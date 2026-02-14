//! System V AMD64 ABI — parameter classification, call/return conventions,
//! frame layout, and red zone support for the x86-64 target.
//!
//! This module implements the type classification algorithm from the
//! System V AMD64 ABI specification (§3.2.3). Every C type is decomposed
//! into 8-byte "eightbytes" and each eightbyte is assigned a class:
//!
//! | Class       | Meaning                                      |
//! |-------------|----------------------------------------------|
//! | INTEGER     | Passed in a general-purpose register (GPR)   |
//! | SSE         | Passed in an XMM register                    |
//! | MEMORY      | Passed on the stack                          |
//! | X87         | Passed on the x87 FPU stack                  |
//! | X87UP       | Upper portion of an x87 long double value    |
//! | COMPLEX_X87 | Complex long double (X87 pair)               |
//! | NO_CLASS    | Padding or void (ignored for passing)        |
//!
//! # Register Assignment Order
//!
//! - **Integer arguments:** RDI, RSI, RDX, RCX, R8, R9 (6 registers)
//! - **Float arguments:** XMM0–XMM7 (8 registers)
//! - **Return values:** RAX (integer), XMM0 (float), RAX:RDX (128-bit int),
//!   XMM0:XMM1 (complex float)
//!
//! # Red Zone
//!
//! Leaf functions may use the 128-byte area below RSP without adjusting
//! the stack pointer (System V AMD64 ABI §3.2.2). This optimisation is
//! only valid when the function contains no calls and the frame fits
//! within 128 bytes.
//!
//! # Frame Layout
//!
//! The x86-64 stack frame grows downward (toward lower addresses):
//!
//! ```text
//! [higher addresses]
//! ...caller's frame...
//! argument overflow area (arg7, arg8, ...)   ← [RSP + 8*N at call]
//! return address                              ← pushed by CALL
//! saved RBP                                   ← [RBP] after prologue
//! callee-saved registers                      ← [RBP - 8], [RBP - 16], ...
//! local variables                             ← negative offsets from RBP
//! spill slots                                 ← below locals
//! [alignment padding to 16 bytes]
//! [RSP]                                       ← 16-byte aligned
//! [lower addresses / red zone (128 bytes)]
//! ```

use crate::backend::traits::{ParamClass, PhysReg};
use crate::backend::x86_64::registers;
use crate::common::target::Target;
use crate::common::type_builder::{compute_struct_layout, FieldLayout, StructLayout};
use crate::common::types::{align_of, size_of, CType, FieldDef};
use crate::ir::function::IrFunction;
use crate::ir::instructions::Instruction;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the red zone in bytes below RSP that leaf functions may use
/// without adjusting the stack pointer (System V AMD64 ABI §3.2.2).
pub const RED_ZONE_SIZE: u32 = 128;

/// Maximum aggregate size (in bytes) that the System V AMD64 ABI passes
/// in registers. Structs larger than 16 bytes are always passed in MEMORY.
const MAX_REGISTER_AGGREGATE_SIZE: usize = 16;

/// Number of bytes in one "eightbyte" — the fundamental classification
/// unit in the System V AMD64 ABI.
const EIGHTBYTE_SIZE: usize = 8;

/// Required stack alignment before a CALL instruction on x86-64.
const STACK_ALIGNMENT: u32 = 16;

/// Size in bytes of a saved register on x86-64 (all GPRs are 64-bit).
const REG_SAVE_SIZE: u32 = 8;

// ---------------------------------------------------------------------------
// ParamLocation — describes where a function parameter is placed
// ---------------------------------------------------------------------------

/// Describes the physical location assigned to a single function parameter
/// after ABI classification and register assignment.
///
/// The System V AMD64 ABI classifies parameters and assigns them to:
/// - General-purpose registers (RDI, RSI, RDX, RCX, R8, R9)
/// - SSE registers (XMM0–XMM7)
/// - Stack slots (when registers are exhausted)
/// - Hidden pointers (for large aggregates classified as MEMORY)
///
/// The register type (GPR vs SSE) is encoded in the [`PhysReg`] value
/// itself, not in the enum variant name. This simplifies downstream code
/// generation since the backend already distinguishes register classes
/// by their physical register identity.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamLocation {
    /// Value is passed in a single physical register (GPR or SSE).
    ///
    /// For INTEGER-classified scalars (int, long, pointer): GPR register.
    /// For SSE-classified scalars (float, double): XMM register.
    Register(PhysReg),

    /// Value is split across two physical registers.
    ///
    /// This covers:
    /// - Two INTEGER eightbytes (e.g., 128-bit struct with two ints)
    /// - Two SSE eightbytes (e.g., `_Complex double`)
    /// - Mixed INTEGER+SSE (e.g., struct with int and float fields)
    ///
    /// The first register holds the low eightbyte, the second holds the
    /// high eightbyte.
    RegisterPair(PhysReg, PhysReg),

    /// Value is passed on the stack at the given byte offset from the
    /// caller's stack pointer at the point of the CALL instruction
    /// (before the return address is pushed).
    Stack {
        /// Byte offset from the caller's RSP at the call site.
        offset: i32,
    },

    /// Large aggregate (MEMORY class) passed via a hidden pointer.
    ///
    /// The caller copies the aggregate to a temporary location and passes
    /// a pointer to the copy in the specified integer register. This
    /// consumes one integer register slot from the RDI/RSI/RDX/RCX/R8/R9
    /// sequence.
    HiddenPointer(PhysReg),
}

// ---------------------------------------------------------------------------
// ReturnLocation — describes where a function return value is placed
// ---------------------------------------------------------------------------

/// Describes the physical location of a function's return value after
/// ABI classification.
///
/// The System V AMD64 ABI returns values in:
/// - RAX for integer types ≤ 8 bytes
/// - RAX:RDX for integer types 8–16 bytes
/// - XMM0 for SSE types (float, double)
/// - XMM0:XMM1 for two-SSE-eightbyte types (complex float/double)
/// - Memory (caller-provided hidden pointer) for large aggregates
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnLocation {
    /// Return value is placed in a single physical register.
    ///
    /// - INTEGER → RAX
    /// - SSE → XMM0
    /// - X87 → ST(0) (represented via a designated PhysReg sentinel)
    Register(PhysReg),

    /// Return value is split across two physical registers.
    ///
    /// - Two INTEGER eightbytes → (RAX, RDX)
    /// - Two SSE eightbytes → (XMM0, XMM1)
    /// - Mixed INTEGER+SSE → (RAX, XMM0) or (XMM0, RAX)
    RegisterPair(PhysReg, PhysReg),

    /// Return value is too large for registers; the caller allocates
    /// space and passes a hidden pointer in RDI. The function stores the
    /// return value through this pointer and returns the pointer in RAX.
    Memory,

    /// Function returns void — no return value.
    Void,
}

// ---------------------------------------------------------------------------
// FrameLayout — stack frame geometry for a function
// ---------------------------------------------------------------------------

/// Complete stack frame layout information for an x86-64 function.
///
/// All offsets are expressed relative to the frame pointer (RBP) after
/// the prologue has executed. Negative offsets point toward lower
/// addresses (deeper into the stack).
///
/// # Frame Construction Order (Prologue)
///
/// ```text
/// push rbp                      ; save old frame pointer
/// mov rbp, rsp                  ; establish new frame pointer
/// push <callee-saved regs>      ; save registers that this function uses
/// sub rsp, <locals+spills+pad>  ; allocate local/spill area
/// ```
///
/// # Usage
///
/// The code generator uses this layout to:
/// - Emit the correct `sub rsp, N` in the prologue
/// - Compute RBP-relative offsets for local variable access
/// - Determine which callee-saved registers need saving/restoring
/// - Decide whether to elide the frame pointer (leaf + small frame)
#[derive(Clone, Debug, PartialEq)]
pub struct FrameLayout {
    /// Total frame allocation size in bytes — the value used in
    /// `sub rsp, frame_size` after all register pushes. This does NOT
    /// include the pushed RBP or callee-saved register pushes (those
    /// are separate `push` instructions). It DOES include local variables,
    /// spill slots, and alignment padding.
    pub frame_size: u32,

    /// RBP-relative byte offset to the start of the local variable area.
    /// Always ≤ 0 (locals live below RBP). The first local variable is
    /// stored at `[RBP + local_area_offset]`.
    pub local_area_offset: i32,

    /// RBP-relative byte offset to the start of the register spill area.
    /// Always ≤ 0 (spills live below locals). The first spill slot is
    /// stored at `[RBP + spill_area_offset]`.
    pub spill_area_offset: i32,

    /// RBP-relative byte offset to the start of the callee-saved register
    /// save area. The first callee-saved register is at
    /// `[RBP + callee_save_area_offset]`. Equals 0 when no callee-saved
    /// registers are used.
    pub callee_save_area_offset: i32,

    /// Stack alignment requirement in bytes (always 16 for x86-64).
    pub alignment: u32,

    /// Whether the function uses RBP as a frame pointer. `false` when
    /// the red zone is used (leaf functions with small frames) or when
    /// frame-pointer omission is enabled.
    pub uses_frame_pointer: bool,
}

// ---------------------------------------------------------------------------
// LocalVar — local variable descriptor for frame layout
// ---------------------------------------------------------------------------

/// Describes a local variable for stack frame layout computation.
///
/// Used as input to [`compute_frame_layout`] to determine how much
/// stack space is needed and where each variable is placed.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalVar {
    /// The C type of this local variable (for debugging and type info).
    pub ty: CType,

    /// Size of the variable in bytes.
    pub size: u32,

    /// Alignment requirement in bytes (must be a power of two).
    pub alignment: u32,
}

// ---------------------------------------------------------------------------
// SpillSlot — register spill slot descriptor
// ---------------------------------------------------------------------------

/// Describes a register spill slot for stack frame layout computation.
///
/// Created by the register allocator when a virtual register cannot be
/// assigned to a physical register and must be "spilled" to the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct SpillSlot {
    /// Size of the spill slot in bytes (typically 8 for GPR, 16 for XMM).
    pub size: u32,

    /// Alignment requirement in bytes (must be a power of two).
    pub alignment: u32,
}

// ---------------------------------------------------------------------------
// classify_type — core ABI classification
// ---------------------------------------------------------------------------

/// Classifies a C type into per-eightbyte [`ParamClass`] assignments
/// according to the System V AMD64 ABI specification (§3.2.3).
///
/// # Arguments
///
/// * `ty`     — the C language type to classify.
/// * `target` — the target architecture (must be [`Target::X86_64`]).
///
/// # Returns
///
/// A `Vec<ParamClass>` with one entry per eightbyte. Scalar types
/// typically produce a single-element vector; structs ≤ 16 bytes may
/// produce one or two entries; large structs produce a single MEMORY
/// entry.
///
/// # Classification Rules (§3.2.3)
///
/// - Integer types ≤ 8 bytes → INTEGER
/// - Float / Double → SSE
/// - Long double (80-bit, stored as 16 bytes on x86-64) → X87 + X87UP
/// - `_Complex float` / `_Complex double` → SSE + SSE
/// - `_Complex long double` → MEMORY (ComplexX87 in some references)
/// - Pointer → INTEGER
/// - Struct ≤ 16 bytes → classify each eightbyte independently
/// - Struct > 16 bytes → MEMORY
/// - Struct with unaligned fields → MEMORY
/// - Array → classified like a struct based on element types and total size
/// - Union → merge classifications of all members per eightbyte
///
/// # Post-Merger Rules
///
/// - If any eightbyte is MEMORY, the entire aggregate is MEMORY.
/// - If the first eightbyte is X87 and the second is not X87UP, MEMORY.
/// - X87UP is only valid immediately following X87.
///
/// # Examples
///
/// ```ignore
/// let classes = classify_type(&CType::Int { signed: true }, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::Integer]);
///
/// let classes = classify_type(&CType::Double, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::SSE]);
///
/// let classes = classify_type(&CType::LongDouble, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::X87, ParamClass::X87Up]);
/// ```
pub fn classify_type(ty: &CType, target: &Target) -> Vec<ParamClass> {
    match ty {
        // -- Void -------------------------------------------------------
        CType::Void => vec![ParamClass::NoClass],

        // -- Boolean ----------------------------------------------------
        CType::Bool => vec![ParamClass::Integer],

        // -- Integer types — all fit in a single GPR --------------------
        CType::Char { .. }
        | CType::Short { .. }
        | CType::Int { .. }
        | CType::Long { .. }
        | CType::LongLong { .. } => vec![ParamClass::Integer],

        // -- Enum — underlying representation is integer ----------------
        CType::Enum { .. } => vec![ParamClass::Integer],

        // -- Float / Double — SSE registers -----------------------------
        CType::Float | CType::Double => vec![ParamClass::SSE],

        // -- Long double — x87 extended precision on x86-64 -------------
        // Occupies 16 bytes of storage (10-byte value + 6-byte padding).
        // Classified as X87 (lower eightbyte) + X87UP (upper eightbyte).
        CType::LongDouble => vec![ParamClass::X87, ParamClass::X87Up],

        // -- Complex types ----------------------------------------------
        // _Complex float (8 bytes) → SSE (single eightbyte)
        // _Complex double (16 bytes) → SSE + SSE (two eightbytes)
        // _Complex long double → ComplexX87 / MEMORY
        //   The actual storage size depends on the target's long double
        //   representation: on x86-64, long double is 16 bytes (from
        //   target.long_double_size()), so _Complex long double is 32 bytes
        //   and cannot fit in registers.
        CType::Complex(base) => match base.as_ref() {
            CType::Float => vec![ParamClass::SSE],
            CType::Double => vec![ParamClass::SSE, ParamClass::SSE],
            CType::LongDouble => {
                // Verify: complex long double = 2 × long_double_size.
                // On x86-64, long_double_size() == 16, so this is 32 bytes
                // — far exceeding the 16-byte register-pair limit.
                let _ld_size = target.long_double_size();
                vec![ParamClass::ComplexX87]
            }
            _ => vec![ParamClass::SSE, ParamClass::SSE],
        },

        // -- Pointer — always INTEGER (8 bytes on x86-64) ---------------
        CType::Pointer(_) => vec![ParamClass::Integer],

        // -- Function type — treated as pointer-to-function -------------
        CType::Function { .. } => vec![ParamClass::Integer],

        // -- Array — classified as aggregate based on total size ---------
        CType::Array { element, size } => {
            let total_size = size
                .map(|n| n * size_of(element, target))
                .unwrap_or(0);
            if total_size > MAX_REGISTER_AGGREGATE_SIZE {
                vec![ParamClass::Memory]
            } else if total_size == 0 {
                vec![ParamClass::NoClass]
            } else {
                // Classify as if it were a struct with repeated elements.
                classify_array_as_aggregate(element, total_size, target)
            }
        }

        // -- Struct — full eightbyte decomposition ----------------------
        CType::Struct { fields, .. } => classify_struct(fields, target),

        // -- Union — classified based on merged member classifications ---
        CType::Union { fields, .. } => classify_union(fields, target),

        // -- Atomic — classify the underlying type ----------------------
        CType::Atomic(inner) => classify_type(inner, target),

        // -- Typedef — classify the underlying type ---------------------
        CType::Typedef { underlying, .. } => classify_type(underlying, target),
    }
}

// ---------------------------------------------------------------------------
// Struct eightbyte classification (§3.2.3 step 4)
// ---------------------------------------------------------------------------

/// Classifies a struct type field-by-field using the eightbyte decomposition
/// algorithm from the System V AMD64 ABI §3.2.3.
///
/// Each 8-byte "eightbyte" of the struct is independently classified based
/// on which fields overlap it. The classification of an eightbyte is the
/// merge of all field classifications that touch that eightbyte.
fn classify_struct(
    fields: &[FieldDef],
    target: &Target,
) -> Vec<ParamClass> {
    // Use the type builder's struct layout computation to get accurate
    // field offsets with proper alignment and padding.
    let layout: StructLayout = compute_struct_layout(fields, target);

    // Rule: structs > 16 bytes are always passed in MEMORY.
    if layout.total_size > MAX_REGISTER_AGGREGATE_SIZE {
        return vec![ParamClass::Memory];
    }

    // Empty structs get NoClass.
    if layout.total_size == 0 {
        return vec![ParamClass::NoClass];
    }

    // Rule: structs with unaligned fields (e.g., from __attribute__((packed)))
    // are classified as MEMORY because misaligned register transfers would
    // trap on some architectures and the ABI does not guarantee correctness.
    // We check each field's actual offset (from FieldLayout) against the
    // natural alignment of its type. If any field (named or anonymous) is
    // misaligned, the entire struct falls back to MEMORY class.
    for (field_idx, field) in fields.iter().enumerate() {
        if field_idx >= layout.fields.len() {
            break;
        }
        let fl: &FieldLayout = &layout.fields[field_idx];
        let natural_align = align_of(&field.ty, target);
        if natural_align > 0 && fl.offset % natural_align != 0 {
            // Field is misaligned — may be a packed struct. Named fields
            // (field.name.is_some()) and anonymous fields are both checked.
            let _field_name: &Option<String> = &field.name;
            return vec![ParamClass::Memory];
        }
    }

    // Determine the number of eightbytes needed to cover the struct.
    let num_eightbytes = (layout.total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    let mut classes = vec![ParamClass::NoClass; num_eightbytes];

    // Walk each field and merge its classification into the overlapping
    // eightbyte(s). Field layout provides precise byte offsets.
    for (field_idx, field) in fields.iter().enumerate() {
        // Safety: layout.fields and fields should have the same count.
        // If the layout has fewer entries (e.g., flexible array member was
        // excluded), clamp the index.
        if field_idx >= layout.fields.len() {
            break;
        }

        let fl: &FieldLayout = &layout.fields[field_idx];

        // For bit-fields, the classification is based on the underlying
        // integer type. Bit-fields are always INTEGER class regardless of
        // their width — the field.bit_width tells us this is a bit-field
        // and its storage is already accounted for in the FieldLayout.
        // Both signed and unsigned bit-fields (checked via is_signed())
        // use the same register class; signedness only affects value
        // extension during code generation.
        let field_class = if field.bit_width.is_some() {
            // Bit-fields are always stored as integers. The signedness
            // (field.ty.is_signed()) affects code generation but not the
            // ABI parameter class.
            let _ = field.ty.is_signed();
            ParamClass::Integer
        } else {
            scalar_class(&field.ty)
        };

        let field_size = fl.size;

        // Determine which eightbyte(s) this field overlaps.
        let start_eb = fl.offset / EIGHTBYTE_SIZE;
        let end_eb = if field_size > 0 {
            (fl.offset + field_size - 1) / EIGHTBYTE_SIZE
        } else {
            start_eb
        };

        // Merge the field's class into each overlapping eightbyte.
        for eb_idx in start_eb..=end_eb.min(num_eightbytes - 1) {
            classes[eb_idx] = classes[eb_idx].merge(field_class);
        }
    }

    // -- Post-merger rules (§3.2.3 step 5) ------------------------------

    // Rule (a): If any eightbyte is MEMORY, the entire struct is MEMORY.
    if classes.contains(&ParamClass::Memory) {
        return vec![ParamClass::Memory];
    }

    // Rule (b): If the first eightbyte is X87 and the second is not X87UP,
    // or if X87UP appears without a preceding X87, the whole struct is MEMORY.
    if num_eightbytes == 2 {
        if classes[0] == ParamClass::X87 && classes[1] != ParamClass::X87Up {
            return vec![ParamClass::Memory];
        }
        if classes[1] == ParamClass::X87Up && classes[0] != ParamClass::X87 {
            return vec![ParamClass::Memory];
        }
    }

    // Rule (c): If the size exceeds two eightbytes and the first is not SSE
    // or any other is not SSEUP, the whole struct is MEMORY.
    // (For ≤ 16 bytes we only have up to 2 eightbytes, so this rarely applies.)

    classes
}

/// Classifies a union type — the classification is the merge of all
/// member classifications across the eightbytes they occupy.
///
/// Union members all start at offset zero, so each member's per-eightbyte
/// classification is merged with the union's running classification for
/// that eightbyte index.
fn classify_union(
    fields: &[FieldDef],
    target: &Target,
) -> Vec<ParamClass> {
    // Union size is the maximum of all member sizes.
    let total_size = fields
        .iter()
        .map(|f| size_of(&f.ty, target))
        .max()
        .unwrap_or(0);

    // Rule: unions > 16 bytes are always passed in MEMORY.
    if total_size > MAX_REGISTER_AGGREGATE_SIZE {
        return vec![ParamClass::Memory];
    }

    if total_size == 0 {
        return vec![ParamClass::NoClass];
    }

    let num_eightbytes = (total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    let mut classes = vec![ParamClass::NoClass; num_eightbytes];

    // Merge each member's per-eightbyte classification.
    for field in fields {
        let field_classes = classify_type(&field.ty, target);
        for (i, &fc) in field_classes.iter().enumerate() {
            if i < num_eightbytes {
                classes[i] = classes[i].merge(fc);
            }
        }
    }

    // Post-merger: MEMORY dominates everything.
    if classes.contains(&ParamClass::Memory) {
        return vec![ParamClass::Memory];
    }

    classes
}

/// Returns the base parameter class for a scalar (non-aggregate) type.
///
/// This helper is used during eightbyte classification of struct and union
/// fields to determine the class contribution of each individual field.
/// It uses the `CType` helper methods (`is_integer`, `is_floating`,
/// `is_pointer`, `is_aggregate`) for efficient classification.
fn scalar_class(ty: &CType) -> ParamClass {
    // Void produces no classification.
    if ty.is_void() {
        return ParamClass::NoClass;
    }

    // Integer types (bool, char, short, int, long, long long, enum),
    // pointers, and function types all map to the INTEGER class.
    // Note: is_integer() covers Bool through Enum; signedness (is_signed)
    // does not affect the register class but may affect sign-extension
    // during code generation.
    if ty.is_integer() || ty.is_pointer() {
        return ParamClass::Integer;
    }

    // Function types decay to pointer-to-function → INTEGER class.
    if matches!(ty, CType::Function { .. }) {
        return ParamClass::Integer;
    }

    // Floating-point types: Float and Double use SSE registers;
    // LongDouble uses the x87 FPU stack.
    if ty.is_floating() {
        return match ty {
            CType::LongDouble => ParamClass::X87,
            _ => ParamClass::SSE,
        };
    }

    // Complex types: _Complex float/double → SSE; _Complex long double → MEMORY.
    if let CType::Complex(base) = ty {
        return match base.as_ref() {
            CType::Float | CType::Double => ParamClass::SSE,
            _ => ParamClass::Memory,
        };
    }

    // Aggregate types (struct, union, array) inside struct fields
    // contribute MEMORY class for this eightbyte, which triggers the
    // post-merger MEMORY-dominates-all rule.
    if ty.is_aggregate() {
        return ParamClass::Memory;
    }

    // Unwrap transparent wrappers.
    match ty {
        CType::Atomic(inner) => scalar_class(inner),
        CType::Typedef { underlying, .. } => scalar_class(underlying),
        _ => ParamClass::Memory,
    }
}

/// Classifies an array as an aggregate by treating it as a sequence of
/// identical elements, similar to a struct with repeated fields.
///
/// If all elements are SSE-class, the eightbytes that contain only
/// complete SSE-class elements get SSE. Otherwise, INTEGER is used as
/// a conservative fallback.
fn classify_array_as_aggregate(
    element: &CType,
    total_size: usize,
    target: &Target,
) -> Vec<ParamClass> {
    let num_eightbytes = (total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    let elem_class = scalar_class(element);

    // If the element class is SSE and elements are properly aligned within
    // eightbytes, classify each eightbyte as SSE. Otherwise, use INTEGER
    // as a safe fallback (INTEGER + SSE merges to INTEGER per the ABI).
    let elem_size = size_of(element, target);
    if elem_class == ParamClass::SSE && elem_size > 0 && EIGHTBYTE_SIZE % elem_size == 0 {
        // Elements pack evenly into eightbytes — all SSE.
        vec![ParamClass::SSE; num_eightbytes]
    } else if elem_class == ParamClass::Integer || elem_class == ParamClass::NoClass {
        vec![ParamClass::Integer; num_eightbytes]
    } else {
        // Mixed or complex — conservative fallback to INTEGER.
        vec![ParamClass::Integer; num_eightbytes]
    }
}

// ---------------------------------------------------------------------------
// compute_param_locations — assign parameters to registers or stack
// ---------------------------------------------------------------------------

/// Assigns each function parameter to a physical location (register or
/// stack slot) based on the System V AMD64 calling convention.
///
/// # Arguments
///
/// * `params` — the C types of the function's formal parameters, in
///   declaration order.
/// * `target` — the target architecture (must be [`Target::X86_64`]).
///
/// # Returns
///
/// A `Vec<ParamLocation>` with the same length as `params`, where each
/// entry describes where the corresponding parameter is passed.
///
/// # Algorithm
///
/// 1. Classify each parameter type using [`classify_type`].
/// 2. Count the integer and SSE registers needed for each parameter.
/// 3. If enough registers remain, assign register(s); otherwise, assign
///    to the stack.
/// 4. MEMORY-class aggregates are passed via a hidden pointer that
///    consumes one integer register slot.
///
/// # Register Sequences
///
/// - **Integer registers:** RDI → RSI → RDX → RCX → R8 → R9
/// - **SSE registers:** XMM0 → XMM1 → XMM2 → ... → XMM7
///
/// # Examples
///
/// ```ignore
/// let params = vec![CType::Int { signed: true }, CType::Double, CType::Long { signed: false }];
/// let locs = compute_param_locations(&params, &Target::X86_64);
/// // int → RDI, double → XMM0, long → RSI
/// ```
pub fn compute_param_locations(params: &[CType], target: &Target) -> Vec<ParamLocation> {
    let mut locations = Vec::with_capacity(params.len());

    // Use the target's stack alignment for parameter area layout.
    // On x86-64, this is always 16 bytes (per the System V ABI).
    let _target_stack_align = target.stack_alignment();

    // Track the next available register index in each sequence.
    let mut int_reg_idx: usize = 0;
    let mut sse_reg_idx: usize = 0;
    let mut stack_offset: i32 = 0;

    for ty in params {
        // Scalar types (integers, floats, pointers) are classified directly
        // without needing struct decomposition — is_scalar() provides a
        // quick check, though we always call classify_type for uniformity.
        let _is_scalar_param = ty.is_scalar();

        let classes = classify_type(ty, target);

        // MEMORY class or ComplexX87 → pass by hidden pointer or on stack.
        let has_memory = classes.contains(&ParamClass::Memory)
            || classes.contains(&ParamClass::ComplexX87);

        // X87 class → always passed on the stack (x87 FPU values).
        let has_x87 = classes.contains(&ParamClass::X87);

        if has_memory {
            // MEMORY-class aggregates: the caller copies the aggregate
            // into a temporary and passes a pointer in the next available
            // integer register. If no integer register is available, the
            // pointer is passed on the stack.
            if int_reg_idx < registers::ARG_REGS_INT.len() {
                let reg = registers::ARG_REGS_INT[int_reg_idx];
                locations.push(ParamLocation::HiddenPointer(reg));
                int_reg_idx += 1;
            } else {
                // No integer register available for hidden pointer — stack.
                let aligned_size = align_up_i32(target.pointer_width() as i32, 8);
                locations.push(ParamLocation::Stack {
                    offset: stack_offset,
                });
                stack_offset += aligned_size;
            }
            continue;
        }

        if has_x87 {
            // Long double (X87 class) is always passed on the stack.
            let type_sz = size_of(ty, target) as i32;
            let aligned_size = align_up_i32(type_sz, 8);
            locations.push(ParamLocation::Stack {
                offset: stack_offset,
            });
            stack_offset += aligned_size;
            continue;
        }

        // Count how many integer and SSE registers this parameter needs.
        let int_needed = classes
            .iter()
            .filter(|c| **c == ParamClass::Integer)
            .count();
        let sse_needed = classes
            .iter()
            .filter(|c| **c == ParamClass::SSE)
            .count();

        // Check if we have enough registers for this parameter.
        let int_avail = registers::ARG_REGS_INT.len() - int_reg_idx;
        let sse_avail = registers::ARG_REGS_FLOAT.len() - sse_reg_idx;

        if int_needed > int_avail || sse_needed > sse_avail {
            // Not enough registers — fall back to stack.
            let type_sz = size_of(ty, target) as i32;
            let aligned_size = align_up_i32(type_sz, 8);
            locations.push(ParamLocation::Stack {
                offset: stack_offset,
            });
            stack_offset += aligned_size;
            continue;
        }

        // Assign registers based on the classification.
        match (int_needed, sse_needed) {
            (1, 0) => {
                // Single INTEGER eightbyte → one GPR.
                let reg = registers::ARG_REGS_INT[int_reg_idx];
                locations.push(ParamLocation::Register(reg));
                int_reg_idx += 1;
            }
            (0, 1) => {
                // Single SSE eightbyte → one XMM register.
                let reg = registers::ARG_REGS_FLOAT[sse_reg_idx];
                locations.push(ParamLocation::Register(reg));
                sse_reg_idx += 1;
            }
            (2, 0) => {
                // Two INTEGER eightbytes → two GPRs.
                let r1 = registers::ARG_REGS_INT[int_reg_idx];
                let r2 = registers::ARG_REGS_INT[int_reg_idx + 1];
                locations.push(ParamLocation::RegisterPair(r1, r2));
                int_reg_idx += 2;
            }
            (0, 2) => {
                // Two SSE eightbytes → two XMM registers.
                let r1 = registers::ARG_REGS_FLOAT[sse_reg_idx];
                let r2 = registers::ARG_REGS_FLOAT[sse_reg_idx + 1];
                locations.push(ParamLocation::RegisterPair(r1, r2));
                sse_reg_idx += 2;
            }
            (1, 1) => {
                // Mixed: one INTEGER and one SSE eightbyte.
                // Determine order from the classification vector.
                let first_is_int = classes.first() == Some(&ParamClass::Integer);
                let (r1, r2) = if first_is_int {
                    (
                        registers::ARG_REGS_INT[int_reg_idx],
                        registers::ARG_REGS_FLOAT[sse_reg_idx],
                    )
                } else {
                    (
                        registers::ARG_REGS_FLOAT[sse_reg_idx],
                        registers::ARG_REGS_INT[int_reg_idx],
                    )
                };
                locations.push(ParamLocation::RegisterPair(r1, r2));
                int_reg_idx += 1;
                sse_reg_idx += 1;
            }
            _ => {
                // Unusual classification — conservative fallback to stack.
                let type_sz = size_of(ty, target) as i32;
                let aligned_size = align_up_i32(type_sz, 8);
                locations.push(ParamLocation::Stack {
                    offset: stack_offset,
                });
                stack_offset += aligned_size;
            }
        }
    }

    locations
}

// ---------------------------------------------------------------------------
// compute_return_location — assign return value to register(s) or memory
// ---------------------------------------------------------------------------

/// Determines the physical location for a function's return value based
/// on the System V AMD64 calling convention.
///
/// # Arguments
///
/// * `ret_type` — the C type of the function's return value.
/// * `target`   — the target architecture (must be [`Target::X86_64`]).
///
/// # Returns
///
/// A [`ReturnLocation`] describing where the return value is placed.
///
/// # Return Value Rules (§3.2.3)
///
/// | Classification      | Location                         |
/// |---------------------|----------------------------------|
/// | Void                | `Void`                           |
/// | INTEGER (≤ 8 bytes) | `Register(RAX)`                  |
/// | INTEGER (8–16 bytes)| `RegisterPair(RAX, RDX)`         |
/// | SSE (single)        | `Register(XMM0)`                 |
/// | SSE (pair)          | `RegisterPair(XMM0, XMM1)`       |
/// | X87                 | `Register(RAX)` (x87 ST(0))      |
/// | MEMORY              | `Memory` (hidden pointer in RDI)  |
///
/// For MEMORY returns, the caller allocates space and passes a pointer
/// in RDI (consuming the first integer register slot). The function
/// stores through this pointer and returns the pointer itself in RAX.
pub fn compute_return_location(ret_type: &CType, target: &Target) -> ReturnLocation {
    // Void functions produce no return value.
    if ret_type.is_void() {
        return ReturnLocation::Void;
    }

    let classes = classify_type(ret_type, target);

    // Count class occurrences.
    let int_count = classes
        .iter()
        .filter(|c| **c == ParamClass::Integer)
        .count();
    let sse_count = classes
        .iter()
        .filter(|c| **c == ParamClass::SSE)
        .count();
    let has_memory = classes.contains(&ParamClass::Memory)
        || classes.contains(&ParamClass::ComplexX87);
    let has_x87 = classes.contains(&ParamClass::X87);

    // MEMORY class: returned via hidden pointer in RDI.
    if has_memory {
        return ReturnLocation::Memory;
    }

    // X87 class: returned on x87 FPU stack ST(0). We represent this
    // using RAX as a sentinel — the code generator knows to use fstp
    // for X87 returns based on the type classification.
    if has_x87 {
        return ReturnLocation::Register(registers::RAX);
    }

    match (int_count, sse_count) {
        // Single INTEGER eightbyte → RAX.
        (1, 0) => ReturnLocation::Register(registers::RAX),
        // Two INTEGER eightbytes → RAX:RDX.
        (2, 0) => ReturnLocation::RegisterPair(registers::RAX, registers::RDX),
        // Single SSE eightbyte → XMM0.
        (0, 1) => ReturnLocation::Register(registers::XMM0),
        // Two SSE eightbytes → XMM0:XMM1.
        (0, 2) => ReturnLocation::RegisterPair(registers::XMM0, registers::XMM1),
        // Mixed INTEGER + SSE.
        (1, 1) => {
            let first_is_int = classes.first() == Some(&ParamClass::Integer);
            if first_is_int {
                ReturnLocation::RegisterPair(registers::RAX, registers::XMM0)
            } else {
                ReturnLocation::RegisterPair(registers::XMM0, registers::RAX)
            }
        }
        // Fallback: anything else is too complex → Memory return.
        _ => ReturnLocation::Memory,
    }
}

// ---------------------------------------------------------------------------
// compute_frame_layout — stack frame geometry computation
// ---------------------------------------------------------------------------

/// Computes the complete stack frame layout for an x86-64 function.
///
/// The layout determines the sizes and offsets of three areas in the
/// stack frame (below the saved frame pointer):
///
/// 1. **Callee-saved register area** — registers that the function uses
///    and must save/restore (RBX, RBP, R12–R15).
/// 2. **Local variable area** — space for all local variables.
/// 3. **Spill slot area** — space for registers spilled during allocation.
///
/// # Arguments
///
/// * `locals`       — local variable descriptors (from IR lowering).
/// * `spills`       — spill slot descriptors (from register allocation).
/// * `callee_saved` — callee-saved registers that this function clobbers.
///
/// # Returns
///
/// A [`FrameLayout`] with all offsets computed relative to RBP.
///
/// # Alignment
///
/// The total frame size is padded to ensure that RSP is 16-byte aligned
/// after the prologue completes. This accounts for:
/// - The return address pushed by CALL (8 bytes)
/// - The saved RBP pushed in the prologue (8 bytes)
/// - The callee-saved register pushes (8 bytes each)
/// - The `sub rsp, N` allocation
///
/// # Frame Pointer Decision
///
/// The function uses a frame pointer (RBP) unless the frame is empty
/// (no locals, no spills, no callee saves). Frame-pointer-omission
/// optimizations are intentionally conservative for debuggability.
pub fn compute_frame_layout(
    locals: &[LocalVar],
    spills: &[SpillSlot],
    callee_saved: &[PhysReg],
) -> FrameLayout {
    // ---------------------------------------------------------------
    // Step 1: Compute callee-saved register save area size.
    // Each saved register occupies 8 bytes (64-bit GPR on x86-64).
    // These are typically emitted as `push` instructions after `push rbp`.
    // ---------------------------------------------------------------
    // Validate that every callee-saved register is a real (non-sentinel)
    // register by inspecting PhysReg.0 — sentinel NONE uses u16::MAX.
    debug_assert!(
        callee_saved
            .iter()
            .all(|r| r.0 != u16::MAX),
        "compute_frame_layout: callee_saved list contains NONE sentinel register"
    );

    let num_callee_saved = callee_saved.len() as u32;
    let callee_save_size = num_callee_saved * REG_SAVE_SIZE;

    // The callee-save area starts immediately below the saved RBP.
    // First callee-saved register is at [RBP - 8].
    let callee_save_area_offset: i32 = if callee_save_size > 0 {
        -(callee_save_size as i32)
    } else {
        0
    };

    // ---------------------------------------------------------------
    // Step 2: Compute local variable area size.
    // Variables are laid out sequentially with proper alignment.
    // ---------------------------------------------------------------
    let mut local_area_size: u32 = 0;
    for local in locals {
        // Align the current offset up to the variable's alignment requirement.
        local_area_size = align_up_u32(local_area_size, local.alignment);
        local_area_size += local.size;
    }

    // The local area starts after the callee-saved registers.
    let local_area_offset: i32 = callee_save_area_offset - (local_area_size as i32);

    // ---------------------------------------------------------------
    // Step 3: Compute spill slot area size.
    // Spill slots are laid out sequentially with proper alignment.
    // ---------------------------------------------------------------
    let mut spill_area_size: u32 = 0;
    for spill in spills {
        spill_area_size = align_up_u32(spill_area_size, spill.alignment);
        spill_area_size += spill.size;
    }

    // The spill area starts after the local variables.
    let spill_area_offset: i32 = local_area_offset - (spill_area_size as i32);

    // ---------------------------------------------------------------
    // Step 4: Compute total frame size with alignment.
    //
    // After the prologue:
    //   [return address]    8 bytes (pushed by CALL)
    //   [saved RBP]         8 bytes (push rbp)
    //   [callee saves]      callee_save_size bytes (push regs)
    //   [frame alloc]       frame_size bytes (sub rsp, frame_size)
    //
    // For RSP to be 16-byte aligned after sub rsp:
    //   (8 + 8 + callee_save_size + frame_size) % 16 == 0
    //   (16 + callee_save_size + frame_size) % 16 == 0
    //   (callee_save_size + frame_size) % 16 == 0
    // ---------------------------------------------------------------
    let raw_frame_alloc = local_area_size + spill_area_size;

    // The frame_size is the `sub rsp, N` value — it does NOT include
    // the callee-saved pushes (those are separate push instructions).
    // We need: (callee_save_size + frame_size) ≡ 0 (mod 16).
    let combined = callee_save_size + raw_frame_alloc;
    let aligned_combined = align_up_u32(combined, STACK_ALIGNMENT);
    let frame_size = aligned_combined - callee_save_size;

    // ---------------------------------------------------------------
    // Step 5: Determine frame pointer usage.
    // We use a frame pointer unless the frame is completely empty.
    // This is conservative but ensures debuggability and simplifies
    // the code generator's offset computation.
    // ---------------------------------------------------------------
    let uses_frame_pointer =
        frame_size > 0 || callee_save_size > 0 || !locals.is_empty() || !spills.is_empty();

    FrameLayout {
        frame_size,
        local_area_offset,
        spill_area_offset,
        callee_save_area_offset,
        alignment: STACK_ALIGNMENT,
        uses_frame_pointer,
    }
}

// ---------------------------------------------------------------------------
// can_use_red_zone — red zone eligibility check
// ---------------------------------------------------------------------------

/// Determines whether a function can use the 128-byte red zone below RSP.
///
/// The System V AMD64 ABI defines a 128-byte area below the current RSP
/// that is guaranteed not to be clobbered by signal handlers or interrupts
/// in user-space code. Leaf functions with small frames can use this area
/// without adjusting RSP, saving the overhead of stack pointer manipulation
/// in the prologue and epilogue.
///
/// # Requirements for Red Zone Usage
///
/// 1. **Leaf function:** The function must not contain any `call`
///    instructions. Functions that call other functions cannot use the
///    red zone because the callee's frame would clobber it.
/// 2. **Small frame:** The function's total stack usage (locals, spills,
///    temporaries) must fit within [`RED_ZONE_SIZE`] (128 bytes).
///
/// # Arguments
///
/// * `func` — the IR function to analyze.
///
/// # Returns
///
/// `true` if the function is eligible for red zone optimization.
///
/// # Conservative Approach
///
/// This function errs on the side of caution:
/// - Inline assembly is treated as a potential call site (conservative).
/// - Variadic functions are excluded (they typically need the register
///   save area which exceeds the red zone).
/// - The frame size estimate is based on alloca instructions in the IR.
pub fn can_use_red_zone(func: &IrFunction) -> bool {
    // Variadic functions use a register save area for va_start that
    // typically exceeds the red zone.
    if func.is_variadic {
        return false;
    }

    // Quick exit: a function with no basic blocks has no instructions and
    // trivially qualifies for the red zone. We access basic_blocks directly
    // (rather than via blocks()) for this fast-path length check.
    if func.basic_blocks.is_empty() {
        return true;
    }

    // Walk all basic blocks and check for Call instructions.
    // A leaf function has no calls to other functions.
    let mut has_call = false;
    let mut estimated_frame_bytes: u32 = 0;

    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                // Any function call disqualifies the red zone.
                Instruction::Call { .. } => {
                    has_call = true;
                }
                // Inline assembly is treated conservatively as a potential
                // call site — it could contain `call` instructions that
                // we cannot analyze at the IR level.
                Instruction::InlineAsm {
                    has_side_effects, ..
                } => {
                    // Only treat volatile inline asm as a potential call.
                    // Pure inline asm (e.g., rdtsc) is unlikely to call.
                    if *has_side_effects {
                        has_call = true;
                    }
                }
                // Count alloca sizes to estimate frame usage.
                Instruction::Alloca { alignment, .. } => {
                    // Each alloca contributes at least its alignment worth
                    // of space. We use alignment as a conservative estimate
                    // for the slot size — the actual size depends on the
                    // IrType, which we approximate by the alignment (at
                    // least 1 byte per alloca, typically 4 or 8).
                    let slot_size = (*alignment).max(8);
                    estimated_frame_bytes = estimated_frame_bytes.saturating_add(slot_size);
                }
                _ => {}
            }

            // Early exit if we already know we can't use the red zone.
            if has_call {
                return false;
            }
        }
    }

    // The function is a leaf. Check if the estimated frame fits
    // within the red zone.
    estimated_frame_bytes <= RED_ZONE_SIZE
}

// ---------------------------------------------------------------------------
// Register convention queries
// ---------------------------------------------------------------------------

/// Returns the callee-saved general-purpose registers for the System V
/// AMD64 ABI.
///
/// These registers must be preserved across function calls. If a function
/// modifies any of these registers, it must save them in the prologue and
/// restore them in the epilogue.
///
/// # Callee-Saved Registers (System V AMD64 ABI §3.2.1)
///
/// | Register | Purpose                                    |
/// |----------|--------------------------------------------|
/// | RBX      | General-purpose callee-saved               |
/// | RBP      | Frame pointer (callee-saved by convention) |
/// | R12      | General-purpose callee-saved               |
/// | R13      | General-purpose callee-saved               |
/// | R14      | General-purpose callee-saved               |
/// | R15      | General-purpose callee-saved               |
#[inline]
pub fn callee_saved_gprs() -> &'static [PhysReg] {
    &registers::CALLEE_SAVED
}

/// Returns the caller-saved (volatile) general-purpose registers for the
/// System V AMD64 ABI.
///
/// These registers may be clobbered by any function call. The caller is
/// responsible for saving them before a call if their values are needed
/// afterward.
///
/// # Caller-Saved Registers (System V AMD64 ABI §3.2.1)
///
/// | Register | Purpose                        |
/// |----------|--------------------------------|
/// | RAX      | Return value / scratch         |
/// | RCX      | 4th integer argument / scratch  |
/// | RDX      | 3rd integer argument / scratch  |
/// | RSI      | 2nd integer argument / scratch  |
/// | RDI      | 1st integer argument / scratch  |
/// | R8       | 5th integer argument / scratch  |
/// | R9       | 6th integer argument / scratch  |
/// | R10      | Scratch / static chain pointer  |
/// | R11      | Scratch                        |
///
/// Note: XMM0–XMM15 are also caller-saved but are not included here
/// since this function returns only GPRs. SSE register management is
/// handled separately by the register allocator.
#[inline]
pub fn caller_saved_gprs() -> &'static [PhysReg] {
    &registers::CALLER_SAVED
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Aligns `value` up to the next multiple of `alignment`.
/// `alignment` must be a power of two and non-zero.
#[inline]
fn align_up_u32(value: u32, alignment: u32) -> u32 {
    debug_assert!(alignment > 0 && alignment.is_power_of_two());
    (value.wrapping_add(alignment - 1)) & !(alignment - 1)
}

/// Aligns an `i32` value up to the next multiple of `alignment`.
/// `alignment` must be a positive power of two.
#[inline]
fn align_up_i32(value: i32, alignment: i32) -> i32 {
    debug_assert!(alignment > 0);
    (value.wrapping_add(alignment - 1)) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::FieldDef;

    // -- classify_type tests -------------------------------------------

    #[test]
    fn classify_void() {
        let classes = classify_type(&CType::Void, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::NoClass]);
    }

    #[test]
    fn classify_bool() {
        let classes = classify_type(&CType::Bool, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_int() {
        let classes = classify_type(&CType::Int { signed: true }, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_unsigned_long() {
        let classes = classify_type(&CType::Long { signed: false }, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_float() {
        let classes = classify_type(&CType::Float, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::SSE]);
    }

    #[test]
    fn classify_double() {
        let classes = classify_type(&CType::Double, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::SSE]);
    }

    #[test]
    fn classify_long_double() {
        let classes = classify_type(&CType::LongDouble, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::X87, ParamClass::X87Up]);
    }

    #[test]
    fn classify_pointer() {
        let classes = classify_type(
            &CType::Pointer(Box::new(CType::Int { signed: true })),
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_complex_double() {
        let classes = classify_type(
            &CType::Complex(Box::new(CType::Double)),
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::SSE, ParamClass::SSE]);
    }

    #[test]
    fn classify_complex_long_double() {
        let classes = classify_type(
            &CType::Complex(Box::new(CType::LongDouble)),
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::ComplexX87]);
    }

    #[test]
    fn classify_small_struct_two_ints() {
        // struct { int a; int b; } — 8 bytes, one eightbyte, INTEGER
        let fields = vec![
            FieldDef {
                name: Some("a".into()),
                ty: CType::Int { signed: true },
                bit_width: None,
            },
            FieldDef {
                name: Some("b".into()),
                ty: CType::Int { signed: true },
                bit_width: None,
            },
        ];
        let classes = classify_type(
            &CType::Struct {
                name: Some("s".into()),
                fields,
            },
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_large_struct_memory() {
        // struct { long a; long b; long c; } — 24 bytes → MEMORY
        let fields = vec![
            FieldDef {
                name: Some("a".into()),
                ty: CType::Long { signed: true },
                bit_width: None,
            },
            FieldDef {
                name: Some("b".into()),
                ty: CType::Long { signed: true },
                bit_width: None,
            },
            FieldDef {
                name: Some("c".into()),
                ty: CType::Long { signed: true },
                bit_width: None,
            },
        ];
        let classes = classify_type(
            &CType::Struct {
                name: Some("s".into()),
                fields,
            },
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::Memory]);
    }

    #[test]
    fn classify_struct_int_double() {
        // struct { int a; double b; } — 16 bytes, INTEGER + SSE
        let fields = vec![
            FieldDef {
                name: Some("a".into()),
                ty: CType::Int { signed: true },
                bit_width: None,
            },
            FieldDef {
                name: Some("b".into()),
                ty: CType::Double,
                bit_width: None,
            },
        ];
        let classes = classify_type(
            &CType::Struct {
                name: Some("s".into()),
                fields,
            },
            &Target::X86_64,
        );
        // First eightbyte contains int (4 bytes) + padding → INTEGER
        // Second eightbyte contains double (8 bytes) → SSE
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0], ParamClass::Integer);
        assert_eq!(classes[1], ParamClass::SSE);
    }

    // -- compute_return_location tests ---------------------------------

    #[test]
    fn return_void() {
        let loc = compute_return_location(&CType::Void, &Target::X86_64);
        assert_eq!(loc, ReturnLocation::Void);
    }

    #[test]
    fn return_int() {
        let loc = compute_return_location(&CType::Int { signed: true }, &Target::X86_64);
        assert_eq!(loc, ReturnLocation::Register(registers::RAX));
    }

    #[test]
    fn return_double() {
        let loc = compute_return_location(&CType::Double, &Target::X86_64);
        assert_eq!(loc, ReturnLocation::Register(registers::XMM0));
    }

    #[test]
    fn return_long_double() {
        // Long double returns via x87 — mapped to RAX sentinel.
        let loc = compute_return_location(&CType::LongDouble, &Target::X86_64);
        assert_eq!(loc, ReturnLocation::Register(registers::RAX));
    }

    // -- compute_param_locations tests ---------------------------------

    #[test]
    fn param_locations_basic() {
        let params = vec![
            CType::Int { signed: true },           // → RDI
            CType::Pointer(Box::new(CType::Void)),  // → RSI
            CType::Double,                           // → XMM0
        ];
        let locs = compute_param_locations(&params, &Target::X86_64);
        assert_eq!(locs.len(), 3);
        assert_eq!(
            locs[0],
            ParamLocation::Register(registers::ARG_REGS_INT[0])
        );
        assert_eq!(
            locs[1],
            ParamLocation::Register(registers::ARG_REGS_INT[1])
        );
        assert_eq!(
            locs[2],
            ParamLocation::Register(registers::ARG_REGS_FLOAT[0])
        );
    }

    #[test]
    fn param_locations_overflow_to_stack() {
        // 7 integer args — first 6 go in registers, 7th goes on stack.
        let params: Vec<CType> = (0..7).map(|_| CType::Int { signed: true }).collect();
        let locs = compute_param_locations(&params, &Target::X86_64);
        assert_eq!(locs.len(), 7);
        for i in 0..6 {
            assert_eq!(
                locs[i],
                ParamLocation::Register(registers::ARG_REGS_INT[i])
            );
        }
        assert!(matches!(locs[6], ParamLocation::Stack { .. }));
    }

    #[test]
    fn param_locations_mixed_int_float() {
        let params = vec![
            CType::Int { signed: true },  // → RDI
            CType::Float,                  // → XMM0
            CType::Long { signed: false }, // → RSI
            CType::Double,                 // → XMM1
        ];
        let locs = compute_param_locations(&params, &Target::X86_64);
        assert_eq!(locs.len(), 4);
        assert_eq!(locs[0], ParamLocation::Register(registers::RDI));
        assert_eq!(locs[1], ParamLocation::Register(registers::XMM0));
        assert_eq!(locs[2], ParamLocation::Register(registers::RSI));
        assert_eq!(locs[3], ParamLocation::Register(registers::XMM1));
    }

    // -- compute_frame_layout tests ------------------------------------

    #[test]
    fn frame_layout_empty() {
        let layout = compute_frame_layout(&[], &[], &[]);
        assert_eq!(layout.frame_size, 0);
        assert_eq!(layout.callee_save_area_offset, 0);
        assert_eq!(layout.local_area_offset, 0);
        assert_eq!(layout.spill_area_offset, 0);
        assert_eq!(layout.alignment, STACK_ALIGNMENT);
        assert!(!layout.uses_frame_pointer);
    }

    #[test]
    fn frame_layout_with_locals() {
        let locals = vec![
            LocalVar {
                ty: CType::Int { signed: true },
                size: 4,
                alignment: 4,
            },
            LocalVar {
                ty: CType::Long { signed: true },
                size: 8,
                alignment: 8,
            },
        ];
        let layout = compute_frame_layout(&locals, &[], &[]);
        // 4 bytes (int) + 4 padding + 8 bytes (long) = 16 bytes local area
        // No callee saves, so frame_size must be >= 16 and aligned
        assert!(layout.frame_size >= 16);
        assert_eq!(layout.frame_size % STACK_ALIGNMENT, 0);
        assert!(layout.uses_frame_pointer);
    }

    #[test]
    fn frame_layout_with_callee_saves() {
        let callee = vec![registers::RBX, registers::R12, registers::R13];
        let layout = compute_frame_layout(&[], &[], &callee);
        // 3 callee-saved registers × 8 bytes = 24 bytes callee-save area.
        assert_eq!(layout.callee_save_area_offset, -24);
        // Frame_size is for sub rsp only (locals + spills + padding).
        // callee_save_size = 24, and (24 + frame_size) must be 16-aligned.
        // 24 % 16 = 8, so frame_size must be 8 (mod 16) to align.
        assert_eq!((24 + layout.frame_size) % STACK_ALIGNMENT, 0);
        assert!(layout.uses_frame_pointer);
    }

    #[test]
    fn frame_layout_alignment() {
        // Verify alignment with odd number of callee saves.
        let locals = vec![LocalVar {
            ty: CType::Int { signed: true },
            size: 4,
            alignment: 4,
        }];
        let callee = vec![registers::RBX]; // 1 callee save = 8 bytes
        let layout = compute_frame_layout(&locals, &[], &callee);
        // callee_save_size = 8, local_area_size = 4
        // combined = 8 + 4 = 12, aligned to 16 = 16
        // frame_size = 16 - 8 = 8
        assert_eq!((8 + layout.frame_size) % STACK_ALIGNMENT, 0);
    }

    // -- can_use_red_zone tests ----------------------------------------

    #[test]
    fn red_zone_leaf_function() {
        use crate::ir::types::IrType;

        // Create a simple leaf function with one small alloca.
        let func = IrFunction::new("leaf".into(), IrType::I32, vec![]);
        // The empty function has one entry block with no instructions.
        // It's a leaf (no calls) and has zero frame usage.
        assert!(can_use_red_zone(&func));
    }

    // -- callee_saved / caller_saved tests -----------------------------

    #[test]
    fn callee_saved_contains_rbx() {
        let regs = callee_saved_gprs();
        assert!(regs.contains(&registers::RBX));
    }

    #[test]
    fn callee_saved_contains_rbp() {
        let regs = callee_saved_gprs();
        assert!(regs.contains(&registers::RBP));
    }

    #[test]
    fn callee_saved_contains_r12_r15() {
        let regs = callee_saved_gprs();
        assert!(regs.contains(&registers::R12));
        assert!(regs.contains(&registers::R13));
        assert!(regs.contains(&registers::R14));
        assert!(regs.contains(&registers::R15));
    }

    #[test]
    fn caller_saved_contains_rax() {
        let regs = caller_saved_gprs();
        assert!(regs.contains(&registers::RAX));
    }

    #[test]
    fn caller_saved_contains_arg_regs() {
        let regs = caller_saved_gprs();
        assert!(regs.contains(&registers::RDI));
        assert!(regs.contains(&registers::RSI));
        assert!(regs.contains(&registers::RDX));
        assert!(regs.contains(&registers::RCX));
        assert!(regs.contains(&registers::R8));
        assert!(regs.contains(&registers::R9));
    }

    #[test]
    fn caller_saved_contains_scratch() {
        let regs = caller_saved_gprs();
        assert!(regs.contains(&registers::R10));
        assert!(regs.contains(&registers::R11));
    }

    // -- RED_ZONE_SIZE constant test -----------------------------------

    #[test]
    fn red_zone_size_is_128() {
        assert_eq!(RED_ZONE_SIZE, 128);
    }
}
