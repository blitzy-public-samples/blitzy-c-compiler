//! cdecl / System V i386 ABI implementation for the BCC compiler.
//!
//! This module implements the complete calling convention for the i686 (IA-32)
//! architecture, following the System V i386 ABI supplement with modern GCC
//! amendments.  It is the i686 counterpart to the x86-64 ABI module and is
//! consumed by the code-generation and register-allocation phases.
//!
//! # cdecl Calling Convention Summary
//!
//! | Property                  | Value / Rule                                        |
//! |--------------------------|-----------------------------------------------------|
//! | Data model               | ILP32 (int, long, pointer are all 32-bit)           |
//! | Argument passing          | ALL parameters on stack (pushed right-to-left)       |
//! | Register arguments        | NONE — no register-based parameter passing           |
//! | Integer return            | EAX (≤ 32-bit), EDX:EAX (64-bit)                   |
//! | FP return                 | x87 ST(0) (float, double, long double)              |
//! | Struct return ≤ 8 bytes   | EAX (≤ 4B) or EDX:EAX (≤ 8B)                       |
//! | Struct return > 8 bytes   | Hidden sret pointer (first stack argument)           |
//! | Caller cleanup            | Caller removes args from stack (ADD ESP, N)          |
//! | Stack alignment           | 16-byte at CALL instruction (modern ABI)            |
//! | Minimum stack slot        | 4 bytes                                              |
//! | Callee-saved registers    | EBX, ESI, EDI, EBP                                  |
//! | Caller-saved registers    | EAX, ECX, EDX                                       |
//!
//! # Key Differences from x86-64 System V
//!
//! - **No register arguments** — x86-64 passes the first 6 integer args in
//!   RDI/RSI/RDX/RCX/R8/R9 and the first 8 FP args in XMM0–XMM7.  On i686
//!   (cdecl) *everything* goes on the stack.
//! - **x87 FPU for floating-point** — x86-64 uses SSE registers for argument
//!   passing and return.  On i686, floating-point return values are placed in
//!   the x87 ST(0) register.  Float arguments are promoted to double (8 bytes)
//!   when pushed to the stack through the x87 FPU.
//! - **4-byte alignment for long long / double** — on i686, `long long` and
//!   `double` are only 4-byte aligned on the stack (not 8-byte as on x86-64).
//!
//! # Usage
//!
//! ```ignore
//! use crate::backend::i686::abi::I686Abi;
//!
//! let abi = I686Abi::new();
//! let ret = abi.classify_return(&return_type, &target);
//! let layout = abi.compute_stack_layout(&param_types, &target);
//! ```

use crate::backend::i686::registers::{
    EAX, ECX, EDX, EBP, ESP, ST0, CALLEE_SAVED, CALLER_SAVED,
};
use crate::backend::traits::{ParamClass, PhysReg};
use crate::common::target::Target;
use crate::common::types::{align_of, size_of, CType, FieldDef};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum stack slot size in bytes for argument passing on i686.
///
/// Every argument occupies at least 4 bytes on the stack, even types smaller
/// than 32 bits (e.g., `char`, `short`, `_Bool`), which are integer-promoted
/// before pushing.
const STACK_SLOT_MIN: u32 = 4;

/// Maximum struct/union size (in bytes) that can be returned in registers.
///
/// Structs ≤ 4 bytes return in EAX; structs ≤ 8 bytes return in the EDX:EAX
/// register pair.  Structs larger than this threshold are returned via a
/// hidden sret pointer passed as the first stack argument.
///
/// This matches GCC's `-freg-struct-return` behaviour, which is the default
/// on Linux i386.
const MAX_REG_RETURN_SIZE: usize = 8;

/// Promoted size (bytes) for `float` arguments on the stack.
///
/// Under the cdecl convention with x87 FPU, `float` values are loaded onto
/// the x87 stack (where they become 80-bit extended precision), then stored
/// to the argument area as 8-byte `double` values.  This constant reflects
/// that promotion.
const FLOAT_PROMOTED_SIZE: u32 = 8;

/// Size of the return address pushed by the CALL instruction (4 bytes).
#[allow(dead_code)]
const RETURN_ADDRESS_SIZE: u32 = 4;

/// Size of a saved frame pointer (EBP) in the standard prologue.
#[allow(dead_code)]
const SAVED_FRAME_POINTER_SIZE: u32 = 4;

// ---------------------------------------------------------------------------
// ArgClassification — argument passing classification
// ---------------------------------------------------------------------------

/// Classification of how a function argument is passed under the cdecl ABI.
///
/// In the cdecl calling convention, **all** arguments are passed on the stack.
/// There is no register-based parameter passing (unlike x86-64 System V or
/// ARM AAPCS64).  The two variants cover:
///
/// - [`Stack`](ArgClassification::Stack) — the argument is copied by value
///   onto the stack at a computed offset.  This is the classification for
///   every normal parameter in cdecl.
///
/// - [`Indirect`](ArgClassification::Indirect) — a hidden pointer to a
///   caller-allocated return buffer is placed on the stack.  This variant is
///   used exclusively for the sret (struct-return) hidden parameter when a
///   function returns a struct/union larger than 8 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgClassification {
    /// Argument passed by value on the stack.
    ///
    /// The `size` field is the number of bytes occupied on the stack (always
    /// rounded up to 4-byte alignment).  The `offset` field is the byte
    /// offset from the start of the argument area (0 = first argument
    /// position, i.e., `[ESP + 4]` from the caller's perspective after
    /// CALL, or `[EBP + 8]` from the callee's perspective after the
    /// standard prologue).
    Stack {
        /// Byte offset from the beginning of the argument area.
        offset: i32,
        /// Size of the argument slot in bytes (≥ 4, 4-byte aligned).
        size: u32,
    },

    /// Hidden sret pointer passed on the stack for large struct returns.
    ///
    /// The caller allocates space for the return value and pushes a pointer
    /// to that space as an invisible first argument.  The callee writes the
    /// return value through this pointer and returns the pointer in EAX.
    Indirect {
        /// Byte offset of the hidden pointer within the argument area.
        stack_offset: i32,
    },
}

// ---------------------------------------------------------------------------
// ReturnClassification — return value classification
// ---------------------------------------------------------------------------

/// Classification of how a function return value is delivered under cdecl.
///
/// The classification controls which registers (or memory location) the
/// code-generation phase must use to produce/consume a function's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReturnClassification {
    /// Scalar value ≤ 32 bits returned in a single general-purpose register.
    ///
    /// Typically `EAX` for integers, enums, pointers, booleans, chars, and
    /// shorts (zero- or sign-extended to 32 bits).  Also used for small
    /// structs/unions (≤ 4 bytes) that fit in a single register.
    InRegister {
        /// The physical register holding the return value (usually `EAX`).
        reg: PhysReg,
    },

    /// 64-bit value returned in a register pair.
    ///
    /// Used for `long long` and small structs/unions in the 5–8 byte range.
    /// The low 32 bits reside in `lo` (EAX) and the high 32 bits in `hi`
    /// (EDX).
    RegisterPair {
        /// Register holding the low 32 bits (EAX).
        lo: PhysReg,
        /// Register holding the high 32 bits (EDX).
        hi: PhysReg,
    },

    /// Floating-point value returned on the x87 FPU stack.
    ///
    /// All floating-point types (`float`, `double`, `long double`) are
    /// returned in ST(0), the top of the x87 floating-point register stack.
    X87 {
        /// The x87 register (always `ST0`).
        reg: PhysReg,
    },

    /// Large struct/union returned via hidden sret pointer.
    ///
    /// The caller allocates space for the return value and passes a hidden
    /// pointer as the first stack argument.  The callee writes through this
    /// pointer.  After the call, EAX contains the sret pointer value.
    Indirect,

    /// No return value (`void` function).
    Void,
}

// ---------------------------------------------------------------------------
// StackLayout — complete call-frame layout
// ---------------------------------------------------------------------------

/// Complete stack-frame layout for a function call under the cdecl ABI.
///
/// This structure encodes the size, alignment, and per-argument classification
/// of the stack argument area.  It is produced by
/// [`I686Abi::compute_stack_layout`] and consumed by the instruction selection
/// and emission phases.
///
/// # Stack Frame Diagram (callee perspective after standard prologue)
///
/// ```text
///     ┌────────────────────────┐  Higher addresses
///     │  ...caller's frame...  │
///     ├────────────────────────┤
///     │  argN                  │  [EBP + 8 + offsetN]
///     │  ...                   │
///     │  arg1                  │  [EBP + 8 + offset1]
///     │  arg0 (or sret ptr)   │  [EBP + 8]
///     ├────────────────────────┤
///     │  return address        │  [EBP + 4]
///     ├────────────────────────┤
///     │  saved EBP             │  [EBP]      ← EBP points here
///     ├────────────────────────┤
///     │  callee-saved regs     │
///     │  local variables       │  [EBP - N]
///     │  spill slots           │
///     └────────────────────────┘  Lower addresses (ESP)
/// ```
#[derive(Clone, Debug)]
pub struct StackLayout {
    /// Total size of the argument area in bytes (including padding to
    /// satisfy the call-site alignment requirement).
    pub total_size: u32,

    /// Per-argument classification with computed byte offsets from the
    /// start of the argument area.  The vector length equals the number of
    /// explicit function parameters (plus one for the hidden sret pointer
    /// when `has_sret` is `true`).
    pub arg_offsets: Vec<ArgClassification>,

    /// Required stack alignment at the CALL instruction site.
    ///
    /// Modern i386 ABI mandates 16-byte alignment; legacy code may only
    /// require 4-byte alignment.  This value comes from
    /// [`Target::stack_alignment()`].
    pub alignment: u32,

    /// `true` when the return type requires a hidden sret pointer to be
    /// passed as the first (invisible) stack argument.
    pub has_sret: bool,

    /// Byte offset of the sret pointer within the argument area, or `None`
    /// when there is no hidden sret argument.  When present, this is
    /// always `0` (the sret pointer occupies the very first stack slot).
    pub sret_offset: Option<i32>,
}

// ---------------------------------------------------------------------------
// I686Abi — main ABI driver
// ---------------------------------------------------------------------------

/// cdecl / System V i386 ABI handler.
///
/// This zero-sized type exposes the complete ABI classification API for the
/// i686 target.  It is deliberately stateless — all target-dependent values
/// come from the [`Target`] parameter passed to each method.
///
/// # Example
///
/// ```ignore
/// let abi = I686Abi::new();
///
/// // Classify how the return type is passed
/// let ret_class = abi.classify_return(&return_ty, &Target::I686);
///
/// // Classify a single argument
/// let arg_class = abi.classify_argument(&param_ty, &Target::I686);
///
/// // Compute the full stack layout for a function call
/// let layout = abi.compute_stack_layout(&param_types, &Target::I686);
/// ```
pub struct I686Abi;

impl I686Abi {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Creates a new `I686Abi` instance.
    ///
    /// The struct is stateless, so this is a zero-cost constructor.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    // -------------------------------------------------------------------
    // classify_argument — single argument classification
    // -------------------------------------------------------------------

    /// Classifies how a single function argument is passed under cdecl.
    ///
    /// In the cdecl ABI, **every** argument is passed on the stack.  This
    /// method returns the [`ArgClassification`] for a given C type, with
    /// `offset` set to `0` (the actual byte offset within the argument area
    /// is computed by [`compute_stack_layout`](Self::compute_stack_layout)).
    ///
    /// # Type Promotion Rules
    ///
    /// - `_Bool`, `char`, `short` → promoted to `int` (4 bytes).
    /// - `float` → promoted to `double` through the x87 FPU (8 bytes).
    /// - `long long` → 8 bytes (4-byte aligned on i686).
    /// - Structs/unions → copied by value, padded to 4-byte alignment.
    /// - Arrays → decay to pointer (4 bytes on ILP32).
    /// - Function types → decay to function pointer (4 bytes).
    pub fn classify_argument(&self, ty: &CType, target: &Target) -> ArgClassification {
        let canonical = ty.canonical();

        // Scalars (integers, floats, pointers) are passed by value on the
        // stack after promotion.  is_scalar() covers arithmetic + pointer
        // types.
        if canonical.is_scalar() {
            let size = self.compute_arg_stack_size(canonical, target);
            return ArgClassification::Stack { offset: 0, size };
        }

        // Aggregates (structs, unions, arrays) are copied by value onto the
        // stack in cdecl.  The total slot is rounded up to 4-byte alignment.
        if canonical.is_aggregate() {
            let size = self.compute_arg_stack_size(canonical, target);
            return ArgClassification::Stack { offset: 0, size };
        }

        // All remaining types (complex, function, void as edge case) are
        // placed on the stack by value with their computed size.
        let size = self.compute_arg_stack_size(canonical, target);
        ArgClassification::Stack { offset: 0, size }
    }

    // -------------------------------------------------------------------
    // classify_return — return value classification
    // -------------------------------------------------------------------

    /// Classifies how a function return value is delivered under cdecl.
    ///
    /// # Return Value Rules (cdecl / System V i386)
    ///
    /// | Type                     | Location                                         |
    /// |--------------------------|--------------------------------------------------|
    /// | `void`                   | No return value                                  |
    /// | `_Bool`, `char`, `short` | EAX (zero/sign-extended to 32 bits)              |
    /// | `int`, `long`, pointer   | EAX                                              |
    /// | `enum`                   | EAX                                              |
    /// | `long long`              | EDX:EAX (EDX = high 32 bits)                     |
    /// | `float`, `double`        | x87 ST(0)                                        |
    /// | `long double`            | x87 ST(0) (80-bit extended precision)            |
    /// | struct/union ≤ 4 bytes   | EAX                                              |
    /// | struct/union 5–8 bytes   | EDX:EAX                                          |
    /// | struct/union > 8 bytes   | Hidden sret pointer (Indirect)                   |
    pub fn classify_return(&self, ty: &CType, target: &Target) -> ReturnClassification {
        let canonical = ty.canonical();
        match canonical {
            // ----- void ------------------------------------------------
            CType::Void => ReturnClassification::Void,

            // ----- Small integers and boolean (≤ 32 bits) → EAX --------
            CType::Bool | CType::Char { .. } | CType::Short { .. } => {
                ReturnClassification::InRegister { reg: EAX }
            }

            // ----- int, long (4 bytes on ILP32), pointer, enum → EAX ---
            CType::Int { .. } | CType::Long { .. } => {
                ReturnClassification::InRegister { reg: EAX }
            }
            CType::Pointer(_) => ReturnClassification::InRegister { reg: EAX },
            CType::Enum { .. } => ReturnClassification::InRegister { reg: EAX },

            // ----- long long (8 bytes) → EDX:EAX -----------------------
            CType::LongLong { .. } => ReturnClassification::RegisterPair {
                lo: EAX,
                hi: EDX,
            },

            // ----- floating-point → x87 ST(0) --------------------------
            CType::Float | CType::Double | CType::LongDouble => {
                ReturnClassification::X87 { reg: ST0 }
            }

            // ----- struct / union — size-dependent ----------------------
            CType::Struct { fields, .. } | CType::Union { fields, .. } => {
                self.classify_struct_return(fields, canonical, target)
            }

            // ----- _Complex — treated as small aggregate ----------------
            CType::Complex(_) => {
                let sz = size_of(canonical, target);
                if sz == 0 {
                    ReturnClassification::Void
                } else if sz <= 4 {
                    ReturnClassification::InRegister { reg: EAX }
                } else if sz <= MAX_REG_RETURN_SIZE {
                    ReturnClassification::RegisterPair {
                        lo: EAX,
                        hi: EDX,
                    }
                } else {
                    ReturnClassification::Indirect
                }
            }

            // ----- array — arrays should not appear as return types -----
            //   but handle gracefully via sret.
            CType::Array { .. } => ReturnClassification::Indirect,

            // ----- function type — not directly returnable; void --------
            CType::Function { .. } => ReturnClassification::Void,

            // ----- _Atomic(T) — canonical() should have unwrapped -------
            CType::Atomic(inner) => self.classify_return(inner, target),

            // ----- typedef — canonical() should have unwrapped ----------
            CType::Typedef { underlying, .. } => self.classify_return(underlying, target),
        }
    }

    // -------------------------------------------------------------------
    // compute_stack_layout — full argument area computation
    // -------------------------------------------------------------------

    /// Computes the complete stack argument area layout for a function call.
    ///
    /// # Parameters
    ///
    /// - `params` — slice of C types for the declared parameters.  If the
    ///   function returns a large struct via sret, the caller should prepend
    ///   a `CType::Pointer(...)` for the hidden sret argument.  The method
    ///   then recognises that the first slot corresponds to the sret pointer.
    /// - `target` — the i686 target providing size/alignment information.
    ///
    /// # Argument Ordering
    ///
    /// Arguments in cdecl are pushed right-to-left.  However, the offsets
    /// produced here are expressed as ascending byte positions from the
    /// *start* of the argument area (arg0 at offset 0).  The push order is
    /// an implementation detail handled by the instruction emitter.
    ///
    /// # Alignment
    ///
    /// Individual argument slots are aligned to 4 bytes (the minimum stack
    /// slot size).  The **total** argument area is then rounded up to the
    /// call-site alignment requirement (16 bytes on modern i386 ABI, per
    /// [`Target::stack_alignment()`]).
    pub fn compute_stack_layout(
        &self,
        params: &[CType],
        target: &Target,
    ) -> StackLayout {
        // Validate we are targeting the right architecture using the ILP32
        // data model check.
        debug_assert!(
            target.data_model() == crate::common::target::DataModel::ILP32,
            "I686Abi::compute_stack_layout called with a non-ILP32 target"
        );

        // Fetch the call-site alignment requirement from the target.
        // On modern i386, this is 16 bytes.  ESP must satisfy this alignment
        // immediately before the CALL instruction.
        let call_site_alignment = target.stack_alignment();

        let mut current_offset: u32 = 0;
        let mut classifications = Vec::with_capacity(params.len());

        for param in params {
            let canonical = param.canonical();
            let slot_size = self.compute_arg_stack_size(canonical, target);

            // Align the current offset to 4 bytes (minimum stack slot).
            current_offset = round_up_u32(current_offset, STACK_SLOT_MIN);

            classifications.push(ArgClassification::Stack {
                offset: current_offset as i32,
                size: slot_size,
            });

            current_offset += slot_size;
        }

        // Pad the total argument area to meet call-site alignment.
        let total_size = round_up_u32(current_offset, call_site_alignment);

        StackLayout {
            total_size,
            arg_offsets: classifications,
            alignment: call_site_alignment,
            has_sret: false,
            sret_offset: None,
        }
    }

    // -------------------------------------------------------------------
    // classify_type — ParamClass classification
    // -------------------------------------------------------------------

    /// Returns the ABI [`ParamClass`] for a given C type on the i686 target.
    ///
    /// This is a lower-level classification used by the codegen layer to
    /// select register classes and instruction patterns:
    ///
    /// - [`ParamClass::Integer`] — scalar integers, pointers, enums.
    /// - [`ParamClass::X87`] — floating-point types processed through the
    ///   x87 FPU.
    /// - [`ParamClass::Memory`] — aggregates (structs, unions, arrays) and
    ///   complex types that must reside in memory.
    /// - [`ParamClass::NoClass`] — `void` and function types that cannot
    ///   be passed or returned in a meaningful register class.
    pub fn classify_type(&self, ty: &CType, target: &Target) -> ParamClass {
        let canonical = ty.canonical();

        // Void yields NoClass — no register assignment possible.
        if matches!(canonical, CType::Void) {
            return ParamClass::NoClass;
        }

        // Integer types (including bool, char, short, int, long, long long,
        // enum) and pointers map to the Integer class.
        if canonical.is_integer() || canonical.is_pointer() {
            return ParamClass::Integer;
        }

        // Floating-point types (float, double, long double) use the x87
        // register class on i686 — there is no SSE-based parameter passing.
        if canonical.is_floating() {
            return ParamClass::X87;
        }

        // Aggregates (struct, union, array) are always classified as Memory.
        if canonical.is_aggregate() {
            return ParamClass::Memory;
        }

        // _Complex types are treated as small aggregates.
        if matches!(canonical, CType::Complex(_)) {
            return ParamClass::Memory;
        }

        // Function types are not directly classifiable.
        if matches!(canonical, CType::Function { .. }) {
            return ParamClass::NoClass;
        }

        // _Atomic(T) — classify the underlying type.
        if let CType::Atomic(inner) = canonical {
            return self.classify_type(inner, target);
        }

        // Fallback for any edge-case types.
        ParamClass::Memory
    }

    // -------------------------------------------------------------------
    // Register helpers
    // -------------------------------------------------------------------

    /// Returns the stack pointer register for frame layout (`ESP` on i686).
    ///
    /// Used by the prologue/epilogue emitters and frame-pointer elimination
    /// passes.  The stack pointer must satisfy the alignment requirement
    /// from [`Target::stack_alignment()`] (16 bytes) at every CALL site.
    #[inline]
    pub fn stack_pointer_reg(&self) -> PhysReg {
        ESP
    }

    /// Returns the frame pointer register (`EBP` on i686).
    ///
    /// Established by the standard prologue (`PUSH EBP; MOV EBP, ESP`).
    /// If frame-pointer omission is active, EBP may instead be used as a
    /// general-purpose callee-saved register.
    #[inline]
    pub fn frame_pointer_reg(&self) -> PhysReg {
        EBP
    }

    /// Returns the registers clobbered (caller-saved) by a CALL instruction.
    ///
    /// Under cdecl, **EAX**, **ECX**, and **EDX** are not preserved across
    /// calls.  The caller must save them before the call if their values are
    /// needed afterwards.
    #[inline]
    pub fn call_clobbered_regs(&self) -> [PhysReg; 3] {
        [EAX, ECX, EDX]
    }

    /// Returns the callee-saved (non-volatile) registers under cdecl.
    ///
    /// The callee must preserve **EBX**, **ESI**, **EDI**, and **EBP** by
    /// saving them in the prologue and restoring them in the epilogue.
    #[inline]
    pub fn callee_preserved_regs(&self) -> &'static [PhysReg] {
        &CALLEE_SAVED
    }

    /// Returns the caller-saved (volatile) register set.
    ///
    /// This is equivalent to [`call_clobbered_regs`](Self::call_clobbered_regs)
    /// but returns the canonical static slice from the register definitions.
    #[inline]
    pub fn caller_saved_regs(&self) -> &'static [PhysReg] {
        &CALLER_SAVED
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Classifies a struct/union return type using field information.
    ///
    /// On i686 with GCC-compatible behaviour (`-freg-struct-return`):
    /// - Struct/union whose total size ≤ 4 bytes → returned in `EAX`.
    /// - Struct/union whose total size is 5–8 bytes → returned in `EDX:EAX`.
    /// - Struct/union > 8 bytes → returned via hidden sret pointer.
    ///
    /// The `fields` slice (from the `CType::Struct` / `CType::Union`
    /// variant) is inspected for ABI-relevant properties such as the
    /// presence of bitfields.
    fn classify_struct_return(
        &self,
        fields: &[FieldDef],
        canonical: &CType,
        target: &Target,
    ) -> ReturnClassification {
        let sz = size_of(canonical, target);

        // Empty struct (GCC extension) — treat as void.
        if sz == 0 || fields.is_empty() {
            return ReturnClassification::Void;
        }

        // Examine aggregate field composition for diagnostics and potential
        // future ABI refinements.  For standard cdecl, the classification
        // is purely size-based, but recording field metadata enables debug
        // verification of struct layout.
        let _field_count = fields.len();
        let _has_bitfield = fields.iter().any(|f: &FieldDef| f.bit_width.is_some());
        let _max_field_align = fields
            .iter()
            .map(|f: &FieldDef| align_of(&f.ty, target))
            .max()
            .unwrap_or(1);

        if sz <= 4 {
            ReturnClassification::InRegister { reg: EAX }
        } else if sz <= MAX_REG_RETURN_SIZE {
            ReturnClassification::RegisterPair {
                lo: EAX,
                hi: EDX,
            }
        } else {
            ReturnClassification::Indirect
        }
    }

    /// Computes the number of bytes an argument occupies on the stack.
    ///
    /// # Promotion Rules
    ///
    /// | C Type                | Stack Size (bytes) | Notes                            |
    /// |-----------------------|--------------------|----------------------------------|
    /// | `_Bool`, `char`       | 4                  | Integer-promoted to `int`        |
    /// | `short`               | 4                  | Integer-promoted to `int`        |
    /// | `int`, `long`         | 4                  | Native 32-bit on ILP32           |
    /// | pointer, enum         | 4                  | 32-bit address / underlying int  |
    /// | `long long`           | 8                  | 4-byte aligned on i686 stack     |
    /// | `float`               | 8                  | Promoted to `double` via x87     |
    /// | `double`              | 8                  | 8-byte IEEE 754                  |
    /// | `long double`         | 12                 | 80-bit extended + 2B padding     |
    /// | struct / union        | `size_of`, ≥ 4     | Copied by value, 4-byte aligned  |
    /// | array                 | 4                  | Decays to pointer                |
    /// | function              | 4                  | Decays to function pointer       |
    /// | `_Complex T`          | `2 × size_of(T)`   | Treated as two components        |
    fn compute_arg_stack_size(&self, ty: &CType, target: &Target) -> u32 {
        match ty {
            // -- Sub-int types: promoted to 4 bytes (minimum stack slot) --
            CType::Bool | CType::Char { .. } | CType::Short { .. } => STACK_SLOT_MIN,

            // -- int, long: 4 bytes on ILP32 --
            CType::Int { .. } | CType::Long { .. } => {
                let natural = size_of(ty, target) as u32;
                // Guarantee the minimum slot size.
                natural.max(STACK_SLOT_MIN)
            }

            // -- pointer: 4 bytes (ILP32 address width) --
            CType::Pointer(_) => {
                // Query the target for pointer width to remain generic.
                let pw = target.pointer_width();
                pw.max(STACK_SLOT_MIN)
            }

            // -- enum: underlying integer size, at least 4 bytes --
            CType::Enum { .. } => {
                let natural = size_of(ty, target) as u32;
                natural.max(STACK_SLOT_MIN)
            }

            // -- long long: 8 bytes (4-byte aligned on i686) --
            CType::LongLong { .. } => {
                let natural = size_of(ty, target) as u32;
                round_up_u32(natural.max(8), STACK_SLOT_MIN)
            }

            // -- float: promoted to double → 8 bytes via x87 FPU --
            CType::Float => FLOAT_PROMOTED_SIZE,

            // -- double: 8 bytes (4-byte aligned on i686 stack) --
            CType::Double => {
                let natural = size_of(ty, target) as u32;
                round_up_u32(natural.max(8), STACK_SLOT_MIN)
            }

            // -- long double: target-specific (12 bytes on i686) --
            CType::LongDouble => {
                let ld_size = target.long_double_size();
                round_up_u32(ld_size.max(STACK_SLOT_MIN), STACK_SLOT_MIN)
            }

            // -- struct / union: entire value copied, 4-byte-aligned --
            CType::Struct { .. } | CType::Union { .. } => {
                let natural = size_of(ty, target) as u32;
                round_up_u32(natural.max(STACK_SLOT_MIN), STACK_SLOT_MIN)
            }

            // -- array: decays to pointer (4 bytes on ILP32) --
            CType::Array { .. } => {
                let pw = target.pointer_width();
                pw.max(STACK_SLOT_MIN)
            }

            // -- function type: decays to function pointer --
            CType::Function { .. } => {
                let pw = target.pointer_width();
                pw.max(STACK_SLOT_MIN)
            }

            // -- _Complex(T): 2 × component size --
            CType::Complex(base) => {
                let component = self.compute_arg_stack_size(base, target);
                let total = 2 * component;
                round_up_u32(total, STACK_SLOT_MIN)
            }

            // -- void: should never be an argument; return 0 --
            CType::Void => 0,

            // -- _Atomic(T): unwrap and classify inner --
            CType::Atomic(inner) => self.compute_arg_stack_size(inner, target),

            // -- typedef(T): unwrap and classify underlying --
            CType::Typedef { underlying, .. } => {
                self.compute_arg_stack_size(underlying, target)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` must be a power of two and greater than zero.
#[inline]
fn round_up_u32(value: u32, align: u32) -> u32 {
    debug_assert!(align > 0 && align.is_power_of_two(), "align must be a positive power of 2");
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a minimal i686 target for tests.
    fn i686_target() -> Target {
        Target::I686
    }

    // ----- round_up_u32 ------------------------------------------------

    #[test]
    fn test_round_up_u32_exact() {
        assert_eq!(round_up_u32(16, 4), 16);
        assert_eq!(round_up_u32(16, 16), 16);
    }

    #[test]
    fn test_round_up_u32_needs_padding() {
        assert_eq!(round_up_u32(1, 4), 4);
        assert_eq!(round_up_u32(5, 4), 8);
        assert_eq!(round_up_u32(13, 16), 16);
        assert_eq!(round_up_u32(17, 16), 32);
    }

    // ----- classify_return for scalar types -----------------------------

    #[test]
    fn test_return_void() {
        let abi = I686Abi::new();
        let t = i686_target();
        assert_eq!(abi.classify_return(&CType::Void, &t), ReturnClassification::Void);
    }

    #[test]
    fn test_return_int_in_eax() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::Int { signed: true }, &t);
        assert_eq!(cls, ReturnClassification::InRegister { reg: EAX });
    }

    #[test]
    fn test_return_long_long_in_edx_eax() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::LongLong { signed: true }, &t);
        assert_eq!(
            cls,
            ReturnClassification::RegisterPair { lo: EAX, hi: EDX }
        );
    }

    #[test]
    fn test_return_double_in_st0() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::Double, &t);
        assert_eq!(cls, ReturnClassification::X87 { reg: ST0 });
    }

    #[test]
    fn test_return_float_in_st0() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::Float, &t);
        assert_eq!(cls, ReturnClassification::X87 { reg: ST0 });
    }

    #[test]
    fn test_return_pointer_in_eax() {
        let abi = I686Abi::new();
        let t = i686_target();
        let ty = CType::Pointer(Box::new(CType::Int { signed: true }));
        let cls = abi.classify_return(&ty, &t);
        assert_eq!(cls, ReturnClassification::InRegister { reg: EAX });
    }

    #[test]
    fn test_return_bool_in_eax() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::Bool, &t);
        assert_eq!(cls, ReturnClassification::InRegister { reg: EAX });
    }

    #[test]
    fn test_return_char_in_eax() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_return(&CType::Char { signed: true }, &t);
        assert_eq!(cls, ReturnClassification::InRegister { reg: EAX });
    }

    // ----- classify_argument -------------------------------------------

    #[test]
    fn test_arg_int_on_stack() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_argument(&CType::Int { signed: true }, &t);
        match cls {
            ArgClassification::Stack { offset, size } => {
                assert_eq!(offset, 0);
                assert_eq!(size, 4);
            }
            _ => panic!("expected Stack classification"),
        }
    }

    #[test]
    fn test_arg_char_promoted_to_4bytes() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_argument(&CType::Char { signed: true }, &t);
        match cls {
            ArgClassification::Stack { size, .. } => assert_eq!(size, 4),
            _ => panic!("expected Stack classification"),
        }
    }

    #[test]
    fn test_arg_float_promoted_to_8bytes() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_argument(&CType::Float, &t);
        match cls {
            ArgClassification::Stack { size, .. } => assert_eq!(size, FLOAT_PROMOTED_SIZE),
            _ => panic!("expected Stack classification"),
        }
    }

    #[test]
    fn test_arg_double_8bytes() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_argument(&CType::Double, &t);
        match cls {
            ArgClassification::Stack { size, .. } => assert_eq!(size, 8),
            _ => panic!("expected Stack classification"),
        }
    }

    #[test]
    fn test_arg_long_long_8bytes() {
        let abi = I686Abi::new();
        let t = i686_target();
        let cls = abi.classify_argument(&CType::LongLong { signed: true }, &t);
        match cls {
            ArgClassification::Stack { size, .. } => assert_eq!(size, 8),
            _ => panic!("expected Stack classification"),
        }
    }

    // ----- compute_stack_layout ----------------------------------------

    #[test]
    fn test_layout_no_params() {
        let abi = I686Abi::new();
        let t = i686_target();
        let layout = abi.compute_stack_layout(&[], &t);
        assert_eq!(layout.total_size, 0);
        assert!(layout.arg_offsets.is_empty());
        assert!(!layout.has_sret);
        assert_eq!(layout.sret_offset, None);
    }

    #[test]
    fn test_layout_two_ints() {
        let abi = I686Abi::new();
        let t = i686_target();
        let params = [
            CType::Int { signed: true },
            CType::Int { signed: false },
        ];
        let layout = abi.compute_stack_layout(&params, &t);

        // Each int is 4 bytes → 8 bytes raw, padded to 16 for alignment.
        assert_eq!(layout.total_size, 16);
        assert_eq!(layout.arg_offsets.len(), 2);

        match &layout.arg_offsets[0] {
            ArgClassification::Stack { offset, size } => {
                assert_eq!(*offset, 0);
                assert_eq!(*size, 4);
            }
            _ => panic!("expected Stack"),
        }
        match &layout.arg_offsets[1] {
            ArgClassification::Stack { offset, size } => {
                assert_eq!(*offset, 4);
                assert_eq!(*size, 4);
            }
            _ => panic!("expected Stack"),
        }
    }

    #[test]
    fn test_layout_mixed_types() {
        let abi = I686Abi::new();
        let t = i686_target();
        let params = [
            CType::Int { signed: true },           // 4 bytes at offset 0
            CType::Double,                         // 8 bytes at offset 4
            CType::Char { signed: false },         // 4 bytes (promoted) at offset 12
        ];
        let layout = abi.compute_stack_layout(&params, &t);

        // 4 + 8 + 4 = 16 raw bytes → 16 already aligned.
        assert_eq!(layout.total_size, 16);
        assert_eq!(layout.arg_offsets.len(), 3);
    }

    // ----- classify_type -----------------------------------------------

    #[test]
    fn test_classify_type_integer() {
        let abi = I686Abi::new();
        let t = i686_target();
        assert_eq!(
            abi.classify_type(&CType::Int { signed: true }, &t),
            ParamClass::Integer
        );
    }

    #[test]
    fn test_classify_type_pointer() {
        let abi = I686Abi::new();
        let t = i686_target();
        let ty = CType::Pointer(Box::new(CType::Void));
        assert_eq!(abi.classify_type(&ty, &t), ParamClass::Integer);
    }

    #[test]
    fn test_classify_type_double() {
        let abi = I686Abi::new();
        let t = i686_target();
        assert_eq!(abi.classify_type(&CType::Double, &t), ParamClass::X87);
    }

    #[test]
    fn test_classify_type_void() {
        let abi = I686Abi::new();
        let t = i686_target();
        assert_eq!(abi.classify_type(&CType::Void, &t), ParamClass::NoClass);
    }

    // ----- register helpers --------------------------------------------

    #[test]
    fn test_stack_pointer_reg() {
        let abi = I686Abi::new();
        assert_eq!(abi.stack_pointer_reg(), ESP);
    }

    #[test]
    fn test_frame_pointer_reg() {
        let abi = I686Abi::new();
        assert_eq!(abi.frame_pointer_reg(), EBP);
    }

    #[test]
    fn test_call_clobbered() {
        let abi = I686Abi::new();
        let regs = abi.call_clobbered_regs();
        // EAX, ECX, EDX are caller-saved.
        assert!(regs.contains(&EAX));
        assert!(regs.contains(&ECX));
        assert!(regs.contains(&EDX));
    }

    #[test]
    fn test_callee_preserved() {
        let abi = I686Abi::new();
        let regs = abi.callee_preserved_regs();
        assert!(!regs.is_empty());
        // EBP must be in the callee-saved set.
        assert!(regs.contains(&EBP));
    }

    #[test]
    fn test_caller_saved_set() {
        let abi = I686Abi::new();
        let regs = abi.caller_saved_regs();
        assert!(regs.contains(&EAX));
        assert!(regs.contains(&ECX));
        assert!(regs.contains(&EDX));
    }

    // ----- Target-specific constants -----------------------------------

    #[test]
    fn test_target_ilp32_properties() {
        let t = i686_target();
        // ILP32: pointer is 4 bytes, long is 4 bytes.
        assert_eq!(t.pointer_width(), 4);
        assert_eq!(t.long_size(), 4);
        // Long double is 12 bytes (80-bit + 2 bytes padding on i686).
        assert_eq!(t.long_double_size(), 12);
        // Call-site alignment is 16 bytes.
        assert_eq!(t.stack_alignment(), 16);
    }
}
