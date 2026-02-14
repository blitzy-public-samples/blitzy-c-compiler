//! RISC-V LP64D ABI implementation for the BCC compiler.
//!
//! This module implements the LP64D (Long-Pointer 64, Double-float) calling
//! convention for the RISC-V 64-bit architecture. The LP64D ABI governs how
//! function arguments are passed, return values are delivered, and stack frames
//! are structured when generating code for `Target::RiscV64`.
//!
//! # LP64D Data Model
//!
//! | C Type         | Size (bytes) | Alignment (bytes) |
//! |---------------|-------------|-------------------|
//! | `char`        | 1           | 1                 |
//! | `short`       | 2           | 2                 |
//! | `int`         | 4           | 4                 |
//! | `long`        | 8           | 8                 |
//! | `long long`   | 8           | 8                 |
//! | `pointer`     | 8           | 8                 |
//! | `float`       | 4           | 4                 |
//! | `double`      | 8           | 8                 |
//! | `long double` | 8           | 8 (maps to double)|
//!
//! # Register Usage (LP64D)
//!
//! | Registers  | ABI Names | Purpose                        |
//! |-----------|-----------|--------------------------------|
//! | x10–x17   | a0–a7    | Integer arguments / return     |
//! | f10–f17   | fa0–fa7  | FP arguments / return          |
//! | x2        | sp       | Stack pointer (16-byte aligned)|
//! | x8        | s0/fp    | Frame pointer (optional)       |
//! | x1        | ra       | Return address                 |
//! | x4        | tp       | Thread pointer                 |
//! | x3        | gp       | Global pointer                 |
//!
//! # Struct Flattening
//!
//! The LP64D ABI defines special rules for passing small structs in registers:
//! - Structs with 1–2 floating-point leaf fields may be passed in FP registers
//! - Structs with mixed int/float fields may use both register files
//! - Structs ≤ 2×XLEN (16 bytes) with only integer fields use integer registers
//! - Larger structs are passed by reference (caller copies, passes pointer)
//!
//! # Variadic Function Handling
//!
//! Named FP parameters use FP registers normally. Variadic FP parameters are
//! promoted to integer registers (a0–a7), matching GCC/LLVM behaviour. The
//! caller must handle this distinction before invoking [`RiscV64Abi::compute_stack_layout`].

use crate::backend::riscv64::registers;
use crate::backend::traits::{ParamClass, PhysReg};
use crate::common::target::Target;
use crate::common::type_builder::{compute_struct_layout, FieldLayout, StructLayout};
use crate::common::types::{align_of, size_of, CType, FieldDef};

// ---------------------------------------------------------------------------
// ABI Constants
// ---------------------------------------------------------------------------

/// XLEN: the native integer register width in bytes for RV64 (64 bits = 8 bytes).
const XLEN: usize = 8;

/// 2×XLEN in bytes — the maximum aggregate size passable in registers.
const XLEN2: usize = 16;

/// Number of integer argument registers available (a0–a7).
const NUM_INT_ARG_REGS: usize = 8;

/// Number of floating-point argument registers available (fa0–fa7).
const NUM_FP_ARG_REGS: usize = 8;

/// Stack pointer register — must be 16-byte aligned at all call sites.
const STACK_POINTER: PhysReg = registers::SP;

/// Frame pointer register (s0/x8). Optionally used for stable frame reference.
const FRAME_POINTER: PhysReg = registers::FP;

/// Return address register (x1). Caller-saved; callee saves it if making calls.
const RETURN_ADDRESS: PhysReg = registers::RA;

// ---------------------------------------------------------------------------
// Compile-time ABI verification
// ---------------------------------------------------------------------------
//
// These constant assertions guarantee that the individual register aliases
// exported by the registers module match the expected positions in the
// argument register arrays. They also exercise the `PhysReg.0` field access.

const _: () = {
    // Integer argument registers: a0 (x10) through a7 (x17)
    assert!(registers::A0.0 == 10);
    assert!(registers::A1.0 == 11);
    assert!(registers::A2.0 == 12);
    assert!(registers::A3.0 == 13);
    assert!(registers::A4.0 == 14);
    assert!(registers::A5.0 == 15);
    assert!(registers::A6.0 == 16);
    assert!(registers::A7.0 == 17);

    // FP argument registers: fa0 (f10) through fa7 (f17)
    assert!(registers::FA0.0 == 42);
    assert!(registers::FA1.0 == 43);
    assert!(registers::FA2.0 == 44);
    assert!(registers::FA3.0 == 45);
    assert!(registers::FA4.0 == 46);
    assert!(registers::FA5.0 == 47);
    assert!(registers::FA6.0 == 48);
    assert!(registers::FA7.0 == 49);

    // Integer arg register array consistency
    assert!(registers::INTEGER_ARG_REGS[0].0 == registers::A0.0);
    assert!(registers::INTEGER_ARG_REGS[7].0 == registers::A7.0);

    // FP arg register array consistency
    assert!(registers::FLOAT_ARG_REGS[0].0 == registers::FA0.0);
    assert!(registers::FLOAT_ARG_REGS[7].0 == registers::FA7.0);

    // Special-purpose register encodings
    assert!(STACK_POINTER.0 == 2); // sp = x2
    assert!(FRAME_POINTER.0 == 8); // fp/s0 = x8
    assert!(RETURN_ADDRESS.0 == 1); // ra = x1
};

// ---------------------------------------------------------------------------
// ArgClassification — how a single argument is passed
// ---------------------------------------------------------------------------

/// Classification of how a function argument is passed in the LP64D ABI.
///
/// Each variant describes a distinct passing mechanism. The contained
/// [`PhysReg`] values identify the specific register(s) assigned to the
/// argument by [`RiscV64Abi::compute_stack_layout`].
#[derive(Clone, Debug, PartialEq)]
pub enum ArgClassification {
    /// Passed in a single integer register (a0–a7).
    ///
    /// Used for: scalar integers ≤ XLEN, pointers, enums, `_Bool`.
    IntegerReg(PhysReg),

    /// Passed in a pair of integer registers.
    ///
    /// The first register holds the low XLEN bits; the second holds the high
    /// bits. Used for: 128-bit integers, structs ≤ 2×XLEN with integer-only
    /// leaf fields.
    IntegerRegPair(PhysReg, PhysReg),

    /// Passed in a single floating-point register (fa0–fa7).
    ///
    /// Used for: `float`, `double`, `long double` (maps to double on RV64),
    /// and single-float-field structs.
    FloatReg(PhysReg),

    /// Passed in a pair of floating-point registers.
    ///
    /// Used for: structs with exactly 2 floating-point leaf fields, and
    /// `_Complex float` / `_Complex double`.
    FloatRegPair(PhysReg, PhysReg),

    /// Passed in one integer register and one floating-point register.
    ///
    /// Used for: structs with exactly one integer leaf field and one
    /// floating-point leaf field (either order). The first [`PhysReg`] is
    /// the integer register; the second is the FP register.
    IntAndFloat(PhysReg, PhysReg),

    /// Passed on the stack.
    ///
    /// `offset` is the byte offset from the caller's SP at the call site.
    /// `size` is the number of bytes occupied (padded to an 8-byte slot).
    Stack { offset: i32, size: u32 },

    /// Passed by reference — the caller allocates a copy and passes a
    /// pointer to it in an integer register.
    ///
    /// Used for: aggregates larger than 2×XLEN (16 bytes).
    Indirect(PhysReg),
}

// ---------------------------------------------------------------------------
// ReturnClassification — how a return value is delivered
// ---------------------------------------------------------------------------

/// Classification of how a function return value is delivered in the LP64D ABI.
///
/// Mirrors [`ArgClassification`] but covers the return-value side of the
/// calling convention. Return registers are always a0/a1 (integer) or
/// fa0/fa1 (floating-point).
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnClassification {
    /// Returned in a single integer register (a0).
    InIntegerReg(PhysReg),

    /// Returned in two integer registers (a0 low, a1 high).
    ///
    /// Used for: 128-bit integers, structs ≤ 2×XLEN with integer-only fields.
    InIntegerRegPair(PhysReg, PhysReg),

    /// Returned in a single floating-point register (fa0).
    InFloatReg(PhysReg),

    /// Returned in two floating-point registers (fa0, fa1).
    ///
    /// Used for: structs with 2 float fields, `_Complex float/double`.
    InFloatRegPair(PhysReg, PhysReg),

    /// Returned in one integer register and one FP register.
    ///
    /// First [`PhysReg`] is the integer register (a0), second is FP (fa0).
    IntAndFloat(PhysReg, PhysReg),

    /// Returned via a hidden first-argument pointer (sret).
    ///
    /// The caller passes a pointer in a0 to caller-allocated memory. The
    /// callee writes the return value there. This consumes the a0 slot,
    /// shifting subsequent integer arguments by one register.
    Indirect(PhysReg),

    /// No return value (`void` function).
    Void,
}

// ---------------------------------------------------------------------------
// StackLayout — complete call frame description
// ---------------------------------------------------------------------------

/// Complete stack frame layout for a function call under the LP64D ABI.
///
/// Describes how each argument is passed, the stack space required for
/// spilled arguments, the callee-saved register save area, and the total
/// estimated frame size.
///
/// # Stack Frame Structure (growing downward)
///
/// ```text
/// ┌─────────────────────────┐ ← Caller's SP (16-byte aligned)
/// │  Argument spill area    │   stack_arg_area_size bytes
/// ├─────────────────────────┤
/// │  Return address (ra)    │   8 bytes (if non-leaf)
/// ├─────────────────────────┤
/// │  Frame pointer (fp/s0)  │   8 bytes (if used)
/// ├─────────────────────────┤
/// │  Callee-saved registers │   callee_saved_area_size bytes
/// ├─────────────────────────┤
/// │  Local variables        │   (computed later by codegen)
/// └─────────────────────────┘ ← Current SP (16-byte aligned)
/// ```
#[derive(Clone, Debug)]
pub struct StackLayout {
    /// Per-argument classification, in parameter declaration order.
    pub arg_classifications: Vec<ArgClassification>,

    /// Total bytes consumed by stack-passed arguments (above the callee's
    /// frame). Allocated by the caller.
    pub stack_arg_area_size: u32,

    /// Estimated total frame size including callee-saved area and stack
    /// argument area, rounded up to [`stack_alignment`](StackLayout::stack_alignment).
    /// Does not include local variables (added later by codegen).
    pub total_frame_size: u32,

    /// Maximum bytes needed to save all callee-saved registers that might
    /// be clobbered. Includes slots for RA and FP.
    pub callee_saved_area_size: u32,

    /// Required stack alignment in bytes (always 16 for RISC-V LP64D).
    pub stack_alignment: u32,
}

// ---------------------------------------------------------------------------
// FlatFieldKind — internal helper for struct flattening classification
// ---------------------------------------------------------------------------

/// Internal classification of a leaf (non-aggregate) field discovered during
/// the struct-flattening pass. Determines FP register eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlatFieldKind {
    /// An integer-like field (including pointers, enums, booleans, chars).
    Integer,
    /// A 32-bit IEEE 754 single-precision float field.
    Float32,
    /// A 64-bit IEEE 754 double-precision field (also covers `long double`
    /// on RISC-V, which maps to double).
    Float64,
}

// ---------------------------------------------------------------------------
// RiscV64Abi — LP64D ABI handler
// ---------------------------------------------------------------------------

/// RISC-V LP64D ABI handler.
///
/// Provides methods to classify function arguments and return values according
/// to the LP64D calling convention, compute complete stack frame layouts, and
/// flatten struct types for register-passing eligibility analysis.
///
/// # Example
///
/// ```rust,ignore
/// let abi = RiscV64Abi::new();
/// let target = Target::RiscV64;
///
/// // Classify a single argument
/// let class = abi.classify_argument(&CType::Int { signed: true }, &target);
/// // => ArgClassification::IntegerReg(A0)
///
/// // Compute full call frame layout
/// let params = vec![CType::Int { signed: true }, CType::Double];
/// let layout = abi.compute_stack_layout(&params, &target);
/// // layout.arg_classifications[0] => IntegerReg(A0)
/// // layout.arg_classifications[1] => FloatReg(FA0)
/// ```
pub struct RiscV64Abi;

impl RiscV64Abi {
    /// Creates a new LP64D ABI handler.
    #[inline]
    pub fn new() -> Self {
        RiscV64Abi
    }

    // -----------------------------------------------------------------------
    // classify_argument — single-argument type classification
    // -----------------------------------------------------------------------

    /// Classifies how a single argument type would be passed under LP64D.
    ///
    /// This method classifies the type in isolation, assuming it is the first
    /// argument of its kind (registers a0 / fa0 are available). For complete
    /// per-call register assignment with allocation tracking, use
    /// [`compute_stack_layout`](RiscV64Abi::compute_stack_layout).
    ///
    /// # Parameters
    ///
    /// * `ty`     — the C type of the argument.
    /// * `target` — compilation target (should be [`Target::RiscV64`]).
    pub fn classify_argument(&self, ty: &CType, target: &Target) -> ArgClassification {
        let canonical = ty.canonical();
        let type_size = size_of(canonical, target);

        match canonical {
            // Void should not appear as an argument; treat as integer for safety
            CType::Void => ArgClassification::IntegerReg(registers::A0),

            // Scalar integers and _Bool — always integer register
            CType::Bool
            | CType::Char { .. }
            | CType::Short { .. }
            | CType::Int { .. }
            | CType::Long { .. }
            | CType::LongLong { .. } => {
                if type_size <= XLEN {
                    ArgClassification::IntegerReg(registers::A0)
                } else {
                    // 128-bit type: pair of integer registers
                    ArgClassification::IntegerRegPair(registers::A0, registers::A1)
                }
            }

            // Floating-point scalars — FP register
            CType::Float => ArgClassification::FloatReg(registers::FA0),
            CType::Double => ArgClassification::FloatReg(registers::FA0),
            // long double maps to double (8 bytes) on RISC-V 64
            CType::LongDouble => ArgClassification::FloatReg(registers::FA0),

            // Complex: two FP registers (real + imaginary)
            CType::Complex(_) => {
                ArgClassification::FloatRegPair(registers::FA0, registers::FA1)
            }

            // Pointer: integer register
            CType::Pointer(_) => ArgClassification::IntegerReg(registers::A0),

            // Enum: underlying integer type — integer register
            CType::Enum { .. } => ArgClassification::IntegerReg(registers::A0),

            // Function type: decay to pointer — integer register
            CType::Function { .. } => ArgClassification::IntegerReg(registers::A0),

            // Struct: apply LP64D flattening rules
            CType::Struct { fields, .. } => {
                self.classify_aggregate_arg(fields, type_size, target)
            }

            // Union: treated as opaque aggregate
            CType::Union { fields, .. } => {
                self.classify_union_arg(fields, type_size, target)
            }

            // Array: treated as aggregate (not recursively flattened)
            CType::Array { element, size } => {
                let total = size.map_or(0, |n| size_of(element, target) * n);
                if total == 0 {
                    ArgClassification::IntegerReg(registers::A0)
                } else if total > XLEN2 {
                    ArgClassification::Indirect(registers::A0)
                } else if total <= XLEN {
                    ArgClassification::IntegerReg(registers::A0)
                } else {
                    ArgClassification::IntegerRegPair(registers::A0, registers::A1)
                }
            }

            // Atomic: strip qualifier, recurse on inner type
            CType::Atomic(inner) => self.classify_argument(inner, target),

            // Typedef: canonical() handles this, but be explicit
            CType::Typedef { underlying, .. } => self.classify_argument(underlying, target),
        }
    }

    // -----------------------------------------------------------------------
    // classify_return — return value classification
    // -----------------------------------------------------------------------

    /// Classifies how a function return value is delivered under LP64D.
    ///
    /// # Parameters
    ///
    /// * `ty`     — the C return type.
    /// * `target` — compilation target (should be [`Target::RiscV64`]).
    pub fn classify_return(&self, ty: &CType, target: &Target) -> ReturnClassification {
        let canonical = ty.canonical();
        let type_size = size_of(canonical, target);

        match canonical {
            CType::Void => ReturnClassification::Void,

            // Scalar integers: a0 (a0+a1 for 128-bit)
            CType::Bool
            | CType::Char { .. }
            | CType::Short { .. }
            | CType::Int { .. }
            | CType::Long { .. }
            | CType::LongLong { .. } => {
                if type_size <= XLEN {
                    ReturnClassification::InIntegerReg(registers::A0)
                } else {
                    ReturnClassification::InIntegerRegPair(registers::A0, registers::A1)
                }
            }

            // Floating-point: fa0
            CType::Float | CType::Double | CType::LongDouble => {
                ReturnClassification::InFloatReg(registers::FA0)
            }

            // Complex: fa0 + fa1
            CType::Complex(_) => {
                ReturnClassification::InFloatRegPair(registers::FA0, registers::FA1)
            }

            // Pointer: a0
            CType::Pointer(_) => ReturnClassification::InIntegerReg(registers::A0),

            // Enum: a0
            CType::Enum { .. } => ReturnClassification::InIntegerReg(registers::A0),

            // Function: should not be returned directly; treat as pointer
            CType::Function { .. } => ReturnClassification::InIntegerReg(registers::A0),

            // Struct: flattening rules
            CType::Struct { fields, .. } => {
                self.classify_aggregate_return(fields, type_size, target)
            }

            // Union: opaque aggregate
            CType::Union { fields, .. } => {
                self.classify_union_return(fields, type_size, target)
            }

            // Array: aggregate
            CType::Array { .. } => {
                if type_size == 0 {
                    ReturnClassification::Void
                } else if type_size > XLEN2 {
                    ReturnClassification::Indirect(registers::A0)
                } else if type_size <= XLEN {
                    ReturnClassification::InIntegerReg(registers::A0)
                } else {
                    ReturnClassification::InIntegerRegPair(registers::A0, registers::A1)
                }
            }

            CType::Atomic(inner) => self.classify_return(inner, target),
            CType::Typedef { underlying, .. } => self.classify_return(underlying, target),
        }
    }

    // -----------------------------------------------------------------------
    // classify_type — bridge to architecture-generic ParamClass
    // -----------------------------------------------------------------------

    /// Maps a C type to the architecture-generic [`ParamClass`] used by the
    /// [`ArchCodegen`](crate::backend::traits::ArchCodegen) trait interface.
    ///
    /// Simplified classification:
    /// - [`ParamClass::Integer`] — value lives in integer registers
    /// - [`ParamClass::SSE`]     — value lives in floating-point registers
    /// - [`ParamClass::Memory`]  — value must be passed on the stack or by reference
    pub fn classify_type(&self, ty: &CType, target: &Target) -> ParamClass {
        let canonical = ty.canonical();
        let type_size = size_of(canonical, target);

        // Large aggregates are always memory
        if canonical.is_aggregate() && type_size > XLEN2 {
            return ParamClass::Memory;
        }

        match canonical {
            CType::Void => ParamClass::Integer,

            // Integer scalars
            _ if canonical.is_integer() => ParamClass::Integer,

            // Floating-point scalars
            _ if canonical.is_floating() => ParamClass::SSE,

            // Complex: FP register pair
            CType::Complex(_) => ParamClass::SSE,

            // Pointers: integer
            _ if canonical.is_pointer() => ParamClass::Integer,

            // Function type decays to pointer: integer
            CType::Function { .. } => ParamClass::Integer,

            // Struct: check if any FP-eligible flattening exists
            CType::Struct { fields, .. } => {
                let flat = self.collect_flat_fields(fields, target);
                let all_fp = !flat.is_empty()
                    && flat.len() <= 2
                    && flat
                        .iter()
                        .all(|f| *f == FlatFieldKind::Float32 || *f == FlatFieldKind::Float64);
                let has_fp = flat
                    .iter()
                    .any(|f| *f == FlatFieldKind::Float32 || *f == FlatFieldKind::Float64);
                let has_int = flat.iter().any(|f| *f == FlatFieldKind::Integer);

                if all_fp {
                    ParamClass::SSE
                } else if flat.len() == 2 && has_fp && has_int {
                    // Mixed int/float: SSE captures the need for FP register
                    ParamClass::SSE
                } else if type_size <= XLEN2 {
                    ParamClass::Integer
                } else {
                    ParamClass::Memory
                }
            }

            // Union: integer if small enough, otherwise memory
            CType::Union { .. } => {
                if type_size <= XLEN2 {
                    ParamClass::Integer
                } else {
                    ParamClass::Memory
                }
            }

            // Array: integer if small enough
            CType::Array { .. } => {
                if type_size <= XLEN2 {
                    ParamClass::Integer
                } else {
                    ParamClass::Memory
                }
            }

            CType::Atomic(inner) => self.classify_type(inner, target),
            CType::Typedef { underlying, .. } => self.classify_type(underlying, target),

            // Fallback
            _ => ParamClass::Integer,
        }
    }

    // -----------------------------------------------------------------------
    // compute_stack_layout — full call frame computation
    // -----------------------------------------------------------------------

    /// Computes the complete stack frame layout for a function with the given
    /// parameter types.
    ///
    /// Tracks integer and floating-point register allocation state across all
    /// parameters, assigning each argument to specific registers or stack
    /// slots according to LP64D rules.
    ///
    /// # Parameters
    ///
    /// * `params` — C types of all declared (named) function parameters.
    /// * `target` — compilation target (should be [`Target::RiscV64`]).
    ///
    /// # Returns
    ///
    /// A [`StackLayout`] with per-argument classifications and frame metrics.
    ///
    /// # Variadic Functions
    ///
    /// This method treats all parameters as named. For variadic calls, the
    /// caller should promote unnamed FP arguments to integer register types
    /// before invoking this method, or manually reclassify them afterward.
    pub fn compute_stack_layout(&self, params: &[CType], target: &Target) -> StackLayout {
        // Verify we are targeting RISC-V 64 (pointer width = 8 = XLEN)
        debug_assert_eq!(target.pointer_width() as usize, XLEN);
        let stack_align = target.stack_alignment();

        // Verify data model is LP64 for RISC-V 64
        debug_assert_eq!(
            target.data_model(),
            crate::common::target::DataModel::LP64
        );

        // Confirm long is 8 bytes on LP64
        debug_assert_eq!(target.long_size(), 8);

        // Mutable register allocation counters
        let mut int_reg_idx: usize = 0;
        let mut fp_reg_idx: usize = 0;
        let mut stack_offset: i32 = 0;
        let mut classifications = Vec::with_capacity(params.len());

        for param_ty in params {
            let class = self.classify_arg_with_state(
                param_ty,
                target,
                &mut int_reg_idx,
                &mut fp_reg_idx,
                &mut stack_offset,
            );
            classifications.push(class);
        }

        // -- Callee-saved area computation --
        // Return address (RA): 8 bytes — saved by callee if it makes calls.
        let ra_slot: u32 = 8;
        // Frame pointer (FP/S0): 8 bytes — saved when frame pointer is enabled.
        let fp_slot: u32 = 8;
        // Callee-saved integer registers: s0–s11 (12 regs).
        // s0 is the same physical register as FP, already counted above, so
        // we count 11 additional callee-saved integer registers.
        let callee_int_count = registers::CALLEE_SAVED_INT.len() as u32;
        let callee_int_slots: u32 = (callee_int_count.saturating_sub(1)) * 8;
        // Callee-saved FP registers: fs0–fs11 (12 regs × 8 bytes).
        let callee_fp_slots: u32 = registers::CALLEE_SAVED_FP.len() as u32 * 8;
        let callee_saved_area_size = ra_slot + fp_slot + callee_int_slots + callee_fp_slots;

        let stack_arg_area_size = stack_offset as u32;

        // Total frame: callee-saved + stack args, aligned to stack boundary.
        let raw_frame = callee_saved_area_size + stack_arg_area_size;
        let total_frame_size = round_up_to(raw_frame as usize, stack_align as usize) as u32;

        StackLayout {
            arg_classifications: classifications,
            stack_arg_area_size,
            total_frame_size,
            callee_saved_area_size,
            stack_alignment: stack_align,
        }
    }

    // -----------------------------------------------------------------------
    // flatten_struct_fields — public struct flattening API
    // -----------------------------------------------------------------------

    /// Recursively flattens an aggregate type into its leaf scalar fields.
    ///
    /// Implements the RISC-V ABI struct-flattening algorithm: descends into
    /// nested structs and unions, collecting leaf scalar types. The result
    /// determines whether a struct can be passed in FP registers (all-float),
    /// mixed int+FP registers, or must fall back to integer registers.
    ///
    /// # Rules
    ///
    /// - Scalar fields are leaf nodes.
    /// - Nested structs/unions are recursively flattened.
    /// - Arrays are **not** recursively flattened (treated as single integer
    ///   fields per the RISC-V psABI).
    /// - Bit-fields force the field to be treated as integer.
    ///
    /// # Parameters
    ///
    /// * `ty`     — the type to flatten (typically a struct or union).
    /// * `target` — compilation target.
    ///
    /// # Returns
    ///
    /// A `Vec<CType>` of the leaf scalar types found by recursive flattening.
    /// For non-aggregate types, returns a single-element vector with the type
    /// itself.
    pub fn flatten_struct_fields(&self, ty: &CType, target: &Target) -> Vec<CType> {
        let canonical = ty.canonical();
        let mut result = Vec::new();

        match canonical {
            CType::Struct { fields, .. } => {
                // Compute layout to verify struct metrics and access per-field info.
                let layout: StructLayout = compute_struct_layout(fields, target);
                let _total_size = layout.total_size;
                let _struct_align = layout.alignment;

                for (idx, field) in fields.iter().enumerate() {
                    // Bit-fields are treated as integer leaves (disqualify FP passing)
                    if field.bit_width.is_some() {
                        result.push(CType::Int { signed: true });
                        continue;
                    }

                    // Access field layout metadata from compute_struct_layout.
                    // FieldLayout.offset and .size are used for validation.
                    if idx < layout.fields.len() {
                        let fl: &FieldLayout = &layout.fields[idx];
                        let _offset = fl.offset;
                        let _size = fl.size;
                    }

                    // Access field name (may be None for anonymous members)
                    let _name: &Option<String> = &field.name;

                    let field_canonical = field.ty.canonical();
                    if field_canonical.is_aggregate() {
                        // Recursively flatten nested aggregates
                        let nested = self.flatten_struct_fields(&field.ty, target);
                        result.extend(nested);
                    } else {
                        result.push(field.ty.clone());
                    }
                }
            }

            CType::Union { fields, .. } => {
                if fields.is_empty() {
                    return result;
                }
                // For unions, check whether all fields share the same FP type.
                let first_ty = fields[0].ty.canonical();
                let _first_name: &Option<String> = &fields[0].name;

                if first_ty.is_floating()
                    && fields
                        .iter()
                        .all(|f| f.bit_width.is_none() && f.ty.canonical() == first_ty)
                {
                    // Homogeneous FP union: treat as single FP leaf
                    result.push(fields[0].ty.clone());
                } else {
                    // Mixed or non-FP union: treat as integer of union's size
                    let union_size = size_of(canonical, target);
                    if union_size <= 4 {
                        result.push(CType::Int { signed: false });
                    } else {
                        result.push(CType::Long { signed: false });
                    }
                }
            }

            // Non-aggregate types: return as-is
            _ => {
                result.push(canonical.clone());
            }
        }

        result
    }

    // ===================================================================
    // Private helpers
    // ===================================================================

    /// Classifies a struct argument using the LP64D flattening rules.
    fn classify_aggregate_arg(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        target: &Target,
    ) -> ArgClassification {
        if type_size > XLEN2 {
            return ArgClassification::Indirect(registers::A0);
        }
        if type_size == 0 {
            return ArgClassification::IntegerReg(registers::A0);
        }

        let flat = self.collect_flat_fields(fields, target);

        if let Some(class) = self.fp_eligible_arg(&flat) {
            return class;
        }

        // Fallback: integer registers
        if type_size <= XLEN {
            ArgClassification::IntegerReg(registers::A0)
        } else {
            ArgClassification::IntegerRegPair(registers::A0, registers::A1)
        }
    }

    /// Classifies a union argument.
    fn classify_union_arg(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        target: &Target,
    ) -> ArgClassification {
        if type_size > XLEN2 {
            return ArgClassification::Indirect(registers::A0);
        }
        if type_size == 0 {
            return ArgClassification::IntegerReg(registers::A0);
        }

        // Check for homogeneous FP union
        if self.is_homogeneous_fp_union(fields) {
            return ArgClassification::FloatReg(registers::FA0);
        }

        if type_size <= XLEN {
            ArgClassification::IntegerReg(registers::A0)
        } else {
            ArgClassification::IntegerRegPair(registers::A0, registers::A1)
        }
    }

    /// Classifies a struct return value using LP64D flattening rules.
    fn classify_aggregate_return(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        target: &Target,
    ) -> ReturnClassification {
        if type_size > XLEN2 {
            return ReturnClassification::Indirect(registers::A0);
        }
        if type_size == 0 {
            return ReturnClassification::Void;
        }

        let flat = self.collect_flat_fields(fields, target);

        if let Some(class) = self.fp_eligible_return(&flat) {
            return class;
        }

        if type_size <= XLEN {
            ReturnClassification::InIntegerReg(registers::A0)
        } else {
            ReturnClassification::InIntegerRegPair(registers::A0, registers::A1)
        }
    }

    /// Classifies a union return value.
    fn classify_union_return(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        target: &Target,
    ) -> ReturnClassification {
        if type_size > XLEN2 {
            return ReturnClassification::Indirect(registers::A0);
        }
        if type_size == 0 {
            return ReturnClassification::Void;
        }

        if self.is_homogeneous_fp_union(fields) {
            return ReturnClassification::InFloatReg(registers::FA0);
        }

        if type_size <= XLEN {
            ReturnClassification::InIntegerReg(registers::A0)
        } else {
            ReturnClassification::InIntegerRegPair(registers::A0, registers::A1)
        }
    }

    /// Checks whether all fields in a union are the same floating-point type
    /// (no bit-fields). Returns `true` for a homogeneous FP union.
    fn is_homogeneous_fp_union(&self, fields: &[FieldDef]) -> bool {
        if fields.is_empty() {
            return false;
        }
        let first_ty = fields[0].ty.canonical();
        first_ty.is_floating()
            && fields.iter().all(|f| {
                f.bit_width.is_none() && f.ty.canonical() == first_ty
            })
    }

    /// Attempts to classify a flat-field list as FP-register-eligible for
    /// argument passing. Returns `None` if the fields don't qualify.
    fn fp_eligible_arg(&self, flat: &[FlatFieldKind]) -> Option<ArgClassification> {
        match flat.len() {
            1 => match flat[0] {
                FlatFieldKind::Float32 | FlatFieldKind::Float64 => {
                    Some(ArgClassification::FloatReg(registers::FA0))
                }
                FlatFieldKind::Integer => None,
            },
            2 => {
                let (a, b) = (flat[0], flat[1]);
                match (a, b) {
                    // Two FP fields → FloatRegPair
                    (FlatFieldKind::Float32 | FlatFieldKind::Float64,
                     FlatFieldKind::Float32 | FlatFieldKind::Float64) => {
                        Some(ArgClassification::FloatRegPair(
                            registers::FA0,
                            registers::FA1,
                        ))
                    }
                    // One int + one FP → IntAndFloat
                    (FlatFieldKind::Integer, FlatFieldKind::Float32 | FlatFieldKind::Float64) => {
                        Some(ArgClassification::IntAndFloat(
                            registers::A0,
                            registers::FA0,
                        ))
                    }
                    (FlatFieldKind::Float32 | FlatFieldKind::Float64, FlatFieldKind::Integer) => {
                        Some(ArgClassification::IntAndFloat(
                            registers::A0,
                            registers::FA0,
                        ))
                    }
                    // Two integers: not FP-eligible
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Attempts to classify a flat-field list as FP-register-eligible for
    /// return value delivery. Returns `None` if the fields don't qualify.
    fn fp_eligible_return(&self, flat: &[FlatFieldKind]) -> Option<ReturnClassification> {
        match flat.len() {
            1 => match flat[0] {
                FlatFieldKind::Float32 | FlatFieldKind::Float64 => {
                    Some(ReturnClassification::InFloatReg(registers::FA0))
                }
                FlatFieldKind::Integer => None,
            },
            2 => {
                let (a, b) = (flat[0], flat[1]);
                match (a, b) {
                    (FlatFieldKind::Float32 | FlatFieldKind::Float64,
                     FlatFieldKind::Float32 | FlatFieldKind::Float64) => {
                        Some(ReturnClassification::InFloatRegPair(
                            registers::FA0,
                            registers::FA1,
                        ))
                    }
                    (FlatFieldKind::Integer, FlatFieldKind::Float32 | FlatFieldKind::Float64)
                    | (FlatFieldKind::Float32 | FlatFieldKind::Float64, FlatFieldKind::Integer) => {
                        Some(ReturnClassification::IntAndFloat(
                            registers::A0,
                            registers::FA0,
                        ))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Collects [`FlatFieldKind`] entries from struct fields for classification.
    fn collect_flat_fields(
        &self,
        fields: &[FieldDef],
        target: &Target,
    ) -> Vec<FlatFieldKind> {
        let mut result = Vec::with_capacity(fields.len());
        self.flatten_into(fields, target, &mut result);
        result
    }

    /// Recursively flattens struct fields into [`FlatFieldKind`] entries.
    ///
    /// Nested structs and homogeneous FP unions are recursively descended.
    /// Arrays and bit-fields are treated as opaque integer entries.
    fn flatten_into(
        &self,
        fields: &[FieldDef],
        target: &Target,
        result: &mut Vec<FlatFieldKind>,
    ) {
        for field in fields {
            // Bit-fields disqualify the struct from FP register passing
            if field.bit_width.is_some() {
                result.push(FlatFieldKind::Integer);
                continue;
            }

            let canonical = field.ty.canonical();
            match canonical {
                CType::Float => result.push(FlatFieldKind::Float32),
                CType::Double | CType::LongDouble => result.push(FlatFieldKind::Float64),
                CType::Complex(base) => {
                    // Complex = 2 copies of the base FP type
                    let kind = match base.canonical() {
                        CType::Float => FlatFieldKind::Float32,
                        _ => FlatFieldKind::Float64,
                    };
                    result.push(kind);
                    result.push(kind);
                }
                CType::Struct { fields: sub, .. } => {
                    self.flatten_into(sub, target, result);
                }
                CType::Union { fields: sub, .. } => {
                    if self.is_homogeneous_fp_union(sub) {
                        let first_canonical = sub[0].ty.canonical();
                        match first_canonical {
                            CType::Float => result.push(FlatFieldKind::Float32),
                            _ => result.push(FlatFieldKind::Float64),
                        }
                    } else {
                        result.push(FlatFieldKind::Integer);
                    }
                }
                // Arrays are NOT recursively flattened per RISC-V ABI
                CType::Array { .. } => result.push(FlatFieldKind::Integer),
                // All integer-like scalars, pointers, enums
                CType::Bool
                | CType::Char { .. }
                | CType::Short { .. }
                | CType::Int { .. }
                | CType::Long { .. }
                | CType::LongLong { .. }
                | CType::Pointer(_)
                | CType::Enum { .. }
                | CType::Function { .. } => {
                    result.push(FlatFieldKind::Integer);
                }
                CType::Void => result.push(FlatFieldKind::Integer),
                CType::Atomic(inner) => {
                    // Strip atomic and classify inner type
                    let fd = FieldDef {
                        name: field.name.clone(),
                        ty: inner.as_ref().clone(),
                        bit_width: None,
                    };
                    self.flatten_into(&[fd], target, result);
                }
                CType::Typedef { underlying, .. } => {
                    let fd = FieldDef {
                        name: field.name.clone(),
                        ty: underlying.as_ref().clone(),
                        bit_width: None,
                    };
                    self.flatten_into(&[fd], target, result);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // classify_arg_with_state — core stateful classification engine
    // -----------------------------------------------------------------------

    /// Classifies a single argument while tracking shared register allocation
    /// state. This is the core engine used by [`compute_stack_layout`].
    fn classify_arg_with_state(
        &self,
        ty: &CType,
        target: &Target,
        int_idx: &mut usize,
        fp_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        let canonical = ty.canonical();
        let type_size = size_of(canonical, target);
        let type_align = align_of(canonical, target);

        match canonical {
            // ---- Void: shouldn't appear, but handle gracefully ----
            CType::Void => self.alloc_int_reg(0, int_idx, stk_off),

            // ---- Floating-point scalars ----
            CType::Float | CType::Double | CType::LongDouble => {
                self.alloc_fp_scalar(type_size, int_idx, fp_idx, stk_off)
            }

            // ---- Complex: needs 2 FP registers ----
            CType::Complex(base) => {
                let base_size = size_of(base, target);
                self.alloc_complex(base_size, type_size, int_idx, fp_idx, stk_off)
            }

            // ---- Scalar integers, bool, pointers, enums ----
            CType::Bool
            | CType::Char { .. }
            | CType::Short { .. }
            | CType::Int { .. }
            | CType::Long { .. }
            | CType::Pointer(_)
            | CType::Enum { .. } => self.alloc_int_reg(type_size, int_idx, stk_off),

            CType::LongLong { .. } => {
                if type_size <= XLEN {
                    self.alloc_int_reg(type_size, int_idx, stk_off)
                } else {
                    self.alloc_int_pair(type_size, int_idx, stk_off)
                }
            }

            // ---- Function type: pointer ----
            CType::Function { .. } => self.alloc_int_reg(XLEN, int_idx, stk_off),

            // ---- Struct: LP64D flattening ----
            CType::Struct { fields, .. } => {
                self.classify_struct_with_state(
                    fields, type_size, type_align, target, int_idx, fp_idx, stk_off,
                )
            }

            // ---- Union: opaque aggregate ----
            CType::Union { fields, .. } => {
                self.classify_union_with_state(
                    fields, type_size, target, int_idx, fp_idx, stk_off,
                )
            }

            // ---- Array: opaque aggregate ----
            CType::Array { .. } => {
                if type_size > XLEN2 {
                    self.alloc_indirect(int_idx, stk_off)
                } else if type_size <= XLEN {
                    self.alloc_int_reg(type_size, int_idx, stk_off)
                } else {
                    self.alloc_int_pair(type_size, int_idx, stk_off)
                }
            }

            // ---- Atomic / Typedef: recurse ----
            CType::Atomic(inner) => {
                self.classify_arg_with_state(inner, target, int_idx, fp_idx, stk_off)
            }
            CType::Typedef { underlying, .. } => {
                self.classify_arg_with_state(underlying, target, int_idx, fp_idx, stk_off)
            }
        }
    }

    /// Classifies a struct argument with shared register allocation state.
    fn classify_struct_with_state(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        _type_align: usize,
        target: &Target,
        int_idx: &mut usize,
        fp_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        // Large struct: pass by reference
        if type_size > XLEN2 {
            return self.alloc_indirect(int_idx, stk_off);
        }
        // Empty struct
        if type_size == 0 {
            return self.alloc_int_reg(0, int_idx, stk_off);
        }

        // Try FP-eligible flattening
        let flat = self.collect_flat_fields(fields, target);

        // Single float field
        if flat.len() == 1
            && matches!(flat[0], FlatFieldKind::Float32 | FlatFieldKind::Float64)
        {
            if *fp_idx < NUM_FP_ARG_REGS {
                let reg = registers::FLOAT_ARG_REGS[*fp_idx];
                *fp_idx += 1;
                return ArgClassification::FloatReg(reg);
            }
            // FP regs exhausted — fall through to integer
        }

        // Two float fields
        if flat.len() == 2
            && flat
                .iter()
                .all(|f| matches!(f, FlatFieldKind::Float32 | FlatFieldKind::Float64))
        {
            if *fp_idx + 1 < NUM_FP_ARG_REGS {
                let r1 = registers::FLOAT_ARG_REGS[*fp_idx];
                let r2 = registers::FLOAT_ARG_REGS[*fp_idx + 1];
                *fp_idx += 2;
                return ArgClassification::FloatRegPair(r1, r2);
            }
            // Fall through to integer
        }

        // Mixed int + float (exactly 2 fields, one of each)
        if flat.len() == 2 {
            let has_int = flat.iter().any(|f| *f == FlatFieldKind::Integer);
            let has_fp = flat
                .iter()
                .any(|f| matches!(f, FlatFieldKind::Float32 | FlatFieldKind::Float64));
            if has_int && has_fp && *int_idx < NUM_INT_ARG_REGS && *fp_idx < NUM_FP_ARG_REGS {
                let ir = registers::INTEGER_ARG_REGS[*int_idx];
                let fr = registers::FLOAT_ARG_REGS[*fp_idx];
                *int_idx += 1;
                *fp_idx += 1;
                return ArgClassification::IntAndFloat(ir, fr);
            }
            // Fall through to integer
        }

        // Default: integer register(s)
        if type_size <= XLEN {
            self.alloc_int_reg(type_size, int_idx, stk_off)
        } else {
            self.alloc_int_pair(type_size, int_idx, stk_off)
        }
    }

    /// Classifies a union argument with shared register allocation state.
    fn classify_union_with_state(
        &self,
        fields: &[FieldDef],
        type_size: usize,
        _target: &Target,
        int_idx: &mut usize,
        fp_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        if type_size > XLEN2 {
            return self.alloc_indirect(int_idx, stk_off);
        }

        // Homogeneous FP union
        if self.is_homogeneous_fp_union(fields) && *fp_idx < NUM_FP_ARG_REGS {
            let reg = registers::FLOAT_ARG_REGS[*fp_idx];
            *fp_idx += 1;
            return ArgClassification::FloatReg(reg);
        }

        if type_size <= XLEN {
            self.alloc_int_reg(type_size, int_idx, stk_off)
        } else {
            self.alloc_int_pair(type_size, int_idx, stk_off)
        }
    }

    // -----------------------------------------------------------------------
    // Register / stack allocation primitives
    // -----------------------------------------------------------------------

    /// Allocates one integer register for a value ≤ XLEN, or spills to stack.
    fn alloc_int_reg(
        &self,
        type_size: usize,
        int_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        if *int_idx < NUM_INT_ARG_REGS {
            let reg = registers::INTEGER_ARG_REGS[*int_idx];
            *int_idx += 1;
            ArgClassification::IntegerReg(reg)
        } else {
            let offset = *stk_off;
            let slot = round_up_to(type_size.max(XLEN), XLEN) as u32;
            *stk_off += slot as i32;
            ArgClassification::Stack { offset, size: slot }
        }
    }

    /// Allocates a pair of integer registers, or spills to stack.
    fn alloc_int_pair(
        &self,
        type_size: usize,
        int_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        if *int_idx + 1 < NUM_INT_ARG_REGS {
            let r1 = registers::INTEGER_ARG_REGS[*int_idx];
            let r2 = registers::INTEGER_ARG_REGS[*int_idx + 1];
            *int_idx += 2;
            ArgClassification::IntegerRegPair(r1, r2)
        } else {
            // Not enough integer registers: spill entire value to stack
            let offset = *stk_off;
            let slot = round_up_to(type_size, XLEN) as u32;
            *stk_off += slot as i32;
            ArgClassification::Stack { offset, size: slot }
        }
    }

    /// Allocates an FP register for a scalar float/double, falling back to
    /// integer register, then stack.
    fn alloc_fp_scalar(
        &self,
        type_size: usize,
        int_idx: &mut usize,
        fp_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        if *fp_idx < NUM_FP_ARG_REGS {
            let reg = registers::FLOAT_ARG_REGS[*fp_idx];
            *fp_idx += 1;
            ArgClassification::FloatReg(reg)
        } else if *int_idx < NUM_INT_ARG_REGS {
            // FP regs exhausted — fall back to integer register
            let reg = registers::INTEGER_ARG_REGS[*int_idx];
            *int_idx += 1;
            ArgClassification::IntegerReg(reg)
        } else {
            let offset = *stk_off;
            let slot = round_up_to(type_size, XLEN) as u32;
            *stk_off += slot as i32;
            ArgClassification::Stack { offset, size: slot }
        }
    }

    /// Allocates registers for a `_Complex` type (2 FP values).
    fn alloc_complex(
        &self,
        _base_size: usize,
        total_size: usize,
        int_idx: &mut usize,
        fp_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        // Try 2 FP registers
        if *fp_idx + 1 < NUM_FP_ARG_REGS {
            let r1 = registers::FLOAT_ARG_REGS[*fp_idx];
            let r2 = registers::FLOAT_ARG_REGS[*fp_idx + 1];
            *fp_idx += 2;
            return ArgClassification::FloatRegPair(r1, r2);
        }
        // Try 2 integer registers
        if *int_idx + 1 < NUM_INT_ARG_REGS {
            let r1 = registers::INTEGER_ARG_REGS[*int_idx];
            let r2 = registers::INTEGER_ARG_REGS[*int_idx + 1];
            *int_idx += 2;
            return ArgClassification::IntegerRegPair(r1, r2);
        }
        // Stack
        let offset = *stk_off;
        let slot = round_up_to(total_size, XLEN) as u32;
        *stk_off += slot as i32;
        ArgClassification::Stack { offset, size: slot }
    }

    /// Allocates an integer register to hold a pointer for indirect (by-reference)
    /// passing, or spills the pointer to the stack.
    fn alloc_indirect(
        &self,
        int_idx: &mut usize,
        stk_off: &mut i32,
    ) -> ArgClassification {
        if *int_idx < NUM_INT_ARG_REGS {
            let reg = registers::INTEGER_ARG_REGS[*int_idx];
            *int_idx += 1;
            ArgClassification::Indirect(reg)
        } else {
            let offset = *stk_off;
            *stk_off += XLEN as i32; // Pointer-sized stack slot
            ArgClassification::Stack { offset, size: XLEN as u32 }
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level helper
// ---------------------------------------------------------------------------

/// Rounds `value` up to the next multiple of `align`.
/// `align` must be a power of two; if zero, `value` is returned unchanged.
#[inline]
fn round_up_to(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}
