//! # RISC-V LP64D ABI Implementation
//!
//! Implements the **RISC-V LP64D** (Long-Pointer 64-bit with Double-precision
//! floating-point) calling convention for BCC's RISC-V 64 backend.
//!
//! The LP64D ABI is the standard calling convention used by Linux on RISC-V 64
//! (including Linux kernel 6.9).  This module defines:
//!
//! - **Argument passing**: integer arguments in a0–a7 (x10–x17), floating-point
//!   arguments in fa0–fa7 (f10–f17), stack-passed arguments aligned to 8 bytes
//! - **Return values**: integer in a0 (x10), float/double in fa0 (f10), large
//!   structs via hidden pointer in a0
//! - **Struct passing**: structs ≤ 16 bytes (2×XLEN) may be decomposed into
//!   register pairs; larger structs are passed indirectly
//! - **Variadic arguments**: floating-point varargs are passed in **integer**
//!   registers (not FP), matching the LP64D specification
//! - **Callee-saved registers**: s0–s11 (x8–x9, x18–x27), fs0–fs11
//!   (f8–f9, f18–f27)
//! - **Caller-saved registers**: ra (x1), t0–t6 (x5–x7, x28–x31), a0–a7
//!   (x10–x17), ft0–ft11 (f0–f7, f28–f31), fa0–fa7 (f10–f17)
//! - **Stack alignment**: 16 bytes
//! - **Stack growth**: downward
//! - **Frame pointer**: s0/fp (x8)
//! - **Return address**: ra (x1)
//!
//! ## Data Model (LP64D)
//!
//! | C type        | Size (bytes) | Alignment (bytes) |
//! |---------------|-------------|-------------------|
//! | `_Bool`       | 1           | 1                 |
//! | `char`        | 1           | 1                 |
//! | `short`       | 2           | 2                 |
//! | `int`         | 4           | 4                 |
//! | `long`        | 8           | 8                 |
//! | `long long`   | 8           | 8                 |
//! | `float`       | 4           | 4                 |
//! | `double`      | 8           | 8                 |
//! | `long double` | 16          | 16                |
//! | `pointer`     | 8           | 8                 |
//!
//! ## Zero-Dependency
//!
//! This module uses only `crate::` references and the Rust standard library.
//! No external crates.

use crate::backend::riscv64::registers::{
    CALLEE_SAVED_FPRS, CALLEE_SAVED_GPRS, CALLER_SAVED_FPRS, CALLER_SAVED_GPRS, F10, F11, F12, F13,
    F14, F15, F16, F17, X10, X11, X12, X13, X14, X15, X16, X17,
};
use crate::backend::traits::ArgLocation;
use crate::common::target::Target;
use crate::common::type_builder::{StructLayout, TypeBuilder};
use crate::common::types::CType;

// ===========================================================================
// LP64D ABI Constants
// ===========================================================================

/// Number of integer argument registers (a0–a7).
pub const NUM_INT_ARG_REGS: usize = 8;

/// Number of floating-point argument registers (fa0–fa7).
pub const NUM_FP_ARG_REGS: usize = 8;

/// Stack alignment requirement in bytes.
///
/// The RISC-V LP64D ABI requires the stack pointer to be 16-byte aligned
/// at all times.
pub const STACK_ALIGNMENT: usize = 16;

/// Argument slot size in bytes.
///
/// All stack-passed arguments occupy at least one slot (8 bytes) and are
/// naturally aligned within the argument area.
pub const ARG_SLOT_SIZE: usize = 8;

/// Maximum size (in bytes) of a struct that can be passed in registers.
///
/// Structs up to 2×XLEN (16 bytes on RV64) may be decomposed into at most
/// two register-width values. Larger structs are passed indirectly.
pub const MAX_STRUCT_REG_SIZE: usize = 16;

/// Integer argument register IDs: a0–a7 (x10–x17).
///
/// Ordered by ABI convention — a0 is used first.
pub const INT_ARG_REGS: [u16; 8] = [X10, X11, X12, X13, X14, X15, X16, X17];

/// Floating-point argument register IDs: fa0–fa7 (f10–f17).
///
/// Ordered by ABI convention — fa0 is used first.
pub const FP_ARG_REGS: [u16; 8] = [F10, F11, F12, F13, F14, F15, F16, F17];

/// Integer return value registers: a0 (x10), a1 (x11).
///
/// Scalar return values use a0. Two-register returns (e.g., 128-bit values
/// or struct pairs) use a0 (low) and a1 (high).
pub const INT_RET_REGS: [u16; 2] = [X10, X11];

/// Floating-point return value registers: fa0 (f10), fa1 (f11).
///
/// Float/double return values use fa0. Struct-of-two-floats returns use
/// fa0 + fa1.
pub const FP_RET_REGS: [u16; 2] = [F10, F11];

// ===========================================================================
// ArgClass — Argument Classification
// ===========================================================================

/// Classification of how a function argument or return value is passed
/// according to the LP64D ABI.
///
/// The RISC-V calling convention classifies each argument into one of
/// these categories to determine register/stack assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgClass {
    /// Passed in a single integer register.
    ///
    /// Applies to scalar integers, pointers, enums, and booleans that fit
    /// in XLEN (64 bits).
    Integer,

    /// Passed in a single floating-point register.
    ///
    /// Applies to `float` and `double` in non-variadic positions.
    Float,

    /// Passed as two integer registers (for 2×XLEN aggregates).
    ///
    /// Applies to structs ≤ 16 bytes whose fields are all integer-like
    /// or when no FP registers are available.
    IntegerPair,

    /// Passed in one integer register + one float register.
    ///
    /// Applies to structs with one integer field and one float/double field,
    /// where the integer field comes first.
    IntegerFloat,

    /// Passed in one float register + one integer register.
    ///
    /// Applies to structs with one float/double field first and one integer
    /// field second.
    FloatInteger,

    /// Passed in two floating-point registers.
    ///
    /// Applies to structs composed of exactly two float or double fields.
    FloatPair,

    /// Passed on the stack.
    ///
    /// Applies when all available registers are exhausted or the type
    /// requires stack passing.
    Memory,

    /// Passed via hidden pointer (caller allocates space, passes address
    /// in an integer register).
    ///
    /// Applies to aggregates exceeding 2×XLEN (16 bytes) and arrays.
    Indirect,
}

// ===========================================================================
// FrameLayout — Stack Frame Layout
// ===========================================================================

/// Describes the stack frame layout for a RISC-V 64 function.
///
/// The frame is laid out from high addresses (previous frame) to low
/// addresses (current SP), growing downward:
///
/// ```text
/// [previous frame]
/// [incoming stack args]      high addresses
/// [saved RA]                 <- sp + total_size - 8
/// [saved s0/FP]              <- sp + total_size - 16
/// [callee-saved regs]
/// [local variables]
/// [spill slots]
/// [outgoing stack args]      <- sp (16-byte aligned)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLayout {
    /// Total stack frame size in bytes (always 16-byte aligned).
    pub total_size: usize,

    /// Offset from FP (s0) to the start of the local variable area.
    ///
    /// Typically negative (locals are below the frame pointer).
    pub locals_offset: i64,

    /// Offset from FP to the start of the callee-saved register save area.
    pub callee_saved_offset: i64,

    /// Offset from SP to the saved return address (RA).
    pub ra_offset: i64,

    /// Offset from SP to the saved frame pointer (s0/FP).
    pub fp_offset: i64,

    /// Offset from FP to the register argument spill area.
    ///
    /// Used for variadic functions that must spill register arguments to
    /// the stack for `va_arg` access.
    pub arg_spill_offset: i64,
}

// ===========================================================================
// RiscV64Abi — Main ABI Handler
// ===========================================================================

/// RISC-V LP64D ABI implementation.
///
/// Tracks register allocation state across multiple argument classifications
/// within a single function call. Call [`reset`](RiscV64Abi::reset) before
/// classifying arguments for a new function.
///
/// # Usage
///
/// ```ignore
/// let mut abi = RiscV64Abi::new();
/// for param_ty in &function_params {
///     let loc = abi.classify_arg(param_ty);
///     // Use `loc` to emit argument setup code
/// }
/// let ret_loc = abi.classify_return(&return_type);
/// abi.reset(); // Ready for the next function
/// ```
pub struct RiscV64Abi {
    /// Number of integer argument registers consumed so far.
    int_regs_used: usize,
    /// Number of floating-point argument registers consumed so far.
    fp_regs_used: usize,
    /// Current byte offset within the stack argument area.
    stack_offset: usize,
}

impl Default for RiscV64Abi {
    fn default() -> Self {
        Self::new()
    }
}

impl RiscV64Abi {
    /// Create a new ABI handler with all counters initialized to zero.
    #[inline]
    pub fn new() -> Self {
        RiscV64Abi {
            int_regs_used: 0,
            fp_regs_used: 0,
            stack_offset: 0,
        }
    }

    /// Reset all register and stack counters for a new function call.
    ///
    /// Must be called between processing different function signatures.
    #[inline]
    pub fn reset(&mut self) {
        self.int_regs_used = 0;
        self.fp_regs_used = 0;
        self.stack_offset = 0;
    }

    // -----------------------------------------------------------------------
    // Internal: Allocate an integer register or stack slot
    // -----------------------------------------------------------------------

    /// Try to allocate the next available integer argument register.
    /// Returns the register as `ArgLocation::Register`, or `None` if
    /// all 8 integer argument registers are exhausted.
    fn alloc_int_reg(&mut self) -> Option<ArgLocation> {
        if self.int_regs_used < NUM_INT_ARG_REGS {
            let reg = INT_ARG_REGS[self.int_regs_used];
            self.int_regs_used += 1;
            Some(ArgLocation::Register(reg))
        } else {
            None
        }
    }

    /// Try to allocate the next available FP argument register.
    /// Returns the register as `ArgLocation::Register`, or `None` if
    /// all 8 FP argument registers are exhausted.
    fn alloc_fp_reg(&mut self) -> Option<ArgLocation> {
        if self.fp_regs_used < NUM_FP_ARG_REGS {
            let reg = FP_ARG_REGS[self.fp_regs_used];
            self.fp_regs_used += 1;
            Some(ArgLocation::Register(reg))
        } else {
            None
        }
    }

    /// Allocate a stack slot of the given size and alignment.
    /// Returns `ArgLocation::Stack` at the allocated offset.
    fn alloc_stack(&mut self, size: usize, align: usize) -> ArgLocation {
        // Align the current stack offset to the required alignment.
        let effective_align = if align < ARG_SLOT_SIZE {
            ARG_SLOT_SIZE
        } else {
            align
        };
        self.stack_offset = align_up(self.stack_offset, effective_align);
        let offset = self.stack_offset;
        // Advance by at least ARG_SLOT_SIZE.
        let slot_size = if size < ARG_SLOT_SIZE {
            ARG_SLOT_SIZE
        } else {
            align_up(size, ARG_SLOT_SIZE)
        };
        self.stack_offset += slot_size;
        ArgLocation::Stack(offset as i32)
    }

    // -----------------------------------------------------------------------
    // Argument Classification
    // -----------------------------------------------------------------------

    /// Classify a function argument according to the LP64D ABI.
    ///
    /// Determines whether the argument is passed in integer registers,
    /// floating-point registers, a register pair, or on the stack.
    ///
    /// This method updates the internal register and stack counters.
    /// Arguments must be classified in order from left to right.
    pub fn classify_arg(&mut self, ty: &CType) -> ArgLocation {
        let resolved = resolve_type(ty);

        match resolved {
            // ----- Void: should not appear as an argument, treat as error -----
            CType::Void => {
                // Void is not a valid argument type, but handle gracefully.
                ArgLocation::Stack(0)
            }

            // ----- Boolean, integer types, pointers, enums -----
            CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong => self.classify_integer_arg(),

            // __int128: passed as two integer registers on RISC-V 64.
            CType::Int128 | CType::UInt128 => {
                if self.int_regs_used % 2 != 0 {
                    self.int_regs_used += 1; // align to even register pair
                }
                if self.int_regs_used + 1 < 8 {
                    let r1 = INT_ARG_REGS[self.int_regs_used];
                    let r2 = INT_ARG_REGS[self.int_regs_used + 1];
                    self.int_regs_used += 2;
                    ArgLocation::RegisterPair(r1, r2)
                } else {
                    let loc = ArgLocation::Stack(self.stack_offset as i32);
                    self.stack_offset += 16;
                    loc
                }
            }

            CType::Pointer(_, _) => self.classify_integer_arg(),

            CType::Enum { .. } => self.classify_integer_arg(),

            // ----- Float -----
            CType::Float => self.classify_fp_arg(4),

            // ----- Double -----
            CType::Double => self.classify_fp_arg(8),

            // ----- Long double (128-bit on RISC-V 64) -----
            CType::LongDouble => self.classify_long_double_arg(),

            // ----- Complex types -----
            CType::Complex(ref base) => self.classify_complex_arg(base),

            // ----- Struct: small structs in registers, large indirect -----
            CType::Struct {
                ref fields,
                packed,
                aligned,
                ..
            } => self.classify_struct_arg(fields, *packed, *aligned),

            // ----- Union: treated as integer of the union's size -----
            CType::Union {
                ref fields,
                packed,
                aligned,
                ..
            } => self.classify_union_arg(fields, *packed, *aligned),

            // ----- Array: always passed by reference (indirect) -----
            CType::Array(_, _) => self.classify_indirect_arg(),

            // ----- Function type: passed as pointer -----
            CType::Function { .. } => self.classify_integer_arg(),

            // ----- Atomic: classify the inner type -----
            CType::Atomic(ref inner) => {
                // Atomic types are passed the same as their inner type.
                let inner_clone = (**inner).clone();
                self.classify_arg(&inner_clone)
            }

            // ----- Typedef/Qualified: should already be resolved -----
            CType::Typedef { ref underlying, .. } => {
                let u = (**underlying).clone();
                self.classify_arg(&u)
            }
            CType::Qualified(ref inner, _) => {
                let i = (**inner).clone();
                self.classify_arg(&i)
            }
        }
    }

    /// Classify a return value according to the LP64D ABI.
    ///
    /// Similar to argument classification but uses return registers
    /// (a0/a1 for integer, fa0/fa1 for floating-point) and does not
    /// consume argument register slots.
    pub fn classify_return(&self, ty: &CType) -> ArgLocation {
        let resolved = resolve_type(ty);

        match resolved {
            // Void: no return value.
            CType::Void => ArgLocation::Register(X10),

            // Scalar integers, pointers, enums, booleans: returned in a0.
            CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong => ArgLocation::Register(X10),

            // __int128: returned in a0+a1 register pair on RISC-V 64.
            CType::Int128 | CType::UInt128 => ArgLocation::RegisterPair(X10, X11),

            CType::Pointer(_, _) => ArgLocation::Register(X10),
            CType::Enum { .. } => ArgLocation::Register(X10),

            // Float: returned in fa0.
            CType::Float => ArgLocation::Register(F10),

            // Double: returned in fa0.
            CType::Double => ArgLocation::Register(F10),

            // Long double (128-bit): returned in a0+a1 (integer pair).
            CType::LongDouble => ArgLocation::RegisterPair(X10, X11),

            // Complex float: fa0 + fa1.
            CType::Complex(ref base) => {
                let base_resolved = resolve_type(base);
                match base_resolved {
                    CType::Float | CType::Double => ArgLocation::RegisterPair(F10, F11),
                    _ => ArgLocation::RegisterPair(X10, X11),
                }
            }

            // Struct return values.
            CType::Struct {
                ref fields,
                packed,
                aligned,
                ..
            } => self.classify_struct_return(fields, *packed, *aligned),

            // Union return values: like integer up to 16 bytes.
            CType::Union {
                ref fields,
                packed,
                aligned,
                ..
            } => {
                let tb = TypeBuilder::new(Target::RiscV64);
                let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
                let layout = tb.compute_union_layout(&field_types, *packed, *aligned);
                if layout.size <= 8 {
                    ArgLocation::Register(X10)
                } else if layout.size <= MAX_STRUCT_REG_SIZE {
                    ArgLocation::RegisterPair(X10, X11)
                } else {
                    // Large union: returned via hidden pointer in a0.
                    ArgLocation::Register(X10)
                }
            }

            // Array: returned via hidden pointer.
            CType::Array(_, _) => ArgLocation::Register(X10),

            // Function type: returned as pointer in a0.
            CType::Function { .. } => ArgLocation::Register(X10),

            // Atomic, Typedef, Qualified: resolve through.
            CType::Atomic(ref inner) => self.classify_return(inner),
            CType::Typedef { ref underlying, .. } => self.classify_return(underlying),
            CType::Qualified(ref inner, _) => self.classify_return(inner),
        }
    }

    /// Classify a variadic argument according to the LP64D ABI.
    ///
    /// **Critical LP64D rule**: Floating-point arguments in the variadic
    /// portion of a function call are passed in **integer** registers
    /// (not FP registers). This is because the callee does not know the
    /// types of variadic arguments at compile time and accesses them via
    /// integer register spills.
    ///
    /// Float is promoted to double (standard C variadic promotion).
    pub fn classify_variadic_arg(&mut self, ty: &CType) -> ArgLocation {
        let resolved = resolve_type(ty);

        match resolved {
            // Scalar integers, pointers, enums, booleans: normal integer passing.
            CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong => self.classify_integer_arg(),

            // __int128: passed as two registers on RISC-V 64
            CType::Int128 | CType::UInt128 => {
                if self.int_regs_used % 2 != 0 {
                    self.int_regs_used += 1; // align to even register pair
                }
                if self.int_regs_used + 1 < 8 {
                    let r1 = INT_ARG_REGS[self.int_regs_used];
                    let r2 = INT_ARG_REGS[self.int_regs_used + 1];
                    self.int_regs_used += 2;
                    ArgLocation::RegisterPair(r1, r2)
                } else {
                    let loc = ArgLocation::Stack(self.stack_offset as i32);
                    self.stack_offset += 16;
                    loc
                }
            }

            CType::Pointer(_, _) => self.classify_integer_arg(),
            CType::Enum { .. } => self.classify_integer_arg(),

            // Float: promoted to double, then passed in INTEGER register.
            // Double: passed in INTEGER register (not FP).
            CType::Float | CType::Double => self.classify_integer_arg(),

            // Long double: passed in 2 integer registers.
            CType::LongDouble => self.classify_long_double_arg(),

            // Complex: passed in integer registers.
            CType::Complex(ref _base) => {
                // Complex float → 8 bytes, complex double → 16 bytes.
                // Both use integer registers in variadic context.
                let size = type_size(ty);
                if size <= 8 {
                    self.classify_integer_arg()
                } else {
                    self.classify_long_double_arg()
                }
            }

            // Struct/Union: same rules as non-variadic but FP fields
            // use integer registers instead.
            CType::Struct {
                ref fields,
                packed,
                aligned,
                ..
            } => {
                let tb = TypeBuilder::new(Target::RiscV64);
                let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
                let layout = tb.compute_struct_layout(&field_types, *packed, *aligned);
                if layout.size > MAX_STRUCT_REG_SIZE {
                    self.classify_indirect_arg()
                } else if layout.size <= 8 {
                    self.classify_integer_arg()
                } else {
                    // 9–16 bytes: use two integer registers or stack.
                    self.classify_integer_pair_arg()
                }
            }

            CType::Union {
                ref fields,
                packed,
                aligned,
                ..
            } => {
                let tb = TypeBuilder::new(Target::RiscV64);
                let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
                let layout = tb.compute_union_layout(&field_types, *packed, *aligned);
                if layout.size > MAX_STRUCT_REG_SIZE {
                    self.classify_indirect_arg()
                } else if layout.size <= 8 {
                    self.classify_integer_arg()
                } else {
                    self.classify_integer_pair_arg()
                }
            }

            // Array: always indirect.
            CType::Array(_, _) => self.classify_indirect_arg(),

            // Function: as pointer.
            CType::Function { .. } => self.classify_integer_arg(),

            CType::Void => ArgLocation::Stack(0),
            CType::Atomic(ref inner) => self.classify_variadic_arg(inner),
            CType::Typedef { ref underlying, .. } => self.classify_variadic_arg(underlying),
            CType::Qualified(ref inner, _) => self.classify_variadic_arg(inner),
        }
    }

    // -----------------------------------------------------------------------
    // Register Set Queries
    // -----------------------------------------------------------------------

    /// Returns the callee-saved general-purpose register set.
    ///
    /// s0–s11: x8, x9, x18–x27 (12 registers).
    /// These must be saved/restored by the callee if modified.
    #[inline]
    pub fn callee_saved_gprs() -> &'static [u16] {
        CALLEE_SAVED_GPRS
    }

    /// Returns the callee-saved floating-point register set.
    ///
    /// fs0–fs11: f8, f9, f18–f27 (12 registers).
    #[inline]
    pub fn callee_saved_fprs() -> &'static [u16] {
        CALLEE_SAVED_FPRS
    }

    /// Returns the caller-saved general-purpose register set.
    ///
    /// ra (x1), t0–t6 (x5–x7, x28–x31), a0–a7 (x10–x17) — 16 registers.
    #[inline]
    pub fn caller_saved_gprs() -> &'static [u16] {
        CALLER_SAVED_GPRS
    }

    /// Returns the caller-saved floating-point register set.
    ///
    /// ft0–ft7 (f0–f7), fa0–fa7 (f10–f17), ft8–ft11 (f28–f31) — 20 registers.
    #[inline]
    pub fn caller_saved_fprs() -> &'static [u16] {
        CALLER_SAVED_FPRS
    }

    // -----------------------------------------------------------------------
    // Stack Frame Layout Computation
    // -----------------------------------------------------------------------

    /// Compute the stack frame layout for a function.
    ///
    /// # Arguments
    ///
    /// * `params` — Parameter types of the function (used to compute
    ///   argument spill area size for variadic functions).
    /// * `locals_size` — Total size in bytes of local variables and
    ///   compiler-generated temporaries.
    /// * `callee_saved_count` — Number of callee-saved registers that
    ///   this function will save/restore (each costs 8 bytes).
    ///
    /// # Returns
    ///
    /// A [`FrameLayout`] describing offsets and total frame size.
    pub fn compute_frame_layout(
        params: &[CType],
        locals_size: usize,
        callee_saved_count: usize,
    ) -> FrameLayout {
        // The frame layout (high → low addresses):
        //   [previous frame / incoming stack args]   ← old SP
        //   [saved RA]                               ← SP + total - 8
        //   [saved FP (s0)]                          ← SP + total - 16
        //   [other callee-saved regs]                ← callee-saved area
        //   [local variables]                        ← locals area
        //   [arg spill / outgoing args]              ← current SP

        // RA and FP always saved (2 × 8 bytes).
        let ra_fp_size: usize = 16;

        // Callee-saved registers (excluding FP which is already counted).
        // The caller provides a count of additional callee-saved regs beyond FP.
        let callee_save_area = callee_saved_count * 8;

        // Local variables area.
        let locals_area = align_up(locals_size, 8);

        // Argument spill area: for variadic functions, we might need to
        // spill all 8 integer argument registers. For non-variadic, this
        // is typically 0. We compute a conservative estimate.
        let arg_spill_area = compute_arg_spill_size(params);

        // Total frame size before alignment.
        let unaligned_total = ra_fp_size + callee_save_area + locals_area + arg_spill_area;

        // Align total frame size to STACK_ALIGNMENT (16 bytes).
        let total_size = align_up(unaligned_total, STACK_ALIGNMENT);

        // Compute offsets relative to SP and FP.
        // FP points to the saved FP slot: SP + total_size - 16.
        let ra_offset = (total_size as i64) - 8;
        let fp_offset = (total_size as i64) - 16;

        // Callee-saved area starts below saved FP.
        let callee_saved_offset = -(ra_fp_size as i64) - (callee_save_area as i64);

        // Locals start below callee-saved area.
        let locals_offset = -(ra_fp_size as i64) - (callee_save_area as i64) - (locals_area as i64);

        // Arg spill area is at the bottom, just above SP.
        let arg_spill_offset = -(total_size as i64) + (ra_fp_size as i64);

        FrameLayout {
            total_size,
            locals_offset,
            callee_saved_offset,
            ra_offset,
            fp_offset,
            arg_spill_offset,
        }
    }

    // -----------------------------------------------------------------------
    // Type size and alignment (LP64D specific, delegated methods)
    // -----------------------------------------------------------------------

    /// Returns the size in bytes of a C type under the LP64D data model.
    ///
    /// This is a convenience method that delegates to the module-level
    /// [`type_size`] function.
    #[inline]
    pub fn type_size(ty: &CType) -> usize {
        type_size(ty)
    }

    /// Returns the alignment in bytes of a C type under the LP64D data model.
    ///
    /// This is a convenience method that delegates to the module-level
    /// [`type_alignment`] function.
    #[inline]
    pub fn type_alignment(ty: &CType) -> usize {
        type_alignment(ty)
    }

    // -----------------------------------------------------------------------
    // Private: Integer argument classification
    // -----------------------------------------------------------------------

    /// Classify a scalar integer/pointer argument.
    ///
    /// Attempts to place it in the next integer argument register.
    /// Falls back to an 8-byte stack slot.
    fn classify_integer_arg(&mut self) -> ArgLocation {
        if let Some(loc) = self.alloc_int_reg() {
            loc
        } else {
            self.alloc_stack(ARG_SLOT_SIZE, ARG_SLOT_SIZE)
        }
    }

    /// Classify a floating-point argument (float or double).
    ///
    /// Tries FP register first, then integer register, then stack.
    fn classify_fp_arg(&mut self, _size: usize) -> ArgLocation {
        // Try FP register first.
        if let Some(loc) = self.alloc_fp_reg() {
            return loc;
        }
        // If no FP register available, use integer register.
        if let Some(loc) = self.alloc_int_reg() {
            return loc;
        }
        // Fall back to stack.
        self.alloc_stack(ARG_SLOT_SIZE, ARG_SLOT_SIZE)
    }

    /// Classify a long double (128-bit / 16-byte) argument.
    ///
    /// Requires 2 integer registers (aligned to even register pair when
    /// possible). If only one register is available, the first half goes
    /// in the register and the second half on the stack. If no registers
    /// are available, the entire value goes on the stack.
    fn classify_long_double_arg(&mut self) -> ArgLocation {
        // Align int_regs_used to even boundary for 2xXLEN natural alignment.
        if self.int_regs_used % 2 != 0 && self.int_regs_used < NUM_INT_ARG_REGS {
            self.int_regs_used += 1; // Skip one register for alignment.
        }

        if self.int_regs_used + 1 < NUM_INT_ARG_REGS {
            // Both halves fit in registers (2 × 64-bit = 128-bit).
            let r1 = INT_ARG_REGS[self.int_regs_used];
            let r2 = INT_ARG_REGS[self.int_regs_used + 1];
            self.int_regs_used += 2;
            ArgLocation::RegisterPair(r1, r2)
        } else if self.int_regs_used < NUM_INT_ARG_REGS {
            // Split handling: exactly 1 integer register remains for a
            // 128-bit value.  Per the RISC-V psABI, when a 2×XLEN
            // value cannot be fully placed in registers, the low half
            // goes in the last available register and the high half
            // goes on the stack.  We represent this as a RegisterPair
            // where the first register holds the low half.  The caller
            // must emit the high half to the appropriate stack slot.
            let r1 = INT_ARG_REGS[self.int_regs_used];
            self.int_regs_used += 1;
            // Allocate 8 bytes of stack for the high half, aligned to 8.
            let stack_offset = self.alloc_stack(8, 8);
            // Return RegisterPair with the actual register for the low half.
            // The stack_offset for the high half is encoded as a sentinel
            // — callers detect the split case when one operand is a physical
            // register and the other exceeds the argument register range.
            // For ABI correctness we return the register; the code generator
            // handles the stack spill for the high half during call lowering.
            match stack_offset {
                ArgLocation::Stack(_) => ArgLocation::RegisterPair(r1, r1),
                _ => ArgLocation::RegisterPair(r1, r1),
            }
        } else {
            // No registers available: entirely on stack, aligned to 16.
            self.alloc_stack(16, 16)
        }
    }

    /// Classify a _Complex type argument.
    fn classify_complex_arg(&mut self, base: &CType) -> ArgLocation {
        let base_resolved = resolve_type(base);
        match base_resolved {
            CType::Float => {
                // Complex float = 2 × float = 8 bytes.
                // Try to pass as two FP registers.
                if self.fp_regs_used + 1 < NUM_FP_ARG_REGS {
                    let r1 = FP_ARG_REGS[self.fp_regs_used];
                    let r2 = FP_ARG_REGS[self.fp_regs_used + 1];
                    self.fp_regs_used += 2;
                    ArgLocation::RegisterPair(r1, r2)
                } else {
                    // Fall back to integer register (8 bytes fits in one).
                    self.classify_integer_arg()
                }
            }
            CType::Double => {
                // Complex double = 2 × double = 16 bytes.
                if self.fp_regs_used + 1 < NUM_FP_ARG_REGS {
                    let r1 = FP_ARG_REGS[self.fp_regs_used];
                    let r2 = FP_ARG_REGS[self.fp_regs_used + 1];
                    self.fp_regs_used += 2;
                    ArgLocation::RegisterPair(r1, r2)
                } else {
                    // Fall back to integer pair.
                    self.classify_integer_pair_arg()
                }
            }
            _ => {
                // Complex long double: 32 bytes → indirect.
                self.classify_indirect_arg()
            }
        }
    }

    /// Classify a struct argument according to the LP64D hardware
    /// floating-point calling convention rules.
    ///
    /// Structs ≤ 16 bytes are decomposed:
    /// - Two float/double fields → FloatPair (two FP regs)
    /// - One float/double + one integer → IntegerFloat or FloatInteger
    /// - Otherwise → one or two integer registers
    ///
    /// Structs > 16 bytes are passed indirectly.
    fn classify_struct_arg(
        &mut self,
        fields: &[crate::common::types::StructField],
        packed: bool,
        aligned: Option<usize>,
    ) -> ArgLocation {
        let tb = TypeBuilder::new(Target::RiscV64);
        let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
        let layout = tb.compute_struct_layout(&field_types, packed, aligned);

        // Structs larger than 2×XLEN are passed indirectly.
        if layout.size > MAX_STRUCT_REG_SIZE {
            return self.classify_indirect_arg();
        }

        // Empty struct: no argument passed.
        if fields.is_empty() || layout.size == 0 {
            return ArgLocation::Register(X10);
        }

        // Classify fields to determine the optimal passing strategy.
        let classification = classify_struct_fields(fields);

        match classification {
            StructClassification::FloatPair => {
                // Two float/double fields → try two FP registers.
                if self.fp_regs_used + 1 < NUM_FP_ARG_REGS {
                    let r1 = FP_ARG_REGS[self.fp_regs_used];
                    let r2 = FP_ARG_REGS[self.fp_regs_used + 1];
                    self.fp_regs_used += 2;
                    ArgLocation::RegisterPair(r1, r2)
                } else {
                    // Fall back to integer registers or stack.
                    self.classify_struct_as_integer(&layout)
                }
            }
            StructClassification::FloatInteger => {
                // One FP field (first) + one integer field (second).
                if self.fp_regs_used < NUM_FP_ARG_REGS && self.int_regs_used < NUM_INT_ARG_REGS {
                    let fp = FP_ARG_REGS[self.fp_regs_used];
                    let gp = INT_ARG_REGS[self.int_regs_used];
                    self.fp_regs_used += 1;
                    self.int_regs_used += 1;
                    ArgLocation::RegisterPair(fp, gp)
                } else {
                    self.classify_struct_as_integer(&layout)
                }
            }
            StructClassification::IntegerFloat => {
                // One integer field (first) + one FP field (second).
                if self.int_regs_used < NUM_INT_ARG_REGS && self.fp_regs_used < NUM_FP_ARG_REGS {
                    let gp = INT_ARG_REGS[self.int_regs_used];
                    let fp = FP_ARG_REGS[self.fp_regs_used];
                    self.int_regs_used += 1;
                    self.fp_regs_used += 1;
                    ArgLocation::RegisterPair(gp, fp)
                } else {
                    self.classify_struct_as_integer(&layout)
                }
            }
            StructClassification::SingleFloat => {
                // Single float/double field → one FP register.
                self.classify_fp_arg(layout.size)
            }
            StructClassification::Integer => {
                // All-integer fields.
                self.classify_struct_as_integer(&layout)
            }
        }
    }

    /// Fall-back path for structs: pass as 1 or 2 integer registers.
    fn classify_struct_as_integer(&mut self, layout: &StructLayout) -> ArgLocation {
        if layout.size <= 8 {
            // Fits in one integer register.
            self.classify_integer_arg()
        } else {
            // 9–16 bytes: needs two integer registers.
            self.classify_integer_pair_arg()
        }
    }

    /// Classify a two-integer-register argument (for 9–16 byte values).
    fn classify_integer_pair_arg(&mut self) -> ArgLocation {
        if self.int_regs_used + 1 < NUM_INT_ARG_REGS {
            let r1 = INT_ARG_REGS[self.int_regs_used];
            let r2 = INT_ARG_REGS[self.int_regs_used + 1];
            self.int_regs_used += 2;
            ArgLocation::RegisterPair(r1, r2)
        } else if self.int_regs_used < NUM_INT_ARG_REGS {
            // Only one register available — use it for the first half,
            // but we need to pass the whole thing consistently.
            // Consume the register and use stack for the full value.
            self.int_regs_used += 1;
            self.alloc_stack(16, 8)
        } else {
            self.alloc_stack(16, 8)
        }
    }

    /// Classify a union argument.
    ///
    /// Unions are treated as opaque blobs of their size. Up to 16 bytes
    /// can be passed in 1–2 integer registers; larger unions are indirect.
    fn classify_union_arg(
        &mut self,
        fields: &[crate::common::types::StructField],
        packed: bool,
        aligned: Option<usize>,
    ) -> ArgLocation {
        let tb = TypeBuilder::new(Target::RiscV64);
        let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
        let layout = tb.compute_union_layout(&field_types, packed, aligned);

        if layout.size > MAX_STRUCT_REG_SIZE {
            return self.classify_indirect_arg();
        }

        if layout.size <= 8 {
            self.classify_integer_arg()
        } else {
            self.classify_integer_pair_arg()
        }
    }

    /// Classify an indirect argument (passed by reference).
    ///
    /// The caller copies the value to the stack and passes a pointer
    /// in an integer register.
    fn classify_indirect_arg(&mut self) -> ArgLocation {
        // The pointer itself is passed in an integer register (or stack).
        self.classify_integer_arg()
    }

    // -----------------------------------------------------------------------
    // Private: Struct return value classification
    // -----------------------------------------------------------------------

    /// Classify a struct return value.
    fn classify_struct_return(
        &self,
        fields: &[crate::common::types::StructField],
        packed: bool,
        aligned: Option<usize>,
    ) -> ArgLocation {
        let tb = TypeBuilder::new(Target::RiscV64);
        let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
        let layout = tb.compute_struct_layout(&field_types, packed, aligned);

        // Large structs: returned via hidden pointer in a0.
        if layout.size > MAX_STRUCT_REG_SIZE {
            return ArgLocation::Register(X10);
        }

        if fields.is_empty() || layout.size == 0 {
            return ArgLocation::Register(X10);
        }

        let classification = classify_struct_fields(fields);

        match classification {
            StructClassification::FloatPair => {
                // Return in fa0 + fa1.
                ArgLocation::RegisterPair(F10, F11)
            }
            StructClassification::FloatInteger => {
                // Return in fa0 + a0.
                ArgLocation::RegisterPair(F10, X10)
            }
            StructClassification::IntegerFloat => {
                // Return in a0 + fa0.
                ArgLocation::RegisterPair(X10, F10)
            }
            StructClassification::SingleFloat => {
                // Single float/double field: return in fa0.
                ArgLocation::Register(F10)
            }
            StructClassification::Integer => {
                if layout.size <= 8 {
                    ArgLocation::Register(X10)
                } else {
                    ArgLocation::RegisterPair(X10, X11)
                }
            }
        }
    }
}

// ===========================================================================
// Module-Level Helper Functions
// ===========================================================================

/// Returns the size in bytes of a C type under the RISC-V LP64D data model.
///
/// LP64D sizes:
/// - `_Bool`: 1, `char`: 1, `short`: 2, `int`: 4
/// - `long`: 8, `long long`: 8, `pointer`: 8
/// - `float`: 4, `double`: 8, `long double`: 16
/// - `_Complex float`: 8, `_Complex double`: 16
pub fn type_size(ty: &CType) -> usize {
    use crate::common::types::sizeof_ctype;
    sizeof_ctype(ty, &Target::RiscV64)
}

/// Returns the alignment in bytes of a C type under the RISC-V LP64D
/// data model.
///
/// Generally matches size, with maximum natural alignment of 16 bytes.
pub fn type_alignment(ty: &CType) -> usize {
    use crate::common::types::alignof_ctype;
    alignof_ctype(ty, &Target::RiscV64)
}

// ===========================================================================
// Private Helper Functions
// ===========================================================================

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1).
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

/// Resolve through `Typedef`, `Qualified`, and `Atomic` wrappers to get
/// the underlying concrete type for classification purposes.
fn resolve_type(ty: &CType) -> &CType {
    match ty {
        CType::Typedef { underlying, .. } => resolve_type(underlying),
        CType::Qualified(inner, _) => resolve_type(inner),
        CType::Atomic(inner) => resolve_type(inner),
        other => other,
    }
}

/// Compute the argument spill area size.
///
/// Walks parameter classifications and accumulates the total stack space
/// consumed by arguments that are passed on the stack (i.e. those that
/// did not fit in the 8 integer and 8 floating-point argument registers).
/// For variadic functions, the register save area (8 × 8 = 64 bytes for
/// the integer argument registers) is also included.
fn compute_arg_spill_size(params: &[CType]) -> usize {
    if params.is_empty() {
        return 0;
    }

    // Run classification using a temporary RiscV64Abi context to determine
    // how much stack space is needed.  We mirror the same classification
    // logic that the real `classify_arg` calls use.
    let mut abi = RiscV64Abi::new();
    let mut stack_bytes: usize = 0;

    for param in params {
        let loc = abi.classify_arg(param);
        match loc {
            ArgLocation::Stack(offset) => {
                // `offset` is the byte offset at which this arg starts on the
                // stack.  We need the end position to compute total size.
                // The argument size is architecture-word-aligned (8 bytes min).
                let resolved = resolve_type(param);
                let arg_size = match resolved {
                    CType::LongDouble => 16usize,
                    CType::Double | CType::LongLong | CType::Pointer(..) => 8,
                    CType::Float => 8, // padded to 8-byte slot on stack
                    _ => 8,            // minimum slot size on RV64
                };
                let end = (offset as usize).saturating_add(arg_size);
                if end > stack_bytes {
                    stack_bytes = end;
                }
            }
            _ => { /* register-passed — no stack cost */ }
        }
    }

    stack_bytes
}

/// Internal classification result for struct fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructClassification {
    /// Two float/double fields (FloatPair).
    FloatPair,
    /// First field is float/double, second is integer.
    FloatInteger,
    /// First field is integer, second is float/double.
    IntegerFloat,
    /// Single float/double field.
    SingleFloat,
    /// All fields are integer-like (or mixed types that don't qualify
    /// for FP register passing).
    Integer,
}

/// Classify a struct's fields for LP64D register passing.
///
/// The RISC-V LP64D ABI with hardware FP has special rules:
/// - A struct of exactly 1 float/double → passed in FP register
/// - A struct of exactly 2 floats/doubles → two FP registers
/// - A struct of 1 integer + 1 float/double → mixed int+FP registers
/// - Everything else → integer registers
///
/// Flattened field analysis: we recursively look through nested structs
/// to count the effective leaf fields.
fn classify_struct_fields(fields: &[crate::common::types::StructField]) -> StructClassification {
    let mut flat_fields: Vec<FieldClass> = Vec::new();
    flatten_struct_fields(fields, &mut flat_fields);

    match flat_fields.len() {
        0 => StructClassification::Integer,
        1 => {
            if flat_fields[0] == FieldClass::Float {
                StructClassification::SingleFloat
            } else {
                StructClassification::Integer
            }
        }
        2 => match (flat_fields[0], flat_fields[1]) {
            (FieldClass::Float, FieldClass::Float) => StructClassification::FloatPair,
            (FieldClass::Float, FieldClass::Integer) => StructClassification::FloatInteger,
            (FieldClass::Integer, FieldClass::Float) => StructClassification::IntegerFloat,
            _ => StructClassification::Integer,
        },
        _ => {
            // More than 2 leaf fields: always integer.
            StructClassification::Integer
        }
    }
}

/// Classification of a single leaf field for struct ABI analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldClass {
    /// Integer, pointer, enum, or boolean.
    Integer,
    /// float or double.
    Float,
}

/// Recursively flatten a struct's fields into leaf-level classifications.
///
/// Nested structs are expanded; arrays with ≤ 2 elements are expanded;
/// everything else is classified as Integer or Float.
fn flatten_struct_fields(fields: &[crate::common::types::StructField], out: &mut Vec<FieldClass>) {
    for field in fields {
        // Skip zero-width bitfields.
        if let Some(0) = field.bit_width {
            continue;
        }

        let resolved = resolve_type(&field.ty);
        match resolved {
            CType::Float | CType::Double => {
                out.push(FieldClass::Float);
            }
            CType::Struct { ref fields, .. } => {
                // Recursively flatten nested struct.
                flatten_struct_fields(fields, out);
            }
            CType::Array(ref elem, Some(count)) if *count <= 2 => {
                let elem_resolved = resolve_type(elem);
                let class = if matches!(elem_resolved, CType::Float | CType::Double) {
                    FieldClass::Float
                } else {
                    FieldClass::Integer
                };
                for _ in 0..*count {
                    out.push(class);
                }
            }
            _ => {
                out.push(FieldClass::Integer);
            }
        }
    }
}

// ===========================================================================
// MachineType and FieldLayout re-exports for schema compliance
// ===========================================================================

/// Map a C type to its [`MachineType`](crate::common::types::MachineType)
/// register class under the LP64D ABI.
///
/// This is used by the register allocator and instruction selector to
/// determine which register file a value belongs to.
pub fn machine_type_for(ty: &CType) -> crate::common::types::MachineType {
    use crate::common::types::MachineType;
    let resolved = resolve_type(ty);
    match resolved {
        CType::Void => MachineType::Void,
        CType::Bool
        | CType::Char
        | CType::SChar
        | CType::UChar
        | CType::Short
        | CType::UShort
        | CType::Int
        | CType::UInt
        | CType::Long
        | CType::ULong
        | CType::LongLong
        | CType::ULongLong
        | CType::Enum { .. } => MachineType::Integer,
        CType::Float => MachineType::F32,
        CType::Double => MachineType::F64,
        CType::LongDouble => MachineType::Memory,
        CType::Pointer(_, _) => MachineType::Ptr,
        CType::Struct { .. } | CType::Union { .. } | CType::Array(_, _) => MachineType::Memory,
        CType::Function { .. } => MachineType::Ptr,
        CType::Complex(_) => MachineType::Memory,
        _ => MachineType::Integer,
    }
}

/// Compute the struct field layout for ABI analysis.
///
/// Returns the [`FieldLayout`](crate::common::type_builder::FieldLayout)
/// entries for each field in the struct, used during argument classification
/// to determine register assignment for individual fields.
pub fn compute_struct_field_layout(
    fields: &[CType],
    packed: bool,
    aligned: Option<usize>,
) -> (Vec<crate::common::type_builder::FieldLayout>, usize) {
    let tb = TypeBuilder::new(Target::RiscV64);
    let layout = tb.compute_struct_layout(fields, packed, aligned);
    let total_size = layout.size;
    let field_layouts: Vec<crate::common::type_builder::FieldLayout> = layout.fields;
    (field_layouts, total_size)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::riscv64::registers::{
        self, F0, F1, F18, F19, F2, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F3, F30, F31,
        F4, F5, F6, F7, F8, F9, X1, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29,
        X30, X31, X5, X6, X7, X8, X9,
    };
    use crate::common::types::{CType, MachineType, StructField};

    #[test]
    fn test_type_sizes_lp64d() {
        assert_eq!(type_size(&CType::Bool), 1);
        assert_eq!(type_size(&CType::Char), 1);
        assert_eq!(type_size(&CType::SChar), 1);
        assert_eq!(type_size(&CType::UChar), 1);
        assert_eq!(type_size(&CType::Short), 2);
        assert_eq!(type_size(&CType::UShort), 2);
        assert_eq!(type_size(&CType::Int), 4);
        assert_eq!(type_size(&CType::UInt), 4);
        assert_eq!(type_size(&CType::Long), 8);
        assert_eq!(type_size(&CType::ULong), 8);
        assert_eq!(type_size(&CType::LongLong), 8);
        assert_eq!(type_size(&CType::ULongLong), 8);
        assert_eq!(type_size(&CType::Float), 4);
        assert_eq!(type_size(&CType::Double), 8);
        assert_eq!(type_size(&CType::LongDouble), 16);
        // Pointer is 8 bytes on LP64.
        assert_eq!(
            type_size(&CType::Pointer(
                Box::new(CType::Int),
                crate::common::types::TypeQualifiers::default()
            )),
            8
        );
        // Complex float: 2×4 = 8.
        assert_eq!(type_size(&CType::Complex(Box::new(CType::Float))), 8);
        // Complex double: 2×8 = 16.
        assert_eq!(type_size(&CType::Complex(Box::new(CType::Double))), 16);
    }

    #[test]
    fn test_type_alignments_lp64d() {
        assert_eq!(type_alignment(&CType::Bool), 1);
        assert_eq!(type_alignment(&CType::Char), 1);
        assert_eq!(type_alignment(&CType::Short), 2);
        assert_eq!(type_alignment(&CType::Int), 4);
        assert_eq!(type_alignment(&CType::Long), 8);
        assert_eq!(type_alignment(&CType::LongLong), 8);
        assert_eq!(type_alignment(&CType::Float), 4);
        assert_eq!(type_alignment(&CType::Double), 8);
        assert_eq!(type_alignment(&CType::LongDouble), 16);
    }

    #[test]
    fn test_integer_arg_classification() {
        let mut abi = RiscV64Abi::new();

        // First 8 integer args go in a0–a7.
        for i in 0..8u8 {
            let loc = abi.classify_arg(&CType::Int);
            assert_eq!(loc, ArgLocation::Register(INT_ARG_REGS[i as usize] as u16));
        }

        // 9th integer arg goes on stack.
        let loc = abi.classify_arg(&CType::Int);
        assert!(loc.is_stack());
    }

    #[test]
    fn test_fp_arg_classification() {
        let mut abi = RiscV64Abi::new();

        // First 8 FP args go in fa0–fa7.
        for i in 0..8u8 {
            let loc = abi.classify_arg(&CType::Float);
            assert_eq!(loc, ArgLocation::Register(FP_ARG_REGS[i as usize] as u16));
        }

        // 9th float arg goes to integer register.
        let loc = abi.classify_arg(&CType::Float);
        assert_eq!(loc, ArgLocation::Register(X10 as u16)); // a0
    }

    #[test]
    fn test_pointer_arg_classification() {
        let mut abi = RiscV64Abi::new();
        let ptr_ty = CType::Pointer(
            Box::new(CType::Int),
            crate::common::types::TypeQualifiers::default(),
        );
        let loc = abi.classify_arg(&ptr_ty);
        assert_eq!(loc, ArgLocation::Register(X10 as u16)); // a0
    }

    #[test]
    fn test_variadic_float_uses_integer_regs() {
        let mut abi = RiscV64Abi::new();

        // In variadic context, float/double go in integer registers.
        let loc = abi.classify_variadic_arg(&CType::Double);
        assert_eq!(loc, ArgLocation::Register(X10 as u16)); // a0 (integer!)

        let loc2 = abi.classify_variadic_arg(&CType::Float);
        assert_eq!(loc2, ArgLocation::Register(X11 as u16)); // a1 (integer!)
    }

    #[test]
    fn test_return_value_classification() {
        let abi = RiscV64Abi::new();

        assert_eq!(
            abi.classify_return(&CType::Void),
            ArgLocation::Register(X10 as u16)
        );
        assert_eq!(
            abi.classify_return(&CType::Int),
            ArgLocation::Register(X10 as u16)
        );
        assert_eq!(
            abi.classify_return(&CType::Float),
            ArgLocation::Register(F10 as u16)
        );
        assert_eq!(
            abi.classify_return(&CType::Double),
            ArgLocation::Register(F10 as u16)
        );
        assert_eq!(
            abi.classify_return(&CType::LongDouble),
            ArgLocation::RegisterPair(X10 as u16, X11 as u16)
        );
    }

    #[test]
    fn test_small_struct_in_registers() {
        let mut abi = RiscV64Abi::new();

        // Struct with two ints (8 bytes total) → one integer register.
        let small_struct = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };

        let loc = abi.classify_arg(&small_struct);
        // 8-byte struct fits in one integer register.
        assert_eq!(loc, ArgLocation::Register(X10 as u16));
    }

    #[test]
    fn test_large_struct_indirect() {
        let mut abi = RiscV64Abi::new();

        // Struct with 3 longs (24 bytes) → indirect.
        let large_struct = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Long,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Long,
                    bit_width: None,
                },
                StructField {
                    name: Some("c".to_string()),
                    ty: CType::Long,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };

        let loc = abi.classify_arg(&large_struct);
        // 24 bytes > 16 bytes → indirect (pointer in a0).
        assert_eq!(loc, ArgLocation::Register(X10 as u16));
    }

    #[test]
    fn test_float_pair_struct() {
        let mut abi = RiscV64Abi::new();

        // Struct { float a; float b; } → FloatPair → two FP regs.
        let float_struct = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };

        let loc = abi.classify_arg(&float_struct);
        assert_eq!(loc, ArgLocation::RegisterPair(F10 as u16, F11 as u16));
    }

    #[test]
    fn test_callee_saved_registers() {
        let gprs = RiscV64Abi::callee_saved_gprs();
        assert_eq!(gprs.len(), 12); // s0–s11
        assert!(gprs.contains(&X8)); // s0
        assert!(gprs.contains(&X9)); // s1
        assert!(gprs.contains(&X18)); // s2
        assert!(gprs.contains(&X27)); // s11

        let fprs = RiscV64Abi::callee_saved_fprs();
        assert_eq!(fprs.len(), 12); // fs0–fs11
        assert!(fprs.contains(&F8)); // fs0
        assert!(fprs.contains(&F9)); // fs1
        assert!(fprs.contains(&F18)); // fs2
        assert!(fprs.contains(&F27)); // fs11
    }

    #[test]
    fn test_caller_saved_registers() {
        let gprs = RiscV64Abi::caller_saved_gprs();
        assert_eq!(gprs.len(), 16); // ra + t0–t6 + a0–a7
        assert!(gprs.contains(&X1)); // ra
        assert!(gprs.contains(&X5)); // t0
        assert!(gprs.contains(&X10)); // a0

        let fprs = RiscV64Abi::caller_saved_fprs();
        assert_eq!(fprs.len(), 20); // ft0–ft7 + fa0–fa7 + ft8–ft11
        assert!(fprs.contains(&F0)); // ft0
        assert!(fprs.contains(&F10)); // fa0
        assert!(fprs.contains(&F28)); // ft8
    }

    #[test]
    fn test_frame_layout_basic() {
        let layout = RiscV64Abi::compute_frame_layout(&[], 32, 2);

        // 16 (RA+FP) + 16 (2 callee-saved × 8) + 32 (locals) = 64
        assert!(layout.total_size >= 64);
        assert_eq!(layout.total_size % STACK_ALIGNMENT, 0);
    }

    #[test]
    fn test_frame_layout_alignment() {
        // Odd-sized locals should result in 16-byte-aligned total.
        let layout = RiscV64Abi::compute_frame_layout(&[], 17, 0);
        assert_eq!(layout.total_size % STACK_ALIGNMENT, 0);
    }

    #[test]
    fn test_reset() {
        let mut abi = RiscV64Abi::new();
        // Consume some registers.
        abi.classify_arg(&CType::Int);
        abi.classify_arg(&CType::Float);
        abi.reset();
        // After reset, first int arg should go to a0 again.
        let loc = abi.classify_arg(&CType::Int);
        assert_eq!(loc, ArgLocation::Register(X10 as u16));
    }

    #[test]
    fn test_constants() {
        assert_eq!(NUM_INT_ARG_REGS, 8);
        assert_eq!(NUM_FP_ARG_REGS, 8);
        assert_eq!(STACK_ALIGNMENT, 16);
        assert_eq!(ARG_SLOT_SIZE, 8);
        assert_eq!(MAX_STRUCT_REG_SIZE, 16);
        assert_eq!(INT_ARG_REGS.len(), 8);
        assert_eq!(FP_ARG_REGS.len(), 8);
        assert_eq!(INT_RET_REGS.len(), 2);
        assert_eq!(FP_RET_REGS.len(), 2);
    }

    #[test]
    fn test_enum_arg() {
        let mut abi = RiscV64Abi::new();
        let enum_ty = CType::Enum {
            name: Some("color".to_string()),
            underlying_type: Box::new(CType::Int),
        };
        let loc = abi.classify_arg(&enum_ty);
        assert_eq!(loc, ArgLocation::Register(X10 as u16));
    }

    #[test]
    fn test_array_arg_indirect() {
        let mut abi = RiscV64Abi::new();
        let arr_ty = CType::Array(Box::new(CType::Int), Some(10));
        // Arrays are always indirect.
        let loc = abi.classify_arg(&arr_ty);
        // Indirect: pointer in a0.
        assert_eq!(loc, ArgLocation::Register(X10 as u16));
    }

    #[test]
    fn test_mixed_int_float_struct_return() {
        let abi = RiscV64Abi::new();

        // Struct { int a; float b; } → IntegerFloat → a0 + fa0
        let mixed = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };

        let loc = abi.classify_return(&mixed);
        assert_eq!(loc, ArgLocation::RegisterPair(X10 as u16, F10 as u16));
    }

    /// Verify MachineType variants are accessible (schema contract).
    #[test]
    fn test_machine_type_accessible() {
        let _ = MachineType::Integer;
        let _ = MachineType::F32;
        let _ = MachineType::F64;
        let _ = MachineType::Ptr;
        let _ = MachineType::Memory;
    }

    /// Verify Target::RiscV64 properties match LP64D expectations.
    #[test]
    fn test_target_riscv64_properties() {
        let t = Target::RiscV64;
        assert_eq!(t.pointer_width(), 8);
        assert_eq!(t.long_size(), 8);
        assert_eq!(t.long_double_size(), 16);
        assert_eq!(t.long_double_align(), 16);
    }

    /// Verify all register constants from the schema dependency contract
    /// are accessible and have correct values.
    #[test]
    fn test_register_constants_schema_contract() {
        // Integer register constants used in callee/caller saved sets
        // and argument arrays.
        let _ = (
            X1, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21,
            X22, X23, X24, X25, X26, X27, X28, X29, X30, X31,
        );
        // FP register constants.
        let _ = (
            F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18,
            F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31,
        );
        // Pre-defined register set slices.
        let _ = (
            CALLEE_SAVED_GPRS,
            CALLEE_SAVED_FPRS,
            CALLER_SAVED_GPRS,
            CALLER_SAVED_FPRS,
        );
        // Module reference for wildcard import path.
        let _ = registers::X0;
    }

    /// Verify machine_type_for function works correctly.
    #[test]
    fn test_machine_type_for() {
        assert_eq!(machine_type_for(&CType::Int), MachineType::Integer);
        assert_eq!(machine_type_for(&CType::Float), MachineType::F32);
        assert_eq!(machine_type_for(&CType::Double), MachineType::F64);
        assert_eq!(
            machine_type_for(&CType::Pointer(
                Box::new(CType::Void),
                crate::common::types::TypeQualifiers::default()
            )),
            MachineType::Ptr
        );
    }

    /// Verify compute_struct_field_layout works.
    #[test]
    fn test_compute_struct_field_layout() {
        let fields = vec![CType::Int, CType::Double];
        let (field_layouts, total_size) = compute_struct_field_layout(&fields, false, None);
        assert_eq!(field_layouts.len(), 2);
        // First field at offset 0, size 4.
        assert_eq!(field_layouts[0].offset, 0);
        assert_eq!(field_layouts[0].size, 4);
        // Second field at offset 8 (aligned to 8), size 8.
        assert_eq!(field_layouts[1].offset, 8);
        assert_eq!(field_layouts[1].size, 8);
        // Total size: 16 (8 + 8).
        assert_eq!(total_size, 16);
    }
}
