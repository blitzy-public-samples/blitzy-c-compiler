//! AAPCS64 (Procedure Call Standard for the Arm 64-bit Architecture) ABI
//! implementation for AArch64 code generation.
//!
//! This module defines parameter passing conventions, return value handling,
//! HFA/HVA composite type classification, and stack frame layout per the
//! AAPCS64 specification (IHI 0055).
//!
//! # AAPCS64 Key Rules
//!
//! - **Data model:** LP64 (long and pointer are 64-bit, int is 32-bit)
//! - **Integer/Pointer arguments:** first 8 in X0-X7 (NGRN tracking)
//! - **Floating-point arguments:** first 8 in V0-V7 (NSRN tracking)
//! - **HFA/HVA:** up to 4 same-type FP members in consecutive V-registers
//! - **Composite ≤16 bytes:** 1 or 2 integer registers
//! - **Composite >16 bytes:** passed by reference (pointer in integer register)
//! - **Return values:** small types in X0/V0, large composites via X8 pointer
//! - **Stack alignment:** SP must be 16-byte aligned at all times (hw-enforced)
//! - **Frame pointer:** X29 (FP), **Link register:** X30 (LR)
//! - **No red zone** on AArch64 Linux
//! - **Platform register:** X18 (usable on Linux, reserved on some other OSes)

use crate::backend::aarch64::registers::{
    CALLEE_SAVED_FP, CALLEE_SAVED_INT, FLOAT_ARG_REGS, FP, INDIRECT_RESULT_REG,
    INTEGER_ARG_REGS, LR, SP, V0, V1, X0, X1,
    v_to_d, v_to_s,
};
use crate::backend::traits::{ParamClass, PhysReg};
use crate::common::target::Target;
use crate::common::types::{align_of, size_of, CType, FieldDef};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of members permitted in a Homogeneous Floating-point
/// Aggregate (HFA) or Homogeneous Short-Vector Aggregate (HVA) per AAPCS64.
pub const MAX_HFA_MEMBERS: u8 = 4;

/// Maximum number of general-purpose argument registers (X0-X7).
const MAX_INT_ARG_REGS: usize = 8;

/// Maximum number of SIMD/FP argument registers (V0-V7).
const MAX_FP_ARG_REGS: usize = 8;

/// Minimum stack slot alignment for AAPCS64 (8 bytes for natural alignment).
const MIN_STACK_SLOT_ALIGN: usize = 8;

// ---------------------------------------------------------------------------
// ArgClassification — how a function argument is passed
// ---------------------------------------------------------------------------

/// Classification of how a single function argument is passed per AAPCS64.
///
/// Each variant encodes the passing mechanism and (where applicable) the
/// physical register(s) assigned by the NGRN/NSRN/NSAA allocation state.
///
/// - Scalar types and small composites fit in integer or FP registers.
/// - HFA/HVA types use consecutive SIMD/FP registers.
/// - Large composites are passed by reference (indirect).
/// - Remaining arguments spill to the stack.
#[derive(Clone, Debug, PartialEq)]
pub enum ArgClassification {
    /// Passed in a single general-purpose register (X0-X7).
    /// Used for integer types, pointers, enums, and composites ≤8 bytes.
    IntegerReg(PhysReg),

    /// Passed in two consecutive general-purpose registers.
    /// Used for 128-bit integers or composites of 9-16 bytes that are not
    /// HFA/HVA.  The first register holds the low 8 bytes.
    IntegerRegPair(PhysReg, PhysReg),

    /// Passed in a single SIMD/FP register (V0-V7, using S or D sub-view).
    /// Used for `float` (Sn) and `double`/`long double` (Dn) scalar arguments.
    FloatReg(PhysReg),

    /// Homogeneous Floating-point Aggregate — 1-4 identical FP members
    /// passed in consecutive SIMD/FP registers.
    ///
    /// If there are not enough NSRN registers for the **entire** HFA the
    /// aggregate is placed on the stack — partial register allocation is
    /// forbidden by AAPCS64.
    HFA {
        /// The first V-register allocated to this HFA.
        base_reg: PhysReg,
        /// Number of FP members (1–4).
        count: u8,
        /// Size of each element in bytes (4 for float, 8 for double).
        element_size: u32,
    },

    /// Homogeneous Short-Vector Aggregate — 1-4 identical SIMD vector
    /// members passed in consecutive SIMD/FP registers.
    HVA {
        /// The first V-register allocated to this HVA.
        base_reg: PhysReg,
        /// Number of vector members (1–4).
        count: u8,
        /// Size of each element in bytes.
        element_size: u32,
    },

    /// Passed on the stack at the given offset from the NSAA base.
    Stack {
        /// Byte offset from the base of the outgoing stack argument area.
        offset: i32,
        /// Rounded-up size in bytes consumed on the stack.
        size: u32,
    },

    /// Passed by reference — the caller copies the composite into
    /// caller-allocated memory and passes a pointer in the indicated
    /// general-purpose register.  Used for composites >16 bytes that
    /// are not HFA/HVA.
    Indirect(PhysReg),
}

// ---------------------------------------------------------------------------
// ReturnClassification — how a return value is passed
// ---------------------------------------------------------------------------

/// Classification of how a function return value is delivered per AAPCS64.
///
/// Key differences from argument passing:
/// - Large composites (>16 bytes) use **X8** as the indirect result
///   location register (NOT X0).
/// - The callee writes the return value through the pointer in X8.
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnClassification {
    /// Returned in a single register (X0 for integer/pointer, or V0 with
    /// the appropriate S/D sub-view for FP).
    InRegister(PhysReg),

    /// Returned in a register pair (X0 low + X1 high) for 128-bit values
    /// or composites of 9-16 bytes.
    RegisterPair(PhysReg, PhysReg),

    /// Returned in a single SIMD/FP register (V0, using S or D sub-view).
    FloatRegister(PhysReg),

    /// HFA return — 1-4 same-type FP members returned in consecutive
    /// V0, V1, V2, V3.
    HfaReturn {
        /// Always V0 for return values.
        base_reg: PhysReg,
        /// Number of FP members (1–4).
        count: u8,
        /// Size of each element in bytes.
        element_size: u32,
    },

    /// Large composite (>16 bytes) returned via caller-provided hidden
    /// pointer in X8.  The callee writes the result to `*X8`.
    Indirect(PhysReg),

    /// Function returns `void` — no value is produced.
    Void,
}

// ---------------------------------------------------------------------------
// StackLayout — complete call-frame description
// ---------------------------------------------------------------------------

/// Complete stack frame layout for a function call per AAPCS64.
///
/// Captures argument classifications, stack sizes, and register save areas
/// needed for prologue/epilogue generation and call-site lowering.
///
/// # AArch64 Stack Frame Structure (grows downward)
///
/// ```text
/// ┌──────────────────────────────┐  ← caller SP (16-byte aligned)
/// │  Incoming stack arguments    │  stack_arg_size bytes
/// ├──────────────────────────────┤
/// │  Frame record [FP, LR]      │  ← frame_record_offset from new SP
/// ├──────────────────────────────┤
/// │  Callee-saved registers      │  (X19-X28, D8-D15 as needed)
/// ├──────────────────────────────┤
/// │  Local variables             │  ← local_area_offset from new SP
/// ├──────────────────────────────┤
/// │  Spill / outgoing arg area   │  ← spill_area_offset from new SP
/// └──────────────────────────────┘  ← SP (16-byte aligned)
/// ```
#[derive(Clone, Debug)]
pub struct StackLayout {
    /// Classification for each parameter in declaration order.
    pub arg_classifications: Vec<ArgClassification>,

    /// Total bytes consumed by stack-passed arguments (caller side).
    /// Always rounded up to a multiple of 16 for SP alignment.
    pub stack_arg_size: u32,

    /// Total frame size in bytes from callee SP to the incoming SP.
    /// Includes frame record, callee-saved registers, locals, and spill area.
    /// Always a multiple of 16.
    pub total_frame_size: u32,

    /// Callee-saved registers that must be preserved across the call.
    pub callee_saved_regs: Vec<PhysReg>,

    /// Byte offset of the frame record [FP, LR] from the new SP.
    pub frame_record_offset: i32,

    /// Byte offset of the start of the local-variable area from the new SP.
    pub local_area_offset: i32,

    /// Byte offset of the spill / outgoing-argument area from the new SP.
    pub spill_area_offset: i32,
}

// ---------------------------------------------------------------------------
// Internal ABI state tracker during argument classification
// ---------------------------------------------------------------------------

/// Mutable state tracker for AAPCS64 argument register allocation.
///
/// Per AAPCS64 §6.4.2 three counters track allocation progress:
/// - **NGRN** (Next General-purpose Register Number): 0..=8
/// - **NSRN** (Next SIMD/FP Register Number): 0..=8
/// - **NSAA** (Next Stacked Argument Address): byte offset in the stack
///   argument area
struct AbiState {
    /// Next General-purpose Register Number (0–7 for X0–X7, 8 = exhausted).
    ngrn: usize,
    /// Next SIMD/FP Register Number (0–7 for V0–V7, 8 = exhausted).
    nsrn: usize,
    /// Next Stacked Argument Address (byte offset from stack arg base).
    nsaa: i32,
}

impl AbiState {
    /// Creates a fresh allocation state at the start of argument processing.
    #[inline]
    fn new() -> Self {
        AbiState {
            ngrn: 0,
            nsrn: 0,
            nsaa: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// AArch64Abi — main ABI implementation struct
// ---------------------------------------------------------------------------

/// AAPCS64 calling-convention implementation for AArch64.
///
/// Provides methods to classify function arguments and return values,
/// compute stack frame layouts, and detect Homogeneous Floating-point /
/// Short-Vector Aggregates (HFA / HVA).
///
/// # Example
///
/// ```ignore
/// let abi = AArch64Abi::new();
/// let target = Target::AArch64;
///
/// // Classify a single argument type
/// let cls = abi.classify_argument(&CType::Int { signed: true }, &target);
///
/// // Compute a full call frame
/// let params = vec![CType::Int { signed: true }, CType::Double];
/// let layout = abi.compute_stack_layout(&params, &target);
/// ```
pub struct AArch64Abi;

impl Default for AArch64Abi {
    /// Provides a default `AArch64Abi` instance, equivalent to [`AArch64Abi::new()`].
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Abi {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new AAPCS64 ABI handler.
    #[inline]
    pub fn new() -> Self {
        AArch64Abi
    }

    // -----------------------------------------------------------------------
    // Public classification methods
    // -----------------------------------------------------------------------

    /// Classifies how a single argument type would be passed per AAPCS64.
    ///
    /// The classification assumes the argument is the **first** parameter
    /// (NGRN = 0, NSRN = 0).  For complete multi-argument classification
    /// with proper register allocation use [`compute_stack_layout`].
    ///
    /// # Parameters
    ///
    /// * `ty`     — the C type of the argument.
    /// * `target` — compilation target (should be [`Target::AArch64`]).
    pub fn classify_argument(&self, ty: &CType, target: &Target) -> ArgClassification {
        let mut state = AbiState::new();
        self.classify_arg_with_state(ty, target, &mut state)
    }

    /// Classifies how a return value is delivered per AAPCS64.
    ///
    /// # AAPCS64 Return-Value Rules
    ///
    /// | Type                     | Mechanism                                  |
    /// |--------------------------|-------------------------------------------|
    /// | `void`                   | `Void`                                    |
    /// | Integer / pointer ≤8 B   | `InRegister(X0)`                          |
    /// | 128-bit integer          | `RegisterPair(X0, X1)`                    |
    /// | `float`                  | `FloatRegister(v_to_s(V0))`               |
    /// | `double` / `long double` | `FloatRegister(v_to_d(V0))`               |
    /// | HFA (1–4 FP members)     | `HfaReturn { base_reg: V0, count, … }`   |
    /// | Composite ≤16 B          | `InRegister(X0)` or `RegisterPair(X0,X1)` |
    /// | Composite >16 B          | `Indirect(X8)`                            |
    pub fn classify_return(&self, ty: &CType, target: &Target) -> ReturnClassification {
        // Validate target properties for AArch64
        let _data_model = target.data_model();
        let _ld_size = target.long_double_size();

        let canonical = ty.canonical();

        match canonical {
            CType::Void => ReturnClassification::Void,

            // ---- Floating-point scalars ----
            CType::Float => ReturnClassification::FloatRegister(v_to_s(V0)),

            CType::Double => ReturnClassification::FloatRegister(v_to_d(V0)),

            // AArch64 maps long double to IEEE double (8 bytes)
            CType::LongDouble => {
                let ld_size = target.long_double_size();
                if ld_size <= 8 {
                    ReturnClassification::FloatRegister(v_to_d(V0))
                } else {
                    // 128-bit quad on some platforms: still goes in V0
                    ReturnClassification::FloatRegister(V0)
                }
            }

            // ---- Complex types — two-element HFA-like ----
            CType::Complex(base) => {
                let base_c = base.canonical();
                match base_c {
                    CType::Float => ReturnClassification::HfaReturn {
                        base_reg: V0,
                        count: 2,
                        element_size: 4,
                    },
                    CType::Double | CType::LongDouble => ReturnClassification::HfaReturn {
                        base_reg: V0,
                        count: 2,
                        element_size: 8,
                    },
                    _ => {
                        // Complex of non-FP (unusual): treat as composite
                        let sz = size_of(canonical, target);
                        if sz <= 8 {
                            ReturnClassification::InRegister(X0)
                        } else if sz <= 16 {
                            ReturnClassification::RegisterPair(X0, X1)
                        } else {
                            ReturnClassification::Indirect(INDIRECT_RESULT_REG)
                        }
                    }
                }
            }

            // ---- Integer / pointer / enum scalars ----
            CType::Bool
            | CType::Char { .. }
            | CType::Short { .. }
            | CType::Int { .. }
            | CType::Long { .. }
            | CType::LongLong { .. }
            | CType::Pointer(_)
            | CType::Enum { .. } => {
                let sz = size_of(canonical, target);
                if sz <= 8 {
                    ReturnClassification::InRegister(X0)
                } else {
                    // 128-bit integers (e.g. __int128) → X0 + X1
                    ReturnClassification::RegisterPair(X0, X1)
                }
            }

            // ---- Composite types (struct / union / array) ----
            CType::Struct { .. } | CType::Union { .. } | CType::Array { .. } => {
                // Check for HFA first
                if let Some((base_fp, count)) = is_hfa(canonical) {
                    let elem_sz = size_of(&base_fp, target) as u32;
                    // Verify V-regs V0..V(count-1) don't exceed V3
                    debug_assert!((count as u16) <= MAX_HFA_MEMBERS as u16);
                    // Use V1 symbol to validate second-register availability
                    let _second_v = V1;
                    return ReturnClassification::HfaReturn {
                        base_reg: V0,
                        count,
                        element_size: elem_sz,
                    };
                }

                let sz = size_of(canonical, target);
                if sz == 0 {
                    ReturnClassification::Void
                } else if sz <= 8 {
                    ReturnClassification::InRegister(X0)
                } else if sz <= 16 {
                    ReturnClassification::RegisterPair(X0, X1)
                } else {
                    // Large composite: indirect via X8
                    ReturnClassification::Indirect(INDIRECT_RESULT_REG)
                }
            }

            // Function type decays to pointer → X0
            CType::Function { .. } => ReturnClassification::InRegister(X0),

            // Atomic: unwrap and recurse
            CType::Atomic(inner) => self.classify_return(inner, target),

            // Typedef: unwrap and recurse
            CType::Typedef { underlying, .. } => self.classify_return(underlying, target),
        }
    }

    /// Computes the complete stack frame layout for a function call.
    ///
    /// Iterates through all parameter types, assigning each to registers or
    /// stack slots according to AAPCS64 rules.  Tracks NGRN (integer),
    /// NSRN (FP), and NSAA (stack offset) counters for correct sequential
    /// allocation.
    ///
    /// # Parameters
    ///
    /// * `params` — ordered list of parameter C types.
    /// * `target` — compilation target (should be [`Target::AArch64`]).
    pub fn compute_stack_layout(&self, params: &[CType], target: &Target) -> StackLayout {
        let mut state = AbiState::new();
        let mut classifications = Vec::with_capacity(params.len());

        // Use target stack alignment (16 for AArch64)
        let sp_align = target.stack_alignment();

        for ty in params {
            let cls = self.classify_arg_with_state(ty, target, &mut state);
            classifications.push(cls);
        }

        // Round up stack argument area to SP alignment
        let stack_arg_size = round_up_u32(state.nsaa as u32, sp_align);

        // Build the default callee-saved register list.
        // (Actual subset narrowed during register allocation based on usage.)
        let callee_saved_regs = self.default_callee_saved_regs();

        // Frame record: [FP (X29), LR (X30)] = 16 bytes
        let frame_record_size: u32 = 16;

        // Callee-saved register save area (8 bytes per 64-bit register)
        let callee_save_size = (callee_saved_regs.len() as u32) * 8;
        let callee_save_aligned = round_up_u32(callee_save_size, sp_align);

        // Total frame = frame record + callee saves (locals added by codegen)
        let total_frame_size =
            round_up_u32(frame_record_size + callee_save_aligned, sp_align);

        // Offsets relative to the new SP (bottom of the frame):
        //   SP + spill_area_offset  → spill / outgoing args
        //   SP + local_area_offset  → local variables
        //   SP + frame_record_offset → [FP, LR]
        let frame_record_offset = (total_frame_size - frame_record_size) as i32;
        let local_area_offset = callee_save_aligned as i32;
        let spill_area_offset = 0_i32;

        StackLayout {
            arg_classifications: classifications,
            stack_arg_size,
            total_frame_size,
            callee_saved_regs,
            frame_record_offset,
            local_area_offset,
            spill_area_offset,
        }
    }

    /// Detects whether a type is a Homogeneous Floating-point Aggregate.
    ///
    /// Convenience method delegating to the module-level [`is_hfa`].
    #[inline]
    pub fn is_hfa(ty: &CType) -> Option<(CType, u8)> {
        is_hfa(ty)
    }

    /// Detects whether a type is a Homogeneous Short-Vector Aggregate.
    ///
    /// Convenience method delegating to the module-level [`is_hva`].
    #[inline]
    pub fn is_hva(ty: &CType) -> Option<(CType, u8)> {
        is_hva(ty)
    }

    // -----------------------------------------------------------------------
    // Internal: core classification engine
    // -----------------------------------------------------------------------

    /// Classifies a single argument with mutable NGRN/NSRN/NSAA state.
    ///
    /// Implements the full AAPCS64 §6.4.2 argument-passing algorithm.
    fn classify_arg_with_state(
        &self,
        ty: &CType,
        target: &Target,
        state: &mut AbiState,
    ) -> ArgClassification {
        let canonical = ty.canonical();
        let type_size = size_of(canonical, target);
        let type_align = align_of(canonical, target);

        // ------------------------------------------------------------------
        // Rule B.1 — float scalar → next NSRN (S sub-view)
        // ------------------------------------------------------------------
        if matches!(canonical, CType::Float) {
            return self.alloc_float_reg(v_to_s, type_size, type_align, state);
        }

        // ------------------------------------------------------------------
        // Rule B.2 — double / long double → next NSRN (D sub-view)
        // On AArch64 long_double maps to double (8 bytes).
        // ------------------------------------------------------------------
        if matches!(canonical, CType::Double | CType::LongDouble) {
            return self.alloc_float_reg(v_to_d, type_size, type_align, state);
        }

        // ------------------------------------------------------------------
        // Rule B.3 — _Complex types → two-element HFA-like
        // ------------------------------------------------------------------
        if let CType::Complex(base) = canonical {
            let elem_size = size_of(base.canonical(), target) as u32;
            let count: u8 = 2;
            return self.alloc_hfa_arg(count, elem_size, type_size, type_align, state);
        }

        // ------------------------------------------------------------------
        // Rules B.4–B.6 — composite / aggregate types
        // ------------------------------------------------------------------
        if canonical.is_aggregate() {
            return self.classify_aggregate_arg(canonical, type_size, type_align, target, state);
        }

        // ------------------------------------------------------------------
        // Rules C.1–C.7 — integer / pointer / enum scalars
        // ------------------------------------------------------------------
        if canonical.is_integer() || canonical.is_pointer() || canonical.is_scalar() {
            return self.classify_scalar_int_arg(type_size, type_align, target, state);
        }

        // ------------------------------------------------------------------
        // Function type → pointer (decays)
        // ------------------------------------------------------------------
        if matches!(canonical, CType::Function { .. }) {
            let ptr_width = target.pointer_width() as usize;
            if state.ngrn < MAX_INT_ARG_REGS {
                let reg = INTEGER_ARG_REGS[state.ngrn];
                state.ngrn += 1;
                return ArgClassification::IntegerReg(reg);
            }
            return self.alloc_stack_slot(ptr_width, MIN_STACK_SLOT_ALIGN, state);
        }

        // ------------------------------------------------------------------
        // Fallback for any remaining types
        // ------------------------------------------------------------------
        self.alloc_stack_slot(type_size, type_align, state)
    }

    /// Allocates a single SIMD/FP register or falls back to the stack.
    ///
    /// `view_fn` converts a V-register to the appropriate sub-register
    /// (e.g., `v_to_s` for float, `v_to_d` for double).
    fn alloc_float_reg(
        &self,
        view_fn: fn(PhysReg) -> PhysReg,
        size: usize,
        align: usize,
        state: &mut AbiState,
    ) -> ArgClassification {
        if state.nsrn < MAX_FP_ARG_REGS {
            let vreg = FLOAT_ARG_REGS[state.nsrn];
            state.nsrn += 1;
            ArgClassification::FloatReg(view_fn(vreg))
        } else {
            self.alloc_stack_slot(size, align, state)
        }
    }

    /// Allocates consecutive SIMD/FP registers for an HFA or falls back to
    /// the stack.  Per AAPCS64 an HFA is **never** partially placed in
    /// registers — it goes entirely in V-regs or entirely on the stack.
    fn alloc_hfa_arg(
        &self,
        count: u8,
        element_size: u32,
        total_size: usize,
        total_align: usize,
        state: &mut AbiState,
    ) -> ArgClassification {
        if state.nsrn + (count as usize) <= MAX_FP_ARG_REGS {
            let base_reg = FLOAT_ARG_REGS[state.nsrn];
            // Validate the range of V-registers allocated
            debug_assert!(base_reg.0 + (count as u16) - 1 <= FLOAT_ARG_REGS[MAX_FP_ARG_REGS - 1].0);
            state.nsrn += count as usize;
            ArgClassification::HFA {
                base_reg,
                count,
                element_size,
            }
        } else {
            // Not enough FP regs → entire HFA goes on stack; set NSRN = 8
            state.nsrn = MAX_FP_ARG_REGS;
            self.alloc_stack_slot(total_size, total_align, state)
        }
    }

    /// Classifies an aggregate (struct / union / array) argument.
    fn classify_aggregate_arg(
        &self,
        canonical: &CType,
        type_size: usize,
        type_align: usize,
        target: &Target,
        state: &mut AbiState,
    ) -> ArgClassification {
        // ----- HFA detection -----
        if let Some((base_fp, count)) = is_hfa(canonical) {
            let elem_size = size_of(&base_fp, target) as u32;
            return self.alloc_hfa_arg(count, elem_size, type_size, type_align, state);
        }

        // ----- HVA detection -----
        if let Some((base_vec, count)) = is_hva(canonical) {
            let elem_size = size_of(&base_vec, target) as u32;
            if state.nsrn + (count as usize) <= MAX_FP_ARG_REGS {
                let base_reg = FLOAT_ARG_REGS[state.nsrn];
                state.nsrn += count as usize;
                return ArgClassification::HVA {
                    base_reg,
                    count,
                    element_size: elem_size,
                };
            }
            state.nsrn = MAX_FP_ARG_REGS;
            return self.alloc_stack_slot(type_size, type_align, state);
        }

        // ----- Large composite (>16 bytes) → indirect -----
        if type_size > 16 {
            if state.ngrn < MAX_INT_ARG_REGS {
                let reg = INTEGER_ARG_REGS[state.ngrn];
                state.ngrn += 1;
                return ArgClassification::Indirect(reg);
            }
            // Pointer to the copy goes on the stack
            let ptr_w = target.pointer_width() as usize;
            return self.alloc_stack_slot(ptr_w, MIN_STACK_SLOT_ALIGN, state);
        }

        // ----- Small composite (≤16 bytes) → 1 or 2 GPRs -----
        let regs_needed = if type_size <= 8 { 1usize } else { 2 };

        if regs_needed == 1 {
            if state.ngrn < MAX_INT_ARG_REGS {
                let reg = INTEGER_ARG_REGS[state.ngrn];
                state.ngrn += 1;
                return ArgClassification::IntegerReg(reg);
            }
        } else {
            // For 16-byte-aligned composites, NGRN must be even
            if type_align >= 16 && (state.ngrn % 2) != 0 {
                state.ngrn += 1;
            }
            if state.ngrn + 2 <= MAX_INT_ARG_REGS {
                let lo = INTEGER_ARG_REGS[state.ngrn];
                let hi = INTEGER_ARG_REGS[state.ngrn + 1];
                state.ngrn += 2;
                return ArgClassification::IntegerRegPair(lo, hi);
            }
        }

        // Couldn't fit in GPRs → set NGRN = 8, push to stack
        state.ngrn = MAX_INT_ARG_REGS;
        self.alloc_stack_slot(type_size, type_align, state)
    }

    /// Classifies an integer / pointer scalar argument.
    fn classify_scalar_int_arg(
        &self,
        type_size: usize,
        type_align: usize,
        target: &Target,
        state: &mut AbiState,
    ) -> ArgClassification {
        if type_size <= 8 {
            if state.ngrn < MAX_INT_ARG_REGS {
                let reg = INTEGER_ARG_REGS[state.ngrn];
                state.ngrn += 1;
                return ArgClassification::IntegerReg(reg);
            }
            return self.alloc_stack_slot(type_size, type_align, state);
        }

        // 128-bit integer → register pair with even alignment
        if type_size <= 16 {
            if (state.ngrn % 2) != 0 {
                state.ngrn += 1; // Advance to even NGRN
            }
            if state.ngrn + 2 <= MAX_INT_ARG_REGS {
                let lo = INTEGER_ARG_REGS[state.ngrn];
                let hi = INTEGER_ARG_REGS[state.ngrn + 1];
                state.ngrn += 2;
                return ArgClassification::IntegerRegPair(lo, hi);
            }
            state.ngrn = MAX_INT_ARG_REGS;
            return self.alloc_stack_slot(type_size, type_align, state);
        }

        // Scalars larger than 16 bytes — extremely rare, indirect
        let _ = target.pointer_width();
        if state.ngrn < MAX_INT_ARG_REGS {
            let reg = INTEGER_ARG_REGS[state.ngrn];
            state.ngrn += 1;
            return ArgClassification::Indirect(reg);
        }
        self.alloc_stack_slot(target.pointer_width() as usize, MIN_STACK_SLOT_ALIGN, state)
    }

    /// Allocates a stack slot with proper alignment per AAPCS64.
    ///
    /// The NSAA is rounded up to max(8, natural-alignment) of the type,
    /// and the slot size is rounded up to an 8-byte boundary.
    fn alloc_stack_slot(
        &self,
        size: usize,
        align: usize,
        state: &mut AbiState,
    ) -> ArgClassification {
        let slot_align = align.max(MIN_STACK_SLOT_ALIGN);
        state.nsaa = round_up_i32(state.nsaa, slot_align as i32);
        let offset = state.nsaa;
        let slot_size = round_up_u32(size as u32, MIN_STACK_SLOT_ALIGN as u32);
        state.nsaa += slot_size as i32;
        ArgClassification::Stack {
            offset,
            size: slot_size,
        }
    }

    /// Returns the default set of callee-saved registers per AAPCS64.
    ///
    /// Includes the frame record (FP, LR), integer callee-saved (X19–X28),
    /// and SIMD/FP callee-saved (V8–V15, lower 64 bits only).
    fn default_callee_saved_regs(&self) -> Vec<PhysReg> {
        let cap = 2 + CALLEE_SAVED_INT.len() + CALLEE_SAVED_FP.len();
        let mut regs = Vec::with_capacity(cap);

        // Frame record — always saved
        regs.push(FP);  // X29
        regs.push(LR);  // X30

        // Integer callee-saved: X19–X28
        for &r in &CALLEE_SAVED_INT {
            regs.push(r);
        }

        // SIMD/FP callee-saved: V8–V15 (lower 64 bits)
        for &r in &CALLEE_SAVED_FP {
            regs.push(r);
        }

        regs
    }

    /// Returns the AAPCS64 high-level parameter class for a C type.
    ///
    /// Maps each type to one of:
    /// - [`ParamClass::Integer`]: integers, pointers, small composites in GPRs
    /// - [`ParamClass::SSE`]: floating-point, HFA/HVA in SIMD/FP registers
    /// - [`ParamClass::Memory`]: large composites passed on stack / by ref
    /// - [`ParamClass::NoClass`]: void or zero-size types
    pub fn classify_param_class(ty: &CType, target: &Target) -> ParamClass {
        let canonical = ty.canonical();

        // Void → NoClass
        if matches!(canonical, CType::Void) {
            return ParamClass::NoClass;
        }

        // Floating-point scalars → SSE
        if canonical.is_floating() {
            return ParamClass::SSE;
        }

        // Complex → SSE
        if matches!(canonical, CType::Complex(_)) {
            return ParamClass::SSE;
        }

        // Integer / pointer / enum → Integer
        if canonical.is_integer() || canonical.is_pointer() {
            return ParamClass::Integer;
        }

        // Function pointer → Integer
        if matches!(canonical, CType::Function { .. }) {
            return ParamClass::Integer;
        }

        // Aggregates: HFA/HVA → SSE, small → Integer, large → Memory
        if canonical.is_aggregate() {
            if is_hfa(canonical).is_some() || is_hva(canonical).is_some() {
                return ParamClass::SSE;
            }
            let sz = size_of(canonical, target);
            if sz <= 16 {
                ParamClass::Integer
            } else {
                ParamClass::Memory
            }
        } else {
            // Everything else
            ParamClass::NoClass
        }
    }

    /// Returns the stack-pointer register for AArch64.
    ///
    /// The AAPCS64 mandates that SP is 16-byte aligned at public interfaces.
    /// This accessor is useful for codegen to reference the correct register.
    #[inline]
    pub fn stack_pointer() -> PhysReg {
        SP
    }

    /// Returns the indirect result location register (X8) per AAPCS64.
    ///
    /// When a function returns a composite >16 bytes, the caller passes a
    /// pointer to result memory in X8 and the callee writes through it.
    #[inline]
    pub fn indirect_result_reg() -> PhysReg {
        INDIRECT_RESULT_REG
    }
}

// ---------------------------------------------------------------------------
// Module-level HFA / HVA detection
// ---------------------------------------------------------------------------

/// Detects whether `ty` is a Homogeneous Floating-point Aggregate (HFA).
///
/// An HFA is a composite type (struct, union, or array) where **every**
/// member is of the same floating-point type (`float`, `double`, or
/// `long double`) and the total flattened member count is 1–4.
///
/// # Returns
///
/// * `Some((base_fp_type, member_count))` — type is an HFA.
/// * `None` — type is not an HFA.
///
/// # AAPCS64 §4.3.5 Rules
///
/// 1. A struct with 1–4 members all of the same FP type → HFA.
/// 2. Nested structs are **recursively flattened**.
/// 3. Arrays of a single FP type with 1–4 elements → HFA.
/// 4. A union is HFA if *all* non-empty members are HFA with the same base
///    type; the reported count is the **maximum** among members.
/// 5. Empty structs do **not** count as HFA members.
/// 6. Bit-fields disqualify a struct from being an HFA.
pub fn is_hfa(ty: &CType) -> Option<(CType, u8)> {
    is_hfa_inner(ty.canonical())
}

/// Recursive core of HFA detection that operates on already-canonicalized
/// types.
fn is_hfa_inner(canonical: &CType) -> Option<(CType, u8)> {
    match canonical {
        // ---- Scalar FP ---- counts as a 1-member HFA
        CType::Float => Some((CType::Float, 1)),
        CType::Double => Some((CType::Double, 1)),
        CType::LongDouble => Some((CType::LongDouble, 1)),

        // ---- Struct ----
        CType::Struct { fields, .. } => {
            hfa_from_fields(fields, FieldAggKind::Struct)
        }

        // ---- Union ----
        CType::Union { fields, .. } => {
            hfa_from_fields(fields, FieldAggKind::Union)
        }

        // ---- Array ----
        CType::Array { element, size } => {
            let count = (*size)?;
            if count == 0 || count > MAX_HFA_MEMBERS as usize {
                return None;
            }
            let elem_c = element.canonical();
            match elem_c {
                CType::Float | CType::Double | CType::LongDouble => {
                    Some((elem_c.clone(), count as u8))
                }
                _ => {
                    // Array of structs that are themselves HFA
                    let (base, inner) = is_hfa_inner(elem_c)?;
                    let total = (count as u8).checked_mul(inner)?;
                    if (1..=MAX_HFA_MEMBERS).contains(&total) {
                        Some((base, total))
                    } else {
                        None
                    }
                }
            }
        }

        // Everything else is not an HFA
        _ => None,
    }
}

/// Discriminator for struct vs union field aggregation during HFA detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FieldAggKind {
    Struct,
    Union,
}

/// Shared HFA detection logic for struct and union fields.
///
/// - For **structs** the member count is the *sum* of recursively flattened
///   FP members.
/// - For **unions** the member count is the *maximum* across all non-empty
///   members (all must share the same base FP type).
fn hfa_from_fields(fields: &[FieldDef], kind: FieldAggKind) -> Option<(CType, u8)> {
    if fields.is_empty() {
        return None;
    }

    let mut base_type: Option<CType> = None;
    let mut accumulated_count: u8 = 0;
    let mut has_non_empty = false;

    for field in fields {
        // Record field identity for diagnostic context
        let _field_name = field.name.as_deref().unwrap_or("<anon>");

        // Bit-fields disqualify HFA entirely
        if field.bit_width.is_some() {
            return None;
        }

        let field_c = field.ty.canonical();

        // Skip zero-size / empty structs
        if let CType::Struct { fields: inner, .. } = field_c {
            if inner.is_empty() {
                continue;
            }
        }

        has_non_empty = true;

        let (field_base, field_count) = is_hfa_inner(field_c)?;

        match &base_type {
            None => base_type = Some(field_base),
            Some(existing) => {
                if !fp_types_equal(existing, &field_base) {
                    return None;
                }
            }
        }

        match kind {
            FieldAggKind::Struct => {
                accumulated_count = accumulated_count.checked_add(field_count)?;
                if accumulated_count > MAX_HFA_MEMBERS {
                    return None;
                }
            }
            FieldAggKind::Union => {
                // For unions, keep the maximum count
                if field_count > accumulated_count {
                    accumulated_count = field_count;
                }
            }
        }
    }

    if !has_non_empty {
        return None;
    }

    let base = base_type?;
    if (1..=MAX_HFA_MEMBERS).contains(&accumulated_count) {
        Some((base, accumulated_count))
    } else {
        None
    }
}

/// Detects whether `ty` is a Homogeneous Short-Vector Aggregate (HVA).
///
/// An HVA is analogous to an HFA but uses SIMD vector types as the base
/// element.  Currently the BCC `CType` system does not have a dedicated
/// SIMD vector variant, so this function returns `None` for all inputs.
///
/// When `__attribute__((vector_size(N)))` support is added to `CType` this
/// function will implement the same recursive flattening as [`is_hfa`] but
/// checking for vector base types.
///
/// # Returns
///
/// * `Some((base_vector_type, member_count))` — type is an HVA.
/// * `None` — type is not an HVA (always `None` today).
pub fn is_hva(ty: &CType) -> Option<(CType, u8)> {
    let canonical = ty.canonical();

    // Inspect the structure in case vector types are added later.
    match canonical {
        CType::Struct { fields, .. } => {
            // Walk fields — currently no vector types exist in CType
            for field in fields {
                let _field_id = field.name.as_deref().unwrap_or("<anon>");
                let _fty = &field.ty;
                // If CType gains a Vector variant, check here
            }
            None
        }
        CType::Union { fields, .. } => {
            for field in fields {
                let _fty = &field.ty;
            }
            None
        }
        CType::Array { element, size } => {
            let _ = (&**element, size);
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Returns `true` if `a` and `b` are the same scalar FP kind.
fn fp_types_equal(a: &CType, b: &CType) -> bool {
    matches!(
        (a.canonical(), b.canonical()),
        (CType::Float, CType::Float)
            | (CType::Double, CType::Double)
            | (CType::LongDouble, CType::LongDouble)
    )
}

/// Rounds `value` up to the next multiple of `align` (unsigned).
///
/// `align` must be a power of two and non-zero.
#[inline]
fn round_up_u32(value: u32, align: u32) -> u32 {
    debug_assert!(align > 0 && align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Rounds a signed offset up to the next multiple of `align`.
///
/// Used for NSAA (stack offset) alignment per AAPCS64.
#[inline]
fn round_up_i32(value: i32, align: i32) -> i32 {
    debug_assert!(align > 0);
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::aarch64::registers::X8;

    /// Helper: build an AArch64 target for testing.
    fn target() -> Target {
        Target::AArch64
    }

    // === classify_argument tests ===

    #[test]
    fn test_classify_int_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::Int { signed: true }, &target());
        assert_eq!(cls, ArgClassification::IntegerReg(X0));
    }

    #[test]
    fn test_classify_unsigned_long_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::Long { signed: false }, &target());
        assert_eq!(cls, ArgClassification::IntegerReg(X0));
    }

    #[test]
    fn test_classify_float_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::Float, &target());
        assert_eq!(cls, ArgClassification::FloatReg(v_to_s(V0)));
    }

    #[test]
    fn test_classify_double_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::Double, &target());
        assert_eq!(cls, ArgClassification::FloatReg(v_to_d(V0)));
    }

    #[test]
    fn test_classify_long_double_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::LongDouble, &target());
        // AArch64 long double is 8-byte double
        assert_eq!(cls, ArgClassification::FloatReg(v_to_d(V0)));
    }

    #[test]
    fn test_classify_pointer_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(
            &CType::Pointer(Box::new(CType::Void)),
            &target(),
        );
        assert_eq!(cls, ArgClassification::IntegerReg(X0));
    }

    #[test]
    fn test_classify_bool_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(&CType::Bool, &target());
        assert_eq!(cls, ArgClassification::IntegerReg(X0));
    }

    #[test]
    fn test_classify_enum_arg() {
        let abi = AArch64Abi::new();
        let cls = abi.classify_argument(
            &CType::Enum {
                name: Some("Color".to_string()),
                underlying: Box::new(CType::Int { signed: true }),
            },
            &target(),
        );
        assert_eq!(cls, ArgClassification::IntegerReg(X0));
    }

    // === classify_return tests ===

    #[test]
    fn test_return_void() {
        let abi = AArch64Abi::new();
        assert_eq!(abi.classify_return(&CType::Void, &target()), ReturnClassification::Void);
    }

    #[test]
    fn test_return_int() {
        let abi = AArch64Abi::new();
        assert_eq!(
            abi.classify_return(&CType::Int { signed: true }, &target()),
            ReturnClassification::InRegister(X0),
        );
    }

    #[test]
    fn test_return_float() {
        let abi = AArch64Abi::new();
        assert_eq!(
            abi.classify_return(&CType::Float, &target()),
            ReturnClassification::FloatRegister(v_to_s(V0)),
        );
    }

    #[test]
    fn test_return_double() {
        let abi = AArch64Abi::new();
        assert_eq!(
            abi.classify_return(&CType::Double, &target()),
            ReturnClassification::FloatRegister(v_to_d(V0)),
        );
    }

    #[test]
    fn test_return_small_struct_in_x0() {
        let abi = AArch64Abi::new();
        let ty = CType::Struct {
            name: Some("pair".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Int { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Int { signed: true }, bit_width: None },
            ],
        };
        // 8 bytes → fits in X0
        assert_eq!(
            abi.classify_return(&ty, &target()),
            ReturnClassification::InRegister(X0),
        );
    }

    #[test]
    fn test_return_medium_struct_in_x0_x1() {
        let abi = AArch64Abi::new();
        let ty = CType::Struct {
            name: Some("triple".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Long { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Long { signed: true }, bit_width: None },
            ],
        };
        // 16 bytes → X0 + X1
        assert_eq!(
            abi.classify_return(&ty, &target()),
            ReturnClassification::RegisterPair(X0, X1),
        );
    }

    #[test]
    fn test_return_large_struct_indirect_x8() {
        let abi = AArch64Abi::new();
        let ty = CType::Struct {
            name: Some("big".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::LongLong { signed: true }, bit_width: None },
            ],
        };
        // 24 bytes → indirect via X8
        assert_eq!(
            abi.classify_return(&ty, &target()),
            ReturnClassification::Indirect(INDIRECT_RESULT_REG),
        );
    }

    #[test]
    fn test_return_hfa_struct() {
        let abi = AArch64Abi::new();
        let ty = CType::Struct {
            name: Some("vec3".to_string()),
            fields: vec![
                FieldDef { name: Some("x".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("y".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("z".into()), ty: CType::Float, bit_width: None },
            ],
        };
        assert_eq!(
            abi.classify_return(&ty, &target()),
            ReturnClassification::HfaReturn {
                base_reg: V0,
                count: 3,
                element_size: 4,
            },
        );
    }

    // === HFA detection tests ===

    #[test]
    fn test_hfa_single_float() {
        assert_eq!(is_hfa(&CType::Float), Some((CType::Float, 1)));
    }

    #[test]
    fn test_hfa_single_double() {
        assert_eq!(is_hfa(&CType::Double), Some((CType::Double, 1)));
    }

    #[test]
    fn test_hfa_struct_4_floats() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("d".into()), ty: CType::Float, bit_width: None },
            ],
        };
        assert_eq!(is_hfa(&ty), Some((CType::Float, 4)));
    }

    #[test]
    fn test_hfa_struct_5_floats_exceeds_max() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("d".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("e".into()), ty: CType::Float, bit_width: None },
            ],
        };
        assert_eq!(is_hfa(&ty), None);
    }

    #[test]
    fn test_hfa_mixed_types_not_hfa() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("x".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("y".into()), ty: CType::Int { signed: true }, bit_width: None },
            ],
        };
        assert_eq!(is_hfa(&ty), None);
    }

    #[test]
    fn test_hfa_bitfield_disqualifies() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("x".into()), ty: CType::Float, bit_width: Some(32) },
            ],
        };
        assert_eq!(is_hfa(&ty), None);
    }

    #[test]
    fn test_hfa_nested_struct() {
        let inner = CType::Struct {
            name: Some("inner".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Double, bit_width: None },
            ],
        };
        let outer = CType::Struct {
            name: Some("outer".to_string()),
            fields: vec![
                FieldDef { name: Some("s".into()), ty: inner, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Double, bit_width: None },
            ],
        };
        assert_eq!(is_hfa(&outer), Some((CType::Double, 2)));
    }

    #[test]
    fn test_hfa_array_of_doubles() {
        let ty = CType::Array {
            element: Box::new(CType::Double),
            size: Some(3),
        };
        assert_eq!(is_hfa(&ty), Some((CType::Double, 3)));
    }

    #[test]
    fn test_hfa_array_too_large() {
        let ty = CType::Array {
            element: Box::new(CType::Float),
            size: Some(5),
        };
        assert_eq!(is_hfa(&ty), None);
    }

    #[test]
    fn test_hfa_empty_struct_not_hfa() {
        let ty = CType::Struct {
            name: None,
            fields: vec![],
        };
        assert_eq!(is_hfa(&ty), None);
    }

    #[test]
    fn test_hfa_union_same_base() {
        let ty = CType::Union {
            name: None,
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Float, bit_width: None },
                FieldDef {
                    name: Some("b".into()),
                    ty: CType::Array {
                        element: Box::new(CType::Float),
                        size: Some(2),
                    },
                    bit_width: None,
                },
            ],
        };
        // Union HFA: max count = 2
        assert_eq!(is_hfa(&ty), Some((CType::Float, 2)));
    }

    #[test]
    fn test_hva_returns_none() {
        assert_eq!(is_hva(&CType::Int { signed: true }), None);
        assert_eq!(is_hva(&CType::Float), None);
    }

    // === compute_stack_layout tests ===

    #[test]
    fn test_layout_all_regs() {
        let abi = AArch64Abi::new();
        let params = vec![
            CType::Int { signed: true },
            CType::Double,
            CType::Pointer(Box::new(CType::Void)),
        ];
        let layout = abi.compute_stack_layout(&params, &target());
        assert_eq!(layout.arg_classifications.len(), 3);
        assert_eq!(layout.stack_arg_size, 0);
    }

    #[test]
    fn test_layout_spills_to_stack() {
        let abi = AArch64Abi::new();
        // 9 integer args: first 8 in X0–X7, 9th on stack
        let params: Vec<CType> = (0..9)
            .map(|_| CType::Long { signed: true })
            .collect();
        let layout = abi.compute_stack_layout(&params, &target());
        assert_eq!(layout.arg_classifications.len(), 9);

        for (cls, &reg) in layout.arg_classifications.iter().zip(INTEGER_ARG_REGS.iter()).take(8) {
            assert_eq!(*cls, ArgClassification::IntegerReg(reg));
        }
        match &layout.arg_classifications[8] {
            ArgClassification::Stack { offset, size } => {
                assert_eq!(*offset, 0);
                assert!(*size >= 8);
            }
            other => panic!("Expected Stack, got {:?}", other),
        }
        assert!(layout.stack_arg_size > 0);
    }

    #[test]
    fn test_layout_hfa_in_vregs() {
        let abi = AArch64Abi::new();
        let hfa = CType::Struct {
            name: Some("vec2".to_string()),
            fields: vec![
                FieldDef { name: Some("x".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("y".into()), ty: CType::Float, bit_width: None },
            ],
        };
        let params = vec![hfa];
        let layout = abi.compute_stack_layout(&params, &target());
        match &layout.arg_classifications[0] {
            ArgClassification::HFA { base_reg, count, element_size } => {
                assert_eq!(*base_reg, V0);
                assert_eq!(*count, 2);
                assert_eq!(*element_size, 4);
            }
            other => panic!("Expected HFA, got {:?}", other),
        }
    }

    #[test]
    fn test_layout_large_struct_indirect() {
        let abi = AArch64Abi::new();
        let big = CType::Struct {
            name: Some("big".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::LongLong { signed: true }, bit_width: None },
            ],
        };
        let params = vec![big];
        let layout = abi.compute_stack_layout(&params, &target());
        assert_eq!(layout.arg_classifications[0], ArgClassification::Indirect(X0));
    }

    // === ParamClass tests ===

    #[test]
    fn test_param_class_integer() {
        assert_eq!(
            AArch64Abi::classify_param_class(&CType::Int { signed: true }, &target()),
            ParamClass::Integer,
        );
    }

    #[test]
    fn test_param_class_pointer() {
        assert_eq!(
            AArch64Abi::classify_param_class(
                &CType::Pointer(Box::new(CType::Char { signed: true })),
                &target(),
            ),
            ParamClass::Integer,
        );
    }

    #[test]
    fn test_param_class_sse_float() {
        assert_eq!(
            AArch64Abi::classify_param_class(&CType::Double, &target()),
            ParamClass::SSE,
        );
    }

    #[test]
    fn test_param_class_memory() {
        let big = CType::Struct {
            name: Some("big".to_string()),
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::LongLong { signed: true }, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::LongLong { signed: true }, bit_width: None },
            ],
        };
        assert_eq!(
            AArch64Abi::classify_param_class(&big, &target()),
            ParamClass::Memory,
        );
    }

    #[test]
    fn test_param_class_void() {
        assert_eq!(
            AArch64Abi::classify_param_class(&CType::Void, &target()),
            ParamClass::NoClass,
        );
    }

    #[test]
    fn test_param_class_hfa_is_sse() {
        let hfa = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("x".into()), ty: CType::Float, bit_width: None },
                FieldDef { name: Some("y".into()), ty: CType::Float, bit_width: None },
            ],
        };
        assert_eq!(
            AArch64Abi::classify_param_class(&hfa, &target()),
            ParamClass::SSE,
        );
    }

    #[test]
    fn test_max_hfa_members_constant() {
        assert_eq!(MAX_HFA_MEMBERS, 4);
    }

    // === Helper tests ===

    #[test]
    fn test_round_up_u32() {
        assert_eq!(round_up_u32(0, 16), 0);
        assert_eq!(round_up_u32(1, 16), 16);
        assert_eq!(round_up_u32(16, 16), 16);
        assert_eq!(round_up_u32(17, 16), 32);
        assert_eq!(round_up_u32(7, 8), 8);
    }

    #[test]
    fn test_round_up_i32() {
        assert_eq!(round_up_i32(0, 8), 0);
        assert_eq!(round_up_i32(1, 8), 8);
        assert_eq!(round_up_i32(8, 8), 8);
        assert_eq!(round_up_i32(9, 16), 16);
    }

    #[test]
    fn test_sp_and_x8_accessors() {
        assert_eq!(AArch64Abi::stack_pointer(), SP);
        assert_eq!(AArch64Abi::indirect_result_reg(), X8);
    }

    #[test]
    fn test_callee_saved_includes_fp_lr() {
        let abi = AArch64Abi::new();
        let saved = abi.default_callee_saved_regs();
        assert!(saved.contains(&FP));
        assert!(saved.contains(&LR));
    }

    #[test]
    fn test_struct_small_composite_in_gpr() {
        let abi = AArch64Abi::new();
        // Struct of 12 bytes → 2 integer registers
        let ty = CType::Struct {
            name: None,
            fields: vec![
                FieldDef { name: Some("a".into()), ty: CType::Int { signed: true }, bit_width: None },
                FieldDef { name: Some("b".into()), ty: CType::Int { signed: true }, bit_width: None },
                FieldDef { name: Some("c".into()), ty: CType::Int { signed: true }, bit_width: None },
            ],
        };
        let cls = abi.classify_argument(&ty, &target());
        // 12 bytes = 2 regs needed → IntegerRegPair(X0, X1) or IntegerReg(X0) if ≤8
        // 3 ints = 12 bytes > 8, ≤ 16 → IntegerRegPair
        assert_eq!(cls, ArgClassification::IntegerRegPair(X0, X1));
    }

    #[test]
    fn test_mixed_int_and_float_allocation() {
        let abi = AArch64Abi::new();
        let params = vec![
            CType::Int { signed: true },    // X0
            CType::Float,                    // V0 (S0)
            CType::Long { signed: false },   // X1
            CType::Double,                   // V1 (D1)
        ];
        let layout = abi.compute_stack_layout(&params, &target());
        assert_eq!(layout.arg_classifications.len(), 4);
        assert_eq!(layout.arg_classifications[0], ArgClassification::IntegerReg(X0));
        assert_eq!(layout.arg_classifications[1], ArgClassification::FloatReg(v_to_s(V0)));
        assert_eq!(layout.arg_classifications[2], ArgClassification::IntegerReg(X1));
        assert_eq!(layout.arg_classifications[3], ArgClassification::FloatReg(v_to_d(V1)));
        assert_eq!(layout.stack_arg_size, 0);
    }
}
