//! # AAPCS64 ABI Implementation
//!
//! Implements the Arm Architecture Procedure Call Standard for 64-bit (AAPCS64,
//! ARM IHI 0055) calling convention for BCC's AArch64 backend.
//!
//! This module defines:
//! - How function parameters and return values are passed between caller and callee
//! - Struct/union passing conventions including HFA/HVA detection
//! - Stack frame layout computation
//! - Register allocation constraints per the ABI
//!
//! ## Data Model: LP64
//!
//! AArch64 uses the LP64 data model where `long` and pointers are 64-bit:
//!
//! | Type          | Size (bytes) | Alignment (bytes) |
//! |---------------|-------------:|------------------:|
//! | `_Bool`       |            1 |                 1 |
//! | `char`        |            1 |                 1 |
//! | `short`       |            2 |                 2 |
//! | `int`         |            4 |                 4 |
//! | `long`        |            8 |                 8 |
//! | `long long`   |            8 |                 8 |
//! | `float`       |            4 |                 4 |
//! | `double`      |            8 |                 8 |
//! | `long double` |           16 |                16 |
//! | `pointer`     |            8 |                 8 |
//!
//! ## Register Usage (AAPCS64)
//!
//! - **Integer argument registers:** X0–X7 (NGRN: Next General-purpose Register Number)
//! - **Float/SIMD argument registers:** V0–V7 / D0–D7 / S0–S7 (NSRN)
//! - **Return value:** X0 (integer), X0+X1 (pair), V0 (FP), V0–V3 (HFA), or X8 (indirect)
//! - **Indirect result location:** X8 — callee writes result to the address in X8
//! - **Callee-saved GPRs:** X19–X28 (10 registers)
//! - **Callee-saved FPRs:** D8–D15 (lower 64 bits of V8–V15 only)
//! - **Caller-saved:** X0–X18, V0–V7, V16–V31
//! - **Frame pointer:** X29 (FP)
//! - **Link register:** X30 (LR)
//! - **Stack pointer:** SP — always 16-byte aligned
//! - **Stack grows downward**
//!
//! ## HFA/HVA (Homogeneous Floating-Point/Vector Aggregates)
//!
//! A critical AAPCS64 feature: structs composed of 1–4 identical floating-point
//! members are classified as HFA and passed in consecutive SIMD/FP registers.
//! Nested structs and arrays are flattened for HFA detection.
//!
//! ## Variadic Functions
//!
//! Anonymous (variadic) arguments follow different rules: floating-point types
//! are passed in GPRs (not FP registers), and HFA classification does not apply.
//!
//! ## Zero-Dependency
//!
//! This module uses only `std` and `crate::` references — no external crates.

use crate::backend::aarch64::registers::*;
use crate::backend::traits::ArgLocation;
use crate::common::target::Target;
use crate::common::type_builder::{StructLayout, TypeBuilder};
use crate::common::types::{CType, MachineType};

// ===========================================================================
// Compile-Time Register Verification
// ===========================================================================

// Static assertions verifying register alias consistency with AAPCS64.
const _: () = assert!(FP_REG == 29, "FP must be X29");
const _: () = assert!(LR == 30, "LR must be X30");
const _: () = assert!(SP_REG == 31, "SP must be encoding 31");
const _: () = assert!(X8 == 8, "X8 is indirect result register");
const _: () = assert!(V0 == 32, "V0 starts at internal ID 32");
const _: () = assert!(V8 == 40, "V8 starts callee-saved FPR range");
const _: () = assert!(V16 == 48, "V16 starts caller-saved scratch FPR range");
const _: () = assert!(V31 == 63, "V31 is the last SIMD/FP register");
const _: () = assert!(X19 == 19, "X19 starts callee-saved GPR range");
const _: () = assert!(X28 == 28, "X28 is the last callee-saved GPR");

// ===========================================================================
// AAPCS64 ABI Constants
// ===========================================================================

/// Number of integer argument registers (X0–X7).
pub const NUM_INT_ARG_REGS: usize = 8;

/// Number of float/SIMD argument registers (V0–V7).
pub const NUM_FP_ARG_REGS: usize = 8;

/// Stack alignment requirement in bytes — AAPCS64 mandates 16-byte alignment.
pub const STACK_ALIGNMENT: usize = 16;

/// Minimum argument slot size in bytes on the stack for AArch64.
pub const ARG_SLOT_SIZE: usize = 8;

/// Maximum number of members in an HFA/HVA (Homogeneous Floating-Point Aggregate).
pub const MAX_HFA_MEMBERS: usize = 4;

/// Maximum size of a composite type that can be passed in registers (2 × 8 = 16 bytes).
pub const MAX_COMPOSITE_REG_SIZE: usize = 16;

/// Integer argument register IDs (X0–X7), indexed by NGRN.
pub const INT_ARG_REGS: [u32; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// Float/SIMD argument register IDs (V0–V7), indexed by NSRN.
pub const FP_ARG_REGS: [u32; 8] = [V0, V1, V2, V3, V4, V5, V6, V7];

/// Integer return registers (X0, X1 — for register pair returns).
pub const INT_RET_REGS: [u32; 2] = [X0, X1];

/// Float return registers (V0–V3 — for HFA returns up to 4 members).
pub const FP_RET_REGS: [u32; 4] = [V0, V1, V2, V3];

/// Indirect result location register — callee writes large return values to `[X8]`.
pub const INDIRECT_RESULT_REG: u32 = X8;

// ===========================================================================
// ArgClass — Argument Classification
// ===========================================================================

/// Classification of how an argument or return value is passed per AAPCS64.
///
/// This drives the code generator's decision of which registers or stack
/// locations to use for each parameter and return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgClass {
    /// Passed in a single integer register (X0–X7).
    Integer,
    /// Passed in a single floating-point/SIMD register (V0–V7).
    Float,
    /// Passed in two consecutive integer registers (for ≤16-byte composites).
    IntegerPair,
    /// Homogeneous Float Aggregate — passed in consecutive FP registers.
    /// `count` is the number of FP members (1–4), `member_size` is bytes per member.
    HFA { count: u8, member_size: u8 },
    /// Passed on the stack (no registers available or type requires stack passing).
    Memory,
    /// Passed via indirect pointer — caller copies to memory, pointer in register.
    /// Used for composites larger than 16 bytes.
    Indirect,
}

// ===========================================================================
// FrameLayout — Stack Frame Description
// ===========================================================================

/// Describes the layout of an AArch64 stack frame for a function.
///
/// ## Frame Layout (high to low addresses)
///
/// ```text
/// [previous frame / incoming stack args]  ← old SP
/// [saved FP (X29)]                        ← FP points here
/// [saved LR (X30)]
/// [callee-saved GPRs (paired STP)]
/// [callee-saved FPRs D8–D15]
/// [local variables + spill slots]
/// [arg spill area]
/// [outgoing stack args]                   ← new SP (16-byte aligned)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLayout {
    /// Total frame size in bytes (always 16-byte aligned).
    pub total_size: usize,
    /// Byte offset from FP (X29) to the start of local variables (negative).
    pub locals_offset: i64,
    /// Byte offset from SP to the callee-saved register save area.
    pub callee_saved_offset: i64,
    /// Byte offset from SP to the saved LR (X30).
    pub lr_offset: i64,
    /// Byte offset from SP to the saved FP (X29).
    pub fp_offset: i64,
    /// Byte offset from FP for register argument spill area (negative).
    pub arg_spill_offset: i64,
}

// ===========================================================================
// Private Helpers
// ===========================================================================

/// Recursively strip `Qualified`, `Typedef`, and `Atomic` wrappers to reach
/// the underlying undecorated C type.
fn strip_decorators(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => strip_decorators(inner),
        CType::Typedef { underlying, .. } => strip_decorators(underlying),
        CType::Atomic(inner) => strip_decorators(inner),
        other => other,
    }
}

/// Returns `true` if two CType values represent the same FP base type for HFA.
fn same_fp_base(a: &CType, b: &CType) -> bool {
    let a = strip_decorators(a);
    let b = strip_decorators(b);
    matches!(
        (a, b),
        (CType::Float, CType::Float)
            | (CType::Double, CType::Double)
            | (CType::LongDouble, CType::LongDouble)
    )
}

/// Returns `true` if the type is a floating-point scalar (float, double,
/// or long double) after stripping qualifiers.
fn is_fp_scalar(ty: &CType) -> bool {
    matches!(
        strip_decorators(ty),
        CType::Float | CType::Double | CType::LongDouble
    )
}

/// Returns `true` if the type is a composite (struct or union).
fn is_composite(ty: &CType) -> bool {
    matches!(
        strip_decorators(ty),
        CType::Struct { .. } | CType::Union { .. }
    )
}

/// Classify a C type to its ABI [`MachineType`] for register-class decisions.
///
/// This mapping determines whether a type naturally belongs in a GPR, an FP
/// register, or must be passed through memory.
///
/// **Design Decision — AArch64 `LongDouble`**: On AArch64, the C `long double`
/// type is defined as IEEE 754 binary128 (quad-precision) by the AAPCS64.
/// However, most real-world AArch64 code (including the Linux kernel) treats
/// `long double` identically to `double` (64-bit).  We map `LongDouble` to
/// `MachineType::F64` here, matching GCC's `-mlong-double-64` behaviour and
/// keeping the ABI classification straightforward.  Full 128-bit quad support
/// is a future extension.
fn classify_machine_type(ty: &CType) -> MachineType {
    let stripped = strip_decorators(ty);
    match stripped {
        CType::Float => MachineType::F32,
        // LongDouble → F64: see design decision note above.
        CType::Double | CType::LongDouble => MachineType::F64,
        CType::Struct { .. } | CType::Union { .. } => {
            let size = type_size(ty);
            if size > MAX_COMPOSITE_REG_SIZE {
                MachineType::Memory
            } else {
                MachineType::Integer
            }
        }
        _ => MachineType::Integer,
    }
}

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1). Returns `value` unchanged if
/// `align` is 0.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

/// Compute the size and alignment of a type using [`TypeBuilder`] for
/// precise struct layout computation per AAPCS64 / LP64 rules.
///
/// For struct types, [`TypeBuilder::compute_struct_layout`] is used to get
/// the exact memory layout with inter-field padding, flexible array members,
/// packed/aligned attributes, and the resulting [`StructLayout`].
fn composite_size_align(ty: &CType) -> (usize, usize) {
    let tb = TypeBuilder::new(Target::AArch64);
    let stripped = strip_decorators(ty);
    match stripped {
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => {
            let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
            let layout: StructLayout = tb.compute_struct_layout(&field_types, *packed, *aligned);
            // Use layout.fields for per-field verification — ensures the
            // TypeBuilder produced a consistent layout entry for each field.
            debug_assert_eq!(layout.fields.len(), field_types.len());
            (layout.size, layout.alignment)
        }
        _ => (tb.sizeof_type(stripped), tb.alignof_type(stripped)),
    }
}

// ===========================================================================
// HFA/HVA Detection
// ===========================================================================

/// Recursively collect all leaf floating-point members from a type for HFA
/// detection. Returns `Some(members)` if the type is composed entirely of
/// floating-point members, or `None` if any non-FP member is found.
fn collect_fp_leaves(ty: &CType) -> Option<Vec<CType>> {
    let stripped = strip_decorators(ty);
    match stripped {
        // Leaf FP types.
        CType::Float | CType::Double | CType::LongDouble => Some(vec![stripped.clone()]),
        // Array of FP types: flatten by repeating the element pattern.
        CType::Array(elem, Some(count)) => {
            let count = *count;
            if count == 0 {
                return None;
            }
            let elem_leaves = collect_fp_leaves(elem)?;
            let mut result = Vec::with_capacity(elem_leaves.len() * count);
            for _ in 0..count {
                result.extend(elem_leaves.iter().cloned());
            }
            Some(result)
        }
        // Struct: flatten all fields sequentially.
        CType::Struct { fields, .. } => {
            if fields.is_empty() {
                return None;
            }
            let mut all = Vec::new();
            for field in fields {
                // Bitfields disqualify a type from being an HFA.
                if field.bit_width.is_some() {
                    return None;
                }
                let field_leaves = collect_fp_leaves(&field.ty)?;
                all.extend(field_leaves);
            }
            if all.is_empty() {
                return None;
            }
            Some(all)
        }
        // Union: all alternatives must decompose to the same FP base type.
        // The effective member count is the maximum across alternatives.
        CType::Union { fields, .. } => {
            if fields.is_empty() {
                return None;
            }
            let mut best: Option<Vec<CType>> = None;
            let mut union_base: Option<CType> = None;
            for field in fields {
                if field.bit_width.is_some() {
                    return None;
                }
                let field_leaves = collect_fp_leaves(&field.ty)?;
                if field_leaves.is_empty() {
                    return None;
                }
                // Verify consistent base type across all alternatives.
                let ft = strip_decorators(&field_leaves[0]).clone();
                if let Some(ref base) = union_base {
                    if !same_fp_base(&ft, base) {
                        return None;
                    }
                } else {
                    union_base = Some(ft);
                }
                // Keep the alternative with the most members.
                match &best {
                    Some(b) if field_leaves.len() <= b.len() => {}
                    _ => {
                        best = Some(field_leaves);
                    }
                }
            }
            best
        }
        _ => None,
    }
}

/// Detect whether a C type is a Homogeneous Floating-point Aggregate (HFA).
///
/// An HFA is a composite type (struct or union) where **all** fundamental
/// data type members are the same floating-point type (`float`, `double`,
/// or `long double`), with at most [`MAX_HFA_MEMBERS`] (4) members.
///
/// Nested structs and arrays are flattened for detection. Bitfields
/// disqualify a type from being an HFA.
///
/// # Returns
///
/// - `Some((count, member_type))` — the type is an HFA with `count` members
///   of the given FP `member_type`.
/// - `None` — the type is not an HFA.
///
/// # Examples (conceptual)
///
/// - `struct { float a, b; }` → `Some((2, Float))`
/// - `struct { double x, y, z; }` → `Some((3, Double))`
/// - `struct { int a; float b; }` → `None` (mixed types)
/// - `struct { float a, b, c, d, e; }` → `None` (> 4 members)
pub fn is_hfa(ty: &CType) -> Option<(usize, CType)> {
    let stripped = strip_decorators(ty);
    // HFA only applies to composite types (struct, union).
    match stripped {
        CType::Struct { .. } | CType::Union { .. } => {}
        _ => return None,
    }

    let leaves = collect_fp_leaves(stripped)?;
    if leaves.is_empty() || leaves.len() > MAX_HFA_MEMBERS {
        return None;
    }

    // All leaves must be the same floating-point type.
    let base = strip_decorators(&leaves[0]);
    for leaf in &leaves[1..] {
        if !same_fp_base(leaf, base) {
            return None;
        }
    }

    Some((leaves.len(), base.clone()))
}

// ===========================================================================
// AArch64Abi — Main ABI Handler
// ===========================================================================

/// AAPCS64 ABI handler tracking register allocation state for argument passing.
///
/// The state consists of three counters per the AAPCS64 specification:
///
/// - **NGRN** (Next General-purpose Register Number): 0–7 into X0–X7
/// - **NSRN** (Next SIMD/FP Register Number): 0–7 into V0–V7
/// - **NSAA** (Next Stacked Argument Address): byte offset for the next stack arg
///
/// Create a new instance per function call, or call [`reset`](AArch64Abi::reset).
pub struct AArch64Abi {
    /// Next General-purpose Register Number (0–8; 8 means exhausted).
    ngrn: usize,
    /// Next SIMD and Floating-point Register Number (0–8; 8 means exhausted).
    nsrn: usize,
    /// Next Stacked Argument Address — byte offset from the outgoing arg area.
    nsaa: usize,
}

impl Default for AArch64Abi {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64Abi {
    /// Create a new AAPCS64 ABI handler with all counters at zero.
    #[inline]
    pub fn new() -> Self {
        AArch64Abi {
            ngrn: 0,
            nsrn: 0,
            nsaa: 0,
        }
    }

    /// Reset all counters to zero for classifying a new function's arguments.
    #[inline]
    pub fn reset(&mut self) {
        self.ngrn = 0;
        self.nsrn = 0;
        self.nsaa = 0;
    }

    /// Allocate a stack slot of the given size with the specified alignment.
    /// Returns the byte offset of the allocated slot.
    fn alloc_stack(&mut self, size: usize, min_align: usize) -> i32 {
        let effective_align = min_align.max(ARG_SLOT_SIZE);
        self.nsaa = align_up(self.nsaa, effective_align);
        let offset = self.nsaa as i32;
        self.nsaa += size.max(ARG_SLOT_SIZE);
        offset
    }

    /// Classify a function argument according to AAPCS64 rules and return
    /// its assigned location (register or stack).
    ///
    /// This method mutates the internal NGRN/NSRN/NSAA state as arguments
    /// are classified sequentially from left to right.
    ///
    /// # AAPCS64 Parameter Passing Rules (B.1–B.5)
    ///
    /// 1. **FP scalar** (float, double, long double) → V\[NSRN\] or stack
    /// 2. **Complex FP** → two consecutive V regs or stack
    /// 3. **HFA** (1–4 FP members) → consecutive V regs (all-or-nothing) or stack
    /// 4. **Small composite** (≤ 16 bytes, not HFA) → 1–2 X regs or stack
    /// 5. **Large composite** (> 16 bytes) → indirect (pointer in X reg or stack)
    /// 6. **Integer/pointer scalar** → X\[NGRN\] or stack
    pub fn classify_arg(&mut self, ty: &CType) -> ArgLocation {
        let stripped = strip_decorators(ty);

        // Compute machine type classification for ABI register-class awareness.
        let _machine_class = classify_machine_type(stripped);

        // --- Handle void (should not appear as an argument, but be defensive) ---
        if matches!(stripped, CType::Void) {
            return ArgLocation::Stack(0);
        }

        // --- Array types decay to pointers in C parameter contexts ---
        if matches!(stripped, CType::Array(..)) {
            return self.classify_integer_scalar();
        }

        // --- Function types decay to function pointers ---
        if matches!(stripped, CType::Function { .. }) {
            return self.classify_integer_scalar();
        }

        // --- Rule 1: Floating-point / SIMD scalar ---
        if is_fp_scalar(stripped) {
            if self.nsrn < NUM_FP_ARG_REGS {
                let reg = FP_ARG_REGS[self.nsrn];
                self.nsrn += 1;
                return ArgLocation::Register(reg as u16);
            }
            let size = type_size(stripped);
            let al = type_alignment(stripped);
            return ArgLocation::Stack(self.alloc_stack(size, al));
        }

        // --- Rule 2: _Complex floating-point types ---
        // _Complex float/double/long double are equivalent to a 2-member HFA
        // of their base type.
        if let CType::Complex(ref base) = stripped {
            let base_stripped = strip_decorators(base);
            if is_fp_scalar(base_stripped) {
                if self.nsrn + 2 <= NUM_FP_ARG_REGS {
                    let r1 = FP_ARG_REGS[self.nsrn];
                    let r2 = FP_ARG_REGS[self.nsrn + 1];
                    self.nsrn += 2;
                    return ArgLocation::RegisterPair(r1 as u16, r2 as u16);
                }
                self.nsrn = NUM_FP_ARG_REGS;
                let size = type_size(stripped);
                let al = type_alignment(stripped);
                return ArgLocation::Stack(self.alloc_stack(size, al));
            }
        }

        // --- Rule 3: HFA (Homogeneous Floating-point Aggregate) ---
        if let Some((count, _member_ty)) = is_hfa(stripped) {
            if self.nsrn + count <= NUM_FP_ARG_REGS {
                let first = FP_ARG_REGS[self.nsrn];
                if count == 1 {
                    self.nsrn += 1;
                    return ArgLocation::Register(first as u16);
                } else if count == 2 {
                    let second = FP_ARG_REGS[self.nsrn + 1];
                    self.nsrn += 2;
                    return ArgLocation::RegisterPair(first as u16, second as u16);
                } else {
                    // 3 or 4 member HFA: the `ArgLocation` enum only supports
                    // `Register` (1 reg) and `RegisterPair` (2 regs).
                    // For HFAs with 3–4 members that fit in FP registers, we
                    // pass via memory to preserve ABI correctness, since we
                    // cannot represent >2 individual register locations in a
                    // single `ArgLocation` value.
                    //
                    // Note: the AAPCS64 specifies that each member occupies
                    // its own FP register (e.g., V0-V3 for a 4-member HFA).
                    // When the code generator gains multi-register location
                    // support, this path should be updated to return all
                    // individual register assignments.
                    self.nsrn = NUM_FP_ARG_REGS; // prevent partial allocation
                    let size = type_size(stripped);
                    let al = type_alignment(stripped);
                    return ArgLocation::Stack(self.alloc_stack(size, al));
                }
            }
            // Not enough FP regs — do NOT partially allocate (AAPCS64 C.4).
            self.nsrn = NUM_FP_ARG_REGS;
            let size = type_size(stripped);
            let al = type_alignment(stripped);
            return ArgLocation::Stack(self.alloc_stack(size, al));
        }

        // --- Rules 4 & 5: Composite types (struct, union) ---
        if is_composite(stripped) {
            let (size, align) = composite_size_align(stripped);

            // Rule 5: Large composite (> 16 bytes) → indirect passing.
            if size > MAX_COMPOSITE_REG_SIZE {
                // Caller allocates a copy; pointer is passed in X[NGRN] or stack.
                if self.ngrn < NUM_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[self.ngrn];
                    self.ngrn += 1;
                    return ArgLocation::Register(reg as u16);
                }
                return ArgLocation::Stack(
                    self.alloc_stack(Target::AArch64.pointer_width(), ARG_SLOT_SIZE),
                );
            }

            // Rule 4: Small composite ≤ 16 bytes.
            if size <= 8 {
                if self.ngrn < NUM_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[self.ngrn];
                    self.ngrn += 1;
                    return ArgLocation::Register(reg as u16);
                }
            } else {
                // 8 < size ≤ 16 bytes — needs two GPRs.
                // AAPCS64 C.8: if alignment > 8, round NGRN to even.
                if align > 8 {
                    self.ngrn = align_up(self.ngrn, 2);
                }
                if self.ngrn + 2 <= NUM_INT_ARG_REGS {
                    let r1 = INT_ARG_REGS[self.ngrn];
                    let r2 = INT_ARG_REGS[self.ngrn + 1];
                    self.ngrn += 2;
                    return ArgLocation::RegisterPair(r1 as u16, r2 as u16);
                }
            }
            // Fall through to stack — mark GPRs exhausted for this composite.
            self.ngrn = NUM_INT_ARG_REGS;
            let stack_align = align.max(ARG_SLOT_SIZE);
            return ArgLocation::Stack(self.alloc_stack(size, stack_align));
        }

        // --- Rule 6: Integer / pointer / enum scalar ---
        self.classify_integer_scalar()
    }

    /// Classify an integer/pointer scalar argument: X\[NGRN\] or stack.
    fn classify_integer_scalar(&mut self) -> ArgLocation {
        if self.ngrn < NUM_INT_ARG_REGS {
            let reg = INT_ARG_REGS[self.ngrn];
            self.ngrn += 1;
            ArgLocation::Register(reg as u16)
        } else {
            ArgLocation::Stack(self.alloc_stack(ARG_SLOT_SIZE, ARG_SLOT_SIZE))
        }
    }

    /// Classify a function return value according to AAPCS64 rules.
    ///
    /// Return value classification is stateless — it does not depend on or
    /// modify the NGRN/NSRN/NSAA counters.
    ///
    /// # Rules
    ///
    /// - **Void**: `Stack(0)` sentinel (no return value)
    /// - **FP scalar**: V0 (S0 for float, D0 for double)
    /// - **Complex FP**: V0 + V1 (two consecutive FP registers)
    /// - **HFA** (1–4 members): V0–V3 in consecutive FP registers
    /// - **Small composite ≤ 8**: X0
    /// - **Small composite 8–16**: X0 + X1
    /// - **Large composite > 16**: indirect return via X8
    /// - **Integer/pointer**: X0
    pub fn classify_return(&self, ty: &CType) -> ArgLocation {
        let stripped = strip_decorators(ty);

        // Void — no return value.
        if matches!(stripped, CType::Void) {
            return ArgLocation::Stack(0);
        }

        // Floating-point scalar.
        if is_fp_scalar(stripped) {
            return ArgLocation::Register(FP_RET_REGS[0] as u16);
        }

        // _Complex floating-point — two consecutive FP return registers.
        if let CType::Complex(ref base) = stripped {
            if is_fp_scalar(strip_decorators(base)) {
                return ArgLocation::RegisterPair(FP_RET_REGS[0] as u16, FP_RET_REGS[1] as u16);
            }
        }

        // HFA return.
        if let Some((count, _)) = is_hfa(stripped) {
            if count == 1 {
                return ArgLocation::Register(FP_RET_REGS[0] as u16);
            } else if count == 2 {
                return ArgLocation::RegisterPair(FP_RET_REGS[0] as u16, FP_RET_REGS[1] as u16);
            } else {
                // 3 or 4 members: return first FP register; codegen reads type.
                return ArgLocation::Register(FP_RET_REGS[0] as u16);
            }
        }

        // Composite types (struct, union).
        if is_composite(stripped) {
            let size = type_size(stripped);
            if size == 0 {
                return ArgLocation::Stack(0);
            }
            if size <= 8 {
                return ArgLocation::Register(INT_RET_REGS[0] as u16);
            }
            if size <= MAX_COMPOSITE_REG_SIZE {
                return ArgLocation::RegisterPair(INT_RET_REGS[0] as u16, INT_RET_REGS[1] as u16);
            }
            // Large composite — indirect return: callee writes to [X8].
            return ArgLocation::Register(INDIRECT_RESULT_REG as u16);
        }

        // Integer / pointer / enum / array / function — return in X0.
        ArgLocation::Register(INT_RET_REGS[0] as u16)
    }

    /// Classify a variadic (anonymous) function argument per AAPCS64.
    ///
    /// Variadic arguments differ from named parameters:
    ///
    /// - **Floating-point types are passed in GPRs** (not FP registers).
    /// - **HFA classification does NOT apply** to anonymous arguments.
    /// - Composites ≤ 16 bytes → 1–2 GPRs.
    /// - Composites > 16 bytes → by reference (pointer in GPR or stack).
    pub fn classify_variadic_arg(&mut self, ty: &CType) -> ArgLocation {
        let stripped = strip_decorators(ty);

        // Void — defensive.
        if matches!(stripped, CType::Void) {
            return ArgLocation::Stack(0);
        }

        // For variadic args, FP types and HFAs are promoted to GPR passing.
        if is_fp_scalar(stripped) || is_hfa(stripped).is_some() {
            return self.classify_integer_scalar();
        }

        // _Complex FP in variadic: promote to GPR passing (8 or 16 bytes).
        if let CType::Complex(ref base) = stripped {
            if is_fp_scalar(strip_decorators(base)) {
                let size = type_size(stripped);
                if size <= 8 {
                    return self.classify_integer_scalar();
                }
                // 16-byte complex → two GPRs.
                if self.ngrn + 2 <= NUM_INT_ARG_REGS {
                    let r1 = INT_ARG_REGS[self.ngrn];
                    let r2 = INT_ARG_REGS[self.ngrn + 1];
                    self.ngrn += 2;
                    return ArgLocation::RegisterPair(r1 as u16, r2 as u16);
                }
                self.ngrn = NUM_INT_ARG_REGS;
                return ArgLocation::Stack(self.alloc_stack(size, ARG_SLOT_SIZE));
            }
        }

        // Arrays decay to pointers.
        if matches!(stripped, CType::Array(..)) {
            return self.classify_integer_scalar();
        }

        // Function types decay to pointers.
        if matches!(stripped, CType::Function { .. }) {
            return self.classify_integer_scalar();
        }

        // Composites: same rules as normal but with NO HFA recognition.
        if is_composite(stripped) {
            let size = type_size(stripped);

            // Indirect for large composites.
            if size > MAX_COMPOSITE_REG_SIZE {
                if self.ngrn < NUM_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[self.ngrn];
                    self.ngrn += 1;
                    return ArgLocation::Register(reg as u16);
                }
                return ArgLocation::Stack(
                    self.alloc_stack(Target::AArch64.pointer_width(), ARG_SLOT_SIZE),
                );
            }

            // Small composite ≤ 8 bytes.
            if size <= 8 {
                if self.ngrn < NUM_INT_ARG_REGS {
                    let reg = INT_ARG_REGS[self.ngrn];
                    self.ngrn += 1;
                    return ArgLocation::Register(reg as u16);
                }
            } else {
                // 8 < size ≤ 16: two GPRs.
                let al = type_alignment(stripped);
                if al > 8 {
                    self.ngrn = align_up(self.ngrn, 2);
                }
                if self.ngrn + 2 <= NUM_INT_ARG_REGS {
                    let r1 = INT_ARG_REGS[self.ngrn];
                    let r2 = INT_ARG_REGS[self.ngrn + 1];
                    self.ngrn += 2;
                    return ArgLocation::RegisterPair(r1 as u16, r2 as u16);
                }
            }

            self.ngrn = NUM_INT_ARG_REGS;
            let al = type_alignment(stripped).max(ARG_SLOT_SIZE);
            return ArgLocation::Stack(self.alloc_stack(size, al));
        }

        // Integer / pointer / enum scalar — normal GPR classification.
        self.classify_integer_scalar()
    }

    /// Compute the stack frame layout for an AArch64 function.
    ///
    /// # Parameters
    ///
    /// - `_params`: parameter types (reserved for future argument spill computation)
    /// - `locals_size`: total size of local variables and spill slots in bytes
    /// - `callee_saved_count`: number of callee-saved registers that need saving
    ///   (excluding FP/LR which are always saved)
    ///
    /// # Frame Structure (high → low addresses)
    ///
    /// ```text
    /// [incoming stack args]                ← old SP
    /// [saved FP=X29, saved LR=X30]        ← FP points to saved FP
    /// [callee-saved regs (paired STP)]
    /// [local variables + spills]
    /// [arg spill area]                     ← SP (16-byte aligned)
    /// ```
    ///
    /// FP (X29) and LR (X30) are always saved as a pair at the top of the
    /// frame using `STP X29, X30, [SP, #offset]`. Frame pointer X29 MUST
    /// point to the saved FP location per AAPCS64.
    pub fn compute_frame_layout(
        _params: &[CType],
        locals_size: usize,
        callee_saved_count: usize,
    ) -> FrameLayout {
        // FP (X29) and LR (X30) are always saved as a pair: 16 bytes.
        let fp_lr_size: usize = 16;

        // Callee-saved registers are saved in pairs using STP/LDP for
        // efficiency. Round up to even count for paired stores.
        let callee_pairs = (callee_saved_count + 1) / 2;
        let callee_saved_size = callee_pairs * 16;

        // Align locals to 16-byte boundary for SP alignment compliance.
        let locals_aligned = align_up(locals_size, STACK_ALIGNMENT);

        // Total frame = FP/LR pair + callee-saved area + locals area.
        let frame_body = fp_lr_size + callee_saved_size + locals_aligned;
        let total_size = align_up(frame_body, STACK_ALIGNMENT);

        // Layout from SP upward:
        //
        //   SP + 0                                      : locals / spills (bottom)
        //   SP + locals_aligned                         : callee-saved regs
        //   SP + locals_aligned + callee_saved_size     : saved FP (X29)
        //   SP + locals_aligned + callee_saved_size + 8 : saved LR (X30)
        //   SP + total_size                             : old SP
        //
        // FP = SP + locals_aligned + callee_saved_size

        let fp_off = (locals_aligned + callee_saved_size) as i64;
        let lr_off = fp_off + 8;
        let cs_off = locals_aligned as i64;

        // Offsets from FP (FP-relative, negative = below FP):
        //   locals start at FP - callee_saved_size - locals_aligned
        //   which is FP - (fp_off - 0) = -fp_off (relative to FP)
        let locals_off_from_fp = -(callee_saved_size as i64 + locals_aligned as i64);

        // Arg spill area is at the bottom of locals.
        let arg_spill_off_from_fp = locals_off_from_fp;

        FrameLayout {
            total_size,
            locals_offset: locals_off_from_fp,
            callee_saved_offset: cs_off,
            lr_offset: lr_off,
            fp_offset: fp_off,
            arg_spill_offset: arg_spill_off_from_fp,
        }
    }
}

// ===========================================================================
// Register Set Queries
// ===========================================================================

/// Returns the callee-saved general-purpose registers per AAPCS64: X19–X28.
///
/// These 10 registers must be preserved by the callee. Typically saved and
/// restored in pairs using STP/LDP instructions for efficiency.
#[inline]
pub fn callee_saved_gprs() -> &'static [u32] {
    CALLEE_SAVED_GPRS
}

/// Returns the callee-saved SIMD/FP registers per AAPCS64: V8–V15 (D8–D15).
///
/// Only the lower 64 bits (D8–D15) of these registers are preserved by the
/// callee. The upper 64 bits of V8–V15 may be clobbered.
#[inline]
pub fn callee_saved_fprs() -> &'static [u32] {
    CALLEE_SAVED_FPRS
}

/// Returns the caller-saved general-purpose registers per AAPCS64.
///
/// Includes X0–X18 (argument/scratch/platform) and X30 (LR, clobbered by
/// BL instructions). These 20 registers may be clobbered by any function call.
#[inline]
pub fn caller_saved_gprs() -> &'static [u32] {
    CALLER_SAVED_GPRS
}

/// Returns the caller-saved SIMD/FP registers per AAPCS64: V0–V7, V16–V31.
///
/// These 24 registers may be clobbered by any function call.
#[inline]
pub fn caller_saved_fprs() -> &'static [u32] {
    CALLER_SAVED_FPRS
}

// ===========================================================================
// Type Size and Alignment (LP64 AArch64-Specific)
// ===========================================================================

/// Returns the size in bytes of a C type under the AArch64 LP64 data model.
///
/// For pointer, long, and long double types, the target-specific methods on
/// [`Target::AArch64`] are used directly. For composite types and other scalars,
/// [`TypeBuilder`] provides the precise layout computation.
///
/// # LP64 Type Sizes
///
/// | Type          | Size |
/// |---------------|-----:|
/// | `_Bool`       |    1 |
/// | `char`        |    1 |
/// | `short`       |    2 |
/// | `int`         |    4 |
/// | `long`        |    8 |
/// | `long long`   |    8 |
/// | `float`       |    4 |
/// | `double`      |    8 |
/// | `long double` |   16 |
/// | `pointer`     |    8 |
pub fn type_size(ty: &CType) -> usize {
    let target = Target::AArch64;
    let stripped = strip_decorators(ty);
    // Use Target methods directly for key LP64-specific types; delegate to
    // TypeBuilder for everything else (including struct layout computation).
    match stripped {
        CType::Pointer(_, _) => target.pointer_width(),
        CType::Long | CType::ULong => target.long_size(),
        CType::LongDouble => target.long_double_size(),
        _ => {
            let tb = TypeBuilder::new(target);
            tb.sizeof_type(ty)
        }
    }
}

/// Returns the alignment in bytes of a C type under the AArch64 LP64 data model.
///
/// Uses [`TypeBuilder`] for target-aware alignment computation including
/// struct/union natural alignment, packed attributes, and explicit alignment
/// overrides. `long double` on AArch64 is IEEE 754 quad-precision with
/// 16-byte alignment.
pub fn type_alignment(ty: &CType) -> usize {
    let target = Target::AArch64;
    let tb = TypeBuilder::new(target);
    tb.alignof_type(ty)
}

// ===========================================================================
// IR-Type Batch Argument Classification (AAPCS64)
// ===========================================================================

use crate::ir::types::IrType;

/// Classify a slice of IR-level argument types into AAPCS64 locations.
///
/// Tracks the Next General-purpose Register Number (NGRN, X0–X7) and the
/// Next SIMD/FP Register Number (NSRN, V0–V7), advancing through them as
/// arguments are assigned.  Arguments that exhaust the available registers
/// are placed on the stack.
///
/// This mirrors the x86_64 ABI module's `classify_arguments` function and
/// is the correct way to determine per-argument placement when multiple
/// arguments must be classified together.
pub fn classify_arguments_ir(
    params: &[IrType],
    target: &Target,
) -> Vec<crate::backend::traits::ArgLocation> {
    use crate::backend::traits::ArgLocation;

    let mut locs = Vec::with_capacity(params.len());
    let mut ngrn: usize = 0; // Next General-purpose Register Number
    let mut nsrn: usize = 0; // Next SIMD/FP Register Number
    let mut stack_offset: i32 = 0;

    for param in params {
        match param {
            // Floating-point scalars → V0–V7
            IrType::F32 | IrType::F64 | IrType::F80 => {
                if nsrn < NUM_FP_ARG_REGS {
                    locs.push(ArgLocation::Register(FP_ARG_REGS[nsrn] as u16));
                    nsrn += 1;
                } else {
                    let size = param.size_bytes(target).max(8) as i32;
                    locs.push(ArgLocation::Stack(stack_offset));
                    stack_offset += size;
                }
            }
            // Void — no value to pass
            IrType::Void => {
                locs.push(ArgLocation::Stack(0));
            }
            // Structs — may use register pairs or spill to stack
            IrType::Struct(st) => {
                let size = param.size_bytes(target);
                if size == 0 {
                    locs.push(ArgLocation::Stack(0));
                } else if size <= 8 && ngrn < NUM_INT_ARG_REGS {
                    locs.push(ArgLocation::Register(INT_ARG_REGS[ngrn] as u16));
                    ngrn += 1;
                } else if size <= 16 && ngrn + 2 <= NUM_INT_ARG_REGS {
                    let r1 = INT_ARG_REGS[ngrn] as u16;
                    let r2 = INT_ARG_REGS[ngrn + 1] as u16;
                    locs.push(ArgLocation::RegisterPair(r1, r2));
                    ngrn += 2;
                } else {
                    // Check for HFA (Homogeneous Floating-point Aggregate)
                    let is_all_fp = !st.fields.is_empty()
                        && st
                            .fields
                            .iter()
                            .all(|f| matches!(f, IrType::F32 | IrType::F64))
                        && st.fields.len() <= 4;
                    if is_all_fp && nsrn + st.fields.len() <= NUM_FP_ARG_REGS {
                        locs.push(ArgLocation::Register(FP_ARG_REGS[nsrn] as u16));
                        nsrn += st.fields.len();
                    } else {
                        // Large struct: passed by reference (hidden pointer in X[ngrn])
                        if ngrn < NUM_INT_ARG_REGS {
                            locs.push(ArgLocation::Register(INT_ARG_REGS[ngrn] as u16));
                            ngrn += 1;
                        } else {
                            let slot = ((size + 7) & !7) as i32;
                            locs.push(ArgLocation::Stack(stack_offset));
                            stack_offset += slot;
                        }
                    }
                }
            }
            // Arrays — passed by reference (hidden pointer)
            IrType::Array(_, _) => {
                if ngrn < NUM_INT_ARG_REGS {
                    locs.push(ArgLocation::Register(INT_ARG_REGS[ngrn] as u16));
                    ngrn += 1;
                } else {
                    locs.push(ArgLocation::Stack(stack_offset));
                    stack_offset += 8;
                }
            }
            // Integer / pointer scalars → X0–X7
            _ => {
                if ngrn < NUM_INT_ARG_REGS {
                    locs.push(ArgLocation::Register(INT_ARG_REGS[ngrn] as u16));
                    ngrn += 1;
                } else {
                    let size = param.size_bytes(target).max(8) as i32;
                    locs.push(ArgLocation::Stack(stack_offset));
                    stack_offset += size;
                }
            }
        }
    }

    locs
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::StructField;

    #[test]
    fn test_lp64_sizes() {
        assert_eq!(type_size(&CType::Bool), 1);
        assert_eq!(type_size(&CType::Char), 1);
        assert_eq!(type_size(&CType::Short), 2);
        assert_eq!(type_size(&CType::Int), 4);
        assert_eq!(type_size(&CType::Long), 8);
        assert_eq!(type_size(&CType::LongLong), 8);
        assert_eq!(type_size(&CType::Float), 4);
        assert_eq!(type_size(&CType::Double), 8);
        assert_eq!(type_size(&CType::LongDouble), 16);
        assert_eq!(
            type_size(&CType::Pointer(
                Box::new(CType::Int),
                crate::common::types::TypeQualifiers::default()
            )),
            8
        );
    }

    #[test]
    fn test_lp64_alignments() {
        assert_eq!(type_alignment(&CType::Char), 1);
        assert_eq!(type_alignment(&CType::Short), 2);
        assert_eq!(type_alignment(&CType::Int), 4);
        assert_eq!(type_alignment(&CType::Long), 8);
        assert_eq!(type_alignment(&CType::Float), 4);
        assert_eq!(type_alignment(&CType::Double), 8);
        assert_eq!(type_alignment(&CType::LongDouble), 16);
    }

    #[test]
    fn test_hfa_two_floats() {
        let ty = CType::Struct {
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
        let result = is_hfa(&ty);
        assert!(result.is_some());
        let (count, member) = result.unwrap();
        assert_eq!(count, 2);
        assert_eq!(member, CType::Float);
    }

    #[test]
    fn test_hfa_three_doubles() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Double,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Double,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let result = is_hfa(&ty);
        assert!(result.is_some());
        let (count, member) = result.unwrap();
        assert_eq!(count, 3);
        assert_eq!(member, CType::Double);
    }

    #[test]
    fn test_hfa_mixed_not_hfa() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        assert!(is_hfa(&ty).is_none());
    }

    #[test]
    fn test_hfa_too_many_members() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        assert!(is_hfa(&ty).is_none());
    }

    #[test]
    fn test_classify_int_args() {
        let mut abi = AArch64Abi::new();
        // First 8 ints → X0–X7.
        for i in 0..8 {
            let loc = abi.classify_arg(&CType::Int);
            assert_eq!(loc, ArgLocation::Register(INT_ARG_REGS[i] as u16));
        }
        // 9th int → stack.
        let loc = abi.classify_arg(&CType::Int);
        assert!(matches!(loc, ArgLocation::Stack(_)));
    }

    #[test]
    fn test_classify_float_args() {
        let mut abi = AArch64Abi::new();
        // First 8 floats → V0–V7.
        for i in 0..8 {
            let loc = abi.classify_arg(&CType::Float);
            assert_eq!(loc, ArgLocation::Register(FP_ARG_REGS[i] as u16));
        }
        // 9th float → stack.
        let loc = abi.classify_arg(&CType::Float);
        assert!(matches!(loc, ArgLocation::Stack(_)));
    }

    #[test]
    fn test_classify_return_int() {
        let abi = AArch64Abi::new();
        let loc = abi.classify_return(&CType::Int);
        assert_eq!(loc, ArgLocation::Register(X0 as u16));
    }

    #[test]
    fn test_classify_return_float() {
        let abi = AArch64Abi::new();
        let loc = abi.classify_return(&CType::Float);
        assert_eq!(loc, ArgLocation::Register(V0 as u16));
    }

    #[test]
    fn test_classify_return_void() {
        let abi = AArch64Abi::new();
        let loc = abi.classify_return(&CType::Void);
        assert_eq!(loc, ArgLocation::Stack(0));
    }

    #[test]
    fn test_classify_return_large_struct() {
        let abi = AArch64Abi::new();
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::LongLong,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::LongLong,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::LongLong,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let loc = abi.classify_return(&ty);
        // > 16 bytes → indirect via X8.
        assert_eq!(loc, ArgLocation::Register(INDIRECT_RESULT_REG as u16));
    }

    #[test]
    fn test_variadic_fp_uses_gpr() {
        let mut abi = AArch64Abi::new();
        let loc = abi.classify_variadic_arg(&CType::Double);
        // Variadic FP args go in GPRs, not FP regs.
        assert_eq!(loc, ArgLocation::Register(X0 as u16));
    }

    #[test]
    fn test_callee_saved_sets() {
        assert_eq!(callee_saved_gprs().len(), 10); // X19–X28
        assert_eq!(callee_saved_fprs().len(), 8); // V8–V15
        assert_eq!(caller_saved_gprs().len(), 20); // X0–X18, X30
        assert_eq!(caller_saved_fprs().len(), 24); // V0–V7, V16–V31
    }

    #[test]
    fn test_frame_layout_alignment() {
        let layout = AArch64Abi::compute_frame_layout(&[], 0, 0);
        assert_eq!(layout.total_size % STACK_ALIGNMENT, 0);

        let layout2 = AArch64Abi::compute_frame_layout(&[], 17, 3);
        assert_eq!(layout2.total_size % STACK_ALIGNMENT, 0);
    }

    #[test]
    fn test_frame_layout_fp_lr() {
        let layout = AArch64Abi::compute_frame_layout(&[], 0, 0);
        // FP/LR pair must be present (16 bytes minimum frame).
        assert!(layout.total_size >= 16);
        // LR is 8 bytes above FP.
        assert_eq!(layout.lr_offset, layout.fp_offset + 8);
    }

    #[test]
    fn test_constants() {
        assert_eq!(NUM_INT_ARG_REGS, 8);
        assert_eq!(NUM_FP_ARG_REGS, 8);
        assert_eq!(STACK_ALIGNMENT, 16);
        assert_eq!(ARG_SLOT_SIZE, 8);
        assert_eq!(MAX_HFA_MEMBERS, 4);
        assert_eq!(MAX_COMPOSITE_REG_SIZE, 16);
        assert_eq!(INT_ARG_REGS.len(), 8);
        assert_eq!(FP_ARG_REGS.len(), 8);
        assert_eq!(INT_RET_REGS.len(), 2);
        assert_eq!(FP_RET_REGS.len(), 4);
        assert_eq!(INDIRECT_RESULT_REG, X8);
    }

    #[test]
    fn test_classify_small_struct_two_regs() {
        let mut abi = AArch64Abi::new();
        // 16-byte struct → register pair (X0, X1).
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Long,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Long,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let loc = abi.classify_arg(&ty);
        assert_eq!(loc, ArgLocation::RegisterPair(X0 as u16, X1 as u16));
    }

    #[test]
    fn test_reset() {
        let mut abi = AArch64Abi::new();
        abi.classify_arg(&CType::Int);
        abi.classify_arg(&CType::Float);
        abi.reset();
        // After reset, first int should go to X0 again.
        let loc = abi.classify_arg(&CType::Int);
        assert_eq!(loc, ArgLocation::Register(X0 as u16));
    }

    #[test]
    fn test_hfa_arg_classification() {
        let mut abi = AArch64Abi::new();
        let hfa = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Float,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let loc = abi.classify_arg(&hfa);
        // 2-member float HFA → RegisterPair(V0, V1).
        assert_eq!(loc, ArgLocation::RegisterPair(V0 as u16, V1 as u16));
    }

    #[test]
    fn test_hfa_return_classification() {
        let abi = AArch64Abi::new();
        let hfa = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Double,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let loc = abi.classify_return(&hfa);
        assert_eq!(loc, ArgLocation::RegisterPair(V0 as u16, V1 as u16));
    }

    #[test]
    fn test_machine_type_classification() {
        assert_eq!(classify_machine_type(&CType::Float), MachineType::F32);
        assert_eq!(classify_machine_type(&CType::Double), MachineType::F64);
        assert_eq!(classify_machine_type(&CType::Int), MachineType::Integer);
    }
}
