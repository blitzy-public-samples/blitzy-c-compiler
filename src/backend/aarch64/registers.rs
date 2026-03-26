//! # AArch64 Register Definitions
//!
//! Defines the complete AArch64 register file:
//! - 31 general-purpose registers: X0–X30 (64-bit) / W0–W30 (32-bit)
//! - Stack pointer: SP (register 31 in some contexts)
//! - Zero register: XZR/WZR (register 31 in other contexts)
//! - 32 SIMD/FP registers: V0–V31 (128-bit), D0–D31 (64-bit), S0–S31 (32-bit),
//!   H0–H31 (16-bit), B0–B31 (8-bit)
//! - NZCV condition flags register
//!
//! ## AAPCS64 Register Roles
//!
//! | Register(s) | Role | Saved By |
//! |-------------|------|----------|
//! | X0–X7       | Arguments / Return values | Caller |
//! | X8          | Indirect result location (struct return ptr) | Caller |
//! | X9–X15      | Temporary (scratch) registers | Caller |
//! | X16 (IP0)   | Intra-procedure-call scratch | Caller |
//! | X17 (IP1)   | Intra-procedure-call scratch | Caller |
//! | X18 (PR)    | Platform register (reserved on some OSes) | Caller |
//! | X19–X28     | Callee-saved registers | Callee |
//! | X29 (FP)    | Frame pointer | Callee |
//! | X30 (LR)    | Link register (return address) | Caller |
//! | SP          | Stack pointer | Special |
//! | XZR         | Zero register (reads as 0, writes discarded) | N/A |
//!
//! ## SIMD/FP Register Roles (AAPCS64)
//!
//! | Register(s) | Role | Saved By |
//! |-------------|------|----------|
//! | V0–V7       | FP/SIMD arguments and return values | Caller |
//! | V8–V15      | Callee-saved (lower 64 bits D8–D15 only) | Callee |
//! | V16–V31     | Temporary (scratch) registers | Caller |
//!
//! ## Note: SP and XZR encoding
//!
//! SP and XZR both use hardware encoding 31, but the distinction is made
//! by the instruction encoding (some instructions use register 31 as SP,
//! others as XZR). Our internal representation uses `SP_REG = 31` and
//! `XZR = 31` with the instruction context determining interpretation.

use crate::backend::traits::RegisterInfo;

// ===========================================================================
// General-Purpose Register ID Constants (X0–X30, SP/ZR)
// ===========================================================================

/// General-purpose register X0 (64-bit) / W0 (32-bit) — argument/return register 0.
pub const X0: u32 = 0;
/// General-purpose register X1 (64-bit) / W1 (32-bit) — argument/return register 1.
pub const X1: u32 = 1;
/// General-purpose register X2 (64-bit) / W2 (32-bit) — argument/return register 2.
pub const X2: u32 = 2;
/// General-purpose register X3 (64-bit) / W3 (32-bit) — argument/return register 3.
pub const X3: u32 = 3;
/// General-purpose register X4 (64-bit) / W4 (32-bit) — argument register 4.
pub const X4: u32 = 4;
/// General-purpose register X5 (64-bit) / W5 (32-bit) — argument register 5.
pub const X5: u32 = 5;
/// General-purpose register X6 (64-bit) / W6 (32-bit) — argument register 6.
pub const X6: u32 = 6;
/// General-purpose register X7 (64-bit) / W7 (32-bit) — argument register 7.
pub const X7: u32 = 7;
/// General-purpose register X8 (64-bit) / W8 (32-bit) — indirect result location.
pub const X8: u32 = 8;
/// General-purpose register X9 (64-bit) / W9 (32-bit) — temporary/scratch.
pub const X9: u32 = 9;
/// General-purpose register X10 (64-bit) / W10 (32-bit) — temporary/scratch.
pub const X10: u32 = 10;
/// General-purpose register X11 (64-bit) / W11 (32-bit) — temporary/scratch.
pub const X11: u32 = 11;
/// General-purpose register X12 (64-bit) / W12 (32-bit) — temporary/scratch.
pub const X12: u32 = 12;
/// General-purpose register X13 (64-bit) / W13 (32-bit) — temporary/scratch.
pub const X13: u32 = 13;
/// General-purpose register X14 (64-bit) / W14 (32-bit) — temporary/scratch.
pub const X14: u32 = 14;
/// General-purpose register X15 (64-bit) / W15 (32-bit) — temporary/scratch.
pub const X15: u32 = 15;
/// General-purpose register X16 (64-bit) / W16 (32-bit) — IP0 intra-procedure-call scratch.
pub const X16: u32 = 16;
/// General-purpose register X17 (64-bit) / W17 (32-bit) — IP1 intra-procedure-call scratch.
pub const X17: u32 = 17;
/// General-purpose register X18 (64-bit) / W18 (32-bit) — platform register.
pub const X18: u32 = 18;
/// General-purpose register X19 (64-bit) / W19 (32-bit) — callee-saved.
pub const X19: u32 = 19;
/// General-purpose register X20 (64-bit) / W20 (32-bit) — callee-saved.
pub const X20: u32 = 20;
/// General-purpose register X21 (64-bit) / W21 (32-bit) — callee-saved.
pub const X21: u32 = 21;
/// General-purpose register X22 (64-bit) / W22 (32-bit) — callee-saved.
pub const X22: u32 = 22;
/// General-purpose register X23 (64-bit) / W23 (32-bit) — callee-saved.
pub const X23: u32 = 23;
/// General-purpose register X24 (64-bit) / W24 (32-bit) — callee-saved.
pub const X24: u32 = 24;
/// General-purpose register X25 (64-bit) / W25 (32-bit) — callee-saved.
pub const X25: u32 = 25;
/// General-purpose register X26 (64-bit) / W26 (32-bit) — callee-saved.
pub const X26: u32 = 26;
/// General-purpose register X27 (64-bit) / W27 (32-bit) — callee-saved.
pub const X27: u32 = 27;
/// General-purpose register X28 (64-bit) / W28 (32-bit) — callee-saved.
pub const X28: u32 = 28;
/// General-purpose register X29 (64-bit) / W29 (32-bit) — frame pointer (FP).
pub const X29: u32 = 29;
/// General-purpose register X30 (64-bit) / W30 (32-bit) — link register (LR).
pub const X30: u32 = 30;

/// Stack pointer register. Shares hardware encoding 31 with XZR, but
/// the distinction is made by instruction context. SP-using instructions
/// (ADD SP, SUB SP, LDR [SP, #imm], STR [SP, #imm]) interpret register 31
/// as the stack pointer.
pub const SP_REG: u32 = 31;

/// Zero register (reads as zero, writes are discarded). Shares hardware
/// encoding 31 with SP. Non-SP instructions (most ALU, logical, move)
/// interpret register 31 as the zero register.
pub const XZR: u32 = 31;

// ===========================================================================
// ABI Name Aliases for Special Registers
// ===========================================================================

/// Frame pointer — alias for X29 per AAPCS64. The frame pointer points to
/// the saved FP location on the stack, forming a linked list of stack frames
/// for unwinding and debugging.
pub const FP_REG: u32 = X29;

/// Link register — alias for X30. Stores the return address set by BL/BLR
/// instructions. The RET instruction defaults to branching to the address in LR.
pub const LR: u32 = X30;

/// Intra-procedure-call scratch register 0 — alias for X16. Used by the
/// linker for veneers and PLT stubs. May be clobbered between a function
/// call and its entry into the callee.
pub const IP0: u32 = X16;

/// Intra-procedure-call scratch register 1 — alias for X17. Used by the
/// linker together with IP0 for longer-range veneers.
pub const IP1: u32 = X17;

/// Platform register — alias for X18. Reserved on some operating systems
/// (e.g., macOS, Windows) but usable on Linux. Treated as caller-saved.
pub const PR: u32 = X18;

// ===========================================================================
// SIMD/FP Register ID Constants (V0–V31)
// ===========================================================================
// Encoded as 32–63 to create a unique register ID namespace separate from
// GPRs. The hardware encoding is obtained by subtracting 32 (i.e., V0 maps
// to hardware register 0, V31 maps to hardware register 31).

/// SIMD/FP register V0 (128-bit) — FP argument/return register 0.
pub const V0: u32 = 32;
/// SIMD/FP register V1 (128-bit) — FP argument/return register 1.
pub const V1: u32 = 33;
/// SIMD/FP register V2 (128-bit) — FP argument/return register 2.
pub const V2: u32 = 34;
/// SIMD/FP register V3 (128-bit) — FP argument/return register 3.
pub const V3: u32 = 35;
/// SIMD/FP register V4 (128-bit) — FP argument register 4.
pub const V4: u32 = 36;
/// SIMD/FP register V5 (128-bit) — FP argument register 5.
pub const V5: u32 = 37;
/// SIMD/FP register V6 (128-bit) — FP argument register 6.
pub const V6: u32 = 38;
/// SIMD/FP register V7 (128-bit) — FP argument register 7.
pub const V7: u32 = 39;
/// SIMD/FP register V8 (128-bit) — callee-saved (lower 64 bits D8 only).
pub const V8: u32 = 40;
/// SIMD/FP register V9 (128-bit) — callee-saved (lower 64 bits D9 only).
pub const V9: u32 = 41;
/// SIMD/FP register V10 (128-bit) — callee-saved (lower 64 bits D10 only).
pub const V10: u32 = 42;
/// SIMD/FP register V11 (128-bit) — callee-saved (lower 64 bits D11 only).
pub const V11: u32 = 43;
/// SIMD/FP register V12 (128-bit) — callee-saved (lower 64 bits D12 only).
pub const V12: u32 = 44;
/// SIMD/FP register V13 (128-bit) — callee-saved (lower 64 bits D13 only).
pub const V13: u32 = 45;
/// SIMD/FP register V14 (128-bit) — callee-saved (lower 64 bits D14 only).
pub const V14: u32 = 46;
/// SIMD/FP register V15 (128-bit) — callee-saved (lower 64 bits D15 only).
pub const V15: u32 = 47;
/// SIMD/FP register V16 (128-bit) — temporary/scratch.
pub const V16: u32 = 48;
/// SIMD/FP register V17 (128-bit) — temporary/scratch.
pub const V17: u32 = 49;
/// SIMD/FP register V18 (128-bit) — temporary/scratch.
pub const V18: u32 = 50;
/// SIMD/FP register V19 (128-bit) — temporary/scratch.
pub const V19: u32 = 51;
/// SIMD/FP register V20 (128-bit) — temporary/scratch.
pub const V20: u32 = 52;
/// SIMD/FP register V21 (128-bit) — temporary/scratch.
pub const V21: u32 = 53;
/// SIMD/FP register V22 (128-bit) — temporary/scratch.
pub const V22: u32 = 54;
/// SIMD/FP register V23 (128-bit) — temporary/scratch.
pub const V23: u32 = 55;
/// SIMD/FP register V24 (128-bit) — temporary/scratch.
pub const V24: u32 = 56;
/// SIMD/FP register V25 (128-bit) — temporary/scratch.
pub const V25: u32 = 57;
/// SIMD/FP register V26 (128-bit) — temporary/scratch.
pub const V26: u32 = 58;
/// SIMD/FP register V27 (128-bit) — temporary/scratch.
pub const V27: u32 = 59;
/// SIMD/FP register V28 (128-bit) — temporary/scratch.
pub const V28: u32 = 60;
/// SIMD/FP register V29 (128-bit) — temporary/scratch.
pub const V29: u32 = 61;
/// SIMD/FP register V30 (128-bit) — temporary/scratch.
pub const V30: u32 = 62;
/// SIMD/FP register V31 (128-bit) — temporary/scratch.
pub const V31: u32 = 63;

// ===========================================================================
// Register Class Definitions
// ===========================================================================

/// Register classes for AArch64, used by the register allocator to
/// differentiate between general-purpose (integer/pointer) and
/// floating-point/SIMD register files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    /// General-purpose registers (X0–X30 / W0–W30).
    /// Used for integer values, pointers, and address computations.
    GPR,
    /// SIMD/Floating-point registers (V0–V31 / D0–D31 / S0–S31).
    /// Used for `float`, `double`, SIMD values, and `long double` on AArch64.
    FPR,
}

// ===========================================================================
// Register Count Constants
// ===========================================================================

/// Total number of GPR encodings (X0–X30 plus SP/ZR encoding = 32).
pub const NUM_GPRS: usize = 32;

/// Total number of SIMD/FP registers (V0–V31 = 32).
pub const NUM_FPRS: usize = 32;

/// Total register count across both classes (GPR + FPR = 64).
pub const TOTAL_REGS: usize = NUM_GPRS + NUM_FPRS;

/// Number of allocatable GPRs: X0–X15, X18–X28 (27 registers).
/// Excludes X16 (IP0/spill scratch), X17 (IP1/second spill scratch),
/// X29 (FP), X30 (LR), and SP/XZR (31).
pub const NUM_ALLOCATABLE_GPRS: usize = 27;

// ===========================================================================
// Register Sets for the Register Allocator
// ===========================================================================

/// Allocatable integer registers — excludes SP (31), X29 (FP), X30 (LR).
///
/// Ordered with caller-saved registers first (preferred for short-lived
/// temporaries) followed by callee-saved registers. This ordering allows
/// the register allocator to prefer registers that do not require
/// save/restore in the prologue/epilogue.
pub const ALLOCATABLE_GPRS: &[u32] = &[
    // Caller-saved temporaries (preferred for allocation — no save overhead)
    X0, X1, X2, X3, X4, X5, X6, X7, // argument registers, scratch after use
    X8, // indirect result location register
    X9, X10, X11, X12, X13, X14, X15, // scratch temporaries
    // X16 (IP0) is RESERVED as the primary spill-reload scratch register
    // and codegen scratch — never allocatable.
    // X17 (IP1) is RESERVED as the secondary spill-reload scratch register.
    // On load-store architectures like AArch64, ALU instructions cannot use
    // Memory operands. When two operands are both spilled, X16 loads the
    // first and X17 loads the second.
    X18, // platform register (usable on Linux)
    // Callee-saved (allocator will prefer caller-saved first)
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Allocatable SIMD/FP registers — all V0–V31 are allocatable.
///
/// Ordered with caller-saved registers first (preferred for short-lived
/// temporaries) followed by callee-saved registers (V8–V15, of which
/// only the lower 64 bits D8–D15 are preserved by the callee).
pub const ALLOCATABLE_FPRS: &[u32] = &[
    // Caller-saved (argument + scratch — no save overhead)
    V0, V1, V2, V3, V4, V5, V6, V7, // FP argument/return registers
    V16, V17, V18, V19, V20, V21, V22, V23, // scratch temporaries
    V24, V25, V26, V27, V28, V29, V30, V31, // scratch temporaries
    // Callee-saved (lower 64 bits D8–D15 preserved by callee)
    V8, V9, V10, V11, V12, V13, V14, V15,
];

// ===========================================================================
// Callee-Saved and Caller-Saved Register Sets
// ===========================================================================

/// Callee-saved GPRs per AAPCS64: X19–X28 (10 registers).
///
/// These registers must be preserved across function calls. If a function
/// uses any of these registers, it must save them in the prologue and
/// restore them in the epilogue.
pub const CALLEE_SAVED_GPRS: &[u32] = &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];

/// Callee-saved FP registers per AAPCS64: V8–V15 (8 registers).
///
/// Only the lower 64 bits (D8–D15) are preserved by the callee.
/// The upper 64 bits of V8–V15 are NOT preserved across calls.
pub const CALLEE_SAVED_FPRS: &[u32] = &[V8, V9, V10, V11, V12, V13, V14, V15];

/// Caller-saved GPRs per AAPCS64: X0–X18, X30 (LR) — 20 registers.
///
/// These registers may be clobbered by any function call. The caller
/// must save any live values in these registers before a call and
/// restore them afterward if needed.
///
/// Note: X29 (FP) is callee-saved but handled specially by the
/// prologue/epilogue, so it does not appear in either set for allocation.
/// X30 (LR) is caller-saved because the callee may use BL which
/// overwrites LR.
pub const CALLER_SAVED_GPRS: &[u32] = &[
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X30,
];

/// Caller-saved FP registers per AAPCS64: V0–V7, V16–V31 — 24 registers.
///
/// These registers may be clobbered by any function call.
pub const CALLER_SAVED_FPRS: &[u32] = &[
    V0, V1, V2, V3, V4, V5, V6, V7, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27,
    V28, V29, V30, V31,
];

/// Reserved registers that are never available for general allocation.
///
/// SP (encoding 31) is the stack pointer and must never be used as a
/// general-purpose register. X29 (FP) and X30 (LR) are handled
/// separately by the prologue/epilogue generator but are excluded from
/// the allocatable set above.  X16 (IP0) is reserved as the spill-
/// reload scratch register used by `apply_allocation_result`.
pub const RESERVED_REGS: &[u32] = &[SP_REG, X16];

// ===========================================================================
// Static Name Tables for Register Name Lookups
// ===========================================================================

/// 64-bit GPR names (X-register view). Index 31 maps to "sp" (stack pointer
/// context).
const GPR_NAMES_X: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "sp",
];

/// 64-bit GPR names with zero register context. Index 31 maps to "xzr"
/// instead of "sp".
const GPR_NAMES_X_ZR: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "xzr",
];

/// SIMD/FP register names (V-register 128-bit view). Indexed 0–31.
const FPR_NAMES_V: [&str; 32] = [
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
    "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
    "v28", "v29", "v30", "v31",
];

/// SIMD/FP register names (D-register 64-bit double view). Indexed 0–31.
const FPR_NAMES_D: [&str; 32] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    "d28", "d29", "d30", "d31",
];

/// SIMD/FP register names (S-register 32-bit float view). Indexed 0–31.
const FPR_NAMES_S: [&str; 32] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31",
];

// ===========================================================================
// Register Name Lookup Functions
// ===========================================================================

/// Returns the 64-bit GPR name for the given register ID (0–31).
///
/// Register 31 is returned as "sp" (stack pointer context). For the zero
/// register context, use [`gpr_name_or_zr`] instead.
///
/// # Arguments
///
/// * `reg` - Register ID in the range 0–31.
///
/// # Returns
///
/// The assembly name (e.g., "x0", "x29", "sp").
///
/// # Panics
///
/// Returns "unknown" for register IDs > 31 (defensive — should not happen
/// in well-formed code).
#[inline]
pub fn gpr_name(reg: u32) -> &'static str {
    if (reg as usize) < GPR_NAMES_X.len() {
        GPR_NAMES_X[reg as usize]
    } else {
        "unknown"
    }
}

/// Returns the 64-bit GPR name for the given register ID (0–31), using
/// "xzr" for register 31 (zero register context).
///
/// Identical to [`gpr_name`] except register 31 maps to "xzr" instead
/// of "sp". Use this for instructions where encoding 31 means the zero
/// register rather than the stack pointer.
///
/// # Arguments
///
/// * `reg` - Register ID in the range 0–31.
///
/// # Returns
///
/// The assembly name (e.g., "x0", "x29", "xzr").
#[inline]
pub fn gpr_name_or_zr(reg: u32) -> &'static str {
    if (reg as usize) < GPR_NAMES_X_ZR.len() {
        GPR_NAMES_X_ZR[reg as usize]
    } else {
        "unknown"
    }
}

/// Returns the 128-bit SIMD/FP V-register name for the given register ID.
///
/// Input is the full internal register ID (32–63). The hardware register
/// number is computed by subtracting 32 (e.g., V0 = 32 maps to "v0").
///
/// # Arguments
///
/// * `reg` - Register ID in the range 32–63 (internal FPR namespace).
///
/// # Returns
///
/// The V-register assembly name (e.g., "v0", "v15", "v31").
#[inline]
pub fn fpr_name(reg: u32) -> &'static str {
    let idx = reg.wrapping_sub(32) as usize;
    if idx < FPR_NAMES_V.len() {
        FPR_NAMES_V[idx]
    } else {
        "unknown"
    }
}

/// Returns the 64-bit D-register (double) name for the given register ID.
///
/// Used when emitting double-precision floating-point instructions.
///
/// # Arguments
///
/// * `reg` - Register ID in the range 32–63 (internal FPR namespace).
///
/// # Returns
///
/// The D-register assembly name (e.g., "d0", "d15", "d31").
#[inline]
pub fn fpr_name_d(reg: u32) -> &'static str {
    let idx = reg.wrapping_sub(32) as usize;
    if idx < FPR_NAMES_D.len() {
        FPR_NAMES_D[idx]
    } else {
        "unknown"
    }
}

/// Returns the 32-bit S-register (single/float) name for the given register ID.
///
/// Used when emitting single-precision floating-point instructions.
///
/// # Arguments
///
/// * `reg` - Register ID in the range 32–63 (internal FPR namespace).
///
/// # Returns
///
/// The S-register assembly name (e.g., "s0", "s15", "s31").
#[inline]
pub fn fpr_name_s(reg: u32) -> &'static str {
    let idx = reg.wrapping_sub(32) as usize;
    if idx < FPR_NAMES_S.len() {
        FPR_NAMES_S[idx]
    } else {
        "unknown"
    }
}

/// Returns the register name for any register ID, dispatching to the
/// appropriate name table based on whether the register is a GPR or FPR.
///
/// - IDs 0–31 → 64-bit GPR name (X-register, with 31 as "sp")
/// - IDs 32–63 → 128-bit FPR name (V-register)
/// - IDs ≥ 64 → "unknown"
///
/// # Arguments
///
/// * `reg` - Register ID (0–63 for valid registers).
///
/// # Returns
///
/// The register assembly name.
#[inline]
pub fn reg_name(reg: u32) -> &'static str {
    if reg < 32 {
        gpr_name(reg)
    } else if reg < 64 {
        fpr_name(reg)
    } else {
        "unknown"
    }
}

// ===========================================================================
// Register Classification Helper Functions
// ===========================================================================

/// Returns `true` if the register ID refers to a general-purpose register
/// (IDs 0–31, including the SP/XZR encoding at 31).
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_gpr(reg: u32) -> bool {
    reg <= 31
}

/// Returns `true` if the register ID refers to a SIMD/floating-point
/// register (IDs 32–63).
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_fpr(reg: u32) -> bool {
    (32..=63).contains(&reg)
}

/// Returns `true` if the register is callee-saved per AAPCS64.
///
/// Checks both the GPR callee-saved set (X19–X28) and the FPR callee-saved
/// set (V8–V15). X29 (FP) is also callee-saved but handled specially.
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_callee_saved(reg: u32) -> bool {
    // GPR callee-saved: X19–X28
    if (X19..=X28).contains(&reg) {
        return true;
    }
    // X29 (FP) is callee-saved but handled specially by prologue/epilogue
    if reg == X29 {
        return true;
    }
    // FPR callee-saved: V8–V15 (IDs 40–47)
    if (V8..=V15).contains(&reg) {
        return true;
    }
    false
}

/// Returns `true` if the register is caller-saved per AAPCS64.
///
/// Checks both the GPR caller-saved set (X0–X18, X30) and the FPR
/// caller-saved set (V0–V7, V16–V31).
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_caller_saved(reg: u32) -> bool {
    // GPR caller-saved: X0–X18, X30 (LR)
    if reg <= X18 || reg == X30 {
        return true;
    }
    // FPR caller-saved: V0–V7 (IDs 32–39) and V16–V31 (IDs 48–63)
    if (V0..=V7).contains(&reg) || (V16..=V31).contains(&reg) {
        return true;
    }
    false
}

/// Returns `true` if the register is reserved and should never be allocated.
///
/// Currently only SP (encoding 31) is strictly reserved. XZR shares the
/// same encoding but is a pseudo-register that cannot hold values. X29 (FP)
/// and X30 (LR) are excluded from the allocatable set but are not in the
/// reserved set because they are managed by the prologue/epilogue.
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_reserved(reg: u32) -> bool {
    // SP is the only unconditionally reserved register
    // Note: SP_REG == XZR == 31, both use the same encoding
    // In practice, encoding 31 in the GPR space is never allocatable
    RESERVED_REGS.contains(&reg)
}

/// Returns `true` if the register can be allocated by the register allocator.
///
/// A register is allocatable if it appears in either [`ALLOCATABLE_GPRS`] or
/// [`ALLOCATABLE_FPRS`]. This excludes SP/XZR (31), X29 (FP), and X30 (LR).
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn is_allocatable(reg: u32) -> bool {
    ALLOCATABLE_GPRS.contains(&reg) || ALLOCATABLE_FPRS.contains(&reg)
}

/// Returns the register class (GPR or FPR) for the given register ID.
///
/// - IDs 0–31 → [`RegClass::GPR`]
/// - IDs 32–63 → [`RegClass::FPR`]
/// - IDs ≥ 64 → defaults to [`RegClass::GPR`] (should not happen in
///   well-formed code)
///
/// # Arguments
///
/// * `reg` - Register ID.
#[inline]
pub fn reg_class(reg: u32) -> RegClass {
    if reg < 32 {
        RegClass::GPR
    } else if reg < 64 {
        RegClass::FPR
    } else {
        // Defensive fallback — should not be reached in well-formed code
        RegClass::GPR
    }
}

/// Returns the 5-bit hardware encoding for the given register ID.
///
/// For GPRs (0–31), the hardware encoding is the register ID itself.
/// For FPRs (32–63), the hardware encoding is the register ID minus 32
/// (i.e., V0 = 32 encodes as hardware register 0).
///
/// The resulting value is always in the range 0–31, fitting in the 5-bit
/// register fields of AArch64 A64 instruction encodings.
///
/// # Arguments
///
/// * `reg` - Register ID (0–63).
///
/// # Returns
///
/// The 5-bit hardware encoding (0–31).
#[inline]
pub fn hw_encoding(reg: u32) -> u8 {
    if reg < 32 {
        reg as u8
    } else {
        (reg - 32) as u8
    }
}

// ===========================================================================
// AArch64RegisterInfo — RegisterInfo Provider
// ===========================================================================

/// AArch64 register information provider.
///
/// Constructs and returns a [`RegisterInfo`] struct populated with the
/// AArch64 register file's allocatable sets, callee/caller-saved sets,
/// reserved registers, and argument/return register assignments per AAPCS64.
///
/// This struct is created by the AArch64 backend module and its
/// [`register_info()`](AArch64RegisterInfo::register_info) method is called
/// by the [`ArchCodegen`](crate::backend::traits::ArchCodegen) trait
/// implementation to provide register metadata to the register allocator.
pub struct AArch64RegisterInfo;

impl Default for AArch64RegisterInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64RegisterInfo {
    /// Create a new AArch64 register information provider.
    #[inline]
    pub fn new() -> Self {
        AArch64RegisterInfo
    }

    /// Build and return a [`RegisterInfo`] struct populated with the
    /// complete AArch64 register allocation metadata.
    ///
    /// The returned `RegisterInfo` contains:
    /// - **allocatable_gpr**: X0–X28 (29 GPRs, caller-saved first)
    /// - **allocatable_fpr**: V0–V31 (32 FPRs, caller-saved first)
    /// - **callee_saved**: X19–X28, V8–V15 (10 GPRs + 8 FPRs = 18 total)
    /// - **caller_saved**: X0–X18, X30, V0–V7, V16–V31 (20 GPRs + 24 FPRs = 44 total)
    /// - **reserved**: SP (encoding 31)
    /// - **argument_gpr**: X0–X7 (8 integer argument registers)
    /// - **argument_fpr**: V0–V7 (8 FP/SIMD argument registers)
    /// - **return_gpr**: X0, X1 (integer return value registers)
    /// - **return_fpr**: V0, V1, V2, V3 (FP return value registers for HFA)
    pub fn register_info(&self) -> RegisterInfo {
        RegisterInfo {
            allocatable_gpr: ALLOCATABLE_GPRS.iter().map(|&r| r as u16).collect(),
            allocatable_fpr: ALLOCATABLE_FPRS.iter().map(|&r| r as u16).collect(),
            callee_saved: {
                let mut saved =
                    Vec::with_capacity(CALLEE_SAVED_GPRS.len() + CALLEE_SAVED_FPRS.len());
                for &r in CALLEE_SAVED_GPRS {
                    saved.push(r as u16);
                }
                for &r in CALLEE_SAVED_FPRS {
                    saved.push(r as u16);
                }
                saved
            },
            caller_saved: {
                let mut saved =
                    Vec::with_capacity(CALLER_SAVED_GPRS.len() + CALLER_SAVED_FPRS.len());
                for &r in CALLER_SAVED_GPRS {
                    saved.push(r as u16);
                }
                for &r in CALLER_SAVED_FPRS {
                    saved.push(r as u16);
                }
                saved
            },
            reserved: RESERVED_REGS.iter().map(|&r| r as u16).collect(),
            argument_gpr: [X0, X1, X2, X3, X4, X5, X6, X7]
                .iter()
                .map(|&r| r as u16)
                .collect(),
            argument_fpr: [V0, V1, V2, V3, V4, V5, V6, V7]
                .iter()
                .map(|&r| r as u16)
                .collect(),
            return_gpr: [X0, X1].iter().map(|&r| r as u16).collect(),
            return_fpr: [V0, V1, V2, V3].iter().map(|&r| r as u16).collect(),
        }
    }
}
