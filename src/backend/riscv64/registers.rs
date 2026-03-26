//! # RISC-V 64 Register Definitions
//!
//! Defines the complete RV64IMAFDC register file:
//! - 32 integer registers: x0 (zero) through x31
//! - 32 floating-point registers: f0 through f31
//! - Control and Status Registers (CSRs) for relevant operations
//!
//! ## ABI Register Names (LP64D)
//! | Register | ABI Name | Role | Saved By |
//! |----------|----------|------|----------|
//! | x0       | zero     | Hard-wired zero | N/A |
//! | x1       | ra       | Return address | Caller |
//! | x2       | sp       | Stack pointer | Callee |
//! | x3       | gp       | Global pointer | — |
//! | x4       | tp       | Thread pointer | — |
//! | x5–x7   | t0–t2    | Temporaries | Caller |
//! | x8       | s0/fp    | Saved reg / Frame pointer | Callee |
//! | x9       | s1       | Saved register | Callee |
//! | x10–x11  | a0–a1   | Args / Return values | Caller |
//! | x12–x17  | a2–a7   | Arguments | Caller |
//! | x18–x27  | s2–s11  | Saved registers | Callee |
//! | x28–x31  | t3–t6   | Temporaries | Caller |
//!
//! ## Floating-Point ABI Names
//! | Register | ABI Name | Role | Saved By |
//! |----------|----------|------|----------|
//! | f0–f7    | ft0–ft7  | FP temporaries | Caller |
//! | f8–f9    | fs0–fs1  | FP saved registers | Callee |
//! | f10–f11  | fa0–fa1  | FP args / return values | Caller |
//! | f12–f17  | fa2–fa7  | FP arguments | Caller |
//! | f18–f27  | fs2–fs11 | FP saved registers | Callee |
//! | f28–f31  | ft8–ft11 | FP temporaries | Caller |
//!
//! ## Register Encoding
//!
//! Integer registers x0–x31 are encoded as IDs 0–31.
//! Floating-point registers f0–f31 are encoded as IDs 32–63, providing
//! a unique namespace so the register allocator can distinguish classes.
//! The 5-bit hardware encoding for instruction fields is obtained via
//! [`hw_encoding`], which maps IDs 32–63 back to 0–31.

use crate::backend::traits::RegisterInfo;

// ===========================================================================
// Integer Register ID Constants (x0–x31) — IDs 0–31
// ===========================================================================

/// x0 — hardwired zero register. Reads always return 0; writes are discarded.
pub const X0: u16 = 0;
/// x1 — return address register (ra). Caller-saved.
pub const X1: u16 = 1;
/// x2 — stack pointer (sp). Callee-saved (by convention, always valid).
pub const X2: u16 = 2;
/// x3 — global pointer (gp). Reserved by the linker for relaxation.
pub const X3: u16 = 3;
/// x4 — thread pointer (tp). Reserved for TLS.
pub const X4: u16 = 4;
/// x5 — temporary register t0. Caller-saved.
pub const X5: u16 = 5;
/// x6 — temporary register t1. Caller-saved.
pub const X6: u16 = 6;
/// x7 — temporary register t2. Caller-saved.
pub const X7: u16 = 7;
/// x8 — saved register s0 / frame pointer (fp). Callee-saved.
pub const X8: u16 = 8;
/// x9 — saved register s1. Callee-saved.
pub const X9: u16 = 9;
/// x10 — argument register a0 / first return value. Caller-saved.
pub const X10: u16 = 10;
/// x11 — argument register a1 / second return value. Caller-saved.
pub const X11: u16 = 11;
/// x12 — argument register a2. Caller-saved.
pub const X12: u16 = 12;
/// x13 — argument register a3. Caller-saved.
pub const X13: u16 = 13;
/// x14 — argument register a4. Caller-saved.
pub const X14: u16 = 14;
/// x15 — argument register a5. Caller-saved.
pub const X15: u16 = 15;
/// x16 — argument register a6. Caller-saved.
pub const X16: u16 = 16;
/// x17 — argument register a7. Caller-saved.
pub const X17: u16 = 17;
/// x18 — saved register s2. Callee-saved.
pub const X18: u16 = 18;
/// x19 — saved register s3. Callee-saved.
pub const X19: u16 = 19;
/// x20 — saved register s4. Callee-saved.
pub const X20: u16 = 20;
/// x21 — saved register s5. Callee-saved.
pub const X21: u16 = 21;
/// x22 — saved register s6. Callee-saved.
pub const X22: u16 = 22;
/// x23 — saved register s7. Callee-saved.
pub const X23: u16 = 23;
/// x24 — saved register s8. Callee-saved.
pub const X24: u16 = 24;
/// x25 — saved register s9. Callee-saved.
pub const X25: u16 = 25;
/// x26 — saved register s10. Callee-saved.
pub const X26: u16 = 26;
/// x27 — saved register s11. Callee-saved.
pub const X27: u16 = 27;
/// x28 — temporary register t3. Caller-saved.
pub const X28: u16 = 28;
/// x29 — temporary register t4. Caller-saved.
pub const X29: u16 = 29;
/// x30 — temporary register t5. Caller-saved.
pub const X30: u16 = 30;
/// x31 — temporary register t6. Caller-saved.
pub const X31: u16 = 31;

// ===========================================================================
// ABI Name Aliases for Integer Registers
// ===========================================================================

/// Hard-wired zero register (x0).
pub const ZERO: u16 = X0;
/// Return address register (x1).
pub const RA: u16 = X1;
/// Stack pointer (x2).
pub const SP: u16 = X2;
/// Global pointer (x3).
pub const GP: u16 = X3;
/// Thread pointer (x4).
pub const TP: u16 = X4;
/// Temporary register 0 (x5).
pub const T0: u16 = X5;
/// Temporary register 1 (x6).
pub const T1: u16 = X6;
/// Temporary register 2 (x7).
pub const T2: u16 = X7;
/// Frame pointer / saved register 0 (x8). Also known as s0.
pub const FP: u16 = X8;
/// Saved register 0 (x8). Same as FP.
pub const S0: u16 = X8;
/// Saved register 1 (x9).
pub const S1: u16 = X9;
/// Argument register 0 / first integer return value (x10).
pub const A0: u16 = X10;
/// Argument register 1 / second integer return value (x11).
pub const A1: u16 = X11;
/// Argument register 2 (x12).
pub const A2: u16 = X12;
/// Argument register 3 (x13).
pub const A3: u16 = X13;
/// Argument register 4 (x14).
pub const A4: u16 = X14;
/// Argument register 5 (x15).
pub const A5: u16 = X15;
/// Argument register 6 (x16).
pub const A6: u16 = X16;
/// Argument register 7 (x17).
pub const A7: u16 = X17;
/// Saved register 2 (x18).
pub const S2: u16 = X18;
/// Saved register 3 (x19).
pub const S3: u16 = X19;
/// Saved register 4 (x20).
pub const S4: u16 = X20;
/// Saved register 5 (x21).
pub const S5: u16 = X21;
/// Saved register 6 (x22).
pub const S6: u16 = X22;
/// Saved register 7 (x23).
pub const S7: u16 = X23;
/// Saved register 8 (x24).
pub const S8: u16 = X24;
/// Saved register 9 (x25).
pub const S9: u16 = X25;
/// Saved register 10 (x26).
pub const S10: u16 = X26;
/// Saved register 11 (x27).
pub const S11: u16 = X27;
/// Temporary register 3 (x28).
pub const T3: u16 = X28;
/// Temporary register 4 (x29).
pub const T4: u16 = X29;
/// Temporary register 5 (x30).
pub const T5: u16 = X30;
/// Temporary register 6 (x31).
pub const T6: u16 = X31;

// ===========================================================================
// Floating-Point Register ID Constants (f0–f31) — IDs 32–63
// ===========================================================================

/// f0 — FP temporary ft0. Caller-saved.
pub const F0: u16 = 32;
/// f1 — FP temporary ft1. Caller-saved.
pub const F1: u16 = 33;
/// f2 — FP temporary ft2. Caller-saved.
pub const F2: u16 = 34;
/// f3 — FP temporary ft3. Caller-saved.
pub const F3: u16 = 35;
/// f4 — FP temporary ft4. Caller-saved.
pub const F4: u16 = 36;
/// f5 — FP temporary ft5. Caller-saved.
pub const F5: u16 = 37;
/// f6 — FP temporary ft6. Caller-saved.
pub const F6: u16 = 38;
/// f7 — FP temporary ft7. Caller-saved.
pub const F7: u16 = 39;
/// f8 — FP saved register fs0. Callee-saved.
pub const F8: u16 = 40;
/// f9 — FP saved register fs1. Callee-saved.
pub const F9: u16 = 41;
/// f10 — FP argument register fa0 / first FP return value. Caller-saved.
pub const F10: u16 = 42;
/// f11 — FP argument register fa1 / second FP return value. Caller-saved.
pub const F11: u16 = 43;
/// f12 — FP argument register fa2. Caller-saved.
pub const F12: u16 = 44;
/// f13 — FP argument register fa3. Caller-saved.
pub const F13: u16 = 45;
/// f14 — FP argument register fa4. Caller-saved.
pub const F14: u16 = 46;
/// f15 — FP argument register fa5. Caller-saved.
pub const F15: u16 = 47;
/// f16 — FP argument register fa6. Caller-saved.
pub const F16: u16 = 48;
/// f17 — FP argument register fa7. Caller-saved.
pub const F17: u16 = 49;
/// f18 — FP saved register fs2. Callee-saved.
pub const F18: u16 = 50;
/// f19 — FP saved register fs3. Callee-saved.
pub const F19: u16 = 51;
/// f20 — FP saved register fs4. Callee-saved.
pub const F20: u16 = 52;
/// f21 — FP saved register fs5. Callee-saved.
pub const F21: u16 = 53;
/// f22 — FP saved register fs6. Callee-saved.
pub const F22: u16 = 54;
/// f23 — FP saved register fs7. Callee-saved.
pub const F23: u16 = 55;
/// f24 — FP saved register fs8. Callee-saved.
pub const F24: u16 = 56;
/// f25 — FP saved register fs9. Callee-saved.
pub const F25: u16 = 57;
/// f26 — FP saved register fs10. Callee-saved.
pub const F26: u16 = 58;
/// f27 — FP saved register fs11. Callee-saved.
pub const F27: u16 = 59;
/// f28 — FP temporary ft8. Caller-saved.
pub const F28: u16 = 60;
/// f29 — FP temporary ft9. Caller-saved.
pub const F29: u16 = 61;
/// f30 — FP temporary ft10. Caller-saved.
pub const F30: u16 = 62;
/// f31 — FP temporary ft11. Caller-saved.
pub const F31: u16 = 63;

// ===========================================================================
// FP ABI Name Aliases
// ===========================================================================

/// FP temporary 0 (f0).
pub const FT0: u16 = F0;
/// FP temporary 1 (f1).
pub const FT1: u16 = F1;
/// FP temporary 2 (f2).
pub const FT2: u16 = F2;
/// FP temporary 3 (f3).
pub const FT3: u16 = F3;
/// FP temporary 4 (f4).
pub const FT4: u16 = F4;
/// FP temporary 5 (f5).
pub const FT5: u16 = F5;
/// FP temporary 6 (f6).
pub const FT6: u16 = F6;
/// FP temporary 7 (f7).
pub const FT7: u16 = F7;
/// FP saved register 0 (f8).
pub const FS0: u16 = F8;
/// FP saved register 1 (f9).
pub const FS1: u16 = F9;
/// FP argument 0 / first FP return value (f10).
pub const FA0: u16 = F10;
/// FP argument 1 / second FP return value (f11).
pub const FA1: u16 = F11;
/// FP argument 2 (f12).
pub const FA2: u16 = F12;
/// FP argument 3 (f13).
pub const FA3: u16 = F13;
/// FP argument 4 (f14).
pub const FA4: u16 = F14;
/// FP argument 5 (f15).
pub const FA5: u16 = F15;
/// FP argument 6 (f16).
pub const FA6: u16 = F16;
/// FP argument 7 (f17).
pub const FA7: u16 = F17;
/// FP saved register 2 (f18).
pub const FS2: u16 = F18;
/// FP saved register 3 (f19).
pub const FS3: u16 = F19;
/// FP saved register 4 (f20).
pub const FS4: u16 = F20;
/// FP saved register 5 (f21).
pub const FS5: u16 = F21;
/// FP saved register 6 (f22).
pub const FS6: u16 = F22;
/// FP saved register 7 (f23).
pub const FS7: u16 = F23;
/// FP saved register 8 (f24).
pub const FS8: u16 = F24;
/// FP saved register 9 (f25).
pub const FS9: u16 = F25;
/// FP saved register 10 (f26).
pub const FS10: u16 = F26;
/// FP saved register 11 (f27).
pub const FS11: u16 = F27;
/// FP temporary 8 (f28).
pub const FT8: u16 = F28;
/// FP temporary 9 (f29).
pub const FT9: u16 = F29;
/// FP temporary 10 (f30).
pub const FT10: u16 = F30;
/// FP temporary 11 (f31).
pub const FT11: u16 = F31;

// ===========================================================================
// Register Count Constants
// ===========================================================================

/// Number of integer general-purpose registers (x0–x31).
pub const NUM_GPRS: usize = 32;
/// Number of floating-point registers (f0–f31).
pub const NUM_FPRS: usize = 32;
/// Total number of architectural registers (GPR + FPR).
pub const TOTAL_REGS: usize = NUM_GPRS + NUM_FPRS;

// ===========================================================================
// Register Class Enum
// ===========================================================================

/// Register classes for RISC-V 64.
///
/// Used by the register allocator and instruction selector to distinguish
/// between integer and floating-point register files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// General-purpose integer registers (x0–x31, IDs 0–31).
    GPR,
    /// Floating-point registers (f0–f31, IDs 32–63).
    FPR,
}

// ===========================================================================
// Register Sets for Register Allocator
// ===========================================================================

/// Allocatable integer registers — excludes x0 (zero), x1 (ra), x2 (sp),
/// x3 (gp), and x4 (tp).
///
/// Ordered with caller-saved temporaries first (preferred for short-lived
/// values) and callee-saved registers last (preferred for long-lived values
/// since they require save/restore in prologue/epilogue).
pub const ALLOCATABLE_GPRS: &[u16] = &[
    // X5 (t0) is EXCLUDED — reserved as primary spill scratch register by
    // apply_allocation_result.  Allocating vregs to t0 would cause
    // spill operations for OTHER vregs to clobber t0's live value.
    // X6 (t1) is EXCLUDED — reserved as secondary spill scratch register.
    // On load-store architectures, ALU instructions cannot use Memory
    // operands. When two operands are both spilled, X5 loads the first
    // and X6 loads the second.
    X7, // t2 (caller-saved)
    X10, X11, X12, X13, X14, X15, X16, X17, // a0–a7 (caller-saved)
    X28, X29, X30, X31, // t3–t6 (caller-saved)
    // X8 (s0/fp) is EXCLUDED — reserved as frame pointer by the prologue.
    X9, // s1 (callee-saved)
    X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, // s2–s11 (callee-saved)
];

/// Allocatable floating-point registers — all f0–f31 are allocatable.
///
/// Ordered with caller-saved registers first (temporaries and arguments)
/// and callee-saved registers last.
pub const ALLOCATABLE_FPRS: &[u16] = &[
    // F0 (ft0) is EXCLUDED — reserved as float spill scratch register by
    // apply_allocation_result.  Same rationale as X5/t0 for GPRs.
    F1, F2, F3, F4, F5, F6, F7, // ft1–ft7 (caller-saved)
    F10, F11, F12, F13, F14, F15, F16, F17, // fa0–fa7 (caller-saved)
    F28, F29, F30, F31, // ft8–ft11 (caller-saved)
    F8, F9, // fs0–fs1 (callee-saved)
    F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, // fs2–fs11 (callee-saved)
];

// ===========================================================================
// Callee-Saved and Caller-Saved Register Sets
// ===========================================================================

/// Callee-saved integer registers: s0–s11 (x8–x9, x18–x27).
///
/// The callee must save and restore these registers if it modifies them.
/// 12 registers total.
pub const CALLEE_SAVED_GPRS: &[u16] = &[X8, X9, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27];

/// Callee-saved floating-point registers: fs0–fs11 (f8–f9, f18–f27).
///
/// 12 registers total.
pub const CALLEE_SAVED_FPRS: &[u16] = &[F8, F9, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27];

/// Caller-saved integer registers: ra, t0–t6, a0–a7.
///
/// The caller must save these registers before a call if their values
/// are needed after the call. 16 registers total.
pub const CALLER_SAVED_GPRS: &[u16] = &[
    X1, X5, X6, X7, X10, X11, X12, X13, X14, X15, X16, X17, X28, X29, X30, X31,
];

/// Caller-saved floating-point registers: ft0–ft7, fa0–fa7, ft8–ft11.
///
/// 20 registers total.
pub const CALLER_SAVED_FPRS: &[u16] = &[
    F0, F1, F2, F3, F4, F5, F6, F7, F10, F11, F12, F13, F14, F15, F16, F17, F28, F29, F30, F31,
];

/// Reserved integer registers that are never allocated by the register
/// allocator: x0 (zero), x2 (sp), x3 (gp), x4 (tp).
///
/// x1 (ra) is special — it is caller-saved and handled by the
/// prologue/epilogue but not available for general allocation.
pub const RESERVED_GPRS: &[u16] = &[X0, X1, X2, X3, X4];

// ===========================================================================
// Argument and Return Register Sets (LP64D ABI)
// ===========================================================================

/// Integer argument registers a0–a7 (x10–x17), ordered by ABI convention.
pub const ARGUMENT_GPRS: &[u16] = &[X10, X11, X12, X13, X14, X15, X16, X17];

/// Floating-point argument registers fa0–fa7 (f10–f17), ordered by ABI convention.
pub const ARGUMENT_FPRS: &[u16] = &[F10, F11, F12, F13, F14, F15, F16, F17];

/// Integer return registers a0–a1 (x10–x11).
pub const RETURN_GPRS: &[u16] = &[X10, X11];

/// Floating-point return registers fa0–fa1 (f10–f11).
pub const RETURN_FPRS: &[u16] = &[F10, F11];

// ===========================================================================
// GPR Name Lookup Tables
// ===========================================================================

/// ABI names for integer registers x0–x31, indexed by register number.
const GPR_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

/// ABI names for floating-point registers f0–f31, indexed by f-register number
/// (0–31, NOT the full register ID 32–63).
const FPR_NAMES: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2",
    "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9",
    "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

// ===========================================================================
// Register Name Lookup Functions
// ===========================================================================

/// Returns the ABI name for the given integer register (0–31).
///
/// # Panics
///
/// Panics if `reg` is not in the range 0–31.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(gpr_name(0), "zero");
/// assert_eq!(gpr_name(1), "ra");
/// assert_eq!(gpr_name(10), "a0");
/// ```
#[inline]
pub fn gpr_name(reg: u16) -> &'static str {
    debug_assert!(reg < 32, "gpr_name: register {} out of range 0–31", reg);
    if reg < 32 {
        GPR_NAMES[reg as usize]
    } else {
        "??gpr"
    }
}

/// Returns the ABI name for the given floating-point register.
///
/// The input `reg` is the **full register ID** (32–63). The function
/// subtracts 32 to obtain the f-register number (0–31) and returns
/// the corresponding ABI name.
///
/// # Panics
///
/// Panics (in debug builds) if `reg` is not in the range 32–63.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(fpr_name(32), "ft0");  // f0
/// assert_eq!(fpr_name(42), "fa0");  // f10
/// ```
#[inline]
pub fn fpr_name(reg: u16) -> &'static str {
    debug_assert!(
        (32..64).contains(&reg),
        "fpr_name: register {} out of range 32–63",
        reg
    );
    let idx = reg.wrapping_sub(32);
    if idx < 32 {
        FPR_NAMES[idx as usize]
    } else {
        "??fpr"
    }
}

/// Returns the ABI name for any register, dispatching to [`gpr_name`] or
/// [`fpr_name`] based on the register ID range.
///
/// - IDs 0–31: integer registers (returns GPR ABI name)
/// - IDs 32–63: floating-point registers (returns FPR ABI name)
/// - IDs >= 64: returns `"??reg"` (invalid)
///
/// # Examples
///
/// ```ignore
/// assert_eq!(reg_name(0), "zero");
/// assert_eq!(reg_name(2), "sp");
/// assert_eq!(reg_name(42), "fa0");
/// ```
#[inline]
pub fn reg_name(reg: u16) -> &'static str {
    if reg < 32 {
        GPR_NAMES[reg as usize]
    } else if reg < 64 {
        FPR_NAMES[(reg - 32) as usize]
    } else {
        "??reg"
    }
}

// ===========================================================================
// Register Helper Functions
// ===========================================================================

/// Returns `true` if `reg` is a general-purpose integer register (ID 0–31).
#[inline]
pub fn is_gpr(reg: u16) -> bool {
    reg < 32
}

/// Returns `true` if `reg` is a floating-point register (ID 32–63).
#[inline]
pub fn is_fpr(reg: u16) -> bool {
    (32..64).contains(&reg)
}

/// Returns `true` if `reg` is a callee-saved register (either GPR or FPR).
///
/// Callee-saved GPRs: s0–s11 (x8–x9, x18–x27)
/// Callee-saved FPRs: fs0–fs11 (f8–f9, f18–f27)
#[inline]
pub fn is_callee_saved(reg: u16) -> bool {
    // GPR callee-saved: x8, x9, x18–x27
    if reg < 32 {
        reg == X8 || reg == X9 || (X18..=X27).contains(&reg)
    } else if reg < 64 {
        // FPR callee-saved: f8, f9, f18–f27
        let f = reg - 32;
        f == 8 || f == 9 || (18..=27).contains(&f)
    } else {
        false
    }
}

/// Returns `true` if `reg` is a caller-saved register (either GPR or FPR).
///
/// Caller-saved GPRs: ra (x1), t0–t2 (x5–x7), a0–a7 (x10–x17), t3–t6 (x28–x31)
/// Caller-saved FPRs: ft0–ft7 (f0–f7), fa0–fa7 (f10–f17), ft8–ft11 (f28–f31)
#[inline]
pub fn is_caller_saved(reg: u16) -> bool {
    if reg < 32 {
        reg == X1
            || (X5..=X7).contains(&reg)
            || (X10..=X17).contains(&reg)
            || (X28..=X31).contains(&reg)
    } else if reg < 64 {
        let f = reg - 32;
        f <= 7 || (10..=17).contains(&f) || (28..=31).contains(&f)
    } else {
        false
    }
}

/// Returns `true` if `reg` is reserved and cannot be used for general
/// register allocation.
///
/// Reserved registers: x0 (zero), x2 (sp), x3 (gp), x4 (tp).
/// x1 (ra) is also reserved from general allocation but is handled
/// specially by prologue/epilogue.
#[inline]
pub fn is_reserved(reg: u16) -> bool {
    reg == X0 || reg == X1 || reg == X2 || reg == X3 || reg == X4
}

/// Returns `true` if `reg` is available for general register allocation.
///
/// A register is allocatable if it is not reserved (x0, x1, x2, x3, x4)
/// and is a valid GPR or FPR. All 32 FPRs are allocatable.
#[inline]
pub fn is_allocatable(reg: u16) -> bool {
    if reg < 32 {
        // GPR: allocatable if not reserved
        !is_reserved(reg)
    } else if reg < 64 {
        // FPR: all 32 FP registers are allocatable
        true
    } else {
        false
    }
}

/// Returns the register class for the given register ID.
///
/// - IDs 0–31: [`RegClass::GPR`]
/// - IDs 32–63: [`RegClass::FPR`]
///
/// # Panics
///
/// Panics (in debug builds) if `reg` >= 64.
#[inline]
pub fn reg_class(reg: u16) -> RegClass {
    debug_assert!(reg < 64, "reg_class: register {} out of range 0–63", reg);
    if reg < 32 {
        RegClass::GPR
    } else {
        RegClass::FPR
    }
}

/// Returns the 5-bit hardware encoding for instruction fields.
///
/// For GPRs (0–31): returns the register number directly (0–31).
/// For FPRs (32–63): returns the f-register number (0–31), i.e., `reg - 32`.
///
/// This value is placed in the `rd`, `rs1`, `rs2`, or `rs3` fields
/// of RISC-V instructions (all 5 bits wide).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(hw_encoding(X0), 0);
/// assert_eq!(hw_encoding(X31), 31);
/// assert_eq!(hw_encoding(F0), 0);   // f0 encodes as 0 in instruction fields
/// assert_eq!(hw_encoding(F31), 31); // f31 encodes as 31
/// ```
#[inline]
pub fn hw_encoding(reg: u16) -> u8 {
    if reg < 32 {
        reg as u8
    } else {
        (reg - 32) as u8
    }
}

// ===========================================================================
// CSR (Control and Status Register) Constants
// ===========================================================================

/// CSR addresses for relevant control/status registers.
///
/// These are used primarily in inline assembly and kernel code that
/// accesses floating-point status, performance counters, and timer
/// registers via `csrr`/`csrw` instructions.
pub mod csr {
    /// Floating-point accrued exception flags (read/write).
    pub const FFLAGS: u16 = 0x001;
    /// Floating-point rounding mode (read/write).
    pub const FRM: u16 = 0x002;
    /// Floating-point control and status register (read/write).
    /// Combines `fflags` and `frm` in a single register.
    pub const FCSR: u16 = 0x003;
    /// Cycle counter (read-only, user mode).
    pub const CYCLE: u16 = 0xC00;
    /// Real-time clock (read-only, user mode).
    pub const TIME: u16 = 0xC01;
    /// Instructions retired counter (read-only, user mode).
    pub const INSTRET: u16 = 0xC02;
}

// ===========================================================================
// RiscV64RegisterInfo — Register Information Provider
// ===========================================================================

/// RISC-V 64 register information provider.
///
/// Implements the register query interface used by the code generation driver,
/// register allocator, and ABI modules. Provides methods to query register
/// names, allocatable sets, callee/caller-saved registers, and special-purpose
/// register identifiers (frame pointer, stack pointer, return address).
///
/// Also provides [`to_register_info`](RiscV64RegisterInfo::to_register_info)
/// to construct the [`RegisterInfo`] struct expected by the
/// [`ArchCodegen`](crate::backend::traits::ArchCodegen) trait.
#[derive(Debug, Clone, Copy)]
pub struct RiscV64RegisterInfo;

impl RiscV64RegisterInfo {
    /// Returns the total number of architectural registers (GPR + FPR = 64).
    #[inline]
    pub fn num_regs(&self) -> usize {
        TOTAL_REGS
    }

    /// Returns the ABI name for the given register ID.
    #[inline]
    pub fn reg_name(&self, reg: u16) -> &'static str {
        reg_name(reg)
    }

    /// Returns the allocatable registers for the given register class.
    ///
    /// - [`RegClass::GPR`]: returns [`ALLOCATABLE_GPRS`]
    /// - [`RegClass::FPR`]: returns [`ALLOCATABLE_FPRS`]
    #[inline]
    pub fn allocatable_regs(&self, class: RegClass) -> &'static [u16] {
        match class {
            RegClass::GPR => ALLOCATABLE_GPRS,
            RegClass::FPR => ALLOCATABLE_FPRS,
        }
    }

    /// Returns the callee-saved GPR register set.
    #[inline]
    pub fn callee_saved(&self) -> &'static [u16] {
        CALLEE_SAVED_GPRS
    }

    /// Returns the caller-saved GPR register set.
    #[inline]
    pub fn caller_saved(&self) -> &'static [u16] {
        CALLER_SAVED_GPRS
    }

    /// Returns the frame pointer register ID (x8 / s0 / fp).
    #[inline]
    pub fn frame_pointer(&self) -> u16 {
        FP
    }

    /// Returns the stack pointer register ID (x2 / sp).
    #[inline]
    pub fn stack_pointer(&self) -> u16 {
        SP
    }

    /// Returns the return address register ID (x1 / ra).
    #[inline]
    pub fn return_address(&self) -> u16 {
        RA
    }

    /// Constructs a [`RegisterInfo`] struct compatible with the
    /// architecture-agnostic backend infrastructure.
    ///
    /// This converts the `u16` register IDs used internally by the RISC-V
    /// backend into the `u16` IDs expected by [`RegisterInfo`] fields.
    pub fn to_register_info(&self) -> RegisterInfo {
        RegisterInfo {
            allocatable_gpr: ALLOCATABLE_GPRS.to_vec(),
            allocatable_fpr: ALLOCATABLE_FPRS.to_vec(),
            callee_saved: CALLEE_SAVED_GPRS
                .iter()
                .chain(CALLEE_SAVED_FPRS.iter())
                .copied()
                .collect(),
            caller_saved: CALLER_SAVED_GPRS
                .iter()
                .chain(CALLER_SAVED_FPRS.iter())
                .copied()
                .collect(),
            reserved: RESERVED_GPRS.to_vec(),
            argument_gpr: ARGUMENT_GPRS.to_vec(),
            argument_fpr: ARGUMENT_FPRS.to_vec(),
            return_gpr: RETURN_GPRS.to_vec(),
            return_fpr: RETURN_FPRS.to_vec(),
        }
    }
}
