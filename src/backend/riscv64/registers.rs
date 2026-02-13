//! RISC-V 64-bit register definitions for the BCC compiler backend.
//!
//! This module provides named constants for the complete RISC-V RV64 register
//! file — 32 integer registers (x0–x31) and 32 floating-point registers
//! (f0–f31) — along with ABI-mandated aliases, register classification arrays,
//! property query functions, and the small set of CSRs relevant to floating-point
//! control.
//!
//! # Register Encoding
//!
//! Physical register identifiers use the [`PhysReg`] wrapper from
//! [`crate::backend::traits`]. Integer registers are assigned indices 0–31
//! and floating-point registers 32–63, matching the hardware 5-bit encoding
//! when masked with `0x1F`.
//!
//! ```text
//! PhysReg(0)  = x0  (zero)       PhysReg(32) = f0  (ft0)
//! PhysReg(1)  = x1  (ra)         PhysReg(33) = f1  (ft1)
//!   ...                            ...
//! PhysReg(31) = x31 (t6)         PhysReg(63) = f31 (ft11)
//! ```
//!
//! # ABI Reference (LP64D)
//!
//! The RISC-V LP64D calling convention divides the integer register file as:
//!
//! | Register(s) | ABI Name | Role                         | Saved By |
//! |------------|----------|------------------------------|----------|
//! | x0         | zero     | Hardwired zero               | —        |
//! | x1         | ra       | Return address               | Caller   |
//! | x2         | sp       | Stack pointer                | Callee   |
//! | x3         | gp       | Global pointer               | —        |
//! | x4         | tp       | Thread pointer               | —        |
//! | x5–x7     | t0–t2    | Temporaries                  | Caller   |
//! | x8         | s0/fp    | Saved register / Frame ptr   | Callee   |
//! | x9         | s1       | Saved register               | Callee   |
//! | x10–x17   | a0–a7    | Arguments / Return values    | Caller   |
//! | x18–x27   | s2–s11   | Saved registers              | Callee   |
//! | x28–x31   | t3–t6    | Temporaries                  | Caller   |
//!
//! The floating-point register file follows a parallel scheme:
//!
//! | Register(s) | ABI Name   | Role                       | Saved By |
//! |------------|------------|----------------------------|----------|
//! | f0–f7      | ft0–ft7    | FP temporaries             | Caller   |
//! | f8–f9      | fs0–fs1    | FP saved registers         | Callee   |
//! | f10–f17    | fa0–fa7    | FP arguments / Return vals | Caller   |
//! | f18–f27    | fs2–fs11   | FP saved registers         | Callee   |
//! | f28–f31    | ft8–ft11   | FP temporaries             | Caller   |

use crate::backend::traits::PhysReg;

// ===========================================================================
// Integer Registers x0–x31 (PhysReg 0–31)
// ===========================================================================

/// x0 — hardwired zero register. Reads always return 0; writes are discarded.
pub const X0: PhysReg = PhysReg(0);
/// x1 — return address (ra).
pub const X1: PhysReg = PhysReg(1);
/// x2 — stack pointer (sp).
pub const X2: PhysReg = PhysReg(2);
/// x3 — global pointer (gp).
pub const X3: PhysReg = PhysReg(3);
/// x4 — thread pointer (tp).
pub const X4: PhysReg = PhysReg(4);
/// x5 — temporary / alternate link register (t0).
pub const X5: PhysReg = PhysReg(5);
/// x6 — temporary (t1).
pub const X6: PhysReg = PhysReg(6);
/// x7 — temporary (t2).
pub const X7: PhysReg = PhysReg(7);
/// x8 — saved register / frame pointer (s0/fp).
pub const X8: PhysReg = PhysReg(8);
/// x9 — saved register (s1).
pub const X9: PhysReg = PhysReg(9);
/// x10 — function argument 0 / return value 0 (a0).
pub const X10: PhysReg = PhysReg(10);
/// x11 — function argument 1 / return value 1 (a1).
pub const X11: PhysReg = PhysReg(11);
/// x12 — function argument 2 (a2).
pub const X12: PhysReg = PhysReg(12);
/// x13 — function argument 3 (a3).
pub const X13: PhysReg = PhysReg(13);
/// x14 — function argument 4 (a4).
pub const X14: PhysReg = PhysReg(14);
/// x15 — function argument 5 (a5).
pub const X15: PhysReg = PhysReg(15);
/// x16 — function argument 6 (a6).
pub const X16: PhysReg = PhysReg(16);
/// x17 — function argument 7 (a7).
pub const X17: PhysReg = PhysReg(17);
/// x18 — saved register (s2).
pub const X18: PhysReg = PhysReg(18);
/// x19 — saved register (s3).
pub const X19: PhysReg = PhysReg(19);
/// x20 — saved register (s4).
pub const X20: PhysReg = PhysReg(20);
/// x21 — saved register (s5).
pub const X21: PhysReg = PhysReg(21);
/// x22 — saved register (s6).
pub const X22: PhysReg = PhysReg(22);
/// x23 — saved register (s7).
pub const X23: PhysReg = PhysReg(23);
/// x24 — saved register (s8).
pub const X24: PhysReg = PhysReg(24);
/// x25 — saved register (s9).
pub const X25: PhysReg = PhysReg(25);
/// x26 — saved register (s10).
pub const X26: PhysReg = PhysReg(26);
/// x27 — saved register (s11).
pub const X27: PhysReg = PhysReg(27);
/// x28 — temporary (t3).
pub const X28: PhysReg = PhysReg(28);
/// x29 — temporary (t4).
pub const X29: PhysReg = PhysReg(29);
/// x30 — temporary (t5).
pub const X30: PhysReg = PhysReg(30);
/// x31 — temporary (t6).
pub const X31: PhysReg = PhysReg(31);

// ===========================================================================
// Integer Register ABI Aliases
// ===========================================================================

/// Hardwired zero register (x0). Reads always yield 0.
pub const ZERO: PhysReg = X0;
/// Return address register (x1). Holds the return address after `jal`/`jalr`.
pub const RA: PhysReg = X1;
/// Stack pointer (x2). Must be 16-byte aligned at function entry per LP64D ABI.
pub const SP: PhysReg = X2;
/// Global pointer (x3). Used for linker-relaxed accesses to global data.
pub const GP: PhysReg = X3;
/// Thread pointer (x4). Points to the thread-local storage block.
pub const TP: PhysReg = X4;

// --- Temporaries (caller-saved) ---

/// Temporary register 0 (x5). Also serves as alternate link register.
pub const T0: PhysReg = X5;
/// Temporary register 1 (x6).
pub const T1: PhysReg = X6;
/// Temporary register 2 (x7).
pub const T2: PhysReg = X7;
/// Temporary register 3 (x28).
pub const T3: PhysReg = X28;
/// Temporary register 4 (x29).
pub const T4: PhysReg = X29;
/// Temporary register 5 (x30).
pub const T5: PhysReg = X30;
/// Temporary register 6 (x31).
pub const T6: PhysReg = X31;

// --- Saved registers (callee-saved) ---

/// Saved register 0 / frame pointer (x8). Callee-saved.
pub const S0: PhysReg = X8;
/// Saved register 1 (x9). Callee-saved.
pub const S1: PhysReg = X9;
/// Saved register 2 (x18). Callee-saved.
pub const S2: PhysReg = X18;
/// Saved register 3 (x19). Callee-saved.
pub const S3: PhysReg = X19;
/// Saved register 4 (x20). Callee-saved.
pub const S4: PhysReg = X20;
/// Saved register 5 (x21). Callee-saved.
pub const S5: PhysReg = X21;
/// Saved register 6 (x22). Callee-saved.
pub const S6: PhysReg = X22;
/// Saved register 7 (x23). Callee-saved.
pub const S7: PhysReg = X23;
/// Saved register 8 (x24). Callee-saved.
pub const S8: PhysReg = X24;
/// Saved register 9 (x25). Callee-saved.
pub const S9: PhysReg = X25;
/// Saved register 10 (x26). Callee-saved.
pub const S10: PhysReg = X26;
/// Saved register 11 (x27). Callee-saved.
pub const S11: PhysReg = X27;

/// Frame pointer alias (x8). Same physical register as [`S0`].
pub const FP: PhysReg = X8;

// --- Argument / return-value registers (caller-saved) ---

/// Integer argument register 0 / first return value (x10).
pub const A0: PhysReg = X10;
/// Integer argument register 1 / second return value (x11).
pub const A1: PhysReg = X11;
/// Integer argument register 2 (x12).
pub const A2: PhysReg = X12;
/// Integer argument register 3 (x13).
pub const A3: PhysReg = X13;
/// Integer argument register 4 (x14).
pub const A4: PhysReg = X14;
/// Integer argument register 5 (x15).
pub const A5: PhysReg = X15;
/// Integer argument register 6 (x16).
pub const A6: PhysReg = X16;
/// Integer argument register 7 (x17).
pub const A7: PhysReg = X17;

// ===========================================================================
// Floating-Point Registers f0–f31 (PhysReg 32–63)
// ===========================================================================

/// f0 — FP temporary (ft0).
pub const F0: PhysReg = PhysReg(32);
/// f1 — FP temporary (ft1).
pub const F1: PhysReg = PhysReg(33);
/// f2 — FP temporary (ft2).
pub const F2: PhysReg = PhysReg(34);
/// f3 — FP temporary (ft3).
pub const F3: PhysReg = PhysReg(35);
/// f4 — FP temporary (ft4).
pub const F4: PhysReg = PhysReg(36);
/// f5 — FP temporary (ft5).
pub const F5: PhysReg = PhysReg(37);
/// f6 — FP temporary (ft6).
pub const F6: PhysReg = PhysReg(38);
/// f7 — FP temporary (ft7).
pub const F7: PhysReg = PhysReg(39);
/// f8 — FP saved register (fs0). Callee-saved.
pub const F8: PhysReg = PhysReg(40);
/// f9 — FP saved register (fs1). Callee-saved.
pub const F9: PhysReg = PhysReg(41);
/// f10 — FP argument 0 / FP return value 0 (fa0).
pub const F10: PhysReg = PhysReg(42);
/// f11 — FP argument 1 / FP return value 1 (fa1).
pub const F11: PhysReg = PhysReg(43);
/// f12 — FP argument 2 (fa2).
pub const F12: PhysReg = PhysReg(44);
/// f13 — FP argument 3 (fa3).
pub const F13: PhysReg = PhysReg(45);
/// f14 — FP argument 4 (fa4).
pub const F14: PhysReg = PhysReg(46);
/// f15 — FP argument 5 (fa5).
pub const F15: PhysReg = PhysReg(47);
/// f16 — FP argument 6 (fa6).
pub const F16: PhysReg = PhysReg(48);
/// f17 — FP argument 7 (fa7).
pub const F17: PhysReg = PhysReg(49);
/// f18 — FP saved register (fs2). Callee-saved.
pub const F18: PhysReg = PhysReg(50);
/// f19 — FP saved register (fs3). Callee-saved.
pub const F19: PhysReg = PhysReg(51);
/// f20 — FP saved register (fs4). Callee-saved.
pub const F20: PhysReg = PhysReg(52);
/// f21 — FP saved register (fs5). Callee-saved.
pub const F21: PhysReg = PhysReg(53);
/// f22 — FP saved register (fs6). Callee-saved.
pub const F22: PhysReg = PhysReg(54);
/// f23 — FP saved register (fs7). Callee-saved.
pub const F23: PhysReg = PhysReg(55);
/// f24 — FP saved register (fs8). Callee-saved.
pub const F24: PhysReg = PhysReg(56);
/// f25 — FP saved register (fs9). Callee-saved.
pub const F25: PhysReg = PhysReg(57);
/// f26 — FP saved register (fs10). Callee-saved.
pub const F26: PhysReg = PhysReg(58);
/// f27 — FP saved register (fs11). Callee-saved.
pub const F27: PhysReg = PhysReg(59);
/// f28 — FP temporary (ft8).
pub const F28: PhysReg = PhysReg(60);
/// f29 — FP temporary (ft9).
pub const F29: PhysReg = PhysReg(61);
/// f30 — FP temporary (ft10).
pub const F30: PhysReg = PhysReg(62);
/// f31 — FP temporary (ft11).
pub const F31: PhysReg = PhysReg(63);

// ===========================================================================
// Floating-Point Register ABI Aliases
// ===========================================================================

// --- FP temporaries (caller-saved) ---

/// FP temporary 0 (f0). Caller-saved.
pub const FT0: PhysReg = F0;
/// FP temporary 1 (f1). Caller-saved.
pub const FT1: PhysReg = F1;
/// FP temporary 2 (f2). Caller-saved.
pub const FT2: PhysReg = F2;
/// FP temporary 3 (f3). Caller-saved.
pub const FT3: PhysReg = F3;
/// FP temporary 4 (f4). Caller-saved.
pub const FT4: PhysReg = F4;
/// FP temporary 5 (f5). Caller-saved.
pub const FT5: PhysReg = F5;
/// FP temporary 6 (f6). Caller-saved.
pub const FT6: PhysReg = F6;
/// FP temporary 7 (f7). Caller-saved.
pub const FT7: PhysReg = F7;
/// FP temporary 8 (f28). Caller-saved.
pub const FT8: PhysReg = F28;
/// FP temporary 9 (f29). Caller-saved.
pub const FT9: PhysReg = F29;
/// FP temporary 10 (f30). Caller-saved.
pub const FT10: PhysReg = F30;
/// FP temporary 11 (f31). Caller-saved.
pub const FT11: PhysReg = F31;

// --- FP saved registers (callee-saved) ---

/// FP saved register 0 (f8). Callee-saved.
pub const FS0: PhysReg = F8;
/// FP saved register 1 (f9). Callee-saved.
pub const FS1: PhysReg = F9;
/// FP saved register 2 (f18). Callee-saved.
pub const FS2: PhysReg = F18;
/// FP saved register 3 (f19). Callee-saved.
pub const FS3: PhysReg = F19;
/// FP saved register 4 (f20). Callee-saved.
pub const FS4: PhysReg = F20;
/// FP saved register 5 (f21). Callee-saved.
pub const FS5: PhysReg = F21;
/// FP saved register 6 (f22). Callee-saved.
pub const FS6: PhysReg = F22;
/// FP saved register 7 (f23). Callee-saved.
pub const FS7: PhysReg = F23;
/// FP saved register 8 (f24). Callee-saved.
pub const FS8: PhysReg = F24;
/// FP saved register 9 (f25). Callee-saved.
pub const FS9: PhysReg = F25;
/// FP saved register 10 (f26). Callee-saved.
pub const FS10: PhysReg = F26;
/// FP saved register 11 (f27). Callee-saved.
pub const FS11: PhysReg = F27;

// --- FP argument / return-value registers (caller-saved) ---

/// FP argument register 0 / first FP return value (f10).
pub const FA0: PhysReg = F10;
/// FP argument register 1 / second FP return value (f11).
pub const FA1: PhysReg = F11;
/// FP argument register 2 (f12).
pub const FA2: PhysReg = F12;
/// FP argument register 3 (f13).
pub const FA3: PhysReg = F13;
/// FP argument register 4 (f14).
pub const FA4: PhysReg = F14;
/// FP argument register 5 (f15).
pub const FA5: PhysReg = F15;
/// FP argument register 6 (f16).
pub const FA6: PhysReg = F16;
/// FP argument register 7 (f17).
pub const FA7: PhysReg = F17;

// ===========================================================================
// Register Classification Arrays
// ===========================================================================

/// Integer argument registers in calling-convention order (a0–a7).
///
/// The LP64D ABI passes the first 8 integer (or pointer) arguments in
/// registers a0 through a7. Excess arguments spill to the stack.
pub const INTEGER_ARG_REGS: [PhysReg; 8] = [A0, A1, A2, A3, A4, A5, A6, A7];

/// Floating-point argument registers in calling-convention order (fa0–fa7).
///
/// The LP64D ABI passes the first 8 floating-point arguments in registers
/// fa0 through fa7. When the D extension is present, both `float` and
/// `double` arguments use these registers.
pub const FLOAT_ARG_REGS: [PhysReg; 8] = [FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7];

/// Callee-saved integer registers (s0–s11).
///
/// These 12 registers must be preserved across function calls. The function
/// prologue saves any of these that it clobbers, and the epilogue restores
/// them before returning. s0 is also the conventional frame pointer.
pub const CALLEE_SAVED_INT: [PhysReg; 12] = [
    S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
];

/// Callee-saved floating-point registers (fs0–fs11).
///
/// These 12 FP registers follow the same save/restore contract as
/// [`CALLEE_SAVED_INT`] — callee must preserve them if modified.
pub const CALLEE_SAVED_FP: [PhysReg; 12] = [
    FS0, FS1, FS2, FS3, FS4, FS5, FS6, FS7, FS8, FS9, FS10, FS11,
];

/// Caller-saved integer registers.
///
/// Includes the return address (`ra`), temporaries (`t0`–`t6`), and
/// argument registers (`a0`–`a7`). The caller must assume these are
/// destroyed by a function call.
///
/// Layout: `[ra, t0, t1, t2, a0, a1, a2, a3, a4, a5, a6, a7, t3, t4, t5, t6]`
pub const CALLER_SAVED_INT: [PhysReg; 16] = [
    RA, T0, T1, T2, A0, A1, A2, A3, A4, A5, A6, A7, T3, T4, T5, T6,
];

/// Caller-saved floating-point registers.
///
/// Includes FP temporaries (`ft0`–`ft11`) and FP argument registers
/// (`fa0`–`fa7`). The caller must assume these are destroyed by a call.
///
/// Layout: `[ft0..ft7, fa0..fa7, ft8..ft11]`
pub const CALLER_SAVED_FP: [PhysReg; 20] = [
    FT0, FT1, FT2, FT3, FT4, FT5, FT6, FT7,
    FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7,
    FT8, FT9, FT10, FT11,
];

/// Integer registers available for the register allocator.
///
/// Excludes registers that have fixed architectural roles and must not be
/// arbitrarily reassigned:
/// - `x0` (hardwired zero — writes are discarded)
/// - `x2` / `sp` (stack pointer — managed by prologue/epilogue)
/// - `x3` / `gp` (global pointer — reserved for linker relaxation)
/// - `x4` / `tp` (thread pointer — reserved for TLS)
///
/// The remaining 28 integer registers are allocatable. The register
/// allocator treats callee-saved registers as higher spill-cost candidates.
pub const ALLOCATABLE_INT: [PhysReg; 28] = [
    // Return address (callee may save/restore if needed)
    X1,  // ra
    // Temporaries t0–t2
    X5, X6, X7,
    // Saved registers s0–s1
    X8, X9,
    // Argument registers a0–a7
    X10, X11, X12, X13, X14, X15, X16, X17,
    // Saved registers s2–s11
    X18, X19, X20, X21, X22, X23, X24, X25, X26, X27,
    // Temporaries t3–t6
    X28, X29, X30, X31,
];

/// Floating-point registers available for the register allocator.
///
/// All 32 FP registers (f0–f31) are allocatable. Unlike the integer file,
/// the FP register file has no hardwired-zero or stack-pointer equivalent,
/// so every register is available.
pub const ALLOCATABLE_FP: [PhysReg; 32] = [
    F0,  F1,  F2,  F3,  F4,  F5,  F6,  F7,
    F8,  F9,  F10, F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23,
    F24, F25, F26, F27, F28, F29, F30, F31,
];

// ===========================================================================
// CSR Constants — Control and Status Registers
// ===========================================================================

/// CSR address for FP Accrued Exception Flags (`fflags`, CSR 0x001).
///
/// Bits [4:0] record which IEEE 754 exceptions have occurred:
/// - bit 0: Inexact (NX)
/// - bit 1: Underflow (UF)
/// - bit 2: Overflow (OF)
/// - bit 3: Divide by Zero (DZ)
/// - bit 4: Invalid Operation (NV)
pub const CSR_FFLAGS: u16 = 0x001;

/// CSR address for FP Dynamic Rounding Mode (`frm`, CSR 0x002).
///
/// Bits [2:0] select the rounding mode:
/// - 0b000: Round to Nearest, ties to Even (RNE)
/// - 0b001: Round towards Zero (RTZ)
/// - 0b010: Round Down (RDN)
/// - 0b011: Round Up (RUP)
/// - 0b100: Round to Nearest, ties to Max Magnitude (RMM)
/// - 0b101–0b110: Reserved
/// - 0b111: Dynamic (use instruction's rm field)
pub const CSR_FRM: u16 = 0x002;

/// CSR address for FP Control and Status Register (`fcsr`, CSR 0x003).
///
/// This register is a combined view of [`CSR_FFLAGS`] (bits [4:0]) and
/// [`CSR_FRM`] (bits [7:5]). Reading/writing `fcsr` is equivalent to
/// accessing both `fflags` and `frm` simultaneously.
pub const CSR_FCSR: u16 = 0x003;

// ===========================================================================
// Register Name Lookup
// ===========================================================================

/// Integer register ABI names indexed by hardware encoding (0–31).
///
/// These are the canonical names emitted in assembly output, matching the
/// RISC-V ABI naming convention used by GCC and LLVM.
const INT_REG_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8",   "s9", "s10","s11","t3", "t4", "t5", "t6",
];

/// Floating-point register ABI names indexed by hardware encoding (0–31).
const FP_REG_NAMES: [&str; 32] = [
    "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6",  "ft7",
    "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4",  "fa5",
    "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6",  "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// Returns the ABI name of an integer register.
///
/// # Panics
///
/// Panics if `reg` does not fall in the integer register range
/// (PhysReg 0–31).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(int_reg_name(X0), "zero");
/// assert_eq!(int_reg_name(A0), "a0");
/// assert_eq!(int_reg_name(T6), "t6");
/// ```
#[inline]
pub fn int_reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(idx < 32, "int_reg_name called with non-integer register PhysReg({})", reg.0);
    if idx < 32 {
        INT_REG_NAMES[idx]
    } else {
        "<invalid-int-reg>"
    }
}

/// Returns the ABI name of a floating-point register.
///
/// # Panics
///
/// Panics (in debug builds) if `reg` does not fall in the FP register range
/// (PhysReg 32–63).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(fp_reg_name(F0), "ft0");
/// assert_eq!(fp_reg_name(FA0), "fa0");
/// assert_eq!(fp_reg_name(FS11), "fs11");
/// ```
#[inline]
pub fn fp_reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(
        (32..=63).contains(&idx),
        "fp_reg_name called with non-FP register PhysReg({})", reg.0
    );
    if idx >= 32 && idx < 64 {
        FP_REG_NAMES[idx - 32]
    } else {
        "<invalid-fp-reg>"
    }
}

/// Returns the ABI name of any RISC-V register (integer or floating-point).
///
/// Dispatches to [`int_reg_name`] for PhysReg 0–31 and [`fp_reg_name`] for
/// PhysReg 32–63. Returns a placeholder string for out-of-range values.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(reg_name(SP), "sp");
/// assert_eq!(reg_name(FA0), "fa0");
/// ```
#[inline]
pub fn reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx < 32 {
        INT_REG_NAMES[idx]
    } else if idx < 64 {
        FP_REG_NAMES[idx - 32]
    } else {
        "<invalid-reg>"
    }
}

// ===========================================================================
// Register Property Queries
// ===========================================================================

/// Returns `true` if `reg` is an integer register (x0–x31, PhysReg 0–31).
#[inline]
pub fn is_integer_reg(reg: PhysReg) -> bool {
    reg.0 < 32
}

/// Returns `true` if `reg` is a floating-point register (f0–f31, PhysReg 32–63).
#[inline]
pub fn is_float_reg(reg: PhysReg) -> bool {
    reg.0 >= 32 && reg.0 < 64
}

/// Returns `true` if `reg` is callee-saved under the LP64D ABI.
///
/// Callee-saved integer registers: s0–s11 (x8–x9, x18–x27).
/// Callee-saved FP registers: fs0–fs11 (f8–f9, f18–f27).
///
/// Note: `sp` (x2) is technically callee-saved in the sense that it must
/// be restored, but it is managed by prologue/epilogue code and not part
/// of the register allocator's callee-saved set.
#[inline]
pub fn is_callee_saved(reg: PhysReg) -> bool {
    let r = reg.0;
    // Integer callee-saved: x8-x9 (s0-s1), x18-x27 (s2-s11)
    if r < 32 {
        matches!(r, 8 | 9 | 18..=27)
    }
    // FP callee-saved: f8-f9 (fs0-fs1), f18-f27 (fs2-fs11)
    else if r < 64 {
        let fp_idx = r - 32;
        matches!(fp_idx, 8 | 9 | 18..=27)
    } else {
        false
    }
}

/// Returns `true` if `reg` is available for the register allocator.
///
/// The following integer registers are **not** allocatable:
/// - `x0` (hardwired zero — writes have no effect)
/// - `x2` / `sp` (stack pointer — reserved for stack management)
/// - `x3` / `gp` (global pointer — reserved for linker relaxation)
/// - `x4` / `tp` (thread pointer — reserved for TLS access)
///
/// All 32 floating-point registers are allocatable — none have a fixed
/// architectural role analogous to `zero` or `sp`.
#[inline]
pub fn is_allocatable(reg: PhysReg) -> bool {
    let r = reg.0;
    if r < 32 {
        // Exclude: x0 (zero), x2 (sp), x3 (gp), x4 (tp)
        !matches!(r, 0 | 2 | 3 | 4)
    } else if r < 64 {
        // All FP registers are allocatable
        true
    } else {
        false
    }
}

/// Returns the 5-bit hardware encoding for `reg`.
///
/// For integer registers (PhysReg 0–31), this is the PhysReg index directly.
/// For floating-point registers (PhysReg 32–63), this is `PhysReg.0 - 32`.
///
/// The returned value fits in bits [4:0] and is used directly in the `rd`,
/// `rs1`, `rs2`, and `rs3` fields of RISC-V instruction encodings.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(encoding(X0), 0);
/// assert_eq!(encoding(X31), 31);
/// assert_eq!(encoding(F0), 0);
/// assert_eq!(encoding(F31), 31);
/// ```
#[inline]
pub fn encoding(reg: PhysReg) -> u8 {
    let r = reg.0;
    if r < 32 {
        r as u8
    } else if r < 64 {
        (r - 32) as u8
    } else {
        // Sentinel / invalid — return 0 as a safe default.
        // Callers should not pass invalid registers; debug builds catch this.
        debug_assert!(
            r < 64,
            "encoding() called with out-of-range register PhysReg({})",
            r
        );
        0
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Integer register constant values
    // -----------------------------------------------------------------------

    #[test]
    fn test_integer_register_indices() {
        assert_eq!(X0.0, 0);
        assert_eq!(X1.0, 1);
        assert_eq!(X2.0, 2);
        assert_eq!(X3.0, 3);
        assert_eq!(X4.0, 4);
        assert_eq!(X5.0, 5);
        assert_eq!(X10.0, 10);
        assert_eq!(X17.0, 17);
        assert_eq!(X27.0, 27);
        assert_eq!(X31.0, 31);
    }

    // -----------------------------------------------------------------------
    // ABI alias correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_abi_aliases() {
        assert_eq!(ZERO, X0);
        assert_eq!(RA, X1);
        assert_eq!(SP, X2);
        assert_eq!(GP, X3);
        assert_eq!(TP, X4);
        assert_eq!(T0, X5);
        assert_eq!(T1, X6);
        assert_eq!(T2, X7);
        assert_eq!(S0, X8);
        assert_eq!(FP, X8);
        assert_eq!(S1, X9);
        assert_eq!(A0, X10);
        assert_eq!(A7, X17);
        assert_eq!(S2, X18);
        assert_eq!(S11, X27);
        assert_eq!(T3, X28);
        assert_eq!(T6, X31);
    }

    // -----------------------------------------------------------------------
    // Floating-point register constant values
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp_register_indices() {
        assert_eq!(F0.0, 32);
        assert_eq!(F1.0, 33);
        assert_eq!(F10.0, 42);
        assert_eq!(F17.0, 49);
        assert_eq!(F27.0, 59);
        assert_eq!(F31.0, 63);
    }

    #[test]
    fn test_fp_abi_aliases() {
        assert_eq!(FT0, F0);
        assert_eq!(FT7, F7);
        assert_eq!(FT8, F28);
        assert_eq!(FT11, F31);
        assert_eq!(FS0, F8);
        assert_eq!(FS1, F9);
        assert_eq!(FS2, F18);
        assert_eq!(FS11, F27);
        assert_eq!(FA0, F10);
        assert_eq!(FA7, F17);
    }

    // -----------------------------------------------------------------------
    // Classification arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_integer_arg_regs() {
        assert_eq!(INTEGER_ARG_REGS.len(), 8);
        assert_eq!(INTEGER_ARG_REGS[0], A0);
        assert_eq!(INTEGER_ARG_REGS[7], A7);
    }

    #[test]
    fn test_float_arg_regs() {
        assert_eq!(FLOAT_ARG_REGS.len(), 8);
        assert_eq!(FLOAT_ARG_REGS[0], FA0);
        assert_eq!(FLOAT_ARG_REGS[7], FA7);
    }

    #[test]
    fn test_callee_saved_int() {
        assert_eq!(CALLEE_SAVED_INT.len(), 12);
        assert_eq!(CALLEE_SAVED_INT[0], S0);
        assert_eq!(CALLEE_SAVED_INT[11], S11);
        for reg in &CALLEE_SAVED_INT {
            assert!(is_callee_saved(*reg), "Expected callee-saved: {:?}", reg);
        }
    }

    #[test]
    fn test_callee_saved_fp() {
        assert_eq!(CALLEE_SAVED_FP.len(), 12);
        assert_eq!(CALLEE_SAVED_FP[0], FS0);
        assert_eq!(CALLEE_SAVED_FP[11], FS11);
        for reg in &CALLEE_SAVED_FP {
            assert!(is_callee_saved(*reg), "Expected callee-saved FP: {:?}", reg);
        }
    }

    #[test]
    fn test_caller_saved_int() {
        assert_eq!(CALLER_SAVED_INT.len(), 16);
        // All caller-saved integer regs must NOT be callee-saved
        for reg in &CALLER_SAVED_INT {
            assert!(!is_callee_saved(*reg), "Unexpected callee-saved in CALLER_SAVED_INT: {:?}", reg);
        }
    }

    #[test]
    fn test_caller_saved_fp() {
        assert_eq!(CALLER_SAVED_FP.len(), 20);
        for reg in &CALLER_SAVED_FP {
            assert!(!is_callee_saved(*reg), "Unexpected callee-saved in CALLER_SAVED_FP: {:?}", reg);
        }
    }

    #[test]
    fn test_allocatable_int() {
        assert_eq!(ALLOCATABLE_INT.len(), 28);
        // x0, x2, x3, x4 must not appear
        for reg in &ALLOCATABLE_INT {
            assert!(is_allocatable(*reg), "Non-allocatable in ALLOCATABLE_INT: {:?}", reg);
            assert_ne!(*reg, X0);
            assert_ne!(*reg, X2);
            assert_ne!(*reg, X3);
            assert_ne!(*reg, X4);
        }
    }

    #[test]
    fn test_allocatable_fp() {
        assert_eq!(ALLOCATABLE_FP.len(), 32);
        for reg in &ALLOCATABLE_FP {
            assert!(is_allocatable(*reg), "Non-allocatable in ALLOCATABLE_FP: {:?}", reg);
        }
    }

    // -----------------------------------------------------------------------
    // Name lookups
    // -----------------------------------------------------------------------

    #[test]
    fn test_int_reg_name() {
        assert_eq!(int_reg_name(X0), "zero");
        assert_eq!(int_reg_name(RA), "ra");
        assert_eq!(int_reg_name(SP), "sp");
        assert_eq!(int_reg_name(GP), "gp");
        assert_eq!(int_reg_name(TP), "tp");
        assert_eq!(int_reg_name(T0), "t0");
        assert_eq!(int_reg_name(S0), "s0");
        assert_eq!(int_reg_name(A0), "a0");
        assert_eq!(int_reg_name(A7), "a7");
        assert_eq!(int_reg_name(S11), "s11");
        assert_eq!(int_reg_name(T6), "t6");
    }

    #[test]
    fn test_fp_reg_name() {
        assert_eq!(fp_reg_name(FT0), "ft0");
        assert_eq!(fp_reg_name(FS0), "fs0");
        assert_eq!(fp_reg_name(FA0), "fa0");
        assert_eq!(fp_reg_name(FA7), "fa7");
        assert_eq!(fp_reg_name(FS11), "fs11");
        assert_eq!(fp_reg_name(FT11), "ft11");
    }

    #[test]
    fn test_reg_name_dispatch() {
        assert_eq!(reg_name(SP), "sp");
        assert_eq!(reg_name(FA0), "fa0");
        assert_eq!(reg_name(X0), "zero");
        assert_eq!(reg_name(F31), "ft11");
    }

    // -----------------------------------------------------------------------
    // Property queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_integer_reg() {
        assert!(is_integer_reg(X0));
        assert!(is_integer_reg(X31));
        assert!(!is_integer_reg(F0));
        assert!(!is_integer_reg(F31));
    }

    #[test]
    fn test_is_float_reg() {
        assert!(!is_float_reg(X0));
        assert!(!is_float_reg(X31));
        assert!(is_float_reg(F0));
        assert!(is_float_reg(F31));
    }

    #[test]
    fn test_is_callee_saved_coverage() {
        // Non-callee-saved integer registers
        assert!(!is_callee_saved(X0));   // zero
        assert!(!is_callee_saved(RA));   // ra
        assert!(!is_callee_saved(SP));   // sp
        assert!(!is_callee_saved(T0));   // t0
        assert!(!is_callee_saved(A0));   // a0
        // Callee-saved integer registers
        assert!(is_callee_saved(S0));
        assert!(is_callee_saved(S1));
        assert!(is_callee_saved(S11));
        // Non-callee-saved FP registers
        assert!(!is_callee_saved(FT0));
        assert!(!is_callee_saved(FA0));
        // Callee-saved FP registers
        assert!(is_callee_saved(FS0));
        assert!(is_callee_saved(FS11));
    }

    #[test]
    fn test_is_allocatable_special_regs() {
        assert!(!is_allocatable(X0));  // zero
        assert!(!is_allocatable(SP));  // sp
        assert!(!is_allocatable(GP));  // gp
        assert!(!is_allocatable(TP));  // tp
        assert!(is_allocatable(RA));   // ra is allocatable
        assert!(is_allocatable(T0));
        assert!(is_allocatable(A0));
        assert!(is_allocatable(S0));
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoding_integer() {
        assert_eq!(encoding(X0), 0);
        assert_eq!(encoding(X1), 1);
        assert_eq!(encoding(X31), 31);
        assert_eq!(encoding(A0), 10);
        assert_eq!(encoding(S0), 8);
    }

    #[test]
    fn test_encoding_fp() {
        assert_eq!(encoding(F0), 0);
        assert_eq!(encoding(F1), 1);
        assert_eq!(encoding(F31), 31);
        assert_eq!(encoding(FA0), 10);
        assert_eq!(encoding(FS0), 8);
    }

    // -----------------------------------------------------------------------
    // CSR constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_csr_addresses() {
        assert_eq!(CSR_FFLAGS, 0x001);
        assert_eq!(CSR_FRM, 0x002);
        assert_eq!(CSR_FCSR, 0x003);
    }

    // -----------------------------------------------------------------------
    // Cross-validation: every allocatable register has a valid encoding
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_allocatable_have_valid_encoding() {
        for reg in &ALLOCATABLE_INT {
            let enc = encoding(*reg);
            assert!(enc < 32, "Invalid encoding {} for {:?}", enc, reg);
        }
        for reg in &ALLOCATABLE_FP {
            let enc = encoding(*reg);
            assert!(enc < 32, "Invalid encoding {} for {:?}", enc, reg);
        }
    }
}
