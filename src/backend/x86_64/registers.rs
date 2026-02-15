//! x86-64 physical register definitions for the BCC compiler.
//!
//! This module defines constants for every physical register in the x86-64
//! architecture that BCC's register allocator, instruction encoder, and ABI
//! modules need to reference. It is the foundational register definition
//! used by all other x86-64 backend modules.
//!
//! # Register Namespace Layout
//!
//! The x86-64 register file is mapped into a flat `PhysReg(u16)` namespace:
//!
//! | Range    | Registers    | Count |
//! |----------|-------------|-------|
//! | 0–15     | GPRs (RAX–R15) | 16 |
//! | 16–31    | SSE  (XMM0–XMM15) | 16 |
//!
//! 32-bit GPR aliases (EAX–R15D) share the same `PhysReg` values as their
//! 64-bit counterparts (RAX–R15). The distinction between operand sizes is
//! handled at the instruction encoding level, not at the register allocation
//! level.
//!
//! # Register Encoding
//!
//! For ModR/M and SIB byte construction, the 3-bit register encoding is
//! derived from `PhysReg.0 & 0x07`. Registers R8–R15 and XMM8–XMM15
//! require a REX prefix (REX.B, REX.R, or REX.X bit set) to address the
//! extended register file.
//!
//! # System V AMD64 ABI Classification
//!
//! Register sets for the System V AMD64 calling convention:
//!
//! - **Integer argument registers:** RDI, RSI, RDX, RCX, R8, R9 (in order)
//! - **Floating-point argument registers:** XMM0–XMM7 (in order)
//! - **Callee-saved (preserved across calls):** RBX, RBP, R12–R15
//! - **Caller-saved (scratch/volatile):** RAX, RCX, RDX, RSI, RDI, R8–R11
//! - **Stack pointer:** RSP — not allocatable
//! - **Return value:** RAX (integer), XMM0 (floating-point)

use crate::backend::traits::PhysReg;

// ---------------------------------------------------------------------------
// Register Count Constants
// ---------------------------------------------------------------------------

/// Total number of general-purpose registers in x86-64 (RAX through R15).
pub const NUM_GPRS: usize = 16;

/// Total number of SSE registers in x86-64 (XMM0 through XMM15).
pub const NUM_SSE: usize = 16;

/// Total number of physical registers tracked by the x86-64 backend
/// (16 GPRs + 16 SSE = 32).
pub const TOTAL_REGS: usize = 32;

// ---------------------------------------------------------------------------
// 64-bit General Purpose Register Constants (GPRs)
// ---------------------------------------------------------------------------
// The encoding order matches the x86-64 register encoding in ModR/M:
//   RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
//   R8=8, R9=9, R10=10, R11=11, R12=12, R13=13, R14=14, R15=15

/// RAX — accumulator, return value register, caller-saved.
pub const RAX: PhysReg = PhysReg(0);
/// RCX — fourth integer argument, caller-saved.
pub const RCX: PhysReg = PhysReg(1);
/// RDX — third integer argument, caller-saved.
pub const RDX: PhysReg = PhysReg(2);
/// RBX — callee-saved general-purpose register.
pub const RBX: PhysReg = PhysReg(3);
/// RSP — stack pointer, NOT allocatable by the register allocator.
pub const RSP: PhysReg = PhysReg(4);
/// RBP — frame pointer, callee-saved.
pub const RBP: PhysReg = PhysReg(5);
/// RSI — second integer argument, caller-saved.
pub const RSI: PhysReg = PhysReg(6);
/// RDI — first integer argument, caller-saved.
pub const RDI: PhysReg = PhysReg(7);
/// R8 — fifth integer argument, caller-saved.
pub const R8: PhysReg = PhysReg(8);
/// R9 — sixth integer argument, caller-saved.
pub const R9: PhysReg = PhysReg(9);
/// R10 — caller-saved scratch register.
pub const R10: PhysReg = PhysReg(10);
/// R11 — caller-saved scratch register, used for indirect calls.
pub const R11: PhysReg = PhysReg(11);
/// R12 — callee-saved general-purpose register.
pub const R12: PhysReg = PhysReg(12);
/// R13 — callee-saved general-purpose register.
pub const R13: PhysReg = PhysReg(13);
/// R14 — callee-saved general-purpose register.
pub const R14: PhysReg = PhysReg(14);
/// R15 — callee-saved general-purpose register.
pub const R15: PhysReg = PhysReg(15);

// ---------------------------------------------------------------------------
// 32-bit General Purpose Register Aliases
// ---------------------------------------------------------------------------
// These share the same PhysReg encoding as their 64-bit counterparts.
// The distinction between EAX and RAX is handled at the instruction
// encoding level (operand-size override prefix or REX.W absence),
// not at the register allocation level.

/// EAX — 32-bit alias for RAX.
pub const EAX: PhysReg = PhysReg(0);
/// ECX — 32-bit alias for RCX.
pub const ECX: PhysReg = PhysReg(1);
/// EDX — 32-bit alias for RDX.
pub const EDX: PhysReg = PhysReg(2);
/// EBX — 32-bit alias for RBX.
pub const EBX: PhysReg = PhysReg(3);
/// ESP — 32-bit alias for RSP.
pub const ESP: PhysReg = PhysReg(4);
/// EBP — 32-bit alias for RBP.
pub const EBP: PhysReg = PhysReg(5);
/// ESI — 32-bit alias for RSI.
pub const ESI: PhysReg = PhysReg(6);
/// EDI — 32-bit alias for RDI.
pub const EDI: PhysReg = PhysReg(7);
/// R8D — 32-bit alias for R8.
pub const R8D: PhysReg = PhysReg(8);
/// R9D — 32-bit alias for R9.
pub const R9D: PhysReg = PhysReg(9);
/// R10D — 32-bit alias for R10.
pub const R10D: PhysReg = PhysReg(10);
/// R11D — 32-bit alias for R11.
pub const R11D: PhysReg = PhysReg(11);
/// R12D — 32-bit alias for R12.
pub const R12D: PhysReg = PhysReg(12);
/// R13D — 32-bit alias for R13.
pub const R13D: PhysReg = PhysReg(13);
/// R14D — 32-bit alias for R14.
pub const R14D: PhysReg = PhysReg(14);
/// R15D — 32-bit alias for R15.
pub const R15D: PhysReg = PhysReg(15);

// ---------------------------------------------------------------------------
// SSE Register Constants (XMM0–XMM15)
// ---------------------------------------------------------------------------
// SSE registers are offset by 16 from GPR numbering to avoid collision
// within the PhysReg namespace. The actual hardware encoding (for VEX/EVEX
// prefix construction) is obtained via `reg_index()` which subtracts the
// offset.

/// XMM0 — first SSE register, first FP argument, FP return value.
pub const XMM0: PhysReg = PhysReg(16);
/// XMM1 — second FP argument register.
pub const XMM1: PhysReg = PhysReg(17);
/// XMM2 — third FP argument register.
pub const XMM2: PhysReg = PhysReg(18);
/// XMM3 — fourth FP argument register.
pub const XMM3: PhysReg = PhysReg(19);
/// XMM4 — fifth FP argument register.
pub const XMM4: PhysReg = PhysReg(20);
/// XMM5 — sixth FP argument register.
pub const XMM5: PhysReg = PhysReg(21);
/// XMM6 — seventh FP argument register.
pub const XMM6: PhysReg = PhysReg(22);
/// XMM7 — eighth FP argument register.
pub const XMM7: PhysReg = PhysReg(23);
/// XMM8 — SSE register, requires REX prefix for encoding.
pub const XMM8: PhysReg = PhysReg(24);
/// XMM9 — SSE register, requires REX prefix for encoding.
pub const XMM9: PhysReg = PhysReg(25);
/// XMM10 — SSE register, requires REX prefix for encoding.
pub const XMM10: PhysReg = PhysReg(26);
/// XMM11 — SSE register, requires REX prefix for encoding.
pub const XMM11: PhysReg = PhysReg(27);
/// XMM12 — SSE register, requires REX prefix for encoding.
pub const XMM12: PhysReg = PhysReg(28);
/// XMM13 — SSE register, requires REX prefix for encoding.
pub const XMM13: PhysReg = PhysReg(29);
/// XMM14 — SSE register, requires REX prefix for encoding.
pub const XMM14: PhysReg = PhysReg(30);
/// XMM15 — SSE register, requires REX prefix for encoding.
pub const XMM15: PhysReg = PhysReg(31);

// ---------------------------------------------------------------------------
// Register Classification Sets
// ---------------------------------------------------------------------------

/// Callee-saved registers — must be preserved across function calls.
///
/// Per the System V AMD64 ABI, these registers must be saved in the
/// function prologue and restored in the epilogue if the function
/// modifies them. The register allocator uses this information to
/// estimate spill costs (callee-saved registers incur prologue/epilogue
/// save/restore overhead only if actually used).
pub const CALLEE_SAVED: &[PhysReg] = &[RBX, RBP, R12, R13, R14, R15];

/// Caller-saved (volatile/scratch) registers — may be clobbered by
/// any function call.
///
/// The register allocator must spill any live value in these registers
/// across a call instruction. These registers are freely usable within
/// a function without prologue/epilogue save/restore overhead.
pub const CALLER_SAVED: &[PhysReg] = &[RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11];

/// Integer argument registers in System V AMD64 calling convention order.
///
/// The first six integer/pointer arguments are passed in these registers,
/// in order: RDI (1st), RSI (2nd), RDX (3rd), RCX (4th), R8 (5th), R9 (6th).
/// Additional integer arguments are passed on the stack.
pub const ARG_REGS_INT: &[PhysReg] = &[RDI, RSI, RDX, RCX, R8, R9];

/// Floating-point argument registers in System V AMD64 calling convention order.
///
/// The first eight floating-point arguments (float, double) are passed in
/// XMM0 through XMM7, in order. Additional FP arguments are passed on
/// the stack.
pub const ARG_REGS_FLOAT: &[PhysReg] = &[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];

/// Allocatable general-purpose registers — all GPRs except RSP.
///
/// RSP is excluded because it serves as the hardware stack pointer and
/// must not be repurposed by the register allocator. RBP is included
/// here because frame-pointer omission is possible (though callee-saved
/// status still applies when RBP is used).
pub const ALLOCATABLE_GPRS: &[PhysReg] = &[
    RAX, RCX, RDX, RBX, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
];

/// Allocatable SSE registers — all XMM0 through XMM15.
///
/// All 16 SSE registers are available for the register allocator. On
/// System V AMD64, all XMM registers are caller-saved (volatile).
pub const ALLOCATABLE_SSE: &[PhysReg] = &[
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14,
    XMM15,
];

// ---------------------------------------------------------------------------
// Register Name Lookup Tables (private)
// ---------------------------------------------------------------------------

/// 64-bit GPR names indexed by register number 0–15.
const GPR_NAMES_64: [&str; NUM_GPRS] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// 32-bit GPR names indexed by register number 0–15.
const GPR_NAMES_32: [&str; NUM_GPRS] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];

/// 16-bit GPR names indexed by register number 0–15.
const GPR_NAMES_16: [&str; NUM_GPRS] = [
    "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w", "r13w",
    "r14w", "r15w",
];

/// 8-bit GPR names indexed by register number 0–15.
///
/// For registers 0–3 (AL, CL, DL, BL), these are the low-byte names.
/// For registers 4–7 in 64-bit mode with REX prefix, SPL/BPL/SIL/DIL
/// are used instead of AH/CH/DH/BH. Registers 8–15 use R8B–R15B.
const GPR_NAMES_8: [&str; NUM_GPRS] = [
    "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
    "r13b", "r14b", "r15b",
];

/// SSE register names indexed by SSE register number 0–15.
const SSE_NAMES: [&str; NUM_SSE] = [
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
    "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

// ---------------------------------------------------------------------------
// Register Encoding Helper Functions
// ---------------------------------------------------------------------------

/// Returns the 3-bit register encoding for ModR/M and SIB byte construction.
///
/// In x86-64 encoding, registers are identified by a 3-bit field within
/// the ModR/M byte (and optionally SIB byte). Registers R8–R15 share
/// encodings 0–7 with RAX–RDI but require REX.B/REX.R/REX.X to
/// distinguish them.
///
/// # Arguments
///
/// * `reg` — A GPR `PhysReg` with value in the range 0–15.
///
/// # Returns
///
/// The low 3 bits of the register number (0–7), suitable for embedding
/// directly into a ModR/M or SIB byte.
///
/// # Panics
///
/// Panics if `reg` is not a GPR (PhysReg 0–15).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(gpr_encoding(RAX), 0); // RAX = 0b000
/// assert_eq!(gpr_encoding(RDI), 7); // RDI = 0b111
/// assert_eq!(gpr_encoding(R8),  0); // R8  = 0b000 (with REX.B)
/// assert_eq!(gpr_encoding(R15), 7); // R15 = 0b111 (with REX.B)
/// ```
#[inline]
pub fn gpr_encoding(reg: PhysReg) -> u8 {
    debug_assert!(
        is_gpr(reg),
        "gpr_encoding called with non-GPR register: PhysReg({})",
        reg.0
    );
    (reg.0 as u8) & 0x07
}

/// Returns `true` if the given register requires a REX prefix for encoding.
///
/// A REX prefix is required to address registers in the extended range:
/// - GPRs R8–R15 (PhysReg 8–15)
/// - SSE registers XMM8–XMM15 (PhysReg 24–31)
///
/// The REX prefix bit placement (REX.B, REX.R, or REX.X) depends on
/// which operand field the register occupies in the instruction encoding.
///
/// # Examples
///
/// ```ignore
/// assert!(!needs_rex(RAX));   // RAX = 0, no REX needed
/// assert!(!needs_rex(RDI));   // RDI = 7, no REX needed
/// assert!(needs_rex(R8));     // R8  = 8, REX.B required
/// assert!(needs_rex(R15));    // R15 = 15, REX.B required
/// assert!(!needs_rex(XMM0));  // XMM0 = 16, no REX for low SSE
/// assert!(needs_rex(XMM8));   // XMM8 = 24, REX required
/// ```
#[inline]
pub fn needs_rex(reg: PhysReg) -> bool {
    let idx = reg.0;
    // GPR range: R8–R15 (indices 8–15)
    // SSE range: XMM8–XMM15 (indices 24–31)
    (8..=15).contains(&idx) || (24..=31).contains(&idx)
}

/// Returns `true` if the given `PhysReg` is a general-purpose register (GPR).
///
/// GPRs occupy PhysReg indices 0–15 (RAX through R15).
///
/// # Examples
///
/// ```ignore
/// assert!(is_gpr(RAX));    // PhysReg(0) is a GPR
/// assert!(is_gpr(R15));    // PhysReg(15) is a GPR
/// assert!(!is_gpr(XMM0));  // PhysReg(16) is SSE, not GPR
/// ```
#[inline]
pub fn is_gpr(reg: PhysReg) -> bool {
    reg.0 < (NUM_GPRS as u16)
}

/// Returns `true` if the given `PhysReg` is an SSE register.
///
/// SSE registers occupy PhysReg indices 16–31 (XMM0 through XMM15).
///
/// # Examples
///
/// ```ignore
/// assert!(!is_sse(RAX));    // PhysReg(0) is GPR, not SSE
/// assert!(is_sse(XMM0));    // PhysReg(16) is SSE
/// assert!(is_sse(XMM15));   // PhysReg(31) is SSE
/// ```
#[inline]
pub fn is_sse(reg: PhysReg) -> bool {
    let idx = reg.0;
    idx >= (NUM_GPRS as u16) && idx < (TOTAL_REGS as u16)
}

/// Returns the 0-based index within the register's class (0–15).
///
/// For GPRs (PhysReg 0–15), this returns the PhysReg value directly.
/// For SSE registers (PhysReg 16–31), this subtracts the SSE offset (16)
/// to yield the hardware XMM register number.
///
/// This is useful for indexing into per-class arrays and for computing
/// hardware register encodings in VEX/EVEX prefixes.
///
/// # Arguments
///
/// * `reg` — A `PhysReg` in the range 0–31.
///
/// # Returns
///
/// A value in the range 0–15 representing the register's position
/// within its class.
///
/// # Panics
///
/// Panics if `reg` is outside the valid x86-64 register range (0–31).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(reg_index(RAX),   0);   // RAX is GPR #0
/// assert_eq!(reg_index(R15),  15);   // R15 is GPR #15
/// assert_eq!(reg_index(XMM0),  0);   // XMM0 is SSE #0
/// assert_eq!(reg_index(XMM15), 15);  // XMM15 is SSE #15
/// ```
#[inline]
pub fn reg_index(reg: PhysReg) -> u8 {
    let idx = reg.0;
    debug_assert!(
        idx < (TOTAL_REGS as u16),
        "reg_index called with out-of-range register: PhysReg({})",
        idx
    );
    if idx < (NUM_GPRS as u16) {
        idx as u8
    } else {
        (idx - NUM_GPRS as u16) as u8
    }
}

// ---------------------------------------------------------------------------
// Register Name Functions
// ---------------------------------------------------------------------------

/// Returns the AT&T-syntax 64-bit register name for a GPR.
///
/// # Arguments
///
/// * `reg` — A GPR `PhysReg` with value in the range 0–15.
///
/// # Returns
///
/// A `&'static str` such as `"rax"`, `"rcx"`, ..., `"r15"`.
///
/// # Panics
///
/// Panics if `reg` is not a GPR (PhysReg 0–15).
#[inline]
pub fn gpr_name_64(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(
        idx < NUM_GPRS,
        "gpr_name_64 called with non-GPR register: PhysReg({})",
        reg.0
    );
    GPR_NAMES_64[idx]
}

/// Returns the AT&T-syntax 32-bit register name for a GPR.
///
/// # Arguments
///
/// * `reg` — A GPR `PhysReg` with value in the range 0–15.
///
/// # Returns
///
/// A `&'static str` such as `"eax"`, `"ecx"`, ..., `"r15d"`.
///
/// # Panics
///
/// Panics if `reg` is not a GPR (PhysReg 0–15).
#[inline]
pub fn gpr_name_32(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(
        idx < NUM_GPRS,
        "gpr_name_32 called with non-GPR register: PhysReg({})",
        reg.0
    );
    GPR_NAMES_32[idx]
}

/// Returns the AT&T-syntax 16-bit register name for a GPR.
///
/// # Arguments
///
/// * `reg` — A GPR `PhysReg` with value in the range 0–15.
///
/// # Returns
///
/// A `&'static str` such as `"ax"`, `"cx"`, ..., `"r15w"`.
///
/// # Panics
///
/// Panics if `reg` is not a GPR (PhysReg 0–15).
#[inline]
pub fn gpr_name_16(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(
        idx < NUM_GPRS,
        "gpr_name_16 called with non-GPR register: PhysReg({})",
        reg.0
    );
    GPR_NAMES_16[idx]
}

/// Returns the AT&T-syntax 8-bit register name for a GPR.
///
/// In 64-bit mode with REX prefix, registers RSP–RDI (indices 4–7) use
/// the SPL/BPL/SIL/DIL names rather than AH/CH/DH/BH. This module
/// always returns the REX-prefix-compatible names since BCC targets
/// 64-bit mode exclusively for x86-64.
///
/// # Arguments
///
/// * `reg` — A GPR `PhysReg` with value in the range 0–15.
///
/// # Returns
///
/// A `&'static str` such as `"al"`, `"cl"`, `"spl"`, ..., `"r15b"`.
///
/// # Panics
///
/// Panics if `reg` is not a GPR (PhysReg 0–15).
#[inline]
pub fn gpr_name_8(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    debug_assert!(
        idx < NUM_GPRS,
        "gpr_name_8 called with non-GPR register: PhysReg({})",
        reg.0
    );
    GPR_NAMES_8[idx]
}

/// Returns the AT&T-syntax name for an SSE register.
///
/// # Arguments
///
/// * `reg` — An SSE `PhysReg` with value in the range 16–31.
///
/// # Returns
///
/// A `&'static str` such as `"xmm0"`, `"xmm1"`, ..., `"xmm15"`.
///
/// # Panics
///
/// Panics if `reg` is not an SSE register (PhysReg 16–31).
#[inline]
pub fn sse_name(reg: PhysReg) -> &'static str {
    debug_assert!(
        is_sse(reg),
        "sse_name called with non-SSE register: PhysReg({})",
        reg.0
    );
    let idx = (reg.0 - NUM_GPRS as u16) as usize;
    SSE_NAMES[idx]
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GPR constant values -----------------------------------------------

    #[test]
    fn test_gpr_constants() {
        assert_eq!(RAX.0, 0);
        assert_eq!(RCX.0, 1);
        assert_eq!(RDX.0, 2);
        assert_eq!(RBX.0, 3);
        assert_eq!(RSP.0, 4);
        assert_eq!(RBP.0, 5);
        assert_eq!(RSI.0, 6);
        assert_eq!(RDI.0, 7);
        assert_eq!(R8.0, 8);
        assert_eq!(R9.0, 9);
        assert_eq!(R10.0, 10);
        assert_eq!(R11.0, 11);
        assert_eq!(R12.0, 12);
        assert_eq!(R13.0, 13);
        assert_eq!(R14.0, 14);
        assert_eq!(R15.0, 15);
    }

    // -- 32-bit alias identity with 64-bit GPRs ---------------------------

    #[test]
    fn test_32bit_aliases_match_64bit() {
        assert_eq!(EAX, RAX);
        assert_eq!(ECX, RCX);
        assert_eq!(EDX, RDX);
        assert_eq!(EBX, RBX);
        assert_eq!(ESP, RSP);
        assert_eq!(EBP, RBP);
        assert_eq!(ESI, RSI);
        assert_eq!(EDI, RDI);
        assert_eq!(R8D, R8);
        assert_eq!(R9D, R9);
        assert_eq!(R10D, R10);
        assert_eq!(R11D, R11);
        assert_eq!(R12D, R12);
        assert_eq!(R13D, R13);
        assert_eq!(R14D, R14);
        assert_eq!(R15D, R15);
    }

    // -- SSE constant values -----------------------------------------------

    #[test]
    fn test_sse_constants() {
        assert_eq!(XMM0.0, 16);
        assert_eq!(XMM1.0, 17);
        assert_eq!(XMM2.0, 18);
        assert_eq!(XMM3.0, 19);
        assert_eq!(XMM4.0, 20);
        assert_eq!(XMM5.0, 21);
        assert_eq!(XMM6.0, 22);
        assert_eq!(XMM7.0, 23);
        assert_eq!(XMM8.0, 24);
        assert_eq!(XMM9.0, 25);
        assert_eq!(XMM10.0, 26);
        assert_eq!(XMM11.0, 27);
        assert_eq!(XMM12.0, 28);
        assert_eq!(XMM13.0, 29);
        assert_eq!(XMM14.0, 30);
        assert_eq!(XMM15.0, 31);
    }

    // -- Count constants ---------------------------------------------------

    #[test]
    fn test_count_constants() {
        assert_eq!(NUM_GPRS, 16);
        assert_eq!(NUM_SSE, 16);
        assert_eq!(TOTAL_REGS, 32);
        assert_eq!(NUM_GPRS + NUM_SSE, TOTAL_REGS);
    }

    // -- is_gpr / is_sse classification ------------------------------------

    #[test]
    fn test_is_gpr() {
        for i in 0..16u16 {
            assert!(is_gpr(PhysReg(i)), "PhysReg({}) should be GPR", i);
        }
        for i in 16..32u16 {
            assert!(!is_gpr(PhysReg(i)), "PhysReg({}) should not be GPR", i);
        }
    }

    #[test]
    fn test_is_sse() {
        for i in 0..16u16 {
            assert!(!is_sse(PhysReg(i)), "PhysReg({}) should not be SSE", i);
        }
        for i in 16..32u16 {
            assert!(is_sse(PhysReg(i)), "PhysReg({}) should be SSE", i);
        }
    }

    // -- gpr_encoding ------------------------------------------------------

    #[test]
    fn test_gpr_encoding() {
        // Low 8 GPRs: encoding matches register index directly
        assert_eq!(gpr_encoding(RAX), 0);
        assert_eq!(gpr_encoding(RCX), 1);
        assert_eq!(gpr_encoding(RDX), 2);
        assert_eq!(gpr_encoding(RBX), 3);
        assert_eq!(gpr_encoding(RSP), 4);
        assert_eq!(gpr_encoding(RBP), 5);
        assert_eq!(gpr_encoding(RSI), 6);
        assert_eq!(gpr_encoding(RDI), 7);
        // High 8 GPRs: encoding wraps to 0-7 (REX.B extends)
        assert_eq!(gpr_encoding(R8), 0);
        assert_eq!(gpr_encoding(R9), 1);
        assert_eq!(gpr_encoding(R10), 2);
        assert_eq!(gpr_encoding(R11), 3);
        assert_eq!(gpr_encoding(R12), 4);
        assert_eq!(gpr_encoding(R13), 5);
        assert_eq!(gpr_encoding(R14), 6);
        assert_eq!(gpr_encoding(R15), 7);
    }

    // -- needs_rex ---------------------------------------------------------

    #[test]
    fn test_needs_rex() {
        // Low GPRs: no REX
        assert!(!needs_rex(RAX));
        assert!(!needs_rex(RCX));
        assert!(!needs_rex(RDX));
        assert!(!needs_rex(RBX));
        assert!(!needs_rex(RSP));
        assert!(!needs_rex(RBP));
        assert!(!needs_rex(RSI));
        assert!(!needs_rex(RDI));
        // High GPRs: REX required
        assert!(needs_rex(R8));
        assert!(needs_rex(R9));
        assert!(needs_rex(R10));
        assert!(needs_rex(R11));
        assert!(needs_rex(R12));
        assert!(needs_rex(R13));
        assert!(needs_rex(R14));
        assert!(needs_rex(R15));
        // Low SSE: no REX
        assert!(!needs_rex(XMM0));
        assert!(!needs_rex(XMM7));
        // High SSE: REX required
        assert!(needs_rex(XMM8));
        assert!(needs_rex(XMM15));
    }

    // -- reg_index ---------------------------------------------------------

    #[test]
    fn test_reg_index() {
        // GPRs: index equals PhysReg value
        assert_eq!(reg_index(RAX), 0);
        assert_eq!(reg_index(R15), 15);
        // SSE: index is PhysReg value minus 16
        assert_eq!(reg_index(XMM0), 0);
        assert_eq!(reg_index(XMM15), 15);
    }

    // -- Register name functions -------------------------------------------

    #[test]
    fn test_gpr_name_64() {
        assert_eq!(gpr_name_64(RAX), "rax");
        assert_eq!(gpr_name_64(RCX), "rcx");
        assert_eq!(gpr_name_64(RDX), "rdx");
        assert_eq!(gpr_name_64(RBX), "rbx");
        assert_eq!(gpr_name_64(RSP), "rsp");
        assert_eq!(gpr_name_64(RBP), "rbp");
        assert_eq!(gpr_name_64(RSI), "rsi");
        assert_eq!(gpr_name_64(RDI), "rdi");
        assert_eq!(gpr_name_64(R8), "r8");
        assert_eq!(gpr_name_64(R9), "r9");
        assert_eq!(gpr_name_64(R10), "r10");
        assert_eq!(gpr_name_64(R11), "r11");
        assert_eq!(gpr_name_64(R12), "r12");
        assert_eq!(gpr_name_64(R13), "r13");
        assert_eq!(gpr_name_64(R14), "r14");
        assert_eq!(gpr_name_64(R15), "r15");
    }

    #[test]
    fn test_gpr_name_32() {
        assert_eq!(gpr_name_32(RAX), "eax");
        assert_eq!(gpr_name_32(RCX), "ecx");
        assert_eq!(gpr_name_32(RDX), "edx");
        assert_eq!(gpr_name_32(RBX), "ebx");
        assert_eq!(gpr_name_32(RSP), "esp");
        assert_eq!(gpr_name_32(RBP), "ebp");
        assert_eq!(gpr_name_32(RSI), "esi");
        assert_eq!(gpr_name_32(RDI), "edi");
        assert_eq!(gpr_name_32(R8), "r8d");
        assert_eq!(gpr_name_32(R15), "r15d");
    }

    #[test]
    fn test_gpr_name_16() {
        assert_eq!(gpr_name_16(RAX), "ax");
        assert_eq!(gpr_name_16(RCX), "cx");
        assert_eq!(gpr_name_16(RDX), "dx");
        assert_eq!(gpr_name_16(RBX), "bx");
        assert_eq!(gpr_name_16(RSP), "sp");
        assert_eq!(gpr_name_16(RBP), "bp");
        assert_eq!(gpr_name_16(RSI), "si");
        assert_eq!(gpr_name_16(RDI), "di");
        assert_eq!(gpr_name_16(R8), "r8w");
        assert_eq!(gpr_name_16(R15), "r15w");
    }

    #[test]
    fn test_gpr_name_8() {
        assert_eq!(gpr_name_8(RAX), "al");
        assert_eq!(gpr_name_8(RCX), "cl");
        assert_eq!(gpr_name_8(RDX), "dl");
        assert_eq!(gpr_name_8(RBX), "bl");
        // In 64-bit mode with REX, these are SPL/BPL/SIL/DIL
        assert_eq!(gpr_name_8(RSP), "spl");
        assert_eq!(gpr_name_8(RBP), "bpl");
        assert_eq!(gpr_name_8(RSI), "sil");
        assert_eq!(gpr_name_8(RDI), "dil");
        assert_eq!(gpr_name_8(R8), "r8b");
        assert_eq!(gpr_name_8(R15), "r15b");
    }

    #[test]
    fn test_sse_name() {
        assert_eq!(sse_name(XMM0), "xmm0");
        assert_eq!(sse_name(XMM1), "xmm1");
        assert_eq!(sse_name(XMM7), "xmm7");
        assert_eq!(sse_name(XMM8), "xmm8");
        assert_eq!(sse_name(XMM15), "xmm15");
    }

    // -- Classification set membership -------------------------------------

    #[test]
    fn test_callee_saved_set() {
        assert_eq!(CALLEE_SAVED.len(), 6);
        assert!(CALLEE_SAVED.contains(&RBX));
        assert!(CALLEE_SAVED.contains(&RBP));
        assert!(CALLEE_SAVED.contains(&R12));
        assert!(CALLEE_SAVED.contains(&R13));
        assert!(CALLEE_SAVED.contains(&R14));
        assert!(CALLEE_SAVED.contains(&R15));
        // Caller-saved registers must NOT be in callee-saved set
        assert!(!CALLEE_SAVED.contains(&RAX));
        assert!(!CALLEE_SAVED.contains(&RCX));
        assert!(!CALLEE_SAVED.contains(&RSP));
    }

    #[test]
    fn test_caller_saved_set() {
        assert_eq!(CALLER_SAVED.len(), 9);
        assert!(CALLER_SAVED.contains(&RAX));
        assert!(CALLER_SAVED.contains(&RCX));
        assert!(CALLER_SAVED.contains(&RDX));
        assert!(CALLER_SAVED.contains(&RSI));
        assert!(CALLER_SAVED.contains(&RDI));
        assert!(CALLER_SAVED.contains(&R8));
        assert!(CALLER_SAVED.contains(&R9));
        assert!(CALLER_SAVED.contains(&R10));
        assert!(CALLER_SAVED.contains(&R11));
        // Callee-saved registers must NOT be in caller-saved set
        assert!(!CALLER_SAVED.contains(&RBX));
        assert!(!CALLER_SAVED.contains(&RBP));
        assert!(!CALLER_SAVED.contains(&R12));
    }

    #[test]
    fn test_arg_regs_int() {
        // System V AMD64 integer argument order: RDI, RSI, RDX, RCX, R8, R9
        assert_eq!(ARG_REGS_INT.len(), 6);
        assert_eq!(ARG_REGS_INT[0], RDI);
        assert_eq!(ARG_REGS_INT[1], RSI);
        assert_eq!(ARG_REGS_INT[2], RDX);
        assert_eq!(ARG_REGS_INT[3], RCX);
        assert_eq!(ARG_REGS_INT[4], R8);
        assert_eq!(ARG_REGS_INT[5], R9);
    }

    #[test]
    fn test_arg_regs_float() {
        assert_eq!(ARG_REGS_FLOAT.len(), 8);
        assert_eq!(ARG_REGS_FLOAT[0], XMM0);
        assert_eq!(ARG_REGS_FLOAT[7], XMM7);
    }

    #[test]
    fn test_allocatable_gprs_excludes_rsp() {
        assert_eq!(ALLOCATABLE_GPRS.len(), 15); // 16 GPRs minus RSP
        assert!(!ALLOCATABLE_GPRS.contains(&RSP));
        // All other GPRs must be present
        for &reg in &[
            RAX, RCX, RDX, RBX, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
        ] {
            assert!(
                ALLOCATABLE_GPRS.contains(&reg),
                "ALLOCATABLE_GPRS missing {:?}",
                reg
            );
        }
    }

    #[test]
    fn test_allocatable_sse_contains_all() {
        assert_eq!(ALLOCATABLE_SSE.len(), 16);
        for i in 0..16u16 {
            let reg = PhysReg(16 + i);
            assert!(
                ALLOCATABLE_SSE.contains(&reg),
                "ALLOCATABLE_SSE missing XMM{}",
                i
            );
        }
    }

    // -- Disjointness checks -----------------------------------------------

    #[test]
    fn test_caller_callee_disjoint() {
        // No register should appear in both caller-saved and callee-saved sets
        for &reg in CALLEE_SAVED {
            assert!(
                !CALLER_SAVED.contains(&reg),
                "Register {:?} in both callee-saved and caller-saved",
                reg
            );
        }
    }

    #[test]
    fn test_gpr_sse_non_overlapping() {
        // GPR and SSE PhysReg ranges must not overlap
        for i in 0..NUM_GPRS as u16 {
            let reg = PhysReg(i);
            assert!(is_gpr(reg));
            assert!(!is_sse(reg));
        }
        for i in NUM_GPRS as u16..TOTAL_REGS as u16 {
            let reg = PhysReg(i);
            assert!(!is_gpr(reg));
            assert!(is_sse(reg));
        }
    }
}
