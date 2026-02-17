//! AArch64 (ARM 64-bit) physical register definitions for the BCC compiler.
//!
//! This module provides named constants for the entire AArch64 register file:
//!
//! - **General-purpose registers (X0–X30):** 64-bit integer registers, with X0–X7
//!   serving as argument/return registers per AAPCS64, X8 as the indirect result
//!   location register, X9–X15 as temporaries, X16–X17 as intra-procedure-call
//!   scratch registers (IP0/IP1), X18 as the platform register, X19–X28 as
//!   callee-saved registers, X29 as the frame pointer (FP), and X30 as the
//!   link register (LR).
//!
//! - **W-register views (W0–W30):** 32-bit views of the lower half of X0–X30.
//!   Writing a W-register zero-extends the result to the full 64-bit X-register.
//!
//! - **Special registers:** SP (stack pointer), XZR/WZR (zero registers that
//!   read as zero and discard writes). In A64 encoding, SP and XZR share
//!   register number 31 but are distinguished by instruction context.
//!
//! - **SIMD/FP registers (V0–V31):** 128-bit vector registers with multiple
//!   sub-register views: Bn (byte, 8-bit), Hn (halfword, 16-bit), Sn (single,
//!   32-bit), Dn (double, 64-bit), Qn (quad, 128-bit). Only the lower 64 bits
//!   (D8–D15) of V8–V15 are callee-saved per AAPCS64.
//!
//! - **NZCV condition codes:** 4-bit condition encodings for conditional
//!   branches (`B.cond`), conditional compare (`CCMP`/`CCMN`), and conditional
//!   select (`CSEL`/`CSINC`/`CSINV`/`CSNEG`) instructions.
//!
//! # PhysReg Encoding Scheme
//!
//! | PhysReg Range | Registers       | Description              |
//! |---------------|-----------------|--------------------------|
//! | 0–30          | X0–X30          | 64-bit GPRs              |
//! | 31            | SP              | Stack pointer            |
//! | 32            | XZR             | 64-bit zero register     |
//! | 33            | WZR             | 32-bit zero register     |
//! | 34–64         | W0–W30          | 32-bit GPR views         |
//! | 65            | WSP             | 32-bit stack pointer     |
//! | 66–97         | V0–V31          | 128-bit SIMD/FP          |
//! | 98–129        | S0–S31          | 32-bit FP views          |
//! | 130–161       | D0–D31          | 64-bit FP views          |
//!
//! # A64 Instruction Encoding
//!
//! All A64 instructions use a 5-bit register field (bits `[4:0]`). The
//! [`encoding`] function extracts the correct 5-bit value (0–31) from any
//! `PhysReg` variant. Note that SP and XZR both encode as 31 — the
//! instruction opcode determines which is used.

use crate::backend::traits::PhysReg;

// ===========================================================================
// General-Purpose Registers (X0–X30) — 64-bit
// ===========================================================================

/// X0: argument/return register 0 (first integer return value per AAPCS64).
pub const X0: PhysReg = PhysReg(0);
/// X1: argument/return register 1 (second part of 128-bit return value).
pub const X1: PhysReg = PhysReg(1);
/// X2: argument register 2.
pub const X2: PhysReg = PhysReg(2);
/// X3: argument register 3.
pub const X3: PhysReg = PhysReg(3);
/// X4: argument register 4.
pub const X4: PhysReg = PhysReg(4);
/// X5: argument register 5.
pub const X5: PhysReg = PhysReg(5);
/// X6: argument register 6.
pub const X6: PhysReg = PhysReg(6);
/// X7: argument register 7 (last integer argument register).
pub const X7: PhysReg = PhysReg(7);
/// X8: indirect result location register — pointer for large struct returns.
pub const X8: PhysReg = PhysReg(8);
/// X9: caller-saved temporary register.
pub const X9: PhysReg = PhysReg(9);
/// X10: caller-saved temporary register.
pub const X10: PhysReg = PhysReg(10);
/// X11: caller-saved temporary register.
pub const X11: PhysReg = PhysReg(11);
/// X12: caller-saved temporary register.
pub const X12: PhysReg = PhysReg(12);
/// X13: caller-saved temporary register.
pub const X13: PhysReg = PhysReg(13);
/// X14: caller-saved temporary register.
pub const X14: PhysReg = PhysReg(14);
/// X15: caller-saved temporary register.
pub const X15: PhysReg = PhysReg(15);
/// X16: intra-procedure-call scratch register 1 (IP0). May be corrupted by
/// linker-inserted PLT/veneer stubs.
pub const X16: PhysReg = PhysReg(16);
/// X17: intra-procedure-call scratch register 2 (IP1). May be corrupted by
/// linker-inserted PLT/veneer stubs.
pub const X17: PhysReg = PhysReg(17);
/// X18: platform register — reserved on some platforms (e.g., macOS), but
/// usable as a general-purpose temporary on Linux per AAPCS64.
pub const X18: PhysReg = PhysReg(18);
/// X19: callee-saved register.
pub const X19: PhysReg = PhysReg(19);
/// X20: callee-saved register.
pub const X20: PhysReg = PhysReg(20);
/// X21: callee-saved register.
pub const X21: PhysReg = PhysReg(21);
/// X22: callee-saved register.
pub const X22: PhysReg = PhysReg(22);
/// X23: callee-saved register.
pub const X23: PhysReg = PhysReg(23);
/// X24: callee-saved register.
pub const X24: PhysReg = PhysReg(24);
/// X25: callee-saved register.
pub const X25: PhysReg = PhysReg(25);
/// X26: callee-saved register.
pub const X26: PhysReg = PhysReg(26);
/// X27: callee-saved register.
pub const X27: PhysReg = PhysReg(27);
/// X28: callee-saved register.
pub const X28: PhysReg = PhysReg(28);
/// X29: frame pointer (FP) — callee-saved; points to the current stack frame
/// record for stack unwinding and debugging.
pub const X29: PhysReg = PhysReg(29);
/// X30: link register (LR) — holds the return address after a BL instruction.
/// Treated as caller-saved because a BL instruction overwrites it.
pub const X30: PhysReg = PhysReg(30);

// ===========================================================================
// Special Registers
// ===========================================================================

/// Stack pointer. In A64 encoding, SP shares register number 31 with XZR —
/// the instruction context disambiguates (e.g., ADD uses SP, AND uses XZR).
pub const SP: PhysReg = PhysReg(31);

/// 64-bit zero register — reads as zero, writes are discarded. Shares A64
/// encoding 31 with SP; the opcode selects which interpretation applies.
pub const XZR: PhysReg = PhysReg(32);

/// 32-bit zero register — the W-register view of XZR. Reads as zero (32-bit),
/// writes are discarded.
pub const WZR: PhysReg = PhysReg(33);

// ===========================================================================
// ABI Name Aliases
// ===========================================================================

/// Frame pointer — alias for X29 per AAPCS64.
pub const FP: PhysReg = X29;
/// Link register — alias for X30 per AAPCS64.
pub const LR: PhysReg = X30;
/// Intra-procedure-call scratch register 1 — alias for X16.
pub const IP0: PhysReg = X16;
/// Intra-procedure-call scratch register 2 — alias for X17.
pub const IP1: PhysReg = X17;

// ===========================================================================
// W-Register Constants (W0–W30) — 32-bit views
// ===========================================================================
//
// W-registers are the lower 32 bits of the corresponding X-registers.
// Writing to a W-register zero-extends the result into the full 64-bit
// X-register (upper 32 bits are zeroed).

/// W0: 32-bit view of X0.
pub const W0: PhysReg = PhysReg(34);
/// W1: 32-bit view of X1.
pub const W1: PhysReg = PhysReg(35);
/// W2: 32-bit view of X2.
pub const W2: PhysReg = PhysReg(36);
/// W3: 32-bit view of X3.
pub const W3: PhysReg = PhysReg(37);
/// W4: 32-bit view of X4.
pub const W4: PhysReg = PhysReg(38);
/// W5: 32-bit view of X5.
pub const W5: PhysReg = PhysReg(39);
/// W6: 32-bit view of X6.
pub const W6: PhysReg = PhysReg(40);
/// W7: 32-bit view of X7.
pub const W7: PhysReg = PhysReg(41);
/// W8: 32-bit view of X8.
pub const W8: PhysReg = PhysReg(42);
/// W9: 32-bit view of X9.
pub const W9: PhysReg = PhysReg(43);
/// W10: 32-bit view of X10.
pub const W10: PhysReg = PhysReg(44);
/// W11: 32-bit view of X11.
pub const W11: PhysReg = PhysReg(45);
/// W12: 32-bit view of X12.
pub const W12: PhysReg = PhysReg(46);
/// W13: 32-bit view of X13.
pub const W13: PhysReg = PhysReg(47);
/// W14: 32-bit view of X14.
pub const W14: PhysReg = PhysReg(48);
/// W15: 32-bit view of X15.
pub const W15: PhysReg = PhysReg(49);
/// W16: 32-bit view of X16 (IP0).
pub const W16: PhysReg = PhysReg(50);
/// W17: 32-bit view of X17 (IP1).
pub const W17: PhysReg = PhysReg(51);
/// W18: 32-bit view of X18 (platform register).
pub const W18: PhysReg = PhysReg(52);
/// W19: 32-bit view of X19 (callee-saved).
pub const W19: PhysReg = PhysReg(53);
/// W20: 32-bit view of X20 (callee-saved).
pub const W20: PhysReg = PhysReg(54);
/// W21: 32-bit view of X21 (callee-saved).
pub const W21: PhysReg = PhysReg(55);
/// W22: 32-bit view of X22 (callee-saved).
pub const W22: PhysReg = PhysReg(56);
/// W23: 32-bit view of X23 (callee-saved).
pub const W23: PhysReg = PhysReg(57);
/// W24: 32-bit view of X24 (callee-saved).
pub const W24: PhysReg = PhysReg(58);
/// W25: 32-bit view of X25 (callee-saved).
pub const W25: PhysReg = PhysReg(59);
/// W26: 32-bit view of X26 (callee-saved).
pub const W26: PhysReg = PhysReg(60);
/// W27: 32-bit view of X27 (callee-saved).
pub const W27: PhysReg = PhysReg(61);
/// W28: 32-bit view of X28 (callee-saved).
pub const W28: PhysReg = PhysReg(62);
/// W29: 32-bit view of X29 (FP).
pub const W29: PhysReg = PhysReg(63);
/// W30: 32-bit view of X30 (LR).
pub const W30: PhysReg = PhysReg(64);
/// WSP: 32-bit view of the stack pointer.
pub const WSP: PhysReg = PhysReg(65);

// ===========================================================================
// SIMD/FP Registers (V0–V31) — 128-bit
// ===========================================================================
//
// V-registers are 128-bit SIMD/FP registers with multiple sub-register views:
// - Bn: byte (8-bit)
// - Hn: halfword (16-bit)
// - Sn: single-precision float (32-bit)
// - Dn: double-precision float (64-bit)
// - Qn: quad (128-bit, same as Vn)
//
// AAPCS64 callee-saved rule: only the lower 64 bits (D8–D15) of V8–V15
// are preserved across calls; the upper 64 bits are caller-saved.

/// V0: SIMD/FP register 0 (first FP argument/return register per AAPCS64).
pub const V0: PhysReg = PhysReg(66);
/// V1: SIMD/FP register 1 (second FP argument register).
pub const V1: PhysReg = PhysReg(67);
/// V2: SIMD/FP register 2.
pub const V2: PhysReg = PhysReg(68);
/// V3: SIMD/FP register 3.
pub const V3: PhysReg = PhysReg(69);
/// V4: SIMD/FP register 4.
pub const V4: PhysReg = PhysReg(70);
/// V5: SIMD/FP register 5.
pub const V5: PhysReg = PhysReg(71);
/// V6: SIMD/FP register 6.
pub const V6: PhysReg = PhysReg(72);
/// V7: SIMD/FP register 7 (last FP argument register).
pub const V7: PhysReg = PhysReg(73);
/// V8: SIMD/FP register 8 — lower 64 bits (D8) are callee-saved.
pub const V8: PhysReg = PhysReg(74);
/// V9: SIMD/FP register 9 — lower 64 bits (D9) are callee-saved.
pub const V9: PhysReg = PhysReg(75);
/// V10: SIMD/FP register 10 — lower 64 bits (D10) are callee-saved.
pub const V10: PhysReg = PhysReg(76);
/// V11: SIMD/FP register 11 — lower 64 bits (D11) are callee-saved.
pub const V11: PhysReg = PhysReg(77);
/// V12: SIMD/FP register 12 — lower 64 bits (D12) are callee-saved.
pub const V12: PhysReg = PhysReg(78);
/// V13: SIMD/FP register 13 — lower 64 bits (D13) are callee-saved.
pub const V13: PhysReg = PhysReg(79);
/// V14: SIMD/FP register 14 — lower 64 bits (D14) are callee-saved.
pub const V14: PhysReg = PhysReg(80);
/// V15: SIMD/FP register 15 — lower 64 bits (D15) are callee-saved.
pub const V15: PhysReg = PhysReg(81);
/// V16: SIMD/FP register 16 — caller-saved.
pub const V16: PhysReg = PhysReg(82);
/// V17: SIMD/FP register 17 — caller-saved.
pub const V17: PhysReg = PhysReg(83);
/// V18: SIMD/FP register 18 — caller-saved.
pub const V18: PhysReg = PhysReg(84);
/// V19: SIMD/FP register 19 — caller-saved.
pub const V19: PhysReg = PhysReg(85);
/// V20: SIMD/FP register 20 — caller-saved.
pub const V20: PhysReg = PhysReg(86);
/// V21: SIMD/FP register 21 — caller-saved.
pub const V21: PhysReg = PhysReg(87);
/// V22: SIMD/FP register 22 — caller-saved.
pub const V22: PhysReg = PhysReg(88);
/// V23: SIMD/FP register 23 — caller-saved.
pub const V23: PhysReg = PhysReg(89);
/// V24: SIMD/FP register 24 — caller-saved.
pub const V24: PhysReg = PhysReg(90);
/// V25: SIMD/FP register 25 — caller-saved.
pub const V25: PhysReg = PhysReg(91);
/// V26: SIMD/FP register 26 — caller-saved.
pub const V26: PhysReg = PhysReg(92);
/// V27: SIMD/FP register 27 — caller-saved.
pub const V27: PhysReg = PhysReg(93);
/// V28: SIMD/FP register 28 — caller-saved.
pub const V28: PhysReg = PhysReg(94);
/// V29: SIMD/FP register 29 — caller-saved.
pub const V29: PhysReg = PhysReg(95);
/// V30: SIMD/FP register 30 — caller-saved.
pub const V30: PhysReg = PhysReg(96);
/// V31: SIMD/FP register 31 — caller-saved.
pub const V31: PhysReg = PhysReg(97);

// ===========================================================================
// S-Register Constants (S0–S31) — 32-bit single-precision FP views
// ===========================================================================
//
// S-registers are the lower 32 bits of the corresponding V-registers,
// used for single-precision floating-point operations (e.g., FADD Sd, Sn, Sm).

pub const S0: PhysReg = PhysReg(98);
pub const S1: PhysReg = PhysReg(99);
pub const S2: PhysReg = PhysReg(100);
pub const S3: PhysReg = PhysReg(101);
pub const S4: PhysReg = PhysReg(102);
pub const S5: PhysReg = PhysReg(103);
pub const S6: PhysReg = PhysReg(104);
pub const S7: PhysReg = PhysReg(105);
pub const S8: PhysReg = PhysReg(106);
pub const S9: PhysReg = PhysReg(107);
pub const S10: PhysReg = PhysReg(108);
pub const S11: PhysReg = PhysReg(109);
pub const S12: PhysReg = PhysReg(110);
pub const S13: PhysReg = PhysReg(111);
pub const S14: PhysReg = PhysReg(112);
pub const S15: PhysReg = PhysReg(113);
pub const S16: PhysReg = PhysReg(114);
pub const S17: PhysReg = PhysReg(115);
pub const S18: PhysReg = PhysReg(116);
pub const S19: PhysReg = PhysReg(117);
pub const S20: PhysReg = PhysReg(118);
pub const S21: PhysReg = PhysReg(119);
pub const S22: PhysReg = PhysReg(120);
pub const S23: PhysReg = PhysReg(121);
pub const S24: PhysReg = PhysReg(122);
pub const S25: PhysReg = PhysReg(123);
pub const S26: PhysReg = PhysReg(124);
pub const S27: PhysReg = PhysReg(125);
pub const S28: PhysReg = PhysReg(126);
pub const S29: PhysReg = PhysReg(127);
pub const S30: PhysReg = PhysReg(128);
pub const S31: PhysReg = PhysReg(129);

// ===========================================================================
// D-Register Constants (D0–D31) — 64-bit double-precision FP views
// ===========================================================================
//
// D-registers are the lower 64 bits of the corresponding V-registers,
// used for double-precision floating-point operations (e.g., FADD Dd, Dn, Dm).
// D8–D15 are the callee-saved portion of V8–V15 per AAPCS64.

pub const D0: PhysReg = PhysReg(130);
pub const D1: PhysReg = PhysReg(131);
pub const D2: PhysReg = PhysReg(132);
pub const D3: PhysReg = PhysReg(133);
pub const D4: PhysReg = PhysReg(134);
pub const D5: PhysReg = PhysReg(135);
pub const D6: PhysReg = PhysReg(136);
pub const D7: PhysReg = PhysReg(137);
pub const D8: PhysReg = PhysReg(138);
pub const D9: PhysReg = PhysReg(139);
pub const D10: PhysReg = PhysReg(140);
pub const D11: PhysReg = PhysReg(141);
pub const D12: PhysReg = PhysReg(142);
pub const D13: PhysReg = PhysReg(143);
pub const D14: PhysReg = PhysReg(144);
pub const D15: PhysReg = PhysReg(145);
pub const D16: PhysReg = PhysReg(146);
pub const D17: PhysReg = PhysReg(147);
pub const D18: PhysReg = PhysReg(148);
pub const D19: PhysReg = PhysReg(149);
pub const D20: PhysReg = PhysReg(150);
pub const D21: PhysReg = PhysReg(151);
pub const D22: PhysReg = PhysReg(152);
pub const D23: PhysReg = PhysReg(153);
pub const D24: PhysReg = PhysReg(154);
pub const D25: PhysReg = PhysReg(155);
pub const D26: PhysReg = PhysReg(156);
pub const D27: PhysReg = PhysReg(157);
pub const D28: PhysReg = PhysReg(158);
pub const D29: PhysReg = PhysReg(159);
pub const D30: PhysReg = PhysReg(160);
pub const D31: PhysReg = PhysReg(161);

// ===========================================================================
// NZCV Condition Code Constants
// ===========================================================================
//
// AArch64 condition codes are 4-bit values used in conditional branch
// (B.cond), conditional compare (CCMP/CCMN), and conditional select
// (CSEL/CSINC/CSINV/CSNEG) instructions. They test the NZCV flags set
// by comparison and flag-setting arithmetic instructions (CMP, ADDS, SUBS).
//
// The condition codes come in complementary pairs — inverting the LSB
// gives the opposite condition (see [`invert_condition`]).

/// Equal (Z flag set).
pub const COND_EQ: u8 = 0b0000;
/// Not equal (Z flag clear).
pub const COND_NE: u8 = 0b0001;
/// Carry set / unsigned higher or same (C flag set). Also known as HS.
pub const COND_CS: u8 = 0b0010;
/// Carry clear / unsigned lower (C flag clear). Also known as LO.
pub const COND_CC: u8 = 0b0011;
/// Minus / negative (N flag set).
pub const COND_MI: u8 = 0b0100;
/// Plus / positive or zero (N flag clear).
pub const COND_PL: u8 = 0b0101;
/// Overflow (V flag set).
pub const COND_VS: u8 = 0b0110;
/// No overflow (V flag clear).
pub const COND_VC: u8 = 0b0111;
/// Unsigned higher (C set AND Z clear).
pub const COND_HI: u8 = 0b1000;
/// Unsigned lower or same (C clear OR Z set).
pub const COND_LS: u8 = 0b1001;
/// Signed greater or equal (N == V).
pub const COND_GE: u8 = 0b1010;
/// Signed less than (N != V).
pub const COND_LT: u8 = 0b1011;
/// Signed greater than (Z clear AND N == V).
pub const COND_GT: u8 = 0b1100;
/// Signed less or equal (Z set OR N != V).
pub const COND_LE: u8 = 0b1101;
/// Always — unconditional (condition always true).
pub const COND_AL: u8 = 0b1110;
/// Never — reserved encoding; behaves as AL in most instruction contexts.
pub const COND_NV: u8 = 0b1111;

// ===========================================================================
// Register Classification Arrays
// ===========================================================================
//
// These arrays group registers by their AAPCS64 ABI roles and are used by
// the register allocator, ABI handling, and prologue/epilogue generation.

/// Integer argument registers (NGRN allocation order per AAPCS64).
/// X0–X7 are used for the first 8 integer/pointer arguments.
pub const INTEGER_ARG_REGS: [PhysReg; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// Floating-point/SIMD argument registers (NSRN allocation order per AAPCS64).
/// V0–V7 are used for the first 8 FP/vector arguments and HFA members.
pub const FLOAT_ARG_REGS: [PhysReg; 8] = [V0, V1, V2, V3, V4, V5, V6, V7];

/// Callee-saved integer registers per AAPCS64.
/// X19–X28 must be preserved across function calls. The function prologue
/// saves these if used and the epilogue restores them.
pub const CALLEE_SAVED_INT: [PhysReg; 10] = [X19, X20, X21, X22, X23, X24, X25, X26, X27, X28];

/// Callee-saved SIMD/FP registers per AAPCS64.
/// Only the lower 64 bits (D8–D15) of V8–V15 are callee-saved. The upper
/// 64 bits are caller-saved and may be freely clobbered by callees.
pub const CALLEE_SAVED_FP: [PhysReg; 8] = [V8, V9, V10, V11, V12, V13, V14, V15];

/// Caller-saved (volatile) integer registers per AAPCS64.
/// These registers may be freely clobbered by a callee and must be saved
/// by the caller if their values are needed after a call.
///
/// Includes: X0–X18 (arguments, indirect result, temporaries, IP0/IP1,
/// platform register).  X30 (LR) is NOT listed here because it is not
/// allocatable — the prologue/epilogue pair handles saving/restoring LR.
pub const CALLER_SAVED_INT: [PhysReg; 19] = [
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18,
];

/// Caller-saved (volatile) SIMD/FP registers per AAPCS64.
/// V0–V7 (argument registers) and V16–V31 are fully caller-saved.
pub const CALLER_SAVED_FP: [PhysReg; 24] = [
    V0, V1, V2, V3, V4, V5, V6, V7, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27,
    V28, V29, V30, V31,
];

/// Allocatable integer registers — all GPRs available for the register
/// allocator. Excludes SP (not a GPR), XZR (zero register, not writable),
/// and X29/FP (reserved for the frame pointer when frame pointer is used,
/// which is always the case for AArch64 in this compiler for correct
/// stack unwinding).
/// Allocatable integer registers — X0–X28 minus X29 (FP).
/// X29 is the frame pointer (reserved by AAPCS64).
/// X30 is the link register (LR) — written implicitly by BL instructions
/// and restored by the epilogue; it must NOT be used for general allocation.
pub const ALLOCATABLE_INT: [PhysReg; 29] = [
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20,
    X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Allocatable SIMD/FP registers — all 32 V-registers are available for
/// the register allocator. Callee-saved status (V8–V15 lower 64 bits)
/// is handled by prologue/epilogue spill generation, not allocation.
pub const ALLOCATABLE_FP: [PhysReg; 32] = [
    V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
    V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
];

/// Indirect result location register per AAPCS64. When a function returns
/// a composite type larger than 16 bytes, the caller allocates memory and
/// passes its address in X8. The callee writes the return value to `*X8`.
pub const INDIRECT_RESULT_REG: PhysReg = X8;

// ===========================================================================
// Register Conversion Functions
// ===========================================================================

/// Convert an X-register (64-bit GPR) to its corresponding W-register
/// (32-bit view). Panics in debug mode if `reg` is not a valid X-register.
///
/// # Mapping
/// X0 (PhysReg 0) → W0 (PhysReg 34), ..., X30 (PhysReg 30) → W30 (PhysReg 64).
#[inline]
pub fn x_to_w(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 <= 30,
        "x_to_w: expected X-register (PhysReg 0–30), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 + 34)
}

/// Convert a W-register (32-bit GPR view) to its corresponding X-register
/// (64-bit GPR). Panics in debug mode if `reg` is not a valid W-register.
///
/// # Mapping
/// W0 (PhysReg 34) → X0 (PhysReg 0), ..., W30 (PhysReg 64) → X30 (PhysReg 30).
#[inline]
pub fn w_to_x(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 >= 34 && reg.0 <= 64,
        "w_to_x: expected W-register (PhysReg 34–64), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 - 34)
}

/// Convert a V-register (128-bit SIMD/FP) to its S-register
/// (32-bit single-precision) view.
///
/// # Mapping
/// V0 (PhysReg 66) → S0 (PhysReg 98), ..., V31 (PhysReg 97) → S31 (PhysReg 129).
#[inline]
pub fn v_to_s(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 >= 66 && reg.0 <= 97,
        "v_to_s: expected V-register (PhysReg 66–97), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 + 32)
}

/// Convert a V-register (128-bit SIMD/FP) to its D-register
/// (64-bit double-precision) view.
///
/// # Mapping
/// V0 (PhysReg 66) → D0 (PhysReg 130), ..., V31 (PhysReg 97) → D31 (PhysReg 161).
#[inline]
pub fn v_to_d(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 >= 66 && reg.0 <= 97,
        "v_to_d: expected V-register (PhysReg 66–97), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 + 64)
}

/// Convert an S-register (32-bit single-precision) to its parent V-register
/// (128-bit SIMD/FP).
///
/// # Mapping
/// S0 (PhysReg 98) → V0 (PhysReg 66), ..., S31 (PhysReg 129) → V31 (PhysReg 97).
#[inline]
pub fn s_to_v(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 >= 98 && reg.0 <= 129,
        "s_to_v: expected S-register (PhysReg 98–129), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 - 32)
}

/// Convert a D-register (64-bit double-precision) to its parent V-register
/// (128-bit SIMD/FP).
///
/// # Mapping
/// D0 (PhysReg 130) → V0 (PhysReg 66), ..., D31 (PhysReg 161) → V31 (PhysReg 97).
#[inline]
pub fn d_to_v(reg: PhysReg) -> PhysReg {
    debug_assert!(
        reg.0 >= 130 && reg.0 <= 161,
        "d_to_v: expected D-register (PhysReg 130–161), got PhysReg({})",
        reg.0
    );
    PhysReg(reg.0 - 64)
}

// ===========================================================================
// Register Name Lookup — Static Name Tables
// ===========================================================================

/// X-register and SP names indexed by hardware register number (0–31).
static X_REG_NAMES: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
    "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27",
    "x28", "x29", "x30", "sp",
];

/// W-register and WSP names indexed by hardware register number (0–31).
static W_REG_NAMES: [&str; 32] = [
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
    "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27",
    "w28", "w29", "w30", "wsp",
];

/// V-register names indexed by hardware register number (0–31).
static V_REG_NAMES: [&str; 32] = [
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
    "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
    "v28", "v29", "v30", "v31",
];

/// S-register (single-precision FP) names indexed by hardware register number (0–31).
static S_REG_NAMES: [&str; 32] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31",
];

/// D-register (double-precision FP) names indexed by hardware register number (0–31).
static D_REG_NAMES: [&str; 32] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    "d28", "d29", "d30", "d31",
];

/// Condition code names indexed by 4-bit condition value (0–15).
static COND_NAMES: [&str; 16] = [
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv",
];

// ===========================================================================
// Register Name Lookup Functions
// ===========================================================================

/// Return the assembly name for an X-register or SP.
///
/// Accepts PhysReg values 0–30 (X0–X30) and 31 (SP).
/// Returns `"<invalid>"` for out-of-range values.
#[inline]
pub fn x_reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0 as usize;
    if idx <= 31 {
        X_REG_NAMES[idx]
    } else {
        "<invalid>"
    }
}

/// Return the assembly name for a W-register or WSP.
///
/// Accepts PhysReg values 34–64 (W0–W30) and 65 (WSP).
/// Returns `"<invalid>"` for out-of-range values.
#[inline]
pub fn w_reg_name(reg: PhysReg) -> &'static str {
    if reg.0 >= 34 && reg.0 <= 65 {
        W_REG_NAMES[(reg.0 - 34) as usize]
    } else {
        "<invalid>"
    }
}

/// Return the assembly name for a V-register (128-bit SIMD/FP).
///
/// Accepts PhysReg values 66–97 (V0–V31).
/// Returns `"<invalid>"` for out-of-range values.
#[inline]
pub fn v_reg_name(reg: PhysReg) -> &'static str {
    if reg.0 >= 66 && reg.0 <= 97 {
        V_REG_NAMES[(reg.0 - 66) as usize]
    } else {
        "<invalid>"
    }
}

/// Return the assembly name for an S-register (32-bit single-precision).
///
/// Accepts PhysReg values 98–129 (S0–S31).
/// Returns `"<invalid>"` for out-of-range values.
#[inline]
pub fn s_reg_name(reg: PhysReg) -> &'static str {
    if reg.0 >= 98 && reg.0 <= 129 {
        S_REG_NAMES[(reg.0 - 98) as usize]
    } else {
        "<invalid>"
    }
}

/// Return the assembly name for a D-register (64-bit double-precision).
///
/// Accepts PhysReg values 130–161 (D0–D31).
/// Returns `"<invalid>"` for out-of-range values.
#[inline]
pub fn d_reg_name(reg: PhysReg) -> &'static str {
    if reg.0 >= 130 && reg.0 <= 161 {
        D_REG_NAMES[(reg.0 - 130) as usize]
    } else {
        "<invalid>"
    }
}

/// Return the assembly name for any AArch64 physical register.
///
/// Dispatches to the appropriate name lookup function based on the
/// PhysReg range:
///
/// | Range   | Result              |
/// |---------|---------------------|
/// | 0–30    | `"x0"`–`"x30"`      |
/// | 31      | `"sp"`              |
/// | 32      | `"xzr"`             |
/// | 33      | `"wzr"`             |
/// | 34–64   | `"w0"`–`"w30"`      |
/// | 65      | `"wsp"`             |
/// | 66–97   | `"v0"`–`"v31"`      |
/// | 98–129  | `"s0"`–`"s31"`      |
/// | 130–161 | `"d0"`–`"d31"`      |
/// | other   | `"<invalid>"`       |
#[inline]
pub fn reg_name(reg: PhysReg) -> &'static str {
    let idx = reg.0;
    match idx {
        0..=31 => X_REG_NAMES[idx as usize],
        32 => "xzr",
        33 => "wzr",
        34..=65 => W_REG_NAMES[(idx - 34) as usize],
        66..=97 => V_REG_NAMES[(idx - 66) as usize],
        98..=129 => S_REG_NAMES[(idx - 98) as usize],
        130..=161 => D_REG_NAMES[(idx - 130) as usize],
        _ => "<invalid>",
    }
}

/// Return the assembly mnemonic for a 4-bit NZCV condition code.
///
/// Returns `"<invalid>"` if `cond > 15`.
#[inline]
pub fn cond_name(cond: u8) -> &'static str {
    if cond <= 15 {
        COND_NAMES[cond as usize]
    } else {
        "<invalid>"
    }
}

// ===========================================================================
// Register Property Functions
// ===========================================================================

/// Returns `true` if `reg` is a 64-bit general-purpose register (X0–X30).
///
/// Does NOT return `true` for SP (PhysReg 31), XZR (PhysReg 32), or
/// W-registers (PhysReg 34–64). Use [`is_gpr_w`] for W-register queries.
#[inline]
pub fn is_gpr(reg: PhysReg) -> bool {
    reg.0 <= 30
}

/// Returns `true` if `reg` is a 32-bit general-purpose register (W0–W30).
///
/// Does NOT include WSP (PhysReg 65) or WZR (PhysReg 33).
#[inline]
pub fn is_gpr_w(reg: PhysReg) -> bool {
    reg.0 >= 34 && reg.0 <= 64
}

/// Returns `true` if `reg` is a SIMD/FP register in the V-register range (V0–V31).
///
/// Also returns `true` for S-register (98–129) and D-register (130–161) views,
/// since they are sub-register views of the same physical V-register.
#[inline]
pub fn is_fp_reg(reg: PhysReg) -> bool {
    reg.0 >= 66 && reg.0 <= 161
}

/// Returns `true` if `reg` is callee-saved per AAPCS64.
///
/// Callee-saved integer: X19–X28 (PhysReg 19–28).
/// Callee-saved FP: V8–V15 (PhysReg 74–81) — only lower 64 bits (D8–D15).
///
/// Note: X29 (FP) and X30 (LR) are saved as part of the frame record but
/// are not included here because they receive special handling in
/// prologue/epilogue generation rather than generic callee-saved spilling.
#[inline]
pub fn is_callee_saved(reg: PhysReg) -> bool {
    let idx = reg.0;
    // X19–X28 (integer callee-saved)
    (19..=28).contains(&idx)
    // V8–V15 (FP callee-saved — lower 64 bits only)
    || (74..=81).contains(&idx)
    // D8–D15 (explicit double-precision view of callee-saved FP)
    || (138..=145).contains(&idx)
    // S8–S15 (single-precision view of callee-saved FP)
    || (106..=113).contains(&idx)
    // W19–W28 (32-bit views of callee-saved integer regs)
    || (53..=62).contains(&idx)
}

/// Returns `true` if `reg` is available for the register allocator.
///
/// Returns `false` for:
/// - SP (PhysReg 31) / WSP (PhysReg 65) — dedicated stack pointer
/// - XZR (PhysReg 32) / WZR (PhysReg 33) — zero register (not writable)
/// - X29/FP (PhysReg 29) / W29 (PhysReg 63) — reserved for frame pointer
///
/// All other X, W, V, S, and D registers are allocatable.
#[inline]
pub fn is_allocatable(reg: PhysReg) -> bool {
    let idx = reg.0;
    match idx {
        // SP, XZR, WZR — never allocatable
        31..=33 => false,
        // X29 (FP) — reserved for frame pointer
        29 => false,
        // X30 (LR) — reserved for link register (clobbered by BL)
        30 => false,
        // WSP — not allocatable
        65 => false,
        // W29 — 32-bit view of FP, also reserved
        63 => false,
        // W30 — 32-bit view of LR, also reserved
        64 => false,
        // X0–X28 — allocatable GPRs
        0..=28 => true,
        // W0–W28 — allocatable 32-bit views
        34..=62 => true,
        // V0–V31 — all allocatable
        66..=97 => true,
        // S0–S31 — all allocatable
        98..=129 => true,
        // D0–D31 — all allocatable
        130..=161 => true,
        // Anything else (including PhysReg::NONE) is not allocatable
        _ => false,
    }
}

/// Return the 5-bit hardware register encoding (0–31) for an AArch64
/// register, as used in A64 instruction fields.
///
/// In A64 encoding:
/// - X0–X30 encode as 0–30
/// - SP and XZR both encode as 31 (instruction context disambiguates)
/// - W-registers encode identically to their X counterparts (the `sf` bit
///   in the instruction selects 32-bit vs 64-bit operation)
/// - V/S/D/Q registers encode as 0–31 in the FP register field
///
/// Returns 0 for unrecognized PhysReg values (should not occur in
/// well-formed code).
#[inline]
pub fn encoding(reg: PhysReg) -> u8 {
    let idx = reg.0;
    match idx {
        // X0–X30 → 0–30
        0..=30 => idx as u8,
        // SP → 31
        31 => 31,
        // XZR → 31 (same encoding as SP; instruction opcode disambiguates)
        32 => 31,
        // WZR → 31
        33 => 31,
        // W0–W30 → 0–30
        34..=64 => (idx - 34) as u8,
        // WSP → 31
        65 => 31,
        // V0–V31 → 0–31
        66..=97 => (idx - 66) as u8,
        // S0–S31 → 0–31
        98..=129 => (idx - 98) as u8,
        // D0–D31 → 0–31
        130..=161 => (idx - 130) as u8,
        // Invalid register — should not occur in correct code
        _ => 0,
    }
}

/// Invert an AArch64 condition code by flipping the least significant bit.
///
/// AArch64 condition codes are arranged in complementary pairs:
/// - EQ (0b0000) ↔ NE (0b0001)
/// - CS (0b0010) ↔ CC (0b0011)
/// - MI (0b0100) ↔ PL (0b0101)
/// - VS (0b0110) ↔ VC (0b0111)
/// - HI (0b1000) ↔ LS (0b1001)
/// - GE (0b1010) ↔ LT (0b1011)
/// - GT (0b1100) ↔ LE (0b1101)
/// - AL (0b1110) ↔ NV (0b1111)
///
/// This property allows efficient condition inversion with a single XOR.
#[inline]
pub fn invert_condition(cond: u8) -> u8 {
    cond ^ 1
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_constants_range() {
        assert_eq!(X0.0, 0);
        assert_eq!(X15.0, 15);
        assert_eq!(X30.0, 30);
        assert_eq!(SP.0, 31);
        assert_eq!(XZR.0, 32);
        assert_eq!(WZR.0, 33);
    }

    #[test]
    fn test_w_register_constants_range() {
        assert_eq!(W0.0, 34);
        assert_eq!(W15.0, 49);
        assert_eq!(W30.0, 64);
        assert_eq!(WSP.0, 65);
    }

    #[test]
    fn test_v_register_constants_range() {
        assert_eq!(V0.0, 66);
        assert_eq!(V31.0, 97);
    }

    #[test]
    fn test_s_register_constants_range() {
        assert_eq!(S0.0, 98);
        assert_eq!(S31.0, 129);
    }

    #[test]
    fn test_d_register_constants_range() {
        assert_eq!(D0.0, 130);
        assert_eq!(D31.0, 161);
    }

    #[test]
    fn test_abi_aliases() {
        assert_eq!(FP, X29);
        assert_eq!(LR, X30);
        assert_eq!(IP0, X16);
        assert_eq!(IP1, X17);
        assert_eq!(INDIRECT_RESULT_REG, X8);
    }

    #[test]
    fn test_x_to_w_conversion() {
        assert_eq!(x_to_w(X0), W0);
        assert_eq!(x_to_w(X15), W15);
        assert_eq!(x_to_w(X30), W30);
    }

    #[test]
    fn test_w_to_x_conversion() {
        assert_eq!(w_to_x(W0), X0);
        assert_eq!(w_to_x(W15), X15);
        assert_eq!(w_to_x(W30), X30);
    }

    #[test]
    fn test_v_to_s_conversion() {
        assert_eq!(v_to_s(V0), S0);
        assert_eq!(v_to_s(V31), S31);
    }

    #[test]
    fn test_v_to_d_conversion() {
        assert_eq!(v_to_d(V0), D0);
        assert_eq!(v_to_d(V31), D31);
    }

    #[test]
    fn test_s_to_v_conversion() {
        assert_eq!(s_to_v(S0), V0);
        assert_eq!(s_to_v(S31), V31);
    }

    #[test]
    fn test_d_to_v_conversion() {
        assert_eq!(d_to_v(D0), V0);
        assert_eq!(d_to_v(D31), V31);
    }

    #[test]
    fn test_roundtrip_x_w() {
        for i in 0..=30 {
            let x = PhysReg(i);
            assert_eq!(w_to_x(x_to_w(x)), x);
        }
    }

    #[test]
    fn test_roundtrip_v_s_d() {
        for i in 66..=97 {
            let v = PhysReg(i);
            assert_eq!(s_to_v(v_to_s(v)), v);
            assert_eq!(d_to_v(v_to_d(v)), v);
        }
    }

    #[test]
    fn test_register_names() {
        assert_eq!(x_reg_name(X0), "x0");
        assert_eq!(x_reg_name(X29), "x29");
        assert_eq!(x_reg_name(X30), "x30");
        assert_eq!(x_reg_name(SP), "sp");
        assert_eq!(w_reg_name(W0), "w0");
        assert_eq!(w_reg_name(W30), "w30");
        assert_eq!(w_reg_name(WSP), "wsp");
        assert_eq!(v_reg_name(V0), "v0");
        assert_eq!(v_reg_name(V31), "v31");
        assert_eq!(s_reg_name(S0), "s0");
        assert_eq!(s_reg_name(S31), "s31");
        assert_eq!(d_reg_name(D0), "d0");
        assert_eq!(d_reg_name(D31), "d31");
    }

    #[test]
    fn test_reg_name_dispatch() {
        assert_eq!(reg_name(X0), "x0");
        assert_eq!(reg_name(SP), "sp");
        assert_eq!(reg_name(XZR), "xzr");
        assert_eq!(reg_name(WZR), "wzr");
        assert_eq!(reg_name(W0), "w0");
        assert_eq!(reg_name(WSP), "wsp");
        assert_eq!(reg_name(V0), "v0");
        assert_eq!(reg_name(S0), "s0");
        assert_eq!(reg_name(D0), "d0");
    }

    #[test]
    fn test_cond_name_lookup() {
        assert_eq!(cond_name(COND_EQ), "eq");
        assert_eq!(cond_name(COND_NE), "ne");
        assert_eq!(cond_name(COND_CS), "cs");
        assert_eq!(cond_name(COND_CC), "cc");
        assert_eq!(cond_name(COND_MI), "mi");
        assert_eq!(cond_name(COND_PL), "pl");
        assert_eq!(cond_name(COND_VS), "vs");
        assert_eq!(cond_name(COND_VC), "vc");
        assert_eq!(cond_name(COND_HI), "hi");
        assert_eq!(cond_name(COND_LS), "ls");
        assert_eq!(cond_name(COND_GE), "ge");
        assert_eq!(cond_name(COND_LT), "lt");
        assert_eq!(cond_name(COND_GT), "gt");
        assert_eq!(cond_name(COND_LE), "le");
        assert_eq!(cond_name(COND_AL), "al");
        assert_eq!(cond_name(COND_NV), "nv");
        assert_eq!(cond_name(16), "<invalid>");
    }

    #[test]
    fn test_is_gpr() {
        assert!(is_gpr(X0));
        assert!(is_gpr(X30));
        assert!(!is_gpr(SP));
        assert!(!is_gpr(XZR));
        assert!(!is_gpr(W0));
        assert!(!is_gpr(V0));
    }

    #[test]
    fn test_is_gpr_w() {
        assert!(is_gpr_w(W0));
        assert!(is_gpr_w(W30));
        assert!(!is_gpr_w(WSP));
        assert!(!is_gpr_w(WZR));
        assert!(!is_gpr_w(X0));
    }

    #[test]
    fn test_is_fp_reg() {
        assert!(is_fp_reg(V0));
        assert!(is_fp_reg(V31));
        assert!(is_fp_reg(S0));
        assert!(is_fp_reg(S31));
        assert!(is_fp_reg(D0));
        assert!(is_fp_reg(D31));
        assert!(!is_fp_reg(X0));
        assert!(!is_fp_reg(W0));
        assert!(!is_fp_reg(SP));
    }

    #[test]
    fn test_is_callee_saved() {
        // X19–X28 are callee-saved
        assert!(is_callee_saved(X19));
        assert!(is_callee_saved(X28));
        // X0–X18, X29, X30 are NOT callee-saved in the general array
        assert!(!is_callee_saved(X0));
        assert!(!is_callee_saved(X18));
        assert!(!is_callee_saved(X29));
        assert!(!is_callee_saved(X30));
        // V8–V15 are callee-saved (lower 64 bits)
        assert!(is_callee_saved(V8));
        assert!(is_callee_saved(V15));
        assert!(!is_callee_saved(V0));
        assert!(!is_callee_saved(V16));
    }

    #[test]
    fn test_is_allocatable() {
        assert!(is_allocatable(X0));
        assert!(is_allocatable(X28));
        assert!(!is_allocatable(X30)); // LR — not allocatable (link register)
        assert!(!is_allocatable(X29)); // FP
        assert!(!is_allocatable(SP));
        assert!(!is_allocatable(XZR));
        assert!(!is_allocatable(WZR));
        assert!(!is_allocatable(WSP));
        assert!(is_allocatable(V0));
        assert!(is_allocatable(V31));
        assert!(is_allocatable(S0));
        assert!(is_allocatable(D0));
    }

    #[test]
    fn test_encoding() {
        assert_eq!(encoding(X0), 0);
        assert_eq!(encoding(X30), 30);
        assert_eq!(encoding(SP), 31);
        assert_eq!(encoding(XZR), 31);
        assert_eq!(encoding(WZR), 31);
        assert_eq!(encoding(W0), 0);
        assert_eq!(encoding(W30), 30);
        assert_eq!(encoding(WSP), 31);
        assert_eq!(encoding(V0), 0);
        assert_eq!(encoding(V31), 31);
        assert_eq!(encoding(S0), 0);
        assert_eq!(encoding(S31), 31);
        assert_eq!(encoding(D0), 0);
        assert_eq!(encoding(D31), 31);
    }

    #[test]
    fn test_encoding_consistency() {
        // X and W register pairs must have the same 5-bit encoding
        for i in 0..=30 {
            let x = PhysReg(i);
            let w = x_to_w(x);
            assert_eq!(encoding(x), encoding(w));
        }
        // V, S, and D register views must have the same 5-bit encoding
        for i in 0..32u16 {
            let v = PhysReg(66 + i);
            let s = PhysReg(98 + i);
            let d = PhysReg(130 + i);
            assert_eq!(encoding(v), encoding(s));
            assert_eq!(encoding(v), encoding(d));
        }
    }

    #[test]
    fn test_invert_condition() {
        assert_eq!(invert_condition(COND_EQ), COND_NE);
        assert_eq!(invert_condition(COND_NE), COND_EQ);
        assert_eq!(invert_condition(COND_CS), COND_CC);
        assert_eq!(invert_condition(COND_CC), COND_CS);
        assert_eq!(invert_condition(COND_MI), COND_PL);
        assert_eq!(invert_condition(COND_PL), COND_MI);
        assert_eq!(invert_condition(COND_VS), COND_VC);
        assert_eq!(invert_condition(COND_VC), COND_VS);
        assert_eq!(invert_condition(COND_HI), COND_LS);
        assert_eq!(invert_condition(COND_LS), COND_HI);
        assert_eq!(invert_condition(COND_GE), COND_LT);
        assert_eq!(invert_condition(COND_LT), COND_GE);
        assert_eq!(invert_condition(COND_GT), COND_LE);
        assert_eq!(invert_condition(COND_LE), COND_GT);
        assert_eq!(invert_condition(COND_AL), COND_NV);
        assert_eq!(invert_condition(COND_NV), COND_AL);
    }

    #[test]
    fn test_invert_condition_double_invert_identity() {
        for cond in 0..=15u8 {
            assert_eq!(invert_condition(invert_condition(cond)), cond);
        }
    }

    #[test]
    fn test_classification_array_sizes() {
        assert_eq!(INTEGER_ARG_REGS.len(), 8);
        assert_eq!(FLOAT_ARG_REGS.len(), 8);
        assert_eq!(CALLEE_SAVED_INT.len(), 10);
        assert_eq!(CALLEE_SAVED_FP.len(), 8);
        assert_eq!(CALLER_SAVED_INT.len(), 19);
        assert_eq!(CALLER_SAVED_FP.len(), 24);
        assert_eq!(ALLOCATABLE_INT.len(), 29);
        assert_eq!(ALLOCATABLE_FP.len(), 32);
    }

    #[test]
    fn test_classification_arrays_valid_registers() {
        for reg in &INTEGER_ARG_REGS {
            assert!(is_gpr(*reg));
            assert!(is_allocatable(*reg));
        }
        for reg in &FLOAT_ARG_REGS {
            assert!(is_fp_reg(*reg));
            assert!(is_allocatable(*reg));
        }
        for reg in &CALLEE_SAVED_INT {
            assert!(is_gpr(*reg));
            assert!(is_callee_saved(*reg));
            assert!(is_allocatable(*reg));
        }
        for reg in &CALLEE_SAVED_FP {
            assert!(is_fp_reg(*reg));
            assert!(is_callee_saved(*reg));
            assert!(is_allocatable(*reg));
        }
        for reg in &ALLOCATABLE_INT {
            assert!(is_gpr(*reg));
            assert!(is_allocatable(*reg));
        }
        for reg in &ALLOCATABLE_FP {
            assert!(is_fp_reg(*reg));
            assert!(is_allocatable(*reg));
        }
    }

    #[test]
    fn test_no_overlap_callee_caller_saved_int() {
        for callee in &CALLEE_SAVED_INT {
            assert!(
                !CALLER_SAVED_INT.contains(callee),
                "Register {:?} found in both CALLEE_SAVED_INT and CALLER_SAVED_INT",
                callee
            );
        }
    }

    #[test]
    fn test_invalid_reg_name_returns_invalid() {
        let invalid = PhysReg(200);
        assert_eq!(reg_name(invalid), "<invalid>");
        assert_eq!(x_reg_name(invalid), "<invalid>");
        assert_eq!(w_reg_name(invalid), "<invalid>");
        assert_eq!(v_reg_name(invalid), "<invalid>");
        assert_eq!(s_reg_name(invalid), "<invalid>");
        assert_eq!(d_reg_name(invalid), "<invalid>");
    }
}
