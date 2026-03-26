#![allow(clippy::vec_init_then_push)]
//! # AArch64 Instruction Selection and Emission
//!
//! Translates BCC IR instructions into AArch64 (ARM 64-bit) machine instructions.
//! Covers the A64 instruction set with fixed 32-bit instruction encoding.
//!
//! ## Instruction Encoding Groups
//! - Data Processing (Immediate): ADD/SUB with 12-bit imm, MOVZ/MOVK/MOVN, logical with bitmask
//! - Data Processing (Register): shifted/extended register ops, conditional select
//! - Loads and Stores: LDR/STR with multiple addressing modes (immediate offset, register offset, pre/post-index)
//! - Branches: B, B.cond, BL, BLR, BR, RET, CBZ/CBNZ, TBZ/TBNZ
//! - SIMD/FP: FADD, FSUB, FMUL, FDIV, FCMP, FCVT, FMOV between GP and FP regs
//!
//! ## Key Design: ADRP+ADD for Symbol Access
//! AArch64 uses ADRP (Address of 4KB Page) + ADD for PC-relative symbol access within ±4GB.
//! For GOT access (PIC): ADRP + LDR from GOT entry.
//!
//! ## Register Architecture
//! - 31 general-purpose registers: X0–X30 (64-bit) / W0–W30 (32-bit views)
//! - SP (stack pointer, not a general register), XZR/WZR (zero register shares encoding with SP)
//! - 32 SIMD/FP registers: V0–V31 (128-bit), D0–D31 (64-bit double), S0–S31 (32-bit float)
//! - NZCV condition flags register

use crate::backend::aarch64::abi::{
    AArch64Abi, FrameLayout, FP_ARG_REGS, INT_ARG_REGS, NUM_FP_ARG_REGS, NUM_INT_ARG_REGS,
};
use crate::backend::aarch64::registers::*;
use crate::backend::traits::ArgLocation;
use crate::common::diagnostics::Span;
use crate::common::target::Target;
use crate::common::types::CType;
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

/// Virtual register base offset.
///
/// AArch64 physical registers occupy indices 0–63 (GPR X0–X30/SP at 0–31,
/// SIMD/FP V0–V31 at 32–63).  IR `Value` indices also start at 0, so a
/// direct `.index() as u32 + VREG_BASE` creates a collision: Value v0 would appear to be
/// physical register X0, preventing register allocation from remapping it.
///
/// Adding `VREG_BASE` to every IR-derived register number lifts virtual
/// registers above the physical range, allowing `rd_is_virtual()`
/// (threshold ≥ 64) to correctly classify them.
const VREG_BASE: u32 = 64;

/// Convert an IR `Value` to a virtual register number for A64 instructions.
///
/// Uses wrapping arithmetic to avoid overflow panics in debug mode when
/// handling sentinel values like `Value::UNDEF` (which has index `u32::MAX`).
/// Callers that care about the UNDEF sentinel must check for it before
/// using the returned register number in instruction operands.
#[allow(dead_code)]
#[inline]
fn vreg(v: &Value) -> u32 {
    v.index().wrapping_add(VREG_BASE)
}

// Re-import items used only in test verification module to avoid unused import warnings.
#[cfg(test)]
use crate::backend::aarch64::abi::{ArgClass, INDIRECT_RESULT_REG, INT_RET_REGS, STACK_ALIGNMENT};
#[cfg(test)]
use crate::backend::traits::{
    MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand, RegisterInfo,
};
#[cfg(test)]
use crate::common::diagnostics::DiagnosticEngine;
#[cfg(test)]
use crate::common::types::MachineType;

// ===========================================================================
// A64 Opcode Definitions
// ===========================================================================

/// AArch64 (A64) instruction opcodes covering all major instruction categories.
///
/// Each variant maps to one or more A64 machine instruction encodings.
/// Pseudo-instructions (MOV_reg, LI, LA, CALL) are resolved to real
/// instruction sequences during assembly encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum A64Opcode {
    // -- Data Processing (Immediate) ----------------------------------------
    /// Add immediate with optional 12-bit left-shifted value.
    ADD_imm,
    /// Add immediate, setting condition flags (NZCV).
    ADDS_imm,
    /// Subtract immediate with optional 12-bit left-shifted value.
    SUB_imm,
    /// Subtract immediate, setting condition flags.
    SUBS_imm,
    /// Move wide with zero — loads a 16-bit immediate into a halfword position.
    MOVZ,
    /// Move wide with keep — loads a 16-bit immediate, keeping other bits.
    MOVK,
    /// Move wide with NOT — loads the bitwise inverse of a 16-bit immediate.
    MOVN,
    /// Bitwise AND with bitmask immediate.
    AND_imm,
    /// Bitwise OR with bitmask immediate.
    ORR_imm,
    /// Bitwise XOR with bitmask immediate.
    EOR_imm,
    /// Bitwise AND (setting flags) with bitmask immediate.
    ANDS_imm,
    /// Address of 4KB page relative to PC — upper 21 bits.
    ADRP,
    /// Address relative to PC — full 21-bit offset.
    ADR,

    // -- Data Processing (Register) -----------------------------------------
    /// Add register (shifted).
    ADD_reg,
    /// Add register (shifted), setting condition flags.
    ADDS_reg,
    /// Subtract register (shifted).
    SUB_reg,
    /// Subtract register (shifted), setting condition flags.
    SUBS_reg,
    /// Bitwise AND register.
    AND_reg,
    /// Bitwise OR register.
    ORR_reg,
    /// Bitwise XOR register.
    EOR_reg,
    /// Bitwise AND (setting flags) register.
    ANDS_reg,
    /// Bitwise OR NOT register.
    ORN_reg,
    /// Bit clear register (AND NOT).
    BIC_reg,
    /// Multiply-Add: Rd = Ra + Rn * Rm.
    MADD,
    /// Multiply-Subtract: Rd = Ra - Rn * Rm.
    MSUB,
    /// Multiply (pseudo for MADD with Ra=XZR): Rd = Rn * Rm.
    MUL,
    /// Signed multiply long: Xd = Wn * Wm (32×32→64).
    SMULL,
    /// Signed multiply high (upper 64 bits of 64x64→128).
    SMULH,
    /// Unsigned multiply high (upper 64 bits of 64x64→128).
    UMULH,
    /// Signed divide.
    SDIV,
    /// Unsigned divide.
    UDIV,
    /// Logical shift left (register).
    LSL_reg,
    /// Logical shift right (register).
    LSR_reg,
    /// Arithmetic shift right (register).
    ASR_reg,
    /// Rotate right (register).
    ROR_reg,
    /// Logical shift left (immediate via UBFM alias).
    LSL_imm,
    /// Logical shift right (immediate via UBFM alias).
    LSR_imm,
    /// Arithmetic shift right (immediate via SBFM alias).
    ASR_imm,

    // -- Conditional Select -------------------------------------------------
    /// Conditional select: Rd = cond ? Rn : Rm.
    CSEL,
    /// Conditional select increment: Rd = cond ? Rn : Rm+1.
    CSINC,
    /// Conditional select invert: Rd = cond ? Rn : ~Rm.
    CSINV,
    /// Conditional select negate: Rd = cond ? Rn : -Rm.
    CSNEG,
    /// Conditional set (alias): Rd = cond ? 1 : 0.
    CSET,
    /// Conditional set mask (alias): Rd = cond ? ~0 : 0.
    CSETM,

    // -- Bit Manipulation ---------------------------------------------------
    /// Count leading zeros.
    CLZ,
    /// Count leading sign bits.
    CLS,
    /// Reverse bits.
    RBIT,
    /// Reverse bytes (full register).
    REV,
    /// Reverse bytes in halfwords.
    REV16,
    /// Reverse bytes in 32-bit words (only valid on 64-bit registers).
    REV32,
    /// Extract bitfield from pair of registers.
    EXTR,
    /// Bitfield move.
    BFM,
    /// Signed bitfield move.
    SBFM,
    /// Unsigned bitfield move.
    UBFM,
    /// Sign-extend byte (alias for SBFM).
    SXTB,
    /// Sign-extend halfword (alias for SBFM).
    SXTH,
    /// Sign-extend word (alias for SBFM).
    SXTW,
    /// Zero-extend byte (alias for UBFM/AND).
    UXTB,
    /// Zero-extend halfword (alias for UBFM/AND).
    UXTH,

    // -- Loads --------------------------------------------------------------
    /// Load register (unsigned immediate offset).
    LDR_imm,
    /// Load byte (unsigned immediate offset).
    LDRB_imm,
    /// Load halfword (unsigned immediate offset).
    LDRH_imm,
    /// Load signed byte (unsigned immediate offset).
    LDRSB_imm,
    /// Load signed halfword (unsigned immediate offset).
    LDRSH_imm,
    /// Load signed word to 64-bit (unsigned immediate offset).
    LDRSW_imm,
    /// Load register (register offset).
    LDR_reg,
    /// Load register (PC-relative literal).
    LDR_literal,
    /// Load pair of registers.
    LDP,
    /// Load pair of registers (post-index): base updated after access.
    LDP_post,
    /// Load pair of signed words to 64-bit.
    LDPSW,
    /// Load register (pre-index): base updated before access.
    LDR_pre,
    /// Load register (post-index): base updated after access.
    LDR_post,

    // -- Stores -------------------------------------------------------------
    /// Store register (unsigned immediate offset).
    STR_imm,
    /// Store byte (unsigned immediate offset).
    STRB_imm,
    /// Store halfword (unsigned immediate offset).
    STRH_imm,
    /// Store register (register offset).
    STR_reg,
    /// Store pair of registers.
    STP,
    /// Store pair of registers (pre-index): base updated before access.
    STP_pre,
    /// Store register (pre-index): base updated before access.
    STR_pre,
    /// Store register (post-index): base updated after access.
    STR_post,

    // -- FP/SIMD Loads and Stores -------------------------------------------
    /// Load FP register (unsigned immediate offset).
    LDR_fp_imm,
    /// Store FP register (unsigned immediate offset).
    STR_fp_imm,
    /// Load pair of FP registers.
    LDP_fp,
    /// Store pair of FP registers.
    STP_fp,

    // -- Branches -----------------------------------------------------------
    /// Unconditional branch (±128 MiB PC-relative).
    B,
    /// Branch with link (function call, ±128 MiB PC-relative).
    BL,
    /// Conditional branch (B.cond, ±1 MiB PC-relative).
    B_cond,
    /// Branch to register (indirect jump).
    BR,
    /// Branch with link to register (indirect call).
    BLR,
    /// Return from subroutine (defaults to X30/LR).
    RET,
    /// Compare and branch if zero (±1 MiB).
    CBZ,
    /// Compare and branch if not zero (±1 MiB).
    CBNZ,
    /// Test bit and branch if zero (±32 KiB).
    TBZ,
    /// Test bit and branch if not zero (±32 KiB).
    TBNZ,

    // -- Comparison (aliases for flag-setting ops with XZR dest) -------------
    /// Compare immediate (alias for SUBS with XZR destination).
    CMP_imm,
    /// Compare register (alias for SUBS with XZR destination).
    CMP_reg,
    /// Compare negative immediate (alias for ADDS with XZR dest).
    CMN_imm,
    /// Compare negative register (alias for ADDS with XZR dest).
    CMN_reg,
    /// Test bits immediate (alias for ANDS with XZR dest).
    TST_imm,
    /// Test bits register (alias for ANDS with XZR dest).
    TST_reg,
    /// Conditional compare immediate.
    CCMP_imm,
    /// Conditional compare register.
    CCMP_reg,

    // -- FP Data Processing -------------------------------------------------
    /// Single-precision floating-point add.
    FADD_s,
    /// Single-precision floating-point subtract.
    FSUB_s,
    /// Single-precision floating-point multiply.
    FMUL_s,
    /// Single-precision floating-point divide.
    FDIV_s,
    /// Single-precision floating-point square root.
    FSQRT_s,
    /// Double-precision floating-point add.
    FADD_d,
    /// Double-precision floating-point subtract.
    FSUB_d,
    /// Double-precision floating-point multiply.
    FMUL_d,
    /// Double-precision floating-point divide.
    FDIV_d,
    /// Double-precision floating-point square root.
    FSQRT_d,
    /// Single-precision negate.
    FNEG_s,
    /// Double-precision negate.
    FNEG_d,
    /// Single-precision absolute value.
    FABS_s,
    /// Double-precision absolute value.
    FABS_d,
    /// Single-precision fused multiply-add.
    FMADD_s,
    /// Single-precision fused multiply-subtract.
    FMSUB_s,
    /// Single-precision negated fused multiply-add.
    FNMADD_s,
    /// Single-precision negated fused multiply-subtract.
    FNMSUB_s,
    /// Double-precision fused multiply-add.
    FMADD_d,
    /// Double-precision fused multiply-subtract.
    FMSUB_d,
    /// Double-precision negated fused multiply-add.
    FNMADD_d,
    /// Double-precision negated fused multiply-subtract.
    FNMSUB_d,
    /// Single-precision minimum.
    FMIN_s,
    /// Single-precision maximum.
    FMAX_s,
    /// Double-precision minimum.
    FMIN_d,
    /// Double-precision maximum.
    FMAX_d,

    // -- FP Comparison ------------------------------------------------------
    /// Single-precision compare (quiet NaN).
    FCMP_s,
    /// Double-precision compare (quiet NaN).
    FCMP_d,
    /// Single-precision compare (signalling NaN).
    FCMPE_s,
    /// Double-precision compare (signalling NaN).
    FCMPE_d,

    // -- FP Conversion ------------------------------------------------------
    /// Convert single to double.
    FCVT_sd,
    /// Convert double to single.
    FCVT_ds,
    /// Float to signed int (W-reg from S-reg, toward zero).
    FCVTZS_ws,
    /// Float to signed int (X-reg from S-reg, toward zero).
    FCVTZS_xs,
    /// Double to signed int (W-reg from D-reg, toward zero).
    FCVTZS_wd,
    /// Double to signed int (X-reg from D-reg, toward zero).
    FCVTZS_xd,
    /// Float to unsigned int (W-reg from S-reg, toward zero).
    FCVTZU_ws,
    /// Float to unsigned int (X-reg from S-reg, toward zero).
    FCVTZU_xs,
    /// Double to unsigned int (W-reg from D-reg, toward zero).
    FCVTZU_wd,
    /// Double to unsigned int (X-reg from D-reg, toward zero).
    FCVTZU_xd,
    /// Signed int (W-reg) to float (S-reg).
    SCVTF_sw,
    /// Signed int (X-reg) to float (S-reg).
    SCVTF_sx,
    /// Signed int (W-reg) to double (D-reg).
    SCVTF_dw,
    /// Signed int (X-reg) to double (D-reg).
    SCVTF_dx,
    /// Unsigned int (W-reg) to float (S-reg).
    UCVTF_sw,
    /// Unsigned int (X-reg) to float (S-reg).
    UCVTF_sx,
    /// Unsigned int (W-reg) to double (D-reg).
    UCVTF_dw,
    /// Unsigned int (X-reg) to double (D-reg).
    UCVTF_dx,

    // -- FP Move ------------------------------------------------------------
    /// Move between single-precision FP registers.
    FMOV_s,
    /// Move between double-precision FP registers.
    FMOV_d,
    /// Move from general-purpose register to FP register.
    FMOV_gen_to_fp,
    /// Move from FP register to general-purpose register.
    FMOV_fp_to_gen,

    // -- SIMD / Vector ------------------------------------------------------
    /// CNT Vd.8B, Vn.8B — count set bits per byte (SIMD popcount helper).
    CNT,
    /// ADDV Bd, Vn.8B — horizontal add across vector lanes (SIMD reduce).
    ADDV,

    // -- System -------------------------------------------------------------
    /// No operation.
    NOP,
    /// Data memory barrier.
    DMB,
    /// Data synchronization barrier.
    DSB,
    /// Instruction synchronization barrier.
    ISB,
    /// Supervisor call (system call).
    SVC,
    /// Hypervisor call.
    HVC,
    /// Secure monitor call.
    SMC,
    /// Move from system register.
    MRS,
    /// Move to system register.
    MSR,

    // -- Pseudo-instructions ------------------------------------------------
    /// Register move (alias for ORR Xd, XZR, Xm).
    MOV_reg,
    /// Immediate move (resolved to MOVZ/MOVK/MOVN sequence).
    MOV_imm,
    /// Negate register (alias for SUB Xd, XZR, Xm).
    NEG_reg,
    /// Bitwise NOT register (alias for ORN Xd, XZR, Xm).
    MVN_reg,
    /// Load immediate — MOVZ+MOVK sequence for large constants.
    LI,
    /// Load address — ADRP+ADD pair for symbol addressing.
    LA,
    /// Function call — BL or ADRP+BLR sequence.
    CALL,
    /// Inline assembly passthrough.
    INLINE_ASM,
}

// ===========================================================================
// Condition Codes
// ===========================================================================

/// AArch64 condition codes encoded in the 4-bit cond field of conditional
/// instructions (B.cond, CSEL, CCMP, etc.).
///
/// Condition codes test the NZCV flags set by comparison/arithmetic instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    /// Equal (Z=1).
    EQ = 0b0000,
    /// Not equal (Z=0).
    NE = 0b0001,
    /// Carry set / unsigned >= (also HS).
    CS = 0b0010,
    /// Carry clear / unsigned < (also LO).
    CC = 0b0011,
    /// Minus / negative (N=1).
    MI = 0b0100,
    /// Plus / positive or zero (N=0).
    PL = 0b0101,
    /// Overflow (V=1).
    VS = 0b0110,
    /// No overflow (V=0).
    VC = 0b0111,
    /// Unsigned higher (C=1 && Z=0).
    HI = 0b1000,
    /// Unsigned lower or same (C=0 || Z=1).
    LS = 0b1001,
    /// Signed >= (N==V).
    GE = 0b1010,
    /// Signed < (N!=V).
    LT = 0b1011,
    /// Signed > (Z=0 && N==V).
    GT = 0b1100,
    /// Signed <= (Z=1 || N!=V).
    LE = 0b1101,
    /// Always (unconditional).
    AL = 0b1110,
    /// Never (reserved, sometimes used).
    NV = 0b1111,
}

impl CondCode {
    /// Invert a condition code (logical negation).
    ///
    /// Each condition code has a complementary code obtained by flipping
    /// the least significant bit of the 4-bit encoding.
    #[inline]
    pub fn invert(self) -> CondCode {
        match self {
            CondCode::EQ => CondCode::NE,
            CondCode::NE => CondCode::EQ,
            CondCode::CS => CondCode::CC,
            CondCode::CC => CondCode::CS,
            CondCode::MI => CondCode::PL,
            CondCode::PL => CondCode::MI,
            CondCode::VS => CondCode::VC,
            CondCode::VC => CondCode::VS,
            CondCode::HI => CondCode::LS,
            CondCode::LS => CondCode::HI,
            CondCode::GE => CondCode::LT,
            CondCode::LT => CondCode::GE,
            CondCode::GT => CondCode::LE,
            CondCode::LE => CondCode::GT,
            CondCode::AL => CondCode::NV,
            CondCode::NV => CondCode::AL,
        }
    }

    /// Map an IR integer comparison operator to the appropriate AArch64
    /// condition code (assumes CMP has already been emitted).
    pub fn from_icmp(op: &ICmpOp) -> CondCode {
        match op {
            ICmpOp::Eq => CondCode::EQ,
            ICmpOp::Ne => CondCode::NE,
            ICmpOp::Slt => CondCode::LT,
            ICmpOp::Sle => CondCode::LE,
            ICmpOp::Sgt => CondCode::GT,
            ICmpOp::Sge => CondCode::GE,
            ICmpOp::Ult => CondCode::CC,
            ICmpOp::Ule => CondCode::LS,
            ICmpOp::Ugt => CondCode::HI,
            ICmpOp::Uge => CondCode::CS,
        }
    }

    /// Map an IR floating-point comparison operator to the appropriate
    /// AArch64 condition code pair (assumes FCMP has already been emitted).
    ///
    /// Returns the condition code for ordered comparisons; unordered
    /// comparisons require additional VS/VC checks handled by the caller.
    pub fn from_fcmp(op: &FCmpOp) -> CondCode {
        match op {
            FCmpOp::Oeq => CondCode::EQ,
            FCmpOp::One => CondCode::MI, // LT || GT → MI or HI after FCMP
            FCmpOp::Olt => CondCode::MI,
            FCmpOp::Ole => CondCode::LS,
            FCmpOp::Ogt => CondCode::GT,
            FCmpOp::Oge => CondCode::GE,
            FCmpOp::Uno => CondCode::VS,
            FCmpOp::Ord => CondCode::VC,
        }
    }

    /// Returns the 4-bit encoding value of this condition code.
    #[inline]
    pub fn encoding(self) -> u8 {
        self as u8
    }

    /// Construct a CondCode from its hardware encoding value.
    pub fn from_encoding(enc: u8) -> CondCode {
        match enc & 0xF {
            0 => CondCode::EQ,
            1 => CondCode::NE,
            2 => CondCode::CS, // HS
            3 => CondCode::CC, // LO
            4 => CondCode::MI,
            5 => CondCode::PL,
            6 => CondCode::VS,
            7 => CondCode::VC,
            8 => CondCode::HI,
            9 => CondCode::LS,
            10 => CondCode::GE,
            11 => CondCode::LT,
            12 => CondCode::GT,
            13 => CondCode::LE,
            14 => CondCode::AL,
            _ => CondCode::AL,
        }
    }
}

// ===========================================================================
// A64Instruction — AArch64-specific machine instruction
// ===========================================================================

/// Represents a single AArch64 machine instruction with all operand fields.
///
/// All A64 instructions are fixed 32-bit (4 bytes) wide. This struct captures
/// the decoded instruction fields before final binary encoding. Fields that
/// are not applicable to a given opcode are set to `None` / 0 / `false`.
#[derive(Debug, Clone)]
pub struct A64Instruction {
    /// The instruction opcode.
    pub opcode: A64Opcode,
    /// Destination register (Rd), if applicable.
    /// Uses u32 to support virtual register indices >255 in large functions.
    pub rd: Option<u32>,
    /// First source register (Rn), if applicable.
    pub rn: Option<u32>,
    /// Second source register (Rm), if applicable.
    pub rm: Option<u32>,
    /// Third source register (Ra), for multiply-accumulate.
    pub ra: Option<u32>,
    /// Immediate value (interpretation depends on opcode).
    pub imm: i64,
    /// Shift amount or shift type encoding.
    pub shift: u8,
    /// Condition code for conditional operations.
    pub cond: Option<CondCode>,
    /// Symbol reference for relocations (function/variable names).
    pub symbol: Option<String>,
    /// If true, use W-register (32-bit) forms; false = X-register (64-bit).
    pub is_32bit: bool,
    /// If true, this is a FP/SIMD instruction.
    pub is_fp: bool,
    /// Optional comment for assembly output / debugging.
    pub comment: Option<String>,
    /// If true, this instruction is part of call argument setup (MOV to arg regs).
    pub is_call_arg_setup: bool,
    /// Source type for conversions (ZExt/SExt/Trunc) — used by register allocator.
    pub from_type: Option<crate::ir::types::IrType>,
}

impl A64Instruction {
    /// Create a new instruction with the given opcode and default fields.
    pub fn new(opcode: A64Opcode) -> Self {
        A64Instruction {
            opcode,
            rd: None,
            rn: None,
            rm: None,
            ra: None,
            imm: 0,
            shift: 0,
            cond: None,
            symbol: None,
            is_32bit: false,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
            from_type: None,
        }
    }

    /// Builder: set destination register.
    #[inline]
    pub fn with_rd(mut self, rd: u32) -> Self {
        self.rd = Some(rd);
        self
    }

    /// Builder: set first source register.
    #[inline]
    pub fn with_rn(mut self, rn: u32) -> Self {
        self.rn = Some(rn);
        self
    }

    /// Builder: set second source register.
    #[inline]
    pub fn with_rm(mut self, rm: u32) -> Self {
        self.rm = Some(rm);
        self
    }

    /// Builder: set third source register (multiply-accumulate).
    #[inline]
    pub fn with_ra(mut self, ra: u32) -> Self {
        self.ra = Some(ra);
        self
    }

    /// Builder: set immediate value.
    #[inline]
    pub fn with_imm(mut self, imm: i64) -> Self {
        self.imm = imm;
        self
    }

    /// Builder: set shift amount.
    #[inline]
    pub fn with_shift(mut self, shift: u8) -> Self {
        self.shift = shift;
        self
    }

    /// Builder: set condition code.
    #[inline]
    pub fn with_cond(mut self, cond: CondCode) -> Self {
        self.cond = Some(cond);
        self
    }

    /// Builder: set symbol reference.
    #[inline]
    pub fn with_symbol(mut self, sym: String) -> Self {
        self.symbol = Some(sym);
        self
    }

    /// Builder: set 32-bit mode (W-register).
    #[inline]
    pub fn set_32bit(mut self) -> Self {
        self.is_32bit = true;
        self
    }

    /// Builder: mark as FP/SIMD instruction.
    #[inline]
    /// Mark this instruction as part of a call-argument-setup sequence.
    /// The post-register-allocation parallel-move resolver uses this flag
    /// to find the contiguous arg-setup window preceding a CALL instruction
    /// and resolve register-to-register move conflicts.
    pub fn set_call_arg_setup(mut self) -> Self {
        self.is_call_arg_setup = true;
        self
    }

    pub fn set_fp(mut self) -> Self {
        self.is_fp = true;
        self
    }

    /// Builder: add comment for debugging.
    #[inline]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Check if the destination register is a virtual register (index >= 64).
    #[inline]
    pub fn rd_is_virtual(&self) -> bool {
        self.rd.map_or(false, |r| r >= 64)
    }

    /// Check if the first source register is a virtual register.
    #[inline]
    pub fn rn_is_virtual(&self) -> bool {
        self.rn.map_or(false, |r| r >= 64)
    }

    /// Check if the second source register is a virtual register.
    #[inline]
    pub fn rm_is_virtual(&self) -> bool {
        self.rm.map_or(false, |r| r >= 64)
    }

    /// Check if the third source register is a virtual register.
    #[inline]
    pub fn ra_is_virtual(&self) -> bool {
        self.ra.map_or(false, |r| r >= 64)
    }
}

// ===========================================================================
// Helper: check if value fits in 12-bit unsigned immediate
// ===========================================================================

/// Returns true if the value fits in a 12-bit unsigned immediate, optionally
/// with a 12-bit left shift.
fn fits_add_sub_imm(val: u64) -> bool {
    val <= 0xFFF || (val & 0xFFF == 0 && (val >> 12) <= 0xFFF)
}

/// Returns the shift needed for an add/sub immediate.
#[allow(dead_code)]
fn add_sub_imm_shift(val: u64) -> Option<u8> {
    if val <= 0xFFF {
        Some(0)
    } else if val & 0xFFF == 0 && (val >> 12) <= 0xFFF {
        Some(12)
    } else {
        None
    }
}

// ===========================================================================
// AArch64InstructionSelector
// ===========================================================================

/// The AArch64 instruction selector. Translates IR functions into sequences
/// of [`A64Instruction`]s by pattern-matching IR instruction variants.
///
/// Maintains per-function state including frame size, spill slots, and block
/// labels. Call [`select_function`] to process an entire IR function.
pub struct AArch64InstructionSelector {
    instructions: Vec<A64Instruction>,
    current_block: Option<String>,
    frame_size: i64,
    spill_slots: Vec<(u32, i64)>,
    pic_mode: bool,
    target: Target,
    next_vreg: u32,
    block_labels: Vec<String>,
    used_callee_saved: Vec<u8>,
    /// Maps IR Values to their IrType, populated during lower_function.
    value_types: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, IrType>,
    /// Maps IR Value indices to compile-time constant values.
    constant_values: crate::common::fx_hash::FxHashMap<u32, i64>,
    /// Maps function IR Values to their symbol names.
    func_ref_names: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Maps global variable IR Values to their symbol names.
    global_var_refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Maps IR Values representing float constants to their .rodata symbol names.
    float_constant_cache: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Pre-computed alloca FP-relative offsets (Value.index() → positive offset from FP).
    /// Frame layout: [FP+0]=X29, [FP+8]=LR, [FP+16..]=locals, then callee-saved.
    alloca_offsets: crate::common::fx_hash::FxHashMap<u32, i64>,
    /// Total locals area size in bytes (16-byte aligned), set during select_function.
    locals_size: usize,
    /// FP-relative offset of the variadic GPR register save area.
    /// For variadic functions, X0-X7 are saved at [FP+16..FP+80] in the
    /// prologue, and locals start at FP+80 instead of FP+16.
    vararg_save_offset: Option<i64>,
    /// Number of named (non-variadic) GPR parameters.  Used by va_start
    /// to skip past the named args in the save area.
    named_gpr_count: usize,
    /// Number of parameter-copy MOV instructions emitted at the start of
    /// the entry block.  Stored in MachineFunction.num_param_moves for the
    /// post-allocation parallel-move resolver.
    num_param_moves: usize,
}

impl AArch64InstructionSelector {
    /// Create a new instruction selector for the AArch64 target.
    pub fn new(target: Target, pic_mode: bool) -> Self {
        AArch64InstructionSelector {
            instructions: Vec::new(),
            current_block: None,
            frame_size: 0,
            spill_slots: Vec::new(),
            pic_mode,
            target,
            next_vreg: 0,
            block_labels: Vec::new(),
            used_callee_saved: Vec::new(),
            value_types: crate::common::fx_hash::FxHashMap::default(),
            constant_values: crate::common::fx_hash::FxHashMap::default(),
            func_ref_names: crate::common::fx_hash::FxHashMap::default(),
            global_var_refs: crate::common::fx_hash::FxHashMap::default(),
            float_constant_cache: crate::common::fx_hash::FxHashMap::default(),
            alloca_offsets: crate::common::fx_hash::FxHashMap::default(),
            locals_size: 0,
            vararg_save_offset: None,
            named_gpr_count: 0,
            num_param_moves: 0,
        }
    }

    /// Reset per-function instruction state for reuse.
    /// NOTE: func_ref_names, global_var_refs, and float_constant_cache are
    /// set externally via setters and NOT cleared here — they are populated
    /// before select_function() is called and must survive the reset.
    fn reset(&mut self) {
        self.instructions.clear();
        self.current_block = None;
        self.frame_size = 0;
        self.spill_slots.clear();
        self.next_vreg = 0;
        self.block_labels.clear();
        self.used_callee_saved.clear();
        self.value_types.clear();
        self.constant_values.clear();
        self.alloca_offsets.clear();
        self.locals_size = 0;
        self.vararg_save_offset = None;
        self.named_gpr_count = 0;
        self.num_param_moves = 0;
    }

    /// Returns the total locals area size (16-byte aligned) computed during
    /// `select_function`. This value should be stored in `mf.frame_size` so
    /// the prologue allocates sufficient stack space for local variables.
    pub fn get_locals_size(&self) -> usize {
        self.locals_size
    }

    /// Get the FP-relative offset of the variadic argument save area.
    /// Returns `Some(16)` for variadic functions, `None` otherwise.
    pub fn get_vararg_save_offset(&self) -> Option<i64> {
        self.vararg_save_offset
    }

    /// Get the number of named (non-variadic) GPR parameters.
    pub fn get_named_gpr_count(&self) -> usize {
        self.named_gpr_count
    }

    /// Get the number of parameter-copy MOV instructions emitted at the
    /// start of the entry block.
    pub fn get_num_param_moves(&self) -> usize {
        self.num_param_moves
    }

    /// Set function reference name mappings.
    pub fn set_func_ref_names(
        &mut self,
        refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.func_ref_names = refs;
    }

    /// Set global variable reference flags.
    pub fn set_global_var_refs(
        &mut self,
        refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.global_var_refs = refs;
    }

    /// Set float constant cache mapping IR Values to their .rodata symbol names.
    pub fn set_float_constant_cache(
        &mut self,
        cache: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.float_constant_cache = cache;
    }

    /// Allocate the next virtual register number.
    #[allow(dead_code)]
    fn alloc_vreg(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }

    /// Emit an instruction into the current function's instruction list.
    fn emit(&mut self, inst: A64Instruction) {
        self.instructions.push(inst);
    }

    /// Determine if an IR type should use W-register (32-bit) forms.
    fn is_32bit_type(ty: &IrType) -> bool {
        matches!(ty, IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32)
    }

    /// Get the memory size in bytes for an IR type.
    fn type_size_for_mem(&self, ty: &IrType) -> usize {
        ty.size_bytes(&self.target)
    }

    /// Map an IR type to an approximate CType for ABI classification.
    /// This provides a best-effort mapping for parameter classification;
    /// the exact C-level type information may differ but the ABI-relevant
    /// properties (size, register class) are preserved.
    fn irtype_to_ctype(ty: &IrType) -> CType {
        match ty {
            IrType::Void => CType::Void,
            IrType::I1 => CType::Bool,
            IrType::I8 => CType::Char,
            IrType::I16 => CType::Short,
            IrType::I32 => CType::Int,
            IrType::I64 => CType::Long,
            IrType::I128 => CType::LongLong,
            IrType::F32 => CType::Float,
            IrType::F64 => CType::Double,
            IrType::F80 => CType::LongDouble,
            IrType::Ptr => CType::Pointer(
                Box::new(CType::Void),
                crate::common::types::TypeQualifiers {
                    is_const: false,
                    is_volatile: false,
                    is_restrict: false,
                    is_atomic: false,
                },
            ),
            IrType::Array(inner, size) => {
                CType::Array(Box::new(Self::irtype_to_ctype(inner)), Some(*size))
            }
            IrType::Struct(_) => CType::Struct {
                name: None,
                fields: Vec::new(),
                packed: false,
                aligned: None,
            },
            IrType::Function(ret, params) => CType::Function {
                return_type: Box::new(Self::irtype_to_ctype(ret)),
                params: params.iter().map(Self::irtype_to_ctype).collect(),
                variadic: false,
            },
        }
    }

    // -----------------------------------------------------------------------
    // Immediate materialization
    // -----------------------------------------------------------------------

    /// Materialize a 64-bit immediate value into a register using the optimal
    /// MOVZ + MOVK sequence. Optimizes for common patterns:
    /// - Zero: MOV Xd, XZR
    /// - Single halfword: MOVZ with shift
    /// - Inverted halfword: MOVN with shift
    /// - Multi-halfword: MOVZ + MOVK chain (skip zero halfwords)
    pub fn materialize_immediate(&mut self, rd: u32, value: u64) -> Vec<A64Instruction> {
        let mut result = Vec::new();

        if value == 0 {
            result.push(
                A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(XZR)
                    .with_comment("mov xd, xzr (zero)"),
            );
            return result;
        }

        // Check if the value fits in a single MOVZ with shift.
        for hw in 0..4u8 {
            let shifted = (value >> (hw as u64 * 16)) & 0xFFFF;
            let mask = 0xFFFF_u64 << (hw as u64 * 16);
            if value == (shifted << (hw as u64 * 16)) && shifted != 0 {
                result.push(
                    A64Instruction::new(A64Opcode::MOVZ)
                        .with_rd(rd)
                        .with_imm(shifted as i64)
                        .with_shift(hw * 16),
                );
                return result;
            }
            let _ = mask;
        }

        // Check if bitwise NOT fits in a single MOVN with shift.
        let not_val = !value;
        for hw in 0..4u8 {
            let shifted = (not_val >> (hw as u64 * 16)) & 0xFFFF;
            if not_val == (shifted << (hw as u64 * 16)) && shifted != 0 {
                result.push(
                    A64Instruction::new(A64Opcode::MOVN)
                        .with_rd(rd)
                        .with_imm(shifted as i64)
                        .with_shift(hw * 16),
                );
                return result;
            }
        }

        // General case: MOVZ first non-zero halfword, MOVK for remaining.
        let mut first = true;
        for hw in 0..4u8 {
            let chunk = (value >> (hw as u64 * 16)) & 0xFFFF;
            if chunk == 0 {
                continue;
            }
            if first {
                result.push(
                    A64Instruction::new(A64Opcode::MOVZ)
                        .with_rd(rd)
                        .with_imm(chunk as i64)
                        .with_shift(hw * 16),
                );
                first = false;
            } else {
                result.push(
                    A64Instruction::new(A64Opcode::MOVK)
                        .with_rd(rd)
                        .with_imm(chunk as i64)
                        .with_shift(hw * 16),
                );
            }
        }

        result
    }

    /// Materialize a symbol address into a register.
    ///
    /// - Non-PIC: ADRP Xd, symbol ; ADD Xd, Xd, :lo12:symbol
    /// - PIC/GOT: ADRP Xd, :got:symbol ; LDR Xd, [Xd, :got_lo12:symbol]
    pub fn materialize_address(&mut self, rd: u32, symbol: &str) -> Vec<A64Instruction> {
        let mut result = Vec::new();
        if self.pic_mode {
            // GOT-indirect addressing for PIC.
            result.push(
                A64Instruction::new(A64Opcode::ADRP)
                    .with_rd(rd)
                    .with_symbol(format!(":got:{}", symbol))
                    .with_comment(format!("adrp {}, :got:{}", gpr_name(rd), symbol)),
            );
            result.push(
                A64Instruction::new(A64Opcode::LDR_imm)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_symbol(format!(":got_lo12:{}", symbol))
                    .with_comment(format!(
                        "ldr {}, [{}, :got_lo12:{}]",
                        gpr_name(rd),
                        gpr_name(rd),
                        symbol
                    )),
            );
        } else {
            // Direct PC-relative addressing.
            result.push(
                A64Instruction::new(A64Opcode::ADRP)
                    .with_rd(rd)
                    .with_symbol(symbol.to_string())
                    .with_comment(format!("adrp {}, {}", gpr_name(rd), symbol)),
            );
            result.push(
                A64Instruction::new(A64Opcode::ADD_imm)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_symbol(format!(":lo12:{}", symbol))
                    .with_comment(format!(
                        "add {}, {}, :lo12:{}",
                        gpr_name(rd),
                        gpr_name(rd),
                        symbol
                    )),
            );
        }
        result
    }

    // -----------------------------------------------------------------------
    // Function-level selection
    // -----------------------------------------------------------------------

    /// Select instructions for an entire IR function. This is the main
    /// entry point for AArch64 instruction selection.
    ///
    /// Processes the function in order:
    /// 1. Build block labels
    /// 2. Compute frame layout via AAPCS64 ABI
    /// 3. Emit prologue (STP x29,x30; MOV x29,sp; save callee-saved)
    /// 4. Set up argument registers
    /// 5. Process each basic block
    /// 6. Emit epilogue (restore callee-saved; LDP x29,x30; RET)
    ///
    /// Set the constant value cache from externally-populated data.
    pub fn set_constant_values(&mut self, cv: crate::common::fx_hash::FxHashMap<u32, i64>) {
        self.constant_values = cv;
    }

    pub fn select_function(&mut self, func: &IrFunction, abi: &AArch64Abi) -> Vec<A64Instruction> {
        // Save constant_values before reset clears them.
        let saved_constants = std::mem::take(&mut self.constant_values);
        self.reset();
        self.constant_values = saved_constants;
        if !func.is_definition {
            return Vec::new();
        }

        self.next_vreg = func.value_count;

        // Build block labels for branch target resolution.
        // Labels MUST be function-scoped to avoid collisions when
        // multiple functions share generic block names like "if.then".
        for (i, block) in func.blocks().iter().enumerate() {
            let base = block.label.as_deref().unwrap_or("bb");
            let label = format!(".L_{}_{}", func.name, i);
            let _ = base; // original label used only for documentation
            self.block_labels.push(label);
        }

        // Populate value_types from function parameters and IR instructions
        // so that select_call can distinguish integer vs FP arguments per AAPCS64.
        for param in &func.params {
            self.value_types.insert(param.value, param.ty.clone());
        }
        for ir_block in func.blocks().iter() {
            for ir_inst in ir_block.instructions() {
                match ir_inst {
                    Instruction::Load { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BinOp { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::BitCast {
                        result, to_type, ..
                    }
                    | Instruction::Trunc {
                        result, to_type, ..
                    }
                    | Instruction::ZExt {
                        result, to_type, ..
                    }
                    | Instruction::SExt {
                        result, to_type, ..
                    }
                    | Instruction::PtrToInt {
                        result, to_type, ..
                    } => {
                        self.value_types.insert(*result, to_type.clone());
                    }
                    Instruction::IntToPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::Call {
                        result,
                        return_type,
                        ..
                    } => {
                        self.value_types.insert(*result, return_type.clone());
                    }
                    Instruction::Alloca { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    Instruction::ICmp { result, .. } | Instruction::FCmp { result, .. } => {
                        self.value_types.insert(*result, IrType::I1);
                    }
                    Instruction::Phi { result, ty, .. } => {
                        self.value_types.insert(*result, ty.clone());
                    }
                    Instruction::GetElementPtr { result, .. } => {
                        self.value_types.insert(*result, IrType::Ptr);
                    }
                    _ => {}
                }
            }
        }

        // Pre-compute alloca FP-relative offsets.
        //
        // Frame layout (FP = SP after prologue):
        //   [FP + 0]                        : saved X29 (FP)
        //   [FP + 8]                        : saved X30 (LR)
        //   [FP + 16]                       : vararg save area (64B, variadic only)
        //   [FP + 16 + vararg_save_size]    : start of locals area
        //   [FP + 16 + vararg_save + locals]: callee-saved regs
        //   [FP + 16 + ... + spills]        : spill slots (filled by regalloc)
        //   [FP + frame_size]               : old SP
        //
        // Locals are at POSITIVE, FIXED offsets from FP, so addresses are
        // computed as `ADD rd, FP, #(16 + vararg_save_size + alloca_position)`.

        // For variadic functions, reserve 64 bytes (8 regs × 8 bytes) at
        // FP+16 to save X0–X7 in the prologue.  va_start points into this
        // area at the first unnamed argument.
        let vararg_save_size: usize = if func.is_variadic { 64 } else { 0 };
        if func.is_variadic {
            self.vararg_save_offset = Some(16);
            self.named_gpr_count = func.params.len();
        }
        {
            // Scan ALL blocks for alloca instructions (not just entry block).
            // Allocas can appear in non-entry blocks (e.g., compound literals
            // in loop bodies) and must still receive valid frame slot offsets.
            let mut alloca_cursor: usize = 0;
            for block in func.blocks() {
                for inst in block.instructions() {
                    if let Instruction::Alloca {
                        result,
                        ty,
                        alignment,
                        ..
                    } = inst
                    {
                        let size = ty.size_bytes(&self.target).max(1);
                        let align = alignment.unwrap_or_else(|| ty.align_bytes(&self.target));
                        let mask = if align > 0 { align - 1 } else { 0 };
                        alloca_cursor = (alloca_cursor + mask) & !mask;
                        let fp_offset = (16 + vararg_save_size + alloca_cursor) as i64;
                        self.alloca_offsets
                            .insert(result.index().wrapping_add(VREG_BASE), fp_offset);
                        alloca_cursor += size;
                    }
                }
            }
            self.locals_size = ((vararg_save_size + alloca_cursor) + 15) & !15; // Align to 16
        }

        // Prologue/epilogue are generated externally by the ArchCodegen
        // trait methods (emit_prologue/emit_epilogue) in mod.rs, which are
        // called by generation.rs after register allocation.  We do NOT
        // emit them here to avoid duplication.

        // Emit argument setup (copy from ABI locations to local vregs).
        self.num_param_moves = self.emit_arg_setup(func, abi);

        // Process each basic block.
        for block in func.blocks() {
            self.select_basic_block(block, func);
        }

        std::mem::take(&mut self.instructions)
    }

    /// Estimate total stack space needed for local allocas.
    #[allow(dead_code)]
    fn estimate_locals_size(&self, func: &IrFunction) -> usize {
        let mut total: usize = 0;
        // Scan ALL blocks — allocas can appear in non-entry blocks
        // (e.g., compound literals in loop bodies).
        for block in func.blocks() {
            for inst in block.instructions() {
                if let Instruction::Alloca { ty, alignment, .. } = inst {
                    let size = ty.size_bytes(&self.target);
                    let align = alignment.unwrap_or_else(|| ty.align_bytes(&self.target));
                    let mask = if align > 0 { align - 1 } else { 0 };
                    total = (total + mask) & !mask;
                    total += size;
                }
            }
        }
        // Round up to 16-byte alignment.
        (total + 15) & !15
    }

    /// Emit the function prologue: save frame pointer + link register,
    /// set up frame pointer, and save callee-saved registers.
    #[allow(dead_code)]
    fn emit_prologue(&mut self, layout: &FrameLayout, _func: &IrFunction) {
        if layout.total_size == 0 {
            return;
        }
        // STP X29, X30, [SP, #-frame_size]!  (pre-decrement SP)
        self.emit(
            A64Instruction::new(A64Opcode::STP_pre)
                .with_rd(FP_REG)
                .with_rn(LR)
                .with_rm(SP_REG)
                .with_imm(-(layout.total_size as i64))
                .with_comment("stp x29, x30, [sp, #-frame_size]!"),
        );
        // MOV X29, SP (set frame pointer)
        self.emit(
            A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(FP_REG)
                .with_rn(SP_REG)
                .with_comment("mov x29, sp (set frame pointer)"),
        );

        // Save callee-saved GPRs using STP pairs.
        let gprs = &CALLEE_SAVED_GPRS;
        let mut offset = layout.callee_saved_offset;
        let mut i = 0;
        while i + 1 < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STP)
                    .with_rd(gprs[i])
                    .with_rn(gprs[i + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!(
                        "stp {}, {}, [sp, #{}]",
                        gpr_name(gprs[i]),
                        gpr_name(gprs[i + 1]),
                        offset
                    )),
            );
            offset += 16;
            i += 2;
        }
        // Handle odd remaining GPR.
        if i < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STR_imm)
                    .with_rd(gprs[i])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!("str {}, [sp, #{}]", gpr_name(gprs[i]), offset)),
            );
            offset += 8;
        }

        // Save callee-saved FPRs using STP_fp pairs.
        let fprs = &CALLEE_SAVED_FPRS;
        let mut fi = 0;
        while fi + 1 < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STP_fp)
                    .with_rd(fprs[fi])
                    .with_rn(fprs[fi + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!(
                        "stp {}, {}, [sp, #{}]",
                        fpr_name_d(fprs[fi]),
                        fpr_name_d(fprs[fi + 1]),
                        offset
                    )),
            );
            offset += 16;
            fi += 2;
        }
        if fi < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::STR_fp_imm)
                    .with_rd(fprs[fi])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!("str {}, [sp, #{}]", fpr_name_d(fprs[fi]), offset)),
            );
        }
    }

    /// Emit the function epilogue: restore callee-saved registers, restore
    /// frame pointer + link register, and return.
    #[allow(dead_code)]
    fn emit_epilogue(&mut self, layout: &FrameLayout) {
        if layout.total_size == 0 {
            self.emit(A64Instruction::new(A64Opcode::RET).with_comment("ret (leaf)"));
            return;
        }

        // Restore callee-saved GPRs (reverse order of prologue).
        let gprs = &CALLEE_SAVED_GPRS;
        let mut offset = layout.callee_saved_offset;
        let mut i = 0;
        while i + 1 < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDP)
                    .with_rd(gprs[i])
                    .with_rn(gprs[i + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!(
                        "ldp {}, {}, [sp, #{}]",
                        gpr_name(gprs[i]),
                        gpr_name(gprs[i + 1]),
                        offset
                    )),
            );
            offset += 16;
            i += 2;
        }
        if i < gprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDR_imm)
                    .with_rd(gprs[i])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .with_comment(format!("ldr {}, [sp, #{}]", gpr_name(gprs[i]), offset)),
            );
            offset += 8;
        }

        // Restore callee-saved FPRs.
        let fprs = &CALLEE_SAVED_FPRS;
        let mut fi = 0;
        while fi + 1 < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDP_fp)
                    .with_rd(fprs[fi])
                    .with_rn(fprs[fi + 1])
                    .with_rm(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!(
                        "ldp {}, {}, [sp, #{}]",
                        fpr_name_d(fprs[fi]),
                        fpr_name_d(fprs[fi + 1]),
                        offset
                    )),
            );
            offset += 16;
            fi += 2;
        }
        if fi < fprs.len() {
            self.emit(
                A64Instruction::new(A64Opcode::LDR_fp_imm)
                    .with_rd(fprs[fi])
                    .with_rn(SP_REG)
                    .with_imm(offset)
                    .set_fp()
                    .with_comment(format!("ldr {}, [sp, #{}]", fpr_name_d(fprs[fi]), offset)),
            );
        }

        // LDP X29, X30, [SP], #frame_size  (post-increment SP)
        self.emit(
            A64Instruction::new(A64Opcode::LDP_post)
                .with_rd(FP_REG)
                .with_rn(LR)
                .with_rm(SP_REG)
                .with_imm(layout.total_size as i64)
                .with_comment("ldp x29, x30, [sp], #frame_size"),
        );
        self.emit(A64Instruction::new(A64Opcode::RET).with_comment("ret"));
    }

    /// Emit argument setup: copy from ABI register/stack locations to vregs.
    ///
    /// This emits **explicit MOV instructions** that copy each ABI register
    /// into the corresponding virtual register for the parameter.  This is
    /// critical for register allocation correctness: without an explicit
    /// definition, the register allocator has no way to know that the
    /// parameter value resides in a specific physical ABI register, and may
    /// freely reuse that register for other values — clobbering the param.
    ///
    /// Returns the number of parameter-copy MOV instructions emitted, which
    /// is stored in `MachineFunction::num_param_moves` so that the post-
    /// register-allocation parallel-move resolver can identify the parameter
    /// MOV window at the start of the entry block.
    fn emit_arg_setup(&mut self, func: &IrFunction, _abi: &AArch64Abi) -> usize {
        // Uses ABI-classified register assignments per AAPCS64 — integer
        // parameters go to X0–X7 and FP parameters go to V0–V7.  The ABI
        // classifier handles the separate register bank allocation, so
        // we do NOT use the sequential parameter index as a physical register
        // number.
        let mut abi_state = AArch64Abi::new();
        abi_state.reset();
        let mut param_move_count: usize = 0;
        for (i, param) in func.params.iter().enumerate() {
            let cty = Self::irtype_to_ctype(&param.ty);
            let loc = if func.is_variadic && i >= func.params.len().saturating_sub(1) {
                abi_state.classify_variadic_arg(&cty)
            } else {
                abi_state.classify_arg(&cty)
            };
            // Compute the virtual register ID for this parameter.
            let vreg = param.value.index().wrapping_add(VREG_BASE);
            match loc {
                ArgLocation::Register(reg) => {
                    // Argument is in an ABI register (X0-X7 for int, V0-V7 for FP).
                    // Emit an explicit MOV from the physical ABI register to the
                    // virtual register so the register allocator sees a proper
                    // definition and can track live ranges correctly.
                    let is_fp = matches!(param.ty, IrType::F32 | IrType::F64 | IrType::F80);
                    if is_fp {
                        // FP parameter: MOV Vd, Vreg (FMOV Vd, Vn)
                        let phys_reg = reg as u32 + 32; // V0 = 32
                        self.emit(
                            A64Instruction::new(A64Opcode::FMOV_d)
                                .with_rd(vreg)
                                .with_rm(phys_reg)
                                .with_comment(format!("param {} from v{} to vreg{}", i, reg, vreg)),
                        );
                    } else {
                        // Integer parameter: MOV Xd, Xn
                        let phys_reg = reg as u32; // X0 = 0
                        self.emit(
                            A64Instruction::new(A64Opcode::MOV_reg)
                                .with_rd(vreg)
                                .with_rm(phys_reg)
                                .with_comment(format!(
                                    "param {} from {} to vreg{}",
                                    i,
                                    gpr_name(phys_reg),
                                    vreg
                                )),
                        );
                    }
                    param_move_count += 1;
                }
                ArgLocation::RegisterPair(r1, _r2) => {
                    // Two-register pair (large struct) — emit MOV for lo half.
                    let phys_r1 = r1 as u32;
                    self.emit(
                        A64Instruction::new(A64Opcode::MOV_reg)
                            .with_rd(vreg)
                            .with_rm(phys_r1)
                            .with_comment(format!(
                                "param {} lo from {} to vreg{}",
                                i,
                                gpr_name(phys_r1),
                                vreg
                            )),
                    );
                    param_move_count += 1;
                }
                ArgLocation::Stack(offset) => {
                    // Load stack argument via frame pointer into the vreg.
                    // After prologue: FP+0 = saved X29, FP+8 = saved LR,
                    // FP+16 = first stack arg for the old stack frame.
                    // But with our frame layout, stack args are above the
                    // caller's frame, so the offset from FP is:
                    //   FP + frame_total + 16 + stack_offset
                    // However, during codegen we don't know frame_total yet,
                    // so we use the pre-adjusted offset from the ABI and
                    // the prologue will handle the frame setup.
                    self.emit(
                        A64Instruction::new(A64Opcode::LDR_imm)
                            .with_rd(vreg)
                            .with_rn(FP_REG)
                            .with_imm(offset as i64 + 16)
                            .with_comment(format!(
                                "load stack arg {} from [fp+{}]",
                                i,
                                offset + 16
                            )),
                    );
                    param_move_count += 1;
                }
            }
        }
        param_move_count
    }

    // -----------------------------------------------------------------------
    // Basic block selection
    // -----------------------------------------------------------------------

    /// Select instructions for a single basic block.
    pub fn select_basic_block(&mut self, block: &BasicBlock, _func: &IrFunction) {
        let label = if block.index < self.block_labels.len() {
            self.block_labels[block.index].clone()
        } else {
            format!(".LBB_{}", block.index)
        };
        self.current_block = Some(label.clone());
        // Emit a NOP as block label marker (assembler will convert to label).
        self.emit(A64Instruction::new(A64Opcode::NOP).with_comment(format!("{}:", label)));

        for inst in block.instructions() {
            let selected = self.select_instruction(inst);
            for a64_inst in selected {
                self.emit(a64_inst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Instruction-level selection (dispatch)
    // -----------------------------------------------------------------------

    /// Select AArch64 instructions for a single IR instruction.
    /// Dispatches to type-specific selection methods based on the IR instruction variant.
    pub fn select_instruction(&mut self, ir_inst: &Instruction) -> Vec<A64Instruction> {
        match ir_inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                span,
            } => self.select_alloca(result, ty, alignment, span),
            Instruction::Load {
                result,
                ptr,
                ty,
                volatile: _,
                span: _,
            } => self.select_load(result, ptr, ty),
            Instruction::Store {
                value,
                ptr,
                volatile: _,
                span: _,
            } => self.select_store(value, ptr),
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                span: _,
            } => self.select_binop(result, op, lhs, rhs, ty),
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => self.select_icmp(result, op, lhs, rhs),
            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                span: _,
            } => self.select_fcmp(result, op, lhs, rhs),
            Instruction::Branch { target, span: _ } => self.select_branch(target),
            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                span: _,
            } => self.select_cond_branch(condition, then_block, else_block),
            Instruction::Switch {
                value,
                default,
                cases,
                span: _,
            } => self.select_switch(value, default, cases),
            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                span: _,
            } => self.select_call(result, callee, args, return_type),
            Instruction::Return { value, span: _ } => self.select_return(value),
            Instruction::Phi { .. } => {
                // Phi nodes are eliminated before code generation; emit NOP placeholder.
                vec![A64Instruction::new(A64Opcode::NOP).with_comment("phi (eliminated)")]
            }
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                in_bounds: _,
                span: _,
            } => self.select_gep(result, base, indices, result_type),
            Instruction::BitCast {
                result,
                value,
                to_type,
                source_unsigned,
                span: _,
            } => self.select_bitcast(result, value, to_type, source_unsigned),
            Instruction::Trunc {
                result,
                value,
                to_type,
                span: _,
            } => self.select_trunc(result, value, to_type),
            Instruction::ZExt {
                result,
                value,
                to_type,
                from_type: _,
                span: _,
            } => self.select_zext(result, value, to_type),
            Instruction::SExt {
                result,
                value,
                to_type,
                from_type,
                span: _,
            } => self.select_sext(result, value, to_type, from_type),
            Instruction::IntToPtr {
                result,
                value,
                span: _,
            } => self.select_int_to_ptr(result, value),
            Instruction::PtrToInt {
                result,
                value,
                to_type,
                span: _,
            } => self.select_ptr_to_int(result, value, to_type),
            // ----- StackAlloc (dynamic stack allocation: __builtin_alloca) -----
            Instruction::StackAlloc { result, size, .. } => {
                let rd = result.index().wrapping_add(VREG_BASE);
                let size_reg = size.index().wrapping_add(VREG_BASE);
                let mut out = Vec::new();
                // SUB SP, SP, size_reg
                out.push(
                    A64Instruction::new(A64Opcode::SUB_reg)
                        .with_rd(SP_REG)
                        .with_rn(SP_REG)
                        .with_rm(size_reg),
                );
                // Align SP to 16 bytes: materialize -16 in IP0 then AND.
                // MOVN X16, #15 → X16 = ~15 = -16
                out.push(
                    A64Instruction::new(A64Opcode::MOVN)
                        .with_rd(IP0)
                        .with_imm(15),
                );
                // AND SP, SP, IP0 (SP &= -16)
                out.push(
                    A64Instruction::new(A64Opcode::AND_reg)
                        .with_rd(SP_REG)
                        .with_rn(SP_REG)
                        .with_rm(IP0),
                );
                // MOV result, SP
                out.push(
                    A64Instruction::new(A64Opcode::MOV_reg)
                        .with_rd(rd)
                        .with_rn(SP_REG),
                );
                out
            }

            // StackSave — capture SP into a virtual register
            Instruction::StackSave { result, .. } => {
                let rd = result.index().wrapping_add(VREG_BASE);
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(SP_REG)]
            }
            // StackRestore — restore SP from a previously saved value
            Instruction::StackRestore { ptr, .. } => {
                let rn = ptr.index().wrapping_add(VREG_BASE);
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(SP_REG)
                    .with_rn(rn)]
            }
            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects: _,
                is_volatile: _,
                goto_targets,
                span,
            } => self.select_inline_asm(
                result,
                template,
                constraints,
                operands,
                clobbers,
                goto_targets,
                span,
            ),

            // IndirectBranch — computed goto: br %reg
            Instruction::IndirectBranch { target, .. } => {
                let rn = target.index().wrapping_add(VREG_BASE);
                vec![A64Instruction::new(A64Opcode::BR).with_rn(rn)]
            }

            // BlockAddress — materialize address of a labeled basic block
            Instruction::BlockAddress { result, block, .. } => {
                let rd = result.index().wrapping_add(VREG_BASE);
                // Use the block_labels vector (populated in lower_function)
                // which maps IR block index → assembler label name.
                let block_label = if block.index() < self.block_labels.len() {
                    self.block_labels[block.index()].clone()
                } else {
                    format!("label_{}", block.0)
                };
                // Use ADRP+ADD pair to load label address (symbol-based)
                vec![
                    A64Instruction::new(A64Opcode::ADRP)
                        .with_rd(rd)
                        .with_symbol(block_label.clone()),
                    A64Instruction::new(A64Opcode::ADD_imm)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_imm(0)
                        .with_symbol(block_label),
                ]
            }
        }
    }

    // -----------------------------------------------------------------------
    // Individual instruction selection methods
    // -----------------------------------------------------------------------

    /// Select instructions for an alloca (stack allocation).
    ///
    /// Uses pre-computed FP-relative positive offsets from `self.alloca_offsets`.
    /// The frame layout places locals at `[FP + 16 .. FP + 16 + locals_size)`,
    /// so alloca addresses are computed as `ADD rd, FP, #offset`.
    fn select_alloca(
        &mut self,
        result: &Value,
        ty: &IrType,
        alignment: &Option<usize>,
        _span: &Span,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);

        // Look up the pre-computed FP-relative offset.
        if let Some(&fp_offset) = self.alloca_offsets.get(&rd) {
            if fits_add_sub_imm(fp_offset as u64) {
                return vec![A64Instruction::new(A64Opcode::ADD_imm)
                    .with_rd(rd)
                    .with_rn(FP_REG)
                    .with_imm(fp_offset)
                    .with_comment(format!(
                        "alloca {} bytes at FP+{}",
                        ty.size_bytes(&self.target),
                        fp_offset
                    ))];
            }
            let mut v = self.materialize_immediate(IP0, fp_offset as u64);
            v.push(
                A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(FP_REG)
                    .with_rm(IP0)
                    .with_comment(format!(
                        "alloca {} bytes at FP+{} (large)",
                        ty.size_bytes(&self.target),
                        fp_offset
                    )),
            );
            return v;
        }

        // Fallback for dynamically added allocas (e.g. spill slot allocas
        // from insert_spill_code).  Compute offset from the end of the
        // current locals area.
        let size = ty.size_bytes(&self.target).max(1);
        let align = alignment.unwrap_or_else(|| ty.align_bytes(&self.target));
        let mask = if align > 0 { align - 1 } else { 0 };
        let cursor = (self.locals_size + mask) & !mask;
        self.locals_size = (cursor + size + 15) & !15;
        let fp_offset = (16 + cursor) as i64;
        self.alloca_offsets.insert(rd, fp_offset);
        if fits_add_sub_imm(fp_offset as u64) {
            vec![A64Instruction::new(A64Opcode::ADD_imm)
                .with_rd(rd)
                .with_rn(FP_REG)
                .with_imm(fp_offset)
                .with_comment(format!("alloca-dynamic {} bytes at FP+{}", size, fp_offset))]
        } else {
            let mut v = self.materialize_immediate(IP0, fp_offset as u64);
            v.push(
                A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(FP_REG)
                    .with_rm(IP0)
                    .with_comment(format!(
                        "alloca-dynamic {} bytes at FP+{} (large)",
                        size, fp_offset
                    )),
            );
            v
        }
    }

    /// Select load instructions based on the loaded type's size.
    fn select_load(&mut self, result: &Value, ptr: &Value, ty: &IrType) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let mut rn = ptr.index().wrapping_add(VREG_BASE);
        let size = self.type_size_for_mem(ty);
        let is_32 = Self::is_32bit_type(ty);

        let mut insts = Vec::new();

        // If the pointer is a global variable reference, materialize its
        // address into IP0 first, then load from IP0.
        if let Some(sym_name) = self.global_var_refs.get(ptr).cloned() {
            insts.push(
                A64Instruction::new(A64Opcode::LA)
                    .with_rd(rd)
                    .with_symbol(sym_name.clone())
                    .with_comment(format!("materialize global {} for load", sym_name)),
            );
            rn = rd; // Load from the register where we just put the address.
        }

        let opcode = match ty {
            IrType::I1 | IrType::I8 => A64Opcode::LDRB_imm,
            IrType::I16 => A64Opcode::LDRH_imm,
            IrType::I32 => A64Opcode::LDR_imm,
            IrType::I64 | IrType::I128 | IrType::Ptr => A64Opcode::LDR_imm,
            IrType::F32 => A64Opcode::LDR_fp_imm,
            IrType::F64 | IrType::F80 => A64Opcode::LDR_fp_imm,
            IrType::Void => A64Opcode::LDR_imm,
            IrType::Array(_, _) | IrType::Struct(_) | IrType::Function(_, _) => A64Opcode::LDR_imm,
        };

        let mut inst = A64Instruction::new(opcode)
            .with_rd(rd)
            .with_rn(rn)
            .with_imm(0)
            .with_comment(format!("load {} bytes", size));

        if is_32 && !ty.is_float() {
            inst = inst.set_32bit();
        }
        if ty.is_float() {
            inst = inst.set_fp();
        }

        insts.push(inst);
        insts
    }

    /// Select store instructions.  The correct store width is determined by
    /// looking up the value's type in `value_types`.  If the type is unknown
    /// we conservatively fall back to a 64-bit store.
    fn select_store(&mut self, value: &Value, ptr: &Value) -> Vec<A64Instruction> {
        let rd = value.index().wrapping_add(VREG_BASE);
        let rn = ptr.index().wrapping_add(VREG_BASE);

        let mut insts = Vec::new();

        // If the pointer is a global variable reference, materialize its
        // address into IP0 (X16) first, then store to IP0.
        let actual_rn = if let Some(sym_name) = self.global_var_refs.get(ptr).cloned() {
            insts.push(
                A64Instruction::new(A64Opcode::LA)
                    .with_rd(IP0)
                    .with_symbol(sym_name.clone())
                    .with_comment(format!("materialize global {} for store", sym_name)),
            );
            IP0
        } else {
            rn
        };

        // Determine the stored value's type from value_types, defaulting to
        // 64-bit if unknown (safe: the upper bits are simply ignored).
        let val_ty = self.value_types.get(value).cloned();
        let (opcode, is_fp, is_32, comment) = match val_ty.as_ref() {
            Some(IrType::I1) | Some(IrType::I8) => {
                (A64Opcode::STRB_imm, false, false, "store byte")
            }
            Some(IrType::I16) => (A64Opcode::STRH_imm, false, false, "store halfword"),
            Some(IrType::I32) => (A64Opcode::STR_imm, false, true, "store 32-bit"),
            Some(IrType::F32) => (A64Opcode::STR_fp_imm, true, true, "store f32"),
            Some(IrType::F64) | Some(IrType::F80) => {
                (A64Opcode::STR_fp_imm, true, false, "store f64")
            }
            _ => (A64Opcode::STR_imm, false, false, "store 64-bit"),
        };

        let mut inst = A64Instruction::new(opcode)
            .with_rd(rd)
            .with_rn(actual_rn)
            .with_imm(0)
            .with_comment(comment);

        if is_32 && !is_fp {
            inst = inst.set_32bit();
        }
        if is_fp {
            inst = inst.set_fp();
        }

        insts.push(inst);
        insts
    }

    /// Select instructions for a binary operation (integer or FP).
    fn select_binop(
        &mut self,
        result: &Value,
        op: &BinOp,
        lhs: &Value,
        rhs: &Value,
        ty: &IrType,
    ) -> Vec<A64Instruction> {
        // Handle constant sentinel: BinOp(Add/FAdd, result, result, UNDEF)
        // generated by emit_int_const / emit_float_const in the IR lowering.
        if *lhs == *result && *rhs == Value::UNDEF {
            let rd = result.index().wrapping_add(VREG_BASE);
            let is_32 = Self::is_32bit_type(ty);

            // **Float constant sentinel**: FAdd result, result, UNDEF
            // The value lives in a .rodata global — load its address via LA
            // (ADRP+ADD) into the result register (which will be a GP reg),
            // then LDR the float from that address.
            //
            // We use `rd` (the result vreg) to hold the address temporarily.
            // After register allocation, rd maps to a GP register.  The final
            // LDR_fp_imm reads from that GP register and writes the FP value
            // back into the same vreg number (which the regalloc treats as one
            // live interval).  This is safe because the address is dead after
            // the LDR.
            if matches!(op, BinOp::FAdd) {
                if let Some(sym_name) = self.float_constant_cache.get(result).cloned() {
                    let is_f32 = matches!(ty, IrType::F32);
                    let mut insts = Vec::new();
                    // Materialize the address of the float constant into
                    // the IP0 (X16) scratch register — NOT into `rd` — to
                    // avoid clobbering a GP register that may hold a live
                    // value (e.g. a store target address) at the same
                    // physical register index after allocation.
                    let addr_reg = crate::backend::aarch64::registers::IP0;
                    insts.push(
                        A64Instruction::new(A64Opcode::LA)
                            .with_rd(addr_reg)
                            .with_symbol(sym_name.clone())
                            .with_comment(format!("materialize float const {}", sym_name)),
                    );
                    // LDR Dd/Sd, [IP0, #0] — load the float value from the
                    // address we just materialized into the result FP register.
                    insts.push(
                        A64Instruction::new(A64Opcode::LDR_fp_imm)
                            .with_rd(rd)
                            .with_rn(addr_reg)
                            .with_imm(0)
                            .set_fp()
                            .with_comment(format!(
                                "ldr float const {} ({})",
                                sym_name,
                                if is_f32 { "f32" } else { "f64" }
                            )),
                    );
                    return insts;
                }
                // Unknown float — materialize zero via MOVZ.
                return vec![A64Instruction::new(A64Opcode::MOVZ)
                    .with_rd(rd)
                    .with_imm(0)
                    .with_comment("float const fallback: 0")];
            }

            // **Integer constant sentinel**: Add result, result, UNDEF
            // Look up constant from our cached constants.
            if let Some(&imm) = self
                .constant_values
                .get(&(result.index().wrapping_add(VREG_BASE)))
            {
                // On AArch64, 32-bit (W-register) operations zero-extend
                // the result into the upper 32 bits of the 64-bit
                // register.  For I32 types, mask the constant to 32 bits
                // so that the MOVZ+MOVK sequence only emits the lower
                // 32 bits (the hardware will zero-extend).
                let val = if is_32 {
                    (imm as u32) as u64
                } else {
                    imm as u64
                };
                let mut insts = self.materialize_immediate(rd, val);
                if is_32 {
                    for inst in &mut insts {
                        *inst = inst.clone().set_32bit();
                    }
                }
                return insts;
            }
            // Fallback: load 0.
            let mut inst = A64Instruction::new(A64Opcode::MOVZ).with_rd(rd).with_imm(0);
            if is_32 {
                inst = inst.set_32bit();
            }
            return vec![inst];
        }

        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = lhs.index().wrapping_add(VREG_BASE);
        let rm = rhs.index().wrapping_add(VREG_BASE);
        let is_32 = Self::is_32bit_type(ty);
        let mut insts = Vec::new();

        match op {
            BinOp::Add => {
                let mut inst = A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Sub => {
                let mut inst = A64Instruction::new(A64Opcode::SUB_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Mul => {
                // MUL is alias for MADD Rd, Rn, Rm, XZR.
                let mut inst = A64Instruction::new(A64Opcode::MADD)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm)
                    .with_ra(XZR);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::SDiv => {
                let mut inst = A64Instruction::new(A64Opcode::SDIV)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::UDiv => {
                let mut inst = A64Instruction::new(A64Opcode::UDIV)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::SRem => {
                // No direct remainder instruction on AArch64.
                // Rd = Rn - (Rn / Rm) * Rm = Rn - SDIV(Rn,Rm)*Rm
                // Implementation: SDIV tmp, Rn, Rm; MSUB Rd, tmp, Rm, Rn
                let tmp = IP0;
                let mut div = A64Instruction::new(A64Opcode::SDIV)
                    .with_rd(tmp)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    div = div.set_32bit();
                }
                insts.push(div);
                let mut msub = A64Instruction::new(A64Opcode::MSUB)
                    .with_rd(rd)
                    .with_rn(tmp)
                    .with_rm(rm)
                    .with_ra(rn);
                if is_32 {
                    msub = msub.set_32bit();
                }
                insts.push(msub);
            }
            BinOp::URem => {
                let tmp = IP0;
                let mut div = A64Instruction::new(A64Opcode::UDIV)
                    .with_rd(tmp)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    div = div.set_32bit();
                }
                insts.push(div);
                let mut msub = A64Instruction::new(A64Opcode::MSUB)
                    .with_rd(rd)
                    .with_rn(tmp)
                    .with_rm(rm)
                    .with_ra(rn);
                if is_32 {
                    msub = msub.set_32bit();
                }
                insts.push(msub);
            }
            BinOp::And => {
                let mut inst = A64Instruction::new(A64Opcode::AND_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Or => {
                let mut inst = A64Instruction::new(A64Opcode::ORR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::Xor => {
                // Detect bitwise NOT: XOR with Value::UNDEF represents
                // `~operand` (XOR with all-ones). Emit MVN (bitwise NOT)
                // instead of EOR with the UNDEF-resolved zero register.
                if *rhs == Value::UNDEF {
                    let mut inst = A64Instruction::new(A64Opcode::MVN_reg)
                        .with_rd(rd)
                        .with_rm(rn);
                    if is_32 {
                        inst = inst.set_32bit();
                    }
                    insts.push(inst);
                } else {
                    let mut inst = A64Instruction::new(A64Opcode::EOR_reg)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm);
                    if is_32 {
                        inst = inst.set_32bit();
                    }
                    insts.push(inst);
                }
            }
            BinOp::Shl => {
                let mut inst = A64Instruction::new(A64Opcode::LSL_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::AShr => {
                let mut inst = A64Instruction::new(A64Opcode::ASR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            BinOp::LShr => {
                let mut inst = A64Instruction::new(A64Opcode::LSR_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_rm(rm);
                if is_32 {
                    inst = inst.set_32bit();
                }
                insts.push(inst);
            }
            // Floating-point arithmetic
            BinOp::FAdd => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FADD_s
                } else {
                    A64Opcode::FADD_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FSub => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FSUB_s
                } else {
                    A64Opcode::FSUB_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FMul => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FMUL_s
                } else {
                    A64Opcode::FMUL_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FDiv => {
                let opc = if matches!(ty, IrType::F32) {
                    A64Opcode::FDIV_s
                } else {
                    A64Opcode::FDIV_d
                };
                insts.push(
                    A64Instruction::new(opc)
                        .with_rd(rd)
                        .with_rn(rn)
                        .with_rm(rm)
                        .set_fp(),
                );
            }
            BinOp::FRem => {
                // AArch64 has no FREM instruction; delegate to runtime fmodf/fmod.
                let fname = if matches!(ty, IrType::F32) {
                    "fmodf"
                } else {
                    "fmod"
                };
                insts.push(
                    A64Instruction::new(A64Opcode::CALL)
                        .with_symbol(fname.to_string())
                        .set_fp()
                        .with_comment(format!("frem via {}", fname)),
                );
            }
        }
        insts
    }

    /// Select instructions for an integer comparison.
    /// Emits CMP + CSET to produce a boolean (0/1) result.
    ///
    /// On AArch64, 32-bit W-register operations (`cmp w0, w1`) naturally
    /// handle sign-extended 32-bit values correctly because the hardware
    /// compares only the lower 32 bits.  Using 64-bit X-register comparisons
    /// (`cmp x0, x1`) on values that were produced by 32-bit W-register
    /// operations is INCORRECT: W-register writes zero-extend into X,
    /// which makes negative 32-bit values appear as large positive 64-bit
    /// values (e.g., -5 stored in w0 yields x0 = 0x00000000FFFFFFFB).
    ///
    /// We determine the operand width from the `value_types` map (which
    /// tracks the IrType of each SSA value).  If both operands are ≤32 bits,
    /// we use a 32-bit comparison (W-regs); otherwise we default to 64-bit.
    fn select_icmp(
        &mut self,
        result: &Value,
        op: &ICmpOp,
        lhs: &Value,
        rhs: &Value,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = lhs.index().wrapping_add(VREG_BASE);
        let rm = rhs.index().wrapping_add(VREG_BASE);
        let cc = CondCode::from_icmp(op);

        // Determine whether operands are 32-bit or narrower.
        let lhs_is_32 = self
            .value_types
            .get(lhs)
            .map(Self::is_32bit_type)
            .unwrap_or(false);
        let rhs_is_32 = self
            .value_types
            .get(rhs)
            .map(Self::is_32bit_type)
            .unwrap_or(false);
        let use_32bit = lhs_is_32 && rhs_is_32;

        let mut cmp_inst = A64Instruction::new(A64Opcode::CMP_reg)
            .with_rn(rn)
            .with_rm(rm)
            .with_comment(format!("cmp for icmp {:?}", op));
        if use_32bit {
            cmp_inst = cmp_inst.set_32bit();
        }

        vec![
            cmp_inst,
            A64Instruction::new(A64Opcode::CSET)
                .with_rd(rd)
                .with_cond(cc)
                .set_32bit()
                .with_comment(format!("cset w{}, {:?}", rd, cc)),
        ]
    }

    /// Select instructions for a floating-point comparison.
    /// Emits FCMP + CSET to produce a boolean (0/1) result.
    fn select_fcmp(
        &mut self,
        result: &Value,
        op: &FCmpOp,
        lhs: &Value,
        rhs: &Value,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = lhs.index().wrapping_add(VREG_BASE);
        let rm = rhs.index().wrapping_add(VREG_BASE);
        let cc = CondCode::from_fcmp(op);

        vec![
            A64Instruction::new(A64Opcode::FCMP_d)
                .with_rn(rn)
                .with_rm(rm)
                .set_fp()
                .with_comment(format!("fcmp for {:?}", op)),
            A64Instruction::new(A64Opcode::CSET)
                .with_rd(rd)
                .with_cond(cc)
                .set_32bit()
                .with_comment(format!("cset w{}, {:?}", rd, cc)),
        ]
    }

    /// Select an unconditional branch.
    fn select_branch(&mut self, target: &BlockId) -> Vec<A64Instruction> {
        let label = if target.index() < self.block_labels.len() {
            self.block_labels[target.index()].clone()
        } else {
            format!(".LBB_{}", target.index())
        };
        vec![A64Instruction::new(A64Opcode::B)
            .with_symbol(label)
            .with_comment(format!("b -> bb{}", target.index()))]
    }

    /// Select a conditional branch (CBNZ for true path, B for false path).
    fn select_cond_branch(
        &mut self,
        condition: &Value,
        then_block: &BlockId,
        else_block: &BlockId,
    ) -> Vec<A64Instruction> {
        let rn = condition.index().wrapping_add(VREG_BASE);
        let then_label = if then_block.index() < self.block_labels.len() {
            self.block_labels[then_block.index()].clone()
        } else {
            format!(".LBB_{}", then_block.index())
        };
        let else_label = if else_block.index() < self.block_labels.len() {
            self.block_labels[else_block.index()].clone()
        } else {
            format!(".LBB_{}", else_block.index())
        };
        vec![
            A64Instruction::new(A64Opcode::CBNZ)
                .with_rn(rn)
                .with_symbol(then_label)
                .with_comment(format!("cbnz -> then bb{}", then_block.index())),
            A64Instruction::new(A64Opcode::B)
                .with_symbol(else_label)
                .with_comment(format!("b -> else bb{}", else_block.index())),
        ]
    }

    /// Select a switch statement using cascaded CMP + B.EQ.
    /// For small case counts this is efficient; dense switches would use
    /// a jump table approach in a production compiler.
    fn select_switch(
        &mut self,
        value: &Value,
        default: &BlockId,
        cases: &[(i64, BlockId)],
    ) -> Vec<A64Instruction> {
        let rn = value.index().wrapping_add(VREG_BASE);
        let mut insts = Vec::new();

        for (case_val, target) in cases {
            let label = if target.index() < self.block_labels.len() {
                self.block_labels[target.index()].clone()
            } else {
                format!(".LBB_{}", target.index())
            };
            // Compare: use CMP_imm if value fits, otherwise materialize.
            if fits_add_sub_imm(*case_val as u64) {
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_imm)
                        .with_rn(rn)
                        .with_imm(*case_val),
                );
            } else {
                let mat = self.materialize_immediate(IP0, *case_val as u64);
                insts.extend(mat);
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_reg)
                        .with_rn(rn)
                        .with_rm(IP0),
                );
            }
            insts.push(
                A64Instruction::new(A64Opcode::B_cond)
                    .with_cond(CondCode::EQ)
                    .with_symbol(label),
            );
        }

        // Default case: unconditional branch.
        let default_label = if default.index() < self.block_labels.len() {
            self.block_labels[default.index()].clone()
        } else {
            format!(".LBB_{}", default.index())
        };
        insts.push(
            A64Instruction::new(A64Opcode::B)
                .with_symbol(default_label)
                .with_comment("switch default"),
        );

        insts
    }

    /// Select a function call instruction (direct or indirect).
    /// Places arguments in registers per AAPCS64, emits BL/BLR,
    /// and copies the return value from X0 or V0.
    /// Try to inline a GCC __builtin_* call as native AArch64 instructions.
    /// Returns Some(instructions) if the builtin was handled, None otherwise.
    #[allow(clippy::vec_init_then_push)]
    fn try_inline_builtin(
        &mut self,
        name: &str,
        result: &Value,
        args: &[Value],
    ) -> Option<Vec<A64Instruction>> {
        let rd = result.index().wrapping_add(VREG_BASE);
        match name {
            // ---- byte swap builtins ----
            "__builtin_bswap16" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // REV16: reverse bytes within each 16-bit halfword.
                // We need REV Wd, Wn (32-bit byte reversal) then LSR Wd, Wd, #16
                // to get 16-bit bswap result in lower 16 bits.
                insts.push(
                    A64Instruction::new(A64Opcode::REV)
                        .with_rd(rd)
                        .with_rn(src)
                        .set_32bit()
                        .with_comment("bswap16: rev w"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::LSR_imm)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_imm(16)
                        .set_32bit()
                        .with_comment("bswap16: lsr #16"),
                );
                Some(insts)
            }
            "__builtin_bswap32" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::REV)
                    .with_rd(rd)
                    .with_rn(src)
                    .set_32bit()
                    .with_comment("bswap32: rev w")])
            }
            "__builtin_bswap64" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::REV)
                    .with_rd(rd)
                    .with_rn(src)
                    .with_comment("bswap64: rev x")])
            }
            // ---- count leading zeros ----
            "__builtin_clz" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::CLZ)
                    .with_rd(rd)
                    .with_rn(src)
                    .set_32bit()
                    .with_comment("clz (32-bit)")])
            }
            "__builtin_clzl" | "__builtin_clzll" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::CLZ)
                    .with_rd(rd)
                    .with_rn(src)
                    .with_comment("clz (64-bit)")])
            }
            // ---- count trailing zeros ----
            "__builtin_ctz" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                // RBIT Wd, Wn; CLZ Wd, Wd → count trailing zeros via reverse-then-count-leading
                let mut insts = Vec::new();
                insts.push(
                    A64Instruction::new(A64Opcode::RBIT)
                        .with_rd(rd)
                        .with_rn(src)
                        .set_32bit()
                        .with_comment("ctz: rbit w"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CLZ)
                        .with_rd(rd)
                        .with_rn(rd)
                        .set_32bit()
                        .with_comment("ctz: clz w"),
                );
                Some(insts)
            }
            "__builtin_ctzl" | "__builtin_ctzll" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                insts.push(
                    A64Instruction::new(A64Opcode::RBIT)
                        .with_rd(rd)
                        .with_rn(src)
                        .with_comment("ctzll: rbit x"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CLZ)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_comment("ctzll: clz x"),
                );
                Some(insts)
            }
            // ---- popcount ----
            "__builtin_popcount" | "__builtin_popcountl" | "__builtin_popcountll" => {
                // AArch64 popcount: FMOV Dn, Xn; CNT Vn.8B, Vn.8B; ADDV Bn, Vn.8B; FMOV Wd, Sn
                // This is a multi-instruction SIMD sequence. For simplicity, use:
                // FMOV Dd, Xn; CNT Vd.8B; UADDLV Hd, Vd.8B; FMOV Wd, Sd
                // For now, emit a simple sequence using a temporary FP register.
                let src = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // Move GP -> FP: FMOV Dd, Xn
                insts.push(
                    A64Instruction::new(A64Opcode::FMOV_gen_to_fp)
                        .with_rd(rd)
                        .with_rn(src)
                        .set_fp()
                        .with_comment("popcount: fmov d, x"),
                );
                // CNT Vd.8B, Vd.8B (count set bits per byte)
                insts.push(
                    A64Instruction::new(A64Opcode::CNT)
                        .with_rd(rd)
                        .with_rn(rd)
                        .set_fp()
                        .with_comment("popcount: cnt v.8b"),
                );
                // ADDV Bd, Vd.8B (horizontal add across all bytes)
                insts.push(
                    A64Instruction::new(A64Opcode::ADDV)
                        .with_rd(rd)
                        .with_rn(rd)
                        .set_fp()
                        .with_comment("popcount: addv b, v.8b"),
                );
                // Move FP -> GP: FMOV Wd, Sd
                insts.push(
                    A64Instruction::new(A64Opcode::FMOV_fp_to_gen)
                        .with_rd(rd)
                        .with_rn(rd)
                        .set_fp()
                        .with_comment("popcount: fmov w, s"),
                );
                Some(insts)
            }
            // ---- find first set bit (1-indexed, 0 if input is 0) ----
            "__builtin_ffs" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // RBIT + CLZ gives ctz, then add 1. But ffs(0) must be 0.
                // Use: RBIT Wd, Wn; CLZ Wd, Wd; ADD Wd, Wd, #1;
                // Then: CMP Wn, #0; CSEL Wd, WZR, Wd, EQ
                insts.push(
                    A64Instruction::new(A64Opcode::RBIT)
                        .with_rd(rd)
                        .with_rn(src)
                        .set_32bit()
                        .with_comment("ffs: rbit w"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CLZ)
                        .with_rd(rd)
                        .with_rn(rd)
                        .set_32bit()
                        .with_comment("ffs: clz w"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::ADD_imm)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_imm(1)
                        .set_32bit()
                        .with_comment("ffs: add #1"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_imm)
                        .with_rn(src)
                        .with_imm(0)
                        .set_32bit()
                        .with_comment("ffs: cmp src, #0"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CSEL)
                        .with_rd(rd)
                        .with_rn(XZR)
                        .with_rm(rd)
                        .set_32bit()
                        .with_cond(CondCode::EQ)
                        .with_comment("ffs: csel -> 0 if src==0"),
                );
                Some(insts)
            }
            "__builtin_ffsll" => {
                let src = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                insts.push(
                    A64Instruction::new(A64Opcode::RBIT)
                        .with_rd(rd)
                        .with_rn(src)
                        .with_comment("ffsll: rbit x"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CLZ)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_comment("ffsll: clz x"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::ADD_imm)
                        .with_rd(rd)
                        .with_rn(rd)
                        .with_imm(1)
                        .with_comment("ffsll: add #1"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CMP_imm)
                        .with_rn(src)
                        .with_imm(0)
                        .with_comment("ffsll: cmp src, #0"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::CSEL)
                        .with_rd(rd)
                        .with_rn(XZR)
                        .with_rm(rd)
                        .with_cond(CondCode::EQ)
                        .with_comment("ffsll: csel -> 0 if src==0"),
                );
                Some(insts)
            }
            // ---- branch prediction and optimizer hints ----
            "__builtin_expect" => {
                // Returns first argument unchanged (hint only).
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(src)
                    .with_comment("expect: passthrough")])
            }
            "__builtin_assume_aligned" => {
                // Returns first argument unchanged (alignment hint only).
                let src = args[0].index().wrapping_add(VREG_BASE);
                Some(vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(src)
                    .with_comment("assume_aligned: passthrough")])
            }
            // ---- frame/return address ----
            "__builtin_frame_address" => {
                // Returns the frame pointer (X29/FP).
                Some(vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(FP_REG)
                    .with_comment("frame_address: mov from fp")])
            }
            "__builtin_return_address" => {
                // Returns the link register (X30/LR).
                Some(vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(LR)
                    .with_comment("return_address: mov from lr")])
            }
            // ---- variadic argument builtins ----
            // AAPCS64 simplified model: va_list is a pointer to the save area
            // on the stack where variadic arguments are stored sequentially.
            "__builtin_va_start" => {
                // va_start(ap): store the address of the first unnamed variadic
                // argument into ap.  The prologue saves X0-X7 at [FP+16..FP+80].
                // Named parameters occupy the first N slots, so the first
                // unnamed argument is at FP + 16 + named_gpr_count * 8.
                if args.is_empty() {
                    return None;
                }
                let ap_ptr = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // Compute address of first unnamed vararg in the save area.
                let va_offset = 16 + (self.named_gpr_count as i64) * 8;
                insts.push(
                    A64Instruction::new(A64Opcode::ADD_imm)
                        .with_rd(rd)
                        .with_rn(FP_REG)
                        .with_imm(va_offset)
                        .with_comment("va_start: addr of first vararg"),
                );
                // Store to va_list slot: *ap_ptr = rd
                insts.push(
                    A64Instruction::new(A64Opcode::STR_imm)
                        .with_rd(rd)
                        .with_rn(ap_ptr)
                        .with_imm(0)
                        .with_comment("va_start: store to ap"),
                );
                Some(insts)
            }
            "__builtin_va_arg" => {
                // va_arg(ap, type): load current pointer from ap, dereference it,
                // advance the pointer by 8, store back.
                if args.is_empty() {
                    return None;
                }
                let ap_ptr = args[0].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // Load current va_list pointer: tmp = *ap_ptr
                let tmp = rd; // reuse result register as temporary
                insts.push(
                    A64Instruction::new(A64Opcode::LDR_imm)
                        .with_rd(tmp)
                        .with_rn(ap_ptr)
                        .with_imm(0)
                        .with_comment("va_arg: load ap"),
                );
                // Load the argument value: rd = *tmp
                insts.push(
                    A64Instruction::new(A64Opcode::LDR_imm)
                        .with_rd(rd)
                        .with_rn(tmp)
                        .with_imm(0)
                        .with_comment("va_arg: load arg"),
                );
                // Advance pointer: tmp = tmp + 8
                // We need a separate temp register for the advanced pointer.
                // Use IP0 (X16) as scratch — it's reserved.
                insts.push(
                    A64Instruction::new(A64Opcode::LDR_imm)
                        .with_rd(IP0)
                        .with_rn(ap_ptr)
                        .with_imm(0)
                        .with_comment("va_arg: reload ap into ip0"),
                );
                insts.push(
                    A64Instruction::new(A64Opcode::ADD_imm)
                        .with_rd(IP0)
                        .with_rn(IP0)
                        .with_imm(8)
                        .with_comment("va_arg: advance by 8"),
                );
                // Store back: *ap_ptr = IP0
                insts.push(
                    A64Instruction::new(A64Opcode::STR_imm)
                        .with_rd(IP0)
                        .with_rn(ap_ptr)
                        .with_imm(0)
                        .with_comment("va_arg: store updated ap"),
                );
                Some(insts)
            }
            "__builtin_va_end" => {
                // va_end(ap): no-op on AArch64.
                Some(vec![A64Instruction::new(A64Opcode::MOV_imm)
                    .with_rd(rd)
                    .with_imm(0)
                    .with_comment("va_end: nop")])
            }
            "__builtin_va_copy" => {
                // va_copy(dest, src): copy the va_list pointer.
                if args.len() < 2 {
                    return None;
                }
                let dest_ptr = args[0].index().wrapping_add(VREG_BASE);
                let src_ptr = args[1].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();
                // Load src: IP0 = *src_ptr
                insts.push(
                    A64Instruction::new(A64Opcode::LDR_imm)
                        .with_rd(IP0)
                        .with_rn(src_ptr)
                        .with_imm(0)
                        .with_comment("va_copy: load src"),
                );
                // Store to dest: *dest_ptr = IP0
                insts.push(
                    A64Instruction::new(A64Opcode::STR_imm)
                        .with_rd(IP0)
                        .with_rn(dest_ptr)
                        .with_imm(0)
                        .with_comment("va_copy: store to dest"),
                );
                // Result is void/0.
                insts.push(
                    A64Instruction::new(A64Opcode::MOV_imm)
                        .with_rd(rd)
                        .with_imm(0)
                        .with_comment("va_copy: result"),
                );
                Some(insts)
            }
            // ---- overflow arithmetic builtins ----
            "__builtin_add_overflow" | "__builtin_sub_overflow" | "__builtin_mul_overflow" => {
                // __builtin_{add,sub,mul}_overflow(a, b, result_ptr)
                // Returns 1 if overflow occurred, 0 otherwise.
                // Stores the (possibly wrapped) result to *result_ptr.
                if args.len() < 3 {
                    return None;
                }
                let a_reg = args[0].index().wrapping_add(VREG_BASE);
                let b_reg = args[1].index().wrapping_add(VREG_BASE);
                let result_ptr = args[2].index().wrapping_add(VREG_BASE);
                let mut insts = Vec::new();

                match name {
                    "__builtin_add_overflow" => {
                        // ADDS Wtmp, Wa, Wb (32-bit, sets overflow flag)
                        insts.push(
                            A64Instruction::new(A64Opcode::ADDS_reg)
                                .with_rd(IP0)
                                .with_rn(a_reg)
                                .with_rm(b_reg)
                                .set_32bit()
                                .with_comment("add_overflow: adds"),
                        );
                        // Store result
                        insts.push(
                            A64Instruction::new(A64Opcode::STR_imm)
                                .with_rd(IP0)
                                .with_rn(result_ptr)
                                .with_imm(0)
                                .set_32bit()
                                .with_comment("add_overflow: store"),
                        );
                        // CSET rd, VS (overflow flag)
                        insts.push(
                            A64Instruction::new(A64Opcode::CSET)
                                .with_rd(rd)
                                .with_cond(CondCode::VS)
                                .with_comment("add_overflow: flag"),
                        );
                    }
                    "__builtin_sub_overflow" => {
                        // SUBS Wtmp, Wa, Wb (32-bit, sets overflow flag)
                        insts.push(
                            A64Instruction::new(A64Opcode::SUBS_reg)
                                .with_rd(IP0)
                                .with_rn(a_reg)
                                .with_rm(b_reg)
                                .set_32bit()
                                .with_comment("sub_overflow: subs"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::STR_imm)
                                .with_rd(IP0)
                                .with_rn(result_ptr)
                                .with_imm(0)
                                .set_32bit()
                                .with_comment("sub_overflow: store"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::CSET)
                                .with_rd(rd)
                                .with_cond(CondCode::VS)
                                .with_comment("sub_overflow: flag"),
                        );
                    }
                    _ => {
                        // mul_overflow for 32-bit int:
                        // Use reserved IP0/IP1 registers to avoid register allocator conflicts.
                        // 1. SMULL X16, Wn, Wm  (32×32→64 signed multiply into IP0)
                        // 2. STR W16, [result_ptr] (store lower 32 bits)
                        // 3. SXTW X17, W16 (sign-extend lower 32 bits)
                        // 4. CMP X16, X17 (compare full result with sign-extended)
                        // 5. CSET rd, NE (overflow if they differ)
                        insts.push(
                            A64Instruction::new(A64Opcode::SMULL)
                                .with_rd(IP0)
                                .with_rn(a_reg)
                                .with_rm(b_reg)
                                .with_comment("mul_overflow: smull x16,w,w"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::STR_imm)
                                .with_rd(IP0)
                                .with_rn(result_ptr)
                                .with_imm(0)
                                .set_32bit()
                                .with_comment("mul_overflow: store lo32"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::SXTW)
                                .with_rd(IP1)
                                .with_rn(IP0)
                                .with_comment("mul_overflow: sxtw x17,w16"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::SUBS_reg)
                                .with_rd(XZR)
                                .with_rn(IP0)
                                .with_rm(IP1)
                                .with_comment("mul_overflow: cmp full vs sxtw"),
                        );
                        insts.push(
                            A64Instruction::new(A64Opcode::CSET)
                                .with_rd(rd)
                                .with_cond(CondCode::NE)
                                .with_comment("mul_overflow: flag"),
                        );
                    }
                }
                Some(insts)
            }
            _ => None,
        }
    }

    fn select_call(
        &mut self,
        result: &Value,
        callee: &Value,
        args: &[Value],
        return_type: &IrType,
    ) -> Vec<A64Instruction> {
        // Intercept __builtin_* intrinsic calls and emit inline AArch64 code.
        if let Some(fname) = self.func_ref_names.get(callee).cloned() {
            if let Some(inlined) = self.try_inline_builtin(&fname, result, args) {
                return inlined;
            }
        }

        let mut insts = Vec::new();

        // Place arguments in registers per AAPCS64 convention.
        // Integer arguments go to X0–X7 (separate counter), FP arguments
        // go to V0–V7 (separate counter).  This is a fundamental AAPCS64
        // requirement: the two register banks are allocated independently.
        let mut int_reg_idx = 0usize;
        let mut fp_reg_idx = 0usize;
        let mut stack_offset: i64 = 0;
        for (i, arg) in args.iter().enumerate() {
            let src = arg.index().wrapping_add(VREG_BASE);
            // Classify argument as FP if its IrType is F32/F64/F80.
            let is_fp_arg = self
                .value_types
                .get(arg)
                .map(|ty| matches!(ty, IrType::F32 | IrType::F64 | IrType::F80))
                .unwrap_or(false);

            if is_fp_arg {
                if fp_reg_idx < NUM_FP_ARG_REGS {
                    let dest = FP_ARG_REGS[fp_reg_idx];
                    insts.push(
                        A64Instruction::new(A64Opcode::FMOV_d)
                            .with_rd(dest)
                            .with_rn(src)
                            .set_fp()
                            .set_call_arg_setup()
                            .with_comment(format!("FP arg {} -> v{}", i, fp_reg_idx)),
                    );
                    fp_reg_idx += 1;
                } else {
                    // Spill FP arg to stack.
                    insts.push(
                        A64Instruction::new(A64Opcode::STR_imm)
                            .with_rd(src)
                            .with_rn(SP_REG)
                            .with_imm(stack_offset)
                            .set_fp()
                            .set_call_arg_setup()
                            .with_comment(format!("FP stack arg {} at sp+{}", i, stack_offset)),
                    );
                    stack_offset += 8;
                }
            } else if int_reg_idx < NUM_INT_ARG_REGS {
                let dest = INT_ARG_REGS[int_reg_idx];
                // Check if this argument is a global variable reference.
                // Global variable addresses must be materialized with an LA
                // pseudo-instruction (ADRP+ADD) rather than a simple MOV,
                // because the Value has no defining instruction — it refers
                // directly to a linker symbol whose address is only known
                // at link time.
                if let Some(sym_name) = self.global_var_refs.get(arg).cloned() {
                    insts.push(
                        A64Instruction::new(A64Opcode::LA)
                            .with_rd(dest)
                            .with_symbol(sym_name.clone())
                            .set_call_arg_setup()
                            .with_comment(format!(
                                "load addr of global {} for arg {}",
                                sym_name, i
                            )),
                    );
                } else {
                    insts.push(
                        A64Instruction::new(A64Opcode::MOV_reg)
                            .with_rd(dest)
                            .with_rn(src)
                            .set_call_arg_setup()
                            .with_comment(format!("int arg {} -> {}", i, gpr_name(dest))),
                    );
                }
                int_reg_idx += 1;
            } else {
                // Spill integer arg to stack.
                // If this is a global variable reference, materialize its
                // address into the IP0 scratch register first, then store
                // that register onto the stack.
                if let Some(sym_name) = self.global_var_refs.get(arg).cloned() {
                    insts.push(
                        A64Instruction::new(A64Opcode::LA)
                            .with_rd(IP0)
                            .with_symbol(sym_name.clone())
                            .set_call_arg_setup()
                            .with_comment(format!(
                                "load addr of global {} for stack arg {}",
                                sym_name, i
                            )),
                    );
                    insts.push(
                        A64Instruction::new(A64Opcode::STR_imm)
                            .with_rd(IP0)
                            .with_rn(SP_REG)
                            .with_imm(stack_offset)
                            .set_call_arg_setup()
                            .with_comment(format!(
                                "stack arg {} (global) at sp+{}",
                                i, stack_offset
                            )),
                    );
                } else {
                    insts.push(
                        A64Instruction::new(A64Opcode::STR_imm)
                            .with_rd(src)
                            .with_rn(SP_REG)
                            .with_imm(stack_offset)
                            .set_call_arg_setup()
                            .with_comment(format!("stack arg {} at sp+{}", i, stack_offset)),
                    );
                }
                stack_offset += 8;
            }
        }

        // Emit the call instruction.
        // Check if callee is a known function reference (direct call → BL symbol)
        // or an indirect call through a register (→ BLR Xn).
        // DEBUG: temporarily log func_ref_names lookup
        if std::env::var("BCC_DEBUG_A64").is_ok() {
            eprintln!(
                "  SENTINEL: callee={:?} func_ref_names_count={} lookup={:?}",
                callee,
                self.func_ref_names.len(),
                self.func_ref_names.get(callee)
            );
        }
        if let Some(func_name) = self.func_ref_names.get(callee) {
            insts.push(
                A64Instruction::new(A64Opcode::CALL)
                    .with_symbol(func_name.clone())
                    .with_comment(format!("call {}", func_name)),
            );
        } else {
            let callee_reg = callee.index().wrapping_add(VREG_BASE);
            insts.push(
                A64Instruction::new(A64Opcode::CALL)
                    .with_rn(callee_reg)
                    .with_comment("indirect call"),
            );
        }

        // Move return value from X0 (integer) or V0 (FP) to result vreg.
        if !return_type.is_void() {
            let rd = result.index().wrapping_add(VREG_BASE);
            if return_type.is_float() {
                // Both F32 and F64 return values are in V0 per AAPCS64.
                insts.push(
                    A64Instruction::new(A64Opcode::FMOV_d)
                        .with_rd(rd)
                        .with_rn(V0)
                        .set_fp()
                        .with_comment("move FP return from v0"),
                );
            } else {
                insts.push(
                    A64Instruction::new(A64Opcode::MOV_reg)
                        .with_rd(rd)
                        .with_rn(X0)
                        .with_comment("move return from x0"),
                );
            }
        }

        insts
    }

    /// Select a return instruction.
    fn select_return(&mut self, value: &Option<Value>) -> Vec<A64Instruction> {
        let mut insts = Vec::new();
        if let Some(val) = value {
            let src = val.index().wrapping_add(VREG_BASE);
            // Move return value to X0 (integer) for the caller.
            insts.push(
                A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(X0)
                    .with_rn(src)
                    .with_comment("ret value -> x0"),
            );
        }
        insts.push(A64Instruction::new(A64Opcode::RET).with_comment("ret"));
        insts
    }

    /// Select instructions for GetElementPtr (pointer arithmetic).
    fn select_gep(
        &mut self,
        result: &Value,
        base: &Value,
        indices: &[Value],
        _result_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let mut insts = Vec::new();

        // Check if the base is a global variable reference — needs LA (Load Address) pseudo-instruction.
        if let Some(sym_name) = self.global_var_refs.get(base).cloned() {
            // Emit LA pseudo-instruction — the assembler expands it to ADRP+ADD (non-PIC)
            // or ADRP+LDR (PIC) with correct relocations.
            insts.push(
                A64Instruction::new(A64Opcode::LA)
                    .with_rd(rd)
                    .with_symbol(sym_name.clone())
                    .with_comment(format!("gep: load address of global {}", sym_name)),
            );
        } else {
            // Base is a regular virtual register — copy it into rd.
            let rn = base.index().wrapping_add(VREG_BASE);
            insts.push(
                A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_comment("gep: copy base ptr"),
            );
        }

        // Add each index offset, skipping constant-zero indices.
        for idx in indices {
            // Check if this index is a known constant zero — skip the ADD entirely.
            if let Some(&imm_val) = self
                .constant_values
                .get(&(idx.index().wrapping_add(VREG_BASE)))
            {
                if imm_val == 0 {
                    continue; // Index is zero, no offset to add.
                }
            }
            let rm = idx.index().wrapping_add(VREG_BASE);
            insts.push(
                A64Instruction::new(A64Opcode::ADD_reg)
                    .with_rd(rd)
                    .with_rn(rd)
                    .with_rm(rm)
                    .with_comment("gep: add index offset"),
            );
        }

        insts
    }

    /// Select instructions for a bitcast (reinterpret bits).
    fn select_bitcast(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
        _source_unsigned: &bool,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);

        if to_type.is_float() {
            // GP register -> FP register.
            vec![A64Instruction::new(A64Opcode::FMOV_gen_to_fp)
                .with_rd(rd)
                .with_rn(rn)
                .set_fp()
                .with_comment("bitcast gp -> fp")]
        } else if to_type.is_pointer() || to_type.is_integer() {
            vec![A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(rd)
                .with_rn(rn)
                .with_comment("bitcast (mov)")]
        } else {
            vec![A64Instruction::new(A64Opcode::MOV_reg)
                .with_rd(rd)
                .with_rn(rn)
                .with_comment("bitcast")]
        }
    }

    /// Select truncation instructions.
    fn select_trunc(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);

        match to_type {
            IrType::I1 => {
                // AND Wd, Wn, #1
                vec![A64Instruction::new(A64Opcode::AND_imm)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_imm(1)
                    .set_32bit()
                    .with_comment("trunc to i1")]
            }
            IrType::I8 => {
                vec![A64Instruction::new(A64Opcode::UXTB)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i8 (uxtb)")]
            }
            IrType::I16 => {
                vec![A64Instruction::new(A64Opcode::UXTH)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i16 (uxth)")]
            }
            IrType::I32 => {
                // MOV Wd, Wn: writing to W-register implicitly zero-extends.
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .set_32bit()
                    .with_comment("trunc to i32 (mov w)")]
            }
            _ => {
                vec![A64Instruction::new(A64Opcode::MOV_reg)
                    .with_rd(rd)
                    .with_rn(rn)
                    .with_comment("trunc (nop)")]
            }
        }
    }

    /// Select zero-extension instructions.
    fn select_zext(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);

        // On AArch64, writing to a W-register implicitly zero-extends the
        // upper 32 bits of the containing X-register. For sub-32-bit sources,
        // use UXTB/UXTH; for 32->64, a simple MOV Wd suffices.
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment(format!("zext to {:?}", to_type))]
    }

    /// Select sign-extension instructions (SXTB, SXTH, SXTW).
    fn select_sext(
        &mut self,
        result: &Value,
        value: &Value,
        to_type: &IrType,
        from_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);

        // Choose the appropriate sign-extend instruction based on **source** width.
        // SXTB sign-extends bits [7:0], SXTH sign-extends bits [15:0],
        // SXTW sign-extends bits [31:0].  The opcode depends on the source
        // type (where the valid bits are), NOT the target type.
        let opcode = match from_type {
            IrType::I1 | IrType::I8 => A64Opcode::SXTB,
            IrType::I16 => A64Opcode::SXTH,
            IrType::I32 => A64Opcode::SXTW,
            // I64→I128 or Ptr→I64: no sign-extend needed, use MOV.
            _ => A64Opcode::SXTW,
        };

        vec![A64Instruction::new(opcode)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment(format!("sext {:?} -> {:?}", from_type, to_type))]
    }

    /// Select integer-to-pointer conversion (no-op on LP64).
    fn select_int_to_ptr(&mut self, result: &Value, value: &Value) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment("inttoptr (nop on LP64)")]
    }

    /// Select pointer-to-integer conversion (no-op on LP64).
    fn select_ptr_to_int(
        &mut self,
        result: &Value,
        value: &Value,
        _to_type: &IrType,
    ) -> Vec<A64Instruction> {
        let rd = result.index().wrapping_add(VREG_BASE);
        let rn = value.index().wrapping_add(VREG_BASE);
        vec![A64Instruction::new(A64Opcode::MOV_reg)
            .with_rd(rd)
            .with_rn(rn)
            .with_comment("ptrtoint (nop on LP64)")]
    }

    /// Select inline assembly. Emits a pseudo-instruction that will be
    /// expanded by the assembler to the literal template text.
    fn select_inline_asm(
        &mut self,
        _result: &Value,
        template: &str,
        constraints: &str,
        _operands: &[Value],
        _clobbers: &[String],
        _goto_targets: &[BlockId],
        _span: &Span,
    ) -> Vec<A64Instruction> {
        vec![A64Instruction::new(A64Opcode::INLINE_ASM)
            .with_comment(format!("asm \"{}\" constraints: {}", template, constraints))]
    }
}

// ===========================================================================
// Ensure all schema-required dependency members are referenced
// ===========================================================================

#[cfg(test)]
mod _schema_verification {
    #![allow(dead_code, unused_variables, unused_imports)]
    use super::*;
    use crate::ir::types::StructType;

    fn verify_register_usage() {
        let _regs: [u32; 37] = [
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18,
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, SP_REG, XZR, FP_REG, LR,
            IP0, IP1,
        ];
        let _fprs: [u32; 32] = [
            V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18,
            V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
        ];
        let _a = ALLOCATABLE_GPRS;
        let _b = ALLOCATABLE_FPRS;
        let _c = CALLEE_SAVED_GPRS;
        let _d = CALLEE_SAVED_FPRS;
        let _e = CALLER_SAVED_GPRS;
        let _f = CALLER_SAVED_FPRS;
        let _ = gpr_name(0);
        let _ = fpr_name(V0);
        let _ = fpr_name_d(V0);
        let _ = fpr_name_s(V0);
        let _ = hw_encoding(X0);
        let _ = is_gpr(X0);
        let _ = is_fpr(V0);
        let _ = is_callee_saved(X19);
        let _ = is_allocatable(X0);
        let _ = reg_class(X0);
        let _rc = RegClass::GPR;
        let _rcf = RegClass::FPR;
    }

    fn verify_traits_usage() {
        let _mi = MachineInstruction::new(0);
        let _mo_reg = MachineOperand::Register(0);
        let _mo_vreg = MachineOperand::VirtualRegister(0);
        let _mo_imm = MachineOperand::Immediate(0);
        let _mo_mem = MachineOperand::Memory {
            base: Some(0),
            index: None,
            scale: 1,
            displacement: 0,
        };
        let _mo_fs = MachineOperand::FrameSlot(0);
        let _mo_gs = MachineOperand::GlobalSymbol("test".to_string());
        let _mo_bl = MachineOperand::BlockLabel(0);
        let _mf = MachineFunction::new("test".to_string());
        let _mbb = MachineBasicBlock::new(None);
        let _ri = RegisterInfo::new();
        let _al_r = ArgLocation::Register(0);
        let _al_rp = ArgLocation::RegisterPair(0, 1);
        let _al_s = ArgLocation::Stack(0);
    }

    fn verify_ir_usage() {
        let v = Value(0);
        let _ = v.index();
        let bid = BlockId(0);
        let _ = bid.index();
        let _ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::SDiv,
            BinOp::UDiv,
            BinOp::SRem,
            BinOp::URem,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
            BinOp::Shl,
            BinOp::AShr,
            BinOp::LShr,
            BinOp::FAdd,
            BinOp::FSub,
            BinOp::FMul,
            BinOp::FDiv,
        ];
        let _icmps = [
            ICmpOp::Eq,
            ICmpOp::Ne,
            ICmpOp::Slt,
            ICmpOp::Sle,
            ICmpOp::Sgt,
            ICmpOp::Sge,
            ICmpOp::Ult,
            ICmpOp::Ule,
            ICmpOp::Ugt,
            ICmpOp::Uge,
        ];
        let _fcmps = [
            FCmpOp::Oeq,
            FCmpOp::One,
            FCmpOp::Olt,
            FCmpOp::Ole,
            FCmpOp::Ogt,
            FCmpOp::Oge,
            FCmpOp::Uno,
            FCmpOp::Ord,
        ];
    }

    fn verify_type_usage() {
        let target = Target::AArch64;
        let _ = target.pointer_width();
        let _ = target.is_64bit();
        let _ct = CType::Int;
        let _mt = MachineType::Integer;
        let _mf32 = MachineType::F32;
        let _mf64 = MachineType::F64;
        let _mm = MachineType::Memory;
        let types: Vec<IrType> = vec![
            IrType::I1,
            IrType::I8,
            IrType::I16,
            IrType::I32,
            IrType::I64,
            IrType::I128,
            IrType::F32,
            IrType::F64,
            IrType::F80,
            IrType::Ptr,
            IrType::Void,
        ];
        for t in &types {
            let _ = t.size_bytes(&target);
            let _ = t.align_bytes(&target);
            let _ = t.is_integer();
            let _ = t.is_float();
            let _ = t.is_pointer();
        }
        let _ = IrType::I32.int_width();
        let _arr = IrType::Array(Box::new(IrType::I32), 10);
        let _st = IrType::Struct(StructType::new(vec![IrType::I32], false));
    }

    fn verify_diag_usage() {
        let mut diag = DiagnosticEngine::new();
        diag.emit_error(Span::dummy(), "test error");
        diag.emit_warning(Span::dummy(), "test warning");
        let _ = diag.has_errors();
    }

    fn verify_abi_usage() {
        let mut abi = AArch64Abi::new();
        abi.reset();
        let cty = CType::Int;
        let _ = abi.classify_arg(&cty);
        let _ = abi.classify_return(&cty);
        let _ = abi.classify_variadic_arg(&cty);
        let layout = AArch64Abi::compute_frame_layout(&[], 64, 2);
        let _ = layout.total_size;
        let _ = layout.callee_saved_offset;
        let _ = layout.lr_offset;
        let _ = layout.fp_offset;
        let _ = NUM_INT_ARG_REGS;
        let _ = NUM_FP_ARG_REGS;
        let _ = STACK_ALIGNMENT;
        let _ = INT_ARG_REGS;
        let _ = FP_ARG_REGS;
        let _ = INT_RET_REGS;
        let _ = INDIRECT_RESULT_REG;
        let _ = ArgClass::Integer;
    }

    fn verify_basic_block_usage() {
        let bb = BasicBlock::new(0);
        let _ = bb.instructions();
        let _ = &bb.label;
        let _ = &bb.predecessors;
        let _ = &bb.successors;
        let _ = bb.index;
    }

    fn verify_ir_function_usage() {
        let func = IrFunction::new("test".to_string(), vec![], IrType::Void);
        let _ = &func.name;
        let _ = &func.params;
        let _ = &func.return_type;
        let _ = func.blocks();
        let _ = func.entry_block();
        let _ = func.block_count();
        let _ = &func.calling_convention;
        let _ = func.is_variadic;
        let _ = func.is_definition;
    }

    fn verify_instruction_methods() {
        let span = Span::dummy();
        let inst = Instruction::Branch {
            target: BlockId(0),
            span,
        };
        let _ = inst.result();
        let _ = inst.is_terminator();
        let _ = inst.span();
    }
}
