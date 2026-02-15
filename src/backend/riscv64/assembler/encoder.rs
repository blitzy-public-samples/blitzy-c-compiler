//! RISC-V 64-bit instruction encoder for the BCC compiler backend.
//!
//! This module implements binary encoding for all six base RISC-V instruction
//! formats (R-type, I-type, S-type, B-type, U-type, J-type), the R4-type
//! format used by fused multiply-add instructions, and the RV64C compressed
//! instruction formats (CR, CI, CSS, CIW, CL, CS, CB, CJ).
//!
//! # Supported ISA Extensions
//!
//! The encoder covers the full **RV64IMAFDC** ISA:
//!
//! - **I** — Base integer instructions (arithmetic, logic, loads, stores, branches, jumps)
//! - **M** — Integer multiply/divide extension
//! - **A** — Atomic memory operation extension (LR/SC, AMO)
//! - **F** — Single-precision floating-point extension
//! - **D** — Double-precision floating-point extension
//! - **C** — Compressed (16-bit) instruction extension
//!
//! # Instruction Encoding Formats
//!
//! ```text
//! R-type:  [funct7 | rs2 | rs1 | funct3 | rd | opcode]
//! I-type:  [imm[11:0]       | rs1 | funct3 | rd | opcode]
//! S-type:  [imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode]
//! B-type:  [imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11] | opcode]
//! U-type:  [imm[31:12]                       | rd | opcode]
//! J-type:  [imm[20|10:1|11|19:12]            | rd | opcode]
//! R4-type: [rs3 | fmt | rs2 | rs1 | rm | rd | opcode]
//! ```
//!
//! All 32-bit instructions are serialized in little-endian byte order.
//! Compressed (16-bit) instructions are also little-endian.
//!
//! # Zero-Dependency Design
//!
//! This encoder is entirely self-contained with no external crate dependencies,
//! fulfilling the BCC project's zero-dependency mandate (Section 0.7.1).

use std::fmt;

use crate::backend::riscv64::registers::encoding;
use crate::backend::traits::PhysReg;

use super::relocations::RiscV64RelocationType;

// ============================================================================
// Base Opcode Constants (bits [6:0])
// ============================================================================

/// Register-register integer operations (ADD, SUB, SLL, SLT, etc.).
pub const OP: u8 = 0x33;
/// 32-bit word variants of register-register operations (ADDW, SUBW, etc.).
pub const OP_32: u8 = 0x3B;
/// Register-immediate integer operations (ADDI, SLTI, ANDI, etc.).
pub const OP_IMM: u8 = 0x13;
/// 32-bit word variants of register-immediate operations (ADDIW, SLLIW, etc.).
pub const OP_IMM_32: u8 = 0x1B;
/// Integer load instructions (LB, LH, LW, LD, LBU, LHU, LWU).
pub const LOAD: u8 = 0x03;
/// Integer store instructions (SB, SH, SW, SD).
pub const STORE: u8 = 0x23;
/// Conditional branch instructions (BEQ, BNE, BLT, BGE, BLTU, BGEU).
pub const BRANCH: u8 = 0x63;
/// Jump and link (JAL) — J-type unconditional jump.
pub const BASE_JAL: u8 = 0x6F;
/// Jump and link register (JALR) — I-type indirect jump.
pub const BASE_JALR: u8 = 0x67;
/// Load upper immediate (LUI) — U-type.
pub const BASE_LUI: u8 = 0x37;
/// Add upper immediate to PC (AUIPC) — U-type.
pub const BASE_AUIPC: u8 = 0x17;
/// System instructions (ECALL, EBREAK, CSR operations).
pub const SYSTEM: u8 = 0x73;
/// Memory fence instructions (FENCE, FENCE.I).
pub const MISC_MEM: u8 = 0x0F;
/// Atomic memory operations (LR, SC, AMO*).
pub const AMO: u8 = 0x2F;
/// Floating-point load instructions (FLW, FLD).
pub const LOAD_FP: u8 = 0x07;
/// Floating-point store instructions (FSW, FSD).
pub const STORE_FP: u8 = 0x27;
/// Floating-point computational instructions (FADD, FSUB, FMUL, etc.).
pub const OP_FP: u8 = 0x53;
/// Fused multiply-add: `rd = rs1*rs2 + rs3`.
pub const BASE_FMADD: u8 = 0x43;
/// Fused multiply-subtract: `rd = rs1*rs2 - rs3`.
pub const BASE_FMSUB: u8 = 0x47;
/// Negated fused multiply-subtract: `rd = -(rs1*rs2) + rs3`.
pub const BASE_FNMSUB: u8 = 0x4B;
/// Negated fused multiply-add: `rd = -(rs1*rs2) - rs3`.
pub const BASE_FNMADD: u8 = 0x4F;

// ============================================================================
// Funct3 Constants
// ============================================================================

// --- Integer arithmetic / logic (OP and OP_IMM) ---
/// ADDI / ADD / SUB funct3.
pub const FUNCT3_ADD: u8 = 0x0;
/// SLL / SLLI funct3.
pub const FUNCT3_SLL: u8 = 0x1;
/// SLT / SLTI funct3.
pub const FUNCT3_SLT: u8 = 0x2;
/// SLTU / SLTIU funct3.
pub const FUNCT3_SLTU: u8 = 0x3;
/// XOR / XORI funct3.
pub const FUNCT3_XOR: u8 = 0x4;
/// SRL / SRLI / SRA / SRAI funct3.
pub const FUNCT3_SRL: u8 = 0x5;
/// OR / ORI funct3.
pub const FUNCT3_OR: u8 = 0x6;
/// AND / ANDI funct3.
pub const FUNCT3_AND: u8 = 0x7;

// --- Branch funct3 ---
/// BEQ funct3.
pub const FUNCT3_BEQ: u8 = 0x0;
/// BNE funct3.
pub const FUNCT3_BNE: u8 = 0x1;
/// BLT funct3.
pub const FUNCT3_BLT: u8 = 0x4;
/// BGE funct3.
pub const FUNCT3_BGE: u8 = 0x5;
/// BLTU funct3.
pub const FUNCT3_BLTU: u8 = 0x6;
/// BGEU funct3.
pub const FUNCT3_BGEU: u8 = 0x7;

// --- Load funct3 ---
/// LB funct3.
pub const FUNCT3_LB: u8 = 0x0;
/// LH funct3.
pub const FUNCT3_LH: u8 = 0x1;
/// LW funct3.
pub const FUNCT3_LW: u8 = 0x2;
/// LD funct3.
pub const FUNCT3_LD: u8 = 0x3;
/// LBU funct3.
pub const FUNCT3_LBU: u8 = 0x4;
/// LHU funct3.
pub const FUNCT3_LHU: u8 = 0x5;
/// LWU funct3.
pub const FUNCT3_LWU: u8 = 0x6;

// --- Store funct3 ---
/// SB funct3.
pub const FUNCT3_SB: u8 = 0x0;
/// SH funct3.
pub const FUNCT3_SH: u8 = 0x1;
/// SW funct3.
pub const FUNCT3_SW: u8 = 0x2;
/// SD funct3.
pub const FUNCT3_SD: u8 = 0x3;

// --- FP load/store funct3 ---
/// FLW / FSW funct3 (32-bit FP).
pub const FUNCT3_W_FP: u8 = 0x2;
/// FLD / FSD funct3 (64-bit FP).
pub const FUNCT3_D_FP: u8 = 0x3;

// --- System funct3 ---
/// ECALL / EBREAK funct3.
pub const FUNCT3_PRIV: u8 = 0x0;
/// CSRRW funct3.
pub const FUNCT3_CSRRW: u8 = 0x1;
/// CSRRS funct3.
pub const FUNCT3_CSRRS: u8 = 0x2;
/// CSRRC funct3.
pub const FUNCT3_CSRRC: u8 = 0x3;

// --- FENCE funct3 ---
/// FENCE funct3.
pub const FUNCT3_FENCE: u8 = 0x0;

// --- Atomic funct3 ---
/// Word-width (32-bit) atomic operation funct3.
pub const FUNCT3_AMO_W: u8 = 0x2;
/// Doubleword-width (64-bit) atomic operation funct3.
pub const FUNCT3_AMO_D: u8 = 0x3;

// ============================================================================
// Funct7 Constants
// ============================================================================

/// ADD, SLL, SLT, SLTU, XOR, SRL, OR, AND — base integer R-type funct7.
pub const FUNCT7_BASE: u8 = 0x00;
/// SUB, SRA — alternate integer R-type funct7.
pub const FUNCT7_ALT: u8 = 0x20;
/// M extension multiply/divide instructions funct7.
pub const FUNCT7_MULDIV: u8 = 0x01;

// --- Atomic funct5 values (upper 5 bits of funct7 for AMO instructions) ---
/// LR (Load Reserved) funct5.
pub const FUNCT5_LR: u8 = 0x02;
/// SC (Store Conditional) funct5.
pub const FUNCT5_SC: u8 = 0x03;
/// AMOSWAP funct5.
pub const FUNCT5_AMOSWAP: u8 = 0x01;
/// AMOADD funct5.
pub const FUNCT5_AMOADD: u8 = 0x00;
/// AMOXOR funct5.
pub const FUNCT5_AMOXOR: u8 = 0x04;
/// AMOAND funct5.
pub const FUNCT5_AMOAND: u8 = 0x0C;
/// AMOOR funct5.
pub const FUNCT5_AMOOR: u8 = 0x08;
/// AMOMIN funct5.
pub const FUNCT5_AMOMIN: u8 = 0x10;
/// AMOMAX funct5.
pub const FUNCT5_AMOMAX: u8 = 0x14;
/// AMOMINU funct5.
pub const FUNCT5_AMOMINU: u8 = 0x18;
/// AMOMAXU funct5.
pub const FUNCT5_AMOMAXU: u8 = 0x1C;

// --- FP funct7 values ---
/// FADD.S funct7.
pub const FUNCT7_FADD_S: u8 = 0x00;
/// FSUB.S funct7.
pub const FUNCT7_FSUB_S: u8 = 0x04;
/// FMUL.S funct7.
pub const FUNCT7_FMUL_S: u8 = 0x08;
/// FDIV.S funct7.
pub const FUNCT7_FDIV_S: u8 = 0x0C;
/// FSQRT.S funct7.
pub const FUNCT7_FSQRT_S: u8 = 0x2C;
/// FSGNJ.S / FSGNJN.S / FSGNJX.S funct7.
pub const FUNCT7_FSGNJ_S: u8 = 0x10;
/// FMIN.S / FMAX.S funct7.
pub const FUNCT7_FMINMAX_S: u8 = 0x14;
/// FCVT.W.S / FCVT.WU.S funct7.
pub const FUNCT7_FCVT_INT_S: u8 = 0x60;
/// FMV.X.W / FCLASS.S funct7.
pub const FUNCT7_FMV_X_W: u8 = 0x70;
/// FEQ.S / FLT.S / FLE.S funct7.
pub const FUNCT7_FCMP_S: u8 = 0x50;
/// FCVT.S.W / FCVT.S.WU funct7.
pub const FUNCT7_FCVT_S_INT: u8 = 0x68;
/// FMV.W.X funct7.
pub const FUNCT7_FMV_W_X: u8 = 0x78;

/// FADD.D funct7.
pub const FUNCT7_FADD_D: u8 = 0x01;
/// FSUB.D funct7.
pub const FUNCT7_FSUB_D: u8 = 0x05;
/// FMUL.D funct7.
pub const FUNCT7_FMUL_D: u8 = 0x09;
/// FDIV.D funct7.
pub const FUNCT7_FDIV_D: u8 = 0x0D;
/// FSQRT.D funct7.
pub const FUNCT7_FSQRT_D: u8 = 0x2D;
/// FSGNJ.D / FSGNJN.D / FSGNJX.D funct7.
pub const FUNCT7_FSGNJ_D: u8 = 0x11;
/// FMIN.D / FMAX.D funct7.
pub const FUNCT7_FMINMAX_D: u8 = 0x15;
/// FCVT.S.D funct7.
pub const FUNCT7_FCVT_S_D: u8 = 0x20;
/// FCVT.D.S funct7.
pub const FUNCT7_FCVT_D_S: u8 = 0x21;
/// FEQ.D / FLT.D / FLE.D funct7.
pub const FUNCT7_FCMP_D: u8 = 0x51;
/// FCLASS.D / FMV.X.D funct7.
pub const FUNCT7_FCLASS_D: u8 = 0x71;
/// FCVT.W.D / FCVT.WU.D / FCVT.L.D / FCVT.LU.D funct7.
pub const FUNCT7_FCVT_INT_D: u8 = 0x61;
/// FCVT.D.W / FCVT.D.WU / FCVT.D.L / FCVT.D.LU funct7.
pub const FUNCT7_FCVT_D_INT: u8 = 0x69;
/// FMV.D.X funct7 (RV64 only).
pub const FUNCT7_FMV_D_X: u8 = 0x79;

/// Single-precision FP format field (bits [26:25] in R4-type).
pub const FMT_S: u8 = 0x00;
/// Double-precision FP format field (bits [26:25] in R4-type).
pub const FMT_D: u8 = 0x01;

/// Dynamic rounding mode (rm field = 0b111, use CSR frm).
pub const RM_DYN: u8 = 0x7;

// ============================================================================
// RvOpcode — comprehensive RISC-V instruction enumeration
// ============================================================================

/// Enumeration of all RISC-V instructions supported by the encoder,
/// covering the full RV64IMAFDC ISA.
///
/// Each variant maps to a specific machine instruction. The
/// [`RiscV64Encoder::encode_instruction`] method dispatches on this enum
/// to select the correct encoding format and field values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum RvOpcode {
    // --- RV64I base integer register-register (R-type) ---
    ADD,
    SUB,
    SLL,
    SLT,
    SLTU,
    XOR,
    SRL,
    SRA,
    OR,
    AND,
    // --- RV64I word variants (R-type, OP_32) ---
    ADDW,
    SUBW,
    SLLW,
    SRLW,
    SRAW,

    // --- RV64M multiply/divide (R-type) ---
    MUL,
    MULH,
    MULHSU,
    MULHU,
    DIV,
    DIVU,
    REM,
    REMU,
    MULW,
    DIVW,
    DIVUW,
    REMW,
    REMUW,

    // --- RV64I immediate operations (I-type) ---
    ADDI,
    SLTI,
    SLTIU,
    XORI,
    ORI,
    ANDI,
    ADDIW,
    // --- Shifts with immediate (I-type specialization) ---
    SLLI,
    SRLI,
    SRAI,
    SLLIW,
    SRLIW,
    SRAIW,

    // --- RV64I loads (I-type) ---
    LB,
    LH,
    LW,
    LD,
    LBU,
    LHU,
    LWU,

    // --- RV64I stores (S-type) ---
    SB,
    SH,
    SW,
    SD,

    // --- RV64I branches (B-type) ---
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,

    // --- RV64I upper immediate (U-type) ---
    LUI,
    AUIPC,

    // --- RV64I jumps ---
    JAL,  // J-type
    JALR, // I-type

    // --- RV64I system / fence ---
    ECALL,
    EBREAK,
    FENCE,
    CSRRW,
    CSRRS,
    CSRRC,

    // --- RV64F single-precision FP loads/stores ---
    FLW,
    FSW,
    // --- RV64D double-precision FP loads/stores ---
    FLD,
    FSD,

    // --- RV64F single-precision FP arithmetic (R-type, OP_FP) ---
    FADD_S,
    FSUB_S,
    FMUL_S,
    FDIV_S,
    FSQRT_S,
    FSGNJ_S,
    FSGNJN_S,
    FSGNJX_S,
    FMIN_S,
    FMAX_S,
    FCVT_W_S,
    FCVT_WU_S,
    FMV_X_W,
    FEQ_S,
    FLT_S,
    FLE_S,
    FCLASS_S,
    FCVT_S_W,
    FCVT_S_WU,
    FMV_W_X,
    FCVT_L_S,
    FCVT_LU_S,
    FCVT_S_L,
    FCVT_S_LU,

    // --- RV64D double-precision FP arithmetic (R-type, OP_FP) ---
    FADD_D,
    FSUB_D,
    FMUL_D,
    FDIV_D,
    FSQRT_D,
    FSGNJ_D,
    FSGNJN_D,
    FSGNJX_D,
    FMIN_D,
    FMAX_D,
    FCVT_S_D,
    FCVT_D_S,
    FEQ_D,
    FLT_D,
    FLE_D,
    FCLASS_D,
    FCVT_W_D,
    FCVT_WU_D,
    FCVT_D_W,
    FCVT_D_WU,
    FCVT_L_D,
    FCVT_LU_D,
    FCVT_D_L,
    FCVT_D_LU,
    FMV_X_D,
    FMV_D_X,

    // --- RV64F/D fused multiply-add (R4-type) ---
    FMADD_S,
    FMSUB_S,
    FNMSUB_S,
    FNMADD_S,
    FMADD_D,
    FMSUB_D,
    FNMSUB_D,
    FNMADD_D,

    // --- RV64A atomics (R-type, AMO opcode) ---
    LR_W,
    SC_W,
    AMOSWAP_W,
    AMOADD_W,
    AMOXOR_W,
    AMOAND_W,
    AMOOR_W,
    AMOMIN_W,
    AMOMAX_W,
    AMOMINU_W,
    AMOMAXU_W,
    LR_D,
    SC_D,
    AMOSWAP_D,
    AMOADD_D,
    AMOXOR_D,
    AMOAND_D,
    AMOOR_D,
    AMOMIN_D,
    AMOMAX_D,
    AMOMINU_D,
    AMOMAXU_D,

    // --- RV64C compressed instructions ---
    C_NOP,
    C_ADDI,
    C_ADDIW,
    C_LI,
    C_LUI,
    C_ADDI16SP,
    C_ADDI4SPN,
    C_SLLI,
    C_SRLI,
    C_SRAI,
    C_ANDI,
    C_MV,
    C_ADD,
    C_AND,
    C_OR,
    C_XOR,
    C_SUB,
    C_ADDW,
    C_SUBW,
    C_LW,
    C_LD,
    C_SW,
    C_SD,
    C_LWSP,
    C_LDSP,
    C_SWSP,
    C_SDSP,
    C_J,
    C_JAL,
    C_JR,
    C_JALR,
    C_BEQZ,
    C_BNEZ,
    C_EBREAK,
    C_FLD,
    C_FSD,
    C_FLDSP,
    C_FSDSP,

    // --- Pseudo-instructions (expand to real instruction sequences) ---
    /// NOP → ADDI x0, x0, 0
    NOP,
    /// LI rd, imm → LUI + ADDI or just ADDI depending on range
    LI,
    /// LA rd, symbol → AUIPC + ADDI pair
    LA,
    /// CALL symbol → AUIPC + JALR pair
    CALL,
    /// TAIL symbol → AUIPC + JALR pair (tail call)
    TAIL,
    /// RET → JALR x0, x1, 0
    RET,
    /// MV rd, rs → ADDI rd, rs, 0
    MV,
    /// NOT rd, rs → XORI rd, rs, -1
    NOT,
    /// NEG rd, rs → SUB rd, x0, rs
    NEG,
    /// SEQZ rd, rs → SLTIU rd, rs, 1
    SEQZ,
    /// SNEZ rd, rs → SLTU rd, x0, rs
    SNEZ,
    /// J offset → JAL x0, offset
    J,
    /// JR rs → JALR x0, rs, 0
    JR,
}

impl fmt::Display for RvOpcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::ADD => "add",
            Self::SUB => "sub",
            Self::SLL => "sll",
            Self::SLT => "slt",
            Self::SLTU => "sltu",
            Self::XOR => "xor",
            Self::SRL => "srl",
            Self::SRA => "sra",
            Self::OR => "or",
            Self::AND => "and",
            Self::ADDW => "addw",
            Self::SUBW => "subw",
            Self::SLLW => "sllw",
            Self::SRLW => "srlw",
            Self::SRAW => "sraw",
            Self::MUL => "mul",
            Self::MULH => "mulh",
            Self::MULHSU => "mulhsu",
            Self::MULHU => "mulhu",
            Self::DIV => "div",
            Self::DIVU => "divu",
            Self::REM => "rem",
            Self::REMU => "remu",
            Self::MULW => "mulw",
            Self::DIVW => "divw",
            Self::DIVUW => "divuw",
            Self::REMW => "remw",
            Self::REMUW => "remuw",
            Self::ADDI => "addi",
            Self::SLTI => "slti",
            Self::SLTIU => "sltiu",
            Self::XORI => "xori",
            Self::ORI => "ori",
            Self::ANDI => "andi",
            Self::ADDIW => "addiw",
            Self::SLLI => "slli",
            Self::SRLI => "srli",
            Self::SRAI => "srai",
            Self::SLLIW => "slliw",
            Self::SRLIW => "srliw",
            Self::SRAIW => "sraiw",
            Self::LB => "lb",
            Self::LH => "lh",
            Self::LW => "lw",
            Self::LD => "ld",
            Self::LBU => "lbu",
            Self::LHU => "lhu",
            Self::LWU => "lwu",
            Self::SB => "sb",
            Self::SH => "sh",
            Self::SW => "sw",
            Self::SD => "sd",
            Self::BEQ => "beq",
            Self::BNE => "bne",
            Self::BLT => "blt",
            Self::BGE => "bge",
            Self::BLTU => "bltu",
            Self::BGEU => "bgeu",
            Self::LUI => "lui",
            Self::AUIPC => "auipc",
            Self::JAL => "jal",
            Self::JALR => "jalr",
            Self::ECALL => "ecall",
            Self::EBREAK => "ebreak",
            Self::FENCE => "fence",
            Self::CSRRW => "csrrw",
            Self::CSRRS => "csrrs",
            Self::CSRRC => "csrrc",
            Self::FLW => "flw",
            Self::FSW => "fsw",
            Self::FLD => "fld",
            Self::FSD => "fsd",
            Self::FADD_S => "fadd.s",
            Self::FSUB_S => "fsub.s",
            Self::FMUL_S => "fmul.s",
            Self::FDIV_S => "fdiv.s",
            Self::FSQRT_S => "fsqrt.s",
            Self::FSGNJ_S => "fsgnj.s",
            Self::FSGNJN_S => "fsgnjn.s",
            Self::FSGNJX_S => "fsgnjx.s",
            Self::FMIN_S => "fmin.s",
            Self::FMAX_S => "fmax.s",
            Self::FCVT_W_S => "fcvt.w.s",
            Self::FCVT_WU_S => "fcvt.wu.s",
            Self::FMV_X_W => "fmv.x.w",
            Self::FEQ_S => "feq.s",
            Self::FLT_S => "flt.s",
            Self::FLE_S => "fle.s",
            Self::FCLASS_S => "fclass.s",
            Self::FCVT_S_W => "fcvt.s.w",
            Self::FCVT_S_WU => "fcvt.s.wu",
            Self::FMV_W_X => "fmv.w.x",
            Self::FCVT_L_S => "fcvt.l.s",
            Self::FCVT_LU_S => "fcvt.lu.s",
            Self::FCVT_S_L => "fcvt.s.l",
            Self::FCVT_S_LU => "fcvt.s.lu",
            Self::FADD_D => "fadd.d",
            Self::FSUB_D => "fsub.d",
            Self::FMUL_D => "fmul.d",
            Self::FDIV_D => "fdiv.d",
            Self::FSQRT_D => "fsqrt.d",
            Self::FSGNJ_D => "fsgnj.d",
            Self::FSGNJN_D => "fsgnjn.d",
            Self::FSGNJX_D => "fsgnjx.d",
            Self::FMIN_D => "fmin.d",
            Self::FMAX_D => "fmax.d",
            Self::FCVT_S_D => "fcvt.s.d",
            Self::FCVT_D_S => "fcvt.d.s",
            Self::FEQ_D => "feq.d",
            Self::FLT_D => "flt.d",
            Self::FLE_D => "fle.d",
            Self::FCLASS_D => "fclass.d",
            Self::FCVT_W_D => "fcvt.w.d",
            Self::FCVT_WU_D => "fcvt.wu.d",
            Self::FCVT_D_W => "fcvt.d.w",
            Self::FCVT_D_WU => "fcvt.d.wu",
            Self::FCVT_L_D => "fcvt.l.d",
            Self::FCVT_LU_D => "fcvt.lu.d",
            Self::FCVT_D_L => "fcvt.d.l",
            Self::FCVT_D_LU => "fcvt.d.lu",
            Self::FMV_X_D => "fmv.x.d",
            Self::FMV_D_X => "fmv.d.x",
            Self::FMADD_S => "fmadd.s",
            Self::FMSUB_S => "fmsub.s",
            Self::FNMSUB_S => "fnmsub.s",
            Self::FNMADD_S => "fnmadd.s",
            Self::FMADD_D => "fmadd.d",
            Self::FMSUB_D => "fmsub.d",
            Self::FNMSUB_D => "fnmsub.d",
            Self::FNMADD_D => "fnmadd.d",
            Self::LR_W => "lr.w",
            Self::SC_W => "sc.w",
            Self::AMOSWAP_W => "amoswap.w",
            Self::AMOADD_W => "amoadd.w",
            Self::AMOXOR_W => "amoxor.w",
            Self::AMOAND_W => "amoand.w",
            Self::AMOOR_W => "amoor.w",
            Self::AMOMIN_W => "amomin.w",
            Self::AMOMAX_W => "amomax.w",
            Self::AMOMINU_W => "amominu.w",
            Self::AMOMAXU_W => "amomaxu.w",
            Self::LR_D => "lr.d",
            Self::SC_D => "sc.d",
            Self::AMOSWAP_D => "amoswap.d",
            Self::AMOADD_D => "amoadd.d",
            Self::AMOXOR_D => "amoxor.d",
            Self::AMOAND_D => "amoand.d",
            Self::AMOOR_D => "amoor.d",
            Self::AMOMIN_D => "amomin.d",
            Self::AMOMAX_D => "amomax.d",
            Self::AMOMINU_D => "amominu.d",
            Self::AMOMAXU_D => "amomaxu.d",
            Self::C_NOP => "c.nop",
            Self::C_ADDI => "c.addi",
            Self::C_ADDIW => "c.addiw",
            Self::C_LI => "c.li",
            Self::C_LUI => "c.lui",
            Self::C_ADDI16SP => "c.addi16sp",
            Self::C_ADDI4SPN => "c.addi4spn",
            Self::C_SLLI => "c.slli",
            Self::C_SRLI => "c.srli",
            Self::C_SRAI => "c.srai",
            Self::C_ANDI => "c.andi",
            Self::C_MV => "c.mv",
            Self::C_ADD => "c.add",
            Self::C_AND => "c.and",
            Self::C_OR => "c.or",
            Self::C_XOR => "c.xor",
            Self::C_SUB => "c.sub",
            Self::C_ADDW => "c.addw",
            Self::C_SUBW => "c.subw",
            Self::C_LW => "c.lw",
            Self::C_LD => "c.ld",
            Self::C_SW => "c.sw",
            Self::C_SD => "c.sd",
            Self::C_LWSP => "c.lwsp",
            Self::C_LDSP => "c.ldsp",
            Self::C_SWSP => "c.swsp",
            Self::C_SDSP => "c.sdsp",
            Self::C_J => "c.j",
            Self::C_JAL => "c.jal",
            Self::C_JR => "c.jr",
            Self::C_JALR => "c.jalr",
            Self::C_BEQZ => "c.beqz",
            Self::C_BNEZ => "c.bnez",
            Self::C_EBREAK => "c.ebreak",
            Self::C_FLD => "c.fld",
            Self::C_FSD => "c.fsd",
            Self::C_FLDSP => "c.fldsp",
            Self::C_FSDSP => "c.fsdsp",
            Self::NOP => "nop",
            Self::LI => "li",
            Self::LA => "la",
            Self::CALL => "call",
            Self::TAIL => "tail",
            Self::RET => "ret",
            Self::MV => "mv",
            Self::NOT => "not",
            Self::NEG => "neg",
            Self::SEQZ => "seqz",
            Self::SNEZ => "snez",
            Self::J => "j",
            Self::JR => "jr",
        };
        write!(f, "{}", name)
    }
}

// ============================================================================
// EncoderOperand — operand types accepted by the encoder
// ============================================================================

/// Operand for the encoder's high-level instruction encoding interface.
///
/// The encoder accepts these operand types from the instruction selection
/// output and maps them to the appropriate bit-fields in the encoded
/// instruction word.
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderOperand {
    /// A 5-bit register encoding (0–31).
    Register(u8),
    /// An immediate integer value.
    Immediate(i64),
    /// A symbolic reference (function name, global variable).
    Symbol(String),
    /// A basic block label ID.
    Label(u32),
}

// ============================================================================
// EncoderError — encoding failure types
// ============================================================================

/// Error type for instruction encoding failures.
///
/// The encoder validates all inputs (immediate ranges, register numbers,
/// alignment constraints) and returns these errors for invalid encodings.
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderError {
    /// An immediate value exceeds the valid range for its bit-width field.
    ImmediateOutOfRange {
        /// The invalid immediate value.
        value: i64,
        /// The expected bit-width of the immediate field.
        bits: u8,
    },
    /// A register number is outside the valid 0–31 range.
    InvalidRegister(u8),
    /// The opcode is not recognized or not yet supported.
    UnsupportedOpcode(String),
    /// A register cannot be encoded in the 3-bit compressed register field
    /// (only x8–x15 are valid for compressed instructions).
    InvalidCompressedReg(u8),
    /// An offset or immediate does not meet the required alignment.
    AlignmentError {
        /// The misaligned value.
        value: i64,
        /// The required alignment in bytes.
        required: u32,
    },
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncoderError::ImmediateOutOfRange { value, bits } => {
                write!(
                    f,
                    "immediate value {} out of range for {}-bit field",
                    value, bits
                )
            }
            EncoderError::InvalidRegister(reg) => {
                write!(f, "invalid register number: {}", reg)
            }
            EncoderError::UnsupportedOpcode(name) => {
                write!(f, "unsupported opcode: {}", name)
            }
            EncoderError::InvalidCompressedReg(reg) => {
                write!(
                    f,
                    "register x{} cannot be encoded in compressed format (only x8-x15)",
                    reg
                )
            }
            EncoderError::AlignmentError { value, required } => {
                write!(f, "value {} is not aligned to {} bytes", value, required)
            }
        }
    }
}

// ============================================================================
// EncodedInstr — encoding result
// ============================================================================

/// The result of encoding a single RISC-V instruction.
///
/// Most instructions produce a single 32-bit word. Compressed instructions
/// produce a 16-bit halfword. Pseudo-instructions that expand to two
/// real instructions (e.g., CALL → AUIPC + JALR) produce a pair.
#[derive(Debug, Clone, PartialEq)]
pub enum EncodedInstr {
    /// A single 32-bit instruction word.
    Word(u32),
    /// A single 16-bit compressed instruction.
    Compressed(u16),
    /// A pair of 32-bit instruction words (for pseudo-instructions).
    Pair(u32, u32),
}

impl EncodedInstr {
    /// Serializes the encoded instruction to little-endian bytes.
    ///
    /// Returns 4 bytes for [`Word`](EncodedInstr::Word), 2 bytes for
    /// [`Compressed`](EncodedInstr::Compressed), or 8 bytes for
    /// [`Pair`](EncodedInstr::Pair).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            EncodedInstr::Word(w) => w.to_le_bytes().to_vec(),
            EncodedInstr::Compressed(c) => c.to_le_bytes().to_vec(),
            EncodedInstr::Pair(w1, w2) => {
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&w1.to_le_bytes());
                bytes.extend_from_slice(&w2.to_le_bytes());
                bytes
            }
        }
    }

    /// Returns the size of the encoded instruction in bytes.
    pub fn size(&self) -> usize {
        match self {
            EncodedInstr::Word(_) => 4,
            EncodedInstr::Compressed(_) => 2,
            EncodedInstr::Pair(_, _) => 8,
        }
    }

    /// Returns the associated relocation type for instructions that
    /// reference symbols, or `None` for fully-resolved instructions.
    ///
    /// This returns the *primary* relocation type for the opcode in a
    /// standard (non-PLT, non-GOT) context. Use
    /// [`lo12_relocation_type`](Self::lo12_relocation_type) for the
    /// lower-12-bit half of a relocation pair, and
    /// [`got_relocation_type`](Self::got_relocation_type) for GOT-relative
    /// addressing.
    pub fn relocation_type(&self, opcode: &RvOpcode) -> Option<RiscV64RelocationType> {
        match opcode {
            RvOpcode::JAL => Some(RiscV64RelocationType::R_RISCV_JAL),
            RvOpcode::BEQ
            | RvOpcode::BNE
            | RvOpcode::BLT
            | RvOpcode::BGE
            | RvOpcode::BLTU
            | RvOpcode::BGEU => Some(RiscV64RelocationType::R_RISCV_BRANCH),
            RvOpcode::AUIPC => Some(RiscV64RelocationType::R_RISCV_PCREL_HI20),
            RvOpcode::LUI => Some(RiscV64RelocationType::R_RISCV_HI20),
            RvOpcode::CALL => Some(RiscV64RelocationType::R_RISCV_CALL),
            // TAIL uses R_RISCV_CALL_PLT because it always goes through the PLT
            // to allow the linker to perform tail-call optimisation via relaxation.
            RvOpcode::TAIL => Some(RiscV64RelocationType::R_RISCV_CALL_PLT),
            RvOpcode::C_J | RvOpcode::C_JAL => Some(RiscV64RelocationType::R_RISCV_RVC_JUMP),
            RvOpcode::C_BEQZ | RvOpcode::C_BNEZ => Some(RiscV64RelocationType::R_RISCV_RVC_BRANCH),
            _ => None,
        }
    }

    /// Returns the relocation type for the lower-12-bit half of an
    /// address-materialisation pair (AUIPC+load/ADDI or LUI+load/ADDI).
    ///
    /// RISC-V splits wide addresses into a HI20 upper part (placed in
    /// AUIPC or LUI) and a LO12 lower part (placed in an I-type or S-type
    /// instruction). The relocation flavour for the LO12 half depends on
    /// *two* factors:
    ///
    /// 1. Whether the upper half was **PC-relative** (AUIPC → PCREL) or
    ///    **absolute** (LUI → ABS).
    /// 2. Whether the lower half is an **I-type** instruction (loads,
    ///    ADDI, JALR, etc.) or an **S-type** instruction (stores).
    ///
    /// # Parameters
    ///
    /// * `opcode` – the opcode of the LO12 instruction (I-type or S-type).
    /// * `pc_relative` – `true` when the HI20 partner is AUIPC,
    ///   `false` when it is LUI.
    pub fn lo12_relocation_type(
        opcode: &RvOpcode,
        pc_relative: bool,
    ) -> Option<RiscV64RelocationType> {
        match opcode {
            // I-type instructions that can carry a LO12 immediate:
            // loads, ADDI/ADDIW, JALR, FP loads.
            RvOpcode::ADDI
            | RvOpcode::ADDIW
            | RvOpcode::LB
            | RvOpcode::LH
            | RvOpcode::LW
            | RvOpcode::LD
            | RvOpcode::LBU
            | RvOpcode::LHU
            | RvOpcode::LWU
            | RvOpcode::JALR
            | RvOpcode::FLW
            | RvOpcode::FLD => {
                if pc_relative {
                    Some(RiscV64RelocationType::R_RISCV_PCREL_LO12_I)
                } else {
                    Some(RiscV64RelocationType::R_RISCV_LO12_I)
                }
            }
            // S-type instructions that can carry a LO12 immediate:
            // stores, FP stores.
            RvOpcode::SB
            | RvOpcode::SH
            | RvOpcode::SW
            | RvOpcode::SD
            | RvOpcode::FSW
            | RvOpcode::FSD => {
                if pc_relative {
                    Some(RiscV64RelocationType::R_RISCV_PCREL_LO12_S)
                } else {
                    Some(RiscV64RelocationType::R_RISCV_LO12_S)
                }
            }
            _ => None,
        }
    }

    /// Returns the GOT-relative relocation type for an instruction that
    /// loads a symbol's address through the Global Offset Table.
    ///
    /// In PIC code, an AUIPC instruction forms the upper 20 bits of the
    /// GOT entry's PC-relative offset. The associated relocation is
    /// [`R_RISCV_GOT_HI20`](RiscV64RelocationType::R_RISCV_GOT_HI20),
    /// which tells the linker to resolve the symbol through a GOT slot.
    pub fn got_relocation_type(opcode: &RvOpcode) -> Option<RiscV64RelocationType> {
        match opcode {
            RvOpcode::AUIPC => Some(RiscV64RelocationType::R_RISCV_GOT_HI20),
            _ => None,
        }
    }
}

// ============================================================================
// Immediate Validation Helpers
// ============================================================================

/// Validates that `imm` fits in a 12-bit signed immediate field (−2048..2047).
pub fn validate_imm12(imm: i32) -> Result<(), EncoderError> {
    if !(-2048..=2047).contains(&imm) {
        Err(EncoderError::ImmediateOutOfRange {
            value: imm as i64,
            bits: 12,
        })
    } else {
        Ok(())
    }
}

/// Validates that `imm` is a valid 20-bit upper immediate value.
/// The upper 20 bits of the instruction encode bits [31:12] of the target
/// address, so the effective range is 0..0xFFFFF (unsigned).
pub fn validate_imm20(imm: i32) -> Result<(), EncoderError> {
    // The upper immediate is treated as the 20-bit value placed in bits [31:12].
    // Accept any 32-bit value where only the upper 20 bits are significant.
    let u = imm as u32;
    if u > 0xFFFFF && !(-(1 << 19)..(1 << 19)).contains(&imm) {
        Err(EncoderError::ImmediateOutOfRange {
            value: imm as i64,
            bits: 20,
        })
    } else {
        Ok(())
    }
}

/// Validates a B-type branch offset: 13-bit signed, must be even (2-byte aligned).
/// Valid range: −4096..4094 (inclusive), and `offset & 1 == 0`.
pub fn validate_branch_offset(offset: i32) -> Result<(), EncoderError> {
    if (offset & 1) != 0 {
        return Err(EncoderError::AlignmentError {
            value: offset as i64,
            required: 2,
        });
    }
    if !(-4096..=4094).contains(&offset) {
        Err(EncoderError::ImmediateOutOfRange {
            value: offset as i64,
            bits: 13,
        })
    } else {
        Ok(())
    }
}

/// Validates a J-type JAL offset: 21-bit signed, must be even (2-byte aligned).
/// Valid range: −1048576..1048574 (inclusive), and `offset & 1 == 0`.
pub fn validate_jal_offset(offset: i32) -> Result<(), EncoderError> {
    if (offset & 1) != 0 {
        return Err(EncoderError::AlignmentError {
            value: offset as i64,
            required: 2,
        });
    }
    if !(-1_048_576..=1_048_574).contains(&offset) {
        Err(EncoderError::ImmediateOutOfRange {
            value: offset as i64,
            bits: 21,
        })
    } else {
        Ok(())
    }
}

// ============================================================================
// Register Encoding Helpers
// ============================================================================

/// Extracts the 5-bit hardware register encoding from a [`PhysReg`].
///
/// Uses [`crate::backend::riscv64::registers::encoding`] to map the PhysReg
/// numbering scheme (integer 0–31, FP 32–63) to the 5-bit field value
/// placed in rd/rs1/rs2/rs3 positions of RISC-V instructions.
#[inline]
pub fn encode_register(reg: PhysReg) -> u8 {
    encoding(reg)
}

/// Attempts to encode a register in the 3-bit compressed register field.
///
/// Only registers x8–x15 (s0/s1, a0–a5) can be encoded in the compressed
/// instruction format. Returns `Some(0..7)` for valid compressed registers,
/// or `None` if the register cannot be used in a compressed instruction.
#[inline]
pub fn encode_compressed_register(reg: PhysReg) -> Option<u8> {
    let hw = encoding(reg);
    // In compressed format, registers x8-x15 map to 3-bit encoding 0-7.
    // This applies only to integer registers (PhysReg 0-31).
    if reg.0 < 32 && (8..=15).contains(&hw) {
        Some(hw - 8)
    } else {
        None
    }
}

// ============================================================================
// RiscV64Encoder — stateless RISC-V 64-bit instruction encoder
// ============================================================================

/// Stateless encoder for RISC-V 64-bit instructions.
///
/// All encoding methods are pure functions: they take instruction parameters
/// and return the encoded bytes. No internal mutable state is maintained,
/// making the encoder safe to share across threads.
///
/// # Encoding Workflow
///
/// 1. The instruction selector produces an [`RvOpcode`] and a list of
///    [`EncoderOperand`] values.
/// 2. [`encode_instruction`](Self::encode_instruction) dispatches to the
///    appropriate format-specific encoder.
/// 3. The format-specific encoder assembles the bit-fields and returns an
///    [`EncodedInstr`].
/// 4. The caller serializes the result to bytes via
///    [`EncodedInstr::to_bytes`].
pub struct RiscV64Encoder;

impl RiscV64Encoder {
    // ========================================================================
    // Base Format Encoding Methods
    // ========================================================================

    /// Encodes an R-type instruction (register-register operations).
    ///
    /// Bit layout: `[31:25] funct7 | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode`
    ///
    /// Used for: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,
    /// ADDW, SUBW, SLLW, SRLW, SRAW, MUL/DIV family, and floating-point
    /// computational instructions.
    #[inline]
    pub fn encode_r_type(
        &self,
        opcode: u8,
        rd: u8,
        funct3: u8,
        rs1: u8,
        rs2: u8,
        funct7: u8,
    ) -> u32 {
        ((funct7 as u32) << 25)
            | ((rs2 as u32 & 0x1F) << 20)
            | ((rs1 as u32 & 0x1F) << 15)
            | ((funct3 as u32 & 0x7) << 12)
            | ((rd as u32 & 0x1F) << 7)
            | (opcode as u32 & 0x7F)
    }

    /// Encodes an I-type instruction (immediate operations, loads, JALR).
    ///
    /// Bit layout: `[31:20] imm[11:0] | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode`
    ///
    /// The 12-bit immediate is sign-extended. Valid range: −2048..2047.
    #[inline]
    pub fn encode_i_type(&self, opcode: u8, rd: u8, funct3: u8, rs1: u8, imm: i32) -> u32 {
        let imm_bits = (imm as u32) & 0xFFF;
        (imm_bits << 20)
            | ((rs1 as u32 & 0x1F) << 15)
            | ((funct3 as u32 & 0x7) << 12)
            | ((rd as u32 & 0x1F) << 7)
            | (opcode as u32 & 0x7F)
    }

    /// Encodes an S-type instruction (stores).
    ///
    /// Bit layout: `[31:25] imm[11:5] | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] imm[4:0] | [6:0] opcode`
    ///
    /// The 12-bit immediate is split: bits [11:5] in the upper field (bits 31:25)
    /// and bits [4:0] in the lower field (bits 11:7).
    #[inline]
    pub fn encode_s_type(&self, opcode: u8, funct3: u8, rs1: u8, rs2: u8, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let imm_11_5 = (imm_u >> 5) & 0x7F;
        let imm_4_0 = imm_u & 0x1F;
        (imm_11_5 << 25)
            | ((rs2 as u32 & 0x1F) << 20)
            | ((rs1 as u32 & 0x1F) << 15)
            | ((funct3 as u32 & 0x7) << 12)
            | (imm_4_0 << 7)
            | (opcode as u32 & 0x7F)
    }

    /// Encodes a B-type instruction (conditional branches).
    ///
    /// Bit layout:
    /// ```text
    /// [31]    imm[12]
    /// [30:25] imm[10:5]
    /// [24:20] rs2
    /// [19:15] rs1
    /// [14:12] funct3
    /// [11:8]  imm[4:1]
    /// [7]     imm[11]
    /// [6:0]   opcode
    /// ```
    ///
    /// The 13-bit signed immediate (PC-relative, always even) has its bits
    /// scattered across four non-contiguous fields.
    #[inline]
    pub fn encode_b_type(&self, opcode: u8, funct3: u8, rs1: u8, rs2: u8, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let bit_12 = (imm_u >> 12) & 0x1;
        let bits_10_5 = (imm_u >> 5) & 0x3F;
        let bits_4_1 = (imm_u >> 1) & 0xF;
        let bit_11 = (imm_u >> 11) & 0x1;
        (bit_12 << 31)
            | (bits_10_5 << 25)
            | ((rs2 as u32 & 0x1F) << 20)
            | ((rs1 as u32 & 0x1F) << 15)
            | ((funct3 as u32 & 0x7) << 12)
            | (bits_4_1 << 8)
            | (bit_11 << 7)
            | (opcode as u32 & 0x7F)
    }

    /// Encodes a U-type instruction (LUI, AUIPC).
    ///
    /// Bit layout: `[31:12] imm[31:12] | [11:7] rd | [6:0] opcode`
    ///
    /// The 20-bit immediate occupies the upper 20 bits of the 32-bit
    /// instruction word, forming bits [31:12] of the resulting value.
    #[inline]
    pub fn encode_u_type(&self, opcode: u8, rd: u8, imm: u32) -> u32 {
        ((imm & 0xFFFFF) << 12) | ((rd as u32 & 0x1F) << 7) | (opcode as u32 & 0x7F)
    }

    /// Encodes a J-type instruction (JAL).
    ///
    /// Bit layout:
    /// ```text
    /// [31]    imm[20]
    /// [30:21] imm[10:1]
    /// [20]    imm[11]
    /// [19:12] imm[19:12]
    /// [11:7]  rd
    /// [6:0]   opcode
    /// ```
    ///
    /// The 21-bit signed immediate (always even) has its bits heavily
    /// permuted across four fields.
    #[inline]
    pub fn encode_j_type(&self, opcode: u8, rd: u8, imm: i32) -> u32 {
        let imm_u = imm as u32;
        let bit_20 = (imm_u >> 20) & 0x1;
        let bits_10_1 = (imm_u >> 1) & 0x3FF;
        let bit_11 = (imm_u >> 11) & 0x1;
        let bits_19_12 = (imm_u >> 12) & 0xFF;
        (bit_20 << 31)
            | (bits_10_1 << 21)
            | (bit_11 << 20)
            | (bits_19_12 << 12)
            | ((rd as u32 & 0x1F) << 7)
            | (opcode as u32 & 0x7F)
    }

    /// Encodes an R4-type instruction (fused multiply-add).
    ///
    /// Bit layout: `[31:27] rs3 | [26:25] fmt | [24:20] rs2 | [19:15] rs1 | [14:12] rm | [11:7] rd | [6:0] opcode`
    ///
    /// Used for FMADD, FMSUB, FNMSUB, FNMADD in both .S and .D variants.
    /// The `fmt` field selects single (0b00) or double (0b01) precision.
    /// The `rm` field specifies the rounding mode.
    #[inline]
    pub fn encode_r4_type(
        &self,
        opcode: u8,
        rd: u8,
        rm: u8,
        rs1: u8,
        rs2: u8,
        rs3: u8,
        fmt: u8,
    ) -> u32 {
        ((rs3 as u32 & 0x1F) << 27)
            | ((fmt as u32 & 0x3) << 25)
            | ((rs2 as u32 & 0x1F) << 20)
            | ((rs1 as u32 & 0x1F) << 15)
            | ((rm as u32 & 0x7) << 12)
            | ((rd as u32 & 0x1F) << 7)
            | (opcode as u32 & 0x7F)
    }

    // ========================================================================
    // Compressed (RV64C) Format Encoding Methods — 16-bit instructions
    // ========================================================================

    /// Encodes a CR-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:12] funct4 | [11:7] rd/rs1 | [6:2] rs2 | [1:0] op`
    ///
    /// Used for C.MV, C.ADD, C.JR, C.JALR, C.EBREAK.
    #[inline]
    pub fn encode_cr(&self, funct4: u8, rd_rs1: u8, rs2: u8, op: u8) -> u16 {
        ((funct4 as u16 & 0xF) << 12)
            | ((rd_rs1 as u16 & 0x1F) << 7)
            | ((rs2 as u16 & 0x1F) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CI-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12] imm[5] | [11:7] rd/rs1 | [6:2] imm[4:0] | [1:0] op`
    ///
    /// Used for C.LI, C.LUI, C.ADDI, C.ADDIW, C.ADDI16SP, C.SLLI,
    /// C.LWSP, C.LDSP, C.FLDSP, C.NOP.
    #[inline]
    pub fn encode_ci(&self, funct3: u8, imm: i32, rd_rs1: u8, op: u8) -> u16 {
        let imm_u = imm as u32;
        let bit_5 = (imm_u >> 5) & 0x1;
        let bits_4_0 = imm_u & 0x1F;
        ((funct3 as u16 & 0x7) << 13)
            | ((bit_5 as u16) << 12)
            | ((rd_rs1 as u16 & 0x1F) << 7)
            | ((bits_4_0 as u16) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CSS-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:7] imm | [6:2] rs2 | [1:0] op`
    ///
    /// Used for C.SWSP, C.SDSP, C.FSDSP.
    #[inline]
    pub fn encode_css(&self, funct3: u8, rs2: u8, imm: u32, op: u8) -> u16 {
        let imm_field = (imm & 0x3F) as u16;
        ((funct3 as u16 & 0x7) << 13)
            | (imm_field << 7)
            | ((rs2 as u16 & 0x1F) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CIW-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:5] imm | [4:2] rd' | [1:0] op`
    ///
    /// Used for C.ADDI4SPN. The 3-bit `rd_prime` is the compressed register
    /// encoding (0–7 representing x8–x15).
    #[inline]
    pub fn encode_ciw(&self, funct3: u8, rd_prime: u8, imm: u32, op: u8) -> u16 {
        let imm_field = (imm & 0xFF) as u16;
        ((funct3 as u16 & 0x7) << 13)
            | (imm_field << 5)
            | ((rd_prime as u16 & 0x7) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CL-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:10] imm_hi | [9:7] rs1' | [6:5] imm_lo | [4:2] rd' | [1:0] op`
    ///
    /// Used for C.LW, C.LD, C.FLD. Register fields use compressed (3-bit)
    /// encoding for x8–x15.
    #[inline]
    pub fn encode_cl(&self, funct3: u8, rd_prime: u8, rs1_prime: u8, imm: u32, op: u8) -> u16 {
        let imm_hi = ((imm >> 2) & 0x7) as u16;
        let imm_lo = (imm & 0x3) as u16;
        ((funct3 as u16 & 0x7) << 13)
            | (imm_hi << 10)
            | ((rs1_prime as u16 & 0x7) << 7)
            | (imm_lo << 5)
            | ((rd_prime as u16 & 0x7) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CS-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:10] imm_hi | [9:7] rs1' | [6:5] imm_lo | [4:2] rs2' | [1:0] op`
    ///
    /// Used for C.SW, C.SD, C.FSD. Same layout as CL-format but the low
    /// register field is rs2' instead of rd'.
    #[inline]
    pub fn encode_cs(&self, funct3: u8, rs2_prime: u8, rs1_prime: u8, imm: u32, op: u8) -> u16 {
        let imm_hi = ((imm >> 2) & 0x7) as u16;
        let imm_lo = (imm & 0x3) as u16;
        ((funct3 as u16 & 0x7) << 13)
            | (imm_hi << 10)
            | ((rs1_prime as u16 & 0x7) << 7)
            | (imm_lo << 5)
            | ((rs2_prime as u16 & 0x7) << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CB-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:10] offset_hi | [9:7] rs1' | [6:2] offset_lo | [1:0] op`
    ///
    /// Used for C.BEQZ, C.BNEZ, C.SRLI, C.SRAI, C.ANDI.
    #[inline]
    pub fn encode_cb(&self, funct3: u8, rs1_prime: u8, offset: i32, op: u8) -> u16 {
        let off_u = offset as u32;
        let off_hi = ((off_u >> 5) & 0x7) as u16;
        let off_lo = (off_u & 0x1F) as u16;
        ((funct3 as u16 & 0x7) << 13)
            | (off_hi << 10)
            | ((rs1_prime as u16 & 0x7) << 7)
            | (off_lo << 2)
            | (op as u16 & 0x3)
    }

    /// Encodes a CJ-format compressed instruction.
    ///
    /// Bit layout (16-bit): `[15:13] funct3 | [12:2] jump_target[11-bit] | [1:0] op`
    ///
    /// Used for C.J (and C.JAL on RV32 only). The 11-bit signed offset
    /// is PC-relative and always even.
    #[inline]
    pub fn encode_cj(&self, funct3: u8, target: i32, op: u8) -> u16 {
        // CJ-format immediate bit rearrangement:
        // Instruction bits [12:2] encode target bits [11|4|9:8|10|6|7|3:1|5]
        let t = target as u32;
        let bit_11 = (t >> 11) & 0x1;
        let bit_4 = (t >> 4) & 0x1;
        let bits_9_8 = (t >> 8) & 0x3;
        let bit_10 = (t >> 10) & 0x1;
        let bit_6 = (t >> 6) & 0x1;
        let bit_7 = (t >> 7) & 0x1;
        let bits_3_1 = (t >> 1) & 0x7;
        let bit_5 = (t >> 5) & 0x1;
        let imm_field = (bit_11 << 10)
            | (bit_4 << 9)
            | (bits_9_8 << 7)
            | (bit_10 << 6)
            | (bit_6 << 5)
            | (bit_7 << 4)
            | (bits_3_1 << 1)
            | bit_5;
        ((funct3 as u16 & 0x7) << 13) | ((imm_field as u16) << 2) | (op as u16 & 0x3)
    }

    // ========================================================================
    // Operand extraction helpers
    // ========================================================================

    /// Extracts a register encoding (0–31) from operand at `idx`.
    fn get_reg(operands: &[EncoderOperand], idx: usize) -> Result<u8, EncoderError> {
        match operands.get(idx) {
            Some(EncoderOperand::Register(r)) => {
                if *r > 31 {
                    Err(EncoderError::InvalidRegister(*r))
                } else {
                    Ok(*r)
                }
            }
            _ => Err(EncoderError::InvalidRegister(255)),
        }
    }

    /// Extracts an immediate value from operand at `idx`.
    fn get_imm(operands: &[EncoderOperand], idx: usize) -> Result<i64, EncoderError> {
        match operands.get(idx) {
            Some(EncoderOperand::Immediate(v)) => Ok(*v),
            _ => Err(EncoderError::ImmediateOutOfRange { value: 0, bits: 0 }),
        }
    }

    // ========================================================================
    // High-level instruction encoding dispatch
    // ========================================================================

    /// Encodes a single RISC-V instruction from its opcode and operands.
    ///
    /// This is the primary entry point for the encoder. It dispatches to the
    /// appropriate format-specific encoder based on the opcode, validates
    /// operands and immediate ranges, and returns the encoded instruction.
    ///
    /// # Operand Conventions
    ///
    /// - **R-type**: `[Register(rd), Register(rs1), Register(rs2)]`
    /// - **I-type**: `[Register(rd), Register(rs1), Immediate(imm12)]`
    /// - **S-type**: `[Register(rs2), Register(rs1), Immediate(imm12)]`
    /// - **B-type**: `[Register(rs1), Register(rs2), Immediate(offset13)]`
    /// - **U-type**: `[Register(rd), Immediate(imm20)]`
    /// - **J-type**: `[Register(rd), Immediate(offset21)]`
    /// - **R4-type**: `[Register(rd), Register(rs1), Register(rs2), Register(rs3)]`
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError`] if operands are invalid, immediates are
    /// out of range, or the opcode is unsupported.
    pub fn encode_instruction(
        &self,
        opcode: RvOpcode,
        operands: &[EncoderOperand],
    ) -> Result<EncodedInstr, EncoderError> {
        match opcode {
            // ================================================================
            // RV64I base integer R-type instructions (OP = 0x33)
            // ================================================================
            RvOpcode::ADD => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_ADD,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SUB => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(
                    self.encode_r_type(OP, rd, FUNCT3_ADD, rs1, rs2, FUNCT7_ALT),
                ))
            }
            RvOpcode::SLL => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLL,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SLT => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLT,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SLTU => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLTU,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::XOR => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_XOR,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SRL => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SRA => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(
                    self.encode_r_type(OP, rd, FUNCT3_SRL, rs1, rs2, FUNCT7_ALT),
                ))
            }
            RvOpcode::OR => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_OR,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::AND => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_AND,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }

            // ================================================================
            // RV64I word-width R-type instructions (OP_32 = 0x3B)
            // ================================================================
            RvOpcode::ADDW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_ADD,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SUBW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32, rd, FUNCT3_ADD, rs1, rs2, FUNCT7_ALT,
                )))
            }
            RvOpcode::SLLW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_SLL,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SRLW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    rs2,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::SRAW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32, rd, FUNCT3_SRL, rs1, rs2, FUNCT7_ALT,
                )))
            }

            // ================================================================
            // RV64M multiply/divide R-type instructions
            // ================================================================
            RvOpcode::MUL => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_ADD,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::MULH => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLL,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::MULHSU => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLT,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::MULHU => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLTU,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::DIV => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_XOR,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::DIVU => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::REM => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_OR,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::REMU => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_AND,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            // RV64M word-width multiply/divide
            RvOpcode::MULW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_ADD,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::DIVW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_XOR,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::DIVUW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::REMW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_OR,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }
            RvOpcode::REMUW => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_32,
                    rd,
                    FUNCT3_AND,
                    rs1,
                    rs2,
                    FUNCT7_MULDIV,
                )))
            }

            // ================================================================
            // RV64I immediate instructions (I-type, OP_IMM = 0x13)
            // ================================================================
            RvOpcode::ADDI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_ADD, rs1, imm),
                ))
            }
            RvOpcode::SLTI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_SLT, rs1, imm),
                ))
            }
            RvOpcode::SLTIU => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM,
                    rd,
                    FUNCT3_SLTU,
                    rs1,
                    imm,
                )))
            }
            RvOpcode::XORI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_XOR, rs1, imm),
                ))
            }
            RvOpcode::ORI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_OR, rs1, imm),
                ))
            }
            RvOpcode::ANDI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_AND, rs1, imm),
                ))
            }
            RvOpcode::ADDIW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM_32, rd, FUNCT3_ADD, rs1, imm),
                ))
            }

            // ================================================================
            // RV64I shifts with immediate (I-type specialization)
            // ================================================================
            // SLLI: funct7=0x00 in upper 6 bits, shamt in lower 6 bits of imm
            RvOpcode::SLLI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                // RV64 SLLI: shamt is 6 bits (0-63)
                let shamt = (imm as u32) & 0x3F;
                let encoded_imm = shamt as i32; // upper bits are 0
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM,
                    rd,
                    FUNCT3_SLL,
                    rs1,
                    encoded_imm,
                )))
            }
            RvOpcode::SRLI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                let shamt = (imm as u32) & 0x3F;
                let encoded_imm = shamt as i32; // funct6=0x00
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    encoded_imm,
                )))
            }
            RvOpcode::SRAI => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                let shamt = (imm as u32) & 0x3F;
                let encoded_imm = (0x400 | shamt) as i32; // funct6=0x10 → bit 10 set
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    encoded_imm,
                )))
            }
            // RV64I word-width shift immediates (OP_IMM_32)
            RvOpcode::SLLIW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                let shamt = (imm as u32) & 0x1F; // 5-bit shamt for W variants
                let encoded_imm = shamt as i32;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM_32,
                    rd,
                    FUNCT3_SLL,
                    rs1,
                    encoded_imm,
                )))
            }
            RvOpcode::SRLIW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                let shamt = (imm as u32) & 0x1F;
                let encoded_imm = shamt as i32;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM_32,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    encoded_imm,
                )))
            }
            RvOpcode::SRAIW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                let shamt = (imm as u32) & 0x1F;
                let encoded_imm = (0x400 | shamt) as i32; // bit 10 set for arithmetic
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM_32,
                    rd,
                    FUNCT3_SRL,
                    rs1,
                    encoded_imm,
                )))
            }

            // ================================================================
            // RV64I loads (I-type, LOAD = 0x03)
            // ================================================================
            RvOpcode::LB => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LB, rs1, imm),
                ))
            }
            RvOpcode::LH => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LH, rs1, imm),
                ))
            }
            RvOpcode::LW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LW, rs1, imm),
                ))
            }
            RvOpcode::LD => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LD, rs1, imm),
                ))
            }
            RvOpcode::LBU => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LBU, rs1, imm),
                ))
            }
            RvOpcode::LHU => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LHU, rs1, imm),
                ))
            }
            RvOpcode::LWU => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(LOAD, rd, FUNCT3_LWU, rs1, imm),
                ))
            }

            // ================================================================
            // RV64I stores (S-type, STORE = 0x23)
            // ================================================================
            RvOpcode::SB => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_s_type(STORE, FUNCT3_SB, rs1, rs2, imm),
                ))
            }
            RvOpcode::SH => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_s_type(STORE, FUNCT3_SH, rs1, rs2, imm),
                ))
            }
            RvOpcode::SW => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_s_type(STORE, FUNCT3_SW, rs1, rs2, imm),
                ))
            }
            RvOpcode::SD => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_s_type(STORE, FUNCT3_SD, rs1, rs2, imm),
                ))
            }

            // ================================================================
            // RV64I branches (B-type, BRANCH = 0x63)
            // ================================================================
            RvOpcode::BEQ => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_b_type(BRANCH, FUNCT3_BEQ, rs1, rs2, imm),
                ))
            }
            RvOpcode::BNE => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_b_type(BRANCH, FUNCT3_BNE, rs1, rs2, imm),
                ))
            }
            RvOpcode::BLT => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_b_type(BRANCH, FUNCT3_BLT, rs1, rs2, imm),
                ))
            }
            RvOpcode::BGE => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_b_type(BRANCH, FUNCT3_BGE, rs1, rs2, imm),
                ))
            }
            RvOpcode::BLTU => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(self.encode_b_type(
                    BRANCH,
                    FUNCT3_BLTU,
                    rs1,
                    rs2,
                    imm,
                )))
            }
            RvOpcode::BGEU => {
                let (rs1, rs2, imm) = self.extract_b_operands(operands)?;
                validate_branch_offset(imm)?;
                Ok(EncodedInstr::Word(self.encode_b_type(
                    BRANCH,
                    FUNCT3_BGEU,
                    rs1,
                    rs2,
                    imm,
                )))
            }

            // ================================================================
            // RV64I upper immediate (U-type)
            // ================================================================
            RvOpcode::LUI => {
                let (rd, imm) = self.extract_u_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_u_type(BASE_LUI, rd, imm)))
            }
            RvOpcode::AUIPC => {
                let (rd, imm) = self.extract_u_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_u_type(BASE_AUIPC, rd, imm)))
            }

            // ================================================================
            // RV64I jumps
            // ================================================================
            RvOpcode::JAL => {
                let rd = Self::get_reg(operands, 0)?;
                let imm_val = Self::get_imm(operands, 1)? as i32;
                validate_jal_offset(imm_val)?;
                Ok(EncodedInstr::Word(
                    self.encode_j_type(BASE_JAL, rd, imm_val),
                ))
            }
            RvOpcode::JALR => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(BASE_JALR, rd, FUNCT3_ADD, rs1, imm),
                ))
            }

            // ================================================================
            // RV64I system / fence
            // ================================================================
            RvOpcode::ECALL => Ok(EncodedInstr::Word(self.encode_i_type(
                SYSTEM,
                0,
                FUNCT3_PRIV,
                0,
                0,
            ))),
            RvOpcode::EBREAK => Ok(EncodedInstr::Word(self.encode_i_type(
                SYSTEM,
                0,
                FUNCT3_PRIV,
                0,
                1,
            ))),
            RvOpcode::FENCE => {
                // Default FENCE: predecessor=iorw, successor=iorw (0x0FF)
                // Operands can optionally carry the fence bits as an immediate.
                let imm = if operands.is_empty() {
                    0x0FF // default: all predecessor and successor bits
                } else {
                    Self::get_imm(operands, 0).unwrap_or(0x0FF) as i32
                };
                Ok(EncodedInstr::Word(self.encode_i_type(
                    MISC_MEM,
                    0,
                    FUNCT3_FENCE,
                    0,
                    imm,
                )))
            }
            RvOpcode::CSRRW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    SYSTEM,
                    rd,
                    FUNCT3_CSRRW,
                    rs1,
                    imm,
                )))
            }
            RvOpcode::CSRRS => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    SYSTEM,
                    rd,
                    FUNCT3_CSRRS,
                    rs1,
                    imm,
                )))
            }
            RvOpcode::CSRRC => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    SYSTEM,
                    rd,
                    FUNCT3_CSRRC,
                    rs1,
                    imm,
                )))
            }

            // ================================================================
            // RV64F/D loads and stores
            // ================================================================
            RvOpcode::FLW => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    LOAD_FP,
                    rd,
                    FUNCT3_W_FP,
                    rs1,
                    imm,
                )))
            }
            RvOpcode::FLD => {
                let (rd, rs1, imm) = self.extract_i_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    LOAD_FP,
                    rd,
                    FUNCT3_D_FP,
                    rs1,
                    imm,
                )))
            }
            RvOpcode::FSW => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(self.encode_s_type(
                    STORE_FP,
                    FUNCT3_W_FP,
                    rs1,
                    rs2,
                    imm,
                )))
            }
            RvOpcode::FSD => {
                let (rs2, rs1, imm) = self.extract_s_operands(operands)?;
                validate_imm12(imm)?;
                Ok(EncodedInstr::Word(self.encode_s_type(
                    STORE_FP,
                    FUNCT3_D_FP,
                    rs1,
                    rs2,
                    imm,
                )))
            }

            // ================================================================
            // RV64F single-precision FP arithmetic
            // ================================================================
            RvOpcode::FADD_S => self.encode_fp_r(FUNCT7_FADD_S, operands),
            RvOpcode::FSUB_S => self.encode_fp_r(FUNCT7_FSUB_S, operands),
            RvOpcode::FMUL_S => self.encode_fp_r(FUNCT7_FMUL_S, operands),
            RvOpcode::FDIV_S => self.encode_fp_r(FUNCT7_FDIV_S, operands),
            RvOpcode::FSQRT_S => {
                // FSQRT only uses rs1, rs2 field is 0
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FSQRT_S,
                )))
            }
            RvOpcode::FSGNJ_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_S,
                )))
            }
            RvOpcode::FSGNJN_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_S,
                )))
            }
            RvOpcode::FSGNJX_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x2,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_S,
                )))
            }
            RvOpcode::FMIN_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FMINMAX_S,
                )))
            }
            RvOpcode::FMAX_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FMINMAX_S,
                )))
            }
            RvOpcode::FCVT_W_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FCVT_INT_S,
                )))
            }
            RvOpcode::FCVT_WU_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    1,
                    FUNCT7_FCVT_INT_S,
                )))
            }
            RvOpcode::FCVT_L_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    2,
                    FUNCT7_FCVT_INT_S,
                )))
            }
            RvOpcode::FCVT_LU_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    3,
                    FUNCT7_FCVT_INT_S,
                )))
            }
            RvOpcode::FMV_X_W => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    0,
                    FUNCT7_FMV_X_W,
                )))
            }
            RvOpcode::FCLASS_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    0,
                    FUNCT7_FMV_X_W,
                )))
            }
            RvOpcode::FEQ_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x2,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_S,
                )))
            }
            RvOpcode::FLT_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_S,
                )))
            }
            RvOpcode::FLE_S => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_S,
                )))
            }
            RvOpcode::FCVT_S_W => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FCVT_S_INT,
                )))
            }
            RvOpcode::FCVT_S_WU => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    1,
                    FUNCT7_FCVT_S_INT,
                )))
            }
            RvOpcode::FCVT_S_L => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    2,
                    FUNCT7_FCVT_S_INT,
                )))
            }
            RvOpcode::FCVT_S_LU => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    3,
                    FUNCT7_FCVT_S_INT,
                )))
            }
            RvOpcode::FMV_W_X => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    0,
                    FUNCT7_FMV_W_X,
                )))
            }

            // ================================================================
            // RV64D double-precision FP arithmetic
            // ================================================================
            RvOpcode::FADD_D => self.encode_fp_r(FUNCT7_FADD_D, operands),
            RvOpcode::FSUB_D => self.encode_fp_r(FUNCT7_FSUB_D, operands),
            RvOpcode::FMUL_D => self.encode_fp_r(FUNCT7_FMUL_D, operands),
            RvOpcode::FDIV_D => self.encode_fp_r(FUNCT7_FDIV_D, operands),
            RvOpcode::FSQRT_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FSQRT_D,
                )))
            }
            RvOpcode::FSGNJ_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_D,
                )))
            }
            RvOpcode::FSGNJN_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_D,
                )))
            }
            RvOpcode::FSGNJX_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x2,
                    rs1,
                    rs2,
                    FUNCT7_FSGNJ_D,
                )))
            }
            RvOpcode::FMIN_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FMINMAX_D,
                )))
            }
            RvOpcode::FMAX_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FMINMAX_D,
                )))
            }
            RvOpcode::FCVT_S_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    1,
                    FUNCT7_FCVT_S_D,
                )))
            }
            RvOpcode::FCVT_D_S => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FCVT_D_S,
                )))
            }
            RvOpcode::FEQ_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x2,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_D,
                )))
            }
            RvOpcode::FLT_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_D,
                )))
            }
            RvOpcode::FLE_D => {
                let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    rs2,
                    FUNCT7_FCMP_D,
                )))
            }
            RvOpcode::FCLASS_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x1,
                    rs1,
                    0,
                    FUNCT7_FCLASS_D,
                )))
            }
            RvOpcode::FCVT_W_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FCVT_INT_D,
                )))
            }
            RvOpcode::FCVT_WU_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    1,
                    FUNCT7_FCVT_INT_D,
                )))
            }
            RvOpcode::FCVT_L_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    2,
                    FUNCT7_FCVT_INT_D,
                )))
            }
            RvOpcode::FCVT_LU_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    3,
                    FUNCT7_FCVT_INT_D,
                )))
            }
            RvOpcode::FCVT_D_W => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    0,
                    FUNCT7_FCVT_D_INT,
                )))
            }
            RvOpcode::FCVT_D_WU => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    1,
                    FUNCT7_FCVT_D_INT,
                )))
            }
            RvOpcode::FCVT_D_L => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    2,
                    FUNCT7_FCVT_D_INT,
                )))
            }
            RvOpcode::FCVT_D_LU => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    RM_DYN,
                    rs1,
                    3,
                    FUNCT7_FCVT_D_INT,
                )))
            }
            RvOpcode::FMV_X_D => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    0,
                    FUNCT7_FCLASS_D,
                )))
            }
            RvOpcode::FMV_D_X => {
                let rd = Self::get_reg(operands, 0)?;
                let rs1 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP_FP,
                    rd,
                    0x0,
                    rs1,
                    0,
                    FUNCT7_FMV_D_X,
                )))
            }

            // ================================================================
            // RV64F/D fused multiply-add (R4-type)
            // ================================================================
            RvOpcode::FMADD_S => self.encode_fma(BASE_FMADD, FMT_S, operands),
            RvOpcode::FMSUB_S => self.encode_fma(BASE_FMSUB, FMT_S, operands),
            RvOpcode::FNMSUB_S => self.encode_fma(BASE_FNMSUB, FMT_S, operands),
            RvOpcode::FNMADD_S => self.encode_fma(BASE_FNMADD, FMT_S, operands),
            RvOpcode::FMADD_D => self.encode_fma(BASE_FMADD, FMT_D, operands),
            RvOpcode::FMSUB_D => self.encode_fma(BASE_FMSUB, FMT_D, operands),
            RvOpcode::FNMSUB_D => self.encode_fma(BASE_FNMSUB, FMT_D, operands),
            RvOpcode::FNMADD_D => self.encode_fma(BASE_FNMADD, FMT_D, operands),

            // ================================================================
            // RV64A atomics
            // ================================================================
            RvOpcode::LR_W => self.encode_amo(FUNCT5_LR, FUNCT3_AMO_W, operands, true),
            RvOpcode::SC_W => self.encode_amo(FUNCT5_SC, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOSWAP_W => self.encode_amo(FUNCT5_AMOSWAP, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOADD_W => self.encode_amo(FUNCT5_AMOADD, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOXOR_W => self.encode_amo(FUNCT5_AMOXOR, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOAND_W => self.encode_amo(FUNCT5_AMOAND, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOOR_W => self.encode_amo(FUNCT5_AMOOR, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOMIN_W => self.encode_amo(FUNCT5_AMOMIN, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOMAX_W => self.encode_amo(FUNCT5_AMOMAX, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOMINU_W => self.encode_amo(FUNCT5_AMOMINU, FUNCT3_AMO_W, operands, false),
            RvOpcode::AMOMAXU_W => self.encode_amo(FUNCT5_AMOMAXU, FUNCT3_AMO_W, operands, false),
            RvOpcode::LR_D => self.encode_amo(FUNCT5_LR, FUNCT3_AMO_D, operands, true),
            RvOpcode::SC_D => self.encode_amo(FUNCT5_SC, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOSWAP_D => self.encode_amo(FUNCT5_AMOSWAP, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOADD_D => self.encode_amo(FUNCT5_AMOADD, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOXOR_D => self.encode_amo(FUNCT5_AMOXOR, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOAND_D => self.encode_amo(FUNCT5_AMOAND, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOOR_D => self.encode_amo(FUNCT5_AMOOR, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOMIN_D => self.encode_amo(FUNCT5_AMOMIN, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOMAX_D => self.encode_amo(FUNCT5_AMOMAX, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOMINU_D => self.encode_amo(FUNCT5_AMOMINU, FUNCT3_AMO_D, operands, false),
            RvOpcode::AMOMAXU_D => self.encode_amo(FUNCT5_AMOMAXU, FUNCT3_AMO_D, operands, false),

            // ================================================================
            // RV64C compressed instructions
            // ================================================================
            RvOpcode::C_NOP => Ok(EncodedInstr::Compressed(self.encode_ci(0b000, 0, 0, 0b01))),
            RvOpcode::C_ADDI => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_ci(0b000, imm, rd, 0b01),
                ))
            }
            RvOpcode::C_ADDIW => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_ci(0b001, imm, rd, 0b01),
                ))
            }
            RvOpcode::C_LI => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_ci(0b010, imm, rd, 0b01),
                ))
            }
            RvOpcode::C_LUI => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_ci(0b011, imm, rd, 0b01),
                ))
            }
            RvOpcode::C_ADDI16SP => {
                // C.ADDI16SP uses rd=2 (sp), special immediate encoding
                let imm = Self::get_imm(operands, 0)? as i32;
                Ok(EncodedInstr::Compressed(self.encode_ci(
                    0b011,
                    imm >> 4,
                    2,
                    0b01,
                )))
            }
            RvOpcode::C_ADDI4SPN => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_ciw(
                    0b000,
                    rd_prime,
                    imm >> 2,
                    0b00,
                )))
            }
            RvOpcode::C_SLLI => {
                let rd = Self::get_reg(operands, 0)?;
                let shamt = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_ci(0b000, shamt, rd, 0b10),
                ))
            }
            RvOpcode::C_SRLI => {
                let rs1_prime = Self::get_reg(operands, 0)?;
                let shamt = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_cb(0b100, rs1_prime, shamt, 0b01),
                ))
            }
            RvOpcode::C_SRAI => {
                let rs1_prime = Self::get_reg(operands, 0)?;
                let shamt = Self::get_imm(operands, 1)? as i32;
                // SRAI has bit 10 set in the immediate encoding
                Ok(EncodedInstr::Compressed(self.encode_cb(
                    0b100,
                    rs1_prime,
                    shamt | 0x20,
                    0b01,
                )))
            }
            RvOpcode::C_ANDI => {
                let rs1_prime = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(self.encode_cb(
                    0b100,
                    rs1_prime,
                    imm | 0x40,
                    0b01,
                )))
            }
            RvOpcode::C_MV => {
                let rd = Self::get_reg(operands, 0)?;
                let rs2 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Compressed(
                    self.encode_cr(0b1000, rd, rs2, 0b10),
                ))
            }
            RvOpcode::C_ADD => {
                let rd = Self::get_reg(operands, 0)?;
                let rs2 = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Compressed(
                    self.encode_cr(0b1001, rd, rs2, 0b10),
                ))
            }
            RvOpcode::C_AND => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                // CA-format: funct6=100011, funct2=11
                let w = self.encode_compressed_alu(0b100011, rd_prime, rs2_prime, 0b11);
                Ok(EncodedInstr::Compressed(w))
            }
            RvOpcode::C_OR => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                let w = self.encode_compressed_alu(0b100011, rd_prime, rs2_prime, 0b10);
                Ok(EncodedInstr::Compressed(w))
            }
            RvOpcode::C_XOR => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                let w = self.encode_compressed_alu(0b100011, rd_prime, rs2_prime, 0b01);
                Ok(EncodedInstr::Compressed(w))
            }
            RvOpcode::C_SUB => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                let w = self.encode_compressed_alu(0b100011, rd_prime, rs2_prime, 0b00);
                Ok(EncodedInstr::Compressed(w))
            }
            RvOpcode::C_ADDW => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                let w = self.encode_compressed_alu(0b100111, rd_prime, rs2_prime, 0b01);
                Ok(EncodedInstr::Compressed(w))
            }
            RvOpcode::C_SUBW => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs2_prime = Self::get_reg(operands, 1)?;
                let w = self.encode_compressed_alu(0b100111, rd_prime, rs2_prime, 0b00);
                Ok(EncodedInstr::Compressed(w))
            }
            // Compressed loads
            RvOpcode::C_LW => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cl(
                    0b010,
                    rd_prime,
                    rs1_prime,
                    imm >> 2,
                    0b00,
                )))
            }
            RvOpcode::C_LD => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cl(
                    0b011,
                    rd_prime,
                    rs1_prime,
                    imm >> 3,
                    0b00,
                )))
            }
            RvOpcode::C_FLD => {
                let rd_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cl(
                    0b001,
                    rd_prime,
                    rs1_prime,
                    imm >> 3,
                    0b00,
                )))
            }
            // Compressed stores
            RvOpcode::C_SW => {
                let rs2_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cs(
                    0b110,
                    rs2_prime,
                    rs1_prime,
                    imm >> 2,
                    0b00,
                )))
            }
            RvOpcode::C_SD => {
                let rs2_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cs(
                    0b111,
                    rs2_prime,
                    rs1_prime,
                    imm >> 3,
                    0b00,
                )))
            }
            RvOpcode::C_FSD => {
                let rs2_prime = Self::get_reg(operands, 0)?;
                let rs1_prime = Self::get_reg(operands, 1)?;
                let imm = Self::get_imm(operands, 2)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_cs(
                    0b101,
                    rs2_prime,
                    rs1_prime,
                    imm >> 3,
                    0b00,
                )))
            }
            // Compressed stack-pointer relative loads/stores
            RvOpcode::C_LWSP => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(self.encode_ci(
                    0b010,
                    imm >> 2,
                    rd,
                    0b10,
                )))
            }
            RvOpcode::C_LDSP => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(self.encode_ci(
                    0b011,
                    imm >> 3,
                    rd,
                    0b10,
                )))
            }
            RvOpcode::C_FLDSP => {
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(self.encode_ci(
                    0b001,
                    imm >> 3,
                    rd,
                    0b10,
                )))
            }
            RvOpcode::C_SWSP => {
                let rs2 = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_css(
                    0b110,
                    rs2,
                    imm >> 2,
                    0b10,
                )))
            }
            RvOpcode::C_SDSP => {
                let rs2 = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_css(
                    0b111,
                    rs2,
                    imm >> 3,
                    0b10,
                )))
            }
            RvOpcode::C_FSDSP => {
                let rs2 = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)? as u32;
                Ok(EncodedInstr::Compressed(self.encode_css(
                    0b101,
                    rs2,
                    imm >> 3,
                    0b10,
                )))
            }
            // Compressed jumps
            RvOpcode::C_J => {
                let target = Self::get_imm(operands, 0)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_cj(0b101, target, 0b01),
                ))
            }
            RvOpcode::C_JAL => {
                let target = Self::get_imm(operands, 0)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_cj(0b001, target, 0b01),
                ))
            }
            RvOpcode::C_JR => {
                let rs1 = Self::get_reg(operands, 0)?;
                Ok(EncodedInstr::Compressed(
                    self.encode_cr(0b1000, rs1, 0, 0b10),
                ))
            }
            RvOpcode::C_JALR => {
                let rs1 = Self::get_reg(operands, 0)?;
                Ok(EncodedInstr::Compressed(
                    self.encode_cr(0b1001, rs1, 0, 0b10),
                ))
            }
            // Compressed branches
            RvOpcode::C_BEQZ => {
                let rs1_prime = Self::get_reg(operands, 0)?;
                let offset = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_cb(0b110, rs1_prime, offset, 0b01),
                ))
            }
            RvOpcode::C_BNEZ => {
                let rs1_prime = Self::get_reg(operands, 0)?;
                let offset = Self::get_imm(operands, 1)? as i32;
                Ok(EncodedInstr::Compressed(
                    self.encode_cb(0b111, rs1_prime, offset, 0b01),
                ))
            }
            RvOpcode::C_EBREAK => Ok(EncodedInstr::Compressed(self.encode_cr(0b1001, 0, 0, 0b10))),

            // ================================================================
            // Pseudo-instructions (expand to real instructions)
            // ================================================================
            RvOpcode::NOP => {
                // NOP → ADDI x0, x0, 0
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, 0, FUNCT3_ADD, 0, 0),
                ))
            }
            RvOpcode::MV => {
                // MV rd, rs → ADDI rd, rs, 0
                let rd = Self::get_reg(operands, 0)?;
                let rs = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_ADD, rs, 0),
                ))
            }
            RvOpcode::NOT => {
                // NOT rd, rs → XORI rd, rs, -1
                let rd = Self::get_reg(operands, 0)?;
                let rs = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(OP_IMM, rd, FUNCT3_XOR, rs, -1),
                ))
            }
            RvOpcode::NEG => {
                // NEG rd, rs → SUB rd, x0, rs
                let rd = Self::get_reg(operands, 0)?;
                let rs = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(
                    self.encode_r_type(OP, rd, FUNCT3_ADD, 0, rs, FUNCT7_ALT),
                ))
            }
            RvOpcode::SEQZ => {
                // SEQZ rd, rs → SLTIU rd, rs, 1
                let rd = Self::get_reg(operands, 0)?;
                let rs = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_i_type(
                    OP_IMM,
                    rd,
                    FUNCT3_SLTU,
                    rs,
                    1,
                )))
            }
            RvOpcode::SNEZ => {
                // SNEZ rd, rs → SLTU rd, x0, rs
                let rd = Self::get_reg(operands, 0)?;
                let rs = Self::get_reg(operands, 1)?;
                Ok(EncodedInstr::Word(self.encode_r_type(
                    OP,
                    rd,
                    FUNCT3_SLTU,
                    0,
                    rs,
                    FUNCT7_BASE,
                )))
            }
            RvOpcode::J => {
                // J offset → JAL x0, offset
                let offset = Self::get_imm(operands, 0)? as i32;
                validate_jal_offset(offset)?;
                Ok(EncodedInstr::Word(self.encode_j_type(BASE_JAL, 0, offset)))
            }
            RvOpcode::JR => {
                // JR rs → JALR x0, rs, 0
                let rs = Self::get_reg(operands, 0)?;
                Ok(EncodedInstr::Word(
                    self.encode_i_type(BASE_JALR, 0, FUNCT3_ADD, rs, 0),
                ))
            }
            RvOpcode::RET => {
                // RET → JALR x0, x1(ra), 0
                Ok(EncodedInstr::Word(
                    self.encode_i_type(BASE_JALR, 0, FUNCT3_ADD, 1, 0),
                ))
            }
            RvOpcode::LI => {
                // LI rd, imm — materialize an immediate into rd.
                // If imm fits in 12 bits: ADDI rd, x0, imm
                // Otherwise: LUI rd, upper20 + ADDI rd, rd, lower12
                let rd = Self::get_reg(operands, 0)?;
                let imm = Self::get_imm(operands, 1)?;
                self.encode_li(rd, imm)
            }
            RvOpcode::LA => {
                // LA rd, symbol → AUIPC rd, %pcrel_hi20(sym) + ADDI rd, rd, %pcrel_lo12(sym)
                // At encoding time with a concrete offset:
                let rd = Self::get_reg(operands, 0)?;
                let offset = Self::get_imm(operands, 1)?;
                self.encode_la(rd, offset as i32)
            }
            RvOpcode::CALL => {
                // CALL symbol → AUIPC x1(ra), upper20 + JALR x1, x1, lower12
                let offset = Self::get_imm(operands, 0)? as i32;
                self.encode_call(offset, 1) // rd = x1 (ra)
            }
            RvOpcode::TAIL => {
                // TAIL symbol → AUIPC x6(t1), upper20 + JALR x0, x6, lower12
                let offset = Self::get_imm(operands, 0)? as i32;
                self.encode_tail(offset)
            }
        }
    }

    // ========================================================================
    // Internal helper methods for common encoding patterns
    // ========================================================================

    /// Extracts (rd, rs1, rs2) for R-type instructions from operands.
    fn extract_r_operands(
        &self,
        operands: &[EncoderOperand],
    ) -> Result<(u8, u8, u8), EncoderError> {
        let rd = Self::get_reg(operands, 0)?;
        let rs1 = Self::get_reg(operands, 1)?;
        let rs2 = Self::get_reg(operands, 2)?;
        Ok((rd, rs1, rs2))
    }

    /// Extracts (rd, rs1, imm) for I-type instructions from operands.
    fn extract_i_operands(
        &self,
        operands: &[EncoderOperand],
    ) -> Result<(u8, u8, i32), EncoderError> {
        let rd = Self::get_reg(operands, 0)?;
        let rs1 = Self::get_reg(operands, 1)?;
        let imm = Self::get_imm(operands, 2)? as i32;
        Ok((rd, rs1, imm))
    }

    /// Extracts (rs2, rs1, imm) for S-type instructions from operands.
    fn extract_s_operands(
        &self,
        operands: &[EncoderOperand],
    ) -> Result<(u8, u8, i32), EncoderError> {
        let rs2 = Self::get_reg(operands, 0)?;
        let rs1 = Self::get_reg(operands, 1)?;
        let imm = Self::get_imm(operands, 2)? as i32;
        Ok((rs2, rs1, imm))
    }

    /// Extracts (rs1, rs2, imm) for B-type instructions from operands.
    fn extract_b_operands(
        &self,
        operands: &[EncoderOperand],
    ) -> Result<(u8, u8, i32), EncoderError> {
        let rs1 = Self::get_reg(operands, 0)?;
        let rs2 = Self::get_reg(operands, 1)?;
        let imm = Self::get_imm(operands, 2)? as i32;
        Ok((rs1, rs2, imm))
    }

    /// Extracts (rd, imm) for U-type instructions from operands.
    /// Returns the 20-bit upper immediate as a u32.
    fn extract_u_operands(&self, operands: &[EncoderOperand]) -> Result<(u8, u32), EncoderError> {
        let rd = Self::get_reg(operands, 0)?;
        let imm = Self::get_imm(operands, 1)? as u32;
        Ok((rd, imm))
    }

    /// Encodes a floating-point R-type instruction with dynamic rounding mode.
    /// Operands: [Register(rd), Register(rs1), Register(rs2)]
    fn encode_fp_r(
        &self,
        funct7: u8,
        operands: &[EncoderOperand],
    ) -> Result<EncodedInstr, EncoderError> {
        let (rd, rs1, rs2) = self.extract_r_operands(operands)?;
        Ok(EncodedInstr::Word(
            self.encode_r_type(OP_FP, rd, RM_DYN, rs1, rs2, funct7),
        ))
    }

    /// Encodes a fused multiply-add R4-type instruction.
    /// Operands: [Register(rd), Register(rs1), Register(rs2), Register(rs3)]
    fn encode_fma(
        &self,
        base_opcode: u8,
        fmt: u8,
        operands: &[EncoderOperand],
    ) -> Result<EncodedInstr, EncoderError> {
        let rd = Self::get_reg(operands, 0)?;
        let rs1 = Self::get_reg(operands, 1)?;
        let rs2 = Self::get_reg(operands, 2)?;
        let rs3 = Self::get_reg(operands, 3)?;
        Ok(EncodedInstr::Word(self.encode_r4_type(
            base_opcode,
            rd,
            RM_DYN,
            rs1,
            rs2,
            rs3,
            fmt,
        )))
    }

    /// Encodes an atomic memory operation (AMO).
    ///
    /// For LR (load-reserved), `is_lr` is true and operands are `[rd, rs1]`.
    /// For SC and other AMO ops, `is_lr` is false and operands are `[rd, rs2, rs1]`.
    ///
    /// Bit layout is R-type with funct7 = `[funct5 | aq | rl]`.
    fn encode_amo(
        &self,
        funct5: u8,
        funct3: u8,
        operands: &[EncoderOperand],
        is_lr: bool,
    ) -> Result<EncodedInstr, EncoderError> {
        let rd = Self::get_reg(operands, 0)?;
        let (rs1, rs2) = if is_lr {
            // LR: rd, (rs1) — rs2 must be 0
            let rs1 = Self::get_reg(operands, 1)?;
            (rs1, 0u8)
        } else {
            // SC / AMO*: rd, rs2, (rs1)
            let rs2 = Self::get_reg(operands, 1)?;
            let rs1 = Self::get_reg(operands, 2)?;
            (rs1, rs2)
        };
        // Default aq=0, rl=0 (can be extended via additional operand if needed)
        let funct7 = funct5 << 2; // aq=0, rl=0
        Ok(EncodedInstr::Word(
            self.encode_r_type(AMO, rd, funct3, rs1, rs2, funct7),
        ))
    }

    /// Encodes a CA-format compressed ALU instruction.
    ///
    /// Bit layout: `[15:10] funct6 | [9:7] rd'/rs1' | [6:5] funct2 | [4:2] rs2' | [1:0] op=01`
    fn encode_compressed_alu(&self, funct6: u8, rd_prime: u8, rs2_prime: u8, funct2: u8) -> u16 {
        ((funct6 as u16 & 0x3F) << 10)
            | ((rd_prime as u16 & 0x7) << 7)
            | ((funct2 as u16 & 0x3) << 5)
            | ((rs2_prime as u16 & 0x7) << 2)
            | 0b01 // op = 01
    }

    /// Encodes a LI pseudo-instruction.
    ///
    /// If the immediate fits in 12 bits, produces a single ADDI.
    /// Otherwise produces LUI + ADDI pair.
    fn encode_li(&self, rd: u8, imm: i64) -> Result<EncodedInstr, EncoderError> {
        let imm32 = imm as i32;
        // Check if fits in 12-bit signed immediate
        if (-2048..=2047).contains(&imm32) {
            Ok(EncodedInstr::Word(
                self.encode_i_type(OP_IMM, rd, FUNCT3_ADD, 0, imm32),
            ))
        } else {
            // Split into upper 20 bits (LUI) and lower 12 bits (ADDI).
            // The lower 12 bits are sign-extended, so if bit 11 is set we need
            // to add 1 to the upper part to compensate.
            let lower = ((imm32 as u32) & 0xFFF) as i32;
            let upper = if lower >= 0x800 {
                // Sign extension of lower 12 bits means the upper portion needs +1
                (((imm32 as u32).wrapping_add(0x1000)) >> 12) & 0xFFFFF
            } else {
                ((imm32 as u32) >> 12) & 0xFFFFF
            };
            let lower_signed = if lower >= 0x800 {
                lower - 0x1000 // sign-extend
            } else {
                lower
            };
            let lui = self.encode_u_type(BASE_LUI, rd, upper);
            let addi = self.encode_i_type(OP_IMM, rd, FUNCT3_ADD, rd, lower_signed);
            Ok(EncodedInstr::Pair(lui, addi))
        }
    }

    /// Encodes a LA pseudo-instruction (load address).
    ///
    /// LA rd, offset → AUIPC rd, %hi(offset) + ADDI rd, rd, %lo(offset)
    fn encode_la(&self, rd: u8, offset: i32) -> Result<EncodedInstr, EncoderError> {
        let lower = (offset as u32) & 0xFFF;
        let upper = if lower >= 0x800 {
            ((offset as u32).wrapping_add(0x1000) >> 12) & 0xFFFFF
        } else {
            ((offset as u32) >> 12) & 0xFFFFF
        };
        let lower_signed = if lower >= 0x800 {
            (lower as i32) - 0x1000
        } else {
            lower as i32
        };
        let auipc = self.encode_u_type(BASE_AUIPC, rd, upper);
        let addi = self.encode_i_type(OP_IMM, rd, FUNCT3_ADD, rd, lower_signed);
        Ok(EncodedInstr::Pair(auipc, addi))
    }

    /// Encodes a CALL pseudo-instruction.
    ///
    /// CALL offset → AUIPC ra, %hi(offset) + JALR ra, ra, %lo(offset)
    fn encode_call(&self, offset: i32, rd: u8) -> Result<EncodedInstr, EncoderError> {
        let lower = (offset as u32) & 0xFFF;
        let upper = if lower >= 0x800 {
            ((offset as u32).wrapping_add(0x1000) >> 12) & 0xFFFFF
        } else {
            ((offset as u32) >> 12) & 0xFFFFF
        };
        let lower_signed = if lower >= 0x800 {
            (lower as i32) - 0x1000
        } else {
            lower as i32
        };
        let auipc = self.encode_u_type(BASE_AUIPC, rd, upper);
        let jalr = self.encode_i_type(BASE_JALR, rd, FUNCT3_ADD, rd, lower_signed);
        Ok(EncodedInstr::Pair(auipc, jalr))
    }

    /// Encodes a TAIL pseudo-instruction (tail call).
    ///
    /// TAIL offset → AUIPC t1, %hi(offset) + JALR x0, t1, %lo(offset)
    /// Uses x6 (t1) as the scratch register.
    fn encode_tail(&self, offset: i32) -> Result<EncodedInstr, EncoderError> {
        let t1: u8 = 6; // x6 = t1
        let lower = (offset as u32) & 0xFFF;
        let upper = if lower >= 0x800 {
            ((offset as u32).wrapping_add(0x1000) >> 12) & 0xFFFFF
        } else {
            ((offset as u32) >> 12) & 0xFFFFF
        };
        let lower_signed = if lower >= 0x800 {
            (lower as i32) - 0x1000
        } else {
            lower as i32
        };
        let auipc = self.encode_u_type(BASE_AUIPC, t1, upper);
        let jalr = self.encode_i_type(BASE_JALR, 0, FUNCT3_ADD, t1, lower_signed);
        Ok(EncodedInstr::Pair(auipc, jalr))
    }
}
