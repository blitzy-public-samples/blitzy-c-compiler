//! # RISC-V 64 Instruction Encoder
//!
//! Encodes RV64IMAFDC instructions into binary machine code bytes.
//!
//! ## Instruction Formats (32-bit standard)
//!
//! ### R-type (Register-Register)
//! ```text
//! [31:25] funct7 | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode
//! ```
//! Examples: ADD, SUB, MUL, AND, OR, XOR, SLL, SRL, SRA
//!
//! ### I-type (Immediate)
//! ```text
//! [31:20] imm[11:0] | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode
//! ```
//! Examples: ADDI, SLTI, ANDI, JALR, LB, LH, LW, LD
//!
//! ### S-type (Store)
//! ```text
//! [31:25] imm[11:5] | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] imm[4:0] | [6:0] opcode
//! ```
//! Examples: SB, SH, SW, SD
//!
//! ### B-type (Branch)
//! ```text
//! [31] imm[12] | [30:25] imm[10:5] | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:8] imm[4:1] | [7] imm[11] | [6:0] opcode
//! ```
//! Examples: BEQ, BNE, BLT, BGE, BLTU, BGEU
//!
//! ### U-type (Upper Immediate)
//! ```text
//! [31:12] imm[31:12] | [11:7] rd | [6:0] opcode
//! ```
//! Examples: LUI, AUIPC
//!
//! ### J-type (Jump)
//! ```text
//! [31] imm[20] | [30:21] imm[10:1] | [20] imm[11] | [19:12] imm[19:12] | [11:7] rd | [6:0] opcode
//! ```
//! Examples: JAL
//!
//! ### R4-type (Fused Multiply-Add)
//! ```text
//! [31:27] rs3 | [26:25] fmt | [24:20] rs2 | [19:15] rs1 | [14:12] rm | [11:7] rd | [6:0] opcode
//! ```
//! Examples: FMADD.S, FMADD.D, FMSUB.S, etc.
//!
//! ## Compressed (C extension) 16-bit Formats
//! Formats: CR, CI, CSS, CIW, CL, CS, CB, CJ
//! Compressed registers use 3-bit encoding mapping to x8–x15 (s0–s1, a0–a5)
//!
//! ## Byte Order: Little-endian (LSB first)

use super::relocations::RiscV64RelocationType;
use crate::backend::riscv64::codegen::{RvInstruction, RvOpcode};
use crate::backend::riscv64::registers;

// ===========================================================================
// RV32I/RV64I Base Opcode Constants (7-bit opcode field, bits 6:0)
// ===========================================================================

const OP_LUI: u32 = 0b0110111;
const OP_AUIPC: u32 = 0b0010111;
const OP_JAL: u32 = 0b1101111;
const OP_JALR: u32 = 0b1100111;
const OP_BRANCH: u32 = 0b1100011;
const OP_LOAD: u32 = 0b0000011;
const OP_STORE: u32 = 0b0100011;
const OP_OP_IMM: u32 = 0b0010011;
const OP_OP: u32 = 0b0110011;
const OP_OP_IMM_32: u32 = 0b0011011;
const OP_OP_32: u32 = 0b0111011;
#[allow(dead_code)]
const OP_MISC_MEM: u32 = 0b0001111;
#[allow(dead_code)]
const OP_SYSTEM: u32 = 0b1110011;
const OP_AMO: u32 = 0b0101111;
const OP_LOAD_FP: u32 = 0b0000111;
const OP_STORE_FP: u32 = 0b0100111;
const OP_FMADD: u32 = 0b1000011;
const OP_FMSUB: u32 = 0b1000111;
const OP_FNMSUB: u32 = 0b1001011;
const OP_FNMADD: u32 = 0b1001111;
const OP_OP_FP: u32 = 0b1010011;

// ===========================================================================
// Branch funct3 values
// ===========================================================================

const FUNCT3_BEQ: u32 = 0b000;
const FUNCT3_BNE: u32 = 0b001;
const FUNCT3_BLT: u32 = 0b100;
const FUNCT3_BGE: u32 = 0b101;
const FUNCT3_BLTU: u32 = 0b110;
const FUNCT3_BGEU: u32 = 0b111;

// ===========================================================================
// Load funct3 values
// ===========================================================================

const FUNCT3_LB: u32 = 0b000;
const FUNCT3_LH: u32 = 0b001;
const FUNCT3_LW: u32 = 0b010;
const FUNCT3_LD: u32 = 0b011;
const FUNCT3_LBU: u32 = 0b100;
const FUNCT3_LHU: u32 = 0b101;
const FUNCT3_LWU: u32 = 0b110;

// ===========================================================================
// Store funct3 values
// ===========================================================================

const FUNCT3_SB: u32 = 0b000;
const FUNCT3_SH: u32 = 0b001;
const FUNCT3_SW: u32 = 0b010;
const FUNCT3_SD: u32 = 0b011;

// ===========================================================================
// ALU immediate funct3 values
// ===========================================================================

const FUNCT3_ADDI: u32 = 0b000;
const FUNCT3_SLTI: u32 = 0b010;
const FUNCT3_SLTIU: u32 = 0b011;
const FUNCT3_XORI: u32 = 0b100;
const FUNCT3_ORI: u32 = 0b110;
const FUNCT3_ANDI: u32 = 0b111;
const FUNCT3_SLLI: u32 = 0b001;
/// funct7=0 for SRLI, funct7=0x20 for SRAI
const FUNCT3_SRLI_SRAI: u32 = 0b101;

// ===========================================================================
// ALU register funct3 values
// ===========================================================================

/// funct7=0 for ADD, funct7=0x20 for SUB
const FUNCT3_ADD_SUB: u32 = 0b000;
const FUNCT3_SLL: u32 = 0b001;
const FUNCT3_SLT: u32 = 0b010;
const FUNCT3_SLTU: u32 = 0b011;
const FUNCT3_XOR: u32 = 0b100;
/// funct7=0 for SRL, funct7=0x20 for SRA
const FUNCT3_SRL_SRA: u32 = 0b101;
const FUNCT3_OR: u32 = 0b110;
const FUNCT3_AND: u32 = 0b111;

// ===========================================================================
// M extension funct3 (MUL/DIV)
// ===========================================================================

const FUNCT3_MUL: u32 = 0b000;
const FUNCT3_MULH: u32 = 0b001;
const FUNCT3_MULHSU: u32 = 0b010;
const FUNCT3_MULHU: u32 = 0b011;
const FUNCT3_DIV: u32 = 0b100;
const FUNCT3_DIVU: u32 = 0b101;
const FUNCT3_REM: u32 = 0b110;
const FUNCT3_REMU: u32 = 0b111;

// ===========================================================================
// Atomic funct3 and funct5 values
// ===========================================================================

const FUNCT3_AMO_W: u32 = 0b010;
const FUNCT3_AMO_D: u32 = 0b011;

const FUNCT5_LR: u32 = 0b00010;
const FUNCT5_SC: u32 = 0b00011;
const FUNCT5_AMOSWAP: u32 = 0b00001;
const FUNCT5_AMOADD: u32 = 0b00000;
const FUNCT5_AMOXOR: u32 = 0b00100;
const FUNCT5_AMOAND: u32 = 0b01100;
const FUNCT5_AMOOR: u32 = 0b01000;
const FUNCT5_AMOMIN: u32 = 0b10000;
const FUNCT5_AMOMAX: u32 = 0b10100;
const FUNCT5_AMOMINU: u32 = 0b11000;
const FUNCT5_AMOMAXU: u32 = 0b11100;

// ===========================================================================
// FP load/store funct3 values
// ===========================================================================

const FUNCT3_FLW: u32 = 0b010;
const FUNCT3_FLD: u32 = 0b011;
const FUNCT3_FSW: u32 = 0b010;
const FUNCT3_FSD: u32 = 0b011;

// ===========================================================================
// funct7 values
// ===========================================================================

const FUNCT7_NORMAL: u32 = 0b0000000;
const FUNCT7_SUB: u32 = 0b0100000;
const FUNCT7_MULDIV: u32 = 0b0000001;

// FP funct7 — Single precision
const FUNCT7_FADD_S: u32 = 0b0000000;
const FUNCT7_FSUB_S: u32 = 0b0000100;
const FUNCT7_FMUL_S: u32 = 0b0001000;
const FUNCT7_FDIV_S: u32 = 0b0001100;
const FUNCT7_FSQRT_S: u32 = 0b0101100;
const FUNCT7_FSGNJ_S: u32 = 0b0010000;
const FUNCT7_FMIN_S: u32 = 0b0010100;
const FUNCT7_FCMP_S: u32 = 0b1010000;
const FUNCT7_FCVT_W_S: u32 = 0b1100000;
const FUNCT7_FCVT_S_W: u32 = 0b1101000;
const FUNCT7_FMV_X_W: u32 = 0b1110000;
const FUNCT7_FMV_W_X: u32 = 0b1111000;

// FP funct7 — Double precision
const FUNCT7_FADD_D: u32 = 0b0000001;
const FUNCT7_FSUB_D: u32 = 0b0000101;
const FUNCT7_FMUL_D: u32 = 0b0001001;
const FUNCT7_FDIV_D: u32 = 0b0001101;
const FUNCT7_FSQRT_D: u32 = 0b0101101;
const FUNCT7_FSGNJ_D: u32 = 0b0010001;
const FUNCT7_FMIN_D: u32 = 0b0010101;
const FUNCT7_FCMP_D: u32 = 0b1010001;
const FUNCT7_FCVT_W_D: u32 = 0b1100001;
const FUNCT7_FCVT_D_W: u32 = 0b1101001;
const FUNCT7_FCVT_S_D: u32 = 0b0100000;
const FUNCT7_FCVT_D_S: u32 = 0b0100001;
const FUNCT7_FMV_X_D: u32 = 0b1110001;
const FUNCT7_FMV_D_X: u32 = 0b1111001;

// ===========================================================================
// Default rounding mode for FP instructions
// ===========================================================================

/// Dynamic rounding mode (uses frm CSR). Value 0b111.
const RM_DYN: u32 = 0b111;

// ===========================================================================
// FP format constants for R4-type encoding
// ===========================================================================

/// Single-precision float format (bits 26:25 = 0b00)
const FMT_S: u32 = 0b00;
/// Double-precision float format (bits 26:25 = 0b01)
const FMT_D: u32 = 0b01;

// ===========================================================================
// Compressed instruction opcode quadrants (bits 1:0)
// ===========================================================================

const C_OP_Q0: u16 = 0b00;
const C_OP_Q1: u16 = 0b01;
const C_OP_Q2: u16 = 0b10;

// ===========================================================================
// Public Types
// ===========================================================================

/// Encoded instruction result.
///
/// Contains the binary machine code bytes for one RISC-V instruction,
/// along with optional relocation information and a continuation
/// instruction for pseudo-ops that expand into two-instruction sequences.
pub struct EncodedInstruction {
    /// Encoded bytes (4 for standard, 2 for compressed).
    pub bytes: Vec<u8>,
    /// Size in bytes (2 or 4).
    pub size: u8,
    /// Optional relocation if instruction references an unresolved symbol.
    pub relocation: Option<EncoderRelocation>,
    /// For pseudo-instructions that expand to 2 real instructions
    /// (e.g., CALL → AUIPC + JALR), the second instruction is stored here.
    pub continuation: Option<Box<EncodedInstruction>>,
}

/// Relocation request from the encoder.
///
/// Emitted when an instruction references a symbol that cannot be resolved
/// at assembly time (function calls, global variable accesses, etc.).
pub struct EncoderRelocation {
    /// Byte offset relative to instruction start (0 for single-instruction relocs).
    pub offset: u8,
    /// Relocation type as a raw ELF relocation number.
    pub reloc_type: u32,
    /// Addend for the relocation computation.
    pub addend: i64,
}

// ===========================================================================
// RiscV64Encoder
// ===========================================================================

/// RISC-V 64 instruction encoder.
///
/// Encodes [`RvInstruction`] into binary machine code bytes (little-endian).
/// Supports all RV64IMAFDC instruction formats plus pseudo-instruction
/// expansion and optional compressed (C extension) encoding.
pub struct RiscV64Encoder {
    // Encoding is stateless — all information comes from the instruction.
}

impl Default for RiscV64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RiscV64Encoder {
    /// Create a new RISC-V 64 instruction encoder.
    pub fn new() -> Self {
        RiscV64Encoder {}
    }

    /// Encode a single [`RvInstruction`] into machine code bytes.
    ///
    /// Returns the encoded instruction including any relocations needed.
    /// For pseudo-instructions that expand to multiple real instructions
    /// (e.g., CALL → AUIPC + JALR), the `continuation` field contains
    /// the second instruction.
    ///
    /// # Match Block Organization
    ///
    /// The main match block is organized by instruction category:
    /// 1. **Pseudo-instructions** — NOP, MV, LI, LA, CALL, J, etc.
    /// 2. **RV64I Base Integer** — ALU, load/store, branch, jump
    /// 3. **RV64M Multiply/Divide** — MUL, DIV, REM variants
    /// 4. **RV64A Atomics** — LR, SC, AMO operations
    /// 5. **RV64F Single-Precision FP** — FLW, FSW, FADD.S, etc.
    /// 6. **RV64D Double-Precision FP** — FLD, FSD, FADD.D, etc.
    /// 7. **RV64C Compressed** — 16-bit compressed instructions
    /// 8. **Pseudo-ops** — Labels, directives, inline asm
    pub fn encode(&self, inst: &RvInstruction) -> Result<EncodedInstruction, String> {
        // Extract register hardware encodings (5-bit values 0–31).
        let rd = registers::hw_encoding(inst.rd.unwrap_or(0));
        let rs1 = registers::hw_encoding(inst.rs1.unwrap_or(0));
        let rs2 = registers::hw_encoding(inst.rs2.unwrap_or(0));
        let rs3 = registers::hw_encoding(inst.rs3.unwrap_or(0));
        let imm = inst.imm;
        let imm32 = imm as i32;

        match inst.opcode {
            // =============================================================
            // Pseudo-instructions — expanded to real instructions
            // =============================================================
            RvOpcode::NOP => {
                // NOP → ADDI x0, x0, 0
                Ok(make_std(encode_i_type(OP_OP_IMM, 0, FUNCT3_ADDI, 0, 0)))
            }

            RvOpcode::MV => {
                // MV rd, rs1 → ADDI rd, rs1, 0
                Ok(make_std(encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rs1, 0)))
            }

            RvOpcode::LI => self.encode_li(rd, imm, &inst.symbol),

            RvOpcode::LA => self.encode_la(rd, imm, &inst.symbol),

            RvOpcode::CALL => self.encode_call(&inst.symbol, imm),

            RvOpcode::RET => {
                // RET → JALR x0, ra, 0
                let ra_hw = registers::hw_encoding(registers::RA);
                let zero_hw = registers::hw_encoding(registers::ZERO);
                Ok(make_std(encode_i_type(OP_JALR, zero_hw, 0, ra_hw, 0)))
            }

            RvOpcode::NEG => {
                // NEG rd, rs2 → SUB rd, x0, rs2
                let zero_hw = registers::hw_encoding(registers::ZERO);
                Ok(make_std(encode_r_type(
                    OP_OP,
                    rd,
                    FUNCT3_ADD_SUB,
                    zero_hw,
                    rs2,
                    FUNCT7_SUB,
                )))
            }

            RvOpcode::NOT => {
                // NOT rd, rs1 → XORI rd, rs1, -1
                Ok(make_std(encode_i_type(OP_OP_IMM, rd, FUNCT3_XORI, rs1, -1)))
            }

            RvOpcode::SEQZ => {
                // SEQZ rd, rs1 → SLTIU rd, rs1, 1
                Ok(make_std(encode_i_type(OP_OP_IMM, rd, FUNCT3_SLTIU, rs1, 1)))
            }

            RvOpcode::SNEZ => {
                // SNEZ rd, rs2 → SLTU rd, x0, rs2
                let zero_hw = registers::hw_encoding(registers::ZERO);
                Ok(make_std(encode_r_type(
                    OP_OP,
                    rd,
                    FUNCT3_SLTU,
                    zero_hw,
                    rs2,
                    FUNCT7_NORMAL,
                )))
            }

            RvOpcode::J => {
                // J offset → JAL x0, offset
                if inst.symbol.is_some() {
                    let word = encode_j_type(OP_JAL, 0, 0);
                    Ok(EncodedInstruction {
                        bytes: word.to_le_bytes().to_vec(),
                        size: 4,
                        relocation: Some(EncoderRelocation {
                            offset: 0,
                            reloc_type: RiscV64RelocationType::Jal.as_elf_type(),
                            addend: imm,
                        }),
                        continuation: None,
                    })
                } else {
                    Ok(make_std(encode_j_type(OP_JAL, 0, imm32)))
                }
            }

            // =============================================================
            // U-type: LUI, AUIPC
            // =============================================================
            RvOpcode::LUI => {
                if inst.symbol.is_some() {
                    let word = encode_u_type(OP_LUI, rd, 0);
                    Ok(EncodedInstruction {
                        bytes: word.to_le_bytes().to_vec(),
                        size: 4,
                        relocation: Some(EncoderRelocation {
                            offset: 0,
                            reloc_type: RiscV64RelocationType::Hi20.as_elf_type(),
                            addend: imm,
                        }),
                        continuation: None,
                    })
                } else {
                    // The codegen provides the raw upper 20-bit value (not
                    // pre-shifted).  encode_u_type expects the value with
                    // significant bits at positions [31:12], so we shift
                    // left by 12 here.  NOTE: encode_li has its own shift
                    // and calls encode_u_type directly, so this only
                    // affects LUI instructions created by the codegen
                    // (e.g., from materialize_constant's split_i32_lui_addi).
                    let shifted = (imm32 as i64) << 12;
                    Ok(make_std(encode_u_type(OP_LUI, rd, shifted as i32)))
                }
            }

            RvOpcode::AUIPC => {
                if inst.symbol.is_some() {
                    let word = encode_u_type(OP_AUIPC, rd, 0);
                    Ok(EncodedInstruction {
                        bytes: word.to_le_bytes().to_vec(),
                        size: 4,
                        relocation: Some(EncoderRelocation {
                            offset: 0,
                            reloc_type: RiscV64RelocationType::PcrelHi20.as_elf_type(),
                            addend: imm,
                        }),
                        continuation: None,
                    })
                } else {
                    Ok(make_std(encode_u_type(OP_AUIPC, rd, imm32)))
                }
            }

            // =============================================================
            // J-type: JAL
            // =============================================================
            RvOpcode::JAL => {
                if inst.symbol.is_some() {
                    let word = encode_j_type(OP_JAL, rd, 0);
                    Ok(EncodedInstruction {
                        bytes: word.to_le_bytes().to_vec(),
                        size: 4,
                        relocation: Some(EncoderRelocation {
                            offset: 0,
                            reloc_type: RiscV64RelocationType::Jal.as_elf_type(),
                            addend: imm,
                        }),
                        continuation: None,
                    })
                } else {
                    Ok(make_std(encode_j_type(OP_JAL, rd, imm32)))
                }
            }

            // =============================================================
            // I-type: JALR
            // =============================================================
            RvOpcode::JALR => Ok(make_std(encode_i_type(OP_JALR, rd, 0, rs1, imm32))),

            // =============================================================
            // B-type: Branch instructions
            // =============================================================
            RvOpcode::BEQ => self.encode_branch(FUNCT3_BEQ, rs1, rs2, imm, &inst.symbol),
            RvOpcode::BNE => self.encode_branch(FUNCT3_BNE, rs1, rs2, imm, &inst.symbol),
            RvOpcode::BLT => self.encode_branch(FUNCT3_BLT, rs1, rs2, imm, &inst.symbol),
            RvOpcode::BGE => self.encode_branch(FUNCT3_BGE, rs1, rs2, imm, &inst.symbol),
            RvOpcode::BLTU => self.encode_branch(FUNCT3_BLTU, rs1, rs2, imm, &inst.symbol),
            RvOpcode::BGEU => self.encode_branch(FUNCT3_BGEU, rs1, rs2, imm, &inst.symbol),

            // =============================================================
            // I-type: Load instructions
            // =============================================================
            RvOpcode::LB => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LB, rs1, imm32))),
            RvOpcode::LH => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LH, rs1, imm32))),
            RvOpcode::LW => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LW, rs1, imm32))),
            RvOpcode::LD => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LD, rs1, imm32))),
            RvOpcode::LBU => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LBU, rs1, imm32))),
            RvOpcode::LHU => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LHU, rs1, imm32))),
            RvOpcode::LWU => Ok(make_std(encode_i_type(OP_LOAD, rd, FUNCT3_LWU, rs1, imm32))),

            // =============================================================
            // S-type: Store instructions
            // =============================================================
            RvOpcode::SB => Ok(make_std(encode_s_type(
                OP_STORE, FUNCT3_SB, rs1, rs2, imm32,
            ))),
            RvOpcode::SH => Ok(make_std(encode_s_type(
                OP_STORE, FUNCT3_SH, rs1, rs2, imm32,
            ))),
            RvOpcode::SW => Ok(make_std(encode_s_type(
                OP_STORE, FUNCT3_SW, rs1, rs2, imm32,
            ))),
            RvOpcode::SD => Ok(make_std(encode_s_type(
                OP_STORE, FUNCT3_SD, rs1, rs2, imm32,
            ))),

            // =============================================================
            // I-type: ALU immediate
            // =============================================================
            RvOpcode::ADDI => Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_ADDI,
                rs1,
                imm32,
            ))),
            RvOpcode::SLTI => Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_SLTI,
                rs1,
                imm32,
            ))),
            RvOpcode::SLTIU => Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_SLTIU,
                rs1,
                imm32,
            ))),
            RvOpcode::XORI => Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_XORI,
                rs1,
                imm32,
            ))),
            RvOpcode::ORI => Ok(make_std(encode_i_type(
                OP_OP_IMM, rd, FUNCT3_ORI, rs1, imm32,
            ))),
            RvOpcode::ANDI => Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_ANDI,
                rs1,
                imm32,
            ))),

            // Shift immediate (I-type with funct7 encoded in upper bits of imm)
            RvOpcode::SLLI => {
                // RV64: shamt is 6 bits (imm[5:0]), funct6=000000 in bits [31:26]
                let shamt = (imm as u32) & 0x3F;
                let imm_field = shamt as i32; // upper bits are 0 for SLLI
                Ok(make_std(encode_i_type(
                    OP_OP_IMM,
                    rd,
                    FUNCT3_SLLI,
                    rs1,
                    imm_field,
                )))
            }
            RvOpcode::SRLI => {
                // funct6=000000 in bits [31:26], shamt in [5:0]
                let shamt = (imm as u32) & 0x3F;
                let imm_field = shamt as i32;
                Ok(make_std(encode_i_type(
                    OP_OP_IMM,
                    rd,
                    FUNCT3_SRLI_SRAI,
                    rs1,
                    imm_field,
                )))
            }
            RvOpcode::SRAI => {
                // funct6=010000 in bits [31:26], shamt in [5:0]
                let shamt = (imm as u32) & 0x3F;
                let imm_field = (0b010000 << 6 | shamt) as i32;
                Ok(make_std(encode_i_type(
                    OP_OP_IMM,
                    rd,
                    FUNCT3_SRLI_SRAI,
                    rs1,
                    imm_field,
                )))
            }

            // =============================================================
            // R-type: ALU register-register
            // =============================================================
            RvOpcode::ADD => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_ADD_SUB,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SUB => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_ADD_SUB,
                rs1,
                rs2,
                FUNCT7_SUB,
            ))),
            RvOpcode::SLL => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_SLL,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SLT => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_SLT,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SLTU => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_SLTU,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::XOR => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_XOR,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SRL => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_SRL_SRA,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SRA => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_SRL_SRA,
                rs1,
                rs2,
                FUNCT7_SUB,
            ))),
            RvOpcode::OR => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_OR,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::AND => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_AND,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),

            // =============================================================
            // RV64I Word variants (32-bit ops, sign-extended result)
            // =============================================================
            RvOpcode::ADDIW => Ok(make_std(encode_i_type(
                OP_OP_IMM_32,
                rd,
                FUNCT3_ADDI,
                rs1,
                imm32,
            ))),
            RvOpcode::SLLIW => {
                // 5-bit shamt, funct7=0000000
                let shamt = (imm as u32) & 0x1F;
                Ok(make_std(encode_i_type(
                    OP_OP_IMM_32,
                    rd,
                    FUNCT3_SLLI,
                    rs1,
                    shamt as i32,
                )))
            }
            RvOpcode::SRLIW => {
                let shamt = (imm as u32) & 0x1F;
                Ok(make_std(encode_i_type(
                    OP_OP_IMM_32,
                    rd,
                    FUNCT3_SRLI_SRAI,
                    rs1,
                    shamt as i32,
                )))
            }
            RvOpcode::SRAIW => {
                let shamt = (imm as u32) & 0x1F;
                let imm_field = (0b0100000 << 5 | shamt) as i32;
                Ok(make_std(encode_i_type(
                    OP_OP_IMM_32,
                    rd,
                    FUNCT3_SRLI_SRAI,
                    rs1,
                    imm_field,
                )))
            }
            RvOpcode::ADDW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_ADD_SUB,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SUBW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_ADD_SUB,
                rs1,
                rs2,
                FUNCT7_SUB,
            ))),
            RvOpcode::SLLW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_SLL,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SRLW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_SRL_SRA,
                rs1,
                rs2,
                FUNCT7_NORMAL,
            ))),
            RvOpcode::SRAW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_SRL_SRA,
                rs1,
                rs2,
                FUNCT7_SUB,
            ))),

            // =============================================================
            // RV64M: Multiply/Divide
            // =============================================================
            RvOpcode::MUL => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_MUL,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::MULH => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_MULH,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::MULHSU => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_MULHSU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::MULHU => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_MULHU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::DIV => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_DIV,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::DIVU => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_DIVU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::REM => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_REM,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::REMU => Ok(make_std(encode_r_type(
                OP_OP,
                rd,
                FUNCT3_REMU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),

            // RV64M Word variants
            RvOpcode::MULW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_MUL,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::DIVW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_DIV,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::DIVUW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_DIVU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::REMW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_REM,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),
            RvOpcode::REMUW => Ok(make_std(encode_r_type(
                OP_OP_32,
                rd,
                FUNCT3_REMU,
                rs1,
                rs2,
                FUNCT7_MULDIV,
            ))),

            // =============================================================
            // RV64A: Atomic instructions
            // =============================================================

            // Atomic Word (32-bit) operations
            RvOpcode::LR_W => Ok(make_std(encode_amo(
                FUNCT5_LR,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                0,
            ))),
            RvOpcode::SC_W => Ok(make_std(encode_amo(
                FUNCT5_SC,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOSWAP_W => Ok(make_std(encode_amo(
                FUNCT5_AMOSWAP,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOADD_W => Ok(make_std(encode_amo(
                FUNCT5_AMOADD,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOAND_W => Ok(make_std(encode_amo(
                FUNCT5_AMOAND,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOOR_W => Ok(make_std(encode_amo(
                FUNCT5_AMOOR,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOXOR_W => Ok(make_std(encode_amo(
                FUNCT5_AMOXOR,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMAX_W => Ok(make_std(encode_amo(
                FUNCT5_AMOMAX,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMIN_W => Ok(make_std(encode_amo(
                FUNCT5_AMOMIN,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMAXU_W => Ok(make_std(encode_amo(
                FUNCT5_AMOMAXU,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMINU_W => Ok(make_std(encode_amo(
                FUNCT5_AMOMINU,
                0,
                0,
                FUNCT3_AMO_W,
                rd,
                rs1,
                rs2,
            ))),

            // Atomic Doubleword (64-bit) operations
            RvOpcode::LR_D => Ok(make_std(encode_amo(
                FUNCT5_LR,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                0,
            ))),
            RvOpcode::SC_D => Ok(make_std(encode_amo(
                FUNCT5_SC,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOSWAP_D => Ok(make_std(encode_amo(
                FUNCT5_AMOSWAP,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOADD_D => Ok(make_std(encode_amo(
                FUNCT5_AMOADD,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOAND_D => Ok(make_std(encode_amo(
                FUNCT5_AMOAND,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOOR_D => Ok(make_std(encode_amo(
                FUNCT5_AMOOR,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOXOR_D => Ok(make_std(encode_amo(
                FUNCT5_AMOXOR,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMAX_D => Ok(make_std(encode_amo(
                FUNCT5_AMOMAX,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMIN_D => Ok(make_std(encode_amo(
                FUNCT5_AMOMIN,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMAXU_D => Ok(make_std(encode_amo(
                FUNCT5_AMOMAXU,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),
            RvOpcode::AMOMINU_D => Ok(make_std(encode_amo(
                FUNCT5_AMOMINU,
                0,
                0,
                FUNCT3_AMO_D,
                rd,
                rs1,
                rs2,
            ))),

            // =============================================================
            // RV64F: Single-precision floating-point
            // =============================================================
            RvOpcode::FLW => Ok(make_std(encode_i_type(
                OP_LOAD_FP, rd, FUNCT3_FLW, rs1, imm32,
            ))),
            RvOpcode::FSW => Ok(make_std(encode_s_type(
                OP_STORE_FP,
                FUNCT3_FSW,
                rs1,
                rs2,
                imm32,
            ))),

            RvOpcode::FADD_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FADD_S,
            ))),
            RvOpcode::FSUB_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FSUB_S,
            ))),
            RvOpcode::FMUL_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FMUL_S,
            ))),
            RvOpcode::FDIV_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FDIV_S,
            ))),
            RvOpcode::FSQRT_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FSQRT_S,
            ))),

            RvOpcode::FMIN_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FMIN_S,
            ))),
            RvOpcode::FMAX_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FMIN_S,
            ))),

            RvOpcode::FSGNJ_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FSGNJ_S,
            ))),
            RvOpcode::FSGNJN_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FSGNJ_S,
            ))),
            RvOpcode::FSGNJX_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b010,
                rs1,
                rs2,
                FUNCT7_FSGNJ_S,
            ))),

            RvOpcode::FCVT_W_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FCVT_W_S,
            ))),
            RvOpcode::FCVT_WU_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                1,
                FUNCT7_FCVT_W_S,
            ))),
            RvOpcode::FCVT_L_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                2,
                FUNCT7_FCVT_W_S,
            ))),
            RvOpcode::FCVT_LU_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                3,
                FUNCT7_FCVT_W_S,
            ))),

            RvOpcode::FCVT_S_W => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FCVT_S_W,
            ))),
            RvOpcode::FCVT_S_WU => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                1,
                FUNCT7_FCVT_S_W,
            ))),
            RvOpcode::FCVT_S_L => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                2,
                FUNCT7_FCVT_S_W,
            ))),
            RvOpcode::FCVT_S_LU => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                3,
                FUNCT7_FCVT_S_W,
            ))),

            RvOpcode::FMV_X_W => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                0,
                FUNCT7_FMV_X_W,
            ))),
            RvOpcode::FMV_W_X => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                0,
                FUNCT7_FMV_W_X,
            ))),

            RvOpcode::FEQ_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b010,
                rs1,
                rs2,
                FUNCT7_FCMP_S,
            ))),
            RvOpcode::FLT_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FCMP_S,
            ))),
            RvOpcode::FLE_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FCMP_S,
            ))),

            RvOpcode::FCLASS_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                0,
                FUNCT7_FMV_X_W,
            ))),

            // R4-type: Fused multiply-add (single)
            RvOpcode::FMADD_S => Ok(make_std(encode_r4_type(
                OP_FMADD, rd, RM_DYN, rs1, rs2, rs3, FMT_S,
            ))),
            RvOpcode::FMSUB_S => Ok(make_std(encode_r4_type(
                OP_FMSUB, rd, RM_DYN, rs1, rs2, rs3, FMT_S,
            ))),
            RvOpcode::FNMSUB_S => Ok(make_std(encode_r4_type(
                OP_FNMSUB, rd, RM_DYN, rs1, rs2, rs3, FMT_S,
            ))),
            RvOpcode::FNMADD_S => Ok(make_std(encode_r4_type(
                OP_FNMADD, rd, RM_DYN, rs1, rs2, rs3, FMT_S,
            ))),

            // =============================================================
            // RV64D: Double-precision floating-point
            // =============================================================
            RvOpcode::FLD => Ok(make_std(encode_i_type(
                OP_LOAD_FP, rd, FUNCT3_FLD, rs1, imm32,
            ))),
            RvOpcode::FSD => Ok(make_std(encode_s_type(
                OP_STORE_FP,
                FUNCT3_FSD,
                rs1,
                rs2,
                imm32,
            ))),

            RvOpcode::FADD_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FADD_D,
            ))),
            RvOpcode::FSUB_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FSUB_D,
            ))),
            RvOpcode::FMUL_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FMUL_D,
            ))),
            RvOpcode::FDIV_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                rs2,
                FUNCT7_FDIV_D,
            ))),
            RvOpcode::FSQRT_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FSQRT_D,
            ))),

            RvOpcode::FMIN_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FMIN_D,
            ))),
            RvOpcode::FMAX_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FMIN_D,
            ))),

            RvOpcode::FSGNJ_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FSGNJ_D,
            ))),
            RvOpcode::FSGNJN_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FSGNJ_D,
            ))),
            RvOpcode::FSGNJX_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b010,
                rs1,
                rs2,
                FUNCT7_FSGNJ_D,
            ))),

            RvOpcode::FCVT_W_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FCVT_W_D,
            ))),
            RvOpcode::FCVT_WU_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                1,
                FUNCT7_FCVT_W_D,
            ))),
            RvOpcode::FCVT_L_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                2,
                FUNCT7_FCVT_W_D,
            ))),
            RvOpcode::FCVT_LU_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                3,
                FUNCT7_FCVT_W_D,
            ))),

            RvOpcode::FCVT_D_W => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FCVT_D_W,
            ))),
            RvOpcode::FCVT_D_WU => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                1,
                FUNCT7_FCVT_D_W,
            ))),
            RvOpcode::FCVT_D_L => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                2,
                FUNCT7_FCVT_D_W,
            ))),
            RvOpcode::FCVT_D_LU => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                3,
                FUNCT7_FCVT_D_W,
            ))),

            // Cross-format conversions
            RvOpcode::FCVT_S_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                1,
                FUNCT7_FCVT_S_D,
            ))),
            RvOpcode::FCVT_D_S => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                RM_DYN,
                rs1,
                0,
                FUNCT7_FCVT_D_S,
            ))),

            // 64-bit FP move (RV64 only)
            RvOpcode::FMV_X_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                0,
                FUNCT7_FMV_X_D,
            ))),
            RvOpcode::FMV_D_X => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                0,
                FUNCT7_FMV_D_X,
            ))),

            RvOpcode::FEQ_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b010,
                rs1,
                rs2,
                FUNCT7_FCMP_D,
            ))),
            RvOpcode::FLT_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                rs2,
                FUNCT7_FCMP_D,
            ))),
            RvOpcode::FLE_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b000,
                rs1,
                rs2,
                FUNCT7_FCMP_D,
            ))),

            RvOpcode::FCLASS_D => Ok(make_std(encode_r_type(
                OP_OP_FP,
                rd,
                0b001,
                rs1,
                0,
                FUNCT7_FMV_X_D,
            ))),

            // R4-type: Fused multiply-add (double)
            RvOpcode::FMADD_D => Ok(make_std(encode_r4_type(
                OP_FMADD, rd, RM_DYN, rs1, rs2, rs3, FMT_D,
            ))),
            RvOpcode::FMSUB_D => Ok(make_std(encode_r4_type(
                OP_FMSUB, rd, RM_DYN, rs1, rs2, rs3, FMT_D,
            ))),
            RvOpcode::FNMSUB_D => Ok(make_std(encode_r4_type(
                OP_FNMSUB, rd, RM_DYN, rs1, rs2, rs3, FMT_D,
            ))),
            RvOpcode::FNMADD_D => Ok(make_std(encode_r4_type(
                OP_FNMADD, rd, RM_DYN, rs1, rs2, rs3, FMT_D,
            ))),

            // =============================================================
            // System / CSR instructions (I-type with OP_SYSTEM opcode)
            // =============================================================

            // CSR Read-Write: CSRRW rd, csr, rs1
            // I-type encoding: imm[11:0]=csr, rs1, funct3=001, rd, opcode=SYSTEM
            RvOpcode::CSRRW => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b001, rs1, csr)))
            }
            // CSR Read-Set: CSRRS rd, csr, rs1
            RvOpcode::CSRRS => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b010, rs1, csr)))
            }
            // CSR Read-Clear: CSRRC rd, csr, rs1
            RvOpcode::CSRRC => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b011, rs1, csr)))
            }
            // CSR Read-Write Immediate: CSRRWI rd, csr, zimm
            // rs1 field holds the 5-bit unsigned immediate (zimm).
            RvOpcode::CSRRWI => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b101, rs1, csr)))
            }
            // CSR Read-Set Immediate: CSRRSI rd, csr, zimm
            RvOpcode::CSRRSI => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b110, rs1, csr)))
            }
            // CSR Read-Clear Immediate: CSRRCI rd, csr, zimm
            RvOpcode::CSRRCI => {
                let csr = imm32 & 0xFFF;
                Ok(make_std(encode_i_type(OP_SYSTEM, rd, 0b111, rs1, csr)))
            }

            // ECALL: system call — SYSTEM + funct3=0, funct12=0x000
            RvOpcode::ECALL => Ok(make_std(0x0000_0073)),
            // EBREAK: breakpoint — SYSTEM + funct3=0, funct12=0x001
            RvOpcode::EBREAK => Ok(make_std(0x0010_0073)),
            // FENCE: memory fence — full ordering (pred=iorw, succ=iorw)
            RvOpcode::FENCE => Ok(make_std(0x0ff0_000f)),
            // FENCE.I: instruction fence
            RvOpcode::FENCE_I => Ok(make_std(0x0000_100f)),
            // WFI: wait for interrupt — SYSTEM, funct12=0x105
            RvOpcode::WFI => Ok(make_std(0x1050_0073)),
            // SFENCE.VMA rs1, rs2 — R-type SYSTEM, funct7=0001001
            RvOpcode::SFENCE_VMA => Ok(make_std(encode_r_type(
                OP_SYSTEM, 0, 0b000, rs1, rs2, 0b0001001,
            ))),
            // MRET: machine return — funct12=0x302
            RvOpcode::MRET => Ok(make_std(0x3020_0073)),
            // SRET: supervisor return — funct12=0x102
            RvOpcode::SRET => Ok(make_std(0x1020_0073)),

            // =============================================================
            // Inline ASM passthrough — raw bytes are handled elsewhere
            // =============================================================
            RvOpcode::INLINE_ASM => {
                // Inline assembly is stored as raw bytes; encode as NOP placeholder
                // if it reaches the encoder (the assembler module handles it separately).
                Ok(make_std(encode_i_type(OP_OP_IMM, 0, FUNCT3_ADDI, 0, 0)))
            }
        }
    }

    /// Try to encode as a compressed (16-bit C extension) instruction.
    ///
    /// Returns `None` if the instruction is not compressible. Compressed
    /// instructions use the RV64C encoding format which is only 16 bits.
    /// Not all instructions can be compressed — only a subset with
    /// restricted register and immediate ranges qualify.
    pub fn try_compress(&self, inst: &RvInstruction) -> Option<EncodedInstruction> {
        let rd = inst.rd.unwrap_or(0);
        let rs1 = inst.rs1.unwrap_or(0);
        let rs2 = inst.rs2.unwrap_or(0);
        let rd_hw = registers::hw_encoding(rd);
        let rs1_hw = registers::hw_encoding(rs1);
        let rs2_hw = registers::hw_encoding(rs2);
        let imm = inst.imm;

        // Cannot compress instructions with symbol references (need full relocation).
        if inst.symbol.is_some() {
            return None;
        }

        match inst.opcode {
            // C.NOP: addi x0, x0, 0
            RvOpcode::NOP => {
                // C.NOP: [15:13]=000, [12]=0, [11:7]=00000, [6:2]=00000, [1:0]=01
                let half: u16 = 0b0000_0000_0000_0001;
                Some(make_compressed(half))
            }

            // C.ADDI: addi rd, rd, imm (rd != 0, imm != 0, 6-bit signed)
            RvOpcode::ADDI
                if rd_hw != 0 && rd_hw == rs1_hw && imm != 0 && fits_imm_bits(imm, 6) =>
            {
                let nzimm = (imm as u16) & 0x3F;
                // [15:13]=000, [12]=nzimm[5], [11:7]=rd, [6:2]=nzimm[4:0], [1:0]=01
                let half: u16 = (((nzimm >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | ((nzimm & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.LI: addi rd, x0, imm (rd != 0, 6-bit signed)
            RvOpcode::ADDI if rd_hw != 0 && rs1_hw == 0 && fits_imm_bits(imm, 6) => {
                let imm6 = (imm as u16) & 0x3F;
                // [15:13]=010, [12]=imm[5], [11:7]=rd, [6:2]=imm[4:0], [1:0]=01
                let half: u16 = (0b010 << 13)
                    | (((imm6 >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | ((imm6 & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.LI for LI pseudo-instruction
            RvOpcode::LI if rd_hw != 0 && fits_imm_bits(imm, 6) => {
                let imm6 = (imm as u16) & 0x3F;
                let half: u16 = (0b010 << 13)
                    | (((imm6 >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | ((imm6 & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.MV: add rd, x0, rs2 → CR format (rd != 0, rs2 != 0)
            RvOpcode::MV if rd_hw != 0 && rs2_hw != 0 => {
                // [15:12]=1000, [11:7]=rd, [6:2]=rs2, [1:0]=10
                let half: u16 =
                    (0b1000 << 12) | ((rd_hw as u16) << 7) | ((rs2_hw as u16) << 2) | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.ADD: add rd, rd, rs2 → CR format (rd != 0, rs2 != 0)
            RvOpcode::ADD if rd_hw != 0 && rd_hw == rs1_hw && rs2_hw != 0 => {
                // [15:12]=1001, [11:7]=rd, [6:2]=rs2, [1:0]=10
                let half: u16 =
                    (0b1001 << 12) | ((rd_hw as u16) << 7) | ((rs2_hw as u16) << 2) | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.JR: jalr x0, rs1, 0 → CR format (rs1 != 0)
            RvOpcode::JALR if rd_hw == 0 && rs1_hw != 0 && imm == 0 => {
                // [15:12]=1000, [11:7]=rs1, [6:2]=00000, [1:0]=10
                let half: u16 = (0b1000 << 12) | ((rs1_hw as u16) << 7) | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.JALR: jalr ra, rs1, 0 → CR format (rs1 != 0)
            RvOpcode::JALR if rd_hw == 1 && rs1_hw != 0 && imm == 0 => {
                // [15:12]=1001, [11:7]=rs1, [6:2]=00000, [1:0]=10
                let half: u16 = (0b1001 << 12) | ((rs1_hw as u16) << 7) | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.RET → C.JR ra
            RvOpcode::RET => {
                let ra_hw = registers::hw_encoding(registers::RA);
                let half: u16 = (0b1000 << 12) | ((ra_hw as u16) << 7) | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.SLLI: slli rd, rd, shamt (rd != 0, shamt 1–63)
            RvOpcode::SLLI if rd_hw != 0 && rd_hw == rs1_hw && imm > 0 && imm < 64 => {
                let shamt = imm as u16;
                // [15:13]=000, [12]=shamt[5], [11:7]=rd, [6:2]=shamt[4:0], [1:0]=10
                let half: u16 = (((shamt >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | ((shamt & 0x1F) << 2)
                    | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.LW: lw rd', offset(rs1') — both in x8–x15, offset 0–124 (word-aligned)
            RvOpcode::LW
                if is_creg(rd) && is_creg(rs1) && (0..128).contains(&imm) && imm % 4 == 0 =>
            {
                let rd_c = creg_encode(rd);
                let rs1_c = creg_encode(rs1);
                let off = imm as u16;
                // [15:13]=010, [12:10]=offset[5:3], [9:7]=rs1', [6]=offset[2],
                // [5]=offset[6], [4:2]=rd', [1:0]=00
                let half: u16 = (0b010 << 13)
                    | (((off >> 3) & 0x7) << 10)
                    | (rs1_c << 7)
                    | (((off >> 2) & 1) << 6)
                    | (((off >> 6) & 1) << 5)
                    | (rd_c << 2)
                    | C_OP_Q0;
                Some(make_compressed(half))
            }

            // C.LD: ld rd', offset(rs1') — both in x8–x15, offset 0–248 (dword-aligned)
            RvOpcode::LD
                if is_creg(rd) && is_creg(rs1) && (0..256).contains(&imm) && imm % 8 == 0 =>
            {
                let rd_c = creg_encode(rd);
                let rs1_c = creg_encode(rs1);
                let off = imm as u16;
                // [15:13]=011, [12:10]=offset[5:3], [9:7]=rs1', [6:5]=offset[7:6],
                // [4:2]=rd', [1:0]=00
                let half: u16 = (0b011 << 13)
                    | (((off >> 3) & 0x7) << 10)
                    | (rs1_c << 7)
                    | (((off >> 6) & 0x3) << 5)
                    | (rd_c << 2)
                    | C_OP_Q0;
                Some(make_compressed(half))
            }

            // C.SW: sw rs2', offset(rs1') — both in x8–x15
            RvOpcode::SW
                if is_creg(rs1) && is_creg(rs2) && (0..128).contains(&imm) && imm % 4 == 0 =>
            {
                let rs2_c = creg_encode(rs2);
                let rs1_c = creg_encode(rs1);
                let off = imm as u16;
                // [15:13]=110, [12:10]=offset[5:3], [9:7]=rs1', [6]=offset[2],
                // [5]=offset[6], [4:2]=rs2', [1:0]=00
                let half: u16 = (0b110 << 13)
                    | (((off >> 3) & 0x7) << 10)
                    | (rs1_c << 7)
                    | (((off >> 2) & 1) << 6)
                    | (((off >> 6) & 1) << 5)
                    | (rs2_c << 2)
                    | C_OP_Q0;
                Some(make_compressed(half))
            }

            // C.SD: sd rs2', offset(rs1') — both in x8–x15
            RvOpcode::SD
                if is_creg(rs1) && is_creg(rs2) && (0..256).contains(&imm) && imm % 8 == 0 =>
            {
                let rs2_c = creg_encode(rs2);
                let rs1_c = creg_encode(rs1);
                let off = imm as u16;
                // [15:13]=111, [12:10]=offset[5:3], [9:7]=rs1', [6:5]=offset[7:6],
                // [4:2]=rs2', [1:0]=00
                let half: u16 = (0b111 << 13)
                    | (((off >> 3) & 0x7) << 10)
                    | (rs1_c << 7)
                    | (((off >> 6) & 0x3) << 5)
                    | (rs2_c << 2)
                    | C_OP_Q0;
                Some(make_compressed(half))
            }

            // C.LDSP: ld rd, offset(sp) — rd != 0, offset 0–504 dword-aligned
            RvOpcode::LD
                if rd_hw != 0
                    && rs1_hw == registers::hw_encoding(registers::SP)
                    && (0..512).contains(&imm)
                    && imm % 8 == 0 =>
            {
                let off = imm as u16;
                // [15:13]=011, [12]=offset[5], [11:7]=rd, [6:4]=offset[4:3|8:6]...
                // Encoding: [12]=off[5], [6:5]=off[4:3], [4:2]=off[8:6]
                let half: u16 = (0b011 << 13)
                    | (((off >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | (((off >> 3) & 0x3) << 5)
                    | (((off >> 6) & 0x7) << 2)
                    | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.SDSP: sd rs2, offset(sp) — offset 0–504 dword-aligned
            RvOpcode::SD
                if rs1_hw == registers::hw_encoding(registers::SP)
                    && (0..512).contains(&imm)
                    && imm % 8 == 0 =>
            {
                let off = imm as u16;
                // [15:13]=111, [12:10]=offset[5:3], [9:7]=offset[8:6], [6:2]=rs2, [1:0]=10
                let half: u16 = (0b111 << 13)
                    | (((off >> 3) & 0x7) << 10)
                    | (((off >> 6) & 0x7) << 7)
                    | ((rs2_hw as u16) << 2)
                    | C_OP_Q2;
                Some(make_compressed(half))
            }

            // C.BEQZ / C.BNEZ: branch if rs1'==0 / rs1'!=0, 9-bit signed even offset
            RvOpcode::BEQ
                if is_creg(rs1) && rs2_hw == 0 && fits_imm_bits(imm, 9) && imm % 2 == 0 =>
            {
                let rs1_c = creg_encode(rs1);
                Some(make_compressed(encode_cb_branch(0b110, rs1_c, imm)))
            }
            RvOpcode::BNE
                if is_creg(rs1) && rs2_hw == 0 && fits_imm_bits(imm, 9) && imm % 2 == 0 =>
            {
                let rs1_c = creg_encode(rs1);
                Some(make_compressed(encode_cb_branch(0b111, rs1_c, imm)))
            }

            // C.J: unconditional jump, 12-bit signed even offset
            RvOpcode::J if fits_imm_bits(imm, 12) && imm % 2 == 0 => {
                Some(make_compressed(encode_cj(imm)))
            }

            // C.SUB, C.XOR, C.OR, C.AND (CA-format, both x8–x15)
            RvOpcode::SUB if is_creg(rd) && rd == rs1 && is_creg(rs2) => Some(make_compressed(
                encode_ca(0b00, creg_encode(rd), creg_encode(rs2)),
            )),
            RvOpcode::XOR if is_creg(rd) && rd == rs1 && is_creg(rs2) => Some(make_compressed(
                encode_ca(0b01, creg_encode(rd), creg_encode(rs2)),
            )),
            RvOpcode::OR if is_creg(rd) && rd == rs1 && is_creg(rs2) => Some(make_compressed(
                encode_ca(0b10, creg_encode(rd), creg_encode(rs2)),
            )),
            RvOpcode::AND if is_creg(rd) && rd == rs1 && is_creg(rs2) => Some(make_compressed(
                encode_ca(0b11, creg_encode(rd), creg_encode(rs2)),
            )),

            // C.ADDW: addw rd', rd', rs2' (both x8–x15)
            RvOpcode::ADDW if is_creg(rd) && rd == rs1 && is_creg(rs2) => {
                let rd_c = creg_encode(rd);
                let rs2_c = creg_encode(rs2);
                // [15:10]=100111, [9:7]=rd', [6:5]=01, [4:2]=rs2', [1:0]=01
                let half: u16 =
                    (0b100111 << 10) | (rd_c << 7) | (0b01 << 5) | (rs2_c << 2) | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.SUBW: subw rd', rd', rs2' (both x8–x15)
            RvOpcode::SUBW if is_creg(rd) && rd == rs1 && is_creg(rs2) => {
                let rd_c = creg_encode(rd);
                let rs2_c = creg_encode(rs2);
                // [15:10]=100111, [9:7]=rd', [6:5]=00, [4:2]=rs2', [1:0]=01
                let half: u16 = (0b100111 << 10) | (rd_c << 7) | (rs2_c << 2) | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.SRLI: srli rd', rd', shamt (rd' in x8–x15, shamt 1–63)
            RvOpcode::SRLI if is_creg(rd) && rd == rs1 && imm > 0 && imm < 64 => {
                let rd_c = creg_encode(rd);
                let shamt = imm as u16;
                // [15:13]=100, [12]=shamt[5], [11:10]=00, [9:7]=rd', [6:2]=shamt[4:0], [1:0]=01
                let half: u16 = (0b100 << 13)
                    | (((shamt >> 5) & 1) << 12)
                    | (rd_c << 7)
                    | ((shamt & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.SRAI: srai rd', rd', shamt (rd' in x8–x15, shamt 1–63)
            RvOpcode::SRAI if is_creg(rd) && rd == rs1 && imm > 0 && imm < 64 => {
                let rd_c = creg_encode(rd);
                let shamt = imm as u16;
                // [15:13]=100, [12]=shamt[5], [11:10]=01, [9:7]=rd', [6:2]=shamt[4:0], [1:0]=01
                let half: u16 = (0b100 << 13)
                    | (((shamt >> 5) & 1) << 12)
                    | (0b01 << 10)
                    | (rd_c << 7)
                    | ((shamt & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.ANDI: andi rd', rd', imm (rd' in x8–x15, 6-bit signed)
            RvOpcode::ANDI if is_creg(rd) && rd == rs1 && fits_imm_bits(imm, 6) => {
                let rd_c = creg_encode(rd);
                let imm6 = (imm as u16) & 0x3F;
                // [15:13]=100, [12]=imm[5], [11:10]=10, [9:7]=rd', [6:2]=imm[4:0], [1:0]=01
                let half: u16 = (0b100 << 13)
                    | (((imm6 >> 5) & 1) << 12)
                    | (0b10 << 10)
                    | (rd_c << 7)
                    | ((imm6 & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // C.LUI: lui rd, nzimm (rd != 0, rd != 2, nzimm != 0, 6-bit signed upper)
            RvOpcode::LUI if rd_hw != 0 && rd_hw != 2 && imm != 0 => {
                // C.LUI encodes nzimm[17:12]. The immediate from LUI is in bits [31:12].
                // For C.LUI we need bits [17:12] to fit in 6-bit signed field.
                let nzimm = ((imm as i32) >> 12) & 0x3F;
                if nzimm != 0 || (((imm as i32) >> 12) & 0x20) != 0 {
                    let nzimm_u = (nzimm as u16) & 0x3F;
                    // [15:13]=011, [12]=nzimm[17], [11:7]=rd, [6:2]=nzimm[16:12], [1:0]=01
                    let half: u16 = (0b011 << 13)
                        | (((nzimm_u >> 5) & 1) << 12)
                        | ((rd_hw as u16) << 7)
                        | ((nzimm_u & 0x1F) << 2)
                        | C_OP_Q1;
                    Some(make_compressed(half))
                } else {
                    None
                }
            }

            // C.ADDIW: addiw rd, rd, imm (rd != 0, 6-bit signed)
            RvOpcode::ADDIW if rd_hw != 0 && rd_hw == rs1_hw && fits_imm_bits(imm, 6) => {
                let imm6 = (imm as u16) & 0x3F;
                // [15:13]=001, [12]=imm[5], [11:7]=rd, [6:2]=imm[4:0], [1:0]=01
                let half: u16 = (0b001 << 13)
                    | (((imm6 >> 5) & 1) << 12)
                    | ((rd_hw as u16) << 7)
                    | ((imm6 & 0x1F) << 2)
                    | C_OP_Q1;
                Some(make_compressed(half))
            }

            // Instruction not compressible
            _ => None,
        }
    }

    // ===================================================================
    // Private helpers for pseudo-instruction encoding
    // ===================================================================

    /// Encode LI (Load Immediate) pseudo-instruction.
    ///
    /// For small immediates (fits 12-bit signed): ADDI rd, x0, imm.
    /// For 32-bit immediates: LUI rd, upper + ADDI rd, rd, lower.
    /// For symbol references: LUI + ADDI with Hi20/Lo12I relocations.
    fn encode_li(
        &self,
        rd: u8,
        imm: i64,
        symbol: &Option<String>,
    ) -> Result<EncodedInstruction, String> {
        if symbol.is_some() {
            // LUI rd, %hi(symbol) + ADDI rd, rd, %lo(symbol)
            let lui_word = encode_u_type(OP_LUI, rd, 0);
            let addi_word = encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, 0);
            Ok(EncodedInstruction {
                bytes: lui_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: Some(EncoderRelocation {
                    offset: 0,
                    reloc_type: RiscV64RelocationType::Hi20.as_elf_type(),
                    addend: imm,
                }),
                continuation: Some(Box::new(EncodedInstruction {
                    bytes: addi_word.to_le_bytes().to_vec(),
                    size: 4,
                    relocation: Some(EncoderRelocation {
                        offset: 0,
                        reloc_type: RiscV64RelocationType::Lo12I.as_elf_type(),
                        addend: imm,
                    }),
                    continuation: None,
                })),
            })
        } else if fits_imm_bits(imm, 12) {
            // Small immediate: ADDI rd, x0, imm
            Ok(make_std(encode_i_type(
                OP_OP_IMM,
                rd,
                FUNCT3_ADDI,
                0,
                imm as i32,
            )))
        } else {
            // 32-bit immediate: LUI + ADDI with sign-extension compensation
            let (hi, lo) = split_i32_lui_addi(imm);
            let lui_word = encode_u_type(OP_LUI, rd, (hi << 12) as i32);
            let addi_word = encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, lo as i32);
            Ok(EncodedInstruction {
                bytes: lui_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: None,
                continuation: Some(Box::new(make_std(addi_word))),
            })
        }
    }

    /// Encode LA (Load Address) pseudo-instruction.
    ///
    /// PC-relative: AUIPC rd, %pcrel_hi(symbol) + ADDI rd, rd, %pcrel_lo(symbol)
    fn encode_la(
        &self,
        rd: u8,
        imm: i64,
        symbol: &Option<String>,
    ) -> Result<EncodedInstruction, String> {
        if symbol.is_some() {
            let auipc_word = encode_u_type(OP_AUIPC, rd, 0);
            let addi_word = encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, 0);
            Ok(EncodedInstruction {
                bytes: auipc_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: Some(EncoderRelocation {
                    offset: 0,
                    reloc_type: RiscV64RelocationType::PcrelHi20.as_elf_type(),
                    addend: imm,
                }),
                continuation: Some(Box::new(EncodedInstruction {
                    bytes: addi_word.to_le_bytes().to_vec(),
                    size: 4,
                    relocation: Some(EncoderRelocation {
                        offset: 0,
                        reloc_type: RiscV64RelocationType::PcrelLo12I.as_elf_type(),
                        addend: 0,
                    }),
                    continuation: None,
                })),
            })
        } else {
            // No symbol — PC-relative offset encoded directly
            let (hi, lo) = split_i32_lui_addi(imm);
            let auipc_word = encode_u_type(OP_AUIPC, rd, (hi << 12) as i32);
            let addi_word = encode_i_type(OP_OP_IMM, rd, FUNCT3_ADDI, rd, lo as i32);
            Ok(EncodedInstruction {
                bytes: auipc_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: None,
                continuation: Some(Box::new(make_std(addi_word))),
            })
        }
    }

    /// Encode CALL pseudo-instruction.
    ///
    /// CALL symbol → AUIPC ra, 0 + JALR ra, ra, 0 with R_RISCV_CALL_PLT relocation.
    fn encode_call(&self, symbol: &Option<String>, imm: i64) -> Result<EncodedInstruction, String> {
        let ra_hw = registers::hw_encoding(registers::RA);

        if symbol.is_some() {
            let auipc_word = encode_u_type(OP_AUIPC, ra_hw, 0);
            let jalr_word = encode_i_type(OP_JALR, ra_hw, 0, ra_hw, 0);
            Ok(EncodedInstruction {
                bytes: auipc_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: Some(EncoderRelocation {
                    offset: 0,
                    reloc_type: RiscV64RelocationType::CallPlt.as_elf_type(),
                    addend: imm,
                }),
                continuation: Some(Box::new(make_std(jalr_word))),
            })
        } else {
            // Near call with known offset — still AUIPC + JALR
            let (hi, lo) = split_i32_lui_addi(imm);
            let auipc_word = encode_u_type(OP_AUIPC, ra_hw, (hi << 12) as i32);
            let jalr_word = encode_i_type(OP_JALR, ra_hw, 0, ra_hw, lo as i32);
            Ok(EncodedInstruction {
                bytes: auipc_word.to_le_bytes().to_vec(),
                size: 4,
                relocation: None,
                continuation: Some(Box::new(make_std(jalr_word))),
            })
        }
    }

    /// Encode a branch instruction, possibly with a relocation.
    fn encode_branch(
        &self,
        funct3: u32,
        rs1: u8,
        rs2: u8,
        imm: i64,
        symbol: &Option<String>,
    ) -> Result<EncodedInstruction, String> {
        if symbol.is_some() {
            let word = encode_b_type(OP_BRANCH, funct3, rs1, rs2, 0);
            Ok(EncodedInstruction {
                bytes: word.to_le_bytes().to_vec(),
                size: 4,
                relocation: Some(EncoderRelocation {
                    offset: 0,
                    reloc_type: RiscV64RelocationType::Branch.as_elf_type(),
                    addend: imm,
                }),
                continuation: None,
            })
        } else {
            Ok(make_std(encode_b_type(
                OP_BRANCH, funct3, rs1, rs2, imm as i32,
            )))
        }
    }
}

// ===========================================================================
// Instruction Format Encoding Functions (module-level)
// ===========================================================================

/// Encode R-type instruction.
///
/// Layout: `funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]`
#[inline]
fn encode_r_type(opcode: u32, rd: u8, funct3: u32, rs1: u8, rs2: u8, funct7: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// Encode I-type instruction.
///
/// Layout: `imm[11:0][31:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]`
///
/// The immediate is a 12-bit sign-extended value. We mask to 12 bits before shifting.
#[inline]
fn encode_i_type(opcode: u32, rd: u8, funct3: u32, rs1: u8, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// Encode S-type instruction.
///
/// Layout: `imm[11:5][31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:0][11:7] | opcode[6:0]`
///
/// The 12-bit immediate is split: upper 7 bits at [31:25], lower 5 bits at [11:7].
#[inline]
fn encode_s_type(opcode: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm_u = (imm as u32) & 0xFFF;
    ((imm_u >> 5) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((imm_u & 0x1F) << 7)
        | opcode
}

/// Encode B-type instruction.
///
/// Layout: `imm[12|10:5][31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:1|11][11:7] | opcode[6:0]`
///
/// The 13-bit signed even immediate is scrambled across the instruction:
/// - bit 31 ← imm[12]
/// - bits 30:25 ← imm[10:5]
/// - bits 11:8 ← imm[4:1]
/// - bit 7 ← imm[11]
#[inline]
fn encode_b_type(opcode: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit12 = (imm_u >> 12) & 1;
    let bit11 = (imm_u >> 11) & 1;
    let bits10_5 = (imm_u >> 5) & 0x3F;
    let bits4_1 = (imm_u >> 1) & 0xF;

    (bit12 << 31)
        | (bits10_5 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | (bits4_1 << 8)
        | (bit11 << 7)
        | opcode
}

/// Encode U-type instruction.
///
/// Layout: `imm[31:12] | rd[11:7] | opcode[6:0]`
///
/// The upper 20 bits of the immediate are placed directly at bits [31:12].
#[inline]
fn encode_u_type(opcode: u32, rd: u8, imm: i32) -> u32 {
    ((imm as u32) & 0xFFFFF000) | ((rd as u32) << 7) | opcode
}

/// Encode J-type instruction.
///
/// Layout: `imm[20|10:1|11|19:12][31:12] | rd[11:7] | opcode[6:0]`
///
/// The 21-bit signed even immediate is scrambled:
/// - bit 31 ← imm[20]
/// - bits 30:21 ← imm[10:1]
/// - bit 20 ← imm[11]
/// - bits 19:12 ← imm[19:12]
#[inline]
fn encode_j_type(opcode: u32, rd: u8, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit20 = (imm_u >> 20) & 1;
    let bits10_1 = (imm_u >> 1) & 0x3FF;
    let bit11 = (imm_u >> 11) & 1;
    let bits19_12 = (imm_u >> 12) & 0xFF;

    (bit20 << 31)
        | (bits10_1 << 21)
        | (bit11 << 20)
        | (bits19_12 << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// Encode R4-type instruction (fused multiply-add).
///
/// Layout: `rs3[31:27] | fmt[26:25] | rs2[24:20] | rs1[19:15] | rm[14:12] | rd[11:7] | opcode[6:0]`
#[inline]
fn encode_r4_type(opcode: u32, rd: u8, rm: u32, rs1: u8, rs2: u8, rs3: u8, fmt: u32) -> u32 {
    ((rs3 as u32) << 27)
        | (fmt << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (rm << 12)
        | ((rd as u32) << 7)
        | opcode
}

/// Encode an atomic (AMO) instruction.
///
/// Atomic instructions use R-type format with funct5 (bits 31:27) selecting
/// the atomic operation, and bits 26:25 for aq/rl ordering.
#[inline]
fn encode_amo(funct5: u32, aq: u32, rl: u32, funct3: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct5 << 27)
        | ((aq & 1) << 26)
        | ((rl & 1) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | OP_AMO
}

// ===========================================================================
// Immediate Validation Helpers
// ===========================================================================

/// Check if a value fits in a signed N-bit immediate.
#[inline]
fn fits_imm_bits(value: i64, bits: u8) -> bool {
    if bits == 0 || bits > 63 {
        return false;
    }
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    value >= min && value <= max
}

/// Decompose a 32-bit value into LUI upper 20 bits and ADDI lower 12 bits,
/// handling the sign-extension gotcha: if bit 11 of the lower portion is set,
/// the ADDI will sign-extend and effectively subtract 0x1000, so we compensate
/// by adding 1 to the upper portion.
fn split_i32_lui_addi(val: i64) -> (i64, i64) {
    let val32 = val as i32;
    let lo12 = ((val32 as i64) << 52 >> 52) as i32; // sign-extend lower 12 bits
    let hi20 = ((val32 as u32).wrapping_sub(lo12 as u32) >> 12) as i32;
    let hi20_masked = hi20 & 0xFFFFF_i32;
    (hi20_masked as i64, lo12 as i64)
}

// ===========================================================================
// Construction Helpers
// ===========================================================================

/// Create a standard (32-bit, no relocation) encoded instruction.
#[inline]
fn make_std(word: u32) -> EncodedInstruction {
    EncodedInstruction {
        bytes: word.to_le_bytes().to_vec(),
        size: 4,
        relocation: None,
        continuation: None,
    }
}

/// Create a compressed (16-bit, no relocation) encoded instruction.
#[inline]
fn make_compressed(half: u16) -> EncodedInstruction {
    EncodedInstruction {
        bytes: half.to_le_bytes().to_vec(),
        size: 2,
        relocation: None,
        continuation: None,
    }
}

// ===========================================================================
// Compressed Instruction Helpers
// ===========================================================================

/// Check if a register ID is in the compressed register set (x8–x15).
///
/// Compressed instructions use a 3-bit register field that can only
/// address registers x8 through x15 (s0, s1, a0–a5).
#[inline]
fn is_creg(reg: u16) -> bool {
    let hw = registers::hw_encoding(reg);
    (8..=15).contains(&hw)
}

/// Encode a register ID into the 3-bit compressed register encoding.
///
/// Maps x8→0, x9→1, ..., x15→7.
///
/// # Panics
///
/// Debug-panics if the register is not in the compressed set.
#[inline]
fn creg_encode(reg: u16) -> u16 {
    let hw = registers::hw_encoding(reg);
    debug_assert!(
        (8..=15).contains(&hw),
        "creg_encode: register {} not in x8–x15",
        hw
    );
    (hw - 8) as u16
}

/// Encode a CB-format branch (C.BEQZ / C.BNEZ).
///
/// Layout: `funct3[15:13] | offset[8|4:3][12:10] | rs1'[9:7] | offset[7:6|2:1|5][6:2] | op[1:0]=01`
fn encode_cb_branch(funct3: u16, rs1_c: u16, imm: i64) -> u16 {
    let off = (imm as u16) & 0x1FF; // 9-bit
                                    // Bit layout in CB:
                                    // [12]=off[8], [11:10]=off[4:3], [9:7]=rs1', [6:5]=off[7:6], [4:3]=off[2:1], [2]=off[5], [1:0]=01
    (funct3 << 13)
        | (((off >> 8) & 1) << 12)
        | (((off >> 3) & 0x3) << 10)
        | (rs1_c << 7)
        | (((off >> 6) & 0x3) << 5)
        | (((off >> 1) & 0x3) << 3)
        | (((off >> 5) & 1) << 2)
        | C_OP_Q1
}

/// Encode a CJ-format jump (C.J).
///
/// Layout: `funct3=101[15:13] | jump target[12:2] | op[1:0]=01`
///
/// The 12-bit signed even offset is scrambled in the instruction:
/// [12]=off[11], [11]=off[4], [10:9]=off[9:8], [8]=off[10],
/// [7]=off[6], [6]=off[7], [5:3]=off[3:1], [2]=off[5]
fn encode_cj(imm: i64) -> u16 {
    let off = (imm as u16) & 0xFFF;
    let bit11 = (off >> 11) & 1;
    let bit4 = (off >> 4) & 1;
    let bits9_8 = (off >> 8) & 0x3;
    let bit10 = (off >> 10) & 1;
    let bit6 = (off >> 6) & 1;
    let bit7 = (off >> 7) & 1;
    let bits3_1 = (off >> 1) & 0x7;
    let bit5 = (off >> 5) & 1;

    (0b101 << 13)
        | (bit11 << 12)
        | (bit4 << 11)
        | (bits9_8 << 9)
        | (bit10 << 8)
        | (bit6 << 7)
        | (bit7 << 6)
        | (bits3_1 << 3)
        | (bit5 << 2)
        | C_OP_Q1
}

/// Encode a CA-format ALU operation (C.SUB, C.XOR, C.OR, C.AND).
///
/// [15:10]=100011, [9:7]=rd'/rs1', [6:5]=funct2, [4:2]=rs2', [1:0]=01
fn encode_ca(funct2: u16, rd_c: u16, rs2_c: u16) -> u16 {
    (0b100011 << 10) | (rd_c << 7) | (funct2 << 5) | (rs2_c << 2) | C_OP_Q1
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::riscv64::codegen::{RvInstruction, RvOpcode};
    use crate::backend::riscv64::registers;

    /// Helper to create a minimal instruction for testing.
    fn test_inst(
        opcode: RvOpcode,
        rd: Option<u16>,
        rs1: Option<u16>,
        rs2: Option<u16>,
        imm: i64,
    ) -> RvInstruction {
        RvInstruction {
            opcode,
            rd,
            rs1,
            rs2,
            rs3: None,
            imm,
            symbol: None,
            is_fp: false,
            comment: None,
            is_call_arg_setup: false,
        }
    }

    fn inst_to_u32(enc: &EncodedInstruction) -> u32 {
        u32::from_le_bytes([enc.bytes[0], enc.bytes[1], enc.bytes[2], enc.bytes[3]])
    }

    #[test]
    fn test_nop_encoding() {
        // NOP → ADDI x0, x0, 0 = 0x00000013
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::NOP, None, None, None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x00000013);
    }

    #[test]
    fn test_ret_encoding() {
        // RET → JALR x0, ra, 0 = 0x00008067
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::RET, None, None, None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x00008067);
    }

    #[test]
    fn test_add_r_type() {
        // ADD x1, x2, x3 → expected: 0x003100B3
        // funct7=0000000, rs2=3, rs1=2, funct3=000, rd=1, opcode=0110011
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::ADD, Some(1), Some(2), Some(3), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x003100B3);
    }

    #[test]
    fn test_addi_i_type() {
        // ADDI x1, x2, 42
        // imm=42=0x02A, rs1=2, funct3=000, rd=1, opcode=0010011
        // Expected: (42 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | 0x13
        //         = 0x02A10093
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::ADDI, Some(1), Some(2), None, 42);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x02A10093);
    }

    #[test]
    fn test_sw_s_type() {
        // SW x1, 8(x2)
        // In the codegen representation: rs1=base(x2), rs2=src(x1), imm=8
        // S-type: imm[11:5]=0, imm[4:0]=01000(=8)
        // Expected: imm[11:5]=0b0000000, rs2=1, rs1=2, funct3=010, imm[4:0]=0b01000, opcode=0100011
        // = 0x00112423
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SW, None, Some(2), Some(1), 8);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x00112423);
    }

    #[test]
    fn test_lui_u_type() {
        // LUI x1, 0x12345 → upper 20 bits = 0x12345
        // The codegen (split_i32_lui_addi) produces the RAW 20-bit
        // upper value, NOT pre-shifted.  The encoder shifts it left
        // by 12 to place it at bits [31:12] for the hardware encoding.
        // Expected: 0x12345000 | (1 << 7) | 0x37 = 0x123450B7
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::LUI, Some(1), None, None, 0x12345);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x123450B7);
    }

    #[test]
    fn test_beq_b_type() {
        // BEQ x1, x2, 8
        // imm=8=0b1000: bit12=0, bits10:5=000000, bits4:1=0100, bit11=0
        // Expected: (0<<31)|(0<<25)|(2<<20)|(1<<15)|(0<<12)|(4<<8)|(0<<7)|0x63
        //         = 0x00208463
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::BEQ, None, Some(1), Some(2), 8);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert_eq!(inst_to_u32(&enc), 0x00208463);
    }

    #[test]
    fn test_jal_j_type() {
        // JAL x1, 0 (offset=0) — useful for placeholder
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::JAL, Some(1), None, None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // JAL x1, 0: opcode=0x6F, rd=1
        // Expected: (0 << 12) | (1 << 7) | 0x6F = 0x000000EF
        assert_eq!(inst_to_u32(&enc), 0x000000EF);
    }

    #[test]
    fn test_mv_pseudo() {
        // MV x10, x11 → ADDI x10, x11, 0
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::MV, Some(10), Some(11), None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // ADDI x10, x11, 0 = (0 << 20) | (11 << 15) | (0 << 12) | (10 << 7) | 0x13
        // = 0x00058513
        assert_eq!(inst_to_u32(&enc), 0x00058513);
    }

    #[test]
    fn test_neg_pseudo() {
        // NEG x10, x11 → SUB x10, x0, x11
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::NEG, Some(10), None, Some(11), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // SUB x10, x0, x11: funct7=0b0100000, rs2=11, rs1=0, funct3=000, rd=10, opcode=0110011
        // = (0x20 << 25) | (11 << 20) | (0 << 15) | (0 << 12) | (10 << 7) | 0x33
        // = 0x40B00533
        assert_eq!(inst_to_u32(&enc), 0x40B00533);
    }

    #[test]
    fn test_not_pseudo() {
        // NOT x10, x11 → XORI x10, x11, -1
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::NOT, Some(10), Some(11), None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // XORI x10, x11, -1: imm=0xFFF(-1 in 12-bit), rs1=11, funct3=100, rd=10, opcode=0010011
        // = (0xFFF << 20) | (11 << 15) | (4 << 12) | (10 << 7) | 0x13
        // = 0xFFF5C513
        assert_eq!(inst_to_u32(&enc), 0xFFF5C513);
    }

    #[test]
    fn test_li_small() {
        // LI x10, 42 → ADDI x10, x0, 42
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::LI, Some(10), None, None, 42);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert!(enc.continuation.is_none());
        // ADDI x10, x0, 42 = (42 << 20) | (0 << 15) | (0 << 12) | (10 << 7) | 0x13
        // = 0x02A00513
        assert_eq!(inst_to_u32(&enc), 0x02A00513);
    }

    #[test]
    fn test_li_large() {
        // LI x10, 0x12345 → LUI x10, hi + ADDI x10, x10, lo
        // 0x12345 = LUI 0x12 + ADDI 0x345
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::LI, Some(10), None, None, 0x12345);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert!(enc.continuation.is_some());
    }

    #[test]
    fn test_call_with_symbol() {
        // CALL with symbol → AUIPC ra, 0 + JALR ra, ra, 0 with relocation
        let encoder = RiscV64Encoder::new();
        let mut inst = test_inst(RvOpcode::CALL, None, None, None, 0);
        inst.symbol = Some("my_func".to_string());
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        assert!(enc.relocation.is_some());
        let reloc = enc.relocation.as_ref().unwrap();
        assert_eq!(
            reloc.reloc_type,
            RiscV64RelocationType::CallPlt.as_elf_type()
        );
        assert!(enc.continuation.is_some());
    }

    #[test]
    fn test_compressed_nop() {
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::NOP, None, None, None, 0);
        let comp = encoder.try_compress(&inst);
        assert!(comp.is_some());
        let enc = comp.unwrap();
        assert_eq!(enc.size, 2);
        // C.NOP = 0x0001
        let half = u16::from_le_bytes([enc.bytes[0], enc.bytes[1]]);
        assert_eq!(half, 0x0001);
    }

    #[test]
    fn test_compressed_ret() {
        // C.RET → C.JR ra
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::RET, None, None, None, 0);
        let comp = encoder.try_compress(&inst);
        assert!(comp.is_some());
        let enc = comp.unwrap();
        assert_eq!(enc.size, 2);
        // C.JR ra: [15:12]=1000, [11:7]=00001 (ra=1), [6:2]=00000, [1:0]=10
        // = 0x8082
        let half = u16::from_le_bytes([enc.bytes[0], enc.bytes[1]]);
        assert_eq!(half, 0x8082);
    }

    #[test]
    fn test_mul_r_type() {
        // MUL x10, x11, x12
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::MUL, Some(10), Some(11), Some(12), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // funct7=0000001, rs2=12, rs1=11, funct3=000, rd=10, opcode=0110011
        // = (1 << 25) | (12 << 20) | (11 << 15) | (0 << 12) | (10 << 7) | 0x33
        // = 0x02C58533
        assert_eq!(inst_to_u32(&enc), 0x02C58533);
    }

    #[test]
    fn test_sub_r_type() {
        // SUB x5, x6, x7
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SUB, Some(5), Some(6), Some(7), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // funct7=0100000, rs2=7, rs1=6, funct3=000, rd=5, opcode=0110011
        // = (0x20 << 25) | (7 << 20) | (6 << 15) | (0 << 12) | (5 << 7) | 0x33
        // = 0x407302B3
        assert_eq!(inst_to_u32(&enc), 0x407302B3);
    }

    #[test]
    fn test_ld_i_type() {
        // LD x10, 16(x2)
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::LD, Some(10), Some(2), None, 16);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // imm=16, rs1=2, funct3=011, rd=10, opcode=0000011
        // = (16 << 20) | (2 << 15) | (3 << 12) | (10 << 7) | 0x03
        // = 0x01013503
        assert_eq!(inst_to_u32(&enc), 0x01013503);
    }

    #[test]
    fn test_sd_s_type() {
        // SD x10, 16(x2)
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SD, None, Some(2), Some(10), 16);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // imm=16=0b10000: imm[11:5]=0, imm[4:0]=10000
        // rs2=10, rs1=2, funct3=011, opcode=0100011
        // = (0 << 25) | (10 << 20) | (2 << 15) | (3 << 12) | (16 << 7) | 0x23
        // = 0x00A13823
        assert_eq!(inst_to_u32(&enc), 0x00A13823);
    }

    #[test]
    fn test_little_endian() {
        // Verify output is little-endian
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::NOP, None, None, None, 0);
        let enc = encoder.encode(&inst).unwrap();
        // NOP = 0x00000013 → bytes: [0x13, 0x00, 0x00, 0x00]
        assert_eq!(enc.bytes, vec![0x13, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_register_hw_encoding() {
        // Verify hw_encoding produces correct 5-bit values
        assert_eq!(registers::hw_encoding(registers::ZERO), 0);
        assert_eq!(registers::hw_encoding(registers::RA), 1);
        assert_eq!(registers::hw_encoding(registers::SP), 2);
        assert_eq!(registers::hw_encoding(31), 31);
        // FP register: F0 (ID 32) → hw 0
        assert_eq!(registers::hw_encoding(32), 0);
        assert_eq!(registers::hw_encoding(63), 31);
    }

    #[test]
    fn test_seqz_pseudo() {
        // SEQZ x10, x11 → SLTIU x10, x11, 1
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SEQZ, Some(10), Some(11), None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // SLTIU x10, x11, 1
        // imm=1, rs1=11, funct3=011, rd=10, opcode=0010011
        // = (1<<20) | (11<<15) | (3<<12) | (10<<7) | 0x13
        // = 0x0015B513
        assert_eq!(inst_to_u32(&enc), 0x0015B513);
    }

    #[test]
    fn test_snez_pseudo() {
        // SNEZ x10, x11 → SLTU x10, x0, x11
        // NOTE: SNEZ uses rs2 (the source register in codegen representation)
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SNEZ, Some(10), None, Some(11), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // SLTU x10, x0, x11
        // funct7=0, rs2=11, rs1=0, funct3=011, rd=10, opcode=0110011
        // = (0<<25)|(11<<20)|(0<<15)|(3<<12)|(10<<7)|0x33
        // = 0x00B03533
        assert_eq!(inst_to_u32(&enc), 0x00B03533);
    }

    #[test]
    fn test_fits_imm_bits() {
        assert!(fits_imm_bits(0, 12));
        assert!(fits_imm_bits(2047, 12));
        assert!(fits_imm_bits(-2048, 12));
        assert!(!fits_imm_bits(2048, 12));
        assert!(!fits_imm_bits(-2049, 12));
        assert!(fits_imm_bits(31, 6));
        assert!(fits_imm_bits(-32, 6));
        assert!(!fits_imm_bits(32, 6));
    }

    #[test]
    fn test_compressed_add() {
        // C.ADD x10, x10, x11
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::ADD, Some(10), Some(10), Some(11), 0);
        let comp = encoder.try_compress(&inst);
        assert!(comp.is_some());
        let enc = comp.unwrap();
        assert_eq!(enc.size, 2);
    }

    #[test]
    fn test_slli_encoding() {
        // SLLI x10, x10, 3
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::SLLI, Some(10), Some(10), None, 3);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // imm=3, rs1=10, funct3=001, rd=10, opcode=0010011
        // = (3<<20)|(10<<15)|(1<<12)|(10<<7)|0x13
        // = 0x00351513
        assert_eq!(inst_to_u32(&enc), 0x00351513);
    }

    #[test]
    fn test_j_pseudo() {
        // J 0 → JAL x0, 0
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::J, None, None, None, 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // JAL x0, 0: rd=0, opcode=0x6F
        // = 0x0000006F
        assert_eq!(inst_to_u32(&enc), 0x0000006F);
    }

    #[test]
    fn test_addw_encoding() {
        // ADDW x5, x6, x7
        let encoder = RiscV64Encoder::new();
        let inst = test_inst(RvOpcode::ADDW, Some(5), Some(6), Some(7), 0);
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // funct7=0, rs2=7, rs1=6, funct3=000, rd=5, opcode=0111011
        // = (0<<25)|(7<<20)|(6<<15)|(0<<12)|(5<<7)|0x3B
        // = 0x007302BB
        assert_eq!(inst_to_u32(&enc), 0x007302BB);
    }

    #[test]
    fn test_fadd_d_encoding() {
        // FADD.D f10, f11, f12 (using FP register IDs 42, 43, 44 → hw 10, 11, 12)
        let encoder = RiscV64Encoder::new();
        let inst = RvInstruction {
            opcode: RvOpcode::FADD_D,
            rd: Some(42),  // f10
            rs1: Some(43), // f11
            rs2: Some(44), // f12
            rs3: None,
            imm: 0,
            symbol: None,
            is_fp: true,
            comment: None,
            is_call_arg_setup: false,
        };
        let enc = encoder.encode(&inst).unwrap();
        assert_eq!(enc.size, 4);
        // funct7=0000001, rs2=12, rs1=11, rm=111(dyn), rd=10, opcode=1010011
        // = (1<<25)|(12<<20)|(11<<15)|(7<<12)|(10<<7)|0x53
        // = 0x02C5F553
        assert_eq!(inst_to_u32(&enc), 0x02C5F553);
    }
}
