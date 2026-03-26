#![allow(clippy::identity_op, clippy::unusual_byte_groupings)]
//! # AArch64 (A64) Instruction Encoder
//!
//! Encodes A64 instructions into fixed 32-bit (4-byte) binary machine code.
//! All instructions are little-endian.
//!
//! ## Instruction Format Groups
//!
//! ### Data Processing — Immediate
//! ADD/SUB with 12-bit immediate (optionally shifted by 12)
//! MOV wide: MOVZ, MOVK, MOVN (16-bit immediate with shift)
//! Logical: AND/ORR/EOR with bitmask immediate (N:immr:imms encoding)
//! PC-relative: ADR (±1 MiB), ADRP (±4 GiB, page-aligned)
//! Bitfield: BFM, SBFM, UBFM
//!
//! ### Data Processing — Register
//! Shifted/extended register operations
//! Conditional select: CSEL, CSINC, CSINV, CSNEG
//! Multiply: MADD, MSUB, SMULH, UMULH
//! Divide: SDIV, UDIV
//! Bit manipulation: CLZ, CLS, RBIT, REV
//!
//! ### Loads and Stores
//! Unsigned offset: LDR/STR \[Xn, #imm\] (scaled by access size)
//! Pre-index: LDR/STR \[Xn, #imm\]!
//! Post-index: LDR/STR \[Xn\], #imm
//! Register offset: LDR/STR \[Xn, Xm{, extend/shift}\]
//! Literal: LDR Xd, label (PC-relative, ±1 MiB)
//! Pair: LDP/STP (load/store pair for efficient stack ops)
//!
//! ### Branches
//! B (unconditional, ±128 MiB, 26-bit signed offset × 4)
//! BL (unconditional with link, ±128 MiB)
//! B.cond (conditional, ±1 MiB, 19-bit signed offset × 4)
//! BR/BLR/RET (register-indirect)
//! CBZ/CBNZ (compare and branch, ±1 MiB)
//! TBZ/TBNZ (test bit and branch, ±32 KiB, 14-bit signed offset × 4)
//!
//! ### SIMD/Floating-Point
//! FADD, FSUB, FMUL, FDIV (single S and double D)
//! FCMP, FCVT, FMOV
//! FMADD, FMSUB (fused multiply-add)
//!
//! ### System
//! NOP, DMB, DSB, ISB, SVC, MRS, MSR
//!
//! ## Encoding Format
//! All 32-bit, little-endian. The `sf` bit (bit 31) selects 64-bit (X) vs 32-bit (W).
//! Register fields: Rd \[4:0\], Rn \[9:5\], Rm \[20:16\], Ra \[14:10\].

use crate::backend::aarch64::codegen::{A64Instruction, A64Opcode};
// CondCode is accessed transitively via A64Instruction.cond field methods
// (.encoding(), .invert()) so no direct import is required.
use super::relocations::AArch64RelocationType;
use crate::backend::aarch64::registers;

// ===========================================================================
// Public Types
// ===========================================================================

/// AArch64 (A64) instruction encoder.
///
/// Encodes [`A64Instruction`] into fixed 32-bit (4-byte) binary machine code.
/// All output is little-endian.
pub struct AArch64Encoder;

/// Result of encoding a single A64 instruction.
///
/// Every AArch64 instruction is exactly 4 bytes (32 bits, fixed-width).
/// If the instruction references an unresolved symbol, a relocation entry
/// is attached for the linker.
pub struct EncodedInstruction {
    /// Encoded bytes — typically exactly 4 for a single AArch64 instruction
    /// (little-endian), but may be longer for multi-instruction expansions
    /// such as `MOV_imm` which emits a MOVZ + up to 3 MOVK instructions
    /// (4–16 bytes).
    pub bytes: Vec<u8>,
    /// Optional relocation if instruction references an unresolved symbol.
    pub relocation: Option<EncoderRelocation>,
}

/// Relocation request emitted by the encoder when an instruction references
/// an unresolved symbol (e.g., a branch target or ADRP page offset).
pub struct EncoderRelocation {
    /// Relocation type — raw ELF `r_type` value from [`AArch64RelocationType`].
    pub reloc_type: u32,
    /// Addend value for the relocation computation.
    pub addend: i64,
}

// ===========================================================================
// Constructor
// ===========================================================================

impl Default for AArch64Encoder {
    fn default() -> Self {
        AArch64Encoder
    }
}

impl AArch64Encoder {
    /// Create a new AArch64 instruction encoder.
    pub fn new() -> Self {
        Self
    }
}

// ===========================================================================
// Internal result helpers
// ===========================================================================

/// Wrap a 32-bit instruction word into an `EncodedInstruction` with no relocation.
#[inline]
fn ok(word: u32) -> Result<EncodedInstruction, String> {
    Ok(EncodedInstruction {
        bytes: word.to_le_bytes().to_vec(),
        relocation: None,
    })
}

/// Wrap a 32-bit instruction word with a relocation into an `EncodedInstruction`.
#[inline]
fn ok_reloc(
    word: u32,
    rtype: AArch64RelocationType,
    addend: i64,
) -> Result<EncodedInstruction, String> {
    Ok(EncodedInstruction {
        bytes: word.to_le_bytes().to_vec(),
        relocation: Some(EncoderRelocation {
            reloc_type: rtype.to_raw(),
            addend,
        }),
    })
}

/// Get the 5-bit hardware register encoding from an `Option<u32>`, defaulting
/// to register 0 if absent.
#[inline]
fn hw(reg: Option<u32>) -> u32 {
    registers::hw_encoding(reg.unwrap_or(0)) as u32
}

/// Get the 5-bit hardware register encoding from an `Option<u32>`, defaulting
/// to XZR (31) if absent. Used for multiply-accumulate Ra field defaults.
#[inline]
fn hw_or_zr(reg: Option<u32>) -> u32 {
    registers::hw_encoding(reg.unwrap_or(registers::XZR)) as u32
}

// ===========================================================================
// Format Encoding Helpers
// ===========================================================================

/// Encode ADD/SUB immediate.
///
/// Format: `sf | op | S | 1 0 0 0 1 0 | sh | imm12 | Rn | Rd`
///
/// - `op`: 0 = ADD, 1 = SUB
/// - `s_flag`: 0 = no flags, 1 = set NZCV flags
/// - `sh`: 0 = no shift, 1 = LSL #12
#[inline]
fn enc_add_sub_imm(sf: u32, op: u32, s_flag: u32, sh: u32, imm12: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s_flag << 29)
        | (0b100010 << 23)
        | (sh << 22)
        | ((imm12 & 0xFFF) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode MOVZ/MOVK/MOVN (move wide immediate).
///
/// Format: `sf | opc | 1 0 0 1 0 1 | hw | imm16 | Rd`
///
/// - `opc`: 0b00 = MOVN, 0b10 = MOVZ, 0b11 = MOVK
/// - `hw`: halfword position (0–3, shift = hw × 16)
#[inline]
fn enc_mov_wide(sf: u32, opc: u32, hw: u32, imm16: u32, rd: u32) -> u32 {
    (sf << 31)
        | ((opc & 0x3) << 29)
        | (0b100101 << 23)
        | ((hw & 0x3) << 21)
        | ((imm16 & 0xFFFF) << 5)
        | (rd & 0x1F)
}

/// Encode ADR or ADRP (PC-relative addressing).
///
/// Format: `op | immlo | 1 0 0 0 0 | immhi | Rd`
///
/// - `op`: 0 = ADR, 1 = ADRP
/// - 21-bit signed offset split: immlo at bits \[30:29\], immhi at bits \[23:5\]
#[inline]
fn enc_pc_rel(op: u32, imm21: i32, rd: u32) -> u32 {
    let imm = imm21 as u32;
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7FFFF;
    (op << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | (rd & 0x1F)
}

/// Encode logical immediate.
///
/// Format: `sf | opc | 1 0 0 1 0 0 | N | immr | imms | Rn | Rd`
///
/// - `opc`: 0b00 = AND, 0b01 = ORR, 0b10 = EOR, 0b11 = ANDS
/// - `n`, `immr`, `imms` from bitmask immediate encoding
#[inline]
fn enc_logical_imm(sf: u32, opc: u32, n: u32, immr: u32, imms: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | ((opc & 0x3) << 29)
        | (0b100100 << 23)
        | ((n & 1) << 22)
        | ((immr & 0x3F) << 16)
        | ((imms & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode bitfield (BFM/SBFM/UBFM).
///
/// Format: `sf | opc | 1 0 0 1 1 0 | N | immr | imms | Rn | Rd`
///
/// - `opc`: 0b00 = SBFM, 0b01 = BFM, 0b10 = UBFM
#[inline]
fn enc_bitfield(sf: u32, opc: u32, n: u32, immr: u32, imms: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | ((opc & 0x3) << 29)
        | (0b100110 << 23)
        | ((n & 1) << 22)
        | ((immr & 0x3F) << 16)
        | ((imms & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode ADD/SUB shifted register.
///
/// Format: `sf | op | S | 0 1 0 1 1 | shift | 0 | Rm | imm6 | Rn | Rd`
///
/// - `shift_type`: 0 = LSL, 1 = LSR, 2 = ASR
#[inline]
fn enc_add_sub_shifted(
    sf: u32,
    op: u32,
    s_flag: u32,
    shift_type: u32,
    rm: u32,
    imm6: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s_flag << 29)
        | (0b01011 << 24)
        | ((shift_type & 0x3) << 22)
        | (0 << 21)
        | ((rm & 0x1F) << 16)
        | ((imm6 & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode logical shifted register.
///
/// Format: `sf | opc | 0 1 0 1 0 | shift | N | Rm | imm6 | Rn | Rd`
///
/// - `opc`: 0b00 = AND, 0b01 = ORR, 0b10 = EOR, 0b11 = ANDS
/// - `invert` (N bit): 1 for BIC/ORN/EON/BICS
#[inline]
fn enc_logical_shifted(
    sf: u32,
    opc: u32,
    shift_type: u32,
    invert: u32,
    rm: u32,
    imm6: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    (sf << 31)
        | ((opc & 0x3) << 29)
        | (0b01010 << 24)
        | ((shift_type & 0x3) << 22)
        | ((invert & 1) << 21)
        | ((rm & 0x1F) << 16)
        | ((imm6 & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode conditional select (CSEL/CSINC/CSINV/CSNEG).
///
/// Format: `sf | op | S | 1 1 0 1 0 1 0 0 | Rm | cond | op2 | Rn | Rd`
///
/// - CSEL:  op=0, S=0, op2=0
/// - CSINC: op=0, S=0, op2=1
/// - CSINV: op=1, S=0, op2=0
/// - CSNEG: op=1, S=0, op2=1
#[inline]
fn enc_cond_select(
    sf: u32,
    op: u32,
    s_flag: u32,
    rm: u32,
    cond: u32,
    op2: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    (sf << 31)
        | (op << 30)
        | (s_flag << 29)
        | (0b11010100 << 21)
        | ((rm & 0x1F) << 16)
        | ((cond & 0xF) << 12)
        | ((op2 & 0x3) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode MADD/MSUB (3-source multiply).
///
/// Format: `sf | 0 0 | 1 1 0 1 1 | 0 0 0 | Rm | o0 | Ra | Rn | Rd`
///
/// - `o0`: 0 = MADD, 1 = MSUB
#[inline]
fn enc_mul(sf: u32, rm: u32, o0: u32, ra: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (0b00_11011_000 << 21)
        | ((rm & 0x1F) << 16)
        | ((o0 & 1) << 15)
        | ((ra & 0x1F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode data-processing (2 source) instructions: UDIV, SDIV, LSLV, LSRV, ASRV, RORV.
///
/// Format: `sf | 0 | 0 | 1 1 0 1 0 1 1 0 | Rm | opcode6 | Rn | Rd`
///
/// Opcode values: UDIV=0b000010, SDIV=0b000011, LSLV=0b001000,
/// LSRV=0b001001, ASRV=0b001010, RORV=0b001011
#[inline]
fn enc_dp_2src(sf: u32, rm: u32, opcode6: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (0b0_0_11010110 << 21)
        | ((rm & 0x1F) << 16)
        | ((opcode6 & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode data-processing (1 source) instructions: CLZ, CLS, RBIT, REV, etc.
///
/// Format: `sf | 1 | 0 | 1 1 0 1 0 1 1 0 | 0 0 0 0 0 | opcode6 | Rn | Rd`
///
/// Opcode values: RBIT=0b000000, REV16=0b000001, REV32=0b000010,
/// REV=0b000010(32-bit)/0b000011(64-bit), CLZ=0b000100, CLS=0b000101
#[inline]
fn enc_dp_1src(sf: u32, opcode6: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (0b1_0_11010110 << 21)
        | (0b00000 << 16)
        | ((opcode6 & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode load/store with unsigned immediate offset.
///
/// Format: `size | 1 1 1 | V | 0 1 | opc | imm12 | Rn | Rt`
///
/// - `size`: 0b00=byte, 0b01=halfword, 0b10=word, 0b11=doubleword
/// - `v`: 1 for SIMD/FP, 0 for GP
/// - `opc`: 0b00=store, 0b01=load, 0b10=load signed (to 64), 0b11=load signed (to 32)
#[inline]
fn enc_ldst_unsigned(size: u32, v: u32, opc: u32, imm12: u32, rn: u32, rt: u32) -> u32 {
    ((size & 0x3) << 30)
        | (0b111 << 27)
        | ((v & 1) << 26)
        | (0b01 << 24)
        | ((opc & 0x3) << 22)
        | ((imm12 & 0xFFF) << 10)
        | ((rn & 0x1F) << 5)
        | (rt & 0x1F)
}

/// Encode load/store with pre-index or post-index addressing.
///
/// Format: `size | 1 1 1 | V | 0 0 | opc | 0 | imm9 | idx | Rn | Rt`
///
/// - `idx`: 0b01 = post-index, 0b11 = pre-index
#[inline]
fn enc_ldst_indexed(size: u32, v: u32, opc: u32, imm9: i32, idx: u32, rn: u32, rt: u32) -> u32 {
    let imm9_bits = (imm9 as u32) & 0x1FF;
    ((size & 0x3) << 30)
        | (0b111 << 27)
        | ((v & 1) << 26)
        | (0b00 << 24)
        | ((opc & 0x3) << 22)
        | (0 << 21)
        | (imm9_bits << 12)
        | ((idx & 0x3) << 10)
        | ((rn & 0x1F) << 5)
        | (rt & 0x1F)
}

/// Encode load/store register (register offset).
///
/// Format: `size | 1 1 1 | V | 0 0 | opc | 1 | Rm | option | S | 1 0 | Rn | Rt`
///
/// - `option`: extend type (0b011 = LSL, 0b010 = UXTW, 0b110 = SXTW, 0b111 = SXTX)
/// - `s_bit`: 1 = shift by access size, 0 = no shift
#[inline]
fn enc_ldst_reg(
    size: u32,
    v: u32,
    opc: u32,
    rm: u32,
    option: u32,
    s_bit: u32,
    rn: u32,
    rt: u32,
) -> u32 {
    ((size & 0x3) << 30)
        | (0b111 << 27)
        | ((v & 1) << 26)
        | (0b00 << 24)
        | ((opc & 0x3) << 22)
        | (1 << 21)
        | ((rm & 0x1F) << 16)
        | ((option & 0x7) << 13)
        | ((s_bit & 1) << 12)
        | (0b10 << 10)
        | ((rn & 0x1F) << 5)
        | (rt & 0x1F)
}

/// Encode load/store pair (LDP/STP) with signed offset.
///
/// Format: `opc | 1 0 1 | V | 0 | idx | L | imm7 | Rt2 | Rn | Rt`
///
/// - `idx`: 0b01=post-index, 0b10=signed-offset, 0b11=pre-index
/// - `l`: 0=store (STP), 1=load (LDP)
#[inline]
fn enc_ldst_pair(opc: u32, v: u32, idx: u32, l: u32, imm7: i32, rt2: u32, rn: u32, rt: u32) -> u32 {
    let imm7_bits = (imm7 as u32) & 0x7F;
    ((opc & 0x3) << 30)
        | (0b101 << 27)
        | ((v & 1) << 26)
        | (0 << 25)
        | ((idx & 0x3) << 23)
        | ((l & 1) << 22)
        | (imm7_bits << 15)
        | ((rt2 & 0x1F) << 10)
        | ((rn & 0x1F) << 5)
        | (rt & 0x1F)
}

/// Encode load register (PC-relative literal).
///
/// Format: `opc | 0 1 1 | V | 0 0 | imm19 | Rt`
#[inline]
fn enc_ldr_literal(opc: u32, v: u32, imm19: i32, rt: u32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x7FFFF;
    ((opc & 0x3) << 30)
        | (0b011 << 27)
        | ((v & 1) << 26)
        | (0b00 << 24)
        | (imm19_bits << 5)
        | (rt & 0x1F)
}

/// Encode B or BL (unconditional branch).
///
/// Format: `op | 0 0 1 0 1 | imm26`
///
/// - `op`: 0 = B, 1 = BL
#[inline]
fn enc_b(op: u32, imm26: i32) -> u32 {
    let imm26_bits = (imm26 as u32) & 0x3FFFFFF;
    ((op & 1) << 31) | (0b00101 << 26) | imm26_bits
}

/// Encode B.cond (conditional branch).
///
/// Format: `0 1 0 1 0 1 0 0 | imm19 | 0 | cond`
#[inline]
fn enc_bcond(imm19: i32, cond: u32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x7FFFF;
    (0b01010100 << 24) | (imm19_bits << 5) | (0 << 4) | (cond & 0xF)
}

/// Encode BR/BLR/RET (unconditional branch to register).
///
/// Format: `1 1 0 1 0 1 1 | opc4 | 1 1 1 1 1 | 0 0 0 0 0 0 | Rn | 0 0 0 0 0`
///
/// - `opc4`: 0b0000 = BR, 0b0001 = BLR, 0b0010 = RET
#[inline]
fn enc_br_reg(opc4: u32, rn: u32) -> u32 {
    (0b1101011 << 25)
        | ((opc4 & 0xF) << 21)
        | (0b11111 << 16)
        | (0b000000 << 10)
        | ((rn & 0x1F) << 5)
        | 0b00000
}

/// Encode CBZ/CBNZ (compare and branch).
///
/// Format: `sf | 0 1 1 0 1 0 | op | imm19 | Rt`
///
/// - `op`: 0 = CBZ, 1 = CBNZ
#[inline]
fn enc_cbz(sf: u32, op: u32, imm19: i32, rt: u32) -> u32 {
    let imm19_bits = (imm19 as u32) & 0x7FFFF;
    (sf << 31) | (0b011010 << 25) | ((op & 1) << 24) | (imm19_bits << 5) | (rt & 0x1F)
}

/// Encode TBZ/TBNZ (test bit and branch).
///
/// Format: `b5 | 0 1 1 0 1 1 | op | b40 | imm14 | Rt`
///
/// - `bit_num`: the bit to test (0–63); b5 is bit\[5\], b40 is bits\[4:0\]
/// - `op`: 0 = TBZ, 1 = TBNZ
#[inline]
fn enc_tbz(bit_num: u32, op: u32, imm14: i32, rt: u32) -> u32 {
    let b5 = (bit_num >> 5) & 1;
    let b40 = bit_num & 0x1F;
    let imm14_bits = (imm14 as u32) & 0x3FFF;
    (b5 << 31) | (0b011011 << 25) | ((op & 1) << 24) | (b40 << 19) | (imm14_bits << 5) | (rt & 0x1F)
}

/// Encode floating-point data-processing (2 source).
///
/// Format: `M=0 | 0 | S=0 | 1 1 1 1 0 | ftype | 1 | Rm | opcode4 | 1 0 | Rn | Rd`
///
/// - `ftype`: 0b00=single, 0b01=double
/// - `opcode4`: FMUL=0b0000, FDIV=0b0001, FADD=0b0010, FSUB=0b0011,
///   FMAX=0b0100, FMIN=0b0101
#[inline]
fn enc_fp_2src(ftype: u32, rm: u32, opcode4: u32, rn: u32, rd: u32) -> u32 {
    (0b0_0_0_11110 << 24)
        | ((ftype & 0x3) << 22)
        | (1 << 21)
        | ((rm & 0x1F) << 16)
        | ((opcode4 & 0xF) << 12)
        | (0b10 << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode floating-point data-processing (1 source).
///
/// Format: `M=0 | 0 | S=0 | 1 1 1 1 0 | ftype | 1 | opcode6 | 1 0 0 0 0 | Rn | Rd`
///
/// - `opcode6`: FMOV=0b000000, FABS=0b000001, FNEG=0b000010, FSQRT=0b000011,
///   FCVT_to_single=0b000100, FCVT_to_double=0b000101
#[inline]
fn enc_fp_1src(ftype: u32, opcode6: u32, rn: u32, rd: u32) -> u32 {
    (0b0_0_0_11110 << 24)
        | ((ftype & 0x3) << 22)
        | (1 << 21)
        | ((opcode6 & 0x3F) << 15)
        | (0b10000 << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode floating-point comparison.
///
/// Format: `M=0 | 0 | S=0 | 1 1 1 1 0 | ftype | 1 | Rm | 0 0 | 1 0 0 0 | Rn | opcode2`
///
/// - `opcode2`: 0b00000=FCMP, 0b01000=FCMP with 0, 0b10000=FCMPE
#[inline]
fn enc_fp_cmp(ftype: u32, rm: u32, rn: u32, opcode2: u32) -> u32 {
    (0b0_0_0_11110 << 24)
        | ((ftype & 0x3) << 22)
        | (1 << 21)
        | ((rm & 0x1F) << 16)
        | (0b00 << 14)
        | (0b1000 << 10)
        | ((rn & 0x1F) << 5)
        | (opcode2 & 0x1F)
}

/// Encode floating-point <-> integer conversion.
///
/// Format: `sf | 0 | S=0 | 1 1 1 1 0 | ftype | 1 | rmode | opcode3 | 0 0 0 0 0 0 | Rn | Rd`
#[inline]
fn enc_fp_int_cvt(sf: u32, ftype: u32, rmode: u32, opcode3: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (0b0_0_11110 << 24)
        | ((ftype & 0x3) << 22)
        | (1 << 21)
        | ((rmode & 0x3) << 19)
        | ((opcode3 & 0x7) << 16)
        | (0b000000 << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode floating-point fused multiply-add (FMADD/FMSUB/FNMADD/FNMSUB).
///
/// Format: `M=0 | 0 | S=0 | 1 1 1 1 1 | ftype | o1 | Rm | o0 | Ra | Rn | Rd`
///
/// - FMADD:  o1=0, o0=0  (Rd = Ra + Rn*Rm)
/// - FMSUB:  o1=0, o0=1  (Rd = Ra - Rn*Rm)
/// - FNMADD: o1=1, o0=0  (Rd = -Ra - Rn*Rm)
/// - FNMSUB: o1=1, o0=1  (Rd = -Ra + Rn*Rm)
#[inline]
fn enc_fp_fma(ftype: u32, o1: u32, rm: u32, o0: u32, ra: u32, rn: u32, rd: u32) -> u32 {
    (0b0_0_0_11111 << 24)
        | ((ftype & 0x3) << 22)
        | ((o1 & 1) << 21)
        | ((rm & 0x1F) << 16)
        | ((o0 & 1) << 15)
        | ((ra & 0x1F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode SVC (supervisor call).
///
/// Format: `1 1 0 1 0 1 0 0 | 0 0 0 | imm16 | 0 0 0 0 1`
#[inline]
fn enc_svc(imm16: u32) -> u32 {
    0xD400_0001 | ((imm16 & 0xFFFF) << 5)
}

/// Encode HVC (hypervisor call).
#[inline]
fn enc_hvc(imm16: u32) -> u32 {
    0xD400_0002 | ((imm16 & 0xFFFF) << 5)
}

/// Encode SMC (secure monitor call).
#[inline]
fn enc_smc(imm16: u32) -> u32 {
    0xD400_0003 | ((imm16 & 0xFFFF) << 5)
}

/// Encode CCMP immediate (conditional compare immediate).
///
/// Format: `sf | op | 1 | 1 1 0 1 0 0 1 0 | imm5 | cond | 1 0 | Rn | 0 | nzcv`
#[inline]
fn enc_ccmp_imm(sf: u32, op: u32, imm5: u32, cond: u32, rn: u32, nzcv: u32) -> u32 {
    (sf << 31)
        | (op << 30)
        | (1 << 29)
        | (0b11010010 << 21)
        | ((imm5 & 0x1F) << 16)
        | ((cond & 0xF) << 12)
        | (0b10 << 10)
        | ((rn & 0x1F) << 5)
        | (0 << 4)
        | (nzcv & 0xF)
}

/// Encode CCMP register (conditional compare register).
///
/// Format: `sf | op | 1 | 1 1 0 1 0 0 1 0 | Rm | cond | 0 0 | Rn | 0 | nzcv`
#[inline]
fn enc_ccmp_reg(sf: u32, op: u32, rm: u32, cond: u32, rn: u32, nzcv: u32) -> u32 {
    (sf << 31)
        | (op << 30)
        | (1 << 29)
        | (0b11010010 << 21)
        | ((rm & 0x1F) << 16)
        | ((cond & 0xF) << 12)
        | (0b00 << 10)
        | ((rn & 0x1F) << 5)
        | (0 << 4)
        | (nzcv & 0xF)
}

/// Encode EXTR (extract bits from register pair).
///
/// Format: `sf | 0 0 | 1 0 0 1 1 1 | N | 0 | Rm | imms | Rn | Rd`
///
/// - N must equal sf (1 for 64-bit, 0 for 32-bit)
#[inline]
fn enc_extr(sf: u32, rm: u32, imms: u32, rn: u32, rd: u32) -> u32 {
    (sf << 31)
        | (0b00 << 29)
        | (0b100111 << 23)
        | (sf << 22)
        | (0 << 21)
        | ((rm & 0x1F) << 16)
        | ((imms & 0x3F) << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode SMULH (signed multiply high, 64-bit only).
///
/// Format: `1 | 0 0 | 1 1 0 1 1 | 0 1 0 | Rm | 0 | Ra=11111 | Rn | Rd`
#[inline]
fn enc_smulh(rm: u32, rn: u32, rd: u32) -> u32 {
    (1 << 31)
        | (0b00_11011_010 << 21)
        | ((rm & 0x1F) << 16)
        | (0 << 15)
        | (0b11111 << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

/// Encode UMULH (unsigned multiply high, 64-bit only).
///
/// Format: `1 | 0 0 | 1 1 0 1 1 | 1 1 0 | Rm | 0 | Ra=11111 | Rn | Rd`
#[inline]
fn enc_umulh(rm: u32, rn: u32, rd: u32) -> u32 {
    (1 << 31)
        | (0b00_11011_110 << 21)
        | ((rm & 0x1F) << 16)
        | (0 << 15)
        | (0b11111 << 10)
        | ((rn & 0x1F) << 5)
        | (rd & 0x1F)
}

// ===========================================================================
// ADD/SUB Immediate Splitting Helper
// ===========================================================================

/// Split an unsigned value into the imm12 + shift fields for ADD/SUB immediate.
fn split_add_sub_imm(value: u64) -> Result<(u32, u32), String> {
    if value <= 0xFFF {
        Ok((0, value as u32))
    } else if (value & 0xFFF) == 0 && (value >> 12) <= 0xFFF {
        Ok((1, (value >> 12) as u32))
    } else {
        Err(format!(
            "ADD/SUB immediate 0x{:X} does not fit in 12-bit encoding",
            value
        ))
    }
}

// ===========================================================================
// Bitmask Immediate Encoder
// ===========================================================================

/// Encode a bitmask immediate value into the AArch64 N:immr:imms format.
///
/// AArch64 logical immediates use a special encoding for repeating bit patterns.
/// Not all 64-bit values are representable.
///
/// Returns `Some((n, immr, imms))` if encodable; `None` otherwise.
pub fn encode_bitmask_immediate(value: u64, is_64bit: bool) -> Option<(bool, u8, u8)> {
    if value == 0 {
        return None;
    }
    let imm = if is_64bit {
        value
    } else {
        let v32 = value as u32;
        (v32 as u64) | ((v32 as u64) << 32)
    };
    if imm == u64::MAX {
        return None;
    }

    // Try each possible element size: 2, 4, 8, 16, 32, 64.
    for log_e in 1u32..=6 {
        let e = 1u32 << log_e;
        let mask: u64 = if e == 64 { u64::MAX } else { (1u64 << e) - 1 };
        let elem = imm & mask;

        // Verify the entire value is this element repeated.
        let mut valid = true;
        let mut check = imm;
        for _ in 0..(64 / e) {
            if (check & mask) != elem {
                valid = false;
                break;
            }
            if e < 64 {
                check >>= e;
            }
        }
        if !valid {
            continue;
        }
        // All-zeros or all-ones elements are not encodable at this size.
        if elem == 0 || elem == mask {
            continue;
        }

        let ones = elem.count_ones() as u8;

        // Find the trailing zeros to locate the start of the ones run.
        let tz = if e < 64 {
            (elem | (u64::MAX << e)).trailing_zeros().min(e) as u8
        } else {
            elem.trailing_zeros().min(64) as u8
        };

        // Rotate right by tz to align ones at LSB; check contiguity.
        let rotated = if tz == 0 {
            elem
        } else {
            ((elem >> tz as u32) | (elem << (e - tz as u32))) & mask
        };
        let expected = if ones >= 64 {
            u64::MAX
        } else {
            (1u64 << ones) - 1
        };
        if rotated != expected {
            continue;
        }

        // immr = rotation amount from canonical LSB-aligned pattern.
        let immr = if tz == 0 { 0u8 } else { (e as u8) - tz };

        // N and imms encoding.
        let n = e == 64;
        let imms = if e == 64 {
            ones - 1
        } else {
            // High bits encode element size; low bits encode (ones - 1).
            let two_e = (e as u16) * 2;
            let size_mask = (!(two_e.wrapping_sub(1)) as u8) & 0x3F;
            size_mask | (ones - 1)
        };

        return Some((n, immr, imms));
    }
    None
}

// ===========================================================================
// Large Immediate Materialization
// ===========================================================================

/// Generate a MOVZ + MOVK instruction sequence to materialize a full 64-bit
/// immediate value into register `rd` (5-bit hardware encoding, 0–31).
///
/// Returns a `Vec<u32>` of raw instruction words (1–4 instructions).
pub fn encode_mov_imm64(rd: u8, value: u64) -> Vec<u32> {
    let rd32 = (rd & 0x1F) as u32;
    let sf: u32 = 1; // 64-bit

    if value == 0 {
        return vec![enc_mov_wide(sf, 0b10, 0, 0, rd32)]; // MOVZ Xd, #0
    }

    let not_val = !value;
    let nz = (0..4u32)
        .filter(|&i| ((value >> (i * 16)) & 0xFFFF) != 0)
        .count();
    let nz_not = (0..4u32)
        .filter(|&i| ((not_val >> (i * 16)) & 0xFFFF) != 0)
        .count();

    // Prefer MOVN when the bitwise-NOT has fewer non-zero halfwords.
    if nz_not < nz {
        let mut result = Vec::with_capacity(4);
        let mut first = true;
        for i in 0u32..4 {
            let not_hw = ((not_val >> (i * 16)) & 0xFFFF) as u32;
            if first && not_hw != 0 {
                result.push(enc_mov_wide(sf, 0b00, i, not_hw, rd32)); // MOVN
                first = false;
            } else if !first {
                let actual_hw = ((value >> (i * 16)) & 0xFFFF) as u32;
                if actual_hw != 0xFFFF {
                    result.push(enc_mov_wide(sf, 0b11, i, actual_hw, rd32)); // MOVK
                }
            }
        }
        if result.is_empty() {
            result.push(enc_mov_wide(sf, 0b00, 0, 0, rd32));
        }
        return result;
    }

    // Standard MOVZ + MOVK sequence, skipping zero halfwords.
    let mut result = Vec::with_capacity(4);
    let mut first = true;
    for i in 0u32..4 {
        let hw_val = ((value >> (i * 16)) & 0xFFFF) as u32;
        if hw_val != 0 {
            if first {
                result.push(enc_mov_wide(sf, 0b10, i, hw_val, rd32)); // MOVZ
                first = false;
            } else {
                result.push(enc_mov_wide(sf, 0b11, i, hw_val, rd32)); // MOVK
            }
        }
    }
    if result.is_empty() {
        result.push(enc_mov_wide(sf, 0b10, 0, 0, rd32));
    }
    result
}

// ===========================================================================
// Load/Store Sizing Helper
// ===========================================================================

/// Determine (size, opc, byte_scale) for GP load/store unsigned offset.
fn ldst_size_opc_gp(opcode: &A64Opcode, is_32bit: bool) -> (u32, u32, u32) {
    match opcode {
        A64Opcode::LDRB_imm => (0b00, 0b01, 1),
        A64Opcode::STRB_imm => (0b00, 0b00, 1),
        A64Opcode::LDRH_imm => (0b01, 0b01, 2),
        A64Opcode::STRH_imm => (0b01, 0b00, 2),
        A64Opcode::LDRSB_imm => (0b00, if is_32bit { 0b11 } else { 0b10 }, 1),
        A64Opcode::LDRSH_imm => (0b01, if is_32bit { 0b11 } else { 0b10 }, 2),
        A64Opcode::LDRSW_imm => (0b10, 0b10, 4),
        A64Opcode::LDR_imm => {
            if is_32bit {
                (0b10, 0b01, 4)
            } else {
                (0b11, 0b01, 8)
            }
        }
        A64Opcode::STR_imm => {
            if is_32bit {
                (0b10, 0b00, 4)
            } else {
                (0b11, 0b00, 8)
            }
        }
        _ => {
            if is_32bit {
                (0b10, 0b01, 4)
            } else {
                (0b11, 0b01, 8)
            }
        }
    }
}

// ===========================================================================
// Main Encoding Dispatch
// ===========================================================================

impl AArch64Encoder {
    /// Encode a single A64 instruction into binary machine code.
    ///
    /// Returns an `EncodedInstruction` containing exactly 4 bytes (32 bits,
    /// little-endian).  When the instruction references an unresolved symbol a
    /// relocation is attached to the result.
    pub fn encode(&self, inst: &A64Instruction) -> Result<EncodedInstruction, String> {
        let rd = hw(inst.rd);
        let rn = hw(inst.rn);
        let rm = hw(inst.rm);
        let ra = hw_or_zr(inst.ra);
        let sf: u32 = if inst.is_32bit { 0 } else { 1 };
        let imm = inst.imm;
        let zr: u32 = registers::XZR;
        // The is_fp flag is consulted by FP load/store to select V-bit encoding
        // and by FP data processing to select ftype. It is carried on the
        // instruction but the opcode enum already encodes the FP vs GP distinction.
        let _is_fp = inst.is_fp;

        match inst.opcode {
            // =========================================================
            // Data Processing — Immediate (ADD / SUB)
            // =========================================================
            A64Opcode::ADD_imm | A64Opcode::ADDS_imm | A64Opcode::SUB_imm | A64Opcode::SUBS_imm => {
                let (op, s_flag): (u32, u32) = match inst.opcode {
                    A64Opcode::ADD_imm => (0, 0),
                    A64Opcode::ADDS_imm => (0, 1),
                    A64Opcode::SUB_imm => (1, 0),
                    _ => (1, 1), // SUBS_imm
                };
                if inst.symbol.is_some() {
                    let word = enc_add_sub_imm(sf, op, s_flag, 0, 0, rn, rd);
                    ok_reloc(word, AArch64RelocationType::AddAbsLo12Nc, imm)
                } else {
                    // Handle negative immediates by flipping ADD↔SUB.
                    let (actual_op, uimm) = if imm < 0 {
                        (op ^ 1, (-imm) as u64)
                    } else {
                        (op, imm as u64)
                    };
                    let (sh, imm12) = split_add_sub_imm(uimm)?;
                    ok(enc_add_sub_imm(sf, actual_op, s_flag, sh, imm12, rn, rd))
                }
            }

            // =========================================================
            // MOV Wide Immediate
            // =========================================================
            A64Opcode::MOVZ => {
                let hw_shift = (inst.shift as u32) / 16;
                let imm16 = (imm as u64 & 0xFFFF) as u32;
                ok(enc_mov_wide(sf, 0b10, hw_shift, imm16, rd))
            }
            A64Opcode::MOVK => {
                let hw_shift = (inst.shift as u32) / 16;
                let imm16 = (imm as u64 & 0xFFFF) as u32;
                ok(enc_mov_wide(sf, 0b11, hw_shift, imm16, rd))
            }
            A64Opcode::MOVN => {
                let hw_shift = (inst.shift as u32) / 16;
                let imm16 = (imm as u64 & 0xFFFF) as u32;
                ok(enc_mov_wide(sf, 0b00, hw_shift, imm16, rd))
            }

            // =========================================================
            // PC-Relative Addressing
            // =========================================================
            A64Opcode::ADRP => {
                if inst.symbol.is_some() {
                    let word = enc_pc_rel(1, 0, rd);
                    // shift==1 signals GOT-relative ADRP (PIC addressing)
                    let rtype = if inst.shift == 1 {
                        AArch64RelocationType::AdrGotPage
                    } else {
                        AArch64RelocationType::AdrPrelPgHi21
                    };
                    ok_reloc(word, rtype, imm)
                } else {
                    let sval = (imm >> 12) as i32;
                    ok(enc_pc_rel(1, sval, rd))
                }
            }
            A64Opcode::ADR => ok(enc_pc_rel(0, imm as i32, rd)),

            // =========================================================
            // Logical Immediate
            // =========================================================
            A64Opcode::AND_imm | A64Opcode::ORR_imm | A64Opcode::EOR_imm | A64Opcode::ANDS_imm => {
                let opc: u32 = match inst.opcode {
                    A64Opcode::AND_imm => 0b00,
                    A64Opcode::ORR_imm => 0b01,
                    A64Opcode::EOR_imm => 0b10,
                    _ => 0b11, // ANDS_imm
                };
                let is_64 = sf == 1;
                let uval = imm as u64;
                let (n, immr, imms) = encode_bitmask_immediate(uval, is_64).ok_or_else(|| {
                    format!("value 0x{:X} cannot be encoded as bitmask immediate", uval)
                })?;
                ok(enc_logical_imm(
                    sf,
                    opc,
                    n as u32,
                    immr as u32,
                    imms as u32,
                    rn,
                    rd,
                ))
            }

            // =========================================================
            // Bitfield (BFM / SBFM / UBFM)
            // =========================================================
            A64Opcode::BFM => {
                let n = sf;
                let immr = (imm as u32) & 0x3F;
                let imms = (inst.shift as u32) & 0x3F;
                ok(enc_bitfield(sf, 0b01, n, immr, imms, rn, rd))
            }
            A64Opcode::SBFM => {
                let n = sf;
                let immr = (imm as u32) & 0x3F;
                let imms = (inst.shift as u32) & 0x3F;
                ok(enc_bitfield(sf, 0b00, n, immr, imms, rn, rd))
            }
            A64Opcode::UBFM => {
                let n = sf;
                let immr = (imm as u32) & 0x3F;
                let imms = (inst.shift as u32) & 0x3F;
                ok(enc_bitfield(sf, 0b10, n, immr, imms, rn, rd))
            }

            // =========================================================
            // Shift Immediate (pseudo-bitfield)
            // =========================================================
            A64Opcode::LSL_imm => {
                let reg_bits: u32 = if inst.is_32bit { 32 } else { 64 };
                let shift_amt = (imm as u32) & (reg_bits - 1);
                let immr = (reg_bits - shift_amt) & (reg_bits - 1);
                let imms = reg_bits - 1 - shift_amt;
                ok(enc_bitfield(sf, 0b10, sf, immr, imms, rn, rd))
            }
            A64Opcode::LSR_imm => {
                let reg_bits: u32 = if inst.is_32bit { 32 } else { 64 };
                let shift_amt = (imm as u32) & (reg_bits - 1);
                ok(enc_bitfield(sf, 0b10, sf, shift_amt, reg_bits - 1, rn, rd))
            }
            A64Opcode::ASR_imm => {
                let reg_bits: u32 = if inst.is_32bit { 32 } else { 64 };
                let shift_amt = (imm as u32) & (reg_bits - 1);
                ok(enc_bitfield(sf, 0b00, sf, shift_amt, reg_bits - 1, rn, rd))
            }

            // Sign/Zero extend (pseudo-bitfield)
            A64Opcode::SXTB => ok(enc_bitfield(sf, 0b00, sf, 0, 7, rn, rd)),
            A64Opcode::SXTH => ok(enc_bitfield(sf, 0b00, sf, 0, 15, rn, rd)),
            A64Opcode::SXTW => ok(enc_bitfield(1, 0b00, 1, 0, 31, rn, rd)),
            A64Opcode::UXTB => ok(enc_bitfield(0, 0b10, 0, 0, 7, rn, rd)),
            A64Opcode::UXTH => ok(enc_bitfield(0, 0b10, 0, 0, 15, rn, rd)),

            // EXTR (extract)
            A64Opcode::EXTR => {
                let lsb = (imm as u32) & 0x3F;
                ok(enc_extr(sf, rm, lsb, rn, rd))
            }

            // =========================================================
            // Data Processing — Register (ADD/SUB/Logical shifted)
            // =========================================================
            A64Opcode::ADD_reg | A64Opcode::ADDS_reg | A64Opcode::SUB_reg | A64Opcode::SUBS_reg => {
                let (op, s_flag): (u32, u32) = match inst.opcode {
                    A64Opcode::ADD_reg => (0, 0),
                    A64Opcode::ADDS_reg => (0, 1),
                    A64Opcode::SUB_reg => (1, 0),
                    _ => (1, 1), // SUBS_reg
                };
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_add_sub_shifted(
                    sf, op, s_flag, shift_type, rm, shift_amt, rn, rd,
                ))
            }

            A64Opcode::AND_reg | A64Opcode::ORR_reg | A64Opcode::EOR_reg | A64Opcode::ANDS_reg => {
                let opc: u32 = match inst.opcode {
                    A64Opcode::AND_reg => 0b00,
                    A64Opcode::ORR_reg => 0b01,
                    A64Opcode::EOR_reg => 0b10,
                    _ => 0b11, // ANDS_reg
                };
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_logical_shifted(
                    sf, opc, shift_type, 0, rm, shift_amt, rn, rd,
                ))
            }

            A64Opcode::ORN_reg => {
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_logical_shifted(
                    sf, 0b01, shift_type, 1, rm, shift_amt, rn, rd,
                ))
            }
            A64Opcode::BIC_reg => {
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_logical_shifted(
                    sf, 0b00, shift_type, 1, rm, shift_amt, rn, rd,
                ))
            }

            // Shift register
            A64Opcode::LSL_reg => ok(enc_dp_2src(sf, rm, 0b001000, rn, rd)),
            A64Opcode::LSR_reg => ok(enc_dp_2src(sf, rm, 0b001001, rn, rd)),
            A64Opcode::ASR_reg => ok(enc_dp_2src(sf, rm, 0b001010, rn, rd)),
            A64Opcode::ROR_reg => ok(enc_dp_2src(sf, rm, 0b001011, rn, rd)),

            // =========================================================
            // Multiply / Divide
            // =========================================================
            A64Opcode::MADD => ok(enc_mul(sf, rm, 0, ra, rn, rd)),
            A64Opcode::MSUB => ok(enc_mul(sf, rm, 1, ra, rn, rd)),
            // MUL pseudo: MADD Rd, Rn, Rm, XZR (ra=31)
            A64Opcode::MUL => ok(enc_mul(sf, rm, 0, 31, rn, rd)),
            // SMULL: signed multiply long (32×32→64).
            // Encoding: sf=1, U=0, Ra=11111 (XZR), opcode bits for SMADDL.
            // SMADDL Xd, Wn, Wm, XZR  → 1 00 11011 001 Rm 0 11111 Rn Rd
            A64Opcode::SMULL => {
                let word = (1u32 << 31) // sf=1
                    | (0b00u32 << 29)   // U=0 (signed)
                    | (0b11011u32 << 24)
                    | (0b001u32 << 21)  // op54=00, op31=1 (long)
                    | ((rm & 0x1F) << 16)
                    | (0u32 << 15)      // o0=0 (add, not sub)
                    | (0b11111u32 << 10) // Ra=XZR (makes it SMULL not SMADDL)
                    | ((rn & 0x1F) << 5)
                    | (rd & 0x1F);
                ok(word)
            }
            A64Opcode::SDIV => ok(enc_dp_2src(sf, rm, 0b000011, rn, rd)),
            A64Opcode::UDIV => ok(enc_dp_2src(sf, rm, 0b000010, rn, rd)),
            A64Opcode::SMULH => ok(enc_smulh(rm, rn, rd)),
            A64Opcode::UMULH => ok(enc_umulh(rm, rn, rd)),

            // =========================================================
            // Conditional Select
            // =========================================================
            A64Opcode::CSEL => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 0, 0, rm, c, 0, rn, rd))
            }
            A64Opcode::CSINC => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 0, 0, rm, c, 1, rn, rd))
            }
            A64Opcode::CSINV => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 1, 0, rm, c, 0, rn, rd))
            }
            A64Opcode::CSNEG => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 1, 0, rm, c, 1, rn, rd))
            }
            A64Opcode::CSET => {
                let c = inst.cond.map(|c| c.invert().encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 0, 0, zr, c, 1, zr, rd))
            }
            A64Opcode::CSETM => {
                let c = inst.cond.map(|c| c.invert().encoding() as u32).unwrap_or(0);
                ok(enc_cond_select(sf, 1, 0, zr, c, 0, zr, rd))
            }

            // =========================================================
            // Bit Manipulation (1-source)
            // =========================================================
            A64Opcode::CLZ => ok(enc_dp_1src(sf, 0b000100, rn, rd)),
            A64Opcode::CLS => ok(enc_dp_1src(sf, 0b000101, rn, rd)),
            A64Opcode::RBIT => ok(enc_dp_1src(sf, 0b000000, rn, rd)),
            A64Opcode::REV => {
                if inst.is_32bit {
                    ok(enc_dp_1src(sf, 0b000010, rn, rd))
                } else {
                    ok(enc_dp_1src(sf, 0b000011, rn, rd))
                }
            }
            A64Opcode::REV16 => ok(enc_dp_1src(sf, 0b000001, rn, rd)),
            A64Opcode::REV32 => ok(enc_dp_1src(1, 0b000010, rn, rd)),

            // =========================================================
            // Loads — Unsigned Offset / Pre / Post / Register
            // =========================================================
            A64Opcode::LDR_imm
            | A64Opcode::LDRB_imm
            | A64Opcode::LDRH_imm
            | A64Opcode::LDRSB_imm
            | A64Opcode::LDRSH_imm
            | A64Opcode::LDRSW_imm => {
                let (size, opc, scale) = ldst_size_opc_gp(&inst.opcode, inst.is_32bit);
                if inst.symbol.is_some() {
                    let word = enc_ldst_unsigned(size, 0, opc, 0, rn, rd);
                    // shift==1 signals GOT-relative LDR (loading from GOT entry)
                    let rtype = if inst.shift == 1 {
                        AArch64RelocationType::Ld64GotLo12Nc
                    } else if scale == 8 {
                        AArch64RelocationType::Ldst64AbsLo12Nc
                    } else {
                        AArch64RelocationType::AddAbsLo12Nc
                    };
                    ok_reloc(word, rtype, imm)
                } else {
                    let offset = imm as u64;
                    let scaled = offset / (scale as u64);
                    ok(enc_ldst_unsigned(size, 0, opc, scaled as u32, rn, rd))
                }
            }

            A64Opcode::STR_imm | A64Opcode::STRB_imm | A64Opcode::STRH_imm => {
                let (size, opc, scale) = ldst_size_opc_gp(&inst.opcode, inst.is_32bit);
                if inst.symbol.is_some() {
                    let word = enc_ldst_unsigned(size, 0, opc, 0, rn, rd);
                    ok_reloc(word, AArch64RelocationType::AddAbsLo12Nc, imm)
                } else {
                    let offset = imm as u64;
                    let scaled = offset / (scale as u64);
                    ok(enc_ldst_unsigned(size, 0, opc, scaled as u32, rn, rd))
                }
            }

            A64Opcode::LDR_reg => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::LDR_imm, inst.is_32bit);
                let s = if inst.shift != 0 { 1u32 } else { 0 };
                ok(enc_ldst_reg(size, 0, opc, rm, 0b011, s, rn, rd))
            }
            A64Opcode::STR_reg => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::STR_imm, inst.is_32bit);
                let s = if inst.shift != 0 { 1u32 } else { 0 };
                ok(enc_ldst_reg(size, 0, opc, rm, 0b011, s, rn, rd))
            }

            A64Opcode::LDR_pre => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::LDR_imm, inst.is_32bit);
                ok(enc_ldst_indexed(size, 0, opc, imm as i32, 0b11, rn, rd))
            }
            A64Opcode::LDR_post => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::LDR_imm, inst.is_32bit);
                ok(enc_ldst_indexed(size, 0, opc, imm as i32, 0b01, rn, rd))
            }
            A64Opcode::STR_pre => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::STR_imm, inst.is_32bit);
                ok(enc_ldst_indexed(size, 0, opc, imm as i32, 0b11, rn, rd))
            }
            A64Opcode::STR_post => {
                let (size, opc, _) = ldst_size_opc_gp(&A64Opcode::STR_imm, inst.is_32bit);
                ok(enc_ldst_indexed(size, 0, opc, imm as i32, 0b01, rn, rd))
            }

            A64Opcode::LDR_literal => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b01 };
                if inst.symbol.is_some() {
                    let word = enc_ldr_literal(opc, 0, 0, rd);
                    ok_reloc(word, AArch64RelocationType::Condbr19, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_ldr_literal(opc, 0, offset, rd))
                }
            }

            // Load/Store pair
            A64Opcode::LDP => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b10 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 0, 0b10, 1, imm7, rt2, rn, rd))
            }
            // LDP post-index: LDP Rt, Rt2, [Rn], #imm  (idx=0b01)
            A64Opcode::LDP_post => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b10 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 0, 0b01, 1, imm7, rt2, rn, rd))
            }
            A64Opcode::STP => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b10 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 0, 0b10, 0, imm7, rt2, rn, rd))
            }
            // STP pre-index: STP Rt, Rt2, [Rn, #imm]!  (idx=0b11)
            A64Opcode::STP_pre => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b10 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 0, 0b11, 0, imm7, rt2, rn, rd))
            }
            A64Opcode::LDPSW => {
                let imm7 = (imm / 4) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(0b01, 0, 0b10, 1, imm7, rt2, rn, rd))
            }

            // FP Load/Store
            A64Opcode::LDR_fp_imm => {
                let (size, scale): (u32, u64) = if inst.is_32bit { (0b10, 4) } else { (0b11, 8) };
                let offset = (imm as u64) / scale;
                ok(enc_ldst_unsigned(size, 1, 0b01, offset as u32, rn, rd))
            }
            A64Opcode::STR_fp_imm => {
                let (size, scale): (u32, u64) = if inst.is_32bit { (0b10, 4) } else { (0b11, 8) };
                let offset = (imm as u64) / scale;
                ok(enc_ldst_unsigned(size, 1, 0b00, offset as u32, rn, rd))
            }
            A64Opcode::LDP_fp => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b01 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 1, 0b10, 1, imm7, rt2, rn, rd))
            }
            A64Opcode::STP_fp => {
                let opc = if inst.is_32bit { 0b00u32 } else { 0b01 };
                let scale = if inst.is_32bit { 4i64 } else { 8 };
                let imm7 = (imm / scale) as i32;
                let rt2 = rm;
                ok(enc_ldst_pair(opc, 1, 0b10, 0, imm7, rt2, rn, rd))
            }

            // =========================================================
            // Branches
            // =========================================================
            A64Opcode::B => {
                if inst.symbol.is_some() {
                    let word = enc_b(0, 0);
                    ok_reloc(word, AArch64RelocationType::Jump26, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_b(0, offset))
                }
            }
            A64Opcode::BL => {
                if inst.symbol.is_some() {
                    let word = enc_b(1, 0);
                    ok_reloc(word, AArch64RelocationType::Call26, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_b(1, offset))
                }
            }
            A64Opcode::B_cond => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                if inst.symbol.is_some() {
                    let word = enc_bcond(0, c);
                    ok_reloc(word, AArch64RelocationType::Condbr19, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_bcond(offset, c))
                }
            }
            A64Opcode::BR => ok(enc_br_reg(0b00, rn)),
            A64Opcode::BLR => ok(enc_br_reg(0b01, rn)),
            A64Opcode::RET => {
                let link = if inst.rn.is_some() { rn } else { 30u32 };
                ok(enc_br_reg(0b10, link))
            }
            A64Opcode::CBZ => {
                // The register to test (Rt) is in rn, not rd —
                // CBNZ/CBZ have no destination register.
                let rt = rn;
                if inst.symbol.is_some() {
                    let word = enc_cbz(sf, 0, 0, rt);
                    ok_reloc(word, AArch64RelocationType::Condbr19, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_cbz(sf, 0, offset, rt))
                }
            }
            A64Opcode::CBNZ => {
                // The register to test (Rt) is in rn, not rd.
                let rt = rn;
                if inst.symbol.is_some() {
                    let word = enc_cbz(sf, 1, 0, rt);
                    ok_reloc(word, AArch64RelocationType::Condbr19, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_cbz(sf, 1, offset, rt))
                }
            }
            A64Opcode::TBZ => {
                // Rt (register to test) is in rn — TBZ has no dest.
                let rt = rn;
                let bit = inst.shift as u32;
                if inst.symbol.is_some() {
                    let word = enc_tbz(bit, 0, 0, rt);
                    ok_reloc(word, AArch64RelocationType::Tstbr14, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_tbz(bit, 0, offset, rt))
                }
            }
            A64Opcode::TBNZ => {
                // Rt (register to test) is in rn — TBNZ has no dest.
                let rt = rn;
                let bit = inst.shift as u32;
                if inst.symbol.is_some() {
                    let word = enc_tbz(bit, 1, 0, rt);
                    ok_reloc(word, AArch64RelocationType::Tstbr14, imm)
                } else {
                    let offset = (imm >> 2) as i32;
                    ok(enc_tbz(bit, 1, offset, rt))
                }
            }

            // =========================================================
            // Comparison Pseudo-instructions
            // =========================================================
            A64Opcode::CMP_imm => {
                // CMP is SUBS with Rd=XZR.  Negative immediates become
                // CMN (ADDS with Rd=XZR).
                let (actual_op, uimm) = if imm < 0 {
                    (0u32, (-imm) as u64) // ADDS (CMN)
                } else {
                    (1u32, imm as u64) // SUBS (CMP)
                };
                let (sh, imm12) = split_add_sub_imm(uimm)?;
                ok(enc_add_sub_imm(sf, actual_op, 1, sh, imm12, rn, zr))
            }
            A64Opcode::CMP_reg => {
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_add_sub_shifted(
                    sf, 1, 1, shift_type, rm, shift_amt, rn, zr,
                ))
            }
            A64Opcode::CMN_imm => {
                // CMN is ADDS with Rd=XZR.  Negative → CMP (SUBS).
                let (actual_op, uimm) = if imm < 0 {
                    (1u32, (-imm) as u64) // SUBS (CMP)
                } else {
                    (0u32, imm as u64) // ADDS (CMN)
                };
                let (sh, imm12) = split_add_sub_imm(uimm)?;
                ok(enc_add_sub_imm(sf, actual_op, 1, sh, imm12, rn, zr))
            }
            A64Opcode::CMN_reg => {
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_add_sub_shifted(
                    sf, 0, 1, shift_type, rm, shift_amt, rn, zr,
                ))
            }
            A64Opcode::TST_imm => {
                let is_64 = sf == 1;
                let uval = imm as u64;
                let (n, immr, imms) = encode_bitmask_immediate(uval, is_64).ok_or_else(|| {
                    format!(
                        "TST: value 0x{:X} cannot be encoded as bitmask immediate",
                        uval
                    )
                })?;
                ok(enc_logical_imm(
                    sf,
                    0b11,
                    n as u32,
                    immr as u32,
                    imms as u32,
                    rn,
                    zr,
                ))
            }
            A64Opcode::TST_reg => {
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_logical_shifted(
                    sf, 0b11, shift_type, 0, rm, shift_amt, rn, zr,
                ))
            }
            A64Opcode::CCMP_imm => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                let nzcv = (inst.shift & 0xF) as u32;
                ok(enc_ccmp_imm(sf, 1, imm as u32, c, rn, nzcv))
            }
            A64Opcode::CCMP_reg => {
                let c = inst.cond.map(|c| c.encoding() as u32).unwrap_or(0);
                let nzcv = (inst.shift & 0xF) as u32;
                ok(enc_ccmp_reg(sf, 1, rm, c, rn, nzcv))
            }

            // =========================================================
            // Floating-Point Arithmetic
            // =========================================================
            A64Opcode::FADD_s => ok(enc_fp_2src(0b00, rm, 0b0010, rn, rd)),
            A64Opcode::FSUB_s => ok(enc_fp_2src(0b00, rm, 0b0011, rn, rd)),
            A64Opcode::FMUL_s => ok(enc_fp_2src(0b00, rm, 0b0000, rn, rd)),
            A64Opcode::FDIV_s => ok(enc_fp_2src(0b00, rm, 0b0001, rn, rd)),
            A64Opcode::FMIN_s => ok(enc_fp_2src(0b00, rm, 0b0100, rn, rd)),
            A64Opcode::FMAX_s => ok(enc_fp_2src(0b00, rm, 0b0101, rn, rd)),
            A64Opcode::FADD_d => ok(enc_fp_2src(0b01, rm, 0b0010, rn, rd)),
            A64Opcode::FSUB_d => ok(enc_fp_2src(0b01, rm, 0b0011, rn, rd)),
            A64Opcode::FMUL_d => ok(enc_fp_2src(0b01, rm, 0b0000, rn, rd)),
            A64Opcode::FDIV_d => ok(enc_fp_2src(0b01, rm, 0b0001, rn, rd)),
            A64Opcode::FMIN_d => ok(enc_fp_2src(0b01, rm, 0b0100, rn, rd)),
            A64Opcode::FMAX_d => ok(enc_fp_2src(0b01, rm, 0b0101, rn, rd)),

            // FP 1-source
            A64Opcode::FSQRT_s => ok(enc_fp_1src(0b00, 0b000011, rn, rd)),
            A64Opcode::FNEG_s => ok(enc_fp_1src(0b00, 0b000010, rn, rd)),
            A64Opcode::FABS_s => ok(enc_fp_1src(0b00, 0b000001, rn, rd)),
            A64Opcode::FSQRT_d => ok(enc_fp_1src(0b01, 0b000011, rn, rd)),
            A64Opcode::FNEG_d => ok(enc_fp_1src(0b01, 0b000010, rn, rd)),
            A64Opcode::FABS_d => ok(enc_fp_1src(0b01, 0b000001, rn, rd)),

            // FP conversion: single <-> double
            A64Opcode::FCVT_sd => ok(enc_fp_1src(0b00, 0b000101, rn, rd)),
            A64Opcode::FCVT_ds => ok(enc_fp_1src(0b01, 0b000100, rn, rd)),

            // FP fused multiply-add/sub
            A64Opcode::FMADD_s => ok(enc_fp_fma(0b00, 0, rm, 0, ra, rn, rd)),
            A64Opcode::FMSUB_s => ok(enc_fp_fma(0b00, 0, rm, 1, ra, rn, rd)),
            A64Opcode::FNMADD_s => ok(enc_fp_fma(0b00, 1, rm, 0, ra, rn, rd)),
            A64Opcode::FNMSUB_s => ok(enc_fp_fma(0b00, 1, rm, 1, ra, rn, rd)),
            A64Opcode::FMADD_d => ok(enc_fp_fma(0b01, 0, rm, 0, ra, rn, rd)),
            A64Opcode::FMSUB_d => ok(enc_fp_fma(0b01, 0, rm, 1, ra, rn, rd)),
            A64Opcode::FNMADD_d => ok(enc_fp_fma(0b01, 1, rm, 0, ra, rn, rd)),
            A64Opcode::FNMSUB_d => ok(enc_fp_fma(0b01, 1, rm, 1, ra, rn, rd)),

            // =========================================================
            // FP Comparison
            // =========================================================
            A64Opcode::FCMP_s => ok(enc_fp_cmp(0b00, rm, rn, 0b00000)),
            A64Opcode::FCMP_d => ok(enc_fp_cmp(0b01, rm, rn, 0b00000)),
            A64Opcode::FCMPE_s => ok(enc_fp_cmp(0b00, rm, rn, 0b10000)),
            A64Opcode::FCMPE_d => ok(enc_fp_cmp(0b01, rm, rn, 0b10000)),

            // =========================================================
            // FP <-> Integer Conversion
            // =========================================================
            A64Opcode::FCVTZS_ws => ok(enc_fp_int_cvt(0, 0b00, 0b11, 0b000, rn, rd)),
            A64Opcode::FCVTZS_xs => ok(enc_fp_int_cvt(1, 0b00, 0b11, 0b000, rn, rd)),
            A64Opcode::FCVTZS_wd => ok(enc_fp_int_cvt(0, 0b01, 0b11, 0b000, rn, rd)),
            A64Opcode::FCVTZS_xd => ok(enc_fp_int_cvt(1, 0b01, 0b11, 0b000, rn, rd)),
            A64Opcode::FCVTZU_ws => ok(enc_fp_int_cvt(0, 0b00, 0b11, 0b001, rn, rd)),
            A64Opcode::FCVTZU_xs => ok(enc_fp_int_cvt(1, 0b00, 0b11, 0b001, rn, rd)),
            A64Opcode::FCVTZU_wd => ok(enc_fp_int_cvt(0, 0b01, 0b11, 0b001, rn, rd)),
            A64Opcode::FCVTZU_xd => ok(enc_fp_int_cvt(1, 0b01, 0b11, 0b001, rn, rd)),
            A64Opcode::SCVTF_sw => ok(enc_fp_int_cvt(0, 0b00, 0b00, 0b010, rn, rd)),
            A64Opcode::SCVTF_sx => ok(enc_fp_int_cvt(1, 0b00, 0b00, 0b010, rn, rd)),
            A64Opcode::SCVTF_dw => ok(enc_fp_int_cvt(0, 0b01, 0b00, 0b010, rn, rd)),
            A64Opcode::SCVTF_dx => ok(enc_fp_int_cvt(1, 0b01, 0b00, 0b010, rn, rd)),
            A64Opcode::UCVTF_sw => ok(enc_fp_int_cvt(0, 0b00, 0b00, 0b011, rn, rd)),
            A64Opcode::UCVTF_sx => ok(enc_fp_int_cvt(1, 0b00, 0b00, 0b011, rn, rd)),
            A64Opcode::UCVTF_dw => ok(enc_fp_int_cvt(0, 0b01, 0b00, 0b011, rn, rd)),
            A64Opcode::UCVTF_dx => ok(enc_fp_int_cvt(1, 0b01, 0b00, 0b011, rn, rd)),

            // =========================================================
            // FP Move
            // =========================================================
            A64Opcode::FMOV_s => ok(enc_fp_1src(0b00, 0b000000, rn, rd)),
            A64Opcode::FMOV_d => ok(enc_fp_1src(0b01, 0b000000, rn, rd)),
            A64Opcode::FMOV_gen_to_fp => {
                let (s, ft) = if inst.is_32bit {
                    (0u32, 0b00u32)
                } else {
                    (1, 0b01)
                };
                ok(enc_fp_int_cvt(s, ft, 0b00, 0b111, rn, rd))
            }
            A64Opcode::FMOV_fp_to_gen => {
                let (s, ft) = if inst.is_32bit {
                    (0u32, 0b00u32)
                } else {
                    (1, 0b01)
                };
                ok(enc_fp_int_cvt(s, ft, 0b00, 0b110, rn, rd))
            }

            // =========================================================
            // SIMD / Vector Instructions
            // =========================================================
            A64Opcode::CNT => {
                // CNT Vd.8B, Vn.8B — AdvSIMD two-reg misc
                // Bit layout: 0 Q U 01110 size 10000 opcode 10 Rn Rd
                //              31 30 29 28-24 23-22 21-17 16-12 11-10 9-5 4-0
                // Q=0 (8B), U=0, size=00, opcode=00101
                let q = 0u32;
                let word = (q << 30)
                    | (0b01110u32 << 24) // bits 28-24: fixed 01110
                    // bits 23-22: size=00 (implicit zero)
                    | (0b10000u32 << 17) // bits 21-17: fixed 10000
                    | (0b00101u32 << 12) // bits 16-12: opcode for CNT
                    | (0b10u32 << 10) // bits 11-10: fixed 10
                    | ((rn & 0x1F) << 5)
                    | (rd & 0x1F);
                ok(word)
            }
            A64Opcode::ADDV => {
                // ADDV Bd, Vn.8B — AdvSIMD across lanes
                // Bit layout: 0 Q U 01110 size 11000 opcode 10 Rn Rd
                //              31 30 29 28-24 23-22 21-17 16-12 11-10 9-5 4-0
                // Q=0 (8B), U=0, size=00, opcode=11011
                let q = 0u32;
                let word = (q << 30)
                    | (0b01110u32 << 24) // bits 28-24: fixed 01110
                    // bits 23-22: size=00 (implicit zero)
                    | (0b11000u32 << 17) // bits 21-17: fixed 11000
                    | (0b11011u32 << 12) // bits 16-12: opcode for ADDV
                    | (0b10u32 << 10) // bits 11-10: fixed 10
                    | ((rn & 0x1F) << 5)
                    | (rd & 0x1F);
                ok(word)
            }

            // =========================================================
            // System Instructions
            // =========================================================
            A64Opcode::NOP => ok(0xD503201F),
            A64Opcode::DMB => {
                let crm = (imm as u32) & 0xF;
                ok(0xD5033000 | (crm << 8) | 0xBF)
            }
            A64Opcode::DSB => {
                let crm = (imm as u32) & 0xF;
                ok(0xD5033000 | (crm << 8) | 0x9F)
            }
            A64Opcode::ISB => {
                let crm = (imm as u32) & 0xF;
                ok(0xD5033000 | (crm << 8) | 0xDF)
            }
            A64Opcode::SVC => ok(enc_svc(imm as u32)),
            A64Opcode::HVC => ok(enc_hvc(imm as u32)),
            A64Opcode::SMC => ok(enc_smc(imm as u32)),
            A64Opcode::MRS => {
                let sysreg = (imm as u32) & 0xFFFF;
                ok(0xD5300000 | (sysreg << 5) | rd)
            }
            A64Opcode::MSR => {
                let sysreg = (imm as u32) & 0xFFFF;
                ok(0xD5100000 | (sysreg << 5) | rd)
            }

            // =========================================================
            // Pseudo-Instructions
            // =========================================================
            A64Opcode::MOV_reg => {
                // MOV Xd, Xm -> ORR Xd, XZR, Xm
                ok(enc_logical_shifted(sf, 0b01, 0, 0, rm, 0, zr, rd))
            }
            A64Opcode::MOV_imm => {
                let val = imm as u64;
                let seq = encode_mov_imm64(rd as u8, val);
                if seq.is_empty() {
                    // Zero value — emit MOVZ Xd, #0
                    ok(enc_mov_wide(sf, 0b10, 0, 0, rd))
                } else {
                    // Emit the COMPLETE MOVZ + MOVK sequence for the full
                    // 64-bit immediate.  `encode_mov_imm64` returns 1–4
                    // instructions depending on how many 16-bit halfwords
                    // are non-zero.  We must emit ALL of them — dropping
                    // MOVK instructions would silently truncate the value.
                    let mut bytes = Vec::with_capacity(seq.len() * 4);
                    for instr_word in &seq {
                        bytes.extend_from_slice(&instr_word.to_le_bytes());
                    }
                    Ok(EncodedInstruction {
                        bytes,
                        relocation: None,
                    })
                }
            }
            A64Opcode::NEG_reg => {
                // NEG Xd, Xm -> SUB Xd, XZR, Xm
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_add_sub_shifted(
                    sf, 1, 0, shift_type, rm, shift_amt, zr, rd,
                ))
            }
            A64Opcode::MVN_reg => {
                // MVN Xd, Xm -> ORN Xd, XZR, Xm
                let shift_type = ((inst.shift >> 6) & 0x3) as u32;
                let shift_amt = (inst.shift & 0x3F) as u32;
                ok(enc_logical_shifted(
                    sf, 0b01, shift_type, 1, rm, shift_amt, zr, rd,
                ))
            }

            // Pseudo-instructions that must be expanded by codegen
            A64Opcode::LI | A64Opcode::LA | A64Opcode::CALL | A64Opcode::INLINE_ASM => {
                Err(format!(
                    "Pseudo-instruction {:?} must be expanded before encoding",
                    inst.opcode
                ))
            }
        }
    }
}
