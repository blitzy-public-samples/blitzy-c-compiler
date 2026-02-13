//! AArch64 relocation application module for the BCC built-in linker.
//!
//! Implements the [`ArchRelocationHandler`] trait for all `R_AARCH64_*` ELF
//! relocation types as defined by the ARM ELF specification (ELF for the Arm®
//! 64-bit Architecture — AArch64).
//!
//! # Relocation Categories
//!
//! - **Absolute data relocations:** `R_AARCH64_ABS64`, `ABS32`, `ABS16`
//! - **PC-relative data relocations:** `R_AARCH64_PREL64`, `PREL32`, `PREL16`
//! - **Direct branch relocations:** `R_AARCH64_CALL26`, `JUMP26` (±128 MiB range)
//! - **ADRP page-relative:** `R_AARCH64_ADR_PREL_PG_HI21` (4 KiB page alignment)
//! - **ADD/LDR low-12 bit:** `R_AARCH64_ADD_ABS_LO12_NC`, `LDST{8,16,32,64,128}_ABS_LO12_NC`
//! - **Conditional/test branches:** `R_AARCH64_CONDBR19`, `TSTBR14`
//! - **MOVW immediate:** `R_AARCH64_MOVW_UABS_G{0..3}`, `MOVW_SABS_G{0..2}`
//! - **GOT-relative (PIC):** `R_AARCH64_ADR_GOT_PAGE`, `LD64_GOT_LO12_NC`
//! - **TLS relocations:** `TLSGD_*`, `TLSIE_*`, `TLSLE_*`
//! - **Dynamic relocations:** `R_AARCH64_GLOB_DAT`, `JUMP_SLOT`, `RELATIVE`, `COPY`
//!
//! # A64 Instruction Encoding
//!
//! All AArch64 instructions are fixed 32-bit little-endian words. Relocation
//! handlers read the existing instruction, mask out the relevant immediate
//! field bits, OR in the computed value, and write back the patched instruction.
//!
//! # Page-Relative Addressing (ADRP)
//!
//! ADRP computes a page-relative offset: `(Page(S+A) - Page(P)) >> 12`, where
//! `Page(x) = x & ~0xFFF`. The 21-bit signed result is split across two
//! instruction fields: `immhi` (bits [23:5]) and `immlo` (bits [30:29]).

use crate::backend::linker_common::relocation::{
    ArchRelocationHandler, RelocationEntry, RelocationError,
};

// ===========================================================================
// AArch64 ELF Relocation Type Constants
// ===========================================================================
// Reference: ELF for the Arm® 64-bit Architecture (AArch64) — Table 4-9

/// No relocation.
pub const R_AARCH64_NONE: u32 = 0;

/// 64-bit absolute: S + A
pub const R_AARCH64_ABS64: u32 = 257;
/// 32-bit absolute: S + A (overflow check)
pub const R_AARCH64_ABS32: u32 = 258;
/// 16-bit absolute: S + A (overflow check)
pub const R_AARCH64_ABS16: u32 = 259;

/// 64-bit PC-relative: S + A - P
pub const R_AARCH64_PREL64: u32 = 260;
/// 32-bit PC-relative: S + A - P (overflow check)
pub const R_AARCH64_PREL32: u32 = 261;
/// 16-bit PC-relative: S + A - P (overflow check)
pub const R_AARCH64_PREL16: u32 = 262;

/// MOVZ/MOVK: unsigned imm16 bits [15:0] of S + A
pub const R_AARCH64_MOVW_UABS_G0: u32 = 263;
/// MOVZ/MOVK: unsigned imm16 bits [15:0], no overflow check
pub const R_AARCH64_MOVW_UABS_G0_NC: u32 = 264;
/// MOVZ/MOVK: unsigned imm16 bits [31:16] of S + A
pub const R_AARCH64_MOVW_UABS_G1: u32 = 265;
/// MOVZ/MOVK: unsigned imm16 bits [31:16], no overflow check
pub const R_AARCH64_MOVW_UABS_G1_NC: u32 = 266;
/// MOVZ/MOVK: unsigned imm16 bits [47:32] of S + A
pub const R_AARCH64_MOVW_UABS_G2: u32 = 267;
/// MOVZ/MOVK: unsigned imm16 bits [47:32], no overflow check
pub const R_AARCH64_MOVW_UABS_G2_NC: u32 = 268;
/// MOVZ/MOVK: unsigned imm16 bits [63:48] of S + A
pub const R_AARCH64_MOVW_UABS_G3: u32 = 269;

/// MOVZ/MOVN: signed imm16 bits [15:0] of S + A
pub const R_AARCH64_MOVW_SABS_G0: u32 = 270;
/// MOVZ/MOVN: signed imm16 bits [31:16] of S + A
pub const R_AARCH64_MOVW_SABS_G1: u32 = 271;
/// MOVZ/MOVN: signed imm16 bits [47:32] of S + A
pub const R_AARCH64_MOVW_SABS_G2: u32 = 272;

/// ADRP: page-relative high 21 bits — (Page(S+A) - Page(P)) >> 12
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;
/// ADRP: page-relative high 21 bits, no overflow check
pub const R_AARCH64_ADR_PREL_PG_HI21_NC: u32 = 276;

/// ADD: low 12 bits of (S + A), no overflow check
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;

/// LDR/STR byte: low 12 bits of (S + A), no scaling (shift=0)
pub const R_AARCH64_LDST8_ABS_LO12_NC: u32 = 278;

/// TBZ/TBNZ: 14-bit PC-relative (S + A - P), ±32 KiB range
pub const R_AARCH64_TSTBR14: u32 = 279;
/// B.cond/CBZ/CBNZ: 19-bit PC-relative (S + A - P), ±1 MiB range
pub const R_AARCH64_CONDBR19: u32 = 280;

/// B (unconditional): 26-bit PC-relative (S + A - P), ±128 MiB range
pub const R_AARCH64_JUMP26: u32 = 282;
/// BL (call): 26-bit PC-relative (S + A - P), ±128 MiB range
pub const R_AARCH64_CALL26: u32 = 283;

/// LDR/STR halfword: low 12 bits of (S + A) >> 1 (scaled by 2)
pub const R_AARCH64_LDST16_ABS_LO12_NC: u32 = 284;
/// LDR/STR word: low 12 bits of (S + A) >> 2 (scaled by 4)
pub const R_AARCH64_LDST32_ABS_LO12_NC: u32 = 285;
/// LDR/STR doubleword: low 12 bits of (S + A) >> 3 (scaled by 8)
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 286;

/// LDR/STR quadword: low 12 bits of (S + A) >> 4 (scaled by 16)
pub const R_AARCH64_LDST128_ABS_LO12_NC: u32 = 299;

/// ADRP to GOT entry page: (Page(G(S)) - Page(P)) >> 12
pub const R_AARCH64_ADR_GOT_PAGE: u32 = 311;
/// LDR from GOT entry: low 12 bits of G(S) >> 3 (8-byte aligned load)
pub const R_AARCH64_LD64_GOT_LO12_NC: u32 = 312;

// TLS relocations
/// TLS GD: ADRP to GOT TLS descriptor page
pub const R_AARCH64_TLSGD_ADR_PAGE21: u32 = 513;
/// TLS GD: ADD low 12 of GOT TLS descriptor
pub const R_AARCH64_TLSGD_ADD_LO12_NC: u32 = 514;
/// TLS IE: ADRP to GOT TP offset page
pub const R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21: u32 = 539;
/// TLS IE: LDR from GOT TP offset
pub const R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC: u32 = 540;
/// TLS LE: ADD high 12 of TP offset
pub const R_AARCH64_TLSLE_ADD_TPREL_HI12: u32 = 549;
/// TLS LE: ADD low 12 of TP offset, no overflow check
pub const R_AARCH64_TLSLE_ADD_TPREL_LO12_NC: u32 = 550;

// Dynamic relocation types
/// Copy symbol at runtime
pub const R_AARCH64_COPY: u32 = 1024;
/// GOT entry: set to S
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
/// PLT jump slot: set to S
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
/// Adjust by base address: B + A
pub const R_AARCH64_RELATIVE: u32 = 1027;
/// TLS module ID
pub const R_AARCH64_TLS_DTPMOD64: u32 = 1028;
/// TLS module offset
pub const R_AARCH64_TLS_DTPREL64: u32 = 1029;
/// TLS TP-relative offset
pub const R_AARCH64_TLS_TPREL64: u32 = 1030;
/// TLS descriptor
pub const R_AARCH64_TLSDESC: u32 = 1031;

// ===========================================================================
// Helper Functions — Little-Endian I/O, Page Arithmetic, Encoding
// ===========================================================================

/// Reads a 32-bit little-endian unsigned integer from `data` at `offset`.
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`.
#[inline]
pub fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    u32::from_le_bytes(bytes)
}

/// Writes a 32-bit little-endian unsigned integer to `data` at `offset`.
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`.
#[inline]
pub fn write_u32_le(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
}

/// Reads a 64-bit little-endian unsigned integer from `data` at `offset`.
///
/// # Panics
///
/// Panics if `offset + 8 > data.len()`.
#[inline]
pub fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    let bytes: [u8; 8] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ];
    u64::from_le_bytes(bytes)
}

/// Writes a 64-bit little-endian unsigned integer to `data` at `offset`.
///
/// # Panics
///
/// Panics if `offset + 8 > data.len()`.
#[inline]
pub fn write_u64_le(data: &mut [u8], offset: usize, value: u64) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
    data[offset + 4] = bytes[4];
    data[offset + 5] = bytes[5];
    data[offset + 6] = bytes[6];
    data[offset + 7] = bytes[7];
}

/// Returns the 4 KiB page address for `addr` by masking out the low 12 bits.
///
/// `Page(x) = x & ~0xFFF`
///
/// This is the fundamental operation behind ADRP instruction relocation,
/// which addresses data relative to 4 KiB page boundaries.
#[inline]
pub fn page(addr: u64) -> u64 {
    addr & !0xFFF
}

/// Sign-extends a value from the specified bit width to a full `i64`.
///
/// For example, `sign_extend(0x1FFFFF, 21)` produces `-1_i64` because the
/// 21st bit (sign bit for a 21-bit field) is set.
///
/// # Parameters
///
/// - `value`: The raw unsigned value to sign-extend.
/// - `bits`: The original bit width of the value (1..=64).
#[inline]
pub fn sign_extend(value: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        return value as i64;
    }
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// Encodes an ADRP page offset into the `immhi:immlo` instruction fields.
///
/// The 21-bit signed page offset is split as follows:
/// - `immlo` = bits [1:0] of the value → encoded in instruction bits [30:29]
/// - `immhi` = bits [20:2] of the value → encoded in instruction bits [23:5]
///
/// The returned `u32` has only the `immhi` and `immlo` bits set; the caller
/// must OR this with the existing instruction (after masking out those fields).
#[inline]
pub fn encode_adrp_immhi_immlo(value: i64) -> u32 {
    let v = value as u32;
    let immlo = (v & 0x3) << 29;           // bits [1:0] → instruction [30:29]
    let immhi = ((v >> 2) & 0x7FFFF) << 5; // bits [20:2] → instruction [23:5]
    immlo | immhi
}

/// Encodes a 12-bit immediate into instruction bits [21:10].
///
/// Used by ADD and load/store instructions that carry a 12-bit unsigned
/// immediate field.
#[inline]
pub fn encode_imm12(value: u32) -> u32 {
    (value & 0xFFF) << 10
}

/// Encodes a 14-bit immediate into instruction bits [18:5].
///
/// Used by TBZ/TBNZ (test-and-branch) instructions. The 14-bit field
/// represents a signed word-aligned offset (shifted left by 2 before use).
#[inline]
pub fn encode_imm14(value: i64) -> u32 {
    ((value as u32) & 0x3FFF) << 5
}

/// Encodes a 16-bit immediate into instruction bits [20:5].
///
/// Used by MOVZ/MOVK/MOVN instructions for 16-bit immediate moves.
#[inline]
pub fn encode_imm16(value: u64) -> u32 {
    ((value as u32) & 0xFFFF) << 5
}

/// Encodes a 19-bit immediate into instruction bits [23:5].
///
/// Used by B.cond, CBZ, and CBNZ instructions. The 19-bit field represents
/// a signed word-aligned offset (shifted left by 2 before use).
#[inline]
pub fn encode_imm19(value: i64) -> u32 {
    ((value as u32) & 0x7FFFF) << 5
}

/// Encodes a 26-bit immediate into instruction bits [25:0].
///
/// Used by B (unconditional branch) and BL (branch-with-link) instructions.
/// The 26-bit field represents a signed word-aligned offset (shifted left by 2
/// before use), giving a ±128 MiB range.
#[inline]
pub fn encode_imm26(value: i64) -> u32 {
    (value as u32) & 0x03FF_FFFF
}

// ===========================================================================
// Instruction field masks for read-modify-write patching
// ===========================================================================

/// Mask for ADRP immhi field: instruction bits [23:5]
const ADRP_IMMHI_MASK: u32 = 0x00FF_FFE0;
/// Mask for ADRP immlo field: instruction bits [30:29]
const ADRP_IMMLO_MASK: u32 = 0x6000_0000;
/// Combined mask for both ADRP immediate fields
const ADRP_IMM_MASK: u32 = ADRP_IMMHI_MASK | ADRP_IMMLO_MASK;

/// Mask for 12-bit immediate field: instruction bits [21:10]
const IMM12_MASK: u32 = 0x003F_FC00;

/// Mask for 14-bit immediate field (TBZ/TBNZ): instruction bits [18:5]
const IMM14_MASK: u32 = 0x0007_FFE0;

/// Mask for 16-bit immediate field (MOVZ/MOVK/MOVN): instruction bits [20:5]
const IMM16_MASK: u32 = 0x001F_FFE0;

/// Mask for 19-bit immediate field (B.cond/CBZ/CBNZ): instruction bits [23:5]
const IMM19_MASK: u32 = 0x00FF_FFE0;

/// Mask for 26-bit immediate field (B/BL): instruction bits [25:0]
const IMM26_MASK: u32 = 0x03FF_FFFF;

// ===========================================================================
// AArch64RelocationHandler — architecture-specific relocation application
// ===========================================================================

/// AArch64 relocation handler implementing [`ArchRelocationHandler`].
///
/// Handles all `R_AARCH64_*` relocation types for the BCC built-in linker,
/// including absolute data, PC-relative, ADRP page-relative, branch,
/// conditional branch, MOVW immediate, GOT-relative, and TLS relocations.
///
/// All relocation application functions follow the read-modify-write pattern:
/// 1. Read the existing instruction or data word at the relocation offset.
/// 2. Mask out the relevant immediate field bits.
/// 3. Compute the relocation value using the architecture-specific formula.
/// 4. Encode the value into the immediate field bits.
/// 5. OR the encoded bits with the masked instruction and write back.
pub struct AArch64RelocationHandler;

impl AArch64RelocationHandler {
    /// Creates a new AArch64 relocation handler.
    pub fn new() -> Self {
        AArch64RelocationHandler
    }

    // =======================================================================
    // Absolute data relocation application
    // =======================================================================

    /// Applies a 64-bit absolute relocation (`R_AARCH64_ABS64`).
    ///
    /// Writes `S + A` as an 8-byte little-endian value at the given offset.
    /// No overflow check is performed for 64-bit absolute relocations.
    pub fn apply_abs64(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 8 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        write_u64_le(data, offset, value as u64);
        Ok(())
    }

    /// Applies a 32-bit absolute relocation (`R_AARCH64_ABS32`).
    ///
    /// Writes `S + A` as a 4-byte little-endian value at the given offset.
    /// Returns [`RelocationError::Overflow`] if the value does not fit in an
    /// unsigned 32-bit range.
    pub fn apply_abs32(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        // The AArch64 ELF spec treats ABS32 as unsigned, but the addend may
        // make it negative. We check both signed and unsigned 32-bit range:
        // the value must be representable as either i32 or u32.
        if value > u32::MAX as i64 || value < i32::MIN as i64 {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_ABS32,
                offset: offset as u64,
                value: value as i128,
                max_value: u32::MAX as i128,
            });
        }
        write_u32_le(data, offset, value as u32);
        Ok(())
    }

    /// Applies a 16-bit absolute relocation (`R_AARCH64_ABS16`).
    ///
    /// Writes `S + A` as a 2-byte little-endian value at the given offset.
    /// Returns [`RelocationError::Overflow`] if the value does not fit in an
    /// unsigned 16-bit range.
    pub fn apply_abs16(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 2 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        if value > u16::MAX as i64 || value < i16::MIN as i64 {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_ABS16,
                offset: offset as u64,
                value: value as i128,
                max_value: u16::MAX as i128,
            });
        }
        data[offset] = value as u8;
        data[offset + 1] = (value >> 8) as u8;
        Ok(())
    }

    // =======================================================================
    // PC-relative data relocation application
    // =======================================================================

    /// Applies a 64-bit PC-relative relocation (`R_AARCH64_PREL64`).
    ///
    /// Writes `S + A - P` as an 8-byte little-endian value. No overflow check.
    pub fn apply_prel64(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 8 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        write_u64_le(data, offset, value as u64);
        Ok(())
    }

    /// Applies a 32-bit PC-relative relocation (`R_AARCH64_PREL32`).
    ///
    /// Writes `S + A - P` as a 4-byte little-endian value.
    /// Returns [`RelocationError::Overflow`] if the value exceeds signed 32-bit range.
    pub fn apply_prel32(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        if value > i32::MAX as i64 || value < i32::MIN as i64 {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_PREL32,
                offset: offset as u64,
                value: value as i128,
                max_value: i32::MAX as i128,
            });
        }
        write_u32_le(data, offset, value as u32);
        Ok(())
    }

    // =======================================================================
    // ADRP and ADD/LDR low-12 relocation application
    // =======================================================================

    /// Applies an ADRP page-relative relocation (`R_AARCH64_ADR_PREL_PG_HI21`).
    ///
    /// Computes `(Page(S + A) - Page(P)) >> 12` and encodes the 21-bit signed
    /// result into the ADRP instruction's `immhi:immlo` fields.
    ///
    /// The ADRP instruction adds the sign-extended, page-scaled immediate to
    /// the page address of the current PC, producing a base address for a
    /// subsequent ADD or LDR instruction that supplies the low 12 bits.
    ///
    /// # Range
    ///
    /// Signed 21-bit page offset covers ±4 GiB of address space.
    pub fn apply_adr_prel_pg_hi21(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        // Range check: signed 21-bit (±2^20 pages = ±4 GiB)
        let max = (1_i64 << 20) - 1;
        let min = -(1_i64 << 20);
        if value > max || value < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_ADR_PREL_PG_HI21,
                offset: offset as u64,
                value: value as i128,
                max_value: max as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !ADRP_IMM_MASK) | encode_adrp_immhi_immlo(value);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies an ADD low-12-bit relocation (`R_AARCH64_ADD_ABS_LO12_NC`).
    ///
    /// Extracts the low 12 bits of `(S + A)` and encodes them into the ADD
    /// instruction's `imm12` field (bits [21:10]). No overflow check is
    /// performed (the `_NC` suffix indicates "no check").
    pub fn apply_add_abs_lo12_nc(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let imm12_value = (value as u64 & 0xFFF) as u32;
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM12_MASK) | encode_imm12(imm12_value);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies a load/store low-12-bit relocation with scaling.
    ///
    /// Used for `R_AARCH64_LDST{8,16,32,64,128}_ABS_LO12_NC`.
    ///
    /// Extracts the low 12 bits of `(S + A)`, right-shifts by `shift` for
    /// alignment scaling, and encodes into the LDR/STR instruction's `imm12`
    /// field (bits [21:10]). No overflow check (`_NC`).
    ///
    /// # Parameters
    ///
    /// - `shift`: Scaling factor — 0 for byte, 1 for halfword, 2 for word,
    ///   3 for doubleword, 4 for quadword.
    pub fn apply_ldst_lo12(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
        shift: u32,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let imm12_value = ((value as u64 & 0xFFF) >> shift) as u32;
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM12_MASK) | encode_imm12(imm12_value);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    // =======================================================================
    // Branch relocation application
    // =======================================================================

    /// Applies a BL (call) relocation (`R_AARCH64_CALL26`).
    ///
    /// Computes `(S + A - P) >> 2` (instructions are 4-byte aligned) and
    /// encodes the 26-bit signed result into the BL instruction's `imm26`
    /// field (bits [25:0]).
    ///
    /// # Range
    ///
    /// Signed 26-bit word offset covers ±128 MiB.
    pub fn apply_call26(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        // The value passed in is already (S + A - P). We need to convert to
        // instruction words by dividing by 4 (right-shift by 2).
        let word_offset = value >> 2;
        // Range check: signed 26-bit (±2^25 words = ±128 MiB)
        let max = (1_i64 << 25) - 1;
        let min = -(1_i64 << 25);
        if word_offset > max || word_offset < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_CALL26,
                offset: offset as u64,
                value: value as i128,
                max_value: (max << 2) as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM26_MASK) | encode_imm26(word_offset);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies a B (unconditional branch) relocation (`R_AARCH64_JUMP26`).
    ///
    /// Same encoding and range as [`apply_call26`] but for the B instruction
    /// instead of BL. The only difference is the opcode portion of the
    /// instruction (preserved by the mask), not the immediate field encoding.
    pub fn apply_jump26(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let word_offset = value >> 2;
        let max = (1_i64 << 25) - 1;
        let min = -(1_i64 << 25);
        if word_offset > max || word_offset < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_JUMP26,
                offset: offset as u64,
                value: value as i128,
                max_value: (max << 2) as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM26_MASK) | encode_imm26(word_offset);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies a conditional branch relocation (`R_AARCH64_CONDBR19`).
    ///
    /// Used by B.cond, CBZ, and CBNZ instructions. Computes
    /// `(S + A - P) >> 2` and encodes the 19-bit signed result into the
    /// instruction's `imm19` field (bits [23:5]).
    ///
    /// # Range
    ///
    /// Signed 19-bit word offset covers ±1 MiB.
    pub fn apply_condbr19(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let word_offset = value >> 2;
        let max = (1_i64 << 18) - 1;
        let min = -(1_i64 << 18);
        if word_offset > max || word_offset < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_CONDBR19,
                offset: offset as u64,
                value: value as i128,
                max_value: (max << 2) as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM19_MASK) | encode_imm19(word_offset);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies a test-and-branch relocation (`R_AARCH64_TSTBR14`).
    ///
    /// Used by TBZ and TBNZ instructions. Computes `(S + A - P) >> 2` and
    /// encodes the 14-bit signed result into the instruction's `imm14` field
    /// (bits [18:5]).
    ///
    /// # Range
    ///
    /// Signed 14-bit word offset covers ±32 KiB.
    pub fn apply_tstbr14(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let word_offset = value >> 2;
        let max = (1_i64 << 13) - 1;
        let min = -(1_i64 << 13);
        if word_offset > max || word_offset < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_TSTBR14,
                offset: offset as u64,
                value: value as i128,
                max_value: (max << 2) as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM14_MASK) | encode_imm14(word_offset);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    // =======================================================================
    // MOVW immediate relocation application
    // =======================================================================

    /// Applies a MOVW immediate relocation for MOVZ/MOVK/MOVN instructions.
    ///
    /// Used for `R_AARCH64_MOVW_UABS_G{0..3}` (unsigned) and
    /// `R_AARCH64_MOVW_SABS_G{0..2}` (signed) relocation types.
    ///
    /// Extracts a 16-bit slice at position `(value >> shift) & 0xFFFF` and
    /// encodes it into the instruction's `imm16` field (bits [20:5]).
    ///
    /// # Parameters
    ///
    /// - `shift`: Bit position of the 16-bit slice (0, 16, 32, or 48).
    pub fn apply_movw(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
        shift: u32,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let imm16_value = ((value as u64) >> shift) & 0xFFFF;
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM16_MASK) | encode_imm16(imm16_value);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    // =======================================================================
    // GOT-relative relocation application
    // =======================================================================

    /// Applies an ADRP-to-GOT-entry relocation (`R_AARCH64_ADR_GOT_PAGE`).
    ///
    /// Computes `(Page(GOT_entry) - Page(P)) >> 12` and encodes into the ADRP
    /// instruction's `immhi:immlo` fields. This is identical in encoding to
    /// [`apply_adr_prel_pg_hi21`] but targets the GOT entry address instead
    /// of the symbol address directly.
    pub fn apply_adr_got_page(
        &self,
        data: &mut [u8],
        offset: usize,
        got_entry_addr: u64,
        pc: u64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let page_delta =
            (page(got_entry_addr) as i64).wrapping_sub(page(pc) as i64) >> 12;
        // Range check: signed 21-bit
        let max = (1_i64 << 20) - 1;
        let min = -(1_i64 << 20);
        if page_delta > max || page_delta < min {
            return Err(RelocationError::Overflow {
                reloc_type: R_AARCH64_ADR_GOT_PAGE,
                offset: offset as u64,
                value: page_delta as i128,
                max_value: max as i128,
            });
        }
        let insn = read_u32_le(data, offset);
        let patched = (insn & !ADRP_IMM_MASK) | encode_adrp_immhi_immlo(page_delta);
        write_u32_le(data, offset, patched);
        Ok(())
    }

    /// Applies a GOT-entry LDR relocation (`R_AARCH64_LD64_GOT_LO12_NC`).
    ///
    /// Extracts the low 12 bits of the GOT entry address, right-shifts by 3
    /// (8-byte aligned LDR), and encodes into the LDR instruction's `imm12`
    /// field (bits [21:10]). No overflow check (`_NC`).
    pub fn apply_ld64_got_lo12_nc(
        &self,
        data: &mut [u8],
        offset: usize,
        got_entry_addr: u64,
    ) -> Result<(), RelocationError> {
        if offset + 4 > data.len() {
            return Err(RelocationError::InvalidOffset {
                offset: offset as u64,
                section_size: data.len() as u64,
            });
        }
        let imm12_value = ((got_entry_addr & 0xFFF) >> 3) as u32;
        let insn = read_u32_le(data, offset);
        let patched = (insn & !IMM12_MASK) | encode_imm12(imm12_value);
        write_u32_le(data, offset, patched);
        Ok(())
    }
}

// ===========================================================================
// ArchRelocationHandler trait implementation
// ===========================================================================

impl ArchRelocationHandler for AArch64RelocationHandler {
    /// Applies a single AArch64 relocation to the output data buffer.
    ///
    /// Dispatches to the appropriate application function based on the
    /// relocation type code. For each type, the correct formula is applied
    /// (absolute, PC-relative, page-relative, etc.) and the result is
    /// encoded into the target instruction or data field.
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), RelocationError> {
        let offset = reloc.offset as usize;
        let s = reloc.symbol_value;
        let a = reloc.addend;
        let p = reloc.offset; // Virtual address of the relocation site

        match reloc.reloc_type {
            R_AARCH64_NONE => Ok(()),

            // ---------------------------------------------------------------
            // Absolute data relocations: value = S + A
            // ---------------------------------------------------------------
            R_AARCH64_ABS64 => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_ABS32 => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs32(output_data, offset, value)
            }
            R_AARCH64_ABS16 => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs16(output_data, offset, value)
            }

            // ---------------------------------------------------------------
            // PC-relative data relocations: value = S + A - P
            // ---------------------------------------------------------------
            R_AARCH64_PREL64 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_prel64(output_data, offset, value)
            }
            R_AARCH64_PREL32 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_prel32(output_data, offset, value)
            }
            R_AARCH64_PREL16 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                if value > i16::MAX as i64 || value < i16::MIN as i64 {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_PREL16,
                        offset: p,
                        value: value as i128,
                        max_value: i16::MAX as i128,
                    });
                }
                if offset + 2 > output_data.len() {
                    return Err(RelocationError::InvalidOffset {
                        offset: p,
                        section_size: output_data.len() as u64,
                    });
                }
                output_data[offset] = value as u8;
                output_data[offset + 1] = (value >> 8) as u8;
                Ok(())
            }

            // ---------------------------------------------------------------
            // ADRP page-relative: value = (Page(S+A) - Page(P)) >> 12
            // ---------------------------------------------------------------
            R_AARCH64_ADR_PREL_PG_HI21 => {
                let target_addr = s.wrapping_add(a as u64);
                let page_delta =
                    (page(target_addr) as i64).wrapping_sub(page(p) as i64) >> 12;
                self.apply_adr_prel_pg_hi21(output_data, offset, page_delta)
            }
            R_AARCH64_ADR_PREL_PG_HI21_NC => {
                // Same as HI21 but without overflow check
                let target_addr = s.wrapping_add(a as u64);
                let page_delta =
                    (page(target_addr) as i64).wrapping_sub(page(p) as i64) >> 12;
                if offset + 4 > output_data.len() {
                    return Err(RelocationError::InvalidOffset {
                        offset: p,
                        section_size: output_data.len() as u64,
                    });
                }
                let insn = read_u32_le(output_data, offset);
                let patched =
                    (insn & !ADRP_IMM_MASK) | encode_adrp_immhi_immlo(page_delta);
                write_u32_le(output_data, offset, patched);
                Ok(())
            }

            // ---------------------------------------------------------------
            // ADD/LDR low 12 bits: value = (S + A) & 0xFFF, possibly scaled
            // ---------------------------------------------------------------
            R_AARCH64_ADD_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_add_abs_lo12_nc(output_data, offset, value)
            }
            R_AARCH64_LDST8_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_ldst_lo12(output_data, offset, value, 0)
            }
            R_AARCH64_LDST16_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_ldst_lo12(output_data, offset, value, 1)
            }
            R_AARCH64_LDST32_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_ldst_lo12(output_data, offset, value, 2)
            }
            R_AARCH64_LDST64_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_ldst_lo12(output_data, offset, value, 3)
            }
            R_AARCH64_LDST128_ABS_LO12_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_ldst_lo12(output_data, offset, value, 4)
            }

            // ---------------------------------------------------------------
            // Branch relocations: value = S + A - P (then >> 2 internally)
            // ---------------------------------------------------------------
            R_AARCH64_CALL26 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_call26(output_data, offset, value)
            }
            R_AARCH64_JUMP26 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_jump26(output_data, offset, value)
            }
            R_AARCH64_CONDBR19 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_condbr19(output_data, offset, value)
            }
            R_AARCH64_TSTBR14 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_tstbr14(output_data, offset, value)
            }

            // ---------------------------------------------------------------
            // MOVW unsigned immediate: 16-bit slices of S + A
            // ---------------------------------------------------------------
            R_AARCH64_MOVW_UABS_G0 => {
                let value = s.wrapping_add(a as u64) as i64;
                // Overflow check: value must fit in 16 bits (bits [15:0])
                if (value as u64) > 0xFFFF {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_UABS_G0,
                        offset: p,
                        value: value as i128,
                        max_value: 0xFFFF,
                    });
                }
                self.apply_movw(output_data, offset, value, 0)
            }
            R_AARCH64_MOVW_UABS_G0_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_movw(output_data, offset, value, 0)
            }
            R_AARCH64_MOVW_UABS_G1 => {
                let value = s.wrapping_add(a as u64) as i64;
                if (value as u64) > 0xFFFF_FFFF {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_UABS_G1,
                        offset: p,
                        value: value as i128,
                        max_value: 0xFFFF_FFFF,
                    });
                }
                self.apply_movw(output_data, offset, value, 16)
            }
            R_AARCH64_MOVW_UABS_G1_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_movw(output_data, offset, value, 16)
            }
            R_AARCH64_MOVW_UABS_G2 => {
                let value = s.wrapping_add(a as u64) as i64;
                if (value as u64) > 0xFFFF_FFFF_FFFF {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_UABS_G2,
                        offset: p,
                        value: value as i128,
                        max_value: 0xFFFF_FFFF_FFFF,
                    });
                }
                self.apply_movw(output_data, offset, value, 32)
            }
            R_AARCH64_MOVW_UABS_G2_NC => {
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_movw(output_data, offset, value, 32)
            }
            R_AARCH64_MOVW_UABS_G3 => {
                // G3 covers bits [63:48] — always fits for 64-bit values
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_movw(output_data, offset, value, 48)
            }

            // ---------------------------------------------------------------
            // MOVW signed immediate: 16-bit slices of S + A (signed)
            // ---------------------------------------------------------------
            R_AARCH64_MOVW_SABS_G0 => {
                let value = (s as i64).wrapping_add(a);
                // Signed check: value must fit in signed 16-bit
                if value > i16::MAX as i64 || value < i16::MIN as i64 {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_SABS_G0,
                        offset: p,
                        value: value as i128,
                        max_value: i16::MAX as i128,
                    });
                }
                self.apply_movw(output_data, offset, value, 0)
            }
            R_AARCH64_MOVW_SABS_G1 => {
                let value = (s as i64).wrapping_add(a);
                if value > i32::MAX as i64 || value < i32::MIN as i64 {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_SABS_G1,
                        offset: p,
                        value: value as i128,
                        max_value: i32::MAX as i128,
                    });
                }
                self.apply_movw(output_data, offset, value, 16)
            }
            R_AARCH64_MOVW_SABS_G2 => {
                let value = (s as i64).wrapping_add(a);
                // Signed 48-bit range
                let max = (1_i64 << 47) - 1;
                let min = -(1_i64 << 47);
                if value > max || value < min {
                    return Err(RelocationError::Overflow {
                        reloc_type: R_AARCH64_MOVW_SABS_G2,
                        offset: p,
                        value: value as i128,
                        max_value: max as i128,
                    });
                }
                self.apply_movw(output_data, offset, value, 32)
            }

            // ---------------------------------------------------------------
            // GOT-relative relocations (PIC)
            // ---------------------------------------------------------------
            R_AARCH64_ADR_GOT_PAGE => {
                // got_address is the base of .got; we compute the GOT entry
                // address as got_address + symbol_offset_in_got. For the
                // simple case where the caller provides got_address as the
                // GOT entry address for this symbol, we use it directly.
                // The linker framework resolves the GOT entry address.
                let got_entry = got_address.wrapping_add(a as u64);
                self.apply_adr_got_page(output_data, offset, got_entry, p)
            }
            R_AARCH64_LD64_GOT_LO12_NC => {
                let got_entry = got_address.wrapping_add(a as u64);
                self.apply_ld64_got_lo12_nc(output_data, offset, got_entry)
            }

            // ---------------------------------------------------------------
            // TLS relocations — handled similarly to GOT-relative
            // ---------------------------------------------------------------
            R_AARCH64_TLSGD_ADR_PAGE21 => {
                // TLS GD: ADRP to the GOT TLS descriptor page
                let target_addr = got_address.wrapping_add(a as u64);
                let page_delta =
                    (page(target_addr) as i64).wrapping_sub(page(p) as i64) >> 12;
                self.apply_adr_prel_pg_hi21(output_data, offset, page_delta)
            }
            R_AARCH64_TLSGD_ADD_LO12_NC => {
                // TLS GD: ADD low 12 of GOT TLS descriptor entry
                let target_addr = got_address.wrapping_add(a as u64);
                self.apply_add_abs_lo12_nc(output_data, offset, target_addr as i64)
            }
            R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => {
                // TLS IE: ADRP to GOT TP offset page
                let target_addr = got_address.wrapping_add(a as u64);
                let page_delta =
                    (page(target_addr) as i64).wrapping_sub(page(p) as i64) >> 12;
                self.apply_adr_prel_pg_hi21(output_data, offset, page_delta)
            }
            R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => {
                // TLS IE: LDR from GOT TP offset, scaled by 8
                let target_addr = got_address.wrapping_add(a as u64);
                self.apply_ld64_got_lo12_nc(output_data, offset, target_addr)
            }
            R_AARCH64_TLSLE_ADD_TPREL_HI12 => {
                // TLS LE: ADD high 12 of TP offset — encode bits [23:12]
                let tp_offset = s.wrapping_add(a as u64);
                let hi12 = ((tp_offset >> 12) & 0xFFF) as u32;
                if offset + 4 > output_data.len() {
                    return Err(RelocationError::InvalidOffset {
                        offset: p,
                        section_size: output_data.len() as u64,
                    });
                }
                let insn = read_u32_le(output_data, offset);
                let patched = (insn & !IMM12_MASK) | encode_imm12(hi12);
                write_u32_le(output_data, offset, patched);
                Ok(())
            }
            R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => {
                // TLS LE: ADD low 12 of TP offset, no overflow check
                let tp_offset = s.wrapping_add(a as u64);
                self.apply_add_abs_lo12_nc(output_data, offset, tp_offset as i64)
            }

            // ---------------------------------------------------------------
            // Dynamic relocations — typically handled by the runtime linker,
            // but we fill in the initial values for the linker output.
            // ---------------------------------------------------------------
            R_AARCH64_GLOB_DAT => {
                // GOT entry: set to S (symbol virtual address)
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_JUMP_SLOT => {
                // PLT jump slot: set to S (resolved at load time, but we
                // fill in the initial PLT stub target)
                let value = plt_address.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_RELATIVE => {
                // Base-relative: B + A (B = base address, stored in symbol_value
                // by the linker framework; addend from the relocation entry)
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_COPY => {
                // Copy relocation: the runtime linker copies the symbol data.
                // Nothing to patch at static link time.
                Ok(())
            }
            R_AARCH64_TLS_DTPMOD64 => {
                // TLS module ID — filled at runtime; write placeholder
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_TLS_DTPREL64 => {
                // TLS module offset — filled at runtime; write symbol+addend
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_TLS_TPREL64 => {
                // TLS TP-relative offset
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }
            R_AARCH64_TLSDESC => {
                // TLS descriptor — write symbol+addend as placeholder
                let value = s.wrapping_add(a as u64) as i64;
                self.apply_abs64(output_data, offset, value)
            }

            // ---------------------------------------------------------------
            // Unsupported relocation type
            // ---------------------------------------------------------------
            _ => Err(RelocationError::UnsupportedType {
                reloc_type: reloc.reloc_type,
            }),
        }
    }

    /// Returns a human-readable name for the given AArch64 relocation type.
    fn relocation_name(&self, reloc_type: u32) -> &'static str {
        match reloc_type {
            R_AARCH64_NONE => "R_AARCH64_NONE",
            R_AARCH64_ABS64 => "R_AARCH64_ABS64",
            R_AARCH64_ABS32 => "R_AARCH64_ABS32",
            R_AARCH64_ABS16 => "R_AARCH64_ABS16",
            R_AARCH64_PREL64 => "R_AARCH64_PREL64",
            R_AARCH64_PREL32 => "R_AARCH64_PREL32",
            R_AARCH64_PREL16 => "R_AARCH64_PREL16",
            R_AARCH64_MOVW_UABS_G0 => "R_AARCH64_MOVW_UABS_G0",
            R_AARCH64_MOVW_UABS_G0_NC => "R_AARCH64_MOVW_UABS_G0_NC",
            R_AARCH64_MOVW_UABS_G1 => "R_AARCH64_MOVW_UABS_G1",
            R_AARCH64_MOVW_UABS_G1_NC => "R_AARCH64_MOVW_UABS_G1_NC",
            R_AARCH64_MOVW_UABS_G2 => "R_AARCH64_MOVW_UABS_G2",
            R_AARCH64_MOVW_UABS_G2_NC => "R_AARCH64_MOVW_UABS_G2_NC",
            R_AARCH64_MOVW_UABS_G3 => "R_AARCH64_MOVW_UABS_G3",
            R_AARCH64_MOVW_SABS_G0 => "R_AARCH64_MOVW_SABS_G0",
            R_AARCH64_MOVW_SABS_G1 => "R_AARCH64_MOVW_SABS_G1",
            R_AARCH64_MOVW_SABS_G2 => "R_AARCH64_MOVW_SABS_G2",
            R_AARCH64_ADR_PREL_PG_HI21 => "R_AARCH64_ADR_PREL_PG_HI21",
            R_AARCH64_ADR_PREL_PG_HI21_NC => "R_AARCH64_ADR_PREL_PG_HI21_NC",
            R_AARCH64_ADD_ABS_LO12_NC => "R_AARCH64_ADD_ABS_LO12_NC",
            R_AARCH64_LDST8_ABS_LO12_NC => "R_AARCH64_LDST8_ABS_LO12_NC",
            R_AARCH64_LDST16_ABS_LO12_NC => "R_AARCH64_LDST16_ABS_LO12_NC",
            R_AARCH64_LDST32_ABS_LO12_NC => "R_AARCH64_LDST32_ABS_LO12_NC",
            R_AARCH64_LDST64_ABS_LO12_NC => "R_AARCH64_LDST64_ABS_LO12_NC",
            R_AARCH64_LDST128_ABS_LO12_NC => "R_AARCH64_LDST128_ABS_LO12_NC",
            R_AARCH64_TSTBR14 => "R_AARCH64_TSTBR14",
            R_AARCH64_CONDBR19 => "R_AARCH64_CONDBR19",
            R_AARCH64_JUMP26 => "R_AARCH64_JUMP26",
            R_AARCH64_CALL26 => "R_AARCH64_CALL26",
            R_AARCH64_ADR_GOT_PAGE => "R_AARCH64_ADR_GOT_PAGE",
            R_AARCH64_LD64_GOT_LO12_NC => "R_AARCH64_LD64_GOT_LO12_NC",
            R_AARCH64_TLSGD_ADR_PAGE21 => "R_AARCH64_TLSGD_ADR_PAGE21",
            R_AARCH64_TLSGD_ADD_LO12_NC => "R_AARCH64_TLSGD_ADD_LO12_NC",
            R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => "R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21",
            R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC",
            R_AARCH64_TLSLE_ADD_TPREL_HI12 => "R_AARCH64_TLSLE_ADD_TPREL_HI12",
            R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC",
            R_AARCH64_COPY => "R_AARCH64_COPY",
            R_AARCH64_GLOB_DAT => "R_AARCH64_GLOB_DAT",
            R_AARCH64_JUMP_SLOT => "R_AARCH64_JUMP_SLOT",
            R_AARCH64_RELATIVE => "R_AARCH64_RELATIVE",
            R_AARCH64_TLS_DTPMOD64 => "R_AARCH64_TLS_DTPMOD64",
            R_AARCH64_TLS_DTPREL64 => "R_AARCH64_TLS_DTPREL64",
            R_AARCH64_TLS_TPREL64 => "R_AARCH64_TLS_TPREL64",
            R_AARCH64_TLSDESC => "R_AARCH64_TLSDESC",
            _ => "R_AARCH64_UNKNOWN",
        }
    }

    /// Returns `true` if the relocation type requires a GOT entry.
    ///
    /// GOT entries are needed for:
    /// - `R_AARCH64_ADR_GOT_PAGE` (ADRP to GOT entry page)
    /// - `R_AARCH64_LD64_GOT_LO12_NC` (LDR from GOT entry)
    /// - `R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21` (TLS IE ADRP)
    /// - `R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC` (TLS IE LDR)
    /// - `R_AARCH64_TLSGD_ADR_PAGE21` (TLS GD ADRP)
    /// - `R_AARCH64_TLSGD_ADD_LO12_NC` (TLS GD ADD)
    fn needs_got_entry(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_AARCH64_ADR_GOT_PAGE
                | R_AARCH64_LD64_GOT_LO12_NC
                | R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
                | R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
                | R_AARCH64_TLSGD_ADR_PAGE21
                | R_AARCH64_TLSGD_ADD_LO12_NC
        )
    }

    /// Returns `true` if the relocation type requires a PLT entry.
    ///
    /// PLT entries are needed for `R_AARCH64_CALL26` and `R_AARCH64_JUMP26`
    /// when targeting external/preemptible symbols in PIC mode. The linker
    /// framework determines preemptibility; this function reports the
    /// relocation types that *may* need PLT entries.
    fn needs_plt_entry(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_AARCH64_CALL26 | R_AARCH64_JUMP26
        )
    }

    /// Returns `true` if the relocation type computes a PC-relative value.
    ///
    /// PC-relative relocations use the formula: `value = S + A - P`.
    fn is_pc_relative(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_AARCH64_ADR_PREL_PG_HI21
                | R_AARCH64_ADR_PREL_PG_HI21_NC
                | R_AARCH64_CALL26
                | R_AARCH64_JUMP26
                | R_AARCH64_CONDBR19
                | R_AARCH64_TSTBR14
                | R_AARCH64_PREL32
                | R_AARCH64_PREL64
                | R_AARCH64_PREL16
                | R_AARCH64_ADR_GOT_PAGE
                | R_AARCH64_TLSGD_ADR_PAGE21
                | R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
        )
    }

    /// Returns the size in bytes of the relocation field.
    ///
    /// - 8 bytes: `ABS64`, `PREL64`, `GLOB_DAT`, `JUMP_SLOT`, `RELATIVE`,
    ///   `TLS_DTPMOD64`, `TLS_DTPREL64`, `TLS_TPREL64`, `TLSDESC`
    /// - 4 bytes: All instruction-level relocations (ADRP, ADD, LDR, B, BL,
    ///   B.cond, TBZ, MOVW, etc.) and `ABS32`, `PREL32`
    /// - 2 bytes: `ABS16`, `PREL16`
    fn relocation_size(&self, reloc_type: u32) -> u8 {
        match reloc_type {
            R_AARCH64_ABS64
            | R_AARCH64_PREL64
            | R_AARCH64_GLOB_DAT
            | R_AARCH64_JUMP_SLOT
            | R_AARCH64_RELATIVE
            | R_AARCH64_TLS_DTPMOD64
            | R_AARCH64_TLS_DTPREL64
            | R_AARCH64_TLS_TPREL64
            | R_AARCH64_TLSDESC => 8,

            R_AARCH64_ABS16 | R_AARCH64_PREL16 => 2,

            R_AARCH64_NONE => 0,

            // All instruction-level relocations (32-bit A64 instructions)
            // and ABS32/PREL32 are 4 bytes
            _ => 4,
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_alignment() {
        assert_eq!(page(0x0000_0000), 0x0000_0000);
        assert_eq!(page(0x0000_0FFF), 0x0000_0000);
        assert_eq!(page(0x0000_1000), 0x0000_1000);
        assert_eq!(page(0x0000_1ABC), 0x0000_1000);
        assert_eq!(page(0xFFFF_FFFF_FFFF_FFFF), 0xFFFF_FFFF_FFFF_F000);
    }

    #[test]
    fn test_sign_extend() {
        // 21-bit sign extension
        assert_eq!(sign_extend(0x0F_FFFF, 21), 0x0F_FFFF);
        assert_eq!(sign_extend(0x10_0000, 21), -0x10_0000_i64);
        assert_eq!(sign_extend(0x1F_FFFF, 21), -1);

        // 14-bit sign extension
        assert_eq!(sign_extend(0x1FFF, 14), 0x1FFF);
        assert_eq!(sign_extend(0x2000, 14), -0x2000_i64);

        // Edge cases
        assert_eq!(sign_extend(0, 1), 0);
        assert_eq!(sign_extend(1, 1), -1);
        assert_eq!(sign_extend(42, 0), 42); // bits=0 returns as-is
    }

    #[test]
    fn test_encode_adrp_immhi_immlo() {
        // Value 0 → both fields zero
        assert_eq!(encode_adrp_immhi_immlo(0), 0);

        // Value 1 → immlo = 1 << 29
        assert_eq!(encode_adrp_immhi_immlo(1), 1 << 29);

        // Value 3 → immlo = 3 << 29
        assert_eq!(encode_adrp_immhi_immlo(3), 3u32 << 29);

        // Value 4 → immhi bit 0 set (bit 5 of instruction)
        assert_eq!(encode_adrp_immhi_immlo(4), 1 << 5);

        // Value 0x1FFFFF (max positive 21-bit) → all bits set
        let encoded = encode_adrp_immhi_immlo(0x1FFFFF);
        let immlo = (encoded >> 29) & 0x3;
        let immhi = (encoded >> 5) & 0x7FFFF;
        assert_eq!(immlo, 3);
        assert_eq!(immhi, 0x7FFFF);
    }

    #[test]
    fn test_encode_imm12() {
        assert_eq!(encode_imm12(0), 0);
        assert_eq!(encode_imm12(1), 1 << 10);
        assert_eq!(encode_imm12(0xFFF), 0xFFF << 10);
        // Mask should only keep low 12 bits
        assert_eq!(encode_imm12(0x1FFF), 0xFFF << 10);
    }

    #[test]
    fn test_encode_imm26() {
        assert_eq!(encode_imm26(0), 0);
        assert_eq!(encode_imm26(1), 1);
        assert_eq!(encode_imm26(0x03FF_FFFF), 0x03FF_FFFF);
        // Mask should only keep low 26 bits
        assert_eq!(encode_imm26(0x0FFF_FFFF), 0x03FF_FFFF);
    }

    #[test]
    fn test_read_write_u32_le() {
        let mut data = [0u8; 8];
        write_u32_le(&mut data, 0, 0xDEAD_BEEF);
        assert_eq!(read_u32_le(&data, 0), 0xDEAD_BEEF);
        assert_eq!(data[0], 0xEF);
        assert_eq!(data[1], 0xBE);
        assert_eq!(data[2], 0xAD);
        assert_eq!(data[3], 0xDE);

        write_u32_le(&mut data, 4, 0x1234_5678);
        assert_eq!(read_u32_le(&data, 4), 0x1234_5678);
    }

    #[test]
    fn test_read_write_u64_le() {
        let mut data = [0u8; 16];
        write_u64_le(&mut data, 0, 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(read_u64_le(&data, 0), 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(data[0], 0xBE);
        assert_eq!(data[1], 0xBA);
    }

    // -----------------------------------------------------------------------
    // AArch64RelocationHandler construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_handler_creation() {
        let handler = AArch64RelocationHandler::new();
        assert_eq!(handler.relocation_name(R_AARCH64_CALL26), "R_AARCH64_CALL26");
    }

    // -----------------------------------------------------------------------
    // Relocation name tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relocation_names() {
        let handler = AArch64RelocationHandler::new();
        assert_eq!(handler.relocation_name(R_AARCH64_NONE), "R_AARCH64_NONE");
        assert_eq!(handler.relocation_name(R_AARCH64_ABS64), "R_AARCH64_ABS64");
        assert_eq!(
            handler.relocation_name(R_AARCH64_ADR_PREL_PG_HI21),
            "R_AARCH64_ADR_PREL_PG_HI21"
        );
        assert_eq!(
            handler.relocation_name(R_AARCH64_ADD_ABS_LO12_NC),
            "R_AARCH64_ADD_ABS_LO12_NC"
        );
        assert_eq!(handler.relocation_name(R_AARCH64_CALL26), "R_AARCH64_CALL26");
        assert_eq!(handler.relocation_name(R_AARCH64_JUMP26), "R_AARCH64_JUMP26");
        assert_eq!(handler.relocation_name(R_AARCH64_GLOB_DAT), "R_AARCH64_GLOB_DAT");
        assert_eq!(handler.relocation_name(9999), "R_AARCH64_UNKNOWN");
    }

    // -----------------------------------------------------------------------
    // GOT/PLT/PC-relative classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_needs_got_entry() {
        let handler = AArch64RelocationHandler::new();
        assert!(handler.needs_got_entry(R_AARCH64_ADR_GOT_PAGE));
        assert!(handler.needs_got_entry(R_AARCH64_LD64_GOT_LO12_NC));
        assert!(handler.needs_got_entry(R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21));
        assert!(handler.needs_got_entry(R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC));
        assert!(!handler.needs_got_entry(R_AARCH64_ABS64));
        assert!(!handler.needs_got_entry(R_AARCH64_CALL26));
    }

    #[test]
    fn test_needs_plt_entry() {
        let handler = AArch64RelocationHandler::new();
        assert!(handler.needs_plt_entry(R_AARCH64_CALL26));
        assert!(handler.needs_plt_entry(R_AARCH64_JUMP26));
        assert!(!handler.needs_plt_entry(R_AARCH64_ABS64));
        assert!(!handler.needs_plt_entry(R_AARCH64_ADR_PREL_PG_HI21));
    }

    #[test]
    fn test_is_pc_relative() {
        let handler = AArch64RelocationHandler::new();
        assert!(handler.is_pc_relative(R_AARCH64_ADR_PREL_PG_HI21));
        assert!(handler.is_pc_relative(R_AARCH64_CALL26));
        assert!(handler.is_pc_relative(R_AARCH64_JUMP26));
        assert!(handler.is_pc_relative(R_AARCH64_CONDBR19));
        assert!(handler.is_pc_relative(R_AARCH64_TSTBR14));
        assert!(handler.is_pc_relative(R_AARCH64_PREL32));
        assert!(handler.is_pc_relative(R_AARCH64_PREL64));
        assert!(handler.is_pc_relative(R_AARCH64_ADR_GOT_PAGE));
        assert!(!handler.is_pc_relative(R_AARCH64_ABS64));
        assert!(!handler.is_pc_relative(R_AARCH64_ADD_ABS_LO12_NC));
    }

    // -----------------------------------------------------------------------
    // Relocation size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relocation_sizes() {
        let handler = AArch64RelocationHandler::new();
        assert_eq!(handler.relocation_size(R_AARCH64_ABS64), 8);
        assert_eq!(handler.relocation_size(R_AARCH64_PREL64), 8);
        assert_eq!(handler.relocation_size(R_AARCH64_GLOB_DAT), 8);
        assert_eq!(handler.relocation_size(R_AARCH64_ABS32), 4);
        assert_eq!(handler.relocation_size(R_AARCH64_PREL32), 4);
        assert_eq!(handler.relocation_size(R_AARCH64_CALL26), 4);
        assert_eq!(handler.relocation_size(R_AARCH64_ADR_PREL_PG_HI21), 4);
        assert_eq!(handler.relocation_size(R_AARCH64_ABS16), 2);
        assert_eq!(handler.relocation_size(R_AARCH64_PREL16), 2);
        assert_eq!(handler.relocation_size(R_AARCH64_NONE), 0);
    }

    // -----------------------------------------------------------------------
    // Absolute relocation application tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_abs64() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        handler
            .apply_abs64(&mut data, 0, 0x0102_0304_0506_0708)
            .unwrap();
        assert_eq!(read_u64_le(&data, 0), 0x0102_0304_0506_0708);
    }

    #[test]
    fn test_apply_abs32_success() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        handler.apply_abs32(&mut data, 0, 0x1234_5678).unwrap();
        assert_eq!(read_u32_le(&data, 0), 0x1234_5678);
    }

    #[test]
    fn test_apply_abs32_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        let result = handler.apply_abs32(&mut data, 0, 0x1_0000_0000);
        assert!(result.is_err());
        if let Err(RelocationError::Overflow { reloc_type, .. }) = result {
            assert_eq!(reloc_type, R_AARCH64_ABS32);
        }
    }

    #[test]
    fn test_apply_abs16_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        let result = handler.apply_abs16(&mut data, 0, 0x1_0000);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // PC-relative relocation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_prel32_success() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        handler.apply_prel32(&mut data, 0, 256).unwrap();
        assert_eq!(read_u32_le(&data, 0), 256);
    }

    #[test]
    fn test_apply_prel32_negative() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        handler.apply_prel32(&mut data, 0, -128).unwrap();
        assert_eq!(read_u32_le(&data, 0), (-128_i32) as u32);
    }

    #[test]
    fn test_apply_prel32_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        let result = handler.apply_prel32(&mut data, 0, 0x8000_0000_i64);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ADRP page-relative tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_adr_prel_pg_hi21_zero_delta() {
        let handler = AArch64RelocationHandler::new();
        // NOP instruction (0xD503201F) as placeholder
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9000_0000); // ADRP X0, #0
        handler.apply_adr_prel_pg_hi21(&mut data, 0, 0).unwrap();
        let insn = read_u32_le(&data, 0);
        // immhi and immlo should be zero, opcode preserved
        assert_eq!(insn & ADRP_IMM_MASK, 0);
        assert_eq!(insn & !ADRP_IMM_MASK, 0x9000_0000);
    }

    #[test]
    fn test_apply_adr_prel_pg_hi21_positive() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9000_0000); // ADRP X0, #0
        // page_delta = 1 (one page forward)
        handler.apply_adr_prel_pg_hi21(&mut data, 0, 1).unwrap();
        let insn = read_u32_le(&data, 0);
        // immlo = 1 << 29
        assert_eq!(insn & ADRP_IMMLO_MASK, 1 << 29);
    }

    #[test]
    fn test_apply_adr_prel_pg_hi21_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9000_0000);
        // Exceeds signed 21-bit range
        let result = handler.apply_adr_prel_pg_hi21(&mut data, 0, 1 << 20);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ADD low-12 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_add_abs_lo12_nc() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9100_0000); // ADD X0, X0, #0
        handler.apply_add_abs_lo12_nc(&mut data, 0, 0x123).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm12 = (insn & IMM12_MASK) >> 10;
        assert_eq!(imm12, 0x123);
    }

    // -----------------------------------------------------------------------
    // Load/store low-12 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_ldst64_lo12() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0xF940_0000); // LDR X0, [X0, #0]
        // Address 0x1F8: low 12 = 0x1F8, shifted by 3 = 0x3F
        handler.apply_ldst_lo12(&mut data, 0, 0x1F8, 3).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm12 = (insn & IMM12_MASK) >> 10;
        assert_eq!(imm12, 0x3F);
    }

    #[test]
    fn test_apply_ldst32_lo12() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0xB940_0000); // LDR W0, [X0, #0]
        // Address 0x100: low 12 = 0x100, shifted by 2 = 0x40
        handler.apply_ldst_lo12(&mut data, 0, 0x100, 2).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm12 = (insn & IMM12_MASK) >> 10;
        assert_eq!(imm12, 0x40);
    }

    // -----------------------------------------------------------------------
    // Branch relocation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_call26_forward() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9400_0000); // BL #0
        // Forward branch by 256 bytes (64 instructions)
        handler.apply_call26(&mut data, 0, 256).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm26 = insn & IMM26_MASK;
        assert_eq!(imm26, 64); // 256 / 4 = 64
    }

    #[test]
    fn test_apply_call26_backward() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9400_0000); // BL #0
        // Backward branch by 256 bytes
        handler.apply_call26(&mut data, 0, -256).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm26 = insn & IMM26_MASK;
        // -64 as 26-bit unsigned = 0x03FF_FFC0
        assert_eq!(imm26, (-64_i32 as u32) & IMM26_MASK);
    }

    #[test]
    fn test_apply_call26_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9400_0000);
        // ±128 MiB = ±0x800_0000 bytes; exceed it
        let result = handler.apply_call26(&mut data, 0, 0x800_0004);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_jump26() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x1400_0000); // B #0
        handler.apply_jump26(&mut data, 0, 128).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm26 = insn & IMM26_MASK;
        assert_eq!(imm26, 32); // 128 / 4 = 32
    }

    #[test]
    fn test_apply_condbr19() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x5400_0000); // B.EQ #0
        handler.apply_condbr19(&mut data, 0, 64).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm19 = (insn & IMM19_MASK) >> 5;
        assert_eq!(imm19, 16); // 64 / 4 = 16
    }

    #[test]
    fn test_apply_condbr19_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x5400_0000);
        // ±1 MiB = ±0x10_0000; exceed it
        let result = handler.apply_condbr19(&mut data, 0, 0x10_0004);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_tstbr14() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x3600_0000); // TBZ X0, #0, #0
        handler.apply_tstbr14(&mut data, 0, 32).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm14 = (insn & IMM14_MASK) >> 5;
        assert_eq!(imm14, 8); // 32 / 4 = 8
    }

    #[test]
    fn test_apply_tstbr14_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x3600_0000);
        // ±32 KiB = ±0x8000; exceed it
        let result = handler.apply_tstbr14(&mut data, 0, 0x8004);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // MOVW immediate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_movw_g0() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0xD280_0000); // MOVZ X0, #0
        handler.apply_movw(&mut data, 0, 0xABCD, 0).unwrap();
        let insn = read_u32_le(&data, 0);
        let imm16 = (insn & IMM16_MASK) >> 5;
        assert_eq!(imm16, 0xABCD);
    }

    #[test]
    fn test_apply_movw_g1() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0xD2A0_0000); // MOVZ X0, #0, LSL #16
        handler
            .apply_movw(&mut data, 0, 0x1234_0000, 16)
            .unwrap();
        let insn = read_u32_le(&data, 0);
        let imm16 = (insn & IMM16_MASK) >> 5;
        assert_eq!(imm16, 0x1234);
    }

    // -----------------------------------------------------------------------
    // GOT-relative relocation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_adr_got_page() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9000_0000); // ADRP X0, #0
        // GOT entry at page 0x2000, PC at page 0x1000
        // page_delta = (0x2000 - 0x1000) >> 12 = 1
        handler
            .apply_adr_got_page(&mut data, 0, 0x2010, 0x1004)
            .unwrap();
        let insn = read_u32_le(&data, 0);
        // immlo should be 1 (page_delta = 1, bits [1:0] = 01)
        assert_eq!((insn >> 29) & 0x3, 1);
    }

    #[test]
    fn test_apply_ld64_got_lo12_nc() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0xF940_0000); // LDR X0, [X0, #0]
        // GOT entry at address 0x2018 → low 12 = 0x018, >> 3 = 3
        handler
            .apply_ld64_got_lo12_nc(&mut data, 0, 0x2018)
            .unwrap();
        let insn = read_u32_le(&data, 0);
        let imm12 = (insn & IMM12_MASK) >> 10;
        assert_eq!(imm12, 3);
    }

    // -----------------------------------------------------------------------
    // InvalidOffset error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_offset_abs64() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4]; // too small for 8-byte write
        let result = handler.apply_abs64(&mut data, 0, 42);
        assert!(result.is_err());
        if let Err(RelocationError::InvalidOffset { .. }) = result {
            // expected
        } else {
            panic!("Expected InvalidOffset error");
        }
    }

    #[test]
    fn test_invalid_offset_call26() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 2]; // too small for 4-byte read/write
        let result = handler.apply_call26(&mut data, 0, 0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Full trait integration test via apply_relocation
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_relocation_abs64() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let entry = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_ABS64,
            symbol_name: "test_sym".to_string(),
            symbol_value: 0x1000,
            addend: 0x100,
            output_section: 0,
        };
        handler
            .apply_relocation(&entry, &mut data, 0, 0)
            .unwrap();
        assert_eq!(read_u64_le(&data, 0), 0x1100);
    }

    #[test]
    fn test_apply_relocation_call26() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        write_u32_le(&mut data, 0, 0x9400_0000); // BL #0
        let entry = RelocationEntry {
            offset: 0x1000,         // P = 0x1000
            reloc_type: R_AARCH64_CALL26,
            symbol_name: "target".to_string(),
            symbol_value: 0x1100,   // S = 0x1100
            addend: 0,              // A = 0
            output_section: 0,
        };
        // S + A - P = 0x1100 - 0x1000 = 0x100 (256 bytes, 64 instructions)
        handler
            .apply_relocation(&entry, &mut data, 0, 0)
            .unwrap();
        let insn = read_u32_le(&data, 0);
        let imm26 = insn & IMM26_MASK;
        assert_eq!(imm26, 64);
    }

    #[test]
    fn test_apply_relocation_adrp_add_pair() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        // ADRP X0, #0 at offset 0 (P = 0x1000)
        write_u32_le(&mut data, 0, 0x9000_0000);
        // ADD X0, X0, #0 at offset 4 (P = 0x1004)
        write_u32_le(&mut data, 4, 0x9100_0000);

        // Target symbol at 0x3ABC
        let s: u64 = 0x3ABC;
        let p_adrp: u64 = 0x1000;

        // ADRP: Page(0x3ABC) = 0x3000, Page(0x1000) = 0x1000
        // page_delta = (0x3000 - 0x1000) >> 12 = 2
        let adrp_entry = RelocationEntry {
            offset: p_adrp,
            reloc_type: R_AARCH64_ADR_PREL_PG_HI21,
            symbol_name: "sym".to_string(),
            symbol_value: s,
            addend: 0,
            output_section: 0,
        };
        handler
            .apply_relocation(&adrp_entry, &mut data, 0, 0)
            .unwrap();

        // ADD: low 12 of S+A = 0xABC
        let add_entry = RelocationEntry {
            offset: 0x1004,
            reloc_type: R_AARCH64_ADD_ABS_LO12_NC,
            symbol_name: "sym".to_string(),
            symbol_value: s,
            addend: 0,
            output_section: 0,
        };
        handler
            .apply_relocation(&add_entry, &mut data, 0, 0)
            .unwrap();

        // Verify ADRP encoding: page_delta = 2
        let adrp_insn = read_u32_le(&data, 0);
        let immlo = (adrp_insn >> 29) & 0x3;
        let immhi = (adrp_insn >> 5) & 0x7FFFF;
        let page_delta_decoded = (immhi << 2) | immlo;
        assert_eq!(page_delta_decoded, 2);

        // Verify ADD encoding: imm12 = 0xABC
        let add_insn = read_u32_le(&data, 4);
        let imm12 = (add_insn & IMM12_MASK) >> 10;
        assert_eq!(imm12, 0xABC);
    }

    #[test]
    fn test_apply_relocation_none() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 4];
        let entry = RelocationEntry {
            offset: 0,
            reloc_type: R_AARCH64_NONE,
            symbol_name: String::new(),
            symbol_value: 0,
            addend: 0,
            output_section: 0,
        };
        handler
            .apply_relocation(&entry, &mut data, 0, 0)
            .unwrap();
        // Data should be unchanged
        assert_eq!(data, vec![0u8; 4]);
    }

    #[test]
    fn test_apply_relocation_unsupported() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        let entry = RelocationEntry {
            offset: 0,
            reloc_type: 9999,
            symbol_name: "bad".to_string(),
            symbol_value: 0,
            addend: 0,
            output_section: 0,
        };
        let result = handler.apply_relocation(&entry, &mut data, 0, 0);
        assert!(result.is_err());
        if let Err(RelocationError::UnsupportedType { reloc_type }) = result {
            assert_eq!(reloc_type, 9999);
        } else {
            panic!("Expected UnsupportedType error");
        }
    }
}
