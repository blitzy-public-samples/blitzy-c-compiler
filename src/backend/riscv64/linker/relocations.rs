//! RISC-V 64-bit relocation application module for the BCC built-in linker.
//!
//! Implements the [`ArchRelocationHandler`] trait for all `R_RISCV_*` ELF
//! relocation types with linker relaxation support. This module is critical
//! for the standalone backend — BCC links RISC-V 64 object files into
//! executables and shared objects without invoking any external linker.
//!
//! # Relocation Types Handled
//!
//! - **Branch/Jump:** `R_RISCV_BRANCH` (B-type), `R_RISCV_JAL` (J-type)
//! - **Call pairs:** `R_RISCV_CALL`, `R_RISCV_CALL_PLT` (AUIPC+JALR)
//! - **PC-relative addressing:** `R_RISCV_PCREL_HI20`, `R_RISCV_PCREL_LO12_I/S`
//! - **Absolute addressing:** `R_RISCV_HI20`, `R_RISCV_LO12_I/S`
//! - **GOT-relative (PIC):** `R_RISCV_GOT_HI20`
//! - **Data:** `R_RISCV_32`, `R_RISCV_64`, `R_RISCV_ADD*/SUB*`
//! - **Relaxation:** `R_RISCV_RELAX`, `R_RISCV_ALIGN`
//! - **TLS:** `R_RISCV_TLS_*` (thread-local storage)
//! - **Compressed:** `R_RISCV_RVC_BRANCH`, `R_RISCV_RVC_JUMP`
//!
//! # Linker Relaxation
//!
//! Implements RISC-V linker relaxation, critical for kernel code size
//! optimization:
//! - **CALL → JAL:** Replaces 8-byte AUIPC+JALR with 4-byte JAL + NOP
//!   when the target is within ±1 MiB range
//! - **AUIPC+LD → AUIPC+ADDI:** Replaces GOT-indirect loads with direct
//!   addressing when possible
//! - **Alignment adjustment:** Reduces NOP padding after code shrinkage
//!
//! # PCREL_LO12 Pairing
//!
//! `R_RISCV_PCREL_LO12_I/S` relocations reference the address of the
//! corresponding `R_RISCV_PCREL_HI20` AUIPC instruction (not the target
//! symbol). A first-pass HI20 map (`FxHashMap<u64, i64>`) is maintained
//! to resolve these pairings efficiently.

use crate::backend::linker_common::relocation::{
    ArchRelocationHandler, RelocationEntry, RelocationError,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;

// ===========================================================================
// RISC-V Relocation Type Constants (hand-defined, zero-dependency)
// ===========================================================================

/// No relocation.
pub const R_RISCV_NONE: u32 = 0;
/// 32-bit absolute data relocation (S + A).
pub const R_RISCV_32: u32 = 1;
/// 64-bit absolute data relocation (S + A).
pub const R_RISCV_64: u32 = 2;
/// Dynamic relocation: adjust by load base address.
pub const R_RISCV_RELATIVE: u32 = 3;
/// Copy symbol at runtime (dynamic linker).
pub const R_RISCV_COPY: u32 = 4;
/// PLT jump slot (S) — filled by the dynamic linker at runtime.
pub const R_RISCV_JUMP_SLOT: u32 = 5;
/// TLS module ID (32-bit).
pub const R_RISCV_TLS_DTPMOD32: u32 = 6;
/// TLS module ID (64-bit).
pub const R_RISCV_TLS_DTPMOD64: u32 = 7;
/// TLS offset within module (32-bit).
pub const R_RISCV_TLS_DTPREL32: u32 = 8;
/// TLS offset within module (64-bit).
pub const R_RISCV_TLS_DTPREL64: u32 = 9;
/// TLS offset from thread pointer (32-bit).
pub const R_RISCV_TLS_TPREL32: u32 = 10;
/// TLS offset from thread pointer (64-bit).
pub const R_RISCV_TLS_TPREL64: u32 = 11;
/// B-type conditional branch — 13-bit signed PC-relative (S + A - P).
pub const R_RISCV_BRANCH: u32 = 16;
/// J-type unconditional jump — 21-bit signed PC-relative (S + A - P).
pub const R_RISCV_JAL: u32 = 17;
/// AUIPC+JALR pair — 32-bit signed PC-relative call (S + A - P).
pub const R_RISCV_CALL: u32 = 18;
/// AUIPC+JALR pair via PLT — 32-bit signed PC-relative call.
pub const R_RISCV_CALL_PLT: u32 = 19;
/// High 20 bits of GOT entry address, PC-relative (AUIPC).
pub const R_RISCV_GOT_HI20: u32 = 20;
/// TLS Initial-Exec: high 20 bits of GOT TLS entry.
pub const R_RISCV_TLS_GOT_HI20: u32 = 21;
/// TLS General-Dynamic: high 20 bits of GOT TLS descriptor pair.
pub const R_RISCV_TLS_GD_HI20: u32 = 22;
/// High 20 bits PC-relative (AUIPC immediate, S + A - P).
pub const R_RISCV_PCREL_HI20: u32 = 23;
/// Low 12 bits PC-relative I-type (ADDI/LD/LW immediate).
pub const R_RISCV_PCREL_LO12_I: u32 = 24;
/// Low 12 bits PC-relative S-type (SW/SD immediate).
pub const R_RISCV_PCREL_LO12_S: u32 = 25;
/// High 20 bits absolute (LUI immediate, S + A).
pub const R_RISCV_HI20: u32 = 26;
/// Low 12 bits absolute I-type (ADDI immediate).
pub const R_RISCV_LO12_I: u32 = 27;
/// Low 12 bits absolute S-type (SW/SD immediate).
pub const R_RISCV_LO12_S: u32 = 28;
/// TLS Local-Exec: high 20 bits of TP-relative offset.
pub const R_RISCV_TPREL_HI20: u32 = 29;
/// TLS Local-Exec: low 12 bits I-type TP-relative.
pub const R_RISCV_TPREL_LO12_I: u32 = 30;
/// TLS Local-Exec: low 12 bits S-type TP-relative.
pub const R_RISCV_TPREL_LO12_S: u32 = 31;
/// TLS Local-Exec: add thread pointer hint.
pub const R_RISCV_TPREL_ADD: u32 = 32;
/// 8-bit addition (existing value + S + A).
pub const R_RISCV_ADD8: u32 = 33;
/// 16-bit addition (existing value + S + A).
pub const R_RISCV_ADD16: u32 = 34;
/// 32-bit addition (existing value + S + A).
pub const R_RISCV_ADD32: u32 = 35;
/// 64-bit addition (existing value + S + A).
pub const R_RISCV_ADD64: u32 = 36;
/// 8-bit subtraction (existing value - (S + A)).
pub const R_RISCV_SUB8: u32 = 37;
/// 16-bit subtraction (existing value - (S + A)).
pub const R_RISCV_SUB16: u32 = 38;
/// 32-bit subtraction (existing value - (S + A)).
pub const R_RISCV_SUB32: u32 = 39;
/// 64-bit subtraction (existing value - (S + A)).
pub const R_RISCV_SUB64: u32 = 40;
/// GNU C++ vtable hierarchy marker (no-op during linking).
pub const R_RISCV_GNU_VTINHERIT: u32 = 41;
/// GNU C++ vtable member usage marker (no-op during linking).
pub const R_RISCV_GNU_VTENTRY: u32 = 42;
/// Alignment NOP padding (relaxation may reduce NOP count).
pub const R_RISCV_ALIGN: u32 = 43;
/// Compressed branch — CB-type 16-bit, 9-bit signed offset.
pub const R_RISCV_RVC_BRANCH: u32 = 44;
/// Compressed jump — CJ-type 16-bit, 12-bit signed offset.
pub const R_RISCV_RVC_JUMP: u32 = 45;
/// Compressed LUI — C.LUI immediate encoding.
pub const R_RISCV_RVC_LUI: u32 = 46;
/// Relaxation hint — instructs the linker that the adjacent relocation
/// may be relaxed to a shorter instruction sequence.
pub const R_RISCV_RELAX: u32 = 51;
/// 6-bit subtraction.
pub const R_RISCV_SUB6: u32 = 52;
/// 6-bit set.
pub const R_RISCV_SET6: u32 = 53;
/// 8-bit set.
pub const R_RISCV_SET8: u32 = 54;
/// 16-bit set.
pub const R_RISCV_SET16: u32 = 55;
/// 32-bit set.
pub const R_RISCV_SET32: u32 = 56;
/// 32-bit PC-relative data relocation (S + A - P).
pub const R_RISCV_32_PCREL: u32 = 57;

// ---------------------------------------------------------------------------
// Internal instruction constants
// ---------------------------------------------------------------------------

/// RISC-V NOP instruction encoding: `ADDI x0, x0, 0`.
const RISCV_NOP: u32 = 0x0000_0013;

/// RISC-V compressed NOP (C.NOP) encoding.
const RISCV_CNOP: u16 = 0x0001;

/// RISC-V JAL opcode (bits [6:0]).
const OPCODE_JAL: u32 = 0b110_1111;

/// RISC-V AUIPC opcode (bits [6:0]).
const OPCODE_AUIPC: u32 = 0b001_0111;

// ===========================================================================
// Low-Level Byte I/O Helpers
// ===========================================================================

/// Reads a little-endian `u32` from `data` at byte offset `offset`.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Writes a little-endian `u32` to `data` at byte offset `offset`.
#[inline]
fn write_u32_le(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
}

/// Reads a little-endian `u64` from `data` at byte offset `offset`.
#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

/// Writes a little-endian `u64` to `data` at byte offset `offset`.
#[inline]
fn write_u64_le(data: &mut [u8], offset: usize, value: u64) {
    let bytes = value.to_le_bytes();
    data[offset..offset + 8].copy_from_slice(&bytes);
}

/// Reads a little-endian `u16` from `data` at byte offset `offset`.
#[inline]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Writes a little-endian `u16` to `data` at byte offset `offset`.
#[inline]
fn write_u16_le(data: &mut [u8], offset: usize, value: u16) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
}

/// Sign-extends a `u32` value from `bits`-wide two's complement to `i64`.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(sign_extend(0xFFF, 12), -1_i64);
/// assert_eq!(sign_extend(0x7FF, 12),  2047_i64);
/// ```
#[inline]
pub fn sign_extend(value: u32, bits: u32) -> i64 {
    let shift = 32 - bits;
    ((value as i32) << shift >> shift) as i64
}

// ===========================================================================
// RISC-V Instruction Encoding / Decoding Helpers
// ===========================================================================
//
// RISC-V uses six core instruction formats (R/I/S/B/U/J). Relocations
// modify the immediate fields of I, S, B, U, and J formats. Each encoder
// takes a signed integer immediate and returns the bits positioned in their
// final instruction-word locations. Each extractor performs the inverse.

/// Extracts the U-type immediate from an instruction word.
///
/// U-type layout: `imm[31:12]` occupies bits [31:12].
/// Returns the value with lower 12 bits zeroed, sign-extended as `i32`.
#[inline]
pub fn extract_u_imm(instruction: u32) -> i32 {
    (instruction & 0xFFFF_F000) as i32
}

/// Extracts the I-type immediate from an instruction word.
///
/// I-type layout: `imm[11:0]` occupies bits [31:20], sign-extended.
#[inline]
pub fn extract_i_imm(instruction: u32) -> i32 {
    (instruction as i32) >> 20
}

/// Extracts the S-type immediate from an instruction word.
///
/// S-type layout: `imm[11:5]` in bits [31:25], `imm[4:0]` in bits [11:7].
/// Returns sign-extended 12-bit value.
#[inline]
pub fn extract_s_imm(instruction: u32) -> i32 {
    let imm_11_5 = (instruction >> 25) & 0x7F;
    let imm_4_0 = (instruction >> 7) & 0x1F;
    let raw = (imm_11_5 << 5) | imm_4_0;
    // Sign-extend from 12 bits
    ((raw as i32) << 20) >> 20
}

/// Extracts the B-type immediate from an instruction word.
///
/// B-type layout: `imm[12]` at bit 31, `imm[10:5]` at bits [30:25],
/// `imm[4:1]` at bits [11:8], `imm[11]` at bit 7.
/// Returns sign-extended 13-bit value (bit 0 is always zero).
#[inline]
pub fn extract_b_imm(instruction: u32) -> i32 {
    let imm_12 = (instruction >> 31) & 1;
    let imm_11 = (instruction >> 7) & 1;
    let imm_10_5 = (instruction >> 25) & 0x3F;
    let imm_4_1 = (instruction >> 8) & 0xF;
    let raw = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1);
    // Sign-extend from 13 bits
    ((raw as i32) << 19) >> 19
}

/// Extracts the J-type immediate from an instruction word.
///
/// J-type layout: `imm[20]` at bit 31, `imm[10:1]` at bits [30:21],
/// `imm[11]` at bit 20, `imm[19:12]` at bits [19:12].
/// Returns sign-extended 21-bit value (bit 0 is always zero).
#[inline]
pub fn extract_j_imm(instruction: u32) -> i32 {
    let imm_20 = (instruction >> 31) & 1;
    let imm_10_1 = (instruction >> 21) & 0x3FF;
    let imm_11 = (instruction >> 20) & 1;
    let imm_19_12 = (instruction >> 12) & 0xFF;
    let raw = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1);
    // Sign-extend from 21 bits
    ((raw as i32) << 11) >> 11
}

/// Encodes a value into the U-type immediate field (bits [31:12]).
///
/// Only the upper 20 bits of `imm` are placed; lower 12 bits are masked.
#[inline]
fn encode_u_imm(imm: i32) -> u32 {
    (imm as u32) & 0xFFFF_F000
}

/// Encodes a value into the I-type immediate field (bits [31:20]).
///
/// Only the low 12 bits of `imm` are used.
#[inline]
fn encode_i_imm(imm: i32) -> u32 {
    ((imm as u32) & 0xFFF) << 20
}

/// Encodes a value into the S-type immediate split field.
///
/// `imm[11:5]` → bits [31:25], `imm[4:0]` → bits [11:7].
#[inline]
fn encode_s_imm(imm: i32) -> u32 {
    let val = imm as u32;
    (((val >> 5) & 0x7F) << 25) | ((val & 0x1F) << 7)
}

/// Encodes a value into the B-type immediate field.
///
/// `imm[12]` → bit 31, `imm[10:5]` → bits [30:25],
/// `imm[4:1]` → bits [11:8], `imm[11]` → bit 7.
#[inline]
fn encode_b_imm(imm: i32) -> u32 {
    let val = imm as u32;
    (((val >> 12) & 1) << 31)
        | (((val >> 5) & 0x3F) << 25)
        | (((val >> 1) & 0xF) << 8)
        | (((val >> 11) & 1) << 7)
}

/// Encodes a value into the J-type immediate field.
///
/// `imm[20]` → bit 31, `imm[10:1]` → bits [30:21],
/// `imm[11]` → bit 20, `imm[19:12]` → bits [19:12].
#[inline]
fn encode_j_imm(imm: i32) -> u32 {
    let val = imm as u32;
    (((val >> 20) & 1) << 31)
        | (((val >> 1) & 0x3FF) << 21)
        | (((val >> 11) & 1) << 20)
        | (((val >> 12) & 0xFF) << 12)
}

/// Encodes a value into the CB-type (compressed branch) immediate field.
///
/// CB-type (C.BEQZ/C.BNEZ) 16-bit layout:
/// `offset[8]` → bit 12, `offset[4:3]` → bits [11:10],
/// `offset[7:6]` → bits [6:5], `offset[2:1]` → bits [4:3],
/// `offset[5]` → bit 2.
#[inline]
fn encode_cb_imm(imm: i32) -> u16 {
    let val = imm as u32;
    ((((val >> 8) & 1) << 12)
        | (((val >> 3) & 3) << 10)
        | (((val >> 7) & 1) << 6)
        | (((val >> 6) & 1) << 5)
        | (((val >> 1) & 3) << 3)
        | (((val >> 5) & 1) << 2)) as u16
}

/// Encodes a value into the CJ-type (compressed jump) immediate field.
///
/// CJ-type (C.J / C.JAL) 16-bit layout:
/// `offset[11]` → bit 12, `offset[4]` → bit 11,
/// `offset[9:8]` → bits [10:9], `offset[10]` → bit 8,
/// `offset[6]` → bit 7, `offset[7]` → bit 6,
/// `offset[3:1]` → bits [5:3], `offset[5]` → bit 2.
#[inline]
fn encode_cj_imm(imm: i32) -> u16 {
    let val = imm as u32;
    ((((val >> 11) & 1) << 12)
        | (((val >> 4) & 1) << 11)
        | (((val >> 8) & 3) << 9)
        | (((val >> 10) & 1) << 8)
        | (((val >> 6) & 1) << 7)
        | (((val >> 7) & 1) << 6)
        | (((val >> 1) & 7) << 3)
        | (((val >> 5) & 1) << 2)) as u16
}

// ===========================================================================
// RelaxationAction — linker relaxation transformation descriptor
// ===========================================================================

/// Describes a single linker relaxation transformation to apply to section
/// data. Relaxation reduces code size by replacing longer instruction
/// sequences with shorter equivalents when the target is within range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelaxationAction {
    /// Replace an 8-byte AUIPC+JALR call pair with a 4-byte JAL instruction
    /// plus a 4-byte NOP, saving decode bandwidth and I-cache pressure.
    /// Applicable when the call target fits in the JAL's ±1 MiB range.
    ReplaceCallWithJal {
        /// Byte offset of the AUIPC instruction being relaxed.
        offset: usize,
    },
    /// Replace an AUIPC+LD GOT-indirect pair with an AUIPC+ADDI direct
    /// pair when the symbol can be addressed without GOT indirection.
    ReplaceAuipcLdWithAuipcAddi {
        /// Byte offset of the AUIPC instruction being relaxed.
        offset: usize,
    },
    /// Delete NOP instructions inserted for alignment that are no longer
    /// needed after other relaxations have shrunk preceding code.
    DeleteNops {
        /// Byte offset of the first NOP to delete.
        offset: usize,
        /// Total number of bytes of NOPs to remove (multiple of 2 or 4).
        count: usize,
    },
    /// No relaxation is applicable for this relocation.
    NoRelaxation,
}

// ===========================================================================
// OffsetAdjustment — byte-removal tracking for relaxation
// ===========================================================================

/// Records a byte-removal event during linker relaxation, used to adjust
/// subsequent symbol addresses and relocation offsets after code shrinkage.
#[derive(Debug, Clone)]
pub struct OffsetAdjustment {
    /// Original byte offset in the section before relaxation.
    pub original_offset: u64,
    /// Number of bytes removed at this offset.
    pub bytes_removed: u64,
}

// ===========================================================================
// RiscV64RelocationHandler
// ===========================================================================

/// Architecture-specific relocation handler for RISC-V 64-bit targets.
///
/// Implements [`ArchRelocationHandler`] for all `R_RISCV_*` ELF relocation
/// types, and provides additional public methods for linker relaxation and
/// PCREL_LO12 pairing resolution.
///
/// # Linker Relaxation
///
/// When `relaxation_enabled` is `true`, the [`try_relax`](Self::try_relax)
/// method will attempt to replace longer instruction sequences with shorter
/// ones (e.g., AUIPC+JALR → JAL). This is critical for kernel text size
/// optimization on RISC-V.
pub struct RiscV64RelocationHandler {
    /// Whether linker relaxation optimizations are enabled.
    relaxation_enabled: bool,
}

impl RiscV64RelocationHandler {
    /// Creates a new RISC-V 64 relocation handler.
    ///
    /// # Arguments
    ///
    /// * `relaxation_enabled` — When `true`, enables linker relaxation
    ///   (CALL→JAL, AUIPC+LD→AUIPC+ADDI, alignment NOP reduction).
    pub fn new(relaxation_enabled: bool) -> Self {
        Self { relaxation_enabled }
    }

    // -----------------------------------------------------------------------
    // Individual Relocation Application Methods
    // -----------------------------------------------------------------------

    /// Applies a B-type branch relocation (`R_RISCV_BRANCH`).
    ///
    /// Encodes a 13-bit signed PC-relative offset into the B-type format.
    /// The offset must be even (bit 0 implied zero) and fit ±4 KiB.
    fn apply_branch(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !(-4096..=4094).contains(&value) || (value & 1) != 0 {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_BRANCH,
                offset: offset as u64,
                value: value as i128,
                max_value: 4094,
            });
        }
        let inst = read_u32_le(data, offset);
        // Mask preserves opcode[6:0], funct3[14:12], rs1[19:15], rs2[24:20]
        let cleared = inst & 0x01FF_F07F;
        write_u32_le(data, offset, cleared | encode_b_imm(value as i32));
        Ok(())
    }

    /// Applies a J-type jump relocation (`R_RISCV_JAL`).
    ///
    /// Encodes a 21-bit signed PC-relative offset into J-type format.
    /// The offset must be even and fit ±1 MiB.
    fn apply_jal(&self, data: &mut [u8], offset: usize, value: i64) -> Result<(), RelocationError> {
        if !(-1_048_576..=1_048_574).contains(&value) || (value & 1) != 0 {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_JAL,
                offset: offset as u64,
                value: value as i128,
                max_value: 1_048_574,
            });
        }
        let inst = read_u32_le(data, offset);
        // Preserve rd[11:7] and opcode[6:0]
        let cleared = inst & 0x0000_0FFF;
        write_u32_le(data, offset, cleared | encode_j_imm(value as i32));
        Ok(())
    }

    /// Applies an AUIPC+JALR call pair relocation (`R_RISCV_CALL` /
    /// `R_RISCV_CALL_PLT`).
    ///
    /// Patches a two-instruction sequence at `offset`:
    /// - **AUIPC** at `offset`: loads upper bits of PC-relative offset
    /// - **JALR** at `offset + 4`: adds lower 12 bits and jumps
    ///
    /// The AUIPC immediate includes a `+0x800` bias to compensate for
    /// JALR's sign extension of the 12-bit immediate.
    fn apply_call(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !(-2_147_483_648..=2_147_483_647).contains(&value) {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_CALL,
                offset: offset as u64,
                value: value as i128,
                max_value: 2_147_483_647,
            });
        }
        // Upper 20 bits with sign-extension compensation, already in U-type
        // format (bits [31:12] populated, bits [11:0] zero). The +0x800 bias
        // compensates for JALR's sign extension of the 12-bit immediate.
        let hi = ((value + 0x800) as i32) & !0xFFF;
        // Lower 12 bits (hardware sign-extends this in JALR)
        let lo = (value as i32) & 0xFFF;

        // Patch AUIPC at offset
        let auipc = read_u32_le(data, offset);
        let auipc_cleared = auipc & 0x0000_0FFF; // Keep rd and opcode
        write_u32_le(data, offset, auipc_cleared | encode_u_imm(hi));

        // Patch JALR at offset + 4
        let jalr = read_u32_le(data, offset + 4);
        let jalr_cleared = jalr & 0x000F_FFFF; // Keep opcode, funct3, rd, rs1
        write_u32_le(data, offset + 4, jalr_cleared | encode_i_imm(lo));

        Ok(())
    }

    /// Applies a U-type AUIPC PC-relative high-20 relocation
    /// (`R_RISCV_PCREL_HI20`).
    ///
    /// Encodes the upper 20 bits of a PC-relative offset with `+0x800`
    /// bias for sign-extension compensation by the paired LO12 instruction.
    fn apply_pcrel_hi20(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !(-2_147_483_648..=2_147_483_647).contains(&value) {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_PCREL_HI20,
                offset: offset as u64,
                value: value as i128,
                max_value: 2_147_483_647,
            });
        }
        let hi = ((value + 0x800) as i32) & !0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x0000_0FFF;
        write_u32_le(data, offset, cleared | encode_u_imm(hi));
        Ok(())
    }

    /// Applies an I-type PC-relative low-12 relocation
    /// (`R_RISCV_PCREL_LO12_I`).
    ///
    /// `value` is the full PC-relative offset from the corresponding HI20
    /// relocation — only the low 12 bits are encoded.
    fn apply_pcrel_lo12_i(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let lo = (value as i32) & 0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x000F_FFFF; // Keep opcode, funct3, rd, rs1
        write_u32_le(data, offset, cleared | encode_i_imm(lo));
        Ok(())
    }

    /// Applies an S-type PC-relative low-12 relocation
    /// (`R_RISCV_PCREL_LO12_S`).
    fn apply_pcrel_lo12_s(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let lo = (value as i32) & 0xFFF;
        let inst = read_u32_le(data, offset);
        // Clear imm[11:5] (bits [31:25]) and imm[4:0] (bits [11:7])
        let cleared = inst & 0x01FF_F07F;
        write_u32_le(data, offset, cleared | encode_s_imm(lo));
        Ok(())
    }

    /// Applies a U-type absolute high-20 relocation (`R_RISCV_HI20`).
    ///
    /// Encodes the upper 20 bits of an absolute address into LUI, with
    /// `+0x800` bias for sign-extension compensation.
    fn apply_hi20(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let hi = ((value + 0x800) as i32) & !0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x0000_0FFF;
        write_u32_le(data, offset, cleared | encode_u_imm(hi));
        Ok(())
    }

    /// Applies an I-type absolute low-12 relocation (`R_RISCV_LO12_I`).
    fn apply_lo12_i(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let lo = (value as i32) & 0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x000F_FFFF;
        write_u32_le(data, offset, cleared | encode_i_imm(lo));
        Ok(())
    }

    /// Applies an S-type absolute low-12 relocation (`R_RISCV_LO12_S`).
    fn apply_lo12_s(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let lo = (value as i32) & 0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x01FF_F07F;
        write_u32_le(data, offset, cleared | encode_s_imm(lo));
        Ok(())
    }

    /// Applies a 32-bit absolute data relocation (`R_RISCV_32`).
    fn apply_abs32(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        write_u32_le(data, offset, value as u32);
        Ok(())
    }

    /// Applies a 64-bit absolute data relocation (`R_RISCV_64`).
    fn apply_abs64(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        write_u64_le(data, offset, value as u64);
        Ok(())
    }

    /// Applies an addition relocation (`R_RISCV_ADD8/16/32/64`).
    ///
    /// Reads the existing value, adds the relocation value, writes back.
    fn apply_add(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
        size: u8,
    ) -> Result<(), RelocationError> {
        match size {
            1 => {
                let existing = data[offset] as i64;
                data[offset] = existing.wrapping_add(value) as u8;
            }
            2 => {
                let existing = read_u16_le(data, offset) as i64;
                write_u16_le(data, offset, existing.wrapping_add(value) as u16);
            }
            4 => {
                let existing = read_u32_le(data, offset) as i64;
                write_u32_le(data, offset, existing.wrapping_add(value) as u32);
            }
            8 => {
                let existing = read_u64_le(data, offset) as i64;
                write_u64_le(data, offset, existing.wrapping_add(value) as u64);
            }
            _ => {
                return Err(RelocationError::UnsupportedType {
                    reloc_type: R_RISCV_ADD8,
                });
            }
        }
        Ok(())
    }

    /// Applies a subtraction relocation (`R_RISCV_SUB8/16/32/64`).
    ///
    /// Reads the existing value, subtracts the relocation value, writes back.
    fn apply_sub(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
        size: u8,
    ) -> Result<(), RelocationError> {
        match size {
            1 => {
                let existing = data[offset] as i64;
                data[offset] = existing.wrapping_sub(value) as u8;
            }
            2 => {
                let existing = read_u16_le(data, offset) as i64;
                write_u16_le(data, offset, existing.wrapping_sub(value) as u16);
            }
            4 => {
                let existing = read_u32_le(data, offset) as i64;
                write_u32_le(data, offset, existing.wrapping_sub(value) as u32);
            }
            8 => {
                let existing = read_u64_le(data, offset) as i64;
                write_u64_le(data, offset, existing.wrapping_sub(value) as u64);
            }
            _ => {
                return Err(RelocationError::UnsupportedType {
                    reloc_type: R_RISCV_SUB8,
                });
            }
        }
        Ok(())
    }

    /// Applies a GOT-relative HI20 relocation (`R_RISCV_GOT_HI20`).
    ///
    /// Encodes the PC-relative offset to the GOT entry into an AUIPC
    /// instruction's U-type immediate with `+0x800` bias.
    fn apply_got_hi20(
        &self,
        data: &mut [u8],
        offset: usize,
        got_entry_addr: u64,
        pc: u64,
    ) -> Result<(), RelocationError> {
        let value = (got_entry_addr as i64).wrapping_sub(pc as i64);
        if !(-2_147_483_648..=2_147_483_647).contains(&value) {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_GOT_HI20,
                offset: offset as u64,
                value: value as i128,
                max_value: 2_147_483_647,
            });
        }
        let hi = ((value + 0x800) as i32) & !0xFFF;
        let inst = read_u32_le(data, offset);
        let cleared = inst & 0x0000_0FFF;
        write_u32_le(data, offset, cleared | encode_u_imm(hi));
        Ok(())
    }

    /// Applies a compressed branch relocation (`R_RISCV_RVC_BRANCH`).
    ///
    /// CB-type 16-bit encoding. Range: ±256 bytes (9-bit signed, bit 0
    /// implied zero).
    fn apply_compressed_branch(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !(-256..=254).contains(&value) || (value & 1) != 0 {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_RVC_BRANCH,
                offset: offset as u64,
                value: value as i128,
                max_value: 254,
            });
        }
        let inst = read_u16_le(data, offset);
        // Clear immediate fields: bits [12:10] and [6:2]
        let cleared = inst & 0xE383;
        write_u16_le(data, offset, cleared | encode_cb_imm(value as i32));
        Ok(())
    }

    /// Applies a compressed jump relocation (`R_RISCV_RVC_JUMP`).
    ///
    /// CJ-type 16-bit encoding. Range: ±2 KiB (12-bit signed, bit 0
    /// implied zero).
    fn apply_compressed_jump(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !(-2048..=2046).contains(&value) || (value & 1) != 0 {
            return Err(RelocationError::Overflow {
                reloc_type: R_RISCV_RVC_JUMP,
                offset: offset as u64,
                value: value as i128,
                max_value: 2046,
            });
        }
        let inst = read_u16_le(data, offset);
        // Clear immediate field: bits [12:2]
        let cleared = inst & 0xE003;
        write_u16_le(data, offset, cleared | encode_cj_imm(value as i32));
        Ok(())
    }

    /// Applies a 32-bit PC-relative data relocation (`R_RISCV_32_PCREL`).
    fn apply_32_pcrel(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        write_u32_le(data, offset, value as u32);
        Ok(())
    }

    /// Applies a SET6 relocation: sets the low 6 bits of the target byte.
    fn apply_set6(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        data[offset] = (data[offset] & 0xC0) | ((value as u8) & 0x3F);
        Ok(())
    }

    /// Applies a SUB6 relocation: subtracts from the low 6 bits of the byte.
    fn apply_sub6(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        let existing = data[offset] & 0x3F;
        let result = existing.wrapping_sub(value as u8) & 0x3F;
        data[offset] = (data[offset] & 0xC0) | result;
        Ok(())
    }

    /// Applies SET8/SET16/SET32 relocations: replaces the value at offset.
    fn apply_set(
        &self,
        data: &mut [u8],
        offset: usize,
        value: i64,
        size: u8,
    ) -> Result<(), RelocationError> {
        match size {
            1 => {
                data[offset] = value as u8;
            }
            2 => {
                write_u16_le(data, offset, value as u16);
            }
            4 => {
                write_u32_le(data, offset, value as u32);
            }
            _ => {
                return Err(RelocationError::UnsupportedType {
                    reloc_type: R_RISCV_SET8,
                });
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // PCREL_LO12 Pairing Resolution
    // -----------------------------------------------------------------------

    /// Resolves a `PCREL_LO12_I` or `PCREL_LO12_S` relocation by looking up
    /// the full PC-relative value from the corresponding HI20 relocation.
    ///
    /// `R_RISCV_PCREL_LO12_I/S` relocations reference the address of the
    /// corresponding `R_RISCV_PCREL_HI20` AUIPC instruction (not the target
    /// symbol). The `hi20_map` maps AUIPC instruction addresses to their
    /// computed full PC-relative values, populated during the HI20 pass.
    ///
    /// # Arguments
    ///
    /// * `lo12_reloc` — The PCREL_LO12 relocation. Its `symbol_value + addend`
    ///   should equal the paired AUIPC instruction's output address.
    /// * `hi20_map` — Map from AUIPC addresses to their full PC-relative values.
    ///
    /// # Returns
    ///
    /// The full PC-relative value whose low 12 bits encode into the LO12
    /// instruction.
    pub fn resolve_pcrel_lo12(
        &self,
        lo12_reloc: &RelocationEntry,
        hi20_map: &FxHashMap<u64, i64>,
    ) -> Result<i64, RelocationError> {
        // The PCREL_LO12's "symbol" points to the AUIPC instruction address
        let hi20_addr = lo12_reloc
            .symbol_value
            .wrapping_add(lo12_reloc.addend as u64);

        // Fast check: verify the HI20 address exists in the map before lookup
        if !hi20_map.contains_key(&hi20_addr) {
            return Err(RelocationError::UndefinedSymbol {
                name: format!(
                    "PCREL_LO12 at offset {:#x} references HI20 at {:#x} which was not found",
                    lo12_reloc.offset, hi20_addr
                ),
            });
        }

        // Safe to unwrap: we just verified the key exists
        match hi20_map.get(&hi20_addr) {
            Some(&value) => Ok(value),
            None => unreachable!("contains_key returned true but get returned None"),
        }
    }

    // -----------------------------------------------------------------------
    // Linker Relaxation Engine
    // -----------------------------------------------------------------------

    /// Reports relocation processing diagnostics using the diagnostic engine.
    ///
    /// This method wraps relocation application with diagnostic reporting,
    /// emitting warnings for overflow edge cases and errors for unsupported
    /// types. Used by the linker driver to collect diagnostics across all
    /// relocations before aborting on fatal errors.
    ///
    /// # Arguments
    ///
    /// * `reloc` — The relocation to apply.
    /// * `output_data` — Mutable output section data.
    /// * `got_address` — Base address of the `.got` section.
    /// * `plt_address` — Base address of the `.plt` section.
    /// * `diag` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// `true` if the relocation was applied successfully, `false` if an error
    /// was reported (the caller should check `diag.has_errors()`).
    pub fn apply_relocation_with_diagnostics(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
        diag: &mut DiagnosticEngine,
    ) -> bool {
        match self.apply_relocation(reloc, output_data, got_address, plt_address) {
            Ok(()) => true,
            Err(RelocationError::Overflow {
                reloc_type,
                offset,
                value,
                max_value,
            }) => {
                diag.error(
                    Span::DUMMY,
                    format!(
                        "relocation {} at offset {:#x} overflows: value {} exceeds maximum {}",
                        self.relocation_name(reloc_type),
                        offset,
                        value,
                        max_value,
                    ),
                );
                false
            }
            Err(RelocationError::UndefinedSymbol { ref name }) => {
                diag.error(Span::DUMMY, format!("undefined symbol: {}", name));
                false
            }
            Err(RelocationError::UnsupportedType { reloc_type }) => {
                diag.warning(
                    Span::DUMMY,
                    format!(
                        "unsupported RISC-V relocation type {} at offset {:#x}",
                        reloc_type, reloc.offset,
                    ),
                );
                false
            }
            Err(RelocationError::InvalidOffset {
                offset,
                section_size,
            }) => {
                diag.error(
                    Span::DUMMY,
                    format!(
                        "relocation offset {:#x} exceeds section size {:#x}",
                        offset, section_size,
                    ),
                );
                false
            }
        }
    }

    /// Reports a relaxation decision to the diagnostic engine for debugging.
    ///
    /// Emits a warning when relaxation was attempted but not possible due to
    /// range constraints, and provides informational diagnostics when
    /// relaxation succeeds.
    pub fn report_relaxation_decision(
        &self,
        reloc: &RelocationEntry,
        action: &RelaxationAction,
        diag: &mut DiagnosticEngine,
    ) {
        match action {
            RelaxationAction::ReplaceCallWithJal { offset } => {
                diag.warning(
                    Span::DUMMY,
                    format!(
                        "relaxed CALL at offset {:#x} to JAL (symbol: {})",
                        offset, reloc.symbol_name,
                    ),
                );
            }
            RelaxationAction::ReplaceAuipcLdWithAuipcAddi { offset } => {
                diag.warning(
                    Span::DUMMY,
                    format!(
                        "relaxed GOT access at offset {:#x} to direct ADDI (symbol: {})",
                        offset, reloc.symbol_name,
                    ),
                );
            }
            RelaxationAction::DeleteNops { offset, count } => {
                diag.warning(
                    Span::DUMMY,
                    format!(
                        "removed {} bytes of alignment NOPs at offset {:#x}",
                        count, offset,
                    ),
                );
            }
            RelaxationAction::NoRelaxation => {
                // No diagnostic needed for no-relaxation decisions
            }
        }
        // Check if any errors accumulated during this session
        if diag.has_errors() {
            diag.warning(
                Span::DUMMY,
                "errors detected during RISC-V relocation processing".to_string(),
            );
        }
    }

    /// Attempts to relax a single relocation, returning the relaxation action
    /// if applicable.
    ///
    /// Relaxation reduces code size by replacing longer instruction sequences
    /// with shorter equivalents when the target is within range:
    ///
    /// - **CALL/CALL_PLT → JAL:** 8-byte AUIPC+JALR → 4-byte JAL + 4-byte
    ///   NOP when target is within ±1 MiB.
    /// - **GOT_HI20 → Direct:** AUIPC+LD → AUIPC+ADDI when the symbol
    ///   can be addressed without GOT indirection.
    /// - **ALIGN:** Reduces NOP padding after preceding code shrinks.
    ///
    /// # Arguments
    ///
    /// * `reloc` — The relocation entry to consider for relaxation.
    /// * `section_data` — Current section data for inspecting instructions.
    /// * `symbol_value` — Resolved virtual address of the target symbol.
    /// * `pc` — Virtual address of the relocation site.
    pub fn try_relax(
        &self,
        reloc: &RelocationEntry,
        section_data: &[u8],
        symbol_value: u64,
        pc: u64,
    ) -> Option<RelaxationAction> {
        if !self.relaxation_enabled {
            return None;
        }

        let offset = reloc.offset as usize;

        match reloc.reloc_type {
            R_RISCV_CALL | R_RISCV_CALL_PLT => {
                // Check if the call target fits in JAL's ±1 MiB range
                let value = (symbol_value as i64)
                    .wrapping_add(reloc.addend)
                    .wrapping_sub(pc as i64);

                if (-1_048_576..=1_048_574).contains(&value)
                    && (value & 1) == 0
                    && offset + 8 <= section_data.len()
                {
                    let auipc_inst = read_u32_le(section_data, offset);
                    if (auipc_inst & 0x7F) == OPCODE_AUIPC {
                        return Some(RelaxationAction::ReplaceCallWithJal { offset });
                    }
                }
                None
            }

            R_RISCV_GOT_HI20 => {
                // Check if we can replace GOT-indirect with direct addressing
                let value = (symbol_value as i64)
                    .wrapping_add(reloc.addend)
                    .wrapping_sub(pc as i64);

                // Direct addressing requires the value to fit in 32-bit range
                if (-2_147_483_648..=2_147_483_647).contains(&value)
                    && offset + 8 <= section_data.len()
                {
                    let auipc_inst = read_u32_le(section_data, offset);
                    let next_inst = read_u32_le(section_data, offset + 4);
                    // AUIPC followed by LD (opcode=0000011, funct3=011)
                    if (auipc_inst & 0x7F) == OPCODE_AUIPC && (next_inst & 0x707F) == 0x3003 {
                        return Some(RelaxationAction::ReplaceAuipcLdWithAuipcAddi { offset });
                    }
                }
                None
            }

            R_RISCV_ALIGN => {
                // Alignment relaxation: count NOP bytes at this location
                // that could be removed if preceding code was shrunk
                let align_bytes = reloc.addend as usize;
                if align_bytes > 0 && offset + align_bytes <= section_data.len() {
                    let mut nop_bytes = 0usize;
                    let mut pos = offset;
                    let end = offset + align_bytes;

                    while pos + 4 <= end {
                        let inst = read_u32_le(section_data, pos);
                        if inst == RISCV_NOP {
                            nop_bytes += 4;
                            pos += 4;
                        } else if pos + 2 <= end {
                            let cinst = read_u16_le(section_data, pos);
                            if cinst == RISCV_CNOP {
                                nop_bytes += 2;
                                pos += 2;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    // Also check for trailing 2-byte CNOPs after 4-byte loop
                    while pos + 2 <= end {
                        let cinst = read_u16_le(section_data, pos);
                        if cinst == RISCV_CNOP {
                            nop_bytes += 2;
                            pos += 2;
                        } else {
                            break;
                        }
                    }

                    if nop_bytes > 0 {
                        return Some(RelaxationAction::DeleteNops {
                            offset,
                            count: nop_bytes,
                        });
                    }
                }
                None
            }

            _ => None,
        }
    }

    /// Applies a batch of relaxation actions to section data, returning
    /// offset adjustments for updating subsequent symbols and relocations.
    ///
    /// Actions are processed in reverse offset order so that byte removals
    /// at higher offsets do not invalidate the positions of earlier actions.
    ///
    /// # Arguments
    ///
    /// * `section_data` — Mutable section data buffer.
    /// * `relaxations` — List of relaxation actions to apply.
    ///
    /// # Returns
    ///
    /// A vector of [`OffsetAdjustment`] records describing where bytes were
    /// removed, for use in adjusting symbol values and relocation offsets.
    pub fn apply_relaxations(
        &self,
        section_data: &mut Vec<u8>,
        relaxations: &[RelaxationAction],
    ) -> Vec<OffsetAdjustment> {
        let mut adjustments = Vec::new();

        // Collect actionable relaxations and sort by offset descending so
        // that removals at higher offsets are processed first.
        let mut sorted: Vec<&RelaxationAction> = relaxations
            .iter()
            .filter(|a| **a != RelaxationAction::NoRelaxation)
            .collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(relaxation_offset(b)));

        for action in &sorted {
            match action {
                RelaxationAction::ReplaceCallWithJal { offset } => {
                    let off = *offset;
                    if off + 8 <= section_data.len() {
                        // Extract the destination register from the AUIPC
                        let auipc = read_u32_le(section_data, off);
                        let rd = (auipc >> 7) & 0x1F;

                        // Build a JAL with the same rd (immediate patched later)
                        let jal = OPCODE_JAL | (rd << 7);
                        write_u32_le(section_data, off, jal);
                        // Fill the second slot with NOP (in-place relaxation)
                        write_u32_le(section_data, off + 4, RISCV_NOP);

                        adjustments.push(OffsetAdjustment {
                            original_offset: off as u64,
                            bytes_removed: 0, // In-place: no size change
                        });
                    }
                }

                RelaxationAction::ReplaceAuipcLdWithAuipcAddi { offset } => {
                    let off = *offset;
                    if off + 8 <= section_data.len() {
                        // Replace LD with ADDI: change opcode from LOAD (0000011)
                        // to OP-IMM (0010011) and funct3 from 011 to 000
                        let ld_inst = read_u32_le(section_data, off + 4);
                        let rd = (ld_inst >> 7) & 0x1F;
                        let rs1 = (ld_inst >> 15) & 0x1F;
                        let imm_bits = ld_inst & 0xFFF0_0000;
                        // ADDI rd, rs1, imm12: opcode=0010011, funct3=000
                        let addi = 0x13 | (rd << 7) | (rs1 << 15) | imm_bits;
                        write_u32_le(section_data, off + 4, addi);

                        adjustments.push(OffsetAdjustment {
                            original_offset: off as u64,
                            bytes_removed: 0,
                        });
                    }
                }

                RelaxationAction::DeleteNops { offset, count } => {
                    let off = *offset;
                    let cnt = *count;
                    if off + cnt <= section_data.len() && cnt > 0 {
                        // Remove the NOP bytes, shifting subsequent data down
                        section_data.drain(off..off + cnt);
                        adjustments.push(OffsetAdjustment {
                            original_offset: off as u64,
                            bytes_removed: cnt as u64,
                        });
                    }
                }

                RelaxationAction::NoRelaxation => {}
            }
        }

        adjustments
    }
}

/// Returns the byte offset associated with a relaxation action, used for
/// sorting actions in reverse offset order during application.
fn relaxation_offset(action: &RelaxationAction) -> usize {
    match action {
        RelaxationAction::ReplaceCallWithJal { offset } => *offset,
        RelaxationAction::ReplaceAuipcLdWithAuipcAddi { offset } => *offset,
        RelaxationAction::DeleteNops { offset, .. } => *offset,
        RelaxationAction::NoRelaxation => usize::MAX,
    }
}

// ===========================================================================
// ArchRelocationHandler Trait Implementation
// ===========================================================================

impl ArchRelocationHandler for RiscV64RelocationHandler {
    /// Applies a single RISC-V 64 relocation to the output data buffer.
    ///
    /// Dispatches to the appropriate encoding-specific method based on
    /// `reloc.reloc_type`. For PC-relative relocations, the computation is
    /// `S + A - P` where S = `reloc.symbol_value`, A = `reloc.addend`,
    /// P = `reloc.offset` (output virtual address of the relocation site).
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), RelocationError> {
        let offset = reloc.offset as usize;
        let pc = reloc.offset;

        // S + A: absolute value
        let sym_plus_addend = (reloc.symbol_value as i64).wrapping_add(reloc.addend);
        // S + A - P: PC-relative value
        let pc_relative = sym_plus_addend.wrapping_sub(pc as i64);

        match reloc.reloc_type {
            // --- No-op relocations ---
            R_RISCV_NONE => Ok(()),

            // --- Absolute data relocations ---
            R_RISCV_32 => self.apply_abs32(output_data, offset, sym_plus_addend),
            R_RISCV_64 => self.apply_abs64(output_data, offset, sym_plus_addend),

            // --- Dynamic relocations (static linking fallback) ---
            R_RISCV_RELATIVE => self.apply_abs64(output_data, offset, sym_plus_addend),
            R_RISCV_COPY | R_RISCV_JUMP_SLOT => {
                self.apply_abs64(output_data, offset, sym_plus_addend)
            }

            // --- Branch and jump ---
            R_RISCV_BRANCH => self.apply_branch(output_data, offset, pc_relative),
            R_RISCV_JAL => self.apply_jal(output_data, offset, pc_relative),

            // --- AUIPC+JALR call pairs ---
            R_RISCV_CALL => self.apply_call(output_data, offset, pc_relative),
            R_RISCV_CALL_PLT => {
                // If PLT address is provided and symbol is unresolved (value=0),
                // use the PLT entry. Otherwise treat as a direct call.
                if plt_address != 0 && reloc.symbol_value == 0 {
                    let plt_rel = (plt_address as i64)
                        .wrapping_add(reloc.addend)
                        .wrapping_sub(pc as i64);
                    self.apply_call(output_data, offset, plt_rel)
                } else {
                    self.apply_call(output_data, offset, pc_relative)
                }
            }

            // --- GOT-relative PC-relative ---
            R_RISCV_GOT_HI20 => {
                // For GOT relocations, symbol_value is the GOT entry address
                // (pre-computed by the linker's GOT allocation), or we use
                // got_address as the base with addend.
                let got_entry = if got_address != 0 {
                    got_address.wrapping_add(reloc.addend as u64)
                } else {
                    reloc.symbol_value.wrapping_add(reloc.addend as u64)
                };
                self.apply_got_hi20(output_data, offset, got_entry, pc)
            }
            R_RISCV_TLS_GOT_HI20 | R_RISCV_TLS_GD_HI20 => {
                let got_entry = if got_address != 0 {
                    got_address.wrapping_add(reloc.addend as u64)
                } else {
                    reloc.symbol_value.wrapping_add(reloc.addend as u64)
                };
                self.apply_got_hi20(output_data, offset, got_entry, pc)
            }

            // --- PC-relative HI20/LO12 pairs ---
            R_RISCV_PCREL_HI20 => self.apply_pcrel_hi20(output_data, offset, pc_relative),
            R_RISCV_PCREL_LO12_I => {
                // For PCREL_LO12, the symbol_value + addend encodes the
                // full value from the paired HI20 (the caller/relocation
                // processor should pre-resolve this via resolve_pcrel_lo12).
                self.apply_pcrel_lo12_i(output_data, offset, sym_plus_addend)
            }
            R_RISCV_PCREL_LO12_S => self.apply_pcrel_lo12_s(output_data, offset, sym_plus_addend),

            // --- Absolute HI20/LO12 pairs ---
            R_RISCV_HI20 => self.apply_hi20(output_data, offset, sym_plus_addend),
            R_RISCV_LO12_I => self.apply_lo12_i(output_data, offset, sym_plus_addend),
            R_RISCV_LO12_S => self.apply_lo12_s(output_data, offset, sym_plus_addend),

            // --- TLS Local-Exec ---
            R_RISCV_TPREL_HI20 => self.apply_hi20(output_data, offset, sym_plus_addend),
            R_RISCV_TPREL_LO12_I => self.apply_lo12_i(output_data, offset, sym_plus_addend),
            R_RISCV_TPREL_LO12_S => self.apply_lo12_s(output_data, offset, sym_plus_addend),
            R_RISCV_TPREL_ADD => {
                // Hint relocation for TP addition — no patching needed
                Ok(())
            }

            // --- ADD/SUB relocations ---
            R_RISCV_ADD8 => self.apply_add(output_data, offset, sym_plus_addend, 1),
            R_RISCV_ADD16 => self.apply_add(output_data, offset, sym_plus_addend, 2),
            R_RISCV_ADD32 => self.apply_add(output_data, offset, sym_plus_addend, 4),
            R_RISCV_ADD64 => self.apply_add(output_data, offset, sym_plus_addend, 8),
            R_RISCV_SUB8 => self.apply_sub(output_data, offset, sym_plus_addend, 1),
            R_RISCV_SUB16 => self.apply_sub(output_data, offset, sym_plus_addend, 2),
            R_RISCV_SUB32 => self.apply_sub(output_data, offset, sym_plus_addend, 4),
            R_RISCV_SUB64 => self.apply_sub(output_data, offset, sym_plus_addend, 8),

            // --- SET/SUB6 relocations ---
            R_RISCV_SUB6 => self.apply_sub6(output_data, offset, sym_plus_addend),
            R_RISCV_SET6 => self.apply_set6(output_data, offset, sym_plus_addend),
            R_RISCV_SET8 => self.apply_set(output_data, offset, sym_plus_addend, 1),
            R_RISCV_SET16 => self.apply_set(output_data, offset, sym_plus_addend, 2),
            R_RISCV_SET32 => self.apply_set(output_data, offset, sym_plus_addend, 4),

            // --- GNU vtable markers (no-op) ---
            R_RISCV_GNU_VTINHERIT | R_RISCV_GNU_VTENTRY => Ok(()),

            // --- Alignment and relaxation markers ---
            R_RISCV_ALIGN => Ok(()), // NOP adjustment handled by relaxation engine
            R_RISCV_RELAX => Ok(()), // Hint only — no direct patching

            // --- Compressed instructions ---
            R_RISCV_RVC_BRANCH => self.apply_compressed_branch(output_data, offset, pc_relative),
            R_RISCV_RVC_JUMP => self.apply_compressed_jump(output_data, offset, pc_relative),
            R_RISCV_RVC_LUI => {
                // C.LUI immediate encoding:
                // nzimm[17] at bit 12, nzimm[16:12] at bits [6:2]
                let value = sym_plus_addend;
                let imm = ((value + 0x800) >> 12) as i32;
                let inst = read_u16_le(output_data, offset);
                let nzimm_17 = (((imm >> 5) & 1) as u16) << 12;
                let nzimm_16_12 = ((imm & 0x1F) as u16) << 2;
                let cleared = inst & 0xEF83;
                write_u16_le(output_data, offset, cleared | nzimm_17 | nzimm_16_12);
                Ok(())
            }

            // --- 32-bit PC-relative data ---
            R_RISCV_32_PCREL => self.apply_32_pcrel(output_data, offset, pc_relative),

            // --- TLS dynamic relocations (static linking fallback) ---
            R_RISCV_TLS_DTPMOD32 | R_RISCV_TLS_DTPREL32 | R_RISCV_TLS_TPREL32 => {
                self.apply_abs32(output_data, offset, sym_plus_addend)
            }
            R_RISCV_TLS_DTPMOD64 | R_RISCV_TLS_DTPREL64 | R_RISCV_TLS_TPREL64 => {
                self.apply_abs64(output_data, offset, sym_plus_addend)
            }

            // --- Unknown relocation type ---
            _ => Err(RelocationError::UnsupportedType {
                reloc_type: reloc.reloc_type,
            }),
        }
    }

    /// Returns a human-readable name for the given RISC-V relocation type.
    fn relocation_name(&self, reloc_type: u32) -> &'static str {
        match reloc_type {
            R_RISCV_NONE => "R_RISCV_NONE",
            R_RISCV_32 => "R_RISCV_32",
            R_RISCV_64 => "R_RISCV_64",
            R_RISCV_RELATIVE => "R_RISCV_RELATIVE",
            R_RISCV_COPY => "R_RISCV_COPY",
            R_RISCV_JUMP_SLOT => "R_RISCV_JUMP_SLOT",
            R_RISCV_TLS_DTPMOD32 => "R_RISCV_TLS_DTPMOD32",
            R_RISCV_TLS_DTPMOD64 => "R_RISCV_TLS_DTPMOD64",
            R_RISCV_TLS_DTPREL32 => "R_RISCV_TLS_DTPREL32",
            R_RISCV_TLS_DTPREL64 => "R_RISCV_TLS_DTPREL64",
            R_RISCV_TLS_TPREL32 => "R_RISCV_TLS_TPREL32",
            R_RISCV_TLS_TPREL64 => "R_RISCV_TLS_TPREL64",
            R_RISCV_BRANCH => "R_RISCV_BRANCH",
            R_RISCV_JAL => "R_RISCV_JAL",
            R_RISCV_CALL => "R_RISCV_CALL",
            R_RISCV_CALL_PLT => "R_RISCV_CALL_PLT",
            R_RISCV_GOT_HI20 => "R_RISCV_GOT_HI20",
            R_RISCV_TLS_GOT_HI20 => "R_RISCV_TLS_GOT_HI20",
            R_RISCV_TLS_GD_HI20 => "R_RISCV_TLS_GD_HI20",
            R_RISCV_PCREL_HI20 => "R_RISCV_PCREL_HI20",
            R_RISCV_PCREL_LO12_I => "R_RISCV_PCREL_LO12_I",
            R_RISCV_PCREL_LO12_S => "R_RISCV_PCREL_LO12_S",
            R_RISCV_HI20 => "R_RISCV_HI20",
            R_RISCV_LO12_I => "R_RISCV_LO12_I",
            R_RISCV_LO12_S => "R_RISCV_LO12_S",
            R_RISCV_TPREL_HI20 => "R_RISCV_TPREL_HI20",
            R_RISCV_TPREL_LO12_I => "R_RISCV_TPREL_LO12_I",
            R_RISCV_TPREL_LO12_S => "R_RISCV_TPREL_LO12_S",
            R_RISCV_TPREL_ADD => "R_RISCV_TPREL_ADD",
            R_RISCV_ADD8 => "R_RISCV_ADD8",
            R_RISCV_ADD16 => "R_RISCV_ADD16",
            R_RISCV_ADD32 => "R_RISCV_ADD32",
            R_RISCV_ADD64 => "R_RISCV_ADD64",
            R_RISCV_SUB8 => "R_RISCV_SUB8",
            R_RISCV_SUB16 => "R_RISCV_SUB16",
            R_RISCV_SUB32 => "R_RISCV_SUB32",
            R_RISCV_SUB64 => "R_RISCV_SUB64",
            R_RISCV_GNU_VTINHERIT => "R_RISCV_GNU_VTINHERIT",
            R_RISCV_GNU_VTENTRY => "R_RISCV_GNU_VTENTRY",
            R_RISCV_ALIGN => "R_RISCV_ALIGN",
            R_RISCV_RVC_BRANCH => "R_RISCV_RVC_BRANCH",
            R_RISCV_RVC_JUMP => "R_RISCV_RVC_JUMP",
            R_RISCV_RVC_LUI => "R_RISCV_RVC_LUI",
            R_RISCV_RELAX => "R_RISCV_RELAX",
            R_RISCV_SUB6 => "R_RISCV_SUB6",
            R_RISCV_SET6 => "R_RISCV_SET6",
            R_RISCV_SET8 => "R_RISCV_SET8",
            R_RISCV_SET16 => "R_RISCV_SET16",
            R_RISCV_SET32 => "R_RISCV_SET32",
            R_RISCV_32_PCREL => "R_RISCV_32_PCREL",
            _ => "R_RISCV_UNKNOWN",
        }
    }

    /// Returns `true` if the relocation type requires a GOT entry.
    fn needs_got_entry(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_RISCV_GOT_HI20 | R_RISCV_TLS_GOT_HI20 | R_RISCV_TLS_GD_HI20
        )
    }

    /// Returns `true` if the relocation type requires a PLT entry.
    fn needs_plt_entry(&self, reloc_type: u32) -> bool {
        matches!(reloc_type, R_RISCV_CALL_PLT)
    }

    /// Returns `true` if the relocation type computes a PC-relative value.
    fn is_pc_relative(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_RISCV_BRANCH
                | R_RISCV_JAL
                | R_RISCV_CALL
                | R_RISCV_CALL_PLT
                | R_RISCV_PCREL_HI20
                | R_RISCV_PCREL_LO12_I
                | R_RISCV_PCREL_LO12_S
                | R_RISCV_GOT_HI20
                | R_RISCV_TLS_GOT_HI20
                | R_RISCV_TLS_GD_HI20
                | R_RISCV_RVC_BRANCH
                | R_RISCV_RVC_JUMP
                | R_RISCV_32_PCREL
        )
    }

    /// Returns the byte size of the relocation field for this relocation type.
    fn relocation_size(&self, reloc_type: u32) -> u8 {
        match reloc_type {
            // No-op / hints
            R_RISCV_NONE | R_RISCV_RELAX | R_RISCV_TPREL_ADD => 0,

            // 1-byte
            R_RISCV_ADD8 | R_RISCV_SUB8 | R_RISCV_SET8 | R_RISCV_SUB6 | R_RISCV_SET6 => 1,

            // 2-byte (compressed instructions)
            R_RISCV_ADD16 | R_RISCV_SUB16 | R_RISCV_SET16 | R_RISCV_RVC_BRANCH
            | R_RISCV_RVC_JUMP | R_RISCV_RVC_LUI => 2,

            // 4-byte (standard 32-bit instructions and data)
            R_RISCV_32 | R_RISCV_32_PCREL | R_RISCV_ADD32 | R_RISCV_SUB32 | R_RISCV_SET32
            | R_RISCV_BRANCH | R_RISCV_JAL | R_RISCV_PCREL_HI20 | R_RISCV_PCREL_LO12_I
            | R_RISCV_PCREL_LO12_S | R_RISCV_HI20 | R_RISCV_LO12_I | R_RISCV_LO12_S
            | R_RISCV_GOT_HI20 | R_RISCV_TLS_GOT_HI20 | R_RISCV_TLS_GD_HI20
            | R_RISCV_TPREL_HI20 | R_RISCV_TPREL_LO12_I | R_RISCV_TPREL_LO12_S
            | R_RISCV_TLS_DTPMOD32 | R_RISCV_TLS_DTPREL32 | R_RISCV_TLS_TPREL32 => 4,

            // 8-byte (AUIPC+JALR pairs and 64-bit data)
            R_RISCV_64 | R_RISCV_ADD64 | R_RISCV_SUB64 | R_RISCV_CALL | R_RISCV_CALL_PLT
            | R_RISCV_TLS_DTPMOD64 | R_RISCV_TLS_DTPREL64 | R_RISCV_TLS_TPREL64
            | R_RISCV_RELATIVE | R_RISCV_COPY | R_RISCV_JUMP_SLOT => 8,

            // Alignment marker — variable size
            R_RISCV_ALIGN => 4,

            // GNU vtable markers — no data modification
            R_RISCV_GNU_VTINHERIT | R_RISCV_GNU_VTENTRY => 0,

            // Unknown — default to pointer size
            _ => 8,
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a RelocationEntry for testing
    fn make_reloc(offset: u64, reloc_type: u32, symbol_value: u64, addend: i64) -> RelocationEntry {
        RelocationEntry {
            offset,
            reloc_type,
            symbol_name: String::from("test_sym"),
            symbol_value,
            addend,
            output_section: 0,
        }
    }

    #[test]
    fn test_encode_decode_b_imm() {
        // Test round-trip for B-type immediates
        for &imm in &[0i32, 2, -2, 4094, -4096, 100, -100, 2048, -2048] {
            let encoded = encode_b_imm(imm);
            // Build a minimal instruction with just the immediate
            let inst = encoded;
            let decoded = extract_b_imm(inst);
            assert_eq!(
                decoded, imm,
                "B-type round-trip failed for imm={}: encoded={:#010x}, decoded={}",
                imm, encoded, decoded
            );
        }
    }

    #[test]
    fn test_encode_decode_j_imm() {
        for &imm in &[0i32, 2, -2, 1048574, -1048576, 1000, -1000, 65536, -65536] {
            let encoded = encode_j_imm(imm);
            let inst = encoded;
            let decoded = extract_j_imm(inst);
            assert_eq!(
                decoded, imm,
                "J-type round-trip failed for imm={}: encoded={:#010x}, decoded={}",
                imm, encoded, decoded
            );
        }
    }

    #[test]
    fn test_encode_decode_u_imm() {
        for &imm in &[
            0i32,
            0x1000,
            -0x1000,
            0x7FFFF000u32 as i32,
            0x80000000u32 as i32,
        ] {
            let aligned = imm & !0xFFF; // U-type only stores bits [31:12]
            let encoded = encode_u_imm(aligned);
            let decoded = extract_u_imm(encoded);
            assert_eq!(
                decoded, aligned,
                "U-type round-trip failed for imm={:#x}",
                imm
            );
        }
    }

    #[test]
    fn test_encode_decode_i_imm() {
        for &imm in &[0i32, 1, -1, 2047, -2048, 100, -100] {
            let encoded = encode_i_imm(imm);
            let decoded = extract_i_imm(encoded);
            assert_eq!(decoded, imm, "I-type round-trip failed for imm={}", imm);
        }
    }

    #[test]
    fn test_encode_decode_s_imm() {
        for &imm in &[0i32, 1, -1, 2047, -2048, 100, -100] {
            let encoded = encode_s_imm(imm);
            // S-type has bits scattered, decode from the full instruction
            let decoded = extract_s_imm(encoded);
            assert_eq!(decoded, imm, "S-type round-trip failed for imm={}", imm);
        }
    }

    #[test]
    fn test_sign_extend() {
        assert_eq!(sign_extend(0xFFF, 12), -1_i64);
        assert_eq!(sign_extend(0x7FF, 12), 2047_i64);
        assert_eq!(sign_extend(0x800, 12), -2048_i64);
        assert_eq!(sign_extend(0x1FFFFF, 21), -1_i64);
        assert_eq!(sign_extend(0x0FFFFF, 21), 1048575_i64);
    }

    #[test]
    fn test_apply_abs32() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 8];
        handler.apply_abs32(&mut data, 2, 0x12345678).unwrap();
        assert_eq!(read_u32_le(&data, 2), 0x12345678);
    }

    #[test]
    fn test_apply_abs64() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 16];
        handler
            .apply_abs64(&mut data, 4, 0x123456789ABCDEF0_u64 as i64)
            .unwrap();
        assert_eq!(read_u64_le(&data, 4), 0x123456789ABCDEF0);
    }

    #[test]
    fn test_apply_branch_in_range() {
        let handler = RiscV64RelocationHandler::new(false);
        // BEQ x0, x0, offset — opcode=1100011, funct3=000
        let beq_base: u32 = 0b1100011;
        let mut data = beq_base.to_le_bytes().to_vec();
        handler.apply_branch(&mut data, 0, 256).unwrap();
        let result = read_u32_le(&data, 0);
        let decoded_imm = extract_b_imm(result);
        assert_eq!(decoded_imm, 256);
    }

    #[test]
    fn test_apply_branch_overflow() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 4];
        let result = handler.apply_branch(&mut data, 0, 8192);
        assert!(result.is_err());
        if let Err(RelocationError::Overflow { reloc_type, .. }) = result {
            assert_eq!(reloc_type, R_RISCV_BRANCH);
        }
    }

    #[test]
    fn test_apply_jal_in_range() {
        let handler = RiscV64RelocationHandler::new(false);
        // JAL x1, offset — opcode=1101111, rd=00001
        let jal_base: u32 = 0b00001_1101111;
        let mut data = jal_base.to_le_bytes().to_vec();
        handler.apply_jal(&mut data, 0, 1024).unwrap();
        let result = read_u32_le(&data, 0);
        let decoded_imm = extract_j_imm(result);
        assert_eq!(decoded_imm, 1024);
        // Verify rd is preserved
        assert_eq!((result >> 7) & 0x1F, 1);
    }

    #[test]
    fn test_apply_call_pair() {
        let handler = RiscV64RelocationHandler::new(false);
        // AUIPC x1, 0 (rd=1) | JALR x1, x1, 0 (rd=1, rs1=1)
        let auipc: u32 = 0b00001_0010111; // AUIPC x1
        let jalr: u32 = 0b0000_0000_0000_0000_1000_0000_1110_0111; // JALR x1,x1,0
        let mut data = Vec::new();
        data.extend_from_slice(&auipc.to_le_bytes());
        data.extend_from_slice(&jalr.to_le_bytes());

        handler.apply_call(&mut data, 0, 0x12345).unwrap();

        // Verify AUIPC got the high bits and JALR got the low bits
        let auipc_result = read_u32_le(&data, 0);
        let jalr_result = read_u32_le(&data, 4);
        let hi = extract_u_imm(auipc_result) as i64;
        let lo = extract_i_imm(jalr_result) as i64;
        // Reconstructed value = hi + lo should equal the original
        assert_eq!(hi + lo, 0x12345);
    }

    #[test]
    fn test_apply_add_sub() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 8];
        // Write initial value 100 as u32
        write_u32_le(&mut data, 0, 100);
        // ADD32: 100 + 50 = 150
        handler.apply_add(&mut data, 0, 50, 4).unwrap();
        assert_eq!(read_u32_le(&data, 0), 150);
        // SUB32: 150 - 30 = 120
        handler.apply_sub(&mut data, 0, 30, 4).unwrap();
        assert_eq!(read_u32_le(&data, 0), 120);
    }

    #[test]
    fn test_relocation_names() {
        let handler = RiscV64RelocationHandler::new(false);
        assert_eq!(handler.relocation_name(R_RISCV_BRANCH), "R_RISCV_BRANCH");
        assert_eq!(handler.relocation_name(R_RISCV_JAL), "R_RISCV_JAL");
        assert_eq!(handler.relocation_name(R_RISCV_CALL), "R_RISCV_CALL");
        assert_eq!(
            handler.relocation_name(R_RISCV_GOT_HI20),
            "R_RISCV_GOT_HI20"
        );
        assert_eq!(handler.relocation_name(999), "R_RISCV_UNKNOWN");
    }

    #[test]
    fn test_needs_got_plt() {
        let handler = RiscV64RelocationHandler::new(false);
        assert!(handler.needs_got_entry(R_RISCV_GOT_HI20));
        assert!(handler.needs_got_entry(R_RISCV_TLS_GOT_HI20));
        assert!(!handler.needs_got_entry(R_RISCV_CALL));
        assert!(handler.needs_plt_entry(R_RISCV_CALL_PLT));
        assert!(!handler.needs_plt_entry(R_RISCV_CALL));
    }

    #[test]
    fn test_is_pc_relative() {
        let handler = RiscV64RelocationHandler::new(false);
        assert!(handler.is_pc_relative(R_RISCV_BRANCH));
        assert!(handler.is_pc_relative(R_RISCV_JAL));
        assert!(handler.is_pc_relative(R_RISCV_CALL));
        assert!(handler.is_pc_relative(R_RISCV_PCREL_HI20));
        assert!(handler.is_pc_relative(R_RISCV_GOT_HI20));
        assert!(!handler.is_pc_relative(R_RISCV_32));
        assert!(!handler.is_pc_relative(R_RISCV_64));
        assert!(!handler.is_pc_relative(R_RISCV_HI20));
    }

    #[test]
    fn test_relocation_sizes() {
        let handler = RiscV64RelocationHandler::new(false);
        assert_eq!(handler.relocation_size(R_RISCV_NONE), 0);
        assert_eq!(handler.relocation_size(R_RISCV_ADD8), 1);
        assert_eq!(handler.relocation_size(R_RISCV_RVC_BRANCH), 2);
        assert_eq!(handler.relocation_size(R_RISCV_BRANCH), 4);
        assert_eq!(handler.relocation_size(R_RISCV_32), 4);
        assert_eq!(handler.relocation_size(R_RISCV_64), 8);
        assert_eq!(handler.relocation_size(R_RISCV_CALL), 8);
    }

    #[test]
    fn test_try_relax_call_to_jal() {
        let handler = RiscV64RelocationHandler::new(true);

        // Build an AUIPC+JALR pair at offset 0
        let auipc: u32 = 0b00001_0010111; // AUIPC x1
        let jalr: u32 = 0b0000_0000_0000_0000_1000_0000_1110_0111; // JALR x1,x1,0
        let mut data = Vec::new();
        data.extend_from_slice(&auipc.to_le_bytes());
        data.extend_from_slice(&jalr.to_le_bytes());

        let reloc = make_reloc(0, R_RISCV_CALL, 0, 0);
        // Target within ±1 MiB (symbol at 0x1000, pc at 0)
        let result = handler.try_relax(&reloc, &data, 0x1000, 0);
        assert_eq!(
            result,
            Some(RelaxationAction::ReplaceCallWithJal { offset: 0 })
        );
    }

    #[test]
    fn test_try_relax_call_out_of_range() {
        let handler = RiscV64RelocationHandler::new(true);

        let auipc: u32 = 0b00001_0010111;
        let jalr: u32 = 0b0000_0000_0000_0000_1000_0000_1110_0111;
        let mut data = Vec::new();
        data.extend_from_slice(&auipc.to_le_bytes());
        data.extend_from_slice(&jalr.to_le_bytes());

        let reloc = make_reloc(0, R_RISCV_CALL, 0, 0);
        // Target beyond ±1 MiB
        let result = handler.try_relax(&reloc, &data, 0x200000, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_relax_disabled() {
        let handler = RiscV64RelocationHandler::new(false);

        let auipc: u32 = 0b00001_0010111;
        let jalr: u32 = 0b0000_0000_0000_0000_1000_0000_1110_0111;
        let mut data = Vec::new();
        data.extend_from_slice(&auipc.to_le_bytes());
        data.extend_from_slice(&jalr.to_le_bytes());

        let reloc = make_reloc(0, R_RISCV_CALL, 0, 0);
        let result = handler.try_relax(&reloc, &data, 0x100, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_pcrel_lo12() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut hi20_map: FxHashMap<u64, i64> = FxHashMap::default();
        hi20_map.insert(0x1000, 0x12345);

        // LO12 reloc referencing HI20 at address 0x1000
        let lo12_reloc = make_reloc(0x1008, R_RISCV_PCREL_LO12_I, 0x1000, 0);
        let value = handler.resolve_pcrel_lo12(&lo12_reloc, &hi20_map).unwrap();
        assert_eq!(value, 0x12345);
    }

    #[test]
    fn test_resolve_pcrel_lo12_missing() {
        let handler = RiscV64RelocationHandler::new(false);
        let hi20_map: FxHashMap<u64, i64> = FxHashMap::default();

        let lo12_reloc = make_reloc(0x1008, R_RISCV_PCREL_LO12_I, 0x2000, 0);
        let result = handler.resolve_pcrel_lo12(&lo12_reloc, &hi20_map);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_relaxations_call_to_jal() {
        let handler = RiscV64RelocationHandler::new(true);

        // Build AUIPC+JALR at offset 0
        let auipc: u32 = 0b00001_0010111;
        let jalr: u32 = 0b0000_0000_0000_0000_1000_0000_1110_0111;
        let mut data = Vec::new();
        data.extend_from_slice(&auipc.to_le_bytes());
        data.extend_from_slice(&jalr.to_le_bytes());

        let relaxations = vec![RelaxationAction::ReplaceCallWithJal { offset: 0 }];
        let adjustments = handler.apply_relaxations(&mut data, &relaxations);

        // Verify: first instruction should be JAL with rd=1
        let first = read_u32_le(&data, 0);
        assert_eq!(first & 0x7F, OPCODE_JAL);
        assert_eq!((first >> 7) & 0x1F, 1); // rd preserved
                                            // Second instruction should be NOP
        assert_eq!(read_u32_le(&data, 4), RISCV_NOP);
        assert_eq!(adjustments.len(), 1);
    }

    #[test]
    fn test_apply_relaxations_delete_nops() {
        let handler = RiscV64RelocationHandler::new(true);

        let mut data = Vec::new();
        // 8 bytes of real code
        data.extend_from_slice(&[0xAA; 4]);
        data.extend_from_slice(&[0xBB; 4]);
        // 8 bytes of NOPs at offset 8
        data.extend_from_slice(&RISCV_NOP.to_le_bytes());
        data.extend_from_slice(&RISCV_NOP.to_le_bytes());
        // 4 more bytes of real code
        data.extend_from_slice(&[0xCC; 4]);

        let original_len = data.len(); // 20
        let relaxations = vec![RelaxationAction::DeleteNops {
            offset: 8,
            count: 8,
        }];
        let adjustments = handler.apply_relaxations(&mut data, &relaxations);

        assert_eq!(data.len(), original_len - 8); // 12 bytes
        assert_eq!(&data[0..4], &[0xAA; 4]);
        assert_eq!(&data[4..8], &[0xBB; 4]);
        assert_eq!(&data[8..12], &[0xCC; 4]);
        assert_eq!(adjustments.len(), 1);
        assert_eq!(adjustments[0].original_offset, 8);
        assert_eq!(adjustments[0].bytes_removed, 8);
    }

    #[test]
    fn test_compressed_branch_encoding() {
        let handler = RiscV64RelocationHandler::new(false);
        // C.BEQZ: opcode=110, funct3=...
        // Build a minimal C.BEQZ at offset 0
        let cbeqz_base: u16 = 0xC001; // C.BEQZ rs1', 0
        let mut data = cbeqz_base.to_le_bytes().to_vec();
        handler.apply_compressed_branch(&mut data, 0, 64).unwrap();
        // Just verify no error for a valid offset
    }

    #[test]
    fn test_compressed_branch_overflow() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 2];
        let result = handler.apply_compressed_branch(&mut data, 0, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_set6_sub6() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0xFF_u8; 4];
        // SET6: set low 6 bits to 0x15 (21)
        handler.apply_set6(&mut data, 1, 0x15).unwrap();
        assert_eq!(data[1], 0xC0 | 0x15); // Upper 2 bits preserved
                                          // SUB6: subtract 5 from low 6 bits
        handler.apply_sub6(&mut data, 1, 5).unwrap();
        assert_eq!(data[1] & 0x3F, 0x10); // 0x15 - 5 = 0x10
    }

    #[test]
    fn test_full_relocation_dispatch() {
        let handler = RiscV64RelocationHandler::new(false);

        // Test R_RISCV_NONE
        let mut data = vec![0u8; 8];
        let reloc = make_reloc(0, R_RISCV_NONE, 0, 0);
        assert!(handler.apply_relocation(&reloc, &mut data, 0, 0).is_ok());

        // Test R_RISCV_32
        let mut data = vec![0u8; 8];
        let reloc = make_reloc(0, R_RISCV_32, 0x1234, 0x10);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(read_u32_le(&data, 0), 0x1244);

        // Test R_RISCV_64
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(0, R_RISCV_64, 0xDEADBEEF, 1);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(read_u64_le(&data, 0), 0xDEADBEF0);
    }

    #[test]
    fn test_unsupported_relocation() {
        let handler = RiscV64RelocationHandler::new(false);
        let mut data = vec![0u8; 8];
        let reloc = make_reloc(0, 999, 0, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(matches!(
            result,
            Err(RelocationError::UnsupportedType { reloc_type: 999 })
        ));
    }
}
