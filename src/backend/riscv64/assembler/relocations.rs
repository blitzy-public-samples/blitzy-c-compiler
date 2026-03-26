#![allow(clippy::manual_range_contains)]
//! # RISC-V 64 ELF Relocation Types
//!
//! Defines all RISC-V 64 relocation types as specified by the RISC-V ELF psABI.
//! This module provides the foundational relocation infrastructure for BCC's
//! built-in RISC-V 64 assembler and linker.
//!
//! ## Relocation Categories
//!
//! ### Absolute relocations (static linking)
//! - `R_RISCV_32`, `R_RISCV_64`: Absolute address (32/64-bit)
//! - `R_RISCV_HI20`, `R_RISCV_LO12_I`, `R_RISCV_LO12_S`: Split absolute
//!   addressing via LUI+ADDI/LD/SW
//!
//! ### PC-relative relocations
//! - `R_RISCV_PCREL_HI20`, `R_RISCV_PCREL_LO12_I`, `R_RISCV_PCREL_LO12_S`:
//!   Split PC-relative addressing via AUIPC+ADDI/LD/SW
//! - `R_RISCV_BRANCH`: B-type conditional branch (±4 KiB range)
//! - `R_RISCV_JAL`: J-type unconditional jump (±1 MiB range)
//! - `R_RISCV_CALL`, `R_RISCV_CALL_PLT`: AUIPC+JALR call sequence (±2 GiB)
//!
//! ### GOT/PLT relocations (dynamic linking, PIC)
//! - `R_RISCV_GOT_HI20`: PC-relative GOT entry for AUIPC
//! - `R_RISCV_COPY`, `R_RISCV_GLOB_DAT`, `R_RISCV_JUMP_SLOT`,
//!   `R_RISCV_RELATIVE`: Dynamic linker relocations
//!
//! ### Linker relaxation
//! - `R_RISCV_RELAX`: Paired with another relocation, hints the linker may
//!   shorten the instruction sequence
//! - `R_RISCV_ALIGN`: Alignment directive (NOP sled that the linker may shrink)
//!
//! ### Addend arithmetic
//! - `R_RISCV_ADD32/64`, `R_RISCV_SUB32/64`: Paired relocations for computed
//!   addresses (DWARF, exception tables)
//! - `R_RISCV_SET6/8/16/32`: Absolute set operations for DWARF/CIE/FDE tables
//! - `R_RISCV_SUB6`: 6-bit subtraction for DWARF
//!
//! ## Instruction Encoding
//!
//! RISC-V uses little-endian instruction encoding. Relocation patching reads,
//! modifies, and writes instruction words in little-endian byte order.
//! Immediate fields in B-type and J-type instructions are scrambled across
//! non-contiguous bit positions — the insertion helpers handle this correctly.
//!
//! ## Zero-Dependency Mandate
//!
//! No external crates. Only `std` and `crate::` references.

use crate::backend::linker_common::relocation::{
    RelocCategory, RelocationError, RelocationHandler, ResolvedRelocation,
};

// ---------------------------------------------------------------------------
// RiscV64RelocationType — RISC-V 64 ELF relocation type enum
// ---------------------------------------------------------------------------

/// RISC-V 64 ELF relocation types.
///
/// Numeric values match the RISC-V ELF psABI specification exactly.
/// We intentionally do NOT use `#[repr(u32)]` because `DynRelative` and
/// `Relative` both map to ELF value 3 (R_RISCV_RELATIVE). Conversion is
/// handled via [`from_u32`] and [`as_elf_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiscV64RelocationType {
    /// R_RISCV_NONE (0) — No relocation.
    None,
    /// R_RISCV_32 (1) — 32-bit absolute address: S + A.
    R32,
    /// R_RISCV_64 (2) — 64-bit absolute address: S + A.
    R64,
    /// R_RISCV_RELATIVE (3) — Base + A (dynamic linker).
    Relative,
    /// R_RISCV_COPY (4) — Copy relocation (dynamic linker).
    Copy,
    /// R_RISCV_JUMP_SLOT (5) — PLT jump slot (dynamic linker).
    JumpSlot,
    /// R_RISCV_GLOB_DAT (6) — GOT data entry (dynamic linker).
    ///
    /// In the canonical psABI value 6 is R_RISCV_TLS_DTPMOD32, but for
    /// 64-bit targets BCC maps this slot to GLOB_DAT for dynamic GOT entries.
    GlobDat,
    /// R_RISCV_TLS_DTPMOD64 (7) — TLS module ID.
    TlsDtpmod64,
    /// R_RISCV_TLS_DTPREL64 (9) — TLS offset within module.
    TlsDtprel64,
    /// R_RISCV_TLS_TPREL64 (11) — TLS offset from TP.
    TlsTprel64,
    /// R_RISCV_BRANCH (16) — B-type conditional branch: S + A - P.
    /// 13-bit signed even offset (±4 KiB range).
    Branch,
    /// R_RISCV_JAL (17) — J-type unconditional jump: S + A - P.
    /// 21-bit signed even offset (±1 MiB range).
    Jal,
    /// R_RISCV_CALL (18) — AUIPC+JALR call: S + A - P (±2 GiB).
    /// Applied to the AUIPC instruction; linker also patches the JALR.
    Call,
    /// R_RISCV_CALL_PLT (19) — Like Call but forces PLT usage.
    CallPlt,
    /// R_RISCV_GOT_HI20 (20) — GOT entry for AUIPC: G + A - P (upper 20).
    GotHi20,
    /// R_RISCV_PCREL_HI20 (23) — PC-relative upper 20 bits for AUIPC.
    PcrelHi20,
    /// R_RISCV_PCREL_LO12_I (24) — PC-relative lower 12 bits, I-type.
    ///
    /// CRITICAL: The symbol references the LABEL at the corresponding AUIPC,
    /// not the final target symbol directly. The actual value is the lo12 of
    /// the PC-relative computation already performed by the PCREL_HI20.
    PcrelLo12I,
    /// R_RISCV_PCREL_LO12_S (25) — PC-relative lower 12 bits, S-type.
    PcrelLo12S,
    /// R_RISCV_HI20 (26) — Absolute upper 20 bits for LUI: S + A.
    Hi20,
    /// R_RISCV_LO12_I (27) — Absolute lower 12 bits, I-type: S + A.
    Lo12I,
    /// R_RISCV_LO12_S (28) — Absolute lower 12 bits, S-type: S + A.
    Lo12S,
    /// R_RISCV_ADD32 (35) — *(loc) += S + A (32-bit).
    Add32,
    /// R_RISCV_ADD64 (36) — *(loc) += S + A (64-bit).
    Add64,
    /// R_RISCV_SUB32 (39) — *(loc) -= (S + A) (32-bit).
    Sub32,
    /// R_RISCV_SUB64 (40) — *(loc) -= (S + A) (64-bit).
    Sub64,
    /// R_RISCV_SUB6 (52) — 6-bit subtraction (DWARF).
    Sub6,
    /// R_RISCV_ALIGN (43) — Alignment NOP sled for linker relaxation.
    Align,
    /// R_RISCV_RELAX (51) — Linker relaxation hint (paired with another reloc).
    Relax,
    /// R_RISCV_SET6 (53) — Set 6-bit value (DWARF).
    Set6,
    /// R_RISCV_SET8 (54) — Set 8-bit value.
    Set8,
    /// R_RISCV_SET16 (55) — Set 16-bit value.
    Set16,
    /// R_RISCV_SET32 (56) — Set 32-bit value.
    Set32,
    /// R_RISCV_32_PCREL (57) — 32-bit PC-relative (.eh_frame).
    R32Pcrel,
    /// Alias for `Relative` used specifically in dynamic linking context.
    /// Maps to ELF value 3 (same as R_RISCV_RELATIVE).
    DynRelative,
}

// ---------------------------------------------------------------------------
// Conversion: u32 <-> RiscV64RelocationType
// ---------------------------------------------------------------------------

impl RiscV64RelocationType {
    /// Convert a raw ELF relocation type code to a `RiscV64RelocationType`.
    ///
    /// Unknown relocation types are mapped to `None` (treated as no-op).
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::R32,
            2 => Self::R64,
            3 => Self::Relative,
            4 => Self::Copy,
            5 => Self::JumpSlot,
            6 => Self::GlobDat,
            7 => Self::TlsDtpmod64,
            9 => Self::TlsDtprel64,
            11 => Self::TlsTprel64,
            16 => Self::Branch,
            17 => Self::Jal,
            18 => Self::Call,
            19 => Self::CallPlt,
            20 => Self::GotHi20,
            23 => Self::PcrelHi20,
            24 => Self::PcrelLo12I,
            25 => Self::PcrelLo12S,
            26 => Self::Hi20,
            27 => Self::Lo12I,
            28 => Self::Lo12S,
            35 => Self::Add32,
            36 => Self::Add64,
            39 => Self::Sub32,
            40 => Self::Sub64,
            43 => Self::Align,
            51 => Self::Relax,
            52 => Self::Sub6,
            53 => Self::Set6,
            54 => Self::Set8,
            55 => Self::Set16,
            56 => Self::Set32,
            57 => Self::R32Pcrel,
            _ => Self::None,
        }
    }

    /// Return the ELF relocation type numeric value for this relocation.
    pub fn as_elf_type(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::R32 => 1,
            Self::R64 => 2,
            Self::Relative | Self::DynRelative => 3,
            Self::Copy => 4,
            Self::JumpSlot => 5,
            Self::GlobDat => 6,
            Self::TlsDtpmod64 => 7,
            Self::TlsDtprel64 => 9,
            Self::TlsTprel64 => 11,
            Self::Branch => 16,
            Self::Jal => 17,
            Self::Call => 18,
            Self::CallPlt => 19,
            Self::GotHi20 => 20,
            Self::PcrelHi20 => 23,
            Self::PcrelLo12I => 24,
            Self::PcrelLo12S => 25,
            Self::Hi20 => 26,
            Self::Lo12I => 27,
            Self::Lo12S => 28,
            Self::Add32 => 35,
            Self::Add64 => 36,
            Self::Sub32 => 39,
            Self::Sub64 => 40,
            Self::Align => 43,
            Self::Relax => 51,
            Self::Sub6 => 52,
            Self::Set6 => 53,
            Self::Set8 => 54,
            Self::Set16 => 55,
            Self::Set32 => 56,
            Self::R32Pcrel => 57,
        }
    }
}

// ---------------------------------------------------------------------------
// Relocation Classification
// ---------------------------------------------------------------------------

impl RiscV64RelocationType {
    /// Whether this relocation is PC-relative.
    pub fn is_pc_relative(&self) -> bool {
        matches!(
            self,
            Self::Branch
                | Self::Jal
                | Self::Call
                | Self::CallPlt
                | Self::PcrelHi20
                | Self::PcrelLo12I
                | Self::PcrelLo12S
                | Self::GotHi20
                | Self::R32Pcrel
        )
    }

    /// Whether this relocation requires a GOT entry.
    pub fn needs_got(&self) -> bool {
        matches!(self, Self::GotHi20)
    }

    /// Whether this relocation requires a PLT stub.
    pub fn needs_plt(&self) -> bool {
        matches!(self, Self::CallPlt)
    }

    /// Whether this relocation is a linker relaxation hint.
    pub fn is_relaxation_hint(&self) -> bool {
        matches!(self, Self::Relax | Self::Align)
    }

    /// Whether this relocation is used by the dynamic linker at runtime.
    pub fn is_dynamic(&self) -> bool {
        matches!(
            self,
            Self::Copy
                | Self::JumpSlot
                | Self::GlobDat
                | Self::Relative
                | Self::DynRelative
                | Self::TlsDtpmod64
                | Self::TlsDtprel64
                | Self::TlsTprel64
        )
    }

    /// Classify this relocation into a [`RelocCategory`] for integration with
    /// the architecture-agnostic linker framework.
    pub fn category(&self) -> RelocCategory {
        match self {
            // PC-relative types (checked first since GotHi20 is PC-relative)
            _ if self.is_pc_relative() => RelocCategory::PcRelative,
            // TLS types
            Self::TlsDtpmod64 | Self::TlsDtprel64 | Self::TlsTprel64 => RelocCategory::Tls,
            // Dynamic types
            Self::Copy | Self::JumpSlot | Self::GlobDat | Self::Relative | Self::DynRelative => {
                RelocCategory::Other
            }
            // Relaxation / metadata types
            Self::Relax | Self::Align => RelocCategory::Other,
            // Absolute types (R32, R64, Hi20, Lo12I, Lo12S, ADD/SUB/SET, None, etc.)
            _ => RelocCategory::Absolute,
        }
    }

    /// Bit width of the value this relocation patches.
    ///
    /// Returns 0 for relocations that don't patch a specific field (e.g. Relax,
    /// None) or for dynamic-only relocations.
    pub fn value_width(&self) -> u8 {
        match self {
            Self::R32 | Self::Add32 | Self::Sub32 | Self::Set32 | Self::R32Pcrel => 32,
            Self::R64 | Self::Add64 | Self::Sub64 => 64,
            Self::Set16 => 16,
            Self::Set8 => 8,
            Self::Set6 | Self::Sub6 => 6,
            Self::Branch => 13,
            Self::Jal => 21,
            Self::Hi20 | Self::PcrelHi20 | Self::GotHi20 => 20,
            Self::Lo12I | Self::Lo12S | Self::PcrelLo12I | Self::PcrelLo12S => 12,
            Self::Call | Self::CallPlt => 32,
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// RelaxationAction — linker relaxation actions
// ---------------------------------------------------------------------------

/// Relaxation action the linker can perform on a RISC-V instruction sequence.
///
/// RISC-V linker relaxation can shorten instruction sequences when the final
/// link-time addresses are known and within narrower ranges than the worst case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxationAction {
    /// No relaxation possible for this relocation/value combination.
    NoRelax,
    /// Convert AUIPC+JALR (8 bytes) to JAL (4 bytes) — saves 4 bytes.
    /// Applicable when the call target is within ±1 MiB of the call site.
    CallToJal,
    /// Convert AUIPC+ADDI (8 bytes) to just ADDI when the PC-relative value
    /// fits in a 12-bit signed immediate.
    PcrelToAddi,
    /// Delete NOP alignment bytes inserted by R_RISCV_ALIGN.
    DeleteAlign {
        /// Number of NOP bytes to remove from the alignment sled.
        bytes_to_remove: usize,
    },
}

// ---------------------------------------------------------------------------
// Little-endian read/write helpers
// ---------------------------------------------------------------------------

/// Read a 32-bit little-endian instruction word from a byte slice.
#[inline]
fn read_insn32(code: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        code[offset],
        code[offset + 1],
        code[offset + 2],
        code[offset + 3],
    ])
}

/// Write a 32-bit little-endian instruction word to a byte slice.
#[inline]
fn write_insn32(code: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    code[offset..offset + 4].copy_from_slice(&bytes);
}

/// Read a 16-bit little-endian value.
#[inline]
#[allow(dead_code)]
fn read_u16_le(code: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([code[offset], code[offset + 1]])
}

/// Write a 16-bit little-endian value.
#[inline]
fn write_u16_le(code: &mut [u8], offset: usize, value: u16) {
    let bytes = value.to_le_bytes();
    code[offset..offset + 2].copy_from_slice(&bytes);
}

/// Read a 32-bit little-endian data value (distinct from instruction reads
/// for semantic clarity; identical implementation).
#[inline]
fn read_u32_le(code: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        code[offset],
        code[offset + 1],
        code[offset + 2],
        code[offset + 3],
    ])
}

/// Write a 32-bit little-endian data value.
#[inline]
fn write_u32_le(code: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    code[offset..offset + 4].copy_from_slice(&bytes);
}

/// Read a 64-bit little-endian value.
#[inline]
fn read_u64_le(code: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        code[offset],
        code[offset + 1],
        code[offset + 2],
        code[offset + 3],
        code[offset + 4],
        code[offset + 5],
        code[offset + 6],
        code[offset + 7],
    ])
}

/// Write a 64-bit little-endian value.
#[inline]
fn write_u64_le(code: &mut [u8], offset: usize, value: u64) {
    let bytes = value.to_le_bytes();
    code[offset..offset + 8].copy_from_slice(&bytes);
}

// ---------------------------------------------------------------------------
// Immediate insertion helpers for RISC-V instruction formats
// ---------------------------------------------------------------------------

/// Insert a B-type immediate into an instruction word.
///
/// B-type immediate encoding (scrambled):
/// - bit 31 ← imm\[12\]
/// - bits 30:25 ← imm\[10:5\]
/// - bits 11:8 ← imm\[4:1\]
/// - bit 7 ← imm\[11\]
///
/// The immediate is always even (bit 0 is implicitly 0); the input `imm`
/// includes bit 0 but it is masked out.
fn insert_b_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit12 = (imm_u >> 12) & 1;
    let bit11 = (imm_u >> 11) & 1;
    let bits10_5 = (imm_u >> 5) & 0x3F;
    let bits4_1 = (imm_u >> 1) & 0xF;
    // Preserve: opcode[6:0], funct3[14:12], rs1[19:15], rs2[24:20]
    let cleared = insn & 0x01FFF07F;
    cleared | (bit12 << 31) | (bits10_5 << 25) | (bits4_1 << 8) | (bit11 << 7)
}

/// Extract a B-type immediate from an instruction word (sign-extended).
#[allow(dead_code)]
fn extract_b_imm(insn: u32) -> i32 {
    let bit12 = (insn >> 31) & 1;
    let bit11 = (insn >> 7) & 1;
    let bits10_5 = (insn >> 25) & 0x3F;
    let bits4_1 = (insn >> 8) & 0xF;
    let raw = (bit12 << 12) | (bit11 << 11) | (bits10_5 << 5) | (bits4_1 << 1);
    // Sign-extend from bit 12
    ((raw as i32) << 19) >> 19
}

/// Insert a J-type immediate into an instruction word.
///
/// J-type immediate encoding (scrambled):
/// - bit 31 ← imm\[20\]
/// - bits 30:21 ← imm\[10:1\]
/// - bit 20 ← imm\[11\]
/// - bits 19:12 ← imm\[19:12\]
fn insert_j_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = imm as u32;
    let bit20 = (imm_u >> 20) & 1;
    let bits10_1 = (imm_u >> 1) & 0x3FF;
    let bit11 = (imm_u >> 11) & 1;
    let bits19_12 = (imm_u >> 12) & 0xFF;
    // Preserve: opcode[6:0] and rd[11:7]
    let cleared = insn & 0x00000FFF;
    cleared | (bit20 << 31) | (bits10_1 << 21) | (bit11 << 20) | (bits19_12 << 12)
}

/// Extract a J-type immediate from an instruction word (sign-extended).
#[allow(dead_code)]
fn extract_j_imm(insn: u32) -> i32 {
    let bit20 = (insn >> 31) & 1;
    let bits10_1 = (insn >> 21) & 0x3FF;
    let bit11 = (insn >> 20) & 1;
    let bits19_12 = (insn >> 12) & 0xFF;
    let raw = (bit20 << 20) | (bits19_12 << 12) | (bit11 << 11) | (bits10_1 << 1);
    // Sign-extend from bit 20
    ((raw as i32) << 11) >> 11
}

/// Insert a U-type upper 20-bit immediate (for LUI/AUIPC).
///
/// The upper 20 bits of the immediate occupy bits \[31:12\] of the instruction.
/// The `imm` parameter should already be the pre-shifted 20-bit value
/// (i.e., imm << 12 has already been done by the caller via `compute_hi_lo`).
fn insert_u_imm(insn: u32, imm: i32) -> u32 {
    // Preserve: opcode[6:0] and rd[11:7]
    let cleared = insn & 0x00000FFF;
    cleared | ((imm as u32) & 0xFFFFF000)
}

/// Insert an I-type 12-bit immediate (for ADDI/LD/JALR).
///
/// The 12-bit immediate occupies bits \[31:20\].
fn insert_i_imm(insn: u32, imm: i32) -> u32 {
    // Preserve: lower 20 bits (opcode, rd, funct3, rs1)
    let cleared = insn & 0x000FFFFF;
    cleared | (((imm as u32) & 0xFFF) << 20)
}

/// Insert an S-type 12-bit split immediate (for SW/SD).
///
/// S-type splits the 12-bit immediate:
/// - bits \[31:25\] ← imm\[11:5\]
/// - bits \[11:7\]  ← imm\[4:0\]
fn insert_s_imm(insn: u32, imm: i32) -> u32 {
    let imm_u = (imm as u32) & 0xFFF;
    let upper = (imm_u >> 5) & 0x7F;
    let lower = imm_u & 0x1F;
    // Preserve: opcode[6:0], funct3[14:12], rs1[19:15], rs2[24:20]
    let cleared = insn & 0x01FFF07F;
    cleared | (upper << 25) | (lower << 7)
}

// ---------------------------------------------------------------------------
// Hi/Lo split computation
// ---------------------------------------------------------------------------

/// Compute the upper 20-bit and lower 12-bit split for a value.
///
/// CRITICAL: When the lower 12 bits are negative (bit 11 is set), the upper
/// 20 bits must be incremented by 1 to compensate for the ADDI sign extension.
/// ADDI sign-extends the 12-bit immediate, effectively subtracting from the
/// upper value if bit 11 is set.
///
/// The identity `(hi << 12) + sign_extend_12(lo) == value` always holds.
pub fn compute_hi_lo(value: i64) -> (i32, i32) {
    // Sign-extend the lower 12 bits to get the actual signed lo value
    let lo = ((value & 0xFFF) as i32) << 20 >> 20;
    // Add 0x800 before shifting to compensate for sign extension of lo
    let hi = ((value.wrapping_add(0x800)) >> 12) as i32;
    (hi, lo)
}

// ---------------------------------------------------------------------------
// Overflow detection
// ---------------------------------------------------------------------------

/// Check if a relocation value overflows its target field.
///
/// Returns `Ok(())` if the value fits, or `Err` with a diagnostic message.
/// ADD/SUB/SET types wrap silently and never overflow.
pub fn check_overflow(reloc_type: RiscV64RelocationType, value: i64) -> Result<(), String> {
    match reloc_type {
        RiscV64RelocationType::Branch => {
            // B-type: 13-bit signed, even aligned (±4 KiB)
            if value < -4096 || value >= 4096 || (value & 1) != 0 {
                return Err(format!(
                    "R_RISCV_BRANCH overflow: offset {} out of ±4KiB range or not aligned",
                    value
                ));
            }
        }
        RiscV64RelocationType::Jal => {
            // J-type: 21-bit signed, even aligned (±1 MiB)
            if value < -(1 << 20) || value >= (1 << 20) || (value & 1) != 0 {
                return Err(format!(
                    "R_RISCV_JAL overflow: offset {} out of ±1MiB range or not aligned",
                    value
                ));
            }
        }
        RiscV64RelocationType::Hi20
        | RiscV64RelocationType::PcrelHi20
        | RiscV64RelocationType::GotHi20 => {
            // U-type: upper 20 bits after sign-extension adjustment
            let hi = value.wrapping_add(0x800) >> 12;
            if hi < -(1i64 << 19) || hi >= (1i64 << 19) {
                return Err(format!(
                    "R_RISCV_*_HI20 overflow: value {} (hi={}) out of 20-bit signed range",
                    value, hi
                ));
            }
        }
        RiscV64RelocationType::Call | RiscV64RelocationType::CallPlt => {
            // AUIPC+JALR pair: ±2 GiB (32-bit signed range)
            if value < -(1i64 << 31) || value >= (1i64 << 31) {
                return Err(format!(
                    "R_RISCV_CALL overflow: offset {} out of ±2GiB range",
                    value
                ));
            }
        }
        // ADD/SUB/SET types wrap, R32/R64 are full width, others are no-op
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Relocation application (instruction patching)
// ---------------------------------------------------------------------------

/// Apply a RISC-V relocation to machine code bytes.
///
/// This is the core function that patches instruction or data bytes at link
/// time based on resolved symbol addresses.
///
/// # Arguments
///
/// * `code` — Mutable byte slice containing the section data.
/// * `offset` — Byte offset within `code` where the relocation applies.
/// * `reloc_type` — The RISC-V relocation type.
/// * `symbol_value` — Resolved absolute address of the target symbol (S).
/// * `addend` — Relocation addend (A) from `Elf64_Rela`.
/// * `reloc_address` — Address of the relocation site itself (P).
///
/// # Returns
///
/// `Ok(())` on success, `Err(String)` with a diagnostic on overflow or error.
pub fn apply_relocation(
    code: &mut [u8],
    offset: usize,
    reloc_type: RiscV64RelocationType,
    symbol_value: u64,
    addend: i64,
    reloc_address: u64,
) -> Result<(), String> {
    match reloc_type {
        // ------------------------------------------------------------------
        // No-op
        // ------------------------------------------------------------------
        RiscV64RelocationType::None => Ok(()),

        // ------------------------------------------------------------------
        // Absolute data relocations
        // ------------------------------------------------------------------
        RiscV64RelocationType::R32 => {
            let val = (symbol_value as i64).wrapping_add(addend) as u32;
            if offset + 4 > code.len() {
                return Err("R_RISCV_32: offset out of bounds".into());
            }
            write_u32_le(code, offset, val);
            Ok(())
        }
        RiscV64RelocationType::R64 => {
            let val = (symbol_value as i64).wrapping_add(addend) as u64;
            if offset + 8 > code.len() {
                return Err("R_RISCV_64: offset out of bounds".into());
            }
            write_u64_le(code, offset, val);
            Ok(())
        }

        // ------------------------------------------------------------------
        // B-type: conditional branch
        // ------------------------------------------------------------------
        RiscV64RelocationType::Branch => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            check_overflow(RiscV64RelocationType::Branch, value)?;
            if offset + 4 > code.len() {
                return Err("R_RISCV_BRANCH: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_b_imm(insn, value as i32);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // J-type: unconditional jump
        // ------------------------------------------------------------------
        RiscV64RelocationType::Jal => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            check_overflow(RiscV64RelocationType::Jal, value)?;
            if offset + 4 > code.len() {
                return Err("R_RISCV_JAL: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_j_imm(insn, value as i32);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // AUIPC+JALR call pair (patches 2 consecutive instructions)
        // ------------------------------------------------------------------
        RiscV64RelocationType::Call | RiscV64RelocationType::CallPlt => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            check_overflow(reloc_type, value)?;
            if offset + 8 > code.len() {
                return Err("R_RISCV_CALL: need 8 bytes for AUIPC+JALR pair".into());
            }
            let (hi, lo) = compute_hi_lo(value);
            // Patch AUIPC at offset (U-type: upper 20 bits)
            let auipc = read_insn32(code, offset);
            let auipc_patched = insert_u_imm(auipc, hi << 12);
            write_insn32(code, offset, auipc_patched);
            // Patch JALR at offset+4 (I-type: lower 12 bits)
            let jalr = read_insn32(code, offset + 4);
            let jalr_patched = insert_i_imm(jalr, lo);
            write_insn32(code, offset + 4, jalr_patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // PC-relative upper 20 bits (AUIPC)
        // ------------------------------------------------------------------
        RiscV64RelocationType::PcrelHi20 => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            check_overflow(RiscV64RelocationType::PcrelHi20, value)?;
            if offset + 4 > code.len() {
                return Err("R_RISCV_PCREL_HI20: offset out of bounds".into());
            }
            let (hi, _lo) = compute_hi_lo(value);
            let insn = read_insn32(code, offset);
            let patched = insert_u_imm(insn, hi << 12);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // PC-relative lower 12 bits, I-type (ADDI/LD after AUIPC)
        //
        // CRITICAL: In a full linker implementation, the actual value comes
        // from the corresponding PCREL_HI20 relocation's computation. Here
        // the caller must pass `symbol_value` as the computed PC-relative
        // result and `reloc_address` as the AUIPC label address.
        // ------------------------------------------------------------------
        RiscV64RelocationType::PcrelLo12I => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            let lo = (value & 0xFFF) as i32;
            if offset + 4 > code.len() {
                return Err("R_RISCV_PCREL_LO12_I: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_i_imm(insn, lo);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // PC-relative lower 12 bits, S-type (SW/SD after AUIPC)
        // ------------------------------------------------------------------
        RiscV64RelocationType::PcrelLo12S => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            let lo = (value & 0xFFF) as i32;
            if offset + 4 > code.len() {
                return Err("R_RISCV_PCREL_LO12_S: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_s_imm(insn, lo);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Absolute upper 20 bits (LUI)
        // ------------------------------------------------------------------
        RiscV64RelocationType::Hi20 => {
            let value = (symbol_value as i64).wrapping_add(addend);
            check_overflow(RiscV64RelocationType::Hi20, value)?;
            if offset + 4 > code.len() {
                return Err("R_RISCV_HI20: offset out of bounds".into());
            }
            let (hi, _lo) = compute_hi_lo(value);
            let insn = read_insn32(code, offset);
            let patched = insert_u_imm(insn, hi << 12);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Absolute lower 12 bits, I-type (ADDI/LD)
        // ------------------------------------------------------------------
        RiscV64RelocationType::Lo12I => {
            let value = (symbol_value as i64).wrapping_add(addend);
            let lo = (value & 0xFFF) as i32;
            if offset + 4 > code.len() {
                return Err("R_RISCV_LO12_I: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_i_imm(insn, lo);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Absolute lower 12 bits, S-type (SW/SD)
        // ------------------------------------------------------------------
        RiscV64RelocationType::Lo12S => {
            let value = (symbol_value as i64).wrapping_add(addend);
            let lo = (value & 0xFFF) as i32;
            if offset + 4 > code.len() {
                return Err("R_RISCV_LO12_S: offset out of bounds".into());
            }
            let insn = read_insn32(code, offset);
            let patched = insert_s_imm(insn, lo);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // GOT upper 20 bits (AUIPC for GOT access)
        // ------------------------------------------------------------------
        RiscV64RelocationType::GotHi20 => {
            // G + A - P where G = symbol_value (GOT entry address)
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            check_overflow(RiscV64RelocationType::GotHi20, value)?;
            if offset + 4 > code.len() {
                return Err("R_RISCV_GOT_HI20: offset out of bounds".into());
            }
            let (hi, _lo) = compute_hi_lo(value);
            let insn = read_insn32(code, offset);
            let patched = insert_u_imm(insn, hi << 12);
            write_insn32(code, offset, patched);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Addend arithmetic: ADD
        // ------------------------------------------------------------------
        RiscV64RelocationType::Add32 => {
            if offset + 4 > code.len() {
                return Err("R_RISCV_ADD32: offset out of bounds".into());
            }
            let existing = read_u32_le(code, offset);
            let delta = (symbol_value as i64).wrapping_add(addend) as u32;
            write_u32_le(code, offset, existing.wrapping_add(delta));
            Ok(())
        }
        RiscV64RelocationType::Add64 => {
            if offset + 8 > code.len() {
                return Err("R_RISCV_ADD64: offset out of bounds".into());
            }
            let existing = read_u64_le(code, offset);
            let delta = (symbol_value as i64).wrapping_add(addend) as u64;
            write_u64_le(code, offset, existing.wrapping_add(delta));
            Ok(())
        }

        // ------------------------------------------------------------------
        // Addend arithmetic: SUB
        // ------------------------------------------------------------------
        RiscV64RelocationType::Sub32 => {
            if offset + 4 > code.len() {
                return Err("R_RISCV_SUB32: offset out of bounds".into());
            }
            let existing = read_u32_le(code, offset);
            let delta = (symbol_value as i64).wrapping_add(addend) as u32;
            write_u32_le(code, offset, existing.wrapping_sub(delta));
            Ok(())
        }
        RiscV64RelocationType::Sub64 => {
            if offset + 8 > code.len() {
                return Err("R_RISCV_SUB64: offset out of bounds".into());
            }
            let existing = read_u64_le(code, offset);
            let delta = (symbol_value as i64).wrapping_add(addend) as u64;
            write_u64_le(code, offset, existing.wrapping_sub(delta));
            Ok(())
        }
        RiscV64RelocationType::Sub6 => {
            if offset >= code.len() {
                return Err("R_RISCV_SUB6: offset out of bounds".into());
            }
            let existing = code[offset];
            let delta = ((symbol_value as i64).wrapping_add(addend) & 0x3F) as u8;
            code[offset] = (existing & 0xC0) | ((existing & 0x3F).wrapping_sub(delta) & 0x3F);
            Ok(())
        }

        // ------------------------------------------------------------------
        // SET operations (DWARF/CIE/FDE)
        // ------------------------------------------------------------------
        RiscV64RelocationType::Set6 => {
            if offset >= code.len() {
                return Err("R_RISCV_SET6: offset out of bounds".into());
            }
            let val = ((symbol_value as i64).wrapping_add(addend) & 0x3F) as u8;
            code[offset] = (code[offset] & 0xC0) | val;
            Ok(())
        }
        RiscV64RelocationType::Set8 => {
            if offset >= code.len() {
                return Err("R_RISCV_SET8: offset out of bounds".into());
            }
            code[offset] = (symbol_value as i64).wrapping_add(addend) as u8;
            Ok(())
        }
        RiscV64RelocationType::Set16 => {
            if offset + 2 > code.len() {
                return Err("R_RISCV_SET16: offset out of bounds".into());
            }
            let val = (symbol_value as i64).wrapping_add(addend) as u16;
            write_u16_le(code, offset, val);
            Ok(())
        }
        RiscV64RelocationType::Set32 => {
            if offset + 4 > code.len() {
                return Err("R_RISCV_SET32: offset out of bounds".into());
            }
            let val = (symbol_value as i64).wrapping_add(addend) as u32;
            write_u32_le(code, offset, val);
            Ok(())
        }

        // ------------------------------------------------------------------
        // 32-bit PC-relative (for .eh_frame)
        // ------------------------------------------------------------------
        RiscV64RelocationType::R32Pcrel => {
            let value = (symbol_value as i64)
                .wrapping_add(addend)
                .wrapping_sub(reloc_address as i64);
            if offset + 4 > code.len() {
                return Err("R_RISCV_32_PCREL: offset out of bounds".into());
            }
            write_u32_le(code, offset, value as u32);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Relaxation hint — no-op at link time
        // ------------------------------------------------------------------
        RiscV64RelocationType::Relax => Ok(()),

        // ------------------------------------------------------------------
        // Alignment — NOP sled management for linker relaxation.
        // Without relaxation, ALIGN is a no-op; the assembler has already
        // inserted the appropriate NOP padding.
        // ------------------------------------------------------------------
        RiscV64RelocationType::Align => Ok(()),

        // ------------------------------------------------------------------
        // Dynamic linker relocations — written into .rela.dyn / .rela.plt
        // sections by the linker and processed by the dynamic linker at
        // load time. For static linking these should not appear.
        // ------------------------------------------------------------------
        RiscV64RelocationType::Relative
        | RiscV64RelocationType::DynRelative
        | RiscV64RelocationType::Copy
        | RiscV64RelocationType::JumpSlot
        | RiscV64RelocationType::GlobDat => Ok(()),

        // ------------------------------------------------------------------
        // TLS relocations — handled by the dynamic linker or resolved by
        // the static linker for local-exec TLS. Written to dynamic
        // relocation sections.
        // ------------------------------------------------------------------
        RiscV64RelocationType::TlsDtpmod64
        | RiscV64RelocationType::TlsDtprel64
        | RiscV64RelocationType::TlsTprel64 => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Linker relaxation analysis
// ---------------------------------------------------------------------------

/// Determine if a relocation+instruction pair can be relaxed (shortened).
///
/// RISC-V linker relaxation can convert:
/// - AUIPC+JALR (8 bytes) → JAL (4 bytes) if target is within ±1 MiB
/// - AUIPC+ADDI (8 bytes) → just ADDI when value fits in 12-bit signed
/// - ALIGN NOP sled → fewer NOPs when surrounding code shrinks
///
/// Returns `Some(action)` describing the relaxation, or `None` if the
/// relocation type is not a candidate at all.
pub fn can_relax(
    reloc_type: RiscV64RelocationType,
    value: i64,
    paired_with_relax: bool,
) -> Option<RelaxationAction> {
    if !paired_with_relax {
        return Some(RelaxationAction::NoRelax);
    }
    match reloc_type {
        RiscV64RelocationType::Call | RiscV64RelocationType::CallPlt => {
            // AUIPC+JALR → JAL if target is within ±1 MiB and even-aligned
            if value >= -(1 << 20) && value < (1 << 20) && (value & 1) == 0 {
                Some(RelaxationAction::CallToJal)
            } else {
                Some(RelaxationAction::NoRelax)
            }
        }
        RiscV64RelocationType::PcrelHi20 => {
            // AUIPC+ADDI → just ADDI if value fits in signed 12-bit
            if value >= -2048 && value < 2048 {
                Some(RelaxationAction::PcrelToAddi)
            } else {
                Some(RelaxationAction::NoRelax)
            }
        }
        RiscV64RelocationType::Align => {
            // Alignment NOP sled — the addend gives the required alignment.
            // After surrounding code shrinks, some NOP bytes may be removed.
            let sled_size = value.unsigned_abs() as usize;
            if sled_size > 0 {
                Some(RelaxationAction::DeleteAlign {
                    bytes_to_remove: sled_size,
                })
            } else {
                Some(RelaxationAction::NoRelax)
            }
        }
        _ => Some(RelaxationAction::NoRelax),
    }
}

// ---------------------------------------------------------------------------
// RiscV64RelocationHandler — RelocationHandler trait implementation
// ---------------------------------------------------------------------------

/// Architecture-specific relocation handler for RISC-V 64.
///
/// Implements the [`RelocationHandler`] trait from `linker_common::relocation`,
/// providing RISC-V–specific relocation classification, naming, sizing, and
/// application for the common linker framework.
pub struct RiscV64RelocationHandler;

impl RelocationHandler for RiscV64RelocationHandler {
    /// Classify a raw ELF relocation type code into a [`RelocCategory`].
    fn classify(&self, rel_type: u32) -> RelocCategory {
        let rt = RiscV64RelocationType::from_u32(rel_type);
        rt.category()
    }

    /// Return the human-readable name for a relocation type code.
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match rel_type {
            0 => "R_RISCV_NONE",
            1 => "R_RISCV_32",
            2 => "R_RISCV_64",
            3 => "R_RISCV_RELATIVE",
            4 => "R_RISCV_COPY",
            5 => "R_RISCV_JUMP_SLOT",
            6 => "R_RISCV_GLOB_DAT",
            7 => "R_RISCV_TLS_DTPMOD64",
            9 => "R_RISCV_TLS_DTPREL64",
            11 => "R_RISCV_TLS_TPREL64",
            16 => "R_RISCV_BRANCH",
            17 => "R_RISCV_JAL",
            18 => "R_RISCV_CALL",
            19 => "R_RISCV_CALL_PLT",
            20 => "R_RISCV_GOT_HI20",
            23 => "R_RISCV_PCREL_HI20",
            24 => "R_RISCV_PCREL_LO12_I",
            25 => "R_RISCV_PCREL_LO12_S",
            26 => "R_RISCV_HI20",
            27 => "R_RISCV_LO12_I",
            28 => "R_RISCV_LO12_S",
            35 => "R_RISCV_ADD32",
            36 => "R_RISCV_ADD64",
            39 => "R_RISCV_SUB32",
            40 => "R_RISCV_SUB64",
            43 => "R_RISCV_ALIGN",
            51 => "R_RISCV_RELAX",
            52 => "R_RISCV_SUB6",
            53 => "R_RISCV_SET6",
            54 => "R_RISCV_SET8",
            55 => "R_RISCV_SET16",
            56 => "R_RISCV_SET32",
            57 => "R_RISCV_32_PCREL",
            _ => "R_RISCV_UNKNOWN",
        }
    }

    /// Return the patch size in bytes for a relocation type code.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        let rt = RiscV64RelocationType::from_u32(rel_type);
        match rt {
            RiscV64RelocationType::R64
            | RiscV64RelocationType::Add64
            | RiscV64RelocationType::Sub64 => 8,
            RiscV64RelocationType::R32
            | RiscV64RelocationType::Add32
            | RiscV64RelocationType::Sub32
            | RiscV64RelocationType::Set32
            | RiscV64RelocationType::R32Pcrel
            | RiscV64RelocationType::Branch
            | RiscV64RelocationType::Jal
            | RiscV64RelocationType::Hi20
            | RiscV64RelocationType::Lo12I
            | RiscV64RelocationType::Lo12S
            | RiscV64RelocationType::PcrelHi20
            | RiscV64RelocationType::PcrelLo12I
            | RiscV64RelocationType::PcrelLo12S
            | RiscV64RelocationType::GotHi20
            | RiscV64RelocationType::Call
            | RiscV64RelocationType::CallPlt => 4,
            RiscV64RelocationType::Set16 => 2,
            RiscV64RelocationType::Set8
            | RiscV64RelocationType::Set6
            | RiscV64RelocationType::Sub6 => 1,
            // RELAX, ALIGN, dynamic, TLS, None → 0 (no direct patching)
            _ => 0,
        }
    }

    /// Apply a resolved relocation to section data.
    ///
    /// Extracts the necessary fields from `rel` and delegates to the
    /// standalone [`apply_relocation`] function.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let rt = RiscV64RelocationType::from_u32(rel.rel_type);
        let offset = rel.patch_offset as usize;

        // For GOT relocations, use the GOT entry address as effective symbol
        let effective_symbol = match rt {
            RiscV64RelocationType::GotHi20 => rel.got_address.unwrap_or(rel.symbol_value),
            _ => rel.symbol_value,
        };

        match apply_relocation(
            section_data,
            offset,
            rt,
            effective_symbol,
            rel.addend,
            rel.patch_address,
        ) {
            Ok(()) => Ok(()),
            Err(_msg) => {
                let computed_value = (effective_symbol as i128)
                    .wrapping_add(rel.addend as i128)
                    .wrapping_sub(rel.patch_address as i128);
                Err(RelocationError::Overflow {
                    reloc_name: self.reloc_name(rel.rel_type).to_string(),
                    value: computed_value,
                    bit_width: rt.value_width(),
                    location: format!("0x{:x}", rel.patch_address),
                })
            }
        }
    }

    /// Whether the given relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool {
        let rt = RiscV64RelocationType::from_u32(rel_type);
        rt.needs_got()
    }

    /// Whether the given relocation type requires a PLT entry.
    fn needs_plt(&self, rel_type: u32) -> bool {
        let rt = RiscV64RelocationType::from_u32(rel_type);
        rt.needs_plt()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reloc_type_round_trip() {
        let types: &[(u32, RiscV64RelocationType)] = &[
            (0, RiscV64RelocationType::None),
            (1, RiscV64RelocationType::R32),
            (2, RiscV64RelocationType::R64),
            (3, RiscV64RelocationType::Relative),
            (4, RiscV64RelocationType::Copy),
            (5, RiscV64RelocationType::JumpSlot),
            (6, RiscV64RelocationType::GlobDat),
            (7, RiscV64RelocationType::TlsDtpmod64),
            (9, RiscV64RelocationType::TlsDtprel64),
            (11, RiscV64RelocationType::TlsTprel64),
            (16, RiscV64RelocationType::Branch),
            (17, RiscV64RelocationType::Jal),
            (18, RiscV64RelocationType::Call),
            (19, RiscV64RelocationType::CallPlt),
            (20, RiscV64RelocationType::GotHi20),
            (23, RiscV64RelocationType::PcrelHi20),
            (24, RiscV64RelocationType::PcrelLo12I),
            (25, RiscV64RelocationType::PcrelLo12S),
            (26, RiscV64RelocationType::Hi20),
            (27, RiscV64RelocationType::Lo12I),
            (28, RiscV64RelocationType::Lo12S),
            (35, RiscV64RelocationType::Add32),
            (36, RiscV64RelocationType::Add64),
            (39, RiscV64RelocationType::Sub32),
            (40, RiscV64RelocationType::Sub64),
            (43, RiscV64RelocationType::Align),
            (51, RiscV64RelocationType::Relax),
            (52, RiscV64RelocationType::Sub6),
            (53, RiscV64RelocationType::Set6),
            (54, RiscV64RelocationType::Set8),
            (55, RiscV64RelocationType::Set16),
            (56, RiscV64RelocationType::Set32),
            (57, RiscV64RelocationType::R32Pcrel),
        ];
        for &(elf_val, ref expected) in types {
            let rt = RiscV64RelocationType::from_u32(elf_val);
            assert_eq!(rt, *expected, "from_u32({}) mismatch", elf_val);
            assert_eq!(rt.as_elf_type(), elf_val, "as_elf_type for {:?}", expected);
        }
        // DynRelative maps to 3 (same as Relative)
        assert_eq!(RiscV64RelocationType::DynRelative.as_elf_type(), 3);
    }

    #[test]
    fn test_unknown_reloc_type() {
        assert_eq!(
            RiscV64RelocationType::from_u32(999),
            RiscV64RelocationType::None,
        );
    }

    #[test]
    fn test_classification() {
        // PC-relative
        assert!(RiscV64RelocationType::Branch.is_pc_relative());
        assert!(RiscV64RelocationType::Jal.is_pc_relative());
        assert!(RiscV64RelocationType::Call.is_pc_relative());
        assert!(RiscV64RelocationType::CallPlt.is_pc_relative());
        assert!(RiscV64RelocationType::PcrelHi20.is_pc_relative());
        assert!(RiscV64RelocationType::PcrelLo12I.is_pc_relative());
        assert!(RiscV64RelocationType::PcrelLo12S.is_pc_relative());
        assert!(RiscV64RelocationType::GotHi20.is_pc_relative());
        assert!(RiscV64RelocationType::R32Pcrel.is_pc_relative());
        assert!(!RiscV64RelocationType::R32.is_pc_relative());
        assert!(!RiscV64RelocationType::Hi20.is_pc_relative());
        assert!(!RiscV64RelocationType::Lo12I.is_pc_relative());

        // GOT
        assert!(RiscV64RelocationType::GotHi20.needs_got());
        assert!(!RiscV64RelocationType::Call.needs_got());
        assert!(!RiscV64RelocationType::PcrelHi20.needs_got());

        // PLT
        assert!(RiscV64RelocationType::CallPlt.needs_plt());
        assert!(!RiscV64RelocationType::Call.needs_plt());

        // Relaxation
        assert!(RiscV64RelocationType::Relax.is_relaxation_hint());
        assert!(RiscV64RelocationType::Align.is_relaxation_hint());
        assert!(!RiscV64RelocationType::Call.is_relaxation_hint());

        // Dynamic
        assert!(RiscV64RelocationType::Copy.is_dynamic());
        assert!(RiscV64RelocationType::JumpSlot.is_dynamic());
        assert!(RiscV64RelocationType::GlobDat.is_dynamic());
        assert!(RiscV64RelocationType::DynRelative.is_dynamic());
        assert!(RiscV64RelocationType::TlsDtpmod64.is_dynamic());
        assert!(!RiscV64RelocationType::Branch.is_dynamic());
    }

    #[test]
    fn test_category_mapping() {
        assert_eq!(
            RiscV64RelocationType::Branch.category(),
            RelocCategory::PcRelative,
        );
        assert_eq!(
            RiscV64RelocationType::R32.category(),
            RelocCategory::Absolute,
        );
        assert_eq!(
            RiscV64RelocationType::Add32.category(),
            RelocCategory::Absolute,
        );
        assert_eq!(
            RiscV64RelocationType::Relax.category(),
            RelocCategory::Other,
        );
        assert_eq!(
            RiscV64RelocationType::TlsDtpmod64.category(),
            RelocCategory::Tls,
        );
    }

    #[test]
    fn test_compute_hi_lo_no_adjustment() {
        let (hi, lo) = compute_hi_lo(0x12345000);
        assert_eq!(hi, 0x12345);
        assert_eq!(lo, 0);
        assert_eq!((hi as i64) * 4096 + lo as i64, 0x12345000);
    }

    #[test]
    fn test_compute_hi_lo_with_adjustment() {
        // 0x800 → bit 11 set → sign-extend to -2048 → hi += 1
        let (hi, lo) = compute_hi_lo(0x12345800);
        assert_eq!(hi, 0x12346);
        assert_eq!(lo, -2048);
        assert_eq!((hi as i64) * 4096 + lo as i64, 0x12345800);
    }

    #[test]
    fn test_compute_hi_lo_positive_lo() {
        // 0x678 → bit 11 = 0 → positive, no adjustment
        let (hi, lo) = compute_hi_lo(0x12345678);
        assert_eq!(hi, 0x12345);
        assert_eq!(lo, 0x678);
        assert_eq!((hi as i64) * 4096 + lo as i64, 0x12345678);
    }

    #[test]
    fn test_compute_hi_lo_max_lo() {
        // 0xFFF → sign-extend to -1 → hi adjusted
        let (hi, lo) = compute_hi_lo(0x12345FFF);
        assert_eq!(lo, -1);
        assert_eq!(hi, 0x12346);
        assert_eq!((hi as i64) * 4096 + lo as i64, 0x12345FFF);
    }

    #[test]
    fn test_compute_hi_lo_zero() {
        let (hi, lo) = compute_hi_lo(0);
        assert_eq!(hi, 0);
        assert_eq!(lo, 0);
    }

    #[test]
    fn test_compute_hi_lo_negative() {
        let (hi, lo) = compute_hi_lo(-4096);
        assert_eq!((hi as i64) * 4096 + lo as i64, -4096);
    }

    #[test]
    fn test_compute_hi_lo_small_negative() {
        let (hi, lo) = compute_hi_lo(-1);
        assert_eq!((hi as i64) * 4096 + lo as i64, -1);
    }

    #[test]
    fn test_b_type_immediate_round_trip() {
        let base_insn = 0x00000063_u32; // BEQ x0, x0
                                        // Positive offset
        let imm = 256i32;
        let patched = insert_b_imm(base_insn, imm);
        assert_eq!(extract_b_imm(patched), imm);
        // Negative offset
        let imm = -128i32;
        let patched = insert_b_imm(base_insn, imm);
        assert_eq!(extract_b_imm(patched), imm);
        // Max positive
        let imm = 4094i32;
        let patched = insert_b_imm(base_insn, imm);
        assert_eq!(extract_b_imm(patched), imm);
        // Max negative
        let imm = -4096i32;
        let patched = insert_b_imm(base_insn, imm);
        assert_eq!(extract_b_imm(patched), imm);
    }

    #[test]
    fn test_j_type_immediate_round_trip() {
        let base_insn = 0x0000006F_u32; // JAL x0
        let imm = 1024i32;
        let patched = insert_j_imm(base_insn, imm);
        assert_eq!(extract_j_imm(patched), imm);
        let imm = -2048i32;
        let patched = insert_j_imm(base_insn, imm);
        assert_eq!(extract_j_imm(patched), imm);
        // Large positive
        let imm = (1 << 20) - 2;
        let patched = insert_j_imm(base_insn, imm);
        assert_eq!(extract_j_imm(patched), imm);
        // Large negative
        let imm = -(1 << 20);
        let patched = insert_j_imm(base_insn, imm);
        assert_eq!(extract_j_imm(patched), imm);
    }

    #[test]
    fn test_apply_r32() {
        let mut code = [0u8; 8];
        apply_relocation(&mut code, 0, RiscV64RelocationType::R32, 0x1000, 0x10, 0).unwrap();
        assert_eq!(read_u32_le(&code, 0), 0x1010);
    }

    #[test]
    fn test_apply_r64() {
        let mut code = [0u8; 8];
        apply_relocation(
            &mut code,
            0,
            RiscV64RelocationType::R64,
            0xFFFF_FFFF_0000,
            0x100,
            0,
        )
        .unwrap();
        assert_eq!(read_u64_le(&code, 0), 0xFFFF_FFFF_0100);
    }

    #[test]
    fn test_apply_branch() {
        let mut code = [0u8; 4];
        write_insn32(&mut code, 0, 0x00000063); // BEQ x0, x0, 0
        apply_relocation(
            &mut code,
            0,
            RiscV64RelocationType::Branch,
            0x1100,
            0,
            0x1000,
        )
        .unwrap();
        let insn = read_insn32(&code, 0);
        assert_eq!(extract_b_imm(insn), 0x100);
    }

    #[test]
    fn test_apply_branch_overflow() {
        let mut code = [0u8; 4];
        write_insn32(&mut code, 0, 0x00000063);
        let result = apply_relocation(
            &mut code,
            0,
            RiscV64RelocationType::Branch,
            0x2000,
            0,
            0x0000,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_jal() {
        let mut code = [0u8; 4];
        write_insn32(&mut code, 0, 0x0000006F); // JAL x0
        apply_relocation(&mut code, 0, RiscV64RelocationType::Jal, 0x1400, 0, 0x1000).unwrap();
        let insn = read_insn32(&code, 0);
        assert_eq!(extract_j_imm(insn), 0x400);
    }

    #[test]
    fn test_apply_call_pair() {
        let mut code = [0u8; 8];
        write_insn32(&mut code, 0, 0x00000097); // AUIPC ra
        write_insn32(&mut code, 4, 0x000080E7); // JALR ra, ra, 0
        let sym = 0x12345000u64;
        let p = 0x10000000u64;
        let value = sym as i64 - p as i64;
        apply_relocation(&mut code, 0, RiscV64RelocationType::Call, sym, 0, p).unwrap();
        let (hi, lo) = compute_hi_lo(value);
        // Verify AUIPC upper 20 bits
        let auipc = read_insn32(&code, 0);
        assert_eq!(auipc & 0xFFFFF000, ((hi as u32) << 12) & 0xFFFFF000);
        // Verify JALR lower 12 bits
        let jalr = read_insn32(&code, 4);
        let jalr_imm = ((jalr >> 20) as i32) << 20 >> 20;
        assert_eq!(jalr_imm, lo);
        // Verify reconstruction
        assert_eq!((hi as i64) * 4096 + lo as i64, value);
    }

    #[test]
    fn test_apply_add32_sub32() {
        let mut code = [0u8; 4];
        write_u32_le(&mut code, 0, 100);
        apply_relocation(&mut code, 0, RiscV64RelocationType::Add32, 50, 10, 0).unwrap();
        assert_eq!(read_u32_le(&code, 0), 160);
        apply_relocation(&mut code, 0, RiscV64RelocationType::Sub32, 20, 5, 0).unwrap();
        assert_eq!(read_u32_le(&code, 0), 135);
    }

    #[test]
    fn test_apply_add64_sub64() {
        let mut code = [0u8; 8];
        write_u64_le(&mut code, 0, 1000);
        apply_relocation(&mut code, 0, RiscV64RelocationType::Add64, 500, 100, 0).unwrap();
        assert_eq!(read_u64_le(&code, 0), 1600);
        apply_relocation(&mut code, 0, RiscV64RelocationType::Sub64, 200, 50, 0).unwrap();
        assert_eq!(read_u64_le(&code, 0), 1350);
    }

    #[test]
    fn test_apply_relax_noop() {
        let mut code = [0u8; 4];
        write_insn32(&mut code, 0, 0xDEADBEEF);
        apply_relocation(&mut code, 0, RiscV64RelocationType::Relax, 0, 0, 0).unwrap();
        assert_eq!(read_insn32(&code, 0), 0xDEADBEEF);
    }

    #[test]
    fn test_apply_set_operations() {
        // SET6
        let mut code = [0u8; 4];
        code[0] = 0xFF;
        apply_relocation(&mut code, 0, RiscV64RelocationType::Set6, 0x15, 0, 0).unwrap();
        assert_eq!(code[0], 0xD5); // 0xC0 | 0x15

        // SET8
        apply_relocation(&mut code, 0, RiscV64RelocationType::Set8, 0xAB, 0, 0).unwrap();
        assert_eq!(code[0], 0xAB);

        // SET16
        apply_relocation(&mut code, 0, RiscV64RelocationType::Set16, 0x1234, 0, 0).unwrap();
        assert_eq!(read_u16_le(&code, 0), 0x1234);

        // SET32
        apply_relocation(&mut code, 0, RiscV64RelocationType::Set32, 0xDEADBEEF, 0, 0).unwrap();
        assert_eq!(read_u32_le(&code, 0), 0xDEADBEEF);
    }

    #[test]
    fn test_apply_sub6() {
        let mut code = [0u8; 1];
        code[0] = 0xDF; // upper 2 bits = 0xC0, lower 6 = 0x1F = 31
        apply_relocation(&mut code, 0, RiscV64RelocationType::Sub6, 5, 0, 0).unwrap();
        // 31 - 5 = 26, upper bits preserved
        assert_eq!(code[0], 0xDA); // 0xC0 | 26
    }

    #[test]
    fn test_apply_r32_pcrel() {
        let mut code = [0u8; 4];
        apply_relocation(
            &mut code,
            0,
            RiscV64RelocationType::R32Pcrel,
            0x2000,
            0,
            0x1000,
        )
        .unwrap();
        assert_eq!(read_u32_le(&code, 0), 0x1000); // S + A - P
    }

    #[test]
    fn test_overflow_branch() {
        assert!(check_overflow(RiscV64RelocationType::Branch, 4000).is_ok());
        assert!(check_overflow(RiscV64RelocationType::Branch, -4096).is_ok());
        assert!(check_overflow(RiscV64RelocationType::Branch, 4096).is_err());
        assert!(check_overflow(RiscV64RelocationType::Branch, 3).is_err());
    }

    #[test]
    fn test_overflow_jal() {
        assert!(check_overflow(RiscV64RelocationType::Jal, 1 << 19).is_ok());
        assert!(check_overflow(RiscV64RelocationType::Jal, 1 << 20).is_err());
        assert!(check_overflow(RiscV64RelocationType::Jal, 1).is_err()); // odd
    }

    #[test]
    fn test_overflow_call() {
        assert!(check_overflow(RiscV64RelocationType::Call, 0).is_ok());
        assert!(check_overflow(RiscV64RelocationType::Call, (1i64 << 31) - 1).is_ok());
        assert!(check_overflow(RiscV64RelocationType::Call, 1i64 << 31).is_err());
    }

    #[test]
    fn test_can_relax_call_to_jal() {
        assert_eq!(
            can_relax(RiscV64RelocationType::Call, 0x1000, true),
            Some(RelaxationAction::CallToJal),
        );
        assert_eq!(
            can_relax(RiscV64RelocationType::Call, 0x200000, true),
            Some(RelaxationAction::NoRelax),
        );
        assert_eq!(
            can_relax(RiscV64RelocationType::Call, 0x1000, false),
            Some(RelaxationAction::NoRelax),
        );
    }

    #[test]
    fn test_can_relax_pcrel_to_addi() {
        assert_eq!(
            can_relax(RiscV64RelocationType::PcrelHi20, 100, true),
            Some(RelaxationAction::PcrelToAddi),
        );
        assert_eq!(
            can_relax(RiscV64RelocationType::PcrelHi20, 5000, true),
            Some(RelaxationAction::NoRelax),
        );
    }

    #[test]
    fn test_handler_classify() {
        let h = RiscV64RelocationHandler;
        assert_eq!(h.classify(16), RelocCategory::PcRelative);
        assert_eq!(h.classify(1), RelocCategory::Absolute);
        assert_eq!(h.classify(51), RelocCategory::Other);
        assert_eq!(h.classify(7), RelocCategory::Tls);
    }

    #[test]
    fn test_handler_reloc_name() {
        let h = RiscV64RelocationHandler;
        assert_eq!(h.reloc_name(0), "R_RISCV_NONE");
        assert_eq!(h.reloc_name(16), "R_RISCV_BRANCH");
        assert_eq!(h.reloc_name(18), "R_RISCV_CALL");
        assert_eq!(h.reloc_name(57), "R_RISCV_32_PCREL");
        assert_eq!(h.reloc_name(999), "R_RISCV_UNKNOWN");
    }

    #[test]
    fn test_handler_reloc_size() {
        let h = RiscV64RelocationHandler;
        assert_eq!(h.reloc_size(2), 8); // R64
        assert_eq!(h.reloc_size(1), 4); // R32
        assert_eq!(h.reloc_size(16), 4); // BRANCH
        assert_eq!(h.reloc_size(55), 2); // SET16
        assert_eq!(h.reloc_size(54), 1); // SET8
        assert_eq!(h.reloc_size(51), 0); // RELAX
    }

    #[test]
    fn test_handler_needs_got_plt() {
        let h = RiscV64RelocationHandler;
        assert!(h.needs_got(20)); // GOT_HI20
        assert!(!h.needs_got(18)); // CALL
        assert!(h.needs_plt(19)); // CALL_PLT
        assert!(!h.needs_plt(18)); // CALL
    }

    #[test]
    fn test_value_width() {
        assert_eq!(RiscV64RelocationType::R32.value_width(), 32);
        assert_eq!(RiscV64RelocationType::R64.value_width(), 64);
        assert_eq!(RiscV64RelocationType::Branch.value_width(), 13);
        assert_eq!(RiscV64RelocationType::Jal.value_width(), 21);
        assert_eq!(RiscV64RelocationType::Hi20.value_width(), 20);
        assert_eq!(RiscV64RelocationType::Lo12I.value_width(), 12);
        assert_eq!(RiscV64RelocationType::Call.value_width(), 32);
        assert_eq!(RiscV64RelocationType::Set6.value_width(), 6);
        assert_eq!(RiscV64RelocationType::Set8.value_width(), 8);
        assert_eq!(RiscV64RelocationType::Set16.value_width(), 16);
        assert_eq!(RiscV64RelocationType::Set32.value_width(), 32);
        assert_eq!(RiscV64RelocationType::Sub6.value_width(), 6);
        assert_eq!(RiscV64RelocationType::None.value_width(), 0);
        assert_eq!(RiscV64RelocationType::Relax.value_width(), 0);
    }

    #[test]
    fn test_out_of_bounds() {
        let mut code = [0u8; 2];
        assert!(apply_relocation(&mut code, 0, RiscV64RelocationType::R32, 0, 0, 0).is_err());
        assert!(apply_relocation(&mut code, 0, RiscV64RelocationType::R64, 0, 0, 0).is_err());
        assert!(apply_relocation(&mut code, 0, RiscV64RelocationType::Branch, 0, 0, 0).is_err());
    }

    #[test]
    fn test_dynamic_relocs_noop() {
        let mut code = [0xAA_u8; 4];
        apply_relocation(&mut code, 0, RiscV64RelocationType::Copy, 0, 0, 0).unwrap();
        apply_relocation(&mut code, 0, RiscV64RelocationType::JumpSlot, 0, 0, 0).unwrap();
        apply_relocation(&mut code, 0, RiscV64RelocationType::GlobDat, 0, 0, 0).unwrap();
        apply_relocation(&mut code, 0, RiscV64RelocationType::DynRelative, 0, 0, 0).unwrap();
        // Code should remain untouched
        assert_eq!(code, [0xAA; 4]);
    }

    #[test]
    fn test_pcrel_hi20_lo12i_pair() {
        // Simulate AUIPC x5, 0 / ADDI x5, x5, 0 pair for a PC-relative access
        let mut code = [0u8; 8];
        write_insn32(&mut code, 0, 0x00000297); // AUIPC x5
        write_insn32(&mut code, 4, 0x00028293); // ADDI x5, x5, 0
        let sym = 0x10002000u64;
        let p_auipc = 0x10000000u64;
        let _p_addi = 0x10000004u64;
        let pcrel_value = sym as i64 - p_auipc as i64; // 0x2000
                                                       // Patch AUIPC with PCREL_HI20
        apply_relocation(
            &mut code,
            0,
            RiscV64RelocationType::PcrelHi20,
            sym,
            0,
            p_auipc,
        )
        .unwrap();
        // Patch ADDI with PCREL_LO12_I — referencing the AUIPC label
        // For PCREL_LO12_I, symbol_value is the pcrel value, reloc_address = p_auipc
        apply_relocation(
            &mut code,
            4,
            RiscV64RelocationType::PcrelLo12I,
            sym,
            0,
            p_auipc,
        )
        .unwrap();
        // Verify the upper/lower split reconstructs to pcrel_value
        let (hi, _lo) = compute_hi_lo(pcrel_value);
        let auipc = read_insn32(&code, 0);
        let hi_actual = ((auipc >> 12) & 0xFFFFF) as i32;
        assert_eq!(hi_actual, hi);
    }
}
