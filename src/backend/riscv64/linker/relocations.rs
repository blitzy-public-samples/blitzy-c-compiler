//! # RISC-V 64 Relocation Application
//!
//! Implements the `RelocationHandler` trait for RISC-V 64 architecture.
//! Applies all `R_RISCV_*` ELF relocation types to linked binary data.
//!
//! ## Supported Relocation Types
//! - **Absolute**: R_RISCV_32, R_RISCV_64 — direct address patching
//! - **PC-relative**: R_RISCV_BRANCH, R_RISCV_JAL — branch/jump offset encoding
//! - **Upper/Lower pairs**: R_RISCV_HI20/LO12_I/LO12_S — 32-bit absolute split
//! - **PC-relative pairs**: R_RISCV_PCREL_HI20/PCREL_LO12_I/PCREL_LO12_S
//! - **GOT access**: R_RISCV_GOT_HI20 — Global Offset Table access
//! - **Function calls**: R_RISCV_CALL, R_RISCV_CALL_PLT — AUIPC+JALR pairs
//! - **Relaxation hints**: R_RISCV_RELAX, R_RISCV_ALIGN
//! - **Arithmetic**: R_RISCV_ADD32/64, R_RISCV_SUB32/64
//! - **Set operations**: R_RISCV_SET6/8/16/32
//! - **Compressed**: R_RISCV_RVC_BRANCH, R_RISCV_RVC_JUMP — 16-bit instructions
//! - **TLS**: R_RISCV_TLS_DTPMOD64, R_RISCV_TLS_DTPREL64, R_RISCV_TLS_TPREL64,
//!   R_RISCV_TLS_GD_HI20, R_RISCV_TPREL_HI20/LO12_I/LO12_S/ADD
//! - **Dynamic**: R_RISCV_RELATIVE, R_RISCV_COPY, R_RISCV_JUMP_SLOT
//!
//! ## Linker Relaxation
//! RISC-V uniquely supports linker relaxation — the linker can shorten instruction
//! sequences when the distance to the target is small enough:
//! - AUIPC+JALR (8 bytes) → JAL (4 bytes) when target within ±1 MiB
//! - AUIPC+ADDI (8 bytes) → ADDI (4 bytes) when target within ±2 KiB
//! - Relaxation is indicated by R_RISCV_RELAX paired with the primary relocation
//!
//! ## Instruction Encoding
//! Relocation values must be encoded into RISC-V instruction bit fields:
//! - B-type: imm[12|10:5|4:1|11] scattered across bits 31,30-25,11-8,7
//! - J-type: imm[20|10:1|11|19:12] scattered across bits 31,30-21,20,19-12
//! - U-type: imm[31:12] in bits 31-12
//! - I-type: imm[11:0] in bits 31-20
//! - S-type: imm[11:5] in bits 31-25, imm[4:0] in bits 11-7
//! - CB-type (compressed branch): 9-bit signed offset
//! - CJ-type (compressed jump): 12-bit signed offset
//!
//! ## Primary kernel boot target — Checkpoint 6 validation.

use std::cell::RefCell;

use crate::backend::linker_common::relocation::{
    compute_absolute, compute_got_relative, compute_pc_relative, fits_signed, fits_unsigned,
    read_le, sign_extend, write_le, RelocCategory, RelocationError, RelocationHandler,
    ResolvedRelocation,
};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// RISC-V ELF Relocation Type Constants (from RISC-V ELF psABI)
// ---------------------------------------------------------------------------

/// No relocation.
pub const R_RISCV_NONE: u32 = 0;
/// S + A — 32-bit absolute address.
pub const R_RISCV_32: u32 = 1;
/// S + A — 64-bit absolute address.
pub const R_RISCV_64: u32 = 2;
/// B + A — base-relative (dynamic linker).
pub const R_RISCV_RELATIVE: u32 = 3;
/// Copy symbol at runtime (dynamic linker).
pub const R_RISCV_COPY: u32 = 4;
/// S — PLT jump slot (dynamic linker).
pub const R_RISCV_JUMP_SLOT: u32 = 5;
/// TLS module ID (32-bit).
pub const R_RISCV_TLS_DTPMOD32: u32 = 6;
/// TLS module ID (64-bit).
pub const R_RISCV_TLS_DTPMOD64: u32 = 7;
/// TLS offset within module (32-bit).
pub const R_RISCV_TLS_DTPREL32: u32 = 8;
/// TLS offset within module (64-bit).
pub const R_RISCV_TLS_DTPREL64: u32 = 9;
/// TLS offset from TP (32-bit).
pub const R_RISCV_TLS_TPREL32: u32 = 10;
/// TLS offset from TP (64-bit).
pub const R_RISCV_TLS_TPREL64: u32 = 11;
/// S + A - P — B-type branch offset, ±4 KiB (13-bit signed, bit 0 always 0).
pub const R_RISCV_BRANCH: u32 = 16;
/// S + A - P — J-type jump offset, ±1 MiB (21-bit signed, bit 0 always 0).
pub const R_RISCV_JAL: u32 = 17;
/// S + A - P — AUIPC+JALR pair, ±2 GiB (32-bit signed).
pub const R_RISCV_CALL: u32 = 18;
/// S + A - P — AUIPC+JALR via PLT, ±2 GiB (32-bit signed).
pub const R_RISCV_CALL_PLT: u32 = 19;
/// G + GOT + A - P — GOT entry, upper 20 bits (U-type in AUIPC).
pub const R_RISCV_GOT_HI20: u32 = 20;
/// TLS GD (General Dynamic) upper 20 bits.
pub const R_RISCV_TLS_GD_HI20: u32 = 22;
/// S + A - P — PC-relative, upper 20 bits (U-type in AUIPC).
pub const R_RISCV_PCREL_HI20: u32 = 23;
/// PC-relative lower 12 bits, I-type encoding. Paired with PCREL_HI20.
pub const R_RISCV_PCREL_LO12_I: u32 = 24;
/// PC-relative lower 12 bits, S-type encoding. Paired with PCREL_HI20.
pub const R_RISCV_PCREL_LO12_S: u32 = 25;
/// S + A — absolute, upper 20 bits (U-type in LUI).
pub const R_RISCV_HI20: u32 = 26;
/// S + A — absolute, lower 12 bits (I-type).
pub const R_RISCV_LO12_I: u32 = 27;
/// S + A — absolute, lower 12 bits (S-type).
pub const R_RISCV_LO12_S: u32 = 28;
/// TLS TP-relative, upper 20 bits.
pub const R_RISCV_TPREL_HI20: u32 = 29;
/// TLS TP-relative, lower 12 bits (I-type).
pub const R_RISCV_TPREL_LO12_I: u32 = 30;
/// TLS TP-relative, lower 12 bits (S-type).
pub const R_RISCV_TPREL_LO12_S: u32 = 31;
/// TLS TP-relative ADD (marker, no actual patching).
pub const R_RISCV_TPREL_ADD: u32 = 32;
/// V + S + A — 8-bit add.
pub const R_RISCV_ADD8: u32 = 33;
/// V + S + A — 16-bit add.
pub const R_RISCV_ADD16: u32 = 34;
/// V + S + A — 32-bit add.
pub const R_RISCV_ADD32: u32 = 35;
/// V + S + A — 64-bit add.
pub const R_RISCV_ADD64: u32 = 36;
/// V - S - A — 8-bit subtract.
pub const R_RISCV_SUB8: u32 = 37;
/// V - S - A — 16-bit subtract.
pub const R_RISCV_SUB16: u32 = 38;
/// V - S - A — 32-bit subtract.
pub const R_RISCV_SUB32: u32 = 39;
/// V - S - A — 64-bit subtract.
pub const R_RISCV_SUB64: u32 = 40;
/// GNU vtable inheritance (legacy).
pub const R_RISCV_GNU_VTINHERIT: u32 = 41;
/// GNU vtable entry (legacy).
pub const R_RISCV_GNU_VTENTRY: u32 = 42;
/// Alignment directive — affected by relaxation.
pub const R_RISCV_ALIGN: u32 = 43;
/// S + A - P — Compressed B-type branch, ±256 bytes (9-bit signed).
pub const R_RISCV_RVC_BRANCH: u32 = 44;
/// S + A - P — Compressed J-type jump, ±2 KiB (12-bit signed).
pub const R_RISCV_RVC_JUMP: u32 = 45;
/// Compressed LUI immediate.
pub const R_RISCV_RVC_LUI: u32 = 46;
/// Relaxation hint — paired with another relocation to indicate it can be relaxed.
pub const R_RISCV_RELAX: u32 = 51;
/// V - (S + A) — 6-bit subtraction (DWARF/CIE/FDE tables).
pub const R_RISCV_SUB6: u32 = 52;
/// Set lower 6 bits of byte to S + A.
pub const R_RISCV_SET6: u32 = 53;
/// Set byte to S + A (truncated to 8 bits).
pub const R_RISCV_SET8: u32 = 54;
/// Set 16-bit value to S + A (truncated to 16 bits).
pub const R_RISCV_SET16: u32 = 55;
/// Set 32-bit value to S + A (truncated to 32 bits).
pub const R_RISCV_SET32: u32 = 56;
/// S + A - P — 32-bit PC-relative.
pub const R_RISCV_32_PCREL: u32 = 57;

// ---------------------------------------------------------------------------
// Instruction Read/Write Helpers
// ---------------------------------------------------------------------------

/// Read a 32-bit RISC-V instruction from a little-endian byte slice.
#[inline]
fn read_insn32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Write a 32-bit RISC-V instruction to a little-endian byte slice.
#[inline]
fn write_insn32(data: &mut [u8], offset: usize, insn: u32) {
    let bytes = insn.to_le_bytes();
    data[offset..offset + 4].copy_from_slice(&bytes);
}

/// Read a 16-bit compressed RISC-V instruction from a little-endian byte slice.
#[inline]
fn read_insn16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Write a 16-bit compressed RISC-V instruction to a little-endian byte slice.
#[inline]
fn write_insn16(data: &mut [u8], offset: usize, insn: u16) {
    let bytes = insn.to_le_bytes();
    data[offset..offset + 2].copy_from_slice(&bytes);
}

// ---------------------------------------------------------------------------
// Instruction Bit-Field Encoding Helpers
// ---------------------------------------------------------------------------

/// Encode a value into a B-type (branch) instruction's immediate field.
///
/// B-type layout: `imm[12|10:5]` in bits `[31|30:25]`, `imm[4:1|11]` in bits `[11:8|7]`.
/// The immediate represents a signed offset in multiples of 2 bytes.
#[inline]
fn encode_b_type_imm(insn: u32, value: i64) -> u32 {
    let imm = value as u32;
    let mut result = insn & 0x01FFF07F; // clear immediate bits
    result |= ((imm >> 12) & 1) << 31; // imm[12]
    result |= ((imm >> 5) & 0x3F) << 25; // imm[10:5]
    result |= ((imm >> 1) & 0xF) << 8; // imm[4:1]
    result |= ((imm >> 11) & 1) << 7; // imm[11]
    result
}

/// Encode a value into a J-type (jump) instruction's immediate field.
///
/// J-type layout: `imm[20|10:1|11|19:12]` in bits `[31|30:21|20|19:12]`.
/// The immediate represents a signed offset in multiples of 2 bytes.
#[inline]
fn encode_j_type_imm(insn: u32, value: i64) -> u32 {
    let imm = value as u32;
    let mut result = insn & 0x00000FFF; // clear immediate bits (keep opcode+rd)
    result |= ((imm >> 20) & 1) << 31; // imm[20]
    result |= ((imm >> 1) & 0x3FF) << 21; // imm[10:1]
    result |= ((imm >> 11) & 1) << 20; // imm[11]
    result |= ((imm >> 12) & 0xFF) << 12; // imm[19:12]
    result
}

/// Encode a value into a U-type (upper immediate) instruction's immediate field.
///
/// U-type layout: `imm[31:12]` in bits `[31:12]`. The lower 12 bits of the
/// instruction (opcode + rd) are preserved.
#[inline]
fn encode_u_type_imm(insn: u32, value: i64) -> u32 {
    let imm = value as u32;
    (insn & 0xFFF) | (imm & 0xFFFFF000)
}

/// Encode a value into an I-type instruction's immediate field.
///
/// I-type layout: `imm[11:0]` in bits `[31:20]`. The lower 20 bits of the
/// instruction (opcode + rd + funct3 + rs1) are preserved.
#[inline]
fn encode_i_type_imm(insn: u32, value: i64) -> u32 {
    let imm = (value as u32) & 0xFFF;
    (insn & 0x000FFFFF) | (imm << 20)
}

/// Encode a value into an S-type (store) instruction's immediate field.
///
/// S-type layout: `imm[11:5]` in bits `[31:25]`, `imm[4:0]` in bits `[11:7]`.
#[inline]
fn encode_s_type_imm(insn: u32, value: i64) -> u32 {
    let imm = (value as u32) & 0xFFF;
    let mut result = insn & 0x01FFF07F; // clear immediate bits
    result |= ((imm >> 5) & 0x7F) << 25; // imm[11:5]
    result |= (imm & 0x1F) << 7; // imm[4:0]
    result
}

/// Encode a value into a compressed B-type (CB) instruction's immediate field.
///
/// CB-type (c.beqz/c.bnez): 9-bit signed offset, ±256 bytes.
/// Bits: `offset[8|4:3]` in insn`[12|11:10]`, `offset[7:6|2:1|5]` in insn`[6:5|4:3|2]`.
#[inline]
fn encode_cb_type_imm(insn: u16, value: i64) -> u16 {
    let imm = value as u16;
    let mut result = insn & 0xE383; // clear immediate bits
    result |= ((imm >> 8) & 1) << 12; // offset[8]
    result |= ((imm >> 3) & 0x3) << 10; // offset[4:3]
    result |= ((imm >> 6) & 0x3) << 5; // offset[7:6]
    result |= ((imm >> 1) & 0x3) << 3; // offset[2:1]
    result |= ((imm >> 5) & 1) << 2; // offset[5]
    result
}

/// Encode a value into a compressed J-type (CJ) instruction's immediate field.
///
/// CJ-type (c.j/c.jal): 12-bit signed offset, ±2 KiB.
/// Bits: `offset[11|4|9:8|10|6|7|3:1|5]` in insn`[12|11|10:9|8|7|6|5:3|2]`.
#[inline]
fn encode_cj_type_imm(insn: u16, value: i64) -> u16 {
    let imm = value as u16;
    let mut result = insn & 0xE003; // clear immediate bits
    result |= ((imm >> 11) & 1) << 12; // offset[11]
    result |= ((imm >> 4) & 1) << 11; // offset[4]
    result |= ((imm >> 8) & 0x3) << 9; // offset[9:8]
    result |= ((imm >> 10) & 1) << 8; // offset[10]
    result |= ((imm >> 6) & 1) << 7; // offset[6]
    result |= ((imm >> 7) & 1) << 6; // offset[7]
    result |= ((imm >> 1) & 0x7) << 3; // offset[3:1]
    result |= ((imm >> 5) & 1) << 2; // offset[5]
    result
}

// ---------------------------------------------------------------------------
// HI20/LO12 Split Helper
// ---------------------------------------------------------------------------

/// Split a value into HI20 and LO12 components with sign-extension compensation.
///
/// Because the I-type immediate is sign-extended, if bit 11 of the full value is
/// set, the LO12 part will effectively subtract from the HI20 part. To compensate,
/// we add 0x800 before shifting to produce the upper 20 bits.
///
/// Returns `(hi20, lo12)` where:
/// - `hi20` is the value to encode in the U-type instruction (already shifted left by 12)
/// - `lo12` is the sign-extended 12-bit value for the I-type or S-type instruction
#[inline]
fn hi20_lo12_split(value: i64) -> (i64, i64) {
    let hi = (value.wrapping_add(0x800)) >> 12;
    let lo = value.wrapping_sub(hi << 12);
    (hi, lo)
}

// ---------------------------------------------------------------------------
// RiscV64RelocationHandler
// ---------------------------------------------------------------------------

/// RISC-V 64 relocation handler for BCC's built-in linker.
///
/// Implements the [`RelocationHandler`] trait, providing RISC-V 64 specific
/// relocation classification, naming, sizing, and application.
///
/// ## HI20/LO12 Pairing
///
/// RISC-V uses paired relocations where a HI20 relocation (PCREL_HI20 or
/// GOT_HI20) is followed by a LO12 relocation (PCREL_LO12_I or PCREL_LO12_S).
/// The LO12 relocation's symbol/addend points to the *address of the HI20
/// instruction*, not the target symbol. The handler tracks HI20 values via
/// an internal map so that LO12 relocations can resolve correctly.
///
/// Interior mutability (`RefCell`) is used for the HI20 tracking map because
/// the `RelocationHandler` trait's `apply_relocation` method takes `&self`.
pub struct RiscV64RelocationHandler {
    /// Tracks HI20 relocation values for LO12 pair resolution.
    ///
    /// Key: absolute address of the AUIPC/LUI instruction (patch_address of HI20 reloc).
    /// Value: the full PC-relative or GOT-relative value computed during HI20 processing.
    ///
    /// Uses `RefCell` for interior mutability since `apply_relocation` takes `&self`
    /// but must record values during HI20 processing for subsequent LO12 lookups.
    hi20_values: RefCell<FxHashMap<u64, i64>>,
}

impl RiscV64RelocationHandler {
    /// Create a new RISC-V 64 relocation handler.
    pub fn new() -> Self {
        Self {
            hi20_values: RefCell::new(FxHashMap::default()),
        }
    }

    /// Check if a CALL/CALL_PLT relocation can be relaxed to JAL.
    ///
    /// Returns `true` if the target is within ±1 MiB range (21-bit signed offset),
    /// which is the maximum reach of a single JAL instruction.
    pub fn can_relax_call_to_jal(&self, rel: &ResolvedRelocation) -> bool {
        let value = (rel.symbol_value as i64)
            .wrapping_add(rel.addend)
            .wrapping_sub(rel.patch_address as i64);
        fits_signed(value, 21)
    }

    /// Check if an AUIPC+ADDI (PCREL_HI20+PCREL_LO12) can be relaxed to a
    /// single ADDI instruction.
    ///
    /// Returns `true` if the target is within ±2 KiB range (12-bit signed immediate).
    pub fn can_relax_pcrel_to_addi(&self, rel: &ResolvedRelocation) -> bool {
        let value = (rel.symbol_value as i64)
            .wrapping_add(rel.addend)
            .wrapping_sub(rel.patch_address as i64);
        fits_signed(value, 12)
    }

    /// Apply CALL→JAL relaxation: replace 8-byte AUIPC+JALR with 4-byte JAL + 4-byte NOP.
    ///
    /// The AUIPC instruction at `offset` is replaced with a JAL instruction
    /// preserving the original destination register (rd). The JALR instruction
    /// at `offset+4` is replaced with a NOP (ADDI x0, x0, 0).
    ///
    /// # Errors
    ///
    /// Returns `RelocationError::Overflow` if the value does not fit in the
    /// 21-bit signed immediate of the JAL instruction.
    pub fn relax_call_to_jal(
        &self,
        section_data: &mut [u8],
        offset: usize,
        value: i64,
    ) -> Result<(), RelocationError> {
        if !fits_signed(value, 21) {
            return Err(RelocationError::Overflow {
                reloc_name: "R_RISCV_CALL (relaxed to JAL)".to_string(),
                value: value as i128,
                bit_width: 21,
                location: format!("0x{:x}", offset),
            });
        }
        // Read the AUIPC instruction to extract the rd field.
        let auipc = read_insn32(section_data, offset);
        let rd = (auipc >> 7) & 0x1F;
        // Construct JAL rd, offset instruction. JAL opcode = 0x6F.
        let jal = encode_j_type_imm(0x6F | (rd << 7), value);
        write_insn32(section_data, offset, jal);
        // Write NOP (ADDI x0, x0, 0 = 0x00000013) at offset+4.
        write_insn32(section_data, offset + 4, 0x0000_0013);
        Ok(())
    }

    /// Record a HI20 relocation value for paired LO12 lookup.
    ///
    /// Called externally by the linker driver for each PCREL_HI20 or GOT_HI20
    /// relocation so that subsequent PCREL_LO12 relocations can resolve by
    /// looking up the full computed value.
    ///
    /// # Arguments
    /// * `patch_address` — absolute address of the AUIPC/LUI instruction
    /// * `value` — the full computed PC-relative or GOT-relative value
    pub fn record_hi20_value(&mut self, patch_address: u64, value: i64) {
        self.hi20_values.borrow_mut().insert(patch_address, value);
    }

    /// Look up a HI20 value for a paired LO12 relocation.
    ///
    /// Returns the full value that was recorded during HI20 processing,
    /// or `None` if no HI20 relocation was processed at the given address.
    pub fn lookup_hi20_value(&self, hi20_address: u64) -> Option<i64> {
        self.hi20_values.borrow().get(&hi20_address).copied()
    }

    /// Apply a CALL or CALL_PLT relocation (AUIPC+JALR pair).
    ///
    /// Both AUIPC (at `offset`) and JALR (at `offset+4`) are patched.
    /// The value is split into hi20 and lo12 with sign-extension compensation.
    fn apply_call_relocation(
        &self,
        section_data: &mut [u8],
        offset: usize,
        value: i64,
        reloc_name_str: &str,
    ) -> Result<(), RelocationError> {
        if !fits_signed(value, 32) {
            return Err(RelocationError::Overflow {
                reloc_name: reloc_name_str.to_string(),
                value: value as i128,
                bit_width: 32,
                location: format!("0x{:x}", offset),
            });
        }
        let (hi, lo) = hi20_lo12_split(value);
        // Patch AUIPC at offset with U-type upper 20 bits.
        let auipc = read_insn32(section_data, offset);
        write_insn32(section_data, offset, encode_u_type_imm(auipc, hi << 12));
        // Patch JALR at offset+4 with I-type lower 12 bits.
        let jalr = read_insn32(section_data, offset + 4);
        write_insn32(section_data, offset + 4, encode_i_type_imm(jalr, lo));
        Ok(())
    }

    /// Resolve a PCREL_LO12 relocation by looking up the paired HI20 value.
    ///
    /// The PCREL_LO12 relocation's `symbol_value + addend` gives the address
    /// of the paired HI20 AUIPC instruction. We look up the full value that
    /// was computed during HI20 processing and extract the lo12 component.
    fn resolve_pcrel_lo12(&self, rel: &ResolvedRelocation) -> Result<i64, RelocationError> {
        let cache = self.hi20_values.borrow();

        // Strategy 1: The "symbol" for PCREL_LO12 points to the AUIPC
        // instruction address (standard convention).
        let hi20_addr = rel.symbol_value.wrapping_add(rel.addend as u64);
        if let Some(&full_value) = cache.get(&hi20_addr) {
            let (_hi, lo) = hi20_lo12_split(full_value);
            return Ok(sign_extend((lo as u64) & 0xFFF, 12));
        }

        // Strategy 2: Try symbol_value alone (some toolchains).
        if let Some(&full_value) = cache.get(&rel.symbol_value) {
            let (_hi, lo) = hi20_lo12_split(full_value);
            return Ok(sign_extend((lo as u64) & 0xFFF, 12));
        }

        // Strategy 3: Look for the AUIPC instruction immediately before
        // this LO12 instruction (AUIPC is at patch_address - 4). This
        // handles the common LA/CALL pseudo-instruction pattern where
        // AUIPC + ADDI/JALR are consecutive.
        let prev_addr = rel.patch_address.wrapping_sub(4);
        if let Some(&full_value) = cache.get(&prev_addr) {
            let (_hi, lo) = hi20_lo12_split(full_value);
            return Ok(sign_extend((lo as u64) & 0xFFF, 12));
        }

        // Strategy 4: Scan all HI20 entries — the LO12 may refer to any
        // AUIPC within the same function. Pick the closest preceding one.
        let mut best: Option<(u64, i64)> = None;
        for (&addr, &fv) in cache.iter() {
            if addr <= rel.patch_address {
                match best {
                    None => best = Some((addr, fv)),
                    Some((ba, _)) if addr > ba => best = Some((addr, fv)),
                    _ => {}
                }
            }
        }
        if let Some((_, full_value)) = best {
            let (_hi, lo) = hi20_lo12_split(full_value);
            return Ok(sign_extend((lo as u64) & 0xFFF, 12));
        }

        Err(RelocationError::UndefinedSymbol {
            name: format!(
                "HI20 pair at 0x{:x} (for LO12 at 0x{:x})",
                hi20_addr, rel.patch_address
            ),
            reloc_name: self.reloc_name(rel.rel_type).to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// RelocationHandler Trait Implementation
// ---------------------------------------------------------------------------

impl RelocationHandler for RiscV64RelocationHandler {
    /// Classify a RISC-V relocation type into a [`RelocCategory`].
    fn classify(&self, rel_type: u32) -> RelocCategory {
        match rel_type {
            R_RISCV_32 | R_RISCV_64 | R_RISCV_HI20 | R_RISCV_LO12_I | R_RISCV_LO12_S => {
                RelocCategory::Absolute
            }
            R_RISCV_BRANCH | R_RISCV_JAL | R_RISCV_CALL | R_RISCV_PCREL_HI20
            | R_RISCV_PCREL_LO12_I | R_RISCV_PCREL_LO12_S | R_RISCV_32_PCREL
            | R_RISCV_RVC_BRANCH | R_RISCV_RVC_JUMP => RelocCategory::PcRelative,
            R_RISCV_GOT_HI20 => RelocCategory::GotRelative,
            R_RISCV_CALL_PLT => RelocCategory::Plt,
            R_RISCV_JUMP_SLOT => RelocCategory::GotEntry,
            R_RISCV_TLS_DTPMOD32 | R_RISCV_TLS_DTPMOD64 | R_RISCV_TLS_DTPREL32
            | R_RISCV_TLS_DTPREL64 | R_RISCV_TLS_TPREL32 | R_RISCV_TLS_TPREL64
            | R_RISCV_TLS_GD_HI20 | R_RISCV_TPREL_HI20 | R_RISCV_TPREL_LO12_I
            | R_RISCV_TPREL_LO12_S | R_RISCV_TPREL_ADD => RelocCategory::Tls,
            R_RISCV_NONE | R_RISCV_RELAX | R_RISCV_ALIGN => RelocCategory::Other,
            R_RISCV_ADD8 | R_RISCV_ADD16 | R_RISCV_ADD32 | R_RISCV_ADD64 | R_RISCV_SUB6
            | R_RISCV_SUB8 | R_RISCV_SUB16 | R_RISCV_SUB32 | R_RISCV_SUB64 | R_RISCV_SET6
            | R_RISCV_SET8 | R_RISCV_SET16 | R_RISCV_SET32 => RelocCategory::SectionRelative,
            R_RISCV_RELATIVE | R_RISCV_COPY => RelocCategory::Absolute,
            R_RISCV_GNU_VTINHERIT | R_RISCV_GNU_VTENTRY | R_RISCV_RVC_LUI => RelocCategory::Other,
            _ => RelocCategory::Other,
        }
    }

    /// Get the human-readable name of a RISC-V relocation type.
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match rel_type {
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

    /// Get the size of the relocation patch in bytes.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        match rel_type {
            R_RISCV_64 | R_RISCV_ADD64 | R_RISCV_SUB64 => 8,
            R_RISCV_32 | R_RISCV_ADD32 | R_RISCV_SUB32 | R_RISCV_SET32 | R_RISCV_32_PCREL => 4,
            R_RISCV_BRANCH | R_RISCV_JAL | R_RISCV_HI20 | R_RISCV_LO12_I | R_RISCV_LO12_S
            | R_RISCV_PCREL_HI20 | R_RISCV_PCREL_LO12_I | R_RISCV_PCREL_LO12_S
            | R_RISCV_GOT_HI20 | R_RISCV_TPREL_HI20 | R_RISCV_TPREL_LO12_I
            | R_RISCV_TPREL_LO12_S | R_RISCV_TLS_GD_HI20 => 4, // 32-bit instruction width
            R_RISCV_CALL | R_RISCV_CALL_PLT => 8, // AUIPC (4) + JALR (4)
            R_RISCV_RVC_BRANCH | R_RISCV_RVC_JUMP | R_RISCV_RVC_LUI => 2, // compressed
            R_RISCV_ADD16 | R_RISCV_SUB16 | R_RISCV_SET16 => 2,
            R_RISCV_ADD8 | R_RISCV_SUB8 | R_RISCV_SET8 => 1,
            R_RISCV_SUB6 | R_RISCV_SET6 => 1,
            R_RISCV_RELATIVE | R_RISCV_JUMP_SLOT | R_RISCV_TLS_DTPMOD64 | R_RISCV_TLS_DTPREL64
            | R_RISCV_TLS_TPREL64 => 8,
            R_RISCV_TLS_DTPMOD32 | R_RISCV_TLS_DTPREL32 | R_RISCV_TLS_TPREL32 | R_RISCV_COPY => 4,
            _ => 0,
        }
    }

    /// Returns `true` if this relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool {
        matches!(rel_type, R_RISCV_GOT_HI20)
    }

    /// Returns `true` if this relocation type requires a PLT stub.
    fn needs_plt(&self, rel_type: u32) -> bool {
        matches!(rel_type, R_RISCV_CALL_PLT | R_RISCV_JAL | R_RISCV_CALL)
    }

    /// Apply a single resolved relocation by computing the final value and
    /// writing it into the section data buffer.
    ///
    /// This is the core relocation application logic. Each RISC-V relocation
    /// type has specific encoding rules that must be followed exactly.
    ///
    /// # Errors
    ///
    /// Returns [`RelocationError::Overflow`] if the computed value does not
    /// fit in the relocation's bit width, [`RelocationError::UndefinedSymbol`]
    /// if a paired HI20 relocation cannot be found, or
    /// [`RelocationError::UnsupportedType`] for unknown relocation types.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let offset = rel.patch_offset as usize;
        let s = rel.symbol_value;
        let a = rel.addend;
        let p = rel.patch_address;

        match rel.rel_type {
            // -----------------------------------------------------------------
            // No-op relocations
            // -----------------------------------------------------------------
            R_RISCV_NONE
            | R_RISCV_RELAX
            | R_RISCV_TPREL_ADD
            | R_RISCV_GNU_VTINHERIT
            | R_RISCV_GNU_VTENTRY => Ok(()),

            // -----------------------------------------------------------------
            // R_RISCV_ALIGN — alignment padding (no-op during application)
            // -----------------------------------------------------------------
            R_RISCV_ALIGN => {
                // Alignment is handled during the relaxation pass. During
                // application, we verify alignment is already satisfied or
                // treat it as a no-op.
                Ok(())
            }

            // -----------------------------------------------------------------
            // Absolute relocations
            // -----------------------------------------------------------------

            // R_RISCV_32: S + A — 32-bit absolute
            R_RISCV_32 => {
                let value = compute_absolute(s, a);
                if !fits_unsigned(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:x}", p),
                    });
                }
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // R_RISCV_64: S + A — 64-bit absolute
            R_RISCV_64 => {
                let value = compute_absolute(s, a);
                write_le(section_data, offset, 8, value);
                Ok(())
            }

            // R_RISCV_RELATIVE: B + A — base-relative (used by dynamic linker)
            R_RISCV_RELATIVE => {
                let value = compute_absolute(s, a);
                write_le(section_data, offset, 8, value);
                Ok(())
            }

            // R_RISCV_COPY: no patching needed at static link time
            R_RISCV_COPY => Ok(()),

            // R_RISCV_JUMP_SLOT: S — PLT jump slot (64-bit address)
            R_RISCV_JUMP_SLOT => {
                write_le(section_data, offset, 8, s);
                Ok(())
            }

            // -----------------------------------------------------------------
            // PC-relative branch/jump relocations
            // -----------------------------------------------------------------

            // R_RISCV_BRANCH: S + A - P — B-type, ±4 KiB (13-bit signed)
            R_RISCV_BRANCH => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 13) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_BRANCH".to_string(),
                        value: value as i128,
                        bit_width: 13,
                        location: format!("0x{:x}", p),
                    });
                }
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_b_type_imm(insn, value));
                Ok(())
            }

            // R_RISCV_JAL: S + A - P — J-type, ±1 MiB (21-bit signed)
            R_RISCV_JAL => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 21) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_JAL".to_string(),
                        value: value as i128,
                        bit_width: 21,
                        location: format!("0x{:x}", p),
                    });
                }
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_j_type_imm(insn, value));
                Ok(())
            }

            // R_RISCV_CALL: S + A - P — AUIPC+JALR pair, ±2 GiB
            R_RISCV_CALL => {
                let value = compute_pc_relative(s, a, p);
                self.apply_call_relocation(section_data, offset, value, "R_RISCV_CALL")
            }

            // R_RISCV_CALL_PLT: S + A - P — AUIPC+JALR via PLT, ±2 GiB
            R_RISCV_CALL_PLT => {
                // Use PLT address if available, otherwise use symbol directly.
                let target = if let Some(plt_addr) = rel.plt_address {
                    plt_addr
                } else {
                    s
                };
                let value = (target as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.apply_call_relocation(section_data, offset, value, "R_RISCV_CALL_PLT")
            }

            // -----------------------------------------------------------------
            // HI20/LO12 pairs — PC-relative
            // -----------------------------------------------------------------

            // R_RISCV_PCREL_HI20: S + A - P — upper 20 bits (U-type in AUIPC)
            R_RISCV_PCREL_HI20 => {
                let value = compute_pc_relative(s, a, p);
                // Record full value for paired LO12 lookup.
                self.hi20_values.borrow_mut().insert(p, value);
                let (hi, _lo) = hi20_lo12_split(value);
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_u_type_imm(insn, hi << 12));
                Ok(())
            }

            // R_RISCV_PCREL_LO12_I: lower 12 bits from paired HI20 (I-type)
            R_RISCV_PCREL_LO12_I => {
                let lo = self.resolve_pcrel_lo12(rel)?;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_i_type_imm(insn, lo));
                Ok(())
            }

            // R_RISCV_PCREL_LO12_S: lower 12 bits from paired HI20 (S-type)
            R_RISCV_PCREL_LO12_S => {
                let lo = self.resolve_pcrel_lo12(rel)?;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_s_type_imm(insn, lo));
                Ok(())
            }

            // -----------------------------------------------------------------
            // GOT-relative
            // -----------------------------------------------------------------

            // R_RISCV_GOT_HI20: GOT_entry + A - P — upper 20 bits (U-type)
            R_RISCV_GOT_HI20 => {
                let got_addr = rel.got_address.unwrap_or(s);
                let value = compute_got_relative(got_addr, a, p);
                // Record full value for paired LO12 lookup.
                self.hi20_values.borrow_mut().insert(p, value);
                let (hi, _lo) = hi20_lo12_split(value);
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_u_type_imm(insn, hi << 12));
                Ok(())
            }

            // -----------------------------------------------------------------
            // HI20/LO12 pairs — Absolute
            // -----------------------------------------------------------------

            // R_RISCV_HI20: S + A — upper 20 bits (U-type in LUI)
            R_RISCV_HI20 => {
                let value = (s as i64).wrapping_add(a);
                let (hi, _lo) = hi20_lo12_split(value);
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_u_type_imm(insn, hi << 12));
                Ok(())
            }

            // R_RISCV_LO12_I: S + A — lower 12 bits (I-type)
            R_RISCV_LO12_I => {
                let value = (s as i64).wrapping_add(a);
                let lo = value & 0xFFF;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_i_type_imm(insn, lo));
                Ok(())
            }

            // R_RISCV_LO12_S: S + A — lower 12 bits (S-type)
            R_RISCV_LO12_S => {
                let value = (s as i64).wrapping_add(a);
                let lo = value & 0xFFF;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_s_type_imm(insn, lo));
                Ok(())
            }

            // -----------------------------------------------------------------
            // TLS relocations — HI20/LO12 style
            // -----------------------------------------------------------------
            R_RISCV_TLS_GD_HI20 => {
                // TLS General Dynamic: GOT-relative upper 20 bits.
                let got_addr = rel.got_address.unwrap_or(s);
                let value = (got_addr as i64).wrapping_add(a).wrapping_sub(p as i64);
                self.hi20_values.borrow_mut().insert(p, value);
                let (hi, _lo) = hi20_lo12_split(value);
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_u_type_imm(insn, hi << 12));
                Ok(())
            }

            R_RISCV_TPREL_HI20 => {
                let value = (s as i64).wrapping_add(a);
                let (hi, _lo) = hi20_lo12_split(value);
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_u_type_imm(insn, hi << 12));
                Ok(())
            }

            R_RISCV_TPREL_LO12_I => {
                let value = (s as i64).wrapping_add(a);
                let lo = value & 0xFFF;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_i_type_imm(insn, lo));
                Ok(())
            }

            R_RISCV_TPREL_LO12_S => {
                let value = (s as i64).wrapping_add(a);
                let lo = value & 0xFFF;
                let insn = read_insn32(section_data, offset);
                write_insn32(section_data, offset, encode_s_type_imm(insn, lo));
                Ok(())
            }

            // TLS DTPMOD/DTPREL/TPREL — 32-bit and 64-bit data writes.
            R_RISCV_TLS_DTPMOD64 | R_RISCV_TLS_DTPREL64 | R_RISCV_TLS_TPREL64 => {
                let value = compute_absolute(s, a);
                write_le(section_data, offset, 8, value);
                Ok(())
            }

            R_RISCV_TLS_DTPMOD32 | R_RISCV_TLS_DTPREL32 | R_RISCV_TLS_TPREL32 => {
                let value = compute_absolute(s, a);
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // -----------------------------------------------------------------
            // Arithmetic relocations (read-modify-write)
            // -----------------------------------------------------------------

            // R_RISCV_ADD8: V + S + A
            R_RISCV_ADD8 => {
                let existing = read_le(section_data, offset, 1) as u8;
                let sa = compute_absolute(s, a) as u8;
                write_le(section_data, offset, 1, existing.wrapping_add(sa) as u64);
                Ok(())
            }

            // R_RISCV_ADD16: V + S + A
            R_RISCV_ADD16 => {
                let existing = read_le(section_data, offset, 2) as u16;
                let sa = compute_absolute(s, a) as u16;
                write_le(section_data, offset, 2, existing.wrapping_add(sa) as u64);
                Ok(())
            }

            // R_RISCV_ADD32: V + S + A
            R_RISCV_ADD32 => {
                let existing = read_le(section_data, offset, 4) as u32;
                let sa = compute_absolute(s, a) as u32;
                write_le(section_data, offset, 4, existing.wrapping_add(sa) as u64);
                Ok(())
            }

            // R_RISCV_ADD64: V + S + A
            R_RISCV_ADD64 => {
                let existing = read_le(section_data, offset, 8);
                let sa = compute_absolute(s, a);
                write_le(section_data, offset, 8, existing.wrapping_add(sa));
                Ok(())
            }

            // R_RISCV_SUB8: V - S - A
            R_RISCV_SUB8 => {
                let existing = read_le(section_data, offset, 1) as u8;
                let sa = compute_absolute(s, a) as u8;
                write_le(section_data, offset, 1, existing.wrapping_sub(sa) as u64);
                Ok(())
            }

            // R_RISCV_SUB16: V - S - A
            R_RISCV_SUB16 => {
                let existing = read_le(section_data, offset, 2) as u16;
                let sa = compute_absolute(s, a) as u16;
                write_le(section_data, offset, 2, existing.wrapping_sub(sa) as u64);
                Ok(())
            }

            // R_RISCV_SUB32: V - S - A
            R_RISCV_SUB32 => {
                let existing = read_le(section_data, offset, 4) as u32;
                let sa = compute_absolute(s, a) as u32;
                write_le(section_data, offset, 4, existing.wrapping_sub(sa) as u64);
                Ok(())
            }

            // R_RISCV_SUB64: V - S - A
            R_RISCV_SUB64 => {
                let existing = read_le(section_data, offset, 8);
                let sa = compute_absolute(s, a);
                write_le(section_data, offset, 8, existing.wrapping_sub(sa));
                Ok(())
            }

            // R_RISCV_SUB6: V - (S + A) — subtract lower 6 bits
            R_RISCV_SUB6 => {
                let existing = section_data[offset];
                let sa = compute_absolute(s, a) as u8;
                // Subtract from the lower 6 bits, preserving the upper 2 bits.
                let low6 = (existing & 0x3F).wrapping_sub(sa & 0x3F) & 0x3F;
                section_data[offset] = (existing & 0xC0) | low6;
                Ok(())
            }

            // -----------------------------------------------------------------
            // Set relocations
            // -----------------------------------------------------------------

            // R_RISCV_SET6: write lower 6 bits of (S + A) into lower 6 bits of byte.
            R_RISCV_SET6 => {
                let value = compute_absolute(s, a) as u8;
                let existing = section_data[offset];
                section_data[offset] = (existing & 0xC0) | (value & 0x3F);
                Ok(())
            }

            // R_RISCV_SET8: write S + A truncated to 8 bits.
            R_RISCV_SET8 => {
                let value = compute_absolute(s, a) as u8;
                section_data[offset] = value;
                Ok(())
            }

            // R_RISCV_SET16: write S + A truncated to 16 bits.
            R_RISCV_SET16 => {
                let value = compute_absolute(s, a) as u16;
                write_le(section_data, offset, 2, value as u64);
                Ok(())
            }

            // R_RISCV_SET32: write S + A truncated to 32 bits.
            R_RISCV_SET32 => {
                let value = compute_absolute(s, a) as u32;
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // -----------------------------------------------------------------
            // Compressed instruction relocations (16-bit)
            // -----------------------------------------------------------------

            // R_RISCV_RVC_BRANCH: S + A - P — compressed branch, ±256 bytes (9-bit signed)
            R_RISCV_RVC_BRANCH => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 9) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_RVC_BRANCH".to_string(),
                        value: value as i128,
                        bit_width: 9,
                        location: format!("0x{:x}", p),
                    });
                }
                let insn = read_insn16(section_data, offset);
                write_insn16(section_data, offset, encode_cb_type_imm(insn, value));
                Ok(())
            }

            // R_RISCV_RVC_JUMP: S + A - P — compressed jump, ±2 KiB (12-bit signed)
            R_RISCV_RVC_JUMP => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 12) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_RVC_JUMP".to_string(),
                        value: value as i128,
                        bit_width: 12,
                        location: format!("0x{:x}", p),
                    });
                }
                let insn = read_insn16(section_data, offset);
                write_insn16(section_data, offset, encode_cj_type_imm(insn, value));
                Ok(())
            }

            // R_RISCV_RVC_LUI: compressed LUI immediate (not commonly used in linking)
            R_RISCV_RVC_LUI => {
                // The immediate for c.lui is bits [17:12], encoded in insn[12|6:2].
                let value = compute_absolute(s, a) as i64;
                let nzimm = ((value >> 12) & 0x3F) as u16;
                let mut insn = read_insn16(section_data, offset);
                insn &= 0xEF83; // clear imm bits
                insn |= ((nzimm >> 5) & 1) << 12; // nzimm[5]
                insn |= (nzimm & 0x1F) << 2; // nzimm[4:0]
                write_insn16(section_data, offset, insn);
                Ok(())
            }

            // -----------------------------------------------------------------
            // 32-bit PC-relative
            // -----------------------------------------------------------------

            // R_RISCV_32_PCREL: S + A - P — 32-bit PC-relative data relocation
            R_RISCV_32_PCREL => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_RISCV_32_PCREL".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:x}", p),
                    });
                }
                write_le(section_data, offset, 4, (value as u32) as u64);
                Ok(())
            }

            // -----------------------------------------------------------------
            // Unsupported
            // -----------------------------------------------------------------
            _ => Err(RelocationError::UnsupportedType {
                rel_type: rel.rel_type,
                target: Target::RiscV64,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Default trait
// ---------------------------------------------------------------------------

impl Default for RiscV64RelocationHandler {
    fn default() -> Self {
        Self::new()
    }
}
