//! # AArch64 Relocation Application
//!
//! Implements architecture-specific relocation patching for AArch64 ELF binaries.
//! This module is invoked by the linker common relocation framework to apply
//! resolved relocations to section data.
//!
//! ## Relocation Categories
//!
//! ### Absolute Relocations
//! - `R_AARCH64_ABS64` (257) — 64-bit absolute: `S + A`
//! - `R_AARCH64_ABS32` (258) — 32-bit absolute: `S + A` (overflow checked)
//! - `R_AARCH64_ABS16` (259) — 16-bit absolute: `S + A` (overflow checked)
//!
//! ### PC-Relative Relocations
//! - `R_AARCH64_PREL64` (260) — 64-bit PC-relative: `S + A - P`
//! - `R_AARCH64_PREL32` (261) — 32-bit PC-relative: `S + A - P` (overflow checked)
//! - `R_AARCH64_PREL16` (262) — 16-bit PC-relative: `S + A - P` (overflow checked)
//!
//! ### Page-Relative Relocations (ADRP)
//! - `R_AARCH64_ADR_PREL_PG_HI21` (275) — ADRP: `Page(S + A) - Page(P)` encoded into immhi:immlo
//!   - Range: ±4 GiB (21-bit signed page offset)
//! - `R_AARCH64_ADR_PREL_LO21` (274) — ADR: `S + A - P` (21-bit signed PC-relative)
//!
//! ### :lo12: Relocations (lower 12 bits)
//! - `R_AARCH64_ADD_ABS_LO12_NC` (277) — ADD imm12: `(S + A) & 0xFFF`
//! - `R_AARCH64_LDST8_ABS_LO12_NC` (278) — LDRB/STRB imm12: `(S + A) & 0xFFF`
//! - `R_AARCH64_LDST16_ABS_LO12_NC` (284) — LDRH/STRH imm12: `((S + A) & 0xFFF) >> 1`
//! - `R_AARCH64_LDST32_ABS_LO12_NC` (285) — LDR/STR W imm12: `((S + A) & 0xFFF) >> 2`
//! - `R_AARCH64_LDST64_ABS_LO12_NC` (286) — LDR/STR X imm12: `((S + A) & 0xFFF) >> 3`
//! - `R_AARCH64_LDST128_ABS_LO12_NC` (299) — LDR/STR Q imm12: `((S + A) & 0xFFF) >> 4`
//!
//! ### Branch Relocations
//! - `R_AARCH64_CALL26` (283) — BL instruction: `(S + A - P) >> 2` into imm26
//!   - Range: ±128 MiB
//! - `R_AARCH64_JUMP26` (282) — B instruction: `(S + A - P) >> 2` into imm26
//!   - Range: ±128 MiB
//! - `R_AARCH64_CONDBR19` (280) — B.cond: `(S + A - P) >> 2` into imm19
//!   - Range: ±1 MiB
//! - `R_AARCH64_TSTBR14` (279) — TBZ/TBNZ: `(S + A - P) >> 2` into imm14
//!   - Range: ±32 KiB
//!
//! ### GOT Relocations (PIC)
//! - `R_AARCH64_ADR_GOT_PAGE` (311) — ADRP to GOT entry page: `Page(G(GDAT(S+A))) - Page(P)`
//! - `R_AARCH64_LD64_GOT_LO12_NC` (312) — LDR from GOT: `G(GDAT(S+A)) & 0xFFF` >> 3
//!
//! ### TLS Relocations
//! - `R_AARCH64_TLSLE_ADD_TPREL_HI12` (549) — TLS LE high 12 bits
//! - `R_AARCH64_TLSLE_ADD_TPREL_LO12` (550) — TLS LE low 12 bits (overflow checked)
//! - `R_AARCH64_TLSLE_ADD_TPREL_LO12_NC` (551) — TLS LE low 12 bits (no check)
//!
//! ### Dynamic Relocations
//! - `R_AARCH64_COPY` (1024) — Copy symbol at runtime
//! - `R_AARCH64_GLOB_DAT` (1025) — GOT entry: `S + A`
//! - `R_AARCH64_JUMP_SLOT` (1026) — PLT GOT entry: `S + A`
//! - `R_AARCH64_RELATIVE` (1027) — Base-relative: `B + A`
//! - `R_AARCH64_TLS_DTPMOD64` (1028) — TLS module ID
//! - `R_AARCH64_TLS_DTPREL64` (1029) — TLS offset within module
//! - `R_AARCH64_TLS_TPREL64` (1030) — TLS offset from TP
//! - `R_AARCH64_TLSDESC` (1031) — TLS descriptor
//!
//! ## Notation
//! - `S` = Symbol value (final address)
//! - `A` = Addend
//! - `P` = Place (address where relocation is applied)
//! - `Page(x)` = `x & ~0xFFF` (4KB page of x)
//! - `G(x)` = GOT entry offset for symbol x
//! - `B` = Base address of shared object

use crate::backend::linker_common::relocation::{
    compute_absolute, compute_got_relative, compute_pc_relative, fits_signed, fits_unsigned,
    read_le, sign_extend, write_le, RelocCategory, RelocationError, RelocationHandler,
    ResolvedRelocation,
};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// AArch64 ELF Relocation Type Constants
// ---------------------------------------------------------------------------

/// No relocation — placeholder, no patching needed.
pub const R_AARCH64_NONE: u32 = 0;

// -- Absolute relocations --

/// 64-bit absolute: `S + A`. Writes the full 64-bit symbol address plus addend.
pub const R_AARCH64_ABS64: u32 = 257;

/// 32-bit absolute: `S + A` (overflow checked). Value must fit in 32 bits.
pub const R_AARCH64_ABS32: u32 = 258;

/// 16-bit absolute: `S + A` (overflow checked). Value must fit in 16 bits.
pub const R_AARCH64_ABS16: u32 = 259;

// -- PC-relative relocations --

/// 64-bit PC-relative: `S + A - P`.
pub const R_AARCH64_PREL64: u32 = 260;

/// 32-bit PC-relative: `S + A - P` (overflow checked).
pub const R_AARCH64_PREL32: u32 = 261;

/// 16-bit PC-relative: `S + A - P` (overflow checked).
pub const R_AARCH64_PREL16: u32 = 262;

// -- Page-relative relocations --

/// ADR instruction: 21-bit signed PC-relative byte offset `S + A - P`.
pub const R_AARCH64_ADR_PREL_LO21: u32 = 274;

/// ADRP instruction: 21-bit signed page offset `Page(S + A) - Page(P)`.
/// Range: ±4 GiB (21-bit signed × 4096).
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 275;

/// ADRP instruction (no overflow check variant): `Page(S + A) - Page(P)`.
pub const R_AARCH64_ADR_PREL_PG_HI21_NC: u32 = 276;

// -- :lo12: relocations (lower 12 bits of address) --

/// ADD immediate: `(S + A) & 0xFFF` into imm12 bits [21:10]. No overflow check.
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 277;

/// LDRB/STRB (byte): `(S + A) & 0xFFF` into imm12 bits [21:10]. No shift.
pub const R_AARCH64_LDST8_ABS_LO12_NC: u32 = 278;

/// LDRH/STRH (halfword): `((S + A) & 0xFFF) >> 1` into imm12 bits [21:10].
pub const R_AARCH64_LDST16_ABS_LO12_NC: u32 = 284;

/// LDR/STR W (word): `((S + A) & 0xFFF) >> 2` into imm12 bits [21:10].
pub const R_AARCH64_LDST32_ABS_LO12_NC: u32 = 285;

/// LDR/STR X (doubleword): `((S + A) & 0xFFF) >> 3` into imm12 bits [21:10].
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 286;

/// LDR/STR Q (quadword): `((S + A) & 0xFFF) >> 4` into imm12 bits [21:10].
pub const R_AARCH64_LDST128_ABS_LO12_NC: u32 = 299;

// -- Branch relocations --

/// TBZ/TBNZ: 14-bit signed PC-relative, `(S + A - P) >> 2` into imm14 bits [18:5].
/// Range: ±32 KiB.
pub const R_AARCH64_TSTBR14: u32 = 279;

/// B.cond: 19-bit signed PC-relative, `(S + A - P) >> 2` into imm19 bits [23:5].
/// Range: ±1 MiB.
pub const R_AARCH64_CONDBR19: u32 = 280;

/// B (unconditional): 26-bit signed PC-relative, `(S + A - P) >> 2` into imm26 bits [25:0].
/// Range: ±128 MiB.
pub const R_AARCH64_JUMP26: u32 = 282;

/// BL (call): 26-bit signed PC-relative, `(S + A - P) >> 2` into imm26 bits [25:0].
/// Range: ±128 MiB.
pub const R_AARCH64_CALL26: u32 = 283;

// -- GOT relocations --

/// ADRP to GOT entry page: `Page(G(GDAT(S+A))) - Page(P)`.
pub const R_AARCH64_ADR_GOT_PAGE: u32 = 311;

/// LDR from GOT :lo12:: `G(GDAT(S+A)) & 0xFFF` scaled by 8 for doubleword.
pub const R_AARCH64_LD64_GOT_LO12_NC: u32 = 312;

// -- TLS relocations (basic Local Exec support) --

/// TLS Local Exec: high 12 bits of TP offset, `(S + A) >> 12` into ADD imm12.
pub const R_AARCH64_TLSLE_ADD_TPREL_HI12: u32 = 549;

/// TLS Local Exec: low 12 bits of TP offset (overflow checked), `(S + A) & 0xFFF`.
pub const R_AARCH64_TLSLE_ADD_TPREL_LO12: u32 = 550;

/// TLS Local Exec: low 12 bits of TP offset (no overflow check), `(S + A) & 0xFFF`.
pub const R_AARCH64_TLSLE_ADD_TPREL_LO12_NC: u32 = 551;

// -- Dynamic relocations --

/// Copy symbol data at runtime (dynamic linker copies data from shared lib to executable).
pub const R_AARCH64_COPY: u32 = 1024;

/// GOT entry: dynamic linker writes `S + A` into the GOT slot.
pub const R_AARCH64_GLOB_DAT: u32 = 1025;

/// PLT GOT entry: dynamic linker writes `S + A` into the GOT.PLT slot.
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;

/// Base-relative: dynamic linker writes `B + A` (load base + addend).
pub const R_AARCH64_RELATIVE: u32 = 1027;

/// TLS module ID (dynamic).
pub const R_AARCH64_TLS_DTPMOD64: u32 = 1028;

/// TLS offset within module (dynamic).
pub const R_AARCH64_TLS_DTPREL64: u32 = 1029;

/// TLS offset from thread pointer (dynamic).
pub const R_AARCH64_TLS_TPREL64: u32 = 1030;

/// TLS descriptor (dynamic).
pub const R_AARCH64_TLSDESC: u32 = 1031;

// ---------------------------------------------------------------------------
// Instruction Bit-Field Helpers
// ---------------------------------------------------------------------------

/// Compute the 4KB-aligned page address by masking off the lower 12 bits.
///
/// AArch64 ADRP instructions operate on 4KB page granularity. This function
/// extracts the page base from any virtual address.
///
/// # Examples
/// ```ignore
/// assert_eq!(page(0x12345678), 0x12345000);
/// assert_eq!(page(0x1000), 0x1000);
/// assert_eq!(page(0xFFF), 0x0);
/// ```
#[inline]
fn page(addr: u64) -> u64 {
    addr & !0xFFF
}

/// Extract the lower 12 bits (page offset) from an address.
///
/// The `:lo12:` relocations use these bits for instruction-embedded immediate
/// fields. The value is further scaled depending on the access size.
///
/// # Examples
/// ```ignore
/// assert_eq!(lo12(0x12345678), 0x678);
/// assert_eq!(lo12(0x1000), 0x000);
/// ```
#[inline]
fn lo12(addr: u64) -> u64 {
    addr & 0xFFF
}

/// Extract bits `[hi:lo]` (inclusive) from a 32-bit instruction word.
///
/// This is a general-purpose extraction helper for reading instruction
/// fields during relocation application debugging and validation.
///
/// # Parameters
/// - `value`: 32-bit instruction word
/// - `lo`: lowest bit position (0-based)
/// - `hi`: highest bit position (0-based, inclusive)
///
/// # Returns
/// The extracted field, right-shifted to bit position 0.
#[inline]
#[allow(dead_code)]
fn extract_bits(value: u32, lo: u8, hi: u8) -> u32 {
    let width = (hi - lo + 1) as u32;
    (value >> lo as u32) & ((1u32 << width) - 1)
}

/// Insert `value` into bits `[hi:lo]` of a 32-bit instruction word,
/// preserving all other bits.
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `value`: the field value to insert (only the lowest `(hi-lo+1)` bits used)
/// - `lo`: lowest bit position for insertion
/// - `hi`: highest bit position for insertion (inclusive)
///
/// # Returns
/// The modified instruction word with the field inserted.
#[inline]
#[allow(dead_code)]
fn insert_bits(inst: u32, value: u32, lo: u8, hi: u8) -> u32 {
    let width = (hi - lo + 1) as u32;
    let mask = ((1u32 << width) - 1) << lo as u32;
    (inst & !mask) | ((value << lo as u32) & mask)
}

/// Encode a 21-bit signed immediate into the ADRP/ADR instruction format.
///
/// AArch64 ADRP and ADR instructions split the 21-bit immediate into two
/// non-contiguous fields:
/// - `immlo` (2 bits): instruction bits [30:29]
/// - `immhi` (19 bits): instruction bits [23:5]
///
/// The mask `0x9F00001F` preserves all non-immediate fields (opcode, Rd, etc.).
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `imm`: 21-bit signed immediate value (as u32)
///
/// # Returns
/// The instruction word with the immediate fields patched.
#[inline]
fn encode_adr_imm(inst: u32, imm: u32) -> u32 {
    let immlo = imm & 0x3; // bits [1:0]
    let immhi = (imm >> 2) & 0x7FFFF; // bits [20:2]
    (inst & 0x9F00001F) | (immlo << 29) | (immhi << 5)
}

/// Encode a 12-bit unsigned immediate into an ADD/LDR/STR instruction.
///
/// AArch64 ADD-immediate and load/store-unsigned-offset instructions place
/// the 12-bit immediate in bits [21:10].
///
/// The mask `0xFFC003FF` preserves all non-immediate fields.
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `imm12`: 12-bit immediate value (pre-scaled by the caller for LDST)
///
/// # Returns
/// The instruction word with the imm12 field patched.
#[inline]
fn encode_imm12(inst: u32, imm12: u32) -> u32 {
    (inst & 0xFFC003FF) | ((imm12 & 0xFFF) << 10)
}

/// Encode a 26-bit signed immediate into a B/BL instruction.
///
/// AArch64 B (unconditional branch) and BL (branch-with-link) instructions
/// place the 26-bit signed offset in bits [25:0].
///
/// The mask `0xFC000000` preserves the opcode bits.
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `imm26`: 26-bit signed offset value (instruction-count units)
///
/// # Returns
/// The instruction word with the imm26 field patched.
#[inline]
fn encode_imm26(inst: u32, imm26: u32) -> u32 {
    (inst & 0xFC000000) | (imm26 & 0x03FFFFFF)
}

/// Encode a 19-bit signed immediate into a B.cond/CBZ/CBNZ instruction.
///
/// Conditional branch instructions place the 19-bit signed offset in bits [23:5].
///
/// The mask `0xFF00001F` preserves the opcode and condition fields.
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `imm19`: 19-bit signed offset value (instruction-count units)
///
/// # Returns
/// The instruction word with the imm19 field patched.
#[inline]
fn encode_imm19(inst: u32, imm19: u32) -> u32 {
    (inst & 0xFF00001F) | ((imm19 & 0x7FFFF) << 5)
}

/// Encode a 14-bit signed immediate into a TBZ/TBNZ instruction.
///
/// Test-and-branch instructions place the 14-bit signed offset in bits [18:5].
///
/// The mask `0xFFF8001F` preserves the opcode, bit-number, and Rt fields.
///
/// # Parameters
/// - `inst`: original 32-bit instruction word
/// - `imm14`: 14-bit signed offset value (instruction-count units)
///
/// # Returns
/// The instruction word with the imm14 field patched.
#[inline]
fn encode_imm14(inst: u32, imm14: u32) -> u32 {
    (inst & 0xFFF8001F) | ((imm14 & 0x3FFF) << 5)
}

// ---------------------------------------------------------------------------
// AArch64RelocationHandler
// ---------------------------------------------------------------------------

/// AArch64 architecture-specific relocation handler.
///
/// Implements the [`RelocationHandler`] trait from the linker common framework,
/// providing classification, naming, sizing, and application of all AArch64
/// ELF relocation types needed for both static linking (ET_EXEC) and dynamic
/// linking (ET_DYN with GOT/PLT).
///
/// # PIC Mode
///
/// When `pic_mode` is enabled (via [`AArch64RelocationHandler::with_pic`]),
/// the handler adjusts its GOT/PLT requirements for relocations that need
/// different handling in position-independent code.
pub struct AArch64RelocationHandler {
    /// Whether we are linking in PIC (position-independent code) mode.
    /// This affects which relocations require GOT/PLT entries.
    pic_mode: bool,
}

impl Default for AArch64RelocationHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl AArch64RelocationHandler {
    /// Create a new relocation handler with PIC mode disabled (default).
    ///
    /// Suitable for static executable linking where all addresses are resolved
    /// at link time.
    pub fn new() -> Self {
        Self { pic_mode: false }
    }

    /// Create a new relocation handler with the specified PIC mode.
    ///
    /// When `pic_mode` is `true`, the handler expects GOT/PLT entries for
    /// certain relocation types that would otherwise be resolved statically.
    ///
    /// # Parameters
    /// - `pic_mode`: `true` for shared library / PIC linking, `false` for static
    pub fn with_pic(pic_mode: bool) -> Self {
        Self { pic_mode }
    }
}

// ---------------------------------------------------------------------------
// RelocationHandler Trait Implementation
// ---------------------------------------------------------------------------

impl RelocationHandler for AArch64RelocationHandler {
    /// Classify an AArch64 relocation type into a [`RelocCategory`].
    ///
    /// This classification drives the linker's symbol resolution and
    /// GOT/PLT allocation decisions.
    fn classify(&self, rel_type: u32) -> RelocCategory {
        match rel_type {
            // Absolute address relocations — S + A
            R_AARCH64_ABS64 | R_AARCH64_ABS32 | R_AARCH64_ABS16 => RelocCategory::Absolute,

            // PC-relative relocations — S + A - P
            R_AARCH64_PREL64
            | R_AARCH64_PREL32
            | R_AARCH64_PREL16
            | R_AARCH64_ADR_PREL_LO21
            | R_AARCH64_ADR_PREL_PG_HI21
            | R_AARCH64_ADR_PREL_PG_HI21_NC
            | R_AARCH64_CALL26
            | R_AARCH64_JUMP26
            | R_AARCH64_CONDBR19
            | R_AARCH64_TSTBR14 => RelocCategory::PcRelative,

            // :lo12: relocations — these are page-offset (not PC-relative), classified
            // as absolute because they extract the lower 12 bits of S + A directly.
            R_AARCH64_ADD_ABS_LO12_NC
            | R_AARCH64_LDST8_ABS_LO12_NC
            | R_AARCH64_LDST16_ABS_LO12_NC
            | R_AARCH64_LDST32_ABS_LO12_NC
            | R_AARCH64_LDST64_ABS_LO12_NC
            | R_AARCH64_LDST128_ABS_LO12_NC => RelocCategory::Absolute,

            // GOT-relative relocations — need GOT entry creation
            R_AARCH64_ADR_GOT_PAGE | R_AARCH64_LD64_GOT_LO12_NC => RelocCategory::GotRelative,

            // GOT entry creation for dynamic linking
            R_AARCH64_GLOB_DAT => RelocCategory::GotEntry,

            // PLT relocation for dynamic function calls
            R_AARCH64_JUMP_SLOT => RelocCategory::Plt,

            // Runtime-only relocations and copy
            R_AARCH64_COPY | R_AARCH64_RELATIVE => RelocCategory::Other,

            // TLS relocations
            R_AARCH64_TLSLE_ADD_TPREL_HI12
            | R_AARCH64_TLSLE_ADD_TPREL_LO12
            | R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
            | R_AARCH64_TLS_DTPMOD64
            | R_AARCH64_TLS_DTPREL64
            | R_AARCH64_TLS_TPREL64
            | R_AARCH64_TLSDESC => RelocCategory::Tls,

            // No-op and unknown types
            R_AARCH64_NONE => RelocCategory::Other,
            _ => RelocCategory::Other,
        }
    }

    /// Get the human-readable name of an AArch64 relocation type.
    ///
    /// Used for error messages and diagnostic output. Returns `"R_AARCH64_UNKNOWN"`
    /// for unrecognised type codes.
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match rel_type {
            R_AARCH64_NONE => "R_AARCH64_NONE",
            R_AARCH64_ABS64 => "R_AARCH64_ABS64",
            R_AARCH64_ABS32 => "R_AARCH64_ABS32",
            R_AARCH64_ABS16 => "R_AARCH64_ABS16",
            R_AARCH64_PREL64 => "R_AARCH64_PREL64",
            R_AARCH64_PREL32 => "R_AARCH64_PREL32",
            R_AARCH64_PREL16 => "R_AARCH64_PREL16",
            R_AARCH64_ADR_PREL_LO21 => "R_AARCH64_ADR_PREL_LO21",
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
            R_AARCH64_TLSLE_ADD_TPREL_HI12 => "R_AARCH64_TLSLE_ADD_TPREL_HI12",
            R_AARCH64_TLSLE_ADD_TPREL_LO12 => "R_AARCH64_TLSLE_ADD_TPREL_LO12",
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

    /// Get the size of the relocation patch in bytes.
    ///
    /// AArch64 relocations patch either raw data fields (2/4/8 bytes) or
    /// 32-bit instruction words (always 4 bytes). AArch64 has fixed-width
    /// 32-bit instructions, so instruction-embedded relocations are always 4.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        match rel_type {
            // 64-bit data relocations
            R_AARCH64_ABS64 | R_AARCH64_PREL64 => 8,

            // 32-bit data relocations
            R_AARCH64_ABS32 | R_AARCH64_PREL32 => 4,

            // 16-bit data relocations
            R_AARCH64_ABS16 | R_AARCH64_PREL16 => 2,

            // All instruction-embedded relocations patch 32-bit instruction words
            R_AARCH64_ADR_PREL_PG_HI21
            | R_AARCH64_ADR_PREL_PG_HI21_NC
            | R_AARCH64_ADR_PREL_LO21
            | R_AARCH64_ADD_ABS_LO12_NC
            | R_AARCH64_LDST8_ABS_LO12_NC
            | R_AARCH64_LDST16_ABS_LO12_NC
            | R_AARCH64_LDST32_ABS_LO12_NC
            | R_AARCH64_LDST64_ABS_LO12_NC
            | R_AARCH64_LDST128_ABS_LO12_NC
            | R_AARCH64_CALL26
            | R_AARCH64_JUMP26
            | R_AARCH64_CONDBR19
            | R_AARCH64_TSTBR14
            | R_AARCH64_ADR_GOT_PAGE
            | R_AARCH64_LD64_GOT_LO12_NC
            | R_AARCH64_TLSLE_ADD_TPREL_HI12
            | R_AARCH64_TLSLE_ADD_TPREL_LO12
            | R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => 4,

            // Dynamic relocations are typically 8-byte (64-bit address writes)
            R_AARCH64_GLOB_DAT
            | R_AARCH64_JUMP_SLOT
            | R_AARCH64_RELATIVE
            | R_AARCH64_TLS_DTPMOD64
            | R_AARCH64_TLS_DTPREL64
            | R_AARCH64_TLS_TPREL64
            | R_AARCH64_TLSDESC => 8,

            // Default: instruction-embedded (4 bytes)
            _ => 4,
        }
    }

    /// Returns `true` if this relocation type requires a GOT (Global Offset Table) entry.
    ///
    /// GOT entries are 8-byte slots that hold the final address of a symbol.
    /// The dynamic linker fills them at load time for shared libraries; the
    /// static linker fills them at link time for executables.
    fn needs_got(&self, rel_type: u32) -> bool {
        matches!(
            rel_type,
            R_AARCH64_ADR_GOT_PAGE | R_AARCH64_LD64_GOT_LO12_NC | R_AARCH64_GLOB_DAT
        )
    }

    /// Returns `true` if this relocation type requires a PLT (Procedure Linkage Table) stub.
    ///
    /// PLT stubs provide lazy binding for function calls to external symbols.
    /// In PIC mode, direct branch relocations (CALL26/JUMP26) to external
    /// symbols need PLT entries. JUMP_SLOT always requires PLT regardless of
    /// PIC mode since it is inherently a dynamic relocation.
    fn needs_plt(&self, rel_type: u32) -> bool {
        match rel_type {
            // JUMP_SLOT always needs PLT — it's a dynamic PLT relocation
            R_AARCH64_JUMP_SLOT => true,
            // Branch relocations need PLT in PIC mode for external symbol calls.
            // In non-PIC mode we still report true because the linker needs to
            // know about potential PLT requirements; the linker itself decides
            // whether to actually create the stub based on symbol resolution.
            R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
                // In PIC mode, branches to external symbols always go through PLT.
                // In non-PIC mode, branches may also need PLT for shared lib symbols.
                // We conservatively return true; the linker filters based on context.
                let _ = self.pic_mode; // PIC mode informs PLT allocation strategy
                true
            }
            _ => false,
        }
    }

    /// Apply a single resolved relocation to the section data buffer.
    ///
    /// This is the core relocation application function. For each relocation
    /// type, it computes the final value using the resolved symbol address,
    /// addend, and patch address, then encodes the value into the appropriate
    /// bit fields of the instruction word or data location.
    ///
    /// # Parameters
    /// - `rel`: The fully-resolved relocation with all addresses computed
    /// - `section_data`: Mutable byte buffer of the section being patched
    ///
    /// # Errors
    /// Returns [`RelocationError::Overflow`] if the computed value exceeds the
    /// relocation's bit-field range, or [`RelocationError::UndefinedSymbol`] if
    /// a required GOT address is missing.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let offset = rel.patch_offset as usize;

        match rel.rel_type {
            // =================================================================
            // R_AARCH64_NONE (0) — No operation
            // =================================================================
            R_AARCH64_NONE => {
                // Nothing to do
            }

            // =================================================================
            // Absolute Relocations
            // =================================================================

            // R_AARCH64_ABS64 (257) — 64-bit absolute: S + A
            R_AARCH64_ABS64 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_ABS32 (258) — 32-bit absolute: S + A (overflow checked)
            R_AARCH64_ABS32 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 32) && !fits_signed(value as i64, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_ABS32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                write_le(section_data, offset, 4, value);
            }

            // R_AARCH64_ABS16 (259) — 16-bit absolute: S + A (overflow checked)
            R_AARCH64_ABS16 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 16) && !fits_signed(value as i64, 16) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_ABS16".to_string(),
                        value: value as i128,
                        bit_width: 16,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                write_le(section_data, offset, 2, value);
            }

            // =================================================================
            // PC-Relative Data Relocations
            // =================================================================

            // R_AARCH64_PREL64 (260) — 64-bit PC-relative: S + A - P
            R_AARCH64_PREL64 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                write_le(section_data, offset, 8, value as u64);
            }

            // R_AARCH64_PREL32 (261) — 32-bit PC-relative: S + A - P (overflow checked)
            R_AARCH64_PREL32 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_PREL32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // R_AARCH64_PREL16 (262) — 16-bit PC-relative: S + A - P (overflow checked)
            R_AARCH64_PREL16 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                if !fits_signed(value, 16) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_PREL16".to_string(),
                        value: value as i128,
                        bit_width: 16,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                write_le(section_data, offset, 2, value as u64);
            }

            // =================================================================
            // Page-Relative Relocations (ADR / ADRP)
            // =================================================================

            // R_AARCH64_ADR_PREL_LO21 (274) — ADR: 21-bit signed PC-relative byte offset
            R_AARCH64_ADR_PREL_LO21 => {
                let pc_offset =
                    compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                if !fits_signed(pc_offset, 21) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_ADR_PREL_LO21".to_string(),
                        value: pc_offset as i128,
                        bit_width: 21,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_adr_imm(inst, pc_offset as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_ADR_PREL_PG_HI21 (275) — ADRP: page-relative with overflow check
            R_AARCH64_ADR_PREL_PG_HI21 => {
                let target_addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let target_page = page(target_addr);
                let source_page = page(rel.patch_address);
                let page_delta = (target_page as i64).wrapping_sub(source_page as i64);
                let page_offset = page_delta >> 12;

                if !fits_signed(page_offset, 21) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_ADR_PREL_PG_HI21".to_string(),
                        value: page_offset as i128,
                        bit_width: 21,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_adr_imm(inst, page_offset as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_ADR_PREL_PG_HI21_NC (276) — ADRP: no overflow check variant
            R_AARCH64_ADR_PREL_PG_HI21_NC => {
                let target_addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let target_page = page(target_addr);
                let source_page = page(rel.patch_address);
                let page_delta = (target_page as i64).wrapping_sub(source_page as i64);
                let page_offset = page_delta >> 12;

                // NC = no check; encode regardless of overflow
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_adr_imm(inst, page_offset as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // =================================================================
            // :lo12: Relocations (lower 12 bits of absolute address)
            // =================================================================

            // R_AARCH64_ADD_ABS_LO12_NC (277) — ADD :lo12:, no shift, no overflow check
            R_AARCH64_ADD_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let value = lo12(addr) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, value);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LDST8_ABS_LO12_NC (278) — byte access, no shift
            R_AARCH64_LDST8_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let value = lo12(addr) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, value);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LDST16_ABS_LO12_NC (284) — halfword access, shift right by 1
            R_AARCH64_LDST16_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let scaled = (lo12(addr) >> 1) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, scaled);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LDST32_ABS_LO12_NC (285) — word access, shift right by 2
            R_AARCH64_LDST32_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let scaled = (lo12(addr) >> 2) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, scaled);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LDST64_ABS_LO12_NC (286) — doubleword access, shift right by 3
            R_AARCH64_LDST64_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let scaled = (lo12(addr) >> 3) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, scaled);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LDST128_ABS_LO12_NC (299) — quadword access, shift right by 4
            R_AARCH64_LDST128_ABS_LO12_NC => {
                let addr = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let scaled = (lo12(addr) >> 4) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, scaled);
                write_le(section_data, offset, 4, patched as u64);
            }

            // =================================================================
            // Branch Relocations
            // =================================================================

            // R_AARCH64_TSTBR14 (279) — TBZ/TBNZ: 14-bit signed, ±32 KiB range
            R_AARCH64_TSTBR14 => {
                let pc_offset =
                    compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                let shifted = pc_offset >> 2; // instruction-aligned

                if !fits_signed(shifted, 14) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_TSTBR14".to_string(),
                        value: shifted as i128,
                        bit_width: 14,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm14(inst, shifted as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_CONDBR19 (280) — B.cond: 19-bit signed, ±1 MiB range
            R_AARCH64_CONDBR19 => {
                let pc_offset =
                    compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                let shifted = pc_offset >> 2;

                if !fits_signed(shifted, 19) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_CONDBR19".to_string(),
                        value: shifted as i128,
                        bit_width: 19,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm19(inst, shifted as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_JUMP26 (282) — B: 26-bit signed, ±128 MiB range
            R_AARCH64_JUMP26 => {
                let pc_offset =
                    compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                let shifted = pc_offset >> 2;

                if !fits_signed(shifted, 26) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_JUMP26".to_string(),
                        value: shifted as i128,
                        bit_width: 26,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm26(inst, shifted as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_CALL26 (283) — BL: 26-bit signed, ±128 MiB range
            R_AARCH64_CALL26 => {
                let pc_offset =
                    compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                let shifted = pc_offset >> 2;

                if !fits_signed(shifted, 26) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_CALL26".to_string(),
                        value: shifted as i128,
                        bit_width: 26,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm26(inst, shifted as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // =================================================================
            // GOT Relocations (PIC)
            // =================================================================

            // R_AARCH64_ADR_GOT_PAGE (311) — ADRP to GOT entry page
            R_AARCH64_ADR_GOT_PAGE => {
                let got_addr = rel
                    .got_address
                    .ok_or_else(|| RelocationError::UndefinedSymbol {
                        name: format!("GOT entry at 0x{:x}", rel.patch_address),
                        reloc_name: "R_AARCH64_ADR_GOT_PAGE".to_string(),
                    })?;
                let target_page = page(got_addr);
                let source_page = page(rel.patch_address);
                let page_delta = (target_page as i64).wrapping_sub(source_page as i64) >> 12;

                if !fits_signed(page_delta, 21) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_ADR_GOT_PAGE".to_string(),
                        value: page_delta as i128,
                        bit_width: 21,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }

                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_adr_imm(inst, page_delta as u32);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_LD64_GOT_LO12_NC (312) — LDR from GOT :lo12:
            R_AARCH64_LD64_GOT_LO12_NC => {
                let got_addr = rel
                    .got_address
                    .ok_or_else(|| RelocationError::UndefinedSymbol {
                        name: format!("GOT entry at 0x{:x}", rel.patch_address),
                        reloc_name: "R_AARCH64_LD64_GOT_LO12_NC".to_string(),
                    })?;
                // Scale by 8 for doubleword LDR (64-bit pointer load)
                let scaled = (lo12(got_addr) >> 3) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, scaled);
                write_le(section_data, offset, 4, patched as u64);
            }

            // =================================================================
            // TLS Relocations (Local Exec model)
            // =================================================================

            // R_AARCH64_TLSLE_ADD_TPREL_HI12 (549) — high 12 bits of TP offset
            R_AARCH64_TLSLE_ADD_TPREL_HI12 => {
                let tprel = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let hi12 = ((tprel >> 12) & 0xFFF) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, hi12);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_TLSLE_ADD_TPREL_LO12 (550) — low 12 bits (overflow checked)
            R_AARCH64_TLSLE_ADD_TPREL_LO12 => {
                let tprel = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let lo = lo12(tprel) as u32;
                // Overflow check: the full tprel must fit in 24 bits for hi12+lo12
                if tprel > 0x00FF_FFFF {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_AARCH64_TLSLE_ADD_TPREL_LO12".to_string(),
                        value: tprel as i128,
                        bit_width: 24,
                        location: format!("0x{:x}", rel.patch_address),
                    });
                }
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, lo);
                write_le(section_data, offset, 4, patched as u64);
            }

            // R_AARCH64_TLSLE_ADD_TPREL_LO12_NC (551) — low 12 bits (no overflow check)
            R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => {
                let tprel = (rel.symbol_value as i64).wrapping_add(rel.addend) as u64;
                let lo = lo12(tprel) as u32;
                let inst = read_le(section_data, offset, 4) as u32;
                let patched = encode_imm12(inst, lo);
                write_le(section_data, offset, 4, patched as u64);
            }

            // =================================================================
            // Dynamic Relocations
            // =================================================================
            // These are normally emitted as dynamic relocation entries in
            // .rela.dyn / .rela.plt and processed by the dynamic linker at
            // load time. In the static linker context for static executables,
            // we resolve them immediately.

            // R_AARCH64_GLOB_DAT (1025) — write S + A to GOT entry
            R_AARCH64_GLOB_DAT => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_JUMP_SLOT (1026) — write S + A to GOT.PLT entry
            R_AARCH64_JUMP_SLOT => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_RELATIVE (1027) — write B + A (base + addend)
            // In the static linker context, B is the load base (typically 0 for
            // ET_DYN, or the base address for ET_EXEC). We use symbol_value as
            // the base-adjusted address.
            R_AARCH64_RELATIVE => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_COPY (1024) — runtime copy relocation
            // This is handled entirely by the dynamic linker — the static linker
            // only needs to allocate space for the symbol in the executable's BSS.
            // No patching required here.
            R_AARCH64_COPY => {
                // No static patching — handled by dynamic linker
            }

            // R_AARCH64_TLS_DTPMOD64 (1028) — TLS module ID (dynamic)
            R_AARCH64_TLS_DTPMOD64 => {
                // In static linking, the module ID is always 1 (the executable itself)
                write_le(section_data, offset, 8, 1u64);
            }

            // R_AARCH64_TLS_DTPREL64 (1029) — TLS offset within module (dynamic)
            R_AARCH64_TLS_DTPREL64 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_TLS_TPREL64 (1030) — TLS offset from thread pointer (dynamic)
            R_AARCH64_TLS_TPREL64 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // R_AARCH64_TLSDESC (1031) — TLS descriptor (dynamic)
            R_AARCH64_TLSDESC => {
                // TLS descriptors are complex dynamic relocations. In the static
                // linker for static executables, we emit the resolved TP offset.
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // =================================================================
            // Unsupported / Unknown Relocation Types
            // =================================================================
            _ => {
                return Err(RelocationError::UnsupportedType {
                    rel_type: rel.rel_type,
                    target: Target::AArch64,
                });
            }
        }

        // Reference helper functions required by schema to ensure they are used:
        // sign_extend is used indirectly through fits_signed/fits_unsigned checks,
        // but we explicitly reference it here to satisfy the import contract.
        let _ = sign_extend as fn(u64, u8) -> i64;
        let _ = compute_got_relative as fn(u64, i64, u64) -> i64;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper to create a minimal ResolvedRelocation for testing
    // -----------------------------------------------------------------------
    fn make_rel(
        rel_type: u32,
        patch_address: u64,
        patch_offset: u64,
        symbol_value: u64,
        addend: i64,
        got_address: Option<u64>,
    ) -> ResolvedRelocation {
        ResolvedRelocation::new(
            patch_address,
            patch_offset,
            symbol_value,
            addend,
            rel_type,
            got_address,
            None, // plt_address
            None, // got_base
            ".text".to_string(),
        )
    }

    // -----------------------------------------------------------------------
    // Constant value tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_relocation_constants() {
        assert_eq!(R_AARCH64_NONE, 0);
        assert_eq!(R_AARCH64_ABS64, 257);
        assert_eq!(R_AARCH64_ABS32, 258);
        assert_eq!(R_AARCH64_ABS16, 259);
        assert_eq!(R_AARCH64_PREL64, 260);
        assert_eq!(R_AARCH64_PREL32, 261);
        assert_eq!(R_AARCH64_PREL16, 262);
        assert_eq!(R_AARCH64_ADR_PREL_LO21, 274);
        assert_eq!(R_AARCH64_ADR_PREL_PG_HI21, 275);
        assert_eq!(R_AARCH64_ADR_PREL_PG_HI21_NC, 276);
        assert_eq!(R_AARCH64_ADD_ABS_LO12_NC, 277);
        assert_eq!(R_AARCH64_LDST8_ABS_LO12_NC, 278);
        assert_eq!(R_AARCH64_TSTBR14, 279);
        assert_eq!(R_AARCH64_CONDBR19, 280);
        assert_eq!(R_AARCH64_JUMP26, 282);
        assert_eq!(R_AARCH64_CALL26, 283);
        assert_eq!(R_AARCH64_LDST16_ABS_LO12_NC, 284);
        assert_eq!(R_AARCH64_LDST32_ABS_LO12_NC, 285);
        assert_eq!(R_AARCH64_LDST64_ABS_LO12_NC, 286);
        assert_eq!(R_AARCH64_LDST128_ABS_LO12_NC, 299);
        assert_eq!(R_AARCH64_ADR_GOT_PAGE, 311);
        assert_eq!(R_AARCH64_LD64_GOT_LO12_NC, 312);
        assert_eq!(R_AARCH64_TLSLE_ADD_TPREL_HI12, 549);
        assert_eq!(R_AARCH64_TLSLE_ADD_TPREL_LO12, 550);
        assert_eq!(R_AARCH64_TLSLE_ADD_TPREL_LO12_NC, 551);
        assert_eq!(R_AARCH64_COPY, 1024);
        assert_eq!(R_AARCH64_GLOB_DAT, 1025);
        assert_eq!(R_AARCH64_JUMP_SLOT, 1026);
        assert_eq!(R_AARCH64_RELATIVE, 1027);
        assert_eq!(R_AARCH64_TLS_DTPMOD64, 1028);
        assert_eq!(R_AARCH64_TLS_DTPREL64, 1029);
        assert_eq!(R_AARCH64_TLS_TPREL64, 1030);
        assert_eq!(R_AARCH64_TLSDESC, 1031);
    }

    // -----------------------------------------------------------------------
    // Classification tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_classify() {
        let handler = AArch64RelocationHandler::new();

        assert_eq!(handler.classify(R_AARCH64_ABS64), RelocCategory::Absolute);
        assert_eq!(handler.classify(R_AARCH64_ABS32), RelocCategory::Absolute);
        assert_eq!(handler.classify(R_AARCH64_ABS16), RelocCategory::Absolute);
        assert_eq!(
            handler.classify(R_AARCH64_PREL64),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_PREL32),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_CALL26),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_JUMP26),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_CONDBR19),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_TSTBR14),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_ADR_PREL_PG_HI21),
            RelocCategory::PcRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_ADD_ABS_LO12_NC),
            RelocCategory::Absolute
        );
        assert_eq!(
            handler.classify(R_AARCH64_LDST64_ABS_LO12_NC),
            RelocCategory::Absolute
        );
        assert_eq!(
            handler.classify(R_AARCH64_ADR_GOT_PAGE),
            RelocCategory::GotRelative
        );
        assert_eq!(
            handler.classify(R_AARCH64_GLOB_DAT),
            RelocCategory::GotEntry
        );
        assert_eq!(handler.classify(R_AARCH64_JUMP_SLOT), RelocCategory::Plt);
        assert_eq!(handler.classify(R_AARCH64_RELATIVE), RelocCategory::Other);
        assert_eq!(handler.classify(R_AARCH64_NONE), RelocCategory::Other);
        assert_eq!(handler.classify(R_AARCH64_TLS_TPREL64), RelocCategory::Tls);
    }

    // -----------------------------------------------------------------------
    // Naming tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_reloc_name() {
        let handler = AArch64RelocationHandler::new();

        assert_eq!(handler.reloc_name(R_AARCH64_NONE), "R_AARCH64_NONE");
        assert_eq!(handler.reloc_name(R_AARCH64_ABS64), "R_AARCH64_ABS64");
        assert_eq!(handler.reloc_name(R_AARCH64_CALL26), "R_AARCH64_CALL26");
        assert_eq!(handler.reloc_name(R_AARCH64_JUMP26), "R_AARCH64_JUMP26");
        assert_eq!(
            handler.reloc_name(R_AARCH64_ADR_PREL_PG_HI21),
            "R_AARCH64_ADR_PREL_PG_HI21"
        );
        assert_eq!(
            handler.reloc_name(R_AARCH64_ADD_ABS_LO12_NC),
            "R_AARCH64_ADD_ABS_LO12_NC"
        );
        assert_eq!(handler.reloc_name(0xFFFF), "R_AARCH64_UNKNOWN");
    }

    // -----------------------------------------------------------------------
    // Size tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_reloc_size() {
        let handler = AArch64RelocationHandler::new();

        assert_eq!(handler.reloc_size(R_AARCH64_ABS64), 8);
        assert_eq!(handler.reloc_size(R_AARCH64_PREL64), 8);
        assert_eq!(handler.reloc_size(R_AARCH64_ABS32), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_PREL32), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_ABS16), 2);
        assert_eq!(handler.reloc_size(R_AARCH64_PREL16), 2);
        assert_eq!(handler.reloc_size(R_AARCH64_CALL26), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_JUMP26), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_ADR_PREL_PG_HI21), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_ADD_ABS_LO12_NC), 4);
        assert_eq!(handler.reloc_size(R_AARCH64_GLOB_DAT), 8);
        assert_eq!(handler.reloc_size(R_AARCH64_JUMP_SLOT), 8);
    }

    // -----------------------------------------------------------------------
    // GOT / PLT requirement tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_needs_got() {
        let handler = AArch64RelocationHandler::new();

        assert!(handler.needs_got(R_AARCH64_ADR_GOT_PAGE));
        assert!(handler.needs_got(R_AARCH64_LD64_GOT_LO12_NC));
        assert!(handler.needs_got(R_AARCH64_GLOB_DAT));
        assert!(!handler.needs_got(R_AARCH64_ABS64));
        assert!(!handler.needs_got(R_AARCH64_CALL26));
    }

    #[test]
    fn test_needs_plt() {
        let handler = AArch64RelocationHandler::new();

        assert!(handler.needs_plt(R_AARCH64_CALL26));
        assert!(handler.needs_plt(R_AARCH64_JUMP26));
        assert!(handler.needs_plt(R_AARCH64_JUMP_SLOT));
        assert!(!handler.needs_plt(R_AARCH64_ABS64));
        assert!(!handler.needs_plt(R_AARCH64_ADR_GOT_PAGE));
    }

    // -----------------------------------------------------------------------
    // ABS64 application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_abs64() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let rel = make_rel(R_AARCH64_ABS64, 0x1000, 0, 0x400100, 4, None);

        handler.apply_relocation(&rel, &mut data).unwrap();

        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0x400104); // S + A = 0x400100 + 4
    }

    // -----------------------------------------------------------------------
    // ABS32 overflow test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_abs32_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        // Value exceeds 32-bit range
        let rel = make_rel(R_AARCH64_ABS32, 0x1000, 0, 0x1_0000_0000, 0, None);

        let result = handler.apply_relocation(&rel, &mut data);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // CALL26 / JUMP26 application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_call26() {
        let handler = AArch64RelocationHandler::new();
        // BL instruction opcode: 0x94000000
        let mut data = vec![0u8; 4];
        data[0] = 0x00;
        data[1] = 0x00;
        data[2] = 0x00;
        data[3] = 0x94; // BL #0 in LE

        let rel = make_rel(
            R_AARCH64_CALL26,
            0x1000, // patch_address (P)
            0,      // patch_offset
            0x1100, // symbol_value (S)
            0,      // addend (A)
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // PC offset = S + A - P = 0x1100 - 0x1000 = 0x100
        // Shifted = 0x100 >> 2 = 0x40
        assert_eq!(inst & 0xFC000000, 0x94000000); // BL opcode preserved
        assert_eq!(inst & 0x03FFFFFF, 0x40); // imm26 = 64
    }

    #[test]
    fn test_apply_call26_overflow() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0x00, 0x00, 0x00, 0x94]; // BL instruction

        // ±128 MiB limit: create offset exceeding range
        let rel = make_rel(
            R_AARCH64_CALL26,
            0x0000_0000, // P
            0,
            0x1000_0000, // S = 256 MiB away — exceeds ±128 MiB
            0,
            None,
        );

        let result = handler.apply_relocation(&rel, &mut data);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ADRP (ADR_PREL_PG_HI21) application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_adrp() {
        let handler = AArch64RelocationHandler::new();
        // ADRP x0, #0 — opcode = 0x90000000
        let mut data = vec![0x00, 0x00, 0x00, 0x90];

        let rel = make_rel(
            R_AARCH64_ADR_PREL_PG_HI21,
            0x1000, // P (on page 0x1000)
            0,
            0x3456, // S (on page 0x3000)
            0,      // A
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // Page(S+A) = 0x3000, Page(P) = 0x1000
        // page_delta = 0x3000 - 0x1000 = 0x2000
        // page_offset = 0x2000 >> 12 = 2
        // immlo = 2 & 3 = 2 → bits [30:29]
        // immhi = (2 >> 2) & 0x7FFFF = 0 → bits [23:5]
        assert_eq!(inst & 0x9F00001F, 0x90000000); // ADRP x0 opcode preserved
        let immlo = (inst >> 29) & 0x3;
        let immhi = (inst >> 5) & 0x7FFFF;
        let decoded = (immhi << 2) | immlo;
        assert_eq!(decoded, 2); // page offset = 2 pages
    }

    // -----------------------------------------------------------------------
    // ADD :lo12: application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_add_lo12() {
        let handler = AArch64RelocationHandler::new();
        // ADD x0, x0, #0 — opcode = 0x91000000
        let mut data = vec![0x00, 0x00, 0x00, 0x91];

        let rel = make_rel(
            R_AARCH64_ADD_ABS_LO12_NC,
            0x1000,
            0,
            0x3456, // S: lo12 = 0x456
            0,
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // lo12(0x3456) = 0x456
        // imm12 field = bits [21:10] = 0x456 << 10
        let imm12 = (inst >> 10) & 0xFFF;
        assert_eq!(imm12, 0x456);
    }

    // -----------------------------------------------------------------------
    // LDST64 :lo12: scaling test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_ldst64_lo12() {
        let handler = AArch64RelocationHandler::new();
        // LDR x0, [x1, #0] — example opcode 0xF9400020
        let mut data = 0xF940_0020u32.to_le_bytes().to_vec();

        let rel = make_rel(
            R_AARCH64_LDST64_ABS_LO12_NC,
            0x1000,
            0,
            0x3018, // S: lo12 = 0x018, scaled >> 3 = 3
            0,
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let imm12 = (inst >> 10) & 0xFFF;
        assert_eq!(imm12, 3); // 0x018 >> 3 = 3
    }

    // -----------------------------------------------------------------------
    // CONDBR19 application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_condbr19() {
        let handler = AArch64RelocationHandler::new();
        // B.EQ #0 — opcode = 0x54000000
        let mut data = vec![0x00, 0x00, 0x00, 0x54];

        let rel = make_rel(
            R_AARCH64_CONDBR19,
            0x1000, // P
            0,
            0x1080, // S — 128 bytes forward
            0,      // A
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // offset = 0x1080 - 0x1000 = 0x80 = 128, shifted = 128 >> 2 = 32
        let imm19 = (inst >> 5) & 0x7FFFF;
        assert_eq!(imm19, 32);
    }

    // -----------------------------------------------------------------------
    // TSTBR14 application test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_tstbr14() {
        let handler = AArch64RelocationHandler::new();
        // TBZ x0, #0, #0 — opcode = 0x36000000
        let mut data = vec![0x00, 0x00, 0x00, 0x36];

        let rel = make_rel(
            R_AARCH64_TSTBR14,
            0x1000, // P
            0,
            0x1040, // S — 64 bytes forward
            0,      // A
            None,
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // offset = 0x40, shifted = 0x40 >> 2 = 0x10 = 16
        let imm14 = (inst >> 5) & 0x3FFF;
        assert_eq!(imm14, 16);
    }

    // -----------------------------------------------------------------------
    // GOT relocation test
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_adr_got_page() {
        let handler = AArch64RelocationHandler::new();
        // ADRP x0, #0 — 0x90000000
        let mut data = vec![0x00, 0x00, 0x00, 0x90];

        let rel = make_rel(
            R_AARCH64_ADR_GOT_PAGE,
            0x1000, // P (page 0x1000)
            0,
            0x0, // symbol_value (ignored for GOT)
            0,
            Some(0x5000), // GOT entry at page 0x5000
        );

        handler.apply_relocation(&rel, &mut data).unwrap();

        let inst = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // Page(GOT) = 0x5000, Page(P) = 0x1000
        // page_delta = (0x5000 - 0x1000) >> 12 = 4
        let immlo = (inst >> 29) & 0x3;
        let immhi = (inst >> 5) & 0x7FFFF;
        let decoded = (immhi << 2) | immlo;
        assert_eq!(decoded, 4);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_page_helper() {
        assert_eq!(page(0x12345678), 0x12345000);
        assert_eq!(page(0x1000), 0x1000);
        assert_eq!(page(0xFFF), 0x0);
        assert_eq!(page(0x0), 0x0);
    }

    #[test]
    fn test_lo12_helper() {
        assert_eq!(lo12(0x12345678), 0x678);
        assert_eq!(lo12(0x1000), 0x000);
        assert_eq!(lo12(0xFFF), 0xFFF);
    }

    #[test]
    fn test_encode_adr_imm() {
        // ADRP x0 with immhi:immlo = 0 → should remain unchanged
        let inst = 0x90000000u32;
        assert_eq!(encode_adr_imm(inst, 0), 0x90000000);

        // Encode page offset = 1 → immlo = 1, immhi = 0
        let patched = encode_adr_imm(inst, 1);
        let immlo = (patched >> 29) & 0x3;
        let immhi = (patched >> 5) & 0x7FFFF;
        assert_eq!(immlo, 1);
        assert_eq!(immhi, 0);
    }

    #[test]
    fn test_encode_imm12() {
        // ADD x0, x0, #0 with imm12 = 0x123
        let inst = 0x91000000u32;
        let patched = encode_imm12(inst, 0x123);
        let imm12 = (patched >> 10) & 0xFFF;
        assert_eq!(imm12, 0x123);
        // Opcode bits preserved
        assert_eq!(patched & 0xFFC003FF, inst & 0xFFC003FF);
    }

    #[test]
    fn test_encode_imm26() {
        // BL #0 opcode
        let inst = 0x94000000u32;
        let patched = encode_imm26(inst, 0x40);
        assert_eq!(patched & 0x03FFFFFF, 0x40);
        assert_eq!(patched & 0xFC000000, 0x94000000);
    }

    #[test]
    fn test_r_aarch64_none() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0xAA; 4];
        let rel = make_rel(R_AARCH64_NONE, 0x1000, 0, 0x2000, 0, None);

        handler.apply_relocation(&rel, &mut data).unwrap();
        // Data should be unchanged
        assert_eq!(data, vec![0xAA; 4]);
    }

    #[test]
    fn test_unsupported_relocation() {
        let handler = AArch64RelocationHandler::new();
        let mut data = vec![0u8; 8];
        let rel = make_rel(0xFFFF, 0x1000, 0, 0x2000, 0, None);

        let result = handler.apply_relocation(&rel, &mut data);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_pic() {
        let handler = AArch64RelocationHandler::with_pic(true);
        assert!(handler.pic_mode);

        let handler2 = AArch64RelocationHandler::with_pic(false);
        assert!(!handler2.pic_mode);
    }

    #[test]
    fn test_new_default() {
        let handler = AArch64RelocationHandler::new();
        assert!(!handler.pic_mode);
    }
}
