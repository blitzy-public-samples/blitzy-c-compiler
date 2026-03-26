//! # AArch64 ELF Relocation Types and Processing
//!
//! Defines all AArch64-specific ELF relocation types, provides functions for
//! creating relocation entries during assembly, and applies relocations during
//! linking.
//!
//! ## Relocation Categories
//!
//! ### Absolute Relocations
//! - R_AARCH64_ABS64: 64-bit absolute address (for data pointers, GOT entries)
//! - R_AARCH64_ABS32: 32-bit absolute address (for 32-bit data, DWARF)
//! - R_AARCH64_ABS16: 16-bit absolute address
//!
//! ### PC-Relative Relocations
//! - R_AARCH64_PREL64: 64-bit PC-relative (rare, debug info)
//! - R_AARCH64_PREL32: 32-bit PC-relative (DWARF, eh_frame)
//! - R_AARCH64_PREL16: 16-bit PC-relative
//!
//! ### Page-Relative (ADRP) Relocations
//! - R_AARCH64_ADR_PREL_PG_HI21: ADRP page-relative offset (bits [32:12])
//! - R_AARCH64_ADD_ABS_LO12_NC: ADD low 12-bit offset (no check)
//! - R_AARCH64_LDST*_ABS_LO12_NC: LDR/STR low 12-bit scaled offset
//!
//! ### Branch Relocations
//! - R_AARCH64_CALL26: BL +/-128 MiB (26-bit signed word offset)
//! - R_AARCH64_JUMP26: B +/-128 MiB (26-bit signed word offset)
//! - R_AARCH64_CONDBR19: B.cond / CBZ / CBNZ +/-1 MiB
//! - R_AARCH64_TSTBR14: TBZ / TBNZ +/-32 KiB
//!
//! ### GOT-Relative Relocations (PIC)
//! - R_AARCH64_ADR_GOT_PAGE: ADRP to GOT entry page
//! - R_AARCH64_LD64_GOT_LO12_NC: LDR from GOT entry (low 12-bit)
//!
//! ### Dynamic Relocations
//! - R_AARCH64_GLOB_DAT: GOT entry (absolute address)
//! - R_AARCH64_JUMP_SLOT: PLT GOT slot (lazy binding)
//! - R_AARCH64_RELATIVE: Base-relative (for -fPIC data)
//! - R_AARCH64_COPY: Copy relocation for data symbols
//!
//! ### TLS Relocations
//! - R_AARCH64_TLSDESC_ADR_PAGE21, R_AARCH64_TLSDESC_LD64_LO12
//! - R_AARCH64_TLSDESC_ADD_LO12, R_AARCH64_TLSDESC_CALL
//!
//! (For Thread-Local Storage support, used by kernel/glibc)

use crate::backend::linker_common::relocation::{
    RelocCategory, RelocationError, RelocationHandler, ResolvedRelocation,
};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// AArch64RelocationType enum
// ---------------------------------------------------------------------------

/// AArch64 ELF relocation type, matching the official ELF ABI values.
///
/// Each variant's discriminant is the raw `r_type` value found in ELF
/// relocation entries (`Elf64_Rela.r_info`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum AArch64RelocationType {
    // === Null ===
    /// No relocation.
    None = 0,

    // === Absolute Relocations (Data) ===
    /// S + A : 64-bit absolute address.
    Abs64 = 257,
    /// S + A : 32-bit absolute address.
    Abs32 = 258,
    /// S + A : 16-bit absolute address.
    Abs16 = 259,

    // === PC-Relative (Data) ===
    /// S + A - P : 64-bit PC-relative.
    Prel64 = 260,
    /// S + A - P : 32-bit PC-relative.
    Prel32 = 261,
    /// S + A - P : 16-bit PC-relative.
    Prel16 = 262,

    // === Group Relocations (Absolute MOVW) ===
    /// MOVZ/MOVK bits [15:0].
    MovwUabsG0 = 263,
    /// MOVZ/MOVK bits [15:0] no overflow check.
    MovwUabsG0Nc = 264,
    /// MOVZ/MOVK bits [31:16].
    MovwUabsG1 = 265,
    /// MOVZ/MOVK bits [31:16] no overflow check.
    MovwUabsG1Nc = 266,
    /// MOVZ/MOVK bits [47:32].
    MovwUabsG2 = 267,
    /// MOVZ/MOVK bits [47:32] no overflow check.
    MovwUabsG2Nc = 268,
    /// MOVZ/MOVK bits [63:48].
    MovwUabsG3 = 269,

    // === Page-Relative (ADRP + ADD/LDR) ===
    /// Page(S + A) - Page(P) : ADRP 21-bit page offset.
    AdrPrelPgHi21 = 275,
    /// Page(S + A) - Page(P) : ADRP 21-bit page offset, no overflow check.
    AdrPrelPgHi21Nc = 276,
    /// (S + A) & 0xFFF : ADD :lo12: 12-bit low offset.
    AddAbsLo12Nc = 277,
    /// (S + A) & 0xFFF : LDR/STR byte :lo12:.
    Ldst8AbsLo12Nc = 278,
    /// ((S + A) & 0xFFF) >> 1 : LDR/STR halfword :lo12:.
    Ldst16AbsLo12Nc = 284,
    /// ((S + A) & 0xFFF) >> 2 : LDR/STR word :lo12:.
    Ldst32AbsLo12Nc = 285,
    /// ((S + A) & 0xFFF) >> 3 : LDR/STR doubleword :lo12:.
    Ldst64AbsLo12Nc = 286,
    /// ((S + A) & 0xFFF) >> 4 : LDR/STR quadword :lo12:.
    Ldst128AbsLo12Nc = 299,

    // === Branch Relocations ===
    /// TBZ/TBNZ +/-32 KiB : 14-bit signed word offset.
    Tstbr14 = 279,
    /// B.cond/CBZ/CBNZ +/-1 MiB : 19-bit signed word offset.
    Condbr19 = 280,
    /// B +/-128 MiB : 26-bit signed word offset.
    Jump26 = 282,
    /// BL +/-128 MiB : 26-bit signed word offset.
    Call26 = 283,

    // === GOT-Relative Relocations (PIC) ===
    /// Page(G(S)) - Page(P) : ADRP to GOT entry page.
    AdrGotPage = 311,
    /// G(S) & 0xFFF : LDR from GOT entry, low 12 bits (8-byte aligned).
    Ld64GotLo12Nc = 312,

    // === TLS Relocations ===
    /// Page offset for TLS descriptor ADRP.
    TlsdescAdrPage21 = 566,
    /// LDR offset for TLS descriptor.
    TlsdescLd64Lo12 = 567,
    /// ADD offset for TLS descriptor.
    TlsdescAddLo12 = 568,
    /// BLR for TLS descriptor call.
    TlsdescCall = 569,

    // === Dynamic Relocations ===
    /// Copy relocation for data symbols.
    Copy = 1024,
    /// GOT entry absolute address (S + A).
    GlobDat = 1025,
    /// PLT GOT slot for lazy binding.
    JumpSlot = 1026,
    /// Base-relative relocation (B + A) for PIC data.
    Relative = 1027,
}

impl AArch64RelocationType {
    /// Convert a raw ELF `r_type` value to an `AArch64RelocationType`.
    ///
    /// Returns `None` for unsupported or unrecognized relocation types.
    pub fn from_raw(r_type: u32) -> Option<Self> {
        match r_type {
            0 => Some(Self::None),
            257 => Some(Self::Abs64),
            258 => Some(Self::Abs32),
            259 => Some(Self::Abs16),
            260 => Some(Self::Prel64),
            261 => Some(Self::Prel32),
            262 => Some(Self::Prel16),
            263 => Some(Self::MovwUabsG0),
            264 => Some(Self::MovwUabsG0Nc),
            265 => Some(Self::MovwUabsG1),
            266 => Some(Self::MovwUabsG1Nc),
            267 => Some(Self::MovwUabsG2),
            268 => Some(Self::MovwUabsG2Nc),
            269 => Some(Self::MovwUabsG3),
            275 => Some(Self::AdrPrelPgHi21),
            276 => Some(Self::AdrPrelPgHi21Nc),
            277 => Some(Self::AddAbsLo12Nc),
            278 => Some(Self::Ldst8AbsLo12Nc),
            279 => Some(Self::Tstbr14),
            280 => Some(Self::Condbr19),
            282 => Some(Self::Jump26),
            283 => Some(Self::Call26),
            284 => Some(Self::Ldst16AbsLo12Nc),
            285 => Some(Self::Ldst32AbsLo12Nc),
            286 => Some(Self::Ldst64AbsLo12Nc),
            299 => Some(Self::Ldst128AbsLo12Nc),
            311 => Some(Self::AdrGotPage),
            312 => Some(Self::Ld64GotLo12Nc),
            566 => Some(Self::TlsdescAdrPage21),
            567 => Some(Self::TlsdescLd64Lo12),
            568 => Some(Self::TlsdescAddLo12),
            569 => Some(Self::TlsdescCall),
            1024 => Some(Self::Copy),
            1025 => Some(Self::GlobDat),
            1026 => Some(Self::JumpSlot),
            1027 => Some(Self::Relative),
            _ => None,
        }
    }

    /// Return the raw ELF `r_type` constant value for this relocation.
    pub fn to_raw(&self) -> u32 {
        *self as u32
    }

    /// Return a human-readable name for diagnostic messages.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "R_AARCH64_NONE",
            Self::Abs64 => "R_AARCH64_ABS64",
            Self::Abs32 => "R_AARCH64_ABS32",
            Self::Abs16 => "R_AARCH64_ABS16",
            Self::Prel64 => "R_AARCH64_PREL64",
            Self::Prel32 => "R_AARCH64_PREL32",
            Self::Prel16 => "R_AARCH64_PREL16",
            Self::MovwUabsG0 => "R_AARCH64_MOVW_UABS_G0",
            Self::MovwUabsG0Nc => "R_AARCH64_MOVW_UABS_G0_NC",
            Self::MovwUabsG1 => "R_AARCH64_MOVW_UABS_G1",
            Self::MovwUabsG1Nc => "R_AARCH64_MOVW_UABS_G1_NC",
            Self::MovwUabsG2 => "R_AARCH64_MOVW_UABS_G2",
            Self::MovwUabsG2Nc => "R_AARCH64_MOVW_UABS_G2_NC",
            Self::MovwUabsG3 => "R_AARCH64_MOVW_UABS_G3",
            Self::AdrPrelPgHi21 => "R_AARCH64_ADR_PREL_PG_HI21",
            Self::AdrPrelPgHi21Nc => "R_AARCH64_ADR_PREL_PG_HI21_NC",
            Self::AddAbsLo12Nc => "R_AARCH64_ADD_ABS_LO12_NC",
            Self::Ldst8AbsLo12Nc => "R_AARCH64_LDST8_ABS_LO12_NC",
            Self::Ldst16AbsLo12Nc => "R_AARCH64_LDST16_ABS_LO12_NC",
            Self::Ldst32AbsLo12Nc => "R_AARCH64_LDST32_ABS_LO12_NC",
            Self::Ldst64AbsLo12Nc => "R_AARCH64_LDST64_ABS_LO12_NC",
            Self::Ldst128AbsLo12Nc => "R_AARCH64_LDST128_ABS_LO12_NC",
            Self::Tstbr14 => "R_AARCH64_TSTBR14",
            Self::Condbr19 => "R_AARCH64_CONDBR19",
            Self::Jump26 => "R_AARCH64_JUMP26",
            Self::Call26 => "R_AARCH64_CALL26",
            Self::AdrGotPage => "R_AARCH64_ADR_GOT_PAGE",
            Self::Ld64GotLo12Nc => "R_AARCH64_LD64_GOT_LO12_NC",
            Self::TlsdescAdrPage21 => "R_AARCH64_TLSDESC_ADR_PAGE21",
            Self::TlsdescLd64Lo12 => "R_AARCH64_TLSDESC_LD64_LO12",
            Self::TlsdescAddLo12 => "R_AARCH64_TLSDESC_ADD_LO12",
            Self::TlsdescCall => "R_AARCH64_TLSDESC_CALL",
            Self::Copy => "R_AARCH64_COPY",
            Self::GlobDat => "R_AARCH64_GLOB_DAT",
            Self::JumpSlot => "R_AARCH64_JUMP_SLOT",
            Self::Relative => "R_AARCH64_RELATIVE",
        }
    }

    /// Classify this relocation into a linker processing category.
    pub fn category(&self) -> RelocCategory {
        match self {
            // Absolute data relocations
            Self::Abs64 | Self::Abs32 | Self::Abs16 => RelocCategory::Absolute,

            // PC-relative data and branch relocations
            Self::Prel64 | Self::Prel32 | Self::Prel16 => RelocCategory::PcRelative,
            Self::AdrPrelPgHi21 | Self::AdrPrelPgHi21Nc => RelocCategory::PcRelative,
            Self::Jump26 | Self::Call26 => RelocCategory::PcRelative,
            Self::Condbr19 | Self::Tstbr14 => RelocCategory::PcRelative,

            // Low-12 section-relative relocations (ADD/LDST)
            Self::AddAbsLo12Nc
            | Self::Ldst8AbsLo12Nc
            | Self::Ldst16AbsLo12Nc
            | Self::Ldst32AbsLo12Nc
            | Self::Ldst64AbsLo12Nc
            | Self::Ldst128AbsLo12Nc => RelocCategory::SectionRelative,

            // MOVW group
            Self::MovwUabsG0
            | Self::MovwUabsG0Nc
            | Self::MovwUabsG1
            | Self::MovwUabsG1Nc
            | Self::MovwUabsG2
            | Self::MovwUabsG2Nc
            | Self::MovwUabsG3 => RelocCategory::Absolute,

            // GOT-relative relocations (require GOT entry)
            Self::AdrGotPage | Self::Ld64GotLo12Nc => RelocCategory::GotRelative,

            // TLS relocations
            Self::TlsdescAdrPage21
            | Self::TlsdescLd64Lo12
            | Self::TlsdescAddLo12
            | Self::TlsdescCall => RelocCategory::Tls,

            // Dynamic relocations
            Self::Copy => RelocCategory::Other,
            Self::GlobDat => RelocCategory::GotEntry,
            Self::JumpSlot => RelocCategory::Plt,
            Self::Relative => RelocCategory::Absolute,

            Self::None => RelocCategory::Other,
        }
    }

    /// Returns `true` if this relocation type requires a GOT entry.
    pub fn needs_got_entry(&self) -> bool {
        matches!(
            self,
            Self::AdrGotPage | Self::Ld64GotLo12Nc | Self::GlobDat | Self::JumpSlot
        )
    }

    /// Returns `true` if this relocation type indicates eligibility for a PLT
    /// stub (the linker ultimately decides based on symbol binding).
    pub fn needs_plt_entry(&self) -> bool {
        matches!(self, Self::Call26 | Self::Jump26 | Self::JumpSlot)
    }
}

// ---------------------------------------------------------------------------
// Overflow checking helpers
// ---------------------------------------------------------------------------

/// Check whether a signed value fits in `bits` bits.
#[inline]
fn check_signed_overflow(value: i64, bits: u32) -> bool {
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    value >= min && value <= max
}

/// Check whether an unsigned value fits in `bits` bits.
#[inline]
fn check_unsigned_overflow(value: u64, bits: u32) -> bool {
    value < (1u64 << bits)
}

// ---------------------------------------------------------------------------
// Instruction / data word I/O helpers (little-endian)
// ---------------------------------------------------------------------------

/// Read a 32-bit instruction word from a byte slice (little-endian).
#[inline]
pub fn read_instruction(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    u32::from_le_bytes(bytes)
}

/// Write a 32-bit instruction word to a byte slice (little-endian).
#[inline]
pub fn write_instruction(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
}

/// Read a 64-bit value from a byte slice (little-endian).
#[inline]
pub fn read_u64(data: &[u8], offset: usize) -> u64 {
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

/// Write a 64-bit value to a byte slice (little-endian).
#[inline]
pub fn write_u64(data: &mut [u8], offset: usize, value: u64) {
    let bytes = value.to_le_bytes();
    data[offset..offset + 8].copy_from_slice(&bytes);
}

/// Write a 32-bit value to a byte slice (little-endian) for data words.
#[inline]
fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset..offset + 4].copy_from_slice(&bytes);
}

/// Write a 16-bit value to a byte slice (little-endian).
#[inline]
fn write_u16(data: &mut [u8], offset: usize, value: u16) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
}

// ---------------------------------------------------------------------------
// Instruction field patching helpers
// ---------------------------------------------------------------------------

/// Insert a 26-bit signed immediate into a B/BL instruction word.
/// Encoding: imm26 at bits [25:0], mask 0x03FF_FFFF.
#[inline]
fn patch_imm26(inst: u32, imm26: i32) -> u32 {
    (inst & !0x03FF_FFFF) | ((imm26 as u32) & 0x03FF_FFFF)
}

/// Insert a 19-bit signed immediate into a B.cond/CBZ/CBNZ instruction.
/// Encoding: imm19 at bits [23:5], mask 0x00FF_FFE0.
#[inline]
fn patch_imm19(inst: u32, imm19: i32) -> u32 {
    (inst & !0x00FF_FFE0) | (((imm19 as u32) & 0x7FFFF) << 5)
}

/// Insert a 14-bit signed immediate into a TBZ/TBNZ instruction.
/// Encoding: imm14 at bits [18:5], mask 0x0007_FFE0.
#[inline]
fn patch_imm14(inst: u32, imm14: i32) -> u32 {
    (inst & !0x0007_FFE0) | (((imm14 as u32) & 0x3FFF) << 5)
}

/// Insert a 21-bit signed immediate into an ADRP/ADR instruction.
/// immhi (19 bits) at bits [23:5], immlo (2 bits) at bits [30:29].
#[inline]
fn patch_adr_imm(inst: u32, imm21: i32) -> u32 {
    let imm = imm21 as u32;
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7FFFF;
    (inst & !(0x00FF_FFE0 | (0x3 << 29))) | (immhi << 5) | (immlo << 29)
}

/// Insert a 12-bit unsigned immediate into ADD/LDR/STR instructions.
/// Encoding: imm12 at bits [21:10], mask 0x003F_FC00.
#[inline]
fn patch_imm12(inst: u32, imm12: u32) -> u32 {
    (inst & !0x003F_FC00) | ((imm12 & 0xFFF) << 10)
}

/// Insert a 16-bit immediate into MOVZ/MOVK/MOVN instructions.
/// Encoding: imm16 at bits [20:5], mask 0x001F_FFE0.
#[inline]
fn patch_imm16(inst: u32, imm16: u16) -> u32 {
    (inst & !0x001F_FFE0) | ((imm16 as u32) << 5)
}

// ---------------------------------------------------------------------------
// apply_relocation -- main relocation application function
// ---------------------------------------------------------------------------

/// Apply a single AArch64 ELF relocation by patching section data.
///
/// # Arguments
/// * `data`       - mutable byte slice of the section being patched
/// * `offset`     - byte offset within data where the instruction/data lives
/// * `reloc_type` - which AArch64 relocation to apply
/// * `sym_value`  - S (resolved symbol address)
/// * `addend`     - A (addend from RELA entry)
/// * `pc`         - P (address of the instruction being patched)
///
/// # Errors
/// Returns `Err(String)` when the computed value overflows the relocation
/// bit width, including the relocation name, expected range, and actual value.
pub fn apply_relocation(
    data: &mut [u8],
    offset: usize,
    reloc_type: AArch64RelocationType,
    sym_value: u64,
    addend: i64,
    pc: u64,
) -> Result<(), String> {
    let sa = sym_value as i64 + addend;

    match reloc_type {
        AArch64RelocationType::None => {}

        // === Absolute data relocations ===
        AArch64RelocationType::Abs64 => {
            write_u64(data, offset, sa as u64);
        }
        AArch64RelocationType::Abs32 => {
            let val = sa as u64;
            if !check_unsigned_overflow(val, 32) && !check_signed_overflow(sa, 32) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in 32 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            write_u32(data, offset, val as u32);
        }
        AArch64RelocationType::Abs16 => {
            let val = sa as u64;
            if !check_unsigned_overflow(val, 16) && !check_signed_overflow(sa, 16) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in 16 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            write_u16(data, offset, val as u16);
        }

        // === PC-relative data relocations ===
        AArch64RelocationType::Prel64 => {
            let val = sa - pc as i64;
            write_u64(data, offset, val as u64);
        }
        AArch64RelocationType::Prel32 => {
            let val = sa - pc as i64;
            if !check_signed_overflow(val, 32) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in signed 32 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            write_u32(data, offset, val as u32);
        }
        AArch64RelocationType::Prel16 => {
            let val = sa - pc as i64;
            if !check_signed_overflow(val, 16) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in signed 16 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            write_u16(data, offset, val as u16);
        }

        // === ADRP page-relative ===
        AArch64RelocationType::AdrPrelPgHi21 => {
            let page_s = sa & !0xFFF;
            let page_p = pc as i64 & !0xFFF;
            let page_off = (page_s - page_p) >> 12;
            if !check_signed_overflow(page_off, 21) {
                return Err(format!(
                    "{}: page offset {} (0x{:x}) does not fit in signed 21 bits",
                    reloc_type.name(),
                    page_off,
                    page_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_adr_imm(inst, page_off as i32));
        }
        AArch64RelocationType::AdrPrelPgHi21Nc => {
            let page_s = sa & !0xFFF;
            let page_p = pc as i64 & !0xFFF;
            let page_off = (page_s - page_p) >> 12;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_adr_imm(inst, page_off as i32));
        }

        // === ADD / LDST low-12 (no overflow check) ===
        AArch64RelocationType::AddAbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12));
        }
        AArch64RelocationType::Ldst8AbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12));
        }
        AArch64RelocationType::Ldst16AbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 1));
        }
        AArch64RelocationType::Ldst32AbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 2));
        }
        AArch64RelocationType::Ldst64AbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 3));
        }
        AArch64RelocationType::Ldst128AbsLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 4));
        }

        // === Branch relocations ===
        AArch64RelocationType::Call26 | AArch64RelocationType::Jump26 => {
            let branch_off = (sa - pc as i64) >> 2;
            if !check_signed_overflow(branch_off, 26) {
                return Err(format!(
                    "{}: branch offset {} (0x{:x}) out of range (signed 26-bit, +/-128 MiB)",
                    reloc_type.name(),
                    branch_off,
                    branch_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm26(inst, branch_off as i32));
        }
        AArch64RelocationType::Condbr19 => {
            let branch_off = (sa - pc as i64) >> 2;
            if !check_signed_overflow(branch_off, 19) {
                return Err(format!(
                    "{}: cond branch offset {} (0x{:x}) out of range (signed 19-bit, +/-1 MiB)",
                    reloc_type.name(),
                    branch_off,
                    branch_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm19(inst, branch_off as i32));
        }
        AArch64RelocationType::Tstbr14 => {
            let branch_off = (sa - pc as i64) >> 2;
            if !check_signed_overflow(branch_off, 14) {
                return Err(format!(
                    "{}: test/branch offset {} (0x{:x}) out of range (signed 14-bit, +/-32 KiB)",
                    reloc_type.name(),
                    branch_off,
                    branch_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm14(inst, branch_off as i32));
        }

        // === GOT relocations ===
        AArch64RelocationType::AdrGotPage => {
            let page_s = sa & !0xFFF;
            let page_p = pc as i64 & !0xFFF;
            let page_off = (page_s - page_p) >> 12;
            if !check_signed_overflow(page_off, 21) {
                return Err(format!(
                    "{}: GOT page offset {} does not fit in signed 21 bits",
                    reloc_type.name(),
                    page_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_adr_imm(inst, page_off as i32));
        }
        AArch64RelocationType::Ld64GotLo12Nc => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 3));
        }

        // === MOVW group relocations ===
        AArch64RelocationType::MovwUabsG0 => {
            let val = sa as u64;
            if !check_unsigned_overflow(val, 16) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in 16 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm16(inst, (val & 0xFFFF) as u16));
        }
        AArch64RelocationType::MovwUabsG0Nc => {
            let val = sa as u64;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm16(inst, (val & 0xFFFF) as u16));
        }
        AArch64RelocationType::MovwUabsG1 => {
            let val = sa as u64;
            if !check_unsigned_overflow(val, 32) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in 32 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(
                data,
                offset,
                patch_imm16(inst, ((val >> 16) & 0xFFFF) as u16),
            );
        }
        AArch64RelocationType::MovwUabsG1Nc => {
            let val = sa as u64;
            let inst = read_instruction(data, offset);
            write_instruction(
                data,
                offset,
                patch_imm16(inst, ((val >> 16) & 0xFFFF) as u16),
            );
        }
        AArch64RelocationType::MovwUabsG2 => {
            let val = sa as u64;
            if !check_unsigned_overflow(val, 48) {
                return Err(format!(
                    "{}: value 0x{:x} does not fit in 48 bits",
                    reloc_type.name(),
                    val,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(
                data,
                offset,
                patch_imm16(inst, ((val >> 32) & 0xFFFF) as u16),
            );
        }
        AArch64RelocationType::MovwUabsG2Nc => {
            let val = sa as u64;
            let inst = read_instruction(data, offset);
            write_instruction(
                data,
                offset,
                patch_imm16(inst, ((val >> 32) & 0xFFFF) as u16),
            );
        }
        AArch64RelocationType::MovwUabsG3 => {
            let val = sa as u64;
            let inst = read_instruction(data, offset);
            write_instruction(
                data,
                offset,
                patch_imm16(inst, ((val >> 48) & 0xFFFF) as u16),
            );
        }

        // === TLS descriptor relocations ===
        AArch64RelocationType::TlsdescAdrPage21 => {
            let page_s = sa & !0xFFF;
            let page_p = pc as i64 & !0xFFF;
            let page_off = (page_s - page_p) >> 12;
            if !check_signed_overflow(page_off, 21) {
                return Err(format!(
                    "{}: TLS page offset {} does not fit in signed 21 bits",
                    reloc_type.name(),
                    page_off,
                ));
            }
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_adr_imm(inst, page_off as i32));
        }
        AArch64RelocationType::TlsdescLd64Lo12 => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12 >> 3));
        }
        AArch64RelocationType::TlsdescAddLo12 => {
            let lo12 = (sa as u64 & 0xFFF) as u32;
            let inst = read_instruction(data, offset);
            write_instruction(data, offset, patch_imm12(inst, lo12));
        }
        AArch64RelocationType::TlsdescCall => {
            // Marker relocation for BLR in TLS descriptor sequence.
            // No instruction patching needed.
        }

        // === Dynamic relocations ===
        AArch64RelocationType::Copy => {
            // Dynamic linker handles this at load time.
        }
        AArch64RelocationType::GlobDat => {
            write_u64(data, offset, sa as u64);
        }
        AArch64RelocationType::JumpSlot => {
            write_u64(data, offset, sa as u64);
        }
        AArch64RelocationType::Relative => {
            write_u64(data, offset, sa as u64);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// AArch64RelocationHandler -- RelocationHandler trait implementation
// ---------------------------------------------------------------------------

/// Architecture-specific relocation handler for AArch64.
///
/// Implements the [`RelocationHandler`] trait so the common linker framework
/// can dispatch AArch64-specific relocation classification, naming, sizing,
/// and application.
pub struct AArch64RelocationHandler;

impl Default for AArch64RelocationHandler {
    fn default() -> Self {
        Self
    }
}

impl AArch64RelocationHandler {
    /// Create a new handler instance.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl RelocationHandler for AArch64RelocationHandler {
    /// Classify a raw `r_type` into a [`RelocCategory`].
    fn classify(&self, rel_type: u32) -> RelocCategory {
        match AArch64RelocationType::from_raw(rel_type) {
            Some(rt) => rt.category(),
            None => RelocCategory::Other,
        }
    }

    /// Return a printable name for the relocation type.
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match AArch64RelocationType::from_raw(rel_type) {
            Some(rt) => rt.name(),
            None => "R_AARCH64_UNKNOWN",
        }
    }

    /// Return the size of the relocation patch in bytes.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        match AArch64RelocationType::from_raw(rel_type) {
            Some(rt) => match rt {
                AArch64RelocationType::Abs64
                | AArch64RelocationType::Prel64
                | AArch64RelocationType::GlobDat
                | AArch64RelocationType::JumpSlot
                | AArch64RelocationType::Relative => 8,
                AArch64RelocationType::Abs16 | AArch64RelocationType::Prel16 => 2,
                _ => 4,
            },
            None => 0,
        }
    }

    /// Apply a resolved relocation to section data.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let reloc_type = match AArch64RelocationType::from_raw(rel.rel_type) {
            Some(r) => r,
            None => {
                return Err(RelocationError::UnsupportedType {
                    rel_type: rel.rel_type,
                    target: Target::AArch64,
                });
            }
        };

        apply_relocation(
            section_data,
            rel.patch_offset as usize,
            reloc_type,
            rel.symbol_value,
            rel.addend,
            rel.patch_address,
        )
        .map_err(|_msg| RelocationError::Overflow {
            reloc_name: reloc_type.name().to_string(),
            value: (rel.symbol_value as i128) + (rel.addend as i128),
            bit_width: match reloc_type {
                AArch64RelocationType::Abs16 | AArch64RelocationType::Prel16 => 16,
                AArch64RelocationType::Abs32 | AArch64RelocationType::Prel32 => 32,
                AArch64RelocationType::AdrPrelPgHi21
                | AArch64RelocationType::AdrGotPage
                | AArch64RelocationType::TlsdescAdrPage21 => 21,
                AArch64RelocationType::Call26 | AArch64RelocationType::Jump26 => 26,
                AArch64RelocationType::Condbr19 => 19,
                AArch64RelocationType::Tstbr14 => 14,
                AArch64RelocationType::MovwUabsG0
                | AArch64RelocationType::MovwUabsG1
                | AArch64RelocationType::MovwUabsG2 => 16,
                _ => 64,
            },
            location: format!("0x{:x}", rel.patch_address),
        })
    }

    /// Returns `true` if this relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool {
        match AArch64RelocationType::from_raw(rel_type) {
            Some(rt) => rt.needs_got_entry(),
            None => false,
        }
    }

    /// Returns `true` if this relocation type requires a PLT stub.
    fn needs_plt(&self, rel_type: u32) -> bool {
        match AArch64RelocationType::from_raw(rel_type) {
            Some(rt) => rt.needs_plt_entry(),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// PLT stub generation
// ---------------------------------------------------------------------------

/// Generate a 16-byte PLT stub entry for AArch64.
///
/// The stub loads the target address from a GOT entry and branches to it:
/// ```text
/// adrp  x16, GOT_ENTRY_PAGE
/// ldr   x17, [x16, #:lo12:GOT_ENTRY]
/// add   x16, x16, #:lo12:GOT_ENTRY
/// br    x17
/// ```
pub fn generate_plt_stub(got_entry_addr: u64, plt_addr: u64) -> [u8; 16] {
    let page_got = (got_entry_addr as i64) & !0xFFF;
    let page_plt = (plt_addr as i64) & !0xFFF;
    let page_off = ((page_got - page_plt) >> 12) as i32;
    let lo12 = (got_entry_addr & 0xFFF) as u32;

    // adrp x16, page_off
    let adrp = patch_adr_imm(0x9000_0010, page_off);
    // ldr x17, [x16, #lo12] (8-byte aligned)
    let ldr = patch_imm12(0xF940_0211, lo12 >> 3);
    // add x16, x16, #lo12
    let add = patch_imm12(0x9100_0210, lo12);
    // br x17
    let br: u32 = 0xD61F_0220;

    let mut stub = [0u8; 16];
    stub[0..4].copy_from_slice(&adrp.to_le_bytes());
    stub[4..8].copy_from_slice(&ldr.to_le_bytes());
    stub[8..12].copy_from_slice(&add.to_le_bytes());
    stub[12..16].copy_from_slice(&br.to_le_bytes());
    stub
}

/// Generate a 32-byte PLT0 header entry (calls dynamic linker resolver).
///
/// ```text
/// stp   x16, x30, [sp, #-16]!
/// adrp  x16, GOT+16
/// ldr   x17, [x16, #:lo12:GOT+16]
/// add   x16, x16, #:lo12:GOT+16
/// br    x17
/// nop; nop; nop
/// ```
pub fn generate_plt0(got_plt_addr: u64, plt0_addr: u64) -> [u8; 32] {
    let got2_addr = got_plt_addr + 16;
    let page_got2 = (got2_addr as i64) & !0xFFF;
    let page_plt0 = (plt0_addr as i64) & !0xFFF;
    let page_off = ((page_got2 - page_plt0) >> 12) as i32;
    let lo12 = (got2_addr & 0xFFF) as u32;

    let stp: u32 = 0xA9BF_7BF0;
    let adrp = patch_adr_imm(0x9000_0010, page_off);
    let ldr = patch_imm12(0xF940_0211, lo12 >> 3);
    let add = patch_imm12(0x9100_0210, lo12);
    let br: u32 = 0xD61F_0220;
    let nop: u32 = 0xD503_201F;

    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&stp.to_le_bytes());
    header[4..8].copy_from_slice(&adrp.to_le_bytes());
    header[8..12].copy_from_slice(&ldr.to_le_bytes());
    header[12..16].copy_from_slice(&add.to_le_bytes());
    header[16..20].copy_from_slice(&br.to_le_bytes());
    header[20..24].copy_from_slice(&nop.to_le_bytes());
    header[24..28].copy_from_slice(&nop.to_le_bytes());
    header[28..32].copy_from_slice(&nop.to_le_bytes());
    header
}

// ---------------------------------------------------------------------------
// AArch64AssemblyReloc
// ---------------------------------------------------------------------------

/// A relocation entry generated during the assembly phase.
///
/// When the assembler encounters an instruction that references an unresolved
/// symbol, it creates one of these entries. The linker later resolves the
/// symbol and applies the relocation via [`apply_relocation`].
#[derive(Debug, Clone)]
pub struct AArch64AssemblyReloc {
    /// Byte offset within the section where the instruction to patch resides.
    pub offset: usize,
    /// The AArch64 ELF relocation type.
    pub reloc_type: AArch64RelocationType,
    /// The name of the target symbol.
    pub symbol: String,
    /// Addend value (typically 0 for most AArch64 RELA relocations).
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_raw_roundtrip() {
        let cases: &[(u32, AArch64RelocationType)] = &[
            (0, AArch64RelocationType::None),
            (257, AArch64RelocationType::Abs64),
            (258, AArch64RelocationType::Abs32),
            (259, AArch64RelocationType::Abs16),
            (260, AArch64RelocationType::Prel64),
            (261, AArch64RelocationType::Prel32),
            (262, AArch64RelocationType::Prel16),
            (263, AArch64RelocationType::MovwUabsG0),
            (264, AArch64RelocationType::MovwUabsG0Nc),
            (265, AArch64RelocationType::MovwUabsG1),
            (266, AArch64RelocationType::MovwUabsG1Nc),
            (267, AArch64RelocationType::MovwUabsG2),
            (268, AArch64RelocationType::MovwUabsG2Nc),
            (269, AArch64RelocationType::MovwUabsG3),
            (275, AArch64RelocationType::AdrPrelPgHi21),
            (276, AArch64RelocationType::AdrPrelPgHi21Nc),
            (277, AArch64RelocationType::AddAbsLo12Nc),
            (278, AArch64RelocationType::Ldst8AbsLo12Nc),
            (279, AArch64RelocationType::Tstbr14),
            (280, AArch64RelocationType::Condbr19),
            (282, AArch64RelocationType::Jump26),
            (283, AArch64RelocationType::Call26),
            (284, AArch64RelocationType::Ldst16AbsLo12Nc),
            (285, AArch64RelocationType::Ldst32AbsLo12Nc),
            (286, AArch64RelocationType::Ldst64AbsLo12Nc),
            (299, AArch64RelocationType::Ldst128AbsLo12Nc),
            (311, AArch64RelocationType::AdrGotPage),
            (312, AArch64RelocationType::Ld64GotLo12Nc),
            (566, AArch64RelocationType::TlsdescAdrPage21),
            (567, AArch64RelocationType::TlsdescLd64Lo12),
            (568, AArch64RelocationType::TlsdescAddLo12),
            (569, AArch64RelocationType::TlsdescCall),
            (1024, AArch64RelocationType::Copy),
            (1025, AArch64RelocationType::GlobDat),
            (1026, AArch64RelocationType::JumpSlot),
            (1027, AArch64RelocationType::Relative),
        ];
        for &(raw, expected) in cases {
            let rt = AArch64RelocationType::from_raw(raw).unwrap();
            assert_eq!(rt, expected, "from_raw({}) mismatch", raw);
            assert_eq!(rt.to_raw(), raw, "to_raw() mismatch for {:?}", expected);
        }
        assert!(AArch64RelocationType::from_raw(9999).is_none());
        assert!(AArch64RelocationType::from_raw(256).is_none());
    }

    #[test]
    fn test_read_write_instruction() {
        let mut buf = [0u8; 8];
        write_instruction(&mut buf, 0, 0xD503_201F);
        assert_eq!(read_instruction(&buf, 0), 0xD503_201F);
        assert_eq!(buf[0], 0x1F);
        assert_eq!(buf[1], 0x20);
        assert_eq!(buf[2], 0x03);
        assert_eq!(buf[3], 0xD5);
    }

    #[test]
    fn test_read_write_u64() {
        let mut buf = [0u8; 16];
        write_u64(&mut buf, 0, 0x0011_2233_4455_6677);
        assert_eq!(read_u64(&buf, 0), 0x0011_2233_4455_6677);
    }

    #[test]
    fn test_overflow_helpers() {
        assert!(check_signed_overflow(0, 1));
        assert!(check_signed_overflow(-1, 1));
        assert!(!check_signed_overflow(1, 1));
        assert!(check_signed_overflow(127, 8));
        assert!(check_signed_overflow(-128, 8));
        assert!(!check_signed_overflow(128, 8));
        assert!(!check_signed_overflow(-129, 8));
        assert!(check_unsigned_overflow(255, 8));
        assert!(!check_unsigned_overflow(256, 8));
        assert!(check_unsigned_overflow(0, 1));
        assert!(!check_unsigned_overflow(2, 1));
    }

    #[test]
    fn test_patch_imm26() {
        let inst = 0x9400_0000u32;
        let patched = patch_imm26(inst, 0x100);
        assert_eq!(patched & 0x03FF_FFFF, 0x100);
        assert_eq!(patched & !0x03FF_FFFFu32, 0x9400_0000);
    }

    #[test]
    fn test_patch_imm19() {
        let inst = 0x5400_0000u32;
        let patched = patch_imm19(inst, 0x42);
        assert_eq!((patched >> 5) & 0x7FFFF, 0x42);
    }

    #[test]
    fn test_patch_imm14() {
        let inst = 0x3600_0000u32;
        let patched = patch_imm14(inst, 0x10);
        assert_eq!((patched >> 5) & 0x3FFF, 0x10);
    }

    #[test]
    fn test_patch_adr_imm() {
        let inst = 0x9000_0000u32;
        let patched = patch_adr_imm(inst, 5);
        let immlo = (patched >> 29) & 0x3;
        let immhi = (patched >> 5) & 0x7FFFF;
        let val = ((immhi << 2) | immlo) as i32;
        assert_eq!(val, 5);
    }

    #[test]
    fn test_patch_imm12() {
        let inst = 0x9100_0000u32;
        let patched = patch_imm12(inst, 0x234);
        assert_eq!((patched >> 10) & 0xFFF, 0x234);
    }

    #[test]
    fn test_patch_imm16() {
        let inst = 0xD280_0000u32;
        let patched = patch_imm16(inst, 0xABCD);
        assert_eq!((patched >> 5) & 0xFFFF, 0xABCD);
    }

    #[test]
    fn test_apply_abs64() {
        let mut data = [0u8; 8];
        apply_relocation(&mut data, 0, AArch64RelocationType::Abs64, 0x1000, 0x10, 0).unwrap();
        assert_eq!(read_u64(&data, 0), 0x1010);
    }

    #[test]
    fn test_apply_abs32_overflow() {
        let mut data = [0u8; 4];
        let result = apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Abs32,
            0x1_0000_0000,
            0,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_call26_in_range() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x9400_0000);
        apply_relocation(&mut data, 0, AArch64RelocationType::Call26, 0x1000, 0, 0x0).unwrap();
        let inst = read_instruction(&data, 0);
        assert_eq!(inst & 0x03FF_FFFF, 0x400);
    }

    #[test]
    fn test_apply_call26_overflow() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x9400_0000);
        let result = apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Call26,
            0x1_0000_0000,
            0,
            0x0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_adrp() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x9000_0000);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::AdrPrelPgHi21,
            0x2000,
            0,
            0x1000,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let immlo = (inst >> 29) & 0x3;
        let immhi = (inst >> 5) & 0x7FFFF;
        let reconstructed = ((immhi << 2) | immlo) as i32;
        assert_eq!(reconstructed, 1);
    }

    #[test]
    fn test_apply_add_lo12() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x9100_0020);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::AddAbsLo12Nc,
            0x1234,
            0,
            0,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm12 = (inst >> 10) & 0xFFF;
        assert_eq!(imm12, 0x234);
    }

    #[test]
    fn test_ldst_scaling() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0xF940_0020);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Ldst64AbsLo12Nc,
            0x1018,
            0,
            0,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm12 = (inst >> 10) & 0xFFF;
        assert_eq!(imm12, 3);
    }

    #[test]
    fn test_prel32() {
        let mut data = [0u8; 4];
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Prel32,
            0x2000,
            0,
            0x1000,
        )
        .unwrap();
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(val, 0x1000);
    }

    #[test]
    fn test_condbr19() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x5400_0000);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Condbr19,
            0x1100,
            0,
            0x1000,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm19 = (inst >> 5) & 0x7FFFF;
        assert_eq!(imm19, 0x40);
    }

    #[test]
    fn test_tstbr14() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0x3600_0000);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::Tstbr14,
            0x1020,
            0,
            0x1000,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm14 = (inst >> 5) & 0x3FFF;
        assert_eq!(imm14, 0x8);
    }

    #[test]
    fn test_movw_g0_nc() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0xD280_0000);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::MovwUabsG0Nc,
            0x1234_5678_9ABC_DEF0,
            0,
            0,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm16 = (inst >> 5) & 0xFFFF;
        assert_eq!(imm16, 0xDEF0);
    }

    #[test]
    fn test_movw_g1_nc() {
        let mut data = [0u8; 4];
        write_instruction(&mut data, 0, 0xF2A0_0000);
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::MovwUabsG1Nc,
            0x1234_5678_9ABC_DEF0,
            0,
            0,
        )
        .unwrap();
        let inst = read_instruction(&data, 0);
        let imm16 = (inst >> 5) & 0xFFFF;
        assert_eq!(imm16, 0x9ABC);
    }

    #[test]
    fn test_plt_stub_size() {
        let stub = generate_plt_stub(0x2000, 0x1000);
        assert_eq!(stub.len(), 16);
    }

    #[test]
    fn test_plt0_size() {
        let header = generate_plt0(0x3000, 0x1000);
        assert_eq!(header.len(), 32);
    }

    #[test]
    fn test_category_classification() {
        assert_eq!(
            AArch64RelocationType::Abs64.category(),
            RelocCategory::Absolute
        );
        assert_eq!(
            AArch64RelocationType::Prel32.category(),
            RelocCategory::PcRelative
        );
        assert_eq!(
            AArch64RelocationType::Call26.category(),
            RelocCategory::PcRelative
        );
        assert_eq!(
            AArch64RelocationType::AdrGotPage.category(),
            RelocCategory::GotRelative
        );
        assert_eq!(
            AArch64RelocationType::GlobDat.category(),
            RelocCategory::GotEntry
        );
        assert_eq!(
            AArch64RelocationType::JumpSlot.category(),
            RelocCategory::Plt
        );
        assert_eq!(
            AArch64RelocationType::TlsdescAdrPage21.category(),
            RelocCategory::Tls
        );
        assert_eq!(
            AArch64RelocationType::AddAbsLo12Nc.category(),
            RelocCategory::SectionRelative
        );
        assert_eq!(
            AArch64RelocationType::MovwUabsG3.category(),
            RelocCategory::Absolute
        );
        assert_eq!(
            AArch64RelocationType::Relative.category(),
            RelocCategory::Absolute
        );
        assert_eq!(AArch64RelocationType::Copy.category(), RelocCategory::Other);
    }

    #[test]
    fn test_needs_got_plt() {
        assert!(AArch64RelocationType::AdrGotPage.needs_got_entry());
        assert!(AArch64RelocationType::Ld64GotLo12Nc.needs_got_entry());
        assert!(AArch64RelocationType::GlobDat.needs_got_entry());
        assert!(AArch64RelocationType::JumpSlot.needs_got_entry());
        assert!(!AArch64RelocationType::Call26.needs_got_entry());
        assert!(!AArch64RelocationType::Abs64.needs_got_entry());

        assert!(AArch64RelocationType::Call26.needs_plt_entry());
        assert!(AArch64RelocationType::Jump26.needs_plt_entry());
        assert!(AArch64RelocationType::JumpSlot.needs_plt_entry());
        assert!(!AArch64RelocationType::Abs64.needs_plt_entry());
        assert!(!AArch64RelocationType::AdrGotPage.needs_plt_entry());
    }

    #[test]
    fn test_handler_classify() {
        let h = AArch64RelocationHandler::new();
        assert_eq!(h.classify(283), RelocCategory::PcRelative);
        assert_eq!(h.classify(257), RelocCategory::Absolute);
        assert_eq!(h.classify(311), RelocCategory::GotRelative);
        assert_eq!(h.classify(9999), RelocCategory::Other);
    }

    #[test]
    fn test_handler_reloc_name() {
        let h = AArch64RelocationHandler::new();
        assert_eq!(h.reloc_name(283), "R_AARCH64_CALL26");
        assert_eq!(h.reloc_name(257), "R_AARCH64_ABS64");
        assert_eq!(h.reloc_name(9999), "R_AARCH64_UNKNOWN");
    }

    #[test]
    fn test_handler_reloc_size() {
        let h = AArch64RelocationHandler::new();
        assert_eq!(h.reloc_size(257), 8);
        assert_eq!(h.reloc_size(260), 8);
        assert_eq!(h.reloc_size(259), 2);
        assert_eq!(h.reloc_size(283), 4);
        assert_eq!(h.reloc_size(1025), 8);
        assert_eq!(h.reloc_size(9999), 0);
    }

    #[test]
    fn test_handler_needs_got_plt() {
        let h = AArch64RelocationHandler::new();
        assert!(h.needs_got(311));
        assert!(h.needs_got(312));
        assert!(!h.needs_got(283));
        assert!(h.needs_plt(283));
        assert!(h.needs_plt(282));
        assert!(!h.needs_plt(257));
    }

    #[test]
    fn test_assembly_reloc_struct() {
        let reloc = AArch64AssemblyReloc {
            offset: 0x100,
            reloc_type: AArch64RelocationType::Call26,
            symbol: "printf".to_string(),
            addend: 0,
        };
        assert_eq!(reloc.offset, 0x100);
        assert_eq!(reloc.reloc_type, AArch64RelocationType::Call26);
        assert_eq!(reloc.symbol, "printf");
        assert_eq!(reloc.addend, 0);
    }

    #[test]
    fn test_apply_none_is_noop() {
        let mut data = [0xAA; 8];
        apply_relocation(&mut data, 0, AArch64RelocationType::None, 0x1000, 0, 0).unwrap();
        assert!(data.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_apply_glob_dat() {
        let mut data = [0u8; 8];
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::GlobDat,
            0xDEAD_BEEF,
            0,
            0,
        )
        .unwrap();
        assert_eq!(read_u64(&data, 0), 0xDEAD_BEEF);
    }

    #[test]
    fn test_apply_jump_slot() {
        let mut data = [0u8; 8];
        apply_relocation(
            &mut data,
            0,
            AArch64RelocationType::JumpSlot,
            0xCAFE_BABE,
            4,
            0,
        )
        .unwrap();
        assert_eq!(read_u64(&data, 0), 0xCAFE_BAC2);
    }

    #[test]
    fn test_plt_stub_instructions() {
        let stub = generate_plt_stub(0x4000, 0x1000);
        // Verify first instruction is ADRP (opcode check: bit 31=1, bits[28:24]=10000)
        let inst0 = u32::from_le_bytes([stub[0], stub[1], stub[2], stub[3]]);
        assert_eq!(inst0 & 0x9F00_0000, 0x9000_0000); // ADRP
                                                      // Last instruction is BR X17
        let inst3 = u32::from_le_bytes([stub[12], stub[13], stub[14], stub[15]]);
        assert_eq!(inst3, 0xD61F_0220);
    }

    #[test]
    fn test_plt0_header_instructions() {
        let header = generate_plt0(0x5000, 0x2000);
        // First instruction is STP
        let inst0 = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        assert_eq!(inst0, 0xA9BF_7BF0);
        // Last three instructions are NOP
        for i in [20, 24, 28] {
            let inst = u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
            assert_eq!(inst, 0xD503_201F);
        }
    }
}
