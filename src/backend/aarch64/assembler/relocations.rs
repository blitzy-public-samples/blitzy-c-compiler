//! AArch64 ELF relocation type definitions for the BCC built-in assembler and linker.
//!
//! Defines the [`AArch64RelocationType`] enum with all standard ELF relocation types
//! required for static and PIC code generation on AArch64. Each variant maps to a
//! specific ELF relocation number and carries metadata about PC-relativity, field
//! size, bit width, and scaling behaviour.
//!
//! # Relocation Categories
//!
//! | Category | Variants | Description |
//! |----------|----------|-------------|
//! | Absolute data | ABS64, ABS32, ABS16 | Direct symbol address |
//! | PC-relative data | PREL64, PREL32, PREL16 | Signed offset from relocation site |
//! | Page-relative | ADR_PREL_PG_HI21*, ADD_ABS_LO12_NC | ADRP+ADD pairs |
//! | Load/store low-12 | LDST{8,16,32,64,128}_ABS_LO12_NC | Scaled page offsets |
//! | Branch | CALL26, JUMP26, CONDBR19, TSTBR14 | Instruction-embedded offsets |
//! | GOT | ADR_GOT_PAGE, LD64_GOT_LO12_NC | PIC global access |
//! | TLS | TLSGD_*, TLSLE_*, TLSIE_* | Thread-local storage |
//! | Dynamic | GLOB_DAT, JUMP_SLOT, RELATIVE, COPY | Runtime linker entries |
//!
//! # Standalone Backend
//!
//! Per Section 0.7.7, all relocation handling is internal to BCC — no external
//! linker or assembler is invoked. This module provides the complete relocation
//! application logic consumed by the AArch64 assembler and linker backends.
//!
//! # Pipeline Position
//!
//! ```text
//! AArch64 Codegen → AArch64 Assembler → [relocations.rs] → AArch64 Linker → ELF
//! ```

use std::fmt;

use crate::backend::linker_common::relocation::RelocationEntry;
use crate::backend::traits::RelocationType;

// ===========================================================================
// AArch64RelocationType — comprehensive AArch64 ELF relocation type enum
// ===========================================================================

/// AArch64 ELF relocation types for the built-in assembler and linker.
///
/// Every variant maps 1:1 to a relocation type defined in the AArch64 ELF
/// specification. The enum provides:
/// - ELF numeric value conversion ([`to_elf_value`](Self::to_elf_value))
/// - Property queries (PC-relative, GOT, TLS, dynamic, etc.)
/// - Instruction patching ([`apply`](Self::apply)) and data patching
///   ([`apply_data`](Self::apply_data))
/// - Range validation ([`check_range`](Self::check_range))
/// - Conversion to the architecture-agnostic linker types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum AArch64RelocationType {
    // ----- Absolute Data Relocations -----

    /// `S + A` — 64-bit absolute address.
    R_AARCH64_ABS64,
    /// `S + A` — 32-bit absolute address (truncated, overflow-checked).
    R_AARCH64_ABS32,
    /// `S + A` — 16-bit absolute address (truncated, overflow-checked).
    R_AARCH64_ABS16,

    // ----- PC-Relative Data Relocations -----

    /// `S + A - P` — 64-bit PC-relative offset.
    R_AARCH64_PREL64,
    /// `S + A - P` — 32-bit PC-relative offset (overflow-checked).
    R_AARCH64_PREL32,
    /// `S + A - P` — 16-bit PC-relative offset (overflow-checked).
    R_AARCH64_PREL16,

    // ----- Group Relocations (page-relative addressing for ADRP+ADD/LDR pairs) -----

    /// `Page(S + A) - Page(P)` — ADRP high 21-bit page offset.
    R_AARCH64_ADR_PREL_PG_HI21,
    /// Same as [`R_AARCH64_ADR_PREL_PG_HI21`] without overflow check.
    R_AARCH64_ADR_PREL_PG_HI21_NC,
    /// `(S + A) & 0xFFF` — ADD low 12-bit page offset (no overflow check).
    R_AARCH64_ADD_ABS_LO12_NC,
    /// `S + A - P` — ADR 21-bit PC-relative offset.
    R_AARCH64_ADR_PREL_LO21,

    // ----- Load/Store Low-12-Bit Relocations (scaled by access size) -----

    /// `(S + A) & 0xFFF` — byte load/store, no scaling.
    R_AARCH64_LDST8_ABS_LO12_NC,
    /// `((S + A) & 0xFFF) >> 1` — halfword load/store, scale by 2.
    R_AARCH64_LDST16_ABS_LO12_NC,
    /// `((S + A) & 0xFFF) >> 2` — word load/store, scale by 4.
    R_AARCH64_LDST32_ABS_LO12_NC,
    /// `((S + A) & 0xFFF) >> 3` — doubleword load/store, scale by 8.
    R_AARCH64_LDST64_ABS_LO12_NC,
    /// `((S + A) & 0xFFF) >> 4` — quadword load/store, scale by 16.
    R_AARCH64_LDST128_ABS_LO12_NC,

    // ----- Branch Relocations -----

    /// `S + A - P` — BL instruction, 26-bit offset (±128 MiB).
    R_AARCH64_CALL26,
    /// `S + A - P` — B instruction, 26-bit offset (±128 MiB).
    R_AARCH64_JUMP26,
    /// `S + A - P` — B.cond / CBZ / CBNZ, 19-bit offset (±1 MiB).
    R_AARCH64_CONDBR19,
    /// `S + A - P` — TBZ / TBNZ, 14-bit offset (±32 KiB).
    R_AARCH64_TSTBR14,

    // ----- GOT-Relative Relocations (for PIC/shared libraries) -----

    /// `Page(G(S)) - Page(P)` — ADRP to GOT entry page.
    R_AARCH64_ADR_GOT_PAGE,
    /// `G(S) & 0xFFF` — LDR from GOT entry, low 12 bits scaled by 8.
    R_AARCH64_LD64_GOT_LO12_NC,

    // ----- TLS Relocations (Thread-Local Storage) -----

    /// `Page(G(TLSIDX(S))) - Page(P)` — General Dynamic TLS, ADRP.
    R_AARCH64_TLSGD_ADR_PAGE21,
    /// `G(TLSIDX(S)) & 0xFFF` — General Dynamic TLS, ADD.
    R_AARCH64_TLSGD_ADD_LO12_NC,
    /// `TPREL(S + A) >> 12` — Local Exec TLS, high 12 bits.
    R_AARCH64_TLSLE_ADD_TPREL_HI12,
    /// `TPREL(S + A) & 0xFFF` — Local Exec TLS, low 12 bits (no check).
    R_AARCH64_TLSLE_ADD_TPREL_LO12_NC,
    /// `Page(G(TPREL(S + A))) - Page(P)` — Initial Exec TLS, ADRP.
    R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21,
    /// `G(TPREL(S + A)) & 0xFFF` — Initial Exec TLS, LDR.
    R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC,

    // ----- Dynamic Relocations (for runtime linker) -----

    /// `S + A` — GOT entry filled by runtime linker.
    R_AARCH64_GLOB_DAT,
    /// `S + A` — PLT entry filled by runtime linker.
    R_AARCH64_JUMP_SLOT,
    /// `B + A` — base-relative fixup for PIE/PIC.
    R_AARCH64_RELATIVE,
    /// Copy symbol data at runtime (for external variables).
    R_AARCH64_COPY,
    /// Module ID for TLS (General Dynamic model).
    R_AARCH64_TLS_DTPMOD64,
    /// TLS offset from DTV (General Dynamic model).
    R_AARCH64_TLS_DTPREL64,
    /// TLS offset from thread pointer (Local Exec / Initial Exec).
    R_AARCH64_TLS_TPREL64,
}

// ===========================================================================
// Bitmask constants for AArch64 instruction field encoding
// ===========================================================================

/// Bits [25:0] in B/BL instructions — 26-bit signed immediate (word-aligned).
const IMM26_MASK: u32 = 0x03FF_FFFF;
/// Bits [23:5] in B.cond/CBZ/CBNZ — 19-bit signed immediate.
const IMM19_MASK: u32 = 0x00FF_FFE0;
/// Bits [18:5] in TBZ/TBNZ — 14-bit signed immediate.
const IMM14_MASK: u32 = 0x0007_FFE0;
/// Combined bits [30:29] (immlo) and [23:5] (immhi) in ADR/ADRP — 21-bit split immediate.
const ADR_IMM_MASK: u32 = 0x60FF_FFE0;
/// Bits [21:10] in ADD/LDR/STR immediate — 12-bit unsigned immediate.
const IMM12_MASK: u32 = 0x003F_FC00;

impl AArch64RelocationType {
    // -----------------------------------------------------------------------
    // Canonical name
    // -----------------------------------------------------------------------

    /// Returns the canonical ELF name of this relocation type.
    ///
    /// The returned string matches the ELF specification name exactly
    /// (e.g., `"R_AARCH64_CALL26"`). Used by [`Display`] and diagnostic messages.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::R_AARCH64_ABS64 => "R_AARCH64_ABS64",
            Self::R_AARCH64_ABS32 => "R_AARCH64_ABS32",
            Self::R_AARCH64_ABS16 => "R_AARCH64_ABS16",
            Self::R_AARCH64_PREL64 => "R_AARCH64_PREL64",
            Self::R_AARCH64_PREL32 => "R_AARCH64_PREL32",
            Self::R_AARCH64_PREL16 => "R_AARCH64_PREL16",
            Self::R_AARCH64_ADR_PREL_PG_HI21 => "R_AARCH64_ADR_PREL_PG_HI21",
            Self::R_AARCH64_ADR_PREL_PG_HI21_NC => "R_AARCH64_ADR_PREL_PG_HI21_NC",
            Self::R_AARCH64_ADD_ABS_LO12_NC => "R_AARCH64_ADD_ABS_LO12_NC",
            Self::R_AARCH64_ADR_PREL_LO21 => "R_AARCH64_ADR_PREL_LO21",
            Self::R_AARCH64_LDST8_ABS_LO12_NC => "R_AARCH64_LDST8_ABS_LO12_NC",
            Self::R_AARCH64_LDST16_ABS_LO12_NC => "R_AARCH64_LDST16_ABS_LO12_NC",
            Self::R_AARCH64_LDST32_ABS_LO12_NC => "R_AARCH64_LDST32_ABS_LO12_NC",
            Self::R_AARCH64_LDST64_ABS_LO12_NC => "R_AARCH64_LDST64_ABS_LO12_NC",
            Self::R_AARCH64_LDST128_ABS_LO12_NC => "R_AARCH64_LDST128_ABS_LO12_NC",
            Self::R_AARCH64_CALL26 => "R_AARCH64_CALL26",
            Self::R_AARCH64_JUMP26 => "R_AARCH64_JUMP26",
            Self::R_AARCH64_CONDBR19 => "R_AARCH64_CONDBR19",
            Self::R_AARCH64_TSTBR14 => "R_AARCH64_TSTBR14",
            Self::R_AARCH64_ADR_GOT_PAGE => "R_AARCH64_ADR_GOT_PAGE",
            Self::R_AARCH64_LD64_GOT_LO12_NC => "R_AARCH64_LD64_GOT_LO12_NC",
            Self::R_AARCH64_TLSGD_ADR_PAGE21 => "R_AARCH64_TLSGD_ADR_PAGE21",
            Self::R_AARCH64_TLSGD_ADD_LO12_NC => "R_AARCH64_TLSGD_ADD_LO12_NC",
            Self::R_AARCH64_TLSLE_ADD_TPREL_HI12 => "R_AARCH64_TLSLE_ADD_TPREL_HI12",
            Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC",
            Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => "R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21",
            Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => {
                "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC"
            }
            Self::R_AARCH64_GLOB_DAT => "R_AARCH64_GLOB_DAT",
            Self::R_AARCH64_JUMP_SLOT => "R_AARCH64_JUMP_SLOT",
            Self::R_AARCH64_RELATIVE => "R_AARCH64_RELATIVE",
            Self::R_AARCH64_COPY => "R_AARCH64_COPY",
            Self::R_AARCH64_TLS_DTPMOD64 => "R_AARCH64_TLS_DTPMOD64",
            Self::R_AARCH64_TLS_DTPREL64 => "R_AARCH64_TLS_DTPREL64",
            Self::R_AARCH64_TLS_TPREL64 => "R_AARCH64_TLS_TPREL64",
        }
    }

    // -----------------------------------------------------------------------
    // ELF numeric value mapping
    // -----------------------------------------------------------------------

    /// Returns the ELF relocation type number as defined in the AArch64 ELF spec.
    ///
    /// These values are written into the `r_info` field of `Elf64_Rela` entries
    /// in relocatable object files (`.o`) and shared objects.
    #[inline]
    pub fn to_elf_value(&self) -> u32 {
        match self {
            Self::R_AARCH64_ABS64 => 257,
            Self::R_AARCH64_ABS32 => 258,
            Self::R_AARCH64_ABS16 => 259,
            Self::R_AARCH64_PREL64 => 260,
            Self::R_AARCH64_PREL32 => 261,
            Self::R_AARCH64_PREL16 => 262,
            Self::R_AARCH64_ADR_PREL_LO21 => 274,
            Self::R_AARCH64_ADR_PREL_PG_HI21 => 275,
            Self::R_AARCH64_ADR_PREL_PG_HI21_NC => 276,
            Self::R_AARCH64_ADD_ABS_LO12_NC => 277,
            Self::R_AARCH64_LDST8_ABS_LO12_NC => 278,
            Self::R_AARCH64_TSTBR14 => 279,
            Self::R_AARCH64_CONDBR19 => 280,
            Self::R_AARCH64_JUMP26 => 282,
            Self::R_AARCH64_CALL26 => 283,
            Self::R_AARCH64_LDST16_ABS_LO12_NC => 284,
            Self::R_AARCH64_LDST32_ABS_LO12_NC => 285,
            Self::R_AARCH64_LDST64_ABS_LO12_NC => 286,
            Self::R_AARCH64_LDST128_ABS_LO12_NC => 299,
            Self::R_AARCH64_ADR_GOT_PAGE => 311,
            Self::R_AARCH64_LD64_GOT_LO12_NC => 312,
            Self::R_AARCH64_TLSGD_ADR_PAGE21 => 514,
            Self::R_AARCH64_TLSGD_ADD_LO12_NC => 515,
            Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => 539,
            Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => 540,
            Self::R_AARCH64_TLSLE_ADD_TPREL_HI12 => 549,
            Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => 551,
            Self::R_AARCH64_COPY => 1024,
            Self::R_AARCH64_GLOB_DAT => 1025,
            Self::R_AARCH64_JUMP_SLOT => 1026,
            Self::R_AARCH64_RELATIVE => 1027,
            Self::R_AARCH64_TLS_DTPMOD64 => 1028,
            Self::R_AARCH64_TLS_DTPREL64 => 1029,
            Self::R_AARCH64_TLS_TPREL64 => 1030,
        }
    }

    /// Converts an ELF relocation number back to the corresponding enum variant.
    ///
    /// Returns `None` for unrecognised values. Used when reading relocatable
    /// object files (`.o`) during the linking phase.
    pub fn from_elf_value(value: u32) -> Option<Self> {
        match value {
            257 => Some(Self::R_AARCH64_ABS64),
            258 => Some(Self::R_AARCH64_ABS32),
            259 => Some(Self::R_AARCH64_ABS16),
            260 => Some(Self::R_AARCH64_PREL64),
            261 => Some(Self::R_AARCH64_PREL32),
            262 => Some(Self::R_AARCH64_PREL16),
            274 => Some(Self::R_AARCH64_ADR_PREL_LO21),
            275 => Some(Self::R_AARCH64_ADR_PREL_PG_HI21),
            276 => Some(Self::R_AARCH64_ADR_PREL_PG_HI21_NC),
            277 => Some(Self::R_AARCH64_ADD_ABS_LO12_NC),
            278 => Some(Self::R_AARCH64_LDST8_ABS_LO12_NC),
            279 => Some(Self::R_AARCH64_TSTBR14),
            280 => Some(Self::R_AARCH64_CONDBR19),
            282 => Some(Self::R_AARCH64_JUMP26),
            283 => Some(Self::R_AARCH64_CALL26),
            284 => Some(Self::R_AARCH64_LDST16_ABS_LO12_NC),
            285 => Some(Self::R_AARCH64_LDST32_ABS_LO12_NC),
            286 => Some(Self::R_AARCH64_LDST64_ABS_LO12_NC),
            299 => Some(Self::R_AARCH64_LDST128_ABS_LO12_NC),
            311 => Some(Self::R_AARCH64_ADR_GOT_PAGE),
            312 => Some(Self::R_AARCH64_LD64_GOT_LO12_NC),
            514 => Some(Self::R_AARCH64_TLSGD_ADR_PAGE21),
            515 => Some(Self::R_AARCH64_TLSGD_ADD_LO12_NC),
            539 => Some(Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21),
            540 => Some(Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC),
            549 => Some(Self::R_AARCH64_TLSLE_ADD_TPREL_HI12),
            551 => Some(Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC),
            1024 => Some(Self::R_AARCH64_COPY),
            1025 => Some(Self::R_AARCH64_GLOB_DAT),
            1026 => Some(Self::R_AARCH64_JUMP_SLOT),
            1027 => Some(Self::R_AARCH64_RELATIVE),
            1028 => Some(Self::R_AARCH64_TLS_DTPMOD64),
            1029 => Some(Self::R_AARCH64_TLS_DTPREL64),
            1030 => Some(Self::R_AARCH64_TLS_TPREL64),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Relocation property queries
    // -----------------------------------------------------------------------

    /// Returns `true` if this relocation type computes a PC-relative value.
    ///
    /// PC-relative relocations use the formula: `value = S + A - P`
    /// where S = symbol value, A = addend, P = relocation virtual address.
    #[inline]
    pub fn is_pc_relative(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_PREL64
                | Self::R_AARCH64_PREL32
                | Self::R_AARCH64_PREL16
                | Self::R_AARCH64_ADR_PREL_PG_HI21
                | Self::R_AARCH64_ADR_PREL_PG_HI21_NC
                | Self::R_AARCH64_ADR_PREL_LO21
                | Self::R_AARCH64_CALL26
                | Self::R_AARCH64_JUMP26
                | Self::R_AARCH64_CONDBR19
                | Self::R_AARCH64_TSTBR14
                | Self::R_AARCH64_ADR_GOT_PAGE
                | Self::R_AARCH64_TLSGD_ADR_PAGE21
                | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
        )
    }

    /// Returns the size of the relocation field in bytes.
    ///
    /// - 2 bytes: 16-bit relocations (ABS16, PREL16)
    /// - 4 bytes: 32-bit data and all instruction-embedded relocations
    /// - 8 bytes: 64-bit relocations and all dynamic/TLS data relocations
    #[inline]
    pub fn size(&self) -> u8 {
        match self {
            // 16-bit data relocations
            Self::R_AARCH64_ABS16 | Self::R_AARCH64_PREL16 => 2,

            // 32-bit data relocations
            Self::R_AARCH64_ABS32 | Self::R_AARCH64_PREL32 => 4,

            // All instruction-embedded relocations operate on 4-byte instruction words
            Self::R_AARCH64_CALL26
            | Self::R_AARCH64_JUMP26
            | Self::R_AARCH64_CONDBR19
            | Self::R_AARCH64_TSTBR14
            | Self::R_AARCH64_ADR_PREL_PG_HI21
            | Self::R_AARCH64_ADR_PREL_PG_HI21_NC
            | Self::R_AARCH64_ADD_ABS_LO12_NC
            | Self::R_AARCH64_ADR_PREL_LO21
            | Self::R_AARCH64_LDST8_ABS_LO12_NC
            | Self::R_AARCH64_LDST16_ABS_LO12_NC
            | Self::R_AARCH64_LDST32_ABS_LO12_NC
            | Self::R_AARCH64_LDST64_ABS_LO12_NC
            | Self::R_AARCH64_LDST128_ABS_LO12_NC
            | Self::R_AARCH64_ADR_GOT_PAGE
            | Self::R_AARCH64_LD64_GOT_LO12_NC
            | Self::R_AARCH64_TLSGD_ADR_PAGE21
            | Self::R_AARCH64_TLSGD_ADD_LO12_NC
            | Self::R_AARCH64_TLSLE_ADD_TPREL_HI12
            | Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
            | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
            | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => 4,

            // 64-bit data and dynamic relocations
            Self::R_AARCH64_ABS64
            | Self::R_AARCH64_PREL64
            | Self::R_AARCH64_GLOB_DAT
            | Self::R_AARCH64_JUMP_SLOT
            | Self::R_AARCH64_RELATIVE
            | Self::R_AARCH64_COPY
            | Self::R_AARCH64_TLS_DTPMOD64
            | Self::R_AARCH64_TLS_DTPREL64
            | Self::R_AARCH64_TLS_TPREL64 => 8,
        }
    }

    /// Returns the number of significant bits in the relocated immediate field.
    ///
    /// For instruction-embedded relocations this is the width of the immediate
    /// operand in the A64 encoding. For data relocations it is the full data width.
    #[inline]
    pub fn bit_width(&self) -> u8 {
        match self {
            Self::R_AARCH64_ABS64 | Self::R_AARCH64_PREL64 => 64,
            Self::R_AARCH64_ABS32 | Self::R_AARCH64_PREL32 => 32,
            Self::R_AARCH64_ABS16 | Self::R_AARCH64_PREL16 => 16,
            Self::R_AARCH64_CALL26 | Self::R_AARCH64_JUMP26 => 26,
            Self::R_AARCH64_ADR_PREL_PG_HI21
            | Self::R_AARCH64_ADR_PREL_PG_HI21_NC
            | Self::R_AARCH64_ADR_PREL_LO21
            | Self::R_AARCH64_ADR_GOT_PAGE
            | Self::R_AARCH64_TLSGD_ADR_PAGE21
            | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => 21,
            Self::R_AARCH64_CONDBR19 => 19,
            Self::R_AARCH64_TSTBR14 => 14,
            Self::R_AARCH64_ADD_ABS_LO12_NC
            | Self::R_AARCH64_LDST8_ABS_LO12_NC
            | Self::R_AARCH64_LDST16_ABS_LO12_NC
            | Self::R_AARCH64_LDST32_ABS_LO12_NC
            | Self::R_AARCH64_LDST64_ABS_LO12_NC
            | Self::R_AARCH64_LDST128_ABS_LO12_NC
            | Self::R_AARCH64_LD64_GOT_LO12_NC
            | Self::R_AARCH64_TLSGD_ADD_LO12_NC
            | Self::R_AARCH64_TLSLE_ADD_TPREL_HI12
            | Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
            | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => 12,
            // Dynamic relocations carry a full 64-bit value
            Self::R_AARCH64_GLOB_DAT
            | Self::R_AARCH64_JUMP_SLOT
            | Self::R_AARCH64_RELATIVE
            | Self::R_AARCH64_COPY
            | Self::R_AARCH64_TLS_DTPMOD64
            | Self::R_AARCH64_TLS_DTPREL64
            | Self::R_AARCH64_TLS_TPREL64 => 64,
        }
    }

    /// Returns `true` if this relocation references a GOT (Global Offset Table) entry.
    #[inline]
    pub fn is_got_relative(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_ADR_GOT_PAGE | Self::R_AARCH64_LD64_GOT_LO12_NC
        )
    }

    /// Returns `true` if this is a TLS (Thread-Local Storage) relocation.
    #[inline]
    pub fn is_tls(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_TLSGD_ADR_PAGE21
                | Self::R_AARCH64_TLSGD_ADD_LO12_NC
                | Self::R_AARCH64_TLSLE_ADD_TPREL_HI12
                | Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
                | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
                | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
                | Self::R_AARCH64_TLS_DTPMOD64
                | Self::R_AARCH64_TLS_DTPREL64
                | Self::R_AARCH64_TLS_TPREL64
        )
    }

    /// Returns `true` if this is a dynamic relocation resolved by the runtime linker.
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_GLOB_DAT
                | Self::R_AARCH64_JUMP_SLOT
                | Self::R_AARCH64_RELATIVE
                | Self::R_AARCH64_COPY
                | Self::R_AARCH64_TLS_DTPMOD64
                | Self::R_AARCH64_TLS_DTPREL64
                | Self::R_AARCH64_TLS_TPREL64
        )
    }

    /// Returns `true` if this relocation requires a PLT (Procedure Linkage Table) entry.
    ///
    /// CALL26 and JUMP26 need PLT stubs when the target symbol resides in a
    /// shared library, providing lazy or eager binding through the PLT.
    #[inline]
    pub fn needs_plt(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_CALL26 | Self::R_AARCH64_JUMP26
        )
    }

    /// Returns `true` if this relocation requires a GOT entry.
    ///
    /// GOT entries are allocated for PIC global data access, GOT-indirect
    /// function calls, and Initial Exec TLS model accesses.
    #[inline]
    pub fn needs_got(&self) -> bool {
        matches!(
            self,
            Self::R_AARCH64_ADR_GOT_PAGE
                | Self::R_AARCH64_LD64_GOT_LO12_NC
                | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
                | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
        )
    }

    /// Returns the alignment scale factor for this relocation type.
    ///
    /// Load/store relocations scale the immediate by the access size:
    /// - LDST8: 1 (byte)
    /// - LDST16: 2 (halfword)
    /// - LDST32: 4 (word)
    /// - LDST64: 8 (doubleword)
    /// - LDST128: 16 (quadword)
    /// - Branch relocations: 4 (instruction-width alignment)
    /// - All others: 1
    #[inline]
    pub fn scale_factor(&self) -> u8 {
        match self {
            Self::R_AARCH64_LDST8_ABS_LO12_NC => 1,
            Self::R_AARCH64_LDST16_ABS_LO12_NC => 2,
            Self::R_AARCH64_LDST32_ABS_LO12_NC => 4,
            Self::R_AARCH64_LDST64_ABS_LO12_NC
            | Self::R_AARCH64_LD64_GOT_LO12_NC
            | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => 8,
            Self::R_AARCH64_LDST128_ABS_LO12_NC => 16,
            Self::R_AARCH64_CALL26
            | Self::R_AARCH64_JUMP26
            | Self::R_AARCH64_CONDBR19
            | Self::R_AARCH64_TSTBR14 => 4,
            _ => 1,
        }
    }

    // -----------------------------------------------------------------------
    // Relocation application — instruction words
    // -----------------------------------------------------------------------

    /// Applies this relocation to a 32-bit AArch64 instruction word.
    ///
    /// The `value` parameter is the fully-computed relocation result
    /// (e.g., `S + A - P` for PC-relative types). This function encodes
    /// the value into the appropriate instruction bitfield per the A64
    /// encoding specification.
    ///
    /// # Encoding Details
    ///
    /// | Relocation | Instruction | Bitfield | Encoding |
    /// |-----------|-------------|----------|----------|
    /// | CALL26/JUMP26 | BL/B | [25:0] | `(value >> 2) & 0x3FFFFFF` |
    /// | CONDBR19 | B.cond/CBZ | [23:5] | `((value >> 2) & 0x7FFFF) << 5` |
    /// | TSTBR14 | TBZ/TBNZ | [18:5] | `((value >> 2) & 0x3FFF) << 5` |
    /// | ADR_PREL_PG_HI21* | ADRP | [30:29],[23:5] | split immlo/immhi from `value >> 12` |
    /// | ADR_PREL_LO21 | ADR | [30:29],[23:5] | split immlo/immhi from value |
    /// | ADD_ABS_LO12_NC | ADD | [21:10] | `(value & 0xFFF) << 10` |
    /// | LDST*_ABS_LO12_NC | LDR/STR | [21:10] | `((value & 0xFFF) >> scale_log2) << 10` |
    ///
    /// For data and dynamic relocations that do not modify instruction words,
    /// the instruction is returned unchanged — use [`apply_data`](Self::apply_data)
    /// for those.
    pub fn apply(&self, instruction: u32, value: i64) -> u32 {
        match self {
            // --- Branch relocations: encode (value >> 2) ---
            Self::R_AARCH64_CALL26 | Self::R_AARCH64_JUMP26 => {
                let imm26 = ((value >> 2) as u32) & IMM26_MASK;
                (instruction & !IMM26_MASK) | imm26
            }

            Self::R_AARCH64_CONDBR19 => {
                let imm19 = (((value >> 2) as u32) & 0x7_FFFF) << 5;
                (instruction & !IMM19_MASK) | imm19
            }

            Self::R_AARCH64_TSTBR14 => {
                let imm14 = (((value >> 2) as u32) & 0x3FFF) << 5;
                (instruction & !IMM14_MASK) | imm14
            }

            // --- ADRP: split 21-bit immediate from (value >> 12) ---
            Self::R_AARCH64_ADR_PREL_PG_HI21
            | Self::R_AARCH64_ADR_PREL_PG_HI21_NC
            | Self::R_AARCH64_ADR_GOT_PAGE
            | Self::R_AARCH64_TLSGD_ADR_PAGE21
            | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => {
                let imm = (value >> 12) as u32;
                let immlo = (imm & 0x3) << 29;
                let immhi = ((imm >> 2) & 0x7_FFFF) << 5;
                (instruction & !ADR_IMM_MASK) | immlo | immhi
            }

            // --- ADR: split 21-bit immediate directly from value ---
            Self::R_AARCH64_ADR_PREL_LO21 => {
                let imm = value as u32;
                let immlo = (imm & 0x3) << 29;
                let immhi = ((imm >> 2) & 0x7_FFFF) << 5;
                (instruction & !ADR_IMM_MASK) | immlo | immhi
            }

            // --- ADD immediate: low 12 bits into [21:10] ---
            Self::R_AARCH64_ADD_ABS_LO12_NC
            | Self::R_AARCH64_TLSGD_ADD_LO12_NC
            | Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => {
                let imm12 = ((value as u32) & 0xFFF) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- TLSLE HI12: extract (value >> 12) & 0xFFF into ADD [21:10] ---
            Self::R_AARCH64_TLSLE_ADD_TPREL_HI12 => {
                let imm12 = (((value >> 12) as u32) & 0xFFF) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Load/store byte: no scaling ---
            Self::R_AARCH64_LDST8_ABS_LO12_NC => {
                let imm12 = ((value as u32) & 0xFFF) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Load/store halfword: scale by 2 ---
            Self::R_AARCH64_LDST16_ABS_LO12_NC => {
                let imm12 = (((value as u32) & 0xFFF) >> 1) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Load/store word: scale by 4 ---
            Self::R_AARCH64_LDST32_ABS_LO12_NC => {
                let imm12 = (((value as u32) & 0xFFF) >> 2) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Load/store doubleword: scale by 8 ---
            Self::R_AARCH64_LDST64_ABS_LO12_NC
            | Self::R_AARCH64_LD64_GOT_LO12_NC
            | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => {
                let imm12 = (((value as u32) & 0xFFF) >> 3) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Load/store quadword: scale by 16 ---
            Self::R_AARCH64_LDST128_ABS_LO12_NC => {
                let imm12 = (((value as u32) & 0xFFF) >> 4) << 10;
                (instruction & !IMM12_MASK) | imm12
            }

            // --- Data and dynamic relocations do not modify instruction words ---
            Self::R_AARCH64_ABS64
            | Self::R_AARCH64_ABS32
            | Self::R_AARCH64_ABS16
            | Self::R_AARCH64_PREL64
            | Self::R_AARCH64_PREL32
            | Self::R_AARCH64_PREL16
            | Self::R_AARCH64_GLOB_DAT
            | Self::R_AARCH64_JUMP_SLOT
            | Self::R_AARCH64_RELATIVE
            | Self::R_AARCH64_COPY
            | Self::R_AARCH64_TLS_DTPMOD64
            | Self::R_AARCH64_TLS_DTPREL64
            | Self::R_AARCH64_TLS_TPREL64 => instruction,
        }
    }

    // -----------------------------------------------------------------------
    // Relocation application — data sections
    // -----------------------------------------------------------------------

    /// Applies this relocation to raw data bytes (e.g., `.data`, `.rodata`).
    ///
    /// For data relocations (ABS64, ABS32, ABS16, PREL*, dynamic), this writes
    /// the value at the specified offset in little-endian format. For
    /// instruction-embedded relocations, it reads the 4-byte instruction at
    /// `offset`, applies via [`apply`](Self::apply), and writes back.
    ///
    /// # Panics
    ///
    /// Panics if `offset + self.size()` exceeds `data.len()`.
    pub fn apply_data(&self, data: &mut [u8], offset: usize, value: i64) {
        match self {
            // 64-bit data relocations — write 8 bytes LE
            Self::R_AARCH64_ABS64
            | Self::R_AARCH64_PREL64
            | Self::R_AARCH64_GLOB_DAT
            | Self::R_AARCH64_JUMP_SLOT
            | Self::R_AARCH64_RELATIVE
            | Self::R_AARCH64_COPY
            | Self::R_AARCH64_TLS_DTPMOD64
            | Self::R_AARCH64_TLS_DTPREL64
            | Self::R_AARCH64_TLS_TPREL64 => {
                let bytes = (value as u64).to_le_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }

            // 32-bit data relocations — write 4 bytes LE
            Self::R_AARCH64_ABS32 | Self::R_AARCH64_PREL32 => {
                let bytes = (value as u32).to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }

            // 16-bit data relocations — write 2 bytes LE
            Self::R_AARCH64_ABS16 | Self::R_AARCH64_PREL16 => {
                let bytes = (value as u16).to_le_bytes();
                data[offset..offset + 2].copy_from_slice(&bytes);
            }

            // Instruction-embedded relocations — read, apply, write back
            _ => {
                let instr_bytes: [u8; 4] = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let instr = u32::from_le_bytes(instr_bytes);
                let patched = self.apply(instr, value);
                let result = patched.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&result);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Range validation
    // -----------------------------------------------------------------------

    /// Checks whether `value` fits within the relocation's immediate field.
    ///
    /// Returns `true` if the value can be encoded without overflow. For
    /// relocation types suffixed with `_NC` (no check), this always returns
    /// `true`.
    ///
    /// Branch relocations additionally verify 4-byte alignment of the value
    /// since AArch64 instructions are word-aligned.
    ///
    /// # Range Summary
    ///
    /// | Relocation | Range |
    /// |-----------|-------|
    /// | CALL26/JUMP26 | `[-134217728, +134217724]` (±128 MiB, 4-aligned) |
    /// | CONDBR19 | `[-1048576, +1048572]` (±1 MiB, 4-aligned) |
    /// | TSTBR14 | `[-32768, +32764]` (±32 KiB, 4-aligned) |
    /// | ADR_PREL_PG_HI21 | `[-4294967296, +4294963200]` (±4 GiB pages) |
    /// | ADR_PREL_LO21 | `[-1048576, +1048575]` (±1 MiB) |
    /// | ABS32 | `[0, 4294967295]` |
    /// | PREL32 | `[-2147483648, +2147483647]` |
    /// | ABS16 | `[0, 65535]` |
    /// | PREL16 | `[-32768, +32767]` |
    /// | *_NC / ABS64 / PREL64 / dynamic | always `true` |
    pub fn check_range(&self, value: i64) -> bool {
        match self {
            // ±128 MiB, 4-byte aligned (26-bit signed × 4)
            Self::R_AARCH64_CALL26 | Self::R_AARCH64_JUMP26 => {
                (value & 0x3) == 0 && value >= -134_217_728 && value <= 134_217_724
            }

            // ±1 MiB, 4-byte aligned (19-bit signed × 4)
            Self::R_AARCH64_CONDBR19 => {
                (value & 0x3) == 0 && value >= -1_048_576 && value <= 1_048_572
            }

            // ±32 KiB, 4-byte aligned (14-bit signed × 4)
            Self::R_AARCH64_TSTBR14 => {
                (value & 0x3) == 0 && value >= -32_768 && value <= 32_764
            }

            // ±4 GiB page range (21-bit signed page offset × 4096)
            Self::R_AARCH64_ADR_PREL_PG_HI21 => {
                value >= -4_294_967_296 && value <= 4_294_963_200
            }

            // ±1 MiB (21-bit signed, no shift)
            Self::R_AARCH64_ADR_PREL_LO21 => {
                value >= -1_048_576 && value <= 1_048_575
            }

            // 32-bit unsigned absolute
            Self::R_AARCH64_ABS32 => value >= 0 && value <= 0xFFFF_FFFF,

            // 32-bit signed PC-relative
            Self::R_AARCH64_PREL32 => {
                value >= i32::MIN as i64 && value <= i32::MAX as i64
            }

            // 16-bit unsigned absolute
            Self::R_AARCH64_ABS16 => value >= 0 && value <= 0xFFFF,

            // 16-bit signed PC-relative
            Self::R_AARCH64_PREL16 => {
                value >= i16::MIN as i64 && value <= i16::MAX as i64
            }

            // TLSLE HI12: check that upper bits beyond [23:12] are zero
            Self::R_AARCH64_TLSLE_ADD_TPREL_HI12 => {
                let hi = (value >> 12) & !0xFFF_i64;
                hi == 0 || hi == -1 // allow sign-extended negatives
            }

            // NC (no check) variants — always pass
            Self::R_AARCH64_ADR_PREL_PG_HI21_NC
            | Self::R_AARCH64_ADD_ABS_LO12_NC
            | Self::R_AARCH64_LDST8_ABS_LO12_NC
            | Self::R_AARCH64_LDST16_ABS_LO12_NC
            | Self::R_AARCH64_LDST32_ABS_LO12_NC
            | Self::R_AARCH64_LDST64_ABS_LO12_NC
            | Self::R_AARCH64_LDST128_ABS_LO12_NC
            | Self::R_AARCH64_LD64_GOT_LO12_NC
            | Self::R_AARCH64_TLSGD_ADD_LO12_NC
            | Self::R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
            | Self::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => true,

            // 64-bit relocations — always valid
            Self::R_AARCH64_ABS64 | Self::R_AARCH64_PREL64 => true,

            // GOT/TLS page relocations — same range as ADR_PREL_PG_HI21
            Self::R_AARCH64_ADR_GOT_PAGE
            | Self::R_AARCH64_TLSGD_ADR_PAGE21
            | Self::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => {
                value >= -4_294_967_296 && value <= 4_294_963_200
            }

            // Dynamic relocations — always valid (runtime linker handles them)
            Self::R_AARCH64_GLOB_DAT
            | Self::R_AARCH64_JUMP_SLOT
            | Self::R_AARCH64_RELATIVE
            | Self::R_AARCH64_COPY
            | Self::R_AARCH64_TLS_DTPMOD64
            | Self::R_AARCH64_TLS_DTPREL64
            | Self::R_AARCH64_TLS_TPREL64 => true,
        }
    }

    // -----------------------------------------------------------------------
    // Conversion to architecture-agnostic linker types
    // -----------------------------------------------------------------------

    /// Creates a skeleton [`RelocationEntry`] for the common linker framework.
    ///
    /// The returned entry has `reloc_type` set to this variant's ELF numeric
    /// value, with all other fields zeroed. The caller fills in `offset`,
    /// `symbol_name`, `symbol_value`, `addend`, and `output_section` as needed.
    ///
    /// This bridges the AArch64-specific relocation enum to the
    /// architecture-agnostic [`RelocationEntry`] consumed by
    /// `linker_common::RelocationProcessor`.
    pub fn to_common_relocation(&self) -> RelocationEntry {
        RelocationEntry {
            offset: 0,
            reloc_type: self.to_elf_value(),
            symbol_name: String::new(),
            symbol_value: 0,
            addend: 0,
            output_section: 0,
        }
    }

    /// Converts this relocation type to the generic [`RelocationType`] descriptor.
    ///
    /// This enables the code generation driver to query relocation properties
    /// through the architecture-agnostic [`ArchCodegen::get_relocation_types()`]
    /// interface.
    pub fn to_relocation_type(&self) -> RelocationType {
        RelocationType {
            name: self.name(),
            value: self.to_elf_value(),
            is_pc_relative: self.is_pc_relative(),
            size: self.size(),
        }
    }
}

// ===========================================================================
// Display implementation
// ===========================================================================

impl fmt::Display for AArch64RelocationType {
    /// Formats the relocation type as its canonical ELF name.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let r = AArch64RelocationType::R_AARCH64_CALL26;
    /// assert_eq!(format!("{}", r), "R_AARCH64_CALL26");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
