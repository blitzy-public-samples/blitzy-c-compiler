//! RISC-V 64-bit relocation type definitions for the built-in assembler and linker.
//!
//! This module defines the [`RiscV64RelocationType`] enum covering all standard
//! RISC-V ELF relocation types needed for static and dynamic linking. It provides:
//!
//! - Relocation type enumeration with ELF value mapping
//! - Relocation property queries (PC-relative, relaxable, GOT/PLT requirements)
//! - Instruction bit-field patching for each relocation type
//! - Hi/Lo address splitting with sign-extension compensation
//! - Linker relaxation eligibility classification
//!
//! # RISC-V Instruction Formats
//!
//! RISC-V relocations patch immediate fields that are scattered across
//! instruction bits in format-specific layouts:
//!
//! - **I-type**: `imm[11:0]` at bits `[31:20]`
//! - **S-type**: `imm[11:5]` at bits `[31:25]`, `imm[4:0]` at bits `[11:7]`
//! - **B-type**: `imm[12|10:5]` at bits `[31|30:25]`, `imm[4:1|11]` at bits `[11:8|7]`
//! - **U-type**: `imm[31:12]` at bits `[31:12]`
//! - **J-type**: `imm[20|10:1|11|19:12]` at bits `[31|30:21|20|19:12]`
//!
//! Compressed (RVC) instructions use 16-bit formats with their own
//! immediate field layouts (CB-type for branches, CJ-type for jumps).
//!
//! This is the foundational relocation infrastructure shared between the
//! RISC-V 64 assembler and linker modules.

use std::fmt;

use crate::backend::traits::RelocationType;

// ============================================================================
// RiscV64RelocationType — RISC-V 64 ELF relocation types
// ============================================================================

/// All standard RISC-V ELF relocation types supported by the BCC assembler
/// and linker.
///
/// Each variant's discriminant matches the ELF `r_type` value as defined
/// in the RISC-V ELF psABI specification. The `#[repr(u32)]` attribute
/// ensures the enum layout matches the ELF relocation type encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RiscV64RelocationType {
    // -- Absolute / Dynamic relocations (0–11) --

    /// No relocation; placeholder entry.
    R_RISCV_NONE = 0,
    /// 32-bit absolute address — writes a 32-bit data value.
    R_RISCV_32 = 1,
    /// 64-bit absolute address — writes a 64-bit data value.
    R_RISCV_64 = 2,
    /// Dynamic: relative to load base address.
    R_RISCV_RELATIVE = 3,
    /// Dynamic: copy symbol data at runtime.
    R_RISCV_COPY = 4,
    /// Dynamic: PLT jump slot (GOT entry filled by the runtime linker).
    R_RISCV_JUMP_SLOT = 5,
    /// TLS module ID (32-bit).
    R_RISCV_TLS_DTPMOD32 = 6,
    /// TLS module ID (64-bit).
    R_RISCV_TLS_DTPMOD64 = 7,
    /// TLS offset within module (32-bit).
    R_RISCV_TLS_DTPREL32 = 8,
    /// TLS offset within module (64-bit).
    R_RISCV_TLS_DTPREL64 = 9,
    /// TLS thread-pointer-relative offset (32-bit, Local Exec model).
    R_RISCV_TLS_TPREL32 = 10,
    /// TLS thread-pointer-relative offset (64-bit, Local Exec model).
    R_RISCV_TLS_TPREL64 = 11,

    // -- PC-relative code relocations (16–19) --

    /// B-type branch offset — 13-bit signed, scrambled across B-type fields.
    /// Value must be 2-byte aligned. Range: −4096 to +4094.
    R_RISCV_BRANCH = 16,
    /// J-type jump offset — 21-bit signed, scrambled across J-type fields.
    /// Value must be 2-byte aligned. Range: −1048576 to +1048574.
    R_RISCV_JAL = 17,
    /// AUIPC+JALR pair — 32-bit PC-relative call, relaxable to JAL or C.JAL.
    /// Applies to two consecutive 4-byte instructions (8 bytes total).
    R_RISCV_CALL = 18,
    /// AUIPC+JALR pair through PLT — 32-bit PC-relative call for PIC.
    /// Applies to two consecutive 4-byte instructions (8 bytes total).
    R_RISCV_CALL_PLT = 19,

    // -- GOT-relative / TLS GOT (20–22) --

    /// High 20 bits of GOT entry PC-relative offset (AUIPC targeting GOT).
    R_RISCV_GOT_HI20 = 20,
    /// TLS GOT entry for Initial Exec model.
    R_RISCV_TLS_GOT_HI20 = 21,
    /// TLS GOT entry for General Dynamic model.
    R_RISCV_TLS_GD_HI20 = 22,

    // -- PC-relative AUIPC-based addressing (23–25) --

    /// High 20 bits of PC-relative offset (for AUIPC instruction, U-type).
    R_RISCV_PCREL_HI20 = 23,
    /// Low 12 bits of PC-relative offset (I-type: loads, ADDI).
    R_RISCV_PCREL_LO12_I = 24,
    /// Low 12 bits of PC-relative offset (S-type: stores).
    R_RISCV_PCREL_LO12_S = 25,

    // -- Absolute LUI-based addressing (26–28) --

    /// High 20 bits of absolute address (for LUI instruction, U-type).
    R_RISCV_HI20 = 26,
    /// Low 12 bits of absolute address (I-type instruction).
    R_RISCV_LO12_I = 27,
    /// Low 12 bits of absolute address (S-type instruction).
    R_RISCV_LO12_S = 28,

    // -- Arithmetic relocations for DWARF / exception tables (33–40) --

    /// Add 8-bit value to relocation site.
    R_RISCV_ADD8 = 33,
    /// Add 16-bit value to relocation site.
    R_RISCV_ADD16 = 34,
    /// Add 32-bit value to relocation site.
    R_RISCV_ADD32 = 35,
    /// Add 64-bit value to relocation site.
    R_RISCV_ADD64 = 36,
    /// Subtract 8-bit value from relocation site.
    R_RISCV_SUB8 = 37,
    /// Subtract 16-bit value from relocation site.
    R_RISCV_SUB16 = 38,
    /// Subtract 32-bit value from relocation site.
    R_RISCV_SUB32 = 39,
    /// Subtract 64-bit value from relocation site.
    R_RISCV_SUB64 = 40,

    // -- Compressed instruction relocations (44–45) --

    /// Compressed branch offset (C.BEQZ / C.BNEZ), 9-bit signed, CB-type.
    /// Value must be 2-byte aligned. Range: −256 to +254.
    R_RISCV_RVC_BRANCH = 44,
    /// Compressed jump offset (C.J / C.JAL), 12-bit signed, CJ-type.
    /// Value must be 2-byte aligned. Range: −2048 to +2046.
    R_RISCV_RVC_JUMP = 45,

    // -- Linker relaxation marker --

    /// Marks the preceding relocation as eligible for linker relaxation.
    /// The linker may shrink the instruction sequence when the target is
    /// close enough (e.g., AUIPC+JALR → JAL, or GOT load → direct access).
    R_RISCV_RELAX = 51,

    // -- SET relocations for DWARF / exception tables (53–56) --

    /// Set low 6 bits at relocation site (preserves upper bits of the byte).
    R_RISCV_SET6 = 53,
    /// Set 8-bit value at relocation site.
    R_RISCV_SET8 = 54,
    /// Set 16-bit value at relocation site.
    R_RISCV_SET16 = 55,
    /// Set 32-bit value at relocation site.
    R_RISCV_SET32 = 56,

    /// Alignment directive — linker must maintain alignment after relaxation.
    /// The addend specifies the required alignment in bytes.
    R_RISCV_ALIGN = 57,
}

// ============================================================================
// RiscV64RelocationType — impl methods
// ============================================================================

impl RiscV64RelocationType {
    /// Converts this relocation type to its ELF `r_type` numeric value.
    ///
    /// Because the enum is `#[repr(u32)]`, this is a zero-cost cast.
    #[inline]
    pub fn to_elf_value(&self) -> u32 {
        *self as u32
    }

    /// Parses an ELF `r_type` value into the corresponding enum variant.
    ///
    /// Returns `None` if the value does not correspond to a known
    /// RISC-V relocation type.
    pub fn from_elf_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::R_RISCV_NONE),
            1 => Some(Self::R_RISCV_32),
            2 => Some(Self::R_RISCV_64),
            3 => Some(Self::R_RISCV_RELATIVE),
            4 => Some(Self::R_RISCV_COPY),
            5 => Some(Self::R_RISCV_JUMP_SLOT),
            6 => Some(Self::R_RISCV_TLS_DTPMOD32),
            7 => Some(Self::R_RISCV_TLS_DTPMOD64),
            8 => Some(Self::R_RISCV_TLS_DTPREL32),
            9 => Some(Self::R_RISCV_TLS_DTPREL64),
            10 => Some(Self::R_RISCV_TLS_TPREL32),
            11 => Some(Self::R_RISCV_TLS_TPREL64),
            16 => Some(Self::R_RISCV_BRANCH),
            17 => Some(Self::R_RISCV_JAL),
            18 => Some(Self::R_RISCV_CALL),
            19 => Some(Self::R_RISCV_CALL_PLT),
            20 => Some(Self::R_RISCV_GOT_HI20),
            21 => Some(Self::R_RISCV_TLS_GOT_HI20),
            22 => Some(Self::R_RISCV_TLS_GD_HI20),
            23 => Some(Self::R_RISCV_PCREL_HI20),
            24 => Some(Self::R_RISCV_PCREL_LO12_I),
            25 => Some(Self::R_RISCV_PCREL_LO12_S),
            26 => Some(Self::R_RISCV_HI20),
            27 => Some(Self::R_RISCV_LO12_I),
            28 => Some(Self::R_RISCV_LO12_S),
            33 => Some(Self::R_RISCV_ADD8),
            34 => Some(Self::R_RISCV_ADD16),
            35 => Some(Self::R_RISCV_ADD32),
            36 => Some(Self::R_RISCV_ADD64),
            37 => Some(Self::R_RISCV_SUB8),
            38 => Some(Self::R_RISCV_SUB16),
            39 => Some(Self::R_RISCV_SUB32),
            40 => Some(Self::R_RISCV_SUB64),
            44 => Some(Self::R_RISCV_RVC_BRANCH),
            45 => Some(Self::R_RISCV_RVC_JUMP),
            51 => Some(Self::R_RISCV_RELAX),
            53 => Some(Self::R_RISCV_SET6),
            54 => Some(Self::R_RISCV_SET8),
            55 => Some(Self::R_RISCV_SET16),
            56 => Some(Self::R_RISCV_SET32),
            57 => Some(Self::R_RISCV_ALIGN),
            _ => None,
        }
    }

    /// Returns the canonical human-readable name of this relocation type,
    /// matching the ELF psABI naming convention (e.g., `"R_RISCV_BRANCH"`).
    pub fn name(&self) -> &'static str {
        match self {
            Self::R_RISCV_NONE => "R_RISCV_NONE",
            Self::R_RISCV_32 => "R_RISCV_32",
            Self::R_RISCV_64 => "R_RISCV_64",
            Self::R_RISCV_RELATIVE => "R_RISCV_RELATIVE",
            Self::R_RISCV_COPY => "R_RISCV_COPY",
            Self::R_RISCV_JUMP_SLOT => "R_RISCV_JUMP_SLOT",
            Self::R_RISCV_TLS_DTPMOD32 => "R_RISCV_TLS_DTPMOD32",
            Self::R_RISCV_TLS_DTPMOD64 => "R_RISCV_TLS_DTPMOD64",
            Self::R_RISCV_TLS_DTPREL32 => "R_RISCV_TLS_DTPREL32",
            Self::R_RISCV_TLS_DTPREL64 => "R_RISCV_TLS_DTPREL64",
            Self::R_RISCV_TLS_TPREL32 => "R_RISCV_TLS_TPREL32",
            Self::R_RISCV_TLS_TPREL64 => "R_RISCV_TLS_TPREL64",
            Self::R_RISCV_BRANCH => "R_RISCV_BRANCH",
            Self::R_RISCV_JAL => "R_RISCV_JAL",
            Self::R_RISCV_CALL => "R_RISCV_CALL",
            Self::R_RISCV_CALL_PLT => "R_RISCV_CALL_PLT",
            Self::R_RISCV_GOT_HI20 => "R_RISCV_GOT_HI20",
            Self::R_RISCV_TLS_GOT_HI20 => "R_RISCV_TLS_GOT_HI20",
            Self::R_RISCV_TLS_GD_HI20 => "R_RISCV_TLS_GD_HI20",
            Self::R_RISCV_PCREL_HI20 => "R_RISCV_PCREL_HI20",
            Self::R_RISCV_PCREL_LO12_I => "R_RISCV_PCREL_LO12_I",
            Self::R_RISCV_PCREL_LO12_S => "R_RISCV_PCREL_LO12_S",
            Self::R_RISCV_HI20 => "R_RISCV_HI20",
            Self::R_RISCV_LO12_I => "R_RISCV_LO12_I",
            Self::R_RISCV_LO12_S => "R_RISCV_LO12_S",
            Self::R_RISCV_ADD8 => "R_RISCV_ADD8",
            Self::R_RISCV_ADD16 => "R_RISCV_ADD16",
            Self::R_RISCV_ADD32 => "R_RISCV_ADD32",
            Self::R_RISCV_ADD64 => "R_RISCV_ADD64",
            Self::R_RISCV_SUB8 => "R_RISCV_SUB8",
            Self::R_RISCV_SUB16 => "R_RISCV_SUB16",
            Self::R_RISCV_SUB32 => "R_RISCV_SUB32",
            Self::R_RISCV_SUB64 => "R_RISCV_SUB64",
            Self::R_RISCV_RVC_BRANCH => "R_RISCV_RVC_BRANCH",
            Self::R_RISCV_RVC_JUMP => "R_RISCV_RVC_JUMP",
            Self::R_RISCV_RELAX => "R_RISCV_RELAX",
            Self::R_RISCV_SET6 => "R_RISCV_SET6",
            Self::R_RISCV_SET8 => "R_RISCV_SET8",
            Self::R_RISCV_SET16 => "R_RISCV_SET16",
            Self::R_RISCV_SET32 => "R_RISCV_SET32",
            Self::R_RISCV_ALIGN => "R_RISCV_ALIGN",
        }
    }

    /// Returns `true` if this relocation computes a PC-relative offset.
    ///
    /// PC-relative relocations require the address of both the relocation
    /// site and the target symbol to compute the final value.
    pub fn is_pc_relative(&self) -> bool {
        matches!(
            self,
            Self::R_RISCV_BRANCH
                | Self::R_RISCV_JAL
                | Self::R_RISCV_CALL
                | Self::R_RISCV_CALL_PLT
                | Self::R_RISCV_GOT_HI20
                | Self::R_RISCV_TLS_GOT_HI20
                | Self::R_RISCV_TLS_GD_HI20
                | Self::R_RISCV_PCREL_HI20
                | Self::R_RISCV_PCREL_LO12_I
                | Self::R_RISCV_PCREL_LO12_S
                | Self::R_RISCV_RVC_BRANCH
                | Self::R_RISCV_RVC_JUMP
        )
    }

    /// Returns the effective number of bits this relocation occupies.
    ///
    /// For instruction relocations this is the width of the encoded value
    /// (including implicit zero bits for alignment-constrained offsets).
    /// For data relocations this is the full data width. Marker-only
    /// relocations (RELAX, ALIGN) return 0.
    pub fn bit_size(&self) -> u8 {
        match self {
            Self::R_RISCV_NONE | Self::R_RISCV_COPY
                | Self::R_RISCV_RELAX | Self::R_RISCV_ALIGN => 0,
            Self::R_RISCV_SET6 => 6,
            Self::R_RISCV_ADD8 | Self::R_RISCV_SUB8
                | Self::R_RISCV_SET8 => 8,
            Self::R_RISCV_RVC_BRANCH => 9,
            Self::R_RISCV_PCREL_LO12_I | Self::R_RISCV_PCREL_LO12_S
                | Self::R_RISCV_LO12_I | Self::R_RISCV_LO12_S
                | Self::R_RISCV_RVC_JUMP => 12,
            Self::R_RISCV_BRANCH => 13,
            Self::R_RISCV_ADD16 | Self::R_RISCV_SUB16
                | Self::R_RISCV_SET16 => 16,
            Self::R_RISCV_PCREL_HI20 | Self::R_RISCV_HI20
                | Self::R_RISCV_GOT_HI20 | Self::R_RISCV_TLS_GOT_HI20
                | Self::R_RISCV_TLS_GD_HI20 => 20,
            Self::R_RISCV_JAL => 21,
            Self::R_RISCV_32 | Self::R_RISCV_TLS_DTPMOD32
                | Self::R_RISCV_TLS_DTPREL32 | Self::R_RISCV_TLS_TPREL32
                | Self::R_RISCV_CALL | Self::R_RISCV_CALL_PLT
                | Self::R_RISCV_ADD32 | Self::R_RISCV_SUB32
                | Self::R_RISCV_SET32 => 32,
            Self::R_RISCV_64 | Self::R_RISCV_RELATIVE
                | Self::R_RISCV_JUMP_SLOT | Self::R_RISCV_TLS_DTPMOD64
                | Self::R_RISCV_TLS_DTPREL64 | Self::R_RISCV_TLS_TPREL64
                | Self::R_RISCV_ADD64 | Self::R_RISCV_SUB64 => 64,
        }
    }

    /// Returns `true` if the linker may optimise (relax) this relocation
    /// to a shorter instruction sequence when the target is within range.
    ///
    /// Relaxable relocations are typically emitted alongside a companion
    /// [`R_RISCV_RELAX`](Self::R_RISCV_RELAX) marker by the assembler.
    pub fn is_relaxable(&self) -> bool {
        matches!(
            self,
            Self::R_RISCV_CALL
                | Self::R_RISCV_CALL_PLT
                | Self::R_RISCV_GOT_HI20
                | Self::R_RISCV_TLS_GD_HI20
                | Self::R_RISCV_TLS_GOT_HI20
        )
    }

    /// Returns `true` if the assembler should typically emit a companion
    /// [`R_RISCV_RELAX`](Self::R_RISCV_RELAX) alongside this relocation.
    ///
    /// This covers directly relaxable relocations and their paired lo-12
    /// counterparts whose instruction may be eliminated or resized when
    /// the HI20 half is relaxed.
    pub fn is_paired_with_relax(&self) -> bool {
        matches!(
            self,
            Self::R_RISCV_CALL
                | Self::R_RISCV_CALL_PLT
                | Self::R_RISCV_GOT_HI20
                | Self::R_RISCV_TLS_GOT_HI20
                | Self::R_RISCV_TLS_GD_HI20
                | Self::R_RISCV_PCREL_HI20
                | Self::R_RISCV_PCREL_LO12_I
                | Self::R_RISCV_PCREL_LO12_S
                | Self::R_RISCV_HI20
                | Self::R_RISCV_LO12_I
                | Self::R_RISCV_LO12_S
        )
    }

    /// Returns `true` if this relocation requires a GOT (Global Offset Table)
    /// entry for the referenced symbol.
    pub fn requires_got_entry(&self) -> bool {
        matches!(
            self,
            Self::R_RISCV_GOT_HI20 | Self::R_RISCV_TLS_GOT_HI20
        )
    }

    /// Returns `true` if this relocation requires a PLT (Procedure Linkage
    /// Table) entry for the referenced symbol.
    pub fn requires_plt_entry(&self) -> bool {
        matches!(self, Self::R_RISCV_CALL_PLT)
    }

    /// Returns the byte size of the memory region this relocation patches.
    ///
    /// Used to populate the [`RelocationType::size`] field in the
    /// architecture-agnostic relocation descriptor table.
    fn byte_size(&self) -> u8 {
        match self {
            Self::R_RISCV_NONE | Self::R_RISCV_COPY
                | Self::R_RISCV_RELAX | Self::R_RISCV_ALIGN => 0,
            Self::R_RISCV_SET6 | Self::R_RISCV_SET8
                | Self::R_RISCV_ADD8 | Self::R_RISCV_SUB8 => 1,
            Self::R_RISCV_ADD16 | Self::R_RISCV_SUB16
                | Self::R_RISCV_SET16 | Self::R_RISCV_RVC_BRANCH
                | Self::R_RISCV_RVC_JUMP => 2,
            Self::R_RISCV_32 | Self::R_RISCV_TLS_DTPMOD32
                | Self::R_RISCV_TLS_DTPREL32 | Self::R_RISCV_TLS_TPREL32
                | Self::R_RISCV_BRANCH | Self::R_RISCV_JAL
                | Self::R_RISCV_PCREL_HI20 | Self::R_RISCV_PCREL_LO12_I
                | Self::R_RISCV_PCREL_LO12_S | Self::R_RISCV_HI20
                | Self::R_RISCV_LO12_I | Self::R_RISCV_LO12_S
                | Self::R_RISCV_GOT_HI20 | Self::R_RISCV_TLS_GOT_HI20
                | Self::R_RISCV_TLS_GD_HI20 | Self::R_RISCV_ADD32
                | Self::R_RISCV_SUB32 | Self::R_RISCV_SET32 => 4,
            Self::R_RISCV_64 | Self::R_RISCV_RELATIVE
                | Self::R_RISCV_JUMP_SLOT | Self::R_RISCV_TLS_DTPMOD64
                | Self::R_RISCV_TLS_DTPREL64 | Self::R_RISCV_TLS_TPREL64
                | Self::R_RISCV_CALL | Self::R_RISCV_CALL_PLT
                | Self::R_RISCV_ADD64 | Self::R_RISCV_SUB64 => 8,
        }
    }
}

impl fmt::Display for RiscV64RelocationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// RelocationError — relocation patching errors
// ============================================================================

/// Error type returned when a relocation cannot be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelocationError {
    /// The computed value exceeds the representable range of the relocation
    /// field. This typically means the target symbol is too far away for the
    /// instruction encoding.
    Overflow {
        /// The relocation type that overflowed.
        reloc_type: RiscV64RelocationType,
        /// The value that could not be encoded.
        value: i64,
        /// Maximum number of bits available in the encoding.
        max_bits: u8,
    },
    /// The computed value violates the alignment constraint required by the
    /// instruction encoding (e.g., branch offsets must be 2-byte aligned).
    AlignmentError {
        /// The misaligned value.
        value: i64,
        /// The required alignment in bytes.
        required: u32,
    },
    /// The relocation type is not supported for instruction patching
    /// (e.g., dynamic-only relocations like COPY or marker-only types).
    UnsupportedType(RiscV64RelocationType),
}

impl fmt::Display for RelocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelocationError::Overflow {
                reloc_type,
                value,
                max_bits,
            } => write!(
                f,
                "relocation overflow: {} value {:#x} exceeds {}-bit range",
                reloc_type.name(),
                value,
                max_bits,
            ),
            RelocationError::AlignmentError { value, required } => write!(
                f,
                "relocation alignment error: value {:#x} is not {}-byte aligned",
                value, required,
            ),
            RelocationError::UnsupportedType(rt) => write!(
                f,
                "unsupported relocation type for instruction patching: {}",
                rt.name(),
            ),
        }
    }
}

// ============================================================================
// RelaxationKind — linker relaxation transformation categories
// ============================================================================

/// Describes the type of linker relaxation that can be applied to a
/// relaxable relocation.
///
/// The linker examines each relaxable relocation (those paired with
/// `R_RISCV_RELAX`) and determines whether the instruction sequence can
/// be shortened based on the resolved distance to the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelaxationKind {
    /// AUIPC+JALR call pair → single JAL (target within ±1 MiB).
    CallToJal,
    /// AUIPC+JALR call pair → compressed C.JAL (target within ±2 KiB,
    /// only valid on RV32 with RVC; on RV64 this variant exists for
    /// completeness but is not normally applied).
    CallToCJal,
    /// GOT-indirect load (AUIPC+LD via GOT) → direct AUIPC+ADDI when
    /// the symbol is defined locally or within the same link unit.
    GotToLocal,
    /// TLS General Dynamic → Local Exec: replace the `__tls_get_addr` call
    /// sequence with a direct TP-relative access when linking a static
    /// executable.
    TlsGdToLe,
    /// TLS Initial Exec → Local Exec: replace the GOT-indirect TP load
    /// with a direct TP-relative access.
    TlsIeToLe,
}

impl fmt::Display for RelaxationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelaxationKind::CallToJal => write!(f, "CALL->JAL"),
            RelaxationKind::CallToCJal => write!(f, "CALL->C.JAL"),
            RelaxationKind::GotToLocal => write!(f, "GOT->LOCAL"),
            RelaxationKind::TlsGdToLe => write!(f, "TLS_GD->LE"),
            RelaxationKind::TlsIeToLe => write!(f, "TLS_IE->LE"),
        }
    }
}

// ============================================================================
// Little-endian byte I/O helpers (internal)
// ============================================================================

/// Reads a 16-bit little-endian value from a byte slice.
#[inline(always)]
fn read_le16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

/// Writes a 16-bit little-endian value into a byte slice.
#[inline(always)]
fn write_le16(bytes: &mut [u8], val: u16) {
    let le = val.to_le_bytes();
    bytes[0] = le[0];
    bytes[1] = le[1];
}

/// Reads a 32-bit little-endian value from a byte slice.
#[inline(always)]
fn read_le32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Writes a 32-bit little-endian value into a byte slice.
#[inline(always)]
fn write_le32(bytes: &mut [u8], val: u32) {
    let le = val.to_le_bytes();
    bytes[0] = le[0];
    bytes[1] = le[1];
    bytes[2] = le[2];
    bytes[3] = le[3];
}

/// Reads a 64-bit little-endian value from a byte slice.
#[inline(always)]
fn read_le64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Writes a 64-bit little-endian value into a byte slice.
#[inline(always)]
fn write_le64(bytes: &mut [u8], val: u64) {
    let le = val.to_le_bytes();
    bytes[0] = le[0];
    bytes[1] = le[1];
    bytes[2] = le[2];
    bytes[3] = le[3];
    bytes[4] = le[4];
    bytes[5] = le[5];
    bytes[6] = le[6];
    bytes[7] = le[7];
}

// ============================================================================
// split_hi_lo — Hi/Lo address splitting with sign-extension compensation
// ============================================================================

/// Splits a value into upper-20-bit and lower-12-bit components for use
/// with LUI/AUIPC + ADDI/LD/SD instruction pairs.
///
/// # Sign-Extension Compensation
///
/// The lower 12 bits are sign-extended by the CPU when used in I-type or
/// S-type instructions. When the low 12 bits have their sign bit set
/// (bit 11 = 1), the upper 20 bits must be incremented by 1 to compensate
/// for the sign extension, ensuring that `(hi << 12) + lo == value`.
///
/// # Formula
///
/// ```text
/// lo = sign_extend_12(value)   // ((value as i32) << 20) >> 20
/// hi = (value - lo) >> 12
/// ```
///
/// # Invariant
///
/// For any input `value` that fits in 32 bits:
/// ```text
/// let (hi, lo) = split_hi_lo(value);
/// assert_eq!((hi as i64) * 4096 + (lo as i64), value);
/// ```
pub fn split_hi_lo(value: i64) -> (i32, i32) {
    // Sign-extend the low 12 bits to 32-bit by shifting left then
    // arithmetically shifting right, preserving the sign.
    let lo: i32 = ((value as i32) << 20) >> 20;
    // Subtract the sign-extended lo from the original value, then shift
    // right by 12 to get the upper 20 bits. This automatically handles
    // the +1 compensation when lo is negative.
    let hi: i32 = ((value.wrapping_sub(lo as i64)) >> 12) as i32;
    (hi, lo)
}

// ============================================================================
// Instruction patching helpers (internal)
// ============================================================================

/// Checks that `value` fits in a signed `bits`-bit integer and returns an
/// overflow error on failure.
#[inline]
fn check_signed_range(
    value: i64,
    bits: u8,
    reloc_type: RiscV64RelocationType,
) -> Result<(), RelocationError> {
    let min = -(1_i64 << (bits - 1));
    let max = (1_i64 << (bits - 1)) - 1;
    if value < min || value > max {
        return Err(RelocationError::Overflow {
            reloc_type,
            value,
            max_bits: bits,
        });
    }
    Ok(())
}

/// Checks that `value` is aligned to `align` bytes and returns an alignment
/// error on failure.
#[inline]
fn check_alignment(value: i64, align: u32) -> Result<(), RelocationError> {
    if (value as u64) & (align as u64 - 1) != 0 {
        return Err(RelocationError::AlignmentError {
            value,
            required: align,
        });
    }
    Ok(())
}

/// Patches a B-type (branch) instruction with a 13-bit signed offset.
///
/// B-type immediate encoding:
/// - inst\[31\]    = offset\[12\]
/// - inst\[30:25\] = offset\[10:5\]
/// - inst\[11:8\]  = offset\[4:1\]
/// - inst\[7\]     = offset\[11\]
fn apply_b_type(bytes: &mut [u8], value: i64) -> Result<(), RelocationError> {
    check_signed_range(value, 13, RiscV64RelocationType::R_RISCV_BRANCH)?;
    check_alignment(value, 2)?;

    let v = value as u32;
    let imm12   = (v >> 12) & 0x1;
    let imm10_5 = (v >> 5) & 0x3F;
    let imm4_1  = (v >> 1) & 0xF;
    let imm11   = (v >> 11) & 0x1;

    let instr = read_le32(bytes);
    // Mask clears: bit 31, bits 30:25, bits 11:8, bit 7
    let mask: u32 = (0x1 << 31) | (0x3F << 25) | (0xF << 8) | (0x1 << 7);
    let patched = (instr & !mask)
        | (imm12 << 31)
        | (imm10_5 << 25)
        | (imm4_1 << 8)
        | (imm11 << 7);
    write_le32(bytes, patched);
    Ok(())
}

/// Patches a J-type (JAL) instruction with a 21-bit signed offset.
///
/// J-type immediate encoding:
/// - inst\[31\]    = offset\[20\]
/// - inst\[30:21\] = offset\[10:1\]
/// - inst\[20\]    = offset\[11\]
/// - inst\[19:12\] = offset\[19:12\]
fn apply_j_type(bytes: &mut [u8], value: i64) -> Result<(), RelocationError> {
    check_signed_range(value, 21, RiscV64RelocationType::R_RISCV_JAL)?;
    check_alignment(value, 2)?;

    let v = value as u32;
    let imm20    = (v >> 20) & 0x1;
    let imm10_1  = (v >> 1) & 0x3FF;
    let imm11    = (v >> 11) & 0x1;
    let imm19_12 = (v >> 12) & 0xFF;

    let instr = read_le32(bytes);
    // Mask clears: bit 31, bits 30:21, bit 20, bits 19:12
    let mask: u32 = (0x1 << 31) | (0x3FF << 21) | (0x1 << 20) | (0xFF << 12);
    let patched = (instr & !mask)
        | (imm20 << 31)
        | (imm10_1 << 21)
        | (imm11 << 20)
        | (imm19_12 << 12);
    write_le32(bytes, patched);
    Ok(())
}

/// Patches a U-type (LUI / AUIPC) instruction with a 20-bit immediate.
///
/// The `hi` value (from [`split_hi_lo`]) is placed at bits \[31:12\].
fn apply_u_type(bytes: &mut [u8], hi: i32) -> Result<(), RelocationError> {
    let instr = read_le32(bytes);
    // Mask clears bits [31:12] (the immediate field).
    let mask: u32 = 0xFFFFF000;
    let patched = (instr & !mask) | (((hi as u32) & 0xFFFFF) << 12);
    write_le32(bytes, patched);
    Ok(())
}

/// Patches an I-type instruction immediate (bits \[31:20\]) with a 12-bit value.
fn apply_i_type_imm(bytes: &mut [u8], lo: i32) -> Result<(), RelocationError> {
    let instr = read_le32(bytes);
    // Mask clears bits [31:20] (the immediate field).
    let mask: u32 = 0xFFF00000;
    let patched = (instr & !mask) | (((lo as u32) & 0xFFF) << 20);
    write_le32(bytes, patched);
    Ok(())
}

/// Patches an S-type instruction split-immediate with a 12-bit value.
///
/// S-type immediate encoding:
/// - inst\[31:25\] = imm\[11:5\]
/// - inst\[11:7\]  = imm\[4:0\]
fn apply_s_type_imm(bytes: &mut [u8], lo: i32) -> Result<(), RelocationError> {
    let v = (lo as u32) & 0xFFF;
    let imm11_5 = (v >> 5) & 0x7F;
    let imm4_0  = v & 0x1F;

    let instr = read_le32(bytes);
    let mask: u32 = (0x7F << 25) | (0x1F << 7);
    let patched = (instr & !mask) | (imm11_5 << 25) | (imm4_0 << 7);
    write_le32(bytes, patched);
    Ok(())
}

/// Patches a CB-type (compressed branch: C.BEQZ / C.BNEZ) instruction
/// with a 9-bit signed offset.
///
/// CB-type immediate encoding:
/// - inst\[12\]   = offset\[8\]
/// - inst\[11:10\] = offset\[4:3\]
/// - inst\[6:5\]  = offset\[7:6\]
/// - inst\[4:3\]  = offset\[2:1\]
/// - inst\[2\]    = offset\[5\]
fn apply_cb_type(bytes: &mut [u8], value: i64) -> Result<(), RelocationError> {
    check_signed_range(value, 9, RiscV64RelocationType::R_RISCV_RVC_BRANCH)?;
    check_alignment(value, 2)?;

    let v = value as u32;
    let mut imm: u16 = 0;
    imm |= (((v >> 8) & 0x1) as u16) << 12;  // offset[8] → inst[12]
    imm |= (((v >> 3) & 0x3) as u16) << 10;  // offset[4:3] → inst[11:10]
    imm |= (((v >> 6) & 0x3) as u16) << 5;   // offset[7:6] → inst[6:5]
    imm |= (((v >> 1) & 0x3) as u16) << 3;   // offset[2:1] → inst[4:3]
    imm |= (((v >> 5) & 0x1) as u16) << 2;   // offset[5] → inst[2]

    let instr = read_le16(bytes);
    // Mask covers bits 12, 11:10, 6:5, 4:3, 2
    let mask: u16 = (0x1 << 12) | (0x3 << 10) | (0x3 << 5) | (0x3 << 3) | (0x1 << 2);
    let patched = (instr & !mask) | imm;
    write_le16(bytes, patched);
    Ok(())
}

/// Patches a CJ-type (compressed jump: C.J / C.JAL) instruction with a
/// 12-bit signed offset.
///
/// CJ-type immediate encoding:
/// - inst\[12\]   = offset\[11\]
/// - inst\[11\]   = offset\[4\]
/// - inst\[10:9\] = offset\[9:8\]
/// - inst\[8\]    = offset\[10\]
/// - inst\[7\]    = offset\[6\]
/// - inst\[6\]    = offset\[7\]
/// - inst\[5:3\]  = offset\[3:1\]
/// - inst\[2\]    = offset\[5\]
fn apply_cj_type(bytes: &mut [u8], value: i64) -> Result<(), RelocationError> {
    check_signed_range(value, 12, RiscV64RelocationType::R_RISCV_RVC_JUMP)?;
    check_alignment(value, 2)?;

    let v = value as u32;
    let mut imm: u16 = 0;
    imm |= (((v >> 11) & 0x1) as u16) << 12;  // offset[11] → inst[12]
    imm |= (((v >> 4) & 0x1) as u16) << 11;   // offset[4]  → inst[11]
    imm |= (((v >> 8) & 0x3) as u16) << 9;    // offset[9:8] → inst[10:9]
    imm |= (((v >> 10) & 0x1) as u16) << 8;   // offset[10] → inst[8]
    imm |= (((v >> 6) & 0x1) as u16) << 7;    // offset[6]  → inst[7]
    imm |= (((v >> 7) & 0x1) as u16) << 6;    // offset[7]  → inst[6]
    imm |= (((v >> 1) & 0x7) as u16) << 3;    // offset[3:1] → inst[5:3]
    imm |= (((v >> 5) & 0x1) as u16) << 2;    // offset[5]  → inst[2]

    let instr = read_le16(bytes);
    // Mask covers bits 12:2 = 0x1FFC
    let mask: u16 = 0x1FFC;
    let patched = (instr & !mask) | imm;
    write_le16(bytes, patched);
    Ok(())
}

// ============================================================================
// apply_relocation — main relocation patching entry point
// ============================================================================

/// Applies a RISC-V 64 relocation to instruction or data bytes at the
/// relocation site.
///
/// # Parameters
///
/// - `reloc_type`: The RISC-V ELF relocation type to apply.
/// - `instruction_bytes`: Mutable reference to the bytes at the relocation
///   site. Must be at least `reloc_type.byte_size()` bytes long.
/// - `value`: The resolved relocation value. For PC-relative relocations
///   this is `S + A - P` (symbol + addend − place). For absolute
///   relocations this is `S + A`.
///
/// # Errors
///
/// Returns [`RelocationError::Overflow`] if the value exceeds the
/// instruction's immediate field width, [`RelocationError::AlignmentError`]
/// if the value violates alignment constraints, or
/// [`RelocationError::UnsupportedType`] for dynamic/marker relocations
/// that are not patched at static link time.
pub fn apply_relocation(
    reloc_type: RiscV64RelocationType,
    instruction_bytes: &mut [u8],
    value: i64,
) -> Result<(), RelocationError> {
    match reloc_type {
        // ----------------------------------------------------------------
        // No-op / marker relocations
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_NONE
        | RiscV64RelocationType::R_RISCV_RELAX
        | RiscV64RelocationType::R_RISCV_ALIGN => Ok(()),

        // ----------------------------------------------------------------
        // Dynamic-only relocations (handled by runtime linker, not patchable)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_COPY => {
            Err(RelocationError::UnsupportedType(reloc_type))
        }

        // ----------------------------------------------------------------
        // Absolute data relocations
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_32 => {
            check_signed_range(value, 32, reloc_type)?;
            write_le32(instruction_bytes, value as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_64 => {
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_RELATIVE => {
            // RELATIVE: base + addend — written as 64-bit for RV64.
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_JUMP_SLOT => {
            // PLT jump slot: write 64-bit function address.
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }

        // ----------------------------------------------------------------
        // TLS data relocations
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_TLS_DTPMOD32 => {
            write_le32(instruction_bytes, value as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_TLS_DTPMOD64 => {
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_TLS_DTPREL32 => {
            write_le32(instruction_bytes, value as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_TLS_DTPREL64 => {
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_TLS_TPREL32 => {
            write_le32(instruction_bytes, value as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_TLS_TPREL64 => {
            write_le64(instruction_bytes, value as u64);
            Ok(())
        }

        // ----------------------------------------------------------------
        // B-type branch
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_BRANCH => {
            apply_b_type(instruction_bytes, value)
        }

        // ----------------------------------------------------------------
        // J-type jump (JAL)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_JAL => {
            apply_j_type(instruction_bytes, value)
        }

        // ----------------------------------------------------------------
        // AUIPC+JALR call pair (CALL / CALL_PLT)
        // Bytes [0..4]: AUIPC (U-type), bytes [4..8]: JALR (I-type)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_CALL
        | RiscV64RelocationType::R_RISCV_CALL_PLT => {
            check_signed_range(value, 32, reloc_type)?;
            let (hi, lo) = split_hi_lo(value);
            apply_u_type(&mut instruction_bytes[0..4], hi)?;
            apply_i_type_imm(&mut instruction_bytes[4..8], lo)?;
            Ok(())
        }

        // ----------------------------------------------------------------
        // U-type HI20 relocations (PCREL / absolute / GOT / TLS)
        // The caller passes the full offset; we split and encode hi.
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_PCREL_HI20
        | RiscV64RelocationType::R_RISCV_HI20
        | RiscV64RelocationType::R_RISCV_GOT_HI20
        | RiscV64RelocationType::R_RISCV_TLS_GOT_HI20
        | RiscV64RelocationType::R_RISCV_TLS_GD_HI20 => {
            let (hi, _lo) = split_hi_lo(value);
            apply_u_type(instruction_bytes, hi)
        }

        // ----------------------------------------------------------------
        // I-type LO12 relocations (PC-relative / absolute)
        // The caller passes the full offset; we split and encode lo.
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_PCREL_LO12_I
        | RiscV64RelocationType::R_RISCV_LO12_I => {
            let (_hi, lo) = split_hi_lo(value);
            apply_i_type_imm(instruction_bytes, lo)
        }

        // ----------------------------------------------------------------
        // S-type LO12 relocations (PC-relative / absolute)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_PCREL_LO12_S
        | RiscV64RelocationType::R_RISCV_LO12_S => {
            let (_hi, lo) = split_hi_lo(value);
            apply_s_type_imm(instruction_bytes, lo)
        }

        // ----------------------------------------------------------------
        // Compressed branch (CB-type)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_RVC_BRANCH => {
            apply_cb_type(instruction_bytes, value)
        }

        // ----------------------------------------------------------------
        // Compressed jump (CJ-type)
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_RVC_JUMP => {
            apply_cj_type(instruction_bytes, value)
        }

        // ----------------------------------------------------------------
        // Arithmetic ADD relocations: add `value` to existing data
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_ADD8 => {
            let existing = instruction_bytes[0] as i8;
            instruction_bytes[0] = existing.wrapping_add(value as i8) as u8;
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_ADD16 => {
            let existing = read_le16(instruction_bytes) as i16;
            write_le16(instruction_bytes, existing.wrapping_add(value as i16) as u16);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_ADD32 => {
            let existing = read_le32(instruction_bytes) as i32;
            write_le32(instruction_bytes, existing.wrapping_add(value as i32) as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_ADD64 => {
            let existing = read_le64(instruction_bytes) as i64;
            write_le64(instruction_bytes, existing.wrapping_add(value) as u64);
            Ok(())
        }

        // ----------------------------------------------------------------
        // Arithmetic SUB relocations: subtract `value` from existing data
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_SUB8 => {
            let existing = instruction_bytes[0] as i8;
            instruction_bytes[0] = existing.wrapping_sub(value as i8) as u8;
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SUB16 => {
            let existing = read_le16(instruction_bytes) as i16;
            write_le16(instruction_bytes, existing.wrapping_sub(value as i16) as u16);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SUB32 => {
            let existing = read_le32(instruction_bytes) as i32;
            write_le32(instruction_bytes, existing.wrapping_sub(value as i32) as u32);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SUB64 => {
            let existing = read_le64(instruction_bytes) as i64;
            write_le64(instruction_bytes, existing.wrapping_sub(value) as u64);
            Ok(())
        }

        // ----------------------------------------------------------------
        // SET relocations: overwrite existing data with `value`
        // ----------------------------------------------------------------
        RiscV64RelocationType::R_RISCV_SET6 => {
            // Preserve the upper 2 bits of the byte, overwrite lower 6.
            let existing = instruction_bytes[0];
            instruction_bytes[0] = (existing & 0xC0) | ((value as u8) & 0x3F);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SET8 => {
            instruction_bytes[0] = value as u8;
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SET16 => {
            write_le16(instruction_bytes, value as u16);
            Ok(())
        }
        RiscV64RelocationType::R_RISCV_SET32 => {
            write_le32(instruction_bytes, value as u32);
            Ok(())
        }
    }
}

// ============================================================================
// relaxation_target — determine relaxation transformation for a relocation
// ============================================================================

/// Determines the type of linker relaxation that can be applied to the
/// given relocation type.
///
/// Returns `None` if the relocation type is not relaxable. The linker
/// should only attempt relaxation when the relocation is accompanied by
/// an `R_RISCV_RELAX` marker.
///
/// # Relaxation Possibilities
///
/// - **`R_RISCV_CALL`** / **`R_RISCV_CALL_PLT`**: Can relax to a single
///   `JAL` instruction if the target is within ±1 MiB, or to `C.JAL` if
///   within ±2 KiB on RV32 with RVC.
/// - **`R_RISCV_GOT_HI20`**: Can relax to direct `AUIPC+ADDI` if the
///   symbol is defined locally.
/// - **`R_RISCV_TLS_GD_HI20`**: Can relax from General Dynamic to Local
///   Exec model for statically linked executables.
/// - **`R_RISCV_TLS_GOT_HI20`**: Can relax from Initial Exec to Local
///   Exec model for statically linked executables.
pub fn relaxation_target(reloc_type: RiscV64RelocationType) -> Option<RelaxationKind> {
    match reloc_type {
        RiscV64RelocationType::R_RISCV_CALL
        | RiscV64RelocationType::R_RISCV_CALL_PLT => Some(RelaxationKind::CallToJal),
        RiscV64RelocationType::R_RISCV_GOT_HI20 => Some(RelaxationKind::GotToLocal),
        RiscV64RelocationType::R_RISCV_TLS_GD_HI20 => Some(RelaxationKind::TlsGdToLe),
        RiscV64RelocationType::R_RISCV_TLS_GOT_HI20 => Some(RelaxationKind::TlsIeToLe),
        _ => None,
    }
}

// ============================================================================
// RISCV64_RELOCATION_TYPES — static relocation descriptor table
// ============================================================================

/// Complete table of all RISC-V 64 relocation types expressed as
/// architecture-agnostic [`RelocationType`] descriptors.
///
/// Returned by the `ArchCodegen::get_relocation_types()` implementation
/// for RISC-V 64 to provide the linker common infrastructure with
/// relocation metadata for validation and dispatch.
///
/// Each entry contains:
/// - `name`: canonical relocation name
/// - `value`: ELF `r_type` numeric value
/// - `is_pc_relative`: whether the relocation is PC-relative
/// - `size`: byte size of the patched region
pub const RISCV64_RELOCATION_TYPES: &[RelocationType] = &[
    RelocationType { name: "R_RISCV_NONE",         value:  0, is_pc_relative: false, size: 0 },
    RelocationType { name: "R_RISCV_32",           value:  1, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_64",           value:  2, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_RELATIVE",     value:  3, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_COPY",         value:  4, is_pc_relative: false, size: 0 },
    RelocationType { name: "R_RISCV_JUMP_SLOT",    value:  5, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_TLS_DTPMOD32", value:  6, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_TLS_DTPMOD64", value:  7, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_TLS_DTPREL32", value:  8, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_TLS_DTPREL64", value:  9, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_TLS_TPREL32",  value: 10, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_TLS_TPREL64",  value: 11, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_BRANCH",       value: 16, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_JAL",          value: 17, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_CALL",         value: 18, is_pc_relative: true,  size: 8 },
    RelocationType { name: "R_RISCV_CALL_PLT",     value: 19, is_pc_relative: true,  size: 8 },
    RelocationType { name: "R_RISCV_GOT_HI20",     value: 20, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_TLS_GOT_HI20", value: 21, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_TLS_GD_HI20",  value: 22, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_PCREL_HI20",   value: 23, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_PCREL_LO12_I", value: 24, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_PCREL_LO12_S", value: 25, is_pc_relative: true,  size: 4 },
    RelocationType { name: "R_RISCV_HI20",         value: 26, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_LO12_I",       value: 27, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_LO12_S",       value: 28, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_ADD8",          value: 33, is_pc_relative: false, size: 1 },
    RelocationType { name: "R_RISCV_ADD16",         value: 34, is_pc_relative: false, size: 2 },
    RelocationType { name: "R_RISCV_ADD32",         value: 35, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_ADD64",         value: 36, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_SUB8",          value: 37, is_pc_relative: false, size: 1 },
    RelocationType { name: "R_RISCV_SUB16",         value: 38, is_pc_relative: false, size: 2 },
    RelocationType { name: "R_RISCV_SUB32",         value: 39, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_SUB64",         value: 40, is_pc_relative: false, size: 8 },
    RelocationType { name: "R_RISCV_RVC_BRANCH",   value: 44, is_pc_relative: true,  size: 2 },
    RelocationType { name: "R_RISCV_RVC_JUMP",     value: 45, is_pc_relative: true,  size: 2 },
    RelocationType { name: "R_RISCV_RELAX",         value: 51, is_pc_relative: false, size: 0 },
    RelocationType { name: "R_RISCV_SET6",          value: 53, is_pc_relative: false, size: 1 },
    RelocationType { name: "R_RISCV_SET8",          value: 54, is_pc_relative: false, size: 1 },
    RelocationType { name: "R_RISCV_SET16",         value: 55, is_pc_relative: false, size: 2 },
    RelocationType { name: "R_RISCV_SET32",         value: 56, is_pc_relative: false, size: 4 },
    RelocationType { name: "R_RISCV_ALIGN",         value: 57, is_pc_relative: false, size: 0 },
];

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- split_hi_lo correctness --

    #[test]
    fn test_split_hi_lo_positive_lo() {
        // value = 0x12345678: lo = 0x678 (positive), no hi adjustment.
        let (hi, lo) = split_hi_lo(0x12345678);
        assert_eq!((hi as i64) * 4096 + (lo as i64), 0x12345678);
        assert_eq!(lo, 0x678);
        assert_eq!(hi, 0x12345);
    }

    #[test]
    fn test_split_hi_lo_negative_lo() {
        // value = 0x12345800: lo sign bit set → hi adjusted by +1.
        let (hi, lo) = split_hi_lo(0x12345800);
        assert_eq!((hi as i64) * 4096 + (lo as i64), 0x12345800);
        assert!(lo < 0, "lo should be negative due to sign extension");
        assert_eq!(hi, 0x12346);
    }

    #[test]
    fn test_split_hi_lo_zero() {
        let (hi, lo) = split_hi_lo(0);
        assert_eq!(hi, 0);
        assert_eq!(lo, 0);
    }

    #[test]
    fn test_split_hi_lo_negative() {
        let (hi, lo) = split_hi_lo(-1);
        assert_eq!((hi as i64) * 4096 + (lo as i64), -1);
    }

    #[test]
    fn test_split_hi_lo_small_positive() {
        let (hi, lo) = split_hi_lo(42);
        assert_eq!(hi, 0);
        assert_eq!(lo, 42);
    }

    #[test]
    fn test_split_hi_lo_boundary() {
        // 0x7FF = 2047, max positive 12-bit signed
        let (hi, lo) = split_hi_lo(0x7FF);
        assert_eq!(hi, 0);
        assert_eq!(lo, 0x7FF);
        // 0x800 = 2048 → lo = -2048, hi = 1
        let (hi, lo) = split_hi_lo(0x800);
        assert_eq!((hi as i64) * 4096 + (lo as i64), 0x800);
    }

    // -- ELF value round-trip --

    #[test]
    fn test_elf_value_roundtrip() {
        let all_types = [
            RiscV64RelocationType::R_RISCV_NONE,
            RiscV64RelocationType::R_RISCV_32,
            RiscV64RelocationType::R_RISCV_64,
            RiscV64RelocationType::R_RISCV_RELATIVE,
            RiscV64RelocationType::R_RISCV_COPY,
            RiscV64RelocationType::R_RISCV_JUMP_SLOT,
            RiscV64RelocationType::R_RISCV_BRANCH,
            RiscV64RelocationType::R_RISCV_JAL,
            RiscV64RelocationType::R_RISCV_CALL,
            RiscV64RelocationType::R_RISCV_CALL_PLT,
            RiscV64RelocationType::R_RISCV_GOT_HI20,
            RiscV64RelocationType::R_RISCV_PCREL_HI20,
            RiscV64RelocationType::R_RISCV_PCREL_LO12_I,
            RiscV64RelocationType::R_RISCV_PCREL_LO12_S,
            RiscV64RelocationType::R_RISCV_HI20,
            RiscV64RelocationType::R_RISCV_LO12_I,
            RiscV64RelocationType::R_RISCV_LO12_S,
            RiscV64RelocationType::R_RISCV_RELAX,
            RiscV64RelocationType::R_RISCV_ALIGN,
            RiscV64RelocationType::R_RISCV_RVC_BRANCH,
            RiscV64RelocationType::R_RISCV_RVC_JUMP,
        ];
        for rt in &all_types {
            let val = rt.to_elf_value();
            let parsed = RiscV64RelocationType::from_elf_value(val);
            assert_eq!(parsed, Some(*rt), "round-trip failed for {:?}", rt);
        }
    }

    #[test]
    fn test_from_elf_value_unknown() {
        assert_eq!(RiscV64RelocationType::from_elf_value(999), None);
        assert_eq!(RiscV64RelocationType::from_elf_value(12), None);
        assert_eq!(RiscV64RelocationType::from_elf_value(50), None);
    }

    // -- Property queries --

    #[test]
    fn test_pc_relative() {
        assert!(RiscV64RelocationType::R_RISCV_BRANCH.is_pc_relative());
        assert!(RiscV64RelocationType::R_RISCV_JAL.is_pc_relative());
        assert!(RiscV64RelocationType::R_RISCV_CALL.is_pc_relative());
        assert!(RiscV64RelocationType::R_RISCV_PCREL_HI20.is_pc_relative());
        assert!(RiscV64RelocationType::R_RISCV_RVC_JUMP.is_pc_relative());
        assert!(!RiscV64RelocationType::R_RISCV_32.is_pc_relative());
        assert!(!RiscV64RelocationType::R_RISCV_HI20.is_pc_relative());
        assert!(!RiscV64RelocationType::R_RISCV_RELAX.is_pc_relative());
    }

    #[test]
    fn test_relaxable() {
        assert!(RiscV64RelocationType::R_RISCV_CALL.is_relaxable());
        assert!(RiscV64RelocationType::R_RISCV_CALL_PLT.is_relaxable());
        assert!(RiscV64RelocationType::R_RISCV_GOT_HI20.is_relaxable());
        assert!(!RiscV64RelocationType::R_RISCV_BRANCH.is_relaxable());
        assert!(!RiscV64RelocationType::R_RISCV_PCREL_HI20.is_relaxable());
    }

    #[test]
    fn test_requires_got_plt() {
        assert!(RiscV64RelocationType::R_RISCV_GOT_HI20.requires_got_entry());
        assert!(RiscV64RelocationType::R_RISCV_TLS_GOT_HI20.requires_got_entry());
        assert!(!RiscV64RelocationType::R_RISCV_PCREL_HI20.requires_got_entry());
        assert!(RiscV64RelocationType::R_RISCV_CALL_PLT.requires_plt_entry());
        assert!(!RiscV64RelocationType::R_RISCV_CALL.requires_plt_entry());
    }

    // -- B-type relocation --

    #[test]
    fn test_apply_branch_zero() {
        // BEQ x0, x0, 0 → offset = 0 should leave immediate bits as 0.
        let mut bytes = [0x63, 0x00, 0x00, 0x00]; // BEQ (opcode 0x63)
        apply_relocation(RiscV64RelocationType::R_RISCV_BRANCH, &mut bytes, 0).unwrap();
        let instr = read_le32(&bytes);
        // Immediate bits should all be zero, opcode/funct3 preserved.
        assert_eq!(instr & 0xFE000F80, 0);
    }

    #[test]
    fn test_apply_branch_positive() {
        let mut bytes = [0x63, 0x00, 0x00, 0x00];
        // Offset = +8 (0b1000): imm[4:1]=0100, imm[10:5]=000000, imm[11]=0, imm[12]=0
        apply_relocation(RiscV64RelocationType::R_RISCV_BRANCH, &mut bytes, 8).unwrap();
        let instr = read_le32(&bytes);
        let extracted_4_1 = (instr >> 8) & 0xF;
        assert_eq!(extracted_4_1, 4); // bits 4:1 of 8 = 0b0100
    }

    #[test]
    fn test_apply_branch_overflow() {
        let mut bytes = [0x63, 0x00, 0x00, 0x00];
        let result = apply_relocation(
            RiscV64RelocationType::R_RISCV_BRANCH,
            &mut bytes,
            8192, // exceeds 13-bit range
        );
        assert!(matches!(result, Err(RelocationError::Overflow { .. })));
    }

    #[test]
    fn test_apply_branch_alignment() {
        let mut bytes = [0x63, 0x00, 0x00, 0x00];
        let result = apply_relocation(
            RiscV64RelocationType::R_RISCV_BRANCH,
            &mut bytes,
            3, // odd → not 2-byte aligned
        );
        assert!(matches!(result, Err(RelocationError::AlignmentError { .. })));
    }

    // -- J-type relocation --

    #[test]
    fn test_apply_jal_zero() {
        let mut bytes = [0x6F, 0x00, 0x00, 0x00]; // JAL (opcode 0x6F)
        apply_relocation(RiscV64RelocationType::R_RISCV_JAL, &mut bytes, 0).unwrap();
        let instr = read_le32(&bytes);
        assert_eq!(instr & 0xFFFFF000, 0);
    }

    #[test]
    fn test_apply_jal_overflow() {
        let mut bytes = [0x6F, 0x00, 0x00, 0x00];
        let result = apply_relocation(
            RiscV64RelocationType::R_RISCV_JAL,
            &mut bytes,
            0x200000, // exceeds 21-bit range (2^20 = 1048576)
        );
        assert!(matches!(result, Err(RelocationError::Overflow { .. })));
    }

    // -- CALL relocation (AUIPC + JALR) --

    #[test]
    fn test_apply_call_zero() {
        // AUIPC x1, 0 (U-type) + JALR x1, x1, 0 (I-type)
        let mut bytes = [
            0x97, 0x00, 0x00, 0x00,  // AUIPC (opcode 0x17 with rd=x1 → 0x97)
            0x67, 0x80, 0x00, 0x00,  // JALR  (opcode 0x67 with rd=x1, rs1=x1)
        ];
        apply_relocation(RiscV64RelocationType::R_RISCV_CALL, &mut bytes, 0).unwrap();
        let auipc = read_le32(&bytes[0..4]);
        let jalr = read_le32(&bytes[4..8]);
        assert_eq!(auipc & 0xFFFFF000, 0);
        assert_eq!(jalr & 0xFFF00000, 0);
    }

    // -- Data relocations --

    #[test]
    fn test_apply_32() {
        let mut bytes = [0x00, 0x00, 0x00, 0x00];
        apply_relocation(RiscV64RelocationType::R_RISCV_32, &mut bytes, 0x12345678).unwrap();
        assert_eq!(read_le32(&bytes), 0x12345678);
    }

    #[test]
    fn test_apply_64() {
        let mut bytes = [0x00; 8];
        apply_relocation(
            RiscV64RelocationType::R_RISCV_64,
            &mut bytes,
            0x123456789ABCDEF0_u64 as i64,
        ).unwrap();
        assert_eq!(read_le64(&bytes), 0x123456789ABCDEF0);
    }

    // -- ADD/SUB relocations --

    #[test]
    fn test_apply_add32() {
        let mut bytes = [0x10, 0x00, 0x00, 0x00]; // existing = 16
        apply_relocation(RiscV64RelocationType::R_RISCV_ADD32, &mut bytes, 4).unwrap();
        assert_eq!(read_le32(&bytes), 20);
    }

    #[test]
    fn test_apply_sub32() {
        let mut bytes = [0x10, 0x00, 0x00, 0x00]; // existing = 16
        apply_relocation(RiscV64RelocationType::R_RISCV_SUB32, &mut bytes, 4).unwrap();
        assert_eq!(read_le32(&bytes), 12);
    }

    // -- SET relocations --

    #[test]
    fn test_apply_set6() {
        let mut bytes = [0xFF]; // all bits set
        apply_relocation(RiscV64RelocationType::R_RISCV_SET6, &mut bytes, 0x15).unwrap();
        // Upper 2 bits preserved (0xC0), lower 6 = 0x15
        assert_eq!(bytes[0], 0xC0 | 0x15);
    }

    #[test]
    fn test_apply_set8() {
        let mut bytes = [0xFF];
        apply_relocation(RiscV64RelocationType::R_RISCV_SET8, &mut bytes, 0x42).unwrap();
        assert_eq!(bytes[0], 0x42);
    }

    // -- Relaxation targets --

    #[test]
    fn test_relaxation_target() {
        assert_eq!(
            relaxation_target(RiscV64RelocationType::R_RISCV_CALL),
            Some(RelaxationKind::CallToJal)
        );
        assert_eq!(
            relaxation_target(RiscV64RelocationType::R_RISCV_GOT_HI20),
            Some(RelaxationKind::GotToLocal)
        );
        assert_eq!(
            relaxation_target(RiscV64RelocationType::R_RISCV_TLS_GD_HI20),
            Some(RelaxationKind::TlsGdToLe)
        );
        assert_eq!(
            relaxation_target(RiscV64RelocationType::R_RISCV_TLS_GOT_HI20),
            Some(RelaxationKind::TlsIeToLe)
        );
        assert_eq!(
            relaxation_target(RiscV64RelocationType::R_RISCV_BRANCH),
            None
        );
    }

    // -- Static table integrity --

    #[test]
    fn test_relocation_table_consistency() {
        for entry in RISCV64_RELOCATION_TYPES {
            // Every table entry must correspond to a valid enum variant.
            let parsed = RiscV64RelocationType::from_elf_value(entry.value);
            assert!(
                parsed.is_some(),
                "table entry {} (value={}) has no enum variant",
                entry.name,
                entry.value,
            );
            let rt = parsed.unwrap();
            // Name must match.
            assert_eq!(rt.name(), entry.name);
            // PC-relative flag must match.
            assert_eq!(
                rt.is_pc_relative(),
                entry.is_pc_relative,
                "is_pc_relative mismatch for {}",
                entry.name,
            );
            // Byte size must match.
            assert_eq!(
                rt.byte_size(),
                entry.size,
                "byte_size mismatch for {}",
                entry.name,
            );
        }
    }

    // -- Unsupported type --

    #[test]
    fn test_unsupported_type() {
        let mut bytes = [0x00; 8];
        let result = apply_relocation(
            RiscV64RelocationType::R_RISCV_COPY,
            &mut bytes,
            0,
        );
        assert!(matches!(result, Err(RelocationError::UnsupportedType(_))));
    }
}
