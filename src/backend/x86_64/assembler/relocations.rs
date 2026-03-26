//! x86-64 ELF relocation type definitions and relocation record construction.
//!
//! This module is the foundational relocation infrastructure for the x86-64
//! backend assembler. It provides:
//!
//! - [`X86_64RelocationType`] — All x86-64 ELF relocation type constants
//!   matching the official *System V Application Binary Interface, AMD64
//!   Architecture Processor Supplement*.
//! - [`RelocationEntry`] — Assembler-level relocation records collected
//!   during instruction encoding, later converted to ELF `Elf64_Rela`
//!   entries for `.rela.text` sections.
//! - [`RelocComputation`] — Relocation value computation formulas and
//!   overflow checking used by the linker during relocation application.
//!
//! # Integration
//!
//! - Imported by `encoder.rs` when encoding instructions that reference
//!   external symbols.
//! - Imported by `mod.rs` when building the `.rela.text` ELF section.
//! - Imported by `src/backend/x86_64/linker/relocations.rs` during
//!   relocation application in the linker.
//! - Relocation entries are serialized into the ELF `.rela.text` section
//!   by [`crate::backend::elf_writer_common::ElfWriter`].
//!
//! # RELA Format
//!
//! x86-64 uses RELA relocations (with explicit addend), NOT REL relocations.
//! Every relocation entry carries: offset, symbol index, type, and addend.

use crate::backend::elf_writer_common::Relocation;

// ===========================================================================
// X86_64RelocationType — ELF Relocation Type Constants
// ===========================================================================

/// x86-64 ELF relocation types.
///
/// Values match the official x86-64 ELF ABI specification.
/// See: *System V Application Binary Interface, AMD64 Architecture
/// Processor Supplement*, Table 4.10.
///
/// The `#[repr(u32)]` attribute ensures that each variant's discriminant
/// is exactly the ELF relocation type value, enabling direct `as u32`
/// conversion without lookup tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum X86_64RelocationType {
    /// `R_X86_64_NONE` (0) — No relocation (placeholder).
    None = 0,

    /// `R_X86_64_64` (1) — 64-bit absolute address: S + A.
    ///
    /// Used for absolute 64-bit references (e.g., `.quad symbol`).
    R64 = 1,

    /// `R_X86_64_PC32` (2) — 32-bit PC-relative: S + A - P.
    ///
    /// Used for direct `call`/`jmp` within ±2 GiB, and local symbol
    /// references in position-independent code.
    Pc32 = 2,

    /// `R_X86_64_GOT32` (3) — 32-bit GOT offset: G + A.
    ///
    /// Offset into the Global Offset Table.
    Got32 = 3,

    /// `R_X86_64_PLT32` (4) — 32-bit PLT-relative: L + A - P.
    ///
    /// Used for function calls through the Procedure Linkage Table.
    Plt32 = 4,

    /// `R_X86_64_COPY` (5) — Copy relocation.
    ///
    /// Copies a symbol's value from a shared library into the
    /// executable's data segment at load time.
    Copy = 5,

    /// `R_X86_64_GLOB_DAT` (6) — GOT entry absolute address: S.
    ///
    /// The dynamic linker creates a GOT entry and fills it with
    /// the symbol's absolute address at load time.
    GlobDat = 6,

    /// `R_X86_64_JUMP_SLOT` (7) — PLT jump slot: S.
    ///
    /// Used for lazy binding: the GOT entry initially points to
    /// the PLT resolver stub and is patched on first call.
    JumpSlot = 7,

    /// `R_X86_64_RELATIVE` (8) — Relative relocation: B + A.
    ///
    /// Base address + addend, used for position-independent data
    /// relocations that do not reference a specific symbol.
    Relative = 8,

    /// `R_X86_64_GOTPCREL` (9) — 32-bit PC-relative GOT entry: G + GOT + A - P.
    ///
    /// Used for PIC global variable access:
    /// `mov rax, [rip + sym@GOTPCREL]`.
    GotPcRel = 9,

    /// `R_X86_64_32` (10) — 32-bit absolute (zero-extended): S + A.
    ///
    /// For 32-bit absolute addresses that are zero-extended to 64 bits.
    Abs32 = 10,

    /// `R_X86_64_32S` (11) — 32-bit absolute (sign-extended): S + A.
    ///
    /// For 32-bit absolute addresses in sign-extending contexts
    /// (e.g., `mov` with 32-bit immediate in 64-bit mode).
    Abs32S = 11,

    /// `R_X86_64_16` (12) — 16-bit absolute: S + A.
    Abs16 = 12,

    /// `R_X86_64_PC16` (13) — 16-bit PC-relative: S + A - P.
    Pc16 = 13,

    /// `R_X86_64_8` (14) — 8-bit absolute: S + A.
    Abs8 = 14,

    /// `R_X86_64_PC8` (15) — 8-bit PC-relative: S + A - P.
    Pc8 = 15,

    /// `R_X86_64_GOTPCRELX` (41) — Relaxable GOTPCREL: G + GOT + A - P.
    ///
    /// Used for `mov eax, [rip + sym@GOTPCREL]` without REX prefix.
    /// The linker may relax (optimize) this to a direct `LEA` if
    /// the symbol is defined locally.
    GotPcRelX = 41,

    /// `R_X86_64_REX_GOTPCRELX` (42) — Relaxable GOTPCREL with REX: G + GOT + A - P.
    ///
    /// Same as [`GotPcRelX`](Self::GotPcRelX) but the instruction
    /// carries a REX prefix. Used for `mov rax, [rip + sym@GOTPCREL]`
    /// with REX.W.
    RexGotPcRelX = 42,
}

// ===========================================================================
// X86_64RelocationType — Conversion and Metadata
// ===========================================================================

impl X86_64RelocationType {
    /// Get the ELF relocation type value as a `u32`.
    ///
    /// Because the enum has `#[repr(u32)]`, this is a zero-cost cast.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Attempt to construct a relocation type from a raw `u32` ELF value.
    ///
    /// Returns `None` for values that do not correspond to a known
    /// x86-64 relocation type handled by this compiler.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::R64),
            2 => Some(Self::Pc32),
            3 => Some(Self::Got32),
            4 => Some(Self::Plt32),
            5 => Some(Self::Copy),
            6 => Some(Self::GlobDat),
            7 => Some(Self::JumpSlot),
            8 => Some(Self::Relative),
            9 => Some(Self::GotPcRel),
            10 => Some(Self::Abs32),
            11 => Some(Self::Abs32S),
            12 => Some(Self::Abs16),
            13 => Some(Self::Pc16),
            14 => Some(Self::Abs8),
            15 => Some(Self::Pc8),
            41 => Some(Self::GotPcRelX),
            42 => Some(Self::RexGotPcRelX),
            _ => None,
        }
    }

    /// Get the relocation field size in bytes.
    ///
    /// This is the number of bytes at the relocation site that will be
    /// patched when the relocation is applied.
    #[inline]
    pub fn size(self) -> u8 {
        match self {
            Self::None => 0,
            // 64-bit field
            Self::R64 | Self::Relative => 8,
            // 32-bit field
            Self::Pc32
            | Self::Got32
            | Self::Plt32
            | Self::GotPcRel
            | Self::Abs32
            | Self::Abs32S
            | Self::GotPcRelX
            | Self::RexGotPcRelX => 4,
            // 16-bit field
            Self::Abs16 | Self::Pc16 => 2,
            // 8-bit field
            Self::Abs8 | Self::Pc8 => 1,
            // Copy/GlobDat/JumpSlot operate on pointer-width (64-bit on x86-64)
            Self::Copy | Self::GlobDat | Self::JumpSlot => 8,
        }
    }

    /// Returns `true` if the relocation is PC-relative.
    ///
    /// PC-relative relocations compute their value using the formula
    /// `result = ... - P` where P is the address of the relocation site.
    #[inline]
    pub fn is_pc_relative(self) -> bool {
        matches!(
            self,
            Self::Pc32
                | Self::Plt32
                | Self::GotPcRel
                | Self::Pc16
                | Self::Pc8
                | Self::GotPcRelX
                | Self::RexGotPcRelX
        )
    }

    /// Returns `true` if the relocation involves the Global Offset Table.
    #[inline]
    pub fn is_got_relative(self) -> bool {
        matches!(
            self,
            Self::Got32 | Self::GotPcRel | Self::GlobDat | Self::GotPcRelX | Self::RexGotPcRelX
        )
    }

    /// Returns `true` if the relocation involves the Procedure Linkage Table.
    #[inline]
    pub fn is_plt_relative(self) -> bool {
        matches!(self, Self::Plt32 | Self::JumpSlot)
    }

    /// Returns `true` if the linker may relax (optimize) this relocation.
    ///
    /// Relaxable relocations allow the linker to convert GOT-indirect
    /// accesses into direct PC-relative accesses when the target symbol
    /// is defined locally within the same link unit.
    #[inline]
    pub fn is_relaxable(self) -> bool {
        matches!(self, Self::GotPcRelX | Self::RexGotPcRelX)
    }

    /// Get the canonical human-readable ELF name for this relocation type.
    ///
    /// The returned string matches the constant names used by `readelf -r`.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "R_X86_64_NONE",
            Self::R64 => "R_X86_64_64",
            Self::Pc32 => "R_X86_64_PC32",
            Self::Got32 => "R_X86_64_GOT32",
            Self::Plt32 => "R_X86_64_PLT32",
            Self::Copy => "R_X86_64_COPY",
            Self::GlobDat => "R_X86_64_GLOB_DAT",
            Self::JumpSlot => "R_X86_64_JUMP_SLOT",
            Self::Relative => "R_X86_64_RELATIVE",
            Self::GotPcRel => "R_X86_64_GOTPCREL",
            Self::Abs32 => "R_X86_64_32",
            Self::Abs32S => "R_X86_64_32S",
            Self::Abs16 => "R_X86_64_16",
            Self::Pc16 => "R_X86_64_PC16",
            Self::Abs8 => "R_X86_64_8",
            Self::Pc8 => "R_X86_64_PC8",
            Self::GotPcRelX => "R_X86_64_GOTPCRELX",
            Self::RexGotPcRelX => "R_X86_64_REX_GOTPCRELX",
        }
    }
}

// ===========================================================================
// Display for X86_64RelocationType
// ===========================================================================

impl std::fmt::Display for X86_64RelocationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ===========================================================================
// RelocationEntry — Assembler-Level Relocation Record
// ===========================================================================

/// A relocation entry produced during x86-64 instruction assembly.
///
/// This struct represents an assembler-level relocation that will be
/// converted to an `Elf64_Rela` entry in the `.rela.text` (or other
/// `.rela.*`) section of the output ELF file.
///
/// # Fields
///
/// | Field    | Description |
/// |----------|-------------|
/// | `offset` | Byte position within the section of the field to patch |
/// | `symbol` | Name of the referenced symbol (resolved to index later) |
/// | `rel_type` | x86-64 relocation type determining the computation |
/// | `addend`  | Explicit addend (RELA format); -4 is standard for PC-rel |
/// | `section` | Section this relocation belongs to (e.g., ".text") |
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Offset within the section (e.g., `.text`) where the relocation
    /// applies. This is the byte position of the 4-byte (or 8-byte)
    /// field to be patched by the linker.
    pub offset: u64,

    /// Name of the referenced symbol. During ELF writing, this name is
    /// resolved to a symbol table index via the string table.
    pub symbol: String,

    /// The relocation type, determining how the linker computes the
    /// final patched value.
    pub rel_type: X86_64RelocationType,

    /// Explicit addend value (RELA format).
    ///
    /// For PC-relative relocations this typically accounts for the
    /// distance from the relocation field to the end of the instruction.
    /// The standard value is -4 for 32-bit PC-relative fields at the
    /// end of an instruction.
    pub addend: i64,

    /// The section this relocation belongs to (e.g., `".text"`,
    /// `".data"`, `".got"`, `".got.plt"`).
    pub section: String,
}

// ===========================================================================
// RelocationEntry — Constructors
// ===========================================================================

impl RelocationEntry {
    /// Create a new relocation entry with all fields specified explicitly.
    pub fn new(
        offset: u64,
        symbol: String,
        rel_type: X86_64RelocationType,
        addend: i64,
        section: String,
    ) -> Self {
        Self {
            offset,
            symbol,
            rel_type,
            addend,
            section,
        }
    }

    /// Create a PC-relative 32-bit relocation for a direct `call` or `jmp`.
    ///
    /// Selects `R_X86_64_PLT32` when `use_plt` is true (PIC external
    /// calls), or `R_X86_64_PC32` otherwise (non-PIC local calls).
    /// The addend is set to -4 to account for the 4-byte relocation
    /// field consumed by the instruction pointer before reaching the
    /// target.
    pub fn pc32_call(offset: u64, symbol: String, use_plt: bool) -> Self {
        Self {
            offset,
            symbol,
            rel_type: if use_plt {
                X86_64RelocationType::Plt32
            } else {
                X86_64RelocationType::Pc32
            },
            addend: -4,
            section: ".text".to_string(),
        }
    }

    /// Create a 64-bit absolute relocation (`R_X86_64_64`).
    ///
    /// Used for `.quad symbol` in data sections and absolute 64-bit
    /// function pointers.
    pub fn abs64(offset: u64, symbol: String, addend: i64, section: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::R64,
            addend,
            section,
        }
    }

    /// Create a 32-bit absolute relocation, zero-extended (`R_X86_64_32`).
    pub fn abs32(offset: u64, symbol: String, addend: i64, section: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::Abs32,
            addend,
            section,
        }
    }

    /// Create a 32-bit absolute relocation, sign-extended (`R_X86_64_32S`).
    ///
    /// Used for 32-bit immediate operands that reference symbols in
    /// contexts where the value is sign-extended to 64 bits.
    pub fn abs32s(offset: u64, symbol: String, addend: i64, section: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::Abs32S,
            addend,
            section,
        }
    }

    /// Create a GOT PC-relative relocation (`R_X86_64_GOTPCREL`).
    ///
    /// Used for PIC global variable access:
    /// `mov rax, [rip + symbol@GOTPCREL]`.
    /// Addend is -4 (standard PC-relative adjustment).
    pub fn got_pcrel(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::GotPcRel,
            addend: -4,
            section: ".text".to_string(),
        }
    }

    /// Create a relaxable GOT PC-relative relocation without REX
    /// (`R_X86_64_GOTPCRELX`).
    ///
    /// Used for `mov eax, [rip + symbol@GOTPCREL]` (32-bit operand).
    /// The linker may relax this to a direct `LEA` if the symbol is
    /// defined locally.
    pub fn got_pcrelx(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::GotPcRelX,
            addend: -4,
            section: ".text".to_string(),
        }
    }

    /// Create a relaxable GOT PC-relative relocation with REX prefix
    /// (`R_X86_64_REX_GOTPCRELX`).
    ///
    /// Used for `mov rax, [rip + symbol@GOTPCREL]` (64-bit operand
    /// with REX.W).
    pub fn rex_got_pcrelx(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::RexGotPcRelX,
            addend: -4,
            section: ".text".to_string(),
        }
    }

    /// Create a PLT-relative relocation (`R_X86_64_PLT32`).
    ///
    /// Used for `call symbol@PLT` in position-independent code.
    /// Addend is -4 (standard PC-relative adjustment).
    pub fn plt32(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::Plt32,
            addend: -4,
            section: ".text".to_string(),
        }
    }

    /// Create a `GLOB_DAT` relocation (`R_X86_64_GLOB_DAT`).
    ///
    /// Used by the dynamic linker to fill GOT entries with the
    /// resolved absolute address of a global symbol.
    pub fn glob_dat(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::GlobDat,
            addend: 0,
            section: ".got".to_string(),
        }
    }

    /// Create a `JUMP_SLOT` relocation (`R_X86_64_JUMP_SLOT`).
    ///
    /// Used for PLT lazy binding entries. The `.got.plt` slot
    /// initially points to the PLT resolver stub and is patched
    /// by the dynamic linker on first call.
    pub fn jump_slot(offset: u64, symbol: String) -> Self {
        Self {
            offset,
            symbol,
            rel_type: X86_64RelocationType::JumpSlot,
            addend: 0,
            section: ".got.plt".to_string(),
        }
    }

    /// Create a `RELATIVE` relocation (`R_X86_64_RELATIVE`).
    ///
    /// Used for position-independent data pointers that require
    /// base address adjustment at load time. `RELATIVE` relocations
    /// do not reference a named symbol.
    pub fn relative(offset: u64, addend: i64, section: String) -> Self {
        Self {
            offset,
            symbol: String::new(),
            rel_type: X86_64RelocationType::Relative,
            addend,
            section,
        }
    }
}

// ===========================================================================
// RelocationEntry — ELF Conversion
// ===========================================================================

impl RelocationEntry {
    /// Convert this assembler-level relocation entry to the ELF
    /// writer's [`Relocation`] struct.
    ///
    /// The `sym_index` parameter is the resolved symbol table index,
    /// determined by the caller (typically the assembler driver in
    /// `mod.rs`) after the symbol table has been finalized.
    pub fn to_elf_relocation(&self, sym_index: u32) -> Relocation {
        Relocation {
            offset: self.offset,
            sym_index,
            rel_type: self.rel_type.as_u32(),
            addend: self.addend,
        }
    }
}

// ===========================================================================
// Display for RelocationEntry
// ===========================================================================

impl std::fmt::Display for RelocationEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}+{:#x}: {} '{}' {:+}",
            self.section,
            self.offset,
            self.rel_type.name(),
            self.symbol,
            self.addend
        )
    }
}

// ===========================================================================
// RelocComputation — Relocation Value Computation
// ===========================================================================

/// Relocation computation formulas for x86-64 ELF.
///
/// Implements the value computation and overflow checking specified by
/// the *System V ABI AMD64 Processor Supplement*.
///
/// # Legend
///
/// | Symbol | Meaning |
/// |--------|---------|
/// | S      | Symbol value (resolved address) |
/// | A      | Addend from the relocation entry |
/// | P      | Position (address of the relocation site) |
/// | B      | Base address of the shared object |
/// | G      | Offset of the symbol's GOT entry from the GOT base |
/// | GOT    | Address of the Global Offset Table |
/// | L      | Address of the PLT entry for the symbol |
pub struct RelocComputation;

impl RelocComputation {
    /// Compute the relocation value to be written at the relocation site.
    ///
    /// # Arguments
    ///
    /// * `rel_type` — The x86-64 relocation type.
    /// * `symbol_value` — Resolved absolute address of the symbol (S).
    /// * `addend` — Explicit addend from the relocation entry (A).
    /// * `relocation_address` — Address of the relocation field (P).
    /// * `got_address` — Base address of the GOT (if applicable).
    /// * `got_entry_offset` — Offset of the symbol's GOT entry from GOT base (G).
    /// * `plt_entry_address` — Address of the PLT entry (L, if applicable).
    /// * `base_address` — Base address of the shared object (B).
    ///
    /// # Returns
    ///
    /// The computed value (as `i64`) to be written at the relocation site.
    /// The caller is responsible for truncating to the correct field width.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        rel_type: X86_64RelocationType,
        symbol_value: u64,
        addend: i64,
        relocation_address: u64,
        got_address: Option<u64>,
        got_entry_offset: Option<u64>,
        plt_entry_address: Option<u64>,
        base_address: u64,
    ) -> i64 {
        let s = symbol_value as i64;
        let a = addend;
        let p = relocation_address as i64;

        match rel_type {
            // No-op relocation.
            X86_64RelocationType::None => 0,

            // S + A — 64-bit absolute.
            X86_64RelocationType::R64 => s.wrapping_add(a),

            // S + A - P — 32-bit PC-relative.
            X86_64RelocationType::Pc32 => s.wrapping_add(a).wrapping_sub(p),

            // G + A — 32-bit GOT offset.
            X86_64RelocationType::Got32 => {
                let g = got_entry_offset.unwrap_or(0) as i64;
                g.wrapping_add(a)
            }

            // L + A - P — 32-bit PLT-relative.
            // Falls back to S + A - P when no PLT entry exists.
            X86_64RelocationType::Plt32 => {
                let l = plt_entry_address.unwrap_or(symbol_value) as i64;
                l.wrapping_add(a).wrapping_sub(p)
            }

            // S — absolute symbol value (GOT entry fill).
            X86_64RelocationType::GlobDat => s,

            // S — absolute symbol value (PLT GOT slot fill).
            X86_64RelocationType::JumpSlot => s,

            // B + A — base-relative (no symbol).
            X86_64RelocationType::Relative => (base_address as i64).wrapping_add(a),

            // G + GOT + A - P — PC-relative GOT access.
            X86_64RelocationType::GotPcRel
            | X86_64RelocationType::GotPcRelX
            | X86_64RelocationType::RexGotPcRelX => {
                let got = got_address.unwrap_or(0) as i64;
                let g = got_entry_offset.unwrap_or(0) as i64;
                g.wrapping_add(got).wrapping_add(a).wrapping_sub(p)
            }

            // S + A — 32-bit absolute (zero-extended).
            X86_64RelocationType::Abs32 => s.wrapping_add(a),

            // S + A — 32-bit absolute (sign-extended).
            X86_64RelocationType::Abs32S => s.wrapping_add(a),

            // S + A — 16-bit absolute.
            X86_64RelocationType::Abs16 => s.wrapping_add(a),

            // S + A - P — 16-bit PC-relative.
            X86_64RelocationType::Pc16 => s.wrapping_add(a).wrapping_sub(p),

            // S + A — 8-bit absolute.
            X86_64RelocationType::Abs8 => s.wrapping_add(a),

            // S + A - P — 8-bit PC-relative.
            X86_64RelocationType::Pc8 => s.wrapping_add(a).wrapping_sub(p),

            // Copy: no computation — handled specially by the dynamic linker.
            X86_64RelocationType::Copy => 0,
        }
    }

    /// Check whether a computed relocation value fits in the target
    /// field size, returning an error message on overflow.
    ///
    /// # Arguments
    ///
    /// * `rel_type` — The relocation type (determines the field width).
    /// * `value` — The computed relocation value to check.
    /// * `symbol_name` — Symbol name for the error message.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` describing the overflow when the value
    /// exceeds the representable range of the target field.
    pub fn check_overflow(
        rel_type: X86_64RelocationType,
        value: i64,
        symbol_name: &str,
    ) -> Result<(), String> {
        let fits = match rel_type {
            // Signed 32-bit: value must be representable as i32.
            X86_64RelocationType::Pc32
            | X86_64RelocationType::Plt32
            | X86_64RelocationType::GotPcRel
            | X86_64RelocationType::GotPcRelX
            | X86_64RelocationType::RexGotPcRelX
            | X86_64RelocationType::Abs32S => value >= i32::MIN as i64 && value <= i32::MAX as i64,

            // Unsigned 32-bit: value must be representable as u32.
            X86_64RelocationType::Abs32 | X86_64RelocationType::Got32 => {
                value >= 0 && value <= u32::MAX as i64
            }

            // Unsigned 16-bit.
            X86_64RelocationType::Abs16 => value >= 0 && value <= u16::MAX as i64,

            // Signed 16-bit.
            X86_64RelocationType::Pc16 => value >= i16::MIN as i64 && value <= i16::MAX as i64,

            // Unsigned 8-bit.
            X86_64RelocationType::Abs8 => value >= 0 && value <= u8::MAX as i64,

            // Signed 8-bit.
            X86_64RelocationType::Pc8 => value >= i8::MIN as i64 && value <= i8::MAX as i64,

            // 64-bit relocations and special types always fit in their
            // target fields (64-bit or not applicable).
            _ => true,
        };

        if fits {
            Ok(())
        } else {
            Err(format!(
                "relocation {} overflow for symbol '{}': value {:#x} does not fit in {} bytes",
                rel_type.name(),
                symbol_name,
                value,
                rel_type.size()
            ))
        }
    }
}
