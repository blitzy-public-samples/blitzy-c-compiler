// SPDX-License-Identifier: MIT
//
// src/backend/i686/assembler/relocations.rs
//
// i686 (IA-32) ELF relocation type definitions for the BCC built-in assembler
// and linker.
//
// This module defines every ELF relocation type needed for position-independent
// and position-dependent code emission on the i686 (32-bit x86) architecture.
// It is consumed by both the assembler (to record relocations during instruction
// encoding) and the linker (to apply relocations when producing final ELF
// executables and shared objects).
//
// The relocation types, ELF constant values, and computation formulas are
// derived from the System V Application Binary Interface — Intel386
// Architecture Processor Supplement (ELF specification for IA-32).
//
// ## Relocation Computation Legend
//
// The following symbols are used in relocation formulas throughout this module:
//
// - **S** — Symbol value (the resolved virtual address of the target symbol)
// - **A** — Addend (a signed constant stored **inline** at the relocation site
//   in the section data — i686 uses REL format, not RELA)
// - **P** — Place (the address of the storage unit being relocated, i.e., the
//   offset of the relocation site within the output section)
// - **B** — Base address (the base virtual address at which the shared object
//   is loaded; used for `R_386_RELATIVE` in position-independent executables)
// - **G** — GOT entry offset (the offset within the GOT of the entry for the
//   referenced symbol, relative to the GOT base)
// - **GOT** — Global Offset Table address (the virtual address of the
//   `_GLOBAL_OFFSET_TABLE_` symbol)
// - **L** — PLT entry address (the virtual address of the PLT stub for the
//   symbol)
//
// ## Critical Difference: REL vs RELA
//
// Unlike x86-64, which uses the **RELA** relocation format (with an explicit
// `r_addend` field in `Elf64_Rela`), the i686 architecture uses the **REL**
// format (`Elf32_Rel`). In REL format:
//
// - Each relocation entry is 8 bytes: `r_offset` (4 bytes) + `r_info` (4 bytes)
// - There is **no** separate addend field; the addend (**A**) is read from the
//   bytes already present at the relocation target location in the section data
// - The linker reads the inline value, computes the relocation, and writes the
//   result back to the same location
// - `.rel.text` and `.rel.dyn` sections are used instead of `.rela.text` and
//   `.rela.dyn`
//
// ## i686 PIC Pattern
//
// Unlike x86-64's RIP-relative addressing, the i686 architecture lacks a
// PC-relative data addressing mode. Position-independent code on i686 uses
// a thunk to establish the GOT base address:
//
// 1. `call __i686.get_pc_thunk.bx` — pushes the return address (current PC)
//    onto the stack and pops it into EBX (or another register)
// 2. `addl $_GLOBAL_OFFSET_TABLE_, %ebx` — adjusts EBX to point to the GOT
//    base, using an `R_386_GOTPC` relocation
// 3. Global data is then accessed via `R_386_GOT32` (through GOT) or
//    `R_386_GOTOFF` (direct offset from GOT base for local symbols)
//
// ## Zero-Dependency Mandate
//
// This module uses only standard Rust types. No external crates are imported.
// All relocation handling is implemented internally per the BCC standalone
// backend architecture (Section 0.7.7 of the Agent Action Plan).

use std::fmt;

use crate::backend::traits::RelocationType;

// ---------------------------------------------------------------------------
// I686RelocType — i386 ELF relocation type enumeration
// ---------------------------------------------------------------------------

/// Enumerates all i686 (IA-32) ELF relocation types supported by the BCC
/// assembler and linker.
///
/// Each variant corresponds to a specific relocation computation as defined by
/// the System V ABI Intel386 Architecture Processor Supplement. The variants
/// carry no data; metadata is accessed via the associated methods
/// ([`to_elf_value`], [`size`], [`is_pc_relative`], etc.).
///
/// # REL Format
///
/// All i686 relocations use the `Elf32_Rel` entry format (8 bytes: offset +
/// info) rather than the `Elf32_Rela` format used by x86-64. The addend is
/// stored inline at the relocation target within the section data.
///
/// # Relocation Size
///
/// All i686 relocations operate on 32-bit (4-byte) fields. There are no 8-bit,
/// 16-bit, or 64-bit relocation types in the standard i686 set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum I686RelocType {
    /// ELF value 0 — `R_386_NONE`: No relocation action required.
    /// Used as a placeholder or padding in relocation tables.
    R386None,

    /// ELF value 1 — `R_386_32`: Direct absolute 32-bit address.
    /// Formula: **S + A** → write 32-bit value.
    /// Used for absolute data references (e.g., function pointers, global
    /// variable addresses in `.data`).
    R386_32,

    /// ELF value 2 — `R_386_PC32`: PC-relative 32-bit signed offset.
    /// Formula: **S + A - P** → write 32-bit signed value.
    /// Used for direct function calls (`CALL rel32`) and relative branch
    /// targets in non-PIC code, as well as PIC internal calls.
    R386Pc32,

    /// ELF value 3 — `R_386_GOT32`: 32-bit offset to GOT entry.
    /// Formula: **G + A** → write 32-bit value.
    /// The offset is relative to the GOT base. Used for loading symbol
    /// addresses indirectly through the GOT in PIC code.
    R386Got32,

    /// ELF value 4 — `R_386_PLT32`: PLT-relative 32-bit offset.
    /// Formula: **L + A - P** → write 32-bit signed value.
    /// Used for external function calls through the PLT in PIC code,
    /// enabling lazy binding and symbol interposition.
    R386Plt32,

    /// ELF value 5 — `R_386_COPY`: Copy relocation for dynamic linking.
    /// The dynamic linker copies the symbol's data from the shared library
    /// into the executable's BSS segment. No static linker computation.
    R386Copy,

    /// ELF value 6 — `R_386_GLOB_DAT`: Set GOT entry to symbol address.
    /// Formula: **S** → write 32-bit absolute address into GOT slot.
    /// Used by the dynamic linker to populate GOT entries at load time.
    R386GlobDat,

    /// ELF value 7 — `R_386_JMP_SLOT`: Set PLT GOT entry for lazy binding.
    /// Formula: **S** → write 32-bit address into the GOT slot associated
    /// with a PLT stub. Used for lazy function resolution.
    R386JmpSlot,

    /// ELF value 8 — `R_386_RELATIVE`: Base-relative relocation.
    /// Formula: **B + A** → write 32-bit value.
    /// Used in shared objects for relocations relative to the load base
    /// address. The dynamic linker adds the base address to the inline
    /// addend at load time.
    R386Relative,

    /// ELF value 9 — `R_386_GOTOFF`: Offset from GOT base to symbol.
    /// Formula: **S + A - GOT** → write 32-bit value.
    /// Used in PIC code to access locally-defined symbols without GOT
    /// indirection. The symbol's distance from the GOT base is computed
    /// at link time.
    R386Gotoff,

    /// ELF value 10 — `R_386_GOTPC`: PC-relative offset to GOT base.
    /// Formula: **GOT + A - P** → write 32-bit value.
    /// Used in the PIC GOT-base establishment pattern:
    /// `addl $_GLOBAL_OFFSET_TABLE_, %ebx` after the thunk call.
    R386Gotpc,

    /// ELF value 11 — `R_386_32PLT`: Alternate PLT 32-bit relocation.
    /// Formula: **L + A** → write 32-bit value.
    /// Less common; used when an absolute reference to a PLT entry is
    /// needed rather than a PC-relative one.
    R386_32Plt,

    /// ELF value 24 — `R_386_TLS_GD_32`: TLS General Dynamic model.
    /// Used to implement `_Thread_local` variable access via the GD
    /// (General Dynamic) TLS model, which supports cross-DSO TLS access.
    R386TlsGd32,

    /// ELF value 34 — `R_386_TLS_LE_32`: TLS Local Exec model.
    /// Used for `_Thread_local` variables when the access is within the
    /// executable itself (local exec model), providing the most efficient
    /// TLS access pattern via a direct offset from the thread pointer.
    R386TlsLe32,
}

// ---------------------------------------------------------------------------
// I686RelocType — core implementation
// ---------------------------------------------------------------------------

impl I686RelocType {
    /// Returns the numeric ELF relocation constant (`R_386_*` value) for this
    /// relocation type.
    ///
    /// These values are defined by the System V ABI Intel386 Architecture
    /// Processor Supplement and are encoded into the `r_info` field of
    /// `Elf32_Rel` relocation entries via [`make_rel_info`].
    ///
    /// The return type is `u8` because all standard i686 relocation type
    /// values fit within one byte (0–34 for the types supported here).
    pub fn to_elf_value(&self) -> u8 {
        match self {
            Self::R386None => 0,
            Self::R386_32 => 1,
            Self::R386Pc32 => 2,
            Self::R386Got32 => 3,
            Self::R386Plt32 => 4,
            Self::R386Copy => 5,
            Self::R386GlobDat => 6,
            Self::R386JmpSlot => 7,
            Self::R386Relative => 8,
            Self::R386Gotoff => 9,
            Self::R386Gotpc => 10,
            Self::R386_32Plt => 11,
            Self::R386TlsGd32 => 24,
            Self::R386TlsLe32 => 34,
        }
    }

    /// Parses a numeric ELF relocation type value into the corresponding enum
    /// variant.
    ///
    /// Returns `None` if the value does not correspond to a relocation type
    /// supported by the BCC i686 assembler/linker. This is used when reading
    /// relocatable object files during the linking phase.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(I686RelocType::from_elf_value(2), Some(I686RelocType::R386Pc32));
    /// assert_eq!(I686RelocType::from_elf_value(99), None);
    /// ```
    pub fn from_elf_value(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::R386None),
            1 => Some(Self::R386_32),
            2 => Some(Self::R386Pc32),
            3 => Some(Self::R386Got32),
            4 => Some(Self::R386Plt32),
            5 => Some(Self::R386Copy),
            6 => Some(Self::R386GlobDat),
            7 => Some(Self::R386JmpSlot),
            8 => Some(Self::R386Relative),
            9 => Some(Self::R386Gotoff),
            10 => Some(Self::R386Gotpc),
            11 => Some(Self::R386_32Plt),
            24 => Some(Self::R386TlsGd32),
            34 => Some(Self::R386TlsLe32),
            _ => None,
        }
    }

    /// Returns `true` if this relocation type computes a PC-relative offset.
    ///
    /// PC-relative relocations subtract the relocation site address (**P**)
    /// from the computed value. On i686, the PC-relative relocations are:
    ///
    /// - `R_386_PC32` — direct PC-relative branch/call
    /// - `R_386_PLT32` — PLT-relative call (also subtracts P)
    /// - `R_386_GOTPC` — PC-relative distance to GOT base
    ///
    /// The linker must account for the fact that on i686, the PC points to
    /// the instruction *following* the relocation site (i.e., the addend
    /// typically compensates for the 4-byte field width).
    #[inline]
    pub fn is_pc_relative(&self) -> bool {
        matches!(
            self,
            Self::R386Pc32 | Self::R386Plt32 | Self::R386Gotpc
        )
    }

    /// Returns `true` if this relocation type produces a direct absolute
    /// address.
    ///
    /// Only `R_386_32` is a pure absolute relocation on i686. `R_386_32PLT`
    /// is also technically absolute (no P subtraction) but references a PLT
    /// entry rather than the symbol directly.
    #[inline]
    pub fn is_absolute(&self) -> bool {
        matches!(self, Self::R386_32)
    }

    /// Returns `true` if this relocation type requires a Global Offset Table
    /// (GOT) entry or involves the GOT base address.
    ///
    /// When the linker encounters one of these relocation types, it must:
    /// - For `R_386_GOT32`: allocate a GOT slot for the symbol and write the
    ///   slot offset
    /// - For `R_386_GOTOFF`: compute the symbol's offset from the GOT base
    /// - For `R_386_GOTPC`: compute the distance from the relocation site to
    ///   the GOT base
    /// - For `R_386_GLOB_DAT`: create a dynamic relocation to fill the GOT
    ///   slot at load time
    #[inline]
    pub fn requires_got(&self) -> bool {
        matches!(
            self,
            Self::R386Got32
                | Self::R386Gotoff
                | Self::R386Gotpc
                | Self::R386GlobDat
        )
    }

    /// Returns `true` if this relocation type requires a Procedure Linkage
    /// Table (PLT) entry for the referenced symbol.
    ///
    /// PLT entries provide lazy binding stubs for function calls in shared
    /// libraries. The linker creates a PLT stub and a corresponding GOT slot
    /// with an initial value pointing back to the PLT resolver.
    ///
    /// - `R_386_PLT32`: PC-relative call through PLT
    /// - `R_386_JMP_SLOT`: dynamic relocation for PLT GOT entry
    #[inline]
    pub fn requires_plt(&self) -> bool {
        matches!(self, Self::R386Plt32 | Self::R386JmpSlot)
    }

    /// Returns the byte size of the relocation field at the relocation site.
    ///
    /// All i686 relocations operate on 32-bit (4-byte) fields, with the
    /// exception of `R_386_NONE` and `R_386_COPY` which do not patch any
    /// data (size 0). This is a key difference from x86-64, which supports
    /// 8-bit, 16-bit, 32-bit, and 64-bit relocation fields.
    #[inline]
    pub fn size(&self) -> u8 {
        match self {
            Self::R386None | Self::R386Copy => 0,
            _ => 4,
        }
    }

    /// Returns `true` if this relocation is processed by the dynamic linker
    /// at load time rather than fully resolved by the static linker.
    ///
    /// Dynamic relocations appear in `.rel.dyn` and `.rel.plt` sections
    /// and are emitted when producing shared objects (`-shared`) or
    /// position-independent executables. The runtime loader (`ld-linux.so.2`)
    /// processes these at load time.
    ///
    /// Dynamic relocation types on i686:
    /// - `R_386_COPY` — copy data from shared library to executable BSS
    /// - `R_386_GLOB_DAT` — fill GOT entry with symbol address
    /// - `R_386_JMP_SLOT` — fill PLT GOT entry (lazy binding)
    /// - `R_386_RELATIVE` — add base address to inline addend
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        matches!(
            self,
            Self::R386Copy
                | Self::R386GlobDat
                | Self::R386JmpSlot
                | Self::R386Relative
        )
    }

    /// Returns the canonical name of this relocation type as it appears in
    /// ELF documentation and `readelf` output.
    ///
    /// This is the same string produced by the [`Display`] implementation,
    /// provided here as a `&'static str` for zero-allocation use in
    /// performance-sensitive paths (e.g., linker error messages, relocation
    /// overflow diagnostics).
    pub fn name(&self) -> &'static str {
        match self {
            Self::R386None => "R_386_NONE",
            Self::R386_32 => "R_386_32",
            Self::R386Pc32 => "R_386_PC32",
            Self::R386Got32 => "R_386_GOT32",
            Self::R386Plt32 => "R_386_PLT32",
            Self::R386Copy => "R_386_COPY",
            Self::R386GlobDat => "R_386_GLOB_DAT",
            Self::R386JmpSlot => "R_386_JMP_SLOT",
            Self::R386Relative => "R_386_RELATIVE",
            Self::R386Gotoff => "R_386_GOTOFF",
            Self::R386Gotpc => "R_386_GOTPC",
            Self::R386_32Plt => "R_386_32PLT",
            Self::R386TlsGd32 => "R_386_TLS_GD_32",
            Self::R386TlsLe32 => "R_386_TLS_LE_32",
        }
    }

    /// Applies this relocation to compute the final value to be written at
    /// the relocation site.
    ///
    /// # Parameters
    ///
    /// - `place` — The virtual address of the relocation site (**P**). For
    ///   relocatable output, this is the section-relative offset of the field
    ///   being patched.
    /// - `symbol_value` — The resolved value of the target symbol (**S**).
    ///   For `R_386_PLT32`, this is the PLT entry address (**L**). For
    ///   `R_386_GOT32`, this is the GOT entry offset (**G**).
    /// - `addend` — The addend (**A**), which on i686 is read from the bytes
    ///   at the relocation site before the fixup is applied (REL format).
    /// - `got_base` — The virtual address of the `_GLOBAL_OFFSET_TABLE_`
    ///   symbol (**GOT**). Used by `R_386_GOTOFF` and `R_386_GOTPC`.
    ///
    /// # Returns
    ///
    /// The computed 32-bit value to write at the relocation site. All
    /// arithmetic uses wrapping 32-bit semantics to handle address space
    /// wraparound correctly.
    ///
    /// # Relocation Formulas
    ///
    /// | Type            | Formula              | Description                    |
    /// |-----------------|----------------------|--------------------------------|
    /// | `R_386_NONE`    | 0                    | No-op                          |
    /// | `R_386_32`      | S + A                | Absolute 32-bit address        |
    /// | `R_386_PC32`    | S + A - P            | PC-relative 32-bit offset      |
    /// | `R_386_GOT32`   | G + A                | GOT entry offset               |
    /// | `R_386_PLT32`   | L + A - P            | PLT-relative 32-bit offset     |
    /// | `R_386_RELATIVE`| B + A                | Base-relative (B = symbol_value)|
    /// | `R_386_GOTOFF`  | S + A - GOT          | Symbol offset from GOT base    |
    /// | `R_386_GOTPC`   | GOT + A - P          | Distance to GOT base           |
    /// | `R_386_32PLT`   | L + A                | Absolute PLT entry address     |
    /// | `R_386_COPY`    | 0                    | No static computation          |
    /// | `R_386_GLOB_DAT`| S                    | Symbol address for GOT         |
    /// | `R_386_JMP_SLOT`| S                    | Symbol address for PLT GOT     |
    /// | TLS types       | S + A                | TLS offset computation         |
    pub fn apply(
        &self,
        place: u32,
        symbol_value: u32,
        addend: i32,
        got_base: u32,
    ) -> u32 {
        let s = symbol_value;
        let a = addend as u32;
        let p = place;
        let got = got_base;

        match self {
            // No relocation — return zero (field is not patched in practice,
            // but returning 0 for consistency)
            Self::R386None => 0,

            // R_386_32: S + A — absolute 32-bit address
            Self::R386_32 => s.wrapping_add(a),

            // R_386_PC32: S + A - P — PC-relative 32-bit offset
            Self::R386Pc32 => s.wrapping_add(a).wrapping_sub(p),

            // R_386_GOT32: G + A — GOT entry offset from GOT base
            // Here `symbol_value` is the GOT entry offset (G), not the
            // symbol's actual address. The GOT slot itself contains the
            // symbol's address.
            Self::R386Got32 => s.wrapping_add(a),

            // R_386_PLT32: L + A - P — PLT-relative 32-bit offset
            // Here `symbol_value` is the PLT entry address (L).
            Self::R386Plt32 => s.wrapping_add(a).wrapping_sub(p),

            // R_386_COPY: no static linker computation — the dynamic
            // linker performs the copy at load time
            Self::R386Copy => 0,

            // R_386_GLOB_DAT: S — symbol address written to GOT slot
            // (dynamic relocation; the static linker may pre-fill)
            Self::R386GlobDat => s,

            // R_386_JMP_SLOT: S — symbol address written to PLT GOT slot
            // (dynamic relocation for lazy binding resolution)
            Self::R386JmpSlot => s,

            // R_386_RELATIVE: B + A — base address + addend
            // In static linking context, `symbol_value` serves as B (base).
            // The dynamic linker adds the actual load base at runtime.
            Self::R386Relative => s.wrapping_add(a),

            // R_386_GOTOFF: S + A - GOT — symbol's offset from GOT base
            // Used for local data access in PIC code without GOT indirection.
            Self::R386Gotoff => s.wrapping_add(a).wrapping_sub(got),

            // R_386_GOTPC: GOT + A - P — PC-relative distance to GOT
            // Used in the `addl $_GLOBAL_OFFSET_TABLE_, %ebx` pattern
            // after the PIC thunk call.
            Self::R386Gotpc => got.wrapping_add(a).wrapping_sub(p),

            // R_386_32PLT: L + A — absolute PLT entry address
            // Similar to R_386_32 but references the PLT stub.
            Self::R386_32Plt => s.wrapping_add(a),

            // R_386_TLS_GD_32: TLS General Dynamic — offset for GD model
            // Computes the offset into the GOT for the TLS descriptor pair.
            Self::R386TlsGd32 => s.wrapping_add(a),

            // R_386_TLS_LE_32: TLS Local Exec — direct TP offset
            // Computes the negative offset from the thread pointer to the
            // TLS variable. The runtime formula is: TP - (S + A).
            Self::R386TlsLe32 => s.wrapping_add(a),
        }
    }

    /// Checks whether the computed relocation `value` fits within the
    /// relocation field without overflow.
    ///
    /// All i686 relocations write 32-bit values. For PC-relative relocations,
    /// the value is treated as signed (`i32` range: −2³¹ to 2³¹−1). For
    /// absolute relocations, the value is treated as unsigned (`u32` range:
    /// 0 to 2³²−1).
    ///
    /// `R_386_NONE` and `R_386_COPY` never overflow (they write no data),
    /// so this method always returns `true` for those types.
    ///
    /// # Parameters
    ///
    /// - `value` — The computed 64-bit relocation value (before truncation
    ///   to 32 bits). This is passed as `i64` to detect both positive and
    ///   negative overflow.
    ///
    /// # Returns
    ///
    /// `true` if the value fits in the relocation field; `false` if it
    /// overflows and the relocation cannot be applied without data loss.
    pub fn check_overflow(&self, value: i64) -> bool {
        match self {
            // No-op relocations never overflow
            Self::R386None | Self::R386Copy => true,

            // PC-relative relocations are signed 32-bit values
            Self::R386Pc32 | Self::R386Plt32 | Self::R386Gotpc => {
                value >= i32::MIN as i64 && value <= i32::MAX as i64
            }

            // Absolute and GOT/GOTOFF relocations: the 32-bit field is
            // interpreted as either signed or unsigned depending on context.
            // We accept any value that fits in either i32 or u32 range to
            // handle both signed offsets and unsigned addresses.
            Self::R386_32
            | Self::R386Got32
            | Self::R386GlobDat
            | Self::R386JmpSlot
            | Self::R386Relative
            | Self::R386Gotoff
            | Self::R386_32Plt
            | Self::R386TlsGd32
            | Self::R386TlsLe32 => {
                // Accept values in the union of i32 and u32 ranges:
                // [-2^31, 2^32 - 1]
                value >= i32::MIN as i64 && value <= u32::MAX as i64
            }
        }
    }

    /// Converts this i686-specific relocation type into the architecture-
    /// agnostic [`RelocationType`] descriptor used by the [`ArchCodegen`]
    /// trait interface.
    ///
    /// This enables the code generation driver and linker common
    /// infrastructure to work with relocations from any architecture
    /// through a uniform descriptor format.
    ///
    /// [`ArchCodegen`]: crate::backend::traits::ArchCodegen
    pub fn to_relocation_type(&self) -> RelocationType {
        RelocationType {
            name: self.name(),
            value: self.to_elf_value() as u32,
            is_pc_relative: self.is_pc_relative(),
            size: self.size(),
        }
    }
}

// ---------------------------------------------------------------------------
// Display — human-readable formatting for diagnostics
// ---------------------------------------------------------------------------

/// Human-readable display of i686 relocation type names for diagnostic output.
///
/// Produces the canonical ELF relocation name string (e.g., `"R_386_PC32"`,
/// `"R_386_PLT32"`), matching the format used by GNU `readelf -r` and
/// standard ELF tooling.
impl fmt::Display for I686RelocType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// From<I686RelocType> for RelocationType — trait integration
// ---------------------------------------------------------------------------

/// Converts an [`I686RelocType`] into the architecture-agnostic
/// [`RelocationType`] descriptor, allowing uniform relocation handling
/// across all target architectures in the BCC backend.
impl From<I686RelocType> for RelocationType {
    fn from(reloc: I686RelocType) -> Self {
        reloc.to_relocation_type()
    }
}

// ---------------------------------------------------------------------------
// REL Format Constants
// ---------------------------------------------------------------------------

/// Size of an `Elf32_Rel` relocation entry in bytes.
///
/// The i686 architecture uses the REL relocation format (not RELA):
///
/// ```text
/// struct Elf32_Rel {
///     Elf32_Addr r_offset;  // 4 bytes — offset of the relocation site
///     Elf32_Word r_info;    // 4 bytes — symbol index (24 bits) + type (8 bits)
/// };
/// // Total: 8 bytes (no r_addend field)
/// ```
///
/// This contrasts with x86-64's `Elf64_Rela` format, which is 24 bytes
/// (offset: 8, info: 8, addend: 8). The absence of a dedicated addend
/// field means the linker must read the addend from the section data
/// at the relocation offset before applying the fixup.
pub const REL_ENTRY_SIZE: usize = 8;

/// Constructs the `r_info` field of an `Elf32_Rel` entry from a symbol
/// table index and relocation type.
///
/// In the ELF32 format, `r_info` is composed as:
///
/// ```text
/// r_info = (symbol_index << 8) | reloc_type
/// ```
///
/// Where:
/// - `symbol_index` occupies the upper 24 bits (supporting up to 16M symbols)
/// - `reloc_type` occupies the lower 8 bits (the relocation type code)
///
/// This encoding differs from ELF64, where `r_info = (sym << 32) | type`
/// with a full 32-bit type field.
///
/// # Parameters
///
/// - `symbol_index` — Index of the symbol in the symbol table (`.symtab`
///   or `.dynsym`). Must be less than 2²⁴ (16,777,216).
/// - `reloc_type` — The [`I686RelocType`] to encode.
///
/// # Returns
///
/// The packed `r_info` value ready to be written into an `Elf32_Rel` entry.
///
/// # Examples
///
/// ```ignore
/// // Symbol at index 42, relocation type R_386_PC32 (value 2)
/// let info = make_rel_info(42, I686RelocType::R386Pc32);
/// assert_eq!(info, (42 << 8) | 2);
/// ```
#[inline]
pub fn make_rel_info(symbol_index: u32, reloc_type: I686RelocType) -> u32 {
    (symbol_index << 8) | (reloc_type.to_elf_value() as u32)
}

// ---------------------------------------------------------------------------
// PIC relocation selection helpers
// ---------------------------------------------------------------------------
// These free functions encapsulate the policy for choosing the appropriate
// relocation type based on PIC mode and instruction context. They are used
// by the i686 code generator and assembler to emit correct relocations
// without duplicating selection logic across multiple call sites.

/// Selects the appropriate relocation type for a function call instruction.
///
/// - In PIC mode (`is_pic == true`), function calls use `R_386_PLT32` so
///   that the call goes through the PLT stub, enabling lazy binding and
///   interposition in shared libraries.
/// - In non-PIC mode (`is_pic == false`), function calls use `R_386_PC32`
///   for a direct PC-relative branch to the target symbol.
///
/// Both relocation types produce a 32-bit signed PC-relative value and are
/// applied to `CALL rel32` instructions.
pub fn select_call_relocation(is_pic: bool) -> I686RelocType {
    if is_pic {
        I686RelocType::R386Plt32
    } else {
        I686RelocType::R386Pc32
    }
}

/// Selects the appropriate relocation type for a data/global variable
/// reference.
///
/// - In PIC mode (`is_pic == true`):
///   - `is_local == true`: Use `R_386_GOTOFF` for a direct offset from
///     the GOT base (more efficient, avoids GOT slot allocation).
///   - `is_local == false`: Use `R_386_GOT32` for an indirect load through
///     the GOT (required for symbols that may be interposed).
/// - In non-PIC mode (`is_pic == false`): Use `R_386_32` for a direct
///   absolute 32-bit address.
///
/// # Parameters
///
/// - `is_pic` — Whether position-independent code generation is active.
/// - `is_local` — Whether the referenced symbol is locally defined (same
///   translation unit or same DSO with hidden/protected visibility).
pub fn select_data_relocation(is_pic: bool, is_local: bool) -> I686RelocType {
    if is_pic {
        if is_local {
            I686RelocType::R386Gotoff
        } else {
            I686RelocType::R386Got32
        }
    } else {
        I686RelocType::R386_32
    }
}

/// Selects the relocation type for establishing the GOT base in PIC code.
///
/// The i686 PIC pattern requires computing the GOT base address in a
/// register (typically EBX). The standard sequence is:
///
/// ```asm
/// call __i686.get_pc_thunk.bx    ; pushes return PC, pops into EBX
/// addl $_GLOBAL_OFFSET_TABLE_, %ebx  ; R_386_GOTPC relocation
/// ```
///
/// This function always returns `R_386_GOTPC`, which computes
/// `GOT + A - P` to produce the distance from the instruction to the
/// GOT base.
#[inline]
pub fn select_got_base_relocation() -> I686RelocType {
    I686RelocType::R386Gotpc
}

// ---------------------------------------------------------------------------
// Utility: extract inline addend from section data
// ---------------------------------------------------------------------------

/// Reads the 32-bit inline addend from section data at the given offset.
///
/// In the i686 REL relocation format, the addend is stored directly in the
/// bytes at the relocation target within the section data. This function
/// extracts that value as a signed `i32` in little-endian byte order.
///
/// # Parameters
///
/// - `data` — The section data byte slice (e.g., `.text` content).
/// - `offset` — The byte offset of the relocation site within the section.
///
/// # Returns
///
/// The 32-bit signed addend value, or `0` if the offset is out of bounds
/// or there are fewer than 4 bytes remaining at the offset.
///
/// # Safety Note
///
/// This function does not panic on out-of-bounds access; it returns 0 as
/// a safe default. Callers should validate offsets before invoking
/// relocation application if strict error handling is desired.
pub fn read_inline_addend(data: &[u8], offset: usize) -> i32 {
    if offset + 4 > data.len() {
        return 0;
    }
    let bytes: [u8; 4] = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    i32::from_le_bytes(bytes)
}

/// Writes a 32-bit value to section data at the given offset in
/// little-endian byte order.
///
/// This is the counterpart to [`read_inline_addend`] — after computing
/// the relocation result via [`I686RelocType::apply`], the linker writes
/// the value back using this function.
///
/// # Parameters
///
/// - `data` — Mutable section data byte slice.
/// - `offset` — The byte offset of the relocation site within the section.
/// - `value` — The computed 32-bit relocation result.
///
/// # Returns
///
/// `true` if the write succeeded; `false` if the offset is out of bounds.
pub fn write_relocation_value(data: &mut [u8], offset: usize, value: u32) -> bool {
    if offset + 4 > data.len() {
        return false;
    }
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
    true
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All enum variants for exhaustive testing.
    fn all_variants() -> [I686RelocType; 14] {
        [
            I686RelocType::R386None,
            I686RelocType::R386_32,
            I686RelocType::R386Pc32,
            I686RelocType::R386Got32,
            I686RelocType::R386Plt32,
            I686RelocType::R386Copy,
            I686RelocType::R386GlobDat,
            I686RelocType::R386JmpSlot,
            I686RelocType::R386Relative,
            I686RelocType::R386Gotoff,
            I686RelocType::R386Gotpc,
            I686RelocType::R386_32Plt,
            I686RelocType::R386TlsGd32,
            I686RelocType::R386TlsLe32,
        ]
    }

    /// Verify that every variant round-trips through to_elf_value → from_elf_value.
    #[test]
    fn test_elf_value_round_trip() {
        for variant in &all_variants() {
            let elf_val = variant.to_elf_value();
            let parsed = I686RelocType::from_elf_value(elf_val);
            assert_eq!(
                parsed,
                Some(*variant),
                "Round-trip failed for {:?} (elf_value={})",
                variant,
                elf_val
            );
        }
    }

    /// Verify that unrecognized ELF values return None.
    #[test]
    fn test_from_elf_value_unknown() {
        assert_eq!(I686RelocType::from_elf_value(12), None);
        assert_eq!(I686RelocType::from_elf_value(13), None);
        assert_eq!(I686RelocType::from_elf_value(23), None);
        assert_eq!(I686RelocType::from_elf_value(25), None);
        assert_eq!(I686RelocType::from_elf_value(33), None);
        assert_eq!(I686RelocType::from_elf_value(35), None);
        assert_eq!(I686RelocType::from_elf_value(99), None);
        assert_eq!(I686RelocType::from_elf_value(u8::MAX), None);
    }

    /// Verify specific ELF constant values match the i386 ABI specification.
    #[test]
    fn test_elf_values_are_correct() {
        assert_eq!(I686RelocType::R386None.to_elf_value(), 0);
        assert_eq!(I686RelocType::R386_32.to_elf_value(), 1);
        assert_eq!(I686RelocType::R386Pc32.to_elf_value(), 2);
        assert_eq!(I686RelocType::R386Got32.to_elf_value(), 3);
        assert_eq!(I686RelocType::R386Plt32.to_elf_value(), 4);
        assert_eq!(I686RelocType::R386Copy.to_elf_value(), 5);
        assert_eq!(I686RelocType::R386GlobDat.to_elf_value(), 6);
        assert_eq!(I686RelocType::R386JmpSlot.to_elf_value(), 7);
        assert_eq!(I686RelocType::R386Relative.to_elf_value(), 8);
        assert_eq!(I686RelocType::R386Gotoff.to_elf_value(), 9);
        assert_eq!(I686RelocType::R386Gotpc.to_elf_value(), 10);
        assert_eq!(I686RelocType::R386_32Plt.to_elf_value(), 11);
        assert_eq!(I686RelocType::R386TlsGd32.to_elf_value(), 24);
        assert_eq!(I686RelocType::R386TlsLe32.to_elf_value(), 34);
    }

    /// Verify PC-relative classification.
    #[test]
    fn test_is_pc_relative() {
        // PC-relative types
        assert!(I686RelocType::R386Pc32.is_pc_relative());
        assert!(I686RelocType::R386Plt32.is_pc_relative());
        assert!(I686RelocType::R386Gotpc.is_pc_relative());

        // Non-PC-relative types
        assert!(!I686RelocType::R386None.is_pc_relative());
        assert!(!I686RelocType::R386_32.is_pc_relative());
        assert!(!I686RelocType::R386Got32.is_pc_relative());
        assert!(!I686RelocType::R386Copy.is_pc_relative());
        assert!(!I686RelocType::R386GlobDat.is_pc_relative());
        assert!(!I686RelocType::R386JmpSlot.is_pc_relative());
        assert!(!I686RelocType::R386Relative.is_pc_relative());
        assert!(!I686RelocType::R386Gotoff.is_pc_relative());
        assert!(!I686RelocType::R386_32Plt.is_pc_relative());
        assert!(!I686RelocType::R386TlsGd32.is_pc_relative());
        assert!(!I686RelocType::R386TlsLe32.is_pc_relative());
    }

    /// Verify absolute classification.
    #[test]
    fn test_is_absolute() {
        assert!(I686RelocType::R386_32.is_absolute());

        assert!(!I686RelocType::R386None.is_absolute());
        assert!(!I686RelocType::R386Pc32.is_absolute());
        assert!(!I686RelocType::R386Plt32.is_absolute());
        assert!(!I686RelocType::R386Got32.is_absolute());
        assert!(!I686RelocType::R386Copy.is_absolute());
        assert!(!I686RelocType::R386GlobDat.is_absolute());
        assert!(!I686RelocType::R386Relative.is_absolute());
        assert!(!I686RelocType::R386Gotoff.is_absolute());
        assert!(!I686RelocType::R386Gotpc.is_absolute());
    }

    /// Verify relocation field sizes.
    #[test]
    fn test_size() {
        // No-op relocations have zero size
        assert_eq!(I686RelocType::R386None.size(), 0);
        assert_eq!(I686RelocType::R386Copy.size(), 0);

        // All other i686 relocations are 32-bit (4 bytes)
        assert_eq!(I686RelocType::R386_32.size(), 4);
        assert_eq!(I686RelocType::R386Pc32.size(), 4);
        assert_eq!(I686RelocType::R386Got32.size(), 4);
        assert_eq!(I686RelocType::R386Plt32.size(), 4);
        assert_eq!(I686RelocType::R386GlobDat.size(), 4);
        assert_eq!(I686RelocType::R386JmpSlot.size(), 4);
        assert_eq!(I686RelocType::R386Relative.size(), 4);
        assert_eq!(I686RelocType::R386Gotoff.size(), 4);
        assert_eq!(I686RelocType::R386Gotpc.size(), 4);
        assert_eq!(I686RelocType::R386_32Plt.size(), 4);
        assert_eq!(I686RelocType::R386TlsGd32.size(), 4);
        assert_eq!(I686RelocType::R386TlsLe32.size(), 4);
    }

    /// Verify GOT requirement classification.
    #[test]
    fn test_requires_got() {
        assert!(I686RelocType::R386Got32.requires_got());
        assert!(I686RelocType::R386Gotoff.requires_got());
        assert!(I686RelocType::R386Gotpc.requires_got());
        assert!(I686RelocType::R386GlobDat.requires_got());

        assert!(!I686RelocType::R386None.requires_got());
        assert!(!I686RelocType::R386_32.requires_got());
        assert!(!I686RelocType::R386Pc32.requires_got());
        assert!(!I686RelocType::R386Plt32.requires_got());
        assert!(!I686RelocType::R386Copy.requires_got());
        assert!(!I686RelocType::R386JmpSlot.requires_got());
        assert!(!I686RelocType::R386Relative.requires_got());
        assert!(!I686RelocType::R386_32Plt.requires_got());
    }

    /// Verify PLT requirement classification.
    #[test]
    fn test_requires_plt() {
        assert!(I686RelocType::R386Plt32.requires_plt());
        assert!(I686RelocType::R386JmpSlot.requires_plt());

        assert!(!I686RelocType::R386None.requires_plt());
        assert!(!I686RelocType::R386_32.requires_plt());
        assert!(!I686RelocType::R386Pc32.requires_plt());
        assert!(!I686RelocType::R386Got32.requires_plt());
        assert!(!I686RelocType::R386Copy.requires_plt());
        assert!(!I686RelocType::R386GlobDat.requires_plt());
        assert!(!I686RelocType::R386Relative.requires_plt());
        assert!(!I686RelocType::R386Gotoff.requires_plt());
        assert!(!I686RelocType::R386Gotpc.requires_plt());
    }

    /// Verify dynamic relocation classification.
    #[test]
    fn test_is_dynamic() {
        assert!(I686RelocType::R386Copy.is_dynamic());
        assert!(I686RelocType::R386GlobDat.is_dynamic());
        assert!(I686RelocType::R386JmpSlot.is_dynamic());
        assert!(I686RelocType::R386Relative.is_dynamic());

        assert!(!I686RelocType::R386None.is_dynamic());
        assert!(!I686RelocType::R386_32.is_dynamic());
        assert!(!I686RelocType::R386Pc32.is_dynamic());
        assert!(!I686RelocType::R386Got32.is_dynamic());
        assert!(!I686RelocType::R386Plt32.is_dynamic());
        assert!(!I686RelocType::R386Gotoff.is_dynamic());
        assert!(!I686RelocType::R386Gotpc.is_dynamic());
        assert!(!I686RelocType::R386_32Plt.is_dynamic());
    }

    /// Verify Display formatting matches canonical ELF names.
    #[test]
    fn test_display() {
        assert_eq!(format!("{}", I686RelocType::R386None), "R_386_NONE");
        assert_eq!(format!("{}", I686RelocType::R386_32), "R_386_32");
        assert_eq!(format!("{}", I686RelocType::R386Pc32), "R_386_PC32");
        assert_eq!(format!("{}", I686RelocType::R386Got32), "R_386_GOT32");
        assert_eq!(format!("{}", I686RelocType::R386Plt32), "R_386_PLT32");
        assert_eq!(format!("{}", I686RelocType::R386Copy), "R_386_COPY");
        assert_eq!(format!("{}", I686RelocType::R386GlobDat), "R_386_GLOB_DAT");
        assert_eq!(format!("{}", I686RelocType::R386JmpSlot), "R_386_JMP_SLOT");
        assert_eq!(format!("{}", I686RelocType::R386Relative), "R_386_RELATIVE");
        assert_eq!(format!("{}", I686RelocType::R386Gotoff), "R_386_GOTOFF");
        assert_eq!(format!("{}", I686RelocType::R386Gotpc), "R_386_GOTPC");
        assert_eq!(format!("{}", I686RelocType::R386_32Plt), "R_386_32PLT");
        assert_eq!(format!("{}", I686RelocType::R386TlsGd32), "R_386_TLS_GD_32");
        assert_eq!(format!("{}", I686RelocType::R386TlsLe32), "R_386_TLS_LE_32");
    }

    /// Verify name() returns the same string as Display.
    #[test]
    fn test_name_matches_display() {
        for variant in &all_variants() {
            assert_eq!(
                variant.name(),
                format!("{}", variant),
                "name() and Display disagree for {:?}",
                variant
            );
        }
    }

    /// Verify R_386_32 relocation formula: S + A.
    #[test]
    fn test_apply_r386_32() {
        // S=0x08048000, A=4
        let result = I686RelocType::R386_32.apply(0, 0x0804_8000, 4, 0);
        assert_eq!(result, 0x0804_8004);

        // S=0, A=0
        let result = I686RelocType::R386_32.apply(0, 0, 0, 0);
        assert_eq!(result, 0);

        // Wrapping addition
        let result = I686RelocType::R386_32.apply(0, 0xFFFF_FFFF, 1, 0);
        assert_eq!(result, 0);
    }

    /// Verify R_386_PC32 relocation formula: S + A - P.
    #[test]
    fn test_apply_r386_pc32() {
        // S=0x08048100, A=-4, P=0x08048050
        // Result = 0x08048100 + (-4 as u32) - 0x08048050
        //        = 0x08048100 + 0xFFFFFFFC - 0x08048050
        //        = 0x000000AC
        let result = I686RelocType::R386Pc32.apply(0x0804_8050, 0x0804_8100, -4, 0);
        assert_eq!(result, 0xAC);

        // S=P, A=0 → result = 0
        let result = I686RelocType::R386Pc32.apply(0x1000, 0x1000, 0, 0);
        assert_eq!(result, 0);
    }

    /// Verify R_386_GOT32 relocation formula: G + A.
    #[test]
    fn test_apply_r386_got32() {
        // GOT entry offset (G) = 0x10, A = 0
        let result = I686RelocType::R386Got32.apply(0, 0x10, 0, 0);
        assert_eq!(result, 0x10);

        // G=0x20, A=4
        let result = I686RelocType::R386Got32.apply(0, 0x20, 4, 0);
        assert_eq!(result, 0x24);
    }

    /// Verify R_386_PLT32 relocation formula: L + A - P.
    #[test]
    fn test_apply_r386_plt32() {
        // PLT entry (L) = 0x08049000, A = -4, P = 0x08048050
        let result = I686RelocType::R386Plt32.apply(0x0804_8050, 0x0804_9000, -4, 0);
        assert_eq!(result, 0x0FAC);
    }

    /// Verify R_386_GOTOFF relocation formula: S + A - GOT.
    #[test]
    fn test_apply_r386_gotoff() {
        // S=0x08049100, A=0, GOT=0x08049000
        let result = I686RelocType::R386Gotoff.apply(0, 0x0804_9100, 0, 0x0804_9000);
        assert_eq!(result, 0x100);

        // S < GOT (negative offset wraps)
        let result = I686RelocType::R386Gotoff.apply(0, 0x0804_8F00, 0, 0x0804_9000);
        // 0x08048F00 - 0x08049000 = -0x100 = 0xFFFFFF00
        assert_eq!(result, 0xFFFF_FF00);
    }

    /// Verify R_386_GOTPC relocation formula: GOT + A - P.
    #[test]
    fn test_apply_r386_gotpc() {
        // GOT=0x08049000, A=2, P=0x08048010
        // Result = 0x08049000 + 2 - 0x08048010 = 0xFF2
        let result = I686RelocType::R386Gotpc.apply(0x0804_8010, 0, 2, 0x0804_9000);
        assert_eq!(result, 0x0FF2);
    }

    /// Verify R_386_RELATIVE relocation formula: B + A.
    #[test]
    fn test_apply_r386_relative() {
        // B (base) = 0x08040000, A = 0x100
        let result = I686RelocType::R386Relative.apply(0, 0x0804_0000, 0x100, 0);
        assert_eq!(result, 0x0804_0100);
    }

    /// Verify R_386_GLOB_DAT and R_386_JMP_SLOT return S.
    #[test]
    fn test_apply_dynamic_relocations() {
        let sym = 0x0804_8000;
        assert_eq!(I686RelocType::R386GlobDat.apply(0, sym, 0, 0), sym);
        assert_eq!(I686RelocType::R386JmpSlot.apply(0, sym, 0, 0), sym);
    }

    /// Verify R_386_NONE and R_386_COPY return 0.
    #[test]
    fn test_apply_noop_relocations() {
        assert_eq!(I686RelocType::R386None.apply(0x1000, 0x2000, 10, 0x3000), 0);
        assert_eq!(I686RelocType::R386Copy.apply(0x1000, 0x2000, 10, 0x3000), 0);
    }

    /// Verify overflow detection for PC-relative relocations.
    #[test]
    fn test_check_overflow_pc_relative() {
        // Values within i32 range
        assert!(I686RelocType::R386Pc32.check_overflow(0));
        assert!(I686RelocType::R386Pc32.check_overflow(i32::MAX as i64));
        assert!(I686RelocType::R386Pc32.check_overflow(i32::MIN as i64));
        assert!(I686RelocType::R386Pc32.check_overflow(100));
        assert!(I686RelocType::R386Pc32.check_overflow(-100));

        // Values outside i32 range
        assert!(!I686RelocType::R386Pc32.check_overflow(i32::MAX as i64 + 1));
        assert!(!I686RelocType::R386Pc32.check_overflow(i32::MIN as i64 - 1));
        assert!(!I686RelocType::R386Pc32.check_overflow(i64::MAX));
        assert!(!I686RelocType::R386Pc32.check_overflow(i64::MIN));

        // Same for PLT32
        assert!(I686RelocType::R386Plt32.check_overflow(i32::MAX as i64));
        assert!(!I686RelocType::R386Plt32.check_overflow(i32::MAX as i64 + 1));

        // Same for GOTPC
        assert!(I686RelocType::R386Gotpc.check_overflow(i32::MIN as i64));
        assert!(!I686RelocType::R386Gotpc.check_overflow(i32::MIN as i64 - 1));
    }

    /// Verify overflow detection for absolute relocations.
    #[test]
    fn test_check_overflow_absolute() {
        // Accepts union of i32 and u32 ranges: [-2^31, 2^32-1]
        assert!(I686RelocType::R386_32.check_overflow(0));
        assert!(I686RelocType::R386_32.check_overflow(u32::MAX as i64));
        assert!(I686RelocType::R386_32.check_overflow(i32::MIN as i64));
        assert!(I686RelocType::R386_32.check_overflow(i32::MAX as i64));

        // Outside the combined range
        assert!(!I686RelocType::R386_32.check_overflow(u32::MAX as i64 + 1));
        assert!(!I686RelocType::R386_32.check_overflow(i32::MIN as i64 - 1));
    }

    /// Verify R_386_NONE and R_386_COPY never overflow.
    #[test]
    fn test_check_overflow_noop() {
        assert!(I686RelocType::R386None.check_overflow(i64::MAX));
        assert!(I686RelocType::R386None.check_overflow(i64::MIN));
        assert!(I686RelocType::R386Copy.check_overflow(i64::MAX));
        assert!(I686RelocType::R386Copy.check_overflow(i64::MIN));
    }

    /// Verify REL_ENTRY_SIZE constant.
    #[test]
    fn test_rel_entry_size() {
        assert_eq!(REL_ENTRY_SIZE, 8);
    }

    /// Verify make_rel_info encoding.
    #[test]
    fn test_make_rel_info() {
        // Symbol index 42, type R_386_PC32 (2)
        let info = make_rel_info(42, I686RelocType::R386Pc32);
        assert_eq!(info, (42 << 8) | 2);

        // Symbol index 0, type R_386_NONE (0)
        let info = make_rel_info(0, I686RelocType::R386None);
        assert_eq!(info, 0);

        // Symbol index 1, type R_386_32 (1)
        let info = make_rel_info(1, I686RelocType::R386_32);
        assert_eq!(info, (1 << 8) | 1);

        // Symbol index 100, type R_386_PLT32 (4)
        let info = make_rel_info(100, I686RelocType::R386Plt32);
        assert_eq!(info, (100 << 8) | 4);

        // Symbol index 0xFFFFFF (max 24-bit), type R_386_RELATIVE (8)
        let info = make_rel_info(0x00FF_FFFF, I686RelocType::R386Relative);
        assert_eq!(info, (0x00FF_FFFF << 8) | 8);

        // Verify extraction: type = info & 0xFF, sym = info >> 8
        let info = make_rel_info(7, I686RelocType::R386Gotoff);
        assert_eq!(info & 0xFF, 9); // R_386_GOTOFF = 9
        assert_eq!(info >> 8, 7); // symbol index = 7
    }

    /// Verify to_relocation_type conversion.
    #[test]
    fn test_to_relocation_type() {
        let rt = I686RelocType::R386Pc32.to_relocation_type();
        assert_eq!(rt.name, "R_386_PC32");
        assert_eq!(rt.value, 2);
        assert!(rt.is_pc_relative);
        assert_eq!(rt.size, 4);

        let rt = I686RelocType::R386_32.to_relocation_type();
        assert_eq!(rt.name, "R_386_32");
        assert_eq!(rt.value, 1);
        assert!(!rt.is_pc_relative);
        assert_eq!(rt.size, 4);

        let rt = I686RelocType::R386None.to_relocation_type();
        assert_eq!(rt.name, "R_386_NONE");
        assert_eq!(rt.value, 0);
        assert!(!rt.is_pc_relative);
        assert_eq!(rt.size, 0);
    }

    /// Verify From<I686RelocType> for RelocationType conversion.
    #[test]
    fn test_from_trait_conversion() {
        let rt: RelocationType = I686RelocType::R386Plt32.into();
        assert_eq!(rt.name, "R_386_PLT32");
        assert_eq!(rt.value, 4);
        assert!(rt.is_pc_relative);
        assert_eq!(rt.size, 4);
    }

    /// Verify PIC call relocation selection.
    #[test]
    fn test_select_call_relocation() {
        assert_eq!(select_call_relocation(true), I686RelocType::R386Plt32);
        assert_eq!(select_call_relocation(false), I686RelocType::R386Pc32);
    }

    /// Verify PIC data relocation selection.
    #[test]
    fn test_select_data_relocation() {
        // PIC, local → GOTOFF
        assert_eq!(
            select_data_relocation(true, true),
            I686RelocType::R386Gotoff
        );
        // PIC, external → GOT32
        assert_eq!(
            select_data_relocation(true, false),
            I686RelocType::R386Got32
        );
        // Non-PIC → absolute R_386_32
        assert_eq!(
            select_data_relocation(false, true),
            I686RelocType::R386_32
        );
        assert_eq!(
            select_data_relocation(false, false),
            I686RelocType::R386_32
        );
    }

    /// Verify GOT base relocation selection.
    #[test]
    fn test_select_got_base_relocation() {
        assert_eq!(select_got_base_relocation(), I686RelocType::R386Gotpc);
    }

    /// Verify inline addend reading from section data.
    #[test]
    fn test_read_inline_addend() {
        // Little-endian: [0xFC, 0xFF, 0xFF, 0xFF] = -4 as i32
        let data: Vec<u8> = vec![0xFC, 0xFF, 0xFF, 0xFF];
        assert_eq!(read_inline_addend(&data, 0), -4);

        // Positive value: [0x04, 0x00, 0x00, 0x00] = 4
        let data: Vec<u8> = vec![0x04, 0x00, 0x00, 0x00];
        assert_eq!(read_inline_addend(&data, 0), 4);

        // Zero
        let data: Vec<u8> = vec![0x00, 0x00, 0x00, 0x00];
        assert_eq!(read_inline_addend(&data, 0), 0);

        // Reading at offset within larger buffer
        let data: Vec<u8> = vec![0xAA, 0xBB, 0x10, 0x00, 0x00, 0x00, 0xCC];
        assert_eq!(read_inline_addend(&data, 2), 0x10);

        // Out-of-bounds returns 0
        let data: Vec<u8> = vec![0xFF, 0xFF];
        assert_eq!(read_inline_addend(&data, 0), 0);

        // Exactly at boundary (offset + 4 > len)
        let data: Vec<u8> = vec![0xFF, 0xFF, 0xFF];
        assert_eq!(read_inline_addend(&data, 0), 0);

        // Empty slice
        assert_eq!(read_inline_addend(&[], 0), 0);
    }

    /// Verify relocation value writing to section data.
    #[test]
    fn test_write_relocation_value() {
        let mut data = vec![0u8; 8];

        // Write at offset 0
        assert!(write_relocation_value(&mut data, 0, 0x0804_8000));
        assert_eq!(&data[0..4], &[0x00, 0x80, 0x04, 0x08]);

        // Write at offset 4
        assert!(write_relocation_value(&mut data, 4, 0xDEAD_BEEF));
        assert_eq!(&data[4..8], &[0xEF, 0xBE, 0xAD, 0xDE]);

        // Out-of-bounds write returns false and doesn't modify data
        let mut small = vec![0u8; 3];
        assert!(!write_relocation_value(&mut small, 0, 0x1234));
        assert_eq!(small, vec![0u8; 3]);

        // Offset past end
        let mut data2 = vec![0u8; 8];
        assert!(!write_relocation_value(&mut data2, 6, 0x1234));
    }

    /// Verify that all enum variants are Clone, Copy, and can be compared.
    #[test]
    fn test_derive_traits() {
        let a = I686RelocType::R386Pc32;
        let b = a; // Copy
        let c = a; // Copy (I686RelocType is Copy)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(a, I686RelocType::R386_32);
    }

    /// Verify Debug formatting works for all variants.
    #[test]
    fn test_debug_formatting() {
        for variant in &all_variants() {
            let debug_str = format!("{:?}", variant);
            assert!(!debug_str.is_empty());
        }
    }

    /// Verify Hash works (via the derive) by inserting into a HashSet.
    #[test]
    fn test_hash_impl() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for variant in &all_variants() {
            set.insert(*variant);
        }
        assert_eq!(set.len(), 14);
    }

    /// End-to-end test: read inline addend, apply relocation, write result.
    #[test]
    fn test_end_to_end_relocation_application() {
        // Simulate a R_386_PC32 relocation:
        // Section data has an inline addend of -4 at offset 10.
        // Place (P) = 0x0804_800A, Symbol (S) = 0x0804_8100
        let mut section_data = vec![0u8; 20];
        // Write -4 as inline addend at offset 10
        write_relocation_value(&mut section_data, 10, (-4_i32) as u32);

        // Read the inline addend
        let addend = read_inline_addend(&section_data, 10);
        assert_eq!(addend, -4);

        // Apply the relocation
        let place = 0x0804_800A_u32;
        let symbol = 0x0804_8100_u32;
        let result = I686RelocType::R386Pc32.apply(place, symbol, addend, 0);

        // Expected: S + A - P = 0x08048100 + (-4 as u32) - 0x0804800A
        //         = 0x08048100 + 0xFFFFFFFC - 0x0804800A = 0xF2
        assert_eq!(result, 0xF2);

        // Check overflow before writing
        assert!(I686RelocType::R386Pc32.check_overflow(result as i64));

        // Write the result back
        assert!(write_relocation_value(&mut section_data, 10, result));

        // Verify the written value
        assert_eq!(read_inline_addend(&section_data, 10), result as i32);
    }
}
