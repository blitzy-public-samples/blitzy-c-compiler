//! # i686 Relocation Application
//!
//! Implements architecture-specific relocation patching for all R_386_* relocation
//! types. This module is used by the i686 built-in ELF linker to resolve and apply
//! relocations during the linking process.
//!
//! ## Supported Relocation Types
//!
//! | Type | Value | Calculation | Description |
//! |------|-------|-------------|-------------|
//! | R_386_NONE | 0 | — | No relocation |
//! | R_386_32 | 1 | S + A | Absolute 32-bit address |
//! | R_386_PC32 | 2 | S + A - P | PC-relative 32-bit |
//! | R_386_GOT32 | 3 | G + A | GOT entry offset |
//! | R_386_PLT32 | 4 | L + A - P | PLT entry PC-relative |
//! | R_386_COPY | 5 | — | Copy symbol data |
//! | R_386_GLOB_DAT | 6 | S | Global data (GOT fill) |
//! | R_386_JMP_SLOT | 7 | S | PLT jump slot |
//! | R_386_RELATIVE | 8 | B + A | Base-relative (PIC) |
//! | R_386_GOTOFF | 9 | S + A - GOT | Offset from GOT base |
//! | R_386_GOTPC | 10 | GOT + A - P | GOT address PC-relative |
//! | R_386_32PLT | 11 | L + A | Absolute PLT address |
//! | R_386_16 | 20 | S + A | Absolute 16-bit |
//! | R_386_PC16 | 21 | S + A - P | PC-relative 16-bit |
//! | R_386_8 | 22 | S + A | Absolute 8-bit |
//! | R_386_PC8 | 23 | S + A - P | PC-relative 8-bit |
//! | R_386_SIZE32 | 38 | Z + A | Symbol size + addend |
//!
//! ## Legend
//! - **S** = Symbol value (final resolved address)
//! - **A** = Addend (from relocation entry)
//! - **P** = Patch address (where relocation is applied)
//! - **G** = GOT entry offset for the symbol
//! - **L** = PLT entry address for the symbol
//! - **B** = Base address of the shared object
//! - **GOT** = Address of the Global Offset Table
//! - **Z** = Symbol size
//!
//! ## Key Differences from x86-64 Relocations
//! - All values are 32-bit (not 64-bit)
//! - No RIP-relative addressing (R_X86_64_PC32 has a different semantic)
//! - GOT access uses `[ebx + offset]`, not `[rip + offset]`
//! - Uses `Elf32_Rel` (8 bytes, implicit addend read from section data)
//!   or `Elf32_Rela` (12 bytes, explicit addend)
//! - R_386_GOTPC is used instead of R_X86_64_GOTPCRELX
//! - No R_386_64 equivalent — maximum relocation width is 32 bits
//!
//! ## Zero-Dependency Mandate
//! No external crates. Only `std` and `crate::` references.

use crate::backend::linker_common::relocation::{
    compute_absolute, compute_got_relative, compute_pc_relative, fits_signed, fits_unsigned,
    read_le, sign_extend, write_le, RelocCategory, RelocationError, RelocationHandler,
    ResolvedRelocation,
};
use crate::backend::traits::RelocationTypeInfo;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// R_386_* Relocation Type Constants (ELF32 i386 ABI)
// ---------------------------------------------------------------------------

/// R_386_NONE — No relocation.
pub const R_386_NONE: u32 = 0;

/// R_386_32 — S + A (absolute 32-bit address).
///
/// Direct 32-bit relocation. The linker computes the symbol's absolute
/// virtual address plus the addend and writes the result as a 32-bit
/// little-endian value at the relocation site.
pub const R_386_32: u32 = 1;

/// R_386_PC32 — S + A - P (PC-relative 32-bit).
///
/// 32-bit PC-relative relocation. Used for relative addressing in CALL
/// and JMP instructions. The result is the signed 32-bit distance from
/// the relocation site to the symbol.
pub const R_386_PC32: u32 = 2;

/// R_386_GOT32 — G + A (GOT entry offset from GOT base).
///
/// Offset of the symbol's GOT entry from the beginning of the GOT.
/// Used in combination with EBX (GOT pointer) for PIC data access:
///   `mov eax, [ebx + symbol@GOT]`
pub const R_386_GOT32: u32 = 3;

/// R_386_PLT32 — L + A - P (PLT entry PC-relative).
///
/// PC-relative reference to the symbol's PLT entry. Used for PIC
/// function calls: `call symbol@PLT`.
pub const R_386_PLT32: u32 = 4;

/// R_386_COPY — Copy symbol data from shared object.
///
/// Used by the dynamic linker to copy data from a shared object into
/// the executable's `.bss` section. The linker creates a R_386_COPY
/// relocation for global data symbols imported from shared libraries.
pub const R_386_COPY: u32 = 5;

/// R_386_GLOB_DAT — S (fill GOT entry with symbol address).
///
/// Dynamic relocation: the dynamic linker fills the GOT entry with
/// the symbol's absolute address at load time.
pub const R_386_GLOB_DAT: u32 = 6;

/// R_386_JMP_SLOT — S (PLT jump slot).
///
/// Dynamic relocation for lazy binding: the dynamic linker patches
/// the GOT.PLT entry when the function is first called.
pub const R_386_JMP_SLOT: u32 = 7;

/// R_386_RELATIVE — B + A (base address + addend).
///
/// Dynamic relocation for PIC: the dynamic linker adds the base
/// address of the shared object to the addend to compute the final
/// absolute address. Used for pointer data in PIC shared objects.
pub const R_386_RELATIVE: u32 = 8;

/// R_386_GOTOFF — S + A - GOT (offset from GOT base).
///
/// Signed offset from the GOT base to the symbol. Used for accessing
/// local data in PIC code without a GOT entry.
pub const R_386_GOTOFF: u32 = 9;

/// R_386_GOTPC — GOT + A - P (GOT address PC-relative).
///
/// PC-relative address of the GOT itself. Used to compute the GOT
/// base address at runtime (typically in `__x86.get_pc_thunk.bx`).
pub const R_386_GOTPC: u32 = 10;

/// R_386_32PLT — L + A (absolute PLT address, rarely used).
pub const R_386_32PLT: u32 = 11;

/// R_386_16 — S + A (absolute 16-bit address, rarely used).
pub const R_386_16: u32 = 20;

/// R_386_PC16 — S + A - P (PC-relative 16-bit, rarely used).
pub const R_386_PC16: u32 = 21;

/// R_386_8 — S + A (absolute 8-bit address, rarely used).
pub const R_386_8: u32 = 22;

/// R_386_PC8 — S + A - P (PC-relative 8-bit, rarely used).
pub const R_386_PC8: u32 = 23;

/// R_386_SIZE32 — Z + A (symbol size + addend).
pub const R_386_SIZE32: u32 = 38;

// ---------------------------------------------------------------------------
// I686RelocationHandler — RelocationHandler trait implementation
// ---------------------------------------------------------------------------

/// i686-specific relocation handler implementing the [`RelocationHandler`] trait.
///
/// Handles all R_386_* relocation types for the i686 built-in ELF linker,
/// including static linking relocations (R_386_32, R_386_PC32) and dynamic
/// linking relocations (R_386_GOT32, R_386_PLT32, R_386_GLOB_DAT,
/// R_386_JMP_SLOT, R_386_RELATIVE, R_386_GOTOFF, R_386_GOTPC).
///
/// This handler is stateless — all relocation logic is purely functional,
/// dispatching on the relocation type code.
pub struct I686RelocationHandler {
    // No state needed — handler is stateless, purely dispatches on relocation type.
}

impl I686RelocationHandler {
    /// Create a new i686 relocation handler.
    pub fn new() -> Self {
        I686RelocationHandler {}
    }
}

impl Default for I686RelocationHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl RelocationHandler for I686RelocationHandler {
    /// Classify a relocation type into a [`RelocCategory`].
    ///
    /// Maps each R_386_* type to the appropriate category for
    /// architecture-agnostic processing in the common linker framework.
    fn classify(&self, rel_type: u32) -> RelocCategory {
        match rel_type {
            R_386_NONE => RelocCategory::Other,
            R_386_32 | R_386_16 | R_386_8 => RelocCategory::Absolute,
            R_386_PC32 | R_386_PC16 | R_386_PC8 => RelocCategory::PcRelative,
            R_386_GOT32 | R_386_GOTOFF | R_386_GOTPC => RelocCategory::GotRelative,
            R_386_PLT32 | R_386_32PLT => RelocCategory::Plt,
            R_386_GLOB_DAT => RelocCategory::GotEntry,
            R_386_JMP_SLOT => RelocCategory::Plt,
            // R_386_RELATIVE is base-relative but classified as Absolute
            // because it results in an absolute address after base addition.
            R_386_RELATIVE => RelocCategory::Absolute,
            R_386_COPY => RelocCategory::Other,
            R_386_SIZE32 => RelocCategory::Other,
            _ => RelocCategory::Other,
        }
    }

    /// Get the canonical ELF name of a relocation type.
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match rel_type {
            R_386_NONE => "R_386_NONE",
            R_386_32 => "R_386_32",
            R_386_PC32 => "R_386_PC32",
            R_386_GOT32 => "R_386_GOT32",
            R_386_PLT32 => "R_386_PLT32",
            R_386_COPY => "R_386_COPY",
            R_386_GLOB_DAT => "R_386_GLOB_DAT",
            R_386_JMP_SLOT => "R_386_JMP_SLOT",
            R_386_RELATIVE => "R_386_RELATIVE",
            R_386_GOTOFF => "R_386_GOTOFF",
            R_386_GOTPC => "R_386_GOTPC",
            R_386_32PLT => "R_386_32PLT",
            R_386_16 => "R_386_16",
            R_386_PC16 => "R_386_PC16",
            R_386_8 => "R_386_8",
            R_386_PC8 => "R_386_PC8",
            R_386_SIZE32 => "R_386_SIZE32",
            _ => "R_386_UNKNOWN",
        }
    }

    /// Get the patch size in bytes for a relocation type.
    ///
    /// Returns 4 for standard 32-bit relocations, 2 for 16-bit, 1 for 8-bit,
    /// and 0 for no-op or symbol-determined sizes.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        match rel_type {
            R_386_NONE => 0,
            R_386_32 | R_386_PC32 | R_386_GOT32 | R_386_PLT32 | R_386_GLOB_DAT | R_386_JMP_SLOT
            | R_386_RELATIVE | R_386_GOTOFF | R_386_GOTPC | R_386_32PLT | R_386_SIZE32 => 4, // 32-bit
            R_386_16 | R_386_PC16 => 2, // 16-bit
            R_386_8 | R_386_PC8 => 1,   // 8-bit
            R_386_COPY => 0,            // size determined by symbol
            _ => 0,
        }
    }

    /// Apply a single resolved relocation by computing the final value and
    /// writing it into the section data buffer.
    ///
    /// This is the core relocation patching function. For each relocation type,
    /// it computes the appropriate formula (see module-level documentation),
    /// validates the result fits in the bit width, and writes the value in
    /// little-endian byte order.
    ///
    /// # Errors
    ///
    /// Returns [`RelocationError::Overflow`] if the computed value does not fit
    /// in the relocation's bit width. Returns [`RelocationError::UndefinedSymbol`]
    /// if a GOT entry is required but not provided. Returns
    /// [`RelocationError::UnsupportedType`] for unknown relocation types.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let offset = rel.patch_offset as usize;
        let s = rel.symbol_value; // S: symbol value (final address)
        let a = rel.addend; // A: addend
        let p = rel.patch_address; // P: patch address

        match rel.rel_type {
            // ---------------------------------------------------------------
            // R_386_NONE: No relocation — nothing to do.
            // ---------------------------------------------------------------
            R_386_NONE => Ok(()),

            // ---------------------------------------------------------------
            // R_386_32: S + A — absolute 32-bit address.
            // ---------------------------------------------------------------
            R_386_32 => {
                let value = compute_absolute(s, a);
                if !fits_unsigned(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_PC32: S + A - P — PC-relative 32-bit.
            // ---------------------------------------------------------------
            R_386_PC32 => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_PC32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_GOT32: G + A — GOT entry offset from GOT base.
            // G = got_entry - got_base, so value = (got_entry - got_base) + A.
            // Uses compute_got_relative(got_entry, A, got_base).
            // ---------------------------------------------------------------
            R_386_GOT32 => {
                let got_entry =
                    rel.got_address
                        .ok_or_else(|| RelocationError::UndefinedSymbol {
                            name: "GOT entry".to_string(),
                            reloc_name: "R_386_GOT32".to_string(),
                        })?;
                let got_base = rel.got_base.unwrap_or(0);
                let value = compute_got_relative(got_entry, a, got_base);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_GOT32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_PLT32: L + A - P — PLT entry PC-relative.
            // Falls back to symbol address if no PLT entry is allocated.
            // ---------------------------------------------------------------
            R_386_PLT32 => {
                let plt_addr = rel.plt_address.unwrap_or(s);
                let value = compute_pc_relative(plt_addr, a, p);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_PLT32".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_COPY: handled by the dynamic linker at load time.
            // The static linker emits this relocation; nothing to patch here.
            // ---------------------------------------------------------------
            R_386_COPY => Ok(()),

            // ---------------------------------------------------------------
            // R_386_GLOB_DAT: S — fill GOT entry with symbol address.
            // ---------------------------------------------------------------
            R_386_GLOB_DAT => {
                write_le(section_data, offset, 4, s);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_JMP_SLOT: S — PLT jump slot (lazy binding target).
            // ---------------------------------------------------------------
            R_386_JMP_SLOT => {
                write_le(section_data, offset, 4, s);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_RELATIVE: B + A — base address + addend.
            // For static linking, base = 0; the dynamic linker adds the
            // actual base address at load time.
            // ---------------------------------------------------------------
            R_386_RELATIVE => {
                let value = compute_absolute(0, a);
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_GOTOFF: S + A - GOT — offset from GOT base.
            // Equivalent to compute_pc_relative(S, A, GOT_base).
            // ---------------------------------------------------------------
            R_386_GOTOFF => {
                let got_base = rel.got_base.unwrap_or(0);
                let value = compute_pc_relative(s, a, got_base);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_GOTOFF".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_GOTPC: GOT + A - P — GOT address PC-relative.
            // ---------------------------------------------------------------
            R_386_GOTPC => {
                let got_base = rel.got_base.unwrap_or(0);
                let value = compute_pc_relative(got_base, a, p);
                if !fits_signed(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_GOTPC".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_32PLT: L + A — absolute PLT address (rarely used).
            // ---------------------------------------------------------------
            R_386_32PLT => {
                let plt_addr = rel.plt_address.unwrap_or(s);
                let value = compute_absolute(plt_addr, a);
                if !fits_unsigned(value, 32) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_32PLT".to_string(),
                        value: value as i128,
                        bit_width: 32,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_16: S + A — absolute 16-bit address.
            // ---------------------------------------------------------------
            R_386_16 => {
                let value = compute_absolute(s, a);
                if !fits_unsigned(value, 16) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_16".to_string(),
                        value: value as i128,
                        bit_width: 16,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 2, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_PC16: S + A - P — PC-relative 16-bit.
            // ---------------------------------------------------------------
            R_386_PC16 => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 16) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_PC16".to_string(),
                        value: value as i128,
                        bit_width: 16,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 2, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_8: S + A — absolute 8-bit address.
            // ---------------------------------------------------------------
            R_386_8 => {
                let value = compute_absolute(s, a);
                if !fits_unsigned(value, 8) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_8".to_string(),
                        value: value as i128,
                        bit_width: 8,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 1, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_PC8: S + A - P — PC-relative 8-bit.
            // ---------------------------------------------------------------
            R_386_PC8 => {
                let value = compute_pc_relative(s, a, p);
                if !fits_signed(value, 8) {
                    return Err(RelocationError::Overflow {
                        reloc_name: "R_386_PC8".to_string(),
                        value: value as i128,
                        bit_width: 8,
                        location: format!("0x{:08x}", p),
                    });
                }
                write_le(section_data, offset, 1, value as u64);
                Ok(())
            }

            // ---------------------------------------------------------------
            // R_386_SIZE32: Z + A — symbol size + addend.
            // The symbol_value field carries the symbol size here.
            // ---------------------------------------------------------------
            R_386_SIZE32 => {
                let value = compute_absolute(s, a);
                write_le(section_data, offset, 4, value);
                Ok(())
            }

            // ---------------------------------------------------------------
            // Unsupported relocation type for i686.
            // ---------------------------------------------------------------
            _ => Err(RelocationError::UnsupportedType {
                rel_type: rel.rel_type,
                target: Target::I686,
            }),
        }
    }

    /// Returns `true` if this relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool {
        matches!(
            rel_type,
            R_386_GOT32 | R_386_GOTOFF | R_386_GOTPC | R_386_GLOB_DAT
        )
    }

    /// Returns `true` if this relocation type requires a PLT stub.
    fn needs_plt(&self, rel_type: u32) -> bool {
        matches!(rel_type, R_386_PLT32 | R_386_32PLT | R_386_JMP_SLOT)
    }
}

// ---------------------------------------------------------------------------
// I686_RELOCATION_TYPES — static relocation type metadata table
// ---------------------------------------------------------------------------

/// Static relocation type information table for i686.
///
/// Used by the linker for relocation type metadata queries and by the
/// [`ArchCodegen`](crate::backend::traits::ArchCodegen) trait implementation
/// via the `relocation_types()` method.
pub static I686_RELOCATION_TYPES: &[RelocationTypeInfo] = &[
    RelocationTypeInfo::new("R_386_NONE", R_386_NONE, 0, false),
    RelocationTypeInfo::new("R_386_32", R_386_32, 4, false),
    RelocationTypeInfo::new("R_386_PC32", R_386_PC32, 4, true),
    RelocationTypeInfo::new("R_386_GOT32", R_386_GOT32, 4, false),
    RelocationTypeInfo::new("R_386_PLT32", R_386_PLT32, 4, true),
    RelocationTypeInfo::new("R_386_COPY", R_386_COPY, 0, false),
    RelocationTypeInfo::new("R_386_GLOB_DAT", R_386_GLOB_DAT, 4, false),
    RelocationTypeInfo::new("R_386_JMP_SLOT", R_386_JMP_SLOT, 4, false),
    RelocationTypeInfo::new("R_386_RELATIVE", R_386_RELATIVE, 4, false),
    RelocationTypeInfo::new("R_386_GOTOFF", R_386_GOTOFF, 4, false),
    RelocationTypeInfo::new("R_386_GOTPC", R_386_GOTPC, 4, true),
];

// ---------------------------------------------------------------------------
// Implicit Addend Reading (Elf32_Rel support)
// ---------------------------------------------------------------------------

/// Read the implicit addend from section data at the relocation site.
///
/// i686 ELF commonly uses `Elf32_Rel` entries which do **not** have an
/// explicit addend field. Instead, the addend is read from the bytes at the
/// relocation site in the section data and sign-extended to `i64`.
///
/// For `Elf32_Rela` entries (which **do** have an explicit addend), this
/// function is not needed — use the addend field directly.
///
/// # Arguments
///
/// - `section_data` — byte buffer of the section containing the relocation.
/// - `offset` — byte offset within `section_data` where the relocation is applied.
/// - `rel_type` — the R_386_* relocation type code.
///
/// # Returns
///
/// The sign-extended addend value read from the relocation site, or `0` for
/// relocation types that have no implicit addend (e.g., `R_386_NONE`).
pub fn read_implicit_addend(section_data: &[u8], offset: usize, rel_type: u32) -> i64 {
    let size = match rel_type {
        R_386_32 | R_386_PC32 | R_386_GOT32 | R_386_PLT32 | R_386_GLOB_DAT | R_386_JMP_SLOT
        | R_386_RELATIVE | R_386_GOTOFF | R_386_GOTPC | R_386_32PLT | R_386_SIZE32 => 4,
        R_386_16 | R_386_PC16 => 2,
        R_386_8 | R_386_PC8 => 1,
        _ => return 0,
    };

    // Read the current little-endian value at the relocation site.
    let raw = read_le(section_data, offset, size);

    // Sign-extend the raw value based on the relocation size.
    sign_extend(raw, (size * 8) as u8)
}

// ---------------------------------------------------------------------------
// Relocation Validation Helpers
// ---------------------------------------------------------------------------

/// Validate that a relocation type code is a known R_386_* type.
///
/// Returns `true` for all defined i686 relocation types, `false` for
/// undefined or reserved codes.
pub fn is_valid_i686_reloc(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_NONE
            | R_386_32
            | R_386_PC32
            | R_386_GOT32
            | R_386_PLT32
            | R_386_COPY
            | R_386_GLOB_DAT
            | R_386_JMP_SLOT
            | R_386_RELATIVE
            | R_386_GOTOFF
            | R_386_GOTPC
            | R_386_32PLT
            | R_386_16
            | R_386_PC16
            | R_386_8
            | R_386_PC8
            | R_386_SIZE32
    )
}

/// Check if a relocation type is a dynamic relocation (for `.rel.dyn`/`.rela.dyn`).
///
/// Dynamic relocations are emitted into the `.rel.dyn` or `.rela.dyn` sections
/// and processed by the dynamic linker at load time.
pub fn is_dynamic_reloc(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_GLOB_DAT | R_386_JMP_SLOT | R_386_RELATIVE | R_386_COPY
    )
}

/// Check if a relocation type requires absolute addressing (not PIC-safe).
///
/// In PIC (position-independent code), these relocations should be avoided
/// because they bake absolute addresses into the code, requiring fixup by
/// the dynamic linker.
pub fn is_absolute_reloc(rel_type: u32) -> bool {
    matches!(rel_type, R_386_32 | R_386_16 | R_386_8)
}

/// Get a human-readable description of a relocation type including the formula.
pub fn reloc_description(rel_type: u32) -> &'static str {
    match rel_type {
        R_386_NONE => "No relocation",
        R_386_32 => "S + A (absolute 32-bit)",
        R_386_PC32 => "S + A - P (PC-relative 32-bit)",
        R_386_GOT32 => "G + A (GOT entry offset)",
        R_386_PLT32 => "L + A - P (PLT entry PC-relative)",
        R_386_COPY => "Copy symbol data from shared object",
        R_386_GLOB_DAT => "S (global data GOT entry)",
        R_386_JMP_SLOT => "S (PLT jump slot)",
        R_386_RELATIVE => "B + A (base address relative)",
        R_386_GOTOFF => "S + A - GOT (offset from GOT base)",
        R_386_GOTPC => "GOT + A - P (GOT address PC-relative)",
        R_386_32PLT => "L + A (absolute PLT address)",
        R_386_16 => "S + A (absolute 16-bit)",
        R_386_PC16 => "S + A - P (PC-relative 16-bit)",
        R_386_8 => "S + A (absolute 8-bit)",
        R_386_PC8 => "S + A - P (PC-relative 8-bit)",
        R_386_SIZE32 => "Z + A (symbol size + addend)",
        _ => "Unknown relocation type",
    }
}

// ---------------------------------------------------------------------------
// Elf32_Rel and Elf32_Rela Encoding / Decoding
// ---------------------------------------------------------------------------
//
// CRITICAL: The Elf32 r_info encoding is DIFFERENT from Elf64:
//   Elf32: r_info = (sym_index << 8)  | (type & 0xFF)    — 24-bit sym, 8-bit type
//   Elf64: r_info = (sym_index << 32) | (type & 0xFFFFFFFF) — 32-bit sym, 32-bit type
// This difference is a common source of bugs when porting between 32-bit and
// 64-bit linkers.

/// Encode an `Elf32_Rel` entry (8 bytes, no explicit addend).
///
/// # Layout
///
/// | Offset | Size | Field      | Description                           |
/// |--------|------|------------|---------------------------------------|
/// | 0      | 4    | `r_offset` | Offset where the relocation applies   |
/// | 4      | 4    | `r_info`   | `(sym_index << 8) | (type & 0xFF)` |
///
/// # Arguments
///
/// - `offset` — byte offset within the section where the relocation is applied.
/// - `sym_index` — index of the symbol in the symbol table (upper 24 bits).
/// - `rel_type` — R_386_* relocation type code (lower 8 bits).
pub fn encode_elf32_rel(offset: u32, sym_index: u32, rel_type: u32) -> [u8; 8] {
    let r_info = make_elf32_r_info(sym_index, rel_type);
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&offset.to_le_bytes());
    buf[4..8].copy_from_slice(&r_info.to_le_bytes());
    buf
}

/// Encode an `Elf32_Rela` entry (12 bytes, with explicit addend).
///
/// # Layout
///
/// | Offset | Size | Field      | Description                           |
/// |--------|------|------------|---------------------------------------|
/// | 0      | 4    | `r_offset` | Offset where the relocation applies   |
/// | 4      | 4    | `r_info`   | `(sym_index << 8) | (type & 0xFF)` |
/// | 8      | 4    | `r_addend` | Signed addend                         |
///
/// # Arguments
///
/// - `offset` — byte offset within the section.
/// - `sym_index` — symbol table index.
/// - `rel_type` — R_386_* relocation type code.
/// - `addend` — signed addend value.
pub fn encode_elf32_rela(offset: u32, sym_index: u32, rel_type: u32, addend: i32) -> [u8; 12] {
    let r_info = make_elf32_r_info(sym_index, rel_type);
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&offset.to_le_bytes());
    buf[4..8].copy_from_slice(&r_info.to_le_bytes());
    buf[8..12].copy_from_slice(&addend.to_le_bytes());
    buf
}

/// Decode the `r_info` field from an `Elf32_Rel` / `Elf32_Rela` entry.
///
/// Returns `(sym_index, rel_type)` where:
/// - `sym_index = r_info >> 8` (upper 24 bits)
/// - `rel_type  = r_info & 0xFF` (lower 8 bits)
#[inline]
pub fn decode_elf32_r_info(r_info: u32) -> (u32, u32) {
    let sym_index = r_info >> 8;
    let rel_type = r_info & 0xFF;
    (sym_index, rel_type)
}

/// Construct the `r_info` field for an `Elf32_Rel` / `Elf32_Rela` entry
/// from a symbol table index and relocation type.
///
/// `r_info = (sym_index << 8) | (rel_type & 0xFF)`
#[inline]
pub fn make_elf32_r_info(sym_index: u32, rel_type: u32) -> u32 {
    (sym_index << 8) | (rel_type & 0xFF)
}
