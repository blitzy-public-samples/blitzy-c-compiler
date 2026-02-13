//! i686 (32-bit x86) relocation application module for the BCC built-in linker.
//!
//! Implements the [`ArchRelocationHandler`] trait for all i386 ELF relocation
//! types as defined by the System V i386 ABI supplement. This module is the
//! architecture-specific back-end that patches machine code bytes at relocation
//! sites in 32-bit ELF output (both `ET_EXEC` and `ET_DYN`).
//!
//! # Supported Relocation Types
//!
//! | Constant          | Value | Formula          | Description                        |
//! |-------------------|-------|------------------|------------------------------------|
//! | `R_386_NONE`      | 0     | —                | No relocation                      |
//! | `R_386_32`        | 1     | S + A            | Absolute 32-bit address            |
//! | `R_386_PC32`      | 2     | S + A - P        | PC-relative 32-bit displacement    |
//! | `R_386_GOT32`     | 3     | G + A            | GOT entry offset from GOT base     |
//! | `R_386_PLT32`     | 4     | L + A - P        | PLT entry PC-relative              |
//! | `R_386_COPY`      | 5     | —                | Runtime copy relocation            |
//! | `R_386_GLOB_DAT`  | 6     | S                | GOT entry ← symbol address         |
//! | `R_386_JMP_SLOT`  | 7     | S                | PLT GOT entry for lazy binding     |
//! | `R_386_RELATIVE`  | 8     | B + A            | Base-relative address              |
//! | `R_386_GOTOFF`    | 9     | S + A - GOT      | Symbol offset from GOT base        |
//! | `R_386_GOTPC`     | 10    | GOT + A - P      | GOT base PC-relative               |
//! | `R_386_32PLT`     | 11    | L + A            | Absolute PLT address               |
//!
//! Where:
//! - **S** = symbol value (resolved virtual address)
//! - **A** = addend (explicit RELA or implicit from section data)
//! - **P** = relocation site address (`reloc.offset`)
//! - **G** = GOT entry offset from GOT base
//! - **L** = PLT entry address
//! - **B** = base load address (for shared libraries)
//! - **GOT** = GOT section base address
//!
//! # REL vs RELA Handling
//!
//! The i386 ABI traditionally uses `Elf32_Rel` (no explicit addend). When the
//! [`RelocationEntry::addend`] is zero, this module reads the implicit addend
//! from the relocation site in the section data, enabling correct handling of
//! both REL-style and RELA-style input.
//!
//! # Overflow Checking
//!
//! All relocation computations are performed in 128-bit arithmetic to detect
//! overflow before truncating to 32 bits. Unsigned relocations (e.g., `R_386_32`)
//! are checked against the `u32` range, and signed relocations (e.g., `R_386_PC32`)
//! are checked against the `i32` range.
//!
//! # Standalone Backend Mode
//!
//! This module is part of BCC's fully self-contained linker — no external `ld`
//! or `gold` linker is ever invoked. All PIC relocation handling (GOT, PLT) is
//! entirely internal. Zero external dependencies are used.

use crate::backend::linker_common::dynamic::DynamicRelocation;
use crate::backend::linker_common::linker_script::OutputType;
use crate::backend::linker_common::relocation::{
    ArchRelocationHandler, RelocationEntry, RelocationError,
};

// ===========================================================================
// i386 Relocation Type Constants (ELF ABI)
// ===========================================================================
// Hand-defined per the zero-dependency mandate. Values match the ELF i386
// ABI supplement and <elf.h> definitions.

/// No relocation — placeholder entry, no patching performed.
pub const R_386_NONE: u32 = 0;

/// Absolute 32-bit address: `S + A`.
/// Writes the full virtual address of the symbol plus addend.
pub const R_386_32: u32 = 1;

/// PC-relative 32-bit displacement: `S + A - P`.
/// Used for relative branches and calls within the same module.
pub const R_386_PC32: u32 = 2;

/// GOT entry offset from GOT base: `G + A`.
/// The symbol's GOT slot offset relative to the GOT section origin.
pub const R_386_GOT32: u32 = 3;

/// PLT entry PC-relative: `L + A - P`.
/// References a PLT stub for external function calls in PIC code.
pub const R_386_PLT32: u32 = 4;

/// Copy relocation — runtime linker copies symbol data from a shared
/// library into the executable's BSS. No direct patching at link time.
pub const R_386_COPY: u32 = 5;

/// Set GOT entry to the symbol's absolute address: `S`.
/// Used to initialize GOT entries for data symbols in shared libraries.
pub const R_386_GLOB_DAT: u32 = 6;

/// Set PLT GOT entry for lazy binding: `S`.
/// The dynamic linker fills this slot when the function is first called.
pub const R_386_JMP_SLOT: u32 = 7;

/// Base-relative address: `B + A`.
/// `B` is the runtime base address of the shared object. Used in
/// `.rel.dyn` for position-independent data references.
pub const R_386_RELATIVE: u32 = 8;

/// GOT-relative symbol offset: `S + A - GOT`.
/// The symbol's address minus the GOT base, for GOT-indirect access.
pub const R_386_GOTOFF: u32 = 9;

/// PC-relative GOT base: `GOT + A - P`.
/// Used to compute the distance from the relocation site to the GOT.
pub const R_386_GOTPC: u32 = 10;

/// Absolute 32-bit PLT address: `L + A`.
/// Rarely used; writes the absolute address of a PLT entry.
pub const R_386_32PLT: u32 = 11;

/// TLS Initial Exec model — GOT entry holding the TP-relative offset.
pub const R_386_TLS_IE: u32 = 15;

/// TLS Local Exec model — direct TP-relative offset for same-module TLS.
pub const R_386_TLS_LE: u32 = 17;

/// TLS General Dynamic model — offset to `tls_index` structure in GOT.
pub const R_386_TLS_GD: u32 = 18;

/// TLS Local Dynamic Module — offset to module's `tls_index` in GOT.
pub const R_386_TLS_LDM: u32 = 19;

/// TLS GD 32-bit offset variant.
pub const R_386_TLS_GD_32: u32 = 24;

/// TLS LDM 32-bit offset variant.
pub const R_386_TLS_LDM_32: u32 = 28;

/// TLS module ID for `dlpi_tls_modid`.
pub const R_386_TLS_DTPMOD32: u32 = 35;

/// TLS offset within module for `dlpi_tls_data`.
pub const R_386_TLS_DTPOFF32: u32 = 36;

/// Negative TP offset (TLS variant II used by i386 Linux).
pub const R_386_TLS_TPOFF32: u32 = 37;

/// Symbol size: `Z + A` where Z is `st_size` of the symbol.
pub const R_386_SIZE32: u32 = 38;

// ===========================================================================
// I686RelocationHandler — architecture-specific relocation processor
// ===========================================================================

/// Handles i386 (IA-32) relocation application for 32-bit x86 ELF output.
///
/// Implements [`ArchRelocationHandler`] to provide architecture-specific
/// relocation patching, overflow checking, and GOT/PLT classification.
/// Used by the i686 linker driver (`mod.rs`) during the link phase.
///
/// # Example
///
/// ```ignore
/// let handler = I686RelocationHandler::new();
/// handler.apply_relocation(&reloc, &mut output, got_addr, plt_addr)?;
/// ```
pub struct I686RelocationHandler;

impl I686RelocationHandler {
    /// Creates a new i686 relocation handler.
    ///
    /// The handler is stateless — all relocation context is provided via
    /// method parameters, making it safe to share across threads.
    #[inline]
    pub fn new() -> Self {
        I686RelocationHandler
    }

    /// Generates a dynamic relocation entry for the runtime linker when
    /// producing shared libraries (`ET_DYN`) or dynamically linked executables.
    ///
    /// # Returns
    ///
    /// - `Some(DynamicRelocation)` if the relocation cannot be fully resolved
    ///   at static link time and must be deferred to the dynamic linker.
    /// - `None` if the relocation is fully resolved and no dynamic entry is needed.
    ///
    /// # Dynamic Relocation Mapping
    ///
    /// | Static Reloc      | Dynamic Reloc       | Section     |
    /// |-------------------|---------------------|-------------|
    /// | `R_386_32`        | `R_386_RELATIVE`    | `.rel.dyn`  |
    /// | `R_386_GOT32`     | `R_386_GLOB_DAT`    | `.rel.dyn`  |
    /// | `R_386_GLOB_DAT`  | `R_386_GLOB_DAT`    | `.rel.dyn`  |
    /// | `R_386_PLT32`     | `R_386_JMP_SLOT`    | `.rel.plt`  |
    /// | `R_386_JMP_SLOT`  | `R_386_JMP_SLOT`    | `.rel.plt`  |
    pub fn generate_dynamic_relocation(
        &self,
        reloc: &RelocationEntry,
        output_type: &OutputType,
    ) -> Option<DynamicRelocation> {
        // Relocatable objects and fully static executables never emit dynamic
        // relocations. For executables, most relocations are resolved at link
        // time unless the binary is dynamically linked (PIE).
        match output_type {
            OutputType::RelocatableObject => return None,
            OutputType::Executable => {
                // For static executables, all relocations are resolved.
                // For dynamically linked executables, only GLOB_DAT/JMP_SLOT
                // from imported symbols generate dynamic relocs.
                match reloc.reloc_type {
                    R_386_GLOB_DAT => {
                        return Some(DynamicRelocation {
                            offset: reloc.offset,
                            reloc_type: R_386_GLOB_DAT,
                            symbol_index: 0, // Caller resolves dynsym index
                            addend: reloc.addend,
                        });
                    }
                    R_386_JMP_SLOT => {
                        return Some(DynamicRelocation {
                            offset: reloc.offset,
                            reloc_type: R_386_JMP_SLOT,
                            symbol_index: 0, // Caller resolves dynsym index
                            addend: reloc.addend,
                        });
                    }
                    _ => return None,
                }
            }
            OutputType::SharedLibrary => {
                // Fall through to shared library logic below.
            }
        }

        // Shared library dynamic relocation generation.
        // In ET_DYN, position-dependent relocations must be converted to
        // dynamic relocations for the runtime linker.
        match reloc.reloc_type {
            R_386_32 => {
                // Absolute 32-bit reference in a shared library:
                // Generate R_386_RELATIVE if the symbol is local (defined
                // within this shared object), allowing base address fixup at
                // load time. The addend encodes the link-time address.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_RELATIVE,
                    symbol_index: 0,
                    addend: (reloc.symbol_value as i64).wrapping_add(reloc.addend),
                })
            }
            R_386_GOT32 | R_386_GLOB_DAT => {
                // GOT entries need GLOB_DAT dynamic relocations so the
                // runtime linker can fill them with the actual symbol address.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_GLOB_DAT,
                    symbol_index: 0, // Caller resolves dynsym index
                    addend: reloc.addend,
                })
            }
            R_386_PLT32 | R_386_JMP_SLOT => {
                // PLT entries need JMP_SLOT dynamic relocations for lazy
                // binding of external function calls.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_JMP_SLOT,
                    symbol_index: 0, // Caller resolves dynsym index
                    addend: reloc.addend,
                })
            }
            R_386_RELATIVE => {
                // R_386_RELATIVE is already a dynamic relocation type;
                // pass it through directly.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_RELATIVE,
                    symbol_index: 0,
                    addend: reloc.addend,
                })
            }
            R_386_TLS_DTPMOD32 => {
                // TLS module ID must be filled by the dynamic linker.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_TLS_DTPMOD32,
                    symbol_index: 0,
                    addend: reloc.addend,
                })
            }
            R_386_TLS_DTPOFF32 => {
                // TLS offset within module, resolved at runtime.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_TLS_DTPOFF32,
                    symbol_index: 0,
                    addend: reloc.addend,
                })
            }
            R_386_TLS_TPOFF32 => {
                // TLS TP-relative offset, resolved at runtime.
                Some(DynamicRelocation {
                    offset: reloc.offset,
                    reloc_type: R_386_TLS_TPOFF32,
                    symbol_index: 0,
                    addend: reloc.addend,
                })
            }
            // PC-relative relocations (R_386_PC32, R_386_GOTPC) and
            // GOT-relative relocations (R_386_GOTOFF) are fully resolved
            // at static link time — no dynamic relocation needed.
            R_386_PC32 | R_386_GOTOFF | R_386_GOTPC | R_386_NONE | R_386_COPY => None,

            // All other types (including R_386_32PLT, TLS GD/LDM variants)
            // are either resolved at link time or unsupported for dynamic.
            _ => None,
        }
    }
}

// ===========================================================================
// ArchRelocationHandler Implementation
// ===========================================================================

impl ArchRelocationHandler for I686RelocationHandler {
    /// Applies a single i386 relocation to the output section data.
    ///
    /// Computes the relocation value using the architecture-specific formula,
    /// checks for 32-bit overflow, and writes the result as a 4-byte
    /// little-endian value at the relocation offset.
    ///
    /// # i386 REL Addend Handling
    ///
    /// When `reloc.addend` is zero (typical for `Elf32_Rel` input), the
    /// implicit addend is read from the existing 4 bytes at the relocation
    /// site. This ensures correct handling of both REL and RELA input formats.
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), RelocationError> {
        let offset = reloc.offset as usize;

        // R_386_NONE requires no processing.
        if reloc.reloc_type == R_386_NONE {
            return Ok(());
        }

        // R_386_COPY is a runtime-only relocation; no static-time patching.
        if reloc.reloc_type == R_386_COPY {
            return Ok(());
        }

        // Bounds check: ensure we can read/write 4 bytes at the offset.
        if offset.checked_add(4).map_or(true, |end| end > output_data.len()) {
            return Err(RelocationError::InvalidOffset {
                offset: reloc.offset,
                section_size: output_data.len() as u64,
            });
        }

        // Determine the effective addend. i386 traditionally uses Elf32_Rel
        // (no explicit addend), so when the RELA addend is zero we read the
        // implicit addend from the relocation site in the section data.
        let implicit_addend = read_le32(output_data, offset) as i32 as i64;
        let addend: i64 = if reloc.addend != 0 {
            reloc.addend
        } else {
            implicit_addend
        };

        // Promote operands to i128 for overflow-safe arithmetic.
        let s = reloc.symbol_value as i128; // S: symbol virtual address
        let a = addend as i128; // A: addend
        let p = reloc.offset as i128; // P: patch site address
        let got = got_address as i128; // GOT base address
        let _plt = plt_address as i128; // PLT base address (unused directly for some)

        match reloc.reloc_type {
            // -----------------------------------------------------------------
            // R_386_32: Absolute 32-bit — S + A
            // -----------------------------------------------------------------
            R_386_32 => {
                let value = s + a;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_PC32: PC-relative 32-bit — S + A - P
            // -----------------------------------------------------------------
            R_386_PC32 => {
                let value = s + a - p;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_GOT32: GOT entry offset — G + A
            // G is the symbol's GOT entry offset from the GOT base.
            // We compute G = symbol_value - got_base.
            // -----------------------------------------------------------------
            R_386_GOT32 => {
                let g = s - got;
                let value = g + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_PLT32: PLT PC-relative — L + A - P
            // L is the PLT entry address for the symbol. The symbol_value
            // is expected to point to the PLT entry when a PLT stub exists.
            // -----------------------------------------------------------------
            R_386_PLT32 => {
                let value = s + a - p;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_GLOB_DAT: Set GOT entry to symbol address — S
            // -----------------------------------------------------------------
            R_386_GLOB_DAT => {
                let value = s;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_JMP_SLOT: Set PLT GOT entry — S
            // -----------------------------------------------------------------
            R_386_JMP_SLOT => {
                let value = s;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_RELATIVE: Base-relative — B + A
            // B is the base load address. For static linking, the symbol_value
            // carries the base address contribution. The final runtime address
            // is computed by the dynamic linker as B + A.
            // -----------------------------------------------------------------
            R_386_RELATIVE => {
                let value = s + a;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_GOTOFF: GOT-relative symbol offset — S + A - GOT
            // -----------------------------------------------------------------
            R_386_GOTOFF => {
                let value = s + a - got;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_GOTPC: PC-relative to GOT base — GOT + A - P
            // -----------------------------------------------------------------
            R_386_GOTPC => {
                let value = got + a - p;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_32PLT: Absolute PLT address — L + A
            // The symbol_value is expected to be the PLT entry address.
            // -----------------------------------------------------------------
            R_386_32PLT => {
                let value = s + a;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // R_386_SIZE32: Symbol size — Z + A
            // Z is the symbol's st_size; conveyed via symbol_value here.
            // -----------------------------------------------------------------
            R_386_SIZE32 => {
                let value = s + a;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // TLS relocations — handled at link time where possible.
            // -----------------------------------------------------------------
            R_386_TLS_LE => {
                // Local Exec: direct TP-relative offset.
                // value = S + A (offset from thread pointer, negative on i386).
                let value = s + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_IE => {
                // Initial Exec: GOT entry holding TP-relative offset.
                // G + A where G is the GOT slot containing the TP offset.
                let g = s - got;
                let value = g + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_GD => {
                // General Dynamic: offset to tls_index pair in GOT.
                let g = s - got;
                let value = g + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_LDM => {
                // Local Dynamic Module: offset to module tls_index in GOT.
                let g = s - got;
                let value = g + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_GD_32 => {
                // TLS GD 32-bit offset variant.
                let value = s + a - p;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_LDM_32 => {
                // TLS LDM 32-bit offset variant.
                let value = s + a - p;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_DTPMOD32 => {
                // TLS module ID — typically filled by the dynamic linker.
                // At static link time, write the symbol value directly.
                let value = s + a;
                check_unsigned_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_DTPOFF32 => {
                // TLS offset within module.
                let value = s + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            R_386_TLS_TPOFF32 => {
                // Negative offset from TP (TLS variant II on i386).
                let value = s + a;
                check_signed_overflow(value, reloc.reloc_type, reloc.offset)?;
                write_le32(output_data, offset, value as u32);
                Ok(())
            }

            // -----------------------------------------------------------------
            // Unsupported / unknown relocation type.
            // -----------------------------------------------------------------
            _ => Err(RelocationError::UnsupportedType {
                reloc_type: reloc.reloc_type,
            }),
        }
    }

    /// Returns a human-readable name for the given i386 relocation type.
    ///
    /// Used in error messages and diagnostic output. Returns `"R_386_UNKNOWN"`
    /// for unrecognized type codes.
    fn relocation_name(&self, reloc_type: u32) -> &'static str {
        match reloc_type {
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
            R_386_TLS_IE => "R_386_TLS_IE",
            R_386_TLS_LE => "R_386_TLS_LE",
            R_386_TLS_GD => "R_386_TLS_GD",
            R_386_TLS_LDM => "R_386_TLS_LDM",
            R_386_TLS_GD_32 => "R_386_TLS_GD_32",
            R_386_TLS_LDM_32 => "R_386_TLS_LDM_32",
            R_386_TLS_DTPMOD32 => "R_386_TLS_DTPMOD32",
            R_386_TLS_DTPOFF32 => "R_386_TLS_DTPOFF32",
            R_386_TLS_TPOFF32 => "R_386_TLS_TPOFF32",
            R_386_SIZE32 => "R_386_SIZE32",
            _ => "R_386_UNKNOWN",
        }
    }

    /// Returns `true` if the given relocation type requires a GOT entry.
    ///
    /// GOT entries are allocated by the dynamic linking section generator
    /// for symbols that must be indirectly accessed in PIC code.
    #[inline]
    fn needs_got_entry(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_386_GOT32 | R_386_GLOB_DAT | R_386_TLS_GD | R_386_TLS_IE
        )
    }

    /// Returns `true` if the given relocation type requires a PLT entry.
    ///
    /// PLT entries are generated for function calls through the Procedure
    /// Linkage Table in shared libraries and PIE executables.
    #[inline]
    fn needs_plt_entry(&self, reloc_type: u32) -> bool {
        matches!(reloc_type, R_386_PLT32 | R_386_JMP_SLOT)
    }

    /// Returns `true` if the given relocation type computes a PC-relative value.
    ///
    /// PC-relative relocations subtract the relocation site address (`P`) from
    /// the computed value, making the result position-independent.
    #[inline]
    fn is_pc_relative(&self, reloc_type: u32) -> bool {
        matches!(reloc_type, R_386_PC32 | R_386_PLT32 | R_386_GOTPC)
    }

    /// Returns the size in bytes of the relocation field.
    ///
    /// All standard i386 relocations write 4 bytes (32 bits). Returns 0 for
    /// `R_386_NONE` (no write) and `R_386_COPY` (runtime-only, no write).
    #[inline]
    fn relocation_size(&self, reloc_type: u32) -> u8 {
        match reloc_type {
            R_386_NONE | R_386_COPY => 0,
            _ => 4,
        }
    }
}

// ===========================================================================
// Helper Functions — Little-Endian I/O and Overflow Checking
// ===========================================================================

/// Writes a 32-bit value in little-endian byte order at the specified offset.
///
/// # Safety Contract
///
/// Callers must ensure `offset + 4 <= data.len()`. This function performs the
/// write unconditionally — bounds checking is the caller's responsibility
/// (validated in `apply_relocation` before dispatch).
#[inline]
fn write_le32(data: &mut [u8], offset: usize, value: u32) {
    data[offset] = (value & 0xFF) as u8;
    data[offset + 1] = ((value >> 8) & 0xFF) as u8;
    data[offset + 2] = ((value >> 16) & 0xFF) as u8;
    data[offset + 3] = ((value >> 24) & 0xFF) as u8;
}

/// Reads a 32-bit value in little-endian byte order from the specified offset.
///
/// Used to extract the implicit addend from the relocation site when processing
/// `Elf32_Rel`-style relocations (which have no explicit addend field).
///
/// # Panics
///
/// Panics if `offset + 4 > data.len()`. Callers must bounds-check first.
#[inline]
fn read_le32(data: &[u8], offset: usize) -> u32 {
    (data[offset] as u32)
        | ((data[offset + 1] as u32) << 8)
        | ((data[offset + 2] as u32) << 16)
        | ((data[offset + 3] as u32) << 24)
}

/// Checks that an unsigned relocation value fits within the 32-bit unsigned range.
///
/// i386 absolute relocations (R_386_32, R_386_GLOB_DAT, R_386_JMP_SLOT, etc.)
/// must produce values in `0..=0xFFFF_FFFF`. Values outside this range indicate
/// a symbol or section layout that exceeds 32-bit addressing.
#[inline]
fn check_unsigned_overflow(
    value: i128,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    const MAX: i128 = u32::MAX as i128;
    if value < 0 || value > MAX {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value,
            max_value: MAX,
        });
    }
    Ok(())
}

/// Checks that a signed relocation value fits within the 32-bit signed range.
///
/// i386 PC-relative and GOT-relative relocations (R_386_PC32, R_386_PLT32,
/// R_386_GOTPC, R_386_GOTOFF, etc.) must produce values in
/// `-2^31..=2^31 - 1`. Overflow typically indicates that the symbol is too
/// far from the relocation site for a 32-bit displacement.
#[inline]
fn check_signed_overflow(
    value: i128,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    const MIN: i128 = i32::MIN as i128;
    const MAX: i128 = i32::MAX as i128;
    if value < MIN || value > MAX {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value,
            max_value: MAX,
        });
    }
    Ok(())
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructs a test RelocationEntry with the given parameters.
    fn make_reloc(
        reloc_type: u32,
        offset: u64,
        symbol_value: u64,
        addend: i64,
    ) -> RelocationEntry {
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
    fn test_r386_none_does_nothing() {
        let handler = I686RelocationHandler::new();
        let mut data = [0xAA; 8];
        let reloc = make_reloc(R_386_NONE, 2, 0x1000, 0);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        // Data must remain unchanged.
        assert_eq!(data, [0xAA; 8]);
    }

    #[test]
    fn test_r386_32_absolute() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0x0804_8000, A = 0x10 (explicit addend)
        let reloc = make_reloc(R_386_32, 0, 0x0804_8000, 0x10);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        // Expected: 0x0804_8010
        let result = read_le32(&data, 0);
        assert_eq!(result, 0x0804_8010);
    }

    #[test]
    fn test_r386_32_implicit_addend() {
        let handler = I686RelocationHandler::new();
        // Pre-fill 4 bytes at offset 0 with implicit addend = 0x20.
        let mut data = [0u8; 8];
        write_le32(&mut data, 0, 0x20);
        // Explicit addend = 0, so implicit addend (0x20) is used.
        let reloc = make_reloc(R_386_32, 0, 0x1000, 0);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        // S + A = 0x1000 + 0x20 = 0x1020
        assert_eq!(read_le32(&data, 0), 0x1020);
    }

    #[test]
    fn test_r386_pc32_pc_relative() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0x2000, A = -4, P = 0x1000
        // value = 0x2000 + (-4) - 0x1000 = 0x0FFC
        let reloc = make_reloc(R_386_PC32, 0x1000, 0x2000, -4);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap_err(); // Offset 0x1000 exceeds 8-byte buffer
    }

    #[test]
    fn test_r386_pc32_in_bounds() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0x100, A = -4, P = 0 (offset in buffer)
        // value = 0x100 + (-4) - 0 = 0xFC = 252
        let reloc = make_reloc(R_386_PC32, 0, 0x100, -4);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0xFC);
    }

    #[test]
    fn test_r386_gotoff() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0x3000, A = 0, GOT = 0x2000
        // value = 0x3000 + 0 - 0x2000 = 0x1000
        let reloc = make_reloc(R_386_GOTOFF, 0, 0x3000, 0x00);
        // Need non-zero explicit addend to avoid reading implicit from data.
        let reloc2 = RelocationEntry {
            addend: 0,
            ..reloc
        };
        // Since addend = 0, implicit addend = read_le32(&data, 0) = 0.
        handler
            .apply_relocation(&reloc2, &mut data, 0x2000, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0x1000);
    }

    #[test]
    fn test_r386_gotpc() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // GOT = 0x5000, A = 0, P = 0 (offset 0 in buffer)
        // value = 0x5000 + 0 - 0 = 0x5000
        let reloc = make_reloc(R_386_GOTPC, 0, 0, 0);
        handler
            .apply_relocation(&reloc, &mut data, 0x5000, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0x5000);
    }

    #[test]
    fn test_r386_copy_no_patching() {
        let handler = I686RelocationHandler::new();
        let mut data = [0xBB; 8];
        let reloc = make_reloc(R_386_COPY, 0, 0x1234, 0);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        // Data unchanged — COPY is runtime-only.
        assert_eq!(data, [0xBB; 8]);
    }

    #[test]
    fn test_relocation_name_coverage() {
        let handler = I686RelocationHandler::new();
        assert_eq!(handler.relocation_name(R_386_NONE), "R_386_NONE");
        assert_eq!(handler.relocation_name(R_386_32), "R_386_32");
        assert_eq!(handler.relocation_name(R_386_PC32), "R_386_PC32");
        assert_eq!(handler.relocation_name(R_386_GOT32), "R_386_GOT32");
        assert_eq!(handler.relocation_name(R_386_PLT32), "R_386_PLT32");
        assert_eq!(handler.relocation_name(R_386_COPY), "R_386_COPY");
        assert_eq!(handler.relocation_name(R_386_GLOB_DAT), "R_386_GLOB_DAT");
        assert_eq!(handler.relocation_name(R_386_JMP_SLOT), "R_386_JMP_SLOT");
        assert_eq!(handler.relocation_name(R_386_RELATIVE), "R_386_RELATIVE");
        assert_eq!(handler.relocation_name(R_386_GOTOFF), "R_386_GOTOFF");
        assert_eq!(handler.relocation_name(R_386_GOTPC), "R_386_GOTPC");
        assert_eq!(handler.relocation_name(R_386_32PLT), "R_386_32PLT");
        assert_eq!(handler.relocation_name(R_386_TLS_IE), "R_386_TLS_IE");
        assert_eq!(handler.relocation_name(R_386_TLS_LE), "R_386_TLS_LE");
        assert_eq!(handler.relocation_name(R_386_TLS_GD), "R_386_TLS_GD");
        assert_eq!(handler.relocation_name(R_386_TLS_LDM), "R_386_TLS_LDM");
        assert_eq!(
            handler.relocation_name(R_386_TLS_DTPMOD32),
            "R_386_TLS_DTPMOD32"
        );
        assert_eq!(
            handler.relocation_name(R_386_TLS_DTPOFF32),
            "R_386_TLS_DTPOFF32"
        );
        assert_eq!(
            handler.relocation_name(R_386_TLS_TPOFF32),
            "R_386_TLS_TPOFF32"
        );
        assert_eq!(handler.relocation_name(R_386_SIZE32), "R_386_SIZE32");
        assert_eq!(handler.relocation_name(999), "R_386_UNKNOWN");
    }

    #[test]
    fn test_needs_got_entry() {
        let handler = I686RelocationHandler::new();
        assert!(handler.needs_got_entry(R_386_GOT32));
        assert!(handler.needs_got_entry(R_386_GLOB_DAT));
        assert!(handler.needs_got_entry(R_386_TLS_GD));
        assert!(handler.needs_got_entry(R_386_TLS_IE));
        assert!(!handler.needs_got_entry(R_386_32));
        assert!(!handler.needs_got_entry(R_386_PC32));
        assert!(!handler.needs_got_entry(R_386_PLT32));
    }

    #[test]
    fn test_needs_plt_entry() {
        let handler = I686RelocationHandler::new();
        assert!(handler.needs_plt_entry(R_386_PLT32));
        assert!(handler.needs_plt_entry(R_386_JMP_SLOT));
        assert!(!handler.needs_plt_entry(R_386_32));
        assert!(!handler.needs_plt_entry(R_386_GOT32));
        assert!(!handler.needs_plt_entry(R_386_PC32));
    }

    #[test]
    fn test_is_pc_relative() {
        let handler = I686RelocationHandler::new();
        assert!(handler.is_pc_relative(R_386_PC32));
        assert!(handler.is_pc_relative(R_386_PLT32));
        assert!(handler.is_pc_relative(R_386_GOTPC));
        assert!(!handler.is_pc_relative(R_386_32));
        assert!(!handler.is_pc_relative(R_386_GOT32));
        assert!(!handler.is_pc_relative(R_386_GOTOFF));
        assert!(!handler.is_pc_relative(R_386_RELATIVE));
    }

    #[test]
    fn test_relocation_size() {
        let handler = I686RelocationHandler::new();
        assert_eq!(handler.relocation_size(R_386_NONE), 0);
        assert_eq!(handler.relocation_size(R_386_COPY), 0);
        assert_eq!(handler.relocation_size(R_386_32), 4);
        assert_eq!(handler.relocation_size(R_386_PC32), 4);
        assert_eq!(handler.relocation_size(R_386_GOT32), 4);
        assert_eq!(handler.relocation_size(R_386_PLT32), 4);
        assert_eq!(handler.relocation_size(R_386_TLS_LE), 4);
        assert_eq!(handler.relocation_size(R_386_SIZE32), 4);
    }

    #[test]
    fn test_overflow_unsigned() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0x1_0000_0000 (exceeds 32-bit), A = 0
        let reloc = make_reloc(R_386_32, 0, 0x1_0000_0000, 1);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        if let Err(RelocationError::Overflow { reloc_type, .. }) = result {
            assert_eq!(reloc_type, R_386_32);
        }
    }

    #[test]
    fn test_overflow_signed_negative() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S = 0, A = i32::MIN as i64 - 1 (exceeds signed 32-bit range)
        let reloc = make_reloc(R_386_PC32, 0, 0, i32::MIN as i64 - 1);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_offset_out_of_bounds() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 4];
        // Offset 2 means we'd need bytes [2..6], but only [0..4] exist.
        let reloc = make_reloc(R_386_32, 2, 0x1000, 1);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        if let Err(RelocationError::InvalidOffset { offset, .. }) = result {
            assert_eq!(offset, 2);
        }
    }

    #[test]
    fn test_unsupported_type() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        let reloc = make_reloc(200, 0, 0, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        if let Err(RelocationError::UnsupportedType { reloc_type }) = result {
            assert_eq!(reloc_type, 200);
        }
    }

    #[test]
    fn test_write_read_le32_roundtrip() {
        let mut buf = [0u8; 8];
        write_le32(&mut buf, 2, 0xDEAD_BEEF);
        assert_eq!(read_le32(&buf, 2), 0xDEAD_BEEF);
        // Verify individual bytes (little-endian).
        assert_eq!(buf[2], 0xEF);
        assert_eq!(buf[3], 0xBE);
        assert_eq!(buf[4], 0xAD);
        assert_eq!(buf[5], 0xDE);
    }

    #[test]
    fn test_generate_dynamic_relocation_executable() {
        let handler = I686RelocationHandler::new();
        // For static executables, most relocations produce no dynamic entry.
        let reloc = make_reloc(R_386_32, 0x100, 0x1000, 0);
        assert!(handler
            .generate_dynamic_relocation(&reloc, &OutputType::Executable)
            .is_none());
        // GLOB_DAT and JMP_SLOT still produce dynamic relocs for executables.
        let reloc_gd = make_reloc(R_386_GLOB_DAT, 0x200, 0x2000, 0);
        let dyn_reloc = handler
            .generate_dynamic_relocation(&reloc_gd, &OutputType::Executable)
            .unwrap();
        assert_eq!(dyn_reloc.reloc_type, R_386_GLOB_DAT);
        assert_eq!(dyn_reloc.offset, 0x200);
    }

    #[test]
    fn test_generate_dynamic_relocation_shared_lib() {
        let handler = I686RelocationHandler::new();
        // R_386_32 in shared lib → R_386_RELATIVE
        let reloc = make_reloc(R_386_32, 0x100, 0x1000, 0x10);
        let dyn_reloc = handler
            .generate_dynamic_relocation(&reloc, &OutputType::SharedLibrary)
            .unwrap();
        assert_eq!(dyn_reloc.reloc_type, R_386_RELATIVE);
        assert_eq!(dyn_reloc.addend, 0x1000 + 0x10);

        // R_386_PLT32 in shared lib → R_386_JMP_SLOT
        let reloc_plt = make_reloc(R_386_PLT32, 0x300, 0x4000, 0);
        let dyn_reloc_plt = handler
            .generate_dynamic_relocation(&reloc_plt, &OutputType::SharedLibrary)
            .unwrap();
        assert_eq!(dyn_reloc_plt.reloc_type, R_386_JMP_SLOT);

        // R_386_PC32 in shared lib → None (resolved at link time)
        let reloc_pc = make_reloc(R_386_PC32, 0x400, 0x5000, 0);
        assert!(handler
            .generate_dynamic_relocation(&reloc_pc, &OutputType::SharedLibrary)
            .is_none());
    }

    #[test]
    fn test_generate_dynamic_relocation_relocatable() {
        let handler = I686RelocationHandler::new();
        // Relocatable objects never produce dynamic relocations.
        let reloc = make_reloc(R_386_32, 0x100, 0x1000, 0);
        assert!(handler
            .generate_dynamic_relocation(&reloc, &OutputType::RelocatableObject)
            .is_none());
    }

    #[test]
    fn test_glob_dat_writes_symbol_address() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        let reloc = make_reloc(R_386_GLOB_DAT, 0, 0xCAFE_BABE, 1);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        // GLOB_DAT writes just S (symbol value), addend is not used.
        assert_eq!(read_le32(&data, 0), 0xCAFE_BABE);
    }

    #[test]
    fn test_jmp_slot_writes_symbol_address() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        let reloc = make_reloc(R_386_JMP_SLOT, 0, 0xDEAD_BEEF, 1);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0xDEAD_BEEF);
    }

    #[test]
    fn test_r386_relative() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // B = symbol_value = 0x0800_0000, A = 0x100
        // value = B + A = 0x0800_0100
        let reloc = make_reloc(R_386_RELATIVE, 0, 0x0800_0000, 0x100);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0x0800_0100);
    }

    #[test]
    fn test_r386_got32() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S (GOT entry address) = 0x3010, GOT base = 0x3000, A = 4 (explicit)
        // G = S - GOT = 0x10
        // value = G + A = 0x10 + 4 = 0x14
        let reloc = make_reloc(R_386_GOT32, 0, 0x3010, 4);
        handler
            .apply_relocation(&reloc, &mut data, 0x3000, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0x14);
    }

    #[test]
    fn test_r386_32plt_absolute_plt() {
        let handler = I686RelocationHandler::new();
        let mut data = [0u8; 8];
        // S (PLT entry address) = 0x4020, A = 0 (implicit, data is zeroed)
        // value = S + A = 0x4020 + 0 = 0x4020
        let reloc = make_reloc(R_386_32PLT, 0, 0x4020, 0);
        handler
            .apply_relocation(&reloc, &mut data, 0, 0)
            .unwrap();
        assert_eq!(read_le32(&data, 0), 0x4020);
    }
}
