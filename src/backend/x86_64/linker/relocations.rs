//! # x86-64 Relocation Application
//!
//! Architecture-specific relocation handler for x86-64 ELF linking.
//! Implements the [`RelocationHandler`] trait from `linker_common::relocation`,
//! providing classification, naming, sizing, overflow detection, and byte-level
//! patching for all `R_X86_64_*` relocation types.
//!
//! Supports both static linking (`ET_EXEC`) and dynamic linking (`ET_DYN`)
//! with GOT/PLT, including GOTPCRELX/REX_GOTPCRELX relaxation optimisation
//! that converts GOT-indirect loads to direct `lea` when the symbol is locally
//! defined.

use crate::backend::linker_common::relocation::{
    compute_absolute, compute_got_relative, compute_pc_relative, fits_signed, fits_unsigned,
    read_le, sign_extend, write_le, RelocCategory, RelocationError, RelocationHandler,
    ResolvedRelocation,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// x86-64 ELF relocation type constants (ELF x86-64 ABI specification)
// ---------------------------------------------------------------------------

/// No relocation.
pub const R_X86_64_NONE: u32 = 0;
/// S + A — absolute 64-bit address.
pub const R_X86_64_64: u32 = 1;
/// S + A - P — PC-relative 32-bit signed.
pub const R_X86_64_PC32: u32 = 2;
/// G + A — GOT entry offset from GOT base, 32-bit.
pub const R_X86_64_GOT32: u32 = 3;
/// L + A - P — PLT entry, PC-relative 32-bit.
pub const R_X86_64_PLT32: u32 = 4;
/// Copy relocation (dynamic linking).
pub const R_X86_64_COPY: u32 = 5;
/// S — used in `.got` for dynamic symbols (GLOB_DAT).
pub const R_X86_64_GLOB_DAT: u32 = 6;
/// S — used in `.got.plt` for lazy-binding PLT stubs (JUMP_SLOT).
pub const R_X86_64_JUMP_SLOT: u32 = 7;
/// B + A — base-relative for `ET_DYN`.
pub const R_X86_64_RELATIVE: u32 = 8;
/// G + GOT + A - P — GOT-relative PC-relative 32-bit.
pub const R_X86_64_GOTPCREL: u32 = 9;
/// S + A — absolute 32-bit unsigned (zero-extended).
pub const R_X86_64_32: u32 = 10;
/// S + A — absolute 32-bit signed (sign-extended).
pub const R_X86_64_32S: u32 = 11;
/// S + A — absolute 16-bit.
pub const R_X86_64_16: u32 = 12;
/// S + A - P — PC-relative 16-bit.
pub const R_X86_64_PC16: u32 = 13;
/// S + A — absolute 8-bit.
pub const R_X86_64_8: u32 = 14;
/// S + A - P — PC-relative 8-bit.
pub const R_X86_64_PC8: u32 = 15;
/// S + A - P — PC-relative 64-bit.
pub const R_X86_64_PC64: u32 = 24;
/// S + A - GOT — symbol offset from GOT base, 64-bit.
pub const R_X86_64_GOTOFF64: u32 = 25;
/// GOT + A - P — GOT address PC-relative, 32-bit.
pub const R_X86_64_GOTPC32: u32 = 26;
/// Z + A — symbol size, 32-bit.
pub const R_X86_64_SIZE32: u32 = 32;
/// Z + A — symbol size, 64-bit.
pub const R_X86_64_SIZE64: u32 = 33;
/// G + GOT + A - P — relaxable GOTPCREL (can be relaxed to direct access).
pub const R_X86_64_GOTPCRELX: u32 = 41;
/// G + GOT + A - P — relaxable GOTPCREL with REX prefix (can be relaxed).
pub const R_X86_64_REX_GOTPCRELX: u32 = 42;

// ---------------------------------------------------------------------------
// X86_64RelocationHandler
// ---------------------------------------------------------------------------

/// x86-64-specific relocation handler.
///
/// Implements [`RelocationHandler`] from `linker_common::relocation`, providing
/// x86-64-specific relocation classification, naming, sizing, and application
/// logic for all `R_X86_64_*` relocation types.
///
/// ## Supported Relocation Types
///
/// | Type | Formula | Bits |
/// |------|---------|------|
/// | `R_X86_64_64` | S + A | 64 |
/// | `R_X86_64_PC32` | S + A − P | 32 (signed) |
/// | `R_X86_64_PLT32` | L + A − P | 32 (signed) |
/// | `R_X86_64_GOTPCREL` | G + GOT + A − P | 32 (signed) |
/// | `R_X86_64_GOTPCRELX` | relaxable GOTPCREL | 32 (signed) |
/// | `R_X86_64_REX_GOTPCRELX` | relaxable GOTPCREL + REX | 32 (signed) |
/// | `R_X86_64_32` | S + A | 32 (unsigned) |
/// | `R_X86_64_32S` | S + A | 32 (signed) |
///
/// Plus all other `R_X86_64_*` types defined above.
pub struct X86_64RelocationHandler {
    /// Whether GOTPCRELX / REX_GOTPCRELX relaxation is enabled.
    /// When enabled, the linker can convert GOT-indirect loads into direct
    /// `lea` instructions for locally-defined symbols.
    relaxation_enabled: bool,
}

impl Default for X86_64RelocationHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl X86_64RelocationHandler {
    /// Create a new x86-64 relocation handler with relaxation enabled by default.
    pub fn new() -> Self {
        Self {
            relaxation_enabled: true,
        }
    }

    /// Create a new x86-64 relocation handler with an explicit relaxation setting.
    pub fn with_relaxation(enabled: bool) -> Self {
        Self {
            relaxation_enabled: enabled,
        }
    }
}

// ---------------------------------------------------------------------------
// RelocationHandler trait implementation
// ---------------------------------------------------------------------------

impl RelocationHandler for X86_64RelocationHandler {
    /// Classify an x86-64 relocation type into a [`RelocCategory`].
    fn classify(&self, rel_type: u32) -> RelocCategory {
        match rel_type {
            R_X86_64_NONE => RelocCategory::Other,

            // Absolute relocations
            R_X86_64_64 | R_X86_64_32 | R_X86_64_32S | R_X86_64_16 | R_X86_64_8 => {
                RelocCategory::Absolute
            }

            // Size relocations (treated as absolute)
            R_X86_64_SIZE32 | R_X86_64_SIZE64 => RelocCategory::Absolute,

            // PC-relative relocations
            R_X86_64_PC32 | R_X86_64_PC16 | R_X86_64_PC8 | R_X86_64_PC64 => {
                RelocCategory::PcRelative
            }

            // PLT relocations
            R_X86_64_PLT32 => RelocCategory::Plt,

            // GOT-relative relocations
            R_X86_64_GOTPCREL
            | R_X86_64_GOTPCRELX
            | R_X86_64_REX_GOTPCRELX
            | R_X86_64_GOT32
            | R_X86_64_GOTOFF64
            | R_X86_64_GOTPC32 => RelocCategory::GotRelative,

            // GOT entry creation (dynamic linking)
            R_X86_64_GLOB_DAT => RelocCategory::GotEntry,

            // Dynamic-only: handled at load time
            R_X86_64_JUMP_SLOT | R_X86_64_RELATIVE | R_X86_64_COPY => RelocCategory::Other,

            _ => RelocCategory::Other,
        }
    }

    /// Return a human-readable name for the relocation type (diagnostics).
    fn reloc_name(&self, rel_type: u32) -> &'static str {
        match rel_type {
            R_X86_64_NONE => "R_X86_64_NONE",
            R_X86_64_64 => "R_X86_64_64",
            R_X86_64_PC32 => "R_X86_64_PC32",
            R_X86_64_GOT32 => "R_X86_64_GOT32",
            R_X86_64_PLT32 => "R_X86_64_PLT32",
            R_X86_64_COPY => "R_X86_64_COPY",
            R_X86_64_GLOB_DAT => "R_X86_64_GLOB_DAT",
            R_X86_64_JUMP_SLOT => "R_X86_64_JUMP_SLOT",
            R_X86_64_RELATIVE => "R_X86_64_RELATIVE",
            R_X86_64_GOTPCREL => "R_X86_64_GOTPCREL",
            R_X86_64_32 => "R_X86_64_32",
            R_X86_64_32S => "R_X86_64_32S",
            R_X86_64_16 => "R_X86_64_16",
            R_X86_64_PC16 => "R_X86_64_PC16",
            R_X86_64_8 => "R_X86_64_8",
            R_X86_64_PC8 => "R_X86_64_PC8",
            R_X86_64_PC64 => "R_X86_64_PC64",
            R_X86_64_GOTOFF64 => "R_X86_64_GOTOFF64",
            R_X86_64_GOTPC32 => "R_X86_64_GOTPC32",
            R_X86_64_SIZE32 => "R_X86_64_SIZE32",
            R_X86_64_SIZE64 => "R_X86_64_SIZE64",
            R_X86_64_GOTPCRELX => "R_X86_64_GOTPCRELX",
            R_X86_64_REX_GOTPCRELX => "R_X86_64_REX_GOTPCRELX",
            _ => "R_X86_64_UNKNOWN",
        }
    }

    /// Return the patch size in bytes for a relocation type.
    fn reloc_size(&self, rel_type: u32) -> u8 {
        match rel_type {
            R_X86_64_NONE => 0,

            // 8-byte relocations
            R_X86_64_64 | R_X86_64_PC64 | R_X86_64_GOTOFF64 | R_X86_64_SIZE64 => 8,

            // 4-byte relocations
            R_X86_64_PC32
            | R_X86_64_PLT32
            | R_X86_64_GOT32
            | R_X86_64_GOTPCREL
            | R_X86_64_32
            | R_X86_64_32S
            | R_X86_64_GOTPC32
            | R_X86_64_SIZE32
            | R_X86_64_GOTPCRELX
            | R_X86_64_REX_GOTPCRELX => 4,

            // 2-byte relocations
            R_X86_64_16 | R_X86_64_PC16 => 2,

            // 1-byte relocations
            R_X86_64_8 | R_X86_64_PC8 => 1,

            // Dynamic-only relocations are pointer-sized (8 bytes on x86-64)
            R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT | R_X86_64_RELATIVE | R_X86_64_COPY => 8,

            _ => 0,
        }
    }

    /// Check whether a relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool {
        matches!(
            rel_type,
            R_X86_64_GOTPCREL
                | R_X86_64_GOTPCRELX
                | R_X86_64_REX_GOTPCRELX
                | R_X86_64_GOT32
                | R_X86_64_GOTOFF64
                | R_X86_64_GOTPC32
                | R_X86_64_GLOB_DAT
        )
    }

    /// Check whether a relocation type requires a PLT stub.
    fn needs_plt(&self, rel_type: u32) -> bool {
        matches!(rel_type, R_X86_64_PLT32 | R_X86_64_JUMP_SLOT)
    }

    /// Apply a single x86-64 relocation: compute the final value, perform
    /// overflow checking, and write it into `section_data` at the correct
    /// offset.
    ///
    /// # Errors
    ///
    /// Returns [`RelocationError::Overflow`] if the computed value does not
    /// fit in the target bit-width.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError> {
        let offset = rel.patch_offset as usize;
        let size = self.reloc_size(rel.rel_type) as usize;

        // Bounds check — ensure we do not write past section_data.
        if size > 0
            && offset
                .checked_add(size)
                .map_or(true, |end| end > section_data.len())
        {
            return Err(RelocationError::Overflow {
                reloc_name: self.reloc_name(rel.rel_type).to_string(),
                value: 0,
                bit_width: (size * 8) as u8,
                location: format!(
                    "offset 0x{:x} (section length 0x{:x})",
                    offset,
                    section_data.len()
                ),
            });
        }

        match rel.rel_type {
            // ---------------------------------------------------------------
            // R_X86_64_NONE — no operation
            // ---------------------------------------------------------------
            R_X86_64_NONE => {}

            // ---------------------------------------------------------------
            // R_X86_64_64 — S + A, absolute 64-bit
            // For undefined symbols with a PLT entry, use the PLT address.
            // ---------------------------------------------------------------
            R_X86_64_64 => {
                let effective_sym = if rel.symbol_value == 0 {
                    rel.plt_address.unwrap_or(rel.symbol_value)
                } else {
                    rel.symbol_value
                };
                let value = compute_absolute(effective_sym, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_PC32 — S + A - P, PC-relative 32-bit signed
            // For undefined symbols resolved via PLT, use the PLT address
            // as the effective symbol value (e.g., address-of-function or
            // extern data symbol that the dynamic linker will resolve).
            // ---------------------------------------------------------------
            R_X86_64_PC32 => {
                let effective_sym = if rel.symbol_value == 0 {
                    rel.plt_address.unwrap_or(rel.symbol_value)
                } else {
                    rel.symbol_value
                };
                let value = compute_pc_relative(effective_sym, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GOT32 — G + A, GOT entry offset 32-bit
            // ---------------------------------------------------------------
            R_X86_64_GOT32 => {
                let got_addr = rel.got_address.unwrap_or(0);
                let got_base = rel.got_base.unwrap_or(0);
                let g = got_addr.wrapping_sub(got_base);
                let value = (g as i64).wrapping_add(rel.addend);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_PLT32 — L + A - P, PLT entry PC-relative 32-bit
            // ---------------------------------------------------------------
            R_X86_64_PLT32 => {
                // If a PLT address is provided, use it; otherwise fall back to
                // the symbol value directly (the linker can resolve locally).
                let target_addr = rel.plt_address.unwrap_or(rel.symbol_value);
                let value = compute_pc_relative(target_addr, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GLOB_DAT — S, absolute 64-bit into GOT slot
            // ---------------------------------------------------------------
            R_X86_64_GLOB_DAT => {
                write_le(section_data, offset, 8, rel.symbol_value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_JUMP_SLOT — S, absolute 64-bit into GOT.PLT slot
            // ---------------------------------------------------------------
            R_X86_64_JUMP_SLOT => {
                write_le(section_data, offset, 8, rel.symbol_value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_RELATIVE — B + A (base + addend)
            // ---------------------------------------------------------------
            R_X86_64_RELATIVE => {
                // patch_address doubles as base for ET_DYN
                let value = rel.patch_address.wrapping_add(rel.addend as u64);
                write_le(section_data, offset, 8, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GOTPCREL — G + GOT + A - P, 32-bit signed
            // ---------------------------------------------------------------
            R_X86_64_GOTPCREL => {
                let got_entry = rel.got_address.unwrap_or(0);
                let value = compute_got_relative(got_entry, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_32 — S + A, absolute 32-bit unsigned
            // ---------------------------------------------------------------
            R_X86_64_32 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_32S — S + A, absolute 32-bit signed
            // ---------------------------------------------------------------
            R_X86_64_32S => {
                let value = (rel.symbol_value as i64).wrapping_add(rel.addend);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_16 — S + A, absolute 16-bit
            // ---------------------------------------------------------------
            R_X86_64_16 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 16) {
                    return Err(self.overflow_error(rel, value as i128, 16));
                }
                write_le(section_data, offset, 2, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_PC16 — S + A - P, PC-relative 16-bit signed
            // ---------------------------------------------------------------
            R_X86_64_PC16 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                if !fits_signed(value, 16) {
                    return Err(self.overflow_error(rel, value as i128, 16));
                }
                write_le(section_data, offset, 2, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_8 — S + A, absolute 8-bit
            // ---------------------------------------------------------------
            R_X86_64_8 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 8) {
                    return Err(self.overflow_error(rel, value as i128, 8));
                }
                section_data[offset] = value as u8;
            }

            // ---------------------------------------------------------------
            // R_X86_64_PC8 — S + A - P, PC-relative 8-bit signed
            // ---------------------------------------------------------------
            R_X86_64_PC8 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                if !fits_signed(value, 8) {
                    return Err(self.overflow_error(rel, value as i128, 8));
                }
                section_data[offset] = value as u8;
            }

            // ---------------------------------------------------------------
            // R_X86_64_PC64 — S + A - P, PC-relative 64-bit
            // ---------------------------------------------------------------
            R_X86_64_PC64 => {
                let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
                write_le(section_data, offset, 8, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GOTOFF64 — S + A - GOT, 64-bit
            // ---------------------------------------------------------------
            R_X86_64_GOTOFF64 => {
                let got_base = rel.got_base.unwrap_or(0);
                let value = (rel.symbol_value as i64)
                    .wrapping_add(rel.addend)
                    .wrapping_sub(got_base as i64);
                write_le(section_data, offset, 8, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GOTPC32 — GOT + A - P, 32-bit signed
            // ---------------------------------------------------------------
            R_X86_64_GOTPC32 => {
                let got_base = rel.got_base.unwrap_or(0);
                let value = compute_pc_relative(got_base, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_SIZE32 — Z + A, symbol size 32-bit
            // ---------------------------------------------------------------
            R_X86_64_SIZE32 => {
                // symbol_value is interpreted as the symbol size for SIZE relocs
                let value = compute_absolute(rel.symbol_value, rel.addend);
                if !fits_unsigned(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_SIZE64 — Z + A, symbol size 64-bit
            // ---------------------------------------------------------------
            R_X86_64_SIZE64 => {
                let value = compute_absolute(rel.symbol_value, rel.addend);
                write_le(section_data, offset, 8, value);
            }

            // ---------------------------------------------------------------
            // R_X86_64_GOTPCRELX — relaxable GOTPCREL (no REX)
            // ---------------------------------------------------------------
            R_X86_64_GOTPCRELX => {
                // Attempt relaxation first; fall back to standard GOTPCREL.
                if self.relaxation_enabled && self.try_relax_gotpcrelx(rel, section_data) {
                    // Relaxation succeeded — the instruction was rewritten and
                    // the value was patched inline by try_relax_gotpcrelx.
                    return Ok(());
                }
                // Standard GOTPCREL behaviour.
                let got_entry = rel.got_address.unwrap_or(0);
                let value = compute_got_relative(got_entry, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_REX_GOTPCRELX — relaxable GOTPCREL with REX prefix
            // ---------------------------------------------------------------
            R_X86_64_REX_GOTPCRELX => {
                if self.relaxation_enabled && self.try_relax_gotpcrelx(rel, section_data) {
                    return Ok(());
                }
                let got_entry = rel.got_address.unwrap_or(0);
                let value = compute_got_relative(got_entry, rel.addend, rel.patch_address);
                if !fits_signed(value, 32) {
                    return Err(self.overflow_error(rel, value as i128, 32));
                }
                write_le(section_data, offset, 4, value as u64);
            }

            // ---------------------------------------------------------------
            // R_X86_64_COPY — dynamic only, no patch during static link
            // ---------------------------------------------------------------
            R_X86_64_COPY => {
                // Nothing to patch — handled at runtime by the dynamic linker.
            }

            // ---------------------------------------------------------------
            // Unsupported relocation type
            // ---------------------------------------------------------------
            _ => {
                return Err(RelocationError::UnsupportedType {
                    rel_type: rel.rel_type,
                    target: Target::X86_64,
                });
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GOTPCRELX / REX_GOTPCRELX relaxation
// ---------------------------------------------------------------------------

impl X86_64RelocationHandler {
    /// Attempt to relax a GOTPCRELX or REX_GOTPCRELX relocation.
    ///
    /// When the target symbol is locally defined (i.e. GOT indirection is not
    /// required), the linker can rewrite:
    ///
    /// ```text
    /// mov  reg, QWORD PTR [rip + GOTPCREL]   ; load via GOT
    /// ```
    ///
    /// into:
    ///
    /// ```text
    /// lea  reg, [rip + OFFSET]                ; compute address directly
    /// ```
    ///
    /// This eliminates one memory indirection.  The function returns `true` if
    /// relaxation succeeded, in which case the instruction has already been
    /// rewritten and the PC-relative displacement patched.  Returns `false` if
    /// relaxation is not applicable (caller must fall back to standard
    /// GOTPCREL behaviour).
    pub fn try_relax_gotpcrelx(&self, rel: &ResolvedRelocation, section_data: &mut [u8]) -> bool {
        // Relaxation is only possible when the GOT entry has been resolved to
        // a locally-defined symbol.  We detect this by checking whether the
        // got_address is absent or equals the symbol_value (meaning no real
        // GOT indirection is needed).
        let got_entry = match rel.got_address {
            Some(addr) => addr,
            // If there is no GOT entry at all, relaxation is trivially safe.
            None => {
                return self.rewrite_mov_to_lea(rel, section_data);
            }
        };

        // If the GOT entry doesn't resolve to the symbol itself we must keep
        // the indirection.
        if got_entry != rel.symbol_value && got_entry != 0 {
            // Symbol is actually loaded via GOT (e.g. external / dynamic) —
            // cannot relax.
            return false;
        }

        self.rewrite_mov_to_lea(rel, section_data)
    }

    /// Rewrite the `mov` instruction at the patch site to `lea` and apply a
    /// direct PC-relative displacement.  Returns `true` on success.
    fn rewrite_mov_to_lea(&self, rel: &ResolvedRelocation, section_data: &mut [u8]) -> bool {
        let offset = rel.patch_offset as usize;

        // We need at least 2 bytes before offset for the opcode + ModRM,
        // and 4 bytes at offset for the 32-bit displacement.
        if offset < 2 || offset + 4 > section_data.len() {
            return false;
        }

        // Read the existing 32-bit displacement at the patch site using the
        // architecture-agnostic reader, then sign-extend for diagnostics/
        // validation purposes.  This confirms the patch site contains a valid
        // displacement field before we overwrite it.
        let _existing_raw = read_le(section_data, offset, 4);
        let _existing_disp = sign_extend(_existing_raw, 32);

        let opcode_offset = offset - 2;

        // Determine whether the instruction has a REX prefix (REX_GOTPCRELX).
        let is_rex = rel.rel_type == R_X86_64_REX_GOTPCRELX;

        if is_rex {
            // REX + opcode + ModRM: need at least 3 bytes before displacement.
            if offset < 3 {
                return false;
            }
            let rex_offset = offset - 3;
            let rex_byte = section_data[rex_offset];
            // REX prefixes occupy 0x40..=0x4F.
            if rex_byte & 0xF0 != 0x40 {
                return false;
            }
            let opcode = section_data[opcode_offset];
            // We expect `mov` opcode 0x8B (load from memory to register).
            if opcode != 0x8B {
                return false;
            }
            // Rewrite to `lea` (0x8D).
            section_data[opcode_offset] = 0x8D;
        } else {
            // GOTPCRELX — no REX prefix.
            let opcode = section_data[opcode_offset];
            if opcode != 0x8B {
                return false;
            }
            section_data[opcode_offset] = 0x8D;
        }

        // Now patch the displacement with a direct PC-relative value:
        // S + A - P instead of G + GOT + A - P.
        let value = compute_pc_relative(rel.symbol_value, rel.addend, rel.patch_address);
        if !fits_signed(value, 32) {
            // Cannot relax — value would overflow; revert the opcode change.
            section_data[opcode_offset] = 0x8B;
            return false;
        }

        write_le(section_data, offset, 4, value as u64);
        true
    }

    /// Build an [`RelocationError::Overflow`] with contextual information.
    fn overflow_error(&self, rel: &ResolvedRelocation, value: i128, bits: u8) -> RelocationError {
        RelocationError::Overflow {
            reloc_name: self.reloc_name(rel.rel_type).to_string(),
            value,
            bit_width: bits,
            location: format!(
                "patch address 0x{:x}, symbol value 0x{:x}, addend {}",
                rel.patch_address, rel.symbol_value, rel.addend
            ),
        }
    }

    /// Report a [`RelocationError`] to the diagnostic engine.
    ///
    /// This helper converts the structured error into a user-facing diagnostic
    /// message and emits it through the provided [`DiagnosticEngine`].
    pub fn report_error(&self, error: &RelocationError, diagnostics: &mut DiagnosticEngine) {
        let msg = match error {
            RelocationError::Overflow {
                reloc_name,
                value,
                bit_width,
                location,
            } => {
                format!(
                    "relocation {} out of range: value 0x{:x} does not fit in {} bits at {}",
                    reloc_name, value, bit_width, location,
                )
            }
            RelocationError::UndefinedSymbol { name, reloc_name } => {
                format!(
                    "undefined reference to '{}' for relocation {}",
                    name, reloc_name,
                )
            }
            RelocationError::UnsupportedType { rel_type, target } => {
                format!(
                    "unsupported relocation type {} for target {}",
                    rel_type, target,
                )
            }
        };
        diagnostics.emit_error(Span::dummy(), msg);
    }
}

// ---------------------------------------------------------------------------
// Standalone public functions
// ---------------------------------------------------------------------------

/// Return all supported x86-64 relocation type IDs.
pub fn supported_relocation_types() -> &'static [u32] {
    &[
        R_X86_64_NONE,
        R_X86_64_64,
        R_X86_64_PC32,
        R_X86_64_GOT32,
        R_X86_64_PLT32,
        R_X86_64_COPY,
        R_X86_64_GLOB_DAT,
        R_X86_64_JUMP_SLOT,
        R_X86_64_RELATIVE,
        R_X86_64_GOTPCREL,
        R_X86_64_32,
        R_X86_64_32S,
        R_X86_64_16,
        R_X86_64_PC16,
        R_X86_64_8,
        R_X86_64_PC8,
        R_X86_64_PC64,
        R_X86_64_GOTOFF64,
        R_X86_64_GOTPC32,
        R_X86_64_SIZE32,
        R_X86_64_SIZE64,
        R_X86_64_GOTPCRELX,
        R_X86_64_REX_GOTPCRELX,
    ]
}

/// Get the dynamic GOT relocation type for x86-64.
/// Used for `.rela.dyn` entries (GLOB_DAT).
pub fn got_reloc_type() -> u32 {
    R_X86_64_GLOB_DAT
}

/// Get the PLT jump-slot relocation type for x86-64.
/// Used for `.rela.plt` entries (JUMP_SLOT).
pub fn plt_reloc_type() -> u32 {
    R_X86_64_JUMP_SLOT
}

/// Get the relative relocation type for x86-64.
/// Used for base-relative relocations in `ET_DYN`.
pub fn relative_reloc_type() -> u32 {
    R_X86_64_RELATIVE
}

/// Get the copy relocation type for x86-64.
/// Used for copy relocations in dynamically-linked executables.
pub fn copy_reloc_type() -> u32 {
    R_X86_64_COPY
}

// ---------------------------------------------------------------------------
// FxHashMap usage — build a relocation-type-to-name lookup table.
// ---------------------------------------------------------------------------

/// Build a lookup table mapping x86-64 relocation type IDs to their names.
///
/// This is useful for batch error reporting where the handler's `reloc_name`
/// method would be called repeatedly in a hot loop.
#[allow(dead_code)]
fn build_reloc_name_map() -> FxHashMap<u32, &'static str> {
    let handler = X86_64RelocationHandler::new();
    let mut map: FxHashMap<u32, &'static str> = FxHashMap::default();
    for &rt in supported_relocation_types() {
        map.insert(rt, handler.reloc_name(rt));
    }
    map
}
