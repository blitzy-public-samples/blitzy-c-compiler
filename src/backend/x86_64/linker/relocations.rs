//! x86-64 architecture-specific relocation application module for the BCC built-in linker.
//!
//! Implements the [`ArchRelocationHandler`] trait for x86-64 ELF relocation types,
//! applying machine-code patches to output section data during the linking phase.
//! All relocation computations follow the System V x86-64 ABI supplement and the
//! ELF specification for AMD64.
//!
//! # Supported Relocation Types
//!
//! - **Absolute:** `R_X86_64_64`, `R_X86_64_32`, `R_X86_64_32S`, `R_X86_64_16`, `R_X86_64_8`
//! - **PC-relative:** `R_X86_64_PC32`, `R_X86_64_PC16`, `R_X86_64_PC8`, `R_X86_64_PC64`
//! - **GOT-relative:** `R_X86_64_GOT32`, `R_X86_64_GOTPCREL`, `R_X86_64_GOTPCRELX`,
//!   `R_X86_64_REX_GOTPCRELX`
//! - **PLT-relative:** `R_X86_64_PLT32`
//! - **Dynamic:** `R_X86_64_GLOB_DAT`, `R_X86_64_JUMP_SLOT`, `R_X86_64_RELATIVE`,
//!   `R_X86_64_COPY`
//!
//! # Overflow Checking
//!
//! Truncated relocations (32-bit, 16-bit, 8-bit) are checked for overflow before
//! writing. The handler returns [`RelocationError::Overflow`] if the computed value
//! does not fit in the target field width.
//!
//! # Zero-Dependency Implementation
//!
//! All relocation constants and encoding logic are hand-implemented per the
//! project's zero-dependency mandate. No external crates are used.

use crate::backend::linker_common::relocation::{
    ArchRelocationHandler, RelocationEntry, RelocationError,
};

// ===========================================================================
// x86-64 ELF Relocation Type Constants (hand-defined, zero-dependency)
// ===========================================================================
//
// These constants duplicate the values from assembler/relocations.rs
// X86_64RelocationType for use in the match dispatch without requiring
// cross-module type conversion. Values are defined by the System V AMD64
// ABI supplement (Table 4.10).

/// No relocation; used as a placeholder or padding entry.
const R_X86_64_NONE: u32 = 0;

/// Direct 64-bit absolute relocation. Computation: S + A.
const R_X86_64_64: u32 = 1;

/// PC-relative 32-bit signed relocation. Computation: S + A - P.
const R_X86_64_PC32: u32 = 2;

/// 32-bit GOT entry offset relocation. Computation: G + A.
const R_X86_64_GOT32: u32 = 3;

/// 32-bit PLT-relative relocation for function calls. Computation: L + A - P.
const R_X86_64_PLT32: u32 = 4;

/// Copy relocation — dynamic linker copies symbol data from shared library.
const R_X86_64_COPY: u32 = 5;

/// GOT entry for data symbol — dynamic linker fills at load time. Computation: S.
const R_X86_64_GLOB_DAT: u32 = 6;

/// GOT entry for PLT lazy binding. Computation: S.
const R_X86_64_JUMP_SLOT: u32 = 7;

/// Base-relative relocation for shared library ASLR. Computation: B + A.
const R_X86_64_RELATIVE: u32 = 8;

/// PC-relative GOT entry relocation. Computation: G + GOT + A - P.
const R_X86_64_GOTPCREL: u32 = 9;

/// Direct 32-bit zero-extended relocation. Computation: S + A.
const R_X86_64_32: u32 = 10;

/// Direct 32-bit sign-extended relocation. Computation: S + A.
const R_X86_64_32S: u32 = 11;

/// Direct 16-bit relocation. Computation: S + A.
const R_X86_64_16: u32 = 12;

/// PC-relative 16-bit relocation. Computation: S + A - P.
const R_X86_64_PC16: u32 = 13;

/// Direct 8-bit relocation. Computation: S + A.
const R_X86_64_8: u32 = 14;

/// PC-relative 8-bit relocation. Computation: S + A - P.
const R_X86_64_PC8: u32 = 15;

/// PC-relative 64-bit relocation. Computation: S + A - P.
const R_X86_64_PC64: u32 = 24;

/// PC-relative GOT entry with relaxation optimization. Computation: G + GOT + A - P.
/// The linker may optimize to LEA if the symbol is locally defined (relaxation
/// is handled at a higher level in mod.rs; this handler applies the unrelaxed form).
const R_X86_64_GOTPCRELX: u32 = 41;

/// Same as GOTPCRELX but with a REX prefix in the instruction encoding.
/// The REX prefix is already in the instruction stream; this relocation targets
/// the disp32 displacement field after the REX + opcode bytes.
const R_X86_64_REX_GOTPCRELX: u32 = 42;

// ===========================================================================
// Little-Endian Write Helpers
// ===========================================================================
//
// Each writer performs a bounds check before writing and returns
// RelocationError::InvalidOffset if the write would extend past the buffer.

/// Writes a single byte to the output buffer at the given offset.
///
/// Returns [`RelocationError::InvalidOffset`] if the offset is out of bounds.
#[inline]
fn write_le_u8(
    data: &mut [u8],
    offset: usize,
    value: u8,
) -> Result<(), RelocationError> {
    if offset >= data.len() {
        return Err(RelocationError::InvalidOffset {
            offset: offset as u64,
            section_size: data.len() as u64,
        });
    }
    data[offset] = value;
    Ok(())
}

/// Writes a 16-bit value in little-endian byte order at the given offset.
///
/// Returns [`RelocationError::InvalidOffset`] if the write extends past the buffer.
#[inline]
fn write_le_u16(
    data: &mut [u8],
    offset: usize,
    value: u16,
) -> Result<(), RelocationError> {
    if offset + 2 > data.len() {
        return Err(RelocationError::InvalidOffset {
            offset: offset as u64,
            section_size: data.len() as u64,
        });
    }
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    Ok(())
}

/// Writes a 32-bit value in little-endian byte order at the given offset.
///
/// Returns [`RelocationError::InvalidOffset`] if the write extends past the buffer.
#[inline]
fn write_le_u32(
    data: &mut [u8],
    offset: usize,
    value: u32,
) -> Result<(), RelocationError> {
    if offset + 4 > data.len() {
        return Err(RelocationError::InvalidOffset {
            offset: offset as u64,
            section_size: data.len() as u64,
        });
    }
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
    Ok(())
}

/// Writes a 64-bit value in little-endian byte order at the given offset.
///
/// Returns [`RelocationError::InvalidOffset`] if the write extends past the buffer.
#[inline]
fn write_le_u64(
    data: &mut [u8],
    offset: usize,
    value: u64,
) -> Result<(), RelocationError> {
    if offset + 8 > data.len() {
        return Err(RelocationError::InvalidOffset {
            offset: offset as u64,
            section_size: data.len() as u64,
        });
    }
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
    data[offset + 4] = bytes[4];
    data[offset + 5] = bytes[5];
    data[offset + 6] = bytes[6];
    data[offset + 7] = bytes[7];
    Ok(())
}

// ===========================================================================
// Overflow Checking Helpers
// ===========================================================================
//
// Each checker validates that a computed i64 value fits within the target
// integer range. On failure, returns RelocationError::Overflow carrying
// the relocation type, site offset, computed value, and maximum allowed value
// for diagnostic reporting.

/// Verifies that `value` fits in a signed 32-bit integer (i32 range: -2^31 .. 2^31-1).
#[inline]
fn check_overflow_i32(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < i64::from(i32::MIN) || value > i64::from(i32::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(i32::MAX),
        });
    }
    Ok(())
}

/// Verifies that `value` fits in an unsigned 32-bit integer (u32 range: 0 .. 2^32-1).
#[inline]
fn check_overflow_u32(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < 0 || value > i64::from(u32::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(u32::MAX),
        });
    }
    Ok(())
}

/// Verifies that `value` fits in a signed 16-bit integer (i16 range: -2^15 .. 2^15-1).
#[inline]
fn check_overflow_i16(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < i64::from(i16::MIN) || value > i64::from(i16::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(i16::MAX),
        });
    }
    Ok(())
}

/// Verifies that `value` fits in an unsigned 16-bit integer (u16 range: 0 .. 2^16-1).
#[inline]
fn check_overflow_u16(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < 0 || value > i64::from(u16::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(u16::MAX),
        });
    }
    Ok(())
}

/// Verifies that `value` fits in a signed 8-bit integer (i8 range: -128 .. 127).
#[inline]
fn check_overflow_i8(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < i64::from(i8::MIN) || value > i64::from(i8::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(i8::MAX),
        });
    }
    Ok(())
}

/// Verifies that `value` fits in an unsigned 8-bit integer (u8 range: 0 .. 255).
#[inline]
fn check_overflow_u8(
    value: i64,
    reloc_type: u32,
    offset: u64,
) -> Result<(), RelocationError> {
    if value < 0 || value > i64::from(u8::MAX) {
        return Err(RelocationError::Overflow {
            reloc_type,
            offset,
            value: i128::from(value),
            max_value: i128::from(u8::MAX),
        });
    }
    Ok(())
}

// ===========================================================================
// X86_64RelocationHandler
// ===========================================================================

/// x86-64 architecture-specific relocation handler for the BCC built-in linker.
///
/// This is a stateless unit struct implementing [`ArchRelocationHandler`]. All
/// relocation context (symbol values, addends, GOT/PLT addresses) is passed via
/// method parameters; the handler carries no internal state.
///
/// # Relocation Conventions
///
/// The handler follows the System V AMD64 ABI supplement relocation formulas:
///
/// | Symbol | Meaning |
/// |--------|---------|
/// | S | Resolved symbol virtual address (`reloc.symbol_value`) |
/// | A | RELA-style addend (`reloc.addend`) |
/// | P | Relocation site offset (`reloc.offset`) |
/// | G | GOT entry offset (carried in `symbol_value` for GOT relocations) |
/// | GOT | Base virtual address of `.got` section (`got_address`) |
/// | L | PLT entry address (carried in `symbol_value` for PLT relocations) |
/// | B | Image base address (carried in `symbol_value` for RELATIVE) |
pub struct X86_64RelocationHandler;

impl X86_64RelocationHandler {
    /// Creates a new x86-64 relocation handler instance.
    ///
    /// The handler is stateless; all relocation context is supplied via the
    /// [`ArchRelocationHandler`] trait method parameters.
    #[inline]
    pub fn new() -> Self {
        X86_64RelocationHandler
    }
}

impl ArchRelocationHandler for X86_64RelocationHandler {
    /// Applies a single x86-64 ELF relocation to the output data buffer.
    ///
    /// Dispatches to the appropriate relocation formula based on `reloc.reloc_type`,
    /// computes the final value using wrapping arithmetic for full-range intermediate
    /// results, performs overflow checking for truncated fields, and writes the
    /// result in little-endian byte order at `reloc.offset`.
    ///
    /// # Parameters
    ///
    /// - `reloc`: Resolved relocation entry containing offset, type, symbol value,
    ///   and addend. For GOT-referencing relocations, `symbol_value` carries the
    ///   GOT entry offset (G). For PLT relocations, it carries the PLT entry
    ///   address (L). For RELATIVE, it carries the image base (B).
    /// - `output_data`: Mutable output section data buffer to patch.
    /// - `got_address`: Base virtual address of the `.got` section.
    /// - `plt_address`: Base virtual address of the `.plt` section (unused by most
    ///   relocation types since PLT addresses are resolved into `symbol_value`).
    ///
    /// # Errors
    ///
    /// - [`RelocationError::Overflow`]: Computed value exceeds target field width.
    /// - [`RelocationError::UnsupportedType`]: Unknown relocation type code.
    /// - [`RelocationError::InvalidOffset`]: Relocation offset out of buffer bounds.
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        _plt_address: u64,
    ) -> Result<(), RelocationError> {
        // Extract commonly used relocation terms.
        //
        // All intermediate arithmetic uses i64 with wrapping operations to
        // prevent Rust panics on overflow. The overflow checkers then validate
        // whether the final value fits in the target field width.
        let s = reloc.symbol_value;
        let a = reloc.addend;
        let p = reloc.offset;
        let off = reloc.offset as usize;

        match reloc.reloc_type {
            // =============================================================
            // R_X86_64_NONE (0) — No relocation action required.
            // =============================================================
            R_X86_64_NONE => Ok(()),

            // =============================================================
            // R_X86_64_64 (1) — Direct 64-bit absolute: S + A
            // Full 64-bit range — no overflow check needed.
            // =============================================================
            R_X86_64_64 => {
                let value = (s as i64).wrapping_add(a) as u64;
                write_le_u64(output_data, off, value)
            }

            // =============================================================
            // R_X86_64_PC32 (2) — PC-relative 32-bit signed: S + A - P
            // Must fit in i32 range.
            // =============================================================
            R_X86_64_PC32 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_GOT32 (3) — 32-bit GOT entry offset: G + A
            // symbol_value carries the GOT entry offset (G).
            // =============================================================
            R_X86_64_GOT32 => {
                let value = (s as i64).wrapping_add(a);
                check_overflow_u32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_PLT32 (4) — PLT-relative 32-bit: L + A - P
            // symbol_value carries the PLT entry address (L).
            // =============================================================
            R_X86_64_PLT32 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_COPY (5) — Copy relocation (dynamic linker action).
            // Static linker marks it; the runtime copies symbol data from
            // the shared library into the executable's .bss segment.
            // No patching is performed by the static linker.
            // =============================================================
            R_X86_64_COPY => Ok(()),

            // =============================================================
            // R_X86_64_GLOB_DAT (6) — GOT data entry fill: S
            // Write the resolved symbol address into the GOT entry.
            // Used by the dynamic linker to fill GOT entries at load time.
            // =============================================================
            R_X86_64_GLOB_DAT => {
                write_le_u64(output_data, off, s)
            }

            // =============================================================
            // R_X86_64_JUMP_SLOT (7) — GOT entry for PLT lazy binding: S
            // Write the resolved symbol address into the GOT.PLT entry.
            // The dynamic linker resolves this on the first call.
            // =============================================================
            R_X86_64_JUMP_SLOT => {
                write_le_u64(output_data, off, s)
            }

            // =============================================================
            // R_X86_64_RELATIVE (8) — Base-relative: B + A
            // symbol_value carries the image base address (B).
            // Used in shared objects for ASLR-compatible address fixups.
            // =============================================================
            R_X86_64_RELATIVE => {
                let value = (s as i64).wrapping_add(a) as u64;
                write_le_u64(output_data, off, value)
            }

            // =============================================================
            // R_X86_64_GOTPCREL (9) — PC-relative GOT entry: G + GOT + A - P
            // symbol_value carries the GOT entry offset (G).
            // =============================================================
            R_X86_64_GOTPCREL => {
                let value = (s as i64)
                    .wrapping_add(got_address as i64)
                    .wrapping_add(a)
                    .wrapping_sub(p as i64);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_32 (10) — Direct 32-bit zero-extended: S + A
            // Must fit in u32 range (0 .. 2^32-1).
            // =============================================================
            R_X86_64_32 => {
                let value = (s as i64).wrapping_add(a);
                check_overflow_u32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_32S (11) — Direct 32-bit sign-extended: S + A
            // Must fit in i32 range (-2^31 .. 2^31-1).
            // =============================================================
            R_X86_64_32S => {
                let value = (s as i64).wrapping_add(a);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_16 (12) — Direct 16-bit: S + A
            // Must fit in u16 range (0 .. 65535).
            // =============================================================
            R_X86_64_16 => {
                let value = (s as i64).wrapping_add(a);
                check_overflow_u16(value, reloc.reloc_type, p)?;
                write_le_u16(output_data, off, value as u16)
            }

            // =============================================================
            // R_X86_64_PC16 (13) — PC-relative 16-bit: S + A - P
            // Must fit in i16 range (-32768 .. 32767).
            // =============================================================
            R_X86_64_PC16 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                check_overflow_i16(value, reloc.reloc_type, p)?;
                write_le_u16(output_data, off, value as u16)
            }

            // =============================================================
            // R_X86_64_8 (14) — Direct 8-bit: S + A
            // Must fit in u8 range (0 .. 255).
            // =============================================================
            R_X86_64_8 => {
                let value = (s as i64).wrapping_add(a);
                check_overflow_u8(value, reloc.reloc_type, p)?;
                write_le_u8(output_data, off, value as u8)
            }

            // =============================================================
            // R_X86_64_PC8 (15) — PC-relative 8-bit: S + A - P
            // Must fit in i8 range (-128 .. 127).
            // =============================================================
            R_X86_64_PC8 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64);
                check_overflow_i8(value, reloc.reloc_type, p)?;
                write_le_u8(output_data, off, value as u8)
            }

            // =============================================================
            // R_X86_64_PC64 (24) — PC-relative 64-bit: S + A - P
            // Full 64-bit range — no overflow check needed.
            // =============================================================
            R_X86_64_PC64 => {
                let value = (s as i64).wrapping_add(a).wrapping_sub(p as i64) as u64;
                write_le_u64(output_data, off, value)
            }

            // =============================================================
            // R_X86_64_GOTPCRELX (41) — PC-relative GOT with relaxation:
            //   G + GOT + A - P
            // symbol_value carries the GOT entry offset (G).
            // The linker may optimize to LEA if the symbol is locally
            // defined; that relaxation is handled at a higher level.
            // =============================================================
            R_X86_64_GOTPCRELX => {
                let value = (s as i64)
                    .wrapping_add(got_address as i64)
                    .wrapping_add(a)
                    .wrapping_sub(p as i64);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // R_X86_64_REX_GOTPCRELX (42) — Same formula as GOTPCRELX,
            // but the instruction has a REX prefix already encoded. The
            // relocation targets the disp32 field after REX + opcode bytes.
            // =============================================================
            R_X86_64_REX_GOTPCRELX => {
                let value = (s as i64)
                    .wrapping_add(got_address as i64)
                    .wrapping_add(a)
                    .wrapping_sub(p as i64);
                check_overflow_i32(value, reloc.reloc_type, p)?;
                write_le_u32(output_data, off, value as u32)
            }

            // =============================================================
            // Unknown relocation type — reject with diagnostic info.
            // =============================================================
            _ => Err(RelocationError::UnsupportedType {
                reloc_type: reloc.reloc_type,
            }),
        }
    }

    /// Returns a human-readable name for the given x86-64 relocation type code.
    ///
    /// Used for linker diagnostic messages and error reporting. Returns
    /// `"R_X86_64_UNKNOWN"` for unrecognized type codes.
    fn relocation_name(&self, reloc_type: u32) -> &'static str {
        match reloc_type {
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
            R_X86_64_GOTPCRELX => "R_X86_64_GOTPCRELX",
            R_X86_64_REX_GOTPCRELX => "R_X86_64_REX_GOTPCRELX",
            _ => "R_X86_64_UNKNOWN",
        }
    }

    /// Returns `true` if the given relocation type requires a GOT (Global Offset
    /// Table) entry to be allocated.
    ///
    /// GOT entries are allocated by the dynamic linking section generator for
    /// PIC and shared library output. The following types need GOT entries:
    /// - `R_X86_64_GOT32` (3), `R_X86_64_GOTPCREL` (9)
    /// - `R_X86_64_GOTPCRELX` (41), `R_X86_64_REX_GOTPCRELX` (42)
    /// - `R_X86_64_GLOB_DAT` (6)
    fn needs_got_entry(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_X86_64_GOT32
                | R_X86_64_GOTPCREL
                | R_X86_64_GOTPCRELX
                | R_X86_64_REX_GOTPCRELX
                | R_X86_64_GLOB_DAT
        )
    }

    /// Returns `true` if the given relocation type requires a PLT (Procedure
    /// Linkage Table) entry to be allocated.
    ///
    /// PLT entries are generated for function calls through the PLT in shared
    /// libraries and PIE executables. The following types need PLT entries:
    /// - `R_X86_64_PLT32` (4), `R_X86_64_JUMP_SLOT` (7)
    fn needs_plt_entry(&self, reloc_type: u32) -> bool {
        matches!(reloc_type, R_X86_64_PLT32 | R_X86_64_JUMP_SLOT)
    }

    /// Returns `true` if the given relocation type uses PC-relative addressing.
    ///
    /// PC-relative relocations subtract the relocation site address (P) from
    /// the symbol value: `value = S + A - P` (or GOT/PLT variant thereof).
    fn is_pc_relative(&self, reloc_type: u32) -> bool {
        matches!(
            reloc_type,
            R_X86_64_PC32
                | R_X86_64_PLT32
                | R_X86_64_GOTPCREL
                | R_X86_64_PC16
                | R_X86_64_PC8
                | R_X86_64_PC64
                | R_X86_64_GOTPCRELX
                | R_X86_64_REX_GOTPCRELX
        )
    }

    /// Returns the size in bytes of the relocation patch field.
    ///
    /// - 8 bytes: `R_X86_64_64`, `GLOB_DAT`, `JUMP_SLOT`, `RELATIVE`, `PC64`
    /// - 4 bytes: `PC32`, `GOT32`, `PLT32`, `GOTPCREL`, `32`, `32S`,
    ///   `GOTPCRELX`, `REX_GOTPCRELX`
    /// - 2 bytes: `16`, `PC16`
    /// - 1 byte: `8`, `PC8`
    /// - 0 bytes: `NONE`, `COPY`, and unknown types (no patch data)
    fn relocation_size(&self, reloc_type: u32) -> u8 {
        match reloc_type {
            // 8-byte (64-bit) relocations
            R_X86_64_64
            | R_X86_64_GLOB_DAT
            | R_X86_64_JUMP_SLOT
            | R_X86_64_RELATIVE
            | R_X86_64_PC64 => 8,

            // 4-byte (32-bit) relocations
            R_X86_64_PC32
            | R_X86_64_GOT32
            | R_X86_64_PLT32
            | R_X86_64_GOTPCREL
            | R_X86_64_32
            | R_X86_64_32S
            | R_X86_64_GOTPCRELX
            | R_X86_64_REX_GOTPCRELX => 4,

            // 2-byte (16-bit) relocations
            R_X86_64_16 | R_X86_64_PC16 => 2,

            // 1-byte (8-bit) relocations
            R_X86_64_8 | R_X86_64_PC8 => 1,

            // NONE, COPY, and unknown types have no data field.
            _ => 0,
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to construct a [`RelocationEntry`] for testing.
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

    // ===================================================================
    // Constructor
    // ===================================================================

    #[test]
    fn test_new_creates_handler() {
        let handler = X86_64RelocationHandler::new();
        // Stateless — just verify it can be used.
        assert_eq!(handler.relocation_name(R_X86_64_NONE), "R_X86_64_NONE");
    }

    // ===================================================================
    // R_X86_64_NONE (0) — no-op
    // ===================================================================

    #[test]
    fn test_apply_none_leaves_data_unchanged() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0xAA; 16];
        let reloc = make_reloc(R_X86_64_NONE, 0, 0, 0);
        assert!(handler.apply_relocation(&reloc, &mut data, 0, 0).is_ok());
        assert!(data.iter().all(|&b| b == 0xAA));
    }

    // ===================================================================
    // R_X86_64_64 (1) — Direct 64-bit absolute: S + A
    // ===================================================================

    #[test]
    fn test_apply_64_basic() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_64, 0, 0x0040_1000, 0x10);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0x0040_1010);
    }

    #[test]
    fn test_apply_64_negative_addend() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_64, 4, 0x0040_1000, -4);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
        ]);
        assert_eq!(result, 0x0040_0FFC);
    }

    #[test]
    fn test_apply_64_at_nonzero_offset() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 32];
        let reloc = make_reloc(R_X86_64_64, 8, 0xCAFE_BABE, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);
        assert_eq!(result, 0xCAFE_BABE);
        // Preceding bytes untouched.
        assert!(data[..8].iter().all(|&b| b == 0));
    }

    // ===================================================================
    // R_X86_64_PC32 (2) — PC-relative 32-bit signed: S + A - P
    // ===================================================================

    #[test]
    fn test_apply_pc32_positive_displacement() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x120, A=-4, P=0 → value = 0x120 + (-4) - 0 = 0x11C = 284
        let reloc = make_reloc(R_X86_64_PC32, 0, 0x120, -4);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0x11C);
    }

    #[test]
    fn test_apply_pc32_negative_displacement() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0, A=-4, P=4 → value = 0 + (-4) - 4 = -8
        let reloc = make_reloc(R_X86_64_PC32, 4, 0, -4);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(result, -8);
    }

    #[test]
    fn test_apply_pc32_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S very far from P — exceeds i32 range.
        let reloc = make_reloc(R_X86_64_PC32, 0, 0x1_0000_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            RelocationError::Overflow { reloc_type, .. } => {
                assert_eq!(reloc_type, R_X86_64_PC32);
            }
            other => panic!("Expected Overflow, got {:?}", other),
        }
    }

    // ===================================================================
    // R_X86_64_GOT32 (3) — 32-bit GOT entry offset: G + A
    // ===================================================================

    #[test]
    fn test_apply_got32() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // G=0x18, A=0 → value = 0x18
        let reloc = make_reloc(R_X86_64_GOT32, 0, 0x18, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0x18);
    }

    #[test]
    fn test_apply_got32_with_addend() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // G=0x20, A=8 → value = 0x28
        let reloc = make_reloc(R_X86_64_GOT32, 0, 0x20, 8);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0x28);
    }

    // ===================================================================
    // R_X86_64_PLT32 (4) — PLT-relative 32-bit: L + A - P
    // ===================================================================

    #[test]
    fn test_apply_plt32() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // L=0x1000 (PLT entry), A=-4, P=0 → value = 0x1000 + (-4) - 0 = 0xFFC
        let reloc = make_reloc(R_X86_64_PLT32, 0, 0x1000, -4);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0xFFC);
    }

    #[test]
    fn test_apply_plt32_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_PLT32, 0, 0x1_0000_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_COPY (5) — no-op for static linker
    // ===================================================================

    #[test]
    fn test_apply_copy_leaves_data_unchanged() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0xBB; 16];
        let reloc = make_reloc(R_X86_64_COPY, 0, 0x1234, 0);
        assert!(handler.apply_relocation(&reloc, &mut data, 0, 0).is_ok());
        assert!(data.iter().all(|&b| b == 0xBB));
    }

    // ===================================================================
    // R_X86_64_GLOB_DAT (6) — GOT data entry: S
    // ===================================================================

    #[test]
    fn test_apply_glob_dat() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_GLOB_DAT, 0, 0xDEAD_BEEF_CAFE_BABE, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0xDEAD_BEEF_CAFE_BABE);
    }

    // ===================================================================
    // R_X86_64_JUMP_SLOT (7) — GOT entry for PLT: S
    // ===================================================================

    #[test]
    fn test_apply_jump_slot() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_JUMP_SLOT, 0, 0x0040_3000, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0x0040_3000);
    }

    // ===================================================================
    // R_X86_64_RELATIVE (8) — Base-relative: B + A
    // ===================================================================

    #[test]
    fn test_apply_relative() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // B=0x200000, A=0x1000 → value = 0x201000
        let reloc = make_reloc(R_X86_64_RELATIVE, 0, 0x20_0000, 0x1000);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0x20_1000);
    }

    // ===================================================================
    // R_X86_64_GOTPCREL (9) — PC-relative GOT: G + GOT + A - P
    // ===================================================================

    #[test]
    fn test_apply_gotpcrel() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // G=0x10, GOT=0x3000, A=-4, P=4
        // value = 0x10 + 0x3000 + (-4) - 4 = 0x3008
        let reloc = make_reloc(R_X86_64_GOTPCREL, 4, 0x10, -4);
        handler
            .apply_relocation(&reloc, &mut data, 0x3000, 0)
            .unwrap();
        let result = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(result, 0x3008);
    }

    // ===================================================================
    // R_X86_64_32 (10) — Direct 32-bit zero-extended: S + A
    // ===================================================================

    #[test]
    fn test_apply_32() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_32, 0, 0x1000, 0x20);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0x1020);
    }

    #[test]
    fn test_apply_32_overflow_negative() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // Negative value does not fit in u32.
        let reloc = make_reloc(R_X86_64_32, 0, 0, -1);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_32_overflow_too_large() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_32, 0, 0x1_0000_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_32S (11) — Direct 32-bit sign-extended: S + A
    // ===================================================================

    #[test]
    fn test_apply_32s_positive() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_32S, 0, 0x1000, 0x20);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, 0x1020);
    }

    #[test]
    fn test_apply_32s_negative() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0, A=-10 → value = -10 (fits in i32)
        let reloc = make_reloc(R_X86_64_32S, 0, 0, -10);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(result, -10);
    }

    #[test]
    fn test_apply_32s_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // Exceeds i32::MAX.
        let reloc = make_reloc(R_X86_64_32S, 0, 0x8000_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_16 (12) — Direct 16-bit: S + A
    // ===================================================================

    #[test]
    fn test_apply_16() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_16, 0, 0x1000, 0x234);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u16::from_le_bytes([data[0], data[1]]);
        assert_eq!(result, 0x1234);
    }

    #[test]
    fn test_apply_16_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_16, 0, 0x1_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_PC16 (13) — PC-relative 16-bit: S + A - P
    // ===================================================================

    #[test]
    fn test_apply_pc16() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x200, A=0, P=0 → value = 0x200
        let reloc = make_reloc(R_X86_64_PC16, 0, 0x200, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i16::from_le_bytes([data[0], data[1]]);
        assert_eq!(result, 0x200);
    }

    #[test]
    fn test_apply_pc16_negative() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0, A=0, P=8 → value = -8
        let reloc = make_reloc(R_X86_64_PC16, 8, 0, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = i16::from_le_bytes([data[8], data[9]]);
        assert_eq!(result, -8);
    }

    #[test]
    fn test_apply_pc16_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // Exceeds i16 range.
        let reloc = make_reloc(R_X86_64_PC16, 0, 0x1_0000, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_8 (14) — Direct 8-bit: S + A
    // ===================================================================

    #[test]
    fn test_apply_8() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_8, 0, 0x42, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(data[0], 0x42);
    }

    #[test]
    fn test_apply_8_with_addend() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_8, 0, 0x40, 2);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(data[0], 0x42);
    }

    #[test]
    fn test_apply_8_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(R_X86_64_8, 0, 0x100, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_PC8 (15) — PC-relative 8-bit: S + A - P
    // ===================================================================

    #[test]
    fn test_apply_pc8_positive() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x10, A=0, P=0 → value = 0x10
        let reloc = make_reloc(R_X86_64_PC8, 0, 0x10, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(data[0] as i8, 0x10);
    }

    #[test]
    fn test_apply_pc8_negative() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0, A=0, P=10 → value = -10
        let reloc = make_reloc(R_X86_64_PC8, 10, 0, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        assert_eq!(data[10] as i8, -10);
    }

    #[test]
    fn test_apply_pc8_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x100, A=0, P=0 → value = 0x100 (exceeds i8)
        let reloc = make_reloc(R_X86_64_PC8, 0, 0x100, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // R_X86_64_PC64 (24) — PC-relative 64-bit: S + A - P
    // ===================================================================

    #[test]
    fn test_apply_pc64() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x2000, A=0, P=0 → value = 0x2000
        let reloc = make_reloc(R_X86_64_PC64, 0, 0x2000, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);
        assert_eq!(result, 0x2000);
    }

    #[test]
    fn test_apply_pc64_large_displacement() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // S=0x800_0000_0000, A=0, P=8 → fits in 64 bits, no overflow.
        let reloc = make_reloc(R_X86_64_PC64, 8, 0x800_0000_0000, 0);
        handler.apply_relocation(&reloc, &mut data, 0, 0).unwrap();
        let result = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);
        // 0x800_0000_0000 - 8 = 0x7FF_FFFF_FFF8
        assert_eq!(result, 0x800_0000_0000_u64.wrapping_sub(8));
    }

    // ===================================================================
    // R_X86_64_GOTPCRELX (41) — PC-relative GOT with relaxation
    // ===================================================================

    #[test]
    fn test_apply_gotpcrelx() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // G=0x20, GOT=0x3000, A=-4, P=4
        // value = 0x20 + 0x3000 + (-4) - 4 = 0x3018
        let reloc = make_reloc(R_X86_64_GOTPCRELX, 4, 0x20, -4);
        handler
            .apply_relocation(&reloc, &mut data, 0x3000, 0)
            .unwrap();
        let result = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(result, 0x3018);
    }

    // ===================================================================
    // R_X86_64_REX_GOTPCRELX (42) — Same as GOTPCRELX with REX prefix
    // ===================================================================

    #[test]
    fn test_apply_rex_gotpcrelx() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // Same computation as GOTPCRELX.
        let reloc = make_reloc(R_X86_64_REX_GOTPCRELX, 4, 0x20, -4);
        handler
            .apply_relocation(&reloc, &mut data, 0x3000, 0)
            .unwrap();
        let result = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(result, 0x3018);
    }

    #[test]
    fn test_apply_gotpcrelx_overflow() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        // GOT entry very far away — overflows i32.
        let reloc = make_reloc(R_X86_64_GOTPCRELX, 0, 0, 0);
        let result = handler.apply_relocation(
            &reloc,
            &mut data,
            0x1_0000_0000, // GOT at 4 GiB — exceeds i32
            0,
        );
        assert!(result.is_err());
    }

    // ===================================================================
    // Unsupported relocation type
    // ===================================================================

    #[test]
    fn test_apply_unsupported_type() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 16];
        let reloc = make_reloc(255, 0, 0, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            RelocationError::UnsupportedType { reloc_type } => {
                assert_eq!(reloc_type, 255);
            }
            other => panic!("Expected UnsupportedType, got {:?}", other),
        }
    }

    // ===================================================================
    // Invalid offset — write past buffer boundary
    // ===================================================================

    #[test]
    fn test_apply_64_invalid_offset() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 4]; // Only 4 bytes — R_X86_64_64 needs 8.
        let reloc = make_reloc(R_X86_64_64, 0, 0x1234, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            RelocationError::InvalidOffset { .. } => {}
            other => panic!("Expected InvalidOffset, got {:?}", other),
        }
    }

    #[test]
    fn test_apply_32_invalid_offset() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 2]; // Only 2 bytes — R_X86_64_32 needs 4.
        let reloc = make_reloc(R_X86_64_32, 0, 0x10, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            RelocationError::InvalidOffset { .. } => {}
            other => panic!("Expected InvalidOffset, got {:?}", other),
        }
    }

    #[test]
    fn test_apply_16_invalid_offset() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 1]; // Only 1 byte — R_X86_64_16 needs 2.
        let reloc = make_reloc(R_X86_64_16, 0, 0x10, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_8_invalid_offset() {
        let handler = X86_64RelocationHandler::new();
        let mut data = vec![0u8; 0]; // Empty buffer.
        let reloc = make_reloc(R_X86_64_8, 0, 0x10, 0);
        let result = handler.apply_relocation(&reloc, &mut data, 0, 0);
        assert!(result.is_err());
    }

    // ===================================================================
    // relocation_name
    // ===================================================================

    #[test]
    fn test_relocation_names_known() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_name(0), "R_X86_64_NONE");
        assert_eq!(h.relocation_name(1), "R_X86_64_64");
        assert_eq!(h.relocation_name(2), "R_X86_64_PC32");
        assert_eq!(h.relocation_name(3), "R_X86_64_GOT32");
        assert_eq!(h.relocation_name(4), "R_X86_64_PLT32");
        assert_eq!(h.relocation_name(5), "R_X86_64_COPY");
        assert_eq!(h.relocation_name(6), "R_X86_64_GLOB_DAT");
        assert_eq!(h.relocation_name(7), "R_X86_64_JUMP_SLOT");
        assert_eq!(h.relocation_name(8), "R_X86_64_RELATIVE");
        assert_eq!(h.relocation_name(9), "R_X86_64_GOTPCREL");
        assert_eq!(h.relocation_name(10), "R_X86_64_32");
        assert_eq!(h.relocation_name(11), "R_X86_64_32S");
        assert_eq!(h.relocation_name(12), "R_X86_64_16");
        assert_eq!(h.relocation_name(13), "R_X86_64_PC16");
        assert_eq!(h.relocation_name(14), "R_X86_64_8");
        assert_eq!(h.relocation_name(15), "R_X86_64_PC8");
        assert_eq!(h.relocation_name(24), "R_X86_64_PC64");
        assert_eq!(h.relocation_name(41), "R_X86_64_GOTPCRELX");
        assert_eq!(h.relocation_name(42), "R_X86_64_REX_GOTPCRELX");
    }

    #[test]
    fn test_relocation_name_unknown() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_name(99), "R_X86_64_UNKNOWN");
        assert_eq!(h.relocation_name(255), "R_X86_64_UNKNOWN");
    }

    // ===================================================================
    // needs_got_entry
    // ===================================================================

    #[test]
    fn test_needs_got_entry_true() {
        let h = X86_64RelocationHandler::new();
        assert!(h.needs_got_entry(R_X86_64_GOT32));
        assert!(h.needs_got_entry(R_X86_64_GOTPCREL));
        assert!(h.needs_got_entry(R_X86_64_GOTPCRELX));
        assert!(h.needs_got_entry(R_X86_64_REX_GOTPCRELX));
        assert!(h.needs_got_entry(R_X86_64_GLOB_DAT));
    }

    #[test]
    fn test_needs_got_entry_false() {
        let h = X86_64RelocationHandler::new();
        assert!(!h.needs_got_entry(R_X86_64_NONE));
        assert!(!h.needs_got_entry(R_X86_64_64));
        assert!(!h.needs_got_entry(R_X86_64_PC32));
        assert!(!h.needs_got_entry(R_X86_64_PLT32));
        assert!(!h.needs_got_entry(R_X86_64_COPY));
        assert!(!h.needs_got_entry(R_X86_64_JUMP_SLOT));
        assert!(!h.needs_got_entry(R_X86_64_RELATIVE));
        assert!(!h.needs_got_entry(R_X86_64_32));
        assert!(!h.needs_got_entry(R_X86_64_32S));
        assert!(!h.needs_got_entry(R_X86_64_16));
        assert!(!h.needs_got_entry(R_X86_64_PC16));
        assert!(!h.needs_got_entry(R_X86_64_8));
        assert!(!h.needs_got_entry(R_X86_64_PC8));
        assert!(!h.needs_got_entry(R_X86_64_PC64));
        assert!(!h.needs_got_entry(255));
    }

    // ===================================================================
    // needs_plt_entry
    // ===================================================================

    #[test]
    fn test_needs_plt_entry_true() {
        let h = X86_64RelocationHandler::new();
        assert!(h.needs_plt_entry(R_X86_64_PLT32));
        assert!(h.needs_plt_entry(R_X86_64_JUMP_SLOT));
    }

    #[test]
    fn test_needs_plt_entry_false() {
        let h = X86_64RelocationHandler::new();
        assert!(!h.needs_plt_entry(R_X86_64_NONE));
        assert!(!h.needs_plt_entry(R_X86_64_64));
        assert!(!h.needs_plt_entry(R_X86_64_PC32));
        assert!(!h.needs_plt_entry(R_X86_64_GOT32));
        assert!(!h.needs_plt_entry(R_X86_64_COPY));
        assert!(!h.needs_plt_entry(R_X86_64_GLOB_DAT));
        assert!(!h.needs_plt_entry(R_X86_64_RELATIVE));
        assert!(!h.needs_plt_entry(R_X86_64_GOTPCREL));
        assert!(!h.needs_plt_entry(R_X86_64_32));
        assert!(!h.needs_plt_entry(R_X86_64_GOTPCRELX));
        assert!(!h.needs_plt_entry(R_X86_64_REX_GOTPCRELX));
    }

    // ===================================================================
    // is_pc_relative
    // ===================================================================

    #[test]
    fn test_is_pc_relative_true() {
        let h = X86_64RelocationHandler::new();
        assert!(h.is_pc_relative(R_X86_64_PC32));
        assert!(h.is_pc_relative(R_X86_64_PLT32));
        assert!(h.is_pc_relative(R_X86_64_GOTPCREL));
        assert!(h.is_pc_relative(R_X86_64_PC16));
        assert!(h.is_pc_relative(R_X86_64_PC8));
        assert!(h.is_pc_relative(R_X86_64_PC64));
        assert!(h.is_pc_relative(R_X86_64_GOTPCRELX));
        assert!(h.is_pc_relative(R_X86_64_REX_GOTPCRELX));
    }

    #[test]
    fn test_is_pc_relative_false() {
        let h = X86_64RelocationHandler::new();
        assert!(!h.is_pc_relative(R_X86_64_NONE));
        assert!(!h.is_pc_relative(R_X86_64_64));
        assert!(!h.is_pc_relative(R_X86_64_GOT32));
        assert!(!h.is_pc_relative(R_X86_64_COPY));
        assert!(!h.is_pc_relative(R_X86_64_GLOB_DAT));
        assert!(!h.is_pc_relative(R_X86_64_JUMP_SLOT));
        assert!(!h.is_pc_relative(R_X86_64_RELATIVE));
        assert!(!h.is_pc_relative(R_X86_64_32));
        assert!(!h.is_pc_relative(R_X86_64_32S));
        assert!(!h.is_pc_relative(R_X86_64_16));
        assert!(!h.is_pc_relative(R_X86_64_8));
        assert!(!h.is_pc_relative(255));
    }

    // ===================================================================
    // relocation_size
    // ===================================================================

    #[test]
    fn test_relocation_sizes_8byte() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_size(R_X86_64_64), 8);
        assert_eq!(h.relocation_size(R_X86_64_GLOB_DAT), 8);
        assert_eq!(h.relocation_size(R_X86_64_JUMP_SLOT), 8);
        assert_eq!(h.relocation_size(R_X86_64_RELATIVE), 8);
        assert_eq!(h.relocation_size(R_X86_64_PC64), 8);
    }

    #[test]
    fn test_relocation_sizes_4byte() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_size(R_X86_64_PC32), 4);
        assert_eq!(h.relocation_size(R_X86_64_GOT32), 4);
        assert_eq!(h.relocation_size(R_X86_64_PLT32), 4);
        assert_eq!(h.relocation_size(R_X86_64_GOTPCREL), 4);
        assert_eq!(h.relocation_size(R_X86_64_32), 4);
        assert_eq!(h.relocation_size(R_X86_64_32S), 4);
        assert_eq!(h.relocation_size(R_X86_64_GOTPCRELX), 4);
        assert_eq!(h.relocation_size(R_X86_64_REX_GOTPCRELX), 4);
    }

    #[test]
    fn test_relocation_sizes_2byte() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_size(R_X86_64_16), 2);
        assert_eq!(h.relocation_size(R_X86_64_PC16), 2);
    }

    #[test]
    fn test_relocation_sizes_1byte() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_size(R_X86_64_8), 1);
        assert_eq!(h.relocation_size(R_X86_64_PC8), 1);
    }

    #[test]
    fn test_relocation_sizes_zero() {
        let h = X86_64RelocationHandler::new();
        assert_eq!(h.relocation_size(R_X86_64_NONE), 0);
        assert_eq!(h.relocation_size(R_X86_64_COPY), 0);
        assert_eq!(h.relocation_size(255), 0);
    }

    // ===================================================================
    // Overflow checking helper boundary tests
    // ===================================================================

    #[test]
    fn test_check_overflow_i32_boundaries() {
        assert!(check_overflow_i32(i64::from(i32::MIN), 2, 0).is_ok());
        assert!(check_overflow_i32(i64::from(i32::MAX), 2, 0).is_ok());
        assert!(check_overflow_i32(0, 2, 0).is_ok());
        assert!(check_overflow_i32(i64::from(i32::MIN) - 1, 2, 0).is_err());
        assert!(check_overflow_i32(i64::from(i32::MAX) + 1, 2, 0).is_err());
    }

    #[test]
    fn test_check_overflow_u32_boundaries() {
        assert!(check_overflow_u32(0, 10, 0).is_ok());
        assert!(check_overflow_u32(i64::from(u32::MAX), 10, 0).is_ok());
        assert!(check_overflow_u32(-1, 10, 0).is_err());
        assert!(check_overflow_u32(i64::from(u32::MAX) + 1, 10, 0).is_err());
    }

    #[test]
    fn test_check_overflow_i16_boundaries() {
        assert!(check_overflow_i16(i64::from(i16::MIN), 13, 0).is_ok());
        assert!(check_overflow_i16(i64::from(i16::MAX), 13, 0).is_ok());
        assert!(check_overflow_i16(i64::from(i16::MIN) - 1, 13, 0).is_err());
        assert!(check_overflow_i16(i64::from(i16::MAX) + 1, 13, 0).is_err());
    }

    #[test]
    fn test_check_overflow_u16_boundaries() {
        assert!(check_overflow_u16(0, 12, 0).is_ok());
        assert!(check_overflow_u16(i64::from(u16::MAX), 12, 0).is_ok());
        assert!(check_overflow_u16(-1, 12, 0).is_err());
        assert!(check_overflow_u16(i64::from(u16::MAX) + 1, 12, 0).is_err());
    }

    #[test]
    fn test_check_overflow_i8_boundaries() {
        assert!(check_overflow_i8(i64::from(i8::MIN), 15, 0).is_ok());
        assert!(check_overflow_i8(i64::from(i8::MAX), 15, 0).is_ok());
        assert!(check_overflow_i8(i64::from(i8::MIN) - 1, 15, 0).is_err());
        assert!(check_overflow_i8(i64::from(i8::MAX) + 1, 15, 0).is_err());
    }

    #[test]
    fn test_check_overflow_u8_boundaries() {
        assert!(check_overflow_u8(0, 14, 0).is_ok());
        assert!(check_overflow_u8(i64::from(u8::MAX), 14, 0).is_ok());
        assert!(check_overflow_u8(-1, 14, 0).is_err());
        assert!(check_overflow_u8(i64::from(u8::MAX) + 1, 14, 0).is_err());
    }

    // ===================================================================
    // Write helper boundary tests
    // ===================================================================

    #[test]
    fn test_write_le_u8_exact_boundary() {
        let mut data = vec![0u8; 1];
        assert!(write_le_u8(&mut data, 0, 0xFF).is_ok());
        assert_eq!(data[0], 0xFF);
        assert!(write_le_u8(&mut data, 1, 0).is_err());
    }

    #[test]
    fn test_write_le_u16_exact_boundary() {
        let mut data = vec![0u8; 2];
        assert!(write_le_u16(&mut data, 0, 0xBEEF).is_ok());
        assert_eq!(data, [0xEF, 0xBE]);
        assert!(write_le_u16(&mut data, 1, 0).is_err());
    }

    #[test]
    fn test_write_le_u32_exact_boundary() {
        let mut data = vec![0u8; 4];
        assert!(write_le_u32(&mut data, 0, 0xDEAD_BEEF).is_ok());
        assert_eq!(data, [0xEF, 0xBE, 0xAD, 0xDE]);
        assert!(write_le_u32(&mut data, 1, 0).is_err());
    }

    #[test]
    fn test_write_le_u64_exact_boundary() {
        let mut data = vec![0u8; 8];
        assert!(write_le_u64(&mut data, 0, 0x0102_0304_0506_0708).is_ok());
        assert_eq!(data, [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
        assert!(write_le_u64(&mut data, 1, 0).is_err());
    }

    #[test]
    fn test_write_le_u64_empty_buffer() {
        let mut data: Vec<u8> = vec![];
        assert!(write_le_u64(&mut data, 0, 0).is_err());
    }
}
