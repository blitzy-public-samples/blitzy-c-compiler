//! Architecture-agnostic relocation processing framework for the BCC built-in linker.
//!
//! Collects relocations from input object files, translates input-section-relative
//! offsets to output-section-relative offsets, classifies relocations for GOT/PLT
//! needs, and dispatches to architecture-specific handlers via the
//! [`ArchRelocationHandler`] trait.
//!
//! Each architecture backend (x86-64, i686, AArch64, RISC-V 64) implements
//! `ArchRelocationHandler` to apply its target-specific relocation types
//! (e.g., `R_X86_64_PC32`, `R_AARCH64_CALL26`, `R_RISCV_JAL`).
//!
//! Removing this module would break all relocation processing, making it
//! impossible to link object files into executables or shared objects.

use crate::common::fx_hash::FxHashSet;
use std::fmt;

// Re-import InputRelocation from section_merger (unified input data model)
use super::section_merger::InputRelocation;

// ===========================================================================
// RelocationEntry — resolved relocation for output
// ===========================================================================

/// A fully resolved relocation ready for application to output section data.
///
/// Created by resolving an [`InputRelocation`] against the final symbol table
/// and output section layout.
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Offset within the output section where the relocation is applied.
    pub offset: u64,
    /// Architecture-specific relocation type code.
    pub reloc_type: u32,
    /// Name of the target symbol.
    pub symbol_name: String,
    /// Resolved value (virtual address) of the target symbol.
    pub symbol_value: u64,
    /// Addend for RELA-style relocations.
    pub addend: i64,
    /// Index of the output section containing this relocation.
    pub output_section: usize,
}

// ===========================================================================
// RelocationError — relocation processing errors
// ===========================================================================

/// Errors that can occur during relocation processing and application.
#[derive(Debug, Clone)]
pub enum RelocationError {
    /// The computed relocation value does not fit in the target field.
    /// For example, a 32-bit PC-relative relocation to a symbol more than
    /// 2 GiB away.
    Overflow {
        /// Architecture-specific relocation type code.
        reloc_type: u32,
        /// Offset where the relocation was being applied.
        offset: u64,
        /// Computed value that does not fit.
        value: i128,
        /// Maximum representable value for this relocation type.
        max_value: i128,
    },
    /// The relocation references a symbol that was not resolved.
    UndefinedSymbol {
        /// Name of the undefined symbol.
        name: String,
    },
    /// The relocation type is not supported by the architecture handler.
    UnsupportedType {
        /// The unsupported relocation type code.
        reloc_type: u32,
    },
    /// The relocation offset is outside the bounds of its section.
    InvalidOffset {
        /// The invalid offset.
        offset: u64,
        /// Size of the section.
        section_size: u64,
    },
}

impl fmt::Display for RelocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelocationError::Overflow {
                reloc_type,
                offset,
                value,
                max_value,
            } => {
                write!(
                    f,
                    "relocation overflow: type {} at offset {:#x}, value {} exceeds maximum {}",
                    reloc_type, offset, value, max_value
                )
            }
            RelocationError::UndefinedSymbol { name } => {
                write!(f, "relocation references undefined symbol `{}`", name)
            }
            RelocationError::UnsupportedType { reloc_type } => {
                write!(f, "unsupported relocation type {}", reloc_type)
            }
            RelocationError::InvalidOffset {
                offset,
                section_size,
            } => {
                write!(
                    f,
                    "relocation offset {:#x} exceeds section size {:#x}",
                    offset, section_size
                )
            }
        }
    }
}

// ===========================================================================
// ArchRelocationHandler trait — dispatch to architecture-specific handlers
// ===========================================================================

/// Trait for architecture-specific relocation application.
///
/// Each target backend (x86-64, i686, AArch64, RISC-V 64) implements this
/// trait to handle its relocation types. The [`RelocationProcessor`] dispatches
/// to this trait during the apply phase.
pub trait ArchRelocationHandler {
    /// Applies a single relocation to the output data buffer.
    ///
    /// # Parameters
    /// - `reloc`: The resolved relocation entry.
    /// - `output_data`: Mutable slice of the output section data.
    /// - `got_address`: Base address of the `.got` section (for GOT-relative).
    /// - `plt_address`: Base address of the `.plt` section (for PLT-relative).
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), RelocationError>;

    /// Returns a human-readable name for the given relocation type code.
    fn relocation_name(&self, reloc_type: u32) -> &'static str;

    /// Returns true if this relocation type requires a GOT entry.
    fn needs_got_entry(&self, reloc_type: u32) -> bool;

    /// Returns true if this relocation type requires a PLT entry.
    fn needs_plt_entry(&self, reloc_type: u32) -> bool;

    /// Returns true if this relocation type is PC-relative.
    fn is_pc_relative(&self, reloc_type: u32) -> bool;

    /// Returns the size in bytes of the relocation field.
    fn relocation_size(&self, reloc_type: u32) -> u8;
}

// ===========================================================================
// PendingRelocation — internal tracking of collected relocations
// ===========================================================================

/// Internal representation of a relocation collected from an input section,
/// with the offset translated to the output section coordinate space.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PendingRelocation {
    /// The original input relocation data.
    input_reloc: InputRelocation,
    /// Index of the source object file.
    object_index: usize,
    /// Index of the input section within the source object.
    input_section_index: usize,
    /// Index of the output section this relocation targets.
    output_section_index: usize,
    /// Offset within the output section (input offset + section placement offset).
    output_offset: u64,
}

// ===========================================================================
// RelocationClassification — GOT/PLT needs analysis
// ===========================================================================

/// Result of classifying relocations for PIC/shared library support.
///
/// Identifies which symbols need GOT entries, PLT entries, or copy
/// relocations for dynamic linking.
#[derive(Debug, Clone, Default)]
pub struct RelocationClassification {
    /// Symbols requiring GOT (Global Offset Table) entries.
    pub got_entries: Vec<String>,
    /// Symbols requiring PLT (Procedure Linkage Table) entries.
    pub plt_entries: Vec<String>,
    /// Symbols requiring copy relocations.
    pub copy_relocs: Vec<String>,
}

// ===========================================================================
// RelocationProcessor — collect and apply relocations
// ===========================================================================

/// Architecture-agnostic relocation processing engine.
///
/// Collects relocations from input object files during section merging,
/// then applies them to the output using an architecture-specific handler.
pub struct RelocationProcessor {
    /// Collected relocations awaiting application.
    pending_relocations: Vec<PendingRelocation>,
    /// Count of successfully applied relocations.
    applied_count: usize,
    /// Errors encountered during application.
    errors: Vec<RelocationError>,
}

impl RelocationProcessor {
    /// Creates a new, empty relocation processor.
    pub fn new() -> Self {
        Self {
            pending_relocations: Vec::new(),
            applied_count: 0,
            errors: Vec::new(),
        }
    }

    /// Collects relocations from one input section, translating offsets from
    /// input-section-relative to output-section-relative using the section
    /// merger's placement information.
    ///
    /// # Parameters
    /// - `object_index`: Index of the source object file.
    /// - `section_index`: Index of the input section within the source object.
    /// - `relocations`: Relocation entries from this input section.
    /// - `output_section_index`: Index of the output section these belong to.
    /// - `section_output_offset`: Byte offset of this input section within
    ///   the output section (from `MergedInput::offset_in_output`).
    pub fn collect_relocations(
        &mut self,
        object_index: usize,
        section_index: usize,
        relocations: &[InputRelocation],
        output_section_index: usize,
        section_output_offset: u64,
    ) {
        for reloc in relocations {
            self.pending_relocations.push(PendingRelocation {
                input_reloc: reloc.clone(),
                object_index,
                input_section_index: section_index,
                output_section_index,
                output_offset: reloc.offset + section_output_offset,
            });
        }
    }

    /// Applies all collected relocations to the output section data.
    ///
    /// For each pending relocation:
    /// 1. Resolves the symbol value from `symbols`.
    /// 2. Constructs a `RelocationEntry` with the output-section offset.
    /// 3. Dispatches to the architecture handler for patching.
    ///
    /// # Returns
    /// - `Ok(())` if all relocations applied successfully.
    /// - `Err(Vec<RelocationError>)` if any relocations failed.
    pub fn apply_relocations(
        &mut self,
        handler: &dyn ArchRelocationHandler,
        symbols: &super::symbol_resolver::ResolvedSymbols,
        output_sections: &mut [super::section_merger::OutputSection],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), Vec<RelocationError>> {
        self.errors.clear();
        self.applied_count = 0;

        for pending in &self.pending_relocations {
            // Resolve the symbol — look up by index in the symbol table
            let sym_index = pending.input_reloc.symbol_index as usize;

            let (sym_name, sym_value) = if sym_index < symbols.symbols.len() {
                let sym = &symbols.symbols[sym_index];
                (sym.name.clone(), sym.value)
            } else {
                self.errors.push(RelocationError::UndefinedSymbol {
                    name: format!("symbol_index_{}", sym_index),
                });
                continue;
            };

            let entry = RelocationEntry {
                offset: pending.output_offset,
                reloc_type: pending.input_reloc.reloc_type,
                symbol_name: sym_name,
                symbol_value: sym_value,
                addend: pending.input_reloc.addend,
                output_section: pending.output_section_index,
            };

            // Get mutable access to the output section's collected data
            if pending.output_section_index >= output_sections.len() {
                self.errors.push(RelocationError::InvalidOffset {
                    offset: pending.output_offset,
                    section_size: 0,
                });
                continue;
            }

            let section = &mut output_sections[pending.output_section_index];

            // Collect all input section data into a single buffer for patching
            let mut section_data: Vec<u8> = Vec::new();
            for merged in &section.input_sections {
                // Pad to alignment
                let pad = super::section_merger::compute_padding(
                    section_data.len() as u64,
                    merged.input.alignment.max(1),
                );
                section_data.extend(std::iter::repeat(0u8).take(pad as usize));
                section_data.extend_from_slice(&merged.input.data);
            }

            // Apply the relocation to the collected data
            match handler.apply_relocation(&entry, &mut section_data, got_address, plt_address) {
                Ok(()) => {
                    self.applied_count += 1;
                }
                Err(e) => {
                    self.errors.push(e);
                }
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    /// Classifies all collected relocations to determine which symbols need
    /// GOT entries, PLT entries, or copy relocations for PIC/shared library
    /// support.
    pub fn classify_relocations(
        &self,
        handler: &dyn ArchRelocationHandler,
    ) -> RelocationClassification {
        let mut got_set = FxHashSet::default();
        let mut plt_set = FxHashSet::default();
        let copy_set: FxHashSet<String> = FxHashSet::default();

        for pending in &self.pending_relocations {
            let reloc_type = pending.input_reloc.reloc_type;
            let sym_key = format!("sym_{}", pending.input_reloc.symbol_index);

            if handler.needs_got_entry(reloc_type) {
                got_set.insert(sym_key.clone());
            }
            if handler.needs_plt_entry(reloc_type) {
                plt_set.insert(sym_key);
            }
        }

        RelocationClassification {
            got_entries: got_set.into_iter().collect(),
            plt_entries: plt_set.into_iter().collect(),
            copy_relocs: copy_set.into_iter().collect(),
        }
    }

    /// Returns the number of successfully applied relocations.
    pub fn applied_count(&self) -> usize {
        self.applied_count
    }

    /// Returns the number of pending (unapplied) relocations.
    pub fn pending_count(&self) -> usize {
        self.pending_relocations.len()
    }

    /// Returns any errors accumulated during the last apply pass.
    pub fn errors(&self) -> &[RelocationError] {
        &self.errors
    }
}

impl Default for RelocationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relocation_error_display() {
        let err = RelocationError::Overflow {
            reloc_type: 2,
            offset: 0x1000,
            value: 0x1_0000_0000,
            max_value: 0xFFFF_FFFF,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overflow"));
        assert!(msg.contains("0x1000"));

        let err = RelocationError::UndefinedSymbol {
            name: "missing".to_string(),
        };
        assert!(format!("{}", err).contains("missing"));

        let err = RelocationError::UnsupportedType { reloc_type: 99 };
        assert!(format!("{}", err).contains("99"));

        let err = RelocationError::InvalidOffset {
            offset: 0x500,
            section_size: 0x100,
        };
        assert!(format!("{}", err).contains("0x500"));
    }

    #[test]
    fn test_relocation_processor_new() {
        let proc = RelocationProcessor::new();
        assert_eq!(proc.pending_count(), 0);
        assert_eq!(proc.applied_count(), 0);
        assert!(proc.errors().is_empty());
    }

    #[test]
    fn test_collect_relocations() {
        let mut proc = RelocationProcessor::new();
        let relocs = vec![
            InputRelocation {
                offset: 0x10,
                reloc_type: 1,
                symbol_index: 0,
                addend: 0,
                section_index: 1,
            },
            InputRelocation {
                offset: 0x20,
                reloc_type: 2,
                symbol_index: 1,
                addend: -4,
                section_index: 1,
            },
        ];

        proc.collect_relocations(0, 1, &relocs, 0, 0x100);

        assert_eq!(proc.pending_count(), 2);
    }

    #[test]
    fn test_relocation_classification_default() {
        let class = RelocationClassification::default();
        assert!(class.got_entries.is_empty());
        assert!(class.plt_entries.is_empty());
        assert!(class.copy_relocs.is_empty());
    }

    #[test]
    fn test_relocation_entry_fields() {
        let entry = RelocationEntry {
            offset: 0x1000,
            reloc_type: 10,
            symbol_name: "foo".to_string(),
            symbol_value: 0x4000,
            addend: -4,
            output_section: 0,
        };
        assert_eq!(entry.offset, 0x1000);
        assert_eq!(entry.reloc_type, 10);
        assert_eq!(entry.symbol_name, "foo");
        assert_eq!(entry.symbol_value, 0x4000);
        assert_eq!(entry.addend, -4);
        assert_eq!(entry.output_section, 0);
    }
}
