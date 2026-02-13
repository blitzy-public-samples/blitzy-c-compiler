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
//! # Pipeline Integration
//!
//! The relocation processor sits between the section merger (which determines
//! output section layout) and the ELF writer (which produces the final binary):
//!
//! ```text
//! SectionMerger → RelocationProcessor → ArchRelocationHandler → ELF Writer
//! ```
//!
//! # PIC Support
//!
//! For position-independent code (`-fPIC`) and shared libraries (`-shared`),
//! the [`classify_relocations`](RelocationProcessor::classify_relocations)
//! method identifies which symbols need GOT/PLT entries, enabling the dynamic
//! linking section generator to allocate the necessary structures.

use std::fmt;

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// InputRelocation is canonically defined in section_merger as part of the
// unified input object data model. Re-exported here for linker modules that
// operate primarily on relocation data.
pub use super::section_merger::InputRelocation;
use super::section_merger::{compute_padding, OutputSection};
use super::symbol_resolver::ResolvedSymbols;

// ===========================================================================
// RelocationEntry — fully resolved relocation for output application
// ===========================================================================

/// A fully resolved relocation ready for application to output section data.
///
/// Created by resolving an [`InputRelocation`] against the final symbol table
/// and output section layout. The architecture-specific
/// [`ArchRelocationHandler`] consumes these entries to patch the output binary.
#[derive(Debug, Clone)]
pub struct RelocationEntry {
    /// Byte offset within the output section where the relocation is applied.
    pub offset: u64,
    /// Architecture-specific relocation type code (e.g., `R_X86_64_PC32 = 2`).
    pub reloc_type: u32,
    /// Name of the target symbol (for diagnostics and dynamic symbol references).
    pub symbol_name: String,
    /// Resolved virtual address of the target symbol.
    pub symbol_value: u64,
    /// Addend for RELA-style relocations (all four BCC architectures use RELA).
    pub addend: i64,
    /// Index of the output section containing this relocation.
    pub output_section: usize,
}

// ===========================================================================
// RelocationError — relocation processing errors
// ===========================================================================

/// Errors that can occur during relocation processing and application.
///
/// Each variant carries structured context information for precise error
/// reporting. The [`Display`] implementation formats these into human-readable
/// linker diagnostic messages.
#[derive(Debug, Clone)]
pub enum RelocationError {
    /// The computed relocation value does not fit in the target field.
    ///
    /// For example, a 32-bit PC-relative relocation to a symbol more than
    /// 2 GiB away from the relocation site.
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
    /// The relocation offset is outside the bounds of its output section.
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
                    "relocation overflow: type {} at offset {:#x}, \
                     value {} exceeds maximum {}",
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
/// to this trait during the apply phase, keeping the relocation framework
/// fully architecture-agnostic.
///
/// # Contract
///
/// Implementations MUST:
/// - Patch `output_data` at `reloc.offset` with the computed relocation value.
/// - Return [`RelocationError::Overflow`] if the value exceeds the field width.
/// - Return [`RelocationError::UnsupportedType`] for unknown relocation codes.
/// - Handle both PIC (GOT/PLT-relative) and non-PIC relocation types.
pub trait ArchRelocationHandler {
    /// Applies a single relocation to the output data buffer.
    ///
    /// The handler reads the existing value at `output_data[reloc.offset..]`,
    /// computes the final relocation value using the architecture-specific
    /// formula, and writes the result back.
    ///
    /// # Parameters
    /// - `reloc`: The resolved relocation entry with symbol value and addend.
    /// - `output_data`: Mutable slice of the full output section data.
    /// - `got_address`: Base virtual address of the `.got` section.
    /// - `plt_address`: Base virtual address of the `.plt` section.
    fn apply_relocation(
        &self,
        reloc: &RelocationEntry,
        output_data: &mut [u8],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), RelocationError>;

    /// Returns a human-readable name for the given relocation type code.
    ///
    /// Used for diagnostic messages. For example, type 2 on x86-64 returns
    /// `"R_X86_64_PC32"`.
    fn relocation_name(&self, reloc_type: u32) -> &'static str;

    /// Returns `true` if this relocation type requires a GOT entry.
    ///
    /// GOT entries are allocated by the dynamic linking section generator
    /// for PIC/shared library output.
    fn needs_got_entry(&self, reloc_type: u32) -> bool;

    /// Returns `true` if this relocation type requires a PLT entry.
    ///
    /// PLT entries are allocated for function calls through the Procedure
    /// Linkage Table in shared libraries and PIE executables.
    fn needs_plt_entry(&self, reloc_type: u32) -> bool;

    /// Returns `true` if this relocation type computes a PC-relative value.
    ///
    /// PC-relative relocations use: `value = S + A - P`
    /// where S = symbol value, A = addend, P = relocation virtual address.
    fn is_pc_relative(&self, reloc_type: u32) -> bool;

    /// Returns the size in bytes of the relocation field.
    ///
    /// Common sizes: 1 (byte), 2 (word), 4 (dword), 8 (qword).
    fn relocation_size(&self, reloc_type: u32) -> u8;
}

// ===========================================================================
// PendingRelocation — internal tracking of collected relocations
// ===========================================================================

/// Internal representation of a relocation collected from an input section,
/// with the offset translated to the output section coordinate space.
#[derive(Debug, Clone)]
struct PendingRelocation {
    /// The original input relocation data (offset, type, symbol, addend, section).
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
/// relocations for dynamic linking. Used by the dynamic section generator
/// ([`super::dynamic`]) to allocate the appropriate linking structures.
#[derive(Debug, Clone, Default)]
pub struct RelocationClassification {
    /// Symbol names requiring GOT (Global Offset Table) entries.
    pub got_entries: Vec<String>,
    /// Symbol names requiring PLT (Procedure Linkage Table) entries.
    pub plt_entries: Vec<String>,
    /// Symbol names requiring copy relocations (for shared library data imports).
    pub copy_relocs: Vec<String>,
}

// ===========================================================================
// RelocationProcessor — core collection and application engine
// ===========================================================================

/// Architecture-agnostic relocation processing engine.
///
/// Collects relocations from input object files during section merging,
/// then applies them to the output using an architecture-specific handler.
///
/// # Workflow
///
/// ```text
/// 1. processor.set_target(target);
/// 2. processor.register_object_symbols(0, names_from_obj0);
/// 3. processor.collect_relocations(0, section_idx, &relocs, out_idx, offset);
/// 4. let classification = processor.classify_relocations(&handler);
///    // → allocate GOT/PLT entries based on classification
/// 5. processor.apply_relocations(&handler, &symbols, &mut sections, got, plt)?;
/// ```
pub struct RelocationProcessor {
    /// Collected relocations awaiting application.
    pending_relocations: Vec<PendingRelocation>,
    /// Count of successfully applied relocations.
    applied_count: usize,
    /// Errors encountered during the most recent apply pass.
    errors: Vec<RelocationError>,
    /// Per-object symbol name tables for resolving input symbol indices to
    /// names. Maps `object_index → Vec<symbol_name>` where the vector index
    /// matches the symbol index in the input object's symbol table.
    object_symbol_names: FxHashMap<usize, Vec<String>>,
    /// Optional target architecture for architecture-aware diagnostics and
    /// relocation field size validation.
    target: Option<Target>,
}

impl RelocationProcessor {
    /// Creates a new, empty relocation processor.
    pub fn new() -> Self {
        Self {
            pending_relocations: Vec::new(),
            applied_count: 0,
            errors: Vec::new(),
            object_symbol_names: FxHashMap::default(),
            target: None,
        }
    }

    /// Sets the target architecture for architecture-aware diagnostics.
    ///
    /// When set, error messages include the ELF machine type and relocation
    /// field size validation uses the target's pointer width as a reference.
    pub fn set_target(&mut self, target: Target) {
        self.target = Some(target);
    }

    /// Registers symbol names for an input object file.
    ///
    /// The symbol names are indexed by their position in the input object's
    /// symbol table. During relocation application, the processor uses these
    /// names to look up resolved symbol values by name in the
    /// [`ResolvedSymbols`] table.
    pub fn register_object_symbols(&mut self, object_index: usize, symbol_names: Vec<String>) {
        self.object_symbol_names.insert(object_index, symbol_names);
    }

    /// Collects relocations from one input section, translating offsets from
    /// input-section-relative to output-section-relative using the section
    /// merger's placement information.
    ///
    /// The output offset for each relocation is computed as:
    /// `output_offset = input_reloc.offset + section_output_offset`
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
        self.pending_relocations.reserve(relocations.len());
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
    /// For each pending relocation this method:
    /// 1. Resolves the symbol value from `symbols`.
    /// 2. Constructs a [`RelocationEntry`] with the output-section offset.
    /// 3. Validates the offset against the section bounds.
    /// 4. Dispatches to the architecture handler for patching.
    ///
    /// Relocations are processed in batches grouped by output section to
    /// minimize buffer construction overhead. The flat section data buffer
    /// is built once per section, all relocations for that section are
    /// applied, and the patched data is written back to the individual input
    /// section data buffers.
    ///
    /// # Returns
    /// - `Ok(())` if all relocations applied successfully.
    /// - `Err(Vec<RelocationError>)` containing all errors encountered.
    pub fn apply_relocations(
        &mut self,
        handler: &dyn ArchRelocationHandler,
        symbols: &ResolvedSymbols,
        output_sections: &mut [OutputSection],
        got_address: u64,
        plt_address: u64,
    ) -> Result<(), Vec<RelocationError>> {
        self.errors.clear();
        self.applied_count = 0;

        // Group pending relocations by output section index for batch
        // processing. FxHashMap::entry() amortises allocation for sections
        // with many relocations.
        let mut section_reloc_indices: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (idx, pending) in self.pending_relocations.iter().enumerate() {
            section_reloc_indices
                .entry(pending.output_section_index)
                .or_default()
                .push(idx);
        }

        // Process each output section batch.
        for (&section_idx, reloc_indices) in &section_reloc_indices {
            // Validate section index is in range.
            if section_idx >= output_sections.len() {
                for &ri in reloc_indices {
                    let pending = &self.pending_relocations[ri];
                    self.errors.push(RelocationError::InvalidOffset {
                        offset: pending.output_offset,
                        section_size: 0,
                    });
                }
                continue;
            }

            // Build a flat data buffer from all input sections merged into
            // this output section. Alignment padding is filled with zeros.
            let flat_capacity = compute_flat_size(&output_sections[section_idx]);
            let mut flat_data =
                build_flat_section_data(&output_sections[section_idx], flat_capacity);
            let section_size = flat_data.len() as u64;

            // Apply each relocation to the flat buffer.
            for &ri in reloc_indices {
                let pending = &self.pending_relocations[ri];

                // Resolve the symbol name and value through the multi-strategy
                // resolution pipeline.
                let (sym_name, sym_value) = match self.resolve_symbol(pending, symbols) {
                    Ok(pair) => pair,
                    Err(e) => {
                        self.errors.push(e);
                        continue;
                    }
                };

                let entry = RelocationEntry {
                    offset: pending.output_offset,
                    reloc_type: pending.input_reloc.reloc_type,
                    symbol_name: sym_name,
                    symbol_value: sym_value,
                    addend: pending.input_reloc.addend,
                    output_section: pending.output_section_index,
                };

                // Validate offset against section bounds. The relocation field
                // size comes from the handler; fall back to the target pointer
                // width when the handler returns 0 (unknown type).
                let field_size = handler.relocation_size(entry.reloc_type);
                let effective_size = if field_size > 0 {
                    field_size as u64
                } else {
                    self.default_relocation_size() as u64
                };

                if entry
                    .offset
                    .checked_add(effective_size)
                    .map_or(true, |end| end > section_size)
                {
                    self.errors.push(RelocationError::InvalidOffset {
                        offset: entry.offset,
                        section_size,
                    });
                    continue;
                }

                // Dispatch to the architecture-specific handler.
                match handler.apply_relocation(&entry, &mut flat_data, got_address, plt_address) {
                    Ok(()) => {
                        self.applied_count += 1;
                    }
                    Err(e) => {
                        self.errors.push(e);
                    }
                }
            }

            // Write the patched flat data back to the individual input section
            // data buffers so that the ELF writer can emit them correctly.
            write_back_flat_data(&mut output_sections[section_idx], &flat_data);
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
    ///
    /// Uses [`FxHashSet`] for deduplication of symbol names across multiple
    /// relocations targeting the same symbol.
    pub fn classify_relocations(
        &self,
        handler: &dyn ArchRelocationHandler,
    ) -> RelocationClassification {
        let mut got_set: FxHashSet<String> = FxHashSet::default();
        let mut plt_set: FxHashSet<String> = FxHashSet::default();
        let mut copy_set: FxHashSet<String> = FxHashSet::default();

        // Cache resolved symbol names per (object, symbol_index) pair to avoid
        // redundant lookups across multiple relocations.
        let mut name_cache: FxHashMap<(usize, u32), String> = FxHashMap::default();

        for pending in &self.pending_relocations {
            let reloc_type = pending.input_reloc.reloc_type;
            let cache_key = (pending.object_index, pending.input_reloc.symbol_index);

            // Resolve symbol name, using the per-object registered names if
            // available, otherwise falling back to a synthetic identifier.
            let sym_name = if let Some(cached) = name_cache.get(&cache_key) {
                cached.clone()
            } else {
                let name = self.resolve_symbol_name_for_classification(pending);
                name_cache.insert(cache_key, name.clone());
                name
            };

            // Skip empty/null symbol names (section-relative relocations that
            // don't reference a named symbol).
            if sym_name.is_empty() {
                continue;
            }

            // Check whether this object has registered symbol names to
            // determine classification confidence level.
            let has_names = self.object_symbol_names.contains_key(&pending.object_index);

            // Classify based on the relocation type requirements.
            if handler.needs_got_entry(reloc_type) && !got_set.contains(&sym_name) {
                got_set.insert(sym_name.clone());
            }
            if handler.needs_plt_entry(reloc_type) && !plt_set.contains(&sym_name) {
                plt_set.insert(sym_name.clone());
            }

            // Copy relocations are needed for absolute data references to
            // external symbols in shared libraries. Only consider symbols from
            // objects with registered names (higher confidence in symbol identity).
            if has_names
                && !handler.is_pc_relative(reloc_type)
                && !handler.needs_got_entry(reloc_type)
                && !handler.needs_plt_entry(reloc_type)
                && !copy_set.contains(&sym_name)
            {
                copy_set.insert(sym_name);
            }
        }

        RelocationClassification {
            got_entries: got_set.into_iter().collect(),
            plt_entries: plt_set.into_iter().collect(),
            copy_relocs: copy_set.into_iter().collect(),
        }
    }

    /// Returns the number of successfully applied relocations from the last
    /// [`apply_relocations`](Self::apply_relocations) call.
    #[inline]
    pub fn applied_count(&self) -> usize {
        self.applied_count
    }

    /// Returns the total number of pending (collected) relocations.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_relocations.len()
    }

    /// Returns errors accumulated during the last apply pass.
    #[inline]
    pub fn errors(&self) -> &[RelocationError] {
        &self.errors
    }

    /// Returns the configured target architecture, if set.
    #[inline]
    pub fn target(&self) -> Option<Target> {
        self.target
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Resolves a pending relocation's symbol to a (name, value) pair.
    ///
    /// Resolution strategy (in order of priority):
    /// 1. Look up the symbol name from per-object registered tables, then
    ///    resolve by name through [`ResolvedSymbols::get_symbol`] and
    ///    [`ResolvedSymbols::get_symbol_value`].
    /// 2. For section-relative symbols (empty name), synthesise a section
    ///    identifier from [`InputRelocation::section_index`].
    /// 3. Fall back to direct index into [`ResolvedSymbols::symbols`].
    fn resolve_symbol(
        &self,
        pending: &PendingRelocation,
        symbols: &ResolvedSymbols,
    ) -> Result<(String, u64), RelocationError> {
        let sym_idx = pending.input_reloc.symbol_index as usize;

        // Strategy 1: Named resolution via registered per-object symbol tables.
        // This is the preferred path because input symbol indices do not
        // necessarily correspond to output symbol table indices after merging.
        if let Some(names) = self.object_symbol_names.get(&pending.object_index) {
            if sym_idx < names.len() {
                let name = &names[sym_idx];
                if !name.is_empty() {
                    // Attempt get_symbol for full entry access.
                    if let Some(entry) = symbols.get_symbol(name) {
                        return Ok((entry.name.clone(), entry.value));
                    }
                    // Attempt get_symbol_value for value-only lookup.
                    if let Some(value) = symbols.get_symbol_value(name) {
                        return Ok((name.clone(), value));
                    }
                    // Cross-verify via symbol_map direct access.
                    if let Some(&mapped_idx) = symbols.symbol_map.get(name) {
                        if mapped_idx < symbols.symbols.len() {
                            let sym = &symbols.symbols[mapped_idx];
                            return Ok((sym.name.clone(), sym.value));
                        }
                    }
                    return Err(RelocationError::UndefinedSymbol { name: name.clone() });
                }

                // Empty name — section-relative symbol. The section_index
                // field identifies which input section's base address to use.
                let section_ref = pending.input_reloc.section_index;
                let section_sym_name = format!("__section_{}", section_ref);
                if let Some(value) = symbols.get_symbol_value(&section_sym_name) {
                    return Ok((section_sym_name, value));
                }
                // Fall through to direct index resolution for section symbols
                // that may be represented differently in the resolved table.
            }
        }

        // Strategy 2: Direct index resolution (fallback).
        // Works when the caller has pre-aligned the resolved symbol table
        // indices with the input object's symbol indices, or for simple
        // single-object linking scenarios.
        if sym_idx < symbols.symbols.len() {
            let sym = &symbols.symbols[sym_idx];
            return Ok((sym.name.clone(), sym.value));
        }

        // All resolution paths exhausted.
        Err(RelocationError::UndefinedSymbol {
            name: self.format_symbol_id(pending),
        })
    }

    /// Resolves a symbol name for classification purposes (GOT/PLT analysis).
    ///
    /// Returns the symbol name string, falling back to a synthetic identifier
    /// when the name cannot be resolved from registered per-object tables.
    fn resolve_symbol_name_for_classification(&self, pending: &PendingRelocation) -> String {
        let sym_idx = pending.input_reloc.symbol_index as usize;

        if let Some(names) = self.object_symbol_names.get(&pending.object_index) {
            if sym_idx < names.len() && !names[sym_idx].is_empty() {
                return names[sym_idx].clone();
            }
        }

        // Synthetic identifier as fallback for classification.
        format!("__sym_obj{}_{}", pending.object_index, sym_idx)
    }

    /// Formats a symbol identifier for error messages, including target
    /// architecture context when available via [`Target::elf_machine`].
    fn format_symbol_id(&self, pending: &PendingRelocation) -> String {
        let base = format!(
            "symbol_index_{} (object {}, section {})",
            pending.input_reloc.symbol_index, pending.object_index, pending.input_section_index,
        );
        if let Some(target) = self.target {
            format!("{} [e_machine={}]", base, target.elf_machine())
        } else {
            base
        }
    }

    /// Returns the default relocation field size in bytes based on the
    /// target's pointer width via [`Target::pointer_width`].
    /// Falls back to 8 bytes (64-bit) when no target is configured.
    fn default_relocation_size(&self) -> u8 {
        self.target.map(|t| t.pointer_width() as u8).unwrap_or(8)
    }
}

impl Default for RelocationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Helper functions — flat section data construction
// ===========================================================================

/// Computes the total size of the flat data buffer for an output section,
/// accounting for alignment padding between merged input sections.
fn compute_flat_size(section: &OutputSection) -> usize {
    // Use the recorded section size if available (set by the section merger
    // during layout computation).
    if section.size > 0 {
        return section.size as usize;
    }
    // Otherwise, compute from individual input sections and their alignment.
    let mut size: u64 = 0;
    for merged in &section.input_sections {
        let pad = compute_padding(size, merged.input.alignment.max(1));
        size += pad;
        size += merged.input.data.len() as u64;
    }
    size as usize
}

/// Builds a flat data buffer from all input sections merged into an output
/// section. Alignment padding between input sections is filled with zeros.
///
/// The resulting buffer has the same layout as the final output section data
/// in the ELF file, allowing relocations to be applied at their
/// output-section-relative offsets.
fn build_flat_section_data(section: &OutputSection, capacity: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(capacity);
    for merged in &section.input_sections {
        // Pad to the correct position using the recorded output offset from
        // the section merger's layout computation.
        let target_offset = merged.offset_in_output as usize;
        if data.len() < target_offset {
            data.resize(target_offset, 0u8);
        }
        data.extend_from_slice(&merged.input.data);
    }
    data
}

/// Writes patched flat data back to the individual input section data buffers
/// within an output section.
///
/// After relocations are applied to the flat buffer, this function distributes
/// the patched bytes back to each input section's data vector so that the
/// ELF writer can emit them correctly.
fn write_back_flat_data(section: &mut OutputSection, flat_data: &[u8]) {
    for merged in &mut section.input_sections {
        let start = merged.offset_in_output as usize;
        let len = merged.input.data.len();
        let end = start + len;
        if end <= flat_data.len() {
            // Full copy — the common case.
            merged.input.data.copy_from_slice(&flat_data[start..end]);
        } else if start < flat_data.len() {
            // Partial overlap — copy what is available from the flat buffer.
            let available = flat_data.len() - start;
            merged.input.data[..available].copy_from_slice(&flat_data[start..flat_data.len()]);
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RelocationError Display formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_relocation_error_display_overflow() {
        let err = RelocationError::Overflow {
            reloc_type: 2,
            offset: 0x1000,
            value: 0x1_0000_0000,
            max_value: 0xFFFF_FFFF,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overflow"));
        assert!(msg.contains("0x1000"));
    }

    #[test]
    fn test_relocation_error_display_undefined() {
        let err = RelocationError::UndefinedSymbol {
            name: "missing_func".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("missing_func"));
        assert!(msg.contains("undefined"));
    }

    #[test]
    fn test_relocation_error_display_unsupported() {
        let err = RelocationError::UnsupportedType { reloc_type: 99 };
        let msg = format!("{}", err);
        assert!(msg.contains("99"));
        assert!(msg.contains("unsupported"));
    }

    #[test]
    fn test_relocation_error_display_invalid_offset() {
        let err = RelocationError::InvalidOffset {
            offset: 0x500,
            section_size: 0x100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x500"));
        assert!(msg.contains("0x100"));
    }

    // -----------------------------------------------------------------------
    // RelocationProcessor construction and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_processor_new() {
        let proc = RelocationProcessor::new();
        assert_eq!(proc.pending_count(), 0);
        assert_eq!(proc.applied_count(), 0);
        assert!(proc.errors().is_empty());
        assert!(proc.target().is_none());
    }

    #[test]
    fn test_processor_default() {
        let proc = RelocationProcessor::default();
        assert_eq!(proc.pending_count(), 0);
    }

    #[test]
    fn test_processor_set_target() {
        let mut proc = RelocationProcessor::new();
        proc.set_target(Target::X86_64);
        assert_eq!(proc.target(), Some(Target::X86_64));
    }

    #[test]
    fn test_default_relocation_size_64bit() {
        let mut proc = RelocationProcessor::new();
        proc.set_target(Target::X86_64);
        assert_eq!(proc.default_relocation_size(), 8);
    }

    #[test]
    fn test_default_relocation_size_32bit() {
        let mut proc = RelocationProcessor::new();
        proc.set_target(Target::I686);
        assert_eq!(proc.default_relocation_size(), 4);
    }

    #[test]
    fn test_default_relocation_size_no_target() {
        let proc = RelocationProcessor::new();
        assert_eq!(proc.default_relocation_size(), 8);
    }

    // -----------------------------------------------------------------------
    // Relocation collection
    // -----------------------------------------------------------------------

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
    fn test_collect_relocations_offset_translation() {
        let mut proc = RelocationProcessor::new();
        let relocs = vec![InputRelocation {
            offset: 0x10,
            reloc_type: 1,
            symbol_index: 0,
            addend: 0,
            section_index: 1,
        }];

        proc.collect_relocations(0, 1, &relocs, 0, 0x200);
        assert_eq!(proc.pending_count(), 1);
        // Output offset = 0x10 (input) + 0x200 (section placement) = 0x210
        assert_eq!(proc.pending_relocations[0].output_offset, 0x210);
    }

    #[test]
    fn test_collect_multiple_objects() {
        let mut proc = RelocationProcessor::new();
        let relocs1 = vec![InputRelocation {
            offset: 0x0,
            reloc_type: 1,
            symbol_index: 0,
            addend: 0,
            section_index: 0,
        }];
        let relocs2 = vec![InputRelocation {
            offset: 0x8,
            reloc_type: 2,
            symbol_index: 1,
            addend: 4,
            section_index: 0,
        }];

        proc.collect_relocations(0, 0, &relocs1, 0, 0x0);
        proc.collect_relocations(1, 0, &relocs2, 0, 0x100);
        assert_eq!(proc.pending_count(), 2);
        assert_eq!(proc.pending_relocations[0].object_index, 0);
        assert_eq!(proc.pending_relocations[1].object_index, 1);
    }

    #[test]
    fn test_collect_preserves_section_index() {
        let mut proc = RelocationProcessor::new();
        let relocs = vec![InputRelocation {
            offset: 0x0,
            reloc_type: 1,
            symbol_index: 5,
            addend: 0,
            section_index: 7,
        }];
        proc.collect_relocations(0, 3, &relocs, 0, 0x0);
        // Verify that the input relocation's section_index is preserved
        assert_eq!(proc.pending_relocations[0].input_reloc.section_index, 7);
        assert_eq!(proc.pending_relocations[0].input_section_index, 3);
    }

    // -----------------------------------------------------------------------
    // Symbol registration
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_object_symbols() {
        let mut proc = RelocationProcessor::new();
        proc.register_object_symbols(
            0,
            vec!["".to_string(), "main".to_string(), "printf".to_string()],
        );
        proc.register_object_symbols(1, vec!["".to_string(), "helper".to_string()]);

        assert!(proc.object_symbol_names.contains_key(&0));
        assert!(proc.object_symbol_names.contains_key(&1));
        assert_eq!(proc.object_symbol_names.get(&0).unwrap().len(), 3);
    }

    // -----------------------------------------------------------------------
    // RelocationClassification default
    // -----------------------------------------------------------------------

    #[test]
    fn test_classification_default() {
        let class = RelocationClassification::default();
        assert!(class.got_entries.is_empty());
        assert!(class.plt_entries.is_empty());
        assert!(class.copy_relocs.is_empty());
    }

    // -----------------------------------------------------------------------
    // RelocationEntry fields
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Format symbol ID with/without target
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_symbol_id_with_target() {
        let mut proc = RelocationProcessor::new();
        proc.set_target(Target::X86_64);
        let pending = PendingRelocation {
            input_reloc: InputRelocation {
                offset: 0,
                reloc_type: 1,
                symbol_index: 42,
                addend: 0,
                section_index: 0,
            },
            object_index: 3,
            input_section_index: 1,
            output_section_index: 0,
            output_offset: 0,
        };
        let id = proc.format_symbol_id(&pending);
        assert!(id.contains("42"));
        assert!(id.contains("object 3"));
        assert!(id.contains("section 1"));
        // Target::X86_64.elf_machine() == 62 (EM_X86_64)
        assert!(id.contains("e_machine="));
    }

    #[test]
    fn test_format_symbol_id_without_target() {
        let proc = RelocationProcessor::new();
        let pending = PendingRelocation {
            input_reloc: InputRelocation {
                offset: 0,
                reloc_type: 1,
                symbol_index: 7,
                addend: 0,
                section_index: 0,
            },
            object_index: 1,
            input_section_index: 0,
            output_section_index: 0,
            output_offset: 0,
        };
        let id = proc.format_symbol_id(&pending);
        assert!(id.contains("7"));
        assert!(id.contains("object 1"));
        assert!(id.contains("section 0"));
        assert!(!id.contains("e_machine"));
    }

    // -----------------------------------------------------------------------
    // Helper: build_flat_section_data
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_flat_section_data_empty_section() {
        let section = OutputSection {
            name: ".text".to_string(),
            section_type: 1,
            flags: 0,
            alignment: 1,
            addr: 0,
            offset: 0,
            size: 0,
            input_sections: Vec::new(),
            entry_size: 0,
        };
        let data = build_flat_section_data(&section, 0);
        assert!(data.is_empty());
    }

    #[test]
    fn test_build_flat_section_data_single_input() {
        use super::super::section_merger::{InputSection, MergedInput};

        let section = OutputSection {
            name: ".text".to_string(),
            section_type: 1,
            flags: 0,
            alignment: 1,
            addr: 0,
            offset: 0,
            size: 4,
            input_sections: vec![MergedInput {
                input: InputSection {
                    name: ".text".to_string(),
                    section_type: 1,
                    flags: 0,
                    data: vec![0xCC, 0xCC, 0xCC, 0xCC],
                    alignment: 1,
                    entry_size: 0,
                    group_id: None,
                    object_index: 0,
                    original_index: 0,
                    relocations: Vec::new(),
                },
                offset_in_output: 0,
            }],
            entry_size: 0,
        };
        let data = build_flat_section_data(&section, 4);
        assert_eq!(data, vec![0xCC, 0xCC, 0xCC, 0xCC]);
    }

    #[test]
    fn test_build_and_write_back_flat_data() {
        use super::super::section_merger::{InputSection, MergedInput};

        let mut section = OutputSection {
            name: ".text".to_string(),
            section_type: 1,
            flags: 0,
            alignment: 4,
            addr: 0x1000,
            offset: 0,
            size: 12,
            input_sections: vec![
                MergedInput {
                    input: InputSection {
                        name: ".text".to_string(),
                        section_type: 1,
                        flags: 0,
                        data: vec![0xAA, 0xBB, 0xCC, 0xDD],
                        alignment: 4,
                        entry_size: 0,
                        group_id: None,
                        object_index: 0,
                        original_index: 0,
                        relocations: Vec::new(),
                    },
                    offset_in_output: 0,
                },
                MergedInput {
                    input: InputSection {
                        name: ".text".to_string(),
                        section_type: 1,
                        flags: 0,
                        data: vec![0x11, 0x22, 0x33, 0x44],
                        alignment: 4,
                        entry_size: 0,
                        group_id: None,
                        object_index: 1,
                        original_index: 0,
                        relocations: Vec::new(),
                    },
                    offset_in_output: 8,
                },
            ],
            entry_size: 0,
        };

        let mut flat_data = build_flat_section_data(&section, 12);
        // Verify flat layout: [AA BB CC DD 00 00 00 00 11 22 33 44]
        assert_eq!(flat_data.len(), 12);
        assert_eq!(flat_data[0], 0xAA);
        assert_eq!(flat_data[4], 0x00); // padding
        assert_eq!(flat_data[8], 0x11);

        // Simulate a relocation patching byte at offset 8
        flat_data[8] = 0xFF;
        flat_data[9] = 0xEE;

        // Write back and verify the second input section was updated
        write_back_flat_data(&mut section, &flat_data);
        assert_eq!(
            section.input_sections[0].input.data,
            vec![0xAA, 0xBB, 0xCC, 0xDD]
        );
        assert_eq!(section.input_sections[1].input.data[0], 0xFF);
        assert_eq!(section.input_sections[1].input.data[1], 0xEE);
    }

    // -----------------------------------------------------------------------
    // compute_flat_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_flat_size_uses_section_size() {
        let section = OutputSection {
            name: ".text".to_string(),
            section_type: 1,
            flags: 0,
            alignment: 1,
            addr: 0,
            offset: 0,
            size: 256,
            input_sections: Vec::new(),
            entry_size: 0,
        };
        assert_eq!(compute_flat_size(&section), 256);
    }
}
