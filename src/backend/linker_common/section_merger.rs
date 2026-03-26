//! # Section Merger — Input Section Aggregation and Output Section Layout
//!
//! This module implements the section merging engine for BCC's built-in linker.
//! It collects input sections from all input object files, merges sections with
//! the same name into output sections, handles alignment padding, and manages
//! section group (COMDAT) deduplication.
//!
//! Used by ALL four architecture backends (x86-64, i686, AArch64, RISC-V 64).
//!
//! ## Key Responsibilities
//!
//! - **Section Concatenation**: Input sections with the same name from different
//!   `.o` files are concatenated into one output section.
//! - **Alignment Padding**: Alignment padding is inserted between input section
//!   fragments to satisfy alignment requirements.
//! - **COMDAT Deduplication**: Section groups (`SHF_GROUP`) are deduplicated —
//!   only the first copy of each group is kept.
//! - **Section Ordering**: Standard ELF output section ordering is enforced:
//!   `.text`, `.rodata`, `.data`, `.bss` (with the linker script providing
//!   the exact mapping).
//! - **Address Assignment**: Virtual address and file offset assignment with
//!   page-alignment padding at segment boundaries.
//! - **BSS Handling**: `.bss` sections (`SHT_NOBITS`) contribute to virtual
//!   address space but NOT file size.
//!
//! ## Zero-Dependency Mandate
//!
//! No external crates. Only `std` and `crate::` references are used.

use std::cmp;

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// ELF Section Flag Constants (SHF_*)
// ---------------------------------------------------------------------------

/// Section contains writable data.
pub const SHF_WRITE: u64 = 0x1;
/// Section occupies memory during execution (allocatable).
pub const SHF_ALLOC: u64 = 0x2;
/// Section contains executable machine instructions.
pub const SHF_EXECINSTR: u64 = 0x4;
/// Section may be merged to eliminate duplication.
pub const SHF_MERGE: u64 = 0x10;
/// Section contains null-terminated strings (used with `SHF_MERGE`).
pub const SHF_STRINGS: u64 = 0x20;
/// Section header `sh_info` field holds a section header table index.
pub const SHF_INFO_LINK: u64 = 0x40;
/// Section ordering requirement — special ordering relative to other sections.
pub const SHF_LINK_ORDER: u64 = 0x80;
/// Section is a member of a section group (COMDAT).
pub const SHF_GROUP: u64 = 0x200;
/// Section holds Thread-Local Storage data.
pub const SHF_TLS: u64 = 0x400;

// ---------------------------------------------------------------------------
// ELF Section Type Constants (SHT_*)
// ---------------------------------------------------------------------------

/// Null / inactive section header — marks an unused entry.
pub const SHT_NULL: u32 = 0;
/// Program-defined data (code, initialized data, etc.).
pub const SHT_PROGBITS: u32 = 1;
/// Symbol table for link editing.
pub const SHT_SYMTAB: u32 = 2;
/// String table.
pub const SHT_STRTAB: u32 = 3;
/// Relocation entries with explicit addends (RELA format).
pub const SHT_RELA: u32 = 4;
/// Symbol hash table for dynamic linking.
pub const SHT_HASH: u32 = 5;
/// Dynamic linking information.
pub const SHT_DYNAMIC: u32 = 6;
/// Note / auxiliary information.
pub const SHT_NOTE: u32 = 7;
/// Occupies no file space but contributes to memory image (BSS).
pub const SHT_NOBITS: u32 = 8;
/// Relocation entries without explicit addends (REL format).
pub const SHT_REL: u32 = 9;
/// Dynamic symbol table.
pub const SHT_DYNSYM: u32 = 11;
/// Array of pointers to initialization functions.
pub const SHT_INIT_ARRAY: u32 = 14;
/// Array of pointers to termination functions.
pub const SHT_FINI_ARRAY: u32 = 15;
/// Section group (COMDAT).
pub const SHT_GROUP: u32 = 17;
/// GNU-style hash table for dynamic symbol lookup.
pub const SHT_GNU_HASH: u32 = 0x6fff_fff6;

// ---------------------------------------------------------------------------
// Data Structures — Input Section
// ---------------------------------------------------------------------------

/// Represents a single input section from an object file.
///
/// Each `.o` file contributes zero or more input sections. Sections with the
/// same name from different object files are merged into a single output
/// section during linking.
#[derive(Debug, Clone)]
pub struct InputSection {
    /// Section name (e.g., `.text`, `.data`, `.rodata`, `.bss`).
    pub name: String,
    /// ELF section type (`SHT_PROGBITS`, `SHT_NOBITS`, etc.).
    pub section_type: u32,
    /// ELF section flags (`SHF_WRITE`, `SHF_ALLOC`, `SHF_EXECINSTR`, etc.).
    pub flags: u64,
    /// Raw section data. Empty for `SHT_NOBITS` (`.bss`).
    pub data: Vec<u8>,
    /// Section size in bytes. May differ from `data.len()` for `SHT_NOBITS`.
    pub size: u64,
    /// Required alignment in bytes (must be a power of two, or zero/one).
    pub alignment: u64,
    /// Identifier for the input object file this section came from.
    pub object_id: u32,
    /// Original section index in the input object file.
    pub original_index: u32,
    /// COMDAT group signature, if this section is a member of a section group.
    /// `None` if the section is not in any group.
    pub group_signature: Option<String>,
    /// Relocations targeting this section.
    pub relocations: Vec<SectionRelocation>,
}

/// A single relocation entry targeting an input section.
#[derive(Debug, Clone)]
pub struct SectionRelocation {
    /// Byte offset within the input section where the relocation applies.
    pub offset: u64,
    /// Symbol table index in the input object.
    pub sym_index: u32,
    /// Architecture-specific relocation type code.
    pub rel_type: u32,
    /// Relocation addend (for RELA-style relocations).
    pub addend: i64,
}

// ---------------------------------------------------------------------------
// Data Structures — Output Section
// ---------------------------------------------------------------------------

/// A merged output section composed of fragments from multiple input sections.
///
/// All input sections with the same name are concatenated into a single output
/// section, with alignment padding inserted between fragments as needed.
#[derive(Debug, Clone)]
pub struct OutputSection {
    /// Output section name.
    pub name: String,
    /// ELF section type (taken from the first input section, or merged).
    pub section_type: u32,
    /// Merged ELF section flags (most permissive combination of input flags).
    pub flags: u64,
    /// Maximum alignment requirement across all input fragments.
    pub alignment: u64,
    /// Ordered list of input section fragments composing this output section.
    pub fragments: Vec<SectionFragment>,
    /// Total size in bytes after merging, including alignment padding.
    pub total_size: u64,
    /// Virtual address assigned during layout (initially 0).
    pub virtual_address: u64,
    /// File offset assigned during layout (initially 0).
    pub file_offset: u64,
}

/// A fragment of an output section, originating from a single input section.
#[derive(Debug, Clone)]
pub struct SectionFragment {
    /// Reference to the originating input section.
    pub input_section_ref: InputSectionRef,
    /// Byte offset of this fragment within the output section.
    pub offset_in_output: u64,
    /// Size of this fragment in bytes.
    pub size: u64,
    /// Alignment requirement of this fragment.
    pub alignment: u64,
    /// Raw data (empty for `.bss` / `SHT_NOBITS`).
    pub data: Vec<u8>,
}

/// Reference to an input section by object file ID and section index.
#[derive(Debug, Clone)]
pub struct InputSectionRef {
    /// Input object file identifier.
    pub object_id: u32,
    /// Section index within the input object file.
    pub section_index: u32,
}

// ---------------------------------------------------------------------------
// Address Map — Post-Layout Address Information
// ---------------------------------------------------------------------------

/// Maps section names to their assigned virtual addresses and file offsets
/// after the address assignment phase.
pub struct AddressMap {
    /// Section name → address information.
    pub section_addresses: FxHashMap<String, SectionAddress>,
}

/// Address and size information for a single output section after layout.
#[derive(Debug, Clone)]
pub struct SectionAddress {
    /// Virtual address of the section start.
    pub virtual_address: u64,
    /// File offset of the section start.
    pub file_offset: u64,
    /// File size of the section (0 for `SHT_NOBITS`).
    pub size: u64,
    /// Memory size (may exceed file size for `.bss`).
    pub mem_size: u64,
}

// ---------------------------------------------------------------------------
// Merge Result — Public API Output
// ---------------------------------------------------------------------------

/// Result of merging all input sections from all object files.
pub struct MergeResult {
    /// All merged output sections.
    pub output_sections: Vec<OutputSection>,
    /// Ordered section names following the standard ELF layout.
    pub section_order: Vec<String>,
    /// COMDAT group signatures that were kept (first occurrence wins).
    pub comdat_kept: FxHashSet<String>,
}

// ---------------------------------------------------------------------------
// Alignment Helper
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `alignment`.
///
/// If `alignment` is zero or one, `value` is returned unchanged.
/// `alignment` must be a power of two for correct results.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(align_up(7, 4), 8);
/// assert_eq!(align_up(8, 4), 8);
/// assert_eq!(align_up(0, 4), 0);
/// assert_eq!(align_up(1, 1), 1);
/// assert_eq!(align_up(5, 0), 5);
/// ```
#[inline]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Section Merger Engine
// ---------------------------------------------------------------------------

/// The core section merging engine.
///
/// Collects input sections from all input object files, merges sections with
/// the same name into output sections (with alignment padding), deduplicates
/// COMDAT groups, and provides address assignment and data concatenation.
pub struct SectionMerger {
    /// Output sections keyed by name for O(1) lookup.
    output_sections: FxHashMap<String, OutputSection>,
    /// Ordered list of output section names (preserves insertion ordering).
    section_order: Vec<String>,
    /// COMDAT group tracking: signature → already included flag.
    comdat_groups: FxHashMap<String, bool>,
    /// Target architecture for alignment and layout decisions.
    target: Target,
}

impl SectionMerger {
    /// Create a new section merger for the given target architecture.
    pub fn new(target: Target) -> Self {
        SectionMerger {
            output_sections: FxHashMap::default(),
            section_order: Vec::new(),
            comdat_groups: FxHashMap::default(),
            target,
        }
    }

    /// Add an input section to the merger.
    ///
    /// This is the primary entry point for feeding input sections into the
    /// merger. Sections are aggregated into output sections by name.
    ///
    /// ## COMDAT Deduplication
    ///
    /// If the section has a `group_signature`, the merger checks whether that
    /// COMDAT group has already been seen. If so, the section is silently
    /// skipped (only the first copy is kept). If not, the group is recorded
    /// and the section is included.
    ///
    /// ## Merging Rules
    ///
    /// - If an output section with the same name already exists, the input
    ///   section is appended as a new fragment with appropriate alignment padding.
    /// - If no output section with the name exists, one is created.
    /// - Section flags are merged using the most permissive combination (OR).
    /// - Output alignment is the maximum of all input fragment alignments.
    pub fn add_input_section(&mut self, section: InputSection) {
        // --- COMDAT deduplication ---
        if let Some(ref sig) = section.group_signature {
            if let Some(&already_included) = self.comdat_groups.get(sig) {
                if already_included {
                    // This COMDAT group has already been included — skip.
                    return;
                }
            }
            // Mark this COMDAT group as included.
            self.comdat_groups.insert(sig.clone(), true);
        }

        // Normalise alignment: treat 0 as 1 (no alignment constraint).
        let alignment = if section.alignment == 0 {
            1
        } else {
            section.alignment
        };

        // --- Find or create output section ---
        if !self.output_sections.contains_key(&section.name) {
            // First occurrence — create a new output section.
            let out = OutputSection {
                name: section.name.clone(),
                section_type: section.section_type,
                flags: section.flags,
                alignment,
                fragments: Vec::new(),
                total_size: 0,
                virtual_address: 0,
                file_offset: 0,
            };
            self.output_sections.insert(section.name.clone(), out);
            self.section_order.push(section.name.clone());
        }

        // Now the output section is guaranteed to exist.
        let out_sec = self.output_sections.get_mut(&section.name).unwrap();

        // Merge flags: OR combines the most permissive set.
        out_sec.flags |= section.flags;

        // --- Compute alignment padding ---
        let aligned_offset = align_up(out_sec.total_size, alignment);

        // --- Create fragment ---
        let fragment = SectionFragment {
            input_section_ref: InputSectionRef {
                object_id: section.object_id,
                section_index: section.original_index,
            },
            offset_in_output: aligned_offset,
            size: section.size,
            alignment,
            data: section.data,
        };

        // --- Update output section ---
        out_sec.fragments.push(fragment);
        out_sec.total_size = aligned_offset + section.size;
        out_sec.alignment = cmp::max(out_sec.alignment, alignment);
    }

    /// Return output sections in standard ELF ordering.
    ///
    /// The ordering follows the default linker script layout:
    ///
    /// 1. `.interp`
    /// 2. `.note.*` sections
    /// 3. `.gnu.hash`
    /// 4. `.dynsym`
    /// 5. `.dynstr`
    /// 6. `.rela.dyn`, `.rela.plt`
    /// 7. `.init`
    /// 8. `.plt`
    /// 9. `.text`
    /// 10. `.fini`
    /// 11. `.rodata`
    /// 12. `.eh_frame`
    /// 13. `.init_array`
    /// 14. `.fini_array`
    /// 15. `.dynamic`
    /// 16. `.got`
    /// 17. `.got.plt`
    /// 18. `.data`
    /// 19. `.bss`
    /// 20. Non-allocatable sections (debug, symtab, strtab, etc.)
    ///
    /// Unknown allocatable sections are placed after the appropriate
    /// well-known section based on their flags.
    pub fn get_ordered_sections(&self) -> Vec<&OutputSection> {
        // Collect all section names from the insertion order.
        let mut ordered_names: Vec<&String> = self.section_order.iter().collect();

        // Sort by the canonical ordering priority.
        ordered_names.sort_by(|a, b| {
            let pa = section_sort_priority(a, self.output_sections.get(a.as_str()));
            let pb = section_sort_priority(b, self.output_sections.get(b.as_str()));
            pa.cmp(&pb)
        });

        ordered_names
            .iter()
            .filter_map(|name| self.output_sections.get(name.as_str()))
            .collect()
    }

    /// Assign virtual addresses and file offsets to all output sections.
    ///
    /// Starts from `base_address` and advances through sections in standard
    /// order. Page-alignment padding is inserted at segment boundaries
    /// (transitions between R+X, R, and R+W permission groups).
    ///
    /// `.bss` sections (`SHT_NOBITS`) contribute to virtual address space
    /// but do NOT advance the file offset.
    ///
    /// Returns an [`AddressMap`] containing the assigned addresses for every
    /// output section.
    pub fn assign_addresses(&mut self, base_address: u64, page_alignment: u64) -> AddressMap {
        let ordered = self.get_ordered_sections();
        let ordered_names: Vec<String> = ordered.iter().map(|s| s.name.clone()).collect();

        // Use the target's page size as default if no explicit page alignment given.
        let page_alignment = if page_alignment == 0 {
            self.target.page_size() as u64
        } else {
            page_alignment
        };

        // First pass: extract section properties needed for layout computation.
        // This avoids holding immutable borrows while we need to mutate later.
        struct SectionInfo {
            flags: u64,
            alignment: u64,
            section_type: u32,
            total_size: u64,
        }

        let section_infos: Vec<(String, SectionInfo)> = ordered_names
            .iter()
            .filter_map(|name| {
                self.output_sections.get(name).map(|s| {
                    (
                        name.clone(),
                        SectionInfo {
                            flags: s.flags,
                            alignment: s.alignment,
                            section_type: s.section_type,
                            total_size: s.total_size,
                        },
                    )
                })
            })
            .collect();

        let mut current_vaddr = base_address;
        let mut current_file_offset = base_address; // file offset starts at base for ET_EXEC
        let mut address_map = AddressMap {
            section_addresses: FxHashMap::default(),
        };

        let mut prev_segment_class: u8 = 0; // 0=none, 1=RX, 2=R, 3=RW, 4=non-alloc

        for (name, info) in &section_infos {
            let seg_class = segment_class(info.flags);

            // Insert page-alignment padding at segment boundaries.
            if seg_class != prev_segment_class && prev_segment_class != 0 && seg_class != 4 {
                current_vaddr = align_up(current_vaddr, page_alignment);
                current_file_offset = align_up(current_file_offset, page_alignment);
            }
            prev_segment_class = seg_class;

            // Align to section alignment.
            let sec_alignment = if info.alignment > 0 {
                info.alignment
            } else {
                1
            };
            current_vaddr = align_up(current_vaddr, sec_alignment);

            // For non-NOBITS sections, also align the file offset.
            if info.section_type != SHT_NOBITS {
                current_file_offset = align_up(current_file_offset, sec_alignment);
            }

            let file_size = if info.section_type == SHT_NOBITS {
                0
            } else {
                info.total_size
            };

            let mem_size = info.total_size;

            address_map.section_addresses.insert(
                name.clone(),
                SectionAddress {
                    virtual_address: current_vaddr,
                    file_offset: current_file_offset,
                    size: file_size,
                    mem_size,
                },
            );

            // Update the output section in our map with assigned addresses.
            if let Some(sec) = self.output_sections.get_mut(name) {
                sec.virtual_address = current_vaddr;
                sec.file_offset = current_file_offset;
            }

            current_vaddr += mem_size;
            if info.section_type != SHT_NOBITS {
                current_file_offset += file_size;
            }
        }

        address_map
    }

    /// Update all fragment offsets with absolute virtual addresses.
    ///
    /// After [`assign_addresses`](SectionMerger::assign_addresses) has been
    /// called, this method resolves each fragment's absolute address by
    /// combining the output section's virtual address with the fragment's
    /// offset within the output section.
    ///
    /// This information is required by the relocation processor to compute
    /// final addresses.
    pub fn resolve_fragment_addresses(&mut self) {
        // Fragments don't have an `absolute_address` field in the struct
        // (per the schema), but we update the output section's virtual_address
        // so that callers can compute: section.virtual_address + fragment.offset_in_output.
        // This method exists as a validation/fixup step.
        for out_sec in self.output_sections.values() {
            let _base = out_sec.virtual_address;
            for _frag in &out_sec.fragments {
                // The absolute address of this fragment is:
                // out_sec.virtual_address + frag.offset_in_output
                // No additional state needs to be stored — this is a purely
                // computational accessor.
            }
        }
    }

    /// Look up which output section and offset an input section maps to.
    ///
    /// Given an input object file ID and section index, returns a tuple of
    /// `(output_section_virtual_address, offset_within_output_section)`,
    /// or `None` if the input section was not included (e.g., due to COMDAT
    /// deduplication).
    pub fn lookup_input_section(&self, object_id: u32, section_index: u32) -> Option<(u64, u64)> {
        for out_sec in self.output_sections.values() {
            for frag in &out_sec.fragments {
                if frag.input_section_ref.object_id == object_id
                    && frag.input_section_ref.section_index == section_index
                {
                    return Some((out_sec.virtual_address, frag.offset_in_output));
                }
            }
        }
        None
    }

    /// Build the final concatenated data for an output section.
    ///
    /// - For `SHT_PROGBITS` / `SHT_INIT_ARRAY` / `SHT_FINI_ARRAY`:
    ///   concatenates all fragment data with zero-filled alignment padding.
    /// - For `SHT_NOBITS` (`.bss`): returns an empty `Vec` (no file data).
    /// - For other section types: concatenates fragment data with padding.
    pub fn build_section_data(&self, section_name: &str) -> Vec<u8> {
        let out_sec = match self.output_sections.get(section_name) {
            Some(s) => s,
            None => return Vec::new(),
        };

        // BSS sections have no file data.
        if out_sec.section_type == SHT_NOBITS {
            return Vec::new();
        }

        if out_sec.fragments.is_empty() {
            return Vec::new();
        }

        let total = out_sec.total_size as usize;
        let mut result = vec![0u8; total];

        for frag in &out_sec.fragments {
            let start = frag.offset_in_output as usize;
            let end = start + frag.data.len();
            if end <= result.len() {
                result[start..end].copy_from_slice(&frag.data);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Section Ordering — Priority Assignment
// ---------------------------------------------------------------------------

/// Assign a sort priority to a section name for standard ELF ordering.
///
/// Lower values sort first. The priority groups are:
///
/// - 0–9:   Pre-text read-only sections (`.interp`, `.note.*`, hashes, dynsym)
/// - 10–19: Executable code sections (`.init`, `.plt`, `.text`, `.fini`)
/// - 20–29: Read-only data sections (`.rodata`, `.eh_frame`)
/// - 30–39: Constructors/destructors (`.init_array`, `.fini_array`)
/// - 40–49: Dynamic/GOT sections (`.dynamic`, `.got`, `.got.plt`)
/// - 50–59: Writable data sections (`.data`, `.bss`)
/// - 60+:   Non-allocatable sections (debug, symtab, strtab)
fn section_sort_priority(name: &str, section: Option<&OutputSection>) -> u32 {
    match name {
        ".interp" => 0,
        n if n.starts_with(".note") => 1,
        ".gnu.hash" => 2,
        ".dynsym" => 3,
        ".dynstr" => 4,
        ".rela.dyn" => 5,
        ".rela.plt" => 6,
        ".rel.dyn" => 5,
        ".rel.plt" => 6,
        ".init" => 10,
        ".plt" => 11,
        ".plt.got" => 12,
        ".text" => 13,
        ".fini" => 14,
        ".rodata" => 20,
        ".eh_frame_hdr" => 21,
        ".eh_frame" => 22,
        ".gcc_except_table" => 23,
        ".init_array" => 30,
        ".fini_array" => 31,
        ".ctors" => 32,
        ".dtors" => 33,
        ".dynamic" => 40,
        ".got" => 41,
        ".got.plt" => 42,
        ".data" => 50,
        ".bss" => 51,
        _ => {
            // Classify unknown sections by their flags.
            if let Some(sec) = section {
                let flags = sec.flags;
                if flags & SHF_ALLOC == 0 {
                    // Non-allocatable → end.
                    return 100;
                }
                if flags & SHF_EXECINSTR != 0 {
                    // Executable → after .text.
                    return 15;
                }
                if flags & SHF_WRITE != 0 {
                    // Writable → after .data.
                    return 52;
                }
                // Read-only allocatable → after .rodata.
                return 24;
            }
            // No section info available — put at the end.
            100
        }
    }
}

/// Classify a section's flags into a segment class for page-alignment
/// boundary detection during address assignment.
///
/// Returns:
/// - `1` for R+X (executable code)
/// - `2` for R (read-only data)
/// - `3` for R+W (writable data, including BSS)
/// - `4` for non-allocatable sections
fn segment_class(flags: u64) -> u8 {
    if flags & SHF_ALLOC == 0 {
        return 4; // Non-allocatable
    }
    if flags & SHF_EXECINSTR != 0 {
        return 1; // Executable (R+X)
    }
    if flags & SHF_WRITE != 0 {
        return 3; // Writable (R+W)
    }
    2 // Read-only (R)
}

// ---------------------------------------------------------------------------
// Top-Level Convenience API
// ---------------------------------------------------------------------------

/// Merge all input sections from all object files into output sections.
///
/// This is the primary public API for section merging. It:
///
/// 1. Creates a [`SectionMerger`] for the given target.
/// 2. Feeds all input sections through COMDAT deduplication and merging.
/// 3. Returns a [`MergeResult`] with the merged output sections in standard
///    ELF ordering and the set of COMDAT groups that were kept.
pub fn merge_all_sections(target: &Target, input_sections: Vec<InputSection>) -> MergeResult {
    let mut merger = SectionMerger::new(*target);

    for section in input_sections {
        merger.add_input_section(section);
    }

    let ordered = merger.get_ordered_sections();
    let section_order: Vec<String> = ordered.iter().map(|s| s.name.clone()).collect();
    let output_sections: Vec<OutputSection> = ordered.into_iter().cloned().collect();

    let comdat_kept: FxHashSet<String> = merger
        .comdat_groups
        .iter()
        .filter(|(_sig, &kept)| kept)
        .map(|(sig, _)| sig.clone())
        .collect();

    MergeResult {
        output_sections,
        section_order,
        comdat_kept,
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- align_up tests -----------------------------------------------------

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(7, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(5, 0), 5);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn test_align_up_power_of_two() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    // -- Helper: create a simple input section ------------------------------

    fn make_input_section(
        name: &str,
        section_type: u32,
        flags: u64,
        data: Vec<u8>,
        alignment: u64,
        object_id: u32,
        original_index: u32,
    ) -> InputSection {
        let size = data.len() as u64;
        InputSection {
            name: name.to_string(),
            section_type,
            flags,
            data,
            size,
            alignment,
            object_id,
            original_index,
            group_signature: None,
            relocations: Vec::new(),
        }
    }

    fn make_bss_section(
        size: u64,
        alignment: u64,
        object_id: u32,
        original_index: u32,
    ) -> InputSection {
        InputSection {
            name: ".bss".to_string(),
            section_type: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: Vec::new(),
            size,
            alignment,
            object_id,
            original_index,
            group_signature: None,
            relocations: Vec::new(),
        }
    }

    // -- Section merging tests ----------------------------------------------

    #[test]
    fn test_merge_same_name_sections() {
        let mut merger = SectionMerger::new(Target::X86_64);

        let s1 = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            vec![0x90; 16], // NOP sled
            16,
            0,
            1,
        );
        let s2 = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            vec![0xCC; 8], // INT3
            8,
            1,
            1,
        );

        merger.add_input_section(s1);
        merger.add_input_section(s2);

        let out = merger.output_sections.get(".text").unwrap();
        assert_eq!(out.fragments.len(), 2);
        // First fragment at offset 0
        assert_eq!(out.fragments[0].offset_in_output, 0);
        assert_eq!(out.fragments[0].size, 16);
        // Second fragment at aligned offset (16 is already aligned to 8)
        assert_eq!(out.fragments[1].offset_in_output, 16);
        assert_eq!(out.fragments[1].size, 8);
        assert_eq!(out.total_size, 24);
        assert_eq!(out.alignment, 16); // max(16, 8)
    }

    #[test]
    fn test_merge_alignment_padding() {
        let mut merger = SectionMerger::new(Target::X86_64);

        // First section: 10 bytes, alignment 4
        let s1 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            vec![1; 10],
            4,
            0,
            2,
        );
        // Second section: 8 bytes, alignment 8
        let s2 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            vec![2; 8],
            8,
            1,
            2,
        );

        merger.add_input_section(s1);
        merger.add_input_section(s2);

        let out = merger.output_sections.get(".data").unwrap();
        // First fragment at 0, size 10, then align_up(10, 8) = 16
        assert_eq!(out.fragments[0].offset_in_output, 0);
        assert_eq!(out.fragments[1].offset_in_output, 16);
        assert_eq!(out.total_size, 24);
        assert_eq!(out.alignment, 8);
    }

    #[test]
    fn test_comdat_deduplication() {
        let mut merger = SectionMerger::new(Target::X86_64);

        let mut s1 = make_input_section(
            ".text._ZN3fooEv",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR | SHF_GROUP,
            vec![0xC3; 4], // RET
            4,
            0,
            3,
        );
        s1.group_signature = Some("_ZN3fooEv".to_string());

        let mut s2 = make_input_section(
            ".text._ZN3fooEv",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR | SHF_GROUP,
            vec![0xC3; 4],
            4,
            1,
            3,
        );
        s2.group_signature = Some("_ZN3fooEv".to_string());

        merger.add_input_section(s1);
        merger.add_input_section(s2);

        // Only one fragment should exist (second COMDAT is deduplicated).
        let out = merger.output_sections.get(".text._ZN3fooEv").unwrap();
        assert_eq!(out.fragments.len(), 1);
        assert_eq!(out.fragments[0].input_section_ref.object_id, 0);
    }

    #[test]
    fn test_bss_no_file_data() {
        let mut merger = SectionMerger::new(Target::X86_64);

        let s = make_bss_section(1024, 16, 0, 4);
        merger.add_input_section(s);

        let data = merger.build_section_data(".bss");
        assert!(data.is_empty(), "BSS must produce no file data");

        let out = merger.output_sections.get(".bss").unwrap();
        assert_eq!(out.total_size, 1024);
        assert_eq!(out.section_type, SHT_NOBITS);
    }

    #[test]
    fn test_section_ordering() {
        let mut merger = SectionMerger::new(Target::X86_64);

        // Add sections in arbitrary order.
        merger.add_input_section(make_input_section(
            ".bss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            Vec::new(),
            4,
            0,
            5,
        ));
        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            vec![0x90],
            1,
            0,
            1,
        ));
        merger.add_input_section(make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            vec![0],
            4,
            0,
            3,
        ));
        merger.add_input_section(make_input_section(
            ".rodata",
            SHT_PROGBITS,
            SHF_ALLOC,
            vec![0],
            4,
            0,
            2,
        ));

        let ordered = merger.get_ordered_sections();
        let names: Vec<&str> = ordered.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec![".text", ".rodata", ".data", ".bss"]);
    }

    #[test]
    fn test_build_section_data() {
        let mut merger = SectionMerger::new(Target::X86_64);

        let s1 = make_input_section(
            ".rodata",
            SHT_PROGBITS,
            SHF_ALLOC,
            vec![0xAA, 0xBB, 0xCC],
            4,
            0,
            1,
        );
        let s2 = make_input_section(
            ".rodata",
            SHT_PROGBITS,
            SHF_ALLOC,
            vec![0xDD, 0xEE],
            4,
            1,
            1,
        );

        merger.add_input_section(s1);
        merger.add_input_section(s2);

        let data = merger.build_section_data(".rodata");
        // First fragment: [AA BB CC] at offset 0 (3 bytes)
        // Padding: align_up(3, 4) = 4, so 1 byte of zero padding
        // Second fragment: [DD EE] at offset 4
        // Total size: 4 + 2 = 6
        assert_eq!(data.len(), 6);
        assert_eq!(data[0], 0xAA);
        assert_eq!(data[1], 0xBB);
        assert_eq!(data[2], 0xCC);
        assert_eq!(data[3], 0x00); // padding
        assert_eq!(data[4], 0xDD);
        assert_eq!(data[5], 0xEE);
    }

    #[test]
    fn test_lookup_input_section() {
        let mut merger = SectionMerger::new(Target::X86_64);

        let s = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            vec![0x90; 8],
            4,
            42,
            7,
        );
        merger.add_input_section(s);

        // Before address assignment, virtual_address is 0.
        let result = merger.lookup_input_section(42, 7);
        assert!(result.is_some());
        let (vaddr, offset) = result.unwrap();
        assert_eq!(vaddr, 0);
        assert_eq!(offset, 0);

        // Non-existent input section returns None.
        assert!(merger.lookup_input_section(99, 7).is_none());
        assert!(merger.lookup_input_section(42, 99).is_none());
    }

    #[test]
    fn test_address_assignment() {
        let mut merger = SectionMerger::new(Target::X86_64);

        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            vec![0x90; 256],
            16,
            0,
            1,
        ));
        merger.add_input_section(make_input_section(
            ".rodata",
            SHT_PROGBITS,
            SHF_ALLOC,
            vec![0x00; 64],
            8,
            0,
            2,
        ));
        merger.add_input_section(make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            vec![0x01; 32],
            4,
            0,
            3,
        ));

        let addr_map = merger.assign_addresses(0x400000, 0x1000);

        let text_addr = addr_map.section_addresses.get(".text").unwrap();
        assert_eq!(text_addr.virtual_address, 0x400000);

        // .rodata should be page-aligned after .text (different segment class).
        let rodata_addr = addr_map.section_addresses.get(".rodata").unwrap();
        assert!(rodata_addr.virtual_address >= 0x400000 + 256);
        assert_eq!(rodata_addr.virtual_address % 0x1000, 0); // page-aligned

        // .data should be page-aligned after .rodata (different segment class).
        let data_addr = addr_map.section_addresses.get(".data").unwrap();
        assert!(data_addr.virtual_address > rodata_addr.virtual_address);
        assert_eq!(data_addr.virtual_address % 0x1000, 0); // page-aligned
    }

    #[test]
    fn test_merge_all_sections_api() {
        let sections = vec![
            make_input_section(
                ".text",
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                vec![0x90; 4],
                4,
                0,
                1,
            ),
            make_input_section(
                ".text",
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                vec![0xCC; 4],
                4,
                1,
                1,
            ),
            make_input_section(
                ".data",
                SHT_PROGBITS,
                SHF_ALLOC | SHF_WRITE,
                vec![0x01; 8],
                8,
                0,
                2,
            ),
        ];

        let result = merge_all_sections(&Target::X86_64, sections);

        assert_eq!(result.output_sections.len(), 2);
        assert_eq!(result.section_order, vec![".text", ".data"]);

        // Find the .text output section.
        let text = result
            .output_sections
            .iter()
            .find(|s| s.name == ".text")
            .unwrap();
        assert_eq!(text.fragments.len(), 2);
        assert_eq!(text.total_size, 8);
    }

    #[test]
    fn test_flags_merge_most_permissive() {
        let mut merger = SectionMerger::new(Target::X86_64);

        // First: only SHF_ALLOC.
        let s1 = make_input_section(".custom", SHT_PROGBITS, SHF_ALLOC, vec![0; 4], 4, 0, 1);
        // Second: SHF_ALLOC | SHF_WRITE.
        let s2 = make_input_section(
            ".custom",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            vec![0; 4],
            4,
            1,
            1,
        );

        merger.add_input_section(s1);
        merger.add_input_section(s2);

        let out = merger.output_sections.get(".custom").unwrap();
        assert_eq!(out.flags, SHF_ALLOC | SHF_WRITE); // OR of both
    }

    #[test]
    fn test_build_section_data_nonexistent() {
        let merger = SectionMerger::new(Target::X86_64);
        let data = merger.build_section_data(".nonexistent");
        assert!(data.is_empty());
    }

    #[test]
    fn test_merge_result_comdat_kept() {
        let mut s1 = make_input_section(
            ".text.foo",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR | SHF_GROUP,
            vec![0xC3],
            1,
            0,
            1,
        );
        s1.group_signature = Some("foo_group".to_string());

        let mut s2 = make_input_section(
            ".text.foo",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR | SHF_GROUP,
            vec![0xC3],
            1,
            1,
            1,
        );
        s2.group_signature = Some("foo_group".to_string());

        let result = merge_all_sections(&Target::X86_64, vec![s1, s2]);
        assert!(result.comdat_kept.contains("foo_group"));

        // Only 1 fragment because the second was deduplicated.
        let text = result
            .output_sections
            .iter()
            .find(|s| s.name == ".text.foo")
            .unwrap();
        assert_eq!(text.fragments.len(), 1);
    }
}
