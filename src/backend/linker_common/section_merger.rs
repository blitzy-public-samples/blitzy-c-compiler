//! Input section aggregation and output section layout engine for the BCC
//! built-in linker.
//!
//! This module collects input sections from multiple relocatable object files,
//! groups them by name and type into merged output sections following the
//! standard ELF ordering (`.text`, `.rodata`, `.data`, `.bss`), computes
//! alignment padding between merged sections, handles COMDAT/section groups
//! for C inline function and template deduplication, and produces the final
//! output section layout with assigned virtual addresses and file offsets.
//!
//! # Architecture
//!
//! The merging pipeline operates in four phases:
//!
//! 1. **Collection** — [`SectionMerger::add_input_section`] collects sections
//!    from individual object files, merging same-name sections together.
//! 2. **COMDAT Deduplication** — [`SectionMerger::handle_comdat_group`] ensures
//!    only the first occurrence of each section group is kept.
//! 3. **Ordering** — [`SectionMerger::compute_section_order`] sorts output
//!    sections into the standard ELF layout order.
//! 4. **Layout** — [`SectionMerger::assign_addresses`] and
//!    [`SectionMerger::assign_file_offsets`] compute virtual addresses and
//!    file offsets for the final ELF binary.
//!
//! # Usage by Architecture Backends
//!
//! All four architecture-specific linkers (x86-64, i686, AArch64, RISC-V 64)
//! use this module as their shared section-merging infrastructure. Removing
//! this module would break all multi-object-file linking since input sections
//! could not be combined into output sections.
//!
//! # Zero-Dependency Implementation
//!
//! This module uses only the Rust standard library and internal BCC modules,
//! adhering to the project's strict zero-dependency mandate.

use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;
use crate::backend::elf_writer_common::{
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE,
    SHT_DYNAMIC, SHT_DYNSYM, SHT_NOBITS, SHT_NULL,
    SHT_RELA, SHT_STRTAB, SHT_SYMTAB,
};

// ===========================================================================
// InputRelocation — relocation entry from an input object file
// ===========================================================================

/// A relocation entry from an input object file, associated with its parent
/// [`InputSection`].
///
/// Each relocation specifies a fixup that the linker must apply to the section
/// data at link time. All four supported architectures use the RELA format
/// (explicit addend), so the `addend` field is always populated.
///
/// # Fields
///
/// - `offset` — Byte offset within the section where the relocation applies.
/// - `reloc_type` — Architecture-specific relocation type code
///   (e.g., `R_X86_64_PC32`, `R_AARCH64_CALL26`).
/// - `symbol_index` — Index into the input object's symbol table identifying
///   the target symbol.
/// - `addend` — Constant addend used in relocation computation.
/// - `section_index` — Index of the section within the input object file that
///   this relocation references (for section-relative symbols).
#[derive(Clone, Debug)]
pub struct InputRelocation {
    /// Byte offset from the start of the owning section where the relocation
    /// value must be written.
    pub offset: u64,

    /// Architecture-specific relocation type (e.g., R_X86_64_PC32 = 2,
    /// R_AARCH64_CALL26 = 283, R_RISCV_JAL = 17).
    pub reloc_type: u32,

    /// Index of the target symbol in the input object's symbol table.
    pub symbol_index: u32,

    /// Explicit addend for RELA-style relocations.
    pub addend: i64,

    /// Index of the referenced section in the input object file (for
    /// section-relative symbols, e.g., STT_SECTION).
    pub section_index: usize,
}

// ===========================================================================
// InputSection — a single section from one input object file
// ===========================================================================

/// A single section read from a relocatable object file (`.o`).
///
/// Each input object file contains zero or more sections (`.text`, `.data`,
/// `.bss`, `.rodata`, `.rela.text`, etc.). During linking, sections with the
/// same name across multiple objects are merged into a single output section.
///
/// For `SHT_NOBITS` (`.bss`) sections, the `data` vector carries the virtual
/// size information — its length represents the BSS contribution even though
/// the bytes themselves are never written to the output file.
#[derive(Clone, Debug)]
pub struct InputSection {
    /// Section name from the input object (e.g., `.text`, `.data`, `.bss`).
    pub name: String,

    /// ELF section type (`SHT_PROGBITS`, `SHT_NOBITS`, `SHT_RELA`, etc.).
    pub section_type: u32,

    /// ELF section flags (`SHF_ALLOC`, `SHF_WRITE`, `SHF_EXECINSTR`, etc.).
    pub flags: u64,

    /// Section content bytes. For `SHT_NOBITS` sections, the length encodes
    /// the virtual size but the bytes are not written to the output file.
    pub data: Vec<u8>,

    /// Required alignment in bytes (must be a power of two, minimum 1).
    pub alignment: u64,

    /// Fixed entry size for sections with uniform entries (e.g., symbol tables).
    /// Zero for sections without fixed-size entries.
    pub entry_size: u64,

    /// COMDAT group identifier, if this section belongs to a section group.
    /// `None` for sections not in any group.
    pub group_id: Option<u32>,

    /// Index of the originating object file in the linker's input list.
    pub object_index: usize,

    /// Original section index within the input object's section header table.
    pub original_index: usize,

    /// Relocations that reference locations within this section.
    pub relocations: Vec<InputRelocation>,
}

// ===========================================================================
// MergedInput — tracks placement of one input section within an output section
// ===========================================================================

/// Records the placement of a single [`InputSection`] within its parent
/// [`OutputSection`].
///
/// After merging, the linker can use `offset_in_output` to translate
/// input-section-relative addresses to output-section-relative addresses
/// for relocation processing.
#[derive(Clone, Debug)]
pub struct MergedInput {
    /// The original input section that was merged into the output.
    pub input: InputSection,

    /// Byte offset of this input section's data within the output section,
    /// accounting for alignment padding inserted before it.
    pub offset_in_output: u64,
}

// ===========================================================================
// OutputSection — merged section in the output ELF file
// ===========================================================================

/// A merged output section containing contributions from one or more input
/// object files.
///
/// Each output section has a name (`.text`, `.data`, etc.), assigned virtual
/// address, file offset, and size. The `input_sections` vector records the
/// provenance and placement of every input section that was merged into this
/// output section.
#[derive(Clone, Debug)]
pub struct OutputSection {
    /// Section name in the output ELF file (e.g., `.text`, `.data`, `.bss`).
    pub name: String,

    /// ELF section type — typically `SHT_PROGBITS` for code/data or
    /// `SHT_NOBITS` for `.bss`.
    pub section_type: u32,

    /// Combined ELF section flags from all merged input sections.
    pub flags: u64,

    /// Required alignment in bytes (maximum of all constituent input sections).
    pub alignment: u64,

    /// Assigned virtual address (populated by [`SectionMerger::assign_addresses`]).
    pub addr: u64,

    /// Assigned file offset in the output ELF (populated by
    /// [`SectionMerger::assign_file_offsets`]).
    pub offset: u64,

    /// Total size of this output section in bytes, including alignment padding
    /// between merged input sections.
    pub size: u64,

    /// Ordered list of input sections merged into this output section, each
    /// with its offset within the output.
    pub input_sections: Vec<MergedInput>,

    /// Fixed entry size (non-zero for symbol tables, rela sections, etc.).
    pub entry_size: u64,
}

// ===========================================================================
// SectionMerger — the merging engine
// ===========================================================================

/// The section merging engine for the BCC built-in linker.
///
/// Collects input sections from multiple object files, deduplicates COMDAT
/// groups, merges compatible sections, orders them following the standard ELF
/// layout, and computes virtual addresses and file offsets.
///
/// # Example
///
/// ```ignore
/// let mut merger = SectionMerger::new();
///
/// // Add sections from each input object
/// merger.add_input_section(text_section_from_obj1);
/// merger.add_input_section(data_section_from_obj1);
/// merger.add_input_section(text_section_from_obj2);
///
/// // Sort into standard ELF order
/// merger.compute_section_order();
///
/// // Assign layout
/// merger.assign_addresses(0x400000);
/// merger.assign_file_offsets(0x1000);
///
/// // Access the finalized layout
/// for section in merger.output_sections() {
///     println!("{}: addr=0x{:x} offset=0x{:x} size=0x{:x}",
///              section.name, section.addr, section.offset, section.size);
/// }
/// ```
pub struct SectionMerger {
    /// Ordered list of output sections produced by merging input sections.
    output_sections: Vec<OutputSection>,

    /// Maps output section names to their indices in `output_sections` for
    /// O(1) lookup during input section collection.
    section_map: FxHashMap<String, usize>,

    /// Tracks processed COMDAT group signatures. Maps group_id → group_name.
    /// Used for first-wins deduplication: the first group with a given
    /// signature is kept, subsequent duplicates are discarded.
    comdat_groups: FxHashMap<u32, String>,
}

// ===========================================================================
// Free helper functions
// ===========================================================================

/// Computes the number of padding bytes needed to advance `current_offset`
/// to the next multiple of `alignment`.
///
/// Returns 0 if `current_offset` is already aligned or if `alignment` is 0 or 1.
///
/// # Panics
///
/// This function does not panic. If `alignment` is 0, it returns 0 padding.
#[inline]
pub fn compute_padding(current_offset: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return 0;
    }
    let remainder = current_offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

/// Returns a numeric priority for section ordering that follows the standard
/// ELF output layout. Lower values appear earlier in the output file.
///
/// The ordering is designed to group sections by their ELF segment membership:
///
/// 1. Metadata sections loaded early (`.interp`, `.note.*`, `.gnu.hash`)
/// 2. Dynamic symbol resolution sections (`.dynsym`, `.dynstr`, `.rela.*`)
/// 3. Executable code (`.init`, `.plt`, `.text`, `.fini`)
/// 4. Read-only data (`.rodata`, `.eh_frame`)
/// 5. Writable initialized data (`.init_array`, `.fini_array`, `.dynamic`,
///    `.got`, `.got.plt`, `.data`)
/// 6. Writable zero-initialized data (`.bss`)
/// 7. Non-loadable sections (`.comment`, `.symtab`, `.strtab`, `.shstrtab`)
/// 8. Debug sections (`.debug_*`)
/// 9. Everything else
fn section_order_priority(name: &str, section_type: u32, flags: u64) -> u32 {
    // Special well-known section names get explicit ordering
    match name {
        // ---- Early metadata (loaded in first PT_LOAD or separate segments) ----
        ".interp" => 10,
        n if n.starts_with(".note") => 15,
        ".gnu.hash" => 20,

        // ---- Dynamic linking resolution tables ----
        ".dynsym" => 30,
        ".dynstr" => 31,
        ".rela.dyn" => 40,
        ".rela.plt" => 41,
        n if n.starts_with(".rela.") || n.starts_with(".rel.") => 42,

        // ---- Executable code segment ----
        ".init" => 50,
        ".plt" => 55,
        ".plt.got" => 56,
        ".text" => 60,
        ".fini" => 65,

        // ---- Read-only data segment ----
        ".rodata" => 70,
        n if n.starts_with(".rodata.") => 71,
        ".eh_frame_hdr" => 75,
        ".eh_frame" => 76,

        // ---- Writable data segment ----
        ".preinit_array" => 80,
        ".init_array" => 81,
        ".fini_array" => 82,
        ".dynamic" => 85,
        ".got" => 90,
        ".got.plt" => 91,
        ".data" => 95,
        n if n.starts_with(".data.") => 96,
        ".bss" => 100,
        n if n.starts_with(".bss.") => 101,

        // ---- Non-loadable metadata sections ----
        ".comment" => 200,
        ".symtab" => 210,
        ".strtab" => 220,
        ".shstrtab" => 230,

        // ---- Debug sections ----
        ".debug_info" => 300,
        ".debug_abbrev" => 301,
        ".debug_line" => 302,
        ".debug_str" => 303,
        ".debug_ranges" => 304,
        ".debug_loc" => 305,
        ".debug_aranges" => 306,
        n if n.starts_with(".debug_") => 310,

        // ---- Fallback: classify by flags and type ----
        _ => classify_by_attributes(section_type, flags),
    }
}

/// Fallback section ordering classification based on ELF section attributes
/// when the section name is not a well-known standard name.
///
/// Uses section type and flags to place the section in the appropriate
/// region of the output layout.
fn classify_by_attributes(section_type: u32, flags: u64) -> u32 {
    let allocatable = flags & SHF_ALLOC != 0;
    let writable = flags & SHF_WRITE != 0;
    let executable = flags & SHF_EXECINSTR != 0;

    if !allocatable {
        // Non-loadable sections go after all loadable content
        return match section_type {
            SHT_SYMTAB => 210,
            SHT_STRTAB => 220,
            SHT_RELA => 215,
            SHT_NULL => 999,
            _ => 250,
        };
    }

    // Allocatable sections classified by permissions
    if executable {
        62 // Near .text
    } else if section_type == SHT_NOBITS {
        102 // Near .bss
    } else if writable {
        97 // Near .data
    } else if section_type == SHT_DYNAMIC {
        85 // With .dynamic
    } else if section_type == SHT_DYNSYM {
        30 // With .dynsym
    } else {
        72 // Near .rodata (read-only allocated)
    }
}

/// Returns the minimum alignment recommended for pointer-sized data sections
/// (`.got`, `.got.plt`, pointer tables) for the given target architecture.
///
/// Uses [`Target::pointer_width`] to ensure GOT and PLT entries are naturally
/// aligned for the target's pointer size.
pub fn pointer_alignment_for_target(target: &Target) -> u64 {
    target.pointer_width() as u64
}

/// Returns the minimum code section alignment recommended for the given target
/// architecture.
///
/// Uses [`Target::stack_alignment`] to ensure function entry points respect
/// the ABI's call-site stack alignment requirement, which also serves as a
/// reasonable minimum alignment for executable sections.
pub fn code_alignment_for_target(target: &Target) -> u64 {
    target.stack_alignment() as u64
}

// ===========================================================================
// SectionMerger implementation
// ===========================================================================

impl SectionMerger {
    /// Creates a new, empty section merger.
    ///
    /// The merger starts with no output sections. Call
    /// [`add_input_section`](Self::add_input_section) to begin collecting
    /// sections from input object files.
    pub fn new() -> Self {
        SectionMerger {
            output_sections: Vec::new(),
            section_map: FxHashMap::default(),
            comdat_groups: FxHashMap::default(),
        }
    }

    /// Adds a section from an input object file to the merger.
    ///
    /// Sections with the same name are merged into a single output section.
    /// The output section's alignment is the maximum alignment of all
    /// constituent input sections, and the combined flags are the union of
    /// all input flags.
    ///
    /// `SHT_NULL` sections are silently ignored. Sections belonging to a
    /// COMDAT group that was already processed (i.e., a duplicate) are also
    /// discarded.
    ///
    /// # Alignment Padding
    ///
    /// When a new input section is appended to an existing output section,
    /// zero-padding is inserted as necessary to satisfy the input section's
    /// alignment requirement.
    pub fn add_input_section(&mut self, input: InputSection) {
        // Ignore null sections — they carry no data and serve only as
        // placeholders in the section header table.
        if input.section_type == SHT_NULL {
            return;
        }

        // If this section belongs to a COMDAT group that was already seen,
        // discard it (first-wins deduplication rule).
        if let Some(group_id) = input.group_id {
            if self.comdat_groups.contains_key(&group_id) {
                return;
            }
        }

        // Ensure alignment is at least 1 to avoid division-by-zero in padding
        // calculations. ELF spec requires sh_addralign >= 1 for allocated
        // sections.
        let alignment = input.alignment.max(1);
        let input_size = input.data.len() as u64;
        let section_name = input.name.clone();
        let input_entry_size = input.entry_size;
        let input_flags = input.flags;
        let input_section_type = input.section_type;

        if let Some(&idx) = self.section_map.get(&section_name) {
            // ---- Merge into existing output section ----
            let output = &mut self.output_sections[idx];

            // Raise alignment to the maximum seen across all input sections
            if alignment > output.alignment {
                output.alignment = alignment;
            }

            // Compute alignment padding from the current end-of-section to the
            // next aligned boundary for this input contribution.
            let padding = compute_padding(output.size, alignment);
            let offset_in_output = output.size + padding;

            // Accumulate size: virtual size grows by padding + input size
            output.size = offset_in_output + input_size;

            // Merge entry_size: adopt the first non-zero value (must be
            // consistent across all input sections contributing to the same
            // output section).
            if output.entry_size == 0 && input_entry_size > 0 {
                output.entry_size = input_entry_size;
            }

            // Union of flags — if any input section has SHF_WRITE, the output
            // must also be writable, etc.
            output.flags |= input_flags;

            // Record the merged input section with its placement offset
            output.input_sections.push(MergedInput {
                input,
                offset_in_output,
            });
        } else {
            // ---- Create a new output section ----
            let idx = self.output_sections.len();

            let output = OutputSection {
                name: section_name.clone(),
                section_type: input_section_type,
                flags: input_flags,
                alignment,
                addr: 0,
                offset: 0,
                size: input_size,
                input_sections: vec![MergedInput {
                    input,
                    offset_in_output: 0,
                }],
                entry_size: input_entry_size,
            };

            self.output_sections.push(output);
            self.section_map.insert(section_name, idx);
        }
    }

    /// Processes a COMDAT section group.
    ///
    /// COMDAT groups (also known as section groups with `GRP_COMDAT` flag)
    /// provide deduplication for symbols that may be defined in multiple
    /// translation units — primarily C inline functions and C++ template
    /// instantiations.
    ///
    /// The **first group** with a given `group_id` wins: its sections are
    /// merged into the output normally. All **subsequent groups** with the
    /// same `group_id` are silently discarded.
    ///
    /// # Parameters
    ///
    /// - `group_id` — Unique identifier for the section group (typically
    ///   derived from the group's signature symbol index).
    /// - `group_name` — Human-readable group name (e.g., the signature symbol
    ///   name), stored for diagnostic purposes.
    /// - `sections` — Slice of input sections belonging to this group.
    pub fn handle_comdat_group(
        &mut self,
        group_id: u32,
        group_name: &str,
        sections: &[InputSection],
    ) {
        // Use the entry API for efficient check-and-insert on the COMDAT map.
        // If the group_id is already registered (Occupied), all sections in
        // this duplicate group are silently discarded. If it's new (Vacant),
        // register the group and proceed to add its sections.
        use std::collections::hash_map::Entry;
        match self.comdat_groups.entry(group_id) {
            Entry::Occupied(_) => return,
            Entry::Vacant(vacant) => {
                vacant.insert(group_name.to_string());
            }
        }

        // Add each section from the winning group. Clone is required because
        // we receive the sections by reference but add_input_section takes
        // ownership. We clear the group_id on the cloned sections so they
        // pass through add_input_section() without being filtered by the
        // COMDAT check — the group_id was just registered above, which would
        // otherwise cause add_input_section() to discard these sections.
        // Future duplicate sections arriving via add_input_section() with the
        // same group_id will still be correctly filtered out because the
        // comdat_groups map now contains this group_id.
        for section in sections {
            let mut owned = section.clone();
            // Clear the group_id so add_input_section does not treat this
            // section as a duplicate COMDAT member.
            owned.group_id = None;
            self.add_input_section(owned);
        }
    }

    /// Sorts output sections into the standard ELF layout order.
    ///
    /// The ordering follows the conventional ELF layout:
    ///
    /// 1. `.interp`, `.note.*`, `.gnu.hash` — early metadata
    /// 2. `.dynsym`, `.dynstr`, `.rela.*` — dynamic linking tables
    /// 3. `.init`, `.plt`, `.text`, `.fini` — executable code
    /// 4. `.rodata`, `.eh_frame` — read-only data
    /// 5. `.data`, `.got`, `.got.plt` — writable initialized data
    /// 6. `.bss` — writable zero-initialized data
    /// 7. `.symtab`, `.strtab`, `.shstrtab` — non-loadable metadata
    /// 8. `.debug_*` — debug information
    ///
    /// After sorting, the internal `section_map` is rebuilt to reflect the
    /// new indices.
    pub fn compute_section_order(&mut self) {
        // Sort output sections by their standard ELF ordering priority
        self.output_sections.sort_by(|a, b| {
            let pa = section_order_priority(&a.name, a.section_type, a.flags);
            let pb = section_order_priority(&b.name, b.section_type, b.flags);
            pa.cmp(&pb).then_with(|| a.name.cmp(&b.name))
        });

        // Rebuild the section name → index map to reflect the new ordering
        self.section_map.clear();
        for (idx, section) in self.output_sections.iter().enumerate() {
            self.section_map.insert(section.name.clone(), idx);
        }
    }

    /// Assigns virtual addresses to all output sections starting from
    /// `base_address`.
    ///
    /// Each output section is placed at the next address that satisfies its
    /// alignment requirement. The address is advanced by the section's size
    /// (including any internal padding between merged input sections).
    ///
    /// `.bss` (`SHT_NOBITS`) sections contribute to the virtual address space
    /// but will not occupy file space — their virtual size still advances the
    /// address counter.
    ///
    /// Non-allocatable sections (those without `SHF_ALLOC`) receive an address
    /// of 0, as they are not mapped into the process address space.
    ///
    /// # Parameters
    ///
    /// - `base_address` — The starting virtual address for the first allocatable
    ///   output section (e.g., `0x400000` for x86-64 executables).
    pub fn assign_addresses(&mut self, base_address: u64) {
        let mut current_addr = base_address;

        for section in &mut self.output_sections {
            // Non-allocatable sections are not loaded into memory
            if section.flags & SHF_ALLOC == 0 {
                section.addr = 0;
                continue;
            }

            // Align the current address to the section's alignment requirement
            let padding = compute_padding(current_addr, section.alignment);
            current_addr += padding;

            // Assign the virtual address
            section.addr = current_addr;

            // Advance past this section's content. BSS sections contribute
            // virtual size even though they have no file content.
            current_addr += section.size;
        }
    }

    /// Assigns file offsets to all output sections starting from
    /// `initial_offset`.
    ///
    /// Each section that contributes to the file is placed at the next offset
    /// that satisfies its alignment requirement. `SHT_NOBITS` sections
    /// (`.bss`) do **not** consume file space — they receive the current
    /// offset value but do not advance the file pointer.
    ///
    /// Non-allocatable sections (e.g., `.symtab`, `.strtab`, `.debug_*`) are
    /// also assigned file offsets since they exist in the ELF file but are
    /// not mapped into the process address space.
    ///
    /// # Parameters
    ///
    /// - `initial_offset` — The starting file offset, typically after the ELF
    ///   header and program header table (e.g., `0x1000` or the first
    ///   page-aligned offset after headers).
    pub fn assign_file_offsets(&mut self, initial_offset: u64) {
        let mut current_offset = initial_offset;

        for section in &mut self.output_sections {
            // Align the file offset to the section's alignment
            let padding = compute_padding(current_offset, section.alignment);
            current_offset += padding;

            // Assign the file offset
            section.offset = current_offset;

            // SHT_NOBITS sections (BSS) do not occupy file space.
            // All other sections advance the file offset by their size.
            if section.section_type != SHT_NOBITS {
                current_offset += section.size;
            }
        }
    }

    /// Returns an immutable reference to the ordered list of output sections.
    ///
    /// The returned slice reflects the current ordering and layout. Call
    /// [`compute_section_order`](Self::compute_section_order),
    /// [`assign_addresses`](Self::assign_addresses), and
    /// [`assign_file_offsets`](Self::assign_file_offsets) before accessing
    /// sections to ensure the layout is finalized.
    pub fn output_sections(&self) -> &[OutputSection] {
        &self.output_sections
    }

    /// Returns a mutable reference to the output sections vector.
    ///
    /// This is used by the relocation processor and other linker components
    /// that need to modify output section data (e.g., patching relocation
    /// values into section content).
    pub fn output_sections_mut(&mut self) -> &mut Vec<OutputSection> {
        &mut self.output_sections
    }

    /// Returns the number of output sections currently held by the merger.
    pub fn section_count(&self) -> usize {
        self.output_sections.len()
    }

    /// Looks up an output section by name and returns its index.
    ///
    /// Returns `None` if no output section with the given name exists.
    pub fn find_section(&self, name: &str) -> Option<usize> {
        self.section_map.get(name).copied()
    }

    /// Returns the total file size consumed by all output sections that have
    /// been assigned file offsets and occupy file space.
    ///
    /// `SHT_NOBITS` sections are excluded from this calculation.
    pub fn total_file_size(&self) -> u64 {
        let mut total: u64 = 0;
        for section in &self.output_sections {
            if section.section_type != SHT_NOBITS {
                let end = section.offset + section.size;
                if end > total {
                    total = end;
                }
            }
        }
        total
    }

    /// Returns the total virtual address span consumed by all allocatable
    /// output sections.
    pub fn total_virtual_size(&self) -> u64 {
        let mut max_end: u64 = 0;
        for section in &self.output_sections {
            if section.flags & SHF_ALLOC != 0 {
                let end = section.addr + section.size;
                if end > max_end {
                    max_end = end;
                }
            }
        }
        max_end
    }

    /// Collects the raw data bytes for a given output section by concatenating
    /// all merged input section data with appropriate alignment padding.
    ///
    /// For `SHT_NOBITS` sections, returns an empty vector (BSS has no file
    /// content).
    ///
    /// # Parameters
    ///
    /// - `section_index` — Index into the output sections list.
    ///
    /// # Returns
    ///
    /// The fully assembled section content with padding, or an empty vector
    /// for BSS sections.
    pub fn collect_section_data(&self, section_index: usize) -> Vec<u8> {
        let section = &self.output_sections[section_index];

        // BSS sections have no file content
        if section.section_type == SHT_NOBITS {
            return Vec::new();
        }

        // Pre-allocate the output buffer to the full section size
        let mut data = vec![0u8; section.size as usize];

        // Copy each input section's data into the output at its assigned offset
        for merged in &section.input_sections {
            let start = merged.offset_in_output as usize;
            let end = start + merged.input.data.len();
            if end <= data.len() {
                data[start..end].copy_from_slice(&merged.input.data);
            }
        }

        data
    }
}

impl Default for SectionMerger {
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
    use crate::backend::elf_writer_common::SHT_PROGBITS;

    /// Helper to create a minimal input section for testing.
    fn make_input_section(
        name: &str,
        section_type: u32,
        flags: u64,
        data: &[u8],
        alignment: u64,
    ) -> InputSection {
        InputSection {
            name: name.to_string(),
            section_type,
            flags,
            data: data.to_vec(),
            alignment,
            entry_size: 0,
            group_id: None,
            object_index: 0,
            original_index: 0,
            relocations: Vec::new(),
        }
    }

    #[test]
    fn test_compute_padding_already_aligned() {
        assert_eq!(compute_padding(16, 16), 0);
        assert_eq!(compute_padding(0, 4), 0);
        assert_eq!(compute_padding(256, 16), 0);
    }

    #[test]
    fn test_compute_padding_needs_padding() {
        assert_eq!(compute_padding(1, 4), 3);
        assert_eq!(compute_padding(5, 8), 3);
        assert_eq!(compute_padding(13, 16), 3);
        assert_eq!(compute_padding(7, 4), 1);
    }

    #[test]
    fn test_compute_padding_alignment_one() {
        // Alignment of 1 means everything is already aligned
        assert_eq!(compute_padding(0, 1), 0);
        assert_eq!(compute_padding(7, 1), 0);
        assert_eq!(compute_padding(999, 1), 0);
    }

    #[test]
    fn test_compute_padding_alignment_zero() {
        // Edge case: alignment 0 should not divide-by-zero
        assert_eq!(compute_padding(0, 0), 0);
        assert_eq!(compute_padding(42, 0), 0);
    }

    #[test]
    fn test_add_single_section() {
        let mut merger = SectionMerger::new();
        let section = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 16], // 16 bytes of NOP
            16,
        );

        merger.add_input_section(section);

        assert_eq!(merger.output_sections().len(), 1);
        let out = &merger.output_sections()[0];
        assert_eq!(out.name, ".text");
        assert_eq!(out.size, 16);
        assert_eq!(out.alignment, 16);
        assert_eq!(out.input_sections.len(), 1);
        assert_eq!(out.input_sections[0].offset_in_output, 0);
    }

    #[test]
    fn test_merge_same_name_sections() {
        let mut merger = SectionMerger::new();

        // First .text section: 8 bytes, alignment 4
        let s1 = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 8],
            4,
        );
        merger.add_input_section(s1);

        // Second .text section: 12 bytes, alignment 8
        let s2 = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0xCC; 12],
            8,
        );
        merger.add_input_section(s2);

        assert_eq!(merger.output_sections().len(), 1);
        let out = &merger.output_sections()[0];
        assert_eq!(out.name, ".text");
        // Alignment raised to max(4, 8) = 8
        assert_eq!(out.alignment, 8);
        // s1: 8 bytes at offset 0
        // s2: needs alignment 8, offset 8 is already aligned → offset = 8
        // total size = 8 + 12 = 20
        assert_eq!(out.size, 20);
        assert_eq!(out.input_sections.len(), 2);
        assert_eq!(out.input_sections[0].offset_in_output, 0);
        assert_eq!(out.input_sections[1].offset_in_output, 8);
    }

    #[test]
    fn test_merge_with_alignment_padding() {
        let mut merger = SectionMerger::new();

        // First section: 5 bytes, alignment 1
        let s1 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[1, 2, 3, 4, 5],
            1,
        );
        merger.add_input_section(s1);

        // Second section: 4 bytes, alignment 8
        // Current offset is 5, needs padding to reach 8
        let s2 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[6, 7, 8, 9],
            8,
        );
        merger.add_input_section(s2);

        let out = &merger.output_sections()[0];
        assert_eq!(out.alignment, 8);
        assert_eq!(out.input_sections[0].offset_in_output, 0);
        // 5 bytes + 3 padding = offset 8
        assert_eq!(out.input_sections[1].offset_in_output, 8);
        // total = 8 + 4 = 12
        assert_eq!(out.size, 12);
    }

    #[test]
    fn test_different_sections_stay_separate() {
        let mut merger = SectionMerger::new();

        let text = make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 8],
            4,
        );
        let data = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0xFF; 16],
            4,
        );

        merger.add_input_section(text);
        merger.add_input_section(data);

        assert_eq!(merger.output_sections().len(), 2);
        assert_eq!(merger.output_sections()[0].name, ".text");
        assert_eq!(merger.output_sections()[1].name, ".data");
    }

    #[test]
    fn test_null_section_ignored() {
        let mut merger = SectionMerger::new();

        let null_sec = make_input_section("", SHT_NULL, 0, &[], 0);
        merger.add_input_section(null_sec);

        assert_eq!(merger.output_sections().len(), 0);
    }

    #[test]
    fn test_comdat_first_wins() {
        let mut merger = SectionMerger::new();

        let group1 = [make_input_section(
            ".text.inline_func",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0xAA; 8],
            4,
        )];

        let group2 = [make_input_section(
            ".text.inline_func",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0xBB; 8],
            4,
        )];

        // First group wins
        merger.handle_comdat_group(42, "inline_func", &group1);
        // Second group is discarded
        merger.handle_comdat_group(42, "inline_func", &group2);

        assert_eq!(merger.output_sections().len(), 1);
        // Data should be from group1 (0xAA bytes)
        let merged = &merger.output_sections()[0].input_sections[0];
        assert_eq!(merged.input.data, vec![0xAA; 8]);
    }

    #[test]
    fn test_comdat_different_groups_both_kept() {
        let mut merger = SectionMerger::new();

        let group1 = [make_input_section(
            ".text.func_a",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0xAA; 4],
            4,
        )];

        let group2 = [make_input_section(
            ".text.func_b",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0xBB; 4],
            4,
        )];

        merger.handle_comdat_group(1, "func_a", &group1);
        merger.handle_comdat_group(2, "func_b", &group2);

        // Both groups have different IDs, so both are kept
        assert_eq!(merger.output_sections().len(), 2);
    }

    #[test]
    fn test_section_ordering() {
        let mut merger = SectionMerger::new();

        // Add sections in reverse order
        merger.add_input_section(make_input_section(
            ".debug_info",
            SHT_PROGBITS,
            0,
            &[0; 4],
            1,
        ));
        merger.add_input_section(make_input_section(
            ".bss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 16],
            4,
        ));
        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 8],
            16,
        ));
        merger.add_input_section(make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0xFF; 8],
            4,
        ));
        merger.add_input_section(make_input_section(
            ".rodata",
            SHT_PROGBITS,
            SHF_ALLOC,
            &[0; 8],
            4,
        ));

        merger.compute_section_order();

        let names: Vec<&str> = merger
            .output_sections()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec![".text", ".rodata", ".data", ".bss", ".debug_info"]);
    }

    #[test]
    fn test_assign_addresses() {
        let mut merger = SectionMerger::new();

        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 64],
            16,
        ));
        merger.add_input_section(make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0xFF; 32],
            8,
        ));
        merger.add_input_section(make_input_section(
            ".bss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 128],
            16,
        ));

        merger.compute_section_order();
        merger.assign_addresses(0x400000);

        let sections = merger.output_sections();

        // .text: aligned to 16 from 0x400000 → 0x400000 (already aligned)
        assert_eq!(sections[0].name, ".text");
        assert_eq!(sections[0].addr, 0x400000);

        // .data: after .text (0x400000 + 64 = 0x400040), aligned to 8 → 0x400040
        assert_eq!(sections[1].name, ".data");
        assert_eq!(sections[1].addr, 0x400040);

        // .bss: after .data (0x400040 + 32 = 0x400060), aligned to 16 → 0x400060
        assert_eq!(sections[2].name, ".bss");
        assert_eq!(sections[2].addr, 0x400060);
    }

    #[test]
    fn test_assign_file_offsets_bss_no_file_space() {
        let mut merger = SectionMerger::new();

        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 64],
            16,
        ));
        merger.add_input_section(make_input_section(
            ".bss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 1024],
            16,
        ));
        merger.add_input_section(make_input_section(
            ".symtab",
            SHT_SYMTAB,
            0,
            &[0; 48],
            8,
        ));

        merger.compute_section_order();
        merger.assign_file_offsets(0x1000);

        let sections = merger.output_sections();

        // .text at offset 0x1000 (aligned to 16, already at 0x1000)
        let text = sections.iter().find(|s| s.name == ".text").unwrap();
        assert_eq!(text.offset, 0x1000);

        // .bss gets an offset but does NOT advance the file pointer
        let bss = sections.iter().find(|s| s.name == ".bss").unwrap();
        assert!(bss.offset >= text.offset + text.size);

        // .symtab follows .text in file (since .bss didn't consume file space)
        let symtab = sections.iter().find(|s| s.name == ".symtab").unwrap();
        // .bss didn't advance the offset, so .symtab starts after .bss's offset
        // but its offset is after the .text data
        assert!(symtab.offset > 0);
    }

    #[test]
    fn test_non_alloc_section_gets_zero_address() {
        let mut merger = SectionMerger::new();

        merger.add_input_section(make_input_section(
            ".symtab",
            SHT_SYMTAB,
            0, // No SHF_ALLOC
            &[0; 48],
            8,
        ));

        merger.assign_addresses(0x400000);

        assert_eq!(merger.output_sections()[0].addr, 0);
    }

    #[test]
    fn test_collect_section_data_with_padding() {
        let mut merger = SectionMerger::new();

        // First contribution: 3 bytes
        let s1 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0x11, 0x22, 0x33],
            1,
        );
        merger.add_input_section(s1);

        // Second contribution: 2 bytes, alignment 4
        // Current size = 3, padding needed = 1 to reach offset 4
        let s2 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0xAA, 0xBB],
            4,
        );
        merger.add_input_section(s2);

        let data = merger.collect_section_data(0);
        // Total size = 4 (aligned offset of s2) + 2 = 6
        assert_eq!(data.len(), 6);
        // First 3 bytes from s1
        assert_eq!(data[0], 0x11);
        assert_eq!(data[1], 0x22);
        assert_eq!(data[2], 0x33);
        // 1 byte of padding (zero)
        assert_eq!(data[3], 0x00);
        // 2 bytes from s2
        assert_eq!(data[4], 0xAA);
        assert_eq!(data[5], 0xBB);
    }

    #[test]
    fn test_collect_section_data_bss_empty() {
        let mut merger = SectionMerger::new();

        merger.add_input_section(make_input_section(
            ".bss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 256],
            16,
        ));

        let data = merger.collect_section_data(0);
        assert!(data.is_empty(), "BSS section data should be empty");
    }

    #[test]
    fn test_flags_union_on_merge() {
        let mut merger = SectionMerger::new();

        // First section: only SHF_ALLOC
        let s1 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC,
            &[0; 4],
            4,
        );
        merger.add_input_section(s1);

        // Second section: SHF_ALLOC | SHF_WRITE
        let s2 = make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 4],
            4,
        );
        merger.add_input_section(s2);

        // Merged flags should be the union
        let out = &merger.output_sections()[0];
        assert_ne!(out.flags & SHF_ALLOC, 0);
        assert_ne!(out.flags & SHF_WRITE, 0);
    }

    #[test]
    fn test_find_section() {
        let mut merger = SectionMerger::new();

        merger.add_input_section(make_input_section(
            ".text",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            &[0x90; 8],
            4,
        ));
        merger.add_input_section(make_input_section(
            ".data",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            &[0; 8],
            4,
        ));

        assert_eq!(merger.find_section(".text"), Some(0));
        assert_eq!(merger.find_section(".data"), Some(1));
        assert_eq!(merger.find_section(".bss"), None);
    }

    #[test]
    fn test_target_alignment_helpers() {
        // Verify that Target::pointer_width and Target::stack_alignment are
        // used through our helper functions.
        let x86_64_ptr_align = pointer_alignment_for_target(&Target::X86_64);
        assert_eq!(x86_64_ptr_align, 8);

        let i686_ptr_align = pointer_alignment_for_target(&Target::I686);
        assert_eq!(i686_ptr_align, 4);

        let code_align = code_alignment_for_target(&Target::X86_64);
        assert_eq!(code_align, 16);

        let riscv_code = code_alignment_for_target(&Target::RiscV64);
        assert_eq!(riscv_code, 16);
    }

    #[test]
    fn test_section_merger_default() {
        let merger = SectionMerger::default();
        assert_eq!(merger.output_sections().len(), 0);
        assert_eq!(merger.section_count(), 0);
    }

    #[test]
    fn test_input_relocation_on_section() {
        let mut merger = SectionMerger::new();

        let section = InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0x90; 16],
            alignment: 4,
            entry_size: 0,
            group_id: None,
            object_index: 0,
            original_index: 1,
            relocations: vec![
                InputRelocation {
                    offset: 4,
                    reloc_type: 2, // e.g., R_X86_64_PC32
                    symbol_index: 1,
                    addend: -4,
                    section_index: 0,
                },
                InputRelocation {
                    offset: 12,
                    reloc_type: 4, // e.g., R_X86_64_PLT32
                    symbol_index: 2,
                    addend: -4,
                    section_index: 0,
                },
            ],
        };

        merger.add_input_section(section);

        let out = &merger.output_sections()[0];
        assert_eq!(out.input_sections[0].input.relocations.len(), 2);
        assert_eq!(out.input_sections[0].input.relocations[0].offset, 4);
        assert_eq!(out.input_sections[0].input.relocations[1].offset, 12);
    }

    #[test]
    fn test_full_pipeline() {
        // End-to-end test simulating a simple two-object link
        let mut merger = SectionMerger::new();

        // Object 1: .text (16 bytes), .data (8 bytes)
        merger.add_input_section(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0xCC; 16],
            alignment: 16,
            entry_size: 0,
            group_id: None,
            object_index: 0,
            original_index: 1,
            relocations: Vec::new(),
        });
        merger.add_input_section(InputSection {
            name: ".data".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0x42; 8],
            alignment: 8,
            entry_size: 0,
            group_id: None,
            object_index: 0,
            original_index: 2,
            relocations: Vec::new(),
        });

        // Object 2: .text (24 bytes), .bss (32 bytes)
        merger.add_input_section(InputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0xDD; 24],
            alignment: 4,
            entry_size: 0,
            group_id: None,
            object_index: 1,
            original_index: 1,
            relocations: Vec::new(),
        });
        merger.add_input_section(InputSection {
            name: ".bss".to_string(),
            section_type: SHT_NOBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0; 32],
            alignment: 8,
            entry_size: 0,
            group_id: None,
            object_index: 1,
            original_index: 3,
            relocations: Vec::new(),
        });

        // Order, assign addresses, assign file offsets
        merger.compute_section_order();
        merger.assign_addresses(0x400000);
        merger.assign_file_offsets(0x1000);

        let sections = merger.output_sections();
        assert_eq!(sections.len(), 3); // .text, .data, .bss

        // .text should be first
        assert_eq!(sections[0].name, ".text");
        assert_eq!(sections[0].addr, 0x400000);
        assert_eq!(sections[0].size, 40); // 16 + 24 (aligned to 16, then 4)

        // .data follows .text
        assert_eq!(sections[1].name, ".data");
        assert!(sections[1].addr >= 0x400000 + 40);

        // .bss last among alloc sections
        assert_eq!(sections[2].name, ".bss");
        assert!(sections[2].addr > sections[1].addr);

        // Verify file offsets: .bss should not advance file offset
        assert!(sections[0].offset >= 0x1000);
        assert!(sections[1].offset >= sections[0].offset + sections[0].size);
    }
}
