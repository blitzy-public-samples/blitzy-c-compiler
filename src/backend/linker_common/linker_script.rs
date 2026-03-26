//! # Default Linker Script Handling
//!
//! Implements the default section-to-segment mapping for BCC's built-in linker.
//! This module defines how input sections are placed into ELF output segments,
//! providing the equivalent of a default linker script used by GNU ld, but
//! implemented entirely within BCC — **NO external linker is invoked**.
//!
//! ## Scope
//!
//! Used by ALL four architecture backends (x86-64, i686, AArch64, RISC-V 64)
//! for both `ET_EXEC` (static executables) and `ET_DYN` (shared objects).
//!
//! ## Segment Layout
//!
//! The default layout maps sections to segments as follows:
//!
//! | Segment       | Type           | Flags        | Sections                                     |
//! |---------------|----------------|--------------|----------------------------------------------|
//! | Code          | `PT_LOAD`      | R+X          | `.init`, `.plt`, `.text`, `.fini`             |
//! | Read-only     | `PT_LOAD`      | R            | `.interp`, `.note.*`, `.gnu.hash`, …, `.rodata`, `.eh_frame` |
//! | Read-write    | `PT_LOAD`      | R+W          | `.init_array`, `.fini_array`, `.dynamic`, `.got`, `.got.plt`, `.data`, `.bss` |
//! | Dynamic       | `PT_DYNAMIC`   | R+W          | `.dynamic`                                    |
//! | Stack         | `PT_GNU_STACK` | R+W (no X)   | *(none — NX stack enforcement)*               |
//! | RELRO         | `PT_GNU_RELRO` | R            | `.dynamic`, `.got`                            |
//! | Program hdrs  | `PT_PHDR`      | R            | *(program header table itself)*               |
//! | Interpreter   | `PT_INTERP`    | R            | `.interp`                                     |
//!
//! ## Zero-Dependency Mandate
//!
//! No external crates. Only `std` and `crate::` references.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// Import ELF constants from elf_writer_common
use crate::backend::elf_writer_common::{
    PT_DYNAMIC, PT_GNU_RELRO, PT_GNU_STACK, PT_INTERP, PT_LOAD, PT_PHDR, SHF_ALLOC, SHF_EXECINSTR,
    SHF_WRITE, SHT_DYNAMIC, SHT_DYNSYM, SHT_GNU_HASH, SHT_NOBITS, SHT_NOTE, SHT_PROGBITS, SHT_RELA,
    SHT_STRTAB,
};

// Import SHT_INIT_ARRAY and SHT_FINI_ARRAY from section_merger
// (these are not exported by elf_writer_common)
use crate::backend::linker_common::section_merger::{SHT_FINI_ARRAY, SHT_INIT_ARRAY};

// ===========================================================================
// Segment Permission Flags (p_flags)
// ===========================================================================

/// Execute permission flag for ELF program headers.
pub const PF_X: u32 = 0x1;
/// Write permission flag for ELF program headers.
pub const PF_W: u32 = 0x2;
/// Read permission flag for ELF program headers.
pub const PF_R: u32 = 0x4;

// ===========================================================================
// Page Alignment
// ===========================================================================

/// Default page alignment for all four supported architectures (4 KiB).
///
/// Used for segment boundary alignment in the ELF program header table.
/// All four BCC targets (x86-64, i686, AArch64, RISC-V 64) use 4096-byte
/// pages as the default granularity.
pub const PAGE_ALIGNMENT: u64 = 0x1000; // 4096 bytes

// ===========================================================================
// Utility — Address Alignment
// ===========================================================================

/// Round `value` up to the next multiple of `alignment`.
///
/// If `alignment` is zero or one, `value` is returned unchanged.
/// `alignment` **must** be a power of two for correct results.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(align_up(7, 4), 8);
/// assert_eq!(align_up(8, 4), 8);
/// assert_eq!(align_up(0, 4096), 0);
/// ```
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ===========================================================================
// Default Base Address
// ===========================================================================

/// Return the default virtual base address for the given target and output type.
///
/// Shared objects (`is_shared = true`) are position-independent and always
/// use base address 0.  Static executables use architecture-conventional
/// base addresses that match the system defaults.
///
/// | Target   | Base Address  |
/// |----------|---------------|
/// | x86-64   | `0x0040_0000` |
/// | i686     | `0x0804_8000` |
/// | AArch64  | `0x0040_0000` |
/// | RISC-V64 | `0x0001_0000` |
pub fn default_base_address(target: &Target, is_shared: bool) -> u64 {
    if is_shared {
        return 0; // shared objects are PIC, base = 0
    }
    match target {
        Target::X86_64 => 0x0040_0000,
        Target::I686 => 0x0804_8000,
        Target::AArch64 => 0x0040_0000,
        Target::RiscV64 => 0x0001_0000,
    }
}

// ===========================================================================
// Data Structures — Public API Types
// ===========================================================================

/// Information about a single input section provided to the layout engine.
///
/// The linker driver populates one `InputSectionInfo` per merged output
/// section (or per input section if no merging has occurred yet) before
/// calling [`DefaultLinkerScript::compute_layout`] or [`create_default_layout`].
#[derive(Debug, Clone)]
pub struct InputSectionInfo {
    /// Section name (e.g. `.text`, `.rodata`, `.bss`).
    pub name: String,
    /// Section size in bytes.
    pub size: u64,
    /// Required alignment in bytes (must be a power of two, or 0/1 for none).
    pub alignment: u64,
    /// Section attribute flags — a bitmask of `SHF_*` constants stored as `u32`.
    pub flags: u32,
}

/// Mapping from section names/flags to an ELF output segment.
///
/// Describes which output sections belong to a particular program header
/// entry, along with the permission flags and alignment for that segment.
#[derive(Debug, Clone)]
pub struct SegmentMapping {
    /// Segment type (`PT_LOAD`, `PT_DYNAMIC`, `PT_INTERP`, etc.).
    pub segment_type: u32,
    /// Permission flags (`PF_R`, `PF_W`, `PF_X` combinations).
    pub flags: u32,
    /// Ordered list of section names assigned to this segment.
    pub sections: Vec<String>,
    /// Segment alignment (typically [`PAGE_ALIGNMENT`]).
    pub alignment: u64,
}

/// Definition of an output section in the default linker script.
///
/// Each entry specifies the section's name, ELF type and flags, alignment,
/// and input section name patterns that should be merged into it.
#[derive(Debug, Clone)]
pub struct OutputSection {
    /// Section name (e.g. `.text`).
    pub name: String,
    /// Virtual address (filled during layout; initially 0).
    pub address: u64,
    /// Required alignment in bytes.
    pub alignment: u64,
    /// Section attribute flags — a bitmask of `SHF_*` constants as `u32`.
    pub flags: u32,
    /// Section type (`SHT_PROGBITS`, `SHT_NOBITS`, etc.).
    pub section_type: u32,
    /// Input section name patterns merged into this output section.
    pub input_patterns: Vec<String>,
}

/// Definition of an output segment (program header entry).
#[derive(Debug, Clone)]
pub struct SegmentDef {
    /// Segment type (`PT_LOAD`, `PT_DYNAMIC`, etc.).
    pub seg_type: u32,
    /// Permission flags (`PF_R | PF_W | PF_X`).
    pub flags: u32,
    /// Segment alignment.
    pub alignment: u64,
    /// Names of output sections belonging to this segment.
    pub sections: Vec<String>,
}

/// Final computed layout of a single output section.
#[derive(Debug, Clone)]
pub struct SectionLayout {
    /// Section name.
    pub name: String,
    /// Assigned virtual address.
    pub virtual_address: u64,
    /// Assigned file offset.
    pub file_offset: u64,
    /// File size in bytes (0 for `.bss` / `SHT_NOBITS`).
    pub size: u64,
    /// In-memory size (may exceed `size` for `.bss`).
    pub mem_size: u64,
    /// Alignment used.
    pub alignment: u64,
}

/// Final computed layout of a single program header (segment).
#[derive(Debug, Clone)]
pub struct SegmentLayout {
    /// Segment type.
    pub seg_type: u32,
    /// Permission flags.
    pub flags: u32,
    /// File offset of the segment.
    pub offset: u64,
    /// Virtual address.
    pub vaddr: u64,
    /// Physical address (mirrors `vaddr` for standard Linux ELF).
    pub paddr: u64,
    /// File size of the segment.
    pub filesz: u64,
    /// In-memory size of the segment.
    pub memsz: u64,
    /// Segment alignment.
    pub alignment: u64,
}

/// Complete result of the address layout computation.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Per-section layout information.
    pub sections: Vec<SectionLayout>,
    /// Per-segment (program header) layout information.
    pub segments: Vec<SegmentLayout>,
    /// Resolved entry point virtual address.
    pub entry_point_address: u64,
}

// ===========================================================================
// DefaultLinkerScript — Core Engine
// ===========================================================================

/// Implements the default ELF linker script for BCC.
///
/// Constructs the canonical section ordering and segment mapping, computes
/// virtual-address and file-offset layout, and resolves the entry point.
/// Supports both `ET_EXEC` (static executables) and `ET_DYN` (shared objects).
pub struct DefaultLinkerScript {
    /// Target architecture.
    target: Target,
    /// Whether the output is a shared object (`ET_DYN`).
    is_shared: bool,
    /// Whether the output is fully static (no dynamic linking at all).
    /// Retained for future use by the full linker driver.
    #[allow(dead_code)]
    is_static: bool,
    /// Entry point symbol name (default: `_start`).
    entry_point: String,
    /// Base virtual address for the first loadable segment.
    base_address: u64,
    /// Ordered list of output sections.
    sections: Vec<OutputSection>,
    /// Segment definitions derived from the section list.
    segments: Vec<SegmentDef>,
}

impl DefaultLinkerScript {
    /// Create a new default linker script for the given target and output type.
    ///
    /// - `target`: The compilation target architecture.
    /// - `is_shared`: `true` for shared objects (`ET_DYN`), `false` for
    ///   executables (`ET_EXEC`).
    ///
    /// The constructor determines whether dynamic linking sections are needed
    /// based on `is_shared` — shared objects always include dynamic sections,
    /// while executables only include them if `is_shared` is true (which
    /// represents a PIE or dynamically-linked executable context).
    pub fn new(target: Target, is_shared: bool) -> Self {
        Self::with_dynamic(target, is_shared, is_shared)
    }

    /// Create a linker script with explicit control over dynamic linking
    /// section inclusion.  Use this when building a dynamically-linked
    /// executable (not `is_shared` but still needs `.interp`, `.plt`, etc.).
    pub fn with_dynamic(target: Target, is_shared: bool, has_dynamic: bool) -> Self {
        let entry_point = target.default_entry_point().to_string();
        let base_address = default_base_address(&target, is_shared);
        let is_static = !has_dynamic;

        let sections = Self::build_default_sections(has_dynamic);
        let segments = Self::build_segment_defs(has_dynamic, &sections);

        DefaultLinkerScript {
            target,
            is_shared,
            is_static,
            entry_point,
            base_address,
            sections,
            segments,
        }
    }

    /// Compute the complete address layout for the given input sections.
    ///
    /// Assigns virtual addresses and file offsets to every output section
    /// that has matching input, builds segment (program header) entries,
    /// and resolves the entry point address.
    ///
    /// # Section Matching
    ///
    /// Each `InputSectionInfo` is matched to an output section by name.
    /// Sections that do not match any defined output section are silently
    /// placed after all known sections in the appropriate permission group
    /// (R+X, R, or R+W) based on their flags.
    pub fn compute_layout(&mut self, input_sections: &[InputSectionInfo]) -> LayoutResult {
        // Build a map of input section name → aggregated info.
        let mut input_map: FxHashMap<String, AggregatedInput> = FxHashMap::default();
        for isec in input_sections {
            let entry = input_map
                .entry(isec.name.clone())
                .or_insert(AggregatedInput {
                    total_size: 0,
                    max_alignment: 1,
                    flags: isec.flags,
                    is_nobits: false,
                });
            entry.total_size += isec.size;
            if isec.alignment > entry.max_alignment {
                entry.max_alignment = isec.alignment;
            }
            // Check if this is a NOBITS section by looking at section type from
            // our output section definitions or by checking the name.
            if isec.name == ".bss" {
                entry.is_nobits = true;
            }
        }

        // Also mark .bss from our section defs as NOBITS.
        for sec_def in &self.sections {
            if sec_def.section_type == SHT_NOBITS {
                if let Some(agg) = input_map.get_mut(&sec_def.name) {
                    agg.is_nobits = true;
                }
            }
        }

        // Collect unknown sections that don't match any output section definition.
        let known_names: Vec<String> = self.sections.iter().map(|s| s.name.clone()).collect();
        let mut unknown_sections: Vec<String> = Vec::new();
        for name in input_map.keys() {
            if !known_names.contains(name) {
                unknown_sections.push(name.clone());
            }
        }
        // Sort unknown sections by their flags for deterministic placement.
        unknown_sections.sort();

        // Build ordered list of active sections (sections that have input data).
        let mut active_sections: Vec<ActiveSection> = Vec::new();
        for sec_def in &self.sections {
            if let Some(agg) = input_map.get(&sec_def.name) {
                let alignment = if agg.max_alignment > sec_def.alignment {
                    agg.max_alignment
                } else if sec_def.alignment > 0 {
                    sec_def.alignment
                } else {
                    1
                };
                active_sections.push(ActiveSection {
                    name: sec_def.name.clone(),
                    size: agg.total_size,
                    alignment,
                    flags: sec_def.flags,
                    section_type: sec_def.section_type,
                    is_nobits: agg.is_nobits || sec_def.section_type == SHT_NOBITS,
                });
            }
        }

        // Append unknown sections in appropriate order based on flags.
        for name in &unknown_sections {
            if let Some(agg) = input_map.get(name) {
                active_sections.push(ActiveSection {
                    name: name.clone(),
                    size: agg.total_size,
                    alignment: agg.max_alignment,
                    flags: agg.flags,
                    section_type: if agg.is_nobits {
                        SHT_NOBITS
                    } else {
                        SHT_PROGBITS
                    },
                    is_nobits: agg.is_nobits,
                });
            }
        }

        // --- Address Assignment ---
        let page_size = self.target.page_size() as u64;
        // ELF header + program header table occupy space before the first section.
        // We estimate the header size conservatively.
        let elf_header_size: u64 = if self.target.is_64bit() { 64 } else { 52 };
        let phdr_entry_size: u64 = if self.target.is_64bit() { 56 } else { 32 };
        let estimated_phdr_count = self.segments.len() as u64 + 4; // extra margin
        let headers_size = elf_header_size + phdr_entry_size * estimated_phdr_count;

        let mut current_vaddr = align_up(self.base_address + headers_size, 16);
        let mut current_file_offset = current_vaddr - self.base_address;
        // For shared objects the file offsets start at 0.
        if self.is_shared {
            current_vaddr = align_up(headers_size, 16);
            current_file_offset = current_vaddr;
        }

        let mut section_layouts: Vec<SectionLayout> = Vec::new();
        let mut prev_segment_class: u8 = 0;

        for sec in &active_sections {
            let seg_class = classify_section_flags(sec.flags);

            // Insert page-alignment padding at segment boundaries.
            // Exception: when transitioning from read-only (class 2) to
            // executable (class 1), do NOT insert a page boundary.  Pre-text
            // metadata sections (.interp, .gnu.hash, .dynsym, .dynstr,
            // .rela.plt) are read-only but belong in the same LOAD segment
            // as .plt/.text in the traditional ELF layout.  Inserting a page
            // gap here would place them on separate pages and create
            // overlapping LOAD segments.
            let needs_page_boundary = seg_class != prev_segment_class
                && prev_segment_class != 0
                && seg_class != 0
                && !(prev_segment_class == 2 && seg_class == 1);
            if needs_page_boundary {
                current_vaddr = align_up(current_vaddr, page_size);
                if !sec.is_nobits {
                    current_file_offset = align_up(current_file_offset, page_size);
                }
            }
            prev_segment_class = seg_class;

            // Align to section alignment.
            let sec_align = if sec.alignment > 0 { sec.alignment } else { 1 };
            current_vaddr = align_up(current_vaddr, sec_align);
            if !sec.is_nobits {
                current_file_offset = align_up(current_file_offset, sec_align);
            }

            let file_size = if sec.is_nobits { 0 } else { sec.size };
            let mem_size = sec.size;

            section_layouts.push(SectionLayout {
                name: sec.name.clone(),
                virtual_address: current_vaddr,
                file_offset: current_file_offset,
                size: file_size,
                mem_size,
                alignment: sec_align,
            });

            current_vaddr += mem_size;
            if !sec.is_nobits {
                current_file_offset += file_size;
            }
        }

        // --- Ensure all allocatable sections are covered by a segment ---
        // Unknown sections (e.g. `.data.rel.ro`, `.data.rel.ro.local`,
        // `.text.startup`) are assigned addresses above but may not appear
        // in any `SegmentDef.sections` list.  Walk every active section
        // and inject it into the matching PT_LOAD segment so that the
        // segment layout builder correctly extends the segment to cover
        // all content.
        {
            // Collect the names already present in each PT_LOAD segment.
            let mut covered: FxHashSet<String> = FxHashSet::default();
            for seg in &self.segments {
                for name in &seg.sections {
                    covered.insert(name.clone());
                }
            }

            for sec in &active_sections {
                let f = sec.flags as u64;
                // Only allocatable sections need a LOAD segment.
                if f & SHF_ALLOC == 0 {
                    continue;
                }
                if covered.contains(&sec.name) {
                    continue;
                }

                // Determine which PT_LOAD segment this section belongs to.
                let target_flags: u32 = if f & SHF_EXECINSTR != 0 {
                    PF_R | PF_X
                } else if f & SHF_WRITE != 0 {
                    PF_R | PF_W
                } else {
                    PF_R
                };

                // Find the matching PT_LOAD segment and append.
                let mut found = false;
                for seg in &mut self.segments {
                    if seg.seg_type == PT_LOAD && seg.flags == target_flags {
                        seg.sections.push(sec.name.clone());
                        found = true;
                        break;
                    }
                }

                // If no matching segment exists, create one.
                if !found {
                    self.segments.push(SegmentDef {
                        seg_type: PT_LOAD,
                        flags: target_flags,
                        alignment: PAGE_ALIGNMENT,
                        sections: vec![sec.name.clone()],
                    });
                }
            }
        }

        // --- Build Segment Layouts ---
        let segment_layouts = self.build_segment_layouts(&section_layouts, page_size);

        // --- Resolve entry point ---
        // For the layout result, we set entry_point_address to 0 initially.
        // The actual address is resolved later via resolve_entry_point().
        // However, if .text exists, we can use its start address as a reasonable default.
        let mut entry_addr = 0u64;
        for sl in &section_layouts {
            if sl.name == ".text" {
                entry_addr = sl.virtual_address;
                break;
            }
        }

        LayoutResult {
            sections: section_layouts,
            segments: segment_layouts,
            entry_point_address: entry_addr,
        }
    }

    /// Resolve the entry point symbol to its virtual address.
    ///
    /// For executables, looks up `_start` in the symbol map and returns
    /// its address.  For shared objects, the entry point is 0 (or an
    /// optional init function address if specified).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the entry point symbol is not found and the output
    /// is a static executable (where `_start` must be defined).
    pub fn resolve_entry_point(&self, symbols: &FxHashMap<String, u64>) -> Result<u64, String> {
        if self.is_shared {
            // Shared objects may have no explicit entry point; return 0 or
            // an init function if the entry point symbol is defined.
            if let Some(&addr) = symbols.get(&self.entry_point) {
                return Ok(addr);
            }
            return Ok(0);
        }

        // For executables, _start MUST be defined.
        match symbols.get(&self.entry_point) {
            Some(&addr) => Ok(addr),
            None => Err(format!(
                "linker error: entry point symbol '{}' is undefined; \
                 static executables require a defined '{}' symbol",
                self.entry_point, self.entry_point
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Internal — Section Definition Builder
    // -----------------------------------------------------------------------

    /// Build the default ordered list of output sections.
    ///
    /// The ordering follows the standard ELF layout used by GNU ld.
    /// Sections that are only relevant for dynamically-linked output are
    /// conditionally included based on `has_dynamic`.
    fn build_default_sections(has_dynamic: bool) -> Vec<OutputSection> {
        let shf_a = SHF_ALLOC as u32;
        let shf_ax = (SHF_ALLOC | SHF_EXECINSTR) as u32;
        let shf_aw = (SHF_ALLOC | SHF_WRITE) as u32;

        let mut sections = Vec::new();

        // 1. .interp — only if dynamically linked
        if has_dynamic {
            sections.push(OutputSection {
                name: ".interp".to_string(),
                address: 0,
                alignment: 1,
                flags: shf_a,
                section_type: SHT_PROGBITS,
                input_patterns: vec![".interp".to_string()],
            });
        }

        // 2. .note.* sections
        sections.push(OutputSection {
            name: ".note".to_string(),
            address: 0,
            alignment: 4,
            flags: shf_a,
            section_type: SHT_NOTE,
            input_patterns: vec![".note.*".to_string(), ".note".to_string()],
        });

        // 3–6. Dynamic linking metadata sections
        if has_dynamic {
            sections.push(OutputSection {
                name: ".gnu.hash".to_string(),
                address: 0,
                alignment: 8,
                flags: shf_a,
                section_type: SHT_GNU_HASH,
                input_patterns: vec![".gnu.hash".to_string()],
            });
            sections.push(OutputSection {
                name: ".dynsym".to_string(),
                address: 0,
                alignment: 8,
                flags: shf_a,
                section_type: SHT_DYNSYM,
                input_patterns: vec![".dynsym".to_string()],
            });
            sections.push(OutputSection {
                name: ".dynstr".to_string(),
                address: 0,
                alignment: 1,
                flags: shf_a,
                section_type: SHT_STRTAB,
                input_patterns: vec![".dynstr".to_string()],
            });
            sections.push(OutputSection {
                name: ".rela.dyn".to_string(),
                address: 0,
                alignment: 8,
                flags: shf_a,
                section_type: SHT_RELA,
                input_patterns: vec![".rela.dyn".to_string()],
            });
            sections.push(OutputSection {
                name: ".rela.plt".to_string(),
                address: 0,
                alignment: 8,
                flags: shf_a,
                section_type: SHT_RELA,
                input_patterns: vec![".rela.plt".to_string()],
            });
        }

        // 7. .init (executable code)
        sections.push(OutputSection {
            name: ".init".to_string(),
            address: 0,
            alignment: 4,
            flags: shf_ax,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".init".to_string()],
        });

        // 8. .plt — only if dynamically linked
        if has_dynamic {
            sections.push(OutputSection {
                name: ".plt".to_string(),
                address: 0,
                alignment: 16,
                flags: shf_ax,
                section_type: SHT_PROGBITS,
                input_patterns: vec![".plt".to_string()],
            });
        }

        // 9. .text (main code section)
        sections.push(OutputSection {
            name: ".text".to_string(),
            address: 0,
            alignment: 16,
            flags: shf_ax,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".text".to_string(), ".text.*".to_string()],
        });

        // 10. .fini
        sections.push(OutputSection {
            name: ".fini".to_string(),
            address: 0,
            alignment: 4,
            flags: shf_ax,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".fini".to_string()],
        });

        // 11. .rodata (read-only data)
        sections.push(OutputSection {
            name: ".rodata".to_string(),
            address: 0,
            alignment: 16,
            flags: shf_a,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".rodata".to_string(), ".rodata.*".to_string()],
        });

        // 12. .eh_frame
        sections.push(OutputSection {
            name: ".eh_frame".to_string(),
            address: 0,
            alignment: 8,
            flags: shf_a,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".eh_frame".to_string()],
        });

        // 13. .init_array (constructor pointers)
        sections.push(OutputSection {
            name: ".init_array".to_string(),
            address: 0,
            alignment: 8,
            flags: shf_aw,
            section_type: SHT_INIT_ARRAY,
            input_patterns: vec![".init_array".to_string()],
        });

        // 14. .fini_array (destructor pointers)
        sections.push(OutputSection {
            name: ".fini_array".to_string(),
            address: 0,
            alignment: 8,
            flags: shf_aw,
            section_type: SHT_FINI_ARRAY,
            input_patterns: vec![".fini_array".to_string()],
        });

        // 15. .dynamic — only if dynamically linked
        if has_dynamic {
            sections.push(OutputSection {
                name: ".dynamic".to_string(),
                address: 0,
                alignment: 8,
                flags: shf_aw,
                section_type: SHT_DYNAMIC,
                input_patterns: vec![".dynamic".to_string()],
            });
        }

        // 16. .got
        sections.push(OutputSection {
            name: ".got".to_string(),
            address: 0,
            alignment: 8,
            flags: shf_aw,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".got".to_string()],
        });

        // 17. .got.plt
        sections.push(OutputSection {
            name: ".got.plt".to_string(),
            address: 0,
            alignment: 8,
            flags: shf_aw,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".got.plt".to_string()],
        });

        // 18. .data
        sections.push(OutputSection {
            name: ".data".to_string(),
            address: 0,
            alignment: 16,
            flags: shf_aw,
            section_type: SHT_PROGBITS,
            input_patterns: vec![".data".to_string(), ".data.*".to_string()],
        });

        // 19. .bss (zero-initialized, no file data)
        sections.push(OutputSection {
            name: ".bss".to_string(),
            address: 0,
            alignment: 16,
            flags: shf_aw,
            section_type: SHT_NOBITS,
            input_patterns: vec![".bss".to_string(), ".bss.*".to_string()],
        });

        sections
    }

    // -----------------------------------------------------------------------
    // Internal — Segment Definition Builder
    // -----------------------------------------------------------------------

    /// Build segment definitions from the ordered section list.
    ///
    /// Groups sections into segments by their permission class and adds
    /// special segments (`PT_PHDR`, `PT_INTERP`, `PT_DYNAMIC`,
    /// `PT_GNU_STACK`, `PT_GNU_RELRO`).
    fn build_segment_defs(has_dynamic: bool, sections: &[OutputSection]) -> Vec<SegmentDef> {
        let mut segments = Vec::new();

        // PT_PHDR — program header table
        segments.push(SegmentDef {
            seg_type: PT_PHDR,
            flags: PF_R,
            alignment: 8,
            sections: Vec::new(), // no sections; points to the phdr table itself
        });

        // PT_INTERP — dynamic linker path
        if has_dynamic {
            segments.push(SegmentDef {
                seg_type: PT_INTERP,
                flags: PF_R,
                alignment: 1,
                sections: vec![".interp".to_string()],
            });
        }

        // Dynamic-linker metadata sections that precede .text in the
        // standard ELF layout.  Although these sections are read-only
        // (no SHF_EXECINSTR), they must be placed in the first LOAD
        // segment together with .plt/.text to avoid overlapping LOAD
        // segments.  Real linkers (GNU ld, lld) do the same.
        const FIRST_LOAD_METADATA: &[&str] = &[
            ".interp",
            ".note",
            ".gnu.hash",
            ".dynsym",
            ".dynstr",
            ".rela.dyn",
            ".rela.plt",
        ];

        // PT_LOAD — first LOAD: executable code + pre-text metadata (R+X)
        let mut rx_sections: Vec<String> = Vec::new();
        for sec in sections {
            let f = sec.flags as u64;
            let is_alloc = f & SHF_ALLOC != 0;
            let is_exec = f & SHF_EXECINSTR != 0;
            let is_write = f & SHF_WRITE != 0;
            let is_first_load_meta =
                has_dynamic && FIRST_LOAD_METADATA.contains(&sec.name.as_str());

            if is_alloc && !is_write && (is_exec || is_first_load_meta) {
                rx_sections.push(sec.name.clone());
            }
        }
        if !rx_sections.is_empty() {
            segments.push(SegmentDef {
                seg_type: PT_LOAD,
                flags: PF_R | PF_X,
                alignment: PAGE_ALIGNMENT,
                sections: rx_sections.clone(),
            });
        }

        // PT_LOAD — read-only data (R) — sections NOT already in R+X
        let rx_set: FxHashSet<String> = rx_sections.into_iter().collect();
        let mut ro_sections: Vec<String> = Vec::new();
        for sec in sections {
            let f = sec.flags as u64;
            if f & SHF_ALLOC != 0
                && f & SHF_EXECINSTR == 0
                && f & SHF_WRITE == 0
                && !rx_set.contains(&sec.name)
            {
                ro_sections.push(sec.name.clone());
            }
        }
        if !ro_sections.is_empty() {
            segments.push(SegmentDef {
                seg_type: PT_LOAD,
                flags: PF_R,
                alignment: PAGE_ALIGNMENT,
                sections: ro_sections,
            });
        }

        // PT_LOAD — read-write data (R+W)
        let mut rw_sections: Vec<String> = Vec::new();
        for sec in sections {
            let f = sec.flags as u64;
            if f & SHF_ALLOC != 0 && f & SHF_WRITE != 0 {
                rw_sections.push(sec.name.clone());
            }
        }
        if !rw_sections.is_empty() {
            segments.push(SegmentDef {
                seg_type: PT_LOAD,
                flags: PF_R | PF_W,
                alignment: PAGE_ALIGNMENT,
                sections: rw_sections,
            });
        }

        // PT_DYNAMIC — dynamic linking section
        if has_dynamic {
            segments.push(SegmentDef {
                seg_type: PT_DYNAMIC,
                flags: PF_R | PF_W,
                alignment: 8,
                sections: vec![".dynamic".to_string()],
            });
        }

        // PT_GNU_STACK — NX stack enforcement (no PF_X!)
        segments.push(SegmentDef {
            seg_type: PT_GNU_STACK,
            flags: PF_R | PF_W, // explicitly NO PF_X
            alignment: PAGE_ALIGNMENT,
            sections: Vec::new(), // virtual only, no sections
        });

        // PT_GNU_RELRO — read-only after relocation
        // Protects .dynamic and .got (but NOT .got.plt for lazy binding)
        if has_dynamic {
            let mut relro_sections = Vec::new();
            for sec in sections {
                if sec.name == ".dynamic" || sec.name == ".got" {
                    relro_sections.push(sec.name.clone());
                }
            }
            if !relro_sections.is_empty() {
                segments.push(SegmentDef {
                    seg_type: PT_GNU_RELRO,
                    flags: PF_R,
                    alignment: PAGE_ALIGNMENT,
                    sections: relro_sections,
                });
            }
        }

        segments
    }

    // -----------------------------------------------------------------------
    // Internal — Segment Layout Builder
    // -----------------------------------------------------------------------

    /// Build program header entries from section layouts and segment definitions.
    fn build_segment_layouts(
        &self,
        section_layouts: &[SectionLayout],
        page_size: u64,
    ) -> Vec<SegmentLayout> {
        // Build a quick lookup: section name → SectionLayout
        let sec_map: FxHashMap<String, &SectionLayout> = section_layouts
            .iter()
            .map(|sl| (sl.name.clone(), sl))
            .collect();

        let mut segment_layouts = Vec::new();

        for seg_def in &self.segments {
            match seg_def.seg_type {
                // PT_PHDR — program header table
                t if t == PT_PHDR => {
                    let phdr_entry_size: u64 = if self.target.is_64bit() { 56 } else { 32 };
                    let elf_header_size: u64 = if self.target.is_64bit() { 64 } else { 52 };
                    let phdr_count = self.segments.len() as u64;
                    let phdr_size = phdr_count * phdr_entry_size;

                    segment_layouts.push(SegmentLayout {
                        seg_type: PT_PHDR,
                        flags: PF_R,
                        offset: elf_header_size,
                        vaddr: self.base_address + elf_header_size,
                        paddr: self.base_address + elf_header_size,
                        filesz: phdr_size,
                        memsz: phdr_size,
                        alignment: seg_def.alignment,
                    });
                }

                // PT_GNU_STACK — virtual segment, no file data
                t if t == PT_GNU_STACK => {
                    segment_layouts.push(SegmentLayout {
                        seg_type: PT_GNU_STACK,
                        flags: PF_R | PF_W, // NO PF_X — NX stack
                        offset: 0,
                        vaddr: 0,
                        paddr: 0,
                        filesz: 0,
                        memsz: 0,
                        alignment: page_size,
                    });
                }

                // All other segments (PT_LOAD, PT_DYNAMIC, PT_INTERP, PT_GNU_RELRO)
                _ => {
                    if seg_def.sections.is_empty() {
                        // Segment with no sections — skip unless it's a special type
                        // that we already handled above.
                        continue;
                    }

                    // Find the first and last sections in this segment.
                    let mut first_offset = u64::MAX;
                    let mut first_vaddr = u64::MAX;
                    let mut last_end_file: u64 = 0;
                    let mut last_end_mem: u64 = 0;

                    for sec_name in &seg_def.sections {
                        if let Some(sl) = sec_map.get(sec_name) {
                            if sl.virtual_address < first_vaddr {
                                first_vaddr = sl.virtual_address;
                                first_offset = sl.file_offset;
                            }
                            let sec_file_end = sl.file_offset + sl.size;
                            let sec_mem_end = sl.virtual_address + sl.mem_size;
                            if sec_file_end > last_end_file {
                                last_end_file = sec_file_end;
                            }
                            if sec_mem_end > last_end_mem {
                                last_end_mem = sec_mem_end;
                            }
                        }
                    }

                    if first_vaddr == u64::MAX {
                        // None of the referenced sections exist in the layout.
                        continue;
                    }

                    // For the FIRST PT_LOAD segment in an executable, extend
                    // it backwards to cover the ELF header and program header
                    // table (file offset 0, vaddr = base_address).  This is
                    // standard ELF convention and ensures the PHDR segment is
                    // covered by a LOAD segment, satisfying the kernel loader.
                    let is_first_load = seg_def.seg_type == PT_LOAD
                        && !segment_layouts.iter().any(|s| s.seg_type == PT_LOAD);

                    let (seg_offset, seg_vaddr) = if is_first_load {
                        // Extend to cover from the start of the file
                        // (ELF header + program header table).  This ensures
                        // the PT_PHDR segment is covered by a PT_LOAD
                        // segment, satisfying the ELF specification and
                        // the kernel/dynamic linker loader.  For shared
                        // objects (base_address == 0) we still extend to
                        // offset 0 so the PHDR is covered.
                        (0u64, self.base_address)
                    } else {
                        (first_offset, first_vaddr)
                    };

                    let total_filesz = last_end_file.saturating_sub(seg_offset);
                    let total_memsz = last_end_mem.saturating_sub(seg_vaddr);

                    let seg_alignment = if seg_def.seg_type == PT_LOAD {
                        page_size
                    } else {
                        seg_def.alignment
                    };

                    segment_layouts.push(SegmentLayout {
                        seg_type: seg_def.seg_type,
                        flags: seg_def.flags,
                        offset: seg_offset,
                        vaddr: seg_vaddr,
                        paddr: seg_vaddr, // paddr mirrors vaddr on Linux
                        filesz: total_filesz,
                        memsz: total_memsz,
                        alignment: seg_alignment,
                    });
                }
            }
        }

        segment_layouts
    }
}

// ===========================================================================
// Classify Section Flags for Segment Boundary Detection
// ===========================================================================

/// Classify section flags into a segment class for page-alignment boundary
/// detection during address assignment.
///
/// Returns:
/// - `1` for R+X (executable code)
/// - `2` for R (read-only data)
/// - `3` for R+W (writable data, including BSS)
/// - `0` for non-allocatable sections
fn classify_section_flags(flags: u32) -> u8 {
    let f = flags as u64;
    if f & SHF_ALLOC == 0 {
        return 0; // Non-allocatable
    }
    if f & SHF_EXECINSTR != 0 {
        return 1; // Executable (R+X)
    }
    if f & SHF_WRITE != 0 {
        return 3; // Writable (R+W)
    }
    2 // Read-only (R)
}

// ===========================================================================
// Internal Helper Types
// ===========================================================================

/// Aggregated input section information used during layout computation.
struct AggregatedInput {
    total_size: u64,
    max_alignment: u64,
    flags: u32,
    is_nobits: bool,
}

/// A section that has been matched and is ready for address assignment.
struct ActiveSection {
    name: String,
    size: u64,
    alignment: u64,
    flags: u32,
    /// Retained for ELF section header generation by the linker driver.
    #[allow(dead_code)]
    section_type: u32,
    is_nobits: bool,
}

// ===========================================================================
// Public Convenience API
// ===========================================================================

/// Create the default ELF layout for the given input sections.
///
/// This is the primary entry point for callers that want a one-shot layout
/// computation without constructing a [`DefaultLinkerScript`] manually.
///
/// # Parameters
///
/// - `target`: Target architecture.
/// - `is_shared`: `true` for shared objects (ET_DYN), `false` for executables (ET_EXEC).
/// - `input_sections`: Merged input section information from the section merger.
/// - `has_dynamic`: `true` if dynamic linking sections should be included
///   (e.g. when linking against shared libraries or producing an ET_DYN).
///
/// # Returns
///
/// A [`LayoutResult`] containing per-section addresses, per-segment program
/// header entries, and the entry point address.
pub fn create_default_layout(
    target: &Target,
    is_shared: bool,
    input_sections: &[InputSectionInfo],
    has_dynamic: bool,
) -> LayoutResult {
    // When has_dynamic is true but is_shared is false, we treat this as a
    // dynamically-linked executable — dynamic sections are included.
    let effective_shared = is_shared || has_dynamic;

    let mut script = DefaultLinkerScript {
        target: *target,
        is_shared,
        is_static: !effective_shared,
        entry_point: target.default_entry_point().to_string(),
        base_address: default_base_address(target, is_shared),
        sections: DefaultLinkerScript::build_default_sections(effective_shared),
        segments: Vec::new(), // rebuilt below
    };
    script.segments = DefaultLinkerScript::build_segment_defs(effective_shared, &script.sections);

    script.compute_layout(input_sections)
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Default base address tests ------------------------------------------

    #[test]
    fn test_base_address_x86_64_exec() {
        assert_eq!(default_base_address(&Target::X86_64, false), 0x0040_0000);
    }

    #[test]
    fn test_base_address_i686_exec() {
        assert_eq!(default_base_address(&Target::I686, false), 0x0804_8000);
    }

    #[test]
    fn test_base_address_aarch64_exec() {
        assert_eq!(default_base_address(&Target::AArch64, false), 0x0040_0000);
    }

    #[test]
    fn test_base_address_riscv64_exec() {
        assert_eq!(default_base_address(&Target::RiscV64, false), 0x0001_0000);
    }

    #[test]
    fn test_base_address_shared_is_zero() {
        assert_eq!(default_base_address(&Target::X86_64, true), 0);
        assert_eq!(default_base_address(&Target::I686, true), 0);
        assert_eq!(default_base_address(&Target::AArch64, true), 0);
        assert_eq!(default_base_address(&Target::RiscV64, true), 0);
    }

    // -- Page alignment constant ---------------------------------------------

    #[test]
    fn test_page_alignment() {
        assert_eq!(PAGE_ALIGNMENT, 0x1000);
        assert_eq!(PAGE_ALIGNMENT, 4096);
    }

    // -- PF flag constants ---------------------------------------------------

    #[test]
    fn test_pf_flags() {
        assert_eq!(PF_X, 0x1);
        assert_eq!(PF_W, 0x2);
        assert_eq!(PF_R, 0x4);
    }

    // -- align_up utility ----------------------------------------------------

    #[test]
    fn test_align_up_basic() {
        assert_eq!(align_up(7, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(5, 0), 5);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }

    // -- DefaultLinkerScript construction ------------------------------------

    #[test]
    fn test_new_static_exec() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        assert!(!script.is_shared);
        assert!(script.is_static);
        assert_eq!(script.entry_point, "_start");
        assert_eq!(script.base_address, 0x0040_0000);
        // Static executable should NOT have .interp, .dynsym, etc.
        let section_names: Vec<&str> = script.sections.iter().map(|s| s.name.as_str()).collect();
        assert!(!section_names.contains(&".interp"));
        assert!(!section_names.contains(&".dynsym"));
        // Must have .text, .rodata, .data, .bss
        assert!(section_names.contains(&".text"));
        assert!(section_names.contains(&".rodata"));
        assert!(section_names.contains(&".data"));
        assert!(section_names.contains(&".bss"));
    }

    #[test]
    fn test_new_shared_object() {
        let script = DefaultLinkerScript::new(Target::X86_64, true);
        assert!(script.is_shared);
        assert!(!script.is_static);
        assert_eq!(script.base_address, 0);
        // Shared object should have dynamic sections
        let section_names: Vec<&str> = script.sections.iter().map(|s| s.name.as_str()).collect();
        assert!(section_names.contains(&".interp"));
        assert!(section_names.contains(&".dynsym"));
        assert!(section_names.contains(&".dynstr"));
        assert!(section_names.contains(&".gnu.hash"));
        assert!(section_names.contains(&".plt"));
        assert!(section_names.contains(&".dynamic"));
    }

    // -- Section ordering: .text before .rodata before .data before .bss -----

    #[test]
    fn test_section_ordering() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let names: Vec<&str> = script.sections.iter().map(|s| s.name.as_str()).collect();
        let text_idx = names.iter().position(|&n| n == ".text").unwrap();
        let rodata_idx = names.iter().position(|&n| n == ".rodata").unwrap();
        let data_idx = names.iter().position(|&n| n == ".data").unwrap();
        let bss_idx = names.iter().position(|&n| n == ".bss").unwrap();
        assert!(text_idx < rodata_idx, ".text must come before .rodata");
        assert!(rodata_idx < data_idx, ".rodata must come before .data");
        assert!(data_idx < bss_idx, ".data must come before .bss");
    }

    // -- Segment flags -------------------------------------------------------

    #[test]
    fn test_text_segment_rx() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let rx_seg = script
            .segments
            .iter()
            .find(|s| s.seg_type == PT_LOAD && s.flags == (PF_R | PF_X))
            .expect("must have R+X PT_LOAD segment");
        assert!(rx_seg.sections.contains(&".text".to_string()));
    }

    #[test]
    fn test_rodata_segment_r() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let r_seg = script
            .segments
            .iter()
            .find(|s| s.seg_type == PT_LOAD && s.flags == PF_R)
            .expect("must have R-only PT_LOAD segment");
        assert!(r_seg.sections.contains(&".rodata".to_string()));
    }

    #[test]
    fn test_data_bss_segment_rw() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let rw_seg = script
            .segments
            .iter()
            .find(|s| s.seg_type == PT_LOAD && s.flags == (PF_R | PF_W))
            .expect("must have R+W PT_LOAD segment");
        assert!(rw_seg.sections.contains(&".data".to_string()));
        assert!(rw_seg.sections.contains(&".bss".to_string()));
    }

    #[test]
    fn test_gnu_stack_no_execute() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let stack_seg = script
            .segments
            .iter()
            .find(|s| s.seg_type == PT_GNU_STACK)
            .expect("must have PT_GNU_STACK");
        assert_eq!(stack_seg.flags & PF_X, 0, "PT_GNU_STACK must NOT have PF_X");
        assert_ne!(stack_seg.flags & PF_R, 0, "PT_GNU_STACK must have PF_R");
        assert_ne!(stack_seg.flags & PF_W, 0, "PT_GNU_STACK must have PF_W");
    }

    // -- Entry point resolution ----------------------------------------------

    #[test]
    fn test_resolve_entry_point_exec() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let mut symbols: FxHashMap<String, u64> = FxHashMap::default();
        symbols.insert("_start".to_string(), 0x401000);
        let result = script.resolve_entry_point(&symbols);
        assert_eq!(result, Ok(0x401000));
    }

    #[test]
    fn test_resolve_entry_point_missing() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let symbols: FxHashMap<String, u64> = FxHashMap::default();
        let result = script.resolve_entry_point(&symbols);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_entry_point_shared_missing_ok() {
        let script = DefaultLinkerScript::new(Target::X86_64, true);
        let symbols: FxHashMap<String, u64> = FxHashMap::default();
        let result = script.resolve_entry_point(&symbols);
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn test_resolve_entry_point_shared_present() {
        let script = DefaultLinkerScript::new(Target::X86_64, true);
        let mut symbols: FxHashMap<String, u64> = FxHashMap::default();
        symbols.insert("_start".to_string(), 0x1000);
        let result = script.resolve_entry_point(&symbols);
        assert_eq!(result, Ok(0x1000));
    }

    // -- Layout computation --------------------------------------------------

    #[test]
    fn test_compute_layout_basic() {
        let mut script = DefaultLinkerScript::new(Target::X86_64, false);
        let input = vec![
            InputSectionInfo {
                name: ".text".to_string(),
                size: 256,
                alignment: 16,
                flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
            },
            InputSectionInfo {
                name: ".rodata".to_string(),
                size: 64,
                alignment: 8,
                flags: SHF_ALLOC as u32,
            },
            InputSectionInfo {
                name: ".data".to_string(),
                size: 128,
                alignment: 8,
                flags: (SHF_ALLOC | SHF_WRITE) as u32,
            },
            InputSectionInfo {
                name: ".bss".to_string(),
                size: 512,
                alignment: 16,
                flags: (SHF_ALLOC | SHF_WRITE) as u32,
            },
        ];

        let result = script.compute_layout(&input);

        // Verify sections exist in the result
        let sec_names: Vec<&str> = result.sections.iter().map(|s| s.name.as_str()).collect();
        assert!(sec_names.contains(&".text"));
        assert!(sec_names.contains(&".rodata"));
        assert!(sec_names.contains(&".data"));
        assert!(sec_names.contains(&".bss"));

        // Verify .bss has size 0 in file but mem_size > 0
        let bss = result.sections.iter().find(|s| s.name == ".bss").unwrap();
        assert_eq!(bss.size, 0, ".bss must have file size 0 (SHT_NOBITS)");
        assert_eq!(bss.mem_size, 512, ".bss must have mem_size 512");

        // Verify section ordering by virtual address
        let text_addr = result
            .sections
            .iter()
            .find(|s| s.name == ".text")
            .unwrap()
            .virtual_address;
        let rodata_addr = result
            .sections
            .iter()
            .find(|s| s.name == ".rodata")
            .unwrap()
            .virtual_address;
        let data_addr = result
            .sections
            .iter()
            .find(|s| s.name == ".data")
            .unwrap()
            .virtual_address;
        let bss_addr = bss.virtual_address;

        assert!(text_addr < rodata_addr, ".text addr < .rodata addr");
        assert!(rodata_addr < data_addr, ".rodata addr < .data addr");
        assert!(data_addr < bss_addr, ".data addr < .bss addr");

        // Verify page alignment at segment boundaries
        assert_eq!(
            rodata_addr % PAGE_ALIGNMENT,
            0,
            ".rodata must be page-aligned (new segment)"
        );
        assert_eq!(
            data_addr % PAGE_ALIGNMENT,
            0,
            ".data must be page-aligned (new segment)"
        );
    }

    #[test]
    fn test_compute_layout_shared() {
        let mut script = DefaultLinkerScript::new(Target::X86_64, true);
        let input = vec![InputSectionInfo {
            name: ".text".to_string(),
            size: 100,
            alignment: 16,
            flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
        }];

        let result = script.compute_layout(&input);
        // For shared objects, addresses should start near 0 (after headers)
        let text = result.sections.iter().find(|s| s.name == ".text").unwrap();
        assert!(
            text.virtual_address < PAGE_ALIGNMENT,
            "shared object .text should be near 0"
        );
    }

    // -- create_default_layout convenience API --------------------------------

    #[test]
    fn test_create_default_layout() {
        let input = vec![
            InputSectionInfo {
                name: ".text".to_string(),
                size: 100,
                alignment: 16,
                flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
            },
            InputSectionInfo {
                name: ".data".to_string(),
                size: 50,
                alignment: 8,
                flags: (SHF_ALLOC | SHF_WRITE) as u32,
            },
        ];
        let result = create_default_layout(&Target::X86_64, false, &input, false);
        assert!(!result.sections.is_empty());
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_create_default_layout_with_dynamic() {
        let input = vec![
            InputSectionInfo {
                name: ".text".to_string(),
                size: 100,
                alignment: 16,
                flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
            },
            InputSectionInfo {
                name: ".dynamic".to_string(),
                size: 200,
                alignment: 8,
                flags: (SHF_ALLOC | SHF_WRITE) as u32,
            },
        ];
        let result = create_default_layout(&Target::X86_64, false, &input, true);
        // Should have PT_DYNAMIC segment
        let has_dynamic = result.segments.iter().any(|s| s.seg_type == PT_DYNAMIC);
        assert!(has_dynamic, "must have PT_DYNAMIC when has_dynamic=true");
    }

    // -- Segment layout verification -----------------------------------------

    #[test]
    fn test_segment_layouts_phdr() {
        let mut script = DefaultLinkerScript::new(Target::X86_64, false);
        let input = vec![InputSectionInfo {
            name: ".text".to_string(),
            size: 64,
            alignment: 16,
            flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
        }];
        let result = script.compute_layout(&input);
        let phdr = result.segments.iter().find(|s| s.seg_type == PT_PHDR);
        assert!(phdr.is_some(), "must have PT_PHDR segment");
        let phdr = phdr.unwrap();
        assert_eq!(phdr.flags, PF_R);
        assert!(phdr.filesz > 0, "PT_PHDR must have non-zero filesz");
    }

    #[test]
    fn test_segment_layouts_gnu_stack() {
        let mut script = DefaultLinkerScript::new(Target::X86_64, false);
        let input = vec![InputSectionInfo {
            name: ".text".to_string(),
            size: 64,
            alignment: 16,
            flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
        }];
        let result = script.compute_layout(&input);
        let stack = result
            .segments
            .iter()
            .find(|s| s.seg_type == PT_GNU_STACK)
            .expect("must have PT_GNU_STACK");
        assert_eq!(stack.flags, PF_R | PF_W, "PT_GNU_STACK: R+W, no X");
        assert_eq!(stack.filesz, 0);
        assert_eq!(stack.memsz, 0);
    }

    // -- Classify section flags helper ----------------------------------------

    #[test]
    fn test_classify_section_flags() {
        assert_eq!(
            classify_section_flags((SHF_ALLOC | SHF_EXECINSTR) as u32),
            1
        ); // R+X
        assert_eq!(classify_section_flags(SHF_ALLOC as u32), 2); // R
        assert_eq!(classify_section_flags((SHF_ALLOC | SHF_WRITE) as u32), 3); // R+W
        assert_eq!(classify_section_flags(0), 0); // non-alloc
    }

    // -- All architectures produce sane layouts -------------------------------

    #[test]
    fn test_all_architectures() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let mut script = DefaultLinkerScript::new(*target, false);
            let input = vec![InputSectionInfo {
                name: ".text".to_string(),
                size: 64,
                alignment: 16,
                flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
            }];
            let result = script.compute_layout(&input);
            assert!(!result.sections.is_empty());
            assert!(!result.segments.is_empty());
            // Entry point should be non-zero for executables
            assert!(
                result.entry_point_address > 0,
                "{}: entry point must be > 0",
                target
            );
        }
    }

    // -- init_array / fini_array section types --------------------------------

    #[test]
    fn test_init_fini_array_section_types() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let init = script
            .sections
            .iter()
            .find(|s| s.name == ".init_array")
            .expect(".init_array must exist");
        assert_eq!(init.section_type, SHT_INIT_ARRAY);
        let fini = script
            .sections
            .iter()
            .find(|s| s.name == ".fini_array")
            .expect(".fini_array must exist");
        assert_eq!(fini.section_type, SHT_FINI_ARRAY);
    }

    // -- .bss section type ---------------------------------------------------

    #[test]
    fn test_bss_section_type() {
        let script = DefaultLinkerScript::new(Target::X86_64, false);
        let bss = script
            .sections
            .iter()
            .find(|s| s.name == ".bss")
            .expect(".bss must exist");
        assert_eq!(bss.section_type, SHT_NOBITS);
    }
}
