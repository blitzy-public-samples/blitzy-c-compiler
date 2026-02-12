//! Default linker script handling for the BCC built-in linker.
//!
//! Defines the mapping from ELF sections to ELF segments (program headers).
//! Implements the default section-to-segment mapping used when no external
//! linker script is provided.
//!
//! # Default Segment Layout
//!
//! | Segment         | Sections                              | Permissions |
//! |-----------------|---------------------------------------|-------------|
//! | `PT_PHDR`       | (program headers self-reference)      | R           |
//! | `PT_INTERP`     | `.interp`                             | R           |
//! | `PT_LOAD #1`    | `.text`, `.init`, `.fini`              | R + X       |
//! | `PT_LOAD #2`    | `.rodata`, `.eh_frame`                 | R           |
//! | `PT_LOAD #3`    | `.data`, `.got`, `.got.plt`, `.bss`    | R + W       |
//! | `PT_DYNAMIC`    | `.dynamic`                            | R + W       |
//! | `PT_GNU_STACK`  | (none)                                | R + W       |
//! | `PT_GNU_RELRO`  | `.dynamic`, `.got`                    | R           |
//!
//! # Architecture-Specific Base Addresses
//!
//! | Target    | Base Address  | Description                    |
//! |-----------|---------------|--------------------------------|
//! | x86-64    | `0x400000`    | 4 MiB (standard for ET_EXEC)   |
//! | i686      | `0x08048000`  | Classic Linux i386 base         |
//! | AArch64   | `0x400000`    | Standard AArch64 base           |
//! | RISC-V 64 | `0x10000`     | Standard RISC-V Linux base      |
//! | Shared    | `0x0`         | Position-independent            |
//!
//! # Standalone Backend Mode
//!
//! This module is part of BCC's fully self-contained linker—no external
//! linker is invoked. All four architecture-specific linkers depend on
//! this module for section-to-segment mapping. Removing it breaks all
//! final ELF output.
//!
//! # Zero-Dependency Implementation
//!
//! Uses only the Rust standard library and internal BCC modules, adhering
//! to the project's strict zero-dependency mandate.

use crate::backend::elf_writer_common::{
    ProgramHeader, PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_RELRO, PT_GNU_STACK, PT_INTERP,
    PT_LOAD, PT_PHDR, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS,
};
use crate::backend::linker_common::section_merger::OutputSection;
use crate::backend::linker_common::symbol_resolver::ResolvedSymbols;
use crate::common::target::Target;

// ===========================================================================
// OutputType — kind of binary being produced
// ===========================================================================

/// Kind of output binary being produced by the linker.
///
/// Controls how the linker script configures entry points, base addresses,
/// segment layout, and dynamic linking sections.
///
/// * [`Executable`](OutputType::Executable) — `ET_EXEC` with absolute
///   addresses and a `_start` entry point.
/// * [`SharedLibrary`](OutputType::SharedLibrary) — `ET_DYN` with
///   position-independent addressing and dynamic linking sections.
/// * [`RelocatableObject`](OutputType::RelocatableObject) — `ET_REL`
///   relocatable object produced by the `-c` flag (no linker script needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputType {
    /// `ET_EXEC` — static executable with absolute addresses and `_start`.
    Executable,
    /// `ET_DYN` — shared library with position-independent addressing.
    SharedLibrary,
    /// `ET_REL` — relocatable object (for `-c`, no linker script needed).
    RelocatableObject,
}

impl core::fmt::Display for OutputType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OutputType::Executable => write!(f, "executable (ET_EXEC)"),
            OutputType::SharedLibrary => write!(f, "shared library (ET_DYN)"),
            OutputType::RelocatableObject => write!(f, "relocatable object (ET_REL)"),
        }
    }
}

// ===========================================================================
// SectionAssignment — input-to-output section mapping rule
// ===========================================================================

/// A rule that maps input section name patterns to an output section.
///
/// The default linker script uses these rules to determine which output
/// section receives each input section from the object files being linked.
///
/// # Pattern Matching
///
/// Input patterns support three modes:
/// - **Exact:** `".text"` matches only `".text"`.
/// - **Suffix wildcard:** `".text.*"` matches `".text"`, `".text.startup"`, etc.
/// - **Prefix wildcard:** `".text*"` matches any section starting with `".text"`.
#[derive(Debug, Clone)]
pub struct SectionAssignment {
    /// Name of the output section (e.g., `.text`, `.data`, `.bss`).
    pub output_name: String,
    /// Input section name patterns that map to this output section.
    pub input_patterns: Vec<String>,
    /// ELF section flags (`SHF_ALLOC`, `SHF_WRITE`, `SHF_EXECINSTR`).
    pub flags: u64,
    /// Minimum alignment in bytes for the output section.
    pub alignment: u64,
}

impl SectionAssignment {
    /// Creates a new section assignment rule.
    ///
    /// # Arguments
    ///
    /// * `output_name` — Target output section name.
    /// * `input_patterns` — Patterns matching input section names.
    /// * `flags` — ELF section flags for the output section.
    /// * `alignment` — Minimum alignment for the output section.
    pub fn new(
        output_name: &str,
        input_patterns: Vec<&str>,
        flags: u64,
        alignment: u64,
    ) -> Self {
        Self {
            output_name: output_name.to_string(),
            input_patterns: input_patterns.iter().map(|s| s.to_string()).collect(),
            flags,
            alignment,
        }
    }

    /// Tests whether `section_name` matches any pattern in this rule.
    ///
    /// The three matching modes are:
    /// 1. Exact — pattern `".text"` matches only `".text"`.
    /// 2. Dot-star — pattern `".text.*"` matches `".text"` and any name
    ///    starting with `".text."`.
    /// 3. Star — pattern `".text*"` matches any name starting with `".text"`.
    pub fn matches(&self, section_name: &str) -> bool {
        for pattern in &self.input_patterns {
            if pattern.ends_with(".*") {
                // Dot-star wildcard: ".text.*" matches ".text" and ".text.foo"
                let prefix = &pattern[..pattern.len() - 2];
                if section_name == prefix
                    || (section_name.starts_with(prefix)
                        && section_name.as_bytes().get(prefix.len()) == Some(&b'.'))
                {
                    return true;
                }
            } else if pattern.ends_with('*') {
                // Prefix wildcard: ".text*" matches ".text", ".textual", etc.
                let prefix = &pattern[..pattern.len() - 1];
                if section_name.starts_with(prefix) {
                    return true;
                }
            } else if section_name == pattern.as_str() {
                // Exact match
                return true;
            }
        }
        false
    }
}

// ===========================================================================
// SegmentRule — section-to-segment membership and permissions
// ===========================================================================

/// Rule defining which output sections belong to an ELF program header
/// segment and the segment's memory protection flags.
///
/// Each rule maps a set of output section names to an ELF program header
/// with a specific type (`PT_LOAD`, `PT_DYNAMIC`, etc.) and permission
/// flags (`PF_R`, `PF_W`, `PF_X`).
#[derive(Debug, Clone)]
pub struct SegmentRule {
    /// Program header type (`PT_LOAD`, `PT_DYNAMIC`, `PT_GNU_STACK`, etc.).
    pub segment_type: u32,
    /// Segment permission flags (`PF_R`, `PF_W`, `PF_X` combinations).
    pub flags: u32,
    /// Segment alignment (typically page size for `PT_LOAD` segments).
    pub alignment: u64,
    /// Output section names assigned to this segment.
    pub sections: Vec<String>,
}

impl SegmentRule {
    /// Creates a new segment rule.
    pub fn new(
        segment_type: u32,
        flags: u32,
        alignment: u64,
        sections: Vec<String>,
    ) -> Self {
        Self {
            segment_type,
            flags,
            alignment,
            sections,
        }
    }

    /// Returns `true` if this segment contains the named output section.
    pub fn contains_section(&self, section_name: &str) -> bool {
        self.sections.iter().any(|s| s == section_name)
    }
}

// ===========================================================================
// LinkerScript — default linker script
// ===========================================================================

/// Default linker script providing section placement, segment layout, entry
/// point configuration, and architecture-specific settings for the BCC
/// built-in linker.
///
/// This replaces the need for an external linker script file. The BCC linker
/// always uses this default script — no custom linker script file is supported.
///
/// # Construction
///
/// Use [`LinkerScript::default_for_target`] to create a script configured for
/// a specific target architecture and output type:
///
/// ```ignore
/// let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
/// assert_eq!(script.entry_point(), "_start");
/// assert_eq!(script.base_address(), 0x400000);
/// ```
///
/// # Segment Layout
///
/// Call [`compute_segment_layout`](LinkerScript::compute_segment_layout) with
/// finalized output sections to generate the ELF program header table.
#[derive(Debug, Clone)]
pub struct LinkerScript {
    /// Entry point symbol name (e.g. `_start`; empty for shared libraries).
    entry_symbol: String,
    /// Base virtual address for the first loadable segment.
    base_addr: u64,
    /// Segment rules mapping output sections to program headers.
    segment_rules: Vec<SegmentRule>,
    /// Section assignment rules mapping input sections to output sections.
    section_assignments: Vec<SectionAssignment>,
    /// Memory page size for the target architecture.
    page_sz: u64,
    /// Target architecture.
    target: Target,
    /// Output binary type.
    output_type: OutputType,
}

impl LinkerScript {
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    /// Creates a default linker script for the given target architecture
    /// and output type.
    ///
    /// This is the primary constructor. It configures:
    /// - Architecture-specific base address (see [`base_address`])
    /// - Architecture-specific page size (see [`page_size`])
    /// - Default section assignment rules
    /// - Default segment mapping rules
    /// - Entry point (`_start` for executables, none for shared libraries)
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture enum value.
    /// * `output_type` — Kind of binary to produce.
    pub fn default_for_target(target: &Target, output_type: OutputType) -> Self {
        let base_addr = Self::compute_base_address(target, &output_type);
        let page_sz = Self::compute_page_size(target);

        let entry_symbol = match output_type {
            OutputType::Executable => "_start".to_string(),
            OutputType::SharedLibrary | OutputType::RelocatableObject => String::new(),
        };

        let mut script = Self {
            entry_symbol,
            base_addr,
            segment_rules: Vec::new(),
            section_assignments: Vec::new(),
            page_sz,
            target: *target,
            output_type,
        };

        script.build_default_section_assignments();
        script.build_default_segment_rules();
        script
    }

    // ---------------------------------------------------------------
    // Public accessors
    // ---------------------------------------------------------------

    /// Returns the entry point symbol name.
    ///
    /// For executables this is `"_start"`. For shared libraries and
    /// relocatable objects this returns an empty string (e_entry = 0).
    pub fn entry_point(&self) -> &str {
        &self.entry_symbol
    }

    /// Returns the base virtual address for the first loadable segment.
    ///
    /// Architecture-specific defaults:
    /// - x86-64: `0x400000` (4 MiB)
    /// - i686: `0x08048000` (classic Linux i386)
    /// - AArch64: `0x400000`
    /// - RISC-V 64: `0x10000`
    /// - Shared libraries: `0x0` (position-independent)
    pub fn base_address(&self) -> u64 {
        self.base_addr
    }

    /// Returns the memory page size for this target architecture.
    ///
    /// Default is 4096 (4 KiB) for all supported architectures.
    pub fn page_size(&self) -> u64 {
        self.page_sz
    }

    /// Returns the section assignment rules (input → output mapping).
    pub fn section_rules(&self) -> &[SectionAssignment] {
        &self.section_assignments
    }

    /// Returns the segment mapping rules (output section → program header).
    pub fn segment_mapping(&self) -> &[SegmentRule] {
        &self.segment_rules
    }

    /// Returns the target architecture.
    pub fn target(&self) -> Target {
        self.target
    }

    /// Returns the output type.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    // ---------------------------------------------------------------
    // Entry point resolution
    // ---------------------------------------------------------------

    /// Resolves the entry point symbol to its virtual address.
    ///
    /// Looks up the entry symbol (typically `_start`) in the resolved
    /// symbol table. Returns `None` if:
    /// - The output type is [`SharedLibrary`](OutputType::SharedLibrary) or
    ///   [`RelocatableObject`](OutputType::RelocatableObject)
    /// - The entry symbol is empty
    /// - The symbol is not found in the resolved symbol table
    ///
    /// # Arguments
    ///
    /// * `symbols` — The final resolved symbol table from the symbol resolver.
    pub fn resolve_entry_address(&self, symbols: &ResolvedSymbols) -> Option<u64> {
        if self.entry_symbol.is_empty() {
            return None;
        }
        match self.output_type {
            OutputType::SharedLibrary | OutputType::RelocatableObject => None,
            OutputType::Executable => symbols.get_symbol_value(&self.entry_symbol),
        }
    }

    // ---------------------------------------------------------------
    // Segment layout computation
    // ---------------------------------------------------------------

    /// Generates ELF program header entries from finalized output sections.
    ///
    /// Given a slice of [`OutputSection`]s with assigned virtual addresses
    /// and file offsets, this method produces the program header table by:
    ///
    /// 1. Iterating over segment rules in order
    /// 2. Matching each output section to its rule by name
    /// 3. Computing `p_offset`, `p_vaddr`, `p_paddr`, `p_filesz`, `p_memsz`,
    ///    and `p_align` for each segment
    /// 4. Ensuring `PT_LOAD` segments are page-aligned
    /// 5. Handling `.bss` (`SHT_NOBITS`): contributes to `p_memsz` but NOT
    ///    to `p_filesz`
    /// 6. Classifying unclaimed allocated sections by their ELF flags
    ///    (`SHF_ALLOC`, `SHF_WRITE`, `SHF_EXECINSTR`) into the best-matching
    ///    `PT_LOAD` segment
    ///
    /// # Arguments
    ///
    /// * `output_sections` — Finalized output sections with addresses and
    ///   offsets from the section merger.
    pub fn compute_segment_layout(
        &self,
        output_sections: &[OutputSection],
    ) -> Vec<ProgramHeader> {
        let mut headers: Vec<ProgramHeader> = Vec::new();
        // Track which output sections have been claimed by a segment rule.
        let mut claimed: Vec<bool> = vec![false; output_sections.len()];

        for rule in &self.segment_rules {
            match rule.segment_type {
                // ---- PT_PHDR: program header table self-reference ----
                PT_PHDR => {
                    // The linker must patch p_offset/p_filesz/p_memsz/p_vaddr
                    // after the full layout is finalised. We emit an initial
                    // entry with zeroed extents that the linker fills in.
                    headers.push(ProgramHeader {
                        p_type: PT_PHDR,
                        p_flags: rule.flags,
                        p_offset: 0,
                        p_vaddr: 0,
                        p_paddr: 0,
                        p_filesz: 0,
                        p_memsz: 0,
                        p_align: rule.alignment,
                    });
                }

                // ---- PT_GNU_STACK: non-executable stack ----
                PT_GNU_STACK => {
                    headers.push(Self::gnu_stack_header());
                }

                // ---- PT_INTERP: dynamic linker path ----
                PT_INTERP => {
                    if let Some((idx, interp)) = find_section_indexed(output_sections, ".interp") {
                        claimed[idx] = true;
                        headers.push(ProgramHeader {
                            p_type: PT_INTERP,
                            p_flags: rule.flags,
                            p_offset: interp.offset,
                            p_vaddr: interp.addr,
                            p_paddr: interp.addr,
                            p_filesz: interp.size,
                            p_memsz: interp.size,
                            p_align: rule.alignment,
                        });
                    }
                }

                // ---- PT_LOAD: loadable segments ----
                PT_LOAD => {
                    let matched = collect_matching_indexed(output_sections, &rule.sections);
                    if matched.is_empty() {
                        continue;
                    }
                    // Mark sections claimed
                    for &(idx, _) in &matched {
                        claimed[idx] = true;
                    }
                    let phdr = Self::build_load_header(rule, &matched);
                    headers.push(phdr);
                }

                // ---- PT_DYNAMIC: dynamic linking metadata ----
                PT_DYNAMIC => {
                    if let Some((idx, dynamic)) =
                        find_section_indexed(output_sections, ".dynamic")
                    {
                        claimed[idx] = true;
                        headers.push(ProgramHeader {
                            p_type: PT_DYNAMIC,
                            p_flags: rule.flags,
                            p_offset: dynamic.offset,
                            p_vaddr: dynamic.addr,
                            p_paddr: dynamic.addr,
                            p_filesz: dynamic.size,
                            p_memsz: dynamic.size,
                            p_align: rule.alignment,
                        });
                    }
                }

                // ---- PT_GNU_RELRO: read-only after relocation ----
                PT_GNU_RELRO => {
                    let matched = collect_matching_indexed(output_sections, &rule.sections);
                    if matched.is_empty() {
                        continue;
                    }
                    for &(idx, _) in &matched {
                        claimed[idx] = true;
                    }
                    let phdr = Self::build_relro_header(rule, &matched, self.page_sz);
                    headers.push(phdr);
                }

                // ---- Other / custom segment types ----
                _ => {
                    let matched = collect_matching_indexed(output_sections, &rule.sections);
                    if matched.is_empty() && rule.sections.is_empty() {
                        headers.push(ProgramHeader {
                            p_type: rule.segment_type,
                            p_flags: rule.flags,
                            p_offset: 0,
                            p_vaddr: 0,
                            p_paddr: 0,
                            p_filesz: 0,
                            p_memsz: 0,
                            p_align: rule.alignment,
                        });
                    } else if !matched.is_empty() {
                        for &(idx, _) in &matched {
                            claimed[idx] = true;
                        }
                        let phdr = Self::build_generic_header(rule, &matched);
                        headers.push(phdr);
                    }
                }
            }
        }

        // ---- Fallback: assign unclaimed allocated sections by flags ----
        self.assign_unclaimed_sections(output_sections, &claimed, &mut headers);

        headers
    }

    // ---------------------------------------------------------------
    // Static helpers
    // ---------------------------------------------------------------

    /// Creates a `PT_GNU_STACK` program header for a non-executable stack.
    ///
    /// This signals the Linux kernel that the process does **not** need an
    /// executable stack:
    /// - `p_type` = `PT_GNU_STACK`
    /// - `p_flags` = `PF_R | PF_W` (no `PF_X`)
    /// - All address/size fields = 0
    pub fn gnu_stack_header() -> ProgramHeader {
        ProgramHeader {
            p_type: PT_GNU_STACK,
            p_flags: PF_R | PF_W,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 0x10,
        }
    }

    // ---------------------------------------------------------------
    // Convenience query methods
    // ---------------------------------------------------------------

    /// Resolves an input section name to the output section it maps to.
    ///
    /// Returns `None` if no section assignment rule matches the name.
    pub fn resolve_section(&self, input_name: &str) -> Option<&str> {
        for assignment in &self.section_assignments {
            if assignment.matches(input_name) {
                return Some(&assignment.output_name);
            }
        }
        None
    }

    /// Returns the segment rule containing the given output section, if any.
    pub fn segment_for_section(&self, output_section_name: &str) -> Option<&SegmentRule> {
        self.segment_rules
            .iter()
            .find(|rule| rule.contains_section(output_section_name))
    }

    /// Returns all segment rules of the given type.
    pub fn segments_of_type(&self, segment_type: u32) -> Vec<&SegmentRule> {
        self.segment_rules
            .iter()
            .filter(|r| r.segment_type == segment_type)
            .collect()
    }

    /// Returns the total number of program headers this script defines.
    pub fn program_header_count(&self) -> usize {
        self.segment_rules.len()
    }

    /// Aligns `addr` up to the next page boundary.
    pub fn align_to_page(&self, addr: u64) -> u64 {
        align_up(addr, self.page_sz)
    }

    /// Sets a custom entry point symbol name.
    pub fn set_entry_point(&mut self, name: &str) {
        self.entry_symbol = name.to_string();
    }

    /// Sets a custom base virtual address.
    pub fn set_base_address(&mut self, addr: u64) {
        self.base_addr = addr;
    }

    // ---------------------------------------------------------------
    // Private — architecture-specific defaults
    // ---------------------------------------------------------------

    /// Computes the default base virtual address for the first loadable
    /// segment given a target architecture and output type.
    ///
    /// Shared libraries and relocatable objects always use base address 0
    /// (position-independent, relocated by the dynamic loader).
    fn compute_base_address(target: &Target, output_type: &OutputType) -> u64 {
        match output_type {
            OutputType::SharedLibrary | OutputType::RelocatableObject => 0,
            OutputType::Executable => match target {
                Target::X86_64 => 0x0040_0000,  // 4 MiB — standard Linux x86-64
                Target::I686 => 0x0804_8000,     // classic Linux i386
                Target::AArch64 => 0x0040_0000,  // standard AArch64
                Target::RiscV64 => 0x0001_0000,  // standard RISC-V Linux
            },
        }
    }

    /// Computes the default page size for the target architecture.
    ///
    /// All four supported architectures use 4096 (4 KiB) as the default
    /// page size. AArch64 optionally supports 64 KiB pages but we default
    /// to 4 KiB for maximum compatibility.
    fn compute_page_size(target: &Target) -> u64 {
        match target {
            Target::X86_64 | Target::I686 | Target::AArch64 | Target::RiscV64 => 0x1000,
        }
    }

    // ---------------------------------------------------------------
    // Private — section assignment rules
    // ---------------------------------------------------------------

    /// Populates the default section assignment rules.
    ///
    /// These rules map input section name patterns to output section names,
    /// along with the ELF section flags and minimum alignment for each
    /// output section.
    fn build_default_section_assignments(&mut self) {
        self.section_assignments = vec![
            // ---- Executable code ----
            SectionAssignment::new(
                ".text",
                vec![".text", ".text.*", ".init", ".fini"],
                SHF_ALLOC | SHF_EXECINSTR,
                16,
            ),
            SectionAssignment::new(
                ".plt",
                vec![".plt", ".plt.*"],
                SHF_ALLOC | SHF_EXECINSTR,
                16,
            ),
            // ---- Read-only data ----
            SectionAssignment::new(
                ".rodata",
                vec![".rodata", ".rodata.*"],
                SHF_ALLOC,
                16,
            ),
            SectionAssignment::new(
                ".eh_frame",
                vec![".eh_frame"],
                SHF_ALLOC,
                8,
            ),
            SectionAssignment::new(
                ".eh_frame_hdr",
                vec![".eh_frame_hdr"],
                SHF_ALLOC,
                4,
            ),
            // ---- Read-only dynamic link tables ----
            SectionAssignment::new(
                ".dynsym",
                vec![".dynsym"],
                SHF_ALLOC,
                8,
            ),
            SectionAssignment::new(
                ".dynstr",
                vec![".dynstr"],
                SHF_ALLOC,
                1,
            ),
            SectionAssignment::new(
                ".gnu.hash",
                vec![".gnu.hash"],
                SHF_ALLOC,
                8,
            ),
            SectionAssignment::new(
                ".rela.dyn",
                vec![".rela.dyn", ".rela.*"],
                SHF_ALLOC,
                8,
            ),
            SectionAssignment::new(
                ".rela.plt",
                vec![".rela.plt"],
                SHF_ALLOC,
                8,
            ),
            SectionAssignment::new(
                ".interp",
                vec![".interp"],
                SHF_ALLOC,
                1,
            ),
            // ---- Initialised writable data ----
            SectionAssignment::new(
                ".data",
                vec![".data", ".data.*"],
                SHF_ALLOC | SHF_WRITE,
                16,
            ),
            SectionAssignment::new(
                ".init_array",
                vec![".init_array", ".init_array.*"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            SectionAssignment::new(
                ".fini_array",
                vec![".fini_array", ".fini_array.*"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            SectionAssignment::new(
                ".ctors",
                vec![".ctors"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            SectionAssignment::new(
                ".dtors",
                vec![".dtors"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            // ---- GOT / PLT data ----
            SectionAssignment::new(
                ".got",
                vec![".got"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            SectionAssignment::new(
                ".got.plt",
                vec![".got.plt"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            // ---- Dynamic linking metadata ----
            SectionAssignment::new(
                ".dynamic",
                vec![".dynamic"],
                SHF_ALLOC | SHF_WRITE,
                8,
            ),
            // ---- Uninitialised data (NOBITS) ----
            SectionAssignment::new(
                ".bss",
                vec![".bss", ".bss.*", "COMMON"],
                SHF_ALLOC | SHF_WRITE,
                16,
            ),
            // ---- Note sections ----
            SectionAssignment::new(
                ".note",
                vec![".note", ".note.*"],
                SHF_ALLOC,
                4,
            ),
            // ---- Debug sections (non-loadable, no SHF_ALLOC) ----
            SectionAssignment::new(".debug_info", vec![".debug_info"], 0, 1),
            SectionAssignment::new(".debug_abbrev", vec![".debug_abbrev"], 0, 1),
            SectionAssignment::new(".debug_line", vec![".debug_line"], 0, 1),
            SectionAssignment::new(".debug_str", vec![".debug_str"], 0, 1),
            SectionAssignment::new(".debug_ranges", vec![".debug_ranges"], 0, 1),
            SectionAssignment::new(".debug_loc", vec![".debug_loc"], 0, 1),
            SectionAssignment::new(".debug_frame", vec![".debug_frame"], 0, 1),
            SectionAssignment::new(".debug_aranges", vec![".debug_aranges"], 0, 1),
            // ---- Non-loadable metadata ----
            SectionAssignment::new(".comment", vec![".comment"], 0, 1),
            SectionAssignment::new(".symtab", vec![".symtab"], 0, 8),
            SectionAssignment::new(".strtab", vec![".strtab"], 0, 1),
            SectionAssignment::new(".shstrtab", vec![".shstrtab"], 0, 1),
        ];
    }

    // ---------------------------------------------------------------
    // Private — segment rules
    // ---------------------------------------------------------------

    /// Populates the default segment mapping rules.
    ///
    /// The ordering of rules matters: it determines the program header
    /// order in the final ELF file. `PT_PHDR` comes first, followed by
    /// `PT_INTERP` (if present), then `PT_LOAD` segments ordered
    /// code → rodata → data, then `PT_DYNAMIC` and `PT_GNU_RELRO` for
    /// shared objects, and finally `PT_GNU_STACK`.
    fn build_default_segment_rules(&mut self) {
        let page = self.page_sz;

        // PT_PHDR — program header table self-reference
        self.segment_rules.push(SegmentRule::new(
            PT_PHDR,
            PF_R,
            8,
            vec![],
        ));

        // PT_INTERP — path to dynamic linker (shared libs and dynamically linked exes)
        if self.output_type == OutputType::SharedLibrary {
            self.segment_rules.push(SegmentRule::new(
                PT_INTERP,
                PF_R,
                1,
                vec![".interp".to_string()],
            ));
        }

        // PT_LOAD #1 — executable code segment (R+X)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R | PF_X,
            page,
            vec![
                ".text".to_string(),
                ".init".to_string(),
                ".fini".to_string(),
                ".plt".to_string(),
            ],
        ));

        // PT_LOAD #2 — read-only data segment (R)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R,
            page,
            vec![
                ".rodata".to_string(),
                ".eh_frame".to_string(),
                ".eh_frame_hdr".to_string(),
                ".dynsym".to_string(),
                ".dynstr".to_string(),
                ".gnu.hash".to_string(),
                ".rela.dyn".to_string(),
                ".rela.plt".to_string(),
                ".note".to_string(),
                ".interp".to_string(),
            ],
        ));

        // PT_LOAD #3 — writable data segment (R+W)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R | PF_W,
            page,
            vec![
                ".data".to_string(),
                ".init_array".to_string(),
                ".fini_array".to_string(),
                ".ctors".to_string(),
                ".dtors".to_string(),
                ".dynamic".to_string(),
                ".got".to_string(),
                ".got.plt".to_string(),
                ".bss".to_string(),
            ],
        ));

        // ---- Shared-object-only segments ----
        if self.output_type == OutputType::SharedLibrary {
            // PT_DYNAMIC — dynamic linking information
            self.segment_rules.push(SegmentRule::new(
                PT_DYNAMIC,
                PF_R | PF_W,
                8,
                vec![".dynamic".to_string()],
            ));

            // PT_GNU_RELRO — read-only after relocation
            self.segment_rules.push(SegmentRule::new(
                PT_GNU_RELRO,
                PF_R,
                1,
                vec![".dynamic".to_string(), ".got".to_string()],
            ));
        }

        // PT_GNU_STACK — non-executable stack marker (always last)
        self.segment_rules.push(SegmentRule::new(
            PT_GNU_STACK,
            PF_R | PF_W,
            0x10,
            vec![],
        ));
    }

    // ---------------------------------------------------------------
    // Private — segment header builders
    // ---------------------------------------------------------------

    /// Builds a `PT_LOAD` program header from a set of matched sections.
    ///
    /// Handles the `SHT_NOBITS` distinction: `.bss` sections contribute
    /// to `p_memsz` but NOT to `p_filesz`.
    fn build_load_header(
        rule: &SegmentRule,
        matched: &[(usize, &OutputSection)],
    ) -> ProgramHeader {
        let mut min_offset = u64::MAX;
        let mut min_vaddr = u64::MAX;
        let mut max_file_end: u64 = 0;
        let mut max_mem_end: u64 = 0;

        for &(_idx, section) in matched {
            let is_nobits = section.section_type == SHT_NOBITS;

            if section.addr < min_vaddr {
                min_vaddr = section.addr;
            }
            if !is_nobits && section.offset < min_offset {
                min_offset = section.offset;
            }

            // Memory extent always includes the section
            let mem_end = section.addr.saturating_add(section.size);
            if mem_end > max_mem_end {
                max_mem_end = mem_end;
            }

            // File extent excludes NOBITS (.bss)
            if !is_nobits {
                let file_end = section.offset.saturating_add(section.size);
                if file_end > max_file_end {
                    max_file_end = file_end;
                }
            }
        }

        // If every section is NOBITS, use a zero file offset
        if min_offset == u64::MAX {
            min_offset = if min_vaddr < u64::MAX { 0 } else { 0 };
        }

        let filesz = if max_file_end > min_offset {
            max_file_end - min_offset
        } else {
            0
        };
        let memsz = if max_mem_end > min_vaddr {
            max_mem_end - min_vaddr
        } else {
            0
        };

        ProgramHeader {
            p_type: PT_LOAD,
            p_flags: rule.flags,
            p_offset: min_offset,
            p_vaddr: min_vaddr,
            p_paddr: min_vaddr,
            p_filesz: filesz,
            p_memsz: memsz,
            p_align: rule.alignment,
        }
    }

    /// Builds a `PT_GNU_RELRO` program header from matched sections.
    ///
    /// The RELRO segment size is rounded up to the page boundary so the
    /// kernel can `mprotect` the entire range to read-only after relocations
    /// are applied.
    fn build_relro_header(
        rule: &SegmentRule,
        matched: &[(usize, &OutputSection)],
        page_size: u64,
    ) -> ProgramHeader {
        let mut min_offset = u64::MAX;
        let mut min_vaddr = u64::MAX;
        let mut max_end: u64 = 0;

        for &(_idx, section) in matched {
            if section.offset < min_offset {
                min_offset = section.offset;
            }
            if section.addr < min_vaddr {
                min_vaddr = section.addr;
            }
            let end = section.addr.saturating_add(section.size);
            if end > max_end {
                max_end = end;
            }
        }

        let raw_size = if max_end > min_vaddr {
            max_end - min_vaddr
        } else {
            0
        };
        let aligned_size = align_up(raw_size, page_size);

        ProgramHeader {
            p_type: PT_GNU_RELRO,
            p_flags: rule.flags,
            p_offset: min_offset,
            p_vaddr: min_vaddr,
            p_paddr: min_vaddr,
            p_filesz: aligned_size,
            p_memsz: aligned_size,
            p_align: 1,
        }
    }

    /// Builds a generic program header for an arbitrary segment type.
    fn build_generic_header(
        rule: &SegmentRule,
        matched: &[(usize, &OutputSection)],
    ) -> ProgramHeader {
        let mut min_offset = u64::MAX;
        let mut min_vaddr = u64::MAX;
        let mut max_end: u64 = 0;

        for &(_idx, section) in matched {
            if section.offset < min_offset {
                min_offset = section.offset;
            }
            if section.addr < min_vaddr {
                min_vaddr = section.addr;
            }
            let end = section.addr.saturating_add(section.size);
            if end > max_end {
                max_end = end;
            }
        }

        let size = if max_end > min_vaddr {
            max_end - min_vaddr
        } else {
            0
        };

        ProgramHeader {
            p_type: rule.segment_type,
            p_flags: rule.flags,
            p_offset: min_offset,
            p_vaddr: min_vaddr,
            p_paddr: min_vaddr,
            p_filesz: size,
            p_memsz: size,
            p_align: rule.alignment,
        }
    }

    // ---------------------------------------------------------------
    // Private — unclaimed section classification (flag-based)
    // ---------------------------------------------------------------

    /// Classifies unclaimed allocated sections by their ELF flags and
    /// appends them to the most appropriate existing `PT_LOAD` segment.
    ///
    /// A section is "unclaimed" if it was not matched by name in any
    /// segment rule during `compute_segment_layout`. Non-allocated sections
    /// (those without `SHF_ALLOC`) are skipped as they do not appear in
    /// loadable segments.
    ///
    /// The classification uses the section's `flags` field:
    /// - `SHF_EXECINSTR` → code segment (`PF_R | PF_X`)
    /// - `SHF_WRITE` → data segment (`PF_R | PF_W`)
    /// - Otherwise → read-only data segment (`PF_R`)
    fn assign_unclaimed_sections(
        &self,
        output_sections: &[OutputSection],
        claimed: &[bool],
        headers: &mut Vec<ProgramHeader>,
    ) {
        for (idx, section) in output_sections.iter().enumerate() {
            if claimed[idx] {
                continue;
            }
            // Only handle allocated sections; non-allocated sections are
            // not placed in any loadable segment.
            if section.flags & SHF_ALLOC == 0 {
                continue;
            }

            // Classify permission intent from section flags
            let target_pf = classify_section_permissions(section.flags);

            // Try to find an existing PT_LOAD header with matching flags
            let mut found = false;
            for hdr in headers.iter_mut() {
                if hdr.p_type == PT_LOAD && hdr.p_flags == target_pf {
                    // Expand the segment to include this section
                    let is_nobits = section.section_type == SHT_NOBITS;

                    let sec_end_mem = section.addr.saturating_add(section.size);
                    let current_end_mem = hdr.p_vaddr.saturating_add(hdr.p_memsz);
                    if sec_end_mem > current_end_mem {
                        hdr.p_memsz = sec_end_mem - hdr.p_vaddr;
                    }
                    if section.addr < hdr.p_vaddr {
                        let delta = hdr.p_vaddr - section.addr;
                        hdr.p_vaddr = section.addr;
                        hdr.p_paddr = section.addr;
                        hdr.p_memsz += delta;
                        if !is_nobits {
                            hdr.p_offset = section.offset;
                            hdr.p_filesz += delta;
                        }
                    }
                    if !is_nobits {
                        let sec_end_file = section.offset.saturating_add(section.size);
                        let current_end_file =
                            hdr.p_offset.saturating_add(hdr.p_filesz);
                        if sec_end_file > current_end_file {
                            hdr.p_filesz = sec_end_file - hdr.p_offset;
                        }
                    }
                    found = true;
                    break;
                }
            }

            // If no existing segment matches, create a new PT_LOAD
            if !found {
                let is_nobits = section.section_type == SHT_NOBITS;
                headers.push(ProgramHeader {
                    p_type: PT_LOAD,
                    p_flags: target_pf,
                    p_offset: if is_nobits { 0 } else { section.offset },
                    p_vaddr: section.addr,
                    p_paddr: section.addr,
                    p_filesz: if is_nobits { 0 } else { section.size },
                    p_memsz: section.size,
                    p_align: self.page_sz,
                });
            }
        }
    }
}

// ===========================================================================
// Module-level helper functions
// ===========================================================================

/// Aligns `value` up to the next multiple of `alignment`.
///
/// Returns `value` unchanged if it is already aligned, or if `alignment`
/// is 0 or 1 (no alignment needed).
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let mask = alignment - 1;
    (value + mask) & !mask
}

/// Finds an output section by exact name, returning its index and reference.
fn find_section_indexed<'a>(
    sections: &'a [OutputSection],
    name: &str,
) -> Option<(usize, &'a OutputSection)> {
    sections
        .iter()
        .enumerate()
        .find(|(_, s)| s.name == name)
}

/// Collects all output sections whose names appear in `names`, returning
/// index-reference pairs.
fn collect_matching_indexed<'a>(
    sections: &'a [OutputSection],
    names: &[String],
) -> Vec<(usize, &'a OutputSection)> {
    sections
        .iter()
        .enumerate()
        .filter(|(_, s)| names.iter().any(|n| n == &s.name))
        .collect()
}

/// Classifies a section's permission intent based on its ELF section flags
/// and returns the corresponding `PF_*` program header flags.
///
/// - `SHF_ALLOC | SHF_EXECINSTR` → `PF_R | PF_X` (executable code)
/// - `SHF_ALLOC | SHF_WRITE` → `PF_R | PF_W` (writable data)
/// - `SHF_ALLOC` (only) → `PF_R` (read-only data)
/// - No `SHF_ALLOC` → `0` (not loadable)
fn classify_section_permissions(section_flags: u64) -> u32 {
    if section_flags & SHF_ALLOC == 0 {
        return 0;
    }
    let mut pf: u32 = PF_R;
    if section_flags & SHF_EXECINSTR != 0 {
        pf |= PF_X;
    }
    if section_flags & SHF_WRITE != 0 {
        pf |= PF_W;
    }
    pf
}

// ===========================================================================
// Standalone public convenience functions
// ===========================================================================

/// Returns the default base virtual address for a target and output type.
///
/// Convenience wrapper around [`LinkerScript`]'s internal logic.
///
/// # Architecture-Specific Values
///
/// | Target    | Executable Base | Shared Library |
/// |-----------|-----------------|----------------|
/// | x86-64    | `0x400000`      | `0x0`          |
/// | i686      | `0x08048000`    | `0x0`          |
/// | AArch64   | `0x400000`      | `0x0`          |
/// | RISC-V 64 | `0x10000`       | `0x0`          |
pub fn base_address(target: &Target, output_type: &OutputType) -> u64 {
    LinkerScript::compute_base_address(target, output_type)
}

/// Returns the default page size for a target architecture.
///
/// All four supported architectures default to 4096 bytes (4 KiB).
pub fn page_size(target: &Target) -> u64 {
    LinkerScript::compute_page_size(target)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::elf_writer_common::{
        PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_RELRO, PT_GNU_STACK, PT_INTERP, PT_LOAD,
        PT_PHDR, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS, SHT_PROGBITS,
    };

    // ---- OutputType tests --------------------------------------------

    #[test]
    fn output_type_display() {
        assert_eq!(format!("{}", OutputType::Executable), "executable (ET_EXEC)");
        assert_eq!(
            format!("{}", OutputType::SharedLibrary),
            "shared library (ET_DYN)"
        );
        assert_eq!(
            format!("{}", OutputType::RelocatableObject),
            "relocatable object (ET_REL)"
        );
    }

    #[test]
    fn output_type_equality() {
        assert_eq!(OutputType::Executable, OutputType::Executable);
        assert_ne!(OutputType::Executable, OutputType::SharedLibrary);
        assert_ne!(OutputType::SharedLibrary, OutputType::RelocatableObject);
    }

    #[test]
    fn output_type_clone_copy() {
        let a = OutputType::Executable;
        let b = a;
        assert_eq!(a, b);
    }

    // ---- SectionAssignment tests -------------------------------------

    #[test]
    fn section_assignment_exact_match() {
        let rule = SectionAssignment::new(".text", vec![".text"], 0, 16);
        assert!(rule.matches(".text"));
        assert!(!rule.matches(".text.startup"));
        assert!(!rule.matches(".textual"));
        assert!(!rule.matches(".tex"));
    }

    #[test]
    fn section_assignment_dot_wildcard() {
        let rule = SectionAssignment::new(".text", vec![".text.*"], 0, 16);
        assert!(rule.matches(".text"));
        assert!(rule.matches(".text.startup"));
        assert!(rule.matches(".text.unlikely"));
        assert!(!rule.matches(".textual"));
        assert!(!rule.matches(".tex"));
    }

    #[test]
    fn section_assignment_star_wildcard() {
        let rule = SectionAssignment::new(".text", vec![".text*"], 0, 16);
        assert!(rule.matches(".text"));
        assert!(rule.matches(".text.startup"));
        assert!(rule.matches(".textual"));
        assert!(!rule.matches(".tex"));
        assert!(!rule.matches(".data"));
    }

    #[test]
    fn section_assignment_multiple_patterns() {
        let rule = SectionAssignment::new(
            ".text",
            vec![".text", ".text.*", ".init", ".fini"],
            SHF_ALLOC | SHF_EXECINSTR,
            16,
        );
        assert!(rule.matches(".text"));
        assert!(rule.matches(".text.startup"));
        assert!(rule.matches(".init"));
        assert!(rule.matches(".fini"));
        assert!(!rule.matches(".data"));
        assert!(!rule.matches(".rodata"));
    }

    #[test]
    fn section_assignment_fields() {
        let rule = SectionAssignment::new(".data", vec![".data", ".data.*"], 3, 32);
        assert_eq!(rule.output_name, ".data");
        assert_eq!(rule.input_patterns.len(), 2);
        assert_eq!(rule.flags, 3);
        assert_eq!(rule.alignment, 32);
    }

    #[test]
    fn section_assignment_no_match() {
        let rule = SectionAssignment::new(".text", vec![".text"], 0, 16);
        assert!(!rule.matches(""));
        assert!(!rule.matches(".data"));
        assert!(!rule.matches(".bss"));
    }

    // ---- SegmentRule tests -------------------------------------------

    #[test]
    fn segment_rule_contains_section() {
        let rule = SegmentRule::new(
            PT_LOAD,
            PF_R | PF_X,
            0x1000,
            vec![".text".to_string(), ".init".to_string()],
        );
        assert!(rule.contains_section(".text"));
        assert!(rule.contains_section(".init"));
        assert!(!rule.contains_section(".data"));
        assert!(!rule.contains_section(""));
    }

    #[test]
    fn segment_rule_fields() {
        let rule = SegmentRule::new(PT_LOAD, PF_R, 0x1000, vec![".rodata".to_string()]);
        assert_eq!(rule.segment_type, PT_LOAD);
        assert_eq!(rule.flags, PF_R);
        assert_eq!(rule.alignment, 0x1000);
        assert_eq!(rule.sections.len(), 1);
        assert_eq!(rule.sections[0], ".rodata");
    }

    // ---- LinkerScript construction tests -----------------------------

    #[test]
    fn default_x86_64_executable() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.entry_point(), "_start");
        assert_eq!(script.base_address(), 0x0040_0000);
        assert_eq!(script.page_size(), 0x1000);
        assert_eq!(script.target(), Target::X86_64);
        assert_eq!(script.output_type(), OutputType::Executable);
        assert!(!script.section_rules().is_empty());
        assert!(!script.segment_mapping().is_empty());
    }

    #[test]
    fn default_i686_executable() {
        let script = LinkerScript::default_for_target(&Target::I686, OutputType::Executable);
        assert_eq!(script.entry_point(), "_start");
        assert_eq!(script.base_address(), 0x0804_8000);
        assert_eq!(script.page_size(), 0x1000);
    }

    #[test]
    fn default_aarch64_executable() {
        let script = LinkerScript::default_for_target(&Target::AArch64, OutputType::Executable);
        assert_eq!(script.entry_point(), "_start");
        assert_eq!(script.base_address(), 0x0040_0000);
        assert_eq!(script.page_size(), 0x1000);
    }

    #[test]
    fn default_riscv64_executable() {
        let script = LinkerScript::default_for_target(&Target::RiscV64, OutputType::Executable);
        assert_eq!(script.entry_point(), "_start");
        assert_eq!(script.base_address(), 0x0001_0000);
        assert_eq!(script.page_size(), 0x1000);
    }

    #[test]
    fn shared_library_no_entry() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);
        assert_eq!(script.entry_point(), "");
        assert_eq!(script.base_address(), 0);
    }

    #[test]
    fn relocatable_object_no_entry() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::RelocatableObject);
        assert_eq!(script.entry_point(), "");
        assert_eq!(script.base_address(), 0);
    }

    #[test]
    fn all_architectures_create_valid_scripts() {
        for target in &[Target::X86_64, Target::I686, Target::AArch64, Target::RiscV64] {
            let exe = LinkerScript::default_for_target(target, OutputType::Executable);
            assert_eq!(exe.entry_point(), "_start");
            assert!(exe.base_address() > 0);
            assert_eq!(exe.page_size(), 0x1000);
            assert!(!exe.section_rules().is_empty());
            assert!(!exe.segment_mapping().is_empty());

            let so = LinkerScript::default_for_target(target, OutputType::SharedLibrary);
            assert_eq!(so.entry_point(), "");
            assert_eq!(so.base_address(), 0);
        }
    }

    // ---- Section resolution tests ------------------------------------

    #[test]
    fn resolve_section_text() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".text"), Some(".text"));
        assert_eq!(script.resolve_section(".text.startup"), Some(".text"));
        assert_eq!(script.resolve_section(".init"), Some(".text"));
        assert_eq!(script.resolve_section(".fini"), Some(".text"));
    }

    #[test]
    fn resolve_section_data_bss() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".data"), Some(".data"));
        assert_eq!(script.resolve_section(".data.rel.ro"), Some(".data"));
        assert_eq!(script.resolve_section(".bss"), Some(".bss"));
    }

    #[test]
    fn resolve_section_rodata() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".rodata"), Some(".rodata"));
        assert_eq!(script.resolve_section(".rodata.str1.1"), Some(".rodata"));
    }

    #[test]
    fn resolve_section_debug() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".debug_info"), Some(".debug_info"));
        assert_eq!(script.resolve_section(".debug_line"), Some(".debug_line"));
        assert_eq!(script.resolve_section(".debug_str"), Some(".debug_str"));
    }

    #[test]
    fn resolve_section_unknown() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".nonexistent"), None);
    }

    // ---- Segment query tests -----------------------------------------

    #[test]
    fn segment_for_text() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let seg = script.segment_for_section(".text").unwrap();
        assert_eq!(seg.segment_type, PT_LOAD);
        assert_eq!(seg.flags, PF_R | PF_X);
    }

    #[test]
    fn segment_for_data() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let seg = script.segment_for_section(".data").unwrap();
        assert_eq!(seg.segment_type, PT_LOAD);
        assert_eq!(seg.flags, PF_R | PF_W);
    }

    #[test]
    fn segment_for_rodata() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let seg = script.segment_for_section(".rodata").unwrap();
        assert_eq!(seg.segment_type, PT_LOAD);
        assert_eq!(seg.flags, PF_R);
    }

    #[test]
    fn shared_object_has_dynamic_segment() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);
        let dynamic = script.segments_of_type(PT_DYNAMIC);
        assert!(!dynamic.is_empty());
        assert_eq!(dynamic[0].flags, PF_R | PF_W);
    }

    #[test]
    fn executable_no_dynamic_segment() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let dynamic = script.segments_of_type(PT_DYNAMIC);
        assert!(dynamic.is_empty());
    }

    #[test]
    fn shared_object_has_relro() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);
        let relro = script.segments_of_type(PT_GNU_RELRO);
        assert!(!relro.is_empty());
        assert_eq!(relro[0].flags, PF_R);
    }

    #[test]
    fn shared_object_has_interp() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);
        let interp = script.segments_of_type(PT_INTERP);
        assert!(!interp.is_empty());
    }

    #[test]
    fn shared_object_more_headers_than_executable() {
        let exe = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let so = LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);
        assert!(so.program_header_count() > exe.program_header_count());
    }

    // ---- GNU_STACK tests ---------------------------------------------

    #[test]
    fn gnu_stack_header_values() {
        let hdr = LinkerScript::gnu_stack_header();
        assert_eq!(hdr.p_type, PT_GNU_STACK);
        assert_eq!(hdr.p_flags, PF_R | PF_W);
        assert_eq!(hdr.p_offset, 0);
        assert_eq!(hdr.p_vaddr, 0);
        assert_eq!(hdr.p_paddr, 0);
        assert_eq!(hdr.p_filesz, 0);
        assert_eq!(hdr.p_memsz, 0);
        // Must NOT have PF_X
        assert_eq!(hdr.p_flags & PF_X, 0);
    }

    // ---- Mutator tests -----------------------------------------------

    #[test]
    fn set_entry_point() {
        let mut script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        script.set_entry_point("main");
        assert_eq!(script.entry_point(), "main");
    }

    #[test]
    fn set_base_address() {
        let mut script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        script.set_base_address(0x1000_0000);
        assert_eq!(script.base_address(), 0x1000_0000);
    }

    // ---- Page alignment tests ----------------------------------------

    #[test]
    fn align_to_page_values() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        assert_eq!(script.align_to_page(0), 0);
        assert_eq!(script.align_to_page(1), 0x1000);
        assert_eq!(script.align_to_page(0x1000), 0x1000);
        assert_eq!(script.align_to_page(0x1001), 0x2000);
        assert_eq!(script.align_to_page(0xFFF), 0x1000);
    }

    // ---- Standalone function tests -----------------------------------

    #[test]
    fn standalone_base_address() {
        assert_eq!(base_address(&Target::X86_64, &OutputType::Executable), 0x0040_0000);
        assert_eq!(base_address(&Target::I686, &OutputType::Executable), 0x0804_8000);
        assert_eq!(base_address(&Target::AArch64, &OutputType::Executable), 0x0040_0000);
        assert_eq!(base_address(&Target::RiscV64, &OutputType::Executable), 0x0001_0000);
        assert_eq!(base_address(&Target::X86_64, &OutputType::SharedLibrary), 0);
        assert_eq!(base_address(&Target::I686, &OutputType::RelocatableObject), 0);
    }

    #[test]
    fn standalone_page_size() {
        assert_eq!(page_size(&Target::X86_64), 0x1000);
        assert_eq!(page_size(&Target::I686), 0x1000);
        assert_eq!(page_size(&Target::AArch64), 0x1000);
        assert_eq!(page_size(&Target::RiscV64), 0x1000);
    }

    // ---- align_up helper tests ---------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(5, 0), 5);
        assert_eq!(align_up(5, 1), 5);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }

    // ---- classify_section_permissions tests ---------------------------

    #[test]
    fn test_classify_section_permissions() {
        assert_eq!(classify_section_permissions(0), 0);
        assert_eq!(classify_section_permissions(SHF_ALLOC), PF_R);
        assert_eq!(
            classify_section_permissions(SHF_ALLOC | SHF_WRITE),
            PF_R | PF_W
        );
        assert_eq!(
            classify_section_permissions(SHF_ALLOC | SHF_EXECINSTR),
            PF_R | PF_X
        );
        assert_eq!(
            classify_section_permissions(SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR),
            PF_R | PF_W | PF_X
        );
    }

    // ---- compute_segment_layout tests --------------------------------

    /// Helper: create a minimal OutputSection for testing.
    fn make_section(
        name: &str,
        section_type: u32,
        flags: u64,
        addr: u64,
        offset: u64,
        size: u64,
    ) -> OutputSection {
        OutputSection {
            name: name.to_string(),
            section_type,
            flags,
            alignment: 16,
            addr,
            offset,
            size,
            input_sections: vec![],
            entry_size: 0,
        }
    }

    #[test]
    fn compute_layout_basic_executable() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);

        let sections = vec![
            make_section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x401000, 0x1000, 0x200),
            make_section(".rodata", SHT_PROGBITS, SHF_ALLOC, 0x402000, 0x2000, 0x100),
            make_section(".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x403000, 0x3000, 0x80),
            make_section(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0x403080, 0x3080, 0x40),
        ];

        let headers = script.compute_segment_layout(&sections);

        // Should have: PT_PHDR, PT_LOAD (R+X), PT_LOAD (R), PT_LOAD (R+W), PT_GNU_STACK
        assert!(headers.len() >= 4, "Expected at least 4 headers, got {}", headers.len());

        // Find the PT_PHDR
        let phdr = headers.iter().find(|h| h.p_type == PT_PHDR);
        assert!(phdr.is_some(), "Missing PT_PHDR");

        // Find executable code segment
        let code_seg = headers
            .iter()
            .find(|h| h.p_type == PT_LOAD && h.p_flags == (PF_R | PF_X));
        assert!(code_seg.is_some(), "Missing code PT_LOAD");
        let code_seg = code_seg.unwrap();
        assert_eq!(code_seg.p_vaddr, 0x401000);
        assert_eq!(code_seg.p_filesz, 0x200);
        assert_eq!(code_seg.p_memsz, 0x200);

        // Find read-only data segment
        let ro_seg = headers
            .iter()
            .find(|h| h.p_type == PT_LOAD && h.p_flags == PF_R);
        assert!(ro_seg.is_some(), "Missing rodata PT_LOAD");
        let ro_seg = ro_seg.unwrap();
        assert_eq!(ro_seg.p_vaddr, 0x402000);

        // Find writable data segment (.data + .bss)
        let rw_seg = headers
            .iter()
            .find(|h| h.p_type == PT_LOAD && h.p_flags == (PF_R | PF_W));
        assert!(rw_seg.is_some(), "Missing data PT_LOAD");
        let rw_seg = rw_seg.unwrap();
        assert_eq!(rw_seg.p_vaddr, 0x403000);
        // .data is 0x80 at offset 0x3000, .bss is 0x40 at addr 0x403080
        // p_filesz should cover only .data (not .bss which is NOBITS)
        assert_eq!(rw_seg.p_filesz, 0x80);
        // p_memsz should cover .data + .bss
        assert_eq!(rw_seg.p_memsz, 0x80 + 0x40);

        // Find GNU_STACK
        let stack = headers.iter().find(|h| h.p_type == PT_GNU_STACK);
        assert!(stack.is_some(), "Missing PT_GNU_STACK");
        let stack = stack.unwrap();
        assert_eq!(stack.p_flags & PF_X, 0, "Stack must not be executable");
    }

    #[test]
    fn compute_layout_shared_library() {
        let script =
            LinkerScript::default_for_target(&Target::X86_64, OutputType::SharedLibrary);

        let sections = vec![
            make_section(".interp", SHT_PROGBITS, SHF_ALLOC, 0x200, 0x200, 0x1c),
            make_section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x1000, 0x1000, 0x100),
            make_section(".rodata", SHT_PROGBITS, SHF_ALLOC, 0x2000, 0x2000, 0x50),
            make_section(".dynamic", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x3000, 0x3000, 0xF0),
            make_section(".got", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x30F0, 0x30F0, 0x20),
            make_section(".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 0x3200, 0x3200, 0x40),
        ];

        let headers = script.compute_segment_layout(&sections);

        // Should have PT_INTERP
        let interp = headers.iter().find(|h| h.p_type == PT_INTERP);
        assert!(interp.is_some(), "Missing PT_INTERP for shared library");
        let interp = interp.unwrap();
        assert_eq!(interp.p_vaddr, 0x200);
        assert_eq!(interp.p_filesz, 0x1c);

        // Should have PT_DYNAMIC
        let dynamic = headers.iter().find(|h| h.p_type == PT_DYNAMIC);
        assert!(dynamic.is_some(), "Missing PT_DYNAMIC for shared library");
        let dynamic = dynamic.unwrap();
        assert_eq!(dynamic.p_vaddr, 0x3000);
        assert_eq!(dynamic.p_filesz, 0xF0);

        // Should have PT_GNU_RELRO
        let relro = headers.iter().find(|h| h.p_type == PT_GNU_RELRO);
        assert!(relro.is_some(), "Missing PT_GNU_RELRO for shared library");
    }

    #[test]
    fn compute_layout_empty_sections() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let sections: Vec<OutputSection> = vec![];
        let headers = script.compute_segment_layout(&sections);
        // Should still have PT_PHDR and PT_GNU_STACK at minimum
        assert!(headers.iter().any(|h| h.p_type == PT_PHDR));
        assert!(headers.iter().any(|h| h.p_type == PT_GNU_STACK));
    }

    #[test]
    fn compute_layout_bss_only_segment() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);
        let sections = vec![
            make_section(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0x403000, 0x3000, 0x1000),
        ];
        let headers = script.compute_segment_layout(&sections);

        let rw_seg = headers
            .iter()
            .find(|h| h.p_type == PT_LOAD && h.p_flags == (PF_R | PF_W));
        assert!(rw_seg.is_some(), "Missing RW segment for .bss");
        let rw_seg = rw_seg.unwrap();
        // .bss is NOBITS — p_filesz must be 0
        assert_eq!(rw_seg.p_filesz, 0);
        // p_memsz must include .bss
        assert_eq!(rw_seg.p_memsz, 0x1000);
    }

    #[test]
    fn compute_layout_unclaimed_section_classified_by_flags() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);

        // Create an unusual section name that won't match any named rule
        let sections = vec![
            make_section(
                ".custom_code",
                SHT_PROGBITS,
                SHF_ALLOC | SHF_EXECINSTR,
                0x500000,
                0x5000,
                0x100,
            ),
        ];

        let headers = script.compute_segment_layout(&sections);
        // The unclaimed section should be classified into a PT_LOAD R+X segment
        let code_segs: Vec<_> = headers
            .iter()
            .filter(|h| h.p_type == PT_LOAD && (h.p_flags & PF_X) != 0)
            .collect();
        // Should have at least one code segment containing our custom section
        assert!(
            code_segs.iter().any(|s| s.p_vaddr <= 0x500000
                && s.p_vaddr + s.p_memsz >= 0x500100),
            "Unclaimed code section not placed in any PT_LOAD R+X segment"
        );
    }

    #[test]
    fn compute_layout_non_alloc_not_loaded() {
        let script = LinkerScript::default_for_target(&Target::X86_64, OutputType::Executable);

        // Debug section without SHF_ALLOC — must NOT appear in any PT_LOAD
        let sections = vec![
            make_section(".debug_info", SHT_PROGBITS, 0, 0x0, 0x8000, 0x500),
        ];

        let headers = script.compute_segment_layout(&sections);
        let load_segs: Vec<_> = headers.iter().filter(|h| h.p_type == PT_LOAD).collect();
        // No PT_LOAD should cover the debug section
        for seg in &load_segs {
            assert!(
                seg.p_filesz == 0 || seg.p_offset > 0x8500 || seg.p_offset + seg.p_filesz <= 0x8000,
                "Debug section incorrectly placed in a PT_LOAD segment"
            );
        }
    }
}
