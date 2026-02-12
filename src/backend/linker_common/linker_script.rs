//! Default linker script handling and section-to-segment mapping.
//!
//! When no external linker script is provided (the common case for BCC),
//! this module supplies the default rules for:
//!
//! - **Section placement:** Which input sections map to which output sections.
//! - **Segment layout:** Which output sections are grouped into which ELF
//!   program header (`PT_LOAD`, `PT_DYNAMIC`, `PT_GNU_STACK`, etc.) segments.
//! - **Architecture-specific base addresses:** Different page sizes and
//!   default load addresses per target.
//! - **Entry point:** Default `_start` symbol as program entry.
//!
//! The section-to-segment mapping defines the memory protection attributes
//! (read/write/execute) for each loadable segment:
//!
//! | Segment   | Sections                       | Permissions |
//! |-----------|--------------------------------|-------------|
//! | PT_LOAD 0 | `.text`                        | R + X       |
//! | PT_LOAD 1 | `.rodata`                      | R           |
//! | PT_LOAD 2 | `.data`, `.bss`                | R + W       |
//! | PT_DYNAMIC| `.dynamic`                     | R + W       |
//! | PT_GNU_STACK | (none)                      | R + W       |
//!
//! Removing this module would eliminate default linking behaviour, requiring
//! all BCC invocations to provide an explicit linker script.

use crate::common::target::Target;
use std::fmt;

// ===========================================================================
// ELF program header type constants
// ===========================================================================

/// PT_NULL — unused entry.
pub const PT_NULL: u32 = 0;
/// PT_LOAD — loadable segment.
pub const PT_LOAD: u32 = 1;
/// PT_DYNAMIC — dynamic linking information.
pub const PT_DYNAMIC: u32 = 2;
/// PT_INTERP — path to dynamic linker.
pub const PT_INTERP: u32 = 3;
/// PT_NOTE — auxiliary information.
pub const PT_NOTE: u32 = 4;
/// PT_PHDR — program header table.
pub const PT_PHDR: u32 = 6;
/// PT_GNU_STACK — GNU stack permissions.
pub const PT_GNU_STACK: u32 = 0x6474_E551;
/// PT_GNU_RELRO — GNU read-only after relocation.
pub const PT_GNU_RELRO: u32 = 0x6474_E552;

// ===========================================================================
// ELF segment permission flags
// ===========================================================================

/// PF_X — Execute permission.
pub const PF_X: u32 = 1;
/// PF_W — Write permission.
pub const PF_W: u32 = 2;
/// PF_R — Read permission.
pub const PF_R: u32 = 4;

// ===========================================================================
// OutputType — type of output binary
// ===========================================================================

/// Type of output binary being produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// ET_EXEC — static executable (default).
    Executable,
    /// ET_DYN — shared object (when `-shared` is specified).
    SharedObject,
    /// Relocatable object — when `-c` is used (no linker involved).
    Relocatable,
}

impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputType::Executable => write!(f, "executable (ET_EXEC)"),
            OutputType::SharedObject => write!(f, "shared object (ET_DYN)"),
            OutputType::Relocatable => write!(f, "relocatable object (ET_REL)"),
        }
    }
}

// ===========================================================================
// SectionAssignment — section-to-output mapping rule
// ===========================================================================

/// A rule mapping input section name patterns to output sections.
///
/// Used by the default linker script to determine where each input section
/// is placed in the final output.
#[derive(Debug, Clone)]
pub struct SectionAssignment {
    /// Pattern for input section name matching (e.g., `.text`, `.text.*`).
    /// The `*` wildcard matches any suffix.
    pub input_pattern: String,
    /// Name of the output section this input maps to.
    pub output_section: String,
}

impl SectionAssignment {
    /// Creates a new section assignment rule.
    pub fn new(input_pattern: &str, output_section: &str) -> Self {
        Self {
            input_pattern: input_pattern.to_string(),
            output_section: output_section.to_string(),
        }
    }

    /// Tests whether an input section name matches this rule's pattern.
    ///
    /// Supports exact match and suffix wildcard (e.g., `.text.*` matches
    /// `.text.startup`, `.text.unlikely`).
    pub fn matches(&self, section_name: &str) -> bool {
        if self.input_pattern.ends_with(".*") {
            let prefix = &self.input_pattern[..self.input_pattern.len() - 2];
            section_name == prefix || section_name.starts_with(&format!("{}.", prefix))
        } else if self.input_pattern.ends_with('*') {
            let prefix = &self.input_pattern[..self.input_pattern.len() - 1];
            section_name.starts_with(prefix)
        } else {
            section_name == self.input_pattern
        }
    }
}

// ===========================================================================
// SegmentRule — segment membership and permissions
// ===========================================================================

/// Rule defining which output sections belong to a program header segment
/// and the segment's memory protection flags.
#[derive(Debug, Clone)]
pub struct SegmentRule {
    /// Program header type (PT_LOAD, PT_DYNAMIC, etc.).
    pub segment_type: u32,
    /// Segment flags (PF_R, PF_W, PF_X combinations).
    pub flags: u32,
    /// Output section names assigned to this segment.
    pub sections: Vec<String>,
    /// Segment alignment (typically page size).
    pub alignment: u64,
}

impl SegmentRule {
    /// Creates a new segment rule.
    pub fn new(segment_type: u32, flags: u32, sections: Vec<String>, alignment: u64) -> Self {
        Self {
            segment_type,
            flags,
            sections,
            alignment,
        }
    }

    /// Returns true if this segment contains the given output section.
    pub fn contains_section(&self, section_name: &str) -> bool {
        self.sections.iter().any(|s| s == section_name)
    }
}

// ===========================================================================
// LinkerScript — default linker script
// ===========================================================================

/// Default linker script providing section placement, segment layout, entry
/// point, and architecture-specific configuration.
///
/// This replaces the need for an external linker script file. The BCC built-in
/// linker always uses this default script (no custom linker script support).
#[derive(Debug, Clone)]
pub struct LinkerScript {
    /// Target architecture.
    target: Target,
    /// Type of output binary.
    output_type: OutputType,
    /// Entry point symbol name (default: `_start`).
    entry_point: String,
    /// Base virtual address for the first loadable segment.
    base_address: u64,
    /// Page size for the target architecture.
    page_size: u64,
    /// Section assignment rules (input → output mapping).
    section_assignments: Vec<SectionAssignment>,
    /// Segment rules (output sections → program headers).
    segment_rules: Vec<SegmentRule>,
}

impl LinkerScript {
    /// Creates a new linker script with default rules for the given target
    /// and output type.
    pub fn new(target: Target, output_type: OutputType) -> Self {
        let base_address = Self::default_base_address(target, output_type);
        let page_size = Self::default_page_size(target);

        let mut script = Self {
            target,
            output_type,
            entry_point: "_start".to_string(),
            base_address,
            page_size,
            section_assignments: Vec::new(),
            segment_rules: Vec::new(),
        };

        script.build_default_section_assignments();
        script.build_default_segment_rules();
        script
    }

    /// Returns the default base virtual address for the first loadable segment.
    ///
    /// These are the conventional Linux base addresses:
    /// - x86-64: `0x400000` (4 MiB aligned)
    /// - i686: `0x08048000` (traditional Linux i386 base)
    /// - AArch64: `0x400000`
    /// - RISC-V 64: `0x10000` (conventional for RISC-V)
    pub fn default_base_address(target: Target, output_type: OutputType) -> u64 {
        match output_type {
            OutputType::SharedObject => 0, // PIE/shared objects are position-independent
            OutputType::Relocatable => 0,
            OutputType::Executable => match target {
                Target::X86_64 => 0x0040_0000,
                Target::I686 => 0x0804_8000,
                Target::AArch64 => 0x0040_0000,
                Target::RiscV64 => 0x0001_0000,
            },
        }
    }

    /// Returns the default page size for the target architecture.
    ///
    /// - x86-64, i686: 4 KiB (0x1000)
    /// - AArch64: 4 KiB (some systems use 16 KiB or 64 KiB, but 4 KiB is default)
    /// - RISC-V 64: 4 KiB
    pub fn default_page_size(target: Target) -> u64 {
        match target {
            Target::X86_64 | Target::I686 | Target::AArch64 | Target::RiscV64 => 0x1000,
        }
    }

    /// Populates the default section assignment rules.
    ///
    /// Maps common input section names (including GCC/LLVM variants like
    /// `.text.startup`, `.rodata.str1.1`) to standard output sections.
    fn build_default_section_assignments(&mut self) {
        self.section_assignments = vec![
            // Code sections → .text
            SectionAssignment::new(".text", ".text"),
            SectionAssignment::new(".text.*", ".text"),
            SectionAssignment::new(".init", ".text"),
            SectionAssignment::new(".fini", ".text"),
            // Read-only data → .rodata
            SectionAssignment::new(".rodata", ".rodata"),
            SectionAssignment::new(".rodata.*", ".rodata"),
            SectionAssignment::new(".eh_frame", ".eh_frame"),
            SectionAssignment::new(".eh_frame_hdr", ".eh_frame_hdr"),
            // Initialized writable data → .data
            SectionAssignment::new(".data", ".data"),
            SectionAssignment::new(".data.*", ".data"),
            SectionAssignment::new(".init_array", ".init_array"),
            SectionAssignment::new(".fini_array", ".fini_array"),
            SectionAssignment::new(".ctors", ".ctors"),
            SectionAssignment::new(".dtors", ".dtors"),
            // Uninitialized data → .bss
            SectionAssignment::new(".bss", ".bss"),
            SectionAssignment::new(".bss.*", ".bss"),
            SectionAssignment::new("COMMON", ".bss"),
            // Debug sections pass through
            SectionAssignment::new(".debug_info", ".debug_info"),
            SectionAssignment::new(".debug_abbrev", ".debug_abbrev"),
            SectionAssignment::new(".debug_line", ".debug_line"),
            SectionAssignment::new(".debug_str", ".debug_str"),
            SectionAssignment::new(".debug_ranges", ".debug_ranges"),
            SectionAssignment::new(".debug_loc", ".debug_loc"),
            SectionAssignment::new(".debug_frame", ".debug_frame"),
            SectionAssignment::new(".debug_aranges", ".debug_aranges"),
            // Comment section (non-loadable)
            SectionAssignment::new(".comment", ".comment"),
            SectionAssignment::new(".note.*", ".note"),
        ];
    }

    /// Populates the default segment rules for PT_LOAD and other segments.
    fn build_default_segment_rules(&mut self) {
        let page = self.page_size;

        // PT_PHDR — program header table itself
        self.segment_rules.push(SegmentRule::new(
            PT_PHDR,
            PF_R,
            vec![],
            8, // 8-byte alignment for PHDR
        ));

        // PT_INTERP — dynamic linker path (only for dynamic executables/shared libs)
        if self.output_type == OutputType::SharedObject {
            self.segment_rules.push(SegmentRule::new(
                PT_INTERP,
                PF_R,
                vec![".interp".to_string()],
                1,
            ));
        }

        // PT_LOAD 0 — executable code (.text, .init, .fini)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R | PF_X,
            vec![".text".to_string()],
            page,
        ));

        // PT_LOAD 1 — read-only data (.rodata, .eh_frame)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R,
            vec![
                ".rodata".to_string(),
                ".eh_frame".to_string(),
                ".eh_frame_hdr".to_string(),
            ],
            page,
        ));

        // PT_LOAD 2 — read-write data (.data, .bss, .got, .got.plt)
        self.segment_rules.push(SegmentRule::new(
            PT_LOAD,
            PF_R | PF_W,
            vec![
                ".data".to_string(),
                ".bss".to_string(),
                ".init_array".to_string(),
                ".fini_array".to_string(),
                ".got".to_string(),
                ".got.plt".to_string(),
            ],
            page,
        ));

        // PT_DYNAMIC — dynamic linking metadata (shared objects only)
        if self.output_type == OutputType::SharedObject {
            self.segment_rules.push(SegmentRule::new(
                PT_DYNAMIC,
                PF_R | PF_W,
                vec![".dynamic".to_string()],
                8,
            ));
        }

        // PT_GNU_STACK — stack permissions (non-executable stack)
        self.segment_rules.push(SegmentRule::new(
            PT_GNU_STACK,
            PF_R | PF_W,
            vec![],
            0x10, // 16-byte alignment (conventional)
        ));
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Returns the entry point symbol name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Sets a custom entry point symbol name.
    pub fn set_entry_point(&mut self, name: &str) {
        self.entry_point = name.to_string();
    }

    /// Returns the base virtual address for the first loadable segment.
    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Sets a custom base virtual address.
    pub fn set_base_address(&mut self, addr: u64) {
        self.base_address = addr;
    }

    /// Returns the page size for alignment.
    pub fn page_size(&self) -> u64 {
        self.page_size
    }

    /// Returns the target architecture.
    pub fn target(&self) -> Target {
        self.target
    }

    /// Returns the output type.
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }

    /// Returns a reference to the section assignment rules.
    pub fn section_assignments(&self) -> &[SectionAssignment] {
        &self.section_assignments
    }

    /// Returns a reference to the segment rules.
    pub fn segment_rules(&self) -> &[SegmentRule] {
        &self.segment_rules
    }

    /// Resolves which output section an input section name maps to.
    ///
    /// Returns the output section name if a matching rule is found, or
    /// `None` if the input section should be discarded.
    pub fn resolve_section(&self, input_section_name: &str) -> Option<&str> {
        for assignment in &self.section_assignments {
            if assignment.matches(input_section_name) {
                return Some(&assignment.output_section);
            }
        }
        None
    }

    /// Returns the segment rule that contains the given output section.
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

    /// Returns the total number of program headers that will be generated.
    pub fn program_header_count(&self) -> usize {
        self.segment_rules.len()
    }

    /// Aligns a virtual address up to the page boundary.
    pub fn align_to_page(&self, addr: u64) -> u64 {
        let mask = self.page_size - 1;
        (addr + mask) & !mask
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_type_display() {
        assert_eq!(
            format!("{}", OutputType::Executable),
            "executable (ET_EXEC)"
        );
        assert_eq!(
            format!("{}", OutputType::SharedObject),
            "shared object (ET_DYN)"
        );
        assert_eq!(
            format!("{}", OutputType::Relocatable),
            "relocatable object (ET_REL)"
        );
    }

    #[test]
    fn test_section_assignment_exact_match() {
        let rule = SectionAssignment::new(".text", ".text");
        assert!(rule.matches(".text"));
        assert!(!rule.matches(".text.startup"));
        assert!(!rule.matches(".textual"));
    }

    #[test]
    fn test_section_assignment_wildcard() {
        let rule = SectionAssignment::new(".text.*", ".text");
        assert!(rule.matches(".text"));
        assert!(rule.matches(".text.startup"));
        assert!(rule.matches(".text.unlikely"));
        assert!(!rule.matches(".textual"));
    }

    #[test]
    fn test_segment_rule_contains_section() {
        let rule = SegmentRule::new(
            PT_LOAD,
            PF_R | PF_X,
            vec![".text".to_string(), ".init".to_string()],
            0x1000,
        );
        assert!(rule.contains_section(".text"));
        assert!(rule.contains_section(".init"));
        assert!(!rule.contains_section(".data"));
    }

    #[test]
    fn test_default_base_addresses() {
        assert_eq!(
            LinkerScript::default_base_address(Target::X86_64, OutputType::Executable),
            0x0040_0000
        );
        assert_eq!(
            LinkerScript::default_base_address(Target::I686, OutputType::Executable),
            0x0804_8000
        );
        assert_eq!(
            LinkerScript::default_base_address(Target::AArch64, OutputType::Executable),
            0x0040_0000
        );
        assert_eq!(
            LinkerScript::default_base_address(Target::RiscV64, OutputType::Executable),
            0x0001_0000
        );
        assert_eq!(
            LinkerScript::default_base_address(Target::X86_64, OutputType::SharedObject),
            0
        );
    }

    #[test]
    fn test_default_page_sizes() {
        assert_eq!(LinkerScript::default_page_size(Target::X86_64), 0x1000);
        assert_eq!(LinkerScript::default_page_size(Target::I686), 0x1000);
        assert_eq!(LinkerScript::default_page_size(Target::AArch64), 0x1000);
        assert_eq!(LinkerScript::default_page_size(Target::RiscV64), 0x1000);
    }

    #[test]
    fn test_linker_script_new() {
        let script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        assert_eq!(script.entry_point(), "_start");
        assert_eq!(script.base_address(), 0x0040_0000);
        assert_eq!(script.page_size(), 0x1000);
        assert!(!script.section_assignments().is_empty());
        assert!(!script.segment_rules().is_empty());
    }

    #[test]
    fn test_resolve_section() {
        let script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        assert_eq!(script.resolve_section(".text"), Some(".text"));
        assert_eq!(script.resolve_section(".text.startup"), Some(".text"));
        assert_eq!(script.resolve_section(".rodata"), Some(".rodata"));
        assert_eq!(script.resolve_section(".data"), Some(".data"));
        assert_eq!(script.resolve_section(".bss"), Some(".bss"));
        assert_eq!(script.resolve_section(".debug_info"), Some(".debug_info"));
    }

    #[test]
    fn test_segment_for_section() {
        let script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        let text_seg = script.segment_for_section(".text").unwrap();
        assert_eq!(text_seg.segment_type, PT_LOAD);
        assert_eq!(text_seg.flags, PF_R | PF_X);

        let data_seg = script.segment_for_section(".data").unwrap();
        assert_eq!(data_seg.segment_type, PT_LOAD);
        assert_eq!(data_seg.flags, PF_R | PF_W);
    }

    #[test]
    fn test_align_to_page() {
        let script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        assert_eq!(script.align_to_page(0x0), 0x0);
        assert_eq!(script.align_to_page(0x1), 0x1000);
        assert_eq!(script.align_to_page(0x1000), 0x1000);
        assert_eq!(script.align_to_page(0x1001), 0x2000);
    }

    #[test]
    fn test_shared_object_has_dynamic_segment() {
        let script = LinkerScript::new(Target::X86_64, OutputType::SharedObject);
        let dynamic_segs = script.segments_of_type(PT_DYNAMIC);
        assert!(!dynamic_segs.is_empty());
    }

    #[test]
    fn test_executable_no_dynamic_segment() {
        let script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        let dynamic_segs = script.segments_of_type(PT_DYNAMIC);
        assert!(dynamic_segs.is_empty());
    }

    #[test]
    fn test_set_entry_point() {
        let mut script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        script.set_entry_point("main");
        assert_eq!(script.entry_point(), "main");
    }

    #[test]
    fn test_set_base_address() {
        let mut script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        script.set_base_address(0x1000_0000);
        assert_eq!(script.base_address(), 0x1000_0000);
    }

    #[test]
    fn test_program_header_count() {
        let exe_script = LinkerScript::new(Target::X86_64, OutputType::Executable);
        let so_script = LinkerScript::new(Target::X86_64, OutputType::SharedObject);
        // Shared object has extra PT_INTERP and PT_DYNAMIC
        assert!(so_script.program_header_count() > exe_script.program_header_count());
    }
}
