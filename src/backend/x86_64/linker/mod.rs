//! x86-64 architecture-specific ELF linker module for the BCC standalone backend.
//!
//! This module replaces external `ld` per Section 0.7.7 (standalone backend mode).
//! It produces **ET_EXEC** static executables and **ET_DYN** shared objects for
//! the x86-64 (AMD64) target architecture.
//!
//! The linker orchestrates the full linking pipeline:
//! 1. Parse input assembled objects — extract sections, symbols, relocations.
//! 2. Two-pass symbol resolution (strong/weak binding) via `SymbolResolver`.
//! 3. Section merging (`.text`/`.rodata`/`.data`/`.bss` ordering) via `SectionMerger`.
//! 4. Relocation classification to determine GOT/PLT needs.
//! 5. GOT/PLT/dynamic section generation for PIC/shared library output.
//! 6. Virtual address and file offset assignment.
//! 7. Relocation application with the `X86_64RelocationHandler`.
//! 8. Program header construction based on the default linker script.
//! 9. Final ELF binary emission via `ElfWriter`.
//!
//! # x86-64 Specifics
//!
//! - PLT stubs are 16 bytes using RIP-relative addressing.
//! - GOT entries are 8 bytes (64-bit pointers).
//! - Default base address for ET_EXEC: `0x400000`.
//! - Dynamic linker path: `/lib64/ld-linux-x86-64.so.2`.
//! - Supports GOTPCRELX relaxation: `mov foo@GOTPCREL(%rip), %reg` →
//!   `lea foo(%rip), %reg` when the symbol is locally defined.

pub mod relocations;

// ---------------------------------------------------------------------------
// Imports from linker_common (re-exports from mod.rs)
// ---------------------------------------------------------------------------
use crate::backend::linker_common::{
    InputRelocation, InputSection, InputSymbol, LinkError, LinkerScript, OutputType,
    RelocationClassification, RelocationEntry, RelocationProcessor, ResolvedSymbols,
    SectionMerger, SymbolBinding, SymbolEntry, SymbolResolver, SymbolType, SymbolVisibility,
};

// Dynamic linking types — GotEntry and PltEntry are not re-exported from
// linker_common/mod.rs, so import directly from the dynamic submodule.
use crate::backend::linker_common::dynamic::{
    DynamicLayout, DynamicRelocation, DynamicSectionBuilder, DynamicSymbolTable, GotBuilder,
    GotEntry, PltBuilder, PltEntry,
};

// ---------------------------------------------------------------------------
// Imports from elf_writer_common
// ---------------------------------------------------------------------------
use crate::backend::elf_writer_common::{
    ElfSection, ElfSymbol, ElfWriter, ProgramHeader, ET_DYN, ET_EXEC, PF_R, PF_W, PT_GNU_STACK,
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_DYNAMIC, SHT_DYNSYM, SHT_PROGBITS, SHT_RELA,
    SHT_STRTAB, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC, STT_NOTYPE, STT_OBJECT, STV_DEFAULT,
    STV_HIDDEN, STV_PROTECTED,
};

// ---------------------------------------------------------------------------
// Imports from x86-64 backend modules
// ---------------------------------------------------------------------------
use crate::backend::x86_64::linker::relocations::X86_64RelocationHandler;
use crate::backend::x86_64::assembler::relocations::X86_64RelocationType;

// ---------------------------------------------------------------------------
// Imports from common
// ---------------------------------------------------------------------------
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// Local constants
// ---------------------------------------------------------------------------

/// ELF section type for `.gnu.hash` (GNU-style hash table).
/// Standard ELF value: `SHT_GNU_HASH = 0x6ffffff6`.
const SHT_GNU_HASH: u32 = 0x6fff_fff6;

/// Standard x86-64 base virtual address for ET_EXEC executables.
const X86_64_EXEC_BASE: u64 = 0x400000;

/// x86-64 page size (4 KiB). Used by address alignment calculations.
#[allow(dead_code)]
const PAGE_SIZE: u64 = 0x1000;

/// Size of each PLT entry in bytes on x86-64.
const PLT_ENTRY_SIZE: u64 = 16;

/// Size of the PLT[0] resolver stub on x86-64.
const PLT0_SIZE: u64 = 16;

/// Size of each GOT entry on x86-64 (8-byte pointer).
const GOT_ENTRY_SIZE: u64 = 8;

/// Number of reserved slots at the start of `.got.plt` (dynamic, link_map,
/// dl_runtime_resolve).
const GOT_PLT_RESERVED: usize = 3;

/// Dynamic linker path for x86-64 Linux.
const X86_64_INTERP: &str = "/lib64/ld-linux-x86-64.so.2";

// ===========================================================================
// LinkerConfig — configuration for the x86-64 linker
// ===========================================================================

/// Configuration parameters controlling x86-64 linker behaviour.
///
/// Passed to [`X86_64Linker::new`] to specify output type, paths, PIC mode,
/// debug info inclusion, and library search parameters.
#[derive(Clone, Debug)]
pub struct LinkerConfig {
    /// Output binary type: executable (`ET_EXEC`) or shared library (`ET_DYN`).
    pub output_type: OutputType,
    /// Filesystem path for the output binary.
    pub output_path: String,
    /// Entry point symbol name (default: `"_start"` for executables).
    pub entry_point: String,
    /// Library search directories (`-L` paths).
    pub library_paths: Vec<String>,
    /// Libraries to link against (`-l` names).
    pub linked_libs: Vec<String>,
    /// Whether position-independent code mode is active (`-fPIC`).
    pub pic: bool,
    /// Whether DWARF debug sections should be preserved in the output.
    pub debug_info: bool,
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self {
            output_type: OutputType::Executable,
            output_path: String::from("a.out"),
            entry_point: String::from("_start"),
            library_paths: Vec::new(),
            linked_libs: Vec::new(),
            pic: false,
            debug_info: false,
        }
    }
}

// ===========================================================================
// AssembledObject — input from the x86-64 assembler
// ===========================================================================

/// Represents a single `.o` file produced by the x86-64 assembler.
///
/// The linker consumes one or more of these objects and produces an ELF
/// executable or shared library.
#[derive(Clone, Debug)]
pub struct AssembledObject {
    /// Human-readable name of the source object (e.g., `"hello.o"`).
    pub name: String,
    /// Sections extracted from the object file.
    pub sections: Vec<ObjectSection>,
    /// Symbol table entries from the object file.
    pub symbols: Vec<InputSymbol>,
    /// Relocations referencing symbols and sections within the object.
    pub relocations: Vec<InputRelocation>,
}

// ===========================================================================
// ObjectSection — one section from an input object
// ===========================================================================

/// A single section from an assembled object file.
#[derive(Clone, Debug)]
pub struct ObjectSection {
    /// Section name (e.g., `.text`, `.data`, `.rodata`, `.bss`).
    pub name: String,
    /// ELF section type (`SHT_PROGBITS`, `SHT_NOBITS`, etc.).
    pub section_type: u32,
    /// ELF section flags (`SHF_ALLOC`, `SHF_WRITE`, `SHF_EXECINSTR`, etc.).
    pub flags: u64,
    /// Section content bytes. For `.bss`, length encodes the virtual size.
    pub data: Vec<u8>,
    /// Required alignment (power of two, minimum 1).
    pub alignment: u64,
}

// ===========================================================================
// X86_64Linker — the main linker struct
// ===========================================================================

/// x86-64 ELF linker that produces static executables (ET_EXEC) and shared
/// objects (ET_DYN) from assembled relocatable objects.
///
/// # Usage
///
/// ```ignore
/// let config = LinkerConfig {
///     output_type: OutputType::Executable,
///     output_path: "hello".into(),
///     entry_point: "_start".into(),
///     ..Default::default()
/// };
/// let mut linker = X86_64Linker::new(config);
/// let elf_bytes = linker.link(vec![assembled_obj])?;
/// std::fs::write("hello", elf_bytes).unwrap();
/// ```
pub struct X86_64Linker {
    /// Linker configuration.
    config: LinkerConfig,
    /// Two-pass symbol resolution engine.
    symbol_resolver: SymbolResolver,
    /// Section merging engine.
    section_merger: SectionMerger,
    /// Architecture-agnostic relocation processor.
    relocation_processor: RelocationProcessor,
    /// Diagnostic engine for error reporting.
    diagnostics: DiagnosticEngine,
}

impl X86_64Linker {
    /// Creates a new x86-64 linker with the given configuration.
    pub fn new(config: LinkerConfig) -> Self {
        let mut symbol_resolver = SymbolResolver::new();
        symbol_resolver.set_target(Target::X86_64);

        let mut relocation_processor = RelocationProcessor::new();
        relocation_processor.set_target(Target::X86_64);

        Self {
            config,
            symbol_resolver,
            section_merger: SectionMerger::new(),
            relocation_processor,
            diagnostics: DiagnosticEngine::new(),
        }
    }

    // =======================================================================
    // Primary link entry point
    // =======================================================================

    /// Links one or more assembled objects into an ELF executable or shared
    /// library.
    ///
    /// Returns the raw bytes of the output ELF file on success, or a list of
    /// [`LinkError`]s on failure (undefined symbols, multiple definitions,
    /// relocation overflows, etc.).
    pub fn link(
        &mut self,
        input_objects: Vec<AssembledObject>,
    ) -> Result<Vec<u8>, Vec<LinkError>> {
        // ------------------------------------------------------------------
        // Phase 1: Parse input objects — register symbols and sections
        // ------------------------------------------------------------------
        for (obj_idx, obj) in input_objects.iter().enumerate() {
            self.symbol_resolver
                .register_object(obj_idx, &obj.name);

            // Build per-object symbol name list for relocation resolution.
            let sym_names: Vec<String> =
                obj.symbols.iter().map(|s| s.name.clone()).collect();
            self.relocation_processor
                .register_object_symbols(obj_idx, sym_names);

            // Collect symbols into the resolver.
            self.symbol_resolver
                .collect_symbols(obj_idx, &obj.symbols);

            // Add each section to the merger, and collect its relocations.
            for (sec_idx, sec) in obj.sections.iter().enumerate() {
                let relocations_for_section: Vec<InputRelocation> = obj
                    .relocations
                    .iter()
                    .filter(|r| r.section_index == sec_idx)
                    .cloned()
                    .collect();

                let input_section = InputSection {
                    name: sec.name.clone(),
                    section_type: sec.section_type,
                    flags: sec.flags,
                    data: sec.data.clone(),
                    alignment: sec.alignment.max(1),
                    entry_size: 0,
                    group_id: None,
                    object_index: obj_idx,
                    original_index: sec_idx,
                    relocations: relocations_for_section,
                };
                self.section_merger.add_input_section(input_section);
            }
        }

        // ------------------------------------------------------------------
        // Phase 2: Resolve symbols (two-pass: collect → resolve)
        // ------------------------------------------------------------------
        let resolved = self.symbol_resolver.resolve_references().map_err(|errs| {
            self.symbol_resolver.emit_diagnostics(&mut self.diagnostics);
            errs
        })?;

        // ------------------------------------------------------------------
        // Phase 3: Compute section order (standard ELF layout)
        // ------------------------------------------------------------------
        self.section_merger.compute_section_order();

        // ------------------------------------------------------------------
        // Phase 4: Collect relocations from all merged sections
        // ------------------------------------------------------------------
        self.collect_all_relocations();

        // ------------------------------------------------------------------
        // Phase 5: Classify relocations for GOT/PLT needs
        // ------------------------------------------------------------------
        let reloc_handler = X86_64RelocationHandler::new();
        let classification = self.relocation_processor.classify_relocations(&reloc_handler);

        let needs_dynamic = matches!(self.config.output_type, OutputType::SharedLibrary)
            || self.config.pic
            || !classification.got_entries.is_empty()
            || !classification.plt_entries.is_empty();

        // ------------------------------------------------------------------
        // Phase 6 (Pass 1): Add placeholder sections for GOT/PLT/dynamic
        //
        // We must know how many GOT/PLT entries exist to compute section
        // sizes *before* assigning addresses. Insert placeholder sections
        // with the correct sizes but zero base addresses.
        // ------------------------------------------------------------------
        let got_entry_count = classification.got_entries.len();
        let plt_entry_count = classification.plt_entries.len();

        // Build lookup maps so we can assign GOT/PLT indices.
        let mut got_symbol_indices: FxHashMap<String, usize> = FxHashMap::default();
        for (i, name) in classification.got_entries.iter().enumerate() {
            got_symbol_indices.insert(name.clone(), i);
        }
        let mut plt_symbol_indices: FxHashMap<String, usize> = FxHashMap::default();
        for (i, name) in classification.plt_entries.iter().enumerate() {
            plt_symbol_indices.insert(name.clone(), i);
        }

        // Placeholder sections for dynamic linking artifacts.
        if needs_dynamic {
            self.add_dynamic_placeholder_sections(
                &classification,
                &resolved,
                got_entry_count,
                plt_entry_count,
            );
        }

        // Add .interp section for dynamically linked executables.
        let needs_interp = needs_dynamic
            && matches!(self.config.output_type, OutputType::Executable);
        if needs_interp {
            self.add_interp_section();
        }

        // ------------------------------------------------------------------
        // Phase 7: Assign virtual addresses and file offsets
        // ------------------------------------------------------------------
        let base_address = match self.config.output_type {
            OutputType::Executable => X86_64_EXEC_BASE,
            OutputType::SharedLibrary | OutputType::RelocatableObject => 0,
        };

        self.section_merger.assign_addresses(base_address);
        // Initial file offset after ELF header (64 bytes) + program headers.
        let estimated_phdr_count = self.estimate_program_header_count(needs_dynamic, needs_interp);
        let initial_offset = 64 + (estimated_phdr_count as u64) * 56;
        self.section_merger.assign_file_offsets(initial_offset);

        // ------------------------------------------------------------------
        // Phase 8 (Pass 2): Build GOT/PLT/dynamic data with real addresses
        // ------------------------------------------------------------------
        let mut dynamic_relocs_data: Vec<u8> = Vec::new();
        let mut plt_relocs_data: Vec<u8> = Vec::new();

        if needs_dynamic {
            self.build_dynamic_sections(
                &classification,
                &resolved,
                &mut dynamic_relocs_data,
                &mut plt_relocs_data,
            );
        }

        // ------------------------------------------------------------------
        // Phase 9: Apply relocations
        // ------------------------------------------------------------------
        let got_address = self.find_section_addr(".got").unwrap_or(0);
        let plt_address = self.find_section_addr(".plt").unwrap_or(0);

        let sections = self.section_merger.output_sections_mut();
        if let Err(reloc_errors) = self.relocation_processor.apply_relocations(
            &reloc_handler,
            &resolved,
            sections,
            got_address,
            plt_address,
        ) {
            let mut link_errors = Vec::new();
            for re in &reloc_errors {
                link_errors.push(LinkError::UndefinedSymbol {
                    name: format!("relocation error: {}", re),
                    referenced_by: Vec::new(),
                });
            }
            return Err(link_errors);
        }

        // ------------------------------------------------------------------
        // Phase 10: Build ELF output
        // ------------------------------------------------------------------
        let elf_bytes = self.build_elf(&resolved, needs_dynamic, needs_interp);

        Ok(elf_bytes)
    }

    // =======================================================================
    // Internal helpers — relocation collection
    // =======================================================================

    /// Collects relocations from all merged output sections into the
    /// relocation processor, translating input-section-relative offsets to
    /// output-section-relative offsets.
    fn collect_all_relocations(&mut self) {
        let sections = self.section_merger.output_sections();
        for (out_idx, out_sec) in sections.iter().enumerate() {
            for merged in &out_sec.input_sections {
                if merged.input.relocations.is_empty() {
                    continue;
                }
                self.relocation_processor.collect_relocations(
                    merged.input.object_index,
                    merged.input.original_index,
                    &merged.input.relocations,
                    out_idx,
                    merged.offset_in_output,
                );
            }
        }
    }

    // =======================================================================
    // Internal helpers — dynamic placeholder section insertion
    // =======================================================================

    /// Inserts placeholder sections (`.got`, `.got.plt`, `.plt`, `.dynsym`,
    /// `.dynstr`, `.gnu.hash`, `.rela.dyn`, `.rela.plt`, `.dynamic`) into the
    /// section merger with the correct sizes computed from entry counts.
    ///
    /// The *data* of these sections is empty placeholder bytes. After addresses
    /// are assigned, [`build_dynamic_sections`] overwrites the data.
    fn add_dynamic_placeholder_sections(
        &mut self,
        _classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
        got_count: usize,
        plt_count: usize,
    ) {
        // Compute sizes.
        let got_size = got_count * GOT_ENTRY_SIZE as usize;
        let got_plt_size = (GOT_PLT_RESERVED + plt_count) * GOT_ENTRY_SIZE as usize;
        let plt_total_size = if plt_count > 0 {
            (PLT0_SIZE + plt_count as u64 * PLT_ENTRY_SIZE) as usize
        } else {
            0
        };

        // Build temporary dynamic symbol table to compute .dynsym/.dynstr sizes.
        let mut dsym_tmp = DynamicSymbolTable::new();
        for sym in &resolved.symbols {
            dsym_tmp.add_symbol(sym);
        }
        let dynsym_data = dsym_tmp.build_dynsym();
        let dynstr_data = dsym_tmp.build_dynstr();
        let gnu_hash_data = dsym_tmp.build_gnu_hash();

        // Compute rela sizes. Each RELA entry is 24 bytes on x86-64.
        // .rela.dyn: GOT entries that need R_X86_64_GLOB_DAT
        // .rela.plt: PLT entries that need R_X86_64_JUMP_SLOT
        let rela_dyn_size = got_count * 24;
        let rela_plt_size = plt_count * 24;

        // .dynamic section size — estimate generously (20 entries × 16 bytes).
        let dynamic_size = 20 * 16;

        // Add .got section (writable data for GOT data entries).
        if got_size > 0 {
            self.section_merger.add_input_section(InputSection {
                name: ".got".into(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                data: vec![0u8; got_size],
                alignment: 8,
                entry_size: 8,
                group_id: None,
                object_index: usize::MAX,
                original_index: 0,
                relocations: Vec::new(),
            });
        }

        // Add .got.plt section (GOT entries for PLT lazy binding).
        if got_plt_size > 0 {
            self.section_merger.add_input_section(InputSection {
                name: ".got.plt".into(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                data: vec![0u8; got_plt_size],
                alignment: 8,
                entry_size: 8,
                group_id: None,
                object_index: usize::MAX,
                original_index: 0,
                relocations: Vec::new(),
            });
        }

        // Add .plt section (executable code stubs).
        if plt_total_size > 0 {
            self.section_merger.add_input_section(InputSection {
                name: ".plt".into(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                data: vec![0u8; plt_total_size],
                alignment: 16,
                entry_size: 0,
                group_id: None,
                object_index: usize::MAX,
                original_index: 0,
                relocations: Vec::new(),
            });
        }

        // Add .dynsym section.
        self.section_merger.add_input_section(InputSection {
            name: ".dynsym".into(),
            section_type: SHT_DYNSYM,
            flags: SHF_ALLOC,
            data: vec![0u8; dynsym_data.len()],
            alignment: 8,
            entry_size: 24,
            group_id: None,
            object_index: usize::MAX,
            original_index: 0,
            relocations: Vec::new(),
        });

        // Add .dynstr section.
        self.section_merger.add_input_section(InputSection {
            name: ".dynstr".into(),
            section_type: SHT_STRTAB,
            flags: SHF_ALLOC,
            data: vec![0u8; dynstr_data.len()],
            alignment: 1,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: 0,
            relocations: Vec::new(),
        });

        // Add .gnu.hash section.
        self.section_merger.add_input_section(InputSection {
            name: ".gnu.hash".into(),
            section_type: SHT_GNU_HASH,
            flags: SHF_ALLOC,
            data: vec![0u8; gnu_hash_data.len()],
            alignment: 8,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: 0,
            relocations: Vec::new(),
        });

        // Add .rela.dyn section.
        if rela_dyn_size > 0 {
            self.section_merger.add_input_section(InputSection {
                name: ".rela.dyn".into(),
                section_type: SHT_RELA,
                flags: SHF_ALLOC,
                data: vec![0u8; rela_dyn_size],
                alignment: 8,
                entry_size: 24,
                group_id: None,
                object_index: usize::MAX,
                original_index: 0,
                relocations: Vec::new(),
            });
        }

        // Add .rela.plt section.
        if rela_plt_size > 0 {
            self.section_merger.add_input_section(InputSection {
                name: ".rela.plt".into(),
                section_type: SHT_RELA,
                flags: SHF_ALLOC,
                data: vec![0u8; rela_plt_size],
                alignment: 8,
                entry_size: 24,
                group_id: None,
                object_index: usize::MAX,
                original_index: 0,
                relocations: Vec::new(),
            });
        }

        // Add .dynamic section.
        self.section_merger.add_input_section(InputSection {
            name: ".dynamic".into(),
            section_type: SHT_DYNAMIC,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0u8; dynamic_size],
            alignment: 8,
            entry_size: 16,
            group_id: None,
            object_index: usize::MAX,
            original_index: 0,
            relocations: Vec::new(),
        });
    }

    /// Adds the `.interp` section for dynamically linked executables.
    fn add_interp_section(&mut self) {
        let mut interp_data = X86_64_INTERP.as_bytes().to_vec();
        interp_data.push(0); // NUL terminator
        self.section_merger.add_input_section(InputSection {
            name: ".interp".into(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC,
            data: interp_data,
            alignment: 1,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: 0,
            relocations: Vec::new(),
        });
    }

    // =======================================================================
    // Internal helpers — build dynamic section data (Pass 2)
    // =======================================================================

    /// Builds the actual data for all dynamic linking sections using the
    /// addresses assigned by the section merger.
    ///
    /// Overwrites the placeholder data in `.got`, `.got.plt`, `.plt`,
    /// `.dynsym`, `.dynstr`, `.gnu.hash`, `.rela.dyn`, `.rela.plt`, and
    /// `.dynamic` with correctly computed bytes.
    fn build_dynamic_sections(
        &mut self,
        classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
        dynamic_relocs_out: &mut Vec<u8>,
        plt_relocs_out: &mut Vec<u8>,
    ) {
        // Retrieve assigned addresses for all dynamic sections.
        let got_addr = self.find_section_addr(".got").unwrap_or(0);
        let got_plt_addr = self.find_section_addr(".got.plt").unwrap_or(0);
        let plt_addr = self.find_section_addr(".plt").unwrap_or(0);
        let dynamic_addr = self.find_section_addr(".dynamic").unwrap_or(0);
        let dynsym_addr = self.find_section_addr(".dynsym").unwrap_or(0);
        let dynstr_addr = self.find_section_addr(".dynstr").unwrap_or(0);
        let gnu_hash_addr = self.find_section_addr(".gnu.hash").unwrap_or(0);
        let rela_dyn_addr = self.find_section_addr(".rela.dyn").unwrap_or(0);
        let rela_plt_addr = self.find_section_addr(".rela.plt").unwrap_or(0);
        let interp_addr = self.find_section_addr(".interp").unwrap_or(0);

        let target = Target::X86_64;

        // ----- Build .dynsym / .dynstr / .gnu.hash -----
        let mut dsym_table = DynamicSymbolTable::new();
        for sym in &resolved.symbols {
            dsym_table.add_symbol(sym);
        }
        let dynsym_data = dsym_table.build_dynsym();
        let dynstr_data = dsym_table.build_dynstr();
        let gnu_hash_data = dsym_table.build_gnu_hash();

        // ----- Build GOT -----
        let mut got_builder = GotBuilder::new(got_addr, got_plt_addr, dynamic_addr, &target);

        // Add data GOT entries for symbols requiring GOT slots.
        let mut got_offsets: FxHashMap<String, u64> = FxHashMap::default();
        for name in &classification.got_entries {
            let initial_value = resolved.get_symbol_value(name).unwrap_or(0);
            let offset = got_builder.add_entry(GotEntry {
                symbol_name: name.clone(),
                offset: 0, // filled by add_entry
                initial_value,
            });
            got_offsets.insert(name.clone(), offset);
        }

        // Add PLT GOT entries (.got.plt) for symbols requiring PLT stubs.
        let mut plt_got_addrs: FxHashMap<String, u64> = FxHashMap::default();
        for (i, name) in classification.plt_entries.iter().enumerate() {
            // Initial value: address of PLT[N] push instruction = PLT[N] + 6.
            let plt_n_addr = plt_addr + PLT0_SIZE + (i as u64) * PLT_ENTRY_SIZE;
            let push_addr = plt_n_addr + 6;
            let got_entry_addr = got_builder.add_plt_entry(GotEntry {
                symbol_name: name.clone(),
                offset: 0,
                initial_value: push_addr,
            });
            plt_got_addrs.insert(name.clone(), got_entry_addr);
        }

        let got_data = got_builder.build_got();
        let got_plt_data = got_builder.build_got_plt();

        // ----- Build PLT -----
        let mut plt_builder = PltBuilder::new(plt_addr, got_plt_addr, target);
        for (i, name) in classification.plt_entries.iter().enumerate() {
            let got_offset = (GOT_PLT_RESERVED + i) as u64 * GOT_ENTRY_SIZE;
            plt_builder.add_entry(PltEntry {
                symbol_name: name.clone(),
                got_offset,
                plt_index: i as u32,
            });
        }
        let plt_data = plt_builder.build_plt(&target);

        // ----- Build .rela.dyn (GLOB_DAT relocations) -----
        let mut rela_dyn_bytes: Vec<u8> = Vec::new();
        for (i, name) in classification.got_entries.iter().enumerate() {
            // Find the dynamic symbol index for this name.
            let dynsym_index = self.find_dynsym_index(&dsym_table, name);
            let entry_addr = got_addr + (i as u64) * GOT_ENTRY_SIZE;
            let reloc = DynamicRelocation {
                offset: entry_addr,
                reloc_type: X86_64RelocationType::R_X86_64_GLOB_DAT.elf_value(),
                symbol_index: dynsym_index,
                addend: 0,
            };
            rela_dyn_bytes.extend_from_slice(&reloc.to_bytes_64_le());
        }

        // ----- Build .rela.plt (JUMP_SLOT relocations) -----
        let mut rela_plt_bytes: Vec<u8> = Vec::new();
        for (i, name) in classification.plt_entries.iter().enumerate() {
            let dynsym_index = self.find_dynsym_index(&dsym_table, name);
            let got_entry_addr =
                got_plt_addr + (GOT_PLT_RESERVED as u64 + i as u64) * GOT_ENTRY_SIZE;
            let reloc = DynamicRelocation {
                offset: got_entry_addr,
                reloc_type: X86_64RelocationType::R_X86_64_JUMP_SLOT.elf_value(),
                symbol_index: dynsym_index,
                addend: 0,
            };
            rela_plt_bytes.extend_from_slice(&reloc.to_bytes_64_le());
        }

        // ----- Build .dynamic section -----
        let layout = DynamicLayout {
            dynamic_addr,
            dynsym_addr,
            dynstr_addr,
            gnu_hash_addr,
            got_addr,
            got_plt_addr,
            plt_addr,
            rela_dyn_addr,
            rela_plt_addr,
            interp_addr,
        };

        let mut dyn_builder = DynamicSectionBuilder::new();
        for lib in &self.config.linked_libs {
            dyn_builder.add_needed(lib);
        }
        dyn_builder.set_dynstr_size(dynstr_data.len() as u64);
        dyn_builder.set_rela_dyn_size(rela_dyn_bytes.len() as u64);
        dyn_builder.set_rela_plt_size(rela_plt_bytes.len() as u64);
        let dynamic_data = dyn_builder.build(&layout);

        *dynamic_relocs_out = rela_dyn_bytes.clone();
        *plt_relocs_out = rela_plt_bytes.clone();

        // ----- Overwrite placeholder section data -----
        self.overwrite_section_data(".got", &got_data);
        self.overwrite_section_data(".got.plt", &got_plt_data);
        self.overwrite_section_data(".plt", &plt_data);
        self.overwrite_section_data(".dynsym", &dynsym_data);
        self.overwrite_section_data(".dynstr", &dynstr_data);
        self.overwrite_section_data(".gnu.hash", &gnu_hash_data);
        self.overwrite_section_data(".rela.dyn", &rela_dyn_bytes);
        self.overwrite_section_data(".rela.plt", &rela_plt_bytes);
        self.overwrite_section_data(".dynamic", &dynamic_data);
    }

    // =======================================================================
    // Internal helpers — ELF output construction
    // =======================================================================

    /// Constructs the final ELF binary from the merged sections, resolved
    /// symbols, and program headers.
    fn build_elf(
        &self,
        resolved: &ResolvedSymbols,
        _needs_dynamic: bool,
        _needs_interp: bool,
    ) -> Vec<u8> {
        let mut writer = ElfWriter::new(Target::X86_64);

        // Set ELF type.
        let elf_type = match self.config.output_type {
            OutputType::Executable => ET_EXEC,
            OutputType::SharedLibrary => ET_DYN,
            OutputType::RelocatableObject => ET_EXEC, // shouldn't happen at link stage
        };
        writer.set_type(elf_type);

        // Set entry point.
        if matches!(self.config.output_type, OutputType::Executable) {
            let entry_addr = resolved
                .get_symbol_value(&self.config.entry_point)
                .unwrap_or(0);
            writer.set_entry_point(entry_addr);
        }

        // Add output sections to the ELF writer.
        let sections = self.section_merger.output_sections();
        let mut section_index_map: FxHashMap<String, usize> = FxHashMap::default();

        for out_sec in sections {
            let section_data = self.section_merger.collect_section_data(
                self.section_merger
                    .find_section(&out_sec.name)
                    .unwrap_or(0),
            );

            let mut elf_sec = ElfSection::new(&out_sec.name, out_sec.section_type);
            elf_sec.flags = out_sec.flags;
            elf_sec.data = section_data;
            elf_sec.alignment = out_sec.alignment;
            elf_sec.addr = out_sec.addr;
            elf_sec.entry_size = out_sec.entry_size;

            // Wire up .dynsym link → .dynstr
            if out_sec.name == ".dynsym" {
                if let Some(&dynstr_idx) = section_index_map.get(".dynstr") {
                    elf_sec.link = dynstr_idx as u32;
                }
            }

            let idx = writer.add_section(elf_sec);
            section_index_map.insert(out_sec.name.clone(), idx);
        }

        // Add symbols to the ELF writer.
        for sym in &resolved.symbols {
            let binding = match sym.binding {
                SymbolBinding::Local => STB_LOCAL,
                SymbolBinding::Global => STB_GLOBAL,
                SymbolBinding::Weak => STB_WEAK,
            };
            let sym_type = match sym.sym_type {
                SymbolType::NoType => STT_NOTYPE,
                SymbolType::Func => STT_FUNC,
                SymbolType::Object => STT_OBJECT,
                SymbolType::Section => STT_NOTYPE,
                SymbolType::File => STT_NOTYPE,
            };
            let visibility = match sym.visibility {
                SymbolVisibility::Default => STV_DEFAULT,
                SymbolVisibility::Hidden => STV_HIDDEN,
                SymbolVisibility::Protected => STV_PROTECTED,
            };

            let elf_sym = ElfSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding,
                sym_type,
                visibility,
                section_index: sym.section_index,
            };
            writer.add_symbol(elf_sym);
        }

        // Build program headers using the linker script.
        let linker_script =
            LinkerScript::default_for_target(&Target::X86_64, self.config.output_type);
        let program_headers = linker_script.compute_segment_layout(sections);

        for phdr in &program_headers {
            writer.add_program_header(phdr.clone());
        }

        // Add PT_GNU_STACK (non-executable stack).
        let gnu_stack = ProgramHeader {
            p_type: PT_GNU_STACK,
            p_flags: PF_R | PF_W,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 16,
        };
        writer.add_program_header(gnu_stack);

        writer.write()
    }

    // =======================================================================
    // Internal helpers — utility methods
    // =======================================================================

    /// Looks up the virtual address assigned to an output section by name.
    fn find_section_addr(&self, name: &str) -> Option<u64> {
        let sections = self.section_merger.output_sections();
        for sec in sections {
            if sec.name == name {
                return Some(sec.addr);
            }
        }
        None
    }

    /// Looks up the size of an output section by name.
    #[allow(dead_code)]
    fn find_section_size(&self, name: &str) -> Option<u64> {
        let sections = self.section_merger.output_sections();
        for sec in sections {
            if sec.name == name {
                return Some(sec.size);
            }
        }
        None
    }

    /// Overwrites the data of the first input section in an output section.
    /// Used to replace placeholder bytes with computed dynamic linking data.
    fn overwrite_section_data(&mut self, section_name: &str, data: &[u8]) {
        let sections = self.section_merger.output_sections_mut();
        for sec in sections.iter_mut() {
            if sec.name == section_name {
                if let Some(merged) = sec.input_sections.first_mut() {
                    // Resize data buffer to match the new data, preserving the
                    // section size metadata.
                    merged.input.data = data.to_vec();
                }
                // Update the output section size.
                sec.size = data.len() as u64;
                return;
            }
        }
    }

    /// Finds the 1-based index of a symbol in the dynamic symbol table.
    /// Returns 0 (STN_UNDEF) if the symbol is not found.
    fn find_dynsym_index(&self, dsym: &DynamicSymbolTable, name: &str) -> u32 {
        for (i, sym) in dsym.symbols().iter().enumerate() {
            if sym.name == name {
                return i as u32;
            }
        }
        0
    }

    /// Estimates the number of program headers for initial file offset
    /// calculation.
    fn estimate_program_header_count(&self, needs_dynamic: bool, needs_interp: bool) -> usize {
        // Baseline: PT_PHDR + PT_LOAD (text) + PT_LOAD (rodata) +
        //           PT_LOAD (data) + PT_GNU_STACK
        let mut count: usize = 5;
        if needs_interp {
            count += 1; // PT_INTERP
        }
        if needs_dynamic {
            count += 1; // PT_DYNAMIC
            count += 1; // PT_GNU_RELRO (for .got.plt)
        }
        count
    }

    /// Attempts a GOTPCRELX relaxation: transforms a MOV through the GOT
    /// into a LEA of the symbol's direct address when the symbol is locally
    /// defined in the same output module.
    ///
    /// Returns `true` if the relaxation was applied.
    ///
    /// The relaxation converts:
    ///   `mov foo@GOTPCREL(%rip), %reg`  (opcode `0x8b`)
    /// to:
    ///   `lea foo(%rip), %reg`            (opcode `0x8d`)
    ///
    /// This eliminates the GOT indirection for locally-defined symbols.
    pub fn try_relax_gotpcrelx(
        reloc: &RelocationEntry,
        symbol: &SymbolEntry,
        code: &mut [u8],
    ) -> bool {
        // Only relax if the symbol is defined locally with default visibility.
        if !symbol.is_defined {
            return false;
        }
        if !matches!(symbol.visibility, SymbolVisibility::Default | SymbolVisibility::Protected) {
            return false;
        }

        // Check relocation type is GOTPCRELX or REX_GOTPCRELX.
        let is_gotpcrelx = reloc.reloc_type
            == X86_64RelocationType::R_X86_64_GOTPCRELX.elf_value();
        let is_rex_gotpcrelx = reloc.reloc_type
            == X86_64RelocationType::R_X86_64_REX_GOTPCRELX.elf_value();

        if !is_gotpcrelx && !is_rex_gotpcrelx {
            return false;
        }

        let offset = reloc.offset as usize;

        // The relocation applies to a 32-bit field at `offset`. The MOV
        // instruction has opcode byte at offset-2 (0x8b for MOV r, r/m).
        // We change it to 0x8d (LEA r, m).
        if offset < 2 || offset + 4 > code.len() {
            return false;
        }

        let opcode_offset = offset - 2;

        // Verify the opcode is MOV (0x8b).
        if code[opcode_offset] != 0x8b {
            return false;
        }

        // Verify the ModR/M byte indicates RIP-relative addressing (mod=00, r/m=101).
        let modrm = code[opcode_offset + 1];
        let mod_field = (modrm >> 6) & 0x3;
        let rm_field = modrm & 0x7;
        if mod_field != 0 || rm_field != 5 {
            return false;
        }

        // Transform MOV → LEA.
        code[opcode_offset] = 0x8d;

        // Recompute the displacement: instead of pointing to the GOT entry,
        // point directly to the symbol.
        // New disp32 = symbol_value + addend - (relocation_address + 4)
        // where relocation_address is the address of the disp32 field.
        // The caller handles patching the displacement value in the relocation
        // application pass; we only change the opcode here.

        true
    }
}

// ===========================================================================
// PLT/GOT stub generation helpers (standalone functions for testing)
// ===========================================================================

/// Generates a single PLT[N] stub (16 bytes) for x86-64.
///
/// Layout:
/// ```text
/// ff 25 XX XX XX XX   jmp *(%rip + GOT_offset)     ; indirect jump through GOT
/// 68 YY YY YY YY      push $reloc_index             ; push relocation index
/// e9 ZZ ZZ ZZ ZZ      jmp PLT[0]                    ; jump to resolver
/// ```
///
/// The `got_offset` is the RIP-relative displacement from this instruction
/// to the GOT entry. `reloc_index` is the index into `.rela.plt`.
pub fn generate_plt_stub(got_offset: i32, reloc_index: u32) -> [u8; 16] {
    let mut stub = [0u8; 16];

    // jmp *(%rip + disp32) — FF 25 <disp32>
    stub[0] = 0xff;
    stub[1] = 0x25;
    stub[2..6].copy_from_slice(&got_offset.to_le_bytes());

    // push $reloc_index — 68 <imm32>
    stub[6] = 0x68;
    stub[7..11].copy_from_slice(&reloc_index.to_le_bytes());

    // jmp PLT[0] — E9 <disp32> (filled with placeholder; linker patches this)
    stub[11] = 0xe9;
    // disp32 is relative to (this instruction address + 5) back to PLT[0].
    // This will be patched by the PltBuilder.
    stub[12..16].copy_from_slice(&0i32.to_le_bytes());

    stub
}

/// Generates the PLT[0] resolver stub (16 bytes) for x86-64.
///
/// Layout:
/// ```text
/// ff 35 XX XX XX XX   push *(%rip + GOT[1])   ; push &link_map
/// ff 25 YY YY YY YY   jmp  *(%rip + GOT[2])   ; jump to _dl_runtime_resolve
/// 0f 1f 40 00          nop DWORD PTR [rax+0]   ; 4-byte NOP padding
/// ```
pub fn generate_plt0_stub(got_plus_8: i32, got_plus_16: i32) -> [u8; 16] {
    let mut stub = [0u8; 16];

    // push *(%rip + GOT[1]) — FF 35 <disp32>
    stub[0] = 0xff;
    stub[1] = 0x35;
    stub[2..6].copy_from_slice(&got_plus_8.to_le_bytes());

    // jmp *(%rip + GOT[2]) — FF 25 <disp32>
    stub[6] = 0xff;
    stub[7] = 0x25;
    stub[8..12].copy_from_slice(&got_plus_16.to_le_bytes());

    // 4-byte NOP (0F 1F 40 00)
    stub[12] = 0x0f;
    stub[13] = 0x1f;
    stub[14] = 0x40;
    stub[15] = 0x00;

    stub
}

/// Generates GOT entry data from a list of symbol names and their initial
/// values. Each entry is 8 bytes (64-bit pointer, little-endian).
///
/// For lazy-bound PLT entries, `plt_address` is used to compute the initial
/// GOT value: `PLT[N] + 6` (pointing to the push instruction in the PLT
/// stub).
pub fn generate_got_entries(got_symbols: &[String], plt_address: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(got_symbols.len() * 8);
    for (i, _name) in got_symbols.iter().enumerate() {
        // Initial value for lazy binding: point back into PLT push insn.
        let plt_push_addr = plt_address + PLT0_SIZE + (i as u64) * PLT_ENTRY_SIZE + 6;
        data.extend_from_slice(&plt_push_addr.to_le_bytes());
    }
    data
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linker_config_default() {
        let cfg = LinkerConfig::default();
        assert_eq!(cfg.entry_point, "_start");
        assert!(!cfg.pic);
        assert!(!cfg.debug_info);
        assert!(matches!(cfg.output_type, OutputType::Executable));
    }

    #[test]
    fn test_plt0_stub_encoding() {
        // PLT[0] at 0x401000, GOT.PLT at 0x403000
        // GOT[1] = 0x403008, GOT[2] = 0x403010
        let plt0_addr: i64 = 0x401000;
        let got1_addr: i64 = 0x403008;
        let got2_addr: i64 = 0x403010;

        // push disp = GOT[1] - (PLT[0] + 6)
        let push_disp = (got1_addr - (plt0_addr + 6)) as i32;
        // jmp disp = GOT[2] - (PLT[0] + 12)
        let jmp_disp = (got2_addr - (plt0_addr + 12)) as i32;

        let stub = generate_plt0_stub(push_disp, jmp_disp);
        assert_eq!(stub[0], 0xff);
        assert_eq!(stub[1], 0x35);
        assert_eq!(stub[6], 0xff);
        assert_eq!(stub[7], 0x25);
        assert_eq!(stub[12], 0x0f); // NOP prefix
        assert_eq!(stub[13], 0x1f);
        assert_eq!(stub.len(), 16);
    }

    #[test]
    fn test_plt_stub_encoding() {
        let stub = generate_plt_stub(0x1000, 0);
        assert_eq!(stub[0], 0xff); // jmp opcode
        assert_eq!(stub[1], 0x25); // ModR/M for [rip+disp32]
        assert_eq!(stub[6], 0x68); // push imm32
        assert_eq!(stub[11], 0xe9); // jmp rel32
        assert_eq!(stub.len(), 16);
    }

    #[test]
    fn test_generate_got_entries() {
        let symbols = vec!["foo".to_string(), "bar".to_string()];
        let plt_addr: u64 = 0x401000;
        let data = generate_got_entries(&symbols, plt_addr);
        assert_eq!(data.len(), 16); // 2 entries × 8 bytes

        // First entry: PLT[0] size (16) + 0 * 16 + 6 = plt_addr + 22
        let val0 = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(val0, plt_addr + PLT0_SIZE + 6);

        // Second entry: plt_addr + 16 + 1*16 + 6 = plt_addr + 38
        let val1 = u64::from_le_bytes(data[8..16].try_into().unwrap());
        assert_eq!(val1, plt_addr + PLT0_SIZE + PLT_ENTRY_SIZE + 6);
    }

    #[test]
    fn test_linker_new() {
        let config = LinkerConfig {
            output_type: OutputType::SharedLibrary,
            output_path: "libfoo.so".into(),
            entry_point: String::new(),
            library_paths: vec!["/usr/lib".into()],
            linked_libs: vec!["c".into()],
            pic: true,
            debug_info: false,
        };
        let linker = X86_64Linker::new(config);
        assert_eq!(linker.config.output_path, "libfoo.so");
        assert!(linker.config.pic);
    }

    #[test]
    fn test_assembled_object_creation() {
        let obj = AssembledObject {
            name: "test.o".into(),
            sections: vec![ObjectSection {
                name: ".text".into(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                data: vec![0xc3], // ret
                alignment: 16,
            }],
            symbols: Vec::new(),
            relocations: Vec::new(),
        };
        assert_eq!(obj.name, "test.o");
        assert_eq!(obj.sections.len(), 1);
        assert_eq!(obj.sections[0].data, vec![0xc3]);
    }

    #[test]
    fn test_gotpcrelx_relaxation_non_mov() {
        // Code buffer with a non-MOV opcode at offset-2.
        let mut code = vec![0x00; 16];
        code[2] = 0x90; // NOP, not MOV
        code[3] = 0x05; // Arbitrary ModR/M

        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: X86_64RelocationType::R_X86_64_GOTPCRELX.elf_value(),
            symbol_name: "foo".into(),
            symbol_value: 0x1000,
            addend: -4,
            output_section: 0,
        };
        let sym = SymbolEntry {
            name: "foo".into(),
            value: 0x1000,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };

        let relaxed = X86_64Linker::try_relax_gotpcrelx(&reloc, &sym, &mut code);
        assert!(!relaxed); // 0x90 != 0x8b
    }

    #[test]
    fn test_gotpcrelx_relaxation_success() {
        // Simulate: MOV with RIP-relative addressing [rip+disp32]
        // At offset 2: opcode 0x8b, ModR/M 0x05 (mod=00, reg=rax, rm=101)
        let mut code = vec![0x00; 16];
        code[2] = 0x8b; // MOV opcode
        code[3] = 0x05; // ModR/M: mod=00, reg=000 (rax), rm=101 (RIP-relative)
        // disp32 at offset 4..8
        code[4] = 0x10;
        code[5] = 0x00;
        code[6] = 0x00;
        code[7] = 0x00;

        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: X86_64RelocationType::R_X86_64_GOTPCRELX.elf_value(),
            symbol_name: "foo".into(),
            symbol_value: 0x2000,
            addend: -4,
            output_section: 0,
        };
        let sym = SymbolEntry {
            name: "foo".into(),
            value: 0x2000,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };

        let relaxed = X86_64Linker::try_relax_gotpcrelx(&reloc, &sym, &mut code);
        assert!(relaxed);
        // Verify opcode changed from MOV (0x8b) to LEA (0x8d).
        assert_eq!(code[2], 0x8d);
    }

    #[test]
    fn test_gotpcrelx_relaxation_undefined_symbol() {
        let mut code = vec![0x00; 16];
        code[2] = 0x8b;
        code[3] = 0x05;

        let reloc = RelocationEntry {
            offset: 4,
            reloc_type: X86_64RelocationType::R_X86_64_GOTPCRELX.elf_value(),
            symbol_name: "bar".into(),
            symbol_value: 0,
            addend: -4,
            output_section: 0,
        };
        let sym = SymbolEntry {
            name: "bar".into(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 0,
            defining_object: 0,
            is_defined: false, // Not defined → no relaxation
        };

        let relaxed = X86_64Linker::try_relax_gotpcrelx(&reloc, &sym, &mut code);
        assert!(!relaxed);
        assert_eq!(code[2], 0x8b); // Unchanged
    }
}
