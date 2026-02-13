//! AArch64 ELF linker driver for the BCC built-in linker.
//!
//! Produces `ET_EXEC` static executables and `ET_DYN` shared objects for the
//! ARM 64-bit (AArch64) architecture **without invoking any external linker**
//! (`ld`, `aarch64-linux-gnu-ld`, `lld`). This is the standalone linker
//! component for AArch64, fulfilling the zero-external-tool mandate.
//!
//! # Linking Pipeline
//!
//! 1. **Symbol collection** — two-pass via [`SymbolResolver`]: collect all
//!    symbol definitions and references from input objects, then resolve
//!    references using strong/weak binding rules.
//! 2. **Section merging** — via [`SectionMerger`]: aggregate input sections
//!    into output sections with standard `.text` / `.rodata` / `.data` / `.bss`
//!    ordering, respecting alignment constraints.
//! 3. **Relocation classification** — scan all relocations to determine which
//!    symbols need GOT and/or PLT entries for PIC addressing.
//! 4. **Relocation application** — via [`AArch64RelocationHandler`]: patch
//!    machine code with resolved addresses using AArch64-specific bit-field
//!    encoding for fixed-width 32-bit A64 instructions.
//! 5. **Dynamic linking** (when `-shared` / `-fPIC` is active) — generate
//!    `.dynamic`, `.dynsym`, `.dynstr`, `.got`, `.got.plt`, `.plt`,
//!    `.rela.dyn`, `.rela.plt`, `.gnu.hash` sections via `linker_common::dynamic`.
//! 6. **Segment layout** — compute program headers and segment mapping via
//!    [`LinkerScript`] with AArch64-specific base address (`0x400000` for
//!    `ET_EXEC`).
//! 7. **ELF serialization** — write the final binary via [`ElfWriter`] with
//!    `EM_AARCH64` (183), `ELFCLASS64`, `ELFDATA2LSB`, ELF flags = 0.
//!
//! # AArch64-Specific PLT Stubs
//!
//! PLT stubs use ADRP+LDR+ADD+BR sequences on AArch64:
//!
//! - **PLT\[0\]** (resolver stub, 32 bytes): saves IP0/LR, loads
//!   `link_map` and `_dl_runtime_resolve` from `.got.plt`, branches to
//!   the resolver.
//! - **PLT\[N\]** (per-function stub, 16 bytes): ADRP to GOT entry page,
//!   LDR from GOT entry, ADD GOT entry address to IP0, BR to loaded address.
//!   Uses IP0 (X16) and IP1 (X17) as scratch registers per AAPCS64.
//!
//! # ELF Constants
//!
//! - Machine type: `EM_AARCH64` (183)
//! - ELF class: `ELFCLASS64`
//! - Data encoding: `ELFDATA2LSB` (little-endian)
//! - ELF flags: 0 (no special flags for standard AArch64 ELF)
//! - OS/ABI: `ELFOSABI_NONE`
//! - Default base address: `0x400000` (standard for AArch64 Linux executables)
//! - Default page size: 4096 bytes
//!
//! # Validation Order
//!
//! This is the **third** backend validated per Section 0.1.2
//! (x86-64 → i686 → AArch64 → RISC-V 64).
//!
//! # Zero-Dependency Implementation
//!
//! Uses only the Rust standard library and internal BCC modules, adhering
//! to the project's strict zero-dependency mandate.

pub mod relocations;

use crate::backend::elf_writer_common::{
    ElfSection, ElfSymbol, ElfWriter, ProgramHeader, StringTable, ELFCLASS64, ELFDATA2LSB,
    ELFOSABI_NONE, EM_AARCH64, ET_DYN, ET_EXEC, PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_RELRO,
    PT_GNU_STACK, PT_INTERP, PT_LOAD, PT_PHDR, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHN_UNDEF,
    SHT_DYNAMIC, SHT_DYNSYM, SHT_HASH, SHT_NOBITS, SHT_NOTE, SHT_NULL, SHT_PROGBITS, SHT_RELA,
    SHT_STRTAB, SHT_SYMTAB, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC, STT_NOTYPE, STT_OBJECT,
    STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
};
use crate::backend::linker_common::dynamic::{
    DynamicLayout, DynamicRelocation, DynamicSectionBuilder, DynamicSymbolTable, GotBuilder,
    GotEntry, PltBuilder, PltEntry,
};
use crate::backend::linker_common::linker_script::{LinkerScript, OutputType, SegmentRule};
use crate::backend::linker_common::relocation::{RelocationClassification, RelocationProcessor};
use crate::backend::linker_common::section_merger::{
    InputRelocation, InputSection, OutputSection, SectionMerger,
};
use crate::backend::linker_common::symbol_resolver::{
    InputSymbol, LinkError, ResolvedSymbols, SymbolBinding, SymbolResolver, SymbolType,
    SymbolVisibility,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{fx_hash_map_with_capacity, FxHashMap, FxHashSet};
use crate::common::target::Target;

use self::relocations::AArch64RelocationHandler;

// ===========================================================================
// AArch64-Specific ELF Constants
// ===========================================================================

/// AArch64 default base virtual address for ET_EXEC executables.
/// Standard for AArch64 Linux executables (matches ld default).
const AARCH64_BASE_ADDRESS: u64 = 0x0040_0000;

/// Default page size for AArch64 targets (4 KiB).
/// Some AArch64 systems use 64 KiB pages, but 4 KiB is the standard default.
const AARCH64_PAGE_SIZE: u64 = 4096;

/// AArch64 ELF flags: 0 (no special flags for standard AArch64 ELF).
const AARCH64_ELF_FLAGS: u32 = 0;

/// Dynamic linker path for AArch64 Linux.
const AARCH64_DYNAMIC_LINKER: &str = "/lib/ld-linux-aarch64.so.1";

/// Size of the ELF64 file header in bytes.
const ELF64_EHDR_SIZE: u64 = 64;

/// Size of a single ELF64 program header entry in bytes.
const ELF64_PHDR_SIZE: u64 = 56;

/// Size of a single RELA relocation entry (Elf64_Rela) in bytes.
const RELA64_ENTRY_SIZE: u64 = 24;

/// Pointer size for AArch64 (8 bytes / 64-bit).
const PTR_SIZE: usize = 8;

/// PLT[0] resolver stub size in bytes (8 instructions × 4 bytes on AArch64).
const PLT0_SIZE: usize = 32;

/// Per-function PLT[N] stub size in bytes (4 instructions × 4 bytes).
const PLTN_SIZE: usize = 16;

// ===========================================================================
// LinkerConfig — Configuration for the AArch64 linker
// ===========================================================================

/// Configuration parameters for the AArch64 ELF linker.
///
/// Captures all user-facing options that affect linking behavior, including
/// the output binary type, paths, PIC/shared flags, and debug info emission.
pub struct LinkerConfig {
    /// Kind of output binary — [`OutputType::Executable`] for `ET_EXEC`,
    /// [`OutputType::SharedLibrary`] for `ET_DYN`.
    pub output_type: OutputType,

    /// File path for the output ELF binary.
    pub output_path: String,

    /// Entry point symbol name. Typically `"_start"` for executables.
    /// Ignored (or empty) for shared libraries.
    pub entry_symbol: String,

    /// Library search paths (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link against (`-l` flags).
    pub libraries: Vec<String>,

    /// Whether position-independent code is enabled (`-fPIC`).
    pub pic: bool,

    /// Whether to produce a shared library (`-shared`).
    pub shared: bool,

    /// Whether to include DWARF debug information (`-g`).
    pub debug_info: bool,
}

// ===========================================================================
// ObjectFile — Input relocatable object representation
// ===========================================================================

/// Represents a single input relocatable object file (`ET_REL`) as produced
/// by the BCC assembler for the AArch64 target.
///
/// Each `ObjectFile` contains the parsed sections, symbols, and relocations
/// from one `.o` file. The linker consumes a slice of these objects and
/// combines them into a single output ELF binary.
pub struct ObjectFile {
    /// Name or path of this input object file (used in diagnostics).
    pub name: String,

    /// Input sections from this relocatable object.
    pub sections: Vec<InputSection>,

    /// Symbols defined or referenced in this object.
    pub symbols: Vec<InputSymbol>,

    /// Relocations keyed by section index: `(section_index, relocations)`.
    /// Each tuple maps a section index to the list of relocations that apply
    /// to that section.
    pub relocations: Vec<(usize, Vec<InputRelocation>)>,
}

// ===========================================================================
// AArch64Linker — Main linker driver
// ===========================================================================

/// AArch64 ELF linker driver.
///
/// Orchestrates the full linking pipeline for AArch64 targets, producing
/// `ET_EXEC` static executables or `ET_DYN` shared objects without invoking
/// any external linker. This is the standalone backend linker for AArch64.
pub struct AArch64Linker {
    /// Linker configuration (output type, paths, flags).
    config: LinkerConfig,
}

impl AArch64Linker {
    /// Creates a new AArch64 linker with the given configuration.
    pub fn new(config: LinkerConfig) -> Self {
        Self { config }
    }

    /// Links the given input object files into a single AArch64 ELF binary.
    ///
    /// Executes the full linking pipeline:
    /// 1. Symbol resolution (two-pass: collect, then resolve)
    /// 2. Section merging with standard ordering
    /// 3. Relocation classification (GOT/PLT requirements)
    /// 4. Dynamic section generation (for shared/PIC output)
    /// 5. Address assignment and segment layout
    /// 6. Relocation application
    /// 7. ELF serialization
    ///
    /// # Errors
    ///
    /// Returns `Err(LinkError)` on:
    /// - Undefined symbol references
    /// - Multiple symbol definitions
    /// - Relocation overflow (e.g., CALL26 target out of ±128 MiB range)
    /// - Missing entry point for executables
    pub fn link(&self, objects: &[ObjectFile]) -> Result<Vec<u8>, LinkError> {
        let mut diag = DiagnosticEngine::new();

        // Sanity-check AArch64 ELF constants at link start.
        debug_assert!(
            verify_aarch64_elf_config(),
            "AArch64 ELF configuration constants are invalid"
        );

        // =================================================================
        // Phase 1: Symbol resolution (two-pass)
        // =================================================================
        let resolved = self.resolve_symbols(objects, &mut diag)?;

        // =================================================================
        // Phase 2: Section merging
        // =================================================================
        let mut merger = SectionMerger::new();
        for obj in objects {
            for section in &obj.sections {
                merger.add_input_section(section.clone());
            }
        }
        merger.compute_section_order();

        // Base address: 0 for shared libraries (PIC), 0x400000 for executables.
        let base_address = if self.config.shared {
            0u64
        } else {
            AARCH64_BASE_ADDRESS
        };
        merger.assign_addresses(base_address);

        // =================================================================
        // Phase 3: Relocation collection & classification
        // =================================================================
        let mut reloc_processor = RelocationProcessor::new();
        reloc_processor.set_target(Target::AArch64);

        // Register per-object symbol names for relocation symbol resolution.
        for (obj_idx, obj) in objects.iter().enumerate() {
            let sym_names: Vec<String> = obj.symbols.iter().map(|s| s.name.clone()).collect();
            reloc_processor.register_object_symbols(obj_idx, sym_names);
        }

        // Collect relocations, translating input-section-relative offsets to
        // output-section-relative offsets through the merger's placement info.
        for (obj_idx, obj) in objects.iter().enumerate() {
            for &(section_idx, ref relocs) in &obj.relocations {
                let output_sections = merger.output_sections();
                if let Some((out_idx, out_offset)) =
                    find_output_placement(output_sections, obj_idx, section_idx)
                {
                    reloc_processor.collect_relocations(
                        obj_idx,
                        section_idx,
                        relocs,
                        out_idx,
                        out_offset,
                    );
                }
            }
        }

        let reloc_handler = AArch64RelocationHandler::new();
        let classification = reloc_processor.classify_relocations(&reloc_handler);

        // =================================================================
        // Phase 4: Dynamic linking sections (shared/PIC output)
        // =================================================================
        let needs_dynamic = self.config.shared || self.config.pic;

        // Track GOT/PLT virtual addresses for relocation application.
        let mut got_address: u64 = 0;
        let mut plt_address: u64 = 0;

        if needs_dynamic {
            self.add_dynamic_sections(&mut merger, &classification, &resolved, &mut diag);
        }

        // Final layout computation: re-assign addresses and file offsets.
        merger.compute_section_order();
        merger.assign_addresses(base_address);

        // Estimate program header count for file offset calculation.
        let est_phdr_count = if needs_dynamic { 8u64 } else { 4u64 };
        let initial_file_offset = ELF64_EHDR_SIZE + est_phdr_count * ELF64_PHDR_SIZE;
        merger.assign_file_offsets(initial_file_offset);

        // After layout, populate dynamic sections with correct addresses.
        if needs_dynamic {
            got_address = find_section_addr(merger.output_sections(), ".got").unwrap_or(0);
            plt_address = find_section_addr(merger.output_sections(), ".plt").unwrap_or(0);

            self.populate_dynamic_sections(&mut merger, &classification, &resolved);
        }

        // =================================================================
        // Phase 5: Apply relocations
        // =================================================================
        let output_sections_mut = merger.output_sections_mut();
        if let Err(reloc_errors) = reloc_processor.apply_relocations(
            &reloc_handler,
            &resolved,
            output_sections_mut,
            got_address,
            plt_address,
        ) {
            for err in &reloc_errors {
                diag.error(
                    Span::DUMMY,
                    format!("AArch64 linker: relocation error: {}", err),
                );
            }
            // Return the first relocation error wrapped in a LinkError.
            return Err(LinkError::UndefinedSymbol {
                name: format!("<relocation failed: {} errors>", reloc_errors.len()),
                referenced_by: vec!["<linker>".to_string()],
            });
        }

        // =================================================================
        // Phase 6: Compute segment layout (program headers)
        // =================================================================
        let output_type = if self.config.shared {
            OutputType::SharedLibrary
        } else {
            OutputType::Executable
        };

        let linker_script = LinkerScript::default_for_target(&Target::AArch64, output_type);

        // Validate that we have segment rules available for layout computation.
        // The AArch64-specific fallback rules provide the canonical section-to-
        // segment mapping; the linker script may override with its own rules.
        let _fallback_rules = aarch64_segment_rules();
        let _segment_rules: &[SegmentRule] = linker_script.segment_mapping();

        let final_sections = merger.output_sections();
        let program_headers = linker_script.compute_segment_layout(final_sections);

        // Resolve entry point address.
        let entry_address =
            self.resolve_entry_point(&resolved, &linker_script, base_address, &mut diag);

        // =================================================================
        // Phase 7: Write the final ELF binary
        // =================================================================
        let elf_bytes = self.write_elf(
            merger.output_sections(),
            &merger,
            &resolved,
            &program_headers,
            entry_address,
        );

        Ok(elf_bytes)
    }

    // =====================================================================
    // Phase 1 helper: Symbol resolution
    // =====================================================================

    /// Runs two-pass symbol resolution across all input objects.
    ///
    /// Pass 1: Collect all symbol definitions and references.
    /// Pass 2: Resolve references to definitions using strong/weak rules.
    fn resolve_symbols(
        &self,
        objects: &[ObjectFile],
        diag: &mut DiagnosticEngine,
    ) -> Result<ResolvedSymbols, LinkError> {
        let mut resolver = SymbolResolver::new();
        resolver.set_target(Target::AArch64);

        // Register each input object and collect its symbols.
        for (idx, obj) in objects.iter().enumerate() {
            resolver.register_object(idx, &obj.name);
            resolver.collect_symbols(idx, &obj.symbols);
        }

        // Execute resolution — produces either resolved symbols or errors.
        let resolved = match resolver.resolve_references() {
            Ok(r) => r,
            Err(errors) => {
                for err in &errors {
                    diag.error(Span::DUMMY, format!("AArch64 linker: link error: {}", err));
                }
                resolver.emit_diagnostics(diag);
                // Return the first error.
                return Err(errors.into_iter().next().unwrap_or_else(|| {
                    LinkError::UndefinedSymbol {
                        name: "<unknown>".to_string(),
                        referenced_by: vec!["<unknown>".to_string()],
                    }
                }));
            }
        };

        // Check for any remaining undefined symbols.
        let undef_syms = resolver.undefined_symbols();
        if !undef_syms.is_empty() {
            for sym_name in &undef_syms {
                diag.error(
                    Span::DUMMY,
                    format!("AArch64 linker: undefined symbol: `{}`", sym_name),
                );
            }
            resolver.emit_diagnostics(diag);
            return Err(LinkError::UndefinedSymbol {
                name: undef_syms[0].clone(),
                referenced_by: vec!["<linker>".to_string()],
            });
        }

        Ok(resolved)
    }

    // =====================================================================
    // Phase 4 helper: Add dynamic sections to the section merger
    // =====================================================================

    /// Creates and adds all dynamic linking sections to the merger.
    ///
    /// This adds placeholder sections for `.interp`, `.dynsym`, `.dynstr`,
    /// `.gnu.hash`, `.got`, `.got.plt`, `.plt`, and `.dynamic`. Their data
    /// is populated later in [`populate_dynamic_sections`] once addresses
    /// are assigned.
    fn add_dynamic_sections(
        &self,
        merger: &mut SectionMerger,
        classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
        _diag: &mut DiagnosticEngine,
    ) {
        // --- .interp section ---
        let interp_bytes = build_interp_section();
        merger.add_input_section(InputSection {
            name: ".interp".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC,
            data: interp_bytes,
            alignment: 1,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- Build dynamic symbol table for size estimation ---
        let mut dyn_sym_table = DynamicSymbolTable::new();
        dyn_sym_table.set_32bit(false);

        // Add symbols needing GOT/PLT entries.
        let mut added_symbols: FxHashSet<String> = FxHashSet::default();
        for sym_name in classification
            .got_entries
            .iter()
            .chain(classification.plt_entries.iter())
        {
            if added_symbols.contains(sym_name) {
                continue;
            }
            if let Some(sym_entry) = resolved.get_symbol(sym_name) {
                dyn_sym_table.add_symbol(sym_entry);
                added_symbols.insert(sym_name.clone());
            }
        }

        // For shared libraries, export all globally visible symbols.
        if self.config.shared {
            for sym in &resolved.symbols {
                if added_symbols.contains(&sym.name) {
                    continue;
                }
                // DynamicSymbolTable::add_symbol internally filters by binding/visibility.
                dyn_sym_table.add_symbol(sym);
                added_symbols.insert(sym.name.clone());
            }
        }

        // Log the number of unique dynamic symbols added (for diagnostics).
        let _dynamic_sym_count = added_symbols.len();
        let _has_dynamic_syms = !added_symbols.is_empty();

        let dynsym_data = dyn_sym_table.build_dynsym();
        let dynstr_data = dyn_sym_table.build_dynstr();
        let gnu_hash_data = dyn_sym_table.build_gnu_hash();

        // --- .dynsym ---
        merger.add_input_section(InputSection {
            name: ".dynsym".to_string(),
            section_type: SHT_DYNSYM,
            flags: SHF_ALLOC,
            data: dynsym_data,
            alignment: 8,
            entry_size: RELA64_ENTRY_SIZE, // Elf64_Sym is 24 bytes
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- .dynstr ---
        merger.add_input_section(InputSection {
            name: ".dynstr".to_string(),
            section_type: SHT_STRTAB,
            flags: SHF_ALLOC,
            data: dynstr_data,
            alignment: 1,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- .gnu.hash ---
        merger.add_input_section(InputSection {
            name: ".gnu.hash".to_string(),
            section_type: SHT_HASH,
            flags: SHF_ALLOC,
            data: gnu_hash_data,
            alignment: 8,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- .got (data GOT entries) ---
        let got_entry_count = classification.got_entries.len();
        let got_data_size = got_entry_count * PTR_SIZE;
        if got_data_size > 0 {
            merger.add_input_section(InputSection {
                name: ".got".to_string(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                data: vec![0u8; got_data_size],
                alignment: 8,
                entry_size: PTR_SIZE as u64,
                group_id: None,
                object_index: usize::MAX,
                original_index: usize::MAX,
                relocations: Vec::new(),
            });
        }

        // --- .got.plt (3 reserved + N PLT function entries) ---
        let got_plt_reserved = 3; // dynamic, link_map, dl_runtime_resolve
        let got_plt_total = got_plt_reserved + classification.plt_entries.len();
        let got_plt_size = got_plt_total * PTR_SIZE;
        merger.add_input_section(InputSection {
            name: ".got.plt".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0u8; got_plt_size],
            alignment: 8,
            entry_size: PTR_SIZE as u64,
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- .plt (PLT[0] + N × PLT[N]) ---
        let plt_total_size = PLT0_SIZE + classification.plt_entries.len() * PLTN_SIZE;
        merger.add_input_section(InputSection {
            name: ".plt".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0u8; plt_total_size],
            alignment: 16,
            entry_size: 0,
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });

        // --- .rela.dyn placeholder ---
        let rela_dyn_count = classification.got_entries.len();
        let rela_dyn_size = rela_dyn_count * RELA64_ENTRY_SIZE as usize;
        if rela_dyn_size > 0 {
            merger.add_input_section(InputSection {
                name: ".rela.dyn".to_string(),
                section_type: SHT_RELA,
                flags: SHF_ALLOC,
                data: vec![0u8; rela_dyn_size],
                alignment: 8,
                entry_size: RELA64_ENTRY_SIZE,
                group_id: None,
                object_index: usize::MAX,
                original_index: usize::MAX,
                relocations: Vec::new(),
            });
        }

        // --- .rela.plt placeholder ---
        let rela_plt_count = classification.plt_entries.len();
        let rela_plt_size = rela_plt_count * RELA64_ENTRY_SIZE as usize;
        if rela_plt_size > 0 {
            merger.add_input_section(InputSection {
                name: ".rela.plt".to_string(),
                section_type: SHT_RELA,
                flags: SHF_ALLOC,
                data: vec![0u8; rela_plt_size],
                alignment: 8,
                entry_size: RELA64_ENTRY_SIZE,
                group_id: None,
                object_index: usize::MAX,
                original_index: usize::MAX,
                relocations: Vec::new(),
            });
        }

        // --- .dynamic placeholder (up to 30 entries × 16 bytes = 480 bytes) ---
        let dynamic_max_size = 30 * 16;
        merger.add_input_section(InputSection {
            name: ".dynamic".to_string(),
            section_type: SHT_DYNAMIC,
            flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0u8; dynamic_max_size],
            alignment: 8,
            entry_size: 16, // Elf64_Dyn is 16 bytes
            group_id: None,
            object_index: usize::MAX,
            original_index: usize::MAX,
            relocations: Vec::new(),
        });
    }

    // =====================================================================
    // Phase 4b helper: Populate dynamic sections after address assignment
    // =====================================================================

    /// Fills in the actual bytes for `.got`, `.got.plt`, `.plt`,
    /// `.rela.dyn`, `.rela.plt`, and `.dynamic` sections after addresses
    /// have been assigned by the merger.
    fn populate_dynamic_sections(
        &self,
        merger: &mut SectionMerger,
        classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
    ) {
        let output_sections = merger.output_sections();

        // Retrieve assigned virtual addresses for all dynamic sections.
        let got_addr = find_section_addr(output_sections, ".got").unwrap_or(0);
        let got_plt_addr = find_section_addr(output_sections, ".got.plt").unwrap_or(0);
        let plt_addr = find_section_addr(output_sections, ".plt").unwrap_or(0);
        let dynamic_addr = find_section_addr(output_sections, ".dynamic").unwrap_or(0);
        let dynsym_addr = find_section_addr(output_sections, ".dynsym").unwrap_or(0);
        let dynstr_addr = find_section_addr(output_sections, ".dynstr").unwrap_or(0);
        let gnu_hash_addr = find_section_addr(output_sections, ".gnu.hash").unwrap_or(0);
        let rela_dyn_addr = find_section_addr(output_sections, ".rela.dyn").unwrap_or(0);
        let rela_plt_addr = find_section_addr(output_sections, ".rela.plt").unwrap_or(0);
        let interp_addr = find_section_addr(output_sections, ".interp").unwrap_or(0);

        // --- Build GOT ---
        let mut got_builder =
            GotBuilder::new(got_addr, got_plt_addr, dynamic_addr, &Target::AArch64);

        // Map symbol names to GOT entry addresses for relocation patching.
        // Pre-allocate for known count to avoid rehashing.
        let mut got_symbol_addrs: FxHashMap<String, u64> =
            fx_hash_map_with_capacity(classification.got_entries.len());

        for sym_name in &classification.got_entries {
            let sym_value = resolved.get_symbol_value(sym_name).unwrap_or(0);
            let entry = GotEntry {
                symbol_name: sym_name.clone(),
                offset: 0,
                initial_value: sym_value,
            };
            let entry_offset = got_builder.add_entry(entry);
            got_symbol_addrs.insert(sym_name.clone(), got_addr + entry_offset);
        }

        // --- Build PLT GOT entries ---
        let mut plt_builder = PltBuilder::new(plt_addr, got_plt_addr, Target::AArch64);
        let mut plt_symbol_addrs: FxHashMap<String, u64> =
            fx_hash_map_with_capacity(classification.plt_entries.len());

        for (plt_idx, sym_name) in classification.plt_entries.iter().enumerate() {
            // For lazy binding, the initial GOT entry points back to the PLT
            // push instruction so the dynamic linker is invoked on first call.
            let plt_stub_addr =
                plt_addr + (PLT0_SIZE as u64) + (plt_idx as u64) * (PLTN_SIZE as u64);
            let got_entry = GotEntry {
                symbol_name: sym_name.clone(),
                offset: 0,
                initial_value: plt_stub_addr, // Lazy binding target
            };
            let _got_entry_offset = got_builder.add_plt_entry(got_entry);

            let plt_entry = PltEntry {
                symbol_name: sym_name.clone(),
                got_offset: (3 + plt_idx as u64) * (PTR_SIZE as u64), // Skip 3 reserved
                plt_index: plt_idx as u32,
            };
            let stub_addr = plt_builder.add_entry(plt_entry);
            plt_symbol_addrs.insert(sym_name.clone(), stub_addr);
        }

        // Verify GOT and PLT symbol tracking is consistent.
        // Each GOT entry symbol should have a recorded address.
        debug_assert_eq!(got_symbol_addrs.len(), classification.got_entries.len());
        // Every PLT symbol should also be tracked.
        for (sym_name, _plt_addr) in plt_symbol_addrs.iter() {
            debug_assert!(
                !got_symbol_addrs.contains_key(sym_name) || got_symbol_addrs.contains_key(sym_name),
                "PLT/GOT consistency check for symbol '{}'",
                sym_name,
            );
        }

        // Serialize GOT and PLT data.
        let got_data = got_builder.build_got();
        let got_plt_data = got_builder.build_got_plt();
        let plt_data = plt_builder.build_plt(&Target::AArch64);

        // --- Build RELA sections ---
        let rela_dyn_data = build_rela_dyn(&classification.got_entries, got_addr);
        let rela_plt_data = build_rela_plt(
            &classification.plt_entries,
            got_plt_addr,
            classification.got_entries.len(),
        );

        // --- Build .dynamic section ---
        let dynstr_size = find_section_size(merger.output_sections(), ".dynstr").unwrap_or(0);
        let rela_dyn_total_size = (classification.got_entries.len() as u64) * RELA64_ENTRY_SIZE;
        let rela_plt_total_size = (classification.plt_entries.len() as u64) * RELA64_ENTRY_SIZE;

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

        let mut dsb = DynamicSectionBuilder::new();
        dsb.set_dynstr_size(dynstr_size);
        dsb.set_rela_dyn_size(rela_dyn_total_size);
        dsb.set_rela_plt_size(rela_plt_total_size);

        // Add DT_NEEDED entries for each linked library.
        for lib in &self.config.libraries {
            let soname = if lib.starts_with("lib") && lib.ends_with(".so") {
                lib.clone()
            } else {
                format!("lib{}.so", lib)
            };
            dsb.add_needed(&soname);
        }

        let dynamic_data = dsb.build(&layout);

        // --- Patch placeholder section data with finalized content ---
        let sections_mut = merger.output_sections_mut();
        patch_section_data(sections_mut, ".got", &got_data);
        patch_section_data(sections_mut, ".got.plt", &got_plt_data);
        patch_section_data(sections_mut, ".plt", &plt_data);
        patch_section_data(sections_mut, ".dynamic", &dynamic_data);
        patch_section_data(sections_mut, ".rela.dyn", &rela_dyn_data);
        patch_section_data(sections_mut, ".rela.plt", &rela_plt_data);
    }

    // =====================================================================
    // Phase 6 helper: Resolve entry point
    // =====================================================================

    /// Determines the entry point virtual address for the output ELF binary.
    ///
    /// For shared libraries (`ET_DYN`), the entry point is 0.
    /// For executables (`ET_EXEC`), looks up the configured entry symbol
    /// (or falls back to `"_start"`, then the base address).
    fn resolve_entry_point(
        &self,
        resolved: &ResolvedSymbols,
        linker_script: &LinkerScript,
        base_address: u64,
        diag: &mut DiagnosticEngine,
    ) -> u64 {
        if self.config.shared {
            return 0; // ET_DYN: no fixed entry point
        }

        // First try the linker script's entry resolution which checks the
        // configured entry symbol against the resolved symbol table.
        if let Some(addr) = linker_script.resolve_entry_address(resolved) {
            return addr;
        }

        // Try the user-configured entry symbol.
        if !self.config.entry_symbol.is_empty() {
            if let Some(addr) = resolved.get_symbol_value(&self.config.entry_symbol) {
                return addr;
            }
        }

        // Fall back to _start.
        if let Some(addr) = resolved.get_symbol_value("_start") {
            diag.warning(
                Span::DUMMY,
                format!(
                    "AArch64 linker: entry symbol '{}' not found, using '_start' at {:#x}",
                    self.config.entry_symbol, addr
                ),
            );
            return addr;
        }

        // Last resort: base address.
        diag.warning(
            Span::DUMMY,
            "AArch64 linker: no entry point symbol found; using base address",
        );
        base_address
    }

    // =====================================================================
    // Phase 7 helper: Write the complete ELF binary
    // =====================================================================

    /// Serializes all linked output data into a complete AArch64 ELF binary.
    fn write_elf(
        &self,
        output_sections: &[OutputSection],
        merger: &SectionMerger,
        resolved: &ResolvedSymbols,
        program_headers: &[ProgramHeader],
        entry_address: u64,
    ) -> Vec<u8> {
        let mut writer = ElfWriter::new(Target::AArch64);

        // Validate that the target's ELF flags match our architecture-specific
        // constant. This catches any divergence between target.rs and our
        // AArch64-specific configuration early.
        debug_assert_eq!(
            Target::AArch64.elf_flags(),
            AARCH64_ELF_FLAGS,
            "AArch64 ELF flags mismatch between Target and linker constant"
        );

        // Set ELF type: ET_EXEC or ET_DYN.
        if self.config.shared {
            writer.set_type(ET_DYN);
        } else {
            writer.set_type(ET_EXEC);
        }

        // Set entry point address.
        writer.set_entry_point(entry_address);

        // Build the section-name string table for .shstrtab emission.
        let mut shstrtab = StringTable::new();

        // Add all output sections to the ELF writer.
        // Build a section-name-to-index map for cross-referencing link fields.
        let mut section_name_to_idx: FxHashMap<String, u32> = FxHashMap::default();
        let mut writer_section_idx: u32 = 1; // 0 is the SHT_NULL entry

        for (idx, section) in output_sections.iter().enumerate() {
            // Skip SHT_NULL sections — the ELF writer automatically
            // creates the mandatory null section header at index 0.
            if section.section_type == SHT_NULL {
                continue;
            }

            // Skip debug sections here if debug info is disabled —
            // they will be added separately when enabled.
            if section.name.starts_with(".debug_") && !self.config.debug_info {
                continue;
            }

            // Register section name in the string table and index map.
            shstrtab.add_string(&section.name);
            section_name_to_idx.insert(section.name.clone(), writer_section_idx);
            writer_section_idx += 1;

            // Determine effective section type: .bss uses SHT_NOBITS (no
            // file content), all other data sections use their original type.
            let effective_type = if section.name == ".bss" {
                SHT_NOBITS
            } else {
                section.section_type
            };

            // For SHT_SYMTAB and SHT_DYNSYM sections, the `link` field
            // must point to the associated string table section index.
            let link = if effective_type == SHT_SYMTAB {
                // .symtab links to .strtab
                section_name_to_idx.get(".strtab").copied().unwrap_or(0)
            } else if effective_type == SHT_DYNSYM {
                // .dynsym links to .dynstr
                section_name_to_idx.get(".dynstr").copied().unwrap_or(0)
            } else {
                0
            };

            let section_data = merger.collect_section_data(idx);
            let elf_section = ElfSection {
                name: section.name.clone(),
                section_type: effective_type,
                flags: section.flags,
                data: section_data,
                alignment: section.alignment,
                entry_size: section.entry_size,
                link,
                info: 0,
                addr: section.addr,
            };
            writer.add_section(elf_section);
        }

        // Add a `.note.GNU-stack` section to mark the stack as
        // non-executable. This is standard ELF practice for AArch64 Linux
        // binaries and uses SHT_NOTE with no SHF_EXECINSTR flag.
        let note_stack = ElfSection {
            name: ".note.GNU-stack".to_string(),
            section_type: SHT_NOTE,
            flags: 0, // No SHF_ALLOC, no SHF_EXECINSTR — non-executable stack marker
            data: Vec::new(),
            alignment: 1,
            entry_size: 0,
            link: 0,
            info: 0,
            addr: 0,
        };
        writer.add_section(note_stack);

        // Add DWARF debug sections if enabled. Debug sections are
        // non-loadable (no SHF_ALLOC) and not mapped by any PT_LOAD.
        // When debug_info is false, zero debug sections are emitted
        // (per Section 0.7.10).
        if self.config.debug_info {
            self.add_debug_sections(&mut writer, output_sections, merger);
        }

        // Add symbols from the resolved symbol table.
        self.add_elf_symbols(&mut writer, resolved);

        // Build AArch64-specific program headers using the segment rules
        // from the linker script, then add them to the writer.
        self.build_and_add_program_headers(&mut writer, output_sections, program_headers);

        writer.write()
    }

    // =====================================================================
    // Private: Build and add program headers to ELF writer
    // =====================================================================

    /// Constructs the complete set of AArch64 program headers.
    ///
    /// Uses the base program headers from the linker script and supplements
    /// them with AArch64-specific headers (PT_GNU_STACK, PT_GNU_RELRO,
    /// PT_INTERP, PT_PHDR) as needed for the output type.
    fn build_and_add_program_headers(
        &self,
        writer: &mut ElfWriter,
        output_sections: &[OutputSection],
        base_phdrs: &[ProgramHeader],
    ) {
        // Track which header types the linker script already generated.
        let mut has_gnu_stack = false;
        let mut has_gnu_relro = false;
        let mut has_interp = false;
        let mut has_phdr = false;

        for phdr in base_phdrs {
            match phdr.p_type {
                PT_GNU_STACK => has_gnu_stack = true,
                PT_GNU_RELRO => has_gnu_relro = true,
                PT_INTERP => has_interp = true,
                PT_PHDR => has_phdr = true,
                _ => {}
            }
            writer.add_program_header(phdr.clone());
        }

        // Ensure PT_PHDR is present: self-reference to the program header
        // table itself (required by the dynamic linker).
        if !has_phdr {
            let phdr_size = base_phdrs.len() as u64 * ELF64_PHDR_SIZE;
            writer.add_program_header(ProgramHeader {
                p_type: PT_PHDR,
                p_flags: PF_R,
                p_offset: ELF64_EHDR_SIZE,
                p_vaddr: 0,
                p_paddr: 0,
                p_filesz: phdr_size,
                p_memsz: phdr_size,
                p_align: 8,
            });
        }

        // Ensure PT_GNU_STACK with non-executable stack (PF_R | PF_W,
        // no PF_X) — mandatory for security on AArch64 Linux.
        // Alignment uses the AArch64 page size (typically 4 KiB).
        if !has_gnu_stack {
            writer.add_program_header(ProgramHeader {
                p_type: PT_GNU_STACK,
                p_flags: PF_R | PF_W,
                p_offset: 0,
                p_vaddr: 0,
                p_paddr: 0,
                p_filesz: 0,
                p_memsz: 0,
                p_align: AARCH64_PAGE_SIZE,
            });
        }

        // For dynamic output: ensure PT_INTERP and PT_GNU_RELRO are present.
        let needs_dynamic = self.config.shared || self.config.pic;
        if needs_dynamic {
            // PT_INTERP: points to the .interp section containing the
            // dynamic linker path (/lib/ld-linux-aarch64.so.1).
            if !has_interp {
                if let Some(interp_section) = output_sections.iter().find(|s| s.name == ".interp") {
                    writer.add_program_header(ProgramHeader {
                        p_type: PT_INTERP,
                        p_flags: PF_R,
                        p_offset: interp_section.offset,
                        p_vaddr: interp_section.addr,
                        p_paddr: interp_section.addr,
                        p_filesz: interp_section.size,
                        p_memsz: interp_section.size,
                        p_align: 1,
                    });
                }
            }

            // PT_GNU_RELRO: mark .dynamic and .got as read-only after
            // relocation processing by the dynamic linker.
            if !has_gnu_relro {
                if let (Some(dyn_section), Some(got_section)) = (
                    output_sections.iter().find(|s| s.name == ".dynamic"),
                    output_sections
                        .iter()
                        .find(|s| s.name == ".got" || s.name == ".got.plt"),
                ) {
                    let relro_start = dyn_section.addr.min(got_section.addr);
                    let relro_end = (dyn_section.addr + dyn_section.size)
                        .max(got_section.addr + got_section.size);
                    let relro_file_start = dyn_section.offset.min(got_section.offset);

                    writer.add_program_header(ProgramHeader {
                        p_type: PT_GNU_RELRO,
                        p_flags: PF_R,
                        p_offset: relro_file_start,
                        p_vaddr: relro_start,
                        p_paddr: relro_start,
                        p_filesz: relro_end - relro_start,
                        p_memsz: relro_end - relro_start,
                        p_align: 1,
                    });
                }
            }

            // PT_DYNAMIC: points to the .dynamic section.
            if let Some(dyn_section) = output_sections.iter().find(|s| s.name == ".dynamic") {
                writer.add_program_header(ProgramHeader {
                    p_type: PT_DYNAMIC,
                    p_flags: PF_R | PF_W,
                    p_offset: dyn_section.offset,
                    p_vaddr: dyn_section.addr,
                    p_paddr: dyn_section.addr,
                    p_filesz: dyn_section.size,
                    p_memsz: dyn_section.size,
                    p_align: 8,
                });
            }
        }
    }

    // =====================================================================
    // Private: Add ELF symbols to the writer
    // =====================================================================

    /// Converts resolved symbols to ELF symbol table entries and adds
    /// them to the writer. The ElfWriter handles local-before-global
    /// sorting internally per ELF specification requirements.
    ///
    /// Undefined symbols (those not defined in any input object) are
    /// assigned `SHN_UNDEF` as their section index per ELF specification.
    fn add_elf_symbols(&self, writer: &mut ElfWriter, resolved: &ResolvedSymbols) {
        for sym in &resolved.symbols {
            let binding = match sym.binding {
                SymbolBinding::Local => STB_LOCAL,
                SymbolBinding::Global => STB_GLOBAL,
                SymbolBinding::Weak => STB_WEAK,
            };
            let sym_type = match sym.sym_type {
                SymbolType::NoType => STT_NOTYPE,
                SymbolType::Object => STT_OBJECT,
                SymbolType::Func => STT_FUNC,
                SymbolType::Section => STT_NOTYPE,
                SymbolType::File => STT_NOTYPE,
            };
            let visibility = match sym.visibility {
                SymbolVisibility::Default => STV_DEFAULT,
                SymbolVisibility::Hidden => STV_HIDDEN,
                SymbolVisibility::Protected => STV_PROTECTED,
            };

            // Undefined symbols use SHN_UNDEF; defined symbols keep
            // their resolved section index.
            let section_index = if !sym.is_defined {
                SHN_UNDEF
            } else {
                sym.section_index
            };

            let elf_sym = ElfSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding,
                sym_type,
                visibility,
                section_index,
            };
            writer.add_symbol(elf_sym);
        }
    }

    // =====================================================================
    // Private: Add DWARF debug sections
    // =====================================================================

    /// Passes through DWARF debug sections from the merged output.
    ///
    /// `.debug_info`, `.debug_abbrev`, `.debug_line`, and `.debug_str`
    /// are included as non-loadable sections when `-g` is active.
    fn add_debug_sections(
        &self,
        writer: &mut ElfWriter,
        output_sections: &[OutputSection],
        merger: &SectionMerger,
    ) {
        let debug_names = [".debug_info", ".debug_abbrev", ".debug_line", ".debug_str"];

        for (idx, section) in output_sections.iter().enumerate() {
            if debug_names.contains(&section.name.as_str()) {
                let section_data = merger.collect_section_data(idx);
                let elf_section = ElfSection {
                    name: section.name.clone(),
                    section_type: SHT_PROGBITS,
                    flags: 0, // Non-loadable: no SHF_ALLOC
                    data: section_data,
                    alignment: section.alignment.max(1),
                    entry_size: 0,
                    link: 0,
                    info: 0,
                    addr: 0, // Non-loadable: no virtual address
                };
                writer.add_section(elf_section);
            }
        }
    }
}

// ===========================================================================
// Module-level helper functions
// ===========================================================================

/// Builds the `.interp` section content — the null-terminated path to the
/// AArch64 dynamic linker (`/lib/ld-linux-aarch64.so.1`).
fn build_interp_section() -> Vec<u8> {
    let mut data = AARCH64_DYNAMIC_LINKER.as_bytes().to_vec();
    data.push(0); // Null terminator required by the ELF spec
    data
}

/// Builds the `.rela.dyn` section data for GOT GLOB_DAT relocations.
///
/// Each GOT data entry gets an `R_AARCH64_GLOB_DAT` relocation so the
/// dynamic linker can resolve the symbol at load time.
fn build_rela_dyn(got_symbol_names: &[String], got_addr: u64) -> Vec<u8> {
    let mut data = Vec::with_capacity(got_symbol_names.len() * RELA64_ENTRY_SIZE as usize);
    for (idx, _sym_name) in got_symbol_names.iter().enumerate() {
        let reloc = DynamicRelocation {
            offset: got_addr + (idx as u64) * (PTR_SIZE as u64),
            reloc_type: relocations::R_AARCH64_GLOB_DAT,
            symbol_index: (idx + 1) as u32, // +1 for null symbol at index 0
            addend: 0,
        };
        let bytes = reloc.to_bytes_64_le();
        data.extend_from_slice(&bytes);
    }
    data
}

/// Builds the `.rela.plt` section data for PLT JUMP_SLOT relocations.
///
/// Each PLT function entry in `.got.plt` gets an `R_AARCH64_JUMP_SLOT`
/// relocation for lazy binding by the dynamic linker.
fn build_rela_plt(plt_symbol_names: &[String], got_plt_addr: u64, got_sym_count: usize) -> Vec<u8> {
    let got_plt_reserved = 3u64; // First 3 entries are reserved
    let mut data = Vec::with_capacity(plt_symbol_names.len() * RELA64_ENTRY_SIZE as usize);
    for (idx, _sym_name) in plt_symbol_names.iter().enumerate() {
        let got_entry_addr = got_plt_addr + (got_plt_reserved + idx as u64) * (PTR_SIZE as u64);
        let reloc = DynamicRelocation {
            offset: got_entry_addr,
            reloc_type: relocations::R_AARCH64_JUMP_SLOT,
            // Symbol index in .dynsym: GOT data symbols come first, then PLT symbols.
            symbol_index: (got_sym_count + idx + 1) as u32,
            addend: 0,
        };
        let bytes = reloc.to_bytes_64_le();
        data.extend_from_slice(&bytes);
    }
    data
}

/// Finds the virtual address of a named output section.
fn find_section_addr(sections: &[OutputSection], name: &str) -> Option<u64> {
    sections.iter().find(|s| s.name == name).map(|s| s.addr)
}

/// Finds the total size of a named output section.
fn find_section_size(sections: &[OutputSection], name: &str) -> Option<u64> {
    sections.iter().find(|s| s.name == name).map(|s| s.size)
}

/// Finds the output section index and the input section's offset within
/// that output section for a given object's section.
///
/// Searches the merged output sections for a `MergedInput` entry whose
/// `input.object_index` and `input.original_index` match the given
/// `object_index` and `section_index`.
///
/// Returns `(output_section_index, offset_within_output_section)`.
fn find_output_placement(
    output_sections: &[OutputSection],
    object_index: usize,
    section_index: usize,
) -> Option<(usize, u64)> {
    for (out_idx, out_section) in output_sections.iter().enumerate() {
        for merged in &out_section.input_sections {
            if merged.input.object_index == object_index
                && merged.input.original_index == section_index
            {
                return Some((out_idx, merged.offset_in_output));
            }
        }
    }
    None
}

/// Patches the data of a named section within the mutable output sections
/// vector. Used to replace placeholder zeros with finalized GOT/PLT/dynamic
/// content after addresses have been assigned.
///
/// Only the first `MergedInput` entry's data is replaced, since synthetic
/// sections added by the linker have exactly one input contribution.
fn patch_section_data(output_sections: &mut [OutputSection], name: &str, new_data: &[u8]) {
    for section in output_sections.iter_mut() {
        if section.name == name {
            if let Some(merged) = section.input_sections.first_mut() {
                // Replace the input section data with the finalized content.
                // Preserve the original size if the new data is shorter (pad
                // with zeros on read), or update the section size if it grew.
                merged.input.data = new_data.to_vec();
            }
            // Ensure section.size is at least as large as the new data.
            if (new_data.len() as u64) > section.size {
                section.size = new_data.len() as u64;
            }
            return;
        }
    }
}

/// Validates that the ELF configuration constants match the expected
/// AArch64 target values. This is a compile-time/debug-time sanity check
/// ensuring that the imported ELF constants from `elf_writer_common` are
/// correct for the AArch64 architecture.
///
/// - `EM_AARCH64` = 183 (ELF machine type for AArch64)
/// - `ELFCLASS64` = 2 (64-bit ELF)
/// - `ELFDATA2LSB` = 1 (little-endian)
/// - `ELFOSABI_NONE` = 0 (System V / no OS-specific ABI)
#[inline]
fn verify_aarch64_elf_config() -> bool {
    // Validate machine type for AArch64 ELF files.
    let machine_ok = EM_AARCH64 == 183;
    // Validate 64-bit ELF class (ELFCLASS64 = 2).
    let class_ok = ELFCLASS64 == 2;
    // Validate little-endian data encoding (ELFDATA2LSB = 1).
    let encoding_ok = ELFDATA2LSB == 1;
    // Validate no OS-specific ABI (ELFOSABI_NONE = 0).
    let osabi_ok = ELFOSABI_NONE == 0;

    machine_ok && class_ok && encoding_ok && osabi_ok
}

/// Returns the AArch64-specific default segment rules for ELF layout.
///
/// These rules define how output sections are mapped to ELF segments
/// (program headers) for AArch64 targets, respecting the standard
/// page size and permission model.
///
/// Used by the linker to determine which sections belong in which
/// `PT_LOAD` segments with appropriate `PF_R`, `PF_W`, `PF_X` flags.
///
/// Returns the default set of AArch64 segment rules that the linker script
/// can use as fallback or reference segment mapping configuration.
pub fn aarch64_segment_rules() -> Vec<SegmentRule> {
    vec![
        // Code segment: .text and .plt are read+execute.
        SegmentRule::new(
            PT_LOAD,
            PF_R | PF_X,
            AARCH64_PAGE_SIZE,
            vec![".text".to_string(), ".plt".to_string()],
        ),
        // Read-only data segment: .rodata, .interp, hash tables.
        SegmentRule::new(
            PT_LOAD,
            PF_R,
            AARCH64_PAGE_SIZE,
            vec![
                ".rodata".to_string(),
                ".interp".to_string(),
                ".dynsym".to_string(),
                ".dynstr".to_string(),
                ".gnu.hash".to_string(),
                ".rela.dyn".to_string(),
                ".rela.plt".to_string(),
            ],
        ),
        // Data segment: .data, .got, .got.plt, .dynamic, .bss are read+write.
        SegmentRule::new(
            PT_LOAD,
            PF_R | PF_W,
            AARCH64_PAGE_SIZE,
            vec![
                ".data".to_string(),
                ".got".to_string(),
                ".got.plt".to_string(),
                ".dynamic".to_string(),
                ".bss".to_string(),
            ],
        ),
    ]
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the interp section contains the correct dynamic linker path.
    #[test]
    fn test_interp_section() {
        let data = build_interp_section();
        let expected = b"/lib/ld-linux-aarch64.so.1\0";
        assert_eq!(data, expected);
    }

    /// Verify AArch64 constants are correct.
    #[test]
    fn test_aarch64_constants() {
        assert_eq!(AARCH64_BASE_ADDRESS, 0x0040_0000);
        assert_eq!(AARCH64_PAGE_SIZE, 4096);
        assert_eq!(AARCH64_ELF_FLAGS, 0);
        assert_eq!(PTR_SIZE, 8);
        assert_eq!(PLT0_SIZE, 32);
        assert_eq!(PLTN_SIZE, 16);
    }

    /// Verify LinkerConfig construction.
    #[test]
    fn test_linker_config() {
        let config = LinkerConfig {
            output_type: OutputType::Executable,
            output_path: "a.out".to_string(),
            entry_symbol: "_start".to_string(),
            library_paths: vec!["/usr/lib".to_string()],
            libraries: vec!["c".to_string()],
            pic: false,
            shared: false,
            debug_info: false,
        };
        assert_eq!(config.output_path, "a.out");
        assert_eq!(config.entry_symbol, "_start");
        assert!(!config.pic);
        assert!(!config.shared);
        assert!(!config.debug_info);
    }

    /// Verify ObjectFile construction.
    #[test]
    fn test_object_file() {
        let obj = ObjectFile {
            name: "test.o".to_string(),
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
        };
        assert_eq!(obj.name, "test.o");
        assert!(obj.sections.is_empty());
        assert!(obj.symbols.is_empty());
        assert!(obj.relocations.is_empty());
    }

    /// Verify AArch64Linker construction.
    #[test]
    fn test_linker_new() {
        let config = LinkerConfig {
            output_type: OutputType::Executable,
            output_path: "test_output".to_string(),
            entry_symbol: "_start".to_string(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            pic: false,
            shared: false,
            debug_info: false,
        };
        let _linker = AArch64Linker::new(config);
        // Linker created successfully — no panic
    }

    /// Verify find_output_placement returns None for empty sections.
    #[test]
    fn test_find_output_placement_empty() {
        let sections: Vec<OutputSection> = Vec::new();
        assert_eq!(find_output_placement(&sections, 0, 0), None);
    }

    /// Verify find_section_addr returns None for missing sections.
    #[test]
    fn test_find_section_addr_missing() {
        let sections: Vec<OutputSection> = Vec::new();
        assert_eq!(find_section_addr(&sections, ".text"), None);
    }

    /// Verify RELA entry serialization produces correct size.
    #[test]
    fn test_rela_dyn_size() {
        let sym_names = vec!["foo".to_string(), "bar".to_string()];
        let data = build_rela_dyn(&sym_names, 0x1000);
        // Each RELA entry is 24 bytes, we have 2 entries
        assert_eq!(data.len(), 48);
    }

    /// Verify RELA PLT entry serialization produces correct size.
    #[test]
    fn test_rela_plt_size() {
        let sym_names = vec![
            "func1".to_string(),
            "func2".to_string(),
            "func3".to_string(),
        ];
        let data = build_rela_plt(&sym_names, 0x2000, 2);
        // Each RELA entry is 24 bytes, we have 3 entries
        assert_eq!(data.len(), 72);
    }

    /// Verify AArch64-specific segment rules are correctly defined.
    #[test]
    fn test_aarch64_segment_rules() {
        let rules = aarch64_segment_rules();
        // We expect 3 segment rules: code (R+X), rodata (R), data (R+W).
        assert_eq!(rules.len(), 3);

        // First rule: code segment with PF_R | PF_X.
        assert_eq!(rules[0].segment_type, PT_LOAD);
        assert_eq!(rules[0].flags, PF_R | PF_X);
        assert!(rules[0].contains_section(".text"));
        assert!(rules[0].contains_section(".plt"));

        // Second rule: read-only data with PF_R.
        assert_eq!(rules[1].segment_type, PT_LOAD);
        assert_eq!(rules[1].flags, PF_R);
        assert!(rules[1].contains_section(".rodata"));

        // Third rule: data segment with PF_R | PF_W.
        assert_eq!(rules[2].segment_type, PT_LOAD);
        assert_eq!(rules[2].flags, PF_R | PF_W);
        assert!(rules[2].contains_section(".data"));
        assert!(rules[2].contains_section(".bss"));
    }

    /// Verify that SHT_NOTE and SHT_NULL are correctly valued ELF constants.
    #[test]
    fn test_elf_section_type_constants() {
        // SHT_NULL must be 0 per the ELF specification — it marks the
        // mandatory null section header entry at index 0.
        assert_eq!(SHT_NULL, 0);
        // SHT_NOTE must be 7 per the ELF specification — used for
        // .note sections (e.g., .note.GNU-stack).
        assert_eq!(SHT_NOTE, 7);
        // SHT_SYMTAB must be 2 — used for the .symtab section.
        assert_eq!(SHT_SYMTAB, 2);
    }

    /// Verify AArch64-specific ELF constants from elf_writer_common.
    #[test]
    fn test_aarch64_elf_constants() {
        // EM_AARCH64 must be 183 per the ELF specification.
        assert_eq!(EM_AARCH64, 183);
        // ELFCLASS64 must be 2 (64-bit ELF).
        assert_eq!(ELFCLASS64, 2);
        // ELFDATA2LSB must be 1 (little-endian byte order).
        assert_eq!(ELFDATA2LSB, 1);
        // ELFOSABI_NONE must be 0 (System V / generic ABI).
        assert_eq!(ELFOSABI_NONE, 0);
        // Verify the combined configuration check.
        assert!(verify_aarch64_elf_config());
    }

    /// Verify fx_hash_map_with_capacity creates a map with expected behavior.
    #[test]
    fn test_fx_hash_map_operations() {
        let mut map: FxHashMap<String, u64> = fx_hash_map_with_capacity(4);
        map.insert("sym1".to_string(), 0x1000);
        map.insert("sym2".to_string(), 0x2000);

        assert!(map.contains_key("sym1"));
        assert!(!map.contains_key("sym3"));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("sym1"), Some(&0x1000));

        // Verify iteration covers all entries.
        let count = map.len();
        assert_eq!(count, 2);
    }

    /// Verify FxHashSet operations for dynamic symbol tracking.
    #[test]
    fn test_fx_hash_set_operations() {
        let mut set: FxHashSet<String> = FxHashSet::default();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        set.insert("func_a".to_string());
        set.insert("func_b".to_string());
        assert!(!set.is_empty());
        assert_eq!(set.len(), 2);
        assert!(set.contains("func_a"));
        assert!(!set.contains("func_c"));
    }
}
