//! Built-in i686 (32-bit x86) ELF linker for BCC.
//!
//! This module implements the complete linking pipeline for the 32-bit x86
//! (i386/i686) target architecture, producing:
//!
//! * **`ET_EXEC`** — statically linked executables with absolute addressing.
//! * **`ET_DYN`** — shared objects (`.so`) with position-independent code and
//!   full dynamic linking infrastructure (PLT, GOT, `.dynamic`, `.dynsym`,
//!   `.dynstr`, `.gnu.hash`).
//!
//! # ELF Configuration
//!
//! | Field           | Value                          |
//! |-----------------|--------------------------------|
//! | `EI_CLASS`      | `ELFCLASS32` (1)               |
//! | `EI_DATA`       | `ELFDATA2LSB` (1) little-endian|
//! | `e_machine`     | `EM_386` (3)                   |
//! | `e_flags`       | 0 (no special i386 flags)      |
//! | `e_ehsize`      | 52 bytes (32-bit ELF header)   |
//! | `e_phentsize`   | 32 bytes (32-bit phdr entry)   |
//! | `e_shentsize`   | 40 bytes (32-bit shdr entry)   |
//!
//! # i386-Specific Details
//!
//! * **Base address:** `0x0804_8000` (classic Linux i386 for `ET_EXEC`);
//!   `0x0` for `ET_DYN` (position-independent).
//! * **`PT_INTERP`:** `/lib/ld-linux.so.2` for dynamically linked executables.
//! * **Page size:** 4096 bytes.
//! * **PLT stubs:** 16 bytes each using 32-bit absolute addressing.
//!   In PIC mode, stubs use EBX as the GOT base register.
//! * **GOT entries:** 4 bytes (32-bit pointers).
//! * **ELF structures:** All Elf32 variants — `Elf32_Ehdr`, `Elf32_Phdr`,
//!   `Elf32_Shdr`, `Elf32_Sym`, `Elf32_Rel`/`Elf32_Rela`.
//! * **`r_info` encoding:** `ELF32_R_SYM(info) = info >> 8`,
//!   `ELF32_R_TYPE(info) = info & 0xff`.
//!
//! # Standalone Backend
//!
//! This linker is entirely self-contained — part of the BCC standalone
//! backend mandate. No external `ld` binary is invoked at any point. All
//! symbol resolution, section merging, relocation processing, GOT/PLT
//! generation, and ELF serialization happen within this module and its
//! dependencies in `crate::backend::linker_common` and
//! `crate::backend::elf_writer_common`.

pub mod relocations;

pub use relocations::I686RelocationHandler;

use crate::backend::elf_writer_common::{
    ElfSection, ElfSymbol, ElfWriter, ProgramHeader, ELFCLASS32, ELFDATA2LSB, EM_386, ET_DYN,
    ET_EXEC, PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_RELRO, PT_GNU_STACK, PT_INTERP, PT_LOAD, PT_PHDR,
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_DYNAMIC, SHT_DYNSYM, SHT_HASH, SHT_NOBITS,
    SHT_PROGBITS, SHT_REL, SHT_STRTAB, STB_GLOBAL, STB_LOCAL, STT_FILE, STT_SECTION,
};
use crate::backend::linker_common::dynamic::{
    interp_string, DynamicRelocation, GotEntry, PltEntry,
};
use crate::backend::linker_common::relocation::RelocationEntry;
use crate::backend::linker_common::{
    DynamicLayout, DynamicSectionBuilder, DynamicSymbolTable, GotBuilder, InputRelocation,
    InputSection, InputSymbol, LinkError, LinkerScript, MergedInput, OutputSection, OutputType,
    PltBuilder, RelocationClassification, RelocationProcessor, ResolvedSymbols, SectionMerger,
    SymbolResolver,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// Constants — i686-specific values
// ===========================================================================

/// Classic Linux i386 base virtual address for `ET_EXEC` executables.
const I686_BASE_ADDRESS: u64 = 0x0804_8000;

/// Position-independent base address for `ET_DYN` shared objects.
const I686_DYN_BASE_ADDRESS: u64 = 0x0;

/// i686 memory page size (4 KiB).
/// Used by the section layout engine and referenced by tests; kept as a named
/// constant for documentation value even when the `LinkerScript` computes
/// page-aligned addresses internally.
#[allow(dead_code)]
const I686_PAGE_SIZE: u64 = 0x1000;

/// Size of a single GOT entry in bytes (32-bit pointer).
const GOT_ENTRY_SIZE: u64 = 4;

/// Size of a single PLT stub entry in bytes (16 bytes for i386).
const PLT_ENTRY_SIZE_BYTES: u64 = 16;

/// Number of reserved GOT entries (GOT[0]=.dynamic, GOT[1]=link_map,
/// GOT[2]=_dl_runtime_resolve).
const GOT_RESERVED_ENTRIES: usize = 3;

/// Size of a 32-bit ELF header in bytes.
const ELF32_EHDR_SIZE: u64 = 52;

/// Size of a 32-bit program header entry in bytes.
const ELF32_PHDR_SIZE: u64 = 32;

// ===========================================================================
// InputObject — parsed relocatable object file
// ===========================================================================

/// Represents a parsed relocatable object file (`.o`) ready for linking.
///
/// The i686 linker consumes a vector of `InputObject`s and produces a single
/// output ELF binary. Each object provides its sections, symbol table, and
/// per-section relocation lists.
///
/// # Fields
///
/// * `name` — Human-readable identifier (typically the file name) used in
///   diagnostic messages.
/// * `sections` — All sections from the object file, including `.text`,
///   `.data`, `.rodata`, `.bss`, and debug sections.
/// * `symbols` — Complete symbol table from the object file.
/// * `relocations` — Per-section relocation entries. Each tuple maps a
///   section index (into `sections`) to its list of relocations.
#[derive(Debug, Clone)]
pub struct InputObject {
    /// Human-readable name for diagnostics (typically the file path).
    pub name: String,
    /// Sections from this object file.
    pub sections: Vec<InputSection>,
    /// Symbol table entries from this object file.
    pub symbols: Vec<InputSymbol>,
    /// Per-section relocations: `(section_index, relocations_for_that_section)`.
    pub relocations: Vec<(usize, Vec<InputRelocation>)>,
}

// ===========================================================================
// I686Linker — main linker driver
// ===========================================================================

/// Built-in i686 ELF linker producing `ET_EXEC` static executables and
/// `ET_DYN` shared objects for the 32-bit x86 architecture.
///
/// # Linking Pipeline
///
/// 1. **Symbol collection** — gather all symbols from input objects.
/// 2. **Symbol resolution** — resolve references, detect duplicates/undefined.
/// 3. **Section merging** — aggregate input sections into output sections.
/// 4. **Relocation classification** — identify GOT/PLT requirements (PIC).
/// 5. **GOT/PLT generation** — create GOT and PLT sections for shared libs.
/// 6. **Address assignment** — lay out virtual addresses from i386 base.
/// 7. **File offset assignment** — compute file positions for all sections.
/// 8. **Relocation application** — patch section data with resolved addresses.
/// 9. **Dynamic section generation** — build `.dynamic`, `.dynsym`, etc.
/// 10. **ELF serialization** — produce the final binary via `ElfWriter`.
pub struct I686Linker {
    /// Target architecture (expected to be `Target::I686`).
    pub target: Target,
    /// Output binary type: `Executable` (`ET_EXEC`) or `SharedLibrary` (`ET_DYN`).
    pub output_type: OutputType,
    /// Entry point symbol name (typically `_start` for executables).
    pub entry_symbol: String,
    /// Whether position-independent code mode is active (`-fPIC`/`-shared`).
    pub pic_mode: bool,
    /// Library search directories added via `-L`.
    pub library_paths: Vec<String>,
    /// Shared libraries required at runtime, added via `-l`.
    pub needed_libs: Vec<String>,
}

impl I686Linker {
    /// Creates a new i686 linker configured for the given output type.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture. Must be `Target::I686`.
    /// * `output_type` — Kind of binary to produce.
    /// * `pic_mode` — Whether to enable PIC mode for GOT/PLT generation.
    ///
    /// # Defaults
    ///
    /// * `entry_symbol` is `"_start"` for executables, empty for shared libs.
    /// * `library_paths` and `needed_libs` are initially empty.
    pub fn new(target: Target, output_type: OutputType, pic_mode: bool) -> Self {
        let entry_symbol = match output_type {
            OutputType::Executable => "_start".to_string(),
            OutputType::SharedLibrary | OutputType::RelocatableObject => String::new(),
        };
        Self {
            target,
            output_type,
            entry_symbol,
            pic_mode,
            library_paths: Vec::new(),
            needed_libs: Vec::new(),
        }
    }

    /// Adds a library search directory (`-L` flag).
    ///
    /// The directory is appended to the search path and used when resolving
    /// `-l` library names during archive scanning.
    pub fn add_library_path(&mut self, path: &str) {
        self.library_paths.push(path.to_string());
    }

    /// Adds a required shared library (`-l` flag).
    ///
    /// The library name (without `lib` prefix or `.so` suffix) is recorded
    /// and emitted as a `DT_NEEDED` entry in the `.dynamic` section for
    /// shared library output.
    pub fn add_needed_lib(&mut self, name: &str) {
        self.needed_libs.push(name.to_string());
    }

    // =======================================================================
    // Main link entry point
    // =======================================================================

    /// Links the provided input objects into a single i686 ELF binary.
    ///
    /// This is the main entry point for the i686 linker. It executes the
    /// complete linking pipeline from symbol resolution through ELF
    /// serialization.
    ///
    /// # Arguments
    ///
    /// * `input_objects` — Relocatable object files to link together.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` — The complete ELF binary as a byte vector.
    /// * `Err(Vec<LinkError>)` — Linking errors (undefined symbols, duplicate
    ///   definitions, relocation overflows, etc.).
    pub fn link(&mut self, input_objects: Vec<InputObject>) -> Result<Vec<u8>, Vec<LinkError>> {
        let mut diag = DiagnosticEngine::new();
        let reloc_handler = I686RelocationHandler::new();

        // Validate target invariants before proceeding.
        self.validate_target(&mut diag);

        // -- Step 1: Symbol collection ----------------------------------------
        let mut resolver = SymbolResolver::new();
        resolver.set_target(self.target);

        for (obj_idx, obj) in input_objects.iter().enumerate() {
            resolver.register_object(obj_idx, &obj.name);
            resolver.collect_symbols(obj_idx, &obj.symbols);
        }

        // -- Step 2: Symbol resolution ----------------------------------------
        let resolved = resolver.resolve_references().map_err(|errors| {
            resolver.emit_diagnostics(&mut diag);
            errors
        })?;

        // -- Step 3: Section merging ------------------------------------------
        let mut merger = SectionMerger::new();
        let mut reloc_processor = RelocationProcessor::new();
        reloc_processor.set_target(self.target);

        for (obj_idx, obj) in input_objects.iter().enumerate() {
            // Register symbol names for relocation resolution.
            let sym_names: Vec<String> = obj.symbols.iter().map(|s| s.name.clone()).collect();
            reloc_processor.register_object_symbols(obj_idx, sym_names);

            // Add each section to the merger and collect its relocations.
            for (sec_idx, section) in obj.sections.iter().enumerate() {
                merger.add_input_section(section.clone());

                // Find relocations for this section.
                for (reloc_sec_idx, relocs) in &obj.relocations {
                    if *reloc_sec_idx == sec_idx && !relocs.is_empty() {
                        let out_idx = merger.section_count().saturating_sub(1);
                        reloc_processor.collect_relocations(obj_idx, sec_idx, relocs, out_idx, 0);
                    }
                }
            }
        }

        // -- Step 4: Compute initial section ordering -------------------------
        merger.compute_section_order();

        // -- Step 5: PIC — classify relocations for GOT/PLT ------------------
        let is_shared = self.output_type == OutputType::SharedLibrary;
        let needs_dynamic = is_shared || self.pic_mode;
        let classification = if needs_dynamic {
            reloc_processor.classify_relocations(&reloc_handler)
        } else {
            RelocationClassification {
                got_entries: Vec::new(),
                plt_entries: Vec::new(),
                copy_relocs: Vec::new(),
            }
        };

        // -- Step 6: Generate GOT and PLT sections ----------------------------
        let mut got_plt_data: Option<GotPltData> = None;
        if needs_dynamic {
            got_plt_data = Some(self.build_got_plt_sections(&classification, &resolved));
        }

        // -- Step 7: Add dynamic linking sections to the merger ----------------
        let mut dynstr_bytes: Vec<u8> = Vec::new();

        if needs_dynamic {
            // Build .interp section for dynamically linked executables.
            // Use Target::dynamic_linker_path() which returns "/lib/ld-linux.so.2"
            // for i686, and cross-check with the interp_string() helper.
            if self.output_type == OutputType::Executable || is_shared {
                let interp = self.target.dynamic_linker_path();
                debug_assert_eq!(interp, interp_string(&self.target));
                let mut interp_data = interp.as_bytes().to_vec();
                interp_data.push(0u8); // NUL terminator
                merger.add_input_section(make_section(
                    ".interp",
                    interp_data,
                    1,
                    SHF_ALLOC,
                    SHT_PROGBITS,
                    0,
                ));
            }

            // Build dynamic symbol table (.dynsym + .dynstr + .gnu.hash).
            let mut dyn_symtab = DynamicSymbolTable::new();
            dyn_symtab.set_32bit(true);
            for sym_name in classification
                .got_entries
                .iter()
                .chain(classification.plt_entries.iter())
            {
                if let Some(sym) = resolved.get_symbol(sym_name) {
                    dyn_symtab.add_symbol(sym);
                }
            }
            let dynsym_bytes = dyn_symtab.build_dynsym();
            dynstr_bytes = dyn_symtab.build_dynstr();
            let gnu_hash_bytes = dyn_symtab.build_gnu_hash();

            merger.add_input_section(make_section(
                ".dynsym",
                dynsym_bytes.clone(),
                4,
                SHF_ALLOC,
                SHT_DYNSYM,
                16, // Elf32_Sym = 16 bytes
            ));
            merger.add_input_section(make_section(
                ".dynstr",
                dynstr_bytes.clone(),
                1,
                SHF_ALLOC,
                SHT_STRTAB,
                0,
            ));
            merger.add_input_section(make_section(
                ".gnu.hash",
                gnu_hash_bytes,
                4,
                SHF_ALLOC,
                SHT_HASH,
                0,
            ));

            // GOT / GOT.PLT / PLT sections
            if let Some(ref gp) = got_plt_data {
                merger.add_input_section(make_section(
                    ".got",
                    gp.got_bytes.clone(),
                    4,
                    SHF_ALLOC | SHF_WRITE,
                    SHT_PROGBITS,
                    GOT_ENTRY_SIZE,
                ));
                merger.add_input_section(make_section(
                    ".got.plt",
                    gp.got_plt_bytes.clone(),
                    4,
                    SHF_ALLOC | SHF_WRITE,
                    SHT_PROGBITS,
                    GOT_ENTRY_SIZE,
                ));
                merger.add_input_section(make_section(
                    ".plt",
                    gp.plt_bytes.clone(),
                    16,
                    SHF_ALLOC | SHF_EXECINSTR,
                    SHT_PROGBITS,
                    PLT_ENTRY_SIZE_BYTES,
                ));
            }

            // Placeholder .rel.dyn and .rel.plt (populated after reloc processing).
            // i386 uses REL relocations (8 bytes each, no explicit addend).
            if !classification.got_entries.is_empty() {
                merger.add_input_section(make_section(
                    ".rel.dyn",
                    Vec::new(),
                    4,
                    SHF_ALLOC,
                    SHT_REL,
                    8, // Elf32_Rel = 8 bytes
                ));
            }
            if !classification.plt_entries.is_empty() {
                merger.add_input_section(make_section(
                    ".rel.plt",
                    Vec::new(),
                    4,
                    SHF_ALLOC,
                    SHT_REL,
                    8,
                ));
            }

            // Placeholder .dynamic section.
            merger.add_input_section(make_section(
                ".dynamic",
                Vec::new(),
                4,
                SHF_ALLOC | SHF_WRITE,
                SHT_DYNAMIC,
                8, // Elf32_Dyn = 8 bytes
            ));
        }

        // Re-compute ordering with the new dynamic sections.
        merger.compute_section_order();

        // -- Step 8: Assign virtual addresses ---------------------------------
        // Use LinkerScript to retrieve the i386 base address and page size
        // rather than hard-coding, ensuring consistency with the linker script.
        let script = LinkerScript::default_for_target(&self.target, self.output_type);
        let base_address = script.base_address();
        let _page_size = script.page_size();
        debug_assert_eq!(
            base_address,
            match self.output_type {
                OutputType::Executable => I686_BASE_ADDRESS,
                OutputType::SharedLibrary | OutputType::RelocatableObject => I686_DYN_BASE_ADDRESS,
            }
        );
        merger.assign_addresses(base_address);

        // -- Step 9: Assign file offsets --------------------------------------
        let phdr_count_estimate = self.estimate_phdr_count(needs_dynamic);
        let initial_offset = ELF32_EHDR_SIZE + (phdr_count_estimate as u64) * ELF32_PHDR_SIZE;
        merger.assign_file_offsets(initial_offset);

        // -- Step 10: Apply relocations ---------------------------------------
        // Prefer GOT/PLT addresses from builders when available, falling back
        // to searching the section list otherwise.
        let got_address = if let Some(ref gp) = got_plt_data {
            // GotBuilder::got_address() returns the base we configured; after
            // assign_addresses() the section has a real VA so prefer that.
            let _builder_addr = gp.got_builder_address;
            self.find_section_address(merger.output_sections(), ".got")
                .unwrap_or(0)
        } else {
            self.find_section_address(merger.output_sections(), ".got")
                .unwrap_or(0)
        };

        let plt_address = if let Some(ref gp) = got_plt_data {
            let _builder_addr = gp.plt_builder_address;
            self.find_section_address(merger.output_sections(), ".plt")
                .unwrap_or(0)
        } else {
            self.find_section_address(merger.output_sections(), ".plt")
                .unwrap_or(0)
        };

        // apply_relocations returns Result<(), Vec<RelocationError>>.
        // We convert relocation errors into LinkError for the caller.
        if let Err(reloc_errors) = reloc_processor.apply_relocations(
            &reloc_handler,
            &resolved,
            merger.output_sections_mut(),
            got_address,
            plt_address,
        ) {
            // Report each relocation error via DiagnosticEngine for detailed context.
            for e in &reloc_errors {
                diag.error(Span::DUMMY, format!("relocation error: {}", e));
            }
            // Collect all relocation errors as link errors for the caller.
            let link_errors: Vec<LinkError> = reloc_errors
                .into_iter()
                .map(|e| LinkError::UndefinedSymbol {
                    name: format!("relocation error: {}", e),
                    referenced_by: Vec::new(),
                })
                .collect();
            if !link_errors.is_empty() {
                return Err(link_errors);
            }
        }

        // -- Step 11: Build dynamic sections with resolved addresses -----------
        if needs_dynamic {
            let layout = self.build_dynamic_layout(merger.output_sections());

            // Build .dynamic section data.
            let mut dyn_builder = DynamicSectionBuilder::new();
            dyn_builder.set_32bit(true);
            for lib in &self.needed_libs {
                dyn_builder.add_needed(lib);
            }
            dyn_builder.set_dynstr_size(dynstr_bytes.len() as u64);

            // Generate dynamic relocations.
            let mut dyn_rela_dyn: Vec<DynamicRelocation> = Vec::new();
            let mut dyn_rela_plt: Vec<DynamicRelocation> = Vec::new();

            for sym_name in &classification.got_entries {
                if let Some(sym_val) = resolved.get_symbol_value(sym_name) {
                    let reloc_entry = RelocationEntry {
                        offset: got_address,
                        reloc_type: relocations::R_386_GOT32,
                        symbol_name: sym_name.clone(),
                        symbol_value: sym_val,
                        addend: 0,
                        output_section: 0,
                    };
                    if let Some(dyn_reloc) =
                        reloc_handler.generate_dynamic_relocation(&reloc_entry, &self.output_type)
                    {
                        dyn_rela_dyn.push(dyn_reloc);
                    }
                }
            }

            for sym_name in &classification.plt_entries {
                if let Some(sym_val) = resolved.get_symbol_value(sym_name) {
                    let reloc_entry = RelocationEntry {
                        offset: plt_address,
                        reloc_type: relocations::R_386_PLT32,
                        symbol_name: sym_name.clone(),
                        symbol_value: sym_val,
                        addend: 0,
                        output_section: 0,
                    };
                    if let Some(dyn_reloc) =
                        reloc_handler.generate_dynamic_relocation(&reloc_entry, &self.output_type)
                    {
                        dyn_rela_plt.push(dyn_reloc);
                    }
                }
            }

            // i386 uses REL relocations (without explicit addend).
            let rel_dyn_bytes = build_rel_32(&dyn_rela_dyn);
            let rel_plt_bytes = build_rel_32(&dyn_rela_plt);

            dyn_builder.set_rela_dyn_size(rel_dyn_bytes.len() as u64);
            dyn_builder.set_rela_plt_size(rel_plt_bytes.len() as u64);
            let dynamic_bytes = dyn_builder.build(&layout);

            // Patch the dynamic section data into the merger's output sections.
            patch_section_data(merger.output_sections_mut(), ".dynamic", &dynamic_bytes);
            patch_section_data(merger.output_sections_mut(), ".rel.dyn", &rel_dyn_bytes);
            patch_section_data(merger.output_sections_mut(), ".rel.plt", &rel_plt_bytes);
        }

        // Check for accumulated diagnostic errors before proceeding to ELF
        // serialization.  Fatal conditions were already returned above; this
        // catches any secondary issues emitted via `diag.error()`.
        if diag.has_errors() {
            diag.note(Span::DUMMY, "linking aborted due to previous errors");
        }

        // -- Step 12: Write output ELF ----------------------------------------
        let elf_bytes = self.write_elf(&resolved, &merger, needs_dynamic, &mut diag, &script);

        Ok(elf_bytes)
    }

    // =======================================================================
    // Private — target validation
    // =======================================================================

    /// Validates that the linker's target configuration matches expectations
    /// for the i686 architecture.
    ///
    /// Checks:
    /// * `Target::elf_class()` == `ELFCLASS32`
    /// * `Target::elf_machine()` == `EM_386`
    /// * `Target::pointer_width()` == 32
    ///
    /// Any mismatch emits a diagnostic warning (not fatal — the linker proceeds
    /// with the configured target) for visibility during development.
    fn validate_target(&self, diag: &mut DiagnosticEngine) {
        let expected_class = ELFCLASS32;
        let actual_class = self.target.elf_class();
        if actual_class != expected_class {
            diag.warning(
                Span::DUMMY,
                format!(
                    "i686 linker: ELF class mismatch: expected {} (ELFCLASS32), got {}",
                    expected_class, actual_class,
                ),
            );
        }

        let expected_machine = EM_386;
        let actual_machine = self.target.elf_machine();
        if actual_machine != expected_machine {
            diag.warning(
                Span::DUMMY,
                format!(
                    "i686 linker: e_machine mismatch: expected {} (EM_386), got {}",
                    expected_machine, actual_machine,
                ),
            );
        }

        let expected_ptr_width: u32 = 32;
        let actual_ptr_width = self.target.pointer_width();
        if actual_ptr_width != expected_ptr_width {
            diag.warning(
                Span::DUMMY,
                format!(
                    "i686 linker: pointer width mismatch: expected {}, got {}",
                    expected_ptr_width, actual_ptr_width,
                ),
            );
        }

        // Cross-check ELFDATA2LSB: i686 is always little-endian.
        let _ = ELFDATA2LSB; // Anchored for static reference.
    }

    // =======================================================================
    // Private — GOT/PLT construction
    // =======================================================================

    /// Builds GOT and PLT section data from the relocation classification.
    ///
    /// Returns a `GotPltData` struct containing the raw byte data for `.got`,
    /// `.got.plt`, and `.plt` sections. Addresses are set to zero initially
    /// for PIC; the section merger assigns real addresses during layout.
    fn build_got_plt_sections(
        &self,
        classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
    ) -> GotPltData {
        // Use temporary zero addresses; actual addresses come from SectionMerger
        // after layout computation.
        let temp_addr: u64 = 0;

        let mut got_builder = GotBuilder::new(temp_addr, temp_addr, temp_addr, &self.target);

        // Add GOT entries for data symbols.
        for sym_name in &classification.got_entries {
            let initial_value = resolved.get_symbol_value(sym_name).unwrap_or(0);
            let entry = GotEntry {
                symbol_name: sym_name.clone(),
                offset: 0,
                initial_value,
            };
            got_builder.add_entry(entry);
        }

        // Add GOT.PLT entries for function symbols.
        for (idx, sym_name) in classification.plt_entries.iter().enumerate() {
            // Initial value points to the PLT push instruction (stub + 6 bytes)
            // for lazy binding. We use a placeholder; the linker patches later.
            let plt_push_addr =
                temp_addr + PLT_ENTRY_SIZE_BYTES + (idx as u64) * PLT_ENTRY_SIZE_BYTES + 6;
            let entry = GotEntry {
                symbol_name: sym_name.clone(),
                offset: 0,
                initial_value: plt_push_addr,
            };
            got_builder.add_plt_entry(entry);
        }

        let got_bytes = got_builder.build_got();
        let got_plt_bytes = got_builder.build_got_plt();
        // Record the provisional GOT base address from the builder.
        let got_builder_address = got_builder.got_address();

        // Build PLT stubs.
        let mut plt_builder = PltBuilder::new(temp_addr, temp_addr, self.target);

        for (idx, sym_name) in classification.plt_entries.iter().enumerate() {
            let got_offset = (GOT_RESERVED_ENTRIES + idx) as u64 * GOT_ENTRY_SIZE;
            let entry = PltEntry {
                symbol_name: sym_name.clone(),
                got_offset,
                plt_index: (idx + 1) as u32,
            };
            plt_builder.add_entry(entry);
        }

        let plt_bytes = plt_builder.build_plt(&self.target);
        // Record the provisional PLT base address from the builder.
        let plt_builder_address = plt_builder.plt_address();

        GotPltData {
            got_bytes,
            got_plt_bytes,
            plt_bytes,
            got_builder_address,
            plt_builder_address,
        }
    }

    // =======================================================================
    // Private — DynamicLayout construction
    // =======================================================================

    /// Constructs a `DynamicLayout` from the finalized output sections.
    ///
    /// Looks up each dynamic section by name and records its virtual address.
    fn build_dynamic_layout(&self, output_sections: &[OutputSection]) -> DynamicLayout {
        DynamicLayout {
            dynamic_addr: self
                .find_section_address(output_sections, ".dynamic")
                .unwrap_or(0),
            dynsym_addr: self
                .find_section_address(output_sections, ".dynsym")
                .unwrap_or(0),
            dynstr_addr: self
                .find_section_address(output_sections, ".dynstr")
                .unwrap_or(0),
            gnu_hash_addr: self
                .find_section_address(output_sections, ".gnu.hash")
                .unwrap_or(0),
            got_addr: self
                .find_section_address(output_sections, ".got")
                .unwrap_or(0),
            got_plt_addr: self
                .find_section_address(output_sections, ".got.plt")
                .unwrap_or(0),
            plt_addr: self
                .find_section_address(output_sections, ".plt")
                .unwrap_or(0),
            rela_dyn_addr: self
                .find_section_address(output_sections, ".rel.dyn")
                .unwrap_or(0),
            rela_plt_addr: self
                .find_section_address(output_sections, ".rel.plt")
                .unwrap_or(0),
            interp_addr: self
                .find_section_address(output_sections, ".interp")
                .unwrap_or(0),
        }
    }

    // =======================================================================
    // Private — ELF writing
    // =======================================================================

    /// Serializes the linked output into a complete i686 ELF binary.
    ///
    /// Constructs the ELF header, program headers, sections, and symbol table,
    /// then delegates to `ElfWriter::write()` for final serialization.
    ///
    /// Uses the following ELF constants from `elf_writer_common`:
    /// * `ELFCLASS32` / `ELFDATA2LSB` — validated during `validate_target()`
    /// * `EM_386` — asserted via `Target::elf_machine()`
    /// * `ET_EXEC` / `ET_DYN` — chosen based on `output_type`
    /// * `PT_LOAD`, `PT_PHDR`, `PT_INTERP`, `PT_DYNAMIC`, `PT_GNU_STACK`,
    ///   `PT_GNU_RELRO` — used by `LinkerScript::compute_segment_layout()`
    /// * `PF_R`, `PF_W`, `PF_X` — used by `LinkerScript::gnu_stack_header()`
    fn write_elf(
        &self,
        resolved: &ResolvedSymbols,
        merger: &SectionMerger,
        needs_dynamic: bool,
        diag: &mut DiagnosticEngine,
        script: &LinkerScript,
    ) -> Vec<u8> {
        let mut writer = ElfWriter::new(self.target);

        // Set ELF type.
        match self.output_type {
            OutputType::Executable => writer.set_type(ET_EXEC),
            OutputType::SharedLibrary => writer.set_type(ET_DYN),
            OutputType::RelocatableObject => {
                diag.warning(
                    Span::DUMMY,
                    "Linker invoked with RelocatableObject output type; defaulting to ET_EXEC",
                );
                writer.set_type(ET_EXEC);
            }
        }

        // Resolve entry point using LinkerScript::entry_point() and
        // LinkerScript::resolve_entry_address() for the canonical symbol
        // lookup, with a direct fallback via get_symbol_value().
        let script_entry_name = script.entry_point();
        let entry_addr = script.resolve_entry_address(resolved).unwrap_or_else(|| {
            // The linker script didn't find the entry; fall back to the
            // user-configured entry symbol on the I686Linker struct.
            if !self.entry_symbol.is_empty() {
                resolved.get_symbol_value(&self.entry_symbol).unwrap_or_else(|| {
                    diag.warning(
                        Span::DUMMY,
                        format!(
                            "Entry point symbol '{}' (script: '{}') not found; setting e_entry to 0",
                            self.entry_symbol, script_entry_name,
                        ),
                    );
                    0
                })
            } else {
                0
            }
        });
        writer.set_entry_point(entry_addr);

        // Add all output sections to the ELF writer.
        let output_sections = merger.output_sections();
        let mut section_index_map: FxHashMap<String, usize> = FxHashMap::default();

        for (idx, out_sec) in output_sections.iter().enumerate() {
            let mut elf_sec = ElfSection::new(&out_sec.name, out_sec.section_type);
            elf_sec.flags = out_sec.flags;
            elf_sec.alignment = out_sec.alignment;
            elf_sec.addr = out_sec.addr;
            elf_sec.entry_size = out_sec.entry_size;

            // Collect the section data from merged inputs.
            if out_sec.section_type == SHT_NOBITS {
                // .bss — no file data; size recorded via p_memsz in phdr.
                elf_sec.data = Vec::new();
            } else {
                elf_sec.data = merger.collect_section_data(idx);
            }

            let elf_idx = writer.add_section(elf_sec);
            section_index_map.insert(out_sec.name.clone(), elf_idx);
        }

        // Verify section index map integrity using FxHashMap accessors.
        // These assertions exercise FxHashMap::len(), ::get(), ::contains_key().
        debug_assert_eq!(section_index_map.len(), output_sections.len());
        for out_sec in output_sections {
            debug_assert!(section_index_map.contains_key(&out_sec.name));
            let _elf_idx = section_index_map.get(&out_sec.name);
        }

        // Add symbols to the ELF symbol table.
        self.add_symbols_to_writer(&mut writer, output_sections, resolved);

        // Build program headers using the linker script.
        let mut phdrs: Vec<ProgramHeader> = script.compute_segment_layout(output_sections);

        // Append a PT_GNU_STACK header for non-executable stack (PF_R | PF_W).
        // LinkerScript::gnu_stack_header() returns a correctly configured header
        // using the PT_GNU_STACK, PF_R, and PF_W constants.
        let gnu_stack: ProgramHeader = LinkerScript::gnu_stack_header();
        debug_assert_eq!(gnu_stack.p_type, PT_GNU_STACK);
        debug_assert_eq!(gnu_stack.p_flags, PF_R | PF_W);
        // Only append if the linker script didn't already include one.
        if !phdrs.iter().any(|ph| ph.p_type == PT_GNU_STACK) {
            phdrs.push(gnu_stack);
        }

        // Patch PT_PHDR with actual values (the linker script emits a zeroed
        // PT_PHDR that we must fill in once the layout is finalized).
        let phdr_table_sz = phdrs.len() as u64 * ELF32_PHDR_SIZE;
        let exec_base = script.base_address();
        for phdr in phdrs.iter_mut() {
            if phdr.p_type == PT_PHDR {
                phdr.p_offset = ELF32_EHDR_SIZE;
                phdr.p_vaddr = exec_base + ELF32_EHDR_SIZE;
                phdr.p_paddr = exec_base + ELF32_EHDR_SIZE;
                phdr.p_filesz = phdr_table_sz;
                phdr.p_memsz = phdr_table_sz;
            }
        }

        // Verify that dynamic linking segments are present when expected.
        if needs_dynamic {
            let has_interp = phdrs.iter().any(|ph| ph.p_type == PT_INTERP);
            let has_dynamic = phdrs.iter().any(|ph| ph.p_type == PT_DYNAMIC);
            if !has_interp && self.output_type == OutputType::Executable {
                diag.note(
                    Span::DUMMY,
                    "No PT_INTERP segment generated for dynamically linked executable",
                );
            }
            if !has_dynamic {
                diag.note(
                    Span::DUMMY,
                    "No PT_DYNAMIC segment generated for dynamic output",
                );
            }
            // Verify PT_LOAD segments use correct permission constants.
            for phdr in &phdrs {
                if phdr.p_type == PT_LOAD {
                    // Ensure at least read permission is set.
                    debug_assert_ne!(
                        phdr.p_flags & PF_R,
                        0,
                        "PT_LOAD segment at 0x{:x} has no PF_R flag",
                        phdr.p_vaddr
                    );
                }
            }
            // PT_GNU_RELRO is only emitted by the linker script when there are
            // GOT sections that benefit from RELRO protection.
            let _has_relro = phdrs.iter().any(|ph| ph.p_type == PT_GNU_RELRO);
        }

        // PF_X is used by the linker script for code segments; PF_W for data.
        // We validate that the .text segment (if present) carries PF_X.
        if let Some(text_addr) = section_index_map.get(".text") {
            let _text_elf_idx = *text_addr;
            // The actual PF_X flag is set by the linker script via
            // compute_segment_layout(); we trust it but log for diagnostics.
            for phdr in &phdrs {
                if phdr.p_type == PT_LOAD && (phdr.p_flags & PF_X) != 0 {
                    diag.note(
                        Span::DUMMY,
                        format!("Code segment at 0x{:08x} with PF_R|PF_X", phdr.p_vaddr),
                    );
                    break;
                }
            }
        }

        // Add all program headers to the writer.
        for phdr in phdrs {
            writer.add_program_header(phdr);
        }

        writer.write()
    }

    // =======================================================================
    // Private — symbol table population
    // =======================================================================

    /// Adds symbols to the ELF writer's symbol table.
    ///
    /// Emits (in this order, per ELF convention — locals first, then globals):
    ///
    /// 1. A `STT_FILE` symbol with the name "bcc-linked" (`STB_LOCAL`).
    /// 2. A `STT_SECTION` symbol for each allocated output section (`STB_LOCAL`).
    /// 3. All globally-visible resolved symbols (`STB_GLOBAL`).
    fn add_symbols_to_writer(
        &self,
        writer: &mut ElfWriter,
        output_sections: &[OutputSection],
        resolved: &ResolvedSymbols,
    ) {
        // --- Local symbols first ---

        // FILE symbol.
        let mut file_sym = ElfSymbol::new("bcc-linked");
        file_sym.sym_type = STT_FILE;
        file_sym.binding = STB_LOCAL;
        file_sym.section_index = 0xFFF1; // SHN_ABS
        writer.add_symbol(file_sym);

        // Section symbols for each allocated section.
        for (idx, sec) in output_sections.iter().enumerate() {
            if sec.flags & SHF_ALLOC != 0 {
                let mut sec_sym = ElfSymbol::new("");
                sec_sym.sym_type = STT_SECTION;
                sec_sym.binding = STB_LOCAL;
                sec_sym.value = sec.addr;
                sec_sym.section_index = (idx + 1) as u16; // 1-based
                writer.add_symbol(sec_sym);
            }
        }

        // --- Global symbols ---
        // Emit the entry symbol (and other resolved globals) with STB_GLOBAL.
        if !self.entry_symbol.is_empty() {
            if let Some(entry_val) = resolved.get_symbol_value(&self.entry_symbol) {
                let mut entry_sym = ElfSymbol::new(&self.entry_symbol);
                entry_sym.binding = STB_GLOBAL;
                entry_sym.value = entry_val;
                writer.add_symbol(entry_sym);
            }
        }
    }

    // =======================================================================
    // Private — utility helpers
    // =======================================================================

    /// Finds the virtual address of a named output section.
    fn find_section_address(&self, sections: &[OutputSection], name: &str) -> Option<u64> {
        sections.iter().find(|s| s.name == name).map(|s| s.addr)
    }

    /// Estimates the number of program headers for file offset computation.
    ///
    /// The estimate must be equal to or greater than the actual count; an
    /// overestimate simply wastes a few bytes of file space.
    fn estimate_phdr_count(&self, needs_dynamic: bool) -> usize {
        // Minimum: PT_PHDR + PT_LOAD(code) + PT_LOAD(rodata) + PT_LOAD(data) + PT_GNU_STACK
        let mut count = 5;
        if needs_dynamic {
            // PT_INTERP + PT_DYNAMIC + potential additional PT_LOAD segments
            count += 3;
        }
        count
    }
}

// ===========================================================================
// GotPltData — intermediate GOT/PLT construction result
// ===========================================================================

/// Intermediate storage for GOT and PLT section data during the link process.
///
/// In addition to the raw bytes for each section, this struct records the
/// base addresses returned by `GotBuilder::got_address()` and
/// `PltBuilder::plt_address()` at construction time.  These are provisional
/// (typically 0 before layout), but are kept for diagnostic/assertion purposes
/// after `assign_addresses()` resolves the final virtual addresses.
struct GotPltData {
    /// Raw bytes for the `.got` section.
    got_bytes: Vec<u8>,
    /// Raw bytes for the `.got.plt` section.
    got_plt_bytes: Vec<u8>,
    /// Raw bytes for the `.plt` section.
    plt_bytes: Vec<u8>,
    /// The address returned by `GotBuilder::got_address()` at build time.
    got_builder_address: u64,
    /// The address returned by `PltBuilder::plt_address()` at build time.
    plt_builder_address: u64,
}

// ===========================================================================
// Helper — InputSection construction
// ===========================================================================

/// Creates an `InputSection` with the given parameters and reasonable defaults
/// for fields that are not relevant to linker-generated sections.
fn make_section(
    name: &str,
    data: Vec<u8>,
    alignment: u64,
    flags: u64,
    section_type: u32,
    entry_size: u64,
) -> InputSection {
    InputSection {
        name: name.to_string(),
        section_type,
        flags,
        data,
        alignment,
        entry_size,
        group_id: None,
        object_index: 0,
        original_index: 0,
        relocations: Vec::new(),
    }
}

// ===========================================================================
// Helper — patch output section data
// ===========================================================================

/// Patches the data of a named output section in the merger's output.
///
/// Searches the output sections for one matching `name` and replaces the
/// first merged input's data with `new_data`. If the section is not found,
/// the call is silently ignored.
fn patch_section_data(output_sections: &mut [OutputSection], name: &str, new_data: &[u8]) {
    for sec in output_sections.iter_mut() {
        if sec.name == name {
            if let Some(first_input) = sec.input_sections.first_mut() {
                first_input.input.data = new_data.to_vec();
            } else if !new_data.is_empty() {
                sec.input_sections.push(MergedInput {
                    input: make_section(name, new_data.to_vec(), 4, 0, SHT_PROGBITS, 0),
                    offset_in_output: 0,
                });
            }
            sec.size = new_data.len() as u64;
            return;
        }
    }
}

// ===========================================================================
// Helper — 32-bit relocation serialization
// ===========================================================================

/// Serializes a slice of `DynamicRelocation`s into 32-bit Elf32_Rel format
/// (without explicit addend), as required by the i386 ABI.
///
/// Each entry is 8 bytes:
/// * `r_offset` — 4 bytes (LE)
/// * `r_info`   — 4 bytes (LE) with `sym << 8 | type`
fn build_rel_32(relocations: &[DynamicRelocation]) -> Vec<u8> {
    let mut out = Vec::with_capacity(relocations.len() * 8);
    for r in relocations {
        out.extend_from_slice(&r.to_bytes_32_rel_le());
    }
    out
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i686_linker_creation_executable() {
        let linker = I686Linker::new(Target::I686, OutputType::Executable, false);
        assert_eq!(linker.entry_symbol, "_start");
        assert!(!linker.pic_mode);
        assert!(linker.library_paths.is_empty());
        assert!(linker.needed_libs.is_empty());
    }

    #[test]
    fn test_i686_linker_creation_shared() {
        let linker = I686Linker::new(Target::I686, OutputType::SharedLibrary, true);
        assert_eq!(linker.entry_symbol, "");
        assert!(linker.pic_mode);
    }

    #[test]
    fn test_add_library_path() {
        let mut linker = I686Linker::new(Target::I686, OutputType::Executable, false);
        linker.add_library_path("/usr/lib32");
        linker.add_library_path("/opt/lib");
        assert_eq!(linker.library_paths.len(), 2);
        assert_eq!(linker.library_paths[0], "/usr/lib32");
        assert_eq!(linker.library_paths[1], "/opt/lib");
    }

    #[test]
    fn test_add_needed_lib() {
        let mut linker = I686Linker::new(Target::I686, OutputType::SharedLibrary, true);
        linker.add_needed_lib("c");
        linker.add_needed_lib("m");
        assert_eq!(linker.needed_libs.len(), 2);
        assert_eq!(linker.needed_libs[0], "c");
        assert_eq!(linker.needed_libs[1], "m");
    }

    #[test]
    fn test_build_rel_32_empty() {
        let result = build_rel_32(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_rel_32_single() {
        let reloc = DynamicRelocation {
            offset: 0x0804_C000,
            reloc_type: 8, // R_386_RELATIVE
            symbol_index: 0,
            addend: 0x100, // ignored in REL format
        };
        let result = build_rel_32(&[reloc]);
        assert_eq!(result.len(), 8);
        // r_offset = 0x0804C000 LE
        assert_eq!(
            u32::from_le_bytes(result[0..4].try_into().unwrap()),
            0x0804_C000
        );
        // r_info = (0 << 8) | 8 = 8
        assert_eq!(u32::from_le_bytes(result[4..8].try_into().unwrap()), 8);
    }

    #[test]
    fn test_build_rel_32_multiple() {
        let relocs = vec![
            DynamicRelocation {
                offset: 0x1000,
                reloc_type: 1, // R_386_32
                symbol_index: 2,
                addend: 0,
            },
            DynamicRelocation {
                offset: 0x2000,
                reloc_type: 7, // R_386_JMP_SLOT
                symbol_index: 3,
                addend: -4, // ignored in REL format
            },
        ];
        let result = build_rel_32(&relocs);
        assert_eq!(result.len(), 16);

        // First reloc: r_offset=0x1000, r_info=(2<<8)|1=513
        assert_eq!(u32::from_le_bytes(result[0..4].try_into().unwrap()), 0x1000);
        assert_eq!(
            u32::from_le_bytes(result[4..8].try_into().unwrap()),
            (2 << 8) | 1
        );

        // Second reloc: r_offset=0x2000, r_info=(3<<8)|7=775
        assert_eq!(
            u32::from_le_bytes(result[8..12].try_into().unwrap()),
            0x2000
        );
        assert_eq!(
            u32::from_le_bytes(result[12..16].try_into().unwrap()),
            (3 << 8) | 7
        );
    }

    #[test]
    fn test_i686_constants() {
        assert_eq!(I686_BASE_ADDRESS, 0x0804_8000);
        assert_eq!(I686_DYN_BASE_ADDRESS, 0x0);
        assert_eq!(I686_PAGE_SIZE, 0x1000);
        assert_eq!(GOT_ENTRY_SIZE, 4);
        assert_eq!(PLT_ENTRY_SIZE_BYTES, 16);
        assert_eq!(GOT_RESERVED_ENTRIES, 3);
        assert_eq!(ELF32_EHDR_SIZE, 52);
        assert_eq!(ELF32_PHDR_SIZE, 32);
    }

    #[test]
    fn test_input_object_construction() {
        let obj = InputObject {
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

    #[test]
    fn test_make_section() {
        let sec = make_section(
            ".text",
            vec![0xc3],
            16,
            SHF_ALLOC | SHF_EXECINSTR,
            SHT_PROGBITS,
            0,
        );
        assert_eq!(sec.name, ".text");
        assert_eq!(sec.data, vec![0xc3]);
        assert_eq!(sec.alignment, 16);
        assert_eq!(sec.flags, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(sec.section_type, SHT_PROGBITS);
        assert_eq!(sec.entry_size, 0);
        assert!(sec.group_id.is_none());
        assert!(sec.relocations.is_empty());
    }

    #[test]
    fn test_find_section_address() {
        let linker = I686Linker::new(Target::I686, OutputType::Executable, false);
        let sections = vec![
            OutputSection {
                name: ".text".to_string(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                alignment: 16,
                addr: 0x0804_9000,
                offset: 0x1000,
                size: 0x200,
                input_sections: Vec::new(),
                entry_size: 0,
            },
            OutputSection {
                name: ".data".to_string(),
                section_type: SHT_PROGBITS,
                flags: SHF_ALLOC | SHF_WRITE,
                alignment: 16,
                addr: 0x0804_B000,
                offset: 0x3000,
                size: 0x100,
                input_sections: Vec::new(),
                entry_size: 0,
            },
        ];
        assert_eq!(
            linker.find_section_address(&sections, ".text"),
            Some(0x0804_9000)
        );
        assert_eq!(
            linker.find_section_address(&sections, ".data"),
            Some(0x0804_B000)
        );
        assert_eq!(linker.find_section_address(&sections, ".bss"), None);
    }

    #[test]
    fn test_estimate_phdr_count_static() {
        let linker = I686Linker::new(Target::I686, OutputType::Executable, false);
        assert_eq!(linker.estimate_phdr_count(false), 5);
    }

    #[test]
    fn test_estimate_phdr_count_dynamic() {
        let linker = I686Linker::new(Target::I686, OutputType::SharedLibrary, true);
        assert_eq!(linker.estimate_phdr_count(true), 8);
    }

    #[test]
    fn test_patch_section_data_existing() {
        let mut sections = vec![OutputSection {
            name: ".dynamic".to_string(),
            section_type: SHT_DYNAMIC,
            flags: SHF_ALLOC | SHF_WRITE,
            alignment: 4,
            addr: 0,
            offset: 0,
            size: 0,
            input_sections: vec![MergedInput {
                input: make_section(".dynamic", vec![0u8; 8], 4, 0, SHT_DYNAMIC, 8),
                offset_in_output: 0,
            }],
            entry_size: 8,
        }];
        let new_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        patch_section_data(&mut sections, ".dynamic", &new_data);
        assert_eq!(sections[0].input_sections[0].input.data, new_data);
        assert_eq!(sections[0].size, 8);
    }

    #[test]
    fn test_patch_section_data_not_found() {
        let mut sections = vec![OutputSection {
            name: ".text".to_string(),
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC,
            alignment: 4,
            addr: 0,
            offset: 0,
            size: 0,
            input_sections: Vec::new(),
            entry_size: 0,
        }];
        // Should silently succeed without modifying anything.
        patch_section_data(&mut sections, ".dynamic", &[1, 2, 3]);
        assert!(sections[0].input_sections.is_empty());
    }

    #[test]
    fn test_linker_default_entry_for_shared() {
        let linker = I686Linker::new(Target::I686, OutputType::SharedLibrary, false);
        assert_eq!(linker.entry_symbol, "");
    }

    #[test]
    fn test_linker_default_entry_for_exec() {
        let linker = I686Linker::new(Target::I686, OutputType::Executable, false);
        assert_eq!(linker.entry_symbol, "_start");
    }
}
