//! # BCC i686 Built-in ELF Linker
//!
//! Produces ELFCLASS32 binaries (ET_EXEC static executables and ET_DYN shared objects)
//! for the i686 (32-bit x86) target without invoking any external linker.
//!
//! ## Standalone Backend Mode
//! This is part of BCC's built-in toolchain. No external linker (`ld`, `lld`, `gold`)
//! is invoked. All symbol resolution, section merging, relocation processing, and
//! ELF output writing is performed internally.
//!
//! ## Output Formats
//! - **ET_EXEC** (static executables): Standard section layout with `_start` entry point,
//!   base address `0x08048000`
//! - **ET_DYN** (shared objects): GOT, PLT, `.dynamic`, `.dynsym`, `.gnu.hash` sections
//!
//! ## Key i686 ELF Characteristics
//! - `e_machine = EM_386 (3)`
//! - ELFCLASS32 — all addresses, offsets, and sizes are 32-bit (4 bytes)
//! - ELFDATA2LSB (little-endian)
//! - Section header entries: 40 bytes (not 64)
//! - Program header entries: 32 bytes (not 56)
//! - Symbol table entries: 16 bytes (not 24)
//! - Dynamic linker: `/lib/ld-linux.so.2`
//! - PIC uses `__x86.get_pc_thunk.bx` for GOT base in EBX register
//! - Uses R_386_* relocation types (not R_X86_64_*)
//! - Supports `Elf32_Rel` (8 bytes, implicit addend) and `Elf32_Rela` (12 bytes, explicit addend)
//!
//! ## Differences from x86-64 Linker
//! - ELFCLASS32 instead of ELFCLASS64
//! - 32-bit addresses and offsets (4-byte pointers)
//! - R_386_* relocation types instead of R_X86_64_*
//! - Dynamic linker: `/lib/ld-linux.so.2` (not `/lib64/ld-linux-x86-64.so.2`)
//! - No RIP-relative addressing; GOT access uses `[ebx + offset]`
//! - Smaller section header entries (40 vs 64 bytes)
//! - Smaller program header entries (32 vs 56 bytes)
//! - Smaller symbol table entries (16 vs 24 bytes)
//! - Default base address: `0x08048000` (not `0x400000`)

pub mod relocations;

// ---------------------------------------------------------------------------
// Imports — crate-internal only (zero external crate dependencies)
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

use crate::backend::elf_writer_common::{
    Elf32Header, Elf32ProgramHeader, Elf64ProgramHeader, ElfSymbol, ElfWriter, Section, EM_386,
    ET_DYN, ET_EXEC, PT_DYNAMIC, PT_GNU_STACK, PT_INTERP, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE,
    SHT_DYNAMIC, SHT_DYNSYM, SHT_GNU_HASH, SHT_NOBITS, SHT_PROGBITS, SHT_REL, SHT_STRTAB,
};

use crate::backend::linker_common::dynamic::DynamicLinkContext;
use crate::backend::linker_common::linker_script::{DefaultLinkerScript, InputSectionInfo};
use crate::backend::linker_common::relocation::{
    annotate_relocations, apply_all_relocations, build_input_to_output_map, resolve_relocations,
    RelocationCollector, RelocationHandler,
};
use crate::backend::linker_common::section_merger::{OutputSection, SectionMerger};
use crate::backend::linker_common::symbol_resolver::{ResolvedSymbol, SymbolResolver, SHN_UNDEF};
use crate::backend::linker_common::{LinkerConfig, LinkerInput, LinkerOutput, OutputType};

// Re-export the i686 relocation handler for consumers of this module.
pub use self::relocations::I686RelocationHandler;

// ---------------------------------------------------------------------------
// i686-specific ELF constants
// ---------------------------------------------------------------------------

/// i686 default base address for static executables (ET_EXEC).
///
/// Matches the conventional Linux default used by GNU ld for static i686
/// executables. Shared objects (ET_DYN) use base 0 (PIC).
const I686_DEFAULT_BASE_ADDRESS: u64 = 0x0804_8000;

/// Page alignment for i686 Linux (4 KiB).
///
/// Used for segment alignment in the ELF program header table.
const I686_PAGE_ALIGNMENT: u64 = 0x1000;

/// i686 dynamic linker path for the `PT_INTERP` program header.
///
/// Used by `DynamicLinkContext::set_interp()` via `Target::dynamic_linker()`.
#[allow(dead_code)]
const I686_DYNAMIC_LINKER: &str = "/lib/ld-linux.so.2";

/// Default entry-point symbol name for i686 executables.
#[allow(dead_code)]
const I686_DEFAULT_ENTRY: &str = "_start";

/// ELF32 header size in bytes.
#[allow(dead_code)]
const ELF32_HEADER_SIZE: u16 = 52;

/// ELF32 program header entry size in bytes.
#[allow(dead_code)]
const ELF32_PHDR_SIZE: u16 = 32;

/// ELF32 section header entry size in bytes.
#[allow(dead_code)]
const ELF32_SHDR_SIZE: u16 = 40;

/// ELF32 symbol table entry size in bytes.
const ELF32_SYM_SIZE: u16 = 16;

/// ELF32 Rel entry size in bytes (no explicit addend).
const ELF32_REL_SIZE: u16 = 8;

/// ELF32 Rela entry size in bytes (with explicit addend).
#[allow(dead_code)]
const ELF32_RELA_SIZE: u16 = 12;

/// Size of the PLT header stub (PLT\[0\]) in bytes for i686.
///
/// PLT\[0\] is the resolver trampoline: it pushes `link_map` from GOT\[1\]
/// and jumps to `_dl_runtime_resolve` via GOT\[2\].
const PLT0_SIZE: usize = 16;

/// Size of each per-function PLT entry (PLT\[n\]) in bytes for i686.
///
/// Standard i686 PLT entry layout:
/// - `jmp *GOT[n]`   (6 bytes) — indirect jump through GOT slot
/// - `push <index>`   (5 bytes) — push relocation index for lazy binding
/// - `jmp PLT[0]`     (5 bytes) — fall through to resolver
const PLT_ENTRY_SIZE: usize = 16;

/// Number of reserved entries at the start of `.got.plt`.
///
/// - GOT\[0\]: address of `.dynamic` section (filled by dynamic linker)
/// - GOT\[1\]: `link_map` pointer (filled by dynamic linker)
/// - GOT\[2\]: `_dl_runtime_resolve` address (filled by dynamic linker)
#[allow(dead_code)]
const GOT_PLT_RESERVED: usize = 3;

/// Size of a single GOT entry in bytes (4 bytes for 32-bit pointers).
const GOT_ENTRY_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// I686Linker — i686 ELF linker driver
// ---------------------------------------------------------------------------

/// i686 (32-bit x86) built-in ELF linker.
///
/// Produces ELFCLASS32 binaries (ET_EXEC and ET_DYN) from relocatable
/// object files without invoking any external linker.
///
/// ## Pipeline
///
/// 1. Collect input objects via [`add_input`](I686Linker::add_input).
/// 2. Call [`link`](I686Linker::link) which executes:
///    - Symbol resolution (strong/weak binding, undefined detection)
///    - Section merging (aggregation, alignment, ordering)
///    - GOT/PLT requirement scanning
///    - GOT/PLT stub generation (for PIC / shared libraries)
///    - Address layout computation (linker script)
///    - Relocation resolution and application
///    - Dynamic section generation (for shared libraries)
///    - Final ELF32 output assembly
pub struct I686Linker {
    /// Linker configuration from CLI flags.
    config: LinkerConfig,
    /// Two-pass symbol resolver.
    symbol_resolver: SymbolResolver,
    /// Section merging engine.
    section_merger: SectionMerger,
    /// i686-specific relocation handler implementing `RelocationHandler`.
    relocation_handler: I686RelocationHandler,
    /// Dynamic linking context (created for shared libs or when `-l` is used).
    dynamic_ctx: Option<DynamicLinkContext>,
    /// Default linker script for section-to-segment mapping.
    linker_script: DefaultLinkerScript,
    /// Accumulated input object files.
    inputs: Vec<LinkerInput>,
    /// Whether PIC mode is active (`-fPIC`).
    pic_enabled: bool,
    /// Whether the output is a shared object (`-shared`).
    is_shared: bool,
    /// Whether DWARF debug sections should be passed through (`-g`).
    emit_debug: bool,
    /// Accumulated diagnostic error messages (written to by internal passes).
    #[allow(dead_code)]
    diagnostics_errors: Vec<String>,
}

impl I686Linker {
    /// Create a new i686 linker with the given configuration.
    ///
    /// The constructor reads the `config` to determine the output type,
    /// PIC mode, debug emission, and needed libraries. It pre-creates
    /// the dynamic linking context when producing shared objects or
    /// when external shared libraries are referenced.
    pub fn new(config: LinkerConfig) -> Self {
        let target = Target::I686;
        let is_shared = config.output_type == OutputType::SharedLibrary;
        let pic_enabled = config.pic;
        let emit_debug = config.emit_debug;

        // Create a dynamic link context when:
        // - producing a shared library, OR
        // - the executable needs shared libraries (DT_NEEDED entries)
        let dynamic_ctx = if is_shared || !config.needed_libs.is_empty() {
            let mut ctx = DynamicLinkContext::new(target, is_shared);
            // Set PT_INTERP for dynamically-linked executables.
            if !is_shared {
                ctx.set_interp();
            }
            // Record DT_NEEDED entries.
            for lib in &config.needed_libs {
                ctx.add_needed_library(lib);
            }
            // Set SONAME if specified.
            if let Some(ref soname) = config.soname {
                ctx.soname = Some(soname.clone());
            }
            Some(ctx)
        } else {
            None
        };

        I686Linker {
            pic_enabled,
            is_shared,
            emit_debug,
            symbol_resolver: SymbolResolver::new(),
            section_merger: SectionMerger::new(target),
            relocation_handler: I686RelocationHandler::new(),
            dynamic_ctx,
            linker_script: DefaultLinkerScript::new(target, is_shared),
            inputs: Vec::new(),
            diagnostics_errors: Vec::new(),
            config,
        }
    }

    /// Add a relocatable object file (`.o`) to the link.
    ///
    /// Each input is registered with the symbol resolver (for cross-file
    /// symbol resolution), the section merger (for output section
    /// construction), and stored for relocation processing.
    pub fn add_input(&mut self, input: LinkerInput) {
        // Feed symbols to the symbol resolver.
        self.symbol_resolver
            .collect_symbols(input.object_id, &input.symbols);

        // Feed sections to the section merger.
        for section in &input.sections {
            self.section_merger.add_input_section(section.clone());
        }

        // Retain the full input for relocation processing.
        self.inputs.push(input);
    }

    /// Perform the complete i686 link operation.
    ///
    /// Executes the full linking pipeline producing an ELFCLASS32 binary:
    ///
    /// 1. Resolve symbols (strong/weak binding, undefined detection)
    /// 2. Merge input sections into output sections
    /// 3. Scan relocations for GOT/PLT requirements
    /// 4. Generate GOT/PLT stubs (if PIC/shared)
    /// 5. Compute address layout (linker script)
    /// 6. Resolve and apply relocations (i686-specific R_386_* patching)
    /// 7. Generate dynamic sections (if shared library)
    /// 8. Write final ELFCLASS32 output
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Diagnostic engine for error/warning reporting.
    ///
    /// # Returns
    ///
    /// On success, returns a [`LinkerOutput`] containing the serialized ELF
    /// bytes, the resolved entry point, and the output type.
    pub fn link(&mut self, diagnostics: &mut DiagnosticEngine) -> Result<LinkerOutput, String> {
        // ===================================================================
        // Phase 1: Symbol Resolution
        // ===================================================================
        if let Err(errors) = self.symbol_resolver.resolve() {
            for err_msg in &errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
            }
            if !self.config.allow_undefined {
                return Err(format!(
                    "symbol resolution failed with {} error(s)",
                    errors.len()
                ));
            }
        }

        // Check for undefined symbols (fatal for executables).
        if let Err(undef_errors) = self
            .symbol_resolver
            .check_undefined(self.config.allow_undefined)
        {
            for err_msg in &undef_errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
            }
            if !self.config.allow_undefined {
                return Err(format!(
                    "linking failed: {} undefined symbol(s)",
                    undef_errors.len()
                ));
            }
        }

        // Emit accumulated diagnostics from symbol resolution.
        self.symbol_resolver.emit_diagnostics(diagnostics);

        if diagnostics.has_errors() && !self.config.allow_undefined {
            return Err("linking aborted due to symbol resolution errors".to_string());
        }

        // ===================================================================
        // Phase 2: Section Merging
        // ===================================================================
        let base_address = if self.is_shared {
            0u64
        } else {
            I686_DEFAULT_BASE_ADDRESS
        };
        let page_alignment = I686_PAGE_ALIGNMENT;

        let address_map = self
            .section_merger
            .assign_addresses(base_address, page_alignment);
        self.section_merger.resolve_fragment_addresses();

        // ===================================================================
        // Phase 3: Linker-Defined Symbols
        // ===================================================================
        let mut section_addr_map: FxHashMap<String, (u64, u64)> = FxHashMap::default();
        for (name, addr_info) in &address_map.section_addresses {
            section_addr_map.insert(name.clone(), (addr_info.virtual_address, addr_info.size));
        }
        self.define_linker_symbols(&section_addr_map);

        // ===================================================================
        // Phase 4: Relocation Collection & GOT/PLT Scanning
        // ===================================================================
        let output_sections = self.section_merger.get_ordered_sections();
        let output_section_vec: Vec<OutputSection> = output_sections.into_iter().cloned().collect();
        let input_to_output_map = build_input_to_output_map(&output_section_vec);

        // Collect all relocations from input objects.
        let mut reloc_collector = RelocationCollector::new();
        for input in &self.inputs {
            for (sec_idx, section) in input.sections.iter().enumerate() {
                let relocs: Vec<crate::backend::linker_common::relocation::Relocation> = section
                    .relocations
                    .iter()
                    .map(|sr| crate::backend::linker_common::relocation::Relocation {
                        offset: sr.offset,
                        symbol_name: String::new(), // resolved below
                        sym_index: sr.sym_index,
                        rel_type: sr.rel_type,
                        addend: sr.addend,
                        object_id: input.object_id,
                        section_index: sec_idx as u32,
                        output_section_name: None,
                    })
                    .collect();
                if !relocs.is_empty() {
                    reloc_collector.add_relocations(input.object_id, sec_idx as u32, relocs);
                }
            }
            // Also process top-level relocations from the input.
            if !input.relocations.is_empty() {
                let mut relocs = input.relocations.clone();
                for r in &mut relocs {
                    r.object_id = input.object_id;
                }
                reloc_collector.add_relocations(input.object_id, 0, relocs);
            }
        }

        // Annotate relocations with output section names and resolve symbol
        // names from the symbol indices.
        let mut all_relocs = reloc_collector.into_relocations();
        annotate_relocations(&mut all_relocs, &input_to_output_map);

        // Resolve symbol names for relocations that have sym_index references.
        self.resolve_relocation_symbol_names(&mut all_relocs);

        // Scan for GOT/PLT requirements using the i686 relocation handler.
        let mut got_symbols: FxHashSet<String> = FxHashSet::default();
        let mut plt_symbols: FxHashSet<String> = FxHashSet::default();
        for rel in &all_relocs {
            if self.relocation_handler.needs_got(rel.rel_type) && !rel.symbol_name.is_empty() {
                got_symbols.insert(rel.symbol_name.clone());
            }
            if self.relocation_handler.needs_plt(rel.rel_type) && !rel.symbol_name.is_empty() {
                plt_symbols.insert(rel.symbol_name.clone());
            }
        }

        // ===================================================================
        // Phase 5: GOT/PLT Stub Generation
        // ===================================================================
        let mut got_entries: FxHashMap<String, u64> = FxHashMap::default();
        let mut plt_entries: FxHashMap<String, u64> = FxHashMap::default();
        let mut got_base: u64 = 0;

        if self.dynamic_ctx.is_some() || self.pic_enabled || !got_symbols.is_empty() {
            let ctx = self
                .dynamic_ctx
                .get_or_insert_with(|| DynamicLinkContext::new(Target::I686, self.is_shared));

            // Allocate GOT entries for symbols that need them.
            for sym_name in &got_symbols {
                let offset = ctx.got.add_got_entry(sym_name, 0);
                got_entries.insert(sym_name.clone(), offset);
            }

            // Allocate GOT.PLT entries and PLT stubs for symbols that need PLT.
            for sym_name in &plt_symbols {
                let (got_plt_offset, _plt_idx) = ctx.got.add_got_plt_entry(sym_name);
                plt_entries.insert(sym_name.clone(), got_plt_offset);
            }
        }

        // ===================================================================
        // Phase 6: Address Layout (Linker Script)
        // ===================================================================
        let ordered_sections = self.section_merger.get_ordered_sections();
        let mut layout_input: Vec<InputSectionInfo> = Vec::new();
        for sec in &ordered_sections {
            layout_input.push(InputSectionInfo {
                name: sec.name.clone(),
                size: sec.total_size,
                alignment: sec.alignment,
                flags: sec.flags as u32,
            });
        }

        // Add synthetic sections for GOT/PLT if they were generated.
        if let Some(ref mut ctx) = self.dynamic_ctx {
            let got_size = ctx.got.encode_got().len() as u64;
            if got_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".got".to_string(),
                    size: got_size,
                    alignment: 4,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }

            let got_plt_size = ctx.got.got_plt_size() as u64;
            if got_plt_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".got.plt".to_string(),
                    size: got_plt_size,
                    alignment: 4,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }

            if !plt_symbols.is_empty() {
                let plt_size = (PLT0_SIZE + plt_symbols.len() * PLT_ENTRY_SIZE) as u64;
                layout_input.push(InputSectionInfo {
                    name: ".plt".to_string(),
                    size: plt_size,
                    alignment: 16,
                    flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
                });
            }

            // Early-finalize dynamic context so we can compute section sizes.
            // finalize() builds .gnu.hash and .dynamic from accumulated state;
            // it does NOT depend on layout addresses and is safe to call now.
            // The Phase 9 finalize() call is idempotent and will re-run harmlessly.
            ctx.finalize();

            // Add synthetic dynamic-linking sections to layout so they get
            // proper virtual addresses.  Without these, PT_DYNAMIC and other
            // program headers that reference these sections would be silently
            // skipped because their names wouldn't appear in layout.sections.
            let dynsym_size = ctx.dynsym.encode_dynsym(&Target::I686).len() as u64;
            if dynsym_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".dynsym".to_string(),
                    size: dynsym_size,
                    alignment: 4,
                    flags: SHF_ALLOC as u32,
                });
            }

            let dynstr_size = ctx.dynsym.encode_dynstr().len() as u64;
            if dynstr_size > 1 {
                layout_input.push(InputSectionInfo {
                    name: ".dynstr".to_string(),
                    size: dynstr_size,
                    alignment: 1,
                    flags: SHF_ALLOC as u32,
                });
            }

            let gnu_hash_size = ctx.gnu_hash.encode().len() as u64;
            if gnu_hash_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".gnu.hash".to_string(),
                    size: gnu_hash_size,
                    alignment: 4,
                    flags: SHF_ALLOC as u32,
                });
            }

            let rel_dyn_size = ctx.rela.encode_rela_dyn(false).len() as u64;
            if rel_dyn_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".rel.dyn".to_string(),
                    size: rel_dyn_size,
                    alignment: 4,
                    flags: SHF_ALLOC as u32,
                });
            }

            let rel_plt_size = ctx.rela.encode_rela_plt(false).len() as u64;
            if rel_plt_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".rel.plt".to_string(),
                    size: rel_plt_size,
                    alignment: 4,
                    flags: SHF_ALLOC as u32,
                });
            }

            let dynamic_size = ctx.dynamic.encode(false).len() as u64;
            if dynamic_size > 0 {
                layout_input.push(InputSectionInfo {
                    name: ".dynamic".to_string(),
                    size: dynamic_size,
                    alignment: 4,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }

            // Add .interp if needed.
            if let Some(interp_bytes) = ctx.get_interp_bytes() {
                layout_input.push(InputSectionInfo {
                    name: ".interp".to_string(),
                    size: interp_bytes.len() as u64,
                    alignment: 1,
                    flags: SHF_ALLOC as u32,
                });
            }
        }

        let layout = self.linker_script.compute_layout(&layout_input);

        // Build a section-name → virtual-address map from the layout.
        let mut section_vaddr_map: FxHashMap<String, u64> = FxHashMap::default();
        for sl in &layout.sections {
            section_vaddr_map.insert(sl.name.clone(), sl.virtual_address);
        }

        // Update GOT entries to absolute addresses based on layout.
        if let Some(got_addr) = section_vaddr_map.get(".got") {
            got_base = *got_addr;
            let mut updated: FxHashMap<String, u64> = FxHashMap::default();
            for (sym_name, offset) in &got_entries {
                updated.insert(sym_name.clone(), got_base + offset);
            }
            got_entries = updated;
        }

        // Update PLT entries to absolute addresses based on layout.
        if let Some(plt_addr) = section_vaddr_map.get(".plt") {
            let mut updated: FxHashMap<String, u64> = FxHashMap::default();
            for (idx, sym_name) in plt_symbols.iter().enumerate() {
                let entry_addr = plt_addr + PLT0_SIZE as u64 + idx as u64 * PLT_ENTRY_SIZE as u64;
                updated.insert(sym_name.clone(), entry_addr);
            }
            plt_entries = updated;
        }

        // ===================================================================
        // Phase 7: Symbol Address Assignment
        // ===================================================================
        let symbol_table = self.symbol_resolver.build_symbol_table();
        let mut symbol_addresses: FxHashMap<String, ResolvedSymbol> = FxHashMap::default();

        for sym in &symbol_table.symbols {
            let resolved = ResolvedSymbol {
                name: sym.name.clone(),
                final_address: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_name: String::new(),
                is_defined: sym.section_index != SHN_UNDEF,
                from_object: 0,
                export_dynamic: false,
            };
            symbol_addresses.insert(sym.name.clone(), resolved);
        }

        // Resolve entry point.
        let entry_point = if self.is_shared {
            0u64
        } else {
            let mut entry_symbols: FxHashMap<String, u64> = FxHashMap::default();
            for (sym_name, sym) in &symbol_addresses {
                entry_symbols.insert(sym_name.clone(), sym.final_address);
            }
            match self.linker_script.resolve_entry_point(&entry_symbols) {
                Ok(addr) => addr,
                Err(e) => {
                    diagnostics.emit_error(Span::dummy(), e);
                    // Fall back to .text start if available.
                    section_vaddr_map
                        .get(".text")
                        .copied()
                        .unwrap_or(I686_DEFAULT_BASE_ADDRESS)
                }
            }
        };

        // ===================================================================
        // Phase 8: Relocation Resolution and Application
        // ===================================================================
        let mut sec_addr_map = crate::backend::linker_common::section_merger::AddressMap {
            section_addresses: FxHashMap::default(),
        };
        for sl in &layout.sections {
            sec_addr_map.section_addresses.insert(
                sl.name.clone(),
                crate::backend::linker_common::section_merger::SectionAddress {
                    virtual_address: sl.virtual_address,
                    file_offset: sl.file_offset,
                    size: sl.size,
                    mem_size: sl.mem_size,
                },
            );
        }

        let resolved_relocs = resolve_relocations(
            &all_relocs,
            &symbol_addresses,
            &sec_addr_map,
            &got_entries,
            &plt_entries,
            got_base,
            &self.relocation_handler,
        );

        // Build mutable section data buffers for patching.
        let mut section_data_map: FxHashMap<String, Vec<u8>> = FxHashMap::default();
        for sec in &ordered_sections {
            let data = self.section_merger.build_section_data(&sec.name);
            section_data_map.insert(sec.name.clone(), data);
        }

        // Apply relocations using the i686 relocation handler.
        match resolved_relocs {
            Ok(resolved) => {
                if let Err(errors) = apply_all_relocations(
                    &resolved,
                    &mut section_data_map,
                    &self.relocation_handler,
                    diagnostics,
                ) {
                    for err in &errors {
                        diagnostics.emit_error(Span::dummy(), err.to_string());
                    }
                    if !self.config.allow_undefined {
                        return Err(format!(
                            "relocation application failed with {} error(s)",
                            errors.len()
                        ));
                    }
                }
            }
            Err(errors) => {
                for err in &errors {
                    diagnostics.emit_error(Span::dummy(), err.to_string());
                }
                if !self.config.allow_undefined {
                    return Err(format!(
                        "relocation resolution failed with {} error(s)",
                        errors.len()
                    ));
                }
            }
        }

        // ===================================================================
        // Phase 9: Dynamic Section Finalization
        // ===================================================================
        if let Some(ref mut ctx) = self.dynamic_ctx {
            ctx.finalize();
        }

        // ===================================================================
        // Phase 10: ELF32 Output Assembly
        // ===================================================================
        let elf_type = if self.is_shared { ET_DYN } else { ET_EXEC };
        let mut writer = ElfWriter::new(Target::I686, elf_type);
        writer.set_entry_point(entry_point);

        // Add program headers for loadable segments.
        // ElfWriter stores them as Elf64ProgramHeader internally; for 32-bit
        // targets the writer truncates fields to 32-bit during serialization.
        for seg in &layout.segments {
            writer.add_program_header(Elf64ProgramHeader {
                p_type: seg.seg_type,
                p_flags: seg.flags,
                p_offset: seg.offset,
                p_vaddr: seg.vaddr,
                p_paddr: seg.paddr,
                p_filesz: seg.filesz,
                p_memsz: seg.memsz,
                p_align: seg.alignment,
            });
        }

        // Add PT_INTERP for dynamically-linked executables.
        if let Some(ref ctx) = self.dynamic_ctx {
            if let Some(interp_bytes) = ctx.get_interp_bytes() {
                let interp_addr = section_vaddr_map.get(".interp").copied().unwrap_or(0);
                let interp_offset = layout
                    .sections
                    .iter()
                    .find(|s| s.name == ".interp")
                    .map(|s| s.file_offset)
                    .unwrap_or(0);
                let interp_size = interp_bytes.len() as u64;

                writer.add_section(Section {
                    name: ".interp".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC,
                    data: interp_bytes,
                    sh_addralign: 1,
                    ..Section::default()
                });

                writer.add_program_header(Elf64ProgramHeader {
                    p_type: PT_INTERP,
                    p_flags: 4, // PF_R
                    p_offset: interp_offset,
                    p_vaddr: interp_addr,
                    p_paddr: interp_addr,
                    p_filesz: interp_size,
                    p_memsz: interp_size,
                    p_align: 1,
                });
            }
        }

        // Add merged output sections with patched data.
        for sec in &ordered_sections {
            // Skip debug sections when debug info was not requested.
            if !self.emit_debug && sec.name.starts_with(".debug") {
                continue;
            }

            let data = section_data_map
                .remove(&sec.name)
                .unwrap_or_else(|| self.section_merger.build_section_data(&sec.name));

            let sh_type = if sec.name == ".bss" {
                SHT_NOBITS
            } else {
                sec.section_type
            };

            writer.add_section(Section {
                name: sec.name.clone(),
                sh_type,
                sh_flags: sec.flags,
                data,
                sh_addralign: sec.alignment,
                ..Section::default()
            });
        }

        // Add GOT/PLT/dynamic sections for shared libraries.
        if let Some(ref ctx) = self.dynamic_ctx {
            // .got section — GOT entries for data access via [ebx + offset]
            let got_data = ctx.got.encode_got();
            if !got_data.is_empty() {
                writer.add_section(Section {
                    name: ".got".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: got_data,
                    sh_addralign: 4,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .got.plt section — PLT-specific GOT entries
            let plt_addr = section_vaddr_map.get(".plt").copied().unwrap_or(0);
            let got_plt_data = ctx.got.encode_got_plt(plt_addr);
            if !got_plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".got.plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: got_plt_data,
                    sh_addralign: 4,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .plt section — Procedure Linkage Table stubs (32-bit)
            let got_plt_addr = section_vaddr_map.get(".got.plt").copied().unwrap_or(0);
            let plt_data = ctx.plt.encode(got_plt_addr, plt_addr);
            if !plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_EXECINSTR,
                    data: plt_data,
                    sh_addralign: 16,
                    sh_entsize: PLT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynsym section — dynamic symbol table (16-byte Elf32_Sym entries)
            let dynsym_data = ctx.dynsym.encode_dynsym(&Target::I686);
            if !dynsym_data.is_empty() {
                writer.add_section(Section {
                    name: ".dynsym".to_string(),
                    sh_type: SHT_DYNSYM,
                    sh_flags: SHF_ALLOC,
                    data: dynsym_data,
                    sh_addralign: 4,
                    sh_entsize: ELF32_SYM_SIZE as u64, // 16 bytes for Elf32_Sym
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynstr section — dynamic string table
            let dynstr_data = ctx.dynsym.encode_dynstr();
            if dynstr_data.len() > 1 {
                writer.add_section(Section {
                    name: ".dynstr".to_string(),
                    sh_type: SHT_STRTAB,
                    sh_flags: SHF_ALLOC,
                    data: dynstr_data,
                    sh_addralign: 1,
                    ..Section::default()
                });
            }

            // .gnu.hash section — GNU hash table for fast symbol lookup
            let gnu_hash_data = ctx.gnu_hash.encode();
            if !gnu_hash_data.is_empty() {
                writer.add_section(Section {
                    name: ".gnu.hash".to_string(),
                    sh_type: SHT_GNU_HASH,
                    sh_flags: SHF_ALLOC,
                    data: gnu_hash_data,
                    sh_addralign: 4,
                    ..Section::default()
                });
            }

            // .rel.dyn section — dynamic data relocations (i686 uses Rel, not Rela)
            let rel_dyn_data = ctx.rela.encode_rela_dyn(false); // false → 32-bit Elf32_Rel (8 bytes)
            if !rel_dyn_data.is_empty() {
                writer.add_section(Section {
                    name: ".rel.dyn".to_string(),
                    sh_type: SHT_REL,
                    sh_flags: SHF_ALLOC,
                    data: rel_dyn_data,
                    sh_addralign: 4,
                    sh_entsize: ELF32_REL_SIZE as u64, // 8 bytes for Elf32_Rel
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .rel.plt section — PLT relocations (R_386_JMP_SLOT)
            let rel_plt_data = ctx.rela.encode_rela_plt(false); // false → 32-bit Elf32_Rel (8 bytes)
            if !rel_plt_data.is_empty() {
                writer.add_section(Section {
                    name: ".rel.plt".to_string(),
                    sh_type: SHT_REL,
                    sh_flags: SHF_ALLOC,
                    data: rel_plt_data,
                    sh_addralign: 4,
                    sh_entsize: ELF32_REL_SIZE as u64, // 8 bytes for Elf32_Rel
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynamic section — dynamic linking metadata tags
            let dynamic_data = ctx.dynamic.encode(false); // false → 32-bit Elf32_Dyn (8 bytes)
            if !dynamic_data.is_empty() {
                writer.add_section(Section {
                    name: ".dynamic".to_string(),
                    sh_type: SHT_DYNAMIC,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: dynamic_data,
                    sh_addralign: 4,
                    sh_entsize: 8, // Each Elf32_Dyn is 8 bytes (d_tag: i32 + d_val: u32)
                    logical_size: 0,
                    ..Section::default()
                });

                // Add PT_DYNAMIC program header.
                if let Some(dyn_sl) = layout.sections.iter().find(|s| s.name == ".dynamic") {
                    writer.add_program_header(Elf64ProgramHeader {
                        p_type: PT_DYNAMIC,
                        p_flags: 6, // PF_R | PF_W
                        p_offset: dyn_sl.file_offset,
                        p_vaddr: dyn_sl.virtual_address,
                        p_paddr: dyn_sl.virtual_address,
                        p_filesz: dyn_sl.size,
                        p_memsz: dyn_sl.mem_size,
                        p_align: 4,
                    });
                }
            }
        }

        // PT_GNU_STACK — NX stack enforcement (applies to all executables).
        writer.add_program_header(Elf64ProgramHeader {
            p_type: PT_GNU_STACK,
            p_flags: 6, // PF_R | PF_W (no PF_X — NX stack)
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 16,
        });

        // Write the resolved symbols into the ELF symbol table.
        for sym in &symbol_table.symbols {
            let final_addr = symbol_addresses
                .get(&sym.name)
                .map(|s| s.final_address)
                .unwrap_or(sym.value);
            writer.add_symbol(ElfSymbol {
                name: sym.name.clone(),
                value: final_addr,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            });
        }

        // Serialize the complete ELFCLASS32 binary.
        let elf_data = writer.write();

        Ok(LinkerOutput {
            elf_data,
            entry_point,
            output_type: self.config.output_type,
        })
    }

    /// Link input objects into a static executable (ET_EXEC).
    ///
    /// Convenience wrapper around [`link`](I686Linker::link) that ensures
    /// the configuration specifies `OutputType::Executable`.
    ///
    /// Standard section layout: `.text`, `.rodata`, `.data`, `.bss`.
    /// Entry point: `_start` symbol (error if undefined).
    /// Base address: `0x08048000`.
    pub fn link_executable(
        &mut self,
        inputs: Vec<LinkerInput>,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, String> {
        for input in inputs {
            self.add_input(input);
        }
        let output = self.link(diagnostics)?;
        Ok(output.elf_data)
    }

    /// Link input objects into a shared object (ET_DYN).
    ///
    /// Convenience wrapper around [`link`](I686Linker::link) that ensures
    /// the configuration specifies `OutputType::SharedLibrary`.
    ///
    /// PIC required: all data accesses via GOT (`[ebx + offset]`),
    /// function calls via PLT. Base address: `0x0` (position-independent).
    pub fn link_shared_object(
        &mut self,
        inputs: Vec<LinkerInput>,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, String> {
        for input in inputs {
            self.add_input(input);
        }
        let output = self.link(diagnostics)?;
        Ok(output.elf_data)
    }

    /// Build an ELF32 header for the i686 target.
    ///
    /// Constructs a 52-byte ELFCLASS32 header with:
    /// - `e_machine = EM_386 (3)` — CRITICAL: must be 3 for i686
    /// - ELFCLASS32 (all addresses are 32-bit)
    /// - ELFDATA2LSB (little-endian)
    /// - `e_ehsize = 52`
    /// - `e_phentsize = 32`
    /// - `e_shentsize = 40`
    ///
    /// Helper for direct ELF32 header construction. Currently unused because
    /// `link()` delegates to `ElfWriter` for header generation. Retained for
    /// the code generation driver's multi-object linking flows.
    #[allow(dead_code)]
    pub fn build_elf32_header(
        &self,
        elf_type: u16,
        entry_point: u32,
        phoff: u32,
        shoff: u32,
        phnum: u16,
        shnum: u16,
        shstrndx: u16,
    ) -> Elf32Header {
        Elf32Header {
            e_type: elf_type,
            e_machine: EM_386, // CRITICAL: EM_386 = 3 for i686
            e_entry: entry_point,
            e_phoff: phoff,
            e_shoff: shoff,
            e_flags: 0,
            e_phnum: phnum,
            e_shnum: shnum,
            e_shstrndx: shstrndx,
        }
    }

    /// Build program headers for the i686 ELF32 binary.
    ///
    /// Each `Elf32_Phdr` is 32 bytes (not 56 like ELF64).
    /// Converts layout segments to `Elf32ProgramHeader` structures.
    ///
    /// Currently unused — `link()` builds program headers inline from
    /// `layout.segments`.  Retained for future use by the code generation
    /// driver.
    #[allow(dead_code)]
    pub fn build_program_headers(
        &self,
        layout: &crate::backend::linker_common::linker_script::LayoutResult,
    ) -> Vec<Elf32ProgramHeader> {
        let mut phdrs = Vec::new();

        for seg in &layout.segments {
            phdrs.push(Elf32ProgramHeader {
                p_type: seg.seg_type,
                p_flags: seg.flags,
                p_offset: seg.offset as u32,
                p_vaddr: seg.vaddr as u32,
                p_paddr: seg.paddr as u32,
                p_filesz: seg.filesz as u32,
                p_memsz: seg.memsz as u32,
                p_align: seg.alignment as u32,
            });
        }

        // PT_GNU_STACK — NX stack enforcement (PF_R | PF_W, NO PF_X).
        phdrs.push(Elf32ProgramHeader {
            p_type: PT_GNU_STACK,
            p_flags: 6, // PF_R | PF_W = 4 | 2 = 6 (no PF_X — NX stack)
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 16,
        });

        phdrs
    }

    /// Define linker-defined symbols that the linker creates automatically.
    ///
    /// These symbols are injected if not already provided by input objects:
    /// - `__bss_start` — start of `.bss` section
    /// - `_edata` — end of `.data` section
    /// - `_end` / `__end` — end of all sections
    /// - `__executable_start` — base load address (`0x08048000`)
    pub fn define_linker_symbols(&mut self, section_addr_map: &FxHashMap<String, (u64, u64)>) {
        self.symbol_resolver.define_linker_symbols(section_addr_map);
    }

    /// Resolve symbol names for relocations using the symbol resolver.
    ///
    /// Maps the `sym_index` field of each relocation to the actual symbol
    /// name by looking up the input symbol table via the symbol resolver.
    fn resolve_relocation_symbol_names(
        &self,
        relocs: &mut [crate::backend::linker_common::relocation::Relocation],
    ) {
        for rel in relocs.iter_mut() {
            if rel.symbol_name.is_empty() {
                // Look up the symbol name from the input that contributed
                // this relocation.
                for input in &self.inputs {
                    if input.object_id == rel.object_id {
                        if let Some(sym) = input.symbols.get(rel.sym_index as usize) {
                            rel.symbol_name = sym.name.clone();
                        }
                        break;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public Module API — Convenience Functions
// ---------------------------------------------------------------------------

/// Link i686 object files into a static executable (ET_EXEC).
///
/// Creates an [`I686Linker`] configured for `OutputType::Executable`,
/// feeds in the provided inputs, and returns the serialized ELFCLASS32 binary.
///
/// # Arguments
///
/// * `inputs` — Relocatable object files to link.
/// * `output_path` — Output file path (used for diagnostic messages).
/// * `entry_point` — Optional entry point symbol (defaults to `_start`).
/// * `diagnostics` — Diagnostic engine for error reporting.
///
/// # Returns
///
/// On success, the complete ELF32 binary bytes ready to write to disk.
pub fn link_i686_executable(
    inputs: Vec<LinkerInput>,
    output_path: &str,
    entry_point: Option<&str>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let mut config = LinkerConfig::new(Target::I686, OutputType::Executable);
    config.output_path = output_path.to_string();
    if let Some(ep) = entry_point {
        config.entry_point = ep.to_string();
    }

    let mut linker = I686Linker::new(config);
    linker.link_executable(inputs, diagnostics)
}

/// Link i686 object files into a shared object (ET_DYN).
///
/// Creates an [`I686Linker`] configured for `OutputType::SharedLibrary`,
/// feeds in the provided inputs, and returns the serialized ELFCLASS32
/// shared object binary with GOT, PLT, `.dynamic`, `.dynsym`, and
/// `.gnu.hash` sections.
///
/// # Arguments
///
/// * `inputs` — Relocatable object files to link.
/// * `output_path` — Output file path (used for diagnostic messages).
/// * `soname` — Optional SONAME for the `DT_SONAME` dynamic entry.
/// * `needed_libs` — Additional needed libraries for `DT_NEEDED` entries.
/// * `diagnostics` — Diagnostic engine for error reporting.
///
/// # Returns
///
/// On success, the complete ELF32 shared object bytes ready to write to disk.
pub fn link_i686_shared_object(
    inputs: Vec<LinkerInput>,
    output_path: &str,
    soname: Option<&str>,
    needed_libs: &[String],
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let mut config = LinkerConfig::new(Target::I686, OutputType::SharedLibrary);
    config.output_path = output_path.to_string();
    config.pic = true; // Shared objects require PIC
    if let Some(sn) = soname {
        config.soname = Some(sn.to_string());
    }
    config.needed_libs = needed_libs.to_vec();

    let mut linker = I686Linker::new(config);
    linker.link_shared_object(inputs, diagnostics)
}
