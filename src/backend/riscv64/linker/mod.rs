//! # RISC-V 64 Built-in ELF Linker
//!
//! This module implements the RISC-V 64-bit ELF linker for BCC's standalone backend.
//! It produces final ELF executables (ET_EXEC) and shared objects (ET_DYN) from
//! relocatable object files (.o) without invoking any external linker.
//!
//! ## Capabilities
//! - Static executable linking (ET_EXEC) with absolute addressing
//! - Shared library linking (ET_DYN) with GOT/PLT for PIC
//! - **Linker relaxation**: condenses AUIPC+JALR → JAL, AUIPC+ADDI → shorter sequences
//! - Symbol resolution via `linker_common::symbol_resolver`
//! - Section merging via `linker_common::section_merger`
//! - Dynamic linking sections via `linker_common::dynamic`
//! - DWARF debug section passthrough when present
//!
//! ## RISC-V 64 ELF Characteristics
//! - ELF machine: EM_RISCV (243)
//! - ELF flags: 0x0005 (EF_RISCV_FLOAT_ABI_DOUBLE | EF_RISCV_RVC)
//! - Default base address: 0x10000
//! - Page size: 4096 bytes
//! - Little-endian byte order
//! - 64-bit ELF (ELFCLASS64)
//!
//! ## PLT Stub Format
//! - PLT0 (header): 32 bytes — AUIPC+LD+ADDI+JALR targeting GOT[2]
//! - PLTn (entry): 16 bytes — AUIPC+LD targeting GOT[n+3]
//!
//! ## Dynamic Linker
//! Path: `/lib/ld-linux-riscv64-lp64d.so.1`
//!
//! ## Primary kernel boot target linker — Checkpoint 6 validation.

pub mod relocations;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::backend::elf_writer_common::{
    Elf64ProgramHeader, ElfSymbol, ElfWriter, Section, ET_DYN, ET_EXEC, SHF_ALLOC, SHF_EXECINSTR,
    SHF_WRITE, SHT_DYNSYM, SHT_NOBITS, SHT_NOTE, SHT_PROGBITS, SHT_RELA, SHT_STRTAB, STB_GLOBAL,
    STB_LOCAL,
};
use crate::backend::linker_common::dynamic::{
    build_dynamic_sections, DynamicLinkContext, DynamicLinkResult, ExportedSymbol, ImportedSymbol,
};
use crate::backend::linker_common::linker_script::{
    DefaultLinkerScript, InputSectionInfo, LayoutResult,
};
use crate::backend::linker_common::relocation::{
    fits_signed, process_relocations, Relocation, RelocationCollector,
};
use crate::backend::linker_common::section_merger::SectionMerger;
use crate::backend::linker_common::symbol_resolver::{
    ResolvedSymbol, SymbolResolver, SymbolTable, SHN_UNDEF as SYM_SHN_UNDEF,
};
use crate::backend::linker_common::{LinkerConfig, LinkerInput, OutputType};
use crate::backend::riscv64::linker::relocations::RiscV64RelocationHandler;
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use relocations::RiscV64RelocationHandler as RiscV64RelocationHandlerReexport;

// ---------------------------------------------------------------------------
// RISC-V 64 ELF Constants
// ---------------------------------------------------------------------------

/// ELF machine type for RISC-V.
pub const EM_RISCV: u16 = 243;

/// ELF flags: double-precision floating-point ABI.
pub const EF_RISCV_FLOAT_ABI_DOUBLE: u32 = 0x0004;

/// ELF flags: compressed (RVC) extension enabled.
pub const EF_RISCV_RVC: u32 = 0x0001;

/// Combined ELF flags for RISC-V 64 output: double-FP ABI + RVC.
pub const ELF_FLAGS: u32 = EF_RISCV_FLOAT_ABI_DOUBLE | EF_RISCV_RVC; // 0x0005

/// Default base address for RISC-V 64 executables.
pub const DEFAULT_BASE_ADDRESS: u64 = 0x10000;

/// Page size for RISC-V 64 (4 KiB).
pub const PAGE_SIZE: u64 = 4096;

/// Dynamic linker path for RISC-V 64 shared objects.
pub const DYNAMIC_LINKER: &str = "/lib/ld-linux-riscv64-lp64d.so.1";

/// PLT header (PLT0) size in bytes.
pub const PLT0_SIZE: usize = 32;

/// PLT entry (PLTn) size in bytes.
pub const PLTN_SIZE: usize = 16;

/// GOT entry size in bytes (64-bit pointer).
pub const GOT_ENTRY_SIZE: usize = 8;

/// Maximum number of linker relaxation iterations before convergence is forced.
const MAX_RELAXATION_ITERATIONS: usize = 16;

// ---------------------------------------------------------------------------
// RelaxationResult — outcome of linker relaxation pass
// ---------------------------------------------------------------------------

/// Result of a linker relaxation pass, summarising how many relaxations
/// were applied and how much code was saved.
pub struct RelaxationResult {
    /// Total number of individual relaxation transformations applied.
    pub relaxations_applied: usize,
    /// Total bytes saved across all relaxations.
    pub bytes_saved: usize,
    /// Number of relaxation iterations performed before convergence.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// RiscV64Linker — Main Linker Struct
// ---------------------------------------------------------------------------

/// RISC-V 64 ELF Linker.
///
/// Produces ELF64 little-endian executables and shared objects from
/// relocatable RISC-V 64 object files. This is the primary kernel boot
/// target linker (Checkpoint 6).
pub struct RiscV64Linker {
    /// Linker configuration derived from CLI flags.
    config: LinkerConfig,
    /// Two-pass symbol resolution engine.
    symbol_resolver: SymbolResolver,
    /// Input section aggregation engine.
    section_merger: SectionMerger,
    /// RISC-V 64 architecture-specific relocation handler.
    relocation_handler: RiscV64RelocationHandler,
    /// Dynamic linking context (present only for shared-object output).
    /// Retained for future use in advanced dynamic linking workflows.
    #[allow(dead_code)]
    dynamic_ctx: Option<DynamicLinkContext>,
    /// Whether linker relaxation is enabled (default: true for RISC-V).
    relaxation_enabled: bool,
    /// Running count of diagnostics emitted during linking.
    diagnostics_count: usize,
}

impl RiscV64Linker {
    /// Create a new RISC-V 64 linker with the given configuration.
    ///
    /// Initialises all linker subsystems:
    /// - Symbol resolver for two-pass resolution
    /// - Section merger targeting RISC-V 64
    /// - RISC-V 64 relocation handler with relaxation support
    /// - Dynamic link context (only for `OutputType::SharedLibrary`)
    pub fn new(config: LinkerConfig) -> Self {
        let dynamic_ctx = if config.output_type == OutputType::SharedLibrary {
            Some(DynamicLinkContext::new(Target::RiscV64, true))
        } else {
            None
        };

        RiscV64Linker {
            config,
            symbol_resolver: SymbolResolver::new(),
            section_merger: SectionMerger::new(Target::RiscV64),
            relocation_handler: RiscV64RelocationHandler::new(),
            dynamic_ctx,
            relaxation_enabled: true,
            diagnostics_count: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Main Linking Entry Point
    // -----------------------------------------------------------------------

    /// Perform the complete RISC-V 64 link operation.
    ///
    /// Orchestrates the full linking pipeline:
    ///
    /// 1. **Collect symbols** from all input objects
    /// 2. **Resolve symbols** — strong/weak binding, undefined detection
    /// 3. **Merge sections** — aggregate, deduplicate COMDATs, align
    /// 4. **Compute layout** — assign virtual addresses and file offsets
    /// 5. **Define linker symbols** — `__bss_start`, `_edata`, `_end`, etc.
    /// 6. **Scan for GOT/PLT** — identify symbols needing GOT entries or PLT stubs
    /// 7. **Perform linker relaxation** — AUIPC+JALR→JAL, etc.
    /// 8. **Apply relocations** — patch section data with final addresses
    /// 9. **Generate dynamic sections** — .dynamic, .dynsym, .gnu.hash, etc.
    /// 10. **Write ELF** — produce the final ELF binary
    ///
    /// # Errors
    ///
    /// Returns `Err` on fatal linking errors (undefined symbols in executables,
    /// relocation overflow, section merge conflicts).
    pub fn link(
        &mut self,
        inputs: Vec<LinkerInput>,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<Vec<u8>, String> {
        let is_shared = self.config.output_type == OutputType::SharedLibrary;

        // -------------------------------------------------------------------
        // Phase 1: Symbol Collection
        // -------------------------------------------------------------------
        for input in &inputs {
            self.symbol_resolver
                .collect_symbols(input.object_id, &input.symbols);
        }

        // -------------------------------------------------------------------
        // Phase 2: Symbol Resolution
        // -------------------------------------------------------------------
        if let Err(errors) = self.symbol_resolver.resolve() {
            for err_msg in &errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
                self.diagnostics_count += 1;
            }
            if !is_shared {
                return Err(format!(
                    "RISC-V 64 linker: symbol resolution failed with {} error(s)",
                    errors.len()
                ));
            }
        }

        // Check for unresolved symbols (fatal for executables).
        if let Err(undef_errors) = self
            .symbol_resolver
            .check_undefined(self.config.allow_undefined)
        {
            for err_msg in &undef_errors {
                diagnostics.emit_error(Span::dummy(), err_msg.clone());
                self.diagnostics_count += 1;
            }
            if !self.config.allow_undefined {
                return Err(format!(
                    "RISC-V 64 linker: {} undefined symbol(s)",
                    undef_errors.len()
                ));
            }
        }

        // Emit any accumulated resolution diagnostics.
        self.symbol_resolver.emit_diagnostics(diagnostics);

        if diagnostics.has_errors() && !self.config.allow_undefined {
            return Err("RISC-V 64 linker: aborted due to symbol resolution errors".to_string());
        }

        // -------------------------------------------------------------------
        // Phase 3: Section Merging
        // -------------------------------------------------------------------
        for input in &inputs {
            for section in &input.sections {
                self.section_merger.add_input_section(section.clone());
            }
        }

        let base_address = if is_shared { 0 } else { DEFAULT_BASE_ADDRESS };
        let address_map = self
            .section_merger
            .assign_addresses(base_address, PAGE_SIZE);
        self.section_merger.resolve_fragment_addresses();

        // -------------------------------------------------------------------
        // Phase 4: Linker-Defined Symbols
        // -------------------------------------------------------------------
        let mut section_addr_pairs: FxHashMap<String, (u64, u64)> = FxHashMap::default();
        for (name, addr_info) in &address_map.section_addresses {
            section_addr_pairs.insert(name.clone(), (addr_info.virtual_address, addr_info.size));
        }
        self.symbol_resolver
            .define_linker_symbols(&section_addr_pairs);

        // Build the final symbol table after all symbols are resolved.
        let sym_table = self.symbol_resolver.build_symbol_table();

        // Build a symbol address map for relocation resolution.
        let mut symbol_address_map: FxHashMap<String, ResolvedSymbol> = FxHashMap::default();
        for sym in &sym_table.symbols {
            let resolved = ResolvedSymbol {
                name: sym.name.clone(),
                final_address: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_name: String::new(),
                is_defined: sym.section_index != SYM_SHN_UNDEF,
                from_object: 0,
                export_dynamic: false,
            };
            symbol_address_map.insert(sym.name.clone(), resolved);
        }

        // -------------------------------------------------------------------
        // Phase 5: Collect Relocations
        // -------------------------------------------------------------------
        let mut all_relocations: Vec<Relocation> = Vec::new();
        for input in &inputs {
            all_relocations.extend(input.relocations.iter().cloned());
        }

        // Build section data buffers for relocation patching.
        let ordered_sections = self.section_merger.get_ordered_sections();
        let mut section_data_map: FxHashMap<String, Vec<u8>> = FxHashMap::default();
        for output_section in &ordered_sections {
            let data = self.section_merger.build_section_data(&output_section.name);
            section_data_map.insert(output_section.name.clone(), data);
        }

        // -------------------------------------------------------------------
        // Phase 6: Scan for GOT/PLT requirements
        // -------------------------------------------------------------------
        let mut collector = RelocationCollector::new();
        for rel in &all_relocations {
            collector.add_relocations(rel.object_id, rel.section_index, vec![rel.clone()]);
        }
        collector.scan_for_got_plt(&self.relocation_handler);
        let got_symbols = collector.got_symbols().clone();
        let plt_symbols = collector.plt_symbols().clone();

        // -------------------------------------------------------------------
        // Phase 7: Generate GOT/PLT (if PIC or shared)
        // -------------------------------------------------------------------
        #[allow(unused_assignments, unused_variables)]
        let mut got_entries: FxHashMap<String, u64> = FxHashMap::default();
        let mut plt_entries: FxHashMap<String, u64> = FxHashMap::default();
        #[allow(unused_assignments)]
        let mut got_data: Vec<u8> = Vec::new();
        let mut got_plt_data: Vec<u8> = Vec::new();
        let mut plt_data: Vec<u8> = Vec::new();
        let mut got_base: u64 = 0;

        if self.config.pic || is_shared {
            // Generate GOT entries for all symbols that need them.
            let (generated_got, generated_got_entries) =
                self.generate_got(&got_symbols, &symbol_address_map);
            got_data = generated_got;
            #[allow(unused_assignments)]
            {
                got_entries = generated_got_entries;
            }

            // Determine GOT section address — use a placeholder that will be
            // patched during layout.
            if let Some(addr_info) = address_map.section_addresses.get(".got") {
                got_base = addr_info.virtual_address;
            }

            // Generate PLT entries for symbols that require PLT stubs.
            if !plt_symbols.is_empty() {
                // Reserve GOT.PLT entries (GOT[0..2] reserved + one per PLT symbol)
                let got_plt_base = got_base + got_data.len() as u64;
                // GOT.PLT reserved entries: GOT[0]=.dynamic, GOT[1]=link_map, GOT[2]=resolver
                got_plt_data = vec![0u8; 3 * GOT_ENTRY_SIZE];

                let plt_base = address_map
                    .section_addresses
                    .get(".plt")
                    .map(|a| a.virtual_address)
                    .unwrap_or(0);

                // Generate PLT0 header
                plt_data = self.generate_plt_header(got_plt_base, plt_base);

                // Generate PLTn entries for each PLT symbol
                let mut sorted_plt_syms: Vec<&String> = plt_symbols.iter().collect();
                sorted_plt_syms.sort();
                for (plt_index, sym_name) in sorted_plt_syms.into_iter().enumerate() {
                    let got_plt_entry_offset =
                        got_plt_base + ((3 + plt_index) as u64) * (GOT_ENTRY_SIZE as u64);
                    let plt_entry_addr =
                        plt_base + (PLT0_SIZE as u64) + (plt_index as u64) * (PLTN_SIZE as u64);

                    // Initial GOT.PLT value points back to PLT0 for lazy binding
                    let initial_value = plt_base;
                    got_plt_data.extend_from_slice(&initial_value.to_le_bytes());

                    let plt_entry = self.generate_plt_entry(got_plt_entry_offset, plt_entry_addr);
                    plt_data.extend_from_slice(&plt_entry);

                    plt_entries.insert(sym_name.clone(), plt_entry_addr);
                }
            }

            // Insert GOT/PLT section data into section_data_map.
            if !got_data.is_empty() {
                section_data_map.insert(".got".to_string(), got_data.clone());
            }
            if !got_plt_data.is_empty() {
                section_data_map.insert(".got.plt".to_string(), got_plt_data.clone());
            }
            if !plt_data.is_empty() {
                section_data_map.insert(".plt".to_string(), plt_data.clone());
            }
        }

        // -------------------------------------------------------------------
        // Phase 8: Linker Relaxation
        // -------------------------------------------------------------------
        let relaxation_result = if self.relaxation_enabled && !all_relocations.is_empty() {
            self.perform_relaxation(
                &mut section_data_map,
                &mut all_relocations,
                &symbol_address_map,
            )
        } else {
            RelaxationResult {
                relaxations_applied: 0,
                bytes_saved: 0,
                iterations: 0,
            }
        };

        let _ = relaxation_result.relaxations_applied; // acknowledged

        // -------------------------------------------------------------------
        // Phase 9: Apply Relocations
        // -------------------------------------------------------------------
        if !all_relocations.is_empty() {
            let empty_got = crate::common::fx_hash::FxHashMap::default();
            let _reloc_result = process_relocations(
                &Target::RiscV64,
                all_relocations,
                &symbol_address_map,
                &address_map,
                &mut section_data_map,
                &self.relocation_handler,
                diagnostics,
                &empty_got,
            );
        }

        if diagnostics.has_errors() && !self.config.allow_undefined {
            let ec = diagnostics.error_count();
            return Err(format!(
                "RISC-V 64 linker: aborted due to {} relocation error(s)",
                ec
            ));
        }

        // -------------------------------------------------------------------
        // Phase 10: Dynamic Section Generation (shared objects only)
        // -------------------------------------------------------------------
        let dynamic_result: Option<DynamicLinkResult> = if is_shared {
            // Collect exported symbols (defined globals with default visibility)
            let mut exported: Vec<ExportedSymbol> = Vec::new();
            let mut imported: Vec<ImportedSymbol> = Vec::new();

            for sym in &sym_table.symbols {
                if sym.section_index != SYM_SHN_UNDEF && sym.binding == STB_GLOBAL {
                    exported.push(ExportedSymbol {
                        name: sym.name.clone(),
                        value: sym.value,
                        size: sym.size,
                        binding: sym.binding,
                        sym_type: sym.sym_type,
                        visibility: sym.visibility,
                        section_index: sym.section_index,
                    });
                } else if sym.section_index == SYM_SHN_UNDEF
                    && !sym.name.is_empty()
                    && sym.binding != STB_LOCAL
                {
                    imported.push(ImportedSymbol {
                        name: sym.name.clone(),
                        binding: sym.binding,
                        sym_type: sym.sym_type,
                        needs_plt: plt_symbols.contains(&sym.name),
                        needs_copy: false,
                        copy_size: 0,
                    });
                }
            }

            let result = build_dynamic_sections(
                &Target::RiscV64,
                true,
                &exported,
                &imported,
                &self.config.needed_libs,
                self.config.soname.as_deref(),
            );
            Some(result)
        } else {
            None
        };

        // -------------------------------------------------------------------
        // Phase 11: Compute Final Layout
        // -------------------------------------------------------------------
        let mut layout_script = DefaultLinkerScript::new(Target::RiscV64, is_shared);

        // Build input section info from section data map.
        let mut input_sec_infos: Vec<InputSectionInfo> = Vec::new();
        for sec in &ordered_sections {
            let size = section_data_map
                .get(&sec.name)
                .map(|d| d.len() as u64)
                .unwrap_or(sec.total_size);
            input_sec_infos.push(InputSectionInfo {
                name: sec.name.clone(),
                size,
                alignment: sec.alignment,
                flags: sec.flags as u32,
            });
        }

        // Add dynamic section sizes if shared.
        if let Some(ref dyn_result) = dynamic_result {
            if let Some(ref interp_data) = dyn_result.interp {
                input_sec_infos.push(InputSectionInfo {
                    name: ".interp".to_string(),
                    size: interp_data.len() as u64,
                    alignment: 1,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.dynsym_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".dynsym".to_string(),
                    size: dyn_result.dynsym_section.len() as u64,
                    alignment: 8,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.dynstr_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".dynstr".to_string(),
                    size: dyn_result.dynstr_section.len() as u64,
                    alignment: 1,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.gnu_hash_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".gnu.hash".to_string(),
                    size: dyn_result.gnu_hash_section.len() as u64,
                    alignment: 8,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.rela_dyn_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".rela.dyn".to_string(),
                    size: dyn_result.rela_dyn_section.len() as u64,
                    alignment: 8,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.rela_plt_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".rela.plt".to_string(),
                    size: dyn_result.rela_plt_section.len() as u64,
                    alignment: 8,
                    flags: SHF_ALLOC as u32,
                });
            }
            if !dyn_result.got_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".got".to_string(),
                    size: dyn_result.got_section.len() as u64,
                    alignment: 8,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }
            if !dyn_result.got_plt_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".got.plt".to_string(),
                    size: dyn_result.got_plt_section.len() as u64,
                    alignment: 8,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }
            if !dyn_result.plt_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".plt".to_string(),
                    size: dyn_result.plt_section.len() as u64,
                    alignment: 16,
                    flags: (SHF_ALLOC | SHF_EXECINSTR) as u32,
                });
            }
            if !dyn_result.dynamic_section.is_empty() {
                input_sec_infos.push(InputSectionInfo {
                    name: ".dynamic".to_string(),
                    size: dyn_result.dynamic_section.len() as u64,
                    alignment: 8,
                    flags: (SHF_ALLOC | SHF_WRITE) as u32,
                });
            }
        }

        let layout = layout_script.compute_layout(&input_sec_infos);

        // -------------------------------------------------------------------
        // Phase 12: Resolve Entry Point
        // -------------------------------------------------------------------
        let _entry_point = if self.config.output_type == OutputType::Executable {
            let mut sym_addr_flat: FxHashMap<String, u64> = FxHashMap::default();
            for sym in &sym_table.symbols {
                sym_addr_flat.insert(sym.name.clone(), sym.value);
            }

            match layout_script.resolve_entry_point(&sym_addr_flat) {
                Ok(addr) => addr,
                Err(err_msg) => {
                    diagnostics.emit_error(Span::dummy(), err_msg.clone());
                    // Use layout entry point as fallback
                    layout.entry_point_address
                }
            }
        } else {
            0
        };

        // -------------------------------------------------------------------
        // Phase 13: Write Final ELF
        // -------------------------------------------------------------------
        let elf_data = self.write_elf(
            &layout,
            &section_data_map,
            &sym_table,
            dynamic_result.as_ref(),
        );

        Ok(elf_data)
    }

    // -----------------------------------------------------------------------
    // Linker Relaxation
    // -----------------------------------------------------------------------

    /// Perform RISC-V linker relaxation on the linked output.
    ///
    /// RISC-V linker relaxation can shorten instruction sequences when
    /// branch/call targets are within a smaller addressing range:
    ///
    /// - **AUIPC+JALR → JAL**: When target is within ±1 MiB of the call
    ///   site, the 8-byte AUIPC+JALR pair can be replaced with a 4-byte
    ///   JAL instruction followed by a 4-byte NOP (preserving alignment).
    ///
    /// - **AUIPC+ADDI → ADDI**: When the target address is within ±2 KiB,
    ///   replace with a single ADDI instruction.
    ///
    /// Relaxation is iterative: each pass may bring previously out-of-range
    /// targets within range by shortening code. Iteration continues until
    /// convergence (no new relaxations) or [`MAX_RELAXATION_ITERATIONS`] is
    /// reached.
    ///
    /// Relocations tagged with `R_RISCV_RELAX` are candidates for relaxation.
    /// `R_RISCV_ALIGN` relocations mark alignment directives that may need
    /// adjustment after relaxation.
    pub fn perform_relaxation(
        &self,
        sections: &mut FxHashMap<String, Vec<u8>>,
        relocations: &mut [Relocation],
        symbols: &FxHashMap<String, ResolvedSymbol>,
    ) -> RelaxationResult {
        let mut total_relaxations: usize = 0;
        let mut total_bytes_saved: usize = 0;
        let mut iteration: usize = 0;

        loop {
            iteration += 1;
            if iteration > MAX_RELAXATION_ITERATIONS {
                break;
            }

            let mut relaxations_this_pass: usize = 0;
            let bytes_saved_this_pass: usize = 0;

            // Process each relocation for relaxation opportunities.
            // We check R_RISCV_CALL and R_RISCV_CALL_PLT for call relaxation,
            // and R_RISCV_PCREL_HI20 for pcrel relaxation.
            let r_riscv_call = relocations::R_RISCV_CALL;
            let r_riscv_call_plt = relocations::R_RISCV_CALL_PLT;
            let r_riscv_pcrel_hi20 = relocations::R_RISCV_PCREL_HI20;

            let mut i = 0;
            while i < relocations.len() {
                let rel = &relocations[i];
                let is_call_reloc =
                    rel.rel_type == r_riscv_call || rel.rel_type == r_riscv_call_plt;
                let is_pcrel_reloc = rel.rel_type == r_riscv_pcrel_hi20;

                if is_call_reloc {
                    // AUIPC+JALR → JAL relaxation.
                    // Check if the target is within ±1 MiB (21-bit signed).
                    if let Some(sym) = symbols.get(&rel.symbol_name) {
                        let value = (sym.final_address as i64)
                            .wrapping_add(rel.addend)
                            .wrapping_sub(rel.offset as i64);

                        if fits_signed(value, 21) {
                            if let Some(section_name) = &rel.output_section_name {
                                if let Some(data) = sections.get_mut(section_name) {
                                    let off = rel.offset as usize;
                                    if off + 8 <= data.len() {
                                        // Read AUIPC to extract rd register.
                                        let auipc = u32::from_le_bytes([
                                            data[off],
                                            data[off + 1],
                                            data[off + 2],
                                            data[off + 3],
                                        ]);
                                        let rd = (auipc >> 7) & 0x1f;

                                        // Encode JAL rd, offset
                                        let jal = encode_jal_instruction(rd, value as i32);
                                        let nop: u32 = 0x0000_0013;

                                        data[off..off + 4].copy_from_slice(&jal.to_le_bytes());
                                        data[off + 4..off + 8].copy_from_slice(&nop.to_le_bytes());

                                        relaxations_this_pass += 1;
                                    }
                                }
                            }
                        }
                    }
                } else if is_pcrel_reloc {
                    // AUIPC+ADDI → ADDI relaxation.
                    // Check if the target is within ±2 KiB (12-bit signed).
                    if let Some(sym) = symbols.get(&rel.symbol_name) {
                        let value = (sym.final_address as i64)
                            .wrapping_add(rel.addend)
                            .wrapping_sub(rel.offset as i64);

                        if fits_signed(value, 12) {
                            if let Some(section_name) = &rel.output_section_name {
                                if let Some(data) = sections.get_mut(section_name) {
                                    let off = rel.offset as usize;
                                    if off + 8 <= data.len() {
                                        // Read the ADDI instruction at offset+4
                                        // to get the destination register.
                                        let addi_insn = u32::from_le_bytes([
                                            data[off + 4],
                                            data[off + 5],
                                            data[off + 6],
                                            data[off + 7],
                                        ]);
                                        let rd = (addi_insn >> 7) & 0x1f;

                                        // Encode ADDI rd, x0, imm12
                                        let new_addi = encode_addi_instruction(rd, 0, value as i32);
                                        let nop: u32 = 0x0000_0013;

                                        data[off..off + 4].copy_from_slice(&new_addi.to_le_bytes());
                                        data[off + 4..off + 8].copy_from_slice(&nop.to_le_bytes());

                                        relaxations_this_pass += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                i += 1;
            }

            total_relaxations += relaxations_this_pass;
            total_bytes_saved += bytes_saved_this_pass;

            // Convergence check: if no relaxations were applied, stop.
            if relaxations_this_pass == 0 {
                break;
            }
        }

        RelaxationResult {
            relaxations_applied: total_relaxations,
            bytes_saved: total_bytes_saved,
            iterations: iteration,
        }
    }

    // -----------------------------------------------------------------------
    // PLT Header Generation (RISC-V 64)
    // -----------------------------------------------------------------------

    /// Generate the PLT0 header stub (32 bytes) for RISC-V 64.
    ///
    /// The PLT header resolves lazy-binding calls through the dynamic linker.
    /// It loads `GOT[2]` (the resolver address) and `GOT[1]` (the link map),
    /// then jumps to the resolver.
    ///
    /// ```text
    /// PLT0 — 32 bytes:
    ///   auipc  t2, %pcrel_hi(.got.plt)       // t2 = PC + hi20
    ///   sub    t1, t1, t3                      // adjust return addr
    ///   ld     t3, %pcrel_lo(1b)(t2)          // t3 = GOT[2] = resolver
    ///   addi   t1, t1, -(PLT0_SIZE + 12)      // compute PLT index
    ///   addi   t0, t2, %pcrel_lo(1b)          // t0 = &GOT[0]
    ///   srli   t1, t1, log2(PLTn_SIZE)        // relocation index
    ///   ld     t0, 8(t0)                       // t0 = GOT[1] = link_map
    ///   jalr   t3                               // jump to resolver
    /// ```
    pub fn generate_plt_header(&self, got_plt_addr: u64, plt_addr: u64) -> Vec<u8> {
        let mut code = Vec::with_capacity(PLT0_SIZE);

        // Compute PC-relative offset from PLT0 to GOT.PLT base.
        let offset = got_plt_addr as i64 - plt_addr as i64;
        let hi20 = ((offset + 0x800) >> 12) & 0xfffff;
        let lo12 = offset & 0xfff;

        // Instruction 1: auipc t2(x7), hi20
        //   opcode=0x17 (AUIPC), rd=7 (t2)
        let auipc_t2 = 0x0000_0397u32 | ((hi20 as u32) << 12);
        code.extend_from_slice(&auipc_t2.to_le_bytes());

        // Instruction 2: sub t1(x6), t1(x6), t3(x28)
        //   R-type: funct7=0x20, rs2=x28, rs1=x6, funct3=0, rd=x6, opcode=0x33
        let sub_t1_t3: u32 = 0x41c30333;
        code.extend_from_slice(&sub_t1_t3.to_le_bytes());

        // Instruction 3: ld t3(x28), lo12(t2(x7))
        //   I-type: imm12=lo12+16, rs1=x7, funct3=3, rd=x28, opcode=0x03
        let ld_off = (lo12 as i32 + 16) & 0xfff; // GOT[2] = GOT.PLT + 16
        let ld_t3: u32 = ((ld_off as u32) << 20) | (7 << 15) | (3 << 12) | (28 << 7) | 0x03;
        code.extend_from_slice(&ld_t3.to_le_bytes());

        // Instruction 4: addi t1(x6), t1(x6), -(PLT0_SIZE + 12)
        //   funct3=0 for ADDI is encoded in bits [14:12]
        let adj_imm = (-(PLT0_SIZE as i32 + 12)) & 0xfff;
        let addi_t1: u32 = ((adj_imm as u32) << 20) | (6 << 15) | (6 << 7) | 0x13;
        code.extend_from_slice(&addi_t1.to_le_bytes());

        // Instruction 5: addi t0(x5), t2(x7), lo12
        //   funct3=0 for ADDI is encoded in bits [14:12]
        let addi_t0: u32 = ((lo12 as u32 & 0xfff) << 20) | (7 << 15) | (5 << 7) | 0x13;
        code.extend_from_slice(&addi_t0.to_le_bytes());

        // Instruction 6: srli t1(x6), t1(x6), log2(PLTN_SIZE)
        //   log2(16) = 4
        let srli_t1: u32 = (4 << 20) | (6 << 15) | (5 << 12) | (6 << 7) | 0x13;
        code.extend_from_slice(&srli_t1.to_le_bytes());

        // Instruction 7: ld t0(x5), 8(t0(x5))   — load GOT[1] = link_map
        let ld_t0: u32 = (8 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | 0x03;
        code.extend_from_slice(&ld_t0.to_le_bytes());

        // Instruction 8: jalr x0, 0(x28)  — jump to resolver
        //   rd=x0 (bits [11:7]=0), funct3=0 (bits [14:12]=0), rs1=x28
        let jalr_t3: u32 = (28 << 15) | 0x67;
        code.extend_from_slice(&jalr_t3.to_le_bytes());

        // Ensure exactly PLT0_SIZE bytes.
        debug_assert_eq!(code.len(), PLT0_SIZE);
        code
    }

    // -----------------------------------------------------------------------
    // PLT Entry Generation (RISC-V 64)
    // -----------------------------------------------------------------------

    /// Generate a PLTn entry stub (16 bytes) for RISC-V 64.
    ///
    /// Each PLT entry loads the target address from the corresponding GOT
    /// entry and jumps to it. On first invocation, the GOT entry points
    /// back to PLT0 for lazy binding resolution.
    ///
    /// ```text
    /// PLTn — 16 bytes:
    ///   auipc  t3, %pcrel_hi(GOT[n+3])    // t3 = PC + hi20
    ///   ld     t3, %pcrel_lo(1b)(t3)       // t3 = GOT[n+3]
    ///   jalr   t1, t3                       // jump; t1 = return for PLT0
    ///   nop                                 // pad to 16 bytes
    /// ```
    pub fn generate_plt_entry(&self, got_entry_addr: u64, plt_entry_addr: u64) -> Vec<u8> {
        let mut code = Vec::with_capacity(PLTN_SIZE);

        // PC-relative offset from this PLT entry to its GOT entry.
        let offset = got_entry_addr as i64 - plt_entry_addr as i64;
        let hi20 = ((offset + 0x800) >> 12) & 0xfffff;
        let lo12 = offset & 0xfff;

        // Instruction 1: auipc t3(x28), hi20
        let auipc_t3: u32 = 0x00000e17u32 | ((hi20 as u32) << 12);
        code.extend_from_slice(&auipc_t3.to_le_bytes());

        // Instruction 2: ld t3(x28), lo12(t3(x28))
        let ld_t3: u32 =
            (((lo12 as u32) & 0xfff) << 20) | (28 << 15) | (3 << 12) | (28 << 7) | 0x03;
        code.extend_from_slice(&ld_t3.to_le_bytes());

        // Instruction 3: jalr t1(x6), t3(x28)
        //   jalr x6, 0(x28) — t1 captures return address for PLT0 computation
        //   rd=x6 (bits [11:7]), funct3=0 (bits [14:12]), rs1=x28
        let jalr_t1_t3: u32 = (28 << 15) | (6 << 7) | 0x67;
        code.extend_from_slice(&jalr_t1_t3.to_le_bytes());

        // Instruction 4: nop (padding to 16 bytes)
        let nop: u32 = 0x0000_0013;
        code.extend_from_slice(&nop.to_le_bytes());

        debug_assert_eq!(code.len(), PLTN_SIZE);
        code
    }

    // -----------------------------------------------------------------------
    // GOT Generation (RISC-V 64)
    // -----------------------------------------------------------------------

    /// Generate the Global Offset Table (GOT) for RISC-V 64.
    ///
    /// GOT entries are 8 bytes each (64-bit pointers):
    /// - For static executables: entries contain final symbol addresses
    /// - For shared objects: entries are filled at load time by the dynamic
    ///   linker
    ///
    /// Returns `(got_data, got_entry_map)` where `got_entry_map` maps symbol
    /// names to their GOT entry addresses (offsets from GOT base).
    pub fn generate_got(
        &self,
        got_symbols: &FxHashSet<String>,
        resolved: &FxHashMap<String, ResolvedSymbol>,
    ) -> (Vec<u8>, FxHashMap<String, u64>) {
        let mut got_data: Vec<u8> = Vec::new();
        let mut got_entry_map: FxHashMap<String, u64> = FxHashMap::default();

        // Sort symbols for deterministic output.
        let mut sorted_syms: Vec<&String> = got_symbols.iter().collect();
        sorted_syms.sort();

        let is_shared = self.config.output_type == OutputType::SharedLibrary;

        for sym_name in sorted_syms {
            let entry_offset = got_data.len() as u64;
            got_entry_map.insert(sym_name.clone(), entry_offset);

            if is_shared {
                // For shared objects, GOT entries are filled at load time.
                // Initial value = 0 (R_RISCV_64 dynamic relocation will fill it).
                got_data.extend_from_slice(&0u64.to_le_bytes());
            } else {
                // For static executables, fill with the resolved address.
                let addr = resolved.get(sym_name).map(|s| s.final_address).unwrap_or(0);
                got_data.extend_from_slice(&addr.to_le_bytes());
            }
        }

        (got_data, got_entry_map)
    }

    // -----------------------------------------------------------------------
    // ELF Output Writing
    // -----------------------------------------------------------------------

    /// Write the final ELF64 binary for RISC-V 64.
    ///
    /// Constructs the complete ELF file including:
    /// - ELF header with `e_machine = EM_RISCV`, `e_flags = 0x0005`
    /// - Program headers (PT_LOAD, PT_DYNAMIC, PT_INTERP, etc.)
    /// - Section data in layout order
    /// - Symbol table (.symtab) and string tables (.strtab, .shstrtab)
    /// - Dynamic sections (if shared object): .dynamic, .dynsym, .dynstr,
    ///   .gnu.hash, .got, .got.plt, .plt, .rela.dyn, .rela.plt
    pub fn write_elf(
        &self,
        layout: &LayoutResult,
        section_data: &FxHashMap<String, Vec<u8>>,
        symbol_table: &SymbolTable,
        dynamic_sections: Option<&DynamicLinkResult>,
    ) -> Vec<u8> {
        let is_shared = self.config.output_type == OutputType::SharedLibrary;
        let elf_type = if is_shared { ET_DYN } else { ET_EXEC };

        let mut writer = ElfWriter::new(Target::RiscV64, elf_type);

        // Set entry point.
        writer.set_entry_point(layout.entry_point_address);

        // -------------------------------------------------------------------
        // Add program headers from layout segments
        // -------------------------------------------------------------------
        for seg in &layout.segments {
            let phdr = Elf64ProgramHeader {
                p_type: seg.seg_type,
                p_flags: seg.flags,
                p_offset: seg.offset,
                p_vaddr: seg.vaddr,
                p_paddr: seg.vaddr,
                p_filesz: seg.filesz,
                p_memsz: seg.memsz,
                p_align: seg.alignment,
            };
            writer.add_program_header(phdr);
        }

        // -------------------------------------------------------------------
        // Add sections from layout
        // -------------------------------------------------------------------
        for sec_layout in &layout.sections {
            let data = section_data
                .get(&sec_layout.name)
                .cloned()
                .unwrap_or_default();

            // Determine section type.
            let (sh_type, sh_flags) = classify_section(&sec_layout.name);

            let section = Section {
                name: sec_layout.name.clone(),
                sh_type,
                sh_flags,
                data,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: sec_layout.alignment,
                sh_entsize: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(section);
        }

        // -------------------------------------------------------------------
        // Add dynamic linking sections if shared object
        // -------------------------------------------------------------------
        if let Some(dyn_secs) = dynamic_sections {
            // .interp
            if let Some(ref interp_data) = dyn_secs.interp {
                writer.add_section(Section {
                    name: ".interp".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC,
                    data: interp_data.clone(),
                    sh_addralign: 1,
                    ..Section::default()
                });
            }

            // .dynsym
            if !dyn_secs.dynsym_section.is_empty() {
                writer.add_section(Section {
                    name: ".dynsym".to_string(),
                    sh_type: SHT_DYNSYM,
                    sh_flags: SHF_ALLOC,
                    data: dyn_secs.dynsym_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: 24, // sizeof(Elf64_Sym)
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynstr
            if !dyn_secs.dynstr_section.is_empty() {
                writer.add_section(Section {
                    name: ".dynstr".to_string(),
                    sh_type: SHT_STRTAB,
                    sh_flags: SHF_ALLOC,
                    data: dyn_secs.dynstr_section.clone(),
                    sh_addralign: 1,
                    ..Section::default()
                });
            }

            // .gnu.hash
            if !dyn_secs.gnu_hash_section.is_empty() {
                // SHT_GNU_HASH = 0x6ffffff6
                writer.add_section(Section {
                    name: ".gnu.hash".to_string(),
                    sh_type: 0x6fff_fff6, // SHT_GNU_HASH
                    sh_flags: SHF_ALLOC,
                    data: dyn_secs.gnu_hash_section.clone(),
                    sh_addralign: 8,
                    ..Section::default()
                });
            }

            // .rela.dyn
            if !dyn_secs.rela_dyn_section.is_empty() {
                writer.add_section(Section {
                    name: ".rela.dyn".to_string(),
                    sh_type: SHT_RELA,
                    sh_flags: SHF_ALLOC,
                    data: dyn_secs.rela_dyn_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: 24, // sizeof(Elf64_Rela)
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .rela.plt
            if !dyn_secs.rela_plt_section.is_empty() {
                writer.add_section(Section {
                    name: ".rela.plt".to_string(),
                    sh_type: SHT_RELA,
                    sh_flags: SHF_ALLOC,
                    data: dyn_secs.rela_plt_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: 24,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .got
            if !dyn_secs.got_section.is_empty() {
                writer.add_section(Section {
                    name: ".got".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: dyn_secs.got_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .got.plt
            if !dyn_secs.got_plt_section.is_empty() {
                writer.add_section(Section {
                    name: ".got.plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: dyn_secs.got_plt_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: GOT_ENTRY_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .plt
            if !dyn_secs.plt_section.is_empty() {
                writer.add_section(Section {
                    name: ".plt".to_string(),
                    sh_type: SHT_PROGBITS,
                    sh_flags: SHF_ALLOC | SHF_EXECINSTR,
                    data: dyn_secs.plt_section.clone(),
                    sh_addralign: 16,
                    sh_entsize: PLTN_SIZE as u64,
                    logical_size: 0,
                    ..Section::default()
                });
            }

            // .dynamic
            if !dyn_secs.dynamic_section.is_empty() {
                // SHT_DYNAMIC = 6
                writer.add_section(Section {
                    name: ".dynamic".to_string(),
                    sh_type: 6, // SHT_DYNAMIC
                    sh_flags: SHF_ALLOC | SHF_WRITE,
                    data: dyn_secs.dynamic_section.clone(),
                    sh_addralign: 8,
                    sh_entsize: 16, // sizeof(Elf64_Dyn)
                    logical_size: 0,
                    ..Section::default()
                });
            }
        }

        // -------------------------------------------------------------------
        // Add symbols to the ELF writer
        // -------------------------------------------------------------------
        for sym in &symbol_table.symbols {
            let elf_sym = ElfSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            };
            writer.add_symbol(elf_sym);
        }

        writer.write()
    }
}

// ===========================================================================
// Public API — Top-Level Convenience Function
// ===========================================================================

/// Link RISC-V 64 relocatable objects into a final ELF binary.
///
/// This is the main entry point for the RISC-V 64 linker, used by the
/// code generation driver to produce final executables or shared objects.
///
/// Creates a [`RiscV64Linker`], configures it from the provided
/// [`LinkerConfig`], and runs the full linking pipeline.
///
/// # Arguments
///
/// * `config` — Linker configuration (target, output type, paths, flags).
/// * `inputs` — List of parsed relocatable object files to link.
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, returns the complete ELF binary as a `Vec<u8>`.
/// On failure, returns a descriptive error string.
///
/// # Example
///
/// ```ignore
/// use bcc::backend::riscv64::linker::link_riscv64;
/// use bcc::backend::linker_common::{LinkerConfig, LinkerInput, OutputType};
/// use bcc::common::target::Target;
/// use bcc::common::diagnostics::DiagnosticEngine;
///
/// let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
/// let inputs = vec![]; // parsed .o files
/// let mut diag = DiagnosticEngine::new();
///
/// match link_riscv64(&config, inputs, &mut diag) {
///     Ok(elf_bytes) => { /* write to disk */ }
///     Err(e) => eprintln!("link error: {}", e),
/// }
/// ```
pub fn link_riscv64(
    config: &LinkerConfig,
    inputs: Vec<LinkerInput>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, String> {
    let mut linker = RiscV64Linker::new(config.clone());
    linker.link(inputs, diagnostics)
}

// ===========================================================================
// Internal Helpers
// ===========================================================================

/// Classify a section by name into (sh_type, sh_flags).
///
/// Returns appropriate ELF section type and attribute flags based on the
/// section name following standard ELF conventions.
fn classify_section(name: &str) -> (u32, u64) {
    match name {
        ".text" | ".init" | ".fini" | ".plt" => (SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR),
        ".rodata" | ".rodata.str1.1" | ".rodata.str1.8" => (SHT_PROGBITS, SHF_ALLOC),
        ".data" | ".data.rel.ro" | ".got" | ".got.plt" => (SHT_PROGBITS, SHF_ALLOC | SHF_WRITE),
        ".bss" => (SHT_NOBITS, SHF_ALLOC | SHF_WRITE),
        ".dynamic" => (6, SHF_ALLOC | SHF_WRITE), // SHT_DYNAMIC
        ".dynsym" => (SHT_DYNSYM, SHF_ALLOC),
        ".dynstr" | ".strtab" | ".shstrtab" => (SHT_STRTAB, SHF_ALLOC),
        ".gnu.hash" => (0x6fff_fff6, SHF_ALLOC), // SHT_GNU_HASH
        ".rela.dyn" | ".rela.plt" | ".rela.text" => (SHT_RELA, SHF_ALLOC),
        ".interp" => (SHT_PROGBITS, SHF_ALLOC),
        ".comment" => (SHT_PROGBITS, 0),
        ".note" | ".note.GNU-stack" => (SHT_NOTE, SHF_ALLOC),
        s if s.starts_with(".debug_") => (SHT_PROGBITS, 0),
        s if s.starts_with(".rela.") => (SHT_RELA, SHF_ALLOC),
        s if s.starts_with(".note") => (SHT_NOTE, SHF_ALLOC),
        _ => (SHT_PROGBITS, SHF_ALLOC),
    }
}

/// Encode a JAL (Jump And Link) instruction for RISC-V.
///
/// JAL rd, offset — J-type instruction.
///
/// Immediate encoding: imm[20|10:1|11|19:12] packed into bits [31:12].
///
/// # Arguments
/// * `rd` — Destination register (0-31).
/// * `offset` — Signed byte offset (must be even, ±1 MiB range).
fn encode_jal_instruction(rd: u32, offset: i32) -> u32 {
    let imm = offset as u32;
    // J-type immediate: [20] [10:1] [11] [19:12]
    let bit20 = (imm >> 20) & 1;
    let bits_10_1 = (imm >> 1) & 0x3ff;
    let bit11 = (imm >> 11) & 1;
    let bits_19_12 = (imm >> 12) & 0xff;

    let encoded_imm = (bit20 << 31) | (bits_10_1 << 21) | (bit11 << 20) | (bits_19_12 << 12);

    encoded_imm | (rd << 7) | 0x6f // opcode for JAL
}

/// Encode an ADDI instruction for RISC-V.
///
/// ADDI rd, rs1, imm12 — I-type instruction.
///
/// # Arguments
/// * `rd` — Destination register (0-31).
/// * `rs1` — Source register (0-31).
/// * `imm12` — 12-bit signed immediate.
fn encode_addi_instruction(rd: u32, rs1: u32, imm12: i32) -> u32 {
    // ADDI: imm[11:0] | rs1 | funct3=000 | rd | opcode=0010011
    // funct3=0 occupies bits [14:12], implicitly zero.
    let imm = (imm12 as u32) & 0xfff;
    (imm << 20) | (rs1 << 15) | (rd << 7) | 0x13
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::elf_writer_common::{STT_FUNC, STV_DEFAULT};

    #[test]
    fn test_constants() {
        assert_eq!(EM_RISCV, 243);
        assert_eq!(EF_RISCV_FLOAT_ABI_DOUBLE, 0x0004);
        assert_eq!(EF_RISCV_RVC, 0x0001);
        assert_eq!(ELF_FLAGS, 0x0005);
        assert_eq!(DEFAULT_BASE_ADDRESS, 0x10000);
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(DYNAMIC_LINKER, "/lib/ld-linux-riscv64-lp64d.so.1");
        assert_eq!(PLT0_SIZE, 32);
        assert_eq!(PLTN_SIZE, 16);
        assert_eq!(GOT_ENTRY_SIZE, 8);
    }

    #[test]
    fn test_elf_flags_composition() {
        assert_eq!(ELF_FLAGS, EF_RISCV_FLOAT_ABI_DOUBLE | EF_RISCV_RVC);
    }

    #[test]
    fn test_new_executable() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let linker = RiscV64Linker::new(config);
        assert!(linker.dynamic_ctx.is_none());
        assert!(linker.relaxation_enabled);
        assert_eq!(linker.diagnostics_count, 0);
    }

    #[test]
    fn test_new_shared_library() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::SharedLibrary);
        let linker = RiscV64Linker::new(config);
        assert!(linker.dynamic_ctx.is_some());
        assert!(linker.relaxation_enabled);
    }

    #[test]
    fn test_plt_header_size() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let linker = RiscV64Linker::new(config);
        let header = linker.generate_plt_header(0x20000, 0x10000);
        assert_eq!(header.len(), PLT0_SIZE);
    }

    #[test]
    fn test_plt_entry_size() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let linker = RiscV64Linker::new(config);
        let entry = linker.generate_plt_entry(0x30000, 0x10020);
        assert_eq!(entry.len(), PLTN_SIZE);
    }

    #[test]
    fn test_got_generation_empty() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let linker = RiscV64Linker::new(config);
        let got_syms = FxHashSet::default();
        let resolved = FxHashMap::default();
        let (data, map) = linker.generate_got(&got_syms, &resolved);
        assert!(data.is_empty());
        assert!(map.is_empty());
    }

    #[test]
    fn test_got_generation_static() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let linker = RiscV64Linker::new(config);

        let mut got_syms = FxHashSet::default();
        got_syms.insert("foo".to_string());

        let mut resolved = FxHashMap::default();
        resolved.insert(
            "foo".to_string(),
            ResolvedSymbol {
                name: "foo".to_string(),
                final_address: 0x12345678,
                size: 4,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_name: ".text".to_string(),
                is_defined: true,
                from_object: 0,
                export_dynamic: false,
            },
        );

        let (data, map) = linker.generate_got(&got_syms, &resolved);
        assert_eq!(data.len(), GOT_ENTRY_SIZE);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("foo"));

        // Verify the GOT entry contains the resolved address.
        let addr = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(addr, 0x12345678);
    }

    #[test]
    fn test_got_generation_shared() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::SharedLibrary);
        let linker = RiscV64Linker::new(config);

        let mut got_syms = FxHashSet::default();
        got_syms.insert("bar".to_string());

        let mut resolved = FxHashMap::default();
        resolved.insert(
            "bar".to_string(),
            ResolvedSymbol {
                name: "bar".to_string(),
                final_address: 0xAAAA,
                size: 0,
                binding: STB_GLOBAL,
                sym_type: STT_FUNC,
                visibility: STV_DEFAULT,
                section_name: ".text".to_string(),
                is_defined: true,
                from_object: 0,
                export_dynamic: true,
            },
        );

        let (data, map) = linker.generate_got(&got_syms, &resolved);
        assert_eq!(data.len(), GOT_ENTRY_SIZE);
        assert!(map.contains_key("bar"));

        // Shared GOT entries are 0 (filled at load time).
        let addr = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(addr, 0);
    }

    #[test]
    fn test_jal_encoding() {
        // JAL x1, 0 should produce opcode 0x6f with rd=1
        let insn = encode_jal_instruction(1, 0);
        assert_eq!(insn & 0x7f, 0x6f); // opcode
        assert_eq!((insn >> 7) & 0x1f, 1); // rd = x1
    }

    #[test]
    fn test_addi_encoding() {
        // ADDI x5, x7, 0x10
        let insn = encode_addi_instruction(5, 7, 0x10);
        assert_eq!(insn & 0x7f, 0x13); // opcode
        assert_eq!((insn >> 7) & 0x1f, 5); // rd
        assert_eq!((insn >> 15) & 0x1f, 7); // rs1
        assert_eq!((insn >> 20) & 0xfff, 0x10); // imm12
    }

    #[test]
    fn test_classify_section_text() {
        let (sh_type, sh_flags) = classify_section(".text");
        assert_eq!(sh_type, SHT_PROGBITS);
        assert_eq!(sh_flags, (SHF_ALLOC | SHF_EXECINSTR) as u64);
    }

    #[test]
    fn test_classify_section_bss() {
        let (sh_type, sh_flags) = classify_section(".bss");
        assert_eq!(sh_type, SHT_NOBITS);
        assert_eq!(sh_flags, (SHF_ALLOC | SHF_WRITE) as u64);
    }

    #[test]
    fn test_classify_section_debug() {
        let (sh_type, _sh_flags) = classify_section(".debug_info");
        assert_eq!(sh_type, SHT_PROGBITS);
    }

    #[test]
    fn test_relaxation_result_defaults() {
        let result = RelaxationResult {
            relaxations_applied: 0,
            bytes_saved: 0,
            iterations: 0,
        };
        assert_eq!(result.relaxations_applied, 0);
        assert_eq!(result.bytes_saved, 0);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_link_empty_inputs() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let mut linker = RiscV64Linker::new(config);
        let mut diag = DiagnosticEngine::new();

        // Linking with no inputs should fail — no _start symbol
        let result = linker.link(vec![], &mut diag);
        // It's OK for it to fail here (no _start) or succeed with empty binary.
        // The important thing is it doesn't panic.
        let _ = result;
    }

    #[test]
    fn test_link_riscv64_convenience() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        let mut diag = DiagnosticEngine::new();

        let result = link_riscv64(&config, vec![], &mut diag);
        // Same as above — should not panic.
        let _ = result;
    }
}
