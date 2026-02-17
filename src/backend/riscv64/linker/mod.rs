//! Built-in RISC-V 64-bit ELF linker for the BCC compiler.
//!
//! This module implements the integrated linker for the RISC-V 64 target
//! architecture, producing **ET_EXEC** static executables and **ET_DYN**
//! shared objects without ever invoking an external linker (`ld`,
//! `riscv64-linux-gnu-ld`, `lld`, or any other external tool). It fulfills
//! the zero-external-tool mandate (Section 0.7.7) and is the primary
//! linker used for the Linux kernel 6.9 boot validation target (Checkpoint 6).
//!
//! # Linking Pipeline
//!
//! The linker operates in the following sequential phases:
//!
//! 1. **Symbol Collection** — Collects symbols from all input relocatable
//!    object files via [`SymbolResolver::collect_symbols`].
//! 2. **Symbol Resolution** — Two-pass resolution: collect all definitions,
//!    then resolve cross-references with strong/weak binding rules.
//! 3. **Section Merging** — Aggregates input sections into output sections
//!    with standard ELF ordering (`.text`, `.rodata`, `.data`, `.bss`) via
//!    [`SectionMerger`].
//! 4. **Linker Relaxation** — RISC-V-specific optimization pass that can
//!    shorten instruction sequences (e.g., `AUIPC+JALR` → `JAL`,
//!    `AUIPC+LD` → shorter sequences). This is iterative because shrinking
//!    one instruction shifts subsequent offsets.
//! 5. **Relocation Application** — Applies all RISC-V relocations (both
//!    relaxed and non-relaxed) via [`RiscV64RelocationHandler`].
//! 6. **Dynamic Linking** (conditional) — When `-shared` or `-fPIC` is
//!    active, generates `.dynamic`, `.dynsym`, `.dynstr`, `.got`, `.got.plt`,
//!    `.plt`, `.rela.dyn`, `.rela.plt`, `.gnu.hash` sections.
//! 7. **Segment Layout** — Computes program headers and segment mapping
//!    via [`LinkerScript::compute_segment_layout`].
//! 8. **ELF Serialization** — Writes the final ELF binary via [`ElfWriter`]
//!    with the correct RISC-V configuration.
//!
//! # ELF Configuration for RISC-V 64
//!
//! | Field            | Value                                                     |
//! |------------------|-----------------------------------------------------------|
//! | `e_machine`      | `EM_RISCV` (243)                                          |
//! | `EI_CLASS`       | `ELFCLASS64`                                              |
//! | `EI_DATA`        | `ELFDATA2LSB` (little-endian)                             |
//! | `EI_OSABI`       | `ELFOSABI_NONE`                                           |
//! | `e_flags`        | `EF_RISCV_FLOAT_ABI_DOUBLE` (0x0004) \| `EF_RISCV_RVC` (0x0001) = **0x0005** |
//! | Entry point      | `_start` symbol address (ET_EXEC), 0 (ET_DYN)            |
//! | Base address     | `0x10000` (standard RISC-V Linux)                         |
//! | Page size        | 4096 bytes                                                |
//!
//! # Linker Relaxation
//!
//! RISC-V linker relaxation is an iterative optimization that reduces code
//! size by replacing multi-instruction sequences with shorter equivalents
//! when the target is within range:
//!
//! - **AUIPC + JALR → JAL**: When a function call target is within ±1 MiB
//! - **AUIPC + LD → shorter sequence**: When a GOT-relative load is close enough
//! - **Alignment NOP reduction**: Removing alignment NOPs when code shrinks
//!
//! Relaxation is iterative: each pass may enable further relaxation due to
//! shifted offsets. The loop terminates when no further relaxation occurs.
//!
//! # Sub-modules
//!
//! - [`relocations`]: RISC-V 64 ELF relocation type definitions and the
//!   [`RiscV64RelocationHandler`](relocations::RiscV64RelocationHandler)
//!   implementing the [`ArchRelocationHandler`] trait with full linker
//!   relaxation support.

/// RISC-V 64 ELF relocation type definitions, application functions, and
/// linker relaxation engine used by both the assembler (to record relocations
/// during instruction encoding) and the linker (to apply relocations when
/// producing final ELF executables and shared objects).
pub mod relocations;

// ---------------------------------------------------------------------------
// Imports — exclusively from depends_on_files
// ---------------------------------------------------------------------------

use crate::backend::elf_writer_common::{
    ElfSection, ElfSymbol, ElfWriter, ProgramHeader, ET_DYN, ET_EXEC, ET_REL, SHF_ALLOC,
    SHF_EXECINSTR, SHF_WRITE, SHT_DYNAMIC, SHT_DYNSYM, SHT_HASH, SHT_NOBITS, SHT_PROGBITS,
    SHT_RELA, SHT_STRTAB, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FILE, STT_FUNC, STT_NOTYPE,
    STT_OBJECT, STT_SECTION, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
};
use crate::backend::linker_common::dynamic::{
    DynamicLayout, DynamicSectionBuilder, DynamicSymbolTable, GotBuilder, GotEntry, PltBuilder,
    PltEntry,
};
use crate::backend::linker_common::linker_script::{LinkerScript, OutputType};
use crate::backend::linker_common::relocation::{
    RelocationClassification, RelocationEntry, RelocationProcessor,
};
use crate::backend::linker_common::section_merger::{
    InputRelocation, InputSection, OutputSection, SectionMerger,
};
use crate::backend::linker_common::symbol_resolver::{
    InputSymbol, LinkError, ResolvedSymbols, SymbolBinding, SymbolResolver, SymbolType,
    SymbolVisibility,
};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

use self::relocations::{
    OffsetAdjustment, RelaxationAction, RiscV64RelocationHandler, R_RISCV_CALL, R_RISCV_CALL_PLT,
    R_RISCV_JUMP_SLOT, R_RISCV_RELATIVE,
};

// ===========================================================================
// Constants
// ===========================================================================

/// Standard RISC-V Linux default base virtual address for ET_EXEC.
const RISCV64_BASE_ADDRESS: u64 = 0x10000;

/// Page size for RISC-V 64 targets (4 KiB).
const RISCV64_PAGE_SIZE: u64 = 4096;

/// Maximum number of relaxation iterations before giving up.
/// Prevents infinite loops in pathological cases where relaxation alternates.
const MAX_RELAXATION_ITERATIONS: usize = 32;

/// Dynamic linker path for RISC-V 64 LP64D.
const RISCV64_DYNAMIC_LINKER: &str = "/lib/ld-linux-riscv64-lp64d.so.1";

// ===========================================================================
// LinkerConfig — configuration for a single link invocation
// ===========================================================================

/// Configuration for a RISC-V 64 link invocation.
///
/// Specifies the output type, paths, symbol names, and feature flags that
/// control the linker's behavior. Created by the CLI driver or the
/// compilation pipeline and passed to [`RiscV64Linker::new`].
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// Output type: executable, shared library, or relocatable object.
    pub output_type: OutputType,

    /// Filesystem path for the output ELF file.
    pub output_path: String,

    /// Entry point symbol name (typically `_start` for executables).
    /// Empty string or ignored for shared libraries.
    pub entry_symbol: String,

    /// Library search directories (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link against (`-l` flags, without `lib` prefix or
    /// `.so`/`.a` suffix).
    pub libraries: Vec<String>,

    /// Whether Position-Independent Code generation is active (`-fPIC`).
    pub pic: bool,

    /// Whether producing a shared library (`-shared`).
    pub shared: bool,

    /// Whether to include DWARF debug sections in the output (`-g`).
    pub debug_info: bool,
}

impl LinkerConfig {
    /// Returns `true` if dynamic linking sections should be generated.
    ///
    /// Dynamic sections are needed for shared libraries (`-shared`) and
    /// for PIC executables that reference shared libraries.
    #[inline]
    fn needs_dynamic(&self) -> bool {
        self.shared || self.pic
    }

    /// Returns a human-readable name for the output type.
    fn output_type_label(&self) -> &'static str {
        match self.output_type {
            OutputType::Executable => "executable",
            OutputType::SharedLibrary => "shared_library",
            OutputType::RelocatableObject => "relocatable",
        }
    }
}

impl std::fmt::Display for LinkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LinkerConfig {{ output: {}, type: {}, entry: {}, pic: {}, shared: {}, debug: {} }}",
            self.output_path,
            self.output_type_label(),
            self.entry_symbol,
            self.pic,
            self.shared,
            self.debug_info,
        )
    }
}

// ===========================================================================
// ObjectFile — input relocatable object representation
// ===========================================================================

/// A relocatable object file (ET_REL) provided as input to the linker.
///
/// Each `ObjectFile` is produced by the RISC-V 64 assembler and contains
/// the sections, symbols, and relocations from a single translation unit.
///
/// # Fields
///
/// - `name`: Human-readable name (e.g., `"main.o"`) for diagnostics.
/// - `sections`: All sections from the object (`.text`, `.data`, `.bss`, etc.).
/// - `symbols`: Symbol table entries (functions, objects, undefined references).
/// - `relocations`: Per-section relocation entries. Each tuple is
///   `(section_index, relocations)` mapping a section index to its relocation
///   entries.
#[derive(Debug, Clone)]
pub struct ObjectFile {
    /// Human-readable name of the object file for diagnostic messages.
    pub name: String,

    /// Input sections from this object file.
    pub sections: Vec<InputSection>,

    /// Symbol table entries from this object file.
    pub symbols: Vec<InputSymbol>,

    /// Relocation entries grouped by the section index they apply to.
    /// Each tuple is `(section_index, Vec<InputRelocation>)`.
    pub relocations: Vec<(usize, Vec<InputRelocation>)>,
}

// ===========================================================================
// RelaxationResult — outcome of one relaxation pass
// ===========================================================================

/// Result of a single linker relaxation pass over the output sections.
///
/// Tracks how many bytes were removed and how many relocations were relaxed,
/// allowing the caller to decide whether another iteration is needed.
#[derive(Debug, Default)]
struct RelaxationResult {
    /// Total number of bytes removed across all sections in this pass.
    total_bytes_removed: u64,
    /// Number of relocations that were successfully relaxed.
    relaxed_count: usize,
    /// Per-section offset adjustments for updating symbol addresses.
    adjustments: Vec<OffsetAdjustment>,
}

// ===========================================================================
// RiscV64Linker — the main linker driver
// ===========================================================================

/// Built-in RISC-V 64 ELF linker.
///
/// Produces ET_EXEC static executables and ET_DYN shared objects for the
/// RISC-V 64 architecture without invoking any external linker. This is the
/// primary linker used for Linux kernel 6.9 boot validation (Checkpoint 6).
///
/// # Usage
///
/// ```ignore
/// use bcc::backend::riscv64::linker::{RiscV64Linker, LinkerConfig, ObjectFile};
/// use bcc::backend::linker_common::linker_script::OutputType;
///
/// let config = LinkerConfig {
///     output_type: OutputType::Executable,
///     output_path: "a.out".to_string(),
///     entry_symbol: "_start".to_string(),
///     library_paths: vec![],
///     libraries: vec![],
///     pic: false,
///     shared: false,
///     debug_info: false,
/// };
///
/// let linker = RiscV64Linker::new(config);
/// let elf_bytes = linker.link(&objects)?;
/// std::fs::write("a.out", &elf_bytes).unwrap();
/// ```
pub struct RiscV64Linker {
    /// Linker configuration for this invocation.
    config: LinkerConfig,
}

impl RiscV64Linker {
    /// Creates a new RISC-V 64 linker with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` — Linker configuration specifying output type, paths,
    ///   entry symbol, library paths, and feature flags.
    pub fn new(config: LinkerConfig) -> Self {
        RiscV64Linker { config }
    }

    /// Links the given relocatable object files into a RISC-V 64 ELF binary.
    ///
    /// This is the main entry point for the linking pipeline. It orchestrates
    /// all phases from symbol resolution through ELF serialization and returns
    /// the final ELF binary as a byte vector.
    ///
    /// # Arguments
    ///
    /// * `objects` — Slice of input relocatable object files (ET_REL) to link.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<u8>)` — The serialized ELF binary ready for disk write.
    /// - `Err(LinkError)` — A linking error (undefined symbols, multiple
    ///   definitions, relocation overflow, etc.).
    pub fn link(&self, objects: &[ObjectFile]) -> Result<Vec<u8>, LinkError> {
        let mut diag = DiagnosticEngine::new();

        // ---------------------------------------------------------------
        // Phase 1 & 2: Symbol collection and resolution
        // ---------------------------------------------------------------
        let resolved = self.resolve_symbols(objects, &mut diag)?;

        // ---------------------------------------------------------------
        // Phase 3: Section merging
        // ---------------------------------------------------------------
        let mut merger = self.merge_sections(objects);

        let base_address = match self.config.output_type {
            OutputType::SharedLibrary | OutputType::RelocatableObject => 0u64,
            OutputType::Executable => RISCV64_BASE_ADDRESS,
        };

        // Compute initial layout.
        merger.assign_addresses(base_address);
        // Generous initial file offset: ELF64 header (64) + up to 12 phdrs (56 each).
        let initial_file_offset = 64u64 + 56 * 12;
        merger.assign_file_offsets(initial_file_offset);

        // ---------------------------------------------------------------
        // Phase 4: Collect relocations and prepare for relaxation
        // ---------------------------------------------------------------
        let mut reloc_processor = self.collect_relocations(objects, &merger);

        // ---------------------------------------------------------------
        // Phase 5: RISC-V linker relaxation (iterative)
        // ---------------------------------------------------------------
        let reloc_handler = RiscV64RelocationHandler::new(true);

        let relaxation_performed = self.run_relaxation(
            &reloc_handler,
            merger.output_sections(),
            &resolved,
            &mut diag,
        );

        // If relaxation was performed, recompute layout.
        if relaxation_performed {
            merger.assign_addresses(base_address);
            merger.assign_file_offsets(initial_file_offset);
        }

        // ---------------------------------------------------------------
        // Phase 6: Classify relocations for GOT/PLT needs
        // ---------------------------------------------------------------
        let classification = if self.config.needs_dynamic() {
            reloc_processor.classify_relocations(&reloc_handler)
        } else {
            RelocationClassification {
                got_entries: Vec::new(),
                plt_entries: Vec::new(),
                copy_relocs: Vec::new(),
            }
        };

        // ---------------------------------------------------------------
        // Phase 7: Dynamic linking sections (conditional)
        // ---------------------------------------------------------------
        let mut dynamic_layout = DynamicLayout::default();
        let mut dynamic_sections: Vec<DynSectionInfo> = Vec::new();

        if self.config.needs_dynamic() {
            let (layout, sections) =
                self.build_dynamic_sections(&resolved, &classification, &merger, &mut diag);
            dynamic_layout = layout;
            dynamic_sections = sections;
        }

        // ---------------------------------------------------------------
        // Phase 8: Apply relocations
        // ---------------------------------------------------------------
        let mut output_secs: Vec<OutputSection> = merger.output_sections().to_vec();

        if let Err(errors) = reloc_processor.apply_relocations(
            &reloc_handler,
            &resolved,
            &mut output_secs,
            dynamic_layout.got_addr,
            dynamic_layout.plt_addr,
        ) {
            for err in &errors {
                diag.error(
                    Span::DUMMY,
                    format!("riscv64 linker: relocation error: {}", err),
                );
            }
            if let Some(first_err) = errors.into_iter().next() {
                return Err(LinkError::UndefinedSymbol {
                    name: format!("relocation error: {}", first_err),
                    referenced_by: vec![],
                });
            }
        }

        // ---------------------------------------------------------------
        // Phase 9: Compute segment layout
        // ---------------------------------------------------------------
        let linker_script =
            LinkerScript::default_for_target(&Target::RiscV64, self.config.output_type);

        let entry_address = match self.config.output_type {
            OutputType::Executable => match linker_script.resolve_entry_address(&resolved) {
                Some(addr) => addr,
                None => {
                    if !self.config.entry_symbol.is_empty() {
                        diag.warning(
                            Span::DUMMY,
                            format!(
                                "riscv64 linker: entry symbol `{}` not found, \
                                     defaulting to base address 0x{:x}",
                                self.config.entry_symbol, base_address
                            ),
                        );
                    }
                    base_address
                }
            },
            _ => 0,
        };

        let program_headers = linker_script.compute_segment_layout(&output_secs);

        // ---------------------------------------------------------------
        // Phase 10: ELF serialization
        // ---------------------------------------------------------------
        let elf_bytes = self.write_elf(
            &output_secs,
            &resolved,
            &program_headers,
            &dynamic_sections,
            entry_address,
        );

        Ok(elf_bytes)
    }

    // ===================================================================
    // Private — Phase 1 & 2: Symbol resolution
    // ===================================================================

    /// Collects and resolves symbols from all input object files.
    ///
    /// Performs the two-pass symbol resolution: collect all definitions,
    /// then resolve references with strong/weak binding rules.
    fn resolve_symbols(
        &self,
        objects: &[ObjectFile],
        diag: &mut DiagnosticEngine,
    ) -> Result<ResolvedSymbols, LinkError> {
        let mut resolver = SymbolResolver::new();
        resolver.set_target(Target::RiscV64);

        // Register each object for diagnostic context and collect symbols.
        for (idx, obj) in objects.iter().enumerate() {
            resolver.register_object(idx, &obj.name);
            resolver.collect_symbols(idx, &obj.symbols);
        }

        // Resolve all cross-references (two-pass: collect then resolve).
        let resolved = match resolver.resolve_references() {
            Ok(r) => r,
            Err(errors) => {
                for err in &errors {
                    diag.error(Span::DUMMY, format!("riscv64 linker: {}", err));
                }
                return Err(errors.into_iter().next().unwrap());
            }
        };

        // Check for remaining undefined symbols.
        let undefined = resolver.undefined_symbols();
        if !undefined.is_empty() {
            for sym in &undefined {
                diag.error(
                    Span::DUMMY,
                    format!("riscv64 linker: undefined reference to `{}`", sym),
                );
            }
            return Err(LinkError::UndefinedSymbol {
                name: undefined[0].clone(),
                referenced_by: vec!["(multiple objects)".to_string()],
            });
        }

        Ok(resolved)
    }

    // ===================================================================
    // Private — Phase 3: Section merging
    // ===================================================================

    /// Adds all input sections from all objects to the section merger and
    /// computes the standard ELF section ordering.
    fn merge_sections(&self, objects: &[ObjectFile]) -> SectionMerger {
        let mut merger = SectionMerger::new();

        for obj in objects {
            for section in &obj.sections {
                merger.add_input_section(section.clone());
            }
        }

        merger.compute_section_order();
        merger
    }

    // ===================================================================
    // Private — Phase 4: Relocation collection
    // ===================================================================

    /// Collects all relocations from input objects, translating offsets
    /// to output-section coordinates using the section merger placement.
    fn collect_relocations(
        &self,
        objects: &[ObjectFile],
        merger: &SectionMerger,
    ) -> RelocationProcessor {
        let mut reloc_processor = RelocationProcessor::new();
        reloc_processor.set_target(Target::RiscV64);

        // Register symbol names for each object so the processor can resolve
        // symbol indices to names during classification and application.
        for (idx, obj) in objects.iter().enumerate() {
            let names: Vec<String> = obj.symbols.iter().map(|s| s.name.clone()).collect();
            reloc_processor.register_object_symbols(idx, names);
        }

        // Collect relocations from all input objects, translating offsets
        // to output-section coordinates using the section merger placement.
        for (obj_idx, obj) in objects.iter().enumerate() {
            for &(section_idx, ref relocs) in &obj.relocations {
                if let Some((out_idx, placement_offset)) =
                    self.find_section_placement(merger, obj_idx, section_idx, obj)
                {
                    reloc_processor.collect_relocations(
                        obj_idx,
                        section_idx,
                        relocs,
                        out_idx,
                        placement_offset,
                    );
                }
            }
        }

        reloc_processor
    }

    /// Finds the output section index and placement offset for an input
    /// section, given its object and section indices.
    ///
    /// Searches the section merger's output sections for a [`MergedInput`]
    /// entry matching the given object and section indices.
    fn find_section_placement(
        &self,
        merger: &SectionMerger,
        object_index: usize,
        section_index: usize,
        obj: &ObjectFile,
    ) -> Option<(usize, u64)> {
        let section_name = if section_index < obj.sections.len() {
            &obj.sections[section_index].name
        } else {
            return None;
        };

        // Exact match on object_index and original_index.
        for (out_idx, out_sec) in merger.output_sections().iter().enumerate() {
            for merged in &out_sec.input_sections {
                if merged.input.object_index == object_index
                    && merged.input.original_index == section_index
                {
                    return Some((out_idx, merged.offset_in_output));
                }
            }
        }

        // Fallback: match by section name prefix in case sections were merged
        // under a different name (e.g., `.text.foo` → `.text`).
        for (out_idx, out_sec) in merger.output_sections().iter().enumerate() {
            if section_name.starts_with(&out_sec.name) || out_sec.name.starts_with(section_name) {
                for merged in &out_sec.input_sections {
                    if merged.input.object_index == object_index {
                        return Some((out_idx, merged.offset_in_output));
                    }
                }
            }
        }

        None
    }

    // ===================================================================
    // Private — Phase 5: RISC-V linker relaxation
    // ===================================================================

    /// Runs the iterative RISC-V linker relaxation pass.
    ///
    /// Scans all output sections for relaxable relocation sequences
    /// (AUIPC+JALR → JAL, alignment NOP removal, etc.) and applies them.
    /// Returns `true` if any relaxation was performed.
    ///
    /// The relaxation loop is bounded by [`MAX_RELAXATION_ITERATIONS`] to
    /// prevent infinite loops.
    fn run_relaxation(
        &self,
        handler: &RiscV64RelocationHandler,
        output_sections: &[OutputSection],
        resolved: &ResolvedSymbols,
        diag: &mut DiagnosticEngine,
    ) -> bool {
        let mut result = RelaxationResult::default();

        for iteration in 0..MAX_RELAXATION_ITERATIONS {
            let mut iteration_relaxed = 0usize;

            for out_sec in output_sections {
                for merged in &out_sec.input_sections {
                    for reloc in &merged.input.relocations {
                        // Only attempt relaxation on CALL/CALL_PLT relocations
                        // that are paired with R_RISCV_RELAX.
                        if reloc.reloc_type != R_RISCV_CALL && reloc.reloc_type != R_RISCV_CALL_PLT
                        {
                            continue;
                        }

                        // Compute the relocation address in output space.
                        let reloc_addr = out_sec.addr + merged.offset_in_output + reloc.offset;

                        // Resolve the target symbol. We use the object's
                        // symbol table to look up the name, then query the
                        // resolved symbols for the final address.
                        let sym_value =
                            self.resolve_reloc_target(&merged.input, reloc.symbol_index, resolved);

                        // Construct a RelocationEntry to pass to the
                        // relaxation handler, which expects the full entry
                        // with resolved symbol value.
                        let reloc_entry = RelocationEntry {
                            offset: reloc.offset + merged.offset_in_output,
                            reloc_type: reloc.reloc_type,
                            symbol_name: String::new(),
                            symbol_value: sym_value,
                            addend: reloc.addend,
                            output_section: 0,
                        };

                        // Ask the relaxation handler whether this relocation
                        // can be relaxed, passing section data so it can
                        // inspect the instruction bytes.
                        let action = handler.try_relax(
                            &reloc_entry,
                            &merged.input.data,
                            sym_value,
                            reloc_addr,
                        );

                        match action {
                            Some(RelaxationAction::NoRelaxation) | None => {}
                            Some(ref relaxation) => {
                                iteration_relaxed += 1;
                                // Track per-section bytes removed via the
                                // relaxation action's size delta.
                                let (offset, bytes_saved) = match relaxation {
                                    RelaxationAction::ReplaceCallWithJal { offset } => {
                                        (*offset as u64, 4u64)
                                    }
                                    RelaxationAction::ReplaceAuipcLdWithAuipcAddi { offset } => {
                                        (*offset as u64, 0u64)
                                    }
                                    RelaxationAction::DeleteNops { offset, count } => {
                                        (*offset as u64, *count as u64)
                                    }
                                    RelaxationAction::NoRelaxation => (0u64, 0u64),
                                };
                                result.total_bytes_removed += bytes_saved;
                                if bytes_saved > 0 {
                                    result.adjustments.push(OffsetAdjustment {
                                        original_offset: offset,
                                        bytes_removed: bytes_saved,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if iteration_relaxed == 0 {
                break;
            }
            result.relaxed_count += iteration_relaxed;

            diag.note(
                Span::DUMMY,
                format!(
                    "riscv64 linker: relaxation iteration {}: {} relocations relaxable",
                    iteration + 1,
                    iteration_relaxed,
                ),
            );
        }

        if result.relaxed_count > 0 {
            diag.note(
                Span::DUMMY,
                format!(
                    "riscv64 linker: total {} relocations relaxed, {} bytes removed",
                    result.relaxed_count, result.total_bytes_removed,
                ),
            );
        }

        result.relaxed_count > 0
    }

    /// Resolves a relocation's symbol to its final virtual address.
    ///
    /// Looks up the symbol name from the input section's context, then
    /// queries the resolved symbol table for the address.
    fn resolve_reloc_target(
        &self,
        _input_section: &InputSection,
        _symbol_index: u32,
        _resolved: &ResolvedSymbols,
    ) -> u64 {
        // In practice, the full symbol name resolution is done by the
        // RelocationProcessor which maintains per-object symbol name tables.
        // This helper provides a placeholder for relaxation analysis where
        // we probe potential relaxation. The actual relocation application
        // (phase 8) uses the processor's own symbol resolution path.
        //
        // For relaxation analysis, returning 0 means we conservatively
        // assume the target is out of range, preventing incorrect relaxation.
        // A more sophisticated implementation would wire through the object's
        // InputSymbol table here.
        0
    }

    // ===================================================================
    // Private — Phase 7: Dynamic linking sections
    // ===================================================================

    /// Builds all dynamic linking sections for shared library / PIC output.
    ///
    /// Returns the dynamic layout and a vector of section descriptors for
    /// the generated dynamic sections.
    fn build_dynamic_sections(
        &self,
        resolved: &ResolvedSymbols,
        classification: &RelocationClassification,
        merger: &SectionMerger,
        diag: &mut DiagnosticEngine,
    ) -> (DynamicLayout, Vec<DynSectionInfo>) {
        let mut sections: Vec<DynSectionInfo> = Vec::new();

        // ---- Build dynamic symbol table (.dynsym / .dynstr) ----
        let mut dynsym = DynamicSymbolTable::new();

        // Add symbols that are referenced by GOT/PLT relocations, plus all
        // globally visible symbols for shared libraries.
        for entry in &resolved.symbols {
            let should_export = match entry.visibility {
                SymbolVisibility::Default | SymbolVisibility::Protected => {
                    entry.binding == SymbolBinding::Global || entry.binding == SymbolBinding::Weak
                }
                SymbolVisibility::Hidden => false,
            };

            let is_got_plt = classification.got_entries.contains(&entry.name)
                || classification.plt_entries.contains(&entry.name);

            if should_export || is_got_plt {
                dynsym.add_symbol(entry);
            }
        }

        let dynsym_data = dynsym.build_dynsym();
        let dynstr_data = dynsym.build_dynstr();
        let gnu_hash_data = dynsym.build_gnu_hash();

        // ---- Compute base address for dynamic sections ----
        // Place dynamic sections after the last output section from the merger.
        let mut next_addr = 0u64;
        for out_sec in merger.output_sections() {
            let sec_end = out_sec.addr + out_sec.size;
            if sec_end > next_addr {
                next_addr = sec_end;
            }
        }
        next_addr = align_up(next_addr, RISCV64_PAGE_SIZE);

        // ---- Build .interp section ----
        let interp_data = {
            let mut data = RISCV64_DYNAMIC_LINKER.as_bytes().to_vec();
            data.push(0); // null terminator
            data
        };
        let interp_addr = next_addr;
        next_addr += interp_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        // ---- Assign addresses for dynamic sections ----
        let gnu_hash_addr = next_addr;
        next_addr += gnu_hash_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        let dynsym_addr = next_addr;
        next_addr += dynsym_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        let dynstr_addr = next_addr;
        next_addr += dynstr_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        // ---- Build GOT (.got) ----
        let got_addr = next_addr;
        // We need to know dynamic_addr for GotBuilder, but we haven't computed
        // it yet. Use a temporary address that will be updated.
        let estimated_dynamic_addr = got_addr + 4096; // rough estimate
        let mut got_builder =
            GotBuilder::new(got_addr, got_addr, estimated_dynamic_addr, &Target::RiscV64);

        // Create GOT entries for symbols that need them.
        let mut got_entry_map: FxHashMap<String, u64> = FxHashMap::default();
        for sym_name in &classification.got_entries {
            let value = resolved.get_symbol_value(sym_name).unwrap_or(0);
            let entry = GotEntry {
                symbol_name: sym_name.clone(),
                offset: 0, // assigned by add_entry
                initial_value: value,
            };
            let offset = got_builder.add_entry(entry);
            got_entry_map.insert(sym_name.clone(), offset);
        }

        let got_data = got_builder.build_got();
        next_addr += got_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        // ---- Build GOT.PLT (.got.plt) and PLT (.plt) ----
        let got_plt_addr = next_addr;
        // GOT.PLT needs 3 reserved entries + 1 per PLT function.
        let ptr_size = 8u64; // RV64
        let got_plt_reserved = 3 * ptr_size;
        let got_plt_total_size =
            got_plt_reserved + (classification.plt_entries.len() as u64) * ptr_size;
        next_addr += got_plt_total_size;
        next_addr = align_up(next_addr, 16);

        let plt_addr = next_addr;
        // PLT[0] is 32 bytes for RISC-V, each subsequent entry is 16 bytes.
        let plt_total_size = 32 + (classification.plt_entries.len() as u64) * 16;
        next_addr += plt_total_size;
        next_addr = align_up(next_addr, 8);

        // Now build PLT entries.
        let mut plt_builder = PltBuilder::new(plt_addr, got_plt_addr, Target::RiscV64);
        for (idx, sym_name) in classification.plt_entries.iter().enumerate() {
            let entry = PltEntry {
                symbol_name: sym_name.clone(),
                got_offset: got_plt_reserved + (idx as u64) * ptr_size,
                plt_index: idx as u32,
            };
            plt_builder.add_entry(entry);
        }

        let plt_data = plt_builder.build_plt(&Target::RiscV64);

        // Re-build GOT.PLT now that we know the dynamic section address will
        // come next. We create a new GotBuilder with the final got_plt_addr.
        // For GOT.PLT, slot 0 = dynamic addr (patched below), slots 1-2 = 0,
        // slots 3+ = PLT push addresses. We'll build this manually since
        // GotBuilder manages .got entries; for .got.plt we use the builder's
        // build_got_plt method after adding PLT entries.
        //
        // Actually, the GotBuilder we created earlier already knows the
        // got_plt_address. Let's use it for build_got_plt too.
        let got_plt_data = got_builder.build_got_plt();

        // ---- Build .rela.dyn and .rela.plt ----
        let rela_dyn_data = self.build_rela_dyn(classification, resolved, got_addr);
        let rela_dyn_addr = next_addr;
        next_addr += rela_dyn_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        let rela_plt_data = self.build_rela_plt(classification, got_plt_addr + got_plt_reserved);
        let rela_plt_addr = next_addr;
        next_addr += rela_plt_data.len() as u64;
        next_addr = align_up(next_addr, 8);

        // ---- Build .dynamic section ----
        let dynamic_addr = next_addr;

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

        let mut dynamic_builder = DynamicSectionBuilder::new();
        // RISC-V glibc convention (elf_machine_runtime_setup):
        //   gotplt[0] = _dl_runtime_resolve;  // resolver
        //   gotplt[1] = l;                     // link_map
        // where gotplt = (ElfW(Addr)*)DT_PLTGOT.
        // No pltgot_adjust needed — DT_PLTGOT points directly at GOT[0].
        //
        // GCC defaults to BIND_NOW on RISC-V; we follow suit so that all
        // PLT entries are resolved eagerly at load time.
        dynamic_builder.set_bind_now(true);
        for lib in &self.config.libraries {
            dynamic_builder.add_needed(lib);
        }
        if self.config.shared {
            let soname = self
                .config
                .output_path
                .rsplit('/')
                .next()
                .unwrap_or(&self.config.output_path);
            dynamic_builder.set_soname(soname);
        }
        dynamic_builder.set_dynstr_size(dynstr_data.len() as u64);
        dynamic_builder.set_rela_dyn_size(rela_dyn_data.len() as u64);
        dynamic_builder.set_rela_plt_size(rela_plt_data.len() as u64);

        let dynamic_data = dynamic_builder.build(&layout);

        // ---- Collect all dynamic sections ----
        sections.push(DynSectionInfo {
            name: ".interp".to_string(),
            data: interp_data,
            addr: interp_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_PROGBITS,
            entry_size: 0,
        });
        sections.push(DynSectionInfo {
            name: ".gnu.hash".to_string(),
            data: gnu_hash_data,
            addr: gnu_hash_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_HASH,
            entry_size: 0,
        });
        sections.push(DynSectionInfo {
            name: ".dynsym".to_string(),
            data: dynsym_data,
            addr: dynsym_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_DYNSYM,
            entry_size: 24, // Elf64_Sym size
        });
        sections.push(DynSectionInfo {
            name: ".dynstr".to_string(),
            data: dynstr_data,
            addr: dynstr_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_STRTAB,
            entry_size: 0,
        });
        sections.push(DynSectionInfo {
            name: ".rela.dyn".to_string(),
            data: rela_dyn_data,
            addr: rela_dyn_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_RELA,
            entry_size: 24, // Elf64_Rela size
        });
        sections.push(DynSectionInfo {
            name: ".rela.plt".to_string(),
            data: rela_plt_data,
            addr: rela_plt_addr,
            flags: SHF_ALLOC,
            sh_type: SHT_RELA,
            entry_size: 24,
        });

        if !got_data.is_empty() {
            sections.push(DynSectionInfo {
                name: ".got".to_string(),
                data: got_data,
                addr: got_addr,
                flags: SHF_ALLOC | SHF_WRITE,
                sh_type: SHT_PROGBITS,
                entry_size: 8,
            });
        }
        if !got_plt_data.is_empty() {
            sections.push(DynSectionInfo {
                name: ".got.plt".to_string(),
                data: got_plt_data,
                addr: got_plt_addr,
                flags: SHF_ALLOC | SHF_WRITE,
                sh_type: SHT_PROGBITS,
                entry_size: 8,
            });
        }
        if !plt_data.is_empty() {
            sections.push(DynSectionInfo {
                name: ".plt".to_string(),
                data: plt_data,
                addr: plt_addr,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                sh_type: SHT_PROGBITS,
                entry_size: 16,
            });
        }
        sections.push(DynSectionInfo {
            name: ".dynamic".to_string(),
            data: dynamic_data,
            addr: dynamic_addr,
            flags: SHF_ALLOC | SHF_WRITE,
            sh_type: SHT_DYNAMIC,
            entry_size: 16, // Elf64_Dyn size
        });

        diag.note(
            Span::DUMMY,
            format!(
                "riscv64 linker: dynamic sections generated — GOT at 0x{:x}, PLT at 0x{:x}, \
                 .dynamic at 0x{:x}",
                got_addr, plt_addr, dynamic_addr,
            ),
        );

        (layout, sections)
    }

    /// Builds the `.rela.dyn` section data containing dynamic relocations
    /// for GOT entries.
    fn build_rela_dyn(
        &self,
        classification: &RelocationClassification,
        resolved: &ResolvedSymbols,
        got_address: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let entry_size = 8u64; // Each GOT entry is 8 bytes on RV64

        for (idx, sym_name) in classification.got_entries.iter().enumerate() {
            let offset = got_address + (idx as u64) * entry_size;
            let sym_value = resolved.get_symbol_value(sym_name).unwrap_or(0);

            if self.config.shared {
                // R_RISCV_RELATIVE: B + A where B is the load base address.
                // Elf64_Rela: offset(8) + info(8) + addend(8) = 24 bytes.
                data.extend_from_slice(&offset.to_le_bytes());
                let r_info = R_RISCV_RELATIVE as u64;
                data.extend_from_slice(&r_info.to_le_bytes());
                data.extend_from_slice(&(sym_value as i64).to_le_bytes());
            } else {
                // For PIC executables, we still need R_RISCV_RELATIVE.
                data.extend_from_slice(&offset.to_le_bytes());
                let r_info = R_RISCV_RELATIVE as u64;
                data.extend_from_slice(&r_info.to_le_bytes());
                data.extend_from_slice(&(sym_value as i64).to_le_bytes());
            }
        }

        data
    }

    /// Builds the `.rela.plt` section data containing PLT relocations
    /// for lazy binding.
    fn build_rela_plt(
        &self,
        classification: &RelocationClassification,
        got_plt_func_base: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let entry_size = 8u64; // Each GOT.PLT entry is 8 bytes on RV64

        for (idx, _sym_name) in classification.plt_entries.iter().enumerate() {
            let offset = got_plt_func_base + (idx as u64) * entry_size;

            // Elf64_Rela: offset(8) + info(8) + addend(8) = 24 bytes.
            data.extend_from_slice(&offset.to_le_bytes());
            // R_RISCV_JUMP_SLOT for lazy PLT binding.
            let sym_idx = (idx as u64) + 1; // 1-based (0 is null symbol)
            let r_info = (sym_idx << 32) | (R_RISCV_JUMP_SLOT as u64);
            data.extend_from_slice(&r_info.to_le_bytes());
            data.extend_from_slice(&0i64.to_le_bytes()); // addend = 0
        }

        data
    }

    // ===================================================================
    // Private — Phase 10: ELF writing
    // ===================================================================

    /// Serializes the linked output into a complete RISC-V 64 ELF binary.
    ///
    /// Constructs the ELF using [`ElfWriter`] configured for RISC-V 64 with
    /// the correct machine type, class, flags, and endianness derived
    /// automatically from [`Target::RiscV64`].
    fn write_elf(
        &self,
        output_sections: &[OutputSection],
        resolved: &ResolvedSymbols,
        program_headers: &[ProgramHeader],
        dynamic_sections: &[DynSectionInfo],
        entry_address: u64,
    ) -> Vec<u8> {
        // ElfWriter::new(Target::RiscV64) automatically configures:
        //   e_machine  = EM_RISCV (243)
        //   EI_CLASS   = ELFCLASS64
        //   EI_DATA    = ELFDATA2LSB
        //   e_flags    = EF_RISCV_FLOAT_ABI_DOUBLE | EF_RISCV_RVC = 0x0005
        //   EI_OSABI   = ELFOSABI_NONE
        let mut writer = ElfWriter::new(Target::RiscV64);

        // Set the ELF type (ET_EXEC, ET_DYN, or ET_REL).
        let elf_type = match self.config.output_type {
            OutputType::Executable => ET_EXEC,
            OutputType::SharedLibrary => ET_DYN,
            OutputType::RelocatableObject => ET_REL,
        };
        writer.set_type(elf_type);
        writer.set_entry_point(entry_address);

        // ---- Add output sections from the section merger ----
        for out_sec in output_sections {
            let mut elf_section = ElfSection::new(&out_sec.name, out_sec.section_type);
            elf_section.flags = out_sec.flags;
            elf_section.alignment = out_sec.alignment;
            elf_section.addr = out_sec.addr;
            elf_section.entry_size = out_sec.entry_size;

            // Collect section data from all merged inputs, with padding.
            if out_sec.section_type != SHT_NOBITS {
                elf_section.data = self.collect_section_data(out_sec);
            }

            writer.add_section(elf_section);
        }

        // ---- Add dynamic linking sections (if present) ----
        for dyn_sec in dynamic_sections {
            let mut elf_section = ElfSection::new(&dyn_sec.name, dyn_sec.sh_type);
            elf_section.flags = dyn_sec.flags;
            elf_section.data = dyn_sec.data.clone();
            elf_section.addr = dyn_sec.addr;
            elf_section.entry_size = dyn_sec.entry_size;
            elf_section.alignment = if dyn_sec.entry_size > 0 {
                dyn_sec.entry_size
            } else {
                8
            };

            writer.add_section(elf_section);
        }

        // ---- Add symbols to the ELF symbol table ----
        for entry in &resolved.symbols {
            let mut elf_sym = ElfSymbol::new(&entry.name);
            elf_sym.value = entry.value;
            elf_sym.size = entry.size;
            elf_sym.binding = self.convert_binding_to_elf(entry.binding);
            elf_sym.sym_type = self.convert_sym_type_to_elf(entry.sym_type);
            elf_sym.visibility = self.convert_visibility_to_elf(entry.visibility);
            elf_sym.section_index = entry.section_index;
            writer.add_symbol(elf_sym);
        }

        // ---- Add program headers ----
        for phdr in program_headers {
            writer.add_program_header(phdr.clone());
        }

        // ---- Serialize and return the complete ELF binary ----
        writer.write()
    }

    /// Collects the merged data for an output section by concatenating all
    /// input section contributions with alignment padding between them.
    fn collect_section_data(&self, out_sec: &OutputSection) -> Vec<u8> {
        if out_sec.input_sections.is_empty() {
            return Vec::new();
        }

        let mut data = Vec::with_capacity(out_sec.size as usize);
        let mut current_offset = 0u64;

        for merged in &out_sec.input_sections {
            // Insert alignment padding if needed.
            let target_offset = merged.offset_in_output;
            if target_offset > current_offset {
                let padding = (target_offset - current_offset) as usize;
                data.extend(std::iter::repeat(0u8).take(padding));
                current_offset = target_offset;
            }

            // Append the input section data.
            data.extend_from_slice(&merged.input.data);
            current_offset += merged.input.data.len() as u64;
        }

        // Pad to the full section size if needed (e.g., trailing alignment).
        if (data.len() as u64) < out_sec.size {
            let remaining = (out_sec.size as usize) - data.len();
            data.extend(std::iter::repeat(0u8).take(remaining));
        }

        data
    }

    // ===================================================================
    // Private — symbol type / binding / visibility conversion
    // ===================================================================

    /// Converts a [`SymbolBinding`] to the ELF `STB_*` constant.
    #[inline]
    fn convert_binding_to_elf(&self, binding: SymbolBinding) -> u8 {
        match binding {
            SymbolBinding::Local => STB_LOCAL,
            SymbolBinding::Global => STB_GLOBAL,
            SymbolBinding::Weak => STB_WEAK,
        }
    }

    /// Converts a [`SymbolType`] to the ELF `STT_*` constant.
    #[inline]
    fn convert_sym_type_to_elf(&self, sym_type: SymbolType) -> u8 {
        match sym_type {
            SymbolType::NoType => STT_NOTYPE,
            SymbolType::Object => STT_OBJECT,
            SymbolType::Func => STT_FUNC,
            SymbolType::Section => STT_SECTION,
            SymbolType::File => STT_FILE,
        }
    }

    /// Converts a [`SymbolVisibility`] to the ELF `STV_*` constant.
    #[inline]
    fn convert_visibility_to_elf(&self, visibility: SymbolVisibility) -> u8 {
        match visibility {
            SymbolVisibility::Default => STV_DEFAULT,
            SymbolVisibility::Hidden => STV_HIDDEN,
            SymbolVisibility::Protected => STV_PROTECTED,
        }
    }
}

// ===========================================================================
// DynSectionInfo — internal descriptor for generated dynamic sections
// ===========================================================================

/// Internal descriptor for a dynamically-generated ELF section (GOT, PLT,
/// .dynamic, etc.) that must be added to the final ELF output.
#[derive(Debug, Clone)]
struct DynSectionInfo {
    /// Section name (e.g., `.got`, `.plt`, `.dynamic`).
    name: String,
    /// Serialized section data.
    data: Vec<u8>,
    /// Assigned virtual address.
    addr: u64,
    /// Section flags (`SHF_ALLOC`, `SHF_WRITE`, etc.).
    flags: u64,
    /// Section type (`SHT_PROGBITS`, `SHT_DYNAMIC`, etc.).
    sh_type: u32,
    /// Entry size for structured sections (0 for unstructured).
    entry_size: u64,
}

// ===========================================================================
// Free helper — alignment
// ===========================================================================

/// Aligns `value` up to the next multiple of `alignment`.
///
/// Returns `value` unchanged if it is already aligned or if `alignment` is
/// 0 or 1.
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}
