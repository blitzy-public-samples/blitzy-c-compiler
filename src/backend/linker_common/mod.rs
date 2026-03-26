//! # BCC Linker Common — Shared Linker Infrastructure
//!
//! This module provides architecture-agnostic linker infrastructure shared by all
//! four target architecture linkers (x86-64, i686, AArch64, RISC-V 64).
//!
//! ## Standalone Backend Mode
//! BCC includes its own built-in linker. NO external linker is invoked
//! (no `ld`, `lld`, `gold`). This module implements the core linker logic.
//!
//! ## Submodules
//! - [`symbol_resolver`] — Two-pass symbol resolution: collect all symbols from input objects,
//!   resolve references using strong/weak binding rules, report undefined symbol errors,
//!   and handle archive scanning for static libraries.
//! - [`section_merger`] — Input section aggregation from multiple object files into output
//!   sections, with alignment padding, COMDAT group deduplication, and standard section ordering
//!   (`.text`, `.rodata`, `.data`, `.bss`).
//! - [`relocation`] — Architecture-agnostic relocation processing framework: collects relocations
//!   from input objects, resolves targets, and dispatches application to architecture-specific
//!   handlers via the `RelocationHandler` trait.
//! - [`dynamic`] — Dynamic linking section generation for shared objects (ET_DYN):
//!   `.dynamic`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt`, `.gnu.hash`,
//!   `.got`, `.got.plt`, `.plt` stub generation, `PT_DYNAMIC`, `PT_INTERP`.
//! - [`linker_script`] — Default section-to-segment mapping: `.text` → `PT_LOAD (R+X)`,
//!   `.rodata` → `PT_LOAD (R)`, `.data`/`.bss` → `PT_LOAD (R+W)`, entry point (`_start`),
//!   `PT_PHDR`, `PT_GNU_STACK`.
//!
//! ## Supported Output Formats
//! - `ET_EXEC` — Static ELF executables
//! - `ET_DYN` — Shared objects (`.so` files)
//!
//! ## Architecture-Specific Delegation
//! Architecture-specific relocation application is delegated to each backend's
//! `linker/relocations.rs` module via the [`relocation::RelocationHandler`] trait:
//! - `src/backend/x86_64/linker/relocations.rs`
//! - `src/backend/i686/linker/relocations.rs`
//! - `src/backend/aarch64/linker/relocations.rs`
//! - `src/backend/riscv64/linker/relocations.rs`

// ============================================================================
// Submodule declarations — exactly 5 public submodules
// ============================================================================

/// Two-pass symbol resolution engine with strong/weak binding rules, archive
/// scanning, undefined symbol detection, and final symbol table generation.
pub mod symbol_resolver;

/// Input section aggregation from multiple object files into output sections,
/// with alignment padding, COMDAT group deduplication, and standard section ordering.
pub mod section_merger;

/// Architecture-agnostic relocation processing framework: collection,
/// classification, resolution, and architecture-dispatched application.
pub mod relocation;

/// Dynamic linking section generation for shared objects (ET_DYN):
/// `.dynamic`, `.dynsym`, `.dynstr`, `.gnu.hash`, `.got`, `.got.plt`, `.plt`,
/// `.rela.dyn`, `.rela.plt`, `PT_DYNAMIC`, `PT_INTERP`.
pub mod dynamic;

/// Default section-to-segment mapping and address layout computation for
/// both `ET_EXEC` (static executables) and `ET_DYN` (shared objects).
pub mod linker_script;

// ============================================================================
// Re-exports — key types for convenient access by architecture backends
// ============================================================================

pub use dynamic::{
    DynamicLinkContext, DynamicSymbol, DynamicSymbolTable, GlobalOffsetTable, ProcedureLinkageTable,
};
pub use linker_script::{DefaultLinkerScript, LayoutResult, SectionLayout, SegmentLayout};
pub use relocation::{
    RelocCategory, Relocation, RelocationError, RelocationHandler, ResolvedRelocation,
};
pub use section_merger::{InputSection, OutputSection, SectionFragment, SectionMerger};
pub use symbol_resolver::{InputSymbol, OutputSymbol, ResolvedSymbol, SymbolResolver, SymbolTable};

// ============================================================================
// Imports from crate::common
// ============================================================================

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

use crate::backend::elf_writer_common::{ElfSymbol, ElfWriter, Section, ET_DYN, ET_EXEC};

// ============================================================================
// OutputType — linker output format selection
// ============================================================================

/// Represents the output type for linking.
///
/// Maps directly to ELF object types:
/// - [`Executable`](OutputType::Executable) → `ET_EXEC`
/// - [`SharedLibrary`](OutputType::SharedLibrary) → `ET_DYN`
/// - [`Relocatable`](OutputType::Relocatable) → `ET_REL` (passthrough, no linking)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// ET_EXEC — Static ELF executable.
    Executable,
    /// ET_DYN — Shared object (`.so`).
    SharedLibrary,
    /// ET_REL — Relocatable object (`.o`) — passthrough, no linking needed.
    Relocatable,
}

impl OutputType {
    /// Convert to the corresponding ELF `e_type` header value.
    ///
    /// Returns the ELF type constant for `ET_EXEC`, `ET_DYN`, or `ET_REL`.
    #[inline]
    pub fn to_elf_type(self) -> u16 {
        match self {
            OutputType::Executable => ET_EXEC,
            OutputType::SharedLibrary => ET_DYN,
            OutputType::Relocatable => crate::backend::elf_writer_common::ET_REL,
        }
    }

    /// Returns `true` if this output type requires position-independent code.
    #[inline]
    pub fn requires_pic(self) -> bool {
        matches!(self, OutputType::SharedLibrary)
    }

    /// Returns `true` if this output type produces a linked (non-relocatable) binary.
    #[inline]
    pub fn is_linked(self) -> bool {
        !matches!(self, OutputType::Relocatable)
    }
}

impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputType::Executable => write!(f, "executable"),
            OutputType::SharedLibrary => write!(f, "shared library"),
            OutputType::Relocatable => write!(f, "relocatable"),
        }
    }
}

// ============================================================================
// LinkerConfig — configuration collected from CLI flags
// ============================================================================

/// Linker configuration options collected from CLI flags.
///
/// Aggregates all linker-relevant settings parsed from the BCC command line,
/// including target architecture, output type, library search paths, and
/// flags controlling PIC mode, debug emission, and security features.
///
/// # Example
///
/// ```ignore
/// use bcc::backend::linker_common::{LinkerConfig, OutputType};
/// use bcc::common::target::Target;
///
/// let config = LinkerConfig {
///     target: Target::X86_64,
///     output_type: OutputType::Executable,
///     output_path: "a.out".to_string(),
///     entry_point: "_start".to_string(),
///     library_paths: vec!["/usr/lib".to_string()],
///     libraries: vec!["c".to_string()],
///     pic: false,
///     allow_undefined: false,
///     soname: None,
///     needed_libs: Vec::new(),
///     emit_debug: false,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LinkerConfig {
    /// Target architecture (x86-64, i686, AArch64, or RISC-V 64).
    pub target: Target,

    /// Output type (executable, shared library, or relocatable).
    pub output_type: OutputType,

    /// Output file path (e.g., `"a.out"`, `"libfoo.so"`).
    pub output_path: String,

    /// Entry point symbol name. Defaults to `"_start"` for executables.
    /// Ignored for shared libraries and relocatable objects.
    pub entry_point: String,

    /// Library search paths (`-L` flags).
    pub library_paths: Vec<String>,

    /// Libraries to link (`-l` flags — names without the `lib` prefix).
    pub libraries: Vec<String>,

    /// Whether to generate position-independent code (`-fPIC`).
    pub pic: bool,

    /// Whether to allow undefined symbols at link time.
    /// Typically `true` for shared libraries, `false` for executables.
    pub allow_undefined: bool,

    /// SONAME for shared libraries (`-soname` / `DT_SONAME`).
    /// `None` if no SONAME was specified.
    pub soname: Option<String>,

    /// Additional needed libraries (names for `DT_NEEDED` entries).
    pub needed_libs: Vec<String>,

    /// Whether to emit DWARF debug sections in the output.
    pub emit_debug: bool,
}

impl LinkerConfig {
    /// Create a minimal default configuration for the given target and output type.
    ///
    /// All paths are empty and optional fields default to `None` / `false`.
    pub fn new(target: Target, output_type: OutputType) -> Self {
        LinkerConfig {
            target,
            output_type,
            output_path: String::from("a.out"),
            entry_point: String::from("_start"),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            pic: output_type.requires_pic(),
            allow_undefined: matches!(output_type, OutputType::SharedLibrary),
            soname: None,
            needed_libs: Vec::new(),
            emit_debug: false,
        }
    }

    /// Returns `true` if the output is a shared library.
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.output_type == OutputType::SharedLibrary
    }

    /// Returns `true` if the output is an executable.
    #[inline]
    pub fn is_executable(&self) -> bool {
        self.output_type == OutputType::Executable
    }

    /// Returns `true` if the output is a relocatable object (no linking).
    #[inline]
    pub fn is_relocatable(&self) -> bool {
        self.output_type == OutputType::Relocatable
    }
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self::new(Target::X86_64, OutputType::Executable)
    }
}

// ============================================================================
// LinkerInput — parsed input object file ready for linking
// ============================================================================

/// Represents a parsed input object file ready for linking.
///
/// Each `LinkerInput` corresponds to one `.o` file (or archive member)
/// provided to the linker. It carries the object's sections, symbols, and
/// relocations in the common representation used by the linker infrastructure.
#[derive(Debug)]
pub struct LinkerInput {
    /// Unique identifier for this input object. Used for provenance tracking
    /// in error messages and for correlating symbols back to their origin.
    pub object_id: u32,

    /// Original file name or archive member name.
    pub filename: String,

    /// Input sections extracted from this object file.
    pub sections: Vec<section_merger::InputSection>,

    /// Symbols defined or referenced by this object file.
    pub symbols: Vec<symbol_resolver::InputSymbol>,

    /// Relocations from this object file.
    pub relocations: Vec<relocation::Relocation>,
}

impl LinkerInput {
    /// Create a new `LinkerInput` with the given object ID and filename.
    ///
    /// Sections, symbols, and relocations are initially empty and should
    /// be populated by the object file parser.
    pub fn new(object_id: u32, filename: String) -> Self {
        LinkerInput {
            object_id,
            filename,
            sections: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
        }
    }

    /// Returns the number of input sections in this object.
    #[inline]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Returns the number of symbols in this object.
    #[inline]
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the number of relocations in this object.
    #[inline]
    pub fn relocation_count(&self) -> usize {
        self.relocations.len()
    }
}

// ============================================================================
// LinkerOutput — final linked output
// ============================================================================

/// Represents the final linked output — a complete ELF binary.
///
/// Contains the serialized ELF byte stream, the resolved entry point address,
/// and the output type for downstream processing (e.g., writing to disk,
/// verification).
pub struct LinkerOutput {
    /// The complete ELF binary as a byte vector.
    ///
    /// This is a valid ELF file that can be written directly to disk,
    /// inspected with `readelf`/`objdump`, and (for executables) executed.
    pub elf_data: Vec<u8>,

    /// Resolved entry point virtual address.
    ///
    /// For executables, this is the address of the `_start` symbol (or
    /// whatever symbol was specified as the entry point). For shared
    /// libraries, this is typically 0.
    pub entry_point: u64,

    /// The output type that was produced.
    pub output_type: OutputType,
}

impl LinkerOutput {
    /// Returns the size of the ELF output in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.elf_data.len()
    }

    /// Returns `true` if the output is an executable.
    #[inline]
    pub fn is_executable(&self) -> bool {
        self.output_type == OutputType::Executable
    }

    /// Returns `true` if the output is a shared library.
    #[inline]
    pub fn is_shared_library(&self) -> bool {
        self.output_type == OutputType::SharedLibrary
    }
}

impl std::fmt::Debug for LinkerOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinkerOutput")
            .field("elf_data_len", &self.elf_data.len())
            .field("entry_point", &format_args!("0x{:x}", self.entry_point))
            .field("output_type", &self.output_type)
            .finish()
    }
}

// ============================================================================
// link() — High-Level Linker Driver
// ============================================================================

/// Perform the complete link operation.
///
/// This function orchestrates the full linking pipeline:
///
/// 1. **Symbol Resolution** — Collect all symbols from input objects, apply
///    strong/weak binding rules, detect multiple definitions, and report
///    undefined symbols.
/// 2. **Section Merging** — Aggregate input sections into output sections,
///    apply alignment padding, and deduplicate COMDAT groups.
/// 3. **Address Layout** — Compute virtual addresses and file offsets for all
///    output sections using the default linker script.
/// 4. **Relocation Processing** — Resolve and apply all relocations using the
///    architecture-specific `RelocationHandler`.
/// 5. **Dynamic Section Generation** (if `ET_DYN`) — Build `.dynamic`,
///    `.dynsym`, `.dynstr`, `.gnu.hash`, `.got`, `.got.plt`, `.plt`,
///    `.rela.dyn`, `.rela.plt`, and `PT_INTERP`.
/// 6. **ELF Output Writing** — Serialize the linked binary using `ElfWriter`.
///
/// # Arguments
///
/// * `config` — Linker configuration (target, output type, paths, flags).
/// * `inputs` — List of parsed input object files to link.
/// * `handler` — Architecture-specific relocation handler (trait object).
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// On success, returns a [`LinkerOutput`] containing the complete ELF binary.
/// On failure, returns a descriptive error string.
///
/// # Errors
///
/// Returns `Err` if:
/// - There are undefined symbol errors (and `allow_undefined` is false)
/// - Relocation processing encounters overflow or unsupported types
/// - Section merging fails due to incompatible section attributes
/// - The entry point symbol cannot be resolved (for executables)
pub fn link(
    config: &LinkerConfig,
    inputs: Vec<LinkerInput>,
    handler: &dyn relocation::RelocationHandler,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    // -----------------------------------------------------------------------
    // Relocatable passthrough — no linking needed
    // -----------------------------------------------------------------------
    if config.output_type == OutputType::Relocatable {
        return link_relocatable(config, inputs, diagnostics);
    }

    // -----------------------------------------------------------------------
    // Phase 1: Symbol Resolution
    // -----------------------------------------------------------------------
    let mut resolver = symbol_resolver::SymbolResolver::new();

    for input in &inputs {
        resolver.collect_symbols(input.object_id, &input.symbols);
    }

    if let Err(errors) = resolver.resolve() {
        for err_msg in &errors {
            diagnostics.emit_error(Span::dummy(), err_msg.clone());
        }
        if !config.allow_undefined {
            return Err(format!(
                "symbol resolution failed with {} error(s)",
                errors.len()
            ));
        }
    }

    // Check for undefined symbols (fatal for executables unless allow_undefined).
    if let Err(undef_errors) = resolver.check_undefined(config.allow_undefined) {
        for err_msg in &undef_errors {
            diagnostics.emit_error(Span::dummy(), err_msg.clone());
        }
        if !config.allow_undefined {
            return Err(format!(
                "linking failed: {} undefined symbol(s)",
                undef_errors.len()
            ));
        }
    }

    // Emit accumulated symbol resolution diagnostics.
    resolver.emit_diagnostics(diagnostics);

    if diagnostics.has_errors() && !config.allow_undefined {
        return Err("linking aborted due to symbol resolution errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase 2: Section Merging
    // -----------------------------------------------------------------------
    let mut merger = section_merger::SectionMerger::new(config.target);

    for input in &inputs {
        for section in &input.sections {
            merger.add_input_section(section.clone());
        }
    }

    // Compute the base address for the target and output type.
    let is_shared = config.is_shared();

    // -----------------------------------------------------------------------
    // Authoritative Address Layout via Linker Script
    // -----------------------------------------------------------------------
    // Use the linker script as the SINGLE source of truth for virtual addresses
    // and file offsets.  Both the symbol table and the ELF program headers are
    // derived from this layout, which guarantees they are mutually consistent.
    let ordered_sections_for_layout = merger.get_ordered_sections();
    let section_infos_for_layout: Vec<linker_script::InputSectionInfo> =
        ordered_sections_for_layout
            .iter()
            .map(|sec| linker_script::InputSectionInfo {
                name: sec.name.clone(),
                size: sec.total_size,
                alignment: sec.alignment,
                flags: sec.flags as u32,
            })
            .collect();

    let mut layout_script = linker_script::DefaultLinkerScript::new(config.target, is_shared);
    let layout_result = layout_script.compute_layout(&section_infos_for_layout);

    // Build the address map from the linker-script layout.
    let mut address_map = section_merger::AddressMap {
        section_addresses: FxHashMap::default(),
    };
    for sl in &layout_result.sections {
        address_map.section_addresses.insert(
            sl.name.clone(),
            section_merger::SectionAddress {
                virtual_address: sl.virtual_address,
                file_offset: sl.file_offset,
                size: sl.size,
                mem_size: sl.mem_size,
            },
        );
    }

    // -----------------------------------------------------------------------
    // Phase 2b: Relocate Symbol Values
    // -----------------------------------------------------------------------
    // After section merging assigned virtual addresses, update each symbol's
    // value from a section-relative offset to an absolute virtual address.
    // Build a mapping: (object_id, section_index) → section_name.
    let mut section_index_to_name: FxHashMap<(u32, u16), String> = FxHashMap::default();
    for input in &inputs {
        for section in &input.sections {
            section_index_to_name.insert(
                (section.object_id, section.original_index as u16),
                section.name.clone(),
            );
        }
    }

    // Build a fragment-offset map so that symbols from multi-object
    // linking get correct addresses within merged output sections.
    // (object_id, section_original_index) → offset_in_merged_output_section
    let mut fragment_offset_map: FxHashMap<(u32, u16), u64> = FxHashMap::default();
    for out_sec in merger.get_ordered_sections() {
        for frag in &out_sec.fragments {
            fragment_offset_map.insert(
                (
                    frag.input_section_ref.object_id,
                    frag.input_section_ref.section_index as u16,
                ),
                frag.offset_in_output,
            );
        }
    }

    // For each resolved symbol, compute:
    //   final_address = section_va + fragment_offset + symbol_offset.
    resolver.relocate_symbol_addresses(&address_map, &section_index_to_name, &fragment_offset_map);

    // Also relocate local symbols.
    resolver.relocate_local_symbol_addresses(
        &address_map,
        &section_index_to_name,
        &fragment_offset_map,
    );

    // -----------------------------------------------------------------------
    // Phase 3: Linker-Defined Symbols
    // -----------------------------------------------------------------------
    // Build a map of section name → (virtual_address, size) for linker symbols.
    let mut section_addr_map: FxHashMap<String, (u64, u64)> = FxHashMap::default();
    for (name, addr_info) in &address_map.section_addresses {
        section_addr_map.insert(name.clone(), (addr_info.virtual_address, addr_info.size));
    }
    resolver.define_linker_symbols(&section_addr_map);

    // Build the final resolved symbol table.
    let mut sym_table = resolver.build_symbol_table();

    // -----------------------------------------------------------------------
    // Phase 4: Build Symbol Address Map for Relocations
    // -----------------------------------------------------------------------
    let mut symbol_address_map: FxHashMap<String, symbol_resolver::ResolvedSymbol> =
        FxHashMap::default();

    // Populate from the resolved output symbols.
    for sym in &sym_table.symbols {
        let resolved = symbol_resolver::ResolvedSymbol {
            name: sym.name.clone(),
            final_address: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_name: String::new(),
            is_defined: sym.section_index != symbol_resolver::SHN_UNDEF,
            from_object: 0,
            export_dynamic: false,
        };
        symbol_address_map.insert(sym.name.clone(), resolved);
    }

    // -----------------------------------------------------------------------
    // Phase 5: Collect Relocations & Build Section Data
    // -----------------------------------------------------------------------
    let mut all_relocations: Vec<relocation::Relocation> = Vec::new();
    for input in &inputs {
        all_relocations.extend(input.relocations.iter().cloned());
    }

    // For multi-object linking, adjust relocation offsets by the fragment
    // offset within merged output sections.  Build a
    // (object_id, section_index) → (output_section_name, fragment_offset)
    // mapping and annotate each relocation.
    {
        let ordered_secs = merger.get_ordered_sections();
        let input_to_output = relocation::build_input_to_output_map(
            &ordered_secs
                .iter()
                .map(|s| (*s).clone())
                .collect::<Vec<_>>(),
        );
        relocation::annotate_relocations(&mut all_relocations, &input_to_output);
    }

    let mut section_data_map: FxHashMap<String, Vec<u8>> = FxHashMap::default();
    let ordered_sections = merger.get_ordered_sections();
    for output_section in &ordered_sections {
        let data = merger.build_section_data(&output_section.name);
        section_data_map.insert(output_section.name.clone(), data);
    }

    // -----------------------------------------------------------------------
    // Detect if `atexit` is referenced but cannot be resolved from shared
    // libraries (glibc ≥ 2.34 moved it into libc_nonshared.a).  When
    // detected we synthesise a tiny wrapper: atexit(fn) → __cxa_atexit(fn, 0, 0).
    // -----------------------------------------------------------------------
    let need_synthetic_atexit = config.is_executable() && !config.needed_libs.is_empty() && {
        let has_atexit_ref = inputs
            .iter()
            .any(|inp| inp.relocations.iter().any(|r| r.symbol_name == "atexit"));
        let has_atexit_def = sym_table
            .symbols
            .iter()
            .any(|s| s.name == "atexit" && s.section_index != symbol_resolver::SHN_UNDEF);
        has_atexit_ref && !has_atexit_def
    };

    // If we need synthetic atexit, ensure __cxa_atexit is imported.
    if need_synthetic_atexit && !symbol_address_map.contains_key("__cxa_atexit") {
        let cxa_sym = symbol_resolver::OutputSymbol {
            name: "__cxa_atexit".to_string(),
            value: 0,
            size: 0,
            binding: 1,  // STB_GLOBAL
            sym_type: 2, // STT_FUNC
            visibility: 0,
            section_index: symbol_resolver::SHN_UNDEF,
        };
        sym_table.symbols.push(cxa_sym);
        symbol_address_map.insert(
            "__cxa_atexit".to_string(),
            symbol_resolver::ResolvedSymbol {
                name: "__cxa_atexit".to_string(),
                final_address: 0,
                size: 0,
                binding: 1,
                sym_type: 2,
                visibility: 0,
                section_name: String::new(),
                is_defined: false,
                from_object: 0,
                export_dynamic: false,
            },
        );
    }

    // -----------------------------------------------------------------------
    // Phase 5.4: Synthetic atexit stub (glibc ≥ 2.34 compatibility)
    // -----------------------------------------------------------------------
    // In glibc ≥ 2.34, `atexit` is no longer in `libc.so.6`; it lives in the
    // static `libc_nonshared.a`.  We synthesise a small wrapper that calls
    // `__cxa_atexit(fn, 0, 0)` — injected into .text with a known address
    // BEFORE dynamic section generation so that:
    //   • atexit becomes a *defined* symbol (excluded from .dynsym imports)
    //   • relocation processing (Phase 6) can resolve references to atexit
    //   • the wrapper's call to __cxa_atexit will be patched after PLT is built
    if need_synthetic_atexit {
        let text_vaddr = address_map
            .section_addresses
            .get(".text")
            .map(|a| a.virtual_address)
            .unwrap_or(0x401000);
        let text_size = section_data_map
            .get(".text")
            .map(|d| d.len() as u64)
            .unwrap_or(0);

        // Align stub to 16 bytes within .text.
        let padding = ((16 - (text_size % 16)) % 16) as usize;
        let atexit_offset = text_size + padding as u64;
        let atexit_vaddr = text_vaddr + atexit_offset;

        // Generate a *placeholder* stub.  We fill the __cxa_atexit address
        // field with zeros for now; it will be patched in Phase 7b after PLT
        // addresses are known.
        let placeholder_code = generate_synthetic_atexit(config.target, atexit_vaddr, 0);
        let atexit_len = placeholder_code.len() as u64;

        // Extend .text section data.
        if let Some(text_data) = section_data_map.get_mut(".text") {
            text_data.extend(std::iter::repeat(0xCC).take(padding));
            text_data.extend_from_slice(&placeholder_code);
        } else {
            let mut data = vec![0xCC; padding];
            data.extend_from_slice(&placeholder_code);
            section_data_map.insert(".text".to_string(), data);
        }

        // Update .text section size.
        if let Some(text_addr) = address_map.section_addresses.get_mut(".text") {
            text_addr.size = atexit_offset + atexit_len;
            text_addr.mem_size = atexit_offset + atexit_len;
        }

        // Replace any existing UND `atexit` entry with the DEFINED version.
        // This prevents the dynamic section generator from creating a PLT
        // stub for atexit that would fail resolution at runtime (since atexit
        // is not in libc.so.6 on glibc ≥ 2.34).
        let mut replaced = false;
        for sym in sym_table.symbols.iter_mut() {
            if sym.name == "atexit" && sym.section_index == symbol_resolver::SHN_UNDEF {
                sym.value = atexit_vaddr;
                sym.size = atexit_len;
                sym.section_index = symbol_resolver::SHN_ABS;
                replaced = true;
                break;
            }
        }
        if !replaced {
            sym_table.symbols.push(symbol_resolver::OutputSymbol {
                name: "atexit".to_string(),
                value: atexit_vaddr,
                size: atexit_len,
                binding: 1,    // STB_GLOBAL
                sym_type: 2,   // STT_FUNC
                visibility: 0, // STV_DEFAULT
                section_index: symbol_resolver::SHN_ABS,
            });
        }

        // Also register in symbol_address_map so relocation processing finds it.
        symbol_address_map.insert(
            "atexit".to_string(),
            symbol_resolver::ResolvedSymbol {
                name: "atexit".to_string(),
                final_address: atexit_vaddr,
                size: atexit_len,
                binding: 1,
                sym_type: 2,
                visibility: 0,
                section_name: ".text".to_string(),
                is_defined: true,
                from_object: 0,
                export_dynamic: false,
            },
        );
    }

    // -----------------------------------------------------------------------
    // Phase 5.5: Pre-generate Dynamic Sections (PLT/GOT) BEFORE relocations
    // -----------------------------------------------------------------------
    // For dynamically-linked executables and shared libraries, we must
    // generate PLT stubs and assign them virtual addresses BEFORE relocation
    // processing, so that CALL/BL relocations to external symbols can be
    // resolved to PLT stub addresses.
    //
    // We also pre-compute virtual addresses for each dynamic section using
    // the same algorithm as write_elf_output, starting from the end of the
    // existing layout. This ensures the PLT addresses used during relocation
    // processing match the final ELF output.
    let mut dyn_section_addresses: FxHashMap<String, (u64, u64)> = FxHashMap::default();

    // When producing a dynamically-linked executable and `_start` is missing
    // but `main` is defined, we will synthesise a `_start` stub in Phase 7.
    // That stub calls `exit()` to flush stdio buffers after `main` returns.
    // Ensure `exit` is registered as an undefined import symbol so it gets a
    // `.dynsym` entry and PLT stub.
    let need_synthetic_start = config.is_executable()
        && !config.needed_libs.is_empty()
        && !sym_table
            .symbols
            .iter()
            .any(|s| s.name == "_start" || s.name == "__start")
        && sym_table.symbols.iter().any(|s| s.name == "main");

    if need_synthetic_start && !symbol_address_map.contains_key("exit") {
        // Inject `exit` as an undefined import symbol.
        let exit_sym = symbol_resolver::OutputSymbol {
            name: "exit".to_string(),
            value: 0,
            size: 0,
            binding: 1,  // STB_GLOBAL
            sym_type: 2, // STT_FUNC
            visibility: 0,
            section_index: symbol_resolver::SHN_UNDEF,
        };
        sym_table.symbols.push(exit_sym);
        symbol_address_map.insert(
            "exit".to_string(),
            symbol_resolver::ResolvedSymbol {
                name: "exit".to_string(),
                final_address: 0,
                size: 0,
                binding: 1,
                sym_type: 2,
                visibility: 0,
                section_name: String::new(),
                is_defined: false,
                from_object: 0,
                export_dynamic: false,
            },
        );
    }

    if config.is_shared() || !config.needed_libs.is_empty() {
        // Collect referenced symbols from relocations for .dynsym filtering.
        let mut referenced_symbols: FxHashSet<String> = FxHashSet::default();
        for input in &inputs {
            for reloc in &input.relocations {
                if !reloc.symbol_name.is_empty() {
                    referenced_symbols.insert(reloc.symbol_name.clone());
                }
            }
        }

        // When producing an executable and _start is missing but main is
        // defined, we will inject a synthetic _start stub in Phase 7.
        // That stub needs to call exit() via PLT to flush stdio buffers,
        // so we must ensure `exit` appears in the dynamic symbol table
        // (and therefore gets a PLT entry) before generating sections.
        if config.is_executable() {
            let has_start = sym_table
                .symbols
                .iter()
                .any(|s| s.name == "_start" || s.name == "__start");
            let has_main = sym_table.symbols.iter().any(|s| s.name == "main");
            if !has_start && has_main {
                referenced_symbols.insert("exit".to_string());
            }
        }

        // Build a set of symbols that need PLT stubs for dynamic resolution.
        //
        // Strategy: any undefined symbol referenced by ANY relocation needs
        // dynamic resolution. For symbols referenced by PLT-specific reloc
        // types (R_X86_64_PLT32, etc.) this is obvious. But BCC-generated
        // .o files also reference external functions via R_X86_64_64 and
        // R_X86_64_PC32, and libc data symbols (stdout, stderr, stdin) use
        // R_X86_64_PC32. All of these need PLT/GOT entries to be resolvable
        // at runtime by the dynamic linker.
        let mut plt_needed_symbols: FxHashSet<String> = FxHashSet::default();

        // First pass: collect symbols from PLT-type relocations.
        for input in &inputs {
            for reloc in &input.relocations {
                if !reloc.symbol_name.is_empty() && handler.needs_plt(reloc.rel_type) {
                    plt_needed_symbols.insert(reloc.symbol_name.clone());
                }
            }
        }

        // Second pass: for any undefined symbol referenced by non-PLT
        // relocations, also add it to the PLT set so it gets a dynamic
        // resolution stub. Known data symbols (stderr, stdout, stdin, etc.)
        // are excluded from PLT and collected separately for copy relocation.
        let mut copy_symbols: Vec<(String, usize)> = Vec::new();
        {
            let defined_names: FxHashSet<String> = sym_table
                .symbols
                .iter()
                .filter(|s| s.section_index != symbol_resolver::SHN_UNDEF)
                .map(|s| s.name.clone())
                .collect();
            for input in &inputs {
                for reloc in &input.relocations {
                    if !reloc.symbol_name.is_empty()
                        && !defined_names.contains(&reloc.symbol_name)
                        && !plt_needed_symbols.contains(&reloc.symbol_name)
                    {
                        // Check if this is a known data symbol — if so, it
                        // needs a copy relocation instead of a PLT entry.
                        if is_known_data_symbol(&reloc.symbol_name) {
                            if !copy_symbols.iter().any(|(n, _)| *n == reloc.symbol_name) {
                                let sz = known_data_symbol_size(&reloc.symbol_name);
                                copy_symbols.push((reloc.symbol_name.clone(), sz));
                            }
                        } else {
                            plt_needed_symbols.insert(reloc.symbol_name.clone());
                        }
                    }
                }
            }
        }
        // The synthetic _start stub calls exit via PLT.
        if need_synthetic_start {
            plt_needed_symbols.insert("exit".to_string());
        }

        // When synthesising atexit, add __cxa_atexit to PLT and keep atexit out.
        if need_synthetic_atexit {
            // atexit is defined by us (synthetic) — remove from PLT/import set.
            plt_needed_symbols.remove("atexit");
            referenced_symbols.remove("atexit");
            // __cxa_atexit (in libc.so.6) must be importable via PLT.
            plt_needed_symbols.insert("__cxa_atexit".to_string());
            referenced_symbols.insert("__cxa_atexit".to_string());
        }

        // Generate all dynamic linking sections (PLT, GOT, .dynamic, etc.).
        generate_dynamic_sections(
            config,
            &sym_table,
            &mut section_data_map,
            &address_map,
            &referenced_symbols,
            &plt_needed_symbols,
            &copy_symbols,
            diagnostics,
        );

        // Create .bss.copy section for copy-relocated data symbols.
        // This section holds space that the dynamic linker fills at load
        // time via R_*_COPY relocations (e.g. stderr, stdout, stdin).
        let copy_total_size: u64 = copy_symbols.iter().map(|(_, sz)| *sz as u64).sum();
        if copy_total_size > 0 {
            // .bss.copy is NOBITS — we store an empty Vec to represent it;
            // the actual size is tracked via copy_total_size. However, for
            // section_data_map we store zeros so the ELF writer allocates
            // the right amount of space in the file layout.
            section_data_map.insert(".bss.copy".to_string(), vec![0u8; copy_total_size as usize]);
        }

        // Pre-compute virtual addresses for dynamic sections using the same
        // algorithm as write_elf_output. This ensures consistency between
        // the addresses used for relocation patching and the final ELF layout.
        let dynamic_section_names = [
            ".interp",
            ".dynamic",
            ".dynsym",
            ".dynstr",
            ".gnu.hash",
            ".got",
            ".got.plt",
            ".plt",
            ".rela.dyn",
            ".rela.plt",
            ".bss.copy",
        ];

        let mut dyn_vaddr_cursor: u64 = 0;
        let mut dyn_foff_cursor: u64 = 0;
        for info in address_map.section_addresses.values() {
            let sec_end_va = info.virtual_address + info.mem_size;
            let sec_end_fo = info.file_offset + info.size;
            if sec_end_va > dyn_vaddr_cursor {
                dyn_vaddr_cursor = sec_end_va;
            }
            if sec_end_fo > dyn_foff_cursor {
                dyn_foff_cursor = sec_end_fo;
            }
        }
        let page_align = 0x1000u64;
        dyn_vaddr_cursor = (dyn_vaddr_cursor + page_align - 1) & !(page_align - 1);
        dyn_foff_cursor = (dyn_foff_cursor + page_align - 1) & !(page_align - 1);

        let align = if config.target.is_64bit() { 8u64 } else { 4u64 };

        for &sec_name in &dynamic_section_names {
            let already_in_main_layout = ordered_sections.iter().any(|s| s.name == sec_name);
            if already_in_main_layout {
                continue;
            }
            if let Some(data) = section_data_map.get(sec_name) {
                if !data.is_empty() {
                    dyn_vaddr_cursor = (dyn_vaddr_cursor + align - 1) & !(align - 1);
                    dyn_foff_cursor = (dyn_foff_cursor + align - 1) & !(align - 1);
                    let va = dyn_vaddr_cursor;
                    let fo = dyn_foff_cursor;
                    dyn_section_addresses.insert(sec_name.to_string(), (va, fo));
                    dyn_vaddr_cursor += data.len() as u64;
                    dyn_foff_cursor += data.len() as u64;
                }
            }
        }

        // Add PLT symbol entries to the symbol address map so that
        // resolve_relocations can resolve CALL/BL to PLT stubs.
        if let Some(&(plt_vaddr, _plt_foff)) = dyn_section_addresses.get(".plt") {
            // Determine PLT entry sizes for this architecture.
            let (plt0_size, pltn_size): (u64, u64) = match config.target {
                Target::AArch64 => (32, 16),
                Target::RiscV64 => (32, 16),
                Target::X86_64 => (16, 16),
                Target::I686 => (16, 16),
            };

            // Assign PLT addresses to ALL symbols in plt_needed_symbols
            // (which includes both PLT-reloc and non-PLT-reloc undefined
            // symbols). Order: first by relocation encounter order for
            // PLT-type relocs, then alphabetically for the rest. This
            // must match the order used by generate_dynamic_sections.
            let mut plt_index: u64 = 0;
            let mut plt_sym_addrs: Vec<(String, u64)> = Vec::new();
            let mut seen: FxHashSet<String> = FxHashSet::default();

            // First pass: PLT-type relocations (maintains original order).
            for reloc in all_relocations.iter() {
                if handler.needs_plt(reloc.rel_type) && !reloc.symbol_name.is_empty() {
                    if let Some(sym) = symbol_address_map.get(&reloc.symbol_name) {
                        if !sym.is_defined && !seen.contains(&reloc.symbol_name) {
                            seen.insert(reloc.symbol_name.clone());
                            let entry_addr = plt_vaddr + plt0_size + plt_index * pltn_size;
                            plt_sym_addrs.push((reloc.symbol_name.clone(), entry_addr));
                            plt_index += 1;
                        }
                    }
                }
            }

            // Second pass: non-PLT-reloc undefined symbols that still need
            // PLT stubs (e.g. close, read, stdout referenced via R_X86_64_64
            // or R_X86_64_PC32). Order by relocation encounter order.
            for reloc in all_relocations.iter() {
                if !reloc.symbol_name.is_empty()
                    && !seen.contains(&reloc.symbol_name)
                    && plt_needed_symbols.contains(&reloc.symbol_name)
                {
                    if let Some(sym) = symbol_address_map.get(&reloc.symbol_name) {
                        if !sym.is_defined {
                            seen.insert(reloc.symbol_name.clone());
                            let entry_addr = plt_vaddr + plt0_size + plt_index * pltn_size;
                            plt_sym_addrs.push((reloc.symbol_name.clone(), entry_addr));
                            plt_index += 1;
                        }
                    }
                }
            }

            // When generating a synthetic _start, ensure `exit` gets a PLT
            // entry even though no relocation explicitly references it.
            // This lets the _start stub call exit() to flush stdio buffers.
            if config.is_executable() {
                let has_start_sym = sym_table
                    .symbols
                    .iter()
                    .any(|s| s.name == "_start" || s.name == "__start");
                let has_main_sym = sym_table.symbols.iter().any(|s| s.name == "main");
                if !has_start_sym && has_main_sym && !seen.contains("exit") {
                    if let Some(sym) = symbol_address_map.get("exit") {
                        if !sym.is_defined {
                            seen.insert("exit".to_string());
                            let entry_addr = plt_vaddr + plt0_size + plt_index * pltn_size;
                            plt_sym_addrs.push(("exit".to_string(), entry_addr));
                            let _ = plt_index; // used for address computation above
                        }
                    }
                }
            }

            // Update the symbol address map so resolve_relocations sees
            // these PLT-bound symbols as "defined" with PLT stub addresses.
            for (name, addr) in &plt_sym_addrs {
                if let Some(sym) = symbol_address_map.get_mut(name) {
                    sym.final_address = *addr;
                    sym.is_defined = true;
                }
            }
        }

        // Resolve copy-relocated data symbols: assign .bss.copy addresses
        // to the symbol address map and patch the R_*_COPY entries in
        // .rela.dyn with the correct r_offset values.
        if !copy_symbols.is_empty() {
            if let Some(&(bss_copy_va, _bss_copy_fo)) = dyn_section_addresses.get(".bss.copy") {
                let is64 = config.target.is_64bit();
                let mut offset: u64 = 0;
                for (sym_name, sym_size) in &copy_symbols {
                    let addr = bss_copy_va + offset;
                    // Update symbol_address_map so R_X86_64_PC32 / R_X86_64_64
                    // relocations to this data symbol resolve to .bss.copy.
                    if let Some(sym) = symbol_address_map.get_mut(sym_name) {
                        sym.final_address = addr;
                        sym.is_defined = true;
                        sym.section_name = ".bss.copy".to_string();
                    }
                    offset += *sym_size as u64;
                }

                // Patch R_*_COPY entries in .rela.dyn — their r_offset was 0
                // when generated and must now point to .bss.copy VA + offset.
                //
                // IMPORTANT: We must match each COPY relocation to its
                // symbol by looking up the r_info sym_index in .dynsym/
                // .dynstr, NOT by sequential index.  The copy_symbols
                // list is ordered by first encounter in input relocations
                // while the COPY relocs in .rela.dyn follow the
                // imported_symbols / sym_table iteration order — these
                // two orders can diverge for multi-file links with
                // several copy-relocated data symbols (e.g. stderr,
                // stdin, stdout).  Sequential matching causes
                // relocation-address / symbol-address mismatches that
                // make the dynamic linker copy the wrong data, silently
                // swapping stdin and stdout.
                let copy_reloc_type = dynamic::default_copy_reloc(config.target);

                // Build a name→.bss.copy VA map for the copy symbols.
                let mut copy_addr_by_name: FxHashMap<String, u64> = FxHashMap::default();
                {
                    let mut coff: u64 = 0;
                    for (cname, csize) in &copy_symbols {
                        copy_addr_by_name.insert(cname.clone(), bss_copy_va + coff);
                        coff += *csize as u64;
                    }
                }

                // Pre-read .dynsym and .dynstr so we can resolve the
                // sym_index inside each COPY relocation to a symbol name.
                let dynsym_snap = section_data_map.get(".dynsym").cloned();
                let dynstr_snap = section_data_map.get(".dynstr").cloned();

                if let Some(rela_data) = section_data_map.get_mut(".rela.dyn") {
                    let entry_size = if is64 { 24usize } else { 12usize };
                    let sym_entry_size = if is64 { 24usize } else { 16usize };
                    let num_entries = rela_data.len() / entry_size;
                    for i in 0..num_entries {
                        let base = i * entry_size;
                        if is64 {
                            let r_info = u64::from_le_bytes(
                                rela_data[base + 8..base + 16].try_into().unwrap(),
                            );
                            let rtype = (r_info & 0xFFFF_FFFF) as u32;
                            if rtype != copy_reloc_type {
                                continue;
                            }
                            let sym_idx = (r_info >> 32) as usize;
                            // Resolve sym_idx → name via .dynsym/.dynstr
                            let name = resolve_dynsym_name(
                                sym_idx,
                                sym_entry_size,
                                dynsym_snap.as_deref(),
                                dynstr_snap.as_deref(),
                                is64,
                            );
                            if let Some(addr) = name
                                .as_ref()
                                .and_then(|n| copy_addr_by_name.get(n.as_str()))
                            {
                                rela_data[base..base + 8].copy_from_slice(&addr.to_le_bytes());
                            }
                        } else {
                            let r_info = u32::from_le_bytes(
                                rela_data[base + 4..base + 8].try_into().unwrap(),
                            );
                            let rtype = r_info & 0xFF;
                            if rtype != copy_reloc_type {
                                continue;
                            }
                            let sym_idx = (r_info >> 8) as usize;
                            let name = resolve_dynsym_name(
                                sym_idx,
                                sym_entry_size,
                                dynsym_snap.as_deref(),
                                dynstr_snap.as_deref(),
                                is64,
                            );
                            if let Some(addr) = name
                                .as_ref()
                                .and_then(|n| copy_addr_by_name.get(n.as_str()))
                            {
                                let addr32 = *addr as u32;
                                rela_data[base..base + 4].copy_from_slice(&addr32.to_le_bytes());
                            }
                        }
                    }
                }

                // Also patch the .dynsym entries for copy symbols to have the
                // correct st_value (pointing to .bss.copy) and st_size.
                // Clone .dynstr first to avoid borrow conflict.
                let dynstr_data = section_data_map.get(".dynstr").cloned();
                if let Some(dynsym_data) = section_data_map.get_mut(".dynsym") {
                    let sym_entry_size = if is64 { 24usize } else { 16usize };
                    let num_syms = dynsym_data.len() / sym_entry_size;
                    let mut copy_offset_patch: u64 = 0;

                    for (copy_name, copy_size) in &copy_symbols {
                        let target_addr = bss_copy_va + copy_offset_patch;
                        // Find the dynsym entry by matching name via .dynstr
                        for si in 0..num_syms {
                            let base = si * sym_entry_size;
                            if is64 {
                                let st_name = u32::from_le_bytes(
                                    dynsym_data[base..base + 4].try_into().unwrap(),
                                );
                                // Look up name in dynstr
                                if let Some(ref dynstr) = dynstr_data {
                                    let name_start = st_name as usize;
                                    if name_start < dynstr.len() {
                                        let name_end = dynstr[name_start..]
                                            .iter()
                                            .position(|&b| b == 0)
                                            .map(|p| name_start + p)
                                            .unwrap_or(dynstr.len());
                                        let name =
                                            std::str::from_utf8(&dynstr[name_start..name_end])
                                                .unwrap_or("");
                                        if name == copy_name {
                                            // Patch st_value (offset 8 in Elf64_Sym)
                                            dynsym_data[base + 8..base + 16]
                                                .copy_from_slice(&target_addr.to_le_bytes());
                                            // Patch st_size (offset 16 in Elf64_Sym)
                                            dynsym_data[base + 16..base + 24].copy_from_slice(
                                                &(*copy_size as u64).to_le_bytes(),
                                            );
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        copy_offset_patch += *copy_size as u64;
                    }
                }
            }
        }

        // Re-encode PLT and GOT.PLT sections with correct virtual addresses.
        // The initial encoding in generate_dynamic_sections used base=0
        // because addresses weren't known yet. For architectures with
        // PC-relative PLT stubs (AArch64 ADRP, RISC-V AUIPC), we must
        // re-encode with the final addresses.
        if let (Some(&(plt_va, _)), Some(&(got_plt_va, _))) = (
            dyn_section_addresses.get(".plt"),
            dyn_section_addresses.get(".got.plt"),
        ) {
            // Re-build the PLT section bytes with correct addresses.
            // We need the PltTable and stubs — reconstruct from the .rela.plt
            // section data and the imported symbol list.
            let plt_table = dynamic::ProcedureLinkageTable::new_public(config.target);
            let mut plt_buf = plt_table.generate_plt0(got_plt_va, plt_va);

            // Determine PLT entry sizes for re-encoding.
            let (plt0_sz, pltn_sz): (u64, u64) = match config.target {
                Target::AArch64 | Target::RiscV64 => (32, 16),
                Target::X86_64 | Target::I686 => (16, 16),
            };

            // Reconstruct stubs from the symbols that were assigned PLT addresses.
            // We need (symbol_name, got_plt_offset, index) for each PLT entry.
            // CRITICAL: We also regenerate .rela.plt in this same loop to
            // guarantee JUMP_SLOT entries are in the same order as PLT stubs.
            // The original .rela.plt from build_dynamic_sections may have a
            // different symbol ordering which would cause PLT stubs to load
            // from the wrong GOT.PLT slots.
            let pw = if config.target.is_64bit() { 8u64 } else { 4u64 };
            let is64 = config.target.is_64bit();
            let js_reltype = dynamic::default_jump_slot_reloc(config.target);
            let reserved = dynamic::got_plt_reserved_count(&config.target) as u64;
            let mut stub_idx: u32 = 0;
            let mut seen_stubs: FxHashSet<String> = FxHashSet::default();
            // Collect ordered stub info for .rela.plt regeneration.
            let mut ordered_plt_syms: Vec<(String, u64)> = Vec::new();
            for reloc in all_relocations.iter() {
                if handler.needs_plt(reloc.rel_type) && !reloc.symbol_name.is_empty() {
                    if let Some(sym) = symbol_address_map.get(&reloc.symbol_name) {
                        if !seen_stubs.contains(&reloc.symbol_name) {
                            let expected_addr = plt_va + plt0_sz + stub_idx as u64 * pltn_sz;
                            if sym.final_address == expected_addr {
                                seen_stubs.insert(reloc.symbol_name.clone());
                                let gp_off = (reserved + stub_idx as u64) * pw;
                                let stub = dynamic::PltStub {
                                    symbol_name: reloc.symbol_name.clone(),
                                    got_plt_offset: gp_off,
                                    index: stub_idx,
                                };
                                let off = plt0_sz + stub_idx as u64 * pltn_sz;
                                plt_buf.extend_from_slice(
                                    &plt_table.generate_plt_entry(&stub, got_plt_va, plt_va, off),
                                );
                                ordered_plt_syms.push((reloc.symbol_name.clone(), gp_off));
                                stub_idx += 1;
                            }
                        }
                    }
                }
            }

            // Second pass: add PLT stubs for non-PLT-reloc undefined symbols
            // (e.g. close, read, stdout referenced via R_X86_64_64 / PC32).
            for reloc in all_relocations.iter() {
                if !reloc.symbol_name.is_empty()
                    && !seen_stubs.contains(&reloc.symbol_name)
                    && plt_needed_symbols.contains(&reloc.symbol_name)
                {
                    if let Some(sym) = symbol_address_map.get(&reloc.symbol_name) {
                        let expected_addr = plt_va + plt0_sz + stub_idx as u64 * pltn_sz;
                        if sym.final_address == expected_addr {
                            seen_stubs.insert(reloc.symbol_name.clone());
                            let gp_off = (reserved + stub_idx as u64) * pw;
                            let stub = dynamic::PltStub {
                                symbol_name: reloc.symbol_name.clone(),
                                got_plt_offset: gp_off,
                                index: stub_idx,
                            };
                            let off = plt0_sz + stub_idx as u64 * pltn_sz;
                            plt_buf.extend_from_slice(
                                &plt_table.generate_plt_entry(&stub, got_plt_va, plt_va, off),
                            );
                            ordered_plt_syms.push((reloc.symbol_name.clone(), gp_off));
                            stub_idx += 1;
                        }
                    }
                }
            }

            // Add PLT stubs for synthetic imports (e.g. `exit` injected for
            // the synthetic _start stub, `__cxa_atexit` for the synthetic
            // atexit wrapper) that were assigned PLT addresses in the first
            // PLT assignment pass but were NOT covered by the relocation-based
            // loop above.  Without this, the re-encoded .plt/.got.plt will
            // have fewer entries than the initial layout, causing an address
            // mismatch that shifts all GOT.PLT lazy-binding targets.
            let mut synthetic_imports: Vec<String> = Vec::new();
            if need_synthetic_start {
                synthetic_imports.push("exit".to_string());
            }
            if need_synthetic_atexit {
                synthetic_imports.push("__cxa_atexit".to_string());
            }
            for name in synthetic_imports {
                if !seen_stubs.contains(&name) && symbol_address_map.contains_key(&name) {
                    seen_stubs.insert(name.clone());
                    let gp_off = (reserved + stub_idx as u64) * pw;
                    let stub = dynamic::PltStub {
                        symbol_name: name.clone(),
                        got_plt_offset: gp_off,
                        index: stub_idx,
                    };
                    let off = plt0_sz + stub_idx as u64 * pltn_sz;
                    plt_buf.extend_from_slice(
                        &plt_table.generate_plt_entry(&stub, got_plt_va, plt_va, off),
                    );
                    // Update the symbol address to the re-encoded
                    // PLT position (may differ from the original
                    // assignment in build_dynamic_sections).
                    let actual_addr = plt_va + plt0_sz + stub_idx as u64 * pltn_sz;
                    if let Some(sym) = symbol_address_map.get_mut(&name) {
                        sym.final_address = actual_addr;
                    }
                    ordered_plt_syms.push((name, gp_off));
                    stub_idx += 1;
                }
            }

            if !plt_buf.is_empty() {
                section_data_map.insert(".plt".to_string(), plt_buf);
            }

            // Re-encode .got.plt with correct lazy-binding targets.
            let num_stubs = stub_idx as usize;
            let total_entries = reserved as usize + num_stubs;
            let mut gotplt_buf = Vec::with_capacity(total_entries * pw as usize);
            // Reserved entries (filled by dynamic linker at load time).
            for _ in 0..reserved {
                if pw == 8 {
                    gotplt_buf.extend_from_slice(&0u64.to_le_bytes());
                } else {
                    gotplt_buf.extend_from_slice(&0u32.to_le_bytes());
                }
            }
            // Per-stub entries: initially point to the push instruction
            // inside each PLT entry for lazy resolution, so the dynamic
            // linker can identify which symbol to resolve.
            let (plt0_sz, pltn_sz) = dynamic::plt_sizes(&config.target);
            for i in 0..num_stubs {
                let plt_entry_va = plt_va + plt0_sz as u64 + i as u64 * pltn_sz as u64;
                let lazy_target = match config.target {
                    Target::AArch64 | Target::RiscV64 => plt_va,
                    Target::X86_64 | Target::I686 => plt_entry_va + 6,
                };
                if pw == 8 {
                    gotplt_buf.extend_from_slice(&lazy_target.to_le_bytes());
                } else {
                    gotplt_buf.extend_from_slice(&(lazy_target as u32).to_le_bytes());
                }
            }
            section_data_map.insert(".got.plt".to_string(), gotplt_buf);

            // Regenerate .rela.plt so JUMP_SLOT entries match PLT stub order.
            // Look up each symbol's dynsym index from the existing .dynsym.
            // If we can't look up indices, fall back to sequential 1-based.
            let mut rela_plt_data: Vec<u8> = Vec::new();
            for (i, (sym_name, gp_off)) in ordered_plt_syms.iter().enumerate() {
                // GOT.PLT slot VA = got_plt_va + gp_off (absolute, pre-patching
                // happens below so store section-relative here — the offset
                // patching pass below will add got_plt_va).
                let r_offset = *gp_off;
                // dynsym index: use 1-based sequential order matching the
                // order symbols appear in the .dynsym we generated.
                // The .dynsym from build_dynamic_sections may have a different
                // order, so we look it up from the section data.
                let sym_idx = dynsym_index_for_name(
                    section_data_map.get(".dynsym"),
                    section_data_map.get(".dynstr"),
                    sym_name,
                    is64,
                )
                .unwrap_or((i + 1) as u32);
                if is64 {
                    let r_info = ((sym_idx as u64) << 32) | (js_reltype as u64);
                    rela_plt_data.extend_from_slice(&r_offset.to_le_bytes());
                    rela_plt_data.extend_from_slice(&r_info.to_le_bytes());
                    rela_plt_data.extend_from_slice(&0i64.to_le_bytes()); // addend
                } else {
                    // i686: Elf32_Rel format (8 bytes, no addend)
                    let r_info = (sym_idx << 8) | (js_reltype & 0xFF);
                    rela_plt_data.extend_from_slice(&(r_offset as u32).to_le_bytes());
                    rela_plt_data.extend_from_slice(&r_info.to_le_bytes());
                }
            }
            if !rela_plt_data.is_empty() {
                section_data_map.insert(".rela.plt".to_string(), rela_plt_data);
            }
        }

        // Patch .rela.dyn and .rela.plt r_offset fields from section-relative
        // to absolute virtual addresses. The dynamic linker expects r_offset
        // to be the VA of the GOT/GOT.PLT slot, not a byte offset within the
        // section.
        let entry_size: usize = if config.target.is_64bit() { 24 } else { 8 };

        let copy_reloc_type_val = dynamic::default_copy_reloc(config.target);
        if let Some(&(got_va, _)) = dyn_section_addresses.get(".got") {
            if let Some(rela_dyn_data) = section_data_map.get_mut(".rela.dyn") {
                let num_entries = rela_dyn_data.len() / entry_size;
                for i in 0..num_entries {
                    let base = i * entry_size;
                    if config.target.is_64bit() {
                        // Skip R_*_COPY entries — they already have absolute
                        // .bss.copy VA addresses set by the copy-relocation
                        // patching pass above.
                        let r_info = u64::from_le_bytes(
                            rela_dyn_data[base + 8..base + 16].try_into().unwrap(),
                        );
                        let rtype = (r_info & 0xFFFF_FFFF) as u32;
                        if rtype == copy_reloc_type_val {
                            continue;
                        }
                        let old_off =
                            u64::from_le_bytes(rela_dyn_data[base..base + 8].try_into().unwrap());
                        let new_off = old_off + got_va;
                        rela_dyn_data[base..base + 8].copy_from_slice(&new_off.to_le_bytes());
                    } else {
                        let r_info = u32::from_le_bytes(
                            rela_dyn_data[base + 4..base + 8].try_into().unwrap(),
                        );
                        let rtype = r_info & 0xFF;
                        if rtype == copy_reloc_type_val {
                            continue;
                        }
                        let old_off =
                            u32::from_le_bytes(rela_dyn_data[base..base + 4].try_into().unwrap());
                        let new_off = old_off + got_va as u32;
                        rela_dyn_data[base..base + 4].copy_from_slice(&new_off.to_le_bytes());
                    }
                }
            }
        }

        if let Some(&(got_plt_va, _)) = dyn_section_addresses.get(".got.plt") {
            if let Some(rela_plt_data) = section_data_map.get_mut(".rela.plt") {
                let num_entries = rela_plt_data.len() / entry_size;
                for i in 0..num_entries {
                    let base = i * entry_size;
                    if config.target.is_64bit() {
                        let old_off =
                            u64::from_le_bytes(rela_plt_data[base..base + 8].try_into().unwrap());
                        let new_off = old_off + got_plt_va;
                        rela_plt_data[base..base + 8].copy_from_slice(&new_off.to_le_bytes());
                    } else {
                        let old_off =
                            u32::from_le_bytes(rela_plt_data[base..base + 4].try_into().unwrap());
                        let new_off = old_off + got_plt_va as u32;
                        rela_plt_data[base..base + 4].copy_from_slice(&new_off.to_le_bytes());
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 5.9: Build per-symbol GOT address map
    // -----------------------------------------------------------------------
    // Scan relocations for GOT-needing symbols and assign sequential GOT
    // slot addresses using the .got VA computed in Phase 5.5.  This map
    // is consumed by the relocation processor so that GOT-relative
    // relocations (e.g. R_AARCH64_ADR_GOT_PAGE, R_X86_64_GOTPCRELX) can
    // resolve to the correct GOT entry address.
    // Build GOT section if relocations need it.
    // In PIC shared libraries, even locally-defined symbols (e.g. global
    // variables, string literals) are accessed through the GOT because the
    // shared library's load address is not known at link time. The
    // build_dynamic_sections only creates GOT entries for imported symbols.
    // Here we scan all relocations, create GOT entries for any symbol with
    // a GOT-needing relocation, and ensure the .got section exists.
    let mut precomputed_got: FxHashMap<String, u64> = FxHashMap::default();
    {
        let pw = if config.target.is_64bit() { 8u64 } else { 4u64 };
        let mut got_syms_ordered: Vec<String> = Vec::new();
        let mut seen_got: FxHashSet<String> = FxHashSet::default();
        for reloc in all_relocations.iter() {
            if handler.needs_got(reloc.rel_type)
                && !reloc.symbol_name.is_empty()
                && !seen_got.contains(&reloc.symbol_name)
            {
                seen_got.insert(reloc.symbol_name.clone());
                got_syms_ordered.push(reloc.symbol_name.clone());
            }
        }

        if !got_syms_ordered.is_empty() {
            // Ensure a .got section exists with the right number of slots.
            let existing_got_size = section_data_map.get(".got").map(|d| d.len()).unwrap_or(0);
            let needed_size = got_syms_ordered.len() * pw as usize;
            if needed_size > existing_got_size {
                // Create or extend the .got section with zero-filled slots.
                // The dynamic linker (or our static initialization below)
                // fills these at load time.
                section_data_map.insert(".got".to_string(), vec![0u8; needed_size]);
            }

            // Ensure .got has a VA in dyn_section_addresses.
            if !dyn_section_addresses.contains_key(".got") {
                // Compute VA for .got following the existing dynamic sections.
                let align = if config.target.is_64bit() { 8u64 } else { 4u64 };
                let mut cursor_va: u64 = 0;
                let mut cursor_fo: u64 = 0;
                for info in address_map.section_addresses.values() {
                    let end_va = info.virtual_address + info.mem_size;
                    let end_fo = info.file_offset + info.size;
                    if end_va > cursor_va {
                        cursor_va = end_va;
                    }
                    if end_fo > cursor_fo {
                        cursor_fo = end_fo;
                    }
                }
                for (_, &(va, fo)) in dyn_section_addresses.iter() {
                    let data_len = section_data_map
                        .values()
                        .map(|d| d.len() as u64)
                        .max()
                        .unwrap_or(0);
                    let end_va = va + data_len;
                    let end_fo = fo + data_len;
                    if end_va > cursor_va {
                        cursor_va = end_va;
                    }
                    if end_fo > cursor_fo {
                        cursor_fo = end_fo;
                    }
                }
                // Re-walk dyn_section_addresses to find the true max end
                for (sec_name, &(va, fo)) in dyn_section_addresses.iter() {
                    if let Some(data) = section_data_map.get(sec_name.as_str()) {
                        let end_va = va + data.len() as u64;
                        let end_fo = fo + data.len() as u64;
                        if end_va > cursor_va {
                            cursor_va = end_va;
                        }
                        if end_fo > cursor_fo {
                            cursor_fo = end_fo;
                        }
                    }
                }
                cursor_va = (cursor_va + align - 1) & !(align - 1);
                cursor_fo = (cursor_fo + align - 1) & !(align - 1);
                dyn_section_addresses.insert(".got".to_string(), (cursor_va, cursor_fo));
            }

            // Now compute per-symbol GOT addresses.
            if let Some(&(got_va, _)) = dyn_section_addresses.get(".got") {
                for (idx, sym_name) in got_syms_ordered.iter().enumerate() {
                    let slot_addr = got_va + idx as u64 * pw;
                    precomputed_got.insert(sym_name.clone(), slot_addr);
                }

                // Pre-fill GOT entries for defined symbols with their
                // final addresses. This is needed for non-lazy binding of
                // locally-defined symbols in PIC shared libraries.
                if let Some(got_data) = section_data_map.get_mut(".got") {
                    for (idx, sym_name) in got_syms_ordered.iter().enumerate() {
                        if let Some(sym) = symbol_address_map.get(sym_name) {
                            if sym.is_defined {
                                let offset = idx * pw as usize;
                                if offset + pw as usize <= got_data.len() {
                                    if pw == 8 {
                                        got_data[offset..offset + 8]
                                            .copy_from_slice(&sym.final_address.to_le_bytes());
                                    } else {
                                        got_data[offset..offset + 4].copy_from_slice(
                                            &(sym.final_address as u32).to_le_bytes(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 6: Relocation Processing
    // -----------------------------------------------------------------------
    // Now that PLT addresses are available in symbol_address_map, CALL/BL
    // relocations to external symbols can be resolved.
    if !all_relocations.is_empty() {
        let _reloc_result = relocation::process_relocations(
            &config.target,
            all_relocations,
            &symbol_address_map,
            &address_map,
            &mut section_data_map,
            handler,
            diagnostics,
            &precomputed_got,
        );
    }

    if diagnostics.has_errors() && !config.allow_undefined {
        // Print all diagnostic messages so the user can see which relocations failed.
        for diag in diagnostics.diagnostics() {
            eprintln!("  linker diagnostic: {}", diag.message);
        }
        return Err("linking aborted due to relocation errors".to_string());
    }

    // -----------------------------------------------------------------------
    // Phase 7: Entry Point Resolution (with synthetic _start fallback)
    // -----------------------------------------------------------------------
    let entry_point = if config.is_executable() {
        // Build a simple symbol name → address map for the entry point lookup.
        // Includes both defined symbols and PLT-resolved external symbols.
        let mut sym_addr_flat: FxHashMap<String, u64> = FxHashMap::default();
        for sym in &sym_table.symbols {
            sym_addr_flat.insert(sym.name.clone(), sym.value);
        }
        // Overlay PLT addresses from symbol_address_map so synthetic _start
        // can resolve `exit` (and other PLT-bound imports).
        for (name, resolved) in &symbol_address_map {
            if resolved.is_defined && resolved.final_address != 0 {
                sym_addr_flat.insert(name.clone(), resolved.final_address);
            }
        }

        match resolve_entry_point(&config.entry_point, &sym_addr_flat) {
            Ok(addr) => addr,
            Err(_no_start) => {
                // `_start` is not defined among the input objects.  If `main`
                // IS defined, synthesise a minimal C-runtime stub that:
                //   1. Extracts argc/argv from the Linux kernel ABI stack
                //   2. Calls `main(argc, argv)`
                //   3. Passes main's return value to exit_group(2)
                //
                // This lets `bcc foo.o bar.o -o prog` "just work" the same
                // way that `gcc foo.o bar.o -o prog` does.
                if let Some(&main_addr) = sym_addr_flat.get("main") {
                    // Append the synthetic stub to the .text section data.
                    // The stub's virtual address = .text vaddr + current .text size.
                    let text_vaddr = address_map
                        .section_addresses
                        .get(".text")
                        .map(|a| a.virtual_address)
                        .unwrap_or(0x401000);
                    let text_size = section_data_map
                        .get(".text")
                        .map(|d| d.len() as u64)
                        .unwrap_or(0);

                    // Align stub to 16 bytes within .text.
                    let padding = ((16 - (text_size % 16)) % 16) as usize;
                    let stub_offset = text_size + padding as u64;
                    let stub_vaddr = text_vaddr + stub_offset;

                    // If `exit` is available (via PLT from libc), use it
                    // instead of raw syscall so stdio buffers get flushed.
                    let exit_addr = sym_addr_flat.get("exit").copied();

                    let stub_code =
                        generate_synthetic_start(config.target, main_addr, stub_vaddr, exit_addr);
                    let stub_len = stub_code.len() as u64;

                    // Extend .text section data with alignment padding and stub.
                    if let Some(text_data) = section_data_map.get_mut(".text") {
                        text_data.extend(std::iter::repeat(0xCC).take(padding)); // INT3 fill
                        text_data.extend_from_slice(&stub_code);
                    } else {
                        // No .text section yet — create one with just the stub.
                        let mut data = vec![0xCC; padding];
                        data.extend_from_slice(&stub_code);
                        section_data_map.insert(".text".to_string(), data);
                    }

                    // Update .text section size in address map.
                    if let Some(text_addr) = address_map.section_addresses.get_mut(".text") {
                        text_addr.size = stub_offset + stub_len;
                        text_addr.mem_size = stub_offset + stub_len;
                    }

                    // Register `_start` in the symbol table.
                    sym_table.symbols.push(symbol_resolver::OutputSymbol {
                        name: "_start".to_string(),
                        value: stub_vaddr,
                        size: stub_len,
                        binding: 1,    // STB_GLOBAL
                        sym_type: 2,   // STT_FUNC
                        visibility: 0, // STV_DEFAULT
                        section_index: symbol_resolver::SHN_ABS,
                    });

                    stub_vaddr
                } else {
                    // Neither _start nor main — fatal error.
                    let err_msg =
                        "linker error: neither '_start' nor 'main' symbol is defined".to_string();
                    diagnostics.emit_error(Span::dummy(), err_msg.clone());
                    return Err(err_msg);
                }
            }
        }
    } else {
        0
    };

    // -----------------------------------------------------------------------
    // Phase 7b: Patch synthetic atexit stub (glibc ≥ 2.34 compatibility)
    // -----------------------------------------------------------------------
    // The placeholder was injected in Phase 5.4 with __cxa_atexit address = 0.
    // Now that PLT stubs are built and relocated, patch in the real address.
    if need_synthetic_atexit {
        // Build updated address map including PLT stubs.
        let mut sa: FxHashMap<String, u64> = FxHashMap::default();
        for sym in sym_table.symbols.iter() {
            sa.insert(sym.name.clone(), sym.value);
        }
        for (name, resolved) in &symbol_address_map {
            // Include both defined symbols AND undefined symbols that have
            // a non-zero final_address (i.e. they have a PLT stub assigned).
            // This is critical for __cxa_atexit which is an imported symbol
            // resolved through PLT — its PLT stub address must be used when
            // patching the synthetic atexit wrapper.
            if resolved.final_address != 0 {
                sa.insert(name.clone(), resolved.final_address);
            }
        }

        let cxa_addr = sa.get("__cxa_atexit").copied();
        let atexit_addr = sa.get("atexit").copied();

        if let (Some(cxa_target), Some(atexit_vaddr)) = (cxa_addr, atexit_addr) {
            let text_vaddr = address_map
                .section_addresses
                .get(".text")
                .map(|a| a.virtual_address)
                .unwrap_or(0x401000);

            // Re-generate the stub with the real __cxa_atexit address.
            let patched_code = generate_synthetic_atexit(config.target, atexit_vaddr, cxa_target);

            // Locate the stub within .text and overwrite the placeholder.
            if let Some(text_data) = section_data_map.get_mut(".text") {
                let offset_in_text = (atexit_vaddr - text_vaddr) as usize;
                if offset_in_text + patched_code.len() <= text_data.len() {
                    text_data[offset_in_text..offset_in_text + patched_code.len()]
                        .copy_from_slice(&patched_code);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 8: ELF Output Writing
    // -----------------------------------------------------------------------
    let elf_data = write_elf_output(
        config,
        &ordered_sections,
        &section_data_map,
        &address_map,
        &sym_table,
        entry_point,
    );

    Ok(LinkerOutput {
        elf_data,
        entry_point,
        output_type: config.output_type,
    })
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

/// Look up a symbol's index in an encoded `.dynsym` section by searching the
/// companion `.dynstr` for the symbol's name.
///
/// Returns `None` if the symbol is not found or the section data is missing.
fn dynsym_index_for_name(
    dynsym_data: Option<&Vec<u8>>,
    dynstr_data: Option<&Vec<u8>>,
    name: &str,
    is_64bit: bool,
) -> Option<u32> {
    let dynsym = dynsym_data?;
    let dynstr = dynstr_data?;
    let entry_size: usize = if is_64bit { 24 } else { 16 };
    if dynsym.is_empty() || entry_size == 0 {
        return None;
    }
    let num_entries = dynsym.len() / entry_size;
    for i in 0..num_entries {
        let base = i * entry_size;
        // st_name is always a 32-bit offset in both ELF32 and ELF64 symbol entries.
        let _ = is_64bit;
        let st_name = u32::from_le_bytes(dynsym[base..base + 4].try_into().ok()?) as usize;
        if st_name < dynstr.len() {
            // Read null-terminated string from dynstr.
            let end = dynstr[st_name..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| st_name + p)
                .unwrap_or(dynstr.len());
            if &dynstr[st_name..end] == name.as_bytes() {
                return Some(i as u32);
            }
        }
    }
    None
}

/// Handle relocatable output (passthrough — no actual linking).
///
/// For `OutputType::Relocatable`, we simply concatenate input sections into
/// a single relocatable ELF object file without performing symbol resolution,
/// relocation processing, or address assignment.
fn link_relocatable(
    config: &LinkerConfig,
    inputs: Vec<LinkerInput>,
    diagnostics: &mut DiagnosticEngine,
) -> Result<LinkerOutput, String> {
    let elf_type = crate::backend::elf_writer_common::ET_REL;
    let mut writer = ElfWriter::new(config.target, elf_type);

    // Add all input sections to the output.
    for input in &inputs {
        for section in &input.sections {
            let elf_section = Section {
                name: section.name.clone(),
                sh_type: section.section_type,
                sh_flags: section.flags,
                data: section.data.clone(),
                sh_link: 0,
                sh_info: 0,
                sh_addralign: section.alignment,
                sh_entsize: 0,
                logical_size: 0,
                virtual_address: 0,
                file_offset_hint: 0,
            };
            writer.add_section(elf_section);
        }

        // Add symbols from this input.
        for sym in &input.symbols {
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
    }

    let elf_data = writer.write();

    // Emit a note if no inputs were provided.
    if inputs.is_empty() {
        diagnostics.emit(Diagnostic::warning(
            Span::dummy(),
            "no input files provided for relocatable output",
        ));
    }

    Ok(LinkerOutput {
        elf_data,
        entry_point: 0,
        output_type: OutputType::Relocatable,
    })
}

/// Resolve the entry point symbol to a virtual address.
///
/// Returns `Ok(address)` if the entry point symbol is found, or `Err`
/// with a descriptive message if the symbol is undefined.
fn resolve_entry_point(entry_name: &str, symbols: &FxHashMap<String, u64>) -> Result<u64, String> {
    match symbols.get(entry_name) {
        Some(&addr) => Ok(addr),
        None => Err(format!(
            "entry point symbol '{}' is undefined; \
             cannot produce executable without an entry point",
            entry_name
        )),
    }
}

/// Generate architecture-specific machine code for a minimal synthetic
/// `_start` entry point that calls `main(argc, argv)` and then terminates
/// the process via the `exit_group` syscall.
///
/// Linux kernel ABI guarantees the following stack layout at process entry:
///
/// ```text
///   (%rsp)       → argc          (x86-64 / i686: top of stack)
///   8(%rsp)      → argv[0]       (subsequent entries follow)
/// ```
///
/// The generated stub is a self-contained blob of machine code with all
/// addresses resolved — no relocations are emitted.
fn generate_synthetic_start(
    target: Target,
    main_addr: u64,
    _stub_vaddr: u64,
    exit_addr: Option<u64>,
) -> Vec<u8> {
    match target {
        Target::X86_64 => {
            // xor  %ebp, %ebp        ; ABI: zero the frame pointer
            // pop  %rdi               ; argc  (first arg for main)
            // mov  %rsp, %rsi         ; argv  (second arg for main)
            // and  $-16, %rsp         ; align stack to 16 bytes
            // movabs $main, %rax      ; load main address
            // call *%rax              ; main(argc, argv)
            // mov  %eax, %edi         ; exit code = main return value
            //
            // If `exit` symbol is available (via PLT/libc), call it so that
            // stdio buffers (stdout, stderr) are flushed before process
            // termination.  Otherwise fall back to the raw exit_group
            // syscall which is safe but skips libc atexit handlers.
            let mut code = Vec::with_capacity(48);
            code.extend_from_slice(&[0x31, 0xed]); // xor %ebp, %ebp
            code.push(0x5f); // pop %rdi
            code.extend_from_slice(&[0x48, 0x89, 0xe6]); // mov %rsp, %rsi
            code.extend_from_slice(&[0x48, 0x83, 0xe4, 0xf0]); // and $-16, %rsp
            code.extend_from_slice(&[0x48, 0xb8]); // movabs $imm64, %rax
            code.extend_from_slice(&main_addr.to_le_bytes()); // imm64 = main
            code.extend_from_slice(&[0xff, 0xd0]); // call *%rax
            code.extend_from_slice(&[0x89, 0xc7]); // mov %eax, %edi
            if let Some(eaddr) = exit_addr {
                // movabs $exit_addr, %rax
                code.extend_from_slice(&[0x48, 0xb8]);
                code.extend_from_slice(&eaddr.to_le_bytes());
                // call *%rax  (calls exit() which flushes stdio)
                code.extend_from_slice(&[0xff, 0xd0]);
            }
            // Always emit the raw syscall as a fallback — if exit()
            // returns (it shouldn't) we still terminate the process.
            code.extend_from_slice(&[0xb8, 0xe7, 0x00, 0x00, 0x00]); // mov $231, %eax
            code.extend_from_slice(&[0x0f, 0x05]); // syscall
            code
        }
        Target::I686 => {
            // xor  %ebp, %ebp        ; ABI: zero the frame pointer
            // pop  %esi               ; argc (save for later)
            // mov  %esp, %ecx         ; argv
            // and  $-16, %esp         ; align stack to 16 bytes
            // push %ecx               ; argv  (arg 2)
            // push %esi               ; argc  (arg 1)
            // mov  $MAIN, %eax        ; load main address
            // call *%eax              ; main(argc, argv)
            //
            // After main returns, call exit() if available (flushes stdio),
            // otherwise fall back to raw exit_group syscall.
            let mut code = Vec::with_capacity(40);
            code.extend_from_slice(&[0x31, 0xed]); // xor %ebp, %ebp
            code.push(0x5e); // pop %esi
            code.extend_from_slice(&[0x89, 0xe1]); // mov %esp, %ecx
            code.extend_from_slice(&[0x83, 0xe4, 0xf0]); // and $-16, %esp
            code.push(0x51); // push %ecx  (argv)
            code.push(0x56); // push %esi  (argc)
            code.extend_from_slice(&[0xb8]); // mov $imm32, %eax
            code.extend_from_slice(&(main_addr as u32).to_le_bytes()); // imm32 = main
            code.extend_from_slice(&[0xff, 0xd0]); // call *%eax
            if let Some(eaddr) = exit_addr {
                // push %eax           ; exit code as arg on stack
                code.push(0x50);
                // mov $exit_addr, %eax
                code.extend_from_slice(&[0xb8]);
                code.extend_from_slice(&(eaddr as u32).to_le_bytes());
                // call *%eax          ; exit(main_return_value)
                code.extend_from_slice(&[0xff, 0xd0]);
            }
            // Fallback raw syscall
            code.extend_from_slice(&[0x89, 0xc3]); // mov %eax, %ebx
            code.extend_from_slice(&[0xb8, 0xfc, 0x00, 0x00, 0x00]); // mov $252, %eax
            code.extend_from_slice(&[0xcd, 0x80]); // int $0x80
            code
        }
        Target::AArch64 => {
            // AArch64 Linux kernel entry: x0 = 0, sp points to argc.
            // ldr  x0, [sp]           ; argc
            // add  x1, sp, #8         ; argv
            // and  sp, sp, #-16       ; align stack
            // ldr  x16, =main         ; load main address (literal pool)
            // blr  x16                ; call main(argc, argv)
            //
            // If exit_addr available: call exit(retval), else raw syscall.
            let mut code = Vec::with_capacity(64);
            // ldr x0, [sp]  → 0xF94003E0
            code.extend_from_slice(&0xF94003E0u32.to_le_bytes());
            // add x1, sp, #8 → 0x910023E1
            code.extend_from_slice(&0x910023E1u32.to_le_bytes());
            // and sp, sp, #-16 → 0x927CEBFF
            code.extend_from_slice(&0x927CEBFFu32.to_le_bytes());

            if let Some(eaddr) = exit_addr {
                // --- With exit() via PLT ---
                // ldr x16, .+28 → literal pool at offset 28 from this instruction
                // (7 more instructions * 4 = 28 bytes to main_addr pool entry)
                let ldr_main: u32 = 0x58000000 | (16) | ((7 & 0x7FFFF) << 5);
                code.extend_from_slice(&ldr_main.to_le_bytes());
                // blr x16  → 0xD63F0200
                code.extend_from_slice(&0xD63F0200u32.to_le_bytes());
                // mov x0, x0 — exit code is already in x0 from main return
                // ldr x16, .+16 → literal pool for exit_addr (4 more insns * 4 = 16)
                let ldr_exit: u32 = 0x58000000 | (16) | ((4 & 0x7FFFF) << 5);
                code.extend_from_slice(&ldr_exit.to_le_bytes());
                // blr x16  → call exit()
                code.extend_from_slice(&0xD63F0200u32.to_le_bytes());
                // Fallback syscall in case exit() somehow returns
                // mov x8, #94 → movz x8, #94 → 0xD2800BC8
                code.extend_from_slice(&0xD2800BC8u32.to_le_bytes());
                // svc #0 → 0xD4000001
                code.extend_from_slice(&0xD4000001u32.to_le_bytes());
                // Literal pool (8 bytes each, aligned)
                code.extend_from_slice(&main_addr.to_le_bytes());
                code.extend_from_slice(&eaddr.to_le_bytes());
            } else {
                // --- Raw syscall fallback (no exit() available) ---
                // ldr x16, .+20 (skip 5 instructions to literal pool)
                code.extend_from_slice(&0x580000B0u32.to_le_bytes());
                // blr x16 → 0xD63F0200
                code.extend_from_slice(&0xD63F0200u32.to_le_bytes());
                // mov x8, #94 → movz x8, #94 → 0xD2800BC8
                code.extend_from_slice(&0xD2800BC8u32.to_le_bytes());
                // svc #0 → 0xD4000001
                code.extend_from_slice(&0xD4000001u32.to_le_bytes());
                // nop for alignment → 0xD503201F
                code.extend_from_slice(&0xD503201Fu32.to_le_bytes());
                // Literal pool: 8-byte address of main
                code.extend_from_slice(&main_addr.to_le_bytes());
            }
            code
        }
        Target::RiscV64 => {
            // RISC-V Linux entry: sp points to argc.
            // ld   a0, 0(sp)          ; argc
            // addi a1, sp, 8          ; argv
            // andi sp, sp, -16        ; align stack
            // Load main from literal pool via auipc+ld, call main.
            // Then call exit() via PLT if available, else raw ecall.
            let mut code = Vec::with_capacity(64);
            // ld a0, 0(sp)
            code.extend_from_slice(&0x00013503u32.to_le_bytes());
            // addi a1, sp, 8
            code.extend_from_slice(&0x00810593u32.to_le_bytes());
            // andi sp, sp, -16
            code.extend_from_slice(&0xFF017113u32.to_le_bytes());

            if let Some(eaddr) = exit_addr {
                // With exit() available:
                // auipc t0, 0   → get PC
                code.extend_from_slice(&0x00000297u32.to_le_bytes());
                // ld t0, 28(t0) → load main_addr from literal pool
                // offset 28 = 7 more instructions * 4 bytes
                let ld_main: u32 = (28 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | 0x03;
                code.extend_from_slice(&ld_main.to_le_bytes());
                // jalr ra, t0, 0  → call main
                code.extend_from_slice(&0x000280E7u32.to_le_bytes());
                // a0 already has return value from main
                // auipc t0, 0   → get PC for exit literal pool load
                code.extend_from_slice(&0x00000297u32.to_le_bytes());
                // ld t0, 20(t0) → load exit_addr (5 more insns * 4 = 20)
                let ld_exit: u32 = (20 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | 0x03;
                code.extend_from_slice(&ld_exit.to_le_bytes());
                // jalr ra, t0, 0 → call exit()
                code.extend_from_slice(&0x000280E7u32.to_le_bytes());
                // Fallback raw syscall
                // addi a7, x0, 94 — RISC-V I-type: imm[11:0]|rs1|funct3|rd|opcode
                let li_a7: u32 = (94 << 20) | (17 << 7) | 0x13; // rs1=x0, funct3=0
                code.extend_from_slice(&li_a7.to_le_bytes());
                // ecall
                code.extend_from_slice(&0x00000073u32.to_le_bytes());
                // Literal pool (8 bytes each)
                code.extend_from_slice(&main_addr.to_le_bytes());
                code.extend_from_slice(&eaddr.to_le_bytes());
            } else {
                // No exit() — use raw syscall only
                // auipc t0, 0
                code.extend_from_slice(&0x00000297u32.to_le_bytes());
                // ld t0, 16(t0) → offset to literal pool
                let ld_inst: u32 = (16 << 20) | (5 << 15) | (3 << 12) | (5 << 7) | 0x03;
                code.extend_from_slice(&ld_inst.to_le_bytes());
                // jalr ra, t0, 0
                code.extend_from_slice(&0x000280E7u32.to_le_bytes());
                // addi a7, x0, 94 — RISC-V I-type: imm[11:0]|rs1|funct3|rd|opcode
                let li_a7: u32 = (94 << 20) | (17 << 7) | 0x13; // rs1=x0, funct3=0
                code.extend_from_slice(&li_a7.to_le_bytes());
                // ecall
                code.extend_from_slice(&0x00000073u32.to_le_bytes());
                // Literal pool: main address
                code.extend_from_slice(&main_addr.to_le_bytes());
            }
            code
        }
    }
}

/// Synthesise a tiny `atexit` wrapper that forwards to `__cxa_atexit`.
///
/// In glibc ≥ 2.34 the `atexit` symbol is no longer exported from `libc.so.6`;
/// it lives in the static `libc_nonshared.a` archive.  Rather than requiring
/// the user to supply CRT objects we emit an equivalent stub inline:
///
/// ```asm
/// atexit:            ; fn already in %rdi (x86-64) / x0 (AArch64) / a0 (RV64)
///   xor %esi, %esi   ; arg  = NULL
///   xor %edx, %edx   ; dso  = 0  (main executable)
///   jmp __cxa_atexit  ; tail-call
/// ```
fn generate_synthetic_atexit(target: Target, _stub_vaddr: u64, cxa_addr: u64) -> Vec<u8> {
    match target {
        Target::X86_64 => {
            // fn already in %rdi
            // xor %esi, %esi       → 31 f6
            // xor %edx, %edx       → 31 d2
            // movabs $cxa, %rax    → 48 b8 <imm64>
            // jmp  *%rax           → ff e0
            let mut code = Vec::with_capacity(16);
            code.extend_from_slice(&[0x31, 0xf6]); // xor %esi, %esi
            code.extend_from_slice(&[0x31, 0xd2]); // xor %edx, %edx
            code.extend_from_slice(&[0x48, 0xb8]); // movabs $imm64, %rax
            code.extend_from_slice(&cxa_addr.to_le_bytes());
            code.extend_from_slice(&[0xff, 0xe0]); // jmp *%rax
            code
        }
        Target::I686 => {
            // cdecl: arg is on stack at [esp+4].  We need to push three args
            // for __cxa_atexit(fn, NULL, 0) then jump.
            // push $0                ; __dso_handle = 0
            // push $0                ; arg = NULL
            // push [esp+8+4]         ; fn (original arg, now at esp+8 due to 2 pushes)
            // Actually simpler: rewrite stack frame.  But a movabs+jmp is simplest
            // for a stub.  Since i686 uses the stack, let's do:
            //   mov 4(%esp), %eax     ; fn
            //   push $0               ; __dso_handle
            //   push $0               ; arg
            //   push %eax             ; fn
            //   mov $cxa, %eax
            //   jmp *%eax             ; tail-call (caller will ret past our pushes)
            // Actually for a proper tailcall we need to adjust the stack.
            // Simplest correct approach: just call __cxa_atexit and ret.
            let mut code = Vec::with_capacity(24);
            // mov 4(%esp), %eax
            code.extend_from_slice(&[0x8b, 0x44, 0x24, 0x04]);
            // sub $12, %esp  (align + space for 3 args)
            // Actually, for i686 cdecl __cxa_atexit(fn, arg, dso):
            // The caller already set up the stack with fn at [esp+4].
            // We need to set up [esp+4]=fn, [esp+8]=0, [esp+12]=0 then call.
            // Easiest: push 0, push 0, push fn (from [esp+4+8]), call, add $12 esp, ret
            // But that's not a tailcall. Let's just do call+ret:
            code.clear();
            // push $0                  ; dso_handle = 0
            code.extend_from_slice(&[0x6a, 0x00]);
            // push $0                  ; arg = NULL
            code.extend_from_slice(&[0x6a, 0x00]);
            // push [esp+4+8]           ; fn (offset 4 for ret addr + 8 for our pushes = 12)
            code.extend_from_slice(&[0xff, 0x74, 0x24, 0x0c]);
            // mov $cxa_addr, %eax
            code.push(0xb8);
            code.extend_from_slice(&(cxa_addr as u32).to_le_bytes());
            // call *%eax
            code.extend_from_slice(&[0xff, 0xd0]);
            // add $12, %esp
            code.extend_from_slice(&[0x83, 0xc4, 0x0c]);
            // ret
            code.push(0xc3);
            code
        }
        Target::AArch64 => {
            // fn already in x0
            // mov x1, #0       ; arg = NULL
            // mov x2, #0       ; dso = 0
            // br  x16 or b cxa ; jump to __cxa_atexit
            let mut code = Vec::with_capacity(20);
            // mov x1, #0: 0xD2800001
            code.extend_from_slice(&0xD280_0001u32.to_le_bytes());
            // mov x2, #0: 0xD2800002
            code.extend_from_slice(&0xD280_0002u32.to_le_bytes());
            // ldr x16, [pc, #8]: load cxa_addr from literal pool after next instr
            // 0x58000050  → ldr x16, .+8
            code.extend_from_slice(&0x5800_0050u32.to_le_bytes());
            // br x16: 0xD61F0200
            code.extend_from_slice(&0xD61F_0200u32.to_le_bytes());
            // Literal pool: cxa_addr (8 bytes)
            code.extend_from_slice(&cxa_addr.to_le_bytes());
            code
        }
        Target::RiscV64 => {
            // fn already in a0
            // li a1, 0         ; arg = NULL
            // li a2, 0         ; dso = 0
            // load cxa_addr into t1 via AUIPC+LD, then jr t1
            let mut code = Vec::with_capacity(28);
            // li a1, 0 → addi a1, zero, 0 : 0x00000593
            code.extend_from_slice(&0x0000_0593u32.to_le_bytes());
            // li a2, 0 → addi a2, zero, 0 : 0x00000613
            code.extend_from_slice(&0x0000_0613u32.to_le_bytes());
            // Load cxa_addr into t1 (x6) using lui+addi+jalr pattern:
            // lui  t1, hi20(cxa_addr)
            // jalr zero, t1, lo12(cxa_addr)  (i.e. jr with offset)
            let addr = cxa_addr;
            let lo12 = (addr & 0xFFF) as i32;
            let mut hi20 = ((addr >> 12) & 0xFFFFF) as u32;
            // Adjust for sign-extension of lo12
            if lo12 >= 0x800 {
                hi20 = hi20.wrapping_add(1) & 0xFFFFF;
            }
            // lui t1, hi20 : imm[31:12] | rd=6 | opcode=0110111
            let lui_instr = (hi20 << 12) | (6 << 7) | 0x37;
            code.extend_from_slice(&lui_instr.to_le_bytes());
            // jalr zero, lo12(t1) : imm[11:0] | rs1=6 | funct3=0 | rd=0 | opcode=1100111
            let lo12_u = (lo12 as u32) & 0xFFF;
            let jalr_instr = (lo12_u << 20) | (6 << 15) | 0x67;
            code.extend_from_slice(&jalr_instr.to_le_bytes());
            code
        }
    }
}

/// Generate dynamic linking sections for shared library output.
///
/// Populates section data buffers with `.dynamic`, `.dynsym`, `.dynstr`,
/// `.gnu.hash`, `.got`, `.got.plt`, `.plt`, `.rela.dyn`, and `.rela.plt`
/// section content.
fn generate_dynamic_sections(
    config: &LinkerConfig,
    sym_table: &symbol_resolver::SymbolTable,
    section_data_map: &mut FxHashMap<String, Vec<u8>>,
    _address_map: &section_merger::AddressMap,
    referenced_symbols: &FxHashSet<String>,
    plt_needed_symbols: &FxHashSet<String>,
    copy_symbols: &[(String, usize)],
    _diagnostics: &mut DiagnosticEngine,
) {
    // Build lists of exported and imported symbols for the dynamic linker.
    let mut exported: Vec<dynamic::ExportedSymbol> = Vec::new();
    let mut imported: Vec<dynamic::ImportedSymbol> = Vec::new();

    let is_executable = config.is_executable();

    for sym in &sym_table.symbols {
        let is_defined = sym.section_index != symbol_resolver::SHN_UNDEF;
        let is_global_or_weak =
            sym.binding == symbol_resolver::STB_GLOBAL || sym.binding == symbol_resolver::STB_WEAK;

        if is_defined && is_global_or_weak && !sym.name.is_empty() {
            // For executables, do NOT export defined symbols into .dynsym
            // unless they have default visibility and the user requested
            // -export-dynamic (not yet implemented). For shared libraries,
            // export all defined global/weak symbols so consumers can link
            // against them.
            if !is_executable {
                exported.push(dynamic::ExportedSymbol {
                    name: sym.name.clone(),
                    value: sym.value,
                    size: sym.size,
                    binding: sym.binding,
                    sym_type: sym.sym_type,
                    visibility: sym.visibility,
                    section_index: sym.section_index,
                });
            }
        } else if !is_defined && is_global_or_weak && !sym.name.is_empty() {
            // For executables, only import symbols that are actually
            // referenced by relocations. Declared-but-unused symbols
            // (e.g. from stdio.h) must NOT appear in .dynsym, otherwise
            // the dynamic linker will try and fail to resolve them.
            if is_executable && !referenced_symbols.contains(&sym.name) {
                // However, copy symbols still need .dynsym entries even
                // if they are not in referenced_symbols from PLT relocs.
                if !copy_symbols.iter().any(|(n, _)| *n == sym.name) {
                    continue;
                }
            }
            // Check if this symbol is a copy-relocated data symbol.
            let is_copy = copy_symbols.iter().any(|(n, _)| *n == sym.name);
            let copy_sz = copy_symbols
                .iter()
                .find(|(n, _)| *n == sym.name)
                .map(|(_, sz)| *sz as u64)
                .unwrap_or(0);
            imported.push(dynamic::ImportedSymbol {
                name: sym.name.clone(),
                binding: sym.binding,
                sym_type: sym.sym_type,
                needs_plt: !is_copy
                    && (sym.sym_type == symbol_resolver::STT_FUNC
                        || plt_needed_symbols.contains(&sym.name)),
                needs_copy: is_copy,
                copy_size: copy_sz,
            });
        }
    }

    // Collect needed libraries.
    let needed: Vec<String> = config.needed_libs.clone();

    // Build the dynamic linking sections.
    let dynamic_result = dynamic::build_dynamic_sections(
        &config.target,
        config.is_shared(),
        &exported,
        &imported,
        &needed,
        config.soname.as_deref(),
    );

    // Store the generated section data into the section data map.
    if !dynamic_result.dynamic_section.is_empty() {
        section_data_map.insert(".dynamic".to_string(), dynamic_result.dynamic_section);
    }
    if !dynamic_result.dynsym_section.is_empty() {
        section_data_map.insert(".dynsym".to_string(), dynamic_result.dynsym_section);
    }
    if !dynamic_result.dynstr_section.is_empty() {
        section_data_map.insert(".dynstr".to_string(), dynamic_result.dynstr_section);
    }
    if !dynamic_result.gnu_hash_section.is_empty() {
        section_data_map.insert(".gnu.hash".to_string(), dynamic_result.gnu_hash_section);
    }
    if !dynamic_result.got_section.is_empty() {
        section_data_map.insert(".got".to_string(), dynamic_result.got_section);
    }
    if !dynamic_result.got_plt_section.is_empty() {
        section_data_map.insert(".got.plt".to_string(), dynamic_result.got_plt_section);
    }
    if !dynamic_result.plt_section.is_empty() {
        section_data_map.insert(".plt".to_string(), dynamic_result.plt_section);
    }
    if !dynamic_result.rela_dyn_section.is_empty() {
        section_data_map.insert(".rela.dyn".to_string(), dynamic_result.rela_dyn_section);
    }
    if !dynamic_result.rela_plt_section.is_empty() {
        section_data_map.insert(".rela.plt".to_string(), dynamic_result.rela_plt_section);
    }
    if let Some(interp_bytes) = dynamic_result.interp {
        section_data_map.insert(".interp".to_string(), interp_bytes);
    }
}

/// Write the final ELF output binary.
///
/// Assembles sections, symbols, and program headers into a complete ELF
/// file using the `ElfWriter` infrastructure.
fn write_elf_output(
    config: &LinkerConfig,
    ordered_sections: &[&section_merger::OutputSection],
    section_data_map: &FxHashMap<String, Vec<u8>>,
    address_map: &section_merger::AddressMap,
    sym_table: &symbol_resolver::SymbolTable,
    entry_point: u64,
) -> Vec<u8> {
    let elf_type = config.output_type.to_elf_type();
    let mut writer = ElfWriter::new(config.target, elf_type);

    // Set the entry point for executables.
    if config.is_executable() {
        writer.set_entry_point(entry_point);
    }

    // Add output sections to the ELF writer.
    // For linked output (ET_EXEC / ET_DYN), populate virtual addresses and
    // file-offset hints from the address map so the ELF writer places
    // section data at the exact offsets the program headers reference.
    let is_linked = config.output_type.is_linked();
    for &output_section in ordered_sections {
        // Use patched data from the section data map if available,
        // otherwise fall back to the merged section data.
        let data = section_data_map
            .get(&output_section.name)
            .cloned()
            .unwrap_or_default();

        let (vaddr, foff) = if is_linked {
            if let Some(addr_info) = address_map.section_addresses.get(&output_section.name) {
                (addr_info.virtual_address, addr_info.file_offset)
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };

        // For SHT_NOBITS sections (.bss) the data vec is empty but the
        // in-memory size is non-zero.  Use the address-map mem_size so
        // that the ELF writer emits the correct sh_size and the
        // corresponding LOAD segment p_memsz covers the full BSS area.
        let logical_size = if output_section.section_type
            == crate::backend::linker_common::section_merger::SHT_NOBITS
        {
            if let Some(addr_info) = address_map.section_addresses.get(&output_section.name) {
                addr_info.mem_size
            } else {
                0
            }
        } else {
            0
        };

        let elf_section = Section {
            name: output_section.name.clone(),
            sh_type: output_section.section_type,
            sh_flags: output_section.flags,
            data,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: output_section.alignment,
            sh_entsize: 0,
            logical_size,
            virtual_address: vaddr,
            file_offset_hint: foff,
        };
        writer.add_section(elf_section);
    }

    // Add dynamic linking sections that were generated separately.
    // Track section indices (1-based ELF indices) for sh_link fixup.
    let dynamic_section_names = [
        ".interp",
        ".dynamic",
        ".dynsym",
        ".dynstr",
        ".gnu.hash",
        ".got",
        ".got.plt",
        ".plt",
        ".rela.dyn",
        ".rela.plt",
    ];

    // Compute a virtual address base for dynamic sections by finding the
    // highest address in the existing layout and aligning up.
    let mut dyn_vaddr_cursor: u64 = 0;
    let mut dyn_foff_cursor: u64 = 0;
    if is_linked {
        for info in address_map.section_addresses.values() {
            let sec_end_va = info.virtual_address + info.mem_size;
            let sec_end_fo = info.file_offset + info.size;
            if sec_end_va > dyn_vaddr_cursor {
                dyn_vaddr_cursor = sec_end_va;
            }
            if sec_end_fo > dyn_foff_cursor {
                dyn_foff_cursor = sec_end_fo;
            }
        }
        // Align to page boundary for clean segment mapping.
        let page_align = 0x1000u64;
        dyn_vaddr_cursor = (dyn_vaddr_cursor + page_align - 1) & !(page_align - 1);
        dyn_foff_cursor = (dyn_foff_cursor + page_align - 1) & !(page_align - 1);
    }

    // Map from section name → 1-based ELF section index for sh_link resolution.
    let mut dyn_section_indices: FxHashMap<String, u32> = FxHashMap::default();
    // Track dynamic sections' virtual addresses for PT_DYNAMIC generation.
    let mut dyn_section_vaddrs: FxHashMap<String, (u64, u64, u64)> = FxHashMap::default(); // (vaddr, foff, size)

    for &sec_name in &dynamic_section_names {
        let already_present = ordered_sections.iter().any(|s| s.name == sec_name);
        if already_present {
            continue;
        }

        if let Some(data) = section_data_map.get(sec_name) {
            if !data.is_empty() {
                let (sh_type, sh_flags, sh_entsize) = dynamic_section_attributes(sec_name, config);
                let align = if config.target.is_64bit() { 8u64 } else { 4u64 };

                // Assign virtual address and file offset for linked output.
                let (vaddr, foff) = if is_linked {
                    dyn_vaddr_cursor = (dyn_vaddr_cursor + align - 1) & !(align - 1);
                    dyn_foff_cursor = (dyn_foff_cursor + align - 1) & !(align - 1);
                    let va = dyn_vaddr_cursor;
                    let fo = dyn_foff_cursor;
                    dyn_vaddr_cursor += data.len() as u64;
                    dyn_foff_cursor += data.len() as u64;
                    (va, fo)
                } else {
                    (0, 0)
                };

                let elf_section = Section {
                    name: sec_name.to_string(),
                    sh_type,
                    sh_flags,
                    data: data.clone(),
                    sh_link: 0, // Fixed up below after all sections are added.
                    sh_info: 0,
                    sh_addralign: align,
                    sh_entsize,
                    logical_size: 0,
                    virtual_address: vaddr,
                    file_offset_hint: foff,
                };
                let idx = writer.add_section(elf_section);
                dyn_section_indices.insert(sec_name.to_string(), idx as u32);
                if is_linked {
                    dyn_section_vaddrs
                        .insert(sec_name.to_string(), (vaddr, foff, data.len() as u64));
                }
            }
        }
    }

    // ----- Fix up sh_link fields for dynamic sections -----
    // .dynsym → sh_link must point to .dynstr section index
    // .dynamic → sh_link must point to .dynstr section index
    // .gnu.hash → sh_link must point to .dynsym section index
    // .rela.dyn → sh_link must point to .dynsym section index
    // .rela.plt → sh_link must point to .dynsym section index
    let dynstr_idx = dyn_section_indices.get(".dynstr").copied().unwrap_or(0);
    let dynsym_idx = dyn_section_indices.get(".dynsym").copied().unwrap_or(0);

    // Resolve the 0-based vector indices from the 1-based ELF indices.
    {
        let sections = writer.sections_mut();
        for (sec_name, &elf_idx) in &dyn_section_indices {
            let vec_idx = (elf_idx as usize).saturating_sub(1);
            if vec_idx < sections.len() {
                match sec_name.as_str() {
                    ".dynsym" => {
                        sections[vec_idx].sh_link = dynstr_idx;
                        // sh_info = index of first non-local symbol (usually 1)
                        sections[vec_idx].sh_info = 1;
                    }
                    ".dynamic" => {
                        sections[vec_idx].sh_link = dynstr_idx;
                    }
                    ".gnu.hash" => {
                        sections[vec_idx].sh_link = dynsym_idx;
                    }
                    ".rela.dyn" | ".rela.plt" => {
                        sections[vec_idx].sh_link = dynsym_idx;
                        // sh_info for .rela.plt = section index of .plt (or .got.plt)
                        if sec_name == ".rela.plt" {
                            if let Some(&plt_idx) = dyn_section_indices.get(".got.plt") {
                                sections[vec_idx].sh_info = plt_idx;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // ----- Patch .dynamic section entries with resolved virtual addresses -----
    // The DynamicSection::build() initially sets DT_STRTAB, DT_SYMTAB,
    // DT_GNU_HASH, DT_RELA, DT_JMPREL, DT_PLTGOT to zero as placeholders.
    // Now that dyn_section_vaddrs has the final virtual addresses for each
    // dynamic section, we patch the .dynamic section data in-place.
    if is_linked && !dyn_section_vaddrs.is_empty() {
        if let Some(&dyn_elf_idx) = dyn_section_indices.get(".dynamic") {
            let vec_idx = (dyn_elf_idx as usize).saturating_sub(1);
            let is64 = config.target.is_64bit();
            let entry_size: usize = if is64 { 16 } else { 8 };
            let sections = writer.sections_mut();
            if vec_idx < sections.len() {
                let data = &mut sections[vec_idx].data;
                // Build a tag → vaddr map for the sections we need to patch.
                let mut patch_map: Vec<(i64, u64)> = Vec::new();
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".dynstr") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_STRTAB, va));
                }
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".dynsym") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_SYMTAB, va));
                }
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".gnu.hash") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_GNU_HASH, va));
                }
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".rela.dyn") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_RELA, va));
                }
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".rela.plt") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_JMPREL, va));
                }
                if let Some(&(va, _, _)) = dyn_section_vaddrs.get(".got.plt") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_PLTGOT, va));
                }
                // Patch PLTRELSZ and RELASZ to match actual section sizes,
                // which may differ from the initial encoding after PLT/GOT
                // regeneration (e.g. when synthetic atexit removed an entry).
                if let Some(&(_, _, sz)) = dyn_section_vaddrs.get(".rela.plt") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_PLTRELSZ, sz));
                }
                if let Some(&(_, _, sz)) = dyn_section_vaddrs.get(".rela.dyn") {
                    patch_map.push((crate::backend::linker_common::dynamic::DT_RELASZ, sz));
                }
                // Iterate over each .dynamic entry and patch matching tags.
                let num_entries = data.len() / entry_size;
                for i in 0..num_entries {
                    let offset = i * entry_size;
                    let tag = if is64 {
                        i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]))
                    } else {
                        i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]))
                            as i64
                    };
                    for &(patch_tag, patch_va) in &patch_map {
                        if tag == patch_tag {
                            if is64 {
                                let val_offset = offset + 8;
                                if val_offset + 8 <= data.len() {
                                    data[val_offset..val_offset + 8]
                                        .copy_from_slice(&patch_va.to_le_bytes());
                                }
                            } else {
                                let val_offset = offset + 4;
                                if val_offset + 4 <= data.len() {
                                    data[val_offset..val_offset + 4]
                                        .copy_from_slice(&(patch_va as u32).to_le_bytes());
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    // Build a mapping from output section virtual-address ranges to their
    // ELF section header indices so that defined symbols receive the
    // correct st_shndx value.  ELF section header index = vector_index + 1
    // because section 0 is the mandatory null section.
    let sec_ranges: Vec<(u64, u64, u16)> = {
        let secs = writer.sections_mut();
        secs.iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let va = s.virtual_address;
                let sz = if s.logical_size > 0 {
                    s.logical_size
                } else {
                    s.data.len() as u64
                };
                if va > 0 || sz > 0 {
                    Some((va, va.saturating_add(sz), (i + 1) as u16))
                } else {
                    None
                }
            })
            .collect()
    };

    // Add symbols to the output symbol table.
    for sym in &sym_table.symbols {
        // For defined symbols, resolve the correct output section index
        // by finding which section's address range contains the symbol.
        let out_shndx = if sym.section_index != symbol_resolver::SHN_UNDEF && sym.value > 0 {
            sec_ranges
                .iter()
                .find(|(lo, hi, _)| sym.value >= *lo && sym.value < *hi)
                .map(|(_, _, idx)| *idx)
                .unwrap_or(sym.section_index)
        } else {
            sym.section_index
        };

        let elf_sym = ElfSymbol {
            name: sym.name.clone(),
            value: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_index: out_shndx,
        };
        writer.add_symbol(elf_sym);
    }

    // Build program headers for linked output.
    if config.output_type.is_linked() {
        build_program_headers(config, ordered_sections, address_map, &mut writer);

        // ----- Add PT_INTERP program header for the .interp section -----
        // PT_INTERP tells the kernel which dynamic linker to invoke.
        // Must be present for dynamically-linked executables and shared libs.
        if let Some(&(interp_vaddr, interp_foff, interp_size)) = dyn_section_vaddrs.get(".interp") {
            let pt_interp = crate::backend::elf_writer_common::Elf64ProgramHeader {
                p_type: 3,  // PT_INTERP
                p_flags: 4, // PF_R
                p_offset: interp_foff,
                p_vaddr: interp_vaddr,
                p_paddr: interp_vaddr,
                p_filesz: interp_size,
                p_memsz: interp_size,
                p_align: 1,
            };
            writer.add_program_header(pt_interp);
        }

        // ----- Add PT_DYNAMIC program header for the .dynamic section -----
        if let Some(&(dyn_vaddr, dyn_foff, dyn_size)) = dyn_section_vaddrs.get(".dynamic") {
            let pt_dynamic = crate::backend::elf_writer_common::Elf64ProgramHeader {
                p_type: 2,  // PT_DYNAMIC
                p_flags: 6, // PF_R | PF_W
                p_offset: dyn_foff,
                p_vaddr: dyn_vaddr,
                p_paddr: dyn_vaddr,
                p_filesz: dyn_size,
                p_memsz: dyn_size,
                p_align: if config.target.is_64bit() { 8 } else { 4 },
            };
            writer.add_program_header(pt_dynamic);
        }

        // ----- Add PT_LOAD segment for dynamic sections if they have addresses -----
        // Gather the full range of dynamic sections to create a LOAD segment.
        if !dyn_section_vaddrs.is_empty() {
            let mut min_vaddr = u64::MAX;
            let mut min_foff = u64::MAX;
            let mut max_end_va = 0u64;
            let mut max_end_fo = 0u64;
            for (va, fo, sz) in dyn_section_vaddrs.values() {
                if *va < min_vaddr {
                    min_vaddr = *va;
                }
                if *fo < min_foff {
                    min_foff = *fo;
                }
                let end_va = va + sz;
                let end_fo = fo + sz;
                if end_va > max_end_va {
                    max_end_va = end_va;
                }
                if end_fo > max_end_fo {
                    max_end_fo = end_fo;
                }
            }
            let total_filesz = max_end_fo - min_foff;
            let total_memsz = max_end_va - min_vaddr;

            let pt_load_dyn = crate::backend::elf_writer_common::Elf64ProgramHeader {
                p_type: 1,  // PT_LOAD
                p_flags: 7, // PF_R | PF_W | PF_X (includes .plt which needs execute)
                p_offset: min_foff,
                p_vaddr: min_vaddr,
                p_paddr: min_vaddr,
                p_filesz: total_filesz,
                p_memsz: total_memsz,
                p_align: 0x1000,
            };
            writer.add_program_header(pt_load_dyn);
        }
    }

    // Serialize to bytes.
    writer.write()
}

/// Build ELF program headers based on the computed layout.
///
/// Creates `PT_LOAD`, `PT_DYNAMIC`, `PT_GNU_STACK`, and other segment
/// descriptors from the section layout and address map.
fn build_program_headers(
    config: &LinkerConfig,
    ordered_sections: &[&section_merger::OutputSection],
    address_map: &section_merger::AddressMap,
    writer: &mut ElfWriter,
) {
    // Use the linker script to compute the segment layout from section info.
    let is_shared = config.is_shared();
    let mut script = linker_script::DefaultLinkerScript::new(config.target, is_shared);

    // Build InputSectionInfo for each output section.
    // Use updated sizes from address_map when available — the address map
    // is patched after late additions (e.g. synthetic _start stub appended
    // to .text), whereas ordered_sections retains original pre-stub sizes.
    // CRITICAL: use mem_size (not file size) because compute_layout treats
    // InputSectionInfo.size as the *memory* size.  For SHT_NOBITS sections
    // like .bss, file size is 0 but mem_size is the actual allocation size.
    let section_infos: Vec<linker_script::InputSectionInfo> = ordered_sections
        .iter()
        .map(|sec| {
            let size = address_map
                .section_addresses
                .get(&sec.name)
                .map(|a| a.mem_size)
                .unwrap_or(sec.total_size);
            linker_script::InputSectionInfo {
                name: sec.name.clone(),
                size,
                alignment: sec.alignment,
                flags: sec.flags as u32,
            }
        })
        .collect();

    let layout = script.compute_layout(&section_infos);

    // Convert SegmentLayout entries to ELF program headers.
    for seg in &layout.segments {
        let phdr = crate::backend::elf_writer_common::Elf64ProgramHeader {
            p_type: seg.seg_type,
            p_flags: seg.flags,
            p_offset: seg.offset,
            p_vaddr: seg.vaddr,
            p_paddr: seg.paddr,
            p_filesz: seg.filesz,
            p_memsz: seg.memsz,
            p_align: seg.alignment,
        };
        writer.add_program_header(phdr);
    }
}

/// Return ELF section attributes (type, flags, entry size) for dynamically
/// generated sections.
fn dynamic_section_attributes(name: &str, config: &LinkerConfig) -> (u32, u64, u64) {
    let ptr_size = config.target.pointer_width() as u64;

    match name {
        ".interp" => (section_merger::SHT_PROGBITS, section_merger::SHF_ALLOC, 0),
        ".dynamic" => (
            section_merger::SHT_DYNAMIC,
            section_merger::SHF_ALLOC | section_merger::SHF_WRITE,
            ptr_size * 2, // Each dynamic entry is 2 × pointer width
        ),
        ".dynsym" => (
            section_merger::SHT_DYNSYM,
            section_merger::SHF_ALLOC,
            if config.target.is_64bit() { 24 } else { 16 },
        ),
        ".dynstr" => (section_merger::SHT_STRTAB, section_merger::SHF_ALLOC, 0),
        ".gnu.hash" => (section_merger::SHT_GNU_HASH, section_merger::SHF_ALLOC, 0),
        ".got" | ".got.plt" => (
            section_merger::SHT_PROGBITS,
            section_merger::SHF_ALLOC | section_merger::SHF_WRITE,
            ptr_size,
        ),
        ".plt" => (
            section_merger::SHT_PROGBITS,
            section_merger::SHF_ALLOC | section_merger::SHF_EXECINSTR,
            0,
        ),
        ".rela.dyn" | ".rela.plt" => {
            if config.target.is_64bit() {
                (section_merger::SHT_RELA, section_merger::SHF_ALLOC, 24)
            } else {
                // i686 uses SHT_REL (no addend) — glibc i386 dynamic linker
                // asserts DT_PLTREL == DT_REL.
                (section_merger::SHT_REL, section_merger::SHF_ALLOC, 8)
            }
        }
        ".bss.copy" => (
            section_merger::SHT_NOBITS,
            section_merger::SHF_ALLOC | section_merger::SHF_WRITE,
            0,
        ),
        _ => (section_merger::SHT_PROGBITS, section_merger::SHF_ALLOC, 0),
    }
}

// ============================================================================
// Copy Relocation Helpers
// ============================================================================

/// Resolve a `.dynsym` index to its symbol name string via `.dynstr`.
///
/// This is used when patching `R_*_COPY` relocations so that we match each
/// relocation to the correct copy symbol **by name** rather than by
/// sequential position (which can diverge between the `copy_symbols` list
/// and the `.rela.dyn` emission order).
fn resolve_dynsym_name(
    sym_idx: usize,
    sym_entry_size: usize,
    dynsym: Option<&[u8]>,
    dynstr: Option<&[u8]>,
    is64: bool,
) -> Option<String> {
    let dynsym = dynsym?;
    let dynstr = dynstr?;
    let base = sym_idx * sym_entry_size;
    if base + sym_entry_size > dynsym.len() {
        return None;
    }
    // st_name is always the first 4 bytes of Elf32_Sym / Elf64_Sym.
    let st_name = u32::from_le_bytes(dynsym[base..base + 4].try_into().ok()?) as usize;
    let _ = is64; // st_name offset is identical for both Elf32 and Elf64
    if st_name >= dynstr.len() {
        return None;
    }
    let end = dynstr[st_name..]
        .iter()
        .position(|&b| b == 0)
        .map(|p| st_name + p)
        .unwrap_or(dynstr.len());
    std::str::from_utf8(&dynstr[st_name..end])
        .ok()
        .map(|s| s.to_string())
}

/// Returns `true` if `name` is a known libc **data** symbol that requires a
/// copy relocation (`R_*_COPY`) rather than a PLT entry. Data symbols need
/// their *value* accessible at a fixed address in the executable's `.bss`
/// segment; a PLT stub is only suitable for *function* symbols.
fn is_known_data_symbol(name: &str) -> bool {
    matches!(
        name,
        "stdout"
            | "stderr"
            | "stdin"
            | "environ"
            | "__environ"
            | "optarg"
            | "optind"
            | "opterr"
            | "optopt"
            | "program_invocation_name"
            | "program_invocation_short_name"
            | "daylight"
            | "timezone"
            | "tzname"
            | "sys_nerr"
            | "sys_errlist"
            | "__progname"
            | "__progname_full"
    )
}

/// Returns the size in bytes for a known libc data symbol. Most are 8-byte
/// pointers; a few small integers are 4 bytes.
fn known_data_symbol_size(name: &str) -> usize {
    match name {
        "optind" | "opterr" | "optopt" | "daylight" | "sys_nerr" => 4,
        "tzname" => 16, // char *[2]
        _ => 8,         // pointer or long
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_type_display() {
        assert_eq!(format!("{}", OutputType::Executable), "executable");
        assert_eq!(format!("{}", OutputType::SharedLibrary), "shared library");
        assert_eq!(format!("{}", OutputType::Relocatable), "relocatable");
    }

    #[test]
    fn test_output_type_elf_type() {
        assert_eq!(OutputType::Executable.to_elf_type(), ET_EXEC);
        assert_eq!(OutputType::SharedLibrary.to_elf_type(), ET_DYN);
        assert_eq!(
            OutputType::Relocatable.to_elf_type(),
            crate::backend::elf_writer_common::ET_REL
        );
    }

    #[test]
    fn test_output_type_requires_pic() {
        assert!(!OutputType::Executable.requires_pic());
        assert!(OutputType::SharedLibrary.requires_pic());
        assert!(!OutputType::Relocatable.requires_pic());
    }

    #[test]
    fn test_output_type_is_linked() {
        assert!(OutputType::Executable.is_linked());
        assert!(OutputType::SharedLibrary.is_linked());
        assert!(!OutputType::Relocatable.is_linked());
    }

    #[test]
    fn test_linker_config_default() {
        let config = LinkerConfig::default();
        assert_eq!(config.target, Target::X86_64);
        assert_eq!(config.output_type, OutputType::Executable);
        assert_eq!(config.output_path, "a.out");
        assert_eq!(config.entry_point, "_start");
        assert!(!config.pic);
        assert!(!config.allow_undefined);
        assert!(config.soname.is_none());
        assert!(config.needed_libs.is_empty());
        assert!(!config.emit_debug);
    }

    #[test]
    fn test_linker_config_new_shared() {
        let config = LinkerConfig::new(Target::AArch64, OutputType::SharedLibrary);
        assert_eq!(config.target, Target::AArch64);
        assert!(config.pic); // Shared libraries require PIC
        assert!(config.allow_undefined); // Shared libraries allow undefined
        assert!(config.is_shared());
        assert!(!config.is_executable());
        assert!(!config.is_relocatable());
    }

    #[test]
    fn test_linker_config_new_executable() {
        let config = LinkerConfig::new(Target::RiscV64, OutputType::Executable);
        assert_eq!(config.target, Target::RiscV64);
        assert!(!config.pic);
        assert!(!config.allow_undefined);
        assert!(config.is_executable());
        assert!(!config.is_shared());
    }

    #[test]
    fn test_linker_input_new() {
        let input = LinkerInput::new(42, "test.o".to_string());
        assert_eq!(input.object_id, 42);
        assert_eq!(input.filename, "test.o");
        assert_eq!(input.section_count(), 0);
        assert_eq!(input.symbol_count(), 0);
        assert_eq!(input.relocation_count(), 0);
    }

    #[test]
    fn test_linker_output_debug() {
        let output = LinkerOutput {
            elf_data: vec![0x7f, b'E', b'L', b'F'],
            entry_point: 0x400000,
            output_type: OutputType::Executable,
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("LinkerOutput"));
        assert!(debug_str.contains("elf_data_len"));
        assert!(debug_str.contains("0x400000"));
        assert!(output.is_executable());
        assert!(!output.is_shared_library());
        assert_eq!(output.size(), 4);
    }

    #[test]
    fn test_resolve_entry_point_found() {
        let mut symbols = FxHashMap::default();
        symbols.insert("_start".to_string(), 0x401000u64);
        symbols.insert("main".to_string(), 0x401100u64);

        assert_eq!(resolve_entry_point("_start", &symbols), Ok(0x401000));
        assert_eq!(resolve_entry_point("main", &symbols), Ok(0x401100));
    }

    #[test]
    fn test_resolve_entry_point_not_found() {
        let symbols: FxHashMap<String, u64> = FxHashMap::default();
        let result = resolve_entry_point("_start", &symbols);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("_start"));
    }

    #[test]
    fn test_dynamic_section_attributes() {
        let config = LinkerConfig::new(Target::X86_64, OutputType::SharedLibrary);

        let (sh_type, sh_flags, _) = dynamic_section_attributes(".interp", &config);
        assert_eq!(sh_type, section_merger::SHT_PROGBITS);
        assert_eq!(sh_flags, section_merger::SHF_ALLOC);

        let (sh_type, sh_flags, sh_entsize) = dynamic_section_attributes(".dynamic", &config);
        assert_eq!(sh_type, section_merger::SHT_DYNAMIC);
        assert_ne!(sh_flags & section_merger::SHF_WRITE, 0);
        assert_eq!(sh_entsize, 16); // 2 × 8 bytes for 64-bit

        let (sh_type, _, sh_entsize) = dynamic_section_attributes(".dynsym", &config);
        assert_eq!(sh_type, section_merger::SHT_DYNSYM);
        assert_eq!(sh_entsize, 24); // Elf64_Sym size

        let (sh_type, _, _) = dynamic_section_attributes(".plt", &config);
        assert_eq!(sh_type, section_merger::SHT_PROGBITS);
    }

    #[test]
    fn test_output_type_equality() {
        assert_eq!(OutputType::Executable, OutputType::Executable);
        assert_ne!(OutputType::Executable, OutputType::SharedLibrary);
        assert_ne!(OutputType::SharedLibrary, OutputType::Relocatable);
    }

    #[test]
    fn test_linker_config_all_targets() {
        for &target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let config = LinkerConfig::new(target, OutputType::Executable);
            assert_eq!(config.target, target);
            assert_eq!(config.output_type, OutputType::Executable);
        }
    }
}
