//! Shared linker infrastructure layer for BCC's built-in linker.
//!
//! This module provides the common linker functionality used by all four
//! architecture-specific linkers (x86-64, i686, AArch64, RISC-V 64),
//! enabling the standalone backend mode where BCC includes its own linker
//! without invoking any external `ld` or `lld` (per Section 0.7.7 of the
//! design specification).
//!
//! # Submodules
//!
//! - [`section_merger`]: Input section aggregation into output sections,
//!   alignment padding, COMDAT/section group deduplication, standard ELF
//!   output section ordering, and virtual address / file offset assignment.
//!
//! - [`symbol_resolver`]: Two-pass symbol resolution — collect definitions
//!   from all input objects, then resolve undefined references with strong/weak
//!   binding rules.
//!
//! - [`relocation`]: Architecture-agnostic relocation collection and dispatch
//!   framework. Architecture backends implement the [`ArchRelocationHandler`]
//!   trait for target-specific relocation application.
//!
//! - [`dynamic`]: Dynamic linking section generation for shared library
//!   (`-shared` / `-fPIC`) output — `.dynamic`, `.dynsym`, `.dynstr`,
//!   `.gnu.hash`, `.got`, `.got.plt`, `.plt`, `.rela.dyn`, `.rela.plt`.
//!
//! - [`linker_script`]: Default section-to-segment mapping rules,
//!   architecture-specific base addresses, page sizes, and entry point
//!   (`_start`) configuration.
//!
//! # Import Architecture
//!
//! Per the project's import architecture rules:
//! - `linker_common` modules import from `crate::common::` for types, target,
//!   diagnostics, and FxHash.
//! - `linker_common` modules import from `crate::backend::elf_writer_common`
//!   for ELF structures and constants.
//! - `linker_common` is imported by architecture-specific linkers
//!   (`src/backend/x86_64/linker/`, etc.).
//! - `linker_common` is imported by `src/backend/generation.rs` for linker
//!   invocation.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Input section aggregation and output section layout engine.
///
/// Collects sections from multiple input object files, groups them by name
/// and type into output sections following standard ELF ordering (`.text`,
/// `.rodata`, `.data`, `.bss`), handles COMDAT deduplication, and assigns
/// virtual addresses and file offsets.
pub mod section_merger;

/// Two-pass symbol resolution engine.
///
/// First pass collects all symbol definitions from input object files;
/// second pass resolves all undefined references with strong/weak binding
/// rules, multiple-definition detection, and visibility enforcement.
pub mod symbol_resolver;

/// Architecture-agnostic relocation processing framework.
///
/// Collects relocations from input object files, translates input-section
/// offsets to output-section offsets, classifies relocations for GOT/PLT
/// needs, and dispatches to architecture-specific handlers for application.
pub mod relocation;

/// Dynamic linking section generator for shared library (ET_DYN) output.
///
/// Generates `.dynamic`, `.dynsym` / `.dynstr`, `.gnu.hash`, `.got` /
/// `.got.plt`, `.plt`, `.rela.dyn` / `.rela.plt`, and `PT_DYNAMIC` /
/// `PT_INTERP` program headers for `-shared` and `-fPIC` linking.
pub mod dynamic;

/// Default linker script handling and section-to-segment mapping.
///
/// Provides the default section placement rules, segment layout,
/// architecture-specific base addresses, page sizes, and `_start` entry
/// point configuration used when no external linker script is specified.
pub mod linker_script;

// ---------------------------------------------------------------------------
// Convenience re-exports for ergonomic access from architecture linkers
// ---------------------------------------------------------------------------

// From section_merger — core section merging types
pub use section_merger::{InputRelocation, InputSection, MergedInput, OutputSection, SectionMerger};

// From symbol_resolver — symbol resolution types
pub use symbol_resolver::{
    ArchiveSymbol, InputSymbol, LinkError, ResolvedSymbols, SymbolBinding, SymbolEntry,
    SymbolResolver, SymbolType, SymbolVisibility,
};

// From relocation — relocation processing types
pub use relocation::{
    ArchRelocationHandler, RelocationClassification, RelocationEntry, RelocationError,
    RelocationProcessor,
};

// From dynamic — dynamic linking types
pub use dynamic::{
    DynamicLayout, DynamicRelocation, DynamicSectionBuilder, DynamicSymbolTable, GotBuilder,
    PltBuilder,
};

// From linker_script — linker script types
pub use linker_script::{LinkerScript, OutputType, SectionAssignment, SegmentRule};
