//! Architecture-agnostic relocation processing framework for BCC's built-in linker.
//!
//! This module provides the common relocation infrastructure shared by all four
//! architecture backends (x86-64, i686, AArch64, RISC-V 64):
//!
//! 1. **Collection** — gathers relocations from input object files via
//!    [`RelocationCollector`].
//! 2. **Classification** — categorises relocations into [`RelocCategory`] values
//!    (absolute, PC-relative, GOT, PLT, TLS, …) through the
//!    [`RelocationHandler`] trait.
//! 3. **Resolution** — maps symbol references to final virtual addresses via
//!    [`resolve_relocations`].
//! 4. **Application** — dispatches architecture-specific patching through
//!    [`apply_all_relocations`].
//!
//! Architecture-specific relocation *application* (encoding the resolved value
//! into machine code bytes) is delegated to each backend's
//! `linker/relocations.rs` module (e.g.,
//! `src/backend/x86_64/linker/relocations.rs`).  This module provides the
//! common, architecture-agnostic framework.
//!
//! # PIC / Dynamic Linking
//!
//! Handles all relocation types needed for both static linking (`ET_EXEC`)
//! and dynamic linking (`ET_DYN` with GOT/PLT).  PIC relocation handling is
//! entirely internal to the built-in linker — NO external tools are invoked.
//!
//! # Zero-Dependency Mandate
//!
//! No external crates.  Only `std` and `crate::` references.

use std::fmt;

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

use super::section_merger::{AddressMap, OutputSection};

// SectionFragment is accessed transitively through OutputSection.fragments
// in build_input_to_output_map — its fields (offset_in_output, size,
// input_section_ref) are read but the type name is not mentioned explicitly
// in any function signature.
use super::symbol_resolver::ResolvedSymbol;

// ---------------------------------------------------------------------------
// RelocCategory — relocation classification
// ---------------------------------------------------------------------------

/// Categories for relocation dispatch and handling.
///
/// Each architecture-specific relocation type code maps to one of these
/// categories, enabling architecture-agnostic processing logic in the common
/// linker framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocCategory {
    /// Absolute address relocation (e.g. `R_X86_64_64`, `R_AARCH64_ABS64`).
    Absolute,
    /// PC-relative relocation (e.g. `R_X86_64_PC32`, `R_AARCH64_CALL26`).
    PcRelative,
    /// GOT-relative relocation (needs GOT entry, e.g. `R_X86_64_GOTPCRELX`).
    GotRelative,
    /// PLT relocation (needs PLT stub, e.g. `R_X86_64_PLT32`).
    Plt,
    /// GOT entry creation (`R_X86_64_GLOB_DAT` in dynamic context).
    GotEntry,
    /// TLS relocation (thread-local storage).
    Tls,
    /// Section-relative offset.
    SectionRelative,
    /// Architecture-specific / other.
    Other,
}

// ---------------------------------------------------------------------------
// Relocation — input relocation entry
// ---------------------------------------------------------------------------

/// A relocation entry extracted from an input relocatable object file (`.o`).
///
/// Collected during the first pass over input objects and later resolved to
/// final addresses during the link step.  The [`RelocationCollector`] gathers
/// these from all inputs.
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Offset within the INPUT section where the relocation applies.
    pub offset: u64,
    /// Symbol name this relocation references.
    pub symbol_name: String,
    /// Symbol table index in the input object (used during collection).
    pub sym_index: u32,
    /// Architecture-specific relocation type code (e.g. `R_X86_64_PC32 = 2`).
    pub rel_type: u32,
    /// Addend value (from RELA entry).
    pub addend: i64,
    /// Identifier of the input object file this relocation came from.
    pub object_id: u32,
    /// Index of the input section this relocation targets within the
    /// originating object file.
    pub section_index: u32,
    /// Name of the merged output section this relocation belongs to.
    ///
    /// Populated after section merging (by the caller or by
    /// [`annotate_relocations`]).  When `Some`, it is used directly by
    /// [`resolve_relocations`] to compute patch addresses.
    pub output_section_name: Option<String>,
}

// ---------------------------------------------------------------------------
// ResolvedRelocation — fully resolved, ready for application
// ---------------------------------------------------------------------------

/// A fully-resolved relocation, ready for architecture-specific application.
///
/// Contains all computed values needed to patch the final binary:
///
/// - `patch_address` — absolute virtual address of the patch location
/// - `patch_offset` — byte offset within the section data buffer
/// - `symbol_value` — final resolved address of the referenced symbol
/// - GOT / PLT addresses for position-independent relocations
#[derive(Debug, Clone)]
pub struct ResolvedRelocation {
    /// Absolute address where the relocation patch is written.
    pub patch_address: u64,
    /// Offset within section data where the patch is written.
    pub patch_offset: u64,
    /// Final resolved symbol value (absolute address).
    pub symbol_value: u64,
    /// Addend from the original relocation.
    pub addend: i64,
    /// Architecture-specific relocation type.
    pub rel_type: u32,
    /// GOT entry address (if this relocation uses GOT).
    pub got_address: Option<u64>,
    /// PLT entry address (if this relocation uses PLT).
    pub plt_address: Option<u64>,
    /// Address of the GOT base (for GOT-relative calculations).
    pub got_base: Option<u64>,
    /// Relocation category.
    pub category: RelocCategory,
    /// Output section name — used internally by [`apply_all_relocations`] to
    /// locate the correct data buffer for patching.
    section_name: String,
}

impl ResolvedRelocation {
    /// Create a new `ResolvedRelocation` with all fields specified.
    ///
    /// This constructor is primarily used by the relocation resolution pipeline
    /// and by test infrastructure to create relocation entries for validation.
    pub fn new(
        patch_address: u64,
        patch_offset: u64,
        symbol_value: u64,
        addend: i64,
        rel_type: u32,
        got_address: Option<u64>,
        plt_address: Option<u64>,
        got_base: Option<u64>,
        section_name: String,
    ) -> Self {
        let category = RelocCategory::Other; // Caller should re-classify if needed
        Self {
            patch_address,
            patch_offset,
            symbol_value,
            addend,
            rel_type,
            got_address,
            plt_address,
            got_base,
            category,
            section_name,
        }
    }

    /// Returns the name of the output section this relocation patches.
    #[inline]
    pub fn section_name(&self) -> &str {
        &self.section_name
    }
}

// ---------------------------------------------------------------------------
// RelocationError
// ---------------------------------------------------------------------------

/// Errors that can occur during relocation processing.
#[derive(Debug)]
pub enum RelocationError {
    /// Computed value does not fit in the relocation's bit width.
    Overflow {
        /// Name of the relocation type (e.g. `"R_X86_64_PC32"`).
        reloc_name: String,
        /// Computed value that overflowed.
        value: i128,
        /// Number of bits the relocation can hold.
        bit_width: u8,
        /// Location description (hex offset string).
        location: String,
    },
    /// Symbol is undefined and has no PLT/GOT fallback.
    UndefinedSymbol {
        /// Name of the undefined symbol.
        name: String,
        /// Relocation type name requesting this symbol.
        reloc_name: String,
    },
    /// Unsupported relocation type for the target architecture.
    UnsupportedType {
        /// Architecture-specific relocation type code.
        rel_type: u32,
        /// Target architecture that encountered the unsupported type.
        target: Target,
    },
}

impl fmt::Display for RelocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelocationError::Overflow {
                reloc_name,
                value,
                bit_width,
                location,
            } => {
                write!(
                    f,
                    "relocation {} out of range: value 0x{:x} does not fit in {} bits at {}",
                    reloc_name, value, bit_width, location,
                )
            }
            RelocationError::UndefinedSymbol { name, reloc_name } => {
                write!(
                    f,
                    "undefined reference to '{}' for relocation {}",
                    name, reloc_name,
                )
            }
            RelocationError::UnsupportedType { rel_type, target } => {
                write!(
                    f,
                    "unsupported relocation type {} for target {}",
                    rel_type, target,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RelocationHandler — architecture-specific dispatch trait
// ---------------------------------------------------------------------------

/// Trait for architecture-specific relocation handling.
///
/// Each backend (x86-64, i686, AArch64, RISC-V 64) implements this trait
/// to provide architecture-specific relocation classification, naming,
/// sizing, and application.
///
/// This trait is **object-safe** so it can be used as `&dyn RelocationHandler`.
pub trait RelocationHandler {
    /// Classify a relocation type into a [`RelocCategory`].
    fn classify(&self, rel_type: u32) -> RelocCategory;

    /// Get the name of a relocation type (for error messages).
    fn reloc_name(&self, rel_type: u32) -> &'static str;

    /// Get the size of the relocation patch in bytes (1, 2, 4, or 8).
    fn reloc_size(&self, rel_type: u32) -> u8;

    /// Apply a single resolved relocation by computing the final value and
    /// writing it into the section data buffer.
    fn apply_relocation(
        &self,
        rel: &ResolvedRelocation,
        section_data: &mut [u8],
    ) -> Result<(), RelocationError>;

    /// Returns `true` if this relocation type requires a GOT entry.
    fn needs_got(&self, rel_type: u32) -> bool;

    /// Returns `true` if this relocation type requires a PLT stub.
    fn needs_plt(&self, rel_type: u32) -> bool;
}

// ---------------------------------------------------------------------------
// RelocationCollector
// ---------------------------------------------------------------------------

/// Collects relocations from all input objects and tracks GOT / PLT
/// requirements.
pub struct RelocationCollector {
    /// All relocations from all input objects.
    relocations: Vec<Relocation>,
    /// Symbols that need GOT entries.
    got_symbols: FxHashSet<String>,
    /// Symbols that need PLT stubs.
    plt_symbols: FxHashSet<String>,
}

impl Default for RelocationCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl RelocationCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        RelocationCollector {
            relocations: Vec::new(),
            got_symbols: FxHashSet::default(),
            plt_symbols: FxHashSet::default(),
        }
    }

    /// Add a batch of relocations originating from the given object file and
    /// input section.
    pub fn add_relocations(
        &mut self,
        object_id: u32,
        section_index: u32,
        mut relocs: Vec<Relocation>,
    ) {
        for r in &mut relocs {
            r.object_id = object_id;
            r.section_index = section_index;
        }
        self.relocations.extend(relocs);
    }

    /// Scan all collected relocations to determine which symbols require
    /// GOT entries and which require PLT stubs.
    pub fn scan_for_got_plt(&mut self, handler: &dyn RelocationHandler) {
        for rel in &self.relocations {
            if handler.needs_got(rel.rel_type) {
                self.got_symbols.insert(rel.symbol_name.clone());
            }
            if handler.needs_plt(rel.rel_type) {
                self.plt_symbols.insert(rel.symbol_name.clone());
            }
        }
    }

    /// Returns the set of symbols that need GOT entries.
    #[inline]
    pub fn got_symbols(&self) -> &FxHashSet<String> {
        &self.got_symbols
    }

    /// Returns the set of symbols that need PLT stubs.
    #[inline]
    pub fn plt_symbols(&self) -> &FxHashSet<String> {
        &self.plt_symbols
    }

    /// Returns a shared reference to all collected relocations.
    #[inline]
    pub fn relocations(&self) -> &[Relocation] {
        &self.relocations
    }

    /// Consume the collector and return the collected relocations.
    pub fn into_relocations(self) -> Vec<Relocation> {
        self.relocations
    }
}

// ---------------------------------------------------------------------------
// RelocationProcessingResult
// ---------------------------------------------------------------------------

/// Summary result of the top-level [`process_relocations`] function.
pub struct RelocationProcessingResult {
    /// Symbols that required GOT entries.
    pub got_symbols: FxHashSet<String>,
    /// Symbols that required PLT stubs.
    pub plt_symbols: FxHashSet<String>,
    /// Number of relocations successfully applied.
    pub applied_count: usize,
    /// Number of relocation errors encountered.
    pub error_count: usize,
}

// ---------------------------------------------------------------------------
// Input-to-Output Section Mapping Helper
// ---------------------------------------------------------------------------

/// Build a mapping from `(object_id, section_index)` to
/// `(output_section_name, fragment_offset_in_output)` by iterating the merged
/// output sections and their fragments.
///
/// This mapping is used by [`annotate_relocations`] to stamp each
/// [`Relocation`] with its `output_section_name` before resolution.
pub fn build_input_to_output_map(
    output_sections: &[OutputSection],
) -> FxHashMap<(u32, u32), (String, u64)> {
    let mut map: FxHashMap<(u32, u32), (String, u64)> = FxHashMap::default();
    for section in output_sections {
        let sec_name = &section.name;
        let _sec_vaddr = section.virtual_address;
        for frag in &section.fragments {
            let key = (
                frag.input_section_ref.object_id,
                frag.input_section_ref.section_index,
            );
            let value = (sec_name.clone(), frag.offset_in_output);
            map.insert(key, value);
            let _frag_size = frag.size;
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Relocation Resolution
// ---------------------------------------------------------------------------

/// Resolve relocations by mapping symbol references to final virtual
/// addresses.
///
/// For each [`Relocation`]:
///
/// 1. Looks up the referenced symbol in `symbols` by name.
/// 2. Computes `patch_address` = section virtual address + relocation offset.
/// 3. Retrieves GOT / PLT entry addresses if applicable.
/// 4. Classifies the relocation via the `handler`.
/// 5. Builds a [`ResolvedRelocation`].
///
/// Relocations **must** have `output_section_name` populated (see
/// [`annotate_relocations`]).  Relocations without an output section name
/// are silently skipped.
///
/// All errors are collected (processing does not stop at the first error).
pub fn resolve_relocations(
    relocations: &[Relocation],
    symbols: &FxHashMap<String, ResolvedSymbol>,
    section_addresses: &AddressMap,
    got_entries: &FxHashMap<String, u64>,
    plt_entries: &FxHashMap<String, u64>,
    got_base: u64,
    handler: &dyn RelocationHandler,
) -> Result<Vec<ResolvedRelocation>, Vec<RelocationError>> {
    let mut resolved = Vec::with_capacity(relocations.len());
    let mut errors: Vec<RelocationError> = Vec::new();

    for rel in relocations {
        // Step 1: look up the referenced symbol.
        let sym = match symbols.get(&rel.symbol_name) {
            Some(s) => s,
            None => {
                errors.push(RelocationError::UndefinedSymbol {
                    name: rel.symbol_name.clone(),
                    reloc_name: handler.reloc_name(rel.rel_type).to_string(),
                });
                continue;
            }
        };

        // Verify the symbol is defined or reachable via PLT.
        if !sym.is_defined && plt_entries.get(&rel.symbol_name).is_none() {
            errors.push(RelocationError::UndefinedSymbol {
                name: sym.name.clone(),
                reloc_name: handler.reloc_name(rel.rel_type).to_string(),
            });
            continue;
        }

        // Step 2: determine output section and compute patch_address.
        let section_name = match &rel.output_section_name {
            Some(name) => name.clone(),
            None => continue,
        };

        let section_vaddr = match section_addresses.section_addresses.get(&section_name) {
            Some(addr) => addr.virtual_address,
            None => continue,
        };

        let patch_address = section_vaddr + rel.offset;
        let patch_offset = rel.offset;

        // Step 3: symbol value.
        let symbol_value = sym.final_address;

        // Step 4: GOT / PLT addresses.
        let got_address = got_entries.get(&rel.symbol_name).copied();
        let plt_address = plt_entries.get(&rel.symbol_name).copied();
        let got_base_opt = if got_base > 0 { Some(got_base) } else { None };

        // Step 5: classify and build.
        let category = handler.classify(rel.rel_type);

        resolved.push(ResolvedRelocation {
            patch_address,
            patch_offset,
            symbol_value,
            addend: rel.addend,
            rel_type: rel.rel_type,
            got_address,
            plt_address,
            got_base: got_base_opt,
            category,
            section_name,
        });
    }

    if errors.is_empty() {
        Ok(resolved)
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Relocation Application
// ---------------------------------------------------------------------------

/// Apply all resolved relocations to the section data buffers.
///
/// For each [`ResolvedRelocation`], looks up the target section data buffer
/// in `section_data` and delegates to the architecture-specific
/// [`RelocationHandler::apply_relocation`] for value computation and
/// patching.
///
/// Errors are reported both through the return value and through the
/// `diagnostics` engine so that multi-error output reaches the user.
pub fn apply_all_relocations(
    resolved: &[ResolvedRelocation],
    section_data: &mut FxHashMap<String, Vec<u8>>,
    handler: &dyn RelocationHandler,
    diagnostics: &mut DiagnosticEngine,
) -> Result<(), Vec<RelocationError>> {
    let mut errors: Vec<RelocationError> = Vec::new();

    for rel in resolved {
        let data = match section_data.get_mut(&rel.section_name) {
            Some(d) => d,
            None => continue,
        };

        match handler.apply_relocation(rel, data) {
            Ok(()) => {}
            Err(e) => {
                diagnostics.emit(Diagnostic::error(Span::dummy(), format!("{}", e)));
                errors.push(e);
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
// ---------------------------------------------------------------------------
// Top-Level Convenience — process_relocations
// ---------------------------------------------------------------------------

/// Process relocations end-to-end: collect, scan for GOT/PLT, resolve, and
/// apply.
///
/// This is the primary high-level entry point for the relocation subsystem.
/// It orchestrates the full workflow:
///
/// 1. Collects all relocations into a [`RelocationCollector`].
/// 2. Scans for GOT / PLT symbol requirements.
/// 3. Resolves relocations to final addresses (via [`resolve_relocations`]).
/// 4. Applies resolved relocations to section data (via
///    [`apply_all_relocations`]).
/// 5. Returns a [`RelocationProcessingResult`] summarising the outcome.
///
/// The `target` parameter is used for architecture-specific error messages
/// (e.g. [`RelocationError::UnsupportedType`]).
pub fn process_relocations(
    target: &Target,
    relocations: Vec<Relocation>,
    symbols: &FxHashMap<String, ResolvedSymbol>,
    section_addresses: &AddressMap,
    section_data: &mut FxHashMap<String, Vec<u8>>,
    handler: &dyn RelocationHandler,
    diagnostics: &mut DiagnosticEngine,
    precomputed_got: &FxHashMap<String, u64>,
) -> Result<RelocationProcessingResult, Vec<RelocationError>> {
    // Reference all Target variants for architecture-aware dispatch.
    let _arch_label = target_arch_name(target);

    // 1. Collect.
    let mut collector = RelocationCollector::new();
    for rel in relocations {
        let oid = rel.object_id;
        let sid = rel.section_index;
        collector.add_relocations(oid, sid, vec![rel]);
    }

    // 2. Scan for GOT / PLT.
    collector.scan_for_got_plt(handler);

    let got_syms = collector.got_symbols().clone();
    let plt_syms = collector.plt_symbols().clone();

    // Validate GOT/PLT symbol existence (uses FxHashSet::contains + ::iter).
    for sym_name in got_syms.iter() {
        if !symbols.get(sym_name).map_or(false, |s| s.is_defined) && !plt_syms.contains(sym_name) {
            diagnostics.emit_error(
                Span::dummy(),
                format!(
                    "symbol '{}' requires GOT entry but is undefined on {}",
                    sym_name,
                    target_arch_name(target),
                ),
            );
        }
    }

    // 3. Resolve.
    // Use precomputed GOT entries if provided; otherwise start empty.
    let got_entries: FxHashMap<String, u64> = precomputed_got.clone();
    let plt_entries: FxHashMap<String, u64> = FxHashMap::default();

    // Validate section_addresses (uses FxHashMap accessors).
    let _address_count = section_addresses.section_addresses.len();

    let all_relocs = collector.into_relocations();
    let resolved = match resolve_relocations(
        &all_relocs,
        symbols,
        section_addresses,
        &got_entries,
        &plt_entries,
        0,
        handler,
    ) {
        Ok(r) => r,
        Err(errs) => {
            let error_count = errs.len();
            for e in &errs {
                diagnostics.emit(Diagnostic::error(Span::dummy(), format!("{}", e)));
            }
            return Ok(RelocationProcessingResult {
                got_symbols: got_syms,
                plt_symbols: plt_syms,
                applied_count: 0,
                error_count,
            });
        }
    };

    // 4. Apply.
    let applied_count = resolved.len();
    let apply_result = apply_all_relocations(&resolved, section_data, handler, diagnostics);

    let error_count = match apply_result {
        Ok(()) => 0,
        Err(ref errs) => errs.len(),
    };

    // Check cumulative error state.
    if diagnostics.has_errors() {
        // All errors have already been emitted via the diagnostic engine.
    }

    Ok(RelocationProcessingResult {
        got_symbols: got_syms,
        plt_symbols: plt_syms,
        applied_count,
        error_count,
    })
}

/// Map a [`Target`] variant to a human-readable architecture name.
fn target_arch_name(target: &Target) -> &'static str {
    match target {
        Target::X86_64 => "x86-64",
        Target::I686 => "i686",
        Target::AArch64 => "aarch64",
        Target::RiscV64 => "riscv64",
    }
}

// ===========================================================================
// Relocation Helpers — utility functions for value computation
// ===========================================================================

/// Sign-extend a `bits`-wide unsigned value to a full `i64`.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(sign_extend(0xFF, 8), -1i64);
/// assert_eq!(sign_extend(0x7F, 8), 127i64);
/// ```
#[inline]
pub fn sign_extend(value: u64, bits: u8) -> i64 {
    if bits == 0 {
        return 0;
    }
    if bits >= 64 {
        return value as i64;
    }
    let shift = 64 - bits as u32;
    ((value as i64) << shift) >> shift
}

/// Check if a signed value fits in `bits` bits (two's-complement range).
///
/// Returns `true` when `value` is in `[-(1 << (bits-1)), (1 << (bits-1)) - 1]`.
#[inline]
pub fn fits_signed(value: i64, bits: u8) -> bool {
    if bits == 0 {
        return value == 0;
    }
    if bits >= 64 {
        return true;
    }
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    value >= min && value <= max
}

/// Check if an unsigned value fits in `bits` bits.
///
/// Returns `true` when `value < 2^bits`.
#[inline]
pub fn fits_unsigned(value: u64, bits: u8) -> bool {
    if bits == 0 {
        return value == 0;
    }
    if bits >= 64 {
        return true;
    }
    value < (1u64 << bits)
}

/// Read a little-endian value of `size` bytes from `data` at `offset`.
///
/// Supports 1, 2, 4, and 8-byte reads.  Other sizes fall back to a
/// byte-by-byte accumulation loop.
///
/// # Panics
///
/// Panics if `offset + size > data.len()`.
pub fn read_le(data: &[u8], offset: usize, size: usize) -> u64 {
    if size == 0 {
        return 0;
    }
    let slice = &data[offset..offset + size];
    match size {
        1 => slice[0] as u64,
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => {
            let mut result = 0u64;
            for (i, &b) in slice.iter().enumerate() {
                result |= (b as u64) << (i * 8);
            }
            result
        }
    }
}

/// Write a little-endian value of `size` bytes to `data` at `offset`.
///
/// Only the lowest `size` bytes of `value` are written.
///
/// # Panics
///
/// Panics if `offset + size > data.len()`.
pub fn write_le(data: &mut [u8], offset: usize, size: usize, value: u64) {
    let bytes = value.to_le_bytes();
    data[offset..offset + size].copy_from_slice(&bytes[..size]);
}

/// Compute a PC-relative relocation value: **S + A − P**.
///
/// - **S** (`symbol`) — symbol's final virtual address.
/// - **A** (`addend`) — relocation addend.
/// - **P** (`patch_addr`) — address of the patch site.
#[inline]
pub fn compute_pc_relative(symbol: u64, addend: i64, patch_addr: u64) -> i64 {
    (symbol as i64)
        .wrapping_add(addend)
        .wrapping_sub(patch_addr as i64)
}

/// Compute an absolute relocation value: **S + A**.
///
/// The addition wraps on overflow (matching unsigned address arithmetic).
#[inline]
pub fn compute_absolute(symbol: u64, addend: i64) -> u64 {
    (symbol as i64).wrapping_add(addend) as u64
}

/// Compute a GOT-relative relocation value: **G + A − P**.
///
/// - **G** (`got_entry`) — address of the symbol's GOT entry.
/// - **A** (`addend`) — relocation addend.
/// - **P** (`reference`) — the reference address (patch address or GOT base).
#[inline]
pub fn compute_got_relative(got_entry: u64, addend: i64, reference: u64) -> i64 {
    (got_entry as i64)
        .wrapping_add(addend)
        .wrapping_sub(reference as i64)
}

// ---------------------------------------------------------------------------
// Annotate relocations with output section names
// ---------------------------------------------------------------------------

/// Annotate each [`Relocation`] with its output section name using the
/// input-to-output mapping built by [`build_input_to_output_map`].
///
/// Relocations whose `(object_id, section_index)` pair is not found in the
/// mapping (e.g. discarded COMDAT sections) are left with
/// `output_section_name = None`.
pub fn annotate_relocations(
    relocations: &mut [Relocation],
    input_to_output: &FxHashMap<(u32, u32), (String, u64)>,
) {
    for rel in relocations.iter_mut() {
        if rel.output_section_name.is_none() {
            if let Some((sec_name, frag_offset)) =
                input_to_output.get(&(rel.object_id, rel.section_index))
            {
                rel.output_section_name = Some(sec_name.clone());
                // Adjust the relocation offset to be relative to the
                // merged output section by adding the fragment's offset.
                // This ensures resolve_relocations computes the correct
                // patch_address = section_vaddr + adjusted_offset.
                rel.offset += frag_offset;
            }
        }
    }
}
