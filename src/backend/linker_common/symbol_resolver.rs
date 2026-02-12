//! Two-pass symbol resolution engine for the BCC built-in linker.
//!
//! Implements collect-then-resolve symbol processing:
//!
//! 1. **Pass 1 — Symbol Collection:** Collects all symbol definitions from
//!    input object files into a global symbol table. Applies strong/weak
//!    binding rules: strong (Global) definitions override weak definitions;
//!    multiple strong definitions of the same symbol produce a linker error.
//!    Local symbols are stored per-object and never enter the global table.
//!
//! 2. **Pass 2 — Reference Resolution:** Resolves all undefined references
//!    against collected definitions. Unresolved references produce structured
//!    `LinkError` diagnostics with source object context. The result is a
//!    `ResolvedSymbols` table with O(1) name-to-index lookup for efficient
//!    relocation processing.
//!
//! Used by all four architecture-specific linkers (x86-64, i686, AArch64,
//! RISC-V 64) as the shared symbol resolution infrastructure. Removing this
//! would break all linking operations since symbols could not be resolved
//! across object files.
//!
//! # Symbol Visibility
//!
//! Symbols carry ELF visibility attributes (Default, Hidden, Protected) that
//! control dynamic symbol table export:
//! - **Default:** Exported in `.dynsym`, subject to preemption.
//! - **Hidden:** Not exported in `.dynsym`; invisible to dynamic linker.
//! - **Protected:** Exported in `.dynsym` but cannot be preempted.
//!
//! # Archive Scanning
//!
//! Static archive (.a) files are processed lazily: only archive members that
//! define currently-undefined symbols are included in the link. This matches
//! the standard UNIX linker archive inclusion semantics.
//!
//! # Diagnostics Integration
//!
//! The resolver integrates with the BCC diagnostic engine for supplementary
//! reporting (weak-overrides-strong notes, type mismatch warnings). Critical
//! errors are captured structurally as `LinkError` variants returned from
//! `resolve_references()`.

use std::fmt;

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// SymbolBinding — ELF symbol binding strength
// ===========================================================================

/// Symbol binding strength, mapping to ELF `STB_LOCAL` / `STB_GLOBAL` / `STB_WEAK`.
///
/// Strong (`Global`) definitions override `Weak` definitions during resolution.
/// Multiple `Global` definitions of the same symbol produce a linker error.
/// `Local` symbols are visible only within their defining object file and never
/// participate in cross-object resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolBinding {
    /// `STB_LOCAL` — visible only within the defining object file.
    Local,
    /// `STB_GLOBAL` — visible across all object files; strong binding.
    Global,
    /// `STB_WEAK` — visible across all object files; overridden by strong (`Global`).
    Weak,
}

impl fmt::Display for SymbolBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolBinding::Local => write!(f, "LOCAL"),
            SymbolBinding::Global => write!(f, "GLOBAL"),
            SymbolBinding::Weak => write!(f, "WEAK"),
        }
    }
}

// ===========================================================================
// SymbolVisibility — ELF symbol visibility
// ===========================================================================

/// Symbol visibility, mapping to ELF `STV_DEFAULT` / `STV_HIDDEN` / `STV_PROTECTED`.
///
/// Controls whether a symbol is exported in the dynamic symbol table (`.dynsym`)
/// for shared library output and whether it can be preempted by the dynamic linker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolVisibility {
    /// `STV_DEFAULT` — exported in `.dynsym`, can be preempted by the dynamic linker.
    Default,
    /// `STV_HIDDEN` — not exported in `.dynsym`; invisible to the dynamic linker.
    Hidden,
    /// `STV_PROTECTED` — exported in `.dynsym` but cannot be preempted.
    Protected,
}

impl fmt::Display for SymbolVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolVisibility::Default => write!(f, "DEFAULT"),
            SymbolVisibility::Hidden => write!(f, "HIDDEN"),
            SymbolVisibility::Protected => write!(f, "PROTECTED"),
        }
    }
}

// ===========================================================================
// SymbolType — ELF symbol type classification
// ===========================================================================

/// Symbol type classification, mapping to ELF `STT_NOTYPE` / `STT_OBJECT` /
/// `STT_FUNC` / `STT_SECTION` / `STT_FILE`.
///
/// Used for symbol type consistency checking across object files and for
/// generating the correct ELF symbol table entry type field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolType {
    /// `STT_NOTYPE` — unspecified type.
    NoType,
    /// `STT_OBJECT` — data object (variable, array, etc.).
    Object,
    /// `STT_FUNC` — function entry point.
    Func,
    /// `STT_SECTION` — associated with a section (used internally by the linker).
    Section,
    /// `STT_FILE` — source file name.
    File,
}

impl fmt::Display for SymbolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolType::NoType => write!(f, "NOTYPE"),
            SymbolType::Object => write!(f, "OBJECT"),
            SymbolType::Func => write!(f, "FUNC"),
            SymbolType::Section => write!(f, "SECTION"),
            SymbolType::File => write!(f, "FILE"),
        }
    }
}

// ===========================================================================
// InputSymbol — symbol from an input object file
// ===========================================================================

/// A symbol from an input object file, before resolution.
///
/// Each input object provides a list of `InputSymbol`s that the resolver
/// collects during Pass 1. A `section_index` of 0 (`SHN_UNDEF`) indicates
/// that the symbol is an undefined reference to be resolved against other objects.
#[derive(Debug, Clone)]
pub struct InputSymbol {
    /// Symbol name (empty string for anonymous/null symbols, which are skipped).
    pub name: String,
    /// Symbol value (address/offset within its section).
    pub value: u64,
    /// Symbol size in bytes (0 if unknown or not applicable).
    pub size: u64,
    /// Binding strength: Local, Global, or Weak.
    pub binding: SymbolBinding,
    /// Symbol type: NoType, Object, Func, Section, or File.
    pub sym_type: SymbolType,
    /// Visibility: Default, Hidden, or Protected.
    pub visibility: SymbolVisibility,
    /// Index of the section this symbol is defined in.
    /// `SHN_UNDEF` (0) means the symbol is an undefined reference.
    pub section_index: u16,
}

// ===========================================================================
// SymbolEntry — resolved symbol entry in the output
// ===========================================================================

/// A resolved symbol entry tracking identity, value, binding, and origin.
///
/// This is the "merged" symbol that appears in the output ELF's `.symtab`.
/// It carries provenance information (`defining_object`) for diagnostic
/// reporting and relocation processing.
#[derive(Debug, Clone)]
pub struct SymbolEntry {
    /// Symbol name.
    pub name: String,
    /// Resolved virtual address / value. Updated during final layout.
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Binding strength (Global or Weak; Local symbols are separate).
    pub binding: SymbolBinding,
    /// Symbol type classification.
    pub sym_type: SymbolType,
    /// Visibility for dynamic symbol table control.
    pub visibility: SymbolVisibility,
    /// Output section index containing this symbol.
    pub section_index: u16,
    /// Index of the input object that defines this symbol.
    pub defining_object: usize,
    /// Whether this symbol has a definition (`true`) or is an undefined import
    /// (`false`, resolved from a shared library at runtime).
    pub is_defined: bool,
}

// ===========================================================================
// ArchiveSymbol — symbol from a static archive (.a)
// ===========================================================================

/// A symbol from an archive (.a) symbol table, used for lazy inclusion of
/// archive members.
///
/// Only archive members that define currently-undefined symbols are pulled
/// into the link, matching standard UNIX linker semantics.
#[derive(Debug, Clone)]
pub struct ArchiveSymbol {
    /// Symbol name exported by this archive member.
    pub name: String,
    /// Index of the object within the archive.
    pub object_index: usize,
    /// Descriptive name of the archive member (e.g., `"libfoo.a(bar.o)"`).
    pub object_name: String,
}

// ===========================================================================
// LinkError — linker error reporting
// ===========================================================================

/// Linker errors produced during symbol resolution.
///
/// These are the critical errors that prevent successful linking. They are
/// returned from `resolve_references()` as a `Vec<LinkError>` and carry
/// structured information for formatted error output.
#[derive(Debug, Clone)]
pub enum LinkError {
    /// An undefined symbol was referenced but never defined in any input object.
    UndefinedSymbol {
        /// Name of the undefined symbol.
        name: String,
        /// List of object file names that reference this symbol.
        referenced_by: Vec<String>,
    },
    /// A symbol was defined in multiple object files with strong (`Global`)
    /// binding — only one strong definition is allowed.
    MultipleDefinition {
        /// Name of the multiply-defined symbol.
        name: String,
        /// List of object file names containing strong definitions.
        defined_in: Vec<String>,
    },
    /// A symbol was defined with incompatible types across object files
    /// (e.g., `STT_FUNC` in one object and `STT_OBJECT` in another).
    SymbolTypeMismatch {
        /// Name of the symbol.
        name: String,
        /// Expected type (from the first definition encountered).
        expected: SymbolType,
        /// Found type (from a subsequent conflicting definition).
        found: SymbolType,
    },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinkError::UndefinedSymbol {
                name,
                referenced_by,
            } => {
                write!(f, "undefined reference to `{}`", name)?;
                if !referenced_by.is_empty() {
                    write!(f, " (referenced by: {})", referenced_by.join(", "))?;
                }
                Ok(())
            }
            LinkError::MultipleDefinition { name, defined_in } => {
                write!(
                    f,
                    "multiple definition of `{}` in: {}",
                    name,
                    defined_in.join(", ")
                )
            }
            LinkError::SymbolTypeMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "symbol type mismatch for `{}`: expected {}, found {}",
                    name, expected, found
                )
            }
        }
    }
}

// ===========================================================================
// Internal: UndefinedRef — pending undefined reference
// ===========================================================================

/// Internal representation of a pending undefined reference encountered during
/// Pass 1 symbol collection. Tracks both the symbol name and the source object
/// for diagnostic context.
#[derive(Debug, Clone)]
struct UndefinedRef {
    /// Name of the referenced symbol.
    name: String,
    /// Index of the object file containing this reference.
    object_index: usize,
    /// Descriptive name of the referencing object (for diagnostics).
    object_name: String,
}

// ===========================================================================
// Internal: MultiDefRecord — structured multiple-definition tracking
// ===========================================================================

/// Internal structured record for tracking multiple-definition conflicts.
/// Collects all object files that provide strong definitions of the same symbol.
#[derive(Debug, Clone)]
struct MultiDefRecord {
    /// Name of the multiply-defined symbol.
    name: String,
    /// Object file names containing strong definitions.
    defined_in: Vec<String>,
}

// ===========================================================================
// Internal: DiagnosticNote — supplementary diagnostic messages
// ===========================================================================

/// Internal supplementary diagnostic message collected during symbol resolution.
/// These are non-critical notes and warnings flushed to a `DiagnosticEngine`
/// via `emit_diagnostics()`.
#[derive(Debug, Clone)]
struct DiagnosticNote {
    /// Severity: true = warning, false = note.
    is_warning: bool,
    /// Human-readable message.
    message: String,
}

// ===========================================================================
// ResolvedSymbols — final symbol table
// ===========================================================================

/// Final resolved symbol table produced by Pass 2 of the symbol resolver.
///
/// Provides O(1) name-to-index lookup via `symbol_map` for efficient relocation
/// processing. Symbols are ordered with locals first (per-object order), then
/// globals sorted by name for deterministic output.
#[derive(Debug, Clone)]
pub struct ResolvedSymbols {
    /// All resolved symbols in output order (locals first, then globals sorted).
    pub symbols: Vec<SymbolEntry>,
    /// Name-to-index mapping for O(1) lookup during relocation processing.
    /// For global symbols, the key is the symbol name directly.
    /// For local symbols, the key is `__local_{object_index}_{name}` to avoid
    /// collisions between local symbols of the same name in different objects.
    pub symbol_map: FxHashMap<String, usize>,
}

impl ResolvedSymbols {
    /// Looks up a symbol by name, returning a reference to its entry if found.
    ///
    /// Searches the global symbol namespace. For local symbols, use the
    /// mangled key format `__local_{object_index}_{name}`.
    pub fn get_symbol(&self, name: &str) -> Option<&SymbolEntry> {
        self.symbol_map.get(name).map(|&idx| &self.symbols[idx])
    }

    /// Looks up a symbol's resolved value (virtual address) by name.
    ///
    /// Returns `None` if the symbol is not in the resolved table.
    pub fn get_symbol_value(&self, name: &str) -> Option<u64> {
        self.get_symbol(name).map(|entry| entry.value)
    }

    /// Returns the total number of resolved symbols (locals + globals).
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Returns `true` if no symbols were resolved.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

// ===========================================================================
// SymbolResolver — two-pass symbol resolution engine
// ===========================================================================

/// Two-pass symbol resolution engine for the BCC built-in linker.
///
/// # Usage
///
/// ```text
/// let mut resolver = SymbolResolver::new();
/// resolver.set_target(Target::X86_64);
///
/// // Register objects for diagnostic context
/// resolver.register_object(0, "main.o");
/// resolver.register_object(1, "lib.o");
///
/// // Pass 1 — collect symbols from each input object
/// for (i, obj) in objects.iter().enumerate() {
///     resolver.collect_symbols(i, &obj.symbols);
/// }
///
/// // Optional — scan archives for lazy inclusion
/// resolver.scan_archive(&archive_symbols);
///
/// // Optional — emit supplementary diagnostics
/// resolver.emit_diagnostics(&mut diag_engine);
///
/// // Pass 2 — resolve references
/// let resolved = resolver.resolve_references()?;
/// ```
///
/// # Architecture Awareness
///
/// The resolver carries a `Target` for architecture-specific diagnostic
/// messages, allowing linker errors to include the target architecture
/// context (e.g., "x86-64 linker: undefined reference to `foo`").
pub struct SymbolResolver {
    /// Global symbol table mapping symbol name → entry.
    /// Populated during Pass 1 with Global and Weak definitions.
    global_symbols: FxHashMap<String, SymbolEntry>,

    /// Per-object local symbol lists. Local symbols are not globally
    /// visible and are stored separately, indexed by object index.
    local_symbols: Vec<Vec<SymbolEntry>>,

    /// Pending undefined references encountered during Pass 1.
    undefined_refs: Vec<UndefinedRef>,

    /// Structured multiple-definition records for error reporting.
    multi_defs: Vec<MultiDefRecord>,

    /// Accumulated error messages during Pass 1 (for legacy string-based reporting).
    errors: Vec<String>,

    /// Supplementary diagnostic notes/warnings collected during resolution.
    supplementary_diagnostics: Vec<DiagnosticNote>,

    /// Object names for diagnostic messages (indexed by object_index).
    object_names: Vec<String>,

    /// Tracks which archive members have been marked for inclusion.
    included_objects: Vec<bool>,

    /// Target architecture for diagnostic context.
    target: Target,
}

impl SymbolResolver {
    /// Creates a new, empty symbol resolver with default target (`X86_64`).
    pub fn new() -> Self {
        Self {
            global_symbols: FxHashMap::default(),
            local_symbols: Vec::new(),
            undefined_refs: Vec::new(),
            multi_defs: Vec::new(),
            errors: Vec::new(),
            supplementary_diagnostics: Vec::new(),
            object_names: Vec::new(),
            included_objects: Vec::new(),
            target: Target::X86_64,
        }
    }

    /// Sets the target architecture for diagnostic context messages.
    ///
    /// The target is used to format architecture-specific linker error messages,
    /// allowing users to identify which architecture's linker encountered the
    /// error in cross-compilation scenarios.
    pub fn set_target(&mut self, target: Target) {
        self.target = target;
    }

    /// Returns the human-readable architecture name for the current target.
    ///
    /// Maps each `Target` variant to its canonical name string used in
    /// diagnostic output.
    fn target_arch_name(&self) -> &'static str {
        match self.target {
            Target::X86_64 => "x86-64",
            Target::I686 => "i686",
            Target::AArch64 => "aarch64",
            Target::RiscV64 => "riscv64",
        }
    }

    /// Registers an object name for diagnostic messages.
    ///
    /// Call this before `collect_symbols()` for each object to provide
    /// meaningful names in error messages (e.g., `"main.o"`, `"libfoo.a(bar.o)"`).
    pub fn register_object(&mut self, object_index: usize, name: &str) {
        // Extend the object_names vector to accommodate the given index.
        while self.object_names.len() <= object_index {
            self.object_names
                .push(format!("object_{}", self.object_names.len()));
        }
        self.object_names[object_index] = name.to_string();

        // Extend included_objects tracking vector.
        while self.included_objects.len() <= object_index {
            self.included_objects.push(true);
        }
    }

    /// Returns the name of the object at the given index (for diagnostics).
    /// Falls back to a generated name if no name was registered.
    fn object_name(&self, index: usize) -> String {
        self.object_names
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("object_{}", index))
    }

    // -----------------------------------------------------------------------
    // Pass 1 — Symbol Collection
    // -----------------------------------------------------------------------

    /// **Pass 1:** Collect all symbols from one input object file.
    ///
    /// Processes each symbol according to its binding:
    ///
    /// - **Local (defined):** stored per-object in `local_symbols` (not globally
    ///   visible). Multiple objects may define locals with the same name without
    ///   conflict.
    /// - **Global (defined):** inserted into `global_symbols`. If a `Global`
    ///   definition already exists → multiple definition error. If the existing
    ///   definition is `Weak` → the new strong definition overrides it. If the
    ///   new symbol is `Weak` and the existing is `Global` → keep the existing
    ///   strong definition.
    /// - **Undefined (section_index == 0):** recorded as a pending reference
    ///   for resolution in Pass 2.
    ///
    /// # Arguments
    ///
    /// * `object_index` — Zero-based index identifying this object file.
    /// * `symbols` — Slice of all symbols from this object's symbol table.
    pub fn collect_symbols(&mut self, object_index: usize, symbols: &[InputSymbol]) {
        // Extend local_symbols vector to accommodate this object.
        while self.local_symbols.len() <= object_index {
            self.local_symbols.push(Vec::new());
        }

        // Ensure a default object name exists for diagnostics.
        if self.object_names.len() <= object_index {
            self.register_object(object_index, &format!("object_{}", object_index));
        }

        for sym in symbols {
            // Skip null/anonymous symbols (empty name entries).
            if sym.name.is_empty() {
                continue;
            }

            // Check for undefined reference: section_index == 0 means SHN_UNDEF.
            // File-type symbols and locals with section_index 0 are not undefined refs.
            let is_undefined = sym.section_index == 0
                && sym.binding != SymbolBinding::Local
                && sym.sym_type != SymbolType::File;

            if is_undefined {
                self.undefined_refs.push(UndefinedRef {
                    name: sym.name.clone(),
                    object_index,
                    object_name: self.object_name(object_index),
                });
                continue;
            }

            // Build a SymbolEntry from the input symbol.
            let entry = SymbolEntry {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
                defining_object: object_index,
                is_defined: sym.section_index != 0,
            };

            match sym.binding {
                SymbolBinding::Local => {
                    // Local symbols are stored per-object, never globally visible.
                    self.local_symbols[object_index].push(entry);
                }
                SymbolBinding::Global | SymbolBinding::Weak => {
                    // Only defined symbols (section_index != 0) enter the global table.
                    if sym.section_index == 0 {
                        continue;
                    }
                    self.insert_global_symbol(entry);
                }
            }
        }
    }

    /// Inserts or merges a global/weak symbol into the global symbol table.
    ///
    /// Resolution rules (ELF specification):
    /// - New `Global` + existing `Global` → multiple definition error
    /// - New `Global` + existing `Weak` → new wins (strong overrides weak)
    /// - New `Weak` + existing `Global` → existing wins (keep strong)
    /// - New `Weak` + existing `Weak` → first wins (keep existing)
    ///
    /// Additionally checks for symbol type mismatches between definitions
    /// and records supplementary diagnostics for override events.
    fn insert_global_symbol(&mut self, new_entry: SymbolEntry) {
        let name = new_entry.name.clone();

        if let Some(existing) = self.global_symbols.get(&name) {
            // Check for symbol type mismatch (FUNC vs OBJECT, etc.)
            // NoType is compatible with any other type.
            if existing.sym_type != SymbolType::NoType
                && new_entry.sym_type != SymbolType::NoType
                && existing.sym_type != new_entry.sym_type
            {
                self.supplementary_diagnostics.push(DiagnosticNote {
                    is_warning: true,
                    message: format!(
                        "symbol type mismatch for `{}`: {} in {} vs {} in {}",
                        name,
                        existing.sym_type,
                        self.object_name(existing.defining_object),
                        new_entry.sym_type,
                        self.object_name(new_entry.defining_object),
                    ),
                });
            }

            // Merge visibility: most restrictive visibility wins.
            // Hidden < Protected < Default (in terms of restrictiveness).
            let merged_visibility =
                merge_visibility(existing.visibility, new_entry.visibility);

            match (existing.binding, new_entry.binding) {
                // Both strong → multiple definition error.
                (SymbolBinding::Global, SymbolBinding::Global) => {
                    if existing.is_defined && new_entry.is_defined {
                        let first_obj = self.object_name(existing.defining_object);
                        let second_obj = self.object_name(new_entry.defining_object);
                        let err_msg = format!(
                            "multiple definition of `{}`: first defined in {}, \
                             also defined in {}",
                            name, first_obj, second_obj,
                        );
                        self.errors.push(err_msg);
                        self.multi_defs.push(MultiDefRecord {
                            name: name.clone(),
                            defined_in: vec![first_obj, second_obj],
                        });
                    }
                    // If existing is undefined but new is defined, the defined one wins.
                    if !existing.is_defined && new_entry.is_defined {
                        let mut updated = new_entry;
                        updated.visibility = merged_visibility;
                        self.global_symbols.insert(name, updated);
                    }
                }
                // New is strong, existing is weak → new overrides.
                (SymbolBinding::Weak, SymbolBinding::Global) => {
                    self.supplementary_diagnostics.push(DiagnosticNote {
                        is_warning: false,
                        message: format!(
                            "strong definition of `{}` in {} overrides weak definition in {}",
                            name,
                            self.object_name(new_entry.defining_object),
                            self.object_name(existing.defining_object),
                        ),
                    });
                    let mut updated = new_entry;
                    updated.visibility = merged_visibility;
                    self.global_symbols.insert(name, updated);
                }
                // New is weak, existing is strong → keep existing.
                (SymbolBinding::Global, SymbolBinding::Weak) => {
                    // Keep the existing strong definition. Apply merged visibility.
                    if merged_visibility != existing.visibility {
                        let mut updated = existing.clone();
                        updated.visibility = merged_visibility;
                        self.global_symbols.insert(name, updated);
                    }
                }
                // Both weak → first wins (keep existing).
                (SymbolBinding::Weak, SymbolBinding::Weak) => {
                    // Keep the first weak definition encountered.
                    if merged_visibility != existing.visibility {
                        let mut updated = existing.clone();
                        updated.visibility = merged_visibility;
                        self.global_symbols.insert(name, updated);
                    }
                }
                // Local binding should never reach here (filtered in collect_symbols).
                _ => {}
            }
        } else {
            // First definition of this symbol — insert directly.
            self.global_symbols.insert(name, new_entry);
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2 — Reference Resolution
    // -----------------------------------------------------------------------

    /// **Pass 2:** Resolve all undefined references against collected definitions.
    ///
    /// Returns `Ok(ResolvedSymbols)` containing the final symbol table if all
    /// references are resolved successfully, or `Err(Vec<LinkError>)` if there
    /// are unresolved symbols, multiple definitions, or type mismatches.
    ///
    /// The resolved symbol table is ordered with local symbols first (in
    /// per-object order), followed by global/weak symbols sorted by name
    /// for deterministic ELF output.
    pub fn resolve_references(&mut self) -> Result<ResolvedSymbols, Vec<LinkError>> {
        let mut link_errors: Vec<LinkError> = Vec::new();

        // Convert structured multiple-definition records into LinkError variants.
        for mdef in &self.multi_defs {
            link_errors.push(LinkError::MultipleDefinition {
                name: mdef.name.clone(),
                defined_in: mdef.defined_in.clone(),
            });
        }

        // Collect truly unresolved references by checking each undefined ref
        // against the global symbol table.
        let mut unresolved: FxHashMap<String, Vec<String>> = FxHashMap::default();

        for undef_ref in &self.undefined_refs {
            if !self.global_symbols.contains_key(&undef_ref.name) {
                unresolved
                    .entry(undef_ref.name.clone())
                    .or_default()
                    .push(undef_ref.object_name.clone());
            }
        }

        // Generate LinkError for each truly unresolved symbol.
        // Use iter() to traverse the unresolved map for structured error generation.
        for (name, referenced_by) in unresolved.iter() {
            link_errors.push(LinkError::UndefinedSymbol {
                name: name.clone(),
                referenced_by: referenced_by.clone(),
            });
        }

        // Record object indices that have unresolved references for diagnostic context.
        let mut referencing_objects: Vec<usize> = self
            .undefined_refs
            .iter()
            .filter(|r| !self.global_symbols.contains_key(&r.name))
            .map(|r| r.object_index)
            .collect();
        referencing_objects.sort();
        referencing_objects.dedup();
        if !referencing_objects.is_empty() {
            let obj_names: Vec<String> = referencing_objects
                .iter()
                .map(|&idx| self.object_name(idx))
                .collect();
            self.supplementary_diagnostics.push(DiagnosticNote {
                is_warning: false,
                message: format!(
                    "objects with unresolved references: {}",
                    obj_names.join(", "),
                ),
            });
        }

        // Early return on errors — do not produce a symbol table if linking fails.
        if !link_errors.is_empty() {
            return Err(link_errors);
        }

        // Build the final resolved symbol table.
        let mut symbols = Vec::new();
        let mut symbol_map: FxHashMap<String, usize> = FxHashMap::default();

        // Phase 1: Add all local symbols (in object order, preserving per-object order).
        // Local symbols are namespaced by object index to avoid collisions.
        for obj_locals in &self.local_symbols {
            for local in obj_locals {
                let idx = symbols.len();
                // Local symbols use a mangled key to avoid name collisions.
                let mangled_key =
                    format!("__local_{}_{}", local.defining_object, local.name);
                symbol_map.insert(mangled_key, idx);
                symbols.push(local.clone());
            }
        }

        // Phase 2: Add all global/weak symbols using iter() for traversal.
        // Collect and sort by name for deterministic output.
        let mut global_entries: Vec<(&String, &SymbolEntry)> =
            self.global_symbols.iter().collect();
        global_entries.sort_by_key(|(name, _)| (*name).clone());

        for (name, entry) in global_entries {
            let idx = symbols.len();
            symbol_map.insert(name.clone(), idx);
            symbols.push(entry.clone());
        }

        Ok(ResolvedSymbols {
            symbols,
            symbol_map,
        })
    }

    // -----------------------------------------------------------------------
    // Archive Scanning
    // -----------------------------------------------------------------------

    /// Scans an archive (.a) symbol table, lazily including only those archive
    /// members that define currently-undefined symbols.
    ///
    /// This implements the standard UNIX linker archive semantics:
    /// - Iterate over archive symbol entries.
    /// - For each entry, if its name matches a currently-undefined symbol,
    ///   mark that archive member for inclusion.
    /// - The caller is responsible for loading the included archive members
    ///   and calling `collect_symbols()` with their symbol tables.
    ///
    /// # Arguments
    ///
    /// * `archive_symbols` — Slice of archive symbol table entries, each
    ///   mapping a symbol name to an archive member (object) index.
    pub fn scan_archive(&mut self, archive_symbols: &[ArchiveSymbol]) {
        // Build a set of currently undefined symbol names for O(1) lookup.
        let mut undefined_names: FxHashMap<String, bool> = FxHashMap::default();
        for undef_ref in &self.undefined_refs {
            if !self.global_symbols.contains_key(&undef_ref.name) {
                undefined_names.insert(undef_ref.name.clone(), true);
            }
        }

        // Identify which archive members need to be included.
        let mut included_indices: FxHashMap<usize, bool> = FxHashMap::default();
        for asym in archive_symbols {
            if undefined_names.contains_key(&asym.name) {
                included_indices.insert(asym.object_index, true);
            }
        }

        // Mark included archive members in the tracking vector.
        for asym in archive_symbols {
            if included_indices.contains_key(&asym.object_index) {
                while self.included_objects.len() <= asym.object_index {
                    self.included_objects.push(false);
                }
                self.included_objects[asym.object_index] = true;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Returns a deduplicated, sorted list of currently undefined symbol names.
    ///
    /// A symbol is "undefined" if it was referenced but not yet collected as
    /// a definition in the global symbol table.
    pub fn undefined_symbols(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .undefined_refs
            .iter()
            .filter(|r| !self.global_symbols.contains_key(&r.name))
            .map(|r| r.name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Returns the number of global symbols collected so far (Pass 1).
    pub fn global_symbol_count(&self) -> usize {
        self.global_symbols.len()
    }

    /// Returns the number of errors accumulated during Pass 1.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Checks whether a given archive member was marked for inclusion during
    /// archive scanning.
    pub fn is_archive_member_included(&self, object_index: usize) -> bool {
        self.included_objects
            .get(object_index)
            .copied()
            .unwrap_or(false)
    }

    /// Looks up a global symbol by name before resolution completes.
    ///
    /// Useful for early queries during archive scanning or diagnostic generation.
    pub fn lookup_global(&self, name: &str) -> Option<&SymbolEntry> {
        self.global_symbols.get(name)
    }

    /// Removes a global symbol by name from the global symbol table.
    ///
    /// Returns the removed `SymbolEntry` if it existed, or `None` otherwise.
    /// This is used for linker script `DISCARD` directives or garbage collection
    /// passes that remove unreferenced symbols.
    pub fn remove_global(&mut self, name: &str) -> Option<SymbolEntry> {
        self.global_symbols.remove(name)
    }

    // -----------------------------------------------------------------------
    // Diagnostics Integration
    // -----------------------------------------------------------------------

    /// Flushes all supplementary diagnostics to the provided `DiagnosticEngine`.
    ///
    /// Critical errors (multiple definitions, undefined symbols) are returned
    /// structurally from `resolve_references()` as `LinkError` variants. This
    /// method reports *supplementary* diagnostics collected during Pass 1:
    ///
    /// - **Errors:** Multiple-definition error messages with architecture context.
    /// - **Warnings:** Symbol type mismatches between definitions.
    /// - **Notes:** Strong-overrides-weak notifications for debugging link order.
    ///
    /// The target architecture name is included in error messages to help
    /// identify which architecture's linker encountered the issue.
    ///
    /// # Arguments
    ///
    /// * `diag` — The diagnostic engine to receive supplementary diagnostics.
    pub fn emit_diagnostics(&self, diag: &mut DiagnosticEngine) {
        let arch = self.target_arch_name();

        // Report critical multiple-definition errors through the diagnostic engine
        // with architecture context.
        for err_msg in &self.errors {
            diag.error(
                Span::DUMMY,
                format!("{} linker: {}", arch, err_msg),
            );
        }

        // Report supplementary diagnostics (warnings and notes).
        for note in &self.supplementary_diagnostics {
            if note.is_warning {
                diag.warning(
                    Span::DUMMY,
                    format!("{} linker: {}", arch, note.message),
                );
            } else {
                diag.note(
                    Span::DUMMY,
                    format!("{} linker: {}", arch, note.message),
                );
            }
        }

        // Log a summary note if the diagnostic engine already had errors from
        // earlier pipeline stages (e.g., compilation errors before linking).
        if diag.has_errors() && !self.errors.is_empty() {
            diag.note(
                Span::DUMMY,
                format!(
                    "{} linker: {} symbol resolution error(s) detected",
                    arch,
                    self.errors.len(),
                ),
            );
        }
    }
}

impl Default for SymbolResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Merges two symbol visibilities, returning the most restrictive.
///
/// ELF visibility merging rule: when multiple definitions provide different
/// visibilities, the most restrictive visibility wins.
/// Hidden > Protected > Default (in terms of restrictiveness).
fn merge_visibility(a: SymbolVisibility, b: SymbolVisibility) -> SymbolVisibility {
    // Assign a restrictiveness score (higher = more restrictive).
    let score = |v: SymbolVisibility| -> u8 {
        match v {
            SymbolVisibility::Default => 0,
            SymbolVisibility::Protected => 1,
            SymbolVisibility::Hidden => 2,
        }
    };
    if score(a) >= score(b) { a } else { b }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Creates a defined Global function symbol.
    fn global_sym(name: &str, value: u64, section: u16) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: section,
        }
    }

    /// Creates a defined Weak function symbol.
    fn weak_sym(name: &str, value: u64, section: u16) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value,
            size: 0,
            binding: SymbolBinding::Weak,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: section,
        }
    }

    /// Creates an undefined Global reference.
    fn undef_sym(name: &str) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: 0,
        }
    }

    /// Creates a defined Local data object symbol.
    fn local_sym(name: &str, value: u64, section: u16) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value,
            size: 0,
            binding: SymbolBinding::Local,
            sym_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
            section_index: section,
        }
    }

    /// Creates a defined Global symbol with a specific visibility.
    fn global_sym_with_vis(
        name: &str,
        value: u64,
        section: u16,
        vis: SymbolVisibility,
    ) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: vis,
            section_index: section,
        }
    }

    /// Creates a defined Global symbol with a specific type.
    fn global_sym_typed(
        name: &str,
        value: u64,
        section: u16,
        sym_type: SymbolType,
    ) -> InputSymbol {
        InputSymbol {
            name: name.to_string(),
            value,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type,
            visibility: SymbolVisibility::Default,
            section_index: section,
        }
    }

    // -----------------------------------------------------------------------
    // Pass 1 + Pass 2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_resolution() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");
        resolver.register_object(1, "lib.o");

        // main.o: defines main, references printf
        resolver.collect_symbols(
            0,
            &[global_sym("main", 0x1000, 1), undef_sym("printf")],
        );

        // lib.o: defines printf
        resolver.collect_symbols(1, &[global_sym("printf", 0x2000, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        assert!(resolved.get_symbol("main").is_some());
        assert!(resolved.get_symbol("printf").is_some());
        assert_eq!(resolved.get_symbol_value("main"), Some(0x1000));
        assert_eq!(resolved.get_symbol_value("printf"), Some(0x2000));
    }

    #[test]
    fn test_undefined_symbol_error() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");

        resolver.collect_symbols(0, &[undef_sym("missing_func")]);

        let result = resolver.resolve_references();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            LinkError::UndefinedSymbol {
                name,
                referenced_by,
            } => {
                assert_eq!(name, "missing_func");
                assert!(!referenced_by.is_empty());
                assert!(referenced_by.contains(&"main.o".to_string()));
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_multiple_undefined_references() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Both objects reference the same undefined symbol.
        resolver.collect_symbols(0, &[undef_sym("missing")]);
        resolver.collect_symbols(1, &[undef_sym("missing")]);

        let result = resolver.resolve_references();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            LinkError::UndefinedSymbol {
                name,
                referenced_by,
            } => {
                assert_eq!(name, "missing");
                assert_eq!(referenced_by.len(), 2);
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_multiple_definition_error() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Both objects define the same Global symbol.
        resolver.collect_symbols(0, &[global_sym("foo", 0x100, 1)]);
        resolver.collect_symbols(1, &[global_sym("foo", 0x200, 1)]);

        assert!(resolver.error_count() > 0);

        let result = resolver.resolve_references();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        let multi_def = errors
            .iter()
            .find(|e| matches!(e, LinkError::MultipleDefinition { .. }));
        assert!(multi_def.is_some());
        if let Some(LinkError::MultipleDefinition { name, defined_in }) = multi_def {
            assert_eq!(name, "foo");
            assert!(defined_in.contains(&"a.o".to_string()));
            assert!(defined_in.contains(&"b.o".to_string()));
        }
    }

    #[test]
    fn test_weak_overridden_by_strong() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "weak.o");
        resolver.register_object(1, "strong.o");

        // Weak definition followed by strong definition.
        resolver.collect_symbols(0, &[weak_sym("handler", 0x100, 1)]);
        resolver.collect_symbols(1, &[global_sym("handler", 0x200, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        // Strong definition wins.
        assert_eq!(resolved.get_symbol_value("handler"), Some(0x200));
    }

    #[test]
    fn test_strong_not_overridden_by_weak() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "strong.o");
        resolver.register_object(1, "weak.o");

        // Strong definition followed by weak definition.
        resolver.collect_symbols(0, &[global_sym("handler", 0x100, 1)]);
        resolver.collect_symbols(1, &[weak_sym("handler", 0x200, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        // Strong definition wins (first strong keeps).
        assert_eq!(resolved.get_symbol_value("handler"), Some(0x100));
    }

    #[test]
    fn test_two_weak_first_wins() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Two weak definitions — first one wins.
        resolver.collect_symbols(0, &[weak_sym("handler", 0x100, 1)]);
        resolver.collect_symbols(1, &[weak_sym("handler", 0x200, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        assert_eq!(resolved.get_symbol_value("handler"), Some(0x100));
    }

    #[test]
    fn test_local_symbols_not_globally_visible() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Both define a local symbol with same name — no conflict.
        resolver.collect_symbols(0, &[local_sym("helper", 0x100, 1)]);
        resolver.collect_symbols(1, &[local_sym("helper", 0x200, 1)]);

        assert_eq!(resolver.error_count(), 0);
        let resolved = resolver.resolve_references().unwrap();
        assert!(resolved.symbols.len() >= 2);
    }

    #[test]
    fn test_local_and_global_same_name() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Local in a.o and global in b.o with the same name — no conflict.
        resolver.collect_symbols(0, &[local_sym("data", 0x100, 1)]);
        resolver.collect_symbols(1, &[global_sym("data", 0x200, 1)]);

        assert_eq!(resolver.error_count(), 0);
        let resolved = resolver.resolve_references().unwrap();
        // Global symbol should be accessible by name.
        assert!(resolved.get_symbol("data").is_some());
        assert_eq!(resolved.get_symbol_value("data"), Some(0x200));
    }

    #[test]
    fn test_empty_name_symbols_skipped() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");

        let null_sym = InputSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: 0,
        };

        resolver.collect_symbols(0, &[null_sym, global_sym("main", 0x1000, 1)]);
        let resolved = resolver.resolve_references().unwrap();
        assert_eq!(resolver.global_symbol_count(), 1);
        assert!(resolved.get_symbol("main").is_some());
    }

    // -----------------------------------------------------------------------
    // Archive scanning tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_archive_scanning_basic() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");

        // main.o references foo and bar.
        resolver.collect_symbols(0, &[undef_sym("foo"), undef_sym("bar")]);

        // Archive has foo in member 1 and baz in member 2.
        let archive = vec![
            ArchiveSymbol {
                name: "foo".to_string(),
                object_index: 1,
                object_name: "libx.a(foo.o)".to_string(),
            },
            ArchiveSymbol {
                name: "baz".to_string(),
                object_index: 2,
                object_name: "libx.a(baz.o)".to_string(),
            },
        ];

        resolver.scan_archive(&archive);

        // Only member 1 (defining foo) should be included.
        assert!(resolver.is_archive_member_included(1));
        // Member 2 (defining baz) is not needed.
        assert!(!resolver.is_archive_member_included(2));
    }

    #[test]
    fn test_archive_scanning_no_matches() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");

        // main.o has no undefined symbols.
        resolver.collect_symbols(0, &[global_sym("main", 0x1000, 1)]);

        let archive = vec![ArchiveSymbol {
            name: "unused".to_string(),
            object_index: 1,
            object_name: "libx.a(unused.o)".to_string(),
        }];

        resolver.scan_archive(&archive);
        assert!(!resolver.is_archive_member_included(1));
    }

    // -----------------------------------------------------------------------
    // Query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_undefined_symbols_query() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");
        resolver.collect_symbols(
            0,
            &[undef_sym("alpha"), undef_sym("beta"), undef_sym("alpha")],
        );

        let undefs = resolver.undefined_symbols();
        assert!(undefs.contains(&"alpha".to_string()));
        assert!(undefs.contains(&"beta".to_string()));
        // Deduplicated: alpha appears only once.
        assert_eq!(
            undefs.iter().filter(|n| n.as_str() == "alpha").count(),
            1
        );
    }

    #[test]
    fn test_undefined_symbols_query_after_partial_resolution() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");
        resolver.register_object(1, "lib.o");

        resolver.collect_symbols(
            0,
            &[undef_sym("found"), undef_sym("missing")],
        );
        // lib.o defines "found" but not "missing".
        resolver.collect_symbols(1, &[global_sym("found", 0x2000, 1)]);

        let undefs = resolver.undefined_symbols();
        assert!(!undefs.contains(&"found".to_string()));
        assert!(undefs.contains(&"missing".to_string()));
    }

    #[test]
    fn test_global_symbol_count() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.collect_symbols(
            0,
            &[
                global_sym("foo", 0x100, 1),
                global_sym("bar", 0x200, 2),
                local_sym("baz", 0x300, 1),
            ],
        );
        // Only foo and bar are global; baz is local.
        assert_eq!(resolver.global_symbol_count(), 2);
    }

    #[test]
    fn test_lookup_global() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.collect_symbols(0, &[global_sym("main", 0x1000, 1)]);

        let sym = resolver.lookup_global("main");
        assert!(sym.is_some());
        assert_eq!(sym.unwrap().value, 0x1000);
        assert!(resolver.lookup_global("nonexistent").is_none());
    }

    #[test]
    fn test_remove_global() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.collect_symbols(
            0,
            &[global_sym("foo", 0x100, 1), global_sym("bar", 0x200, 2)],
        );

        assert_eq!(resolver.global_symbol_count(), 2);
        let removed = resolver.remove_global("foo");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().value, 0x100);
        assert_eq!(resolver.global_symbol_count(), 1);
        assert!(resolver.lookup_global("foo").is_none());
        assert!(resolver.remove_global("nonexistent").is_none());
    }

    // -----------------------------------------------------------------------
    // Visibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_visibility_merge_hidden_wins() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // a.o defines foo as Default, b.o defines foo as Hidden (weak).
        resolver.collect_symbols(
            0,
            &[global_sym_with_vis("foo", 0x100, 1, SymbolVisibility::Default)],
        );
        resolver.collect_symbols(
            1,
            &[InputSymbol {
                name: "foo".to_string(),
                value: 0x200,
                size: 0,
                binding: SymbolBinding::Weak,
                sym_type: SymbolType::Func,
                visibility: SymbolVisibility::Hidden,
                section_index: 1,
            }],
        );

        let resolved = resolver.resolve_references().unwrap();
        let sym = resolved.get_symbol("foo").unwrap();
        // Strong definition value kept, but visibility merges to most restrictive.
        assert_eq!(sym.value, 0x100);
        assert_eq!(sym.visibility, SymbolVisibility::Hidden);
    }

    #[test]
    fn test_visibility_merge_protected() {
        assert_eq!(
            merge_visibility(SymbolVisibility::Default, SymbolVisibility::Protected),
            SymbolVisibility::Protected,
        );
        assert_eq!(
            merge_visibility(SymbolVisibility::Protected, SymbolVisibility::Default),
            SymbolVisibility::Protected,
        );
        assert_eq!(
            merge_visibility(SymbolVisibility::Hidden, SymbolVisibility::Protected),
            SymbolVisibility::Hidden,
        );
    }

    // -----------------------------------------------------------------------
    // Target and diagnostics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_target() {
        let mut resolver = SymbolResolver::new();
        resolver.set_target(Target::AArch64);
        assert_eq!(resolver.target_arch_name(), "aarch64");

        resolver.set_target(Target::RiscV64);
        assert_eq!(resolver.target_arch_name(), "riscv64");

        resolver.set_target(Target::I686);
        assert_eq!(resolver.target_arch_name(), "i686");

        resolver.set_target(Target::X86_64);
        assert_eq!(resolver.target_arch_name(), "x86-64");
    }

    #[test]
    fn test_resolved_symbols_len_and_is_empty() {
        let mut resolver = SymbolResolver::new();
        let resolved = resolver.resolve_references().unwrap();
        assert!(resolved.is_empty());
        assert_eq!(resolved.len(), 0);

        resolver.register_object(0, "a.o");
        resolver.collect_symbols(0, &[global_sym("main", 0x1000, 1)]);
        let resolved = resolver.resolve_references().unwrap();
        assert!(!resolved.is_empty());
        assert_eq!(resolved.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Display implementation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_symbol_binding() {
        assert_eq!(format!("{}", SymbolBinding::Global), "GLOBAL");
        assert_eq!(format!("{}", SymbolBinding::Local), "LOCAL");
        assert_eq!(format!("{}", SymbolBinding::Weak), "WEAK");
    }

    #[test]
    fn test_display_symbol_visibility() {
        assert_eq!(format!("{}", SymbolVisibility::Default), "DEFAULT");
        assert_eq!(format!("{}", SymbolVisibility::Hidden), "HIDDEN");
        assert_eq!(format!("{}", SymbolVisibility::Protected), "PROTECTED");
    }

    #[test]
    fn test_display_symbol_type() {
        assert_eq!(format!("{}", SymbolType::NoType), "NOTYPE");
        assert_eq!(format!("{}", SymbolType::Object), "OBJECT");
        assert_eq!(format!("{}", SymbolType::Func), "FUNC");
        assert_eq!(format!("{}", SymbolType::Section), "SECTION");
        assert_eq!(format!("{}", SymbolType::File), "FILE");
    }

    #[test]
    fn test_display_link_error_undefined() {
        let err = LinkError::UndefinedSymbol {
            name: "missing".to_string(),
            referenced_by: vec!["main.o".to_string(), "util.o".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("missing"));
        assert!(msg.contains("main.o"));
        assert!(msg.contains("util.o"));
    }

    #[test]
    fn test_display_link_error_undefined_no_refs() {
        let err = LinkError::UndefinedSymbol {
            name: "orphan".to_string(),
            referenced_by: Vec::new(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("orphan"));
        assert!(!msg.contains("referenced by"));
    }

    #[test]
    fn test_display_link_error_multiple_def() {
        let err = LinkError::MultipleDefinition {
            name: "foo".to_string(),
            defined_in: vec!["a.o".to_string(), "b.o".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("foo"));
        assert!(msg.contains("a.o"));
        assert!(msg.contains("b.o"));
    }

    #[test]
    fn test_display_link_error_type_mismatch() {
        let err = LinkError::SymbolTypeMismatch {
            name: "data".to_string(),
            expected: SymbolType::Func,
            found: SymbolType::Object,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("data"));
        assert!(msg.contains("FUNC"));
        assert!(msg.contains("OBJECT"));
    }

    // -----------------------------------------------------------------------
    // Default trait test
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_resolver() {
        let resolver = SymbolResolver::default();
        assert_eq!(resolver.global_symbol_count(), 0);
        assert_eq!(resolver.error_count(), 0);
        assert!(resolver.undefined_symbols().is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge case: symbol type mismatch warning
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_mismatch_warning_during_collection() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // a.o defines foo as FUNC, b.o defines foo as OBJECT (weak).
        resolver.collect_symbols(
            0,
            &[global_sym_typed("foo", 0x100, 1, SymbolType::Func)],
        );
        resolver.collect_symbols(
            1,
            &[InputSymbol {
                name: "foo".to_string(),
                value: 0x200,
                size: 4,
                binding: SymbolBinding::Weak,
                sym_type: SymbolType::Object,
                visibility: SymbolVisibility::Default,
                section_index: 1,
            }],
        );

        // No hard error (strong definition kept), but a supplementary warning
        // was recorded.
        assert_eq!(resolver.error_count(), 0);
        assert!(!resolver.supplementary_diagnostics.is_empty());
        assert!(resolver.supplementary_diagnostics[0].is_warning);
    }

    // -----------------------------------------------------------------------
    // Complex multi-object resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_complex_multi_object() {
        let mut resolver = SymbolResolver::new();
        resolver.set_target(Target::RiscV64);
        resolver.register_object(0, "main.o");
        resolver.register_object(1, "lib_a.o");
        resolver.register_object(2, "lib_b.o");

        // main.o: defines _start, references init and cleanup
        resolver.collect_symbols(
            0,
            &[
                global_sym("_start", 0x1000, 1),
                undef_sym("init"),
                undef_sym("cleanup"),
            ],
        );

        // lib_a.o: defines init (weak), references helper
        resolver.collect_symbols(
            1,
            &[weak_sym("init", 0x2000, 1), undef_sym("helper")],
        );

        // lib_b.o: defines init (strong), cleanup, helper
        resolver.collect_symbols(
            2,
            &[
                global_sym("init", 0x3000, 1),
                global_sym("cleanup", 0x3100, 1),
                global_sym("helper", 0x3200, 2),
            ],
        );

        let resolved = resolver.resolve_references().unwrap();
        // Strong init (0x3000 from lib_b.o) overrides weak init (0x2000).
        assert_eq!(resolved.get_symbol_value("init"), Some(0x3000));
        assert_eq!(resolved.get_symbol_value("cleanup"), Some(0x3100));
        assert_eq!(resolved.get_symbol_value("helper"), Some(0x3200));
        assert_eq!(resolved.get_symbol_value("_start"), Some(0x1000));
    }
}
