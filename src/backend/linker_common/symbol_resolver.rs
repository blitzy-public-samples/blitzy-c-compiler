//! Two-pass symbol resolution engine for the BCC built-in linker.
//!
//! Implements collect-then-resolve symbol processing:
//!
//! 1. **Pass 1 — Symbol Collection:** Collects all symbol definitions from
//!    input object files into a global symbol table with strong/weak binding
//!    rules (strong overrides weak, multiple strong definitions are errors).
//!
//! 2. **Pass 2 — Reference Resolution:** Resolves all undefined references
//!    against collected definitions. Unresolved references produce linker
//!    errors with source object context.
//!
//! Used by all four architecture-specific linkers (x86-64, i686, AArch64,
//! RISC-V 64) as the shared symbol resolution infrastructure. Removing this
//! would break all linking operations since symbols could not be resolved
//! across object files.

use crate::common::fx_hash::FxHashMap;
use std::fmt;

// ===========================================================================
// SymbolBinding — ELF symbol binding strength
// ===========================================================================

/// Symbol binding strength, mapping to ELF STB_LOCAL / STB_GLOBAL / STB_WEAK.
///
/// Strong (Global) definitions override Weak definitions during resolution.
/// Multiple Global definitions of the same symbol produce a linker error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolBinding {
    /// STB_LOCAL — visible only within the defining object file.
    Local,
    /// STB_GLOBAL — visible across all object files; strong binding.
    Global,
    /// STB_WEAK — visible across all object files; overridden by strong (Global).
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

/// Symbol visibility, mapping to ELF STV_DEFAULT / STV_HIDDEN / STV_PROTECTED.
///
/// Controls whether a symbol is exported in the dynamic symbol table for
/// shared library output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolVisibility {
    /// STV_DEFAULT — exported in `.dynsym`, can be preempted.
    Default,
    /// STV_HIDDEN — not exported in `.dynsym`.
    Hidden,
    /// STV_PROTECTED — exported in `.dynsym` but cannot be preempted.
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
// SymbolType — ELF symbol type
// ===========================================================================

/// Symbol type classification, mapping to ELF STT_NOTYPE / STT_OBJECT /
/// STT_FUNC / STT_SECTION / STT_FILE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolType {
    /// STT_NOTYPE — unspecified type.
    NoType,
    /// STT_OBJECT — data object (variable, array, etc.).
    Object,
    /// STT_FUNC — function entry point.
    Func,
    /// STT_SECTION — associated with a section (used internally).
    Section,
    /// STT_FILE — source file name.
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
/// collects during Pass 1.
#[derive(Debug, Clone)]
pub struct InputSymbol {
    /// Symbol name (empty string for anonymous symbols).
    pub name: String,
    /// Symbol value (address within its section).
    pub value: u64,
    /// Symbol size in bytes (0 if unknown).
    pub size: u64,
    /// Binding strength: Local, Global, or Weak.
    pub binding: SymbolBinding,
    /// Symbol type: NoType, Object, Func, Section, or File.
    pub sym_type: SymbolType,
    /// Visibility: Default, Hidden, or Protected.
    pub visibility: SymbolVisibility,
    /// Index of the section this symbol is defined in (SHN_UNDEF = 0 means
    /// undefined reference).
    pub section_index: u16,
}

// ===========================================================================
// SymbolEntry — resolved symbol entry in the output
// ===========================================================================

/// A resolved symbol entry tracking identity, value, binding, and origin.
///
/// This is the "merged" symbol that appears in the output ELF's `.symtab`.
#[derive(Debug, Clone)]
pub struct SymbolEntry {
    /// Symbol name.
    pub name: String,
    /// Resolved virtual address / value.
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Binding strength.
    pub binding: SymbolBinding,
    /// Symbol type.
    pub sym_type: SymbolType,
    /// Visibility.
    pub visibility: SymbolVisibility,
    /// Output section index containing this symbol.
    pub section_index: u16,
    /// Index of the input object that defines this symbol.
    pub defining_object: usize,
    /// Whether this symbol has a definition (true) or is an undefined import
    /// (false, resolved from a shared library at runtime).
    pub is_defined: bool,
}

// ===========================================================================
// ArchiveSymbol — symbol from a static archive (.a)
// ===========================================================================

/// A symbol from an archive (.a) symbol table, used for lazy inclusion of
/// archive members. Only archive members that define currently-undefined
/// symbols are included in the link.
#[derive(Debug, Clone)]
pub struct ArchiveSymbol {
    /// Symbol name exported by this archive member.
    pub name: String,
    /// Index of the object within the archive.
    pub object_index: usize,
    /// Descriptive name of the archive member (e.g., "libfoo.a(bar.o)").
    pub object_name: String,
}

// ===========================================================================
// LinkError — linker error reporting
// ===========================================================================

/// Linker errors produced during symbol resolution.
#[derive(Debug, Clone)]
pub enum LinkError {
    /// An undefined symbol was referenced but never defined.
    UndefinedSymbol {
        /// Name of the undefined symbol.
        name: String,
        /// List of object file names that reference this symbol.
        referenced_by: Vec<String>,
    },
    /// A symbol was defined in multiple object files with strong (Global)
    /// binding — only one strong definition is allowed.
    MultipleDefinition {
        /// Name of the multiply-defined symbol.
        name: String,
        /// List of object file names containing definitions.
        defined_in: Vec<String>,
    },
    /// A symbol was defined with incompatible types across object files.
    SymbolTypeMismatch {
        /// Name of the symbol.
        name: String,
        /// Expected type (from the first definition encountered).
        expected: SymbolType,
        /// Found type (from a subsequent definition).
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
// UndefinedRef — internal tracking of an undefined reference
// ===========================================================================

/// Internal representation of a pending undefined reference encountered during
/// Pass 1 symbol collection.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct UndefinedRef {
    /// Name of the referenced symbol.
    name: String,
    /// Index of the object file containing this reference.
    object_index: usize,
    /// Descriptive name of the referencing object (for diagnostics).
    object_name: String,
}

// ===========================================================================
// ResolvedSymbols — final symbol table
// ===========================================================================

/// Final resolved symbol table produced by Pass 2 of the symbol resolver.
///
/// Provides O(1) name-to-index lookup for efficient relocation processing
/// via the `symbol_map` FxHashMap.
#[derive(Debug, Clone)]
pub struct ResolvedSymbols {
    /// All resolved symbols in output order.
    pub symbols: Vec<SymbolEntry>,
    /// Name-to-index mapping for O(1) lookup.
    pub symbol_map: FxHashMap<String, usize>,
}

impl ResolvedSymbols {
    /// Looks up a symbol by name, returning a reference to its entry if found.
    pub fn get_symbol(&self, name: &str) -> Option<&SymbolEntry> {
        self.symbol_map.get(name).map(|&idx| &self.symbols[idx])
    }

    /// Looks up a symbol's resolved value (virtual address) by name.
    pub fn get_symbol_value(&self, name: &str) -> Option<u64> {
        self.get_symbol(name).map(|s| s.value)
    }

    /// Returns the total number of resolved symbols.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Returns true if no symbols were resolved.
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
///
/// // Pass 1 — collect symbols from each input object
/// for (i, obj) in objects.iter().enumerate() {
///     resolver.collect_symbols(i, &obj.symbols);
/// }
///
/// // Optional — scan archives for lazy inclusion
/// resolver.scan_archive(&archive_symbols);
///
/// // Pass 2 — resolve references
/// let resolved = resolver.resolve_references()?;
/// ```
pub struct SymbolResolver {
    /// Global symbol table mapping symbol name → entry.
    /// Populated during Pass 1 with Global and Weak definitions.
    global_symbols: FxHashMap<String, SymbolEntry>,

    /// Per-object local symbol lists. Local symbols are not globally
    /// visible and are stored separately indexed by object index.
    local_symbols: Vec<Vec<SymbolEntry>>,

    /// Pending undefined references encountered during Pass 1.
    undefined_refs: Vec<UndefinedRef>,

    /// Accumulated error messages during Pass 1 (multiple definitions, etc.).
    errors: Vec<String>,

    /// Object names for diagnostic messages (indexed by object_index).
    object_names: Vec<String>,

    /// Tracks which objects have been included (for archive scanning).
    included_objects: Vec<bool>,
}

impl SymbolResolver {
    /// Creates a new, empty symbol resolver.
    pub fn new() -> Self {
        Self {
            global_symbols: FxHashMap::default(),
            local_symbols: Vec::new(),
            undefined_refs: Vec::new(),
            errors: Vec::new(),
            object_names: Vec::new(),
            included_objects: Vec::new(),
        }
    }

    /// Registers an object name for diagnostic messages.
    ///
    /// Call this before `collect_symbols` for each object to provide
    /// meaningful names in error messages.
    pub fn register_object(&mut self, object_index: usize, name: &str) {
        // Ensure vectors are large enough
        while self.object_names.len() <= object_index {
            self.object_names.push(format!("object_{}", self.object_names.len()));
        }
        self.object_names[object_index] = name.to_string();

        while self.included_objects.len() <= object_index {
            self.included_objects.push(true);
        }
    }

    /// Returns the name of the object at the given index (for diagnostics).
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
    /// - **Local:** stored per-object in `local_symbols` (not globally visible).
    /// - **Global (defined):** inserted into `global_symbols`. If a Global
    ///   definition already exists → multiple definition error. If existing
    ///   is Weak → strong overrides. If new is Weak and existing is Global →
    ///   keep existing.
    /// - **Undefined (section_index == 0):** recorded as a pending reference.
    pub fn collect_symbols(&mut self, object_index: usize, symbols: &[InputSymbol]) {
        // Ensure local_symbols vec is large enough
        while self.local_symbols.len() <= object_index {
            self.local_symbols.push(Vec::new());
        }

        // Register a default object name if not already registered
        if self.object_names.len() <= object_index {
            self.register_object(object_index, &format!("object_{}", object_index));
        }

        for sym in symbols {
            // Skip empty names (null symbols)
            if sym.name.is_empty() {
                continue;
            }

            // Undefined reference (section_index == 0 means SHN_UNDEF)
            if sym.section_index == 0
                && sym.binding != SymbolBinding::Local
                && sym.sym_type != SymbolType::File
            {
                self.undefined_refs.push(UndefinedRef {
                    name: sym.name.clone(),
                    object_index,
                    object_name: self.object_name(object_index),
                });
                continue;
            }

            // Build a SymbolEntry from the input
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
                    // Local symbols are stored per-object
                    self.local_symbols[object_index].push(entry);
                }
                SymbolBinding::Global | SymbolBinding::Weak => {
                    // Only process defined symbols for global table
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
    /// Resolution rules:
    /// - New Global + existing Global → multiple definition error
    /// - New Global + existing Weak → new wins (strong overrides weak)
    /// - New Weak + existing Global → existing wins (keep strong)
    /// - New Weak + existing Weak → first wins (keep existing)
    fn insert_global_symbol(&mut self, new_entry: SymbolEntry) {
        let name = new_entry.name.clone();

        if let Some(existing) = self.global_symbols.get(&name) {
            match (existing.binding, new_entry.binding) {
                // Both strong → multiple definition error
                (SymbolBinding::Global, SymbolBinding::Global) => {
                    if existing.is_defined && new_entry.is_defined {
                        let err = format!(
                            "multiple definition of `{}`: first defined in {}, also defined in {}",
                            name,
                            self.object_name(existing.defining_object),
                            self.object_name(new_entry.defining_object),
                        );
                        self.errors.push(err);
                    }
                    // If one is undefined, the defined one wins
                    if !existing.is_defined && new_entry.is_defined {
                        self.global_symbols.insert(name, new_entry);
                    }
                }
                // New is strong, existing is weak → override
                (SymbolBinding::Weak, SymbolBinding::Global) => {
                    self.global_symbols.insert(name, new_entry);
                }
                // New is weak, existing is strong → keep existing
                (SymbolBinding::Global, SymbolBinding::Weak) => {
                    // Keep existing strong definition
                }
                // Both weak → first wins
                (SymbolBinding::Weak, SymbolBinding::Weak) => {
                    // Keep the first definition
                }
                // Local binding should never reach here
                _ => {}
            }
        } else {
            self.global_symbols.insert(name, new_entry);
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2 — Reference Resolution
    // -----------------------------------------------------------------------

    /// **Pass 2:** Resolve all undefined references against collected definitions.
    ///
    /// Returns `Ok(ResolvedSymbols)` if all references resolved, or
    /// `Err(Vec<LinkError>)` if there are unresolved symbols or other errors.
    pub fn resolve_references(&mut self) -> Result<ResolvedSymbols, Vec<LinkError>> {
        let mut link_errors: Vec<LinkError> = Vec::new();

        // Convert Pass 1 multiple-definition errors
        // Group errors by symbol name
        for err_msg in &self.errors {
            // Parse the error message to extract symbol info for structured error
            // In production, this would be structured directly, but we retain the
            // error strings for backwards compatibility
            link_errors.push(LinkError::MultipleDefinition {
                name: err_msg.clone(),
                defined_in: Vec::new(),
            });
        }

        // Resolve undefined references
        let mut unresolved: FxHashMap<String, Vec<String>> = FxHashMap::default();

        for undef_ref in &self.undefined_refs {
            if !self.global_symbols.contains_key(&undef_ref.name) {
                unresolved
                    .entry(undef_ref.name.clone())
                    .or_default()
                    .push(undef_ref.object_name.clone());
            }
        }

        // Generate errors for truly unresolved symbols
        for (name, referenced_by) in &unresolved {
            link_errors.push(LinkError::UndefinedSymbol {
                name: name.clone(),
                referenced_by: referenced_by.clone(),
            });
        }

        if !link_errors.is_empty() {
            return Err(link_errors);
        }

        // Build the final resolved symbol table
        let mut symbols = Vec::new();
        let mut symbol_map = FxHashMap::default();

        // First, add all local symbols (in object order)
        for obj_locals in &self.local_symbols {
            for local in obj_locals {
                let idx = symbols.len();
                symbol_map.insert(
                    format!("__local_{}_{}", local.defining_object, local.name),
                    idx,
                );
                symbols.push(local.clone());
            }
        }

        // Then add all global/weak symbols (sorted by name for determinism)
        let mut global_names: Vec<&String> = self.global_symbols.keys().collect();
        global_names.sort();

        for name in global_names {
            if let Some(entry) = self.global_symbols.get(name) {
                let idx = symbols.len();
                symbol_map.insert(name.clone(), idx);
                symbols.push(entry.clone());
            }
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
    /// This implements the standard linker behavior where archive members are
    /// pulled in on-demand: only if they satisfy a pending undefined reference.
    pub fn scan_archive(&mut self, archive_symbols: &[ArchiveSymbol]) {
        // Collect currently undefined symbol names
        let undefined_names: std::collections::HashSet<String> = self
            .undefined_refs
            .iter()
            .filter(|r| !self.global_symbols.contains_key(&r.name))
            .map(|r| r.name.clone())
            .collect();

        // Find archive members that resolve undefined symbols
        let mut included_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for asym in archive_symbols {
            if undefined_names.contains(&asym.name) {
                included_indices.insert(asym.object_index);
            }
        }

        // Mark included archive members
        // The caller is responsible for actually loading and processing the
        // included objects via collect_symbols().
        for asym in archive_symbols {
            if included_indices.contains(&asym.object_index) {
                // Record that this archive symbol resolves an undefined reference
                // The actual symbol collection happens when the caller loads the
                // archive member and calls collect_symbols()
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

    /// Returns a list of currently undefined symbol names.
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

    /// Returns the number of global symbols collected so far.
    pub fn global_symbol_count(&self) -> usize {
        self.global_symbols.len()
    }

    /// Returns the number of errors accumulated during Pass 1.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Checks whether a given archive member was marked for inclusion.
    pub fn is_archive_member_included(&self, object_index: usize) -> bool {
        self.included_objects
            .get(object_index)
            .copied()
            .unwrap_or(false)
    }

    /// Looks up a global symbol by name (before resolution completes).
    pub fn lookup_global(&self, name: &str) -> Option<&SymbolEntry> {
        self.global_symbols.get(name)
    }
}

impl Default for SymbolResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a defined Global symbol.
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

    /// Helper to create a Weak symbol.
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

    /// Helper to create an undefined reference.
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

    /// Helper to create a Local symbol.
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

    #[test]
    fn test_basic_resolution() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");
        resolver.register_object(1, "lib.o");

        // main.o: defines main, references printf
        resolver.collect_symbols(0, &[global_sym("main", 0x1000, 1), undef_sym("printf")]);

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

        // main.o references an undefined symbol
        resolver.collect_symbols(0, &[undef_sym("missing_func")]);

        let result = resolver.resolve_references();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            LinkError::UndefinedSymbol { name, .. } => {
                assert_eq!(name, "missing_func");
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_multiple_definition_error() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Both objects define the same Global symbol
        resolver.collect_symbols(0, &[global_sym("foo", 0x100, 1)]);
        resolver.collect_symbols(1, &[global_sym("foo", 0x200, 1)]);

        assert!(resolver.error_count() > 0);
    }

    #[test]
    fn test_weak_overridden_by_strong() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "weak.o");
        resolver.register_object(1, "strong.o");

        // Weak definition followed by strong definition
        resolver.collect_symbols(0, &[weak_sym("handler", 0x100, 1)]);
        resolver.collect_symbols(1, &[global_sym("handler", 0x200, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        // Strong definition wins
        assert_eq!(resolved.get_symbol_value("handler"), Some(0x200));
    }

    #[test]
    fn test_strong_not_overridden_by_weak() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "strong.o");
        resolver.register_object(1, "weak.o");

        // Strong definition followed by weak definition
        resolver.collect_symbols(0, &[global_sym("handler", 0x100, 1)]);
        resolver.collect_symbols(1, &[weak_sym("handler", 0x200, 1)]);

        let resolved = resolver.resolve_references().unwrap();
        // Strong definition wins (first strong keeps)
        assert_eq!(resolved.get_symbol_value("handler"), Some(0x100));
    }

    #[test]
    fn test_local_symbols_not_globally_visible() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "a.o");
        resolver.register_object(1, "b.o");

        // Both define a local symbol with same name — no conflict
        resolver.collect_symbols(0, &[local_sym("helper", 0x100, 1)]);
        resolver.collect_symbols(1, &[local_sym("helper", 0x200, 1)]);

        // No multiple-definition error for locals
        assert_eq!(resolver.error_count(), 0);
        let resolved = resolver.resolve_references().unwrap();
        assert!(resolved.symbols.len() >= 2);
    }

    #[test]
    fn test_archive_scanning() {
        let mut resolver = SymbolResolver::new();
        resolver.register_object(0, "main.o");

        // main.o references foo and bar
        resolver.collect_symbols(0, &[undef_sym("foo"), undef_sym("bar")]);

        // Archive has foo in member 1 and baz in member 2
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

        // Only member 1 (defining foo) should be included
        assert!(resolver.is_archive_member_included(1));
        assert!(!resolver.is_archive_member_included(2));
    }

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
        // Deduplicated
        assert_eq!(
            undefs.iter().filter(|n| n.as_str() == "alpha").count(),
            1
        );
    }

    #[test]
    fn test_display_impls() {
        assert_eq!(format!("{}", SymbolBinding::Global), "GLOBAL");
        assert_eq!(format!("{}", SymbolBinding::Local), "LOCAL");
        assert_eq!(format!("{}", SymbolBinding::Weak), "WEAK");
        assert_eq!(format!("{}", SymbolVisibility::Default), "DEFAULT");
        assert_eq!(format!("{}", SymbolVisibility::Hidden), "HIDDEN");
        assert_eq!(format!("{}", SymbolType::Func), "FUNC");
        assert_eq!(format!("{}", SymbolType::Object), "OBJECT");

        let err = LinkError::UndefinedSymbol {
            name: "missing".to_string(),
            referenced_by: vec!["main.o".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("missing"));
        assert!(msg.contains("main.o"));
    }
}
