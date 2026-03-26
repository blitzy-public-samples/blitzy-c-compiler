//! Symbol table for Phase 5 (semantic analysis) of the BCC C11 compiler.
//!
//! Manages symbol entries with name (interned [`Symbol`]), type ([`CType`]),
//! linkage ([`Linkage`]: external/internal/none), storage class
//! ([`StorageClass`]: auto/register/static/extern/typedef), definition vs.
//! declaration tracking, and `weak` attribute handling for ELF symbol binding.
//!
//! Uses [`FxHashMap`] from `crate::common::fx_hash` for O(1) average-case
//! lookup performance, critical for kernel-scale compilation where symbol
//! tables may contain tens of thousands of entries.
//!
//! # C11 Conformance
//!
//! - **Tentative definitions** (C11 §6.9.2): file-scope variable declarations
//!   without an initialiser are tentative; if no actual definition appears by
//!   end of translation unit, the tentative definition becomes an actual
//!   definition with a zero initialiser.
//!
//! - **Linkage** (C11 §6.2.2): external, internal, or none — resolved from
//!   scope depth, storage class, and prior declarations.
//!
//! - **Redeclaration rules**: same-scope redeclaration of the same identifier
//!   is an error unless both are compatible `extern` declarations or a prior
//!   declaration is being completed by a definition.
//!
//! # Architecture
//!
//! This module does **NOT** depend on `crate::ir`, `crate::passes`, or
//! `crate::backend`. It is a pure frontend data structure consumed by the
//! semantic analyser and later read (but not mutated) by the IR lowering phase.
//!
//! # Zero-Dependency Compliance
//!
//! No external crates. Only `std` and `crate::` references.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::Symbol;
use crate::common::types::CType;
use crate::frontend::sema::attribute_handler::{SymbolVisibility, ValidatedAttribute};

// ===========================================================================
// SymbolId — unique handle for a symbol table entry
// ===========================================================================

/// Unique identifier for a symbol entry within a [`SymbolTable`].
///
/// The inner `u32` is the index into the table's internal `Vec<SymbolEntry>`.
/// Once allocated, a `SymbolId` is stable for the lifetime of the table —
/// leaving a scope removes the name mapping but the entry remains accessible
/// by ID.
///
/// # Ordering
///
/// `SymbolId` values are assigned in declaration order (monotonically
/// increasing). Earlier declarations receive lower IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

impl SymbolId {
    /// Returns the raw `u32` index backing this symbol ID.
    ///
    /// Useful for serialisation, diagnostics, or external data structures
    /// that need a numeric handle.
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

// ===========================================================================
// SymbolKind — classification of declared identifiers
// ===========================================================================

/// Classification of a symbol in the ordinary identifier namespace.
///
/// C11 has four kinds of identifiers in the ordinary namespace:
/// - object (variable)
/// - function
/// - typedef name
/// - enumeration constant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// A variable (object) symbol — includes file-scope globals, block-scope
    /// locals, function parameters, and static locals.
    Variable,
    /// A function symbol — includes both definitions and declarations
    /// (prototypes).
    Function,
    /// A typedef name — introduces a type alias in the ordinary namespace.
    TypedefName,
    /// An enumeration constant — a named integer value declared inside an
    /// `enum` definition.
    EnumConstant,
}

// ===========================================================================
// Linkage — C11 §6.2.2
// ===========================================================================

/// Linkage classification per C11 §6.2.2.
///
/// Linkage determines whether two declarations of the same identifier in
/// different scopes or translation units refer to the same entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// **External linkage** — the identifier denotes the same entity across
    /// all translation units. This is the default for functions and file-scope
    /// variables without `static`.
    External,
    /// **Internal linkage** — the identifier is visible only within the
    /// current translation unit. Applied by `static` at file scope.
    Internal,
    /// **No linkage** — the identifier denotes a unique entity. Block-scope
    /// variables (except `extern`), function parameters, typedefs, and
    /// enum constants have no linkage.
    None,
}

// ===========================================================================
// StorageClass — C11 §6.7.1
// ===========================================================================

/// Storage class specifier per C11 §6.7.1.
///
/// At most one storage class specifier may appear in a declaration, except
/// that `_Thread_local` may combine with `static` or `extern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// `auto` — block-scope automatic storage duration (the default for local
    /// variables when no specifier is given).
    Auto,
    /// `register` — block-scope with a hint for register allocation; the
    /// address of a `register` variable cannot be taken.
    Register,
    /// `static` — static storage duration. At file scope, also implies
    /// internal linkage.
    Static,
    /// `extern` — external declaration; the definition is expected elsewhere.
    Extern,
    /// `typedef` — introduces a type alias, not a variable or function.
    Typedef,
    /// `_Thread_local` — thread-local storage duration (C11 §6.7.1p3).
    ThreadLocal,
}

// ===========================================================================
// SymbolEntry — full description of a declared identifier
// ===========================================================================

/// Complete metadata for a single symbol declaration.
///
/// Every identifier in the ordinary namespace — variables, functions,
/// typedefs, and enum constants — is represented by a `SymbolEntry`.
///
/// # Lifecycle
///
/// 1. A `SymbolEntry` is created via [`SymbolTable::declare`] (or one of the
///    convenience methods like [`SymbolTable::declare_function`]).
/// 2. The entry's `is_defined` flag is initially `false` for forward
///    declarations and `true` for definitions.
/// 3. [`SymbolTable::define`] promotes a declaration to a definition.
/// 4. At end of translation unit, [`SymbolTable::finalize_tentative_definitions`]
///    converts file-scope tentative definitions (§6.9.2) into actual definitions.
/// 5. [`SymbolTable::check_unused_symbols`] emits warnings for unused entries.
#[derive(Debug, Clone)]
pub struct SymbolEntry {
    /// Interned name handle for O(1) comparison.
    pub name: Symbol,
    /// C type of this symbol (e.g. `CType::Int`, `CType::Function { .. }`).
    pub ty: CType,
    /// What kind of identifier this is.
    pub kind: SymbolKind,
    /// Linkage classification — external, internal, or none.
    pub linkage: Linkage,
    /// Storage class specifier from the declaration.
    pub storage_class: StorageClass,
    /// `true` if this entry represents a definition (not just a declaration).
    pub is_defined: bool,
    /// `true` if this is a tentative definition (C11 §6.9.2).
    pub is_tentative: bool,
    /// Source span of the declaration/definition site.
    pub span: Span,
    /// Validated GCC `__attribute__` annotations applied to this symbol.
    pub attributes: Vec<ValidatedAttribute>,
    /// `true` if `__attribute__((weak))` is applied — `STB_WEAK` in ELF.
    pub is_weak: bool,
    /// ELF symbol visibility (`STV_DEFAULT`, `STV_HIDDEN`, etc.).
    pub visibility: Option<SymbolVisibility>,
    /// ELF section override from `__attribute__((section("...")))`.
    pub section: Option<String>,
    /// Whether the symbol has been referenced/used (for `-Wunused`).
    pub is_used: bool,
    /// Scope depth at which this symbol was declared (0 = file scope).
    pub scope_depth: u32,
}

// ===========================================================================
// SymbolTable — the main data structure
// ===========================================================================

/// The symbol table for a single translation unit.
///
/// # Internal Layout
///
/// - `symbols: Vec<SymbolEntry>` — arena-style storage indexed by [`SymbolId`].
///   Entries are never removed; leaving a scope only removes the name mapping.
///
/// - `name_to_ids: FxHashMap<Symbol, Vec<SymbolId>>` — maps each interned name
///   to a stack of symbol IDs ordered by declaration (most recent last).
///   When a scope is exited, IDs belonging to that scope are popped from the
///   vectors. This gives correct shadowing semantics.
///
/// - `current_scope_depth: u32` — tracks lexical nesting depth. `0` represents
///   file scope; each `enter_scope()` increments, each `leave_scope()`
///   decrements.
pub struct SymbolTable {
    /// All symbol entries, indexed by [`SymbolId`].
    symbols: Vec<SymbolEntry>,
    /// Name → list of symbol IDs, supporting shadowing across scopes.
    name_to_ids: FxHashMap<Symbol, Vec<SymbolId>>,
    /// Current lexical scope depth (0 = file scope).
    current_scope_depth: u32,
}

impl SymbolTable {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Create a new, empty symbol table at file scope (depth 0).
    pub fn new() -> Self {
        SymbolTable {
            symbols: Vec::new(),
            name_to_ids: FxHashMap::default(),
            current_scope_depth: 0,
        }
    }

    // ===================================================================
    // Symbol Declaration
    // ===================================================================

    /// Declare a symbol, performing redeclaration conflict checking.
    ///
    /// # Redeclaration Rules (C11 §6.7p2, §6.2.2)
    ///
    /// When the same name already exists **in the same scope**:
    ///
    /// - Two `extern` declarations with compatible types → OK (merge).
    /// - An `extern` declaration followed by a definition with a compatible
    ///   type → OK (promote to definition).
    /// - Two definitions → **error** `"redefinition of 'name'"`.
    /// - Declarations with incompatible types → **error**
    ///   `"conflicting types for 'name'"`.
    ///
    /// When the same name exists in an **outer scope**, the new declaration
    /// shadows the outer one. Shadowing is legal per C11.
    ///
    /// # Returns
    ///
    /// `Ok(SymbolId)` on success, `Err(())` if a fatal redeclaration error
    /// was emitted.
    pub fn declare(
        &mut self,
        mut entry: SymbolEntry,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<SymbolId, ()> {
        entry.scope_depth = self.current_scope_depth;

        // Check for an existing declaration of the same name in the current scope.
        if let Some(existing_id) = self.find_in_current_scope(entry.name) {
            let existing = &self.symbols[existing_id.0 as usize];

            // Typedef redeclaration: same type is OK (C11 §6.7p3).
            if existing.kind == SymbolKind::TypedefName && entry.kind == SymbolKind::TypedefName {
                if self.types_are_compatible(&existing.ty, &entry.ty) {
                    // Benign re-typedef — just return the existing ID.
                    return Ok(existing_id);
                } else {
                    diagnostics.emit(
                        Diagnostic::error(entry.span, "conflicting types for typedef".to_string())
                            .with_note(existing.span, "previous declaration is here"),
                    );
                    return Err(());
                }
            }

            // Enum constant redeclaration is always an error.
            if existing.kind == SymbolKind::EnumConstant || entry.kind == SymbolKind::EnumConstant {
                diagnostics.emit(
                    Diagnostic::error(entry.span, "redefinition of enumerator".to_string())
                        .with_note(existing.span, "previous definition is here"),
                );
                return Err(());
            }

            // Check type compatibility.
            if !self.types_are_compatible(&existing.ty, &entry.ty) {
                diagnostics.emit(
                    Diagnostic::error(entry.span, "conflicting types for symbol".to_string())
                        .with_note(existing.span, "previous declaration is here"),
                );
                return Err(());
            }

            // Both extern declarations with compatible types → merge.
            if existing.storage_class == StorageClass::Extern
                && entry.storage_class == StorageClass::Extern
                && !existing.is_defined
                && !entry.is_defined
            {
                // Merge attributes from the new declaration.
                let sym = &mut self.symbols[existing_id.0 as usize];
                Self::merge_attributes(sym, &entry.attributes);
                return Ok(existing_id);
            }

            // Existing is a declaration, new entry is a definition → promote.
            if !existing.is_defined && entry.is_defined {
                let sym = &mut self.symbols[existing_id.0 as usize];
                sym.is_defined = true;
                sym.is_tentative = false;
                sym.span = entry.span;
                Self::merge_attributes(sym, &entry.attributes);
                if entry.is_weak {
                    sym.is_weak = true;
                }
                if entry.visibility.is_some() {
                    sym.visibility = entry.visibility;
                }
                if entry.section.is_some() {
                    sym.section = entry.section;
                }
                return Ok(existing_id);
            }

            // Existing is tentative definition and new entry is compatible
            // extern declaration → OK (C11 §6.9.2p2).
            if existing.is_tentative && entry.storage_class == StorageClass::Extern {
                let sym = &mut self.symbols[existing_id.0 as usize];
                Self::merge_attributes(sym, &entry.attributes);
                return Ok(existing_id);
            }

            // Both tentative definitions → valid per C11, merge.
            if existing.is_tentative && entry.is_tentative {
                let sym = &mut self.symbols[existing_id.0 as usize];
                sym.span = entry.span;
                Self::merge_attributes(sym, &entry.attributes);
                return Ok(existing_id);
            }

            // Existing forward declaration (not defined) and new is also not
            // a definition (just a redeclaration) → merge if compatible.
            if !existing.is_defined && !entry.is_defined {
                let sym = &mut self.symbols[existing_id.0 as usize];
                Self::merge_attributes(sym, &entry.attributes);
                return Ok(existing_id);
            }

            // Two actual definitions in the same scope → error.
            if existing.is_defined && entry.is_defined {
                diagnostics.emit(
                    Diagnostic::error(entry.span, "redefinition of symbol".to_string())
                        .with_note(existing.span, "previous definition is here"),
                );
                return Err(());
            }

            // Existing definition + new declaration → OK, just merge attrs.
            if existing.is_defined && !entry.is_defined {
                let sym = &mut self.symbols[existing_id.0 as usize];
                Self::merge_attributes(sym, &entry.attributes);
                return Ok(existing_id);
            }
        }

        // No conflict — allocate a fresh SymbolId and insert.
        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(entry);
        let name = self.symbols[id.0 as usize].name;
        self.name_to_ids.entry(name).or_default().push(id);
        Ok(id)
    }

    // ===================================================================
    // Definition Promotion
    // ===================================================================

    /// Mark an existing declaration as defined.
    ///
    /// If the symbol was a tentative definition, it is promoted to an
    /// actual definition. The `is_tentative` flag is cleared.
    pub fn define(&mut self, id: SymbolId) {
        let entry = &mut self.symbols[id.0 as usize];
        entry.is_defined = true;
        entry.is_tentative = false;
    }

    // ===================================================================
    // Lookup
    // ===================================================================

    /// Look up the most recent (innermost-scope) declaration of `name`.
    ///
    /// Returns `None` if no visible declaration exists.
    ///
    /// # Performance
    ///
    /// O(1) average-case via FxHashMap + a reverse scan of the per-name ID
    /// list (typically very short — one or two entries for most identifiers).
    pub fn lookup(&self, name: Symbol) -> Option<&SymbolEntry> {
        self.name_to_ids
            .get(&name)
            .and_then(|ids| ids.last().map(|id| &self.symbols[id.0 as usize]))
    }

    /// Look up a declaration of `name` **only** in the current scope depth.
    ///
    /// Used for redeclaration checking — determines whether the new
    /// declaration conflicts with an existing one at the same scope level.
    pub fn lookup_in_current_scope(&self, name: Symbol) -> Option<&SymbolEntry> {
        self.name_to_ids.get(&name).and_then(|ids| {
            ids.iter().rev().find_map(|id| {
                let entry = &self.symbols[id.0 as usize];
                if entry.scope_depth == self.current_scope_depth {
                    Some(entry)
                } else {
                    Option::None
                }
            })
        })
    }

    /// Mutable lookup of the most recent declaration of `name`.
    ///
    /// Useful for updating symbol properties (e.g. marking as used,
    /// applying late attributes).
    pub fn lookup_mut(&mut self, name: Symbol) -> Option<&mut SymbolEntry> {
        if let Some(ids) = self.name_to_ids.get(&name) {
            if let Some(id) = ids.last() {
                let idx = id.0 as usize;
                return Some(&mut self.symbols[idx]);
            }
        }
        Option::None
    }

    /// Direct lookup by [`SymbolId`].
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of range (debug assertion).
    #[inline]
    pub fn get(&self, id: SymbolId) -> &SymbolEntry {
        debug_assert!(
            (id.0 as usize) < self.symbols.len(),
            "SymbolId({}) out of range for symbol table with {} entries",
            id.0,
            self.symbols.len()
        );
        &self.symbols[id.0 as usize]
    }

    /// Mutable direct lookup by [`SymbolId`].
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of range (debug assertion).
    #[inline]
    pub fn get_mut(&mut self, id: SymbolId) -> &mut SymbolEntry {
        debug_assert!(
            (id.0 as usize) < self.symbols.len(),
            "SymbolId({}) out of range for symbol table with {} entries",
            id.0,
            self.symbols.len()
        );
        &mut self.symbols[id.0 as usize]
    }

    // ===================================================================
    // Scope Management
    // ===================================================================

    /// Enter a new lexical scope.
    ///
    /// Increments the scope depth counter. All subsequent declarations
    /// will be associated with this new depth level.
    #[inline]
    pub fn enter_scope(&mut self) {
        self.current_scope_depth += 1;
    }

    /// Leave the current lexical scope.
    ///
    /// Decrements the scope depth counter and removes symbol-name mappings
    /// that belong to the scope being exited. The underlying
    /// [`SymbolEntry`] objects remain in the `symbols` vector (so that
    /// [`SymbolId`]s remain stable), but they are no longer discoverable
    /// via name lookup.
    ///
    /// This is critical for correct shadowing: once a block-scope variable
    /// goes out of scope, any outer-scope variable of the same name
    /// becomes visible again.
    pub fn leave_scope(&mut self) {
        let depth = self.current_scope_depth;

        // Prune name_to_ids entries that belong to the exiting scope.
        // We iterate over all names and pop any IDs whose scope_depth
        // matches the current depth.
        //
        // For large programs this could be optimised by maintaining a
        // per-scope list of names to clean up. The current approach is
        // simple and correct, and in practice the number of names per
        // scope is small.
        let symbols = &self.symbols;
        self.name_to_ids.retain(|_name, ids| {
            // Remove all IDs declared at the exiting scope depth.
            ids.retain(|id| symbols[id.0 as usize].scope_depth != depth);
            // Keep the entry if there are still IDs remaining.
            !ids.is_empty()
        });

        if self.current_scope_depth > 0 {
            self.current_scope_depth -= 1;
        }
    }

    // ===================================================================
    // Linkage Resolution (C11 §6.2.2)
    // ===================================================================

    /// Determine the linkage for a symbol based on its storage class
    /// specifier, scope depth, and any prior visible declaration.
    ///
    /// # Linkage Rules (C11 §6.2.2)
    ///
    /// | Scope | Storage Class | Linkage |
    /// |-------|--------------|---------|
    /// | File  | `static`     | Internal |
    /// | File  | *(none)*     | External |
    /// | File  | `extern`     | External (or Internal if prior `static`) |
    /// | Block | `extern`     | External (matches file-scope if any) |
    /// | Block | *(none)*     | None |
    /// | Block | `static`     | None (static duration, no linkage) |
    /// | Any   | `typedef`    | None |
    /// | Any   | `register`   | None |
    pub fn resolve_linkage(
        &self,
        name: Symbol,
        storage: StorageClass,
        scope_depth: u32,
    ) -> Linkage {
        // Typedefs and register variables never have linkage.
        match storage {
            StorageClass::Typedef | StorageClass::Register => return Linkage::None,
            _ => {}
        }

        let is_file_scope = scope_depth == 0;

        if is_file_scope {
            match storage {
                StorageClass::Static => Linkage::Internal,
                StorageClass::Extern => {
                    // If a prior visible declaration of the same name exists
                    // with internal linkage, the extern declaration inherits
                    // internal linkage (C11 §6.2.2p4).
                    if let Some(prior) = self.lookup(name) {
                        if prior.linkage == Linkage::Internal {
                            return Linkage::Internal;
                        }
                    }
                    Linkage::External
                }
                // Auto, Register, ThreadLocal at file scope default to external
                // for functions and variables per C11 §6.2.2p5.
                _ => Linkage::External,
            }
        } else {
            // Block scope.
            match storage {
                StorageClass::Extern => {
                    // Block-scope extern inherits the linkage of a prior
                    // file-scope declaration if one exists.
                    if let Some(prior) = self.lookup(name) {
                        if prior.linkage == Linkage::Internal || prior.linkage == Linkage::External
                        {
                            return prior.linkage;
                        }
                    }
                    Linkage::External
                }
                StorageClass::Static => {
                    // Block-scope static has no linkage (but static storage
                    // duration).
                    Linkage::None
                }
                _ => Linkage::None,
            }
        }
    }

    /// Check linkage compatibility between an existing and a new declaration.
    ///
    /// # Rules
    ///
    /// - Both external: types must be compatible.
    /// - Existing internal + new external: error (C11 §6.2.2p7).
    /// - Prior `static` at file scope overrides subsequent `extern`.
    pub fn check_linkage_compatibility(
        &self,
        existing: &SymbolEntry,
        new_entry: &SymbolEntry,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<(), ()> {
        // If existing has internal linkage and new has external → error.
        if existing.linkage == Linkage::Internal && new_entry.linkage == Linkage::External {
            diagnostics.emit(
                Diagnostic::error(
                    new_entry.span,
                    "non-static declaration follows static declaration",
                )
                .with_note(existing.span, "previous declaration is here"),
            );
            return Err(());
        }

        // Both external linkage: types must be compatible.
        if existing.linkage == Linkage::External
            && new_entry.linkage == Linkage::External
            && !self.types_are_compatible(&existing.ty, &new_entry.ty)
        {
            diagnostics.emit(
                Diagnostic::error(
                    new_entry.span,
                    "conflicting types for symbol with external linkage",
                )
                .with_note(existing.span, "previous declaration is here"),
            );
            return Err(());
        }

        Ok(())
    }

    // ===================================================================
    // Tentative Definition Handling (C11 §6.9.2)
    // ===================================================================

    /// Finalise tentative definitions at the end of a translation unit.
    ///
    /// Per C11 §6.9.2p2: if a file-scope variable has one or more tentative
    /// definitions and no actual definition, the compiler acts as though a
    /// file-scope definition with zero initialiser appeared.
    ///
    /// Multiple tentative definitions of the same identifier are valid and
    /// are merged into a single definition.
    pub fn finalize_tentative_definitions(&mut self, _diagnostics: &mut DiagnosticEngine) {
        for entry in &mut self.symbols {
            // Only file-scope variables with tentative status.
            if entry.scope_depth == 0
                && entry.kind == SymbolKind::Variable
                && entry.is_tentative
                && !entry.is_defined
            {
                // Promote tentative definition to actual definition with
                // implicit zero initialiser.
                entry.is_defined = true;
                entry.is_tentative = false;
            }
        }
    }

    // ===================================================================
    // Weak Symbol Support
    // ===================================================================

    /// Mark a symbol as having weak linkage (`STB_WEAK` in ELF).
    ///
    /// Weak definitions can be overridden by strong definitions at link time.
    /// This is applied when `__attribute__((weak))` is present.
    pub fn mark_weak(&mut self, id: SymbolId) {
        self.symbols[id.0 as usize].is_weak = true;
    }

    // ===================================================================
    // Usage Tracking
    // ===================================================================

    /// Mark a symbol as used/referenced.
    ///
    /// Called whenever an identifier is referenced in an expression context.
    /// Symbols that are never marked as used may generate `-Wunused`
    /// warnings at scope exit.
    pub fn mark_used(&mut self, name: Symbol) {
        if let Some(ids) = self.name_to_ids.get(&name) {
            if let Some(id) = ids.last() {
                self.symbols[id.0 as usize].is_used = true;
            }
        }
    }

    /// Emit warnings for unused symbols.
    ///
    /// Scans all symbols and emits a warning for each unused variable or
    /// function that does not carry `__attribute__((unused))` or
    /// `__attribute__((used))`.
    ///
    /// This should be called at the end of the translation unit (or at
    /// scope exit for block-scope variables).
    pub fn check_unused_symbols(
        &self,
        diagnostics: &mut DiagnosticEngine,
        interner: &crate::common::string_interner::Interner,
    ) {
        for entry in &self.symbols {
            if entry.is_used {
                continue;
            }

            // Skip symbols with __attribute__((unused)) or __attribute__((used)).
            let has_unused_attr = entry
                .attributes
                .iter()
                .any(|a| matches!(a, ValidatedAttribute::Unused | ValidatedAttribute::Used));
            if has_unused_attr {
                continue;
            }

            // Skip extern declarations without definitions — they are
            // forward references and not expected to be "used" locally.
            if entry.storage_class == StorageClass::Extern && !entry.is_defined {
                continue;
            }

            // Skip typedefs — unused typedef warnings are less common and
            // can be very noisy for system headers.
            if entry.kind == SymbolKind::TypedefName {
                continue;
            }

            // Skip enum constants — they are part of an enum definition
            // and not individually "used".
            if entry.kind == SymbolKind::EnumConstant {
                continue;
            }

            // Skip well-known entry-point functions — `main` and `_start`
            // are called by the runtime/linker and should never be flagged.
            let sym_name = interner.resolve(entry.name);
            if sym_name == "main" || sym_name == "_start" {
                continue;
            }

            // Emit appropriate warning based on kind.
            let msg = match entry.kind {
                SymbolKind::Variable => "unused variable",
                SymbolKind::Function => "unused function",
                _ => continue,
            };

            diagnostics.emit(Diagnostic::warning(entry.span, msg));
        }
    }

    // ===================================================================
    // Enum Constant Declaration
    // ===================================================================

    /// Declare an enumeration constant.
    ///
    /// Enum constants have **no linkage** and exist in the ordinary
    /// identifier namespace. They must not conflict with other identifiers
    /// declared in the same scope.
    ///
    /// # Arguments
    ///
    /// - `name`: interned identifier name
    /// - `_value`: the integer value of the enumerator (stored for later use)
    /// - `ty`: the type of the enum constant (typically `CType::Int` or the
    ///   enum's underlying type)
    /// - `span`: source location
    ///
    /// # Returns
    ///
    /// `Ok(SymbolId)` on success, `Err(())` on redeclaration conflict.
    pub fn declare_enum_constant(
        &mut self,
        name: Symbol,
        _value: i128,
        ty: CType,
        span: Span,
    ) -> Result<SymbolId, ()> {
        // Check for redeclaration in the current scope.
        if let Some(existing_id) = self.find_in_current_scope(name) {
            let existing = &self.symbols[existing_id.0 as usize];
            // Enum constants cannot be redeclared.
            // We produce an error but do not go through DiagnosticEngine here
            // because we return Err(()) — the caller is expected to have a
            // diagnostics reference to use. For consistency with declare(),
            // we use a lightweight approach.
            //
            // Actually, the caller can use the standard declare() path.
            // We keep this as a convenience, returning Err on conflict.
            let _ = existing; // suppress unused warning in this branch
            return Err(());
        }

        let entry = SymbolEntry {
            name,
            ty,
            kind: SymbolKind::EnumConstant,
            linkage: Linkage::None,
            storage_class: StorageClass::Auto, // Enum constants have no storage class.
            is_defined: true,
            is_tentative: false,
            span,
            attributes: Vec::new(),
            is_weak: false,
            visibility: Option::None,
            section: Option::None,
            is_used: false,
            scope_depth: self.current_scope_depth,
        };

        let id = SymbolId(self.symbols.len() as u32);
        self.symbols.push(entry);
        self.name_to_ids.entry(name).or_default().push(id);
        Ok(id)
    }

    // ===================================================================
    // Function Declaration
    // ===================================================================

    /// Declare or re-declare a function symbol.
    ///
    /// Handles:
    /// - File-scope functions default to external linkage.
    /// - `static` functions have internal linkage.
    /// - Prototype merging: if a function was previously declared, the new
    ///   declaration is checked for parameter type compatibility.
    ///
    /// # Returns
    ///
    /// `Ok(SymbolId)` on success, `Err(())` if a conflicting redeclaration
    /// is detected.
    pub fn declare_function(
        &mut self,
        name: Symbol,
        ty: CType,
        storage: StorageClass,
        span: Span,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<SymbolId, ()> {
        // Determine linkage for the function.
        let linkage = self.resolve_linkage(name, storage, self.current_scope_depth);

        let entry = SymbolEntry {
            name,
            ty,
            kind: SymbolKind::Function,
            linkage,
            storage_class: storage,
            is_defined: false, // Caller will call define() if this is a definition.
            is_tentative: false,
            span,
            attributes: Vec::new(),
            is_weak: false,
            visibility: Option::None,
            section: Option::None,
            is_used: false,
            scope_depth: self.current_scope_depth,
        };

        self.declare(entry, diagnostics)
    }

    // ===================================================================
    // Internal Helpers
    // ===================================================================

    /// Find the [`SymbolId`] of a declaration with the given `name` in the
    /// current scope depth, or `None` if no such declaration exists.
    fn find_in_current_scope(&self, name: Symbol) -> Option<SymbolId> {
        self.name_to_ids.get(&name).and_then(|ids| {
            ids.iter().rev().find_map(|&id| {
                if self.symbols[id.0 as usize].scope_depth == self.current_scope_depth {
                    Some(id)
                } else {
                    Option::None
                }
            })
        })
    }

    /// Simplified type compatibility check for redeclaration purposes.
    ///
    /// For the symbol table, we use structural equality via `CType`'s
    /// `PartialEq`. More nuanced compatibility (e.g. ignoring parameter
    /// names, qualifier differences on function parameters) is handled by
    /// the `TypeChecker` in `type_checker.rs`.
    ///
    /// For function types, we compare return types and parameter lists
    /// while allowing a prototype with no parameters (`()` in C means
    /// unspecified parameters) to be compatible with any parameter list.
    fn types_are_compatible(&self, a: &CType, b: &CType) -> bool {
        // Shortcut: structural equality covers most cases.
        if a == b {
            return true;
        }

        // Function type compatibility: a function declared with no parameters
        // (empty params, non-variadic) is compatible with any prototype.
        match (a, b) {
            (
                CType::Function {
                    return_type: ret_a,
                    params: params_a,
                    variadic: var_a,
                },
                CType::Function {
                    return_type: ret_b,
                    params: params_b,
                    variadic: var_b,
                },
            ) => {
                // Return types must be compatible.
                if !self.types_are_compatible(ret_a, ret_b) {
                    return false;
                }
                // An empty parameter list (K&R style) is compatible with any.
                if params_a.is_empty() || params_b.is_empty() {
                    return true;
                }
                // Same variadic flag required.
                if var_a != var_b {
                    return false;
                }
                // Parameter count must match.
                if params_a.len() != params_b.len() {
                    return false;
                }
                // Each parameter type must be compatible.
                params_a
                    .iter()
                    .zip(params_b.iter())
                    .all(|(pa, pb)| self.types_are_compatible(pa, pb))
            }

            // Pointer compatibility: compare pointee types.
            (CType::Pointer(inner_a, _quals_a), CType::Pointer(inner_b, _quals_b)) => {
                self.types_are_compatible(inner_a, inner_b)
            }

            // Array compatibility: compare element types; size may differ
            // (incomplete array vs sized array).
            (CType::Array(elem_a, _size_a), CType::Array(elem_b, _size_b)) => {
                self.types_are_compatible(elem_a, elem_b)
            }

            // Array ↔ Pointer compatibility: In C, function parameters
            // declared as `T name[]` are adjusted to `T *name` (C11 §6.7.6.3p7).
            // A forward declaration may use `T[]` while the definition uses
            // `T *` (or vice versa).  Treat these as compatible.
            (CType::Array(elem, _), CType::Pointer(inner, _))
            | (CType::Pointer(inner, _), CType::Array(elem, _)) => {
                self.types_are_compatible(elem, inner)
            }

            // Function type vs pointer-to-function type: C99 §6.7.6.3/8
            // states that a parameter declared as "function returning type"
            // is adjusted to "pointer to function returning type".  This
            // makes `filler_t filler` and `filler_t *filler` compatible in
            // parameter contexts, which the Linux kernel relies upon.
            (CType::Function { .. }, CType::Pointer(inner, _))
                if matches!(inner.as_ref(), CType::Function { .. }) =>
            {
                self.types_are_compatible(a, inner)
            }
            (CType::Pointer(inner, _), CType::Function { .. })
                if matches!(inner.as_ref(), CType::Function { .. }) =>
            {
                self.types_are_compatible(inner, b)
            }

            // Typedef: peel and compare.
            (CType::Typedef { underlying: u, .. }, other)
            | (other, CType::Typedef { underlying: u, .. }) => self.types_are_compatible(u, other),

            // Qualified type: peel qualifiers and compare base.
            (CType::Qualified(inner, _), other) | (other, CType::Qualified(inner, _)) => {
                self.types_are_compatible(inner, other)
            }

            // Struct compatibility: named structs with the same tag are
            // compatible even if their field lists differ (one may be a
            // forward-declared snapshot captured before the full definition).
            (CType::Struct { name: Some(na), .. }, CType::Struct { name: Some(nb), .. }) => {
                na == nb
            }

            // Union compatibility: same as struct — tag-name match suffices.
            (CType::Union { name: Some(na), .. }, CType::Union { name: Some(nb), .. }) => na == nb,

            // Enum compatibility: named enums with the same tag are compatible,
            // mirroring the struct/union treatment.  This covers cases like:
            //   extern enum system_states { ... } system_state;
            //   enum system_states system_state;  // re-declaration
            (CType::Enum { name: Some(na), .. }, CType::Enum { name: Some(nb), .. }) => na == nb,

            // Enum compatible with its underlying type (e.g. `int`).
            (
                CType::Enum {
                    underlying_type, ..
                },
                other,
            )
            | (
                other,
                CType::Enum {
                    underlying_type, ..
                },
            ) => self.types_are_compatible(underlying_type, other),

            _ => false,
        }
    }

    /// Merge attributes from a new declaration into an existing entry.
    ///
    /// Avoids duplicating attributes that are already present.
    fn merge_attributes(entry: &mut SymbolEntry, new_attrs: &[ValidatedAttribute]) {
        for attr in new_attrs {
            // Check for weak attribute.
            if matches!(attr, ValidatedAttribute::Weak) {
                entry.is_weak = true;
            }
            // Check for visibility attribute.
            if let ValidatedAttribute::Visibility(vis) = attr {
                entry.visibility = Some(*vis);
            }
            // Check for section attribute.
            if let ValidatedAttribute::Section(ref s) = attr {
                entry.section = Some(s.clone());
            }
            // Check for used attribute — mark as used to suppress warnings.
            if matches!(attr, ValidatedAttribute::Used) {
                entry.is_used = true;
            }
            // Add to attributes list if not already present.
            if !entry.attributes.contains(attr) {
                entry.attributes.push(attr.clone());
            }
        }
    }
}

// ===========================================================================
// Default trait
// ===========================================================================

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
