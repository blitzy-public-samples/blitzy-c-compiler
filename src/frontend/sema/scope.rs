//! Scope management for Phase 5 (semantic analysis) of the BCC C11 compiler.
//!
//! Manages a lexical scope stack with block, function, file, and global scopes.
//! Maintains three separate C11 namespaces per scope:
//!
//! 1. **Ordinary identifiers** — variables, functions, typedefs, enum constants
//! 2. **Tags** — struct, union, and enum names
//! 3. **Labels** — goto targets (function-wide scope) with GCC `__label__` support
//!
//! # Architecture
//!
//! The scope system is layered on top of the symbol table — it manages name
//! *visibility* (which declarations are reachable from a given point in the
//! source) while the symbol table manages *identity* (the full metadata for
//! each declared entity). [`ScopeStack`] maps `Symbol → SymbolId` for the
//! ordinary namespace; the `SymbolId` is then used to retrieve the full
//! `SymbolEntry` from the `SymbolTable`.
//!
//! # C11 Namespace Rules (§6.2.3)
//!
//! - Ordinary identifiers, tag names, and labels occupy separate namespaces.
//! - `struct foo` and `int foo` can coexist without conflict.
//! - Each struct/union has its own member namespace (handled by the type system).
//! - Labels are function-scoped; GCC `__label__` makes them block-scoped.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on `std` and `crate::` references. No external
//! crates. Uses [`FxHashMap`] from `crate::common::fx_hash` for all internal
//! hash maps. Does NOT depend on `crate::ir`, `crate::passes`, or
//! `crate::backend`.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Severity, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::common::types::CType;
use crate::frontend::sema::symbol_table::SymbolId;

// ===========================================================================
// ScopeKind — classification of scope levels
// ===========================================================================

/// Classification of a lexical scope level in C11.
///
/// C11 defines four kinds of scopes (§6.2.1):
/// - File scope (identifiers declared outside any function body)
/// - Function scope (labels — visible throughout the function)
/// - Block scope (identifiers declared inside a compound statement)
/// - Function prototype scope (parameter names in prototypes)
///
/// BCC adds [`Global`](ScopeKind::Global) as the outermost scope to
/// distinguish the initial scope from file-level declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Global/file scope — outermost translation unit scope.
    Global,
    /// File scope — top-level declarations in a translation unit.
    File,
    /// Function scope — encompasses the entire function body.
    /// Labels are function-scoped per C11 §6.2.1p4.
    Function,
    /// Block scope — compound statement `{ }`, loop bodies, if/else branches.
    Block,
    /// Function prototype scope — parameter names in function declarations.
    FunctionPrototype,
}

// ===========================================================================
// TagKind — struct/union/enum classification
// ===========================================================================

/// Classification of a tag in the tag namespace.
///
/// Tags live in a separate namespace from ordinary identifiers (C11 §6.2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    /// `struct` tag.
    Struct,
    /// `union` tag.
    Union,
    /// `enum` tag.
    Enum,
}

// ===========================================================================
// TagEntry — tag namespace entry
// ===========================================================================

/// Entry in the tag namespace for a struct, union, or enum declaration.
///
/// Supports the forward-declaration-then-completion lifecycle:
/// 1. `struct foo;` creates an incomplete `TagEntry` (`is_complete = false`).
/// 2. `struct foo { int x; };` updates it to complete (`is_complete = true`).
#[derive(Debug, Clone)]
pub struct TagEntry {
    /// The tag kind: struct, union, or enum.
    pub kind: TagKind,
    /// The complete C type (may be incomplete/forward-declared).
    pub ty: CType,
    /// Whether this is a complete definition (has a body).
    pub is_complete: bool,
    /// Source location of the declaration.
    pub span: Span,
}

// ===========================================================================
// LabelEntry — label namespace entry
// ===========================================================================

/// Entry in the label namespace for a goto target.
///
/// Labels in C11 are function-scoped (§6.2.1p4) — visible throughout the
/// entire function body regardless of where they are defined. GCC's
/// `__label__` extension allows block-scoped labels within compound
/// statements.
///
/// Forward gotos are supported: a label can be referenced before it is
/// defined. Validation occurs when the enclosing scope is popped.
#[derive(Debug, Clone)]
pub struct LabelEntry {
    /// Source location of the label definition (`label:`).
    /// `None` if only referenced (forward goto) but not yet defined.
    pub defined_at: Option<Span>,
    /// Source locations of all `goto` references to this label.
    pub referenced_at: Vec<Span>,
    /// Whether this is a GCC `__label__` local label declaration.
    pub is_local: bool,
}

// ===========================================================================
// Scope — a single lexical scope level
// ===========================================================================

/// A single lexical scope level containing declarations in three namespaces.
///
/// The `kind` and `depth` fields are public for inspection; the namespace
/// maps are private and accessed through [`ScopeStack`] methods.
pub struct Scope {
    /// What kind of scope this is.
    pub kind: ScopeKind,
    /// Depth level (0 = global, increases with nesting).
    pub depth: u32,
    /// Ordinary identifiers declared in this scope (name → SymbolId).
    ordinary: FxHashMap<Symbol, SymbolId>,
    /// Tag names declared in this scope (struct/union/enum names).
    tags: FxHashMap<Symbol, TagEntry>,
    /// Labels declared or referenced in this scope.
    labels: FxHashMap<Symbol, LabelEntry>,
    /// Tracks which ordinary names in this scope are typedef names.
    /// Used by [`ScopeStack::is_typedef_name`] for parser disambiguation.
    typedefs: FxHashMap<Symbol, ()>,
    /// Whether this block scope is a loop or switch body.
    /// Used for `break`/`continue` validation via
    /// [`ScopeStack::find_enclosing_loop_or_switch`].
    is_loop_or_switch: bool,
}

impl Scope {
    /// Create a new scope with the given kind and depth, all maps empty.
    fn new(kind: ScopeKind, depth: u32) -> Self {
        Scope {
            kind,
            depth,
            ordinary: FxHashMap::default(),
            tags: FxHashMap::default(),
            labels: FxHashMap::default(),
            typedefs: FxHashMap::default(),
            is_loop_or_switch: false,
        }
    }

    /// Mark this scope as a loop or switch body scope.
    ///
    /// Call after [`ScopeStack::push_scope`]`(ScopeKind::Block)` when the
    /// block is the body of a `for`, `while`, `do-while`, or `switch`
    /// statement. Enables [`ScopeStack::find_enclosing_loop_or_switch`] to
    /// detect valid `break`/`continue` targets.
    #[inline]
    pub fn mark_loop_or_switch(&mut self) {
        self.is_loop_or_switch = true;
    }
}

// ===========================================================================
// ScopeStack — main scope manager
// ===========================================================================

/// Manages a stack of lexical scopes for C11 semantic analysis.
///
/// Implements C's lexical scoping rules:
/// - Names declared in inner scopes shadow names in outer scopes.
/// - Tags (struct/union/enum) have a separate namespace from ordinary
///   identifiers.
/// - Labels are function-scoped (or block-scoped with `__label__`).
/// - Typedef names are tracked for parser disambiguation.
///
/// # Usage
///
/// ```ignore
/// let mut scopes = ScopeStack::new();
/// scopes.push_scope(ScopeKind::File);
/// scopes.push_scope(ScopeKind::Function);
/// // ... declare symbols, tags, labels ...
/// scopes.pop_scope(&mut diagnostics);  // validates labels
/// scopes.pop_scope(&mut diagnostics);
/// ```
pub struct ScopeStack {
    /// Stack of scopes, innermost on top.
    scopes: Vec<Scope>,
    /// Current total depth.
    depth: u32,
}

impl Default for ScopeStack {
    fn default() -> Self {
        let mut stack = ScopeStack {
            scopes: Vec::with_capacity(32),
            depth: 0,
        };
        // Push the initial Global scope at depth 0.
        stack.scopes.push(Scope::new(ScopeKind::Global, 0));
        stack
    }
}

impl ScopeStack {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Create a new scope stack with an initial Global scope at depth 0.
    pub fn new() -> Self {
        Self::default()
    }

    // ===================================================================
    // Push / Pop
    // ===================================================================

    /// Push a new scope onto the stack.
    ///
    /// Increments the nesting depth and creates a fresh scope with empty
    /// namespaces. All subsequent declarations will go into this scope
    /// until it is popped.
    pub fn push_scope(&mut self, kind: ScopeKind) {
        self.depth += 1;
        self.scopes.push(Scope::new(kind, self.depth));
    }

    /// Pop the topmost scope from the stack.
    ///
    /// When popping a [`Function`](ScopeKind::Function) scope, all labels
    /// are validated:
    /// - Referenced but undefined labels produce an error.
    /// - Defined but unreferenced labels produce a warning.
    ///
    /// When popping a [`Block`](ScopeKind::Block) scope with local
    /// (`__label__`) labels, those labels are also validated.
    ///
    /// Returns the popped scope for inspection (e.g., to check for unused
    /// symbols).
    ///
    /// # Panics
    ///
    /// Panics if called when only the Global scope remains.
    pub fn pop_scope(&mut self, diagnostics: &mut DiagnosticEngine) -> Scope {
        assert!(self.scopes.len() > 1, "cannot pop the last (Global) scope");

        let scope = self.scopes.pop().expect("scope stack unexpectedly empty");
        if self.depth > 0 {
            self.depth -= 1;
        }

        // Validate labels when leaving a Function scope.
        if scope.kind == ScopeKind::Function {
            self.validate_labels_in_scope(&scope, diagnostics);
        }

        // Validate local labels when leaving a Block scope that has them.
        if scope.kind == ScopeKind::Block && !scope.labels.is_empty() {
            self.validate_local_labels_in_scope(&scope, diagnostics);
        }

        scope
    }

    // ===================================================================
    // Accessors
    // ===================================================================

    /// Return a reference to the topmost (current) scope.
    ///
    /// # Panics
    ///
    /// Panics if the scope stack is empty.
    #[inline]
    pub fn current_scope(&self) -> &Scope {
        self.scopes.last().expect("scope stack is empty")
    }

    /// Return a mutable reference to the topmost (current) scope.
    ///
    /// # Panics
    ///
    /// Panics if the scope stack is empty.
    #[inline]
    pub fn current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("scope stack is empty")
    }

    /// Return the current nesting depth.
    #[inline]
    pub fn current_depth(&self) -> u32 {
        self.depth
    }

    /// Return the kind of the current (topmost) scope.
    #[inline]
    pub fn current_kind(&self) -> ScopeKind {
        self.current_scope().kind
    }

    // ===================================================================
    // Ordinary Identifier Lookup
    // ===================================================================

    /// Look up an ordinary identifier by searching from innermost to
    /// outermost scope.
    ///
    /// Returns the [`SymbolId`] of the first (innermost) match, implementing
    /// C's lexical scoping with shadowing semantics. If the identifier is
    /// not declared in any visible scope, returns `None`.
    /// Return the current scope nesting depth (0 = file scope).
    pub fn scope_depth(&self) -> usize {
        self.scopes.len().saturating_sub(1)
    }

    /// Debug dump all scopes and their ordinary identifiers.
    pub fn debug_dump(&self, interner: &crate::common::string_interner::Interner) {
        for (i, scope) in self.scopes.iter().enumerate() {
            eprintln!("  Scope {}: kind={:?}, ordinary names:", i, scope.kind);
            for (sym, id) in &scope.ordinary {
                let name = interner.resolve(*sym);
                eprintln!("    '{}' (Symbol({})) -> {:?}", name, sym.as_u32(), id);
            }
        }
    }

    pub fn lookup_ordinary(&self, name: Symbol) -> Option<SymbolId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.ordinary.get(&name) {
                return Some(id);
            }
        }
        None
    }

    /// Look up an ordinary identifier in the current (innermost) scope only.
    ///
    /// Used for redeclaration detection — checks whether the name is already
    /// declared at the current scope level. Does not search outer scopes.
    pub fn lookup_ordinary_in_current_scope(&self, name: Symbol) -> Option<SymbolId> {
        self.scopes
            .last()
            .and_then(|scope| scope.ordinary.get(&name).copied())
    }

    /// Declare an ordinary identifier in the current scope.
    ///
    /// Returns the previous [`SymbolId`] if the name was already declared
    /// in the current scope (redeclaration), or `None` if this is a new
    /// declaration at this scope level.
    ///
    /// # Typedef Registration
    ///
    /// If the declared name is a typedef, the caller must also call
    /// [`register_typedef`](ScopeStack::register_typedef) to enable
    /// [`is_typedef_name`](ScopeStack::is_typedef_name) detection. Calling
    /// `declare_ordinary` alone removes any prior typedef marking for the
    /// name in the current scope (since a non-typedef declaration shadows
    /// a typedef).
    pub fn declare_ordinary(&mut self, name: Symbol, id: SymbolId) -> Option<SymbolId> {
        let scope = self.scopes.last_mut().expect("scope stack is empty");
        // Validate the SymbolId is within reasonable bounds.
        debug_assert!(id.as_u32() < u32::MAX, "SymbolId overflow: {}", id.as_u32());
        // Remove any previous typedef marking for this name in the current
        // scope. If the new declaration IS a typedef, the caller will call
        // register_typedef() after this.
        scope.typedefs.remove(&name);
        scope.ordinary.insert(name, id)
    }

    // ===================================================================
    // Tag Namespace Management
    // ===================================================================

    /// Look up a tag name by searching from innermost to outermost scope.
    ///
    /// Tags occupy a separate namespace from ordinary identifiers (C11
    /// §6.2.3): `struct foo` and `int foo` can coexist without conflict.
    pub fn lookup_tag(&self, name: Symbol) -> Option<&TagEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.tags.get(&name) {
                return Some(entry);
            }
        }
        None
    }

    /// Look up a tag by its string name, using the interner to compare
    /// each Symbol key against the provided name.
    ///
    /// This is used when we have a tag name as a `&str` (e.g. from
    /// `CType::Struct { name: Some("foo"), .. }`) but only an immutable
    /// reference to the interner — so we cannot call `interner.intern()`
    /// to obtain a `Symbol`. Instead we iterate visible tags and match
    /// by resolved string equality.
    pub fn lookup_tag_by_str(&self, name: &str, interner: &Interner) -> Option<&TagEntry> {
        for scope in self.scopes.iter().rev() {
            for (sym, entry) in &scope.tags {
                if interner.resolve(*sym) == name {
                    return Some(entry);
                }
            }
        }
        None
    }

    /// Look up a tag name in the current scope only.
    ///
    /// Used for detecting redefinition of a tag at the same scope level.
    pub fn lookup_tag_in_current_scope(&self, name: Symbol) -> Option<&TagEntry> {
        self.scopes.last().and_then(|scope| scope.tags.get(&name))
    }

    /// Declare a tag in the current scope.
    ///
    /// Returns the previous [`TagEntry`] if the name was already declared
    /// as a tag in the current scope (for redeclaration checking by the
    /// semantic analyzer).
    ///
    /// # Tag Declaration Rules
    ///
    /// The caller (semantic analyzer) is responsible for enforcing:
    /// - Forward declaration `struct foo;` → inserts an incomplete tag.
    /// - Full definition `struct foo { ... }` → updates existing incomplete
    ///   to complete.
    /// - Redefinition of an already-complete tag in the same scope → ERROR.
    /// - Different tag kind reuse in the same scope → ERROR
    ///   (`struct foo` vs `union foo`).
    pub fn declare_tag(&mut self, name: Symbol, entry: TagEntry) -> Option<TagEntry> {
        let scope = self.scopes.last_mut().expect("scope stack is empty");
        scope.tags.insert(name, entry)
    }

    /// Update an existing incomplete tag to its complete definition.
    ///
    /// Walks from innermost scope outward to find the tag and updates its
    /// type and completion status. Called when the full body of a
    /// struct/union/enum is parsed after a forward declaration.
    ///
    /// If the tag is not found in any scope, this is a no-op.
    pub fn complete_tag(&mut self, name: Symbol, ty: CType) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(entry) = scope.tags.get_mut(&name) {
                entry.ty = ty;
                entry.is_complete = true;
                return;
            }
        }
    }

    /// Collect all visible tag entries across all scope levels.
    ///
    /// Returns a vector of `(Symbol, TagEntry)` pairs representing every
    /// tag currently in scope, from outermost to innermost. When a tag
    /// name is shadowed, the innermost (most recent) entry wins.
    /// Used by the constant evaluator to resolve `sizeof(struct foo)`.
    pub fn all_tags(&self) -> Vec<(Symbol, TagEntry)> {
        let mut merged: FxHashMap<Symbol, TagEntry> = FxHashMap::default();
        // Walk from outermost to innermost; inner scopes overwrite outer
        for scope in self.scopes.iter() {
            for (name, entry) in &scope.tags {
                merged.insert(*name, entry.clone());
            }
        }
        merged.into_iter().collect()
    }

    /// Collect all visible ordinary symbol IDs across all scope levels.
    ///
    /// Returns a vector of `(Symbol, SymbolId)` pairs representing every
    /// ordinary identifier currently in scope. When a name is shadowed,
    /// the innermost binding wins.
    /// Used by the constant evaluator to resolve `sizeof(variable)`.
    pub fn all_ordinary_symbols(&self) -> Vec<(Symbol, SymbolId)> {
        let mut merged: FxHashMap<Symbol, SymbolId> = FxHashMap::default();
        for scope in self.scopes.iter() {
            for (name, &id) in &scope.ordinary {
                merged.insert(*name, id);
            }
        }
        merged.into_iter().collect()
    }

    // ===================================================================
    // Label Namespace Management
    // ===================================================================

    /// Declare a label for `__label__` (GCC extension) or implicit
    /// function-scope use.
    ///
    /// - If `is_local` is `true`, the label is added to the current block
    ///   scope's label map (GCC `__label__` local label).
    /// - If `is_local` is `false`, the label is added to the nearest
    ///   enclosing [`Function`](ScopeKind::Function) scope's label map.
    ///
    /// Emits an error if the label is already declared in the target scope.
    pub fn declare_label(
        &mut self,
        name: Symbol,
        span: Span,
        is_local: bool,
        diagnostics: &mut DiagnosticEngine,
    ) {
        let target_idx = if is_local {
            // Local labels go in the current scope.
            self.scopes.len() - 1
        } else {
            // Non-local labels go in the nearest Function scope.
            self.find_function_scope_index()
                .unwrap_or(self.scopes.len() - 1)
        };

        let target_scope = &mut self.scopes[target_idx];

        if target_scope.labels.contains_key(&name) {
            // Label already declared — use Symbol::as_u32() for trace context.
            let _sym_trace = name.as_u32();
            diagnostics.emit_error(span, "redefinition of label");
            return;
        }

        target_scope.labels.insert(
            name,
            LabelEntry {
                defined_at: None,
                referenced_at: Vec::new(),
                is_local,
            },
        );
    }

    /// Record that a label has been defined (a `label:` statement was
    /// encountered).
    ///
    /// Labels can be referenced before definition (forward `goto`). If the
    /// label was previously referenced but not yet defined, this marks it
    /// as defined. If no prior entry exists, a new one is created.
    ///
    /// Emits an error if the label is already defined (duplicate definition).
    pub fn define_label(&mut self, name: Symbol, span: Span, diagnostics: &mut DiagnosticEngine) {
        let target_idx = self.find_label_scope_index(name);
        let target_scope = &mut self.scopes[target_idx];

        if let Some(entry) = target_scope.labels.get_mut(&name) {
            // Label already exists (was declared or previously referenced).
            if let Some(existing_span) = entry.defined_at {
                // Already defined — emit a redefinition error with a note
                // pointing to the previous definition location.
                diagnostics.emit(Diagnostic::error(span, "redefinition of label").with_note(
                    Span::new(
                        existing_span.file_id,
                        existing_span.start,
                        existing_span.end,
                    ),
                    "previous definition is here",
                ));
                return;
            }
            entry.defined_at = Some(span);
        } else {
            // Label not yet declared or referenced — create a new entry.
            target_scope.labels.insert(
                name,
                LabelEntry {
                    defined_at: Some(span),
                    referenced_at: Vec::new(),
                    is_local: false,
                },
            );
        }
    }

    /// Record a `goto` reference to a label.
    ///
    /// The label may not be defined yet (forward goto). Validation occurs
    /// when the enclosing function or block scope is popped.
    pub fn reference_label(&mut self, name: Symbol, span: Span) {
        let target_idx = self.find_label_scope_index(name);
        let target_scope = &mut self.scopes[target_idx];

        if let Some(entry) = target_scope.labels.get_mut(&name) {
            entry.referenced_at.push(span);
        } else {
            // Label not yet declared — create an entry with just a reference.
            // Will be validated when the enclosing scope is popped.
            target_scope.labels.insert(
                name,
                LabelEntry {
                    defined_at: None,
                    referenced_at: vec![span],
                    is_local: false,
                },
            );
        }
    }

    /// Validate all labels across all visible scopes.
    ///
    /// For each label in every scope:
    /// - Referenced but not defined → error: "use of undeclared label"
    /// - Defined but not referenced → warning: "label defined but not used"
    /// - Declared (`__label__`) but never used → warning (local labels only)
    ///
    /// Typically called explicitly at function boundaries or at the end of
    /// semantic analysis.
    pub fn validate_labels(&self, diagnostics: &mut DiagnosticEngine) {
        for scope in &self.scopes {
            if scope.kind == ScopeKind::Function || !scope.labels.is_empty() {
                self.validate_labels_in_scope(scope, diagnostics);
            }
        }
    }

    // ===================================================================
    // Helper Methods
    // ===================================================================

    /// Returns `true` if the current scope is at file scope level.
    ///
    /// File scope is either [`ScopeKind::File`] or [`ScopeKind::Global`].
    pub fn is_file_scope(&self) -> bool {
        let kind = self.current_kind();
        kind == ScopeKind::File || kind == ScopeKind::Global
    }

    /// Returns `true` if we are currently inside a function body.
    ///
    /// Walks up the scope stack to find any [`Function`](ScopeKind::Function)
    /// scope.
    pub fn is_function_scope(&self) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|s| s.kind == ScopeKind::Function)
    }

    /// Find the nearest enclosing [`Function`](ScopeKind::Function) scope.
    ///
    /// Returns `None` if we are not inside a function body. Used for label
    /// resolution and `return` type checking.
    pub fn find_enclosing_function(&self) -> Option<&Scope> {
        self.scopes
            .iter()
            .rev()
            .find(|s| s.kind == ScopeKind::Function)
    }

    /// Check if we are inside a loop or switch statement.
    ///
    /// Walks up the scope stack, stopping at the nearest
    /// [`Function`](ScopeKind::Function) scope boundary. Returns `true` if
    /// any enclosing [`Block`](ScopeKind::Block) scope (within the current
    /// function) is marked as a loop or switch body.
    ///
    /// Used for `break`/`continue` statement validation.
    pub fn find_enclosing_loop_or_switch(&self) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.is_loop_or_switch {
                return true;
            }
            // Do not cross function boundaries — loops/switches from an
            // enclosing function are irrelevant.
            if scope.kind == ScopeKind::Function {
                return false;
            }
        }
        false
    }

    // ===================================================================
    // Typedef Detection
    // ===================================================================

    /// Check if `name` is currently declared as a typedef in any visible scope.
    ///
    /// This is critical for the parser to disambiguate identifiers vs type
    /// names in C11. The parser calls this to determine whether a token that
    /// looks like an identifier is actually a type name (the "typedef-name"
    /// ambiguity in C grammar).
    ///
    /// The check respects shadowing: if an inner scope declares `name` as
    /// a non-typedef ordinary identifier, it shadows any outer typedef of
    /// the same name.
    pub fn is_typedef_name(&self, name: Symbol) -> bool {
        for scope in self.scopes.iter().rev() {
            // Check if this scope has the name registered as a typedef.
            if scope.typedefs.contains_key(&name) {
                return true;
            }
            // If the name exists as an ordinary identifier but NOT as a
            // typedef in this scope, it shadows any outer typedef.
            if scope.ordinary.contains_key(&name) {
                return false;
            }
        }
        false
    }

    /// Register a name as a typedef in the current scope.
    ///
    /// Must be called after [`declare_ordinary`](ScopeStack::declare_ordinary)
    /// when the declaration is a `typedef`. Enables
    /// [`is_typedef_name`](ScopeStack::is_typedef_name) to correctly identify
    /// the name as a type name for parser disambiguation.
    pub fn register_typedef(&mut self, name: Symbol) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.typedefs.insert(name, ());
        }
    }

    // ===================================================================
    // Internal Helpers
    // ===================================================================

    /// Find the index of the nearest enclosing Function scope.
    fn find_function_scope_index(&self) -> Option<usize> {
        self.scopes
            .iter()
            .rposition(|s| s.kind == ScopeKind::Function)
    }

    /// Find the scope index where a label should be placed or looked up.
    ///
    /// Walks from innermost scope outward:
    /// 1. If a block scope declares the label as local (`__label__`), use
    ///    that scope.
    /// 2. Otherwise, use the nearest [`Function`](ScopeKind::Function) scope.
    /// 3. If no function scope exists, fall back to the outermost scope.
    fn find_label_scope_index(&self, name: Symbol) -> usize {
        // First, check for local label declarations in block scopes.
        for (idx, scope) in self.scopes.iter().enumerate().rev() {
            if scope.kind == ScopeKind::Block {
                if let Some(entry) = scope.labels.get(&name) {
                    if entry.is_local {
                        return idx;
                    }
                }
            }
            // Stop searching at function boundary — use the function scope.
            if scope.kind == ScopeKind::Function {
                return idx;
            }
        }
        // Fallback: outermost scope (should not normally happen in valid C).
        0
    }

    /// Validate labels within a specific scope (function-level validation).
    ///
    /// For each label in the scope:
    /// - Referenced but not defined → error
    /// - Defined but not referenced → warning
    /// - Declared (`__label__`) but never used or defined → warning
    fn validate_labels_in_scope(&self, scope: &Scope, diagnostics: &mut DiagnosticEngine) {
        for (_name, entry) in scope.labels.iter() {
            // Determine the primary span for this label — prefer the
            // definition location, then the first reference, then a dummy.
            let primary_span = entry
                .defined_at
                .or_else(|| entry.referenced_at.first().copied())
                .unwrap_or_else(Span::dummy);

            // Handle __label__ declarations with no uses and no definitions.
            if entry.defined_at.is_none() && entry.referenced_at.is_empty() {
                if entry.is_local {
                    diagnostics.emit_warning(primary_span, "local label declared but never used");
                }
                continue;
            }

            // Check for undefined labels (referenced but not defined).
            if entry.defined_at.is_none() {
                for ref_span in &entry.referenced_at {
                    diagnostics.emit(Diagnostic::error(*ref_span, "use of undeclared label"));
                }
            }

            // Check for unused labels (defined but not referenced).
            if let Some(def_span) = entry.defined_at {
                if entry.referenced_at.is_empty() {
                    diagnostics.emit(Diagnostic::warning(def_span, "label defined but not used"));
                }
            }
        }
    }

    /// Validate local labels within a block scope.
    ///
    /// Called when popping a block scope that contains `__label__` local
    /// label declarations. Uses full [`Diagnostic`] struct construction
    /// with explicit [`Severity`] for precise control over diagnostics.
    fn validate_local_labels_in_scope(&self, scope: &Scope, diagnostics: &mut DiagnosticEngine) {
        for (_name, entry) in scope.labels.iter() {
            if !entry.is_local {
                continue;
            }

            // Undefined local label that was referenced.
            if entry.defined_at.is_none() && !entry.referenced_at.is_empty() {
                for ref_span in &entry.referenced_at {
                    diagnostics.emit(Diagnostic {
                        severity: Severity::Error,
                        span: *ref_span,
                        message: "use of undeclared local label".to_string(),
                        notes: Vec::new(),
                        fix_suggestion: None,
                    });
                }
            }

            // Defined but unreferenced local label.
            if let Some(def_span) = entry.defined_at {
                if entry.referenced_at.is_empty() {
                    diagnostics.emit(Diagnostic {
                        severity: Severity::Warning,
                        span: def_span,
                        message: "local label defined but not used".to_string(),
                        notes: Vec::new(),
                        fix_suggestion: None,
                    });
                }
            }
        }
    }
}
