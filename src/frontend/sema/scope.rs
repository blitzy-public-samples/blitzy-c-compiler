// src/frontend/sema/scope.rs
//
// Scope management module for Phase 5 (semantic analysis) of the BCC compiler.
//
// Implements the C11 lexical scope system (§6.2.1) with a scope stack
// supporting four scope levels: global, file, function, and block. The module
// maintains three separate namespaces per C11 §6.2.3:
//
// 1. **Ordinary identifiers** — variables, functions, typedefs, enum constants.
//    Each entry maps an interned `Symbol` name to a `SymbolId` handle into the
//    external `SymbolTable`, avoiding tight coupling to symbol entry internals.
//
// 2. **Tag names** — struct, union, and enum tags. Tags occupy their own
//    namespace so that `struct foo` and `int foo` can coexist without conflict.
//    Tag entries carry the `CType`, completion status, and declaration span.
//
// 3. **Label names** — goto target labels. In standard C11 labels have function
//    scope; the GCC `__label__` extension creates block-scoped labels that
//    shadow outer labels of the same name within their enclosing block.
//
// Name lookup traverses the scope stack from innermost (current) to outermost
// (global), returning the first match — implementing the C11 shadowing rules
// where inner declarations hide outer ones.
//
// Additionally, the module tracks typedef names via a per-scope `FxHashSet`
// to support parser disambiguation between type names and ordinary identifiers
// (the "typedef-name problem" in C parsing).
//
// Integration points:
// - `crate::frontend::sema::symbol_table` — provides `SymbolId` handles stored
//   in the ordinary namespace; the scope module never inspects symbol entries.
// - `crate::common::diagnostics` — used for label validation errors at function
//   exit (undefined labels referenced by goto statements).
// - `crate::common::types` — `CType` stored in `TagEntry` for struct/union/enum
//   type tracking.
// - `crate::common::fx_hash` — `FxHashMap` and `FxHashSet` for performant
//   hash-based namespace lookups with Fibonacci hashing.
// - `crate::common::string_interner` — `Symbol` handles for zero-cost name
//   comparison across all three namespaces.

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::string_interner::Symbol;
use crate::common::types::CType;
use crate::frontend::sema::symbol_table::SymbolId;

// ---------------------------------------------------------------------------
// ScopeLevel — the four C11 scope levels
// ---------------------------------------------------------------------------

/// Identifies the kind of scope a [`Scope`] represents.
///
/// C11 §6.2.1 defines four kinds of scope. The ordering here reflects
/// nesting depth from outermost (Global) to innermost (Block). Every
/// `Scope` in the [`ScopeStack`] carries a `ScopeLevel` that determines
/// how names declared within it behave with respect to visibility, linkage,
/// and lifetime.
///
/// # Variants
///
/// - `Global` — Predefined names visible everywhere (`__builtin_*`, etc.).
///   There is exactly one global scope at the bottom of every scope stack.
/// - `File` — Top-level declarations in a translation unit. Names declared
///   here have file scope (visible from declaration to end of TU).
/// - `Function` — The scope of a function body. Labels (goto targets) have
///   function scope in standard C11: they are visible throughout the entire
///   function regardless of where they are defined.
/// - `Block` — A compound statement `{ ... }`. Names declared here have
///   block scope: they are visible from the point of declaration to the
///   closing brace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScopeLevel {
    /// Predefined names (builtins, compiler-generated entities).
    Global,
    /// Top-level declarations in a translation unit (file scope).
    File,
    /// Function body scope — labels have function scope per C11 §6.2.1p3.
    Function,
    /// Compound statement scope `{ ... }` (block scope).
    Block,
}

// ---------------------------------------------------------------------------
// TagKind — classification of struct/union/enum tags
// ---------------------------------------------------------------------------

/// Distinguishes the three kinds of aggregate/enumeration tags in C11.
///
/// Used in [`TagEntry`] to record whether a tag name refers to a struct,
/// union, or enum declaration. This is necessary because the tag namespace
/// is shared across all three kinds — redeclaring `struct foo` as `enum foo`
/// in the same scope is an error, but `struct foo` in an inner scope can
/// shadow `enum foo` from an outer scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TagKind {
    /// `struct` aggregate type.
    Struct,
    /// `union` aggregate type.
    Union,
    /// `enum` enumerated type.
    Enum,
}

// ---------------------------------------------------------------------------
// TagEntry — a struct/union/enum tag declaration in the tag namespace
// ---------------------------------------------------------------------------

/// Records a struct, union, or enum tag declaration within a scope.
///
/// Tags occupy their own namespace per C11 §6.2.3, separate from ordinary
/// identifiers and labels. This means `struct foo` and `int foo` can coexist
/// in the same scope without conflict.
///
/// A tag may initially be incomplete (forward declaration, e.g. `struct foo;`)
/// and later completed with a full definition (`struct foo { int x; };`). The
/// `is_complete` flag tracks this status, and [`ScopeStack::update_tag`] is
/// used to transition from incomplete to complete.
///
/// # Fields
///
/// - `kind` — Whether this is a struct, union, or enum tag.
/// - `ty` — The C type associated with this tag declaration.
/// - `is_complete` — `true` if the tag has been fully defined (all members
///   known); `false` for forward declarations.
/// - `span` — Source location of the tag declaration for diagnostic reporting.
#[derive(Clone, Debug)]
pub struct TagEntry {
    /// Whether this tag is a struct, union, or enum.
    pub kind: TagKind,
    /// The C type associated with this tag declaration.
    pub ty: CType,
    /// Whether the type definition is complete (all members known).
    pub is_complete: bool,
    /// Source location of the tag declaration.
    pub span: Span,
}

// ---------------------------------------------------------------------------
// LabelEntry — a goto label in the label namespace
// ---------------------------------------------------------------------------

/// Records a goto label declaration or reference within a scope.
///
/// In standard C11 (§6.2.1p3), labels have function scope — they are visible
/// throughout the entire enclosing function regardless of where they appear.
/// The GCC `__label__` extension (§6.8 in GCC docs) creates block-scoped
/// labels that shadow any function-scope label of the same name within the
/// declaring block.
///
/// A label may be referenced (by a `goto` statement) before it is defined
/// (by a label statement `name:`). The `is_defined` flag distinguishes these
/// states, enabling detection of undefined labels at function exit.
///
/// # Fields
///
/// - `name` — Interned label name for zero-cost comparison.
/// - `is_defined` — `true` when the label statement (`name:`) has been
///   encountered; `false` when only referenced by `goto name;`.
/// - `is_local` — `true` for labels declared via `__label__` (GCC extension);
///   these have block scope instead of function scope.
/// - `span` — Source location of the label definition or first reference.
#[derive(Clone, Debug)]
pub struct LabelEntry {
    /// Interned label name.
    pub name: Symbol,
    /// Whether the label definition (`name:`) has been encountered.
    pub is_defined: bool,
    /// Whether this is a `__label__` block-scoped label (GCC extension).
    pub is_local: bool,
    /// Source location of the label declaration or first reference.
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Scope — a single scope layer with three namespaces
// ---------------------------------------------------------------------------

/// A single scope layer within the [`ScopeStack`], containing three separate
/// namespaces per C11 §6.2.3 and a typedef name tracker.
///
/// Each scope represents one level of the C lexical scope hierarchy (global,
/// file, function, or block). Names declared within a scope are visible from
/// the point of declaration until the scope is exited (popped from the stack).
///
/// The three namespaces ensure that struct/union/enum tags, goto labels, and
/// ordinary identifiers do not conflict with each other, as required by C11.
///
/// # Fields
///
/// - `level` — The kind of scope (Global, File, Function, Block).
/// - `ordinary` — Ordinary identifier namespace: variables, functions,
///   typedefs, and enum constants, mapped to `SymbolId` handles.
/// - `tags` — Tag namespace: struct, union, and enum tag names mapped to
///   [`TagEntry`] records.
/// - `labels` — Label namespace: goto target labels mapped to [`LabelEntry`]
///   records.
/// - `depth` — Nesting depth from the outermost scope (0 = global). Used
///   for diagnostic context and debug output.
pub struct Scope {
    /// The kind of scope this layer represents.
    pub level: ScopeLevel,
    /// Ordinary identifier namespace: variables, functions, typedefs,
    /// enum constants → SymbolId handles into the SymbolTable.
    pub ordinary: FxHashMap<Symbol, SymbolId>,
    /// Tag namespace: struct/union/enum tag names → TagEntry records.
    pub tags: FxHashMap<Symbol, TagEntry>,
    /// Label namespace: goto label names → LabelEntry records.
    pub labels: FxHashMap<Symbol, LabelEntry>,
    /// Nesting depth from outermost scope (0 = global).
    pub depth: u32,
    /// Typedef name tracker for parser disambiguation.
    ///
    /// When a name is inserted as a typedef (via the symbol table), it is
    /// also recorded here so that [`ScopeStack::is_typedef`] can efficiently
    /// determine whether a name refers to a type without inspecting the
    /// symbol table's internals. This avoids coupling the scope module to
    /// `SymbolEntry` details.
    typedefs: FxHashSet<Symbol>,
}

impl Scope {
    /// Creates a new, empty scope at the given level and depth.
    ///
    /// All three namespaces and the typedef tracker start empty. The caller
    /// is responsible for inserting names as declarations are processed.
    ///
    /// # Arguments
    ///
    /// * `level` — The kind of scope (Global, File, Function, Block).
    /// * `depth` — Nesting depth from outermost scope (0 = global).
    fn new(level: ScopeLevel, depth: u32) -> Self {
        Scope {
            level,
            ordinary: FxHashMap::default(),
            tags: FxHashMap::default(),
            labels: FxHashMap::default(),
            depth,
            typedefs: FxHashSet::default(),
        }
    }

    /// Records a name as a typedef in this scope.
    ///
    /// Called by the semantic analyzer after inserting a typedef declaration
    /// into the ordinary namespace. This enables the `is_typedef` query on
    /// [`ScopeStack`] without requiring access to the symbol table.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned typedef name to record.
    #[inline]
    pub fn mark_typedef(&mut self, name: Symbol) {
        self.typedefs.insert(name);
    }

    /// Returns `true` if the given name is recorded as a typedef in this scope.
    #[inline]
    pub fn has_typedef(&self, name: Symbol) -> bool {
        self.typedefs.contains(&name)
    }
}

// ---------------------------------------------------------------------------
// ScopeStack — the main scope management structure
// ---------------------------------------------------------------------------

/// A stack of nested [`Scope`] layers implementing C11 lexical scoping.
///
/// The scope stack is the central data structure for name resolution during
/// semantic analysis. It maintains a vector of `Scope` layers where index 0
/// is the global (outermost) scope and the last element is the current
/// (innermost) scope.
///
/// # Scope Lifecycle
///
/// 1. [`ScopeStack::new`] creates a stack with a single global scope.
/// 2. [`push`](ScopeStack::push) enters a new scope (file, function, or block).
/// 3. Names are inserted and looked up via the namespace-specific methods.
/// 4. [`pop`](ScopeStack::pop) exits the current scope, removing all its names.
///
/// # Name Resolution Order
///
/// All lookup methods search from the innermost scope outward, returning the
/// first match. This implements C11's shadowing rules where inner declarations
/// hide outer ones of the same name.
///
/// # Namespaces
///
/// Three independent namespaces are maintained per C11 §6.2.3:
/// - **Ordinary** — variables, functions, typedefs, enum constants
/// - **Tags** — struct/union/enum tag names
/// - **Labels** — goto target labels
///
/// # Label Scope Rules
///
/// In standard C11, labels have function scope — they are visible throughout
/// the entire function. The GCC `__label__` extension creates block-scoped
/// labels that shadow function-scope labels within their declaring block.
///
/// # Typedef Tracking
///
/// The scope stack tracks typedef names independently of the symbol table,
/// enabling the parser to disambiguate type names from identifiers without
/// inspecting symbol entries.
pub struct ScopeStack {
    /// Stack of scope layers. Index 0 is the global scope (always present).
    /// The last element is the current (innermost) scope.
    scopes: Vec<Scope>,
}

impl Default for ScopeStack {
    /// Returns a `ScopeStack` initialised with a single global scope at depth 0.
    fn default() -> Self {
        Self::new()
    }
}

impl ScopeStack {
    // -----------------------------------------------------------------------
    // Construction and scope lifecycle
    // -----------------------------------------------------------------------

    /// Creates a new scope stack with an initial global scope at depth 0.
    ///
    /// The global scope is the outermost scope layer, intended for predefined
    /// names such as `__builtin_*` identifiers. It is never popped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut scopes = ScopeStack::new();
    /// assert_eq!(scopes.depth(), 0);
    /// assert_eq!(scopes.current().level, ScopeLevel::Global);
    /// ```
    pub fn new() -> Self {
        ScopeStack {
            scopes: vec![Scope::new(ScopeLevel::Global, 0)],
        }
    }

    /// Pushes a new scope onto the stack, entering a deeper nesting level.
    ///
    /// The new scope becomes the current (innermost) scope. Its depth is
    /// one greater than its parent scope's depth.
    ///
    /// # Arguments
    ///
    /// * `level` — The kind of scope to enter (File, Function, or Block).
    ///
    /// # Panics
    ///
    /// Does not panic. The scope stack can grow to arbitrary depth, limited
    /// only by available memory and the 512-depth recursion limit enforced
    /// by the parser/macro expander.
    pub fn push(&mut self, level: ScopeLevel) {
        let new_depth = self.scopes.len() as u32;
        self.scopes.push(Scope::new(level, new_depth));
    }

    /// Pops the current (innermost) scope from the stack and returns it.
    ///
    /// All names declared in the popped scope become invisible. The caller
    /// receives the scope for any final processing (e.g., checking for
    /// unused variables or validating local label definitions).
    ///
    /// # Panics
    ///
    /// Panics if called when only the global scope remains. The global scope
    /// must never be popped.
    ///
    /// # Returns
    ///
    /// The popped `Scope` with all its namespace contents.
    pub fn pop(&mut self) -> Scope {
        assert!(
            self.scopes.len() > 1,
            "BCC internal error: cannot pop the global scope"
        );
        self.scopes
            .pop()
            .expect("BCC internal error: scope stack unexpectedly empty")
    }

    /// Returns an immutable reference to the current (innermost) scope.
    ///
    /// This is the scope at the top of the stack — the most recently pushed
    /// scope that has not yet been popped.
    ///
    /// # Panics
    ///
    /// Never panics — the scope stack always contains at least the global scope.
    #[inline]
    pub fn current(&self) -> &Scope {
        self.scopes
            .last()
            .expect("BCC internal error: scope stack is empty")
    }

    /// Returns a mutable reference to the current (innermost) scope.
    ///
    /// Allows direct modification of the scope's namespace maps, for example
    /// to call [`Scope::mark_typedef`] after inserting a typedef declaration.
    ///
    /// # Panics
    ///
    /// Never panics — the scope stack always contains at least the global scope.
    #[inline]
    pub fn current_mut(&mut self) -> &mut Scope {
        self.scopes
            .last_mut()
            .expect("BCC internal error: scope stack is empty")
    }

    /// Returns the current nesting depth (0 = global scope).
    ///
    /// The depth equals the number of scopes pushed beyond the initial global
    /// scope: `depth() == scopes.len() - 1`.
    #[inline]
    pub fn depth(&self) -> u32 {
        // depth of the current scope
        self.current().depth
    }

    // -----------------------------------------------------------------------
    // Ordinary namespace — variables, functions, typedefs, enum constants
    // -----------------------------------------------------------------------

    /// Looks up an ordinary identifier by searching from the innermost scope
    /// outward to the global scope.
    ///
    /// Returns the `SymbolId` associated with the first matching declaration
    /// found during the outward search. This implements C11's shadowing rules:
    /// an inner declaration of the same name hides any outer declaration.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier to look up.
    ///
    /// # Returns
    ///
    /// - `Some(SymbolId)` if the name is found in any enclosing scope.
    /// - `None` if the name is not declared in any visible scope.
    pub fn lookup(&self, name: Symbol) -> Option<SymbolId> {
        // Iterate from innermost (last) to outermost (first) scope.
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.ordinary.get(&name) {
                return Some(id);
            }
        }
        None
    }

    /// Looks up an ordinary identifier in the current scope only.
    ///
    /// This is used for redeclaration checking: when processing a new
    /// declaration, we need to know if the same name was already declared
    /// in the *same* scope (as opposed to an outer scope where it would
    /// simply be shadowed rather than redeclared).
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier to look up.
    ///
    /// # Returns
    ///
    /// - `Some(SymbolId)` if the name exists in the current scope.
    /// - `None` if the name is not in the current scope (may still exist
    ///   in outer scopes).
    pub fn lookup_in_current(&self, name: Symbol) -> Option<SymbolId> {
        self.current().ordinary.get(&name).copied()
    }

    /// Inserts an ordinary identifier into the current scope.
    ///
    /// Maps the given `name` to the given `SymbolId` in the current scope's
    /// ordinary namespace. If the name already exists in the current scope
    /// (a redeclaration), the previous `SymbolId` is returned.
    ///
    /// This method does **not** affect outer scopes. If the same name exists
    /// in an outer scope, it is shadowed (hidden) by the new declaration
    /// in the current scope.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier to insert.
    /// * `id` — The symbol table handle for the declaration.
    ///
    /// # Returns
    ///
    /// - `Some(SymbolId)` — the previous entry in the current scope if
    ///   the name was already declared there (redeclaration).
    /// - `None` — the name is new to the current scope (may shadow outer).
    pub fn insert(&mut self, name: Symbol, id: SymbolId) -> Option<SymbolId> {
        self.current_mut().ordinary.insert(name, id)
    }

    // -----------------------------------------------------------------------
    // Tag namespace — struct/union/enum tags
    // -----------------------------------------------------------------------

    /// Looks up a tag name (struct, union, or enum) by searching from the
    /// innermost scope outward.
    ///
    /// Tags occupy a separate namespace from ordinary identifiers (C11 §6.2.3),
    /// so `struct foo` does not conflict with a variable named `foo`.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned tag name to look up.
    ///
    /// # Returns
    ///
    /// - `Some(&TagEntry)` — the first matching tag found in an enclosing scope.
    /// - `None` — the tag is not declared in any visible scope.
    pub fn lookup_tag(&self, name: Symbol) -> Option<&TagEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.tags.get(&name) {
                return Some(entry);
            }
        }
        None
    }

    /// Looks up a tag name in the current scope only.
    ///
    /// Used for checking whether a tag is already declared in the current
    /// scope (redeclaration vs. new declaration). A tag in an outer scope
    /// is not returned — it would be shadowed by the new declaration.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned tag name to look up.
    ///
    /// # Returns
    ///
    /// - `Some(&TagEntry)` — the tag exists in the current scope.
    /// - `None` — the tag is not in the current scope.
    pub fn lookup_tag_in_current(&self, name: Symbol) -> Option<&TagEntry> {
        self.current().tags.get(&name)
    }

    /// Inserts a tag entry into the current scope's tag namespace.
    ///
    /// If a tag with the same name already exists in the current scope,
    /// the previous entry is returned (for redeclaration validation by the
    /// semantic analyzer).
    ///
    /// # Arguments
    ///
    /// * `name` — The interned tag name.
    /// * `entry` — The tag entry to insert.
    ///
    /// # Returns
    ///
    /// - `Some(TagEntry)` — the previous tag in the current scope (displaced).
    /// - `None` — no previous tag with this name in the current scope.
    pub fn insert_tag(&mut self, name: Symbol, entry: TagEntry) -> Option<TagEntry> {
        self.current_mut().tags.insert(name, entry)
    }

    /// Updates an existing tag entry, typically transitioning from an
    /// incomplete (forward) declaration to a complete definition.
    ///
    /// Searches the scope stack from innermost to outermost for the first
    /// scope containing a tag with the given name, then replaces that
    /// entry with the provided updated entry.
    ///
    /// This is used when a forward declaration like `struct foo;` (incomplete)
    /// is followed by a definition `struct foo { int x; };` (complete) in a
    /// scope where the forward declaration is visible.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned tag name to update.
    /// * `entry` — The new (typically complete) tag entry.
    ///
    /// # Behavior
    ///
    /// If the tag is not found in any scope, this method does nothing.
    /// The caller should have already verified the tag exists via
    /// [`lookup_tag`](ScopeStack::lookup_tag) or
    /// [`lookup_tag_in_current`](ScopeStack::lookup_tag_in_current).
    pub fn update_tag(&mut self, name: Symbol, entry: TagEntry) {
        // Search from innermost to outermost for the scope containing this tag.
        for scope in self.scopes.iter_mut().rev() {
            if let Some(existing) = scope.tags.get_mut(&name) {
                *existing = entry;
                return;
            }
        }
        // Tag not found in any scope — no action. The semantic analyzer
        // should have validated existence before calling update_tag.
    }

    // -----------------------------------------------------------------------
    // Label namespace — goto targets (function scope + __label__ block scope)
    // -----------------------------------------------------------------------

    /// Looks up a label by searching from the innermost scope outward.
    ///
    /// In standard C11, labels have function scope — they are visible
    /// throughout the entire function. The GCC `__label__` extension creates
    /// block-scoped labels that shadow function-scope labels of the same name.
    ///
    /// The search proceeds from the current (innermost) scope outward. A
    /// `__label__` block-scoped label in an inner scope will be found before
    /// a function-scope label of the same name, implementing the expected
    /// shadowing behavior.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned label name to look up.
    ///
    /// # Returns
    ///
    /// - `Some(&LabelEntry)` — the first matching label found.
    /// - `None` — no label with this name is visible.
    pub fn lookup_label(&self, name: Symbol) -> Option<&LabelEntry> {
        for scope in self.scopes.iter().rev() {
            if let Some(entry) = scope.labels.get(&name) {
                return Some(entry);
            }
            // Labels can only exist in function or block scopes.
            // Once we pass the function scope, stop searching — labels
            // declared in one function are not visible in another.
            if scope.level == ScopeLevel::Function {
                break;
            }
        }
        None
    }

    /// Marks a label as defined (the `name:` label statement was encountered).
    ///
    /// If the label was previously declared via `__label__` (block-scoped),
    /// the definition is placed in the scope containing the `__label__`
    /// declaration. Otherwise, the label is defined in the nearest enclosing
    /// function scope, per C11's rule that labels have function scope.
    ///
    /// If the label was already defined (duplicate definition), an error
    /// is returned.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned label name being defined.
    /// * `span` — Source location of the label definition.
    ///
    /// # Returns
    ///
    /// - `Ok(())` — label successfully defined.
    /// - `Err(())` — duplicate label definition (same name already defined
    ///   in the relevant scope).
    #[allow(clippy::result_unit_err)]
    pub fn define_label(&mut self, name: Symbol, span: Span) -> Result<(), ()> {
        // First, search for a local (__label__) label declaration in block
        // scopes from innermost outward, stopping at function scope.
        for scope in self.scopes.iter_mut().rev() {
            if let Some(entry) = scope.labels.get_mut(&name) {
                if entry.is_local {
                    // Found a __label__ declaration — define it here.
                    if entry.is_defined {
                        return Err(()); // Duplicate label definition.
                    }
                    entry.is_defined = true;
                    entry.span = span;
                    return Ok(());
                }
            }
            if scope.level == ScopeLevel::Function {
                break;
            }
        }

        // No local label found. Define in the function scope.
        if let Some(func_scope) = self.find_function_scope_mut() {
            if let Some(entry) = func_scope.labels.get_mut(&name) {
                // Label was previously referenced (forward goto) or defined.
                if entry.is_defined {
                    return Err(()); // Duplicate label definition.
                }
                entry.is_defined = true;
                entry.span = span;
                return Ok(());
            }
            // First occurrence of this label — create a defined entry.
            func_scope.labels.insert(
                name,
                LabelEntry {
                    name,
                    is_defined: true,
                    is_local: false,
                    span,
                },
            );
            return Ok(());
        }

        // No function scope found — we're at file/global scope where labels
        // are not valid. Return error; the caller should emit a diagnostic.
        Err(())
    }

    /// Records a label reference (a `goto name;` statement was encountered).
    ///
    /// If the label has already been declared (via `__label__` or a previous
    /// definition/reference), this is a no-op. Otherwise, a new entry is
    /// created in the function scope with `is_defined = false`, indicating
    /// a forward reference that must be resolved before function exit.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned label name being referenced.
    /// * `span` — Source location of the `goto` statement.
    pub fn reference_label(&mut self, name: Symbol, span: Span) {
        // Check if the label already exists as a local (__label__) label
        // in any block scope from innermost outward.
        for scope in self.scopes.iter().rev() {
            if scope.labels.contains_key(&name) {
                // Label already known — nothing to do.
                return;
            }
            if scope.level == ScopeLevel::Function {
                break;
            }
        }

        // Label not found in block scopes. Check and create in function scope.
        if let Some(func_scope) = self.find_function_scope_mut() {
            if func_scope.labels.contains_key(&name) {
                // Already referenced or defined — nothing to do.
                return;
            }
            // Create a forward-reference entry (is_defined = false).
            func_scope.labels.insert(
                name,
                LabelEntry {
                    name,
                    is_defined: false,
                    is_local: false,
                    span,
                },
            );
        }
        // If no function scope exists, this is a goto at file/global scope
        // which is invalid C. The parser/sema should catch this separately.
    }

    /// Declares a block-scoped local label via the GCC `__label__` extension.
    ///
    /// The `__label__` declaration creates a label name in the current block
    /// scope that shadows any function-scope label of the same name. The label
    /// must be defined (with a `name:` statement) within the same block before
    /// the block exits.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned label name to declare.
    /// * `span` — Source location of the `__label__` declaration.
    ///
    /// # Example (GCC Extension)
    ///
    /// ```c
    /// void foo(void) {
    ///     {
    ///         __label__ done;
    ///         if (cond) goto done;
    ///         // ...
    ///         done:;
    ///     }
    ///     // 'done' label is no longer in scope here.
    /// }
    /// ```
    pub fn declare_local_label(&mut self, name: Symbol, span: Span) {
        let current = self.current_mut();
        current.labels.insert(
            name,
            LabelEntry {
                name,
                is_defined: false,
                is_local: true,
                span,
            },
        );
    }

    /// Validates that all referenced labels are defined within the function.
    ///
    /// Called when exiting a function scope. Scans all label entries in the
    /// scope stack from the current scope down to (and including) the function
    /// scope, checking that every label entry has `is_defined == true`.
    ///
    /// For any label that was referenced by a `goto` statement but never
    /// defined, emits an error diagnostic: "use of undeclared label 'name'".
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — The diagnostic engine for emitting label errors.
    ///
    /// # Notes
    ///
    /// This method should be called before popping the function scope so that
    /// all label entries (including those in nested block scopes that haven't
    /// been popped yet) are still visible. In typical usage, all inner block
    /// scopes will have been popped already, so only the function scope's
    /// labels remain.
    pub fn validate_labels_on_function_exit(&self, diagnostics: &mut DiagnosticEngine) {
        // Scan from innermost scope to the function scope (inclusive).
        for scope in self.scopes.iter().rev() {
            for entry in scope.labels.values() {
                if !entry.is_defined {
                    // Label was referenced (goto) but never defined (name:).
                    diagnostics.error(
                        entry.span,
                        format!("use of undeclared label '{:?}'", entry.name),
                    );
                }
            }
            // Stop after processing the function scope.
            if scope.level == ScopeLevel::Function {
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Typedef name queries — parser disambiguation support
    // -----------------------------------------------------------------------

    /// Checks whether a name resolves to a typedef in any enclosing scope.
    ///
    /// This is essential for the parser's disambiguation of type names vs.
    /// ordinary identifiers (the "typedef-name problem" in C). When the parser
    /// encounters an identifier token, it calls this method to determine if
    /// the identifier should be treated as a type name or a variable/function
    /// name.
    ///
    /// The search proceeds from the innermost scope outward, matching C11's
    /// shadowing rules. If a non-typedef declaration of the same name exists
    /// in an inner scope, it shadows the typedef from an outer scope, and this
    /// method correctly returns `false` because the typedef tracker only
    /// contains names explicitly marked as typedefs.
    ///
    /// # Arguments
    ///
    /// * `name` — The interned identifier to check.
    ///
    /// # Returns
    ///
    /// `true` if the name is recorded as a typedef in any visible scope,
    /// `false` otherwise.
    ///
    /// # Note on Shadowing
    ///
    /// If a variable shadows a typedef in an inner scope, the typedef tracker
    /// will not contain the name in the inner scope, so the search will
    /// continue outward and potentially find the typedef in an outer scope.
    /// To handle this correctly, the caller should check both `is_typedef`
    /// and `lookup` — if `lookup` returns a non-typedef `SymbolId` in a
    /// nearer scope, the name is not a typedef despite appearing as one in
    /// an outer scope.
    pub fn is_typedef(&self, name: Symbol) -> bool {
        // We need to check whether the nearest declaration of `name` is a
        // typedef. Walk from innermost to outermost scope:
        for scope in self.scopes.iter().rev() {
            // If the name exists in the ordinary namespace of this scope,
            // check if it's marked as a typedef. If it exists but is NOT
            // a typedef, it shadows any outer typedef → return false.
            if scope.ordinary.contains_key(&name) {
                return scope.typedefs.contains(&name);
            }
        }
        // Name not found in any scope's ordinary namespace → not a typedef.
        false
    }

    // -----------------------------------------------------------------------
    // Convenience methods — scope level queries
    // -----------------------------------------------------------------------

    /// Returns `true` if the current (innermost) scope is a file scope.
    ///
    /// File scope corresponds to the top level of a translation unit,
    /// outside any function body.
    #[inline]
    pub fn in_file_scope(&self) -> bool {
        self.current().level == ScopeLevel::File
    }

    /// Returns `true` if the scope stack contains a function scope at or
    /// above the current scope.
    ///
    /// This indicates we are currently inside a function body (possibly
    /// within a nested block scope). Used by the semantic analyzer to
    /// determine whether local variable declarations, `return` statements,
    /// and labels are valid at the current position.
    pub fn in_function_scope(&self) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|s| s.level == ScopeLevel::Function)
    }

    /// Returns `true` if the current (innermost) scope is a block scope.
    ///
    /// Block scope corresponds to a compound statement `{ ... }` within
    /// a function body.
    #[inline]
    pub fn in_block_scope(&self) -> bool {
        self.current().level == ScopeLevel::Block
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Finds the nearest enclosing function scope (mutable) by searching
    /// from the innermost scope outward.
    ///
    /// Returns `None` if no function scope exists in the stack (e.g., we
    /// are at file or global scope).
    fn find_function_scope_mut(&mut self) -> Option<&mut Scope> {
        self.scopes
            .iter_mut()
            .rev()
            .find(|s| s.level == ScopeLevel::Function)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test Symbol from a raw u32 index.
    fn sym(n: u32) -> Symbol {
        Symbol::new(n)
    }

    /// Helper to create a test SymbolId from a raw u32 index.
    fn sid(n: u32) -> SymbolId {
        SymbolId::from_raw(n)
    }

    // -- ScopeStack construction and lifecycle --

    #[test]
    fn test_new_scope_stack_has_global_scope() {
        let stack = ScopeStack::new();
        assert_eq!(stack.current().level, ScopeLevel::Global);
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_push_and_pop_scopes() {
        let mut stack = ScopeStack::new();

        stack.push(ScopeLevel::File);
        assert_eq!(stack.current().level, ScopeLevel::File);
        assert_eq!(stack.depth(), 1);

        stack.push(ScopeLevel::Function);
        assert_eq!(stack.current().level, ScopeLevel::Function);
        assert_eq!(stack.depth(), 2);

        stack.push(ScopeLevel::Block);
        assert_eq!(stack.current().level, ScopeLevel::Block);
        assert_eq!(stack.depth(), 3);

        let popped = stack.pop();
        assert_eq!(popped.level, ScopeLevel::Block);
        assert_eq!(stack.current().level, ScopeLevel::Function);
        assert_eq!(stack.depth(), 2);
    }

    #[test]
    #[should_panic(expected = "cannot pop the global scope")]
    fn test_cannot_pop_global_scope() {
        let mut stack = ScopeStack::new();
        stack.pop(); // Should panic
    }

    // -- Ordinary namespace lookup --

    #[test]
    fn test_ordinary_insert_and_lookup() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);
        let id = sid(10);

        assert!(stack.lookup(name).is_none());
        let prev = stack.insert(name, id);
        assert!(prev.is_none());
        assert_eq!(stack.lookup(name), Some(id));
    }

    #[test]
    fn test_ordinary_shadowing() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);
        let outer_id = sid(10);
        stack.insert(name, outer_id);

        stack.push(ScopeLevel::Function);
        stack.push(ScopeLevel::Block);

        let inner_id = sid(20);
        stack.insert(name, inner_id);

        // Lookup should find the inner (shadowing) declaration.
        assert_eq!(stack.lookup(name), Some(inner_id));

        // lookup_in_current should find the inner one.
        assert_eq!(stack.lookup_in_current(name), Some(inner_id));

        // Pop the block scope — the outer declaration should be visible again.
        stack.pop();
        // Now we're in function scope, which doesn't have this name.
        assert_eq!(stack.lookup(name), Some(outer_id));
    }

    #[test]
    fn test_lookup_in_current_scope_only() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);
        let id = sid(10);
        stack.insert(name, id);

        stack.push(ScopeLevel::Function);

        // Name exists in file scope but NOT in function scope.
        assert!(stack.lookup_in_current(name).is_none());
        // Full lookup still finds it.
        assert_eq!(stack.lookup(name), Some(id));
    }

    #[test]
    fn test_insert_returns_previous_on_redeclaration() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);
        let first_id = sid(10);
        let second_id = sid(20);

        let prev1 = stack.insert(name, first_id);
        assert!(prev1.is_none());

        let prev2 = stack.insert(name, second_id);
        assert_eq!(prev2, Some(first_id));
        assert_eq!(stack.lookup(name), Some(second_id));
    }

    // -- Tag namespace --

    #[test]
    fn test_tag_insert_and_lookup() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let tag_name = sym(5);
        let entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: false,
            span: Span::DUMMY,
        };

        assert!(stack.lookup_tag(tag_name).is_none());
        let prev = stack.insert_tag(tag_name, entry);
        assert!(prev.is_none());

        let found = stack.lookup_tag(tag_name).unwrap();
        assert_eq!(found.kind, TagKind::Struct);
        assert!(!found.is_complete);
    }

    #[test]
    fn test_tag_update_to_complete() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let tag_name = sym(5);

        // Insert incomplete forward declaration.
        let incomplete_entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: false,
            span: Span::DUMMY,
        };
        stack.insert_tag(tag_name, incomplete_entry);

        // Update to complete definition.
        let complete_entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: true,
            span: Span::DUMMY,
        };
        stack.update_tag(tag_name, complete_entry);

        let found = stack.lookup_tag(tag_name).unwrap();
        assert!(found.is_complete);
    }

    #[test]
    fn test_tag_and_ordinary_coexist() {
        // C11 §6.2.3: Tags and ordinary identifiers are in separate namespaces.
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);

        // Insert ordinary identifier.
        stack.insert(name, sid(10));

        // Insert tag with same name — should not conflict.
        let tag_entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: true,
            span: Span::DUMMY,
        };
        stack.insert_tag(name, tag_entry);

        // Both should be independently retrievable.
        assert_eq!(stack.lookup(name), Some(sid(10)));
        assert!(stack.lookup_tag(name).is_some());
    }

    #[test]
    fn test_tag_lookup_in_current_only() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let tag_name = sym(5);
        let entry = TagEntry {
            kind: TagKind::Enum,
            ty: CType::Enum {
                name: Some("color".to_string()),
                underlying: Box::new(CType::Int { signed: true }),
            },
            is_complete: true,
            span: Span::DUMMY,
        };
        stack.insert_tag(tag_name, entry);

        stack.push(ScopeLevel::Function);

        // Not in current (function) scope.
        assert!(stack.lookup_tag_in_current(tag_name).is_none());
        // But found via full lookup.
        assert!(stack.lookup_tag(tag_name).is_some());
    }

    // -- Label namespace --

    #[test]
    fn test_label_define_and_lookup() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        let span = Span::new(0, 10, 15);

        assert!(stack.define_label(label, span).is_ok());

        let found = stack.lookup_label(label).unwrap();
        assert!(found.is_defined);
        assert!(!found.is_local);
    }

    #[test]
    fn test_label_reference_then_define() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        let ref_span = Span::new(0, 10, 15);
        let def_span = Span::new(0, 50, 55);

        // Forward reference via goto.
        stack.reference_label(label, ref_span);
        let found = stack.lookup_label(label).unwrap();
        assert!(!found.is_defined);

        // Definition.
        assert!(stack.define_label(label, def_span).is_ok());
        let found = stack.lookup_label(label).unwrap();
        assert!(found.is_defined);
    }

    #[test]
    fn test_duplicate_label_definition_fails() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        let span1 = Span::new(0, 10, 15);
        let span2 = Span::new(0, 50, 55);

        assert!(stack.define_label(label, span1).is_ok());
        assert!(stack.define_label(label, span2).is_err());
    }

    #[test]
    fn test_local_label_declaration() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);
        stack.push(ScopeLevel::Block);

        let label = sym(100);
        let span = Span::new(0, 10, 15);

        // Declare local label via __label__.
        stack.declare_local_label(label, span);

        let found = stack.lookup_label(label).unwrap();
        assert!(!found.is_defined);
        assert!(found.is_local);

        // Define the local label.
        let def_span = Span::new(0, 30, 35);
        assert!(stack.define_label(label, def_span).is_ok());

        let found = stack.lookup_label(label).unwrap();
        assert!(found.is_defined);
        assert!(found.is_local);
    }

    #[test]
    fn test_local_label_shadows_function_label() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);

        // Define a function-scope label.
        stack.define_label(label, Span::new(0, 0, 5)).unwrap();

        // Enter a block scope with a __label__ of the same name.
        stack.push(ScopeLevel::Block);
        stack.declare_local_label(label, Span::new(0, 20, 25));

        // The local label should shadow the function-scope label.
        let found = stack.lookup_label(label).unwrap();
        assert!(found.is_local);
        assert!(!found.is_defined); // Not yet defined in block scope.

        // Pop the block — function-scope label visible again.
        stack.pop();
        let found = stack.lookup_label(label).unwrap();
        assert!(!found.is_local);
        assert!(found.is_defined);
    }

    #[test]
    fn test_validate_labels_reports_undefined() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        // Reference label without defining it (forward goto with no target).
        stack.reference_label(label, Span::new(0, 10, 15));

        let mut diag = DiagnosticEngine::new();
        stack.validate_labels_on_function_exit(&mut diag);

        assert!(diag.has_errors());
        assert_eq!(diag.error_count(), 1);
    }

    #[test]
    fn test_validate_labels_passes_when_all_defined() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        stack.reference_label(label, Span::new(0, 10, 15));
        stack.define_label(label, Span::new(0, 50, 55)).unwrap();

        let mut diag = DiagnosticEngine::new();
        stack.validate_labels_on_function_exit(&mut diag);

        assert!(!diag.has_errors());
    }

    // -- Typedef tracking --

    #[test]
    fn test_is_typedef() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);
        let id = sid(10);

        // Insert as a normal identifier — not a typedef.
        stack.insert(name, id);
        assert!(!stack.is_typedef(name));

        // Now mark it as a typedef.
        stack.current_mut().mark_typedef(name);
        assert!(stack.is_typedef(name));
    }

    #[test]
    fn test_typedef_shadowed_by_variable() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let name = sym(1);

        // File-scope typedef.
        stack.insert(name, sid(10));
        stack.current_mut().mark_typedef(name);
        assert!(stack.is_typedef(name));

        // Block scope with a variable of the same name (not typedef).
        stack.push(ScopeLevel::Function);
        stack.push(ScopeLevel::Block);
        stack.insert(name, sid(20));
        // NOT marking as typedef — this shadows the outer typedef.

        // is_typedef should return false because the nearest declaration
        // is a variable, not a typedef.
        assert!(!stack.is_typedef(name));

        // Pop block — typedef visible again.
        stack.pop();
        stack.pop();
        assert!(stack.is_typedef(name));
    }

    #[test]
    fn test_typedef_not_found() {
        let stack = ScopeStack::new();
        assert!(!stack.is_typedef(sym(999)));
    }

    // -- Convenience methods --

    #[test]
    fn test_scope_level_queries() {
        let mut stack = ScopeStack::new();
        assert!(!stack.in_file_scope());
        assert!(!stack.in_function_scope());
        assert!(!stack.in_block_scope());

        stack.push(ScopeLevel::File);
        assert!(stack.in_file_scope());
        assert!(!stack.in_function_scope());
        assert!(!stack.in_block_scope());

        stack.push(ScopeLevel::Function);
        assert!(!stack.in_file_scope());
        assert!(stack.in_function_scope());
        assert!(!stack.in_block_scope());

        stack.push(ScopeLevel::Block);
        assert!(!stack.in_file_scope());
        assert!(stack.in_function_scope()); // Still inside a function.
        assert!(stack.in_block_scope());
    }

    #[test]
    fn test_multiple_nested_blocks() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let x = sym(1);
        let y = sym(2);

        // Outer block.
        stack.push(ScopeLevel::Block);
        stack.insert(x, sid(10));

        // Inner block.
        stack.push(ScopeLevel::Block);
        stack.insert(y, sid(20));

        assert_eq!(stack.lookup(x), Some(sid(10))); // From outer block.
        assert_eq!(stack.lookup(y), Some(sid(20))); // From inner block.

        // Pop inner block — y disappears.
        stack.pop();
        assert_eq!(stack.lookup(x), Some(sid(10)));
        assert!(stack.lookup(y).is_none());

        // Pop outer block — x disappears.
        stack.pop();
        assert!(stack.lookup(x).is_none());
    }

    #[test]
    fn test_tag_shadowing_across_scopes() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let tag = sym(5);

        // File-scope struct foo (incomplete).
        let outer_entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: false,
            span: Span::DUMMY,
        };
        stack.insert_tag(tag, outer_entry);

        // Block scope with a different struct foo (complete).
        stack.push(ScopeLevel::Function);
        stack.push(ScopeLevel::Block);

        let inner_entry = TagEntry {
            kind: TagKind::Struct,
            ty: CType::Struct {
                name: Some("foo".to_string()),
                fields: vec![],
            },
            is_complete: true,
            span: Span::DUMMY,
        };
        stack.insert_tag(tag, inner_entry);

        // Inner complete version should shadow outer incomplete.
        assert!(stack.lookup_tag(tag).unwrap().is_complete);

        // Pop block — outer incomplete version visible again.
        stack.pop();
        stack.pop();
        assert!(!stack.lookup_tag(tag).unwrap().is_complete);
    }

    #[test]
    fn test_update_tag_in_outer_scope() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);

        let tag = sym(5);

        // File-scope forward declaration.
        let incomplete = TagEntry {
            kind: TagKind::Union,
            ty: CType::Union {
                name: Some("bar".to_string()),
                fields: vec![],
            },
            is_complete: false,
            span: Span::DUMMY,
        };
        stack.insert_tag(tag, incomplete);

        // Enter function scope, then update the outer tag to complete.
        stack.push(ScopeLevel::Function);

        let complete = TagEntry {
            kind: TagKind::Union,
            ty: CType::Union {
                name: Some("bar".to_string()),
                fields: vec![],
            },
            is_complete: true,
            span: Span::DUMMY,
        };
        stack.update_tag(tag, complete);

        // Even from function scope, lookup should find the updated tag.
        assert!(stack.lookup_tag(tag).unwrap().is_complete);
    }

    #[test]
    fn test_label_not_visible_across_functions() {
        let mut stack = ScopeStack::new();
        stack.push(ScopeLevel::File);
        stack.push(ScopeLevel::Function);

        let label = sym(100);
        stack.define_label(label, Span::DUMMY).unwrap();

        // Pop function, push a new one.
        stack.pop();
        stack.push(ScopeLevel::Function);

        // Label from first function should not be visible in second.
        assert!(stack.lookup_label(label).is_none());
    }
}
