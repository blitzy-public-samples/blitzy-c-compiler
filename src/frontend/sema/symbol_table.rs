// src/frontend/sema/symbol_table.rs
//
// Symbol table module for Phase 5 (semantic analysis) of the BCC compiler.
//
// Manages all declared names within a translation unit, recording each
// symbol's resolved C type, linkage (external, internal, none per C11
// §6.2.2), storage class (auto, register, static, extern, typedef per
// C11 §6.2.4), definition vs. declaration status, and GCC attribute
// annotations (weak binding, visibility, section placement, etc.).
//
// Key responsibilities:
//
// - **Symbol storage:** Every declared identifier (variable, function,
//   typedef, enum constant) is assigned a compact `SymbolId` handle and
//   stored in a flat `Vec<SymbolEntry>` for O(1) random access.
//
// - **Declaration/definition merging:** When the same identifier is
//   declared multiple times (e.g. `extern int x;` followed by `int x = 5;`),
//   the symbol table merges the declarations according to C11 rules:
//   compatible types are unified, linkage is resolved, and conflicting
//   redeclarations produce diagnostic errors.
//
// - **Tentative definitions (C11 §6.9.2):** File-scope variable
//   declarations without initializers are tentative until the end of the
//   translation unit. The symbol table tracks this status and can finalize
//   tentative definitions after all declarations are processed.
//
// - **Linkage resolution (C11 §6.2.2):** The `resolve_linkage()` function
//   determines the linkage of a declaration based on its storage class,
//   scope level, and any prior visible declaration of the same name.
//
// - **Weak symbol handling:** GCC `__attribute__((weak))` marks a symbol
//   as having weak binding, allowing it to be overridden by a strong
//   definition at link time.
//
// Integration points:
// - `crate::frontend::sema::scope` — the scope stack maps names to
//   `SymbolId` handles; this module stores the actual `SymbolEntry` data.
// - `crate::frontend::sema::type_checker` — resolves identifiers to
//   their declared types via `SymbolTable::get()`.
// - `crate::frontend::sema::attribute_handler` — propagates validated
//   GCC attributes into `SymbolEntry::attributes`.
// - `crate::ir::lowering` — reads linkage, storage class, and type
//   information from symbol entries to generate correct IR.

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::Symbol;
use crate::common::types::CType;

// ---------------------------------------------------------------------------
// VisibilityKind — ELF symbol visibility (GCC __attribute__((visibility)))
// ---------------------------------------------------------------------------

/// ELF symbol visibility as set by `__attribute__((visibility("...")))`.
///
/// Controls how the dynamic linker resolves references to this symbol in
/// shared libraries. Maps directly to the `STV_*` values in the ELF spec.
///
/// # Variants
///
/// - `Default` — Symbol is visible to other shared objects (STV_DEFAULT).
/// - `Hidden` — Symbol is not visible outside the shared object (STV_HIDDEN).
/// - `Protected` — Symbol is visible but cannot be preempted (STV_PROTECTED).
/// - `Internal` — Symbol uses processor-specific semantics (STV_INTERNAL).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisibilityKind {
    /// Visible to other shared objects (STV_DEFAULT).
    Default,
    /// Not visible outside the defining shared object (STV_HIDDEN).
    Hidden,
    /// Visible but not preemptible by other shared objects (STV_PROTECTED).
    Protected,
    /// Processor-specific hidden visibility (STV_INTERNAL).
    Internal,
}

// ---------------------------------------------------------------------------
// Linkage — C11 §6.2.2 identifier linkage classes
// ---------------------------------------------------------------------------

/// Linkage class for an identifier per C11 §6.2.2.
///
/// Linkage determines whether multiple declarations of the same name in
/// different scopes or translation units refer to the same entity.
///
/// # Variants
///
/// - `External` — Visible across translation units. Default for file-scope
///   functions and `extern` variables.
/// - `Internal` — Visible only within the current translation unit.
///   Applies to `static` file-scope declarations.
/// - `None` — Local to the enclosing block scope. Applies to block-scope
///   variables without `extern`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// Visible across translation units (file-scope functions, extern vars).
    External,
    /// Visible only within the current translation unit (static file-scope).
    Internal,
    /// Local to the enclosing scope (block-scope variables).
    None,
}

// ---------------------------------------------------------------------------
// StorageClass — C11 §6.2.4 storage class specifiers
// ---------------------------------------------------------------------------

/// Storage class specifier for a declaration per C11 §6.2.4.
///
/// Maps directly from the parsed declaration specifier list. Each
/// declaration may have at most one storage class specifier.
///
/// # Variants
///
/// - `Auto` — Automatic storage duration (default for block-scope).
/// - `Register` — Hint for register allocation; address-of is forbidden.
/// - `Static` — Static storage duration with internal linkage at file scope,
///   or static local at block scope.
/// - `Extern` — External linkage declaration; may or may not be a definition.
/// - `Typedef` — Not truly a storage class but syntactically occupies the
///   same position in declaration specifiers.
/// - `ThreadLocal` — `_Thread_local` storage duration (C11 §6.7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Automatic storage duration (block-scope default).
    Auto,
    /// Register hint; address-of forbidden.
    Register,
    /// Static storage duration / internal linkage at file scope.
    Static,
    /// External linkage declaration.
    Extern,
    /// Type alias (not a true storage class, but parsed as one).
    Typedef,
    /// Thread-local storage duration (`_Thread_local`).
    ThreadLocal,
}

// ---------------------------------------------------------------------------
// SymbolAttributes — GCC attribute annotations on a symbol
// ---------------------------------------------------------------------------

/// Collection of GCC `__attribute__` annotations applied to a symbol.
///
/// Each field corresponds to a specific GCC attribute that the semantic
/// analyzer validates and propagates from the parsed `__attribute__((...))`.
/// These attributes influence code generation (e.g. section placement,
/// alignment), linking (e.g. weak binding, visibility), and diagnostics
/// (e.g. deprecated, warn_unused_result).
///
/// All boolean fields default to `false` and `Option` fields to `None`,
/// representing the absence of the corresponding attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolAttributes {
    /// `__attribute__((weak))` — weak binding; overridable at link time.
    pub is_weak: bool,
    /// `__attribute__((visibility("...")))` — ELF symbol visibility.
    pub visibility: Option<VisibilityKind>,
    /// `__attribute__((section("...")))` — place symbol in named ELF section.
    pub section: Option<String>,
    /// `__attribute__((used))` — prevent linker from discarding this symbol.
    pub is_used: bool,
    /// `__attribute__((unused))` — suppress "unused variable" warnings.
    pub is_unused: bool,
    /// `__attribute__((deprecated))` or `__attribute__((deprecated("msg")))`.
    /// `Some(msg)` carries the optional deprecation message; `None` means
    /// the attribute is not present.
    pub is_deprecated: Option<String>,
    /// `__attribute__((noreturn))` or `_Noreturn` — function never returns.
    pub is_noreturn: bool,
    /// `__attribute__((inline))` — hint for inlining.
    pub is_inline: bool,
    /// `__attribute__((always_inline))` — force inlining.
    pub is_always_inline: bool,
    /// `__attribute__((noinline))` — prevent inlining.
    pub is_noinline: bool,
    /// `__attribute__((cold))` — unlikely execution path.
    pub is_cold: bool,
    /// `__attribute__((hot))` — likely execution path.
    pub is_hot: bool,
    /// `__attribute__((pure))` — function has no side effects except
    /// reading memory through its arguments.
    pub is_pure: bool,
    /// `__attribute__((const))` — function has no side effects and does
    /// not read global memory; result depends only on arguments.
    pub is_const: bool,
    /// `__attribute__((malloc))` — returned pointer does not alias any
    /// existing pointer.
    pub is_malloc: bool,
    /// `__attribute__((warn_unused_result))` — warn if return value is
    /// discarded.
    pub is_warn_unused_result: bool,
    /// `__attribute__((constructor(priority)))` — called before `main`.
    /// `Some(priority)` carries the optional priority value (lower = earlier).
    pub constructor_priority: Option<u32>,
    /// `__attribute__((destructor(priority)))` — called after `main` exits.
    /// `Some(priority)` carries the optional priority value.
    pub destructor_priority: Option<u32>,
    /// `__attribute__((aligned(N)))` — minimum alignment in bytes.
    pub alignment: Option<u64>,
}

impl Default for SymbolAttributes {
    /// Returns a `SymbolAttributes` with no attributes set.
    fn default() -> Self {
        SymbolAttributes {
            is_weak: false,
            visibility: None,
            section: None,
            is_used: false,
            is_unused: false,
            is_deprecated: None,
            is_noreturn: false,
            is_inline: false,
            is_always_inline: false,
            is_noinline: false,
            is_cold: false,
            is_hot: false,
            is_pure: false,
            is_const: false,
            is_malloc: false,
            is_warn_unused_result: false,
            constructor_priority: None,
            destructor_priority: None,
            alignment: None,
        }
    }
}

impl SymbolAttributes {
    /// Creates a new `SymbolAttributes` with all attributes disabled.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if any attribute is set (non-default).
    pub fn has_any(&self) -> bool {
        self.is_weak
            || self.visibility.is_some()
            || self.section.is_some()
            || self.is_used
            || self.is_unused
            || self.is_deprecated.is_some()
            || self.is_noreturn
            || self.is_inline
            || self.is_always_inline
            || self.is_noinline
            || self.is_cold
            || self.is_hot
            || self.is_pure
            || self.is_const
            || self.is_malloc
            || self.is_warn_unused_result
            || self.constructor_priority.is_some()
            || self.destructor_priority.is_some()
            || self.alignment.is_some()
    }

    /// Merges attributes from `other` into `self`.
    ///
    /// For boolean flags, the result is the logical OR (either source
    /// having the attribute enables it). For `Option` fields, `other`
    /// takes precedence if it is `Some`.
    pub fn merge_from(&mut self, other: &SymbolAttributes) {
        self.is_weak |= other.is_weak;
        if other.visibility.is_some() {
            self.visibility = other.visibility;
        }
        if other.section.is_some() {
            self.section = other.section.clone();
        }
        self.is_used |= other.is_used;
        self.is_unused |= other.is_unused;
        if other.is_deprecated.is_some() {
            self.is_deprecated = other.is_deprecated.clone();
        }
        self.is_noreturn |= other.is_noreturn;
        self.is_inline |= other.is_inline;
        self.is_always_inline |= other.is_always_inline;
        self.is_noinline |= other.is_noinline;
        self.is_cold |= other.is_cold;
        self.is_hot |= other.is_hot;
        self.is_pure |= other.is_pure;
        self.is_const |= other.is_const;
        self.is_malloc |= other.is_malloc;
        self.is_warn_unused_result |= other.is_warn_unused_result;
        if other.constructor_priority.is_some() {
            self.constructor_priority = other.constructor_priority;
        }
        if other.destructor_priority.is_some() {
            self.destructor_priority = other.destructor_priority;
        }
        if other.alignment.is_some() {
            self.alignment = other.alignment;
        }
    }
}

// ---------------------------------------------------------------------------
// SymbolId — compact handle into the symbol table
// ---------------------------------------------------------------------------

/// Compact 4-byte handle referencing a symbol entry in the [`SymbolTable`].
///
/// `SymbolId` is a newtype wrapper around `u32` that serves as an index
/// into the symbol table's internal `Vec<SymbolEntry>`. Because it is
/// `Copy`, `Eq`, `Hash`, and `Ord`, it can be used efficiently as a value
/// in scope hash maps, AST nodes, and IR instructions.
///
/// # Safety
///
/// A `SymbolId` is only valid for the `SymbolTable` that created it.
/// Using a `SymbolId` from one table with a different table may panic
/// or return incorrect results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(u32);

impl SymbolId {
    /// Returns the underlying index as a `usize`, suitable for use as an
    /// array/vector index.
    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    /// Returns the raw `u32` value of this symbol ID.
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// Creates a `SymbolId` from a raw `u32` index.
    ///
    /// This is intended for internal use by the symbol table and scope
    /// modules. External callers should obtain `SymbolId` values via
    /// `SymbolTable::insert()`.
    #[inline]
    pub fn from_raw(index: u32) -> Self {
        SymbolId(index)
    }
}

// ---------------------------------------------------------------------------
// SymbolEntry — a single declared name with full semantic information
// ---------------------------------------------------------------------------

/// A single symbol entry recording everything the compiler needs to know
/// about a declared name for semantic analysis, IR lowering, and linking.
///
/// Every variable, function, typedef, and enum constant in the translation
/// unit has a corresponding `SymbolEntry` in the symbol table.
///
/// # Fields
///
/// - `name` — Interned identifier handle for zero-cost comparison.
/// - `ty` — The resolved C type of this symbol.
/// - `linkage` — External, internal, or no linkage per C11 §6.2.2.
/// - `storage_class` — Storage class specifier from the declaration.
/// - `is_definition` — `true` if this entry represents a definition
///   (not merely a declaration).
/// - `is_tentative` — `true` for file-scope variable declarations
///   without initializers (C11 §6.9.2 tentative definitions).
/// - `span` — Source location of this declaration for diagnostics.
/// - `attributes` — GCC `__attribute__` annotations.
#[derive(Clone, Debug)]
pub struct SymbolEntry {
    /// Interned name of this symbol.
    pub name: Symbol,
    /// Resolved C type of this symbol.
    pub ty: CType,
    /// Linkage class (external, internal, none).
    pub linkage: Linkage,
    /// Storage class specifier.
    pub storage_class: StorageClass,
    /// Whether this entry is a definition (vs. merely a declaration).
    pub is_definition: bool,
    /// Whether this is a tentative definition (C11 §6.9.2).
    pub is_tentative: bool,
    /// Source location of this declaration.
    pub span: Span,
    /// GCC attribute annotations.
    pub attributes: SymbolAttributes,
}

impl SymbolEntry {
    /// Creates a new symbol entry with the given core fields.
    /// Attributes default to none and tentative defaults to false.
    pub fn new(
        name: Symbol,
        ty: CType,
        linkage: Linkage,
        storage_class: StorageClass,
        is_definition: bool,
        span: Span,
    ) -> Self {
        SymbolEntry {
            name,
            ty,
            linkage,
            storage_class,
            is_definition,
            is_tentative: false,
            span,
            attributes: SymbolAttributes::default(),
        }
    }

    /// Creates a tentative definition entry (C11 §6.9.2).
    ///
    /// Tentative definitions are file-scope variable declarations without
    /// an initializer. They act as definitions if no other definition of
    /// the same identifier appears in the translation unit.
    pub fn new_tentative(
        name: Symbol,
        ty: CType,
        linkage: Linkage,
        storage_class: StorageClass,
        span: Span,
    ) -> Self {
        SymbolEntry {
            name,
            ty,
            linkage,
            storage_class,
            is_definition: false,
            is_tentative: true,
            span,
            attributes: SymbolAttributes::default(),
        }
    }

    /// Returns `true` if this symbol has external linkage.
    #[inline]
    pub fn is_external(&self) -> bool {
        self.linkage == Linkage::External
    }

    /// Returns `true` if this symbol has internal linkage (`static`
    /// at file scope).
    #[inline]
    pub fn is_internal(&self) -> bool {
        self.linkage == Linkage::Internal
    }

    /// Returns `true` if this symbol has no linkage (block-scope local).
    #[inline]
    pub fn is_no_linkage(&self) -> bool {
        self.linkage == Linkage::None
    }

    /// Returns `true` if the symbol is a typedef.
    #[inline]
    pub fn is_typedef(&self) -> bool {
        self.storage_class == StorageClass::Typedef
    }

    /// Returns `true` if the symbol is declared `extern`.
    #[inline]
    pub fn is_extern(&self) -> bool {
        self.storage_class == StorageClass::Extern
    }

    /// Returns `true` if the symbol has weak binding.
    #[inline]
    pub fn is_weak(&self) -> bool {
        self.attributes.is_weak
    }

    /// Creates a compiler-generated symbol entry with a dummy source span.
    ///
    /// Used for implicit declarations (e.g. predefined `__func__`) and
    /// compiler-synthesized entities that have no source location.
    pub fn compiler_generated(
        name: Symbol,
        ty: CType,
        linkage: Linkage,
        storage_class: StorageClass,
    ) -> Self {
        SymbolEntry {
            name,
            ty,
            linkage,
            storage_class,
            is_definition: true,
            is_tentative: false,
            span: Span::DUMMY,
            attributes: SymbolAttributes::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// resolve_linkage — C11 §6.2.2 linkage determination
// ---------------------------------------------------------------------------

/// Determines the linkage of a declaration based on its storage class,
/// scope level, and any prior visible declaration of the same name.
///
/// Implements the linkage rules from C11 §6.2.2:
///
/// | Scope       | Storage class         | Result                                 |
/// |-------------|-----------------------|----------------------------------------|
/// | File        | (none)                | External                               |
/// | File        | `static`              | Internal                               |
/// | File        | `extern`              | External (or inherit prior linkage)    |
/// | Block       | `extern`              | External (or inherit prior linkage)    |
/// | Block       | `static`              | None (static local — storage duration) |
/// | Block       | (none) / auto / reg   | None                                   |
///
/// # Arguments
///
/// - `storage_class` — The storage class specifier of the new declaration.
/// - `is_file_scope` — `true` if the declaration appears at file scope.
///   Uses a `bool` rather than importing `ScopeLevel` to avoid a circular
///   dependency between `symbol_table` and `scope`.
/// - `prior_linkage` — If a prior visible declaration of the same name
///   exists, its linkage; otherwise `None`.
///
/// # Returns
///
/// The resolved `Linkage` for the new declaration.
pub fn resolve_linkage(
    storage_class: &StorageClass,
    is_file_scope: bool,
    prior_linkage: Option<Linkage>,
) -> Linkage {
    match (storage_class, is_file_scope) {
        // File scope, no storage class or implicit → external linkage.
        // Functions and non-static variables at file scope default to external.
        (StorageClass::Auto, true) | (StorageClass::Register, true) => {
            // `auto` and `register` are technically invalid at file scope in
            // strict C11, but the semantic analyzer emits a diagnostic for
            // this. For linkage purposes we treat them like no-storage-class
            // → External.
            Linkage::External
        }

        // File scope, static → internal linkage.
        (StorageClass::Static, true) => Linkage::Internal,

        // File scope, extern → inherit prior linkage if any, else external.
        // C11 §6.2.2p4: "If the declaration of an identifier for a function
        // has no storage-class specifier, its linkage is determined exactly
        // as if it were declared with the storage-class specifier extern."
        (StorageClass::Extern, true) => {
            prior_linkage.unwrap_or(Linkage::External)
        }

        // Block scope, extern → inherit prior linkage if any, else external.
        // C11 §6.2.2p4: "If ... with the storage-class specifier extern
        // in a scope in which a prior declaration of that identifier is
        // visible, ... the linkage of the identifier at the later
        // declaration is the same as the linkage specified at the prior
        // declaration."
        (StorageClass::Extern, false) => {
            prior_linkage.unwrap_or(Linkage::External)
        }

        // Block scope, static → no linkage (static local variable).
        // The variable has static storage duration but no linkage.
        (StorageClass::Static, false) => Linkage::None,

        // Block scope, no storage class / auto / register → no linkage.
        (StorageClass::Auto, false)
        | (StorageClass::Register, false) => Linkage::None,

        // Typedef → no linkage (type aliases are not objects/functions).
        (StorageClass::Typedef, _) => Linkage::None,

        // ThreadLocal at file scope → external by default (similar to
        // no-storage-class) unless combined with static (which is a
        // separate storage class in our model; the parser handles the
        // combination and sets StorageClass::Static + ThreadLocal).
        (StorageClass::ThreadLocal, true) => {
            prior_linkage.unwrap_or(Linkage::External)
        }

        // ThreadLocal at block scope → no linkage.
        (StorageClass::ThreadLocal, false) => Linkage::None,
    }
}

// ---------------------------------------------------------------------------
// Type compatibility checking for redeclarations
// ---------------------------------------------------------------------------

/// Checks whether two types are compatible for redeclaration merging
/// per C11 §6.2.7 and constructs a composite type where applicable.
///
/// Compatible types can be merged — for example, an incomplete array
/// declaration `extern int arr[];` is compatible with a later complete
/// definition `int arr[10];`, and the composite type has the known size.
///
/// Function types with and without prototypes are also merged: a
/// declaration `int f();` (unprototyped) followed by `int f(int x);`
/// (prototyped) yields the prototyped form.
///
/// # Returns
///
/// - `Ok(composite_type)` if the types are compatible.
/// - `Err(())` if the types are incompatible (diagnostic emitted).
fn check_redeclaration_compatibility(
    existing: &CType,
    new: &CType,
    diagnostics: &mut DiagnosticEngine,
    existing_span: Span,
    new_span: Span,
) -> Result<CType, ()> {
    // Identical types are trivially compatible.
    if types_structurally_equal(existing, new) {
        return Ok(existing.clone());
    }

    // Function type merging: merge prototyped with unprototyped forms.
    if existing.is_function() && new.is_function() {
        return merge_function_types(existing, new, diagnostics, existing_span, new_span);
    }

    // Array type merging: merge known and unknown sizes.
    if existing.is_array() && new.is_array() {
        return merge_array_types(existing, new, diagnostics, existing_span, new_span);
    }

    // Pointer type: check pointee compatibility.
    if let (CType::Pointer(pointee_a), CType::Pointer(pointee_b)) = (existing, new) {
        let merged_pointee = check_redeclaration_compatibility(
            pointee_a, pointee_b, diagnostics, existing_span, new_span,
        )?;
        return Ok(CType::Pointer(Box::new(merged_pointee)));
    }

    // Types are incompatible.
    diagnostics.error(
        new_span,
        "conflicting types for redeclaration",
    );
    Err(())
}

/// Performs a structural equality check on two `CType` values.
///
/// This is a deep comparison that recurses into pointer targets, array
/// element types, function parameter types, and struct/union tag names.
/// It covers all `CType` struct-variant forms with named fields.
fn types_structurally_equal(a: &CType, b: &CType) -> bool {
    match (a, b) {
        (CType::Void, CType::Void)
        | (CType::Bool, CType::Bool)
        | (CType::Float, CType::Float)
        | (CType::Double, CType::Double)
        | (CType::LongDouble, CType::LongDouble) => true,

        // Integer types with signedness.
        (CType::Char { signed: sa }, CType::Char { signed: sb }) => sa == sb,
        (CType::Short { signed: sa }, CType::Short { signed: sb }) => sa == sb,
        (CType::Int { signed: sa }, CType::Int { signed: sb }) => sa == sb,
        (CType::Long { signed: sa }, CType::Long { signed: sb }) => sa == sb,
        (CType::LongLong { signed: sa }, CType::LongLong { signed: sb }) => sa == sb,

        // Complex type — compare base type.
        (CType::Complex(inner_a), CType::Complex(inner_b)) => {
            types_structurally_equal(inner_a, inner_b)
        }

        // Pointer — compare pointee type.
        (CType::Pointer(target_a), CType::Pointer(target_b)) => {
            types_structurally_equal(target_a, target_b)
        }

        // Array — compare element type and size.
        (
            CType::Array { element: elem_a, size: size_a },
            CType::Array { element: elem_b, size: size_b },
        ) => size_a == size_b && types_structurally_equal(elem_a, elem_b),

        // Function — compare return type, params, and variadic flag.
        (
            CType::Function { return_type: ret_a, params: params_a, variadic: var_a },
            CType::Function { return_type: ret_b, params: params_b, variadic: var_b },
        ) => {
            if var_a != var_b {
                return false;
            }
            if !types_structurally_equal(ret_a, ret_b) {
                return false;
            }
            if params_a.len() != params_b.len() {
                return false;
            }
            params_a
                .iter()
                .zip(params_b.iter())
                .all(|(pa, pb)| types_structurally_equal(pa, pb))
        }

        // Struct — tag name identity suffices for redeclaration.
        (
            CType::Struct { name: name_a, .. },
            CType::Struct { name: name_b, .. },
        ) => name_a == name_b,

        // Union — tag name identity suffices.
        (
            CType::Union { name: name_a, .. },
            CType::Union { name: name_b, .. },
        ) => name_a == name_b,

        // Enum — tag name identity suffices.
        (
            CType::Enum { name: name_a, .. },
            CType::Enum { name: name_b, .. },
        ) => name_a == name_b,

        // Atomic — compare inner type.
        (CType::Atomic(inner_a), CType::Atomic(inner_b)) => {
            types_structurally_equal(inner_a, inner_b)
        }

        // Typedef — same name suffices for redeclaration.
        (
            CType::Typedef { name: name_a, .. },
            CType::Typedef { name: name_b, .. },
        ) => name_a == name_b,

        _ => false,
    }
}

/// Merges two function types, one of which may be unprototyped.
///
/// C11 §6.2.7p3: A function with no parameter type list is compatible
/// with a function having a parameter type list, provided each parameter
/// type is compatible with the default argument promotions.
fn merge_function_types(
    existing: &CType,
    new: &CType,
    diagnostics: &mut DiagnosticEngine,
    existing_span: Span,
    new_span: Span,
) -> Result<CType, ()> {
    if let (
        CType::Function { return_type: ret_a, params: params_a, variadic: var_a },
        CType::Function { return_type: ret_b, params: params_b, variadic: var_b },
    ) = (existing, new)
    {
        // Return types must be compatible.
        let merged_ret = check_redeclaration_compatibility(
            ret_a, ret_b, diagnostics, existing_span, new_span,
        )?;

        // If one has no parameters (unprototyped), use the other's parameters.
        let (merged_params, merged_variadic) = if params_a.is_empty() && !params_b.is_empty() {
            (params_b.clone(), *var_b)
        } else if params_b.is_empty() && !params_a.is_empty() {
            (params_a.clone(), *var_a)
        } else if params_a.len() == params_b.len() && var_a == var_b {
            // Both prototyped: merge each parameter type.
            let mut merged = Vec::with_capacity(params_a.len());
            for (pa, pb) in params_a.iter().zip(params_b.iter()) {
                merged.push(check_redeclaration_compatibility(
                    pa, pb, diagnostics, existing_span, new_span,
                )?);
            }
            (merged, *var_a)
        } else {
            diagnostics.error(
                new_span,
                "conflicting function types in redeclaration",
            );
            return Err(());
        };

        return Ok(CType::Function {
            return_type: Box::new(merged_ret),
            params: merged_params,
            variadic: merged_variadic,
        });
    }
    // Should not be reached if callers check is_function().
    Err(())
}

/// Merges two array types where one may have unknown size.
///
/// C11 §6.2.7p1: An array of unknown size is compatible with an array
/// of known size if the element types are compatible. The composite
/// type has the known size.
fn merge_array_types(
    existing: &CType,
    new: &CType,
    diagnostics: &mut DiagnosticEngine,
    existing_span: Span,
    new_span: Span,
) -> Result<CType, ()> {
    if let (
        CType::Array { element: elem_a, size: size_a },
        CType::Array { element: elem_b, size: size_b },
    ) = (existing, new)
    {
        let merged_elem = check_redeclaration_compatibility(
            elem_a, elem_b, diagnostics, existing_span, new_span,
        )?;

        // Merge sizes: known overrides unknown; both known must match.
        let merged_size = match (size_a, size_b) {
            (Some(sa), Some(sb)) if sa == sb => Some(*sa),
            (Some(sa), None) => Some(*sa),
            (None, Some(sb)) => Some(*sb),
            (None, None) => None,
            _ => {
                diagnostics.error(
                    new_span,
                    "conflicting array sizes in redeclaration",
                );
                return Err(());
            }
        };

        return Ok(CType::Array {
            element: Box::new(merged_elem),
            size: merged_size,
        });
    }
    Err(())
}

// ---------------------------------------------------------------------------
// SymbolTable — the central symbol registry for a translation unit
// ---------------------------------------------------------------------------

/// Central registry of all declared symbols within a translation unit.
///
/// The symbol table stores `SymbolEntry` records in a flat `Vec`, indexed
/// by compact `SymbolId` handles. It provides insertion, lookup, and
/// iteration methods, as well as declaration/definition merging logic
/// per C11 §6.2.2 and §6.9.2.
///
/// # Architecture
///
/// The symbol table is intentionally **flat** — it does not maintain
/// scope information itself. Scope management (block, function, file,
/// global) is handled by the separate `scope` module, which maps
/// names to `SymbolId` handles. This separation allows the symbol
/// table to provide O(1) lookup by ID while the scope module handles
/// the stack of visibility contexts.
///
/// # Name-to-ID Auxiliary Index
///
/// An `FxHashMap<Symbol, Vec<SymbolId>>` auxiliary index tracks all
/// symbol IDs that share the same interned name. This enables efficient
/// linkage resolution during `extern` redeclaration, where the compiler
/// must find a prior visible declaration of the same name at file scope.
pub struct SymbolTable {
    /// All symbol entries, indexed by `SymbolId`.
    symbols: Vec<SymbolEntry>,

    /// Auxiliary index: interned name → list of SymbolIds with that name.
    /// Used for linkage resolution during `extern` redeclaration merging.
    name_index: FxHashMap<Symbol, Vec<SymbolId>>,
}

impl SymbolTable {
    /// Creates a new, empty symbol table.
    pub fn new() -> Self {
        SymbolTable {
            symbols: Vec::with_capacity(1024),
            name_index: FxHashMap::default(),
        }
    }

    /// Inserts a new symbol entry and returns its unique `SymbolId`.
    ///
    /// The entry is appended to the end of the internal vector, and the
    /// name-to-ID auxiliary index is updated.
    pub fn insert(&mut self, entry: SymbolEntry) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u32);
        let name = entry.name;
        self.symbols.push(entry);

        // Update the name-to-ID auxiliary index. If this is the first
        // symbol with this name, create a new vector; otherwise append.
        if self.name_index.contains_key(&name) {
            // Name already present — push the new ID onto the existing list.
            self.name_index.get_mut(&name).unwrap().push(id);
        } else {
            // First occurrence — create a new entry in the index.
            self.name_index.insert(name, vec![id]);
        }
        id
    }

    /// Returns `true` if any symbol with the given interned name exists
    /// in the table.
    #[inline]
    pub fn contains_name(&self, name: &Symbol) -> bool {
        self.name_index.contains_key(name)
    }

    /// Returns an immutable reference to the symbol entry for `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds (i.e. was not produced by this table).
    #[inline]
    pub fn get(&self, id: SymbolId) -> &SymbolEntry {
        &self.symbols[id.as_usize()]
    }

    /// Returns a mutable reference to the symbol entry for `id`.
    ///
    /// This is used by the attribute handler to propagate validated GCC
    /// attributes onto an existing symbol, and by declaration merging
    /// to update tentative definitions.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, id: SymbolId) -> &mut SymbolEntry {
        &mut self.symbols[id.as_usize()]
    }

    /// Returns the total number of symbol entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Returns `true` if the symbol table contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    // -------------------------------------------------------------------
    // Declaration / definition merging (C11 §6.2.2, §6.9.2)
    // -------------------------------------------------------------------

    /// Merges a new declaration or definition with an existing symbol.
    ///
    /// This implements the C11 redeclaration rules:
    ///
    /// - **extern → extern:** Types must be compatible; keep existing entry,
    ///   update type to composite type.
    /// - **extern → definition:** Update to definition status; types must
    ///   be compatible.
    /// - **definition → extern:** Keep definition; check type compatibility.
    /// - **definition → definition:** Error — multiple definitions.
    /// - **static → extern:** Error — conflicting linkage (C11 §6.2.2p7).
    /// - **tentative → definition:** Upgrade tentative to definition.
    /// - **tentative → tentative:** Merge types, keep tentative.
    ///
    /// Attributes from the new declaration are merged into the existing
    /// symbol's attribute set.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the merge succeeded.
    /// - `Err(())` if the redeclaration is invalid (diagnostic emitted).
    pub fn merge_declaration(
        &mut self,
        existing_id: SymbolId,
        new_entry: &SymbolEntry,
        diagnostics: &mut DiagnosticEngine,
    ) -> Result<(), ()> {
        let existing = &self.symbols[existing_id.as_usize()];
        let existing_span = existing.span;
        let existing_is_def = existing.is_definition;
        let existing_is_tentative = existing.is_tentative;
        let existing_linkage = existing.linkage;
        let existing_storage = existing.storage_class;
        let existing_ty = existing.ty.clone();

        let new_is_def = new_entry.is_definition;
        let new_linkage = new_entry.linkage;
        let new_storage = new_entry.storage_class;
        let new_span = new_entry.span;
        let new_ty = &new_entry.ty;

        // ------------------------------------------------------------------
        // Rule: typedef → anything or anything → typedef
        // Typedef redeclarations must declare the exact same type.
        // ------------------------------------------------------------------
        if existing_storage == StorageClass::Typedef || new_storage == StorageClass::Typedef {
            if existing_storage != new_storage {
                diagnostics.error(
                    new_span,
                    "redeclaration of typedef as a different kind of symbol",
                );
                return Err(());
            }
            // Both are typedefs — check type compatibility.
            let _merged = check_redeclaration_compatibility(
                &existing_ty, new_ty, diagnostics, existing_span, new_span,
            )?;
            // Typedefs don't need further merging; the type is identical.
            // Merge attributes from the new declaration.
            let attrs = new_entry.attributes.clone();
            self.symbols[existing_id.as_usize()]
                .attributes
                .merge_from(&attrs);
            return Ok(());
        }

        // ------------------------------------------------------------------
        // Rule: conflicting linkage — static (internal) vs extern (external)
        // C11 §6.2.2p7: If a prior declaration at file scope has internal
        // linkage, a later extern declaration is a constraint violation.
        // ------------------------------------------------------------------
        if existing_linkage == Linkage::Internal && new_linkage == Linkage::External {
            diagnostics.error(
                new_span,
                "static declaration followed by non-static declaration",
            );
            return Err(());
        }
        if existing_linkage == Linkage::External && new_linkage == Linkage::Internal {
            diagnostics.error(
                new_span,
                "non-static declaration followed by static declaration",
            );
            return Err(());
        }

        // ------------------------------------------------------------------
        // Check type compatibility and compute composite type.
        // ------------------------------------------------------------------
        let composite_ty = check_redeclaration_compatibility(
            &existing_ty, new_ty, diagnostics, existing_span, new_span,
        )?;

        // ------------------------------------------------------------------
        // Rule: multiple definitions
        // ------------------------------------------------------------------
        if existing_is_def && new_is_def {
            diagnostics.error(
                new_span,
                "redefinition of symbol",
            );
            return Err(());
        }

        // ------------------------------------------------------------------
        // Rule: definitions require complete types (C11 §6.7p7).
        // A definition of an object requires that its type be complete
        // at the point of definition (arrays with unknown size are
        // completed by an initializer or by a later compatible declaration).
        // Functions may be declared with incomplete return types for forward
        // declarations but must be complete at the definition.
        // ------------------------------------------------------------------
        if new_is_def && !composite_ty.is_complete() && !composite_ty.is_function() {
            diagnostics.error(
                new_span,
                "definition of variable with incomplete type",
            );
            return Err(());
        }

        // ------------------------------------------------------------------
        // Apply the merge to the existing entry.
        // ------------------------------------------------------------------
        let entry = &mut self.symbols[existing_id.as_usize()];
        entry.ty = composite_ty;

        // Upgrade from declaration/tentative to definition.
        if new_is_def {
            entry.is_definition = true;
            entry.is_tentative = false;
            entry.span = new_span;
        } else if new_entry.is_tentative && !existing_is_def {
            // Both tentative: keep tentative status, but update span to
            // the latest tentative declaration.
            entry.is_tentative = true;
            entry.span = new_span;
        }

        // Merge attributes from the new declaration.
        let new_attrs = new_entry.attributes.clone();
        entry.attributes.merge_from(&new_attrs);

        Ok(())
    }

    // -------------------------------------------------------------------
    // Weak symbol handling
    // -------------------------------------------------------------------

    /// Marks the given symbol as having weak binding.
    ///
    /// Weak symbols can be overridden by a strong definition at link time.
    /// If the symbol already has a strong (non-weak) definition in the same
    /// translation unit, a warning is emitted (it's valid but unusual).
    pub fn mark_weak(&mut self, id: SymbolId) {
        self.symbols[id.as_usize()].attributes.is_weak = true;
    }

    // -------------------------------------------------------------------
    // Iterators and query methods
    // -------------------------------------------------------------------

    /// Returns an iterator over all symbol entries.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &SymbolEntry> {
        self.symbols.iter()
    }

    /// Returns an iterator over only the externally-visible symbols
    /// (those with `Linkage::External`).
    pub fn external_symbols(&self) -> impl Iterator<Item = &SymbolEntry> {
        self.symbols
            .iter()
            .filter(|s| s.linkage == Linkage::External)
    }

    /// Returns an iterator over only the definitions (not mere declarations).
    pub fn definitions(&self) -> impl Iterator<Item = &SymbolEntry> {
        self.symbols.iter().filter(|s| s.is_definition)
    }

    /// Returns an iterator over symbols that are declared but never defined.
    ///
    /// Useful for detecting missing definitions and emitting linker-like
    /// "undefined reference" diagnostics within a single translation unit.
    pub fn undefined_symbols(&self) -> impl Iterator<Item = &SymbolEntry> {
        self.symbols
            .iter()
            .filter(|s| !s.is_definition && !s.is_tentative)
    }

    /// Returns an iterator over (SymbolId, &SymbolEntry) pairs.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (SymbolId, &SymbolEntry)> {
        self.symbols
            .iter()
            .enumerate()
            .map(|(i, e)| (SymbolId(i as u32), e))
    }

    /// Looks up all `SymbolId`s with the given interned name.
    ///
    /// Returns an empty slice if no symbol with this name has been inserted.
    pub fn lookup_by_name(&self, name: Symbol) -> &[SymbolId] {
        self.name_index
            .get(&name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Finds the most recent file-scope declaration of the given name,
    /// if any, by scanning the name index in reverse order.
    ///
    /// This is used during linkage resolution to find a prior visible
    /// declaration at file scope for `extern` redeclaration merging.
    pub fn find_file_scope_declaration(&self, name: Symbol) -> Option<SymbolId> {
        let ids = self.lookup_by_name(name);
        // Iterate in reverse so the most recent declaration wins.
        for &id in ids.iter().rev() {
            let entry = &self.symbols[id.as_usize()];
            // File-scope declarations have external or internal linkage.
            if entry.linkage == Linkage::External || entry.linkage == Linkage::Internal {
                return Some(id);
            }
        }
        None
    }

    /// Finalizes all tentative definitions at the end of a translation unit.
    ///
    /// C11 §6.9.2: "If ... a translation unit contains one or more
    /// tentative definitions for an identifier, and the translation unit
    /// contains no external definition for that identifier, then the
    /// behavior is exactly as if the translation unit contains a file scope
    /// declaration of that identifier, with the composite type as of the
    /// end of the translation unit, with an initializer equal to 0."
    ///
    /// This function promotes all remaining tentative definitions to
    /// full definitions.
    pub fn finalize_tentative_definitions(&mut self) {
        for entry in &mut self.symbols {
            if entry.is_tentative && !entry.is_definition {
                entry.is_definition = true;
                entry.is_tentative = false;
            }
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
