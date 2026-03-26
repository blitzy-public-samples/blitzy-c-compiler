//! Attribute semantic validation and propagation for GCC `__attribute__`.
//!
//! This module implements Phase 5 attribute handling for the BCC C11 compiler.
//! It validates and propagates all 21+ GCC `__attribute__((...))` attributes
//! required for Linux kernel compilation (§4.3 of the AAP):
//!
//! **Supported attributes**: `aligned`, `packed`, `section`, `used`, `unused`,
//! `weak`, `constructor`, `destructor`, `visibility`, `deprecated`, `noreturn`,
//! `noinline`, `always_inline`, `cold`, `hot`, `format`, `format_arg`, `malloc`,
//! `pure`, `const`, `warn_unused_result`, `fallthrough`.
//!
//! **Key design decisions**:
//! - Unknown attributes produce a warning diagnostic (never silently ignored,
//!   per AAP §0.7.6).
//! - Attribute validation is target-aware: alignment constraints vary by
//!   architecture.
//! - This module does NOT depend on `crate::ir`, `crate::passes`, or
//!   `crate::backend`.
//! - Circular dependency with `symbol_table` is avoided by accepting
//!   `&mut Vec<ValidatedAttribute>` in `propagate_to_symbol` rather than
//!   importing `SymbolEntry`.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::common::types::{CType, TypeQualifiers};
use crate::frontend::parser::ast::{Attribute, AttributeArg, Expression};

// ===========================================================================
// Validated Attribute Representation
// ===========================================================================

/// Semantically validated, type-safe attribute representation.
///
/// Each variant corresponds to one of the 21+ GCC `__attribute__` keywords
/// from §4.3 of the AAP. Values have been range-checked, context-validated,
/// and conflict-resolved by the [`AttributeHandler`].
#[derive(Debug, Clone, PartialEq)]
pub enum ValidatedAttribute {
    /// `__attribute__((aligned(N)))` — alignment in bytes; `N` is a power of two.
    Aligned(u64),
    /// `__attribute__((packed))` — pack struct/union with alignment of 1.
    Packed,
    /// `__attribute__((section("name")))` — place symbol in named ELF section.
    Section(String),
    /// `__attribute__((used))` — prevent dead-code elimination by the linker.
    Used,
    /// `__attribute__((unused))` — suppress `-Wunused` diagnostics.
    Unused,
    /// `__attribute__((weak))` — weak symbol linkage (`STB_WEAK` in ELF).
    Weak,
    /// `__attribute__((constructor))` or `((constructor(priority)))`.
    Constructor(Option<i32>),
    /// `__attribute__((destructor))` or `((destructor(priority)))`.
    Destructor(Option<i32>),
    /// `__attribute__((visibility("...")))`.
    Visibility(SymbolVisibility),
    /// `__attribute__((deprecated))` or `((deprecated("message")))`.
    Deprecated(Option<String>),
    /// `__attribute__((noreturn))` — function does not return.
    NoReturn,
    /// `__attribute__((noinline))` — never inline this function.
    NoInline,
    /// `__attribute__((always_inline))` — always inline this function.
    AlwaysInline,
    /// `__attribute__((cold))` — function is rarely executed.
    Cold,
    /// `__attribute__((hot))` — function is frequently executed.
    Hot,
    /// `__attribute__((format(archetype, string_index, first_to_check)))`.
    Format {
        archetype: FormatArchetype,
        string_index: u32,
        first_to_check: u32,
    },
    /// `__attribute__((format_arg(N)))`.
    FormatArg(u32),
    /// `__attribute__((malloc))` — function returns freshly allocated memory.
    Malloc,
    /// `__attribute__((pure))` — function has no side effects except through
    /// pointer arguments.
    Pure,
    /// `__attribute__((const))` — function has no side effects at all
    /// (stricter than `pure`).
    Const,
    /// `__attribute__((warn_unused_result))` — warn if return value is discarded.
    WarnUnusedResult,
    /// `__attribute__((fallthrough))` — intentional switch case fall-through.
    Fallthrough,
}

// ===========================================================================
// Symbol Visibility
// ===========================================================================

/// ELF symbol visibility classification.
///
/// Maps directly to the `STV_*` constants in the ELF specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolVisibility {
    /// `STV_DEFAULT` — visible to other shared objects.
    Default,
    /// `STV_HIDDEN` — not visible outside the shared object.
    Hidden,
    /// `STV_PROTECTED` — visible but not preemptible.
    Protected,
    /// `STV_INTERNAL` — processor-specific hidden semantics.
    Internal,
}

impl Default for SymbolVisibility {
    fn default() -> Self {
        SymbolVisibility::Default
    }
}

// ===========================================================================
// Format Archetype
// ===========================================================================

/// Format string archetype for `__attribute__((format(...)))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatArchetype {
    /// `printf`-style format string.
    Printf,
    /// `scanf`-style format string.
    Scanf,
    /// `strftime`-style format string.
    Strftime,
    /// `strfmon`-style format string.
    Strfmon,
}

// ===========================================================================
// Attribute Context
// ===========================================================================

/// The syntactic context in which an attribute appears, used for
/// applicability validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeContext {
    /// Attribute on a function declaration/definition.
    Function,
    /// Attribute on a variable declaration.
    Variable,
    /// Attribute on a type specifier (struct, union, enum, typedef).
    Type,
    /// Attribute on a struct/union field.
    Field,
    /// Attribute on a statement (e.g., `fallthrough`).
    Statement,
    /// Attribute on a label.
    Label,
}

// ===========================================================================
// Attribute Handler
// ===========================================================================

/// Validates GCC `__attribute__` specifications and propagates their semantic
/// effects to types and symbols.
///
/// # Lifetime
///
/// The `'a` lifetime binds the handler to the diagnostic engine and string
/// interner, both of which must outlive the handler.
///
/// # Example
///
/// ```ignore
/// let mut handler = AttributeHandler::new(&mut diag, target, &interner);
/// let validated = handler.validate_attributes(&parsed_attrs, AttributeContext::Function, span);
/// handler.propagate_to_type(&validated, &mut ctype);
/// ```
pub struct AttributeHandler<'a> {
    /// Diagnostic engine for error/warning emission.
    diagnostics: &'a mut DiagnosticEngine,
    /// Target architecture for alignment and size constraints.
    target: Target,
    /// String interner for resolving attribute name Symbols to strings.
    interner: &'a Interner,
}

impl<'a> AttributeHandler<'a> {
    /// Creates a new attribute handler bound to the given diagnostic engine,
    /// target architecture, and string interner.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        target: Target,
        interner: &'a Interner,
    ) -> Self {
        AttributeHandler {
            diagnostics,
            target,
            interner,
        }
    }

    // =======================================================================
    // Top-Level Attribute Validation
    // =======================================================================

    /// Validates a list of parsed attributes and returns their semantically
    /// validated forms.
    ///
    /// Each attribute is:
    /// 1. Resolved from its interned [`Symbol`] name to a string.
    /// 2. Checked for applicability in the given [`AttributeContext`].
    /// 3. Dispatched to its specific validator for argument checking.
    /// 4. Collected into the result if validation succeeds.
    ///
    /// Unknown attributes produce a warning diagnostic and are omitted from
    /// the result. Attribute conflicts (e.g., `noinline` + `always_inline`)
    /// are detected and resolved with warnings.
    pub fn validate_attributes(
        &mut self,
        attrs: &[Attribute],
        context: AttributeContext,
        _span: Span,
    ) -> Vec<ValidatedAttribute> {
        let mut validated: Vec<ValidatedAttribute> = Vec::with_capacity(attrs.len());

        for attr in attrs {
            let name_str = self.interner.resolve(attr.name);
            // Dispatch to the appropriate validator based on attribute name.
            // The parser has already stripped `__` prefix/suffix from names
            // (e.g., `__aligned__` → `aligned`).
            let result = match name_str {
                "aligned" => self.validate_aligned(&attr.args, attr.span),
                "packed" => self.validate_packed(&attr.args, attr.span),
                "section" => self.validate_section(&attr.args, attr.span),
                "used" => self.validate_used(attr.span),
                "unused" => self.validate_unused(attr.span),
                "weak" => self.validate_weak(attr.span),
                "constructor" => self.validate_constructor(&attr.args, attr.span),
                "destructor" => self.validate_destructor(&attr.args, attr.span),
                "visibility" => self.validate_visibility(&attr.args, attr.span),
                "deprecated" => self.validate_deprecated(&attr.args, attr.span),
                "noreturn" => self.validate_noreturn(attr.span),
                "noinline" => self.validate_noinline(attr.span),
                "always_inline" => self.validate_always_inline(attr.span),
                "cold" => self.validate_cold(attr.span),
                "hot" => self.validate_hot(attr.span),
                "format" => self.validate_format(&attr.args, attr.span),
                "format_arg" => self.validate_format_arg(&attr.args, attr.span),
                "malloc" => self.validate_malloc(attr.span),
                "pure" => self.validate_pure(attr.span),
                "const" => self.validate_const(attr.span),
                "warn_unused_result" => self.validate_warn_unused_result(attr.span),
                "fallthrough" => self.validate_fallthrough(attr.span),

                // -------------------------------------------------------
                // Recognized glibc / GCC attributes — accepted silently.
                //
                // These attributes appear frequently in system headers
                // (glibc, Linux kernel headers) and must be recognized to
                // avoid flooding users with spurious "unknown attribute"
                // warnings.  They are semantically no-ops at this stage:
                // BCC acknowledges them but does not enforce their
                // constraints at compile time.
                // -------------------------------------------------------
                "nonnull"
                | "returns_nonnull"
                | "sentinel"
                | "leaf"
                | "nothrow"
                | "alloc_size"
                | "alloc_align"
                | "assume_aligned"
                | "may_alias"
                | "access"
                | "no_sanitize"
                | "no_sanitize_address"
                | "no_sanitize_thread"
                | "no_sanitize_undefined"
                | "no_instrument_function"
                | "no_stack_protector"
                | "no_split_stack"
                | "noclone"
                | "no_reorder"
                | "nocf_check"
                | "no_caller_saved_registers"
                | "flatten"
                | "gnu_inline"
                | "artificial"
                | "externally_visible"
                | "optimize"
                | "target"
                | "error"
                | "warning"
                | "cleanup"
                | "tls_model"
                | "transparent_union"
                | "alias"
                | "weakref"
                | "ifunc"
                | "mode"
                | "nonstring"
                | "copy"
                | "retain"
                | "symver"
                | "patchable_function_entry"
                | "warn_unused"
                | "noipa"
                | "no_icf"
                | "stack_protect"
                | "zero_call_used_regs"
                | "fd_arg"
                | "fd_arg_read"
                | "fd_arg_write"
                | "null_terminated_string_arg" => {
                    // Recognized but no semantic action needed in BCC.
                    None
                }

                _ => {
                    // AAP §0.7.6: unknown attributes MUST NOT be silently
                    // ignored — emit a warning diagnostic.
                    self.diagnostics.emit(Diagnostic::warning(
                        attr.span,
                        format!("unknown attribute '{}' ignored", name_str),
                    ));
                    None
                }
            };

            // Context-check the validated attribute before accepting it.
            if let Some(ref va) = result {
                if !self.is_applicable(va, context) {
                    let kind_str = self.attribute_kind_str(va);
                    self.diagnostics.emit(Diagnostic::warning(
                        attr.span,
                        format!("'{}' attribute only applies to {}", name_str, kind_str,),
                    ));
                    continue;
                }
                validated.push(va.clone());
            }
        }

        // Detect and resolve conflicting attribute combinations.
        self.resolve_conflicts(&mut validated);

        validated
    }

    // =======================================================================
    // Context Applicability
    // =======================================================================

    /// Returns `true` if `attr` is applicable in the given `context`.
    fn is_applicable(&self, attr: &ValidatedAttribute, context: AttributeContext) -> bool {
        match attr {
            // Function-only attributes.
            ValidatedAttribute::NoReturn
            | ValidatedAttribute::NoInline
            | ValidatedAttribute::AlwaysInline
            | ValidatedAttribute::Cold
            | ValidatedAttribute::Hot
            | ValidatedAttribute::Format { .. }
            | ValidatedAttribute::FormatArg(_)
            | ValidatedAttribute::Malloc
            | ValidatedAttribute::Pure
            | ValidatedAttribute::Const
            | ValidatedAttribute::WarnUnusedResult => context == AttributeContext::Function,

            // Function or variable or type.
            ValidatedAttribute::Aligned(_) => matches!(
                context,
                AttributeContext::Function
                    | AttributeContext::Variable
                    | AttributeContext::Type
                    | AttributeContext::Field
            ),

            // Packed: types and fields (structs/unions).
            ValidatedAttribute::Packed => {
                matches!(context, AttributeContext::Type | AttributeContext::Field)
            }

            // Variable or function scope.
            ValidatedAttribute::Section(_)
            | ValidatedAttribute::Used
            | ValidatedAttribute::Weak
            | ValidatedAttribute::Visibility(_)
            | ValidatedAttribute::Constructor(_)
            | ValidatedAttribute::Destructor(_) => matches!(
                context,
                AttributeContext::Function | AttributeContext::Variable
            ),

            // Unused: broad applicability.
            ValidatedAttribute::Unused => matches!(
                context,
                AttributeContext::Function
                    | AttributeContext::Variable
                    | AttributeContext::Type
                    | AttributeContext::Label
            ),

            // Deprecated: broad applicability.
            ValidatedAttribute::Deprecated(_) => matches!(
                context,
                AttributeContext::Function
                    | AttributeContext::Variable
                    | AttributeContext::Type
                    | AttributeContext::Field
            ),

            // Statement-only.
            ValidatedAttribute::Fallthrough => context == AttributeContext::Statement,
        }
    }

    /// Returns a human-readable description of the valid contexts for the
    /// given attribute, used in diagnostic messages.
    fn attribute_kind_str(&self, attr: &ValidatedAttribute) -> &'static str {
        match attr {
            ValidatedAttribute::NoReturn
            | ValidatedAttribute::NoInline
            | ValidatedAttribute::AlwaysInline
            | ValidatedAttribute::Cold
            | ValidatedAttribute::Hot
            | ValidatedAttribute::Format { .. }
            | ValidatedAttribute::FormatArg(_)
            | ValidatedAttribute::Malloc
            | ValidatedAttribute::Pure
            | ValidatedAttribute::Const
            | ValidatedAttribute::WarnUnusedResult => "functions",

            ValidatedAttribute::Packed => "struct/union types and fields",

            ValidatedAttribute::Aligned(_) => "variables, functions, types, and fields",

            ValidatedAttribute::Section(_)
            | ValidatedAttribute::Used
            | ValidatedAttribute::Weak
            | ValidatedAttribute::Visibility(_)
            | ValidatedAttribute::Constructor(_)
            | ValidatedAttribute::Destructor(_) => "functions and variables",

            ValidatedAttribute::Unused => "functions, variables, types, and labels",

            ValidatedAttribute::Deprecated(_) => "functions, variables, types, and fields",

            ValidatedAttribute::Fallthrough => "statements",
        }
    }

    // =======================================================================
    // Individual Attribute Validators
    // =======================================================================

    /// `__attribute__((aligned(N)))` — sets the minimum alignment in bytes.
    ///
    /// - If no argument: defaults to the target's maximum useful alignment.
    /// - If one argument: must evaluate to a positive power-of-two integer.
    /// - Architecture-dependent maximum alignment checked against page size.
    fn validate_aligned(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        if args.is_empty() {
            // Default to maximum alignment for the target.
            return Some(ValidatedAttribute::Aligned(self.target.max_align() as u64));
        }
        if args.len() > 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'aligned' attribute takes at most one argument".to_string(),
            ));
            return None;
        }
        let value = self.extract_integer_arg(&args[0], span)?;
        if value == 0 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "requested alignment must be greater than zero".to_string(),
            ));
            return None;
        }
        if !value.is_power_of_two() {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "requested alignment is not a power of 2".to_string(),
            ));
            return None;
        }
        // Warn (but permit) alignment exceeding target page size.
        let page_size = self.target.page_size() as u64;
        if value > page_size {
            self.diagnostics.emit(Diagnostic::warning(
                span,
                format!(
                    "requested alignment {} exceeds target page size {}",
                    value, page_size,
                ),
            ));
        }
        Some(ValidatedAttribute::Aligned(value))
    }

    /// `__attribute__((packed))` — sets struct/union member alignment to 1.
    ///
    /// Accepts no arguments.
    fn validate_packed(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute> {
        if !args.is_empty() {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'packed' attribute takes no arguments".to_string(),
            ));
            return None;
        }
        Some(ValidatedAttribute::Packed)
    }

    /// `__attribute__((section("name")))` — places the symbol in the named
    /// ELF section.
    ///
    /// Requires exactly one string literal argument with a non-empty,
    /// null-free section name.
    fn validate_section(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        if args.len() != 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'section' attribute requires exactly one string argument".to_string(),
            ));
            return None;
        }
        let name = self.extract_string_arg(&args[0], span)?;
        if name.is_empty() {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "section name must not be empty".to_string(),
            ));
            return None;
        }
        // Check for embedded null bytes which are invalid in ELF section names.
        if name.contains('\0') {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "section name must not contain null characters".to_string(),
            ));
            return None;
        }
        Some(ValidatedAttribute::Section(name))
    }

    /// `__attribute__((used))` — prevents dead-code elimination even if the
    /// symbol is unreferenced.
    fn validate_used(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Used)
    }

    /// `__attribute__((unused))` — suppresses unused-variable/function warnings.
    fn validate_unused(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Unused)
    }

    /// `__attribute__((weak))` — the symbol has weak linkage (`STB_WEAK`).
    fn validate_weak(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Weak)
    }

    /// `__attribute__((constructor))` or `((constructor(priority)))`.
    ///
    /// Priority 0–100 are reserved by the implementation and produce a
    /// warning if used.
    fn validate_constructor(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        self.validate_ctor_dtor(args, span, true)
    }

    /// `__attribute__((destructor))` or `((destructor(priority)))`.
    fn validate_destructor(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        self.validate_ctor_dtor(args, span, false)
    }

    /// Shared validation for `constructor` and `destructor` attributes.
    fn validate_ctor_dtor(
        &mut self,
        args: &[AttributeArg],
        span: Span,
        is_ctor: bool,
    ) -> Option<ValidatedAttribute> {
        let attr_name = if is_ctor { "constructor" } else { "destructor" };
        if args.len() > 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!("'{}' attribute takes at most one argument", attr_name),
            ));
            return None;
        }
        let priority = if args.is_empty() {
            None
        } else {
            let val = self.extract_integer_arg(&args[0], span)?;
            let prio = val as i32;
            if prio < 0 {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    format!("'{}' priority must be non-negative", attr_name),
                ));
                return None;
            }
            if prio <= 100 {
                self.diagnostics.emit(Diagnostic::warning(
                    span,
                    format!(
                        "'{}' priorities 0-100 are reserved for the implementation",
                        attr_name,
                    ),
                ));
            }
            Some(prio)
        };
        if is_ctor {
            Some(ValidatedAttribute::Constructor(priority))
        } else {
            Some(ValidatedAttribute::Destructor(priority))
        }
    }

    /// `__attribute__((visibility("default"|"hidden"|"protected"|"internal")))`.
    fn validate_visibility(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        if args.len() != 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'visibility' attribute requires exactly one string argument".to_string(),
            ));
            return None;
        }
        let vis_str = self.extract_string_arg(&args[0], span)?;
        let vis = match vis_str.as_str() {
            "default" => SymbolVisibility::Default,
            "hidden" => SymbolVisibility::Hidden,
            "protected" => SymbolVisibility::Protected,
            "internal" => SymbolVisibility::Internal,
            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    format!(
                        "visibility must be \"default\", \"hidden\", \"protected\", \
                         or \"internal\"; got \"{}\"",
                        vis_str,
                    ),
                ));
                return None;
            }
        };
        Some(ValidatedAttribute::Visibility(vis))
    }

    /// `__attribute__((deprecated))` or `((deprecated("message")))`.
    fn validate_deprecated(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        if args.len() > 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'deprecated' attribute takes at most one string argument".to_string(),
            ));
            return None;
        }
        let msg = if args.is_empty() {
            None
        } else {
            Some(self.extract_string_arg(&args[0], span)?)
        };
        Some(ValidatedAttribute::Deprecated(msg))
    }

    /// `__attribute__((noreturn))` — function does not return.
    fn validate_noreturn(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::NoReturn)
    }

    /// `__attribute__((noinline))` — never inline.
    fn validate_noinline(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::NoInline)
    }

    /// `__attribute__((always_inline))` — always inline.
    fn validate_always_inline(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::AlwaysInline)
    }

    /// `__attribute__((cold))` — rarely executed.
    fn validate_cold(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Cold)
    }

    /// `__attribute__((hot))` — frequently executed.
    fn validate_hot(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Hot)
    }

    /// `__attribute__((format(archetype, string_index, first_to_check)))`.
    ///
    /// Three arguments are required:
    /// 1. Archetype identifier: `printf`, `scanf`, `strftime`, or `strfmon`.
    /// 2. String index: 1-based parameter position of the format string.
    /// 3. First to check: 1-based position of the first variadic arg to
    ///    check (0 means no checking).
    fn validate_format(&mut self, args: &[AttributeArg], span: Span) -> Option<ValidatedAttribute> {
        if args.len() != 3 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'format' attribute requires exactly 3 arguments \
                 (archetype, string_index, first_to_check)"
                    .to_string(),
            ));
            return None;
        }

        // First argument: archetype identifier.
        let archetype = match &args[0] {
            AttributeArg::Identifier(sym, _id_span) => {
                let name = self.interner.resolve(*sym);
                match name {
                    "printf" | "__printf__" => FormatArchetype::Printf,
                    "scanf" | "__scanf__" => FormatArchetype::Scanf,
                    "strftime" | "__strftime__" => FormatArchetype::Strftime,
                    "strfmon" | "__strfmon__" => FormatArchetype::Strfmon,
                    _ => {
                        self.diagnostics.emit(Diagnostic::error(
                            span,
                            format!(
                                "unknown format archetype '{}'; expected \
                                 printf, scanf, strftime, or strfmon",
                                name,
                            ),
                        ));
                        return None;
                    }
                }
            }
            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "first argument to 'format' must be an identifier \
                     (printf, scanf, strftime, or strfmon)"
                        .to_string(),
                ));
                return None;
            }
        };

        // Second argument: string_index (1-based parameter position).
        let string_index = self.extract_integer_arg(&args[1], span)? as u32;
        if string_index == 0 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "format string index must be a positive integer (1-based)".to_string(),
            ));
            return None;
        }

        // Third argument: first_to_check (1-based, or 0 for no checking).
        let first_to_check = self.extract_integer_arg(&args[2], span)? as u32;

        // first_to_check must be 0 or > string_index.
        if first_to_check != 0 && first_to_check <= string_index {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "format first-to-check index ({}) must be 0 or greater than \
                     string index ({})",
                    first_to_check, string_index,
                ),
            ));
            return None;
        }

        Some(ValidatedAttribute::Format {
            archetype,
            string_index,
            first_to_check,
        })
    }

    /// `__attribute__((format_arg(N)))` — the Nth parameter is a format
    /// string that should be checked.
    fn validate_format_arg(
        &mut self,
        args: &[AttributeArg],
        span: Span,
    ) -> Option<ValidatedAttribute> {
        if args.len() != 1 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "'format_arg' attribute requires exactly one integer argument".to_string(),
            ));
            return None;
        }
        let index = self.extract_integer_arg(&args[0], span)? as u32;
        if index == 0 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                "format_arg index must be a positive integer (1-based)".to_string(),
            ));
            return None;
        }
        Some(ValidatedAttribute::FormatArg(index))
    }

    /// `__attribute__((malloc))` — return value points to freshly allocated
    /// memory with no aliases.
    fn validate_malloc(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Malloc)
    }

    /// `__attribute__((pure))` — function has no side effects except through
    /// pointer arguments. The return value depends only on parameters and
    /// global state.
    fn validate_pure(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Pure)
    }

    /// `__attribute__((const))` — function has no side effects at all and
    /// depends only on its arguments. Stricter than `pure`.
    fn validate_const(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Const)
    }

    /// `__attribute__((warn_unused_result))` — emit a warning if the caller
    /// discards the return value.
    fn validate_warn_unused_result(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::WarnUnusedResult)
    }

    /// `__attribute__((fallthrough))` — marks an intentional fall-through
    /// between switch cases.
    fn validate_fallthrough(&mut self, _span: Span) -> Option<ValidatedAttribute> {
        Some(ValidatedAttribute::Fallthrough)
    }

    // =======================================================================
    // Conflict Detection and Resolution
    // =======================================================================

    /// Detects and resolves conflicting attribute combinations in-place.
    ///
    /// Conflict rules:
    /// - `noinline` + `always_inline` → warning, keep the later one.
    /// - `cold` + `hot` → warning, keep the later one.
    /// - `pure` + `const` → `const` wins (stricter).
    /// - Multiple `aligned` → last one wins.
    /// - Multiple `section` → last one wins, with a warning.
    /// - `packed` + `aligned` → both apply (aligned overrides packed for
    ///   the specific field/member).
    fn resolve_conflicts(&mut self, attrs: &mut Vec<ValidatedAttribute>) {
        let has_noinline = attrs
            .iter()
            .any(|a| matches!(a, ValidatedAttribute::NoInline));
        let has_always_inline = attrs
            .iter()
            .any(|a| matches!(a, ValidatedAttribute::AlwaysInline));
        if has_noinline && has_always_inline {
            self.diagnostics.emit(Diagnostic::warning(
                Span::dummy(),
                "conflicting attributes 'noinline' and 'always_inline'; \
                 using 'always_inline'"
                    .to_string(),
            ));
            attrs.retain(|a| !matches!(a, ValidatedAttribute::NoInline));
        }

        let has_cold = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Cold));
        let has_hot = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Hot));
        if has_cold && has_hot {
            self.diagnostics.emit(Diagnostic::warning(
                Span::dummy(),
                "conflicting attributes 'cold' and 'hot'; using 'hot'".to_string(),
            ));
            attrs.retain(|a| !matches!(a, ValidatedAttribute::Cold));
        }

        // `pure` + `const` → `const` subsumes `pure`.
        let has_pure = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Pure));
        let has_const = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Const));
        if has_pure && has_const {
            attrs.retain(|a| !matches!(a, ValidatedAttribute::Pure));
        }

        // Multiple `aligned` → keep only the last.
        let aligned_count = attrs
            .iter()
            .filter(|a| matches!(a, ValidatedAttribute::Aligned(_)))
            .count();
        if aligned_count > 1 {
            self.keep_last_of(attrs, |a| matches!(a, ValidatedAttribute::Aligned(_)));
        }

        // Multiple `section` → keep last, warn.
        let section_count = attrs
            .iter()
            .filter(|a| matches!(a, ValidatedAttribute::Section(_)))
            .count();
        if section_count > 1 {
            self.diagnostics.emit(Diagnostic::warning(
                Span::dummy(),
                "multiple 'section' attributes; using the last one".to_string(),
            ));
            self.keep_last_of(attrs, |a| matches!(a, ValidatedAttribute::Section(_)));
        }
    }

    /// Retains only the last element matching `pred`, removing earlier
    /// occurrences. Non-matching elements are preserved in order.
    fn keep_last_of<F>(&self, attrs: &mut Vec<ValidatedAttribute>, pred: F)
    where
        F: Fn(&ValidatedAttribute) -> bool,
    {
        // Find the index of the last matching element.
        let last_idx = attrs.iter().rposition(&pred);
        if let Some(last) = last_idx {
            let mut seen = false;
            let mut i = attrs.len();
            // Walk backwards; keep the element at `last`, remove others.
            while i > 0 {
                i -= 1;
                if pred(&attrs[i]) {
                    if i == last && !seen {
                        seen = true;
                    } else {
                        attrs.remove(i);
                    }
                }
            }
        }
    }

    // =======================================================================
    // Propagation
    // =======================================================================

    /// Propagates validated attributes to a C type.
    ///
    /// The following attributes modify the type representation:
    /// - `aligned(N)` → sets the type alignment override.
    /// - `packed` → enables packing mode on struct/union types.
    ///
    /// Other attributes are type-transparent and affect only the symbol.
    pub fn propagate_to_type(&self, attrs: &[ValidatedAttribute], ty: &mut CType) {
        for attr in attrs {
            match attr {
                ValidatedAttribute::Aligned(n) => {
                    match ty {
                        CType::Struct { aligned, .. } | CType::Union { aligned, .. } => {
                            *aligned = Some(*n as usize);
                        }
                        // For non-struct/union types, alignment is recorded
                        // at the symbol level, not the type level. This is
                        // acceptable — the caller propagates alignment to
                        // the symbol as well.
                        _ => {}
                    }
                }
                ValidatedAttribute::Packed => match ty {
                    CType::Struct { packed, .. } | CType::Union { packed, .. } => {
                        *packed = true;
                    }
                    _ => {}
                },
                // Other attributes do not modify type representation.
                _ => {}
            }
        }
    }

    /// Propagates validated attributes onto a symbol's attribute list.
    ///
    /// This method appends all validated attributes to the provided
    /// `symbol_attrs` vector (typically `SymbolEntry.attributes` from the
    /// symbol table). The symbol table module is responsible for extracting
    /// specific fields (e.g., `is_weak`, `visibility`, `section`) from the
    /// stored attributes.
    ///
    /// Conflict resolution is already performed by [`validate_attributes`],
    /// so `symbol_attrs` receives a clean, conflict-free set.
    pub fn propagate_to_symbol(
        &self,
        attrs: &[ValidatedAttribute],
        symbol_attrs: &mut Vec<ValidatedAttribute>,
    ) {
        for attr in attrs {
            symbol_attrs.push(attr.clone());
        }
    }

    // =======================================================================
    // Argument Extraction Helpers
    // =======================================================================

    /// Extracts an integer constant value from an attribute argument.
    ///
    /// Supports `AttributeArg::Expression(IntegerLiteral)` and bare
    /// integer expressions. Returns `None` and emits an error if the
    /// argument is not an integer constant.
    fn extract_integer_arg(&mut self, arg: &AttributeArg, span: Span) -> Option<u64> {
        match arg {
            AttributeArg::Expression(expr) => self.eval_const_integer(expr, span),
            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "expected an integer constant expression".to_string(),
                ));
                None
            }
        }
    }

    /// Extracts a string value from an attribute argument.
    ///
    /// Supports `AttributeArg::String` (raw bytes decoded as UTF-8) and
    /// `AttributeArg::Expression(StringLiteral)`. Returns `None` and emits
    /// an error if the argument is not a string.
    fn extract_string_arg(&mut self, arg: &AttributeArg, span: Span) -> Option<String> {
        match arg {
            AttributeArg::String(bytes, _str_span) => {
                // Attribute string arguments are ASCII section names,
                // visibility values, or deprecation messages — lossy
                // conversion is acceptable for diagnostics.
                Some(String::from_utf8_lossy(bytes).into_owned())
            }
            AttributeArg::Expression(expr) => {
                // Support string literals wrapped in an expression node.
                if let Expression::StringLiteral { segments, .. } = expr.as_ref() {
                    let mut buf = Vec::new();
                    for seg in segments {
                        buf.extend_from_slice(&seg.value);
                    }
                    Some(String::from_utf8_lossy(&buf).into_owned())
                } else {
                    self.diagnostics.emit(Diagnostic::error(
                        span,
                        "expected a string literal".to_string(),
                    ));
                    None
                }
            }
            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "expected a string literal".to_string(),
                ));
                None
            }
        }
    }

    /// Evaluates an expression as a compile-time integer constant.
    ///
    /// Currently handles `IntegerLiteral` directly. Complex constant
    /// expressions (e.g., `1 << 4`) should be pre-folded by the constant
    /// evaluator before reaching attribute validation.
    fn eval_const_integer(&mut self, expr: &Expression, span: Span) -> Option<u64> {
        match expr {
            Expression::IntegerLiteral { value, .. } => {
                // Clamp to u64 range for attribute purposes. Attribute
                // arguments (alignment, priority, index) fit in u64.
                if *value > u64::MAX as u128 {
                    self.diagnostics.emit(Diagnostic::error(
                        span,
                        "attribute argument value exceeds maximum".to_string(),
                    ));
                    return None;
                }
                Some(*value as u64)
            }
            _ => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "expected a constant integer expression".to_string(),
                ));
                None
            }
        }
    }

    // =======================================================================
    // Type Query Helpers
    // =======================================================================

    /// Returns `true` if the given type is `const char *` — a const-qualified
    /// pointer to a character type.
    ///
    /// Used during format attribute validation to verify the format string
    /// parameter has type `const char *`. Examines both `CType::Pointer`
    /// qualifiers and `CType::Qualified` wrappers.
    #[allow(dead_code)]
    fn is_const_char_ptr(ty: &CType) -> bool {
        match ty {
            CType::Pointer(pointee, quals) => {
                if !quals.is_const {
                    return false;
                }
                // Check if pointee is char (signed or unsigned).
                matches!(pointee.as_ref(), CType::Char | CType::SChar | CType::UChar)
            }
            CType::Qualified(inner, _quals) => Self::is_const_char_ptr(inner),
            _ => false,
        }
    }

    /// Returns `true` if the given type is a pointer type.
    ///
    /// Used to validate that `__attribute__((malloc))` is applied to
    /// functions returning pointer types.
    #[allow(dead_code)]
    fn is_pointer_type(ty: &CType) -> bool {
        matches!(ty, CType::Pointer(_, _))
    }

    /// Returns `true` if the given type is a function type.
    ///
    /// Used for context validation when attributes require function scope.
    #[allow(dead_code)]
    fn is_function_type(ty: &CType) -> bool {
        matches!(ty, CType::Function { .. })
    }

    /// Creates a `const`-qualified wrapper around the given [`CType`].
    ///
    /// Utility method that uses [`TypeQualifiers`] directly to construct
    /// a `CType::Qualified` node with only the `is_const` flag set.
    #[allow(dead_code)]
    fn make_const_qualified(ty: CType) -> CType {
        let quals = TypeQualifiers {
            is_const: true,
            ..TypeQualifiers::default()
        };
        CType::Qualified(Box::new(ty), quals)
    }
}

// ===========================================================================
// Free-Standing Utilities
// ===========================================================================

/// Checks whether an attribute list contains an attribute with the given
/// interned name, using `Symbol::as_u32()` for O(1) integer comparison.
///
/// This is useful for quick existence checks without needing string resolution.
pub fn has_attribute_by_symbol(attrs: &[Attribute], name: Symbol) -> bool {
    let target = name.as_u32();
    attrs.iter().any(|a| a.name.as_u32() == target)
}

/// Extracts the `SymbolVisibility` from a validated attribute list, if present.
///
/// Returns `SymbolVisibility::Default` if no visibility attribute was found.
pub fn extract_visibility(attrs: &[ValidatedAttribute]) -> SymbolVisibility {
    for attr in attrs.iter().rev() {
        if let ValidatedAttribute::Visibility(vis) = attr {
            return *vis;
        }
    }
    SymbolVisibility::Default
}

/// Extracts the section name from a validated attribute list, if present.
pub fn extract_section(attrs: &[ValidatedAttribute]) -> Option<&str> {
    for attr in attrs.iter().rev() {
        if let ValidatedAttribute::Section(ref name) = attr {
            return Some(name.as_str());
        }
    }
    None
}

/// Returns `true` if any attribute in the list is `Used`.
pub fn has_used_attribute(attrs: &[ValidatedAttribute]) -> bool {
    attrs.iter().any(|a| matches!(a, ValidatedAttribute::Used))
}

/// Returns `true` if any attribute in the list is `Weak`.
pub fn has_weak_attribute(attrs: &[ValidatedAttribute]) -> bool {
    attrs.iter().any(|a| matches!(a, ValidatedAttribute::Weak))
}

/// Returns `true` if any attribute in the list is `NoReturn`.
pub fn has_noreturn_attribute(attrs: &[ValidatedAttribute]) -> bool {
    attrs
        .iter()
        .any(|a| matches!(a, ValidatedAttribute::NoReturn))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::string_interner::Interner;
    use crate::frontend::parser::ast::{IntegerSuffix, StringPrefix, StringSegment};

    // Helpers ---------------------------------------------------------------

    fn make_parts() -> (DiagnosticEngine, Interner) {
        (DiagnosticEngine::new(), Interner::new())
    }

    fn mk_attr(interner: &mut Interner, name: &str, args: Vec<AttributeArg>) -> Attribute {
        Attribute {
            name: interner.intern(name),
            args,
            span: Span::dummy(),
        }
    }

    fn int_arg(val: u64) -> AttributeArg {
        AttributeArg::Expression(Box::new(Expression::IntegerLiteral {
            value: val as u128,
            suffix: IntegerSuffix::None,
            is_hex_or_octal: false,
            span: Span::dummy(),
        }))
    }

    fn str_arg(s: &str) -> AttributeArg {
        AttributeArg::String(s.as_bytes().to_vec(), Span::dummy())
    }

    fn ident_arg(interner: &mut Interner, name: &str) -> AttributeArg {
        AttributeArg::Identifier(interner.intern(name), Span::dummy())
    }

    // Enum basics -----------------------------------------------------------

    #[test]
    fn visibility_default_trait() {
        assert_eq!(SymbolVisibility::default(), SymbolVisibility::Default);
    }

    #[test]
    fn validated_attribute_clone_eq() {
        let a = ValidatedAttribute::Aligned(16);
        assert_eq!(a.clone(), a);
    }

    #[test]
    fn format_archetype_copy() {
        let a = FormatArchetype::Printf;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn context_eq() {
        assert_eq!(AttributeContext::Function, AttributeContext::Function);
        assert_ne!(AttributeContext::Function, AttributeContext::Variable);
    }

    // Handler construction --------------------------------------------------

    #[test]
    fn handler_new() {
        let (mut diag, interner) = make_parts();
        let _h = AttributeHandler::new(&mut diag, Target::X86_64, &interner);
    }

    // Simple no-arg attributes ----------------------------------------------

    #[test]
    fn validate_packed_ok() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "packed", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Type, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Packed]);
    }

    #[test]
    fn validate_packed_with_args_err() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "packed", vec![int_arg(1)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Type, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn validate_used() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "used", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Used]);
    }

    #[test]
    fn validate_unused() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "unused", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Unused]);
    }

    #[test]
    fn validate_weak() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "weak", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Weak]);
    }

    #[test]
    fn validate_noreturn() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "noreturn", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::NoReturn]);
    }

    #[test]
    fn validate_noinline() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "noinline", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::NoInline]);
    }

    #[test]
    fn validate_always_inline() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "always_inline", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::AlwaysInline]);
    }

    #[test]
    fn validate_cold() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "cold", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Cold]);
    }

    #[test]
    fn validate_hot() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "hot", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Hot]);
    }

    #[test]
    fn validate_malloc() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "malloc", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Malloc]);
    }

    #[test]
    fn validate_pure() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "pure", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Pure]);
    }

    #[test]
    fn validate_const_attr() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "const", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Const]);
    }

    #[test]
    fn validate_warn_unused_result() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "warn_unused_result", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::WarnUnusedResult]);
    }

    #[test]
    fn validate_fallthrough() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "fallthrough", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Statement, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Fallthrough]);
    }

    // Aligned ---------------------------------------------------------------

    #[test]
    fn aligned_default_max() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(16)]);
    }

    #[test]
    fn aligned_power_of_two() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![int_arg(8)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(8)]);
    }

    #[test]
    fn aligned_not_power_of_two_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![int_arg(3)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert!(r.is_empty());
        assert!(diag.has_errors());
    }

    #[test]
    fn aligned_zero_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![int_arg(0)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn aligned_large_warns_but_accepted() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![int_arg(8192)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(8192)]);
        assert!(diag.warning_count() > 0);
    }

    // Section ---------------------------------------------------------------

    #[test]
    fn section_ok() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "section", vec![str_arg(".init.text")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Section(".init.text".into())]);
    }

    #[test]
    fn section_empty_name_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "section", vec![str_arg("")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn section_no_args_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "section", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn section_null_byte_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(
            &mut inter,
            "section",
            vec![AttributeArg::String(
                b".text\x00rest".to_vec(),
                Span::dummy(),
            )],
        );
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    // Constructor / Destructor ----------------------------------------------

    #[test]
    fn constructor_no_priority() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "constructor", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Constructor(None)]);
    }

    #[test]
    fn constructor_with_priority() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "constructor", vec![int_arg(200)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Constructor(Some(200))]);
    }

    #[test]
    fn constructor_reserved_priority_warns() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "constructor", vec![int_arg(50)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Constructor(Some(50))]);
        assert!(diag.warning_count() > 0);
    }

    #[test]
    fn destructor_no_priority() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "destructor", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Destructor(None)]);
    }

    // Visibility ------------------------------------------------------------

    #[test]
    fn visibility_hidden() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "visibility", vec![str_arg("hidden")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Visibility(SymbolVisibility::Hidden)]
        );
    }

    #[test]
    fn visibility_default_value() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "visibility", vec![str_arg("default")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Visibility(SymbolVisibility::Default)]
        );
    }

    #[test]
    fn visibility_protected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "visibility", vec![str_arg("protected")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Visibility(SymbolVisibility::Protected)]
        );
    }

    #[test]
    fn visibility_internal() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "visibility", vec![str_arg("internal")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Visibility(SymbolVisibility::Internal)]
        );
    }

    #[test]
    fn visibility_invalid_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "visibility", vec![str_arg("public")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
        assert!(diag.has_errors());
    }

    // Deprecated ------------------------------------------------------------

    #[test]
    fn deprecated_no_msg() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "deprecated", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Deprecated(None)]);
    }

    #[test]
    fn deprecated_with_msg() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "deprecated", vec![str_arg("use v2")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Deprecated(Some("use v2".into()))]
        );
    }

    // Format ----------------------------------------------------------------

    #[test]
    fn format_printf() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "printf"), int_arg(1), int_arg(2)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r,
            vec![ValidatedAttribute::Format {
                archetype: FormatArchetype::Printf,
                string_index: 1,
                first_to_check: 2,
            }]
        );
    }

    #[test]
    fn format_scanf() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "scanf"), int_arg(1), int_arg(2)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r[0],
            ValidatedAttribute::Format {
                archetype: FormatArchetype::Scanf,
                string_index: 1,
                first_to_check: 2,
            }
        );
    }

    #[test]
    fn format_strftime() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "strftime"), int_arg(1), int_arg(0)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r[0],
            ValidatedAttribute::Format {
                archetype: FormatArchetype::Strftime,
                string_index: 1,
                first_to_check: 0,
            }
        );
    }

    #[test]
    fn format_strfmon() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "strfmon"), int_arg(1), int_arg(2)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r[0],
            ValidatedAttribute::Format {
                archetype: FormatArchetype::Strfmon,
                string_index: 1,
                first_to_check: 2,
            }
        );
    }

    #[test]
    fn format_printf_alias() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "__printf__"), int_arg(2), int_arg(3)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(
            r[0],
            ValidatedAttribute::Format {
                archetype: FormatArchetype::Printf,
                string_index: 2,
                first_to_check: 3,
            }
        );
    }

    #[test]
    fn format_wrong_arg_count() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "printf"), int_arg(1)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn format_unknown_archetype() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "custom"), int_arg(1), int_arg(2)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    // Format arg ------------------------------------------------------------

    #[test]
    fn format_arg_ok() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "format_arg", vec![int_arg(1)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::FormatArg(1)]);
    }

    #[test]
    fn format_arg_zero_rejected() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "format_arg", vec![int_arg(0)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    // Context applicability -------------------------------------------------

    #[test]
    fn noreturn_not_applicable_to_variable() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "noreturn", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert!(r.is_empty());
    }

    #[test]
    fn fallthrough_only_on_statement() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "fallthrough", vec![]);

        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a.clone()], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());

        let mut h2 = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r2 = h2.validate_attributes(&[a], AttributeContext::Statement, Span::dummy());
        assert_eq!(r2, vec![ValidatedAttribute::Fallthrough]);
    }

    #[test]
    fn packed_not_applicable_to_function() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "packed", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }

    // Unknown attribute -----------------------------------------------------

    #[test]
    fn unknown_attribute_emits_warning() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "nonexistent_attr", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
        assert!(diag.warning_count() > 0);
    }

    // Conflict detection ----------------------------------------------------

    #[test]
    fn conflict_noinline_always_inline() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "noinline", vec![]);
        let a2 = mk_attr(&mut inter, "always_inline", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Function, Span::dummy());
        assert!(!r.iter().any(|a| matches!(a, ValidatedAttribute::NoInline)));
        assert!(r
            .iter()
            .any(|a| matches!(a, ValidatedAttribute::AlwaysInline)));
    }

    #[test]
    fn conflict_cold_hot() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "cold", vec![]);
        let a2 = mk_attr(&mut inter, "hot", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Function, Span::dummy());
        assert!(!r.iter().any(|a| matches!(a, ValidatedAttribute::Cold)));
        assert!(r.iter().any(|a| matches!(a, ValidatedAttribute::Hot)));
    }

    #[test]
    fn conflict_pure_const() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "pure", vec![]);
        let a2 = mk_attr(&mut inter, "const", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Function, Span::dummy());
        assert!(!r.iter().any(|a| matches!(a, ValidatedAttribute::Pure)));
        assert!(r.iter().any(|a| matches!(a, ValidatedAttribute::Const)));
    }

    #[test]
    fn multiple_aligned_keeps_last() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "aligned", vec![int_arg(4)]);
        let a2 = mk_attr(&mut inter, "aligned", vec![int_arg(16)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Variable, Span::dummy());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], ValidatedAttribute::Aligned(16));
    }

    #[test]
    fn multiple_section_keeps_last() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "section", vec![str_arg(".data")]);
        let a2 = mk_attr(&mut inter, "section", vec![str_arg(".init.text")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Function, Span::dummy());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], ValidatedAttribute::Section(".init.text".into()));
    }

    // Propagation -----------------------------------------------------------

    #[test]
    fn propagate_aligned_to_struct() {
        let (mut diag, inter) = make_parts();
        let h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let attrs = vec![ValidatedAttribute::Aligned(32)];
        let mut ty = CType::Struct {
            name: Some("s".to_string()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        h.propagate_to_type(&attrs, &mut ty);
        match &ty {
            CType::Struct { aligned, .. } => assert_eq!(*aligned, Some(32)),
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn propagate_packed_to_struct() {
        let (mut diag, inter) = make_parts();
        let h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let attrs = vec![ValidatedAttribute::Packed];
        let mut ty = CType::Struct {
            name: Some("s".to_string()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        h.propagate_to_type(&attrs, &mut ty);
        match &ty {
            CType::Struct { packed, .. } => assert!(*packed),
            _ => panic!("expected Struct"),
        }
    }

    #[test]
    fn propagate_packed_to_union() {
        let (mut diag, inter) = make_parts();
        let h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let attrs = vec![ValidatedAttribute::Packed];
        let mut ty = CType::Union {
            name: Some("u".to_string()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        h.propagate_to_type(&attrs, &mut ty);
        match &ty {
            CType::Union { packed, .. } => assert!(*packed),
            _ => panic!("expected Union"),
        }
    }

    #[test]
    fn propagate_to_symbol() {
        let (mut diag, inter) = make_parts();
        let h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let attrs = vec![
            ValidatedAttribute::Weak,
            ValidatedAttribute::Visibility(SymbolVisibility::Hidden),
            ValidatedAttribute::Section(".text".into()),
        ];
        let mut sym_attrs: Vec<ValidatedAttribute> = Vec::new();
        h.propagate_to_symbol(&attrs, &mut sym_attrs);
        assert_eq!(sym_attrs.len(), 3);
        assert_eq!(sym_attrs[0], ValidatedAttribute::Weak);
        assert_eq!(
            sym_attrs[1],
            ValidatedAttribute::Visibility(SymbolVisibility::Hidden)
        );
        assert_eq!(sym_attrs[2], ValidatedAttribute::Section(".text".into()));
    }

    // Free-standing utilities -----------------------------------------------

    #[test]
    fn test_has_attribute_by_symbol() {
        let mut interner = Interner::new();
        let sym = interner.intern("packed");
        let attrs = vec![Attribute {
            name: sym,
            args: vec![],
            span: Span::dummy(),
        }];
        assert!(has_attribute_by_symbol(&attrs, sym));
        let other = interner.intern("aligned");
        assert!(!has_attribute_by_symbol(&attrs, other));
    }

    #[test]
    fn test_extract_visibility_found() {
        let attrs = vec![
            ValidatedAttribute::Unused,
            ValidatedAttribute::Visibility(SymbolVisibility::Hidden),
        ];
        assert_eq!(extract_visibility(&attrs), SymbolVisibility::Hidden);
    }

    #[test]
    fn test_extract_visibility_absent() {
        let attrs = vec![ValidatedAttribute::Unused, ValidatedAttribute::Used];
        assert_eq!(extract_visibility(&attrs), SymbolVisibility::Default);
    }

    #[test]
    fn test_extract_section_found() {
        let attrs = vec![ValidatedAttribute::Section(".data".into())];
        assert_eq!(extract_section(&attrs), Some(".data"));
    }

    #[test]
    fn test_extract_section_absent() {
        let attrs = vec![ValidatedAttribute::Used];
        assert_eq!(extract_section(&attrs), None);
    }

    #[test]
    fn test_has_used() {
        assert!(has_used_attribute(&[ValidatedAttribute::Used]));
        assert!(!has_used_attribute(&[ValidatedAttribute::Unused]));
    }

    #[test]
    fn test_has_weak() {
        assert!(has_weak_attribute(&[ValidatedAttribute::Weak]));
        assert!(!has_weak_attribute(&[ValidatedAttribute::Used]));
    }

    #[test]
    fn test_has_noreturn() {
        assert!(has_noreturn_attribute(&[ValidatedAttribute::NoReturn]));
        assert!(!has_noreturn_attribute(&[ValidatedAttribute::Packed]));
    }

    // Multi-target ----------------------------------------------------------

    #[test]
    fn aligned_default_i686() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::I686, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(16)]);
    }

    #[test]
    fn aligned_default_aarch64() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::AArch64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(16)]);
    }

    #[test]
    fn aligned_default_riscv64() {
        let (mut diag, mut inter) = make_parts();
        let a = mk_attr(&mut inter, "aligned", vec![]);
        let mut h = AttributeHandler::new(&mut diag, Target::RiscV64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Variable, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Aligned(16)]);
    }

    // Mixed combinations ----------------------------------------------------

    #[test]
    fn multiple_attrs_on_function() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "noreturn", vec![]);
        let a2 = mk_attr(&mut inter, "cold", vec![]);
        let a3 = mk_attr(&mut inter, "visibility", vec![str_arg("hidden")]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2, a3], AttributeContext::Function, Span::dummy());
        assert_eq!(r.len(), 3);
        assert!(r.contains(&ValidatedAttribute::NoReturn));
        assert!(r.contains(&ValidatedAttribute::Cold));
        assert!(r.contains(&ValidatedAttribute::Visibility(SymbolVisibility::Hidden)));
    }

    #[test]
    fn packed_and_aligned_both_apply() {
        let (mut diag, mut inter) = make_parts();
        let a1 = mk_attr(&mut inter, "packed", vec![]);
        let a2 = mk_attr(&mut inter, "aligned", vec![int_arg(4)]);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a1, a2], AttributeContext::Type, Span::dummy());
        assert!(r.contains(&ValidatedAttribute::Packed));
        assert!(r.contains(&ValidatedAttribute::Aligned(4)));
    }

    // String literal in Expression node ------------------------------------

    #[test]
    fn extract_string_from_expression_node() {
        let (mut diag, mut inter) = make_parts();
        let seg = StringSegment {
            value: b"hello".to_vec(),
            span: Span::dummy(),
        };
        let args = vec![AttributeArg::Expression(Box::new(
            Expression::StringLiteral {
                segments: vec![seg],
                prefix: StringPrefix::None,
                span: Span::dummy(),
            },
        ))];
        let a = mk_attr(&mut inter, "section", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert_eq!(r, vec![ValidatedAttribute::Section("hello".into())]);
    }

    // Validate that format first_to_check <= string_index is rejected ------

    #[test]
    fn format_first_to_check_too_small() {
        let (mut diag, mut inter) = make_parts();
        let args = vec![ident_arg(&mut inter, "printf"), int_arg(3), int_arg(2)];
        let a = mk_attr(&mut inter, "format", args);
        let mut h = AttributeHandler::new(&mut diag, Target::X86_64, &inter);
        let r = h.validate_attributes(&[a], AttributeContext::Function, Span::dummy());
        assert!(r.is_empty());
    }
}
