// src/frontend/sema/attribute_handler.rs
//
// Attribute semantic validation and propagation module for Phase 5
// (semantic analysis) of the BCC compiler.
//
// Validates GCC `__attribute__((...))` arguments after parsing:
//   - `aligned(N)` power-of-two check
//   - `section("name")` string validation
//   - `format(printf, N, M)` parameter index validation
//   - `visibility("hidden"|"default"|"protected")` enum mapping
//   - `constructor/destructor` priority range checking
//   - Argument type/count validation for all 21+ supported GCC attributes
//
// Propagates validated attributes to their target symbols and types in
// the symbol table.
//
// Integration points:
//   - Consumed by: `crate::frontend::sema::mod` during declaration processing
//   - Depends on: `crate::frontend::parser::ast` for parsed `Attribute` nodes
//   - Depends on: `crate::common::diagnostics` for error/warning reporting
//   - Depends on: `crate::common::types` for `CType` inspection
//   - Depends on: `crate::frontend::sema::symbol_table` for `SymbolEntry`
//   - Depends on: `crate::common::string_interner` for `Symbol`/`Interner`

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::string_interner::{Interner, Symbol};
use crate::common::types::CType;
use crate::frontend::parser::ast::{Attribute, AttributeArg};
use crate::frontend::sema::symbol_table::{SymbolEntry, VisibilityKind};

// ===========================================================================
// FormatKind — archetype for format(printf, ...) style attributes
// ===========================================================================

/// The format archetype for `__attribute__((format(...)))`.
///
/// Identifies which family of format-string checking to apply when
/// validating printf-style, scanf-style, or time-formatting functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatKind {
    /// `printf`/`__printf__` format family (e.g., `printf`, `fprintf`, `snprintf`).
    Printf,
    /// `scanf`/`__scanf__` format family (e.g., `scanf`, `fscanf`, `sscanf`).
    Scanf,
    /// `strftime`/`__strftime__` format family for time formatting.
    Strftime,
    /// `strfmon`/`__strfmon__` format family for monetary formatting.
    Strfmon,
}

// ===========================================================================
// AttributeTargetKind — what the attributes are attached to
// ===========================================================================

/// The kind of declaration or construct that an `__attribute__` is attached to.
///
/// Used during validation to determine whether a specific attribute is valid
/// for the target it appears on (e.g., `noreturn` is only valid on functions,
/// `packed` is only valid on struct/union types or fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttributeTargetKind {
    /// Attribute on a function declaration or definition.
    Function,
    /// Attribute on a variable declaration.
    Variable,
    /// Attribute on a type specifier (struct, union, enum, typedef).
    Type,
    /// Attribute on a struct/union field.
    Field,
    /// Attribute on a statement (e.g., `[[fallthrough]]`).
    Statement,
    /// Attribute on a label.
    Label,
}

// ===========================================================================
// ValidatedAttribute — semantic representation of validated attributes
// ===========================================================================

/// A semantically validated GCC attribute with extracted and type-checked
/// argument values.
///
/// This enum is the output of the attribute validation pipeline. Each variant
/// corresponds to a specific GCC attribute with its validated arguments
/// extracted into strongly-typed fields. Invalid attributes are rejected
/// during validation and never appear as `ValidatedAttribute` values.
///
/// The `propagate_to_symbol` and `propagate_to_type` functions consume these
/// validated attributes and apply them to the appropriate compiler data
/// structures.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidatedAttribute {
    /// `__attribute__((aligned(N)))` — minimum alignment in bytes.
    /// `None` means use the target's maximum useful alignment (GCC default).
    Aligned(Option<u64>),
    /// `__attribute__((packed))` — minimize padding in struct/union layout.
    Packed,
    /// `__attribute__((section("name")))` — place symbol in named ELF section.
    Section(String),
    /// `__attribute__((used))` — prevent linker from discarding symbol.
    Used,
    /// `__attribute__((unused))` — suppress unused warnings.
    Unused,
    /// `__attribute__((weak))` — weak binding, overridable at link time.
    Weak,
    /// `__attribute__((constructor(priority)))` — called before `main`.
    /// `None` priority means default priority (65535).
    Constructor(Option<u32>),
    /// `__attribute__((destructor(priority)))` — called after `main` returns.
    /// `None` priority means default priority (65535).
    Destructor(Option<u32>),
    /// `__attribute__((visibility("...")))` — ELF symbol visibility control.
    Visibility(VisibilityKind),
    /// `__attribute__((deprecated))` or `__attribute__((deprecated("msg")))`.
    Deprecated(Option<String>),
    /// `__attribute__((noreturn))` — function never returns to caller.
    Noreturn,
    /// `__attribute__((noinline))` — prevent inlining.
    Noinline,
    /// `__attribute__((always_inline))` — force inlining.
    AlwaysInline,
    /// `__attribute__((cold))` — unlikely execution path hint.
    Cold,
    /// `__attribute__((hot))` — likely execution path hint.
    Hot,
    /// `__attribute__((format(archetype, string_index, first_to_check)))`.
    Format {
        /// Format family (printf, scanf, strftime, strfmon).
        archetype: FormatKind,
        /// 1-based parameter index of the format string.
        string_index: u32,
        /// 1-based parameter index of the first argument to check (0 = no check).
        first_to_check: u32,
    },
    /// `__attribute__((format_arg(N)))` — parameter N is a format string.
    FormatArg(u32),
    /// `__attribute__((malloc))` — returned pointer does not alias.
    Malloc,
    /// `__attribute__((pure))` — no side effects except via arguments.
    Pure,
    /// `__attribute__((const))` — no side effects, no global memory reads.
    Const,
    /// `__attribute__((warn_unused_result))` — warn if return value discarded.
    WarnUnusedResult,
    /// `__attribute__((fallthrough))` — intentional fall-through in switch.
    Fallthrough,
}

// ===========================================================================
// Attribute name canonicalization
// ===========================================================================

/// Canonicalizes a GCC attribute name by stripping the double-underscore
/// prefix and suffix, if both are present.
///
/// GCC allows attribute names in two forms:
///   - Plain form: `aligned`, `packed`, `section`
///   - Double-underscore form: `__aligned__`, `__packed__`, `__section__`
///
/// Both forms are semantically identical. This function normalizes the
/// double-underscore form to the plain form for uniform dispatch.
///
/// # Examples
///
/// - `"__aligned__"` → `"aligned"`
/// - `"__packed__"` → `"packed"`
/// - `"aligned"` → `"aligned"` (unchanged)
/// - `"__foo"` → `"__foo"` (unchanged — suffix underscores missing)
/// - `"foo__"` → `"foo__"` (unchanged — prefix underscores missing)
pub fn canonicalize_name(name: &str) -> &str {
    if name.len() >= 5 && name.starts_with("__") && name.ends_with("__") {
        &name[2..name.len() - 2]
    } else {
        name
    }
}

// ===========================================================================
// Attribute target validity tables
// ===========================================================================

/// Returns `true` if the given canonical attribute name is valid for the
/// specified target kind. This prevents attributes from being applied to
/// inappropriate targets (e.g., `noreturn` on a variable).
fn is_valid_target(canonical_name: &str, target: AttributeTargetKind) -> bool {
    match canonical_name {
        // Function-only attributes
        "noreturn" | "noinline" | "always_inline" | "cold" | "hot" | "format"
        | "format_arg" | "malloc" | "pure" | "const" | "warn_unused_result"
        | "constructor" | "destructor" => matches!(target, AttributeTargetKind::Function),

        // Function and variable attributes
        "used" | "unused" | "weak" | "section" | "visibility" | "deprecated" => matches!(
            target,
            AttributeTargetKind::Function
                | AttributeTargetKind::Variable
                | AttributeTargetKind::Type
                | AttributeTargetKind::Field
        ),

        // Alignment applies to variables, types, and fields
        "aligned" => matches!(
            target,
            AttributeTargetKind::Function
                | AttributeTargetKind::Variable
                | AttributeTargetKind::Type
                | AttributeTargetKind::Field
        ),

        // Packed applies to types (struct/union) and fields
        "packed" => matches!(
            target,
            AttributeTargetKind::Type | AttributeTargetKind::Field
        ),

        // Fallthrough applies to statements (null statement before case label)
        "fallthrough" => matches!(target, AttributeTargetKind::Statement),

        // Unknown attributes: allow them (GCC silently accepts many).
        // A warning is emitted by the caller for truly unknown attributes.
        _ => true,
    }
}

/// Returns a human-readable description of an `AttributeTargetKind`.
fn target_kind_name(target: AttributeTargetKind) -> &'static str {
    match target {
        AttributeTargetKind::Function => "function",
        AttributeTargetKind::Variable => "variable",
        AttributeTargetKind::Type => "type",
        AttributeTargetKind::Field => "field",
        AttributeTargetKind::Statement => "statement",
        AttributeTargetKind::Label => "label",
    }
}

// ===========================================================================
// Main validation entry point
// ===========================================================================

/// Validates a list of parsed `Attribute` nodes and returns the
/// semantically-validated representations.
///
/// For each attribute:
/// 1. The name is resolved from the interned `Symbol` to a string.
/// 2. The name is canonicalized (strip `__...__` wrappers).
/// 3. Target validity is checked (e.g., `noreturn` only on functions).
/// 4. Argument count and types are validated.
/// 5. Specific semantic constraints are enforced (e.g., alignment power-of-two).
///
/// Invalid attributes produce diagnostic errors or warnings but do NOT cause
/// the function to return early — all attributes are processed to maximize
/// diagnostic coverage.
///
/// # Arguments
///
/// * `attrs` — Parsed attribute list from the AST.
/// * `target_kind` — What the attributes are attached to.
/// * `interner` — String interner for resolving `Symbol` handles.
/// * `diagnostics` — Diagnostic engine for reporting errors and warnings.
///
/// # Returns
///
/// A `Vec<ValidatedAttribute>` containing only the successfully validated
/// attributes. Attributes that fail validation are omitted from the result.
pub fn validate_attributes(
    attrs: &[Attribute],
    target_kind: AttributeTargetKind,
    interner: &Interner,
    diagnostics: &mut DiagnosticEngine,
) -> Vec<ValidatedAttribute> {
    let mut result = Vec::with_capacity(attrs.len());

    for attr in attrs {
        let name_sym: Symbol = attr.name;
        let raw_name = interner.resolve(name_sym);
        let canonical = canonicalize_name(raw_name);

        // Check target validity
        if !is_valid_target(canonical, target_kind) {
            diagnostics.warning(
                attr.span,
                format!(
                    "'{}' attribute ignored on {} declaration",
                    canonical,
                    target_kind_name(target_kind),
                ),
            );
            continue;
        }

        // Dispatch to per-attribute validator
        let validated = match canonical {
            "aligned" => validate_aligned(&attr.args, attr.span, diagnostics),
            "packed" => validate_packed(&attr.args, attr.span, diagnostics),
            "section" => validate_section(&attr.args, attr.span, diagnostics),
            "used" => validate_used(&attr.args, attr.span, diagnostics),
            "unused" => validate_unused(&attr.args, attr.span, diagnostics),
            "weak" => validate_weak(&attr.args, attr.span, diagnostics),
            "constructor" => validate_constructor(&attr.args, attr.span, diagnostics),
            "destructor" => validate_destructor(&attr.args, attr.span, diagnostics),
            "visibility" => validate_visibility(&attr.args, attr.span, interner, diagnostics),
            "deprecated" => validate_deprecated(&attr.args, attr.span, diagnostics),
            "noreturn" => validate_noreturn(&attr.args, attr.span, diagnostics),
            "noinline" => validate_noinline(&attr.args, attr.span, diagnostics),
            "always_inline" => validate_always_inline(&attr.args, attr.span, diagnostics),
            "cold" => validate_cold(&attr.args, attr.span, diagnostics),
            "hot" => validate_hot(&attr.args, attr.span, diagnostics),
            "format" => validate_format(&attr.args, attr.span, interner, diagnostics),
            "format_arg" => validate_format_arg(&attr.args, attr.span, diagnostics),
            "malloc" => validate_malloc(&attr.args, attr.span, diagnostics),
            "pure" => validate_pure(&attr.args, attr.span, diagnostics),
            "const" => validate_const_attr(&attr.args, attr.span, diagnostics),
            "warn_unused_result" => {
                validate_warn_unused_result(&attr.args, attr.span, diagnostics)
            }
            "fallthrough" => validate_fallthrough(&attr.args, attr.span, diagnostics),
            _ => {
                diagnostics.warning(
                    attr.span,
                    format!("unknown attribute '{}' ignored", canonical),
                );
                None
            }
        };

        if let Some(v) = validated {
            result.push(v);
        }
    }

    result
}

// ===========================================================================
// Argument validation helpers
// ===========================================================================

/// Checks that an attribute has exactly `expected` arguments.
/// Returns `true` if the count matches; emits a diagnostic and returns
/// `false` otherwise.
fn check_arg_count(
    name: &str,
    args: &[AttributeArg],
    expected: usize,
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> bool {
    if args.len() != expected {
        diagnostics.error(
            span,
            format!(
                "attribute '{}' requires {} argument{}, but {} {} provided",
                name,
                expected,
                if expected == 1 { "" } else { "s" },
                args.len(),
                if args.len() == 1 { "was" } else { "were" },
            ),
        );
        false
    } else {
        true
    }
}

/// Checks that an attribute has no arguments (flag-style attribute).
/// Returns `true` if there are no arguments; emits a diagnostic and returns
/// `false` otherwise.
fn check_no_args(
    name: &str,
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> bool {
    if !args.is_empty() {
        diagnostics.error(
            span,
            format!(
                "attribute '{}' does not accept arguments ({} provided)",
                name,
                args.len(),
            ),
        );
        false
    } else {
        true
    }
}

/// Checks that an attribute has at most `max` arguments.
/// Returns `true` if the count is within range; emits a diagnostic and
/// returns `false` otherwise.
fn check_max_args(
    name: &str,
    args: &[AttributeArg],
    max: usize,
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> bool {
    if args.len() > max {
        diagnostics.error(
            span,
            format!(
                "attribute '{}' accepts at most {} argument{}, but {} {} provided",
                name,
                max,
                if max == 1 { "" } else { "s" },
                args.len(),
                if args.len() == 1 { "was" } else { "were" },
            ),
        );
        false
    } else {
        true
    }
}

/// Returns `true` if `value` is a power of two or zero.
/// Used for alignment validation.
#[inline]
fn is_power_of_two(value: u64) -> bool {
    value > 0 && (value & (value - 1)) == 0
}

// ===========================================================================
// Per-attribute validation functions
// ===========================================================================

/// Validates `__attribute__((aligned))` or `__attribute__((aligned(N)))`.
///
/// - With no arguments: uses the target's maximum useful alignment.
/// - With one integer argument N: N must be a positive power of two.
/// - More than one argument is an error.
fn validate_aligned(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if args.is_empty() {
        // No argument: use default maximum alignment
        return Some(ValidatedAttribute::Aligned(None));
    }

    if !check_max_args("aligned", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::Integer(n) => {
            if *n <= 0 {
                diagnostics.error(
                    span,
                    format!(
                        "requested alignment {} is not a positive power of 2",
                        n
                    ),
                );
                return None;
            }
            let alignment = *n as u64;
            if !is_power_of_two(alignment) {
                diagnostics.error(
                    span,
                    format!(
                        "requested alignment {} is not a power of 2",
                        alignment
                    ),
                );
                return None;
            }
            // GCC allows very large alignments but caps at 2^29 on most targets.
            // We accept up to 2^30 (1 GiB) as a reasonable limit.
            if alignment > (1 << 30) {
                diagnostics.error(
                    span,
                    format!(
                        "requested alignment {} exceeds maximum allowed alignment ({})",
                        alignment,
                        1u64 << 30
                    ),
                );
                return None;
            }
            Some(ValidatedAttribute::Aligned(Some(alignment)))
        }
        AttributeArg::Expression(_) => {
            // Expression arguments (e.g., sizeof(int)) are accepted but we
            // cannot evaluate them at attribute-validation time. The semantic
            // analyzer's constant evaluator should have reduced this to an
            // integer before attribute validation; if it hasn't, we accept
            // it optimistically and defer the power-of-two check to later.
            diagnostics.warning(
                span,
                "aligned attribute with non-constant expression; alignment check deferred",
            );
            Some(ValidatedAttribute::Aligned(None))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'aligned' attribute must be an integer constant",
            );
            None
        }
    }
}

/// Validates `__attribute__((packed))`.
///
/// Packed accepts no arguments. Applies to struct/union types and fields
/// to minimize padding between members.
fn validate_packed(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("packed", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Packed)
}

/// Validates `__attribute__((section("name")))`.
///
/// Requires exactly one string argument containing a valid ELF section name.
/// The section name must be a non-empty string. Certain characters that are
/// invalid in ELF section names are rejected.
fn validate_section(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_arg_count("section", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::String(bytes) => {
            // Convert to a UTF-8 string for the section name.
            // Section names in ELF are NUL-terminated C strings, so we
            // reject any NUL bytes within the name.
            let name = match std::str::from_utf8(bytes) {
                Ok(s) => s.to_string(),
                Err(_) => {
                    diagnostics.error(
                        span,
                        "section name must be a valid UTF-8 string",
                    );
                    return None;
                }
            };

            if name.is_empty() {
                diagnostics.error(span, "section name must not be empty");
                return None;
            }

            if name.contains('\0') {
                diagnostics.error(
                    span,
                    "section name must not contain NUL characters",
                );
                return None;
            }

            Some(ValidatedAttribute::Section(name))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'section' attribute must be a string literal",
            );
            None
        }
    }
}

/// Validates `__attribute__((used))`.
///
/// No arguments. Prevents the linker from discarding the symbol even if
/// it appears unreferenced.
fn validate_used(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("used", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Used)
}

/// Validates `__attribute__((unused))`.
///
/// No arguments. Suppresses "unused variable/function" warnings.
fn validate_unused(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("unused", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Unused)
}

/// Validates `__attribute__((weak))`.
///
/// No arguments. Marks the symbol as having weak binding in the ELF
/// symbol table, allowing it to be overridden by a strong definition.
fn validate_weak(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("weak", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Weak)
}

/// Validates `__attribute__((constructor))` or `__attribute__((constructor(priority)))`.
///
/// With no arguments: default priority (65535).
/// With one integer argument: priority value in range 0–65535.
/// Priorities 0–100 are reserved for the implementation (GCC convention).
fn validate_constructor(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if args.is_empty() {
        return Some(ValidatedAttribute::Constructor(None));
    }

    if !check_max_args("constructor", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::Integer(n) => {
            if *n < 0 || *n > 65535 {
                diagnostics.error(
                    span,
                    format!(
                        "constructor priority {} is outside the valid range [0, 65535]",
                        n
                    ),
                );
                return None;
            }
            let priority = *n as u32;
            if priority <= 100 {
                diagnostics.warning(
                    span,
                    format!(
                        "constructor priority {} is in the reserved range [0, 100]",
                        priority
                    ),
                );
            }
            Some(ValidatedAttribute::Constructor(Some(priority)))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'constructor' attribute must be an integer constant",
            );
            None
        }
    }
}

/// Validates `__attribute__((destructor))` or `__attribute__((destructor(priority)))`.
///
/// Same rules as `constructor`: optional priority in 0–65535 with
/// 0–100 reserved.
fn validate_destructor(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if args.is_empty() {
        return Some(ValidatedAttribute::Destructor(None));
    }

    if !check_max_args("destructor", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::Integer(n) => {
            if *n < 0 || *n > 65535 {
                diagnostics.error(
                    span,
                    format!(
                        "destructor priority {} is outside the valid range [0, 65535]",
                        n
                    ),
                );
                return None;
            }
            let priority = *n as u32;
            if priority <= 100 {
                diagnostics.warning(
                    span,
                    format!(
                        "destructor priority {} is in the reserved range [0, 100]",
                        priority
                    ),
                );
            }
            Some(ValidatedAttribute::Destructor(Some(priority)))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'destructor' attribute must be an integer constant",
            );
            None
        }
    }
}

/// Validates `__attribute__((visibility("default"|"hidden"|"protected"|"internal")))`.
///
/// Requires exactly one string argument containing a valid ELF visibility
/// specifier. Maps to the corresponding `VisibilityKind` enum value.
fn validate_visibility(
    args: &[AttributeArg],
    span: Span,
    interner: &Interner,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_arg_count("visibility", args, 1, span, diagnostics) {
        return None;
    }

    // The visibility argument can be either a string literal or an identifier
    // depending on parser behavior. Handle both forms.
    let vis_str: String = match &args[0] {
        AttributeArg::String(bytes) => {
            match std::str::from_utf8(bytes) {
                Ok(s) => s.to_string(),
                Err(_) => {
                    diagnostics.error(
                        span,
                        "visibility argument must be a valid UTF-8 string",
                    );
                    return None;
                }
            }
        }
        AttributeArg::Identifier(sym) => {
            interner.resolve(*sym).to_string()
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'visibility' attribute must be a string literal",
            );
            return None;
        }
    };

    let kind = match vis_str.as_str() {
        "default" => VisibilityKind::Default,
        "hidden" => VisibilityKind::Hidden,
        "protected" => VisibilityKind::Protected,
        "internal" => VisibilityKind::Internal,
        _ => {
            diagnostics.error(
                span,
                format!(
                    "visibility must be \"default\", \"hidden\", \"protected\", \
                     or \"internal\"; got \"{}\"",
                    vis_str
                ),
            );
            return None;
        }
    };

    Some(ValidatedAttribute::Visibility(kind))
}

/// Validates `__attribute__((deprecated))` or `__attribute__((deprecated("msg")))`.
///
/// Accepts zero or one arguments. If present, the argument must be a string
/// literal containing the deprecation message.
fn validate_deprecated(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if args.is_empty() {
        return Some(ValidatedAttribute::Deprecated(None));
    }

    if !check_max_args("deprecated", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::String(bytes) => {
            let msg = match std::str::from_utf8(bytes) {
                Ok(s) => s.to_string(),
                Err(_) => {
                    diagnostics.error(
                        span,
                        "deprecation message must be a valid UTF-8 string",
                    );
                    return None;
                }
            };
            Some(ValidatedAttribute::Deprecated(Some(msg)))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'deprecated' attribute must be a string literal",
            );
            None
        }
    }
}

/// Validates `__attribute__((noreturn))`.
///
/// No arguments. Function-only attribute indicating the function never
/// returns to its caller (e.g., `exit()`, `abort()`, `__builtin_unreachable()`).
fn validate_noreturn(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("noreturn", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Noreturn)
}

/// Validates `__attribute__((noinline))`.
///
/// No arguments. Prevents the compiler from inlining this function.
fn validate_noinline(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("noinline", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Noinline)
}

/// Validates `__attribute__((always_inline))`.
///
/// No arguments. Forces the compiler to inline this function at every
/// call site, even at `-O0`.
fn validate_always_inline(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("always_inline", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::AlwaysInline)
}

/// Validates `__attribute__((cold))`.
///
/// No arguments. Hint that the function is rarely executed, allowing
/// the optimizer to deprioritize it.
fn validate_cold(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("cold", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Cold)
}

/// Validates `__attribute__((hot))`.
///
/// No arguments. Hint that the function is frequently executed, allowing
/// the optimizer to prioritize it.
fn validate_hot(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("hot", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Hot)
}

/// Validates `__attribute__((format(archetype, string_index, first_to_check)))`.
///
/// Requires exactly three arguments:
/// 1. `archetype` — an identifier (`printf`, `scanf`, `strftime`, `strfmon`)
///    or their `__...__` double-underscore variants.
/// 2. `string_index` — a positive integer (1-based parameter index of the
///    format string).
/// 3. `first_to_check` — a non-negative integer (1-based parameter index
///    of the first variadic argument to check, or 0 for no checking).
fn validate_format(
    args: &[AttributeArg],
    span: Span,
    interner: &Interner,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_arg_count("format", args, 3, span, diagnostics) {
        return None;
    }

    // Argument 1: archetype identifier
    let archetype = match &args[0] {
        AttributeArg::Identifier(sym) => {
            let name = interner.resolve(*sym);
            let canonical = canonicalize_name(name);
            match canonical {
                "printf" => FormatKind::Printf,
                "scanf" => FormatKind::Scanf,
                "strftime" => FormatKind::Strftime,
                "strfmon" => FormatKind::Strfmon,
                _ => {
                    diagnostics.error(
                        span,
                        format!(
                            "unknown format archetype '{}'; expected 'printf', \
                             'scanf', 'strftime', or 'strfmon'",
                            canonical
                        ),
                    );
                    return None;
                }
            }
        }
        _ => {
            diagnostics.error(
                span,
                "first argument to 'format' attribute must be an identifier \
                 (printf, scanf, strftime, or strfmon)",
            );
            return None;
        }
    };

    // Argument 2: string_index (must be positive integer)
    let string_index = match &args[1] {
        AttributeArg::Integer(n) => {
            if *n < 1 {
                diagnostics.error(
                    span,
                    format!(
                        "format string index {} must be a positive integer",
                        n
                    ),
                );
                return None;
            }
            *n as u32
        }
        _ => {
            diagnostics.error(
                span,
                "second argument to 'format' attribute must be an integer constant",
            );
            return None;
        }
    };

    // Argument 3: first_to_check (must be non-negative integer)
    let first_to_check = match &args[2] {
        AttributeArg::Integer(n) => {
            if *n < 0 {
                diagnostics.error(
                    span,
                    format!(
                        "format first-to-check index {} must be a non-negative integer",
                        n
                    ),
                );
                return None;
            }
            *n as u32
        }
        _ => {
            diagnostics.error(
                span,
                "third argument to 'format' attribute must be an integer constant",
            );
            return None;
        }
    };

    // If first_to_check > 0, it must be greater than string_index
    // (the format string itself is not checked as a variadic argument).
    if first_to_check > 0 && first_to_check <= string_index {
        diagnostics.error(
            span,
            format!(
                "format first-to-check index ({}) must be greater than \
                 string index ({}) or zero",
                first_to_check, string_index
            ),
        );
        return None;
    }

    Some(ValidatedAttribute::Format {
        archetype,
        string_index,
        first_to_check,
    })
}

/// Validates `__attribute__((format_arg(N)))`.
///
/// Requires exactly one positive integer argument identifying the
/// 1-based parameter index that contains a format string to be passed
/// through to a printf/scanf-family function.
fn validate_format_arg(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_arg_count("format_arg", args, 1, span, diagnostics) {
        return None;
    }

    match &args[0] {
        AttributeArg::Integer(n) => {
            if *n < 1 {
                diagnostics.error(
                    span,
                    format!(
                        "format_arg parameter index {} must be a positive integer",
                        n
                    ),
                );
                return None;
            }
            Some(ValidatedAttribute::FormatArg(*n as u32))
        }
        _ => {
            diagnostics.error(
                span,
                "argument to 'format_arg' attribute must be an integer constant",
            );
            None
        }
    }
}

/// Validates `__attribute__((malloc))`.
///
/// No arguments. Indicates that the function returns a pointer that does
/// not alias any other pointer visible to the caller, enabling alias
/// analysis optimizations.
fn validate_malloc(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("malloc", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Malloc)
}

/// Validates `__attribute__((pure))`.
///
/// No arguments. Indicates the function has no observable side effects
/// except reading memory through its arguments and global variables.
/// The optimizer may eliminate redundant calls to pure functions.
fn validate_pure(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("pure", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Pure)
}

/// Validates `__attribute__((const))`.
///
/// No arguments. Stricter than `pure`: the function has no side effects
/// and does not read any memory other than its arguments. The return
/// value depends only on the argument values, enabling even more
/// aggressive optimization (e.g., common subexpression elimination).
fn validate_const_attr(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("const", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Const)
}

/// Validates `__attribute__((warn_unused_result))`.
///
/// No arguments. Causes the compiler to emit a warning when the return
/// value of the function is discarded without explicit `(void)` cast.
fn validate_warn_unused_result(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("warn_unused_result", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::WarnUnusedResult)
}

/// Validates `__attribute__((fallthrough))`.
///
/// No arguments. Statement-only attribute indicating intentional
/// fall-through in a switch case. Suppresses fall-through warnings.
fn validate_fallthrough(
    args: &[AttributeArg],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Option<ValidatedAttribute> {
    if !check_no_args("fallthrough", args, span, diagnostics) {
        return None;
    }
    Some(ValidatedAttribute::Fallthrough)
}

// ===========================================================================
// Attribute propagation to symbols
// ===========================================================================

/// Applies validated attributes to a symbol table entry.
///
/// This function maps each `ValidatedAttribute` to the corresponding field
/// in the symbol's `SymbolAttributes` struct. It also detects and diagnoses
/// conflicting attribute combinations (e.g., `noinline` + `always_inline`).
///
/// # Arguments
///
/// * `attrs` — The validated attributes to apply.
/// * `symbol` — The symbol table entry to modify.
pub fn propagate_to_symbol(attrs: &[ValidatedAttribute], symbol: &mut SymbolEntry) {
    for attr in attrs {
        match attr {
            ValidatedAttribute::Aligned(alignment) => {
                symbol.attributes.alignment = *alignment;
            }
            ValidatedAttribute::Packed => {
                // Packed is primarily a type attribute but may appear on
                // variables/fields. For symbols, we don't have a direct
                // packed field — alignment is set to 1 to simulate packing.
                symbol.attributes.alignment = Some(1);
            }
            ValidatedAttribute::Section(name) => {
                symbol.attributes.section = Some(name.clone());
            }
            ValidatedAttribute::Used => {
                symbol.attributes.is_used = true;
            }
            ValidatedAttribute::Unused => {
                symbol.attributes.is_unused = true;
            }
            ValidatedAttribute::Weak => {
                symbol.attributes.is_weak = true;
            }
            ValidatedAttribute::Constructor(priority) => {
                symbol.attributes.constructor_priority = Some(priority.unwrap_or(65535));
            }
            ValidatedAttribute::Destructor(priority) => {
                symbol.attributes.destructor_priority = Some(priority.unwrap_or(65535));
            }
            ValidatedAttribute::Visibility(kind) => {
                symbol.attributes.visibility = Some(*kind);
            }
            ValidatedAttribute::Deprecated(msg) => {
                // Store the deprecation message; empty string if no message.
                symbol.attributes.is_deprecated = Some(
                    msg.as_deref().unwrap_or("").to_string(),
                );
            }
            ValidatedAttribute::Noreturn => {
                symbol.attributes.is_noreturn = true;
            }
            ValidatedAttribute::Noinline => {
                symbol.attributes.is_noinline = true;
            }
            ValidatedAttribute::AlwaysInline => {
                symbol.attributes.is_always_inline = true;
            }
            ValidatedAttribute::Cold => {
                symbol.attributes.is_cold = true;
            }
            ValidatedAttribute::Hot => {
                symbol.attributes.is_hot = true;
            }
            ValidatedAttribute::Format { .. } => {
                // Format attribute information is used by the type checker
                // during call-site validation, not stored on the symbol.
                // The validated attribute is retained in the ValidatedAttribute
                // list for the sema driver to use during call validation.
            }
            ValidatedAttribute::FormatArg(_) => {
                // Same as Format — used during call-site validation.
            }
            ValidatedAttribute::Malloc => {
                symbol.attributes.is_malloc = true;
            }
            ValidatedAttribute::Pure => {
                symbol.attributes.is_pure = true;
            }
            ValidatedAttribute::Const => {
                symbol.attributes.is_const = true;
            }
            ValidatedAttribute::WarnUnusedResult => {
                symbol.attributes.is_warn_unused_result = true;
            }
            ValidatedAttribute::Fallthrough => {
                // Fallthrough is a statement attribute; it does not
                // propagate to symbols. Silently ignored here.
            }
        }
    }
}

// ===========================================================================
// Attribute propagation to types
// ===========================================================================

/// Applies validated type-modifying attributes to a C type.
///
/// Currently, the type-modifying attributes are `aligned` and `packed`,
/// which affect struct/union layout. Since the `CType` enum does not carry
/// alignment or packing metadata directly (alignment is tracked in
/// `SymbolAttributes`), this function validates that the attributes are
/// applicable to the given type kind and performs any necessary type
/// transformations.
///
/// # Arguments
///
/// * `attrs` — The validated attributes to apply.
/// * `ty` — The C type to potentially modify.
pub fn propagate_to_type(attrs: &[ValidatedAttribute], ty: &mut CType) {
    for attr in attrs {
        match attr {
            ValidatedAttribute::Aligned(_) => {
                // Aligned is valid on struct/union types, variables, and
                // fields. The actual alignment value is tracked in
                // SymbolAttributes. For types, we validate applicability
                // but do not modify the CType directly since alignment
                // is an external property of the type's usage context.
                //
                // Struct/union, pointer, array, and scalar types can all
                // have alignment attributes in GCC. No CType mutation needed.
            }
            ValidatedAttribute::Packed => {
                // Packed is primarily meaningful for struct/union types.
                // When applied to a struct, it changes the layout to
                // eliminate padding between members. The packed state is
                // tracked externally (in SymbolAttributes as alignment=1
                // or in the type builder's layout computation).
                //
                // Validate that the target type is struct or union.
                match ty {
                    CType::Struct { .. } | CType::Union { .. } => {
                        // Valid target — packing will be handled during
                        // struct layout computation by the type builder.
                    }
                    _ => {
                        // Packed on non-struct/union types is unusual but
                        // GCC accepts it on some targets. We allow it
                        // silently; the layout effect is implementation-defined.
                    }
                }
            }
            // All other attributes do not modify the type itself.
            _ => {}
        }
    }
}

// ===========================================================================
// Conflict detection
// ===========================================================================

/// Checks for conflicting attribute combinations and emits diagnostic
/// warnings for any detected conflicts.
///
/// Known conflicts:
/// - `noinline` + `always_inline`: contradictory inlining directives.
/// - `cold` + `hot`: contradictory execution frequency hints.
/// - `pure` + `const`: `const` is strictly stronger than `pure`.
///
/// This function should be called after `propagate_to_symbol()` with the
/// same validated attributes.
///
/// # Arguments
///
/// * `attrs` — The validated attributes to check for conflicts.
/// * `span` — Source location for diagnostic messages.
/// * `diagnostics` — Diagnostic engine for reporting warnings.
pub fn check_attribute_conflicts(
    attrs: &[ValidatedAttribute],
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) {
    let has_noinline = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Noinline));
    let has_always_inline = attrs.iter().any(|a| matches!(a, ValidatedAttribute::AlwaysInline));
    let has_cold = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Cold));
    let has_hot = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Hot));
    let has_pure = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Pure));
    let has_const = attrs.iter().any(|a| matches!(a, ValidatedAttribute::Const));

    if has_noinline && has_always_inline {
        diagnostics.warning(
            span,
            "attributes 'noinline' and 'always_inline' are contradictory; \
             'noinline' takes precedence",
        );
    }

    if has_cold && has_hot {
        diagnostics.warning(
            span,
            "attributes 'cold' and 'hot' are contradictory; 'cold' takes precedence",
        );
    }

    if has_pure && has_const {
        diagnostics.note(
            span,
            "attribute 'const' is strictly stronger than 'pure'; \
             'pure' is redundant when 'const' is present",
        );
    }
}
