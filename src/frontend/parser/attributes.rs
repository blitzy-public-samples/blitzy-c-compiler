#![allow(clippy::result_unit_err)]
//! `__attribute__((...))` parsing for all 21+ required GCC attributes.
//!
//! This module implements parsing for the GCC `__attribute__` specifier syntax
//! used extensively throughout the Linux kernel and other C codebases. It handles
//! the double-parenthesis convention `__attribute__((attr1, attr2, ...))` and
//! produces [`Attribute`] AST nodes with accurate source spans.
//!
//! # Supported Attributes (22)
//!
//! ## Simple (no arguments):
//! - `packed` — struct packing, remove inter-member padding
//! - `used` — mark symbol as used (prevents dead code elimination by the linker)
//! - `unused` — suppress "unused variable/function" warnings
//! - `noreturn` — function does not return to its caller
//! - `noinline` — prevent the compiler from inlining this function
//! - `always_inline` — force inlining of this function
//! - `cold` — unlikely execution path (affects code layout heuristics)
//! - `hot` — likely execution path (affects code layout heuristics)
//! - `malloc` — return value is a fresh, unaliased pointer
//! - `pure` — function has no side effects (reads global memory only)
//! - `const` — function has no side effects and reads nothing from memory
//! - `warn_unused_result` — warn if the return value is discarded
//! - `weak` — emit the symbol with weak linkage binding
//! - `fallthrough` — suppress implicit fallthrough warning in switch
//!
//! ## With arguments:
//! - `aligned(N)` — set alignment (optional N, default = maximum alignment)
//! - `section("name")` — place the symbol in the named ELF section
//! - `visibility("vis")` — set symbol visibility (default/hidden/protected/internal)
//! - `constructor(priority)` — mark as constructor (optional priority integer)
//! - `destructor(priority)` — mark as destructor (optional priority integer)
//! - `format(archetype, str_idx, first_to_check)` — enable printf/scanf format checking
//! - `format_arg(str_idx)` — identify the format-string parameter index
//! - `deprecated("msg")` — mark as deprecated (optional message string)
//!
//! # Spelling Conventions
//!
//! Both the plain spelling (`aligned`) and the double-underscore spelling
//! (`__aligned__`) are accepted. The canonical (stripped) name is stored
//! in the [`Attribute`] AST node for uniform downstream handling.
//!
//! # Error Handling
//!
//! - Unknown attributes emit a **warning** (not a hard error) for forward
//!   compatibility with newer GCC versions.
//! - Malformed attributes attempt recovery by scanning to the next `,` or
//!   closing `)` inside the specifier.
//! - Missing double parentheses produce a clear diagnostic.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules (`crate::common`, `crate::frontend`). No external crates.

use super::ast::{Attribute, AttributeArg};
use super::expressions;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ============================================================================
// Public Entry Point
// ============================================================================

/// Parse a `__attribute__((...))` specifier.
///
/// Called when the parser encounters the `__attribute__` keyword token
/// (which is still the current token — this function consumes it).
///
/// Returns a `Vec<Attribute>` because a single `__attribute__` specifier
/// may contain multiple comma-separated attributes:
///
/// ```c
/// __attribute__((aligned(16), packed, section(".data")))
/// //            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/// //            One specifier, three attributes
/// ```
///
/// # Grammar
///
/// ```text
/// attribute-specifier:
///     '__attribute__' '(' '(' attribute-list ')' ')'
///
/// attribute-list:
///     attribute
///     attribute-list ',' attribute
///     /* empty — GCC allows __attribute__(()) */
///
/// attribute:
///     attribute-name
///     attribute-name '(' argument-list ')'
///
/// attribute-name:
///     identifier
///     keyword   /* e.g., 'const', 'volatile' inside attribute context */
/// ```
///
/// # Errors
///
/// Returns `Err(())` on unrecoverable parse failures (missing `((`). On
/// per-attribute failures, error recovery skips to the next `,` or `)` and
/// continues parsing remaining attributes.
pub fn parse_attribute_specifier(parser: &mut Parser<'_>) -> Result<Vec<Attribute>, ()> {
    // Consume the `__attribute__` keyword token.
    parser.expect(TokenKind::Attribute)?;

    // Expect the first opening parenthesis of the double-paren syntax.
    if parser.expect(TokenKind::LeftParen).is_err() {
        return Err(());
    }

    // Expect the second opening parenthesis.
    if parser.expect(TokenKind::LeftParen).is_err() {
        // Recovery: try to skip to closing `)` and bail.
        skip_to_right_paren(parser);
        let _ = parser.match_token(&TokenKind::RightParen);
        return Err(());
    }

    let mut attributes = Vec::new();

    // Parse comma-separated attribute list.
    // An empty list is valid: `__attribute__(())`.
    while !parser.check(&TokenKind::RightParen) && !parser.current.is_eof() {
        // Allow empty attributes (e.g., trailing comma before `))`)
        if parser.check(&TokenKind::Comma) {
            parser.advance();
            continue;
        }

        match parse_single_attribute(parser) {
            Ok(attr) => attributes.push(attr),
            Err(()) => {
                // Error recovery: skip to next comma or closing paren.
                skip_to_comma_or_close(parser);
            }
        }

        // Consume comma separator between attributes.
        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    // Expect the first closing parenthesis.
    parser.expect(TokenKind::RightParen)?;

    // Expect the second closing parenthesis.
    parser.expect(TokenKind::RightParen)?;

    Ok(attributes)
}

// ============================================================================
// Single Attribute Parsing
// ============================================================================

/// Parse a single attribute within the comma-separated attribute list.
///
/// Extracts the attribute name, strips leading/trailing `__`, checks for
/// an argument list (opening `(`), and dispatches to the appropriate
/// argument parser based on the canonical attribute name.
///
/// Unknown attributes emit a warning and have their arguments skipped
/// (balanced parenthesis consumption) for forward compatibility.
fn parse_single_attribute(parser: &mut Parser<'_>) -> Result<Attribute, ()> {
    let start_span = parser.current_span();

    // Extract the attribute name (may be a keyword token like `const`).
    let raw_name_sym = match extract_attribute_name(parser) {
        Some(sym) => sym,
        None => {
            let span = parser.current_span();
            parser.error(span, "expected attribute name");
            return Err(());
        }
    };

    // Resolve the raw name string and compute the canonical (stripped) name.
    let raw_name = parser.resolve_symbol(raw_name_sym).to_owned();
    let canonical = strip_underscores(&raw_name);

    // Intern the canonical name for the Attribute AST node.
    let name_sym = if canonical != raw_name.as_str() {
        parser.intern(canonical)
    } else {
        raw_name_sym
    };

    // Determine whether arguments follow (opening parenthesis).
    let has_args = parser.check(&TokenKind::LeftParen);

    // Dispatch to the appropriate argument parser based on canonical name.
    let args = parse_attribute_args(parser, canonical, has_args, start_span)?;

    let span = parser.make_span(start_span);
    Ok(Attribute {
        name: name_sym,
        args,
        span,
    })
}

/// Dispatch attribute argument parsing based on the canonical attribute name.
///
/// This function encodes the knowledge of which attributes take which kinds
/// of arguments. It routes to specialised parsers for known attributes and
/// falls back to a generic "skip balanced parens" strategy for unknown ones.
fn parse_attribute_args(
    parser: &mut Parser<'_>,
    canonical_name: &str,
    has_args: bool,
    start_span: Span,
) -> Result<Vec<AttributeArg>, ()> {
    match canonical_name {
        // =================================================================
        // Simple attributes — no arguments expected
        // =================================================================
        "packed" | "used" | "unused" | "noreturn" | "noinline" | "always_inline"
        | "cold" | "hot" | "malloc" | "pure" | "const" | "warn_unused_result"
        | "weak" | "fallthrough"
        // Additional common simple attributes (forward-compat with kernel code):
        | "may_alias" | "no_instrument_function" | "noclone" | "no_reorder"
        | "flatten" | "externally_visible" | "no_sanitize_address"
        | "no_sanitize_thread" | "no_sanitize_undefined" | "no_split_stack"
        | "leaf" | "nothrow" | "returns_nonnull" | "returns_twice"
        | "no_stack_protector" | "transparent_union" | "artificial"
        | "no_caller_saved_registers" | "naked" | "target"
        | "optimize" | "no_profile_instrument_function"
        // More glibc/GCC attributes recognised as no-ops:
        | "nocf_check" | "gnu_inline" | "nonstring" | "retain"
        | "warn_unused" => {
            if has_args {
                // Some nominally-simple attributes may have optional/ignorable
                // arguments (e.g., GCC tolerates `used(0)` in some versions).
                // Parse them generically so we don't choke.
                parse_generic_parenthesized_args(parser)
            } else {
                Ok(Vec::new())
            }
        }

        // =================================================================
        // aligned — optional constant-expression argument
        // =================================================================
        "aligned" => {
            if has_args {
                parse_aligned_args(parser)
            } else {
                // `aligned` without arguments means maximum target alignment.
                Ok(Vec::new())
            }
        }

        // =================================================================
        // section — required string argument
        // =================================================================
        "section" => {
            if has_args {
                parse_string_arg_attribute(parser)
            } else {
                let span = parser.current_span();
                parser.error(span, "'section' attribute requires a string argument");
                Err(())
            }
        }

        // =================================================================
        // visibility — required string argument with validation
        // =================================================================
        "visibility" => {
            if has_args {
                parse_visibility_args(parser)
            } else {
                let span = parser.current_span();
                parser.error(span, "'visibility' attribute requires a string argument");
                Err(())
            }
        }

        // =================================================================
        // deprecated — optional string message argument
        // =================================================================
        "deprecated" => {
            if has_args {
                parse_deprecated_args(parser)
            } else {
                Ok(Vec::new())
            }
        }

        // =================================================================
        // constructor / destructor — optional priority integer
        // =================================================================
        "constructor" | "destructor" => {
            if has_args {
                parse_priority_args(parser)
            } else {
                Ok(Vec::new())
            }
        }

        // =================================================================
        // format — (archetype, string_index, first_to_check)
        // =================================================================
        "format" => {
            if has_args {
                parse_format_args(parser)
            } else {
                let span = parser.current_span();
                parser.error(span, "'format' attribute requires arguments");
                Err(())
            }
        }

        // =================================================================
        // format_arg — (string_index)
        // =================================================================
        "format_arg" => {
            if has_args {
                parse_format_arg_args(parser)
            } else {
                let span = parser.current_span();
                parser.error(
                    span,
                    "'format_arg' attribute requires an integer argument",
                );
                Err(())
            }
        }

        // =================================================================
        // sentinel — optional integer argument
        // =================================================================
        "sentinel" => {
            if has_args {
                parse_aligned_args(parser) // Same pattern: optional int
            } else {
                Ok(Vec::new())
            }
        }

        // =================================================================
        // error / warning — required string message
        // =================================================================
        "error" | "warning" => {
            if has_args {
                parse_string_arg_attribute(parser)
            } else {
                let span = parser.current_span();
                parser.error(
                    span,
                    &format!(
                        "'{}' attribute requires a string argument",
                        canonical_name,
                    ),
                );
                Err(())
            }
        }

        // =================================================================
        // Attributes with optional / generic arguments from glibc / GCC
        // headers. Recognised to avoid "unknown attribute" warnings.
        // Arguments are consumed generically; semantic validation (if any)
        // happens in the attribute handler.
        // =================================================================
        "nonnull" | "alloc_size" | "alloc_align" | "assume_aligned"
        | "access" | "no_sanitize" | "cleanup" | "tls_model"
        | "alias" | "weakref" | "ifunc" | "mode" | "copy"
        | "symver" | "patchable_function_entry" => {
            if has_args {
                parse_generic_parenthesized_args(parser)
            } else {
                Ok(Vec::new())
            }
        }

        // =================================================================
        // Unknown attribute — emit warning and skip arguments
        // =================================================================
        _ => {
            let span = parser.make_span(start_span);
            parser.warn(
                span,
                &format!("unknown attribute '{}' ignored", canonical_name),
            );
            if has_args {
                skip_balanced_parens_consume(parser);
            }
            Ok(Vec::new())
        }
    }
}

// ============================================================================
// Attribute Name Extraction
// ============================================================================

/// Extract the attribute name from the current token.
///
/// Inside `__attribute__((...))`, C keywords like `const` and `volatile`
/// are treated as plain identifiers (attribute names). This function handles
/// both regular identifiers and keyword tokens that can appear as attribute
/// names.
///
/// Returns `None` if the current token cannot be an attribute name (e.g.,
/// a punctuator or literal).
fn extract_attribute_name(parser: &mut Parser<'_>) -> Option<Symbol> {
    match &parser.current.kind {
        // Regular identifier — the common case.
        TokenKind::Identifier(sym) => {
            let s = *sym;
            parser.advance();
            Some(s)
        }

        // C keywords that can appear as attribute names.
        // GCC treats these as identifiers within __attribute__((...)).
        TokenKind::Const => {
            parser.advance();
            Some(parser.intern("const"))
        }
        TokenKind::Volatile => {
            parser.advance();
            Some(parser.intern("volatile"))
        }
        TokenKind::Inline => {
            parser.advance();
            Some(parser.intern("inline"))
        }

        // Catch-all: any other keyword token. Inside attribute context,
        // keywords are treated as plain identifier names. Use the keyword's
        // canonical C spelling.
        kind if kind.is_keyword() => {
            if let Some(spelling) = kind.keyword_str() {
                let sym = parser.intern(spelling);
                parser.advance();
                Some(sym)
            } else {
                None
            }
        }

        _ => None,
    }
}

/// Strip leading and trailing double underscores from an attribute name.
///
/// GCC allows both `aligned` and `__aligned__` spellings. The canonical
/// form is the one without underscores:
/// - `__aligned__` → `aligned`
/// - `__packed__`  → `packed`
/// - `aligned`     → `aligned`  (unchanged)
/// - `__foo`       → `__foo`    (only stripped if BOTH pairs present)
fn strip_underscores(name: &str) -> &str {
    let bytes = name.as_bytes();
    let len = bytes.len();
    if len > 4
        && bytes[0] == b'_'
        && bytes[1] == b'_'
        && bytes[len - 2] == b'_'
        && bytes[len - 1] == b'_'
    {
        &name[2..len - 2]
    } else {
        name
    }
}

// ============================================================================
// Attribute-Specific Argument Parsers
// ============================================================================

/// Parse arguments for the `aligned` attribute.
///
/// ```c
/// aligned       /* max alignment — no args */
/// aligned(16)   /* explicit alignment */
/// aligned(N)    /* constant expression */
/// ```
fn parse_aligned_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    // Parse the alignment value as a constant expression.
    let expr = expressions::parse_constant_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::Expression(Box::new(expr))])
}

/// Parse a single string argument attribute (used by `section`, `error`, `warning`).
///
/// ```c
/// section(".data")
/// error("do not call this function")
/// ```
fn parse_string_arg_attribute(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    let (bytes, span) = parse_string_literal(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::String(bytes, span)])
}

/// Parse arguments for the `visibility` attribute with value validation.
///
/// ```c
/// visibility("default")
/// visibility("hidden")
/// visibility("protected")
/// visibility("internal")
/// ```
fn parse_visibility_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    let (bytes, span) = parse_string_literal(parser)?;

    // Validate the visibility string value.
    let vis_str = String::from_utf8_lossy(&bytes);
    match vis_str.as_ref() {
        "default" | "hidden" | "protected" | "internal" => { /* valid */ }
        _ => {
            parser.warn(
                span,
                &format!(
                    "unknown visibility '{}'; expected 'default', 'hidden', 'protected', or 'internal'",
                    vis_str
                ),
            );
        }
    }

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::String(bytes, span)])
}

/// Parse arguments for the `deprecated` attribute.
///
/// ```c
/// deprecated            /* no message */
/// deprecated("use bar instead")  /* with message */
/// ```
fn parse_deprecated_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    // The argument is a string literal message.
    let (bytes, span) = parse_string_literal(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::String(bytes, span)])
}

/// Parse arguments for `constructor`/`destructor` attributes (optional priority).
///
/// ```c
/// constructor           /* default priority */
/// constructor(101)      /* explicit priority */
/// destructor(65535)
/// ```
fn parse_priority_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    let expr = expressions::parse_constant_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::Expression(Box::new(expr))])
}

/// Parse arguments for the `format` attribute.
///
/// ```c
/// format(printf, 1, 2)
/// format(scanf, 2, 3)
/// format(__printf__, 1, 2)   /* double-underscore variant */
/// ```
///
/// The first argument is an archetype identifier, followed by two integer
/// constant expressions separated by commas.
fn parse_format_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    // First argument: archetype identifier (printf, scanf, strftime, etc.)
    let archetype_span = parser.current_span();
    let archetype_sym = match extract_attribute_name(parser) {
        Some(sym) => sym,
        None => {
            let span = parser.current_span();
            parser.error(span, "expected format archetype (e.g., 'printf', 'scanf')");
            return Err(());
        }
    };

    // Comma separator.
    parser.expect(TokenKind::Comma)?;

    // Second argument: string-index (integer constant expression).
    let string_idx_expr = expressions::parse_constant_expression(parser)?;

    // Comma separator.
    parser.expect(TokenKind::Comma)?;

    // Third argument: first-to-check (integer constant expression).
    let first_to_check_expr = expressions::parse_constant_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![
        AttributeArg::Identifier(archetype_sym, archetype_span),
        AttributeArg::Expression(Box::new(string_idx_expr)),
        AttributeArg::Expression(Box::new(first_to_check_expr)),
    ])
}

/// Parse arguments for the `format_arg` attribute.
///
/// ```c
/// format_arg(1)    /* the first parameter is the format string */
/// ```
fn parse_format_arg_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    let expr = expressions::parse_constant_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    Ok(vec![AttributeArg::Expression(Box::new(expr))])
}

// ============================================================================
// Generic Argument Parsing
// ============================================================================

/// Parse a parenthesized argument list generically.
///
/// Used for attributes whose argument structure is not specifically known
/// but whose arguments should still be captured in the AST. Each argument
/// is classified as an identifier, string literal, or expression.
fn parse_generic_parenthesized_args(parser: &mut Parser<'_>) -> Result<Vec<AttributeArg>, ()> {
    parser.expect(TokenKind::LeftParen)?;

    let mut args = Vec::new();

    while !parser.check(&TokenKind::RightParen) && !parser.current.is_eof() {
        let arg = parse_single_generic_arg(parser)?;
        args.push(arg);

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }
    }

    parser.expect(TokenKind::RightParen)?;

    Ok(args)
}

/// Parse a single generic attribute argument.
///
/// Tries to classify the argument as one of:
/// 1. A string literal → `AttributeArg::String`
/// 2. An identifier → `AttributeArg::Identifier`
/// 3. An expression → `AttributeArg::Expression`
fn parse_single_generic_arg(parser: &mut Parser<'_>) -> Result<AttributeArg, ()> {
    match &parser.current.kind {
        // String literal argument.
        TokenKind::StringLiteral { value, .. } => {
            let bytes = value.as_bytes().to_vec();
            let span = parser.current_span();
            parser.advance();
            Ok(AttributeArg::String(bytes, span))
        }

        // Identifier argument (not followed by `(` — otherwise it might be
        // a nested function-like attribute which we treat as an identifier
        // for simplicity).
        TokenKind::Identifier(sym) => {
            let s = *sym;
            let span = parser.current_span();
            parser.advance();
            Ok(AttributeArg::Identifier(s, span))
        }

        // Everything else: parse as an expression.
        _ => {
            let expr = expressions::parse_assignment_expression(parser)?;
            Ok(AttributeArg::Expression(Box::new(expr)))
        }
    }
}

// ============================================================================
// String Literal Helper
// ============================================================================

/// Parse a string literal from the current token.
///
/// Returns the raw byte content and the source span. Handles adjacent
/// string literal concatenation (e.g., `"hello" " world"`).
fn parse_string_literal(parser: &mut Parser<'_>) -> Result<(Vec<u8>, Span), ()> {
    match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => {
            let mut bytes = value.as_bytes().to_vec();
            let start_span = parser.current_span();
            parser.advance();

            // Concatenate adjacent string literals (C11 §6.4.5).
            while let TokenKind::StringLiteral { value, .. } = &parser.current.kind {
                bytes.extend_from_slice(value.as_bytes());
                parser.advance();
            }

            let span = parser.make_span(start_span);
            Ok((bytes, span))
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected string literal");
            Err(())
        }
    }
}

// ============================================================================
// Error Recovery Helpers
// ============================================================================

/// Skip tokens until we find a comma or a closing right-parenthesis at the
/// current nesting level inside the attribute list.
///
/// This is used for per-attribute error recovery: if parsing one attribute
/// fails, we skip to the next `,` or `)` so the remaining attributes in the
/// specifier can still be parsed.
fn skip_to_comma_or_close(parser: &mut Parser<'_>) {
    let mut depth: u32 = 0;
    while !parser.current.is_eof() {
        match &parser.current.kind {
            TokenKind::LeftParen => {
                depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                if depth == 0 {
                    // We've reached the attribute list's closing paren — stop
                    // without consuming so the caller can handle it.
                    return;
                }
                depth -= 1;
                parser.advance();
            }
            TokenKind::Comma if depth == 0 => {
                // Found a comma at the top level of the attribute list — stop
                // without consuming so the caller can handle the separator.
                return;
            }
            _ => {
                parser.advance();
            }
        }
    }
}

/// Skip a balanced parenthesized group, consuming the opening `(` and the
/// matching closing `)`.
///
/// Used for unknown attributes that have arguments we cannot parse but
/// must consume to continue parsing.
fn skip_balanced_parens_consume(parser: &mut Parser<'_>) {
    if !parser.match_token(&TokenKind::LeftParen) {
        return;
    }

    let mut depth: u32 = 1;
    while depth > 0 && !parser.current.is_eof() {
        match &parser.current.kind {
            TokenKind::LeftParen => depth += 1,
            TokenKind::RightParen => depth -= 1,
            _ => {}
        }
        parser.advance();
    }
}

/// Skip tokens until a right parenthesis is found (not consumed).
///
/// Used for recovery when the double-paren opening failed partway.
fn skip_to_right_paren(parser: &mut Parser<'_>) {
    while !parser.current.is_eof() && !parser.check(&TokenKind::RightParen) {
        parser.advance();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_underscores_basic() {
        assert_eq!(strip_underscores("aligned"), "aligned");
        assert_eq!(strip_underscores("__aligned__"), "aligned");
        assert_eq!(strip_underscores("__packed__"), "packed");
        assert_eq!(strip_underscores("__section__"), "section");
        assert_eq!(strip_underscores("__visibility__"), "visibility");
    }

    #[test]
    fn test_strip_underscores_edge_cases() {
        // Only strip when BOTH leading and trailing __ are present.
        assert_eq!(strip_underscores("__foo"), "__foo");
        assert_eq!(strip_underscores("foo__"), "foo__");
        // Minimum length: must be > 4 characters (__ + at least 1 char + __).
        assert_eq!(strip_underscores("____"), "____");
        assert_eq!(strip_underscores("__a__"), "a");
        // Empty and short strings.
        assert_eq!(strip_underscores(""), "");
        assert_eq!(strip_underscores("_"), "_");
        assert_eq!(strip_underscores("__"), "__");
        assert_eq!(strip_underscores("___"), "___");
    }
}
