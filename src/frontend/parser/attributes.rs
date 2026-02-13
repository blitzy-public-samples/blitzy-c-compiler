//! GCC `__attribute__((...))` parsing for the BCC C11 parser.
//!
//! Parses the GCC `__attribute__` syntax and produces [`Attribute`] AST nodes
//! that are attached to declarations, types, and statements. The semantic
//! validation of individual attributes (e.g., checking that `aligned(N)` is a
//! power of two, or that `visibility` receives a valid string) is performed
//! later in the semantic analysis phase (`src/frontend/sema/attribute_handler.rs`).
//!
//! # Supported Attributes
//!
//! This parser recognises all 21+ GCC attributes required for Linux kernel
//! compilation:
//!
//! | Category              | Attributes                                                        |
//! |-----------------------|-------------------------------------------------------------------|
//! | Alignment / Layout    | `aligned`, `packed`                                               |
//! | Section / Linkage     | `section`, `used`, `unused`, `weak`                               |
//! | Constructor / Destructor | `constructor`, `destructor`                                    |
//! | Visibility            | `visibility`                                                      |
//! | Function Behaviour    | `deprecated`, `noreturn`, `noinline`, `always_inline`, `cold`,    |
//! |                       | `hot`, `malloc`, `pure`, `const`, `warn_unused_result`            |
//! | Format Checking       | `format`, `format_arg`                                            |
//! | Statement             | `fallthrough`                                                     |
//!
//! Additional attributes commonly encountered in the Linux kernel source are
//! also recognised to suppress spurious warnings.
//!
//! # Syntax
//!
//! ```text
//! gnu-attributes:
//!     __attribute__ '(' '(' attribute-list ')' ')'
//!     gnu-attributes __attribute__ '(' '(' attribute-list ')' ')'
//!
//! attribute-list:
//!     attribute
//!     attribute-list ',' attribute
//!
//! attribute:
//!     <empty>
//!     attribute-name
//!     attribute-name '(' argument-list ')'
//!
//! argument-list:
//!     argument
//!     argument-list ',' argument
//!
//! argument:
//!     integer-literal
//!     string-literal
//!     identifier
//!     constant-expression
//!     assignment-expression
//! ```
//!
//! # GCC Underscore Convention
//!
//! Each attribute may be spelled in two equivalent forms:
//! - Plain form: `aligned`, `packed`, `section`
//! - Double-underscore form: `__aligned__`, `__packed__`, `__section__`
//!
//! Both forms are semantically identical. The [`normalize_attribute_name`]
//! function strips the leading and trailing `__` to produce the canonical
//! form used for matching in the semantic analyzer.
//!
//! # Error Recovery
//!
//! - **Unknown attribute names:** parsed generically with a warning diagnostic.
//! - **Malformed arguments:** skip to closing `)` and continue parsing
//!   remaining attributes in the list.
//! - **Missing `))`:** skip to a recovery point (`)` or `;`) and report an
//!   error diagnostic.
//! - **Premature EOF:** detected explicitly to prevent infinite loops.

use super::ast::{Attribute, AttributeArg, Expression, Span};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Known attribute registry
// ===========================================================================

/// All known GCC attribute names supported by BCC.
///
/// This list is used for diagnostic purposes: when an attribute name is not
/// recognised, a warning is emitted but parsing continues normally. The
/// actual semantic validation and interpretation of each attribute happens
/// in `src/frontend/sema/attribute_handler.rs`.
///
/// The list includes the 21 core attributes required by the specification
/// plus additional common attributes encountered in the Linux kernel source.
const KNOWN_ATTRIBUTES: &[&str] = &[
    // ---- Core required attributes (§4.3 / §0.6.1) — the primary 21+ ----
    "aligned",
    "packed",
    "section",
    "used",
    "unused",
    "weak",
    "constructor",
    "destructor",
    "visibility",
    "deprecated",
    "noreturn",
    "noinline",
    "always_inline",
    "cold",
    "hot",
    "format",
    "format_arg",
    "malloc",
    "pure",
    "const",
    "warn_unused_result",
    "fallthrough",
    // ---- Additional kernel-common attributes ----
    "nonnull",
    "returns_nonnull",
    "sentinel",
    "cleanup",
    "may_alias",
    "transparent_union",
    "mode",
    "no_instrument_function",
    "noclone",
    "no_sanitize",
    "no_sanitize_address",
    "no_sanitize_thread",
    "no_sanitize_undefined",
    "no_split_stack",
    "assume_aligned",
    "alloc_size",
    "alloc_align",
    "warn_if_not_aligned",
    "error",
    "warning",
    "externally_visible",
    "no_reorder",
    "leaf",
    "nothrow",
    "artificial",
    "flatten",
    "target",
    "optimize",
    "no_caller_saved_registers",
    "nocf_check",
    "gnu_inline",
    "naked",
    "ifunc",
    "interrupt",
    "alias",
    "weakref",
    "common",
    "nocommon",
    "designated_init",
    "nonstring",
    "noplt",
    "regparm",
    "stdcall",
    "cdecl",
    "fastcall",
    "returns_twice",
    "force_align_arg_pointer",
    "retain",
    "copy",
    "access",
    "tls_model",
    "selectany",
    "init_priority",
    "ms_struct",
    "gcc_struct",
    "weak_import",
    "volatile",
    "inline",
    "signal",
];

/// Attributes whose arguments are expected to be constant expressions.
///
/// For these attributes, the expression parser dispatches to
/// [`parse_constant_expression`](super::expressions::parse_constant_expression)
/// rather than the more general assignment-expression parser. This does not
/// affect parsing correctness (both accept the same token sequences at the
/// grammar level), but it documents the intent and enables the semantic
/// analyzer to enforce the constant-expression constraint.
const CONST_EXPR_ATTRIBUTES: &[&str] = &[
    "aligned",
    "constructor",
    "destructor",
    "format_arg",
    "alloc_size",
    "alloc_align",
    "warn_if_not_aligned",
    "assume_aligned",
    "init_priority",
    "regparm",
];

// ===========================================================================
// Public API
// ===========================================================================

/// Normalizes a GCC attribute name by stripping the leading `__` and
/// trailing `__` if both are present.
///
/// # GCC Convention
///
/// GCC allows each attribute to be spelled two ways:
/// - Plain form: `aligned`, `packed`, `section`
/// - Double-underscore form: `__aligned__`, `__packed__`, `__section__`
///
/// Both forms are semantically identical. This function maps the
/// double-underscore form to the canonical plain form so that the
/// semantic analyzer can match on a single canonical spelling.
///
/// The original spelling is preserved via the interned [`Symbol`] in the
/// AST for diagnostic messages that wish to show the user's exact text.
///
/// # Arguments
///
/// * `name` — The raw attribute name as written in source code.
///
/// # Returns
///
/// A string slice with the leading and trailing `__` removed if both
/// were present, or the original string unchanged.
///
/// # Examples
///
/// ```text
/// normalize_attribute_name("__aligned__") == "aligned"
/// normalize_attribute_name("packed")      == "packed"
/// normalize_attribute_name("__section__") == "section"
/// normalize_attribute_name("__x")         == "__x"   // only prefix, unchanged
/// normalize_attribute_name("x__")         == "x__"   // only suffix, unchanged
/// normalize_attribute_name("____")        == "____"   // 4 chars < 5, unchanged
/// ```
pub fn normalize_attribute_name(name: &str) -> &str {
    // The GCC convention requires BOTH a leading `__` and a trailing `__`.
    // The minimum valid name is `__X__` (5 characters), so we reject any
    // name shorter than 5 characters.
    if name.len() >= 5 && name.starts_with("__") && name.ends_with("__") {
        &name[2..name.len() - 2]
    } else {
        name
    }
}

/// Parses one or more consecutive `__attribute__((...))` specifiers and
/// returns a flat list of all parsed attributes.
///
/// This function is called from declaration specifier parsing, declarator
/// parsing, and label statement parsing — anywhere GCC attributes may
/// appear in the grammar.
///
/// Multiple `__attribute__((...))` specifiers may appear in sequence on
/// the same declaration. This function collects all attributes from all
/// specifiers into a single flat vector.
///
/// # Grammar
///
/// ```text
/// gnu-attributes:
///     __attribute__ '(' '(' attribute-list ')' ')'
///     gnu-attributes __attribute__ '(' '(' attribute-list ')' ')'
/// ```
///
/// # Error Recovery
///
/// - On premature EOF: returns an error immediately.
/// - On missing `))`: attempts to skip to a recovery point and returns
///   the attributes parsed so far.
/// - Individual attribute parsing errors are caught, allowing subsequent
///   attributes in the same list to be parsed.
///
/// # Returns
///
/// A vector of [`Attribute`] nodes, one for each attribute across all
/// consecutive `__attribute__` specifiers. May be empty if no
/// `__attribute__` tokens are present at the current position.
pub fn parse_attribute_list(parser: &mut Parser<'_>) -> Result<Vec<Attribute>, ParseError> {
    let mut attrs: Vec<Attribute> = Vec::new();

    // Loop over consecutive __attribute__ specifiers.
    // GCC allows multiple adjacent specifiers:
    //   __attribute__((a)) __attribute__((b, c))
    while parser.check(TokenKind::Attribute) {
        parser.advance(); // consume `__attribute__` or `__attribute`

        // Expect the opening double-parenthesis `((`
        parser.expect(TokenKind::LeftParen)?;
        parser.expect(TokenKind::LeftParen)?;

        // Parse the comma-separated attribute list.
        // An empty list is valid: __attribute__(())
        if !parser.check(TokenKind::RightParen) {
            loop {
                // Guard against premature EOF to prevent an infinite loop
                // on malformed input missing the closing `))`.
                if parser.check(TokenKind::Eof) {
                    let span = parser.current().span;
                    parser.diagnostics.error(
                        span,
                        "unexpected end of file in __attribute__ specification",
                    );
                    return Err(ParseError {
                        span,
                        message: "unexpected end of file in __attribute__ specification"
                            .to_string(),
                        expected: Some("))".to_string()),
                    });
                }

                // Attempt to parse a single attribute. On failure, recover
                // by skipping to the next comma or closing `)`.
                match parse_single_attribute(parser) {
                    Ok(attr) => attrs.push(attr),
                    Err(_) => {
                        // Error recovery: skip tokens until we find a comma,
                        // right paren, or EOF. This allows parsing to continue
                        // with the next attribute in the list.
                        skip_to_arg_recovery(parser);
                    }
                }

                // Consume the separating comma; break if none found.
                if !parser.eat(TokenKind::Comma) {
                    break;
                }

                // Handle trailing comma before `))`:
                //   __attribute__((aligned(16),))  — valid in GCC
                if parser.check(TokenKind::RightParen) {
                    break;
                }
            }
        }

        // Expect closing `))`
        // If the first `)` is missing, attempt recovery.
        if parser.expect(TokenKind::RightParen).is_err() {
            skip_to_outer_recovery(parser);
            // Try to consume remaining `)` if we recovered to one
            let _ = parser.eat(TokenKind::RightParen);
            continue;
        }
        if parser.expect(TokenKind::RightParen).is_err() {
            skip_to_outer_recovery(parser);
            continue;
        }
    }

    Ok(attrs)
}

// ===========================================================================
// Internal — single attribute parsing
// ===========================================================================

/// Parses a single attribute within an attribute list.
///
/// ```text
/// attribute:
///     <empty>
///     attribute-name
///     attribute-name '(' argument-list ')'
///
/// attribute-name:
///     identifier
///     keyword        (GCC allows `const`, `volatile`, etc.)
/// ```
///
/// Empty attributes arise from sequences like `__attribute__((a, , b))`
/// where adjacent commas create an empty slot.
fn parse_single_attribute(parser: &mut Parser<'_>) -> Result<Attribute, ParseError> {
    let start = parser.current().span;

    // Extract the attribute name from the current token.
    // Most attribute names are identifiers, but GCC also permits certain
    // keywords (e.g., `const`, `volatile`) and their GCC alternate forms
    // (e.g., `__const__`, `__volatile__`) as attribute names.
    let (raw_sym, raw_name_string) = extract_attribute_name(parser)?;

    // Normalize the GCC underscore convention: __aligned__ → aligned
    let normalized = normalize_attribute_name(&raw_name_string);

    // Emit a diagnostic for unrecognised attribute names. The warning
    // allows compilation to proceed (attributes are not required to be
    // known for parsing to succeed), but informs the developer that the
    // attribute will have no effect.
    if !raw_name_string.is_empty() && !is_known_attribute(normalized) {
        parser.diagnostics.warning(
            start,
            &format!("unknown attribute '{}'", raw_name_string),
        );
    }

    // Intern the normalised form for the AST node's `name` field.
    // This allows the semantic analyzer to match on canonical names
    // without worrying about `__foo__` vs `foo` spelling differences.
    let name_sym: Symbol = if normalized != raw_name_string.as_str() {
        parser.interner.intern(normalized)
    } else {
        raw_sym
    };

    // Parse the optional argument list: attribute-name '(' args... ')'
    let args: Vec<AttributeArg> = if parser.check(TokenKind::LeftParen) {
        parse_attribute_args(parser, normalized)?
    } else {
        Vec::new()
    };

    // Compute the span covering the entire attribute (name + arguments).
    let end_span = parser.current().span;
    let span = Span::merge(start, end_span);

    Ok(Attribute {
        name: name_sym,
        args,
        span,
    })
}

/// Extracts an attribute name from the current token position.
///
/// Returns a tuple of `(interned_symbol, raw_string)`:
/// - `interned_symbol`: the [`Symbol`] for the raw attribute name text
/// - `raw_string`: the string representation used for normalisation
///
/// GCC allows identifiers, C keywords (`const`, `volatile`), and GCC
/// alternate keyword forms (`__const__`, `__volatile__`, `__inline__`)
/// as attribute names. All of these are handled here.
///
/// If the current token is not a valid attribute name (e.g., a comma
/// creating an empty slot), returns the empty symbol.
fn extract_attribute_name(
    parser: &mut Parser<'_>,
) -> Result<(Symbol, String), ParseError> {
    match &parser.current().kind {
        // Standard identifier — covers the vast majority of attribute names
        // (aligned, packed, section, used, unused, weak, noreturn, etc.)
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            let name = parser.interner.resolve(sym).to_string();
            parser.advance();
            Ok((sym, name))
        }

        // C `const` keyword used as an attribute name.
        // GCC interprets `__attribute__((const))` as the "pure function with
        // no side effects and no reads of global memory" attribute.
        TokenKind::Const => {
            let name = "const".to_string();
            let sym: Symbol = parser.interner.intern("const");
            parser.advance();
            Ok((sym, name))
        }

        // C `volatile` keyword — rare but valid as an attribute name.
        TokenKind::Volatile => {
            let name = "volatile".to_string();
            let sym: Symbol = parser.interner.intern("volatile");
            parser.advance();
            Ok((sym, name))
        }

        // GCC `__const__` / `__const` keyword form.
        // Normalises to `const` via the underscore convention.
        TokenKind::ConstGcc => {
            let name = "__const__".to_string();
            let sym: Symbol = parser.interner.intern("__const__");
            parser.advance();
            Ok((sym, name))
        }

        // GCC `__volatile__` / `__volatile` keyword form.
        TokenKind::VolatileGcc => {
            let name = "__volatile__".to_string();
            let sym: Symbol = parser.interner.intern("__volatile__");
            parser.advance();
            Ok((sym, name))
        }

        // GCC `__inline__` / `__inline` keyword form.
        TokenKind::InlineGcc => {
            let name = "__inline__".to_string();
            let sym: Symbol = parser.interner.intern("__inline__");
            parser.advance();
            Ok((sym, name))
        }

        // Empty or unrecognised token in attribute name position.
        // GCC allows empty entries between commas:
        //   __attribute__((,))        — one empty slot
        //   __attribute__((a,,b))     — empty slot between a and b
        // We return the empty symbol so the caller can include a zero-width
        // attribute in the list (which is silently ignored during sema).
        _ => {
            let _span = parser.current().span;
            let empty_sym: Symbol = parser.interner.intern("");
            Ok((empty_sym, String::new()))
        }
    }
}

// ===========================================================================
// Internal — argument list parsing
// ===========================================================================

/// Parses the parenthesised argument list for a single attribute.
///
/// The parsing strategy adapts based on the normalised attribute name:
///
/// - Attributes known to expect constant expressions (aligned, constructor,
///   destructor, format_arg) use [`parse_constant_expression`] for their
///   expression-form arguments.
/// - Unknown or generic attributes use [`parse_assignment_expression`] as
///   a fallback for maximum flexibility.
/// - Simple arguments (string literals, integer literals, standalone
///   identifiers) are consumed directly without invoking the full
///   expression parser.
///
/// # Arguments
///
/// * `parser` — The parser state.
/// * `normalized_name` — The canonical (normalised) attribute name, used
///   to select the expression parsing strategy.
fn parse_attribute_args(
    parser: &mut Parser<'_>,
    normalized_name: &str,
) -> Result<Vec<AttributeArg>, ParseError> {
    // Consume the opening `(`
    parser.expect(TokenKind::LeftParen)?;

    let mut args: Vec<AttributeArg> = Vec::new();

    // Empty argument lists are valid: `noreturn()` ≡ `noreturn`.
    if !parser.check(TokenKind::RightParen) {
        loop {
            // Guard against premature EOF inside the argument list
            if parser.check(TokenKind::Eof) {
                let span = parser.current().span;
                parser.diagnostics.error(
                    span,
                    &format!(
                        "unexpected end of file in arguments for attribute '{}'",
                        normalized_name,
                    ),
                );
                return Err(ParseError {
                    span,
                    message: format!(
                        "unexpected end of file in arguments for attribute '{}'",
                        normalized_name,
                    ),
                    expected: Some(")".to_string()),
                });
            }

            // Attempt to parse a single argument. On failure, recover by
            // skipping to the next comma or closing paren so that
            // subsequent arguments (or the attribute list) can continue.
            match parse_single_arg(parser, normalized_name) {
                Ok(arg) => args.push(arg),
                Err(_) => {
                    skip_to_arg_recovery(parser);
                    // If we landed on a comma, continue with the next arg
                    if parser.check(TokenKind::Comma) {
                        parser.advance();
                        continue;
                    }
                    // Otherwise we're at `)` or EOF — exit the loop
                    break;
                }
            }

            // Consume the separating comma; break if none found
            if !parser.eat(TokenKind::Comma) {
                break;
            }

            // Handle trailing comma: `aligned(16,)` is valid in GCC
            if parser.check(TokenKind::RightParen) {
                break;
            }
        }
    }

    // Consume the closing `)`
    parser.expect(TokenKind::RightParen)?;
    Ok(args)
}

/// Parses a single argument to an attribute.
///
/// Dispatches to the appropriate parsing strategy based on the current
/// token and the attribute context:
///
/// 1. **Integer literal** — If the token is a standalone integer (followed
///    by `,` or `)`), consumed directly as [`AttributeArg::Integer`].
///    Otherwise parsed as a full expression (e.g., `1 << 4`).
///
/// 2. **String literal** — Always consumed directly as
///    [`AttributeArg::String`]. String literals in attribute arguments
///    are never part of larger expressions.
///
/// 3. **Identifier** — If standalone (followed by `,` or `)`), consumed
///    as [`AttributeArg::Identifier`]. Otherwise parsed as an expression
///    (e.g., function-call-like constructs).
///
/// 4. **Other tokens** — Parsed as a constant expression (for attributes
///    in [`CONST_EXPR_ATTRIBUTES`]) or an assignment expression (for all
///    others), producing [`AttributeArg::Expression`].
fn parse_single_arg(
    parser: &mut Parser<'_>,
    normalized_name: &str,
) -> Result<AttributeArg, ParseError> {
    // -------------------------------------------------------------------
    // Fast path: standalone integer literal
    //   e.g., 16 in aligned(16), or 101 in constructor(101)
    // -------------------------------------------------------------------
    if let TokenKind::IntegerLiteral { value, .. } = &parser.current().kind {
        let val = *value as i64;
        // Check whether this is a standalone integer by peeking at the
        // next token. If followed by `,` or `)`, it is a simple argument
        // and we can avoid invoking the full expression parser.
        let next_is_terminator = matches!(
            parser.peek_ahead(1).kind,
            TokenKind::Comma | TokenKind::RightParen
        );
        if next_is_terminator {
            parser.advance();
            return Ok(AttributeArg::Integer(val));
        }
        // Fall through to expression parsing for compound expressions
        // such as `1 << 4`, `PAGE_SIZE - 1`, etc.
    }

    // -------------------------------------------------------------------
    // Fast path: string literal
    //   e.g., ".init.text" in section(".init.text"),
    //         "default" in visibility("default"),
    //         "deprecated msg" in deprecated("deprecated msg")
    // -------------------------------------------------------------------
    if let TokenKind::StringLiteral { value, .. } = &parser.current().kind {
        let val: Vec<u8> = value.clone();
        parser.advance();
        return Ok(AttributeArg::String(val));
    }

    // -------------------------------------------------------------------
    // Fast path: standalone identifier
    //   e.g., printf in format(printf, 1, 2),
    //         strftime in format(strftime, 3, 0)
    // -------------------------------------------------------------------
    if let TokenKind::Identifier(sym) = &parser.current().kind {
        let sym_copy: Symbol = *sym;
        let next_is_terminator = matches!(
            parser.peek_ahead(1).kind,
            TokenKind::Comma | TokenKind::RightParen
        );
        if next_is_terminator {
            parser.advance();
            return Ok(AttributeArg::Identifier(sym_copy));
        }
        // Fall through to expression parsing for identifiers that are
        // part of larger expressions (e.g., macro-expanded constants).
    }

    // -------------------------------------------------------------------
    // Slow path: full expression parsing
    // -------------------------------------------------------------------
    // Use parse_constant_expression for attributes that expect compile-time
    // constant values (aligned, constructor priority, etc.), and
    // parse_assignment_expression for all other attributes to provide
    // maximum flexibility in accepted syntax.
    let expr: Expression = if expects_constant_expression(normalized_name) {
        super::expressions::parse_constant_expression(parser)?
    } else {
        super::expressions::parse_assignment_expression(parser)?
    };

    Ok(AttributeArg::Expression(Box::new(expr)))
}

// ===========================================================================
// Internal — helpers
// ===========================================================================

/// Returns `true` if the given normalised attribute name is in the set of
/// known attributes recognised by BCC.
///
/// Unknown attributes are still parsed (with a warning), but this function
/// is used to decide whether to emit the warning.
#[inline]
fn is_known_attribute(normalized_name: &str) -> bool {
    KNOWN_ATTRIBUTES.contains(&normalized_name)
}

/// Returns `true` if the given normalised attribute name expects its
/// arguments to be constant expressions rather than general assignment
/// expressions.
///
/// This distinction is primarily informational at the parser level (both
/// parse the same token sequences), but it guides the semantic analyzer
/// to enforce the constant-expression requirement.
#[inline]
fn expects_constant_expression(normalized_name: &str) -> bool {
    CONST_EXPR_ATTRIBUTES.contains(&normalized_name)
}

/// Error recovery helper: skip tokens until we find a comma, right
/// parenthesis, or EOF at the current nesting depth.
///
/// This is used when an individual attribute argument fails to parse.
/// The function tracks parenthesis nesting so that commas and closing
/// parens inside nested sub-expressions are not treated as recovery
/// points.
fn skip_to_arg_recovery(parser: &mut Parser<'_>) {
    let mut paren_depth: u32 = 0;
    loop {
        match &parser.current().kind {
            // Stop at end of file
            TokenKind::Eof => break,
            // Track nested parentheses
            TokenKind::LeftParen => {
                paren_depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                if paren_depth == 0 {
                    // At the argument-list level — don't consume; let
                    // the caller's expect(RightParen) handle it.
                    break;
                }
                paren_depth -= 1;
                parser.advance();
            }
            // Comma at the argument-list level is a recovery point
            TokenKind::Comma if paren_depth == 0 => {
                // Don't consume — let the caller's comma loop handle it.
                break;
            }
            // Skip all other tokens
            _ => {
                parser.advance();
            }
        }
    }
}

/// Error recovery helper for the outer `))` of `__attribute__((...))`.
///
/// Skips tokens until we find a right parenthesis, semicolon, or EOF.
/// This is used when the expected closing `))` is missing or malformed,
/// to resynchronise the parser with the surrounding declaration context.
fn skip_to_outer_recovery(parser: &mut Parser<'_>) {
    loop {
        match &parser.current().kind {
            // Stop at end of file — can't recover further
            TokenKind::Eof => break,
            // Found a right paren — consume it (might be part of `))`)
            TokenKind::RightParen => {
                parser.advance();
                break;
            }
            // Semicolons delimit declarations — stop but don't consume,
            // so the declaration parser can see the `;`.
            TokenKind::Semicolon => break,
            // Skip everything else
            _ => {
                parser.advance();
            }
        }
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_plain_name() {
        assert_eq!(normalize_attribute_name("aligned"), "aligned");
        assert_eq!(normalize_attribute_name("packed"), "packed");
        assert_eq!(normalize_attribute_name("section"), "section");
        assert_eq!(normalize_attribute_name("used"), "used");
        assert_eq!(normalize_attribute_name("unused"), "unused");
        assert_eq!(normalize_attribute_name("weak"), "weak");
        assert_eq!(normalize_attribute_name("constructor"), "constructor");
        assert_eq!(normalize_attribute_name("destructor"), "destructor");
        assert_eq!(normalize_attribute_name("visibility"), "visibility");
        assert_eq!(normalize_attribute_name("deprecated"), "deprecated");
        assert_eq!(normalize_attribute_name("noreturn"), "noreturn");
        assert_eq!(normalize_attribute_name("noinline"), "noinline");
        assert_eq!(normalize_attribute_name("always_inline"), "always_inline");
        assert_eq!(normalize_attribute_name("cold"), "cold");
        assert_eq!(normalize_attribute_name("hot"), "hot");
        assert_eq!(normalize_attribute_name("format"), "format");
        assert_eq!(normalize_attribute_name("format_arg"), "format_arg");
        assert_eq!(normalize_attribute_name("malloc"), "malloc");
        assert_eq!(normalize_attribute_name("pure"), "pure");
        assert_eq!(normalize_attribute_name("const"), "const");
        assert_eq!(
            normalize_attribute_name("warn_unused_result"),
            "warn_unused_result"
        );
        assert_eq!(normalize_attribute_name("fallthrough"), "fallthrough");
    }

    #[test]
    fn test_normalize_underscore_form() {
        assert_eq!(normalize_attribute_name("__aligned__"), "aligned");
        assert_eq!(normalize_attribute_name("__packed__"), "packed");
        assert_eq!(normalize_attribute_name("__section__"), "section");
        assert_eq!(normalize_attribute_name("__used__"), "used");
        assert_eq!(normalize_attribute_name("__unused__"), "unused");
        assert_eq!(normalize_attribute_name("__weak__"), "weak");
        assert_eq!(normalize_attribute_name("__constructor__"), "constructor");
        assert_eq!(normalize_attribute_name("__destructor__"), "destructor");
        assert_eq!(normalize_attribute_name("__visibility__"), "visibility");
        assert_eq!(normalize_attribute_name("__deprecated__"), "deprecated");
        assert_eq!(normalize_attribute_name("__noreturn__"), "noreturn");
        assert_eq!(normalize_attribute_name("__noinline__"), "noinline");
        assert_eq!(
            normalize_attribute_name("__always_inline__"),
            "always_inline"
        );
        assert_eq!(normalize_attribute_name("__cold__"), "cold");
        assert_eq!(normalize_attribute_name("__hot__"), "hot");
        assert_eq!(normalize_attribute_name("__format__"), "format");
        assert_eq!(normalize_attribute_name("__format_arg__"), "format_arg");
        assert_eq!(normalize_attribute_name("__malloc__"), "malloc");
        assert_eq!(normalize_attribute_name("__pure__"), "pure");
        assert_eq!(normalize_attribute_name("__const__"), "const");
        assert_eq!(
            normalize_attribute_name("__warn_unused_result__"),
            "warn_unused_result"
        );
        assert_eq!(normalize_attribute_name("__fallthrough__"), "fallthrough");
    }

    #[test]
    fn test_normalize_edge_cases() {
        // Only prefix — no stripping
        assert_eq!(normalize_attribute_name("__x"), "__x");
        // Only suffix — no stripping
        assert_eq!(normalize_attribute_name("x__"), "x__");
        // Too short to contain both __ and __
        assert_eq!(normalize_attribute_name("____"), "____");
        // Empty string
        assert_eq!(normalize_attribute_name(""), "");
        // Single underscore prefix/suffix
        assert_eq!(normalize_attribute_name("_aligned_"), "_aligned_");
        // Minimum valid case: __X__ (5 chars)
        assert_eq!(normalize_attribute_name("__X__"), "X");
        // Just underscores, exactly 5
        assert_eq!(normalize_attribute_name("_____"), "_");
    }

    #[test]
    fn test_known_attributes_recognised() {
        // All 21 core required attributes must be recognised
        assert!(is_known_attribute("aligned"));
        assert!(is_known_attribute("packed"));
        assert!(is_known_attribute("section"));
        assert!(is_known_attribute("used"));
        assert!(is_known_attribute("unused"));
        assert!(is_known_attribute("weak"));
        assert!(is_known_attribute("constructor"));
        assert!(is_known_attribute("destructor"));
        assert!(is_known_attribute("visibility"));
        assert!(is_known_attribute("deprecated"));
        assert!(is_known_attribute("noreturn"));
        assert!(is_known_attribute("noinline"));
        assert!(is_known_attribute("always_inline"));
        assert!(is_known_attribute("cold"));
        assert!(is_known_attribute("hot"));
        assert!(is_known_attribute("format"));
        assert!(is_known_attribute("format_arg"));
        assert!(is_known_attribute("malloc"));
        assert!(is_known_attribute("pure"));
        assert!(is_known_attribute("const"));
        assert!(is_known_attribute("warn_unused_result"));
        assert!(is_known_attribute("fallthrough"));
    }

    #[test]
    fn test_unknown_attribute_not_recognised() {
        assert!(!is_known_attribute("totally_made_up"));
        assert!(!is_known_attribute("not_a_real_attribute"));
    }

    #[test]
    fn test_const_expr_attributes() {
        assert!(expects_constant_expression("aligned"));
        assert!(expects_constant_expression("constructor"));
        assert!(expects_constant_expression("destructor"));
        assert!(expects_constant_expression("format_arg"));
        // Non-const-expr attributes
        assert!(!expects_constant_expression("section"));
        assert!(!expects_constant_expression("format"));
        assert!(!expects_constant_expression("visibility"));
        assert!(!expects_constant_expression("deprecated"));
    }
}
