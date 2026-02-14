//! Inline assembly parsing module for the BCC C11 parser.
//!
//! Parses `asm` / `__asm__` statements with the full GCC extended syntax:
//!
//! - `asm volatile` / `__asm__ __volatile__` — volatile qualifier prevents
//!   the compiler from optimizing away or reordering the assembly block.
//! - `asm goto` — enables the assembly to jump to C labels in the enclosing
//!   function scope.
//! - Template strings — one or more adjacent string literal fragments
//!   concatenated. Contain AT&T syntax assembly with operand placeholders
//!   (`%0`, `%1`, `%[name]`, `%%` for literal `%`). May include
//!   `.pushsection` / `.popsection` directives used by the Linux kernel
//!   for alternative instruction patching.
//! - Output operands — expressions written to by the assembly, with
//!   constraint strings prefixed by `=` (write-only) or `+` (read-write).
//! - Input operands — expressions read by the assembly, with constraint
//!   strings such as `"r"`, `"i"`, `"n"`, `"m"`, `"g"`, or digit
//!   constraints `"0"`–`"9"` matching an output operand.
//! - Clobber lists — string literals naming registers or special flags
//!   (`"memory"`, `"cc"`) that the assembly modifies beyond outputs.
//! - Named operands — `[name]` prefix syntax allowing template references
//!   via `%[name]` instead of positional `%N`.
//! - Goto labels — for `asm goto`, a fourth colon section lists C label
//!   identifiers reachable from the assembly via `%l[name]` or `%l[N]`.
//!
//! # Grammar
//!
//! ```text
//! asm-statement:
//!     ('asm' | '__asm__') ['volatile' | '__volatile__'] ['goto']
//!     '(' asm-string-literal
//!         [':' [asm-operand-list]]              // outputs
//!         [':' [asm-operand-list]]              // inputs
//!         [':' [asm-clobber-list]]              // clobbers
//!         [':' [asm-goto-labels]]               // goto labels (asm goto only)
//!     ')' ';'
//!
//! asm-operand-list:
//!     asm-operand (',' asm-operand)*
//!
//! asm-operand:
//!     ['[' identifier ']'] string-literal '(' expression ')'
//!
//! asm-clobber-list:
//!     string-literal (',' string-literal)*
//!
//! asm-goto-labels:
//!     identifier (',' identifier)*
//! ```
//!
//! # Error Recovery
//!
//! On parse errors within asm statements, the parser attempts to skip to
//! the closing `)` and `;` to allow continued parsing of subsequent
//! statements. Diagnostics are emitted with clear messages describing the
//! expected syntax element.
//!
//! # Integration
//!
//! - **Called by:** `src/frontend/parser/statements.rs` when `asm`/`__asm__`
//!   keyword is encountered.
//! - **Produces:** `AsmStatement` AST nodes consumed by:
//!   - `src/frontend/sema/attribute_handler.rs` for full constraint validation
//!   - `src/ir/lowering/asm_lowering.rs` for IR inline assembly generation
//! - **Required for:** Linux kernel compilation which uses inline assembly
//!   extensively for architecture-specific operations, atomic primitives,
//!   context switching, and memory barriers.

use super::ast::{AsmOperand, AsmStatement, Expression, Span};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public API — four exported functions per schema
// ===========================================================================

/// Parses a complete inline assembly statement.
///
/// This is the primary entry point, called from the statement parser when
/// the current token is `asm` or `__asm__` (`TokenKind::AsmKeyword`).
///
/// # Syntax
///
/// ```c
/// asm volatile goto (
///     "template %0, %[name]"
///     : [output] "=r" (out_var)           // output operands
///     : [name] "r" (in_var)               // input operands
///     : "memory", "cc"                    // clobbers
///     : label1, label2                    // goto labels
/// );
/// ```
///
/// # Arguments
///
/// * `parser` — Mutable reference to the parser state. The current token
///   must be `TokenKind::AsmKeyword`.
///
/// # Returns
///
/// An `AsmStatement` AST node fully populated with all parsed components,
/// or a `ParseError` if the syntax is malformed.
///
/// # Error Recovery
///
/// On internal parse failures, the function attempts to skip to the
/// closing `)` and `;` before returning the error, allowing the parser
/// to continue with subsequent statements.
pub fn parse_asm_statement(parser: &mut Parser<'_>) -> Result<AsmStatement, ParseError> {
    // Precondition: the caller must have verified that the current token
    // is `asm` or `__asm__` before dispatching to this function.
    debug_assert!(
        parser.current().is(TokenKind::AsmKeyword),
        "parse_asm_statement called with non-asm token: {:?}",
        parser.current().kind,
    );

    let start_span = parser.advance(); // consume `asm` / `__asm__`

    // Parse optional qualifiers: volatile / __volatile__ and goto.
    // Both `volatile` (C keyword) and `__volatile__` (GCC form) are accepted.
    // The qualifiers may appear in any order; each may appear at most once.
    let mut is_volatile = false;
    let mut is_goto = false;

    loop {
        if parser.check(TokenKind::Volatile) || parser.check(TokenKind::VolatileGcc) {
            is_volatile = true;
            parser.advance();
        } else if parser.check(TokenKind::Goto) {
            is_goto = true;
            parser.advance();
        } else {
            break;
        }
    }

    // Opening parenthesis is mandatory.
    if let Err(e) = parser.expect(TokenKind::LeftParen) {
        skip_to_asm_end(parser);
        return Err(e);
    }

    // Parse template string(s) — one or more adjacent string literals.
    let template = match parse_asm_template(parser) {
        Ok(t) => t,
        Err(e) => {
            skip_to_asm_end(parser);
            return Err(e);
        }
    };

    // The four colon-separated sections are all optional. If a colon is
    // absent, all subsequent sections are also absent.
    let mut outputs: Vec<AsmOperand> = Vec::new();
    let mut inputs: Vec<AsmOperand> = Vec::new();
    let mut clobbers: Vec<String> = Vec::new();
    let mut goto_labels: Vec<Symbol> = Vec::new();

    // First colon: output operands.
    if parser.eat(TokenKind::Colon) {
        if !is_colon_or_rparen(parser) {
            match parse_asm_operands_internal(parser, true) {
                Ok(ops) => outputs = ops,
                Err(e) => {
                    skip_to_asm_end(parser);
                    return Err(e);
                }
            }
        }

        // Second colon: input operands.
        if parser.eat(TokenKind::Colon) {
            if !is_colon_or_rparen(parser) {
                match parse_asm_operands_internal(parser, false) {
                    Ok(ops) => inputs = ops,
                    Err(e) => {
                        skip_to_asm_end(parser);
                        return Err(e);
                    }
                }
            }

            // Third colon: clobber list.
            if parser.eat(TokenKind::Colon) {
                if !is_colon_or_rparen(parser) {
                    match parse_clobber_list(parser) {
                        Ok(c) => clobbers = c,
                        Err(e) => {
                            skip_to_asm_end(parser);
                            return Err(e);
                        }
                    }
                }

                // Fourth colon: goto labels (only valid when `goto` qualifier present).
                if is_goto && parser.eat(TokenKind::Colon) && !parser.check(TokenKind::RightParen) {
                    match parse_goto_labels(parser) {
                        Ok(l) => goto_labels = l,
                        Err(e) => {
                            skip_to_asm_end(parser);
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    // Closing parenthesis and semicolon.
    let rparen_span = match parser.expect(TokenKind::RightParen) {
        Ok(s) => s,
        Err(e) => {
            skip_to_asm_end(parser);
            return Err(e);
        }
    };
    let end_span = match parser.expect_semicolon() {
        Ok(s) => s,
        Err(_) => {
            // If semicolon is missing, use the rparen span as fallback end.
            rparen_span
        }
    };

    let span = Span::merge(start_span, end_span);

    Ok(AsmStatement {
        is_volatile,
        is_goto,
        template,
        outputs,
        inputs,
        clobbers,
        goto_labels,
        span,
    })
}

/// Parses a comma-separated list of assembly operands (output or input).
///
/// This is the public API entry point for operand list parsing. It delegates
/// to the internal operand parser without performing output-specific constraint
/// validation (the `is_output` flag is set to `false`). Callers that need
/// output constraint validation should use `parse_asm_statement` which
/// handles the distinction internally.
///
/// # Operand Syntax
///
/// ```text
/// asm-operand:
///     ['[' identifier ']'] string-literal '(' expression ')'
/// ```
///
/// ## Output Constraints
///
/// Output operands require a constraint string beginning with `=` (write-only
/// output) or `+` (read-write). An optional `&` after the prefix marks the
/// operand as early-clobber (e.g., `"=&r"`).
///
/// ## Input Constraints
///
/// Input operands use constraint strings without `=` or `+` prefix:
/// - `"r"` — general-purpose register
/// - `"i"` — immediate constant
/// - `"n"` — numeric immediate
/// - `"m"` — memory operand
/// - `"g"` — general (register, memory, or immediate)
/// - `"0"` through `"9"` — matching constraint (same location as output N)
///
/// # Arguments
///
/// * `parser` — Mutable reference to the parser state.
///
/// # Returns
///
/// A vector of `AsmOperand` nodes, or a `ParseError` on syntax errors.
pub fn parse_asm_operands(parser: &mut Parser<'_>) -> Result<Vec<AsmOperand>, ParseError> {
    parse_asm_operands_internal(parser, false)
}

/// Parses a comma-separated list of clobber string literals.
///
/// Clobber strings inform the compiler which registers and state the
/// inline assembly may modify beyond the declared output operands.
///
/// # Common Clobber Values
///
/// - `"memory"` — the assembly reads or writes arbitrary memory locations
///   not covered by input/output operands. Forces the compiler to flush
///   all cached memory values before the asm and reload after.
/// - `"cc"` — the assembly modifies the condition codes / flags register.
/// - Named registers (architecture-specific): `"rax"`, `"rbx"`, `"x0"`,
///   `"a0"`, etc.
///
/// # Syntax
///
/// ```text
/// asm-clobber-list:
///     string-literal (',' string-literal)*
/// ```
///
/// # Arguments
///
/// * `parser` — Mutable reference to the parser state.
///
/// # Returns
///
/// A vector of clobber name strings, or a `ParseError` on syntax errors.
pub fn parse_clobber_list(parser: &mut Parser<'_>) -> Result<Vec<String>, ParseError> {
    let mut clobbers: Vec<String> = Vec::new();

    loop {
        match &parser.current().kind {
            TokenKind::StringLiteral { value, .. } => {
                let clobber_name = String::from_utf8_lossy(value).into_owned();
                parser.advance();
                clobbers.push(clobber_name);
            }
            _ => {
                // If we haven't parsed any clobbers yet, this is an error.
                // If we have at least one, the caller may have hit the next
                // section separator, so we stop gracefully.
                if clobbers.is_empty() {
                    let span = parser.current().span;
                    let msg = "expected clobber string literal".to_string();
                    parser.diagnostics.error(span, &msg);
                    return Err(ParseError {
                        span,
                        message: msg,
                        expected: Some("string literal".to_string()),
                    });
                }
                break;
            }
        }

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(clobbers)
}

/// Parses a comma-separated list of goto label identifiers for `asm goto`.
///
/// In `asm goto(...)` statements, a fourth colon section lists identifiers
/// that name C labels in the enclosing function. The assembly template can
/// reference these labels using `%l[name]` (by name) or `%l[N]` (by
/// positional index in this list).
///
/// The labels are interned as `Symbol` handles for zero-cost comparison
/// during subsequent semantic analysis (which validates that the labels
/// actually exist in the enclosing function scope) and IR lowering (which
/// wires the labels to basic block targets).
///
/// # Syntax
///
/// ```text
/// asm-goto-labels:
///     identifier (',' identifier)*
/// ```
///
/// # Arguments
///
/// * `parser` — Mutable reference to the parser state.
///
/// # Returns
///
/// A vector of `Symbol` handles for the goto label identifiers, or a
/// `ParseError` if a non-identifier token is encountered.
pub fn parse_goto_labels(parser: &mut Parser<'_>) -> Result<Vec<Symbol>, ParseError> {
    let mut labels: Vec<Symbol> = Vec::new();

    loop {
        match &parser.current().kind {
            TokenKind::Identifier(sym) => {
                labels.push(*sym);
                parser.advance();
            }
            _ => {
                if labels.is_empty() {
                    let span = parser.current().span;
                    let msg = "expected goto label identifier".to_string();
                    parser.diagnostics.error(span, &msg);
                    return Err(ParseError {
                        span,
                        message: msg,
                        expected: Some("identifier".to_string()),
                    });
                }
                break;
            }
        }

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(labels)
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Returns `true` if the current token is `:` or `)` — used to detect
/// empty operand / clobber / label sections between colons or before the
/// closing parenthesis.
#[inline]
fn is_colon_or_rparen(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Colon | TokenKind::RightParen
    )
}

/// Error recovery helper: skip tokens until we reach `)` then `;`, or EOF.
///
/// When a parse error occurs inside an asm statement, this function attempts
/// to consume tokens up through the closing `)` and `;` so that the parser
/// can continue with subsequent statements without cascading errors.
fn skip_to_asm_end(parser: &mut Parser<'_>) {
    // First, skip to the closing `)`, respecting nesting.
    let mut paren_depth: u32 = 1; // We are inside the opening `(`
    while !parser.at_end() {
        match parser.current().kind {
            TokenKind::LeftParen => {
                paren_depth += 1;
                parser.advance();
            }
            TokenKind::RightParen => {
                paren_depth -= 1;
                parser.advance();
                if paren_depth == 0 {
                    break;
                }
            }
            _ => {
                parser.advance();
            }
        }
    }
    // Consume trailing semicolon if present.
    parser.eat(TokenKind::Semicolon);
}

/// Parses one or more adjacent string literal fragments that form the
/// assembly template.
///
/// Template strings contain AT&T syntax assembly with operand placeholders:
/// - `%0`, `%1`, ... — positional operand references
/// - `%[name]` — named operand references
/// - `%%` — literal `%` character
///
/// The template may also contain `.pushsection` / `.popsection` assembler
/// directives, commonly used by the Linux kernel for alternative instruction
/// patching and static key implementations.
///
/// Multiple adjacent string literals are collected as separate fragments
/// (matching C's string literal concatenation semantics) and stored as
/// `Vec<Vec<u8>>` to preserve PUA-encoded non-UTF-8 bytes with byte-exact
/// fidelity.
fn parse_asm_template(parser: &mut Parser<'_>) -> Result<Vec<Vec<u8>>, ParseError> {
    let mut fragments: Vec<Vec<u8>> = Vec::new();

    // Collect all adjacent string literal tokens as template fragments.
    while let TokenKind::StringLiteral { value, .. } = &parser.current().kind {
        fragments.push(value.clone());
        parser.advance();
    }

    if fragments.is_empty() {
        let span = parser.current().span;
        let msg = "expected assembly template string".to_string();
        parser.diagnostics.error(span, &msg);
        return Err(ParseError {
            span,
            message: msg,
            expected: Some("string literal".to_string()),
        });
    }

    Ok(fragments)
}

/// Internal operand list parser with output/input constraint validation.
///
/// # Arguments
///
/// * `parser` — Mutable reference to the parser state.
/// * `is_output` — When `true`, validates that constraint strings begin with
///   `=` or `+` (output constraint prefix). When `false`, validates that
///   constraints do NOT begin with `=` or `+` (input constraint).
fn parse_asm_operands_internal(
    parser: &mut Parser<'_>,
    is_output: bool,
) -> Result<Vec<AsmOperand>, ParseError> {
    let mut operands: Vec<AsmOperand> = Vec::new();

    loop {
        let operand = parse_single_operand(parser, is_output)?;
        operands.push(operand);

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(operands)
}

/// Parses a single assembly operand: `[name] "constraint" (expression)`.
///
/// # Named Operand Syntax
///
/// The optional `[name]` prefix assigns a symbolic name to the operand.
/// This name can then be referenced in the template string via `%[name]`
/// instead of the positional `%N` form, improving readability:
///
/// ```c
/// asm("mov %[input], %[output]"
///     : [output] "=r" (out)
///     : [input] "r" (in));
/// ```
///
/// The identifier between brackets is interned via the parser's `Interner`
/// and stored as `Option<Symbol>` in `AsmOperand.name`.
///
/// # Constraint Validation
///
/// Basic constraint format validation is performed here:
/// - Output operands (`is_output == true`): constraint must start with `=`
///   (write-only) or `+` (read-write). An optional `&` marks early-clobber.
/// - Input operands (`is_output == false`): constraint must NOT start with
///   `=` or `+`.
///
/// Full semantic constraint validation (register class compatibility,
/// matching constraint indices, architecture-specific constraints) is
/// deferred to `sema/attribute_handler.rs` and `ir/lowering/asm_lowering.rs`.
///
/// # Expression Parsing
///
/// The operand expression is parsed between parentheses using
/// `expressions::parse_expression`. For output operands, this should be an
/// lvalue expression (variable, array element, struct member) that receives
/// the assembly output. For input operands, any rvalue expression is valid.
fn parse_single_operand(
    parser: &mut Parser<'_>,
    is_output: bool,
) -> Result<AsmOperand, ParseError> {
    let operand_start = parser.current().span;

    // Optional symbolic name: [name]
    let name: Option<Symbol> = if parser.check(TokenKind::LeftBracket) {
        parser.advance(); // consume `[`

        let sym = match &parser.current().kind {
            TokenKind::Identifier(sym) => {
                let s = *sym;
                parser.advance();
                s
            }
            _ => {
                let span = parser.current().span;
                let msg = "expected operand name identifier after '['".to_string();
                parser.diagnostics.error(span, &msg);
                return Err(ParseError {
                    span,
                    message: msg,
                    expected: Some("identifier".to_string()),
                });
            }
        };

        parser.expect(TokenKind::RightBracket)?;
        Some(sym)
    } else {
        None
    };

    // Constraint string literal — e.g., "=r", "+m", "r", "i", "0"
    let (constraint, constraint_span) = match &parser.current().kind {
        TokenKind::StringLiteral { value, .. } => {
            let span = parser.current().span;
            let s = String::from_utf8_lossy(value).into_owned();
            parser.advance();
            (s, span)
        }
        _ => {
            let span = parser.current().span;
            let msg = "expected constraint string for asm operand".to_string();
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("string literal".to_string()),
            });
        }
    };

    // Basic constraint validation (§9 from key changes).
    validate_constraint(parser, &constraint, constraint_span, is_output)?;

    // Expression in parentheses: (expression)
    parser.expect(TokenKind::LeftParen)?;
    let expression: Expression = super::expressions::parse_expression(parser)?;
    let rparen_span = parser.expect(TokenKind::RightParen)?;

    let operand_span = Span::merge(operand_start, rparen_span);

    Ok(AsmOperand {
        name,
        constraint,
        expression: Box::new(expression),
        span: operand_span,
    })
}

/// Performs basic constraint string validation.
///
/// Output constraints must begin with `=` (write-only) or `+` (read-write).
/// Input constraints must NOT begin with `=` or `+`.
///
/// This is intentionally a shallow check — full semantic validation of
/// constraint characters against the target architecture's register classes,
/// matching constraint index bounds, and constraint/expression compatibility
/// is deferred to the semantic analyzer and IR lowering passes.
///
/// # Arguments
///
/// * `parser` — Used for diagnostic emission.
/// * `constraint` — The constraint string to validate.
/// * `constraint_span` — Source span of the constraint string literal.
/// * `is_output` — Whether this constraint belongs to an output operand.
///
/// # Errors
///
/// Returns `ParseError` if the constraint format is invalid:
/// - Empty constraint string.
/// - Output constraint missing `=` or `+` prefix.
/// - Input constraint incorrectly prefixed with `=` or `+`.
fn validate_constraint(
    parser: &mut Parser<'_>,
    constraint: &str,
    constraint_span: Span,
    is_output: bool,
) -> Result<(), ParseError> {
    if constraint.is_empty() {
        let msg = "empty constraint string in asm operand".to_string();
        parser.diagnostics.error(constraint_span, &msg);
        return Err(ParseError {
            span: constraint_span,
            message: msg,
            expected: Some("non-empty constraint string".to_string()),
        });
    }

    let first_char = constraint.as_bytes()[0];
    let has_output_prefix = first_char == b'=' || first_char == b'+';

    if is_output && !has_output_prefix {
        // Create a span for the precise error location using Span::new
        let err_span = Span::new(
            constraint_span.file_id,
            constraint_span.start,
            constraint_span.end,
        );
        let msg = format!(
            "output constraint '{}' must begin with '=' or '+'",
            constraint
        );
        parser.diagnostics.error(err_span, &msg);
        return Err(ParseError {
            span: err_span,
            message: msg,
            expected: Some("'=' or '+' prefix".to_string()),
        });
    }

    if !is_output && has_output_prefix {
        let msg = format!(
            "input constraint '{}' must not begin with '=' or '+'",
            constraint
        );
        parser.diagnostics.warning(constraint_span, &msg);
        // This is a warning rather than a hard error, as some GCC versions
        // are lenient with this. The semantic analyzer performs the final check.
    }

    Ok(())
}
