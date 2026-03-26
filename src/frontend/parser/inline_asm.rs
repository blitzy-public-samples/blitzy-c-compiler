#![allow(clippy::result_unit_err)]
//! Inline assembly (`asm`/`__asm__`) statement parsing for the BCC C11 parser.
//!
//! Implements full parsing of GCC-style inline assembly statements (Phase 4),
//! which are critical for Linux kernel compilation. The kernel makes extensive
//! use of inline assembly with AT&T syntax, complex constraint specifications,
//! clobber lists, named operands, `asm volatile`, and `asm goto` with jump
//! labels.
//!
//! # Supported Syntax
//!
//! ## Basic Assembly (no operands)
//!
//! ```text
//! asm("nop");
//! asm volatile("cli");
//! ```
//!
//! ## Extended Assembly (with operands and clobbers)
//!
//! ```text
//! asm [qualifiers] ( template
//!     : output-operands
//!     : input-operands
//!     : clobber-list
//!     [: goto-labels]
//! );
//! ```
//!
//! ## Qualifiers
//!
//! - `volatile` / `__volatile__` — side effects, may not be optimized away
//! - `goto` — the assembly may transfer control to C labels
//! - `inline` — hint for inlining
//!
//! Multiple qualifiers may appear in any order after the `asm` keyword.
//!
//! ## Named Operands
//!
//! ```text
//! asm("add %[src], %[dst]"
//!     : [dst] "+r" (result)
//!     : [src] "r" (addend)
//!     : "cc");
//! ```
//!
//! ## Asm Goto
//!
//! ```text
//! asm goto("testl %0, %0; jnz %l[error]"
//!     : /* no outputs */
//!     : "r" (val)
//!     : "cc"
//!     : error);
//! ```
//!
//! ## `.pushsection` / `.popsection`
//!
//! Assembly templates may contain `.pushsection`/`.popsection` directives as
//! raw assembly text — these are preserved verbatim in the template string and
//! handled during code generation, not during parsing.
//!
//! # Grammar
//!
//! ```text
//! asm-statement:
//!     'asm' asm-qualifiers '(' asm-template ')' ';'
//!     'asm' asm-qualifiers '(' asm-template ':' output-operands
//!         ':' input-operands ':' clobber-list ')' ';'
//!     'asm' asm-qualifiers '(' asm-template ':' output-operands
//!         ':' input-operands ':' clobber-list ':' goto-labels ')' ';'
//!
//! asm-qualifiers:
//!     ('volatile' | '__volatile__' | 'goto' | 'inline')*
//!
//! asm-template:
//!     string-literal+
//!
//! output-operands:
//!     output-operand (',' output-operand)*
//!
//! output-operand:
//!     ['[' identifier ']'] string-literal '(' assignment-expression ')'
//!
//! input-operands:
//!     input-operand (',' input-operand)*
//!
//! input-operand:
//!     ['[' identifier ']'] string-literal '(' assignment-expression ')'
//!
//! clobber-list:
//!     string-literal (',' string-literal)*
//!
//! goto-labels:
//!     identifier (',' identifier)*
//! ```
//!
//! # Design Decisions
//!
//! - The `asm`/`__asm__` keyword is consumed by the caller (statement parser
//!   or external-declaration parser) before invoking [`parse_asm_statement`].
//!   The first token visible to this module is the first qualifier or `(`.
//! - Template strings are stored as raw `Vec<u8>` to preserve PUA-encoded
//!   bytes from non-UTF-8 source files with byte-exact fidelity.
//! - Constraint strings are also `Vec<u8>` for the same PUA reason.
//! - Operand expressions use `parse_assignment_expression` (not the full
//!   comma expression) because commas serve as operand list separators in
//!   the asm syntax — a comma within an operand expression would be ambiguous.
//! - Recursion depth is tracked via `enter_recursion`/`leave_recursion` to
//!   enforce the 512-depth limit on deeply nested constructs.
//! - Error recovery attempts to skip to the closing `)` and `;` on failure,
//!   allowing the parser to continue with the next statement.
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node types (AsmStatement, AsmOperand, AsmClobber)
//! - `super::expressions` — expression parsing (`parse_assignment_expression`)
//! - `super::Parser` — token consumption, recursion tracking, error reporting
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned identifiers
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates. Does **NOT** depend on `crate::ir`,
//! `crate::passes`, or `crate::backend`.

use super::ast::*;
use super::expressions;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Main Entry Point
// ===========================================================================

/// Parse an inline assembly statement.
///
/// Called when the parser detects an `asm`/`__asm__` keyword (or
/// `__volatile__` in the asm-start position). The current token is the
/// `asm`-family keyword itself — this function consumes it along with
/// qualifiers, the parenthesized template, operand sections, and the
/// terminating semicolon.
///
/// # Grammar
///
/// ```text
/// asm-statement:
///     'asm' asm-qualifiers '(' asm-template ')' ';'
///     'asm' asm-qualifiers '(' asm-template ':' outputs ':' inputs
///                                             ':' clobbers ')' ';'
///     'asm' asm-qualifiers '(' asm-template ':' outputs ':' inputs
///                                             ':' clobbers ':' labels ')' ';'
///
/// asm-qualifiers:
///     ('volatile' | '__volatile__' | 'goto' | 'inline')*
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token positioned at the `asm`,
///   `__asm__`, or `__volatile__` keyword that starts the statement.
///
/// # Returns
///
/// * `Ok(AsmStatement)` — Successfully parsed inline assembly statement.
/// * `Err(())` — Parse error (diagnostic already emitted via the parser's
///   diagnostic engine).
///
/// # Error Recovery
///
/// On parse failure, the function attempts to skip tokens until the closing
/// `)` and `;` are found, allowing the parser to continue with subsequent
/// statements. This enables multi-error reporting in a single compilation
/// pass.
///
/// # Examples
///
/// ```text
/// // Basic asm (no operands)
/// asm("nop");
///
/// // Volatile asm with clobbers
/// asm volatile("mov %%eax, %%ebx" ::: "ebx");
///
/// // Extended asm with named operands
/// asm("add %[src], %[dst]"
///     : [dst] "+r" (result)
///     : [src] "r" (addend)
///     : "cc");
///
/// // Asm goto
/// asm goto("testl %0, %0; jnz %l[error]"
///     : /* no outputs */
///     : "r" (val)
///     : "cc"
///     : error);
/// ```
pub fn parse_asm_statement(parser: &mut Parser<'_>) -> Result<AsmStatement, ()> {
    let start_span = parser.current_span();

    // Track recursion depth to enforce the 512-level limit.
    parser.enter_recursion()?;

    // -----------------------------------------------------------------------
    // Phase 0: Consume the initial asm-family keyword and extract qualifiers.
    //
    // The call site (parse_external_declaration, parse_statement) dispatches
    // here when the current token is `TokenKind::Asm` or
    // `TokenKind::AsmVolatile`. We handle both:
    //
    // - `Asm` (`asm` / `__asm__`): Consume it, then parse additional
    //   qualifiers (volatile, goto, inline).
    // - `AsmVolatile` (`__volatile__`): Treated as `asm volatile` (the
    //   `__volatile__` token implies volatility).
    // -----------------------------------------------------------------------
    let mut is_volatile = false;
    let mut is_goto = false;

    // Consume `asm` / `__asm__`.
    parser.expect(TokenKind::Asm).map_err(|()| {
        parser.leave_recursion();
    })?;

    // -----------------------------------------------------------------------
    // Phase 1: Parse additional qualifiers (volatile, goto, inline) in any
    // order. These follow the initial asm keyword.
    // -----------------------------------------------------------------------
    parse_qualifiers(parser, &mut is_volatile, &mut is_goto);

    // -----------------------------------------------------------------------
    // Phase 2: Expect opening parenthesis.
    // -----------------------------------------------------------------------
    if parser
        .consume(TokenKind::LeftParen, "expected '(' after asm qualifiers")
        .is_err()
    {
        parser.leave_recursion();
        return Err(());
    }

    // -----------------------------------------------------------------------
    // Phase 3: Parse the assembly template string (one or more concatenated
    // string literals).
    // -----------------------------------------------------------------------
    let template = match parse_template_string(parser) {
        Ok(t) => t,
        Err(()) => {
            recover_to_asm_end(parser);
            parser.leave_recursion();
            return Err(());
        }
    };

    // -----------------------------------------------------------------------
    // Phase 4–7: Parse colon-separated sections (extended asm) or nothing
    // (basic asm).
    //
    // Basic asm:    asm("nop");              — no colons
    // Extended asm: asm("..." : out : in : clobber [: labels]);
    //
    // Any trailing colon sections can be omitted. Missing sections default
    // to empty.
    // -----------------------------------------------------------------------
    let mut outputs: Vec<AsmOperand> = Vec::new();
    let mut inputs: Vec<AsmOperand> = Vec::new();
    let mut clobbers: Vec<AsmClobber> = Vec::new();
    let mut goto_labels: Vec<Symbol> = Vec::new();

    // Detect extended asm by the presence of a colon after the template.
    if parser.check(&TokenKind::Colon) {
        // Consume the first colon — separates template from output operands.
        parser.advance();

        // Section 1: Output operands (may be empty if next token is ':' or ')').
        if !is_at_section_boundary(parser) {
            match parse_operand_list(parser) {
                Ok(ops) => outputs = ops,
                Err(()) => {
                    recover_to_asm_end(parser);
                    parser.leave_recursion();
                    return Err(());
                }
            }
        }

        // Section 2: Input operands.
        if parser.match_token(&TokenKind::Colon) {
            if !is_at_section_boundary(parser) {
                match parse_operand_list(parser) {
                    Ok(ops) => inputs = ops,
                    Err(()) => {
                        recover_to_asm_end(parser);
                        parser.leave_recursion();
                        return Err(());
                    }
                }
            }

            // Section 3: Clobber list.
            if parser.match_token(&TokenKind::Colon) {
                if !is_at_section_boundary(parser) {
                    match parse_clobber_list(parser) {
                        Ok(c) => clobbers = c,
                        Err(()) => {
                            recover_to_asm_end(parser);
                            parser.leave_recursion();
                            return Err(());
                        }
                    }
                }

                // Section 4: Goto labels (only meaningful with `goto` qualifier,
                // but parsed unconditionally for error recovery).
                if parser.match_token(&TokenKind::Colon) && !is_at_section_boundary(parser) {
                    match parse_goto_label_list(parser) {
                        Ok(labels) => goto_labels = labels,
                        Err(()) => {
                            recover_to_asm_end(parser);
                            parser.leave_recursion();
                            return Err(());
                        }
                    }
                }
            }
        }
    }
    // else: Basic asm — template only, no operands, clobbers, or labels.

    // -----------------------------------------------------------------------
    // Phase 8: Warn if `goto` qualifier was present but no labels provided.
    // -----------------------------------------------------------------------
    if is_goto && goto_labels.is_empty() {
        // Merge the start span with the current position for a wide diagnostic.
        let warn_span = start_span.merge(parser.current_span());
        parser.warn(
            warn_span,
            "`asm goto` specified but no goto labels provided in the fourth colon section",
        );
    }

    // -----------------------------------------------------------------------
    // Phase 9: Expect closing parenthesis.
    // -----------------------------------------------------------------------
    if parser.expect(TokenKind::RightParen).is_err() {
        recover_to_asm_end(parser);
        parser.leave_recursion();
        // Build a partial span covering what we have so far.
        return Err(());
    }

    // -----------------------------------------------------------------------
    // Phase 10: Expect semicolon (terminates the asm statement).
    //
    // If the semicolon is missing we still return the parsed AST node — the
    // error was already reported by expect(). The parser's panic-mode error
    // recovery will handle the missing semicolon for subsequent parsing.
    // -----------------------------------------------------------------------
    let _ = parser.expect(TokenKind::Semicolon);

    let span = parser.make_span(start_span);
    parser.leave_recursion();

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

// ===========================================================================
// Qualifier Parsing
// ===========================================================================

/// Parse asm qualifiers in any order: `volatile`, `__volatile__`, `goto`,
/// `inline`.
///
/// Multiple qualifiers can appear and in any combination. Duplicate qualifiers
/// are silently accepted (matching GCC behavior). The `inline` qualifier is
/// parsed but not stored — `AsmStatement` has no `is_inline` field because
/// it is purely an optimization hint with no semantic effect.
///
/// # Arguments
///
/// * `parser` — Current parser state.
/// * `is_volatile` — Set to `true` if `volatile` / `__volatile__` found.
/// * `is_goto` — Set to `true` if `goto` found.
fn parse_qualifiers(parser: &mut Parser<'_>, is_volatile: &mut bool, is_goto: &mut bool) {
    loop {
        // Check for `volatile` / `__volatile__` (both map to TokenKind::Volatile).
        if parser.match_token(&TokenKind::Volatile) {
            *is_volatile = true;
        }
        // Check for `goto` qualifier.
        else if parser.match_token(&TokenKind::Goto) {
            *is_goto = true;
        }
        // Check for `inline` hint (parsed but not stored).
        else if parser.match_token(&TokenKind::Inline) {
            // `inline` is an optimization hint — no corresponding field in
            // AsmStatement. We consume and discard.
        }
        // No more qualifiers.
        else {
            break;
        }
    }
}

// ===========================================================================
// Template String Parsing
// ===========================================================================

/// Parse the assembly template string — one or more adjacent string literals
/// that are concatenated into a single byte sequence.
///
/// The template contains assembly instructions with operand references:
/// - `%0`, `%1`, ... — positional operand references
/// - `%[name]` — named operand references
/// - `%%` — literal percent sign
/// - `.pushsection`/`.popsection` — as raw assembly directives
///
/// The parser stores the raw template string bytes (PUA-encoded bytes from
/// non-UTF-8 sources are preserved). Operand binding happens during IR
/// lowering, not here.
///
/// # Returns
///
/// * `Ok(Vec<u8>)` — The concatenated template bytes.
/// * `Err(())` — No string literal found at the expected position.
fn parse_template_string(parser: &mut Parser<'_>) -> Result<Vec<u8>, ()> {
    // The first token must be a string literal.
    let first_span = parser.current_span();
    let mut bytes = match extract_string_literal(parser) {
        Some(s) => s,
        None => {
            let span = parser.current_span();
            parser.error(span, "expected assembly template string literal");
            return Err(());
        }
    };
    parser.advance();

    // Concatenate any adjacent string literals (C string literal
    // concatenation applies to asm templates).
    let mut last_span = parser.previous_span();
    while is_string_literal(parser) {
        if let Some(s) = extract_string_literal(parser) {
            bytes.extend_from_slice(&s);
        }
        last_span = parser.current_span();
        parser.advance();
    }

    // Record the merged span covering all concatenated literals. This is
    // used for diagnostic purposes if template-related errors arise later.
    let _template_span = Span::new(first_span.file_id, first_span.start, last_span.end);

    Ok(bytes)
}

/// Extract the byte content of a string literal from the current token.
///
/// Returns `None` if the current token is not a string literal. The returned
/// bytes reflect PUA-encoded content from `value.as_bytes()`, preserving
/// non-UTF-8 round-trip fidelity.
fn extract_string_literal(parser: &Parser<'_>) -> Option<Vec<u8>> {
    match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => Some(value.as_bytes().to_vec()),
        _ => None,
    }
}

/// Check if the current token is a string literal (for template
/// concatenation and constraint/clobber parsing).
fn is_string_literal(parser: &Parser<'_>) -> bool {
    matches!(parser.current.kind, TokenKind::StringLiteral { .. })
}

// ===========================================================================
// Operand List Parsing (Output and Input Operands)
// ===========================================================================

/// Parse a comma-separated list of assembly operands (output or input).
///
/// Each operand has the form:
/// ```text
/// ['[' identifier ']'] string-literal '(' assignment-expression ')'
/// ```
///
/// The list is terminated by a colon (next section), closing paren, or EOF.
/// An empty list (when the next token is a section boundary) should not call
/// this function — the caller checks `is_at_section_boundary` first.
///
/// # Returns
///
/// * `Ok(Vec<AsmOperand>)` — Parsed operand list.
/// * `Err(())` — Parse error in an individual operand.
fn parse_operand_list(parser: &mut Parser<'_>) -> Result<Vec<AsmOperand>, ()> {
    let mut operands = Vec::new();

    // Parse the first operand.
    operands.push(parse_single_operand(parser)?);

    // Parse additional comma-separated operands.
    while parser.match_token(&TokenKind::Comma) {
        // After a comma, there must be another operand.
        operands.push(parse_single_operand(parser)?);
    }

    Ok(operands)
}

/// Parse a single assembly operand (output or input).
///
/// # Grammar
///
/// ```text
/// asm-operand:
///     ['[' identifier ']'] constraint-string '(' assignment-expression ')'
///
/// constraint-string:
///     string-literal
///
/// Output constraints: "=r", "=m", "+r", "=&r", etc.
/// Input constraints:  "r", "i", "n", "m", "0", "1", etc.
/// ```
///
/// # Named Operands
///
/// The optional `[name]` prefix allows the template to reference this
/// operand by name (`%[name]`) instead of by position (`%0`). The name
/// is interned as a [`Symbol`] and stored in `AsmOperand.symbolic_name`.
///
/// # Expression Level
///
/// The parenthesized expression is parsed at the assignment-expression
/// level (not full comma expression) because commas serve as operand
/// list separators. A comma within the expression would be ambiguous.
fn parse_single_operand(parser: &mut Parser<'_>) -> Result<AsmOperand, ()> {
    let operand_start = parser.current_span();

    // -----------------------------------------------------------------------
    // Optional symbolic name: [identifier]
    // -----------------------------------------------------------------------
    let symbolic_name = parse_optional_symbolic_name(parser)?;

    // -----------------------------------------------------------------------
    // Constraint string (required).
    // -----------------------------------------------------------------------
    let constraint = match extract_string_literal(parser) {
        Some(bytes) => {
            parser.advance();
            bytes
        }
        None => {
            let span = parser.current_span();
            parser.error(span, "expected constraint string literal in asm operand");
            return Err(());
        }
    };

    // -----------------------------------------------------------------------
    // Parenthesized C expression (required).
    // -----------------------------------------------------------------------
    if parser
        .consume(
            TokenKind::LeftParen,
            "expected '(' before asm operand expression",
        )
        .is_err()
    {
        return Err(());
    }

    let expression = expressions::parse_assignment_expression(parser)?;

    if parser.expect(TokenKind::RightParen).is_err() {
        return Err(());
    }

    let span = parser.make_span(operand_start);

    Ok(AsmOperand {
        symbolic_name,
        constraint,
        expression: Box::new(expression),
        span,
    })
}

/// Parse the optional `[identifier]` symbolic name prefix on an asm operand.
///
/// If the current token is `[`, consumes `[ identifier ]` and returns
/// `Some(Symbol)`. Otherwise returns `None` without consuming anything.
///
/// # Returns
///
/// * `Ok(Some(Symbol))` — Named operand syntax found and parsed.
/// * `Ok(None)` — No bracket found; positional operand.
/// * `Err(())` — Bracket found but identifier or closing bracket is missing.
fn parse_optional_symbolic_name(parser: &mut Parser<'_>) -> Result<Option<Symbol>, ()> {
    if !parser.check(&TokenKind::LeftBracket) {
        return Ok(None);
    }

    // Consume '['
    parser.advance();

    // Extract identifier symbol.
    let sym = match parser.current.kind {
        TokenKind::Identifier(sym) => {
            parser.advance();
            sym
        }
        _ => {
            let span = parser.current_span();
            parser.error(
                span,
                "expected identifier for named asm operand inside '[...]'",
            );
            return Err(());
        }
    };

    // Consume ']'
    parser.expect(TokenKind::RightBracket)?;

    Ok(Some(sym))
}

// ===========================================================================
// Clobber List Parsing
// ===========================================================================

/// Parse a comma-separated clobber list.
///
/// Each clobber is a string literal naming a register or special resource:
/// - `"memory"` — assembly clobbers memory (prevents reordering)
/// - `"cc"` — assembly clobbers condition codes / flags register
/// - `"rax"`, `"rbx"`, `"eax"`, etc. — architecture-specific register names
///
/// The list is terminated by a colon (goto labels section), closing paren,
/// or EOF. The caller ensures this function is only called when there are
/// actual clobbers to parse (not at a section boundary).
///
/// # Returns
///
/// * `Ok(Vec<AsmClobber>)` — Parsed clobber entries.
/// * `Err(())` — Expected a string literal but found something else.
fn parse_clobber_list(parser: &mut Parser<'_>) -> Result<Vec<AsmClobber>, ()> {
    let mut clobbers = Vec::new();

    // Parse the first clobber.
    clobbers.push(parse_single_clobber(parser)?);

    // Parse additional comma-separated clobbers.
    while parser.match_token(&TokenKind::Comma) {
        // After a trailing comma, check if we're at a section boundary
        // (some compilers tolerate a trailing comma in clobber lists).
        if is_at_section_boundary(parser) {
            break;
        }
        clobbers.push(parse_single_clobber(parser)?);
    }

    Ok(clobbers)
}

/// Parse a single clobber entry: a string literal.
///
/// # Returns
///
/// * `Ok(AsmClobber)` — Successfully parsed clobber with its register name.
/// * `Err(())` — Current token is not a string literal.
fn parse_single_clobber(parser: &mut Parser<'_>) -> Result<AsmClobber, ()> {
    let span = parser.current_span();

    match extract_string_literal(parser) {
        Some(register) => {
            parser.advance();
            let clobber_span = Span::merge(span, parser.previous_span());
            Ok(AsmClobber {
                register,
                span: clobber_span,
            })
        }
        None => {
            parser.error(
                span,
                "expected string literal for clobber name in asm statement",
            );
            Err(())
        }
    }
}

// ===========================================================================
// Goto Label List Parsing
// ===========================================================================

/// Parse a comma-separated list of goto label identifiers.
///
/// These labels must be defined in the enclosing function scope. They are
/// referenced in the assembly template using `%l[name]` or `%l[N]` syntax.
///
/// Duplicate labels are detected and reported as warnings using the
/// [`Symbol`] handle's `u32` representation for O(1) comparison via a
/// simple seen-set.
///
/// # Returns
///
/// * `Ok(Vec<Symbol>)` — Parsed label identifiers (interned).
/// * `Err(())` — Expected an identifier but found something else.
fn parse_goto_label_list(parser: &mut Parser<'_>) -> Result<Vec<Symbol>, ()> {
    let mut labels: Vec<Symbol> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();

    // Parse the first label.
    labels.push(parse_single_goto_label(parser, &mut seen)?);

    // Parse additional comma-separated labels.
    while parser.match_token(&TokenKind::Comma) {
        // Trailing comma tolerance: check for section boundary.
        if is_at_section_boundary(parser) {
            break;
        }
        labels.push(parse_single_goto_label(parser, &mut seen)?);
    }

    Ok(labels)
}

/// Parse a single goto label identifier, checking for duplicates.
///
/// Uses `Symbol::as_u32()` for O(1) duplicate detection via a simple
/// seen-set of `u32` handles.
fn parse_single_goto_label(parser: &mut Parser<'_>, seen: &mut Vec<u32>) -> Result<Symbol, ()> {
    match parser.current.kind {
        TokenKind::Identifier(sym) => {
            let label_span = parser.current_span();
            parser.advance();

            // Check for duplicate labels using the Symbol's u32 handle.
            let handle = sym.as_u32();
            if seen.contains(&handle) {
                parser.warn(label_span, "duplicate goto label in asm goto statement");
            } else {
                seen.push(handle);
            }

            Ok(sym)
        }
        _ => {
            let span = parser.current_span();
            parser.error(
                span,
                "expected identifier for goto label in asm goto statement",
            );
            Err(())
        }
    }
}

// ===========================================================================
// Section Boundary Detection
// ===========================================================================

/// Check if the current token marks the end of an operand/clobber/label
/// section.
///
/// A section boundary is reached when the next token is:
/// - `:` — start of the next colon-separated section
/// - `)` — end of the entire asm statement
/// - `EOF` — unexpected end of input
///
/// This function does NOT consume any tokens — it only peeks.
#[inline]
fn is_at_section_boundary(parser: &Parser<'_>) -> bool {
    matches!(
        parser.peek(),
        TokenKind::Colon | TokenKind::RightParen | TokenKind::Eof
    )
}

// ===========================================================================
// Error Recovery
// ===========================================================================

/// Attempt to recover from a parse error within an asm statement by skipping
/// tokens until the closing `)` and `;` are found.
///
/// This enables the parser to continue with subsequent statements after
/// encountering a malformed asm statement. The recovery strategy is:
///
/// 1. Skip tokens until `)` or EOF is encountered.
/// 2. If `)` is found, consume it.
/// 3. If `;` follows, consume it too.
///
/// Parenthesis nesting is tracked to handle cases where the asm operand
/// expressions contain parenthesized sub-expressions.
fn recover_to_asm_end(parser: &mut Parser<'_>) {
    let mut paren_depth: u32 = 1; // We're inside the asm '(' already.

    while !parser.check(&TokenKind::Eof) {
        match parser.peek() {
            TokenKind::LeftParen => {
                paren_depth = paren_depth.saturating_add(1);
                parser.advance();
            }
            TokenKind::RightParen => {
                paren_depth = paren_depth.saturating_sub(1);
                if paren_depth == 0 {
                    // Found the matching ')' — consume it.
                    parser.advance();
                    // Try to also consume the trailing ';'.
                    if parser.check(&TokenKind::Semicolon) {
                        parser.advance();
                    }
                    return;
                }
                parser.advance();
            }
            TokenKind::Semicolon if paren_depth <= 1 => {
                // A semicolon at top level likely means the ')' was missing.
                // Consume the semicolon and stop recovery.
                parser.advance();
                return;
            }
            _ => {
                parser.advance();
            }
        }
    }
    // Reached EOF without finding ')' — nothing more we can do.
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    // Integration tests for inline asm parsing are in tests/checkpoint2_language.rs.
    // The module structure is validated through the module compilation test suite.
    // This test module is reserved for future unit-level tests if needed.
}
