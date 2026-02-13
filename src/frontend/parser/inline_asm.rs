//! Inline assembly parsing module for the BCC C11 parser.
//!
//! Parses `asm` / `__asm__` statements with the full GCC extended syntax:
//! - `asm volatile` / `asm goto`
//! - Template strings (multiple fragments concatenated)
//! - Output operands with constraints
//! - Input operands with constraints
//! - Clobber lists
//! - Named operands (`[name] "constraint" (expr)`)
//! - Goto labels for `asm goto`
//! - `.pushsection` / `.popsection` directives in templates
//!
//! # Grammar
//!
//! ```text
//! asm-statement:
//!     'asm' ['volatile'] ['goto'] '(' asm-string-literal
//!         [':' [asm-operand-list]]              // outputs
//!         [':' [asm-operand-list]]              // inputs
//!         [':' [asm-clobber-list]]              // clobbers
//!         [':' [asm-goto-labels]]               // goto labels
//!     ')' ';'
//! ```

use super::ast::{AsmOperand, AsmStatement, Span};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public API
// ===========================================================================

/// Parses a complete inline assembly statement.
///
/// The caller has already verified that the current token is `asm` or
/// `__asm__`.
pub fn parse_asm_statement(parser: &mut Parser<'_>) -> Result<AsmStatement, ParseError> {
    let start = parser.advance(); // consume `asm` / `__asm__`

    // Optional qualifiers: volatile, goto
    let mut is_volatile = false;
    let mut is_goto = false;

    loop {
        if parser.check(TokenKind::Volatile) {
            is_volatile = true;
            parser.advance();
        } else if parser.check(TokenKind::Goto) {
            is_goto = true;
            parser.advance();
        } else {
            break;
        }
    }

    parser.expect(TokenKind::LeftParen)?;

    // Parse template string(s) — one or more adjacent string literals
    let template = parse_asm_template(parser)?;

    let mut outputs: Vec<AsmOperand> = Vec::new();
    let mut inputs: Vec<AsmOperand> = Vec::new();
    let mut clobbers: Vec<String> = Vec::new();
    let mut goto_labels: Vec<Symbol> = Vec::new();

    // First colon: output operands
    if parser.eat(TokenKind::Colon) {
        if !is_colon_or_paren(parser) {
            outputs = parse_asm_operand_list(parser)?;
        }

        // Second colon: input operands
        if parser.eat(TokenKind::Colon) {
            if !is_colon_or_paren(parser) {
                inputs = parse_asm_operand_list(parser)?;
            }

            // Third colon: clobber list
            if parser.eat(TokenKind::Colon) {
                if !is_colon_or_paren(parser) {
                    clobbers = parse_clobber_list(parser)?;
                }

                // Fourth colon: goto labels
                if is_goto && parser.eat(TokenKind::Colon) && !parser.check(TokenKind::RightParen) {
                    goto_labels = parse_goto_labels(parser)?;
                }
            }
        }
    }

    parser.expect(TokenKind::RightParen)?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

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
// Internal helpers
// ===========================================================================

/// Returns `true` if the current token is `:` or `)` — used to detect
/// empty operand sections.
fn is_colon_or_paren(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Colon | TokenKind::RightParen
    )
}

/// Parses one or more adjacent string literal fragments that form the
/// assembly template.
fn parse_asm_template(parser: &mut Parser<'_>) -> Result<Vec<Vec<u8>>, ParseError> {
    let mut fragments: Vec<Vec<u8>> = Vec::new();

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

/// Parses a comma-separated list of assembly operands.
///
/// ```text
/// asm-operand: [name] "constraint" (expression)
/// name: '[' identifier ']'
/// ```
fn parse_asm_operand_list(parser: &mut Parser<'_>) -> Result<Vec<AsmOperand>, ParseError> {
    let mut operands = Vec::new();

    loop {
        let op = parse_asm_operand(parser)?;
        operands.push(op);

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(operands)
}

/// Parses a single asm operand: `[name] "constraint" (expression)`.
fn parse_asm_operand(parser: &mut Parser<'_>) -> Result<AsmOperand, ParseError> {
    let start = parser.current().span;

    // Optional symbolic name: [name]
    let name = if parser.check(TokenKind::LeftBracket) {
        parser.advance(); // consume `[`
        let sym = match &parser.current().kind {
            TokenKind::Identifier(sym) => {
                let s = *sym;
                parser.advance();
                s
            }
            _ => {
                let span = parser.current().span;
                let msg = "expected operand name identifier".to_string();
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

    // Constraint string literal
    let constraint = match &parser.current().kind {
        TokenKind::StringLiteral { value, .. } => {
            let s = String::from_utf8_lossy(value).into_owned();
            parser.advance();
            s
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

    // Expression in parentheses: (expr)
    parser.expect(TokenKind::LeftParen)?;
    let expression = super::expressions::parse_expression(parser)?;
    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);

    Ok(AsmOperand {
        name,
        constraint,
        expression: Box::new(expression),
        span,
    })
}

/// Parses a comma-separated list of clobber string literals.
fn parse_clobber_list(parser: &mut Parser<'_>) -> Result<Vec<String>, ParseError> {
    let mut clobbers = Vec::new();

    loop {
        match &parser.current().kind {
            TokenKind::StringLiteral { value, .. } => {
                clobbers.push(String::from_utf8_lossy(value).into_owned());
                parser.advance();
            }
            _ => {
                let span = parser.current().span;
                let msg = "expected clobber string literal".to_string();
                parser.diagnostics.error(span, &msg);
                return Err(ParseError {
                    span,
                    message: msg,
                    expected: Some("string literal".to_string()),
                });
            }
        }

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(clobbers)
}

/// Parses a comma-separated list of goto label identifiers.
fn parse_goto_labels(parser: &mut Parser<'_>) -> Result<Vec<Symbol>, ParseError> {
    let mut labels = Vec::new();

    loop {
        match &parser.current().kind {
            TokenKind::Identifier(sym) => {
                labels.push(*sym);
                parser.advance();
            }
            _ => {
                let span = parser.current().span;
                let msg = "expected goto label identifier".to_string();
                parser.diagnostics.error(span, &msg);
                return Err(ParseError {
                    span,
                    message: msg,
                    expected: Some("identifier".to_string()),
                });
            }
        }

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    Ok(labels)
}
