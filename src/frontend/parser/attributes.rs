//! `__attribute__((...))` parsing for the BCC C11 parser.
//!
//! Parses the GCC `__attribute__` syntax and produces [`Attribute`] AST nodes.
//! The semantic validation of individual attributes (e.g., checking that
//! `aligned(N)` is a power of two) is performed later in the semantic
//! analysis phase (`src/frontend/sema/attribute_handler.rs`).
//!
//! # Syntax
//!
//! ```text
//! __attribute__ (( attribute-list ))
//! attribute-list: attribute (',' attribute)*
//! attribute: identifier
//!          | identifier '(' argument-list ')'
//! argument-list: argument (',' argument)*
//! argument: integer-literal | string-literal | identifier | expression
//! ```
//!
//! Multiple `__attribute__` specifiers may appear consecutively; this module
//! collects them all into a single `Vec<Attribute>`.

use super::ast::{Attribute, AttributeArg, Span};
use super::{ParseError, Parser};
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public API
// ===========================================================================

/// Parses one or more consecutive `__attribute__((...))` specifiers and
/// returns a flat list of all attributes.
///
/// This function is called from declaration specifier parsing, declarator
/// parsing, and label statement parsing — anywhere GCC attributes may appear.
///
/// # Grammar
///
/// ```text
/// gnu-attributes: __attribute__ '(' '(' attribute-list ')' ')'
///               | gnu-attributes __attribute__ '(' '(' attribute-list ')' ')'
/// ```
pub fn parse_attribute_list(parser: &mut Parser<'_>) -> Result<Vec<Attribute>, ParseError> {
    let mut attrs: Vec<Attribute> = Vec::new();

    while parser.check(TokenKind::Attribute) {
        parser.advance(); // consume `__attribute__`

        // Expect `((`
        parser.expect(TokenKind::LeftParen)?;
        parser.expect(TokenKind::LeftParen)?;

        // Parse comma-separated attribute list
        if !parser.check(TokenKind::RightParen) {
            loop {
                let attr = parse_single_attribute(parser)?;
                attrs.push(attr);

                if !parser.eat(TokenKind::Comma) {
                    break;
                }
            }
        }

        // Expect `))`
        parser.expect(TokenKind::RightParen)?;
        parser.expect(TokenKind::RightParen)?;
    }

    Ok(attrs)
}

// ===========================================================================
// Internal
// ===========================================================================

/// Parses a single attribute: `identifier` or `identifier(args...)`.
fn parse_single_attribute(parser: &mut Parser<'_>) -> Result<Attribute, ParseError> {
    let start = parser.current().span;

    // Attribute name — must be an identifier (or a keyword-like token that
    // GCC treats as an attribute name, such as `const`, `volatile`, etc.)
    let name = match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            parser.advance();
            sym
        }
        // GCC allows some keywords as attribute names (e.g., `const`)
        TokenKind::Const | TokenKind::Volatile => {
            let sym = parser
                .interner
                .intern(&format!("{}", parser.current().kind));
            parser.advance();
            sym
        }
        _ => {
            let span = parser.current().span;
            // Empty attribute (e.g., `__attribute__((,))`) — skip
            let empty_sym = parser.interner.intern("");
            return Ok(Attribute {
                name: empty_sym,
                args: Vec::new(),
                span,
            });
        }
    };

    // Optional argument list
    let args = if parser.check(TokenKind::LeftParen) {
        parser.advance(); // consume `(`
        let mut args = Vec::new();

        if !parser.check(TokenKind::RightParen) {
            loop {
                let arg = parse_attribute_arg(parser)?;
                args.push(arg);

                if !parser.eat(TokenKind::Comma) {
                    break;
                }
            }
        }

        parser.expect(TokenKind::RightParen)?;
        args
    } else {
        Vec::new()
    };

    let end = parser.current().span;
    let span = Span::merge(start, end);

    Ok(Attribute { name, args, span })
}

/// Parses a single attribute argument.
///
/// Arguments can be:
/// - Integer literal
/// - String literal
/// - Identifier
/// - General expression (e.g., `sizeof(int)`)
fn parse_attribute_arg(parser: &mut Parser<'_>) -> Result<AttributeArg, ParseError> {
    match &parser.current().kind {
        TokenKind::IntegerLiteral { value, .. } => {
            let val = *value as i64;
            parser.advance();
            Ok(AttributeArg::Integer(val))
        }
        TokenKind::StringLiteral { value, .. } => {
            let val = value.clone();
            parser.advance();
            Ok(AttributeArg::String(val))
        }
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            parser.advance();
            Ok(AttributeArg::Identifier(sym))
        }
        _ => {
            // General expression argument
            let expr = super::expressions::parse_conditional_expression(parser)?;
            Ok(AttributeArg::Expression(Box::new(expr)))
        }
    }
}
