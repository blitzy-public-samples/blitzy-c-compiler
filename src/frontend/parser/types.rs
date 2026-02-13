//! Type specifier and qualifier parsing module for the BCC C11 parser.
//!
//! Handles parsing of:
//! - `typeof` / `__typeof__` (GCC extension)
//! - `_Atomic(type-name)` (C11)
//! - Type name parsing for casts, sizeof, _Alignof, compound literals
//! - Specifier-qualifier lists (used in struct/union members, type names)
//! - Abstract declarators (declarators without names)
//!
//! # Integration
//!
//! These functions are called from:
//! - `expressions.rs` — for sizeof/alignof/cast type operands
//! - `declarations.rs` — for specifier-qualifier lists in struct fields
//! - `gcc_extensions.rs` — for typeof in extension contexts

use super::ast::{
    AbstractDeclarator, DerivedDeclarator, Span, SpecifierQualifierList, TypeName, TypeQualifiers,
    TypeSpecifier,
};
use super::{ParseError, Parser};
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Helpers
// ===========================================================================

/// Extracts the source span from a `TypeSpecifier` variant, if it carries one.
fn spec_span(spec: &TypeSpecifier) -> Option<Span> {
    match spec {
        TypeSpecifier::Struct { span, .. }
        | TypeSpecifier::Union { span, .. }
        | TypeSpecifier::Enum { span, .. }
        | TypeSpecifier::TypedefName { span, .. }
        | TypeSpecifier::Typeof { span, .. } => Some(*span),
        _ => None,
    }
}

// ===========================================================================
// Public API
// ===========================================================================

/// Parses a type-name: specifier-qualifier-list followed by an optional
/// abstract declarator.
///
/// ```text
/// type-name:
///     specifier-qualifier-list abstract-declarator?
/// ```
///
/// Used in cast expressions `(type-name) expr`, sizeof, _Alignof,
/// compound literals, _Generic associations.
pub fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ParseError> {
    let start = parser.current().span;
    let specifiers = parse_specifier_qualifier_list(parser)?;

    let declarator = if can_start_abstract_declarator(parser) {
        Some(parse_abstract_declarator(parser)?)
    } else {
        None
    };

    let end = declarator.as_ref().map_or(specifiers.span, |d| d.span);
    let span = Span::merge(start, end);

    Ok(TypeName {
        specifiers,
        declarator,
        span,
    })
}

/// Parses a specifier-qualifier list (type specifiers and qualifiers only,
/// no storage class or function specifiers).
///
/// ```text
/// specifier-qualifier-list:
///     type-specifier specifier-qualifier-list?
///     type-qualifier specifier-qualifier-list?
/// ```
pub fn parse_specifier_qualifier_list(
    parser: &mut Parser<'_>,
) -> Result<SpecifierQualifierList, ParseError> {
    let start = parser.current().span;
    let mut specifiers: Vec<TypeSpecifier> = Vec::new();
    let mut qualifiers = TypeQualifiers::default();
    let mut last_span = start;

    loop {
        match parser.current().kind.clone() {
            // Type specifier keywords
            TokenKind::Void => {
                specifiers.push(TypeSpecifier::Void);
                last_span = parser.advance();
            }
            TokenKind::Char => {
                specifiers.push(TypeSpecifier::Char);
                last_span = parser.advance();
            }
            TokenKind::Short => {
                specifiers.push(TypeSpecifier::Short);
                last_span = parser.advance();
            }
            TokenKind::Int => {
                specifiers.push(TypeSpecifier::Int);
                last_span = parser.advance();
            }
            TokenKind::Long => {
                specifiers.push(TypeSpecifier::Long);
                last_span = parser.advance();
            }
            TokenKind::Float => {
                specifiers.push(TypeSpecifier::Float);
                last_span = parser.advance();
            }
            TokenKind::Double => {
                specifiers.push(TypeSpecifier::Double);
                last_span = parser.advance();
            }
            TokenKind::Signed => {
                specifiers.push(TypeSpecifier::Signed);
                last_span = parser.advance();
            }
            TokenKind::Unsigned => {
                specifiers.push(TypeSpecifier::Unsigned);
                last_span = parser.advance();
            }
            TokenKind::Bool => {
                specifiers.push(TypeSpecifier::Bool);
                last_span = parser.advance();
            }
            TokenKind::Complex => {
                specifiers.push(TypeSpecifier::Complex);
                last_span = parser.advance();
            }

            // struct / union / enum
            TokenKind::Struct => {
                let spec = super::declarations::parse_struct_or_union_specifier(parser, true)?;
                last_span = spec_span(&spec).unwrap_or(last_span);
                specifiers.push(spec);
            }
            TokenKind::Union => {
                let spec = super::declarations::parse_struct_or_union_specifier(parser, false)?;
                last_span = spec_span(&spec).unwrap_or(last_span);
                specifiers.push(spec);
            }
            TokenKind::Enum => {
                let spec = super::declarations::parse_enum_specifier(parser)?;
                last_span = spec_span(&spec).unwrap_or(last_span);
                specifiers.push(spec);
            }

            // Type qualifiers
            TokenKind::Const => {
                qualifiers.is_const = true;
                last_span = parser.advance();
            }
            TokenKind::Volatile => {
                qualifiers.is_volatile = true;
                last_span = parser.advance();
            }
            TokenKind::Restrict => {
                qualifiers.is_restrict = true;
                last_span = parser.advance();
            }
            TokenKind::Atomic => {
                qualifiers.is_atomic = true;
                last_span = parser.advance();
            }

            // typeof / __typeof__
            TokenKind::TypeofKeyword => {
                let typeof_start = parser.advance();
                parser.expect(TokenKind::LeftParen)?;
                let expr = super::expressions::parse_expression(parser)?;
                let end = parser.expect(TokenKind::RightParen)?;
                let span = Span::merge(typeof_start, end);
                specifiers.push(TypeSpecifier::Typeof {
                    operand: super::ast::TypeofOperand::Expression(Box::new(expr)),
                    span,
                });
                last_span = span;
            }

            // __attribute__
            TokenKind::Attribute => {
                // Attributes on specifier-qualifier lists are allowed by GCC.
                // We parse and discard them here (they'll be collected elsewhere).
                let _attrs = super::attributes::parse_attribute_list(parser)?;
            }

            // Typedef name
            TokenKind::Identifier(sym) => {
                if specifiers.is_empty() && parser.is_typedef_name(sym) {
                    let tspan = parser.current().span;
                    specifiers.push(TypeSpecifier::TypedefName {
                        name: sym,
                        span: tspan,
                    });
                    last_span = parser.advance();
                } else {
                    break;
                }
            }

            _ => break,
        }
    }

    let span = Span::merge(start, last_span);
    Ok(SpecifierQualifierList {
        specifiers,
        qualifiers,
        span,
    })
}

/// Returns `true` if the current token can begin an abstract declarator.
fn can_start_abstract_declarator(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Star | TokenKind::LeftParen | TokenKind::LeftBracket
    )
}

/// Parses an abstract declarator: `pointer? direct-abstract-declarator?`.
///
/// An abstract declarator is a declarator without a name — used in type
/// names for casts, sizeof, function parameter types, etc.
pub fn parse_abstract_declarator(
    parser: &mut Parser<'_>,
) -> Result<AbstractDeclarator, ParseError> {
    let start = parser.current().span;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Pointer derivations
    while parser.check(TokenKind::Star) {
        parser.advance();
        let quals = parse_qualifier_list(parser);
        derived.push(DerivedDeclarator::Pointer { qualifiers: quals });
    }

    // Direct abstract declarator postfix: arrays and functions
    loop {
        match parser.current().kind {
            TokenKind::LeftBracket => {
                parser.advance(); // consume `[`
                let size = if parser.check(TokenKind::RightBracket) {
                    None
                } else {
                    Some(Box::new(super::expressions::parse_assignment_expression(
                        parser,
                    )?))
                };
                parser.expect(TokenKind::RightBracket)?;
                derived.push(DerivedDeclarator::Array {
                    size,
                    is_static: false,
                    qualifiers: TypeQualifiers::default(),
                });
            }
            TokenKind::LeftParen => {
                // Could be a grouping `(abstract-declarator)` or a function
                // parameter list `(parameter-type-list)`.
                // Heuristic: if after `(` we see `)` or a declaration-start
                // token, it's a function parameter list.
                if matches!(
                    parser.peek_ahead(1).kind,
                    TokenKind::RightParen
                        | TokenKind::Void
                        | TokenKind::Int
                        | TokenKind::Char
                        | TokenKind::Short
                        | TokenKind::Long
                        | TokenKind::Float
                        | TokenKind::Double
                        | TokenKind::Signed
                        | TokenKind::Unsigned
                        | TokenKind::Struct
                        | TokenKind::Union
                        | TokenKind::Enum
                        | TokenKind::Const
                        | TokenKind::Volatile
                        | TokenKind::Restrict
                        | TokenKind::Ellipsis
                ) || parser.peek_ahead(1).kind == TokenKind::RightParen
                {
                    // Function parameter list
                    let func = super::declarations::parse_function_declarator(parser)?;
                    derived.push(func);
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    let end = parser.current().span;
    let span = Span::merge(start, end);

    Ok(AbstractDeclarator { derived, span })
}

/// Parses zero or more type qualifiers.
fn parse_qualifier_list(parser: &mut Parser<'_>) -> TypeQualifiers {
    let mut quals = TypeQualifiers::default();
    loop {
        match parser.current().kind {
            TokenKind::Const => {
                quals.is_const = true;
                parser.advance();
            }
            TokenKind::Volatile => {
                quals.is_volatile = true;
                parser.advance();
            }
            TokenKind::Restrict => {
                quals.is_restrict = true;
                parser.advance();
            }
            TokenKind::Atomic => {
                quals.is_atomic = true;
                parser.advance();
            }
            _ => break,
        }
    }
    quals
}
