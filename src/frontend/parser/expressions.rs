//! Expression parsing module for the BCC C11 parser.
//!
//! Implements Pratt / precedence-climbing expression parsing for all C11
//! operators, primary expressions, casts, sizeof, `_Alignof`, `_Generic`,
//! compound literals, and GCC extensions (statement expressions, label
//! addresses, conditional omission).
//!
//! # Operator Precedence
//!
//! The parser follows the standard C operator precedence table (15 levels).
//! Assignment operators are right-associative; all other binary operators
//! are left-associative.
//!
//! # Architecture
//!
//! All public functions accept a `&mut Parser` and return
//! `Result<Expression, ParseError>`.

use super::ast::{
    AlignofOperand, BinaryOperator, Expression, GenericAssociation, SizeofOperand, Span,
    UnaryOperator,
};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public entry points — called from statements.rs, declarations.rs,
// gcc_extensions.rs, and mod.rs.
// ===========================================================================

/// Parses a full expression (comma-separated expression list).
///
/// This is the top-level expression rule in C:
///
/// ```text
/// expression:
///     assignment-expression
///     expression ',' assignment-expression
/// ```
///
/// The comma operator evaluates all sub-expressions left-to-right, discarding
/// all values except the last.
pub fn parse_expression(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    parser.enter_recursion()?;
    let first = parse_assignment_expression(parser)?;

    if !parser.check(TokenKind::Comma) {
        parser.leave_recursion();
        return Ok(first);
    }

    // Collect all comma-separated sub-expressions into a Vec.
    let mut exprs = vec![first];
    while parser.eat(TokenKind::Comma) {
        exprs.push(parse_assignment_expression(parser)?);
    }

    let start = exprs.first().unwrap().span();
    let end = exprs.last().unwrap().span();
    let span = Span::merge(start, end);

    parser.leave_recursion();
    Ok(Expression::Comma {
        expressions: exprs,
        span,
    })
}

/// Parses an assignment expression.
///
/// ```text
/// assignment-expression:
///     conditional-expression
///     unary-expression assignment-operator assignment-expression
/// ```
///
/// Since we cannot trivially distinguish unary-expression from conditional-
/// expression by lookahead, we parse a conditional-expression and then check
/// if it is followed by an assignment operator. If so, we reinterpret the
/// LHS as an lvalue and emit a `BinaryOp` with the assignment operator.
pub fn parse_assignment_expression(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    parser.enter_recursion()?;
    let lhs = parse_conditional_expression(parser)?;

    if let Some(op) = try_assignment_operator(&parser.current().kind) {
        let _op_span = parser.advance();
        let rhs = parse_assignment_expression(parser)?; // right-associative
        let span = Span::merge(lhs.span(), rhs.span());
        parser.leave_recursion();
        return Ok(Expression::BinaryOp {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
            span,
        });
    }

    parser.leave_recursion();
    Ok(lhs)
}

/// Parses a conditional (ternary) expression.
///
/// ```text
/// conditional-expression:
///     logical-or-expression
///     logical-or-expression '?' expression ':' conditional-expression
///     logical-or-expression '?' ':' conditional-expression   (GCC omission)
/// ```
///
/// The GCC extension form `x ?: y` omits the "then" operand; it is
/// dispatched to [`gcc_extensions::parse_conditional_omission`].
pub fn parse_conditional_expression(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    parser.enter_recursion()?;
    let condition = parse_logical_or(parser)?;

    if parser.eat(TokenKind::Question) {
        let question_span = parser.current().span;
        // GCC extension: `x ?: y` (conditional omission)
        if parser.check(TokenKind::Colon) {
            parser.leave_recursion();
            return super::gcc_extensions::parse_conditional_omission(
                parser,
                condition,
                question_span,
            );
        }

        let then_expr = parse_expression(parser)?;
        parser.expect(TokenKind::Colon)?;
        let else_expr = parse_conditional_expression(parser)?;
        let span = Span::merge(condition.span(), else_expr.span());
        parser.leave_recursion();
        return Ok(Expression::Conditional {
            condition: Box::new(condition),
            then_expr: Some(Box::new(then_expr)),
            else_expr: Box::new(else_expr),
            span,
        });
    }

    parser.leave_recursion();
    Ok(condition)
}

// ===========================================================================
// Binary operators — precedence climbing
// ===========================================================================

/// Parses logical-or: `a || b`
fn parse_logical_or(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_logical_and(parser)?;
    while parser.check(TokenKind::PipePipe) {
        parser.advance();
        let right = parse_logical_and(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op: BinaryOperator::LogOr,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses logical-and: `a && b`
fn parse_logical_and(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_bitwise_or(parser)?;
    while parser.check(TokenKind::AmpAmp) {
        parser.advance();
        let right = parse_bitwise_or(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op: BinaryOperator::LogAnd,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses bitwise-or: `a | b`
fn parse_bitwise_or(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_bitwise_xor(parser)?;
    while parser.check(TokenKind::Pipe) {
        parser.advance();
        let right = parse_bitwise_xor(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op: BinaryOperator::BitOr,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses bitwise-xor: `a ^ b`
fn parse_bitwise_xor(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_bitwise_and(parser)?;
    while parser.check(TokenKind::Caret) {
        parser.advance();
        let right = parse_bitwise_and(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op: BinaryOperator::BitXor,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses bitwise-and: `a & b`
fn parse_bitwise_and(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_equality(parser)?;
    while parser.check(TokenKind::Ampersand) {
        parser.advance();
        let right = parse_equality(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op: BinaryOperator::BitAnd,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses equality: `a == b`, `a != b`
fn parse_equality(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_relational(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::EqualEqual => BinaryOperator::Eq,
            TokenKind::NotEqual => BinaryOperator::Ne,
            _ => break,
        };
        parser.advance();
        let right = parse_relational(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses relational: `a < b`, `a > b`, `a <= b`, `a >= b`
fn parse_relational(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_shift(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::Less => BinaryOperator::Lt,
            TokenKind::Greater => BinaryOperator::Gt,
            TokenKind::LessEqual => BinaryOperator::Le,
            TokenKind::GreaterEqual => BinaryOperator::Ge,
            _ => break,
        };
        parser.advance();
        let right = parse_shift(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses shift: `a << b`, `a >> b`
fn parse_shift(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_additive(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::LeftShift => BinaryOperator::Shl,
            TokenKind::RightShift => BinaryOperator::Shr,
            _ => break,
        };
        parser.advance();
        let right = parse_additive(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses additive: `a + b`, `a - b`
fn parse_additive(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_multiplicative(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::Plus => BinaryOperator::Add,
            TokenKind::Minus => BinaryOperator::Sub,
            _ => break,
        };
        parser.advance();
        let right = parse_multiplicative(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

/// Parses multiplicative: `a * b`, `a / b`, `a % b`
fn parse_multiplicative(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_unary(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::Star => BinaryOperator::Mul,
            TokenKind::Slash => BinaryOperator::Div,
            TokenKind::Percent => BinaryOperator::Mod,
            _ => break,
        };
        parser.advance();
        let right = parse_unary(parser)?;
        let span = Span::merge(left.span(), right.span());
        left = Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }
    Ok(left)
}

// ===========================================================================
// Unary expressions
// ===========================================================================

/// Parses a unary expression.
///
/// ```text
/// unary-expression:
///     postfix-expression
///     '++' unary-expression
///     '--' unary-expression
///     unary-operator cast-expression
///     'sizeof' unary-expression
///     'sizeof' '(' type-name ')'
///     '_Alignof' '(' type-name ')'
///     '&&' identifier              (GCC label address)
///     '__extension__' expr         (GCC extension prefix)
/// ```
fn parse_unary(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    parser.enter_recursion()?;

    let result = match parser.current().kind {
        // Prefix increment / decrement
        TokenKind::PlusPlus => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::PreInc,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }
        TokenKind::MinusMinus => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::PreDec,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // GCC label address: `&&label`
        TokenKind::AmpAmp => super::gcc_extensions::parse_label_address(parser),

        // Address-of: `&expr`
        TokenKind::Ampersand => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::AddressOf {
                operand: Box::new(operand),
                span,
            })
        }

        // Dereference: `*expr`
        TokenKind::Star => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::Dereference {
                operand: Box::new(operand),
                span,
            })
        }

        // Unary plus: `+expr`
        TokenKind::Plus => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Plus,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // Unary minus: `-expr`
        TokenKind::Minus => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Neg,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // Bitwise NOT: `~expr`
        TokenKind::Tilde => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::BitNot,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // Logical NOT: `!expr`
        TokenKind::Exclaim => {
            let start = parser.advance();
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::LogNot,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // sizeof
        TokenKind::Sizeof => {
            let start = parser.advance();
            // sizeof(x) — could be type or expression
            let operand = parse_unary(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::Sizeof {
                operand: SizeofOperand::Expression(Box::new(operand)),
                span,
            })
        }

        // _Alignof
        TokenKind::Alignof => {
            let start = parser.advance();
            parser.expect(TokenKind::LeftParen)?;
            // For now, parse as expression. Sema resolves type vs expression.
            let operand = parse_expression(parser)?;
            let close = parser.expect(TokenKind::RightParen)?;
            let span = Span::merge(start, close);
            Ok(Expression::Alignof {
                operand: AlignofOperand::Expression(Box::new(operand)),
                span,
            })
        }

        // __extension__ expr (GCC extension)
        TokenKind::Extension => super::gcc_extensions::parse_extension_expr(parser),

        // Fall through to postfix expression
        _ => parse_postfix(parser),
    };

    parser.leave_recursion();
    result
}

// ===========================================================================
// Postfix expressions
// ===========================================================================

/// Parses a postfix expression.
///
/// ```text
/// postfix-expression:
///     primary-expression
///     postfix-expression '[' expression ']'
///     postfix-expression '(' argument-expression-list? ')'
///     postfix-expression '.' identifier
///     postfix-expression '->' identifier
///     postfix-expression '++'
///     postfix-expression '--'
/// ```
fn parse_postfix(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut expr = parse_primary(parser)?;

    loop {
        match parser.current().kind {
            // Array subscript: expr[index]
            TokenKind::LeftBracket => {
                parser.advance();
                let index = parse_expression(parser)?;
                let end = parser.expect(TokenKind::RightBracket)?;
                let span = Span::merge(expr.span(), end);
                expr = Expression::ArraySubscript {
                    array: Box::new(expr),
                    index: Box::new(index),
                    span,
                };
            }

            // Function call: expr(args)
            TokenKind::LeftParen => {
                parser.advance();
                let mut args = Vec::new();
                if !parser.check(TokenKind::RightParen) {
                    args.push(parse_assignment_expression(parser)?);
                    while parser.eat(TokenKind::Comma) {
                        args.push(parse_assignment_expression(parser)?);
                    }
                }
                let end = parser.expect(TokenKind::RightParen)?;
                let span = Span::merge(expr.span(), end);
                expr = Expression::FunctionCall {
                    callee: Box::new(expr),
                    args,
                    span,
                };
            }

            // Member access: expr.member
            TokenKind::Dot => {
                parser.advance();
                let (member, end) = parse_field_identifier(parser)?;
                let span = Span::merge(expr.span(), end);
                expr = Expression::MemberAccess {
                    object: Box::new(expr),
                    member,
                    span,
                };
            }

            // Arrow access: expr->member
            TokenKind::Arrow => {
                parser.advance();
                let (member, end) = parse_field_identifier(parser)?;
                let span = Span::merge(expr.span(), end);
                expr = Expression::ArrowAccess {
                    pointer: Box::new(expr),
                    member,
                    span,
                };
            }

            // Post-increment: expr++
            TokenKind::PlusPlus => {
                let end = parser.advance();
                let span = Span::merge(expr.span(), end);
                expr = Expression::UnaryOp {
                    op: UnaryOperator::PostInc,
                    operand: Box::new(expr),
                    is_postfix: true,
                    span,
                };
            }

            // Post-decrement: expr--
            TokenKind::MinusMinus => {
                let end = parser.advance();
                let span = Span::merge(expr.span(), end);
                expr = Expression::UnaryOp {
                    op: UnaryOperator::PostDec,
                    operand: Box::new(expr),
                    is_postfix: true,
                    span,
                };
            }

            _ => break,
        }
    }

    Ok(expr)
}

// ===========================================================================
// Primary expressions
// ===========================================================================

/// Parses a primary expression.
///
/// ```text
/// primary-expression:
///     identifier
///     integer-literal
///     float-literal
///     string-literal
///     char-literal
///     '(' expression ')'
///     '(' '{' ... '}' ')'           (GCC statement expression)
///     _Generic( ... )
/// ```
fn parse_primary(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    match parser.current().kind.clone() {
        // Identifier
        TokenKind::Identifier(sym) => {
            let span = parser.advance();
            Ok(Expression::Identifier { name: sym, span })
        }

        // Integer literal
        TokenKind::IntegerLiteral { value, suffix } => {
            let span = parser.advance();
            Ok(Expression::IntegerLiteral {
                value,
                suffix,
                span,
            })
        }

        // Float literal
        TokenKind::FloatLiteral { value, suffix } => {
            let span = parser.advance();
            Ok(Expression::FloatLiteral {
                value,
                suffix,
                span,
            })
        }

        // String literal
        TokenKind::StringLiteral { value, prefix } => {
            let span = parser.advance();
            Ok(Expression::StringLiteral {
                value,
                prefix,
                span,
            })
        }

        // Character literal
        TokenKind::CharLiteral { value, prefix } => {
            let span = parser.advance();
            Ok(Expression::CharLiteral {
                value,
                prefix,
                span,
            })
        }

        // Parenthesized expression or GCC statement expression
        TokenKind::LeftParen => {
            let _start = parser.advance();
            // GCC statement expression: `({ ... })`
            if parser.check(TokenKind::LeftBrace) {
                let expr = super::gcc_extensions::parse_statement_expression(parser)?;
                parser.expect(TokenKind::RightParen)?;
                return Ok(expr);
            }

            let inner = parse_expression(parser)?;
            parser.expect(TokenKind::RightParen)?;
            // Parenthesized expressions are not a distinct AST node — they
            // are just structural grouping in the source.
            Ok(inner)
        }

        // _Generic selection
        TokenKind::Generic => parse_generic_selection(parser),

        // Unexpected token
        _ => {
            let tok = parser.current();
            let span = tok.span;
            let msg = format!("expected expression, found '{}'", tok.kind);
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("expression".to_string()),
            })
        }
    }
}

/// Parses a `_Generic` selection expression (C11 §6.5.1.1).
///
/// ```text
/// _Generic ( assignment-expression , generic-assoc-list )
/// generic-assoc-list: generic-association (',' generic-association)*
/// generic-association: type-name ':' assignment-expression
///                    | 'default' ':' assignment-expression
/// ```
fn parse_generic_selection(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let start = parser.advance(); // consume _Generic
    parser.expect(TokenKind::LeftParen)?;
    let controlling = parse_assignment_expression(parser)?;
    parser.expect(TokenKind::Comma)?;

    let mut associations = Vec::new();
    loop {
        let assoc_start = parser.current().span;
        if parser.check(TokenKind::Default) {
            parser.advance();
            parser.expect(TokenKind::Colon)?;
            let expr = parse_assignment_expression(parser)?;
            let assoc_span = Span::merge(assoc_start, expr.span());
            associations.push(GenericAssociation {
                type_name: None,
                expression: Box::new(expr),
                span: assoc_span,
            });
        } else {
            // Parse type-name as expression (simplified; sema resolves).
            let type_expr = parse_assignment_expression(parser)?;
            parser.expect(TokenKind::Colon)?;
            let expr = parse_assignment_expression(parser)?;
            let assoc_span = Span::merge(assoc_start, expr.span());
            // For the minimal parser, we treat the type as an expression
            // and record it. The semantic analyzer resolves type names.
            let _ = type_expr; // The type name information is lost here;
                               // TODO: Proper type-name parsing in the types module will fill this in.
            associations.push(GenericAssociation {
                type_name: None, // Will be filled in by sema/types module
                expression: Box::new(expr),
                span: assoc_span,
            });
        }

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);
    Ok(Expression::Generic {
        controlling: Box::new(controlling),
        associations,
        span,
    })
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parses a field/member identifier for `.member` or `->member` access.
fn parse_field_identifier(parser: &mut Parser<'_>) -> Result<(Symbol, Span), ParseError> {
    match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            let span = parser.advance();
            Ok((sym, span))
        }
        _ => {
            let span = parser.current().span;
            let msg = "expected field/member name (identifier)".to_string();
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier".to_string()),
            })
        }
    }
}

/// Attempts to match an assignment operator token, returning the
/// corresponding `BinaryOperator` variant if successful.
fn try_assignment_operator(kind: &TokenKind) -> Option<BinaryOperator> {
    match kind {
        TokenKind::Assign => Some(BinaryOperator::Assign),
        TokenKind::PlusAssign => Some(BinaryOperator::AddAssign),
        TokenKind::MinusAssign => Some(BinaryOperator::SubAssign),
        TokenKind::StarAssign => Some(BinaryOperator::MulAssign),
        TokenKind::SlashAssign => Some(BinaryOperator::DivAssign),
        TokenKind::PercentAssign => Some(BinaryOperator::ModAssign),
        TokenKind::AmpAssign => Some(BinaryOperator::BitAndAssign),
        TokenKind::PipeAssign => Some(BinaryOperator::BitOrAssign),
        TokenKind::CaretAssign => Some(BinaryOperator::BitXorAssign),
        TokenKind::LeftShiftAssign => Some(BinaryOperator::ShlAssign),
        TokenKind::RightShiftAssign => Some(BinaryOperator::ShrAssign),
        _ => None,
    }
}
