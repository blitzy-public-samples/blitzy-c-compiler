//! Expression parsing module for the BCC C11 parser.
//!
//! Implements recursive-descent expression parsing for all C11 operators,
//! primary expressions, casts, sizeof, `_Alignof`, `_Generic`, compound
//! literals, and GCC extensions (statement expressions, label addresses,
//! conditional omission `x ?: y`, `__extension__`).
//!
//! # Operator Precedence
//!
//! The parser follows the standard C operator precedence table (15 levels).
//! Each level is implemented as a separate function in the recursive-descent
//! chain.  Assignment operators and the conditional operator (`?:`) are
//! right-associative; all other binary operators are left-associative.
//!
//! # Type-Name Disambiguation
//!
//! Parenthesised constructs `(...)` in expression context may represent:
//! - Parenthesised expression:   `(expr)`
//! - GCC statement expression:   `({ ... })`
//! - Cast expression:            `(type-name) unary-expression`
//! - Compound literal:           `(type-name) { initializer-list }`
//!
//! The disambiguator peeks at the token after `(`.  If it can start a
//! type-name (type specifier keyword, type qualifier, `struct`/`union`/`enum`,
//! `typeof`/`__typeof__`, or a registered typedef name) we parse a
//! type-name.  After closing `)`, a following `{` selects compound-literal;
//! otherwise it is a cast.  All other cases fall through to a regular
//! parenthesised (or statement) expression.
//!
//! # Architecture
//!
//! All public functions accept a `&mut Parser` and return
//! `Result<Expression, ParseError>`.  Private helpers are used for each
//! precedence level.

use super::ast::{
    AlignofOperand, BinaryOperator, Designator, Expression, GenericAssociation, Initializer,
    InitializerItem, SizeofOperand, Span, UnaryOperator,
};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public entry points — called from statements.rs, declarations.rs,
// gcc_extensions.rs, types.rs, inline_asm.rs, attributes.rs, and mod.rs.
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

/// Parses an assignment expression — used in function arguments, initializers,
/// and anywhere the comma operator must *not* be consumed.
///
/// ```text
/// assignment-expression:
///     conditional-expression
///     unary-expression assignment-operator assignment-expression
/// ```
///
/// Since we cannot trivially distinguish unary-expression from conditional-
/// expression by lookahead, we parse a conditional-expression first, then
/// check if it is followed by an assignment operator.  If so, we reinterpret
/// the LHS as an lvalue and emit a `BinaryOp` with the assignment operator.
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

/// Parses a constant expression — used for array sizes, case values, enum
/// values, bitfield widths, `_Static_assert` conditions, and similar
/// contexts where only compile-time-evaluable expressions are allowed.
///
/// ```text
/// constant-expression:
///     conditional-expression
/// ```
///
/// The grammar for constant expressions is identical to conditional
/// expressions.  The compile-time-evaluability constraint is enforced by
/// the semantic analyzer, not the parser.
pub fn parse_constant_expression(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    parse_conditional_expression(parser)
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
/// dispatched to [`super::gcc_extensions::parse_conditional_omission`].
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
// Binary operators — one function per precedence level (recursive-descent)
// ===========================================================================

/// Parses logical-or: `a || b`   (precedence 4)
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

/// Parses logical-and: `a && b`   (precedence 5)
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

/// Parses bitwise-or: `a | b`    (precedence 6)
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

/// Parses bitwise-xor: `a ^ b`   (precedence 7)
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

/// Parses bitwise-and: `a & b`   (precedence 8)
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

/// Parses equality: `a == b`, `a != b`   (precedence 9)
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

/// Parses relational: `a < b`, `a > b`, `a <= b`, `a >= b`   (precedence 10)
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

/// Parses shift: `a << b`, `a >> b`   (precedence 11)
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

/// Parses additive: `a + b`, `a - b`   (precedence 12)
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

/// Parses multiplicative: `a * b`, `a / b`, `a % b`   (precedence 13)
fn parse_multiplicative(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let mut left = parse_cast(parser)?;
    loop {
        let op = match parser.current().kind {
            TokenKind::Star => BinaryOperator::Mul,
            TokenKind::Slash => BinaryOperator::Div,
            TokenKind::Percent => BinaryOperator::Mod,
            _ => break,
        };
        parser.advance();
        let right = parse_cast(parser)?;
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
// Cast expression — the bridge between binary and unary operators
// ===========================================================================

/// Parses a cast expression (C11 §6.5.4).
///
/// ```text
/// cast-expression:
///     unary-expression
///     '(' type-name ')' cast-expression
/// ```
///
/// When the parser sees `(`, it must disambiguate between:
/// 1. `(type-name) cast-expression` — explicit cast
/// 2. `(type-name) { init-list }` — compound literal (C11)
/// 3. `({ ... })` — GCC statement expression
/// 4. `( expression )` — parenthesized expression (handled inside
///    `parse_unary` → `parse_postfix` → `parse_primary`)
///
/// The heuristic: after consuming `(`, if the current token can begin a
/// type-name *and* the construct is not a statement expression, we attempt
/// to parse as type-name.  After the closing `)`, a `{` yields a compound
/// literal, otherwise a cast.  If the token after `(` cannot start a
/// type-name, we fall through to `parse_unary`.
fn parse_cast(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    // Only attempt the type-name path when `(` is current **and** the token
    // after `(` looks like it can start a type-name.
    if parser.check(TokenKind::LeftParen) && is_type_name_start_at(parser, 1) {
        // Speculatively peek: if the token after `(` is `{` this is actually
        // a GCC statement expression `({ ... })`, not a cast.
        if matches!(parser.peek_ahead(1).kind, TokenKind::LeftBrace) {
            return parse_unary(parser);
        }

        let start = parser.current().span;
        parser.advance(); // consume `(`
        let type_name = super::types::parse_type_name(parser)?;
        parser.expect(TokenKind::RightParen)?;

        // Compound literal: `(type-name) { initializer-list }`
        if parser.check(TokenKind::LeftBrace) {
            let initializer = parse_brace_initializer(parser)?;
            let end_span = initializer_span(&initializer);
            let span = Span::merge(start, end_span);
            return Ok(Expression::CompoundLiteral {
                type_name: Box::new(type_name),
                initializer,
                span,
            });
        }

        // Cast: `(type-name) cast-expression`
        let operand = parse_cast(parser)?;
        let span = Span::merge(start, operand.span());
        return Ok(Expression::Cast {
            type_name: Box::new(type_name),
            operand: Box::new(operand),
            span,
        });
    }

    parse_unary(parser)
}

// ===========================================================================
// Unary expressions
// ===========================================================================

/// Parses a unary expression (C11 §6.5.3).
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
///     '__alignof__' '(' type-name ')'   (GCC synonym)
///     '&&' identifier                   (GCC label address)
///     '__extension__' expression         (GCC extension prefix)
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
            let operand = parse_cast(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::AddressOf {
                operand: Box::new(operand),
                span,
            })
        }

        // Dereference: `*expr`
        TokenKind::Star => {
            let start = parser.advance();
            let operand = parse_cast(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::Dereference {
                operand: Box::new(operand),
                span,
            })
        }

        // Unary plus: `+expr`
        TokenKind::Plus => {
            let start = parser.advance();
            let operand = parse_cast(parser)?;
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
            let operand = parse_cast(parser)?;
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
            let operand = parse_cast(parser)?;
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
            let operand = parse_cast(parser)?;
            let span = Span::merge(start, operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOperator::LogNot,
                operand: Box::new(operand),
                is_postfix: false,
                span,
            })
        }

        // sizeof (C11 §6.5.3.4)
        TokenKind::Sizeof => parse_sizeof(parser),

        // _Alignof / __alignof__ (C11 §6.5.3.4)
        TokenKind::Alignof => parse_alignof(parser),

        // __extension__ expr (GCC extension prefix)
        TokenKind::Extension => super::gcc_extensions::parse_extension_expr(parser),

        // Fall through to postfix expression
        _ => parse_postfix(parser),
    };

    parser.leave_recursion();
    result
}

/// Parses a `sizeof` expression (C11 §6.5.3.4).
///
/// ```text
/// sizeof unary-expression
/// sizeof ( type-name )
/// ```
///
/// Disambiguation: if the token after `sizeof` is `(` and the token after
/// *that* can start a type-name, we parse `sizeof(type-name)`.  Otherwise
/// we parse `sizeof unary-expression`.
fn parse_sizeof(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let start = parser.advance(); // consume `sizeof`

    // sizeof ( type-name )
    if parser.check(TokenKind::LeftParen) && is_type_name_start_at(parser, 1) {
        // Guard against `sizeof({ ... })` which is a GCC statement expression
        if !matches!(parser.peek_ahead(1).kind, TokenKind::LeftBrace) {
            parser.advance(); // consume `(`
            let type_name = super::types::parse_type_name(parser)?;
            let end = parser.expect(TokenKind::RightParen)?;
            let span = Span::merge(start, end);
            return Ok(Expression::Sizeof {
                operand: SizeofOperand::TypeName(Box::new(type_name)),
                span,
            });
        }
    }

    // sizeof unary-expression
    let operand = parse_unary(parser)?;
    let span = Span::merge(start, operand.span());
    Ok(Expression::Sizeof {
        operand: SizeofOperand::Expression(Box::new(operand)),
        span,
    })
}

/// Parses an `_Alignof` / `__alignof__` expression (C11 §6.5.3.4).
///
/// ```text
/// _Alignof ( type-name )
/// __alignof__ ( expression )        (GCC extension)
/// ```
///
/// C11 mandates that `_Alignof` takes a type-name in parentheses.  GCC
/// additionally allows `__alignof__(expr)`.  We attempt type-name first;
/// if that fails (the current token after `(` does not start a type-name),
/// we fall back to an expression operand.
fn parse_alignof(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let start = parser.advance(); // consume `_Alignof` / `__alignof__`
    parser.expect(TokenKind::LeftParen)?;

    // Attempt type-name
    if is_type_name_start(parser) {
        let type_name = super::types::parse_type_name(parser)?;
        let end = parser.expect(TokenKind::RightParen)?;
        let span = Span::merge(start, end);
        return Ok(Expression::Alignof {
            operand: AlignofOperand::TypeName(Box::new(type_name)),
            span,
        });
    }

    // GCC extension: __alignof__(expression)
    let expr = parse_expression(parser)?;
    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);
    Ok(Expression::Alignof {
        operand: AlignofOperand::Expression(Box::new(expr)),
        span,
    })
}

// ===========================================================================
// Postfix expressions
// ===========================================================================

/// Parses a postfix expression (C11 §6.5.2).
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

/// Parses a primary expression (C11 §6.5.1).
///
/// ```text
/// primary-expression:
///     identifier
///     constant (integer / float / char)
///     string-literal
///     '(' expression ')'
///     '(' '{' ... '}' ')'        (GCC statement expression)
///     _Generic( ... )
/// ```
///
/// Note: cast expressions `(type-name) expr` and compound literals
/// `(type-name) { init }` are handled by [`parse_cast`] *before* this
/// function is reached.  By the time we enter `parse_primary`, any `(`
/// we encounter is guaranteed to be either a parenthesized expression
/// or a GCC statement expression.
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

        // String literal — handles adjacent string concatenation (C11 §5.1.1.2
        // translation phase 6). Adjacent string literals are concatenated into a
        // single string literal. This must happen in the parser because the
        // preprocessor may output multiple adjacent string tokens across lines.
        TokenKind::StringLiteral { value, prefix } => {
            let span = parser.advance();
            let mut combined_value = value;
            let mut combined_prefix = prefix;
            let mut combined_span = span;

            // Consume any adjacent string literals.
            loop {
                let (next_val, next_pfx) = match &parser.current().kind {
                    TokenKind::StringLiteral { value, prefix } => (value.clone(), *prefix),
                    _ => break,
                };
                let next_span = parser.advance();
                combined_value.extend_from_slice(&next_val);
                combined_span = Span::merge(combined_span, next_span);
                // Resolve prefix: if both are None, stays None; otherwise
                // wider prefix wins (simplified; C11 §6.4.5p5).
                if combined_prefix == crate::frontend::lexer::token::StringPrefix::None {
                    combined_prefix = next_pfx;
                }
            }

            Ok(Expression::StringLiteral {
                value: combined_value,
                prefix: combined_prefix,
                span: combined_span,
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

        // Parenthesized expression or GCC statement expression.
        //
        // Cast/compound-literal cases have already been handled by
        // `parse_cast` before we reach here, so we only need to deal
        // with plain parenthesized expressions and statement expressions.
        TokenKind::LeftParen => {
            // GCC statement expression: `({ ... })`
            // Check for `{` at peek position 1 (i.e. after `(`) BEFORE
            // consuming the `(`.  `parse_statement_expression` owns the
            // full `({ ... })` production including both outer parens.
            if matches!(parser.peek_ahead(1).kind, TokenKind::LeftBrace) {
                return super::gcc_extensions::parse_statement_expression(parser);
            }

            let _start = parser.advance(); // consume `(`
            let inner = parse_expression(parser)?;
            parser.expect(TokenKind::RightParen)?;
            // Parenthesized expressions are not a distinct AST node — they
            // are just structural grouping in the source.
            Ok(inner)
        }

        // _Generic selection expression (C11 §6.5.1.1)
        TokenKind::Generic => parse_generic_selection(parser),

        // ------------------------------------------------------------------
        // GCC __builtin_offsetof(type, member) — special syntax because the
        // first argument is a type-name, not an expression.
        // We parse it into an AST node that the semantic analyser can handle.
        // ------------------------------------------------------------------
        TokenKind::BuiltinOffsetof => {
            let start = parser.advance(); // consume __builtin_offsetof
            parser.expect(TokenKind::LeftParen)?;
            // Parse the type name
            let type_name = super::types::parse_type_name(parser)?;
            parser.expect(TokenKind::Comma)?;
            // The member designator is parsed as an identifier (possibly with
            // nested . and [] access, but for now just an identifier suffices
            // for the most common usage).
            let member = parse_assignment_expression(parser)?;
            let end = parser.expect(TokenKind::RightParen)?;
            // Represent as a call with the type stringified — the sema will
            // recognise BuiltinOffsetof calls by name and handle type arg.
            let _name = parser.interner.intern("__builtin_offsetof");
            Ok(Expression::BuiltinOffsetof {
                type_name: Box::new(type_name),
                member: Box::new(member),
                span: Span::merge(start, end),
            })
        }

        // ------------------------------------------------------------------
        // GCC __builtin_types_compatible_p(type1, type2) — both arguments
        // are type-names.
        // ------------------------------------------------------------------
        TokenKind::BuiltinTypesCompatibleP => {
            let start = parser.advance();
            parser.expect(TokenKind::LeftParen)?;
            let type1 = super::types::parse_type_name(parser)?;
            parser.expect(TokenKind::Comma)?;
            let type2 = super::types::parse_type_name(parser)?;
            let end = parser.expect(TokenKind::RightParen)?;
            Ok(Expression::BuiltinTypesCompatibleP {
                type1: Box::new(type1),
                type2: Box::new(type2),
                span: Span::merge(start, end),
            })
        }

        // ------------------------------------------------------------------
        // GCC __builtin_choose_expr(const_expr, expr1, expr2) — all are
        // expressions, so treat like a regular function call via identifier.
        // ------------------------------------------------------------------
        TokenKind::BuiltinChooseExpr => {
            let name = parser.interner.intern("__builtin_choose_expr");
            let span = parser.advance();
            Ok(Expression::Identifier { name, span })
        }

        // ------------------------------------------------------------------
        // GCC __builtin_va_arg(ap, type) — second argument is a type-name.
        // ------------------------------------------------------------------
        TokenKind::BuiltinVaArg => {
            let start = parser.advance();
            parser.expect(TokenKind::LeftParen)?;
            let ap_expr = parse_assignment_expression(parser)?;
            parser.expect(TokenKind::Comma)?;
            let type_name = super::types::parse_type_name(parser)?;
            let end = parser.expect(TokenKind::RightParen)?;
            Ok(Expression::BuiltinVaArg {
                ap: Box::new(ap_expr),
                type_name: Box::new(type_name),
                span: Span::merge(start, end),
            })
        }

        // ------------------------------------------------------------------
        // GCC builtin function calls — these are lexed as keyword tokens
        // but behave like function-call identifiers at the expression level.
        // We convert them to regular identifiers so that `parse_postfix`
        // can handle the `(args...)` call syntax.
        // ------------------------------------------------------------------
        TokenKind::BuiltinConstantP
        | TokenKind::BuiltinExpect
        | TokenKind::BuiltinUnreachable
        | TokenKind::BuiltinTrap
        | TokenKind::BuiltinClz
        | TokenKind::BuiltinCtz
        | TokenKind::BuiltinPopcount
        | TokenKind::BuiltinBswap16
        | TokenKind::BuiltinBswap32
        | TokenKind::BuiltinBswap64
        | TokenKind::BuiltinFfs
        | TokenKind::BuiltinFrameAddress
        | TokenKind::BuiltinReturnAddress
        | TokenKind::BuiltinAssumeAligned
        | TokenKind::BuiltinAddOverflow
        | TokenKind::BuiltinSubOverflow
        | TokenKind::BuiltinMulOverflow
        | TokenKind::BuiltinVaStart
        | TokenKind::BuiltinVaEnd
        | TokenKind::BuiltinVaCopy => {
            let name = format!("{}", parser.current().kind);
            let sym = parser.interner.intern(&name);
            let span = parser.advance();
            Ok(Expression::Identifier { name: sym, span })
        }

        // Unexpected token — emit diagnostic and return error
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

// ===========================================================================
// _Generic selection expression
// ===========================================================================

/// Parses a `_Generic` selection expression (C11 §6.5.1.1).
///
/// ```text
/// _Generic ( assignment-expression , generic-assoc-list )
/// generic-assoc-list:
///     generic-association
///     generic-assoc-list ',' generic-association
/// generic-association:
///     type-name ':' assignment-expression
///     'default' ':' assignment-expression
/// ```
///
/// The controlling expression's type is evaluated at compile time and
/// matched against the type-name associations.  Only the matched
/// association's expression is selected (the others are not evaluated).
fn parse_generic_selection(parser: &mut Parser<'_>) -> Result<Expression, ParseError> {
    let start = parser.advance(); // consume `_Generic`
    parser.expect(TokenKind::LeftParen)?;
    let controlling = parse_assignment_expression(parser)?;
    parser.expect(TokenKind::Comma)?;

    let mut associations = Vec::new();
    loop {
        let assoc_start = parser.current().span;

        if parser.check(TokenKind::Default) {
            // default: assignment-expression
            parser.advance(); // consume `default`
            parser.expect(TokenKind::Colon)?;
            let expr = parse_assignment_expression(parser)?;
            let assoc_span = Span::merge(assoc_start, expr.span());
            associations.push(GenericAssociation {
                type_name: None,
                expression: Box::new(expr),
                span: assoc_span,
            });
        } else {
            // type-name : assignment-expression
            let type_name = super::types::parse_type_name(parser)?;
            parser.expect(TokenKind::Colon)?;
            let expr = parse_assignment_expression(parser)?;
            let assoc_span = Span::merge(assoc_start, expr.span());
            associations.push(GenericAssociation {
                type_name: Some(type_name),
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
// Brace-enclosed initializer parsing (for compound literals)
// ===========================================================================

/// Parses a brace-enclosed initializer list (C11 §6.7.9).
///
/// ```text
/// { initializer-list [,] }
/// { }
/// ```
///
/// This is used by compound literals `(type-name) { init-list }`.  The
/// logic mirrors `declarations::parse_brace_initializer` but is local to
/// this module to avoid cross-module visibility issues.
fn parse_brace_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ParseError> {
    let start = parser.expect(TokenKind::LeftBrace)?;
    let mut items: Vec<InitializerItem> = Vec::new();

    while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
        let item_start = parser.current().span;
        let mut designators: Vec<Designator> = Vec::new();

        // Parse designators: `.field` or `[index]`
        while parser.check(TokenKind::Dot) || parser.check(TokenKind::LeftBracket) {
            if parser.eat(TokenKind::Dot) {
                // Field designator: `.field`
                let field = parse_designator_field(parser)?;
                designators.push(Designator::Field(field));
            } else {
                // Array index designator: `[expr]`
                parser.advance(); // consume `[`
                let index = parse_conditional_expression(parser)?;
                parser.expect(TokenKind::RightBracket)?;
                designators.push(Designator::Index(Box::new(index)));
            }
        }

        // After designators, expect `=`
        if !designators.is_empty() {
            parser.expect(TokenKind::Assign)?;
        }

        // Initializer value — nested braces or a single expression
        let init = if parser.check(TokenKind::LeftBrace) {
            parse_brace_initializer(parser)?
        } else {
            let expr = parse_assignment_expression(parser)?;
            Initializer::Expression(Box::new(expr))
        };
        let item_end = parser.current().span;
        let item_span = Span::merge(item_start, item_end);

        items.push(InitializerItem {
            designators,
            initializer: init,
            span: item_span,
        });

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    let end = parser.expect(TokenKind::RightBrace)?;
    let span = Span::merge(start, end);
    Ok(Initializer::List { items, span })
}

/// Parses a field name in a designator context (`.field`).
fn parse_designator_field(parser: &mut Parser<'_>) -> Result<Symbol, ParseError> {
    match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            parser.advance();
            Ok(sym)
        }
        _ => {
            let span = parser.current().span;
            let msg = "expected field name after '.' in designator".to_string();
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier".to_string()),
            })
        }
    }
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

/// Returns `true` if the *current* token of the parser can begin a
/// C type-name (specifier-qualifier-list followed by an optional abstract
/// declarator).
///
/// This is a narrower check than [`Parser::is_declaration_start`] because
/// type-names cannot contain storage-class specifiers, function specifiers,
/// `_Alignas`, `_Static_assert`, or `__extension__`.
fn is_type_name_start(parser: &Parser<'_>) -> bool {
    match &parser.current().kind {
        // Type specifier keywords
        TokenKind::Void
        | TokenKind::Char
        | TokenKind::Short
        | TokenKind::Int
        | TokenKind::Long
        | TokenKind::Float
        | TokenKind::Double
        | TokenKind::Signed
        | TokenKind::Unsigned
        | TokenKind::Bool
        | TokenKind::Complex => true,

        // Aggregate / enum specifiers
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // Type qualifiers
        TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict | TokenKind::Atomic => true,

        // typeof / __typeof__ (GCC extension — produces a type)
        TokenKind::TypeofKeyword => true,

        // __attribute__ can precede a type-name
        TokenKind::Attribute => true,

        // Identifier that is a registered typedef name
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),

        _ => false,
    }
}

/// Returns `true` if the token at `parser.pos + offset` can begin a
/// type-name.  This is used to peek *past* a `(` without consuming it.
fn is_type_name_start_at(parser: &Parser<'_>, offset: usize) -> bool {
    let tok = parser.peek_ahead(offset);
    match &tok.kind {
        // Type specifier keywords
        TokenKind::Void
        | TokenKind::Char
        | TokenKind::Short
        | TokenKind::Int
        | TokenKind::Long
        | TokenKind::Float
        | TokenKind::Double
        | TokenKind::Signed
        | TokenKind::Unsigned
        | TokenKind::Bool
        | TokenKind::Complex => true,

        // Aggregate / enum specifiers
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // Type qualifiers
        TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict | TokenKind::Atomic => true,

        // typeof / __typeof__
        TokenKind::TypeofKeyword => true,

        // __attribute__
        TokenKind::Attribute => true,

        // Typedef name
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),

        _ => false,
    }
}

/// Extracts the span from an `Initializer` value.
fn initializer_span(init: &Initializer) -> Span {
    match init {
        Initializer::Expression(expr) => expr.span(),
        Initializer::List { span, .. } => *span,
    }
}
