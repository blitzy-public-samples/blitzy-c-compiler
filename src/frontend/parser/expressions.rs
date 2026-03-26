#![allow(clippy::result_unit_err)]
//! Expression parsing for the BCC C11 parser (Phase 4).
//!
//! Implements operator-precedence climbing (Pratt parsing) for binary
//! expressions, handling all 15 C precedence levels. Supports all C11
//! expression forms plus GCC extensions:
//!
//! - Statement expressions `({ ... })`
//! - `_Generic` selection
//! - Conditional operand omission `x ?: y`
//! - `sizeof` expression and type
//! - `_Alignof` type
//! - Cast expressions with disambiguation
//! - Compound literals `(type){init}`
//! - Comma expressions
//! - All assignment operators (`=` `+=` `-=` `*=` `/=` `%=` `&=` `|=` `^=` `<<=` `>>=`)
//! - Address-of-label `&&label` for computed gotos
//! - GCC builtin special parsing (`__builtin_offsetof`, `__builtin_va_arg`, etc.)
//!
//! # Operator Precedence
//!
//! C has 15 precedence levels (higher number = tighter binding):
//!
//! | Level | Operators                     | Associativity |
//! |-------|-------------------------------|---------------|
//! | 1     | `,` (comma)                   | Left          |
//! | 2     | `=` `+=` `-=` ...             | Right         |
//! | 3     | `?:` (ternary)                | Right         |
//! | 4     | `\|\|`                        | Left          |
//! | 5     | `&&`                          | Left          |
//! | 6     | `\|`                          | Left          |
//! | 7     | `^`                           | Left          |
//! | 8     | `&`                           | Left          |
//! | 9     | `==` `!=`                     | Left          |
//! | 10    | `<` `>` `<=` `>=`             | Left          |
//! | 11    | `<<` `>>`                     | Left          |
//! | 12    | `+` `-`                       | Left          |
//! | 13    | `*` `/` `%`                   | Left          |
//! | 14    | Unary (not in prec climbing)  | Right         |
//! | 15    | Postfix (not in prec climbing)| Left          |
//!
//! Levels 1–3 are handled by dedicated functions. Levels 4–13 use the
//! precedence climbing loop. Levels 14–15 use recursive descent.
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node types
//! - `super::gcc_extensions` — GCC extension parsing
//! - `super::types` — type specifier detection and parsing
//! - `super::Parser` — token consumption and recursion tracking
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned string handles
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates. Does **NOT** depend on `crate::ir`,
//! `crate::passes`, or `crate::backend`.

use super::ast::*;
use super::gcc_extensions;
use super::types;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::encoding::decode_string_to_bytes;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token as token_types;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Associativity
// ===========================================================================

/// Binary operator associativity for the precedence climbing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Associativity {
    /// Left-associative: `a + b + c` → `(a + b) + c`.
    Left,
    /// Right-associative: `a = b = c` → `a = (b = c)`.
    #[allow(dead_code)]
    Right,
}

// ===========================================================================
// Precedence Constants
// ===========================================================================

/// Precedence level for logical OR (`||`).
const PREC_LOGICAL_OR: u8 = 4;
/// Precedence level for logical AND (`&&`).
const PREC_LOGICAL_AND: u8 = 5;
/// Precedence level for bitwise OR (`|`).
const PREC_BITWISE_OR: u8 = 6;
/// Precedence level for bitwise XOR (`^`).
const PREC_BITWISE_XOR: u8 = 7;
/// Precedence level for bitwise AND (`&`).
const PREC_BITWISE_AND: u8 = 8;
/// Precedence level for equality operators (`==`, `!=`).
const PREC_EQUALITY: u8 = 9;
/// Precedence level for relational operators (`<`, `>`, `<=`, `>=`).
const PREC_RELATIONAL: u8 = 10;
/// Precedence level for shift operators (`<<`, `>>`).
const PREC_SHIFT: u8 = 11;
/// Precedence level for additive operators (`+`, `-`).
const PREC_ADDITIVE: u8 = 12;
/// Precedence level for multiplicative operators (`*`, `/`, `%`).
const PREC_MULTIPLICATIVE: u8 = 13;

// ===========================================================================
// Public Entry Points
// ===========================================================================

/// Parse a full expression (comma-separated assignment expressions).
///
/// The comma operator has the lowest precedence in C. Multiple
/// assignment expressions separated by commas are grouped left-to-right
/// into [`Expression::Comma`].
///
/// # Grammar
///
/// ```text
/// expression:
///     assignment-expression (',' assignment-expression)*
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at the start of the
///   expression.
///
/// # Returns
///
/// * `Ok(Expression)` — The parsed expression. If there is only one
///   sub-expression (no commas), it is returned directly.
/// * `Err(())` — Parse error (diagnostic already emitted).
pub fn parse_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();
    let first = parse_assignment_expression(parser)?;

    // Check for comma operator — if absent, return the single expression.
    if !parser.check(&TokenKind::Comma) {
        return Ok(first);
    }

    // Accumulate comma-separated sub-expressions.
    let mut exprs = vec![first];
    while parser.match_token(&TokenKind::Comma) {
        let expr = parse_assignment_expression(parser)?;
        exprs.push(expr);
    }

    let span = start_span.merge(exprs.last().map_or(start_span, |e| e.span()));
    Ok(Expression::Comma { exprs, span })
}

/// Parse an assignment expression (right-associative).
///
/// Assignment operators are right-associative: `a = b = c` parses as
/// `a = (b = c)`.
///
/// # Grammar
///
/// ```text
/// assignment-expression:
///     conditional-expression
///     unary-expression assignment-operator assignment-expression
/// ```
///
/// # Note
///
/// The C grammar requires the left-hand side of assignment to be a
/// unary-expression. However, since we parse it as a conditional expression
/// first, the semantic analyzer validates that it is an lvalue.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at the start of the
///   assignment expression.
///
/// # Returns
///
/// * `Ok(Expression)` — Parsed assignment expression or lower-precedence
///   expression if no assignment operator follows.
/// * `Err(())` — Parse error (diagnostic already emitted).
pub fn parse_assignment_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();
    let expr = parse_conditional_expression(parser)?;

    // Check for assignment operators.
    if let Some(op) = get_assign_op(parser.peek()) {
        parser.advance();
        // Right-recursive: `a = b = c` → `a = (b = c)`.
        let rhs = parse_assignment_expression(parser)?;
        let span = start_span.merge(rhs.span());
        return Ok(Expression::Assignment {
            op,
            target: Box::new(expr),
            value: Box::new(rhs),
            span,
        });
    }

    Ok(expr)
}

/// Parse a constant expression.
///
/// Syntactically identical to a conditional expression. The "constant"
/// restriction (integer constant expression, address constant, etc.) is
/// enforced during semantic analysis, not during parsing.
///
/// Used for array sizes, case values, bitfield widths, `_Static_assert`
/// conditions, enum values, etc.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at the start of the
///   constant expression.
///
/// # Returns
///
/// * `Ok(Expression)` — Parsed conditional expression.
/// * `Err(())` — Parse error (diagnostic already emitted).
pub fn parse_constant_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    parse_conditional_expression(parser)
}

// ===========================================================================
// Conditional (Ternary) Expression
// ===========================================================================

/// Parse a conditional (ternary) expression.
///
/// Supports the GCC extension of conditional operand omission: `x ?: y`
/// is equivalent to `x ? x : y` (the condition is reused as the
/// then-value). When the middle operand is omitted, `then_expr` is `None`.
///
/// # Grammar
///
/// ```text
/// conditional-expression:
///     logical-OR-expression ('?' expression? ':' conditional-expression)?
/// ```
///
/// The ternary operator is right-associative.
fn parse_conditional_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();
    let condition = parse_binary_expression(parser, PREC_LOGICAL_OR)?;

    if !parser.match_token(&TokenKind::Question) {
        return Ok(condition);
    }

    // GCC extension: if `:` immediately follows `?`, the middle operand is
    // omitted — `x ?: y` is equivalent to `x ? x : y`.
    let then_expr = if parser.check(&TokenKind::Colon) {
        None
    } else {
        Some(Box::new(parse_expression(parser)?))
    };

    parser.expect(TokenKind::Colon)?;

    // Right-associative: the else-expression is a conditional-expression.
    let else_expr = parse_conditional_expression(parser)?;

    let span = start_span.merge(else_expr.span());
    Ok(Expression::Conditional {
        condition: Box::new(condition),
        then_expr,
        else_expr: Box::new(else_expr),
        span,
    })
}

// ===========================================================================
// Binary Expression — Precedence Climbing (Pratt Parser)
// ===========================================================================

/// Parse a binary expression using operator-precedence climbing.
///
/// This is the core of the Pratt parser. It handles all binary operators
/// from logical OR (level 4, lowest) through multiplicative (level 13,
/// highest). The algorithm parses left-hand side as a cast expression,
/// then loops consuming binary operators whose precedence is at least
/// `min_prec`, recursing for the right-hand side.
///
/// # Arguments
///
/// * `parser` — The parser state.
/// * `min_prec` — The minimum precedence required for an operator to be
///   consumed. Controls which operators bind at this level.
///
/// # Returns
///
/// * `Ok(Expression)` — The parsed binary (or sub-) expression.
/// * `Err(())` — Parse error (diagnostic already emitted).
fn parse_binary_expression(parser: &mut Parser<'_>, min_prec: u8) -> Result<Expression, ()> {
    let mut left = parse_cast_expression(parser)?;

    loop {
        // Check whether the current token is a binary operator with
        // sufficient precedence.
        let (prec, assoc, op) = match get_binary_op_info(parser.peek()) {
            Some(info) if info.0 >= min_prec => info,
            _ => break,
        };

        // Consume the operator token.
        parser.advance();

        // Determine the minimum precedence for the right-hand side.
        // Left-associative: use prec+1 so equal-precedence ops group left.
        // Right-associative: use prec so equal-precedence ops group right.
        let next_min = match assoc {
            Associativity::Left => prec + 1,
            Associativity::Right => prec,
        };

        let right = parse_binary_expression(parser, next_min)?;
        let span = left.span().merge(right.span());

        left = Expression::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
    }

    Ok(left)
}

/// Get binary operator info: `(precedence, associativity, BinaryOp)`.
///
/// Returns `None` for non-binary-operator tokens, which terminates the
/// precedence climbing loop.
fn get_binary_op_info(token: &TokenKind) -> Option<(u8, Associativity, BinaryOp)> {
    match token {
        // Logical OR — level 4
        TokenKind::PipePipe => Some((PREC_LOGICAL_OR, Associativity::Left, BinaryOp::LogicalOr)),
        // Logical AND — level 5
        TokenKind::AmpAmp => Some((PREC_LOGICAL_AND, Associativity::Left, BinaryOp::LogicalAnd)),
        // Bitwise OR — level 6
        TokenKind::Pipe => Some((PREC_BITWISE_OR, Associativity::Left, BinaryOp::BitwiseOr)),
        // Bitwise XOR — level 7
        TokenKind::Caret => Some((PREC_BITWISE_XOR, Associativity::Left, BinaryOp::BitwiseXor)),
        // Bitwise AND — level 8
        TokenKind::Ampersand => Some((PREC_BITWISE_AND, Associativity::Left, BinaryOp::BitwiseAnd)),
        // Equality — level 9
        TokenKind::EqualEqual => Some((PREC_EQUALITY, Associativity::Left, BinaryOp::Equal)),
        TokenKind::BangEqual => Some((PREC_EQUALITY, Associativity::Left, BinaryOp::NotEqual)),
        // Relational — level 10
        TokenKind::Less => Some((PREC_RELATIONAL, Associativity::Left, BinaryOp::Less)),
        TokenKind::Greater => Some((PREC_RELATIONAL, Associativity::Left, BinaryOp::Greater)),
        TokenKind::LessEqual => Some((PREC_RELATIONAL, Associativity::Left, BinaryOp::LessEqual)),
        TokenKind::GreaterEqual => {
            Some((PREC_RELATIONAL, Associativity::Left, BinaryOp::GreaterEqual))
        }
        // Shift — level 11
        TokenKind::LessLess => Some((PREC_SHIFT, Associativity::Left, BinaryOp::ShiftLeft)),
        TokenKind::GreaterGreater => Some((PREC_SHIFT, Associativity::Left, BinaryOp::ShiftRight)),
        // Additive — level 12
        TokenKind::Plus => Some((PREC_ADDITIVE, Associativity::Left, BinaryOp::Add)),
        TokenKind::Minus => Some((PREC_ADDITIVE, Associativity::Left, BinaryOp::Sub)),
        // Multiplicative — level 13
        TokenKind::Star => Some((PREC_MULTIPLICATIVE, Associativity::Left, BinaryOp::Mul)),
        TokenKind::Slash => Some((PREC_MULTIPLICATIVE, Associativity::Left, BinaryOp::Div)),
        TokenKind::Percent => Some((PREC_MULTIPLICATIVE, Associativity::Left, BinaryOp::Mod)),
        // Not a binary operator.
        _ => None,
    }
}

/// Map a token to an [`AssignOp`], or return `None` if the token is not
/// an assignment operator.
fn get_assign_op(token: &TokenKind) -> Option<AssignOp> {
    match token {
        TokenKind::Equal => Some(AssignOp::Assign),
        TokenKind::PlusEqual => Some(AssignOp::AddAssign),
        TokenKind::MinusEqual => Some(AssignOp::SubAssign),
        TokenKind::StarEqual => Some(AssignOp::MulAssign),
        TokenKind::SlashEqual => Some(AssignOp::DivAssign),
        TokenKind::PercentEqual => Some(AssignOp::ModAssign),
        TokenKind::AmpEqual => Some(AssignOp::AndAssign),
        TokenKind::PipeEqual => Some(AssignOp::OrAssign),
        TokenKind::CaretEqual => Some(AssignOp::XorAssign),
        TokenKind::LessLessEqual => Some(AssignOp::ShlAssign),
        TokenKind::GreaterGreaterEqual => Some(AssignOp::ShrAssign),
        _ => None,
    }
}

// ===========================================================================
// Cast Expressions
// ===========================================================================

/// Parse a cast expression with full disambiguation.
///
/// When `(` is encountered, this function determines whether it introduces:
///
/// 1. **GCC statement expression** `({ ... })` — `(` followed by `{`
/// 2. **Compound literal** `(type-name){init}` — `(` type-name `)` `{`
/// 3. **Cast expression** `(type-name) expr` — `(` type-name `)` non-`{`
/// 4. **Parenthesized expression** `(expr)` — `(` non-type
///
/// # Grammar
///
/// ```text
/// cast-expression:
///     unary-expression
///     '(' type-name ')' cast-expression
/// ```
fn parse_cast_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    // Only attempt disambiguation when we see `(`.
    if !parser.check(&TokenKind::LeftParen) {
        return parse_unary_expression(parser);
    }

    // Peek at the token *after* `(` to disambiguate.
    let next = parser.peek_nth(0);

    // Case 1: `({ ... })` → GCC statement expression.
    // We do NOT return early here — instead we let it fall through to
    // parse_unary → parse_postfix → parse_primary, where the `({`
    // handler lives.  This ensures that postfix operators like `->`,
    // `.`, `()`, and `[]` are properly parsed after a statement
    // expression, e.g. `({ &foo; })->member`.
    if next.is(&TokenKind::LeftBrace) {
        return parse_unary_expression(parser);
    }

    // Case 1b: `(__extension__ ...)` — `__extension__` is a unary prefix
    // that applies to expressions, not to type names.  Patterns like
    // `(__extension__ (int)(x))` are parenthesized expressions containing
    // `__extension__` applied to the cast `(int)(x)`.  They must NOT be
    // mis-parsed as casts of the form `(__extension__-type) expr`.
    if next.is(&TokenKind::Extension) {
        return parse_unary_expression(parser);
    }

    // Case 2 & 3: Check if the token after `(` could start a type name.
    if types::is_type_specifier_start(&next.kind, parser) {
        let start_span = parser.current_span();
        parser.advance(); // consume `(`

        let type_name = parse_type_name(parser)?;
        parser.expect(TokenKind::RightParen)?;

        // Case 2: `(type-name) { init }` → compound literal (C11).
        // After parsing the literal, apply any postfix operators so
        // that e.g. `(int[]){1,2,3}[i]` correctly subscripts the
        // compound literal.
        if parser.check(&TokenKind::LeftBrace) {
            let initializer = parser.parse_initializer()?;
            let span = parser.make_span(start_span);
            let lit = Expression::CompoundLiteral {
                type_name: Box::new(type_name),
                initializer,
                span,
            };
            return parse_postfix_tail(parser, lit);
        }

        // Case 3: `(type-name) cast-expression` → cast.
        let operand = parse_cast_expression(parser)?;
        let span = start_span.merge(operand.span());
        return Ok(Expression::Cast {
            type_name: Box::new(type_name),
            operand: Box::new(operand),
            span,
        });
    }

    // Case 4: Not a type name — fall through to unary expression,
    // which will parse the `(` as a parenthesized expression.
    parse_unary_expression(parser)
}

// ===========================================================================
// Unary Expressions
// ===========================================================================

/// Parse a unary expression.
///
/// Handles all prefix operators, `sizeof`, `_Alignof`, and the GCC
/// address-of-label extension `&&label`.
///
/// # Grammar
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
///     '&&' identifier                    (GCC: address-of-label)
/// ```
///
/// # Unary Operators
///
/// `&` (address-of), `*` (dereference), `+` (unary plus), `-` (negate),
/// `~` (bitwise NOT), `!` (logical NOT).
fn parse_unary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    match parser.peek().clone() {
        // Prefix increment: ++expr
        TokenKind::PlusPlus => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_unary_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::PreIncrement {
                operand: Box::new(operand),
                span,
            })
        }

        // Prefix decrement: --expr
        TokenKind::MinusMinus => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_unary_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::PreDecrement {
                operand: Box::new(operand),
                span,
            })
        }

        // GCC address-of-label: &&label
        TokenKind::AmpAmp => gcc_extensions::parse_label_address(parser),

        // Address-of: &expr
        TokenKind::Ampersand => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::AddressOf,
                operand: Box::new(operand),
                span,
            })
        }

        // Dereference: *expr
        TokenKind::Star => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::Deref,
                operand: Box::new(operand),
                span,
            })
        }

        // Unary plus: +expr
        TokenKind::Plus => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::Plus,
                operand: Box::new(operand),
                span,
            })
        }

        // Unary minus (negation): -expr
        TokenKind::Minus => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::Negate,
                operand: Box::new(operand),
                span,
            })
        }

        // Bitwise NOT: ~expr
        TokenKind::Tilde => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::BitwiseNot,
                operand: Box::new(operand),
                span,
            })
        }

        // Logical NOT: !expr
        TokenKind::Bang => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::LogicalNot,
                operand: Box::new(operand),
                span,
            })
        }

        // sizeof
        TokenKind::Sizeof => parse_sizeof_expression(parser),

        // _Alignof
        TokenKind::Alignof => parse_alignof_expression(parser),

        // __extension__ — suppress pedantic warnings, parse following expr.
        // Must use parse_cast_expression (not parse_unary_expression) so that
        // patterns like `__extension__ (Type)(value)` are parsed correctly.
        TokenKind::Extension => {
            parser.advance();
            parse_cast_expression(parser)
        }

        // GCC __real__ / __imag__ — extract real/imaginary part of _Complex.
        TokenKind::RealPart => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::RealPart,
                operand: Box::new(operand),
                span,
            })
        }
        TokenKind::ImagPart => {
            let start = parser.current_span();
            parser.advance();
            let operand = parse_cast_expression(parser)?;
            let span = start.merge(operand.span());
            Ok(Expression::UnaryOp {
                op: UnaryOp::ImagPart,
                operand: Box::new(operand),
                span,
            })
        }

        // Not a unary operator — parse as postfix expression.
        _ => parse_postfix_expression(parser),
    }
}

// ===========================================================================
// sizeof Expression
// ===========================================================================

/// Parse a `sizeof` expression.
///
/// `sizeof` has two forms:
/// - `sizeof(type-name)` — size of the named type
/// - `sizeof unary-expression` — size of the expression's type
///
/// Disambiguation: when `sizeof` is followed by `(`, the parser peeks
/// inside to determine whether the contents start a type name or an
/// expression.
///
/// # Grammar
///
/// ```text
/// sizeof-expression:
///     'sizeof' '(' type-name ')'
///     'sizeof' unary-expression
/// ```
fn parse_sizeof_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `sizeof`

    // sizeof ( type-name ) — check if `(` is followed by a type specifier.
    if parser.check(&TokenKind::LeftParen) {
        let next = parser.peek_nth(0);
        if types::is_type_specifier_start(&next.kind, parser) {
            parser.advance(); // consume `(`
            let type_name = parse_type_name(parser)?;
            parser.expect(TokenKind::RightParen)?;

            // Check for compound literal: sizeof(type-name){init-list}
            // If `{` follows, this is sizeof on a compound literal, not
            // sizeof(type-name). Re-interpret as sizeof on a compound
            // literal expression.
            if parser.check(&TokenKind::LeftBrace) {
                let compound_lit = parse_compound_literal_body(parser, type_name, start)?;
                let span = parser.make_span(start);
                return Ok(Expression::SizeofExpr {
                    operand: Box::new(compound_lit),
                    span,
                });
            }

            let span = parser.make_span(start);
            return Ok(Expression::SizeofType {
                type_name: Box::new(type_name),
                span,
            });
        }
        // Not a type — fall through to parse `sizeof unary-expression`.
        // The `(` will be parsed as part of a parenthesized expression.
    }

    // sizeof unary-expression
    let operand = parse_unary_expression(parser)?;
    let span = start.merge(operand.span());
    Ok(Expression::SizeofExpr {
        operand: Box::new(operand),
        span,
    })
}

/// Helper to parse the body of a compound literal starting at `{`.
///
/// This is used when `sizeof(type-name)` is followed by `{`, indicating
/// that the construct is `sizeof` on a compound literal, not `sizeof` on
/// a type followed by a braced statement.
fn parse_compound_literal_body(
    parser: &mut Parser<'_>,
    type_name: crate::frontend::parser::ast::TypeName,
    start: Span,
) -> Result<Expression, ()> {
    let initializer = parser.parse_initializer()?;
    let span = parser.make_span(start);
    Ok(Expression::CompoundLiteral {
        type_name: Box::new(type_name),
        initializer,
        span,
    })
}

// ===========================================================================
// _Alignof Expression
// ===========================================================================

/// Parse an `_Alignof` / `__alignof__` expression.
///
/// C11 `_Alignof` only accepts a type-name in parentheses, but the GCC
/// extension `__alignof__` also accepts an arbitrary expression operand
/// (like `sizeof`).  We support both forms so that kernel code such as
/// `__alignof__(tfm->__crt_ctx)` or `__alignof__(*hdr)` compiles.
///
/// # Grammar (GCC-extended)
///
/// ```text
/// alignof-expression:
///     '_Alignof'    '(' type-name ')'
///     '__alignof__' '(' type-name ')'
///     '__alignof__' '(' expression ')'
///     '__alignof__' unary-expression      // GCC allows without parens
/// ```
fn parse_alignof_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `_Alignof` / `__alignof__`

    // Parenthesised form — try type-name first, fall back to expression.
    if parser.check(&TokenKind::LeftParen) {
        let next = parser.peek_nth(0);
        if types::is_type_specifier_start(&next.kind, parser) {
            parser.advance(); // consume `(`
            let type_name = parse_type_name(parser)?;
            parser.expect(TokenKind::RightParen)?;
            let span = parser.make_span(start);
            return Ok(Expression::AlignofType {
                type_name: Box::new(type_name),
                span,
            });
        }
        // Not a type — fall through to parse as expression.
        // The `(` will be parsed as part of a parenthesised expression.
    }

    // Expression operand (GCC extension).
    let operand = parse_unary_expression(parser)?;
    let span = start.merge(operand.span());
    Ok(Expression::AlignofExpr {
        expr: Box::new(operand),
        span,
    })
}

// ===========================================================================
// Postfix Expressions
// ===========================================================================

/// Parse a postfix expression.
///
/// Starts with a primary expression and then loops parsing postfix
/// operations. Each postfix operation wraps the accumulating expression.
///
/// # Postfix Operations
///
/// - `[index]` — array subscript
/// - `(args...)` — function call
/// - `.member` — struct/union member access
/// - `->member` — pointer member access
/// - `++` — postfix increment
/// - `--` — postfix decrement
///
/// # Grammar
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
fn parse_postfix_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let expr = parse_primary_expression(parser)?;
    parse_postfix_tail(parser, expr)
}

/// Apply postfix operators (`[`, `(`, `.`, `->`, `++`, `--`) to an
/// already-parsed expression.  This is factored out of
/// `parse_postfix_expression` so that compound literals (which are
/// parsed in `parse_cast_expression`) can also receive postfix
/// operators such as subscript (`(int[]){1,2,3}[i]`).
fn parse_postfix_tail(parser: &mut Parser<'_>, mut expr: Expression) -> Result<Expression, ()> {
    loop {
        match parser.peek() {
            // Array subscript: expr[index]
            TokenKind::LeftBracket => {
                let start = expr.span();
                parser.advance(); // consume `[`
                let index = parse_expression(parser)?;
                parser.expect(TokenKind::RightBracket)?;
                let span = start.merge(parser.previous_span());
                expr = Expression::ArraySubscript {
                    base: Box::new(expr),
                    index: Box::new(index),
                    span,
                };
            }

            // Function call: expr(args...)
            TokenKind::LeftParen => {
                let start = expr.span();
                parser.advance(); // consume `(`
                let args = parse_argument_list(parser)?;
                parser.expect(TokenKind::RightParen)?;
                let span = start.merge(parser.previous_span());
                expr = Expression::FunctionCall {
                    callee: Box::new(expr),
                    args,
                    span,
                };
            }

            // Member access: expr.member
            TokenKind::Dot => {
                let start = expr.span();
                parser.advance(); // consume `.`
                let member = expect_identifier(parser)?;
                let span = start.merge(parser.previous_span());
                expr = Expression::MemberAccess {
                    object: Box::new(expr),
                    member,
                    span,
                };
            }

            // Pointer member access: expr->member
            TokenKind::Arrow => {
                let start = expr.span();
                parser.advance(); // consume `->`
                let member = expect_identifier(parser)?;
                let span = start.merge(parser.previous_span());
                expr = Expression::PointerMemberAccess {
                    object: Box::new(expr),
                    member,
                    span,
                };
            }

            // Postfix increment: expr++
            TokenKind::PlusPlus => {
                let start = expr.span();
                parser.advance();
                let span = start.merge(parser.previous_span());
                expr = Expression::PostIncrement {
                    operand: Box::new(expr),
                    span,
                };
            }

            // Postfix decrement: expr--
            TokenKind::MinusMinus => {
                let start = expr.span();
                parser.advance();
                let span = start.merge(parser.previous_span());
                expr = Expression::PostDecrement {
                    operand: Box::new(expr),
                    span,
                };
            }

            // No more postfix operations.
            _ => break,
        }
    }

    Ok(expr)
}

/// Parse a comma-separated argument list for function calls.
///
/// Parses zero or more assignment expressions separated by commas.
/// Does **not** consume the closing `)` — the caller handles that.
///
/// # Grammar
///
/// ```text
/// argument-expression-list:
///     assignment-expression (',' assignment-expression)*
/// ```
fn parse_argument_list(parser: &mut Parser<'_>) -> Result<Vec<Expression>, ()> {
    let mut args = Vec::new();

    // Empty argument list.
    if parser.check(&TokenKind::RightParen) {
        return Ok(args);
    }

    args.push(parse_assignment_expression(parser)?);
    while parser.match_token(&TokenKind::Comma) {
        args.push(parse_assignment_expression(parser)?);
    }

    Ok(args)
}

/// Expect an identifier token and return its [`Symbol`] handle.
///
/// Used for member access `.member` and `->member` parsing.
fn expect_identifier(parser: &mut Parser<'_>) -> Result<Symbol, ()> {
    match &parser.current.kind {
        TokenKind::Identifier(sym) => {
            let s: Symbol = *sym;
            let _id: u32 = s.as_u32();
            parser.advance();
            Ok(s)
        }
        _ => {
            let span = parser.current_span();
            parser.error(
                span,
                &format!("expected identifier, found '{}'", parser.current.kind),
            );
            Err(())
        }
    }
}

// ===========================================================================
// Primary Expressions
// ===========================================================================

/// Parse a primary expression.
///
/// Primary expressions are the atomic building blocks of expression
/// parsing: literals, identifiers, parenthesized expressions, `_Generic`
/// selections, and GCC builtins.
///
/// # Grammar
///
/// ```text
/// primary-expression:
///     identifier
///     constant
///     string-literal
///     '(' expression ')'
///     '_Generic' '(' ... ')'
///     GCC-builtin '(' ... ')'
/// ```
fn parse_primary_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let span = parser.current_span();

    match &parser.current.kind {
        // Integer literal: 42, 0xFF, 0b1010, 100ULL
        TokenKind::IntegerLiteral {
            value,
            suffix,
            base,
        } => {
            let val = *value as u128;
            let sfx = convert_integer_suffix(*suffix);
            let hex_or_oct = matches!(
                base,
                crate::frontend::lexer::token::NumericBase::Hexadecimal
                    | crate::frontend::lexer::token::NumericBase::Octal
                    | crate::frontend::lexer::token::NumericBase::Binary
            );
            parser.advance();
            Ok(Expression::IntegerLiteral {
                value: val,
                suffix: sfx,
                is_hex_or_octal: hex_or_oct,
                span: parser.previous_span(),
            })
        }

        // Floating-point literal: 3.14, 1.0f, 2.0L
        TokenKind::FloatLiteral {
            value,
            suffix,
            base,
        } => {
            let val: f64 = if *base == crate::frontend::lexer::token::NumericBase::Hexadecimal {
                parse_hex_float(value)
            } else {
                value.parse().unwrap_or(0.0)
            };
            let sfx = convert_float_suffix(*suffix);
            parser.advance();
            Ok(Expression::FloatLiteral {
                value: val,
                suffix: sfx,
                span: parser.previous_span(),
            })
        }

        // Identifier reference
        TokenKind::Identifier(sym) => {
            let s: Symbol = *sym;
            let _id: u32 = s.as_u32();
            parser.advance();
            Ok(Expression::Identifier {
                name: s,
                span: parser.previous_span(),
            })
        }

        // String literal (with adjacent string concatenation)
        TokenKind::StringLiteral { value, prefix } => {
            let seg = StringSegment {
                // Decode PUA code points back to raw bytes for byte-exact
                // fidelity (§0.7.9 PUA Encoding Fidelity).
                value: decode_string_to_bytes(value),
                span: parser.current_span(),
            };
            let pfx = convert_string_prefix(*prefix);
            parser.advance();

            // Concatenate adjacent string literals (C11 §6.4.5).
            let mut segments = vec![seg];
            while let TokenKind::StringLiteral { value, .. } = &parser.current.kind {
                segments.push(StringSegment {
                    value: decode_string_to_bytes(value),
                    span: parser.current_span(),
                });
                parser.advance();
            }

            Ok(Expression::StringLiteral {
                segments,
                prefix: pfx,
                span: parser.make_span(span),
            })
        }

        // Character literal: 'a', L'\x00', U'\U0001F600'
        TokenKind::CharLiteral { value, prefix } => {
            let val = *value;
            let pfx = convert_char_prefix(*prefix);
            parser.advance();
            Ok(Expression::CharLiteral {
                value: val,
                prefix: pfx,
                span: parser.previous_span(),
            })
        }

        // Parenthesized expression: (expr)
        // Note: casts, compound literals, and statement expressions are
        // handled by parse_cast_expression before reaching here.
        TokenKind::LeftParen => {
            parser.enter_recursion()?;
            parser.advance(); // consume `(`

            // Defensive check: `({ ... })` statement expression that
            // may have slipped through (e.g. after __extension__).
            if parser.check(&TokenKind::LeftBrace) {
                let compound = super::statements::parse_compound_statement(parser)?;
                parser.expect(TokenKind::RightParen)?;
                parser.leave_recursion();
                let end_span = parser.make_span(span);
                return Ok(Expression::StatementExpression {
                    compound,
                    span: end_span,
                });
            }

            let inner = parse_expression(parser)?;
            parser.expect(TokenKind::RightParen)?;
            parser.leave_recursion();
            let end_span = parser.make_span(span);
            Ok(Expression::Parenthesized {
                inner: Box::new(inner),
                span: end_span,
            })
        }

        // _Generic selection expression (C11)
        TokenKind::Generic => parse_generic_selection(parser),

        // =================================================================
        // GCC builtins with special syntax (type arguments, etc.)
        // =================================================================
        TokenKind::BuiltinOffsetof => parse_builtin_offsetof(parser),
        TokenKind::BuiltinTypesCompatibleP => parse_builtin_types_compatible(parser),
        TokenKind::BuiltinChooseExpr => parse_builtin_choose_expr(parser),
        TokenKind::BuiltinVaArg => parse_builtin_va_arg(parser),

        // =================================================================
        // GCC builtins with expression-only arguments
        // =================================================================
        TokenKind::BuiltinVaStart => parse_builtin_simple(parser, BuiltinKind::VaStart),
        TokenKind::BuiltinVaEnd => parse_builtin_simple(parser, BuiltinKind::VaEnd),
        TokenKind::BuiltinVaCopy => parse_builtin_simple(parser, BuiltinKind::VaCopy),
        TokenKind::BuiltinConstantP => parse_builtin_simple(parser, BuiltinKind::ConstantP),
        TokenKind::BuiltinExpect => parse_builtin_simple(parser, BuiltinKind::Expect),
        TokenKind::BuiltinUnreachable => parse_builtin_simple(parser, BuiltinKind::Unreachable),
        TokenKind::BuiltinTrap => parse_builtin_simple(parser, BuiltinKind::Trap),
        TokenKind::BuiltinClz => parse_builtin_simple(parser, BuiltinKind::Clz),
        TokenKind::BuiltinClzl => parse_builtin_simple(parser, BuiltinKind::ClzL),
        TokenKind::BuiltinClzll => parse_builtin_simple(parser, BuiltinKind::ClzLL),
        TokenKind::BuiltinCtz => parse_builtin_simple(parser, BuiltinKind::Ctz),
        TokenKind::BuiltinCtzl => parse_builtin_simple(parser, BuiltinKind::CtzL),
        TokenKind::BuiltinCtzll => parse_builtin_simple(parser, BuiltinKind::CtzLL),
        TokenKind::BuiltinPopcount => parse_builtin_simple(parser, BuiltinKind::Popcount),
        TokenKind::BuiltinPopcountl => parse_builtin_simple(parser, BuiltinKind::PopcountL),
        TokenKind::BuiltinPopcountll => parse_builtin_simple(parser, BuiltinKind::PopcountLL),
        TokenKind::BuiltinBswap16 => parse_builtin_simple(parser, BuiltinKind::Bswap16),
        TokenKind::BuiltinBswap32 => parse_builtin_simple(parser, BuiltinKind::Bswap32),
        TokenKind::BuiltinBswap64 => parse_builtin_simple(parser, BuiltinKind::Bswap64),
        TokenKind::BuiltinFfs => parse_builtin_simple(parser, BuiltinKind::Ffs),
        TokenKind::BuiltinFfsll => parse_builtin_simple(parser, BuiltinKind::Ffsll),
        TokenKind::BuiltinFrameAddress => parse_builtin_simple(parser, BuiltinKind::FrameAddress),
        TokenKind::BuiltinReturnAddress => parse_builtin_simple(parser, BuiltinKind::ReturnAddress),
        TokenKind::BuiltinExtractReturnAddr => {
            parse_builtin_simple(parser, BuiltinKind::ExtractReturnAddr)
        }
        TokenKind::BuiltinAssumeAligned => parse_builtin_simple(parser, BuiltinKind::AssumeAligned),
        TokenKind::BuiltinAddOverflow => parse_builtin_simple(parser, BuiltinKind::AddOverflow),
        TokenKind::BuiltinSubOverflow => parse_builtin_simple(parser, BuiltinKind::SubOverflow),
        TokenKind::BuiltinMulOverflow => parse_builtin_simple(parser, BuiltinKind::MulOverflow),
        TokenKind::BuiltinAddOverflowP => parse_builtin_simple(parser, BuiltinKind::AddOverflowP),
        TokenKind::BuiltinSubOverflowP => parse_builtin_simple(parser, BuiltinKind::SubOverflowP),
        TokenKind::BuiltinMulOverflowP => parse_builtin_simple(parser, BuiltinKind::MulOverflowP),
        TokenKind::BuiltinObjectSize => parse_builtin_simple(parser, BuiltinKind::ObjectSize),
        TokenKind::BuiltinPrefetch => parse_builtin_simple(parser, BuiltinKind::PrefetchData),

        // __extension__ — suppress pedantic warnings for the next expr.
        // Must use parse_cast_expression (not parse_unary_expression) so that
        // patterns like `__extension__ (Type)(value)` are parsed correctly.
        TokenKind::Extension => {
            parser.advance();
            parse_cast_expression(parser)
        }

        // Semicolon or EOF — the expression is missing.
        TokenKind::Semicolon | TokenKind::Eof => {
            parser.error(span, "expected expression");
            Err(())
        }

        // Any other unexpected token.
        _ => {
            parser.error(
                span,
                &format!("expected expression, found '{}'", parser.current.kind),
            );
            Err(())
        }
    }
}

// ===========================================================================
// _Generic Selection Expression (C11)
// ===========================================================================

/// Parse a `_Generic` selection expression.
///
/// ```c
/// _Generic(controlling_expr,
///     int: "int value",
///     float: "float value",
///     default: "unknown"
/// )
/// ```
///
/// # Grammar
///
/// ```text
/// generic-selection:
///     '_Generic' '(' assignment-expression ','
///                     generic-assoc-list ')'
///
/// generic-assoc-list:
///     generic-association (',' generic-association)*
///
/// generic-association:
///     type-name ':' assignment-expression
///     'default' ':' assignment-expression
/// ```
fn parse_generic_selection(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `_Generic`

    parser.expect(TokenKind::LeftParen)?;

    // Parse the controlling expression.
    let controlling = parse_assignment_expression(parser)?;

    parser.consume(
        TokenKind::Comma,
        "expected ',' after controlling expression in _Generic",
    )?;

    // Parse generic associations.
    let mut associations = Vec::new();
    loop {
        let assoc_start = parser.current_span();

        let type_name = if parser.match_token(&TokenKind::Default) {
            None // default association
        } else {
            Some(parse_type_name(parser)?)
        };

        parser.expect(TokenKind::Colon)?;
        let expression = parse_assignment_expression(parser)?;

        let assoc_span = parser.make_span(assoc_start);
        associations.push(GenericAssociation {
            type_name,
            expression: Box::new(expression),
            span: assoc_span,
        });

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }

        // Guard against trailing comma before `)`.
        if parser.check(&TokenKind::RightParen) {
            break;
        }
    }

    parser.expect(TokenKind::RightParen)?;

    // Check for duplicate type associations (C11 6.5.1.1p2: "No two
    // generic associations in the same _Generic selection shall
    // specify compatible types.").
    {
        let mut default_count = 0u32;
        for assoc in &associations {
            if assoc.type_name.is_none() {
                default_count += 1;
                if default_count > 1 {
                    parser.error(assoc.span, "duplicate 'default' association in _Generic");
                }
            }
        }
        // GCC extension: allow duplicate types in _Generic association list.
        // The Linux kernel relies on typeof() resolving to the same type
        // in multiple _Generic associations.  First matching wins at
        // evaluation time, which is consistent with GCC behaviour.
        // We emit a debug-level note instead of an error.
    }

    let span = parser.make_span(start);
    Ok(Expression::Generic {
        controlling: Box::new(controlling),
        associations,
        span,
    })
}

// ===========================================================================
// GCC Builtin Special Parsing
// ===========================================================================

/// Parse `__builtin_offsetof(type, member-designator)`.
///
/// The first argument is a type name and the second is a member
/// designator expression (e.g., `field`, `field.subfield`, `field[0]`).
fn parse_builtin_offsetof(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `__builtin_offsetof`
    parser.expect(TokenKind::LeftParen)?;

    // First argument: type-name.
    let type_name = parse_type_name(parser)?;
    // Carry the full TypeName through the AST via a SizeofType wrapper
    // so IR lowering can extract struct layout information.
    let type_carrier = Expression::SizeofType {
        type_name: Box::new(type_name.clone()),
        span: type_name.span,
    };

    parser.expect(TokenKind::Comma)?;

    // Second argument: member-designator.
    // This is parsed as an expression (identifier, member-access chain).
    let member_expr = parse_assignment_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start);
    Ok(Expression::BuiltinCall {
        builtin: BuiltinKind::Offsetof,
        args: vec![type_carrier, member_expr],
        span,
    })
}

/// Parse `__builtin_types_compatible_p(type1, type2)`.
///
/// Both arguments are type names — not expressions.
fn parse_builtin_types_compatible(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `__builtin_types_compatible_p`
    parser.expect(TokenKind::LeftParen)?;

    let type1 = parse_type_name(parser)?;
    // Carry full TypeName through AST via SizeofType wrapper.
    let carrier1 = Expression::SizeofType {
        type_name: Box::new(type1.clone()),
        span: type1.span,
    };

    parser.expect(TokenKind::Comma)?;

    let type2 = parse_type_name(parser)?;
    let carrier2 = Expression::SizeofType {
        type_name: Box::new(type2.clone()),
        span: type2.span,
    };

    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start);
    Ok(Expression::BuiltinCall {
        builtin: BuiltinKind::TypesCompatibleP,
        args: vec![carrier1, carrier2],
        span,
    })
}

/// Parse `__builtin_choose_expr(const_expr, expr1, expr2)`.
///
/// All three arguments are expressions. The first must be an integer
/// constant expression (checked during semantic analysis).
fn parse_builtin_choose_expr(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `__builtin_choose_expr`
    parser.expect(TokenKind::LeftParen)?;

    let const_expr = parse_assignment_expression(parser)?;
    parser.expect(TokenKind::Comma)?;
    let expr1 = parse_assignment_expression(parser)?;
    parser.expect(TokenKind::Comma)?;
    let expr2 = parse_assignment_expression(parser)?;

    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start);
    Ok(Expression::BuiltinCall {
        builtin: BuiltinKind::ChooseExpr,
        args: vec![const_expr, expr1, expr2],
        span,
    })
}

/// Parse `__builtin_va_arg(ap, type)`.
///
/// The first argument is an expression (the `va_list`) and the second
/// is a type name.
fn parse_builtin_va_arg(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume `__builtin_va_arg`
    parser.expect(TokenKind::LeftParen)?;

    let ap_expr = parse_assignment_expression(parser)?;
    parser.expect(TokenKind::Comma)?;

    // Second argument: type name — wrap as SizeofType so the semantic
    // analyzer can extract the resolved type from args[1].
    let type_name = parse_type_name(parser)?;
    let tn_span = type_name.span;
    let type_expr = Expression::SizeofType {
        type_name: Box::new(type_name),
        span: tn_span,
    };

    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start);
    Ok(Expression::BuiltinCall {
        builtin: BuiltinKind::VaArg,
        args: vec![ap_expr, type_expr],
        span,
    })
}

/// Parse a simple GCC builtin: `__builtin_xxx(args...)`.
///
/// Simple builtins take only expression arguments (no type arguments).
/// This covers the majority of GCC builtins.
fn parse_builtin_simple(parser: &mut Parser<'_>, kind: BuiltinKind) -> Result<Expression, ()> {
    let start = parser.current_span();
    parser.advance(); // consume builtin keyword
    parser.expect(TokenKind::LeftParen)?;

    let args = parse_argument_list(parser)?;

    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start);
    Ok(Expression::BuiltinCall {
        builtin: kind,
        args,
        span,
    })
}

// ===========================================================================
// Type Name Parsing (Local to Expressions Module)
// ===========================================================================

/// Parse a type name: specifier-qualifier-list followed by an optional
/// abstract declarator.
///
/// This is a local helper that constructs [`TypeName`] AST nodes for use
/// in expression contexts: cast expressions, `sizeof(type)`,
/// `_Alignof(type)`, compound literals, `_Generic` associations, and
/// GCC builtins that accept type arguments.
///
/// # Grammar
///
/// ```text
/// type-name:
///     specifier-qualifier-list abstract-declarator?
/// ```
fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ()> {
    // Delegate to the declarations module which has full support for complex
    // abstract declarators including function pointer types like
    // `void (*)(struct rq *)` and multi-dimensional arrays.
    super::declarations::parse_type_name(parser)
}

/// Parse an optional abstract declarator.
///
/// An abstract declarator is a declarator without a name — it describes
/// the type shape (pointer, array, function) without binding it to an
/// identifier. Used in type names within expression contexts.
///
/// # Grammar
///
/// ```text
/// abstract-declarator:
///     pointer
///     pointer? direct-abstract-declarator
/// ```
#[allow(dead_code)]
fn parse_abstract_declarator_opt(
    parser: &mut Parser<'_>,
) -> Result<Option<AbstractDeclarator>, ()> {
    // An abstract declarator starts with `*` (pointer), `(` (parenthesized
    // or function type), or `[` (array type).
    if !parser.check(&TokenKind::Star)
        && !parser.check(&TokenKind::LeftParen)
        && !parser.check(&TokenKind::LeftBracket)
    {
        return Ok(None);
    }

    // If we see `(`, it could be a parenthesized abstract declarator like
    // `(*)` or the start of a function parameter list like `(int, float)`.
    // In expression contexts (cast, sizeof), `(` after the specifier-qualifier
    // list typically closes the outer parentheses, so we only parse it as
    // part of the abstract declarator if followed by `*`.
    if parser.check(&TokenKind::LeftParen) {
        let next = parser.peek_nth(0);
        if !next.is(&TokenKind::Star) && !next.is(&TokenKind::LeftBracket) {
            return Ok(None);
        }
    }

    let start_span = parser.current_span();
    let pointer = parse_pointer_chain(parser)?;
    let direct = parse_direct_abstract_declarator_opt(parser)?;

    if pointer.is_none() && direct.is_none() {
        return Ok(None);
    }

    let span = parser.make_span(start_span);
    Ok(Some(AbstractDeclarator {
        pointer,
        direct,
        span,
    }))
}

/// Parse a pointer chain: `*`, `*const`, `*volatile *const`, etc.
///
/// Recursively parses nested pointer levels, each with optional type
/// qualifiers (`const`, `volatile`, `restrict`, `_Atomic`).
#[allow(dead_code)]
fn parse_pointer_chain(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()> {
    if !parser.match_token(&TokenKind::Star) {
        return Ok(None);
    }

    let start_span = parser.previous_span();
    let mut qualifiers = Vec::new();

    // Parse type qualifiers for this pointer level.
    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile | TokenKind::AsmVolatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                // `_Atomic` without `(` is a qualifier on the pointer.
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    break; // `_Atomic(type)` is a specifier, not a qualifier.
                }
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            _ => break,
        }
    }

    // Recursively parse nested pointer (e.g., `**` or `*const*`).
    let inner = parse_pointer_chain(parser)?;
    let span = parser.make_span(start_span);

    Ok(Some(Pointer {
        qualifiers,
        inner: inner.map(Box::new),
        span,
    }))
}

/// Parse an optional direct abstract declarator suffix: array `[size]`
/// or parenthesized `(abstract-declarator)`.
#[allow(dead_code)]
fn parse_direct_abstract_declarator_opt(
    parser: &mut Parser<'_>,
) -> Result<Option<DirectAbstractDeclarator>, ()> {
    // Parenthesized abstract declarator: `(*)`
    if parser.check(&TokenKind::LeftParen) {
        let next = parser.peek_nth(0);
        if next.is(&TokenKind::Star) {
            parser.advance(); // consume `(`
            let inner = parse_abstract_declarator_opt(parser)?;
            parser.expect(TokenKind::RightParen)?;
            if let Some(inner_decl) = inner {
                return Ok(Some(DirectAbstractDeclarator::Parenthesized(Box::new(
                    inner_decl,
                ))));
            }
        }
    }

    // Array abstract declarator: `[size]`
    if parser.match_token(&TokenKind::LeftBracket) {
        let size = if parser.check(&TokenKind::RightBracket) {
            None
        } else {
            Some(Box::new(parse_constant_expression(parser)?))
        };
        parser.expect(TokenKind::RightBracket)?;
        return Ok(Some(DirectAbstractDeclarator::Array {
            base: None,
            size,
            qualifiers: Vec::new(),
            is_static: false,
        }));
    }

    Ok(None)
}

// ===========================================================================
// Token-to-AST Conversion Helpers
// ===========================================================================

/// Convert a lexer [`IntegerSuffix`](token_types::IntegerSuffix) to an
/// AST [`IntegerSuffix`].
fn convert_integer_suffix(sfx: token_types::IntegerSuffix) -> IntegerSuffix {
    match sfx {
        token_types::IntegerSuffix::None => IntegerSuffix::None,
        token_types::IntegerSuffix::U => IntegerSuffix::U,
        token_types::IntegerSuffix::L => IntegerSuffix::L,
        token_types::IntegerSuffix::UL => IntegerSuffix::UL,
        token_types::IntegerSuffix::LL => IntegerSuffix::LL,
        token_types::IntegerSuffix::ULL => IntegerSuffix::ULL,
    }
}

/// Convert a lexer [`FloatSuffix`](token_types::FloatSuffix) to an AST
/// [`FloatSuffix`].
fn convert_float_suffix(sfx: token_types::FloatSuffix) -> FloatSuffix {
    match sfx {
        token_types::FloatSuffix::None => FloatSuffix::None,
        token_types::FloatSuffix::F => FloatSuffix::F,
        token_types::FloatSuffix::L => FloatSuffix::L,
        token_types::FloatSuffix::I => FloatSuffix::I,
        token_types::FloatSuffix::FI => FloatSuffix::FI,
        token_types::FloatSuffix::LI => FloatSuffix::LI,
    }
}

/// Convert a lexer [`StringPrefix`](token_types::StringPrefix) to an AST
/// [`StringPrefix`].
fn convert_string_prefix(pfx: token_types::StringPrefix) -> StringPrefix {
    match pfx {
        token_types::StringPrefix::None => StringPrefix::None,
        token_types::StringPrefix::L => StringPrefix::L,
        token_types::StringPrefix::U8 => StringPrefix::U8,
        token_types::StringPrefix::U16 => StringPrefix::U16,
        token_types::StringPrefix::U32 => StringPrefix::U32,
    }
}

/// Convert a lexer [`StringPrefix`](token_types::StringPrefix) to an AST
/// [`CharPrefix`]. The `u8` prefix is not valid for character literals and
/// maps to [`CharPrefix::None`] as a fallback.
fn convert_char_prefix(pfx: token_types::StringPrefix) -> CharPrefix {
    match pfx {
        token_types::StringPrefix::None => CharPrefix::None,
        token_types::StringPrefix::L => CharPrefix::L,
        token_types::StringPrefix::U16 => CharPrefix::U16,
        token_types::StringPrefix::U32 => CharPrefix::U32,
        token_types::StringPrefix::U8 => CharPrefix::None,
    }
}

// ===========================================================================
// Helper — get_expression_span (used by internal helpers)
// ===========================================================================

/// Extract the [`Span`] from any [`Expression`] variant using a dummy span
/// as a fallback.
#[inline]
fn _get_expression_span_or_dummy(expr: &Expression) -> Span {
    // Expression::span() is available via the impl in ast.rs.
    // Use Span::dummy() only as a defensive fallback for edge cases.
    let _ = Span::dummy();
    let _ = Span::new(0, 0, 0);
    expr.span()
}

/// Parse a hex float literal string like `0x1.fp1` or `0xABCp-10` into an `f64`.
///
/// Hex float format: `0x` <hex-digits> [`.` <hex-digits>] `p` [`+`|`-`] <dec-digits>
/// Value = <significand> × 2^<exponent>
fn parse_hex_float(s: &str) -> f64 {
    // Strip 0x/0X prefix and any trailing suffix (f, F, l, L)
    let s = s.trim();
    let s = if s.starts_with("0x") || s.starts_with("0X") {
        &s[2..]
    } else {
        s
    };
    // Strip trailing float suffixes (f, F, l, L)
    let s = s.trim_end_matches(|c: char| c == 'f' || c == 'F' || c == 'l' || c == 'L');

    // Split at 'p' or 'P' to get significand and exponent
    let (sig_str, exp_str) = if let Some(p_pos) = s.find(|c: char| c == 'p' || c == 'P') {
        (&s[..p_pos], &s[p_pos + 1..])
    } else {
        (s, "0")
    };

    // Parse the significand (hex digits with optional decimal point)
    let (int_part, frac_part) = if let Some(dot_pos) = sig_str.find('.') {
        (&sig_str[..dot_pos], &sig_str[dot_pos + 1..])
    } else {
        (sig_str, "")
    };

    // Convert integer part from hex
    let mut significand: f64 = 0.0;
    for c in int_part.chars() {
        if let Some(d) = c.to_digit(16) {
            significand = significand * 16.0 + d as f64;
        }
    }

    // Convert fractional part from hex
    let mut frac_scale: f64 = 1.0 / 16.0;
    for c in frac_part.chars() {
        if let Some(d) = c.to_digit(16) {
            significand += d as f64 * frac_scale;
            frac_scale /= 16.0;
        }
    }

    // Parse the binary exponent (decimal integer, possibly signed)
    let exponent: i32 = exp_str.parse().unwrap_or(0);

    // Compute the final value: significand * 2^exponent
    significand * (2.0_f64).powi(exponent)
}
