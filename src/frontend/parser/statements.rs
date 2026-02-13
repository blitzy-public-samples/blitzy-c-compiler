//! Statement parsing module for the BCC C11 parser.
//!
//! Implements parsing for all C11 statement forms and the GCC extensions
//! (computed gotos, case ranges, local labels, inline assembly dispatch).
//!
//! # Architecture
//!
//! All public functions accept a `&mut Parser` and return
//! `Result<Statement, ParseError>` or `Result<BlockItem, ParseError>`.

use super::ast::{BlockItem, ForInit, Span, Statement};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public entry points
// ===========================================================================

/// Parses a single statement.
///
/// ```text
/// statement:
///     labeled-statement
///     compound-statement
///     expression-statement
///     selection-statement
///     iteration-statement
///     jump-statement
///     asm-statement
///     __label__ ...;       (GCC local labels)
/// ```
///
/// This function is the main dispatcher — it inspects the current token and
/// delegates to the appropriate sub-parser.
pub fn parse_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    parser.enter_recursion()?;

    let result = match parser.current().kind.clone() {
        // Compound statement: { ... }
        TokenKind::LeftBrace => parse_compound_statement(parser),

        // Selection: if / switch
        TokenKind::If => parse_if_statement(parser),
        TokenKind::Switch => parse_switch_statement(parser),

        // Iteration: while / do / for
        TokenKind::While => parse_while_statement(parser),
        TokenKind::Do => parse_do_while_statement(parser),
        TokenKind::For => parse_for_statement(parser),

        // Jump: goto / break / continue / return
        TokenKind::Goto => parse_goto_statement(parser),
        TokenKind::Break => parse_break_statement(parser),
        TokenKind::Continue => parse_continue_statement(parser),
        TokenKind::Return => parse_return_statement(parser),

        // Case / Default labels (inside switch)
        TokenKind::Case => parse_case_label(parser),
        TokenKind::Default => parse_default_label(parser),

        // GCC local labels: __label__ id1, id2, ...;
        TokenKind::Label => parse_local_label_decl(parser),

        // Inline assembly: asm / __asm__
        TokenKind::AsmKeyword => {
            let asm = super::inline_asm::parse_asm_statement(parser)?;
            Ok(Statement::Asm(Box::new(asm)))
        }

        // __extension__ — may precede a declaration or expression-statement
        TokenKind::Extension => {
            // Try parsing as an expression-statement (which will handle
            // the __extension__ prefix via the expression parser).
            parse_expression_statement(parser)
        }

        // Null statement: ;
        TokenKind::Semicolon => {
            let span = parser.advance();
            Ok(Statement::Null { span })
        }

        // Identifier — could be a labeled statement (`label:`) or an
        // expression-statement. Check for `:` after the identifier.
        TokenKind::Identifier(_sym) => {
            // Lookahead: if the next token is `:`, this is a labeled statement.
            if parser.peek_ahead(1).kind == TokenKind::Colon {
                parse_labeled_statement(parser)
            } else {
                parse_expression_statement(parser)
            }
        }

        // Everything else is an expression-statement.
        _ => parse_expression_statement(parser),
    };

    parser.leave_recursion();
    result
}

/// Parses a compound statement (block): `{ block-item* }`.
///
/// A compound statement contains a sequence of block items, where each item
/// is either a declaration or a statement.
///
/// GCC extensions parsed at block scope:
/// - `__label__` local label declarations (handled by `parse_statement`)
/// - Declarations mixed with statements (standard in C99+)
pub fn parse_compound_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.expect(TokenKind::LeftBrace)?;
    let mut items: Vec<BlockItem> = Vec::new();

    while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
        match parse_block_item(parser) {
            Ok(item) => items.push(item),
            Err(e) => {
                // Error recovery: report and skip to next synchronisation point.
                parser.diagnostics.error(e.span, &e.message);
                parser.synchronize();
                if parser.at_end() {
                    break;
                }
            }
        }
    }

    let end = parser.expect(TokenKind::RightBrace)?;
    let span = Span::merge(start, end);
    Ok(Statement::Compound { items, span })
}

/// Parses a block-item: either a declaration or a statement.
///
/// The disambiguating heuristic: if the current token can start a
/// declaration (storage class, type specifier, typedef name, etc.),
/// treat it as a declaration; otherwise treat it as a statement.
pub fn parse_block_item(parser: &mut Parser<'_>) -> Result<BlockItem, ParseError> {
    if parser.is_declaration_start() {
        // Some tokens are ambiguous (e.g., __extension__ can precede either
        // a declaration or an expression-statement). We try declaration first.
        match super::declarations::parse_declaration(parser) {
            Ok(decl) => Ok(BlockItem::Declaration(decl)),
            Err(_) => {
                // If declaration parsing fails, this might be an expression
                // starting with a typedef name — fall through to statement.
                let stmt = parse_statement(parser)?;
                Ok(BlockItem::Statement(stmt))
            }
        }
    } else {
        let stmt = parse_statement(parser)?;
        Ok(BlockItem::Statement(stmt))
    }
}

// ===========================================================================
// Selection statements
// ===========================================================================

/// Parses an `if` statement: `if (condition) then [else else_branch]`.
fn parse_if_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `if`
    parser.expect(TokenKind::LeftParen)?;
    let condition = super::expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    let then_branch = parse_statement(parser)?;

    let else_branch = if parser.eat(TokenKind::Else) {
        Some(Box::new(parse_statement(parser)?))
    } else {
        None
    };

    let end_span = else_branch
        .as_ref()
        .map_or(then_branch.span(), |e| e.span());
    let span = Span::merge(start, end_span);

    Ok(Statement::If {
        condition: Box::new(condition),
        then_branch: Box::new(then_branch),
        else_branch,
        span,
    })
}

/// Parses a `switch` statement: `switch (expression) body`.
fn parse_switch_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `switch`
    parser.expect(TokenKind::LeftParen)?;
    let expression = super::expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    let body = parse_statement(parser)?;
    let span = Span::merge(start, body.span());

    Ok(Statement::Switch {
        expression: Box::new(expression),
        body: Box::new(body),
        span,
    })
}

// ===========================================================================
// Iteration statements
// ===========================================================================

/// Parses a `while` statement: `while (condition) body`.
fn parse_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `while`
    parser.expect(TokenKind::LeftParen)?;
    let condition = super::expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    let body = parse_statement(parser)?;
    let span = Span::merge(start, body.span());

    Ok(Statement::While {
        condition: Box::new(condition),
        body: Box::new(body),
        span,
    })
}

/// Parses a `do ... while (condition);` statement.
fn parse_do_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `do`
    let body = parse_statement(parser)?;
    parser.expect(TokenKind::While)?;
    parser.expect(TokenKind::LeftParen)?;
    let condition = super::expressions::parse_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    Ok(Statement::DoWhile {
        body: Box::new(body),
        condition: Box::new(condition),
        span,
    })
}

/// Parses a `for` statement:
///
/// ```text
/// for ( init? ; condition? ; increment? ) body
/// ```
///
/// The initialiser may be a declaration (`for (int i = 0; ...)`) or an
/// expression.
fn parse_for_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `for`
    parser.expect(TokenKind::LeftParen)?;

    // Parse init clause (may be empty, a declaration, or an expression).
    let init = if parser.check(TokenKind::Semicolon) {
        parser.advance(); // consume `;`
        None
    } else if parser.is_declaration_start() {
        let decl = super::declarations::parse_declaration(parser)?;
        // The declaration already consumed its trailing semicolon.
        Some(ForInit::Declaration(Box::new(decl)))
    } else {
        let expr = super::expressions::parse_expression(parser)?;
        parser.expect_semicolon()?;
        Some(ForInit::Expression(Box::new(expr)))
    };

    // Parse condition (optional).
    let condition = if parser.check(TokenKind::Semicolon) {
        None
    } else {
        Some(Box::new(super::expressions::parse_expression(parser)?))
    };
    parser.expect_semicolon()?;

    // Parse increment (optional).
    let increment = if parser.check(TokenKind::RightParen) {
        None
    } else {
        Some(Box::new(super::expressions::parse_expression(parser)?))
    };
    parser.expect(TokenKind::RightParen)?;

    let body = parse_statement(parser)?;
    let span = Span::merge(start, body.span());

    Ok(Statement::For {
        init,
        condition,
        increment,
        body: Box::new(body),
        span,
    })
}

// ===========================================================================
// Jump statements
// ===========================================================================

/// Parses a `goto` statement.
///
/// Standard: `goto identifier;`
/// GCC extension: `goto *expression;` (computed goto) — dispatched to
/// [`gcc_extensions::parse_computed_goto`].
fn parse_goto_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `goto`

    // GCC computed goto: `goto *expr;`
    if parser.check(TokenKind::Star) {
        return super::gcc_extensions::parse_computed_goto(parser, start);
    }

    // Standard goto: `goto label;`
    let label = expect_identifier(parser, "goto label")?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    Ok(Statement::Goto { label, span })
}

/// Parses a `break;` statement.
fn parse_break_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `break`
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);
    Ok(Statement::Break { span })
}

/// Parses a `continue;` statement.
fn parse_continue_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `continue`
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);
    Ok(Statement::Continue { span })
}

/// Parses a `return [expression];` statement.
fn parse_return_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `return`

    let value = if parser.check(TokenKind::Semicolon) {
        None
    } else {
        Some(Box::new(super::expressions::parse_expression(parser)?))
    };

    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    Ok(Statement::Return { value, span })
}

// ===========================================================================
// Label statements
// ===========================================================================

/// Parses a labeled statement: `identifier : statement`.
///
/// The caller has already verified (via lookahead) that the current token
/// is an identifier followed by `:`.
fn parse_labeled_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let label = expect_identifier(parser, "label name")?;
    let _colon = parser.expect(TokenKind::Colon)?;

    // GCC allows attributes on labels: `label: __attribute__((unused)) stmt`
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    let body = parse_statement(parser)?;
    let span = Span::merge(
        parser
            .tokens
            .get(parser.pos.saturating_sub(2))
            .map_or(Span::DUMMY, |t| t.span),
        body.span(),
    );

    Ok(Statement::Labeled {
        label,
        attrs,
        body: Box::new(body),
        span,
    })
}

/// Parses a `case value:` label (with GCC case-range extension).
///
/// ```text
/// case constant-expression :
/// case constant-expression ... constant-expression :    (GCC extension)
/// ```
fn parse_case_label(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `case`
    let low = super::expressions::parse_conditional_expression(parser)?;

    // GCC case range: `case LOW ... HIGH:`
    if parser.check(TokenKind::Ellipsis) {
        return super::gcc_extensions::parse_case_range(parser, low, start);
    }

    parser.expect(TokenKind::Colon)?;
    let body = parse_statement(parser)?;
    let span = Span::merge(start, body.span());

    Ok(Statement::Case {
        value: Box::new(low),
        body: Box::new(body),
        span,
    })
}

/// Parses a `default:` label in a switch statement.
fn parse_default_label(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.advance(); // consume `default`
    parser.expect(TokenKind::Colon)?;
    let body = parse_statement(parser)?;
    let span = Span::merge(start, body.span());

    Ok(Statement::Default {
        body: Box::new(body),
        span,
    })
}

// ===========================================================================
// Expression statement
// ===========================================================================

/// Parses an expression-statement: `expression ;`.
fn parse_expression_statement(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let expr = super::expressions::parse_expression(parser)?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(expr.span(), end);

    Ok(Statement::Expression {
        expr: Box::new(expr),
        span,
    })
}

// ===========================================================================
// GCC local labels
// ===========================================================================

/// Parses a GCC `__label__` local label declaration as a statement.
///
/// `__label__ id1, id2, ..., idN ;`
///
/// Local labels are scoped to the enclosing compound statement.
/// We parse them and emit a Null statement (their semantic effect is
/// registering the labels in the current scope — handled by sema).
fn parse_local_label_decl(parser: &mut Parser<'_>) -> Result<Statement, ParseError> {
    let start = parser.current().span;
    let labels = super::gcc_extensions::parse_local_labels(parser)?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    // Local labels are registered in scope tracking by sema.
    // The parser emits them as a side-effect and returns a Null statement.
    let _ = labels;
    Ok(Statement::Null { span })
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Expects and consumes an identifier token, returning its `Symbol`.
/// On failure, emits a diagnostic and returns a `ParseError`.
fn expect_identifier(parser: &mut Parser<'_>, context: &str) -> Result<Symbol, ParseError> {
    match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            let sym = *sym;
            parser.advance();
            Ok(sym)
        }
        _ => {
            let span = parser.current().span;
            let msg = format!(
                "expected {} (identifier), found '{}'",
                context,
                parser.current().kind
            );
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier".to_string()),
            })
        }
    }
}
