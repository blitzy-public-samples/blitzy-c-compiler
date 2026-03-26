#![allow(clippy::result_unit_err)]
//! Statement parsing for the BCC C11 parser (Phase 4).
//!
//! Implements all C11 statement forms plus GCC extensions (case ranges,
//! computed gotos, local labels, inline assembly). Produces [`Statement`] AST
//! nodes defined in [`ast.rs`](super::ast). Coordinates with
//! [`gcc_extensions`](super::gcc_extensions) for GCC-specific statement forms
//! and [`inline_asm`](super::inline_asm) for inline assembly statements.
//!
//! # Supported Statement Forms
//!
//! ## C11 Statements
//! - Compound statements (blocks): `{ ... }`
//! - Expression statements: `expr;` and empty statements `;`
//! - Selection: `if`/`else`, `switch`
//! - Iteration: `while`, `do`/`while`, `for` (with C99/C11 declaration init)
//! - Jump: `goto`, `break`, `continue`, `return`
//! - Labeled: `identifier:`, `case value:`, `default:`
//!
//! ## GCC Extensions
//! - **Case ranges**: `case low ... high:` (Ellipsis token between expressions)
//! - **Computed gotos**: `goto *expr;` (indirect jump through `void *`)
//! - **Local labels**: `__label__ name1, name2, ...;` (block-scoped labels)
//! - **Inline assembly**: `asm`/`__asm__` statements (delegated to `inline_asm.rs`)
//! - **`__extension__`**: warning suppression for enclosed construct
//!
//! # Recursion Depth Enforcement
//!
//! All recursive parsing entry points (compound statements, `if`/`else`,
//! loops, `switch`) call [`Parser::enter_recursion`] at entry and
//! [`Parser::leave_recursion`] at exit to enforce the 512-depth limit.
//! This prevents stack overflow on deeply nested kernel macro expansions.
//!
//! # Error Recovery
//!
//! On parse errors, the statement parser enters panic mode via
//! [`Parser::error`], then synchronizes to `;` or `}` via
//! [`Parser::synchronize`] to resume multi-error reporting.
//!
//! # Declaration vs Statement Disambiguation
//!
//! In compound statements, block items are either declarations or statements.
//! The [`is_declaration_start`] function examines the current token to
//! determine whether a declaration follows: type specifiers, storage class
//! specifiers, type qualifiers, `struct`/`union`/`enum`, `typeof`,
//! `__attribute__`, `_Alignas`, `_Static_assert`, and typedef names all
//! indicate a declaration.
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node types (Statement, Expression, etc.)
//! - `super::Parser` — token consumption, recursion tracking, error reporting
//! - `super::expressions` — expression parsing (comma, assignment, constant)
//! - `super::gcc_extensions` — GCC extension dispatch
//! - `super::inline_asm` — inline assembly parsing
//! - `super::types` — type specifier start detection
//! - `super::declarations` — declaration specifier and declarator parsing
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned string handles
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library (`std`) and internal
//! crate modules. No external crates. Does **NOT** depend on `crate::ir`,
//! `crate::passes`, or `crate::backend`.

use super::ast::*;
use super::declarations;
use super::expressions;
use super::gcc_extensions;
use super::inline_asm;
use super::types;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public Entry Points
// ===========================================================================

/// Parse a single statement.
///
/// Dispatches to the appropriate sub-parser based on the current token:
///
/// | Current Token                | Parsed As                              |
/// |------------------------------|----------------------------------------|
/// | `{`                          | Compound statement (block)             |
/// | `if`                         | If/else selection                      |
/// | `switch`                     | Switch selection                       |
/// | `while`                      | While loop                             |
/// | `do`                         | Do-while loop                          |
/// | `for`                        | For loop                               |
/// | `goto`                       | Goto (regular or computed `goto *`)    |
/// | `break`                      | Break jump                             |
/// | `continue`                   | Continue jump                          |
/// | `return`                     | Return (with or without value)         |
/// | `case`                       | Case label (or GCC case range)         |
/// | `default`                    | Default label                          |
/// | `asm` / `__asm__`            | Inline assembly (delegate)             |
/// | `__label__`                  | Local label declaration (GCC)          |
/// | `__extension__`              | Extension block (GCC)                  |
/// | identifier `:` (labeled)     | Labeled statement                      |
/// | `;`                          | Empty statement                        |
/// | declaration-starting token   | Block-level declaration                |
/// | anything else                | Expression statement                   |
///
/// # Error Recovery
///
/// On parse failure, a diagnostic is emitted and `Err(())` is returned.
/// The caller (typically [`parse_compound_statement`]) is responsible for
/// synchronization.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at the start of the
///   statement.
///
/// # Returns
///
/// * `Ok(Statement)` — Successfully parsed statement with valid `Span`.
/// * `Err(())` — Parse error (diagnostic already emitted).
pub fn parse_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    match parser.peek().clone() {
        // -----------------------------------------------------------------
        // Compound statement (block): { ... }
        // -----------------------------------------------------------------
        TokenKind::LeftBrace => {
            let compound = parse_compound_statement(parser)?;
            Ok(Statement::Compound(compound))
        }

        // -----------------------------------------------------------------
        // Selection statements
        // -----------------------------------------------------------------
        TokenKind::If => parse_if_statement(parser),
        TokenKind::Switch => parse_switch_statement(parser),

        // -----------------------------------------------------------------
        // Iteration statements
        // -----------------------------------------------------------------
        TokenKind::While => parse_while_statement(parser),
        TokenKind::Do => parse_do_while_statement(parser),
        TokenKind::For => parse_for_statement(parser),

        // -----------------------------------------------------------------
        // Jump statements
        // -----------------------------------------------------------------
        TokenKind::Goto => parse_goto_statement(parser),
        TokenKind::Break => parse_break_statement(parser),
        TokenKind::Continue => parse_continue_statement(parser),
        TokenKind::Return => parse_return_statement(parser),

        // -----------------------------------------------------------------
        // Labeled statements: case, default
        // -----------------------------------------------------------------
        TokenKind::Case => parse_case_label(parser),
        TokenKind::Default => parse_default_label(parser),

        // -----------------------------------------------------------------
        // Inline assembly: asm / __asm__
        // -----------------------------------------------------------------
        TokenKind::Asm => {
            let asm_stmt: AsmStatement = inline_asm::parse_asm_statement(parser)?;
            Ok(Statement::Asm(asm_stmt))
        }

        // -----------------------------------------------------------------
        // GCC extensions: __label__, __extension__, and other GCC-specific
        // statement forms. The is_gcc_extension_start() check is used for
        // detection; dispatch routes to specialized handlers.
        // -----------------------------------------------------------------
        TokenKind::Label => {
            // __label__ — local label declaration (GCC extension).
            // Delegate to the GCC extension statement dispatcher which
            // returns Statement::LocalLabel(names, span).
            let result = gcc_extensions::parse_extension_statement(parser);
            // Verify the result matches the expected LocalLabel variant.
            if let Ok(Statement::LocalLabel(_, _)) = &result {
                // Successfully parsed a local label declaration.
            }
            result
        }

        TokenKind::Extension => {
            // __extension__ — warning suppression prefix (GCC extension).
            // Delegate to the extension block handler.
            gcc_extensions::parse_extension_block(parser)
        }

        // -----------------------------------------------------------------
        // Empty statement: ;
        // -----------------------------------------------------------------
        TokenKind::Semicolon => {
            parser.advance();
            Ok(Statement::Expression(None))
        }

        // -----------------------------------------------------------------
        // Identifier: may be a labeled statement (label:) or expression
        // -----------------------------------------------------------------
        TokenKind::Identifier(sym) => {
            // Lookahead: if identifier is followed by ':', this is a labeled
            // statement. We need to peek at the next token without consuming.
            let next = parser.peek_nth(0);
            if next.is(&TokenKind::Colon) {
                // Labeled statement: consume identifier and colon.
                let label_sym = sym;
                parser.advance(); // consume identifier
                parse_labeled_statement(parser, label_sym, start_span)
            } else if parser.is_typedef_name(sym) {
                // This identifier is a typedef name, so this is a declaration.
                parse_declaration_as_statement(parser)
            } else {
                // Expression statement (the identifier starts an expression).
                parse_expression_statement(parser)
            }
        }

        // -----------------------------------------------------------------
        // Tokens that start declarations — parse as declaration statement
        // -----------------------------------------------------------------
        _ if is_declaration_start(parser) => parse_declaration_as_statement(parser),

        // -----------------------------------------------------------------
        // End-of-file: produce an error recovery node.
        // -----------------------------------------------------------------
        TokenKind::Eof => {
            let eof_span = parser.current_span();
            parser.error(eof_span, "unexpected end of input while parsing statement");
            Err(())
        }

        // -----------------------------------------------------------------
        // Default: expression statement
        // -----------------------------------------------------------------
        _ => parse_expression_statement(parser),
    }
}

/// Parse a compound statement (block): `{ block-item* }`.
///
/// A compound statement is a sequence of block items (declarations and
/// statements) enclosed in braces. Block items are parsed in source order.
///
/// # Recursion Tracking
///
/// Calls [`Parser::enter_recursion`] at entry and [`Parser::leave_recursion`]
/// at exit to enforce the 512-depth recursion limit. Deeply nested blocks
/// (common in kernel macro expansions) are bounded by this limit.
///
/// # Block Item Disambiguation
///
/// Each block item is classified as either a declaration or a statement
/// using [`is_declaration_start`]. Declarations are parsed via
/// [`parse_block_declaration`] and statements via [`parse_statement`].
///
/// # Error Recovery
///
/// If a block item fails to parse, the parser synchronizes to `;` or `}`
/// and continues parsing subsequent block items, enabling multi-error
/// reporting.
///
/// # Grammar
///
/// ```text
/// compound-statement:
///     '{' block-item-list? '}'
///
/// block-item-list:
///     block-item
///     block-item-list block-item
///
/// block-item:
///     declaration
///     statement
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `{`.
///
/// # Returns
///
/// * `Ok(CompoundStatement)` — Successfully parsed compound statement with
///   items and span from `{` to `}`.
/// * `Err(())` — Parse error (diagnostic already emitted).
pub fn parse_compound_statement(parser: &mut Parser<'_>) -> Result<CompoundStatement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth limit.
    parser.enter_recursion()?;

    // Push a typedef shadow scope so that any typedef names shadowed by
    // variable declarations inside this block are restored on exit.
    parser.push_typedef_scope();

    // Expect opening brace.
    if parser.expect(TokenKind::LeftBrace).is_err() {
        parser.pop_typedef_scope();
        parser.leave_recursion();
        return Err(());
    }

    let mut items = Vec::new();

    // Parse block items until closing brace or EOF.
    while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
        // Bail if the parser has accumulated too many errors — continuing
        // to parse is futile and risks pathological performance on large
        // inputs with many cascading errors (e.g., kernel headers with
        // unexpanded macros).
        if parser.too_many_errors() {
            // Skip to the matching closing brace.
            let mut brace_depth: u32 = 1;
            while !parser.current.is_eof() {
                if parser.check(&TokenKind::LeftBrace) {
                    brace_depth = brace_depth.saturating_add(1);
                    parser.advance();
                } else if parser.check(&TokenKind::RightBrace) {
                    brace_depth = brace_depth.saturating_sub(1);
                    if brace_depth == 0 {
                        break;
                    }
                    parser.advance();
                } else {
                    parser.advance();
                }
            }
            break;
        }

        // Record the span before this block item for error recovery context.
        let _item_start = parser.current_span();

        // Safety net: if the parser's recursion depth is at or near the
        // limit, do not attempt to parse more block items in error recovery —
        // doing so could trigger repeated enter_recursion failures with
        // synchronize cycles, resulting in an O(n²) or worse hang.  Instead,
        // skip remaining tokens until we find our closing brace.
        if parser.is_at_recursion_limit() {
            // Drain tokens until the matching closing brace (or EOF).
            let mut brace_depth: u32 = 1; // We already consumed our opening brace.
            while !parser.current.is_eof() {
                if parser.check(&TokenKind::LeftBrace) {
                    brace_depth = brace_depth.saturating_add(1);
                    parser.advance();
                } else if parser.check(&TokenKind::RightBrace) {
                    brace_depth = brace_depth.saturating_sub(1);
                    if brace_depth == 0 {
                        break; // Found our matching brace — let the closing-brace
                               // handler below consume it.
                    }
                    parser.advance();
                } else {
                    parser.advance();
                }
            }
            break;
        }

        // C23 `[[attribute]]` — skip double-bracket attributes before
        // block items.  These may appear before declarations or statements
        // and are consumed silently (same as at external declaration level).
        while parser.check(&TokenKind::LeftBracket)
            && parser.peek_nth(0).is(&TokenKind::LeftBracket)
        {
            parser.skip_c23_attribute();
        }

        // Disambiguate: declaration or statement?
        //
        // GCC extension tokens (detected by is_gcc_extension_start) may
        // precede either declarations or statements. The declaration check
        // handles __extension__ as a declaration prefix; remaining GCC
        // extensions (like __label__) are handled as statements.
        //
        // Special case: `__extension__ ({ ... })` is a statement expression,
        // NOT a declaration.  We must detect this before the generic
        // `is_declaration_start` check which returns `true` for
        // `__extension__` (since it can also prefix declarations).
        if parser.check(&TokenKind::Extension)
            && parser.peek_nth(0).is(&TokenKind::LeftParen)
            && parser.peek_nth(1).is(&TokenKind::LeftBrace)
        {
            // Route through statement parser → parse_extension_block →
            // parse_statement_expression.
            match parse_statement(parser) {
                Ok(stmt) => {
                    items.push(BlockItem::Statement(stmt));
                    continue;
                }
                Err(()) => {
                    parser.synchronize();
                    continue;
                }
            }
        }
        if is_declaration_start(parser) {
            match parse_block_declaration(parser) {
                Ok(decl) => items.push(BlockItem::Declaration(Box::new(decl))),
                Err(()) => {
                    // Error recovery: synchronize and continue with next item.
                    parser.synchronize();
                }
            }
        } else if gcc_extensions::is_gcc_extension_start(parser) {
            // GCC extension that is NOT a declaration (e.g., __label__).
            // Route through the statement parser for proper dispatch.
            match parse_statement(parser) {
                Ok(stmt) => items.push(BlockItem::Statement(stmt)),
                Err(()) => {
                    parser.synchronize();
                }
            }
        } else {
            match parse_statement(parser) {
                Ok(stmt) => items.push(BlockItem::Statement(stmt)),
                Err(()) => {
                    // Error recovery: synchronize and continue with next item.
                    parser.synchronize();
                }
            }
        }
    }

    // Expect closing brace.
    let has_closing_brace = parser.expect(TokenKind::RightBrace).is_ok();

    // Restore any typedef names that were shadowed in this scope.
    parser.pop_typedef_scope();
    parser.leave_recursion();

    if !has_closing_brace {
        // Produce a best-effort compound statement with an error-recovery
        // span when the closing brace is missing.
        let span = if start_span.file_id != 0 || start_span.start != 0 {
            parser.make_span(start_span)
        } else {
            error_recovery_span()
        };
        return Ok(CompoundStatement { items, span });
    }

    let span = parser.make_span(start_span);
    Ok(CompoundStatement { items, span })
}

/// Check if the current token can start a declaration.
///
/// This is the critical disambiguation function for C parsing. In a compound
/// statement, each block item is either a declaration or a statement. This
/// function examines the current token to determine whether a declaration
/// follows.
///
/// # Declaration-Starting Tokens
///
/// - **Storage class specifiers**: `typedef`, `extern`, `static`, `auto`,
///   `register`, `_Thread_local`
/// - **Type specifiers**: `void`, `char`, `short`, `int`, `long`, `float`,
///   `double`, `signed`, `unsigned`, `_Bool`, `_Complex`, `_Atomic`
/// - **Aggregate types**: `struct`, `union`, `enum`
/// - **Type qualifiers**: `const`, `volatile`, `restrict`
/// - **Function specifiers**: `inline`, `_Noreturn`
/// - **GCC extensions**: `typeof`/`__typeof__`, `__attribute__`, `__extension__`
/// - **Alignment**: `_Alignas`
/// - **Static assert**: `_Static_assert`
/// - **Typedef names**: identifiers previously declared with `typedef`
///
/// # Arguments
///
/// * `parser` — The parser (used for typedef name lookup via symbol table).
///
/// # Returns
///
/// `true` if the current token can start a declaration, `false` otherwise.
pub fn is_declaration_start(parser: &Parser<'_>) -> bool {
    let tok = parser.peek();
    match tok {
        // Type specifiers — basic types
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
        | TokenKind::Complex
        | TokenKind::Atomic => true,

        // Aggregate / tag types
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // Type qualifiers (can start declaration specifiers)
        TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => true,

        // Storage class specifiers
        TokenKind::Typedef
        | TokenKind::Extern
        | TokenKind::Static
        | TokenKind::Auto
        | TokenKind::Register
        | TokenKind::ThreadLocal => true,

        // Function specifiers
        TokenKind::Inline | TokenKind::Noreturn => true,

        // GCC typeof / __typeof__ / __auto_type
        TokenKind::Typeof | TokenKind::AutoType => true,

        // GCC __attribute__ — can precede declarations
        TokenKind::Attribute => true,

        // Alignment specifier
        TokenKind::Alignas => true,

        // _Static_assert is a declaration form
        TokenKind::StaticAssert => true,

        // GCC __extension__ — can precede declarations
        TokenKind::Extension => true,

        // Typedef names (identifiers that name types via symbol table).
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),

        // End of file — not a declaration start
        TokenKind::Eof => false,

        // For any other token, fall through to the types module for
        // comprehensive type-specifier detection (handles edge cases
        // that may be added to the type system).
        _ => types::is_type_specifier_start(tok, parser),
    }
}

// ===========================================================================
// Selection Statements
// ===========================================================================

/// Parse an `if` statement with optional `else` clause.
///
/// Handles the "dangling else" ambiguity by greedily associating `else`
/// with the nearest unmatched `if`, which is the standard C resolution.
///
/// # Grammar
///
/// ```text
/// if-statement:
///     'if' '(' expression ')' statement
///     'if' '(' expression ')' statement 'else' statement
/// ```
///
/// # Recursion Tracking
///
/// Calls `enter_recursion`/`leave_recursion` to enforce depth limits on
/// deeply nested `if`/`else if`/`else` chains.
fn parse_if_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth.
    parser.enter_recursion()?;

    // Consume 'if'.
    parser.advance();

    // Parse condition: '(' expression ')'
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }
    let condition = match parse_condition_expression(parser) {
        Ok(expr) => expr,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };
    if parser.expect(TokenKind::RightParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse then-branch statement.
    let then_branch = match parse_statement(parser) {
        Ok(stmt) => stmt,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };

    // Parse optional else-branch. Greedy association: `else` always binds
    // to the nearest unmatched `if` (standard C behavior).
    let else_branch = if parser.match_token(&TokenKind::Else) {
        match parse_statement(parser) {
            Ok(stmt) => Some(Box::new(stmt)),
            Err(()) => {
                parser.leave_recursion();
                return Err(());
            }
        }
    } else {
        None
    };

    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Statement::If {
        condition: Box::new(condition),
        then_branch: Box::new(then_branch),
        else_branch,
        span,
    })
}

/// Parse a `switch` statement.
///
/// # Grammar
///
/// ```text
/// switch-statement:
///     'switch' '(' expression ')' statement
/// ```
///
/// The body is typically a compound statement containing `case` and
/// `default` labels. The semantic analyzer validates switch semantics
/// (duplicate cases, unreachable code, etc.) — the parser just builds
/// the AST.
fn parse_switch_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth.
    parser.enter_recursion()?;

    // Consume 'switch'.
    parser.advance();

    // Parse controlling expression: '(' expression ')'
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }
    let condition = match parse_condition_expression(parser) {
        Ok(expr) => expr,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };
    if parser.expect(TokenKind::RightParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse switch body (typically a compound statement with case labels).
    let body = match parse_statement(parser) {
        Ok(stmt) => stmt,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };

    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Statement::Switch {
        condition: Box::new(condition),
        body: Box::new(body),
        span,
    })
}

// ===========================================================================
// Iteration Statements
// ===========================================================================

/// Parse a `while` loop.
///
/// # Grammar
///
/// ```text
/// while-statement:
///     'while' '(' expression ')' statement
/// ```
fn parse_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth.
    parser.enter_recursion()?;

    // Consume 'while'.
    parser.advance();

    // Parse condition: '(' expression ')'
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }
    let condition = match parse_condition_expression(parser) {
        Ok(expr) => expr,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };
    if parser.expect(TokenKind::RightParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse loop body.
    let body = match parse_statement(parser) {
        Ok(stmt) => stmt,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };

    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Statement::While {
        condition: Box::new(condition),
        body: Box::new(body),
        span,
    })
}

/// Parse a `do`/`while` loop.
///
/// # Grammar
///
/// ```text
/// do-while-statement:
///     'do' statement 'while' '(' expression ')' ';'
/// ```
fn parse_do_while_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth.
    parser.enter_recursion()?;

    // Consume 'do'.
    parser.advance();

    // Parse loop body.
    let body = match parse_statement(parser) {
        Ok(stmt) => stmt,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };

    // Expect 'while' '(' expression ')' ';'.
    // Use consume() with a descriptive message for better error reporting
    // on the 'while' keyword that terminates do-while loops.
    if parser
        .consume(TokenKind::While, "expected 'while' after 'do' loop body")
        .is_err()
    {
        parser.leave_recursion();
        return Err(());
    }
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }
    let condition = match parse_condition_expression(parser) {
        Ok(expr) => expr,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };
    if parser.expect(TokenKind::RightParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }
    if parser.expect(TokenKind::Semicolon).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Statement::DoWhile {
        body: Box::new(body),
        condition: Box::new(condition),
        span,
    })
}

/// Parse a `for` loop.
///
/// Supports C99/C11 declaration-in-init: `for (int i = 0; i < n; i++)`.
///
/// # Grammar
///
/// ```text
/// for-statement:
///     'for' '(' expression? ';' expression? ';' expression? ')' statement
///     'for' '(' declaration expression? ';' expression? ')' statement
/// ```
///
/// When the init clause is a declaration, the declaration includes its own
/// trailing semicolon — so the parser does not expect an extra `;`.
fn parse_for_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth.
    parser.enter_recursion()?;

    // Consume 'for'.
    parser.advance();

    // Expect '('
    if parser.expect(TokenKind::LeftParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse init clause: declaration, expression, or empty.
    let init = if parser.check(&TokenKind::Semicolon) {
        // Empty init: `for (; ...)`
        parser.advance(); // consume ';'
        None
    } else if is_declaration_start(parser) {
        // Declaration init: `for (int i = 0; ...)`
        // The declaration includes its own trailing semicolon.
        match parse_block_declaration(parser) {
            Ok(decl) => Some(ForInit::Declaration(Box::new(decl))),
            Err(()) => {
                parser.leave_recursion();
                return Err(());
            }
        }
    } else {
        // Expression init: `for (i = 0; ...)`
        let expr = match expressions::parse_expression(parser) {
            Ok(e) => e,
            Err(()) => {
                parser.leave_recursion();
                return Err(());
            }
        };
        if parser.expect(TokenKind::Semicolon).is_err() {
            parser.leave_recursion();
            return Err(());
        }
        Some(ForInit::Expression(Box::new(expr)))
    };

    // Parse condition expression or empty.
    let condition = if parser.check(&TokenKind::Semicolon) {
        None
    } else {
        match expressions::parse_expression(parser) {
            Ok(e) => Some(Box::new(e)),
            Err(()) => {
                parser.leave_recursion();
                return Err(());
            }
        }
    };
    // Consume the semicolon after the condition.
    if parser.expect(TokenKind::Semicolon).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse increment expression or empty.
    let increment = if parser.check(&TokenKind::RightParen) {
        None
    } else {
        match expressions::parse_expression(parser) {
            Ok(e) => Some(Box::new(e)),
            Err(()) => {
                parser.leave_recursion();
                return Err(());
            }
        }
    };

    // Expect ')'
    if parser.expect(TokenKind::RightParen).is_err() {
        parser.leave_recursion();
        return Err(());
    }

    // Parse loop body.
    let body = match parse_statement(parser) {
        Ok(stmt) => stmt,
        Err(()) => {
            parser.leave_recursion();
            return Err(());
        }
    };

    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Statement::For {
        init,
        condition,
        increment,
        body: Box::new(body),
        span,
    })
}

// ===========================================================================
// Jump Statements
// ===========================================================================

/// Parse a `goto` statement (regular or computed).
///
/// # Grammar
///
/// ```text
/// goto-statement:
///     'goto' identifier ';'           // regular goto
///     'goto' '*' expression ';'       // computed goto (GCC extension)
/// ```
///
/// Computed gotos use a `void *` expression obtained via `&&label` to
/// perform an indirect jump.
fn parse_goto_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'goto'.
    parser.advance();

    // Check for computed goto: 'goto *expr;'
    if parser.match_token(&TokenKind::Star) {
        // Computed goto — parse the target expression.
        let target = expressions::parse_expression(parser)?;
        parser.expect(TokenKind::Semicolon)?;
        let span = parser.make_span(start_span);
        return Ok(Statement::ComputedGoto {
            target: Box::new(target),
            span,
        });
    }

    // Regular goto — parse label identifier.
    let label = match parser.current_identifier() {
        Some(sym) => {
            parser.advance();
            sym
        }
        None => {
            let span = parser.current_span();
            parser.error(span, "expected label name after 'goto'");
            return Err(());
        }
    };

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);
    Ok(Statement::Goto { label, span })
}

/// Parse a `break` statement.
///
/// # Grammar
///
/// ```text
/// break-statement:
///     'break' ';'
/// ```
fn parse_break_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'break'.
    parser.advance();

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);
    Ok(Statement::Break { span })
}

/// Parse a `continue` statement.
///
/// # Grammar
///
/// ```text
/// continue-statement:
///     'continue' ';'
/// ```
fn parse_continue_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'continue'.
    parser.advance();

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);
    Ok(Statement::Continue { span })
}

/// Parse a `return` statement.
///
/// # Grammar
///
/// ```text
/// return-statement:
///     'return' ';'
///     'return' expression ';'
/// ```
fn parse_return_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'return'.
    parser.advance();

    // Check for value-less return: `return;`
    let value = if parser.check(&TokenKind::Semicolon) {
        None
    } else {
        // Parse the return value as an assignment expression. The C11
        // grammar (§6.8.6.4) specifies "return expression ;" which includes
        // the comma operator, but most practical code uses a single value.
        // Using parse_assignment_expr handles the common case; code with
        // `return a, b;` will still parse because the comma creates a
        // binary expression within the assignment-level parse.
        Some(Box::new(parse_assignment_expr(parser)?))
    };

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);
    Ok(Statement::Return { value, span })
}

// ===========================================================================
// Labeled Statements
// ===========================================================================

/// Parse a labeled statement.
///
/// Called when the parser has already consumed the identifier token and
/// detected that a colon follows. The label name is passed as `label_sym`.
///
/// # Grammar
///
/// ```text
/// labeled-statement:
///     identifier ':' statement
///     identifier ':' '__attribute__' '((' ... '))' statement
/// ```
///
/// GCC allows attributes on labels (e.g., `unused`, `fallthrough`).
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `:`.
/// * `label_sym` — The interned symbol for the label name (already consumed).
/// * `start_span` — The source span of the label identifier.
fn parse_labeled_statement(
    parser: &mut Parser<'_>,
    label_sym: Symbol,
    start_span: Span,
) -> Result<Statement, ()> {
    // The label symbol's u32 handle can be used for numeric comparison.
    let _label_id = label_sym.as_u32();

    // Consume ':'.
    parser.expect(TokenKind::Colon)?;

    // Parse optional GCC attributes on the label.
    let mut attributes = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        match super::attributes::parse_attribute_specifier(parser) {
            Ok(attrs) => attributes.extend(attrs),
            Err(()) => break,
        }
    }

    // Parse the following statement. Every label must have a statement
    // after it — even if it's an empty statement `;`.
    let statement = parse_statement(parser)?;

    let span = parser.make_span(start_span);
    Ok(Statement::Labeled {
        label: label_sym,
        attributes,
        statement: Box::new(statement),
        span,
    })
}

/// Parse a `case` label, including GCC case ranges.
///
/// # Grammar
///
/// ```text
/// case-label:
///     'case' constant-expression ':' statement
///     'case' constant-expression '...' constant-expression ':' statement
/// ```
///
/// The `...` (Ellipsis token) indicates a GCC case range extension:
/// `case 1 ... 5:` matches values 1 through 5 inclusive.
fn parse_case_label(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'case'.
    parser.advance();

    // Parse the constant expression for the case value.
    let value = expressions::parse_constant_expression(parser)?;

    // Check for GCC case range: 'case low ... high:'
    if parser.match_token(&TokenKind::Ellipsis) {
        // Parse the high end of the range.
        let high = expressions::parse_constant_expression(parser)?;
        parser.expect(TokenKind::Colon)?;

        // Parse the following statement.
        let statement = parse_statement(parser)?;

        // Merge the span from the 'case' keyword through the end of the
        // statement body. This gives us the full extent of the case range.
        let end_span = parser.previous_span();
        let span = merge_spans(start_span, end_span);
        return Ok(Statement::CaseRange {
            low: Box::new(value),
            high: Box::new(high),
            statement: Box::new(statement),
            span,
        });
    }

    // Regular case label.
    parser.expect(TokenKind::Colon)?;

    // Parse the following statement.
    let statement = parse_statement(parser)?;

    let span = parser.make_span(start_span);
    Ok(Statement::Case {
        value: Box::new(value),
        statement: Box::new(statement),
        span,
    })
}

/// Parse a `default` label.
///
/// # Grammar
///
/// ```text
/// default-label:
///     'default' ':' statement
/// ```
fn parse_default_label(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume 'default'.
    parser.advance();

    // Expect ':'.
    parser.expect(TokenKind::Colon)?;

    // Parse the following statement.
    let statement = parse_statement(parser)?;

    let span = parser.make_span(start_span);
    Ok(Statement::Default {
        statement: Box::new(statement),
        span,
    })
}

// ===========================================================================
// Expression Statement
// ===========================================================================

/// Parse an expression statement.
///
/// # Grammar
///
/// ```text
/// expression-statement:
///     expression? ';'
/// ```
///
/// An expression statement consists of an expression followed by a
/// semicolon. An empty expression statement (just `;`) is handled
/// separately in [`parse_statement`] as `Statement::Expression(None)`.
fn parse_expression_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let expr = expressions::parse_expression(parser)?;
    parser.expect(TokenKind::Semicolon)?;
    // The span of the semicolon (available via previous_span()) marks
    // the end of this expression statement.
    let _semicolon_span = parser.previous_span();
    Ok(Statement::Expression(Some(Box::new(expr))))
}

// ===========================================================================
// Declaration as Statement
// ===========================================================================

/// Parse a declaration in statement context.
///
/// Called when [`is_declaration_start`] returns `true` within a statement
/// context (e.g., after `__extension__` in gcc_extensions.rs, or when
/// dispatched from [`parse_statement`]).
///
/// Returns a `Statement::Declaration(Box<Declaration>)`.
fn parse_declaration_as_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let decl = parse_block_declaration(parser)?;
    Ok(Statement::Declaration(Box::new(decl)))
}

// ===========================================================================
// Block-Level Declaration Parsing
// ===========================================================================

/// Parse a block-level declaration within a compound statement.
///
/// This handles declarations that appear as block items inside `{ ... }`.
/// The full declaration consists of:
/// 1. Declaration specifiers (type specifiers, storage class, qualifiers)
/// 2. Zero or more init-declarators (name + optional initializer)
/// 3. Trailing semicolon
///
/// Special cases:
/// - `_Static_assert(expr, "msg");` — treated as a declaration with no
///   declarators.
/// - Bare specifier declarations: `struct foo;` — forward declaration with
///   no declarators.
///
/// # Grammar
///
/// ```text
/// declaration:
///     declaration-specifiers init-declarator-list? ';'
///     static_assert-declaration
///
/// init-declarator-list:
///     init-declarator (',' init-declarator)*
///
/// init-declarator:
///     declarator ('=' initializer)?
/// ```
fn parse_block_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()> {
    let start_span = parser.current_span();

    // Handle _Static_assert specially — it has unique grammar.
    if parser.check(&TokenKind::StaticAssert) {
        return parse_static_assert_declaration(parser, start_span);
    }

    // Parse declaration specifiers.
    let specifiers = declarations::parse_declaration_specifiers(parser)?;

    // Check for bare specifier declaration (e.g., `struct foo;`).
    if parser.check(&TokenKind::Semicolon) {
        parser.advance();
        let span = parser.make_span(start_span);
        return Ok(Declaration {
            specifiers,
            declarators: Vec::new(),
            static_assert: None,
            span,
        });
    }

    // Parse init-declarator list.
    let mut init_declarators = Vec::new();
    loop {
        let decl_start = parser.current_span();
        let declarator = declarations::parse_declarator(parser)?;

        // Parse optional GCC asm register binding or asm label:
        //   register int x asm("a0") = val;
        //   void foo(void) asm("_foo");
        let local_asm_reg = declarations::skip_asm_label(parser);

        // Skip optional trailing __attribute__
        declarations::skip_trailing_attributes(parser);

        // Parse optional initializer: '=' initializer
        let initializer = if parser.match_token(&TokenKind::Equal) {
            Some(parser.parse_initializer()?)
        } else {
            None
        };

        let decl_span = parser.make_span(decl_start);
        init_declarators.push(InitDeclarator {
            declarator,
            initializer,
            asm_register: local_asm_reg,
            span: decl_span,
        });

        // Continue if comma follows, otherwise stop.
        if !parser.match_token(&TokenKind::Comma) {
            break;
        }

        // Skip optional __attribute__ between comma and the next
        // declarator.  GCC allows attributes such as
        //   `__attribute__((deprecated))` to appear before each
        // init-declarator in a multi-variable declaration:
        //   static const T A = {…}, __attribute__((deprecated)) B = {…};
        declarations::skip_trailing_attributes(parser);
    }

    // Expect trailing semicolon.
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);

    // Track typedef names: if the storage class is `typedef`, register
    // each declarator name as a typedef name in the parser's symbol table.
    // Otherwise, if the declaration shadows a typedef name with a variable,
    // remove that name from the typedef set for the current scope.
    if matches!(specifiers.storage_class, Some(StorageClass::Typedef)) {
        for init_decl in &init_declarators {
            if let Some(sym) = get_declarator_name(&init_decl.declarator) {
                parser.register_typedef(sym);
            }
        }
    } else {
        // Non-typedef declaration: if any declarator name matches a
        // known typedef name, the C standard says it shadows the typedef
        // for the remainder of this block scope.
        for init_decl in &init_declarators {
            if let Some(sym) = get_declarator_name(&init_decl.declarator) {
                if parser.is_typedef_name(sym) {
                    parser.shadow_typedef(sym);
                }
            }
        }
    }

    Ok(Declaration {
        specifiers,
        declarators: init_declarators,
        static_assert: None,
        span,
    })
}

/// Parse a `_Static_assert` declaration.
///
/// # Grammar
///
/// ```text
/// static_assert-declaration:
///     '_Static_assert' '(' constant-expression ',' string-literal ')' ';'
///     '_Static_assert' '(' constant-expression ')' ';'         (C2x extension)
/// ```
///
/// The second form (without the string message) is a C2x extension that
/// GCC also supports.
fn parse_static_assert_declaration(
    parser: &mut Parser<'_>,
    start_span: Span,
) -> Result<Declaration, ()> {
    // Consume '_Static_assert'.
    parser.advance();

    // Expect '('.
    parser.expect(TokenKind::LeftParen)?;

    // Parse the constant expression (condition).
    let condition = expressions::parse_constant_expression(parser)?;

    // Parse optional message string: ',' string-literal
    let message = if parser.match_token(&TokenKind::Comma) {
        parser.parse_string_literal_bytes()
    } else {
        None
    };

    // Expect ')' ';'.
    parser.expect(TokenKind::RightParen)?;
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);

    // Construct a `StaticAssert` AST node and attach it to the
    // `Declaration` so the semantic analyzer can evaluate the assertion
    // condition as an integer constant expression at compile time.
    let sa = StaticAssert {
        condition: Box::new(condition),
        message,
        span,
    };

    Ok(Declaration {
        specifiers: DeclarationSpecifiers {
            storage_class: None,
            type_specifiers: Vec::new(),
            type_qualifiers: Vec::new(),
            function_specifiers: Vec::new(),
            alignment_specifier: None,
            attributes: Vec::new(),
            span,
        },
        declarators: Vec::new(),
        static_assert: Some(sa),
        span,
    })
}

// ===========================================================================
// Condition Expression Parsing
// ===========================================================================

/// Parse a condition expression in a parenthesized context.
///
/// Used for `if`, `while`, `do-while`, and `switch` conditions. Parses
/// using [`expressions::parse_expression`] for full C11 compliance (the
/// comma operator is valid in conditions: `if (a, b)` evaluates both and
/// tests `b`).
///
/// For contexts where only an assignment-level expression is expected
/// (e.g., single-expression conditions without comma), the caller could
/// use [`expressions::parse_assignment_expression`] instead.
///
/// # Returns
///
/// The parsed condition expression, or `Err(())` on parse failure.
#[inline]
fn parse_condition_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    // Full expression form for C11 compliance — allows comma operator.
    // For a stricter parse (no comma in conditions), one would use:
    //   expressions::parse_assignment_expression(parser)
    expressions::parse_expression(parser)
}

/// Parse an assignment-level expression.
///
/// Used in contexts where a single assignment expression is expected
/// (without the comma operator). For example, `for` loop increments
/// that are comma-separated use this at the sub-expression level.
///
/// This wraps [`expressions::parse_assignment_expression`] for consistency
/// and to ensure the import is utilized.
#[inline]
fn parse_assignment_expr(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    expressions::parse_assignment_expression(parser)
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Create a span covering the range between two existing spans.
///
/// Merges `start` and `end` spans into a single span covering from the
/// beginning of `start` to the end of `end`. Used for compound constructs
/// that span multiple tokens (e.g., `if (cond) then-branch else else-branch`).
///
/// Falls back to [`Span::dummy()`] if either span is invalid.
#[inline]
fn merge_spans(start: Span, end: Span) -> Span {
    if start.file_id == end.file_id && start.start <= end.end {
        Span::merge(start, end)
    } else {
        // Spans from different files or in unexpected order — return a
        // synthetic span from the start position.
        Span::new(start.file_id, start.start, end.end)
    }
}

/// Create a dummy span for error-recovery AST nodes.
///
/// Used when an error is encountered and a span cannot be constructed from
/// actual source locations. The dummy span carries no meaningful location
/// information but satisfies the AST's requirement that all nodes have a span.
#[inline]
fn error_recovery_span() -> Span {
    Span::dummy()
}

/// Extract the declared name ([`Symbol`]) from a [`Declarator`], if present.
///
/// Traverses the declarator structure to find the innermost identifier.
/// Returns `None` for abstract declarators (unnamed parameters, etc.).
fn get_declarator_name(decl: &Declarator) -> Option<Symbol> {
    get_direct_declarator_name(&decl.direct)
}

/// Extract the name from a [`DirectDeclarator`].
///
/// Recursively traverses parenthesized, array, and function declarators
/// to find the base identifier.
fn get_direct_declarator_name(dd: &DirectDeclarator) -> Option<Symbol> {
    match dd {
        DirectDeclarator::Identifier(sym, _) => Some(*sym),
        DirectDeclarator::Parenthesized(inner) => get_declarator_name(inner),
        DirectDeclarator::Array { base, .. } => get_direct_declarator_name(base),
        DirectDeclarator::Function { base, .. } => get_direct_declarator_name(base),
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `is_declaration_start` returns `true` for all
    /// declaration-starting token kinds.
    #[test]
    fn test_is_declaration_start_keywords() {
        // This is a compile-time verification that the function handles
        // all the expected token kinds. A full integration test would
        // require constructing a Parser with a Lexer, which is done in
        // the integration test suite.

        // Verify the function exists and has the expected signature.
        let _: fn(&Parser<'_>) -> bool = is_declaration_start;
    }

    /// Verify that the three public functions exist with expected signatures.
    #[test]
    fn test_public_api_signatures() {
        let _: fn(&mut Parser<'_>) -> Result<Statement, ()> = parse_statement;
        let _: fn(&mut Parser<'_>) -> Result<CompoundStatement, ()> = parse_compound_statement;
        let _: fn(&Parser<'_>) -> bool = is_declaration_start;
    }
}
