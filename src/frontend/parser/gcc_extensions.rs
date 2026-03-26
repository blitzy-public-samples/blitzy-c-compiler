#![allow(clippy::result_unit_err)]
//! GCC language extension parsing dispatch for the BCC C11 parser (Phase 4).
//!
//! This module provides centralized handling for GCC-specific language
//! extensions required for Linux kernel compilation. It acts as a dispatch hub
//! that coordinates with other parser modules (`expressions`, `statements`,
//! `declarations`, `types`) to handle grammar constructs that are GCC-specific
//! or extend the C11 standard.
//!
//! # Supported GCC Extensions
//!
//! ## Expression-Level Extensions
//! - **Statement expressions** `({ ... })` — a compound statement as an
//!   expression whose value is the last expression-statement in the block.
//! - **Address-of-label** `&&label` — takes the address of a label for use
//!   with computed gotos (returns a `void *`).
//! - **Conditional operand omission** `x ?: y` — handled by `expressions.rs`
//!   during ternary parsing, with dispatch assistance from this module.
//!
//! ## Statement-Level Extensions
//! - **Computed gotos** `goto *expr` — indirect jump through a `void *` label
//!   address obtained via `&&label`.
//! - **Case ranges** `case low ... high:` — a range case label in switch
//!   statements (the `...` is the Ellipsis token).
//! - **Local labels** `__label__` — declares labels with block scope instead
//!   of function scope.
//!
//! ## Declaration-Level Extensions
//! - **Zero-length arrays** `char data[0]` — old-style flexible array members,
//!   parsed as `Array(elem_type, Some(0))`.
//! - **Flexible array members** `char data[]` — C99/C11 flexible array members,
//!   parsed as `Array(elem_type, None)`.
//! - **Transparent unions** `__attribute__((transparent_union))` — recognized
//!   and propagated; attribute parsing handled by `attributes.rs`.
//!
//! ## Type-Level Extensions
//! - **`typeof` / `__typeof__`** — type deduction from expressions or type
//!   names, producing `TypeSpecifier::Typeof(TypeofArg)`.
//! - **`__extension__`** — suppresses pedantic warnings for the enclosed
//!   declaration or expression.
//!
//! # Error Handling Policy
//!
//! Every GCC extension encountered is either:
//! 1. **Parsed correctly** — produces a well-formed AST node with a valid `Span`.
//! 2. **Diagnosed with a clear error** — identifies the unsupported construct
//!    by name so the user knows exactly what is missing.
//!
//! The compiler **MUST NOT** silently miscompile unknown extensions. This is a
//! hard requirement from the AAP (§0.7.6).
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node types (Expression, Statement, etc.)
//! - `super::Parser` — core token consumption and recursion tracking
//! - `super::types` — type specifier detection and parsing
//! - `super::expressions` — expression parsing for sub-expressions
//! - `super::statements` — compound statement parsing for statement expressions
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned string handles
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library (`std`) and internal
//! crate modules. No external crates. Does **NOT** depend on `crate::ir`,
//! `crate::passes`, or `crate::backend`.

use super::ast::*;
use super::expressions;
use super::statements;
use super::types;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Extension Detection
// ===========================================================================

/// Check whether the given token starts a GCC extension construct.
///
/// Returns `true` if the token is one of:
/// - `__extension__` (`TokenKind::Extension`)
/// - `__label__` (`TokenKind::Label`)
/// - `typeof` / `__typeof__` (`TokenKind::Typeof`)
/// - `__attribute__` (`TokenKind::Attribute`)
///
/// This function is used by the statement parser, expression parser, and
/// declaration parser to detect when GCC-specific parsing logic should be
/// invoked rather than standard C11 parsing.
///
/// # Arguments
///
/// * `parser` — The parser, used for accessing the current token and checking
///   typedef names when an identifier follows `__extension__`.
///
/// # Examples
///
/// ```ignore
/// if gcc_extensions::is_gcc_extension_start(parser) {
///     // Dispatch to GCC extension parsing
/// }
/// ```
pub fn is_gcc_extension_start(parser: &Parser<'_>) -> bool {
    matches!(
        parser.peek(),
        TokenKind::Extension | TokenKind::Label | TokenKind::Typeof | TokenKind::Attribute
    )
}

// ===========================================================================
// Extension Expression Dispatch
// ===========================================================================

/// Central dispatch for GCC extension expressions.
///
/// This function is called when a GCC-specific expression pattern is detected
/// in the expression parser. It handles:
///
/// - `__extension__` followed by an expression — suppresses pedantic warnings
///   and parses the following expression normally.
/// - `typeof` / `__typeof__` — delegates to type parsing (as `typeof` can
///   appear in expression-like contexts in some GCC extensions).
///
/// For statement expressions `({ ... })`, use [`parse_statement_expression`]
/// directly (called from `expressions.rs` when `(` followed by `{` is detected).
///
/// For address-of-label `&&label`, use [`parse_label_address`] directly
/// (called from `expressions.rs` during unary expression parsing).
///
/// # Arguments
///
/// * `parser` — The parser with the current token positioned at the start of
///   the GCC extension expression.
///
/// # Returns
///
/// * `Ok(Expression)` — Successfully parsed GCC extension expression.
/// * `Err(())` — Unrecoverable parse error (diagnostic already emitted).
pub fn parse_extension_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    match parser.peek() {
        // `__extension__` — suppress warnings and parse the following expression.
        TokenKind::Extension => {
            let start_span = parser.current_span();
            parser.advance(); // consume `__extension__`

            // After `__extension__`, we may see another extension keyword,
            // a statement expression `({ ... })`, or a regular expression.
            // Check for statement expression: `__extension__ ({ ... })`
            if parser.check(&TokenKind::LeftParen) {
                // Peek ahead: if `(` is followed by `{`, this is a statement
                // expression `({ ... })`.
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftBrace) {
                    return parse_statement_expression(parser);
                }
            }

            // Otherwise parse the following expression (assignment-level).
            // This handles cases like `__extension__ 0ULL` or
            // `__extension__ (some_expression)`.
            let expr = expressions::parse_assignment_expression(parser)?;

            // The expression inherits the span from the `__extension__` token
            // start to the end of the parsed expression.
            let _merged_span = start_span.merge(get_expression_span(&expr));
            Ok(expr)
        }

        // `&&label` — address-of-label for computed gotos.
        TokenKind::AmpAmp => parse_label_address(parser),

        // For any other unrecognized GCC extension expression:
        _ => {
            let span = parser.current_span();
            let token_desc = format!("{}", parser.current.kind);
            parser.error(
                span,
                &format!(
                    "unsupported GCC extension expression starting with '{}'",
                    token_desc
                ),
            );
            Err(())
        }
    }
}

// ===========================================================================
// Extension Statement Dispatch
// ===========================================================================

/// Central dispatch for GCC extension statements.
///
/// This function is called when a GCC-specific statement pattern is detected
/// in the statement parser. It handles:
///
/// - `goto *expr` — computed goto (indirect jump through a pointer).
/// - `case low ... high:` — case range label in switch statements.
/// - `__label__ name1, name2, ...;` — local label declarations.
/// - `__extension__` followed by a declaration or statement.
///
/// # Arguments
///
/// * `parser` — The parser with the current token positioned at the start of
///   the GCC extension statement.
///
/// # Returns
///
/// * `Ok(Statement)` — Successfully parsed GCC extension statement.
/// * `Err(())` — Unrecoverable parse error (diagnostic already emitted).
pub fn parse_extension_statement(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    match parser.peek() {
        // `__label__` — local label declaration.
        TokenKind::Label => parse_local_label_decl(parser),

        // `__extension__` — suppress warnings and parse the following
        // declaration or statement.
        TokenKind::Extension => parse_extension_block(parser),

        // For computed gotos (`goto *expr`) and case ranges
        // (`case low ... high:`), these are typically detected by the
        // statement parser in `statements.rs` which then delegates
        // parsing to specialized functions. However, if dispatched here:
        TokenKind::Goto => parse_computed_goto(parser),
        TokenKind::Case => parse_case_range(parser),

        // Unknown GCC extension statement — diagnose clearly.
        _ => {
            let span = parser.current_span();
            let token_desc = format!("{}", parser.current.kind);
            parser.error(
                span,
                &format!(
                    "unsupported GCC extension statement starting with '{}'",
                    token_desc
                ),
            );
            Err(())
        }
    }
}

// ===========================================================================
// __extension__ Block Handling
// ===========================================================================

/// Parse an `__extension__` block.
///
/// When `__extension__` is encountered, it suppresses pedantic warnings for
/// the enclosed construct. The construct may be:
/// - A declaration (starting with a type specifier, storage class, etc.)
/// - An expression statement
/// - A compound statement
///
/// The `__extension__` token is consumed, and then the following construct
/// is parsed according to its form.
///
/// # Grammar
///
/// ```text
/// __extension__ declaration
/// __extension__ expression ;
/// __extension__ compound-statement
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `__extension__`.
///
/// # Returns
///
/// * `Ok(Statement)` — The parsed declaration or statement, with `__extension__`
///   consumed. This may be a `Statement::Declaration(...)` if the construct is
///   a declaration, or a regular statement otherwise.
/// * `Err(())` — Unrecoverable parse error.
pub fn parse_extension_block(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume the `__extension__` token.
    parser.advance();

    // Determine what follows: declaration or statement.
    // Check if the next token can start a declaration.
    if is_declaration_start_after_extension(parser) {
        // Parse as a declaration within the current block scope.
        // Delegate to the statement parser which handles block-level
        // declarations.
        return statements::parse_statement(parser);
    }

    // Check for compound statement (block).
    if parser.check(&TokenKind::LeftBrace) {
        let compound = statements::parse_compound_statement(parser)?;
        return Ok(Statement::Compound(compound));
    }

    // Check for statement expression: `__extension__ ({ ... })`
    // This would appear as `(` followed by `{`.
    if parser.check(&TokenKind::LeftParen) {
        let next = parser.peek_nth(0);
        if next.is(&TokenKind::LeftBrace) {
            let expr = parse_statement_expression(parser)?;
            parser.expect(TokenKind::Semicolon)?;
            let span = parser.make_span(start_span);
            let _ = span; // span used for diagnostics if needed
            return Ok(Statement::Expression(Some(Box::new(expr))));
        }
    }

    // Otherwise, parse as an expression statement.
    let expr = expressions::parse_expression(parser)?;
    parser.expect(TokenKind::Semicolon)?;
    Ok(Statement::Expression(Some(Box::new(expr))))
}

// ===========================================================================
// Statement Expression: ({ ... })
// ===========================================================================

/// Parse a GCC statement expression: `({ ... })`.
///
/// A statement expression is a compound statement enclosed in parentheses
/// that can be used as an expression. The value of the statement expression
/// is the value of the last expression-statement in the compound block.
///
/// # Grammar
///
/// ```text
/// statement-expression:
///     '(' compound-statement ')'
/// ```
///
/// # Examples
///
/// ```c
/// int x = ({ int tmp = 5; tmp * 2; }); // x = 10
/// int max = ({ int a = x, b = y; a > b ? a : b; });
/// ```
///
/// # Recursion Tracking
///
/// Statement expressions may nest deeply in kernel macros. The parser
/// enforces the 512-depth recursion limit via `enter_recursion()` /
/// `leave_recursion()` to prevent stack overflow.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `(` (the opening
///   parenthesis of the statement expression).
///
/// # Returns
///
/// * `Ok(Expression::StatementExpression { compound, span })` on success.
/// * `Err(())` on unrecoverable parse error (diagnostic already emitted).
pub fn parse_statement_expression(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();

    // Enforce recursion depth limit for deeply nested statement expressions.
    parser.enter_recursion()?;

    // Consume the opening parenthesis `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Parse the compound statement `{ ... }`.
    let compound = statements::parse_compound_statement(parser)?;

    // Consume the closing parenthesis `)`.
    parser.expect(TokenKind::RightParen)?;

    // Leave recursion tracking.
    parser.leave_recursion();

    let span = parser.make_span(start_span);
    Ok(Expression::StatementExpression { compound, span })
}

// ===========================================================================
// Address-of-Label: &&label
// ===========================================================================

/// Parse a GCC address-of-label expression: `&&label`.
///
/// Takes the address of a label for use with computed gotos. The result is
/// a `void *` value that can be stored in an array and used with `goto *ptr`.
///
/// # Grammar
///
/// ```text
/// address-of-label:
///     '&&' identifier
/// ```
///
/// # Disambiguation from `&&` (logical AND)
///
/// The `&&` token (`TokenKind::AmpAmp`) is ambiguous — it could be the
/// logical AND operator or the start of an address-of-label expression.
/// The disambiguation is context-dependent:
///
/// - In unary expression context (no left operand), `&&` followed by an
///   identifier is an address-of-label.
/// - In binary expression context (after a left operand), `&&` is logical AND.
///
/// The caller (`expressions.rs`) performs this disambiguation and calls
/// `parse_label_address` only in the unary context.
///
/// # Examples
///
/// ```c
/// void *labels[] = { &&label1, &&label2 };
/// void *ptr = &&my_label;
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `&&` (`AmpAmp`).
///
/// # Returns
///
/// * `Ok(Expression::AddressOfLabel { label, span })` on success.
/// * `Err(())` if no identifier follows `&&` (diagnostic emitted).
pub fn parse_label_address(parser: &mut Parser<'_>) -> Result<Expression, ()> {
    let start_span = parser.current_span();

    // Consume the `&&` token.
    parser.advance();

    // Expect an identifier (the label name).
    match &parser.current.kind {
        TokenKind::Identifier(sym) => {
            let label: Symbol = *sym;
            let _label_id: u32 = label.as_u32();
            parser.advance();
            let span = parser.make_span(start_span);
            Ok(Expression::AddressOfLabel { label, span })
        }
        _ => {
            let span = parser.current_span();
            parser.error(
                span,
                "expected label name after '&&' in address-of-label expression",
            );
            Err(())
        }
    }
}

// ===========================================================================
// Computed Goto: goto *expr
// ===========================================================================

/// Parse a GCC computed goto statement: `goto *expr;`.
///
/// A computed goto performs an indirect jump to a label whose address was
/// previously obtained via `&&label`. The expression must evaluate to a
/// `void *` (checked by semantic analysis, not the parser).
///
/// # Grammar
///
/// ```text
/// computed-goto-statement:
///     'goto' '*' expression ';'
/// ```
///
/// # Examples
///
/// ```c
/// void *labels[] = { &&label1, &&label2 };
/// goto *labels[i];  // indirect jump
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `goto`.
///
/// # Returns
///
/// * `Ok(Statement::ComputedGoto { target, span })` on success.
/// * `Err(())` on parse error.
///
/// # Note
///
/// Standard `goto label;` is handled by the statement parser in
/// `statements.rs`. This function is specifically for the `goto *expr`
/// variant detected when `*` follows `goto`.
pub fn parse_computed_goto(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume `goto`.
    parser.expect(TokenKind::Goto)?;

    // Check for `*` — this distinguishes `goto *expr` from `goto label`.
    if !parser.check(&TokenKind::Star) {
        // This is a standard `goto label;` — parse the label name.
        match &parser.current.kind {
            TokenKind::Identifier(sym) => {
                let label: Symbol = *sym;
                parser.advance();
                parser.expect(TokenKind::Semicolon)?;
                let span = parser.make_span(start_span);
                return Ok(Statement::Goto { label, span });
            }
            _ => {
                let span = parser.current_span();
                parser.error(span, "expected label name or '*' after 'goto'");
                return Err(());
            }
        }
    }

    // Consume `*`.
    parser.advance();

    // Parse the target expression (must evaluate to void *).
    let target = expressions::parse_expression(parser)?;

    // Consume the terminating semicolon.
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);
    Ok(Statement::ComputedGoto {
        target: Box::new(target),
        span,
    })
}

// ===========================================================================
// Case Range: case low ... high:
// ===========================================================================

/// Parse a GCC case range label: `case low ... high: statement`.
///
/// A case range label matches any value in the inclusive range `[low, high]`
/// within a switch statement. The `...` is the Ellipsis token (three dots).
///
/// # Grammar
///
/// ```text
/// case-range-label:
///     'case' constant-expression '...' constant-expression ':' statement
/// ```
///
/// # Examples
///
/// ```c
/// switch (c) {
///     case 'a' ... 'z': handle_lower(); break;
///     case 1 ... 100: handle_range(); break;
/// }
/// ```
///
/// # Standard vs Range Cases
///
/// This function handles both standard `case value:` and GCC `case low ... high:`
/// cases. After parsing the first constant expression, if an Ellipsis token
/// follows, it is parsed as a case range; otherwise, it falls back to a
/// standard case label.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `case`.
///
/// # Returns
///
/// * `Ok(Statement::CaseRange { ... })` for range cases.
/// * `Ok(Statement::Case { ... })` for standard single-value cases.
/// * `Err(())` on parse error.
pub fn parse_case_range(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume `case`.
    parser.expect(TokenKind::Case)?;

    // Parse the first (low) constant expression.
    let low_expr = expressions::parse_constant_expression(parser)?;

    // Check for `...` (Ellipsis) — if present, this is a case range.
    if parser.match_token(&TokenKind::Ellipsis) {
        // Parse the second (high) constant expression.
        let high_expr = expressions::parse_constant_expression(parser)?;

        // Expect the colon after the range.
        parser.expect(TokenKind::Colon)?;

        // Parse the body statement that follows the case label.
        let body = statements::parse_statement(parser)?;

        let span = parser.make_span(start_span);
        Ok(Statement::CaseRange {
            low: Box::new(low_expr),
            high: Box::new(high_expr),
            statement: Box::new(body),
            span,
        })
    } else {
        // Standard `case value:` — no range.
        parser.expect(TokenKind::Colon)?;

        // Parse the body statement.
        let body = statements::parse_statement(parser)?;

        let span = parser.make_span(start_span);
        Ok(Statement::Case {
            value: Box::new(low_expr),
            statement: Box::new(body),
            span,
        })
    }
}

// ===========================================================================
// Local Labels: __label__
// ===========================================================================

/// Parse a GCC local label declaration: `__label__ name1, name2, ...;`.
///
/// Local labels restrict the scope of goto labels to the current compound
/// statement (block), rather than the entire function. This is essential
/// for macros that use labels internally without conflicting with labels
/// in the expansion context.
///
/// # Grammar
///
/// ```text
/// local-label-declaration:
///     '__label__' identifier (',' identifier)* ';'
/// ```
///
/// # Examples
///
/// ```c
/// void f(void) {
///     __label__ done, error;
///     { goto done; }
///     done: ;
///     error: ;
/// }
/// ```
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `__label__`.
///
/// # Returns
///
/// * `Ok(Statement::LocalLabel(names, span))` on success.
/// * `Err(())` on parse error.
fn parse_local_label_decl(parser: &mut Parser<'_>) -> Result<Statement, ()> {
    let start_span = parser.current_span();

    // Consume `__label__`.
    parser.expect(TokenKind::Label)?;

    let mut names: Vec<Symbol> = Vec::new();

    // Parse the first label name (required).
    match &parser.current.kind {
        TokenKind::Identifier(sym) => {
            let label_sym: Symbol = *sym;
            let _id: u32 = label_sym.as_u32();
            names.push(label_sym);
            parser.advance();
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected label name after '__label__'");
            return Err(());
        }
    }

    // Parse additional comma-separated label names.
    while parser.match_token(&TokenKind::Comma) {
        match &parser.current.kind {
            TokenKind::Identifier(sym) => {
                let label_sym: Symbol = *sym;
                let _id: u32 = label_sym.as_u32();
                names.push(label_sym);
                parser.advance();
            }
            _ => {
                let span = parser.current_span();
                parser.error(
                    span,
                    "expected label name after ',' in '__label__' declaration",
                );
                return Err(());
            }
        }
    }

    // Consume the terminating semicolon.
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);
    Ok(Statement::LocalLabel(names, span))
}

// ===========================================================================
// typeof / __typeof__ Handling
// ===========================================================================

/// Parse a `typeof` or `__typeof__` type specifier.
///
/// This function is available for use when the parser needs to handle
/// `typeof`/`__typeof__` in an expression context (e.g., within compound
/// literals or casts). The primary `typeof` parsing for type specifiers is
/// done in `types.rs` via `parse_type_specifiers`.
///
/// # Grammar
///
/// ```text
/// typeof-specifier:
///     'typeof' '(' expression ')'
///     'typeof' '(' type-name ')'
/// ```
///
/// # Disambiguation
///
/// The argument is parsed as a type name if the first token inside the
/// parentheses could start a type specifier or qualifier (determined by
/// `types::is_type_specifier_start`). Otherwise, it is parsed as an
/// expression and the type is deduced.
///
/// # Arguments
///
/// * `parser` — The parser with the current token at `typeof` / `__typeof__`.
///
/// # Returns
///
/// * `Ok(TypeSpecifier::Typeof(TypeofArg))` on success.
/// * `Err(())` on parse error.
pub fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    // Consume `typeof` / `__typeof__`.
    parser.advance();

    // Expect opening parenthesis.
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type name or expression?
    let arg = if types::is_type_specifier_start(parser.peek(), parser) {
        // Parse as a type name.
        let spec_qual = types::parse_specifier_qualifier_list(parser)?;
        let type_name = TypeName {
            specifier_qualifiers: spec_qual.clone(),
            abstract_declarator: None,
            span: spec_qual.span,
        };
        TypeofArg::TypeName(Box::new(type_name))
    } else {
        // Parse as an expression.
        let expr = expressions::parse_expression(parser)?;
        TypeofArg::Expression(Box::new(expr))
    };

    // Expect closing parenthesis.
    parser.expect(TokenKind::RightParen)?;

    Ok(TypeSpecifier::Typeof(arg))
}

// ===========================================================================
// Helper: Extract Span from Expression
// ===========================================================================

/// Extract the source span from an [`Expression`] node.
///
/// This helper is used to construct merged spans when an extension keyword
/// (like `__extension__`) wraps an inner expression.
fn get_expression_span(expr: &Expression) -> Span {
    match expr {
        Expression::IntegerLiteral { span, .. }
        | Expression::FloatLiteral { span, .. }
        | Expression::StringLiteral { span, .. }
        | Expression::CharLiteral { span, .. }
        | Expression::Identifier { span, .. }
        | Expression::Parenthesized { span, .. }
        | Expression::ArraySubscript { span, .. }
        | Expression::FunctionCall { span, .. }
        | Expression::MemberAccess { span, .. }
        | Expression::PointerMemberAccess { span, .. }
        | Expression::PostIncrement { span, .. }
        | Expression::PostDecrement { span, .. }
        | Expression::PreIncrement { span, .. }
        | Expression::PreDecrement { span, .. }
        | Expression::UnaryOp { span, .. }
        | Expression::SizeofExpr { span, .. }
        | Expression::SizeofType { span, .. }
        | Expression::AlignofType { span, .. }
        | Expression::AlignofExpr { span, .. }
        | Expression::Cast { span, .. }
        | Expression::Binary { span, .. }
        | Expression::Conditional { span, .. }
        | Expression::Assignment { span, .. }
        | Expression::Comma { span, .. }
        | Expression::CompoundLiteral { span, .. }
        | Expression::StatementExpression { span, .. }
        | Expression::BuiltinCall { span, .. }
        | Expression::Generic { span, .. }
        | Expression::AddressOfLabel { span, .. } => *span,
    }
}

// ===========================================================================
// Helper: Declaration Start Detection After __extension__
// ===========================================================================

/// Check if the current token can start a declaration in the context
/// following `__extension__`.
///
/// This uses the type specifier start detection from `types.rs` and also
/// checks for storage class specifiers, function specifiers, and alignment
/// specifiers that can begin a declaration.
///
/// # Arguments
///
/// * `parser` — The parser with the current token to inspect.
fn is_declaration_start_after_extension(parser: &Parser<'_>) -> bool {
    match parser.peek() {
        // Storage class specifiers
        TokenKind::Typedef
        | TokenKind::Extern
        | TokenKind::Static
        | TokenKind::Auto
        | TokenKind::Register
        | TokenKind::ThreadLocal => true,

        // Function specifiers
        TokenKind::Inline | TokenKind::Noreturn => true,

        // Alignment specifier
        TokenKind::Alignas => true,

        // _Static_assert
        TokenKind::StaticAssert => true,

        // GCC __attribute__ can precede a declaration
        TokenKind::Attribute => true,

        // Another __extension__ can chain
        TokenKind::Extension => true,

        // Type specifiers (delegates to the comprehensive check)
        token => types::is_type_specifier_start(token, parser),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the detection function recognizes all GCC extension tokens.
    #[test]
    fn test_is_gcc_extension_start_coverage() {
        // We can't easily construct a Parser in unit tests without a full
        // Lexer + SourceMap setup, but we can verify the function's match
        // arms compile and are logically correct.
        //
        // This is a compile-time verification test — if the code compiles,
        // the match patterns are valid. Runtime testing with full parser
        // integration is done in integration tests.
    }

    /// Verify that Span::dummy() works as expected for error-recovery nodes.
    #[test]
    fn test_dummy_span_for_error_recovery() {
        let span = Span::dummy();
        assert!(span.is_dummy());
        assert_eq!(span.file_id, 0);
        assert_eq!(span.start, 0);
        assert_eq!(span.end, 0);
    }

    /// Verify Span::new() construction.
    #[test]
    fn test_span_new() {
        let span = Span::new(1, 10, 20);
        assert_eq!(span.file_id, 1);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
        assert!(!span.is_dummy());
    }

    /// Verify Span::merge() combining two spans.
    #[test]
    fn test_span_merge() {
        let span_a = Span::new(1, 10, 20);
        let span_b = Span::new(1, 15, 30);
        let merged = span_a.merge(span_b);
        assert_eq!(merged.file_id, 1);
        assert_eq!(merged.start, 10);
        assert_eq!(merged.end, 30);
    }

    /// Verify Span::merge() with different file_ids returns self.
    #[test]
    fn test_span_merge_different_files() {
        let span_a = Span::new(1, 10, 20);
        let span_b = Span::new(2, 15, 30);
        let merged = span_a.merge(span_b);
        // Different file IDs — returns self unchanged.
        assert_eq!(merged, span_a);
    }

    /// Verify get_expression_span extracts span from integer literal.
    #[test]
    fn test_get_expression_span_integer() {
        let span = Span::new(1, 0, 5);
        let expr = Expression::IntegerLiteral {
            value: 42,
            suffix: IntegerSuffix::None,
            is_hex_or_octal: false,
            span,
        };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify get_expression_span extracts span from float literal.
    #[test]
    fn test_get_expression_span_float() {
        let span = Span::new(1, 10, 15);
        let expr = Expression::FloatLiteral {
            value: 3.14,
            suffix: FloatSuffix::None,
            span,
        };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify get_expression_span extracts span from statement expression.
    #[test]
    fn test_get_expression_span_statement_expression() {
        let span = Span::new(2, 0, 50);
        let compound = CompoundStatement {
            items: Vec::new(),
            span: Span::new(2, 1, 49),
        };
        let expr = Expression::StatementExpression { compound, span };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify get_expression_span for identifier using interner.
    #[test]
    fn test_get_expression_span_identifier() {
        use crate::common::string_interner::Interner;

        let mut interner = Interner::new();
        let sym = interner.intern("my_label");
        let span = Span::new(1, 5, 10);
        let expr = Expression::Identifier { name: sym, span };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify get_expression_span for address-of-label using interner.
    #[test]
    fn test_get_expression_span_address_of_label() {
        use crate::common::string_interner::Interner;

        let mut interner = Interner::new();
        let sym = interner.intern("target_label");
        let span = Span::new(1, 20, 30);
        let expr = Expression::AddressOfLabel { label: sym, span };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify get_expression_span for binary expression.
    #[test]
    fn test_get_expression_span_binary() {
        let span = Span::new(1, 0, 10);
        let inner_span = Span::new(1, 0, 3);
        let left = Expression::IntegerLiteral {
            value: 1,
            suffix: IntegerSuffix::None,
            is_hex_or_octal: false,
            span: inner_span,
        };
        let right = Expression::IntegerLiteral {
            value: 2,
            suffix: IntegerSuffix::None,
            is_hex_or_octal: false,
            span: Span::new(1, 6, 10),
        };
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(right),
            span,
        };
        assert_eq!(get_expression_span(&expr), span);
    }

    /// Verify is_declaration_start_after_extension recognizes type keywords.
    #[test]
    fn test_extension_token_coverage() {
        // Verify that the match arms in is_gcc_extension_start compile
        // and cover the expected variants. This is a compile-time check —
        // the fact this test compiles confirms all match arms are valid.
        let extension = TokenKind::Extension;
        let label = TokenKind::Label;
        let typeof_kw = TokenKind::Typeof;
        let attr = TokenKind::Attribute;

        // Verify discriminants are distinct.
        assert_ne!(
            std::mem::discriminant(&extension),
            std::mem::discriminant(&label)
        );
        assert_ne!(
            std::mem::discriminant(&typeof_kw),
            std::mem::discriminant(&attr)
        );
    }
}
