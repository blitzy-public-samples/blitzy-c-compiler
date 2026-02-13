//! GCC language extension dispatch and parsing module for the BCC C11 parser.
//!
//! Implements parsing support for GCC-specific language extensions that are
//! required for Linux kernel compilation. These extensions go beyond the
//! C11 standard and are widely used in system-level C code.
//!
//! # Supported GCC Extensions
//!
//! - **`__extension__` keyword:** Suppresses pedantic warnings for the
//!   following expression or declaration.
//! - **Statement expressions:** `({ ... })` — compound statements used as
//!   sub-expressions (e.g., in kernel `min()`/`max()` macros).
//! - **Computed gotos:** `goto *expr` and `&&label` — indirect jumps via
//!   label addresses for jump tables and threaded interpreters.
//! - **Case ranges:** `case LOW ... HIGH:` — inclusive range matching in
//!   switch statements.
//! - **Conditional omission:** `x ?: y` — shorthand for `x ? x : y` where
//!   the condition is evaluated only once.
//! - **Zero-length arrays:** `int arr[0]` — legacy GCC extension for
//!   variable-size trailing buffers (pre-flexible-array-member era).
//! - **Flexible array members:** `int arr[]` — C99/C11 flexible array
//!   members with validation that they are the last struct field.
//! - **Transparent unions:** `__attribute__((transparent_union))` — union
//!   types whose first member type can be passed directly by callers.
//! - **Local labels:** `__label__ name1, name2, ...;` — block-scoped labels
//!   to avoid name conflicts in macro expansions.
//!
//! # Architecture
//!
//! This module acts as a dispatcher, routing GCC-specific constructs to
//! their parsing logic. It is called from `expressions.rs`, `statements.rs`,
//! and `declarations.rs` when GCC extension tokens are encountered. All
//! functions accept a mutable reference to the `Parser` and return AST nodes
//! or validation results.
//!
//! # Error Handling
//!
//! Per Section 0.7.6, unknown GCC extensions are diagnosed with clear error
//! messages identifying the unsupported construct. The compiler never silently
//! miscompiles unknown extensions.

use super::ast::{
    Attribute, BlockItem, Declaration, DerivedDeclarator, Expression, Span, Statement,
};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ---------------------------------------------------------------------------
// §10: Extension Detection Helper
// ---------------------------------------------------------------------------

/// Checks whether a token kind marks the beginning of a GCC extension construct.
///
/// Returns `true` for any token that can start a GCC-specific language
/// extension, allowing the parser to dispatch into extension-specific parsing
/// paths. This is used by `declarations.rs` and `expressions.rs` to detect
/// when normal C11 parsing should yield to extension handling.
///
/// # Recognised Extension Starters
///
/// | Token | Extension |
/// |-------|-----------|
/// | `__extension__` | Warning suppression prefix |
/// | `__label__` | Local label declarations |
/// | `__attribute__` | GCC attributes |
/// | `typeof` / `__typeof__` | Type-of operator |
/// | `asm` / `__asm__` | Inline assembly |
///
/// # Arguments
///
/// * `token` — The token kind to inspect.
///
/// # Returns
///
/// `true` if `token` can begin a GCC extension construct, `false` otherwise.
pub fn is_gcc_extension_start(token: &TokenKind) -> bool {
    matches!(
        token,
        TokenKind::Extension
            | TokenKind::Label
            | TokenKind::Attribute
            | TokenKind::TypeofKeyword
            | TokenKind::AsmKeyword
    )
}

// ---------------------------------------------------------------------------
// §1: __extension__ Keyword Handling
// ---------------------------------------------------------------------------

/// Parses an expression preceded by the `__extension__` keyword.
///
/// When `__extension__` appears in expression context, GCC pedantic warnings
/// are suppressed for the immediately following expression. This is commonly
/// used in system headers and kernel code to silence warnings about GCC
/// extensions in `-pedantic` mode.
///
/// The current token must be `__extension__` when this function is called.
/// After consuming it, the inner expression is parsed normally. The resulting
/// AST node is not structurally different — the extension flag is carried on
/// declaration specifiers where applicable; for bare expressions the keyword
/// is consumed and the inner expression returned directly.
///
/// # Grammar
///
/// ```text
/// extension-expression:
///     '__extension__' expression
/// ```
///
/// # Errors
///
/// Propagates any parse error from the inner expression.
pub fn parse_extension_expr(parser: &mut Parser) -> Result<Expression, ParseError> {
    let start_span = parser.current().span;
    // Consume the __extension__ keyword.
    parser.advance();

    // Parse the inner expression. The expression parser in the parent module
    // handles all operator precedence. We call the assignment-expression level
    // via `parse_assignment_expression` since __extension__ can precede any
    // assignment-level expression.
    let inner = super::expressions::parse_assignment_expression(parser)?;

    // The __extension__ keyword does not produce a distinct AST node for
    // expressions — its effect is purely diagnostic suppression. We merge
    // the span to cover the keyword and return the inner expression.
    // NOTE: The span of the inner expression already starts after __extension__,
    // but we want the combined span for accurate diagnostics.
    let _ = Span::merge(start_span, inner.span());

    Ok(inner)
}

/// Parses a declaration preceded by the `__extension__` keyword.
///
/// When `__extension__` precedes a declaration, GCC pedantic warnings are
/// suppressed for that declaration. The `has_extension` flag is set on the
/// resulting `DeclarationSpecifiers` so downstream phases know the keyword
/// was present.
///
/// The current token must be `__extension__` when this function is called.
///
/// # Grammar
///
/// ```text
/// extension-declaration:
///     '__extension__' declaration
/// ```
///
/// # Errors
///
/// Propagates any parse error from the inner declaration.
pub fn parse_extension_decl(parser: &mut Parser) -> Result<Declaration, ParseError> {
    let _start_span = parser.current().span;
    // Consume the __extension__ keyword.
    parser.advance();

    // Parse the inner declaration using the standard declaration parser.
    let mut decl = super::declarations::parse_declaration(parser)?;

    // Mark the declaration as having been preceded by __extension__.
    // This flag suppresses certain pedantic diagnostics during semantic
    // analysis.
    mark_declaration_extension(&mut decl);

    Ok(decl)
}

/// Sets the `has_extension` flag on a declaration's specifiers.
///
/// For declaration variants that carry `DeclarationSpecifiers`, this sets
/// the `has_extension` field to `true`. For variants without specifiers
/// (e.g., `Empty`, `Error`), this is a no-op.
fn mark_declaration_extension(decl: &mut Declaration) {
    match decl {
        Declaration::Variable {
            ref mut specifiers, ..
        } => {
            specifiers.has_extension = true;
        }
        Declaration::FunctionDef {
            ref mut specifiers, ..
        } => {
            specifiers.has_extension = true;
        }
        Declaration::FunctionDecl {
            ref mut specifiers, ..
        } => {
            specifiers.has_extension = true;
        }
        Declaration::Typedef {
            ref mut specifiers, ..
        } => {
            specifiers.has_extension = true;
        }
        // StructDef, UnionDef, EnumDef, StaticAssert, Empty, Error do not
        // carry DeclarationSpecifiers at the top level, so the extension
        // flag cannot be propagated. For struct/union/enum definitions the
        // flag would have been on the enclosing variable declaration.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// §2: Statement Expressions — ({ ... })
// ---------------------------------------------------------------------------

/// Parses a GCC statement expression: `({ stmt1; stmt2; expr; })`.
///
/// A statement expression is a compound statement wrapped in parentheses and
/// used as a sub-expression. The value of the statement expression is the
/// value of the last expression-statement in the block. This construct is
/// used extensively in Linux kernel macros (`min()`, `max()`, `container_of()`).
///
/// **Precondition:** The current token is `(` and the next token is `{`.
/// The caller has already verified this two-token lookahead.
///
/// # Grammar
///
/// ```text
/// statement-expression:
///     '(' compound-statement ')'
///
/// compound-statement:
///     '{' block-item-list? '}'
/// ```
///
/// # Returns
///
/// `Expression::StatementExpression` with the block items as the body.
///
/// # Errors
///
/// - Missing `{` after `(`.
/// - Unmatched braces or parentheses.
/// - Errors within the compound statement body.
pub fn parse_statement_expression(parser: &mut Parser) -> Result<Expression, ParseError> {
    let start_span = parser.current().span;

    // Consume the opening `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Consume the opening `{`.
    parser.expect(TokenKind::LeftBrace)?;

    // Parse block items (declarations and statements) until we hit `}`.
    let mut body: Vec<BlockItem> = Vec::new();
    while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
        // Determine whether the next construct is a declaration or statement.
        if parser.is_declaration_start() {
            match super::declarations::parse_declaration(parser) {
                Ok(decl) => body.push(BlockItem::Declaration(decl)),
                Err(e) => {
                    // Emit diagnostic and attempt recovery by synchronizing.
                    parser.diagnostics.error(
                        e.span,
                        format!(
                            "error in statement expression body: {}",
                            e.message
                        ),
                    );
                    parser.synchronize();
                    // If synchronization brought us to `}`, stop.
                    if parser.check(TokenKind::RightBrace) {
                        break;
                    }
                }
            }
        } else {
            match super::statements::parse_statement(parser) {
                Ok(stmt) => body.push(BlockItem::Statement(stmt)),
                Err(e) => {
                    parser.diagnostics.error(
                        e.span,
                        format!(
                            "error in statement expression body: {}",
                            e.message
                        ),
                    );
                    parser.synchronize();
                    if parser.check(TokenKind::RightBrace) {
                        break;
                    }
                }
            }
        }
    }

    // Consume the closing `}`.
    parser.expect(TokenKind::RightBrace)?;

    // Consume the closing `)`.
    let end_token = parser.current();
    let end_span = end_token.span;
    parser.expect(TokenKind::RightParen)?;

    let span = Span::merge(start_span, end_span);

    Ok(Expression::StatementExpression { body, span })
}

// ---------------------------------------------------------------------------
// §3: Computed Goto — goto *expr and &&label
// ---------------------------------------------------------------------------

/// Parses a computed goto statement: `goto *expression;`.
///
/// A computed goto is a GCC extension that performs an indirect jump to the
/// address held in an expression. The target expression must evaluate to
/// `void *` (typically obtained via `&&label`). This is used in the Linux
/// kernel for dispatch tables and threaded interpreters.
///
/// **Precondition:** The `goto` keyword has already been consumed and the
/// current token is `*` (indicating this is a computed goto, not a regular
/// goto).
///
/// # Grammar
///
/// ```text
/// computed-goto-statement:
///     'goto' '*' expression ';'
/// ```
///
/// # Errors
///
/// - Missing `*` after `goto` (should not happen if precondition is met).
/// - Invalid target expression.
/// - Missing semicolon after expression.
pub fn parse_computed_goto(parser: &mut Parser, goto_span: Span) -> Result<Statement, ParseError> {
    // Consume the `*` token.
    parser.expect(TokenKind::Star)?;

    // Parse the target expression. Any expression yielding `void *` is valid.
    let target = super::expressions::parse_expression(parser)?;

    // Consume the terminating semicolon.
    let end_span = parser.current().span;
    parser.expect_semicolon()?;

    let span = Span::merge(goto_span, end_span);

    Ok(Statement::ComputedGoto {
        target: Box::new(target),
        span,
    })
}

/// Parses a label address expression: `&&label`.
///
/// Takes the address of a named label, producing a `void *` value that can
/// be stored and later used with `goto *expr`. The `&&` is two adjacent
/// ampersand characters forming `TokenKind::AmpAmp`.
///
/// **Precondition:** The current token is `&&` (AmpAmp).
///
/// # Grammar
///
/// ```text
/// label-address:
///     '&&' identifier
/// ```
///
/// # Errors
///
/// - Missing identifier after `&&`.
/// - Identifier is not a valid label name.
pub fn parse_label_address(parser: &mut Parser) -> Result<Expression, ParseError> {
    let start_span = parser.current().span;

    // Consume the `&&` token.
    parser.advance(); // consume AmpAmp

    // The next token must be an identifier naming the label.
    let label_token = parser.current().clone();
    match &label_token.kind {
        TokenKind::Identifier(sym) => {
            let label = *sym;
            let end_span = label_token.span;
            parser.advance(); // consume the identifier

            let span = Span::merge(start_span, end_span);
            Ok(Expression::LabelAddress { label, span })
        }
        _ => {
            let err_span = parser.current().span;
            parser.diagnostics.error(
                err_span,
                "expected label name after '&&' in label address expression",
            );
            Err(ParseError {
                span: err_span,
                message: "expected label name after '&&'".to_string(),
                expected: Some("identifier".to_string()),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// §4: Case Ranges — case LOW ... HIGH:
// ---------------------------------------------------------------------------

/// Parses a case range: `case LOW ... HIGH: body`.
///
/// After the caller has parsed `case expr`, this function is called when the
/// next token is `...` (ellipsis), indicating a GCC case range extension. The
/// range is inclusive: all values from `low` through `high` (inclusive) match.
///
/// Both `low` and `high` must be integer constant expressions. The `...` token
/// is `TokenKind::Ellipsis` — the same token used for variadic parameters, but
/// distinguished by context (we are inside a `case` label, not a parameter
/// list).
///
/// **Precondition:** `low` has been parsed, and the current token is `...`.
///
/// # Grammar
///
/// ```text
/// case-range:
///     'case' constant-expression '...' constant-expression ':' statement
/// ```
///
/// # Errors
///
/// - Missing high expression after `...`.
/// - Missing `:` after high expression.
/// - Errors in the body statement.
pub fn parse_case_range(
    parser: &mut Parser,
    low: Expression,
    case_span: Span,
) -> Result<Statement, ParseError> {
    // Consume the `...` (ellipsis) token.
    parser.expect(TokenKind::Ellipsis)?;

    // Parse the high expression of the range.
    let high = super::expressions::parse_conditional_expression(parser)?;

    // Consume the `:` following the range.
    parser.expect(TokenKind::Colon)?;

    // Parse the body statement following the case range label.
    let body = super::statements::parse_statement(parser)?;

    let span = Span::merge(case_span, body.span());

    Ok(Statement::CaseRange {
        low: Box::new(low),
        high: Box::new(high),
        body: Box::new(body),
        span,
    })
}

// ---------------------------------------------------------------------------
// §5: Conditional Expression Omission — x ?: y
// ---------------------------------------------------------------------------

/// Parses a conditional expression with omitted middle operand: `x ?: y`.
///
/// In GCC's extension, `a ?: b` is shorthand for `a ? a : b`, except that
/// `a` is evaluated only once. When the parser encounters `?` immediately
/// followed by `:` (i.e., the "then" expression is omitted), this function
/// is called.
///
/// **Precondition:** The `condition` expression and the `?` token have been
/// consumed. The current token is `:`.
///
/// # Grammar
///
/// ```text
/// conditional-omission:
///     conditional-expression '?' ':' assignment-expression
/// ```
///
/// # Returns
///
/// `Expression::Conditional` with `then_expr: None` to signal the omission.
///
/// # Errors
///
/// - Errors in the alternative (else) expression.
pub fn parse_conditional_omission(
    parser: &mut Parser,
    condition: Expression,
    question_span: Span,
) -> Result<Expression, ParseError> {
    // The `?` has already been consumed by the caller. The current token is `:`.
    // Consume the `:`.
    parser.expect(TokenKind::Colon)?;

    // Parse the alternative ("else") expression at the assignment level.
    let alternative = super::expressions::parse_assignment_expression(parser)?;

    let span = Span::merge(condition.span(), alternative.span());

    Ok(Expression::Conditional {
        condition: Box::new(condition),
        then_expr: None,
        else_expr: Box::new(alternative),
        span: Span::merge(question_span, span),
    })
}

// ---------------------------------------------------------------------------
// §6: Zero-Length Arrays — int arr[0]
// ---------------------------------------------------------------------------

/// Validates whether an array size expression represents a zero-length array.
///
/// Zero-length arrays (`int arr[0]`) are a GCC extension used in older Linux
/// kernel code (pre-flexible-array-member era) for variable-size trailing
/// buffers in structures. This function checks if the size expression is an
/// integer literal with value `0`.
///
/// When a zero-length array is detected, a diagnostic note is emitted to
/// indicate the GCC extension usage, but no error is raised — the construct
/// is permitted.
///
/// # Arguments
///
/// * `parser` — The parser, used for emitting diagnostics.
/// * `size_expr` — The array size expression to check.
///
/// # Returns
///
/// `true` if the expression is a zero-length array (integer literal 0),
/// `false` otherwise.
pub fn validate_zero_length_array(parser: &mut Parser, size_expr: &Expression) -> bool {
    if let Expression::IntegerLiteral { value: 0, span, .. } = size_expr {
        parser.diagnostics.note(
            *span,
            "zero-length array is a GCC extension; consider using a flexible array member instead",
        );
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// §7: Flexible Array Members — int arr[]
// ---------------------------------------------------------------------------

/// Validates that a flexible array member appears as the last field of a struct.
///
/// A flexible array member (`int arr[]`) has an unspecified array size and must
/// be the last member of a structure. The structure must also have at least one
/// other named member before the flexible array member (C11 §6.7.2.1 ¶18).
///
/// This function inspects a `Declaration::StructDef` or equivalent to verify
/// these constraints. It emits appropriate diagnostics if the constraints are
/// violated.
///
/// # Arguments
///
/// * `parser` — The parser, used for emitting diagnostics.
/// * `decl` — The declaration to validate (typically a struct definition).
///
/// # Returns
///
/// `true` if the declaration contains a valid flexible array member (or no
/// flexible array member at all), `false` if the constraints are violated.
pub fn validate_flexible_array_member(parser: &mut Parser, decl: &Declaration) -> bool {
    // Extract the field list from a struct definition.
    let fields = match decl {
        Declaration::StructDef { fields, .. } => fields,
        Declaration::Variable { specifiers, .. } => {
            // A variable declaration might embed an inline struct definition
            // in its type specifiers. For now, we delegate struct-level
            // validation to when the struct is actually defined. Return true
            // to indicate no constraint violation at this level.
            let _ = specifiers;
            return true;
        }
        // Flexible array members only apply to structs.
        _ => return true,
    };

    // No fields means no flexible array member — trivially valid.
    if fields.is_empty() {
        return true;
    }

    // Check each field for a flexible array member (array with no size).
    let field_count = fields.len();
    for (field_idx, field) in fields.iter().enumerate() {
        for declarator_entry in &field.declarators {
            if let Some(ref declarator) = declarator_entry.declarator {
                // Check if any derived declarator is an incomplete array.
                for derived in &declarator.derived {
                    if let DerivedDeclarator::Array {
                        size: None,
                        is_static: false,
                        ..
                    } = derived
                    {
                        // Found a flexible array member. Validate constraints:
                        //
                        // 1. Must be in the last field declaration.
                        if field_idx != field_count - 1 {
                            parser.diagnostics.error(
                                field.span,
                                "flexible array member must be the last field in the struct",
                            );
                            return false;
                        }

                        // 2. Struct must have at least one other named member.
                        if field_count < 2 {
                            parser.diagnostics.error(
                                field.span,
                                "flexible array member in a struct with no other named members",
                            );
                            return false;
                        }

                        // Valid flexible array member.
                        return true;
                    }
                }
            }
        }
    }

    // No flexible array member found — trivially valid.
    true
}

// ---------------------------------------------------------------------------
// §8: Transparent Unions — __attribute__((transparent_union))
// ---------------------------------------------------------------------------

/// Parses and annotates a union declaration with the `transparent_union` attribute.
///
/// When a union type has `__attribute__((transparent_union))`, function
/// parameters of that union type accept any member type directly without
/// requiring an explicit union construction. The attribute itself is parsed
/// by `attributes.rs`; this function handles the semantic dispatch: it
/// accepts an already-parsed union declaration and attaches the transparent
/// union attribute to it.
///
/// **Precondition:** The `__attribute__((transparent_union))` has been parsed
/// and is available in the attribute list. The union declaration body has
/// been parsed normally.
///
/// # Arguments
///
/// * `parser` — The parser, used for diagnostics.
///
/// # Returns
///
/// A `Declaration::UnionDef` with the transparent_union `Attribute` attached.
///
/// # Errors
///
/// - The declaration is not a union.
/// - The union has no members.
/// - Parse errors in the union body.
pub fn parse_transparent_union(parser: &mut Parser) -> Result<Declaration, ParseError> {
    let start_span = parser.current().span;

    // We expect to be positioned at a point where we can parse a declaration.
    // The __attribute__((transparent_union)) may have been parsed before the
    // `union` keyword. We parse the declaration normally, then ensure it is
    // a union and annotate it.
    let decl = super::declarations::parse_declaration(parser)?;

    match decl {
        Declaration::UnionDef {
            name,
            fields,
            mut attrs,
            span,
        } => {
            // Construct the transparent_union attribute and prepend it.
            let attr = Attribute {
                name: parser.interner.intern("transparent_union"),
                args: Vec::new(),
                span: start_span,
            };
            attrs.insert(0, attr);

            // Validate: the union must have at least one member.
            if fields.is_empty() {
                parser.diagnostics.warning(
                    span,
                    "transparent_union applied to union with no members",
                );
            }

            Ok(Declaration::UnionDef {
                name,
                fields,
                attrs,
                span,
            })
        }
        Declaration::Variable {
            specifiers,
            declarators,
            mut attrs,
            span,
        } => {
            // A variable declaration of union type — the transparent_union
            // attribute applies to the union type specifier within.
            let attr = Attribute {
                name: parser.interner.intern("transparent_union"),
                args: Vec::new(),
                span: start_span,
            };
            attrs.insert(0, attr);

            Ok(Declaration::Variable {
                specifiers,
                declarators,
                attrs,
                span,
            })
        }
        other => {
            let err_span = other.span();
            parser.diagnostics.error(
                err_span,
                "transparent_union attribute can only be applied to union declarations",
            );
            // Return the declaration as-is rather than failing hard — the
            // semantic analyzer will catch the invalid attribute application.
            Ok(other)
        }
    }
}

// ---------------------------------------------------------------------------
// §9: Local Labels — __label__ Declarations
// ---------------------------------------------------------------------------

/// Parses a `__label__` local label declaration.
///
/// GCC's `__label__` extension declares label names that are scoped to the
/// enclosing compound statement rather than the entire function. This is
/// critical for kernel macros that use labels internally — without local
/// label scoping, multiple expansions of the same macro would produce
/// duplicate label errors.
///
/// **Precondition:** The current token is `__label__`.
///
/// # Grammar
///
/// ```text
/// local-label-declaration:
///     '__label__' identifier (',' identifier)* ';'
/// ```
///
/// # Returns
///
/// A `Vec<Symbol>` containing the interned names of the declared local labels.
///
/// # Errors
///
/// - Missing identifier after `__label__`.
/// - Missing `;` terminator.
/// - Duplicate label names (emits warning, does not fail).
pub fn parse_local_labels(parser: &mut Parser) -> Result<Vec<Symbol>, ParseError> {
    let start_span = parser.current().span;

    // Consume the `__label__` keyword.
    parser.advance();

    let mut labels: Vec<Symbol> = Vec::new();

    // Parse the first label name (mandatory).
    let first_label = parse_label_identifier(parser, start_span)?;
    labels.push(first_label);

    // Parse additional comma-separated label names.
    while parser.eat(TokenKind::Comma) {
        let label = parse_label_identifier(parser, parser.current().span)?;

        // Check for duplicate label names in this declaration.
        if labels.contains(&label) {
            parser.diagnostics.warning(
                parser.current().span,
                "duplicate label name in __label__ declaration",
            );
        }

        labels.push(label);
    }

    // Consume the terminating semicolon.
    parser.expect_semicolon()?;

    Ok(labels)
}

/// Extracts an identifier as a `Symbol` for local label parsing.
///
/// Helper for `parse_local_labels` — reads the current token, which must be
/// an `Identifier`, and returns its interned `Symbol`.
fn parse_label_identifier(parser: &mut Parser, context_span: Span) -> Result<Symbol, ParseError> {
    let token = parser.current().clone();
    match &token.kind {
        TokenKind::Identifier(sym) => {
            let label = *sym;
            parser.advance(); // consume the identifier
            Ok(label)
        }
        _ => {
            parser.diagnostics.error(
                token.span,
                "expected label name in __label__ declaration",
            );
            Err(ParseError {
                span: Span::merge(context_span, token.span),
                message: "expected identifier in __label__ declaration".to_string(),
                expected: Some("identifier".to_string()),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// §11: Comprehensive Error Handling Utilities
// ---------------------------------------------------------------------------

/// Attempts to parse a GCC extension construct, producing an error expression
/// on failure with graceful recovery.
///
/// This is used by the expression parser when it encounters a potential GCC
/// extension token but cannot determine the specific extension. It attempts
/// to skip the unknown construct and continue parsing.
///
/// Per Section 0.7.6, unknown GCC extensions must never be silently
/// miscompiled — they are always diagnosed.
///
/// # Arguments
///
/// * `parser` — The parser.
/// * `token_kind` — The unrecognised extension token kind.
///
/// # Returns
///
/// An `Expression::Error` node with a diagnostic emitted.
pub fn diagnose_unknown_extension_expr(
    parser: &mut Parser,
    token_kind: &TokenKind,
) -> Expression {
    let span = parser.current().span;
    parser.diagnostics.error(
        span,
        format!(
            "unsupported GCC extension in expression context: '{}'",
            format_token_kind_for_diagnostic(token_kind),
        ),
    );

    // Attempt to skip the unknown token so the parser can continue.
    parser.advance();

    Expression::Error { span }
}

/// Attempts to handle an unknown GCC extension in declaration context,
/// producing an error declaration with graceful recovery.
///
/// Per Section 0.7.6, the compiler must never silently miscompile unknown
/// extensions. This function emits a clear diagnostic and returns an error
/// placeholder.
///
/// # Arguments
///
/// * `parser` — The parser.
/// * `token_kind` — The unrecognised extension token kind.
///
/// # Returns
///
/// A `Declaration::Error` node with a diagnostic emitted.
pub fn diagnose_unknown_extension_decl(
    parser: &mut Parser,
    token_kind: &TokenKind,
) -> Declaration {
    let span = parser.current().span;
    parser.diagnostics.error(
        span,
        format!(
            "unsupported GCC extension in declaration context: '{}'",
            format_token_kind_for_diagnostic(token_kind),
        ),
    );

    // Attempt to skip the unknown token.
    parser.advance();

    Declaration::Error { span }
}

/// Formats a `TokenKind` into a human-readable string for diagnostic messages.
///
/// This avoids depending on `Display` implementations that may not produce
/// user-friendly output for all token variants.
fn format_token_kind_for_diagnostic(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::Extension => "__extension__",
        TokenKind::Label => "__label__",
        TokenKind::Attribute => "__attribute__",
        TokenKind::TypeofKeyword => "typeof/__typeof__",
        TokenKind::AsmKeyword => "asm/__asm__",
        TokenKind::AmpAmp => "&&",
        TokenKind::Star => "*",
        TokenKind::Ellipsis => "...",
        _ => "<unknown GCC extension>",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `is_gcc_extension_start` correctly identifies all GCC
    /// extension starter tokens.
    #[test]
    fn test_is_gcc_extension_start_positive() {
        assert!(is_gcc_extension_start(&TokenKind::Extension));
        assert!(is_gcc_extension_start(&TokenKind::Label));
        assert!(is_gcc_extension_start(&TokenKind::Attribute));
        assert!(is_gcc_extension_start(&TokenKind::TypeofKeyword));
        assert!(is_gcc_extension_start(&TokenKind::AsmKeyword));
    }

    /// Verify that `is_gcc_extension_start` rejects non-extension tokens.
    #[test]
    fn test_is_gcc_extension_start_negative() {
        assert!(!is_gcc_extension_start(&TokenKind::Star));
        assert!(!is_gcc_extension_start(&TokenKind::Semicolon));
        assert!(!is_gcc_extension_start(&TokenKind::LeftParen));
        assert!(!is_gcc_extension_start(&TokenKind::Colon));
        assert!(!is_gcc_extension_start(&TokenKind::Eof));
        assert!(!is_gcc_extension_start(&TokenKind::Identifier(Symbol::EMPTY)));
        assert!(!is_gcc_extension_start(&TokenKind::AmpAmp));
        assert!(!is_gcc_extension_start(&TokenKind::Question));
        assert!(!is_gcc_extension_start(&TokenKind::Ellipsis));
        assert!(!is_gcc_extension_start(&TokenKind::Comma));
        assert!(!is_gcc_extension_start(&TokenKind::LeftBrace));
        assert!(!is_gcc_extension_start(&TokenKind::RightBrace));
        assert!(!is_gcc_extension_start(&TokenKind::RightParen));
    }

    /// Verify the diagnostic formatter for known extension tokens.
    #[test]
    fn test_format_token_kind() {
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Extension), "__extension__");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Label), "__label__");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Attribute), "__attribute__");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::TypeofKeyword), "typeof/__typeof__");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::AsmKeyword), "asm/__asm__");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::AmpAmp), "&&");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Star), "*");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Ellipsis), "...");
        assert_eq!(format_token_kind_for_diagnostic(&TokenKind::Semicolon), "<unknown GCC extension>");
    }
}
