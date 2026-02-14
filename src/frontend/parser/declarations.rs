// =============================================================================
// src/frontend/parser/declarations.rs — C11 Declaration Parsing
// =============================================================================
//
// This module implements parsing for all C11 declaration forms:
//
// - Variable declarations with initializers
// - Function declarations (prototypes) and function definitions
// - Typedef declarations
// - Struct/union declarations with field lists and bitfields
// - Enum declarations with enumerator lists
// - Storage class specifiers: auto, register, static, extern, typedef,
//   _Thread_local
// - _Static_assert compile-time assertions
// - _Alignas alignment specifiers
// - Anonymous struct/union members (C11)
// - Function parameter lists with ellipsis for variadic functions
// - Abstract declarators for type names in casts, sizeof, _Generic
// - GCC __attribute__((...)) annotations at all applicable positions
// - __extension__ prefix handling for declarations
//
// The complex "declaration mirrors use" C syntax is handled via the
// declarator → direct-declarator recursion with pointer, array, and
// function derivations.
//
// All functions produce AST nodes defined in `super::ast`.
// =============================================================================

use super::ast::{
    AbstractDeclarator, AlignasSpecifier, Attribute, Declaration, DeclarationSpecifiers,
    Declarator, DerivedDeclarator, Designator, Enumerator, FieldDeclaration, FieldDeclarator,
    FunctionSpecifiers, InitDeclarator, Initializer, InitializerItem, Parameter, ParameterList,
    SpecifierQualifierList, Span, Statement, StorageClass, TypeName, TypeQualifiers,
    TypeSpecifier, TypeofOperand,
};
use super::Parser;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;
use crate::frontend::parser::ParseError;

// =============================================================================
// Internal helpers
// =============================================================================

/// Extracts the span from a `TypeSpecifier` variant that carries one.
fn type_specifier_span(spec: &TypeSpecifier) -> Option<Span> {
    match spec {
        TypeSpecifier::Struct { span, .. }
        | TypeSpecifier::Union { span, .. }
        | TypeSpecifier::Enum { span, .. }
        | TypeSpecifier::TypedefName { span, .. }
        | TypeSpecifier::Typeof { span, .. } => Some(*span),
        _ => None,
    }
}

/// Heuristic to determine if a declarator's derived list represents a
/// function declaration (vs. a function-pointer variable declaration).
///
/// A declarator is a function declaration when the outermost (last) derived
/// element is `DerivedDeclarator::Function` and the name was parsed without
/// grouping parentheses (i.e., `int f(int)` not `int (*f)(int)`).
fn is_function_declaration(derived: &[DerivedDeclarator], has_grouping: bool) -> bool {
    if has_grouping {
        return false;
    }
    matches!(derived.last(), Some(DerivedDeclarator::Function { .. }))
}

/// Returns `true` if the current token could begin a type specifier or
/// qualifier (used for specifier-qualifier-list parsing context).
fn is_specifier_qualifier_start(parser: &Parser<'_>) -> bool {
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

        // struct / union / enum
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // Type qualifiers
        TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict | TokenKind::Atomic => true,

        // GCC extensions used as type specifiers
        TokenKind::TypeofKeyword | TokenKind::Attribute | TokenKind::Extension => true,

        // Typedef name
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),

        _ => false,
    }
}

/// Returns `true` if the current token could begin an abstract declarator.
fn can_start_abstract_declarator(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Star | TokenKind::LeftParen | TokenKind::LeftBracket
    )
}

/// Helper to extract an identifier `Symbol` from the current token.
/// Emits a diagnostic and returns `Err` if the current token is not an
/// identifier.
fn expect_identifier(parser: &mut Parser<'_>, context: &str) -> Result<Symbol, ParseError> {
    match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            let name = *sym;
            parser.advance();
            Ok(name)
        }
        _ => {
            let span = parser.current().span;
            let msg = format!("expected identifier in {context}, found '{}'", parser.current().kind);
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier".to_string()),
            })
        }
    }
}

// =============================================================================
// §1: Top-level Entry Points
// =============================================================================

/// Parse an external (file-scope) declaration or function definition.
///
/// At file scope, declarations can be:
/// - Variable declarations: `int x = 5;`
/// - Function definitions: `int main(void) { return 0; }`
/// - Function prototypes: `extern int printf(const char *, ...);`
/// - Typedef declarations: `typedef unsigned long size_t;`
/// - Struct/union/enum definitions: `struct point { int x; int y; };`
/// - `_Static_assert(cond, "msg");`
/// - Empty declarations: `;`
///
/// This function also handles `__extension__` prefixes by delegating to
/// `gcc_extensions::parse_extension_decl`.
pub fn parse_external_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.current().span;

    // Handle __extension__ prefix
    if parser.check(TokenKind::Extension)
        && super::gcc_extensions::is_gcc_extension_start(&parser.current().kind)
    {
        return super::gcc_extensions::parse_extension_decl(parser);
    }

    // Handle _Static_assert
    if parser.check(TokenKind::StaticAssert) {
        return parse_static_assert(parser);
    }

    // Handle empty declaration (lone semicolon)
    if parser.check(TokenKind::Semicolon) {
        let span = parser.advance();
        return Ok(Declaration::Empty { span });
    }

    // Parse declaration specifiers
    let specifiers = match parse_declaration_specifiers(parser) {
        Ok(s) => s,
        Err(_) => {
            // Error recovery: skip to synchronization point and return error node
            parser.synchronize();
            return Ok(Declaration::Error {
                span: Span::merge(start, parser.current().span),
            });
        }
    };

    // After specifiers, we might have:
    // - A semicolon (standalone struct/union/enum definition or declaration with no declarator)
    // - A declarator (variable, function, typedef, etc.)

    // Standalone tag definition followed by semicolon
    if parser.check(TokenKind::Semicolon) {
        let end_span = parser.advance();
        return make_standalone_tag_declaration(specifiers, start, end_span);
    }

    // Collect pre-declarator attributes
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Parse the first declarator
    let (first_declarator, has_grouping) = parse_declarator_inner(parser)?;

    // Collect post-declarator attributes
    let mut all_attrs = attrs;
    if parser.check(TokenKind::Attribute) {
        all_attrs.extend(super::attributes::parse_attribute_list(parser)?);
    }

    // Check for function definition: specifiers declarator { body }
    if parser.check(TokenKind::LeftBrace) {
        let body = super::statements::parse_compound_statement(parser)?;
        let end_span = match &body {
            Statement::Compound { span, .. } => *span,
            _ => parser.current().span,
        };
        return Ok(Declaration::FunctionDef {
            specifiers,
            declarator: first_declarator,
            attrs: all_attrs,
            body: Box::new(body),
            span: Span::merge(start, end_span),
        });
    }

    // Otherwise, it's a declaration (variable, function prototype, or typedef)
    parse_declaration_rest(parser, specifiers, first_declarator, all_attrs, has_grouping, start)
}

/// Parse a declaration in block (local) scope.
///
/// Block-scope declarations are similar to external declarations but cannot
/// be function definitions (those are only valid at file scope).
///
/// Handles:
/// - Variable declarations with initializers
/// - Typedef declarations
/// - Struct/union/enum definitions
/// - _Static_assert
/// - Empty declarations
/// - __extension__ prefixed declarations
pub fn parse_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.current().span;

    // Handle __extension__ prefix
    if parser.check(TokenKind::Extension)
        && super::gcc_extensions::is_gcc_extension_start(&parser.current().kind)
    {
        return super::gcc_extensions::parse_extension_decl(parser);
    }

    // Handle _Static_assert
    if parser.check(TokenKind::StaticAssert) {
        return parse_static_assert(parser);
    }

    // Handle empty declaration
    if parser.check(TokenKind::Semicolon) {
        let span = parser.advance();
        return Ok(Declaration::Empty { span });
    }

    // Parse declaration specifiers
    let specifiers = parse_declaration_specifiers(parser)?;

    // Standalone tag definition followed by semicolon
    if parser.check(TokenKind::Semicolon) {
        let end_span = parser.advance();
        return make_standalone_tag_declaration(specifiers, start, end_span);
    }

    // Collect pre-declarator attributes
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Parse the first declarator
    let (first_declarator, has_grouping) = parse_declarator_inner(parser)?;

    // Collect post-declarator attributes
    let mut all_attrs = attrs;
    if parser.check(TokenKind::Attribute) {
        all_attrs.extend(super::attributes::parse_attribute_list(parser)?);
    }

    // Parse the rest (init-declarator-list and semicolon)
    parse_declaration_rest(parser, specifiers, first_declarator, all_attrs, has_grouping, start)
}

/// Given declaration specifiers with no declarator (followed by `;`),
/// determine if this is a standalone struct/union/enum definition and
/// produce the appropriate `Declaration` variant.
fn make_standalone_tag_declaration(
    specifiers: DeclarationSpecifiers,
    start: Span,
    end_span: Span,
) -> Result<Declaration, ParseError> {
    // Check if the type specifiers contain a struct/union/enum definition
    if specifiers.type_specifiers.len() == 1 {
        match &specifiers.type_specifiers[0] {
            TypeSpecifier::Struct {
                name,
                fields: Some(f),
                attrs,
                span: _,
            } => {
                return Ok(Declaration::StructDef {
                    name: *name,
                    fields: f.clone(),
                    attrs: attrs.clone(),
                    span: Span::merge(start, end_span),
                });
            }
            TypeSpecifier::Union {
                name,
                fields: Some(f),
                attrs,
                span: _,
            } => {
                return Ok(Declaration::UnionDef {
                    name: *name,
                    fields: f.clone(),
                    attrs: attrs.clone(),
                    span: Span::merge(start, end_span),
                });
            }
            TypeSpecifier::Enum {
                name,
                enumerators: Some(e),
                attrs,
                span: _,
            } => {
                return Ok(Declaration::EnumDef {
                    name: *name,
                    enumerators: e.clone(),
                    attrs: attrs.clone(),
                    span: Span::merge(start, end_span),
                });
            }
            _ => {}
        }
    }

    // Forward reference or declaration with no declarators (e.g., `struct foo;`)
    Ok(Declaration::Variable {
        specifiers,
        declarators: vec![],
        attrs: vec![],
        span: Span::merge(start, end_span),
    })
}

// =============================================================================
// §2: Declaration Specifiers
// =============================================================================

/// Parse declaration specifiers — the prefix of a declaration before the
/// declarator list.
///
/// Collects:
/// - Storage class specifiers: `auto`, `register`, `static`, `extern`,
///   `typedef`, `_Thread_local`
/// - Type specifiers: `void`, `char`, `int`, `long`, etc., plus struct/union/
///   enum, typedef names, `typeof`
/// - Type qualifiers: `const`, `volatile`, `restrict`, `_Atomic`
/// - Function specifiers: `inline`, `_Noreturn`
/// - Alignment specifiers: `_Alignas(type-name)` or `_Alignas(constant-expr)`
/// - GCC `__attribute__((...))`
/// - `__extension__` marker
///
/// At least one specifier must be present. If none are found, an error is
/// emitted.
pub fn parse_declaration_specifiers(
    parser: &mut Parser<'_>,
) -> Result<DeclarationSpecifiers, ParseError> {
    let start = parser.current().span;
    let mut storage_class: Option<StorageClass> = None;
    let mut type_specifiers: Vec<TypeSpecifier> = Vec::new();
    let mut type_qualifiers = TypeQualifiers::default();
    let mut func_specifiers = FunctionSpecifiers::default();
    let mut alignment: Option<AlignasSpecifier> = None;
    let mut attrs: Vec<Attribute> = Vec::new();
    let mut has_extension = false;
    let mut last_span = start;
    let mut count = 0usize;

    loop {
        match &parser.current().kind {
            // -----------------------------------------------------------
            // Storage class specifiers
            // -----------------------------------------------------------
            TokenKind::Auto => {
                set_storage_class(parser, &mut storage_class, StorageClass::Auto);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Register => {
                set_storage_class(parser, &mut storage_class, StorageClass::Register);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Static => {
                set_storage_class(parser, &mut storage_class, StorageClass::Static);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Extern => {
                set_storage_class(parser, &mut storage_class, StorageClass::Extern);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Typedef => {
                set_storage_class(parser, &mut storage_class, StorageClass::Typedef);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::ThreadLocal => {
                set_storage_class(parser, &mut storage_class, StorageClass::ThreadLocal);
                last_span = parser.advance();
                count += 1;
            }

            // -----------------------------------------------------------
            // Type specifier keywords
            // -----------------------------------------------------------
            TokenKind::Void => {
                type_specifiers.push(TypeSpecifier::Void);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Char => {
                type_specifiers.push(TypeSpecifier::Char);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Short => {
                type_specifiers.push(TypeSpecifier::Short);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Int => {
                type_specifiers.push(TypeSpecifier::Int);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Long => {
                type_specifiers.push(TypeSpecifier::Long);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Float => {
                type_specifiers.push(TypeSpecifier::Float);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Double => {
                type_specifiers.push(TypeSpecifier::Double);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Signed => {
                type_specifiers.push(TypeSpecifier::Signed);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Unsigned => {
                type_specifiers.push(TypeSpecifier::Unsigned);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Bool => {
                type_specifiers.push(TypeSpecifier::Bool);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Complex => {
                type_specifiers.push(TypeSpecifier::Complex);
                last_span = parser.advance();
                count += 1;
            }

            // -----------------------------------------------------------
            // Type qualifiers
            // -----------------------------------------------------------
            TokenKind::Const => {
                type_qualifiers.is_const = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Volatile => {
                type_qualifiers.is_volatile = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Restrict => {
                type_qualifiers.is_restrict = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Atomic => {
                // _Atomic can be either a type qualifier or a type specifier
                // _Atomic(type-name). If followed by '(', it's a type specifier.
                if parser.peek_ahead(1).is(TokenKind::LeftParen) {
                    parser.advance(); // consume _Atomic
                    parser.advance(); // consume (
                    let inner_type = parse_type_name(parser)?;
                    let end = parser.expect(TokenKind::RightParen)?;
                    type_specifiers.push(TypeSpecifier::Atomic(Box::new(inner_type)));
                    last_span = end;
                    count += 1;
                } else {
                    type_qualifiers.is_atomic = true;
                    last_span = parser.advance();
                    count += 1;
                }
            }

            // -----------------------------------------------------------
            // Function specifiers
            // -----------------------------------------------------------
            TokenKind::Inline => {
                func_specifiers.is_inline = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Noreturn => {
                func_specifiers.is_noreturn = true;
                last_span = parser.advance();
                count += 1;
            }

            // -----------------------------------------------------------
            // Alignment specifier: _Alignas
            // -----------------------------------------------------------
            TokenKind::Alignas => {
                alignment = Some(parse_alignas_specifier(parser)?);
                count += 1;
            }

            // -----------------------------------------------------------
            // Struct / Union / Enum
            // -----------------------------------------------------------
            TokenKind::Struct => {
                let spec = parse_struct_or_union_specifier(parser, true)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                type_specifiers.push(spec);
                count += 1;
            }
            TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser, false)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                type_specifiers.push(spec);
                count += 1;
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                type_specifiers.push(spec);
                count += 1;
            }

            // -----------------------------------------------------------
            // typeof / __typeof__
            // -----------------------------------------------------------
            TokenKind::TypeofKeyword => {
                let spec = parse_typeof_specifier(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                type_specifiers.push(spec);
                count += 1;
            }

            // -----------------------------------------------------------
            // GCC __attribute__((...))
            // -----------------------------------------------------------
            TokenKind::Attribute => {
                let new_attrs = super::attributes::parse_attribute_list(parser)?;
                attrs.extend(new_attrs);
                count += 1;
            }

            // -----------------------------------------------------------
            // __extension__
            // -----------------------------------------------------------
            TokenKind::Extension => {
                has_extension = true;
                last_span = parser.advance();
                count += 1;
            }

            // -----------------------------------------------------------
            // Typedef name (identifier registered as typedef)
            // -----------------------------------------------------------
            TokenKind::Identifier(sym) if parser.is_typedef_name(*sym) => {
                // Only consume typedef name if we haven't seen another type
                // specifier yet (avoids misinterpreting variable names as types
                // in declarations like `int size_t;`).
                if type_specifiers.is_empty() {
                    let name = *sym;
                    let span = parser.current().span;
                    type_specifiers.push(TypeSpecifier::TypedefName { name, span });
                    last_span = parser.advance();
                    count += 1;
                } else {
                    break;
                }
            }

            // -----------------------------------------------------------
            // Not a declaration specifier — stop
            // -----------------------------------------------------------
            _ => break,
        }
    }

    if count == 0 {
        let span = parser.current().span;
        let msg = format!(
            "expected declaration specifiers, found '{}'",
            parser.current().kind
        );
        parser.diagnostics.error(span, &msg);
        return Err(ParseError {
            span,
            message: msg,
            expected: Some("declaration specifier".to_string()),
        });
    }

    Ok(DeclarationSpecifiers {
        storage_class,
        type_specifiers,
        type_qualifiers,
        function_specifiers: func_specifiers,
        alignment,
        attrs,
        has_extension,
        span: Span::merge(start, last_span),
    })
}

/// Helper to set storage class with duplicate detection.
fn set_storage_class(
    parser: &mut Parser<'_>,
    current: &mut Option<StorageClass>,
    new: StorageClass,
) {
    if let Some(existing) = current {
        // _Thread_local can combine with static or extern
        let is_thread_local_combo = matches!(
            (&existing, &new),
            (StorageClass::Static, StorageClass::ThreadLocal)
                | (StorageClass::Extern, StorageClass::ThreadLocal)
                | (StorageClass::ThreadLocal, StorageClass::Static)
                | (StorageClass::ThreadLocal, StorageClass::Extern)
        );
        if !is_thread_local_combo {
            parser.diagnostics.warning(
                parser.current().span,
                "multiple storage class specifiers in declaration",
            );
        }
    }
    *current = Some(new);
}

// =============================================================================
// §3: Declarator Parsing
// =============================================================================

/// Parse a declarator (named) — the core of C declaration syntax.
///
/// A declarator consists of optional pointer derivations followed by a
/// direct declarator (identifier with optional array/function postfix).
///
/// # Grammar
///
/// ```text
/// declarator: pointer? direct-declarator
/// pointer: * type-qualifier-list? | * type-qualifier-list? pointer
/// ```
pub fn parse_declarator(parser: &mut Parser<'_>) -> Result<Declarator, ParseError> {
    let (declarator, _has_grouping) = parse_declarator_inner(parser)?;
    Ok(declarator)
}

/// Internal version of `parse_declarator` that also returns whether the
/// declarator used grouping parentheses (needed for function-declaration
/// vs. function-pointer disambiguation).
fn parse_declarator_inner(
    parser: &mut Parser<'_>,
) -> Result<(Declarator, bool), ParseError> {
    let start = parser.current().span;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Parse pointer derivations: * [qualifiers]*
    while parser.check(TokenKind::Star) {
        parser.advance();
        let quals = parse_type_qualifier_list(parser);
        derived.push(DerivedDeclarator::Pointer { qualifiers: quals });
    }

    // Parse direct declarator (name + postfix derivations)
    let (name, mut postfix_derived, has_grouping, name_span) =
        parse_direct_declarator(parser)?;

    // Combine: pointer derivations come first (outermost), then postfix
    derived.append(&mut postfix_derived);

    // Collect trailing attributes on the declarator
    let decl_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    let end_span = if !derived.is_empty() || name.is_some() {
        name_span.unwrap_or(start)
    } else {
        start
    };

    Ok((
        Declarator {
            name,
            derived,
            attrs: decl_attrs,
            span: Span::merge(start, end_span),
        },
        has_grouping,
    ))
}

/// Parse a direct declarator: identifier, grouped `(declarator)`, or
/// abstract (no name), followed by optional postfix `[...]` and `(...)`.
///
/// Returns (name, postfix_derived, has_grouping, last_span).
fn parse_direct_declarator(
    parser: &mut Parser<'_>,
) -> Result<(Option<Symbol>, Vec<DerivedDeclarator>, bool, Option<Span>), ParseError> {
    let mut derived: Vec<DerivedDeclarator> = Vec::new();
    let mut name: Option<Symbol> = None;
    let mut has_grouping = false;
    let mut last_span: Option<Span> = None;

    // Match the core of the direct declarator
    match &parser.current().kind {
        // Named declarator: identifier
        TokenKind::Identifier(sym) if !parser.is_typedef_name(*sym) => {
            name = Some(*sym);
            last_span = Some(parser.current().span);
            parser.advance();
        }

        // Grouped declarator: ( declarator )
        // Disambiguate from function parameter list by peeking:
        // If after '(' we see '*', or another '(' that looks like grouped,
        // or an identifier that isn't a type name, it's grouped.
        TokenKind::LeftParen => {
            let next_kind = &parser.peek_ahead(1).kind;
            let is_grouped = matches!(
                next_kind,
                TokenKind::Star
                    | TokenKind::LeftParen
                    | TokenKind::Attribute
            ) || matches!(next_kind, TokenKind::Identifier(sym) if !parser.is_typedef_name(*sym));

            if is_grouped {
                has_grouping = true;
                parser.advance(); // consume '('
                let (inner_decl, inner_grouping) = parse_declarator_inner(parser)?;
                parser.expect(TokenKind::RightParen)?;
                name = inner_decl.name;
                // Inner derived declarators come first (closer to name)
                derived.extend(inner_decl.derived);
                if inner_grouping {
                    has_grouping = true;
                }
                last_span = Some(parser.current().span);
            }
            // If not grouped, name stays None (abstract declarator case)
        }

        // No name — abstract declarator context
        _ => {}
    }

    // Parse postfix derivations: [] and ()
    loop {
        match parser.current().kind {
            TokenKind::LeftBracket => {
                let arr = parse_array_declarator(parser)?;
                derived.push(arr);
                last_span = Some(parser.current().span);
            }
            TokenKind::LeftParen => {
                // Function parameter list postfix
                let func = parse_function_declarator(parser)?;
                derived.push(func);
                last_span = Some(parser.current().span);
            }
            _ => break,
        }
    }

    Ok((name, derived, has_grouping, last_span))
}

/// Parse an array declarator: `[ [static] [qualifiers] [size-expr] ]`
///
/// Supports C11 array declarator syntax including:
/// - `[]` — incomplete array / flexible array member
/// - `[10]` — fixed-size array
/// - `[static 10]` — parameter array with guaranteed minimum size
/// - `[const 10]` — qualified array (parameter context)
/// - `[*]` — variable-length array with unspecified size
fn parse_array_declarator(parser: &mut Parser<'_>) -> Result<DerivedDeclarator, ParseError> {
    parser.expect(TokenKind::LeftBracket)?;

    let mut is_static = false;

    // Check for `static` before qualifiers
    if parser.check(TokenKind::Static) {
        is_static = true;
        parser.advance();
    }

    // Parse optional type qualifiers inside brackets
    let qualifiers = parse_type_qualifier_list(parser);

    // Check for `static` after qualifiers (if not already seen)
    if !is_static && parser.check(TokenKind::Static) {
        is_static = true;
        parser.advance();
    }

    // Parse size expression (or `*` for VLA)
    let size = if parser.check(TokenKind::RightBracket) {
        None
    } else if parser.check(TokenKind::Star) && parser.peek_ahead(1).is(TokenKind::RightBracket) {
        // [*] — VLA with unspecified size
        parser.advance();
        None
    } else {
        let size_expr = super::expressions::parse_assignment_expression(parser)?;
        // Validate zero-length array extension
        super::gcc_extensions::validate_zero_length_array(parser, &size_expr);
        Some(Box::new(size_expr))
    };

    parser.expect(TokenKind::RightBracket)?;

    Ok(DerivedDeclarator::Array {
        size,
        is_static,
        qualifiers,
    })
}

// =============================================================================
// §4: Abstract Declarator and Type Name
// =============================================================================

/// Parse an abstract declarator — a declarator without a name.
///
/// Used in type names (casts, sizeof, _Generic, _Alignas, _Atomic),
/// and in parameter declarations where the name is omitted.
///
/// # Grammar
///
/// ```text
/// abstract-declarator: pointer | pointer? direct-abstract-declarator
/// ```
pub fn parse_abstract_declarator(
    parser: &mut Parser<'_>,
) -> Result<AbstractDeclarator, ParseError> {
    let start = parser.current().span;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Parse pointer derivations
    while parser.check(TokenKind::Star) {
        parser.advance();
        let quals = parse_type_qualifier_list(parser);
        derived.push(DerivedDeclarator::Pointer { qualifiers: quals });
    }

    // Parse direct abstract declarator (optional)
    let end_span = parse_direct_abstract_declarator(parser, &mut derived)?;

    let final_span = end_span.unwrap_or(start);
    Ok(AbstractDeclarator {
        derived,
        span: Span::merge(start, final_span),
    })
}

/// Parse the direct part of an abstract declarator.
///
/// ```text
/// direct-abstract-declarator:
///     ( abstract-declarator )
///     direct-abstract-declarator? [ ... ]
///     direct-abstract-declarator? ( parameter-type-list? )
/// ```
///
/// Returns the span of the last parsed component, or `None` if nothing
/// was parsed.
fn parse_direct_abstract_declarator(
    parser: &mut Parser<'_>,
    derived: &mut Vec<DerivedDeclarator>,
) -> Result<Option<Span>, ParseError> {
    let mut last_span: Option<Span> = None;

    // Check for grouped abstract declarator: ( abstract-declarator )
    if parser.check(TokenKind::LeftParen) {
        let next_kind = &parser.peek_ahead(1).kind;
        // If next token is '*', '[', or '(' it's a grouped abstract declarator
        let is_grouped = matches!(
            next_kind,
            TokenKind::Star | TokenKind::LeftBracket | TokenKind::LeftParen
        );

        if is_grouped {
            parser.advance(); // consume '('
            let inner = parse_abstract_declarator(parser)?;
            let end = parser.expect(TokenKind::RightParen)?;
            derived.extend(inner.derived);
            last_span = Some(end);
        }
    }

    // Parse postfix: [] and ()
    loop {
        match parser.current().kind {
            TokenKind::LeftBracket => {
                let arr = parse_array_declarator(parser)?;
                derived.push(arr);
                last_span = Some(parser.current().span);
            }
            TokenKind::LeftParen => {
                // Disambiguate: is this another grouped abstract declarator
                // or a function parameter list?
                // In postfix position, '(' always starts a function parameter list.
                let func = parse_function_declarator(parser)?;
                derived.push(func);
                last_span = Some(parser.current().span);
            }
            _ => break,
        }
    }

    Ok(last_span)
}

/// Parse a type name — specifier-qualifier list plus optional abstract
/// declarator.
///
/// Used in cast expressions, sizeof/alignof, compound literals, _Atomic,
/// _Generic, and _Alignas.
///
/// # Grammar
///
/// ```text
/// type-name: specifier-qualifier-list abstract-declarator?
/// ```
pub fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ParseError> {
    let start = parser.current().span;
    let specifiers = parse_specifier_qualifier_list(parser)?;

    let declarator = if can_start_abstract_declarator(parser) {
        Some(parse_abstract_declarator(parser)?)
    } else {
        None
    };

    let end_span = declarator
        .as_ref()
        .map(|d| d.span)
        .unwrap_or(specifiers.span);

    Ok(TypeName {
        specifiers,
        declarator,
        span: Span::merge(start, end_span),
    })
}

/// Parse a specifier-qualifier list — type specifiers and qualifiers only,
/// without storage class, function specifiers, or alignment.
///
/// Used in struct/union field declarations and type names.
fn parse_specifier_qualifier_list(
    parser: &mut Parser<'_>,
) -> Result<SpecifierQualifierList, ParseError> {
    let start = parser.current().span;
    let mut specifiers: Vec<TypeSpecifier> = Vec::new();
    let mut qualifiers = TypeQualifiers::default();
    let mut last_span = start;
    let mut count = 0usize;

    loop {
        match &parser.current().kind {
            // Type specifier keywords
            TokenKind::Void => {
                specifiers.push(TypeSpecifier::Void);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Char => {
                specifiers.push(TypeSpecifier::Char);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Short => {
                specifiers.push(TypeSpecifier::Short);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Int => {
                specifiers.push(TypeSpecifier::Int);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Long => {
                specifiers.push(TypeSpecifier::Long);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Float => {
                specifiers.push(TypeSpecifier::Float);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Double => {
                specifiers.push(TypeSpecifier::Double);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Signed => {
                specifiers.push(TypeSpecifier::Signed);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Unsigned => {
                specifiers.push(TypeSpecifier::Unsigned);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Bool => {
                specifiers.push(TypeSpecifier::Bool);
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Complex => {
                specifiers.push(TypeSpecifier::Complex);
                last_span = parser.advance();
                count += 1;
            }

            // Type qualifiers
            TokenKind::Const => {
                qualifiers.is_const = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Volatile => {
                qualifiers.is_volatile = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Restrict => {
                qualifiers.is_restrict = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::Atomic => {
                if parser.peek_ahead(1).is(TokenKind::LeftParen) {
                    let _atom = parser.advance(); // consume _Atomic
                    parser.advance(); // consume (
                    let inner_type = parse_type_name(parser)?;
                    let end = parser.expect(TokenKind::RightParen)?;
                    specifiers.push(TypeSpecifier::Atomic(Box::new(inner_type)));
                    last_span = end;
                    count += 1;
                } else {
                    qualifiers.is_atomic = true;
                    last_span = parser.advance();
                    count += 1;
                }
            }

            // Struct / Union / Enum
            TokenKind::Struct => {
                let spec = parse_struct_or_union_specifier(parser, true)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }
            TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser, false)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }

            // typeof / __typeof__
            TokenKind::TypeofKeyword => {
                let spec = parse_typeof_specifier(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }

            // GCC __attribute__
            TokenKind::Attribute => {
                // In specifier-qualifier context, attributes are consumed but
                // typically apply to the declared type rather than the specifiers.
                let _attrs = super::attributes::parse_attribute_list(parser)?;
                count += 1;
            }

            // __extension__
            TokenKind::Extension => {
                last_span = parser.advance();
                count += 1;
            }

            // Typedef name
            TokenKind::Identifier(sym) if parser.is_typedef_name(*sym) => {
                if specifiers.is_empty() {
                    let name = *sym;
                    let span = parser.current().span;
                    specifiers.push(TypeSpecifier::TypedefName { name, span });
                    last_span = parser.advance();
                    count += 1;
                } else {
                    break;
                }
            }

            _ => break,
        }
    }

    if count == 0 {
        let span = parser.current().span;
        let msg = format!(
            "expected type specifier or qualifier, found '{}'",
            parser.current().kind
        );
        parser.diagnostics.error(span, &msg);
        return Err(ParseError {
            span,
            message: msg,
            expected: Some("type specifier".to_string()),
        });
    }

    Ok(SpecifierQualifierList {
        specifiers,
        qualifiers,
        span: Span::merge(start, last_span),
    })
}

// =============================================================================
// §5: Parameter Lists
// =============================================================================

/// Parse a parameter type list (without enclosing parentheses).
///
/// # Grammar
///
/// ```text
/// parameter-type-list: parameter-list | parameter-list , ...
/// parameter-list: parameter-declaration | parameter-list , parameter-declaration
/// ```
pub fn parse_parameter_type_list(
    parser: &mut Parser<'_>,
) -> Result<ParameterList, ParseError> {
    let start = parser.current().span;
    let mut params: Vec<Parameter> = Vec::new();
    let mut variadic = false;

    // Handle void parameter: (void) means no parameters
    if parser.check(TokenKind::Void) && parser.peek_ahead(1).is(TokenKind::RightParen) {
        let void_span = parser.advance();
        return Ok(ParameterList {
            params: vec![],
            variadic: false,
            span: Span::merge(start, void_span),
        });
    }

    // Parse first parameter
    if !parser.check(TokenKind::RightParen) && !parser.check(TokenKind::Ellipsis) {
        let param = parse_parameter_declaration(parser)?;
        params.push(param);
    }

    // Parse remaining parameters
    while parser.eat(TokenKind::Comma) {
        if parser.check(TokenKind::Ellipsis) {
            variadic = true;
            parser.advance();
            break;
        }
        let param = parse_parameter_declaration(parser)?;
        params.push(param);
    }

    // Handle standalone ellipsis (old-style variadic)
    if !variadic && parser.check(TokenKind::Ellipsis) {
        variadic = true;
        parser.advance();
    }

    let last_span = if let Some(last) = params.last() {
        last.span
    } else {
        start
    };

    Ok(ParameterList {
        params,
        variadic,
        span: Span::merge(start, last_span),
    })
}

/// Parse a function declarator: `( parameter-type-list )`
///
/// Handles:
/// - `(void)` — no parameters
/// - `(int x, int y)` — named parameters
/// - `(int, int)` — unnamed parameters
/// - `(int x, ...)` — variadic
/// - `()` — unspecified parameters (K&R style)
pub(super) fn parse_function_declarator(
    parser: &mut Parser<'_>,
) -> Result<DerivedDeclarator, ParseError> {
    let start = parser.current().span;
    parser.expect(TokenKind::LeftParen)?;

    // Empty parameter list: ()
    if parser.check(TokenKind::RightParen) {
        let end = parser.advance();
        return Ok(DerivedDeclarator::Function {
            params: ParameterList {
                params: vec![],
                variadic: false,
                span: Span::merge(start, end),
            },
        });
    }

    let mut param_list = parse_parameter_type_list(parser)?;
    let end = parser.expect(TokenKind::RightParen)?;
    param_list.span = Span::merge(start, end);

    // Collect trailing attributes after the function declarator
    if parser.check(TokenKind::Attribute) {
        let _attrs = super::attributes::parse_attribute_list(parser)?;
        // Attributes after function declarator are typically applied to the
        // function type itself; they are consumed here but the caller may
        // need to handle them.
    }

    Ok(DerivedDeclarator::Function {
        params: param_list,
    })
}

/// Parse a single parameter declaration.
///
/// # Grammar
///
/// ```text
/// parameter-declaration:
///     declaration-specifiers declarator
///     declaration-specifiers abstract-declarator?
/// ```
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> Result<Parameter, ParseError> {
    let start = parser.current().span;
    let specifiers = parse_declaration_specifiers(parser)?;

    // Try to parse a declarator. The tricky part is distinguishing between
    // a named declarator and an abstract declarator (no name).
    let declarator = if parser.check(TokenKind::Comma)
        || parser.check(TokenKind::RightParen)
        || parser.check(TokenKind::Ellipsis)
    {
        // No declarator follows — abstract declarator omitted
        None
    } else if can_start_abstract_declarator(parser)
        && !matches!(&parser.current().kind, TokenKind::Identifier(sym) if !parser.is_typedef_name(*sym))
    {
        // Looks like an abstract declarator (starts with *, (, or [)
        // but not a plain identifier (which would be a named declarator)
        let abs = parse_abstract_declarator(parser)?;
        Some(Declarator {
            name: None,
            derived: abs.derived,
            attrs: vec![],
            span: abs.span,
        })
    } else {
        // Try named declarator
        parse_declarator(parser).ok()
    };

    let end_span = declarator
        .as_ref()
        .map(|d| d.span)
        .unwrap_or(specifiers.span);

    Ok(Parameter {
        specifiers,
        declarator,
        span: Span::merge(start, end_span),
    })
}

// =============================================================================
// §6: Initializers
// =============================================================================

/// Parse an initializer expression or brace-enclosed initializer list.
///
/// # Grammar
///
/// ```text
/// initializer:
///     assignment-expression
///     { initializer-list }
///     { initializer-list , }
/// ```
pub fn parse_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ParseError> {
    if parser.check(TokenKind::LeftBrace) {
        parse_brace_initializer(parser)
    } else {
        let expr = super::expressions::parse_assignment_expression(parser)?;
        Ok(Initializer::Expression(Box::new(expr)))
    }
}

/// Parse a brace-enclosed initializer list: `{ initializer-list [,] }`
///
/// Supports designated initializers:
/// - `.field = value` (struct field designator)
/// - `[index] = value` (array index designator)
/// - Nested designators: `.field.subfield = value`
fn parse_brace_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ParseError> {
    let start = parser.current().span;
    parser.expect(TokenKind::LeftBrace)?;

    let mut items: Vec<InitializerItem> = Vec::new();

    // Handle empty initializer: {}
    if parser.check(TokenKind::RightBrace) {
        let end = parser.advance();
        return Ok(Initializer::List {
            items,
            span: Span::merge(start, end),
        });
    }

    loop {
        if parser.check(TokenKind::RightBrace) || parser.at_end() {
            break;
        }

        let item_start = parser.current().span;
        let mut designators: Vec<Designator> = Vec::new();

        // Parse designator chain: .field, [index], or nested
        while parser.check(TokenKind::Dot) || parser.check(TokenKind::LeftBracket) {
            if parser.check(TokenKind::Dot) {
                parser.advance(); // consume '.'
                let field_name = expect_identifier(parser, "designator")?;
                designators.push(Designator::Field(field_name));
            } else {
                // [index]
                parser.advance(); // consume '['
                let index_expr = super::expressions::parse_constant_expression(parser)?;
                parser.expect(TokenKind::RightBracket)?;
                designators.push(Designator::Index(Box::new(index_expr)));
            }
        }

        // Consume '=' after designators (required if designators present)
        if !designators.is_empty() {
            parser.expect(TokenKind::Assign)?;
        }

        // Parse the initializer value
        let initializer = parse_initializer(parser)?;
        let item_end = match &initializer {
            Initializer::Expression(_) => item_start, // approximate span
            Initializer::List { span, .. } => *span,
        };

        items.push(InitializerItem {
            designators,
            initializer,
            span: Span::merge(item_start, item_end),
        });

        // Trailing comma is optional; eat it if present
        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    let end = parser.expect(TokenKind::RightBrace)?;
    Ok(Initializer::List {
        items,
        span: Span::merge(start, end),
    })
}

// =============================================================================
// §7: Struct / Union Parsing
// =============================================================================

/// Parse a struct or union as a standalone declaration.
///
/// Expects the current token to be `struct` or `union`. Returns:
/// - `Declaration::StructDef` / `Declaration::UnionDef` for definitions
/// - `Declaration::Variable` with tag specifier for forward references
///
/// This is a public convenience entry point. For use within declaration
/// specifiers, `parse_struct_or_union_specifier` is called internally.
pub fn parse_struct_or_union(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.current().span;
    let is_struct = match parser.current().kind {
        TokenKind::Struct => true,
        TokenKind::Union => false,
        _ => {
            let span = parser.current().span;
            let msg = format!("expected 'struct' or 'union', found '{}'", parser.current().kind);
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("struct or union".to_string()),
            });
        }
    };

    let spec = parse_struct_or_union_specifier(parser, is_struct)?;

    // Collect trailing attributes
    let trailing_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Determine which Declaration variant to produce
    match spec {
        TypeSpecifier::Struct {
            name,
            fields: Some(f),
            attrs: tag_attrs,
            span,
        } => {
            let mut all_attrs = tag_attrs;
            all_attrs.extend(trailing_attrs);
            Ok(Declaration::StructDef {
                name,
                fields: f,
                attrs: all_attrs,
                span: Span::merge(start, span),
            })
        }
        TypeSpecifier::Union {
            name,
            fields: Some(f),
            attrs: tag_attrs,
            span,
        } => {
            let mut all_attrs = tag_attrs;
            all_attrs.extend(trailing_attrs);
            Ok(Declaration::UnionDef {
                name,
                fields: f,
                attrs: all_attrs,
                span: Span::merge(start, span),
            })
        }
        _ => {
            // Forward reference — wrap as variable declaration
            let end_span = type_specifier_span(&spec).unwrap_or(start);
            let specifiers = DeclarationSpecifiers {
                storage_class: None,
                type_specifiers: vec![spec],
                type_qualifiers: TypeQualifiers::default(),
                function_specifiers: FunctionSpecifiers::default(),
                alignment: None,
                attrs: vec![],
                has_extension: false,
                span: Span::merge(start, end_span),
            };
            Ok(Declaration::Variable {
                specifiers,
                declarators: vec![],
                attrs: trailing_attrs,
                span: Span::merge(start, end_span),
            })
        }
    }
}

/// Parse a struct or union type specifier.
///
/// # Grammar
///
/// ```text
/// struct-or-union-specifier:
///     struct-or-union attributes? identifier? { struct-declaration-list }
///     struct-or-union attributes? identifier
/// ```
///
/// `is_struct` — `true` for `struct`, `false` for `union`.
pub(super) fn parse_struct_or_union_specifier(
    parser: &mut Parser<'_>,
    is_struct: bool,
) -> Result<TypeSpecifier, ParseError> {
    let start = parser.current().span;
    // Consume 'struct' or 'union' keyword
    parser.advance();

    // Optional attributes before the tag name
    let mut tag_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Optional tag name
    let name = if parser.check(TokenKind::Identifier(Symbol::EMPTY)) {
        Some(expect_identifier(parser, "struct/union tag")?)
    } else {
        None
    };

    // Optional attributes after the tag name
    if parser.check(TokenKind::Attribute) {
        tag_attrs.extend(super::attributes::parse_attribute_list(parser)?);
    }

    // Field list: { struct-declaration-list }
    let fields = if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume '{'
        let mut field_list: Vec<FieldDeclaration> = Vec::new();

        while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
            match parse_struct_field(parser) {
                Ok(field) => field_list.push(field),
                Err(_) => {
                    // Error recovery: skip to next field or closing brace
                    parser.synchronize();
                    if parser.check(TokenKind::Semicolon) {
                        parser.advance();
                    }
                }
            }
        }
        let end = parser.expect(TokenKind::RightBrace)?;

        // Validate flexible array member if present
        if is_struct && !field_list.is_empty() {
            let temp_decl = Declaration::StructDef {
                name,
                fields: field_list.clone(),
                attrs: tag_attrs.clone(),
                span: Span::merge(start, end),
            };
            super::gcc_extensions::validate_flexible_array_member(parser, &temp_decl);
        }

        Some(field_list)
    } else {
        // Forward reference: must have a name
        if name.is_none() {
            let span = parser.current().span;
            let tag = if is_struct { "struct" } else { "union" };
            parser
                .diagnostics
                .error(span, format!("{tag} without name or body"));
        }
        None
    };

    let end_span = parser.current().span;
    if is_struct {
        Ok(TypeSpecifier::Struct {
            name,
            fields,
            attrs: tag_attrs,
            span: Span::merge(start, end_span),
        })
    } else {
        Ok(TypeSpecifier::Union {
            name,
            fields,
            attrs: tag_attrs,
            span: Span::merge(start, end_span),
        })
    }
}

/// Parse a single struct or union field declaration.
///
/// # Grammar
///
/// ```text
/// struct-declaration:
///     specifier-qualifier-list struct-declarator-list? ;
///     _Static_assert-declaration
/// ```
///
/// Supports:
/// - Regular fields: `int x;`
/// - Bitfields: `unsigned int flags : 3;`
/// - Anonymous bitfields: `int : 5;`
/// - Anonymous struct/union members (C11): `struct { int x; };`
/// - GCC attributes on fields
fn parse_struct_field(parser: &mut Parser<'_>) -> Result<FieldDeclaration, ParseError> {
    let start = parser.current().span;

    // Handle _Static_assert inside struct (C11 §6.7.2.1)
    if parser.check(TokenKind::StaticAssert) {
        // _Static_assert inside a struct — parse and wrap as a field with
        // empty declarators. The actual assertion is handled at semantic level.
        let _sa = parse_static_assert(parser)?;
        return Ok(FieldDeclaration {
            specifiers: DeclarationSpecifiers {
                storage_class: None,
                type_specifiers: vec![],
                type_qualifiers: TypeQualifiers::default(),
                function_specifiers: FunctionSpecifiers::default(),
                alignment: None,
                attrs: vec![],
                has_extension: false,
                span: start,
            },
            declarators: vec![],
            attrs: vec![],
            span: Span::merge(start, parser.current().span),
        });
    }

    // Parse specifier-qualifier-list
    let spec_quals = parse_specifier_qualifier_list(parser)?;
    let specifiers = DeclarationSpecifiers {
        storage_class: None,
        type_specifiers: spec_quals.specifiers,
        type_qualifiers: spec_quals.qualifiers,
        function_specifiers: FunctionSpecifiers::default(),
        alignment: None,
        attrs: vec![],
        has_extension: false,
        span: spec_quals.span,
    };

    // Check for anonymous struct/union member (C11): just specifiers followed by ';'
    if parser.check(TokenKind::Semicolon) {
        let end = parser.advance();
        return Ok(FieldDeclaration {
            specifiers,
            declarators: vec![],
            attrs: vec![],
            span: Span::merge(start, end),
        });
    }

    // Parse field declarator list
    let mut declarators: Vec<FieldDeclarator> = Vec::new();

    loop {
        let field_start = parser.current().span;

        // Optional declarator (absent for anonymous bitfields like `int : 5;`)
        let declarator = if parser.check(TokenKind::Colon) {
            None
        } else {
            match parse_declarator(parser) {
                Ok(d) => Some(d),
                Err(e) => {
                    // If we can't parse a declarator, try bitfield
                    if parser.check(TokenKind::Colon) {
                        None
                    } else {
                        return Err(e);
                    }
                }
            }
        };

        // Optional bitfield width: `: constant-expression`
        let bit_width = if parser.eat(TokenKind::Colon) {
            Some(Box::new(
                super::expressions::parse_constant_expression(parser)?,
            ))
        } else {
            None
        };

        // Optional attributes on the field declarator
        let _field_attrs = if parser.check(TokenKind::Attribute) {
            super::attributes::parse_attribute_list(parser)?
        } else {
            vec![]
        };

        let field_end = parser.current().span;
        declarators.push(FieldDeclarator {
            declarator,
            bit_width,
            span: Span::merge(field_start, field_end),
        });

        if !parser.eat(TokenKind::Comma) {
            break;
        }
    }

    // Collect trailing attributes on the field declaration
    let field_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    let end = parser.expect(TokenKind::Semicolon)?;
    Ok(FieldDeclaration {
        specifiers,
        declarators,
        attrs: field_attrs,
        span: Span::merge(start, end),
    })
}

// =============================================================================
// §8: Enum Parsing
// =============================================================================

/// Parse an enum as a standalone declaration.
///
/// Expects the current token to be `enum`. Returns:
/// - `Declaration::EnumDef` for definitions with enumerator list
/// - `Declaration::Variable` for forward references
pub fn parse_enum(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.current().span;

    if !parser.check(TokenKind::Enum) {
        let span = parser.current().span;
        let msg = format!("expected 'enum', found '{}'", parser.current().kind);
        parser.diagnostics.error(span, &msg);
        return Err(ParseError {
            span,
            message: msg,
            expected: Some("enum".to_string()),
        });
    }

    let spec = parse_enum_specifier(parser)?;

    // Collect trailing attributes
    let trailing_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    match spec {
        TypeSpecifier::Enum {
            name,
            enumerators: Some(e),
            attrs: tag_attrs,
            span,
        } => {
            let mut all_attrs = tag_attrs;
            all_attrs.extend(trailing_attrs);
            Ok(Declaration::EnumDef {
                name,
                enumerators: e,
                attrs: all_attrs,
                span: Span::merge(start, span),
            })
        }
        _ => {
            let end_span = type_specifier_span(&spec).unwrap_or(start);
            let specifiers = DeclarationSpecifiers {
                storage_class: None,
                type_specifiers: vec![spec],
                type_qualifiers: TypeQualifiers::default(),
                function_specifiers: FunctionSpecifiers::default(),
                alignment: None,
                attrs: vec![],
                has_extension: false,
                span: Span::merge(start, end_span),
            };
            Ok(Declaration::Variable {
                specifiers,
                declarators: vec![],
                attrs: trailing_attrs,
                span: Span::merge(start, end_span),
            })
        }
    }
}

/// Parse an enum type specifier.
///
/// # Grammar
///
/// ```text
/// enum-specifier:
///     enum attributes? identifier? { enumerator-list [,] }
///     enum attributes? identifier
/// ```
pub(super) fn parse_enum_specifier(
    parser: &mut Parser<'_>,
) -> Result<TypeSpecifier, ParseError> {
    let start = parser.current().span;
    parser.expect(TokenKind::Enum)?;

    // Optional attributes before tag name
    let mut tag_attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Optional tag name
    let name = if parser.check(TokenKind::Identifier(Symbol::EMPTY)) {
        Some(expect_identifier(parser, "enum tag")?)
    } else {
        None
    };

    // Optional attributes after tag name
    if parser.check(TokenKind::Attribute) {
        tag_attrs.extend(super::attributes::parse_attribute_list(parser)?);
    }

    // Enumerator list: { enumerator-list [,] }
    let enumerators = if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume '{'
        let mut enum_list: Vec<Enumerator> = Vec::new();

        while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
            match parse_enumerator(parser) {
                Ok(e) => enum_list.push(e),
                Err(_) => {
                    parser.synchronize();
                    if parser.check(TokenKind::Comma) {
                        parser.advance();
                    }
                }
            }

            // Trailing comma is optional
            if !parser.eat(TokenKind::Comma) {
                break;
            }
        }

        parser.expect(TokenKind::RightBrace)?;
        Some(enum_list)
    } else {
        if name.is_none() {
            let span = parser.current().span;
            parser
                .diagnostics
                .error(span, "enum without name or body");
        }
        None
    };

    let end_span = parser.current().span;
    Ok(TypeSpecifier::Enum {
        name,
        enumerators,
        attrs: tag_attrs,
        span: Span::merge(start, end_span),
    })
}

/// Parse a single enumerator: `identifier [= constant-expression]`
fn parse_enumerator(parser: &mut Parser<'_>) -> Result<Enumerator, ParseError> {
    let start = parser.current().span;
    let name = expect_identifier(parser, "enumerator")?;

    // Optional attributes on enumerator
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        vec![]
    };

    // Optional explicit value: = constant-expression
    let value = if parser.eat(TokenKind::Assign) {
        Some(Box::new(
            super::expressions::parse_constant_expression(parser)?,
        ))
    } else {
        None
    };

    let end_span = parser.current().span;
    Ok(Enumerator {
        name,
        value,
        attrs,
        span: Span::merge(start, end_span),
    })
}

// =============================================================================
// §9: _Static_assert
// =============================================================================

/// Parse a `_Static_assert` declaration.
///
/// # Grammar
///
/// ```text
/// _Static_assert ( constant-expression , string-literal ) ;
/// ```
///
/// The condition must be an integer constant expression. If it evaluates
/// to zero at compile time, the compiler emits an error including the
/// message string.
pub fn parse_static_assert(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.current().span;
    parser.expect(TokenKind::StaticAssert)?;
    parser.expect(TokenKind::LeftParen)?;

    // Parse the constant expression condition
    let condition = super::expressions::parse_constant_expression(parser)?;

    parser.expect(TokenKind::Comma)?;

    // Parse the string literal message — must produce Vec<u8> for PUA fidelity
    let message: Vec<u8> = match &parser.current().kind {
        TokenKind::StringLiteral { value, .. } => {
            let val = value.clone();
            parser.advance();
            val
        }
        _ => {
            let span = parser.current().span;
            let msg = format!(
                "expected string literal in _Static_assert, found '{}'",
                parser.current().kind
            );
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("string literal".to_string()),
            });
        }
    };

    parser.expect(TokenKind::RightParen)?;
    let end = parser.expect(TokenKind::Semicolon)?;

    Ok(Declaration::StaticAssert {
        condition: Box::new(condition),
        message,
        span: Span::merge(start, end),
    })
}

// =============================================================================
// §10: _Alignas, typeof
// =============================================================================

/// Parse an `_Alignas` alignment specifier.
///
/// # Grammar
///
/// ```text
/// alignment-specifier:
///     _Alignas ( type-name )
///     _Alignas ( constant-expression )
/// ```
fn parse_alignas_specifier(parser: &mut Parser<'_>) -> Result<AlignasSpecifier, ParseError> {
    parser.expect(TokenKind::Alignas)?;
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type-name vs constant-expression
    // If the first token looks like a type specifier, parse as type name;
    // otherwise parse as expression.
    let spec = if is_specifier_qualifier_start(parser) {
        let type_name = parse_type_name(parser)?;
        AlignasSpecifier::TypeName(Box::new(type_name))
    } else {
        let expr = super::expressions::parse_constant_expression(parser)?;
        AlignasSpecifier::Expression(Box::new(expr))
    };

    parser.expect(TokenKind::RightParen)?;
    Ok(spec)
}

/// Parse a `typeof` / `__typeof__` type specifier.
///
/// # Grammar
///
/// ```text
/// typeof-specifier: typeof ( expression ) | typeof ( type-name )
/// ```
fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ParseError> {
    let start = parser.current().span;
    parser.expect(TokenKind::TypeofKeyword)?;
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type-name vs expression
    let operand = if is_specifier_qualifier_start(parser) {
        let type_name = parse_type_name(parser)?;
        TypeofOperand::TypeName(Box::new(type_name))
    } else {
        let expr = super::expressions::parse_expression(parser)?;
        TypeofOperand::Expression(Box::new(expr))
    };

    let end = parser.expect(TokenKind::RightParen)?;
    Ok(TypeSpecifier::Typeof {
        operand,
        span: Span::merge(start, end),
    })
}

// =============================================================================
// §11: Declaration Rest — the init-declarator-list tail
// =============================================================================

/// Parse the remainder of a declaration after the first declarator has been
/// consumed (by `parse_external_declaration` or `parse_declaration`).
///
/// Handles:
/// - Initializer for first declarator: `= initializer`
/// - Additional declarators: `, declarator [= initializer]`
/// - Trailing semicolon
/// - Typedef registration
/// - Function-declaration vs variable-declaration disambiguation
fn parse_declaration_rest(
    parser: &mut Parser<'_>,
    specifiers: DeclarationSpecifiers,
    first_declarator: Declarator,
    attrs: Vec<Attribute>,
    has_grouping: bool,
    start: Span,
) -> Result<Declaration, ParseError> {
    let is_typedef = matches!(specifiers.storage_class, Some(StorageClass::Typedef));

    // For typedef declarations, collect declarators (no initializers)
    if is_typedef {
        let mut typedef_declarators: Vec<Declarator> = Vec::new();

        // Register the first typedef name
        if let Some(name) = first_declarator.name {
            parser.register_typedef(name);
            // Use as_u32() to verify the symbol is valid for tracking purposes
            let _name_id = name.as_u32();
        }
        typedef_declarators.push(first_declarator);

        // Parse additional typedef declarators
        while parser.eat(TokenKind::Comma) {
            let decl = parse_declarator(parser)?;
            if let Some(name) = decl.name {
                parser.register_typedef(name);
            }
            typedef_declarators.push(decl);
        }

        let end = parser.expect(TokenKind::Semicolon)?;
        return Ok(Declaration::Typedef {
            specifiers,
            declarators: typedef_declarators,
            attrs,
            span: Span::merge(start, end),
        });
    }

    // Check if this is a function declaration (prototype)
    if is_function_declaration(&first_declarator.derived, has_grouping) {
        // A function prototype — no initializer expected, just semicolon
        // But first check for additional declarators (unusual but valid):
        // e.g., `int f(void), g(int);`
        if parser.check(TokenKind::Semicolon) {
            let end = parser.advance();
            return Ok(Declaration::FunctionDecl {
                specifiers,
                declarator: first_declarator,
                attrs,
                span: Span::merge(start, end),
            });
        }

        // If comma follows, fall through to variable declaration handling
        // (though `int f(void), x;` is technically valid C — f is a function
        // declaration and x is a variable)
    }

    // Variable declaration with init-declarator-list
    let mut init_declarators: Vec<InitDeclarator> = Vec::new();

    // First declarator with optional initializer
    let first_init = if parser.eat(TokenKind::Assign) {
        Some(parse_initializer(parser)?)
    } else {
        None
    };
    let first_end = parser.current().span;
    init_declarators.push(InitDeclarator {
        declarator: first_declarator,
        initializer: first_init,
        span: Span::merge(start, first_end),
    });

    // Additional declarators: , declarator [= initializer]
    while parser.eat(TokenKind::Comma) {
        let decl_start = parser.current().span;

        // Optional attributes before declarator
        let extra_attrs = if parser.check(TokenKind::Attribute) {
            super::attributes::parse_attribute_list(parser)?
        } else {
            vec![]
        };

        let mut next_decl = parse_declarator(parser)?;
        // Merge extra attributes onto the declarator
        next_decl.attrs.extend(extra_attrs);

        // Optional attributes after declarator
        if parser.check(TokenKind::Attribute) {
            next_decl
                .attrs
                .extend(super::attributes::parse_attribute_list(parser)?);
        }

        let next_init = if parser.eat(TokenKind::Assign) {
            Some(parse_initializer(parser)?)
        } else {
            None
        };

        let decl_end = parser.current().span;
        init_declarators.push(InitDeclarator {
            declarator: next_decl,
            initializer: next_init,
            span: Span::merge(decl_start, decl_end),
        });
    }

    let end = match parser.expect(TokenKind::Semicolon) {
        Ok(span) => span,
        Err(_) => {
            // Error recovery: emit the error, try to synchronize, and
            // return what we have so far
            let err_span = Span::merge(start, parser.current().span);
            parser.synchronize();
            return Ok(Declaration::Variable {
                specifiers,
                declarators: init_declarators,
                attrs,
                span: err_span,
            });
        }
    };

    Ok(Declaration::Variable {
        specifiers,
        declarators: init_declarators,
        attrs,
        span: Span::merge(start, end),
    })
}

// =============================================================================
// §12: Type Qualifier List Helper
// =============================================================================

/// Parse a sequence of type qualifiers (used in pointer declarators and
/// array brackets).
///
/// Collects `const`, `volatile`, `restrict`, `_Atomic` qualifiers.
fn parse_type_qualifier_list(parser: &mut Parser<'_>) -> TypeQualifiers {
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
