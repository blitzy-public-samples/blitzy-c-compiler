//! Declaration parsing module for the BCC C11 parser.
//!
//! Implements parsing for all forms of C11 declarations:
//! - Variable/object declarations with optional initialisers
//! - Function definitions and forward declarations (prototypes)
//! - Typedefs
//! - Struct / union / enum definitions
//! - `_Static_assert`
//! - `_Alignas` specifiers
//! - GCC `__extension__` prefix on declarations
//! - GCC `__attribute__` annotations on declarations and declarators
//!
//! # Architecture
//!
//! All public functions accept a `&mut Parser` and return
//! `Result<Declaration, ParseError>`.
//!
//! The central entry point for file-scope declarations is
//! [`parse_external_declaration`]. For block-scope declarations (inside
//! compound statements) the same [`parse_declaration`] function is used.

use super::ast::{
    AlignasSpecifier, Attribute, Declaration, DeclarationSpecifiers, Declarator, DerivedDeclarator,
    Enumerator, FieldDeclaration, FieldDeclarator, FunctionSpecifiers, InitDeclarator, Initializer,
    InitializerItem, Parameter, ParameterList, Span, StorageClass, TypeQualifiers, TypeSpecifier,
};
use super::{ParseError, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Helpers
// ===========================================================================

/// Extracts the source span from a `TypeSpecifier` variant, if available.
fn type_specifier_span(spec: &TypeSpecifier) -> Option<Span> {
    match spec {
        TypeSpecifier::Struct { span, .. }
        | TypeSpecifier::Union { span, .. }
        | TypeSpecifier::Enum { span, .. }
        | TypeSpecifier::TypedefName { span, .. }
        | TypeSpecifier::Typeof { span, .. } => Some(*span),
        // Keyword specifiers (Void, Int, etc.) don't carry their own span.
        _ => None,
    }
}

// ===========================================================================
// External declaration (file-scope entry point)
// ===========================================================================

/// Parses an external (file-scope) declaration.
///
/// ```text
/// external-declaration:
///     function-definition
///     declaration
///     ';'              (empty declaration)
/// ```
///
/// A function-definition is distinguished from a declaration only after
/// parsing declaration specifiers and the declarator — if a `{` follows
/// rather than `=`, `,`, or `;`, it is a function definition.
pub fn parse_external_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    // Empty declaration
    if parser.check(TokenKind::Semicolon) {
        let span = parser.advance();
        return Ok(Declaration::Empty { span });
    }

    // _Static_assert
    if parser.check(TokenKind::StaticAssert) {
        return parse_static_assert(parser);
    }

    // Parse declaration specifiers
    let specifiers = parse_declaration_specifiers(parser)?;

    // Check for struct/union/enum standing alone (tag definition without declarator)
    if parser.check(TokenKind::Semicolon) {
        let end = parser.advance();
        let span = Span::merge(specifiers.span, end);
        return Ok(Declaration::Variable {
            specifiers,
            declarators: Vec::new(),
            attrs: Vec::new(),
            span,
        });
    }

    // Parse first declarator
    let declarator = parse_declarator(parser)?;

    // Trailing attributes
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    // Function definition: specifiers declarator { body }
    if parser.check(TokenKind::LeftBrace) {
        let body = super::statements::parse_compound_statement(parser)?;
        let span = Span::merge(specifiers.span, body.span());

        // If typedef storage class was present, register the name
        if specifiers.storage_class == Some(StorageClass::Typedef) {
            if let Some(name) = declarator.name {
                parser.register_typedef(name);
            }
        }

        return Ok(Declaration::FunctionDef {
            specifiers,
            declarator,
            attrs,
            body: Box::new(body),
            span,
        });
    }

    // Variable declaration or function prototype
    parse_declaration_rest(parser, specifiers, declarator, attrs)
}

/// Parses a block-scope declaration.
///
/// ```text
/// declaration:
///     declaration-specifiers init-declarator-list? ';'
///     _Static_assert ( ... ) ;
/// ```
pub fn parse_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    // _Static_assert
    if parser.check(TokenKind::StaticAssert) {
        return parse_static_assert(parser);
    }

    let specifiers = parse_declaration_specifiers(parser)?;

    // Lone specifiers (e.g., `struct foo;`)
    if parser.check(TokenKind::Semicolon) {
        let end = parser.advance();
        let span = Span::merge(specifiers.span, end);
        return Ok(Declaration::Variable {
            specifiers,
            declarators: Vec::new(),
            attrs: Vec::new(),
            span,
        });
    }

    let declarator = parse_declarator(parser)?;

    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    parse_declaration_rest(parser, specifiers, declarator, attrs)
}

// ===========================================================================
// Declaration specifiers
// ===========================================================================

/// Parses declaration specifiers: storage class, type specifiers, type
/// qualifiers, function specifiers, _Alignas, __attribute__, __extension__.
///
/// Specifiers are accumulated in a loop until we see a token that cannot
/// be part of the specifier list.
pub fn parse_declaration_specifiers(
    parser: &mut Parser<'_>,
) -> Result<DeclarationSpecifiers, ParseError> {
    let start = parser.current().span;
    let mut storage_class: Option<StorageClass> = None;
    let mut type_specifiers: Vec<TypeSpecifier> = Vec::new();
    let mut qualifiers = TypeQualifiers::default();
    let mut func_specs = FunctionSpecifiers::default();
    let mut alignment: Option<AlignasSpecifier> = None;
    let mut attrs: Vec<Attribute> = Vec::new();
    let mut has_extension = false;
    let mut last_span = start;

    loop {
        match parser.current().kind.clone() {
            // Storage class specifiers
            TokenKind::Auto => {
                storage_class = Some(StorageClass::Auto);
                last_span = parser.advance();
            }
            TokenKind::Register => {
                storage_class = Some(StorageClass::Register);
                last_span = parser.advance();
            }
            TokenKind::Static => {
                storage_class = Some(StorageClass::Static);
                last_span = parser.advance();
            }
            TokenKind::Extern => {
                storage_class = Some(StorageClass::Extern);
                last_span = parser.advance();
            }
            TokenKind::Typedef => {
                storage_class = Some(StorageClass::Typedef);
                last_span = parser.advance();
            }
            TokenKind::ThreadLocal => {
                // _Thread_local can combine with static/extern
                last_span = parser.advance();
            }

            // Type specifiers (keywords)
            TokenKind::Void => {
                type_specifiers.push(TypeSpecifier::Void);
                last_span = parser.advance();
            }
            TokenKind::Char => {
                type_specifiers.push(TypeSpecifier::Char);
                last_span = parser.advance();
            }
            TokenKind::Short => {
                type_specifiers.push(TypeSpecifier::Short);
                last_span = parser.advance();
            }
            TokenKind::Int => {
                type_specifiers.push(TypeSpecifier::Int);
                last_span = parser.advance();
            }
            TokenKind::Long => {
                type_specifiers.push(TypeSpecifier::Long);
                last_span = parser.advance();
            }
            TokenKind::Float => {
                type_specifiers.push(TypeSpecifier::Float);
                last_span = parser.advance();
            }
            TokenKind::Double => {
                type_specifiers.push(TypeSpecifier::Double);
                last_span = parser.advance();
            }
            TokenKind::Signed => {
                type_specifiers.push(TypeSpecifier::Signed);
                last_span = parser.advance();
            }
            TokenKind::Unsigned => {
                type_specifiers.push(TypeSpecifier::Unsigned);
                last_span = parser.advance();
            }
            TokenKind::Bool => {
                type_specifiers.push(TypeSpecifier::Bool);
                last_span = parser.advance();
            }
            TokenKind::Complex => {
                type_specifiers.push(TypeSpecifier::Complex);
                last_span = parser.advance();
            }

            // struct / union / enum
            TokenKind::Struct => {
                let spec = parse_struct_or_union_specifier(parser, true)?;
                last_span = type_specifier_span(&spec).unwrap_or(last_span);
                type_specifiers.push(spec);
            }
            TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser, false)?;
                last_span = type_specifier_span(&spec).unwrap_or(last_span);
                type_specifiers.push(spec);
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                last_span = type_specifier_span(&spec).unwrap_or(last_span);
                type_specifiers.push(spec);
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

            // Function specifiers
            TokenKind::Inline => {
                func_specs.is_inline = true;
                last_span = parser.advance();
            }
            TokenKind::Noreturn => {
                func_specs.is_noreturn = true;
                last_span = parser.advance();
            }

            // _Alignas
            TokenKind::Alignas => {
                let align = parse_alignas_specifier(parser)?;
                alignment = Some(align);
            }

            // typeof / __typeof__
            TokenKind::TypeofKeyword => {
                let spec = parse_typeof_specifier(parser)?;
                last_span = type_specifier_span(&spec).unwrap_or(last_span);
                type_specifiers.push(spec);
            }

            // __attribute__
            TokenKind::Attribute => {
                let mut new_attrs = super::attributes::parse_attribute_list(parser)?;
                attrs.append(&mut new_attrs);
            }

            // __extension__
            TokenKind::Extension => {
                has_extension = true;
                last_span = parser.advance();
            }

            // typedef name (an identifier previously registered as typedef)
            TokenKind::Identifier(sym) => {
                if type_specifiers.is_empty() && parser.is_typedef_name(sym) {
                    let tspan = parser.current().span;
                    type_specifiers.push(TypeSpecifier::TypedefName {
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
    Ok(DeclarationSpecifiers {
        storage_class,
        type_specifiers,
        type_qualifiers: qualifiers,
        function_specifiers: func_specs,
        alignment,
        attrs,
        has_extension,
        span,
    })
}

// ===========================================================================
// Declarators
// ===========================================================================

/// Parses a declarator: `pointer? direct-declarator`.
pub fn parse_declarator(parser: &mut Parser<'_>) -> Result<Declarator, ParseError> {
    let start = parser.current().span;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Parse pointer derivations: `* [qualifiers] * [qualifiers] ...`
    while parser.check(TokenKind::Star) {
        parser.advance();
        let quals = parse_type_qualifier_list(parser);
        derived.push(DerivedDeclarator::Pointer { qualifiers: quals });
    }

    // Parse direct declarator
    let (name, mut postfix_derived) = parse_direct_declarator(parser)?;

    // Combine: pointer derivations come first, then postfix (array/function)
    derived.append(&mut postfix_derived);

    // Trailing attributes
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    let end = parser.current().span;
    let span = Span::merge(start, end);

    Ok(Declarator {
        name,
        derived,
        attrs,
        span,
    })
}

/// Parses a direct declarator: `identifier | '(' declarator ')' |
/// direct-declarator '[' ... ']' | direct-declarator '(' ... ')'`.
///
/// Returns the identifier (if any) and accumulated postfix derivations.
fn parse_direct_declarator(
    parser: &mut Parser<'_>,
) -> Result<(Option<Symbol>, Vec<DerivedDeclarator>), ParseError> {
    let mut name: Option<Symbol> = None;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Identifier or parenthesized declarator
    match &parser.current().kind {
        TokenKind::Identifier(sym) => {
            name = Some(*sym);
            parser.advance();
        }
        TokenKind::LeftParen => {
            // Could be a function parameter list or a grouped declarator.
            // Heuristic: if the next token after `(` is a `*` or is another `(`,
            // or is an identifier that is not a declaration start, treat as grouped.
            if matches!(parser.peek_ahead(1).kind, TokenKind::Star) {
                parser.advance(); // consume `(`
                let inner = parse_declarator(parser)?;
                parser.expect(TokenKind::RightParen)?;
                name = inner.name;
                // The inner's derived modifiers are "inner" — they should be
                // applied inside any postfix derivations we parse next.
                // For simplicity in this initial implementation, we prepend them.
                derived.extend(inner.derived);
            }
            // Otherwise, fall through — no name, postfix will pick up params
        }
        _ => {
            // Abstract declarator (no name) — valid in parameter declarations
        }
    }

    // Postfix: array subscripts and function parameter lists
    loop {
        match parser.current().kind {
            TokenKind::LeftBracket => {
                let arr = parse_array_declarator(parser)?;
                derived.push(arr);
            }
            TokenKind::LeftParen => {
                let func = parse_function_declarator(parser)?;
                derived.push(func);
            }
            _ => break,
        }
    }

    Ok((name, derived))
}

/// Parses an array declarator: `[ expression? ]` or `[ static expression ]`
/// or `[ * ]`.
fn parse_array_declarator(parser: &mut Parser<'_>) -> Result<DerivedDeclarator, ParseError> {
    parser.expect(TokenKind::LeftBracket)?;

    let quals = parse_type_qualifier_list(parser);
    let mut is_static = false;

    if parser.eat(TokenKind::Static) {
        is_static = true;
    }

    let size = if parser.check(TokenKind::RightBracket) {
        None
    } else if parser.check(TokenKind::Star) {
        parser.advance(); // VLA: [*]
        None
    } else {
        // Check for zero-length array (GCC extension)
        let expr = super::expressions::parse_assignment_expression(parser)?;
        super::gcc_extensions::validate_zero_length_array(parser, &expr);
        Some(Box::new(expr))
    };

    parser.expect(TokenKind::RightBracket)?;

    Ok(DerivedDeclarator::Array {
        size,
        is_static,
        qualifiers: quals,
    })
}

/// Parses a function declarator: `( parameter-type-list )` or
/// `( identifier-list? )`.
pub(super) fn parse_function_declarator(
    parser: &mut Parser<'_>,
) -> Result<DerivedDeclarator, ParseError> {
    let start = parser.expect(TokenKind::LeftParen)?;

    // Empty parameter list: ()
    if parser.check(TokenKind::RightParen) {
        let end = parser.advance();
        let span = Span::merge(start, end);
        return Ok(DerivedDeclarator::Function {
            params: ParameterList {
                params: Vec::new(),
                variadic: false,
                span,
            },
        });
    }

    // (void) — explicitly no parameters
    if parser.check(TokenKind::Void) && parser.peek_ahead(1).kind == TokenKind::RightParen {
        parser.advance(); // consume `void`
        let end = parser.advance(); // consume `)`
        let span = Span::merge(start, end);
        return Ok(DerivedDeclarator::Function {
            params: ParameterList {
                params: Vec::new(),
                variadic: false,
                span,
            },
        });
    }

    // Parse parameter list
    let mut params = Vec::new();
    let mut variadic = false;

    loop {
        if parser.check(TokenKind::Ellipsis) {
            parser.advance();
            variadic = true;
            break;
        }

        let param = parse_parameter_declaration(parser)?;
        params.push(param);

        if !parser.eat(TokenKind::Comma) {
            break;
        }

        // After comma, check for `...`
        if parser.check(TokenKind::Ellipsis) {
            parser.advance();
            variadic = true;
            break;
        }
    }

    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);

    Ok(DerivedDeclarator::Function {
        params: ParameterList {
            params,
            variadic,
            span,
        },
    })
}

/// Parses a single parameter declaration: `declaration-specifiers declarator?`.
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> Result<Parameter, ParseError> {
    let start = parser.current().span;
    let specifiers = parse_declaration_specifiers(parser)?;

    // Optional declarator (may be abstract, may be omitted)
    let declarator = if parser.check(TokenKind::Comma)
        || parser.check(TokenKind::RightParen)
        || parser.check(TokenKind::Ellipsis)
    {
        None
    } else {
        Some(parse_declarator(parser)?)
    };

    let end_span = declarator.as_ref().map_or(specifiers.span, |d| d.span);
    let span = Span::merge(start, end_span);

    Ok(Parameter {
        specifiers,
        declarator,
        span,
    })
}

// ===========================================================================
// Initializers
// ===========================================================================

/// Parses an initializer: `= expression` or `= { initializer-list }`.
fn parse_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ParseError> {
    if parser.check(TokenKind::LeftBrace) {
        parse_brace_initializer(parser)
    } else {
        let expr = super::expressions::parse_assignment_expression(parser)?;
        Ok(Initializer::Expression(Box::new(expr)))
    }
}

/// Parses a brace-enclosed initializer list: `{ item, item, ... }`.
fn parse_brace_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ParseError> {
    let start = parser.expect(TokenKind::LeftBrace)?;
    let mut items: Vec<InitializerItem> = Vec::new();

    while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
        let item_start = parser.current().span;
        let mut designators = Vec::new();

        // Parse designators: `.field` or `[index]`
        while parser.check(TokenKind::Dot) || parser.check(TokenKind::LeftBracket) {
            if parser.eat(TokenKind::Dot) {
                // Field designator: `.field`
                let field = expect_identifier(parser, "designator field name")?;
                designators.push(super::ast::Designator::Field(field));
            } else {
                // Array index designator: `[expr]`
                parser.advance(); // consume `[`
                let index = super::expressions::parse_conditional_expression(parser)?;
                parser.expect(TokenKind::RightBracket)?;
                designators.push(super::ast::Designator::Index(Box::new(index)));
            }
        }

        // After designators, expect `=`
        if !designators.is_empty() {
            parser.expect(TokenKind::Assign)?;
        }

        let init = parse_initializer(parser)?;
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

// ===========================================================================
// Struct / Union / Enum specifiers
// ===========================================================================

/// Parses a struct-or-union specifier.
///
/// ```text
/// struct-or-union-specifier:
///     struct-or-union identifier? { struct-declaration-list }
///     struct-or-union identifier
/// ```
pub(super) fn parse_struct_or_union_specifier(
    parser: &mut Parser<'_>,
    is_struct: bool,
) -> Result<TypeSpecifier, ParseError> {
    let start = parser.advance(); // consume `struct` or `union`

    // Optional attributes after struct/union keyword
    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    // Optional tag name
    let name = if let TokenKind::Identifier(sym) = &parser.current().kind {
        let sym = *sym;
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Definition body?
    if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume `{`
        let mut fields = Vec::new();

        while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
            match parse_struct_field(parser) {
                Ok(field) => fields.push(field),
                Err(e) => {
                    parser.diagnostics.error(e.span, &e.message);
                    parser.synchronize();
                }
            }
        }

        let end = parser.expect(TokenKind::RightBrace)?;
        let span = Span::merge(start, end);

        if is_struct {
            Ok(TypeSpecifier::Struct {
                name,
                fields: Some(fields),
                attrs,
                span,
            })
        } else {
            Ok(TypeSpecifier::Union {
                name,
                fields: Some(fields),
                attrs,
                span,
            })
        }
    } else {
        // Forward reference: `struct foo`
        if name.is_none() {
            let span = parser.current().span;
            let msg = "expected identifier or '{' after struct/union keyword".to_string();
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier or '{'".to_string()),
            });
        }
        let span = Span::merge(start, parser.current().span);
        if is_struct {
            Ok(TypeSpecifier::Struct {
                name,
                fields: None,
                attrs,
                span,
            })
        } else {
            Ok(TypeSpecifier::Union {
                name,
                fields: None,
                attrs,
                span,
            })
        }
    }
}

/// Parses a single struct/union field declaration.
fn parse_struct_field(parser: &mut Parser<'_>) -> Result<FieldDeclaration, ParseError> {
    let start = parser.current().span;
    let specifiers = parse_declaration_specifiers(parser)?;
    let mut declarators = Vec::new();

    // Parse declarator list (may include bitfield widths)
    if !parser.check(TokenKind::Semicolon) {
        loop {
            let decl_start = parser.current().span;
            let declarator = if parser.check(TokenKind::Colon) {
                // Anonymous bitfield
                None
            } else {
                Some(parse_declarator(parser)?)
            };

            let bitfield_width = if parser.eat(TokenKind::Colon) {
                Some(Box::new(super::expressions::parse_conditional_expression(
                    parser,
                )?))
            } else {
                None
            };

            let decl_end = parser.current().span;
            let decl_span = Span::merge(decl_start, decl_end);

            declarators.push(FieldDeclarator {
                declarator,
                bit_width: bitfield_width,
                span: decl_span,
            });

            if !parser.eat(TokenKind::Comma) {
                break;
            }
        }
    }

    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    Ok(FieldDeclaration {
        specifiers,
        declarators,
        attrs: Vec::new(),
        span,
    })
}

/// Parses an enum specifier.
///
/// ```text
/// enum-specifier:
///     'enum' identifier? { enumerator-list }
///     'enum' identifier
/// ```
pub(super) fn parse_enum_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ParseError> {
    let start = parser.advance(); // consume `enum`

    let attrs = if parser.check(TokenKind::Attribute) {
        super::attributes::parse_attribute_list(parser)?
    } else {
        Vec::new()
    };

    let name = if let TokenKind::Identifier(sym) = &parser.current().kind {
        let sym = *sym;
        parser.advance();
        Some(sym)
    } else {
        None
    };

    if parser.check(TokenKind::LeftBrace) {
        parser.advance(); // consume `{`
        let mut enumerators = Vec::new();

        while !parser.check(TokenKind::RightBrace) && !parser.at_end() {
            let e = parse_enumerator(parser)?;
            enumerators.push(e);
            if !parser.eat(TokenKind::Comma) {
                break;
            }
        }

        let end = parser.expect(TokenKind::RightBrace)?;
        let span = Span::merge(start, end);

        Ok(TypeSpecifier::Enum {
            name,
            enumerators: Some(enumerators),
            attrs,
            span,
        })
    } else {
        if name.is_none() {
            let span = parser.current().span;
            let msg = "expected identifier or '{' after 'enum'".to_string();
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("identifier or '{'".to_string()),
            });
        }
        let span = Span::merge(start, parser.current().span);
        Ok(TypeSpecifier::Enum {
            name,
            enumerators: None,
            attrs,
            span,
        })
    }
}

/// Parses a single enumerator: `identifier [= constant-expression]`.
fn parse_enumerator(parser: &mut Parser<'_>) -> Result<Enumerator, ParseError> {
    let start = parser.current().span;
    let name = expect_identifier(parser, "enumerator name")?;

    let value = if parser.eat(TokenKind::Assign) {
        Some(Box::new(super::expressions::parse_conditional_expression(
            parser,
        )?))
    } else {
        None
    };

    let end = parser.current().span;
    let span = Span::merge(start, end);

    Ok(Enumerator {
        name,
        value,
        attrs: Vec::new(),
        span,
    })
}

// ===========================================================================
// _Static_assert, _Alignas, typeof
// ===========================================================================

/// Parses `_Static_assert(condition, message);`.
fn parse_static_assert(parser: &mut Parser<'_>) -> Result<Declaration, ParseError> {
    let start = parser.advance(); // consume `_Static_assert`
    parser.expect(TokenKind::LeftParen)?;
    let condition = super::expressions::parse_conditional_expression(parser)?;
    parser.expect(TokenKind::Comma)?;

    // Message must be a string literal
    let message = match &parser.current().kind {
        TokenKind::StringLiteral { value, .. } => {
            let val = value.clone();
            parser.advance();
            val
        }
        _ => {
            let span = parser.current().span;
            let msg = "expected string literal in _Static_assert".to_string();
            parser.diagnostics.error(span, &msg);
            return Err(ParseError {
                span,
                message: msg,
                expected: Some("string literal".to_string()),
            });
        }
    };

    parser.expect(TokenKind::RightParen)?;
    let end = parser.expect_semicolon()?;
    let span = Span::merge(start, end);

    Ok(Declaration::StaticAssert {
        condition: Box::new(condition),
        message,
        span,
    })
}

/// Parses `_Alignas(type-name)` or `_Alignas(constant-expression)`.
fn parse_alignas_specifier(parser: &mut Parser<'_>) -> Result<AlignasSpecifier, ParseError> {
    parser.advance(); // consume `_Alignas`
    parser.expect(TokenKind::LeftParen)?;

    // For this initial implementation, parse as expression. Sema will
    // distinguish type-name vs constant-expression.
    let expr = super::expressions::parse_conditional_expression(parser)?;
    parser.expect(TokenKind::RightParen)?;

    Ok(AlignasSpecifier::Expression(Box::new(expr)))
}

/// Parses `typeof(expression)` or `typeof(type-name)`.
fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ParseError> {
    let start = parser.advance(); // consume `typeof` / `__typeof__`
    parser.expect(TokenKind::LeftParen)?;
    let expr = super::expressions::parse_expression(parser)?;
    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);

    // Simplified: store as expression. Sema resolves type-name vs expression.
    Ok(TypeSpecifier::Typeof {
        operand: super::ast::TypeofOperand::Expression(Box::new(expr)),
        span,
    })
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Continues parsing a declaration after the first declarator has been parsed.
/// Handles: `= initializer`, additional declarators (`,`-separated), and the
/// trailing `;`.
fn parse_declaration_rest(
    parser: &mut Parser<'_>,
    specifiers: DeclarationSpecifiers,
    first_declarator: Declarator,
    attrs: Vec<Attribute>,
) -> Result<Declaration, ParseError> {
    let mut declarators = Vec::new();

    // First init-declarator
    let init = if parser.eat(TokenKind::Assign) {
        Some(parse_initializer(parser)?)
    } else {
        None
    };

    let first_span = Span::merge(first_declarator.span, parser.current().span);
    declarators.push(InitDeclarator {
        declarator: first_declarator,
        initializer: init,
        span: first_span,
    });

    // Additional declarators
    while parser.eat(TokenKind::Comma) {
        let decl = parse_declarator(parser)?;
        let init = if parser.eat(TokenKind::Assign) {
            Some(parse_initializer(parser)?)
        } else {
            None
        };
        let decl_span = Span::merge(decl.span, parser.current().span);
        declarators.push(InitDeclarator {
            declarator: decl,
            initializer: init,
            span: decl_span,
        });
    }

    let end = parser.expect_semicolon()?;
    let span = Span::merge(specifiers.span, end);

    // Register typedef names
    if specifiers.storage_class == Some(StorageClass::Typedef) {
        for id in &declarators {
            if let Some(name) = id.declarator.name {
                parser.register_typedef(name);
            }
        }
    }

    Ok(Declaration::Variable {
        specifiers,
        declarators,
        attrs,
        span,
    })
}

/// Parses zero or more type qualifiers: const, volatile, restrict, _Atomic.
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

/// Expects and consumes an identifier, returning its `Symbol`.
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
