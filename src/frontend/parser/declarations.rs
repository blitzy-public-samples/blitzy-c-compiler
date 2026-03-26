#![allow(clippy::result_unit_err)]
//! Declaration parsing for the BCC C11 parser (Phase 4).
//!
//! Handles all C11 declaration forms:
//! - Variable declarations with initializers (`int x = 5;`, `int x, y;`)
//! - Function declarations and definitions (`int f(int x) { ... }`)
//! - Typedef declarations with typedef name registration
//! - Struct/union definitions with members, bitfields, anonymous structs/unions,
//!   and flexible array members
//! - Enum definitions with optional values and trailing commas
//! - `_Static_assert` declarations
//! - `_Alignas` alignment specifiers
//! - Storage class specifiers (`auto`, `register`, `static`, `extern`,
//!   `typedef`, `_Thread_local`)
//! - Function specifiers (`inline`, `_Noreturn`)
//! - GCC `__attribute__((...))` in all valid positions
//! - GCC `__extension__` warning suppression
//! - Abstract declarators for type names (casts, sizeof, _Alignof, _Generic,
//!   compound literals)
//! - Function pointer declarations (`int (*fp)(int)`)
//! - Complex nested declarators
//!
//! # Grammar Coverage
//!
//! The C declaration grammar is one of the most complex parts of the language.
//! Declarators are recursive and build types "inside-out":
//! - `int *(*fp)(int, float)` → `fp` is a pointer to a function(int,float)
//!   returning pointer to int
//!
//! # Typedef-vs-Expression Disambiguation
//!
//! After `typedef int myint;`, `myint` must be recognized as a type name.
//! The parser tracks registered typedef names and uses this information to
//! disambiguate between declarations and expressions in block item parsing.
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node definitions
//! - `super::Parser` — token consumption and recursion tracking
//! - `super::types` — type specifier/qualifier parsing
//! - `super::attributes` — `__attribute__((...))` parsing
//! - `super::expressions` — initializer expression parsing
//! - `super::statements` — compound statement parsing for function bodies
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned string handles
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates. Does NOT depend on `crate::ir`, `crate::passes`,
//! or `crate::backend`.

use super::ast::*;
use super::attributes;
use super::expressions;
use super::statements;
use super::types as type_parser;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Public Entry Points
// ===========================================================================

/// Parse a complete declaration: specifiers + init-declarator-list + `;`.
///
/// This is the primary entry point for parsing declarations within compound
/// statements and at file scope (when called from the parser driver).
///
/// Handles:
/// - Variable declarations: `int x;`, `int x = 5;`, `int x, y, z;`
/// - Typedef declarations: `typedef int myint;`
/// - `_Static_assert(expr, "msg");`
/// - Bare type declarations: `struct foo { ... };`, `enum bar { A, B };`
///
/// # Grammar
///
/// ```text
/// declaration:
///     declaration-specifiers init-declarator-list_opt ';'
///     static_assert-declaration
/// ```
///
/// # Errors
///
/// Returns `Err(())` on unrecoverable parse errors. Diagnostics are emitted
/// via `parser.error()` before returning.
pub fn parse_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()> {
    let start_span = parser.current_span();

    // Skip leading `__extension__` tokens (GCC warning suppression).
    while parser.match_token(&TokenKind::Extension) {}

    // Handle _Static_assert as a declaration.
    if parser.check(&TokenKind::StaticAssert) {
        return parse_static_assert_declaration(parser);
    }

    // Parse declaration specifiers (storage class, type specifiers, qualifiers,
    // function specifiers, alignment, attributes).
    let specifiers = parse_declaration_specifiers(parser)?;

    // Check for a bare specifier declaration with no declarator
    // (e.g., `struct foo;` or `enum bar { A, B };`).
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

    // Determine if this is a typedef declaration for name registration.
    let is_typedef = matches!(specifiers.storage_class, Some(StorageClass::Typedef));

    // Parse the init-declarator list.
    let mut init_declarators = Vec::new();

    // Parse first declarator.
    let first_decl_start = parser.current_span();
    let first_declarator = parse_declarator(parser)?;

    // Register typedef name if this is a typedef declaration.
    if is_typedef {
        register_declarator_name(parser, &first_declarator);
    }

    // Parse optional GCC asm label / register binding: `__asm__("symbol_name")`
    let first_asm_reg = skip_asm_label(parser);

    // Parse optional initializer for the first declarator.
    let first_init = if parser.match_token(&TokenKind::Equal) {
        Some(parse_initializer(parser)?)
    } else {
        None
    };

    // Skip optional trailing GCC __attribute__ after initializer
    skip_trailing_attributes(parser);

    let first_span = parser.make_span(first_decl_start);
    init_declarators.push(InitDeclarator {
        declarator: first_declarator,
        initializer: first_init,
        asm_register: first_asm_reg,
        span: first_span,
    });

    // Parse additional comma-separated init-declarators.
    while parser.match_token(&TokenKind::Comma) {
        let decl_start = parser.current_span();
        let decl = parse_declarator(parser)?;

        // Register additional typedef names.
        if is_typedef {
            register_declarator_name(parser, &decl);
        }

        // Parse optional GCC asm label / register binding
        let extra_asm_reg = skip_asm_label(parser);

        let init = if parser.match_token(&TokenKind::Equal) {
            Some(parse_initializer(parser)?)
        } else {
            None
        };

        // Skip optional trailing GCC __attribute__
        skip_trailing_attributes(parser);

        let decl_span = parser.make_span(decl_start);
        init_declarators.push(InitDeclarator {
            declarator: decl,
            initializer: init,
            asm_register: extra_asm_reg,
            span: decl_span,
        });
    }

    // Expect trailing `;`.
    parser.expect(TokenKind::Semicolon)?;

    let span = parser.make_span(start_span);
    Ok(Declaration {
        specifiers,
        declarators: init_declarators,
        static_assert: None,
        span,
    })
}

/// Parse declaration specifiers: an interleaved sequence of storage class
/// specifiers, type specifiers, type qualifiers, function specifiers,
/// alignment specifiers, and GCC attributes.
///
/// These can appear in ANY order (e.g., `static const int` is equivalent to
/// `int static const`).
///
/// # Grammar
///
/// ```text
/// declaration-specifiers:
///     storage-class-specifier declaration-specifiers_opt
///     type-specifier declaration-specifiers_opt
///     type-qualifier declaration-specifiers_opt
///     function-specifier declaration-specifiers_opt
///     alignment-specifier declaration-specifiers_opt
///     attribute-specifier declaration-specifiers_opt  (GCC)
/// ```
///
/// # Constraints
///
/// - At most one storage class specifier, except `_Thread_local` which can
///   combine with `static` or `extern`.
/// - Type specifiers combine per C11 §6.7.2 (e.g., `unsigned long long`).
/// - Duplicate qualifiers are valid per C11 §6.7.3 (silently accepted).
pub fn parse_declaration_specifiers(parser: &mut Parser<'_>) -> Result<DeclarationSpecifiers, ()> {
    let start_span = parser.current_span();

    let mut storage_class: Option<StorageClass> = None;
    let mut type_specifiers = Vec::new();
    let mut type_qualifiers = Vec::new();
    let mut function_specifiers = Vec::new();
    let mut alignment_specifier: Option<AlignmentSpecifier> = None;
    let mut attrs = Vec::new();

    // Track whether we've seen thread-local for combination validation.
    let mut has_thread_local = false;

    loop {
        // Skip __extension__ tokens (GCC warning suppression).
        if parser.check(&TokenKind::Extension) {
            parser.advance();
            continue;
        }

        match parser.peek().clone() {
            // =================================================================
            // Storage class specifiers
            // =================================================================
            TokenKind::Typedef => {
                if let Err(msg) = validate_storage_class(storage_class, has_thread_local, "typedef")
                {
                    let span = parser.current_span();
                    parser.error(span, &msg);
                }
                parser.advance();
                storage_class = Some(StorageClass::Typedef);
            }
            TokenKind::Extern => {
                if let Err(msg) = validate_storage_class(storage_class, has_thread_local, "extern")
                {
                    let span = parser.current_span();
                    parser.error(span, &msg);
                }
                parser.advance();
                storage_class = Some(StorageClass::Extern);
            }
            TokenKind::Static => {
                if let Err(msg) = validate_storage_class(storage_class, has_thread_local, "static")
                {
                    let span = parser.current_span();
                    parser.error(span, &msg);
                }
                parser.advance();
                storage_class = Some(StorageClass::Static);
            }
            TokenKind::Auto => {
                if let Err(msg) = validate_storage_class(storage_class, has_thread_local, "auto") {
                    let span = parser.current_span();
                    parser.error(span, &msg);
                }
                parser.advance();
                storage_class = Some(StorageClass::Auto);
            }
            TokenKind::Register => {
                if let Err(msg) =
                    validate_storage_class(storage_class, has_thread_local, "register")
                {
                    let span = parser.current_span();
                    parser.error(span, &msg);
                }
                parser.advance();
                storage_class = Some(StorageClass::Register);
            }
            TokenKind::ThreadLocal => {
                // _Thread_local can combine with static or extern.
                if has_thread_local {
                    let span = parser.current_span();
                    parser.error(span, "duplicate '_Thread_local' specifier");
                }
                has_thread_local = true;
                parser.advance();
                // If no other storage class yet, set it; otherwise it
                // combines with static/extern (the existing storage_class stays).
                if storage_class.is_none() {
                    storage_class = Some(StorageClass::ThreadLocal);
                }
            }

            // =================================================================
            // Type specifiers — delegated keywords
            // =================================================================
            TokenKind::Void => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Complex);
            }
            TokenKind::Int128Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int128);
            }
            TokenKind::Float128Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float128);
            }
            TokenKind::Float16Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float16);
            }
            TokenKind::Float32Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float32);
            }
            TokenKind::Float64Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float64);
            }

            // =================================================================
            // Elaborated type specifiers: struct/union/enum
            // =================================================================
            TokenKind::Struct | TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                type_specifiers.push(spec);
            }

            // =================================================================
            // typeof / __typeof__
            // =================================================================
            TokenKind::Typeof => {
                let spec = parse_typeof_specifier(parser)?;
                type_specifiers.push(spec);
            }

            // =================================================================
            // __auto_type — GCC auto type inference
            // =================================================================
            TokenKind::AutoType => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::AutoType);
            }

            // =================================================================
            // _Atomic — could be type specifier `_Atomic(type)` or qualifier
            // =================================================================
            TokenKind::Atomic => {
                // Peek at next token to disambiguate.
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    // _Atomic(type-name) — type specifier.
                    parser.advance(); // consume `_Atomic`
                    parser.advance(); // consume `(`
                    let type_name = parse_type_name(parser)?;
                    parser.expect(TokenKind::RightParen)?;
                    type_specifiers.push(TypeSpecifier::Atomic(Box::new(type_name)));
                } else {
                    // _Atomic without parentheses — type qualifier.
                    parser.advance();
                    type_qualifiers.push(TypeQualifier::Atomic);
                }
            }

            // =================================================================
            // Type qualifiers
            // =================================================================
            TokenKind::Const => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Restrict);
            }

            // =================================================================
            // Function specifiers
            // =================================================================
            TokenKind::Inline => {
                parser.advance();
                function_specifiers.push(FunctionSpecifier::Inline);
            }
            TokenKind::Noreturn => {
                parser.advance();
                function_specifiers.push(FunctionSpecifier::Noreturn);
            }

            // =================================================================
            // Alignment specifier: _Alignas
            // =================================================================
            TokenKind::Alignas => {
                let align = parse_alignas_specifier(parser)?;
                alignment_specifier = Some(align);
            }

            // =================================================================
            // GCC __attribute__((...))
            // =================================================================
            TokenKind::Attribute => {
                let attr_list = attributes::parse_attribute_specifier(parser)?;
                attrs.extend(attr_list);
            }

            // =================================================================
            // Identifier — check if it is a typedef name
            // =================================================================
            TokenKind::Identifier(sym) => {
                let sym_val = sym;
                if parser.is_typedef_name(sym_val) && type_specifiers.is_empty() {
                    parser.advance();
                    type_specifiers.push(TypeSpecifier::TypedefName(sym_val));
                } else {
                    break;
                }
            }

            // No more specifiers.
            _ => break,
        }
    }

    let span = parser.make_span(start_span);

    Ok(DeclarationSpecifiers {
        storage_class,
        type_specifiers,
        type_qualifiers,
        function_specifiers,
        alignment_specifier,
        attributes: attrs,
        span,
    })
}

/// Parse a declarator (the "name and shape" part of a declaration).
///
/// Declarators are recursive and build types "inside-out":
/// - `int *p` → pointer to int
/// - `int arr[10]` → array of 10 ints
/// - `int f(int x)` → function taking int, returning int
/// - `int *(*fp)(int)` → pointer to function(int) returning pointer to int
///
/// # Grammar
///
/// ```text
/// declarator:
///     pointer_opt direct-declarator
///
/// pointer:
///     '*' type-qualifier-list_opt
///     '*' type-qualifier-list_opt pointer
/// ```
pub fn parse_declarator(parser: &mut Parser<'_>) -> Result<Declarator, ()> {
    let start_span = parser.current_span();

    // Parse optional pointer chain: `*`, `* const`, `* volatile`, etc.
    let pointer = parse_pointer_chain(parser)?;

    // Parse direct declarator.
    let direct = parse_direct_declarator(parser)?;

    // Parse optional trailing GCC attributes.
    let mut attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    let span = parser.make_span(start_span);
    Ok(Declarator {
        pointer,
        direct,
        attributes: attrs,
        span,
    })
}

/// Parse a function definition given already-parsed specifiers and declarator.
///
/// Called when parsing external declarations and the parser detects a `{`
/// following the first declarator (instead of `;`, `,`, or `=`).
///
/// Handles:
/// - Standard function definitions: `int main(void) { return 0; }`
/// - K&R-style (old-style) parameter declarations between the declarator
///   and the opening brace
/// - GCC attributes on the function definition
///
/// # Grammar
///
/// ```text
/// function-definition:
///     declaration-specifiers declarator declaration-list_opt compound-statement
/// ```
pub fn parse_function_definition(
    parser: &mut Parser<'_>,
    specifiers: DeclarationSpecifiers,
    declarator: Declarator,
) -> Result<FunctionDefinition, ()> {
    let start_span = specifiers.span;

    // Parse optional K&R-style parameter declarations between the declarator
    // and the opening brace. These are declarations (without initializers)
    // that name the parameter types for old-style function definitions:
    //   int f(a, b) int a; double b; { ... }
    let mut old_style_params = Vec::new();
    while !parser.check(&TokenKind::LeftBrace) && !parser.current.is_eof() {
        // Attempt to parse a declaration (K&R parameter declaration).
        match parse_declaration(parser) {
            Ok(decl) => old_style_params.push(decl),
            Err(()) => {
                parser.synchronize();
                break;
            }
        }
    }

    // Parse optional trailing attributes before the body.
    let mut func_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        func_attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    // Shadow any parameter names that collide with typedef names.
    //
    // In C, a parameter declaration like `void f(list *list)` — where
    // `list` is both a typedef name and the parameter name — causes the
    // parameter name to shadow the typedef for the entire function body.
    // We must remove these names from the typedef set *before* parsing
    // the compound statement, otherwise the parser will treat every use
    // of the parameter name as a type specifier and fail to parse member
    // access expressions (`list->head`).
    //
    // We push a typedef scope here so that the shadowed names are restored
    // when we leave this function definition (the compound statement
    // pushes its *own* nested scope internally).
    parser.push_typedef_scope();
    shadow_function_parameter_typedefs(parser, &declarator);

    // Parse the function body (compound statement).
    let body = statements::parse_compound_statement(parser);

    // Pop the function-level typedef shadow scope, restoring any names
    // that were shadowed by parameter declarations.
    parser.pop_typedef_scope();

    let body = body?;

    let span = parser.make_span(start_span);
    Ok(FunctionDefinition {
        specifiers,
        declarator,
        old_style_params,
        body,
        attributes: func_attrs,
        span,
    })
}

/// Parse an abstract declarator — a declarator without a name, used in
/// type names (casts, sizeof, _Alignof, _Generic, compound literals).
///
/// # Examples
///
/// - `int *` → pointer to int (no name)
/// - `int (*)(int, float)` → pointer to function(int,float) returning int
/// - `int [10]` → array of 10 ints
///
/// # Grammar
///
/// ```text
/// abstract-declarator:
///     pointer
///     pointer_opt direct-abstract-declarator
/// ```
pub fn parse_abstract_declarator(parser: &mut Parser<'_>) -> Result<AbstractDeclarator, ()> {
    let start_span = parser.current_span();

    // Parse optional pointer chain.
    let pointer = parse_pointer_chain(parser)?;

    // Parse optional direct abstract declarator (array/function shapes).
    let direct = parse_direct_abstract_declarator_opt(parser)?;

    let span = parser.make_span(start_span);
    Ok(AbstractDeclarator {
        pointer,
        direct,
        span,
    })
}

/// Parse a type name: specifier-qualifier-list + optional abstract-declarator.
///
/// Used in: cast expressions `(type)expr`, `sizeof(type)`, `_Alignof(type)`,
/// `_Generic` associations, compound literals `(type){init}`, `_Atomic(type)`,
/// `_Alignas(type)`, and `typeof(type)`.
///
/// # Grammar
///
/// ```text
/// type-name:
///     specifier-qualifier-list abstract-declarator_opt
/// ```
pub fn parse_type_name(parser: &mut Parser<'_>) -> Result<TypeName, ()> {
    let start_span = parser.current_span();

    // Parse specifier-qualifier list using the types module.
    let specifier_qualifiers = parse_specifier_qualifier_list_for_type_name(parser)?;

    // Parse optional abstract declarator (pointer, array, function shapes).
    let abstract_declarator = if parser.check(&TokenKind::Star)
        || parser.check(&TokenKind::LeftParen)
        || parser.check(&TokenKind::LeftBracket)
    {
        Some(parse_abstract_declarator(parser)?)
    } else {
        None
    };

    let span = parser.make_span(start_span);
    Ok(TypeName {
        specifier_qualifiers,
        abstract_declarator,
        span,
    })
}

// ===========================================================================
// Internal — Storage Class Validation
// ===========================================================================

/// Validate that a new storage class specifier can be added given the current
/// state. Returns `Err(message)` if the combination is invalid.
fn validate_storage_class(
    current: Option<StorageClass>,
    has_thread_local: bool,
    new_specifier: &str,
) -> Result<(), String> {
    if let Some(existing) = current {
        // _Thread_local can combine with static or extern.
        if has_thread_local
            && matches!(existing, StorageClass::ThreadLocal)
            && (new_specifier == "static" || new_specifier == "extern")
        {
            return Ok(());
        }
        if (new_specifier == "_Thread_local")
            && matches!(existing, StorageClass::Static | StorageClass::Extern)
        {
            return Ok(());
        }
        Err(format!(
            "cannot combine '{}' with previous storage class specifier",
            new_specifier
        ))
    } else {
        Ok(())
    }
}

// ===========================================================================
// Internal — Pointer Chain Parsing
// ===========================================================================

/// Parse a pointer chain: `*`, `**`, `* const *`, `* volatile * restrict`, etc.
///
/// # Grammar
///
/// ```text
/// pointer:
///     '*' type-qualifier-list_opt
///     '*' type-qualifier-list_opt pointer
/// ```
fn parse_pointer_chain(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()> {
    if !parser.match_token(&TokenKind::Star) {
        return Ok(None);
    }

    let start_span = parser.previous_span();
    let mut qualifiers = Vec::new();

    // Parse qualifiers for this pointer level.
    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                // _Atomic without `(` is a qualifier on the pointer.
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    break;
                }
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            // GCC __attribute__ on pointer qualifiers.
            TokenKind::Attribute => {
                // GCC allows attributes on pointer levels; consume but don't
                // store in qualifier list (attributes are on the declarator).
                let _ = attributes::parse_attribute_specifier(parser);
            }
            _ => break,
        }
    }

    // Recursively parse inner pointer.
    let inner = parse_pointer_chain(parser)?;
    let span = parser.make_span(start_span);

    Ok(Some(Pointer {
        qualifiers,
        inner: inner.map(Box::new),
        span,
    }))
}

// ===========================================================================
// Internal — Direct Declarator Parsing
// ===========================================================================

/// Parse a direct declarator: identifier, `(declarator)`, followed by optional
/// array `[...]` or function `(...)` suffixes.
///
/// # Grammar
///
/// ```text
/// direct-declarator:
///     identifier
///     '(' declarator ')'
///     direct-declarator '[' type-qualifier-list_opt assignment-expression_opt ']'
///     direct-declarator '[' 'static' type-qualifier-list_opt assignment-expression ']'
///     direct-declarator '[' type-qualifier-list 'static' assignment-expression ']'
///     direct-declarator '[' type-qualifier-list_opt '*' ']'
///     direct-declarator '(' parameter-type-list ')'
///     direct-declarator '(' identifier-list_opt ')'
/// ```
fn parse_direct_declarator(parser: &mut Parser<'_>) -> Result<DirectDeclarator, ()> {
    let start_span = parser.current_span();

    // Base: identifier or parenthesized declarator.
    let mut base = if let Some(sym) = parser.current_identifier() {
        let id_span = parser.current_span();
        parser.advance();
        DirectDeclarator::Identifier(sym, id_span)
    } else if parser.check(&TokenKind::LeftParen) {
        // Disambiguate: parenthesized declarator vs. function parameter list.
        // A parenthesized declarator begins with `(` followed by `*`, `(`
        // (another nested declarator), or an identifier that is NOT a type name
        // (but this heuristic is tricky).
        //
        // Heuristic: if the token after `(` is `*`, `(`, or `__attribute__`,
        // it's likely a parenthesized declarator. If it's a type keyword or
        // typedef name, it's a function parameter list — but since we're
        // parsing a direct declarator that should have a name, and parameter
        // lists are parsed as suffixes, this `(` must be a parenthesized
        // declarator or we have an abstract declarator.
        let lookahead = parser.peek_nth(0);
        let is_paren_declarator = lookahead.is(&TokenKind::Star)
            || lookahead.is(&TokenKind::LeftParen)
            || lookahead.is(&TokenKind::Attribute)
            || matches!(lookahead.kind, TokenKind::Identifier(_));

        if is_paren_declarator {
            // Further check: if the identifier after `(` is a typedef name,
            // and the token after that is `)`, it could be `(type)` — an abstract
            // declarator or a cast. But in a direct-declarator context, we
            // assume parenthesized declarator.
            parser.advance(); // consume `(`
            let inner = parse_declarator(parser)?;
            parser.expect(TokenKind::RightParen)?;
            DirectDeclarator::Parenthesized(Box::new(inner))
        } else {
            // No name — this is used for abstract-ish contexts or error recovery.
            DirectDeclarator::Identifier(parser.intern(""), Span::dummy())
        }
    } else {
        // No identifier and no parenthesized declarator — abstract/unnamed.
        DirectDeclarator::Identifier(parser.intern(""), Span::dummy())
    };

    // Parse suffixes: `[...]` (array) or `(...)` (function parameters).
    loop {
        if parser.check(&TokenKind::LeftBracket) {
            base = parse_array_suffix(parser, base, start_span)?;
        } else if parser.match_token(&TokenKind::LeftParen) {
            base = parse_function_suffix(parser, base, start_span)?;
        } else {
            break;
        }
    }

    Ok(base)
}

/// Parse an array declarator suffix: `[...]`.
///
/// Handles:
/// - `[N]` — fixed-size array
/// - `[]` — incomplete array (flexible array member or extern)
/// - `[*]` — VLA with unspecified size
/// - `[static N]` — C99 static hint for function parameters
/// - `[const N]` — qualifiers in array declarators
/// - `[0]` — GCC zero-length array extension
fn parse_array_suffix(
    parser: &mut Parser<'_>,
    base: DirectDeclarator,
    start_span: Span,
) -> Result<DirectDeclarator, ()> {
    parser.advance(); // consume `[`

    let mut qualifiers = Vec::new();
    let mut is_static = false;
    let mut is_star = false;

    // Parse optional `static` keyword.
    if parser.match_token(&TokenKind::Static) {
        is_static = true;
    }

    // Parse optional type qualifiers.
    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            _ => break,
        }
    }

    // Parse optional `static` after qualifiers (C99).
    if !is_static && parser.match_token(&TokenKind::Static) {
        is_static = true;
    }

    // Parse size expression, `*` for VLA, or empty for incomplete array.
    let size = if parser.check(&TokenKind::RightBracket) {
        None
    } else if parser.check(&TokenKind::Star) && !is_static {
        // `[*]` — VLA with unspecified size (only in function prototype scope).
        let next = parser.peek_nth(0);
        if next.is(&TokenKind::RightBracket) {
            parser.advance(); // consume `*`
            is_star = true;
            None
        } else {
            // `*` is part of an expression, not a VLA wildcard.
            Some(Box::new(expressions::parse_assignment_expression(parser)?))
        }
    } else {
        Some(Box::new(expressions::parse_assignment_expression(parser)?))
    };

    parser.expect(TokenKind::RightBracket)?;
    let span = parser.make_span(start_span);

    Ok(DirectDeclarator::Array {
        base: Box::new(base),
        size,
        qualifiers,
        is_static,
        is_star,
        span,
    })
}

/// Parse a function declarator suffix: `(parameter-type-list)` or `(identifier-list)`.
fn parse_function_suffix(
    parser: &mut Parser<'_>,
    base: DirectDeclarator,
    start_span: Span,
) -> Result<DirectDeclarator, ()> {
    // The `(` has already been consumed.
    let (params, is_variadic) = parse_parameter_type_list(parser)?;
    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start_span);
    Ok(DirectDeclarator::Function {
        base: Box::new(base),
        params,
        is_variadic,
        span,
    })
}

// ===========================================================================
// Internal — Parameter Type List Parsing
// ===========================================================================

/// Parse a function parameter type list.
///
/// Returns the list of parameter declarations and a flag indicating whether
/// the function is variadic (`...` at end).
///
/// # Grammar
///
/// ```text
/// parameter-type-list:
///     parameter-list
///     parameter-list ',' '...'
///
/// parameter-list:
///     parameter-declaration
///     parameter-list ',' parameter-declaration
/// ```
///
/// Special cases:
/// - `()` → empty parameter list (K&R unspecified)
/// - `(void)` → no parameters
fn parse_parameter_type_list(
    parser: &mut Parser<'_>,
) -> Result<(Vec<ParameterDeclaration>, bool), ()> {
    let mut params = Vec::new();
    let mut is_variadic = false;

    // Empty parameter list: `()`.
    if parser.check(&TokenKind::RightParen) {
        return Ok((params, false));
    }

    // Check for `(void)` — no parameters.
    if parser.check(&TokenKind::Void) {
        let lookahead = parser.peek_nth(0);
        if lookahead.is(&TokenKind::RightParen) {
            parser.advance(); // consume `void`
            return Ok((params, false));
        }
    }

    // Parse parameter declarations.
    loop {
        // Check for `...` (variadic) — can only appear after at least one
        // named parameter, but GCC is lenient.
        if parser.check(&TokenKind::Ellipsis) {
            parser.advance();
            is_variadic = true;
            break;
        }

        let param = parse_parameter_declaration(parser)?;
        params.push(param);

        if !parser.match_token(&TokenKind::Comma) {
            break;
        }

        // Check for `...` after the comma.
        if parser.check(&TokenKind::Ellipsis) {
            parser.advance();
            is_variadic = true;
            break;
        }
    }

    Ok((params, is_variadic))
}

/// Parse a single function parameter declaration.
///
/// A parameter may have a named declarator, an abstract declarator (no name),
/// or just specifiers (no declarator at all).
///
/// # Grammar
///
/// ```text
/// parameter-declaration:
///     declaration-specifiers declarator
///     declaration-specifiers abstract-declarator_opt
/// ```
fn parse_parameter_declaration(parser: &mut Parser<'_>) -> Result<ParameterDeclaration, ()> {
    let start_span = parser.current_span();

    // Parse the specifiers.
    let specifiers = parse_declaration_specifiers(parser)?;

    // Try to determine if what follows is a named declarator, an abstract
    // declarator, or nothing at all.
    if parser.check(&TokenKind::Comma)
        || parser.check(&TokenKind::RightParen)
        || parser.check(&TokenKind::Ellipsis)
    {
        // No declarator — just specifiers.
        let span = parser.make_span(start_span);
        return Ok(ParameterDeclaration {
            specifiers,
            declarator: None,
            abstract_declarator: None,
            span,
        });
    }

    // Attempt to parse a declarator. If the first non-pointer token is an
    // identifier (not a typedef name in this context), it's a named declarator.
    // Otherwise, it's an abstract declarator.
    //
    // Heuristic: if we see `*`, it could be either. If we see an identifier
    // that's not a type name, it's named. If we see `(` or `[`, it could be
    // abstract. We try named first and fall back.
    //
    // Special case: if the specifiers already contain type specifiers and the
    // current token is a typedef name, treat it as a named declarator (the
    // parameter name shadows the typedef). Example:
    //   typedef int (*cb_t)(void);
    //   int foo(cb_t cb_t, void *data);  // second `cb_t` is param name
    let has_type_specs = !specifiers.type_specifiers.is_empty();
    let is_typedef_as_name = has_type_specs
        && matches!(parser.peek(), TokenKind::Identifier(sym) if parser.is_typedef_name(*sym));
    if is_declarator_start(parser) || is_typedef_as_name {
        let decl = parse_declarator(parser)?;
        let span = parser.make_span(start_span);
        Ok(ParameterDeclaration {
            specifiers,
            declarator: Some(decl),
            abstract_declarator: None,
            span,
        })
    } else if parser.check(&TokenKind::Star)
        || parser.check(&TokenKind::LeftParen)
        || parser.check(&TokenKind::LeftBracket)
    {
        let abs = parse_abstract_declarator(parser)?;
        let span = parser.make_span(start_span);
        Ok(ParameterDeclaration {
            specifiers,
            declarator: None,
            abstract_declarator: Some(abs),
            span,
        })
    } else {
        // No declarator.
        let span = parser.make_span(start_span);
        Ok(ParameterDeclaration {
            specifiers,
            declarator: None,
            abstract_declarator: None,
            span,
        })
    }
}

/// Check if the current token sequence looks like it starts a named declarator
/// (as opposed to an abstract declarator).
///
/// A named declarator must eventually reach an identifier. We use a simple
/// heuristic: skip `*` and qualifiers, then check if we see an identifier
/// or `(` followed by what looks like a nested declarator.
fn is_declarator_start(parser: &mut Parser<'_>) -> bool {
    // Quick check: if the current token is an identifier (not a type name
    // in context), it's a named declarator.
    if let TokenKind::Identifier(sym) = parser.peek() {
        // If this identifier is a typedef name, it's ambiguous —
        // could be start of a new declaration. Treat as named declarator
        // only if no type specifiers were just parsed (the caller handles this).
        return !parser.is_typedef_name(*sym);
    }

    // If we see `*`, we need to look further to see if a name follows.
    // For simplicity in parameter context, we try to parse as named.
    if parser.check(&TokenKind::Star) {
        return true;
    }

    // `(` could be parenthesized declarator.
    if parser.check(&TokenKind::LeftParen) {
        let lookahead = parser.peek_nth(0);
        return lookahead.is(&TokenKind::Star)
            || lookahead.is(&TokenKind::LeftParen)
            || matches!(lookahead.kind, TokenKind::Identifier(_));
    }

    false
}

// ===========================================================================
// Internal — Struct/Union Specifier Parsing
// ===========================================================================

/// Parse a struct or union specifier (type specifier form).
///
/// # Grammar
///
/// ```text
/// struct-or-union-specifier:
///     struct-or-union attribute-specifier_opt identifier_opt '{' struct-declaration-list '}'
///     struct-or-union attribute-specifier_opt identifier
/// ```
fn parse_struct_or_union_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();

    // Determine struct vs union.
    let kind = if parser.check(&TokenKind::Union) {
        parser.advance();
        StructOrUnion::Union
    } else {
        parser.advance(); // consume `struct`
        StructOrUnion::Struct
    };

    // Optional attributes after struct/union keyword.
    let mut pre_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        pre_attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional tag name.
    let tag = if let Some(sym) = parser.current_identifier() {
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Optional attributes after tag name.
    while parser.check(&TokenKind::Attribute) {
        pre_attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional member list `{ ... }`.
    let members = if parser.match_token(&TokenKind::LeftBrace) {
        parser.enter_recursion()?;
        let mut member_list = Vec::new();
        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            // Skip empty declarations (bare semicolons) inside struct/union bodies.
            // These are null statements that contribute no members and must not
            // affect struct layout.  Example: `struct S { ; int x; };`
            if parser.check(&TokenKind::Semicolon) {
                parser.advance();
                continue;
            }
            match parse_struct_member(parser) {
                Ok(m) => member_list.push(m),
                Err(()) => {
                    parser.synchronize();
                }
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        parser.leave_recursion();
        Some(member_list)
    } else {
        // Forward declaration or usage — must have a tag name.
        if tag.is_none() && pre_attrs.is_empty() {
            let span = parser.current_span();
            parser.error(span, "expected tag name or member list for struct/union");
        }
        None
    };

    // Optional trailing attributes.
    while parser.check(&TokenKind::Attribute) {
        pre_attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    let span = parser.make_span(start_span);

    let spec = StructOrUnionSpecifier {
        kind,
        tag,
        members,
        attributes: pre_attrs,
        span,
    };

    match kind {
        StructOrUnion::Struct => Ok(TypeSpecifier::Struct(spec)),
        StructOrUnion::Union => Ok(TypeSpecifier::Union(spec)),
    }
}

/// Parse a single struct/union member declaration.
///
/// Handles:
/// - Regular members: `int x;`, `int x, y;`
/// - Bitfields: `int flag : 1;`, `int : 3;` (anonymous bitfield)
/// - Anonymous structs/unions (C11): `struct { int a; };` without a name
/// - Flexible array members: `int data[];`, `int data[0];` (GCC)
/// - `_Static_assert` inside struct/union (C11)
/// - `__attribute__` on members
fn parse_struct_member(parser: &mut Parser<'_>) -> Result<StructMember, ()> {
    let start_span = parser.current_span();

    // Skip __extension__ tokens.
    while parser.match_token(&TokenKind::Extension) {}

    // Handle _Static_assert inside structs (C11).
    if parser.check(&TokenKind::StaticAssert) {
        parser.advance();
        parser.expect(TokenKind::LeftParen)?;
        let _cond = expressions::parse_constant_expression(parser)?;
        if parser.match_token(&TokenKind::Comma) {
            let _ = parse_string_literal_value(parser);
        }
        parser.expect(TokenKind::RightParen)?;
        parser.expect(TokenKind::Semicolon)?;
        let span = parser.make_span(start_span);
        return Ok(StructMember {
            specifiers: SpecifierQualifierList {
                type_specifiers: Vec::new(),
                type_qualifiers: Vec::new(),
                attributes: Vec::new(),
                span,
            },
            declarators: Vec::new(),
            attributes: Vec::new(),
            span,
        });
    }

    // Parse specifier-qualifier list.
    let specifiers = parse_specifier_qualifier_list_for_type_name(parser)?;

    // Parse struct declarators (may have bitfield width).
    let mut declarators = Vec::new();
    if !parser.check(&TokenKind::Semicolon) {
        loop {
            let decl_start = parser.current_span();

            // An anonymous bitfield starts with `:` without a declarator.
            let declarator = if parser.check(&TokenKind::Colon) {
                None
            } else {
                Some(parse_declarator(parser)?)
            };

            // Parse optional bitfield width.
            let bit_width = if parser.match_token(&TokenKind::Colon) {
                Some(Box::new(expressions::parse_constant_expression(parser)?))
            } else {
                None
            };

            // Parse optional attributes on the member declarator.
            let mut _decl_attrs = Vec::new();
            while parser.check(&TokenKind::Attribute) {
                _decl_attrs.extend(attributes::parse_attribute_specifier(parser)?);
            }

            let decl_span = parser.make_span(decl_start);
            declarators.push(StructDeclarator {
                declarator,
                bit_width,
                span: decl_span,
            });

            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
    }

    // Optional attributes after declarators.
    let mut member_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        member_attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    parser.expect(TokenKind::Semicolon)?;
    let span = parser.make_span(start_span);

    Ok(StructMember {
        specifiers,
        declarators,
        attributes: member_attrs,
        span,
    })
}

// ===========================================================================
// Internal — Enum Specifier Parsing
// ===========================================================================

/// Parse an enum specifier (type specifier form).
///
/// # Grammar
///
/// ```text
/// enum-specifier:
///     'enum' attribute-specifier_opt identifier_opt '{' enumerator-list ','? '}'
///     'enum' attribute-specifier_opt identifier
/// ```
fn parse_enum_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();
    parser.advance(); // consume `enum`

    // Optional attributes.
    let mut attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional tag name.
    let tag = if let Some(sym) = parser.current_identifier() {
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Optional attributes after tag.
    while parser.check(&TokenKind::Attribute) {
        attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    // Optional enumerator list `{ ... }`.
    let enumerators = if parser.match_token(&TokenKind::LeftBrace) {
        parser.enter_recursion()?;
        let mut list = Vec::new();

        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            // Skip __extension__ tokens.
            while parser.match_token(&TokenKind::Extension) {}

            let enum_start = parser.current_span();
            match parser.current.kind.clone() {
                TokenKind::Identifier(sym) => {
                    parser.advance();

                    // Optional attributes on enumerator.
                    while parser.check(&TokenKind::Attribute) {
                        let _ = attributes::parse_attribute_specifier(parser);
                    }

                    // Optional explicit value.
                    let value = if parser.match_token(&TokenKind::Equal) {
                        Some(Box::new(expressions::parse_constant_expression(parser)?))
                    } else {
                        None
                    };
                    let espan = parser.make_span(enum_start);
                    list.push(Enumerator {
                        name: sym,
                        value,
                        span: espan,
                    });
                }
                _ => {
                    let span = parser.current_span();
                    parser.error(span, "expected enumerator name");
                    parser.synchronize();
                    break;
                }
            }

            // Comma separator (trailing comma allowed).
            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        parser.leave_recursion();
        Some(list)
    } else {
        // Forward reference — must have tag name.
        if tag.is_none() {
            let span = parser.current_span();
            parser.error(span, "expected tag name or enumerator list for enum");
        }
        None
    };

    // Optional trailing attributes.
    while parser.check(&TokenKind::Attribute) {
        attrs.extend(attributes::parse_attribute_specifier(parser)?);
    }

    let span = parser.make_span(start_span);
    Ok(TypeSpecifier::Enum(EnumSpecifier {
        tag,
        enumerators,
        attributes: attrs,
        span,
    }))
}

// ===========================================================================
// Internal — typeof Specifier Parsing
// ===========================================================================

/// Parse a `typeof(...)` or `__typeof__(...)` type specifier.
///
/// Disambiguates between `typeof(type-name)` and `typeof(expression)` by
/// checking if the first token inside the parentheses could start a type
/// specifier.
fn parse_typeof_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    parser.advance(); // consume `typeof` / `__typeof__`

    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type name or expression?
    let arg = if type_parser::is_type_specifier_start(parser.peek(), parser) {
        // Parse as type name.
        let type_name = parse_type_name(parser)?;
        TypeofArg::TypeName(Box::new(type_name))
    } else {
        // Parse as expression.
        let expr = expressions::parse_expression(parser)?;
        TypeofArg::Expression(Box::new(expr))
    };

    parser.expect(TokenKind::RightParen)?;

    Ok(TypeSpecifier::Typeof(arg))
}

// ===========================================================================
// Internal — _Alignas Specifier Parsing
// ===========================================================================

/// Parse `_Alignas(type-name)` or `_Alignas(constant-expression)`.
///
/// # Grammar
///
/// ```text
/// alignment-specifier:
///     '_Alignas' '(' type-name ')'
///     '_Alignas' '(' constant-expression ')'
/// ```
fn parse_alignas_specifier(parser: &mut Parser<'_>) -> Result<AlignmentSpecifier, ()> {
    let start_span = parser.current_span();

    // Consume `_Alignas`.
    parser.advance();

    // Expect `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type name or expression?
    let arg = if type_parser::is_type_specifier_start(parser.peek(), parser) {
        let type_name = parse_type_name(parser)?;
        AlignasArg::Type(type_name)
    } else {
        let expr = expressions::parse_constant_expression(parser)?;
        AlignasArg::Expression(Box::new(expr))
    };

    // Expect `)`.
    parser.expect(TokenKind::RightParen)?;

    let span = parser.make_span(start_span);
    Ok(AlignmentSpecifier { arg, span })
}

// ===========================================================================
// Internal — _Static_assert Parsing
// ===========================================================================

/// Parse a `_Static_assert` declaration.
///
/// # Grammar
///
/// ```text
/// static_assert-declaration:
///     '_Static_assert' '(' constant-expression ',' string-literal ')' ';'
///     '_Static_assert' '(' constant-expression ')' ';'  (C23/GCC lenient)
/// ```
fn parse_static_assert_declaration(parser: &mut Parser<'_>) -> Result<Declaration, ()> {
    let start_span = parser.current_span();

    // Consume `_Static_assert`.
    parser.advance();

    // Expect `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Parse condition expression (constant expression).
    let condition = expressions::parse_constant_expression(parser)?;

    // Parse optional `,` followed by string-literal message.
    // C11 requires the message; C23 makes it optional; GCC is lenient.
    let message = if parser.match_token(&TokenKind::Comma) {
        parse_string_literal_value(parser)
    } else {
        None
    };

    // Expect `)` `;`.
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
// Internal — Initializer Parsing
// ===========================================================================

/// Parse an initializer: either a simple expression or a brace-enclosed
/// initializer list.
///
/// Called after consuming the `=` token.
///
/// # Grammar
///
/// ```text
/// initializer:
///     assignment-expression
///     '{' initializer-list ','? '}'
/// ```
fn parse_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ()> {
    if parser.check(&TokenKind::LeftBrace) {
        parse_brace_initializer(parser)
    } else {
        let expr = expressions::parse_assignment_expression(parser)?;
        Ok(Initializer::Expression(Box::new(expr)))
    }
}

/// Parse a brace-enclosed initializer list: `{ init, init, ... }`.
///
/// Supports C11 designated initializers:
/// - `.member = value` — struct member designation
/// - `[index] = value` — array index designation
/// - `[low ... high] = value` — GCC array index range designation
/// - Nested designations: `.member.submember = value`, `[i][j] = value`
/// - Out-of-order designations are allowed
/// - Trailing comma is allowed
fn parse_brace_initializer(parser: &mut Parser<'_>) -> Result<Initializer, ()> {
    let start = parser.current_span();
    parser.expect(TokenKind::LeftBrace)?;

    parser.enter_recursion()?;

    let mut items = Vec::new();
    let mut trailing_comma = false;

    // Parse initializer list entries until `}`.
    if !parser.check(&TokenKind::RightBrace) {
        loop {
            let item_start = parser.current_span();
            let mut designators = Vec::new();

            // GCC extension: `field: value` labeled initializer (no dot, uses colon).
            // Unlike standard C `.field = value`, GCC uses `field: value`.
            let mut gcc_labeled = false;
            if let TokenKind::Identifier(sym) = parser.current.kind.clone() {
                let next_tok = parser.peek_nth(0);
                if matches!(next_tok.kind, TokenKind::Colon) {
                    let fspan = parser.current_span();
                    parser.advance(); // consume identifier
                    parser.advance(); // consume colon
                    designators.push(Designator::Field(sym, fspan));
                    gcc_labeled = true;
                }
            }

            // Parse optional designators: `.field`, `[index]`, `[low...high]`.
            while parser.check(&TokenKind::Dot) || parser.check(&TokenKind::LeftBracket) {
                if parser.match_token(&TokenKind::Dot) {
                    // Field designator: `.member`.
                    match parser.current.kind.clone() {
                        TokenKind::Identifier(sym) => {
                            let fspan = parser.current_span();
                            parser.advance();
                            designators.push(Designator::Field(sym, fspan));
                        }
                        _ => {
                            let span = parser.current_span();
                            parser.error(span, "expected field name after '.'");
                            return Err(());
                        }
                    }
                } else if parser.match_token(&TokenKind::LeftBracket) {
                    // Array index designator: `[expr]` or `[low ... high]`.
                    let idx = expressions::parse_constant_expression(parser)?;

                    if parser.match_token(&TokenKind::Ellipsis) {
                        // GCC range designator: `[low ... high]`.
                        let high = expressions::parse_constant_expression(parser)?;
                        parser.expect(TokenKind::RightBracket)?;
                        let dspan = parser.make_span(item_start);
                        designators.push(Designator::IndexRange(
                            Box::new(idx),
                            Box::new(high),
                            dspan,
                        ));
                    } else {
                        parser.expect(TokenKind::RightBracket)?;
                        let dspan = parser.make_span(item_start);
                        designators.push(Designator::Index(Box::new(idx), dspan));
                    }
                }
            }

            // If designators are present, expect `=`.
            // GCC labeled inits (`field: value`) use `:` instead of `=`.
            if !designators.is_empty() && !gcc_labeled {
                parser.expect(TokenKind::Equal)?;
            }

            // Parse the initializer value (recursive for nested braces).
            let init = parse_initializer(parser)?;
            let item_span = parser.make_span(item_start);

            items.push(DesignatedInitializer {
                designators,
                initializer: init,
                span: item_span,
            });

            // Check for comma separator.
            if !parser.match_token(&TokenKind::Comma) {
                trailing_comma = false;
                break;
            }

            // Trailing comma before `}` is allowed.
            if parser.check(&TokenKind::RightBrace) {
                trailing_comma = true;
                break;
            }
        }
    }

    parser.expect(TokenKind::RightBrace)?;
    parser.leave_recursion();

    let span = parser.make_span(start);
    Ok(Initializer::List {
        designators_and_initializers: items,
        trailing_comma,
        span,
    })
}

// ===========================================================================
// Internal — Abstract Declarator Parsing
// ===========================================================================

/// Parse an optional direct abstract declarator suffix (array `[...]` or
/// function `(...)` shapes), possibly parenthesized.
///
/// # Grammar
///
/// ```text
/// direct-abstract-declarator:
///     '(' abstract-declarator ')'
///     direct-abstract-declarator_opt '[' type-qualifier-list_opt assignment-expression_opt ']'
///     direct-abstract-declarator_opt '[' 'static' type-qualifier-list_opt assignment-expression ']'
///     direct-abstract-declarator_opt '[' type-qualifier-list 'static' assignment-expression ']'
///     direct-abstract-declarator_opt '[' '*' ']'
///     direct-abstract-declarator_opt '(' parameter-type-list_opt ')'
/// ```
fn parse_direct_abstract_declarator_opt(
    parser: &mut Parser<'_>,
) -> Result<Option<DirectAbstractDeclarator>, ()> {
    // Check for parenthesized abstract declarator.
    if parser.check(&TokenKind::LeftParen) {
        // Disambiguate: `(abstract-declarator)` vs. `(parameter-list)`.
        // If the token after `(` is `*`, it's a pointer abstract declarator.
        // If it's `)`, it's an empty function abstract declarator.
        // If it starts with a type keyword, it's a parameter list (function shape).
        let lookahead = parser.peek_nth(0);

        if lookahead.is(&TokenKind::Star) {
            // Parenthesized abstract declarator: `(*)(int)` etc.
            parser.advance(); // consume `(`
            let inner = parse_abstract_declarator(parser)?;
            parser.expect(TokenKind::RightParen)?;

            // Check for function/array suffixes after the parenthesized part.
            let mut result = DirectAbstractDeclarator::Parenthesized(Box::new(inner));
            result = parse_abstract_suffix_chain(parser, result)?;
            return Ok(Some(result));
        } else if lookahead.is(&TokenKind::RightParen) {
            // Empty function: `()`.
            parser.advance(); // consume `(`
            parser.advance(); // consume `)`
            let result = DirectAbstractDeclarator::Function {
                base: None,
                params: Vec::new(),
                is_variadic: false,
            };
            return Ok(Some(result));
        } else if lookahead.is(&TokenKind::LeftParen) {
            // Nested parenthesized abstract declarator.
            parser.advance(); // consume `(`
            let inner = parse_abstract_declarator(parser)?;
            parser.expect(TokenKind::RightParen)?;

            let mut result = DirectAbstractDeclarator::Parenthesized(Box::new(inner));
            result = parse_abstract_suffix_chain(parser, result)?;
            return Ok(Some(result));
        } else if type_parser::is_type_specifier_start(&lookahead.kind, parser)
            || lookahead.is(&TokenKind::Ellipsis)
        {
            // Function abstract declarator with parameters.
            parser.advance(); // consume `(`
            let (params, is_variadic) = parse_parameter_type_list(parser)?;
            parser.expect(TokenKind::RightParen)?;
            let result = DirectAbstractDeclarator::Function {
                base: None,
                params,
                is_variadic,
            };
            return Ok(Some(result));
        } else {
            // Could be a parenthesized abstract declarator or something else.
            // Try as function parameter list.
            parser.advance(); // consume `(`
            if parser.check(&TokenKind::RightParen) {
                parser.advance(); // consume `)`
                let result = DirectAbstractDeclarator::Function {
                    base: None,
                    params: Vec::new(),
                    is_variadic: false,
                };
                return Ok(Some(result));
            }
            let (params, is_variadic) = parse_parameter_type_list(parser)?;
            parser.expect(TokenKind::RightParen)?;
            let result = DirectAbstractDeclarator::Function {
                base: None,
                params,
                is_variadic,
            };
            return Ok(Some(result));
        }
    }

    // Check for array abstract declarator: `[...]`.
    // After parsing the first dimension, check for additional suffix
    // chains (e.g., `[2][1]` for multi-dimensional array type names
    // used in `sizeof(int[2][1])` expressions).
    if parser.check(&TokenKind::LeftBracket) {
        let mut result = parse_abstract_array_declarator(parser)?;
        if parser.check(&TokenKind::LeftBracket) || parser.check(&TokenKind::LeftParen) {
            result = parse_abstract_suffix_chain(parser, result)?;
        }
        return Ok(Some(result));
    }

    Ok(None)
}

/// Parse an abstract array declarator: `[N]`, `[]`, `[*]`, `[static N]`.
fn parse_abstract_array_declarator(
    parser: &mut Parser<'_>,
) -> Result<DirectAbstractDeclarator, ()> {
    parser.expect(TokenKind::LeftBracket)?;

    let mut qualifiers = Vec::new();
    let mut is_static = false;

    // Parse optional `static`.
    if parser.match_token(&TokenKind::Static) {
        is_static = true;
    }

    // Parse optional qualifiers.
    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                qualifiers.push(TypeQualifier::Restrict);
            }
            TokenKind::Atomic => {
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            _ => break,
        }
    }

    // Parse optional `static` after qualifiers.
    if !is_static && parser.match_token(&TokenKind::Static) {
        is_static = true;
    }

    // Parse size or `*` or empty.
    let size = if parser.check(&TokenKind::RightBracket) {
        None
    } else if parser.check(&TokenKind::Star) {
        let next = parser.peek_nth(0);
        if next.is(&TokenKind::RightBracket) {
            parser.advance(); // consume `*`
            None // VLA with unspecified size
        } else {
            Some(Box::new(expressions::parse_assignment_expression(parser)?))
        }
    } else {
        Some(Box::new(expressions::parse_assignment_expression(parser)?))
    };

    parser.expect(TokenKind::RightBracket)?;

    Ok(DirectAbstractDeclarator::Array {
        base: None,
        size,
        qualifiers,
        is_static,
    })
}

/// Parse a chain of abstract declarator suffixes (array/function) after an
/// initial direct abstract declarator.
fn parse_abstract_suffix_chain(
    parser: &mut Parser<'_>,
    _base: DirectAbstractDeclarator,
) -> Result<DirectAbstractDeclarator, ()> {
    let mut current = _base;

    loop {
        if parser.check(&TokenKind::LeftBracket) {
            // Array suffix.
            let mut arr = parse_abstract_array_declarator(parser)?;
            // Thread the base through so the chain preserves the inner part.
            if let DirectAbstractDeclarator::Array { ref mut base, .. } = arr {
                *base = Some(Box::new(current));
            }
            current = arr;
        } else if parser.check(&TokenKind::LeftParen) {
            // Function suffix.
            parser.advance(); // consume `(`
            let (params, is_variadic) = parse_parameter_type_list(parser)?;
            parser.expect(TokenKind::RightParen)?;
            current = DirectAbstractDeclarator::Function {
                base: Some(Box::new(current)),
                params,
                is_variadic,
            };
        } else {
            break;
        }
    }

    Ok(current)
}

// ===========================================================================
// Internal — Specifier-Qualifier List for Type Names
// ===========================================================================

/// Parse a specifier-qualifier-list: an interleaved sequence of type specifiers
/// and type qualifiers (no storage class specifiers).
///
/// Used for struct/union member declarations and type names.
fn parse_specifier_qualifier_list_for_type_name(
    parser: &mut Parser<'_>,
) -> Result<SpecifierQualifierList, ()> {
    let start_span = parser.current_span();
    let mut type_specifiers = Vec::new();
    let mut type_qualifiers = Vec::new();
    let mut attrs = Vec::new();

    loop {
        // Skip __extension__ tokens.
        if parser.check(&TokenKind::Extension) {
            parser.advance();
            continue;
        }

        match parser.peek().clone() {
            // Type specifiers.
            TokenKind::Void => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Complex);
            }
            TokenKind::Int128Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int128);
            }
            TokenKind::Float128Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float128);
            }
            TokenKind::Float16Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float16);
            }
            TokenKind::Float32Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float32);
            }
            TokenKind::Float64Keyword => {
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float64);
            }

            // Type qualifiers.
            TokenKind::Const => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Const);
            }
            TokenKind::Volatile => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Volatile);
            }
            TokenKind::Restrict => {
                parser.advance();
                type_qualifiers.push(TypeQualifier::Restrict);
            }

            // _Atomic: specifier or qualifier.
            TokenKind::Atomic => {
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    // _Atomic(type-name) — type specifier.
                    parser.advance(); // consume `_Atomic`
                    parser.advance(); // consume `(`
                    let type_name = parse_type_name(parser)?;
                    parser.expect(TokenKind::RightParen)?;
                    type_specifiers.push(TypeSpecifier::Atomic(Box::new(type_name)));
                } else {
                    // _Atomic — type qualifier.
                    parser.advance();
                    type_qualifiers.push(TypeQualifier::Atomic);
                }
            }

            // Elaborated type specifiers.
            TokenKind::Struct | TokenKind::Union => {
                let spec = parse_struct_or_union_specifier(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Enum => {
                let spec = parse_enum_specifier(parser)?;
                type_specifiers.push(spec);
            }

            // typeof / __typeof__.
            TokenKind::Typeof => {
                let spec = parse_typeof_specifier(parser)?;
                type_specifiers.push(spec);
            }

            // __attribute__((...)).
            TokenKind::Attribute => {
                let attr_list = attributes::parse_attribute_specifier(parser)?;
                attrs.extend(attr_list);
            }

            // Typedef names.
            TokenKind::Identifier(sym) => {
                let sym_val = sym;
                if parser.is_typedef_name(sym_val) && type_specifiers.is_empty() {
                    parser.advance();
                    type_specifiers.push(TypeSpecifier::TypedefName(sym_val));
                } else {
                    break;
                }
            }

            _ => break,
        }
    }

    let span = parser.make_span(start_span);
    Ok(SpecifierQualifierList {
        type_specifiers,
        type_qualifiers,
        attributes: attrs,
        span,
    })
}

// ===========================================================================
// Internal — Typedef Name Registration
// ===========================================================================

/// Extract the declared name from a declarator and register it as a typedef
/// name in the parser.
/// Public wrapper for typedef name registration, callable from mod.rs.
pub fn register_declarator_name_pub(parser: &mut Parser<'_>, decl: &Declarator) {
    register_declarator_name(parser, decl);
}

fn register_declarator_name(parser: &mut Parser<'_>, decl: &Declarator) {
    if let Some(sym) = extract_declarator_name(&decl.direct) {
        let _id: u32 = sym.as_u32();
        parser.register_typedef(sym);
    }
}

/// Extract the identifier name from a direct declarator, traversing through
/// parenthesized and suffix declarators.
fn extract_declarator_name(direct: &DirectDeclarator) -> Option<Symbol> {
    match direct {
        DirectDeclarator::Identifier(sym, _) => {
            // Check if the symbol represents an actual name (not empty).
            if sym.as_u32() != 0 {
                Some(*sym)
            } else {
                // Empty symbol from abstract declarator — check if it was
                // interned with content. We can't distinguish here, so
                // return the symbol anyway.
                Some(*sym)
            }
        }
        DirectDeclarator::Parenthesized(inner) => extract_declarator_name(&inner.direct),
        DirectDeclarator::Array { base, .. } => extract_declarator_name(base),
        DirectDeclarator::Function { base, .. } => extract_declarator_name(base),
    }
}

/// Shadow typedef names that collide with function parameter names.
///
/// In C, a parameter name is allowed to shadow a typedef name in the same
/// scope:
///
/// ```c
/// typedef struct list { ... } list;
/// void foo(list *list) {   // parameter "list" shadows typedef "list"
///     list->head = ...;    // "list" here is the variable, not the type
/// }
/// ```
///
/// This function walks the function declarator to find the parameter list,
/// extracts each parameter's name, and calls `shadow_typedef` for any
/// name that currently matches a typedef.  This must be called *before*
/// the function body is parsed so that the parser treats these names as
/// ordinary identifiers inside the body.
fn shadow_function_parameter_typedefs(parser: &mut Parser<'_>, declarator: &Declarator) {
    // Walk through the direct declarator to find the Function variant.
    fn find_params(direct: &DirectDeclarator) -> Option<&Vec<ParameterDeclaration>> {
        match direct {
            DirectDeclarator::Function { params, .. } => Some(params),
            DirectDeclarator::Parenthesized(inner) => find_params(&inner.direct),
            DirectDeclarator::Array { base, .. } => find_params(base),
            DirectDeclarator::Identifier(..) => None,
        }
    }

    if let Some(params) = find_params(&declarator.direct) {
        for param in params {
            // Extract the parameter name from its declarator (if named).
            if let Some(ref decl) = param.declarator {
                if let Some(sym) = extract_declarator_name(&decl.direct) {
                    if parser.is_typedef_name(sym) {
                        parser.shadow_typedef(sym);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Internal — String Literal Helper
// ===========================================================================

/// Parse a string literal token and return its byte content.
///
/// Returns `None` if the current token is not a string literal (emits
/// an error in that case).
fn parse_string_literal_value(parser: &mut Parser<'_>) -> Option<Vec<u8>> {
    match &parser.current.kind {
        TokenKind::StringLiteral { value, .. } => {
            let mut bytes = value.as_bytes().to_vec();
            parser.advance();
            // Concatenate adjacent string literals (C11 §6.4.5):
            // _Static_assert(expr, "part1" "part2" "part3");
            while let TokenKind::StringLiteral { value, .. } = &parser.current.kind {
                bytes.extend_from_slice(value.as_bytes());
                parser.advance();
            }
            Some(bytes)
        }
        _ => {
            let span = parser.current_span();
            parser.error(span, "expected string literal");
            None
        }
    }
}

// ===========================================================================
// GCC asm labels and trailing attributes on declarations
// ===========================================================================

/// Skip an optional GCC `__asm__("symbol_name")` label on a declaration.
///
/// This construct allows specifying the assembly-level symbol name for a
/// function or variable.  It appears after the declarator and before any
/// initializer or trailing `__attribute__`.
///
/// Grammar: `asm-label: '__asm__' '(' string-literal ')'`
pub(crate) fn skip_asm_label(parser: &mut Parser<'_>) -> Option<String> {
    if parser.check(&TokenKind::Asm) {
        parser.advance(); // consume `__asm__` / `asm`
        if parser.match_token(&TokenKind::LeftParen) {
            // Capture the string literal content (register name or asm label).
            let mut label = String::new();
            let mut depth: u32 = 1;
            while depth > 0 && !parser.check(&TokenKind::Eof) {
                if parser.check(&TokenKind::LeftParen) {
                    depth += 1;
                } else if parser.check(&TokenKind::RightParen) {
                    depth -= 1;
                    if depth == 0 {
                        parser.advance(); // consume closing ')'
                        break;
                    }
                } else if depth == 1 {
                    // Extract register/label name from string literal token
                    if let TokenKind::StringLiteral { ref value, .. } = parser.current.kind {
                        label.push_str(value);
                    }
                }
                parser.advance();
            }
            if !label.is_empty() {
                return Some(label);
            }
        }
    }
    None
}

/// Skip optional trailing `__attribute__((...))` lists that appear after a
/// declarator or initializer but before `;` or `,`.
///
/// Glibc headers attach `__attribute__((__nonnull__(1)))` and similar
/// constructs to extern declarations.  The parser's normal attribute
/// handling runs during declarator parsing; this function handles the
/// after-declarator position where additional attributes may appear.
pub(crate) fn skip_trailing_attributes(parser: &mut Parser<'_>) {
    while parser.check(&TokenKind::Attribute) {
        parser.advance(); // consume `__attribute__`
        if parser.match_token(&TokenKind::LeftParen) {
            let mut depth: u32 = 1;
            while depth > 0 && !parser.check(&TokenKind::Eof) {
                if parser.check(&TokenKind::LeftParen) {
                    depth += 1;
                } else if parser.check(&TokenKind::RightParen) {
                    depth -= 1;
                    if depth == 0 {
                        parser.advance(); // consume closing ')'
                        break;
                    }
                }
                parser.advance();
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    // Unit tests are run via the full module test suite and checkpoint
    // integration tests. This section is reserved for future targeted
    // declaration parsing unit tests.
}
