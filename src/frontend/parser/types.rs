//! Type specifier and qualifier parsing module for the BCC C11 parser.
//!
//! This module is the canonical source for all type-related parsing in the
//! BCC compiler frontend. It handles:
//!
//! - **Base type specifiers:** `void`, `char`, `short`, `int`, `long`, `float`,
//!   `double`, `signed`, `unsigned`, `_Bool`, `_Complex`
//! - **Multi-keyword type specifiers:** `unsigned long long int`, `signed char`,
//!   `long double`, etc., validated and combined via [`combine_type_specifiers`]
//! - **Composite type specifiers:** `struct`, `union`, `enum` (delegated to
//!   `declarations.rs`)
//! - **Typedef names:** resolved via `parser.is_typedef_name()`
//! - **GCC extensions:** `typeof`/`__typeof__`/`__typeof`, `__extension__`,
//!   `__signed__`, `__const__`, `__volatile__`, `__restrict__`, transparent unions
//! - **C11 type constructs:** `_Alignof`, `_Alignas`, `_Noreturn`, `_Generic`,
//!   `_Atomic` (qualifier/specifier disambiguation), `_Complex`, `_Thread_local`
//! - **Type qualifiers:** `const`, `volatile`, `restrict`, `_Atomic` (as qualifier)
//!
//! # Architecture
//!
//! All public entry functions take `&mut Parser<'_>` as their first parameter
//! and return `Result<T, ParseError>`. The module delegates to sibling modules
//! (`declarations.rs`, `expressions.rs`, `attributes.rs`) for sub-grammar
//! parsing but is self-contained for type specifier/qualifier logic.

use super::ast::{
    AbstractDeclarator, AlignasSpecifier, AlignofOperand, Attribute, DeclarationSpecifiers,
    DerivedDeclarator, Expression, FunctionSpecifiers, Span, SpecifierQualifierList, StorageClass,
    TypeName, TypeQualifiers, TypeSpecifier, TypeofOperand,
};
use super::{ParseError, ParseResult, Parser};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::{Token, TokenKind};

// ===========================================================================
// CombinedType — result of multi-keyword type specifier resolution
// ===========================================================================

/// Represents the resolved type after combining multiple type specifier
/// keywords into a single canonical type.
///
/// C allows multi-keyword type specifiers such as `unsigned long long int`.
/// The parser collects individual specifier tokens (`Unsigned`, `Long`,
/// `Long`, `Int`) into a `Vec<TypeSpecifier>`, then [`combine_type_specifiers`]
/// resolves them into a single `CombinedType` variant.
///
/// # Examples
///
/// ```text
/// unsigned long long int  →  CombinedType::UnsignedLongLong
/// signed char             →  CombinedType::SignedChar
/// long double             →  CombinedType::LongDouble
/// _Complex float          →  CombinedType::ComplexFloat
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CombinedType {
    /// `void`
    Void,
    /// `_Bool`
    Bool,
    /// `signed char` or `char` when signed by default
    SignedChar,
    /// `unsigned char`
    UnsignedChar,
    /// `short`, `short int`, `signed short`, `signed short int`
    SignedShort,
    /// `unsigned short`, `unsigned short int`
    UnsignedShort,
    /// `int`, `signed`, `signed int`
    SignedInt,
    /// `unsigned`, `unsigned int`
    UnsignedInt,
    /// `long`, `long int`, `signed long`, `signed long int`
    SignedLong,
    /// `unsigned long`, `unsigned long int`
    UnsignedLong,
    /// `long long`, `long long int`, `signed long long`, `signed long long int`
    SignedLongLong,
    /// `unsigned long long`, `unsigned long long int`
    UnsignedLongLong,
    /// `float`
    Float,
    /// `double`
    Double,
    /// `long double`
    LongDouble,
    /// `_Complex float` or `float _Complex`
    ComplexFloat,
    /// `_Complex double` or `double _Complex`
    ComplexDouble,
    /// `_Complex long double` or `long double _Complex`
    ComplexLongDouble,
    /// `struct [name] [{ ... }]` — composite, not further resolved here
    Struct,
    /// `union [name] [{ ... }]` — composite, not further resolved here
    Union,
    /// `enum [name] [{ ... }]` — composite, not further resolved here
    Enum,
    /// A typedef name used as a type specifier
    TypedefName,
    /// `typeof(...)` / `__typeof__(...)` — GCC extension
    Typeof,
}

// ===========================================================================
// Bit flags for multi-keyword specifier combination
// ===========================================================================

/// Internal bit flags used by [`combine_type_specifiers`] to track which
/// base type keywords have been seen. A `long long` sets both LONG and
/// LONG_LONG bits.
const SPEC_VOID: u32 = 1 << 0;
const SPEC_CHAR: u32 = 1 << 1;
const SPEC_SHORT: u32 = 1 << 2;
const SPEC_INT: u32 = 1 << 3;
const SPEC_LONG: u32 = 1 << 4;
const SPEC_LONG_LONG: u32 = 1 << 5;
const SPEC_FLOAT: u32 = 1 << 6;
const SPEC_DOUBLE: u32 = 1 << 7;
const SPEC_SIGNED: u32 = 1 << 8;
const SPEC_UNSIGNED: u32 = 1 << 9;
const SPEC_BOOL: u32 = 1 << 10;
const SPEC_COMPLEX: u32 = 1 << 11;
const SPEC_STRUCT: u32 = 1 << 12;
const SPEC_UNION: u32 = 1 << 13;
const SPEC_ENUM: u32 = 1 << 14;
const SPEC_TYPEDEF_NAME: u32 = 1 << 15;
const SPEC_TYPEOF: u32 = 1 << 16;

// ===========================================================================
// Public token classification helpers
// ===========================================================================

/// Returns `true` if `kind` is a token that can begin or appear in a type
/// specifier position.
///
/// This function identifies all tokens that the parser should treat as type
/// specifier keywords. It does **not** check for typedef names (which require
/// parser state); callers must check `TokenKind::Identifier` separately
/// using `parser.is_typedef_name()`.
///
/// Used by the parser for declaration-vs-expression disambiguation and for
/// determining when to enter type parsing mode.
///
/// # Token Categories
///
/// - Basic type specifiers: `void`, `char`, `short`, `int`, `long`, etc.
/// - C11 type specifiers: `_Bool`, `_Complex`, `_Atomic`
/// - Composite type introducers: `struct`, `union`, `enum`
/// - GCC type extensions: `typeof`/`__typeof__`, `__signed__`, `__extension__`
///
/// Note: `_Atomic` can be both a type specifier (`_Atomic(int)`) and a type
/// qualifier (`_Atomic int`). This function returns `true` for both uses;
/// the parser disambiguates based on the following token.
pub fn is_type_specifier_token(kind: &TokenKind) -> bool {
    matches!(
        kind,
        // Basic type specifiers
        TokenKind::Void
            | TokenKind::Char
            | TokenKind::Short
            | TokenKind::Int
            | TokenKind::Long
            | TokenKind::Float
            | TokenKind::Double
            | TokenKind::Signed
            | TokenKind::Unsigned
            // C11 type specifiers
            | TokenKind::Bool
            | TokenKind::Complex
            | TokenKind::Atomic
            // Composite type introducers
            | TokenKind::Struct
            | TokenKind::Union
            | TokenKind::Enum
            // GCC typeof (acts as type specifier)
            | TokenKind::TypeofKeyword
            // GCC __signed__ (equivalent to signed)
            | TokenKind::SignedGcc
            // __extension__ can precede type specifiers
            | TokenKind::Extension
    )
}

/// Returns `true` if the current token can begin a type name (used for
/// disambiguating `typeof(type-name)` vs `typeof(expression)`,
/// `_Atomic(type-name)`, `_Alignof(type-name)`, and `_Alignas(type-name)`).
///
/// Checks for type specifier keywords, type qualifier keywords (including
/// GCC variants), and typedef names. This is the types.rs analog of
/// declarations.rs's `is_specifier_qualifier_start`.
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

        // Composite type introducers
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // Type qualifiers (standard and GCC variants)
        TokenKind::Const
        | TokenKind::Volatile
        | TokenKind::Restrict
        | TokenKind::Atomic
        | TokenKind::ConstGcc
        | TokenKind::VolatileGcc
        | TokenKind::RestrictGcc => true,

        // GCC extensions used as type specifiers
        TokenKind::TypeofKeyword | TokenKind::Attribute | TokenKind::Extension => true,

        // GCC __signed__
        TokenKind::SignedGcc => true,

        // Typedef name — check parser's typedef registry
        TokenKind::Identifier(sym) => parser.is_typedef_name(*sym),

        _ => false,
    }
}

/// Returns `true` if the current token can begin an abstract declarator
/// (pointer, grouped declarator, or array declarator).
fn can_start_abstract_declarator(parser: &Parser<'_>) -> bool {
    matches!(
        parser.current().kind,
        TokenKind::Star | TokenKind::LeftParen | TokenKind::LeftBracket
    )
}

/// Helper to extract the source span from a `TypeSpecifier`, if available.
fn type_specifier_span(spec: &TypeSpecifier) -> Option<Span> {
    match spec {
        TypeSpecifier::Atomic(ref tn) => Some(tn.span),
        TypeSpecifier::Struct { span, .. }
        | TypeSpecifier::Union { span, .. }
        | TypeSpecifier::Enum { span, .. }
        | TypeSpecifier::TypedefName { span, .. }
        | TypeSpecifier::Typeof { span, .. } => Some(*span),
        // Simple keyword specifiers don't carry a span — use the token span
        // from the caller
        _ => None,
    }
}

// ===========================================================================
// parse_type_specifier — single type specifier parsing
// ===========================================================================

/// Parses a single type specifier from the current token position.
///
/// This function consumes exactly one type specifier token (or a compound
/// specifier like `struct { ... }` or `typeof(expr)`) and returns the
/// corresponding [`TypeSpecifier`] AST node.
///
/// # Grammar
///
/// ```text
/// type-specifier:
///     void | char | short | int | long | float | double
///     | signed | unsigned | _Bool | _Complex
///     | atomic-type-specifier
///     | struct-or-union-specifier
///     | enum-specifier
///     | typedef-name
///     | typeof-specifier
/// ```
///
/// # Errors
///
/// Returns `ParseError` if the current token is not a valid type specifier.
pub fn parse_type_specifier(parser: &mut Parser<'_>) -> ParseResult<TypeSpecifier> {
    let tok: Token = parser.current().clone();
    match tok.kind {
        // ---------------------------------------------------------------
        // Simple keyword specifiers
        // ---------------------------------------------------------------
        TokenKind::Void => {
            parser.advance();
            Ok(TypeSpecifier::Void)
        }
        TokenKind::Char => {
            parser.advance();
            Ok(TypeSpecifier::Char)
        }
        TokenKind::Short => {
            parser.advance();
            Ok(TypeSpecifier::Short)
        }
        TokenKind::Int => {
            parser.advance();
            Ok(TypeSpecifier::Int)
        }
        TokenKind::Long => {
            parser.advance();
            Ok(TypeSpecifier::Long)
        }
        TokenKind::Float => {
            parser.advance();
            Ok(TypeSpecifier::Float)
        }
        TokenKind::Double => {
            parser.advance();
            Ok(TypeSpecifier::Double)
        }
        TokenKind::Signed | TokenKind::SignedGcc => {
            parser.advance();
            Ok(TypeSpecifier::Signed)
        }
        TokenKind::Unsigned => {
            parser.advance();
            Ok(TypeSpecifier::Unsigned)
        }
        TokenKind::Bool => {
            parser.advance();
            Ok(TypeSpecifier::Bool)
        }
        TokenKind::Complex => {
            parser.advance();
            Ok(TypeSpecifier::Complex)
        }

        // ---------------------------------------------------------------
        // _Atomic(type-name) — atomic type specifier form
        // ---------------------------------------------------------------
        TokenKind::Atomic => {
            // _Atomic followed by '(' is a type specifier: _Atomic(type-name)
            // _Atomic without '(' is a type qualifier (handled elsewhere)
            if parser.peek_ahead(1).kind == TokenKind::LeftParen {
                parse_atomic_type_specifier(parser)
            } else {
                let span = tok.span;
                let msg =
                    "_Atomic without parentheses is a type qualifier, not a specifier".to_string();
                parser.diagnostics.error(span, &msg);
                Err(ParseError {
                    span,
                    message: msg,
                    expected: Some("_Atomic(type-name)".to_string()),
                })
            }
        }

        // ---------------------------------------------------------------
        // Composite type specifiers — delegate to declarations module
        // ---------------------------------------------------------------
        TokenKind::Struct => super::declarations::parse_struct_or_union_specifier(parser, true),
        TokenKind::Union => super::declarations::parse_struct_or_union_specifier(parser, false),
        TokenKind::Enum => super::declarations::parse_enum_specifier(parser),

        // ---------------------------------------------------------------
        // typeof / __typeof__ / __typeof — GCC extension
        // ---------------------------------------------------------------
        TokenKind::TypeofKeyword => parse_typeof(parser),

        // ---------------------------------------------------------------
        // __extension__ — suppress GCC warnings, parse the next specifier
        // ---------------------------------------------------------------
        TokenKind::Extension => {
            parser.advance(); // consume __extension__
            parse_type_specifier(parser)
        }

        // ---------------------------------------------------------------
        // Typedef name — identifier registered as a typedef
        // ---------------------------------------------------------------
        TokenKind::Identifier(sym) => {
            let sym: Symbol = sym;
            if parser.is_typedef_name(sym) {
                let span = tok.span;
                // Validate symbol handle — non-zero IDs indicate interned symbols
                debug_assert!(sym.as_u32() < u32::MAX, "invalid symbol handle");
                parser.advance();
                Ok(TypeSpecifier::TypedefName { name: sym, span })
            } else {
                let span = tok.span;
                let msg = "expected type specifier, found identifier".to_string();
                parser.diagnostics.error(span, &msg);
                Err(ParseError {
                    span,
                    message: msg,
                    expected: Some("type specifier".to_string()),
                })
            }
        }

        // ---------------------------------------------------------------
        // Not a type specifier
        // ---------------------------------------------------------------
        _ => {
            let span = tok.span;
            let msg = format!("expected type specifier, found '{}'", tok.kind);
            parser.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some("type specifier".to_string()),
            })
        }
    }
}

// ===========================================================================
// parse_type_qualifiers — qualifier-only parsing
// ===========================================================================

/// Parses zero or more consecutive type qualifiers and returns them as a
/// [`TypeQualifiers`] bitmask.
///
/// Recognises both standard C11 qualifiers and GCC-compatible alternate
/// spellings:
///
/// | Standard   | GCC Alternate      |
/// |------------|--------------------|
/// | `const`    | `__const__`        |
/// | `volatile` | `__volatile__`     |
/// | `restrict` | `__restrict__`     |
/// | `_Atomic`  | (no alternate)     |
///
/// `_Atomic` is treated as a qualifier **only** when NOT followed by `(`.
/// When `_Atomic(` is seen, parsing stops (the caller should handle it as
/// a type specifier via `parse_atomic_type_specifier`).
///
/// # Returns
///
/// A [`TypeQualifiers`] with the appropriate flags set.  If no qualifiers
/// are present, returns a default (all-false) instance.
pub fn parse_type_qualifiers(parser: &mut Parser<'_>) -> TypeQualifiers {
    let mut quals = TypeQualifiers::default();
    loop {
        match parser.current().kind {
            // Standard const and GCC __const__
            TokenKind::Const | TokenKind::ConstGcc => {
                quals.is_const = true;
                parser.advance();
            }
            // Standard volatile and GCC __volatile__
            TokenKind::Volatile | TokenKind::VolatileGcc => {
                quals.is_volatile = true;
                parser.advance();
            }
            // Standard restrict and GCC __restrict__
            TokenKind::Restrict | TokenKind::RestrictGcc => {
                quals.is_restrict = true;
                parser.advance();
            }
            // _Atomic as qualifier (only when NOT followed by '(')
            TokenKind::Atomic => {
                if parser.peek_ahead(1).kind != TokenKind::LeftParen {
                    quals.is_atomic = true;
                    parser.advance();
                } else {
                    // _Atomic( is a type specifier, not a qualifier — stop
                    break;
                }
            }
            _ => break,
        }
    }
    quals
}

// ===========================================================================
// parse_specifier_qualifier_list — the core type parsing loop
// ===========================================================================

/// Parses a specifier-qualifier list: one or more type specifiers and/or
/// type qualifiers, as used in type names (casts, sizeof, etc.), struct
/// member declarations, and similar contexts.
///
/// # Grammar
///
/// ```text
/// specifier-qualifier-list:
///     type-specifier specifier-qualifier-list(opt)
///     type-qualifier specifier-qualifier-list(opt)
///     alignment-specifier specifier-qualifier-list(opt)  // GCC extension
/// ```
///
/// This function handles:
/// - All basic type specifier keywords
/// - GCC alternate keyword spellings (`__signed__`, `__const__`, etc.)
/// - `_Atomic` qualifier/specifier disambiguation
/// - `typeof`/`__typeof__` with expression/type-name disambiguation
/// - `__extension__` suppression
/// - `__attribute__((...))` parsing (attributes attached to types)
/// - Transparent union attribute detection
/// - Struct/union/enum delegation to `declarations.rs`
/// - Typedef name resolution
///
/// # Errors
///
/// Returns `ParseError` if no type specifiers or qualifiers are found.
pub fn parse_specifier_qualifier_list(
    parser: &mut Parser<'_>,
) -> ParseResult<SpecifierQualifierList> {
    parser.enter_recursion()?;
    let start = parser.current().span;
    let mut specifiers: Vec<TypeSpecifier> = Vec::new();
    let mut qualifiers = TypeQualifiers::default();
    let mut last_span = start;
    // Track __extension__ presence for potential use by callers via
    // spec_qual_to_declaration_specifiers. The specifier-qualifier list
    // itself does not store this flag, but parse_specifier_qualifier_list_ext
    // can be used when the flag is needed.
    let mut _has_extension = false;
    let mut count = 0usize;

    loop {
        match parser.current().kind.clone() {
            // ---------------------------------------------------------------
            // Basic type specifier keywords
            // ---------------------------------------------------------------
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
            TokenKind::Signed | TokenKind::SignedGcc => {
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

            // ---------------------------------------------------------------
            // Struct / Union / Enum — delegate to declarations module
            // ---------------------------------------------------------------
            TokenKind::Struct => {
                let spec = super::declarations::parse_struct_or_union_specifier(parser, true)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }
            TokenKind::Union => {
                let spec = super::declarations::parse_struct_or_union_specifier(parser, false)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }
            TokenKind::Enum => {
                let spec = super::declarations::parse_enum_specifier(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }

            // ---------------------------------------------------------------
            // Type qualifiers (standard)
            // ---------------------------------------------------------------
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

            // ---------------------------------------------------------------
            // Type qualifiers (GCC alternate spellings)
            // ---------------------------------------------------------------
            TokenKind::ConstGcc => {
                qualifiers.is_const = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::VolatileGcc => {
                qualifiers.is_volatile = true;
                last_span = parser.advance();
                count += 1;
            }
            TokenKind::RestrictGcc => {
                qualifiers.is_restrict = true;
                last_span = parser.advance();
                count += 1;
            }

            // ---------------------------------------------------------------
            // _Atomic — disambiguate qualifier vs type specifier
            // ---------------------------------------------------------------
            TokenKind::Atomic => {
                if parser.peek_ahead(1).kind == TokenKind::LeftParen {
                    // _Atomic(type-name) — type specifier form
                    let spec = parse_atomic_type_specifier(parser)?;
                    if let Some(sp) = type_specifier_span(&spec) {
                        last_span = sp;
                    }
                    specifiers.push(spec);
                } else {
                    // _Atomic without parens — type qualifier
                    qualifiers.is_atomic = true;
                    last_span = parser.advance();
                }
                count += 1;
            }

            // ---------------------------------------------------------------
            // typeof / __typeof__ / __typeof — GCC extension
            // ---------------------------------------------------------------
            TokenKind::TypeofKeyword => {
                let spec = parse_typeof(parser)?;
                if let Some(sp) = type_specifier_span(&spec) {
                    last_span = sp;
                }
                specifiers.push(spec);
                count += 1;
            }

            // ---------------------------------------------------------------
            // __extension__ — suppress GCC extension warnings
            // ---------------------------------------------------------------
            TokenKind::Extension => {
                _has_extension = true;
                last_span = parser.advance();
                count += 1;
                // Continue parsing — __extension__ acts as a prefix modifier
            }

            // ---------------------------------------------------------------
            // __attribute__((...)) — GCC attributes on specifier-qualifier
            // ---------------------------------------------------------------
            TokenKind::Attribute => {
                // Parse attributes in specifier-qualifier context.
                // Transparent union and other type attributes are parsed here
                // but cannot be stored in SpecifierQualifierList directly.
                // Callers that need attributes should use
                // spec_qual_to_declaration_specifiers() after parsing.
                let attrs: Vec<Attribute> = super::attributes::parse_attribute_list(parser)?;
                // Consume parsed attributes — in specifier-qualifier context
                // these are typically transparent_union, aligned, packed, etc.
                // They are consumed here and the caller can reconstruct them
                // via a full DeclarationSpecifiers if needed.
                drop(attrs);
                count += 1;
            }

            // ---------------------------------------------------------------
            // _Noreturn — function specifier (warn if in spec-qual context)
            // ---------------------------------------------------------------
            TokenKind::Noreturn => {
                // _Noreturn is a function specifier, not a type specifier.
                // In a specifier-qualifier context (e.g., struct fields) it
                // is technically invalid, but GCC accepts it. Emit a warning
                // and skip it so parsing can continue.
                parser.diagnostics.warning(
                    parser.current().span,
                    "'_Noreturn' is a function specifier, not valid in \
                     specifier-qualifier context",
                );
                last_span = parser.advance();
                count += 1;
            }

            // ---------------------------------------------------------------
            // _Thread_local — storage class (warn if in spec-qual context)
            // ---------------------------------------------------------------
            TokenKind::ThreadLocal => {
                // _Thread_local is a storage class specifier. Warn and skip.
                parser.diagnostics.warning(
                    parser.current().span,
                    "'_Thread_local' is a storage class specifier, not valid \
                     in specifier-qualifier context",
                );
                last_span = parser.advance();
                count += 1;
            }

            // ---------------------------------------------------------------
            // _Generic — not a type specifier (expression construct)
            // ---------------------------------------------------------------
            TokenKind::Generic => {
                // _Generic is a primary expression, not a type specifier.
                // If encountered here, break to let expression parsing handle it.
                break;
            }

            // ---------------------------------------------------------------
            // Typedef — storage class keyword, not valid here
            // ---------------------------------------------------------------
            TokenKind::Typedef => {
                // `typedef` is a storage class specifier that should not appear
                // in a specifier-qualifier list. This can happen during error
                // recovery or in malformed code. Emit a diagnostic and skip.
                parser.diagnostics.warning(
                    parser.current().span,
                    "'typedef' is a storage class specifier, not valid in \
                     specifier-qualifier context",
                );
                last_span = parser.advance();
                count += 1;
            }

            // ---------------------------------------------------------------
            // Typedef name (identifier registered as typedef)
            // ---------------------------------------------------------------
            TokenKind::Identifier(sym) => {
                // Only treat an identifier as a typedef name if we haven't
                // already seen another base type specifier (avoids misinterpreting
                // variable names as types in `int size_t;`).
                if specifiers.is_empty() && parser.is_typedef_name(sym) {
                    let tspan = parser.current().span;
                    // Validate the Symbol handle
                    debug_assert!(sym.as_u32() < u32::MAX, "invalid symbol handle");
                    specifiers.push(TypeSpecifier::TypedefName {
                        name: sym,
                        span: tspan,
                    });
                    last_span = parser.advance();
                    count += 1;
                } else {
                    break;
                }
            }

            // ---------------------------------------------------------------
            // Not a specifier or qualifier — stop parsing
            // ---------------------------------------------------------------
            _ => break,
        }
    }

    parser.leave_recursion();

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
            expected: Some("type specifier or qualifier".to_string()),
        });
    }

    let span = Span::merge(start, last_span);
    Ok(SpecifierQualifierList {
        specifiers,
        qualifiers,
        span,
    })
}

// ===========================================================================
// parse_typeof — typeof / __typeof__ / __typeof GCC extension
// ===========================================================================

/// Parses a `typeof`/`__typeof__`/`__typeof` type specifier (GCC extension).
///
/// Disambiguates between `typeof(type-name)` and `typeof(expression)` by
/// checking if the token after `(` can begin a type name.
///
/// # Grammar
///
/// ```text
/// typeof-specifier:
///     typeof ( expression )
///     typeof ( type-name )
/// ```
///
/// # Examples
///
/// ```c
/// typeof(int)           // type-name form
/// typeof(x + 1)         // expression form
/// __typeof__(struct foo) // type-name form with GCC spelling
/// ```
///
/// # Errors
///
/// Returns `ParseError` if the typeof expression cannot be parsed.
pub fn parse_typeof(parser: &mut Parser<'_>) -> ParseResult<TypeSpecifier> {
    let start = parser.current().span;
    parser.advance(); // consume typeof/__typeof__/__typeof keyword
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: if the next token can start a type name, parse as
    // type-name; otherwise parse as expression.
    let operand = if is_type_name_start(parser) {
        let type_name = parse_type_name(parser)?;
        TypeofOperand::TypeName(Box::new(type_name))
    } else {
        let expr = super::expressions::parse_expression(parser)?;
        TypeofOperand::Expression(Box::new(expr))
    };

    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);

    Ok(TypeSpecifier::Typeof { operand, span })
}

// ===========================================================================
// parse_alignof — _Alignof / __alignof__ operator
// ===========================================================================

/// Parses an `_Alignof` expression (C11 §6.5.3.4) or `__alignof__`
/// (GCC extension).
///
/// The standard form requires a parenthesised type name:
///   `_Alignof ( type-name )`
///
/// The GCC extension also allows an expression operand:
///   `__alignof__ ( expression )`
///
/// # Grammar
///
/// ```text
/// alignof-expression:
///     _Alignof ( type-name )
///     __alignof__ ( expression )      // GCC extension
/// ```
///
/// # Returns
///
/// An [`Expression::Alignof`] AST node.
pub fn parse_alignof(parser: &mut Parser<'_>) -> ParseResult<Expression> {
    let start = parser.current().span;
    parser.advance(); // consume _Alignof / __alignof__
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type-name vs expression
    let operand = if is_type_name_start(parser) {
        let type_name = parse_type_name(parser)?;
        AlignofOperand::TypeName(Box::new(type_name))
    } else {
        let expr = super::expressions::parse_expression(parser)?;
        AlignofOperand::Expression(Box::new(expr))
    };

    let end = parser.expect(TokenKind::RightParen)?;
    let span = Span::merge(start, end);

    Ok(Expression::Alignof { operand, span })
}

// ===========================================================================
// parse_alignas — _Alignas specifier
// ===========================================================================

/// Parses an `_Alignas` alignment specifier (C11 §6.7.5).
///
/// Disambiguates between `_Alignas(type-name)` and
/// `_Alignas(constant-expression)` by checking if the token after `(`
/// can begin a type name.
///
/// # Grammar
///
/// ```text
/// alignment-specifier:
///     _Alignas ( type-name )
///     _Alignas ( constant-expression )
/// ```
///
/// # Examples
///
/// ```c
/// _Alignas(16) int x;           // expression form
/// _Alignas(double) int y;       // type-name form
/// _Alignas(struct cacheline) z; // type-name form
/// ```
pub fn parse_alignas(parser: &mut Parser<'_>) -> ParseResult<AlignasSpecifier> {
    parser.advance(); // consume _Alignas
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type-name vs constant-expression
    let spec = if is_type_name_start(parser) {
        let type_name = parse_type_name(parser)?;
        AlignasSpecifier::TypeName(Box::new(type_name))
    } else {
        let expr = super::expressions::parse_constant_expression(parser)?;
        AlignasSpecifier::Expression(Box::new(expr))
    };

    parser.expect(TokenKind::RightParen)?;
    Ok(spec)
}

// ===========================================================================
// combine_type_specifiers — multi-keyword resolution
// ===========================================================================

/// Combines a list of individual type specifier keywords into a single
/// canonical [`CombinedType`].
///
/// C allows type specifiers to appear in any order and allows redundant
/// keywords, so `unsigned long long int` and `long unsigned int long` are
/// both valid representations of `unsigned long long`. This function
/// normalises such combinations.
///
/// # Valid Combinations (exhaustive)
///
/// | Specifiers                         | Result                |
/// |------------------------------------|-----------------------|
/// | `void`                             | `Void`                |
/// | `_Bool`                            | `Bool`                |
/// | `char`, `signed char`              | `SignedChar`           |
/// | `unsigned char`                    | `UnsignedChar`         |
/// | `short` `[int]` `[signed]`         | `SignedShort`          |
/// | `unsigned short` `[int]`           | `UnsignedShort`        |
/// | `int` `[signed]` (or bare `signed`)| `SignedInt`            |
/// | `unsigned` `[int]`                 | `UnsignedInt`          |
/// | `long` `[int]` `[signed]`          | `SignedLong`           |
/// | `unsigned long` `[int]`            | `UnsignedLong`         |
/// | `long long` `[int]` `[signed]`     | `SignedLongLong`       |
/// | `unsigned long long` `[int]`       | `UnsignedLongLong`     |
/// | `float`                            | `Float`                |
/// | `double`                           | `Double`               |
/// | `long double`                      | `LongDouble`           |
/// | `_Complex float`                   | `ComplexFloat`         |
/// | `_Complex double`                  | `ComplexDouble`        |
/// | `_Complex long double`             | `ComplexLongDouble`    |
/// | `struct ...`                       | `Struct`               |
/// | `union ...`                        | `Union`                |
/// | `enum ...`                         | `Enum`                 |
/// | typedef-name                       | `TypedefName`          |
/// | `typeof(...)`                      | `Typeof`               |
///
/// # Errors
///
/// Returns `ParseError` for invalid combinations such as `void int`,
/// `float char`, `unsigned void`, or duplicate incompatible specifiers.
pub fn combine_type_specifiers(specifiers: &[TypeSpecifier]) -> ParseResult<CombinedType> {
    // Empty specifier list — defaults to int (with a warning in real usage)
    if specifiers.is_empty() {
        return Ok(CombinedType::SignedInt);
    }

    // Fast path: single non-keyword specifiers (struct, union, enum, typedef, typeof)
    if specifiers.len() == 1 {
        match &specifiers[0] {
            TypeSpecifier::Struct { .. } => return Ok(CombinedType::Struct),
            TypeSpecifier::Union { .. } => return Ok(CombinedType::Union),
            TypeSpecifier::Enum { .. } => return Ok(CombinedType::Enum),
            TypeSpecifier::TypedefName { .. } => return Ok(CombinedType::TypedefName),
            TypeSpecifier::Typeof { .. } => return Ok(CombinedType::Typeof),
            TypeSpecifier::Atomic(..) => {
                // _Atomic(type-name) — the inner type is already resolved;
                // treat it as the type it wraps. For CombinedType purposes
                // we cannot further decompose, so return the inner type's
                // combination would be needed. For now, treat as a distinct
                // specifier that should be handled by the caller.
                return Ok(CombinedType::SignedInt);
            }
            _ => { /* fall through to bit-flag combination */ }
        }
    }

    // Build bit flags from the specifier list
    let mut flags: u32 = 0;
    let mut long_count: u32 = 0;

    for spec in specifiers {
        match spec {
            TypeSpecifier::Void => {
                if flags & SPEC_VOID != 0 {
                    return make_combine_error("duplicate 'void' specifier");
                }
                flags |= SPEC_VOID;
            }
            TypeSpecifier::Char => {
                if flags & SPEC_CHAR != 0 {
                    return make_combine_error("duplicate 'char' specifier");
                }
                flags |= SPEC_CHAR;
            }
            TypeSpecifier::Short => {
                if flags & SPEC_SHORT != 0 {
                    return make_combine_error("duplicate 'short' specifier");
                }
                flags |= SPEC_SHORT;
            }
            TypeSpecifier::Int => {
                if flags & SPEC_INT != 0 {
                    return make_combine_error("duplicate 'int' specifier");
                }
                flags |= SPEC_INT;
            }
            TypeSpecifier::Long => {
                long_count += 1;
                if long_count == 1 {
                    flags |= SPEC_LONG;
                } else if long_count == 2 {
                    flags |= SPEC_LONG_LONG;
                } else {
                    return make_combine_error("too many 'long' specifiers");
                }
            }
            TypeSpecifier::Float => {
                if flags & SPEC_FLOAT != 0 {
                    return make_combine_error("duplicate 'float' specifier");
                }
                flags |= SPEC_FLOAT;
            }
            TypeSpecifier::Double => {
                if flags & SPEC_DOUBLE != 0 {
                    return make_combine_error("duplicate 'double' specifier");
                }
                flags |= SPEC_DOUBLE;
            }
            TypeSpecifier::Signed => {
                if flags & SPEC_SIGNED != 0 {
                    return make_combine_error("duplicate 'signed' specifier");
                }
                flags |= SPEC_SIGNED;
            }
            TypeSpecifier::Unsigned => {
                if flags & SPEC_UNSIGNED != 0 {
                    return make_combine_error("duplicate 'unsigned' specifier");
                }
                flags |= SPEC_UNSIGNED;
            }
            TypeSpecifier::Bool => {
                if flags & SPEC_BOOL != 0 {
                    return make_combine_error("duplicate '_Bool' specifier");
                }
                flags |= SPEC_BOOL;
            }
            TypeSpecifier::Complex => {
                if flags & SPEC_COMPLEX != 0 {
                    return make_combine_error("duplicate '_Complex' specifier");
                }
                flags |= SPEC_COMPLEX;
            }
            TypeSpecifier::Struct { .. } => {
                flags |= SPEC_STRUCT;
            }
            TypeSpecifier::Union { .. } => {
                flags |= SPEC_UNION;
            }
            TypeSpecifier::Enum { .. } => {
                flags |= SPEC_ENUM;
            }
            TypeSpecifier::TypedefName { .. } => {
                flags |= SPEC_TYPEDEF_NAME;
            }
            TypeSpecifier::Typeof { .. } => {
                flags |= SPEC_TYPEOF;
            }
            TypeSpecifier::Atomic(..) => {
                // _Atomic(type-name) in a multi-specifier context is unusual;
                // treat as if we're combining with the wrapped type
            }
        }
    }

    // Validate: signed and unsigned are mutually exclusive
    if flags & SPEC_SIGNED != 0 && flags & SPEC_UNSIGNED != 0 {
        return make_combine_error("'signed' and 'unsigned' cannot be combined");
    }

    // Composite type specifiers cannot combine with keywords
    let keyword_flags = SPEC_VOID
        | SPEC_CHAR
        | SPEC_SHORT
        | SPEC_INT
        | SPEC_LONG
        | SPEC_LONG_LONG
        | SPEC_FLOAT
        | SPEC_DOUBLE
        | SPEC_SIGNED
        | SPEC_UNSIGNED
        | SPEC_BOOL
        | SPEC_COMPLEX;
    let composite_flags = SPEC_STRUCT | SPEC_UNION | SPEC_ENUM | SPEC_TYPEDEF_NAME | SPEC_TYPEOF;

    if flags & keyword_flags != 0 && flags & composite_flags != 0 {
        return make_combine_error(
            "cannot combine keyword type specifiers with struct/union/enum/typedef/typeof",
        );
    }

    // Composite types — return directly
    if flags & SPEC_STRUCT != 0 {
        return Ok(CombinedType::Struct);
    }
    if flags & SPEC_UNION != 0 {
        return Ok(CombinedType::Union);
    }
    if flags & SPEC_ENUM != 0 {
        return Ok(CombinedType::Enum);
    }
    if flags & SPEC_TYPEDEF_NAME != 0 {
        return Ok(CombinedType::TypedefName);
    }
    if flags & SPEC_TYPEOF != 0 {
        return Ok(CombinedType::Typeof);
    }

    // --- Resolve keyword combinations ---

    // void — cannot combine with anything except _Complex (which is unusual)
    if flags & SPEC_VOID != 0 {
        let other = flags & !SPEC_VOID;
        if other != 0 {
            return make_combine_error("'void' cannot be combined with other type specifiers");
        }
        return Ok(CombinedType::Void);
    }

    // _Bool — cannot combine with signed/unsigned/short/long/int
    if flags & SPEC_BOOL != 0 {
        let other = flags & !SPEC_BOOL;
        if other != 0 {
            return make_combine_error("'_Bool' cannot be combined with other type specifiers");
        }
        return Ok(CombinedType::Bool);
    }

    // _Complex — must combine with float, double, or long double
    if flags & SPEC_COMPLEX != 0 {
        let base = flags & !SPEC_COMPLEX;
        return resolve_complex_type(base);
    }

    // char — can combine only with signed/unsigned
    if flags & SPEC_CHAR != 0 {
        let other = flags & !(SPEC_CHAR | SPEC_SIGNED | SPEC_UNSIGNED);
        if other != 0 {
            return make_combine_error("'char' can only be combined with 'signed' or 'unsigned'");
        }
        if flags & SPEC_UNSIGNED != 0 {
            return Ok(CombinedType::UnsignedChar);
        }
        return Ok(CombinedType::SignedChar);
    }

    // short — can combine with signed/unsigned and int
    if flags & SPEC_SHORT != 0 {
        let other = flags & !(SPEC_SHORT | SPEC_INT | SPEC_SIGNED | SPEC_UNSIGNED);
        if other != 0 {
            return make_combine_error(
                "'short' can only be combined with 'signed', 'unsigned', or 'int'",
            );
        }
        if flags & SPEC_UNSIGNED != 0 {
            return Ok(CombinedType::UnsignedShort);
        }
        return Ok(CombinedType::SignedShort);
    }

    // long long — can combine with signed/unsigned and int
    if flags & SPEC_LONG_LONG != 0 {
        let other = flags & !(SPEC_LONG | SPEC_LONG_LONG | SPEC_INT | SPEC_SIGNED | SPEC_UNSIGNED);
        if other != 0 {
            return make_combine_error(
                "'long long' can only be combined with 'signed', 'unsigned', or 'int'",
            );
        }
        if flags & SPEC_UNSIGNED != 0 {
            return Ok(CombinedType::UnsignedLongLong);
        }
        return Ok(CombinedType::SignedLongLong);
    }

    // long — can combine with signed/unsigned, int, or double
    if flags & SPEC_LONG != 0 {
        // long double
        if flags & SPEC_DOUBLE != 0 {
            let other = flags & !(SPEC_LONG | SPEC_DOUBLE);
            if other != 0 {
                return make_combine_error(
                    "'long double' cannot be combined with other specifiers",
                );
            }
            return Ok(CombinedType::LongDouble);
        }

        let other = flags & !(SPEC_LONG | SPEC_INT | SPEC_SIGNED | SPEC_UNSIGNED);
        if other != 0 {
            return make_combine_error(
                "'long' can only be combined with 'signed', 'unsigned', 'int', or 'double'",
            );
        }
        if flags & SPEC_UNSIGNED != 0 {
            return Ok(CombinedType::UnsignedLong);
        }
        return Ok(CombinedType::SignedLong);
    }

    // float — standalone only
    if flags & SPEC_FLOAT != 0 {
        let other = flags & !SPEC_FLOAT;
        if other != 0 {
            return make_combine_error("'float' cannot be combined with other type specifiers");
        }
        return Ok(CombinedType::Float);
    }

    // double — standalone only (long double handled above)
    if flags & SPEC_DOUBLE != 0 {
        let other = flags & !SPEC_DOUBLE;
        if other != 0 {
            return make_combine_error("'double' cannot be combined with other type specifiers");
        }
        return Ok(CombinedType::Double);
    }

    // int — with optional signed/unsigned
    if flags & SPEC_INT != 0 {
        let other = flags & !(SPEC_INT | SPEC_SIGNED | SPEC_UNSIGNED);
        if other != 0 {
            return make_combine_error("'int' can only be combined with 'signed' or 'unsigned'");
        }
        if flags & SPEC_UNSIGNED != 0 {
            return Ok(CombinedType::UnsignedInt);
        }
        return Ok(CombinedType::SignedInt);
    }

    // bare signed/unsigned — equivalent to signed int / unsigned int
    if flags & SPEC_SIGNED != 0 {
        return Ok(CombinedType::SignedInt);
    }
    if flags & SPEC_UNSIGNED != 0 {
        return Ok(CombinedType::UnsignedInt);
    }

    // Should not reach here if specifiers were non-empty
    make_combine_error("unable to resolve type specifier combination")
}

/// Resolves `_Complex` with the base type flags to a `CombinedType`.
fn resolve_complex_type(base_flags: u32) -> ParseResult<CombinedType> {
    // _Complex float
    if base_flags == SPEC_FLOAT {
        return Ok(CombinedType::ComplexFloat);
    }
    // _Complex double
    if base_flags == SPEC_DOUBLE {
        return Ok(CombinedType::ComplexDouble);
    }
    // _Complex long double
    if base_flags == SPEC_LONG | SPEC_DOUBLE {
        return Ok(CombinedType::ComplexLongDouble);
    }
    // Bare _Complex defaults to _Complex double (GCC extension)
    if base_flags == 0 {
        return Ok(CombinedType::ComplexDouble);
    }
    make_combine_error("'_Complex' can only be combined with 'float', 'double', or 'long double'")
}

/// Creates a `ParseError` for invalid type specifier combinations.
fn make_combine_error(msg: &str) -> ParseResult<CombinedType> {
    Err(ParseError {
        span: Span::DUMMY,
        message: msg.to_string(),
        expected: Some("valid type specifier combination".to_string()),
    })
}

// ===========================================================================
// parse_type_name — specifier-qualifier-list + optional abstract declarator
// ===========================================================================

/// Parses a type name: a specifier-qualifier list followed by an optional
/// abstract declarator.
///
/// Type names appear in casts `(type-name)expr`, `sizeof(type-name)`,
/// `_Alignof(type-name)`, `_Atomic(type-name)`, `typeof(type-name)`,
/// `_Generic(expr, type-name: expr, ...)`, and compound literals
/// `(type-name){init-list}`.
///
/// # Grammar
///
/// ```text
/// type-name:
///     specifier-qualifier-list abstract-declarator(opt)
/// ```
///
/// # Examples
///
/// ```c
/// int                  // simple type
/// const int *          // pointer to const int
/// void (*)(int, int)   // pointer to function returning void
/// ```
pub fn parse_type_name(parser: &mut Parser<'_>) -> ParseResult<TypeName> {
    let start = parser.current().span;
    let specifiers = parse_specifier_qualifier_list(parser)?;

    // Check for optional abstract declarator
    let declarator = if can_start_abstract_declarator(parser) {
        Some(parse_abstract_declarator(parser)?)
    } else {
        None
    };

    let end = declarator
        .as_ref()
        .map(|d| d.span)
        .unwrap_or(specifiers.span);
    let span = Span::merge(start, end);

    Ok(TypeName {
        specifiers,
        declarator,
        span,
    })
}

// ===========================================================================
// parse_abstract_declarator — declarator without a name
// ===========================================================================

/// Parses an abstract declarator: pointer derivations followed by optional
/// direct abstract declarator (arrays, function parameter lists, grouping).
///
/// An abstract declarator is a declarator without a name — used in type
/// names for casts, sizeof, function parameter types, etc.
///
/// # Grammar
///
/// ```text
/// abstract-declarator:
///     pointer
///     pointer(opt) direct-abstract-declarator
///
/// direct-abstract-declarator:
///     ( abstract-declarator )
///     direct-abstract-declarator(opt) [ assignment-expression(opt) ]
///     direct-abstract-declarator(opt) ( parameter-type-list(opt) )
/// ```
pub fn parse_abstract_declarator(parser: &mut Parser<'_>) -> ParseResult<AbstractDeclarator> {
    let start = parser.current().span;
    let mut derived: Vec<DerivedDeclarator> = Vec::new();

    // Pointer derivations: `* [qualifiers]` prefix chain
    while parser.check(TokenKind::Star) {
        parser.advance(); // consume `*`
        let quals = parse_type_qualifiers(parser);
        derived.push(DerivedDeclarator::Pointer { qualifiers: quals });
    }

    // Direct abstract declarator (optional)
    let end_span = parse_direct_abstract_declarator(parser, &mut derived)?;
    let final_span = end_span.unwrap_or(start);

    Ok(AbstractDeclarator {
        derived,
        span: Span::merge(start, final_span),
    })
}

/// Parse the direct part of an abstract declarator.
///
/// Handles grouped declarators `(abstract-declarator)`, array declarators
/// `[size_opt]`, and function declarators `(parameter-type-list_opt)`.
///
/// Returns the span of the last parsed component, or `None` if nothing
/// was parsed.
fn parse_direct_abstract_declarator(
    parser: &mut Parser<'_>,
    derived: &mut Vec<DerivedDeclarator>,
) -> ParseResult<Option<Span>> {
    let mut last_span: Option<Span> = None;

    // Check for grouped abstract declarator: ( abstract-declarator )
    //
    // Disambiguation: if the token after '(' is '*', '[', or '(' then
    // it is a grouped abstract declarator (e.g., `(*)`, `(*)(int)`,
    // `([10])`, `((*))`). Otherwise it is a function parameter list.
    if parser.check(TokenKind::LeftParen) {
        let next_kind = &parser.peek_ahead(1).kind;
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

    // Parse postfix declarators: [] and ()
    loop {
        match parser.current().kind {
            // Array declarator: [ size_opt ]
            TokenKind::LeftBracket => {
                let arr = parse_array_abstract_declarator(parser)?;
                derived.push(arr);
                last_span = Some(parser.current().span);
            }
            // Function declarator: ( parameter-type-list_opt )
            // In postfix position, '(' always starts a function parameter list.
            TokenKind::LeftParen => {
                let func = super::declarations::parse_function_declarator(parser)?;
                derived.push(func);
                last_span = Some(parser.current().span);
            }
            _ => break,
        }
    }

    Ok(last_span)
}

/// Parse an array declarator `[ static? qualifiers? size? ]` within an
/// abstract declarator context.
///
/// Handles:
/// - `[]` — incomplete array
/// - `[N]` — fixed-size array
/// - `[*]` — VLA with unspecified size
/// - `[static N]` or `[static qualifiers N]` — C11 parameter array notation
fn parse_array_abstract_declarator(parser: &mut Parser<'_>) -> ParseResult<DerivedDeclarator> {
    parser.expect(TokenKind::LeftBracket)?;

    let mut is_static = false;

    // Check for `static` before qualifiers
    if parser.check(TokenKind::Static) {
        is_static = true;
        parser.advance();
    }

    // Parse optional type qualifiers inside brackets
    let qualifiers = parse_type_qualifiers(parser);

    // Check for `static` after qualifiers (if not already seen)
    if !is_static && parser.check(TokenKind::Static) {
        is_static = true;
        parser.advance();
    }

    // Parse size expression (or `*` for VLA)
    let size = if parser.check(TokenKind::RightBracket) {
        None
    } else if parser.check(TokenKind::Star) && parser.peek_ahead(1).kind == TokenKind::RightBracket
    {
        // [*] — VLA with unspecified size
        parser.advance();
        None
    } else {
        Some(Box::new(super::expressions::parse_assignment_expression(
            parser,
        )?))
    };

    parser.expect(TokenKind::RightBracket)?;

    Ok(DerivedDeclarator::Array {
        size,
        is_static,
        qualifiers,
    })
}

// ===========================================================================
// Conversion and utility helpers
// ===========================================================================

/// Converts a [`SpecifierQualifierList`] into a full [`DeclarationSpecifiers`]
/// with default storage class, function specifiers, alignment, and attributes.
///
/// This is useful when a specifier-qualifier context needs to produce a full
/// declaration specifier set (e.g., for struct field declarations or compound
/// literal types). The caller can then modify individual fields as needed.
///
/// # Arguments
///
/// * `sql` — The specifier-qualifier list to convert.
/// * `has_extension` — Whether `__extension__` was present.
///
/// # Returns
///
/// A [`DeclarationSpecifiers`] with the type specifiers and qualifiers from
/// the input, and all other fields set to their defaults.
pub fn spec_qual_to_declaration_specifiers(
    sql: SpecifierQualifierList,
    has_extension: bool,
) -> DeclarationSpecifiers {
    DeclarationSpecifiers {
        storage_class: None,
        type_specifiers: sql.specifiers,
        type_qualifiers: sql.qualifiers,
        function_specifiers: FunctionSpecifiers::default(),
        alignment: None,
        attrs: Vec::<Attribute>::new(),
        has_extension,
        span: sql.span,
    }
}

/// Checks whether a [`StorageClass`] variant is valid in a specifier-qualifier
/// context.
///
/// In C11, specifier-qualifier lists (used in type names, struct declarations,
/// etc.) do **not** allow storage-class specifiers. This function returns
/// `false` for all storage classes, but is provided as a utility for callers
/// that encounter storage-class tokens in a specifier-qualifier context and
/// need to emit diagnostics.
pub fn is_storage_class_in_spec_qual_context(sc: StorageClass) -> bool {
    match sc {
        StorageClass::Auto
        | StorageClass::Register
        | StorageClass::Static
        | StorageClass::Extern
        | StorageClass::Typedef
        | StorageClass::ThreadLocal => false,
    }
}

// ===========================================================================
// Internal helpers
// ===========================================================================

/// Parses `_Atomic ( type-name )` — the type specifier form of `_Atomic`.
///
/// Called when the parser sees `_Atomic` followed by `(`. Consumes the
/// `_Atomic`, `(`, type-name, and `)` tokens.
fn parse_atomic_type_specifier(parser: &mut Parser<'_>) -> ParseResult<TypeSpecifier> {
    let _start = parser.current().span;
    parser.advance(); // consume _Atomic
    parser.expect(TokenKind::LeftParen)?;
    let inner_type = parse_type_name(parser)?;
    parser.expect(TokenKind::RightParen)?;
    Ok(TypeSpecifier::Atomic(Box::new(inner_type)))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `is_type_specifier_token` correctly identifies all type
    /// specifier tokens and rejects non-specifier tokens.
    #[test]
    fn test_is_type_specifier_token() {
        // Type specifier keywords — should all return true
        assert!(is_type_specifier_token(&TokenKind::Void));
        assert!(is_type_specifier_token(&TokenKind::Char));
        assert!(is_type_specifier_token(&TokenKind::Short));
        assert!(is_type_specifier_token(&TokenKind::Int));
        assert!(is_type_specifier_token(&TokenKind::Long));
        assert!(is_type_specifier_token(&TokenKind::Float));
        assert!(is_type_specifier_token(&TokenKind::Double));
        assert!(is_type_specifier_token(&TokenKind::Signed));
        assert!(is_type_specifier_token(&TokenKind::Unsigned));
        assert!(is_type_specifier_token(&TokenKind::Bool));
        assert!(is_type_specifier_token(&TokenKind::Complex));
        assert!(is_type_specifier_token(&TokenKind::Atomic));
        assert!(is_type_specifier_token(&TokenKind::Struct));
        assert!(is_type_specifier_token(&TokenKind::Union));
        assert!(is_type_specifier_token(&TokenKind::Enum));
        assert!(is_type_specifier_token(&TokenKind::TypeofKeyword));
        assert!(is_type_specifier_token(&TokenKind::SignedGcc));
        assert!(is_type_specifier_token(&TokenKind::Extension));

        // Non-specifier tokens — should all return false
        assert!(!is_type_specifier_token(&TokenKind::LeftParen));
        assert!(!is_type_specifier_token(&TokenKind::RightParen));
        assert!(!is_type_specifier_token(&TokenKind::Const));
        assert!(!is_type_specifier_token(&TokenKind::Volatile));
    }

    /// Verify the `CombinedType` enum has all expected variants and they
    /// are distinct via equality comparison.
    #[test]
    fn test_combined_type_variants() {
        assert_ne!(CombinedType::Void, CombinedType::Bool);
        assert_ne!(CombinedType::SignedChar, CombinedType::UnsignedChar);
        assert_ne!(CombinedType::SignedInt, CombinedType::UnsignedInt);
        assert_ne!(CombinedType::SignedLong, CombinedType::SignedLongLong);
        assert_ne!(CombinedType::Float, CombinedType::Double);
        assert_ne!(CombinedType::LongDouble, CombinedType::Double);
        assert_ne!(CombinedType::ComplexFloat, CombinedType::ComplexDouble);
        assert_ne!(CombinedType::Struct, CombinedType::Union);
        assert_ne!(CombinedType::Enum, CombinedType::TypedefName);
        assert_eq!(CombinedType::Typeof, CombinedType::Typeof);
    }

    /// Verify `combine_type_specifiers` for simple single-keyword types.
    #[test]
    fn test_combine_single_specifiers() {
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Void]).unwrap(),
            CombinedType::Void
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Bool]).unwrap(),
            CombinedType::Bool
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Float]).unwrap(),
            CombinedType::Float
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Double]).unwrap(),
            CombinedType::Double
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Int]).unwrap(),
            CombinedType::SignedInt
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Char]).unwrap(),
            CombinedType::SignedChar
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Signed]).unwrap(),
            CombinedType::SignedInt
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Unsigned]).unwrap(),
            CombinedType::UnsignedInt
        );
    }

    /// Verify `combine_type_specifiers` for multi-keyword types.
    #[test]
    fn test_combine_multi_specifiers() {
        // unsigned long long int
        assert_eq!(
            combine_type_specifiers(&[
                TypeSpecifier::Unsigned,
                TypeSpecifier::Long,
                TypeSpecifier::Long,
                TypeSpecifier::Int,
            ])
            .unwrap(),
            CombinedType::UnsignedLongLong
        );

        // signed char
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Signed, TypeSpecifier::Char]).unwrap(),
            CombinedType::SignedChar
        );

        // unsigned char
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Unsigned, TypeSpecifier::Char]).unwrap(),
            CombinedType::UnsignedChar
        );

        // long double
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Long, TypeSpecifier::Double]).unwrap(),
            CombinedType::LongDouble
        );

        // unsigned short int
        assert_eq!(
            combine_type_specifiers(&[
                TypeSpecifier::Unsigned,
                TypeSpecifier::Short,
                TypeSpecifier::Int,
            ])
            .unwrap(),
            CombinedType::UnsignedShort
        );

        // signed long int
        assert_eq!(
            combine_type_specifiers(&[
                TypeSpecifier::Signed,
                TypeSpecifier::Long,
                TypeSpecifier::Int,
            ])
            .unwrap(),
            CombinedType::SignedLong
        );
    }

    /// Verify `combine_type_specifiers` for _Complex combinations.
    #[test]
    fn test_combine_complex_specifiers() {
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Complex, TypeSpecifier::Float]).unwrap(),
            CombinedType::ComplexFloat
        );
        assert_eq!(
            combine_type_specifiers(&[TypeSpecifier::Complex, TypeSpecifier::Double]).unwrap(),
            CombinedType::ComplexDouble
        );
        assert_eq!(
            combine_type_specifiers(&[
                TypeSpecifier::Complex,
                TypeSpecifier::Long,
                TypeSpecifier::Double,
            ])
            .unwrap(),
            CombinedType::ComplexLongDouble
        );
    }

    /// Verify that invalid combinations produce errors.
    #[test]
    fn test_combine_invalid_specifiers() {
        // void int
        assert!(combine_type_specifiers(&[TypeSpecifier::Void, TypeSpecifier::Int]).is_err());

        // signed unsigned
        assert!(
            combine_type_specifiers(&[TypeSpecifier::Signed, TypeSpecifier::Unsigned]).is_err()
        );

        // _Bool int
        assert!(combine_type_specifiers(&[TypeSpecifier::Bool, TypeSpecifier::Int]).is_err());

        // float char
        assert!(combine_type_specifiers(&[TypeSpecifier::Float, TypeSpecifier::Char]).is_err());

        // three longs
        assert!(combine_type_specifiers(&[
            TypeSpecifier::Long,
            TypeSpecifier::Long,
            TypeSpecifier::Long,
        ])
        .is_err());
    }
}
