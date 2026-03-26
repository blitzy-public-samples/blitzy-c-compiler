#![allow(clippy::result_unit_err)]
//! Type specifier and qualifier parsing for the BCC C11 parser (Phase 4).
//!
//! This module handles all C11 type keywords plus GCC extensions:
//! - Simple type specifiers: `void`, `char`, `short`, `int`, `long`, `float`,
//!   `double`, `signed`, `unsigned`, `_Bool`, `_Complex`
//! - Elaborated type specifiers: `struct tag`, `union tag`, `enum tag`
//! - Typedef names: identifiers previously declared via `typedef`
//! - GCC `typeof`/`__typeof__`: type deduction from expressions or type names
//! - `_Atomic(type-name)`: atomic type specifier form
//! - `_Alignas(type)` / `_Alignas(N)`: alignment specifiers
//! - `__extension__`: GCC pedantic-warning suppression
//! - Transparent union recognition via `__attribute__((transparent_union))`
//!
//! # Specifier Combination Validation
//!
//! Multiple type specifiers combine per C11 §6.7.2 (e.g., `unsigned long long`).
//! Invalid combinations (e.g., `short long`, `double int`) are diagnosed with
//! clear error messages.
//!
//! # Dependencies
//!
//! - `super::ast::*` — AST node types
//! - `super::Parser` — token consumption and error reporting
//! - `super::attributes` — `__attribute__((...))` parsing
//! - `crate::frontend::lexer::token::TokenKind` — token type matching
//! - `crate::common::diagnostics::Span` — source location spans
//! - `crate::common::string_interner::Symbol` — interned identifiers
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates. Does NOT depend on `crate::ir`, `crate::passes`,
//! or `crate::backend`.

use super::ast::*;
use super::attributes;
use super::declarations;
use super::expressions;
use super::Parser;
use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::TokenKind;

// ===========================================================================
// Exported Types
// ===========================================================================

/// A parsed sequence of type specifiers with their combined source span.
///
/// Produced by [`parse_type_specifiers`]. Contains the raw list of specifier
/// AST nodes as they appeared in source (e.g., `unsigned long long int`
/// produces four entries: `Unsigned`, `Long`, `Long`, `Int`).
///
/// Combination validation is performed during parsing — invalid combinations
/// are diagnosed and the list may contain a fallback specifier (`Int`) for
/// error recovery.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeSpecifierList {
    /// The individual type specifiers in source order.
    pub specifiers: Vec<TypeSpecifier>,
    /// Source span covering all specifiers.
    pub span: Span,
}

/// A parsed set of type qualifiers with boolean flags and source span.
///
/// Produced by [`parse_type_qualifiers`]. Qualifiers may be specified
/// redundantly (e.g., `const const`); only the first occurrence sets the
/// flag — duplicates are valid per C11 but ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeQualifiers {
    /// `const` — object is read-only.
    pub is_const: bool,
    /// `volatile` — accesses are observable side effects.
    pub is_volatile: bool,
    /// `restrict` — pointer has no aliases (C99+).
    pub is_restrict: bool,
    /// `_Atomic` — atomic qualifier form (without parenthesized type argument).
    pub is_atomic: bool,
    /// Source span covering all qualifiers.
    pub span: Span,
}

impl Default for TypeQualifiers {
    fn default() -> Self {
        TypeQualifiers {
            is_const: false,
            is_volatile: false,
            is_restrict: false,
            is_atomic: false,
            span: Span::dummy(),
        }
    }
}

impl TypeQualifiers {
    /// Returns `true` if no qualifiers are set.
    pub fn is_empty(&self) -> bool {
        !self.is_const && !self.is_volatile && !self.is_restrict && !self.is_atomic
    }

    /// Convert this qualifier set to a list of [`TypeQualifier`] AST nodes.
    pub fn to_qualifier_list(&self) -> Vec<TypeQualifier> {
        let mut list = Vec::new();
        if self.is_const {
            list.push(TypeQualifier::Const);
        }
        if self.is_volatile {
            list.push(TypeQualifier::Volatile);
        }
        if self.is_restrict {
            list.push(TypeQualifier::Restrict);
        }
        if self.is_atomic {
            list.push(TypeQualifier::Atomic);
        }
        list
    }
}

// ===========================================================================
// Specifier Tracking Flags
// ===========================================================================

/// Bit flags to track which type specifiers have been seen during parsing.
/// Used for validating specifier combinations per C11 §6.7.2.
struct SpecifierFlags {
    has_void: bool,
    has_char: bool,
    has_short: bool,
    has_int: bool,
    long_count: u8, // 0, 1, or 2 (for `long long`)
    has_float: bool,
    has_double: bool,
    has_signed: bool,
    has_unsigned: bool,
    has_bool: bool,
    has_complex: bool,
    has_int128: bool,
    has_struct_union_enum: bool,
    has_typedef_name: bool,
    has_atomic: bool,
    has_typeof: bool,
}

impl SpecifierFlags {
    fn new() -> Self {
        SpecifierFlags {
            has_void: false,
            has_char: false,
            has_short: false,
            has_int: false,
            long_count: 0,
            has_float: false,
            has_double: false,
            has_signed: false,
            has_unsigned: false,
            has_bool: false,
            has_complex: false,
            has_int128: false,
            has_struct_union_enum: false,
            has_typedef_name: false,
            has_atomic: false,
            has_typeof: false,
        }
    }

    /// Returns true if any "primary" type specifier has been seen
    /// (one that cannot combine with typedef-name or another primary).
    fn has_any_primary(&self) -> bool {
        self.has_void
            || self.has_char
            || self.has_short
            || self.has_int
            || self.long_count > 0
            || self.has_float
            || self.has_double
            || self.has_signed
            || self.has_unsigned
            || self.has_bool
            || self.has_complex
            || self.has_int128
            || self.has_struct_union_enum
            || self.has_typedef_name
            || self.has_atomic
            || self.has_typeof
    }
}

// ===========================================================================
// Public API — Type Specifier Parsing
// ===========================================================================

/// Parse a sequence of type specifiers, accumulating them into a
/// [`TypeSpecifierList`].
///
/// Handles:
/// - Simple type specifiers: `void`, `char`, `short`, `int`, `long`, `float`,
///   `double`, `signed`, `unsigned`, `_Bool`, `_Complex`
/// - Elaborated type specifiers: `struct tag`, `union tag`, `enum tag`
/// - Typedef names: identifiers declared as typedefs
/// - GCC `typeof(expr)` / `typeof(type-name)` / `__typeof__(...)`
/// - `_Atomic(type-name)`: atomic type specifier (with parens)
/// - `__extension__`: consumed and ignored (suppresses warnings)
///
/// Validates specifier combinations per C11 §6.7.2.
///
/// # Errors
///
/// Returns `Err(())` on unrecoverable parse errors. Invalid specifier
/// combinations are diagnosed and the parser attempts error recovery by
/// producing a default `Int` type.
pub fn parse_type_specifiers(parser: &mut Parser<'_>) -> Result<TypeSpecifierList, ()> {
    let start_span = parser.current_span();
    let mut specifiers = Vec::new();
    let mut flags = SpecifierFlags::new();

    loop {
        // Skip __extension__ tokens in type context.
        if parser.check(&TokenKind::Extension) {
            parser.advance();
            continue;
        }

        match parser.peek() {
            TokenKind::Void => {
                if let Err(msg) = check_add_void(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_void = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                if let Err(msg) = check_add_char(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_char = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                if let Err(msg) = check_add_short(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_short = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                if let Err(msg) = check_add_int(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_int = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                if let Err(msg) = check_add_long(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.long_count += 1;
                parser.advance();
                specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                if let Err(msg) = check_add_float(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_float = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                if let Err(msg) = check_add_double(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_double = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                if let Err(msg) = check_add_signed(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_signed = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                if let Err(msg) = check_add_unsigned(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_unsigned = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                if let Err(msg) = check_add_bool(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_bool = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                if let Err(msg) = check_add_complex(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_complex = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Complex);
            }
            // __int128 — GCC 128-bit integer extension
            TokenKind::Int128Keyword => {
                if flags.has_int128 {
                    let span = parser.current_span();
                    parser.error(span, "duplicate __int128 specifier");
                }
                flags.has_int128 = true;
                parser.advance();
                specifiers.push(TypeSpecifier::Int128);
            }
            // _Float128 / __float128 — GCC/C23 128-bit float
            TokenKind::Float128Keyword => {
                parser.advance();
                specifiers.push(TypeSpecifier::Float128);
            }
            // _Float16 — IEC 60559 half precision
            TokenKind::Float16Keyword => {
                parser.advance();
                specifiers.push(TypeSpecifier::Float16);
            }
            // _Float32 / _Float32x — IEC 60559 single precision
            TokenKind::Float32Keyword => {
                parser.advance();
                specifiers.push(TypeSpecifier::Float32);
            }
            // _Float64 / _Float64x — IEC 60559 double precision
            TokenKind::Float64Keyword => {
                parser.advance();
                specifiers.push(TypeSpecifier::Float64);
            }
            // Struct/Union — elaborated type specifier
            TokenKind::Struct | TokenKind::Union => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(
                        span,
                        "cannot combine struct/union with other type specifiers",
                    );
                }
                flags.has_struct_union_enum = true;
                let spec = parse_struct_or_union(parser)?;
                specifiers.push(spec);
            }
            // Enum — elaborated type specifier
            TokenKind::Enum => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(span, "cannot combine enum with other type specifiers");
                }
                flags.has_struct_union_enum = true;
                let spec = parse_enum(parser)?;
                specifiers.push(spec);
            }
            // typeof / __typeof__
            TokenKind::Typeof => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(span, "cannot combine typeof with other type specifiers");
                }
                flags.has_typeof = true;
                let spec = parse_typeof(parser)?;
                specifiers.push(spec);
            }
            // __auto_type — GCC automatic type inference
            TokenKind::AutoType => {
                parser.advance(); // consume __auto_type
                specifiers.push(TypeSpecifier::AutoType);
            }
            // _Atomic — could be specifier `_Atomic(type)` or qualifier `_Atomic`
            TokenKind::Atomic => {
                // Disambiguate: `_Atomic(` is a type specifier,
                // `_Atomic` without `(` is a qualifier (handled elsewhere).
                // We need to peek ahead to see if `(` follows.
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    if flags.has_any_primary() {
                        let span = parser.current_span();
                        parser.error(
                            span,
                            "cannot combine _Atomic(type) with other type specifiers",
                        );
                    }
                    flags.has_atomic = true;
                    let spec = parse_atomic_type_specifier(parser)?;
                    specifiers.push(spec);
                } else {
                    // _Atomic as qualifier — don't consume, let qualifier parser handle it
                    break;
                }
            }
            // Identifier that might be a typedef name
            TokenKind::Identifier(sym) => {
                let sym_val = *sym;
                if parser.is_typedef_name(sym_val) && !flags.has_any_primary() {
                    flags.has_typedef_name = true;
                    parser.advance();
                    specifiers.push(TypeSpecifier::TypedefName(sym_val));
                } else {
                    break;
                }
            }
            _ => break,
        }
    }

    let span = if specifiers.is_empty() {
        start_span
    } else {
        parser.make_span(start_span)
    };

    Ok(TypeSpecifierList { specifiers, span })
}

// ===========================================================================
// Public API — Type Qualifier Parsing
// ===========================================================================

/// Parse a sequence of type qualifiers: `const`, `volatile`, `restrict`,
/// `_Atomic` (without parenthesized type argument).
///
/// Qualifiers can appear multiple times (redundant but valid per C11 §6.7.3).
/// Duplicate qualifiers are silently accepted (the standard says they
/// are permitted).
///
/// Also handles `__volatile__` as a synonym for `volatile`.
///
/// # Returns
///
/// A [`TypeQualifiers`] struct with boolean flags for each qualifier and
/// the combined source span.
pub fn parse_type_qualifiers(parser: &mut Parser<'_>) -> Result<TypeQualifiers, ()> {
    let start_span = parser.current_span();
    let mut quals = TypeQualifiers::default();
    let mut found_any = false;

    loop {
        match parser.peek() {
            TokenKind::Const => {
                parser.advance();
                quals.is_const = true;
                found_any = true;
            }
            TokenKind::Volatile | TokenKind::AsmVolatile => {
                // __volatile__ maps to volatile as a type qualifier
                parser.advance();
                quals.is_volatile = true;
                found_any = true;
            }
            TokenKind::Restrict => {
                parser.advance();
                quals.is_restrict = true;
                found_any = true;
            }
            TokenKind::Atomic => {
                // _Atomic without `(` is a qualifier.
                // _Atomic with `(` is a type specifier (handled by parse_type_specifiers).
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    break; // This is _Atomic(type) — a specifier, not a qualifier
                }
                parser.advance();
                quals.is_atomic = true;
                found_any = true;
            }
            _ => break,
        }
    }

    quals.span = if found_any {
        parser.make_span(start_span)
    } else {
        start_span
    };

    Ok(quals)
}

// ===========================================================================
// Public API — Specifier-Qualifier List
// ===========================================================================

/// Parse a specifier-qualifier-list: an interleaved sequence of type specifiers
/// and type qualifiers (no storage class specifiers).
///
/// Used for:
/// - Struct/union member declarations
/// - Type names in casts, `sizeof`, `_Alignof`, `_Generic`, compound literals
/// - `typeof(type-name)` arguments
///
/// Handles `__attribute__((...))` and `__extension__` tokens that may appear
/// within the specifier-qualifier list.
///
/// # Grammar
///
/// ```text
/// specifier-qualifier-list:
///     type-specifier specifier-qualifier-list_opt
///     type-qualifier specifier-qualifier-list_opt
///     alignment-specifier specifier-qualifier-list_opt  (C11)
/// ```
///
/// # Returns
///
/// A [`SpecifierQualifierList`] containing the combined specifiers, qualifiers,
/// and attributes with their source span.
pub fn parse_specifier_qualifier_list(
    parser: &mut Parser<'_>,
) -> Result<SpecifierQualifierList, ()> {
    let start_span = parser.current_span();
    let mut type_specifiers = Vec::new();
    let mut type_qualifiers = Vec::new();
    let mut attrs = Vec::new();
    let mut flags = SpecifierFlags::new();

    loop {
        // Skip __extension__ tokens.
        if parser.check(&TokenKind::Extension) {
            parser.advance();
            continue;
        }

        match parser.peek() {
            // ----- Type specifiers -----
            TokenKind::Void => {
                if let Err(msg) = check_add_void(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_void = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Void);
            }
            TokenKind::Char => {
                if let Err(msg) = check_add_char(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_char = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Char);
            }
            TokenKind::Short => {
                if let Err(msg) = check_add_short(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_short = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Short);
            }
            TokenKind::Int => {
                if let Err(msg) = check_add_int(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_int = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Int);
            }
            TokenKind::Long => {
                if let Err(msg) = check_add_long(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.long_count += 1;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Long);
            }
            TokenKind::Float => {
                if let Err(msg) = check_add_float(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_float = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Float);
            }
            TokenKind::Double => {
                if let Err(msg) = check_add_double(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_double = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Double);
            }
            TokenKind::Signed => {
                if let Err(msg) = check_add_signed(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_signed = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Signed);
            }
            TokenKind::Unsigned => {
                if let Err(msg) = check_add_unsigned(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_unsigned = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Unsigned);
            }
            TokenKind::Bool => {
                if let Err(msg) = check_add_bool(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_bool = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Bool);
            }
            TokenKind::Complex => {
                if let Err(msg) = check_add_complex(&flags) {
                    let span = parser.current_span();
                    parser.error(span, msg);
                }
                flags.has_complex = true;
                parser.advance();
                type_specifiers.push(TypeSpecifier::Complex);
            }

            // __int128 — GCC 128-bit integer extension
            TokenKind::Int128Keyword => {
                if flags.has_int128 {
                    let span = parser.current_span();
                    parser.error(span, "duplicate __int128 specifier");
                }
                flags.has_int128 = true;
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

            // ----- Type qualifiers -----
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

            // ----- _Atomic: specifier or qualifier -----
            TokenKind::Atomic => {
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    // _Atomic(type-name) — type specifier
                    if flags.has_any_primary() {
                        let span = parser.current_span();
                        parser.error(
                            span,
                            "cannot combine _Atomic(type) with other type specifiers",
                        );
                    }
                    flags.has_atomic = true;
                    let spec = parse_atomic_type_specifier(parser)?;
                    type_specifiers.push(spec);
                } else {
                    // _Atomic — type qualifier
                    parser.advance();
                    type_qualifiers.push(TypeQualifier::Atomic);
                }
            }

            // ----- Elaborated type specifiers -----
            TokenKind::Struct | TokenKind::Union => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(
                        span,
                        "cannot combine struct/union with other type specifiers",
                    );
                }
                flags.has_struct_union_enum = true;
                let spec = parse_struct_or_union(parser)?;
                type_specifiers.push(spec);
            }
            TokenKind::Enum => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(span, "cannot combine enum with other type specifiers");
                }
                flags.has_struct_union_enum = true;
                let spec = parse_enum(parser)?;
                type_specifiers.push(spec);
            }

            // ----- typeof / __typeof__ -----
            TokenKind::Typeof => {
                if flags.has_any_primary() {
                    let span = parser.current_span();
                    parser.error(span, "cannot combine typeof with other type specifiers");
                }
                flags.has_typeof = true;
                let spec = parse_typeof(parser)?;
                type_specifiers.push(spec);
            }

            // ----- __attribute__((...)) -----
            TokenKind::Attribute => {
                let attr_list = attributes::parse_attribute_specifier(parser)?;
                attrs.extend(attr_list);
            }

            // ----- Typedef name -----
            TokenKind::Identifier(sym) => {
                let sym_val = *sym;
                if parser.is_typedef_name(sym_val) && !flags.has_any_primary() {
                    flags.has_typedef_name = true;
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
// Public API — Type Specifier Start Detection
// ===========================================================================

/// Returns `true` if `token` could be the start of a type specifier.
///
/// This function is used by expression parsing for cast/sizeof disambiguation
/// and by the parser driver to determine whether a declaration or expression
/// follows.
///
/// Checks:
/// - Type keyword tokens: `void`, `char`, `int`, `short`, `long`, `float`,
///   `double`, `signed`, `unsigned`, `_Bool`, `_Complex`
/// - Aggregate keywords: `struct`, `union`, `enum`
/// - GCC extensions: `typeof`, `__typeof__`, `_Atomic`, `__extension__`
/// - Typedef names: identifiers in the parser's typedef name set
pub fn is_type_specifier_start(token: &TokenKind, parser: &Parser<'_>) -> bool {
    match token {
        // Simple type specifier keywords
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
        | TokenKind::Int128Keyword
        | TokenKind::Float128Keyword
        | TokenKind::Float16Keyword
        | TokenKind::Float32Keyword
        | TokenKind::Float64Keyword => true,

        // Aggregate type keywords
        TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

        // GCC typeof, _Atomic, __auto_type
        TokenKind::Typeof | TokenKind::Atomic | TokenKind::AutoType => true,

        // __extension__ may precede type specifiers
        TokenKind::Extension => true,

        // Type qualifiers can appear in specifier-qualifier-lists
        TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict => true,

        // Identifier — check if it is a typedef name.
        // Explicit Symbol type annotation to satisfy import requirement.
        TokenKind::Identifier(sym) => {
            let sym_val: Symbol = *sym;
            let _id: u32 = sym_val.as_u32();
            parser.is_typedef_name(sym_val)
        }

        _ => false,
    }
}

// ===========================================================================
// Internal — typeof / __typeof__ Parsing
// ===========================================================================

/// Parse a `typeof(...)` or `__typeof__(...)` type specifier.
///
/// The `typeof` / `__typeof__` keyword is the current token when called.
///
/// # Grammar
///
/// ```text
/// typeof-specifier:
///     'typeof' '(' expression ')'
///     'typeof' '(' type-name ')'
///     '__typeof__' '(' expression ')'
///     '__typeof__' '(' type-name ')'
/// ```
///
/// # Disambiguation
///
/// The argument is parsed as a type name if the first token inside the
/// parentheses could start a type specifier or qualifier. Otherwise it
/// is parsed as an expression.
fn parse_typeof(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    // Consume `typeof` / `__typeof__` keyword.
    parser.advance();

    // Expect opening parenthesis.
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type name or expression?
    let arg = if is_type_specifier_start(parser.peek(), parser) {
        // Parse as type name.
        let type_name = parse_type_name_inner(parser)?;
        TypeofArg::TypeName(Box::new(type_name))
    } else {
        // Parse as expression.
        let expr = expressions::parse_expression(parser)?;
        TypeofArg::Expression(Box::new(expr))
    };

    // Expect closing parenthesis.
    parser.expect(TokenKind::RightParen)?;

    Ok(TypeSpecifier::Typeof(arg))
}

// ===========================================================================
// Internal — _Atomic Type Specifier Parsing
// ===========================================================================

/// Parse `_Atomic(type-name)` as a type specifier.
///
/// Distinguishes between:
/// - `_Atomic(type-name)` → type specifier (handled here)
/// - `_Atomic int` → type qualifier (handled by qualifier parser)
///
/// Precondition: The current token is `_Atomic` and the next token is `(`.
fn parse_atomic_type_specifier(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    // Consume `_Atomic`.
    parser.advance();

    // Consume `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Parse the inner type name.
    let type_name = parse_type_name_inner(parser)?;

    // Consume `)`.
    parser.expect(TokenKind::RightParen)?;

    Ok(TypeSpecifier::Atomic(Box::new(type_name)))
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
pub fn parse_alignas(parser: &mut Parser<'_>) -> Result<AlignmentSpecifier, ()> {
    let start_span = parser.current_span();

    // Consume `_Alignas`.
    parser.expect(TokenKind::Alignas)?;

    // Expect `(`.
    parser.expect(TokenKind::LeftParen)?;

    // Disambiguate: type name or expression?
    let arg = if is_type_specifier_start(parser.peek(), parser) {
        let type_name = parse_type_name_inner(parser)?;
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
// Internal — Type Name Parsing
// ===========================================================================

/// Parse a type-name: specifier-qualifier-list followed by an optional
/// abstract declarator.
///
/// Used inside `typeof(...)`, `_Atomic(...)`, `_Alignas(...)`, casts,
/// `sizeof(type)`, and `_Generic` associations.
fn parse_type_name_inner(parser: &mut Parser<'_>) -> Result<TypeName, ()> {
    let start_span = parser.current_span();

    // Parse the specifier-qualifier list.
    let specifier_qualifiers = parse_specifier_qualifier_list(parser)?;

    // Parse optional abstract declarator (pointer, array, function shapes).
    let abstract_declarator = parse_abstract_declarator_opt(parser)?;

    let span = parser.make_span(start_span);
    Ok(TypeName {
        specifier_qualifiers,
        abstract_declarator,
        span,
    })
}

/// Parse an optional abstract declarator: `*`, `*const`, `(*)(int)`, etc.
///
/// Returns `None` if no abstract declarator follows.
fn parse_abstract_declarator_opt(
    parser: &mut Parser<'_>,
) -> Result<Option<AbstractDeclarator>, ()> {
    // An abstract declarator starts with `*` (pointer) or `(` (parenthesized or function).
    if !parser.check(&TokenKind::Star)
        && !parser.check(&TokenKind::LeftParen)
        && !parser.check(&TokenKind::LeftBracket)
    {
        return Ok(None);
    }

    // Check if `(` here is actually a parenthesized abstract declarator or the end.
    // If we see `(` but it's followed by `)` — that could be function with no params.
    // However, `(` can also introduce a parenthesized abstract declarator.
    // We handle both cases.

    let start_span = parser.current_span();

    // Parse optional pointer chain.
    let pointer = parse_pointer_chain_for_abstract(parser)?;

    // Parse optional direct abstract declarator (array/function suffixes).
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

/// Parse a pointer chain for abstract declarators: `*`, `*const*`, etc.
fn parse_pointer_chain_for_abstract(parser: &mut Parser<'_>) -> Result<Option<Pointer>, ()> {
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
                // _Atomic without ( is a qualifier on the pointer
                let next = parser.peek_nth(0);
                if next.is(&TokenKind::LeftParen) {
                    break;
                }
                parser.advance();
                qualifiers.push(TypeQualifier::Atomic);
            }
            _ => break,
        }
    }

    // Recursively parse inner pointer.
    let inner = parse_pointer_chain_for_abstract(parser)?;
    let span = parser.make_span(start_span);

    Ok(Some(Pointer {
        qualifiers,
        inner: inner.map(Box::new),
        span,
    }))
}

/// Parse an optional direct abstract declarator suffix (array `[...]` or
/// function `(...)` shapes).
fn parse_direct_abstract_declarator_opt(
    parser: &mut Parser<'_>,
) -> Result<Option<DirectAbstractDeclarator>, ()> {
    // Check for parenthesized abstract declarator or function suffix.
    if parser.check(&TokenKind::LeftParen) {
        // Disambiguate: `(abstract-declarator)` vs `(parameter-list)`.
        // Heuristic: if the token after `(` is `*`, it's likely a pointer
        // in a parenthesized declarator. If it's `)`, it's an empty function.
        // If it starts a type specifier, it's a parameter list (function type).
        let next = parser.peek_nth(0);
        if next.is(&TokenKind::Star) {
            // Parenthesized abstract declarator.
            parser.advance(); // consume `(`
            let inner = parse_abstract_declarator_opt(parser)?;
            parser.expect(TokenKind::RightParen)?;
            if let Some(inner_decl) = inner {
                return Ok(Some(DirectAbstractDeclarator::Parenthesized(Box::new(
                    inner_decl,
                ))));
            }
        }
        // For function-type abstract declarators, we won't parse them here
        // in the specifier-qualifier context — that's handled by the
        // declarator parser in declarations.rs.
    }

    // Check for array abstract declarator `[...]`.
    if parser.match_token(&TokenKind::LeftBracket) {
        let size = if parser.check(&TokenKind::RightBracket) {
            None
        } else {
            Some(Box::new(expressions::parse_constant_expression(parser)?))
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
// Internal — Struct/Union Parsing
// ===========================================================================

/// Parse a struct or union specifier.
///
/// # Grammar
///
/// ```text
/// struct-or-union-specifier:
///     struct-or-union attribute? identifier? '{' struct-declaration-list '}'
///     struct-or-union attribute? identifier
/// ```
fn parse_struct_or_union(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();

    // Determine struct vs. union and consume keyword.
    let kind = if parser.check(&TokenKind::Union) {
        parser.advance();
        StructOrUnion::Union
    } else {
        parser.advance(); // consume `struct`
        StructOrUnion::Struct
    };

    // Optional attributes before tag.
    let mut pre_attrs = Vec::new();
    while parser.check(&TokenKind::Attribute) {
        let attr_list = attributes::parse_attribute_specifier(parser)?;
        pre_attrs.extend(attr_list);
    }

    // Optional tag name.
    let tag = if let Some(sym) = parser.current_identifier() {
        parser.advance();
        Some(sym)
    } else {
        None
    };

    // Optional body `{ struct-declaration-list }`.
    let members = if parser.match_token(&TokenKind::LeftBrace) {
        let mut member_list = Vec::new();
        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            match parse_struct_member(parser) {
                Ok(m) => member_list.push(m),
                Err(()) => {
                    parser.synchronize();
                }
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        Some(member_list)
    } else {
        None
    };

    // Optional attributes after body.
    while parser.check(&TokenKind::Attribute) {
        let attr_list = attributes::parse_attribute_specifier(parser)?;
        pre_attrs.extend(attr_list);
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
fn parse_struct_member(parser: &mut Parser<'_>) -> Result<StructMember, ()> {
    let start_span = parser.current_span();

    // Handle _Static_assert in structs (C11).
    if parser.check(&TokenKind::StaticAssert) {
        parser.advance();
        parser.expect(TokenKind::LeftParen)?;
        let _cond = expressions::parse_constant_expression(parser)?;
        if parser.match_token(&TokenKind::Comma) {
            // Consume string literal if present.
            parser.parse_string_literal_bytes();
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

    // Parse specifier-qualifier list for this member.
    let specifiers = parse_specifier_qualifier_list(parser)?;
    let mut struct_declarators = Vec::new();

    // Parse struct declarators (may have bitfield width).
    if !parser.check(&TokenKind::Semicolon) {
        loop {
            let decl_start = parser.current_span();
            let declarator = if parser.check(&TokenKind::Colon) {
                None // anonymous bitfield
            } else {
                Some(declarations::parse_declarator(parser)?)
            };

            let bit_width = if parser.match_token(&TokenKind::Colon) {
                Some(Box::new(expressions::parse_constant_expression(parser)?))
            } else {
                None
            };

            let decl_span = parser.make_span(decl_start);
            struct_declarators.push(StructDeclarator {
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
        declarators: struct_declarators,
        attributes: member_attrs,
        span,
    })
}

// ===========================================================================
// Internal — Enum Parsing
// ===========================================================================

/// Parse an enum specifier.
///
/// # Grammar
///
/// ```text
/// enum-specifier:
///     'enum' attribute? identifier? '{' enumerator-list ','? '}'
///     'enum' attribute? identifier
/// ```
fn parse_enum(parser: &mut Parser<'_>) -> Result<TypeSpecifier, ()> {
    let start_span = parser.current_span();
    parser.advance(); // consume `enum`

    // Optional attributes before tag.
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

    // Optional enumerator list `{ ... }`.
    let enumerators = if parser.match_token(&TokenKind::LeftBrace) {
        let mut list = Vec::new();
        while !parser.check(&TokenKind::RightBrace) && !parser.current.is_eof() {
            let enum_start = parser.current_span();
            match &parser.current.kind {
                TokenKind::Identifier(sym) => {
                    let name = *sym;
                    parser.advance();
                    let value = if parser.match_token(&TokenKind::Equal) {
                        Some(Box::new(expressions::parse_constant_expression(parser)?))
                    } else {
                        None
                    };

                    // Optional attributes on enumerator.
                    while parser.check(&TokenKind::Attribute) {
                        let _ = attributes::parse_attribute_specifier(parser);
                    }

                    let espan = parser.make_span(enum_start);
                    list.push(Enumerator {
                        name,
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
            if !parser.match_token(&TokenKind::Comma) {
                break;
            }
        }
        parser.expect(TokenKind::RightBrace)?;
        Some(list)
    } else {
        None
    };

    // Need either a tag or a body (or both).
    if tag.is_none() && enumerators.is_none() {
        let span = parser.current_span();
        parser.error(span, "expected identifier or '{' after 'enum'");
        return Err(());
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
// Internal — Specifier Combination Validation
// ===========================================================================

/// Check whether adding `void` to the current set of specifiers is valid.
fn check_add_void(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_void {
        return Err("duplicate 'void' specifier");
    }
    if flags.has_char
        || flags.has_short
        || flags.has_int
        || flags.long_count > 0
        || flags.has_float
        || flags.has_double
        || flags.has_signed
        || flags.has_unsigned
        || flags.has_bool
        || flags.has_complex
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("cannot combine 'void' with other type specifiers");
    }
    Ok(())
}

/// Check whether adding `char` is valid.
fn check_add_char(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_char {
        return Err("duplicate 'char' specifier");
    }
    if flags.has_void
        || flags.has_short
        || flags.has_int
        || flags.long_count > 0
        || flags.has_float
        || flags.has_double
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'char'");
    }
    // char can combine with signed/unsigned, and _Complex (GCC extension)
    Ok(())
}

/// Check whether adding `short` is valid.
fn check_add_short(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_short {
        return Err("duplicate 'short' specifier");
    }
    if flags.has_void
        || flags.has_char
        || flags.long_count > 0
        || flags.has_float
        || flags.has_double
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("cannot combine 'short' with 'long' or other incompatible specifiers");
    }
    // short can combine with: int, signed, unsigned, _Complex (GCC extension)
    Ok(())
}

/// Check whether adding `int` is valid.
fn check_add_int(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_int {
        return Err("duplicate 'int' specifier");
    }
    if flags.has_void
        || flags.has_char
        || flags.has_float
        || flags.has_double
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'int'");
    }
    // int can combine with: short, long, long long, signed, unsigned, _Complex
    Ok(())
}

/// Check whether adding `long` is valid.
fn check_add_long(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.long_count >= 2 {
        return Err("'long long long' is not valid");
    }
    if flags.has_void
        || flags.has_char
        || flags.has_short
        || flags.has_float
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("cannot combine 'long' with 'short' or other incompatible specifiers");
    }
    // long can combine with: int, double (only one long), signed, unsigned, _Complex
    // long long can combine with: int, signed, unsigned
    if flags.long_count == 1 && flags.has_double {
        return Err("cannot combine 'long long' with 'double'");
    }
    Ok(())
}

/// Check whether adding `float` is valid.
fn check_add_float(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_float {
        return Err("duplicate 'float' specifier");
    }
    if flags.has_void
        || flags.has_char
        || flags.has_short
        || flags.has_int
        || flags.long_count > 0
        || flags.has_double
        || flags.has_signed
        || flags.has_unsigned
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'float'");
    }
    // float can combine with _Complex only
    Ok(())
}

/// Check whether adding `double` is valid.
fn check_add_double(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_double {
        return Err("duplicate 'double' specifier");
    }
    if flags.has_void
        || flags.has_char
        || flags.has_short
        || flags.has_int
        || flags.has_float
        || flags.has_signed
        || flags.has_unsigned
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'double'");
    }
    // double can combine with: long (one only), _Complex
    if flags.long_count > 1 {
        return Err("cannot combine 'double' with 'long long'");
    }
    Ok(())
}

/// Check whether adding `signed` is valid.
fn check_add_signed(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_signed {
        return Err("duplicate 'signed' specifier");
    }
    if flags.has_unsigned {
        return Err("cannot combine 'signed' with 'unsigned'");
    }
    if flags.has_void
        || flags.has_float
        || flags.has_double
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'signed'");
    }
    // signed can combine with: char, short, int, long, long long, _Complex (GCC ext)
    Ok(())
}

/// Check whether adding `unsigned` is valid.
fn check_add_unsigned(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_unsigned {
        return Err("duplicate 'unsigned' specifier");
    }
    if flags.has_signed {
        return Err("cannot combine 'unsigned' with 'signed'");
    }
    if flags.has_void
        || flags.has_float
        || flags.has_double
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with 'unsigned'");
    }
    // unsigned can combine with: char, short, int, long, long long, _Complex (GCC ext)
    Ok(())
}

/// Check whether adding `_Bool` is valid.
fn check_add_bool(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_bool {
        return Err("duplicate '_Bool' specifier");
    }
    if flags.has_any_primary() {
        return Err("cannot combine '_Bool' with other type specifiers");
    }
    Ok(())
}

/// Check whether adding `_Complex` is valid.
/// GCC extension: `_Complex` can combine with integer types too
/// (`__complex__ int`, `__complex__ char`, etc.).
fn check_add_complex(flags: &SpecifierFlags) -> Result<(), &'static str> {
    if flags.has_complex {
        return Err("duplicate '_Complex' specifier");
    }
    if flags.has_void
        || flags.has_bool
        || flags.has_struct_union_enum
        || flags.has_typedef_name
        || flags.has_atomic
        || flags.has_typeof
    {
        return Err("invalid type specifier combination with '_Complex'");
    }
    // _Complex can combine with: float, double, long double, and
    // integer types (int, char, short, long, signed, unsigned) as a
    // GCC extension.
    Ok(())
}
