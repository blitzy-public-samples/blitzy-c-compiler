//! Frontend module — Phases 1–5 of the BCC compilation pipeline.
//!
//! This module contains the complete C language frontend, organized into
//! four major submodules that sequentially process source text from raw
//! characters to a semantically validated AST:
//!
//! - **Preprocessor** (Phases 1–2): Trigraph replacement, line splicing,
//!   `#include` resolution, `#define`/`#undef` macro management, conditional
//!   compilation (`#if`/`#ifdef`/`#elif`/`#else`/`#endif`), with paint-marker
//!   recursion protection for self-referential macros.
//!
//! - **Lexer** (Phase 3): Character-by-character scanning with PUA-aware
//!   UTF-8 handling, producing a token stream of keywords, identifiers,
//!   literals, operators, and punctuators.
//!
//! - **Parser** (Phase 4): Recursive-descent C11 parser with comprehensive
//!   GCC extension support (statement expressions, `typeof`, computed gotos,
//!   case ranges, `__attribute__`, inline assembly).
//!
//! - **Semantic Analyzer** (Phase 5): Type checking, scope management,
//!   symbol table construction, constant expression evaluation, GCC builtin
//!   evaluation, initializer analysis, and attribute validation.
//!
//! # Processing Order
//!
//! ```text
//! Source File
//!     │
//!     ▼
//! Preprocessor (Phase 1–2)
//!   ├─ Phase 1: trigraph replacement, line splicing
//!   └─ Phase 2: #include, #define, #if, macro expansion (paint-marker protected)
//!     │
//!     ▼  (expanded source text)
//! Lexer (Phase 3)
//!   └─ PUA-aware tokenization, keyword recognition, literal parsing
//!     │
//!     ▼  (Token stream)
//! Parser (Phase 4)
//!   └─ AST construction with full GCC extension support
//!     │
//!     ▼  (TranslationUnit AST)
//! Semantic Analyzer (Phase 5)
//!   └─ Type checking, scope resolution, constant evaluation, attribute validation
//!     │
//!     ▼  (CheckedTranslationUnit — consumed by Phase 6 IR lowering)
//! ```
//!
//! # Dependencies
//!
//! The frontend depends on [`crate::common`] for infrastructure (encoding,
//! types, diagnostics, source map, string interning, target definitions)
//! but does **NOT** depend on [`crate::ir`] or [`crate::backend`].
//!
//! # Convenience Re-exports
//!
//! The most commonly used types from each submodule are re-exported at this
//! level for ergonomic access by the IR lowering phase and main driver:
//!
//! - From [`lexer`]: [`Token`], [`TokenKind`]
//! - From [`parser`]: [`TranslationUnit`], [`Declaration`], [`Expression`],
//!   [`Statement`], [`Parser`], [`ParseError`], [`ParseResult`]
//! - From [`sema`]: [`SemanticAnalyzer`], [`CheckedTranslationUnit`],
//!   [`CheckedDeclaration`]
//! - From [`preprocessor`]: [`Preprocessor`], [`MacroDef`]

// ═══════════════════════════════════════════════════════════════════════════
// Submodule Declarations
// ═══════════════════════════════════════════════════════════════════════════

/// Preprocessor — Phases 1–2: trigraph replacement, line splicing, directive
/// processing (`#include`, `#define`, `#if`/`#ifdef`/`#elif`/`#else`/`#endif`),
/// and macro expansion with paint-marker recursion protection.
///
/// # Submodules
///
/// - [`preprocessor::directives`] — `#include`, `#define`, `#undef`, conditionals
/// - [`preprocessor::macro_expander`] — object-like & function-like macro expansion
/// - [`preprocessor::paint_marker`] — token paint state for recursion suppression
/// - [`preprocessor::include_handler`] — `#include` file resolution, guards, circular detection
/// - [`preprocessor::token_paster`] — `##` concatenation and `#` stringification
/// - [`preprocessor::expression`] — `#if`/`#elif` constant expression evaluation
/// - [`preprocessor::predefined`] — `__FILE__`, `__LINE__`, arch-specific defines
pub mod preprocessor;

/// Lexer — Phase 3 tokenization.
///
/// Converts PUA-encoded source text into a token stream. Contains the
/// character-level scanner and token type definitions, numeric literal
/// parsing, and string literal parsing.
///
/// # Submodules
///
/// - [`lexer::token`] — token kind definitions, span, literal suffix types
/// - [`lexer::scanner`] — PUA-aware character-level scanner
/// - [`lexer::number_literal`] — numeric literal parsing (decimal, hex, octal, binary, float)
/// - [`lexer::string_literal`] — string/character literal parsing with escape sequences
pub mod lexer;

/// Parser — Phase 4: recursive-descent C11 parser with comprehensive GCC
/// extension support.
///
/// Converts a token stream into an Abstract Syntax Tree (AST). Contains the
/// AST node definitions, declaration/expression/statement/type parsing,
/// GCC extension handling, `__attribute__` parsing, and inline assembly parsing.
///
/// # Submodules
///
/// - [`parser::ast`] — AST node definitions (TranslationUnit, Declaration, Expression, etc.)
/// - [`parser::declarations`] — declaration parsing
/// - [`parser::expressions`] — expression parsing with precedence climbing
/// - [`parser::statements`] — statement parsing (control flow, labels, compounds)
/// - [`parser::types`] — type specifier/qualifier parsing
/// - [`parser::gcc_extensions`] — GCC extension dispatch
/// - [`parser::attributes`] — `__attribute__((...))` parsing
/// - [`parser::inline_asm`] — `asm`/`__asm__` statement parsing
pub mod parser;

/// Semantic Analyzer — Phase 5 semantic analysis.
///
/// Type checking, scope management, symbol table construction, constant
/// expression evaluation, GCC builtin evaluation, initializer analysis,
/// and attribute validation. Produces [`CheckedTranslationUnit`] consumed
/// by Phase 6 IR lowering.
///
/// # Submodules
///
/// - [`sema::type_checker`] — type inference and validation
/// - [`sema::scope`] — lexical scope management
/// - [`sema::symbol_table`] — symbol storage and linkage resolution
/// - [`sema::constant_eval`] — compile-time constant expression evaluation
/// - [`sema::builtin_eval`] — GCC `__builtin_*` evaluation
/// - [`sema::initializer`] — designated initializer analysis
/// - [`sema::attribute_handler`] — attribute validation and propagation
pub mod sema;

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Re-exports
// ═══════════════════════════════════════════════════════════════════════════
//
// The following re-exports provide ergonomic access to the most commonly
// used frontend types. Consumers (IR lowering, main driver) can write:
//
//   use crate::frontend::{TranslationUnit, CheckedTranslationUnit, Token};
//
// instead of the fully qualified paths:
//
//   use crate::frontend::parser::ast::TranslationUnit;
//   use crate::frontend::sema::CheckedTranslationUnit;
//   use crate::frontend::lexer::token::Token;

// ── Preprocessor re-exports ────────────────────────────────────────────────
// The preprocessor driver struct and macro definition type are the primary
// interfaces that the main compilation driver uses to invoke preprocessing.
pub use preprocessor::{MacroDef, Preprocessor};

// ── Lexer re-exports ───────────────────────────────────────────────────────
// Token and TokenKind are the fundamental currency of the lexer-parser
// interface. Re-exported here so downstream consumers can access them
// without reaching into the lexer submodule hierarchy.
pub use lexer::{Token, TokenKind};

// ── Parser re-exports ──────────────────────────────────────────────────────
// The parser's AST node types are consumed by the semantic analyzer and
// (indirectly, through sema output) by the IR lowering phase. Parser and
// ParseError/ParseResult are needed by the main driver to invoke parsing.
pub use parser::{
    Declaration, Expression, ParseError, ParseResult, Parser, Statement, TranslationUnit,
};

// ── Semantic analyzer re-exports ───────────────────────────────────────────
// SemanticAnalyzer is the Phase 5 driver invoked by the main compilation
// driver. CheckedTranslationUnit is the primary output consumed by Phase 6
// (IR lowering). CheckedDeclaration provides the per-declaration detail.
pub use sema::{CheckedDeclaration, CheckedTranslationUnit, SemanticAnalyzer};
