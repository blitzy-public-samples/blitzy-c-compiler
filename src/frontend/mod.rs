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
//! # Dependencies
//!
//! The frontend depends on [`crate::common`] for infrastructure (encoding,
//! types, diagnostics, source map, string interning, target definitions)
//! but does NOT depend on [`crate::ir`] or [`crate::backend`].

/// Preprocessor — Phases 1–2: trigraph replacement, line splicing, directive
/// processing, and macro expansion with paint-marker recursion protection.
pub mod preprocessor;

/// Lexer — Phase 3 tokenization.
///
/// Converts PUA-encoded source text into a token stream. Contains the
/// character-level scanner and (future) token type definitions, numeric
/// literal parsing, and string literal parsing.
pub mod lexer;

/// Semantic Analyzer — Phase 5 semantic analysis.
///
/// Type checking, scope management, symbol table construction, constant
/// expression evaluation, GCC builtin evaluation, initializer analysis,
/// and attribute validation.
pub mod sema;
