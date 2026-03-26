//! # BCC Frontend — C11 Compilation Frontend Pipeline
//!
//! This module implements the complete C11 frontend pipeline with GCC extension support:
//!
//! - [`preprocessor`] — Phases 1–2: trigraph replacement, line splicing, `#include`,
//!   `#define`/`#undef`, `#if`/`#ifdef`/`#elif`/`#else`/`#endif`, macro expansion with
//!   paint-marker recursion protection, PUA encoding for non-UTF-8 byte round-tripping
//! - [`lexer`] — Phase 3: tokenization with PUA-aware UTF-8 scanning, all C11 keywords,
//!   GCC extension keywords, numeric/string/character literal parsing
//! - [`parser`] — Phase 4: recursive-descent C11 parser with comprehensive GCC extension
//!   support (statement expressions, typeof, computed gotos, case ranges, attributes,
//!   inline assembly with AT&T syntax)
//! - [`sema`] — Phase 5: semantic analysis including type checking, scope management,
//!   symbol tables, constant evaluation, GCC builtin evaluation, initializer analysis,
//!   and attribute validation
//!
//! ## Dependencies
//!
//! The frontend depends on [`crate::common`] for infrastructure:
//! - FxHash for performant hash maps
//! - PUA/UTF-8 encoding for non-UTF-8 source files
//! - Type system (`CType`, `MachineType`)
//! - Diagnostics engine for error/warning reporting
//! - Source map for file/line tracking
//! - String interner for O(1) identifier comparison
//! - Target definitions for architecture-dependent behavior
//!
//! The frontend does NOT depend on `ir`, `passes`, or `backend` modules.
//!
//! ## Key Architectural Decisions
//!
//! - PUA encoding is transparent at the preprocessor/lexer boundary
//! - Paint-marker recursion protection is architecturally distinct from circular `#include` detection
//! - 512-depth recursion limit is enforced in the parser and macro expander
//! - GCC extension keywords are recognized by the lexer as proper token kinds
//! - All AST nodes carry source spans for diagnostic reporting

pub mod lexer; // Phase 3: tokenization
pub mod parser; // Phase 4: parsing, AST construction
pub mod preprocessor; // Phases 1–2: preprocessing, macro expansion
pub mod sema; // Phase 5: semantic analysis
