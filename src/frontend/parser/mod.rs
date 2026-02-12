//! Parser module — Phase 4 of the BCC compilation pipeline.
//!
//! Recursive-descent C11 parser with comprehensive GCC extension support.
//! Converts a token stream into an Abstract Syntax Tree (AST).
//!
//! # Submodules
//!
//! - [`ast`]: Complete AST node hierarchy — `TranslationUnit`, `Declaration`,
//!   `Statement`, `Expression`, `TypeSpecifier`, `Attribute`, `AsmStatement`,
//!   all carrying [`ast::Span`] for source location tracking.

/// AST node definitions — the foundational types for the entire parser,
/// semantic analyzer, and IR lowering phases.
pub mod ast;
