//! Semantic analysis module — Phase 5 of the BCC compilation pipeline.
//!
//! This module contains the semantic analysis components that transform
//! the parser-produced AST into a semantically validated, type-annotated
//! representation suitable for IR lowering (Phase 6).
//!
//! # Submodules
//!
//! - [`symbol_table`]: Symbol storage, declaration/definition merging,
//!   linkage resolution (C11 §6.2.2), and GCC attribute tracking.
//!
//! # Dependencies
//!
//! The semantic analyzer depends on:
//! - [`crate::common`] for infrastructure (types, diagnostics, source map,
//!   string interning, target definitions, FxHash collections)
//! - [`crate::frontend::lexer`] for token/span types
//!
//! The semantic analyzer does NOT depend on [`crate::ir`] or [`crate::backend`].

/// Symbol table — declaration tracking, linkage resolution, and attribute storage.
///
/// Provides [`SymbolTable`], [`SymbolEntry`], [`SymbolId`], [`Linkage`],
/// [`StorageClass`], and [`SymbolAttributes`] types for tracking all
/// declared identifiers within a translation unit.
pub mod attribute_handler;
pub mod constant_eval;
pub mod scope;
pub mod symbol_table;
