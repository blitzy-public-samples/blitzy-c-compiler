//! Preprocessor module — Phases 1 and 2 of the BCC compilation pipeline.
//!
//! The C preprocessor handles:
//!
//! - **Phase 1:** Trigraph replacement and line splicing (backslash-newline
//!   continuation).
//!
//! - **Phase 2:** Directive processing (`#include`, `#define`, `#undef`,
//!   `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`,
//!   `#warning`, `#line`) and macro expansion with paint-marker recursion
//!   protection for self-referential macros.
//!
//! # Paint-Marker Recursion Protection
//!
//! The [`paint_marker`] submodule implements the C standard's rule (C11
//! §6.10.3.4) that prevents self-referential macro expansion. When a macro
//! is expanded, its name is "painted" onto all replacement tokens; during
//! rescanning, painted tokens matching the macro name are treated as ordinary
//! identifiers and are not re-expanded.
//!
//! This is architecturally distinct from:
//! - Circular `#include` detection (file-level, handled by the include handler)
//! - The 512-depth recursion limit (global safety net for deeply nested chains)
//!
//! # Dependencies
//!
//! The preprocessor depends on [`crate::common`] for infrastructure (encoding,
//! diagnostics, source map, string interning) and [`crate::frontend::lexer`]
//! for token types. It does NOT depend on [`crate::ir`] or [`crate::backend`].

/// Token-level paint marker implementation for macro expansion recursion
/// protection. Tracks which macros have "painted" each token to suppress
/// re-expansion of self-referential macro definitions.
pub mod paint_marker;

/// Token pasting (`##`) and stringification (`#`) operator implementation.
/// Handles concatenation of preprocessing tokens via `##` and conversion of
/// macro arguments to string literals via `#`, with proper whitespace and
/// token-boundary handling per C11 §6.10.3.
pub mod token_paster;

/// `#include` file resolution — system path and user path search distinction,
/// include guard optimization (`#ifndef`/`#define`/`#endif` detection), circular
/// include detection with diagnostic chain reporting, `#pragma once` support,
/// and PUA-aware file loading via [`crate::common::encoding::read_source_file`].
pub mod include_handler;
