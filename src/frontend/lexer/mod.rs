//! Lexer module — Phase 3 tokenization for the BCC compiler.
//!
//! This module provides the lexical analysis pipeline that converts a
//! PUA-encoded source string into a stream of [`Token`] values for
//! consumption by the recursive-descent parser (Phase 4).
//!
//! # Submodules
//!
//! - [`scanner`]: Character-level scanner with PUA-aware UTF-8 reading,
//!   O(1) lookahead, byte-offset position tracking, and line/column
//!   tracking for diagnostic messages.
//!
//! # PUA-Aware Processing
//!
//! The lexer operates on source text that has already been processed by
//! [`crate::common::encoding::read_source_file`], which maps non-UTF-8
//! bytes (0x80–0xFF) to Private Use Area code points (U+E080–U+E0FF).
//! The scanner transparently passes these PUA code points through without
//! interpretation, while character classification helpers explicitly
//! exclude them from identifier and whitespace categories to preserve
//! byte-exact round-tripping fidelity.

/// Token type definitions — the most foundational type in the entire frontend.
///
/// Defines [`TokenKind`], [`Token`], [`Span`] (re-exported from diagnostics),
/// and auxiliary enums for literal suffixes and prefixes. Every subsequent
/// frontend module (preprocessor, parser, sema) imports from this module.
pub mod token;

/// Character-level scanner for PUA-aware UTF-8 source reading.
///
/// Provides the foundational character-reading layer: peek/advance with
/// O(1) lookahead, byte-offset and line/column position tracking, source
/// slice extraction, and character classification that respects PUA opacity.
pub mod scanner;
