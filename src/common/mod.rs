//! Common infrastructure layer for the BCC (Blitzy's C Compiler) compiler.
//!
//! This module provides the foundational utilities and data structures imported
//! by every other layer in the compilation pipeline — `frontend`, `ir`, `passes`,
//! and `backend`. All submodules are implemented from scratch using only the Rust
//! standard library, adhering to the project's strict zero-dependency mandate.
//!
//! # Submodules
//!
//! - [`fx_hash`]: Fast non-cryptographic Fibonacci hashing (FxHash) for symbol
//!   tables, scope lookups, and all performance-critical hash-based data structures.
//!   Provides [`FxHashMap`] and [`FxHashSet`] type aliases that replace the default
//!   SipHash with significantly faster multiplicative hashing for compiler-typical
//!   small string and integer keys.
//!
//! - [`encoding`]: PUA/UTF-8 encoding for non-UTF-8 byte round-tripping. Maps
//!   bytes in the 0x80–0xFF range to Unicode Private Use Area code points
//!   (U+E080–U+E0FF), ensuring byte-exact fidelity of binary data in C string
//!   literals and inline assembly operands through the Rust `String`-based pipeline.
//!
//! - [`long_double`]: Software 80-bit IEEE 754 extended-precision arithmetic
//!   (add, sub, mul, div, comparison, conversion to/from `f64`/`i64`/`u64`)
//!   without external math libraries. Required by the zero-dependency mandate for
//!   compile-time constant evaluation of `long double` values.
//!
//! - [`temp_files`]: RAII-based temporary file and directory management with
//!   automatic cleanup via `Drop`. Used for intermediate object files during
//!   multi-file compilation. Employs an atomic counter for thread-safe unique
//!   name generation.
//!
//! - [`types`]: The dual type system — [`CType`] for C11 language types (Void
//!   through Typedef with all qualifiers) and [`MachineType`] for backend
//!   register-class mappings. Provides target-dependent `sizeof`/`alignof`,
//!   integer promotion, and usual arithmetic conversion rules used by every
//!   pipeline stage.
//!
//! - [`type_builder`]: Builder-pattern API for constructing complex C types
//!   (e.g., `TypeBuilder::new().pointer_to(CType::Int).build()`) and computing
//!   struct/union layouts with `packed`/`aligned` attribute support and flexible
//!   array member handling.
//!
//! - [`diagnostics`]: Multi-error diagnostic reporting engine with source spans,
//!   severity levels (error, warning, note, help), fix suggestions, and
//!   GCC-compatible formatted output. Integrates with [`SourceMap`] for
//!   file/line/column resolution. Every pipeline stage emits diagnostics through
//!   this module.
//!
//! - [`source_map`]: Source file tracking with file IDs, pre-computed line offset
//!   tables for O(log n) line/column lookups from byte offsets, and `#line`
//!   directive remapping. Used by diagnostics for error location formatting and
//!   by the DWARF emitter for debug line information.
//!
//! - [`string_interner`]: String interning with [`FxHashMap`]-backed deduplication.
//!   Each unique string is stored once and represented by a compact 4-byte
//!   [`Symbol`] handle supporting `Copy`, `Eq`, and `Hash` for zero-cost
//!   identifier comparison in symbol tables and AST nodes.
//!
//! - [`target`]: Target architecture definitions for the four supported platforms
//!   (x86-64, i686, AArch64, RISC-V 64). Provides architecture-specific properties
//!   including pointer width, endianness, data model (LP64/ILP32), predefined
//!   preprocessor macros, ELF machine constants, and dynamic linker paths. Target
//!   information flows through every pipeline stage from CLI parsing to code
//!   generation.

// ---------------------------------------------------------------------------
// Submodule declarations — all 10 common infrastructure modules
// ---------------------------------------------------------------------------

/// Multi-error diagnostic reporting with source spans and GCC-compatible output.
pub mod diagnostics;

/// PUA/UTF-8 encoding for non-UTF-8 byte round-tripping in C source files.
pub mod encoding;

/// Fast non-cryptographic Fibonacci hashing for compiler hash maps and sets.
pub mod fx_hash;

/// Software 80-bit IEEE 754 extended-precision arithmetic for `long double`.
pub mod long_double;

/// Source file tracking with line offset tables and `#line` directive remapping.
pub mod source_map;

/// String interning with FxHash-backed deduplication for identifiers and keywords.
pub mod string_interner;

/// Target architecture definitions (x86-64, i686, AArch64, RISC-V 64).
pub mod target;

/// RAII-based temporary file and directory management with automatic cleanup.
pub mod temp_files;

/// Builder-pattern API for constructing complex C types and computing layouts.
pub mod type_builder;

/// Dual type system: C language types (`CType`) and machine types (`MachineType`).
pub mod types;

// ---------------------------------------------------------------------------
// Convenience re-exports — most commonly used types at the module level
// ---------------------------------------------------------------------------
// These re-exports allow other modules to write e.g. `use crate::common::FxHashMap`
// instead of the longer `use crate::common::fx_hash::FxHashMap` path, reducing
// import boilerplate across the codebase.

// --- Diagnostics ---
// Re-export diagnostic types for ergonomic access via `crate::common::Diagnostic`,
// `crate::common::DiagnosticEngine`, etc.
pub use diagnostics::{Diagnostic, DiagnosticEngine, FixSuggestion, Severity, Span};

// --- Encoding ---
// Re-export PUA encoding functions for ergonomic access via
// `crate::common::encode_byte_to_pua`, `crate::common::read_source_file`, etc.
pub use encoding::{
    decode_pua_string, decode_pua_to_byte, encode_byte_to_pua, is_pua_encoded, read_source_file,
    write_decoded_file,
};

// --- FxHash ---
// Re-export FxHash types and constructors for ergonomic access via
// `crate::common::FxHashMap`, `crate::common::fx_hash_map()`, etc.
pub use fx_hash::{fx_hash_map, fx_hash_map_with_capacity, fx_hash_set};
pub use fx_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};

// --- Long Double ---
// Re-export LongDouble for ergonomic access via `crate::common::LongDouble`.
pub use long_double::LongDouble;

// --- Source Map ---
// Re-export source map types for ergonomic access via `crate::common::SourceMap`,
// `crate::common::FileId`, etc.
pub use source_map::{FileId, LineDirective, SourceFile, SourceLocation, SourceMap};

// --- String Interner ---
// Re-export string interner types for ergonomic access via `crate::common::Interner`
// and `crate::common::Symbol`.
pub use string_interner::{Interner, Symbol};

// --- Target ---
// Re-export target architecture types for ergonomic access via
// `crate::common::Target`, `crate::common::DataModel`, etc.
pub use target::{DataModel, Endianness, Target};

// --- Temp Files ---
// Re-export temporary file types for ergonomic access via `crate::common::TempFile`
// and `crate::common::TempDir`.
pub use temp_files::{TempDir, TempFile};

// --- Types ---
// Re-export C type system types and utility functions for ergonomic access via
// `crate::common::CType`, `crate::common::size_of()`, etc.
pub use types::{
    align_of, integer_promote, size_of, usual_arithmetic_conversion, CType, FieldDef, MachineType,
    QualifiedType, TypeQualifiers,
};

// --- Type Builder ---
// Re-export type builder types and layout computation functions for ergonomic access
// via `crate::common::TypeBuilder`, `crate::common::compute_struct_layout()`, etc.
pub use type_builder::{
    composite_type, compute_struct_layout, compute_union_layout, ctype_to_machine_type,
    types_compatible, FieldLayout, StructLayout, TypeBuilder, UnionLayout,
};
