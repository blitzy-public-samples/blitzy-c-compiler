//! BCC — Blitzy's C Compiler Library
//!
//! A complete, self-contained, zero-external-dependency C compilation toolchain
//! implemented in Rust (2021 Edition) that cross-compiles C source code into
//! native Linux ELF executables and shared objects for four target architectures:
//! x86-64, i686, AArch64, and RISC-V 64.
//!
//! # Architecture
//!
//! The compiler is organized as a multi-phase pipeline:
//!
//! 1. **Common Infrastructure** ([`common`]): Foundational utilities including
//!    fast hashing ([`common::fx_hash`]), encoding, type system, diagnostics,
//!    source tracking, string interning, and target definitions.
//!
//! # Zero-Dependency Mandate
//!
//! BCC uses **no external crates**. All functionality — hashing, encoding,
//! math operations, ELF writing, DWARF emission, assemblers, and linkers —
//! is implemented internally using only the Rust standard library.

// Crate-level lint configuration for compiler-project-specific patterns.
// Compiler functions naturally have many parameters due to pipeline context.
#![allow(clippy::too_many_arguments)]
// AST and IR enums naturally have varying-size variants.
#![allow(clippy::large_enum_variant)]
// Allow module inception (e.g., module named same as parent).
#![allow(clippy::module_inception)]
// Type complexity is inherent in compiler type representations.
#![allow(clippy::type_complexity)]

/// Common infrastructure layer — foundational utilities imported by every
/// other module in the compiler. Provides fast hashing, encoding, type system,
/// diagnostics, source map, string interning, and target architecture definitions.
pub mod common;

/// Frontend module — Phases 1–5 of the compilation pipeline: preprocessor,
/// lexer, parser, and semantic analyzer. Depends on `common` but not on `ir`
/// or `backend`.
pub mod frontend;

/// Intermediate representation (IR) module — Phases 6, 7, and 9 of the
/// compilation pipeline: AST-to-IR lowering with alloca-first insertion,
/// SSA construction via alloca promotion (mem2reg), and phi elimination.
/// Depends on `common` and `frontend` (for AST types during lowering).
/// Consumed by `passes` (optimization) and `backend` (code generation).
pub mod ir;

/// Optimization passes module — Phase 8 of the compilation pipeline.
/// Provides constant folding, dead code elimination, and CFG simplification.
/// Runs between SSA construction (Phase 7 / mem2reg) and phi elimination
/// (Phase 9). Depends on `common` and `ir`.
pub mod passes;

/// Backend module — code generation, assemblers, linkers, and DWARF debug
/// information generation for all four target architectures.
pub mod backend;


