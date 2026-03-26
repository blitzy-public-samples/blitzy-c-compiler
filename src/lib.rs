//! # BCC — Blitzy's C Compiler
//!
//! A complete, self-contained, zero-external-dependency C11 compiler implemented
//! in Rust (2021 Edition) that produces native Linux ELF executables and shared
//! objects for four target architectures: x86-64, i686, AArch64, and RISC-V 64.
//!
//! ## Module Structure
//!
//! - [`common`] — Infrastructure layer (hashing, encoding, types, diagnostics)
//! - [`frontend`] — C11 frontend (preprocessor, lexer, parser, semantic analysis)
//! - [`ir`] — Intermediate representation, lowering, SSA construction
//! - [`passes`] — Optimization passes (constant folding, DCE, CFG simplification)
//! - [`backend`] — Code generation, assemblers, linkers, DWARF, ELF writing
//!
//! ## Architecture
//!
//! The compilation pipeline follows 10+ phases from preprocessing through
//! code generation, using the "alloca-then-promote" SSA construction strategy.
//! All components (assembler, linker, DWARF emitter) are built-in — no
//! external toolchain invocation occurs.
//!
//! ## Pipeline Overview
//!
//! 1. **Phases 1–2** (Preprocessor): Trigraph replacement, line splicing,
//!    `#include`/`#define` processing, macro expansion with paint-marker
//!    recursion protection, PUA encoding for non-UTF-8 byte round-tripping.
//! 2. **Phase 3** (Lexer): Tokenization with PUA-aware UTF-8 scanning.
//! 3. **Phase 4** (Parser): Recursive-descent C11 parsing with GCC extensions,
//!    inline assembly, and attribute support.
//! 4. **Phase 5** (Sema): Type checking, scope management, constant evaluation,
//!    GCC builtin evaluation, initializer analysis, attribute validation.
//! 5. **Phase 6** (IR Lowering): AST-to-IR conversion with "alloca-first"
//!    pattern — all locals start as alloca instructions.
//! 6. **Phase 7** (Mem2Reg): SSA construction via dominance-frontier-based
//!    alloca promotion.
//! 7. **Phase 8** (Optimization): Constant folding, dead code elimination,
//!    CFG simplification with fixpoint iteration.
//! 8. **Phase 9** (Phi Elimination): SSA deconstruction — phi nodes are
//!    replaced by parallel copies for register allocation.
//! 9. **Phase 10** (Code Generation): Architecture-specific instruction
//!    selection, register allocation, machine code emission, ELF linking.
//!
//! ## Target Architectures
//!
//! | Architecture | Data Model | ABI           | ELF Machine |
//! |-------------|-----------|---------------|-------------|
//! | x86-64      | LP64      | System V AMD64| EM_X86_64   |
//! | i686        | ILP32     | cdecl / SysV  | EM_386      |
//! | AArch64     | LP64      | AAPCS64       | EM_AARCH64  |
//! | RISC-V 64   | LP64      | LP64D         | EM_RISCV    |
//!
//! ## Zero-Dependency Mandate
//!
//! This crate has **zero** external Rust crate dependencies. Every capability
//! — FxHash, PUA encoding, long-double arithmetic, ELF writing, DWARF
//! emission, assemblers, linkers — is implemented internally using only the
//! Rust standard library (`std`).

// ============================================================================
// Crate-level attributes
// ============================================================================

// Compiler code routinely needs many parameters for pipeline context,
// compilation options, target info, and diagnostic state. Allow this
// pattern without clippy noise.
#![allow(clippy::too_many_arguments)]
// Complex nested types are inherent in AST node definitions, IR type
// hierarchies, and machine instruction representations. Suppress warnings
// for legitimate complexity in compiler data structures.
#![allow(clippy::type_complexity)]
// Large enum variants are expected in AST/IR node representations where
// some variants (e.g. Struct, Function) carry significantly more data
// than simple leaf variants. Boxing every large variant would add
// indirection overhead in the hot compilation path.
#![allow(clippy::large_enum_variant)]

// ============================================================================
// Top-level module declarations
//
// Declared in dependency order: each module may only depend on modules
// declared ABOVE it in this list.
//
//   common    → (no BCC dependencies — only std)
//   frontend  → common
//   ir        → common, frontend
//   passes    → common, ir
//   backend   → common, ir, (optionally frontend for inline asm)
// ============================================================================

/// Infrastructure layer — foundational types imported by all other modules.
///
/// Provides FxHash for performant hash maps, PUA/UTF-8 encoding for non-UTF-8
/// source byte round-tripping, the dual type system (`CType` + `MachineType`),
/// the diagnostic reporting engine, source file tracking, string interning,
/// target architecture definitions, long-double software arithmetic, temporary
/// file management, and the type builder API.
pub mod common;

/// C11 frontend pipeline with GCC extension support.
///
/// Contains the preprocessor (Phases 1–2 with paint-marker recursion
/// protection), lexer (Phase 3 with PUA-aware scanning), recursive-descent
/// parser (Phase 4 with GCC extensions, inline assembly, attributes), and
/// semantic analyzer (Phase 5 with type checking, scope management, builtin
/// evaluation, initializer analysis).
///
/// Depends on [`common`] but does NOT depend on [`ir`], [`passes`], or
/// [`backend`].
pub mod frontend;

/// Intermediate representation, lowering, and SSA construction.
///
/// Defines the IR instruction set, basic blocks, functions, and modules.
/// Implements AST-to-IR lowering (Phase 6 with alloca-first strategy), SSA
/// construction via mem2reg (Phase 7 using dominance frontiers), and phi-node
/// elimination (Phase 9).
///
/// Depends on [`common`] and [`frontend`] (for AST types consumed during
/// lowering).
pub mod ir;

/// Optimization passes (Phase 8).
///
/// Provides a pass manager with fixpoint iteration driving three core passes:
/// constant folding and propagation, dead code elimination, and control-flow
/// graph simplification (unreachable block removal, branch threading).
///
/// Depends on [`common`] and [`ir`]. Does NOT depend on [`frontend`] or
/// [`backend`].
pub mod passes;

/// Code generation backend (Phase 10).
///
/// Implements the `ArchCodegen` trait abstraction, architecture-dispatching
/// code generation driver, register allocator, common ELF writing, DWARF v4
/// debug information generation, shared linker infrastructure with dynamic
/// linking, and four architecture-specific backends (x86-64, i686, AArch64,
/// RISC-V 64) — each with its own built-in assembler and built-in linker.
///
/// Depends on [`common`], [`ir`], and optionally [`frontend`] (for inline
/// assembly processing). Does NOT depend on [`passes`].
pub mod backend;

// ============================================================================
// Key type re-exports
//
// Re-export the most commonly used types at the crate root for convenience.
// This allows `use bcc::Target` instead of `use bcc::common::target::Target`,
// reducing import verbosity in main.rs and integration tests.
// ============================================================================

/// Re-export [`common::target::Target`] — the target architecture enum
/// (X86_64, I686, AArch64, RiscV64) used throughout the entire pipeline.
pub use common::target::Target;

/// Re-export [`common::diagnostics::DiagnosticEngine`] — the multi-error
/// diagnostic reporting engine that accumulates errors, warnings, and notes
/// from all pipeline stages.
pub use common::diagnostics::DiagnosticEngine;

/// Re-export [`common::types::CType`] — the C language type enum representing
/// all C11 types plus GCC extensions (Void, Bool, Char, Int, Pointer, Array,
/// Struct, Union, Enum, Function, Atomic, Typedef, Qualified, etc.).
pub use common::types::CType;

/// Re-export [`common::source_map::SourceMap`] — source file tracking with
/// file IDs, line/column lookups, and `#line` directive remapping.
pub use common::source_map::SourceMap;

/// Re-export [`common::string_interner::Interner`] — string interning for
/// identifiers, keywords, and string literals with O(1) Symbol handle
/// comparison.
pub use common::string_interner::Interner;

/// Re-export [`common::fx_hash::FxHashMap`] — fast, non-cryptographic
/// `HashMap` using FxHash (Fibonacci hashing) for performance-critical
/// compiler-internal data structures.
pub use common::fx_hash::FxHashMap;

/// Re-export [`common::fx_hash::FxHashSet`] — fast, non-cryptographic
/// `HashSet` using FxHash (Fibonacci hashing) for performance-critical
/// compiler-internal data structures.
pub use common::fx_hash::FxHashSet;
