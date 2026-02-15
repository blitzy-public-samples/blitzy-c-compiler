//! BCC — Blitzy's C Compiler Library
//!
//! A complete, self-contained, zero-external-dependency C compilation toolchain
//! implemented in Rust (2021 Edition) that cross-compiles C source code into
//! native Linux ELF executables and shared objects for four target architectures:
//! **x86-64**, **i686**, **AArch64**, and **RISC-V 64**.
//!
//! # Architecture
//!
//! The compiler is organized as a ten-phase pipeline, grouped into five
//! top-level modules:
//!
//! ## [`common`] — Infrastructure Layer
//!
//! Foundational utilities imported by every other layer in the pipeline:
//!
//! - [`common::fx_hash`] — Fast non-cryptographic Fibonacci hashing
//!   ([`FxHashMap`], [`FxHashSet`]) for symbol tables and lookup maps.
//! - [`common::encoding`] — PUA/UTF-8 encoding for non-UTF-8 byte
//!   round-tripping (U+E080–U+E0FF), ensuring byte-exact fidelity of
//!   binary data in C string literals and inline assembly.
//! - [`common::long_double`] — Software 80-bit IEEE 754 extended-precision
//!   arithmetic without external math libraries.
//! - [`common::temp_files`] — RAII-based temporary file and directory
//!   management with automatic cleanup via [`Drop`].
//! - [`common::types`] — Dual type system: [`CType`] for C language types
//!   and [`MachineType`] for backend register-class mappings.
//! - [`common::type_builder`] — Builder-pattern API for constructing complex
//!   C types and computing struct/union layouts.
//! - [`common::diagnostics`] — Multi-error diagnostic reporting engine with
//!   source spans, severity levels, and GCC-compatible output formatting.
//! - [`common::source_map`] — Source file tracking with line offset tables
//!   for O(log n) lookups and `#line` directive remapping.
//! - [`common::string_interner`] — String interning with FxHash-backed
//!   deduplication; [`Symbol`] handles for zero-cost identifier comparison.
//! - [`common::target`] — Target architecture definitions ([`Target`]) for
//!   x86-64, i686, AArch64, and RISC-V 64 with pointer widths, endianness,
//!   predefined macros, and data models.
//!
//! ## [`frontend`] — Phases 1–5
//!
//! The complete C language frontend, organized into four submodules:
//!
//! - [`frontend::preprocessor`] — **Phases 1–2**: Trigraph replacement,
//!   line splicing, `#include` resolution, macro expansion with paint-marker
//!   recursion protection for self-referential macros.
//! - [`frontend::lexer`] — **Phase 3**: PUA-aware tokenization producing
//!   [`Token`] streams with keyword recognition and literal parsing.
//! - [`frontend::parser`] — **Phase 4**: Recursive-descent C11 parser with
//!   comprehensive GCC extension support (statement expressions, `typeof`,
//!   computed gotos, case ranges, `__attribute__`, inline assembly).
//! - [`frontend::sema`] — **Phase 5**: Type checking, scope management,
//!   symbol table construction, constant evaluation, GCC builtin handling,
//!   initializer analysis, and attribute validation.
//!
//! ## [`ir`] — Phases 6, 7, and 9
//!
//! Intermediate representation and SSA transformations:
//!
//! - [`ir::lowering`] — **Phase 6**: AST-to-IR lowering with alloca-first
//!   insertion for all local variables (the "alloca" half of
//!   alloca-then-promote).
//! - [`ir::mem2reg`] — **Phase 7**: SSA construction via dominance frontier
//!   computation, phi-node insertion, and alloca promotion (the "promote"
//!   half). Also contains **Phase 9**: phi-node elimination, converting SSA
//!   back to copy operations for register allocation.
//!
//! ## [`passes`] — Phase 8
//!
//! Optimization passes that preserve SSA invariants:
//!
//! - [`passes::constant_folding`] — Compile-time constant evaluation and
//!   propagation.
//! - [`passes::dead_code_elimination`] — Removal of unused, side-effect-free
//!   instructions and unreachable blocks.
//! - [`passes::simplify_cfg`] — Block merging, empty-block elimination,
//!   and branch chain simplification.
//! - [`passes::pass_manager`] — Fixed-order pass scheduling with fixpoint
//!   iteration via [`PassManager`].
//!
//! ## [`backend`] — Phase 10
//!
//! Code generation, built-in assemblers, built-in linkers, and DWARF:
//!
//! - [`backend::traits`] — [`ArchCodegen`] trait (architecture abstraction
//!   layer) and machine IR types.
//! - [`backend::generation`] — Code generation driver: architecture dispatch,
//!   security mitigation injection, object file emission.
//! - [`backend::register_allocator`] — Linear scan register allocator.
//! - [`backend::elf_writer_common`] — Common ELF binary writing for all
//!   four targets.
//! - [`backend::linker_common`] — Shared linker infrastructure: symbol
//!   resolution, section merging, relocations, dynamic linking.
//! - [`backend::dwarf`] — DWARF v4 debug information (conditional on `-g`).
//! - [`backend::x86_64`] — x86-64 with System V AMD64 ABI, retpoline,
//!   CET/IBT, and stack probe mitigations.
//! - [`backend::i686`] — i686 with cdecl/System V i386 ABI.
//! - [`backend::aarch64`] — AArch64 with AAPCS64 ABI.
//! - [`backend::riscv64`] — RISC-V 64 with LP64D ABI (Linux kernel 6.9
//!   boot target).
//!
//! # Pipeline Data Flow
//!
//! ```text
//! Source File (.c)
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  frontend::preprocessor  (Phase 1–2)                    │
//! │    Trigraphs → line splicing → #include → #define →     │
//! │    macro expansion (paint-marker protected)             │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ expanded source text
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  frontend::lexer  (Phase 3)                             │
//! │    PUA-aware tokenization → Token stream                │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ Token stream
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  frontend::parser  (Phase 4)                            │
//! │    Recursive-descent C11 + GCC extensions → AST         │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ TranslationUnit (AST)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  frontend::sema  (Phase 5)                              │
//! │    Type checking, scopes, constants, builtins, attrs    │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ CheckedTranslationUnit
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  ir::lowering  (Phase 6)                                │
//! │    AST → IR (alloca-first for all locals)               │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ IrModule (with allocas)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  ir::mem2reg  (Phase 7)                                 │
//! │    Dominance frontiers → phi insertion → SSA promotion  │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ IrModule (SSA form)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  passes::pass_manager  (Phase 8)                        │
//! │    Constant folding → DCE → CFG simplification          │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ IrModule (optimized SSA)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  ir::mem2reg::phi_eliminate  (Phase 9)                  │
//! │    Phi nodes → parallel copies → sequentialized copies  │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │ IrModule (phi-eliminated)
//!                        ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │  backend::generation  (Phase 10)                        │
//! │    Target dispatch → instruction selection →            │
//! │    register allocation → assembler → linker → ELF      │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │
//!                        ▼
//!                   Output: .o / ELF / .so
//! ```
//!
//! # Zero-Dependency Mandate
//!
//! BCC uses **no external crates**. The `[dependencies]` section of
//! `Cargo.toml` is empty. All functionality — hashing, encoding, math
//! operations, ELF writing, DWARF emission, assemblers, and linkers — is
//! implemented internally using only the Rust standard library (`std`).
//!
//! # Supported Targets
//!
//! | Architecture | ABI              | Ptr Width | Endian | ELF Machine  |
//! |-------------|------------------|-----------|--------|-------------|
//! | x86-64      | System V AMD64   | 8 bytes   | Little | `EM_X86_64` |
//! | i686        | cdecl/SysV i386  | 4 bytes   | Little | `EM_386`    |
//! | AArch64     | AAPCS64          | 8 bytes   | Little | `EM_AARCH64`|
//! | RISC-V 64   | LP64D            | 8 bytes   | Little | `EM_RISCV`  |
//!
//! # Usage
//!
//! Integration tests and the main driver (`src/main.rs`) access the library
//! via `use bcc::*` or targeted imports such as `use bcc::common::Target`.
//! The most frequently used types are re-exported at the crate root for
//! ergonomic access.

// ═══════════════════════════════════════════════════════════════════════════════
// Crate-level lint configuration
// ═══════════════════════════════════════════════════════════════════════════════
//
// These suppressions address patterns that are idiomatic and unavoidable in
// compiler implementations. Each is deliberately chosen:

/// Compiler pipeline functions naturally accumulate many parameters
/// (target, options, diagnostics, source map, interner, etc.).
#![allow(clippy::too_many_arguments)]

/// AST and IR enums have inherently varying-size variants — a
/// `FunctionDef` is far larger than a `Break` statement, and forcing
/// uniform boxing would add needless indirection.
#![allow(clippy::large_enum_variant)]

/// Module inception (e.g., `parser::parser`) is occasionally useful
/// when a module's primary type shares its name.
#![allow(clippy::module_inception)]

/// Compiler type representations are inherently complex — deeply nested
/// generics for type-safe IR manipulation are expected and correct.
#![allow(clippy::type_complexity)]

// ═══════════════════════════════════════════════════════════════════════════════
// Top-level module declarations
// ═══════════════════════════════════════════════════════════════════════════════
//
// These five modules form the complete BCC compilation pipeline. The import
// dependency order is strictly layered:
//
//   common  (no internal deps)
//     ↑
//   frontend  (depends on common)
//     ↑
//   ir  (depends on common, frontend)
//     ↑
//   passes  (depends on common, ir)
//     ↑
//   backend  (depends on common, ir, optionally frontend for inline asm)

/// Common infrastructure layer — foundational utilities imported by every
/// other module in the compiler. Provides fast hashing ([`FxHashMap`],
/// [`FxHashSet`]), PUA encoding, type system ([`CType`], [`MachineType`]),
/// diagnostics ([`Diagnostic`], [`DiagnosticEngine`]), source tracking
/// ([`SourceMap`]), string interning ([`Interner`], [`Symbol`]), and target
/// architecture definitions ([`Target`]).
pub mod common;

/// Frontend pipeline — Phases 1–5 of the compilation pipeline:
/// preprocessor (trigraphs, line splicing, `#include`, `#define`, macro
/// expansion with paint-marker recursion protection), lexer (PUA-aware
/// tokenization), parser (recursive-descent C11 with GCC extensions), and
/// semantic analyzer (type checking, scopes, builtins, attributes).
/// Depends on [`common`] but not on [`ir`] or [`backend`].
pub mod frontend;

/// Intermediate representation — Phases 6 (AST-to-IR lowering with
/// alloca-first insertion), 7 (SSA construction via alloca promotion /
/// mem2reg), and 9 (phi elimination). Depends on [`common`] and
/// [`frontend`] (for AST types during lowering). Consumed by [`passes`]
/// (optimization) and [`backend`] (code generation).
pub mod ir;

/// Optimization passes — Phase 8 of the compilation pipeline. Provides
/// constant folding, dead code elimination, and CFG simplification in a
/// fixed-order pipeline with fixpoint iteration. All passes preserve SSA
/// invariants. Depends on [`common`] and [`ir`].
pub mod passes;

/// Backend — Phase 10: code generation, built-in assemblers, built-in
/// linkers, and DWARF v4 debug information generation for all four target
/// architectures (x86-64, i686, AArch64, RISC-V 64). Operates in
/// standalone mode — no external toolchain components are invoked.
/// Depends on [`common`], [`ir`], and optionally [`frontend`] for inline
/// assembly support.
pub mod backend;

// ═══════════════════════════════════════════════════════════════════════════════
// Convenience re-exports — key types at the crate root
// ═══════════════════════════════════════════════════════════════════════════════
//
// These re-exports allow consumers (main.rs, integration tests) to write
// ergonomic imports such as:
//
//   use bcc::{Target, CType, IrModule, PassManager, ArchCodegen};
//
// instead of fully qualified paths:
//
//   use bcc::common::target::Target;
//   use bcc::ir::module::IrModule;
//   use bcc::passes::pass_manager::PassManager;
//   use bcc::backend::traits::ArchCodegen;
//
// Only the most frequently used types are re-exported here. Consumers
// needing less common types should import from the appropriate submodule.

// ── From common: foundational infrastructure types ─────────────────────────

/// Target architecture enum — x86-64, i686, AArch64, RISC-V 64.
/// Flows through every pipeline stage from CLI parsing to code generation.
pub use common::Target;

/// C language type representation — the frontend's view of types (Void,
/// Bool, Char, Int, Pointer, Struct, Union, etc.) with qualifiers.
pub use common::CType;

/// Machine type representation — the backend's register-class view of
/// types, used during instruction selection and register allocation.
pub use common::MachineType;

/// Builder-pattern API for constructing complex C types and computing
/// struct/union layouts with packed/aligned attribute support.
pub use common::TypeBuilder;

/// Single diagnostic message with severity, source span, message text,
/// and optional fix suggestion.
pub use common::Diagnostic;

/// Diagnostic collection engine — accumulates errors, warnings, and notes
/// across all pipeline stages with formatted output.
pub use common::DiagnosticEngine;

/// Source file registry — maps file IDs to contents, provides O(log n)
/// byte-offset-to-line/column lookups, handles `#line` remapping.
pub use common::SourceMap;

/// String interner — stores each unique string once, returning compact
/// [`Symbol`] handles for zero-cost comparison and hashing.
pub use common::Interner;

/// Interned string handle — a lightweight 4-byte identifier supporting
/// `Copy`, `Eq`, and `Hash` for use in symbol tables and AST nodes.
pub use common::Symbol;

/// High-performance hash map using Fibonacci hashing (FxHash) instead of
/// the default SipHash. Used throughout the compiler for symbol tables,
/// scope lookups, and all performance-critical map operations.
pub use common::FxHashMap;

/// High-performance hash set using Fibonacci hashing (FxHash). Used for
/// include guard tracking, visited-set algorithms, and deduplication.
pub use common::FxHashSet;

/// RAII temporary file with automatic cleanup on drop. Used for
/// intermediate object files during multi-file compilation.
pub use common::TempFile;

// ── From ir: intermediate representation types ─────────────────────────────

/// Top-level IR container for a compilation unit — holds global variables,
/// function definitions, external declarations, string literal pool, and
/// module-level inline assembly blocks.
pub use ir::IrModule;

/// Function-level IR container — holds basic blocks (CFG), SSA value
/// registry, parameter list, calling convention, and function attributes.
pub use ir::IrFunction;

/// IR type system — bridges C language types from the frontend to machine
/// register classes in the backend (Void, I1, I8, I16, I32, I64, I128,
/// F32, F64, F80, Ptr, Array, Struct, Function).
pub use ir::IrType;

/// IR construction API — provides typed instruction creation methods with
/// insertion point tracking and automatic SSA value numbering.
pub use ir::IrBuilder;

// ── From passes: optimization infrastructure ───────────────────────────────

/// Pass scheduling and execution framework — runs the fixed-order
/// optimization pipeline (constant folding → DCE → CFG simplification)
/// with fixpoint iteration until no pass reports changes.
pub use passes::PassManager;

// ── From backend: code generation infrastructure ───────────────────────────

/// Architecture abstraction trait — the central polymorphism point enabling
/// uniform code generation across x86-64, i686, AArch64, and RISC-V 64.
/// Each target backend implements this trait.
pub use backend::ArchCodegen;

/// Code generation entry point — dispatches to the appropriate architecture
/// backend, injects security mitigations, coordinates assembler and linker
/// invocation, and produces final ELF output.
pub use backend::generate_code;

/// Code generation configuration — target architecture, output mode,
/// optimization level, debug info flag, PIC mode, security mitigation
/// flags, and all other codegen-relevant settings.
pub use backend::CodegenConfig;
