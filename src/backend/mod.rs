//! # BCC Backend — Code Generation, Assemblers, Linkers, DWARF, ELF
//!
//! This module implements the complete backend pipeline for BCC, the Blitzy C
//! Compiler.  It transforms phi-eliminated IR (from Phase 9) into native ELF
//! binaries for four target architectures.
//!
//! ## Core Infrastructure
//!
//! - [`traits`] — `ArchCodegen` trait: the architecture abstraction layer that
//!   every backend must implement.  Defines the interface for instruction
//!   selection, register information, ABI classification, and machine code
//!   emission.
//! - [`generation`] — Phase 10 code generation driver with architecture
//!   dispatch.  Routes compilation to the correct backend based on the
//!   `--target` flag and injects security mitigations for x86-64.
//! - [`register_allocator`] — Linear scan register allocator.  Computes live
//!   intervals, assigns physical registers, and generates spill code.
//!   Architecture-parameterized via [`RegisterInfo`].
//! - [`elf_writer_common`] — Common ELF binary format writing infrastructure.
//!   Hand-implements all ELF construction (headers, sections, symbols, program
//!   headers) in compliance with the zero-dependency mandate.
//!
//! ## Shared Infrastructure
//!
//! - [`linker_common`] — Shared linker infrastructure (symbol resolution,
//!   section merging, relocation processing, dynamic linking support, and
//!   default linker script handling).  Used by all four architecture linkers.
//! - [`dwarf`] — DWARF v4 debug information generation (`.debug_info`,
//!   `.debug_abbrev`, `.debug_line`, `.debug_str`).  Conditional on the `-g`
//!   flag — a binary compiled without `-g` contains zero `.debug_*` sections.
//!
//! ## Architecture Backends
//!
//! Each backend implements the [`ArchCodegen`] trait and includes its own
//! built-in assembler and built-in linker:
//!
//! - [`x86_64`] — x86-64 backend (primary validation target): codegen,
//!   assembler, linker, and security mitigations (retpoline, CET/IBT, stack
//!   guard-page probing).  System V AMD64 ABI with 16 GPRs + 16 SSE
//!   registers.
//! - [`i686`] — i686 (32-bit x86) backend: codegen, assembler, linker.
//!   cdecl ABI with 8 GPRs and x87 FPU.
//! - [`aarch64`] — AArch64 (ARM 64-bit) backend: codegen, assembler, linker.
//!   AAPCS64 ABI with 31 GPRs + 32 SIMD/FP registers, fixed 32-bit
//!   instruction width.
//! - [`riscv64`] — RISC-V 64 backend (kernel boot target): codegen,
//!   assembler, linker.  LP64D ABI with RV64IMAFDC ISA, 32 integer + 32 FP
//!   registers.
//!
//! ## Standalone Backend Mode
//!
//! BCC includes its own built-in assembler and linker for ALL four target
//! architectures.  No external toolchain invocation occurs — no `as`, `ld`,
//! `gcc`, `llvm-mc`, or `lld`.  Every step from instruction selection through
//! final ELF binary production is performed internally.
//!
//! ## Backend Validation Order
//!
//! Backends are validated in a fixed order:
//! 1. x86-64 (primary)
//! 2. i686
//! 3. AArch64
//! 4. RISC-V 64 (kernel boot)
//!
//! ## Dependencies
//!
//! - Depends on `crate::common` (target, types, diagnostics, FxHash, encoding)
//! - Depends on `crate::ir` (IR module, functions, instructions consumed by
//!   codegen)
//! - Optionally depends on `crate::frontend` (for inline assembly processing)
//! - Does **NOT** depend on `crate::passes`
//!
//! ## Zero-Dependency Mandate
//!
//! This module and all submodules use only the Rust standard library (`std`)
//! and `crate::` internal references.  No external crates are used anywhere
//! in the backend.

// ===========================================================================
// Core backend infrastructure
// ===========================================================================

/// `ArchCodegen` trait — the architecture abstraction layer.
///
/// Defines the interface that every architecture backend must implement:
/// instruction selection, register information, ABI classification,
/// prologue/epilogue emission, and machine code encoding.
pub mod traits;

/// Phase 10 — code generation driver.
///
/// Central dispatch point that routes compilation to the correct architecture
/// backend based on the `--target` flag.  Injects security mitigations
/// (retpoline, CET/IBT, stack probe) for x86-64 and orchestrates the full
/// pipeline from IR to final ELF binary.
pub mod generation;

/// Linear scan register allocator.
///
/// Computes live intervals for virtual registers, assigns physical registers
/// using a linear scan algorithm, and generates spill code for values that
/// cannot fit in the physical register file.  Fully parameterized by
/// `RegisterInfo` for architecture independence.
pub mod register_allocator;

/// Common ELF binary format writing infrastructure.
///
/// Hand-implements all ELF construction: file headers (ELF32/ELF64), section
/// header tables, program header tables, string tables (`.strtab`,
/// `.shstrtab`), symbol tables (`.symtab`), and relocation sections.
/// Supports ET_REL, ET_EXEC, and ET_DYN output for all four architectures.
pub mod elf_writer_common;

// ===========================================================================
// Shared infrastructure
// ===========================================================================

/// Shared linker infrastructure.
///
/// Provides architecture-agnostic linker components used by all four
/// architecture-specific linkers: two-pass symbol resolution, input section
/// merging, relocation processing framework, dynamic linking section
/// generation (`.dynamic`, `.dynsym`, `.gnu.hash`, GOT, PLT), and default
/// linker script handling for section-to-segment mapping.
pub mod linker_common;

/// DWARF v4 debug information generation.
///
/// Generates `.debug_info`, `.debug_abbrev`, `.debug_line`, and `.debug_str`
/// sections when the `-g` flag is active (at `-O0` only).  A binary compiled
/// without `-g` contains zero `.debug_*` sections — no debug leakage.
pub mod dwarf;

// ===========================================================================
// Architecture-specific backends
// ===========================================================================

/// x86-64 backend — primary validation target.
///
/// Implements `ArchCodegen` for the x86-64 architecture with System V AMD64
/// ABI.  Features 16 GPRs (RAX–R15), 16 SSE registers (XMM0–XMM15),
/// variable-length instruction encoding with REX prefix support, and
/// security mitigations (retpoline, CET/IBT `endbr64`, stack guard-page
/// probing for frames > 4096 bytes).
pub mod x86_64;

/// i686 (32-bit x86) backend.
///
/// Implements `ArchCodegen` for the i686 architecture with cdecl/System V
/// i386 ABI.  Features 8 GPRs (EAX–EDI), x87 FPU stack for floating-point,
/// all arguments passed on the stack, and 32-bit instruction encoding
/// without REX prefixes.
pub mod i686;

/// AArch64 (ARM 64-bit) backend.
///
/// Implements `ArchCodegen` for the AArch64 architecture with AAPCS64 ABI.
/// Features 31 GPRs (X0–X30), 32 SIMD/FP registers (V0–V31), fixed 32-bit
/// instruction width, and ADRP+ADD pairs for PIC addressing.
pub mod aarch64;

/// RISC-V 64 backend — kernel boot target.
///
/// Implements `ArchCodegen` for the RISC-V 64 architecture with LP64D ABI
/// and RV64IMAFDC ISA.  Features 32 integer registers (x0–x31), 32 FP
/// registers (f0–f31), R/I/S/B/U/J instruction formats, and linker
/// relaxation support.  Primary target for Linux kernel 6.9 build and
/// QEMU boot validation (Checkpoint 6).
pub mod riscv64;

// ===========================================================================
// Key type re-exports for crate-wide convenience access
// ===========================================================================

// From traits module: the core architecture abstraction types
pub use traits::{ArchCodegen, MachineFunction, MachineInstruction, MachineOperand, RegisterInfo};

// From generation module: compilation context carrying CLI flags
pub use generation::CodegenContext;

// From elf_writer_common module: top-level ELF binary writer
pub use elf_writer_common::ElfWriter;
