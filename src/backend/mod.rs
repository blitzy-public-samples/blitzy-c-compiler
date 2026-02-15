//! Backend module — code generation, assemblers, linkers, and debug information.
//!
//! This module contains the Phase 10 code generation pipeline and all supporting
//! infrastructure for producing ELF binaries from the compiler's intermediate
//! representation. The backend is organized into shared infrastructure and
//! four architecture-specific code generators, each containing instruction
//! selection, register definitions, ABI implementation, a built-in assembler,
//! and a built-in ELF linker.
//!
//! # Module Organization
//!
//! ## Shared Infrastructure
//!
//! - [`traits`] — [`ArchCodegen`] trait (the architecture abstraction layer),
//!   machine IR types ([`MachineFunction`], [`MachineBasicBlock`], [`MachineInstr`],
//!   [`MachineOperand`]), physical register identifier ([`PhysReg`]), and ABI
//!   classification ([`ParamClass`]).
//! - [`generation`] — Phase 10 code generation driver: architecture dispatch,
//!   security mitigation injection, object file emission, DWARF coordination.
//!   Provides [`CodegenConfig`], [`OutputMode`], and the [`generate_code`]
//!   entry point.
//! - [`register_allocator`] — Linear scan register allocator: live interval
//!   computation, greedy register assignment, spill code generation. Provides
//!   [`RegisterAllocator`], [`LiveInterval`], and [`RegisterSet`].
//! - [`elf_writer_common`] — Common ELF binary writing infrastructure: section
//!   headers, program headers, symbol tables, string tables, and ELF file
//!   serialization for all four target architectures. Provides [`ElfWriter`],
//!   [`ElfSection`], and [`ElfSymbol`].
//! - [`linker_common`] — Shared linker infrastructure for symbol resolution,
//!   section merging, relocation processing, dynamic linking section generation
//!   (`.dynamic`, `.dynsym`, `.rela.dyn`, `.rela.plt`, `.gnu.hash`), and
//!   default linker script handling.
//! - [`dwarf`] — DWARF v4 debug information generation: `.debug_info`,
//!   `.debug_abbrev`, `.debug_line`, and `.debug_str` section builders.
//!   Conditionally active when the `-g` flag is set; produces zero debug
//!   sections when inactive (Section 0.7.10).
//!
//! ## Architecture-Specific Backends
//!
//! Each backend implements the [`ArchCodegen`] trait and includes its own
//! assembler and linker (standalone backend mode — Section 0.7.7):
//!
//! - [`x86_64`] — x86-64 (AMD64) with System V AMD64 ABI, 16 GPRs + 16 SSE
//!   registers, variable-length instruction encoding, REX prefix handling,
//!   and security mitigations (retpoline, CET/IBT, stack probe).
//! - [`i686`] — i686 (IA-32) with cdecl/System V i386 ABI, 8 GPRs, 32-bit
//!   instruction encoding without REX prefixes.
//! - [`aarch64`] — AArch64 (ARM 64-bit) with AAPCS64 ABI, 31 GPRs + 32
//!   SIMD/FP registers, fixed 32-bit instruction width.
//! - [`riscv64`] — RISC-V 64 with LP64D ABI, 32 integer + 32 FP registers,
//!   RV64IMAFDC ISA, linker relaxation support. Primary target for Linux
//!   kernel 6.9 boot validation (Checkpoint 6).
//!
//! # Supported Architectures
//!
//! | Architecture | ABI         | Ptr Width | Endian   | ELF Machine |
//! |-------------|-------------|-----------|----------|------------|
//! | x86-64      | System V AMD64 | 8 bytes | Little | EM_X86_64 |
//! | i686        | cdecl/SysV i386 | 4 bytes | Little | EM_386    |
//! | AArch64     | AAPCS64     | 8 bytes   | Little   | EM_AARCH64 |
//! | RISC-V 64   | LP64D       | 8 bytes   | Little   | EM_RISCV  |
//!
//! # Standalone Backend
//!
//! BCC includes its own assembler and linker — no external toolchain components
//! (`as`, `ld`, `gcc`, `llvm-mc`, `lld`) are invoked during compilation.
//! This standalone mode is a non-negotiable architectural constraint
//! (Section 0.7.7) that overrides any assumption about external tools.
//!
//! # Backend Validation Order
//!
//! Per Section 0.1.2, backends are validated in a fixed order:
//! **x86-64 → i686 → AArch64 → RISC-V 64**. This order reflects
//! implementation priority and testing confidence levels.
//!
//! # Dependencies
//!
//! Backend modules depend on:
//! - `crate::common::` — types, target definitions, diagnostics, FxHash
//! - `crate::ir::` — IR types, functions, modules (input to code generation)
//! - `crate::frontend::` — optionally, for inline assembly support

// ===========================================================================
// Submodule declarations — all 10 backend submodules
// ===========================================================================

/// `ArchCodegen` trait definition — the central polymorphism point that enables
/// uniform code generation across x86-64, i686, AArch64, and RISC-V 64.
/// Also defines machine IR types (`MachineFunction`, `MachineBasicBlock`,
/// `MachineInstr`, `MachineOperand`), ABI classification (`ParamClass`),
/// register metadata (`PhysReg`, `RegisterClass`, `RegisterInfo`),
/// relocation descriptors, and code generation configuration.
pub mod traits;

/// Phase 10 code generation driver — architecture dispatch, security mitigation
/// injection, object file emission, and DWARF debug info coordination. This
/// module is the primary entry point for converting IR to machine code.
pub mod generation;

/// Linear scan register allocator — assigns physical machine registers to
/// SSA virtual registers across all four target architectures. Computes live
/// intervals, performs greedy allocation with furthest-next-use spilling, and
/// generates spill/reload pseudo-instructions for the prologue/epilogue pass.
pub mod register_allocator;

/// Common ELF writing infrastructure — section headers, program headers,
/// symbol tables, string tables, and ELF file serialization for all four
/// target architectures. Supports ET_REL, ET_EXEC, and ET_DYN output.
pub mod elf_writer_common;

/// Shared linker infrastructure — symbol resolution, section merging,
/// relocation processing, dynamic linking section generation, and default
/// linker script handling. Used by all four architecture-specific linkers
/// as common linking infrastructure for the standalone backend mode.
pub mod linker_common;

/// DWARF v4 debug information generation — `.debug_info`, `.debug_abbrev`,
/// `.debug_line`, and `.debug_str` section builders.
pub mod dwarf;

/// x86-64 (AMD64) backend — `ArchCodegen` trait implementation, instruction
/// selection, built-in assembler, and built-in linker for the x86-64 target.
/// Includes security mitigations: retpoline thunks (`-mretpoline`),
/// CET/IBT `endbr64` (`-fcf-protection`), and stack guard page probing
/// for stack frames exceeding 4,096 bytes.
pub mod x86_64;

/// i686 (IA-32) backend — `ArchCodegen` trait implementation, instruction
/// selection, built-in assembler, and built-in linker for the 32-bit x86
/// target with cdecl/System V i386 ABI conformance.
pub mod i686;

/// AArch64 (ARM 64-bit) backend — `ArchCodegen` trait implementation,
/// instruction selection, built-in assembler, and built-in linker for the
/// AArch64 target with AAPCS64 ABI conformance.
pub mod aarch64;

/// RISC-V 64 (RV64IMAFDC) backend — `ArchCodegen` trait implementation,
/// instruction selection, built-in assembler, and built-in linker for the
/// RISC-V 64 target with linker relaxation support. This is the primary
/// target for Linux kernel 6.9 boot validation (Checkpoint 6).
pub mod riscv64;

// ===========================================================================
// Convenience re-exports — ergonomic access via `crate::backend::*`
// ===========================================================================
//
// These re-exports allow consuming modules (e.g., `src/main.rs`, test code)
// to write `use crate::backend::ArchCodegen` instead of the longer
// `use crate::backend::traits::ArchCodegen`. Only the most commonly
// referenced types are re-exported here.

// --- From traits: Core backend abstractions ---
pub use traits::{
    ArchCodegen, MachineBasicBlock, MachineFunction, MachineInstr, MachineOperand, ParamClass,
    PhysReg,
};

// --- From generation: Pipeline entry point and configuration ---
pub use generation::{generate_code, CodegenConfig, OutputMode};

// --- From register_allocator: Register allocation infrastructure ---
pub use register_allocator::{LiveInterval, RegisterAllocator, RegisterSet};

// --- From elf_writer_common: ELF binary output types ---
pub use elf_writer_common::{ElfSection, ElfSymbol, ElfWriter};
