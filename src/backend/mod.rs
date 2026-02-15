//! Backend module — code generation, assemblers, linkers, and debug information.
//!
//! This module contains the Phase 10 code generation pipeline and all supporting
//! infrastructure for producing ELF binaries from the compiler's intermediate
//! representation. The backend is organized into:
//!
//! - Architecture-specific code generators implementing the `ArchCodegen` trait
//! - Built-in assemblers for each target architecture
//! - Built-in linkers producing ET_EXEC and ET_DYN ELF binaries
//! - DWARF v4 debug information generation (conditional on `-g`)
//! - Common ELF writing infrastructure
//! - Register allocation
//!
//! # Supported Architectures
//!
//! - x86-64 (System V AMD64 ABI)
//! - i686 (cdecl/System V i386 ABI)
//! - AArch64 (AAPCS64 ABI)
//! - RISC-V 64 (LP64D ABI)
//!
//! # Standalone Backend
//!
//! BCC includes its own assembler and linker — no external toolchain components
//! (`as`, `ld`, `gcc`, `llvm-mc`, `lld`) are invoked during compilation.

/// `ArchCodegen` trait definition — the central polymorphism point that enables
/// uniform code generation across x86-64, i686, AArch64, and RISC-V 64.
/// Also defines machine IR types (`MachineFunction`, `MachineBasicBlock`,
/// `MachineInstr`, `MachineOperand`), ABI classification (`ParamClass`),
/// register metadata (`PhysReg`, `RegisterClass`, `RegisterInfo`),
/// relocation descriptors, and code generation configuration.
pub mod traits;

/// DWARF v4 debug information generation — `.debug_info`, `.debug_abbrev`,
/// `.debug_line`, and `.debug_str` section builders.
pub mod dwarf;

/// Common ELF writing infrastructure — section headers, program headers,
/// symbol tables, string tables, and ELF file serialization for all four
/// target architectures. Supports ET_REL, ET_EXEC, and ET_DYN output.
pub mod elf_writer_common;

/// Shared linker infrastructure — symbol resolution, section merging,
/// relocation processing, dynamic linking section generation, and default
/// linker script handling. Used by all four architecture-specific linkers
/// as common linking infrastructure for the standalone backend mode.
pub mod linker_common;

/// x86-64 (AMD64) backend — `ArchCodegen` trait implementation, instruction
/// selection, built-in assembler, and built-in linker for the x86-64 target.
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
/// RISC-V 64 target with linker relaxation support.
pub mod riscv64;

/// Linear scan register allocator — assigns physical machine registers to
/// SSA virtual registers across all four target architectures. Computes live
/// intervals, performs greedy allocation with furthest-next-use spilling, and
/// generates spill/reload pseudo-instructions for the prologue/epilogue pass.
pub mod register_allocator;

/// Phase 10 code generation driver — architecture dispatch, security mitigation
/// injection, object file emission, and DWARF debug info coordination. This
/// module is the primary entry point for converting IR to machine code.
pub mod generation;
