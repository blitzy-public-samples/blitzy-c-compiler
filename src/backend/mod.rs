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

/// DWARF v4 debug information generation — `.debug_info`, `.debug_abbrev`,
/// `.debug_line`, and `.debug_str` section builders.
pub mod dwarf;
