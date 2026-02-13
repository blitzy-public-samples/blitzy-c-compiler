//! Built-in i686 (32-bit x86) ELF linker module.
//!
//! This module implements the integrated linker for the i686 target
//! architecture, producing ET_EXEC static executables and ET_DYN shared
//! objects without invoking any external linker (`ld`, `i686-linux-gnu-ld`,
//! `lld`). All linking is fully self-contained per the standalone backend
//! mandate.
//!
//! # Sub-modules
//!
//! - [`relocations`]: i386 ELF relocation type definitions and the
//!   [`I686RelocationHandler`](relocations::I686RelocationHandler)
//!   implementing the [`ArchRelocationHandler`](crate::backend::linker_common::relocation::ArchRelocationHandler)
//!   trait for all `R_386_*` relocation types with proper 32-bit overflow
//!   checking and implicit (REL-style) addend support.
//!
//! # ELF Configuration
//!
//! - Machine type: `EM_386` (3)
//! - ELF class: `ELFCLASS32`
//! - Data encoding: `ELFDATA2LSB` (little-endian)
//! - ELF flags: `0` (no special flags for standard i386 ELF)
//! - Dynamic linker: `/lib/ld-linux.so.2`
//! - Default base address: `0x08048000` (classic Linux i386 base)
//!
//! # PLT Stub Format
//!
//! i386 PLT stubs use `push`/`jmp` sequences through GOT entries (16 bytes
//! per entry) with 32-bit absolute or GOT-relative addressing (EBX as GOT
//! base in PIC mode).

/// i386 ELF relocation type definitions, application functions, and
/// handler implementing the `ArchRelocationHandler` trait for all
/// `R_386_*` relocation types used by both the assembler (to record
/// relocations during instruction encoding) and the linker (to apply
/// relocations when producing final ELF executables and shared objects).
pub mod relocations;
