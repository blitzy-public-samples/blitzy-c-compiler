//! Built-in x86-64 assembler module.
//!
//! This module implements the integrated assembler for the x86-64 target
//! architecture, encoding x86-64 machine instructions with ModR/M, SIB,
//! REX/VEX prefixes and recording ELF relocations for unresolved symbols.
//!
//! # Sub-modules
//!
//! - [`relocations`]: x86-64 ELF relocation type definitions — `R_X86_64_*`
//!   constants, metadata queries (size, PC-relative, GOT/PLT requirements),
//!   and PIC-aware relocation selection helpers.

/// x86-64 ELF relocation type definitions used by both the assembler (to record
/// relocations during instruction encoding) and the linker (to apply relocations
/// when producing final ELF executables and shared objects).
pub mod relocations;
