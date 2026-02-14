//! Built-in x86-64 assembler module.
//!
//! This module implements the integrated assembler for the x86-64 target
//! architecture, encoding x86-64 machine instructions with ModR/M, SIB,
//! REX/VEX prefixes and recording ELF relocations for unresolved symbols.
//!
//! # Sub-modules
//!
//! - [`encoder`]: Core instruction encoder — translates [`MachineInstr`]s
//!   into binary machine code with proper prefix handling and relocation
//!   record generation.
//! - [`relocations`]: x86-64 ELF relocation type definitions — `R_X86_64_*`
//!   constants, metadata queries (size, PC-relative, GOT/PLT requirements),
//!   and PIC-aware relocation selection helpers.

/// Core x86-64 instruction encoder — translates machine instructions into
/// binary x86-64 machine code with REX, ModR/M, SIB prefix encoding and
/// relocation record generation for unresolved symbolic references.
pub mod encoder;

/// x86-64 ELF relocation type definitions used by both the assembler (to record
/// relocations during instruction encoding) and the linker (to apply relocations
/// when producing final ELF executables and shared objects).
pub mod relocations;

// Re-export the primary entry point so callers can use
// `assembler::encode_function(...)` directly without navigating into
// the encoder submodule.
pub use encoder::{encode_function, AssembledFunction, Relocation};
