//! Built-in RISC-V 64 linker module.
//!
//! This module implements the integrated linker for the RISC-V 64 target
//! architecture, applying `R_RISCV_*` relocations with full linker relaxation
//! support and producing ET_EXEC and ET_DYN ELF binaries.
//!
//! # Sub-modules
//!
//! - [`relocations`]: RISC-V 64 ELF relocation type definitions and the
//!   [`RiscV64RelocationHandler`](relocations::RiscV64RelocationHandler)
//!   implementing the [`ArchRelocationHandler`](crate::backend::linker_common::relocation::ArchRelocationHandler)
//!   trait with linker relaxation support (CALL→JAL, alignment NOP reduction).

/// RISC-V 64 ELF relocation type definitions, application functions, and
/// linker relaxation engine used by both the assembler (to record relocations
/// during instruction encoding) and the linker (to apply relocations when
/// producing final ELF executables and shared objects).
pub mod relocations;
