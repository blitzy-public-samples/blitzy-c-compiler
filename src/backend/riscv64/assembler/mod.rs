//! Built-in RISC-V 64 assembler module.
//!
//! This module implements the integrated assembler for the RISC-V 64 target
//! architecture, encoding RV64IMAFDC machine instructions in R/I/S/B/U/J
//! formats and recording ELF relocations for unresolved symbols. RISC-V's
//! variable-length encoding (32-bit base + 16-bit compressed RVC extensions)
//! and linker relaxation model require careful relocation handling.
//!
//! # Sub-modules
//!
//! - [`relocations`]: RISC-V 64 ELF relocation type definitions —
//!   `R_RISCV_*` constants, metadata queries (size, PC-relative, GOT/PLT
//!   requirements, relaxation eligibility), hi/lo splitting helpers, and
//!   relocation application functions for all RISC-V instruction formats.

/// RISC-V 64 ELF relocation type definitions used by both the assembler
/// (to record relocations during instruction encoding) and the linker (to
/// apply relocations when producing final ELF executables and shared objects).
/// Includes support for linker relaxation markers (`R_RISCV_RELAX`) and
/// compressed instruction relocations (`R_RISCV_RVC_BRANCH`, `R_RISCV_RVC_JUMP`).
pub mod relocations;

/// RISC-V 64-bit instruction encoder — encodes all RV64IMAFDC instructions
/// in R/I/S/B/U/J/R4 formats plus 16-bit compressed (RVC) formats. Provides
/// the primary encoding entry point [`encoder::RiscV64Encoder::encode_instruction`],
/// immediate validation, register encoding helpers, relocation type inference,
/// and pseudo-instruction expansion (LI, LA, CALL, TAIL, RET, NOP, etc.).
pub mod encoder;
