//! Built-in AArch64 assembler producing relocatable object code without
//! invoking any external tools.
//!
//! This module accepts `MachineFunction` output from the AArch64 instruction
//! selector and produces binary `.text` sections with relocation entries. It
//! operates entirely in-process — no external `as` or `llvm-mc` is ever
//! invoked, enforcing the standalone backend mandate (Section 0.7.7).
//!
//! All A64 instructions are fixed-width 32-bit (4-byte) little-endian words.
//!
//! # Sub-modules
//!
//! - [`encoder`]: A64 instruction encoding — translates `MachineInstr` into
//!   32-bit instruction words with optional relocation annotations.
//! - [`relocations`]: AArch64-specific ELF relocation type definitions used
//!   by both the assembler and the built-in linker.

/// A64 instruction encoder — encodes `MachineInstr` operands into 32-bit
/// instruction words following the A64 encoding specification. Each encoded
/// instruction may carry an optional relocation for unresolved symbol
/// references (ADRP, B/BL, LDR GOT, ADD lo12, etc.).
pub mod encoder;

/// AArch64 ELF relocation type definitions — `AArch64RelocationType` enum
/// covering absolute, PC-relative, page-relative, GOT, TLS, branch, and
/// dynamic relocation types. Provides ELF numeric values, relocation
/// property queries, range validation, and instruction-level application.
pub mod relocations;

// Re-export primary types for convenient access from parent modules.
pub use relocations::AArch64RelocationType;
