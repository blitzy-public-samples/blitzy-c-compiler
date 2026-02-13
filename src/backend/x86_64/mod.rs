//! x86-64 backend module.
//!
//! Implements the `ArchCodegen` trait for the x86-64 (AMD64) target architecture,
//! providing instruction selection, register allocation configuration, and
//! ABI-conformant code generation following the System V AMD64 ABI.
//!
//! # Sub-modules
//!
//! - [`assembler`]: Built-in x86-64 assembler — instruction encoding with
//!   ModR/M, SIB, REX/VEX prefixes and ELF relocation recording.
//!
//! # Architecture Characteristics
//!
//! - Variable-length instruction encoding (1–15 bytes)
//! - 16 general-purpose registers (RAX–R15) + 16 SSE registers (XMM0–XMM15)
//! - REX prefix for 64-bit operand size and extended register access
//! - RIP-relative addressing for position-independent code
//! - System V AMD64 calling convention: RDI, RSI, RDX, RCX, R8, R9 for
//!   integer arguments; XMM0–XMM7 for floating-point arguments

/// Built-in x86-64 assembler — instruction encoding, ModR/M, SIB, REX/VEX
/// prefixes, and ELF relocation recording for the x86-64 target.
pub mod assembler;

/// Built-in x86-64 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code and GOTPCRELX
/// relaxation optimization.
pub mod linker;
