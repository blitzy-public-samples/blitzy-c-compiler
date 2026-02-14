//! AArch64 (ARM 64-bit) backend module.
//!
//! This module implements the AArch64 target backend for BCC, providing
//! instruction selection, register allocation support, ABI conformance
//! (AAPCS64), a built-in assembler for A64 instruction encoding, and an
//! integrated ELF linker producing native AArch64 Linux executables and
//! shared objects.
//!
//! # Architecture Overview
//!
//! - 31 general-purpose 64-bit registers (X0–X30), each aliased as 32-bit
//!   (W0–W30)
//! - Stack pointer: SP (not a general-purpose register)
//! - Link register: X30 (LR)
//! - Frame pointer: X29 (FP)
//! - 32 SIMD/floating-point 128-bit registers (V0–V31), aliased as
//!   D0–D31 (64-bit), S0–S31 (32-bit), H0–H31 (16-bit), B0–B31 (8-bit)
//! - Fixed-width 32-bit instruction encoding (A64)
//! - PC-relative addressing via ADRP + ADD/LDR pairs for ±4 GiB reach
//!
//! # Sub-modules
//!
//! - [`linker`]: Built-in AArch64 ELF linker with relocation support

/// AArch64 physical register definitions — X0–X30, W0–W30, SP, XZR/WZR,
/// V0–V31, S0–S31, D0–D31, NZCV condition codes, ABI classification arrays,
/// register name lookup, property queries, and 5-bit A64 encoding helpers.
pub mod registers;

/// Built-in AArch64 assembler producing relocatable object code from
/// `MachineFunction` output without invoking any external tools.
/// Contains the A64 instruction encoder and AArch64-specific ELF
/// relocation type definitions.
pub mod assembler;

/// Built-in AArch64 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code.
pub mod linker;
