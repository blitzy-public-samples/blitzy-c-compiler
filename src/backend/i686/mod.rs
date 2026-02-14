//! i686 (32-bit x86) backend module.
//!
//! This module implements the i686 target backend for BCC, providing
//! instruction selection, register allocation support, ABI conformance
//! (cdecl / System V i386 ABI), a built-in assembler for IA-32 instruction
//! encoding, and an integrated ELF linker producing native i686 Linux
//! executables and shared objects.
//!
//! # Architecture Overview
//!
//! - 8 general-purpose 32-bit registers (EAX–EDI)
//! - x87 FPU stack for floating-point operations
//! - Variable-length instruction encoding (1–15 bytes, no REX prefix)
//! - Stack-based parameter passing (cdecl convention)
//! - Classic Linux base address: `0x08048000`
//! - 32-bit addressing only (no RIP-relative; GOT-relative via EBX in PIC)
//!
//! # Sub-modules
//!
//! - [`registers`]: i686 register definitions — 8 GPRs, sub-registers, x87 FPU stack
//! - [`linker`]: Built-in i686 ELF linker with relocation support

/// i686 register definitions — 8 GPRs (EAX–EDI), 16-bit and 8-bit
/// sub-register aliases, x87 FPU stack registers (ST0–ST7), EFLAGS,
/// register classification arrays, and property query functions.
pub mod registers;

/// Built-in i686 ELF linker producing ET_EXEC and ET_DYN binaries
/// with full GOT/PLT relocation support for PIC code.
pub mod linker;
