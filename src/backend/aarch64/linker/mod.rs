//! Built-in AArch64 (ARM 64-bit) ELF linker module.
//!
//! This module implements the integrated linker for the AArch64 target
//! architecture, producing ET_EXEC static executables and ET_DYN shared
//! objects without invoking any external linker (`ld`, `aarch64-linux-gnu-ld`,
//! `lld`).
//!
//! # Sub-modules
//!
//! - [`relocations`]: AArch64 ELF relocation type definitions and the
//!   [`AArch64RelocationHandler`](relocations::AArch64RelocationHandler)
//!   implementing the [`ArchRelocationHandler`](crate::backend::linker_common::relocation::ArchRelocationHandler)
//!   trait for all `R_AARCH64_*` relocation types with correct A64 instruction
//!   field encoding and page-relative (ADRP) addressing support.
//!
//! # ELF Configuration
//!
//! - Machine type: `EM_AARCH64` (183)
//! - ELF class: `ELFCLASS64`
//! - Data encoding: `ELFDATA2LSB` (little-endian)
//! - ELF flags: `0` (no special flags for standard AArch64 ELF)
//! - Dynamic linker: `/lib/ld-linux-aarch64.so.1`
//!
//! # PLT Stub Format
//!
//! AArch64 PLT stubs use ADRP + LDR + ADD + BR sequences (16 bytes per entry)
//! with IP0 (X16) and IP1 (X17) as scratch registers per AAPCS64.

/// AArch64 ELF relocation type definitions, application functions, and
/// handler implementing the `ArchRelocationHandler` trait for all
/// `R_AARCH64_*` relocation types used by both the assembler (to record
/// relocations during A64 instruction encoding) and the linker (to apply
/// relocations when producing final ELF executables and shared objects).
pub mod relocations;
