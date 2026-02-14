//! Built-in i686 assembler module.
//!
//! This module implements the integrated assembler for the i686 (IA-32) target
//! architecture, encoding 32-bit x86 machine instructions without REX prefixes
//! and recording ELF relocations for unresolved symbols. Unlike x86-64, i686
//! uses the classic ModR/M and SIB encoding without REX byte extensions, and
//! relocations use the REL format (inline addends) rather than RELA.
//!
//! The assembler operates entirely in-process — no external `as` or `llvm-mc`
//! is ever invoked, enforcing the standalone backend mandate (Section 0.7.7).
//!
//! # Sub-modules
//!
//! - [`relocations`]: i686-specific ELF relocation type definitions —
//!   `R_386_*` constants, metadata queries (size, PC-relative, GOT/PLT
//!   requirements), REL format helpers, PIC-aware relocation selection,
//!   and relocation application functions for all i686 relocation types.
//!
//! # Key Differences from x86-64
//!
//! - No REX prefix — only 8 GPRs (EAX–EDI) directly encodable
//! - REL relocations (8-byte entries: offset + info) instead of RELA
//!   (24-byte entries: offset + info + addend) — addend is inline in the
//!   instruction stream
//! - PIC code uses `__i686.get_pc_thunk.bx` + `R_386_GOTPC` to establish
//!   the GOT base in EBX, rather than RIP-relative addressing
//! - All addresses and relocations are 32-bit (no 64-bit relocations)

/// i686 ELF relocation type definitions used by both the assembler (to record
/// relocations during instruction encoding) and the linker (to apply relocations
/// when producing final ELF executables and shared objects).
///
/// Implements the `I686RelocType` enum covering all required `R_386_*` relocation
/// types, property queries, relocation application formulas, overflow detection,
/// and REL-format encoding helpers.
pub mod relocations;
