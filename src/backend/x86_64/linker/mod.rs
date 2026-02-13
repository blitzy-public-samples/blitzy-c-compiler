//! Built-in x86-64 (AMD64) ELF linker module.
//!
//! This module implements the integrated linker for the x86-64 target
//! architecture, producing ET_EXEC static executables and ET_DYN shared
//! objects without invoking any external linker (`ld`, `gold`, `lld`).
//! It replaces the external `ld` per the standalone backend mandate
//! (Section 0.7.7), using the shared [`linker_common`](crate::backend::linker_common)
//! infrastructure for symbol resolution, section merging, and relocation
//! dispatch.
//!
//! # Sub-modules
//!
//! - [`relocations`]: x86-64 ELF relocation type application and the
//!   [`X86_64RelocationHandler`](relocations::X86_64RelocationHandler)
//!   implementing the [`ArchRelocationHandler`](crate::backend::linker_common::relocation::ArchRelocationHandler)
//!   trait for all `R_X86_64_*` relocation types, including GOT/PLT-relative
//!   relocations (`GOTPCREL`, `GOTPCRELX`, `REX_GOTPCRELX`, `PLT32`) and
//!   truncated-field overflow checking for 32-bit, 16-bit, and 8-bit
//!   relocation fields.
//!
//! # ELF Configuration
//!
//! - Machine type: `EM_X86_64` (62)
//! - ELF class: `ELFCLASS64`
//! - Data encoding: `ELFDATA2LSB` (little-endian)
//! - Default base address for `ET_EXEC`: `0x400000`
//! - Default base address for `ET_DYN`: `0x0` (position-independent)
//! - Dynamic linker: `/lib64/ld-linux-x86-64.so.2`
//!
//! # PLT Stub Format
//!
//! x86-64 PLT stubs use RIP-relative GOT addressing (16 bytes per entry):
//! - PLT\[0\] (resolver): `push *(GOT+8)(%rip); jmp *(GOT+16)(%rip)`
//! - PLT\[N\] (per-function): `jmp *(GOT+N*8)(%rip); push $reloc_index; jmp PLT[0]`
//!
//! # GOTPCRELX Relaxation
//!
//! When a `R_X86_64_GOTPCRELX` or `R_X86_64_REX_GOTPCRELX` relocation
//! references a symbol defined in the same output module, the linker may
//! transform `mov foo@GOTPCREL(%rip), %reg` → `lea foo(%rip), %reg`,
//! eliminating the GOT indirection.

/// x86-64 ELF relocation type application functions and handler implementing
/// the `ArchRelocationHandler` trait for all `R_X86_64_*` relocation types
/// used by the linker to apply relocations when producing final ELF
/// executables and shared objects.
pub mod relocations;
