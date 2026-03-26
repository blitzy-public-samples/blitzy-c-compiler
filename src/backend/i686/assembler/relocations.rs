//! # i686 (i386) ELF Relocation Types
//!
//! Defines all `R_386_*` relocation types from the ELF specification for EM_386.
//! These relocations are used by BCC's built-in i686 assembler to mark references
//! that need resolution by the linker, and by the built-in i686 linker to apply
//! the final address patches in the output ELF binary.
//!
//! ## Key Relocations
//! - `R_386_32` — Absolute 32-bit address (`S + A`)
//! - `R_386_PC32` — PC-relative 32-bit offset (`S + A - P`)
//! - `R_386_GOT32` — GOT entry offset (`G + A`)
//! - `R_386_PLT32` — PLT entry relative address (`L + A - P`)
//! - `R_386_GOTOFF` — GOT-relative offset (`S + A - GOT`)
//! - `R_386_GOTPC` — GOT address, PC-relative (`GOT + A - P`)
//!
//! ## Relocation Application Formulas
//!
//! The following symbols are used in the relocation formulas:
//!
//! | Symbol | Meaning                                                    |
//! |--------|------------------------------------------------------------|
//! | `S`    | Symbol value (resolved address after linking)              |
//! | `A`    | Addend (explicit or implicit from the relocation site)     |
//! | `P`    | Relocation location (address where the patch is applied)   |
//! | `B`    | Base address (shared library load address at runtime)      |
//! | `G`    | GOT entry offset for the symbol from the GOT base         |
//! | `GOT`  | Address of the Global Offset Table                        |
//! | `L`    | PLT entry address for the symbol                          |
//!
//! ### Core Relocation Formulas
//!
//! | Type            | Formula              | Description                        |
//! |-----------------|----------------------|------------------------------------|
//! | `R_386_32`      | `word32 = S + A`     | Absolute 32-bit                    |
//! | `R_386_PC32`    | `word32 = S + A - P` | PC-relative 32-bit                 |
//! | `R_386_GOT32`   | `word32 = G + A`     | GOT entry offset                   |
//! | `R_386_PLT32`   | `word32 = L + A - P` | PLT entry, PC-relative             |
//! | `R_386_COPY`    | *(runtime copy)*     | Dynamic linker copies data         |
//! | `R_386_GLOB_DAT`| `word32 = S`         | Set GOT entry to symbol address    |
//! | `R_386_JMP_SLOT`| `word32 = S`         | Set PLT GOT slot to symbol address |
//! | `R_386_RELATIVE`| `word32 = B + A`     | Base-relative (load-time adjust)   |
//! | `R_386_GOTOFF`  | `word32 = S + A - GOT` | Symbol offset from GOT base      |
//! | `R_386_GOTPC`   | `word32 = GOT + A - P` | GOT address, PC-relative         |
//!
//! ## i386 REL vs RELA
//!
//! The i386 ABI uses `Elf32_Rel` (implicit addend read from the relocation site),
//! not `Elf32_Rela`. The addend `A` is extracted from the existing content at the
//! relocation offset before patching.
//!
//! ## Reference
//!
//! * System V ABI — Intel386 Architecture Processor Supplement (SCO, 1997)
//! * ELF Handling for Thread-Local Storage (Ulrich Drepper, 2013)
//! * GNU Binutils `elf/i386.h` relocation type definitions

use crate::backend::traits::RelocationTypeInfo;

// ===========================================================================
// Core Relocation Type Constants (ELF EM_386)
// ===========================================================================

/// No relocation. Placeholder entry with no effect.
pub const R_386_NONE: u32 = 0;

/// Absolute 32-bit relocation: `S + A`
///
/// Direct 32-bit address of the symbol plus addend.
/// Used for absolute address references in data sections and non-PIC code.
pub const R_386_32: u32 = 1;

/// PC-relative 32-bit relocation: `S + A - P`
///
/// Symbol address plus addend minus the relocation location.
/// Used for direct function calls (`CALL rel32`) and PC-relative branches.
pub const R_386_PC32: u32 = 2;

/// GOT entry 32-bit offset: `G + A`
///
/// Offset into the Global Offset Table for this symbol.
/// Used in PIC code to access global data via the GOT.
pub const R_386_GOT32: u32 = 3;

/// PLT entry 32-bit offset: `L + A - P`
///
/// Address of the PLT entry for this symbol, relative to relocation location.
/// Used for function calls in PIC code that go through the PLT.
pub const R_386_PLT32: u32 = 4;

/// Copy relocation.
///
/// Used by the dynamic linker to copy data from shared libraries into the
/// executable's BSS segment. Created by the link-editor for dynamic
/// executables to preserve a read-only text segment.
pub const R_386_COPY: u32 = 5;

/// Global data relocation: `S`
///
/// Set GOT entry to the symbol's absolute address.
/// Used by the dynamic linker to fill in GOT entries at load time.
pub const R_386_GLOB_DAT: u32 = 6;

/// PLT jump slot relocation: `S`
///
/// Set PLT GOT entry to the symbol's absolute address.
/// Used for lazy binding in the dynamic linker; the PLT stub initially
/// jumps to the resolver, which patches the GOT slot on first call.
pub const R_386_JMP_SLOT: u32 = 7;

/// Relative relocation: `B + A`
///
/// Base address plus addend. Used in shared libraries and PIE executables
/// for position-independent data references within the same object.
pub const R_386_RELATIVE: u32 = 8;

/// GOT-relative offset: `S + A - GOT`
///
/// Symbol address relative to the start of the GOT.
/// Used in PIC code to compute offsets from the GOT base held in EBX.
pub const R_386_GOTOFF: u32 = 9;

/// GOT PC-relative: `GOT + A - P`
///
/// GOT address relative to the relocation point.
/// Used with `__x86.get_pc_thunk.*` to compute the GOT pointer in PIC code.
pub const R_386_GOTPC: u32 = 10;

// ===========================================================================
// TLS (Thread-Local Storage) Relocation Types
// ===========================================================================
//
// These relocations implement the four TLS access models defined in the
// "ELF Handling for Thread-Local Storage" specification:
//   - General Dynamic (GD): for dynamically-loaded TLS in shared libraries
//   - Local Dynamic (LDM/LD): optimized for multiple TLS vars in one module
//   - Initial Exec (IE): for TLS in the initial executable or preloaded libs
//   - Local Exec (LE): for TLS known to be in the executable itself
//
// The GD and LDM sequences use multi-instruction patterns (push/call/pop)
// that the linker can relax to IE or LE when linking statically.

/// TLS Initial-Exec: Address of GOT entry for static TLS block offset.
///
/// Loads the TP-relative offset of the TLS variable from the GOT.
/// Only valid when the module is part of the initial TLS image.
pub const R_386_TLS_IE: u32 = 15;

/// TLS Initial-Exec (GOT-indirect): GOT entry for negated static TLS offset.
///
/// References a GOT slot containing the negated offset from the thread pointer.
pub const R_386_TLS_GOTIE: u32 = 16;

/// TLS Local-Exec offset: `S - TLS_BASE`
///
/// Direct offset relative to the thread pointer for TLS variables known
/// to be in the executable. The simplest and most efficient TLS model.
pub const R_386_TLS_LE: u32 = 17;

/// TLS General Dynamic: Direct 32-bit for GD model.
///
/// References a `tls_index` structure in the GOT for `__tls_get_addr`.
pub const R_386_TLS_GD: u32 = 18;

/// TLS Local Dynamic: Direct 32-bit for LDM model.
///
/// Similar to GD but for module-local TLS variables; the dtv offset
/// is computed once and reused for multiple TLS accesses in the same module.
pub const R_386_TLS_LDM: u32 = 19;

/// TLS GD 32-bit: Tag for `leal x@tlsgd(,%ebx,1), %eax` in GD TLS code.
pub const R_386_TLS_GD_32: u32 = 24;

/// TLS GD push: Tag for `pushl %eax` in GD TLS code sequence.
pub const R_386_TLS_GD_PUSH: u32 = 25;

/// TLS GD call: Relocation for `call ___tls_get_addr@PLT` in GD TLS code.
pub const R_386_TLS_GD_CALL: u32 = 26;

/// TLS GD pop: Tag for `popl %ebx` in GD TLS code sequence.
pub const R_386_TLS_GD_POP: u32 = 27;

/// TLS LDM 32-bit: Tag for `leal x@tlsldm(%ebx), %eax` in LDM TLS code.
pub const R_386_TLS_LDM_32: u32 = 28;

/// TLS LDM push: Tag for `pushl %eax` in LDM TLS code sequence.
pub const R_386_TLS_LDM_PUSH: u32 = 29;

/// TLS LDM call: Relocation for `call ___tls_get_addr@PLT` in LDM TLS code.
pub const R_386_TLS_LDM_CALL: u32 = 30;

/// TLS LDM pop: Tag for `popl %ebx` in LDM TLS code sequence.
pub const R_386_TLS_LDM_POP: u32 = 31;

/// TLS Local Dynamic Offset: `S + A - TLS_BLOCK`
///
/// Offset relative to the TLS block base for a local dynamic variable.
/// Used after the dtv base has been obtained via `__tls_get_addr`.
pub const R_386_TLS_LDO_32: u32 = 32;

/// TLS Initial-Exec 32-bit: GOT entry for negated static TLS block offset.
///
/// References a GOT entry containing the negated TP-relative offset.
/// Used in the IE→LE relaxation pathway.
pub const R_386_TLS_IE_32: u32 = 33;

/// TLS Local-Exec negated offset: `TLS_BASE - S`
///
/// Negated offset relative to the thread pointer. Some TLS implementations
/// use negated offsets for historical/ABI compatibility reasons.
pub const R_386_TLS_LE_32: u32 = 34;

/// TLS descriptor: GOT offset for TLS descriptor.
///
/// References a TLSDESC structure in the GOT, used by the TLSDESC ABI
/// for more efficient TLS access on modern glibc.
pub const R_386_TLS_GOTDESC: u32 = 39;

/// TLS descriptor call: Marker for the call through a TLS descriptor.
///
/// Annotates the `call *x@tlscall(%eax)` instruction in the TLSDESC sequence.
pub const R_386_TLS_DESC_CALL: u32 = 40;

/// TLS descriptor entry: Full TLS descriptor containing pointer and argument.
///
/// A two-word GOT entry: function pointer + argument, filled by the dynamic
/// linker for TLSDESC resolution.
pub const R_386_TLS_DESC: u32 = 41;

// ===========================================================================
// Legacy and GNU Extension Relocation Types
// ===========================================================================

/// 16-bit absolute relocation (legacy).
///
/// Used in 16-bit code segments or mixed 16/32-bit environments.
/// Rarely encountered in modern Linux binaries.
pub const R_386_16: u32 = 20;

/// 16-bit PC-relative relocation (legacy).
///
/// PC-relative variant of `R_386_16` for 16-bit code.
pub const R_386_PC16: u32 = 21;

/// 8-bit absolute relocation (legacy).
///
/// Used for byte-sized relocations in special contexts.
pub const R_386_8: u32 = 22;

/// 8-bit PC-relative relocation (legacy).
///
/// PC-relative variant for byte-sized offsets.
pub const R_386_PC8: u32 = 23;

/// GNU IFUNC (indirect function) relocation.
///
/// Similar to `R_386_RELATIVE` but the resolved address is a function
/// pointer that the dynamic linker calls to determine the final address.
/// Used for hardware capability-based function dispatch (e.g., memcpy
/// selecting SSE vs. non-SSE implementation at load time).
pub const R_386_IRELATIVE: u32 = 42;

/// Relaxable GOT load: `G + A`
///
/// Like `R_386_GOT32` but marks the instruction as eligible for linker
/// relaxation. When the symbol is non-preemptible, the linker can convert
/// a GOT-indirect load (`mov foo@GOT(%ebx), %reg`) into a direct LEA
/// (`lea foo@GOTOFF(%ebx), %reg`), eliminating the GOT indirection.
pub const R_386_GOT32X: u32 = 43;

// ===========================================================================
// Relocation Type Info Array — for ArchCodegen::relocation_types()
// ===========================================================================

/// Complete table of i686 relocation types with metadata.
///
/// Used by [`ArchCodegen::relocation_types()`](crate::backend::traits::ArchCodegen::relocation_types)
/// and the linker for relocation dispatch. Each entry describes:
/// - `name`: Human-readable identifier matching the ELF specification
/// - `type_id`: Numeric ELF relocation type value (`Elf32_Rel.r_info` low byte)
/// - `size`: Width of the relocation field in bytes (0, 1, 2, or 4)
/// - `is_pc_relative`: Whether the formula subtracts the relocation address (`P`)
///
/// The table is ordered by type_id for efficient lookup. Core relocations
/// (0–10) are listed first, followed by TLS, legacy, and GNU extensions.
pub static I686_RELOCATION_TYPES: &[RelocationTypeInfo] = &[
    // Core relocations (0–10)
    RelocationTypeInfo {
        name: "R_386_NONE",
        type_id: R_386_NONE,
        size: 0,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_32",
        type_id: R_386_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_PC32",
        type_id: R_386_PC32,
        size: 4,
        is_pc_relative: true,
    },
    RelocationTypeInfo {
        name: "R_386_GOT32",
        type_id: R_386_GOT32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_PLT32",
        type_id: R_386_PLT32,
        size: 4,
        is_pc_relative: true,
    },
    RelocationTypeInfo {
        name: "R_386_COPY",
        type_id: R_386_COPY,
        size: 0,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_GLOB_DAT",
        type_id: R_386_GLOB_DAT,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_JMP_SLOT",
        type_id: R_386_JMP_SLOT,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_RELATIVE",
        type_id: R_386_RELATIVE,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_GOTOFF",
        type_id: R_386_GOTOFF,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_GOTPC",
        type_id: R_386_GOTPC,
        size: 4,
        is_pc_relative: true,
    },
    // TLS relocations (15–19, 24–34, 39–41)
    RelocationTypeInfo {
        name: "R_386_TLS_IE",
        type_id: R_386_TLS_IE,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GOTIE",
        type_id: R_386_TLS_GOTIE,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LE",
        type_id: R_386_TLS_LE,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GD",
        type_id: R_386_TLS_GD,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDM",
        type_id: R_386_TLS_LDM,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GD_32",
        type_id: R_386_TLS_GD_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GD_PUSH",
        type_id: R_386_TLS_GD_PUSH,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GD_CALL",
        type_id: R_386_TLS_GD_CALL,
        size: 4,
        is_pc_relative: true,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GD_POP",
        type_id: R_386_TLS_GD_POP,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDM_32",
        type_id: R_386_TLS_LDM_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDM_PUSH",
        type_id: R_386_TLS_LDM_PUSH,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDM_CALL",
        type_id: R_386_TLS_LDM_CALL,
        size: 4,
        is_pc_relative: true,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDM_POP",
        type_id: R_386_TLS_LDM_POP,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LDO_32",
        type_id: R_386_TLS_LDO_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_IE_32",
        type_id: R_386_TLS_IE_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_LE_32",
        type_id: R_386_TLS_LE_32,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_GOTDESC",
        type_id: R_386_TLS_GOTDESC,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_DESC_CALL",
        type_id: R_386_TLS_DESC_CALL,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_TLS_DESC",
        type_id: R_386_TLS_DESC,
        size: 4,
        is_pc_relative: false,
    },
    // Legacy 16-bit and 8-bit relocations (20–23)
    RelocationTypeInfo {
        name: "R_386_16",
        type_id: R_386_16,
        size: 2,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_PC16",
        type_id: R_386_PC16,
        size: 2,
        is_pc_relative: true,
    },
    RelocationTypeInfo {
        name: "R_386_8",
        type_id: R_386_8,
        size: 1,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_PC8",
        type_id: R_386_PC8,
        size: 1,
        is_pc_relative: true,
    },
    // GNU extensions (42–43)
    RelocationTypeInfo {
        name: "R_386_IRELATIVE",
        type_id: R_386_IRELATIVE,
        size: 4,
        is_pc_relative: false,
    },
    RelocationTypeInfo {
        name: "R_386_GOT32X",
        type_id: R_386_GOT32X,
        size: 4,
        is_pc_relative: false,
    },
];

// ===========================================================================
// Helper Functions — Relocation Classification
// ===========================================================================

/// Check if a relocation type computes a PC-relative offset.
///
/// PC-relative relocations subtract the relocation site address (`P`) from
/// the target address. They are essential for position-independent code
/// and for `CALL`/`JMP` instructions that use relative offsets.
///
/// # Examples
///
/// ```ignore
/// assert!(is_pc_relative(R_386_PC32));
/// assert!(is_pc_relative(R_386_PLT32));
/// assert!(!is_pc_relative(R_386_32));
/// ```
pub fn is_pc_relative(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_PC32 | R_386_PLT32 | R_386_GOTPC | R_386_PC16 | R_386_PC8
    )
}

/// Check if a relocation type references the Global Offset Table (GOT).
///
/// GOT-referencing relocations are used in position-independent code to
/// access global data through the GOT indirection. The linker must create
/// GOT entries for symbols referenced by these relocations.
///
/// # Includes
///
/// - `R_386_GOT32` — offset into GOT for the symbol
/// - `R_386_GLOB_DAT` — dynamic linker fills GOT entry
/// - `R_386_GOTOFF` — offset from GOT base to symbol
/// - `R_386_GOTPC` — address of GOT, PC-relative
/// - `R_386_GOT32X` — relaxable GOT load
pub fn is_got_relocation(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_GOT32 | R_386_GLOB_DAT | R_386_GOTOFF | R_386_GOTPC | R_386_GOT32X
    )
}

/// Check if a relocation type is for dynamic linking (PLT/GOT slots).
///
/// Dynamic relocations are processed by the runtime dynamic linker (`ld.so`)
/// rather than the static linker. They appear in `.rel.dyn` and `.rel.plt`
/// sections of shared libraries and dynamically-linked executables.
///
/// # Includes
///
/// - `R_386_GLOB_DAT` — fill GOT entry at load time
/// - `R_386_JMP_SLOT` — lazy-bind PLT entry
/// - `R_386_RELATIVE` — base-relative adjustment
/// - `R_386_COPY` — copy from shared library
/// - `R_386_IRELATIVE` — IFUNC resolver dispatch
pub fn is_dynamic_relocation(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_GLOB_DAT | R_386_JMP_SLOT | R_386_RELATIVE | R_386_COPY | R_386_IRELATIVE
    )
}

/// Check if a relocation is TLS (Thread-Local Storage) related.
///
/// TLS relocations implement the four TLS access models (GD, LD, IE, LE)
/// and the TLSDESC extension. They are needed for `_Thread_local` variable
/// support and appear in both static and dynamic linking contexts.
///
/// The linker may relax GD/LD relocations to IE or LE when the TLS variable
/// is known to be in the executable or initial TLS image.
pub fn is_tls_relocation(rel_type: u32) -> bool {
    matches!(
        rel_type,
        R_386_TLS_IE
            | R_386_TLS_GOTIE
            | R_386_TLS_LE
            | R_386_TLS_GD
            | R_386_TLS_LDM
            | R_386_TLS_GD_32
            | R_386_TLS_GD_PUSH
            | R_386_TLS_GD_CALL
            | R_386_TLS_GD_POP
            | R_386_TLS_LDM_32
            | R_386_TLS_LDM_PUSH
            | R_386_TLS_LDM_CALL
            | R_386_TLS_LDM_POP
            | R_386_TLS_LDO_32
            | R_386_TLS_IE_32
            | R_386_TLS_LE_32
            | R_386_TLS_GOTDESC
            | R_386_TLS_DESC_CALL
            | R_386_TLS_DESC
    )
}

/// Get the size of a relocation patch in bytes.
///
/// Returns the number of bytes that the linker writes at the relocation site:
/// - `0` for `R_386_NONE` and `R_386_COPY` (no patch / runtime-only)
/// - `1` for 8-bit relocations (`R_386_8`, `R_386_PC8`)
/// - `2` for 16-bit relocations (`R_386_16`, `R_386_PC16`)
/// - `4` for all standard 32-bit relocations (the vast majority)
///
/// # Examples
///
/// ```ignore
/// assert_eq!(relocation_size(R_386_NONE), 0);
/// assert_eq!(relocation_size(R_386_32), 4);
/// assert_eq!(relocation_size(R_386_8), 1);
/// assert_eq!(relocation_size(R_386_16), 2);
/// ```
pub fn relocation_size(rel_type: u32) -> u8 {
    match rel_type {
        R_386_NONE | R_386_COPY => 0,
        R_386_8 | R_386_PC8 => 1,
        R_386_16 | R_386_PC16 => 2,
        _ => 4, // All other i386 relocations are 32-bit
    }
}

/// Get the human-readable name of a relocation type.
///
/// Returns the canonical ELF ABI supplement name for the relocation type.
/// Unknown relocation type values return `"R_386_UNKNOWN"`.
///
/// Useful for diagnostic messages, debug output, and linker error reporting.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(relocation_name(R_386_32), "R_386_32");
/// assert_eq!(relocation_name(R_386_PC32), "R_386_PC32");
/// assert_eq!(relocation_name(999), "R_386_UNKNOWN");
/// ```
pub fn relocation_name(rel_type: u32) -> &'static str {
    match rel_type {
        R_386_NONE => "R_386_NONE",
        R_386_32 => "R_386_32",
        R_386_PC32 => "R_386_PC32",
        R_386_GOT32 => "R_386_GOT32",
        R_386_PLT32 => "R_386_PLT32",
        R_386_COPY => "R_386_COPY",
        R_386_GLOB_DAT => "R_386_GLOB_DAT",
        R_386_JMP_SLOT => "R_386_JMP_SLOT",
        R_386_RELATIVE => "R_386_RELATIVE",
        R_386_GOTOFF => "R_386_GOTOFF",
        R_386_GOTPC => "R_386_GOTPC",
        R_386_TLS_IE => "R_386_TLS_IE",
        R_386_TLS_GOTIE => "R_386_TLS_GOTIE",
        R_386_TLS_LE => "R_386_TLS_LE",
        R_386_TLS_GD => "R_386_TLS_GD",
        R_386_TLS_LDM => "R_386_TLS_LDM",
        R_386_16 => "R_386_16",
        R_386_PC16 => "R_386_PC16",
        R_386_8 => "R_386_8",
        R_386_PC8 => "R_386_PC8",
        R_386_TLS_GD_32 => "R_386_TLS_GD_32",
        R_386_TLS_GD_PUSH => "R_386_TLS_GD_PUSH",
        R_386_TLS_GD_CALL => "R_386_TLS_GD_CALL",
        R_386_TLS_GD_POP => "R_386_TLS_GD_POP",
        R_386_TLS_LDM_32 => "R_386_TLS_LDM_32",
        R_386_TLS_LDM_PUSH => "R_386_TLS_LDM_PUSH",
        R_386_TLS_LDM_CALL => "R_386_TLS_LDM_CALL",
        R_386_TLS_LDM_POP => "R_386_TLS_LDM_POP",
        R_386_TLS_LDO_32 => "R_386_TLS_LDO_32",
        R_386_TLS_IE_32 => "R_386_TLS_IE_32",
        R_386_TLS_LE_32 => "R_386_TLS_LE_32",
        R_386_TLS_GOTDESC => "R_386_TLS_GOTDESC",
        R_386_TLS_DESC_CALL => "R_386_TLS_DESC_CALL",
        R_386_TLS_DESC => "R_386_TLS_DESC",
        R_386_IRELATIVE => "R_386_IRELATIVE",
        R_386_GOT32X => "R_386_GOT32X",
        _ => "R_386_UNKNOWN",
    }
}

// ===========================================================================
// Unit-Test Support (compile-time assertions)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_relocation_values_match_elf_spec() {
        // Verify against the canonical ELF i386 ABI supplement values
        assert_eq!(R_386_NONE, 0);
        assert_eq!(R_386_32, 1);
        assert_eq!(R_386_PC32, 2);
        assert_eq!(R_386_GOT32, 3);
        assert_eq!(R_386_PLT32, 4);
        assert_eq!(R_386_COPY, 5);
        assert_eq!(R_386_GLOB_DAT, 6);
        assert_eq!(R_386_JMP_SLOT, 7);
        assert_eq!(R_386_RELATIVE, 8);
        assert_eq!(R_386_GOTOFF, 9);
        assert_eq!(R_386_GOTPC, 10);
    }

    #[test]
    fn tls_relocation_values_match_elf_spec() {
        assert_eq!(R_386_TLS_IE, 15);
        assert_eq!(R_386_TLS_GOTIE, 16);
        assert_eq!(R_386_TLS_LE, 17);
        assert_eq!(R_386_TLS_GD, 18);
        assert_eq!(R_386_TLS_LDM, 19);
        assert_eq!(R_386_TLS_GD_32, 24);
        assert_eq!(R_386_TLS_GD_PUSH, 25);
        assert_eq!(R_386_TLS_GD_CALL, 26);
        assert_eq!(R_386_TLS_GD_POP, 27);
        assert_eq!(R_386_TLS_LDM_32, 28);
        assert_eq!(R_386_TLS_LDM_PUSH, 29);
        assert_eq!(R_386_TLS_LDM_CALL, 30);
        assert_eq!(R_386_TLS_LDM_POP, 31);
        assert_eq!(R_386_TLS_LDO_32, 32);
        assert_eq!(R_386_TLS_IE_32, 33);
        assert_eq!(R_386_TLS_LE_32, 34);
        assert_eq!(R_386_TLS_GOTDESC, 39);
        assert_eq!(R_386_TLS_DESC_CALL, 40);
        assert_eq!(R_386_TLS_DESC, 41);
    }

    #[test]
    fn legacy_and_gnu_relocation_values() {
        assert_eq!(R_386_16, 20);
        assert_eq!(R_386_PC16, 21);
        assert_eq!(R_386_8, 22);
        assert_eq!(R_386_PC8, 23);
        assert_eq!(R_386_IRELATIVE, 42);
        assert_eq!(R_386_GOT32X, 43);
    }

    #[test]
    fn pc_relative_classification() {
        assert!(is_pc_relative(R_386_PC32));
        assert!(is_pc_relative(R_386_PLT32));
        assert!(is_pc_relative(R_386_GOTPC));
        assert!(is_pc_relative(R_386_PC16));
        assert!(is_pc_relative(R_386_PC8));
        // Non-PC-relative types:
        assert!(!is_pc_relative(R_386_NONE));
        assert!(!is_pc_relative(R_386_32));
        assert!(!is_pc_relative(R_386_GOT32));
        assert!(!is_pc_relative(R_386_GOTOFF));
        assert!(!is_pc_relative(R_386_RELATIVE));
    }

    #[test]
    fn got_relocation_classification() {
        assert!(is_got_relocation(R_386_GOT32));
        assert!(is_got_relocation(R_386_GLOB_DAT));
        assert!(is_got_relocation(R_386_GOTOFF));
        assert!(is_got_relocation(R_386_GOTPC));
        assert!(is_got_relocation(R_386_GOT32X));
        assert!(!is_got_relocation(R_386_32));
        assert!(!is_got_relocation(R_386_PC32));
    }

    #[test]
    fn dynamic_relocation_classification() {
        assert!(is_dynamic_relocation(R_386_GLOB_DAT));
        assert!(is_dynamic_relocation(R_386_JMP_SLOT));
        assert!(is_dynamic_relocation(R_386_RELATIVE));
        assert!(is_dynamic_relocation(R_386_COPY));
        assert!(is_dynamic_relocation(R_386_IRELATIVE));
        assert!(!is_dynamic_relocation(R_386_32));
        assert!(!is_dynamic_relocation(R_386_PC32));
    }

    #[test]
    fn tls_relocation_classification() {
        assert!(is_tls_relocation(R_386_TLS_GD));
        assert!(is_tls_relocation(R_386_TLS_LDM));
        assert!(is_tls_relocation(R_386_TLS_IE));
        assert!(is_tls_relocation(R_386_TLS_LE));
        assert!(is_tls_relocation(R_386_TLS_GOTIE));
        assert!(is_tls_relocation(R_386_TLS_GD_32));
        assert!(is_tls_relocation(R_386_TLS_GD_POP));
        assert!(is_tls_relocation(R_386_TLS_LDM_POP));
        assert!(is_tls_relocation(R_386_TLS_GOTDESC));
        assert!(is_tls_relocation(R_386_TLS_DESC));
        assert!(!is_tls_relocation(R_386_32));
        assert!(!is_tls_relocation(R_386_PC32));
    }

    #[test]
    fn relocation_size_values() {
        assert_eq!(relocation_size(R_386_NONE), 0);
        assert_eq!(relocation_size(R_386_COPY), 0);
        assert_eq!(relocation_size(R_386_32), 4);
        assert_eq!(relocation_size(R_386_PC32), 4);
        assert_eq!(relocation_size(R_386_GOT32), 4);
        assert_eq!(relocation_size(R_386_PLT32), 4);
        assert_eq!(relocation_size(R_386_RELATIVE), 4);
        assert_eq!(relocation_size(R_386_16), 2);
        assert_eq!(relocation_size(R_386_PC16), 2);
        assert_eq!(relocation_size(R_386_8), 1);
        assert_eq!(relocation_size(R_386_PC8), 1);
        assert_eq!(relocation_size(R_386_TLS_GD), 4);
        assert_eq!(relocation_size(R_386_IRELATIVE), 4);
        assert_eq!(relocation_size(R_386_GOT32X), 4);
    }

    #[test]
    fn relocation_name_lookup() {
        assert_eq!(relocation_name(R_386_NONE), "R_386_NONE");
        assert_eq!(relocation_name(R_386_32), "R_386_32");
        assert_eq!(relocation_name(R_386_PC32), "R_386_PC32");
        assert_eq!(relocation_name(R_386_GOT32), "R_386_GOT32");
        assert_eq!(relocation_name(R_386_PLT32), "R_386_PLT32");
        assert_eq!(relocation_name(R_386_IRELATIVE), "R_386_IRELATIVE");
        assert_eq!(relocation_name(R_386_GOT32X), "R_386_GOT32X");
        assert_eq!(relocation_name(R_386_TLS_GD), "R_386_TLS_GD");
        assert_eq!(relocation_name(R_386_TLS_DESC), "R_386_TLS_DESC");
        assert_eq!(relocation_name(999), "R_386_UNKNOWN");
    }

    #[test]
    fn relocation_type_info_array_completeness() {
        // Verify the array contains all core relocations
        let find_by_id =
            |id: u32| -> bool { I686_RELOCATION_TYPES.iter().any(|r| r.type_id == id) };
        assert!(find_by_id(R_386_NONE));
        assert!(find_by_id(R_386_32));
        assert!(find_by_id(R_386_PC32));
        assert!(find_by_id(R_386_GOT32));
        assert!(find_by_id(R_386_PLT32));
        assert!(find_by_id(R_386_COPY));
        assert!(find_by_id(R_386_GLOB_DAT));
        assert!(find_by_id(R_386_JMP_SLOT));
        assert!(find_by_id(R_386_RELATIVE));
        assert!(find_by_id(R_386_GOTOFF));
        assert!(find_by_id(R_386_GOTPC));
        assert!(find_by_id(R_386_TLS_GD));
        assert!(find_by_id(R_386_TLS_LDM));
        assert!(find_by_id(R_386_IRELATIVE));
        assert!(find_by_id(R_386_GOT32X));
    }

    #[test]
    fn relocation_type_info_metadata_correct() {
        // Verify metadata for a few key entries
        let pc32 = I686_RELOCATION_TYPES
            .iter()
            .find(|r| r.type_id == R_386_PC32)
            .unwrap();
        assert_eq!(pc32.name, "R_386_PC32");
        assert_eq!(pc32.size, 4);
        assert!(pc32.is_pc_relative);

        let abs32 = I686_RELOCATION_TYPES
            .iter()
            .find(|r| r.type_id == R_386_32)
            .unwrap();
        assert_eq!(abs32.name, "R_386_32");
        assert_eq!(abs32.size, 4);
        assert!(!abs32.is_pc_relative);

        let none = I686_RELOCATION_TYPES
            .iter()
            .find(|r| r.type_id == R_386_NONE)
            .unwrap();
        assert_eq!(none.size, 0);
        assert!(!none.is_pc_relative);
    }

    #[test]
    fn no_duplicate_type_ids() {
        // Verify no two entries have the same type_id
        for (i, a) in I686_RELOCATION_TYPES.iter().enumerate() {
            for (j, b) in I686_RELOCATION_TYPES.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a.type_id, b.type_id,
                        "Duplicate type_id {} between {} and {}",
                        a.type_id, a.name, b.name
                    );
                }
            }
        }
    }
}
