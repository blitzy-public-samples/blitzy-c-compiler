// SPDX-License-Identifier: MIT
//
// src/backend/x86_64/assembler/relocations.rs
//
// x86-64 ELF relocation type definitions for the BCC built-in assembler and linker.
//
// This module defines every ELF relocation type needed for position-independent and
// position-dependent code emission on the x86-64 architecture. It is consumed by both
// the assembler (to record relocations during instruction encoding) and the linker
// (to apply relocations when producing final ELF executables and shared objects).
//
// The relocation types, ELF constant values, and computation formulas are derived from
// the System V Application Binary Interface — AMD64 Architecture Processor Supplement
// (https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf) and the ELF
// specification for x86-64.
//
// ## Relocation Computation Legend
//
// The following symbols are used in relocation formulas throughout this module:
//
// - **S** — Symbol value (the resolved virtual address of the target symbol)
// - **A** — Addend (a signed constant stored in the relocation entry)
// - **P** — Place (the address of the storage unit being relocated, i.e., the
//   offset of the relocation site within the output section)
// - **B** — Base address (the base virtual address at which the shared object is
//   loaded; used for `R_X86_64_RELATIVE` in position-independent executables)
// - **G** — GOT entry offset (the offset within the GOT of the entry for the
//   referenced symbol)
// - **GOT** — Global Offset Table address (the virtual address of the `.got` section)
// - **L** — PLT entry address (the virtual address of the PLT stub for the symbol)
//
// ## Zero-Dependency Mandate
//
// This module uses only standard Rust types. No external crates are imported.
// All relocation handling is implemented internally per the BCC standalone backend
// architecture (Section 0.7.7 of the Agent Action Plan).

use std::fmt;

/// Enumerates all x86-64 ELF relocation types supported by the BCC assembler and linker.
///
/// Each variant corresponds to a specific relocation computation as defined by the
/// x86-64 ABI supplement. The variants carry no data; metadata is accessed via the
/// associated methods (`elf_value`, `size`, `is_pc_relative`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86_64RelocationType {
    /// ELF value 0 — No relocation action required. Used as a placeholder or padding
    /// in relocation tables.
    R_X86_64_NONE,

    /// ELF value 1 — Direct 64-bit absolute address.
    /// Formula: S + A → write 64-bit value.
    /// Used for absolute data references (e.g., function pointers in `.data`).
    R_X86_64_64,

    /// ELF value 2 — PC-relative 32-bit signed offset.
    /// Formula: S + A - P → write 32-bit signed value.
    /// Used for direct function calls (`CALL rel32`) and non-PIC data references.
    R_X86_64_PC32,

    /// ELF value 3 — 32-bit GOT entry offset.
    /// Formula: G + A → write 32-bit value.
    /// Used for GOT-indirect data access in position-independent code.
    R_X86_64_GOT32,

    /// ELF value 4 — 32-bit PLT-relative offset.
    /// Formula: L + A - P → write 32-bit signed value.
    /// Used for function calls through the Procedure Linkage Table in PIC code.
    R_X86_64_PLT32,

    /// ELF value 5 — Copy relocation (dynamic linker).
    /// The dynamic linker copies the symbol's data into the executable's BSS segment.
    /// No computation by the static linker; the runtime linker handles it.
    R_X86_64_COPY,

    /// ELF value 6 — Set GOT entry to the symbol's absolute address.
    /// Formula: S → write 64-bit absolute address into the GOT slot.
    /// Used by the dynamic linker to populate GOT entries at load time.
    R_X86_64_GLOB_DAT,

    /// ELF value 7 — Set GOT entry for PLT lazy binding.
    /// Formula: S → write 64-bit address into the GOT slot for PLT resolution.
    /// Used for lazy symbol resolution via the PLT/GOT mechanism.
    R_X86_64_JUMP_SLOT,

    /// ELF value 8 — Base-relative relocation for shared objects.
    /// Formula: B + A → write 64-bit value.
    /// Used in `-shared` output for relocations relative to the load base address.
    R_X86_64_RELATIVE,

    /// ELF value 9 — PC-relative GOT entry offset.
    /// Formula: G + GOT + A - P → write 32-bit signed value.
    /// Used for GOT-indirect addressing in position-independent code.
    R_X86_64_GOTPCREL,

    /// ELF value 10 — Direct 32-bit zero-extended absolute address.
    /// Formula: S + A → write 32-bit value (must fit in u32).
    /// Used for 32-bit absolute addresses in the lower 4 GiB address space.
    R_X86_64_32,

    /// ELF value 11 — Direct 32-bit sign-extended absolute address.
    /// Formula: S + A → write 32-bit value (must fit in i32).
    /// Used when the address is known to reside in the signed 32-bit range.
    R_X86_64_32S,

    /// ELF value 12 — Direct 16-bit zero-extended absolute address.
    /// Formula: S + A → write 16-bit value.
    R_X86_64_16,

    /// ELF value 13 — PC-relative 16-bit signed offset.
    /// Formula: S + A - P → write 16-bit signed value.
    R_X86_64_PC16,

    /// ELF value 14 — Direct 8-bit absolute value.
    /// Formula: S + A → write 8-bit value.
    R_X86_64_8,

    /// ELF value 15 — PC-relative 8-bit signed offset.
    /// Formula: S + A - P → write 8-bit signed value.
    R_X86_64_PC8,

    /// ELF value 24 — PC-relative 64-bit signed offset.
    /// Formula: S + A - P → write 64-bit value.
    /// Rare; used for 64-bit PC-relative references.
    R_X86_64_PC64,

    /// ELF value 41 — PC-relative GOT entry with linker relaxation hint.
    /// Formula: G + GOT + A - P → write 32-bit signed value.
    /// The linker may optimize (relax) this to a direct LEA if the symbol is locally
    /// defined, eliminating the GOT indirection. Instruction does NOT have a REX prefix.
    R_X86_64_GOTPCRELX,

    /// ELF value 42 — Same as `GOTPCRELX` but the instruction carries a REX prefix.
    /// Formula: G + GOT + A - P → write 32-bit signed value.
    /// Indicates a 64-bit GOT access with REX.W=1; the linker may relax similarly.
    R_X86_64_REX_GOTPCRELX,
}

impl X86_64RelocationType {
    /// Returns the numeric ELF relocation constant (`R_X86_64_*` value) for this type.
    ///
    /// These values are defined by the x86-64 ABI supplement and are written into the
    /// `r_info` field of `Elf64_Rela` relocation entries in the ELF output.
    pub fn elf_value(&self) -> u32 {
        match self {
            Self::R_X86_64_NONE => 0,
            Self::R_X86_64_64 => 1,
            Self::R_X86_64_PC32 => 2,
            Self::R_X86_64_GOT32 => 3,
            Self::R_X86_64_PLT32 => 4,
            Self::R_X86_64_COPY => 5,
            Self::R_X86_64_GLOB_DAT => 6,
            Self::R_X86_64_JUMP_SLOT => 7,
            Self::R_X86_64_RELATIVE => 8,
            Self::R_X86_64_GOTPCREL => 9,
            Self::R_X86_64_32 => 10,
            Self::R_X86_64_32S => 11,
            Self::R_X86_64_16 => 12,
            Self::R_X86_64_PC16 => 13,
            Self::R_X86_64_8 => 14,
            Self::R_X86_64_PC8 => 15,
            Self::R_X86_64_PC64 => 24,
            Self::R_X86_64_GOTPCRELX => 41,
            Self::R_X86_64_REX_GOTPCRELX => 42,
        }
    }

    /// Parses a numeric ELF relocation type value into the corresponding enum variant.
    ///
    /// Returns `None` if the value does not correspond to a relocation type supported
    /// by this assembler/linker. This is used when reading relocatable object files
    /// during the linking phase.
    pub fn from_elf_value(value: u32) -> Option<X86_64RelocationType> {
        match value {
            0 => Some(Self::R_X86_64_NONE),
            1 => Some(Self::R_X86_64_64),
            2 => Some(Self::R_X86_64_PC32),
            3 => Some(Self::R_X86_64_GOT32),
            4 => Some(Self::R_X86_64_PLT32),
            5 => Some(Self::R_X86_64_COPY),
            6 => Some(Self::R_X86_64_GLOB_DAT),
            7 => Some(Self::R_X86_64_JUMP_SLOT),
            8 => Some(Self::R_X86_64_RELATIVE),
            9 => Some(Self::R_X86_64_GOTPCREL),
            10 => Some(Self::R_X86_64_32),
            11 => Some(Self::R_X86_64_32S),
            12 => Some(Self::R_X86_64_16),
            13 => Some(Self::R_X86_64_PC16),
            14 => Some(Self::R_X86_64_8),
            15 => Some(Self::R_X86_64_PC8),
            24 => Some(Self::R_X86_64_PC64),
            41 => Some(Self::R_X86_64_GOTPCRELX),
            42 => Some(Self::R_X86_64_REX_GOTPCRELX),
            _ => None,
        }
    }

    /// Returns `true` if this relocation type computes a PC-relative offset.
    ///
    /// PC-relative relocations subtract the relocation site address (`P`) from the
    /// target address, producing a displacement suitable for RIP-relative addressing
    /// and branch instructions. The linker must account for the instruction length
    /// when applying these relocations (the addend typically includes -4 for the
    /// 4-byte relocation field width).
    pub fn is_pc_relative(&self) -> bool {
        matches!(
            self,
            Self::R_X86_64_PC32
                | Self::R_X86_64_PLT32
                | Self::R_X86_64_GOTPCREL
                | Self::R_X86_64_GOTPCRELX
                | Self::R_X86_64_REX_GOTPCRELX
                | Self::R_X86_64_PC64
                | Self::R_X86_64_PC16
                | Self::R_X86_64_PC8
        )
    }

    /// Returns the byte size of the relocation field written at the relocation site.
    ///
    /// - 1 byte: `R_X86_64_8`, `R_X86_64_PC8`
    /// - 2 bytes: `R_X86_64_16`, `R_X86_64_PC16`
    /// - 4 bytes: `R_X86_64_PC32`, `R_X86_64_PLT32`, `R_X86_64_GOT32`, `R_X86_64_32`,
    ///   `R_X86_64_32S`, `R_X86_64_GOTPCREL`, `R_X86_64_GOTPCRELX`, `R_X86_64_REX_GOTPCRELX`
    /// - 8 bytes: `R_X86_64_64`, `R_X86_64_PC64`, `R_X86_64_GLOB_DAT`, `R_X86_64_JUMP_SLOT`,
    ///   `R_X86_64_RELATIVE`
    /// - 0 bytes: `R_X86_64_NONE`, `R_X86_64_COPY` (no field is patched)
    pub fn size(&self) -> u8 {
        match self {
            Self::R_X86_64_NONE | Self::R_X86_64_COPY => 0,
            Self::R_X86_64_8 | Self::R_X86_64_PC8 => 1,
            Self::R_X86_64_16 | Self::R_X86_64_PC16 => 2,
            Self::R_X86_64_PC32
            | Self::R_X86_64_PLT32
            | Self::R_X86_64_GOT32
            | Self::R_X86_64_32
            | Self::R_X86_64_32S
            | Self::R_X86_64_GOTPCREL
            | Self::R_X86_64_GOTPCRELX
            | Self::R_X86_64_REX_GOTPCRELX => 4,
            Self::R_X86_64_64
            | Self::R_X86_64_PC64
            | Self::R_X86_64_GLOB_DAT
            | Self::R_X86_64_JUMP_SLOT
            | Self::R_X86_64_RELATIVE => 8,
        }
    }

    /// Returns `true` if this relocation type produces a sign-extended 32-bit value.
    ///
    /// `R_X86_64_32S` writes a 32-bit value that is sign-extended to 64 bits by the
    /// processor. The linker must verify that the computed value fits in the range
    /// `[-2^31, 2^31 - 1]` (i.e., `i32`). All PC-relative 32-bit relocations are
    /// also inherently signed, but this method specifically identifies the
    /// `R_X86_64_32S` type versus `R_X86_64_32` (zero-extended).
    ///
    /// For `R_X86_64_32`, the value must fit in `[0, 2^32 - 1]` (i.e., `u32`).
    pub fn is_signed(&self) -> bool {
        matches!(self, Self::R_X86_64_32S)
    }

    /// Returns `true` if this relocation type requires a Global Offset Table (GOT) entry.
    ///
    /// When the linker encounters one of these relocation types, it must allocate a GOT
    /// slot for the referenced symbol (if one does not already exist) and populate it
    /// with the symbol's absolute address at load time (or via dynamic relocation).
    pub fn requires_got(&self) -> bool {
        matches!(
            self,
            Self::R_X86_64_GOT32
                | Self::R_X86_64_GOTPCREL
                | Self::R_X86_64_GOTPCRELX
                | Self::R_X86_64_REX_GOTPCRELX
                | Self::R_X86_64_GLOB_DAT
        )
    }

    /// Returns `true` if this relocation type requires a Procedure Linkage Table (PLT)
    /// entry for the referenced symbol.
    ///
    /// PLT entries provide lazy binding stubs for function calls in shared libraries.
    /// The linker creates a PLT stub and a corresponding GOT slot with an initial value
    /// pointing back to the PLT resolver when it encounters these relocations.
    pub fn requires_plt(&self) -> bool {
        matches!(self, Self::R_X86_64_PLT32 | Self::R_X86_64_JUMP_SLOT)
    }

    /// Returns `true` if this relocation is processed by the dynamic linker at load time
    /// rather than fully resolved by the static linker.
    ///
    /// Dynamic relocations appear in `.rela.dyn` and `.rela.plt` sections and are
    /// emitted when producing shared objects (`-shared`) or position-independent
    /// executables. The static linker records them in the output ELF for the runtime
    /// loader (`ld-linux-x86-64.so.2`) to process.
    pub fn is_dynamic(&self) -> bool {
        matches!(
            self,
            Self::R_X86_64_COPY
                | Self::R_X86_64_GLOB_DAT
                | Self::R_X86_64_JUMP_SLOT
                | Self::R_X86_64_RELATIVE
        )
    }

    /// Returns the canonical name of this relocation type as it appears in ELF
    /// documentation and `readelf` output.
    ///
    /// This is the same string produced by the `Display` implementation, provided
    /// here as a `&'static str` for zero-allocation use in performance-sensitive paths.
    pub fn name(&self) -> &'static str {
        match self {
            Self::R_X86_64_NONE => "R_X86_64_NONE",
            Self::R_X86_64_64 => "R_X86_64_64",
            Self::R_X86_64_PC32 => "R_X86_64_PC32",
            Self::R_X86_64_GOT32 => "R_X86_64_GOT32",
            Self::R_X86_64_PLT32 => "R_X86_64_PLT32",
            Self::R_X86_64_COPY => "R_X86_64_COPY",
            Self::R_X86_64_GLOB_DAT => "R_X86_64_GLOB_DAT",
            Self::R_X86_64_JUMP_SLOT => "R_X86_64_JUMP_SLOT",
            Self::R_X86_64_RELATIVE => "R_X86_64_RELATIVE",
            Self::R_X86_64_GOTPCREL => "R_X86_64_GOTPCREL",
            Self::R_X86_64_32 => "R_X86_64_32",
            Self::R_X86_64_32S => "R_X86_64_32S",
            Self::R_X86_64_16 => "R_X86_64_16",
            Self::R_X86_64_PC16 => "R_X86_64_PC16",
            Self::R_X86_64_8 => "R_X86_64_8",
            Self::R_X86_64_PC8 => "R_X86_64_PC8",
            Self::R_X86_64_PC64 => "R_X86_64_PC64",
            Self::R_X86_64_GOTPCRELX => "R_X86_64_GOTPCRELX",
            Self::R_X86_64_REX_GOTPCRELX => "R_X86_64_REX_GOTPCRELX",
        }
    }
}

/// Human-readable display of relocation type names for diagnostic output.
///
/// Produces the canonical ELF relocation name string (e.g., `"R_X86_64_PC32"`,
/// `"R_X86_64_PLT32"`), matching the format used by GNU `readelf -r` and
/// standard ELF tooling.
impl fmt::Display for X86_64RelocationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// PIC-specific relocation selection helpers
// ---------------------------------------------------------------------------
// These free functions encapsulate the policy for choosing the appropriate
// relocation type based on PIC mode and instruction context. They are used
// by the x86-64 code generator and assembler to emit correct relocations
// without duplicating selection logic across multiple call sites.

/// Selects the appropriate relocation type for a function call instruction.
///
/// - In PIC mode (`is_pic == true`), function calls use `R_X86_64_PLT32` so that
///   the call goes through the PLT stub, enabling lazy binding and interposition
///   in shared libraries.
/// - In non-PIC mode (`is_pic == false`), function calls use `R_X86_64_PC32` for
///   a direct PC-relative branch to the target symbol.
///
/// Both relocation types produce a 32-bit signed PC-relative value and are applied
/// to `CALL rel32` and `JMP rel32` instructions.
pub fn select_call_relocation(is_pic: bool) -> X86_64RelocationType {
    if is_pic {
        X86_64RelocationType::R_X86_64_PLT32
    } else {
        X86_64RelocationType::R_X86_64_PC32
    }
}

/// Selects the appropriate relocation type for a data/global variable reference.
///
/// - In PIC mode (`is_pic == true`), global data accesses are routed through the
///   GOT using `R_X86_64_GOTPCRELX`. The linker may relax this to a direct LEA
///   if the symbol is locally defined (same DSO), eliminating the GOT indirection.
/// - In non-PIC mode, the relocation depends on the operand `size`:
///   - 8 bytes → `R_X86_64_64` (absolute 64-bit address)
///   - 4 bytes → `R_X86_64_32` (absolute 32-bit zero-extended address)
///   - Any other size defaults to `R_X86_64_32` for data references.
///
/// This function covers the common case; callers requiring `R_X86_64_32S`
/// (sign-extended 32-bit) should handle that explicitly.
pub fn select_data_relocation(is_pic: bool, size: u8) -> X86_64RelocationType {
    if is_pic {
        X86_64RelocationType::R_X86_64_GOTPCRELX
    } else {
        match size {
            8 => X86_64RelocationType::R_X86_64_64,
            _ => X86_64RelocationType::R_X86_64_32,
        }
    }
}

/// Selects the correct GOT-relative relocation type based on the presence of a
/// REX prefix in the accessing instruction.
///
/// - If the instruction has a REX prefix (`has_rex == true`), use
///   `R_X86_64_REX_GOTPCRELX`. This indicates a 64-bit GOT access with REX.W=1
///   (e.g., `MOV RAX, [RIP + symbol@GOTPCREL]`).
/// - Otherwise, use `R_X86_64_GOTPCRELX` for a non-REX GOT access
///   (e.g., `MOV EAX, [RIP + symbol@GOTPCREL]`).
///
/// Both variants carry a linker relaxation hint: the linker may optimize the
/// GOT-indirect access to a direct `LEA` when the referenced symbol is defined
/// within the same shared object.
pub fn select_got_relocation(has_rex: bool) -> X86_64RelocationType {
    if has_rex {
        X86_64RelocationType::R_X86_64_REX_GOTPCRELX
    } else {
        X86_64RelocationType::R_X86_64_GOTPCRELX
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that every variant round-trips through elf_value → from_elf_value.
    #[test]
    fn test_elf_value_round_trip() {
        let all_variants = [
            X86_64RelocationType::R_X86_64_NONE,
            X86_64RelocationType::R_X86_64_64,
            X86_64RelocationType::R_X86_64_PC32,
            X86_64RelocationType::R_X86_64_GOT32,
            X86_64RelocationType::R_X86_64_PLT32,
            X86_64RelocationType::R_X86_64_COPY,
            X86_64RelocationType::R_X86_64_GLOB_DAT,
            X86_64RelocationType::R_X86_64_JUMP_SLOT,
            X86_64RelocationType::R_X86_64_RELATIVE,
            X86_64RelocationType::R_X86_64_GOTPCREL,
            X86_64RelocationType::R_X86_64_32,
            X86_64RelocationType::R_X86_64_32S,
            X86_64RelocationType::R_X86_64_16,
            X86_64RelocationType::R_X86_64_PC16,
            X86_64RelocationType::R_X86_64_8,
            X86_64RelocationType::R_X86_64_PC8,
            X86_64RelocationType::R_X86_64_PC64,
            X86_64RelocationType::R_X86_64_GOTPCRELX,
            X86_64RelocationType::R_X86_64_REX_GOTPCRELX,
        ];

        for variant in &all_variants {
            let elf_val = variant.elf_value();
            let parsed = X86_64RelocationType::from_elf_value(elf_val);
            assert_eq!(
                parsed,
                Some(*variant),
                "Round-trip failed for {:?} (elf_value={})",
                variant,
                elf_val
            );
        }
    }

    /// Verify that unrecognized ELF values return None.
    #[test]
    fn test_from_elf_value_unknown() {
        assert_eq!(X86_64RelocationType::from_elf_value(99), None);
        assert_eq!(X86_64RelocationType::from_elf_value(16), None);
        assert_eq!(X86_64RelocationType::from_elf_value(23), None);
        assert_eq!(X86_64RelocationType::from_elf_value(40), None);
        assert_eq!(X86_64RelocationType::from_elf_value(43), None);
        assert_eq!(X86_64RelocationType::from_elf_value(u32::MAX), None);
    }

    /// Verify specific ELF constant values match the x86-64 ABI specification.
    #[test]
    fn test_elf_values_are_correct() {
        assert_eq!(X86_64RelocationType::R_X86_64_NONE.elf_value(), 0);
        assert_eq!(X86_64RelocationType::R_X86_64_64.elf_value(), 1);
        assert_eq!(X86_64RelocationType::R_X86_64_PC32.elf_value(), 2);
        assert_eq!(X86_64RelocationType::R_X86_64_GOT32.elf_value(), 3);
        assert_eq!(X86_64RelocationType::R_X86_64_PLT32.elf_value(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_COPY.elf_value(), 5);
        assert_eq!(X86_64RelocationType::R_X86_64_GLOB_DAT.elf_value(), 6);
        assert_eq!(X86_64RelocationType::R_X86_64_JUMP_SLOT.elf_value(), 7);
        assert_eq!(X86_64RelocationType::R_X86_64_RELATIVE.elf_value(), 8);
        assert_eq!(X86_64RelocationType::R_X86_64_GOTPCREL.elf_value(), 9);
        assert_eq!(X86_64RelocationType::R_X86_64_32.elf_value(), 10);
        assert_eq!(X86_64RelocationType::R_X86_64_32S.elf_value(), 11);
        assert_eq!(X86_64RelocationType::R_X86_64_16.elf_value(), 12);
        assert_eq!(X86_64RelocationType::R_X86_64_PC16.elf_value(), 13);
        assert_eq!(X86_64RelocationType::R_X86_64_8.elf_value(), 14);
        assert_eq!(X86_64RelocationType::R_X86_64_PC8.elf_value(), 15);
        assert_eq!(X86_64RelocationType::R_X86_64_PC64.elf_value(), 24);
        assert_eq!(X86_64RelocationType::R_X86_64_GOTPCRELX.elf_value(), 41);
        assert_eq!(X86_64RelocationType::R_X86_64_REX_GOTPCRELX.elf_value(), 42);
    }

    /// Verify PC-relative classification.
    #[test]
    fn test_is_pc_relative() {
        // PC-relative types
        assert!(X86_64RelocationType::R_X86_64_PC32.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_PLT32.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_GOTPCREL.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_GOTPCRELX.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_REX_GOTPCRELX.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_PC64.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_PC16.is_pc_relative());
        assert!(X86_64RelocationType::R_X86_64_PC8.is_pc_relative());

        // Non-PC-relative types
        assert!(!X86_64RelocationType::R_X86_64_NONE.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_64.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_GOT32.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_32.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_32S.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_COPY.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_GLOB_DAT.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_JUMP_SLOT.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_RELATIVE.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_16.is_pc_relative());
        assert!(!X86_64RelocationType::R_X86_64_8.is_pc_relative());
    }

    /// Verify relocation field sizes.
    #[test]
    fn test_size() {
        assert_eq!(X86_64RelocationType::R_X86_64_NONE.size(), 0);
        assert_eq!(X86_64RelocationType::R_X86_64_COPY.size(), 0);
        assert_eq!(X86_64RelocationType::R_X86_64_8.size(), 1);
        assert_eq!(X86_64RelocationType::R_X86_64_PC8.size(), 1);
        assert_eq!(X86_64RelocationType::R_X86_64_16.size(), 2);
        assert_eq!(X86_64RelocationType::R_X86_64_PC16.size(), 2);
        assert_eq!(X86_64RelocationType::R_X86_64_PC32.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_PLT32.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_GOT32.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_32.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_32S.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_GOTPCREL.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_GOTPCRELX.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_REX_GOTPCRELX.size(), 4);
        assert_eq!(X86_64RelocationType::R_X86_64_64.size(), 8);
        assert_eq!(X86_64RelocationType::R_X86_64_PC64.size(), 8);
        assert_eq!(X86_64RelocationType::R_X86_64_GLOB_DAT.size(), 8);
        assert_eq!(X86_64RelocationType::R_X86_64_JUMP_SLOT.size(), 8);
        assert_eq!(X86_64RelocationType::R_X86_64_RELATIVE.size(), 8);
    }

    /// Verify signed classification.
    #[test]
    fn test_is_signed() {
        assert!(X86_64RelocationType::R_X86_64_32S.is_signed());
        assert!(!X86_64RelocationType::R_X86_64_32.is_signed());
        assert!(!X86_64RelocationType::R_X86_64_64.is_signed());
        assert!(!X86_64RelocationType::R_X86_64_PC32.is_signed());
        assert!(!X86_64RelocationType::R_X86_64_NONE.is_signed());
    }

    /// Verify GOT requirement classification.
    #[test]
    fn test_requires_got() {
        assert!(X86_64RelocationType::R_X86_64_GOT32.requires_got());
        assert!(X86_64RelocationType::R_X86_64_GOTPCREL.requires_got());
        assert!(X86_64RelocationType::R_X86_64_GOTPCRELX.requires_got());
        assert!(X86_64RelocationType::R_X86_64_REX_GOTPCRELX.requires_got());
        assert!(X86_64RelocationType::R_X86_64_GLOB_DAT.requires_got());

        assert!(!X86_64RelocationType::R_X86_64_PC32.requires_got());
        assert!(!X86_64RelocationType::R_X86_64_PLT32.requires_got());
        assert!(!X86_64RelocationType::R_X86_64_64.requires_got());
        assert!(!X86_64RelocationType::R_X86_64_JUMP_SLOT.requires_got());
    }

    /// Verify PLT requirement classification.
    #[test]
    fn test_requires_plt() {
        assert!(X86_64RelocationType::R_X86_64_PLT32.requires_plt());
        assert!(X86_64RelocationType::R_X86_64_JUMP_SLOT.requires_plt());

        assert!(!X86_64RelocationType::R_X86_64_PC32.requires_plt());
        assert!(!X86_64RelocationType::R_X86_64_GOTPCRELX.requires_plt());
        assert!(!X86_64RelocationType::R_X86_64_64.requires_plt());
        assert!(!X86_64RelocationType::R_X86_64_GLOB_DAT.requires_plt());
    }

    /// Verify dynamic relocation classification.
    #[test]
    fn test_is_dynamic() {
        assert!(X86_64RelocationType::R_X86_64_COPY.is_dynamic());
        assert!(X86_64RelocationType::R_X86_64_GLOB_DAT.is_dynamic());
        assert!(X86_64RelocationType::R_X86_64_JUMP_SLOT.is_dynamic());
        assert!(X86_64RelocationType::R_X86_64_RELATIVE.is_dynamic());

        assert!(!X86_64RelocationType::R_X86_64_PC32.is_dynamic());
        assert!(!X86_64RelocationType::R_X86_64_PLT32.is_dynamic());
        assert!(!X86_64RelocationType::R_X86_64_64.is_dynamic());
        assert!(!X86_64RelocationType::R_X86_64_32.is_dynamic());
        assert!(!X86_64RelocationType::R_X86_64_GOTPCRELX.is_dynamic());
    }

    /// Verify Display formatting matches canonical ELF names.
    #[test]
    fn test_display() {
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_NONE),
            "R_X86_64_NONE"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_PC32),
            "R_X86_64_PC32"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_PLT32),
            "R_X86_64_PLT32"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_GOTPCRELX),
            "R_X86_64_GOTPCRELX"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_REX_GOTPCRELX),
            "R_X86_64_REX_GOTPCRELX"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_64),
            "R_X86_64_64"
        );
        assert_eq!(
            format!("{}", X86_64RelocationType::R_X86_64_RELATIVE),
            "R_X86_64_RELATIVE"
        );
    }

    /// Verify PIC call relocation selection.
    #[test]
    fn test_select_call_relocation() {
        assert_eq!(
            select_call_relocation(true),
            X86_64RelocationType::R_X86_64_PLT32
        );
        assert_eq!(
            select_call_relocation(false),
            X86_64RelocationType::R_X86_64_PC32
        );
    }

    /// Verify PIC data relocation selection.
    #[test]
    fn test_select_data_relocation() {
        // PIC mode always returns GOTPCRELX regardless of size
        assert_eq!(
            select_data_relocation(true, 4),
            X86_64RelocationType::R_X86_64_GOTPCRELX
        );
        assert_eq!(
            select_data_relocation(true, 8),
            X86_64RelocationType::R_X86_64_GOTPCRELX
        );

        // Non-PIC mode: 8-byte → R_X86_64_64, otherwise → R_X86_64_32
        assert_eq!(
            select_data_relocation(false, 8),
            X86_64RelocationType::R_X86_64_64
        );
        assert_eq!(
            select_data_relocation(false, 4),
            X86_64RelocationType::R_X86_64_32
        );
        assert_eq!(
            select_data_relocation(false, 2),
            X86_64RelocationType::R_X86_64_32
        );
        assert_eq!(
            select_data_relocation(false, 1),
            X86_64RelocationType::R_X86_64_32
        );
    }

    /// Verify GOT relocation selection based on REX prefix presence.
    #[test]
    fn test_select_got_relocation() {
        assert_eq!(
            select_got_relocation(true),
            X86_64RelocationType::R_X86_64_REX_GOTPCRELX
        );
        assert_eq!(
            select_got_relocation(false),
            X86_64RelocationType::R_X86_64_GOTPCRELX
        );
    }

    /// Verify Clone and Copy derive traits work correctly.
    #[test]
    fn test_clone_copy() {
        let original = X86_64RelocationType::R_X86_64_PLT32;
        let cloned = original.clone();
        let copied = original;
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    /// Verify Hash derive works for use in HashMap/HashSet.
    #[test]
    fn test_hash_in_collection() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(X86_64RelocationType::R_X86_64_PC32);
        set.insert(X86_64RelocationType::R_X86_64_PLT32);
        set.insert(X86_64RelocationType::R_X86_64_PC32); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&X86_64RelocationType::R_X86_64_PC32));
        assert!(set.contains(&X86_64RelocationType::R_X86_64_PLT32));
        assert!(!set.contains(&X86_64RelocationType::R_X86_64_64));
    }

    /// Verify Debug formatting produces the expected variant names.
    #[test]
    fn test_debug_format() {
        let debug_str = format!("{:?}", X86_64RelocationType::R_X86_64_GOTPCRELX);
        assert_eq!(debug_str, "R_X86_64_GOTPCRELX");
    }

    /// Verify that name() and Display produce identical output.
    #[test]
    fn test_name_matches_display() {
        let all_variants = [
            X86_64RelocationType::R_X86_64_NONE,
            X86_64RelocationType::R_X86_64_64,
            X86_64RelocationType::R_X86_64_PC32,
            X86_64RelocationType::R_X86_64_GOT32,
            X86_64RelocationType::R_X86_64_PLT32,
            X86_64RelocationType::R_X86_64_COPY,
            X86_64RelocationType::R_X86_64_GLOB_DAT,
            X86_64RelocationType::R_X86_64_JUMP_SLOT,
            X86_64RelocationType::R_X86_64_RELATIVE,
            X86_64RelocationType::R_X86_64_GOTPCREL,
            X86_64RelocationType::R_X86_64_32,
            X86_64RelocationType::R_X86_64_32S,
            X86_64RelocationType::R_X86_64_16,
            X86_64RelocationType::R_X86_64_PC16,
            X86_64RelocationType::R_X86_64_8,
            X86_64RelocationType::R_X86_64_PC8,
            X86_64RelocationType::R_X86_64_PC64,
            X86_64RelocationType::R_X86_64_GOTPCRELX,
            X86_64RelocationType::R_X86_64_REX_GOTPCRELX,
        ];

        for variant in &all_variants {
            assert_eq!(
                variant.name(),
                format!("{}", variant),
                "name() and Display mismatch for {:?}",
                variant
            );
        }
    }
}
