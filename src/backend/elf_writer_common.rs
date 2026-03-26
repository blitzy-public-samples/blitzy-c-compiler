//! Common ELF binary format writing infrastructure for BCC.
//!
//! This module hand-implements all ELF binary format construction, replacing
//! external crates such as `object` or `elf` in compliance with BCC's
//! zero-dependency mandate. It provides:
//!
//! - **ELF constants** — Magic bytes, class, data encoding, section types,
//!   section flags, program header types, symbol bindings/types/visibility.
//! - **Low-level ELF structures** — `Elf64Header`, `Elf32Header`,
//!   `Elf64SectionHeader`, `Elf32SectionHeader`, `Elf64ProgramHeader`,
//!   `Elf32ProgramHeader`, `Elf64Symbol`, `Elf32Symbol`, `Elf64Rela`,
//!   `Elf32Rel`, all with `to_bytes()` little-endian serialization.
//! - **String table builder** ([`StringTable`]) — Constructs `.strtab` and
//!   `.shstrtab` with FxHashMap-backed deduplication.
//! - **Symbol table builder** ([`SymbolTable`]) — Manages symbols with the
//!   ELF-mandated local-before-global ordering.
//! - **Section management** ([`Section`]) — Generic section container for
//!   `.text`, `.data`, `.rodata`, `.bss`, etc.
//! - **ELF writer** ([`ElfWriter`]) — Top-level writer that assembles
//!   a complete ELF binary (ET_REL, ET_EXEC, or ET_DYN) from sections,
//!   symbols, and program headers for all four target architectures.
//!
//! # Supported Formats
//!
//! - **ELF32** for i686 (ILP32)
//! - **ELF64** for x86-64, AArch64, RISC-V 64 (LP64)
//!
//! # Architecture Support
//!
//! Architecture-specific ELF header values (`e_machine`, `EI_CLASS`,
//! `EI_DATA`, `e_flags`) are obtained from [`Target`] methods, ensuring
//! correct output for all four backends.

use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// ELF Magic
// ===========================================================================

/// ELF magic number: `\x7fELF`.
pub const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

// ===========================================================================
// EI_CLASS — ELF Class (32-bit vs 64-bit)
// ===========================================================================

/// 32-bit ELF objects (i686).
pub const ELFCLASS32: u8 = 1;
/// 64-bit ELF objects (x86-64, AArch64, RISC-V 64).
pub const ELFCLASS64: u8 = 2;

// ===========================================================================
// EI_DATA — Data Encoding (Endianness)
// ===========================================================================

/// Little-endian data encoding. All four BCC targets are little-endian.
pub const ELFDATA2LSB: u8 = 1;

// ===========================================================================
// EI_VERSION
// ===========================================================================

/// Current ELF version.
const EV_CURRENT: u8 = 1;

// ===========================================================================
// e_type — Object File Type
// ===========================================================================

/// Relocatable object file (`.o`).
pub const ET_REL: u16 = 1;
/// Executable file.
pub const ET_EXEC: u16 = 2;
/// Shared object file (`.so`).
pub const ET_DYN: u16 = 3;

// ===========================================================================
// e_machine — Architecture
// ===========================================================================

/// Intel 80386 (i686).
pub const EM_386: u16 = 3;
/// AMD x86-64 (x86-64).
pub const EM_X86_64: u16 = 62;
/// ARM AARCH64 (AArch64).
pub const EM_AARCH64: u16 = 183;
/// RISC-V.
pub const EM_RISCV: u16 = 243;

// ===========================================================================
// sh_type — Section Types
// ===========================================================================

/// Inactive section header table entry.
pub const SHT_NULL: u32 = 0;
/// Program-defined data (`.text`, `.rodata`, `.data`, etc.).
pub const SHT_PROGBITS: u32 = 1;
/// Symbol table.
pub const SHT_SYMTAB: u32 = 2;
/// String table.
pub const SHT_STRTAB: u32 = 3;
/// Relocation entries with explicit addends (`.rela.*`).
pub const SHT_RELA: u32 = 4;
/// Symbol hash table.
pub const SHT_HASH: u32 = 5;
/// Dynamic linking information (`.dynamic`).
pub const SHT_DYNAMIC: u32 = 6;
/// Information for the object file (`.note.*`).
pub const SHT_NOTE: u32 = 7;
/// Section occupies no file space (`.bss`).
pub const SHT_NOBITS: u32 = 8;
/// Relocation entries without explicit addends (`.rel.*`).
pub const SHT_REL: u32 = 9;
/// Dynamic linker symbol table (`.dynsym`).
pub const SHT_DYNSYM: u32 = 11;
/// GNU hash table (`.gnu.hash`).
pub const SHT_GNU_HASH: u32 = 0x6fff_fff6;

// ===========================================================================
// sh_flags — Section Attribute Flags
// ===========================================================================

/// Section data is writable during execution.
pub const SHF_WRITE: u64 = 0x1;
/// Section occupies memory during process execution.
pub const SHF_ALLOC: u64 = 0x2;
/// Section contains executable machine instructions.
pub const SHF_EXECINSTR: u64 = 0x4;
/// Section data may be merged to eliminate duplication.
pub const SHF_MERGE: u64 = 0x10;
/// Section data consists of null-terminated character strings.
pub const SHF_STRINGS: u64 = 0x20;
/// Section header `sh_info` field holds a section header table index.
pub const SHF_INFO_LINK: u64 = 0x40;
/// Section is a member of a section group.
pub const SHF_GROUP: u64 = 0x200;

// ===========================================================================
// p_type — Program Header Types
// ===========================================================================

/// Unused program header table entry.
pub const PT_NULL: u32 = 0;
/// Loadable segment.
pub const PT_LOAD: u32 = 1;
/// Dynamic linking information.
pub const PT_DYNAMIC: u32 = 2;
/// Pathname of the interpreter.
pub const PT_INTERP: u32 = 3;
/// Auxiliary information.
pub const PT_NOTE: u32 = 4;
/// Program header table.
pub const PT_PHDR: u32 = 6;
/// GNU extension: stack executability control.
pub const PT_GNU_STACK: u32 = 0x6474_e551;
/// GNU extension: read-only after relocation.
pub const PT_GNU_RELRO: u32 = 0x6474_e552;

// ===========================================================================
// Symbol Binding (STB_*)
// ===========================================================================

/// Local symbol — not visible outside the object file.
pub const STB_LOCAL: u8 = 0;
/// Global symbol — visible to all object files.
pub const STB_GLOBAL: u8 = 1;
/// Weak symbol — like global but with lower precedence.
pub const STB_WEAK: u8 = 2;

// ===========================================================================
// Symbol Type (STT_*)
// ===========================================================================

/// Symbol type is not specified.
pub const STT_NOTYPE: u8 = 0;
/// Symbol is a data object (variable, array, etc.).
pub const STT_OBJECT: u8 = 1;
/// Symbol is a function entry point.
pub const STT_FUNC: u8 = 2;
/// Symbol is associated with a section.
pub const STT_SECTION: u8 = 3;
/// Symbol gives the name of the source file.
pub const STT_FILE: u8 = 4;

// ===========================================================================
// Symbol Visibility (STV_*)
// ===========================================================================

/// Default visibility — determined by binding type.
pub const STV_DEFAULT: u8 = 0;
/// Hidden visibility — not visible from other components.
pub const STV_HIDDEN: u8 = 2;
/// Protected visibility — visible but not preemptable.
pub const STV_PROTECTED: u8 = 3;

// ===========================================================================
// Special Section Indices
// ===========================================================================

/// Undefined, missing, irrelevant, or meaningless section reference.
const SHN_UNDEF: u16 = 0;

// ===========================================================================
// ELF64 Header (64 bytes)
// ===========================================================================

/// 64-bit ELF file header.
///
/// Serializes to exactly 64 bytes in little-endian format.
/// The `e_ident` array is constructed automatically by [`to_bytes`] using
/// ELFCLASS64 and ELFDATA2LSB; the caller need only set the variable fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Elf64Header {
    /// Object file type (ET_REL, ET_EXEC, ET_DYN).
    pub e_type: u16,
    /// Architecture (EM_X86_64, EM_AARCH64, EM_RISCV).
    pub e_machine: u16,
    /// Entry point virtual address.
    pub e_entry: u64,
    /// Program header table file offset.
    pub e_phoff: u64,
    /// Section header table file offset.
    pub e_shoff: u64,
    /// Processor-specific flags.
    pub e_flags: u32,
    /// Number of program header table entries.
    pub e_phnum: u16,
    /// Number of section header table entries.
    pub e_shnum: u16,
    /// Section header table index of `.shstrtab`.
    pub e_shstrndx: u16,
}

impl Elf64Header {
    /// Serialize to 64 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        // e_ident[0..4]: magic
        buf.extend_from_slice(&ELF_MAGIC);
        // e_ident[4]: EI_CLASS = ELFCLASS64
        buf.push(ELFCLASS64);
        // e_ident[5]: EI_DATA = ELFDATA2LSB
        buf.push(ELFDATA2LSB);
        // e_ident[6]: EI_VERSION = EV_CURRENT
        buf.push(EV_CURRENT);
        // e_ident[7]: EI_OSABI = ELFOSABI_NONE
        buf.push(0);
        // e_ident[8..16]: EI_ABIVERSION + padding
        buf.extend_from_slice(&[0u8; 8]);
        // e_type
        buf.extend_from_slice(&self.e_type.to_le_bytes());
        // e_machine
        buf.extend_from_slice(&self.e_machine.to_le_bytes());
        // e_version
        buf.extend_from_slice(&1u32.to_le_bytes());
        // e_entry
        buf.extend_from_slice(&self.e_entry.to_le_bytes());
        // e_phoff
        buf.extend_from_slice(&self.e_phoff.to_le_bytes());
        // e_shoff
        buf.extend_from_slice(&self.e_shoff.to_le_bytes());
        // e_flags
        buf.extend_from_slice(&self.e_flags.to_le_bytes());
        // e_ehsize = 64 for ELF64
        buf.extend_from_slice(&64u16.to_le_bytes());
        // e_phentsize = 56 for ELF64
        buf.extend_from_slice(&56u16.to_le_bytes());
        // e_phnum
        buf.extend_from_slice(&self.e_phnum.to_le_bytes());
        // e_shentsize = 64 for ELF64
        buf.extend_from_slice(&64u16.to_le_bytes());
        // e_shnum
        buf.extend_from_slice(&self.e_shnum.to_le_bytes());
        // e_shstrndx
        buf.extend_from_slice(&self.e_shstrndx.to_le_bytes());
        debug_assert_eq!(buf.len(), 64);
        buf
    }
}

// ===========================================================================
// ELF32 Header (52 bytes)
// ===========================================================================

/// 32-bit ELF file header.
///
/// Serializes to exactly 52 bytes in little-endian format.
/// Used for i686 targets with ELFCLASS32.
#[derive(Debug, Clone, PartialEq)]
pub struct Elf32Header {
    /// Object file type (ET_REL, ET_EXEC, ET_DYN).
    pub e_type: u16,
    /// Architecture (EM_386).
    pub e_machine: u16,
    /// Entry point virtual address (32-bit).
    pub e_entry: u32,
    /// Program header table file offset (32-bit).
    pub e_phoff: u32,
    /// Section header table file offset (32-bit).
    pub e_shoff: u32,
    /// Processor-specific flags.
    pub e_flags: u32,
    /// Number of program header table entries.
    pub e_phnum: u16,
    /// Number of section header table entries.
    pub e_shnum: u16,
    /// Section header table index of `.shstrtab`.
    pub e_shstrndx: u16,
}

impl Elf32Header {
    /// Serialize to 52 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(52);
        // e_ident[0..4]: magic
        buf.extend_from_slice(&ELF_MAGIC);
        // e_ident[4]: EI_CLASS = ELFCLASS32
        buf.push(ELFCLASS32);
        // e_ident[5]: EI_DATA = ELFDATA2LSB
        buf.push(ELFDATA2LSB);
        // e_ident[6]: EI_VERSION = EV_CURRENT
        buf.push(EV_CURRENT);
        // e_ident[7]: EI_OSABI = ELFOSABI_NONE
        buf.push(0);
        // e_ident[8..16]: EI_ABIVERSION + padding
        buf.extend_from_slice(&[0u8; 8]);
        // e_type
        buf.extend_from_slice(&self.e_type.to_le_bytes());
        // e_machine
        buf.extend_from_slice(&self.e_machine.to_le_bytes());
        // e_version
        buf.extend_from_slice(&1u32.to_le_bytes());
        // e_entry
        buf.extend_from_slice(&self.e_entry.to_le_bytes());
        // e_phoff
        buf.extend_from_slice(&self.e_phoff.to_le_bytes());
        // e_shoff
        buf.extend_from_slice(&self.e_shoff.to_le_bytes());
        // e_flags
        buf.extend_from_slice(&self.e_flags.to_le_bytes());
        // e_ehsize = 52 for ELF32
        buf.extend_from_slice(&52u16.to_le_bytes());
        // e_phentsize = 32 for ELF32
        buf.extend_from_slice(&32u16.to_le_bytes());
        // e_phnum
        buf.extend_from_slice(&self.e_phnum.to_le_bytes());
        // e_shentsize = 40 for ELF32
        buf.extend_from_slice(&40u16.to_le_bytes());
        // e_shnum
        buf.extend_from_slice(&self.e_shnum.to_le_bytes());
        // e_shstrndx
        buf.extend_from_slice(&self.e_shstrndx.to_le_bytes());
        debug_assert_eq!(buf.len(), 52);
        buf
    }
}

// ===========================================================================
// ELF64 Section Header (64 bytes)
// ===========================================================================

/// 64-bit ELF section header entry.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf64SectionHeader {
    /// Offset into the section header string table for this section's name.
    pub sh_name: u32,
    /// Section type (SHT_*).
    pub sh_type: u32,
    /// Section attribute flags (SHF_*).
    pub sh_flags: u64,
    /// Virtual address of the section in memory (0 for non-loaded sections).
    pub sh_addr: u64,
    /// File offset of the section data.
    pub sh_offset: u64,
    /// Size of the section in the file (0 for SHT_NOBITS in file, but
    /// set to the in-memory size for `.bss`).
    pub sh_size: u64,
    /// Section header table index link (meaning depends on sh_type).
    pub sh_link: u32,
    /// Extra information (meaning depends on sh_type).
    pub sh_info: u32,
    /// Required alignment of the section.
    pub sh_addralign: u64,
    /// Size of each entry for fixed-size entry sections (0 otherwise).
    pub sh_entsize: u64,
}

impl Elf64SectionHeader {
    /// Serialize to 64 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.sh_name.to_le_bytes());
        buf.extend_from_slice(&self.sh_type.to_le_bytes());
        buf.extend_from_slice(&self.sh_flags.to_le_bytes());
        buf.extend_from_slice(&self.sh_addr.to_le_bytes());
        buf.extend_from_slice(&self.sh_offset.to_le_bytes());
        buf.extend_from_slice(&self.sh_size.to_le_bytes());
        buf.extend_from_slice(&self.sh_link.to_le_bytes());
        buf.extend_from_slice(&self.sh_info.to_le_bytes());
        buf.extend_from_slice(&self.sh_addralign.to_le_bytes());
        buf.extend_from_slice(&self.sh_entsize.to_le_bytes());
        debug_assert_eq!(buf.len(), 64);
        buf
    }
}

// ===========================================================================
// ELF32 Section Header (40 bytes)
// ===========================================================================

/// 32-bit ELF section header entry.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf32SectionHeader {
    /// Offset into the section header string table for this section's name.
    pub sh_name: u32,
    /// Section type (SHT_*).
    pub sh_type: u32,
    /// Section attribute flags (SHF_*). Stored as u32 for ELF32.
    pub sh_flags: u32,
    /// Virtual address of the section in memory.
    pub sh_addr: u32,
    /// File offset of the section data.
    pub sh_offset: u32,
    /// Size of the section.
    pub sh_size: u32,
    /// Section header table index link.
    pub sh_link: u32,
    /// Extra information.
    pub sh_info: u32,
    /// Required alignment of the section.
    pub sh_addralign: u32,
    /// Size of each entry for fixed-size entry sections.
    pub sh_entsize: u32,
}

impl Elf32SectionHeader {
    /// Serialize to 40 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(40);
        buf.extend_from_slice(&self.sh_name.to_le_bytes());
        buf.extend_from_slice(&self.sh_type.to_le_bytes());
        buf.extend_from_slice(&self.sh_flags.to_le_bytes());
        buf.extend_from_slice(&self.sh_addr.to_le_bytes());
        buf.extend_from_slice(&self.sh_offset.to_le_bytes());
        buf.extend_from_slice(&self.sh_size.to_le_bytes());
        buf.extend_from_slice(&self.sh_link.to_le_bytes());
        buf.extend_from_slice(&self.sh_info.to_le_bytes());
        buf.extend_from_slice(&self.sh_addralign.to_le_bytes());
        buf.extend_from_slice(&self.sh_entsize.to_le_bytes());
        debug_assert_eq!(buf.len(), 40);
        buf
    }
}

// ===========================================================================
// ELF64 Program Header (56 bytes)
// ===========================================================================

/// 64-bit ELF program header entry.
///
/// Note: In ELF64, `p_flags` appears **after** `p_type` and **before**
/// `p_offset`, which differs from the ELF32 layout where `p_flags`
/// appears after `p_memsz`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf64ProgramHeader {
    /// Segment type (PT_*).
    pub p_type: u32,
    /// Segment-dependent flags (PF_X=1, PF_W=2, PF_R=4).
    pub p_flags: u32,
    /// Offset from the beginning of the file.
    pub p_offset: u64,
    /// Virtual address in memory.
    pub p_vaddr: u64,
    /// Physical address (usually same as vaddr on Linux).
    pub p_paddr: u64,
    /// Size of the segment in the file image.
    pub p_filesz: u64,
    /// Size of the segment in memory (may be larger than filesz for BSS).
    pub p_memsz: u64,
    /// Alignment of the segment (must be a power of 2).
    pub p_align: u64,
}

impl Elf64ProgramHeader {
    /// Serialize to 56 little-endian bytes.
    ///
    /// ELF64 layout: p_type, p_flags, p_offset, p_vaddr, p_paddr,
    /// p_filesz, p_memsz, p_align.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(56);
        buf.extend_from_slice(&self.p_type.to_le_bytes());
        buf.extend_from_slice(&self.p_flags.to_le_bytes());
        buf.extend_from_slice(&self.p_offset.to_le_bytes());
        buf.extend_from_slice(&self.p_vaddr.to_le_bytes());
        buf.extend_from_slice(&self.p_paddr.to_le_bytes());
        buf.extend_from_slice(&self.p_filesz.to_le_bytes());
        buf.extend_from_slice(&self.p_memsz.to_le_bytes());
        buf.extend_from_slice(&self.p_align.to_le_bytes());
        debug_assert_eq!(buf.len(), 56);
        buf
    }
}

// ===========================================================================
// ELF32 Program Header (32 bytes)
// ===========================================================================

/// 32-bit ELF program header entry.
///
/// Note: In ELF32, `p_flags` appears **after** `p_memsz`, which differs
/// from the ELF64 layout where `p_flags` is the second field.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf32ProgramHeader {
    /// Segment type (PT_*).
    pub p_type: u32,
    /// Segment-dependent flags (PF_X=1, PF_W=2, PF_R=4).
    pub p_flags: u32,
    /// Offset from the beginning of the file.
    pub p_offset: u32,
    /// Virtual address in memory.
    pub p_vaddr: u32,
    /// Physical address.
    pub p_paddr: u32,
    /// Size of the segment in the file image.
    pub p_filesz: u32,
    /// Size of the segment in memory.
    pub p_memsz: u32,
    /// Alignment of the segment.
    pub p_align: u32,
}

impl Elf32ProgramHeader {
    /// Serialize to 32 little-endian bytes.
    ///
    /// ELF32 layout: p_type, p_offset, p_vaddr, p_paddr, p_filesz,
    /// p_memsz, p_flags, p_align.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&self.p_type.to_le_bytes());
        // NOTE: ELF32 has p_offset here, NOT p_flags
        buf.extend_from_slice(&self.p_offset.to_le_bytes());
        buf.extend_from_slice(&self.p_vaddr.to_le_bytes());
        buf.extend_from_slice(&self.p_paddr.to_le_bytes());
        buf.extend_from_slice(&self.p_filesz.to_le_bytes());
        buf.extend_from_slice(&self.p_memsz.to_le_bytes());
        // p_flags comes AFTER p_memsz in ELF32
        buf.extend_from_slice(&self.p_flags.to_le_bytes());
        buf.extend_from_slice(&self.p_align.to_le_bytes());
        debug_assert_eq!(buf.len(), 32);
        buf
    }
}

// ===========================================================================
// ELF64 Symbol Table Entry (24 bytes)
// ===========================================================================

/// 64-bit ELF symbol table entry.
///
/// Note: The ELF64 symbol layout places `st_info`, `st_other`, and
/// `st_shndx` **before** `st_value` and `st_size`, which differs from
/// the ELF32 layout.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf64Symbol {
    /// Symbol name index into the associated string table.
    pub st_name: u32,
    /// Symbol type and binding: `(binding << 4) | sym_type`.
    pub st_info: u8,
    /// Symbol visibility (STV_*).
    pub st_other: u8,
    /// Section header table index this symbol is defined in.
    pub st_shndx: u16,
    /// Symbol value (address or offset).
    pub st_value: u64,
    /// Symbol size in bytes.
    pub st_size: u64,
}

impl Elf64Symbol {
    /// Serialize to 24 little-endian bytes.
    ///
    /// ELF64 layout: st_name(4), st_info(1), st_other(1), st_shndx(2),
    /// st_value(8), st_size(8).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.st_name.to_le_bytes());
        buf.push(self.st_info);
        buf.push(self.st_other);
        buf.extend_from_slice(&self.st_shndx.to_le_bytes());
        buf.extend_from_slice(&self.st_value.to_le_bytes());
        buf.extend_from_slice(&self.st_size.to_le_bytes());
        debug_assert_eq!(buf.len(), 24);
        buf
    }
}

// ===========================================================================
// ELF32 Symbol Table Entry (16 bytes)
// ===========================================================================

/// 32-bit ELF symbol table entry.
///
/// Note: The ELF32 symbol layout places `st_value` and `st_size`
/// **before** `st_info`, `st_other`, and `st_shndx`, which differs
/// from the ELF64 layout.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf32Symbol {
    /// Symbol name index into the associated string table.
    pub st_name: u32,
    /// Symbol value (address or offset, 32-bit).
    pub st_value: u32,
    /// Symbol size in bytes (32-bit).
    pub st_size: u32,
    /// Symbol type and binding: `(binding << 4) | sym_type`.
    pub st_info: u8,
    /// Symbol visibility (STV_*).
    pub st_other: u8,
    /// Section header table index this symbol is defined in.
    pub st_shndx: u16,
}

impl Elf32Symbol {
    /// Serialize to 16 little-endian bytes.
    ///
    /// ELF32 layout: st_name(4), st_value(4), st_size(4),
    /// st_info(1), st_other(1), st_shndx(2).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.st_name.to_le_bytes());
        buf.extend_from_slice(&self.st_value.to_le_bytes());
        buf.extend_from_slice(&self.st_size.to_le_bytes());
        buf.push(self.st_info);
        buf.push(self.st_other);
        buf.extend_from_slice(&self.st_shndx.to_le_bytes());
        debug_assert_eq!(buf.len(), 16);
        buf
    }
}

// ===========================================================================
// ELF64 Relocation with Addend (24 bytes)
// ===========================================================================

/// 64-bit ELF relocation entry with explicit addend (`.rela.*` sections).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf64Rela {
    /// Offset of the relocation target within the section.
    pub r_offset: u64,
    /// Relocation info: `(sym_index << 32) | rel_type`.
    pub r_info: u64,
    /// Constant addend to compute the relocation value.
    pub r_addend: i64,
}

impl Elf64Rela {
    /// Create from separate symbol index and relocation type.
    pub fn new(offset: u64, sym_index: u32, rel_type: u32, addend: i64) -> Self {
        Self {
            r_offset: offset,
            r_info: ((sym_index as u64) << 32) | (rel_type as u64),
            r_addend: addend,
        }
    }

    /// Serialize to 24 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.r_offset.to_le_bytes());
        buf.extend_from_slice(&self.r_info.to_le_bytes());
        buf.extend_from_slice(&self.r_addend.to_le_bytes());
        debug_assert_eq!(buf.len(), 24);
        buf
    }
}

// ===========================================================================
// ELF32 Relocation without Addend (8 bytes)
// ===========================================================================

/// 32-bit ELF relocation entry without explicit addend (`.rel.*` sections).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Elf32Rel {
    /// Offset of the relocation target within the section.
    pub r_offset: u32,
    /// Relocation info: `(sym_index << 8) | rel_type`.
    pub r_info: u32,
}

impl Elf32Rel {
    /// Create from separate symbol index and relocation type.
    pub fn new(offset: u32, sym_index: u32, rel_type: u8) -> Self {
        Self {
            r_offset: offset,
            r_info: (sym_index << 8) | (rel_type as u32),
        }
    }

    /// Serialize to 8 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&self.r_offset.to_le_bytes());
        buf.extend_from_slice(&self.r_info.to_le_bytes());
        debug_assert_eq!(buf.len(), 8);
        buf
    }
}

/// 32-bit ELF relocation entry with addend (`.rela.*` sections).
///
/// Each `Elf32Rela` is a 12-byte structure:
/// - `r_offset`: Virtual address of the relocation target.
/// - `r_info`: `(sym_index << 8) | rel_type`.
/// - `r_addend`: Signed addend value.
pub struct Elf32Rela {
    /// Offset of the relocation target within the section.
    pub r_offset: u32,
    /// Relocation info: `(sym_index << 8) | rel_type`.
    pub r_info: u32,
    /// Explicit addend for the relocation computation.
    pub r_addend: i32,
}

impl Elf32Rela {
    /// Create from separate symbol index, relocation type, and addend.
    pub fn new(offset: u32, sym_index: u32, rel_type: u8, addend: i32) -> Self {
        Self {
            r_offset: offset,
            r_info: (sym_index << 8) | (rel_type as u32),
            r_addend: addend,
        }
    }

    /// Serialize to 12 little-endian bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12);
        buf.extend_from_slice(&self.r_offset.to_le_bytes());
        buf.extend_from_slice(&self.r_info.to_le_bytes());
        buf.extend_from_slice(&self.r_addend.to_le_bytes());
        debug_assert_eq!(buf.len(), 12);
        buf
    }
}

// ===========================================================================
// StringTable — .strtab / .shstrtab Builder
// ===========================================================================

/// ELF string table builder for `.strtab` and `.shstrtab` sections.
///
/// Strings are stored contiguously as null-terminated byte sequences.
/// The table always begins with a null byte at offset 0 (the empty
/// string). Duplicate strings are coalesced via an [`FxHashMap`]-backed
/// lookup, yielding O(1) average-case deduplication — critical for
/// large kernel builds with many symbols.
#[derive(Debug, Clone)]
pub struct StringTable {
    /// Raw byte buffer holding the concatenated null-terminated strings.
    data: Vec<u8>,
    /// Deduplication map: string content → byte offset within `data`.
    map: FxHashMap<String, u32>,
}

impl StringTable {
    /// Create a new string table with only the mandatory null byte at offset 0.
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        // The empty string maps to offset 0 (the initial null byte).
        map.insert(String::new(), 0);
        Self { data: vec![0], map }
    }

    /// Add a string to the table, returning its byte offset.
    ///
    /// If the string was already added, the existing offset is returned
    /// without duplicating the data. The empty string always returns 0.
    pub fn add_string(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.map.get(s) {
            return offset;
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0); // null terminator
        self.map.insert(s.to_owned(), offset);
        offset
    }

    /// Check whether a string is already present in the table.
    pub fn contains(&self, s: &str) -> bool {
        self.map.contains_key(s)
    }

    /// Look up the offset of a previously added string.
    ///
    /// Returns `None` if the string has not been added.
    pub fn get_offset(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }

    /// Return the raw byte contents of the string table for section data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Return the total byte length of the string table.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the string table contains only the initial null byte.
    pub fn is_empty(&self) -> bool {
        self.data.len() <= 1
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// ElfSymbol — High-Level Symbol Representation
// ===========================================================================

/// High-level ELF symbol representation used by the compiler.
///
/// This is a target-independent view. The [`SymbolTable`] builder converts
/// these into either [`Elf64Symbol`] or [`Elf32Symbol`] entries during
/// binary serialization.
#[derive(Debug, Clone, PartialEq)]
pub struct ElfSymbol {
    /// Symbol name (stored in the associated string table).
    pub name: String,
    /// Symbol value (virtual address or offset).
    pub value: u64,
    /// Symbol size in bytes.
    pub size: u64,
    /// Symbol binding (STB_LOCAL, STB_GLOBAL, STB_WEAK).
    pub binding: u8,
    /// Symbol type (STT_NOTYPE, STT_FUNC, STT_OBJECT, etc.).
    pub sym_type: u8,
    /// Symbol visibility (STV_DEFAULT, STV_HIDDEN, STV_PROTECTED).
    pub visibility: u8,
    /// Section index this symbol is defined in (0 = SHN_UNDEF).
    pub section_index: u16,
}

impl Default for ElfSymbol {
    fn default() -> Self {
        Self {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: SHN_UNDEF,
        }
    }
}

// ===========================================================================
// SymbolTable — .symtab Builder
// ===========================================================================

/// ELF symbol table builder.
///
/// Manages the collection of symbols and their associated string table.
/// During serialization, symbols are reordered so that all `STB_LOCAL`
/// symbols precede all `STB_GLOBAL`/`STB_WEAK` symbols — an ELF
/// specification requirement. The `sh_info` field of the `.symtab`
/// section header is set to [`first_global_index()`], indicating the
/// boundary between local and non-local symbols.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Symbols added by the user (not yet reordered).
    symbols: Vec<ElfSymbol>,
    /// String table holding symbol names.
    string_table: StringTable,
}

impl SymbolTable {
    /// Create an empty symbol table.
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            string_table: StringTable::new(),
        }
    }

    /// Add a symbol to the table.
    ///
    /// The symbol's name is automatically interned in the associated
    /// string table.
    pub fn add_symbol(&mut self, sym: ElfSymbol) {
        self.string_table.add_string(&sym.name);
        self.symbols.push(sym);
    }

    /// Compute the 1-based index of the first non-local (global/weak)
    /// symbol, accounting for the mandatory null symbol at index 0.
    ///
    /// This value is stored in `sh_info` of the `.symtab` section header.
    pub fn first_global_index(&self) -> u32 {
        let num_locals = self
            .symbols
            .iter()
            .filter(|s| s.binding == STB_LOCAL)
            .count();
        // +1 for the null symbol at index 0
        (num_locals as u32) + 1
    }

    /// Return a reference to the symbol list.
    pub fn symbols(&self) -> &[ElfSymbol] {
        &self.symbols
    }

    /// Return a reference to the associated string table.
    pub fn string_table(&self) -> &StringTable {
        &self.string_table
    }

    /// Serialize all symbols into binary format suitable for a `.symtab`
    /// section, respecting the local-before-global ordering.
    ///
    /// Returns `(serialized_bytes, first_global_index)`.
    ///
    /// The `is_64` flag selects between [`Elf64Symbol`] and [`Elf32Symbol`]
    /// encoding.
    pub fn build_bytes(&self, is_64: bool) -> (Vec<u8>, u32) {
        // Separate locals and globals.
        let (locals, globals): (Vec<&ElfSymbol>, Vec<&ElfSymbol>) =
            self.symbols.iter().partition(|s| s.binding == STB_LOCAL);

        let total_count = 1 + locals.len() + globals.len(); // +1 for null
        let entry_size = if is_64 { 24 } else { 16 };
        let mut bytes = Vec::with_capacity(total_count * entry_size);

        // Null symbol at index 0.
        if is_64 {
            bytes.extend_from_slice(&Elf64Symbol::default().to_bytes());
        } else {
            bytes.extend_from_slice(&Elf32Symbol::default().to_bytes());
        }

        let first_global = (1 + locals.len()) as u32;

        // Write local symbols, then global symbols.
        for sym in locals.iter().chain(globals.iter()) {
            let name_offset = self.string_table.get_offset(&sym.name).unwrap_or(0);
            let st_info = (sym.binding << 4) | (sym.sym_type & 0x0f);
            let st_other = sym.visibility & 0x03;

            if is_64 {
                let raw = Elf64Symbol {
                    st_name: name_offset,
                    st_info,
                    st_other,
                    st_shndx: sym.section_index,
                    st_value: sym.value,
                    st_size: sym.size,
                };
                bytes.extend_from_slice(&raw.to_bytes());
            } else {
                let raw = Elf32Symbol {
                    st_name: name_offset,
                    st_value: sym.value as u32,
                    st_size: sym.size as u32,
                    st_info,
                    st_other,
                    st_shndx: sym.section_index,
                };
                bytes.extend_from_slice(&raw.to_bytes());
            }
        }

        (bytes, first_global)
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Section — Generic ELF Section Container
// ===========================================================================

/// Generic section container used to build user-defined ELF sections.
///
/// Callers populate instances with section metadata and raw data bytes;
/// the [`ElfWriter`] handles computing file offsets and writing section
/// header entries.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name (stored in `.shstrtab`).
    pub name: String,
    /// Section type (SHT_PROGBITS, SHT_NOBITS, SHT_NOTE, etc.).
    pub sh_type: u32,
    /// Section attribute flags (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR, etc.).
    pub sh_flags: u64,
    /// Section content bytes. For SHT_NOBITS, this can be an empty `Vec`
    /// since no bytes are written to the file — use `logical_size` to set
    /// the in-memory size instead.
    pub data: Vec<u8>,
    /// Associated section index (meaning depends on sh_type).
    pub sh_link: u32,
    /// Extra information (meaning depends on sh_type).
    pub sh_info: u32,
    /// Required alignment. Must be a power of two (or 0/1 for none).
    pub sh_addralign: u64,
    /// Entry size for fixed-size entry tables (0 for variable-size).
    pub sh_entsize: u64,
    /// Explicit logical (in-memory) size override.  When non-zero, this
    /// value is used as `sh_size` instead of `data.len()`.  This is useful
    /// for `SHT_NOBITS` sections (e.g., `.bss`) where the in-memory size
    /// is non-zero but no bytes need to be stored in the data vector.
    /// When zero, `data.len()` is used as the logical size.
    pub logical_size: u64,
    /// Virtual address hint for linked output (ET_EXEC / ET_DYN).
    /// When non-zero, used as `sh_addr` in the section header, and
    /// the section data is placed at `file_offset_hint` in the file.
    /// When zero, the section is placed sequentially after the preceding
    /// section with `sh_addr = 0` (relocatable mode).
    pub virtual_address: u64,
    /// File offset hint for linked output (ET_EXEC / ET_DYN).
    /// When `virtual_address` is non-zero, the ELF writer places this
    /// section's data at exactly this file offset, inserting padding as
    /// necessary.  This ensures the file layout matches the program
    /// headers produced by the linker script.
    pub file_offset_hint: u64,
}

impl Default for Section {
    fn default() -> Self {
        Self {
            name: String::new(),
            sh_type: SHT_NULL,
            sh_flags: 0,
            data: Vec::new(),
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 1,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        }
    }
}

// ===========================================================================
// Relocation — Architecture-Independent Relocation Entry
// ===========================================================================

/// Architecture-independent relocation entry.
///
/// This is a high-level representation used throughout the compiler.
/// Convert to [`Elf64Rela`] or [`Elf32Rel`] for binary serialization
/// in `.rela.*` or `.rel.*` sections.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Relocation {
    /// Byte offset within the section where the relocation applies.
    pub offset: u64,
    /// Symbol table index of the target symbol.
    pub sym_index: u32,
    /// Architecture-specific relocation type.
    pub rel_type: u32,
    /// Constant addend (used for RELA relocations; ignored for REL).
    pub addend: i64,
}

impl Relocation {
    /// Convert to an [`Elf64Rela`] entry for `.rela.*` sections.
    pub fn to_elf64_rela(&self) -> Elf64Rela {
        Elf64Rela::new(self.offset, self.sym_index, self.rel_type, self.addend)
    }

    /// Convert to an [`Elf32Rel`] entry for `.rel.*` sections.
    ///
    /// The addend is dropped (ELF32 REL relocations use implicit addends
    /// stored at the relocation site).
    pub fn to_elf32_rel(&self) -> Elf32Rel {
        Elf32Rel::new(self.offset as u32, self.sym_index, self.rel_type as u8)
    }
}

// ===========================================================================
// ElfWriter — Top-Level ELF Binary Writer
// ===========================================================================

/// Top-level ELF binary writer.
///
/// Assembles a complete ELF file from sections, symbols, program headers,
/// and metadata for any of BCC's four target architectures. Supports
/// `ET_REL` (relocatable `.o`), `ET_EXEC` (static executable), and
/// `ET_DYN` (shared object) output.
///
/// # Usage
///
/// ```ignore
/// use bcc::backend::elf_writer_common::*;
/// use bcc::common::target::Target;
///
/// let mut writer = ElfWriter::new(Target::X86_64, ET_REL);
///
/// // Add a .text section
/// writer.add_section(Section {
///     name: ".text".to_string(),
///     sh_type: SHT_PROGBITS,
///     sh_flags: SHF_ALLOC | SHF_EXECINSTR,
///     data: vec![0xcc], // int3
///     sh_addralign: 16,
///     ..Section::default()
/// });
///
/// // Add a symbol
/// writer.add_symbol(ElfSymbol {
///     name: "_start".to_string(),
///     binding: STB_GLOBAL,
///     sym_type: STT_FUNC,
///     section_index: 1,
///     ..ElfSymbol::default()
/// });
///
/// let elf_bytes = writer.write();
/// ```
#[derive(Debug)]
pub struct ElfWriter {
    /// Target architecture.
    target: Target,
    /// ELF object type (ET_REL, ET_EXEC, or ET_DYN).
    elf_type: u16,
    /// User-defined sections (indexed starting from 1 in the final ELF).
    sections: Vec<Section>,
    /// Symbol table builder.
    symbols: SymbolTable,
    /// Program headers (for ET_EXEC / ET_DYN).
    program_headers: Vec<Elf64ProgramHeader>,
    /// Entry point virtual address.
    entry_point: u64,
}

impl ElfWriter {
    /// Create a new ELF writer for the given target and object type.
    ///
    /// # Arguments
    ///
    /// * `target` — Target architecture (determines ELF class, machine, flags).
    /// * `elf_type` — Object file type: [`ET_REL`], [`ET_EXEC`], or [`ET_DYN`].
    pub fn new(target: Target, elf_type: u16) -> Self {
        Self {
            target,
            elf_type,
            sections: Vec::new(),
            symbols: SymbolTable::new(),
            program_headers: Vec::new(),
            entry_point: 0,
        }
    }

    /// Add a section and return its 1-based index in the section header
    /// table (index 0 is the mandatory null section).
    pub fn add_section(&mut self, section: Section) -> usize {
        self.sections.push(section);
        self.sections.len() // 1-based index
    }

    /// Add a symbol to the symbol table.
    pub fn add_symbol(&mut self, sym: ElfSymbol) {
        self.symbols.add_symbol(sym);
    }

    /// Set the entry point virtual address for the ELF header.
    pub fn set_entry_point(&mut self, addr: u64) {
        self.entry_point = addr;
    }

    /// Add a program header segment descriptor.
    ///
    /// Program headers are only meaningful for `ET_EXEC` and `ET_DYN`
    /// output. They are stored as [`Elf64ProgramHeader`] internally;
    /// for 32-bit targets, fields are truncated during serialization.
    pub fn add_program_header(&mut self, phdr: Elf64ProgramHeader) {
        self.program_headers.push(phdr);
    }

    /// Return the number of user-defined sections (excluding the null
    /// section and internal bookkeeping sections).
    pub fn sections_count(&self) -> usize {
        self.sections.len()
    }

    /// Return a mutable reference to the internal sections vector.
    ///
    /// This allows post-hoc fixup of section header fields (e.g. `sh_link`,
    /// `sh_info`) after all sections have been added and their indices are
    /// known.  The vector uses **0-based** indexing into the user sections;
    /// the ELF section header index is `vector_index + 1` (because section 0
    /// is the mandatory null section).
    pub fn sections_mut(&mut self) -> &mut Vec<Section> {
        &mut self.sections
    }

    /// Return the target architecture.
    pub fn target(&self) -> Target {
        self.target
    }

    /// Return the dynamic linker path for the target architecture.
    ///
    /// Used when constructing `PT_INTERP` program headers for
    /// `ET_EXEC` and `ET_DYN` output.
    pub fn dynamic_linker(&self) -> &str {
        self.target.dynamic_linker()
    }

    /// Return the target page size for segment alignment.
    ///
    /// Used when constructing `PT_LOAD` program headers to ensure
    /// correct `p_align` values.
    pub fn page_size(&self) -> usize {
        self.target.page_size()
    }

    /// Serialize the complete ELF binary to a byte vector.
    ///
    /// The output byte vector is a valid ELF file that can be parsed by
    /// `readelf`, `objdump`, and the BCC built-in linker. The layout is:
    ///
    /// 1. ELF header
    /// 2. Program headers (ET_EXEC / ET_DYN only)
    /// 3. Section data (padded for alignment)
    /// 4. Section header table
    pub fn write(&self) -> Vec<u8> {
        let is_64 = self.target.is_64bit();

        // Validate consistency between is_64bit() and elf_class().
        debug_assert_eq!(
            self.target.elf_class(),
            if is_64 { ELFCLASS64 } else { ELFCLASS32 },
            "elf_class() must agree with is_64bit()"
        );
        // All BCC targets are little-endian.
        debug_assert_eq!(
            self.target.elf_data(),
            ELFDATA2LSB,
            "all BCC targets must be little-endian"
        );

        let ptr_width = self.target.pointer_width();
        debug_assert!(ptr_width == 4 || ptr_width == 8);

        let ehdr_size: usize = if is_64 { 64 } else { 52 };
        let phdr_entry_size: usize = if is_64 { 56 } else { 32 };
        let shdr_entry_size: usize = if is_64 { 64 } else { 40 };
        let sym_entry_size: usize = if is_64 { 24 } else { 16 };

        // -----------------------------------------------------------------
        // Phase 1: Build internal data structures
        // -----------------------------------------------------------------

        // Build section header string table locally.
        let mut shstrtab = StringTable::new();

        // Reserve name offsets for all sections (including null at index 0).
        let mut section_name_offsets: Vec<u32> = Vec::with_capacity(self.sections.len() + 4);
        section_name_offsets.push(shstrtab.add_string("")); // null section

        for sec in &self.sections {
            section_name_offsets.push(shstrtab.add_string(&sec.name));
        }

        let symtab_name_off = shstrtab.add_string(".symtab");
        let strtab_name_off = shstrtab.add_string(".strtab");
        let shstrtab_name_off = shstrtab.add_string(".shstrtab");

        // Build binary symbol table.
        let (sym_bytes, first_global_idx) = self.symbols.build_bytes(is_64);
        let strtab_bytes = self.symbols.string_table().as_bytes().to_vec();
        let shstrtab_bytes = shstrtab.as_bytes().to_vec();

        // -----------------------------------------------------------------
        // Phase 2: Compute section indices
        // -----------------------------------------------------------------

        let num_user_sections = self.sections.len();
        let symtab_section_idx = num_user_sections + 1;
        let strtab_section_idx = num_user_sections + 2;
        let shstrtab_section_idx = num_user_sections + 3;
        // null + user + symtab + strtab + shstrtab
        let total_sections = num_user_sections + 4;

        // -----------------------------------------------------------------
        // Phase 3: Compute file layout (offsets)
        // -----------------------------------------------------------------

        let num_phdrs = self.program_headers.len();
        let phdrs_total_size = num_phdrs * phdr_entry_size;

        let mut current_offset = ehdr_size;
        if num_phdrs > 0 {
            current_offset += phdrs_total_size;
        }

        // Track (file_offset, logical_size, virtual_address) for every section.
        let mut section_file_offsets: Vec<(u64, u64, u64)> = Vec::with_capacity(total_sections);
        section_file_offsets.push((0, 0, 0)); // null section

        for sec in &self.sections {
            // Use the explicit logical_size if non-zero, otherwise data.len().
            let logical_size = if sec.logical_size > 0 {
                sec.logical_size
            } else {
                sec.data.len() as u64
            };

            // For linked output, use the file_offset_hint if provided.
            // This ensures the file layout matches the program headers.
            if sec.file_offset_hint > 0 {
                current_offset = sec.file_offset_hint as usize;
                section_file_offsets.push((
                    sec.file_offset_hint,
                    logical_size,
                    sec.virtual_address,
                ));
                if sec.sh_type != SHT_NOBITS {
                    current_offset += sec.data.len();
                }
            } else {
                let align = (sec.sh_addralign as usize).max(1);
                current_offset = align_up(current_offset, align);
                section_file_offsets.push((
                    current_offset as u64,
                    logical_size,
                    sec.virtual_address,
                ));
                // NOBITS sections take no file space.
                if sec.sh_type != SHT_NOBITS {
                    current_offset += sec.data.len();
                }
            }
        }

        // When file_offset_hint is used, current_offset tracks the LAST
        // hinted position, which may not be the MAXIMUM.  Ensure we place
        // .symtab/.strtab/.shstrtab and the section header table after ALL
        // section data by choosing the maximum across every section's end.
        let ptr_align = ptr_width;
        {
            let mut max_end = current_offset;
            for (i, sec) in self.sections.iter().enumerate() {
                let (off, _, _) = section_file_offsets[i + 1];
                let data_len = if sec.sh_type != SHT_NOBITS {
                    sec.data.len()
                } else {
                    0
                };
                let end = off as usize + data_len;
                if end > max_end {
                    max_end = end;
                }
            }
            current_offset = max_end;
        }

        // .symtab — aligned to pointer width
        current_offset = align_up(current_offset, ptr_align);
        section_file_offsets.push((current_offset as u64, sym_bytes.len() as u64, 0));
        current_offset += sym_bytes.len();

        // .strtab — alignment 1
        section_file_offsets.push((current_offset as u64, strtab_bytes.len() as u64, 0));
        current_offset += strtab_bytes.len();

        // .shstrtab — alignment 1
        section_file_offsets.push((current_offset as u64, shstrtab_bytes.len() as u64, 0));
        current_offset += shstrtab_bytes.len();

        // Section header table — aligned to pointer width
        current_offset = align_up(current_offset, ptr_align);
        let shdr_table_offset = current_offset;

        let total_file_size = shdr_table_offset + total_sections * shdr_entry_size;

        // -----------------------------------------------------------------
        // Phase 4: Write ELF header
        // -----------------------------------------------------------------

        let mut buf = Vec::with_capacity(total_file_size);

        let phdr_offset_val = if num_phdrs > 0 { ehdr_size } else { 0 };

        if is_64 {
            let hdr = Elf64Header {
                e_type: self.elf_type,
                e_machine: self.target.elf_machine(),
                e_entry: self.entry_point,
                e_phoff: phdr_offset_val as u64,
                e_shoff: shdr_table_offset as u64,
                e_flags: self.target.elf_flags(),
                e_phnum: num_phdrs as u16,
                e_shnum: total_sections as u16,
                e_shstrndx: shstrtab_section_idx as u16,
            };
            buf.extend_from_slice(&hdr.to_bytes());
        } else {
            let hdr = Elf32Header {
                e_type: self.elf_type,
                e_machine: self.target.elf_machine(),
                e_entry: self.entry_point as u32,
                e_phoff: phdr_offset_val as u32,
                e_shoff: shdr_table_offset as u32,
                e_flags: self.target.elf_flags(),
                e_phnum: num_phdrs as u16,
                e_shnum: total_sections as u16,
                e_shstrndx: shstrtab_section_idx as u16,
            };
            buf.extend_from_slice(&hdr.to_bytes());
        }

        // -----------------------------------------------------------------
        // Phase 5: Write program headers
        // -----------------------------------------------------------------

        for phdr in &self.program_headers {
            if is_64 {
                buf.extend_from_slice(&phdr.to_bytes());
            } else {
                let phdr32 = Elf32ProgramHeader {
                    p_type: phdr.p_type,
                    p_flags: phdr.p_flags,
                    p_offset: phdr.p_offset as u32,
                    p_vaddr: phdr.p_vaddr as u32,
                    p_paddr: phdr.p_paddr as u32,
                    p_filesz: phdr.p_filesz as u32,
                    p_memsz: phdr.p_memsz as u32,
                    p_align: phdr.p_align as u32,
                };
                buf.extend_from_slice(&phdr32.to_bytes());
            }
        }

        // -----------------------------------------------------------------
        // Phase 6: Write section data
        // -----------------------------------------------------------------
        // Write sections in ascending file-offset order to ensure `pad_to`
        // always moves the buffer forward.  Sections with `file_offset_hint`
        // may have been added to the writer in an order that does not match
        // their file layout, so we sort indices by target offset first.
        {
            let mut write_order: Vec<usize> = (0..self.sections.len()).collect();
            write_order.sort_by_key(|&i| section_file_offsets[i + 1].0);
            for &i in &write_order {
                let (target_offset, _, _) = section_file_offsets[i + 1];
                pad_to(&mut buf, target_offset as usize);
                if self.sections[i].sh_type != SHT_NOBITS {
                    buf.extend_from_slice(&self.sections[i].data);
                }
            }
        }

        // Write .symtab data
        pad_to(
            &mut buf,
            section_file_offsets[symtab_section_idx].0 as usize,
        );
        buf.extend_from_slice(&sym_bytes);

        // Write .strtab data
        pad_to(
            &mut buf,
            section_file_offsets[strtab_section_idx].0 as usize,
        );
        buf.extend_from_slice(&strtab_bytes);

        // Write .shstrtab data
        pad_to(
            &mut buf,
            section_file_offsets[shstrtab_section_idx].0 as usize,
        );
        buf.extend_from_slice(&shstrtab_bytes);

        // -----------------------------------------------------------------
        // Phase 7: Write section header table
        // -----------------------------------------------------------------

        pad_to(&mut buf, shdr_table_offset);

        // Section 0: null entry (all zeros)
        self.write_shdr(&mut buf, is_64, &Elf64SectionHeader::default());

        // User sections
        for (i, sec) in self.sections.iter().enumerate() {
            let (file_off, logical_size, virt_addr) = section_file_offsets[i + 1];
            let name_off = section_name_offsets[i + 1];

            let shdr = Elf64SectionHeader {
                sh_name: name_off,
                sh_type: sec.sh_type,
                sh_flags: sec.sh_flags,
                sh_addr: virt_addr,
                sh_offset: file_off,
                sh_size: logical_size,
                sh_link: sec.sh_link,
                sh_info: sec.sh_info,
                sh_addralign: sec.sh_addralign,
                sh_entsize: sec.sh_entsize,
            };
            self.write_shdr(&mut buf, is_64, &shdr);
        }

        // .symtab section header
        {
            let (off, size, _) = section_file_offsets[symtab_section_idx];
            let shdr = Elf64SectionHeader {
                sh_name: symtab_name_off,
                sh_type: SHT_SYMTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: off,
                sh_size: size,
                sh_link: strtab_section_idx as u32,
                sh_info: first_global_idx,
                sh_addralign: ptr_align as u64,
                sh_entsize: sym_entry_size as u64,
            };
            self.write_shdr(&mut buf, is_64, &shdr);
        }

        // .strtab section header
        {
            let (off, size, _) = section_file_offsets[strtab_section_idx];
            let shdr = Elf64SectionHeader {
                sh_name: strtab_name_off,
                sh_type: SHT_STRTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: off,
                sh_size: size,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 1,
                sh_entsize: 0,
            };
            self.write_shdr(&mut buf, is_64, &shdr);
        }

        // .shstrtab section header
        {
            let (off, size, _) = section_file_offsets[shstrtab_section_idx];
            let shdr = Elf64SectionHeader {
                sh_name: shstrtab_name_off,
                sh_type: SHT_STRTAB,
                sh_flags: 0,
                sh_addr: 0,
                sh_offset: off,
                sh_size: size,
                sh_link: 0,
                sh_info: 0,
                sh_addralign: 1,
                sh_entsize: 0,
            };
            self.write_shdr(&mut buf, is_64, &shdr);
        }

        buf
    }

    /// Internal helper: write a section header entry, dispatching between
    /// 64-bit and 32-bit formats based on target.
    fn write_shdr(&self, buf: &mut Vec<u8>, is_64: bool, shdr: &Elf64SectionHeader) {
        if is_64 {
            buf.extend_from_slice(&shdr.to_bytes());
        } else {
            let shdr32 = Elf32SectionHeader {
                sh_name: shdr.sh_name,
                sh_type: shdr.sh_type,
                sh_flags: shdr.sh_flags as u32,
                sh_addr: shdr.sh_addr as u32,
                sh_offset: shdr.sh_offset as u32,
                sh_size: shdr.sh_size as u32,
                sh_link: shdr.sh_link,
                sh_info: shdr.sh_info,
                sh_addralign: shdr.sh_addralign as u32,
                sh_entsize: shdr.sh_entsize as u32,
            };
            buf.extend_from_slice(&shdr32.to_bytes());
        }
    }
}

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Align `value` up to the next multiple of `align`.
///
/// `align` must be a power of two (or 0/1 for no alignment).
/// Returns `value` unchanged if already aligned.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

/// Pad a byte vector with zeros until it reaches the target length.
///
/// Does nothing if the vector is already at or past `target_len`.
#[inline]
fn pad_to(buf: &mut Vec<u8>, target_len: usize) {
    if buf.len() < target_len {
        buf.resize(target_len, 0);
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ELF Constants -------------------------------------------------------

    #[test]
    fn elf_magic_bytes_correct() {
        assert_eq!(ELF_MAGIC, [0x7f, b'E', b'L', b'F']);
    }

    #[test]
    fn elf_class_constants() {
        assert_eq!(ELFCLASS32, 1);
        assert_eq!(ELFCLASS64, 2);
    }

    #[test]
    fn elf_data_constant() {
        assert_eq!(ELFDATA2LSB, 1);
    }

    #[test]
    fn elf_type_constants() {
        assert_eq!(ET_REL, 1);
        assert_eq!(ET_EXEC, 2);
        assert_eq!(ET_DYN, 3);
    }

    #[test]
    fn elf_machine_constants() {
        assert_eq!(EM_386, 3);
        assert_eq!(EM_X86_64, 62);
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(EM_RISCV, 243);
    }

    #[test]
    fn section_type_constants() {
        assert_eq!(SHT_NULL, 0);
        assert_eq!(SHT_PROGBITS, 1);
        assert_eq!(SHT_SYMTAB, 2);
        assert_eq!(SHT_STRTAB, 3);
        assert_eq!(SHT_RELA, 4);
        assert_eq!(SHT_HASH, 5);
        assert_eq!(SHT_DYNAMIC, 6);
        assert_eq!(SHT_NOTE, 7);
        assert_eq!(SHT_NOBITS, 8);
        assert_eq!(SHT_REL, 9);
        assert_eq!(SHT_DYNSYM, 11);
        assert_eq!(SHT_GNU_HASH, 0x6fff_fff6);
    }

    #[test]
    fn section_flag_constants() {
        assert_eq!(SHF_WRITE, 0x1);
        assert_eq!(SHF_ALLOC, 0x2);
        assert_eq!(SHF_EXECINSTR, 0x4);
        assert_eq!(SHF_MERGE, 0x10);
        assert_eq!(SHF_STRINGS, 0x20);
        assert_eq!(SHF_INFO_LINK, 0x40);
        assert_eq!(SHF_GROUP, 0x200);
    }

    #[test]
    fn program_header_type_constants() {
        assert_eq!(PT_NULL, 0);
        assert_eq!(PT_LOAD, 1);
        assert_eq!(PT_DYNAMIC, 2);
        assert_eq!(PT_INTERP, 3);
        assert_eq!(PT_NOTE, 4);
        assert_eq!(PT_PHDR, 6);
        assert_eq!(PT_GNU_STACK, 0x6474_e551);
        assert_eq!(PT_GNU_RELRO, 0x6474_e552);
    }

    #[test]
    fn symbol_binding_type_visibility() {
        assert_eq!(STB_LOCAL, 0);
        assert_eq!(STB_GLOBAL, 1);
        assert_eq!(STB_WEAK, 2);
        assert_eq!(STT_NOTYPE, 0);
        assert_eq!(STT_OBJECT, 1);
        assert_eq!(STT_FUNC, 2);
        assert_eq!(STT_SECTION, 3);
        assert_eq!(STT_FILE, 4);
        assert_eq!(STV_DEFAULT, 0);
        assert_eq!(STV_HIDDEN, 2);
        assert_eq!(STV_PROTECTED, 3);
    }

    // -- Header Serialization ------------------------------------------------

    #[test]
    fn elf64_header_size_and_magic() {
        let hdr = Elf64Header {
            e_type: ET_REL,
            e_machine: EM_X86_64,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 0,
            e_flags: 0,
            e_phnum: 0,
            e_shnum: 0,
            e_shstrndx: 0,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[0..4], &ELF_MAGIC);
        assert_eq!(bytes[4], ELFCLASS64);
        assert_eq!(bytes[5], ELFDATA2LSB);
        assert_eq!(bytes[6], 1); // EV_CURRENT
    }

    #[test]
    fn elf32_header_size_and_magic() {
        let hdr = Elf32Header {
            e_type: ET_REL,
            e_machine: EM_386,
            e_entry: 0,
            e_phoff: 0,
            e_shoff: 0,
            e_flags: 0,
            e_phnum: 0,
            e_shnum: 0,
            e_shstrndx: 0,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), 52);
        assert_eq!(&bytes[0..4], &ELF_MAGIC);
        assert_eq!(bytes[4], ELFCLASS32);
        assert_eq!(bytes[5], ELFDATA2LSB);
    }

    // -- Section Header Serialization ----------------------------------------

    #[test]
    fn elf64_section_header_size() {
        assert_eq!(Elf64SectionHeader::default().to_bytes().len(), 64);
    }

    #[test]
    fn elf32_section_header_size() {
        assert_eq!(Elf32SectionHeader::default().to_bytes().len(), 40);
    }

    // -- Program Header Serialization ----------------------------------------

    #[test]
    fn elf64_program_header_size() {
        assert_eq!(Elf64ProgramHeader::default().to_bytes().len(), 56);
    }

    #[test]
    fn elf32_program_header_size() {
        assert_eq!(Elf32ProgramHeader::default().to_bytes().len(), 32);
    }

    // -- Symbol Serialization ------------------------------------------------

    #[test]
    fn elf64_symbol_size() {
        assert_eq!(Elf64Symbol::default().to_bytes().len(), 24);
    }

    #[test]
    fn elf32_symbol_size() {
        assert_eq!(Elf32Symbol::default().to_bytes().len(), 16);
    }

    // -- Relocation Serialization --------------------------------------------

    #[test]
    fn elf64_rela_size_and_encoding() {
        let rela = Elf64Rela::new(0x100, 5, 10, -42);
        let bytes = rela.to_bytes();
        assert_eq!(bytes.len(), 24);
        assert_eq!(rela.r_info, (5u64 << 32) | 10u64);
        assert_eq!(rela.r_addend, -42);
    }

    #[test]
    fn elf32_rel_size_and_encoding() {
        let rel = Elf32Rel::new(0x200, 7, 3);
        let bytes = rel.to_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(rel.r_info, (7 << 8) | 3);
    }

    // -- StringTable ---------------------------------------------------------

    #[test]
    fn string_table_starts_with_null() {
        let strtab = StringTable::new();
        assert_eq!(strtab.as_bytes()[0], 0);
        assert_eq!(strtab.len(), 1);
    }

    #[test]
    fn string_table_add_and_retrieve() {
        let mut strtab = StringTable::new();
        let off = strtab.add_string("hello");
        assert_eq!(off, 1);
        assert_eq!(strtab.get_offset("hello"), Some(1));
    }

    #[test]
    fn string_table_deduplication() {
        let mut strtab = StringTable::new();
        let off1 = strtab.add_string("world");
        let off2 = strtab.add_string("world");
        assert_eq!(off1, off2);
    }

    #[test]
    fn string_table_empty_returns_zero() {
        let mut strtab = StringTable::new();
        assert_eq!(strtab.add_string(""), 0);
        assert_eq!(strtab.get_offset(""), Some(0));
    }

    #[test]
    fn string_table_null_terminated() {
        let mut strtab = StringTable::new();
        strtab.add_string("abc");
        assert_eq!(strtab.as_bytes(), &[0, b'a', b'b', b'c', 0]);
    }

    // -- SymbolTable ---------------------------------------------------------

    #[test]
    fn symbol_table_first_global_only_locals() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol(ElfSymbol {
            name: "l1".into(),
            binding: STB_LOCAL,
            ..ElfSymbol::default()
        });
        symtab.add_symbol(ElfSymbol {
            name: "l2".into(),
            binding: STB_LOCAL,
            ..ElfSymbol::default()
        });
        // 1 (null) + 2 locals = 3
        assert_eq!(symtab.first_global_index(), 3);
    }

    #[test]
    fn symbol_table_first_global_mixed() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol(ElfSymbol {
            name: "local".into(),
            binding: STB_LOCAL,
            ..ElfSymbol::default()
        });
        symtab.add_symbol(ElfSymbol {
            name: "global".into(),
            binding: STB_GLOBAL,
            ..ElfSymbol::default()
        });
        // 1 (null) + 1 local = 2
        assert_eq!(symtab.first_global_index(), 2);
    }

    #[test]
    fn symbol_table_build_bytes_ordering() {
        let mut symtab = SymbolTable::new();
        // Add global first, local second — output must have local first.
        symtab.add_symbol(ElfSymbol {
            name: "global_fn".into(),
            binding: STB_GLOBAL,
            sym_type: STT_FUNC,
            section_index: 1,
            ..ElfSymbol::default()
        });
        symtab.add_symbol(ElfSymbol {
            name: "local_var".into(),
            binding: STB_LOCAL,
            sym_type: STT_OBJECT,
            section_index: 2,
            ..ElfSymbol::default()
        });

        let (bytes, first_global) = symtab.build_bytes(true);
        // first_global = 1 (null) + 1 (local) = 2
        assert_eq!(first_global, 2);
        // 3 entries * 24 = 72 bytes
        assert_eq!(bytes.len(), 72);
    }

    // -- Relocation ----------------------------------------------------------

    #[test]
    fn relocation_to_elf64_rela() {
        let rel = Relocation {
            offset: 0x10,
            sym_index: 3,
            rel_type: 2,
            addend: -8,
        };
        let rela = rel.to_elf64_rela();
        assert_eq!(rela.r_offset, 0x10);
        assert_eq!(rela.r_info, (3u64 << 32) | 2);
        assert_eq!(rela.r_addend, -8);
    }

    #[test]
    fn relocation_to_elf32_rel() {
        let rel = Relocation {
            offset: 0x20,
            sym_index: 5,
            rel_type: 7,
            addend: 100,
        };
        let rel32 = rel.to_elf32_rel();
        assert_eq!(rel32.r_offset, 0x20);
        assert_eq!(rel32.r_info, (5 << 8) | 7);
    }

    // -- ElfWriter Integration -----------------------------------------------

    #[test]
    fn elf_writer_basic_x86_64() {
        let mut writer = ElfWriter::new(Target::X86_64, ET_REL);
        writer.add_section(Section {
            name: ".text".into(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0xcc],
            sh_addralign: 16,
            ..Section::default()
        });
        writer.add_symbol(ElfSymbol {
            name: "_start".into(),
            binding: STB_GLOBAL,
            sym_type: STT_FUNC,
            section_index: 1,
            ..ElfSymbol::default()
        });

        let elf = writer.write();
        assert_eq!(&elf[0..4], &ELF_MAGIC);
        assert_eq!(elf[4], ELFCLASS64);
        assert_eq!(elf[5], ELFDATA2LSB);
        assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), ET_REL);
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), EM_X86_64);
    }

    #[test]
    fn elf_writer_basic_i686() {
        let mut writer = ElfWriter::new(Target::I686, ET_REL);
        writer.add_section(Section {
            name: ".text".into(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_EXECINSTR,
            data: vec![0xcc],
            sh_addralign: 4,
            ..Section::default()
        });
        let elf = writer.write();
        assert_eq!(&elf[0..4], &ELF_MAGIC);
        assert_eq!(elf[4], ELFCLASS32);
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), EM_386);
    }

    #[test]
    fn elf_writer_aarch64() {
        let writer = ElfWriter::new(Target::AArch64, ET_REL);
        let elf = writer.write();
        assert_eq!(&elf[0..4], &ELF_MAGIC);
        assert_eq!(elf[4], ELFCLASS64);
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), EM_AARCH64);
    }

    #[test]
    fn elf_writer_riscv64_flags() {
        let writer = ElfWriter::new(Target::RiscV64, ET_REL);
        let elf = writer.write();
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), EM_RISCV);
        let e_flags = u32::from_le_bytes([elf[48], elf[49], elf[50], elf[51]]);
        assert_eq!(e_flags, 0x0001);
    }

    #[test]
    fn elf_writer_sections_count() {
        let mut writer = ElfWriter::new(Target::X86_64, ET_REL);
        assert_eq!(writer.sections_count(), 0);
        writer.add_section(Section {
            name: ".text".into(),
            sh_type: SHT_PROGBITS,
            ..Section::default()
        });
        assert_eq!(writer.sections_count(), 1);
    }

    #[test]
    fn elf_writer_target_accessor() {
        let writer = ElfWriter::new(Target::AArch64, ET_EXEC);
        assert_eq!(writer.target(), Target::AArch64);
    }

    #[test]
    fn elf_writer_entry_point() {
        let mut writer = ElfWriter::new(Target::X86_64, ET_EXEC);
        writer.set_entry_point(0x400000);
        let elf = writer.write();
        let entry = u64::from_le_bytes(elf[24..32].try_into().unwrap());
        assert_eq!(entry, 0x400000);
    }

    #[test]
    fn elf_writer_with_program_header() {
        let mut writer = ElfWriter::new(Target::X86_64, ET_EXEC);
        writer.add_program_header(Elf64ProgramHeader {
            p_type: PT_LOAD,
            p_flags: 5,
            p_offset: 0,
            p_vaddr: 0x400000,
            p_paddr: 0x400000,
            p_filesz: 0x1000,
            p_memsz: 0x1000,
            p_align: 0x1000,
        });
        let elf = writer.write();
        let phoff = u64::from_le_bytes(elf[32..40].try_into().unwrap());
        assert_eq!(phoff, 64);
        let phnum = u16::from_le_bytes([elf[56], elf[57]]);
        assert_eq!(phnum, 1);
    }

    #[test]
    fn elf_writer_bss_no_file_space() {
        let mut writer = ElfWriter::new(Target::X86_64, ET_REL);
        writer.add_section(Section {
            name: ".bss".into(),
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: vec![0; 4096], // logical size
            sh_addralign: 16,
            ..Section::default()
        });
        let elf = writer.write();
        // BSS should not inflate the file size proportionally
        assert!(elf.len() < 4096);
    }

    #[test]
    fn elf_writer_i686_entry_point_32bit() {
        let mut writer = ElfWriter::new(Target::I686, ET_EXEC);
        writer.set_entry_point(0x08048000);
        let elf = writer.write();
        let entry = u32::from_le_bytes(elf[24..28].try_into().unwrap());
        assert_eq!(entry, 0x08048000);
    }

    // -- align_up helper -----------------------------------------------------

    #[test]
    fn align_up_tests() {
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(17, 8), 24);
        assert_eq!(align_up(100, 0), 100);
        assert_eq!(align_up(100, 1), 100);
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
    }
}
