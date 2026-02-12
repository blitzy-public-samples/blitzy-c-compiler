//! Common ELF binary format writing infrastructure for the BCC compiler.
//!
//! This module provides the low-level ELF (Executable and Linkable Format)
//! structure creation API shared by all four architecture backends (x86-64,
//! i686, AArch64, RISC-V 64). It constructs ELF headers, section header
//! tables, program header tables, string tables (`.strtab`, `.shstrtab`),
//! symbol tables (`.symtab`), and note sections.
//!
//! # Architecture
//!
//! The [`ElfWriter`] struct is the main entry point. Callers configure the
//! writer with a target architecture, add sections, symbols, and program
//! headers, then call [`ElfWriter::write()`] to serialize the complete ELF
//! file to a byte vector.
//!
//! The writer supports all three ELF object types:
//!
//! - **`ET_REL`** — Relocatable object files (produced by `-c` flag)
//! - **`ET_EXEC`** — Static executables
//! - **`ET_DYN`** — Shared objects / position-independent executables
//!
//! # Zero-Dependency Implementation
//!
//! This module replaces external `object` or `elf` crates, adhering to the
//! project's strict zero-dependency mandate. All ELF constants, header
//! formats, and serialization logic are implemented from scratch using only
//! the Rust standard library.
//!
//! # Platform Restriction
//!
//! Output is exclusively Linux ELF. No Mach-O, PE/COFF, or other binary
//! formats are supported.

use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// ELF Constants — Hand-defined, no external crate
// ===========================================================================

// ---------------------------------------------------------------------------
// ELF Identification (e_ident)
// ---------------------------------------------------------------------------

/// ELF magic number bytes: `\x7fELF`.
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];

/// ELF class for 32-bit object files.
pub const ELFCLASS32: u8 = 1;

/// ELF class for 64-bit object files.
pub const ELFCLASS64: u8 = 2;

/// ELF data encoding: little-endian (2's complement, least significant byte
/// at lowest address). All four BCC targets use little-endian byte order.
pub const ELFDATA2LSB: u8 = 1;

/// Current ELF format version.
const EV_CURRENT: u8 = 1;

/// OS/ABI identification: UNIX System V (no extensions).
pub const ELFOSABI_NONE: u8 = 0;

// ---------------------------------------------------------------------------
// ELF Object File Types (e_type)
// ---------------------------------------------------------------------------

/// Relocatable object file — produced by the `-c` flag.
pub const ET_REL: u16 = 1;

/// Executable file — static ELF binary.
pub const ET_EXEC: u16 = 2;

/// Shared object file — dynamic library or PIE executable.
pub const ET_DYN: u16 = 3;

// ---------------------------------------------------------------------------
// ELF Machine Types (e_machine)
// ---------------------------------------------------------------------------

/// Intel 80386 (i686 / IA-32).
pub const EM_386: u16 = 3;

/// AMD x86-64 architecture.
pub const EM_X86_64: u16 = 62;

/// ARM AARCH64 (64-bit ARM).
pub const EM_AARCH64: u16 = 183;

/// RISC-V.
pub const EM_RISCV: u16 = 243;

// ---------------------------------------------------------------------------
// Section Header Types (sh_type)
// ---------------------------------------------------------------------------

/// Inactive/null section header entry.
pub const SHT_NULL: u32 = 0;

/// Program data — code, initialized data, etc.
pub const SHT_PROGBITS: u32 = 1;

/// Symbol table (`.symtab`).
pub const SHT_SYMTAB: u32 = 2;

/// String table (`.strtab`, `.shstrtab`, `.dynstr`).
pub const SHT_STRTAB: u32 = 3;

/// Relocation entries with explicit addends (`.rela.*`).
pub const SHT_RELA: u32 = 4;

/// Symbol hash table (`.hash`).
pub const SHT_HASH: u32 = 5;

/// Dynamic linking information (`.dynamic`).
pub const SHT_DYNAMIC: u32 = 6;

/// Auxiliary note information (`.note.*`).
pub const SHT_NOTE: u32 = 7;

/// Section occupies no space in the file (`.bss`).
pub const SHT_NOBITS: u32 = 8;

/// Relocation entries without explicit addends (`.rel.*`).
pub const SHT_REL: u32 = 9;

/// Dynamic linker symbol table (`.dynsym`).
pub const SHT_DYNSYM: u32 = 11;

// ---------------------------------------------------------------------------
// Section Header Flags (sh_flags)
// ---------------------------------------------------------------------------

/// Section is writable at runtime.
pub const SHF_WRITE: u64 = 0x1;

/// Section occupies memory during process execution.
pub const SHF_ALLOC: u64 = 0x2;

/// Section contains executable machine instructions.
pub const SHF_EXECINSTR: u64 = 0x4;

// ---------------------------------------------------------------------------
// Special Section Indices
// ---------------------------------------------------------------------------

/// Undefined/meaningless section reference.
pub const SHN_UNDEF: u16 = 0;

// ---------------------------------------------------------------------------
// Program Header Types (p_type)
// ---------------------------------------------------------------------------

/// Unused program header table entry.
pub const PT_NULL: u32 = 0;

/// Loadable segment.
pub const PT_LOAD: u32 = 1;

/// Dynamic linking information.
pub const PT_DYNAMIC: u32 = 2;

/// Path to program interpreter (dynamic linker).
pub const PT_INTERP: u32 = 3;

/// Auxiliary note information.
pub const PT_NOTE: u32 = 4;

/// Program header table itself (when loaded into memory).
pub const PT_PHDR: u32 = 6;

/// GNU extension: stack executability control.
pub const PT_GNU_STACK: u32 = 0x6474_e551;

/// GNU extension: read-only after relocation segment.
pub const PT_GNU_RELRO: u32 = 0x6474_e552;

// ---------------------------------------------------------------------------
// Program Header Flags (p_flags)
// ---------------------------------------------------------------------------

/// Segment is executable.
pub const PF_X: u32 = 0x1;

/// Segment is writable.
pub const PF_W: u32 = 0x2;

/// Segment is readable.
pub const PF_R: u32 = 0x4;

// ---------------------------------------------------------------------------
// Symbol Binding (upper 4 bits of st_info)
// ---------------------------------------------------------------------------

/// Local symbol — not visible outside the object file.
pub const STB_LOCAL: u8 = 0;

/// Global symbol — visible to all combined object files.
pub const STB_GLOBAL: u8 = 1;

/// Weak symbol — like global but may be overridden.
pub const STB_WEAK: u8 = 2;

// ---------------------------------------------------------------------------
// Symbol Types (lower 4 bits of st_info)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Symbol Visibility (lower 2 bits of st_other)
// ---------------------------------------------------------------------------

/// Default visibility — symbol may be overridden by another definition.
pub const STV_DEFAULT: u8 = 0;

/// Hidden visibility — symbol is not visible outside the shared object.
pub const STV_HIDDEN: u8 = 2;

/// Protected visibility — symbol is visible but cannot be preempted.
pub const STV_PROTECTED: u8 = 3;

// ===========================================================================
// ELF Header Size Constants
// ===========================================================================

/// Size of the 32-bit ELF header in bytes (Elf32_Ehdr).
const ELF32_EHDR_SIZE: usize = 52;

/// Size of the 64-bit ELF header in bytes (Elf64_Ehdr).
const ELF64_EHDR_SIZE: usize = 64;

/// Size of a 32-bit section header entry (Elf32_Shdr).
const ELF32_SHDR_SIZE: usize = 40;

/// Size of a 64-bit section header entry (Elf64_Shdr).
const ELF64_SHDR_SIZE: usize = 64;

/// Size of a 32-bit program header entry (Elf32_Phdr).
const ELF32_PHDR_SIZE: usize = 32;

/// Size of a 64-bit program header entry (Elf64_Phdr).
const ELF64_PHDR_SIZE: usize = 56;

/// Size of a 32-bit symbol table entry (Elf32_Sym).
const ELF32_SYM_SIZE: usize = 16;

/// Size of a 64-bit symbol table entry (Elf64_Sym).
const ELF64_SYM_SIZE: usize = 24;

// ===========================================================================
// StringTable — ELF string table with deduplication
// ===========================================================================

/// An ELF string table with O(1) deduplication via [`FxHashMap`].
///
/// ELF string tables (`.strtab`, `.shstrtab`, `.dynstr`) are contiguous byte
/// arrays of null-terminated strings. Each string is identified by its byte
/// offset from the table start. The first byte is always a null byte (the
/// empty string at offset 0, representing `SHN_UNDEF` or unnamed entries).
///
/// This implementation deduplicates strings — if the same string is added
/// twice, the same offset is returned both times, saving space in the final
/// ELF binary.
pub struct StringTable {
    /// Raw byte data of the string table (null-terminated strings concatenated).
    data: Vec<u8>,
    /// Map from string content to its byte offset within `data`, used for
    /// deduplication and O(1) lookup of previously added strings.
    offsets: FxHashMap<String, u32>,
}

impl StringTable {
    /// Creates a new, empty string table.
    ///
    /// The table is initialized with a single null byte at offset 0,
    /// which is required by the ELF specification for the undefined/empty
    /// string entry.
    pub fn new() -> Self {
        StringTable {
            data: vec![0u8], // null byte at offset 0
            offsets: FxHashMap::default(),
        }
    }

    /// Adds a string to the table and returns its byte offset.
    ///
    /// If the string has already been added, the existing offset is returned
    /// without duplicating the data. Empty strings always return offset 0
    /// (the null byte at the table start).
    ///
    /// # Arguments
    ///
    /// * `s` — The string to add. Will be stored as a null-terminated byte
    ///   sequence in the table.
    ///
    /// # Returns
    ///
    /// The byte offset of the string within the table, suitable for use in
    /// ELF header fields that reference string table entries (e.g.,
    /// `sh_name`, `st_name`).
    pub fn add_string(&mut self, s: &str) -> u32 {
        // Empty string maps to offset 0 (the initial null byte).
        if s.is_empty() {
            return 0;
        }

        // Check for existing entry using the FxHashMap for O(1) deduplication.
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }

        // Record offset before appending — this is where the string starts.
        let offset = self.data.len() as u32;

        // Append the string bytes followed by a null terminator.
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0u8);

        // Store in the deduplication map for future lookups.
        self.offsets.insert(s.to_string(), offset);

        offset
    }

    /// Looks up the byte offset of a previously added string.
    ///
    /// Returns `None` if the string has not been added to this table.
    /// Empty strings always return `Some(0)`.
    pub fn get_offset(&self, s: &str) -> Option<u32> {
        if s.is_empty() {
            return Some(0);
        }
        self.offsets.get(s).copied()
    }

    /// Returns the raw byte data of the string table.
    ///
    /// This is the complete, ready-to-emit byte sequence for the ELF section
    /// data, including the leading null byte and all null terminators.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns the current size of the string table in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the string table contains only the initial null byte.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.len() <= 1
    }

    /// Returns `true` if the given string has already been added to the table.
    ///
    /// Empty strings always return `true` (the empty string is implicitly
    /// present at offset 0).
    #[inline]
    pub fn contains(&self, s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        self.offsets.contains_key(s)
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// ElfSection — Section descriptor
// ===========================================================================

/// Describes an ELF section to be written into the output file.
///
/// Each section carries its metadata (type, flags, alignment) and its raw
/// byte data. The [`ElfWriter`] collects sections and serializes them into
/// the final ELF layout including proper section header table entries.
///
/// # Fields
///
/// All fields are public to allow direct construction by architecture-specific
/// backends that need fine-grained control over section attributes.
#[derive(Clone)]
pub struct ElfSection {
    /// Section name (e.g., `.text`, `.data`, `.bss`, `.rodata`).
    /// Stored in the section header string table (`.shstrtab`).
    pub name: String,

    /// Section type — one of the `SHT_*` constants (e.g., [`SHT_PROGBITS`],
    /// [`SHT_SYMTAB`], [`SHT_NOBITS`]).
    pub section_type: u32,

    /// Section attribute flags — a bitwise OR of `SHF_*` constants
    /// (e.g., [`SHF_ALLOC`] | [`SHF_EXECINSTR`] for executable code).
    pub flags: u64,

    /// Raw section data. Empty for `SHT_NOBITS` sections (`.bss`).
    pub data: Vec<u8>,

    /// Required alignment for the section's data (must be a power of two,
    /// or zero for no alignment constraint).
    pub alignment: u64,

    /// Size of each entry for sections with fixed-size entries (e.g.,
    /// symbol tables, relocation tables). Zero for sections without
    /// fixed-size entries.
    pub entry_size: u64,

    /// Section header table index link. Interpretation depends on section type:
    /// - `SHT_SYMTAB`/`SHT_DYNSYM`: index of associated string table section
    /// - `SHT_RELA`/`SHT_REL`: index of the section to which relocations apply
    /// - `SHT_HASH`: index of the symbol table section
    /// - `SHT_DYNAMIC`: index of the string table used by entries
    pub link: u32,

    /// Extra information. Interpretation depends on section type:
    /// - `SHT_SYMTAB`/`SHT_DYNSYM`: one greater than the index of the last
    ///   local symbol (`STB_LOCAL`)
    /// - `SHT_RELA`/`SHT_REL`: index of the section to which relocations apply
    pub info: u32,

    /// Virtual address at which the section should be loaded (for allocated
    /// sections). Zero for non-loaded sections.
    pub addr: u64,
}

impl ElfSection {
    /// Creates a new section with the given name and type, using sensible
    /// defaults for all other fields (zero flags, empty data, alignment 1).
    pub fn new(name: &str, section_type: u32) -> Self {
        ElfSection {
            name: name.to_string(),
            section_type,
            flags: 0,
            data: Vec::new(),
            alignment: 1,
            entry_size: 0,
            link: 0,
            info: 0,
            addr: 0,
        }
    }
}

// ===========================================================================
// ElfSymbol — Symbol table entry
// ===========================================================================

/// Describes an ELF symbol to be written into the `.symtab` section.
///
/// Symbols represent named entities (functions, variables, sections, files)
/// that are defined or referenced within the ELF object. The symbol table
/// is used by the linker for relocation, by the dynamic linker for runtime
/// binding, and by debuggers for name resolution.
#[derive(Clone)]
pub struct ElfSymbol {
    /// Symbol name. Stored in the string table (`.strtab`) referenced by the
    /// symbol table section's `sh_link` field. Empty string for unnamed symbols.
    pub name: String,

    /// Symbol value — depends on symbol type:
    /// - For defined functions/objects: virtual address or section offset
    /// - For common symbols: alignment constraint
    /// - For undefined symbols: zero
    pub value: u64,

    /// Size of the object associated with the symbol (in bytes).
    /// Zero if the size is unknown or not applicable.
    pub size: u64,

    /// Symbol binding — one of [`STB_LOCAL`], [`STB_GLOBAL`], [`STB_WEAK`].
    /// Stored in the upper 4 bits of `st_info`.
    pub binding: u8,

    /// Symbol type — one of [`STT_NOTYPE`], [`STT_OBJECT`], [`STT_FUNC`],
    /// [`STT_SECTION`], [`STT_FILE`]. Stored in the lower 4 bits of `st_info`.
    pub sym_type: u8,

    /// Symbol visibility — one of [`STV_DEFAULT`], [`STV_HIDDEN`],
    /// [`STV_PROTECTED`]. Stored in the lower 2 bits of `st_other`.
    pub visibility: u8,

    /// Section table index of the section in which the symbol is defined.
    /// [`SHN_UNDEF`] (0) for undefined symbols, `SHN_ABS` (0xfff1) for
    /// absolute symbols, or a valid section index.
    pub section_index: u16,
}

impl ElfSymbol {
    /// Creates a new symbol with sensible defaults (undefined, no type,
    /// local binding, default visibility).
    pub fn new(name: &str) -> Self {
        ElfSymbol {
            name: name.to_string(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: SHN_UNDEF,
        }
    }

    /// Computes the `st_info` byte from binding and type fields.
    ///
    /// `st_info = (binding << 4) | (type & 0xf)`
    #[inline]
    fn st_info(&self) -> u8 {
        (self.binding << 4) | (self.sym_type & 0xf)
    }

    /// Computes the `st_other` byte from the visibility field.
    ///
    /// `st_other = visibility & 0x3`
    #[inline]
    fn st_other(&self) -> u8 {
        self.visibility & 0x3
    }
}

// ===========================================================================
// ProgramHeader — Program header table entry
// ===========================================================================

/// Describes an ELF program header (segment descriptor).
///
/// Program headers define how the ELF loader maps the file into memory.
/// Each header describes a segment with its type, permissions, file offset,
/// virtual address, and size. The loader uses this information to create
/// the process image.
#[derive(Clone)]
pub struct ProgramHeader {
    /// Segment type — one of the `PT_*` constants (e.g., [`PT_LOAD`],
    /// [`PT_DYNAMIC`], [`PT_INTERP`]).
    pub p_type: u32,

    /// Segment permission flags — a bitwise OR of `PF_*` constants
    /// (e.g., [`PF_R`] | [`PF_X`] for readable, executable code).
    pub p_flags: u32,

    /// Offset from the beginning of the file to the first byte of the segment.
    pub p_offset: u64,

    /// Virtual address at which the segment is loaded into memory.
    pub p_vaddr: u64,

    /// Physical address (relevant for systems where physical addressing is used).
    /// Typically set equal to `p_vaddr` on Linux.
    pub p_paddr: u64,

    /// Size of the segment in the file (in bytes). May be zero for `.bss`-only
    /// segments.
    pub p_filesz: u64,

    /// Size of the segment in memory (in bytes). If `p_memsz > p_filesz`, the
    /// extra bytes are zero-filled (used for `.bss` sections).
    pub p_memsz: u64,

    /// Alignment of the segment in memory and in the file. Must be a power of
    /// two. `p_vaddr ≡ p_offset (mod p_align)`.
    pub p_align: u64,
}

impl ProgramHeader {
    /// Creates a new program header with sensible defaults (PT_NULL, no flags,
    /// all offsets and sizes zero, alignment 1).
    pub fn new(p_type: u32) -> Self {
        ProgramHeader {
            p_type,
            p_flags: 0,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 1,
        }
    }
}

// ===========================================================================
// ElfWriter — Main ELF file construction API
// ===========================================================================

/// ELF file writer that serializes sections, symbols, and program headers
/// into a complete ELF binary.
///
/// # Usage
///
/// ```ignore
/// use bcc::backend::elf_writer_common::*;
/// use bcc::common::target::Target;
///
/// let mut writer = ElfWriter::new(Target::X86_64);
/// writer.set_type(ET_EXEC);
/// writer.set_entry_point(0x401000);
///
/// let mut text = ElfSection::new(".text", SHT_PROGBITS);
/// text.flags = SHF_ALLOC | SHF_EXECINSTR;
/// text.data = vec![0xc3]; // ret instruction
/// text.alignment = 16;
/// writer.add_section(text);
///
/// let bytes = writer.write();
/// // `bytes` is now a valid ELF executable
/// ```
///
/// # Layout
///
/// The [`write()`](ElfWriter::write) method produces the following ELF layout:
///
/// 1. ELF header (52 bytes for 32-bit, 64 bytes for 64-bit)
/// 2. Program header table (if program headers were added)
/// 3. Section data (user sections, then auto-generated `.symtab`, `.strtab`)
/// 4. Section header string table (`.shstrtab`) data
/// 5. Section header table
pub struct ElfWriter {
    /// Target architecture — determines ELF class (32/64), machine type,
    /// flags, and header sizes.
    target: Target,

    /// ELF object file type — [`ET_REL`], [`ET_EXEC`], or [`ET_DYN`].
    elf_type: u16,

    /// Entry point virtual address (`e_entry`). Zero for relocatable objects.
    entry_point: u64,

    /// User-added sections (does not include auto-generated `.symtab`,
    /// `.strtab`, `.shstrtab` which are created during [`write()`]).
    sections: Vec<ElfSection>,

    /// Symbol table entries. Local symbols are sorted before global symbols
    /// during serialization as required by the ELF specification.
    symbols: Vec<ElfSymbol>,

    /// String table for symbol names (`.strtab`). Built incrementally as
    /// symbols are added, then serialized during [`write()`].
    string_table: StringTable,

    /// String table for section names (`.shstrtab`). Built during [`write()`]
    /// from the names of all sections.
    section_string_table: StringTable,

    /// Program header table entries. Only present for `ET_EXEC` and `ET_DYN`
    /// output; empty for `ET_REL`.
    program_headers: Vec<ProgramHeader>,
}

impl ElfWriter {
    /// Creates a new ELF writer configured for the given target architecture.
    ///
    /// The writer is initialized with:
    /// - ELF type set to [`ET_REL`] (relocatable object) by default
    /// - Entry point at 0
    /// - Empty section, symbol, and program header lists
    /// - Pre-initialized string tables with null byte at offset 0
    pub fn new(target: Target) -> Self {
        ElfWriter {
            target,
            elf_type: ET_REL,
            entry_point: 0,
            sections: Vec::new(),
            symbols: Vec::new(),
            string_table: StringTable::new(),
            section_string_table: StringTable::new(),
            program_headers: Vec::new(),
        }
    }

    /// Returns the pointer size (in bytes) for the configured target architecture.
    ///
    /// This value is derived from the target's `pointer_width()` method and
    /// reflects the native address size: **4** for 32-bit targets (i686),
    /// **8** for 64-bit targets (x86-64, AArch64, RISC-V 64).
    ///
    /// Callers (such as architecture-specific linkers) may use this to
    /// determine relocation sizes, GOT entry widths, and PLT stub layouts.
    #[inline]
    pub fn pointer_size(&self) -> u8 {
        self.target.pointer_width() as u8
    }

    /// Sets the ELF object file type.
    ///
    /// # Arguments
    ///
    /// * `elf_type` — One of [`ET_REL`] (relocatable), [`ET_EXEC`]
    ///   (executable), or [`ET_DYN`] (shared object).
    pub fn set_type(&mut self, elf_type: u16) {
        self.elf_type = elf_type;
    }

    /// Sets the program entry point virtual address (`e_entry`).
    ///
    /// This is the address where the system transfers control when executing
    /// the program. Should be zero for relocatable objects (`ET_REL`).
    pub fn set_entry_point(&mut self, addr: u64) {
        self.entry_point = addr;
    }

    /// Adds a section to the ELF file and returns its 1-based section index.
    ///
    /// Section index 0 is always the null section (`SHN_UNDEF`), which is
    /// automatically created during serialization. The returned index accounts
    /// for this, so the first user section has index 1.
    ///
    /// # Returns
    ///
    /// The section index that can be used in symbol `section_index` fields,
    /// relocation section `sh_info` fields, etc.
    pub fn add_section(&mut self, section: ElfSection) -> usize {
        // Pre-register the section name in the section header string table
        // for deduplication during serialization.
        if !section.name.is_empty() {
            self.section_string_table.add_string(&section.name);
        }
        self.sections.push(section);
        // Section index = position + 1 (accounting for the null section at index 0)
        self.sections.len()
    }

    /// Adds a symbol to the symbol table.
    ///
    /// Symbols are collected and sorted during [`write()`] — local symbols
    /// (`STB_LOCAL`) are placed before global/weak symbols as required by the
    /// ELF specification. The symbol's name is automatically added to the
    /// string table (`.strtab`).
    pub fn add_symbol(&mut self, symbol: ElfSymbol) {
        // Pre-register the symbol name in the string table for deduplication.
        if !symbol.name.is_empty() {
            self.string_table.add_string(&symbol.name);
        }
        self.symbols.push(symbol);
    }

    /// Adds a program header (segment descriptor) to the ELF file.
    ///
    /// Program headers are only meaningful for `ET_EXEC` and `ET_DYN` files.
    /// They describe how the ELF loader maps the file into memory.
    pub fn add_program_header(&mut self, phdr: ProgramHeader) {
        self.program_headers.push(phdr);
    }

    // -----------------------------------------------------------------------
    // Core serialization
    // -----------------------------------------------------------------------

    /// Serializes the complete ELF file to a byte vector.
    ///
    /// This method performs the following steps:
    ///
    /// 1. Determines 32-bit vs 64-bit format from the target architecture
    /// 2. Sorts symbols (locals before globals, as required by ELF spec)
    /// 3. Builds the section header string table (`.shstrtab`)
    /// 4. Computes layout offsets for all components
    /// 5. Writes the ELF header
    /// 6. Writes program headers (if any)
    /// 7. Writes section data
    /// 8. Writes the section header table
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the complete, valid ELF binary ready to be
    /// written to disk.
    pub fn write(&self) -> Vec<u8> {
        let is_64bit = self.target.elf_class() == ELFCLASS64;

        // ---- Step 1: Sort symbols (locals before globals) -----------------
        let (sorted_symbols, first_global_index) = self.sort_symbols();

        // ---- Step 2: Build string tables ----------------------------------
        // Symbol string table (.strtab): collect all symbol names.
        let mut strtab = StringTable::new();
        for sym in &sorted_symbols {
            if !sym.name.is_empty() {
                strtab.add_string(&sym.name);
            }
        }

        // Section header string table (.shstrtab): collect all section names
        // including the auto-generated sections.
        let mut shstrtab = StringTable::new();
        for section in &self.sections {
            shstrtab.add_string(&section.name);
        }
        // Names for auto-generated sections.
        shstrtab.add_string(".symtab");
        shstrtab.add_string(".strtab");
        shstrtab.add_string(".shstrtab");

        // ---- Step 3: Build .symtab section data ---------------------------
        let sym_entry_size = if is_64bit { ELF64_SYM_SIZE } else { ELF32_SYM_SIZE };
        let symtab_data = self.serialize_symbols(&sorted_symbols, &strtab, is_64bit);

        // ---- Step 4: Compute layout offsets -------------------------------
        let ehdr_size = if is_64bit { ELF64_EHDR_SIZE } else { ELF32_EHDR_SIZE };
        let phdr_entry_size = if is_64bit { ELF64_PHDR_SIZE } else { ELF32_PHDR_SIZE };
        let shdr_entry_size = if is_64bit { ELF64_SHDR_SIZE } else { ELF32_SHDR_SIZE };

        // Program header table immediately follows the ELF header.
        let phdr_offset = if self.program_headers.is_empty() {
            0usize
        } else {
            ehdr_size
        };
        let phdr_total_size = self.program_headers.len() * phdr_entry_size;

        // Section data starts after the ELF header and program headers.
        let mut current_offset = ehdr_size + phdr_total_size;

        // Compute file offsets for each user section.
        let mut section_offsets: Vec<usize> = Vec::with_capacity(self.sections.len());
        for section in &self.sections {
            // Align to the section's required alignment.
            if section.alignment > 1 {
                current_offset = align_up(current_offset, section.alignment as usize);
            }
            section_offsets.push(current_offset);
            // SHT_NOBITS sections occupy no file space.
            if section.section_type != SHT_NOBITS {
                current_offset += section.data.len();
            }
        }

        // Total number of sections in the section header table:
        // null(0) + user sections + .symtab + .strtab + .shstrtab
        let num_user_sections = self.sections.len();
        let symtab_section_idx = num_user_sections + 1; // +1 for null section
        let strtab_section_idx = symtab_section_idx + 1;
        let shstrtab_section_idx = strtab_section_idx + 1;
        let total_sections = shstrtab_section_idx + 1; // includes null section

        // .symtab data offset
        if sym_entry_size > 1 {
            current_offset = align_up(current_offset, if is_64bit { 8 } else { 4 });
        }
        let symtab_offset = current_offset;
        current_offset += symtab_data.len();

        // .strtab data offset
        let strtab_offset = current_offset;
        current_offset += strtab.as_bytes().len();

        // .shstrtab data offset
        let shstrtab_offset = current_offset;
        current_offset += shstrtab.as_bytes().len();

        // Section header table offset — align to natural word boundary.
        let shdr_align = if is_64bit { 8 } else { 4 };
        current_offset = align_up(current_offset, shdr_align);
        let shdr_offset = current_offset;

        // ---- Step 5: Build the output buffer ------------------------------
        let total_size = shdr_offset + total_sections * shdr_entry_size;
        let mut output = Vec::with_capacity(total_size);

        // ---- Step 6: Write ELF header -------------------------------------
        self.write_elf_header(
            &mut output,
            is_64bit,
            phdr_offset,
            shdr_offset,
            phdr_entry_size,
            shdr_entry_size,
            total_sections,
            shstrtab_section_idx,
        );

        // ---- Step 7: Write program headers --------------------------------
        for phdr in &self.program_headers {
            self.write_program_header(&mut output, phdr, is_64bit);
        }

        // ---- Step 8: Write section data -----------------------------------
        for (i, section) in self.sections.iter().enumerate() {
            // Pad to the computed section offset.
            pad_to(&mut output, section_offsets[i]);
            // Write section data (skip for NOBITS).
            if section.section_type != SHT_NOBITS {
                output.extend_from_slice(&section.data);
            }
        }

        // .symtab data
        pad_to(&mut output, symtab_offset);
        output.extend_from_slice(&symtab_data);

        // .strtab data
        pad_to(&mut output, strtab_offset);
        output.extend_from_slice(strtab.as_bytes());

        // .shstrtab data
        pad_to(&mut output, shstrtab_offset);
        output.extend_from_slice(shstrtab.as_bytes());

        // ---- Step 9: Write section header table ---------------------------
        pad_to(&mut output, shdr_offset);

        // Section 0: null entry (SHN_UNDEF)
        self.write_section_header(
            &mut output,
            is_64bit,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );

        // User sections
        for (i, section) in self.sections.iter().enumerate() {
            let sh_name = shstrtab.get_offset(&section.name).unwrap_or(0);
            // For SHT_NOBITS, sh_offset is still set to the computed offset
            // (indicating the conceptual position) and sh_size reflects the
            // virtual allocation size stored in the data vector's length.
            let sh_offset = section_offsets[i] as u64;
            let sh_size = section.data.len() as u64;
            self.write_section_header(
                &mut output,
                is_64bit,
                sh_name,
                section.section_type,
                section.flags,
                section.addr,
                sh_offset,
                sh_size,
                section.link,
                section.info,
                section.alignment,
                section.entry_size,
            );
        }

        // .symtab section header
        let symtab_name = shstrtab.get_offset(".symtab").unwrap_or(0);
        self.write_section_header(
            &mut output,
            is_64bit,
            symtab_name,
            SHT_SYMTAB,
            0, // no flags
            0, // no address
            symtab_offset as u64,
            symtab_data.len() as u64,
            strtab_section_idx as u32, // sh_link → .strtab section index
            first_global_index as u32, // sh_info → index of first non-local symbol
            if is_64bit { 8 } else { 4 }, // alignment
            sym_entry_size as u64,
        );

        // .strtab section header
        let strtab_name = shstrtab.get_offset(".strtab").unwrap_or(0);
        self.write_section_header(
            &mut output,
            is_64bit,
            strtab_name,
            SHT_STRTAB,
            0,
            0,
            strtab_offset as u64,
            strtab.as_bytes().len() as u64,
            0,
            0,
            1,
            0,
        );

        // .shstrtab section header
        let shstrtab_name = shstrtab.get_offset(".shstrtab").unwrap_or(0);
        self.write_section_header(
            &mut output,
            is_64bit,
            shstrtab_name,
            SHT_STRTAB,
            0,
            0,
            shstrtab_offset as u64,
            shstrtab.as_bytes().len() as u64,
            0,
            0,
            1,
            0,
        );

        output
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Sorts symbols with local symbols before global/weak symbols.
    ///
    /// The ELF specification requires that in `.symtab`, all `STB_LOCAL`
    /// symbols precede all `STB_GLOBAL` and `STB_WEAK` symbols. The section
    /// header's `sh_info` field records the index of the first non-local
    /// symbol.
    ///
    /// Returns the sorted symbol list and the index of the first global symbol.
    /// The first entry in the sorted list is always the null symbol (index 0).
    fn sort_symbols(&self) -> (Vec<ElfSymbol>, usize) {
        let mut locals: Vec<ElfSymbol> = Vec::new();
        let mut globals: Vec<ElfSymbol> = Vec::new();

        for sym in &self.symbols {
            if sym.binding == STB_LOCAL {
                locals.push(sym.clone());
            } else {
                globals.push(sym.clone());
            }
        }

        // Build sorted list: null symbol + locals + globals.
        let mut sorted = Vec::with_capacity(1 + locals.len() + globals.len());

        // Entry 0: null/undefined symbol (required by ELF spec).
        sorted.push(ElfSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: SHN_UNDEF,
        });

        // Index of first non-local (global) symbol = 1 + locals.len()
        let first_global = 1 + locals.len();

        sorted.extend(locals);
        sorted.extend(globals);

        (sorted, first_global)
    }

    /// Serializes the symbol table to bytes.
    ///
    /// Produces either Elf32_Sym or Elf64_Sym entries depending on the
    /// `is_64bit` flag.
    fn serialize_symbols(
        &self,
        symbols: &[ElfSymbol],
        strtab: &StringTable,
        is_64bit: bool,
    ) -> Vec<u8> {
        let entry_size = if is_64bit { ELF64_SYM_SIZE } else { ELF32_SYM_SIZE };
        let mut data = Vec::with_capacity(symbols.len() * entry_size);

        for sym in symbols {
            let st_name = if sym.name.is_empty() {
                0u32
            } else {
                strtab.get_offset(&sym.name).unwrap_or(0)
            };

            if is_64bit {
                // Elf64_Sym layout:
                //   st_name  (4 bytes) — offset into string table
                //   st_info  (1 byte)  — binding + type
                //   st_other (1 byte)  — visibility
                //   st_shndx (2 bytes) — section index
                //   st_value (8 bytes) — symbol value
                //   st_size  (8 bytes) — symbol size
                data.extend_from_slice(&st_name.to_le_bytes());
                data.push(sym.st_info());
                data.push(sym.st_other());
                data.extend_from_slice(&sym.section_index.to_le_bytes());
                data.extend_from_slice(&sym.value.to_le_bytes());
                data.extend_from_slice(&sym.size.to_le_bytes());
            } else {
                // Elf32_Sym layout:
                //   st_name  (4 bytes) — offset into string table
                //   st_value (4 bytes) — symbol value
                //   st_size  (4 bytes) — symbol size
                //   st_info  (1 byte)  — binding + type
                //   st_other (1 byte)  — visibility
                //   st_shndx (2 bytes) — section index
                data.extend_from_slice(&st_name.to_le_bytes());
                data.extend_from_slice(&(sym.value as u32).to_le_bytes());
                data.extend_from_slice(&(sym.size as u32).to_le_bytes());
                data.push(sym.st_info());
                data.push(sym.st_other());
                data.extend_from_slice(&sym.section_index.to_le_bytes());
            }
        }

        data
    }

    /// Writes the ELF file header to the output buffer.
    ///
    /// Produces either a 52-byte Elf32_Ehdr or a 64-byte Elf64_Ehdr depending
    /// on the `is_64bit` flag.
    #[allow(clippy::too_many_arguments)]
    fn write_elf_header(
        &self,
        output: &mut Vec<u8>,
        is_64bit: bool,
        phdr_offset: usize,
        shdr_offset: usize,
        phdr_entry_size: usize,
        shdr_entry_size: usize,
        num_sections: usize,
        shstrndx: usize,
    ) {
        // e_ident[0..4]: ELF magic
        output.extend_from_slice(&ELF_MAGIC);

        // e_ident[4]: EI_CLASS
        output.push(self.target.elf_class());

        // e_ident[5]: EI_DATA — little-endian for all BCC targets
        output.push(ELFDATA2LSB);

        // e_ident[6]: EI_VERSION — always EV_CURRENT (1)
        output.push(EV_CURRENT);

        // e_ident[7]: EI_OSABI — ELFOSABI_NONE (UNIX System V)
        output.push(ELFOSABI_NONE);

        // e_ident[8..16]: EI_ABIVERSION + padding (8 zero bytes)
        output.extend_from_slice(&[0u8; 8]);

        // e_type (2 bytes)
        output.extend_from_slice(&self.elf_type.to_le_bytes());

        // e_machine (2 bytes)
        output.extend_from_slice(&self.target.elf_machine().to_le_bytes());

        // e_version (4 bytes)
        output.extend_from_slice(&(EV_CURRENT as u32).to_le_bytes());

        if is_64bit {
            // --- Elf64_Ehdr specific fields ---
            // e_entry (8 bytes)
            output.extend_from_slice(&self.entry_point.to_le_bytes());

            // e_phoff (8 bytes) — program header table offset
            output.extend_from_slice(&(phdr_offset as u64).to_le_bytes());

            // e_shoff (8 bytes) — section header table offset
            output.extend_from_slice(&(shdr_offset as u64).to_le_bytes());
        } else {
            // --- Elf32_Ehdr specific fields ---
            // e_entry (4 bytes)
            output.extend_from_slice(&(self.entry_point as u32).to_le_bytes());

            // e_phoff (4 bytes)
            output.extend_from_slice(&(phdr_offset as u32).to_le_bytes());

            // e_shoff (4 bytes)
            output.extend_from_slice(&(shdr_offset as u32).to_le_bytes());
        }

        // e_flags (4 bytes) — architecture-specific flags
        output.extend_from_slice(&self.target.elf_flags().to_le_bytes());

        // e_ehsize (2 bytes) — ELF header size
        let ehdr_size: u16 = if is_64bit {
            ELF64_EHDR_SIZE as u16
        } else {
            ELF32_EHDR_SIZE as u16
        };
        output.extend_from_slice(&ehdr_size.to_le_bytes());

        // e_phentsize (2 bytes) — program header entry size
        let phdr_ent_size: u16 = if self.program_headers.is_empty() {
            0
        } else {
            phdr_entry_size as u16
        };
        output.extend_from_slice(&phdr_ent_size.to_le_bytes());

        // e_phnum (2 bytes) — number of program header entries
        output.extend_from_slice(&(self.program_headers.len() as u16).to_le_bytes());

        // e_shentsize (2 bytes) — section header entry size
        output.extend_from_slice(&(shdr_entry_size as u16).to_le_bytes());

        // e_shnum (2 bytes) — number of section header entries
        output.extend_from_slice(&(num_sections as u16).to_le_bytes());

        // e_shstrndx (2 bytes) — section header string table index
        output.extend_from_slice(&(shstrndx as u16).to_le_bytes());
    }

    /// Writes a single program header entry to the output buffer.
    fn write_program_header(
        &self,
        output: &mut Vec<u8>,
        phdr: &ProgramHeader,
        is_64bit: bool,
    ) {
        if is_64bit {
            // Elf64_Phdr layout:
            //   p_type   (4 bytes)
            //   p_flags  (4 bytes)  ← note: flags come BEFORE offset in 64-bit!
            //   p_offset (8 bytes)
            //   p_vaddr  (8 bytes)
            //   p_paddr  (8 bytes)
            //   p_filesz (8 bytes)
            //   p_memsz  (8 bytes)
            //   p_align  (8 bytes)
            output.extend_from_slice(&phdr.p_type.to_le_bytes());
            output.extend_from_slice(&phdr.p_flags.to_le_bytes());
            output.extend_from_slice(&phdr.p_offset.to_le_bytes());
            output.extend_from_slice(&phdr.p_vaddr.to_le_bytes());
            output.extend_from_slice(&phdr.p_paddr.to_le_bytes());
            output.extend_from_slice(&phdr.p_filesz.to_le_bytes());
            output.extend_from_slice(&phdr.p_memsz.to_le_bytes());
            output.extend_from_slice(&phdr.p_align.to_le_bytes());
        } else {
            // Elf32_Phdr layout:
            //   p_type   (4 bytes)
            //   p_offset (4 bytes)  ← offset comes BEFORE flags in 32-bit!
            //   p_vaddr  (4 bytes)
            //   p_paddr  (4 bytes)
            //   p_filesz (4 bytes)
            //   p_memsz  (4 bytes)
            //   p_flags  (4 bytes)
            //   p_align  (4 bytes)
            output.extend_from_slice(&phdr.p_type.to_le_bytes());
            output.extend_from_slice(&(phdr.p_offset as u32).to_le_bytes());
            output.extend_from_slice(&(phdr.p_vaddr as u32).to_le_bytes());
            output.extend_from_slice(&(phdr.p_paddr as u32).to_le_bytes());
            output.extend_from_slice(&(phdr.p_filesz as u32).to_le_bytes());
            output.extend_from_slice(&(phdr.p_memsz as u32).to_le_bytes());
            output.extend_from_slice(&phdr.p_flags.to_le_bytes());
            output.extend_from_slice(&(phdr.p_align as u32).to_le_bytes());
        }
    }

    /// Writes a single section header entry to the output buffer.
    ///
    /// Produces either a 40-byte Elf32_Shdr or a 64-byte Elf64_Shdr.
    #[allow(clippy::too_many_arguments)]
    fn write_section_header(
        &self,
        output: &mut Vec<u8>,
        is_64bit: bool,
        sh_name: u32,
        sh_type: u32,
        sh_flags: u64,
        sh_addr: u64,
        sh_offset: u64,
        sh_size: u64,
        sh_link: u32,
        sh_info: u32,
        sh_addralign: u64,
        sh_entsize: u64,
    ) {
        if is_64bit {
            // Elf64_Shdr layout (64 bytes total):
            output.extend_from_slice(&sh_name.to_le_bytes());     // 4 bytes
            output.extend_from_slice(&sh_type.to_le_bytes());     // 4 bytes
            output.extend_from_slice(&sh_flags.to_le_bytes());    // 8 bytes
            output.extend_from_slice(&sh_addr.to_le_bytes());     // 8 bytes
            output.extend_from_slice(&sh_offset.to_le_bytes());   // 8 bytes
            output.extend_from_slice(&sh_size.to_le_bytes());     // 8 bytes
            output.extend_from_slice(&sh_link.to_le_bytes());     // 4 bytes
            output.extend_from_slice(&sh_info.to_le_bytes());     // 4 bytes
            output.extend_from_slice(&sh_addralign.to_le_bytes()); // 8 bytes
            output.extend_from_slice(&sh_entsize.to_le_bytes());  // 8 bytes
        } else {
            // Elf32_Shdr layout (40 bytes total):
            output.extend_from_slice(&sh_name.to_le_bytes());                // 4 bytes
            output.extend_from_slice(&sh_type.to_le_bytes());                // 4 bytes
            output.extend_from_slice(&(sh_flags as u32).to_le_bytes());      // 4 bytes
            output.extend_from_slice(&(sh_addr as u32).to_le_bytes());       // 4 bytes
            output.extend_from_slice(&(sh_offset as u32).to_le_bytes());     // 4 bytes
            output.extend_from_slice(&(sh_size as u32).to_le_bytes());       // 4 bytes
            output.extend_from_slice(&sh_link.to_le_bytes());                // 4 bytes
            output.extend_from_slice(&sh_info.to_le_bytes());                // 4 bytes
            output.extend_from_slice(&(sh_addralign as u32).to_le_bytes());  // 4 bytes
            output.extend_from_slice(&(sh_entsize as u32).to_le_bytes());    // 4 bytes
        }
    }
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Rounds `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two (or 1). If `value` is already aligned,
/// it is returned unchanged.
#[inline]
fn align_up(value: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        return value;
    }
    // For power-of-two alignment: (value + alignment - 1) & !(alignment - 1)
    (value + alignment - 1) & !(alignment - 1)
}

/// Pads the output buffer with zero bytes until it reaches the target offset.
///
/// If the buffer is already at or past the target offset, this is a no-op.
#[inline]
fn pad_to(output: &mut Vec<u8>, target_offset: usize) {
    if output.len() < target_offset {
        output.resize(target_offset, 0u8);
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // StringTable tests
    // -----------------------------------------------------------------------

    #[test]
    fn string_table_starts_with_null_byte() {
        let st = StringTable::new();
        assert_eq!(st.as_bytes(), &[0u8]);
        assert_eq!(st.len(), 1);
    }

    #[test]
    fn string_table_add_and_lookup() {
        let mut st = StringTable::new();
        let offset = st.add_string(".text");
        assert_eq!(offset, 1); // first string starts at byte 1
        assert_eq!(st.get_offset(".text"), Some(1));

        // Verify the raw bytes: \0 . t e x t \0
        let expected: Vec<u8> = vec![0, b'.', b't', b'e', b'x', b't', 0];
        assert_eq!(st.as_bytes(), &expected[..]);
    }

    #[test]
    fn string_table_deduplication() {
        let mut st = StringTable::new();
        let off1 = st.add_string(".text");
        let off2 = st.add_string(".data");
        let off3 = st.add_string(".text"); // duplicate

        assert_eq!(off1, off3); // same offset for duplicate
        assert_ne!(off1, off2); // different strings get different offsets
    }

    #[test]
    fn string_table_empty_string_returns_zero() {
        let mut st = StringTable::new();
        assert_eq!(st.add_string(""), 0);
        assert_eq!(st.get_offset(""), Some(0));
    }

    #[test]
    fn string_table_unknown_string_returns_none() {
        let st = StringTable::new();
        assert_eq!(st.get_offset("nonexistent"), None);
    }

    // -----------------------------------------------------------------------
    // ElfSection tests
    // -----------------------------------------------------------------------

    #[test]
    fn elf_section_defaults() {
        let section = ElfSection::new(".text", SHT_PROGBITS);
        assert_eq!(section.name, ".text");
        assert_eq!(section.section_type, SHT_PROGBITS);
        assert_eq!(section.flags, 0);
        assert!(section.data.is_empty());
        assert_eq!(section.alignment, 1);
        assert_eq!(section.entry_size, 0);
        assert_eq!(section.link, 0);
        assert_eq!(section.info, 0);
        assert_eq!(section.addr, 0);
    }

    // -----------------------------------------------------------------------
    // ElfSymbol tests
    // -----------------------------------------------------------------------

    #[test]
    fn elf_symbol_defaults() {
        let sym = ElfSymbol::new("main");
        assert_eq!(sym.name, "main");
        assert_eq!(sym.value, 0);
        assert_eq!(sym.size, 0);
        assert_eq!(sym.binding, STB_LOCAL);
        assert_eq!(sym.sym_type, STT_NOTYPE);
        assert_eq!(sym.visibility, STV_DEFAULT);
        assert_eq!(sym.section_index, SHN_UNDEF);
    }

    #[test]
    fn elf_symbol_st_info_encoding() {
        let mut sym = ElfSymbol::new("func");
        sym.binding = STB_GLOBAL;
        sym.sym_type = STT_FUNC;
        // st_info = (1 << 4) | 2 = 0x12
        assert_eq!(sym.st_info(), 0x12);
    }

    #[test]
    fn elf_symbol_st_other_encoding() {
        let mut sym = ElfSymbol::new("hidden_func");
        sym.visibility = STV_HIDDEN;
        assert_eq!(sym.st_other(), 2);
    }

    // -----------------------------------------------------------------------
    // ProgramHeader tests
    // -----------------------------------------------------------------------

    #[test]
    fn program_header_defaults() {
        let phdr = ProgramHeader::new(PT_LOAD);
        assert_eq!(phdr.p_type, PT_LOAD);
        assert_eq!(phdr.p_flags, 0);
        assert_eq!(phdr.p_offset, 0);
        assert_eq!(phdr.p_vaddr, 0);
        assert_eq!(phdr.p_paddr, 0);
        assert_eq!(phdr.p_filesz, 0);
        assert_eq!(phdr.p_memsz, 0);
        assert_eq!(phdr.p_align, 1);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — basic construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn elf_writer_new_defaults() {
        let writer = ElfWriter::new(Target::X86_64);
        assert_eq!(writer.elf_type, ET_REL);
        assert_eq!(writer.entry_point, 0);
        assert!(writer.sections.is_empty());
        assert!(writer.symbols.is_empty());
        assert!(writer.program_headers.is_empty());
    }

    #[test]
    fn elf_writer_set_type() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_type(ET_EXEC);
        assert_eq!(writer.elf_type, ET_EXEC);
    }

    #[test]
    fn elf_writer_set_entry_point() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_entry_point(0x401000);
        assert_eq!(writer.entry_point, 0x401000);
    }

    #[test]
    fn elf_writer_add_section_returns_correct_index() {
        let mut writer = ElfWriter::new(Target::X86_64);
        let idx1 = writer.add_section(ElfSection::new(".text", SHT_PROGBITS));
        let idx2 = writer.add_section(ElfSection::new(".data", SHT_PROGBITS));
        assert_eq!(idx1, 1); // index 0 is null section
        assert_eq!(idx2, 2);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — ELF64 header validation
    // -----------------------------------------------------------------------

    #[test]
    fn elf64_header_magic_and_class() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_type(ET_EXEC);
        let output = writer.write();

        // ELF magic
        assert_eq!(&output[0..4], &ELF_MAGIC);
        // EI_CLASS = ELFCLASS64
        assert_eq!(output[4], ELFCLASS64);
        // EI_DATA = ELFDATA2LSB
        assert_eq!(output[5], ELFDATA2LSB);
        // EI_VERSION = EV_CURRENT
        assert_eq!(output[6], EV_CURRENT);
        // EI_OSABI = ELFOSABI_NONE
        assert_eq!(output[7], ELFOSABI_NONE);
    }

    #[test]
    fn elf64_header_machine_x86_64() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        // e_machine at offset 18 (2 bytes, little-endian)
        let e_machine = u16::from_le_bytes([output[18], output[19]]);
        assert_eq!(e_machine, EM_X86_64);
    }

    #[test]
    fn elf64_header_machine_aarch64() {
        let writer = ElfWriter::new(Target::AArch64);
        let output = writer.write();

        let e_machine = u16::from_le_bytes([output[18], output[19]]);
        assert_eq!(e_machine, EM_AARCH64);
    }

    #[test]
    fn elf64_header_machine_riscv64() {
        let writer = ElfWriter::new(Target::RiscV64);
        let output = writer.write();

        let e_machine = u16::from_le_bytes([output[18], output[19]]);
        assert_eq!(e_machine, EM_RISCV);
    }

    #[test]
    fn elf64_header_size() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        // e_ehsize at offset 52 (2 bytes)
        let e_ehsize = u16::from_le_bytes([output[52], output[53]]);
        assert_eq!(e_ehsize, ELF64_EHDR_SIZE as u16);
    }

    #[test]
    fn elf64_entry_point() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_type(ET_EXEC);
        writer.set_entry_point(0x401000);
        let output = writer.write();

        // e_entry at offset 24 (8 bytes for ELF64)
        let e_entry = u64::from_le_bytes([
            output[24], output[25], output[26], output[27],
            output[28], output[29], output[30], output[31],
        ]);
        assert_eq!(e_entry, 0x401000);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — ELF32 header validation
    // -----------------------------------------------------------------------

    #[test]
    fn elf32_header_magic_and_class() {
        let writer = ElfWriter::new(Target::I686);
        let output = writer.write();

        assert_eq!(&output[0..4], &ELF_MAGIC);
        assert_eq!(output[4], ELFCLASS32);
        assert_eq!(output[5], ELFDATA2LSB);
    }

    #[test]
    fn elf32_header_machine_i686() {
        let writer = ElfWriter::new(Target::I686);
        let output = writer.write();

        let e_machine = u16::from_le_bytes([output[18], output[19]]);
        assert_eq!(e_machine, EM_386);
    }

    #[test]
    fn elf32_header_size() {
        let writer = ElfWriter::new(Target::I686);
        let output = writer.write();

        // e_ehsize at offset 40 for ELF32 header
        let e_ehsize = u16::from_le_bytes([output[40], output[41]]);
        assert_eq!(e_ehsize, ELF32_EHDR_SIZE as u16);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — Section header table validation
    // -----------------------------------------------------------------------

    #[test]
    fn elf64_has_null_section_plus_auto_sections() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        // e_shnum at offset 60 (2 bytes for ELF64)
        let e_shnum = u16::from_le_bytes([output[60], output[61]]);
        // With no user sections: null + .symtab + .strtab + .shstrtab = 4
        assert_eq!(e_shnum, 4);
    }

    #[test]
    fn elf64_shstrndx_points_to_last_section() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        let e_shnum = u16::from_le_bytes([output[60], output[61]]);
        let e_shstrndx = u16::from_le_bytes([output[62], output[63]]);
        // .shstrtab is always the last section
        assert_eq!(e_shstrndx, e_shnum - 1);
    }

    #[test]
    fn elf64_with_user_section_count() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.add_section(ElfSection::new(".text", SHT_PROGBITS));
        writer.add_section(ElfSection::new(".data", SHT_PROGBITS));
        let output = writer.write();

        let e_shnum = u16::from_le_bytes([output[60], output[61]]);
        // null + .text + .data + .symtab + .strtab + .shstrtab = 6
        assert_eq!(e_shnum, 6);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — Symbol table validation
    // -----------------------------------------------------------------------

    #[test]
    fn elf64_symbol_table_has_null_entry() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        // Find the .symtab section. Its offset is in the section header table.
        // We know .symtab is at section index 1 (after null section) when
        // no user sections are added.
        // With no user sections: section layout is:
        //   [0] null, [1] .symtab, [2] .strtab, [3] .shstrtab
        // So .symtab section header is at index 1.
        // The section header table starts at e_shoff.
        let e_shoff = u64::from_le_bytes([
            output[40], output[41], output[42], output[43],
            output[44], output[45], output[46], output[47],
        ]) as usize;

        // Skip null section header (64 bytes for ELF64), read .symtab header.
        let symtab_shdr_offset = e_shoff + ELF64_SHDR_SIZE; // section index 1

        // sh_offset is at byte 24 of the section header (8 bytes in Elf64_Shdr)
        let sh_offset = u64::from_le_bytes([
            output[symtab_shdr_offset + 24],
            output[symtab_shdr_offset + 25],
            output[symtab_shdr_offset + 26],
            output[symtab_shdr_offset + 27],
            output[symtab_shdr_offset + 28],
            output[symtab_shdr_offset + 29],
            output[symtab_shdr_offset + 30],
            output[symtab_shdr_offset + 31],
        ]) as usize;

        // sh_size is at byte 32 of the section header (8 bytes in Elf64_Shdr)
        let sh_size = u64::from_le_bytes([
            output[symtab_shdr_offset + 32],
            output[symtab_shdr_offset + 33],
            output[symtab_shdr_offset + 34],
            output[symtab_shdr_offset + 35],
            output[symtab_shdr_offset + 36],
            output[symtab_shdr_offset + 37],
            output[symtab_shdr_offset + 38],
            output[symtab_shdr_offset + 39],
        ]) as usize;

        // With no user symbols, .symtab should have exactly one entry (null symbol).
        assert_eq!(sh_size, ELF64_SYM_SIZE);

        // The null symbol entry should be all zeros.
        let null_sym_bytes = &output[sh_offset..sh_offset + ELF64_SYM_SIZE];
        assert!(null_sym_bytes.iter().all(|&b| b == 0));
    }

    // -----------------------------------------------------------------------
    // ElfWriter — Program header validation
    // -----------------------------------------------------------------------

    #[test]
    fn elf64_no_program_headers_by_default() {
        let writer = ElfWriter::new(Target::X86_64);
        let output = writer.write();

        // e_phoff at offset 32 (8 bytes for ELF64) should be 0
        let e_phoff = u64::from_le_bytes([
            output[32], output[33], output[34], output[35],
            output[36], output[37], output[38], output[39],
        ]);
        assert_eq!(e_phoff, 0);

        // e_phnum at offset 56 (2 bytes) should be 0
        let e_phnum = u16::from_le_bytes([output[56], output[57]]);
        assert_eq!(e_phnum, 0);
    }

    #[test]
    fn elf64_with_program_headers() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_type(ET_EXEC);

        let mut phdr = ProgramHeader::new(PT_LOAD);
        phdr.p_flags = PF_R | PF_X;
        phdr.p_vaddr = 0x400000;
        phdr.p_paddr = 0x400000;
        phdr.p_align = 0x200000;
        writer.add_program_header(phdr);

        let output = writer.write();

        // e_phoff should point immediately after the header (offset 64 for ELF64)
        let e_phoff = u64::from_le_bytes([
            output[32], output[33], output[34], output[35],
            output[36], output[37], output[38], output[39],
        ]);
        assert_eq!(e_phoff, ELF64_EHDR_SIZE as u64);

        // e_phnum should be 1
        let e_phnum = u16::from_le_bytes([output[56], output[57]]);
        assert_eq!(e_phnum, 1);

        // Read the program header p_type
        let phdr_offset = ELF64_EHDR_SIZE;
        let p_type = u32::from_le_bytes([
            output[phdr_offset],
            output[phdr_offset + 1],
            output[phdr_offset + 2],
            output[phdr_offset + 3],
        ]);
        assert_eq!(p_type, PT_LOAD);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — Complete ELF with section data
    // -----------------------------------------------------------------------

    #[test]
    fn elf64_with_text_section_data() {
        let mut writer = ElfWriter::new(Target::X86_64);
        writer.set_type(ET_EXEC);

        let mut text = ElfSection::new(".text", SHT_PROGBITS);
        text.flags = SHF_ALLOC | SHF_EXECINSTR;
        text.data = vec![0xc3]; // ret instruction
        text.alignment = 16;
        writer.add_section(text);

        let output = writer.write();

        // Verify the output is non-empty and starts with ELF magic.
        assert!(output.len() > ELF64_EHDR_SIZE);
        assert_eq!(&output[0..4], &ELF_MAGIC);

        // Verify the .text section data (0xc3 byte) appears in the output.
        assert!(
            output.windows(1).any(|w| w == [0xc3]),
            ".text section data (ret instruction) not found in output"
        );
    }

    #[test]
    fn elf64_riscv_flags() {
        let writer = ElfWriter::new(Target::RiscV64);
        let output = writer.write();

        // e_flags at offset 48 (4 bytes for ELF64)
        let e_flags = u32::from_le_bytes([
            output[48], output[49], output[50], output[51],
        ]);
        // RISC-V should have RVC (0x1) | FLOAT_ABI_DOUBLE (0x4) = 0x5
        assert_eq!(e_flags, 0x5);
    }

    // -----------------------------------------------------------------------
    // ElfWriter — Symbol sorting validation
    // -----------------------------------------------------------------------

    #[test]
    fn symbols_locals_before_globals() {
        let mut writer = ElfWriter::new(Target::X86_64);

        // Add a global symbol first, then a local symbol.
        let mut global_sym = ElfSymbol::new("global_func");
        global_sym.binding = STB_GLOBAL;
        global_sym.sym_type = STT_FUNC;
        global_sym.section_index = 1;
        writer.add_symbol(global_sym);

        let mut local_sym = ElfSymbol::new("local_var");
        local_sym.binding = STB_LOCAL;
        local_sym.sym_type = STT_OBJECT;
        local_sym.section_index = 1;
        writer.add_symbol(local_sym);

        let (sorted, first_global) = writer.sort_symbols();

        // sorted[0] = null symbol, sorted[1] = local_var, sorted[2] = global_func
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].name, "");           // null symbol
        assert_eq!(sorted[1].name, "local_var");   // local first
        assert_eq!(sorted[2].name, "global_func"); // global after
        assert_eq!(first_global, 2);               // first global at index 2
    }

    // -----------------------------------------------------------------------
    // Utility function tests
    // -----------------------------------------------------------------------

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }

    #[test]
    fn align_up_no_alignment() {
        assert_eq!(align_up(42, 1), 42);
        assert_eq!(align_up(0, 1), 0);
    }

    #[test]
    fn pad_to_extends_buffer() {
        let mut buf = vec![0xAA, 0xBB];
        pad_to(&mut buf, 5);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf, vec![0xAA, 0xBB, 0, 0, 0]);
    }

    #[test]
    fn pad_to_noop_when_already_past() {
        let mut buf = vec![0xAA, 0xBB, 0xCC];
        pad_to(&mut buf, 2);
        assert_eq!(buf.len(), 3); // unchanged
    }

    // -----------------------------------------------------------------------
    // ELF constant value verification
    // -----------------------------------------------------------------------

    #[test]
    fn elf_constants_correct_values() {
        // ELF types
        assert_eq!(ET_REL, 1);
        assert_eq!(ET_EXEC, 2);
        assert_eq!(ET_DYN, 3);

        // ELF classes
        assert_eq!(ELFCLASS32, 1);
        assert_eq!(ELFCLASS64, 2);

        // Data encoding
        assert_eq!(ELFDATA2LSB, 1);

        // Machine types
        assert_eq!(EM_386, 3);
        assert_eq!(EM_X86_64, 62);
        assert_eq!(EM_AARCH64, 183);
        assert_eq!(EM_RISCV, 243);

        // Section types
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

        // Section flags
        assert_eq!(SHF_WRITE, 0x1);
        assert_eq!(SHF_ALLOC, 0x2);
        assert_eq!(SHF_EXECINSTR, 0x4);

        // Program header types
        assert_eq!(PT_NULL, 0);
        assert_eq!(PT_LOAD, 1);
        assert_eq!(PT_DYNAMIC, 2);
        assert_eq!(PT_INTERP, 3);
        assert_eq!(PT_NOTE, 4);
        assert_eq!(PT_PHDR, 6);
        assert_eq!(PT_GNU_STACK, 0x6474_e551);
        assert_eq!(PT_GNU_RELRO, 0x6474_e552);

        // Program header flags
        assert_eq!(PF_X, 0x1);
        assert_eq!(PF_W, 0x2);
        assert_eq!(PF_R, 0x4);

        // Symbol binding
        assert_eq!(STB_LOCAL, 0);
        assert_eq!(STB_GLOBAL, 1);
        assert_eq!(STB_WEAK, 2);

        // Symbol types
        assert_eq!(STT_NOTYPE, 0);
        assert_eq!(STT_OBJECT, 1);
        assert_eq!(STT_FUNC, 2);
        assert_eq!(STT_SECTION, 3);
        assert_eq!(STT_FILE, 4);

        // Symbol visibility
        assert_eq!(STV_DEFAULT, 0);
        assert_eq!(STV_HIDDEN, 2);
        assert_eq!(STV_PROTECTED, 3);

        // Special section indices
        assert_eq!(SHN_UNDEF, 0);

        // OS/ABI
        assert_eq!(ELFOSABI_NONE, 0);
    }
}
