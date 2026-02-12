//! Dynamic linking section generator for ELF shared library (ET_DYN) output.
//!
//! When linking with `-shared` or `-fPIC`, the linker must produce several
//! additional ELF sections for the runtime dynamic linker (`ld-linux.so`):
//!
//! - **`.dynamic`** — array of `Elf64_Dyn` entries describing dynamic linking
//!   metadata (needed libraries, symbol tables, relocations, init/fini).
//! - **`.dynsym` / `.dynstr`** — dynamic symbol table and its string table,
//!   containing symbols exported from / imported by this shared object.
//! - **`.gnu.hash`** — GNU-style hash table for efficient dynamic symbol lookup.
//! - **`.got` / `.got.plt`** — Global Offset Table for PIC data/function access.
//! - **`.plt`** — Procedure Linkage Table stubs for lazy function binding.
//! - **`.rela.dyn` / `.rela.plt`** — RELA relocations for the dynamic linker.
//! - **`PT_DYNAMIC`** — program header referencing the `.dynamic` section.
//! - **`PT_INTERP`** — program header naming the dynamic linker path.
//!
//! Removing this module would break all shared library output and PIC linking.

use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// ELF dynamic tag constants
// ===========================================================================

/// ELF dynamic section tag for needed shared library (DT_NEEDED).
pub const DT_NEEDED: u64 = 1;
/// DT_HASH — ELF hash table (we emit DT_GNU_HASH instead, but support both).
pub const DT_HASH: u64 = 4;
/// DT_STRTAB — dynamic string table address.
pub const DT_STRTAB: u64 = 5;
/// DT_SYMTAB — dynamic symbol table address.
pub const DT_SYMTAB: u64 = 6;
/// DT_RELA — RELA relocation table address.
pub const DT_RELA: u64 = 7;
/// DT_RELASZ — total size of RELA table in bytes.
pub const DT_RELASZ: u64 = 8;
/// DT_RELAENT — size of one RELA entry.
pub const DT_RELAENT: u64 = 9;
/// DT_STRSZ — size of the dynamic string table.
pub const DT_STRSZ: u64 = 10;
/// DT_SYMENT — size of one dynamic symbol table entry.
pub const DT_SYMENT: u64 = 11;
/// DT_INIT — address of initialization function.
pub const DT_INIT: u64 = 12;
/// DT_FINI — address of finalization function.
pub const DT_FINI: u64 = 13;
/// DT_SONAME — name of this shared object.
pub const DT_SONAME: u64 = 14;
/// DT_PLTGOT — address of PLT/GOT.
pub const DT_PLTGOT: u64 = 3;
/// DT_PLTRELSZ — total size of PLT relocations.
pub const DT_PLTRELSZ: u64 = 2;
/// DT_PLTREL — type of PLT relocations (DT_RELA = 7).
pub const DT_PLTREL: u64 = 20;
/// DT_JMPREL — address of PLT relocations.
pub const DT_JMPREL: u64 = 23;
/// DT_FLAGS — dynamic flags.
pub const DT_FLAGS: u64 = 30;
/// DT_GNU_HASH — GNU hash table address.
pub const DT_GNU_HASH: u64 = 0x6FFF_FEFF;
/// DT_NULL — terminator entry.
pub const DT_NULL: u64 = 0;

// ===========================================================================
// DynamicEntry — single .dynamic section entry
// ===========================================================================

/// A single entry in the `.dynamic` section (`Elf64_Dyn`).
///
/// Each entry is a tag-value pair. The interpretation of `val` depends
/// on the `tag`.
#[derive(Debug, Clone)]
pub struct DynamicEntry {
    /// Dynamic tag (DT_NEEDED, DT_STRTAB, DT_SYMTAB, etc.).
    pub tag: u64,
    /// Value (address, size, string table offset, etc.).
    pub val: u64,
}

impl DynamicEntry {
    /// Creates a new dynamic entry.
    pub fn new(tag: u64, val: u64) -> Self {
        Self { tag, val }
    }

    /// Serializes this entry to bytes (16 bytes for 64-bit ELF).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.tag.to_le_bytes());
        buf.extend_from_slice(&self.val.to_le_bytes());
        buf
    }
}

// ===========================================================================
// DynamicRelocation — relocation for the dynamic linker
// ===========================================================================

/// A relocation for the runtime dynamic linker, placed in `.rela.dyn` or
/// `.rela.plt`.
///
/// These are `Elf64_Rela` entries processed by `ld-linux.so` at load time.
#[derive(Debug, Clone)]
pub struct DynamicRelocation {
    /// Virtual address where the relocation is applied.
    pub offset: u64,
    /// Architecture-specific relocation type.
    pub reloc_type: u32,
    /// Index into the dynamic symbol table (`.dynsym`).
    pub symbol_index: u32,
    /// Addend.
    pub addend: i64,
}

impl DynamicRelocation {
    /// Serializes this relocation to bytes (24 bytes for 64-bit RELA).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);
        buf.extend_from_slice(&self.offset.to_le_bytes());
        // r_info = (sym_index << 32) | reloc_type
        let r_info: u64 = ((self.symbol_index as u64) << 32) | (self.reloc_type as u64);
        buf.extend_from_slice(&r_info.to_le_bytes());
        buf.extend_from_slice(&self.addend.to_le_bytes());
        buf
    }
}

// ===========================================================================
// DynamicSymbolEntry — entry in .dynsym
// ===========================================================================

/// A symbol entry in the dynamic symbol table (`.dynsym`).
///
/// Only symbols visible at dynamic link time (exported or imported) appear here.
#[derive(Debug, Clone)]
pub struct DynamicSymbolEntry {
    /// Symbol name.
    pub name: String,
    /// Symbol value (virtual address).
    pub value: u64,
    /// Symbol size.
    pub size: u64,
    /// ELF st_info byte (binding << 4 | type).
    pub info: u8,
    /// ELF st_other byte (visibility).
    pub other: u8,
    /// Section header table index (SHN_UNDEF for imports).
    pub shndx: u16,
}

// ===========================================================================
// DynamicSymbolTable — .dynsym + .dynstr builder
// ===========================================================================

/// Builder for the `.dynsym` dynamic symbol table and its associated
/// `.dynstr` string table.
///
/// Maintains deduplication of symbol names in the string table.
#[derive(Debug, Clone)]
pub struct DynamicSymbolTable {
    /// Dynamic symbol entries in output order.
    pub symbols: Vec<DynamicSymbolEntry>,
    /// String table bytes (including leading NUL byte).
    pub strtab: Vec<u8>,
    /// Map from symbol name to index in `symbols`.
    pub symbol_map: FxHashMap<String, usize>,
    /// Map from string to offset in `strtab` for deduplication.
    name_offsets: FxHashMap<String, u32>,
}

impl DynamicSymbolTable {
    /// Creates a new dynamic symbol table with the mandatory null entry.
    pub fn new() -> Self {
        // Leading NUL byte for the string table
        let strtab = vec![0u8];

        // First entry is always the null symbol
        let syms = vec![DynamicSymbolEntry {
            name: String::new(),
            value: 0,
            size: 0,
            info: 0,
            other: 0,
            shndx: 0,
        }];

        Self {
            symbols: syms,
            strtab,
            symbol_map: FxHashMap::default(),
            name_offsets: FxHashMap::default(),
        }
    }

    /// Adds a symbol to the dynamic symbol table.
    ///
    /// Returns the index of the new symbol in the table.
    pub fn add_symbol(&mut self, sym: DynamicSymbolEntry) -> usize {
        let name = sym.name.clone();
        let idx = self.symbols.len();

        // Add name to strtab (with deduplication)
        if !name.is_empty() && !self.name_offsets.contains_key(&name) {
            let offset = self.strtab.len() as u32;
            self.strtab.extend_from_slice(name.as_bytes());
            self.strtab.push(0); // NUL terminator
            self.name_offsets.insert(name.clone(), offset);
        }

        self.symbol_map.insert(name, idx);
        self.symbols.push(sym);
        idx
    }

    /// Returns the string table offset for the given name.
    pub fn get_name_offset(&self, name: &str) -> Option<u32> {
        self.name_offsets.get(name).copied()
    }

    /// Returns the index of a symbol by name.
    pub fn get_symbol_index(&self, name: &str) -> Option<usize> {
        self.symbol_map.get(name).copied()
    }

    /// Returns the number of symbols (including the null entry).
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the size of the string table in bytes.
    pub fn strtab_size(&self) -> usize {
        self.strtab.len()
    }

    /// Serializes the symbol table to bytes for 64-bit ELF.
    /// Each Elf64_Sym is 24 bytes.
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.symbols.len() * 24);
        for sym in &self.symbols {
            let name_offset = self
                .name_offsets
                .get(&sym.name)
                .copied()
                .unwrap_or(0);
            buf.extend_from_slice(&name_offset.to_le_bytes()); // st_name (4 bytes)
            buf.push(sym.info); // st_info (1 byte)
            buf.push(sym.other); // st_other (1 byte)
            buf.extend_from_slice(&sym.shndx.to_le_bytes()); // st_shndx (2 bytes)
            buf.extend_from_slice(&sym.value.to_le_bytes()); // st_value (8 bytes)
            buf.extend_from_slice(&sym.size.to_le_bytes()); // st_size (8 bytes)
        }
        buf
    }
}

impl Default for DynamicSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// GnuHashTable — .gnu.hash builder
// ===========================================================================

/// GNU hash function — the standard hash for `.gnu.hash` sections.
///
/// Computes the hash value used by the `.gnu.hash` section in ELF files
/// for efficient dynamic symbol lookup. The algorithm multiplies the running
/// hash by 33 and adds each byte.
pub fn gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

// ===========================================================================
// GotBuilder — .got / .got.plt builder
// ===========================================================================

/// Builder for the Global Offset Table (`.got` and `.got.plt`) sections.
///
/// The GOT contains entries that the dynamic linker patches at load time
/// with the actual addresses of global data and function pointers.
#[derive(Debug, Clone)]
pub struct GotBuilder {
    /// GOT entries: symbol name → GOT slot index.
    entries: FxHashMap<String, usize>,
    /// Number of reserved entries (GOT[0] is _DYNAMIC, etc.).
    reserved_count: usize,
    /// Address size in bytes (8 for 64-bit, 4 for 32-bit).
    entry_size: usize,
}

impl GotBuilder {
    /// Creates a new GOT builder.
    ///
    /// - `is_64bit`: True for 64-bit targets, false for 32-bit.
    pub fn new(is_64bit: bool) -> Self {
        Self {
            entries: FxHashMap::default(),
            reserved_count: 3, // GOT[0]=_DYNAMIC, GOT[1]=link_map, GOT[2]=resolver
            entry_size: if is_64bit { 8 } else { 4 },
        }
    }

    /// Adds a symbol to the GOT, returning its slot index.
    ///
    /// If the symbol already has a GOT entry, returns the existing index.
    pub fn add_entry(&mut self, symbol_name: &str) -> usize {
        let next_idx = self.reserved_count + self.entries.len();
        *self.entries.entry(symbol_name.to_string()).or_insert(next_idx)
    }

    /// Returns the GOT slot index for a symbol, if it has an entry.
    pub fn get_entry(&self, symbol_name: &str) -> Option<usize> {
        self.entries.get(symbol_name).copied()
    }

    /// Returns the total number of GOT slots (reserved + user entries).
    pub fn total_entries(&self) -> usize {
        self.reserved_count + self.entries.len()
    }

    /// Returns the total size of the GOT in bytes.
    pub fn total_size(&self) -> usize {
        self.total_entries() * self.entry_size
    }

    /// Returns the entry size in bytes.
    pub fn entry_size(&self) -> usize {
        self.entry_size
    }

    /// Serializes the GOT to bytes (all zeros initially; patched by the
    /// dynamic linker at load time).
    pub fn to_bytes(&self) -> Vec<u8> {
        vec![0u8; self.total_size()]
    }
}

// ===========================================================================
// PltBuilder — .plt stub generator
// ===========================================================================

/// Builder for the Procedure Linkage Table (`.plt`) stubs.
///
/// Each PLT entry is a small code stub that performs lazy binding:
/// on first call, it jumps to the dynamic linker resolver; on subsequent
/// calls, it jumps through the GOT (already patched by the resolver).
///
/// PLT stub format is architecture-specific; this builder stores the
/// abstract mapping and delegates concrete stub generation to the arch backend.
#[derive(Debug, Clone)]
pub struct PltBuilder {
    /// PLT entries: symbol name → PLT slot index.
    entries: FxHashMap<String, usize>,
    /// Size of each PLT entry in bytes (architecture-dependent).
    entry_size: usize,
    /// Size of the PLT header (PLT[0]) in bytes.
    header_size: usize,
}

impl PltBuilder {
    /// Creates a new PLT builder.
    ///
    /// - `entry_size`: Size of each PLT entry in bytes (16 for x86-64).
    /// - `header_size`: Size of the PLT header (PLT[0]) in bytes (16 for x86-64).
    pub fn new(entry_size: usize, header_size: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            entry_size,
            header_size,
        }
    }

    /// Adds a symbol to the PLT, returning its slot index.
    pub fn add_entry(&mut self, symbol_name: &str) -> usize {
        let next_idx = self.entries.len();
        *self.entries.entry(symbol_name.to_string()).or_insert(next_idx)
    }

    /// Returns the PLT slot index for a symbol, if it has an entry.
    pub fn get_entry(&self, symbol_name: &str) -> Option<usize> {
        self.entries.get(symbol_name).copied()
    }

    /// Returns the total number of PLT entries (excluding header).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total size of the PLT section in bytes (header + entries).
    pub fn total_size(&self) -> usize {
        self.header_size + self.entries.len() * self.entry_size
    }

    /// Returns the byte offset of a PLT entry given its slot index.
    pub fn entry_offset(&self, slot_index: usize) -> usize {
        self.header_size + slot_index * self.entry_size
    }
}

// ===========================================================================
// DynamicLayout — computed addresses for all dynamic sections
// ===========================================================================

/// Final computed addresses and sizes for all dynamic linking sections.
///
/// Populated after section merging and address assignment, used during
/// final ELF writing to fill in cross-references between dynamic sections.
#[derive(Debug, Clone, Default)]
pub struct DynamicLayout {
    /// Virtual address of `.dynamic`.
    pub dynamic_addr: u64,
    /// Size of `.dynamic` in bytes.
    pub dynamic_size: u64,
    /// Virtual address of `.dynsym`.
    pub dynsym_addr: u64,
    /// Size of `.dynsym` in bytes.
    pub dynsym_size: u64,
    /// Virtual address of `.dynstr`.
    pub dynstr_addr: u64,
    /// Size of `.dynstr` in bytes.
    pub dynstr_size: u64,
    /// Virtual address of `.gnu.hash`.
    pub gnu_hash_addr: u64,
    /// Virtual address of `.got`.
    pub got_addr: u64,
    /// Size of `.got` in bytes.
    pub got_size: u64,
    /// Virtual address of `.got.plt`.
    pub got_plt_addr: u64,
    /// Size of `.got.plt` in bytes.
    pub got_plt_size: u64,
    /// Virtual address of `.plt`.
    pub plt_addr: u64,
    /// Size of `.plt` in bytes.
    pub plt_size: u64,
    /// Virtual address of `.rela.dyn`.
    pub rela_dyn_addr: u64,
    /// Size of `.rela.dyn` in bytes.
    pub rela_dyn_size: u64,
    /// Virtual address of `.rela.plt`.
    pub rela_plt_addr: u64,
    /// Size of `.rela.plt` in bytes.
    pub rela_plt_size: u64,
    /// Path to the dynamic linker (for `PT_INTERP`).
    pub interp_path: String,
}

// ===========================================================================
// DynamicSectionBuilder — orchestrator for dynamic section generation
// ===========================================================================

/// Top-level builder that orchestrates construction of all dynamic linking
/// sections for a shared library or dynamically-linked executable.
///
/// # Usage
///
/// ```text
/// let mut builder = DynamicSectionBuilder::new(Target::X86_64);
///
/// // Add needed libraries
/// builder.add_needed_library("libc.so.6");
///
/// // Add exported/imported symbols
/// builder.add_dynamic_symbol(sym);
///
/// // Set section addresses (after address assignment)
/// builder.set_layout(layout);
///
/// // Generate .dynamic entries
/// let entries = builder.build_dynamic_entries();
/// ```
pub struct DynamicSectionBuilder {
    /// Target architecture.
    target: Target,
    /// Needed shared libraries (`-l` flags resolved to SO names).
    needed_libraries: Vec<String>,
    /// Dynamic symbol table builder.
    pub dynsym: DynamicSymbolTable,
    /// GOT builder.
    pub got: GotBuilder,
    /// PLT builder.
    pub plt: PltBuilder,
    /// Dynamic relocations for `.rela.dyn`.
    pub rela_dyn: Vec<DynamicRelocation>,
    /// PLT relocations for `.rela.plt`.
    pub rela_plt: Vec<DynamicRelocation>,
    /// Section layout (set after address assignment).
    pub layout: DynamicLayout,
    /// Optional SONAME for this shared object.
    pub soname: Option<String>,
    /// Optional INIT function address.
    pub init_addr: Option<u64>,
    /// Optional FINI function address.
    pub fini_addr: Option<u64>,
}

impl DynamicSectionBuilder {
    /// Creates a new dynamic section builder for the given target architecture.
    pub fn new(target: Target) -> Self {
        let is_64bit = target.pointer_width() == 8;
        let (plt_entry_size, plt_header_size) = match target {
            Target::X86_64 => (16, 16),
            Target::I686 => (16, 16),
            Target::AArch64 => (16, 32),
            Target::RiscV64 => (16, 32),
        };

        Self {
            target,
            needed_libraries: Vec::new(),
            dynsym: DynamicSymbolTable::new(),
            got: GotBuilder::new(is_64bit),
            plt: PltBuilder::new(plt_entry_size, plt_header_size),
            rela_dyn: Vec::new(),
            rela_plt: Vec::new(),
            layout: DynamicLayout::default(),
            soname: None,
            init_addr: None,
            fini_addr: None,
        }
    }

    /// Adds a needed shared library (DT_NEEDED entry).
    pub fn add_needed_library(&mut self, name: &str) {
        self.needed_libraries.push(name.to_string());
    }

    /// Adds a symbol to the dynamic symbol table.
    pub fn add_dynamic_symbol(&mut self, sym: DynamicSymbolEntry) -> usize {
        self.dynsym.add_symbol(sym)
    }

    /// Adds a GOT entry for a symbol.
    pub fn add_got_entry(&mut self, symbol_name: &str) -> usize {
        self.got.add_entry(symbol_name)
    }

    /// Adds a PLT entry for a symbol.
    pub fn add_plt_entry(&mut self, symbol_name: &str) -> usize {
        self.plt.add_entry(symbol_name)
    }

    /// Adds a dynamic relocation to `.rela.dyn`.
    pub fn add_rela_dyn(&mut self, reloc: DynamicRelocation) {
        self.rela_dyn.push(reloc);
    }

    /// Adds a PLT relocation to `.rela.plt`.
    pub fn add_rela_plt(&mut self, reloc: DynamicRelocation) {
        self.rela_plt.push(reloc);
    }

    /// Sets the computed layout (addresses and sizes) for all dynamic sections.
    pub fn set_layout(&mut self, layout: DynamicLayout) {
        self.layout = layout;
    }

    /// Returns the dynamic linker path for the target architecture.
    pub fn interp_path(&self) -> &str {
        match self.target {
            Target::X86_64 => "/lib64/ld-linux-x86-64.so.2",
            Target::I686 => "/lib/ld-linux.so.2",
            Target::AArch64 => "/lib/ld-linux-aarch64.so.1",
            Target::RiscV64 => "/lib/ld-linux-riscv64-lp64d.so.1",
        }
    }

    /// Builds the `.dynamic` section entries based on the current layout
    /// and configuration.
    ///
    /// Returns a vector of [`DynamicEntry`] that can be serialized to bytes.
    pub fn build_dynamic_entries(&self) -> Vec<DynamicEntry> {
        let mut entries = Vec::new();

        // DT_NEEDED entries for shared library dependencies
        for lib_name in &self.needed_libraries {
            let offset = self.dynsym.get_name_offset(lib_name).unwrap_or(0) as u64;
            entries.push(DynamicEntry::new(DT_NEEDED, offset));
        }

        // SONAME
        if let Some(ref soname) = self.soname {
            let offset = self.dynsym.get_name_offset(soname).unwrap_or(0) as u64;
            entries.push(DynamicEntry::new(DT_SONAME, offset));
        }

        // Symbol table and string table
        entries.push(DynamicEntry::new(DT_SYMTAB, self.layout.dynsym_addr));
        entries.push(DynamicEntry::new(DT_SYMENT, 24)); // sizeof(Elf64_Sym)
        entries.push(DynamicEntry::new(DT_STRTAB, self.layout.dynstr_addr));
        entries.push(DynamicEntry::new(DT_STRSZ, self.layout.dynstr_size));

        // GNU hash table
        if self.layout.gnu_hash_addr != 0 {
            entries.push(DynamicEntry::new(DT_GNU_HASH, self.layout.gnu_hash_addr));
        }

        // GOT/PLT
        if self.layout.got_plt_addr != 0 {
            entries.push(DynamicEntry::new(DT_PLTGOT, self.layout.got_plt_addr));
        }

        // RELA relocations
        if self.layout.rela_dyn_size > 0 {
            entries.push(DynamicEntry::new(DT_RELA, self.layout.rela_dyn_addr));
            entries.push(DynamicEntry::new(DT_RELASZ, self.layout.rela_dyn_size));
            entries.push(DynamicEntry::new(DT_RELAENT, 24)); // sizeof(Elf64_Rela)
        }

        // PLT relocations
        if self.layout.rela_plt_size > 0 {
            entries.push(DynamicEntry::new(DT_JMPREL, self.layout.rela_plt_addr));
            entries.push(DynamicEntry::new(DT_PLTRELSZ, self.layout.rela_plt_size));
            entries.push(DynamicEntry::new(DT_PLTREL, 7)); // DT_RELA
        }

        // INIT / FINI
        if let Some(addr) = self.init_addr {
            entries.push(DynamicEntry::new(DT_INIT, addr));
        }
        if let Some(addr) = self.fini_addr {
            entries.push(DynamicEntry::new(DT_FINI, addr));
        }

        // DT_NULL terminator
        entries.push(DynamicEntry::new(DT_NULL, 0));

        entries
    }

    /// Serializes all `.dynamic` entries to bytes.
    pub fn dynamic_section_bytes(&self) -> Vec<u8> {
        let entries = self.build_dynamic_entries();
        let mut buf = Vec::with_capacity(entries.len() * 16);
        for entry in &entries {
            buf.extend_from_slice(&entry.to_bytes_le());
        }
        buf
    }

    /// Serializes `.rela.dyn` to bytes.
    pub fn rela_dyn_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.rela_dyn.len() * 24);
        for reloc in &self.rela_dyn {
            buf.extend_from_slice(&reloc.to_bytes_le());
        }
        buf
    }

    /// Serializes `.rela.plt` to bytes.
    pub fn rela_plt_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.rela_plt.len() * 24);
        for reloc in &self.rela_plt {
            buf.extend_from_slice(&reloc.to_bytes_le());
        }
        buf
    }

    /// Returns the target architecture.
    pub fn target(&self) -> Target {
        self.target
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_entry_to_bytes() {
        let entry = DynamicEntry::new(DT_NEEDED, 42);
        let bytes = entry.to_bytes_le();
        assert_eq!(bytes.len(), 16);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), DT_NEEDED);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 42);
    }

    #[test]
    fn test_dynamic_relocation_to_bytes() {
        let reloc = DynamicRelocation {
            offset: 0x1000,
            reloc_type: 7,
            symbol_index: 3,
            addend: -4,
        };
        let bytes = reloc.to_bytes_le();
        assert_eq!(bytes.len(), 24);
        assert_eq!(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            0x1000
        );
    }

    #[test]
    fn test_dynamic_symbol_table_new() {
        let table = DynamicSymbolTable::new();
        // Should have the mandatory null symbol
        assert_eq!(table.symbol_count(), 1);
        // String table should start with NUL
        assert_eq!(table.strtab[0], 0);
    }

    #[test]
    fn test_add_dynamic_symbol() {
        let mut table = DynamicSymbolTable::new();
        let idx = table.add_symbol(DynamicSymbolEntry {
            name: "printf".to_string(),
            value: 0,
            size: 0,
            info: 0x12, // STB_GLOBAL | STT_FUNC
            other: 0,
            shndx: 0,
        });
        assert_eq!(idx, 1);
        assert_eq!(table.symbol_count(), 2);
        assert!(table.get_name_offset("printf").is_some());
        assert_eq!(table.get_symbol_index("printf"), Some(1));
    }

    #[test]
    fn test_got_builder() {
        let mut got = GotBuilder::new(true);
        assert_eq!(got.total_entries(), 3); // Reserved entries
        assert_eq!(got.entry_size(), 8);

        let idx = got.add_entry("foo");
        assert_eq!(idx, 3);
        assert_eq!(got.total_entries(), 4);

        // Adding same symbol returns same index
        let idx2 = got.add_entry("foo");
        assert_eq!(idx2, 3);
        assert_eq!(got.total_entries(), 4);
    }

    #[test]
    fn test_plt_builder() {
        let mut plt = PltBuilder::new(16, 16);
        assert_eq!(plt.entry_count(), 0);
        assert_eq!(plt.total_size(), 16); // Header only

        let idx = plt.add_entry("printf");
        assert_eq!(idx, 0);
        assert_eq!(plt.entry_count(), 1);
        assert_eq!(plt.total_size(), 32); // Header + 1 entry
        assert_eq!(plt.entry_offset(0), 16);
    }

    #[test]
    fn test_dynamic_layout_default() {
        let layout = DynamicLayout::default();
        assert_eq!(layout.dynamic_addr, 0);
        assert_eq!(layout.plt_size, 0);
        assert!(layout.interp_path.is_empty());
    }

    #[test]
    fn test_dynamic_section_builder() {
        let mut builder = DynamicSectionBuilder::new(Target::X86_64);
        builder.add_needed_library("libc.so.6");

        let entries = builder.build_dynamic_entries();
        // Should have at least: DT_NEEDED, DT_SYMTAB, DT_SYMENT, DT_STRTAB,
        // DT_STRSZ, DT_NULL
        assert!(entries.len() >= 6);

        // Last entry should be DT_NULL
        assert_eq!(entries.last().unwrap().tag, DT_NULL);
    }

    #[test]
    fn test_interp_paths() {
        assert_eq!(
            DynamicSectionBuilder::new(Target::X86_64).interp_path(),
            "/lib64/ld-linux-x86-64.so.2"
        );
        assert_eq!(
            DynamicSectionBuilder::new(Target::I686).interp_path(),
            "/lib/ld-linux.so.2"
        );
        assert_eq!(
            DynamicSectionBuilder::new(Target::AArch64).interp_path(),
            "/lib/ld-linux-aarch64.so.1"
        );
        assert_eq!(
            DynamicSectionBuilder::new(Target::RiscV64).interp_path(),
            "/lib/ld-linux-riscv64-lp64d.so.1"
        );
    }

    #[test]
    fn test_gnu_hash_function() {
        // Known test vector: gnu_hash("") == 5381
        assert_eq!(gnu_hash(b""), 5381);
        // Verify determinism
        assert_eq!(gnu_hash(b"printf"), gnu_hash(b"printf"));
        // Different names produce different hashes (with high probability)
        assert_ne!(gnu_hash(b"printf"), gnu_hash(b"scanf"));
    }
}
