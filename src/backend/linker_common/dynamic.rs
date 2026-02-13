//! Dynamic linking section generator for ELF shared library (ET_DYN) output.
//!
//! When linking with `-shared` or `-fPIC`, the linker must produce several
//! additional ELF sections for the runtime dynamic linker (`ld-linux.so`):
//!
//! - **`.dynamic`** — array of `Elf64_Dyn`/`Elf32_Dyn` entries describing
//!   dynamic linking metadata (needed libraries, symbol tables, relocations,
//!   init/fini addresses).
//! - **`.dynsym` / `.dynstr`** — dynamic symbol table and its string table,
//!   containing symbols exported from / imported by this shared object.
//! - **`.gnu.hash`** — GNU-style hash table for O(1) dynamic symbol lookup.
//! - **`.got` / `.got.plt`** — Global Offset Table for PIC data/function access.
//! - **`.plt`** — Procedure Linkage Table stubs for lazy function binding.
//! - **`.rela.dyn` / `.rela.plt`** — RELA relocations for the dynamic linker.
//! - **`PT_DYNAMIC`** — program header referencing the `.dynamic` section.
//! - **`PT_INTERP`** — program header naming the dynamic linker path.
//!
//! This module supports all four BCC target architectures: x86-64, i686,
//! AArch64, and RISC-V 64. Architecture-specific PLT stub code generation
//! is dispatched based on the [`Target`] enum.
//!
//! Removing this module would break all shared library generation and PIC linking.

use crate::backend::elf_writer_common::{ProgramHeader, PF_R, PF_W, PT_DYNAMIC, PT_INTERP};
use crate::backend::linker_common::symbol_resolver::{
    SymbolBinding, SymbolEntry, SymbolType, SymbolVisibility,
};
use crate::common::fx_hash::{fx_hash_map, FxHashMap};
use crate::common::target::Target;

// ===========================================================================
// ELF Dynamic Tag Constants (DT_*)
// ===========================================================================
// Hand-defined per zero-dependency mandate. All tags are i64 to match
// the signed `d_tag` field of `Elf64_Dyn` / `Elf32_Dyn`.

/// Marks the end of the `_DYNAMIC` array.
pub const DT_NULL: i64 = 0;
/// String-table offset of a needed shared library name.
pub const DT_NEEDED: i64 = 1;
/// Total size in bytes of the PLT relocation entries.
pub const DT_PLTRELSZ: i64 = 2;
/// Address of the PLT and/or GOT.
pub const DT_PLTGOT: i64 = 3;
/// Address of the symbol hash table (SysV-style; superseded by DT_GNU_HASH).
pub const DT_HASH: i64 = 4;
/// Address of the dynamic string table (`.dynstr`).
pub const DT_STRTAB: i64 = 5;
/// Address of the dynamic symbol table (`.dynsym`).
pub const DT_SYMTAB: i64 = 6;
/// Address of the Rela relocation table (`.rela.dyn`).
pub const DT_RELA: i64 = 7;
/// Total size in bytes of the Rela relocation table.
pub const DT_RELASZ: i64 = 8;
/// Size of a single Rela relocation entry (24 bytes for 64-bit).
pub const DT_RELAENT: i64 = 9;
/// Total size in bytes of the dynamic string table.
pub const DT_STRSZ: i64 = 10;
/// Size of a single symbol table entry (24 bytes for 64-bit, 16 for 32-bit).
pub const DT_SYMENT: i64 = 11;
/// Address of the initialization function.
pub const DT_INIT: i64 = 12;
/// Address of the finalization function.
pub const DT_FINI: i64 = 13;
/// String-table offset of this shared object's name (SONAME).
pub const DT_SONAME: i64 = 14;
/// String-table offset of library search path (deprecated in favour of runpath).
pub const DT_RPATH: i64 = 15;
/// Object uses symbolic binding (rarely used).
pub const DT_SYMBOLIC: i64 = 16;
/// Type of relocation entry used for PLT (7 = DT_RELA, 17 = DT_REL).
pub const DT_PLTREL: i64 = 20;
/// Address of the PLT relocation entries (`.rela.plt`).
pub const DT_JMPREL: i64 = 23;
/// Dynamic flags word.
pub const DT_FLAGS: i64 = 30;
/// Extended dynamic flags word.
pub const DT_FLAGS_1: i64 = 0x6fff_fffb;
/// Address of the GNU hash table (`.gnu.hash`).
pub const DT_GNU_HASH: i64 = 0x6fff_fef5;

// Size of Elf64_Sym (used for DT_SYMENT on 64-bit targets).
const ELF64_SYM_SIZE: u64 = 24;
// Size of Elf32_Sym (used for DT_SYMENT on 32-bit targets).
const ELF32_SYM_SIZE: u64 = 16;
// Size of Elf64_Rela (used for DT_RELAENT on 64-bit targets).
const ELF64_RELA_SIZE: u64 = 24;
// Size of Elf32_Rela (used for DT_RELAENT on 32-bit targets).
const ELF32_RELA_SIZE: u64 = 12;

// ===========================================================================
// DynamicEntry — single .dynamic section entry
// ===========================================================================

/// A single entry in the `.dynamic` section, corresponding to `Elf64_Dyn`
/// (or `Elf32_Dyn` on 32-bit targets).
///
/// Each entry is a tag/value pair. The tag identifies the entry type and the
/// value is interpreted according to the tag (address, size, string offset, …).
#[derive(Debug, Clone, Copy)]
pub struct DynamicEntry {
    /// Dynamic tag — one of the `DT_*` constants. Signed because the ELF
    /// specification defines `d_tag` as `Elf64_Sxword` / `Elf32_Sword`.
    pub tag: i64,
    /// Associated value (address, byte count, or string-table offset).
    pub value: u64,
}

impl DynamicEntry {
    /// Creates a new dynamic entry with the given tag and value.
    #[inline]
    pub fn new(tag: i64, value: u64) -> Self {
        Self { tag, value }
    }

    /// Serializes this entry to little-endian bytes for a 64-bit ELF
    /// (16 bytes: 8-byte tag + 8-byte value).
    pub fn to_bytes_64_le(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.tag.to_le_bytes());
        buf[8..16].copy_from_slice(&self.value.to_le_bytes());
        buf
    }

    /// Serializes this entry to little-endian bytes for a 32-bit ELF
    /// (8 bytes: 4-byte tag + 4-byte value).
    pub fn to_bytes_32_le(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&(self.tag as i32).to_le_bytes());
        buf[4..8].copy_from_slice(&(self.value as u32).to_le_bytes());
        buf
    }
}

// ===========================================================================
// DynamicRelocation — entry in .rela.dyn or .rela.plt
// ===========================================================================

/// Represents a single RELA relocation that the dynamic linker must process
/// at load time. Used for both `.rela.dyn` (data relocations) and
/// `.rela.plt` (PLT/function relocations).
#[derive(Debug, Clone)]
pub struct DynamicRelocation {
    /// Virtual address where the relocation is applied (r_offset).
    pub offset: u64,
    /// Architecture-specific relocation type (r_type).
    pub reloc_type: u32,
    /// Index into `.dynsym` of the referenced symbol (r_sym).
    pub symbol_index: u32,
    /// Addend for the relocation computation (r_addend).
    pub addend: i64,
}

impl DynamicRelocation {
    /// Serializes as Elf64_Rela (24 bytes, little-endian).
    /// Layout: r_offset (8) | r_info (8 = sym<<32 | type) | r_addend (8).
    pub fn to_bytes_64_le(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        let r_info: u64 = ((self.symbol_index as u64) << 32) | (self.reloc_type as u64);
        buf[8..16].copy_from_slice(&r_info.to_le_bytes());
        buf[16..24].copy_from_slice(&self.addend.to_le_bytes());
        buf
    }

    /// Serializes as Elf32_Rela (12 bytes, little-endian).
    /// Layout: r_offset (4) | r_info (4 = sym<<8 | type) | r_addend (4).
    pub fn to_bytes_32_le(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&(self.offset as u32).to_le_bytes());
        let r_info: u32 = (self.symbol_index << 8) | (self.reloc_type & 0xFF);
        buf[4..8].copy_from_slice(&r_info.to_le_bytes());
        buf[8..12].copy_from_slice(&(self.addend as i32).to_le_bytes());
        buf
    }
}

// ===========================================================================
// DynSymEntry — processed dynamic symbol for .dynsym
// ===========================================================================

/// A symbol destined for the `.dynsym` dynamic symbol table. Produced from
/// the linker's resolved [`SymbolEntry`] after filtering for appropriate
/// binding (Global/Weak) and visibility (Default/Protected).
#[derive(Debug, Clone)]
pub struct DynSymEntry {
    /// Symbol name (also stored in `.dynstr`).
    pub name: String,
    /// Symbol value — typically the virtual address of the symbol definition.
    pub value: u64,
    /// Symbol size in bytes (0 if unknown).
    pub size: u64,
    /// Symbol binding: Global or Weak (Local is excluded from `.dynsym`).
    pub binding: SymbolBinding,
    /// Symbol type: Func, Object, NoType, etc.
    pub sym_type: SymbolType,
    /// Symbol visibility: Default or Protected (Hidden excluded).
    pub visibility: SymbolVisibility,
    /// Section index where the symbol is defined, or 0 (SHN_UNDEF) if
    /// imported from another shared object.
    pub section_index: u16,
}

impl DynSymEntry {
    /// Packs the ELF `st_info` field: `(binding << 4) | type`.
    fn elf_st_info(&self) -> u8 {
        let bind = match self.binding {
            SymbolBinding::Local => 0,
            SymbolBinding::Global => 1,
            SymbolBinding::Weak => 2,
        };
        let stype = match self.sym_type {
            SymbolType::NoType => 0,
            SymbolType::Object => 1,
            SymbolType::Func => 2,
            SymbolType::Section => 3,
            SymbolType::File => 4,
        };
        (bind << 4) | stype
    }

    /// Packs the ELF `st_other` field (visibility in low 2 bits).
    fn elf_st_other(&self) -> u8 {
        match self.visibility {
            SymbolVisibility::Default => 0,
            SymbolVisibility::Hidden => 2,
            SymbolVisibility::Protected => 3,
        }
    }

    /// Serializes as Elf64_Sym (24 bytes, little-endian).
    /// Order: st_name(4), st_info(1), st_other(1), st_shndx(2),
    ///        st_value(8), st_size(8).
    pub fn to_bytes_64_le(&self, name_offset: u32) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&name_offset.to_le_bytes());
        buf[4] = self.elf_st_info();
        buf[5] = self.elf_st_other();
        buf[6..8].copy_from_slice(&self.section_index.to_le_bytes());
        buf[8..16].copy_from_slice(&self.value.to_le_bytes());
        buf[16..24].copy_from_slice(&self.size.to_le_bytes());
        buf
    }

    /// Serializes as Elf32_Sym (16 bytes, little-endian).
    /// Order: st_name(4), st_value(4), st_size(4),
    ///        st_info(1), st_other(1), st_shndx(2).
    pub fn to_bytes_32_le(&self, name_offset: u32) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&name_offset.to_le_bytes());
        buf[4..8].copy_from_slice(&(self.value as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&(self.size as u32).to_le_bytes());
        buf[12] = self.elf_st_info();
        buf[13] = self.elf_st_other();
        buf[14..16].copy_from_slice(&self.section_index.to_le_bytes());
        buf
    }
}

// ===========================================================================
// GotEntry — single .got / .got.plt entry
// ===========================================================================

/// Represents a single slot in the Global Offset Table. Each entry holds
/// an address that the dynamic linker patches at load time (or lazily for
/// `.got.plt` function entries).
#[derive(Debug, Clone)]
pub struct GotEntry {
    /// Name of the associated symbol (for diagnostics and relocation lookup).
    pub symbol_name: String,
    /// Byte offset of this entry within the GOT section.
    pub offset: u64,
    /// Initial value written into the entry before dynamic relocation.
    /// For `.got.plt` function slots this is the address of the PLT push
    /// instruction (enabling lazy resolution); for data slots it is 0.
    pub initial_value: u64,
}

// ===========================================================================
// PltEntry — single .plt stub descriptor
// ===========================================================================

/// Describes one function's PLT (Procedure Linkage Table) stub. The stub
/// performs an indirect jump through the corresponding `.got.plt` slot.
#[derive(Debug, Clone)]
pub struct PltEntry {
    /// Symbol name of the function being called through this PLT entry.
    pub symbol_name: String,
    /// Byte offset of the corresponding `.got.plt` entry (from the start of
    /// `.got.plt`).
    pub got_offset: u64,
    /// Zero-based index of this entry in the PLT (PLT[0] is the resolver
    /// stub; user entries start at index 1).
    pub plt_index: u32,
}

// ===========================================================================
// DynamicLayout — coordinating addresses for cross-section references
// ===========================================================================

/// Holds the resolved virtual addresses of all dynamic linking sections.
/// Passed to [`DynamicSectionBuilder::build`] so that `.dynamic` entries
/// can reference the correct addresses of `.dynsym`, `.dynstr`, `.gnu.hash`,
/// `.got`, `.got.plt`, `.plt`, `.rela.dyn`, and `.rela.plt`.
///
/// The linker must compute these addresses during section layout before
/// building the `.dynamic` section.
#[derive(Debug, Clone, Default)]
pub struct DynamicLayout {
    /// Virtual address of the `.dynamic` section itself.
    pub dynamic_addr: u64,
    /// Virtual address of the `.dynsym` section.
    pub dynsym_addr: u64,
    /// Virtual address of the `.dynstr` section.
    pub dynstr_addr: u64,
    /// Virtual address of the `.gnu.hash` section.
    pub gnu_hash_addr: u64,
    /// Virtual address of the `.got` section.
    pub got_addr: u64,
    /// Virtual address of the `.got.plt` section.
    pub got_plt_addr: u64,
    /// Virtual address of the `.plt` section.
    pub plt_addr: u64,
    /// Virtual address of the `.rela.dyn` section.
    pub rela_dyn_addr: u64,
    /// Virtual address of the `.rela.plt` section.
    pub rela_plt_addr: u64,
    /// Virtual address of the `.interp` section.
    pub interp_addr: u64,
}

// ===========================================================================
// GNU Hash function
// ===========================================================================

/// Computes the GNU hash of a symbol name. This is the standard ELF GNU
/// hash function used by `.gnu.hash` for fast dynamic symbol lookup.
///
/// Algorithm: `h = 5381; for each byte: h = h * 33 + byte; return h`.
pub fn gnu_hash(name: &str) -> u32 {
    let mut h: u32 = 5381;
    for byte in name.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u32);
    }
    h
}

// ===========================================================================
// DynamicSymbolTable — builds .dynsym and .dynstr
// ===========================================================================

/// Builds the `.dynsym` (dynamic symbol table) and `.dynstr` (dynamic string
/// table) sections. Also capable of building the `.gnu.hash` section from the
/// collected symbols.
///
/// # Usage
/// ```ignore
/// let mut dsym = DynamicSymbolTable::new();
/// for sym in resolved_symbols {
///     dsym.add_symbol(&sym);
/// }
/// let dynsym_bytes = dsym.build_dynsym();
/// let dynstr_bytes = dsym.build_dynstr();
/// let gnu_hash_bytes = dsym.build_gnu_hash();
/// ```
pub struct DynamicSymbolTable {
    /// Processed dynamic symbols (index 0 is always the null symbol).
    symbols: Vec<DynSymEntry>,
    /// Raw `.dynstr` string table bytes. Begins with a NUL byte (index 0 is
    /// the empty string) and contains NUL-terminated symbol names.
    string_table: Vec<u8>,
    /// Maps symbol name → byte offset within `string_table` for O(1)
    /// deduplication during string table construction.
    string_offsets: FxHashMap<String, u32>,
    /// Whether the target is 32-bit (i686) or 64-bit.
    is_32bit: bool,
}

impl DynamicSymbolTable {
    /// Creates a new, empty dynamic symbol table.
    /// The table is pre-populated with a null symbol at index 0 (ELF
    /// requirement) and a NUL byte at string-table offset 0.
    pub fn new() -> Self {
        let null_sym = DynSymEntry {
            name: String::new(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Local,
            sym_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: 0,
        };
        let mut string_table = Vec::with_capacity(256);
        string_table.push(0u8); // NUL byte at offset 0
        let mut string_offsets = fx_hash_map();
        string_offsets.insert(String::new(), 0);

        Self {
            symbols: vec![null_sym],
            string_table,
            string_offsets,
            is_32bit: false,
        }
    }

    /// Sets whether this table targets a 32-bit ELF (i686). Defaults to
    /// 64-bit. Must be called before `build_dynsym`.
    pub fn set_32bit(&mut self, is_32bit: bool) {
        self.is_32bit = is_32bit;
    }

    /// Adds a resolved symbol to the dynamic symbol table. Only symbols with
    /// Global or Weak binding **and** Default or Protected visibility are
    /// included. Local binding and Hidden visibility symbols are silently
    /// filtered out (they do not belong in `.dynsym`).
    pub fn add_symbol(&mut self, sym: &SymbolEntry) {
        // Filter: only Global or Weak binding symbols qualify for .dynsym.
        match sym.binding {
            SymbolBinding::Global | SymbolBinding::Weak => {}
            SymbolBinding::Local => return,
        }
        // Filter: Hidden symbols are excluded from the dynamic symbol table.
        if matches!(sym.visibility, SymbolVisibility::Hidden) {
            return;
        }

        // Intern the symbol name in .dynstr.
        let name_offset = self.intern_string(&sym.name);
        let _ = name_offset; // offset is recorded in string_offsets

        // Determine section index: 0 (SHN_UNDEF) for undefined imports,
        // otherwise preserve the section index from the resolved symbol.
        let section_index = if sym.is_defined {
            sym.section_index
        } else {
            0 // SHN_UNDEF
        };

        self.symbols.push(DynSymEntry {
            name: sym.name.clone(),
            value: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_index,
        });
    }

    /// Returns the number of symbols in the table (including the null symbol
    /// at index 0).
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    /// Returns a reference to the internal symbol list.
    pub fn symbols(&self) -> &[DynSymEntry] {
        &self.symbols
    }

    /// Interns a string in the `.dynstr` table and returns its byte offset.
    /// Duplicate strings are deduplicated via `FxHashMap`.
    fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.string_offsets.get(s) {
            return offset;
        }
        let offset = self.string_table.len() as u32;
        self.string_table.extend_from_slice(s.as_bytes());
        self.string_table.push(0u8); // NUL terminator
        self.string_offsets.insert(s.to_owned(), offset);
        offset
    }

    /// Returns the `.dynstr` byte offset for a given name. Panics if the
    /// name was never interned (should not happen for symbols already added).
    fn name_offset(&self, name: &str) -> u32 {
        *self
            .string_offsets
            .get(name)
            .unwrap_or_else(|| panic!("dynamic: symbol '{}' not found in .dynstr", name))
    }

    /// Serializes the `.dynsym` section as a contiguous byte vector.
    /// Each symbol is encoded as Elf64_Sym (24 bytes) or Elf32_Sym (16 bytes).
    pub fn build_dynsym(&self) -> Vec<u8> {
        let entry_size = if self.is_32bit { 16usize } else { 24usize };
        let mut out = Vec::with_capacity(self.symbols.len() * entry_size);
        for sym in &self.symbols {
            let name_off = self.name_offset(&sym.name);
            if self.is_32bit {
                out.extend_from_slice(&sym.to_bytes_32_le(name_off));
            } else {
                out.extend_from_slice(&sym.to_bytes_64_le(name_off));
            }
        }
        out
    }

    /// Returns a clone of the `.dynstr` string table bytes (NUL-terminated
    /// concatenation of all interned symbol names).
    pub fn build_dynstr(&self) -> Vec<u8> {
        self.string_table.clone()
    }

    /// Builds the `.gnu.hash` section from the current symbol list.
    /// Delegates to the standalone [`build_gnu_hash`] function.
    pub fn build_gnu_hash(&self) -> Vec<u8> {
        build_gnu_hash(&self.symbols, &self.string_table)
    }
}

impl Default for DynamicSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// build_gnu_hash — standalone GNU hash table builder
// ===========================================================================

/// Builds a `.gnu.hash` section from a slice of dynamic symbols and the
/// associated `.dynstr` data.
///
/// The GNU hash table layout is:
/// ```text
/// Header:     nbuckets(4)  symoffset(4)  bloom_size(4)  bloom_shift(4)
/// Bloom:      bloom_size × 8-byte words (Bloom filter for fast rejection)
/// Buckets:    nbuckets × 4-byte indices (first symbol index per bucket)
/// Chains:     (nsyms - symoffset) × 4-byte hash values (chain terminators)
/// ```
///
/// Symbol index 0 (null) is never hashed. Symbols whose index is less than
/// `symoffset` (typically 1) are skipped. The caller must ensure that the
/// symbols slice is consistent with the `.dynsym` ordering.
pub fn build_gnu_hash(symbols: &[DynSymEntry], _dynstr: &[u8]) -> Vec<u8> {
    // Collect hashable symbols (skip index 0 = null symbol).
    // Each tuple: (original index in .dynsym, hash value).
    let hashable: Vec<(usize, u32)> = symbols
        .iter()
        .enumerate()
        .skip(1) // skip null symbol at index 0
        .filter(|(_, s)| !s.name.is_empty())
        .map(|(idx, s)| (idx, gnu_hash(&s.name)))
        .collect();

    let nsyms = hashable.len();
    // symoffset: the index of the first hashed symbol in .dynsym.
    // Typically 1 (the null symbol at 0 is never hashed).
    let symoffset: u32 = if hashable.is_empty() {
        1
    } else {
        hashable[0].0 as u32
    };

    // Choose bucket count: at least 1, roughly nsyms/1 for small tables,
    // capped to next power of two for alignment efficiency.
    let nbuckets: u32 = if nsyms == 0 {
        1
    } else {
        (nsyms as u32).next_power_of_two().max(1)
    };

    // Bloom filter: 1 word per 8 symbols (minimum 1 word). 64-bit words.
    let bloom_size: u32 = if nsyms == 0 {
        1
    } else {
        ((nsyms + 7) / 8).next_power_of_two().max(1) as u32
    };
    let bloom_shift: u32 = 6; // standard shift for 64-bit bloom words

    // Allocate bloom filter, buckets, and chain arrays.
    let mut bloom: Vec<u64> = vec![0u64; bloom_size as usize];
    let mut buckets: Vec<u32> = vec![0u32; nbuckets as usize];
    // Chain: one entry per hashed symbol (from symoffset to last).
    let chain_len = if nsyms == 0 { 0 } else { nsyms };
    let mut chains: Vec<u32> = vec![0u32; chain_len];

    // Sort hashable symbols by bucket to produce contiguous chains.
    // Collect: (bucket_index, hash, original_dynsym_index)
    let mut sorted: Vec<(u32, u32, usize)> = hashable
        .iter()
        .map(|&(idx, h)| (h % nbuckets, h, idx))
        .collect();
    sorted.sort_by_key(|&(bucket, _, idx)| (bucket, idx));

    // Fill bloom filter.
    for &(_, h, _) in &sorted {
        let word_idx = ((h / 64) as usize) % bloom.len();
        let bit0 = h % 64;
        let bit1 = (h >> bloom_shift) % 64;
        bloom[word_idx] |= 1u64 << bit0;
        bloom[word_idx] |= 1u64 << bit1;
    }

    // Fill buckets: for each bucket, record the chain index of its first symbol.
    // Chains use indices relative to symoffset.
    let mut last_bucket: Option<u32> = None;
    for (chain_idx, &(bucket, _, _)) in sorted.iter().enumerate() {
        if last_bucket != Some(bucket) {
            buckets[bucket as usize] = symoffset + chain_idx as u32;
            last_bucket = Some(bucket);
        }
    }

    // Fill chains: each chain entry is `hash & ~1`, with the LSB set on the
    // last element of each bucket's chain (end-of-chain marker).
    for (chain_idx, &(bucket, h, _)) in sorted.iter().enumerate() {
        let mut chain_val = h & !1u32; // clear LSB
                                       // Check if this is the last entry in its bucket.
        let is_last = chain_idx + 1 >= sorted.len() || sorted[chain_idx + 1].0 != bucket;
        if is_last {
            chain_val |= 1; // set end-of-chain bit
        }
        chains[chain_idx] = chain_val;
    }

    // Serialize the .gnu.hash section.
    let header_size = 4 * 4; // nbuckets, symoffset, bloom_size, bloom_shift
    let total_size =
        header_size + (bloom_size as usize * 8) + (nbuckets as usize * 4) + (chain_len * 4);
    let mut out = Vec::with_capacity(total_size);

    // Header
    out.extend_from_slice(&nbuckets.to_le_bytes());
    out.extend_from_slice(&symoffset.to_le_bytes());
    out.extend_from_slice(&bloom_size.to_le_bytes());
    out.extend_from_slice(&bloom_shift.to_le_bytes());

    // Bloom filter words
    for &word in &bloom {
        out.extend_from_slice(&word.to_le_bytes());
    }

    // Buckets
    for &b in &buckets {
        out.extend_from_slice(&b.to_le_bytes());
    }

    // Chains
    for &c in &chains {
        out.extend_from_slice(&c.to_le_bytes());
    }

    out
}

// ===========================================================================
// DynamicSectionBuilder — builds the .dynamic section
// ===========================================================================

/// Constructs the `.dynamic` section — an array of `DynamicEntry` tag/value
/// pairs that the runtime dynamic linker reads to locate `.dynsym`,
/// `.dynstr`, `.gnu.hash`, `.rela.dyn`, `.rela.plt`, `.got.plt`, etc.
///
/// # Usage
/// ```ignore
/// let mut dsb = DynamicSectionBuilder::new();
/// dsb.add_needed("libc.so.6");
/// dsb.set_soname("libfoo.so.1");
/// let bytes = dsb.build(&layout);
/// ```
pub struct DynamicSectionBuilder {
    /// Accumulated dynamic entries (pre-build).
    entries: Vec<DynamicEntry>,
    /// Library names required at runtime (DT_NEEDED strings).
    needed_libs: Vec<String>,
    /// The SONAME of this shared object, if set.
    soname: Option<String>,
    /// Address of the `.init` function, if any.
    init_addr: Option<u64>,
    /// Address of the `.fini` function, if any.
    fini_addr: Option<u64>,
    /// Cached sizes of sections needed for DT_*SZ entries.
    dynstr_size: u64,
    rela_dyn_size: u64,
    rela_plt_size: u64,
    /// Whether to emit 32-bit entries (i686) instead of 64-bit.
    is_32bit: bool,
}

impl DynamicSectionBuilder {
    /// Creates a new, empty `.dynamic` section builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(32),
            needed_libs: Vec::new(),
            soname: None,
            init_addr: None,
            fini_addr: None,
            dynstr_size: 0,
            rela_dyn_size: 0,
            rela_plt_size: 0,
            is_32bit: false,
        }
    }

    /// Adds a `DT_NEEDED` entry for a required shared library. The `lib`
    /// string is the library's SONAME (e.g., `"libc.so.6"`).
    pub fn add_needed(&mut self, lib: &str) {
        self.needed_libs.push(lib.to_owned());
    }

    /// Sets the SONAME of the shared object being linked (DT_SONAME).
    pub fn set_soname(&mut self, name: &str) {
        self.soname = Some(name.to_owned());
    }

    /// Sets the `.init` function address (DT_INIT).
    pub fn set_init(&mut self, addr: u64) {
        self.init_addr = Some(addr);
    }

    /// Sets the `.fini` function address (DT_FINI).
    pub fn set_fini(&mut self, addr: u64) {
        self.fini_addr = Some(addr);
    }

    /// Records the size of `.dynstr` for DT_STRSZ.
    pub fn set_dynstr_size(&mut self, size: u64) {
        self.dynstr_size = size;
    }

    /// Records the size of `.rela.dyn` for DT_RELASZ.
    pub fn set_rela_dyn_size(&mut self, size: u64) {
        self.rela_dyn_size = size;
    }

    /// Records the size of `.rela.plt` for DT_PLTRELSZ.
    pub fn set_rela_plt_size(&mut self, size: u64) {
        self.rela_plt_size = size;
    }

    /// Sets whether to emit 32-bit entries for i686 targets.
    pub fn set_32bit(&mut self, is_32bit: bool) {
        self.is_32bit = is_32bit;
    }

    /// Builds and serializes the complete `.dynamic` section.
    ///
    /// Uses `layout` to reference the virtual addresses of all dynamic linking
    /// sections. Returns the section bytes ready for ELF embedding.
    ///
    /// The produced `.dynamic` array contains entries in this order:
    /// 1. DT_NEEDED (one per required library)
    /// 2. DT_SONAME (if set)
    /// 3. DT_INIT / DT_FINI (if set)
    /// 4. DT_GNU_HASH / DT_STRTAB / DT_SYMTAB / DT_STRSZ / DT_SYMENT
    /// 5. DT_PLTGOT / DT_PLTRELSZ / DT_PLTREL / DT_JMPREL
    /// 6. DT_RELA / DT_RELASZ / DT_RELAENT
    /// 7. DT_NULL (terminator)
    pub fn build(&mut self, layout: &DynamicLayout) -> Vec<u8> {
        self.entries.clear();

        // Build a temporary .dynstr-like table for DT_NEEDED/DT_SONAME
        // string offsets. In practice the linker has already built .dynstr;
        // here we compute offsets into that table.
        let mut dynstr_map: FxHashMap<String, u64> = fx_hash_map();
        let mut offset: u64 = 1; // offset 0 = empty string (NUL byte)
        for lib in &self.needed_libs {
            if !dynstr_map.contains_key(lib) {
                dynstr_map.insert(lib.clone(), offset);
                offset += lib.len() as u64 + 1;
            }
        }
        if let Some(ref soname) = self.soname {
            if !dynstr_map.contains_key(soname) {
                dynstr_map.insert(soname.clone(), offset);
            }
        }

        // 1. DT_NEEDED entries.
        for lib in &self.needed_libs {
            let str_offset = dynstr_map.get(lib).copied().unwrap_or(0);
            self.entries.push(DynamicEntry::new(DT_NEEDED, str_offset));
        }

        // 2. DT_SONAME.
        if let Some(ref soname) = self.soname {
            let str_offset = dynstr_map.get(soname).copied().unwrap_or(0);
            self.entries.push(DynamicEntry::new(DT_SONAME, str_offset));
        }

        // 3. DT_INIT / DT_FINI.
        if let Some(addr) = self.init_addr {
            self.entries.push(DynamicEntry::new(DT_INIT, addr));
        }
        if let Some(addr) = self.fini_addr {
            self.entries.push(DynamicEntry::new(DT_FINI, addr));
        }

        // 4. Symbol lookup structures.
        self.entries
            .push(DynamicEntry::new(DT_GNU_HASH, layout.gnu_hash_addr));
        self.entries
            .push(DynamicEntry::new(DT_STRTAB, layout.dynstr_addr));
        self.entries
            .push(DynamicEntry::new(DT_SYMTAB, layout.dynsym_addr));
        self.entries
            .push(DynamicEntry::new(DT_STRSZ, self.dynstr_size));
        let syment = if self.is_32bit {
            ELF32_SYM_SIZE
        } else {
            ELF64_SYM_SIZE
        };
        self.entries.push(DynamicEntry::new(DT_SYMENT, syment));

        // 5. PLT/GOT entries.
        self.entries
            .push(DynamicEntry::new(DT_PLTGOT, layout.got_plt_addr));
        if self.rela_plt_size > 0 {
            self.entries
                .push(DynamicEntry::new(DT_PLTRELSZ, self.rela_plt_size));
            // DT_PLTREL value 7 = DT_RELA (we use RELA format for all arches).
            self.entries
                .push(DynamicEntry::new(DT_PLTREL, DT_RELA as u64));
            self.entries
                .push(DynamicEntry::new(DT_JMPREL, layout.rela_plt_addr));
        }

        // 6. RELA relocations (non-PLT).
        if self.rela_dyn_size > 0 {
            self.entries
                .push(DynamicEntry::new(DT_RELA, layout.rela_dyn_addr));
            self.entries
                .push(DynamicEntry::new(DT_RELASZ, self.rela_dyn_size));
            let relaent = if self.is_32bit {
                ELF32_RELA_SIZE
            } else {
                ELF64_RELA_SIZE
            };
            self.entries.push(DynamicEntry::new(DT_RELAENT, relaent));
        }

        // 7. Terminator.
        self.entries.push(DynamicEntry::new(DT_NULL, 0));

        // Serialize all entries.
        let entry_size = if self.is_32bit { 8 } else { 16 };
        let mut out = Vec::with_capacity(self.entries.len() * entry_size);
        for entry in &self.entries {
            if self.is_32bit {
                out.extend_from_slice(&entry.to_bytes_32_le());
            } else {
                out.extend_from_slice(&entry.to_bytes_64_le());
            }
        }
        out
    }
}

impl Default for DynamicSectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// GotBuilder — builds .got and .got.plt sections
// ===========================================================================

/// Builds the Global Offset Table (`.got`) for PIC data addressing and the
/// `.got.plt` section for lazy function binding through the PLT.
///
/// **`.got`** contains entries for data symbols that need PIC-relative
/// addressing. Each entry is patched by the dynamic linker with the actual
/// runtime address of the symbol.
///
/// **`.got.plt`** contains:
/// - `GOT[0]` = address of `.dynamic` section (convention).
/// - `GOT[1]` = reserved for `link_map` pointer (filled by `ld.so`).
/// - `GOT[2]` = reserved for `_dl_runtime_resolve` (filled by `ld.so`).
/// - `GOT[3+]` = one entry per PLT function, initially pointing back to the
///   PLT push instruction for lazy resolution.
pub struct GotBuilder {
    /// Data GOT entries (for `.got` section).
    entries: Vec<GotEntry>,
    /// PLT GOT entries (for `.got.plt` section, excluding the 3 reserved).
    plt_entries: Vec<GotEntry>,
    /// Base virtual address of the `.got` section.
    base_address: u64,
    /// Base virtual address of the `.got.plt` section.
    got_plt_address: u64,
    /// Address of the `.dynamic` section (written into GOT.PLT[0]).
    dynamic_addr: u64,
    /// Pointer width in bytes (4 for i686, 8 for 64-bit targets).
    ptr_size: usize,
}

impl GotBuilder {
    /// Creates a new GOT builder.
    ///
    /// - `got_address`: virtual address assigned to `.got`.
    /// - `got_plt_address`: virtual address assigned to `.got.plt`.
    /// - `dynamic_addr`: virtual address of `.dynamic` (goes into GOT.PLT[0]).
    /// - `target`: determines pointer width (4 or 8 bytes).
    pub fn new(got_address: u64, got_plt_address: u64, dynamic_addr: u64, target: &Target) -> Self {
        Self {
            entries: Vec::new(),
            plt_entries: Vec::new(),
            base_address: got_address,
            got_plt_address,
            dynamic_addr,
            ptr_size: target.pointer_width() as usize,
        }
    }

    /// Adds a data GOT entry (goes into `.got`). Returns the byte offset of
    /// this entry within `.got`.
    pub fn add_entry(&mut self, entry: GotEntry) -> u64 {
        let offset = (self.entries.len() * self.ptr_size) as u64;
        self.entries.push(entry);
        offset
    }

    /// Adds a PLT GOT entry (goes into `.got.plt`). The first three slots
    /// are reserved and managed internally; this adds to slot 3+.
    /// Returns the virtual address of the new `.got.plt` entry.
    pub fn add_plt_entry(&mut self, entry: GotEntry) -> u64 {
        let slot_index = 3 + self.plt_entries.len();
        let addr = self.got_plt_address + (slot_index * self.ptr_size) as u64;
        self.plt_entries.push(entry);
        addr
    }

    /// Returns the base virtual address of `.got`.
    pub fn got_address(&self) -> u64 {
        self.base_address
    }

    /// Returns the base virtual address of `.got.plt`.
    pub fn got_plt_address(&self) -> u64 {
        self.got_plt_address
    }

    /// Returns the number of data GOT entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total size of `.got` in bytes.
    pub fn got_size(&self) -> usize {
        self.entries.len() * self.ptr_size
    }

    /// Returns the total size of `.got.plt` in bytes (3 reserved + N PLT entries).
    pub fn got_plt_size(&self) -> usize {
        (3 + self.plt_entries.len()) * self.ptr_size
    }

    /// Serializes the `.got` section. Each entry is `ptr_size` bytes
    /// containing the initial value (typically 0, patched at load time).
    pub fn build_got(&self) -> Vec<u8> {
        let total = self.entries.len() * self.ptr_size;
        let mut out = Vec::with_capacity(total);
        for entry in &self.entries {
            if self.ptr_size == 4 {
                out.extend_from_slice(&(entry.initial_value as u32).to_le_bytes());
            } else {
                out.extend_from_slice(&entry.initial_value.to_le_bytes());
            }
        }
        out
    }

    /// Serializes the `.got.plt` section. Layout:
    /// - Slot 0: address of `.dynamic`
    /// - Slot 1: 0 (reserved for `link_map`)
    /// - Slot 2: 0 (reserved for `_dl_runtime_resolve`)
    /// - Slot 3+: one per PLT function (initial value = PLT push address)
    pub fn build_got_plt(&self) -> Vec<u8> {
        let total = (3 + self.plt_entries.len()) * self.ptr_size;
        let mut out = Vec::with_capacity(total);

        if self.ptr_size == 4 {
            // Slot 0: .dynamic address
            out.extend_from_slice(&(self.dynamic_addr as u32).to_le_bytes());
            // Slot 1: reserved (link_map)
            out.extend_from_slice(&0u32.to_le_bytes());
            // Slot 2: reserved (_dl_runtime_resolve)
            out.extend_from_slice(&0u32.to_le_bytes());
            // PLT function entries
            for entry in &self.plt_entries {
                out.extend_from_slice(&(entry.initial_value as u32).to_le_bytes());
            }
        } else {
            // Slot 0: .dynamic address
            out.extend_from_slice(&self.dynamic_addr.to_le_bytes());
            // Slot 1: reserved (link_map)
            out.extend_from_slice(&0u64.to_le_bytes());
            // Slot 2: reserved (_dl_runtime_resolve)
            out.extend_from_slice(&0u64.to_le_bytes());
            // PLT function entries
            for entry in &self.plt_entries {
                out.extend_from_slice(&entry.initial_value.to_le_bytes());
            }
        }
        out
    }
}

// ===========================================================================
// PltBuilder — builds the .plt section with architecture-specific stubs
// ===========================================================================

/// Builds the Procedure Linkage Table (`.plt`) — a set of small code stubs
/// that indirect function calls through `.got.plt` for lazy binding.
///
/// Each PLT entry is architecture-specific machine code:
/// - **x86-64**: 16-byte stubs using RIP-relative addressing.
/// - **i686**: 16-byte stubs using absolute addressing.
/// - **AArch64**: 16-byte stubs using ADRP/LDR/BR sequences.
/// - **RISC-V 64**: 16-byte stubs using AUIPC/LD/JR sequences.
///
/// PLT[0] is a special resolver stub that pushes `link_map` and jumps to
/// `_dl_runtime_resolve`. PLT[1..] are per-function stubs.
pub struct PltBuilder {
    /// Per-function PLT entries (not including PLT[0]).
    entries: Vec<PltEntry>,
    /// Base virtual address of the `.plt` section.
    plt_address: u64,
    /// Base virtual address of the `.got.plt` section.
    got_plt_address: u64,
    /// Target architecture (determines stub machine code).
    target: Target,
}

/// Size of each PLT entry in bytes (consistent across architectures).
const PLT_ENTRY_SIZE: u64 = 16;

/// Size of the PLT[0] resolver stub in bytes.
/// x86-64/i686 use 16 bytes; AArch64/RISC-V use 32 bytes.
fn plt0_size(target: &Target) -> u64 {
    match target {
        Target::X86_64 | Target::I686 => 16,
        Target::AArch64 | Target::RiscV64 => 32,
    }
}

impl PltBuilder {
    /// Creates a new PLT builder.
    ///
    /// - `plt_address`: virtual address where `.plt` is mapped.
    /// - `got_plt_address`: virtual address of `.got.plt`.
    /// - `target`: determines architecture-specific stub code.
    pub fn new(plt_address: u64, got_plt_address: u64, target: Target) -> Self {
        Self {
            entries: Vec::new(),
            plt_address,
            got_plt_address,
            target,
        }
    }

    /// Adds a function entry to the PLT. Returns the virtual address of the
    /// new PLT stub (callers use this address to resolve function calls).
    pub fn add_entry(&mut self, entry: PltEntry) -> u64 {
        let idx = self.entries.len();
        let addr = self.plt_address + plt0_size(&self.target) + (idx as u64) * PLT_ENTRY_SIZE;
        self.entries.push(entry);
        addr
    }

    /// Returns the base virtual address of `.plt`.
    pub fn plt_address(&self) -> u64 {
        self.plt_address
    }

    /// Returns the total size of the `.plt` section in bytes.
    pub fn plt_size(&self) -> u64 {
        plt0_size(&self.target) + (self.entries.len() as u64) * PLT_ENTRY_SIZE
    }

    /// Returns the number of per-function PLT entries (excluding PLT[0]).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Serializes the `.plt` section with architecture-specific stub code.
    ///
    /// Generates PLT[0] (resolver) and PLT[1..N] (per-function stubs) using
    /// machine code appropriate for the target architecture.
    pub fn build_plt(&self, target: &Target) -> Vec<u8> {
        match target {
            Target::X86_64 => self.build_plt_x86_64(),
            Target::I686 => self.build_plt_i686(),
            Target::AArch64 => self.build_plt_aarch64(),
            Target::RiscV64 => self.build_plt_riscv64(),
        }
    }

    // -----------------------------------------------------------------------
    // x86-64 PLT stubs (16 bytes each)
    // -----------------------------------------------------------------------

    /// Generates x86-64 PLT code.
    ///
    /// PLT[0] (16 bytes):
    /// ```text
    /// push   QWORD PTR [rip + (GOT+8 - PLT0 - 6)]  ; push &link_map
    /// jmp    QWORD PTR [rip + (GOT+16 - PLT0 - 12)] ; jmp dl_runtime_resolve
    /// nop DWORD PTR [rax]                            ; 4-byte NOP padding
    /// ```
    ///
    /// PLT[N] (16 bytes):
    /// ```text
    /// jmp    QWORD PTR [rip + (GOT_ENTRY - PLT_N - 6)] ; jmp *GOT entry
    /// push   RELOC_INDEX                                 ; push reloc index
    /// jmp    PLT[0]                                      ; jmp resolver
    /// ```
    fn build_plt_x86_64(&self) -> Vec<u8> {
        let plt0_addr = self.plt_address;
        let got_plt_base = self.got_plt_address;
        let ptr_size: u64 = 8;

        let mut out = Vec::with_capacity((1 + self.entries.len()) * 16);

        // PLT[0]: resolver stub
        {
            // ff 35 XX XX XX XX  : push [rip+disp32] ; GOT[1] = link_map
            // ff 25 XX XX XX XX  : jmp  [rip+disp32] ; GOT[2] = resolver
            // 0f 1f 40 00        : nop DWORD [rax+0] (4-byte NOP)
            let got1 = got_plt_base + ptr_size; // GOT[1]
            let got2 = got_plt_base + 2 * ptr_size; // GOT[2]

            // push [rip+disp32]: instruction at PLT0+0, length=6, so rip=PLT0+6
            let push_disp = (got1 as i64) - (plt0_addr as i64) - 6;
            // jmp [rip+disp32]: instruction at PLT0+6, length=6, so rip=PLT0+12
            let jmp_disp = (got2 as i64) - (plt0_addr as i64) - 12;

            out.extend_from_slice(&[0xff, 0x35]);
            out.extend_from_slice(&(push_disp as i32).to_le_bytes());
            out.extend_from_slice(&[0xff, 0x25]);
            out.extend_from_slice(&(jmp_disp as i32).to_le_bytes());
            out.extend_from_slice(&[0x0f, 0x1f, 0x40, 0x00]);
        }

        // PLT[1..N]: per-function stubs
        for (i, entry) in self.entries.iter().enumerate() {
            let plt_n_addr = plt0_addr + 16 + (i as u64) * 16;

            // GOT.PLT entry address for this function
            let got_entry_addr = got_plt_base + (3 + entry.plt_index as u64) * ptr_size;

            // jmp [rip+disp32]: instruction at PLT_N+0, length=6, rip=PLT_N+6
            let jmp_disp = (got_entry_addr as i64) - (plt_n_addr as i64) - 6;

            // push relocation index (index into .rela.plt)
            let reloc_idx = i as u32;

            // jmp PLT[0]: instruction at PLT_N+11, length=5, rip=PLT_N+16
            let jmp_plt0_disp = (plt0_addr as i64) - (plt_n_addr as i64) - 16;

            out.extend_from_slice(&[0xff, 0x25]);
            out.extend_from_slice(&(jmp_disp as i32).to_le_bytes());
            out.extend_from_slice(&[0x68]);
            out.extend_from_slice(&reloc_idx.to_le_bytes());
            out.extend_from_slice(&[0xe9]);
            out.extend_from_slice(&(jmp_plt0_disp as i32).to_le_bytes());
        }
        out
    }

    // -----------------------------------------------------------------------
    // i686 PLT stubs (16 bytes each)
    // -----------------------------------------------------------------------

    /// Generates i686 PLT code with absolute addressing.
    ///
    /// PLT[0] (16 bytes):
    /// ```text
    /// push   DWORD PTR [GOT+4]      ; push &link_map
    /// jmp    DWORD PTR [GOT+8]      ; jmp dl_runtime_resolve
    /// nop; nop; nop; nop             ; padding
    /// ```
    ///
    /// PLT[N] (16 bytes):
    /// ```text
    /// jmp    DWORD PTR [GOT_ENTRY]  ; jmp *GOT entry (absolute)
    /// push   RELOC_INDEX             ; push relocation index
    /// jmp    PLT[0]                  ; jmp resolver
    /// ```
    fn build_plt_i686(&self) -> Vec<u8> {
        let plt0_addr = self.plt_address;
        let got_plt_base = self.got_plt_address;
        let ptr_size: u64 = 4;

        let mut out = Vec::with_capacity((1 + self.entries.len()) * 16);

        // PLT[0]
        {
            let got1 = got_plt_base + ptr_size; // GOT[1]
            let got2 = got_plt_base + 2 * ptr_size; // GOT[2]

            // ff 35 XX XX XX XX : push dword ptr [abs32] GOT[1]
            out.extend_from_slice(&[0xff, 0x35]);
            out.extend_from_slice(&(got1 as u32).to_le_bytes());
            // ff 25 XX XX XX XX : jmp  dword ptr [abs32] GOT[2]
            out.extend_from_slice(&[0xff, 0x25]);
            out.extend_from_slice(&(got2 as u32).to_le_bytes());
            // 4-byte NOP padding
            out.extend_from_slice(&[0x90, 0x90, 0x90, 0x90]);
        }

        // PLT[1..N]
        for (i, entry) in self.entries.iter().enumerate() {
            let plt_n_addr = plt0_addr + 16 + (i as u64) * 16;
            let got_entry_addr = got_plt_base + (3 + entry.plt_index as u64) * ptr_size;
            let reloc_idx = i as u32;

            // jmp dword ptr [abs32]
            out.extend_from_slice(&[0xff, 0x25]);
            out.extend_from_slice(&(got_entry_addr as u32).to_le_bytes());
            // push reloc_index
            out.extend_from_slice(&[0x68]);
            out.extend_from_slice(&reloc_idx.to_le_bytes());
            // jmp PLT[0] (relative)
            let jmp_target = (plt0_addr as i64) - (plt_n_addr as i64) - 16;
            out.extend_from_slice(&[0xe9]);
            out.extend_from_slice(&(jmp_target as i32).to_le_bytes());
        }
        out
    }

    // -----------------------------------------------------------------------
    // AArch64 PLT stubs
    // -----------------------------------------------------------------------

    /// Generates AArch64 PLT code using ADRP/LDR/ADD/BR sequences.
    ///
    /// PLT[0] (32 bytes):
    /// ```text
    /// stp  x16, x30, [sp, #-16]!
    /// adrp x16, PAGE(GOT+16)
    /// ldr  x17, [x16, #PAGEOFF(GOT+16)]
    /// add  x16, x16, #PAGEOFF(GOT+8)
    /// br   x17
    /// nop
    /// nop
    /// nop
    /// ```
    ///
    /// PLT[N] (16 bytes):
    /// ```text
    /// adrp x16, PAGE(GOT_ENTRY)
    /// ldr  x17, [x16, #PAGEOFF(GOT_ENTRY)]
    /// add  x16, x16, #PAGEOFF(GOT_ENTRY)
    /// br   x17
    /// ```
    fn build_plt_aarch64(&self) -> Vec<u8> {
        let plt0_addr = self.plt_address;
        let got_plt_base = self.got_plt_address;
        let ptr_size: u64 = 8;

        let plt0_bytes = 32usize;
        let mut out = Vec::with_capacity(plt0_bytes + self.entries.len() * 16);

        // PLT[0] — resolver stub (32 bytes = 8 instructions)
        {
            let got1_addr = got_plt_base + ptr_size; // GOT[1] link_map
            let got2_addr = got_plt_base + 2 * ptr_size; // GOT[2] resolver

            // stp x16, x30, [sp, #-16]!
            out.extend_from_slice(&0xa9bf_7bf0u32.to_le_bytes());

            // adrp x16, PAGE(GOT+16) — page-relative offset from PLT0+4
            let pc_adrp = plt0_addr + 4;
            let page_diff = aarch64_page(got2_addr) as i64 - aarch64_page(pc_adrp) as i64;
            out.extend_from_slice(&encode_adrp(16, page_diff).to_le_bytes());

            // ldr x17, [x16, #PAGEOFF(GOT+16)]
            let pageoff_got2 = (got2_addr & 0xFFF) as u32;
            out.extend_from_slice(&encode_ldr_imm64(17, 16, pageoff_got2).to_le_bytes());

            // add x16, x16, #PAGEOFF(GOT+8)
            let pageoff_got1 = (got1_addr & 0xFFF) as u32;
            out.extend_from_slice(&encode_add_imm(16, 16, pageoff_got1).to_le_bytes());

            // br x17
            out.extend_from_slice(&0xd61f_0220u32.to_le_bytes());

            // 3× NOP padding to fill to 32 bytes
            out.extend_from_slice(&0xd503_201fu32.to_le_bytes()); // nop
            out.extend_from_slice(&0xd503_201fu32.to_le_bytes()); // nop
            out.extend_from_slice(&0xd503_201fu32.to_le_bytes()); // nop
        }

        // PLT[1..N] — per-function stubs (16 bytes = 4 instructions)
        for (i, entry) in self.entries.iter().enumerate() {
            let plt_n_addr = plt0_addr + plt0_bytes as u64 + (i as u64) * 16;
            let got_entry_addr = got_plt_base + (3 + entry.plt_index as u64) * ptr_size;

            // adrp x16, PAGE(GOT_ENTRY) — from PLT_N
            let page_diff = aarch64_page(got_entry_addr) as i64 - aarch64_page(plt_n_addr) as i64;
            out.extend_from_slice(&encode_adrp(16, page_diff).to_le_bytes());

            // ldr x17, [x16, #PAGEOFF(GOT_ENTRY)]
            let pageoff = (got_entry_addr & 0xFFF) as u32;
            out.extend_from_slice(&encode_ldr_imm64(17, 16, pageoff).to_le_bytes());

            // add x16, x16, #PAGEOFF(GOT_ENTRY)
            out.extend_from_slice(&encode_add_imm(16, 16, pageoff).to_le_bytes());

            // br x17
            out.extend_from_slice(&0xd61f_0220u32.to_le_bytes());
        }
        out
    }

    // -----------------------------------------------------------------------
    // RISC-V 64 PLT stubs
    // -----------------------------------------------------------------------

    /// Generates RISC-V 64 PLT code using AUIPC/LD/JALR sequences.
    ///
    /// PLT[0] (32 bytes):
    /// ```text
    /// auipc  t2, %pcrel_hi(GOT+8)     ; t2 = PC + hi20(GOT+8)
    /// sub    t1, t1, t3                ; (caller convention)
    /// ld     t3, %pcrel_lo(GOT+16)(t2) ; t3 = resolver entry
    /// addi   t1, t1, -(plt_header_size + 12)
    /// addi   t0, t2, %pcrel_lo(GOT+8)  ; t0 = &link_map
    /// srli   t1, t1, (log2(plt_entry_size)) ; reloc index
    /// ld     t0, 0(t0)                  ; t0 = link_map value
    /// jr     t3
    /// ```
    ///
    /// PLT[N] (16 bytes):
    /// ```text
    /// auipc  t3, %pcrel_hi(GOT_ENTRY)
    /// ld     t3, %pcrel_lo(GOT_ENTRY)(t3)
    /// jalr   t1, t3
    /// nop
    /// ```
    fn build_plt_riscv64(&self) -> Vec<u8> {
        let plt0_addr = self.plt_address;
        let got_plt_base = self.got_plt_address;
        let ptr_size: u64 = 8;

        let plt0_bytes = 32usize;
        let mut out = Vec::with_capacity(plt0_bytes + self.entries.len() * 16);

        // PLT[0] — resolver stub (32 bytes = 8 instructions × 4 bytes)
        {
            let got1_addr = got_plt_base + ptr_size; // GOT[1] link_map
            let got2_addr = got_plt_base + 2 * ptr_size; // GOT[2] resolver

            // auipc t2(x7), %pcrel_hi(GOT+8)
            let offset1 = got1_addr as i64 - plt0_addr as i64;
            let (hi20_1, lo12_1) = riscv_split_imm(offset1 as i32);
            out.extend_from_slice(&riscv_auipc(7, hi20_1 as u32).to_le_bytes());

            // sub t1(x6), t1, t3(x28) — convention for lazy binding index
            out.extend_from_slice(&riscv_sub(6, 6, 28).to_le_bytes());

            // ld t3(x28), lo12(GOT+16)(t2)
            let offset2 = got2_addr as i64 - plt0_addr as i64;
            let (_hi20_2, lo12_2) = riscv_split_imm(offset2 as i32);
            out.extend_from_slice(&riscv_ld(28, 7, lo12_2).to_le_bytes());

            // addi t1, t1, -(plt_header_size + 12)
            let neg_off = -((plt0_bytes as i32) + 12);
            out.extend_from_slice(&riscv_addi(6, 6, neg_off).to_le_bytes());

            // addi t0(x5), t2, lo12(GOT+8) — pointer to link_map
            out.extend_from_slice(&riscv_addi(5, 7, lo12_1).to_le_bytes());

            // srli t1, t1, 4 (log2(16) = 4, plt entry size)
            out.extend_from_slice(&riscv_srli(6, 6, 4).to_le_bytes());

            // ld t0, 0(t0) — dereference link_map pointer
            out.extend_from_slice(&riscv_ld(5, 5, 0).to_le_bytes());

            // jr t3 (jalr x0, t3, 0)
            out.extend_from_slice(&riscv_jalr(0, 28, 0).to_le_bytes());
        }

        // PLT[1..N] — per-function stubs (16 bytes = 4 instructions)
        for (i, entry) in self.entries.iter().enumerate() {
            let plt_n_addr = plt0_addr + plt0_bytes as u64 + (i as u64) * 16;
            let got_entry_addr = got_plt_base + (3 + entry.plt_index as u64) * ptr_size;

            let offset = got_entry_addr as i64 - plt_n_addr as i64;
            let (hi20, lo12) = riscv_split_imm(offset as i32);

            // auipc t3(x28), hi20
            out.extend_from_slice(&riscv_auipc(28, hi20 as u32).to_le_bytes());

            // ld t3, lo12(t3)
            out.extend_from_slice(&riscv_ld(28, 28, lo12).to_le_bytes());

            // jalr t1(x6), t3, 0  (t1 = return address for lazy resolver)
            out.extend_from_slice(&riscv_jalr(6, 28, 0).to_le_bytes());

            // nop (addi x0, x0, 0)
            out.extend_from_slice(&riscv_addi(0, 0, 0).to_le_bytes());
        }
        out
    }
}

// ===========================================================================
// Architecture-specific instruction encoding helpers
// ===========================================================================

// ---- AArch64 helpers ----

/// Returns the 4K page address (bits [63:12]).
#[inline]
fn aarch64_page(addr: u64) -> u64 {
    addr & !0xFFF
}

/// Encodes an ADRP instruction: `adrp Xd, #page_offset`.
/// `page_diff` is in bytes and must be page-aligned (multiple of 4096).
fn encode_adrp(rd: u32, page_diff: i64) -> u32 {
    let imm = (page_diff >> 12) as i32;
    let immlo = (imm & 0x3) as u32;
    let immhi = ((imm >> 2) & 0x7_FFFF) as u32;
    0x9000_0000 | (immlo << 29) | (immhi << 5) | (rd & 0x1F)
}

/// Encodes `LDR Xt, [Xn, #imm12]` (64-bit load, unsigned offset).
/// `byte_offset` must be 8-byte aligned.
fn encode_ldr_imm64(rt: u32, rn: u32, byte_offset: u32) -> u32 {
    let imm12 = (byte_offset / 8) & 0xFFF;
    0xf940_0000 | (imm12 << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
}

/// Encodes `ADD Xd, Xn, #imm12` (64-bit add immediate, no shift).
fn encode_add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    let imm = imm12 & 0xFFF;
    0x9100_0000 | (imm << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
}

// ---- RISC-V 64 helpers ----

/// Splits a 32-bit signed immediate into (hi20, lo12) for RISC-V
/// AUIPC+load/addi pairs, with lo12 sign-extension compensation.
fn riscv_split_imm(imm: i32) -> (i32, i32) {
    let lo = (imm << 20) >> 20; // sign-extend low 12 bits
    let mut hi = imm.wrapping_sub(lo);
    // hi must be aligned to 0x1000 (the low 12 bits are zero after the sub).
    // Shift right by 12 to get the hi20 value for AUIPC.
    hi >>= 12;
    (hi, lo)
}

/// Encodes RISC-V AUIPC: `auipc rd, imm20`.
fn riscv_auipc(rd: u32, imm20: u32) -> u32 {
    ((imm20 & 0xF_FFFF) << 12) | ((rd & 0x1F) << 7) | 0x17
}

/// Encodes RISC-V LD (load doubleword): `ld rd, offset(rs1)`.
fn riscv_ld(rd: u32, rs1: u32, offset: i32) -> u32 {
    let imm = (offset as u32) & 0xFFF;
    (imm << 20) | ((rs1 & 0x1F) << 15) | (0x3 << 12) | ((rd & 0x1F) << 7) | 0x03
}

/// Encodes RISC-V ADDI: `addi rd, rs1, imm12`.
/// funct3 = 0b000 (bits [14:12]), opcode = 0x13 (OP-IMM).
#[allow(clippy::identity_op)]
fn riscv_addi(rd: u32, rs1: u32, imm12: i32) -> u32 {
    let imm = (imm12 as u32) & 0xFFF;
    (imm << 20) | ((rs1 & 0x1F) << 15) | (0x0 << 12) | ((rd & 0x1F) << 7) | 0x13
}

/// Encodes RISC-V JALR: `jalr rd, rs1, offset`.
/// funct3 = 0b000 (bits [14:12]), opcode = 0x67 (JALR).
#[allow(clippy::identity_op)]
fn riscv_jalr(rd: u32, rs1: u32, offset: i32) -> u32 {
    let imm = (offset as u32) & 0xFFF;
    (imm << 20) | ((rs1 & 0x1F) << 15) | (0x0 << 12) | ((rd & 0x1F) << 7) | 0x67
}

/// Encodes RISC-V SUB: `sub rd, rs1, rs2`.
/// funct7 = 0x20 (bits [31:25]), funct3 = 0b000 (bits [14:12]), opcode = 0x33 (OP).
#[allow(clippy::identity_op)]
fn riscv_sub(rd: u32, rs1: u32, rs2: u32) -> u32 {
    (0x20 << 25)
        | ((rs2 & 0x1F) << 20)
        | ((rs1 & 0x1F) << 15)
        | (0x0 << 12)
        | ((rd & 0x1F) << 7)
        | 0x33
}

/// Encodes RISC-V SRLI: `srli rd, rs1, shamt`.
fn riscv_srli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    ((shamt & 0x3F) << 20) | ((rs1 & 0x1F) << 15) | (0x5 << 12) | ((rd & 0x1F) << 7) | 0x13
}

// ===========================================================================
// Dynamic Relocation Table serializers
// ===========================================================================

/// Serializes a slice of dynamic relocations into a `.rela.dyn` section
/// (data relocations, patched at load time by the dynamic linker).
///
/// Each entry is encoded as Elf64_Rela (24 bytes). For 32-bit targets, the
/// caller should use the 32-bit serialization path separately.
pub fn build_rela_dyn(relocations: &[DynamicRelocation]) -> Vec<u8> {
    let mut out = Vec::with_capacity(relocations.len() * 24);
    for r in relocations {
        out.extend_from_slice(&r.to_bytes_64_le());
    }
    out
}

/// Serializes a slice of dynamic relocations into a `.rela.plt` section
/// (PLT/function relocations, resolved lazily or at load time).
///
/// Format is identical to `.rela.dyn` — each entry is Elf64_Rela (24 bytes).
pub fn build_rela_plt(relocations: &[DynamicRelocation]) -> Vec<u8> {
    let mut out = Vec::with_capacity(relocations.len() * 24);
    for r in relocations {
        out.extend_from_slice(&r.to_bytes_64_le());
    }
    out
}

// ===========================================================================
// Program Header constructors
// ===========================================================================

/// Creates a `PT_DYNAMIC` program header referencing the `.dynamic` section.
///
/// - `dynamic_section_addr`: virtual address of `.dynamic`.
/// - `dynamic_section_size`: total size of `.dynamic` in bytes.
///
/// Flags: PF_R | PF_W (the GOT entries referenced by `.dynamic` are writable).
/// Alignment: 8 bytes (natural alignment for 64-bit ELF).
pub fn create_dynamic_phdr(dynamic_section_addr: u64, dynamic_section_size: u64) -> ProgramHeader {
    ProgramHeader {
        p_type: PT_DYNAMIC,
        p_flags: PF_R | PF_W,
        p_offset: dynamic_section_addr,
        p_vaddr: dynamic_section_addr,
        p_paddr: dynamic_section_addr,
        p_filesz: dynamic_section_size,
        p_memsz: dynamic_section_size,
        p_align: 8,
    }
}

/// Creates a `PT_INTERP` program header referencing the `.interp` section
/// (which contains the NUL-terminated dynamic linker path string).
///
/// - `interp_section_addr`: virtual address of `.interp`.
/// - `interp_section_size`: size of the interp string including NUL terminator.
///
/// Flags: PF_R (read-only).
/// Alignment: 1 byte.
pub fn create_interp_phdr(interp_section_addr: u64, interp_section_size: u64) -> ProgramHeader {
    ProgramHeader {
        p_type: PT_INTERP,
        p_flags: PF_R,
        p_offset: interp_section_addr,
        p_vaddr: interp_section_addr,
        p_paddr: interp_section_addr,
        p_filesz: interp_section_size,
        p_memsz: interp_section_size,
        p_align: 1,
    }
}

/// Returns the correct dynamic linker path string for the given target
/// architecture. This path is written into the `.interp` section and
/// referenced by `PT_INTERP`.
///
/// Each Linux architecture has a different conventional path:
/// - x86-64: `/lib64/ld-linux-x86-64.so.2`
/// - i686: `/lib/ld-linux.so.2`
/// - AArch64: `/lib/ld-linux-aarch64.so.1`
/// - RISC-V 64: `/lib/ld-linux-riscv64-lp64d.so.1`
pub fn interp_string(target: &Target) -> &'static str {
    target.dynamic_linker_path()
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_entry_serialization_64() {
        let entry = DynamicEntry::new(DT_NEEDED, 42);
        let bytes = entry.to_bytes_64_le();
        assert_eq!(bytes.len(), 16);
        // tag = 1 (DT_NEEDED) as i64 LE
        assert_eq!(i64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1);
        // value = 42
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 42);
    }

    #[test]
    fn test_dynamic_entry_serialization_32() {
        let entry = DynamicEntry::new(DT_FINI, 0x1234);
        let bytes = entry.to_bytes_32_le();
        assert_eq!(bytes.len(), 8);
        assert_eq!(
            i32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            DT_FINI as i32
        );
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 0x1234);
    }

    #[test]
    fn test_dynamic_relocation_64() {
        let reloc = DynamicRelocation {
            offset: 0x400100,
            reloc_type: 7,
            symbol_index: 3,
            addend: -8,
        };
        let bytes = reloc.to_bytes_64_le();
        assert_eq!(bytes.len(), 24);
        assert_eq!(
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            0x400100
        );
        let r_info = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(r_info >> 32, 3); // symbol index
        assert_eq!(r_info & 0xFFFF_FFFF, 7); // reloc type
        assert_eq!(i64::from_le_bytes(bytes[16..24].try_into().unwrap()), -8);
    }

    #[test]
    fn test_gnu_hash_function() {
        // Verify the standard GNU hash algorithm against known values.
        assert_eq!(gnu_hash(""), 5381);
        // "printf" hash = well-known value from glibc
        let h = gnu_hash("printf");
        assert_ne!(h, 0); // basic sanity
                          // Determinism: hashing the same string twice gives the same result.
        assert_eq!(gnu_hash("main"), gnu_hash("main"));
        // Different strings should (very likely) produce different hashes.
        assert_ne!(gnu_hash("foo"), gnu_hash("bar"));
    }

    #[test]
    fn test_gnu_hash_known_values() {
        // GNU hash is: h=5381; for each byte c: h = h*33 + c
        // For "a" (0x61): h = 5381*33 + 97 = 177573 + 97 = 177670 = 0x2B606
        assert_eq!(gnu_hash("a"), 177670);
    }

    #[test]
    fn test_dynsym_table_filtering() {
        let mut table = DynamicSymbolTable::new();

        // Local symbol should be filtered out.
        let local_sym = SymbolEntry {
            name: "local_func".to_owned(),
            value: 0x1000,
            size: 64,
            binding: SymbolBinding::Local,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };
        table.add_symbol(&local_sym);
        assert_eq!(table.symbol_count(), 1); // only null symbol

        // Global/Default should be added.
        let global_sym = SymbolEntry {
            name: "global_func".to_owned(),
            value: 0x2000,
            size: 128,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };
        table.add_symbol(&global_sym);
        assert_eq!(table.symbol_count(), 2);

        // Hidden symbol should be filtered out.
        let hidden_sym = SymbolEntry {
            name: "hidden_func".to_owned(),
            value: 0x3000,
            size: 32,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Hidden,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };
        table.add_symbol(&hidden_sym);
        assert_eq!(table.symbol_count(), 2); // still 2

        // Weak/Protected should be added.
        let weak_sym = SymbolEntry {
            name: "weak_var".to_owned(),
            value: 0x4000,
            size: 8,
            binding: SymbolBinding::Weak,
            sym_type: SymbolType::Object,
            visibility: SymbolVisibility::Protected,
            section_index: 2,
            defining_object: 0,
            is_defined: true,
        };
        table.add_symbol(&weak_sym);
        assert_eq!(table.symbol_count(), 3);
    }

    #[test]
    fn test_dynsym_serialization() {
        let mut table = DynamicSymbolTable::new();
        let sym = SymbolEntry {
            name: "test_sym".to_owned(),
            value: 0x1000,
            size: 16,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
            defining_object: 0,
            is_defined: true,
        };
        table.add_symbol(&sym);

        let dynsym = table.build_dynsym();
        // 2 symbols × 24 bytes = 48
        assert_eq!(dynsym.len(), 48);

        let dynstr = table.build_dynstr();
        // "\0" + "test_sym\0" = 1 + 9 = 10
        assert_eq!(dynstr.len(), 10);
        assert_eq!(dynstr[0], 0);
        assert_eq!(&dynstr[1..9], b"test_sym");
        assert_eq!(dynstr[9], 0);
    }

    #[test]
    fn test_build_gnu_hash_empty() {
        // Building a hash table with only the null symbol should not panic.
        let symbols = vec![DynSymEntry {
            name: String::new(),
            value: 0,
            size: 0,
            binding: SymbolBinding::Local,
            sym_type: SymbolType::NoType,
            visibility: SymbolVisibility::Default,
            section_index: 0,
        }];
        let hash = build_gnu_hash(&symbols, &[0]);
        assert!(!hash.is_empty());
        // Header should have: nbuckets(4) + symoffset(4) + bloom_size(4) + bloom_shift(4)
        assert!(hash.len() >= 16);
    }

    #[test]
    fn test_got_builder() {
        let target = Target::X86_64;
        let mut got = GotBuilder::new(0x200000, 0x201000, 0x100000, &target);

        assert_eq!(got.got_address(), 0x200000);
        assert_eq!(got.got_plt_address(), 0x201000);

        got.add_entry(GotEntry {
            symbol_name: "data_sym".to_owned(),
            offset: 0,
            initial_value: 0,
        });
        assert_eq!(got.entry_count(), 1);
        assert_eq!(got.got_size(), 8);

        let got_bytes = got.build_got();
        assert_eq!(got_bytes.len(), 8);
        assert_eq!(u64::from_le_bytes(got_bytes[0..8].try_into().unwrap()), 0);

        let got_plt_bytes = got.build_got_plt();
        // 3 reserved slots × 8 bytes = 24
        assert_eq!(got_plt_bytes.len(), 24);
        // Slot 0 = dynamic addr
        assert_eq!(
            u64::from_le_bytes(got_plt_bytes[0..8].try_into().unwrap()),
            0x100000
        );
    }

    #[test]
    fn test_plt_builder_x86_64() {
        let target = Target::X86_64;
        let mut plt = PltBuilder::new(0x1000, 0x2000, target);
        assert_eq!(plt.plt_address(), 0x1000);

        plt.add_entry(PltEntry {
            symbol_name: "puts".to_owned(),
            got_offset: 24,
            plt_index: 0,
        });

        let bytes = plt.build_plt(&target);
        // PLT[0] = 16 bytes + PLT[1] = 16 bytes = 32
        assert_eq!(bytes.len(), 32);
        // PLT[0] starts with ff 35 (push [rip+disp32])
        assert_eq!(bytes[0], 0xff);
        assert_eq!(bytes[1], 0x35);
        // PLT[1] starts with ff 25 (jmp [rip+disp32])
        assert_eq!(bytes[16], 0xff);
        assert_eq!(bytes[17], 0x25);
    }

    #[test]
    fn test_plt_builder_i686() {
        let target = Target::I686;
        let mut plt = PltBuilder::new(0x1000, 0x2000, target);
        plt.add_entry(PltEntry {
            symbol_name: "puts".to_owned(),
            got_offset: 12,
            plt_index: 0,
        });

        let bytes = plt.build_plt(&target);
        assert_eq!(bytes.len(), 32);
        // PLT[0] starts with ff 35 (push [abs32])
        assert_eq!(bytes[0], 0xff);
        assert_eq!(bytes[1], 0x35);
    }

    #[test]
    fn test_dynamic_section_builder() {
        let mut dsb = DynamicSectionBuilder::new();
        dsb.add_needed("libc.so.6");
        dsb.set_soname("libtest.so.1");
        dsb.set_dynstr_size(64);
        dsb.set_rela_dyn_size(48);
        dsb.set_rela_plt_size(24);

        let layout = DynamicLayout {
            dynamic_addr: 0x100000,
            dynsym_addr: 0x100100,
            dynstr_addr: 0x100200,
            gnu_hash_addr: 0x100300,
            got_addr: 0x200000,
            got_plt_addr: 0x200100,
            plt_addr: 0x300000,
            rela_dyn_addr: 0x100400,
            rela_plt_addr: 0x100500,
            interp_addr: 0x100600,
        };

        let bytes = dsb.build(&layout);
        assert!(!bytes.is_empty());
        // Each entry is 16 bytes (64-bit). Last entry is DT_NULL.
        assert_eq!(bytes.len() % 16, 0);
        let last_tag =
            i64::from_le_bytes(bytes[bytes.len() - 16..bytes.len() - 8].try_into().unwrap());
        assert_eq!(last_tag, DT_NULL);
    }

    #[test]
    fn test_interp_string() {
        assert_eq!(
            interp_string(&Target::X86_64),
            "/lib64/ld-linux-x86-64.so.2"
        );
        assert_eq!(interp_string(&Target::I686), "/lib/ld-linux.so.2");
        assert_eq!(
            interp_string(&Target::AArch64),
            "/lib/ld-linux-aarch64.so.1"
        );
        assert_eq!(
            interp_string(&Target::RiscV64),
            "/lib/ld-linux-riscv64-lp64d.so.1"
        );
    }

    #[test]
    fn test_create_dynamic_phdr() {
        let phdr = create_dynamic_phdr(0x400000, 256);
        assert_eq!(phdr.p_type, PT_DYNAMIC);
        assert_eq!(phdr.p_flags, PF_R | PF_W);
        assert_eq!(phdr.p_vaddr, 0x400000);
        assert_eq!(phdr.p_filesz, 256);
        assert_eq!(phdr.p_memsz, 256);
    }

    #[test]
    fn test_create_interp_phdr() {
        let phdr = create_interp_phdr(0x200000, 28);
        assert_eq!(phdr.p_type, PT_INTERP);
        assert_eq!(phdr.p_flags, PF_R);
        assert_eq!(phdr.p_vaddr, 0x200000);
        assert_eq!(phdr.p_filesz, 28);
        assert_eq!(phdr.p_align, 1);
    }

    #[test]
    fn test_build_rela_dyn() {
        let relocs = vec![
            DynamicRelocation {
                offset: 0x200000,
                reloc_type: 8,
                symbol_index: 1,
                addend: 0,
            },
            DynamicRelocation {
                offset: 0x200008,
                reloc_type: 8,
                symbol_index: 2,
                addend: -4,
            },
        ];
        let bytes = build_rela_dyn(&relocs);
        assert_eq!(bytes.len(), 48); // 2 × 24
    }

    #[test]
    fn test_build_rela_plt() {
        let relocs = vec![DynamicRelocation {
            offset: 0x201018,
            reloc_type: 7,
            symbol_index: 1,
            addend: 0,
        }];
        let bytes = build_rela_plt(&relocs);
        assert_eq!(bytes.len(), 24); // 1 × 24
    }

    #[test]
    fn test_dt_constants_values() {
        // Verify exact constant values per ELF specification.
        assert_eq!(DT_NULL, 0);
        assert_eq!(DT_NEEDED, 1);
        assert_eq!(DT_PLTRELSZ, 2);
        assert_eq!(DT_PLTGOT, 3);
        assert_eq!(DT_HASH, 4);
        assert_eq!(DT_STRTAB, 5);
        assert_eq!(DT_SYMTAB, 6);
        assert_eq!(DT_RELA, 7);
        assert_eq!(DT_RELASZ, 8);
        assert_eq!(DT_RELAENT, 9);
        assert_eq!(DT_STRSZ, 10);
        assert_eq!(DT_SYMENT, 11);
        assert_eq!(DT_INIT, 12);
        assert_eq!(DT_FINI, 13);
        assert_eq!(DT_SONAME, 14);
        assert_eq!(DT_RPATH, 15);
        assert_eq!(DT_SYMBOLIC, 16);
        assert_eq!(DT_PLTREL, 20);
        assert_eq!(DT_JMPREL, 23);
        assert_eq!(DT_FLAGS, 30);
        assert_eq!(DT_FLAGS_1, 0x6fff_fffb);
        assert_eq!(DT_GNU_HASH, 0x6fff_fef5_i64);
    }

    #[test]
    fn test_dynsym_entry_elf_format() {
        let sym = DynSymEntry {
            name: "test".to_owned(),
            value: 0x1000,
            size: 4,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Func,
            visibility: SymbolVisibility::Default,
            section_index: 1,
        };
        let bytes = sym.to_bytes_64_le(5);
        // st_name offset = 5
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 5);
        // st_info = (1 << 4) | 2 = 0x12 (GLOBAL + FUNC)
        assert_eq!(bytes[4], 0x12);
        // st_other = 0 (DEFAULT visibility)
        assert_eq!(bytes[5], 0x00);
        // st_shndx = 1
        assert_eq!(u16::from_le_bytes(bytes[6..8].try_into().unwrap()), 1);
        // st_value = 0x1000
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 0x1000);
        // st_size = 4
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 4);
    }

    #[test]
    fn test_got_builder_32bit() {
        let target = Target::I686;
        let mut got = GotBuilder::new(0x200000, 0x201000, 0x100000, &target);
        got.add_entry(GotEntry {
            symbol_name: "x".to_owned(),
            offset: 0,
            initial_value: 0xDEAD,
        });
        let bytes = got.build_got();
        assert_eq!(bytes.len(), 4);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0xDEAD);

        let plt_bytes = got.build_got_plt();
        // 3 reserved × 4 = 12 bytes
        assert_eq!(plt_bytes.len(), 12);
        assert_eq!(
            u32::from_le_bytes(plt_bytes[0..4].try_into().unwrap()),
            0x100000
        );
    }
}
