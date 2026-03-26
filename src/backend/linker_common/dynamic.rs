//! Dynamic linking support for BCC's built-in linker.
//!
//! Generates ALL ELF sections required for shared objects (ET_DYN) and
//! dynamically-linked executables across all four target architectures:
//! x86-64, i686, AArch64, and RISC-V 64.
//!
//! ## Generated Sections
//!
//! | Section        | Purpose                                    |
//! |----------------|--------------------------------------------|
//! | `.dynamic`     | Dynamic linking metadata tags               |
//! | `.dynsym`      | Dynamic symbol table                        |
//! | `.dynstr`      | Dynamic string table                        |
//! | `.gnu.hash`    | GNU hash table for fast symbol lookup       |
//! | `.got`         | Global Offset Table (data relocations)      |
//! | `.got.plt`     | GOT entries for PLT lazy binding            |
//! | `.plt`         | Procedure Linkage Table stubs               |
//! | `.rela.dyn`    | RELA relocations for data references        |
//! | `.rela.plt`    | RELA relocations for PLT/GOT entries        |
//!
//! ## Standalone Backend
//!
//! NO external linker is invoked. This module is part of BCC's built-in
//! linker pipeline that produces complete ELF binaries from object files.

use crate::backend::elf_writer_common::{
    Elf32Rel, Elf32Symbol, Elf64Rela, Elf64Symbol, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC,
    STT_NOTYPE, STT_OBJECT, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED,
};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// ===========================================================================
// ELF Dynamic Section Tag Constants (d_tag values)
// ===========================================================================

pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_PLTGOT: i64 = 3;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_INIT: i64 = 12;
pub const DT_FINI: i64 = 13;
pub const DT_SONAME: i64 = 14;
pub const DT_RPATH: i64 = 15;
pub const DT_SYMBOLIC: i64 = 16;
pub const DT_REL: i64 = 17;
pub const DT_RELSZ: i64 = 18;
pub const DT_RELENT: i64 = 19;
pub const DT_PLTREL: i64 = 20;
pub const DT_DEBUG: i64 = 21;
pub const DT_TEXTREL: i64 = 22;
pub const DT_JMPREL: i64 = 23;
pub const DT_BIND_NOW: i64 = 24;
pub const DT_INIT_ARRAY: i64 = 25;
pub const DT_FINI_ARRAY: i64 = 26;
pub const DT_INIT_ARRAYSZ: i64 = 27;
pub const DT_FINI_ARRAYSZ: i64 = 28;
pub const DT_FLAGS: i64 = 30;
pub const DT_FLAGS_1: i64 = 0x6fff_fffb;
pub const DT_GNU_HASH: i64 = 0x6fff_fef5;
pub const DT_RELACOUNT: i64 = 0x6fff_fff9;
pub const DT_RELCOUNT: i64 = 0x6fff_fffa;
pub const DT_VERSYM: i64 = 0x6fff_fff0;
pub const DT_VERDEF: i64 = 0x6fff_fffc;
pub const DT_VERDEFNUM: i64 = 0x6fff_fffd;
pub const DT_VERNEED: i64 = 0x6fff_fffe;
pub const DT_VERNEEDNUM: i64 = 0x6fff_ffff;

// PLT sizes are computed via `plt_sizes()` at the bottom of this module.
// Architecture-specific sizes: x86-64/i686 = 16/16, AArch64/RV64 = 32/16.

// ===========================================================================
// Symbol Structures
// ===========================================================================

/// A symbol in the dynamic symbol table (`.dynsym`).
#[derive(Debug, Clone)]
pub struct DynamicSymbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub binding: u8,
    pub sym_type: u8,
    pub visibility: u8,
    pub section_index: u16,
    pub is_defined: bool,
    pub is_plt_entry: bool,
    pub got_offset: Option<u64>,
    pub plt_index: Option<u32>,
}

/// A symbol exported from the shared object being linked.
#[derive(Debug, Clone)]
pub struct ExportedSymbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub binding: u8,
    pub sym_type: u8,
    pub visibility: u8,
    pub section_index: u16,
}

/// A symbol imported from another shared object.
#[derive(Debug, Clone)]
pub struct ImportedSymbol {
    pub name: String,
    pub binding: u8,
    pub sym_type: u8,
    pub needs_plt: bool,
    /// If true, this is a data symbol requiring `R_*_COPY` relocation
    /// rather than GOT/PLT entries. The linker allocates space in
    /// `.bss.copy` and the dynamic linker copies the initial value
    /// from the shared library at load time.
    pub needs_copy: bool,
    /// Size in bytes for copy-relocated data symbols.
    pub copy_size: u64,
}

// ===========================================================================
// Dynamic Symbol Table (.dynsym / .dynstr)
// ===========================================================================

/// Builder for the `.dynsym` and `.dynstr` ELF sections.
pub struct DynamicSymbolTable {
    symbols: Vec<DynamicSymbol>,
    string_table: Vec<u8>,
    string_offsets: FxHashMap<String, u32>,
}

impl Default for DynamicSymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicSymbolTable {
    /// Create with mandatory null entry at index 0 and null byte in `.dynstr`.
    pub fn new() -> Self {
        let null_sym = DynamicSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: 0,
            is_defined: false,
            is_plt_entry: false,
            got_offset: None,
            plt_index: None,
        };
        let mut string_offsets = FxHashMap::default();
        string_offsets.insert(String::new(), 0);
        Self {
            symbols: vec![null_sym],
            string_table: vec![0],
            string_offsets,
        }
    }

    /// Add a dynamic symbol, return its `.dynsym` index.
    pub fn add_symbol(&mut self, sym: DynamicSymbol) -> u32 {
        let name_clone = sym.name.clone();
        self.add_string(&name_clone);
        let index = self.symbols.len() as u32;
        self.symbols.push(sym);
        index
    }

    /// Add a string to `.dynstr`, return its byte offset. Deduplicates via FxHashMap.
    pub fn add_string(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.string_offsets.get(s) {
            return offset;
        }
        let offset = self.string_table.len() as u32;
        self.string_table.extend_from_slice(s.as_bytes());
        self.string_table.push(0);
        self.string_offsets.insert(s.to_owned(), offset);
        offset
    }

    /// Add a string to `.dynstr` without creating a symbol.
    ///
    /// Used to insert needed-library names (for `DT_NEEDED`) into the
    /// dynamic string table.
    pub fn add_dynstr_string(&mut self, s: &str) -> u32 {
        self.add_string(s)
    }

    /// Encode `.dynsym` to bytes. Locals before globals per ELF spec.
    ///
    /// Symbol binding (`STB_LOCAL`, `STB_GLOBAL`, `STB_WEAK`) determines
    /// ordering — locals must precede globals. Visibility values (`STV_DEFAULT`,
    /// `STV_HIDDEN`, `STV_PROTECTED`) are encoded into `st_other`.
    pub fn encode_dynsym(&self, target: &Target) -> Vec<u8> {
        let mut locals: Vec<&DynamicSymbol> = Vec::new();
        let mut globals: Vec<&DynamicSymbol> = Vec::new();
        for (i, sym) in self.symbols.iter().enumerate() {
            if i == 0 {
                // Null entry is always local
                locals.push(sym);
            } else if sym.binding == STB_LOCAL {
                locals.push(sym);
            } else {
                // STB_GLOBAL and STB_WEAK go after locals
                globals.push(sym);
            }
        }
        let ordered: Vec<&DynamicSymbol> = locals.iter().chain(globals.iter()).copied().collect();
        let entry_size = if target.is_64bit() { 24usize } else { 16usize };
        let mut buf = Vec::with_capacity(ordered.len() * entry_size);
        for sym in &ordered {
            let name_offset = self.string_offsets.get(&sym.name).copied().unwrap_or(0);
            let st_info = (sym.binding << 4) | (sym.sym_type & 0x0f);
            // Visibility: STV_DEFAULT(0), STV_HIDDEN(2), STV_PROTECTED(3)
            let st_other = match sym.visibility {
                STV_HIDDEN => STV_HIDDEN,
                STV_PROTECTED => STV_PROTECTED,
                _ => STV_DEFAULT,
            };
            let shndx = if sym.is_defined { sym.section_index } else { 0 };
            if target.is_64bit() {
                buf.extend_from_slice(
                    &Elf64Symbol {
                        st_name: name_offset,
                        st_info,
                        st_other,
                        st_shndx: shndx,
                        st_value: sym.value,
                        st_size: sym.size,
                    }
                    .to_bytes(),
                );
            } else {
                buf.extend_from_slice(
                    &Elf32Symbol {
                        st_name: name_offset,
                        st_value: sym.value as u32,
                        st_size: sym.size as u32,
                        st_info,
                        st_other,
                        st_shndx: shndx,
                    }
                    .to_bytes(),
                );
            }
        }
        buf
    }

    /// Return the raw `.dynstr` section bytes.
    pub fn encode_dynstr(&self) -> Vec<u8> {
        self.string_table.clone()
    }

    /// Total number of symbols (including the null entry).
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }
    /// Borrow the symbol list.
    pub fn symbols(&self) -> &[DynamicSymbol] {
        &self.symbols
    }
    /// Mutable borrow of the symbol list — used to patch values
    /// after address layout computation.
    pub fn symbols_mut(&mut self) -> &mut [DynamicSymbol] {
        &mut self.symbols
    }
    /// Count of local-binding symbols (used for `.dynsym` sh_info).
    pub fn local_count(&self) -> u32 {
        self.symbols
            .iter()
            .filter(|s| s.binding == STB_LOCAL)
            .count() as u32
    }
}

// ===========================================================================
// GNU Hash Function and Table (.gnu.hash)
// ===========================================================================

/// Compute the GNU hash of a symbol name (`dl_new_hash` algorithm).
pub fn gnu_hash(name: &str) -> u32 {
    let mut h: u32 = 5381;
    for byte in name.bytes() {
        h = h.wrapping_mul(33).wrapping_add(byte as u32);
    }
    h
}

/// GNU hash table (`.gnu.hash` section) for fast dynamic symbol lookup.
pub struct GnuHashTable {
    pub bucket_count: u32,
    pub symoffset: u32,
    pub bloom_size: u32,
    pub bloom_shift: u32,
    pub bloom: Vec<u64>,
    pub buckets: Vec<u32>,
    pub chains: Vec<u32>,
}

impl GnuHashTable {
    /// Build from dynamic symbols and reorder them so that the `.dynsym`
    /// layout matches the `.gnu.hash` lookup expectations.
    ///
    /// The GNU hash format requires that:
    ///  1. Symbols 0..symoffset-1 are NOT in the hash table (null + undefined).
    ///  2. Symbols symoffset..end are in the hash table, sorted by bucket.
    ///  3. Symbols within a single bucket are contiguous — the dynamic linker
    ///     walks chains by incrementing the dynsym index.
    ///
    /// This function partitions and reorders `symbols` in place to satisfy
    /// those constraints, then builds the matching hash table.
    pub fn build_and_reorder(symbols: &mut Vec<DynamicSymbol>) -> Self {
        let mut seen = FxHashSet::default();

        // Classify each symbol as hashable or non-hashable.
        let mut hashable_indices: Vec<(usize, u32)> = Vec::new(); // (orig_index, hash)
        let mut non_hashable_indices: Vec<usize> = Vec::new();

        for (i, sym) in symbols.iter().enumerate() {
            if i == 0 {
                non_hashable_indices.push(i);
                continue;
            }
            let is_exportable = sym.is_defined
                && (sym.binding == STB_GLOBAL || sym.binding == STB_WEAK)
                && !sym.name.is_empty()
                && sym.visibility != STV_HIDDEN;
            let valid_type = sym.sym_type == STT_FUNC
                || sym.sym_type == STT_OBJECT
                || sym.sym_type == STT_NOTYPE;
            if is_exportable && valid_type && seen.insert(sym.name.clone()) {
                hashable_indices.push((i, gnu_hash(&sym.name)));
            } else {
                non_hashable_indices.push(i);
            }
        }

        if hashable_indices.is_empty() {
            return Self {
                bucket_count: 1,
                symoffset: symbols.len() as u32,
                bloom_size: 1,
                bloom_shift: 6,
                bloom: vec![0],
                buckets: vec![0],
                chains: Vec::new(),
            };
        }

        // Choose bucket count (power of two, ~4/3 of symbol count).
        let bucket_count = std::cmp::max(
            1u32,
            ((hashable_indices.len() * 4 / 3) as u32).next_power_of_two(),
        );

        // Sort hashable symbols by bucket index for contiguous placement.
        hashable_indices.sort_by_key(|&(_, h)| h % bucket_count);

        let symoffset = non_hashable_indices.len() as u32;

        // Reorder the symbol table: non-hashable first, then hashable.
        let old_symbols = symbols.clone();
        symbols.clear();
        for &idx in &non_hashable_indices {
            symbols.push(old_symbols[idx].clone());
        }
        for &(idx, _) in &hashable_indices {
            symbols.push(old_symbols[idx].clone());
        }

        // Now symbols[symoffset + ci] corresponds to hashable_indices[ci].
        // Build Bloom filter.
        let bloom_shift: u32 = 6;
        let bloom_size = std::cmp::max(1u32, (hashable_indices.len() as u32).next_power_of_two());
        let bloom_mask = bloom_size as u64 - 1;
        let mut bloom = vec![0u64; bloom_size as usize];
        for &(_, h) in &hashable_indices {
            let wi = ((h as u64 / 64) & bloom_mask) as usize;
            bloom[wi] |= 1u64 << (h % 64);
            bloom[wi] |= 1u64 << ((h >> bloom_shift) % 64);
        }

        // Build buckets and chains.
        let mut buckets = vec![0u32; bucket_count as usize];
        let mut chains = vec![0u32; hashable_indices.len()];

        // Group symbols by bucket (indices into hashable_indices, which is
        // already sorted by bucket so groups are contiguous).
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); bucket_count as usize];
        for (ci, &(_, h)) in hashable_indices.iter().enumerate() {
            groups[(h % bucket_count) as usize].push(ci);
        }

        for indices in &groups {
            if let Some(&first) = indices.first() {
                let bkt = (hashable_indices[first].1 % bucket_count) as usize;
                // Point to the actual dynsym index.
                buckets[bkt] = first as u32 + symoffset;
            }
            for (pos, &ci) in indices.iter().enumerate() {
                let h = hashable_indices[ci].1;
                // Store hash with low bit = 1 for last entry in chain.
                chains[ci] = (h & !1) | u32::from(pos == indices.len() - 1);
            }
        }

        Self {
            bucket_count,
            symoffset,
            bloom_size,
            bloom_shift,
            bloom,
            buckets,
            chains,
        }
    }

    /// Legacy build function (does NOT reorder symbols).
    /// Retained for backward compatibility with code paths that manage
    /// symbol ordering externally. Prefer `build_and_reorder` for new code.
    pub fn build(symbols: &[DynamicSymbol]) -> Self {
        let mut v = symbols.to_vec();
        Self::build_and_reorder(&mut v)
    }

    /// Encode the `.gnu.hash` section to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let total = 16 + self.bloom.len() * 8 + self.buckets.len() * 4 + self.chains.len() * 4;
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.bucket_count.to_le_bytes());
        buf.extend_from_slice(&self.symoffset.to_le_bytes());
        buf.extend_from_slice(&self.bloom_size.to_le_bytes());
        buf.extend_from_slice(&self.bloom_shift.to_le_bytes());
        for &w in &self.bloom {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for &b in &self.buckets {
            buf.extend_from_slice(&b.to_le_bytes());
        }
        for &c in &self.chains {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf
    }
}

// ===========================================================================
// Global Offset Table (.got / .got.plt)
// ===========================================================================

/// Number of reserved entries at the start of `.got.plt`.
///
/// - **x86-64 / i686 / AArch64**: 3 entries — `[0]` = `.dynamic` address,
///   `[1]` = link\_map, `[2]` = resolver (filled by ld.so).
/// - **RISC-V 64**: 2 entries — `[0]` = resolver, `[1]` = link\_map
///   (the RISC-V ABI stores the resolver at `[0]`, not `[2]`).
pub fn got_plt_reserved_count(target: &Target) -> usize {
    match target {
        Target::RiscV64 => 2,
        _ => 3,
    }
}

/// An entry in the `.got` section.
#[derive(Debug, Clone)]
pub struct GotEntry {
    pub symbol_name: String,
    pub offset: u64,
    pub addend: i64,
    pub reloc_type: u32,
}

/// An entry in the `.got.plt` section.
#[derive(Debug, Clone)]
pub struct GotPltEntry {
    pub symbol_name: String,
    pub offset: u64,
    pub plt_index: u32,
}

/// Builder for `.got` and `.got.plt` sections.
pub struct GlobalOffsetTable {
    entries: Vec<GotEntry>,
    got_plt_entries: Vec<GotPltEntry>,
    target: Target,
}

impl GlobalOffsetTable {
    pub fn new(target: Target) -> Self {
        Self {
            entries: Vec::new(),
            got_plt_entries: Vec::new(),
            target,
        }
    }

    /// Add a `.got` entry, return its byte offset.
    pub fn add_got_entry(&mut self, sym: &str, reloc_type: u32) -> u64 {
        let pw = self.target.pointer_width() as u64;
        let offset = self.entries.len() as u64 * pw;
        self.entries.push(GotEntry {
            symbol_name: sym.to_owned(),
            offset,
            addend: 0,
            reloc_type,
        });
        offset
    }

    /// Add a `.got.plt` entry, return (offset, plt_index).
    pub fn add_got_plt_entry(&mut self, sym: &str) -> (u64, u32) {
        let pw = self.target.pointer_width() as u64;
        let idx = self.got_plt_entries.len() as u32;
        let reserved = got_plt_reserved_count(&self.target) as u64;
        let offset = (reserved + idx as u64) * pw;
        self.got_plt_entries.push(GotPltEntry {
            symbol_name: sym.to_owned(),
            offset,
            plt_index: idx,
        });
        (offset, idx)
    }

    /// Encode `.got` (all zeros; dynamic linker fills via GLOB_DAT).
    pub fn encode_got(&self) -> Vec<u8> {
        vec![0u8; self.entries.len() * self.target.pointer_width()]
    }

    /// Encode `.got.plt` with reserved entries + lazy-binding targets.
    pub fn encode_got_plt(&self, plt_base: u64) -> Vec<u8> {
        let pw = self.target.pointer_width();
        let (plt0_sz, pltn_sz) = plt_sizes(&self.target);
        let reserved = got_plt_reserved_count(&self.target);
        let total = reserved + self.got_plt_entries.len();
        let mut buf = Vec::with_capacity(total * pw);
        // Reserved entries (filled by dynamic linker at load time).
        for _ in 0..reserved {
            if pw == 8 {
                buf.extend_from_slice(&0u64.to_le_bytes());
            } else {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        for e in &self.got_plt_entries {
            let plt_entry = plt_base + plt0_sz as u64 + e.plt_index as u64 * pltn_sz as u64;
            let lazy = match self.target {
                Target::X86_64 | Target::I686 => plt_entry + 6,
                Target::AArch64 | Target::RiscV64 => plt_base,
            };
            if pw == 8 {
                buf.extend_from_slice(&lazy.to_le_bytes());
            } else {
                buf.extend_from_slice(&(lazy as u32).to_le_bytes());
            }
        }
        buf
    }

    /// Borrow the `.got` entry list.
    pub fn got_entries(&self) -> &[GotEntry] {
        &self.entries
    }
    /// Borrow the `.got.plt` entry list.
    pub fn got_plt_entries(&self) -> &[GotPltEntry] {
        &self.got_plt_entries
    }
    /// Total size of `.got.plt` in bytes (reserved + N stubs × pointer width).
    pub fn got_plt_size(&self) -> usize {
        (got_plt_reserved_count(&self.target) + self.got_plt_entries.len())
            * self.target.pointer_width()
    }
}

// ===========================================================================
// Procedure Linkage Table (.plt)
// ===========================================================================

/// A PLT stub entry for a single dynamically-resolved function.
#[derive(Debug, Clone)]
pub struct PltStub {
    pub symbol_name: String,
    pub got_plt_offset: u64,
    pub index: u32,
}

/// Builder for the `.plt` section with architecture-specific stubs.
pub struct ProcedureLinkageTable {
    /// PLT stub entries for each dynamically-linked function symbol.
    pub stubs: Vec<PltStub>,
    target: Target,
}

impl ProcedureLinkageTable {
    fn new(target: Target) -> Self {
        Self {
            stubs: Vec::new(),
            target,
        }
    }

    /// Public constructor for use by the linker common infrastructure
    /// when re-encoding PLT sections with final virtual addresses.
    pub fn new_public(target: Target) -> Self {
        Self::new(target)
    }

    /// Generate PLT0 header stub (resolver trampoline).
    pub fn generate_plt0(&self, got_plt_addr: u64, plt_addr: u64) -> Vec<u8> {
        match self.target {
            Target::X86_64 => self.plt0_x86_64(got_plt_addr, plt_addr),
            Target::I686 => self.plt0_i686(got_plt_addr, plt_addr),
            Target::AArch64 => self.plt0_aarch64(got_plt_addr, plt_addr),
            Target::RiscV64 => self.plt0_riscv64(got_plt_addr, plt_addr),
        }
    }

    /// Generate a per-function PLT entry (PLTn).
    pub fn generate_plt_entry(
        &self,
        stub: &PltStub,
        got_plt_addr: u64,
        plt_addr: u64,
        plt_offset: u64,
    ) -> Vec<u8> {
        match self.target {
            Target::X86_64 => self.pltn_x86_64(stub, got_plt_addr, plt_addr, plt_offset),
            Target::I686 => self.pltn_i686(stub, got_plt_addr, plt_addr, plt_offset),
            Target::AArch64 => self.pltn_aarch64(stub, got_plt_addr, plt_addr, plt_offset),
            Target::RiscV64 => self.pltn_riscv64(stub, got_plt_addr, plt_addr, plt_offset),
        }
    }

    // -- x86-64 PLT (RIP-relative) --

    fn plt0_x86_64(&self, gpa: u64, pa: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        // push [rip + GOT[1]] ; RIP after = pa + 6
        let d1 = ((gpa + 8) as i64 - (pa as i64 + 6)) as i32;
        b.extend_from_slice(&[0xff, 0x35]);
        b.extend_from_slice(&d1.to_le_bytes());
        // jmp [rip + GOT[2]] ; RIP after = pa + 12
        let d2 = ((gpa + 16) as i64 - (pa as i64 + 12)) as i32;
        b.extend_from_slice(&[0xff, 0x25]);
        b.extend_from_slice(&d2.to_le_bytes());
        // 4-byte NOP
        b.extend_from_slice(&[0x0f, 0x1f, 0x40, 0x00]);
        debug_assert_eq!(b.len(), 16);
        b
    }

    fn pltn_x86_64(&self, s: &PltStub, gpa: u64, pa: u64, po: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        let ea = pa + po;
        // jmp [rip + GOT_entry] ; RIP after = ea + 6
        let d = ((gpa + s.got_plt_offset) as i64 - (ea as i64 + 6)) as i32;
        b.extend_from_slice(&[0xff, 0x25]);
        b.extend_from_slice(&d.to_le_bytes());
        // push index
        b.push(0x68);
        b.extend_from_slice(&s.index.to_le_bytes());
        // jmp PLT0 ; RIP after = ea + 16
        let j = (pa as i64 - (ea as i64 + 16)) as i32;
        b.push(0xe9);
        b.extend_from_slice(&j.to_le_bytes());
        debug_assert_eq!(b.len(), 16);
        b
    }

    // -- i686 PLT (absolute addressing) --

    fn plt0_i686(&self, gpa: u64, _pa: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        let ga = gpa as u32;
        // push [GOT+4]
        b.extend_from_slice(&[0xff, 0x35]);
        b.extend_from_slice(&(ga + 4).to_le_bytes());
        // jmp [GOT+8]
        b.extend_from_slice(&[0xff, 0x25]);
        b.extend_from_slice(&(ga + 8).to_le_bytes());
        b.extend_from_slice(&[0x00; 4]);
        debug_assert_eq!(b.len(), 16);
        b
    }

    fn pltn_i686(&self, s: &PltStub, gpa: u64, pa: u64, po: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        let ea = (pa + po) as u32;
        // jmp [GOT_entry]
        b.extend_from_slice(&[0xff, 0x25]);
        b.extend_from_slice(&((gpa + s.got_plt_offset) as u32).to_le_bytes());
        // push reloc_offset (index * 8 for Elf32Rel)
        b.push(0x68);
        b.extend_from_slice(&(s.index * 8).to_le_bytes());
        // jmp PLT0
        let j = (pa as i32) - (ea as i32 + 16);
        b.push(0xe9);
        b.extend_from_slice(&j.to_le_bytes());
        debug_assert_eq!(b.len(), 16);
        b
    }

    // -- AArch64 PLT (ADRP + LDR PC-relative) --

    fn plt0_aarch64(&self, gpa: u64, pa: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(32);
        // stp x16, x30, [sp, #-16]!
        b.extend_from_slice(&0xA9BF7BF0u32.to_le_bytes());
        // adrp x16, PAGE(GOT+16) relative to pa+4
        let got2 = gpa + 16;
        let pd = (aarch64_page(got2) as i64 - aarch64_page(pa + 4) as i64) >> 12;
        b.extend_from_slice(&encode_aarch64_adrp(16, pd as i32).to_le_bytes());
        // ldr x17, [x16, #PAGEOFF(GOT+16)]
        b.extend_from_slice(&encode_aarch64_ldr64(17, 16, (got2 & 0xFFF) as u32 / 8).to_le_bytes());
        // add x16, x16, #PAGEOFF(GOT+16) — must match the LDR target (GOT[2])
        let got2_lo = (got2 & 0xFFF) as u32;
        b.extend_from_slice(&encode_aarch64_add_imm(16, 16, got2_lo).to_le_bytes());
        // br x17
        b.extend_from_slice(&0xD61F0220u32.to_le_bytes());
        // 3 × nop
        for _ in 0..3 {
            b.extend_from_slice(&0xD503201Fu32.to_le_bytes());
        }
        debug_assert_eq!(b.len(), 32);
        b
    }

    fn pltn_aarch64(&self, s: &PltStub, gpa: u64, pa: u64, po: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        let ea = pa + po;
        let ge = gpa + s.got_plt_offset;
        let pd = (aarch64_page(ge) as i64 - aarch64_page(ea) as i64) >> 12;
        let lo = (ge & 0xFFF) as u32;
        // adrp x16, PAGE(GOT_entry)
        b.extend_from_slice(&encode_aarch64_adrp(16, pd as i32).to_le_bytes());
        // ldr x17, [x16, #PAGEOFF]
        b.extend_from_slice(&encode_aarch64_ldr64(17, 16, lo / 8).to_le_bytes());
        // add x16, x16, #PAGEOFF
        b.extend_from_slice(&encode_aarch64_add_imm(16, 16, lo).to_le_bytes());
        // br x17
        b.extend_from_slice(&0xD61F0220u32.to_le_bytes());
        debug_assert_eq!(b.len(), 16);
        b
    }

    // -- RISC-V 64 PLT (AUIPC + LD PC-relative) --

    fn plt0_riscv64(&self, gpa: u64, pa: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(32);
        // RISC-V GOT.PLT layout (2-entry reserved model):
        //   [0] = _dl_runtime_resolve (resolver, filled by ld.so)
        //   [1] = link_map             (filled by ld.so)
        //   [2+] = JUMP_SLOT entries
        //
        // PLT0 loads resolver from GOT[0] into t3, link_map from
        // GOT[1] into t0, then jumps to resolver.
        let off = gpa as i64 - pa as i64; // PC-relative to GOT[0]
        let (hi, lo) = riscv_hi20_lo12(off as i32);
        // auipc t2(x7), hi20  — t2 = PLT0 + hi_page, base for
        //                        PC-relative GOT access
        b.extend_from_slice(&encode_rv_auipc(7, hi).to_le_bytes());
        // sub t1(x6), t1(x6), t3(x28) — PLT index bookkeeping
        b.extend_from_slice(&encode_rv_sub(6, 6, 28).to_le_bytes());
        // ld t3(x28), lo12(t2/x7) — load resolver from GOT[0]
        b.extend_from_slice(&encode_rv_ld(28, 7, lo).to_le_bytes());
        // addi t1(x6), t1(x6), -(32+12) — PLT index bookkeeping
        b.extend_from_slice(&encode_rv_addi(6, 6, -44).to_le_bytes());
        // addi t0(x5), t2(x7), lo12 — t0 = address of GOT[0]
        b.extend_from_slice(&encode_rv_addi(5, 7, lo).to_le_bytes());
        // srli t1(x6), t1(x6), 1 — PLT index bookkeeping
        b.extend_from_slice(&encode_rv_srli(6, 6, 1).to_le_bytes());
        // ld t0(x5), 8(t0/x5) — load link_map from GOT[1] = GOT[0]+8
        b.extend_from_slice(&encode_rv_ld(5, 5, 8).to_le_bytes());
        // jr t3(x28) — jump to resolver (_dl_runtime_resolve)
        b.extend_from_slice(&encode_rv_jalr(0, 28, 0).to_le_bytes());
        debug_assert_eq!(b.len(), 32);
        b
    }

    fn pltn_riscv64(&self, s: &PltStub, gpa: u64, pa: u64, po: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(16);
        let ea = pa + po;
        let ge = gpa + s.got_plt_offset;
        let off = ge as i64 - ea as i64;
        let (hi, lo) = riscv_hi20_lo12(off as i32);
        // auipc t3(x28), hi20
        b.extend_from_slice(&encode_rv_auipc(28, hi).to_le_bytes());
        // ld t3(x28), lo12(t3/x28)
        b.extend_from_slice(&encode_rv_ld(28, 28, lo).to_le_bytes());
        // jalr t1(x6), t3(x28), 0
        b.extend_from_slice(&encode_rv_jalr(6, 28, 0).to_le_bytes());
        // nop
        b.extend_from_slice(&encode_rv_addi(0, 0, 0).to_le_bytes());
        debug_assert_eq!(b.len(), 16);
        b
    }

    /// Encode the complete `.plt` section (PLT0 + all PLTn stubs).
    pub fn encode(&self, got_plt_addr: u64, plt_addr: u64) -> Vec<u8> {
        let (p0, pn) = plt_sizes(&self.target);
        let mut buf = Vec::with_capacity(p0 + self.stubs.len() * pn);
        buf.extend_from_slice(&self.generate_plt0(got_plt_addr, plt_addr));
        for (i, stub) in self.stubs.iter().enumerate() {
            let off = p0 as u64 + i as u64 * pn as u64;
            buf.extend_from_slice(&self.generate_plt_entry(stub, got_plt_addr, plt_addr, off));
        }
        buf
    }
}

// ===========================================================================
// AArch64 Instruction Encoding Helpers
// ===========================================================================

fn aarch64_page(addr: u64) -> u64 {
    addr & !0xFFF
}

fn encode_aarch64_adrp(rd: u32, imm: i32) -> u32 {
    let imm21 = imm as u32 & 0x1F_FFFF;
    (1u32 << 31)
        | ((imm21 & 3) << 29)
        | (0b10000 << 24)
        | (((imm21 >> 2) & 0x7_FFFF) << 5)
        | (rd & 0x1f)
}

fn encode_aarch64_ldr64(rt: u32, rn: u32, pimm: u32) -> u32 {
    // LDR X<rt>, [X<rn>, #<pimm>] — opc=11 V=1 size=11 => 0xF9400000 base
    (0xF940_0000u32) | ((pimm & 0xFFF) << 10) | ((rn & 0x1f) << 5) | (rt & 0x1f)
}

fn encode_aarch64_add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    // ADD X<rd>, X<rn>, #<imm12> — sf=1 op=0 S=0 => 0x91000000 base
    (0x9100_0000u32) | ((imm12 & 0xFFF) << 10) | ((rn & 0x1f) << 5) | (rd & 0x1f)
}

// ===========================================================================
// RISC-V 64 Instruction Encoding Helpers
// ===========================================================================

fn riscv_hi20_lo12(offset: i32) -> (i32, i32) {
    let lo = (offset << 20) >> 20;
    let hi = offset.wrapping_sub(lo) >> 12;
    (hi, lo)
}

fn encode_rv_auipc(rd: u32, imm20: i32) -> u32 {
    ((imm20 as u32) << 12) | ((rd & 0x1f) << 7) | 0b0010111
}

fn encode_rv_sub(rd: u32, rs1: u32, rs2: u32) -> u32 {
    (0b0100000u32 << 25)
        | ((rs2 & 0x1f) << 20)
        | ((rs1 & 0x1f) << 15)
        | ((rd & 0x1f) << 7)
        | 0b0110011
}

fn encode_rv_ld(rd: u32, rs1: u32, imm12: i32) -> u32 {
    (((imm12 as u32) & 0xFFF) << 20)
        | ((rs1 & 0x1f) << 15)
        | (0b011 << 12)
        | ((rd & 0x1f) << 7)
        | 0b0000011
}

fn encode_rv_addi(rd: u32, rs1: u32, imm12: i32) -> u32 {
    (((imm12 as u32) & 0xFFF) << 20) | ((rs1 & 0x1f) << 15) | ((rd & 0x1f) << 7) | 0b0010011
}

fn encode_rv_srli(rd: u32, rs1: u32, shamt: u32) -> u32 {
    ((shamt & 0x3f) << 20) | ((rs1 & 0x1f) << 15) | (0b101 << 12) | ((rd & 0x1f) << 7) | 0b0010011
}

fn encode_rv_jalr(rd: u32, rs1: u32, imm12: i32) -> u32 {
    (((imm12 as u32) & 0xFFF) << 20) | ((rs1 & 0x1f) << 15) | ((rd & 0x1f) << 7) | 0b1100111
}

// ===========================================================================
// Dynamic Relocations (.rela.dyn / .rela.plt)
// ===========================================================================

/// A single dynamic relocation entry.
#[derive(Debug, Clone)]
pub struct DynamicRelocation {
    pub offset: u64,
    pub sym_index: u32,
    pub rel_type: u32,
    pub addend: i64,
}

/// Builder for `.rela.dyn` and `.rela.plt` sections.
pub struct DynamicRelocationTable {
    rela_dyn: Vec<DynamicRelocation>,
    rela_plt: Vec<DynamicRelocation>,
}

impl Default for DynamicRelocationTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicRelocationTable {
    pub fn new() -> Self {
        Self {
            rela_dyn: Vec::new(),
            rela_plt: Vec::new(),
        }
    }

    pub fn add_rela_dyn(&mut self, reloc: DynamicRelocation) {
        self.rela_dyn.push(reloc);
    }

    pub fn add_rela_plt(&mut self, reloc: DynamicRelocation) {
        self.rela_plt.push(reloc);
    }

    /// Adjust all `.rela.plt` entry offsets from section-relative (within
    /// `.got.plt`) to absolute virtual addresses by adding `got_plt_vaddr`.
    /// The dynamic linker expects `r_offset` to be the absolute address
    /// of the GOT.PLT slot, not a section-relative byte offset.
    pub fn patch_rela_plt_offsets(&mut self, got_plt_vaddr: u64) {
        for r in &mut self.rela_plt {
            r.offset += got_plt_vaddr;
        }
    }

    /// Adjust all `.rela.dyn` entry offsets from section-relative to
    /// absolute virtual addresses.
    pub fn patch_rela_dyn_offsets(&mut self, got_vaddr: u64) {
        for r in &mut self.rela_dyn {
            r.offset += got_vaddr;
        }
    }

    /// Return a mutable slice of `.rela.dyn` entries so Phase 9 can patch
    /// offsets and addends after the address layout is finalized.
    pub fn rela_dyn_mut(&mut self) -> &mut [DynamicRelocation] {
        &mut self.rela_dyn
    }

    pub fn encode_rela_dyn(&self, is_64bit: bool) -> Vec<u8> {
        encode_relocations(&self.rela_dyn, is_64bit)
    }

    pub fn encode_rela_plt(&self, is_64bit: bool) -> Vec<u8> {
        encode_relocations(&self.rela_plt, is_64bit)
    }
}

/// Serialise a list of dynamic relocations in ELF RELA (64-bit) or REL (i686) format.
fn encode_relocations(relocs: &[DynamicRelocation], is_64bit: bool) -> Vec<u8> {
    let entry_sz = if is_64bit { 24 } else { 8 };
    let mut buf = Vec::with_capacity(relocs.len() * entry_sz);
    if is_64bit {
        for r in relocs {
            let e = Elf64Rela::new(r.offset, r.sym_index, r.rel_type, r.addend);
            buf.extend_from_slice(&e.to_bytes());
        }
    } else {
        // i686 uses Elf32_Rel (8 bytes, no addend) — the glibc i386
        // dynamic linker asserts DT_PLTREL == DT_REL.
        for r in relocs {
            let e = Elf32Rel::new(r.offset as u32, r.sym_index, r.rel_type as u8);
            buf.extend_from_slice(&e.to_bytes());
        }
    }
    buf
}

// ===========================================================================
// .dynamic Section
// ===========================================================================

/// A single entry in the `.dynamic` section (tag + value).
#[derive(Debug, Clone)]
pub struct DynamicEntry {
    pub tag: i64,
    pub value: u64,
}

/// Builder for the `.dynamic` section.
pub struct DynamicSection {
    entries: Vec<DynamicEntry>,
}

impl Default for DynamicSection {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicSection {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, tag: i64, value: u64) {
        self.entries.push(DynamicEntry { tag, value });
    }

    /// Construct the `.dynamic` section from a fully finalised `DynamicLinkContext`.
    ///
    /// All address values (section base addresses) are set to zero here and
    /// must be patched by the linker once final segment layout is determined.
    pub fn build(ctx: &DynamicLinkContext) -> Self {
        let mut s = Self::new();
        let is64 = ctx.target.is_64bit();

        // Symbol table metadata
        s.add_entry(DT_STRTAB, 0);
        s.add_entry(DT_STRSZ, ctx.dynsym.encode_dynstr().len() as u64);
        s.add_entry(DT_SYMTAB, 0);
        s.add_entry(DT_SYMENT, if is64 { 24 } else { 16 });
        s.add_entry(DT_GNU_HASH, 0);

        // Relocations
        let rela_dyn_bytes = ctx.rela.encode_rela_dyn(is64);
        let rela_plt_bytes = ctx.rela.encode_rela_plt(is64);

        if is64 {
            if !rela_dyn_bytes.is_empty() {
                s.add_entry(DT_RELA, 0);
                s.add_entry(DT_RELASZ, rela_dyn_bytes.len() as u64);
                s.add_entry(DT_RELAENT, 24);
            }
            if !rela_plt_bytes.is_empty() {
                s.add_entry(DT_JMPREL, 0);
                s.add_entry(DT_PLTRELSZ, rela_plt_bytes.len() as u64);
                s.add_entry(DT_PLTREL, DT_RELA as u64);
                s.add_entry(DT_PLTGOT, 0);
            }
        } else {
            // 32-bit ELF (i686) uses REL entries (8 bytes, no addend)
            // because the i386 glibc dynamic linker asserts DT_PLTREL==DT_REL.
            if !rela_dyn_bytes.is_empty() {
                s.add_entry(DT_REL, 0);
                s.add_entry(DT_RELSZ, rela_dyn_bytes.len() as u64);
                s.add_entry(DT_RELENT, 8);
            }
            if !rela_plt_bytes.is_empty() {
                s.add_entry(DT_JMPREL, 0);
                s.add_entry(DT_PLTRELSZ, rela_plt_bytes.len() as u64);
                s.add_entry(DT_PLTREL, DT_REL as u64);
                s.add_entry(DT_PLTGOT, 0);
            }
        }

        // Needed libraries
        for _lib in &ctx.needed_libs {
            s.add_entry(DT_NEEDED, 0);
        }

        // SONAME
        if ctx.soname.is_some() {
            s.add_entry(DT_SONAME, 0);
        }

        // Flags
        s.add_entry(DT_FLAGS, 0);
        s.add_entry(DT_FLAGS_1, 0);

        // Debug marker for non-shared (executable) objects
        if !ctx.is_shared {
            s.add_entry(DT_DEBUG, 0);
        }

        // Terminator (required)
        s.add_entry(DT_NULL, 0);
        s
    }

    /// Patch the first entry matching `tag` to have the given `value`.
    ///
    /// Used by the linker to fill in section base addresses after layout.
    pub fn patch_address(&mut self, tag: i64, value: u64) {
        for e in &mut self.entries {
            if e.tag == tag {
                e.value = value;
                return;
            }
        }
    }

    /// Patch the `DT_NEEDED` entries with the correct `.dynstr` offsets for
    /// each needed library name.
    ///
    /// Relies on the `.dynstr` table having been built by the same
    /// `DynamicSymbolTable` that was used during `build()`.
    pub fn patch_needed_libs(&mut self, lib_names: &[String], dynstr_data: &[u8]) {
        let mut lib_iter = lib_names.iter();
        for e in &mut self.entries {
            if e.tag == DT_NEEDED {
                if let Some(lib_name) = lib_iter.next() {
                    // Find the library name in the dynstr blob.
                    let name_bytes = lib_name.as_bytes();
                    if let Some(pos) = dynstr_data
                        .windows(name_bytes.len() + 1)
                        .position(|w| w.starts_with(name_bytes) && w[name_bytes.len()] == 0)
                    {
                        e.value = pos as u64;
                    }
                }
            }
        }
    }

    /// Encode the `.dynamic` section into a byte vector.
    pub fn encode(&self, is_64bit: bool) -> Vec<u8> {
        if is_64bit {
            let mut buf = Vec::with_capacity(self.entries.len() * 16);
            for e in &self.entries {
                buf.extend_from_slice(&e.tag.to_le_bytes());
                buf.extend_from_slice(&e.value.to_le_bytes());
            }
            buf
        } else {
            let mut buf = Vec::with_capacity(self.entries.len() * 8);
            for e in &self.entries {
                buf.extend_from_slice(&(e.tag as i32).to_le_bytes());
                buf.extend_from_slice(&(e.value as u32).to_le_bytes());
            }
            buf
        }
    }
}

// ===========================================================================
// DynamicLinkContext — top-level orchestrator
// ===========================================================================

/// Orchestrates the generation of all dynamic linking ELF sections.
///
/// Holds the accumulated state for `.dynsym`, `.dynstr`, `.gnu.hash`,
/// `.got`, `.got.plt`, `.plt`, `.rela.dyn`, `.rela.plt`, `.dynamic`
/// and the `PT_INTERP` program header content.
pub struct DynamicLinkContext {
    pub target: Target,
    pub is_shared: bool,
    pub soname: Option<String>,
    pub needed_libs: Vec<String>,
    pub dynsym: DynamicSymbolTable,
    pub gnu_hash: GnuHashTable,
    pub got: GlobalOffsetTable,
    pub plt: ProcedureLinkageTable,
    pub rela: DynamicRelocationTable,
    pub dynamic: DynamicSection,
    pub interp: Option<String>,
}

impl DynamicLinkContext {
    /// Create a new context for the given target and shared-object flag.
    pub fn new(target: Target, is_shared: bool) -> Self {
        Self {
            target,
            is_shared,
            soname: None,
            needed_libs: Vec::new(),
            dynsym: DynamicSymbolTable::new(),
            gnu_hash: GnuHashTable {
                bucket_count: 0,
                symoffset: 0,
                bloom_size: 0,
                bloom_shift: 0,
                bloom: Vec::new(),
                buckets: Vec::new(),
                chains: Vec::new(),
            },
            got: GlobalOffsetTable::new(target),
            plt: ProcedureLinkageTable::new(target),
            rela: DynamicRelocationTable::new(),
            dynamic: DynamicSection::new(),
            interp: None,
        }
    }

    /// Set the `PT_INTERP` path from the target's dynamic linker.
    pub fn set_interp(&mut self) {
        self.interp = Some(self.target.dynamic_linker().to_string());
    }

    /// Record a needed shared library (`DT_NEEDED`).
    pub fn add_needed_library(&mut self, name: &str) {
        self.needed_libs.push(name.to_string());
    }

    /// Build all section contents once symbol / relocation data has been
    /// collected. After this call the various `encode_*()` helpers can be
    /// used to obtain the raw byte vectors.
    pub fn finalize(&mut self) {
        // Add needed library names to .dynstr BEFORE building .dynamic,
        // so that patch_needed_libs can find them in the string table.
        for lib in &self.needed_libs.clone() {
            self.dynsym.add_dynstr_string(lib);
        }

        // Build .gnu.hash from the symbol table, reordering symbols to
        // satisfy the contiguous-bucket layout required by the dynamic linker.
        self.gnu_hash = GnuHashTable::build_and_reorder(&mut self.dynsym.symbols);

        // Build .dynamic from accumulated metadata
        self.dynamic = DynamicSection::build(self);

        // Patch DT_NEEDED entries with the correct dynstr offsets
        let dynstr_data = self.dynsym.encode_dynstr();
        self.dynamic
            .patch_needed_libs(&self.needed_libs, &dynstr_data);
    }

    /// Return the null-terminated `PT_INTERP` content, if one was set.
    pub fn get_interp_bytes(&self) -> Option<Vec<u8>> {
        self.interp.as_ref().map(|p| {
            let mut v = p.as_bytes().to_vec();
            v.push(0);
            v
        })
    }
}

// ===========================================================================
// DynamicLinkResult — public output type
// ===========================================================================

/// The final encoded bytes for every dynamic-linking ELF section.
pub struct DynamicLinkResult {
    pub dynamic_section: Vec<u8>,
    pub dynsym_section: Vec<u8>,
    pub dynstr_section: Vec<u8>,
    pub gnu_hash_section: Vec<u8>,
    pub got_section: Vec<u8>,
    pub got_plt_section: Vec<u8>,
    pub plt_section: Vec<u8>,
    pub rela_dyn_section: Vec<u8>,
    pub rela_plt_section: Vec<u8>,
    pub interp: Option<Vec<u8>>,
}

// ===========================================================================
// Public API — build_dynamic_sections
// ===========================================================================

/// Create all ELF dynamic-linking sections for a shared object
/// (`-shared`) or dynamically-linked executable.
///
/// # Arguments
///
/// * `target` — Target architecture.
/// * `is_shared` — `true` for `ET_DYN` (shared lib), `false` for exec with dynamic deps.
/// * `exported_symbols` — Symbols defined here and visible to the dynamic linker.
/// * `imported_symbols` — Symbols that must be resolved at load time.
/// * `needed_libs` — List of shared libraries required (`DT_NEEDED`).
/// * `soname` — Optional `DT_SONAME` value.
pub fn build_dynamic_sections(
    target: &Target,
    is_shared: bool,
    exported_symbols: &[ExportedSymbol],
    imported_symbols: &[ImportedSymbol],
    needed_libs: &[String],
    soname: Option<&str>,
) -> DynamicLinkResult {
    let mut ctx = DynamicLinkContext::new(*target, is_shared);

    // Optionally set SONAME
    if let Some(sn) = soname {
        ctx.soname = Some(sn.to_string());
    }

    // Record needed libraries
    for lib in needed_libs {
        ctx.add_needed_library(lib);
    }

    // Set interpreter for non-shared executables
    if !is_shared {
        ctx.set_interp();
    }

    // Add all exported (defined) symbols to .dynsym
    for sym in exported_symbols {
        let ds = DynamicSymbol {
            name: sym.name.clone(),
            value: sym.value,
            size: sym.size,
            binding: sym.binding,
            sym_type: sym.sym_type,
            visibility: sym.visibility,
            section_index: sym.section_index,
            is_defined: true,
            is_plt_entry: false,
            got_offset: None,
            plt_index: None,
        };
        ctx.dynsym.add_symbol(ds);
    }

    // Add all imported (undefined) symbols — give them GOT/PLT entries as needed.
    // Data symbols that need copy relocations are handled separately below.
    for sym in imported_symbols {
        // Copy-relocated data symbols (stderr, stdout, stdin, etc.) are
        // handled after the main imported-symbol loop — they need
        // R_*_COPY in .rela.dyn and no PLT/GOT.PLT entries.
        if sym.needs_copy {
            continue;
        }

        // Trust the needs_plt flag computed from relocation types — do NOT
        // re-filter on sym_type because GCC .o files mark external
        // functions as STT_NOTYPE rather than STT_FUNC.
        let needs_plt = sym.needs_plt;
        // If the symbol needs PLT, it is a function — upgrade its type
        // for the .dynsym entry so the dynamic linker knows.
        let effective_sym_type = if needs_plt && sym.sym_type == STT_NOTYPE {
            STT_FUNC
        } else {
            sym.sym_type
        };
        let mut ds = DynamicSymbol {
            name: sym.name.clone(),
            value: 0,
            size: 0,
            binding: sym.binding,
            sym_type: effective_sym_type,
            visibility: STV_DEFAULT,
            section_index: 0, // SHN_UNDEF
            is_defined: false,
            is_plt_entry: needs_plt,
            got_offset: None,
            plt_index: None,
        };

        // Allocate a GOT entry for every imported symbol
        let reloc_type = default_glob_dat_reloc(*target);
        let goff = ctx.got.add_got_entry(&sym.name, reloc_type);
        ds.got_offset = Some(goff);

        // For function imports that need lazy binding → GOT.PLT + PLT stub
        if needs_plt {
            let (gp_off, plt_idx) = ctx.got.add_got_plt_entry(&sym.name);
            ds.plt_index = Some(plt_idx);

            ctx.plt.stubs.push(PltStub {
                symbol_name: sym.name.clone(),
                got_plt_offset: gp_off,
                index: plt_idx,
            });
        }

        let sym_idx = ctx.dynsym.add_symbol(ds);

        // Emit .rela.dyn relocation for the GOT entry
        ctx.rela.add_rela_dyn(DynamicRelocation {
            offset: goff,
            sym_index: sym_idx,
            rel_type: reloc_type,
            addend: 0,
        });

        // Emit .rela.plt relocation for the GOT.PLT entry (JUMP_SLOT)
        if needs_plt {
            let js_type = default_jump_slot_reloc(*target);
            let gp = ctx.got.got_plt_entries.last().unwrap();
            ctx.rela.add_rela_plt(DynamicRelocation {
                offset: gp.offset,
                sym_index: sym_idx,
                rel_type: js_type,
                addend: 0,
            });
        }
    }

    // Add copy-relocated data symbols (stderr, stdout, stdin, environ, etc.).
    // These symbols are placed in .bss.copy in the executable and the dynamic
    // linker copies their initial values from the shared library at load time
    // via R_*_COPY relocations. They do NOT get PLT/GOT.PLT entries.
    let copy_reloc_type = default_copy_reloc(*target);
    for sym in imported_symbols {
        if !sym.needs_copy {
            continue;
        }

        // Data symbols are added to .dynsym as STT_OBJECT with a
        // placeholder value of 0. The caller (linker_common::link) will
        // patch the value to the .bss.copy VA after layout.
        let ds = DynamicSymbol {
            name: sym.name.clone(),
            value: 0, // patched to .bss.copy VA later
            size: sym.copy_size,
            binding: sym.binding,
            sym_type: STT_OBJECT,
            visibility: STV_DEFAULT,
            section_index: 0,  // will be patched to .bss.copy shndx later
            is_defined: false, // UND — ld.so resolves from shared lib
            is_plt_entry: false,
            got_offset: None,
            plt_index: None,
        };

        let sym_idx = ctx.dynsym.add_symbol(ds);

        // Emit R_*_COPY in .rela.dyn. The r_offset is 0 here — the
        // caller patches it to the .bss.copy VA after layout.
        ctx.rela.add_rela_dyn(DynamicRelocation {
            offset: 0, // patched by linker_common::link
            sym_index: sym_idx,
            rel_type: copy_reloc_type,
            addend: 0,
        });
    }

    // Finalise (builds .gnu.hash and .dynamic from accumulated state)
    ctx.finalize();

    let is64 = target.is_64bit();

    // Encode all section byte streams
    DynamicLinkResult {
        dynamic_section: ctx.dynamic.encode(is64),
        dynsym_section: ctx.dynsym.encode_dynsym(target),
        dynstr_section: ctx.dynsym.encode_dynstr(),
        gnu_hash_section: ctx.gnu_hash.encode(),
        got_section: ctx.got.encode_got(),
        got_plt_section: ctx.got.encode_got_plt(0), // base patched by linker
        plt_section: ctx.plt.encode(0, 0),          // bases patched by linker
        rela_dyn_section: ctx.rela.encode_rela_dyn(is64),
        rela_plt_section: ctx.rela.encode_rela_plt(is64),
        interp: ctx.get_interp_bytes(),
    }
}

// ===========================================================================
// Internal Helpers
// ===========================================================================

/// Returns the `R_*_GLOB_DAT` relocation type for the given target.
fn default_glob_dat_reloc(target: Target) -> u32 {
    match target {
        Target::X86_64 => 6,     // R_X86_64_GLOB_DAT
        Target::I686 => 6,       // R_386_GLOB_DAT
        Target::AArch64 => 1025, // R_AARCH64_GLOB_DAT
        Target::RiscV64 => 2,    // R_RISCV_64 (used as GLOB_DAT equivalent)
    }
}

/// Returns the `R_*_JUMP_SLOT` relocation type for the given target.
pub fn default_jump_slot_reloc(target: Target) -> u32 {
    match target {
        Target::X86_64 => 7,     // R_X86_64_JUMP_SLOT
        Target::I686 => 7,       // R_386_JMP_SLOT
        Target::AArch64 => 1026, // R_AARCH64_JUMP_SLOT
        Target::RiscV64 => 5,    // R_RISCV_JUMP_SLOT
    }
}

/// Returns the `R_*_COPY` relocation type for the given target.
/// Copy relocations instruct the dynamic linker to copy the initial
/// value of a data symbol from a shared library into the executable's
/// `.bss.copy` segment at load time.
pub fn default_copy_reloc(target: Target) -> u32 {
    match target {
        Target::X86_64 => 5,     // R_X86_64_COPY
        Target::I686 => 5,       // R_386_COPY
        Target::AArch64 => 1024, // R_AARCH64_COPY
        Target::RiscV64 => 4,    // R_RISCV_COPY
    }
}

/// Return `(plt0_size, pltn_size)` for the given target.
pub fn plt_sizes(target: &Target) -> (usize, usize) {
    match target {
        Target::X86_64 | Target::I686 => (16, 16),
        Target::AArch64 | Target::RiscV64 => (32, 16),
    }
}
