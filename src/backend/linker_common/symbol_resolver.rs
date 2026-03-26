//! Two-pass symbol resolution engine for BCC's built-in ELF linker.
//!
//! This module implements the core symbol resolution algorithm used by all four
//! architecture backends (x86-64, i686, AArch64, RISC-V 64). It follows
//! standard ELF semantics:
//!
//! - **Strong global symbols** (`STB_GLOBAL` with a definition) override weak
//!   symbols; duplicate strong definitions produce a "multiple definition" error.
//! - **Weak symbols** (`STB_WEAK`) can be overridden by strong symbols without
//!   error; if only weak definitions exist, any one is chosen.
//! - **Local symbols** (`STB_LOCAL`) are visible only within their originating
//!   object file and never participate in cross-file resolution.
//! - **Symbol visibility** (`STV_DEFAULT`, `STV_HIDDEN`, `STV_PROTECTED`,
//!   `STV_INTERNAL`) is merged by choosing the most restrictive level.
//!
//! The resolution algorithm has two passes:
//! 1. **Collection** — gather all symbols from all input object files, grouping
//!    globals/weaks by name and storing locals separately.
//! 2. **Resolution** — for each unique symbol name, apply binding precedence
//!    rules, merge visibility, and produce a single [`ResolvedSymbol`].
//!
//! Archive scanning follows the standard iterative algorithm: pull in archive
//! members that define currently-undefined symbols, repeating until a fixpoint.
//!
//! # Zero-Dependency
//!
//! This module uses only the Rust standard library and `crate::common` modules.
//! No external crates are permitted per the zero-dependency mandate.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;

// ---------------------------------------------------------------------------
// ELF Symbol Binding Constants (from elf.h)
// ---------------------------------------------------------------------------

/// Local symbol — visible only within the object file that defines it.
pub const STB_LOCAL: u8 = 0;

/// Global symbol — visible to all object files being combined.
pub const STB_GLOBAL: u8 = 1;

/// Weak symbol — like global but with lower precedence; can be overridden
/// by a strong (global) definition without error.
pub const STB_WEAK: u8 = 2;

// ---------------------------------------------------------------------------
// ELF Symbol Type Constants (from elf.h)
// ---------------------------------------------------------------------------

/// Symbol type is not specified.
pub const STT_NOTYPE: u8 = 0;

/// Symbol is a data object (variable, array, etc.).
pub const STT_OBJECT: u8 = 1;

/// Symbol is a function or other executable code.
pub const STT_FUNC: u8 = 2;

/// Symbol is associated with a section (used internally by the linker).
pub const STT_SECTION: u8 = 3;

/// Symbol gives the name of the source file associated with the object.
pub const STT_FILE: u8 = 4;

/// Symbol labels a common block (uninitialized data with COMMON allocation).
pub const STT_COMMON: u8 = 5;

/// Symbol is a thread-local storage entity.
pub const STT_TLS: u8 = 6;

// ---------------------------------------------------------------------------
// ELF Symbol Visibility Constants (from elf.h)
// ---------------------------------------------------------------------------

/// Default visibility — symbol is visible according to its binding.
pub const STV_DEFAULT: u8 = 0;

/// Internal visibility — most restrictive; reserved for processor-specific
/// semantics. Treated like hidden but may have additional constraints.
pub const STV_INTERNAL: u8 = 1;

/// Hidden visibility — symbol is not visible outside the defining
/// shared object or executable.
pub const STV_HIDDEN: u8 = 2;

/// Protected visibility — symbol is visible in other components but
/// cannot be preempted (references from within the defining component
/// always bind to the local definition).
pub const STV_PROTECTED: u8 = 3;

// ---------------------------------------------------------------------------
// Special Section Index Constants (from elf.h)
// ---------------------------------------------------------------------------

/// Undefined section — the symbol is referenced but not defined in this
/// object file.
pub const SHN_UNDEF: u16 = 0;

/// Absolute symbol — the symbol has an absolute value that will not change
/// because of relocation.
pub const SHN_ABS: u16 = 0xFFF1;

/// Common symbol — the symbol labels a common block that has not yet been
/// allocated. The linker allocates storage in `.bss`.
pub const SHN_COMMON: u16 = 0xFFF2;

// ---------------------------------------------------------------------------
// InputSymbol — symbol as read from an input object file
// ---------------------------------------------------------------------------

/// Represents a symbol extracted from an input relocatable object file (`.o`).
///
/// Each input object contributes a list of `InputSymbol`s. Local symbols
/// are stored separately and never participate in cross-file resolution.
/// Global and weak symbols are grouped by name for the resolution pass.
#[derive(Debug, Clone)]
pub struct InputSymbol {
    /// The symbol's name (e.g. `"main"`, `"printf"`, `"__bss_start"`).
    pub name: String,

    /// Symbol value — typically the offset within its containing section.
    pub value: u64,

    /// Size of the symbol's associated data or code in bytes.
    pub size: u64,

    /// Binding attribute: [`STB_LOCAL`], [`STB_GLOBAL`], or [`STB_WEAK`].
    pub binding: u8,

    /// Type attribute: [`STT_FUNC`], [`STT_OBJECT`], [`STT_NOTYPE`], etc.
    pub sym_type: u8,

    /// Visibility attribute: [`STV_DEFAULT`], [`STV_HIDDEN`], etc.
    pub visibility: u8,

    /// Section header index of the section containing this symbol.
    /// [`SHN_UNDEF`] if the symbol is undefined (referenced but not defined).
    /// [`SHN_ABS`] for absolute symbols.
    /// [`SHN_COMMON`] for common symbols.
    pub section_index: u16,

    /// Identifier of the input object file that contributed this symbol.
    /// Used for error messages and provenance tracking.
    pub object_file_id: u32,
}

impl InputSymbol {
    /// Returns `true` if this symbol has a definition (is not undefined).
    #[inline]
    pub fn is_defined(&self) -> bool {
        self.section_index != SHN_UNDEF
    }

    /// Returns `true` if this symbol has global binding (`STB_GLOBAL`).
    #[inline]
    pub fn is_global(&self) -> bool {
        self.binding == STB_GLOBAL
    }

    /// Returns `true` if this symbol has weak binding (`STB_WEAK`).
    #[inline]
    pub fn is_weak(&self) -> bool {
        self.binding == STB_WEAK
    }

    /// Returns `true` if this symbol has local binding (`STB_LOCAL`).
    #[inline]
    pub fn is_local(&self) -> bool {
        self.binding == STB_LOCAL
    }

    /// Returns `true` if this symbol is a function (`STT_FUNC`).
    #[inline]
    pub fn is_function(&self) -> bool {
        self.sym_type == STT_FUNC
    }

    /// Returns `true` if this symbol has hidden visibility (`STV_HIDDEN`).
    #[inline]
    pub fn is_hidden(&self) -> bool {
        self.visibility == STV_HIDDEN
    }
}

// ---------------------------------------------------------------------------
// ResolvedSymbol — the final resolved symbol after two-pass resolution
// ---------------------------------------------------------------------------

/// A symbol after cross-file resolution, with a definitive binding,
/// visibility, and provenance.
///
/// `ResolvedSymbol` instances are produced by [`SymbolResolver::resolve`]
/// and consumed by the ELF writer and architecture-specific linkers to
/// produce the output symbol table.
#[derive(Debug, Clone)]
pub struct ResolvedSymbol {
    /// The symbol's name.
    pub name: String,

    /// Absolute virtual address after section layout. Initially 0 until
    /// the linker performs address assignment.
    pub final_address: u64,

    /// Size in bytes.
    pub size: u64,

    /// Binding attribute after resolution: [`STB_GLOBAL`] or [`STB_WEAK`].
    pub binding: u8,

    /// Type attribute: [`STT_FUNC`], [`STT_OBJECT`], etc.
    pub sym_type: u8,

    /// Merged visibility (most restrictive wins).
    pub visibility: u8,

    /// Name of the output section containing this symbol (e.g. `".text"`).
    pub section_name: String,

    /// Whether this symbol has a definition.
    pub is_defined: bool,

    /// Object file ID that provided the winning definition.
    pub from_object: u32,

    /// If `true`, this symbol should appear in `.dynsym` for shared
    /// library export.
    pub export_dynamic: bool,
}

// ---------------------------------------------------------------------------
// OutputSymbol — symbol for the final ELF symbol table
// ---------------------------------------------------------------------------

/// A symbol ready for emission into the ELF `.symtab` section.
///
/// Local symbols appear first (indices 0..`first_global_index`), followed
/// by global and weak symbols, per the ELF specification requirement that
/// `sh_info` equals the index of the first non-local symbol.
#[derive(Debug, Clone)]
pub struct OutputSymbol {
    /// Symbol name.
    pub name: String,

    /// Symbol value (absolute virtual address or section-relative offset).
    pub value: u64,

    /// Symbol size in bytes.
    pub size: u64,

    /// Binding: [`STB_LOCAL`], [`STB_GLOBAL`], or [`STB_WEAK`].
    pub binding: u8,

    /// Type: [`STT_FUNC`], [`STT_OBJECT`], [`STT_NOTYPE`], etc.
    pub sym_type: u8,

    /// Visibility: [`STV_DEFAULT`], [`STV_HIDDEN`], etc.
    pub visibility: u8,

    /// Section header table index for the section containing this symbol.
    pub section_index: u16,
}

// ---------------------------------------------------------------------------
// SymbolTable — final symbol table for ELF emission
// ---------------------------------------------------------------------------

/// Complete symbol table ready for ELF `.symtab` emission.
///
/// Symbols are ordered with all locals first, then all globals/weaks.
/// The `first_global_index` field becomes the `sh_info` value of the
/// `.symtab` section header.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Ordered list of output symbols (locals first, then globals).
    pub symbols: Vec<OutputSymbol>,

    /// Index of the first non-local (global or weak) symbol.
    /// This value is written as `sh_info` in the `.symtab` section header.
    pub first_global_index: u32,
}

// ---------------------------------------------------------------------------
// ArchiveMember — a single member within a static library (.a)
// ---------------------------------------------------------------------------

/// Represents one object file within a static library archive (`.a`).
///
/// Archive scanning inspects each member's symbols to determine whether
/// the member should be included in the link (i.e. it defines a symbol
/// that is currently undefined).
#[derive(Debug, Clone)]
pub struct ArchiveMember {
    /// Index of this member within the archive (zero-based).
    pub index: u32,

    /// All symbols defined or referenced by this archive member.
    pub symbols: Vec<InputSymbol>,
}

// ---------------------------------------------------------------------------
// ObjectSymbols — symbols from a single input object file
// ---------------------------------------------------------------------------

/// Bundles all symbols from one input relocatable object file with the
/// object's unique identifier.
#[derive(Debug, Clone)]
pub struct ObjectSymbols {
    /// Unique identifier for this input object file.
    pub object_id: u32,

    /// All symbols (local, global, weak) from this object file.
    pub symbols: Vec<InputSymbol>,
}

// ---------------------------------------------------------------------------
// Archive — a static library containing multiple object files
// ---------------------------------------------------------------------------

/// Represents a static library archive (`.a` file) containing multiple
/// relocatable object files.
#[derive(Debug, Clone)]
pub struct Archive {
    /// The archive members (object files) within this archive.
    pub members: Vec<ArchiveMember>,
}

// ---------------------------------------------------------------------------
// SymbolResolutionResult — output of the complete resolution process
// ---------------------------------------------------------------------------

/// The complete result of symbol resolution, containing the final symbol
/// table, all resolved symbols, undefined weak symbols, and information
/// about which archive members were included.
#[derive(Debug)]
pub struct SymbolResolutionResult {
    /// The final symbol table for ELF emission.
    pub symbol_table: SymbolTable,

    /// Map from symbol name to its resolved definition.
    pub resolved: FxHashMap<String, ResolvedSymbol>,

    /// Set of weak symbol names that remain undefined. These resolve to
    /// address 0 (NULL) at runtime and are not errors.
    pub undefined_weak: FxHashSet<String>,

    /// Archive members that were pulled into the link. Each entry is
    /// `(archive_index, member_indices)` where `archive_index` is the
    /// position of the archive in the input archive list.
    pub included_archive_members: Vec<(usize, Vec<u32>)>,
}

// ---------------------------------------------------------------------------
// Visibility merging helper
// ---------------------------------------------------------------------------

/// Merge two ELF symbol visibility values by selecting the most restrictive.
///
/// The restrictiveness ordering (most to least) is:
/// `STV_INTERNAL` > `STV_HIDDEN` > `STV_PROTECTED` > `STV_DEFAULT`.
///
/// If either operand is `STV_DEFAULT`, the other's visibility wins.
/// Among non-default values, the numerically smaller value is more
/// restrictive (INTERNAL=1 < HIDDEN=2 < PROTECTED=3).
fn merge_visibility(a: u8, b: u8) -> u8 {
    if a == STV_DEFAULT {
        return b;
    }
    if b == STV_DEFAULT {
        return a;
    }
    // Both non-default: the smaller numeric value is more restrictive.
    a.min(b)
}

/// Returns the default executable base load address for a given target
/// architecture. This is used by the linker to set up the virtual address
/// space origin for `ET_EXEC` executables.
///
/// | Architecture | Base Address |
/// |-------------|-------------|
/// | x86-64      | `0x400000`   |
/// | i686        | `0x08048000` |
/// | AArch64     | `0x400000`   |
/// | RISC-V 64   | `0x10000`    |
pub fn default_base_address(target: Target) -> u64 {
    match target {
        Target::X86_64 => 0x0040_0000,
        Target::I686 => 0x0804_8000,
        Target::AArch64 => 0x0040_0000,
        Target::RiscV64 => 0x0001_0000,
    }
}

// ---------------------------------------------------------------------------
// SymbolResolver — the core resolution engine
// ---------------------------------------------------------------------------

/// Two-pass symbol resolution engine.
///
/// Usage pattern:
/// 1. Create with [`SymbolResolver::new`].
/// 2. Call [`collect_symbols`](SymbolResolver::collect_symbols) for each
///    input object file.
/// 3. Optionally call [`scan_archive`](SymbolResolver::scan_archive) for
///    each static library.
/// 4. Call [`resolve`](SymbolResolver::resolve) to perform binding/visibility
///    resolution.
/// 5. Call [`check_undefined`](SymbolResolver::check_undefined) to verify
///    all referenced symbols are defined.
/// 6. Call [`define_linker_symbols`](SymbolResolver::define_linker_symbols)
///    to inject linker-generated symbols.
/// 7. Call [`build_symbol_table`](SymbolResolver::build_symbol_table) to
///    produce the final ELF symbol table.
pub struct SymbolResolver {
    /// All input symbols grouped by name. Only global and weak symbols
    /// participate in cross-file resolution.
    global_symbols: FxHashMap<String, Vec<InputSymbol>>,

    /// Local symbols from all input objects. These are kept separately
    /// because they never participate in cross-file resolution.
    local_symbols: Vec<InputSymbol>,

    /// Final resolution results, keyed by symbol name.
    resolved: FxHashMap<String, ResolvedSymbol>,

    /// Names of symbols that are referenced (undefined) but have no
    /// definition in any input object or archive member.
    undefined: FxHashSet<String>,

    /// Collected error messages (e.g. multiple definition conflicts).
    errors: Vec<String>,
}

impl Default for SymbolResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolResolver {
    // -- Construction ------------------------------------------------------

    /// Create a new, empty symbol resolver.
    pub fn new() -> Self {
        SymbolResolver {
            global_symbols: FxHashMap::default(),
            local_symbols: Vec::new(),
            resolved: FxHashMap::default(),
            undefined: FxHashSet::default(),
            errors: Vec::new(),
        }
    }

    // -- Pass 1: Symbol Collection ----------------------------------------

    /// Collect symbols from a single input object file.
    ///
    /// Local symbols (`STB_LOCAL`) are stored separately and never
    /// participate in cross-file resolution. Global and weak symbols
    /// are grouped by name for the resolution pass. Undefined references
    /// (global symbols with `section_index == SHN_UNDEF`) are tracked
    /// so archive scanning can resolve them.
    pub fn collect_symbols(&mut self, object_id: u32, symbols: &[InputSymbol]) {
        for sym in symbols {
            // Skip empty-name symbols (e.g. the null symbol at index 0).
            if sym.name.is_empty() {
                continue;
            }

            // Clone and stamp with the object file ID.
            let mut input_sym = sym.clone();
            input_sym.object_file_id = object_id;

            if input_sym.is_local() {
                // Local symbols: store separately, no cross-file resolution.
                self.local_symbols.push(input_sym);
            } else {
                // Global or weak: group by name for resolution.
                self.global_symbols
                    .entry(input_sym.name.clone())
                    .or_default()
                    .push(input_sym);
            }
        }
    }

    // -- Pass 2: Symbol Resolution ----------------------------------------

    /// Resolve all collected global and weak symbols according to ELF
    /// binding precedence rules.
    ///
    /// For each unique symbol name:
    /// - Multiple strong (`STB_GLOBAL`) definitions → error.
    /// - Exactly one strong definition → it wins (overrides any weak).
    /// - Only weak definitions → any one is chosen (first encountered).
    /// - No definitions, only references → symbol is undefined.
    ///
    /// Visibility is merged by selecting the most restrictive value across
    /// all instances of the same symbol name.
    ///
    /// Returns `Ok(())` on success or `Err(errors)` if multiple-definition
    /// conflicts were detected.
    pub fn resolve(&mut self) -> Result<(), Vec<String>> {
        // Snapshot all symbol groups to avoid borrow conflicts when
        // calling mutable methods during resolution.
        let snapshot: Vec<(String, Vec<InputSymbol>)> = self
            .global_symbols
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (name, entries) in &snapshot {
            // Partition into definitions and references.
            let mut strong_defs: Vec<&InputSymbol> = Vec::new();
            let mut weak_defs: Vec<&InputSymbol> = Vec::new();
            let mut has_reference = false;
            let mut merged_vis: u8 = STV_DEFAULT;

            for sym in entries {
                // Merge visibility from ALL instances (defs and refs).
                merged_vis = merge_visibility(merged_vis, sym.visibility);

                if sym.is_defined() {
                    if sym.is_global() {
                        strong_defs.push(sym);
                    } else if sym.is_weak() {
                        weak_defs.push(sym);
                    }
                } else {
                    has_reference = true;
                }
            }

            // Apply binding precedence rules.
            if strong_defs.len() > 1 {
                // ERROR: multiple strong definitions.
                let locations: Vec<String> = strong_defs
                    .iter()
                    .map(|s| format!("object {}", s.object_file_id))
                    .collect();
                self.errors.push(format!(
                    "multiple definition of '{}' (defined in: {})",
                    name,
                    locations.join(", ")
                ));
                // Still pick the first strong definition so resolution can
                // continue and report as many errors as possible.
                let winner = strong_defs[0];
                self.insert_resolved(name, winner, merged_vis, true);
            } else if strong_defs.len() == 1 {
                // Exactly one strong definition — it wins over any weak.
                let winner = strong_defs[0];
                self.insert_resolved(name, winner, merged_vis, true);
            } else if !weak_defs.is_empty() {
                // Only weak definitions — pick the first one.
                let winner = weak_defs[0];
                self.insert_resolved(name, winner, merged_vis, true);
            } else if has_reference {
                // No definitions at all — symbol is undefined.
                let is_weak_ref = entries.iter().all(|s| s.is_weak());
                self.undefined.insert(name.clone());
                // Always add to `resolved` so that the relocation resolver
                // can look up the symbol (and then check for PLT entries
                // for dynamically-linked executables).  Without this entry,
                // PLT-based relocations against libc functions would fail
                // with "undefined reference" even when `allow_undefined` is
                // set and the symbol has a PLT stub.
                let ref_sym = entries.first().unwrap();
                self.resolved.insert(
                    name.clone(),
                    ResolvedSymbol {
                        name: name.clone(),
                        final_address: 0,
                        size: 0,
                        binding: if is_weak_ref {
                            STB_WEAK
                        } else {
                            ref_sym.binding
                        },
                        sym_type: ref_sym.sym_type,
                        visibility: merged_vis,
                        section_name: String::new(),
                        is_defined: false,
                        from_object: ref_sym.object_file_id,
                        export_dynamic: false,
                    },
                );
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    /// Internal helper to insert a resolved symbol from a winning definition.
    fn insert_resolved(
        &mut self,
        name: &str,
        winner: &InputSymbol,
        merged_vis: u8,
        is_defined: bool,
    ) {
        // Remove from undefined set if it was previously tracked there.
        self.undefined.remove(name);

        // Determine whether this symbol should be exported to .dynsym.
        // Symbols with default or protected visibility and global binding
        // are candidates for dynamic export.
        let export_dynamic =
            (merged_vis == STV_DEFAULT || merged_vis == STV_PROTECTED) && !winner.is_weak();

        self.resolved.insert(
            name.to_string(),
            ResolvedSymbol {
                name: name.to_string(),
                final_address: winner.value,
                size: winner.size,
                binding: winner.binding,
                sym_type: winner.sym_type,
                visibility: merged_vis,
                section_name: format!("section_{}", winner.section_index),
                is_defined,
                from_object: winner.object_file_id,
                export_dynamic,
            },
        );
    }

    // -- Undefined Symbol Checking ----------------------------------------

    /// Check for unresolved symbol references.
    ///
    /// When `allow_undefined` is `false` (static linking), any undefined
    /// non-weak symbol produces an error. When `true` (shared library
    /// linking with `-shared`), undefined symbols are permitted because
    /// they will be resolved at runtime by the dynamic linker.
    ///
    /// Weak undefined references are never errors — they resolve to
    /// address 0 (NULL) at runtime.
    pub fn check_undefined(&self, allow_undefined: bool) -> Result<(), Vec<String>> {
        if allow_undefined {
            return Ok(());
        }

        let mut errors = Vec::new();

        for name in self.undefined.iter() {
            // Check if this undefined symbol is weak (not an error).
            let is_weak_undef = self
                .global_symbols
                .get(name)
                .map(|entries| entries.iter().all(|s| s.is_weak()))
                .unwrap_or(false);

            if !is_weak_undef {
                // Check if it was resolved by archive scanning or linker symbols.
                if !self.resolved.contains_key(name) {
                    errors.push(format!("undefined reference to '{}'", name));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // -- Archive Scanning -------------------------------------------------

    /// Scan archive members and include those that define currently-undefined
    /// symbols.
    ///
    /// Implements the standard iterative archive scanning algorithm:
    /// 1. Compute the set of currently-undefined symbols from collected globals.
    /// 2. For each undefined symbol, check if any archive member defines it.
    /// 3. If yes, include that member (collect all its symbols).
    /// 4. Repeat until no more members are pulled in (fixpoint reached).
    ///
    /// Returns the indices of all archive members that were included.
    pub fn scan_archive(&mut self, archive_members: &[ArchiveMember]) -> Vec<u32> {
        let mut included: FxHashSet<u32> = FxHashSet::default();
        let mut changed = true;

        while changed {
            changed = false;

            // Compute currently-undefined symbols: names in global_symbols
            // that have references but no definitions yet.
            let currently_undefined = self.compute_undefined_names();

            for member in archive_members {
                // Skip already-included members.
                if included.contains(&member.index) {
                    continue;
                }

                // Check if this member defines any currently-undefined symbol.
                let defines_needed = member.symbols.iter().any(|sym| {
                    sym.is_defined() && !sym.is_local() && currently_undefined.contains(&sym.name)
                });

                if defines_needed {
                    // Include this archive member.
                    included.insert(member.index);
                    changed = true;

                    // Collect all symbols from this member into global_symbols.
                    self.collect_symbols(member.index, &member.symbols);
                }
            }
        }

        included.iter().copied().collect()
    }

    /// Compute the set of symbol names that are currently undefined:
    /// they appear in `global_symbols` but have no definition (all entries
    /// have `section_index == SHN_UNDEF`).
    fn compute_undefined_names(&self) -> FxHashSet<String> {
        let mut undef = FxHashSet::default();

        for (name, entries) in self.global_symbols.iter() {
            let has_definition = entries.iter().any(|s| s.is_defined());
            let has_reference = entries.iter().any(|s| !s.is_defined());

            if has_reference && !has_definition {
                undef.insert(name.clone());
            }
        }

        undef
    }

    // -- Linker-Defined Symbols -------------------------------------------

    /// Inject linker-generated symbols based on output section addresses.
    ///
    /// Standard linker symbols created if not already defined by input:
    /// - `__bss_start` — start address of `.bss`
    /// - `_edata` — end address of `.data`
    /// - `_end` / `__end` — end of all loadable sections
    /// - `__executable_start` — base load address of the executable
    ///
    /// Relocate resolved symbol values from section-relative offsets to
    /// absolute virtual addresses after section merging.
    ///
    /// For each resolved symbol, looks up its input section via
    /// `(object_file_id, section_index)` → section_name, then adds the
    /// section's virtual address to the symbol's `final_address`.
    ///
    /// When multiple object files contribute sections with the same name
    /// (e.g. `.text`), the `fragment_offset_map` provides the byte offset
    /// of each input section fragment within the merged output section.
    /// The final address is: `section_va + fragment_offset + symbol_value`.
    pub fn relocate_symbol_addresses(
        &mut self,
        address_map: &crate::backend::linker_common::section_merger::AddressMap,
        section_index_to_name: &FxHashMap<(u32, u16), String>,
        fragment_offset_map: &FxHashMap<(u32, u16), u64>,
    ) {
        for resolved in self.resolved.values_mut() {
            if !resolved.is_defined {
                continue;
            }
            // Try to find the section this symbol belongs to.
            // The section_name is "section_N" where N is the section index.
            if let Some(idx_str) = resolved.section_name.strip_prefix("section_") {
                if let Ok(idx) = idx_str.parse::<u16>() {
                    let lookup_key = (resolved.from_object, idx);
                    if let Some(section_name) = section_index_to_name.get(&lookup_key) {
                        if let Some(addr_info) = address_map.section_addresses.get(section_name) {
                            // Add output section VA + fragment offset within
                            // the merged output section.
                            let frag_off =
                                fragment_offset_map.get(&lookup_key).copied().unwrap_or(0);
                            resolved.final_address += addr_info.virtual_address + frag_off;
                            // Update section_name to the real name.
                            resolved.section_name = section_name.clone();
                        }
                    }
                }
            }
        }
    }

    /// Relocate local symbol values from section-relative offsets to
    /// absolute virtual addresses after section merging.
    ///
    /// Like [`relocate_symbol_addresses`], this accounts for fragment
    /// offsets within merged output sections when multiple objects
    /// contribute the same section.
    pub fn relocate_local_symbol_addresses(
        &mut self,
        address_map: &crate::backend::linker_common::section_merger::AddressMap,
        section_index_to_name: &FxHashMap<(u32, u16), String>,
        fragment_offset_map: &FxHashMap<(u32, u16), u64>,
    ) {
        for sym in &mut self.local_symbols {
            let lookup_key = (sym.object_file_id, sym.section_index);
            if let Some(section_name) = section_index_to_name.get(&lookup_key) {
                if let Some(addr_info) = address_map.section_addresses.get(section_name) {
                    let frag_off = fragment_offset_map.get(&lookup_key).copied().unwrap_or(0);
                    sym.value += addr_info.virtual_address + frag_off;
                }
            }
        }
    }

    ///
    /// The `section_addresses` map provides `(start_address, end_address)`
    /// for each output section name (e.g. `".text"`, `".data"`, `".bss"`).
    pub fn define_linker_symbols(&mut self, section_addresses: &FxHashMap<String, (u64, u64)>) {
        // Helper: define a symbol if it doesn't already have a strong definition.
        let mut define = |name: &str, value: u64, section: &str| {
            if !self.resolved.contains_key(name) {
                self.resolved.insert(
                    name.to_string(),
                    ResolvedSymbol {
                        name: name.to_string(),
                        final_address: value,
                        size: 0,
                        binding: STB_GLOBAL,
                        sym_type: STT_NOTYPE,
                        visibility: STV_HIDDEN,
                        section_name: section.to_string(),
                        is_defined: true,
                        from_object: 0, // linker-generated
                        export_dynamic: false,
                    },
                );
                // Remove from undefined set if it was there.
                self.undefined.remove(name);
            }
        };

        // __bss_start — start of .bss section
        if let Some(&(bss_start, _)) = section_addresses.get(".bss") {
            define("__bss_start", bss_start, ".bss");
        }

        // _edata — end of .data section
        if let Some(&(_, data_end)) = section_addresses.get(".data") {
            define("_edata", data_end, ".data");
        }

        // _end / __end — end of all loadable sections
        // Find the maximum end address across all sections.
        let max_end = section_addresses
            .values()
            .map(|&(_, end)| end)
            .max()
            .unwrap_or(0);
        define("_end", max_end, "");
        define("__end", max_end, "");

        // __executable_start — base load address
        // Use the minimum start address across all sections.
        let min_start = section_addresses
            .values()
            .map(|&(start, _)| start)
            .min()
            .unwrap_or(0);
        define("__executable_start", min_start, "");
    }

    // -- Constructor / Destructor Collection -------------------------------

    /// Collect constructor and destructor function addresses for
    /// `.init_array` and `.fini_array` sections.
    ///
    /// Returns `(constructors, destructors)` where each vector contains
    /// absolute addresses of the corresponding functions, sorted for
    /// deterministic initialization/finalization order.
    ///
    /// Constructor functions are identified by being in sections named
    /// `.init_array` or `.ctors`. Destructor functions are in `.fini_array`
    /// or `.dtors`.
    pub fn collect_init_fini_symbols(&self) -> (Vec<u64>, Vec<u64>) {
        let mut constructors: Vec<u64> = Vec::new();
        let mut destructors: Vec<u64> = Vec::new();

        for resolved in self.resolved.values() {
            if !resolved.is_defined {
                continue;
            }

            // Check section name for init/fini arrays.
            let section = resolved.section_name.as_str();
            if section.starts_with(".init_array") || section.starts_with(".ctors") {
                constructors.push(resolved.final_address);
            } else if section.starts_with(".fini_array") || section.starts_with(".dtors") {
                destructors.push(resolved.final_address);
            }
        }

        // Sort for deterministic ordering.
        constructors.sort_unstable();
        destructors.sort_unstable();

        (constructors, destructors)
    }

    // -- Symbol Table Construction ----------------------------------------

    /// Build the final symbol table for ELF `.symtab` emission.
    ///
    /// The output symbol table follows ELF conventions:
    /// - Index 0 is the null symbol (all fields zero).
    /// - Local symbols appear first (indices 1..first_global_index).
    /// - Global and weak symbols follow.
    /// - `first_global_index` becomes the `sh_info` value of the
    ///   `.symtab` section header.
    pub fn build_symbol_table(&self) -> SymbolTable {
        let mut locals: Vec<OutputSymbol> = Vec::new();
        let mut globals: Vec<OutputSymbol> = Vec::new();

        // Null symbol at index 0 (ELF requirement).
        locals.push(OutputSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_NOTYPE,
            visibility: STV_DEFAULT,
            section_index: SHN_UNDEF,
        });

        // Add local symbols.
        for sym in &self.local_symbols {
            locals.push(OutputSymbol {
                name: sym.name.clone(),
                value: sym.value,
                size: sym.size,
                binding: sym.binding,
                sym_type: sym.sym_type,
                visibility: sym.visibility,
                section_index: sym.section_index,
            });
        }

        // Add resolved global and weak symbols.
        // Sort by name for deterministic output.
        let mut resolved_names: Vec<&String> = self.resolved.keys().collect();
        resolved_names.sort();

        for name in resolved_names {
            if let Some(resolved) = self.resolved.get(name) {
                globals.push(OutputSymbol {
                    name: resolved.name.clone(),
                    value: resolved.final_address,
                    size: resolved.size,
                    binding: resolved.binding,
                    sym_type: resolved.sym_type,
                    visibility: resolved.visibility,
                    section_index: if resolved.is_defined {
                        // Use a placeholder section index; the actual index
                        // is assigned by the ELF writer during section layout.
                        1
                    } else {
                        SHN_UNDEF
                    },
                });
            }
        }

        let first_global_index = locals.len() as u32;

        // Combine: locals first, then globals.
        let mut symbols = locals;
        symbols.extend(globals);

        SymbolTable {
            symbols,
            first_global_index,
        }
    }

    // -- Diagnostic Emission ----------------------------------------------

    /// Emit all collected errors to a [`DiagnosticEngine`] as structured
    /// diagnostics.
    ///
    /// Each error string is wrapped in a [`Diagnostic::error`] with a
    /// [`Span::dummy`] since linker errors don't correspond to specific
    /// source locations.
    pub fn emit_diagnostics(&self, engine: &mut DiagnosticEngine) {
        for err in &self.errors {
            engine.emit(Diagnostic::error(Span::dummy(), err.clone()));
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level convenience function
// ---------------------------------------------------------------------------

/// Resolve all symbols from input objects and archives in a single call.
///
/// This is the primary public API for the symbol resolver. It orchestrates
/// the complete two-pass resolution process:
///
/// 1. Collect symbols from all input objects.
/// 2. Scan archives iteratively to resolve undefined references.
/// 3. Resolve symbol bindings and merge visibility.
/// 4. Check for unresolved references (unless `is_shared` is `true`).
/// 5. Build the final symbol table.
///
/// # Parameters
///
/// - `input_objects` — symbols from each input relocatable object file.
/// - `archives` — static library archives to scan for definitions.
/// - `is_shared` — if `true`, undefined symbols are permitted (shared library).
///
/// # Returns
///
/// `Ok(SymbolResolutionResult)` on success, or `Err(errors)` if resolution
/// fails (multiple definitions or unresolved references).
pub fn resolve_all_symbols(
    input_objects: &[ObjectSymbols],
    archives: &[Archive],
    is_shared: bool,
) -> Result<SymbolResolutionResult, Vec<String>> {
    let mut resolver = SymbolResolver::new();

    // Pass 1: Collect symbols from all input objects.
    for obj in input_objects {
        resolver.collect_symbols(obj.object_id, &obj.symbols);
    }

    // Archive scanning: iteratively include archive members that define
    // currently-undefined symbols.
    let mut all_included_members: Vec<(usize, Vec<u32>)> = Vec::new();
    for (archive_idx, archive) in archives.iter().enumerate() {
        let included = resolver.scan_archive(&archive.members);
        if !included.is_empty() {
            all_included_members.push((archive_idx, included));
        }
    }

    // Pass 2: Resolve symbol bindings and visibility.
    if let Err(errors) = resolver.resolve() {
        // Use DiagnosticEngine to report multiple-definition errors as
        // structured diagnostics before returning the error list.
        let mut diag_engine = DiagnosticEngine::new();
        for err in &errors {
            diag_engine.emit_error(Span::dummy(), err.as_str());
        }
        // The engine has errors — this confirms the failure.
        if diag_engine.has_errors() {
            return Err(errors);
        }
    }

    // Check for unresolved references.
    if let Err(undef_errors) = resolver.check_undefined(is_shared) {
        // Emit undefined reference errors via diagnostic engine.
        let mut diag_engine = DiagnosticEngine::new();
        resolver.emit_diagnostics(&mut diag_engine);
        for err in &undef_errors {
            diag_engine.emit(Diagnostic::error(Span::dummy(), err.clone()));
        }
        return Err(undef_errors);
    }

    // Build the final symbol table.
    let symbol_table = resolver.build_symbol_table();

    // Collect undefined weak symbols.
    let undefined_weak: FxHashSet<String> = resolver
        .undefined
        .iter()
        .filter(|name| {
            resolver
                .global_symbols
                .get(*name)
                .map(|entries| entries.iter().all(|s| s.is_weak()))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    // Collect init/fini symbols (not directly included in the result
    // struct but exercised for validation).
    let (_constructors, _destructors) = resolver.collect_init_fini_symbols();

    Ok(SymbolResolutionResult {
        symbol_table,
        resolved: resolver.resolved,
        undefined_weak,
        included_archive_members: all_included_members,
    })
}
