//! String interning module — zero-cost identifier comparison through compact Symbol handles.
//!
//! This module provides the [`Interner`] struct backed by [`FxHashMap`] for O(1)
//! deduplication of identifiers, keywords, and string literals. Each unique string
//! is stored exactly once and represented by a lightweight 4-byte [`Symbol`] handle
//! that supports `Copy`, `Eq`, `Hash`, and `Ord` for efficient use as keys in
//! symbol tables, scope maps, and AST nodes.
//!
//! # Architecture
//!
//! The `Interner` maintains two data structures in tandem:
//!
//! - **`strings: Vec<String>`** — an arena of all interned strings, indexed by
//!   the `Symbol`'s `u32` value. This provides O(1) resolution from `Symbol` back
//!   to `&str`.
//!
//! - **`map: FxHashMap<String, Symbol>`** — a reverse-lookup map from string
//!   content to `Symbol`. This provides O(1) amortised deduplication during
//!   interning via FxHash's fast Fibonacci hashing.
//!
//! # Pre-interned Keywords
//!
//! On construction, the `Interner` pre-interns the empty string (at index 0,
//! accessible via [`Symbol::EMPTY`]), all C11 keywords, C11 special keywords
//! (`_Alignas`, `_Atomic`, etc.), and common GCC extension keywords. This ensures
//! that frequently-used compiler strings have stable, low-valued `Symbol` indices
//! and avoids redundant hash lookups during tokenization.
//!
//! # Usage
//!
//! ```rust
//! use bcc::common::string_interner::{Interner, Symbol};
//!
//! let mut interner = Interner::new();
//!
//! let sym_main = interner.intern("main");
//! let sym_main2 = interner.intern("main");
//! assert_eq!(sym_main, sym_main2); // same Symbol for same string
//!
//! assert_eq!(interner.resolve(sym_main), "main");
//! assert!(interner.contains("main"));
//! ```
//!
//! # Thread Safety
//!
//! The `Interner` is **not** thread-safe by design. BCC uses a single compilation
//! thread with a 64 MiB stack, so no synchronisation is needed. If thread safety
//! is required in the future, the `Interner` can be wrapped in a `Mutex`.
//!
//! # Zero-Dependency Implementation
//!
//! This module replaces the external `string-interner` or `lasso` crates,
//! adhering to the project's zero-dependency mandate. Only the internal
//! [`FxHashMap`](crate::common::fx_hash::FxHashMap) and the Rust standard
//! library are used.

use crate::common::fx_hash::FxHashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Symbol — compact 4-byte handle to an interned string
// ---------------------------------------------------------------------------

/// A compact, 4-byte handle representing an interned string.
///
/// `Symbol` is a newtype wrapper around `u32` that serves as an index into the
/// [`Interner`]'s string arena. Because it is `Copy`, `Eq`, `Hash`, and `Ord`,
/// it can be used as an efficient key in hash maps, B-trees, and as a field in
/// AST nodes without incurring any allocation or string-comparison overhead.
///
/// # Comparison Semantics
///
/// Two `Symbol` values are equal if and only if they were produced by interning
/// the same string content through the same `Interner` instance. Comparing
/// `Symbol` values from different `Interner` instances is **undefined** and will
/// produce incorrect results.
///
/// # Memory Layout
///
/// `Symbol` is exactly 4 bytes (`u32`), making it suitable for dense storage
/// in arrays, vectors, and struct fields throughout the compilation pipeline.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Symbol(u32);

impl Symbol {
    /// Sentinel symbol representing the empty string `""`.
    ///
    /// This is always the first entry (index 0) in every `Interner` instance
    /// and can be used as a default/missing identifier placeholder in AST nodes
    /// and symbol table entries where no name is present.
    pub const EMPTY: Symbol = Symbol(0);

    /// Creates a new `Symbol` from a raw `u32` index.
    ///
    /// This is the primary constructor for `Symbol` values created outside the
    /// `Interner`. It is typically used when reconstructing symbols from
    /// serialised data, test fixtures, or when a raw index is known.
    ///
    /// # Safety (Logical)
    ///
    /// The caller is responsible for ensuring the index is valid within the
    /// relevant `Interner`. Using an out-of-range index will cause a panic
    /// when resolved via [`Interner::resolve`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::Symbol;
    /// let sym = Symbol::new(42);
    /// assert_eq!(sym.as_u32(), 42);
    /// ```
    #[inline]
    pub fn new(index: u32) -> Self {
        Symbol(index)
    }

    /// Returns the raw `u32` index of this symbol within the interner's arena.
    ///
    /// This is primarily useful for serialisation, compact encoding in bitfields,
    /// or for interfacing with external data structures that require integer keys.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::Symbol;
    /// assert_eq!(Symbol::EMPTY.as_u32(), 0);
    /// ```
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Debug for Symbol {
    /// Formats the symbol as `Symbol(<index>)` for debug output.
    ///
    /// To see the actual string content, use [`Interner::resolve`] instead.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Pre-interned keyword tables
// ---------------------------------------------------------------------------

/// C11 standard keywords (ISO/IEC 9899:2011 §6.4.1).
///
/// These are pre-interned at `Interner` construction so that every keyword
/// token produced by the lexer maps to a stable, pre-existing `Symbol` without
/// requiring a fresh hash-map lookup on first encounter.
const C11_KEYWORDS: &[&str] = &[
    // Storage class specifiers
    "auto", "extern", "register", "static", "typedef", // Type specifiers
    "char", "double", "enum", "float", "int", "long", "short", "signed", "struct", "union",
    "unsigned", "void", // Type qualifiers
    "const", "restrict", "volatile", // Function specifiers
    "inline",   // Control flow
    "break", "case", "continue", "default", "do", "else", "for", "goto", "if", "return", "switch",
    "while", // Operators and misc
    "sizeof",
];

/// C11 special keywords introduced by the standard with underscore-uppercase
/// naming convention (§6.4.1, §6.7.2.4, §6.7.5, §6.10.1).
const C11_SPECIAL_KEYWORDS: &[&str] = &[
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
];

/// GCC extension keywords required for Linux kernel compilation.
///
/// These cover the `__attribute__`, `__typeof__`, inline assembly, and builtin
/// families that the Linux kernel source extensively relies upon.
const GCC_EXTENSION_KEYWORDS: &[&str] = &[
    // Attribute and extension markers
    "__attribute__",
    "__attribute",
    "__extension__",
    // Alternate type specifiers
    "__typeof__",
    "__typeof",
    "typeof",
    // Inline assembly
    "asm",
    "__asm__",
    "__asm",
    // Qualifier alternates
    "__volatile__",
    "__volatile",
    "__const__",
    "__const",
    "__inline__",
    "__inline",
    "__restrict__",
    "__restrict",
    "__signed__",
    "__signed",
    "__unsigned",
    // Labels
    "__label__",
    // Alignment
    "__alignof__",
    "__alignof",
    // Builtin prefix forms (the most common builtins used in kernel headers)
    "__builtin_va_list",
    "__builtin_va_start",
    "__builtin_va_end",
    "__builtin_va_arg",
    "__builtin_va_copy",
    "__builtin_offsetof",
    "__builtin_constant_p",
    "__builtin_expect",
    "__builtin_unreachable",
    "__builtin_types_compatible_p",
    "__builtin_choose_expr",
    "__builtin_clz",
    "__builtin_ctz",
    "__builtin_popcount",
    "__builtin_clzl",
    "__builtin_ctzl",
    "__builtin_popcountl",
    "__builtin_clzll",
    "__builtin_ctzll",
    "__builtin_popcountll",
    "__builtin_bswap16",
    "__builtin_bswap32",
    "__builtin_bswap64",
    "__builtin_ffs",
    "__builtin_ffsl",
    "__builtin_ffsll",
    "__builtin_trap",
    "__builtin_assume_aligned",
    "__builtin_frame_address",
    "__builtin_return_address",
    "__builtin_add_overflow",
    "__builtin_sub_overflow",
    "__builtin_mul_overflow",
    "__builtin_object_size",
    "__builtin_prefetch",
    "__builtin_huge_val",
    "__builtin_huge_valf",
    "__builtin_inf",
    "__builtin_inff",
    "__builtin_nan",
    "__builtin_nanf",
    "__builtin_alloca",
    "__builtin_memcpy",
    "__builtin_memset",
    "__builtin_memmove",
    "__builtin_memcmp",
    "__builtin_strcmp",
    "__builtin_strlen",
    "__builtin_abs",
    "__builtin_labs",
    "__builtin_llabs",
];

/// Common identifiers and type aliases frequently encountered in C code,
/// especially in the Linux kernel. Pre-interning these avoids repeated
/// hash-map probes during parsing and semantic analysis.
const COMMON_IDENTIFIERS: &[&str] = &[
    // Standard library type aliases
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "intptr_t",
    "uintptr_t",
    "wchar_t",
    // Fixed-width integer types
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    // Boolean
    "bool",
    "true",
    "false",
    // Common kernel identifiers
    "NULL",
    "main",
    // Visibility attributes
    "aligned",
    "packed",
    "section",
    "used",
    "unused",
    "weak",
    "constructor",
    "destructor",
    "visibility",
    "deprecated",
    "noreturn",
    "noinline",
    "always_inline",
    "cold",
    "hot",
    "format",
    "format_arg",
    "malloc",
    "pure",
    "warn_unused_result",
    "fallthrough",
    // Inline asm clobbers and constraints commonly used in kernel
    "memory",
    "cc",
];

// ---------------------------------------------------------------------------
// Interner — FxHashMap-backed string deduplication engine
// ---------------------------------------------------------------------------

/// A string interner that deduplicates strings and assigns each unique string
/// a compact [`Symbol`] handle for O(1) comparison.
///
/// # Invariants
///
/// - `strings.len() == map.len()` — every string in the arena has a
///   corresponding map entry, and vice versa.
/// - `strings[0] == ""` — the empty string is always at index 0
///   (corresponding to [`Symbol::EMPTY`]).
/// - `strings.len() <= u32::MAX as usize` — the arena cannot exceed the
///   `u32` index space. In practice, no C compilation unit approaches this
///   limit (~4 billion unique strings).
pub struct Interner {
    /// Arena of all interned strings. The index of each string corresponds to
    /// its `Symbol` value (i.e., `strings[sym.as_u32() as usize]` is the
    /// string that `sym` represents).
    strings: Vec<String>,

    /// Reverse-lookup map from string content to `Symbol` for O(1) amortised
    /// deduplication during interning. Backed by [`FxHashMap`] for fast
    /// Fibonacci hashing of the small-string keys typical in compilers.
    map: FxHashMap<String, Symbol>,
}

impl Interner {
    /// Creates a new `Interner` pre-populated with the empty string sentinel
    /// and all C11 keywords, C11 special keywords, GCC extension keywords,
    /// and common identifiers.
    ///
    /// After construction, [`Symbol::EMPTY`] is guaranteed to resolve to `""`,
    /// and all pre-interned keywords have stable `Symbol` values that remain
    /// consistent across compilation runs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::{Interner, Symbol};
    ///
    /// let interner = Interner::new();
    /// assert_eq!(interner.resolve(Symbol::EMPTY), "");
    /// assert!(interner.contains("int"));
    /// assert!(interner.contains("__attribute__"));
    /// ```
    pub fn new() -> Self {
        // Calculate total pre-interned count for capacity pre-allocation.
        // +1 for the empty string sentinel at index 0.
        let preinterned_count = 1
            + C11_KEYWORDS.len()
            + C11_SPECIAL_KEYWORDS.len()
            + GCC_EXTENSION_KEYWORDS.len()
            + COMMON_IDENTIFIERS.len();

        let mut interner = Interner {
            strings: Vec::with_capacity(preinterned_count),
            map: FxHashMap::default(),
        };

        // Index 0: empty string sentinel (Symbol::EMPTY)
        interner.intern_raw("");

        // Pre-intern all keyword and identifier tables. The `intern_raw`
        // helper handles deduplication if any string appears in multiple
        // tables (which it shouldn't, but safety first).
        for &kw in C11_KEYWORDS {
            interner.intern_raw(kw);
        }
        for &kw in C11_SPECIAL_KEYWORDS {
            interner.intern_raw(kw);
        }
        for &kw in GCC_EXTENSION_KEYWORDS {
            interner.intern_raw(kw);
        }
        for &id in COMMON_IDENTIFIERS {
            interner.intern_raw(id);
        }

        interner
    }

    /// Interns a string, returning a [`Symbol`] handle that uniquely identifies
    /// the string content within this interner.
    ///
    /// If the string has already been interned, the existing `Symbol` is returned
    /// in O(1) time via the `FxHashMap` lookup. Otherwise, the string is allocated
    /// in the arena and a new `Symbol` is assigned.
    ///
    /// # Panics
    ///
    /// Panics if the number of unique interned strings exceeds `u32::MAX`
    /// (approximately 4.29 billion). In practice, this limit is unreachable
    /// for any C compilation unit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::Interner;
    ///
    /// let mut interner = Interner::new();
    /// let s1 = interner.intern("my_variable");
    /// let s2 = interner.intern("my_variable");
    /// assert_eq!(s1, s2);
    /// ```
    #[inline]
    pub fn intern(&mut self, s: &str) -> Symbol {
        // Fast path: check if already interned via FxHashMap lookup.
        // FxHashMap::get uses Borrow<str> on String keys, so &str works
        // directly without allocation.
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }

        // Slow path: allocate the string in the arena and record the mapping.
        self.intern_fresh(s)
    }

    /// Resolves a [`Symbol`] back to its string content.
    ///
    /// This is an O(1) operation — a direct index into the string arena.
    ///
    /// # Panics
    ///
    /// Panics if `sym` was not produced by this `Interner` (i.e., the index
    /// is out of bounds). In a correctly functioning compiler pipeline, this
    /// should never occur.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::Interner;
    ///
    /// let mut interner = Interner::new();
    /// let sym = interner.intern("printf");
    /// assert_eq!(interner.resolve(sym), "printf");
    /// ```
    #[inline]
    pub fn resolve(&self, sym: Symbol) -> &str {
        let idx = sym.0 as usize;
        debug_assert!(
            idx < self.strings.len(),
            "Symbol({}) is out of bounds (interner has {} entries)",
            idx,
            self.strings.len()
        );
        // Safety: the debug_assert above catches out-of-bounds in debug builds.
        // In release builds, the index operation will panic on out-of-bounds,
        // which is the correct behaviour for a corrupted Symbol.
        &self.strings[idx]
    }

    /// Returns `true` if the given string has already been interned.
    ///
    /// This does not allocate or modify the interner — it is a pure lookup
    /// in the `FxHashMap`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use bcc::common::string_interner::Interner;
    ///
    /// let mut interner = Interner::new();
    /// assert!(!interner.contains("my_func"));
    /// interner.intern("my_func");
    /// assert!(interner.contains("my_func"));
    /// ```
    #[inline]
    pub fn contains(&self, s: &str) -> bool {
        self.map.contains_key(s)
    }

    /// Returns the number of unique strings currently interned.
    ///
    /// This includes all pre-interned keywords and any strings added via
    /// [`intern`](Interner::intern).
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if the interner contains no strings.
    ///
    /// Note: a freshly constructed `Interner` via [`new`](Interner::new) is
    /// never empty because it pre-interns the empty string and all keywords.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Looks up the `Symbol` for a previously interned string, returning `None`
    /// if the string has not been interned.
    ///
    /// Unlike [`intern`](Interner::intern), this method does not allocate or
    /// modify the interner.
    #[inline]
    pub fn lookup(&self, s: &str) -> Option<Symbol> {
        self.map.get(s).copied()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Interns a string without checking if it already exists.
    ///
    /// Used during pre-interning where we know the keyword tables contain
    /// no duplicates (but we handle duplicates gracefully via the `intern_raw`
    /// wrapper).
    fn intern_fresh(&mut self, s: &str) -> Symbol {
        let idx = self.strings.len();
        assert!(
            idx <= u32::MAX as usize,
            "String interner overflow: cannot intern more than {} unique strings",
            u32::MAX
        );
        let sym = Symbol(idx as u32);
        let owned = s.to_owned();
        self.strings.push(owned.clone());
        self.map.insert(owned, sym);
        sym
    }

    /// Interns a string if not already present; used during construction to
    /// build the pre-interned keyword set. Handles potential duplicates across
    /// keyword tables gracefully.
    fn intern_raw(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        self.intern_fresh(s)
    }
}

impl Default for Interner {
    /// Creates a new `Interner` with all pre-interned keywords.
    ///
    /// This is equivalent to [`Interner::new()`].
    #[inline]
    fn default() -> Self {
        Interner::new()
    }
}

impl fmt::Debug for Interner {
    /// Formats the interner showing the number of interned strings and a
    /// preview of the first few entries for diagnostic purposes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show a concise summary rather than dumping every string.
        let preview_count = 8.min(self.strings.len());
        let preview: Vec<&str> = self.strings[..preview_count]
            .iter()
            .map(|s| s.as_str())
            .collect();
        f.debug_struct("Interner")
            .field("count", &self.strings.len())
            .field("preview", &preview)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Symbol tests -------------------------------------------------------

    #[test]
    fn test_symbol_empty_is_zero() {
        assert_eq!(Symbol::EMPTY.as_u32(), 0);
    }

    #[test]
    fn test_symbol_copy() {
        let s = Symbol(42);
        let s2 = s; // Copy
        assert_eq!(s, s2);
        assert_eq!(s.as_u32(), 42);
    }

    #[test]
    fn test_symbol_eq_and_ne() {
        let a = Symbol(1);
        let b = Symbol(1);
        let c = Symbol(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_symbol_ord() {
        let a = Symbol(1);
        let b = Symbol(2);
        let c = Symbol(3);
        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn test_symbol_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let s1 = Symbol(100);
        let s2 = Symbol(100);
        let s3 = Symbol(200);

        let hash_of = |sym: Symbol| -> u64 {
            let mut h = DefaultHasher::new();
            sym.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash_of(s1), hash_of(s2));
        assert_ne!(hash_of(s1), hash_of(s3));
    }

    #[test]
    fn test_symbol_debug_format() {
        let s = Symbol(42);
        let dbg = format!("{:?}", s);
        assert_eq!(dbg, "Symbol(42)");
    }

    // -- Interner construction tests ----------------------------------------

    #[test]
    fn test_interner_new_is_not_empty() {
        let interner = Interner::new();
        assert!(!interner.is_empty());
        // At minimum: empty string + C11 keywords + special + GCC + common
        assert!(interner.len() > 50);
    }

    #[test]
    fn test_interner_empty_string_at_index_zero() {
        let interner = Interner::new();
        assert_eq!(interner.resolve(Symbol::EMPTY), "");
    }

    #[test]
    fn test_interner_default_equals_new() {
        let a = Interner::new();
        let b = Interner::default();
        assert_eq!(a.len(), b.len());
    }

    // -- Interner pre-interned keywords tests --------------------------------

    #[test]
    fn test_c11_keywords_pre_interned() {
        let interner = Interner::new();
        for &kw in C11_KEYWORDS {
            assert!(
                interner.contains(kw),
                "C11 keyword '{}' should be pre-interned",
                kw
            );
        }
    }

    #[test]
    fn test_c11_special_keywords_pre_interned() {
        let interner = Interner::new();
        for &kw in C11_SPECIAL_KEYWORDS {
            assert!(
                interner.contains(kw),
                "C11 special keyword '{}' should be pre-interned",
                kw
            );
        }
    }

    #[test]
    fn test_gcc_extension_keywords_pre_interned() {
        let interner = Interner::new();
        for &kw in GCC_EXTENSION_KEYWORDS {
            assert!(
                interner.contains(kw),
                "GCC extension keyword '{}' should be pre-interned",
                kw
            );
        }
    }

    #[test]
    fn test_common_identifiers_pre_interned() {
        let interner = Interner::new();
        for &id in COMMON_IDENTIFIERS {
            assert!(
                interner.contains(id),
                "Common identifier '{}' should be pre-interned",
                id
            );
        }
    }

    // -- Interner::intern tests ---------------------------------------------

    #[test]
    fn test_intern_new_string() {
        let mut interner = Interner::new();
        let initial_len = interner.len();
        let sym = interner.intern("my_unique_variable_name_xyz");
        assert_eq!(interner.len(), initial_len + 1);
        assert_eq!(interner.resolve(sym), "my_unique_variable_name_xyz");
    }

    #[test]
    fn test_intern_deduplication() {
        let mut interner = Interner::new();
        let s1 = interner.intern("hello_world");
        let s2 = interner.intern("hello_world");
        assert_eq!(s1, s2);
        // Length should only increase by 1 for the first intern
        let len_after = interner.len();
        let s3 = interner.intern("hello_world");
        assert_eq!(s1, s3);
        assert_eq!(interner.len(), len_after);
    }

    #[test]
    fn test_intern_distinct_strings_produce_distinct_symbols() {
        let mut interner = Interner::new();
        let a = interner.intern("alpha");
        let b = interner.intern("beta");
        assert_ne!(a, b);
    }

    #[test]
    fn test_intern_empty_string_returns_empty_symbol() {
        let mut interner = Interner::new();
        let sym = interner.intern("");
        assert_eq!(sym, Symbol::EMPTY);
        assert_eq!(sym.as_u32(), 0);
    }

    #[test]
    fn test_intern_pre_interned_keyword_returns_same_symbol() {
        let mut interner = Interner::new();
        let sym_int1 = interner.intern("int");
        let sym_int2 = interner.intern("int");
        assert_eq!(sym_int1, sym_int2);
        assert_eq!(interner.resolve(sym_int1), "int");

        // Interning a pre-interned keyword should not increase length
        let len = interner.len();
        let _ = interner.intern("void");
        assert_eq!(interner.len(), len);
    }

    #[test]
    fn test_intern_unicode_identifier() {
        let mut interner = Interner::new();
        let sym = interner.intern("変数名");
        assert_eq!(interner.resolve(sym), "変数名");
        assert!(interner.contains("変数名"));
    }

    #[test]
    fn test_intern_long_string() {
        let mut interner = Interner::new();
        let long = "a".repeat(10_000);
        let sym = interner.intern(&long);
        assert_eq!(interner.resolve(sym), long.as_str());
    }

    // -- Interner::resolve tests --------------------------------------------

    #[test]
    fn test_resolve_sequential() {
        let mut interner = Interner::new();
        let syms: Vec<Symbol> = (0..100)
            .map(|i| interner.intern(&format!("var_{}", i)))
            .collect();

        for (i, &sym) in syms.iter().enumerate() {
            assert_eq!(interner.resolve(sym), format!("var_{}", i));
        }
    }

    #[test]
    #[should_panic]
    fn test_resolve_out_of_bounds_panics() {
        let interner = Interner::new();
        let bad_sym = Symbol(u32::MAX);
        let _ = interner.resolve(bad_sym);
    }

    // -- Interner::contains tests -------------------------------------------

    #[test]
    fn test_contains_returns_false_for_absent_string() {
        let interner = Interner::new();
        assert!(!interner.contains("this_string_does_not_exist_in_any_table"));
    }

    #[test]
    fn test_contains_returns_true_after_intern() {
        let mut interner = Interner::new();
        assert!(!interner.contains("newly_interned"));
        interner.intern("newly_interned");
        assert!(interner.contains("newly_interned"));
    }

    // -- Interner::len tests ------------------------------------------------

    #[test]
    fn test_len_increases_on_new_intern() {
        let mut interner = Interner::new();
        let len_before = interner.len();
        interner.intern("brand_new_string");
        assert_eq!(interner.len(), len_before + 1);
    }

    #[test]
    fn test_len_unchanged_on_duplicate_intern() {
        let mut interner = Interner::new();
        interner.intern("test_dup");
        let len = interner.len();
        interner.intern("test_dup");
        assert_eq!(interner.len(), len);
    }

    // -- Interner::lookup tests ---------------------------------------------

    #[test]
    fn test_lookup_existing() {
        let mut interner = Interner::new();
        let sym = interner.intern("lookup_test");
        assert_eq!(interner.lookup("lookup_test"), Some(sym));
    }

    #[test]
    fn test_lookup_missing() {
        let interner = Interner::new();
        assert_eq!(interner.lookup("nonexistent_string_abcdef"), None);
    }

    #[test]
    fn test_lookup_pre_interned() {
        let interner = Interner::new();
        let sym = interner.lookup("int");
        assert!(sym.is_some());
        assert_eq!(interner.resolve(sym.unwrap()), "int");
    }

    // -- Interner::is_empty tests -------------------------------------------

    #[test]
    fn test_is_empty_false_after_construction() {
        let interner = Interner::new();
        assert!(!interner.is_empty());
    }

    // -- Stress / integration tests -----------------------------------------

    #[test]
    fn test_many_unique_strings() {
        let mut interner = Interner::new();
        let initial_len = interner.len();
        let count = 10_000;

        let symbols: Vec<Symbol> = (0..count)
            .map(|i| interner.intern(&format!("sym_stress_{}", i)))
            .collect();

        assert_eq!(interner.len(), initial_len + count);

        // Verify all symbols resolve correctly
        for (i, &sym) in symbols.iter().enumerate() {
            assert_eq!(interner.resolve(sym), format!("sym_stress_{}", i));
        }

        // Verify deduplication: re-interning should return same symbols
        for (i, &original_sym) in symbols.iter().enumerate() {
            let re_interned = interner.intern(&format!("sym_stress_{}", i));
            assert_eq!(re_interned, original_sym);
        }

        assert_eq!(interner.len(), initial_len + count);
    }

    #[test]
    fn test_symbol_as_hash_map_key() {
        use crate::common::fx_hash::FxHashMap;

        let mut interner = Interner::new();
        let sym_x = interner.intern("x");
        let sym_y = interner.intern("y");

        let mut table: FxHashMap<Symbol, i32> = FxHashMap::default();
        table.insert(sym_x, 10);
        table.insert(sym_y, 20);

        assert_eq!(table.get(&sym_x), Some(&10));
        assert_eq!(table.get(&sym_y), Some(&20));
        assert_eq!(table.get(&interner.intern("x")), Some(&10));
    }

    #[test]
    fn test_symbol_ordering_reflects_insertion_order() {
        let mut interner = Interner::new();
        let first = interner.intern("aaa_first");
        let second = interner.intern("zzz_second");
        // Symbol ordering is by index (insertion order), not alphabetical
        assert!(first < second);
    }

    #[test]
    fn test_debug_format_interner() {
        let interner = Interner::new();
        let dbg = format!("{:?}", interner);
        assert!(dbg.contains("Interner"));
        assert!(dbg.contains("count"));
    }

    #[test]
    fn test_empty_string_roundtrip() {
        let mut interner = Interner::new();
        let sym = interner.intern("");
        assert_eq!(sym, Symbol::EMPTY);
        assert_eq!(interner.resolve(sym), "");
        assert!(interner.contains(""));
    }

    #[test]
    fn test_whitespace_strings_are_distinct() {
        let mut interner = Interner::new();
        let space = interner.intern(" ");
        let tab = interner.intern("\t");
        let newline = interner.intern("\n");
        assert_ne!(space, tab);
        assert_ne!(tab, newline);
        assert_ne!(space, newline);
    }

    #[test]
    fn test_case_sensitive() {
        let mut interner = Interner::new();
        let lower = interner.intern("main");
        let upper = interner.intern("Main");
        let all_upper = interner.intern("MAIN");
        assert_ne!(lower, upper);
        assert_ne!(upper, all_upper);
        assert_ne!(lower, all_upper);
    }
}
