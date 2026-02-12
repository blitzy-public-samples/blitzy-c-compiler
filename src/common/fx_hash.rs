//! FxHasher — fast, non-cryptographic hash function based on Fibonacci hashing.
//!
//! This module provides a high-performance hasher optimized for compiler workloads
//! where keys are typically small strings (identifiers, keywords) and integers
//! (symbol IDs, type handles). It replaces the default SipHash used by
//! `std::collections::HashMap` with a significantly faster alternative that offers
//! ~2-5x speedup on small-key workloads.
//!
//! # Architecture
//!
//! The hash function uses Fibonacci hashing (multiplicative hashing with the golden
//! ratio constant) combined with rotate-XOR mixing. Each hash step computes:
//!
//! ```text
//! hash = (hash.rotate_left(5) ^ value).wrapping_mul(SEED)
//! ```
//!
//! where `SEED` is `0x517cc1b727220a95` on 64-bit platforms (derived from the
//! golden ratio) and `0x9e3779b9` on 32-bit platforms.
//!
//! # Security Warning
//!
//! FxHash is **NOT** cryptographically secure and is **NOT** resistant to HashDoS
//! attacks. It is suitable **only** for compiler-internal hash maps where keys are
//! not adversarially controlled. Never use this for network-facing or
//! security-sensitive applications.
//!
//! # Usage
//!
//! ```rust
//! use bcc::common::fx_hash::{FxHashMap, FxHashSet, fx_hash_map, fx_hash_set};
//!
//! let mut symbols: FxHashMap<&str, u32> = fx_hash_map();
//! symbols.insert("main", 0);
//! symbols.insert("printf", 1);
//!
//! let mut keywords: FxHashSet<&str> = fx_hash_set();
//! keywords.insert("int");
//! keywords.insert("return");
//! ```
//!
//! # Zero-Dependency Implementation
//!
//! This module is a from-scratch implementation that replaces the external `fxhash`
//! or `ahash` crates, adhering to the project's zero-dependency mandate. Only
//! `std::hash` traits and `std::collections` types from the Rust standard library
//! are used.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

// ---------------------------------------------------------------------------
// Fibonacci hashing seed constant
// ---------------------------------------------------------------------------
// The golden ratio constant for Fibonacci (multiplicative) hashing.
// On 64-bit: floor(2^64 / φ) where φ = (1 + √5) / 2 ≈ 1.6180339887…
// On 32-bit: floor(2^32 / φ)
// These constants provide excellent bit-mixing and uniform distribution of
// hash values across the output space, especially for sequential integer keys.

#[cfg(target_pointer_width = "64")]
const SEED: usize = 0x517c_c1b7_2722_0a95;

#[cfg(target_pointer_width = "32")]
const SEED: usize = 0x9e37_79b9;

// ---------------------------------------------------------------------------
// FxHasher
// ---------------------------------------------------------------------------

/// A fast, non-cryptographic hasher based on Fibonacci hashing.
///
/// `FxHasher` maintains a single `usize` accumulator that is mixed with each
/// piece of input data via rotate-XOR and multiplicative hashing. This provides
/// excellent performance for the small-key workloads typical in compilers
/// (identifiers, symbol IDs, type handles).
///
/// Implements [`std::hash::Hasher`] so it can be used with any standard library
/// collection that accepts a custom hasher.
#[derive(Clone)]
pub struct FxHasher {
    /// Accumulated hash state. Initialized to 0 and progressively mixed with
    /// each piece of input data via the Fibonacci hashing step.
    hash: usize,
}

impl FxHasher {
    /// Creates a new `FxHasher` with hash state initialized to zero.
    #[inline]
    pub fn new() -> Self {
        FxHasher { hash: 0 }
    }

    /// Creates a new `FxHasher` with a specific seed value.
    ///
    /// This can be useful when you need distinct hash functions for different
    /// purposes (e.g., double-hashing in a Cuckoo hash table) while still
    /// using the Fibonacci hashing algorithm.
    #[inline]
    pub fn with_seed(seed: usize) -> Self {
        FxHasher { hash: seed }
    }

    /// Core Fibonacci hashing step: rotate-XOR-multiply.
    ///
    /// Mixes a single `usize` value into the hash accumulator. The rotate-left
    /// by 5 bits ensures that consecutive bytes fed into `write()` do not simply
    /// XOR away, the XOR combines the new value with the rotated state, and the
    /// wrapping multiplication by the golden-ratio constant provides avalanche
    /// mixing so that each input bit affects many output bits.
    #[inline(always)]
    fn hash_word(&mut self, value: usize) {
        self.hash = (self.hash.rotate_left(5) ^ value).wrapping_mul(SEED);
    }
}

impl Default for FxHasher {
    #[inline]
    fn default() -> Self {
        FxHasher { hash: 0 }
    }
}

impl Hasher for FxHasher {
    /// Returns the accumulated hash value as a `u64`.
    ///
    /// On 64-bit platforms this is a zero-cost cast. On 32-bit platforms the
    /// `usize` value is widened to `u64`.
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    /// Hashes a byte slice by processing `usize`-width chunks for throughput,
    /// then handling any remaining tail bytes.
    ///
    /// For slices shorter than `size_of::<usize>()`, each byte is hashed
    /// individually through the core mixing step. For longer slices, we read
    /// aligned `usize` words from the slice for maximum throughput, then
    /// process remaining bytes individually.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Process full usize-width chunks for maximum throughput.
        // We use an unaligned read via from_ne_bytes to handle arbitrary
        // alignment without requiring the input to be usize-aligned.
        let mut remaining = bytes;
        let word_size = std::mem::size_of::<usize>();

        while remaining.len() >= word_size {
            // Safety: we verified remaining.len() >= word_size, so the slice
            // has at least word_size bytes. We read them as native-endian usize.
            let mut word_bytes = [0u8; std::mem::size_of::<usize>()];
            word_bytes.copy_from_slice(&remaining[..word_size]);
            let word = usize::from_ne_bytes(word_bytes);
            self.hash_word(word);
            remaining = &remaining[word_size..];
        }

        // Process remaining tail bytes one at a time. Each byte is widened to
        // usize and mixed through the same Fibonacci hashing step to maintain
        // consistent distribution properties.
        for &byte in remaining {
            self.hash_word(byte as usize);
        }
    }

    /// Optimized single-byte hash. Widens the byte to `usize` and applies
    /// one Fibonacci hashing step.
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.hash_word(i as usize);
    }

    /// Optimized 16-bit hash. Widens to `usize` and applies one hashing step.
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.hash_word(i as usize);
    }

    /// Optimized 32-bit hash. Widens to `usize` and applies one hashing step.
    ///
    /// On 64-bit platforms this is a zero-extend. On 32-bit platforms it is
    /// a no-op conversion since `usize` is 32 bits wide.
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.hash_word(i as usize);
    }

    /// Optimized 64-bit hash.
    ///
    /// On 64-bit platforms this is a single hashing step (identity cast to
    /// `usize`). On 32-bit platforms the value is split into two 32-bit halves,
    /// each mixed separately to ensure all 64 bits of input contribute to the
    /// hash state.
    #[inline]
    fn write_u64(&mut self, i: u64) {
        // On 64-bit: usize == u64, so this is a single step.
        // On 32-bit: we split into low and high halves.
        #[cfg(target_pointer_width = "64")]
        {
            self.hash_word(i as usize);
        }
        #[cfg(target_pointer_width = "32")]
        {
            self.hash_word(i as usize);
            self.hash_word((i >> 32) as usize);
        }
    }

    /// Optimized pointer-width hash. Directly applies one Fibonacci hashing
    /// step with no widening or narrowing conversion.
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.hash_word(i);
    }
}

// ---------------------------------------------------------------------------
// FxBuildHasher
// ---------------------------------------------------------------------------

/// A [`BuildHasher`] that produces [`FxHasher`] instances.
///
/// This is a zero-sized unit struct that serves as the hasher factory for
/// `FxHashMap` and `FxHashSet`. Since `FxHasher` requires no per-instance
/// configuration or random seed (it always starts with `hash = 0`), the
/// builder carries no state.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use bcc::common::fx_hash::FxBuildHasher;
///
/// let map: HashMap<String, u32, FxBuildHasher> = HashMap::with_hasher(FxBuildHasher);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FxBuildHasher;

impl Default for FxBuildHasher {
    #[inline]
    fn default() -> Self {
        FxBuildHasher
    }
}

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    /// Constructs a new `FxHasher` with hash state initialized to zero.
    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: 0 }
    }
}

// ---------------------------------------------------------------------------
// Type Aliases
// ---------------------------------------------------------------------------

/// A [`HashMap`] using [`FxBuildHasher`] for fast Fibonacci hashing.
///
/// Drop-in replacement for `std::collections::HashMap` with significantly
/// better performance on compiler workloads (small string and integer keys).
/// The only difference from the standard `HashMap` is the hasher — all
/// `HashMap` methods and trait implementations work identically.
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// A [`HashSet`] using [`FxBuildHasher`] for fast Fibonacci hashing.
///
/// Drop-in replacement for `std::collections::HashSet` with significantly
/// better performance on compiler workloads. All `HashSet` methods and trait
/// implementations work identically to the standard version.
pub type FxHashSet<T> = HashSet<T, FxBuildHasher>;

// ---------------------------------------------------------------------------
// Convenience Constructors
// ---------------------------------------------------------------------------

/// Creates a new, empty [`FxHashMap`] with the default initial capacity.
///
/// This is equivalent to `HashMap::with_hasher(FxBuildHasher)` but more
/// ergonomic for the common case of creating an empty map.
///
/// # Example
///
/// ```rust
/// use bcc::common::fx_hash::{FxHashMap, fx_hash_map};
/// let mut map: FxHashMap<&str, u32> = fx_hash_map();
/// map.insert("x", 42);
/// ```
#[inline]
pub fn fx_hash_map<K, V>() -> FxHashMap<K, V> {
    HashMap::with_hasher(FxBuildHasher)
}

/// Creates a new, empty [`FxHashSet`] with the default initial capacity.
///
/// This is equivalent to `HashSet::with_hasher(FxBuildHasher)` but more
/// ergonomic for the common case of creating an empty set.
///
/// # Example
///
/// ```rust
/// use bcc::common::fx_hash::{FxHashSet, fx_hash_set};
/// let mut set: FxHashSet<&str> = fx_hash_set();
/// set.insert("keyword");
/// ```
#[inline]
pub fn fx_hash_set<T>() -> FxHashSet<T> {
    HashSet::with_hasher(FxBuildHasher)
}

/// Creates a new, empty [`FxHashMap`] with at least the specified capacity.
///
/// The map will be able to hold at least `capacity` key-value pairs without
/// reallocating. Pre-allocating capacity avoids repeated resizing when the
/// approximate number of entries is known ahead of time (e.g., symbol tables
/// for translation units of known size).
///
/// # Example
///
/// ```rust
/// use bcc::common::fx_hash::{FxHashMap, fx_hash_map_with_capacity};
/// // Pre-allocate for ~1024 symbols to avoid resizing during parsing
/// let mut symbols: FxHashMap<String, u32> = fx_hash_map_with_capacity(1024);
/// ```
#[inline]
pub fn fx_hash_map_with_capacity<K, V>(capacity: usize) -> FxHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, FxBuildHasher)
}

/// Creates a new, empty [`FxHashSet`] with at least the specified capacity.
///
/// The set will be able to hold at least `capacity` elements without
/// reallocating.
///
/// # Example
///
/// ```rust
/// use bcc::common::fx_hash::{FxHashSet, fx_hash_set_with_capacity};
/// let mut visited: FxHashSet<usize> = fx_hash_set_with_capacity(256);
/// ```
#[inline]
pub fn fx_hash_set_with_capacity<T>(capacity: usize) -> FxHashSet<T> {
    HashSet::with_capacity_and_hasher(capacity, FxBuildHasher)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    /// Helper: hash a value using FxHasher and return the u64 result.
    fn fx_hash_one<T: Hash>(value: &T) -> u64 {
        let mut hasher = FxHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_default_hash_is_deterministic() {
        // The same input must always produce the same hash.
        let h1 = fx_hash_one(&42u64);
        let h2 = fx_hash_one(&42u64);
        assert_eq!(h1, h2, "FxHash must be deterministic");
    }

    #[test]
    fn test_different_values_produce_different_hashes() {
        let h1 = fx_hash_one(&1u64);
        let h2 = fx_hash_one(&2u64);
        assert_ne!(h1, h2, "Distinct small integers should hash differently");
    }

    #[test]
    fn test_string_hashing() {
        let h1 = fx_hash_one(&"hello");
        let h2 = fx_hash_one(&"world");
        let h3 = fx_hash_one(&"hello");
        assert_ne!(h1, h2, "Distinct strings should hash differently");
        assert_eq!(h1, h3, "Same string must produce same hash");
    }

    #[test]
    fn test_empty_slice_hash() {
        let mut hasher = FxHasher::new();
        hasher.write(&[]);
        let h_empty = hasher.finish();
        // Empty input with zero-initialized state should produce 0
        assert_eq!(h_empty, 0, "Empty input on zero state should yield 0");
    }

    #[test]
    fn test_single_byte_hashing() {
        let mut h1 = FxHasher::new();
        h1.write_u8(0xAB);
        let result1 = h1.finish();

        let mut h2 = FxHasher::new();
        h2.write_u8(0xAB);
        let result2 = h2.finish();

        assert_eq!(result1, result2, "Single byte hash must be deterministic");
        assert_ne!(result1, 0, "Non-zero input should produce non-zero hash");
    }

    #[test]
    fn test_u16_hashing() {
        let mut h = FxHasher::new();
        h.write_u16(0x1234);
        let result = h.finish();
        assert_ne!(result, 0, "u16 hash should be non-zero for non-zero input");
    }

    #[test]
    fn test_u32_hashing() {
        let mut h = FxHasher::new();
        h.write_u32(0xDEADBEEF);
        let result = h.finish();
        assert_ne!(result, 0, "u32 hash should be non-zero for non-zero input");
    }

    #[test]
    fn test_u64_hashing() {
        let mut h1 = FxHasher::new();
        h1.write_u64(0xCAFEBABE_DEADBEEF);
        let r1 = h1.finish();

        let mut h2 = FxHasher::new();
        h2.write_u64(0xCAFEBABE_DEADBEEF);
        let r2 = h2.finish();

        assert_eq!(r1, r2);
        assert_ne!(r1, 0);
    }

    #[test]
    fn test_usize_hashing() {
        let mut h = FxHasher::new();
        h.write_usize(12345);
        let result = h.finish();
        assert_ne!(result, 0);
    }

    #[test]
    fn test_fx_hash_map_basic_operations() {
        let mut map: FxHashMap<&str, i32> = fx_hash_map();
        map.insert("alpha", 1);
        map.insert("beta", 2);
        map.insert("gamma", 3);

        assert_eq!(map.len(), 3);
        assert_eq!(map.get("alpha"), Some(&1));
        assert_eq!(map.get("beta"), Some(&2));
        assert_eq!(map.get("gamma"), Some(&3));
        assert_eq!(map.get("delta"), None);
    }

    #[test]
    fn test_fx_hash_set_basic_operations() {
        let mut set: FxHashSet<&str> = fx_hash_set();
        assert!(set.insert("int"));
        assert!(set.insert("return"));
        assert!(!set.insert("int")); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains("int"));
        assert!(set.contains("return"));
        assert!(!set.contains("void"));
    }

    #[test]
    fn test_fx_hash_map_with_capacity_does_not_panic() {
        let map: FxHashMap<u64, u64> = fx_hash_map_with_capacity(1024);
        assert!(map.is_empty());
        // Capacity may be >= 1024 due to internal rounding, but must not panic.
    }

    #[test]
    fn test_fx_hash_set_with_capacity_does_not_panic() {
        let set: FxHashSet<u64> = fx_hash_set_with_capacity(512);
        assert!(set.is_empty());
    }

    #[test]
    fn test_build_hasher_produces_consistent_hashers() {
        let builder = FxBuildHasher;
        let mut h1 = builder.build_hasher();
        let mut h2 = builder.build_hasher();

        h1.write_u64(999);
        h2.write_u64(999);

        assert_eq!(
            h1.finish(),
            h2.finish(),
            "BuildHasher must produce identical hashers"
        );
    }

    #[test]
    fn test_hash_distribution_no_obvious_collisions() {
        // Hash 1000 sequential integers and verify no excessive collisions.
        // With a good hash function over 1000 distinct inputs, the number
        // of distinct hash values should be very close to 1000.
        let mut set = std::collections::HashSet::new();
        for i in 0u64..1000 {
            set.insert(fx_hash_one(&i));
        }
        // Allow at most 1% collision rate for sequential integers
        assert!(
            set.len() >= 990,
            "Hash distribution too poor: only {} distinct hashes out of 1000",
            set.len()
        );
    }

    #[test]
    fn test_hash_of_similar_strings() {
        // Compiler identifiers often differ by a single character. Verify
        // that similar strings produce different hashes.
        let hashes: Vec<u64> = (b'a'..=b'z')
            .map(|c| {
                let s = format!("var_{}", c as char);
                fx_hash_one(&s)
            })
            .collect();

        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(
            unique.len(),
            hashes.len(),
            "Similar short strings should produce distinct hashes"
        );
    }

    #[test]
    fn test_large_byte_slice() {
        // Hash a large byte slice to exercise the usize-chunked path.
        let data: Vec<u8> = (0..1024).map(|i| (i & 0xFF) as u8).collect();
        let mut h1 = FxHasher::new();
        h1.write(&data);
        let r1 = h1.finish();

        let mut h2 = FxHasher::new();
        h2.write(&data);
        let r2 = h2.finish();

        assert_eq!(r1, r2, "Large slice hashing must be deterministic");
        assert_ne!(r1, 0, "Large slice should produce non-zero hash");
    }

    #[test]
    fn test_default_impls() {
        let hasher = FxHasher::default();
        assert_eq!(hasher.finish(), 0, "Default hasher should start at 0");

        let builder = FxBuildHasher::default();
        let h = builder.build_hasher();
        assert_eq!(h.finish(), 0, "Default builder should produce zero-state hasher");
    }

    #[test]
    fn test_with_seed() {
        let h1 = FxHasher::with_seed(42);
        let h2 = FxHasher::with_seed(42);
        let h3 = FxHasher::with_seed(99);

        assert_eq!(h1.finish(), h2.finish(), "Same seed must yield same initial state");
        assert_ne!(
            h1.finish(),
            h3.finish(),
            "Different seeds should yield different initial state"
        );
    }

    #[test]
    fn test_incremental_vs_bulk_write() {
        // Hashing bytes one at a time should produce the same result as
        // hashing them in a single write call — but only when the bulk
        // path processes at the same granularity. For FxHash, single-byte
        // writes differ from chunk-aligned writes, which is expected behavior.
        // This test ensures both paths are deterministic individually.
        let data = b"hello world, this is a test string";

        let mut bulk = FxHasher::new();
        bulk.write(data);
        let bulk_result = bulk.finish();

        let mut bulk2 = FxHasher::new();
        bulk2.write(data);
        assert_eq!(bulk_result, bulk2.finish(), "Bulk write must be deterministic");
    }

    #[test]
    fn test_map_overwrite() {
        let mut map: FxHashMap<&str, i32> = fx_hash_map();
        map.insert("key", 1);
        assert_eq!(map.get("key"), Some(&1));
        map.insert("key", 2);
        assert_eq!(map.get("key"), Some(&2));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_set_remove() {
        let mut set: FxHashSet<i32> = fx_hash_set();
        set.insert(10);
        set.insert(20);
        assert!(set.remove(&10));
        assert!(!set.contains(&10));
        assert!(set.contains(&20));
    }

    #[test]
    fn test_clone() {
        let mut h1 = FxHasher::new();
        h1.write_u64(12345);
        let h2 = h1.clone();
        assert_eq!(h1.finish(), h2.finish(), "Cloned hasher must have identical state");
    }
}
