//! Fast, non-cryptographic Fibonacci hashing (FxHash) implementation.
//!
//! Provides `FxHashMap<K,V>` and `FxHashSet<K>` type aliases that wrap
//! `std::collections::HashMap`/`HashSet` with `BuildHasherDefault<FxHasher>`.
//! This replaces external `fxhash` or `ahash` crates in compliance with the
//! zero-dependency mandate.
//!
//! # Performance
//!
//! FxHash uses Fibonacci hashing with architecture-width multiply for speed.
//! It is significantly faster than the default `SipHash` used by
//! `std::collections::HashMap`, making it ideal for compiler-internal data
//! structures such as symbol tables, string interning maps, and all lookup
//! structures throughout the BCC pipeline.
//!
//! # Security
//!
//! This hasher is intentionally **NOT** cryptographic and is **NOT**
//! HashDoS-resistant. It is designed solely for compiler-internal use
//! where input is trusted (source code identifiers, types, IR nodes).
//!
//! # Algorithm
//!
//! The core operation is: `hash = (hash.rotate_left(5) ^ word).wrapping_mul(SEED)`
//! where `SEED` is the golden-ratio constant for the architecture's pointer width:
//! - 64-bit: `0x517cc1b727220a95` (floor(2^64 / φ))
//! - 32-bit: `0x9e3779b9` (floor(2^32 / φ))

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

// ---------------------------------------------------------------------------
// Fibonacci hashing constants (golden-ratio derived)
// ---------------------------------------------------------------------------

/// Golden-ratio constant for 64-bit architectures: floor(2^64 / φ).
/// This is the same constant used by rustc's internal `FxHasher`.
#[cfg(target_pointer_width = "64")]
const SEED: usize = 0x517c_c1b7_2722_0a95;

/// Golden-ratio constant for 32-bit architectures: floor(2^32 / φ).
#[cfg(target_pointer_width = "32")]
const SEED: usize = 0x9e37_79b9;

// ---------------------------------------------------------------------------
// FxHasher — the core hasher
// ---------------------------------------------------------------------------

/// Fast, non-cryptographic hasher based on FxHash (Firefox/Rustc hash).
///
/// Uses Fibonacci hashing with architecture-width multiply for speed.
/// Implements [`std::hash::Hasher`] so it integrates seamlessly with
/// `HashMap` and `HashSet` via [`BuildHasherDefault`].
///
/// # Examples
///
/// ```ignore
/// use bcc::common::fx_hash::{FxHashMap, FxHashSet};
///
/// let mut map: FxHashMap<&str, i32> = FxHashMap::default();
/// map.insert("hello", 42);
/// assert_eq!(map.get("hello"), Some(&42));
/// ```
pub struct FxHasher {
    /// Accumulated hash state. Starts at 0 and is updated by each `write*`
    /// call through the `add_to_hash` mixing function.
    hash: usize,
}

impl FxHasher {
    /// Create a new `FxHasher` with initial state of zero.
    #[inline]
    pub fn new() -> Self {
        FxHasher { hash: 0 }
    }

    /// Core hash accumulation step.
    ///
    /// 1. `rotate_left(5)` — spreads existing bits to avoid clustering.
    /// 2. `^ word` — incorporates new data via XOR.
    /// 3. `.wrapping_mul(SEED)` — the Fibonacci multiplication provides
    ///    excellent avalanche properties, ensuring small input changes
    ///    produce large hash changes.
    #[inline]
    fn add_to_hash(&mut self, word: usize) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

// ---------------------------------------------------------------------------
// Default trait — required by BuildHasherDefault<FxHasher>
// ---------------------------------------------------------------------------

impl Default for FxHasher {
    /// Returns a new `FxHasher` with initial hash state of zero.
    /// This is invoked by [`BuildHasherDefault`] each time a hash operation
    /// begins on a `HashMap`/`HashSet`.
    #[inline]
    fn default() -> Self {
        FxHasher { hash: 0 }
    }
}

// ---------------------------------------------------------------------------
// Clone trait — allows copying hasher state
// ---------------------------------------------------------------------------

impl Clone for FxHasher {
    #[inline]
    fn clone(&self) -> Self {
        FxHasher { hash: self.hash }
    }
}

// ---------------------------------------------------------------------------
// Hasher trait implementation
// ---------------------------------------------------------------------------

impl Hasher for FxHasher {
    /// Returns the accumulated hash value as a `u64`.
    ///
    /// On 64-bit platforms this is a zero-cost conversion. On 32-bit
    /// platforms the `usize` value is zero-extended to `u64`.
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    /// Hash an arbitrary byte slice.
    ///
    /// Processes data in `usize`-sized chunks (8 bytes on 64-bit, 4 bytes
    /// on 32-bit) for maximum throughput. Trailing bytes that do not fill
    /// a complete `usize` are zero-padded into a final chunk.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Process full usize-width chunks for speed.
        let chunks = bytes.chunks(std::mem::size_of::<usize>());
        for chunk in chunks {
            // Zero-pad partial trailing chunks into a usize buffer.
            let mut buf = [0u8; std::mem::size_of::<usize>()];
            buf[..chunk.len()].copy_from_slice(chunk);
            let word = usize::from_ne_bytes(buf);
            self.add_to_hash(word);
        }
    }

    /// Hash a single `u8` value.
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(i as usize);
    }

    /// Hash a single `u16` value.
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_to_hash(i as usize);
    }

    /// Hash a single `u32` value.
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(i as usize);
    }

    /// Hash a single `u64` value.
    ///
    /// On 32-bit platforms, the upper 32 bits are truncated. This is
    /// acceptable for a non-cryptographic compiler-internal hasher.
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i as usize);
    }

    /// Hash a single `usize` value — the most common hot path for
    /// hashing integer keys and interned symbol handles.
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i);
    }
}

// ---------------------------------------------------------------------------
// Type aliases — the primary public API for the rest of the compiler
// ---------------------------------------------------------------------------

/// `HashMap` using `FxHasher` — drop-in replacement for `std::collections::HashMap`
/// with significantly faster hashing for compiler-internal use.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::fx_hash::{FxHashMap, fx_hash_map};
///
/// let mut symbols: FxHashMap<String, u32> = fx_hash_map();
/// symbols.insert("main".to_string(), 0);
/// ```
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// `HashSet` using `FxHasher` — drop-in replacement for `std::collections::HashSet`
/// with significantly faster hashing for compiler-internal use.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::fx_hash::{FxHashSet, fx_hash_set};
///
/// let mut keywords: FxHashSet<&str> = fx_hash_set();
/// keywords.insert("int");
/// keywords.insert("return");
/// ```
pub type FxHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

// ---------------------------------------------------------------------------
// Convenience constructor functions
// ---------------------------------------------------------------------------

/// Create an empty [`FxHashMap`].
///
/// Equivalent to `FxHashMap::default()` but reads more clearly at call sites.
#[inline]
pub fn fx_hash_map<K, V>() -> FxHashMap<K, V> {
    FxHashMap::default()
}

/// Create an empty [`FxHashMap`] with pre-allocated capacity.
///
/// Use this when the approximate number of entries is known in advance
/// to avoid rehashing during population. Critical for large symbol tables.
#[inline]
pub fn fx_hash_map_with_capacity<K, V>(capacity: usize) -> FxHashMap<K, V> {
    FxHashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::default())
}

/// Create an empty [`FxHashSet`].
///
/// Equivalent to `FxHashSet::default()` but reads more clearly at call sites.
#[inline]
pub fn fx_hash_set<K>() -> FxHashSet<K> {
    FxHashSet::default()
}

/// Create an empty [`FxHashSet`] with pre-allocated capacity.
///
/// Use this when the approximate number of entries is known in advance
/// to avoid rehashing during population.
#[inline]
pub fn fx_hash_set_with_capacity<K>(capacity: usize) -> FxHashSet<K> {
    FxHashSet::with_capacity_and_hasher(capacity, BuildHasherDefault::default())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    /// Helper: hash a value using FxHasher and return the u64 result.
    fn hash_one<T: Hash>(val: &T) -> u64 {
        let mut hasher = FxHasher::default();
        val.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_default_creates_zero_state() {
        let h = FxHasher::default();
        assert_eq!(h.hash, 0);
    }

    #[test]
    fn test_new_creates_zero_state() {
        let h = FxHasher::new();
        assert_eq!(h.hash, 0);
    }

    #[test]
    fn test_determinism() {
        // Same input must always produce the same hash.
        let h1 = hash_one(&42u64);
        let h2 = hash_one(&42u64);
        assert_eq!(h1, h2);

        let h3 = hash_one(&"hello");
        let h4 = hash_one(&"hello");
        assert_eq!(h3, h4);
    }

    #[test]
    fn test_different_inputs_different_hashes() {
        // Basic collision resistance — different inputs should (with very
        // high probability) produce different hashes.
        let h1 = hash_one(&1u64);
        let h2 = hash_one(&2u64);
        assert_ne!(h1, h2);

        let h3 = hash_one(&"foo");
        let h4 = hash_one(&"bar");
        assert_ne!(h3, h4);
    }

    #[test]
    fn test_write_u8() {
        let mut h = FxHasher::default();
        h.write_u8(0xFF);
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_u16() {
        let mut h = FxHasher::default();
        h.write_u16(0xBEEF);
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_u32() {
        let mut h = FxHasher::default();
        h.write_u32(0xDEAD_BEEF);
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_u64() {
        let mut h = FxHasher::default();
        h.write_u64(0xDEAD_BEEF_CAFE_BABEu64);
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_usize() {
        let mut h = FxHasher::default();
        h.write_usize(12345);
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_bytes() {
        let mut h = FxHasher::default();
        h.write(b"hello world");
        assert_ne!(h.finish(), 0);
    }

    #[test]
    fn test_write_empty_bytes() {
        // Hashing empty bytes should still be valid (hash stays at 0
        // since chunks() produces no items).
        let h1 = FxHasher::default();
        let mut h2 = FxHasher::default();
        h2.write(b"");
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_fx_hash_map_default() {
        let map: FxHashMap<&str, i32> = FxHashMap::default();
        assert!(map.is_empty());
    }

    #[test]
    fn test_fx_hash_map_insert_lookup() {
        let mut map: FxHashMap<String, i32> = fx_hash_map();
        map.insert("alpha".to_string(), 1);
        map.insert("beta".to_string(), 2);
        map.insert("gamma".to_string(), 3);

        assert_eq!(map.get("alpha"), Some(&1));
        assert_eq!(map.get("beta"), Some(&2));
        assert_eq!(map.get("gamma"), Some(&3));
        assert_eq!(map.get("delta"), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_fx_hash_map_with_capacity() {
        let mut map: FxHashMap<u32, u32> = fx_hash_map_with_capacity(1024);
        for i in 0..1024 {
            map.insert(i, i * 2);
        }
        for i in 0..1024 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_fx_hash_set_default() {
        let set: FxHashSet<&str> = FxHashSet::default();
        assert!(set.is_empty());
    }

    #[test]
    fn test_fx_hash_set_insert_contains() {
        let mut set: FxHashSet<String> = fx_hash_set();
        set.insert("int".to_string());
        set.insert("return".to_string());
        set.insert("void".to_string());

        assert!(set.contains("int"));
        assert!(set.contains("return"));
        assert!(set.contains("void"));
        assert!(!set.contains("float"));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_fx_hash_set_with_capacity() {
        let mut set: FxHashSet<u64> = fx_hash_set_with_capacity(256);
        for i in 0..256u64 {
            set.insert(i);
        }
        assert_eq!(set.len(), 256);
        for i in 0..256u64 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn test_fx_hash_map_overwrite() {
        let mut map: FxHashMap<&str, i32> = fx_hash_map();
        map.insert("key", 1);
        assert_eq!(map.get("key"), Some(&1));
        map.insert("key", 2);
        assert_eq!(map.get("key"), Some(&2));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_fx_hash_map_remove() {
        let mut map: FxHashMap<&str, i32> = fx_hash_map();
        map.insert("a", 1);
        map.insert("b", 2);
        assert_eq!(map.remove("a"), Some(1));
        assert_eq!(map.get("a"), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_fx_hash_set_remove() {
        let mut set: FxHashSet<i32> = fx_hash_set();
        set.insert(10);
        set.insert(20);
        assert!(set.remove(&10));
        assert!(!set.contains(&10));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_clone_hasher() {
        let mut h1 = FxHasher::default();
        h1.write_u64(999);
        let h2 = h1.clone();
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_many_keys_no_panic() {
        // Stress test — ensure no panics with many distinct keys.
        let mut map: FxHashMap<u64, u64> = fx_hash_map_with_capacity(10_000);
        for i in 0..10_000u64 {
            map.insert(i, i);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000u64 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_string_key_hashing() {
        // Verify string keys work correctly (common use case for symbol tables).
        let mut map: FxHashMap<String, usize> = fx_hash_map();
        let identifiers = vec![
            "main",
            "printf",
            "argc",
            "argv",
            "i",
            "j",
            "k",
            "__builtin_expect",
            "__attribute__",
            "struct",
            "union",
        ];
        for (idx, &id) in identifiers.iter().enumerate() {
            map.insert(id.to_string(), idx);
        }
        for (idx, &id) in identifiers.iter().enumerate() {
            assert_eq!(map.get(id), Some(&idx));
        }
    }
}
