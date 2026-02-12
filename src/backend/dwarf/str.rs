//! DWARF `.debug_str` section string table for debug information.
//!
//! This module implements `DebugStrTable`, the string table backing the DWARF v4
//! `.debug_str` section. All string-valued attributes in `.debug_info` DIEs
//! (producer name, source file names, compilation directory, function names,
//! variable names, type names) are stored here using `DW_FORM_strp` encoding:
//! the `.debug_info` section records a 4-byte offset into `.debug_str` rather
//! than embedding the string inline.
//!
//! # Architecture
//!
//! The table stores deduplicated, null-terminated UTF-8 strings in a contiguous
//! `Vec<u8>` buffer. An `FxHashMap<String, u32>` provides O(1) deduplication
//! lookups — when a string is added for the first time, it is appended to the
//! buffer and its byte offset is recorded in the map. Subsequent additions of
//! the same string return the already-recorded offset without appending any
//! additional data.
//!
//! # Initial State
//!
//! Per the DWARF specification, the `.debug_str` section begins with a single
//! null byte at offset 0, representing the empty string. This ensures that any
//! `DW_FORM_strp` attribute with value 0 resolves to an empty string rather
//! than pointing to invalid data.
//!
//! # Offset Stability
//!
//! The table is strictly append-only: once a string is added and its offset is
//! returned, that offset is guaranteed to remain valid and unchanged for the
//! lifetime of the table. This is critical because `DW_FORM_strp` offsets are
//! written into `.debug_info` during emission and must not be invalidated by
//! later additions.
//!
//! # Conditionality (Section 0.7.10)
//!
//! This table is only constructed and serialized when the `-g` flag is active.
//! When `-g` is absent, no `DebugStrTable` instance is created, and no
//! `.debug_str` section appears in the ELF output — enforcing the zero-leakage
//! requirement.
//!
//! # Usage Example
//!
//! ```rust
//! use bcc::backend::dwarf::debug_str::DebugStrTable;
//!
//! let mut table = DebugStrTable::new();
//!
//! // Add strings used by DW_FORM_strp attributes in .debug_info:
//! let producer_off = table.add_string("bcc 1.0.0");
//! let file_off     = table.add_string("main.c");
//! let func_off     = table.add_string("main");
//!
//! // Duplicate additions return the same offset:
//! assert_eq!(table.add_string("main.c"), file_off);
//!
//! // Retrieve the raw section bytes for inclusion in ELF output:
//! let section_data = table.as_bytes();
//! ```
//!
//! # Zero-Dependency Implementation
//!
//! This module uses only the Rust standard library and the project-internal
//! `FxHashMap` from `crate::common::fx_hash`, adhering to the project's strict
//! zero-external-dependency mandate.

use crate::common::fx_hash::FxHashMap;

// ---------------------------------------------------------------------------
// DebugStrTable — DWARF .debug_str section builder
// ---------------------------------------------------------------------------

/// DWARF `.debug_str` section string table.
///
/// Stores deduplicated, null-terminated UTF-8 strings and provides
/// `DW_FORM_strp` byte-offset management for referencing strings from
/// `.debug_info` DIE attributes.
///
/// # Layout
///
/// ```text
/// Offset  Content
/// ------  -------
/// 0x0000  0x00                     (empty string — required by DWARF spec)
/// 0x0001  b'b' b'c' b'c' 0x00     ("bcc")
/// 0x0005  b'm' b'a' b'i' b'n' 0x00 ("main")
/// ...
/// ```
///
/// Each string is immediately followed by a null terminator byte (`0x00`).
/// The offset of a string is the byte position of its first character within
/// the `data` buffer (or offset 0 for the empty string, which is the initial
/// null byte).
pub struct DebugStrTable {
    /// Raw `.debug_str` section bytes: concatenated null-terminated strings.
    ///
    /// Initialized with a single `0x00` byte at index 0 representing the
    /// empty string entry mandated by the DWARF specification.
    data: Vec<u8>,

    /// String-to-offset deduplication map.
    ///
    /// Maps each unique string (as a `String` key) to its byte offset within
    /// `data`. Provides O(1) lookup to avoid appending duplicate strings.
    /// Uses `FxHashMap` (Fibonacci hashing) instead of the standard library
    /// `HashMap` for faster hashing on the small-string workloads typical in
    /// compiler debug information (function names, variable names, file paths).
    offsets: FxHashMap<String, u32>,
}

impl DebugStrTable {
    /// Creates a new `DebugStrTable` with the DWARF-mandated initial state.
    ///
    /// The table starts with:
    /// - `data` containing a single null byte at offset 0 (the empty string).
    /// - `offsets` containing the mapping `"" → 0`.
    ///
    /// This initial null byte is required by the DWARF specification so that
    /// any `DW_FORM_strp` attribute with value 0 resolves to an empty string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let table = DebugStrTable::new();
    /// assert_eq!(table.section_size(), 1);   // Just the initial null byte.
    /// assert!(table.is_empty());             // No "real" strings yet.
    /// assert!(table.contains(""));           // Empty string is always present.
    /// assert_eq!(table.get_offset(""), Some(0));
    /// ```
    pub fn new() -> Self {
        // Start with the DWARF-mandated null byte at offset 0.
        let data = vec![0u8];

        // Pre-populate the offsets map with the empty string entry.
        let mut offsets: FxHashMap<String, u32> =
            FxHashMap::with_hasher(crate::common::fx_hash::FxBuildHasher);
        offsets.insert(String::new(), 0);

        DebugStrTable { data, offsets }
    }

    // -----------------------------------------------------------------------
    // String addition with deduplication
    // -----------------------------------------------------------------------

    /// Adds a string to the table and returns its byte offset within the
    /// `.debug_str` section data.
    ///
    /// If the string is already present (exact match via `FxHashMap` lookup),
    /// the existing offset is returned without modifying the table. Otherwise
    /// the string bytes and a null terminator are appended, the new offset is
    /// recorded in the deduplication map, and the offset is returned.
    ///
    /// The returned `u32` offset is suitable for direct use as a
    /// `DW_FORM_strp` attribute value in `.debug_info`.
    ///
    /// # Parameters
    ///
    /// - `s` — The string to add. Stored as UTF-8 followed by a null byte.
    ///
    /// # Returns
    ///
    /// The byte offset of `s` within the `.debug_str` section buffer.
    ///
    /// # Panics
    ///
    /// Panics if the section size would exceed `u32::MAX` bytes, which would
    /// overflow the 4-byte `DW_FORM_strp` offset encoding. In practice this
    /// limit (~4 GiB) is far beyond any realistic debug string table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// let off1 = table.add_string("main");
    /// let off2 = table.add_string("main"); // Deduplication — same offset.
    /// assert_eq!(off1, off2);
    /// ```
    pub fn add_string(&mut self, s: &str) -> u32 {
        // Fast path: return the existing offset if the string is already present.
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }

        // Record the offset where this new string will begin.
        let offset = self.data.len() as u32;

        // Assert that we haven't exceeded the DW_FORM_strp 4-byte offset limit.
        // The cast above is safe as long as data.len() fits in u32.
        // The condition: total_after_append = data.len() + s.len() + 1 (null byte)
        // must not exceed u32::MAX. Equivalent: data.len() + s.len() < u32::MAX.
        assert!(
            (self.data.len() as u64) + (s.len() as u64) < u32::MAX as u64,
            "DWARF .debug_str section exceeds u32::MAX ({} bytes); \
             DW_FORM_strp offsets cannot represent positions beyond 4 GiB",
            self.data.len() + s.len() + 1,
        );

        // Append the string bytes followed by a null terminator.
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0u8);

        // Record the offset in the deduplication map.
        self.offsets.insert(s.to_owned(), offset);

        offset
    }

    /// Looks up the byte offset of an existing string without adding it.
    ///
    /// Returns `Some(offset)` if the string has previously been added via
    /// [`add_string`](Self::add_string), or `None` if the string is not in
    /// the table.
    ///
    /// # Parameters
    ///
    /// - `s` — The string to look up.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// assert_eq!(table.get_offset("main"), None);
    /// table.add_string("main");
    /// assert!(table.get_offset("main").is_some());
    /// ```
    pub fn get_offset(&self, s: &str) -> Option<u32> {
        self.offsets.get(s).copied()
    }

    /// Checks whether the given string is already present in the table.
    ///
    /// Equivalent to `self.get_offset(s).is_some()` but more semantically
    /// expressive when only a boolean check is needed.
    ///
    /// # Parameters
    ///
    /// - `s` — The string to test for membership.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// assert!(table.contains(""));       // Empty string is always present.
    /// assert!(!table.contains("main"));
    /// table.add_string("main");
    /// assert!(table.contains("main"));
    /// ```
    pub fn contains(&self, s: &str) -> bool {
        self.offsets.contains_key(s)
    }

    // -----------------------------------------------------------------------
    // Section data access
    // -----------------------------------------------------------------------

    /// Returns the raw `.debug_str` section bytes for inclusion in ELF output.
    ///
    /// The returned slice contains all stored null-terminated strings in the
    /// order they were added, starting with the initial null byte at offset 0.
    /// This data is written verbatim into the `.debug_str` ELF section with
    /// `SHT_PROGBITS` type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// table.add_string("abc");
    /// // data = [0x00, b'a', b'b', b'c', 0x00]
    /// assert_eq!(table.as_bytes(), &[0x00, b'a', b'b', b'c', 0x00]);
    /// ```
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns the current total size (in bytes) of the `.debug_str` section.
    ///
    /// This includes the initial null byte and all appended string data
    /// (string bytes plus their null terminators). Useful for computing
    /// section header sizes during ELF construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// assert_eq!(table.section_size(), 1); // Initial null byte only.
    /// table.add_string("hi");
    /// // 1 (initial null) + 2 (string "hi") + 1 (null terminator) = 4
    /// assert_eq!(table.section_size(), 4);
    /// ```
    pub fn section_size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no "real" strings have been added to the table.
    ///
    /// The table always contains the initial null byte (the empty string
    /// entry), so `is_empty` returns `true` only when the section size is
    /// exactly 1 — meaning no calls to [`add_string`](Self::add_string)
    /// have been made with a non-empty string argument.
    ///
    /// # Note
    ///
    /// After `add_string("")` (which is a no-op due to the pre-populated
    /// empty-string entry), `is_empty()` still returns `true` because no
    /// new data was appended.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bcc::backend::dwarf::debug_str::DebugStrTable;
    /// let mut table = DebugStrTable::new();
    /// assert!(table.is_empty());
    /// table.add_string("main");
    /// assert!(!table.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        // The initial state has exactly 1 byte (the null byte at offset 0).
        // Any non-empty string addition increases the length beyond 1.
        self.data.len() <= 1
    }
}

impl Default for DebugStrTable {
    /// Equivalent to [`DebugStrTable::new()`].
    fn default() -> Self {
        Self::new()
    }
}

// Implement Debug manually so large section data is summarized rather than
// dumped in full, which would overwhelm diagnostic output for large builds.
impl std::fmt::Debug for DebugStrTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebugStrTable")
            .field("section_size", &self.data.len())
            .field("string_count", &self.offsets.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction -------------------------------------------------------

    #[test]
    fn test_new_table_has_initial_null_byte() {
        let table = DebugStrTable::new();
        assert_eq!(table.section_size(), 1);
        assert_eq!(table.as_bytes(), &[0x00]);
    }

    #[test]
    fn test_new_table_is_empty() {
        let table = DebugStrTable::new();
        assert!(table.is_empty());
    }

    #[test]
    fn test_new_table_contains_empty_string() {
        let table = DebugStrTable::new();
        assert!(table.contains(""));
        assert_eq!(table.get_offset(""), Some(0));
    }

    #[test]
    fn test_default_trait_matches_new() {
        let a = DebugStrTable::new();
        let b = DebugStrTable::default();
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_eq!(a.section_size(), b.section_size());
    }

    // -- Adding strings -----------------------------------------------------

    #[test]
    fn test_add_single_string() {
        let mut table = DebugStrTable::new();
        let offset = table.add_string("main");

        // The string starts right after the initial null byte.
        assert_eq!(offset, 1);

        // Section layout: [0x00, b'm', b'a', b'i', b'n', 0x00]
        assert_eq!(table.section_size(), 6);
        assert_eq!(table.as_bytes(), &[0x00, b'm', b'a', b'i', b'n', 0x00]);
    }

    #[test]
    fn test_add_multiple_strings() {
        let mut table = DebugStrTable::new();

        let off_abc = table.add_string("abc");
        let off_xyz = table.add_string("xyz");

        // "abc" starts at offset 1 (after initial null byte).
        assert_eq!(off_abc, 1);
        // "xyz" starts at offset 1 + 3 (abc) + 1 (null) = 5.
        assert_eq!(off_xyz, 5);

        // Total: 1 (init null) + 4 (abc\0) + 4 (xyz\0) = 9.
        assert_eq!(table.section_size(), 9);

        let expected: &[u8] = &[
            0x00, // initial empty string
            b'a', b'b', b'c', 0x00, // "abc"
            b'x', b'y', b'z', 0x00, // "xyz"
        ];
        assert_eq!(table.as_bytes(), expected);
    }

    #[test]
    fn test_add_empty_string_returns_zero() {
        let mut table = DebugStrTable::new();
        // Empty string is pre-populated at offset 0.
        let offset = table.add_string("");
        assert_eq!(offset, 0);
        // No additional data should have been appended.
        assert_eq!(table.section_size(), 1);
        assert!(table.is_empty());
    }

    // -- Deduplication ------------------------------------------------------

    #[test]
    fn test_deduplication_returns_same_offset() {
        let mut table = DebugStrTable::new();
        let off1 = table.add_string("hello");
        let off2 = table.add_string("hello");
        assert_eq!(off1, off2);

        // Section should contain the string only once.
        // 1 (init null) + 5 (hello) + 1 (null) = 7
        assert_eq!(table.section_size(), 7);
    }

    #[test]
    fn test_deduplication_does_not_affect_other_strings() {
        let mut table = DebugStrTable::new();
        let off_a = table.add_string("alpha");
        let off_b = table.add_string("beta");
        let off_a2 = table.add_string("alpha"); // Duplicate.
        let off_b2 = table.add_string("beta"); // Duplicate.

        assert_eq!(off_a, off_a2);
        assert_eq!(off_b, off_b2);
        assert_ne!(off_a, off_b);
    }

    #[test]
    fn test_prefix_strings_are_distinct() {
        // "main" and "main.c" are different strings even though one is a
        // prefix of the other. They must occupy separate entries.
        let mut table = DebugStrTable::new();
        let off1 = table.add_string("main");
        let off2 = table.add_string("main.c");
        assert_ne!(off1, off2);
    }

    // -- Lookup and containment ---------------------------------------------

    #[test]
    fn test_get_offset_for_existing_string() {
        let mut table = DebugStrTable::new();
        let offset = table.add_string("printf");
        assert_eq!(table.get_offset("printf"), Some(offset));
    }

    #[test]
    fn test_get_offset_for_missing_string_returns_none() {
        let table = DebugStrTable::new();
        assert_eq!(table.get_offset("missing"), None);
    }

    #[test]
    fn test_contains_for_existing_and_missing() {
        let mut table = DebugStrTable::new();
        table.add_string("foo");
        assert!(table.contains("foo"));
        assert!(!table.contains("bar"));
    }

    // -- Section data access ------------------------------------------------

    #[test]
    fn test_as_bytes_returns_complete_section() {
        let mut table = DebugStrTable::new();
        table.add_string("A");
        table.add_string("BB");

        // Layout: [0x00, b'A', 0x00, b'B', b'B', 0x00]
        let expected: &[u8] = &[0x00, b'A', 0x00, b'B', b'B', 0x00];
        assert_eq!(table.as_bytes(), expected);
    }

    #[test]
    fn test_section_size_grows_correctly() {
        let mut table = DebugStrTable::new();
        assert_eq!(table.section_size(), 1);

        table.add_string("x"); // +2 bytes (x + null)
        assert_eq!(table.section_size(), 3);

        table.add_string("yy"); // +3 bytes (yy + null)
        assert_eq!(table.section_size(), 6);

        table.add_string("x"); // Duplicate — no growth.
        assert_eq!(table.section_size(), 6);
    }

    #[test]
    fn test_is_empty_becomes_false_after_add() {
        let mut table = DebugStrTable::new();
        assert!(table.is_empty());

        table.add_string("test");
        assert!(!table.is_empty());
    }

    #[test]
    fn test_is_empty_remains_true_after_adding_empty_string() {
        let mut table = DebugStrTable::new();
        table.add_string("");
        assert!(table.is_empty());
    }

    // -- Null termination ---------------------------------------------------

    #[test]
    fn test_strings_are_null_terminated() {
        let mut table = DebugStrTable::new();
        let off = table.add_string("test") as usize;
        let bytes = table.as_bytes();

        // The string "test" should be at bytes[off..off+4].
        assert_eq!(&bytes[off..off + 4], b"test");
        // Followed by a null terminator.
        assert_eq!(bytes[off + 4], 0x00);
    }

    #[test]
    fn test_consecutive_strings_are_separated_by_null() {
        let mut table = DebugStrTable::new();
        let off1 = table.add_string("aa") as usize;
        let off2 = table.add_string("bb") as usize;

        let bytes = table.as_bytes();

        // "aa" at off1, followed by null at off1+2.
        assert_eq!(&bytes[off1..off1 + 2], b"aa");
        assert_eq!(bytes[off1 + 2], 0x00);

        // "bb" starts at off2 = off1 + 3.
        assert_eq!(off2, off1 + 3);
        assert_eq!(&bytes[off2..off2 + 2], b"bb");
        assert_eq!(bytes[off2 + 2], 0x00);
    }

    // -- Offset stability ---------------------------------------------------

    #[test]
    fn test_offsets_stable_across_additions() {
        let mut table = DebugStrTable::new();
        let off_a = table.add_string("alpha");
        let off_b = table.add_string("beta");

        // Add more strings and verify earlier offsets are unchanged.
        table.add_string("gamma");
        table.add_string("delta");

        assert_eq!(table.get_offset("alpha"), Some(off_a));
        assert_eq!(table.get_offset("beta"), Some(off_b));
    }

    // -- Realistic DWARF usage patterns -------------------------------------

    #[test]
    fn test_typical_dwarf_usage() {
        let mut table = DebugStrTable::new();

        // Typical strings added during DWARF emission:
        let producer = table.add_string("bcc 1.0.0");
        let comp_dir = table.add_string("/home/user/project");
        let source = table.add_string("main.c");
        let func = table.add_string("main");
        let var_name = table.add_string("argc");
        let type_name = table.add_string("int");

        // All offsets should be distinct (no two strings are the same).
        let offsets = [producer, comp_dir, source, func, var_name, type_name];
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                assert_ne!(
                    offsets[i], offsets[j],
                    "Offsets for different strings must be distinct"
                );
            }
        }

        // Verify round-trip: each string can be found in the raw bytes.
        let bytes = table.as_bytes();
        for (s, off) in [
            ("bcc 1.0.0", producer),
            ("/home/user/project", comp_dir),
            ("main.c", source),
            ("main", func),
            ("argc", var_name),
            ("int", type_name),
        ] {
            let off = off as usize;
            let end = off + s.len();
            assert_eq!(
                &bytes[off..end],
                s.as_bytes(),
                "String '{}' not found at offset {}",
                s,
                off,
            );
            assert_eq!(
                bytes[end], 0x00,
                "String '{}' missing null terminator at offset {}",
                s, end,
            );
        }
    }

    #[test]
    fn test_unicode_string() {
        let mut table = DebugStrTable::new();
        // Multi-byte UTF-8: U+00E9 (é) is 2 bytes, U+2603 (☃) is 3 bytes.
        let off = table.add_string("café☃");
        let bytes = table.as_bytes();
        let s = "café☃";
        let start = off as usize;
        let end = start + s.len();
        assert_eq!(&bytes[start..end], s.as_bytes());
        assert_eq!(bytes[end], 0x00);
    }

    // -- Debug trait ---------------------------------------------------------

    #[test]
    fn test_debug_trait_does_not_panic() {
        let mut table = DebugStrTable::new();
        table.add_string("debug");
        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("DebugStrTable"));
        assert!(debug_str.contains("section_size"));
        assert!(debug_str.contains("string_count"));
    }

    // -- Large string count -------------------------------------------------

    #[test]
    fn test_many_strings() {
        let mut table = DebugStrTable::new();
        let count = 1000;
        let mut recorded_offsets = Vec::with_capacity(count);

        for i in 0..count {
            let s = format!("string_{}", i);
            let off = table.add_string(&s);
            recorded_offsets.push((s, off));
        }

        // Verify all offsets are recoverable and still correct.
        for (s, expected_off) in &recorded_offsets {
            assert_eq!(
                table.get_offset(s),
                Some(*expected_off),
                "Offset mismatch for '{}'",
                s,
            );
        }

        // Verify deduplication: adding the same strings again yields same offsets.
        for (s, expected_off) in &recorded_offsets {
            let off = table.add_string(s);
            assert_eq!(off, *expected_off, "Deduplication failed for '{}'", s,);
        }

        // Size should not have grown after the deduplication pass.
        let size_before = table.section_size();
        for (s, _) in &recorded_offsets {
            table.add_string(s);
        }
        assert_eq!(table.section_size(), size_before);
    }
}
