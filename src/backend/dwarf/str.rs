//! # `.debug_str` Section Generation
//!
//! Generates the DWARF v4 string table section for BCC. The `.debug_str` section
//! is a pool of null-terminated UTF-8 strings referenced by Debug Information
//! Entries (DIEs) in `.debug_info` using the `DW_FORM_strp` form.
//!
//! Instead of embedding string data inline in every DIE, the `DW_FORM_strp` form
//! stores a 4-byte offset (in 32-bit DWARF format) into this string table,
//! achieving significant deduplication and size savings.
//!
//! The `.debug_str` section is the most foundational of the four DWARF sections
//! because it is consumed by `.debug_info` (and potentially `.debug_line` for
//! file names via `DW_AT_name`), but depends on no other debug sections.
//!
//! ## String Deduplication
//! Identical strings share the same offset — adding the same string twice
//! returns the original offset. This reduces the size of the debug
//! information significantly for common type names and repeated identifiers.
//!
//! ## Binary Format
//! The section is a simple concatenation of null-terminated strings:
//! ```text
//! offset 0: "int\0"
//! offset 4: "main\0"
//! offset 9: "char\0"
//! offset 14: "BCC 0.1.0\0"
//! ...
//! ```
//!
//! ## Usage
//! ```ignore
//! let mut str_table = DebugStrTable::new();
//! let int_offset = str_table.add_string("int");       // e.g., returns 0
//! let main_offset = str_table.add_string("main");     // e.g., returns 4
//! let int_again = str_table.add_string("int");        // returns 0 (deduplicated)
//! let section_data = str_table.generate();
//! ```
//!
//! ## References
//! - DWARF v4, Section 7.5.5 (`DW_FORM_strp`)
//! - DWARF v4, Section 7.26 (String Table)

use crate::common::fx_hash::FxHashMap;

// ============================================================================
// Producer Version Constant
// ============================================================================

/// The producer identification string embedded in the compilation unit DIE
/// via `DW_AT_producer`. Identifies this compiler and its version to debuggers.
const BCC_PRODUCER_STRING: &str = "BCC 0.1.0";

// ============================================================================
// DebugStrTable — `.debug_str` Section Builder
// ============================================================================

/// Builder for the `.debug_str` DWARF v4 section.
///
/// Collects null-terminated UTF-8 strings and provides byte-offset-based
/// references suitable for `DW_FORM_strp` attributes in `.debug_info`.
/// Identical strings are deduplicated — inserting the same string twice
/// returns the same offset, saving space in the debug output.
///
/// # Architecture
///
/// Internally, the string table maintains two data structures:
/// - A `Vec<u8>` buffer that accumulates the raw section bytes
///   (concatenated null-terminated strings).
/// - An `FxHashMap<String, u32>` that maps previously inserted string
///   content to its byte offset within the buffer, enabling O(1)
///   average-case deduplication lookups.
///
/// # DWARF v4 Format
///
/// In 32-bit DWARF format (the only format BCC emits), `DW_FORM_strp`
/// references are 4-byte little-endian offsets pointing into this section.
/// The section itself has no header or trailer — it is purely a
/// concatenation of null-terminated strings starting at offset 0.
///
/// # Examples
///
/// ```ignore
/// use bcc::backend::dwarf::str::DebugStrTable;
///
/// let mut table = DebugStrTable::new();
/// let off_int = table.add_string("int");     // offset 0
/// let off_main = table.add_string("main");   // offset 4
/// let off_dup = table.add_string("int");     // offset 0 (deduplicated)
///
/// assert_eq!(off_int, 0);
/// assert_eq!(off_main, 4);
/// assert_eq!(off_dup, off_int);
///
/// let data = table.generate();
/// assert_eq!(&data, b"int\0main\0");
/// ```
pub struct DebugStrTable {
    /// The accumulated section data — concatenated null-terminated strings.
    /// Each string's bytes are followed immediately by a single `0x00` byte.
    /// The first string starts at offset 0.
    buffer: Vec<u8>,

    /// Deduplication map: maps string content to its byte offset within
    /// `buffer`. Uses FxHashMap for fast, non-cryptographic hashing — ideal
    /// for compiler-internal data structures where input is trusted.
    offsets: FxHashMap<String, u32>,
}

// ============================================================================
// Core Implementation
// ============================================================================

impl DebugStrTable {
    /// Create a new, empty `.debug_str` section builder.
    ///
    /// The resulting table contains no strings and produces an empty section.
    /// Strings are added via [`add_string`](Self::add_string) or the
    /// convenience methods ([`add_name`](Self::add_name),
    /// [`add_type_name`](Self::add_type_name), etc.).
    #[inline]
    pub fn new() -> Self {
        DebugStrTable {
            buffer: Vec::new(),
            offsets: FxHashMap::default(),
        }
    }

    /// Add a string to the table and return its byte offset within the
    /// `.debug_str` section.
    ///
    /// If the string already exists in the table, the existing offset is
    /// returned without modifying the buffer (deduplication). Otherwise,
    /// the string bytes are appended to the buffer followed by a null
    /// terminator (`0x00`), and the new offset is returned.
    ///
    /// The returned offset is suitable for encoding as a `DW_FORM_strp`
    /// value in `.debug_info` using [`encode_strp_offset`](Self::encode_strp_offset).
    ///
    /// # Parameters
    /// - `s`: The string to add. May be empty (adds a single null byte).
    ///   UTF-8 bytes are preserved exactly — no re-encoding occurs.
    ///
    /// # Returns
    /// The byte offset from the start of the `.debug_str` section where
    /// this string begins.
    ///
    /// # Panics
    /// This method does not panic under normal usage. In theory, if the
    /// accumulated buffer exceeds `u32::MAX` bytes (4 GiB), the `as u32`
    /// cast would truncate — but debug string tables of that size are
    /// unrealistic for any practical compilation.
    pub fn add_string(&mut self, s: &str) -> u32 {
        // Fast path: check for an existing entry in the deduplication map.
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }

        // Record the current buffer length as the offset for this new string.
        let offset = self.buffer.len() as u32;

        // Append the string bytes followed by a null terminator.
        // UTF-8 bytes are copied verbatim — no transcoding.
        self.buffer.extend_from_slice(s.as_bytes());
        self.buffer.push(0); // null terminator

        // Store the offset in the deduplication map for future lookups.
        self.offsets.insert(s.to_string(), offset);

        offset
    }

    /// Look up the offset of a previously added string without inserting it.
    ///
    /// Returns `Some(offset)` if the string was previously added via
    /// [`add_string`](Self::add_string) (or any convenience method),
    /// or `None` if the string has never been inserted.
    ///
    /// # Parameters
    /// - `s`: The string to look up.
    ///
    /// # Returns
    /// The byte offset if the string exists, or `None` otherwise.
    #[inline]
    pub fn get_offset(&self, s: &str) -> Option<u32> {
        self.offsets.get(s).copied()
    }

    /// Return the finalized `.debug_str` section data as an owned byte vector.
    ///
    /// The returned bytes are the complete, ready-to-emit section contents:
    /// concatenated null-terminated strings starting at offset 0. There is
    /// no section header or trailer — the raw bytes are written directly
    /// into the ELF `.debug_str` section.
    ///
    /// This method clones the internal buffer. For read-only access without
    /// allocation, use [`data`](Self::data) instead.
    pub fn generate(&self) -> Vec<u8> {
        self.buffer.clone()
    }

    /// Return a reference to the current buffer contents without copying.
    ///
    /// The returned slice contains all strings added so far, concatenated
    /// with null terminators. This is useful for writing the section data
    /// directly into an ELF writer without an intermediate allocation.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.buffer
    }

    /// Return the current size of the string table in bytes.
    ///
    /// This is the total number of bytes that will be emitted in the
    /// `.debug_str` section, including all null terminators. An empty
    /// table returns 0.
    ///
    /// Useful for computing section header `sh_size` values and for
    /// pre-allocating output buffers.
    #[inline]
    pub fn size(&self) -> u32 {
        self.buffer.len() as u32
    }

    /// Check whether the string table is empty (no strings have been added).
    ///
    /// An empty string table means no `.debug_str` section needs to be
    /// emitted in the output ELF. This can be used to conditionally skip
    /// section creation when debug information is not requested.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ============================================================================
// Predefined String Convenience Methods
// ============================================================================

impl DebugStrTable {
    /// Add the BCC producer identification string (currently "BCC 0.1.0").
    ///
    /// This string is used for the `DW_AT_producer` attribute in the
    /// `DW_TAG_compile_unit` DIE, identifying the compiler that produced
    /// the debug information. Debuggers (GDB, LLDB) display this in
    /// `info source` or equivalent commands.
    ///
    /// # Returns
    /// The byte offset of the producer string in the `.debug_str` section.
    /// Subsequent calls return the same offset (deduplication).
    pub fn add_producer_string(&mut self) -> u32 {
        self.add_string(BCC_PRODUCER_STRING)
    }

    /// Add a source file name string.
    ///
    /// Used for the `DW_AT_name` attribute in the `DW_TAG_compile_unit`
    /// DIE, identifying the primary source file of the compilation unit.
    /// Debuggers use this to locate and display source code.
    ///
    /// # Parameters
    /// - `file_name`: The file name (e.g., `"main.c"`, `"kernel/sched/core.c"`).
    ///
    /// # Returns
    /// The byte offset of the file name string.
    pub fn add_file_name(&mut self, file_name: &str) -> u32 {
        self.add_string(file_name)
    }

    /// Add a compilation directory string.
    ///
    /// Used for the `DW_AT_comp_dir` attribute in the `DW_TAG_compile_unit`
    /// DIE. This is the working directory from which the compiler was invoked,
    /// allowing debuggers to resolve relative file paths.
    ///
    /// # Parameters
    /// - `dir`: The directory path (e.g., `"/home/user/project"`).
    ///
    /// # Returns
    /// The byte offset of the directory string.
    pub fn add_comp_dir(&mut self, dir: &str) -> u32 {
        self.add_string(dir)
    }

    /// Add a function or variable name string.
    ///
    /// Used for the `DW_AT_name` attribute in `DW_TAG_subprogram` (function)
    /// and `DW_TAG_variable` DIEs. Debuggers display these names in
    /// backtraces, variable watches, and symbol lookups.
    ///
    /// # Parameters
    /// - `name`: The identifier name (e.g., `"main"`, `"counter"`, `"do_fork"`).
    ///
    /// # Returns
    /// The byte offset of the name string.
    pub fn add_name(&mut self, name: &str) -> u32 {
        self.add_string(name)
    }

    /// Add a type name string.
    ///
    /// Used for the `DW_AT_name` attribute in `DW_TAG_base_type`,
    /// `DW_TAG_structure_type`, `DW_TAG_typedef`, and other type-describing
    /// DIEs. Debuggers display these when printing variable types.
    ///
    /// # Parameters
    /// - `type_name`: The type name (e.g., `"int"`, `"char"`, `"struct task_struct"`).
    ///
    /// # Returns
    /// The byte offset of the type name string.
    pub fn add_type_name(&mut self, type_name: &str) -> u32 {
        self.add_string(type_name)
    }
}

// ============================================================================
// DW_FORM_strp Offset Encoding
// ============================================================================

impl DebugStrTable {
    /// Encode a `DW_FORM_strp` reference as a 4-byte little-endian offset
    /// and append it to the given output buffer.
    ///
    /// This is a convenience method for `.debug_info` emission — it writes
    /// the 4-byte offset that a DIE attribute uses to point into the
    /// `.debug_str` section.
    ///
    /// In 32-bit DWARF format (the only format BCC emits), `strp` offsets
    /// are always exactly 4 bytes, encoded in the target byte order
    /// (little-endian for all BCC-supported architectures).
    ///
    /// # Parameters
    /// - `offset`: The byte offset into `.debug_str` (as returned by
    ///   [`add_string`](Self::add_string) or any convenience method).
    /// - `output`: The buffer to append the 4-byte encoded offset to.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut info_buf: Vec<u8> = Vec::new();
    /// let str_offset = str_table.add_string("main");
    /// DebugStrTable::encode_strp_offset(str_offset, &mut info_buf);
    /// // info_buf now contains the 4-byte LE offset pointing to "main" in .debug_str
    /// ```
    #[inline]
    pub fn encode_strp_offset(offset: u32, output: &mut Vec<u8>) {
        output.extend_from_slice(&offset.to_le_bytes());
    }
}

// ============================================================================
// Default Trait Implementation
// ============================================================================

impl Default for DebugStrTable {
    /// Create a new, empty `DebugStrTable`.
    ///
    /// Equivalent to [`DebugStrTable::new()`].
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_table_is_empty() {
        let table = DebugStrTable::new();
        assert!(table.is_empty());
        assert_eq!(table.size(), 0);
        assert_eq!(table.data(), &[] as &[u8]);
        assert_eq!(table.generate(), Vec::<u8>::new());
    }

    #[test]
    fn test_default_is_same_as_new() {
        let table = DebugStrTable::default();
        assert!(table.is_empty());
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_add_single_string() {
        let mut table = DebugStrTable::new();
        let offset = table.add_string("int");

        // First string starts at offset 0.
        assert_eq!(offset, 0);
        // "int" is 3 bytes + 1 null terminator = 4 bytes.
        assert_eq!(table.size(), 4);
        assert!(!table.is_empty());
        assert_eq!(table.data(), b"int\0");
    }

    #[test]
    fn test_add_multiple_strings() {
        let mut table = DebugStrTable::new();

        let off_int = table.add_string("int");
        let off_main = table.add_string("main");
        let off_char = table.add_string("char");

        assert_eq!(off_int, 0);
        // "int\0" = 4 bytes, so "main" starts at offset 4.
        assert_eq!(off_main, 4);
        // "int\0main\0" = 9 bytes, so "char" starts at offset 9.
        assert_eq!(off_char, 9);
        // Total: "int\0main\0char\0" = 14 bytes.
        assert_eq!(table.size(), 14);
        assert_eq!(table.data(), b"int\0main\0char\0");
    }

    #[test]
    fn test_deduplication() {
        let mut table = DebugStrTable::new();

        let off1 = table.add_string("int");
        let off2 = table.add_string("main");
        let off3 = table.add_string("int"); // duplicate

        // Duplicate returns the same offset without growing the buffer.
        assert_eq!(off1, off3);
        assert_eq!(off1, 0);
        assert_eq!(off2, 4);
        // Buffer should only contain "int\0main\0" — no second "int\0".
        assert_eq!(table.size(), 9);
        assert_eq!(table.data(), b"int\0main\0");
    }

    #[test]
    fn test_get_offset_existing() {
        let mut table = DebugStrTable::new();
        table.add_string("hello");

        assert_eq!(table.get_offset("hello"), Some(0));
    }

    #[test]
    fn test_get_offset_nonexistent() {
        let table = DebugStrTable::new();
        assert_eq!(table.get_offset("missing"), None);
    }

    #[test]
    fn test_get_offset_after_multiple_inserts() {
        let mut table = DebugStrTable::new();
        table.add_string("aaa");
        table.add_string("bbb");
        table.add_string("ccc");

        assert_eq!(table.get_offset("aaa"), Some(0));
        assert_eq!(table.get_offset("bbb"), Some(4)); // "aaa\0" = 4
        assert_eq!(table.get_offset("ccc"), Some(8)); // "aaa\0bbb\0" = 8
        assert_eq!(table.get_offset("ddd"), None);
    }

    #[test]
    fn test_empty_string() {
        let mut table = DebugStrTable::new();
        let offset = table.add_string("");

        // Empty string adds a single null byte.
        assert_eq!(offset, 0);
        assert_eq!(table.size(), 1);
        assert_eq!(table.data(), b"\0");
    }

    #[test]
    fn test_empty_string_deduplication() {
        let mut table = DebugStrTable::new();
        let off1 = table.add_string("");
        let off2 = table.add_string("");

        assert_eq!(off1, off2);
        assert_eq!(table.size(), 1); // Only one null byte.
    }

    #[test]
    fn test_generate_returns_clone() {
        let mut table = DebugStrTable::new();
        table.add_string("foo");
        table.add_string("bar");

        let generated = table.generate();
        assert_eq!(generated, table.data());
        assert_eq!(generated, b"foo\0bar\0");
    }

    #[test]
    fn test_null_terminated_strings() {
        let mut table = DebugStrTable::new();
        table.add_string("hello");

        let data = table.data();
        // Verify the null terminator is present.
        assert_eq!(data[5], 0);
        // Verify the string bytes are correct.
        assert_eq!(&data[0..5], b"hello");
    }

    #[test]
    fn test_utf8_preservation() {
        let mut table = DebugStrTable::new();
        // UTF-8 multi-byte string.
        let offset = table.add_string("héllo");

        assert_eq!(offset, 0);
        // "héllo" in UTF-8: 'h'(1) + 'é'(2) + 'l'(1) + 'l'(1) + 'o'(1) = 6 bytes + null = 7.
        assert_eq!(table.size(), 7);
        // Verify raw bytes are preserved exactly.
        let expected = b"h\xc3\xa9llo\0";
        assert_eq!(table.data(), expected);
    }

    #[test]
    fn test_producer_string() {
        let mut table = DebugStrTable::new();
        let offset = table.add_producer_string();

        assert_eq!(offset, 0);
        assert_eq!(table.get_offset("BCC 0.1.0"), Some(0));
    }

    #[test]
    fn test_producer_string_deduplication() {
        let mut table = DebugStrTable::new();
        let off1 = table.add_producer_string();
        let off2 = table.add_producer_string();

        assert_eq!(off1, off2);
    }

    #[test]
    fn test_add_file_name() {
        let mut table = DebugStrTable::new();
        let offset = table.add_file_name("main.c");

        assert_eq!(offset, 0);
        assert_eq!(table.get_offset("main.c"), Some(0));
    }

    #[test]
    fn test_add_comp_dir() {
        let mut table = DebugStrTable::new();
        let offset = table.add_comp_dir("/home/user/project");

        assert_eq!(offset, 0);
        assert_eq!(table.get_offset("/home/user/project"), Some(0));
    }

    #[test]
    fn test_add_name() {
        let mut table = DebugStrTable::new();
        let offset = table.add_name("do_fork");

        assert_eq!(offset, 0);
        assert_eq!(table.get_offset("do_fork"), Some(0));
    }

    #[test]
    fn test_add_type_name() {
        let mut table = DebugStrTable::new();
        let offset = table.add_type_name("struct task_struct");

        assert_eq!(offset, 0);
        assert_eq!(table.get_offset("struct task_struct"), Some(0));
    }

    #[test]
    fn test_convenience_methods_share_deduplication() {
        let mut table = DebugStrTable::new();

        // Add "main" via add_name, then try to add it via add_string.
        let off_name = table.add_name("main");
        let off_str = table.add_string("main");

        // They should return the same offset due to shared deduplication.
        assert_eq!(off_name, off_str);
        assert_eq!(table.size(), 5); // "main\0" = 5 bytes
    }

    #[test]
    fn test_encode_strp_offset_zero() {
        let mut buf = Vec::new();
        DebugStrTable::encode_strp_offset(0, &mut buf);

        assert_eq!(buf.len(), 4);
        assert_eq!(buf, vec![0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_encode_strp_offset_small() {
        let mut buf = Vec::new();
        DebugStrTable::encode_strp_offset(4, &mut buf);

        assert_eq!(buf.len(), 4);
        // 4 in little-endian 32-bit: 0x04, 0x00, 0x00, 0x00
        assert_eq!(buf, vec![0x04, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_encode_strp_offset_large() {
        let mut buf = Vec::new();
        DebugStrTable::encode_strp_offset(0x12345678, &mut buf);

        assert_eq!(buf.len(), 4);
        // 0x12345678 in little-endian: 0x78, 0x56, 0x34, 0x12
        assert_eq!(buf, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_encode_strp_offset_max() {
        let mut buf = Vec::new();
        DebugStrTable::encode_strp_offset(u32::MAX, &mut buf);

        assert_eq!(buf.len(), 4);
        assert_eq!(buf, vec![0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_encode_strp_offset_appends() {
        let mut buf = vec![0xAA, 0xBB];
        DebugStrTable::encode_strp_offset(9, &mut buf);

        // The encoded offset should be appended, not replace existing data.
        assert_eq!(buf.len(), 6);
        assert_eq!(buf, vec![0xAA, 0xBB, 0x09, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_realistic_compilation_unit() {
        // Simulate adding strings for a typical compilation unit.
        let mut table = DebugStrTable::new();

        let off_producer = table.add_producer_string(); // "BCC 0.1.0"
        let off_file = table.add_file_name("hello.c"); // "hello.c"
        let off_dir = table.add_comp_dir("/tmp/build"); // "/tmp/build"
        let off_main = table.add_name("main"); // "main"
        let off_int = table.add_type_name("int"); // "int"
        let off_char = table.add_type_name("char"); // "char"
        let off_argc = table.add_name("argc"); // "argc"
        let off_argv = table.add_name("argv"); // "argv"

        // Deduplication: adding "int" again returns the same offset.
        let off_int_dup = table.add_type_name("int");
        assert_eq!(off_int, off_int_dup);

        // All offsets should be distinct (except the deduplicated one).
        let offsets = [
            off_producer,
            off_file,
            off_dir,
            off_main,
            off_int,
            off_char,
            off_argc,
            off_argv,
        ];
        for i in 0..offsets.len() {
            for j in (i + 1)..offsets.len() {
                assert_ne!(
                    offsets[i], offsets[j],
                    "offsets[{}] == offsets[{}] == {}",
                    i, j, offsets[i]
                );
            }
        }

        // Verify the buffer is well-formed: every string is null-terminated.
        let data = table.data();
        let mut pos = 0;
        let strings = [
            "BCC 0.1.0",
            "hello.c",
            "/tmp/build",
            "main",
            "int",
            "char",
            "argc",
            "argv",
        ];
        for s in &strings {
            let end = pos + s.len();
            assert_eq!(&data[pos..end], s.as_bytes());
            assert_eq!(data[end], 0); // null terminator
            pos = end + 1;
        }
        assert_eq!(pos, data.len());
    }

    #[test]
    fn test_encode_then_lookup() {
        // End-to-end: add strings, encode offsets, verify they decode correctly.
        let mut table = DebugStrTable::new();
        let off_int = table.add_string("int");
        let off_main = table.add_string("main");

        let mut info_buf = Vec::new();
        DebugStrTable::encode_strp_offset(off_int, &mut info_buf);
        DebugStrTable::encode_strp_offset(off_main, &mut info_buf);

        // Decode the offsets from the info buffer.
        let decoded_off_int =
            u32::from_le_bytes([info_buf[0], info_buf[1], info_buf[2], info_buf[3]]);
        let decoded_off_main =
            u32::from_le_bytes([info_buf[4], info_buf[5], info_buf[6], info_buf[7]]);

        assert_eq!(decoded_off_int, off_int);
        assert_eq!(decoded_off_main, off_main);

        // Verify the strings can be read back from the section data.
        let str_data = table.generate();
        let int_start = decoded_off_int as usize;
        let int_end = str_data[int_start..].iter().position(|&b| b == 0).unwrap() + int_start;
        assert_eq!(&str_data[int_start..int_end], b"int");

        let main_start = decoded_off_main as usize;
        let main_end = str_data[main_start..].iter().position(|&b| b == 0).unwrap() + main_start;
        assert_eq!(&str_data[main_start..main_end], b"main");
    }

    #[test]
    fn test_string_with_special_characters() {
        let mut table = DebugStrTable::new();

        // Strings containing path separators, spaces, and other characters.
        let off1 = table.add_string("/usr/src/linux-6.9/kernel/sched/core.c");
        let off2 = table.add_string("struct task_struct *");
        let off3 = table.add_string("unsigned long long");

        assert_eq!(off1, 0);
        assert!(off2 > off1);
        assert!(off3 > off2);

        // Verify null termination for each string.
        let data = table.data();
        let check_null = |start: u32, s: &str| {
            let start = start as usize;
            let end = start + s.len();
            assert_eq!(&data[start..end], s.as_bytes());
            assert_eq!(data[end], 0);
        };
        check_null(off1, "/usr/src/linux-6.9/kernel/sched/core.c");
        check_null(off2, "struct task_struct *");
        check_null(off3, "unsigned long long");
    }

    #[test]
    fn test_single_character_strings() {
        let mut table = DebugStrTable::new();

        let off_a = table.add_string("a");
        let off_b = table.add_string("b");

        assert_eq!(off_a, 0);
        assert_eq!(off_b, 2); // "a\0" = 2 bytes
        assert_eq!(table.data(), b"a\0b\0");
    }

    #[test]
    fn test_long_string() {
        let mut table = DebugStrTable::new();
        let long_str = "a".repeat(1000);
        let offset = table.add_string(&long_str);

        assert_eq!(offset, 0);
        assert_eq!(table.size(), 1001); // 1000 chars + null
        assert_eq!(table.data().last(), Some(&0)); // ends with null
        assert_eq!(&table.data()[..1000], long_str.as_bytes());
    }
}
