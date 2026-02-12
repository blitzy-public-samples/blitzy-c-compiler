// src/common/source_map.rs
//
// Source file tracking module for the BCC compiler.
//
// Maintains a registry of all source files loaded during compilation with
// pre-computed line offset tables enabling O(log n) binary search for
// line/column lookups from byte offsets. Supports `#line` directive remapping
// for preprocessor-generated source locations.
//
// Used by:
// - The diagnostic engine (`diagnostics.rs`) for error/warning source location formatting
// - DWARF debug information generation for source file/line mapping
//
// Design decisions:
// - Line offsets are computed once on file load and stored as Vec<u32>,
//   enabling efficient binary search without rescanning content.
// - FileId is a lightweight newtype wrapper (4 bytes, Copy) that indexes
//   directly into the SourceMap's file vector.
// - Line and column numbers are 1-indexed in all user-facing contexts
//   (SourceLocation), matching GCC/Clang convention.
// - `#line` directives are stored separately and applied lazily during
//   location lookup, avoiding mutation of the underlying line offset tables.

/// Unique identifier for a loaded source file within the compilation session.
///
/// `FileId` is a lightweight, copyable handle that indexes into the `SourceMap`'s
/// internal file registry. It is used throughout the compiler pipeline to reference
/// source files without carrying ownership of file data.
///
/// The inner `u32` value corresponds to the index in `SourceMap::files`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FileId(pub u32);

/// Represents a single source file loaded into the compilation session.
///
/// Each `SourceFile` stores the file's content as a Rust `String` (with
/// non-UTF-8 bytes PUA-encoded by the encoding module) and a pre-computed
/// table of line start byte offsets for efficient line/column resolution.
///
/// # Line Offset Table
///
/// `line_offsets[i]` contains the byte offset of the start of line `i`
/// (0-indexed internally). The table always starts with offset 0 (the
/// beginning of the file is always line 0). Each subsequent entry records
/// the byte position immediately after a newline character.
///
/// Example for content `"abc\ndef\n"`:
/// ```text
/// line_offsets = [0, 4]  // line 0 starts at byte 0, line 1 starts at byte 4
/// ```
pub struct SourceFile {
    /// Unique identifier for this file within the source map.
    pub id: FileId,
    /// File name or path as provided to the compiler (may be relative or absolute).
    pub name: String,
    /// Full content of the file as a Rust String (with PUA encoding for non-UTF-8 bytes).
    pub content: String,
    /// Pre-computed byte offsets of each line start.
    /// Index 0 is always 0 (start of file). Index `i` gives the byte offset
    /// of line `i` (0-indexed). Used for O(log n) binary search lookups.
    pub line_offsets: Vec<u32>,
}

/// A resolved source location with file, line, and column information.
///
/// All values are 1-indexed for user-facing display, matching the convention
/// used by GCC, Clang, and most editors. This struct is produced by
/// `SourceMap::lookup_location()` and consumed by the diagnostic engine
/// for formatting error/warning messages.
#[derive(Clone, Debug)]
pub struct SourceLocation {
    /// The file containing this location.
    pub file_id: FileId,
    /// 1-indexed line number within the file.
    pub line: u32,
    /// 1-indexed column number within the line (byte offset from line start + 1).
    pub column: u32,
    /// Human-readable file name for display. This may differ from the original
    /// file name if a `#line` directive has remapped the location.
    pub file_name: String,
}

/// A `#line` directive remapping entry.
///
/// When the preprocessor encounters `#line N` or `#line N "filename"`,
/// a `LineDirective` is registered in the `SourceMap`. During location
/// lookup, any byte offset at or after `byte_offset` in the specified
/// file will have its line number (and optionally file name) remapped.
///
/// Directives are stored in insertion order and searched in reverse
/// during lookup to find the most recently applicable directive.
#[derive(Clone, Debug)]
pub struct LineDirective {
    /// The file in which this directive appears.
    pub file_id: FileId,
    /// Byte offset in the file content where this directive takes effect.
    /// All byte offsets >= this value (in the same file) are remapped.
    pub byte_offset: u32,
    /// The new line number that `byte_offset` maps to (1-indexed).
    pub new_line: u32,
    /// Optional new file name. If `Some`, the file name in the resolved
    /// `SourceLocation` is replaced with this value.
    pub new_file: Option<String>,
}

/// Central registry of all source files and `#line` directives in a compilation session.
///
/// The `SourceMap` is created once at the start of compilation and shared
/// (by reference) across all pipeline stages. Source files are added as they
/// are loaded by the preprocessor's `#include` handler or the CLI driver.
///
/// # Thread Safety
///
/// `SourceMap` is not thread-safe by default. It is designed for use within
/// a single compilation thread (the 64 MiB worker thread). If concurrent
/// access is needed in the future, it can be wrapped in a `Mutex`.
pub struct SourceMap {
    /// All loaded source files, indexed by `FileId.0`.
    files: Vec<SourceFile>,
    /// Registered `#line` directives for source location remapping.
    /// Stored in insertion order; searched in reverse during lookup.
    line_directives: Vec<LineDirective>,
}

impl SourceMap {
    /// Creates a new, empty source map with no files or directives.
    pub fn new() -> Self {
        SourceMap {
            files: Vec::new(),
            line_directives: Vec::new(),
        }
    }

    /// Registers a source file in the map and returns its unique `FileId`.
    ///
    /// The file's content is stored and a line offset table is pre-computed
    /// for efficient subsequent lookups. The returned `FileId` can be used
    /// with all other `SourceMap` methods.
    ///
    /// # Arguments
    ///
    /// * `name` — The file name or path (used in diagnostic messages and DWARF info).
    /// * `content` — The full file content as a Rust String (PUA-encoded if needed).
    ///
    /// # Returns
    ///
    /// A `FileId` uniquely identifying this file within this `SourceMap`.
    pub fn add_file(&mut self, name: String, content: String) -> FileId {
        let id = FileId(self.files.len() as u32);
        let line_offsets = compute_line_offsets(&content);
        self.files.push(SourceFile {
            id,
            name,
            content,
            line_offsets,
        });
        id
    }

    /// Retrieves a reference to the `SourceFile` for the given `FileId`.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not correspond to a file in this source map.
    /// This is a programming error — all `FileId` values should originate
    /// from `add_file()` on the same `SourceMap` instance.
    pub fn get_file(&self, id: FileId) -> &SourceFile {
        let idx = id.0 as usize;
        assert!(
            idx < self.files.len(),
            "source_map: invalid FileId({}) — only {} files registered",
            id.0,
            self.files.len()
        );
        &self.files[idx]
    }

    /// Resolves a byte offset within a file to a `SourceLocation` with
    /// 1-indexed line and column numbers.
    ///
    /// Uses binary search on the pre-computed line offset table for O(log n)
    /// performance. If any `#line` directives are applicable, the returned
    /// location reflects the remapped line number and (optionally) file name.
    ///
    /// # Arguments
    ///
    /// * `file_id` — The file containing the byte offset.
    /// * `byte_offset` — Byte position within the file content (0-indexed).
    ///
    /// # Returns
    ///
    /// A `SourceLocation` with 1-indexed line and column numbers and the
    /// (possibly remapped) file name.
    ///
    /// # Panics
    ///
    /// Panics if `file_id` is invalid.
    pub fn lookup_location(&self, file_id: FileId, byte_offset: u32) -> SourceLocation {
        let file = self.get_file(file_id);
        let offsets = &file.line_offsets;

        // Binary search: find the rightmost line whose start offset is <= byte_offset.
        // `partition_point` returns the first index where the predicate is false,
        // so we subtract 1 to get the line index.
        let line_index = match offsets.partition_point(|&off| off <= byte_offset) {
            0 => 0, // Should not happen since offsets[0] is always 0, but handle gracefully
            n => n - 1,
        };

        let line_start = offsets[line_index];
        // Column is 1-indexed: byte_offset at the start of a line is column 1
        let column = byte_offset - line_start + 1;
        // Line is 1-indexed for display
        let raw_line = line_index as u32 + 1;

        // Apply #line directive remapping if applicable
        let (display_line, display_file) =
            self.apply_line_directive(file_id, byte_offset, raw_line, &file.name);

        SourceLocation {
            file_id,
            line: display_line,
            column,
            file_name: display_file,
        }
    }

    /// Extracts the content of a single source line for diagnostic display.
    ///
    /// Returns the text of the specified line (1-indexed) without the trailing
    /// newline character. If the line number is out of range, returns an
    /// empty string.
    ///
    /// # Arguments
    ///
    /// * `file_id` — The file to extract from.
    /// * `line` — 1-indexed line number.
    ///
    /// # Returns
    ///
    /// The line content as a string slice, without trailing newline.
    pub fn get_line_content(&self, file_id: FileId, line: u32) -> &str {
        let file = self.get_file(file_id);
        let offsets = &file.line_offsets;

        // Convert 1-indexed line to 0-indexed
        if line == 0 {
            return "";
        }
        let line_idx = (line - 1) as usize;

        if line_idx >= offsets.len() {
            return "";
        }

        let start = offsets[line_idx] as usize;

        // End is either the start of the next line or the end of file
        let end = if line_idx + 1 < offsets.len() {
            offsets[line_idx + 1] as usize
        } else {
            file.content.len()
        };

        // Trim trailing newline characters (\n or \r\n)
        let line_bytes = &file.content.as_bytes()[start..end];
        let mut trim_end = end;
        if trim_end > start && line_bytes[trim_end - start - 1] == b'\n' {
            trim_end -= 1;
        }
        if trim_end > start && line_bytes[trim_end - start - 1] == b'\r' {
            trim_end -= 1;
        }

        &file.content[start..trim_end]
    }

    /// Extracts a substring of source content for a given byte range.
    ///
    /// Returns the content between `start` (inclusive) and `end` (exclusive)
    /// byte offsets. If the range exceeds the file content, it is clamped
    /// to the available range.
    ///
    /// # Arguments
    ///
    /// * `file_id` — The file to extract from.
    /// * `start` — Start byte offset (inclusive).
    /// * `end` — End byte offset (exclusive).
    ///
    /// # Returns
    ///
    /// The extracted source text as a string slice.
    pub fn get_snippet(&self, file_id: FileId, start: u32, end: u32) -> &str {
        let file = self.get_file(file_id);
        let content_len = file.content.len();

        let s = (start as usize).min(content_len);
        let e = (end as usize).min(content_len);

        if s >= e {
            return "";
        }

        &file.content[s..e]
    }

    /// Registers a `#line` directive remapping.
    ///
    /// After this call, any `lookup_location()` for byte offsets at or after
    /// `directive.byte_offset` in `directive.file_id` will use the remapped
    /// line number (and optionally file name) instead of the physical line.
    ///
    /// Directives should be added in the order they appear in the source
    /// (i.e., with monotonically non-decreasing byte offsets per file).
    /// The most recently applicable directive (highest byte_offset <=
    /// query offset) takes precedence.
    pub fn add_line_directive(&mut self, directive: LineDirective) {
        self.line_directives.push(directive);
    }

    /// Returns the total number of source files registered in this map.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Applies `#line` directive remapping to a raw location.
    ///
    /// Searches the directive list in reverse to find the most recently
    /// applicable directive for the given file and byte offset. If found,
    /// the raw line number is adjusted by the delta between the directive's
    /// position and the query position, and the file name may be replaced.
    ///
    /// # Arguments
    ///
    /// * `file_id` — The file being queried.
    /// * `byte_offset` — The byte offset being resolved.
    /// * `raw_line` — The physical (1-indexed) line number before remapping.
    /// * `default_name` — The original file name (used if no directive overrides it).
    ///
    /// # Returns
    ///
    /// A tuple of `(remapped_line, remapped_file_name)`.
    fn apply_line_directive(
        &self,
        file_id: FileId,
        byte_offset: u32,
        raw_line: u32,
        default_name: &str,
    ) -> (u32, String) {
        // Search directives in reverse to find the most recent applicable one.
        // Directives are stored in insertion (source) order, so the last one
        // with byte_offset <= query offset for the same file wins.
        let mut best: Option<&LineDirective> = None;

        for directive in self.line_directives.iter().rev() {
            if directive.file_id == file_id && directive.byte_offset <= byte_offset {
                // Found the most recent applicable directive
                best = Some(directive);
                break;
            }
        }

        match best {
            Some(directive) => {
                // Compute the physical line of the directive's position
                let file = &self.files[file_id.0 as usize];
                let directive_physical_line = {
                    let offsets = &file.line_offsets;
                    match offsets.partition_point(|&off| off <= directive.byte_offset) {
                        0 => 1u32, // First line
                        n => n as u32,
                    }
                };

                // The remapped line = directive's new_line + (current physical line - directive physical line)
                // This preserves the line delta from the directive to the current position
                let line_delta = raw_line.saturating_sub(directive_physical_line);
                let remapped_line = directive.new_line + line_delta;

                let remapped_file = directive
                    .new_file
                    .as_deref()
                    .unwrap_or(default_name)
                    .to_string();

                (remapped_line, remapped_file)
            }
            None => (raw_line, default_name.to_string()),
        }
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the line offset table for a source file's content.
///
/// Scans the content for newline characters and records the byte offset of
/// each line start. The table always begins with 0 (the start of the file
/// is always line 0). Handles both Unix (`\n`) and Windows (`\r\n`) line
/// endings correctly — in both cases, the line start after the newline is
/// the byte immediately following the `\n`.
///
/// # Arguments
///
/// * `content` — The file content to analyze.
///
/// # Returns
///
/// A `Vec<u32>` where `result[i]` is the byte offset of the start of
/// line `i` (0-indexed).
///
/// # Examples
///
/// ```text
/// ""         -> [0]                     // Empty file: one line starting at 0
/// "abc"      -> [0]                     // No newline: one line
/// "abc\n"    -> [0, 4]                  // One newline: two lines
/// "abc\ndef" -> [0, 4]                  // Two lines, no trailing newline
/// "abc\r\n"  -> [0, 5]                  // Windows line ending
/// "\n\n"     -> [0, 1, 2]              // Two empty lines + third empty line
/// ```
fn compute_line_offsets(content: &str) -> Vec<u32> {
    let bytes = content.as_bytes();
    let len = bytes.len();

    // Pre-allocate with an estimate. Typical source files have ~40-80 chars per line.
    // Start with at least capacity for the first entry plus a reasonable estimate.
    let estimated_lines = if len > 0 { len / 40 + 2 } else { 1 };
    let mut offsets = Vec::with_capacity(estimated_lines);

    // Line 0 always starts at offset 0
    offsets.push(0u32);

    let mut i = 0;
    while i < len {
        if bytes[i] == b'\n' {
            // Start of next line is the byte after '\n'
            offsets.push((i + 1) as u32);
            i += 1;
        } else if bytes[i] == b'\r' {
            // Handle \r\n (Windows) and lone \r (old Mac)
            if i + 1 < len && bytes[i + 1] == b'\n' {
                // \r\n — next line starts after the \n
                offsets.push((i + 2) as u32);
                i += 2;
            } else {
                // Lone \r — treat as line ending (old Mac style)
                offsets.push((i + 1) as u32);
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    offsets
}

/// Implements `Display` for `FileId` to support debug formatting.
impl core::fmt::Display for FileId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "FileId({})", self.0)
    }
}

/// Implements `Display` for `SourceLocation` for diagnostic messages.
///
/// Formats as `filename:line:column` matching GCC/Clang convention.
impl core::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}:{}", self.file_name, self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_file() {
        let mut sm = SourceMap::new();
        let fid = sm.add_file("empty.c".to_string(), String::new());
        assert_eq!(fid, FileId(0));

        let file = sm.get_file(fid);
        assert_eq!(file.name, "empty.c");
        assert_eq!(file.content, "");
        assert_eq!(file.line_offsets, vec![0]);
    }

    #[test]
    fn test_single_line_no_newline() {
        let mut sm = SourceMap::new();
        let fid = sm.add_file("test.c".to_string(), "int x = 0;".to_string());

        let file = sm.get_file(fid);
        assert_eq!(file.line_offsets, vec![0]);

        let loc = sm.lookup_location(fid, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        let loc = sm.lookup_location(fid, 4);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 5);
    }

    #[test]
    fn test_multi_line_unix() {
        let mut sm = SourceMap::new();
        let content = "int x;\nint y;\nint z;\n".to_string();
        let fid = sm.add_file("test.c".to_string(), content);

        let file = sm.get_file(fid);
        assert_eq!(file.line_offsets, vec![0, 7, 14, 21]);

        // "int x;\n" = 7 bytes, line 1
        let loc = sm.lookup_location(fid, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // byte 4 = 'x' on line 1
        let loc = sm.lookup_location(fid, 4);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 5);

        // byte 7 = start of line 2
        let loc = sm.lookup_location(fid, 7);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);

        // byte 14 = start of line 3
        let loc = sm.lookup_location(fid, 14);
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 1);

        // byte 17 = 'z' on line 3 (offset 14 + 3 = 17, column = 17-14+1 = 4)
        let loc = sm.lookup_location(fid, 17);
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 4);
    }

    #[test]
    fn test_multi_line_windows() {
        let mut sm = SourceMap::new();
        // "ab\r\ncd\r\n" = bytes: a(0) b(1) \r(2) \n(3) c(4) d(5) \r(6) \n(7)
        let content = "ab\r\ncd\r\n".to_string();
        let fid = sm.add_file("win.c".to_string(), content);

        let file = sm.get_file(fid);
        assert_eq!(file.line_offsets, vec![0, 4, 8]);

        let loc = sm.lookup_location(fid, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // byte 4 = 'c', start of line 2
        let loc = sm.lookup_location(fid, 4);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_get_line_content() {
        let mut sm = SourceMap::new();
        let content = "first line\nsecond line\nthird\n".to_string();
        let fid = sm.add_file("lines.c".to_string(), content);

        assert_eq!(sm.get_line_content(fid, 1), "first line");
        assert_eq!(sm.get_line_content(fid, 2), "second line");
        assert_eq!(sm.get_line_content(fid, 3), "third");
        // Line beyond content
        assert_eq!(sm.get_line_content(fid, 5), "");
        // Line 0 is invalid (1-indexed)
        assert_eq!(sm.get_line_content(fid, 0), "");
    }

    #[test]
    fn test_get_line_content_windows() {
        let mut sm = SourceMap::new();
        let content = "hello\r\nworld\r\n".to_string();
        let fid = sm.add_file("win.c".to_string(), content);

        assert_eq!(sm.get_line_content(fid, 1), "hello");
        assert_eq!(sm.get_line_content(fid, 2), "world");
    }

    #[test]
    fn test_get_snippet() {
        let mut sm = SourceMap::new();
        let content = "abcdefghij".to_string();
        let fid = sm.add_file("snip.c".to_string(), content);

        assert_eq!(sm.get_snippet(fid, 0, 3), "abc");
        assert_eq!(sm.get_snippet(fid, 3, 6), "def");
        assert_eq!(sm.get_snippet(fid, 7, 10), "hij");
        // Beyond end
        assert_eq!(sm.get_snippet(fid, 8, 100), "ij");
        // Empty range
        assert_eq!(sm.get_snippet(fid, 5, 5), "");
        // Inverted range
        assert_eq!(sm.get_snippet(fid, 5, 3), "");
    }

    #[test]
    fn test_multiple_files() {
        let mut sm = SourceMap::new();
        let fid0 = sm.add_file("a.c".to_string(), "int a;\n".to_string());
        let fid1 = sm.add_file("b.c".to_string(), "int b;\nint c;\n".to_string());

        assert_eq!(fid0, FileId(0));
        assert_eq!(fid1, FileId(1));
        assert_eq!(sm.file_count(), 2);

        assert_eq!(sm.get_file(fid0).name, "a.c");
        assert_eq!(sm.get_file(fid1).name, "b.c");

        let loc0 = sm.lookup_location(fid0, 0);
        assert_eq!(loc0.file_name, "a.c");
        assert_eq!(loc0.line, 1);

        let loc1 = sm.lookup_location(fid1, 7);
        assert_eq!(loc1.file_name, "b.c");
        assert_eq!(loc1.line, 2);
        assert_eq!(loc1.column, 1);
    }

    #[test]
    fn test_line_directive_basic() {
        let mut sm = SourceMap::new();
        // Content: line1(0-5) \n(6) line2(7-12) \n(13) line3(14-19) \n(20)
        let content = "line 1\nline 2\nline 3\n".to_string();
        let fid = sm.add_file("orig.c".to_string(), content);

        // #line 100 at byte offset 7 (start of physical line 2)
        sm.add_line_directive(LineDirective {
            file_id: fid,
            byte_offset: 7,
            new_line: 100,
            new_file: Some("remapped.h".to_string()),
        });

        // Before directive: physical location
        let loc = sm.lookup_location(fid, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.file_name, "orig.c");

        // At directive position: remapped to line 100
        let loc = sm.lookup_location(fid, 7);
        assert_eq!(loc.line, 100);
        assert_eq!(loc.file_name, "remapped.h");

        // Next physical line (line 3) should be line 101
        let loc = sm.lookup_location(fid, 14);
        assert_eq!(loc.line, 101);
        assert_eq!(loc.file_name, "remapped.h");
    }

    #[test]
    fn test_line_directive_no_file() {
        let mut sm = SourceMap::new();
        let content = "aaa\nbbb\nccc\n".to_string();
        let fid = sm.add_file("test.c".to_string(), content);

        // #line 50 without file name change
        sm.add_line_directive(LineDirective {
            file_id: fid,
            byte_offset: 4,
            new_line: 50,
            new_file: None,
        });

        let loc = sm.lookup_location(fid, 4);
        assert_eq!(loc.line, 50);
        assert_eq!(loc.file_name, "test.c"); // Original file name preserved
    }

    #[test]
    fn test_compute_line_offsets_empty() {
        let offsets = compute_line_offsets("");
        assert_eq!(offsets, vec![0]);
    }

    #[test]
    fn test_compute_line_offsets_no_newline() {
        let offsets = compute_line_offsets("hello world");
        assert_eq!(offsets, vec![0]);
    }

    #[test]
    fn test_compute_line_offsets_consecutive_newlines() {
        let offsets = compute_line_offsets("\n\n\n");
        assert_eq!(offsets, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_compute_line_offsets_mixed_endings() {
        // "a\r\nb\nc\r" = a(0) \r(1) \n(2) b(3) \n(4) c(5) \r(6)
        let offsets = compute_line_offsets("a\r\nb\nc\r");
        assert_eq!(offsets, vec![0, 3, 5, 7]);
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation {
            file_id: FileId(0),
            line: 42,
            column: 7,
            file_name: "test.c".to_string(),
        };
        assert_eq!(format!("{}", loc), "test.c:42:7");
    }

    #[test]
    fn test_file_id_display() {
        let fid = FileId(5);
        assert_eq!(format!("{}", fid), "FileId(5)");
    }

    #[test]
    fn test_boundary_offsets() {
        let mut sm = SourceMap::new();
        let content = "a\nb\nc".to_string();
        let fid = sm.add_file("test.c".to_string(), content);

        // Byte 0 = 'a', line 1 col 1
        let loc = sm.lookup_location(fid, 0);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 1);

        // Byte 1 = '\n', still line 1 col 2
        let loc = sm.lookup_location(fid, 1);
        assert_eq!(loc.line, 1);
        assert_eq!(loc.column, 2);

        // Byte 2 = 'b', line 2 col 1
        let loc = sm.lookup_location(fid, 2);
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 1);

        // Byte 4 = 'c', line 3 col 1
        let loc = sm.lookup_location(fid, 4);
        assert_eq!(loc.line, 3);
        assert_eq!(loc.column, 1);
    }

    #[test]
    fn test_get_line_content_last_line_no_newline() {
        let mut sm = SourceMap::new();
        let content = "first\nsecond".to_string();
        let fid = sm.add_file("test.c".to_string(), content);

        assert_eq!(sm.get_line_content(fid, 1), "first");
        assert_eq!(sm.get_line_content(fid, 2), "second");
    }

    #[test]
    fn test_default_impl() {
        let sm = SourceMap::default();
        assert_eq!(sm.file_count(), 0);
    }
}
