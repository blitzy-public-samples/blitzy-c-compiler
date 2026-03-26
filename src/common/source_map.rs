//! Source file tracking for the BCC compiler.
//!
//! Provides [`SourceMap`] for tracking loaded source files by ID, line offset
//! tables for efficient O(log n) line/column lookups via binary search, and
//! `#line` directive remapping. This module supplies the source location context
//! needed by `diagnostics.rs` for formatted, GCC-compatible error messages.
//!
//! # Design
//!
//! Each source file is assigned a stable `u32` file ID when added to the
//! [`SourceMap`]. Line offset tables are precomputed by scanning for `\n`
//! characters, enabling O(log n) lookups using `binary_search` on the
//! `line_offsets` vector. `#line` directives are stored per-file, sorted
//! by byte offset, for efficient binary-search–based remapping.
//!
//! # Zero-Dependency
//!
//! This module depends only on the Rust standard library (`std`). It has
//! NO dependencies on other BCC modules (fully standalone in `common/`).

use std::fmt;

// ---------------------------------------------------------------------------
// SourceLocation — a resolved position in a source file
// ---------------------------------------------------------------------------

/// A resolved source location with file name, line, and column information.
///
/// Both `line` and `column` are **1-indexed** for display purposes, matching
/// the convention used by GCC, Clang, and most other compilers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    /// File ID in the [`SourceMap`] that owns this location.
    pub file_id: u32,
    /// Human-readable filename (may be remapped by `#line`).
    pub filename: String,
    /// 1-indexed line number.
    pub line: u32,
    /// 1-indexed column number (byte-based).
    pub column: u32,
}

impl fmt::Display for SourceLocation {
    /// Formats as `filename:line:column` (e.g. `hello.c:42:10`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.filename, self.line, self.column)
    }
}

// ---------------------------------------------------------------------------
// LineDirective — a `#line` remapping entry
// ---------------------------------------------------------------------------

/// Represents a `#line` preprocessor directive that remaps the reported
/// filename and/or line number for subsequent source locations.
///
/// When the preprocessor encounters `#line N "file"`, it registers a
/// `LineDirective` so that diagnostics report the adjusted location rather
/// than the physical one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineDirective {
    /// Which source file the directive appears in (file ID).
    pub file_id: u32,
    /// Byte offset in the source where the `#line` directive occurs.
    pub directive_offset: u32,
    /// New line number specified by the directive.
    pub new_line: u32,
    /// New filename if specified (e.g. `#line 100 "foo.h"`), or `None` if
    /// only the line number is changed.
    pub new_filename: Option<String>,
}

// ---------------------------------------------------------------------------
// SourceFile — a single loaded source file with precomputed line offsets
// ---------------------------------------------------------------------------

/// A source file tracked by the [`SourceMap`].
///
/// Stores the full source content together with a precomputed line-offset
/// table that enables O(log n) byte-offset → (line, column) lookups via
/// binary search.
#[derive(Debug)]
pub struct SourceFile {
    /// Stable unique file ID (index into [`SourceMap::files`]).
    pub id: u32,
    /// Original filename / path.
    pub filename: String,
    /// Full source content (may contain PUA-encoded bytes for non-UTF-8).
    pub content: String,
    /// Byte offsets of each line start. `line_offsets[0] == 0` (line 1 starts
    /// at offset 0). Each subsequent entry is the byte offset immediately
    /// following a `\n` character.
    line_offsets: Vec<u32>,
}

impl SourceFile {
    /// Create a new `SourceFile`, computing the line-offset table by scanning
    /// for `\n` characters in `content`.
    fn new(id: u32, filename: String, content: String) -> Self {
        let line_offsets = Self::compute_line_offsets(&content);
        SourceFile {
            id,
            filename,
            content,
            line_offsets,
        }
    }

    /// Scan the content for newline characters and build the line-offset table.
    ///
    /// The first entry is always `0` (line 1 starts at byte 0). Each
    /// additional entry records the byte offset of the character immediately
    /// after a `\n`.
    fn compute_line_offsets(content: &str) -> Vec<u32> {
        let bytes = content.as_bytes();
        // Pre-allocate with a rough estimate (one line per 40 bytes on average).
        let mut offsets = Vec::with_capacity(1 + bytes.len() / 40);
        offsets.push(0); // Line 1 starts at offset 0.

        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' {
                // The next line starts at the byte after the newline.
                offsets.push((i + 1) as u32);
            }
        }
        offsets
    }

    /// Look up the (line, column) for a given byte offset using binary search.
    ///
    /// Both line and column are **1-indexed**.
    ///
    /// # Complexity
    ///
    /// O(log n) where n is the number of lines.
    ///
    /// # Edge cases
    ///
    /// * If `byte_offset` is beyond the end of the file the last line is used.
    /// * An empty file has a single line starting at offset 0.
    pub fn lookup_line_col(&self, byte_offset: u32) -> (u32, u32) {
        // Binary search: find the rightmost line whose start offset is <= byte_offset.
        let line_index = match self.line_offsets.binary_search(&byte_offset) {
            // Exact match — byte_offset is at the start of a line.
            Ok(idx) => idx,
            // Not an exact match — `idx` is where byte_offset *would* be inserted,
            // so the line containing byte_offset is `idx - 1`.
            Err(idx) => {
                if idx == 0 {
                    0
                } else {
                    idx - 1
                }
            }
        };

        let line_start = self.line_offsets[line_index];
        let line_number = (line_index as u32) + 1; // 1-indexed
        let column = byte_offset.saturating_sub(line_start) + 1; // 1-indexed

        (line_number, column)
    }

    /// Return the content of the given line (1-indexed).
    ///
    /// The returned slice does **not** include the trailing `\n`, if any.
    /// Returns an empty string for out-of-range line numbers.
    pub fn get_line_content(&self, line: u32) -> &str {
        if line == 0 || (line as usize) > self.line_offsets.len() {
            return "";
        }
        let idx = (line - 1) as usize;
        let start = self.line_offsets[idx] as usize;

        // Determine end: either the start of the next line (minus 1 for the \n)
        // or the end of the content.
        let end = if idx + 1 < self.line_offsets.len() {
            // Strip the trailing '\n' character.
            let next_start = self.line_offsets[idx + 1] as usize;
            if next_start > 0 && self.content.as_bytes().get(next_start - 1) == Some(&b'\n') {
                next_start - 1
            } else {
                next_start
            }
        } else {
            self.content.len()
        };

        // Clamp to content bounds for safety.
        let start = start.min(self.content.len());
        let end = end.min(self.content.len());

        // Also strip a trailing '\r' for Windows-style line endings that
        // might appear in cross-platform sources.
        let slice = &self.content[start..end];
        slice.strip_suffix('\r').unwrap_or(slice)
    }

    /// Return the total number of lines in this source file.
    ///
    /// An empty file is considered to have 1 line.
    pub fn line_count(&self) -> usize {
        self.line_offsets.len()
    }
}

// ---------------------------------------------------------------------------
// SourceMap — the central registry of all loaded source files
// ---------------------------------------------------------------------------

/// Central registry for all source files loaded during compilation.
///
/// Files are added via [`add_file`](SourceMap::add_file) and assigned stable
/// `u32` IDs (starting from `0`). Line/column lookups are O(log n) via
/// precomputed line-offset tables. `#line` directive remapping is supported
/// through [`add_line_directive`](SourceMap::add_line_directive) and
/// [`resolve_location`](SourceMap::resolve_location).
#[derive(Debug)]
pub struct SourceMap {
    /// All loaded files, indexed by their `u32` file ID.
    files: Vec<SourceFile>,
    /// `#line` directives grouped by file ID. Each inner `Vec` is kept sorted
    /// by `directive_offset` for binary-search lookups.
    line_directives: Vec<Vec<LineDirective>>,
}

impl SourceMap {
    /// Create a new, empty `SourceMap`.
    pub fn new() -> Self {
        SourceMap {
            files: Vec::new(),
            line_directives: Vec::new(),
        }
    }

    /// Add a source file to the map.
    ///
    /// Returns the stable `u32` file ID assigned to this file. IDs are
    /// sequential starting from `0`.
    pub fn add_file(&mut self, filename: String, content: String) -> u32 {
        let id = self.files.len() as u32;
        let file = SourceFile::new(id, filename, content);
        self.files.push(file);
        // Ensure the directives vector has a slot for the new file.
        self.line_directives.push(Vec::new());
        id
    }

    /// Look up a file by its ID.
    ///
    /// Returns `None` if `file_id` is out of range.
    pub fn get_file(&self, file_id: u32) -> Option<&SourceFile> {
        self.files.get(file_id as usize)
    }

    /// Resolve a byte offset within a file to a [`SourceLocation`] using the
    /// **physical** line-offset table (ignoring `#line` directives).
    ///
    /// Returns `None` if `file_id` is invalid.
    pub fn lookup_location(&self, file_id: u32, byte_offset: u32) -> Option<SourceLocation> {
        let file = self.files.get(file_id as usize)?;
        let (line, column) = file.lookup_line_col(byte_offset);
        Some(SourceLocation {
            file_id,
            filename: file.filename.clone(),
            line,
            column,
        })
    }

    /// Convenience accessor for a file's name.
    ///
    /// Returns `None` if `file_id` is out of range.
    pub fn get_filename(&self, file_id: u32) -> Option<&str> {
        self.files
            .get(file_id as usize)
            .map(|f| f.filename.as_str())
    }

    /// Return all `#line` directive filenames for a given file.
    ///
    /// This is used by the DWARF debug information generator to include
    /// `#line`-specified filenames in the `.debug_line` file table.
    pub fn get_line_directive_filenames(&self, file_id: u32) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(directives) = self.line_directives.get(file_id as usize) {
            for d in directives {
                if let Some(ref name) = d.new_filename {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
            }
        }
        names
    }

    /// Register a `#line` directive for future
    /// [`resolve_location`](SourceMap::resolve_location) lookups.
    ///
    /// Directives are maintained in sorted order by `directive_offset` within
    /// each file's directive list.
    pub fn add_line_directive(&mut self, directive: LineDirective) {
        let fid = directive.file_id as usize;

        // Grow the per-file directive storage if the file was added after
        // the initial creation (should not normally happen, but be safe).
        while self.line_directives.len() <= fid {
            self.line_directives.push(Vec::new());
        }

        let directives = &mut self.line_directives[fid];

        // Insert in sorted order by directive_offset. Most additions will be
        // sequential (ascending offsets) so a push-to-end fast-path is used.
        let offset = directive.directive_offset;
        if directives.is_empty()
            || directives
                .last()
                .map_or(false, |d| d.directive_offset <= offset)
        {
            directives.push(directive);
        } else {
            // Find insertion point via binary search on directive_offset.
            let pos = directives
                .binary_search_by_key(&offset, |d| d.directive_offset)
                .unwrap_or_else(|e| e);
            directives.insert(pos, directive);
        }
    }

    /// Resolve a byte offset to a [`SourceLocation`], applying `#line`
    /// directive remapping when applicable.
    ///
    /// 1. Performs a physical line/column lookup via the line-offset table.
    /// 2. Searches for the most recent `#line` directive at or before
    ///    `byte_offset` in the same file.
    /// 3. If such a directive exists, adjusts the reported line number and
    ///    optionally the filename.
    ///
    /// If `file_id` is invalid, returns a dummy location with
    /// `"<unknown>"` filename.
    pub fn resolve_location(&self, file_id: u32, byte_offset: u32) -> SourceLocation {
        // Perform physical lookup first.
        let file = match self.files.get(file_id as usize) {
            Some(f) => f,
            None => {
                return SourceLocation {
                    file_id,
                    filename: "<unknown>".to_string(),
                    line: 0,
                    column: 0,
                };
            }
        };

        let (phys_line, column) = file.lookup_line_col(byte_offset);
        let mut resolved_line = phys_line;
        let mut resolved_filename = file.filename.clone();

        // Check for #line directive remapping.
        if let Some(directives) = self.line_directives.get(file_id as usize) {
            if !directives.is_empty() {
                // Find the last directive whose offset is <= byte_offset.
                let search = directives.binary_search_by_key(&byte_offset, |d| d.directive_offset);
                let idx = match search {
                    Ok(i) => Some(i),
                    Err(0) => None, // All directives are after byte_offset.
                    Err(i) => Some(i - 1),
                };

                if let Some(di) = idx {
                    let dir = &directives[di];

                    // Compute the physical line of the directive itself so we
                    // can calculate how many lines have elapsed since.
                    let (dir_phys_line, _) = file.lookup_line_col(dir.directive_offset);

                    // Lines after the directive are reported relative to
                    // the directive's new_line value.
                    let lines_since_directive = phys_line.saturating_sub(dir_phys_line);
                    resolved_line = dir.new_line + lines_since_directive;

                    if let Some(ref new_name) = dir.new_filename {
                        resolved_filename = new_name.clone();
                    }
                }
            }
        }

        SourceLocation {
            file_id,
            filename: resolved_filename,
            line: resolved_line,
            column,
        }
    }

    /// Format a source span as a GCC-compatible `filename:line:column` string.
    ///
    /// Uses [`resolve_location`](SourceMap::resolve_location) on the *start*
    /// offset so `#line` remapping is applied. The `end` offset is accepted
    /// for future use (e.g. multi-line span display) but currently only the
    /// start is formatted.
    pub fn format_span(&self, file_id: u32, start: u32, _end: u32) -> String {
        let loc = self.resolve_location(file_id, start);
        format!("{}:{}:{}", loc.filename, loc.line, loc.column)
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SourceFile tests --------------------------------------------------

    #[test]
    fn test_line_offsets_simple() {
        let sf = SourceFile::new(0, "test.c".into(), "abc\ndef\nghi\n".into());
        // Lines: "abc\n" starts at 0, "def\n" at 4, "ghi\n" at 8, "" at 12
        assert_eq!(sf.line_offsets, vec![0, 4, 8, 12]);
    }

    #[test]
    fn test_line_offsets_no_trailing_newline() {
        let sf = SourceFile::new(0, "test.c".into(), "abc\ndef".into());
        // Lines: "abc\n" starts at 0, "def" at 4
        assert_eq!(sf.line_offsets, vec![0, 4]);
    }

    #[test]
    fn test_line_offsets_empty() {
        let sf = SourceFile::new(0, "empty.c".into(), "".into());
        // Even an empty file has line 1 starting at offset 0.
        assert_eq!(sf.line_offsets, vec![0]);
    }

    #[test]
    fn test_line_offsets_single_newline() {
        let sf = SourceFile::new(0, "nl.c".into(), "\n".into());
        assert_eq!(sf.line_offsets, vec![0, 1]);
    }

    #[test]
    fn test_lookup_line_col_first_char() {
        let sf = SourceFile::new(0, "t.c".into(), "abc\ndef\n".into());
        assert_eq!(sf.lookup_line_col(0), (1, 1));
    }

    #[test]
    fn test_lookup_line_col_middle_of_line() {
        let sf = SourceFile::new(0, "t.c".into(), "abc\ndef\n".into());
        // 'b' is at offset 1 → line 1, column 2
        assert_eq!(sf.lookup_line_col(1), (1, 2));
        // 'c' is at offset 2 → line 1, column 3
        assert_eq!(sf.lookup_line_col(2), (1, 3));
    }

    #[test]
    fn test_lookup_line_col_second_line() {
        let sf = SourceFile::new(0, "t.c".into(), "abc\ndef\n".into());
        // 'd' is at offset 4 → line 2, column 1
        assert_eq!(sf.lookup_line_col(4), (2, 1));
        // 'e' is at offset 5 → line 2, column 2
        assert_eq!(sf.lookup_line_col(5), (2, 2));
    }

    #[test]
    fn test_lookup_line_col_newline_char() {
        let sf = SourceFile::new(0, "t.c".into(), "abc\ndef\n".into());
        // The '\n' after "abc" is at offset 3 → line 1, column 4
        assert_eq!(sf.lookup_line_col(3), (1, 4));
    }

    #[test]
    fn test_lookup_line_col_empty_file() {
        let sf = SourceFile::new(0, "e.c".into(), "".into());
        assert_eq!(sf.lookup_line_col(0), (1, 1));
    }

    #[test]
    fn test_lookup_line_col_beyond_end() {
        let sf = SourceFile::new(0, "t.c".into(), "ab".into());
        // Offset 10 is beyond end; should clamp to last line.
        let (line, _col) = sf.lookup_line_col(10);
        assert_eq!(line, 1);
    }

    #[test]
    fn test_get_line_content() {
        let sf = SourceFile::new(0, "t.c".into(), "hello\nworld\nfoo".into());
        assert_eq!(sf.get_line_content(1), "hello");
        assert_eq!(sf.get_line_content(2), "world");
        assert_eq!(sf.get_line_content(3), "foo");
    }

    #[test]
    fn test_get_line_content_trailing_newline() {
        let sf = SourceFile::new(0, "t.c".into(), "aaa\nbbb\n".into());
        assert_eq!(sf.get_line_content(1), "aaa");
        assert_eq!(sf.get_line_content(2), "bbb");
        // Line 3 is the empty line after the trailing \n
        assert_eq!(sf.get_line_content(3), "");
    }

    #[test]
    fn test_get_line_content_out_of_range() {
        let sf = SourceFile::new(0, "t.c".into(), "x".into());
        assert_eq!(sf.get_line_content(0), ""); // 0 is invalid (1-indexed)
        assert_eq!(sf.get_line_content(99), ""); // Way beyond
    }

    #[test]
    fn test_line_count() {
        let sf = SourceFile::new(0, "t.c".into(), "a\nb\nc".into());
        assert_eq!(sf.line_count(), 3);

        let sf2 = SourceFile::new(0, "t.c".into(), "a\nb\nc\n".into());
        assert_eq!(sf2.line_count(), 4); // Trailing newline creates an empty 4th line entry

        let sf3 = SourceFile::new(0, "t.c".into(), "".into());
        assert_eq!(sf3.line_count(), 1);
    }

    // -- SourceLocation tests -----------------------------------------------

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation {
            file_id: 0,
            filename: "hello.c".into(),
            line: 42,
            column: 10,
        };
        assert_eq!(format!("{}", loc), "hello.c:42:10");
    }

    #[test]
    fn test_source_location_display_path() {
        let loc = SourceLocation {
            file_id: 1,
            filename: "/usr/include/stdio.h".into(),
            line: 1,
            column: 1,
        };
        assert_eq!(format!("{}", loc), "/usr/include/stdio.h:1:1");
    }

    // -- SourceMap tests ----------------------------------------------------

    #[test]
    fn test_source_map_add_file_sequential_ids() {
        let mut sm = SourceMap::new();
        let id0 = sm.add_file("a.c".into(), "int x;".into());
        let id1 = sm.add_file("b.c".into(), "int y;".into());
        let id2 = sm.add_file("c.c".into(), "int z;".into());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_source_map_get_file() {
        let mut sm = SourceMap::new();
        let id = sm.add_file("test.c".into(), "hello".into());
        let file = sm.get_file(id).unwrap();
        assert_eq!(file.filename, "test.c");
        assert_eq!(file.content, "hello");
        assert!(sm.get_file(999).is_none());
    }

    #[test]
    fn test_source_map_get_filename() {
        let mut sm = SourceMap::new();
        sm.add_file("foo.c".into(), "".into());
        assert_eq!(sm.get_filename(0), Some("foo.c"));
        assert_eq!(sm.get_filename(5), None);
    }

    #[test]
    fn test_source_map_lookup_location() {
        let mut sm = SourceMap::new();
        sm.add_file("test.c".into(), "abc\ndef\nghi".into());
        let loc = sm.lookup_location(0, 5).unwrap(); // 'e' in "def"
        assert_eq!(loc.filename, "test.c");
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 2);
        assert_eq!(loc.file_id, 0);
    }

    #[test]
    fn test_source_map_lookup_location_invalid_file() {
        let sm = SourceMap::new();
        assert!(sm.lookup_location(0, 0).is_none());
    }

    #[test]
    fn test_source_map_default() {
        let sm = SourceMap::default();
        assert!(sm.get_file(0).is_none());
    }

    // -- #line directive tests -----------------------------------------------

    #[test]
    fn test_line_directive_simple() {
        let mut sm = SourceMap::new();
        // File content: line 1 = "aaa", line 2 = "#line 100 \"foo.h\"", line 3 = "bbb"
        let content = "aaa\n#line 100 \"foo.h\"\nbbb\n";
        let fid = sm.add_file("real.c".into(), content.into());

        // The #line directive is at the start of physical line 2, offset 4.
        sm.add_line_directive(LineDirective {
            file_id: fid,
            directive_offset: 4,
            new_line: 100,
            new_filename: Some("foo.h".into()),
        });

        // Before the directive — physical location unchanged.
        let loc0 = sm.resolve_location(fid, 0);
        assert_eq!(loc0.line, 1);
        assert_eq!(loc0.filename, "real.c");

        // On the directive line itself.
        let loc1 = sm.resolve_location(fid, 4);
        assert_eq!(loc1.line, 100);
        assert_eq!(loc1.filename, "foo.h");

        // One line after the directive → 100 + 1 = 101
        let loc2 = sm.resolve_location(fid, 24); // "bbb\n" starts at offset 24
        assert_eq!(loc2.line, 101);
        assert_eq!(loc2.filename, "foo.h");
    }

    #[test]
    fn test_line_directive_no_filename() {
        let mut sm = SourceMap::new();
        let content = "aaa\nbbb\nccc\n";
        let fid = sm.add_file("orig.c".into(), content.into());

        // #line 50 without a filename — only changes line number.
        sm.add_line_directive(LineDirective {
            file_id: fid,
            directive_offset: 4, // Start of "bbb"
            new_line: 50,
            new_filename: None,
        });

        let loc = sm.resolve_location(fid, 4);
        assert_eq!(loc.line, 50);
        assert_eq!(loc.filename, "orig.c"); // Filename unchanged.
    }

    #[test]
    fn test_line_directive_multiple() {
        let mut sm = SourceMap::new();
        // 5 physical lines: offsets 0, 4, 8, 12, 16
        let content = "aaa\nbbb\nccc\nddd\neee\n";
        let fid = sm.add_file("x.c".into(), content.into());

        sm.add_line_directive(LineDirective {
            file_id: fid,
            directive_offset: 4, // line 2
            new_line: 200,
            new_filename: Some("a.h".into()),
        });
        sm.add_line_directive(LineDirective {
            file_id: fid,
            directive_offset: 12, // line 4
            new_line: 300,
            new_filename: Some("b.h".into()),
        });

        // Before first directive.
        let loc0 = sm.resolve_location(fid, 0);
        assert_eq!(loc0.line, 1);
        assert_eq!(loc0.filename, "x.c");

        // In first directive region.
        let loc1 = sm.resolve_location(fid, 8); // line 3 physically
        assert_eq!(loc1.line, 201);
        assert_eq!(loc1.filename, "a.h");

        // In second directive region.
        let loc2 = sm.resolve_location(fid, 16); // line 5 physically
        assert_eq!(loc2.line, 301);
        assert_eq!(loc2.filename, "b.h");
    }

    #[test]
    fn test_resolve_location_invalid_file() {
        let sm = SourceMap::new();
        let loc = sm.resolve_location(99, 0);
        assert_eq!(loc.filename, "<unknown>");
        assert_eq!(loc.line, 0);
        assert_eq!(loc.column, 0);
    }

    // -- format_span tests --------------------------------------------------

    #[test]
    fn test_format_span_basic() {
        let mut sm = SourceMap::new();
        sm.add_file("hello.c".into(), "int main() {}\n".into());
        let s = sm.format_span(0, 4, 8); // "main" starts at col 5
        assert_eq!(s, "hello.c:1:5");
    }

    #[test]
    fn test_format_span_with_directive() {
        let mut sm = SourceMap::new();
        let content = "aaa\nbbb\n";
        let fid = sm.add_file("real.c".into(), content.into());
        sm.add_line_directive(LineDirective {
            file_id: fid,
            directive_offset: 0,
            new_line: 42,
            new_filename: Some("mapped.h".into()),
        });
        let s = sm.format_span(fid, 0, 3);
        assert_eq!(s, "mapped.h:42:1");
    }

    // -- Stress / edge-case tests -------------------------------------------

    #[test]
    fn test_large_file_binary_search_correctness() {
        // Build a file with 10,000 lines to exercise binary search.
        let mut content = String::new();
        for i in 0..10_000 {
            content.push_str(&format!("line {}\n", i));
        }
        let mut sm = SourceMap::new();
        let fid = sm.add_file("big.c".into(), content.clone());
        let file = sm.get_file(fid).unwrap();

        // Check a few known positions.
        assert_eq!(file.lookup_line_col(0), (1, 1));
        // "line 0\n" is 7 bytes, so line 2 starts at offset 7.
        assert_eq!(file.lookup_line_col(7), (2, 1));
        // Offset 8 is the second character of line 2.
        assert_eq!(file.lookup_line_col(8), (2, 2));

        // Verify last line.
        let total_lines = file.line_count();
        assert!(total_lines >= 10_000);
    }

    #[test]
    fn test_windows_line_endings_get_line_content() {
        let sf = SourceFile::new(0, "win.c".into(), "aaa\r\nbbb\r\n".into());
        // With \r\n, the \n positions give line_offsets of [0, 5, 10].
        // get_line_content should strip the \r.
        assert_eq!(sf.get_line_content(1), "aaa");
        assert_eq!(sf.get_line_content(2), "bbb");
    }
}
