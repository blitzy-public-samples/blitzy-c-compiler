//! Character-level scanner with PUA-aware UTF-8 handling.
//!
//! This module provides the lowest-level input abstraction for the BCC lexer.
//! The [`Scanner`] reads individual characters from the preprocessor's expanded
//! source string, provides lookahead buffering (at least 2 characters), tracks
//! position (byte offset, line number, column number), and handles PUA-encoded
//! bytes transparently.
//!
//! # PUA Transparency
//!
//! Non-UTF-8 bytes in C source files are encoded as Unicode Private Use Area
//! (PUA) code points (U+E080–U+E0FF) during source file reading. The scanner
//! treats these as normal characters — they flow through to string literals
//! and identifiers without special treatment. Since the source is a valid
//! Rust `&str`, all PUA code points are valid UTF-8-encoded Unicode characters
//! and [`CharIndices`] handles them correctly.
//!
//! # Line Ending Normalization
//!
//! The scanner normalizes all line endings to `\n`:
//! - `\n` → returned as `\n`
//! - `\r\n` → consumed as a unit, returned as `\n`
//! - `\r` alone → returned as `\n`
//!
//! Byte offsets correctly account for the actual bytes consumed (1 for `\n`
//! or standalone `\r`, 2 for `\r\n`).
//!
//! # Position Tracking
//!
//! Positions use byte offsets (0-indexed) for `Span` construction, compatible
//! with [`crate::common::diagnostics::Span`]. Line numbers are 1-indexed,
//! column numbers are 1-indexed and measured in bytes (not characters).

use crate::common::encoding;

use std::iter::Peekable;
use std::str::CharIndices;

// ---------------------------------------------------------------------------
// Position — source location tracking
// ---------------------------------------------------------------------------

/// Source position within a file — byte offset, line, and column.
///
/// The byte offset is the definitive position for `Span` construction. The
/// line and column numbers are for human-readable diagnostic display only.
///
/// # Invariants
///
/// - `offset` is a 0-indexed byte offset from the start of the source string.
/// - `line` starts at 1 (first line of the file).
/// - `column` starts at 1 (first byte of the line), measured in bytes not
///   characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// Byte offset from the start of the source string (0-indexed).
    pub offset: u32,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (1-indexed, in bytes not characters).
    pub column: u32,
}

impl Position {
    /// Create a new position with explicit values.
    #[inline]
    pub fn new(offset: u32, line: u32, column: u32) -> Self {
        Self {
            offset,
            line,
            column,
        }
    }
}

// ---------------------------------------------------------------------------
// Scanner — character-level input with lookahead and position tracking
// ---------------------------------------------------------------------------

/// Character-level scanner for the BCC lexer.
///
/// Reads individual characters from a preprocessor-expanded source string,
/// provides bounded lookahead (via [`peek`], [`peek_nth`], and [`unget`]),
/// tracks byte offset / line / column, and normalizes line endings.
///
/// PUA-encoded code points (U+E080–U+E0FF) pass through transparently as
/// normal Unicode characters.
///
/// # Example
///
/// ```ignore
/// use bcc::frontend::lexer::scanner::Scanner;
///
/// let mut sc = Scanner::new("hello\nworld");
/// assert_eq!(sc.advance(), Some('h'));
/// assert_eq!(sc.peek(), Some('e'));
/// assert_eq!(sc.line(), 1);
/// ```
///
/// [`peek`]: Scanner::peek
/// [`peek_nth`]: Scanner::peek_nth
/// [`unget`]: Scanner::unget
pub struct Scanner<'src> {
    /// The full source string (PUA-encoded for non-UTF-8 files).
    source: &'src str,

    /// Peekable iterator over characters with their byte indices.
    chars: Peekable<CharIndices<'src>>,

    /// Current byte offset in the source string — position of the next
    /// character to be read (when the lookahead buffer is empty).
    offset: usize,

    /// Current line number (1-indexed).
    line: u32,

    /// Current column number (1-indexed, in bytes).
    column: u32,

    /// Lookahead buffer storing `(character, position_of_character)` pairs.
    ///
    /// Each entry's [`Position`] represents the scanner state immediately
    /// **before** consuming that character. Characters are pushed here by
    /// [`unget`] or pre-read by [`peek_nth`] when the lexer needs to look
    /// further ahead than what the underlying iterator provides.
    ///
    /// [`unget`]: Scanner::unget
    /// [`peek_nth`]: Scanner::peek_nth
    lookahead: Vec<(char, Position)>,
}

impl<'src> Scanner<'src> {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Create a new scanner for the given source string.
    ///
    /// The scanner starts at byte offset 0, line 1, column 1.
    pub fn new(source: &'src str) -> Self {
        Scanner {
            source,
            chars: source.char_indices().peekable(),
            offset: 0,
            line: 1,
            column: 1,
            lookahead: Vec::with_capacity(4),
        }
    }

    // ===================================================================
    // Core character operations
    // ===================================================================

    /// Advance to the next character, consuming it.
    ///
    /// Returns `Some(char)` with the next character (after line-ending
    /// normalization), or `None` at end-of-file.
    ///
    /// # Line-Ending Normalization
    ///
    /// - `\n` is returned as `\n` (1 byte consumed).
    /// - `\r\n` is returned as `\n` (2 bytes consumed).
    /// - `\r` alone is returned as `\n` (1 byte consumed).
    ///
    /// All other characters, including PUA code points (U+E080–U+E0FF),
    /// are returned as-is.
    pub fn advance(&mut self) -> Option<char> {
        // Priority 1: consume from the lookahead buffer.
        if !self.lookahead.is_empty() {
            let (ch, pos) = self.lookahead.remove(0);
            // Determine the actual byte length this character spans in the
            // original source. For normalized newlines this may be 1 or 2.
            let byte_len = self.source_byte_len(ch, pos.offset as usize);
            // Update scanner state to the position AFTER this character.
            self.offset = pos.offset as usize + byte_len;
            if ch == '\n' {
                self.line = pos.line + 1;
                self.column = 1;
            } else {
                self.line = pos.line;
                self.column = pos.column + byte_len as u32;
            }
            return Some(ch);
        }

        // Priority 2: read directly from the character iterator.
        let (byte_idx, raw_ch) = self.chars.next()?;

        match raw_ch {
            '\r' => {
                // Normalize \r\n (Windows) or standalone \r (old Mac) to \n.
                if let Some(&(_, '\n')) = self.chars.peek() {
                    self.chars.next(); // consume the '\n' half of \r\n
                    self.offset = byte_idx + 2;
                } else {
                    self.offset = byte_idx + 1;
                }
                self.line += 1;
                self.column = 1;
                Some('\n') // normalized
            }
            '\n' => {
                self.offset = byte_idx + 1;
                self.line += 1;
                self.column = 1;
                Some('\n')
            }
            ch => {
                let len = ch.len_utf8();
                self.offset = byte_idx + len;
                self.column += len as u32;
                Some(ch)
            }
        }
    }

    /// Peek at the next character without consuming it.
    ///
    /// Returns `Some(char)` with the next character (after line-ending
    /// normalization), or `None` at end-of-file.
    ///
    /// This method does **not** advance the scanner's position.
    pub fn peek(&mut self) -> Option<char> {
        // Check the lookahead buffer first.
        if let Some(&(ch, _)) = self.lookahead.first() {
            return Some(ch);
        }
        // Peek at the raw iterator with \r normalization.
        self.chars
            .peek()
            .map(|&(_, ch)| if ch == '\r' { '\n' } else { ch })
    }

    /// Look ahead `n` characters (0-indexed) without consuming.
    ///
    /// `peek_nth(0)` is equivalent to [`peek()`](Scanner::peek).
    /// `peek_nth(1)` returns the character after the next one, and so on.
    ///
    /// The lookahead buffer is filled as needed by pre-reading from the
    /// underlying iterator. All pre-read characters undergo line-ending
    /// normalization.
    pub fn peek_nth(&mut self, n: usize) -> Option<char> {
        // If already buffered, return immediately.
        if n < self.lookahead.len() {
            return Some(self.lookahead[n].0);
        }
        // Fill the lookahead up to n+1 entries.
        self.fill_lookahead(n + 1);
        self.lookahead.get(n).map(|&(ch, _)| ch)
    }

    /// Push a character back to the front of the scanner's input.
    ///
    /// The `pos` argument must be the [`Position`] that the scanner had
    /// **before** this character was consumed — i.e., the value returned by
    /// [`position()`](Scanner::position) immediately before the
    /// corresponding [`advance()`](Scanner::advance) call.
    ///
    /// # Usage
    ///
    /// Used when the lexer "over-reads" a character and needs to put it
    /// back for re-scanning.
    pub fn unget(&mut self, ch: char, pos: Position) {
        self.lookahead.insert(0, (ch, pos));
        // Restore the scanner's current-position state so that position(),
        // offset(), line(), column() all reflect this character's location
        // (the next character to be read).
        self.offset = pos.offset as usize;
        self.line = pos.line;
        self.column = pos.column;
    }

    // ===================================================================
    // Position query methods
    // ===================================================================

    /// Return the current scanner position.
    ///
    /// This is the position of the **next** character to be read — i.e.,
    /// what [`advance()`](Scanner::advance) would consume.
    #[inline]
    pub fn position(&self) -> Position {
        Position {
            offset: self.offset as u32,
            line: self.line,
            column: self.column,
        }
    }

    /// Return the current byte offset (shorthand for `position().offset`).
    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset as u32
    }

    /// Return the current line number (1-indexed).
    #[inline]
    pub fn line(&self) -> u32 {
        self.line
    }

    /// Return the current column number (1-indexed, in bytes).
    #[inline]
    pub fn column(&self) -> u32 {
        self.column
    }

    // ===================================================================
    // Utility methods
    // ===================================================================

    /// Check whether the scanner has reached end-of-file.
    ///
    /// Returns `true` when there are no more characters to read.
    #[inline]
    pub fn is_eof(&mut self) -> bool {
        self.peek().is_none()
    }

    /// Return the full source string.
    ///
    /// Useful for extracting substrings between known byte offsets.
    #[inline]
    pub fn source(&self) -> &'src str {
        self.source
    }

    /// Return a substring of the source between byte offsets `start`
    /// (inclusive) and `end` (exclusive).
    ///
    /// Both offsets must lie on valid UTF-8 character boundaries within
    /// the source string.
    ///
    /// # Panics
    ///
    /// Panics if `start > end`, if either offset exceeds the source
    /// length, or if an offset is not on a UTF-8 boundary.
    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> &'src str {
        &self.source[start..end]
    }

    /// Consume characters as long as `predicate` returns `true`.
    ///
    /// Stops at the first character for which `predicate` returns `false`,
    /// or at end-of-file.
    pub fn skip_while(&mut self, predicate: impl Fn(char) -> bool) {
        loop {
            match self.peek() {
                Some(ch) if predicate(ch) => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    /// If the next character equals `ch`, consume it and return `true`.
    /// Otherwise return `false` without consuming.
    ///
    /// Used for optional character matching (e.g., checking for `=`
    /// after `+` to form `+=`).
    pub fn consume_if(&mut self, ch: char) -> bool {
        if self.peek() == Some(ch) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// If the next character satisfies `pred`, consume and return
    /// `Some(ch)`. Otherwise return `None` without consuming.
    pub fn consume_if_pred(&mut self, pred: impl Fn(char) -> bool) -> Option<char> {
        match self.peek() {
            Some(ch) if pred(ch) => {
                self.advance();
                Some(ch)
            }
            _ => None,
        }
    }

    // ===================================================================
    // PUA utility
    // ===================================================================

    /// Check if a character is a PUA-encoded non-UTF-8 byte (U+E080–U+E0FF).
    ///
    /// Convenience wrapper around [`encoding::is_pua_encoded`]. The scanner
    /// passes PUA code points through transparently, but callers (such as
    /// the lexer or diagnostic system) may use this to identify characters
    /// that represent original non-UTF-8 source bytes.
    ///
    /// [`encoding::is_pua_encoded`]: crate::common::encoding::is_pua_encoded
    #[inline]
    pub fn is_pua_char(ch: char) -> bool {
        encoding::is_pua_encoded(ch)
    }

    // ===================================================================
    // Internal helpers
    // ===================================================================

    /// Fill the lookahead buffer until it contains at least `target_len`
    /// entries.
    ///
    /// Pre-reads characters from the underlying [`CharIndices`] iterator,
    /// computes their positions, and applies line-ending normalization.
    /// Stops early if the iterator is exhausted (EOF).
    fn fill_lookahead(&mut self, target_len: usize) {
        while self.lookahead.len() < target_len {
            // Compute the position this next character will occupy.
            let pos = self.next_lookahead_position();

            // Read the next raw character from the iterator.
            match self.chars.next() {
                Some((_, '\r')) => {
                    // Normalize \r\n or standalone \r to \n.
                    if let Some(&(_, '\n')) = self.chars.peek() {
                        self.chars.next(); // consume the '\n' of \r\n
                    }
                    self.lookahead.push(('\n', pos));
                }
                Some((_, ch)) => {
                    self.lookahead.push((ch, pos));
                }
                None => break, // EOF reached
            }
        }
    }

    /// Compute the [`Position`] for the next character to be appended to
    /// the lookahead buffer.
    ///
    /// If the buffer is empty, this equals the current scanner position.
    /// Otherwise it is the position immediately after the last buffered
    /// character.
    fn next_lookahead_position(&self) -> Position {
        match self.lookahead.last() {
            Some(&(ch, pos)) => {
                let byte_len = self.source_byte_len(ch, pos.offset as usize);
                if ch == '\n' {
                    Position {
                        offset: pos.offset + byte_len as u32,
                        line: pos.line + 1,
                        column: 1,
                    }
                } else {
                    Position {
                        offset: pos.offset + byte_len as u32,
                        line: pos.line,
                        column: pos.column + byte_len as u32,
                    }
                }
            }
            None => Position {
                offset: self.offset as u32,
                line: self.line,
                column: self.column,
            },
        }
    }

    /// Determine how many bytes a character occupies at a given source
    /// offset.
    ///
    /// For most characters this is simply `ch.len_utf8()`. For normalised
    /// newlines (stored as `'\n'` in the lookahead) we inspect the raw
    /// source bytes to distinguish between `\n` (1 byte), standalone `\r`
    /// (1 byte), and `\r\n` (2 bytes).
    #[inline]
    fn source_byte_len(&self, ch: char, at_offset: usize) -> usize {
        if ch == '\n' {
            let bytes = self.source.as_bytes();
            match bytes.get(at_offset) {
                Some(&b'\r') => {
                    // \r followed by \n is 2 bytes; standalone \r is 1.
                    if bytes.get(at_offset + 1) == Some(&b'\n') {
                        2
                    } else {
                        1
                    }
                }
                Some(&b'\n') => 1,
                // Defensive fallback — should not be reached in normal
                // operation.
                _ => 1,
            }
        } else {
            ch.len_utf8()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // Construction and empty input
    // ===================================================================

    #[test]
    fn test_new_scanner_empty_input() {
        let mut sc = Scanner::new("");
        assert!(sc.is_eof());
        assert_eq!(sc.advance(), None);
        assert_eq!(sc.peek(), None);
    }

    #[test]
    fn test_new_scanner_initial_position() {
        let sc = Scanner::new("abc");
        assert_eq!(sc.line(), 1);
        assert_eq!(sc.column(), 1);
        assert_eq!(sc.offset(), 0);
    }

    // ===================================================================
    // Advance through ASCII characters
    // ===================================================================

    #[test]
    fn test_advance_ascii_single_char() {
        let mut sc = Scanner::new("A");
        assert_eq!(sc.advance(), Some('A'));
        assert_eq!(sc.advance(), None);
    }

    #[test]
    fn test_advance_ascii_multiple_chars() {
        let mut sc = Scanner::new("abc");
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.advance(), Some('b'));
        assert_eq!(sc.advance(), Some('c'));
        assert_eq!(sc.advance(), None);
    }

    #[test]
    fn test_advance_updates_column() {
        let mut sc = Scanner::new("xyz");
        assert_eq!(sc.column(), 1);
        sc.advance(); // 'x'
        assert_eq!(sc.column(), 2);
        sc.advance(); // 'y'
        assert_eq!(sc.column(), 3);
        sc.advance(); // 'z'
        assert_eq!(sc.column(), 4);
    }

    #[test]
    fn test_advance_updates_offset() {
        let mut sc = Scanner::new("ab");
        assert_eq!(sc.offset(), 0);
        sc.advance(); // 'a' (1 byte)
        assert_eq!(sc.offset(), 1);
        sc.advance(); // 'b' (1 byte)
        assert_eq!(sc.offset(), 2);
    }

    // ===================================================================
    // Line-ending normalization
    // ===================================================================

    #[test]
    fn test_lf_normalization() {
        let mut sc = Scanner::new("a\nb");
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.advance(), Some('\n'));
        assert_eq!(sc.line(), 2);
        assert_eq!(sc.column(), 1);
        assert_eq!(sc.advance(), Some('b'));
    }

    #[test]
    fn test_cr_normalization() {
        let mut sc = Scanner::new("a\rb");
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.advance(), Some('\n')); // \r → \n
        assert_eq!(sc.line(), 2);
        assert_eq!(sc.column(), 1);
        assert_eq!(sc.advance(), Some('b'));
    }

    #[test]
    fn test_crlf_normalization() {
        let mut sc = Scanner::new("a\r\nb");
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.advance(), Some('\n')); // \r\n → \n (2 bytes consumed)
        assert_eq!(sc.line(), 2);
        assert_eq!(sc.column(), 1);
        assert_eq!(sc.advance(), Some('b'));
    }

    #[test]
    fn test_crlf_offset_accounts_for_two_bytes() {
        let mut sc = Scanner::new("x\r\ny");
        sc.advance(); // 'x'
        sc.advance(); // '\r\n' → '\n'
                      // After consuming 'x' (1 byte) and '\r\n' (2 bytes), offset should be 3.
        assert_eq!(sc.offset(), 3);
    }

    #[test]
    fn test_multiple_line_endings() {
        let mut sc = Scanner::new("a\n\n\nb");
        sc.advance(); // 'a'
        sc.advance(); // '\n'
        assert_eq!(sc.line(), 2);
        sc.advance(); // '\n'
        assert_eq!(sc.line(), 3);
        sc.advance(); // '\n'
        assert_eq!(sc.line(), 4);
        assert_eq!(sc.advance(), Some('b'));
    }

    // ===================================================================
    // Peek — non-consuming lookahead
    // ===================================================================

    #[test]
    fn test_peek_returns_next_without_consuming() {
        let mut sc = Scanner::new("ab");
        assert_eq!(sc.peek(), Some('a'));
        assert_eq!(sc.peek(), Some('a')); // still 'a'
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.peek(), Some('b'));
    }

    #[test]
    fn test_peek_at_eof() {
        let mut sc = Scanner::new("");
        assert_eq!(sc.peek(), None);
    }

    #[test]
    fn test_peek_normalizes_cr() {
        let mut sc = Scanner::new("\r");
        assert_eq!(sc.peek(), Some('\n')); // \r normalized to \n
    }

    // ===================================================================
    // peek_nth — multi-character lookahead
    // ===================================================================

    #[test]
    fn test_peek_nth_zero_same_as_peek() {
        let mut sc = Scanner::new("abc");
        assert_eq!(sc.peek_nth(0), sc.peek());
    }

    #[test]
    fn test_peek_nth_lookahead() {
        let mut sc = Scanner::new("abcd");
        assert_eq!(sc.peek_nth(0), Some('a'));
        assert_eq!(sc.peek_nth(1), Some('b'));
        assert_eq!(sc.peek_nth(2), Some('c'));
        assert_eq!(sc.peek_nth(3), Some('d'));
        assert_eq!(sc.peek_nth(4), None);
    }

    #[test]
    fn test_peek_nth_does_not_consume() {
        let mut sc = Scanner::new("xyz");
        sc.peek_nth(2);
        // First advance should still return 'x'
        assert_eq!(sc.advance(), Some('x'));
    }

    #[test]
    fn test_peek_nth_beyond_eof() {
        let mut sc = Scanner::new("a");
        assert_eq!(sc.peek_nth(0), Some('a'));
        assert_eq!(sc.peek_nth(1), None);
    }

    // ===================================================================
    // unget — push character back
    // ===================================================================

    #[test]
    fn test_unget_restores_character() {
        let mut sc = Scanner::new("ab");
        let pos = sc.position();
        let ch = sc.advance().unwrap();
        assert_eq!(ch, 'a');
        sc.unget(ch, pos);
        // After unget, the next advance should return 'a' again.
        assert_eq!(sc.advance(), Some('a'));
        assert_eq!(sc.advance(), Some('b'));
        assert_eq!(sc.advance(), None);
    }

    #[test]
    fn test_unget_restores_position() {
        let mut sc = Scanner::new("xy");
        let pos_before = sc.position();
        sc.advance(); // 'x'
        sc.unget('x', pos_before);
        assert_eq!(sc.position(), pos_before);
    }

    // ===================================================================
    // Position tracking — multi-line
    // ===================================================================

    #[test]
    fn test_position_tracking_multiline() {
        let mut sc = Scanner::new("ab\ncd\ne");
        // Line 1: a, b, \n
        sc.advance(); // 'a' → line 1, col 2
        sc.advance(); // 'b' → line 1, col 3
        sc.advance(); // '\n' → line 2, col 1
        assert_eq!(sc.line(), 2);
        assert_eq!(sc.column(), 1);

        // Line 2: c, d, \n
        sc.advance(); // 'c' → line 2, col 2
        assert_eq!(sc.column(), 2);
        sc.advance(); // 'd' → line 2, col 3
        sc.advance(); // '\n' → line 3, col 1
        assert_eq!(sc.line(), 3);

        // Line 3: e
        assert_eq!(sc.advance(), Some('e'));
    }

    #[test]
    fn test_position_method_returns_current() {
        let mut sc = Scanner::new("a\nb");
        let pos = sc.position();
        assert_eq!(pos.offset, 0);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);

        sc.advance(); // 'a'
        let pos = sc.position();
        assert_eq!(pos.offset, 1);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 2);
    }

    // ===================================================================
    // EOF detection
    // ===================================================================

    #[test]
    fn test_is_eof_empty_input() {
        let mut sc = Scanner::new("");
        assert!(sc.is_eof());
    }

    #[test]
    fn test_is_eof_after_consuming_all() {
        let mut sc = Scanner::new("x");
        assert!(!sc.is_eof());
        sc.advance();
        assert!(sc.is_eof());
    }

    // ===================================================================
    // Multi-byte UTF-8 handling
    // ===================================================================

    #[test]
    fn test_advance_multibyte_utf8() {
        let mut sc = Scanner::new("é"); // 2-byte UTF-8 (U+00E9)
        assert_eq!(sc.advance(), Some('é'));
        assert_eq!(sc.offset(), 2); // 2 bytes consumed
        assert!(sc.is_eof());
    }

    #[test]
    fn test_advance_three_byte_utf8() {
        let mut sc = Scanner::new("€"); // 3-byte UTF-8 (U+20AC)
        assert_eq!(sc.advance(), Some('€'));
        assert_eq!(sc.offset(), 3);
    }

    #[test]
    fn test_advance_four_byte_utf8() {
        let mut sc = Scanner::new("🦀"); // 4-byte UTF-8 (U+1F980)
        assert_eq!(sc.advance(), Some('🦀'));
        assert_eq!(sc.offset(), 4);
    }

    #[test]
    fn test_column_counts_bytes_not_chars() {
        let mut sc = Scanner::new("aé"); // 'a' (1 byte) + 'é' (2 bytes)
        sc.advance(); // 'a'
        assert_eq!(sc.column(), 2); // column 2 (after 1 byte)
        sc.advance(); // 'é'
        assert_eq!(sc.column(), 4); // column 4 (after 1 + 2 = 3 bytes)
    }

    // ===================================================================
    // PUA character handling
    // ===================================================================

    #[test]
    fn test_advance_pua_encoded_character() {
        // U+E080 is PUA-encoded byte 0x80, encoded in UTF-8 as 3 bytes: EE 82 80
        let pua_char = char::from_u32(0xE080).unwrap();
        let input = String::from(pua_char);
        let mut sc = Scanner::new(&input);
        assert_eq!(sc.advance(), Some(pua_char));
        assert_eq!(sc.offset(), 3); // 3 UTF-8 bytes
    }

    #[test]
    fn test_is_pua_char_true() {
        let pua = char::from_u32(0xE080).unwrap();
        assert!(Scanner::is_pua_char(pua));
    }

    #[test]
    fn test_is_pua_char_false_for_ascii() {
        assert!(!Scanner::is_pua_char('a'));
        assert!(!Scanner::is_pua_char('\n'));
    }

    #[test]
    fn test_is_pua_char_boundary() {
        let low = char::from_u32(0xE080).unwrap();
        let high = char::from_u32(0xE0FF).unwrap();
        assert!(Scanner::is_pua_char(low));
        assert!(Scanner::is_pua_char(high));

        // Just below and above the range
        let below = char::from_u32(0xE07F).unwrap();
        let above = char::from_u32(0xE100).unwrap();
        assert!(!Scanner::is_pua_char(below));
        assert!(!Scanner::is_pua_char(above));
    }

    // ===================================================================
    // Utility methods — skip_while, consume_if, consume_if_pred
    // ===================================================================

    #[test]
    fn test_skip_while_digits() {
        let mut sc = Scanner::new("123abc");
        sc.skip_while(|c| c.is_ascii_digit());
        assert_eq!(sc.advance(), Some('a'));
    }

    #[test]
    fn test_skip_while_no_match() {
        let mut sc = Scanner::new("abc");
        sc.skip_while(|c| c.is_ascii_digit());
        assert_eq!(sc.advance(), Some('a')); // nothing skipped
    }

    #[test]
    fn test_skip_while_all_match() {
        let mut sc = Scanner::new("999");
        sc.skip_while(|c| c.is_ascii_digit());
        assert!(sc.is_eof());
    }

    #[test]
    fn test_consume_if_matching() {
        let mut sc = Scanner::new("+=");
        assert!(sc.consume_if('+'));
        assert_eq!(sc.advance(), Some('='));
    }

    #[test]
    fn test_consume_if_not_matching() {
        let mut sc = Scanner::new("ab");
        assert!(!sc.consume_if('x'));
        assert_eq!(sc.advance(), Some('a')); // 'a' not consumed
    }

    #[test]
    fn test_consume_if_pred_matching() {
        let mut sc = Scanner::new("abc");
        let result = sc.consume_if_pred(|c| c == 'a');
        assert_eq!(result, Some('a'));
        assert_eq!(sc.advance(), Some('b'));
    }

    #[test]
    fn test_consume_if_pred_not_matching() {
        let mut sc = Scanner::new("abc");
        let result = sc.consume_if_pred(|c| c == 'z');
        assert_eq!(result, None);
        assert_eq!(sc.advance(), Some('a')); // not consumed
    }

    // ===================================================================
    // source() and slice()
    // ===================================================================

    #[test]
    fn test_source_returns_full_input() {
        let sc = Scanner::new("hello world");
        assert_eq!(sc.source(), "hello world");
    }

    #[test]
    fn test_slice_extracts_substring() {
        let sc = Scanner::new("hello world");
        assert_eq!(sc.slice(0, 5), "hello");
        assert_eq!(sc.slice(6, 11), "world");
    }

    // ===================================================================
    // Position::new
    // ===================================================================

    #[test]
    fn test_position_new() {
        let pos = Position::new(10, 2, 5);
        assert_eq!(pos.offset, 10);
        assert_eq!(pos.line, 2);
        assert_eq!(pos.column, 5);
    }

    #[test]
    fn test_position_default() {
        let pos = Position::default();
        assert_eq!(pos.offset, 0);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.column, 0);
    }

    #[test]
    fn test_position_equality() {
        let a = Position::new(0, 1, 1);
        let b = Position::new(0, 1, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_position_clone_copy() {
        let pos = Position::new(5, 3, 2);
        let copied = pos;
        let cloned = pos.clone();
        assert_eq!(pos, copied);
        assert_eq!(pos, cloned);
    }
}
