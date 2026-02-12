//! Character-level scanner module for PUA-aware UTF-8 source reading.
//!
//! Provides the foundational character-reading layer for the BCC lexer, wrapping
//! a PUA-encoded source string (already processed by
//! [`crate::common::encoding::read_source_file`]) and offering:
//!
//! - **Character-by-character reading** with O(1) lookahead via a pre-computed
//!   character index array
//! - **Byte-offset position tracking** for constructing `Span` values used
//!   throughout the compiler for source location metadata
//! - **Line/column tracking** for diagnostic messages, handling Unix (`\n`),
//!   Windows (`\r\n`), and legacy Mac (`\r`) line endings
//! - **Source slice extraction** for capturing identifier text, number literal
//!   text, and other lexeme strings
//! - **Character classification helpers** (digits, hex digits, identifier chars,
//!   whitespace) that explicitly exclude PUA code points (U+E080–U+E0FF) from
//!   being misidentified as identifier characters or whitespace
//!
//! # PUA Transparency
//!
//! PUA code points representing non-UTF-8 source bytes appear as regular `char`
//! values to [`Scanner::peek`] and [`Scanner::advance`]. The scanner does not
//! interpret or transform them — they flow through unchanged. However, the
//! classification helpers ([`Scanner::is_identifier_start`],
//! [`Scanner::is_identifier_continue`], [`Scanner::is_whitespace`]) explicitly
//! reject PUA code points so they are treated as opaque/inert characters,
//! preserving byte-exact round-tripping fidelity per Section 0.7.9.
//!
//! # Performance
//!
//! The constructor pre-computes a `Vec<(usize, char)>` of all character
//! boundaries, enabling O(1) random-access `peek_ahead` without redundant
//! UTF-8 re-decoding. This is critical for processing very large files
//! (Linux kernel sources can exceed 10K lines after preprocessing).

use crate::common::encoding::is_pua_encoded;

// ---------------------------------------------------------------------------
// ScannerMark — saved position for backtracking
// ---------------------------------------------------------------------------

/// A saved scanner position that can be restored via [`Scanner::reset_to`].
///
/// Captures the complete scanner state required for exact position restoration:
/// byte offset, character index, line number, and column number. Used for
/// speculative parsing (e.g., checking if an identifier prefix like `L` is
/// followed by a quote for string literal detection) and for newline-crossing
/// detection across token boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScannerMark {
    /// Byte offset into the source string at the time of the mark.
    pub pos: usize,
    /// Character index into the pre-computed character array.
    pub index: usize,
    /// Line number at the time of the mark (1-indexed).
    pub line: u32,
    /// Column number at the time of the mark (1-indexed).
    pub column: u32,
}

// ---------------------------------------------------------------------------
// C11 Annex D — Unicode identifier character classification
// ---------------------------------------------------------------------------

/// Check if a Unicode code point is allowed in C11 identifiers (Annex D.1).
///
/// Returns `true` for non-ASCII code points that C11 permits in identifiers.
/// This covers a wide range of international scripts and symbols as specified
/// in ISO/IEC 9899:2011 Annex D.1. ASCII letters, digits, and underscore are
/// handled separately by the caller; this function only checks extended Unicode
/// ranges.
///
/// The PUA range U+E000–U+F8FF is NOT included in Annex D.1 (there is a gap
/// between 0xD7FF and 0xF900), but we still rely on the explicit
/// [`is_pua_encoded`] check in the public API for defense-in-depth.
#[inline]
fn is_c11_annex_d1_char(cp: u32) -> bool {
    // Fast rejection for ASCII range (handled by caller) and low Latin-1
    if cp < 0x00A8 {
        return false;
    }

    matches!(cp,
        // Latin-1 Supplement selected code points
        0x00A8 | 0x00AA | 0x00AD | 0x00AF |
        0x00B2..=0x00B5 | 0x00B7..=0x00BA | 0x00BC..=0x00BE |
        0x00C0..=0x00D6 | 0x00D8..=0x00F6 | 0x00F8..=0x00FF |
        // Latin Extended, IPA, Spacing Modifiers, Combining Marks, Greek,
        // Cyrillic, Armenian, Hebrew, Arabic, Syriac, Thaana, Devanagari,
        // Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
        // Malayalam, Sinhala, Thai, Lao, Tibetan, Myanmar, Georgian,
        // Hangul Jamo, Ethiopic, Cherokee, Canadian Aboriginal, Ogham,
        // Runic, Tagalog, Hanunoo, Buhid, Tagbanwa, Khmer, Mongolian
        0x0100..=0x167F |
        // Limbu through Ol Chiki (skip 0x1680 OGHAM SPACE MARK)
        0x1681..=0x180D |
        // Mongolian continuation through Phonetic Extensions Supplement
        0x180F..=0x1FFF |
        // General Punctuation (selected)
        0x200B..=0x200D | 0x202A..=0x202E | 0x203F..=0x2040 |
        0x2054 |
        0x2060..=0x206F |
        // Superscripts, Subscripts, Currency, Letterlike, Number Forms,
        // Arrows, Math Operators, Misc Technical
        0x2070..=0x218F |
        // Enclosed Alphanumerics, Dingbats
        0x2460..=0x24FF | 0x2776..=0x2793 |
        // Glagolitic, Latin Extended-C, Coptic, Georgian Supplement,
        // Tifinagh, Ethiopic Extended, CJK Radicals
        0x2C00..=0x2DFF | 0x2E80..=0x2FFF |
        // CJK Symbols, Hiragana, Katakana, Bopomofo, Hangul Compat,
        // Kanbun, Bopomofo Extended, CJK Strokes, CJK Unified to
        // Hangul Syllables (excluding surrogates)
        0x3004..=0x3007 | 0x3021..=0x302F | 0x3031..=0xD7FF |
        // CJK Compatibility Ideographs through Arabic Presentation Forms-A
        0xF900..=0xFD3D | 0xFD40..=0xFDCF |
        // Arabic Presentation Forms-A continued, Variation Selectors,
        // CJK Compatibility Forms, Small Form Variants, Arabic
        // Presentation Forms-B, Half/Fullwidth Forms, Specials
        0xFDF0..=0xFE44 | 0xFE47..=0xFFFD |
        // Supplementary planes: Linear B through Supplementary Private Use
        // (each plane allows 0x0000 through 0xFFFD, excluding 0xFFFE/0xFFFF)
        0x10000..=0x1FFFD | 0x20000..=0x2FFFD | 0x30000..=0x3FFFD |
        0x40000..=0x4FFFD | 0x50000..=0x5FFFD | 0x60000..=0x6FFFD |
        0x70000..=0x7FFFD | 0x80000..=0x8FFFD | 0x90000..=0x9FFFD |
        0xA0000..=0xAFFFD | 0xB0000..=0xBFFFD | 0xC0000..=0xCFFFD |
        0xD0000..=0xDFFFD | 0xE0000..=0xEFFFD
    )
}

/// Check if a Unicode code point is disallowed at the start of a C11 identifier.
///
/// C11 Annex D.2 lists combining characters and other code points that may
/// appear within identifiers but not as the first character. These are
/// primarily combining diacritical marks (e.g., accents, cedillas) and
/// variation selectors.
#[inline]
fn is_c11_annex_d2_disallowed_initial(cp: u32) -> bool {
    matches!(cp,
        // Combining Diacritical Marks
        0x0300..=0x036F |
        // Combining Diacritical Marks Supplement
        0x1DC0..=0x1DFF |
        // Combining Diacritical Marks for Symbols
        0x20D0..=0x20FF |
        // Combining Half Marks
        0xFE20..=0xFE2F
    )
}

// ---------------------------------------------------------------------------
// Scanner — PUA-aware character-level source reader
// ---------------------------------------------------------------------------

/// Character-level scanner with PUA-aware UTF-8 decoding and lookahead.
///
/// This is the foundational character-reading layer that all lexer sub-modules
/// ([`super::mod`], [`super::number_literal`], [`super::string_literal`]) build
/// upon for character-by-character source consumption.
///
/// # Construction
///
/// ```ignore
/// use crate::frontend::lexer::scanner::Scanner;
///
/// let source = "int main() { return 0; }\n";
/// let mut scanner = Scanner::new(source, /* file_id */ 0);
/// assert_eq!(scanner.peek(), Some('i'));
/// ```
///
/// # Byte-Offset Tracking
///
/// Every character consumed through [`advance`](Scanner::advance) updates the
/// internal byte offset, which can be queried via [`byte_offset`](Scanner::byte_offset).
/// This byte offset is used by the lexer to construct `Span { file_id, start, end }`
/// values attached to each token for precise source location reporting.
pub struct Scanner<'src> {
    /// The PUA-encoded source text (already processed by
    /// `crate::common::encoding::read_source_file`).
    source: &'src str,

    /// Pre-decoded `(byte_offset, char)` pairs for O(1) indexed access.
    ///
    /// Constructed once in [`Scanner::new`] from `source.char_indices()`.
    /// This avoids repeated UTF-8 decoding during `peek_ahead` calls and
    /// provides efficient random access into the character stream.
    chars: Vec<(usize, char)>,

    /// Current byte offset into `source`.
    ///
    /// Always points to the first byte of the character at `chars[index]`,
    /// or equals `source.len()` when at EOF.
    pos: usize,

    /// Current character index into `chars`.
    ///
    /// Invariant: `0 <= index <= chars.len()`. When `index == chars.len()`,
    /// the scanner is at EOF.
    index: usize,

    /// Current line number (1-indexed).
    ///
    /// Incremented on `\n`, on standalone `\r`, and on `\r` not followed by
    /// `\n`. The `\r\n` sequence counts as a single line ending.
    line: u32,

    /// Current column number (1-indexed).
    ///
    /// Reset to 1 on each line break. Incremented by 1 for each character
    /// consumed (including multi-byte UTF-8 characters, which each count as
    /// one column — consistent with GCC behavior).
    column: u32,

    /// File ID of the source file being scanned.
    ///
    /// Passed through to `Span` construction by the lexer; identifies which
    /// source file a token originated from in multi-file compilation.
    file_id: u32,
}

impl<'src> Scanner<'src> {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// Create a new scanner over the given PUA-encoded source text.
    ///
    /// The source string should already have been processed by
    /// `crate::common::encoding::read_source_file`, ensuring that any non-UTF-8
    /// bytes are represented as PUA code points (U+E080–U+E0FF).
    ///
    /// Pre-computes a character index array for O(1) lookahead access.
    ///
    /// # Arguments
    ///
    /// * `source` — The PUA-encoded source text to scan.
    /// * `file_id` — Identifier for the source file (embedded in Spans).
    pub fn new(source: &'src str, file_id: u32) -> Self {
        // Pre-compute all character boundaries for O(1) random access.
        // char_indices() yields (byte_offset, char) pairs by decoding UTF-8 once.
        let chars: Vec<(usize, char)> = source.char_indices().collect();

        Scanner {
            source,
            chars,
            pos: 0,
            index: 0,
            line: 1,
            column: 1,
            file_id,
        }
    }

    // -----------------------------------------------------------------------
    // Character Reading (PUA-Transparent)
    // -----------------------------------------------------------------------

    /// Return the current character without advancing the scanner.
    ///
    /// Returns `None` at end-of-file. PUA code points (U+E080–U+E0FF) appear
    /// as regular `char` values — the scanner does not interpret or filter them.
    #[inline]
    pub fn peek(&self) -> Option<char> {
        self.chars.get(self.index).map(|&(_, ch)| ch)
    }

    /// Lookahead `n` characters from the current position without consuming.
    ///
    /// `peek_ahead(0)` is equivalent to `peek()`. Returns `None` if fewer than
    /// `n + 1` characters remain in the source.
    ///
    /// This is an O(1) operation thanks to the pre-computed character array.
    #[inline]
    pub fn peek_ahead(&self, n: usize) -> Option<char> {
        self.chars.get(self.index + n).map(|&(_, ch)| ch)
    }

    /// Consume and return the current character, advancing all position state.
    ///
    /// Updates byte offset, character index, line number, and column number.
    /// Returns `None` at end-of-file (no state change in that case).
    ///
    /// # Line-Ending Handling
    ///
    /// - `\n` — increments line, resets column to 1
    /// - `\r\n` — treated as a single newline: `\r` advances position without
    ///   incrementing line (the subsequent `\n` will do so)
    /// - `\r` alone — increments line, resets column to 1
    pub fn advance(&mut self) -> Option<char> {
        if self.index >= self.chars.len() {
            return None;
        }

        let (_, ch) = self.chars[self.index];
        self.index += 1;

        // Update byte offset to the start of the next character, or to
        // source.len() if we just consumed the last character.
        self.pos = if self.index < self.chars.len() {
            self.chars[self.index].0
        } else {
            self.source.len()
        };

        // Update line/column tracking based on the consumed character.
        match ch {
            '\n' => {
                self.line += 1;
                self.column = 1;
            }
            '\r' => {
                // Check if this \r is part of a \r\n pair (Windows line ending).
                // If the next character is \n, we let \n handle the line increment
                // to avoid double-counting. The column value here is transient —
                // the subsequent \n will reset it to 1 anyway.
                if self.peek() == Some('\n') {
                    // \r before \n: don't increment line yet; \n will handle it.
                    // Column is irrelevant since \n will reset it.
                    self.column += 1;
                } else {
                    // Standalone \r (legacy Mac line ending): treat as newline.
                    self.line += 1;
                    self.column = 1;
                }
            }
            _ => {
                self.column += 1;
            }
        }

        Some(ch)
    }

    /// Consume the current character only if it matches `expected`.
    ///
    /// Returns `true` if the character was consumed, `false` otherwise (no
    /// state change on mismatch or EOF).
    #[inline]
    pub fn advance_if(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume characters while `predicate` returns `true`.
    ///
    /// Returns a source slice covering all consumed characters (from the
    /// byte position at the start of the call to the byte position after the
    /// last consumed character). Returns an empty slice if the predicate is
    /// immediately false or the scanner is at EOF.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let digits = scanner.advance_while(|ch| Scanner::is_digit(ch));
    /// // `digits` is e.g. "12345"
    /// ```
    pub fn advance_while(&mut self, predicate: impl Fn(char) -> bool) -> &'src str {
        let start_pos = self.pos;
        while let Some(ch) = self.peek() {
            if !predicate(ch) {
                break;
            }
            self.advance();
        }
        &self.source[start_pos..self.pos]
    }

    // -----------------------------------------------------------------------
    // Position Tracking
    // -----------------------------------------------------------------------

    /// Return the current byte offset into the source string.
    ///
    /// This value is used as `Span.start` or `Span.end` by the lexer. It
    /// always falls on a valid UTF-8 character boundary.
    #[inline]
    pub fn byte_offset(&self) -> u32 {
        self.pos as u32
    }

    /// Return the current line number (1-indexed).
    #[inline]
    pub fn current_line(&self) -> u32 {
        self.line
    }

    /// Return the current column number (1-indexed).
    #[inline]
    pub fn current_column(&self) -> u32 {
        self.column
    }

    /// Save the current scanner position for later restoration.
    ///
    /// The returned [`ScannerMark`] can be passed to [`reset_to`](Scanner::reset_to)
    /// to restore the scanner to this exact position, including byte offset,
    /// character index, line, and column. Used for speculative parsing and
    /// backtracking in the lexer.
    #[inline]
    pub fn mark(&self) -> ScannerMark {
        ScannerMark {
            pos: self.pos,
            index: self.index,
            line: self.line,
            column: self.column,
        }
    }

    /// Restore the scanner to a previously saved position.
    ///
    /// All state (byte offset, character index, line, column) is restored
    /// to the values captured by [`mark`](Scanner::mark). This enables
    /// backtracking after speculative lookahead (e.g., checking if an `L`
    /// identifier prefix is followed by a quote for wide string detection).
    #[inline]
    pub fn reset_to(&mut self, mark: ScannerMark) {
        self.pos = mark.pos;
        self.index = mark.index;
        self.line = mark.line;
        self.column = mark.column;
    }

    // -----------------------------------------------------------------------
    // Source Slice Extraction
    // -----------------------------------------------------------------------

    /// Extract a source text slice from `start` byte offset to the current position.
    ///
    /// Used by the lexer to capture the textual representation of tokens
    /// (identifiers, numbers, operators) for interning or diagnostic display.
    ///
    /// # Panics
    ///
    /// Panics if `start` does not fall on a valid UTF-8 character boundary
    /// within the source string. In practice this never occurs because `start`
    /// is always obtained from a prior [`byte_offset`](Scanner::byte_offset) call.
    #[inline]
    pub fn slice_from(&self, start: u32) -> &'src str {
        &self.source[start as usize..self.pos]
    }

    /// Extract a source text slice between two byte offsets.
    ///
    /// Returns `&source[start..end]`. Both `start` and `end` must be valid
    /// UTF-8 character boundaries within the source string.
    ///
    /// # Panics
    ///
    /// Panics if either offset is out of bounds or not on a character boundary.
    #[inline]
    pub fn slice_range(&self, start: u32, end: u32) -> &'src str {
        &self.source[start as usize..end as usize]
    }

    // -----------------------------------------------------------------------
    // Character Classification Helpers
    // -----------------------------------------------------------------------

    /// Check whether the scanner has reached end-of-file.
    #[inline]
    pub fn is_at_end(&self) -> bool {
        self.index >= self.chars.len()
    }

    /// Check if a character is an ASCII decimal digit (`0`–`9`).
    #[inline]
    pub fn is_digit(ch: char) -> bool {
        ch.is_ascii_digit()
    }

    /// Check if a character is an ASCII hexadecimal digit (`0`–`9`, `a`–`f`, `A`–`F`).
    #[inline]
    pub fn is_hex_digit(ch: char) -> bool {
        ch.is_ascii_hexdigit()
    }

    /// Check if a character is an ASCII octal digit (`0`–`7`).
    #[inline]
    pub fn is_octal_digit(ch: char) -> bool {
        matches!(ch, '0'..='7')
    }

    /// Check if a character can start a C11 identifier.
    ///
    /// Allowed: ASCII letters (`a`–`z`, `A`–`Z`), underscore (`_`), and
    /// Unicode code points permitted by C11 Annex D.1 (excluding those
    /// disallowed initially by Annex D.2, which are combining marks).
    ///
    /// **PUA code points (U+E080–U+E0FF) are explicitly excluded** even though
    /// some fall within Annex D.1 ranges in supplementary planes. This ensures
    /// PUA-encoded non-UTF-8 bytes are treated as opaque, preserving byte-exact
    /// round-tripping fidelity per Section 0.7.9.
    #[inline]
    pub fn is_identifier_start(ch: char) -> bool {
        // Fast path: ASCII letters and underscore cover 99%+ of real-world C code
        if matches!(ch, 'a'..='z' | 'A'..='Z' | '_') {
            return true;
        }

        // Reject PUA code points before checking extended Unicode ranges.
        // The BMP PUA range (U+E000–U+F8FF) is NOT in C11 Annex D.1 (there is
        // a gap from 0xD800 to 0xF8FF), but we check explicitly for safety.
        if is_pua_encoded(ch) {
            return false;
        }

        // Extended Unicode identifier start: Annex D.1 minus Annex D.2.
        let cp = ch as u32;
        is_c11_annex_d1_char(cp) && !is_c11_annex_d2_disallowed_initial(cp)
    }

    /// Check if a character can continue a C11 identifier.
    ///
    /// Allowed: everything in [`is_identifier_start`](Scanner::is_identifier_start)
    /// plus ASCII digits (`0`–`9`) and C11 Annex D.2 combining marks (which are
    /// allowed within identifiers but not at the start).
    ///
    /// **PUA code points (U+E080–U+E0FF) are explicitly excluded.**
    #[inline]
    pub fn is_identifier_continue(ch: char) -> bool {
        // Fast path: ASCII letters, digits, and underscore
        if matches!(ch, 'a'..='z' | 'A'..='Z' | '_' | '0'..='9') {
            return true;
        }

        // Reject PUA code points
        if is_pua_encoded(ch) {
            return false;
        }

        // Extended Unicode identifier continuation: full Annex D.1 range
        // (includes combining marks from Annex D.2 that are not allowed at start)
        is_c11_annex_d1_char(ch as u32)
    }

    /// Check if a character is C whitespace.
    ///
    /// Matches: space (`0x20`), horizontal tab (`0x09`), vertical tab (`0x0B`),
    /// form feed (`0x0C`), carriage return (`0x0D`), and newline (`0x0A`).
    ///
    /// PUA code points are **not** whitespace — they are opaque bytes that
    /// must flow through unchanged.
    #[inline]
    pub fn is_whitespace(ch: char) -> bool {
        matches!(ch, ' ' | '\t' | '\x0B' | '\x0C' | '\r' | '\n')
    }

    // -----------------------------------------------------------------------
    // Newline Tracking
    // -----------------------------------------------------------------------

    /// Check if a newline was crossed between a saved mark and the current position.
    ///
    /// Returns `true` if the current line number is greater than the line number
    /// recorded in `mark`. This is critical for the preprocessor to detect
    /// directive boundaries, as preprocessor directives are line-delimited.
    #[inline]
    pub fn saw_newline_since(&self, mark: &ScannerMark) -> bool {
        self.line > mark.line
    }

    // -----------------------------------------------------------------------
    // File ID accessor
    // -----------------------------------------------------------------------

    /// Return the file ID of the source file being scanned.
    ///
    /// This value is embedded in every `Span` constructed by the lexer to
    /// identify which source file a token originated from.
    #[inline]
    pub fn file_id(&self) -> u32 {
        self.file_id
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Constructor and basic state
    // =======================================================================

    #[test]
    fn test_new_empty_source() {
        let scanner = Scanner::new("", 0);
        assert!(scanner.is_at_end());
        assert_eq!(scanner.peek(), None);
        assert_eq!(scanner.byte_offset(), 0);
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.current_column(), 1);
        assert_eq!(scanner.file_id(), 0);
    }

    #[test]
    fn test_new_with_content() {
        let scanner = Scanner::new("abc", 42);
        assert!(!scanner.is_at_end());
        assert_eq!(scanner.peek(), Some('a'));
        assert_eq!(scanner.byte_offset(), 0);
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.current_column(), 1);
        assert_eq!(scanner.file_id(), 42);
    }

    // =======================================================================
    // peek and peek_ahead
    // =======================================================================

    #[test]
    fn test_peek_returns_current_char() {
        let scanner = Scanner::new("xyz", 0);
        assert_eq!(scanner.peek(), Some('x'));
    }

    #[test]
    fn test_peek_at_eof() {
        let scanner = Scanner::new("", 0);
        assert_eq!(scanner.peek(), None);
    }

    #[test]
    fn test_peek_ahead_zero_equals_peek() {
        let scanner = Scanner::new("hello", 0);
        assert_eq!(scanner.peek_ahead(0), scanner.peek());
    }

    #[test]
    fn test_peek_ahead_multiple() {
        let scanner = Scanner::new("abcde", 0);
        assert_eq!(scanner.peek_ahead(0), Some('a'));
        assert_eq!(scanner.peek_ahead(1), Some('b'));
        assert_eq!(scanner.peek_ahead(2), Some('c'));
        assert_eq!(scanner.peek_ahead(3), Some('d'));
        assert_eq!(scanner.peek_ahead(4), Some('e'));
        assert_eq!(scanner.peek_ahead(5), None);
    }

    #[test]
    fn test_peek_ahead_beyond_end() {
        let scanner = Scanner::new("ab", 0);
        assert_eq!(scanner.peek_ahead(10), None);
    }

    // =======================================================================
    // advance
    // =======================================================================

    #[test]
    fn test_advance_returns_chars_in_order() {
        let mut scanner = Scanner::new("abc", 0);
        assert_eq!(scanner.advance(), Some('a'));
        assert_eq!(scanner.advance(), Some('b'));
        assert_eq!(scanner.advance(), Some('c'));
        assert_eq!(scanner.advance(), None);
    }

    #[test]
    fn test_advance_updates_byte_offset_ascii() {
        let mut scanner = Scanner::new("ab", 0);
        assert_eq!(scanner.byte_offset(), 0);
        scanner.advance(); // consume 'a'
        assert_eq!(scanner.byte_offset(), 1);
        scanner.advance(); // consume 'b'
        assert_eq!(scanner.byte_offset(), 2);
    }

    #[test]
    fn test_advance_updates_byte_offset_multibyte() {
        // 'é' is 2 bytes in UTF-8 (0xC3 0xA9)
        let mut scanner = Scanner::new("éa", 0);
        assert_eq!(scanner.byte_offset(), 0);
        scanner.advance(); // consume 'é' (2 bytes)
        assert_eq!(scanner.byte_offset(), 2);
        scanner.advance(); // consume 'a' (1 byte)
        assert_eq!(scanner.byte_offset(), 3);
    }

    #[test]
    fn test_advance_updates_column() {
        let mut scanner = Scanner::new("abc", 0);
        assert_eq!(scanner.current_column(), 1);
        scanner.advance();
        assert_eq!(scanner.current_column(), 2);
        scanner.advance();
        assert_eq!(scanner.current_column(), 3);
        scanner.advance();
        assert_eq!(scanner.current_column(), 4);
    }

    #[test]
    fn test_advance_at_eof_returns_none() {
        let mut scanner = Scanner::new("", 0);
        assert_eq!(scanner.advance(), None);
        // State should not change
        assert_eq!(scanner.byte_offset(), 0);
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.current_column(), 1);
    }

    // =======================================================================
    // advance_if
    // =======================================================================

    #[test]
    fn test_advance_if_match() {
        let mut scanner = Scanner::new("ab", 0);
        assert!(scanner.advance_if('a'));
        assert_eq!(scanner.peek(), Some('b'));
    }

    #[test]
    fn test_advance_if_no_match() {
        let mut scanner = Scanner::new("ab", 0);
        assert!(!scanner.advance_if('x'));
        assert_eq!(scanner.peek(), Some('a'));
    }

    #[test]
    fn test_advance_if_at_eof() {
        let mut scanner = Scanner::new("", 0);
        assert!(!scanner.advance_if('a'));
    }

    // =======================================================================
    // advance_while
    // =======================================================================

    #[test]
    fn test_advance_while_digits() {
        let mut scanner = Scanner::new("123abc", 0);
        let digits = scanner.advance_while(|ch| Scanner::is_digit(ch));
        assert_eq!(digits, "123");
        assert_eq!(scanner.peek(), Some('a'));
    }

    #[test]
    fn test_advance_while_no_match() {
        let mut scanner = Scanner::new("abc", 0);
        let result = scanner.advance_while(|ch| Scanner::is_digit(ch));
        assert_eq!(result, "");
        assert_eq!(scanner.peek(), Some('a'));
    }

    #[test]
    fn test_advance_while_consumes_all() {
        let mut scanner = Scanner::new("12345", 0);
        let result = scanner.advance_while(|ch| Scanner::is_digit(ch));
        assert_eq!(result, "12345");
        assert!(scanner.is_at_end());
    }

    #[test]
    fn test_advance_while_at_eof() {
        let mut scanner = Scanner::new("", 0);
        let result = scanner.advance_while(|_| true);
        assert_eq!(result, "");
    }

    // =======================================================================
    // Line tracking — Unix newlines (\n)
    // =======================================================================

    #[test]
    fn test_newline_increments_line() {
        let mut scanner = Scanner::new("a\nb", 0);
        scanner.advance(); // 'a'
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.current_column(), 2);
        scanner.advance(); // '\n'
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 1);
        scanner.advance(); // 'b'
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 2);
    }

    #[test]
    fn test_multiple_newlines() {
        let mut scanner = Scanner::new("\n\n\n", 0);
        scanner.advance();
        assert_eq!(scanner.current_line(), 2);
        scanner.advance();
        assert_eq!(scanner.current_line(), 3);
        scanner.advance();
        assert_eq!(scanner.current_line(), 4);
    }

    // =======================================================================
    // Line tracking — Windows newlines (\r\n)
    // =======================================================================

    #[test]
    fn test_crlf_single_newline() {
        let mut scanner = Scanner::new("a\r\nb", 0);
        scanner.advance(); // 'a' → line 1, col 2
        assert_eq!(scanner.current_line(), 1);
        scanner.advance(); // '\r' → does NOT increment line (next is '\n')
        assert_eq!(scanner.current_line(), 1);
        scanner.advance(); // '\n' → line 2, col 1
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 1);
        scanner.advance(); // 'b' → line 2, col 2
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 2);
    }

    #[test]
    fn test_multiple_crlf() {
        let mut scanner = Scanner::new("\r\n\r\n", 0);
        scanner.advance(); // '\r'
        scanner.advance(); // '\n' → line 2
        assert_eq!(scanner.current_line(), 2);
        scanner.advance(); // '\r'
        scanner.advance(); // '\n' → line 3
        assert_eq!(scanner.current_line(), 3);
    }

    // =======================================================================
    // Line tracking — Legacy Mac newlines (\r alone)
    // =======================================================================

    #[test]
    fn test_standalone_cr_newline() {
        let mut scanner = Scanner::new("a\rb", 0);
        scanner.advance(); // 'a'
        assert_eq!(scanner.current_line(), 1);
        scanner.advance(); // '\r' (standalone) → line 2
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 1);
        scanner.advance(); // 'b'
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 2);
    }

    #[test]
    fn test_cr_at_end_of_file() {
        let mut scanner = Scanner::new("a\r", 0);
        scanner.advance(); // 'a'
        scanner.advance(); // '\r' at EOF → standalone, line 2
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.current_column(), 1);
    }

    // =======================================================================
    // Mark and reset
    // =======================================================================

    #[test]
    fn test_mark_and_reset() {
        let mut scanner = Scanner::new("abcdef", 0);
        scanner.advance(); // 'a'
        scanner.advance(); // 'b'
        let mark = scanner.mark();
        assert_eq!(mark.pos, 2);
        assert_eq!(mark.index, 2);
        assert_eq!(mark.line, 1);
        assert_eq!(mark.column, 3);

        scanner.advance(); // 'c'
        scanner.advance(); // 'd'
        assert_eq!(scanner.peek(), Some('e'));
        assert_eq!(scanner.byte_offset(), 4);

        scanner.reset_to(mark);
        assert_eq!(scanner.peek(), Some('c'));
        assert_eq!(scanner.byte_offset(), 2);
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.current_column(), 3);
    }

    #[test]
    fn test_mark_at_start() {
        let scanner = Scanner::new("abc", 0);
        let mark = scanner.mark();
        assert_eq!(mark.pos, 0);
        assert_eq!(mark.index, 0);
        assert_eq!(mark.line, 1);
        assert_eq!(mark.column, 1);
    }

    #[test]
    fn test_mark_across_newlines() {
        let mut scanner = Scanner::new("a\nb\nc", 0);
        let mark_start = scanner.mark();
        scanner.advance(); // 'a'
        scanner.advance(); // '\n'
        scanner.advance(); // 'b'
        let mark_mid = scanner.mark();
        assert_eq!(mark_mid.line, 2);

        scanner.advance(); // '\n'
        scanner.advance(); // 'c'
        assert_eq!(scanner.current_line(), 3);

        scanner.reset_to(mark_start);
        assert_eq!(scanner.current_line(), 1);
        assert_eq!(scanner.peek(), Some('a'));

        scanner.reset_to(mark_mid);
        assert_eq!(scanner.current_line(), 2);
        assert_eq!(scanner.peek(), Some('\n'));
    }

    // =======================================================================
    // Slice extraction
    // =======================================================================

    #[test]
    fn test_slice_from() {
        let mut scanner = Scanner::new("hello world", 0);
        let start = scanner.byte_offset();
        scanner.advance_while(|ch| ch != ' ');
        let text = scanner.slice_from(start);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_slice_from_empty() {
        let scanner = Scanner::new("abc", 0);
        let start = scanner.byte_offset();
        let text = scanner.slice_from(start);
        assert_eq!(text, "");
    }

    #[test]
    fn test_slice_range() {
        let scanner = Scanner::new("hello world", 0);
        let text = scanner.slice_range(6, 11);
        assert_eq!(text, "world");
    }

    #[test]
    fn test_slice_range_with_multibyte() {
        // 'é' is 2 bytes: positions 0-1 = 'é', position 2 = 'a'
        let scanner = Scanner::new("éa", 0);
        let text = scanner.slice_range(0, 2);
        assert_eq!(text, "é");
    }

    // =======================================================================
    // Character classification — digits
    // =======================================================================

    #[test]
    fn test_is_digit() {
        for ch in '0'..='9' {
            assert!(Scanner::is_digit(ch), "expected '{}' to be a digit", ch);
        }
        assert!(!Scanner::is_digit('a'));
        assert!(!Scanner::is_digit('A'));
        assert!(!Scanner::is_digit(' '));
        assert!(!Scanner::is_digit('_'));
    }

    #[test]
    fn test_is_hex_digit() {
        for ch in '0'..='9' {
            assert!(Scanner::is_hex_digit(ch));
        }
        for ch in 'a'..='f' {
            assert!(Scanner::is_hex_digit(ch));
        }
        for ch in 'A'..='F' {
            assert!(Scanner::is_hex_digit(ch));
        }
        assert!(!Scanner::is_hex_digit('g'));
        assert!(!Scanner::is_hex_digit('G'));
        assert!(!Scanner::is_hex_digit(' '));
    }

    #[test]
    fn test_is_octal_digit() {
        for ch in '0'..='7' {
            assert!(Scanner::is_octal_digit(ch));
        }
        assert!(!Scanner::is_octal_digit('8'));
        assert!(!Scanner::is_octal_digit('9'));
        assert!(!Scanner::is_octal_digit('a'));
    }

    // =======================================================================
    // Character classification — identifiers
    // =======================================================================

    #[test]
    fn test_is_identifier_start_ascii() {
        assert!(Scanner::is_identifier_start('a'));
        assert!(Scanner::is_identifier_start('z'));
        assert!(Scanner::is_identifier_start('A'));
        assert!(Scanner::is_identifier_start('Z'));
        assert!(Scanner::is_identifier_start('_'));
        assert!(!Scanner::is_identifier_start('0'));
        assert!(!Scanner::is_identifier_start('9'));
        assert!(!Scanner::is_identifier_start(' '));
        assert!(!Scanner::is_identifier_start('+'));
    }

    #[test]
    fn test_is_identifier_continue_ascii() {
        assert!(Scanner::is_identifier_continue('a'));
        assert!(Scanner::is_identifier_continue('Z'));
        assert!(Scanner::is_identifier_continue('_'));
        assert!(Scanner::is_identifier_continue('0'));
        assert!(Scanner::is_identifier_continue('9'));
        assert!(!Scanner::is_identifier_continue(' '));
        assert!(!Scanner::is_identifier_continue('+'));
    }

    #[test]
    fn test_is_identifier_start_unicode() {
        // Latin Extended-A: 'Ā' (U+0100) should be valid identifier start
        assert!(Scanner::is_identifier_start('\u{0100}'));
        // CJK character should be valid identifier start
        assert!(Scanner::is_identifier_start('\u{4E00}')); // 一
        // Combining mark (U+0300) should NOT be valid identifier start (Annex D.2)
        assert!(!Scanner::is_identifier_start('\u{0300}'));
    }

    #[test]
    fn test_is_identifier_continue_unicode() {
        // Combining mark (U+0300) IS valid identifier continuation
        assert!(Scanner::is_identifier_continue('\u{0300}'));
        // Latin Extended-A: 'Ā' (U+0100) should be valid continuation
        assert!(Scanner::is_identifier_continue('\u{0100}'));
    }

    #[test]
    fn test_pua_not_identifier() {
        // PUA code points must NOT be treated as identifier characters
        assert!(!Scanner::is_identifier_start('\u{E080}'));
        assert!(!Scanner::is_identifier_start('\u{E0A0}'));
        assert!(!Scanner::is_identifier_start('\u{E0FF}'));
        assert!(!Scanner::is_identifier_continue('\u{E080}'));
        assert!(!Scanner::is_identifier_continue('\u{E0A0}'));
        assert!(!Scanner::is_identifier_continue('\u{E0FF}'));
    }

    // =======================================================================
    // Character classification — whitespace
    // =======================================================================

    #[test]
    fn test_is_whitespace() {
        assert!(Scanner::is_whitespace(' '));
        assert!(Scanner::is_whitespace('\t'));
        assert!(Scanner::is_whitespace('\n'));
        assert!(Scanner::is_whitespace('\r'));
        assert!(Scanner::is_whitespace('\x0B')); // vertical tab
        assert!(Scanner::is_whitespace('\x0C')); // form feed
        assert!(!Scanner::is_whitespace('a'));
        assert!(!Scanner::is_whitespace('0'));
        assert!(!Scanner::is_whitespace('_'));
    }

    #[test]
    fn test_pua_not_whitespace() {
        // PUA code points must NOT be treated as whitespace
        assert!(!Scanner::is_whitespace('\u{E080}'));
        assert!(!Scanner::is_whitespace('\u{E0FF}'));
    }

    // =======================================================================
    // saw_newline_since
    // =======================================================================

    #[test]
    fn test_saw_newline_since_true() {
        let mut scanner = Scanner::new("a\nb", 0);
        let mark = scanner.mark();
        scanner.advance(); // 'a'
        scanner.advance(); // '\n'
        assert!(scanner.saw_newline_since(&mark));
    }

    #[test]
    fn test_saw_newline_since_false() {
        let mut scanner = Scanner::new("abc", 0);
        let mark = scanner.mark();
        scanner.advance(); // 'a'
        scanner.advance(); // 'b'
        assert!(!scanner.saw_newline_since(&mark));
    }

    #[test]
    fn test_saw_newline_since_at_same_position() {
        let scanner = Scanner::new("abc", 0);
        let mark = scanner.mark();
        assert!(!scanner.saw_newline_since(&mark));
    }

    #[test]
    fn test_saw_newline_since_crlf() {
        let mut scanner = Scanner::new("a\r\nb", 0);
        let mark = scanner.mark();
        scanner.advance(); // 'a'
        scanner.advance(); // '\r' (no line increment — next is '\n')
        assert!(!scanner.saw_newline_since(&mark)); // not yet
        scanner.advance(); // '\n' (line increments now)
        assert!(scanner.saw_newline_since(&mark));
    }

    // =======================================================================
    // PUA character transparency
    // =======================================================================

    #[test]
    fn test_pua_chars_transparent_to_advance() {
        // Simulate PUA-encoded source with U+E080 (representing byte 0x80)
        let source = "a\u{E080}b";
        let mut scanner = Scanner::new(source, 0);
        assert_eq!(scanner.advance(), Some('a'));
        assert_eq!(scanner.advance(), Some('\u{E080}')); // PUA char flows through
        assert_eq!(scanner.advance(), Some('b'));
        assert_eq!(scanner.advance(), None);
    }

    #[test]
    fn test_pua_chars_transparent_to_peek() {
        let source = "\u{E0FF}x";
        let scanner = Scanner::new(source, 0);
        assert_eq!(scanner.peek(), Some('\u{E0FF}'));
        assert_eq!(scanner.peek_ahead(1), Some('x'));
    }

    // =======================================================================
    // is_at_end
    // =======================================================================

    #[test]
    fn test_is_at_end_empty() {
        let scanner = Scanner::new("", 0);
        assert!(scanner.is_at_end());
    }

    #[test]
    fn test_is_at_end_after_consuming_all() {
        let mut scanner = Scanner::new("ab", 0);
        assert!(!scanner.is_at_end());
        scanner.advance();
        assert!(!scanner.is_at_end());
        scanner.advance();
        assert!(scanner.is_at_end());
    }

    // =======================================================================
    // file_id
    // =======================================================================

    #[test]
    fn test_file_id() {
        let scanner = Scanner::new("", 99);
        assert_eq!(scanner.file_id(), 99);
    }

    // =======================================================================
    // Integration-style tests
    // =======================================================================

    #[test]
    fn test_lex_identifier_pattern() {
        // Simulate how the lexer would use the scanner to lex an identifier
        let mut scanner = Scanner::new("my_var123 + rest", 0);
        let start = scanner.byte_offset();
        // Consume identifier start
        assert!(Scanner::is_identifier_start(scanner.peek().unwrap()));
        scanner.advance();
        // Consume identifier continuation
        scanner.advance_while(|ch| Scanner::is_identifier_continue(ch));
        let ident = scanner.slice_from(start);
        assert_eq!(ident, "my_var123");
        // Next char should be space
        assert_eq!(scanner.peek(), Some(' '));
    }

    #[test]
    fn test_lex_number_pattern() {
        // Simulate how the lexer would lex a hex number
        let mut scanner = Scanner::new("0xFF + 1", 0);
        let start = scanner.byte_offset();
        // Consume '0'
        scanner.advance();
        // Consume 'x'
        scanner.advance_if('x');
        // Consume hex digits
        scanner.advance_while(|ch| Scanner::is_hex_digit(ch));
        let num = scanner.slice_from(start);
        assert_eq!(num, "0xFF");
    }

    #[test]
    fn test_mark_reset_for_prefix_detection() {
        // Simulate string prefix detection: check if 'L' is followed by '"'
        let mut scanner = Scanner::new("L\"hello\"", 0);
        let mark = scanner.mark();
        scanner.advance(); // consume 'L'
        let is_string_prefix = scanner.peek() == Some('"');
        if !is_string_prefix {
            scanner.reset_to(mark);
        }
        assert!(is_string_prefix);
        assert_eq!(scanner.peek(), Some('"'));
    }

    #[test]
    fn test_large_source_performance() {
        // Ensure the scanner handles moderately large sources correctly.
        // (Not a true performance benchmark, but validates correctness at scale.)
        let large_source: String = "abcdefghij\n".repeat(1000);
        let mut scanner = Scanner::new(&large_source, 0);
        let mut char_count = 0u32;
        while scanner.advance().is_some() {
            char_count += 1;
        }
        // 10 chars + 1 newline = 11 chars per line, 1000 lines
        assert_eq!(char_count, 11_000);
        assert_eq!(scanner.current_line(), 1001); // 1000 newlines → line 1001
        assert!(scanner.is_at_end());
    }

    #[test]
    fn test_mixed_line_endings() {
        // Mix of \n, \r\n, and \r line endings
        let source = "a\nb\r\nc\rd";
        let mut scanner = Scanner::new(source, 0);

        scanner.advance(); // 'a' → line 1
        assert_eq!(scanner.current_line(), 1);

        scanner.advance(); // '\n' → line 2
        assert_eq!(scanner.current_line(), 2);

        scanner.advance(); // 'b' → line 2
        assert_eq!(scanner.current_line(), 2);

        scanner.advance(); // '\r' (part of \r\n) → still line 2
        assert_eq!(scanner.current_line(), 2);

        scanner.advance(); // '\n' → line 3
        assert_eq!(scanner.current_line(), 3);

        scanner.advance(); // 'c' → line 3
        assert_eq!(scanner.current_line(), 3);

        scanner.advance(); // '\r' (standalone) → line 4
        assert_eq!(scanner.current_line(), 4);

        scanner.advance(); // 'd' → line 4
        assert_eq!(scanner.current_line(), 4);
    }
}
