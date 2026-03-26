//! String and character literal parsing for the BCC lexer (Phase 3).
//!
//! This module handles all C11 string and character literal lexing:
//!
//! - **Encoding prefixes**: `L`, `u8`, `u`, `U` for wide and Unicode literals.
//! - **Escape sequences**: simple (`\n`, `\t`, `\\`, …), octal (`\0`–`\377`),
//!   hexadecimal (`\xFF`), and Unicode (`\u0041`, `\U00000041`).
//! - **PUA transparency**: Non-UTF-8 bytes encoded as PUA code points
//!   (U+E080–U+E0FF) by the source-file reader flow through untouched.
//!   Escape sequences producing byte values 0x80–0xFF are also stored as
//!   PUA code points so the code generator can decode them back to exact bytes.
//! - **Adjacent string concatenation**: Detection and merging of adjacent
//!   string literals (e.g., `"hello" " world"` → `"hello world"`).
//! - **Diagnostic reporting**: Unterminated literals, empty character constants,
//!   invalid escape sequences, and incompatible prefix concatenation are all
//!   reported through [`DiagnosticEngine`] — no panics.
//!
//! # PUA Encoding Fidelity
//!
//! The Linux kernel contains binary data in string literals that **must** survive
//! the entire pipeline with byte-exact fidelity. PUA code points from the source
//! (representing non-UTF-8 bytes) are preserved as-is in the literal value string.
//! Escape sequences like `\xFF` also produce PUA code points so the code generator
//! can use [`encoding::decode_pua_to_byte()`] for faithful byte emission.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules. No external crates are used.

use super::scanner::Scanner;
use super::token::{StringPrefix, TokenKind};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::encoding;

// ---------------------------------------------------------------------------
// Internal helper — byte-value to char conversion with PUA encoding
// ---------------------------------------------------------------------------

/// Convert a byte value (0–255) from an escape sequence into a `char` suitable
/// for storage in a string literal's value `String`.
///
/// - Values 0x00–0x7F are stored as their direct ASCII/Unicode `char`
///   equivalents (single-byte UTF-8).
/// - Values 0x80–0xFF are stored as PUA code points (U+E080–U+E0FF) so the
///   code generator can later decode them to the exact original byte via
///   [`encoding::decode_pua_to_byte()`].
///
/// This maintains consistency with source-level PUA encoding and guarantees
/// byte-exact round-tripping for kernel binary data in string literals.
#[inline]
fn escape_byte_to_char(value: u8) -> char {
    if value < 0x80 {
        value as char
    } else {
        // Map 0x80–0xFF to PUA U+E080–U+E0FF for byte-exact fidelity.
        // SAFETY: 0xE000 + 0x80..=0xFF = 0xE080..=0xE0FF — always valid Unicode.
        char::from_u32(0xE000 + value as u32).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Escape sequence processing
// ---------------------------------------------------------------------------

/// Process a single escape sequence starting **after** the `\` has been consumed.
///
/// The scanner must be positioned at the character immediately following the
/// backslash. This function consumes the escape characters and returns the
/// decoded value, or `None` on error (with diagnostics emitted).
///
/// # Supported Escape Sequences
///
/// | Escape  | Value | Description              |
/// |---------|-------|--------------------------|
/// | `\n`    | 0x0A  | Newline (line feed)      |
/// | `\t`    | 0x09  | Horizontal tab           |
/// | `\r`    | 0x0D  | Carriage return          |
/// | `\a`    | 0x07  | Alert (bell)             |
/// | `\b`    | 0x08  | Backspace                |
/// | `\f`    | 0x0C  | Form feed                |
/// | `\v`    | 0x0B  | Vertical tab             |
/// | `\\`    | 0x5C  | Backslash                |
/// | `\"`    | 0x22  | Double quote             |
/// | `\'`    | 0x27  | Single quote             |
/// | `\?`    | 0x3F  | Question mark            |
/// | `\0`–`\377` | 0–255 | Octal (1–3 digits) |
/// | `\xHH`  | 0–…   | Hexadecimal (1+ digits)  |
/// | `\uHHHH`| U+… | Unicode (4 hex digits)     |
/// | `\UHHHHHHHH`| U+… | Unicode (8 hex digits) |
fn process_escape_sequence(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> Option<char> {
    let esc_start = scanner.offset();
    let ch = match scanner.advance() {
        Some(c) => c,
        None => {
            // EOF right after backslash
            let span = Span::new(file_id, esc_start.saturating_sub(1), esc_start);
            diagnostics.emit_error(span, "unexpected end of file in escape sequence");
            return None;
        }
    };

    match ch {
        // -- Simple escapes -----------------------------------------------
        'n' => Some('\n'),   // 0x0A
        't' => Some('\t'),   // 0x09
        'r' => Some('\r'),   // 0x0D
        'a' => Some('\x07'), // 0x07 BEL
        'b' => Some('\x08'), // 0x08 BS
        'f' => Some('\x0C'), // 0x0C FF
        'v' => Some('\x0B'), // 0x0B VT
        '\\' => Some('\\'),  // 0x5C
        '"' => Some('"'),    // 0x22
        '\'' => Some('\''),  // 0x27
        '?' => Some('?'),    // 0x3F (trigraph prevention)

        // -- Octal escapes (\0 through \377) ------------------------------
        '0'..='7' => {
            let mut value: u32 = (ch as u32) - ('0' as u32);
            // Read up to 2 more octal digits (3 total).
            for _ in 0..2 {
                match scanner.peek() {
                    Some(d @ '0'..='7') => {
                        scanner.advance();
                        value = value * 8 + (d as u32 - '0' as u32);
                    }
                    _ => break,
                }
            }
            // Octal escapes in C produce a single byte (0–255).
            if value > 255 {
                let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
                diagnostics.emit_warning(span, "octal escape sequence out of range");
                value &= 0xFF;
            }
            Some(escape_byte_to_char(value as u8))
        }

        // -- Hexadecimal escapes (\xHH…) ----------------------------------
        'x' => {
            let hex_start = scanner.offset();
            let mut value: u32 = 0;
            let mut count: u32 = 0;
            while let Some(d) = scanner.peek() {
                let digit = match d {
                    '0'..='9' => d as u32 - '0' as u32,
                    'a'..='f' => d as u32 - 'a' as u32 + 10,
                    'A'..='F' => d as u32 - 'A' as u32 + 10,
                    _ => break,
                };
                scanner.advance();
                value = value.wrapping_mul(16).wrapping_add(digit);
                count += 1;
            }
            if count == 0 {
                let span = Span::new(file_id, esc_start.saturating_sub(1), hex_start);
                diagnostics.emit_error(span, "\\x used with no following hex digits");
                return None;
            }
            // For byte strings, values > 0xFF are implementation-defined.
            // We truncate with a warning for very large values.
            if value > 0xFF {
                let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
                diagnostics.emit_warning(
                    span,
                    format!(
                        "hex escape sequence \\x{:X} out of range for character",
                        value
                    ),
                );
                // Truncate to byte — the caller can use the full value for wide strings.
                value &= 0xFF;
            }
            Some(escape_byte_to_char(value as u8))
        }

        // -- Unicode escape \uHHHH (4 hex digits) -------------------------
        'u' => process_unicode_escape(scanner, diagnostics, file_id, 4, esc_start),

        // -- Unicode escape \UHHHHHHHH (8 hex digits) ---------------------
        'U' => process_unicode_escape(scanner, diagnostics, file_id, 8, esc_start),

        // -- Unknown escape: warn and use the literal character ------------
        other => {
            let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
            diagnostics.emit_warning(span, format!("unknown escape sequence '\\{}'", other));
            Some(other)
        }
    }
}

/// Process a `\u` (4 digits) or `\U` (8 digits) Unicode escape sequence.
///
/// Reads exactly `num_digits` hexadecimal digits, validates the resulting
/// code point is a valid Unicode scalar value (not a surrogate), and returns
/// the decoded character.
fn process_unicode_escape(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    num_digits: usize,
    esc_start: u32,
) -> Option<char> {
    let mut value: u32 = 0;
    let mut count: usize = 0;

    for _ in 0..num_digits {
        match scanner.peek() {
            Some(d) => {
                let digit = match d {
                    '0'..='9' => d as u32 - '0' as u32,
                    'a'..='f' => d as u32 - 'a' as u32 + 10,
                    'A'..='F' => d as u32 - 'A' as u32 + 10,
                    _ => break,
                };
                scanner.advance();
                value = value * 16 + digit;
                count += 1;
            }
            None => break,
        }
    }

    if count != num_digits {
        let prefix_char = if num_digits == 4 { 'u' } else { 'U' };
        let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
        diagnostics.emit_error(
            span,
            format!(
                "\\{} escape requires exactly {} hex digits, found {}",
                prefix_char, num_digits, count
            ),
        );
        return None;
    }

    // Reject surrogate code points (U+D800–U+DFFF).
    if (0xD800..=0xDFFF).contains(&value) {
        let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
        diagnostics.emit_error(
            span,
            format!(
                "\\{} escape sequence value U+{:04X} is a surrogate code point",
                if num_digits == 4 { 'u' } else { 'U' },
                value
            ),
        );
        return None;
    }

    match char::from_u32(value) {
        Some(c) => Some(c),
        None => {
            let span = Span::new(file_id, esc_start.saturating_sub(1), scanner.offset());
            diagnostics.emit_error(span, format!("invalid Unicode code point U+{:04X}", value));
            None
        }
    }
}

// ---------------------------------------------------------------------------
// String literal lexing
// ---------------------------------------------------------------------------

/// Lex a string literal starting at the opening `"`.
///
/// The encoding prefix (if any) has already been consumed by the caller;
/// `prefix` indicates what was found. The scanner must be positioned **at**
/// the opening `"` character.
///
/// # Processing
///
/// 1. Consumes the opening `"`.
/// 2. Reads characters, resolving escape sequences and preserving PUA code
///    points, until the closing `"` or an error condition (EOF / unescaped
///    newline).
/// 3. Returns `TokenKind::StringLiteral` with the processed value.
///
/// # Diagnostics
///
/// - **Error**: unterminated string literal (EOF or unescaped newline).
/// - **Warning**: unknown escape sequence.
pub fn lex_string_literal(
    scanner: &mut Scanner,
    prefix: StringPrefix,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> TokenKind {
    let start_offset = scanner.offset();

    // Consume the opening double-quote.
    debug_assert_eq!(scanner.peek(), Some('"'));
    scanner.advance();

    let mut value = String::new();

    loop {
        match scanner.peek() {
            None => {
                // EOF — unterminated string literal.
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(span, "unterminated string literal");
                break;
            }
            Some('\n') => {
                // Unescaped newline — unterminated string literal.
                // (Line splicing was handled in Phase 1.)
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(span, "missing terminating '\"' character");
                break;
            }
            Some('"') => {
                // Closing quote — end of this string literal.
                scanner.advance();
                break;
            }
            Some('\\') => {
                // Escape sequence.
                scanner.advance(); // consume '\'
                if let Some(ch) = process_escape_sequence(scanner, diagnostics, file_id) {
                    value.push(ch);
                }
                // On None (error), we skip the bad escape and continue lexing.
            }
            Some(ch) => {
                // Regular character — including PUA code points which are
                // preserved as-is for byte-exact round-tripping.
                scanner.advance();
                value.push(ch);
            }
        }
    }

    TokenKind::StringLiteral { value, prefix }
}

// ---------------------------------------------------------------------------
// Character literal lexing
// ---------------------------------------------------------------------------

/// Lex a character literal starting at the opening `'`.
///
/// The encoding prefix (if any) has already been consumed by the caller;
/// `prefix` indicates what was found. The scanner must be positioned **at**
/// the opening `'` character.
///
/// # Character Constant Semantics
///
/// - Empty character constant `''` is an error.
/// - Single character (or single escape sequence) produces the character's
///   numeric code point value.
/// - Multi-character constants (e.g., `'AB'`) are implementation-defined —
///   supported with a warning. The value is computed by packing characters
///   in big-endian byte order.
///
/// # Diagnostics
///
/// - **Error**: empty character constant, unterminated literal.
/// - **Warning**: multi-character character constant, unknown escape.
pub fn lex_char_literal(
    scanner: &mut Scanner,
    prefix: StringPrefix,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> TokenKind {
    let start_offset = scanner.offset();

    // Consume the opening single-quote.
    debug_assert_eq!(scanner.peek(), Some('\''));
    scanner.advance();

    let mut chars: Vec<u32> = Vec::new();

    loop {
        match scanner.peek() {
            None => {
                // EOF — unterminated character literal.
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(span, "unterminated character constant");
                break;
            }
            Some('\n') => {
                // Unescaped newline — unterminated character literal.
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(span, "missing terminating \"'\" character");
                break;
            }
            Some('\'') => {
                // Closing quote — end of this character literal.
                scanner.advance();
                break;
            }
            Some('\\') => {
                // Escape sequence.
                scanner.advance(); // consume '\'
                if let Some(ch) = process_escape_sequence(scanner, diagnostics, file_id) {
                    chars.push(ch as u32);
                }
            }
            Some(ch) => {
                // Regular character — PUA code points are decoded to their
                // byte value for numeric representation.
                scanner.advance();
                if encoding::is_pua_encoded(ch) {
                    if let Some(byte) = encoding::decode_pua_to_byte(ch) {
                        chars.push(byte as u32);
                    } else {
                        chars.push(ch as u32);
                    }
                } else {
                    chars.push(ch as u32);
                }
            }
        }
    }

    // Validate the character constant.
    let end_offset = scanner.offset();
    let span = Span::new(file_id, start_offset, end_offset);

    if chars.is_empty() {
        diagnostics.emit_error(span, "empty character constant");
        return TokenKind::CharLiteral { value: 0, prefix };
    }

    if chars.len() > 1 {
        diagnostics.emit_warning(
            span,
            format!(
                "multi-character character constant '{}'",
                chars
                    .iter()
                    .map(|&v| {
                        if let Some(c) = char::from_u32(v) {
                            if c.is_ascii_graphic() {
                                return c.to_string();
                            }
                        }
                        format!("\\x{:02X}", v)
                    })
                    .collect::<String>()
            ),
        );
    }

    // Compute the multi-char packed value (big-endian byte packing).
    // For a single char, this is just the char's code point value.
    let value = chars
        .iter()
        .fold(0u32, |acc, &v| acc.wrapping_shl(8) | (v & 0xFF));

    TokenKind::CharLiteral { value, prefix }
}

// ---------------------------------------------------------------------------
// Prefix detection
// ---------------------------------------------------------------------------

/// Detect if the current scanner position starts a string/character literal
/// prefix (`L`, `u`, `u8`, `U`).
///
/// Called from the lexer main loop when an `L`, `u`, or `U` is seen. The
/// scanner must be positioned **at** the potential prefix character (it has
/// NOT been consumed yet).
///
/// # Returns
///
/// - `Some((prefix, is_string))` if a valid prefix pattern is detected.
///   `is_string` is `true` for string literals (`"`) and `false` for character
///   literals (`'`). The prefix characters are consumed; the scanner is left
///   positioned at the opening quote.
/// - `None` if the character sequence does not form a valid literal prefix.
///   The scanner is left unmodified.
///
/// # Recognized Patterns
///
/// | Pattern | Prefix | Literal Type |
/// |---------|--------|-------------|
/// | `L"`    | L      | string      |
/// | `L'`    | L      | char        |
/// | `u"`    | U16    | string      |
/// | `u'`    | U16    | char        |
/// | `u8"`   | U8     | string      |
/// | `U"`    | U32    | string      |
/// | `U'`    | U32    | char        |
///
/// Note: `u8'…'` (u8 char literal) is not supported in C11.
pub fn detect_prefix(scanner: &mut Scanner) -> Option<(StringPrefix, bool)> {
    let first = scanner.peek()?;

    match first {
        'L' => {
            match scanner.peek_nth(1) {
                Some('"') => {
                    scanner.advance(); // consume 'L'
                    Some((StringPrefix::L, true))
                }
                Some('\'') => {
                    scanner.advance(); // consume 'L'
                    Some((StringPrefix::L, false))
                }
                _ => None,
            }
        }
        'u' => {
            match scanner.peek_nth(1) {
                Some('8') => {
                    // Check for u8" (u8 string literal).
                    if scanner.peek_nth(2) == Some('"') {
                        scanner.advance(); // consume 'u'
                        scanner.advance(); // consume '8'
                        Some((StringPrefix::U8, true))
                    } else {
                        // u8 but not followed by " — not a string prefix.
                        // Note: u8'…' is NOT valid in C11.
                        None
                    }
                }
                Some('"') => {
                    scanner.advance(); // consume 'u'
                    Some((StringPrefix::U16, true))
                }
                Some('\'') => {
                    scanner.advance(); // consume 'u'
                    Some((StringPrefix::U16, false))
                }
                _ => None,
            }
        }
        'U' => {
            match scanner.peek_nth(1) {
                Some('"') => {
                    scanner.advance(); // consume 'U'
                    Some((StringPrefix::U32, true))
                }
                Some('\'') => {
                    scanner.advance(); // consume 'U'
                    Some((StringPrefix::U32, false))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Adjacent string literal detection and concatenation
// ---------------------------------------------------------------------------

/// Check if there is an adjacent string literal following the current
/// scanner position (separated only by whitespace).
///
/// Looks ahead past whitespace characters (spaces, tabs, newlines) to
/// determine whether the next non-whitespace content starts a string literal
/// (either a bare `"` or a prefix + `"`).
///
/// This function does **not** consume any characters from the scanner — it
/// uses only look-ahead via [`Scanner::peek_nth`].
///
/// # Note
///
/// Comments have been stripped in preprocessing (Phase 1-2), so only
/// whitespace can appear between adjacent string literals at this stage.
pub fn has_adjacent_string(scanner: &mut Scanner) -> bool {
    let mut n: usize = 0;
    // Skip whitespace characters by advancing the peek index.
    while let Some(' ') | Some('\t') | Some('\n') = scanner.peek_nth(n) {
        n += 1;
    }
    // Check if what follows is a string literal (with or without prefix).
    match scanner.peek_nth(n) {
        Some('"') => true,
        Some('L') => scanner.peek_nth(n + 1) == Some('"'),
        Some('u') => match scanner.peek_nth(n + 1) {
            Some('"') => true,
            Some('8') => scanner.peek_nth(n + 2) == Some('"'),
            _ => false,
        },
        Some('U') => scanner.peek_nth(n + 1) == Some('"'),
        _ => false,
    }
}

/// Skip whitespace characters (spaces, tabs, newlines) in the scanner.
///
/// Consumes all consecutive whitespace at the current position.
fn skip_whitespace(scanner: &mut Scanner) {
    while let Some(' ') | Some('\t') | Some('\n') = scanner.peek() {
        scanner.advance();
    }
}

/// Merge two string prefixes during adjacent literal concatenation.
///
/// # C11 Rules (§6.4.5 p5)
///
/// - If both prefixes are the same, the result has that prefix.
/// - If one is `None` and the other is a specific prefix, the result has
///   the specific prefix.
/// - If both are different non-`None` prefixes, the behavior is
///   implementation-defined — we emit a diagnostic error and use the first
///   prefix.
fn merge_prefix(
    a: StringPrefix,
    b: StringPrefix,
    diagnostics: &mut DiagnosticEngine,
    _file_id: u32,
    span: Span,
) -> StringPrefix {
    match (a, b) {
        // Both None → None.
        (StringPrefix::None, StringPrefix::None) => StringPrefix::None,
        // One None, one specific → use the specific prefix.
        (StringPrefix::None, other) | (other, StringPrefix::None) => other,
        // Same non-None prefix → keep it.
        (a_pref, b_pref) if a_pref == b_pref => a_pref,
        // Different non-None prefixes → diagnostic error.
        (a_pref, b_pref) => {
            diagnostics.emit_error(
                span,
                format!(
                    "unsupported non-standard concatenation of string literals \
                     with incompatible encoding prefixes ({:?} and {:?})",
                    a_pref, b_pref
                ),
            );
            // Use the first prefix as a recovery strategy.
            a_pref
        }
    }
}

/// Consume whitespace and detect the prefix (if any) of the next adjacent
/// string literal.
///
/// Returns the detected prefix and advances the scanner past the prefix
/// characters to the opening `"`. Returns `None` if the next non-whitespace
/// token is not a string literal.
fn consume_adjacent_prefix(scanner: &mut Scanner) -> Option<StringPrefix> {
    skip_whitespace(scanner);
    match scanner.peek() {
        Some('"') => Some(StringPrefix::None),
        Some('L') => {
            if scanner.peek_nth(1) == Some('"') {
                scanner.advance(); // consume 'L'
                Some(StringPrefix::L)
            } else {
                None
            }
        }
        Some('u') => match scanner.peek_nth(1) {
            Some('8') if scanner.peek_nth(2) == Some('"') => {
                scanner.advance(); // consume 'u'
                scanner.advance(); // consume '8'
                Some(StringPrefix::U8)
            }
            Some('"') => {
                scanner.advance(); // consume 'u'
                Some(StringPrefix::U16)
            }
            _ => None,
        },
        Some('U') => {
            if scanner.peek_nth(1) == Some('"') {
                scanner.advance(); // consume 'U'
                Some(StringPrefix::U32)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Lex a string literal and concatenate any adjacent string literals.
///
/// This function lexes the first string literal (the scanner must be at the
/// opening `"`) and then greedily concatenates any adjacent string literals
/// separated only by whitespace.
///
/// # Concatenation Rules (C11 §6.4.5)
///
/// - Adjacent string literals with the same prefix are concatenated normally.
/// - If one literal has no prefix (`None`) and the other has a specific
///   prefix, the result adopts the specific prefix.
/// - If both literals have different non-`None` prefixes, a diagnostic error
///   is emitted and the first prefix is used.
///
/// # Example
///
/// ```c
/// "hello" " world"       // → "hello world"  (None)
/// L"wide" " string"      // → L"wide string" (L)
/// "a" u8"b"              // → u8"ab"         (U8)
/// L"a" U"b"              // → error          (incompatible prefixes)
/// ```
pub fn lex_string_with_concatenation(
    scanner: &mut Scanner,
    prefix: StringPrefix,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> TokenKind {
    // Lex the first string literal.
    let first = lex_string_literal(scanner, prefix, diagnostics, file_id);
    let (mut combined_value, mut combined_prefix) = match first {
        TokenKind::StringLiteral { value, prefix: p } => (value, p),
        other => return other,
    };

    // Greedily concatenate adjacent string literals.
    while has_adjacent_string(scanner) {
        let concat_start = scanner.offset();

        // Consume whitespace and detect the next literal's prefix.
        let next_prefix = match consume_adjacent_prefix(scanner) {
            Some(p) => p,
            None => break, // Shouldn't happen since has_adjacent_string was true.
        };

        // Merge prefixes with compatibility checking.
        let span = Span::new(file_id, concat_start, scanner.offset());
        combined_prefix = merge_prefix(combined_prefix, next_prefix, diagnostics, file_id, span);

        // Lex the next string literal and append its value.
        match lex_string_literal(scanner, next_prefix, diagnostics, file_id) {
            TokenKind::StringLiteral { value, .. } => {
                combined_value.push_str(&value);
            }
            _ => break,
        }
    }

    TokenKind::StringLiteral {
        value: combined_value,
        prefix: combined_prefix,
    }
}
