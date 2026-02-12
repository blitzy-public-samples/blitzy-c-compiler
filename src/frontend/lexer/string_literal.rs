//! String and character literal lexing module for the BCC compiler.
//!
//! Handles parsing of C11 string and character literals with full escape
//! sequence support and wide/unicode encoding prefixes. Supports adjacent
//! string literal concatenation per C11 §6.4.5.
//!
//! # Features
//!
//! - **Escape Sequences**: `\n`, `\t`, `\r`, `\0`, `\\`, `\"`, `\'`, `\a`,
//!   `\b`, `\f`, `\v`, `\?`, hex `\xHH`, octal `\NNN`, universal `\uNNNN`
//!   and `\UNNNNNNNN`
//! - **Encoding Prefixes**: `L` (wchar_t), `u8` (UTF-8), `u` (char16_t),
//!   `U` (char32_t)
//! - **Adjacent String Concatenation**: Consecutive string literals are merged
//!   into a single token per C11 §6.4.5
//! - **PUA Transparency**: Non-UTF-8 bytes encoded as PUA code points by the
//!   source encoding layer pass through string bodies unchanged (Section 0.7.9)
//!
//! # Usage
//!
//! Called by the lexer's main tokenization loop when a quote character or
//! encoding prefix is encountered.

use super::scanner::{Scanner, ScannerMark};
use super::token::{CharPrefix, Span, StringPrefix, Token, TokenKind};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::encoding::{decode_pua_to_byte, is_pua_encoded};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert a hexadecimal digit character to its numeric value (0–15).
///
/// Returns the numeric value for '0'–'9', 'a'–'f', and 'A'–'F'.
/// Panics in debug mode if `ch` is not a valid hex digit.
#[inline]
fn hex_digit_value(ch: char) -> u32 {
    match ch {
        '0'..='9' => ch as u32 - '0' as u32,
        'a'..='f' => ch as u32 - 'a' as u32 + 10,
        'A'..='F' => ch as u32 - 'A' as u32 + 10,
        _ => {
            debug_assert!(false, "hex_digit_value called with non-hex char '{}'", ch);
            0
        }
    }
}

/// Push a Unicode code point encoded as UTF-8 bytes into a byte vector.
///
/// If the code point is not a valid Unicode scalar value (e.g., surrogates
/// or values above U+10FFFF), pushes U+FFFD REPLACEMENT CHARACTER instead.
fn push_code_point_utf8(value: u32, bytes: &mut Vec<u8>) {
    if let Some(ch) = char::from_u32(value) {
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        bytes.extend_from_slice(encoded.as_bytes());
    } else {
        // Invalid code point — emit U+FFFD replacement character in UTF-8
        bytes.extend_from_slice(&[0xEF, 0xBF, 0xBD]);
    }
}

/// Push a character's byte representation into a byte vector.
///
/// PUA-encoded characters (U+E080–U+E0FF) are decoded back to their original
/// raw byte value (0x80–0xFF), preserving byte-exact fidelity for non-UTF-8
/// source content (Section 0.7.9). Regular characters are encoded as UTF-8.
#[inline]
fn push_char_to_bytes(ch: char, bytes: &mut Vec<u8>) {
    if is_pua_encoded(ch) {
        // PUA code point from source encoding layer — decode to original byte
        if let Some(raw_byte) = decode_pua_to_byte(ch) {
            bytes.push(raw_byte);
        }
    } else if (ch as u32) < 0x80 {
        // ASCII fast path: single byte
        bytes.push(ch as u8);
    } else {
        // Multi-byte Unicode character: encode as UTF-8
        let mut buf = [0u8; 4];
        let encoded = ch.encode_utf8(&mut buf);
        bytes.extend_from_slice(encoded.as_bytes());
    }
}

/// Get the numeric value of a character, decoding PUA back to the original
/// byte value if applicable.
///
/// For regular characters, returns the Unicode code point value. For PUA
/// characters, returns the decoded raw byte value (0x80–0xFF).
#[inline]
fn char_value(ch: char) -> u32 {
    if is_pua_encoded(ch) {
        if let Some(raw_byte) = decode_pua_to_byte(ch) {
            return raw_byte as u32;
        }
    }
    ch as u32
}

// ---------------------------------------------------------------------------
// Escape Sequence Processing
// ---------------------------------------------------------------------------

/// Result of processing a single escape sequence.
///
/// Encodes the three possible outcomes: a single byte (most escapes), a
/// Unicode code point (universal character names), or raw bytes when an
/// error occurs and the literal characters are preserved.
enum EscapeResult {
    /// Single byte value — used for simple escapes, octal, and hex.
    Byte(u8),
    /// Unicode code point — used for `\uNNNN` and `\UNNNNNNNN`.
    CodePoint(u32),
    /// Raw byte sequence — error fallback, includes the literal backslash
    /// and the unrecognized character(s).
    RawBytes(Vec<u8>),
}

/// Process a single escape sequence starting after the backslash character.
///
/// The caller has already consumed the `\` character and recorded its byte
/// offset. This function reads the escape specifier and any additional
/// characters (hex digits, octal digits, etc.) and returns the resulting
/// value.
///
/// Handles all C11 §6.4.4.4 escape sequences plus universal character names
/// (C11 §6.4.3). Unknown escape sequences emit a warning and are preserved
/// literally.
fn process_escape(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    file_id: u32,
    backslash_offset: u32,
) -> EscapeResult {
    match scanner.advance() {
        None => {
            // EOF immediately after backslash
            let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
            diag.error(span, "unexpected end of file in escape sequence");
            EscapeResult::RawBytes(vec![b'\\'])
        }
        Some(ch) => match ch {
            // ---------------------------------------------------------------
            // Simple escape sequences (C11 §6.4.4.4)
            // ---------------------------------------------------------------
            'n' => EscapeResult::Byte(0x0A),   // newline
            't' => EscapeResult::Byte(0x09),   // horizontal tab
            'r' => EscapeResult::Byte(0x0D),   // carriage return
            'a' => EscapeResult::Byte(0x07),   // alert (bell)
            'b' => EscapeResult::Byte(0x08),   // backspace
            'f' => EscapeResult::Byte(0x0C),   // form feed
            'v' => EscapeResult::Byte(0x0B),   // vertical tab
            '\\' => EscapeResult::Byte(b'\\'), // backslash
            '\'' => EscapeResult::Byte(b'\''), // single quote
            '"' => EscapeResult::Byte(b'"'),   // double quote
            '?' => EscapeResult::Byte(b'?'),   // question mark (trigraph avoidance)

            // ---------------------------------------------------------------
            // Octal escape: \NNN — 1 to 3 octal digits (C11 §6.4.4.4)
            // The first octal digit is the character `ch` already consumed.
            // ---------------------------------------------------------------
            '0'..='7' => {
                let mut value: u32 = (ch as u32) - ('0' as u32);
                // Read up to 2 more octal digits (3 total maximum)
                for _ in 0..2 {
                    match scanner.peek() {
                        Some(next) if Scanner::is_octal_digit(next) => {
                            value = value * 8 + (next as u32 - '0' as u32);
                            scanner.advance();
                        }
                        _ => break,
                    }
                }
                if value > 255 {
                    let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                    diag.warning(span, "octal escape sequence out of range");
                }
                EscapeResult::Byte((value & 0xFF) as u8)
            }

            // ---------------------------------------------------------------
            // Hex escape: \xHH... — one or more hex digits (C11 §6.4.4.4)
            // Reads ALL consecutive hex digits; warns if value > 255 for
            // narrow string context.
            // ---------------------------------------------------------------
            'x' => {
                let has_digits = scanner.peek().is_some_and(Scanner::is_hex_digit);
                if !has_digits {
                    let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                    diag.warning(span, "\\x used with no following hex digits");
                    EscapeResult::RawBytes(vec![b'\\', b'x'])
                } else {
                    let mut value: u64 = 0;
                    let mut overflow = false;
                    while let Some(next) = scanner.peek() {
                        if Scanner::is_hex_digit(next) {
                            let digit = hex_digit_value(next) as u64;
                            value = value.wrapping_mul(16).wrapping_add(digit);
                            if value > 0xFFFF_FFFF {
                                overflow = true;
                            }
                            scanner.advance();
                        } else {
                            break;
                        }
                    }
                    if overflow || value > 255 {
                        let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                        diag.warning(span, "hex escape sequence out of range");
                    }
                    EscapeResult::Byte((value & 0xFF) as u8)
                }
            }

            // ---------------------------------------------------------------
            // Universal character name: \uNNNN — exactly 4 hex digits
            // (C11 §6.4.3)
            // ---------------------------------------------------------------
            'u' => {
                let mut value: u32 = 0;
                let mut count: u32 = 0;
                for _ in 0..4 {
                    match scanner.peek() {
                        Some(next) if Scanner::is_hex_digit(next) => {
                            value = value * 16 + hex_digit_value(next);
                            scanner.advance();
                            count += 1;
                        }
                        _ => break,
                    }
                }
                if count != 4 {
                    let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                    diag.error(
                        span,
                        "incomplete universal character name: \\u requires exactly 4 hex digits",
                    );
                    EscapeResult::RawBytes(vec![b'\\', b'u'])
                } else {
                    EscapeResult::CodePoint(value)
                }
            }

            // ---------------------------------------------------------------
            // Universal character name: \UNNNNNNNN — exactly 8 hex digits
            // (C11 §6.4.3)
            // ---------------------------------------------------------------
            'U' => {
                let mut value: u64 = 0;
                let mut count: u32 = 0;
                for _ in 0..8 {
                    match scanner.peek() {
                        Some(next) if Scanner::is_hex_digit(next) => {
                            value = value * 16 + hex_digit_value(next) as u64;
                            scanner.advance();
                            count += 1;
                        }
                        _ => break,
                    }
                }
                if count != 8 {
                    let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                    diag.error(
                        span,
                        "incomplete universal character name: \\U requires exactly 8 hex digits",
                    );
                    EscapeResult::RawBytes(vec![b'\\', b'U'])
                } else if value > 0x10FFFF {
                    let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                    diag.error(
                        span,
                        "universal character name value exceeds Unicode maximum (U+10FFFF)",
                    );
                    // Use replacement character U+FFFD
                    EscapeResult::CodePoint(0xFFFD)
                } else {
                    EscapeResult::CodePoint(value as u32)
                }
            }

            // ---------------------------------------------------------------
            // Unknown escape sequence — emit warning, include literally
            // ---------------------------------------------------------------
            other => {
                let span = Span::new(file_id, backslash_offset, scanner.byte_offset());
                let msg = format!("unknown escape sequence '\\{}'", other);
                diag.warning(span, msg);
                // Preserve the backslash and the unrecognized character
                let mut bytes = vec![b'\\'];
                if is_pua_encoded(other) {
                    if let Some(raw) = decode_pua_to_byte(other) {
                        bytes.push(raw);
                    }
                } else {
                    let mut buf = [0u8; 4];
                    let s = other.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
                EscapeResult::RawBytes(bytes)
            }
        },
    }
}

// ---------------------------------------------------------------------------
// String Literal Lexing
// ---------------------------------------------------------------------------

/// Lex a string literal starting at the opening double-quote.
///
/// The scanner must be positioned at the `"` character. The `prefix`
/// parameter indicates the encoding prefix that was already detected
/// and consumed by the caller.
///
/// # Arguments
///
/// * `scanner` — Character scanner positioned at the opening `"`.
/// * `prefix` — The encoding prefix (None, L, U8, SmallU, BigU).
/// * `diag` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// A `Token` with kind `TokenKind::StringLiteral` containing the
/// processed byte vector and encoding prefix. On error (unterminated
/// string), returns a token with whatever bytes were collected so far
/// to allow continued parsing.
pub fn lex_string_literal(
    scanner: &mut Scanner,
    prefix: StringPrefix,
    diag: &mut DiagnosticEngine,
) -> Token {
    let file_id = scanner.file_id();
    let start_offset = scanner.byte_offset();

    // Safety check: we expect the scanner to be at the opening double-quote.
    // If not (programming error), produce an error token.
    if !scanner.advance_if('"') {
        let span = Span::new(file_id, start_offset, scanner.byte_offset());
        diag.error(span, "expected opening double-quote for string literal");
        return Token::new(TokenKind::Error, span);
    }

    let mut bytes: Vec<u8> = Vec::new();

    loop {
        // Check for EOF before reading the next character
        if scanner.is_at_end() {
            let span = Span::new(file_id, start_offset, scanner.byte_offset());
            diag.error(span, "unterminated string literal");
            return Token::new(
                TokenKind::StringLiteral {
                    value: bytes,
                    prefix,
                },
                span,
            );
        }

        match scanner.peek() {
            None => {
                // Redundant with is_at_end() above, but included for completeness
                let span = Span::new(file_id, start_offset, scanner.byte_offset());
                diag.error(span, "unterminated string literal");
                return Token::new(
                    TokenKind::StringLiteral {
                        value: bytes,
                        prefix,
                    },
                    span,
                );
            }
            Some('\n') | Some('\r') => {
                // Newline before closing quote — unterminated string literal.
                // Do NOT consume the newline — let the outer lexer handle it.
                let span = Span::new(file_id, start_offset, scanner.byte_offset());
                diag.error(span, "unterminated string literal (missing closing \")");
                return Token::new(
                    TokenKind::StringLiteral {
                        value: bytes,
                        prefix,
                    },
                    span,
                );
            }
            Some('"') => {
                // Closing double-quote — end of string literal
                scanner.advance();
                let span = Span::new(file_id, start_offset, scanner.byte_offset());
                return Token::new(
                    TokenKind::StringLiteral {
                        value: bytes,
                        prefix,
                    },
                    span,
                );
            }
            Some('\\') => {
                // Start of an escape sequence
                let backslash_offset = scanner.byte_offset();
                scanner.advance(); // consume the backslash
                match process_escape(scanner, diag, file_id, backslash_offset) {
                    EscapeResult::Byte(b) => bytes.push(b),
                    EscapeResult::CodePoint(cp) => push_code_point_utf8(cp, &mut bytes),
                    EscapeResult::RawBytes(raw) => bytes.extend_from_slice(&raw),
                }
            }
            Some(ch) => {
                // Regular character — push its byte representation.
                // PUA characters are decoded to their original byte value per
                // Section 0.7.9 (byte-exact fidelity for non-UTF-8 source).
                push_char_to_bytes(ch, &mut bytes);
                scanner.advance();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Character Literal Lexing
// ---------------------------------------------------------------------------

/// Lex a character literal starting at the opening single-quote.
///
/// The scanner must be positioned at the `'` character. The `prefix`
/// parameter indicates the encoding prefix that was already detected
/// and consumed by the caller.
///
/// Character literal semantics (C11 §6.4.4.4):
/// - Exactly one character (after escape processing) for regular char literals.
/// - Multi-character constants produce a warning and an implementation-defined
///   value (bytes are combined by shifting: `'ab'` → `('a' << 8) | 'b'`).
/// - Empty character constants (`''`) produce an error diagnostic.
/// - Wide char (`L'x'`) stores the value at wchar_t width.
///
/// # Arguments
///
/// * `scanner` — Character scanner positioned at the opening `'`.
/// * `prefix` — The encoding prefix (None, L, SmallU, BigU).
/// * `diag` — Diagnostic engine for error/warning reporting.
///
/// # Returns
///
/// A `Token` with kind `TokenKind::CharLiteral` containing the character
/// value (as u32) and encoding prefix. On error, returns a token with
/// the best-effort value to allow continued parsing.
pub fn lex_char_literal(
    scanner: &mut Scanner,
    prefix: CharPrefix,
    diag: &mut DiagnosticEngine,
) -> Token {
    let file_id = scanner.file_id();
    let start_offset = scanner.byte_offset();

    // Safety check: we expect the scanner to be at the opening single-quote.
    if !scanner.advance_if('\'') {
        let span = Span::new(file_id, start_offset, scanner.byte_offset());
        diag.error(span, "expected opening single-quote for character literal");
        return Token::new(TokenKind::Error, span);
    }

    let mut value: u32 = 0;
    let mut char_count: usize = 0;

    loop {
        // Check for EOF
        if scanner.is_at_end() {
            let span = Span::new(file_id, start_offset, scanner.byte_offset());
            diag.error(span, "unterminated character literal");
            return Token::new(TokenKind::CharLiteral { value, prefix }, span);
        }

        match scanner.peek() {
            None => {
                // EOF before closing quote
                let span = Span::new(file_id, start_offset, scanner.byte_offset());
                diag.error(span, "unterminated character literal");
                return Token::new(TokenKind::CharLiteral { value, prefix }, span);
            }
            Some('\n') | Some('\r') => {
                // Newline in char literal — unterminated
                let span = Span::new(file_id, start_offset, scanner.byte_offset());
                diag.error(span, "unterminated character literal (missing closing ')");
                return Token::new(TokenKind::CharLiteral { value, prefix }, span);
            }
            Some('\'') => {
                // Closing single-quote — end of character literal
                scanner.advance();
                let span = Span::new(file_id, start_offset, scanner.byte_offset());

                if char_count == 0 {
                    diag.error(span, "empty character constant");
                } else if char_count > 1 {
                    diag.warning(span, "multi-character character constant");
                }

                // Validate character value range based on encoding prefix
                if char_count == 1 {
                    match prefix {
                        CharPrefix::None => {
                            // Plain char: value must fit in 8 bits
                            if value > 0xFF {
                                diag.warning(
                                    span,
                                    "character constant too large for type 'char'",
                                );
                                value &= 0xFF;
                            }
                        }
                        CharPrefix::L | CharPrefix::BigU => {
                            // wchar_t / char32_t: full 32-bit range is valid
                        }
                        CharPrefix::SmallU => {
                            // char16_t: value must fit in 16 bits
                            if value > 0xFFFF {
                                diag.warning(
                                    span,
                                    "character constant too large for type 'char16_t'",
                                );
                                value &= 0xFFFF;
                            }
                        }
                    }
                }

                return Token::new(TokenKind::CharLiteral { value, prefix }, span);
            }
            Some('\\') => {
                // Start of an escape sequence
                let backslash_offset = scanner.byte_offset();
                scanner.advance(); // consume the backslash

                let esc = process_escape(scanner, diag, file_id, backslash_offset);
                let char_val = match esc {
                    EscapeResult::Byte(b) => b as u32,
                    EscapeResult::CodePoint(cp) => cp,
                    EscapeResult::RawBytes(ref raw) => {
                        // Fallback: combine bytes into a single value
                        let mut v: u32 = 0;
                        for b in raw {
                            v = (v << 8) | (*b as u32);
                        }
                        v
                    }
                };

                // Multi-character constant: combine by left-shifting existing value
                if char_count > 0 {
                    value = (value << 8) | (char_val & 0xFF);
                } else {
                    value = char_val;
                }
                char_count += 1;
            }
            Some(ch) => {
                // Regular character
                let char_val = char_value(ch);
                scanner.advance();

                // Multi-character constant: combine by left-shifting existing value
                if char_count > 0 {
                    value = (value << 8) | (char_val & 0xFF);
                } else {
                    value = char_val;
                }
                char_count += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// String Prefix Detection
// ---------------------------------------------------------------------------

/// Detect if the current scanner position starts a string/char literal prefix.
///
/// When the scanner is positioned on `L`, `u`, or `U`, this function uses
/// save/restore lookahead to check whether the next character(s) form a
/// string or character literal prefix followed by a quote character:
///
/// | Prefix | Pattern                      |
/// |--------|------------------------------|
/// | `L`    | `L` followed by `"` or `'`   |
/// | `u8`   | `u` `8` followed by `"`      |
/// | `u`    | `u` followed by `"` or `'`   |
/// | `U`    | `U` followed by `"` or `'`   |
///
/// Note: `u8` prefix is valid only for string literals (C11), not char
/// literals, so `u8'x'` is NOT detected as a prefix.
///
/// The scanner position is saved and restored — this is a pure detection
/// function that does **not** advance the scanner. The caller is responsible
/// for advancing past the prefix characters before calling the appropriate
/// literal lexing function.
///
/// # Arguments
///
/// * `scanner` — Character scanner positioned at a potential prefix char.
///
/// # Returns
///
/// `Some(StringPrefix)` if a valid prefix is detected, `None` otherwise.
pub fn detect_string_prefix(scanner: &mut Scanner) -> Option<StringPrefix> {
    let mark: ScannerMark = scanner.mark();

    let result = match scanner.peek() {
        Some('L') => {
            scanner.advance();
            match scanner.peek() {
                Some('"') | Some('\'') => Some(StringPrefix::L),
                _ => None,
            }
        }
        Some('u') => {
            scanner.advance();
            if scanner.peek() == Some('8') {
                scanner.advance();
                // u8 prefix is only valid for string literals ("), not char literals (')
                match scanner.peek() {
                    Some('"') => Some(StringPrefix::U8),
                    _ => None,
                }
            } else {
                match scanner.peek() {
                    Some('"') | Some('\'') => Some(StringPrefix::SmallU),
                    _ => None,
                }
            }
        }
        Some('U') => {
            scanner.advance();
            match scanner.peek() {
                Some('"') | Some('\'') => Some(StringPrefix::BigU),
                _ => None,
            }
        }
        _ => None,
    };

    // Always restore scanner position — this is a non-consuming lookahead
    scanner.reset_to(mark);
    result
}

// ---------------------------------------------------------------------------
// Adjacent String Literal Concatenation
// ---------------------------------------------------------------------------

/// Resolve the encoding prefix for concatenated string literals.
///
/// Per C11 §6.4.5p5:
/// - If one literal has no prefix and the other does, the result takes the
///   non-empty prefix.
/// - If both have the same prefix (or both have no prefix), that prefix is used.
/// - If they have different non-empty prefixes, a warning is emitted and the
///   first prefix is used (this is a constraint violation in strict C11).
fn resolve_concat_prefix(
    first: StringPrefix,
    second: StringPrefix,
    diag: &mut DiagnosticEngine,
    span: Span,
) -> StringPrefix {
    match (first, second) {
        // Same prefix or trivial case
        (a, b) if std::mem::discriminant(&a) == std::mem::discriminant(&b) => a,
        // One is None — take the other
        (StringPrefix::None, b) => b,
        (a, StringPrefix::None) => a,
        // Different non-None prefixes — constraint violation
        (a, _b) => {
            diag.warning(
                span,
                "concatenation of string literals with different encoding prefixes",
            );
            a // First prefix wins
        }
    }
}

/// Skip whitespace characters and C-style comments between adjacent string
/// literals.
///
/// Advances the scanner past:
/// - Whitespace characters (spaces, tabs, newlines, carriage returns, etc.)
/// - Single-line comments (`// ...` to end of line)
/// - Multi-line comments (`/* ... */`)
///
/// This enables correct adjacent string concatenation for cases like:
/// ```c
/// "hello" /* comment */ "world"
/// "hello" // comment
/// "world"
/// ```
fn skip_whitespace_and_comments(scanner: &mut Scanner) {
    loop {
        // Skip any whitespace characters
        match scanner.peek() {
            Some(ch) if Scanner::is_whitespace(ch) => {
                scanner.advance();
                continue;
            }
            _ => {}
        }

        // Check for comments: // or /*
        if scanner.peek() == Some('/') {
            match scanner.peek_ahead(1) {
                Some('/') => {
                    // Single-line comment: skip to end of line
                    scanner.advance(); // consume first '/'
                    scanner.advance(); // consume second '/'
                    loop {
                        match scanner.peek() {
                            None | Some('\n') => break,
                            _ => {
                                scanner.advance();
                            }
                        }
                    }
                    // Consume the newline character if present
                    if scanner.peek() == Some('\n') {
                        scanner.advance();
                    }
                    continue;
                }
                Some('*') => {
                    // Multi-line comment: skip to closing */
                    scanner.advance(); // consume '/'
                    scanner.advance(); // consume '*'
                    loop {
                        match scanner.advance() {
                            None => break, // EOF inside comment — stop
                            Some('*') => {
                                if scanner.peek() == Some('/') {
                                    scanner.advance(); // consume closing '/'
                                    break;
                                }
                            }
                            _ => {} // skip character inside comment
                        }
                    }
                    continue;
                }
                _ => {} // Not a comment start
            }
        }

        // Nothing more to skip
        break;
    }
}

/// Attempt to concatenate adjacent string literals.
///
/// After lexing a string literal, this function looks ahead (skipping
/// whitespace and comments) for another string literal. If found, the
/// literals are concatenated into a single `StringLiteral` token whose
/// byte vector is the concatenation of all adjacent strings and whose
/// span covers from the first string to the last.
///
/// This implements C11 §6.4.5 (translation phase 6) adjacent string
/// literal concatenation at the lexer level.
///
/// # Arguments
///
/// * `scanner` — Character scanner positioned after the first string literal.
/// * `first` — The first string literal token already lexed.
/// * `diag` — Diagnostic engine for warnings about prefix conflicts.
///
/// # Returns
///
/// A single `Token` with kind `TokenKind::StringLiteral` containing the
/// concatenated byte content. If no adjacent string is found, returns
/// the original token unchanged. If the input is not a `StringLiteral`
/// token, it is returned as-is.
pub fn try_concatenate_strings(
    scanner: &mut Scanner,
    first: Token,
    diag: &mut DiagnosticEngine,
) -> Token {
    // Extract the value and prefix from the first token; if it is not a
    // string literal, return it unchanged.
    let (mut value, mut prefix, mut span) = match first.kind {
        TokenKind::StringLiteral { value, prefix } => (value, prefix, first.span),
        _ => return first,
    };

    loop {
        // Save scanner position for backtracking if no adjacent string found
        let mark: ScannerMark = scanner.mark();

        // Skip whitespace and comments between potential adjacent strings
        skip_whitespace_and_comments(scanner);

        // Determine if the next token starts another string literal.
        // We check for: `"`, `L"`, `u"`, `u8"`, `U"`.
        let next_prefix = match scanner.peek() {
            Some('"') => Some(StringPrefix::None),
            Some('L') => {
                if scanner.peek_ahead(1) == Some('"') {
                    Some(StringPrefix::L)
                } else {
                    None
                }
            }
            Some('u') => {
                if scanner.peek_ahead(1) == Some('8')
                    && scanner.peek_ahead(2) == Some('"')
                {
                    Some(StringPrefix::U8)
                } else if scanner.peek_ahead(1) == Some('"') {
                    Some(StringPrefix::SmallU)
                } else {
                    None
                }
            }
            Some('U') => {
                if scanner.peek_ahead(1) == Some('"') {
                    Some(StringPrefix::BigU)
                } else {
                    None
                }
            }
            _ => None,
        };

        match next_prefix {
            None => {
                // No adjacent string literal found — restore position and stop
                scanner.reset_to(mark);
                break;
            }
            Some(adj_prefix) => {
                // Advance the scanner past the prefix characters
                match adj_prefix {
                    StringPrefix::None => {
                        // No prefix chars to consume — scanner is at the '"'
                    }
                    StringPrefix::L | StringPrefix::SmallU | StringPrefix::BigU => {
                        scanner.advance(); // consume the single prefix char
                    }
                    StringPrefix::U8 => {
                        scanner.advance(); // consume 'u'
                        scanner.advance(); // consume '8'
                    }
                }

                // Lex the adjacent string literal (scanner is now at '"')
                let next_token = lex_string_literal(scanner, adj_prefix, diag);

                match next_token.kind {
                    TokenKind::StringLiteral {
                        value: next_value,
                        prefix: next_pfx,
                    } => {
                        // Resolve prefix for the concatenation
                        prefix = resolve_concat_prefix(prefix, next_pfx, diag, span);
                        // Concatenate byte content
                        value.extend_from_slice(&next_value);
                        // Merge spans to cover the full range
                        span = Span::merge(span, next_token.span);
                    }
                    _ => {
                        // Error during adjacent string lexing — stop concatenation
                        break;
                    }
                }
            }
        }
    }

    Token::new(TokenKind::StringLiteral { value, prefix }, span)
}
