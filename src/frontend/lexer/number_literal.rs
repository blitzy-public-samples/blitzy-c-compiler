//! Numeric literal parsing for the BCC C11 lexer (Phase 3).
//!
//! Implements lexing of all C11 §6.4.4 numeric literal forms:
//!
//! - **Decimal integers**: `42`, `123456`
//! - **Hexadecimal integers**: `0xFF`, `0X1A`
//! - **Octal integers**: `0755`, `0177`
//! - **Binary integers** (GCC extension): `0b1010`, `0B1111`
//! - **Integer suffixes**: `u`/`U`, `l`/`L`, `ll`/`LL`, and all valid
//!   combinations (`ul`, `lu`, `ull`, `llu`, etc.) per C11 §6.4.4.1
//! - **Decimal floats**: `3.14`, `1e10`, `1.5e-3`, `.5`
//! - **Hex floats**: `0x1.0p+0`, `0x1.fp10`, `0xAp-2`
//! - **Float suffixes**: `f`/`F` (float), `l`/`L` (long double)
//!
//! # Entry Point
//!
//! [`lex_number`] is the sole public function. The main lexer calls it when it
//! encounters a leading digit character (`0`–`9`). The scanner must be
//! positioned at that first digit on entry.
//!
//! # Error Handling
//!
//! All malformed input (invalid suffix, invalid digit for base, missing
//! exponent digits, overflow) is diagnosed through [`DiagnosticEngine`] with
//! precise [`Span`] locations. The function never panics on malformed input;
//! it always returns a valid [`TokenKind`] (possibly [`TokenKind::Error`])
//! for error recovery.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules (`super::token`, `super::scanner`, `crate::common::diagnostics`).

use super::scanner::Scanner;
use super::token::{FloatSuffix, IntegerSuffix, NumericBase, TokenKind};
use crate::common::diagnostics::{DiagnosticEngine, Span};

// ---------------------------------------------------------------------------
// Helper: digit value conversion
// ---------------------------------------------------------------------------

/// Convert an ASCII digit character to its numeric value (0–15).
///
/// Supports decimal (`0`–`9`) and hexadecimal (`a`–`f`, `A`–`F`) digits.
/// Returns `0` for any character outside these ranges (caller-validated).
#[inline]
fn digit_value(ch: char) -> u64 {
    match ch {
        '0'..='9' => (ch as u64) - ('0' as u64),
        'a'..='f' => (ch as u64) - ('a' as u64) + 10,
        'A'..='F' => (ch as u64) - ('A' as u64) + 10,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Helper: parse integer value from digit slice
// ---------------------------------------------------------------------------

/// Parse a sequence of digit characters into a `u64` with the given radix.
///
/// Digits that are invalid for the given radix (e.g., `8` in an octal literal)
/// are silently skipped — the caller has already diagnosed them. Returns a
/// tuple of `(value, overflowed)` where `overflowed` is `true` if the result
/// exceeded `u64::MAX` (in which case `value` is saturated to `u64::MAX`).
fn parse_integer_value(digits: &str, radix: u32) -> (u64, bool) {
    let mut value: u64 = 0;
    let mut overflowed = false;
    let radix_u64 = radix as u64;

    for ch in digits.chars() {
        let d = digit_value(ch);
        // Skip digits invalid for this radix (already diagnosed by caller).
        if d >= radix_u64 {
            continue;
        }
        match value.checked_mul(radix_u64) {
            Some(v) => match v.checked_add(d) {
                Some(v2) => value = v2,
                None => {
                    overflowed = true;
                    value = u64::MAX;
                }
            },
            None => {
                overflowed = true;
                value = u64::MAX;
            }
        }
    }

    (value, overflowed)
}

// ---------------------------------------------------------------------------
// Suffix parsing: integer
// ---------------------------------------------------------------------------

/// Parse an integer type suffix from the current scanner position.
///
/// Recognises all valid C11 §6.4.4.1 suffix combinations:
///
/// | Suffix pattern       | Result              |
/// |----------------------|---------------------|
/// | (none)               | `IntegerSuffix::None` |
/// | `u` / `U`            | `IntegerSuffix::U`    |
/// | `l` / `L`            | `IntegerSuffix::L`    |
/// | `ul` / `UL` / `lu` … | `IntegerSuffix::UL`   |
/// | `ll` / `LL`          | `IntegerSuffix::LL`   |
/// | `ull` / `ULL` / `llu` … | `IntegerSuffix::ULL` |
///
/// Mixed-case `ll` (e.g., `lL`, `Ll`) is diagnosed as an error but recovered
/// to the best-matching suffix. Invalid trailing characters are left for the
/// caller's [`check_trailing_invalid_chars`] to diagnose.
fn parse_integer_suffix(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> IntegerSuffix {
    let suffix_start = scanner.offset();

    match scanner.peek() {
        // ---- unsigned prefix: u/U ----
        Some('u') | Some('U') => {
            scanner.advance(); // consume u/U
            match scanner.peek() {
                Some(l1 @ 'l') | Some(l1 @ 'L') => {
                    scanner.advance(); // consume first l/L
                    match scanner.peek() {
                        Some(l2) if l2 == l1 => {
                            // ull / ULL — matched case
                            scanner.advance();
                            IntegerSuffix::ULL
                        }
                        Some('l') | Some('L') => {
                            // Mixed case: uLl or ulL — INVALID but recover as ULL
                            scanner.advance();
                            let span = Span::new(file_id, suffix_start, scanner.offset());
                            diagnostics.emit_error(
                                span,
                                "invalid integer suffix: mixed-case 'lL' is not allowed; \
                                 use 'll' or 'LL'",
                            );
                            IntegerSuffix::ULL
                        }
                        _ => IntegerSuffix::UL,
                    }
                }
                _ => IntegerSuffix::U,
            }
        }

        // ---- long prefix: l/L ----
        Some(l1 @ 'l') | Some(l1 @ 'L') => {
            scanner.advance(); // consume first l/L
            match scanner.peek() {
                Some(l2) if l2 == l1 => {
                    // ll or LL — matched case
                    scanner.advance();
                    // Optional trailing u/U for llu / LLU
                    match scanner.peek() {
                        Some('u') | Some('U') => {
                            scanner.advance();
                            IntegerSuffix::ULL
                        }
                        _ => IntegerSuffix::LL,
                    }
                }
                Some('l') | Some('L') => {
                    // Mixed case: lL or Ll — INVALID but recover as LL
                    scanner.advance();
                    // Check for trailing u/U
                    let has_u = match scanner.peek() {
                        Some('u') | Some('U') => {
                            scanner.advance();
                            true
                        }
                        _ => false,
                    };
                    let span = Span::new(file_id, suffix_start, scanner.offset());
                    diagnostics.emit_error(
                        span,
                        "invalid integer suffix: mixed-case 'lL' is not allowed; \
                         use 'll' or 'LL'",
                    );
                    if has_u {
                        IntegerSuffix::ULL
                    } else {
                        IntegerSuffix::LL
                    }
                }
                Some('u') | Some('U') => {
                    // lu / LU
                    scanner.advance();
                    IntegerSuffix::UL
                }
                _ => IntegerSuffix::L,
            }
        }

        // ---- no suffix ----
        _ => IntegerSuffix::None,
    }
}

// ---------------------------------------------------------------------------
// Suffix parsing: float
// ---------------------------------------------------------------------------

/// Parse a float type suffix from the current scanner position.
///
/// - `f` / `F` → [`FloatSuffix::F`] (float)
/// - `l` / `L` → [`FloatSuffix::L`] (long double)
/// - `i` / `j` → [`FloatSuffix::I`] (imaginary double, GCC extension)
/// - `fi` / `if` / `fj` / `jf` → [`FloatSuffix::FI`] (imaginary float)
/// - `li` / `il` / `lj` / `jl` → [`FloatSuffix::LI`] (imaginary long double)
/// - (nothing) → [`FloatSuffix::None`] (double)
fn parse_float_suffix(scanner: &mut Scanner) -> FloatSuffix {
    match scanner.peek() {
        Some('f') | Some('F') => {
            scanner.advance();
            // Check for trailing imaginary suffix: fi / fj
            match scanner.peek() {
                Some('i') | Some('j') | Some('I') | Some('J') => {
                    scanner.advance();
                    FloatSuffix::FI
                }
                _ => FloatSuffix::F,
            }
        }
        Some('l') | Some('L') => {
            scanner.advance();
            // Check for trailing imaginary suffix: li / lj
            match scanner.peek() {
                Some('i') | Some('j') | Some('I') | Some('J') => {
                    scanner.advance();
                    FloatSuffix::LI
                }
                _ => FloatSuffix::L,
            }
        }
        Some('i') | Some('j') | Some('I') | Some('J') => {
            scanner.advance();
            // Check for trailing type suffix: if / il
            match scanner.peek() {
                Some('f') | Some('F') => {
                    scanner.advance();
                    FloatSuffix::FI
                }
                Some('l') | Some('L') => {
                    scanner.advance();
                    FloatSuffix::LI
                }
                _ => FloatSuffix::I,
            }
        }
        _ => FloatSuffix::None,
    }
}

// ---------------------------------------------------------------------------
// Trailing invalid character detection
// ---------------------------------------------------------------------------

/// Check for and diagnose invalid trailing characters after a numeric literal
/// and its suffix.
///
/// In C, a numeric literal immediately followed by identifier-like characters
/// (e.g., `42abc`, `0xFFgh`) is ill-formed. This function consumes all such
/// trailing characters and emits a single diagnostic error.
fn check_trailing_invalid_chars(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    _literal_start: u32,
) {
    if let Some(ch) = scanner.peek() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            let bad_start = scanner.offset();
            // Consume all contiguous identifier-like characters.
            while let Some(c) = scanner.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    scanner.advance();
                } else {
                    break;
                }
            }
            let span = Span::new(file_id, bad_start, scanner.offset());
            diagnostics.emit_error(span, "invalid suffix on numeric literal");
        }
    }
}

/// GCC extension: Convert an integer literal to a float imaginary literal if
/// the next character is 'i' or 'j'. For example, `200i` is equivalent to
/// `200.0i` and represents a `_Complex double` imaginary constant.
///
/// Returns the original token unchanged if no imaginary suffix follows.
fn maybe_convert_to_imaginary(scanner: &mut Scanner, token: TokenKind) -> TokenKind {
    match scanner.peek() {
        Some('i') | Some('j') | Some('I') | Some('J') => {
            // Consume the imaginary suffix
            scanner.advance();
            // Check for additional type suffix after i/j: fi/li
            let suffix = match scanner.peek() {
                Some('f') | Some('F') => {
                    scanner.advance();
                    FloatSuffix::FI
                }
                Some('l') | Some('L') => {
                    scanner.advance();
                    FloatSuffix::LI
                }
                _ => FloatSuffix::I,
            };
            // Extract the integer value and convert to float string
            let value_str = match &token {
                TokenKind::IntegerLiteral { value, .. } => {
                    format!("{}.0", value)
                }
                _ => "0.0".to_string(),
            };
            TokenKind::FloatLiteral {
                value: value_str,
                suffix,
                base: NumericBase::Decimal,
            }
        }
        _ => token,
    }
}

// ===========================================================================
// Main entry point
// ===========================================================================

/// Lex a numeric literal starting at the current scanner position.
///
/// The scanner **must** be positioned at the first digit character (`0`–`9`).
/// On return, the scanner is positioned immediately after the last character
/// of the literal (including any type suffix).
///
/// # Returns
///
/// - [`TokenKind::IntegerLiteral`] for integer constants.
/// - [`TokenKind::FloatLiteral`] for floating-point constants.
/// - [`TokenKind::Error`] if the input is so malformed that no reasonable
///   literal can be produced (e.g., EOF at the start).
///
/// # Diagnostics
///
/// Errors and warnings are emitted through `diagnostics` rather than panics:
/// - Invalid suffix combinations (e.g., `123abc`)
/// - Invalid digit for the detected base (e.g., `09`, `0b2`)
/// - Missing digits after base prefix (e.g., `0x`, `0b`)
/// - Missing exponent digits (e.g., `1e`, `0x1.0p`)
/// - Integer overflow past `u64::MAX`
pub fn lex_number(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
) -> TokenKind {
    // Record the start position for Span construction.
    let start_pos = scanner.position();
    let start_offset = start_pos.offset;

    // Guard: the scanner must have at least one character.
    if scanner.is_eof() {
        let span = Span::new(file_id, start_offset, start_offset);
        diagnostics.emit_error(span, "unexpected end of file in numeric literal");
        return TokenKind::Error;
    }

    let first_char = match scanner.peek() {
        Some(ch) if ch.is_ascii_digit() => ch,
        _ => {
            let span = Span::new(file_id, start_offset, start_offset + 1);
            diagnostics.emit_error(span, "expected digit in numeric literal");
            return TokenKind::Error;
        }
    };

    if first_char == '0' {
        // Use peek_nth to inspect the character *after* '0' before consuming.
        let second_char = scanner.peek_nth(1);
        scanner.advance(); // consume '0'

        match second_char {
            Some('x') | Some('X') => {
                scanner.advance(); // consume 'x' / 'X'
                lex_hex_literal(scanner, diagnostics, file_id, start_offset)
            }
            Some('b') | Some('B') => {
                scanner.advance(); // consume 'b' / 'B'
                lex_binary_literal(scanner, diagnostics, file_id, start_offset)
            }
            _ => {
                // Octal integer, plain `0`, or decimal float starting with `0`.
                lex_after_leading_zero(scanner, diagnostics, file_id, start_offset)
            }
        }
    } else {
        // Decimal literal starting with 1–9 (first digit NOT yet consumed).
        lex_decimal_literal(scanner, diagnostics, file_id, start_offset)
    }
}

// ===========================================================================
// Hexadecimal literal
// ===========================================================================

/// Lex a hexadecimal integer or hex float literal.
///
/// Called after `0x` / `0X` has been consumed. The scanner is positioned at
/// the first character after the prefix.
fn lex_hex_literal(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    let digits_start = scanner.offset();
    let mut has_int_digits = false;

    // Consume hexadecimal digit sequence before the (optional) dot.
    while scanner
        .consume_if_pred(|ch| ch.is_ascii_hexdigit())
        .is_some()
    {
        has_int_digits = true;
    }

    // Check for hex float transition: `.` or `p`/`P`.
    match scanner.peek() {
        Some('.') => {
            scanner.advance(); // consume '.'

            // Consume hex digits after the dot.
            let mut has_frac_digits = false;
            while scanner
                .consume_if_pred(|ch| ch.is_ascii_hexdigit())
                .is_some()
            {
                has_frac_digits = true;
            }

            if !has_int_digits && !has_frac_digits {
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(
                    span,
                    "hexadecimal floating literal requires at least one hex digit",
                );
            }

            // Hex float MUST have a `p`/`P` exponent.
            return lex_hex_float_exponent(scanner, diagnostics, file_id, start_offset);
        }
        Some('p') | Some('P') => {
            if !has_int_digits {
                let span = Span::new(file_id, start_offset, scanner.offset());
                diagnostics.emit_error(
                    span,
                    "hexadecimal floating literal requires at least one hex digit \
                     before 'p' exponent",
                );
            }
            return lex_hex_float_exponent(scanner, diagnostics, file_id, start_offset);
        }
        _ => { /* fall through to hex integer */ }
    }

    // ---- Hex integer ----
    if !has_int_digits {
        let span = Span::new(file_id, start_offset, scanner.offset());
        diagnostics.emit_error(span, "no digits after '0x' in hexadecimal literal");
        return TokenKind::IntegerLiteral {
            value: 0,
            suffix: IntegerSuffix::None,
            base: NumericBase::Hexadecimal,
        };
    }

    let digits_end = scanner.offset();
    let digit_str = scanner.slice(digits_start as usize, digits_end as usize);
    let (value, overflowed) = parse_integer_value(digit_str, 16);

    if overflowed {
        let span = Span::new(file_id, start_offset, digits_end);
        diagnostics.emit_warning(
            span,
            "integer literal is too large for type 'unsigned long long'",
        );
    }

    let suffix = parse_integer_suffix(scanner, diagnostics, file_id);
    let int_tok = TokenKind::IntegerLiteral {
        value,
        suffix,
        base: NumericBase::Hexadecimal,
    };
    let tok = maybe_convert_to_imaginary(scanner, int_tok);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    tok
}

// ---------------------------------------------------------------------------
// Hex float exponent
// ---------------------------------------------------------------------------

/// Lex the exponent part of a hex float literal (`p`/`P` followed by decimal
/// exponent digits), then return the complete [`TokenKind::FloatLiteral`].
///
/// Called after the hex significand (integer + optional fractional part) has
/// been consumed. The scanner is positioned at the `p`/`P` character (or at
/// the point where `p`/`P` is expected if missing).
fn lex_hex_float_exponent(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    // Consume required `p`/`P`.
    match scanner.peek() {
        Some('p') | Some('P') => {
            scanner.advance();
        }
        _ => {
            // Missing exponent — hex float requires it.
            let span = Span::new(file_id, start_offset, scanner.offset());
            diagnostics.emit_error(
                span,
                "hexadecimal floating literal requires 'p' or 'P' exponent",
            );
            let before_suffix = scanner.offset();
            let suffix = parse_float_suffix(scanner);
            check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
            let raw = scanner
                .slice(start_offset as usize, before_suffix as usize)
                .to_string();
            return TokenKind::FloatLiteral {
                value: raw,
                suffix,
                base: NumericBase::Hexadecimal,
            };
        }
    }

    // Optional sign.
    let _ = scanner.consume_if('+') || scanner.consume_if('-');

    // Decimal exponent digits (mandatory).
    let mut has_exp_digits = false;
    while scanner.consume_if_pred(|ch| ch.is_ascii_digit()).is_some() {
        has_exp_digits = true;
    }

    if !has_exp_digits {
        let span = Span::new(file_id, start_offset, scanner.offset());
        diagnostics.emit_error(
            span,
            "expected decimal digits after exponent in hexadecimal floating literal",
        );
    }

    let before_suffix = scanner.offset();
    let suffix = parse_float_suffix(scanner);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    let raw = scanner
        .slice(start_offset as usize, before_suffix as usize)
        .to_string();

    TokenKind::FloatLiteral {
        value: raw,
        suffix,
        base: NumericBase::Hexadecimal,
    }
}

// ===========================================================================
// Binary literal (GCC extension)
// ===========================================================================

/// Lex a binary integer literal (GCC extension).
///
/// Called after `0b` / `0B` has been consumed. The scanner is positioned at
/// the first character after the prefix.
fn lex_binary_literal(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    let digits_start = scanner.offset();
    let mut has_digits = false;

    // Consume binary digits (0 and 1).
    loop {
        match scanner.peek() {
            Some('0') | Some('1') => {
                scanner.advance();
                has_digits = true;
            }
            Some(ch) if ch.is_ascii_digit() => {
                // Invalid digit for binary (2–9) — diagnose but consume for
                // error recovery.
                let bad_pos = scanner.offset();
                scanner.advance();
                let span = Span::new(file_id, bad_pos, bad_pos + 1);
                diagnostics.emit_error(span, format!("invalid digit '{}' in binary literal", ch));
                has_digits = true; // still count as "having something"
            }
            _ => break,
        }
    }

    if !has_digits {
        let span = Span::new(file_id, start_offset, scanner.offset());
        diagnostics.emit_error(span, "no digits after '0b' in binary literal");
        return TokenKind::IntegerLiteral {
            value: 0,
            suffix: IntegerSuffix::None,
            base: NumericBase::Binary,
        };
    }

    let digits_end = scanner.offset();
    let digit_str = scanner.slice(digits_start as usize, digits_end as usize);
    let (value, overflowed) = parse_integer_value(digit_str, 2);

    if overflowed {
        let span = Span::new(file_id, start_offset, digits_end);
        diagnostics.emit_warning(
            span,
            "integer literal is too large for type 'unsigned long long'",
        );
    }

    let suffix = parse_integer_suffix(scanner, diagnostics, file_id);
    let int_tok = TokenKind::IntegerLiteral {
        value,
        suffix,
        base: NumericBase::Binary,
    };
    let tok = maybe_convert_to_imaginary(scanner, int_tok);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    tok
}

// ===========================================================================
// Octal / decimal-from-zero
// ===========================================================================

/// Lex a number that starts with `0` but is NOT hex or binary.
///
/// Possible results:
/// - **Octal integer**: `0`, `0755`, `0177`
/// - **Decimal float**: `0.5`, `0e10`, `0123.5` (octal prefix + `.` → decimal
///   float)
///
/// Called after the leading `0` has been consumed. The scanner is positioned
/// at the character immediately after `0`.
fn lex_after_leading_zero(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    let mut has_non_octal_digit = false;

    // Consume digits (0–9). Track whether any digit is 8 or 9 (invalid in
    // octal but valid if the literal transitions to a decimal float).
    while let Some(ch) = scanner.consume_if_pred(|c| c.is_ascii_digit()) {
        if ch >= '8' {
            has_non_octal_digit = true;
        }
    }

    // Check for decimal float transition via `.` or `e`/`E`.
    match scanner.peek() {
        Some('.') => {
            scanner.advance(); // consume '.'
                               // This is now a decimal float (e.g., `0.5`, `0123.5`).
            return lex_decimal_float_after_dot(scanner, diagnostics, file_id, start_offset);
        }
        Some('e') | Some('E') => {
            // Decimal float with exponent (e.g., `0e10`, `0123e5`).
            return lex_decimal_exponent(scanner, diagnostics, file_id, start_offset);
        }
        _ => { /* octal integer */ }
    }

    // ---- Octal integer (or plain `0`) ----
    if has_non_octal_digit {
        let span = Span::new(file_id, start_offset, scanner.offset());
        diagnostics.emit_error(span, "invalid digit in octal literal");
    }

    // Parse the octal value from the digit string (including leading '0').
    let digits_end = scanner.offset();
    let all_text = scanner.slice(start_offset as usize, digits_end as usize);

    // `all_text` is "0", "0755", "08", etc. Skip the leading '0' for parsing
    // since it doesn't contribute to the value (radix is already 8).
    let octal_digits = &all_text[1..];

    let (value, overflowed) = if octal_digits.is_empty() {
        // Just the literal `0`.
        (0u64, false)
    } else {
        parse_integer_value(octal_digits, 8)
    };

    if overflowed {
        let span = Span::new(file_id, start_offset, digits_end);
        diagnostics.emit_warning(
            span,
            "integer literal is too large for type 'unsigned long long'",
        );
    }

    let suffix = parse_integer_suffix(scanner, diagnostics, file_id);
    let int_tok = TokenKind::IntegerLiteral {
        value,
        suffix,
        base: NumericBase::Octal,
    };
    let tok = maybe_convert_to_imaginary(scanner, int_tok);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    tok
}

// ===========================================================================
// Decimal literal
// ===========================================================================

/// Lex a decimal integer or decimal float literal starting with `1`–`9`.
///
/// Called when the first digit is NOT `0`. The first digit has **not** been
/// consumed yet — the scanner is positioned at it.
fn lex_decimal_literal(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    // Consume all leading decimal digits (including the first 1–9 digit).
    while scanner.consume_if_pred(|ch| ch.is_ascii_digit()).is_some() {}

    // Check for float transition via `.` or `e`/`E`.
    match scanner.peek() {
        Some('.') => {
            scanner.advance(); // consume '.'
            return lex_decimal_float_after_dot(scanner, diagnostics, file_id, start_offset);
        }
        Some('e') | Some('E') => {
            return lex_decimal_exponent(scanner, diagnostics, file_id, start_offset);
        }
        _ => { /* decimal integer */ }
    }

    // ---- Decimal integer ----
    let digits_end = scanner.offset();
    let digit_str = scanner.slice(start_offset as usize, digits_end as usize);
    let (value, overflowed) = parse_integer_value(digit_str, 10);

    if overflowed {
        let span = Span::new(file_id, start_offset, digits_end);
        diagnostics.emit_warning(
            span,
            "integer literal is too large for type 'unsigned long long'",
        );
    }

    let suffix = parse_integer_suffix(scanner, diagnostics, file_id);
    let int_tok = TokenKind::IntegerLiteral {
        value,
        suffix,
        base: NumericBase::Decimal,
    };
    let tok = maybe_convert_to_imaginary(scanner, int_tok);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    tok
}

// ===========================================================================
// Decimal float helpers
// ===========================================================================

/// Lex the fractional and optional exponent part of a decimal float literal.
///
/// Called after the integer part and `.` have already been consumed (e.g.,
/// after `3.` or `0.`). The scanner is positioned at the first character
/// after the dot.
fn lex_decimal_float_after_dot(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    // Consume fractional decimal digits.
    while scanner.consume_if_pred(|ch| ch.is_ascii_digit()).is_some() {}

    // Optional exponent part.
    if matches!(scanner.peek(), Some('e') | Some('E')) {
        scanner.advance(); // consume 'e' / 'E'

        // Optional sign.
        let _ = scanner.consume_if('+') || scanner.consume_if('-');

        // Exponent decimal digits (mandatory after 'e'/'E').
        let mut has_exp_digits = false;
        while scanner.consume_if_pred(|ch| ch.is_ascii_digit()).is_some() {
            has_exp_digits = true;
        }

        if !has_exp_digits {
            let span = Span::new(file_id, start_offset, scanner.offset());
            diagnostics.emit_error(
                span,
                "expected digits after exponent in floating-point literal",
            );
        }
    }

    let before_suffix = scanner.offset();
    let suffix = parse_float_suffix(scanner);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    let raw = scanner
        .slice(start_offset as usize, before_suffix as usize)
        .to_string();

    TokenKind::FloatLiteral {
        value: raw,
        suffix,
        base: NumericBase::Decimal,
    }
}

/// Lex a decimal float literal whose exponent is the next thing to parse.
///
/// Called after the integer part has been consumed and `e`/`E` is the next
/// character. This handles numbers like `1e10`, `0e5`, `42E-3`.
fn lex_decimal_exponent(
    scanner: &mut Scanner,
    diagnostics: &mut DiagnosticEngine,
    file_id: u32,
    start_offset: u32,
) -> TokenKind {
    // Consume 'e' / 'E'.
    scanner.advance();

    // Optional sign.
    let _ = scanner.consume_if('+') || scanner.consume_if('-');

    // Exponent decimal digits (mandatory).
    let mut has_exp_digits = false;
    while scanner.consume_if_pred(|ch| ch.is_ascii_digit()).is_some() {
        has_exp_digits = true;
    }

    if !has_exp_digits {
        let span = Span::new(file_id, start_offset, scanner.offset());
        diagnostics.emit_error(
            span,
            "expected digits after exponent in floating-point literal",
        );
    }

    let before_suffix = scanner.offset();
    let suffix = parse_float_suffix(scanner);
    check_trailing_invalid_chars(scanner, diagnostics, file_id, start_offset);
    let raw = scanner
        .slice(start_offset as usize, before_suffix as usize)
        .to_string();

    TokenKind::FloatLiteral {
        value: raw,
        suffix,
        base: NumericBase::Decimal,
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::frontend::lexer::scanner::Scanner;
    use crate::frontend::lexer::token::{FloatSuffix, IntegerSuffix, NumericBase, TokenKind};

    /// Helper: lex a number from the given source string and return the
    /// `TokenKind` plus any diagnostics emitted.
    fn lex(src: &str) -> (TokenKind, DiagnosticEngine) {
        let mut scanner = Scanner::new(src);
        let mut diag = DiagnosticEngine::new();
        let tok = lex_number(&mut scanner, &mut diag, 0);
        (tok, diag)
    }

    // -- Decimal integers --------------------------------------------------

    #[test]
    fn test_decimal_zero() {
        let (tok, diag) = lex("0");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
                base: NumericBase::Octal, // plain '0' is technically octal
            }
        );
    }

    #[test]
    fn test_decimal_integer() {
        let (tok, diag) = lex("42");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 42,
                suffix: IntegerSuffix::None,
                base: NumericBase::Decimal,
            }
        );
    }

    #[test]
    fn test_decimal_large() {
        let (tok, diag) = lex("123456789");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 123456789,
                suffix: IntegerSuffix::None,
                base: NumericBase::Decimal,
            }
        );
    }

    // -- Hexadecimal integers ----------------------------------------------

    #[test]
    fn test_hex_integer() {
        let (tok, diag) = lex("0xFF");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 255,
                suffix: IntegerSuffix::None,
                base: NumericBase::Hexadecimal,
            }
        );
    }

    #[test]
    fn test_hex_upper_prefix() {
        let (tok, diag) = lex("0X1A");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 26,
                suffix: IntegerSuffix::None,
                base: NumericBase::Hexadecimal,
            }
        );
    }

    #[test]
    fn test_hex_no_digits() {
        let (tok, diag) = lex("0x ");
        assert!(diag.has_errors());
        // Still produces a fallback token.
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
                base: NumericBase::Hexadecimal,
            }
        );
    }

    // -- Octal integers ----------------------------------------------------

    #[test]
    fn test_octal_integer() {
        let (tok, diag) = lex("0755");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 493, // 7*64 + 5*8 + 5
                suffix: IntegerSuffix::None,
                base: NumericBase::Octal,
            }
        );
    }

    #[test]
    fn test_octal_invalid_digit() {
        let (tok, diag) = lex("08");
        assert!(diag.has_errors());
        // The invalid digit is diagnosed; value recovery skips '8'.
        match tok {
            TokenKind::IntegerLiteral { base, .. } => {
                assert_eq!(base, NumericBase::Octal);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    // -- Binary integers ---------------------------------------------------

    #[test]
    fn test_binary_integer() {
        let (tok, diag) = lex("0b1010");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 10,
                suffix: IntegerSuffix::None,
                base: NumericBase::Binary,
            }
        );
    }

    #[test]
    fn test_binary_upper_prefix() {
        let (tok, diag) = lex("0B1111");
        assert!(!diag.has_errors());
        assert_eq!(
            tok,
            TokenKind::IntegerLiteral {
                value: 15,
                suffix: IntegerSuffix::None,
                base: NumericBase::Binary,
            }
        );
    }

    #[test]
    fn test_binary_no_digits() {
        let (tok, diag) = lex("0b ");
        assert!(diag.has_errors());
        match tok {
            TokenKind::IntegerLiteral { base, .. } => {
                assert_eq!(base, NumericBase::Binary);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    // -- Integer suffixes --------------------------------------------------

    #[test]
    fn test_suffix_u() {
        let (tok, _) = lex("42u");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::U);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_upper_u() {
        let (tok, _) = lex("42U");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::U);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_l() {
        let (tok, _) = lex("42l");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::L);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_ul() {
        let (tok, _) = lex("42ul");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::UL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_lu() {
        let (tok, _) = lex("42lu");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::UL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_ll() {
        let (tok, _) = lex("42ll");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::LL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_ull() {
        let (tok, _) = lex("42ull");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::ULL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_llu() {
        let (tok, _) = lex("42llu");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::ULL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_upper_ll() {
        let (tok, _) = lex("42LL");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::LL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_upper_ull() {
        let (tok, _) = lex("42ULL");
        match tok {
            TokenKind::IntegerLiteral { suffix, .. } => {
                assert_eq!(suffix, IntegerSuffix::ULL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_mixed_case_ll_invalid() {
        let (_, diag) = lex("42lL");
        assert!(diag.has_errors());
    }

    // -- Decimal floats ----------------------------------------------------

    #[test]
    fn test_decimal_float_basic() {
        let (tok, diag) = lex("3.14");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value,
                suffix,
                base,
            } => {
                assert_eq!(value, "3.14");
                assert_eq!(suffix, FloatSuffix::None);
                assert_eq!(base, NumericBase::Decimal);
            }
            _ => panic!("expected FloatLiteral, got {:?}", tok),
        }
    }

    #[test]
    fn test_decimal_float_exponent() {
        let (tok, diag) = lex("1e10");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value, base, ..
            } => {
                assert_eq!(value, "1e10");
                assert_eq!(base, NumericBase::Decimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_decimal_float_negative_exp() {
        let (tok, diag) = lex("1.5e-3");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value,
                suffix,
                base,
            } => {
                assert_eq!(value, "1.5e-3");
                assert_eq!(suffix, FloatSuffix::None);
                assert_eq!(base, NumericBase::Decimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_decimal_float_suffix_f() {
        let (tok, diag) = lex("1.0f");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral { suffix, .. } => {
                assert_eq!(suffix, FloatSuffix::F);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_decimal_float_suffix_l() {
        let (tok, diag) = lex("1.0L");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral { suffix, .. } => {
                assert_eq!(suffix, FloatSuffix::L);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_zero_dot() {
        let (tok, diag) = lex("0.");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value, base, ..
            } => {
                assert_eq!(value, "0.");
                assert_eq!(base, NumericBase::Decimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    // -- Hex floats --------------------------------------------------------

    #[test]
    fn test_hex_float_basic() {
        let (tok, diag) = lex("0x1.0p0");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value,
                suffix,
                base,
            } => {
                assert_eq!(value, "0x1.0p0");
                assert_eq!(suffix, FloatSuffix::None);
                assert_eq!(base, NumericBase::Hexadecimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_hex_float_with_exp() {
        let (tok, diag) = lex("0x1.fp10");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value, base, ..
            } => {
                assert_eq!(value, "0x1.fp10");
                assert_eq!(base, NumericBase::Hexadecimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_hex_float_no_dot() {
        let (tok, diag) = lex("0xAp-2");
        assert!(!diag.has_errors());
        match tok {
            TokenKind::FloatLiteral {
                ref value, base, ..
            } => {
                assert_eq!(value, "0xAp-2");
                assert_eq!(base, NumericBase::Hexadecimal);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    // -- Overflow ----------------------------------------------------------

    #[test]
    fn test_integer_overflow_warning() {
        // u64::MAX + 1
        let (_, diag) = lex("18446744073709551616");
        assert!(diag.warning_count() > 0);
    }

    // -- Invalid suffix on integer -----------------------------------------

    #[test]
    fn test_invalid_suffix() {
        let (_, diag) = lex("123abc");
        assert!(diag.has_errors());
    }

    // -- Exponent without digits -------------------------------------------

    #[test]
    fn test_missing_exponent_digits() {
        let (tok, diag) = lex("1e ");
        assert!(diag.has_errors());
        // Should still return a FloatLiteral for recovery.
        match tok {
            TokenKind::FloatLiteral { .. } => {}
            _ => panic!("expected FloatLiteral for recovery"),
        }
    }
}
