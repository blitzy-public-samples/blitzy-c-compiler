//! Numeric literal lexing module for the BCC compiler.
//!
//! Handles parsing of all C11 numeric literal forms:
//! - Decimal integers (`123`, `42u`, `100L`)
//! - Hexadecimal integers (`0xFF`, `0xDEADul`)
//! - Octal integers (`0755`, `0644`)
//! - Binary integers (`0b1010`, `0B11110000`) — GCC extension
//! - Decimal floats (`1.5`, `.5`, `1.`, `1e10`, `1.5e-3`)
//! - Hexadecimal floats (`0x1.ap5`, `0x1p10`)
//!
//! Integer suffixes: `u`/`U`, `l`/`L`, `ll`/`LL`, and all valid combinations.
//! Float suffixes: `f`/`F` (float), `l`/`L` (long double).
//!
//! # Entry Point
//!
//! [`lex_number`] is called from the main lexer tokenization loop
//! (`src/frontend/lexer/mod.rs`) when the scanner encounters a digit
//! character (`0`–`9`) or a dot followed by a digit (`.N`).
//!
//! # Error Recovery
//!
//! On malformed literals, remaining alphanumeric characters are consumed
//! to prevent cascading errors. Clear diagnostics are emitted via the
//! [`DiagnosticEngine`].
//!
//! # Zero Dependencies
//!
//! No external crates are used. Decimal float parsing uses
//! `str::parse::<f64>()` from the Rust standard library. Hex float
//! values are computed with a manual significand × 2^exponent algorithm.

use super::scanner::{Scanner, ScannerMark};
use super::token::{FloatSuffix, IntegerSuffix, Span, Token, TokenKind};
use crate::common::diagnostics::DiagnosticEngine;

// ===========================================================================
// Public API
// ===========================================================================

/// Lex a numeric literal token from the scanner.
///
/// Entry point called by the main tokenization loop when the scanner
/// encounters a digit character (`0`–`9`) or a dot followed by a digit
/// (`.N`). The scanner must be positioned at that first character.
///
/// # Returns
///
/// A [`Token`] with one of the following kinds:
/// - [`TokenKind::IntegerLiteral`] — integer constant with value and suffix
/// - [`TokenKind::FloatLiteral`] — floating-point constant with value and suffix
/// - [`TokenKind::Error`] — malformed numeric literal (diagnostics emitted)
///
/// The returned token's [`Span`] covers the entire literal from the first
/// digit (or dot) through the last suffix character.
pub fn lex_number(scanner: &mut Scanner, diag: &mut DiagnosticEngine) -> Token {
    let start = scanner.byte_offset();
    let file_id = scanner.file_id();

    // Guard: if scanner is at end of input, produce an error token.
    // This should not normally happen (caller guarantees a digit or dot),
    // but defensive programming prevents panics.
    if scanner.is_at_end() {
        let span = Span::new(file_id, start, start);
        diag.error(span, "unexpected end of file in numeric literal");
        return Token::new(TokenKind::Error, span);
    }

    // Case 1: Starts with dot — must be a decimal float like .123
    if scanner.peek() == Some('.') {
        return lex_float_starting_with_dot(scanner, diag, start, file_id);
    }

    // Case 2: Starts with '0' — hex (0x), binary (0b), octal, or decimal float
    if scanner.peek() == Some('0') {
        scanner.advance(); // consume the leading '0'

        match scanner.peek() {
            // Hex prefix: 0x or 0X
            Some('x') | Some('X') => {
                scanner.advance(); // consume 'x'/'X'
                return lex_hex_literal(scanner, diag, start, file_id);
            }
            // Binary prefix: 0b or 0B (GCC extension)
            Some('b') | Some('B') => {
                scanner.advance(); // consume 'b'/'B'
                return lex_binary_literal(scanner, diag, start, file_id);
            }
            // Decimal point — float like 0.5
            // Guard against ".." or "..." (ellipsis) which are separate tokens.
            // Use mark/reset_to to speculatively consume the dot and undo if
            // it turns out to be the start of an ellipsis or range operator.
            Some('.') => {
                let dot_mark = scanner.mark();
                scanner.advance(); // speculatively consume '.'
                if scanner.peek() == Some('.') {
                    // This is ".." — the dot belongs to an ellipsis or range
                    // operator, not a float. Undo the dot consumption.
                    scanner.reset_to(dot_mark);
                    return finish_integer(scanner, diag, start, file_id, 0);
                }
                return continue_decimal_float_after_dot(scanner, diag, start, file_id);
            }
            // Exponent — float like 0e5
            Some('e') | Some('E') => {
                return continue_decimal_float_exponent(scanner, diag, start, file_id);
            }
            // Digit — octal or possibly decimal float
            Some(ch) if Scanner::is_digit(ch) => {
                return lex_after_leading_zero(scanner, diag, start, file_id);
            }
            // Anything else — just the integer 0 (possibly with suffix)
            _ => {
                return finish_integer(scanner, diag, start, file_id, 0);
            }
        }
    }

    // Case 3: Decimal number starting with 1–9
    lex_decimal_literal(scanner, diag, start, file_id)
}

// ===========================================================================
// Float Starting with Dot (.123)
// ===========================================================================

/// Parse a float literal starting with a dot: `.123`, `.5e10`, `.1f`.
///
/// The scanner is positioned at the leading dot. The calling convention
/// guarantees the dot is followed by at least one digit.
fn lex_float_starting_with_dot(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    scanner.advance(); // consume '.'

    // Collect fractional digits (at least one, guaranteed by caller).
    let frac_digits = scanner.advance_while(|ch| Scanner::is_digit(ch));
    if frac_digits.is_empty() {
        // Defensive: should not happen given calling convention.
        let span = Span::new(file_id, start, scanner.byte_offset());
        diag.error(span, "expected digit after '.' in floating-point literal");
        return Token::new(TokenKind::Error, span);
    }

    // Optional exponent (e/E).
    if matches!(scanner.peek(), Some('e') | Some('E')) {
        consume_decimal_exponent(scanner, diag, start, file_id);
    }

    // Parse the accumulated text as f64.
    let literal_text = scanner.slice_from(start);
    let value = safe_parse_f64(literal_text, scanner, diag, start, file_id);

    finish_float(scanner, diag, start, file_id, value)
}

// ===========================================================================
// Hexadecimal Literal (0x / 0X)
// ===========================================================================

/// Parse a hexadecimal literal after the `0x` or `0X` prefix has been consumed.
///
/// Handles both hex integers (`0xFF`, `0xDEADULL`) and hex floats
/// (`0x1.ap5`, `0x.8p-2`). Hex floats require a binary exponent (`p`/`P`).
fn lex_hex_literal(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    // Collect hex digits for the integer part (may be empty for 0x.5p1).
    let int_digits = scanner.advance_while(|ch| Scanner::is_hex_digit(ch));

    match scanner.peek() {
        // Dot — hex float with fractional part.
        Some('.') => {
            scanner.advance(); // consume '.'
            let frac_digits = scanner.advance_while(|ch| Scanner::is_hex_digit(ch));

            // At least one hex digit required across integer + fractional parts.
            if int_digits.is_empty() && frac_digits.is_empty() {
                let span = Span::new(file_id, start, scanner.byte_offset());
                diag.error(
                    span,
                    "no hex digits in hexadecimal floating constant",
                );
                consume_invalid_suffix(scanner);
                return Token::new(
                    TokenKind::Error,
                    Span::new(file_id, start, scanner.byte_offset()),
                );
            }

            // Binary exponent (p/P) is mandatory for hex floats.
            if !matches!(scanner.peek(), Some('p') | Some('P')) {
                let span = Span::new(file_id, start, scanner.byte_offset());
                diag.error(
                    span,
                    "hexadecimal floating constant requires an exponent",
                );
                consume_invalid_suffix(scanner);
                return Token::new(
                    TokenKind::Error,
                    Span::new(file_id, start, scanner.byte_offset()),
                );
            }

            let (exp_sign, exp_value) =
                consume_binary_exponent(scanner, diag, start, file_id);
            let value =
                compute_hex_float_value(int_digits, frac_digits, exp_sign, exp_value);
            finish_float(scanner, diag, start, file_id, value)
        }

        // Binary exponent without fractional part — hex float like 0x1p5.
        Some('p') | Some('P') => {
            if int_digits.is_empty() {
                let span = Span::new(file_id, start, scanner.byte_offset());
                diag.error(span, "no hex digits in hexadecimal floating constant");
                consume_invalid_suffix(scanner);
                return Token::new(
                    TokenKind::Error,
                    Span::new(file_id, start, scanner.byte_offset()),
                );
            }
            let (exp_sign, exp_value) =
                consume_binary_exponent(scanner, diag, start, file_id);
            let value = compute_hex_float_value(int_digits, "", exp_sign, exp_value);
            finish_float(scanner, diag, start, file_id, value)
        }

        // No dot and no exponent — hex integer.
        _ => {
            if int_digits.is_empty() {
                // "0x" with no hex digits at all.
                let span = Span::new(file_id, start, scanner.byte_offset());
                diag.error(span, "no hex digits in hexadecimal constant");
                consume_invalid_suffix(scanner);
                return Token::new(
                    TokenKind::Error,
                    Span::new(file_id, start, scanner.byte_offset()),
                );
            }
            match hex_str_to_u128(int_digits) {
                Some(value) => finish_integer(scanner, diag, start, file_id, value),
                None => {
                    let span = Span::new(file_id, start, scanner.byte_offset());
                    diag.error(span, "hexadecimal integer constant is too large");
                    consume_invalid_suffix(scanner);
                    Token::new(
                        TokenKind::Error,
                        Span::new(file_id, start, scanner.byte_offset()),
                    )
                }
            }
        }
    }
}

// ===========================================================================
// Binary Literal (0b / 0B) — GCC Extension
// ===========================================================================

/// Parse a binary literal after the `0b` or `0B` prefix has been consumed.
///
/// Binary literals are a GCC extension; at least one binary digit (`0` or `1`)
/// is required after the prefix.
fn lex_binary_literal(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    let bin_digits = scanner.advance_while(|ch| ch == '0' || ch == '1');

    if bin_digits.is_empty() {
        // "0b" with no binary digits.
        let span = Span::new(file_id, start, scanner.byte_offset());
        diag.error(span, "no binary digits in binary constant after '0b' prefix");
        consume_invalid_suffix(scanner);
        return Token::new(
            TokenKind::Error,
            Span::new(file_id, start, scanner.byte_offset()),
        );
    }

    // Check for invalid digits (2-9) that might follow binary digits.
    if let Some(ch) = scanner.peek() {
        if Scanner::is_digit(ch) && ch != '0' && ch != '1' {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(
                span,
                format!("invalid digit '{}' in binary constant", ch),
            );
            consume_invalid_suffix(scanner);
            return Token::new(
                TokenKind::Error,
                Span::new(file_id, start, scanner.byte_offset()),
            );
        }
    }

    match binary_str_to_u128(bin_digits) {
        Some(value) => finish_integer(scanner, diag, start, file_id, value),
        None => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(span, "binary integer constant is too large");
            consume_invalid_suffix(scanner);
            Token::new(
                TokenKind::Error,
                Span::new(file_id, start, scanner.byte_offset()),
            )
        }
    }
}

// ===========================================================================
// After Leading Zero (Octal / Decimal Float)
// ===========================================================================

/// Parse a number after a leading `0` followed by another digit.
///
/// The leading `0` has been consumed; the scanner is at the next digit.
/// This could be:
/// - Octal integer (`0755`)
/// - Decimal float (`0123.45`, `0999e2`)
/// - Invalid octal (`089`)
fn lex_after_leading_zero(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    // Consume all decimal digits following the leading zero.
    // We accept 0-9 here; validation for octal (0-7 only) happens later
    // if this turns out not to be a float.
    let extra_digits = scanner.advance_while(|ch| Scanner::is_digit(ch));

    match scanner.peek() {
        // Dot — decimal float (e.g. 0123.45, 0895.0)
        // Guard against ".." / "..." (ellipsis) which are separate tokens.
        // Use mark/reset_to to speculatively consume the dot and undo if
        // it is part of a ".." or "..." range/ellipsis.
        Some('.') => {
            let dot_mark = scanner.mark();
            scanner.advance(); // speculatively consume '.'
            if scanner.peek() == Some('.') {
                // ".." — undo the dot and fall through to octal validation.
                scanner.reset_to(dot_mark);
            } else {
                return continue_decimal_float_after_dot(scanner, diag, start, file_id);
            }
        }
        // Exponent — decimal float (e.g. 0123e5)
        Some('e') | Some('E') => {
            return continue_decimal_float_exponent(scanner, diag, start, file_id);
        }
        _ => {}
    }

    // Not a float — interpret as octal integer.
    // Validate that all digits are in range 0-7 using the Scanner's
    // octal digit classifier for consistency.
    for ch in extra_digits.chars() {
        if !Scanner::is_octal_digit(ch) {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(
                span,
                format!("invalid digit '{}' in octal constant", ch),
            );
            consume_invalid_suffix(scanner);
            return Token::new(
                TokenKind::Error,
                Span::new(file_id, start, scanner.byte_offset()),
            );
        }
    }

    // Parse the octal value from the digits after '0'.
    match octal_str_to_u128(extra_digits) {
        Some(value) => finish_integer(scanner, diag, start, file_id, value),
        None => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(span, "octal integer constant is too large");
            consume_invalid_suffix(scanner);
            Token::new(
                TokenKind::Error,
                Span::new(file_id, start, scanner.byte_offset()),
            )
        }
    }
}

// ===========================================================================
// Decimal Literal (starts with 1–9)
// ===========================================================================

/// Parse a decimal literal starting with a digit `1`–`9`.
///
/// This can be a plain integer (`123`), an integer with suffix (`123ULL`),
/// or a decimal float (`123.45`, `123e10`, `123.45e-3f`).
fn lex_decimal_literal(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    // Consume the full sequence of decimal digits.
    let digits = scanner.advance_while(|ch| Scanner::is_digit(ch));

    match scanner.peek() {
        // Dot — decimal float.
        // Guard against ".." / "..." which are separate tokens.
        Some('.') => {
            if scanner.peek_ahead(1) != Some('.') {
                scanner.advance(); // consume '.'
                return continue_decimal_float_after_dot(scanner, diag, start, file_id);
            }
            // Dot is part of ".." or "..." — fall through to integer.
        }
        // Exponent — decimal float (e.g. 123e5).
        Some('e') | Some('E') => {
            return continue_decimal_float_exponent(scanner, diag, start, file_id);
        }
        _ => {}
    }

    // Plain decimal integer.
    match decimal_str_to_u128(digits) {
        Some(value) => finish_integer(scanner, diag, start, file_id, value),
        None => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(span, "integer constant is too large for any integer type");
            consume_invalid_suffix(scanner);
            Token::new(
                TokenKind::Error,
                Span::new(file_id, start, scanner.byte_offset()),
            )
        }
    }
}

// ===========================================================================
// Decimal Float Continuation Helpers
// ===========================================================================

/// Continue parsing a decimal float after the dot has been consumed.
///
/// The scanner has consumed everything up to and including the `.`.
/// Remaining: optional fractional digits, optional exponent, optional suffix.
/// Examples at this point: integer part already consumed, scanner past `.`.
fn continue_decimal_float_after_dot(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    // Collect optional fractional digits.
    let _frac = scanner.advance_while(|ch| Scanner::is_digit(ch));

    // Optional exponent.
    if matches!(scanner.peek(), Some('e') | Some('E')) {
        consume_decimal_exponent(scanner, diag, start, file_id);
    }

    // Get the full literal text from start to current position (before suffix).
    let literal_text = scanner.slice_from(start);
    let value = safe_parse_f64(literal_text, scanner, diag, start, file_id);

    finish_float(scanner, diag, start, file_id, value)
}

/// Continue parsing a decimal float when an exponent (`e`/`E`) is seen.
///
/// The scanner is positioned at the `e`/`E` character. Everything before
/// (integer digits and possibly a dot + fractional digits) has been consumed.
fn continue_decimal_float_exponent(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> Token {
    consume_decimal_exponent(scanner, diag, start, file_id);

    let literal_text = scanner.slice_from(start);
    let value = safe_parse_f64(literal_text, scanner, diag, start, file_id);

    finish_float(scanner, diag, start, file_id, value)
}

// ===========================================================================
// Exponent Consumption
// ===========================================================================

/// Consume a decimal exponent: `e`/`E`, optional sign, decimal digits.
///
/// The scanner must be positioned at the `e`/`E` character. After this
/// function returns, the scanner is past the last exponent digit.
///
/// If the exponent is malformed (no digits after sign), a diagnostic
/// is emitted but the scanner is left past whatever was consumed.
fn consume_decimal_exponent(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) {
    // Consume 'e' or 'E'.
    scanner.advance();

    // Optional sign — use advance_if for concise conditional consumption.
    if !scanner.advance_if('+') {
        scanner.advance_if('-');
    }

    // Require at least one digit.
    let exp_digits = scanner.advance_while(|ch| Scanner::is_digit(ch));
    if exp_digits.is_empty() {
        let span = Span::new(file_id, start, scanner.byte_offset());
        diag.error(span, "exponent has no digits");
    }
}

/// Consume a binary (hex-float) exponent: `p`/`P`, optional sign, decimal digits.
///
/// Returns `(sign, absolute_value)` where sign is `1` or `-1`.
/// The scanner must be positioned at the `p`/`P` character.
fn consume_binary_exponent(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> (i32, u32) {
    // Consume 'p' or 'P'.
    scanner.advance();

    // Optional sign — use advance_if for concise conditional consumption.
    let sign: i32 = if scanner.advance_if('-') {
        -1
    } else {
        // Consume '+' if present; default to positive.
        scanner.advance_if('+');
        1
    };

    // Require at least one decimal digit for the exponent value.
    let exp_digits = scanner.advance_while(|ch| Scanner::is_digit(ch));
    if exp_digits.is_empty() {
        let span = Span::new(file_id, start, scanner.byte_offset());
        diag.error(span, "exponent has no digits in hexadecimal floating constant");
        return (sign, 0);
    }

    // Parse the exponent magnitude. Exponents above a few thousand are
    // effectively ±infinity for f64, so u32 is more than sufficient.
    let exp_val: u32 = exp_digits
        .chars()
        .fold(0u32, |acc, ch| {
            acc.saturating_mul(10)
                .saturating_add((ch as u32).wrapping_sub('0' as u32))
        });

    (sign, exp_val)
}

// ===========================================================================
// Integer Finishing (Suffix + Token Creation)
// ===========================================================================

/// Finish an integer literal: parse suffix, validate, create token.
///
/// The scanner is positioned immediately after the last digit of the
/// integer literal. Any identifier-continuation characters following the
/// digits are consumed as a potential suffix.
fn finish_integer(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
    value: u128,
) -> Token {
    // Consume any trailing ASCII alphanumeric characters and underscores
    // as a potential integer suffix. Unicode identifier continuations are
    // excluded — only C suffix letters (u/U, l/L) and error-recovery
    // characters (other ASCII letters and digits) are consumed.
    let suffix_text = scanner.advance_while(|ch| {
        matches!(ch, 'a'..='z' | 'A'..='Z' | '_' | '0'..='9')
    });

    if suffix_text.is_empty() {
        let span = Span::new(file_id, start, scanner.byte_offset());
        return Token::new(
            TokenKind::IntegerLiteral {
                value,
                suffix: IntegerSuffix::None,
            },
            span,
        );
    }

    // Attempt to parse a valid integer suffix.
    match parse_integer_suffix_text(suffix_text) {
        Some(suffix) => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            Token::new(TokenKind::IntegerLiteral { value, suffix }, span)
        }
        None => {
            // Invalid suffix — report the offending text. We use the text
            // directly from advance_while rather than re-slicing.
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(
                span,
                format!("invalid suffix '{}' on integer constant", suffix_text),
            );
            Token::new(TokenKind::Error, span)
        }
    }
}

// ===========================================================================
// Float Finishing (Suffix + Token Creation)
// ===========================================================================

/// Finish a floating-point literal: parse suffix, validate, create token.
///
/// The scanner is positioned immediately after the last character of the
/// numeric portion (digits, dot, exponent). Any trailing identifier
/// characters are consumed as a potential suffix.
fn finish_float(
    scanner: &mut Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
    value: f64,
) -> Token {
    // Consume potential float suffix (f/F/l/L) and any erroneous trailing
    // identifier characters for error recovery.
    let suffix_text = scanner.advance_while(|ch| {
        matches!(ch, 'a'..='z' | 'A'..='Z' | '_' | '0'..='9')
    });

    if suffix_text.is_empty() {
        let span = Span::new(file_id, start, scanner.byte_offset());
        return Token::new(
            TokenKind::FloatLiteral {
                value,
                suffix: FloatSuffix::None,
            },
            span,
        );
    }

    match parse_float_suffix_text(suffix_text) {
        Some(suffix) => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            Token::new(TokenKind::FloatLiteral { value, suffix }, span)
        }
        None => {
            let span = Span::new(file_id, start, scanner.byte_offset());
            diag.error(
                span,
                format!("invalid suffix '{}' on floating constant", suffix_text),
            );
            Token::new(TokenKind::Error, span)
        }
    }
}

// ===========================================================================
// Suffix Parsing
// ===========================================================================

/// Attempt to parse an integer suffix string.
///
/// Valid suffixes (case-insensitive):
/// - `""` → [`IntegerSuffix::None`]
/// - `"u"` → [`IntegerSuffix::U`]
/// - `"l"` → [`IntegerSuffix::L`]
/// - `"ul"` / `"lu"` → [`IntegerSuffix::UL`]
/// - `"ll"` → [`IntegerSuffix::LL`]
/// - `"ull"` / `"llu"` → [`IntegerSuffix::ULL`]
///
/// Returns `None` for invalid suffixes.
fn parse_integer_suffix_text(text: &str) -> Option<IntegerSuffix> {
    // Convert to lowercase for case-insensitive matching.
    // Suffix strings are at most 3 characters, so this allocation is negligible.
    let lower: String = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    match lower.as_str() {
        "u" => Some(IntegerSuffix::U),
        "l" => Some(IntegerSuffix::L),
        "ul" | "lu" => Some(IntegerSuffix::UL),
        "ll" => Some(IntegerSuffix::LL),
        "ull" | "llu" => Some(IntegerSuffix::ULL),
        _ => None,
    }
}

/// Attempt to parse a floating-point suffix string.
///
/// Valid suffixes (case-insensitive):
/// - `""` → [`FloatSuffix::None`] (double)
/// - `"f"` → [`FloatSuffix::F`] (float)
/// - `"l"` → [`FloatSuffix::L`] (long double)
///
/// Returns `None` for invalid suffixes.
fn parse_float_suffix_text(text: &str) -> Option<FloatSuffix> {
    let lower: String = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    match lower.as_str() {
        "f" => Some(FloatSuffix::F),
        "l" => Some(FloatSuffix::L),
        _ => None,
    }
}

// ===========================================================================
// Value Parsing — String-to-Integer Conversion
// ===========================================================================

/// Parse a decimal digit string into a `u128`, returning `None` on overflow.
///
/// The input string must consist entirely of ASCII digits (`0`–`9`).
/// An empty string returns `Some(0)`.
fn decimal_str_to_u128(s: &str) -> Option<u128> {
    if s.is_empty() {
        return Some(0);
    }
    let mut value: u128 = 0;
    for ch in s.chars() {
        let digit = (ch as u32).wrapping_sub('0' as u32) as u128;
        value = value.checked_mul(10)?;
        value = value.checked_add(digit)?;
    }
    Some(value)
}

/// Parse a hexadecimal digit string into a `u128`, returning `None` on overflow.
///
/// The input string must consist entirely of hex digits (`0`–`9`, `a`–`f`, `A`–`F`).
/// An empty string returns `Some(0)`.
fn hex_str_to_u128(s: &str) -> Option<u128> {
    if s.is_empty() {
        return Some(0);
    }
    let mut value: u128 = 0;
    for ch in s.chars() {
        let digit = hex_char_value(ch) as u128;
        value = value.checked_mul(16)?;
        value = value.checked_add(digit)?;
    }
    Some(value)
}

/// Parse an octal digit string into a `u128`, returning `None` on overflow.
///
/// The input string must consist entirely of octal digits (`0`–`7`).
/// An empty string returns `Some(0)`.
fn octal_str_to_u128(s: &str) -> Option<u128> {
    if s.is_empty() {
        return Some(0);
    }
    let mut value: u128 = 0;
    for ch in s.chars() {
        let digit = (ch as u32).wrapping_sub('0' as u32) as u128;
        value = value.checked_mul(8)?;
        value = value.checked_add(digit)?;
    }
    Some(value)
}

/// Parse a binary digit string into a `u128`, returning `None` on overflow.
///
/// The input string must consist entirely of binary digits (`0` or `1`).
/// An empty string returns `Some(0)`.
fn binary_str_to_u128(s: &str) -> Option<u128> {
    if s.is_empty() {
        return Some(0);
    }
    let mut value: u128 = 0;
    for ch in s.chars() {
        let digit = (ch as u32).wrapping_sub('0' as u32) as u128;
        value = value.checked_mul(2)?;
        value = value.checked_add(digit)?;
    }
    Some(value)
}

/// Convert a single hex character to its numeric value (0–15).
///
/// # Panics
///
/// Panics if `ch` is not a valid hex digit. Callers must pre-validate
/// using [`Scanner::is_hex_digit`].
#[inline]
fn hex_char_value(ch: char) -> u8 {
    match ch {
        '0'..='9' => (ch as u8) - b'0',
        'a'..='f' => (ch as u8) - b'a' + 10,
        'A'..='F' => (ch as u8) - b'A' + 10,
        _ => unreachable!("hex_char_value called with non-hex char: {:?}", ch),
    }
}

// ===========================================================================
// Hex Float Computation
// ===========================================================================

/// Compute the `f64` value of a hexadecimal floating-point literal.
///
/// The value is `significand × 2^(sign × exponent)` where the significand
/// is constructed from the hex integer and fractional digit strings.
///
/// # Arguments
///
/// * `int_digits` — Hex digits before the dot (may be empty).
/// * `frac_digits` — Hex digits after the dot (may be empty).
/// * `exp_sign` — Exponent sign: `1` for positive, `-1` for negative.
/// * `exp_value` — Absolute value of the binary exponent.
///
/// # Examples
///
/// ```text
/// 0x1.ap5 → int_digits="1", frac_digits="a", exp=+5
///           significand = 1 + 10/16 = 1.625
///           value = 1.625 × 2^5 = 52.0
///
/// 0x.8p-2 → int_digits="", frac_digits="8", exp=-2
///           significand = 0 + 8/16 = 0.5
///           value = 0.5 × 2^(-2) = 0.125
/// ```
fn compute_hex_float_value(
    int_digits: &str,
    frac_digits: &str,
    exp_sign: i32,
    exp_value: u32,
) -> f64 {
    // Build the significand from the hex digit strings.
    let mut significand: f64 = 0.0;

    // Integer part: each digit contributes digit × 16^position.
    for ch in int_digits.chars() {
        significand = significand * 16.0 + hex_char_value(ch) as f64;
    }

    // Fractional part: each digit contributes digit × 16^(-position).
    let mut place_value: f64 = 1.0 / 16.0;
    for ch in frac_digits.chars() {
        significand += hex_char_value(ch) as f64 * place_value;
        place_value /= 16.0;
    }

    // Apply the binary exponent: result = significand × 2^(sign × value).
    // Clamp exponent to avoid overflow in powi; extreme exponents yield
    // ±infinity or 0.0 naturally via f64 arithmetic.
    let exponent = exp_sign * (exp_value.min(10000) as i32);

    if exponent >= 0 {
        significand * 2.0f64.powi(exponent)
    } else {
        significand / 2.0f64.powi(-exponent)
    }
}

// ===========================================================================
// Float Parsing Helper
// ===========================================================================

/// Safely parse a decimal float literal text to `f64`.
///
/// Handles edge cases where the Rust parser might reject the input
/// (e.g., malformed text from error recovery) by returning 0.0 and
/// emitting a warning.
fn safe_parse_f64(
    text: &str,
    _scanner: &Scanner,
    diag: &mut DiagnosticEngine,
    start: u32,
    file_id: u32,
) -> f64 {
    match text.parse::<f64>() {
        Ok(v) => {
            // Warn on infinity (literal too large for f64).
            if v.is_infinite() {
                let span = Span::new(file_id, start, start + text.len() as u32);
                diag.warning(span, "floating constant exceeds range of 'double'");
            }
            v
        }
        Err(_) => {
            // Should not happen for well-formed literals, but be defensive.
            let span = Span::new(file_id, start, start + text.len() as u32);
            diag.warning(span, "unable to parse floating-point literal value");
            0.0
        }
    }
}

// ===========================================================================
// Error Recovery
// ===========================================================================

/// Consume remaining identifier-continuation characters after an error.
///
/// When a malformed numeric literal is detected, this function advances
/// the scanner past any trailing letters, digits, or underscores so that
/// the lexer does not produce cascading errors on the leftover characters.
fn consume_invalid_suffix(scanner: &mut Scanner) {
    scanner.advance_while(|ch| {
        matches!(ch, 'a'..='z' | 'A'..='Z' | '_' | '0'..='9') || ch == '.'
    });
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a Scanner and DiagnosticEngine, lex one number token.
    fn lex_one(src: &str) -> (Token, DiagnosticEngine) {
        let mut scanner = Scanner::new(src, 0);
        let mut diag = DiagnosticEngine::new();
        let tok = lex_number(&mut scanner, &mut diag);
        (tok, diag)
    }

    // -----------------------------------------------------------------------
    // Decimal integers
    // -----------------------------------------------------------------------

    #[test]
    fn test_decimal_zero() {
        let (tok, diag) = lex_one("0");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 0);
                assert_eq!(suffix, IntegerSuffix::None);
            }
            _ => panic!("expected IntegerLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_decimal_simple() {
        let (tok, diag) = lex_one("42");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 42);
                assert_eq!(suffix, IntegerSuffix::None);
            }
            _ => panic!("expected IntegerLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_decimal_large() {
        let (tok, diag) = lex_one("18446744073709551615");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, .. } => {
                assert_eq!(value, u64::MAX as u128);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    // -----------------------------------------------------------------------
    // Integer suffixes
    // -----------------------------------------------------------------------

    #[test]
    fn test_suffix_u() {
        let (tok, _) = lex_one("42u");
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 42);
                assert_eq!(suffix, IntegerSuffix::U);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_ul() {
        let (tok, _) = lex_one("100UL");
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 100);
                assert_eq!(suffix, IntegerSuffix::UL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_llu() {
        let (tok, _) = lex_one("999LLU");
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 999);
                assert_eq!(suffix, IntegerSuffix::ULL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_suffix_ll() {
        let (tok, _) = lex_one("1LL");
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 1);
                assert_eq!(suffix, IntegerSuffix::LL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_invalid_suffix() {
        let (tok, diag) = lex_one("123xyz");
        assert!(diag.has_errors());
        assert!(matches!(tok.kind, TokenKind::Error));
    }

    // -----------------------------------------------------------------------
    // Hexadecimal integers
    // -----------------------------------------------------------------------

    #[test]
    fn test_hex_simple() {
        let (tok, diag) = lex_one("0xFF");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 255);
                assert_eq!(suffix, IntegerSuffix::None);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_hex_upper() {
        let (tok, _) = lex_one("0XDEAD");
        match tok.kind {
            TokenKind::IntegerLiteral { value, .. } => {
                assert_eq!(value, 0xDEAD);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_hex_no_digits() {
        let (tok, diag) = lex_one("0x");
        assert!(diag.has_errors());
        assert!(matches!(tok.kind, TokenKind::Error));
    }

    // -----------------------------------------------------------------------
    // Octal integers
    // -----------------------------------------------------------------------

    #[test]
    fn test_octal_simple() {
        let (tok, diag) = lex_one("0755");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 0o755);
                assert_eq!(suffix, IntegerSuffix::None);
            }
            _ => panic!("expected IntegerLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_octal_invalid_digit() {
        let (tok, diag) = lex_one("089");
        assert!(diag.has_errors());
        assert!(matches!(tok.kind, TokenKind::Error));
    }

    // -----------------------------------------------------------------------
    // Binary integers
    // -----------------------------------------------------------------------

    #[test]
    fn test_binary_simple() {
        let (tok, diag) = lex_one("0b1010");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 0b1010);
                assert_eq!(suffix, IntegerSuffix::None);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_binary_no_digits() {
        let (tok, diag) = lex_one("0b");
        assert!(diag.has_errors());
        assert!(matches!(tok.kind, TokenKind::Error));
    }

    // -----------------------------------------------------------------------
    // Decimal floats
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_dot_digits() {
        let (tok, diag) = lex_one(".5");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, suffix } => {
                assert!((value - 0.5).abs() < 1e-15);
                assert_eq!(suffix, FloatSuffix::None);
            }
            _ => panic!("expected FloatLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_float_digits_dot() {
        let (tok, diag) = lex_one("1.");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 1.0).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_float_digits_dot_digits() {
        let (tok, diag) = lex_one("3.14");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 3.14).abs() < 1e-10);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_float_exponent() {
        let (tok, diag) = lex_one("1e10");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 1e10).abs() < 1.0);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_float_suffix_f() {
        let (tok, diag) = lex_one("1.5f");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, suffix } => {
                assert!((value - 1.5).abs() < 1e-15);
                assert_eq!(suffix, FloatSuffix::F);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_float_suffix_l() {
        let (tok, _) = lex_one("1.0L");
        match tok.kind {
            TokenKind::FloatLiteral { suffix, .. } => {
                assert_eq!(suffix, FloatSuffix::L);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    // -----------------------------------------------------------------------
    // Hex floats
    // -----------------------------------------------------------------------

    #[test]
    fn test_hex_float_simple() {
        let (tok, diag) = lex_one("0x1.0p0");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 1.0).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_hex_float_complex() {
        // 0x1.ap5 = (1 + 10/16) * 2^5 = 1.625 * 32 = 52.0
        let (tok, diag) = lex_one("0x1.ap5");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 52.0).abs() < 1e-10);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_hex_float_no_exponent() {
        let (tok, diag) = lex_one("0x1.0");
        assert!(diag.has_errors());
        assert!(matches!(tok.kind, TokenKind::Error));
    }

    // -----------------------------------------------------------------------
    // Span coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_span_covers_literal() {
        let (tok, _) = lex_one("0xFFul");
        assert_eq!(tok.span.start, 0);
        assert_eq!(tok.span.end, 6); // "0xFFul" = 6 chars
    }

    #[test]
    fn test_span_covers_float() {
        let (tok, _) = lex_one("3.14f");
        assert_eq!(tok.span.start, 0);
        assert_eq!(tok.span.end, 5); // "3.14f" = 5 chars
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_with_suffix() {
        let (tok, diag) = lex_one("0ULL");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 0);
                assert_eq!(suffix, IntegerSuffix::ULL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_zero_dot_is_float() {
        let (tok, diag) = lex_one("0.");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 0.0).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_hex_float_negative_exponent() {
        // 0x1p-2 = 1 * 2^(-2) = 0.25
        let (tok, diag) = lex_one("0x1p-2");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 0.25).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_octal_with_suffix() {
        let (tok, diag) = lex_one("0777UL");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::IntegerLiteral { value, suffix } => {
                assert_eq!(value, 0o777);
                assert_eq!(suffix, IntegerSuffix::UL);
            }
            _ => panic!("expected IntegerLiteral"),
        }
    }

    #[test]
    fn test_zero_e_is_float() {
        // 0e0 = 0.0
        let (tok, diag) = lex_one("0e0");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 0.0).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral, got {:?}", tok.kind),
        }
    }

    #[test]
    fn test_float_with_negative_exponent() {
        // 1.5e-3 = 0.0015
        let (tok, diag) = lex_one("1.5e-3");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 0.0015).abs() < 1e-10);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }

    #[test]
    fn test_hex_float_frac_only() {
        // 0x.8p0 = 0.5 * 2^0 = 0.5
        let (tok, diag) = lex_one("0x.8p0");
        assert!(!diag.has_errors());
        match tok.kind {
            TokenKind::FloatLiteral { value, .. } => {
                assert!((value - 0.5).abs() < 1e-15);
            }
            _ => panic!("expected FloatLiteral"),
        }
    }
}
