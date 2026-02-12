//! PUA (Private Use Area) / UTF-8 encoding module for byte-exact round-tripping.
//!
//! Maps non-UTF-8 bytes (0x80–0xFF) to Unicode Private Use Area code points
//! (U+E080–U+E0FF) during source file reading and decodes them back to original
//! bytes during code generation output.
//!
//! This ensures that binary data in C string literals and inline assembly operands
//! survives the Rust `String`-based pipeline with byte-exact fidelity. Critical
//! for Linux kernel compilation where source files contain raw binary data.
//!
//! # PUA Mapping Scheme
//!
//! - Bytes `0x00`–`0x7F`: Valid ASCII, passed through unchanged
//! - Bytes `0x80`–`0xFF`: Mapped to U+E080–U+E0FF (one-to-one bijection)
//! - Formula: `PUA_code_point = 0xE000 + byte_value`
//! - Reverse: `byte_value = PUA_code_point - 0xE000`
//!
//! # Round-Trip Guarantee
//!
//! For any byte sequence `B`:
//! ```text
//! decode_pua_string(&encode_bytes_to_pua_string(B)) == B
//! ```
//!
//! # UTF-8 Handling
//!
//! During source file reading, the encoder performs strict RFC 3629 validation:
//! - Valid multi-byte UTF-8 sequences are preserved as-is
//! - Only genuinely invalid bytes are PUA-encoded
//! - Overlong encodings, surrogate code points, and over-range values are rejected
//!
//! # Design Note
//!
//! The PUA sub-range U+E080–U+E0FF was chosen because:
//! 1. It lies within the BMP Private Use Area (U+E000–U+F8FF)
//! 2. No standard character set assigns meaning to these code points
//! 3. C source files (including the Linux kernel) do not contain these code points
//! 4. The 128-entry range provides a perfect 1:1 mapping for bytes 0x80–0xFF

use std::fs;
use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// PUA Encoding Constants
// ---------------------------------------------------------------------------

/// Start of the Private Use Area mapping range (U+E080).
/// Corresponds to the encoding of byte `0x80`.
const PUA_BASE: u32 = 0xE080;

/// End of the Private Use Area mapping range, inclusive (U+E0FF).
/// Corresponds to the encoding of byte `0xFF`.
const PUA_END: u32 = 0xE0FF;

/// Offset applied to convert between raw byte values and PUA code points.
///
/// The mapping is: `PUA_code_point = PUA_OFFSET + byte_value`
/// (for bytes 0x80–0xFF this yields U+E080–U+E0FF).
///
/// The reverse is: `byte_value = PUA_code_point - PUA_OFFSET`.
const PUA_OFFSET: u32 = 0xE000;

// ---------------------------------------------------------------------------
// Core Encoding Functions
// ---------------------------------------------------------------------------

/// Encode a single byte to its character representation, using PUA for non-ASCII.
///
/// - Bytes `0x00`–`0x7F` (valid ASCII) are returned as the corresponding ASCII `char`.
/// - Bytes `0x80`–`0xFF` are mapped to PUA code points U+E080–U+E0FF.
///
/// This function is the atomic unit of the PUA encoding scheme: it handles
/// individual bytes that are known to be outside valid UTF-8 multi-byte sequences.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(encode_byte_to_pua(b'A'), 'A');       // ASCII passthrough
/// assert_eq!(encode_byte_to_pua(0x80), '\u{E080}'); // PUA encoding
/// assert_eq!(encode_byte_to_pua(0xFF), '\u{E0FF}'); // PUA encoding
/// ```
#[inline]
pub fn encode_byte_to_pua(byte: u8) -> char {
    if byte < 0x80 {
        // ASCII byte: return directly as a char (0x00–0x7F are valid Unicode scalar values)
        byte as char
    } else {
        let code_point = PUA_OFFSET + byte as u32;
        // Debug-mode validation that the computed code point is within the expected PUA range
        debug_assert!(
            (PUA_BASE..=PUA_END).contains(&code_point),
            "PUA encoding out of range: U+{:04X} for byte 0x{:02X}",
            code_point,
            byte
        );
        // SAFETY: PUA_OFFSET (0xE000) + byte (0x80..=0xFF) produces code points in the range
        // 0xE080..=0xE0FF, all of which are valid Unicode scalar values within the BMP
        // Private Use Area. They are not surrogates, not above U+10FFFF, and not invalid.
        unsafe { char::from_u32_unchecked(code_point) }
    }
}

/// Decode a PUA-encoded character back to its original byte value.
///
/// Returns `Some(byte)` if the character is in the PUA mapping range U+E080–U+E0FF,
/// indicating it was a PUA-encoded non-UTF-8 byte. Returns `None` for all other
/// characters, indicating they are ordinary Unicode characters.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(decode_pua_to_byte('\u{E080}'), Some(0x80));
/// assert_eq!(decode_pua_to_byte('\u{E0FF}'), Some(0xFF));
/// assert_eq!(decode_pua_to_byte('A'), None);         // Not PUA-encoded
/// assert_eq!(decode_pua_to_byte('\u{E07F}'), None);   // Below PUA range
/// assert_eq!(decode_pua_to_byte('\u{E100}'), None);   // Above PUA range
/// ```
#[inline]
pub fn decode_pua_to_byte(ch: char) -> Option<u8> {
    let code_point = ch as u32;
    if (PUA_BASE..=PUA_END).contains(&code_point) {
        // Reverse the encoding: byte = code_point - PUA_OFFSET
        // This yields a value in 0x80..=0xFF, which fits in u8.
        Some((code_point - PUA_OFFSET) as u8)
    } else {
        None
    }
}

/// Check whether a character is a PUA-encoded byte.
///
/// Returns `true` if the character falls within the PUA mapping range
/// U+E080–U+E0FF, meaning it represents a PUA-encoded non-UTF-8 byte
/// from the original source file.
///
/// This is a convenience predicate equivalent to `decode_pua_to_byte(ch).is_some()`.
#[inline]
pub fn is_pua_encoded(ch: char) -> bool {
    let cp = ch as u32;
    (PUA_BASE..=PUA_END).contains(&cp)
}

// ---------------------------------------------------------------------------
// File I/O with PUA Encoding
// ---------------------------------------------------------------------------

/// Read a source file with PUA encoding applied to non-UTF-8 bytes.
///
/// Reads the file as raw bytes, then processes the byte stream:
/// - Valid UTF-8 multi-byte sequences are preserved unchanged
/// - Invalid bytes (0x80–0xFF not part of any valid UTF-8 sequence) are encoded
///   as PUA characters (U+E080–U+E0FF)
///
/// The returned `String` is always valid Rust UTF-8, with non-UTF-8 bytes
/// represented as PUA code points that can later be decoded back to the
/// original bytes via [`decode_pua_string`].
///
/// # Performance
///
/// Uses a fast path for files that are entirely valid UTF-8 (the common case
/// for most C source files). Only falls back to byte-by-byte processing when
/// `String::from_utf8` detects at least one invalid byte.
///
/// # Errors
///
/// Returns `io::Error` if the file cannot be read (not found, permission denied, etc.).
pub fn read_source_file(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;

    // Fast path: if the entire file is valid UTF-8, no PUA encoding is needed.
    // This avoids the byte-by-byte processing overhead for the common case.
    // Most C source files are pure ASCII or valid UTF-8.
    match String::from_utf8(bytes) {
        Ok(valid_string) => Ok(valid_string),
        Err(from_utf8_err) => {
            // Slow path: the file contains at least one non-UTF-8 byte.
            // Recover the original byte vector and process with PUA encoding.
            let raw_bytes = from_utf8_err.into_bytes();
            Ok(encode_bytes_to_pua_string(&raw_bytes))
        }
    }
}

/// Decode a PUA-encoded string back to a raw byte vector.
///
/// Iterates over all characters in the string:
/// - PUA characters (U+E080–U+E0FF) are decoded back to the original single byte
///   (0x80–0xFF) that was PUA-encoded during source file reading
/// - All other characters are emitted as their standard UTF-8 byte sequences
///
/// The result is byte-exact relative to the original file content, preserving
/// the round-trip guarantee required for binary data in C string literals
/// and inline assembly operands.
///
/// # Examples
///
/// ```ignore
/// // ASCII passes through unchanged
/// assert_eq!(decode_pua_string("Hello"), b"Hello");
///
/// // PUA characters decode to original non-UTF-8 bytes
/// let s = "\u{E080}\u{E0FF}";
/// assert_eq!(decode_pua_string(s), &[0x80, 0xFF]);
/// ```
pub fn decode_pua_string(s: &str) -> Vec<u8> {
    let mut result = Vec::with_capacity(s.len());

    for ch in s.chars() {
        if let Some(byte) = decode_pua_to_byte(ch) {
            // PUA-encoded byte: emit the original single byte value (0x80–0xFF).
            // This reverses the encoding performed by encode_bytes_to_pua_string.
            result.push(byte);
        } else {
            // Regular Unicode character (including ASCII): emit its standard
            // UTF-8 encoding. For ASCII, this is a single byte. For multi-byte
            // characters, this emits 2–4 bytes matching the original file content.
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            result.extend_from_slice(encoded.as_bytes());
        }
    }

    result
}

/// Write a PUA-encoded string to a file, decoding PUA characters back to
/// their original byte values before writing.
///
/// This is the output counterpart to [`read_source_file`]. Any PUA characters
/// in the content string are decoded back to the original non-UTF-8 bytes,
/// ensuring byte-exact fidelity with the original source material.
///
/// # Errors
///
/// Returns `io::Error` if the file cannot be written (permission denied,
/// disk full, invalid path, etc.).
pub fn write_decoded_file(path: &Path, content: &str) -> io::Result<()> {
    let decoded_bytes = decode_pua_string(content);
    fs::write(path, decoded_bytes)
}

// ---------------------------------------------------------------------------
// Internal: Byte-Stream to PUA-Encoded String Conversion
// ---------------------------------------------------------------------------

/// Convert a raw byte slice to a PUA-encoded `String`.
///
/// Walks through the byte stream performing strict RFC 3629 UTF-8 validation:
/// - Valid single-byte characters (ASCII, 0x00–0x7F) pass through directly
/// - Valid multi-byte UTF-8 sequences (2–4 bytes) are preserved as-is
/// - Invalid bytes (lone continuation bytes, invalid lead bytes, truncated
///   sequences, overlong encodings, surrogates, over-range code points) are
///   each individually encoded as PUA characters
///
/// The result is always a valid Rust `String` that preserves the ability to
/// reconstruct the original byte stream via [`decode_pua_string`].
fn encode_bytes_to_pua_string(bytes: &[u8]) -> String {
    // Pre-allocate with some headroom: PUA characters take 3 UTF-8 bytes each,
    // but most bytes will be ASCII (1 byte) or valid UTF-8 multi-byte sequences.
    let mut result = String::with_capacity(bytes.len() + bytes.len() / 8);
    let mut pos = 0;

    while pos < bytes.len() {
        let byte = bytes[pos];

        // ASCII byte (0x00–0x7F): always valid, pass through directly.
        // This is the fast path for the majority of C source content.
        if byte < 0x80 {
            result.push(byte as char);
            pos += 1;
            continue;
        }

        // Attempt to parse a valid UTF-8 multi-byte sequence starting at this byte.
        // First, check if it's a valid lead byte and determine the expected length.
        if let Some(seq_len) = utf8_lead_byte_sequence_length(byte) {
            // Validate the complete sequence (continuation bytes, overlong check,
            // surrogate check, range check).
            if let Some(code_point) = validate_utf8_sequence(&bytes[pos..], seq_len) {
                // The sequence is valid UTF-8. Convert the code point to a char
                // and push it to the result string.
                if let Some(ch) = char::from_u32(code_point) {
                    result.push(ch);
                    pos += seq_len;
                    continue;
                }
                // char::from_u32 returning None here should be impossible because
                // validate_utf8_sequence already rejects surrogates and over-range
                // values. Fall through to PUA encoding as a safety measure.
            }
        }

        // The byte is not part of a valid UTF-8 sequence. This covers:
        // - Lone continuation bytes (0x80–0xBF without a valid lead byte)
        // - Invalid lead bytes (0xC0, 0xC1 for overlong; 0xF5–0xFF for over-range)
        // - Lead bytes of truncated sequences (insufficient following bytes)
        // - Lead bytes where continuation bytes are malformed
        // - Sequences that decode to surrogates or overlong encodings
        //
        // Encode this single byte as a PUA character and advance by exactly 1,
        // so subsequent bytes (including would-be continuation bytes of a broken
        // sequence) are processed individually on subsequent iterations.
        result.push(encode_byte_to_pua(byte));
        pos += 1;
    }

    result
}

// ---------------------------------------------------------------------------
// Internal: UTF-8 Validation Helpers (RFC 3629 Compliant)
// ---------------------------------------------------------------------------

/// Determine the expected byte length of a UTF-8 sequence from its lead byte.
///
/// Returns `Some(length)` for bytes that are valid UTF-8 lead bytes:
/// - `0x00`–`0x7F` → 1 (ASCII, though ASCII is handled before reaching this function)
/// - `0xC2`–`0xDF` → 2 (excludes `0xC0`–`0xC1` which can only produce overlong encodings)
/// - `0xE0`–`0xEF` → 3
/// - `0xF0`–`0xF4` → 4 (excludes `0xF5`+ which would encode values above U+10FFFF)
///
/// Returns `None` for invalid lead bytes:
/// - `0x80`–`0xBF`: Continuation bytes, not valid as sequence starts
/// - `0xC0`–`0xC1`: Would produce overlong encodings of ASCII values
/// - `0xF5`–`0xFF`: Would encode code points above the Unicode maximum U+10FFFF
#[inline]
fn utf8_lead_byte_sequence_length(byte: u8) -> Option<usize> {
    match byte {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

/// Validate a UTF-8 multi-byte sequence starting at the given byte slice.
///
/// Performs five checks mandated by RFC 3629:
/// 1. Sufficient bytes are available in the slice for the expected sequence length
/// 2. All continuation bytes (positions 1..expected_len) have the form `10xxxxxx`
/// 3. The decoded code point is not an overlong encoding (uses minimum bytes needed)
/// 4. The code point is not in the surrogate range U+D800–U+DFFF
/// 5. The code point does not exceed the Unicode maximum U+10FFFF
///
/// Returns the decoded Unicode code point on success, or `None` if any check fails.
/// The caller is responsible for converting the code point to a `char` via
/// `char::from_u32()`.
fn validate_utf8_sequence(bytes: &[u8], expected_len: usize) -> Option<u32> {
    // Check 1: Ensure we have enough bytes remaining in the input
    if bytes.len() < expected_len {
        return None;
    }

    // Check 2: Verify all continuation bytes are in the form 10xxxxxx (0x80–0xBF).
    // We check bytes[1] through bytes[expected_len - 1].
    for &byte in &bytes[1..expected_len] {
        if byte & 0xC0 != 0x80 {
            return None;
        }
    }

    // Decode the code point by extracting payload bits from each byte
    let code_point = match expected_len {
        1 => {
            // Single byte: 0xxxxxxx — 7 payload bits
            bytes[0] as u32
        }
        2 => {
            // Two bytes: 110xxxxx 10xxxxxx — 11 payload bits
            ((bytes[0] as u32 & 0x1F) << 6) | (bytes[1] as u32 & 0x3F)
        }
        3 => {
            // Three bytes: 1110xxxx 10xxxxxx 10xxxxxx — 16 payload bits
            ((bytes[0] as u32 & 0x0F) << 12)
                | ((bytes[1] as u32 & 0x3F) << 6)
                | (bytes[2] as u32 & 0x3F)
        }
        4 => {
            // Four bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx — 21 payload bits
            ((bytes[0] as u32 & 0x07) << 18)
                | ((bytes[1] as u32 & 0x3F) << 12)
                | ((bytes[2] as u32 & 0x3F) << 6)
                | (bytes[3] as u32 & 0x3F)
        }
        _ => return None,
    };

    // Check 3: Reject overlong encodings.
    // Each sequence length has a minimum code point value that justifies that length.
    // If the decoded value is below the minimum, it could have been encoded in fewer bytes.
    let min_code_point = match expected_len {
        1 => 0x0000,   // ASCII: any value 0x00–0x7F is valid
        2 => 0x0080,   // 2-byte: must encode U+0080 or higher
        3 => 0x0800,   // 3-byte: must encode U+0800 or higher
        4 => 0x10000,  // 4-byte: must encode U+10000 or higher
        _ => return None,
    };
    if code_point < min_code_point {
        return None;
    }

    // Check 4: Reject surrogate code points (U+D800–U+DFFF).
    // These are reserved for UTF-16 surrogate pairs and are not valid Unicode
    // scalar values. They must never appear in UTF-8.
    if (0xD800..=0xDFFF).contains(&code_point) {
        return None;
    }

    // Check 5: Reject code points above the Unicode maximum (U+10FFFF).
    // While the 4-byte encoding format could theoretically encode up to U+1FFFFF,
    // Unicode restricts the maximum to U+10FFFF.
    if code_point > 0x10FFFF {
        return None;
    }

    Some(code_point)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Core encoding function tests
    // =======================================================================

    #[test]
    fn test_encode_ascii_passthrough() {
        // Every ASCII byte (0x00–0x7F) must pass through as itself
        for byte in 0x00..=0x7Fu8 {
            let ch = encode_byte_to_pua(byte);
            assert_eq!(
                ch, byte as char,
                "ASCII byte 0x{:02X} should pass through unchanged",
                byte
            );
        }
    }

    #[test]
    fn test_encode_non_utf8_bytes_to_pua() {
        // Specific boundary and representative values
        assert_eq!(encode_byte_to_pua(0x80), '\u{E080}');
        assert_eq!(encode_byte_to_pua(0x81), '\u{E081}');
        assert_eq!(encode_byte_to_pua(0xA0), '\u{E0A0}');
        assert_eq!(encode_byte_to_pua(0xC0), '\u{E0C0}');
        assert_eq!(encode_byte_to_pua(0xFE), '\u{E0FE}');
        assert_eq!(encode_byte_to_pua(0xFF), '\u{E0FF}');
    }

    #[test]
    fn test_encode_all_non_ascii_bytes_in_pua_range() {
        // Every byte 0x80–0xFF must map to the PUA range U+E080–U+E0FF
        for byte in 0x80..=0xFFu8 {
            let ch = encode_byte_to_pua(byte);
            assert!(
                is_pua_encoded(ch),
                "Byte 0x{:02X} should be PUA-encoded, got U+{:04X}",
                byte,
                ch as u32
            );
        }
    }

    #[test]
    fn test_decode_pua_to_byte_valid() {
        assert_eq!(decode_pua_to_byte('\u{E080}'), Some(0x80));
        assert_eq!(decode_pua_to_byte('\u{E081}'), Some(0x81));
        assert_eq!(decode_pua_to_byte('\u{E0A0}'), Some(0xA0));
        assert_eq!(decode_pua_to_byte('\u{E0C0}'), Some(0xC0));
        assert_eq!(decode_pua_to_byte('\u{E0FE}'), Some(0xFE));
        assert_eq!(decode_pua_to_byte('\u{E0FF}'), Some(0xFF));
    }

    #[test]
    fn test_decode_pua_to_byte_invalid() {
        // Characters outside the PUA mapping range must return None
        assert_eq!(decode_pua_to_byte('A'), None);
        assert_eq!(decode_pua_to_byte('\0'), None);
        assert_eq!(decode_pua_to_byte('\u{E07F}'), None); // Just below PUA_BASE
        assert_eq!(decode_pua_to_byte('\u{E100}'), None); // Just above PUA_END
        assert_eq!(decode_pua_to_byte('\u{E000}'), None); // Start of broader PUA
        assert_eq!(decode_pua_to_byte('\u{F8FF}'), None); // End of broader PUA
        assert_eq!(decode_pua_to_byte('\u{FFFF}'), None);
        assert_eq!(decode_pua_to_byte('\u{10FFFF}'), None);
    }

    #[test]
    fn test_is_pua_encoded() {
        // In range
        assert!(is_pua_encoded('\u{E080}'));
        assert!(is_pua_encoded('\u{E0A0}'));
        assert!(is_pua_encoded('\u{E0FF}'));

        // Out of range
        assert!(!is_pua_encoded('A'));
        assert!(!is_pua_encoded('\u{E07F}'));
        assert!(!is_pua_encoded('\u{E100}'));
        assert!(!is_pua_encoded('\u{0000}'));
    }

    // =======================================================================
    // Round-trip tests (encode then decode)
    // =======================================================================

    #[test]
    fn test_roundtrip_all_128_non_utf8_bytes() {
        // Verify lossless round-trip for every byte in the 0x80–0xFF range.
        // This is the core guarantee from Section 0.7.9.
        for byte in 0x80..=0xFFu8 {
            let ch = encode_byte_to_pua(byte);
            let decoded = decode_pua_to_byte(ch);
            assert_eq!(
                decoded,
                Some(byte),
                "Round-trip failed for byte 0x{:02X}: encoded to U+{:04X}",
                byte,
                ch as u32
            );
        }
    }

    #[test]
    fn test_roundtrip_mixed_ascii_and_binary() {
        // Simulate: "Hello\x80\xFF world"
        let original: Vec<u8> = vec![
            b'H', b'e', b'l', b'l', b'o', 0x80, 0xFF, b' ', b'w', b'o', b'r', b'l', b'd',
        ];
        let encoded = encode_bytes_to_pua_string(&original);
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, original, "Mixed content round-trip failed");
    }

    #[test]
    fn test_roundtrip_pure_ascii() {
        let original = b"int main(void) { return 0; }\n";
        let encoded = encode_bytes_to_pua_string(original);
        // Pure ASCII should be identical as a string
        assert_eq!(encoded, "int main(void) { return 0; }\n");
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_valid_utf8_multibyte() {
        // Valid UTF-8 multi-byte characters should pass through unchanged
        let original_str = "/* 日本語コメント */\n";
        let original = original_str.as_bytes();
        let encoded = encode_bytes_to_pua_string(original);
        assert_eq!(encoded, original_str);
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_all_256_bytes() {
        // Full 256-byte round-trip: every possible byte value appears exactly once.
        // In this specific sequence (0, 1, 2, ..., 255), no valid multi-byte UTF-8
        // sequences are formed because lead bytes are never immediately followed by
        // appropriate continuation bytes at adjacent positions.
        let original: Vec<u8> = (0..=255u8).collect();
        let encoded = encode_bytes_to_pua_string(&original);
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, original, "Full 256-byte round-trip failed");
    }

    #[test]
    fn test_roundtrip_empty() {
        let encoded = encode_bytes_to_pua_string(&[]);
        assert_eq!(encoded, "");
        let decoded = decode_pua_string(&encoded);
        assert!(decoded.is_empty());
    }

    // =======================================================================
    // UTF-8 validation: valid sequences preserved
    // =======================================================================

    #[test]
    fn test_valid_utf8_2byte_preserved() {
        // U+00E9 (é) = [0xC3, 0xA9]
        let bytes = [0xC3, 0xA9];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result, "é");
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_3byte_preserved() {
        // U+4E16 (世) = [0xE4, 0xB8, 0x96]
        let bytes = [0xE4, 0xB8, 0x96];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result, "世");
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_4byte_preserved() {
        // U+1F600 (😀) = [0xF0, 0x9F, 0x98, 0x80]
        let bytes = [0xF0, 0x9F, 0x98, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result, "😀");
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_boundary_2byte_min() {
        // U+0080 = [0xC2, 0x80] (minimum valid 2-byte)
        let bytes = [0xC2, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result.chars().next().unwrap() as u32, 0x0080);
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_boundary_2byte_max() {
        // U+07FF = [0xDF, 0xBF] (maximum valid 2-byte)
        let bytes = [0xDF, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result.chars().next().unwrap() as u32, 0x07FF);
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_boundary_3byte_min() {
        // U+0800 = [0xE0, 0xA0, 0x80] (minimum valid 3-byte)
        let bytes = [0xE0, 0xA0, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result.chars().next().unwrap() as u32, 0x0800);
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_boundary_4byte_min() {
        // U+10000 = [0xF0, 0x90, 0x80, 0x80] (minimum valid 4-byte)
        let bytes = [0xF0, 0x90, 0x80, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result.chars().next().unwrap() as u32, 0x10000);
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_valid_utf8_boundary_4byte_max() {
        // U+10FFFF = [0xF4, 0x8F, 0xBF, 0xBF] (maximum valid Unicode code point)
        let bytes = [0xF4, 0x8F, 0xBF, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result.chars().next().unwrap() as u32, 0x10FFFF);
        assert_eq!(decode_pua_string(&result), bytes);
    }

    // =======================================================================
    // UTF-8 validation: invalid sequences PUA-encoded
    // =======================================================================

    #[test]
    fn test_invalid_lone_continuation_byte() {
        // 0x80 alone is a continuation byte without a lead byte
        let bytes = [0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 1);
        assert!(is_pua_encoded(result.chars().next().unwrap()));
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_invalid_all_continuation_bytes() {
        // Multiple lone continuation bytes
        let bytes = [0x80, 0x85, 0x90, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        assert_eq!(result.chars().count(), 4);
        for ch in result.chars() {
            assert!(is_pua_encoded(ch));
        }
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_invalid_overlong_c0() {
        // 0xC0 is rejected as a lead byte (would create overlong 2-byte for ASCII)
        let bytes = [0xC0, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        // Both bytes should be individually PUA-encoded
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_overlong_c1() {
        // 0xC1 is rejected as a lead byte (would create overlong 2-byte for ASCII)
        let bytes = [0xC1, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_overlong_3byte() {
        // 0xE0, 0x80, 0x80 → U+0000 (overlong 3-byte for NULL)
        let bytes = [0xE0, 0x80, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_overlong_3byte_below_min() {
        // 0xE0, 0x9F, 0xBF → U+07FF (overlong: should be 2-byte)
        let bytes = [0xE0, 0x9F, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_overlong_4byte() {
        // 0xF0, 0x80, 0x80, 0x80 → U+0000 (overlong 4-byte)
        let bytes = [0xF0, 0x80, 0x80, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_surrogate_low() {
        // 0xED, 0xA0, 0x80 → U+D800 (low surrogate boundary, invalid in UTF-8)
        let bytes = [0xED, 0xA0, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_surrogate_high() {
        // 0xED, 0xBF, 0xBF → U+DFFF (high surrogate boundary, invalid in UTF-8)
        let bytes = [0xED, 0xBF, 0xBF];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_surrogate_middle() {
        // 0xED, 0xAC, 0x80 → U+DB00 (middle of surrogate range)
        let bytes = [0xED, 0xAC, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_over_range_f5() {
        // 0xF5 and above are always invalid lead bytes
        let bytes = [0xF5, 0x80, 0x80, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_over_range_f4_90() {
        // 0xF4, 0x90, 0x80, 0x80 → U+110000 (just above maximum)
        let bytes = [0xF4, 0x90, 0x80, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_invalid_fe_byte() {
        // 0xFE is never a valid UTF-8 byte
        let bytes = [0xFE];
        let result = encode_bytes_to_pua_string(&bytes);
        assert!(is_pua_encoded(result.chars().next().unwrap()));
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_invalid_ff_byte() {
        // 0xFF is never a valid UTF-8 byte
        let bytes = [0xFF];
        let result = encode_bytes_to_pua_string(&bytes);
        assert!(is_pua_encoded(result.chars().next().unwrap()));
        assert_eq!(decode_pua_string(&result), bytes);
    }

    #[test]
    fn test_truncated_2byte_sequence() {
        // 0xC3 alone at end of input (missing continuation byte)
        let bytes = [0xC3];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_truncated_3byte_sequence() {
        // 0xE4, 0xB8 at end of input (missing third byte)
        let bytes = [0xE4, 0xB8];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_truncated_4byte_sequence() {
        // 0xF0, 0x9F, 0x98 at end of input (missing fourth byte)
        let bytes = [0xF0, 0x9F, 0x98];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_broken_continuation_in_3byte() {
        // 0xE4, 0x00, 0x96 — second byte is ASCII, not continuation
        let bytes = [0xE4, 0x00, 0x96];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_broken_continuation_in_4byte() {
        // 0xF0, 0x9F, 0x41, 0x80 — third byte is ASCII 'A', not continuation
        let bytes = [0xF0, 0x9F, 0x41, 0x80];
        let result = encode_bytes_to_pua_string(&bytes);
        let decoded = decode_pua_string(&result);
        assert_eq!(decoded, bytes);
    }

    // =======================================================================
    // Mixed valid and invalid byte tests
    // =======================================================================

    #[test]
    fn test_valid_utf8_followed_by_invalid_byte() {
        // Valid 2-byte 'é' followed by lone 0x80
        let bytes = vec![0xC3, 0xA9, 0x80];
        let encoded = encode_bytes_to_pua_string(&bytes);

        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0], 'é');
        assert!(is_pua_encoded(chars[1]));

        assert_eq!(decode_pua_string(&encoded), bytes);
    }

    #[test]
    fn test_invalid_byte_followed_by_valid_utf8() {
        // Lone 0x80 followed by valid 2-byte 'é'
        let bytes = vec![0x80, 0xC3, 0xA9];
        let encoded = encode_bytes_to_pua_string(&bytes);

        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert!(is_pua_encoded(chars[0]));
        assert_eq!(chars[1], 'é');

        assert_eq!(decode_pua_string(&encoded), bytes);
    }

    #[test]
    fn test_ascii_then_invalid_then_valid_utf8() {
        // "A" + 0xFF + "世" (3-byte UTF-8)
        let bytes = vec![0x41, 0xFF, 0xE4, 0xB8, 0x96];
        let encoded = encode_bytes_to_pua_string(&bytes);

        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 3);
        assert_eq!(chars[0], 'A');
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(chars[2], '世');

        assert_eq!(decode_pua_string(&encoded), bytes);
    }

    // =======================================================================
    // decode_pua_string specific tests
    // =======================================================================

    #[test]
    fn test_decode_pure_ascii() {
        let result = decode_pua_string("Hello, World!");
        assert_eq!(result, b"Hello, World!");
    }

    #[test]
    fn test_decode_with_pua_chars() {
        let s = format!(
            "test{}data{}end",
            encode_byte_to_pua(0x80),
            encode_byte_to_pua(0xFF)
        );
        let result = decode_pua_string(&s);
        assert_eq!(
            result,
            vec![b't', b'e', b's', b't', 0x80, b'd', b'a', b't', b'a', 0xFF, b'e', b'n', b'd']
        );
    }

    #[test]
    fn test_decode_empty_string() {
        assert_eq!(decode_pua_string(""), Vec::<u8>::new());
    }

    #[test]
    fn test_decode_preserves_regular_utf8() {
        let input = "日本語";
        let result = decode_pua_string(input);
        assert_eq!(result, input.as_bytes());
    }

    // =======================================================================
    // Internal UTF-8 helper tests
    // =======================================================================

    #[test]
    fn test_utf8_lead_byte_sequence_length_ascii() {
        assert_eq!(utf8_lead_byte_sequence_length(0x00), Some(1));
        assert_eq!(utf8_lead_byte_sequence_length(0x41), Some(1)); // 'A'
        assert_eq!(utf8_lead_byte_sequence_length(0x7F), Some(1));
    }

    #[test]
    fn test_utf8_lead_byte_sequence_length_continuation() {
        // Continuation bytes are invalid as lead bytes
        assert_eq!(utf8_lead_byte_sequence_length(0x80), None);
        assert_eq!(utf8_lead_byte_sequence_length(0x90), None);
        assert_eq!(utf8_lead_byte_sequence_length(0xBF), None);
    }

    #[test]
    fn test_utf8_lead_byte_sequence_length_overlong() {
        // 0xC0 and 0xC1 are rejected (would create overlong encodings)
        assert_eq!(utf8_lead_byte_sequence_length(0xC0), None);
        assert_eq!(utf8_lead_byte_sequence_length(0xC1), None);
    }

    #[test]
    fn test_utf8_lead_byte_sequence_length_valid_multi() {
        assert_eq!(utf8_lead_byte_sequence_length(0xC2), Some(2));
        assert_eq!(utf8_lead_byte_sequence_length(0xDF), Some(2));
        assert_eq!(utf8_lead_byte_sequence_length(0xE0), Some(3));
        assert_eq!(utf8_lead_byte_sequence_length(0xEF), Some(3));
        assert_eq!(utf8_lead_byte_sequence_length(0xF0), Some(4));
        assert_eq!(utf8_lead_byte_sequence_length(0xF4), Some(4));
    }

    #[test]
    fn test_utf8_lead_byte_sequence_length_over_range() {
        assert_eq!(utf8_lead_byte_sequence_length(0xF5), None);
        assert_eq!(utf8_lead_byte_sequence_length(0xF8), None);
        assert_eq!(utf8_lead_byte_sequence_length(0xFE), None);
        assert_eq!(utf8_lead_byte_sequence_length(0xFF), None);
    }

    #[test]
    fn test_validate_utf8_boundary_values() {
        // Minimum valid 2-byte: U+0080
        assert_eq!(validate_utf8_sequence(&[0xC2, 0x80], 2), Some(0x0080));
        // Maximum valid 2-byte: U+07FF
        assert_eq!(validate_utf8_sequence(&[0xDF, 0xBF], 2), Some(0x07FF));
        // Minimum valid 3-byte: U+0800
        assert_eq!(validate_utf8_sequence(&[0xE0, 0xA0, 0x80], 3), Some(0x0800));
        // U+D7FF (last before surrogates)
        assert_eq!(validate_utf8_sequence(&[0xED, 0x9F, 0xBF], 3), Some(0xD7FF));
        // U+E000 (first after surrogates)
        assert_eq!(validate_utf8_sequence(&[0xEE, 0x80, 0x80], 3), Some(0xE000));
        // Maximum valid 3-byte: U+FFFF
        assert_eq!(validate_utf8_sequence(&[0xEF, 0xBF, 0xBF], 3), Some(0xFFFF));
        // Minimum valid 4-byte: U+10000
        assert_eq!(
            validate_utf8_sequence(&[0xF0, 0x90, 0x80, 0x80], 4),
            Some(0x10000)
        );
        // Maximum valid 4-byte: U+10FFFF
        assert_eq!(
            validate_utf8_sequence(&[0xF4, 0x8F, 0xBF, 0xBF], 4),
            Some(0x10FFFF)
        );
    }

    #[test]
    fn test_validate_utf8_rejects_overlong() {
        // Overlong 3-byte for U+007F
        assert_eq!(validate_utf8_sequence(&[0xE0, 0x81, 0xBF], 3), None);
        // Overlong 4-byte for U+07FF
        assert_eq!(validate_utf8_sequence(&[0xF0, 0x80, 0x9F, 0xBF], 4), None);
    }

    #[test]
    fn test_validate_utf8_rejects_surrogates() {
        assert_eq!(validate_utf8_sequence(&[0xED, 0xA0, 0x80], 3), None); // U+D800
        assert_eq!(validate_utf8_sequence(&[0xED, 0xBF, 0xBF], 3), None); // U+DFFF
        assert_eq!(validate_utf8_sequence(&[0xED, 0xAC, 0x80], 3), None); // U+DB00
    }

    #[test]
    fn test_validate_utf8_rejects_beyond_max() {
        assert_eq!(
            validate_utf8_sequence(&[0xF4, 0x90, 0x80, 0x80], 4),
            None
        ); // U+110000
    }

    #[test]
    fn test_validate_utf8_insufficient_bytes() {
        assert_eq!(validate_utf8_sequence(&[0xE4, 0xB8], 3), None);
        assert_eq!(validate_utf8_sequence(&[0xC3], 2), None);
        assert_eq!(validate_utf8_sequence(&[0xF0, 0x9F], 4), None);
        assert_eq!(validate_utf8_sequence(&[], 1), None);
    }

    // =======================================================================
    // File I/O tests
    // =======================================================================

    #[test]
    fn test_file_roundtrip_with_binary_content() {
        let dir = std::env::temp_dir();
        let path = dir.join("bcc_test_encoding_binary_roundtrip.bin");

        // Original content with various non-UTF-8 bytes interspersed with ASCII
        let original: Vec<u8> = vec![
            b'H', b'e', b'l', b'l', b'o', 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0,
            0xFE, 0xFF, b'\n',
        ];

        // Write the raw bytes directly
        fs::write(&path, &original).expect("write test file");

        // Read with PUA encoding
        let encoded = read_source_file(&path).expect("read source file");

        // Decode and verify byte-exact match
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, original, "File round-trip with binary content failed");

        // Also test write_decoded_file
        let out_path = dir.join("bcc_test_encoding_binary_roundtrip_out.bin");
        write_decoded_file(&out_path, &encoded).expect("write decoded file");
        let result = fs::read(&out_path).expect("read output file");
        assert_eq!(result, original, "write_decoded_file round-trip failed");

        // Cleanup
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn test_file_read_pure_ascii() {
        let dir = std::env::temp_dir();
        let path = dir.join("bcc_test_encoding_ascii.c");

        let content = b"int main(void) { return 0; }\n";
        fs::write(&path, content).expect("write test file");

        let result = read_source_file(&path).expect("read source file");
        // Pure ASCII file should come through the fast path unchanged
        assert_eq!(result, "int main(void) { return 0; }\n");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_file_read_nonexistent() {
        let result = read_source_file(Path::new("/nonexistent/directory/file.c"));
        assert!(result.is_err());
    }

    #[test]
    fn test_file_write_nonexistent_dir() {
        let result = write_decoded_file(
            Path::new("/nonexistent/directory/output.c"),
            "content",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_file_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join("bcc_test_encoding_empty.c");

        fs::write(&path, b"").expect("write test file");

        let result = read_source_file(&path).expect("read source file");
        assert_eq!(result, "");
        assert_eq!(decode_pua_string(&result), Vec::<u8>::new());

        let _ = fs::remove_file(&path);
    }

    // =======================================================================
    // Kernel-relevant scenario tests
    // =======================================================================

    #[test]
    fn test_kernel_binary_string_literal_80_ff() {
        // Simulate source file containing: char data[] = "\x80\xFF";
        // The 0x80 and 0xFF bytes appear directly in the file content.
        // After PUA encoding → pipeline → PUA decoding, the .rodata section
        // must contain exact bytes 80 FF. (Section 0.7.9 validation)
        let mut source_bytes: Vec<u8> = Vec::new();
        source_bytes.extend_from_slice(b"char data[] = \"");
        source_bytes.push(0x80);
        source_bytes.push(0xFF);
        source_bytes.extend_from_slice(b"\";\n");

        let encoded = encode_bytes_to_pua_string(&source_bytes);
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, source_bytes, "Kernel binary string literal round-trip failed");

        // Verify the 0x80 and 0xFF bytes are at the expected positions
        let prefix = b"char data[] = \"";
        assert_eq!(decoded[prefix.len()], 0x80);
        assert_eq!(decoded[prefix.len() + 1], 0xFF);
    }

    #[test]
    fn test_kernel_inline_asm_binary_operand() {
        // Simulate inline assembly with binary data
        let mut source: Vec<u8> = Vec::new();
        source.extend_from_slice(b"asm volatile(\".byte 0x");
        source.push(0xDE);
        source.push(0xAD);
        source.extend_from_slice(b"\");\n");

        let encoded = encode_bytes_to_pua_string(&source);
        let decoded = decode_pua_string(&encoded);
        assert_eq!(decoded, source);
    }

    #[test]
    fn test_all_non_utf8_bytes_individually() {
        // Each non-UTF-8 byte value in isolation must round-trip correctly
        for byte in 0x80..=0xFFu8 {
            let input = vec![byte];
            let encoded = encode_bytes_to_pua_string(&input);
            let decoded = decode_pua_string(&encoded);
            assert_eq!(
                decoded, input,
                "Individual byte 0x{:02X} failed round-trip",
                byte
            );
        }
    }
}
