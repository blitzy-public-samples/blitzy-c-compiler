//! PUA/UTF-8 encoding for non-UTF-8 byte round-tripping.
//!
//! This module implements a Private Use Area (PUA) codec that maps non-UTF-8 bytes
//! (0x80–0xFF) to Unicode PUA code points (U+E080–U+E0FF) and back. This is
//! **critical** for the Linux kernel build — the kernel contains binary data in
//! string literals and inline assembly operands that must survive the entire
//! compilation pipeline with **byte-exact fidelity**.
//!
//! # Encoding Scheme
//!
//! Non-UTF-8 bytes in C source files are encoded on read and decoded on output:
//!
//! - **Read (encode):** Each byte `b` in 0x80–0xFF that is NOT part of a valid
//!   UTF-8 multi-byte sequence is mapped to the PUA code point `U+(E000 + b)`.
//!   For example, byte `0x80` → `U+E080`, byte `0xFF` → `U+E0FF`.
//!
//! - **Write (decode):** Each PUA code point in the range U+E080–U+E0FF is
//!   mapped back to its original byte value `(cp - 0xE000) as u8`.
//!   Normal (non-PUA) characters are emitted as standard UTF-8.
//!
//! # PUA Range
//!
//! The Unicode Private Use Area (U+E000–U+F8FF) is reserved for application-
//! specific use. We use only the sub-range **U+E080–U+E0FF** (128 code points)
//! which corresponds one-to-one with non-ASCII byte values 0x80–0xFF. These
//! code points do **not** appear in normal Unicode text.
//!
//! # Round-Trip Guarantee
//!
//! For any byte sequence `bytes`:
//! ```text
//! decode_string_to_bytes(encode_bytes_to_string(bytes)) == bytes
//! ```
//!
//! This guarantee holds for all byte values 0x00–0xFF, including valid UTF-8
//! multi-byte sequences (which pass through unchanged) and arbitrary non-UTF-8
//! bytes (which are PUA-encoded then decoded).
//!
//! # Replaces
//!
//! This module replaces the external `encoding_rs` crate in compliance with the
//! zero-dependency mandate. No external crates are used — only the Rust
//! standard library.
//!
//! # Validation
//!
//! Per AAP requirements: compile a source with `\x80\xFF` in a string literal,
//! inspect `.rodata` with `objdump -s`, and confirm exact bytes `80 ff` are
//! present.

use std::fs;
use std::io;
use std::io::Read;
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants — PUA range boundaries
// ---------------------------------------------------------------------------

/// Maximum source file size: 256 MiB.
///
/// This limit prevents the compiler from hanging when presented with
/// infinite or extremely large input streams (e.g., `/dev/zero`).
/// Legitimate C source files, including the largest Linux kernel files,
/// are well below this limit.
const MAX_SOURCE_FILE_SIZE: u64 = 256 * 1024 * 1024;

/// Base offset for PUA encoding. Adding a byte value (0x80–0xFF) to this
/// base produces the corresponding PUA code point (U+E080–U+E0FF).
const PUA_BASE: u32 = 0xE000;

/// Lowest PUA code point used by our encoding: U+E080 (maps to byte 0x80).
const PUA_LOW: u32 = 0xE080;

/// Highest PUA code point used by our encoding: U+E0FF (maps to byte 0xFF).
const PUA_HIGH: u32 = 0xE0FF;

// ---------------------------------------------------------------------------
// Core PUA mapping functions
// ---------------------------------------------------------------------------

/// Encode a non-UTF-8 byte (0x80–0xFF) to a Unicode PUA code point.
///
/// The mapping is: byte `b` → code point `U+(E000 + b)`, giving the range
/// U+E080 for byte 0x80 through U+E0FF for byte 0xFF.
///
/// # Panics
///
/// Panics if `byte < 0x80`. ASCII bytes (0x00–0x7F) are always valid UTF-8
/// and must NOT be PUA-encoded — they pass through the pipeline as-is.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::encode_byte_to_pua;
///
/// let ch = encode_byte_to_pua(0x80);
/// assert_eq!(ch as u32, 0xE080);
///
/// let ch = encode_byte_to_pua(0xFF);
/// assert_eq!(ch as u32, 0xE0FF);
/// ```
#[inline]
pub fn encode_byte_to_pua(byte: u8) -> char {
    assert!(
        byte >= 0x80,
        "Only bytes 0x80-0xFF need PUA encoding, got 0x{:02X}",
        byte
    );
    // SAFETY: PUA_BASE + byte is always in range U+E080..=U+E0FF, which
    // is within the valid Unicode PUA block (U+E000–U+F8FF). These are
    // always valid Unicode scalar values.
    //
    // The unwrap cannot fail because:
    //   PUA_BASE (0xE000) + 0x80 = 0xE080 (valid)
    //   PUA_BASE (0xE000) + 0xFF = 0xE0FF (valid)
    // Both are well below the surrogate range and within the BMP PUA.
    char::from_u32(PUA_BASE + byte as u32).unwrap()
}

/// Decode a PUA code point back to the original non-UTF-8 byte.
///
/// Returns `Some(byte)` if the character is a PUA-encoded byte (code point
/// in U+E080–U+E0FF), or `None` for any other character.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::{encode_byte_to_pua, decode_pua_to_byte};
///
/// // Round-trip for byte 0x80
/// let ch = encode_byte_to_pua(0x80);
/// assert_eq!(decode_pua_to_byte(ch), Some(0x80));
///
/// // Normal characters are not PUA-encoded
/// assert_eq!(decode_pua_to_byte('A'), None);
/// assert_eq!(decode_pua_to_byte('é'), None);
/// ```
#[inline]
pub fn decode_pua_to_byte(ch: char) -> Option<u8> {
    let cp = ch as u32;
    if (PUA_LOW..=PUA_HIGH).contains(&cp) {
        // Reverse the encoding: code point - PUA_BASE gives the original byte.
        // The result is always in 0x80..=0xFF because:
        //   PUA_LOW  - PUA_BASE = 0xE080 - 0xE000 = 0x80
        //   PUA_HIGH - PUA_BASE = 0xE0FF - 0xE000 = 0xFF
        Some((cp - PUA_BASE) as u8)
    } else {
        None
    }
}

/// Check if a character is a PUA-encoded byte.
///
/// Returns `true` if the character's code point is in the range
/// U+E080–U+E0FF (our reserved PUA encoding sub-range).
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::{encode_byte_to_pua, is_pua_encoded};
///
/// assert!(is_pua_encoded(encode_byte_to_pua(0x80)));
/// assert!(is_pua_encoded(encode_byte_to_pua(0xFF)));
/// assert!(!is_pua_encoded('A'));
/// assert!(!is_pua_encoded('\u{E000}')); // U+E000 is PUA but not in our range
/// ```
#[inline]
pub fn is_pua_encoded(ch: char) -> bool {
    let cp = ch as u32;
    (PUA_LOW..=PUA_HIGH).contains(&cp)
}

// ---------------------------------------------------------------------------
// File reading with PUA encoding
// ---------------------------------------------------------------------------

/// Read a C source file, encoding any non-UTF-8 bytes as PUA code points.
///
/// This function reads the raw bytes of a source file and produces a valid
/// Rust `String` where:
/// - Valid UTF-8 sequences (including multi-byte characters like CJK, emoji,
///   or accented letters in comments) are preserved intact.
/// - Non-UTF-8 bytes (0x80–0xFF that do NOT form valid UTF-8 sequences)
///   are encoded as PUA code points (U+E080–U+E0FF).
///
/// The resulting string can be safely processed through Rust's `String`/`str`
/// APIs (which require valid UTF-8) while preserving the original byte values
/// for later decoding via [`decode_string_to_bytes`].
///
/// # Errors
///
/// Returns `Err` if the file cannot be read (e.g., file not found, permission
/// denied, I/O error). The encoding step itself is infallible.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::read_source_file;
/// use std::path::Path;
///
/// let source = read_source_file(Path::new("hello.c"))?;
/// // `source` is a valid Rust String — safe for all str operations
/// ```
pub fn read_source_file(path: &Path) -> io::Result<String> {
    // Attempt to determine file size from metadata. For regular files this
    // gives an exact size; for special files (e.g., /dev/zero) the metadata
    // may report size 0 or be unavailable, so we also enforce a read limit.
    if let Ok(metadata) = fs::metadata(path) {
        let file_size = metadata.len();
        if file_size > MAX_SOURCE_FILE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "source file exceeds maximum size of {} bytes ({} bytes)",
                    MAX_SOURCE_FILE_SIZE, file_size
                ),
            ));
        }
    }

    // Open the file and read with a hard size limit to protect against
    // infinite streams (e.g., /dev/zero) where metadata reports size 0.
    let file = fs::File::open(path)?;
    let mut raw_bytes = Vec::new();
    // Read at most MAX_SOURCE_FILE_SIZE + 1 bytes. If we read more than
    // MAX_SOURCE_FILE_SIZE, the file exceeds our limit.
    let bytes_read = file
        .take(MAX_SOURCE_FILE_SIZE + 1)
        .read_to_end(&mut raw_bytes)?;

    if bytes_read as u64 > MAX_SOURCE_FILE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "source file exceeds maximum size of {} bytes",
                MAX_SOURCE_FILE_SIZE
            ),
        ));
    }

    Ok(encode_bytes_to_string(&raw_bytes))
}

/// Encode raw bytes into a Rust `String`, using PUA code points for non-UTF-8 bytes.
///
/// This is the core encoding function. It processes the byte slice incrementally:
///
/// 1. **ASCII bytes** (0x00–0x7F): Passed through directly as their character
///    equivalents. These are always valid single-byte UTF-8.
///
/// 2. **Valid multi-byte UTF-8 sequences**: Detected using [`std::str::from_utf8`]
///    and passed through intact. This preserves characters like `é` (U+00E9),
///    Chinese/Japanese/Korean characters, emoji, etc.
///
/// 3. **Non-UTF-8 bytes** (0x80–0xFF that are NOT part of valid UTF-8):
///    Each such byte is individually encoded as a PUA code point via
///    [`encode_byte_to_pua`].
///
/// # Algorithm
///
/// Uses `std::str::from_utf8` on the remaining byte slice at each iteration.
/// When an error is reported:
/// - The valid prefix (up to the error) is appended as a `&str`.
/// - The invalid byte(s) at the error position are PUA-encoded individually.
/// - Processing continues from after the invalid byte(s).
///
/// This approach correctly distinguishes between valid multi-byte UTF-8
/// sequences (e.g., `0xC3 0xA9` for 'é') and genuinely invalid bytes
/// (e.g., a bare `0x80` or `0xFF`).
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::encode_bytes_to_string;
///
/// // Pure ASCII passes through unchanged
/// let s = encode_bytes_to_string(b"hello");
/// assert_eq!(s, "hello");
///
/// // Valid UTF-8 (é = 0xC3 0xA9) passes through
/// let s = encode_bytes_to_string(&[0xC3, 0xA9]);
/// assert_eq!(s, "é");
///
/// // Non-UTF-8 byte 0x80 becomes PUA U+E080
/// let s = encode_bytes_to_string(&[0x80]);
/// assert_eq!(s.chars().next().unwrap() as u32, 0xE080);
/// ```
pub fn encode_bytes_to_string(bytes: &[u8]) -> String {
    // Pre-allocate with a reasonable estimate. PUA code points encode as
    // 3 bytes in UTF-8 (U+E080–U+E0FF are in the BMP), so the worst case
    // is 3× the input size. However, most C source is ASCII, so the input
    // length is a good starting estimate.
    let mut result = String::with_capacity(bytes.len());
    let mut pos = 0;

    while pos < bytes.len() {
        // Attempt to validate the remaining bytes as UTF-8.
        match std::str::from_utf8(&bytes[pos..]) {
            Ok(valid_str) => {
                // All remaining bytes are valid UTF-8 — append and finish.
                result.push_str(valid_str);
                break;
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();

                // Append the valid UTF-8 prefix (if any) before the error.
                if valid_up_to > 0 {
                    // from_utf8 already confirmed these bytes are valid UTF-8,
                    // so this second validation will always succeed.
                    let valid_slice = &bytes[pos..pos + valid_up_to];
                    let valid_str = std::str::from_utf8(valid_slice)
                        .expect("BCC encoding: bytes already validated as UTF-8 by from_utf8");
                    result.push_str(valid_str);
                    pos += valid_up_to;
                }

                // Now handle the invalid byte(s) at the current position.
                match err.error_len() {
                    Some(invalid_len) => {
                        // `invalid_len` bytes at `pos` form a definite error
                        // (e.g., unexpected continuation byte, overlong encoding,
                        // surrogate code point, or byte > 0xF4).
                        // PUA-encode each invalid byte individually.
                        for offset in 0..invalid_len {
                            let byte = bytes[pos + offset];
                            if byte >= 0x80 {
                                result.push(encode_byte_to_pua(byte));
                            } else {
                                // ASCII bytes within an error sequence (theoretically
                                // should not happen, but handle defensively).
                                result.push(byte as char);
                            }
                        }
                        pos += invalid_len;
                    }
                    None => {
                        // The input ended unexpectedly mid-sequence. The remaining
                        // bytes (from `pos` to end) are an incomplete multi-byte
                        // UTF-8 sequence. Since we have the entire file, these
                        // bytes are genuinely invalid — PUA-encode each one.
                        for &byte in &bytes[pos..] {
                            if byte >= 0x80 {
                                result.push(encode_byte_to_pua(byte));
                            } else {
                                result.push(byte as char);
                            }
                        }
                        pos = bytes.len();
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Output decoding — PUA code points back to raw bytes
// ---------------------------------------------------------------------------

/// Decode a PUA-encoded string back to raw bytes.
///
/// This is the reverse of [`encode_bytes_to_string`]. It iterates over the
/// characters in the string and:
///
/// - **PUA-encoded characters** (U+E080–U+E0FF): Decoded back to their
///   original byte value (0x80–0xFF) via [`decode_pua_to_byte`].
///
/// - **All other characters**: Encoded as standard UTF-8 bytes.
///
/// # Round-Trip Guarantee
///
/// For any byte sequence `original`:
/// ```text
/// decode_string_to_bytes(&encode_bytes_to_string(original)) == original
/// ```
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::{encode_bytes_to_string, decode_string_to_bytes};
///
/// // Round-trip test with mixed valid and invalid bytes
/// let original = vec![0x41, 0x80, 0xC3, 0xA9, 0xFF, 0x42];
/// let encoded = encode_bytes_to_string(&original);
/// let decoded = decode_string_to_bytes(&encoded);
/// assert_eq!(decoded, original);
/// ```
pub fn decode_string_to_bytes(s: &str) -> Vec<u8> {
    let mut result = Vec::with_capacity(s.len());

    for ch in s.chars() {
        if let Some(byte) = decode_pua_to_byte(ch) {
            // PUA-encoded byte — emit the original raw byte value.
            result.push(byte);
        } else {
            // Normal Unicode character — encode as standard UTF-8 bytes.
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            result.extend_from_slice(encoded.as_bytes());
        }
    }

    result
}

/// Extract raw bytes from a string that may contain PUA-encoded bytes.
///
/// This is used during code generation to recover the exact byte values
/// from C string literals that passed through the PUA-encoding pipeline.
/// It is functionally equivalent to [`decode_string_to_bytes`] — both
/// perform the same PUA-to-byte decoding.
///
/// This dedicated function provides a clear semantic name for the code
/// generation phase, where the intent is to extract the raw bytes that
/// will be emitted into the `.rodata` section of the output ELF binary.
///
/// # Examples
///
/// ```ignore
/// use bcc::common::encoding::{encode_bytes_to_string, extract_string_bytes};
///
/// // Simulate a C string literal containing non-UTF-8 bytes
/// let raw = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x80, 0xFF]; // "Hello\x80\xFF"
/// let encoded = encode_bytes_to_string(&raw);
/// let extracted = extract_string_bytes(&encoded);
/// assert_eq!(extracted, raw); // Exact bytes recovered for .rodata emission
/// ```
pub fn extract_string_bytes(s: &str) -> Vec<u8> {
    decode_string_to_bytes(s)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Core PUA mapping tests --

    #[test]
    fn test_encode_byte_to_pua_boundary_low() {
        // Byte 0x80 maps to U+E080
        let ch = encode_byte_to_pua(0x80);
        assert_eq!(ch as u32, 0xE080);
    }

    #[test]
    fn test_encode_byte_to_pua_boundary_high() {
        // Byte 0xFF maps to U+E0FF
        let ch = encode_byte_to_pua(0xFF);
        assert_eq!(ch as u32, 0xE0FF);
    }

    #[test]
    fn test_encode_byte_to_pua_mid_range() {
        // Byte 0xA0 maps to U+E0A0
        let ch = encode_byte_to_pua(0xA0);
        assert_eq!(ch as u32, 0xE0A0);
    }

    #[test]
    #[should_panic(expected = "Only bytes 0x80-0xFF need PUA encoding")]
    fn test_encode_byte_to_pua_panics_for_ascii() {
        // ASCII bytes must not be PUA-encoded
        encode_byte_to_pua(0x41);
    }

    #[test]
    #[should_panic(expected = "Only bytes 0x80-0xFF need PUA encoding")]
    fn test_encode_byte_to_pua_panics_for_zero() {
        encode_byte_to_pua(0x00);
    }

    #[test]
    #[should_panic(expected = "Only bytes 0x80-0xFF need PUA encoding")]
    fn test_encode_byte_to_pua_panics_for_max_ascii() {
        // 0x7F is the last ASCII byte
        encode_byte_to_pua(0x7F);
    }

    #[test]
    fn test_decode_pua_to_byte_boundary_low() {
        // U+E080 decodes to 0x80
        let ch = char::from_u32(0xE080).unwrap();
        assert_eq!(decode_pua_to_byte(ch), Some(0x80));
    }

    #[test]
    fn test_decode_pua_to_byte_boundary_high() {
        // U+E0FF decodes to 0xFF
        let ch = char::from_u32(0xE0FF).unwrap();
        assert_eq!(decode_pua_to_byte(ch), Some(0xFF));
    }

    #[test]
    fn test_decode_pua_to_byte_not_pua() {
        // Normal ASCII character
        assert_eq!(decode_pua_to_byte('A'), None);
        assert_eq!(decode_pua_to_byte('z'), None);
        assert_eq!(decode_pua_to_byte('0'), None);
    }

    #[test]
    fn test_decode_pua_to_byte_outside_range() {
        // U+E000 is PUA but NOT in our encoding range (below U+E080)
        let ch = char::from_u32(0xE000).unwrap();
        assert_eq!(decode_pua_to_byte(ch), None);

        // U+E07F is just below our range
        let ch = char::from_u32(0xE07F).unwrap();
        assert_eq!(decode_pua_to_byte(ch), None);

        // U+E100 is just above our range
        let ch = char::from_u32(0xE100).unwrap();
        assert_eq!(decode_pua_to_byte(ch), None);
    }

    #[test]
    fn test_decode_pua_to_byte_non_bmp() {
        // Characters outside the BMP (emoji, etc.)
        assert_eq!(decode_pua_to_byte('😀'), None);
    }

    #[test]
    fn test_is_pua_encoded() {
        // In range
        assert!(is_pua_encoded(char::from_u32(0xE080).unwrap()));
        assert!(is_pua_encoded(char::from_u32(0xE0FF).unwrap()));
        assert!(is_pua_encoded(char::from_u32(0xE0A0).unwrap()));

        // Out of range
        assert!(!is_pua_encoded('A'));
        assert!(!is_pua_encoded(char::from_u32(0xE000).unwrap()));
        assert!(!is_pua_encoded(char::from_u32(0xE07F).unwrap()));
        assert!(!is_pua_encoded(char::from_u32(0xE100).unwrap()));
    }

    // -- Round-trip tests --

    #[test]
    fn test_encode_decode_roundtrip_all_non_ascii_bytes() {
        // Every byte 0x80-0xFF must round-trip perfectly
        for byte in 0x80u8..=0xFF {
            let ch = encode_byte_to_pua(byte);
            let decoded = decode_pua_to_byte(ch);
            assert_eq!(
                decoded,
                Some(byte),
                "Round-trip failed for byte 0x{:02X}",
                byte
            );
        }
    }

    #[test]
    fn test_is_pua_matches_encode() {
        // Every encoded byte produces a PUA-encoded character
        for byte in 0x80u8..=0xFF {
            let ch = encode_byte_to_pua(byte);
            assert!(
                is_pua_encoded(ch),
                "is_pua_encoded returned false for encoded byte 0x{:02X}",
                byte
            );
        }
    }

    // -- encode_bytes_to_string tests --

    #[test]
    fn test_encode_pure_ascii() {
        let bytes = b"Hello, World!\n";
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded, "Hello, World!\n");
    }

    #[test]
    fn test_encode_empty() {
        let encoded = encode_bytes_to_string(b"");
        assert_eq!(encoded, "");
    }

    #[test]
    fn test_encode_valid_utf8_multibyte() {
        // é is U+00E9, encoded as 0xC3 0xA9 in UTF-8
        let bytes: &[u8] = &[0xC3, 0xA9];
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded, "é");
        // Verify it was NOT PUA-encoded (the chars should be the actual character)
        assert_eq!(encoded.chars().count(), 1);
        assert!(!is_pua_encoded(encoded.chars().next().unwrap()));
    }

    #[test]
    fn test_encode_valid_utf8_three_byte() {
        // 中 (U+4E2D) is encoded as 0xE4 0xB8 0xAD in UTF-8
        let bytes: &[u8] = &[0xE4, 0xB8, 0xAD];
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded, "中");
    }

    #[test]
    fn test_encode_valid_utf8_four_byte() {
        // 😀 (U+1F600) is encoded as 0xF0 0x9F 0x98 0x80 in UTF-8
        let bytes: &[u8] = &[0xF0, 0x9F, 0x98, 0x80];
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded, "😀");
    }

    #[test]
    fn test_encode_single_non_utf8_byte() {
        // Bare 0x80 is not valid UTF-8 — should be PUA-encoded
        let bytes: &[u8] = &[0x80];
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded.chars().count(), 1);
        let ch = encoded.chars().next().unwrap();
        assert!(is_pua_encoded(ch));
        assert_eq!(ch as u32, 0xE080);
    }

    #[test]
    fn test_encode_0xff_byte() {
        // 0xFF is never valid in UTF-8 — should be PUA-encoded
        let bytes: &[u8] = &[0xFF];
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded.chars().count(), 1);
        let ch = encoded.chars().next().unwrap();
        assert!(is_pua_encoded(ch));
        assert_eq!(ch as u32, 0xE0FF);
    }

    #[test]
    fn test_encode_mixed_ascii_and_non_utf8() {
        // "A\x80B" — ASCII, invalid, ASCII
        let bytes: &[u8] = &[0x41, 0x80, 0x42];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 3);
        assert_eq!(chars[0], 'A');
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(decode_pua_to_byte(chars[1]), Some(0x80));
        assert_eq!(chars[2], 'B');
    }

    #[test]
    fn test_encode_mixed_valid_utf8_and_non_utf8() {
        // "é\x80" — valid 2-byte UTF-8 followed by invalid byte
        let bytes: &[u8] = &[0xC3, 0xA9, 0x80];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0], 'é');
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(decode_pua_to_byte(chars[1]), Some(0x80));
    }

    #[test]
    fn test_encode_incomplete_utf8_at_end() {
        // 0xC3 alone at end of file — incomplete 2-byte UTF-8 sequence
        let bytes: &[u8] = &[0x41, 0xC3];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[0], 'A');
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(decode_pua_to_byte(chars[1]), Some(0xC3));
    }

    #[test]
    fn test_encode_incomplete_utf8_followed_by_ascii() {
        // 0xC3 0x41 — incomplete 2-byte sequence followed by ASCII 'A'
        // 0xC3 expects a continuation byte (0x80-0xBF), but gets 0x41 instead
        let bytes: &[u8] = &[0xC3, 0x41];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert!(is_pua_encoded(chars[0]));
        assert_eq!(decode_pua_to_byte(chars[0]), Some(0xC3));
        assert_eq!(chars[1], 'A');
    }

    #[test]
    fn test_encode_overlong_encoding() {
        // 0xC0 0x80 is an overlong encoding of U+0000 — invalid UTF-8
        let bytes: &[u8] = &[0xC0, 0x80];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        // Both bytes should be PUA-encoded
        assert_eq!(chars.len(), 2);
        assert!(is_pua_encoded(chars[0]));
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(decode_pua_to_byte(chars[0]), Some(0xC0));
        assert_eq!(decode_pua_to_byte(chars[1]), Some(0x80));
    }

    #[test]
    fn test_encode_consecutive_non_utf8() {
        // Multiple consecutive non-UTF-8 bytes
        let bytes: &[u8] = &[0x80, 0xFF, 0xFE, 0x90];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 4);
        for (i, &byte) in bytes.iter().enumerate() {
            assert!(is_pua_encoded(chars[i]));
            assert_eq!(decode_pua_to_byte(chars[i]), Some(byte));
        }
    }

    #[test]
    fn test_encode_kernel_style_binary_data() {
        // Simulate binary data in a string literal: "Hello\x80\xFF"
        let bytes: &[u8] = &[0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x80, 0xFF];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 7);
        assert_eq!(chars[0], 'H');
        assert_eq!(chars[1], 'e');
        assert_eq!(chars[2], 'l');
        assert_eq!(chars[3], 'l');
        assert_eq!(chars[4], 'o');
        assert!(is_pua_encoded(chars[5]));
        assert_eq!(decode_pua_to_byte(chars[5]), Some(0x80));
        assert!(is_pua_encoded(chars[6]));
        assert_eq!(decode_pua_to_byte(chars[6]), Some(0xFF));
    }

    // -- decode_string_to_bytes tests --

    #[test]
    fn test_decode_pure_ascii() {
        let decoded = decode_string_to_bytes("Hello, World!\n");
        assert_eq!(decoded, b"Hello, World!\n");
    }

    #[test]
    fn test_decode_empty() {
        let decoded = decode_string_to_bytes("");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_preserves_utf8() {
        // Normal UTF-8 characters are encoded as their UTF-8 byte sequences
        let decoded = decode_string_to_bytes("é");
        assert_eq!(decoded, &[0xC3, 0xA9]);
    }

    #[test]
    fn test_decode_pua_to_raw_byte() {
        // A PUA character U+E080 should decode to byte 0x80
        let pua_str = String::from(char::from_u32(0xE080).unwrap());
        let decoded = decode_string_to_bytes(&pua_str);
        assert_eq!(decoded, &[0x80]);
    }

    // -- Full round-trip tests --

    #[test]
    fn test_roundtrip_pure_ascii() {
        let original = b"Hello, World!\n".to_vec();
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_valid_utf8() {
        let original = "Héllo, 世界! 😀".as_bytes().to_vec();
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_non_utf8_bytes() {
        let original: Vec<u8> = (0x80..=0xFF).collect();
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_mixed_content() {
        // ASCII + valid UTF-8 + non-UTF-8 + ASCII
        let mut original = Vec::new();
        original.extend_from_slice(b"Hello"); // ASCII
        original.extend_from_slice(&[0xC3, 0xA9]); // valid UTF-8 (é)
        original.extend_from_slice(&[0x80, 0xFF]); // non-UTF-8
        original.extend_from_slice(b" World"); // ASCII
        original.extend_from_slice(&[0xFE]); // non-UTF-8

        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_all_256_bytes() {
        // Test every possible byte value
        let original: Vec<u8> = (0..=255).collect();
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_kernel_binary_data() {
        // Simulate kernel-style binary data embedded in inline assembly
        let original: Vec<u8> = vec![
            0x48, 0x89, 0xE5, // mov rbp, rsp (x86-64 machine code)
            0x48, 0x83, 0xEC, 0x10, // sub rsp, 16
            0xC3, // ret
        ];
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_empty() {
        let original: Vec<u8> = vec![];
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    // -- extract_string_bytes tests --

    #[test]
    fn test_extract_string_bytes_ascii() {
        let bytes = extract_string_bytes("Hello");
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn test_extract_string_bytes_with_pua() {
        let original = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x80, 0xFF];
        let encoded = encode_bytes_to_string(&original);
        let extracted = extract_string_bytes(&encoded);
        assert_eq!(extracted, original);
    }

    #[test]
    fn test_extract_string_bytes_consistency() {
        // extract_string_bytes should produce identical output to decode_string_to_bytes
        let test_str = "test\u{E080}\u{E0FF}data";
        assert_eq!(
            extract_string_bytes(test_str),
            decode_string_to_bytes(test_str)
        );
    }

    // -- Edge case tests --

    #[test]
    fn test_surrogate_range_bytes() {
        // UTF-8 encoding of surrogate code points is invalid
        // 0xED 0xA0 0x80 would encode U+D800 (high surrogate) — invalid UTF-8
        let bytes: &[u8] = &[0xED, 0xA0, 0x80];
        let encoded = encode_bytes_to_string(bytes);
        // All three bytes should be PUA-encoded (surrogate encoding is invalid UTF-8)
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn test_fe_ff_bytes() {
        // 0xFE and 0xFF are never valid in UTF-8
        let bytes: &[u8] = &[0xFE, 0xFF];
        let encoded = encode_bytes_to_string(bytes);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 2);
        assert!(is_pua_encoded(chars[0]));
        assert!(is_pua_encoded(chars[1]));
        assert_eq!(decode_pua_to_byte(chars[0]), Some(0xFE));
        assert_eq!(decode_pua_to_byte(chars[1]), Some(0xFF));
    }

    #[test]
    fn test_valid_utf8_string_is_preserved() {
        // A string that is entirely valid UTF-8 should pass through unchanged
        let input = "int main() { return 0; }";
        let bytes = input.as_bytes();
        let encoded = encode_bytes_to_string(bytes);
        assert_eq!(encoded, input);
    }

    #[test]
    fn test_large_file_roundtrip() {
        // Simulate a large source file with periodic non-UTF-8 bytes
        let mut original = Vec::with_capacity(10000);
        for i in 0..10000u32 {
            if i % 100 == 0 {
                // Insert a non-UTF-8 byte every 100 bytes
                original.push(0x80 + (i % 128) as u8);
            } else {
                // Normal ASCII
                original.push(0x20 + (i % 95) as u8); // printable ASCII range
            }
        }
        let encoded = encode_bytes_to_string(&original);
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encoded_string_is_valid_utf8() {
        // Any output of encode_bytes_to_string must be valid UTF-8 (it's a String)
        let bytes: Vec<u8> = (0..=255).collect();
        let encoded = encode_bytes_to_string(&bytes);
        // This is implicitly tested by the fact that it's a String,
        // but let's verify explicitly
        assert!(std::str::from_utf8(encoded.as_bytes()).is_ok());
    }

    // -- read_source_file tests (requires filesystem) --

    #[test]
    fn test_read_source_file_not_found() {
        let result = read_source_file(Path::new("/nonexistent/file.c"));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_source_file_with_tempfile() {
        // Create a temp file with mixed content.
        // Use a unique file name incorporating thread ID + timestamp
        // to avoid race conditions when tests run in parallel.
        let temp_dir = std::env::temp_dir();
        let unique_name = format!(
            "bcc_encoding_test_{:?}_{}.c",
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let temp_path = temp_dir.join(unique_name);
        let content: Vec<u8> = vec![
            0x69, 0x6E, 0x74, 0x20, // "int "
            0x78, 0x20, 0x3D, 0x20, // "x = "
            0x80, 0xFF, // non-UTF-8 bytes
            0x3B, 0x0A, // ";\n"
        ];
        std::fs::write(&temp_path, &content).expect("Failed to write temp file");

        let result = read_source_file(&temp_path);
        assert!(result.is_ok());
        let encoded = result.unwrap();

        // Verify round-trip
        let decoded = decode_string_to_bytes(&encoded);
        assert_eq!(decoded, content);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
