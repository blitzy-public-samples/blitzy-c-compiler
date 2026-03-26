//! `##` token concatenation and `#` stringification operators for C11 macro
//! expansion.
//!
//! Implements the two macro-replacement-list operators defined in C11 §6.10.3:
//!
//! - **`##` concatenation** (§6.10.3.3): joins two preprocessing tokens into
//!   one by concatenating their text and re-classifying the result as a single
//!   preprocessing token.  If the concatenated text does not form a valid
//!   preprocessing token, the behaviour is undefined per C11, but BCC diagnoses
//!   it with a warning and preserves the original tokens.
//!
//! - **`#` stringification** (§6.10.3.2): converts a macro argument token
//!   sequence to a string literal with proper escaping of `\` and `"`
//!   characters and whitespace normalisation.
//!
//! - **Placemarker tokens**: when `##` involves an empty macro argument, a
//!   "placemarker" token stands in.  Concatenating a placemarker with a real
//!   token yields the real token unchanged.
//!
//! These operators are invoked exclusively by [`super::macro_expander`] during
//! macro substitution.  Arguments used with `#` or `##` are **not**
//! macro-expanded before the operation; only normal-context arguments undergo
//! pre-expansion (C11 §6.10.3.1¶1).
//!
//! # Zero-Dependency
//!
//! This module depends only on `std` and `crate::` references — no external
//! crates.

use super::{PPToken, PPTokenKind};
use crate::common::diagnostics::{Diagnostic, Span};

// ===========================================================================
// PasteError — ## concatenation error
// ===========================================================================

/// Error produced when `##` token concatenation yields an invalid result.
///
/// Per C11 §6.10.3.3, if the concatenated text does not form a valid
/// preprocessing token, the behaviour is undefined.  BCC reports this as a
/// warning rather than silently miscompiling, and preserves the original
/// tokens in the output.
#[derive(Debug)]
pub enum PasteError {
    /// The concatenated text does not form a valid preprocessing token.
    ///
    /// Contains the invalid result text for diagnostic reporting.
    InvalidToken(String),
}

// ===========================================================================
// paste_tokens — ## operator on two tokens
// ===========================================================================

/// Paste two preprocessing tokens via the `##` operator (C11 §6.10.3.3).
///
/// Concatenates the text of `left` and `right`, then re-tokenises the result
/// as a single preprocessing token.  The resulting token is always marked
/// `from_macro = true` since it is produced by macro expansion.
///
/// # Placemarker Handling
///
/// When `##` is applied to an empty macro argument, a placemarker token is
/// used as the operand.  The rules are:
///
/// | Left            | Right           | Result                      |
/// |-----------------|-----------------|-----------------------------|
/// | Placemarker     | Placemarker     | Placemarker                 |
/// | Placemarker     | Real token      | Real token (from_macro=true)|
/// | Real token      | Placemarker     | Real token (from_macro=true)|
/// | Real token      | Real token      | Concatenated & re-classified|
///
/// # Errors
///
/// Returns [`PasteError::InvalidToken`] if the concatenated text does not
/// form a valid preprocessing token (identifier, pp-number, or punctuator).
///
/// # Examples
///
/// ```ignore
/// // Identifier concatenation: foo ## bar → foobar
/// let result = paste_tokens(&foo_tok, &bar_tok)?;
/// assert_eq!(result.text, "foobar");
/// assert_eq!(result.kind, PPTokenKind::Identifier);
///
/// // Number concatenation: 1 ## 2 → 12
/// let result = paste_tokens(&one_tok, &two_tok)?;
/// assert_eq!(result.text, "12");
/// assert_eq!(result.kind, PPTokenKind::Number);
///
/// // Mixed: VAR ## 1 → VAR1
/// let result = paste_tokens(&var_tok, &one_tok)?;
/// assert_eq!(result.text, "VAR1");
/// assert_eq!(result.kind, PPTokenKind::Identifier);
/// ```
pub fn paste_tokens(left: &PPToken, right: &PPToken) -> Result<PPToken, PasteError> {
    // -------------------------------------------------------------------
    // Placemarker handling (C11 §6.10.3.3¶2 footnote)
    // -------------------------------------------------------------------

    // Both placemarkers → return a new placemarker.
    if left.kind == PPTokenKind::PlacemarkerToken && right.kind == PPTokenKind::PlacemarkerToken {
        return Ok(PPToken {
            kind: PPTokenKind::PlacemarkerToken,
            text: String::new(),
            span: left.span.merge(right.span),
            from_macro: true,
            painted: false,
        });
    }

    // Left is placemarker → return right (marked as from macro expansion).
    if left.kind == PPTokenKind::PlacemarkerToken {
        return Ok(PPToken {
            kind: right.kind.clone(),
            text: right.text.clone(),
            span: left.span.merge(right.span),
            from_macro: true,
            painted: false,
        });
    }

    // Right is placemarker → return left (marked as from macro expansion).
    if right.kind == PPTokenKind::PlacemarkerToken {
        return Ok(PPToken {
            kind: left.kind.clone(),
            text: left.text.clone(),
            span: left.span.merge(right.span),
            from_macro: true,
            painted: false,
        });
    }

    // -------------------------------------------------------------------
    // Normal concatenation: join texts and re-classify the result
    // -------------------------------------------------------------------

    let mut result_text = String::with_capacity(left.text.len() + right.text.len());
    result_text.push_str(&left.text);
    result_text.push_str(&right.text);

    let kind = classify_concatenated_token(&result_text)
        .ok_or_else(|| PasteError::InvalidToken(result_text.clone()))?;

    Ok(PPToken {
        kind,
        text: result_text,
        span: left.span.merge(right.span),
        from_macro: true,
        painted: false,
    })
}

// ===========================================================================
// process_concatenation — scan replacement list for ## operators
// ===========================================================================

/// Process all `##` concatenation operators in a macro replacement token list.
///
/// Scans through `tokens` for `##` punctuators and pastes the adjacent
/// non-whitespace tokens together.  Handles chained operators such as
/// `A ## B ## C` by processing left-to-right: first `A ## B` produces `AB`,
/// then `AB ## C` produces `ABC`.
///
/// # Whitespace Handling
///
/// Whitespace tokens are **not significant** around `##` (C11 §6.10.3.3).
/// When `##` operators are present, the input list is filtered to significant
/// (non-whitespace) tokens for processing.  If **no** `##` operators exist,
/// the original token list is returned unmodified (whitespace preserved).
///
/// # Diagnostics
///
/// - `##` at the **start** or **end** of a replacement list is undefined
///   behaviour per C11 §6.10.3.3¶1 and is diagnosed as an error.
/// - An **invalid** concatenation result (e.g., `@` `##` `!` producing `@!`)
///   is diagnosed as a warning; the original tokens are preserved verbatim.
///
/// Returns a tuple of `(processed_tokens, diagnostics)`.
pub fn process_concatenation(tokens: &[PPToken]) -> (Vec<PPToken>, Vec<Diagnostic>) {
    let mut diagnostics: Vec<Diagnostic> = Vec::new();

    // Collect significant (non-whitespace) tokens for concatenation analysis.
    let sig: Vec<PPToken> = tokens
        .iter()
        .filter(|t| !matches!(t.kind, PPTokenKind::Whitespace | PPTokenKind::Newline))
        .cloned()
        .collect();

    // No significant tokens → nothing to process.
    if sig.is_empty() {
        return (tokens.to_vec(), diagnostics);
    }

    // Early exit when no ## operator is present — return original tokens
    // (with whitespace) unmodified for efficiency.
    let has_concat = sig.iter().any(is_hashhash);
    if !has_concat {
        return (tokens.to_vec(), diagnostics);
    }

    // -------------------------------------------------------------------
    // Boundary checks — ## at start or end is undefined behaviour
    // -------------------------------------------------------------------

    if is_hashhash(&sig[0]) {
        diagnostics.push(Diagnostic::error(
            sig[0].span,
            "'##' cannot appear at the start of a macro replacement list",
        ));
    }

    if let Some(last) = sig.last() {
        if sig.len() > 1 && is_hashhash(last) {
            diagnostics.push(Diagnostic::error(
                last.span,
                "'##' cannot appear at the end of a macro replacement list",
            ));
        }
    }

    // -------------------------------------------------------------------
    // Left-to-right concatenation pass
    // -------------------------------------------------------------------

    let mut result: Vec<PPToken> = Vec::with_capacity(sig.len());
    let mut i: usize = 0;

    while i < sig.len() {
        if is_hashhash(&sig[i]) {
            // ## operator found — need left and right operands.

            if result.is_empty() {
                // ## at start — already diagnosed above.  Skip the operator
                // so the right operand is processed normally.
                i += 1;
                continue;
            }

            if i + 1 >= sig.len() {
                // ## at end — already diagnosed above.  Skip the operator.
                i += 1;
                continue;
            }

            let left = result.pop().unwrap();
            let right = &sig[i + 1];

            match paste_tokens(&left, right) {
                Ok(pasted) => {
                    // Push the pasted result — it may be consumed by a
                    // subsequent ## in a chained expression.
                    result.push(pasted);
                }
                Err(PasteError::InvalidToken(ref bad_text)) => {
                    // Undefined behaviour — warn and preserve originals.
                    diagnostics.push(Diagnostic::warning(
                        left.span.merge(right.span),
                        format!(
                            "pasting \"{}\" and \"{}\" does not give a valid \
                             preprocessing token: \"{}\"",
                            left.text, right.text, bad_text
                        ),
                    ));
                    result.push(left);
                    result.push(right.clone());
                }
            }

            i += 2; // Skip the ## and the right operand.
        } else {
            result.push(sig[i].clone());
            i += 1;
        }
    }

    (result, diagnostics)
}

// ===========================================================================
// stringify_tokens — # operator
// ===========================================================================

/// Stringify a macro argument's token sequence via the `#` operator
/// (C11 §6.10.3.2).
///
/// Converts `tokens` into a single string literal token following the C11
/// stringification rules:
///
/// 1. Leading and trailing whitespace tokens are removed.
/// 2. Each internal sequence of one or more whitespace tokens is collapsed
///    to a single space character.
/// 3. `\` and `"` characters within string or character literal tokens in the
///    argument are escaped with an additional `\`.
/// 4. The entire result is surrounded by `"` delimiters.
///
/// The returned token has:
/// - `kind`: [`PPTokenKind::StringLiteral`]
/// - `span`: [`Span::dummy()`] (compiler-generated, no real source location)
/// - `from_macro`: `true`
///
/// # Examples
///
/// ```ignore
/// // #define STR(x) #x
/// // STR(hello world)  →  "hello world"
/// // STR(a "b" c)      →  "a \"b\" c"
/// // STR(a\nb)         →  "a\\nb"
/// ```
pub fn stringify_tokens(tokens: &[PPToken]) -> PPToken {
    let body = normalize_whitespace_for_stringify(tokens);

    // Wrap in double-quote delimiters to form a string literal.
    let mut literal = String::with_capacity(body.len() + 2);
    literal.push('"');
    literal.push_str(&body);
    literal.push('"');

    PPToken {
        kind: PPTokenKind::StringLiteral,
        text: literal,
        span: Span::dummy(),
        from_macro: true,
        painted: false,
    }
}

// ===========================================================================
// Internal helper functions
// ===========================================================================

// ---------------------------------------------------------------------------
// Token classification after concatenation
// ---------------------------------------------------------------------------

/// Classify the concatenated text as a preprocessing token kind.
///
/// Returns `None` if the text does not form a single valid preprocessing
/// token.  Used by [`paste_tokens`] to validate the concatenation result.
///
/// Classification order: identifier → pp-number → punctuator → string
/// literal → character literal.
fn classify_concatenated_token(text: &str) -> Option<PPTokenKind> {
    if text.is_empty() {
        return None;
    }

    let bytes = text.as_bytes();

    // --- Identifier: [a-zA-Z_][a-zA-Z0-9_]* ---
    if is_identifier_start(bytes[0]) {
        if bytes.iter().all(|&b| is_identifier_continue(b)) {
            return Some(PPTokenKind::Identifier);
        }
        // Starts like an identifier but contains invalid continuation
        // characters — not a valid single token.
        return None;
    }

    // --- pp-number: digit ... | . digit ... (C11 §6.4.8) ---
    if bytes[0].is_ascii_digit()
        || (bytes[0] == b'.' && bytes.len() > 1 && bytes[1].is_ascii_digit())
    {
        if is_valid_pp_number(text) {
            return Some(PPTokenKind::Number);
        }
        return None;
    }

    // --- Punctuator (all C11 punctuators + digraphs) ---
    if is_valid_punctuator(text) {
        return Some(PPTokenKind::Punctuator);
    }

    // --- String literal: "..." with optional prefix ---
    if is_valid_string_literal(text) {
        return Some(PPTokenKind::StringLiteral);
    }

    // --- Character literal: '...' with optional prefix ---
    if is_valid_char_literal(text) {
        return Some(PPTokenKind::CharLiteral);
    }

    // No valid classification found.
    None
}

/// Check whether `text` forms a valid preprocessing token.
///
/// Returns `true` for identifiers, pp-numbers, punctuators, string literals,
/// and character literals.  This is the top-level validity predicate used
/// internally by the concatenation logic and exposed for testing.
#[allow(dead_code)]
fn is_valid_preprocessing_token(text: &str) -> bool {
    classify_concatenated_token(text).is_some()
}

// ---------------------------------------------------------------------------
// Identifier helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `b` can start a C identifier (`[a-zA-Z_]`).
#[inline]
fn is_identifier_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Returns `true` if `b` can continue a C identifier (`[a-zA-Z0-9_]`).
#[inline]
fn is_identifier_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// pp-number validation (C11 §6.4.8)
// ---------------------------------------------------------------------------

/// Validate that `text` matches the C11 pp-number grammar.
///
/// A pp-number is:
/// ```text
/// pp-number:
///     digit
///     . digit
///     pp-number digit
///     pp-number identifier-nondigit
///     pp-number e sign
///     pp-number E sign
///     pp-number p sign
///     pp-number P sign
///     pp-number .
/// ```
///
/// In practice this means: starts with a digit or `.digit`, then any
/// combination of alphanumerics, `_`, `.`, and `eEpP` immediately
/// followed by `+` or `-`.
fn is_valid_pp_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let mut i: usize = 0;

    // Must start with digit, or `.` followed by digit.
    if bytes[i] == b'.' {
        if i + 1 >= bytes.len() || !bytes[i + 1].is_ascii_digit() {
            return false;
        }
        i += 2;
    } else if bytes[i].is_ascii_digit() {
        i += 1;
    } else {
        return false;
    }

    // Continuation characters.
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_alphanumeric()
            || b == b'_'
            || b == b'.'
            || ((b == b'+' || b == b'-')
                && i > 0
                && matches!(bytes[i - 1], b'e' | b'E' | b'p' | b'P'))
        {
            i += 1;
        } else {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Punctuator validation
// ---------------------------------------------------------------------------

/// Check whether `text` is a valid C11 punctuator (including digraphs).
fn is_valid_punctuator(text: &str) -> bool {
    matches!(
        text,
        // --- 4-character (digraph alternative) ---
        "%:%:" |
        // --- 3-character ---
        "<<=" | ">>=" | "..." |
        // --- 2-character ---
        "++" | "--" | "->" | "<<" | ">>" | "<=" | ">=" | "==" | "!=" |
        "&&" | "||" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" |
        "^=" | "##" | "<:" | ":>" | "<%" | "%>" | "%:" |
        // --- 1-character ---
        "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "~" | "!" |
        "<" | ">" | "=" | "(" | ")" | "[" | "]" | "{" | "}" |
        ";" | ":" | "," | "." | "?" | "#"
    )
}

// ---------------------------------------------------------------------------
// String / character literal validation (for concatenation results)
// ---------------------------------------------------------------------------

/// Check whether `text` forms a valid (properly terminated) string literal.
///
/// Validates an optional prefix (`L`, `u`, `U`, `u8`) followed by a
/// properly terminated `"..."` sequence with escape-sequence awareness.
fn is_valid_string_literal(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let mut i: usize = 0;

    // Optional prefix: L, U, u, u8.
    match bytes.get(i).copied() {
        Some(b'L' | b'U') => i += 1,
        Some(b'u') => {
            i += 1;
            if i < bytes.len() && bytes[i] == b'8' {
                i += 1;
            }
        }
        _ => {}
    }

    // Opening `"`.
    if i >= bytes.len() || bytes[i] != b'"' {
        return false;
    }
    i += 1;

    // Scan to closing `"`, respecting `\` escape sequences.
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2; // Skip escaped character.
        } else if bytes[i] == b'"' {
            // Closing quote — must be the last character.
            return i + 1 == bytes.len();
        } else {
            i += 1;
        }
    }

    false // Unterminated.
}

/// Check whether `text` forms a valid (properly terminated) character literal.
///
/// Validates an optional prefix (`L`, `u`, `U`) followed by a properly
/// terminated `'...'` sequence with escape-sequence awareness.
fn is_valid_char_literal(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let mut i: usize = 0;

    // Optional prefix: L, U, u.
    if matches!(bytes.get(i), Some(b'L' | b'U' | b'u')) {
        i += 1;
    }

    // Opening `'`.
    if i >= bytes.len() || bytes[i] != b'\'' {
        return false;
    }
    i += 1;

    // Scan to closing `'`.
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
        } else if bytes[i] == b'\'' {
            return i + 1 == bytes.len();
        } else {
            i += 1;
        }
    }

    false // Unterminated.
}

// ---------------------------------------------------------------------------
// Stringification helpers
// ---------------------------------------------------------------------------

/// Escape `\` and `"` characters for inclusion inside a string literal.
///
/// Per C11 §6.10.3.2¶2: each `\` character and each `"` character within the
/// argument's string or character literal content is preceded by a `\` in the
/// stringified result.
fn escape_for_string_literal(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            _ => result.push(ch),
        }
    }
    result
}

/// Normalise whitespace and build the stringified body from argument tokens.
///
/// Implements the whitespace normalisation specified in C11 §6.10.3.2:
///
/// 1. **Leading** whitespace tokens are removed (not emitted).
/// 2. Each **internal** sequence of one or more whitespace tokens is replaced
///    with a single space character.
/// 3. **Trailing** whitespace is stripped from the result.
/// 4. The text of [`PPTokenKind::StringLiteral`] and [`PPTokenKind::CharLiteral`]
///    tokens is escaped via [`escape_for_string_literal`] to preserve `\` and
///    `"` in the output.
fn normalize_whitespace_for_stringify(tokens: &[PPToken]) -> String {
    let mut result = String::new();
    // `prev_was_space` starts `true` so that leading whitespace is skipped.
    let mut prev_was_space = true;

    for token in tokens {
        match token.kind {
            PPTokenKind::Whitespace | PPTokenKind::Newline => {
                // Collapse consecutive whitespace into a single space, but
                // only once non-whitespace content has been accumulated
                // (this prevents leading spaces).
                if !prev_was_space && !result.is_empty() {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            PPTokenKind::StringLiteral | PPTokenKind::CharLiteral => {
                // String and character literal text includes surrounding
                // quotes and internal escape sequences.  Per C11 §6.10.3.2,
                // internal `\` and `"` must be escaped when stringifying.
                result.push_str(&escape_for_string_literal(&token.text));
                prev_was_space = false;
            }
            _ => {
                // All other token kinds: emit text verbatim.
                result.push_str(&token.text);
                prev_was_space = false;
            }
        }
    }

    // Trim trailing space (from a whitespace token at the end of the argument).
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

// ---------------------------------------------------------------------------
// Utility predicates
// ---------------------------------------------------------------------------

/// Returns `true` if `token` is the `##` concatenation punctuator.
#[inline]
fn is_hashhash(token: &PPToken) -> bool {
    token.kind == PPTokenKind::Punctuator && token.text == "##"
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple identifier token.
    fn ident(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Identifier, text, Span::dummy())
    }

    // Helper to create a number token.
    fn number(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Number, text, Span::dummy())
    }

    // Helper to create a punctuator token.
    fn punct(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Punctuator, text, Span::dummy())
    }

    // Helper to create a whitespace token.
    fn ws() -> PPToken {
        PPToken::new(PPTokenKind::Whitespace, " ", Span::dummy())
    }

    // Helper to create a placemarker token.
    fn placemarker() -> PPToken {
        PPToken::placemarker(Span::dummy())
    }

    // Helper for ## punctuator.
    fn hashhash() -> PPToken {
        punct("##")
    }

    // Helper to create a string literal token.
    fn strlit(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::StringLiteral, text, Span::dummy())
    }

    // Helper to create a char literal token.
    fn chrlit(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::CharLiteral, text, Span::dummy())
    }

    // Helper to create a newline token.
    fn newline() -> PPToken {
        PPToken::new(PPTokenKind::Newline, "\n", Span::dummy())
    }

    // -----------------------------------------------------------------------
    // paste_tokens tests
    // -----------------------------------------------------------------------

    #[test]
    fn paste_two_identifiers() {
        let result = paste_tokens(&ident("foo"), &ident("bar")).unwrap();
        assert_eq!(result.text, "foobar");
        assert_eq!(result.kind, PPTokenKind::Identifier);
        assert!(result.from_macro);
    }

    #[test]
    fn paste_two_numbers() {
        let result = paste_tokens(&number("1"), &number("2")).unwrap();
        assert_eq!(result.text, "12");
        assert_eq!(result.kind, PPTokenKind::Number);
    }

    #[test]
    fn paste_identifier_and_number() {
        let result = paste_tokens(&ident("VAR"), &number("1")).unwrap();
        assert_eq!(result.text, "VAR1");
        assert_eq!(result.kind, PPTokenKind::Identifier);
    }

    #[test]
    fn paste_number_and_identifier_gives_ppnumber() {
        // "1e" is a valid pp-number (starts with digit, followed by letter)
        let result = paste_tokens(&number("1"), &ident("e")).unwrap();
        assert_eq!(result.text, "1e");
        assert_eq!(result.kind, PPTokenKind::Number);
    }

    #[test]
    fn paste_punctuators() {
        // < ## = → <=
        let result = paste_tokens(&punct("<"), &punct("=")).unwrap();
        assert_eq!(result.text, "<=");
        assert_eq!(result.kind, PPTokenKind::Punctuator);
    }

    #[test]
    fn paste_into_shift_assign() {
        // << ## = → <<=
        let result = paste_tokens(&punct("<<"), &punct("=")).unwrap();
        assert_eq!(result.text, "<<=");
        assert_eq!(result.kind, PPTokenKind::Punctuator);
    }

    #[test]
    fn paste_invalid_token() {
        // " ## + → "+ which is not a valid preprocessing token
        let result = paste_tokens(&punct("\""), &punct("+"));
        assert!(result.is_err());
        if let Err(PasteError::InvalidToken(text)) = result {
            assert_eq!(text, "\"+");
        }
    }

    #[test]
    fn paste_left_placemarker() {
        let result = paste_tokens(&placemarker(), &ident("foo")).unwrap();
        assert_eq!(result.text, "foo");
        assert_eq!(result.kind, PPTokenKind::Identifier);
        assert!(result.from_macro);
    }

    #[test]
    fn paste_right_placemarker() {
        let result = paste_tokens(&ident("bar"), &placemarker()).unwrap();
        assert_eq!(result.text, "bar");
        assert_eq!(result.kind, PPTokenKind::Identifier);
        assert!(result.from_macro);
    }

    #[test]
    fn paste_both_placemarkers() {
        let result = paste_tokens(&placemarker(), &placemarker()).unwrap();
        assert_eq!(result.kind, PPTokenKind::PlacemarkerToken);
        assert!(result.text.is_empty());
        assert!(result.from_macro);
    }

    // -----------------------------------------------------------------------
    // process_concatenation tests
    // -----------------------------------------------------------------------

    #[test]
    fn concat_simple() {
        // [foo, ##, bar] → [foobar]
        let tokens = vec![ident("foo"), hashhash(), ident("bar")];
        let (result, diags) = process_concatenation(&tokens);
        assert!(diags.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "foobar");
    }

    #[test]
    fn concat_with_whitespace() {
        // [foo, WS, ##, WS, bar] → [foobar]
        let tokens = vec![ident("foo"), ws(), hashhash(), ws(), ident("bar")];
        let (result, diags) = process_concatenation(&tokens);
        assert!(diags.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "foobar");
    }

    #[test]
    fn concat_chained() {
        // [A, ##, B, ##, C] → [ABC]
        let tokens = vec![ident("A"), hashhash(), ident("B"), hashhash(), ident("C")];
        let (result, diags) = process_concatenation(&tokens);
        assert!(diags.is_empty());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "ABC");
    }

    #[test]
    fn concat_no_hashhash_preserves_whitespace() {
        // [A, WS, B] without ## → returns original tokens unchanged
        let tokens = vec![ident("A"), ws(), ident("B")];
        let (result, diags) = process_concatenation(&tokens);
        assert!(diags.is_empty());
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "A");
        assert_eq!(result[1].kind, PPTokenKind::Whitespace);
        assert_eq!(result[2].text, "B");
    }

    #[test]
    fn concat_at_start_diagnosed() {
        // [##, foo] → error at start
        let tokens = vec![hashhash(), ident("foo")];
        let (_, diags) = process_concatenation(&tokens);
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("start"));
    }

    #[test]
    fn concat_at_end_diagnosed() {
        // [foo, ##] → error at end
        let tokens = vec![ident("foo"), hashhash()];
        let (_, diags) = process_concatenation(&tokens);
        assert!(!diags.is_empty());
        assert!(diags[0].message.contains("end"));
    }

    #[test]
    fn concat_invalid_result_warning() {
        // Concatenating two punctuators that don't form a valid combined punctuator
        // e.g., `@` ## `!` → `@!` (invalid)
        let at_tok = PPToken::new(PPTokenKind::Punctuator, "@", Span::dummy());
        let bang_tok = punct("!");
        let tokens = vec![at_tok, hashhash(), bang_tok];
        let (result, diags) = process_concatenation(&tokens);
        // Should have a warning about invalid concatenation.
        assert!(diags
            .iter()
            .any(|d| d.message.contains("does not give a valid")));
        // Original tokens should be preserved.
        assert!(result.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // stringify_tokens tests
    // -----------------------------------------------------------------------

    #[test]
    fn stringify_simple() {
        // [hello, WS, world] → "hello world"
        let tokens = vec![ident("hello"), ws(), ident("world")];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.kind, PPTokenKind::StringLiteral);
        assert_eq!(result.text, "\"hello world\"");
        assert!(result.from_macro);
    }

    #[test]
    fn stringify_empty() {
        let result = stringify_tokens(&[]);
        assert_eq!(result.text, "\"\"");
        assert_eq!(result.kind, PPTokenKind::StringLiteral);
    }

    #[test]
    fn stringify_escapes_backslash_in_string_literal() {
        // Token containing a string literal with backslash: "a\nb"
        let tokens = vec![strlit("\"a\\nb\"")];
        let result = stringify_tokens(&tokens);
        // The \ and " in the string literal text are escaped.
        assert_eq!(result.text, "\"\\\"a\\\\nb\\\"\"");
    }

    #[test]
    fn stringify_escapes_quote_in_string_literal() {
        // Argument tokens: a "b" c → "a \"b\" c"
        let tokens = vec![ident("a"), ws(), strlit("\"b\""), ws(), ident("c")];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"a \\\"b\\\" c\"");
    }

    #[test]
    fn stringify_collapses_whitespace() {
        // [a, WS, WS, WS, b] → "a b" (multiple spaces → one)
        let tokens = vec![ident("a"), ws(), ws(), ws(), ident("b")];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"a b\"");
    }

    #[test]
    fn stringify_strips_leading_whitespace() {
        let tokens = vec![ws(), ws(), ident("x")];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"x\"");
    }

    #[test]
    fn stringify_strips_trailing_whitespace() {
        let tokens = vec![ident("x"), ws(), ws()];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"x\"");
    }

    #[test]
    fn stringify_newline_treated_as_whitespace() {
        let tokens = vec![ident("a"), newline(), ident("b")];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"a b\"");
    }

    #[test]
    fn stringify_only_whitespace_gives_empty_string() {
        let tokens = vec![ws(), ws(), newline()];
        let result = stringify_tokens(&tokens);
        assert_eq!(result.text, "\"\"");
    }

    #[test]
    fn stringify_char_literal_escaped() {
        // Char literal with backslash: '\n'
        let tokens = vec![chrlit("'\\n'")];
        let result = stringify_tokens(&tokens);
        // The ' characters pass through unchanged; the \ is escaped.
        assert_eq!(result.text, "\"'\\\\n'\"");
    }

    // -----------------------------------------------------------------------
    // Internal helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn valid_identifiers() {
        assert!(is_valid_preprocessing_token("foo"));
        assert!(is_valid_preprocessing_token("_bar"));
        assert!(is_valid_preprocessing_token("__VA_ARGS__"));
        assert!(is_valid_preprocessing_token("x1"));
    }

    #[test]
    fn valid_numbers() {
        assert!(is_valid_preprocessing_token("42"));
        assert!(is_valid_preprocessing_token("0x1F"));
        assert!(is_valid_preprocessing_token("3.14"));
        assert!(is_valid_preprocessing_token(".5"));
        assert!(is_valid_preprocessing_token("1e10"));
        assert!(is_valid_preprocessing_token("1e+10"));
        assert!(is_valid_preprocessing_token("0x1p-3"));
    }

    #[test]
    fn valid_punctuators() {
        assert!(is_valid_preprocessing_token("+"));
        assert!(is_valid_preprocessing_token("->"));
        assert!(is_valid_preprocessing_token("<<="));
        assert!(is_valid_preprocessing_token("..."));
        assert!(is_valid_preprocessing_token("##"));
    }

    #[test]
    fn invalid_tokens() {
        assert!(!is_valid_preprocessing_token(""));
        assert!(!is_valid_preprocessing_token("\"unterminated"));
        assert!(!is_valid_preprocessing_token("'a"));
        assert!(!is_valid_preprocessing_token("@!"));
    }

    #[test]
    fn escape_for_string_literal_basic() {
        assert_eq!(escape_for_string_literal("hello"), "hello");
        assert_eq!(escape_for_string_literal("a\\b"), "a\\\\b");
        assert_eq!(escape_for_string_literal("a\"b"), "a\\\"b");
        assert_eq!(escape_for_string_literal("\"a\\b\""), "\\\"a\\\\b\\\"");
    }

    #[test]
    fn pp_number_validation() {
        assert!(is_valid_pp_number("42"));
        assert!(is_valid_pp_number("0xFF"));
        assert!(is_valid_pp_number("3.14"));
        assert!(is_valid_pp_number(".5"));
        assert!(is_valid_pp_number("1e+10"));
        assert!(is_valid_pp_number("0x1p-3"));
        assert!(is_valid_pp_number("0b1010"));
        assert!(!is_valid_pp_number(""));
        assert!(!is_valid_pp_number("abc"));
        assert!(!is_valid_pp_number("."));
    }
}
