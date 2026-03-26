//! # BCC Preprocessor — Phases 1–2 of the C11 Compilation Pipeline
//!
//! Implements the C preprocessor with:
//! - Phase 1: Trigraph replacement (`??=` → `#`, etc.) and backslash-newline line splicing
//! - Phase 2: Directive processing (`#include`, `#define`, `#if`, etc.) and macro expansion
//!
//! ## Key Features
//! - Paint-marker recursion protection for self-referential macros (`#define A A` terminates correctly)
//! - 512-depth recursion limit for macro expansion (prevents stack overflow on kernel macros)
//! - PUA encoding for non-UTF-8 source file byte round-tripping (U+E080–U+E0FF)
//! - Full C11 preprocessor compliance with GCC extensions
//! - `__VA_ARGS__` and `__VA_OPT__` variadic macro support
//!
//! ## Usage
//! ```ignore
//! use crate::frontend::preprocessor::Preprocessor;
//! let mut pp = Preprocessor::new(source_map, diagnostics, target, interner);
//! pp.add_include_path("/usr/include");
//! pp.add_define("DEBUG", "1");
//! let tokens = pp.preprocess_file("hello.c");
//! ```
//!
//! ## Dependencies
//! - `crate::common::encoding` — PUA/UTF-8 encoding for source file reading
//! - `crate::common::source_map` — File tracking and `#line` directive handling
//! - `crate::common::diagnostics` — Error/warning reporting (unterminated `#if`, circular includes, etc.)
//! - `crate::common::string_interner` — Macro name interning
//! - `crate::common::target` — Architecture-specific predefined macros
//! - `crate::common::fx_hash` — FxHashMap for macro definition storage

// ---------------------------------------------------------------------------
// Submodule declarations — all 7 preprocessor sub-components
// ---------------------------------------------------------------------------

pub mod directives;
pub mod expression;
pub mod include_handler;
pub mod macro_expander;
pub mod paint_marker;
pub mod predefined;
pub mod token_paster;

// ---------------------------------------------------------------------------
// Imports from crate::common — infrastructure layer
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::encoding::read_source_file;
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::string_interner::Interner;
use crate::common::target::Target;

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// PPTokenKind — preprocessing token classification
// ---------------------------------------------------------------------------

/// Classification of a preprocessing token.
///
/// Preprocessing tokens are coarser than lexer tokens — they do not
/// distinguish keywords from identifiers, and numeric literals are not
/// yet classified by type (int, float, hex, etc.).  The lexer performs
/// that finer classification after preprocessing is complete.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PPTokenKind {
    /// An identifier (or keyword — not distinguished at this stage).
    Identifier,
    /// A preprocessing number: a token that starts with a digit or `.` followed
    /// by a digit.  Not yet classified as integer or floating-point.
    Number,
    /// A string literal, including the surrounding `"` quotes and any
    /// prefix (`u8`, `u`, `U`, `L`).
    StringLiteral,
    /// A character literal, including the surrounding `'` quotes and any
    /// prefix (`L`, `u`, `U`).
    CharLiteral,
    /// An operator or punctuator: `+`, `-`, `->`, `(`, `)`, `{`, `}`, `;`, etc.
    Punctuator,
    /// Whitespace (spaces, tabs) — NOT newlines.
    Whitespace,
    /// A newline character (`\n`).  Significant for directive recognition:
    /// a `#` that is the first non-whitespace token on a line introduces
    /// a preprocessing directive.
    Newline,
    /// A header name in an `#include` directive: either `<path>` or `"path"`.
    HeaderName,
    /// A placemarker token produced by `##` concatenation when one operand is
    /// an empty macro argument.  Concatenating a placemarker with a real token
    /// yields the real token.
    PlacemarkerToken,
    /// End-of-file sentinel.
    EndOfFile,
}

// ---------------------------------------------------------------------------
// PPToken — a single preprocessing token
// ---------------------------------------------------------------------------

/// A preprocessing token — the output of the preprocessor, input to the lexer.
///
/// Carries the token text, source location span, and macro-expansion metadata
/// (whether it was produced by macro expansion and whether it has been painted
/// by the paint-marker system to suppress re-expansion).
#[derive(Debug, Clone)]
pub struct PPToken {
    /// The kind (classification) of this token.
    pub kind: PPTokenKind,
    /// The textual representation of this token exactly as it appears in the
    /// (possibly macro-expanded) source.
    pub text: String,
    /// Source location span for diagnostic reporting.
    pub span: Span,
    /// `true` if this token was produced by macro expansion rather than
    /// appearing directly in the source file.
    pub from_macro: bool,
    /// Paint-marker state.  When `true`, this token must NOT be re-expanded
    /// even if it matches a defined macro name.  Set by the paint-marker
    /// system when a macro name is encountered during its own expansion
    /// (C11 §6.10.3.4 rescanning suppression).
    pub painted: bool,
}

impl PPToken {
    /// Create a new preprocessing token with default expansion metadata.
    ///
    /// `from_macro` and `painted` are both set to `false` — the caller
    /// should override these when constructing tokens during macro expansion.
    #[inline]
    pub fn new(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self {
        PPToken {
            kind,
            text: text.into(),
            span,
            from_macro: false,
            painted: false,
        }
    }

    /// Create a new preprocessing token produced by macro expansion.
    #[inline]
    pub fn from_expansion(kind: PPTokenKind, text: impl Into<String>, span: Span) -> Self {
        PPToken {
            kind,
            text: text.into(),
            span,
            from_macro: true,
            painted: false,
        }
    }

    /// Create an end-of-file sentinel token.
    #[inline]
    pub fn eof(span: Span) -> Self {
        PPToken {
            kind: PPTokenKind::EndOfFile,
            text: String::new(),
            span,
            from_macro: false,
            painted: false,
        }
    }

    /// Create a placemarker token (used by `##` concatenation with empty args).
    #[inline]
    pub fn placemarker(span: Span) -> Self {
        PPToken {
            kind: PPTokenKind::PlacemarkerToken,
            text: String::new(),
            span,
            from_macro: true,
            painted: false,
        }
    }

    /// Returns `true` if this token is whitespace or newline.
    #[inline]
    pub fn is_whitespace(&self) -> bool {
        matches!(self.kind, PPTokenKind::Whitespace | PPTokenKind::Newline)
    }

    /// Returns `true` if this token is end-of-file.
    #[inline]
    pub fn is_eof(&self) -> bool {
        self.kind == PPTokenKind::EndOfFile
    }
}

// ---------------------------------------------------------------------------
// MacroKind — object-like vs function-like macros
// ---------------------------------------------------------------------------

/// The kind of a macro definition.
///
/// Object-like macros expand to their replacement list without argument
/// substitution.  Function-like macros accept a parenthesised argument list
/// and perform parameter substitution in the replacement.
#[derive(Debug, Clone)]
pub enum MacroKind {
    /// An object-like macro: `#define FOO 42` — expands to replacement
    /// tokens without any argument processing.
    ObjectLike,
    /// A function-like macro: `#define MAX(a, b) ((a) > (b) ? (a) : (b))`.
    FunctionLike {
        /// Parameter names in declaration order.
        params: Vec<String>,
        /// `true` if the macro is variadic (last parameter is `...`).
        /// Variadic arguments are accessible via `__VA_ARGS__`.
        variadic: bool,
    },
}

// ---------------------------------------------------------------------------
// MacroDef — a complete macro definition
// ---------------------------------------------------------------------------

/// A macro definition created by `#define` or predefined by the compiler.
///
/// Stores the macro name, kind (object-like or function-like), replacement
/// token list, and metadata about whether it is a compiler-predefined macro
/// and where it was defined in the source.
#[derive(Debug, Clone)]
pub struct MacroDef {
    /// The macro name (e.g., `"FOO"`, `"__STDC__"`).
    pub name: String,
    /// Whether this is an object-like or function-like macro.
    pub kind: MacroKind,
    /// The replacement token list.  For an empty `#define FOO`, this is empty.
    pub replacement: Vec<PPToken>,
    /// `true` for compiler-predefined macros (`__FILE__`, `__LINE__`,
    /// `__STDC__`, etc.) — these cannot be `#undef`'d.
    pub is_predefined: bool,
    /// Source location where the macro was defined.  [`Span::dummy()`] for
    /// predefined macros.
    pub definition_span: Span,
}

// ---------------------------------------------------------------------------
// ConditionalState — #if/#ifdef/#else/#endif nesting tracker
// ---------------------------------------------------------------------------

/// State for a single level of nested `#if`/`#ifdef`/`#elif`/`#else`/`#endif`
/// conditional compilation.
///
/// The preprocessor maintains a stack of these to handle arbitrary nesting of
/// conditional blocks.
#[derive(Debug, Clone)]
pub struct ConditionalState {
    /// Whether the current branch is active (tokens should be included in output).
    pub active: bool,
    /// Whether *any* branch in this `#if`/`#elif`/`#else` group has been active.
    /// Once a branch has been taken, subsequent `#elif` and `#else` branches
    /// are skipped even if their conditions evaluate to true.
    pub seen_active: bool,
    /// Whether `#else` has been seen for this group.  A second `#else` or
    /// `#elif` after `#else` is an error.
    pub seen_else: bool,
    /// Source span of the opening `#if`/`#ifdef`/`#ifndef` directive, used
    /// for diagnostic reporting of unterminated conditional blocks.
    pub opening_span: Span,
}

impl ConditionalState {
    /// Create a new conditional state for an `#if`/`#ifdef`/`#ifndef` directive.
    ///
    /// `active` indicates whether the condition evaluated to true (non-zero).
    /// `opening_span` is the source location of the directive for diagnostics.
    #[inline]
    pub fn new(active: bool, opening_span: Span) -> Self {
        ConditionalState {
            active,
            seen_active: active,
            seen_else: false,
            opening_span,
        }
    }
}

// ===========================================================================
// Phase 1 — Trigraph Replacement and Line Splicing
// ===========================================================================

/// Replace C trigraph sequences with their single-character equivalents.
///
/// Trigraphs are three-character sequences beginning with `??` that represent
/// characters not available in some legacy character sets.  C11 retains them
/// for backwards compatibility (§5.2.1.1), though they are rarely used in
/// modern code.
///
/// | Trigraph | Replacement |
/// |----------|-------------|
/// | `??=`    | `#`         |
/// | `??/`    | `\`         |
/// | `??'`    | `^`         |
/// | `??(`    | `[`         |
/// | `??)`    | `]`         |
/// | `??!`    | `|`         |
/// | `??<`    | `{`         |
/// | `??>`    | `}`         |
/// | `??-`    | `~`         |
pub fn phase1_trigraphs(input: &str) -> String {
    let bytes = input.as_bytes();
    let len = bytes.len();
    if len < 3 {
        return input.to_string();
    }

    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if i + 2 < len && bytes[i] == b'?' && bytes[i + 1] == b'?' {
            let replacement = match bytes[i + 2] {
                b'=' => Some('#'),
                b'/' => Some('\\'),
                b'\'' => Some('^'),
                b'(' => Some('['),
                b')' => Some(']'),
                b'!' => Some('|'),
                b'<' => Some('{'),
                b'>' => Some('}'),
                b'-' => Some('~'),
                _ => None,
            };
            if let Some(ch) = replacement {
                result.push(ch);
                i += 3;
                continue;
            }
        }
        // Multi-byte UTF-8 characters (including PUA code points) must be
        // copied as complete sequences. `bytes[i] as char` would treat each
        // byte individually, converting e.g. 0xEE to U+00EE instead of
        // recognising the full 3-byte PUA sequence EE 83 BF as U+E0FF.
        if bytes[i] < 0x80 {
            result.push(bytes[i] as char);
            i += 1;
        } else {
            // Determine UTF-8 sequence length from the lead byte.
            let seq_len = if bytes[i] >= 0xF0 {
                4
            } else if bytes[i] >= 0xE0 {
                3
            } else if bytes[i] >= 0xC0 {
                2
            } else {
                1 // bare continuation byte — push as-is
            };
            let end = std::cmp::min(i + seq_len, len);
            if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                result.push_str(s);
            } else {
                // Fallback for malformed sequences: push each byte as a char.
                for &b in &bytes[i..end] {
                    result.push(b as char);
                }
            }
            i = end;
        }
    }

    result
}

/// Splice lines that are continued with a backslash immediately before a
/// newline, producing a single logical line.
///
/// Handles both Unix (`\n`) and Windows (`\r\n`) line endings.  The
/// backslash and the following newline (and optional carriage return) are
/// removed, concatenating the physical lines into one logical line.
pub fn phase1_line_splice(input: &str) -> String {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\\' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
                continue;
            }
            if i + 2 < len && bytes[i + 1] == b'\r' && bytes[i + 2] == b'\n' {
                i += 3;
                continue;
            }
        }
        // Preserve multi-byte UTF-8 sequences (including PUA code points).
        if bytes[i] < 0x80 {
            result.push(bytes[i] as char);
            i += 1;
        } else {
            let seq_len = if bytes[i] >= 0xF0 {
                4
            } else if bytes[i] >= 0xE0 {
                3
            } else if bytes[i] >= 0xC0 {
                2
            } else {
                1
            };
            let end = std::cmp::min(i + seq_len, len);
            if let Ok(s) = std::str::from_utf8(&bytes[i..end]) {
                result.push_str(s);
            } else {
                for &b in &bytes[i..end] {
                    result.push(b as char);
                }
            }
            i = end;
        }
    }

    result
}

// ===========================================================================
// Preprocessing Tokenizer — splits input text into PPToken streams
// ===========================================================================

/// Tokenize a string of (Phase-1-processed) source text into preprocessing
/// tokens.
///
/// This tokeniser is intentionally simpler than the full lexer (Phase 3):
/// - It does NOT distinguish C keywords from identifiers.
/// - It does NOT parse numeric literal suffixes in detail.
/// - It does NOT handle adjacent string literal concatenation.
///
/// Its job is solely to split the input into the coarse token categories
/// that the preprocessor needs: identifiers, numbers, strings, character
/// literals, punctuators, whitespace, and newlines.
pub fn tokenize_preprocessing(input: &str, file_id: u32) -> Vec<PPToken> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 4);
    let mut pos: usize = 0;

    while pos < len {
        let start = pos;
        let b = bytes[pos];

        // --- Newline ---
        if b == b'\n' {
            tokens.push(PPToken::new(
                PPTokenKind::Newline,
                "\n",
                Span::new(file_id, start as u32, (start + 1) as u32),
            ));
            pos += 1;
            continue;
        }

        // --- Whitespace (space, tab, \r, form-feed — but NOT \n) ---
        if b == b' ' || b == b'\t' || b == b'\r' || b == 0x0C {
            let ws_start = pos;
            while pos < len {
                let c = bytes[pos];
                if c == b' ' || c == b'\t' || c == b'\r' || c == 0x0C {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[ws_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                text,
                Span::new(file_id, ws_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Line comment: // ... ---
        if b == b'/' && pos + 1 < len && bytes[pos + 1] == b'/' {
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            // C11 §5.1.1.2 — each comment is replaced by one space.
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                " ",
                Span::new(file_id, start as u32, pos as u32),
            ));
            continue;
        }

        // --- Block comment: /* ... */ ---
        if b == b'/' && pos + 1 < len && bytes[pos + 1] == b'*' {
            let cmt_start = pos;
            pos += 2;
            loop {
                if pos + 1 >= len {
                    pos = len;
                    break;
                }
                if bytes[pos] == b'*' && bytes[pos + 1] == b'/' {
                    pos += 2;
                    break;
                }
                pos += 1;
            }
            tokens.push(PPToken::new(
                PPTokenKind::Whitespace,
                " ",
                Span::new(file_id, cmt_start as u32, pos as u32),
            ));
            continue;
        }

        // --- String literal: "..." (with optional prefix L, u, U, u8) ---
        if b == b'"' || is_string_prefix(bytes, pos) {
            let (tok, new_pos) = lex_string_literal(input, pos, file_id);
            tokens.push(tok);
            pos = new_pos;
            continue;
        }

        // --- Character literal: '...' (with optional prefix L, u, U) ---
        if b == b'\'' || is_char_prefix(bytes, pos) {
            let (tok, new_pos) = lex_char_literal(input, pos, file_id);
            tokens.push(tok);
            pos = new_pos;
            continue;
        }

        // --- Identifier: [a-zA-Z_][a-zA-Z0-9_]* ---
        if b.is_ascii_alphabetic() || b == b'_' {
            let id_start = pos;
            pos += 1;
            while pos < len {
                let c = bytes[pos];
                if c.is_ascii_alphanumeric() || c == b'_' {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[id_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Identifier,
                text,
                Span::new(file_id, id_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Number: starts with digit, or `.` followed by digit ---
        if b.is_ascii_digit() || (b == b'.' && pos + 1 < len && bytes[pos + 1].is_ascii_digit()) {
            let num_start = pos;
            pos += 1;
            while pos < len {
                let c = bytes[pos];
                let is_number_continuation = c.is_ascii_alphanumeric()
                    || c == b'_'
                    || c == b'.'
                    || ((c == b'+' || c == b'-')
                        && pos > num_start + 1
                        && matches!(bytes[pos - 1], b'e' | b'E' | b'p' | b'P'));
                if is_number_continuation {
                    pos += 1;
                } else {
                    break;
                }
            }
            let text = &input[num_start..pos];
            tokens.push(PPToken::new(
                PPTokenKind::Number,
                text,
                Span::new(file_id, num_start as u32, pos as u32),
            ));
            continue;
        }

        // --- Multi-character punctuators (longest match) ---
        if let Some((punct_text, plen)) = match_punctuator(bytes, pos) {
            tokens.push(PPToken::new(
                PPTokenKind::Punctuator,
                punct_text,
                Span::new(file_id, start as u32, (start + plen) as u32),
            ));
            pos += plen;
            continue;
        }

        // --- Fallback: single character ---
        let ch = get_char_at(input, pos);
        let ch_len = ch.len_utf8();
        pos += ch_len;
        if ch_len > 1 {
            // Multi-byte UTF-8 or PUA — treat as identifier for pass-through.
            tokens.push(PPToken::new(
                PPTokenKind::Identifier,
                &input[start..pos],
                Span::new(file_id, start as u32, pos as u32),
            ));
        } else {
            tokens.push(PPToken::new(
                PPTokenKind::Punctuator,
                &input[start..pos],
                Span::new(file_id, start as u32, pos as u32),
            ));
        }
    }

    // Append EOF sentinel.
    tokens.push(PPToken::eof(Span::new(file_id, pos as u32, pos as u32)));
    tokens
}

// ---------------------------------------------------------------------------
// Tokenizer helper functions
// ---------------------------------------------------------------------------

/// Check if position starts a string literal prefix (`u8"`, `u"`, `U"`, `L"`).
fn is_string_prefix(bytes: &[u8], pos: usize) -> bool {
    let len = bytes.len();
    if pos >= len {
        return false;
    }
    match bytes[pos] {
        b'L' | b'U' => pos + 1 < len && bytes[pos + 1] == b'"',
        b'u' => {
            (pos + 1 < len && bytes[pos + 1] == b'"')
                || (pos + 2 < len && bytes[pos + 1] == b'8' && bytes[pos + 2] == b'"')
        }
        _ => false,
    }
}

/// Check if position starts a character literal prefix (`L'`, `u'`, `U'`).
fn is_char_prefix(bytes: &[u8], pos: usize) -> bool {
    let len = bytes.len();
    if pos >= len {
        return false;
    }
    matches!(bytes[pos], b'L' | b'U' | b'u') && pos + 1 < len && bytes[pos + 1] == b'\''
}

/// Lex a string literal starting at `pos`.
fn lex_string_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize) {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let start = pos;
    let mut i = pos;

    // Skip prefix.
    match bytes[i] {
        b'L' | b'U' => i += 1,
        b'u' => {
            i += 1;
            if i < len && bytes[i] == b'8' {
                i += 1;
            }
        }
        _ => {}
    }
    // Skip opening `"`.
    if i < len && bytes[i] == b'"' {
        i += 1;
    }
    // Scan until closing `"`.
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
        } else if bytes[i] == b'"' {
            i += 1;
            break;
        } else if bytes[i] == b'\n' {
            break;
        } else {
            i += 1;
        }
    }
    let text = &input[start..i];
    (
        PPToken::new(
            PPTokenKind::StringLiteral,
            text,
            Span::new(file_id, start as u32, i as u32),
        ),
        i,
    )
}

/// Lex a character literal starting at `pos`.
fn lex_char_literal(input: &str, pos: usize, file_id: u32) -> (PPToken, usize) {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let start = pos;
    let mut i = pos;

    // Skip prefix.
    if matches!(bytes[i], b'L' | b'U' | b'u') {
        i += 1;
    }
    if i < len && bytes[i] == b'\'' {
        i += 1;
    }
    while i < len {
        if bytes[i] == b'\\' && i + 1 < len {
            i += 2;
        } else if bytes[i] == b'\'' {
            i += 1;
            break;
        } else if bytes[i] == b'\n' {
            break;
        } else {
            i += 1;
        }
    }
    let text = &input[start..i];
    (
        PPToken::new(
            PPTokenKind::CharLiteral,
            text,
            Span::new(file_id, start as u32, i as u32),
        ),
        i,
    )
}

/// Match the longest punctuator at the given position.
fn match_punctuator(bytes: &[u8], pos: usize) -> Option<(&'static str, usize)> {
    let len = bytes.len();
    let remaining = len - pos;

    // 3-character punctuators.
    if remaining >= 3 {
        let t = &bytes[pos..pos + 3];
        let m = match t {
            b"<<=" => Some("<<="),
            b">>=" => Some(">>="),
            b"..." => Some("..."),
            _ => None,
        };
        if m.is_some() {
            return m.map(|s| (s, 3));
        }
    }

    // 2-character punctuators.
    if remaining >= 2 {
        let t = &bytes[pos..pos + 2];
        let m = match t {
            b"++" => Some("++"),
            b"--" => Some("--"),
            b"->" => Some("->"),
            b"<<" => Some("<<"),
            b">>" => Some(">>"),
            b"<=" => Some("<="),
            b">=" => Some(">="),
            b"==" => Some("=="),
            b"!=" => Some("!="),
            b"&&" => Some("&&"),
            b"||" => Some("||"),
            b"+=" => Some("+="),
            b"-=" => Some("-="),
            b"*=" => Some("*="),
            b"/=" => Some("/="),
            b"%=" => Some("%="),
            b"&=" => Some("&="),
            b"|=" => Some("|="),
            b"^=" => Some("^="),
            b"##" => Some("##"),
            _ => None,
        };
        if m.is_some() {
            return m.map(|s| (s, 2));
        }
    }

    // 1-character punctuators.
    if remaining >= 1 {
        let text: &'static str = match bytes[pos] {
            b'+' => "+",
            b'-' => "-",
            b'*' => "*",
            b'/' => "/",
            b'%' => "%",
            b'&' => "&",
            b'|' => "|",
            b'^' => "^",
            b'~' => "~",
            b'!' => "!",
            b'<' => "<",
            b'>' => ">",
            b'=' => "=",
            b'(' => "(",
            b')' => ")",
            b'[' => "[",
            b']' => "]",
            b'{' => "{",
            b'}' => "}",
            b';' => ";",
            b':' => ":",
            b',' => ",",
            b'.' => ".",
            b'?' => "?",
            b'#' => "#",
            _ => return None,
        };
        return Some((text, 1));
    }

    None
}

/// Get the Unicode character at a byte position in a string.
#[inline]
fn get_char_at(s: &str, pos: usize) -> char {
    s[pos..].chars().next().unwrap_or('\0')
}

// ===========================================================================
// Preprocessor — the main state machine
// ===========================================================================

/// The main preprocessor state machine.
///
/// Coordinates all seven submodules to transform raw C source text into
/// a fully macro-expanded preprocessing token stream.  The preprocessor
/// is the first stage of the compilation pipeline and is invoked by the
/// CLI driver (`src/main.rs`).
///
/// # Lifetime Parameter
///
/// The `'a` lifetime binds the preprocessor to its borrowed infrastructure:
/// source map, diagnostic engine, and string interner.  All three are owned
/// by the compilation driver and passed by mutable reference.
pub struct Preprocessor<'a> {
    /// Source file registry for tracking loaded files and line/column lookups.
    pub source_map: &'a mut SourceMap,
    /// Diagnostic reporting engine for errors, warnings, and notes.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Target architecture — determines predefined macros and type sizes.
    pub target: Target,
    /// String interner for deduplicating identifiers and macro names.
    pub interner: &'a mut Interner,
    /// Macro definitions: macro name → [`MacroDef`].
    pub macro_defs: FxHashMap<String, MacroDef>,
    /// User include search paths from `-I` flags (searched first for `"..."` includes).
    pub include_paths: Vec<PathBuf>,
    /// System include paths (searched for `<...>` includes).
    pub system_include_paths: Vec<PathBuf>,
    /// Command-line `-D` defines: `(name, value)` pairs.
    pub cli_defines: Vec<(String, String)>,
    /// Current include nesting depth for depth-limit enforcement.
    pub include_depth: usize,
    /// Maximum include nesting depth (default: 200, matching GCC).
    /// Exceeding this limit produces a clean diagnostic error rather than
    /// a stack overflow, protecting against deeply-chained or circular
    /// `#include` directives.
    pub max_include_depth: usize,
    /// Stack of file paths currently being processed, from outermost to
    /// innermost.  Used for circular `#include` detection — if a file
    /// appears in this stack when an `#include` directive tries to include
    /// it again, the preprocessor emits an error instead of infinitely
    /// recursing.
    include_stack: Vec<PathBuf>,
    /// Maximum recursion depth for macro expansion (default: 512).
    pub max_recursion_depth: usize,
    /// Stack of conditional compilation states for nested `#if`/`#endif` blocks.
    conditional_stack: Vec<ConditionalState>,
    /// Auto-incrementing counter for `__COUNTER__` macro (GCC extension).
    /// Each expansion of `__COUNTER__` returns the current value and increments it.
    pub counter_value: u64,
    /// Stack for `#pragma push_macro("name")` / `#pragma pop_macro("name")`.
    /// Maps macro name → stack of saved definitions (None = was undefined).
    pub macro_push_stack: FxHashMap<String, Vec<Option<MacroDef>>>,
    /// When `true`, `#pragma` directives are preserved as output tokens
    /// (for `-E` preprocess-only mode).  When `false` (default), pragmas
    /// are consumed silently and not emitted into the token stream.
    pub preserve_pragmas: bool,
}

impl<'a> Preprocessor<'a> {
    /// Create a new preprocessor with the given infrastructure references.
    ///
    /// Initialises the macro definition table with predefined macros for the
    /// target architecture (via [`predefined::register_predefined_macros`]),
    /// sets the recursion depth limit to 512, and prepares an empty include
    /// path list.
    pub fn new(
        source_map: &'a mut SourceMap,
        diagnostics: &'a mut DiagnosticEngine,
        target: Target,
        interner: &'a mut Interner,
    ) -> Self {
        let mut macro_defs = FxHashMap::default();

        // Register compiler-predefined macros (__STDC__, __linux__, arch-specific, etc.).
        predefined::register_predefined_macros(&mut macro_defs, &target);

        Preprocessor {
            source_map,
            diagnostics,
            target,
            interner,
            macro_defs,
            include_paths: Vec::new(),
            system_include_paths: Vec::new(),
            cli_defines: Vec::new(),
            include_depth: 0,
            max_include_depth: 200,
            include_stack: Vec::new(),
            max_recursion_depth: 512,
            conditional_stack: Vec::new(),
            counter_value: 0,
            macro_push_stack: FxHashMap::default(),
            preserve_pragmas: false,
        }
    }

    /// Add a user include search path (`-I` flag).
    pub fn add_include_path(&mut self, path: &str) {
        self.include_paths.push(PathBuf::from(path));
    }

    /// Add a system include search path.
    pub fn add_system_include_path(&mut self, path: &str) {
        self.system_include_paths.push(PathBuf::from(path));
    }

    /// Define a macro from a `-D` command-line flag.
    ///
    /// `name` is the macro name, `value` is the replacement text.
    /// An empty `value` defines the macro with an empty replacement list.
    pub fn add_define(&mut self, name: &str, value: &str) {
        // Store for later processing.
        self.cli_defines.push((name.to_string(), value.to_string()));

        // Also immediately register as an object-like macro.
        let replacement = if value.is_empty() {
            Vec::new()
        } else {
            // Tokenise the value string into preprocessing tokens.
            tokenize_preprocessing(value, 0)
                .into_iter()
                .filter(|t| !t.is_eof())
                .collect()
        };

        let _ = self.interner.intern(name);
        self.macro_defs.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                kind: MacroKind::ObjectLike,
                replacement,
                is_predefined: false,
                definition_span: Span::dummy(),
            },
        );
    }

    /// Remove a macro definition, implementing the `-U` command-line flag.
    ///
    /// This is applied after all `-D` defines during preprocessor
    /// initialization, matching GCC's behaviour where `-U` overrides
    /// earlier `-D` for the same macro name.
    pub fn add_undef(&mut self, name: &str) {
        self.macro_defs.remove(name);
    }

    /// Preprocess a source file, returning the fully macro-expanded token stream.
    ///
    /// This is the main entry point for the preprocessor.  It:
    /// 1. Reads the file with PUA encoding for non-UTF-8 bytes.
    /// 2. Applies Phase 1 (trigraph replacement + line splicing).
    /// 3. Tokenizes into preprocessing tokens.
    /// 4. Processes directives and expands macros (Phase 2).
    /// 5. Returns the expanded token stream (minus whitespace/newlines) for the lexer.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the file cannot be read or if a fatal preprocessing
    /// error occurs (e.g., `#error` directive).  Non-fatal errors (warnings,
    /// recoverable syntax issues) are accumulated in the diagnostic engine.
    #[allow(clippy::result_unit_err)]
    pub fn preprocess_file(&mut self, filename: &str) -> Result<Vec<PPToken>, ()> {
        // Step 1: Read source file with PUA encoding.
        // Special case: "-" means read from stdin, "/dev/null" means empty input.
        let source = if filename == "-" {
            // Read from stdin
            use std::io::Read;
            let mut raw_bytes = Vec::new();
            match std::io::stdin().read_to_end(&mut raw_bytes) {
                Ok(_) => crate::common::encoding::encode_bytes_to_string(&raw_bytes),
                Err(e) => {
                    self.diagnostics.emit(Diagnostic::error(
                        Span::dummy(),
                        format!("cannot read from stdin: {}", e),
                    ));
                    return Err(());
                }
            }
        } else if filename == "/dev/null" {
            String::new()
        } else {
            let path = Path::new(filename);
            match read_source_file(path) {
                Ok(s) => s,
                Err(e) => {
                    self.diagnostics.emit(Diagnostic::error(
                        Span::dummy(),
                        format!("cannot open source file '{}': {}", filename, e),
                    ));
                    return Err(());
                }
            }
        };

        // Push the initial source file onto the include stack so that
        // circular detection works when an included file tries to
        // re-include the original source.
        let path = Path::new(filename);
        let canonical = if filename == "-" || filename == "/dev/null" {
            std::path::PathBuf::from(filename)
        } else {
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        };
        self.include_stack.push(canonical);

        // Step 2: Register the file in the source map.
        let file_id = self
            .source_map
            .add_file(filename.to_string(), source.clone());

        // Step 3: Apply Phase 1 transformations.
        let after_trigraphs = phase1_trigraphs(&source);
        let after_splice = phase1_line_splice(&after_trigraphs);

        // Step 4: Tokenize into preprocessing tokens.
        let tokens = tokenize_preprocessing(&after_splice, file_id);

        // Step 5: Process directives and expand macros (Phase 2).
        let expanded = self.process_tokens(&tokens);

        // Pop the initial source file from the include stack.
        self.include_stack.pop();

        // Step 6: Check for unterminated conditional blocks.
        if let Some(cond) = self.conditional_stack.last() {
            self.diagnostics.emit(Diagnostic::error(
                cond.opening_span,
                "unterminated #if/#ifdef/#ifndef — missing #endif",
            ));
        }

        if self.diagnostics.has_errors() {
            return Err(());
        }

        Ok(expanded)
    }

    /// Process a token stream, handling directives and macro expansion.
    ///
    /// This is the main Phase 2 processing loop.  It iterates through the
    /// preprocessing token stream line by line:
    /// - Lines beginning with `#` are dispatched to directive handling.
    /// - Non-directive lines in active conditional blocks are macro-expanded.
    /// - Lines in inactive conditional blocks are skipped (except for
    ///   nesting-tracking directives like `#if`/`#endif`).
    fn process_tokens(&mut self, tokens: &[PPToken]) -> Vec<PPToken> {
        let mut output = Vec::new();
        let mut pos = 0;
        let len = tokens.len();

        while pos < len {
            // End-of-file sentinel — nothing more to process.
            if tokens[pos].kind == PPTokenKind::EndOfFile {
                break;
            }

            // Skip to the first non-whitespace token to check for directive.
            let line_start = pos;

            // Collect whitespace at the start of the line.
            while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                pos += 1;
            }

            // Check for directive: `#` as first non-whitespace token.
            if pos < len && tokens[pos].kind == PPTokenKind::Punctuator && tokens[pos].text == "#" {
                let hash_pos = pos;
                pos += 1; // skip `#`

                // Skip whitespace after `#`.
                while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                    pos += 1;
                }

                // Collect tokens until end of line (newline or EOF).
                let mut directive_tokens = Vec::new();
                while pos < len
                    && tokens[pos].kind != PPTokenKind::Newline
                    && tokens[pos].kind != PPTokenKind::EndOfFile
                {
                    directive_tokens.push(tokens[pos].clone());
                    pos += 1;
                }

                // Skip the newline.
                if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                    pos += 1;
                }

                // Process the directive.  #include may produce output tokens.
                if let Some(included) =
                    self.process_directive_line(&tokens[hash_pos], &directive_tokens)
                {
                    output.extend(included);
                }
                continue;
            }

            // Not a directive line — check if we are in an active region.
            if !self.is_active() {
                // Skip tokens until end of line.
                while pos < len
                    && tokens[pos].kind != PPTokenKind::Newline
                    && tokens[pos].kind != PPTokenKind::EndOfFile
                {
                    pos += 1;
                }
                if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                    pos += 1;
                }
                continue;
            }

            // Active region — collect and macro-expand the line.
            //
            // C11 §6.10.3.4: After substitution, replacement tokens are
            // rescanned together with subsequent source tokens for further
            // macro names to replace.  We collect all non-directive tokens on
            // this line, then run them through the full MacroExpander which
            // handles multi-level expansion and paint-marker recursion
            // protection correctly.
            let restore_pos = line_start;
            pos = restore_pos;

            // Collect raw tokens for the current line.  In C preprocessing,
            // function-like macro invocations may span multiple source lines
            // (after line-splicing).  We must collect tokens across newlines
            // when a function-like macro argument list is open (unbalanced
            // parentheses).
            let mut line_tokens: Vec<PPToken> = Vec::new();
            let mut paren_depth: i32 = 0;
            // Stack of paren depths at which function-like macro invocations
            // opened.  We continue collecting across newlines whenever this
            // stack is non-empty (i.e., any macro arg list is open).
            let mut macro_arg_depths: Vec<i32> = Vec::new();
            while pos < len && tokens[pos].kind != PPTokenKind::EndOfFile {
                let tok = &tokens[pos];

                let in_macro_args = !macro_arg_depths.is_empty();

                // If we hit a newline and we're NOT inside macro arguments,
                // the line ends here — UNLESS the last non-whitespace token
                // is a function-like macro name and the next non-whitespace
                // token on the following line is `(`.  C11 §6.10.3 requires
                // that whitespace (including newlines) between a function-like
                // macro name and its opening `(` be ignored.
                if tok.kind == PPTokenKind::Newline && !in_macro_args {
                    // Peek: is the last non-whitespace token a function-like
                    // macro name?
                    let last_nws = line_tokens
                        .iter()
                        .rev()
                        .find(|t| t.kind != PPTokenKind::Whitespace);
                    let ends_with_func_macro = if let Some(lt) = last_nws {
                        if lt.kind == PPTokenKind::Identifier {
                            self.macro_defs
                                .get(&lt.text)
                                .map(|md| matches!(md.kind, MacroKind::FunctionLike { .. }))
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if ends_with_func_macro {
                        // Peek ahead past whitespace and newlines in the
                        // remaining token stream to find `(`.
                        let mut peek = pos + 1; // skip the current newline
                        while peek < len
                            && matches!(
                                tokens[peek].kind,
                                PPTokenKind::Whitespace | PPTokenKind::Newline
                            )
                        {
                            peek += 1;
                        }
                        if peek < len
                            && tokens[peek].kind == PPTokenKind::Punctuator
                            && tokens[peek].text == "("
                        {
                            // The `(` follows — skip the newline (and any
                            // whitespace/newlines before the paren) and
                            // continue collecting tokens on this logical
                            // line so `expand_tokens` sees the full
                            // function-like macro invocation.
                            pos += 1; // skip newline token
                            continue;
                        }
                    }
                    break;
                }

                // When inside macro argument lists and we hit a newline,
                // check whether subsequent lines start with preprocessing
                // directives (#ifdef, #ifndef, #if, #elif, #else, #endif).
                // GCC processes conditional directives that appear inside
                // macro arguments — the Linux kernel relies on this (e.g.,
                // `unlikely(expr | \n #ifdef CONFIG_X \n val | \n #endif \n rest)`).
                // §6.10.3p11: behavior is undefined but GCC extends it.
                //
                // We loop here to handle consecutive and nested directive
                // blocks (e.g., `#ifdef A \n #ifdef B \n ... #endif \n #endif`).
                if tok.kind == PPTokenKind::Newline && in_macro_args {
                    pos += 1;
                    // Process any run of consecutive directive lines.
                    loop {
                        // Skip whitespace to find start of the next logical line.
                        let saved_pos = pos;
                        while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                            pos += 1;
                        }
                        // Not a `#` → not a directive, stop loop.
                        if pos >= len
                            || tokens[pos].kind != PPTokenKind::Punctuator
                            || tokens[pos].text != "#"
                        {
                            pos = saved_pos;
                            break;
                        }
                        let hash_pos = pos;
                        pos += 1;
                        while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                            pos += 1;
                        }
                        // Check for a conditional directive keyword.
                        if pos >= len || tokens[pos].kind != PPTokenKind::Identifier {
                            pos = saved_pos;
                            break;
                        }
                        let dname = tokens[pos].text.clone();
                        if !matches!(
                            dname.as_str(),
                            "ifdef" | "ifndef" | "if" | "elif" | "else" | "endif"
                        ) {
                            // Not a conditional directive — restore and
                            // treat as ordinary tokens.
                            pos = saved_pos;
                            break;
                        }
                        // Consume directive keyword.
                        pos += 1;
                        // Skip leading whitespace before directive arguments.
                        while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                            pos += 1;
                        }
                        // Collect directive argument tokens until end of line.
                        let mut dir_tokens = Vec::new();
                        while pos < len
                            && tokens[pos].kind != PPTokenKind::Newline
                            && tokens[pos].kind != PPTokenKind::EndOfFile
                        {
                            dir_tokens.push(tokens[pos].clone());
                            pos += 1;
                        }
                        if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                            pos += 1;
                        }
                        // Process the conditional directive with proper
                        // inactive-region nesting (matching the behaviour
                        // of the top-level process_directive_line).
                        match dname.as_str() {
                            "ifdef" | "ifndef" | "if" => {
                                if !self.is_active() {
                                    self.conditional_stack
                                        .push(ConditionalState::new(false, tokens[hash_pos].span));
                                } else {
                                    match dname.as_str() {
                                        "ifdef" => {
                                            self.process_ifdef(&dir_tokens, tokens[hash_pos].span)
                                        }
                                        "ifndef" => {
                                            self.process_ifndef(&dir_tokens, tokens[hash_pos].span)
                                        }
                                        "if" => self.process_if(&dir_tokens, tokens[hash_pos].span),
                                        _ => unreachable!(),
                                    }
                                }
                            }
                            "elif" => self.process_elif(&tokens[hash_pos], &dir_tokens),
                            "else" => self.process_else(&tokens[hash_pos]),
                            "endif" => self.process_endif(&tokens[hash_pos]),
                            _ => unreachable!(),
                        }
                        // If we just entered an inactive region, skip non-
                        // directive lines until we become active again (or
                        // hit more directives which the next loop iteration
                        // handles).
                        while pos < len && !self.is_active() {
                            // Skip whitespace.
                            while pos < len && tokens[pos].kind == PPTokenKind::Whitespace {
                                pos += 1;
                            }
                            // Check for a directive line.
                            if pos < len
                                && tokens[pos].kind == PPTokenKind::Punctuator
                                && tokens[pos].text == "#"
                            {
                                // This is a directive — break out so the
                                // outer loop processes it (it might be
                                // #elif, #else, #endif, or nested #ifdef).
                                break;
                            }
                            // Non-directive line in inactive region — skip.
                            while pos < len
                                && tokens[pos].kind != PPTokenKind::Newline
                                && tokens[pos].kind != PPTokenKind::EndOfFile
                            {
                                pos += 1;
                            }
                            if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                                pos += 1;
                            }
                        }
                        // Loop back to check if the next line is also a directive.
                    }
                    continue;
                }

                // Track parenthesis depth for macro argument spanning.
                if tok.kind == PPTokenKind::Punctuator {
                    match tok.text.as_str() {
                        "(" => {
                            paren_depth += 1;
                            // Check if the preceding non-whitespace token
                            // is a function-like macro name — at any depth.
                            let last_ident = line_tokens
                                .iter()
                                .rev()
                                .find(|t| t.kind != PPTokenKind::Whitespace);
                            if let Some(prev) = last_ident {
                                if prev.kind == PPTokenKind::Identifier {
                                    if let Some(md) = self.macro_defs.get(&prev.text) {
                                        if matches!(md.kind, MacroKind::FunctionLike { .. }) {
                                            macro_arg_depths.push(paren_depth);
                                        } else if matches!(md.kind, MacroKind::ObjectLike) {
                                            // Object-like macro followed by `(`: check
                                            // if the replacement list is a single token
                                            // that is itself a function-like macro name.
                                            // This handles the kernel pattern:
                                            //   #define ALIAS FUNC
                                            //   ALIAS(args across\n lines)
                                            // where FUNC is a function-like macro.
                                            if md.replacement.len() == 1
                                                && md.replacement[0].kind == PPTokenKind::Identifier
                                            {
                                                if let Some(target_md) =
                                                    self.macro_defs.get(&md.replacement[0].text)
                                                {
                                                    if matches!(
                                                        target_md.kind,
                                                        MacroKind::FunctionLike { .. }
                                                    ) {
                                                        macro_arg_depths.push(paren_depth);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        ")" => {
                            // If this `)` closes the innermost macro arg list,
                            // pop it off the stack.
                            if let Some(&depth) = macro_arg_depths.last() {
                                if paren_depth == depth {
                                    macro_arg_depths.pop();
                                }
                            }
                            paren_depth -= 1;
                        }
                        _ => {}
                    }
                }

                if tok.kind != PPTokenKind::Whitespace || !line_tokens.is_empty() {
                    line_tokens.push(tok.clone());
                }
                pos += 1;
            }

            // Expand all macros (including multi-level rescanning) using
            // the full MacroExpander with source map for __LINE__/__FILE__.
            if !line_tokens.is_empty() {
                let expanded = {
                    let mut expander = macro_expander::MacroExpander::new_with_source_map(
                        &self.macro_defs,
                        &mut *self.diagnostics,
                        self.max_recursion_depth,
                        &mut self.counter_value,
                        self.source_map,
                    );
                    expander.expand_tokens(&line_tokens)
                };
                // C99/C11 §6.10.9: Handle `_Pragma( string-literal )`
                // operator.  After macro expansion, any remaining
                // `_Pragma("...")` sequences are consumed and discarded
                // (BCC ignores GCC diagnostic pragmas, etc.).
                let filtered = strip_pragma_operator(&expanded);
                output.extend(filtered);
            }

            // Emit a newline token so that `-E` output preserves line
            // structure.  The newline is part of the preprocessed output
            // (C11 §5.1.1.2 phase 4 produces a stream of pp-tokens and
            // white-space characters).
            if pos < len && tokens[pos].kind == PPTokenKind::Newline {
                output.push(tokens[pos].clone());
                pos += 1;
            }
        }

        // Always append an EndOfFile sentinel so downstream consumers can rely
        // on the token stream being terminated.
        let eof_span = if let Some(last) = tokens.last() {
            last.span
        } else {
            Span::dummy()
        };
        output.push(PPToken::eof(eof_span));

        output
    }

    /// Returns `true` if the current preprocessing position is in an active
    /// conditional compilation region (i.e., all enclosing `#if` blocks are active).
    fn is_active(&self) -> bool {
        self.conditional_stack.iter().all(|c| c.active)
    }

    /// Process a single preprocessor directive line.
    ///
    /// `hash_token` is the `#` token, `tokens` are the remaining tokens on the
    /// directive line (after `#` and whitespace).
    fn process_directive_line(
        &mut self,
        _hash_token: &PPToken,
        tokens: &[PPToken],
    ) -> Option<Vec<PPToken>> {
        // Null directive: `#` alone on a line — valid C11 no-op.
        if tokens.is_empty() {
            return None;
        }

        let directive = &tokens[0];
        let rest = if tokens.len() > 1 { &tokens[1..] } else { &[] };

        // Strip leading whitespace from rest.
        let rest: Vec<PPToken> = rest
            .iter()
            .skip_while(|t| t.kind == PPTokenKind::Whitespace)
            .cloned()
            .collect();

        let directive_name = directive.text.as_str();

        // Directives that must be processed even in inactive regions
        // (for nesting tracking).
        match directive_name {
            "if" | "ifdef" | "ifndef" => {
                if !self.is_active() {
                    // In inactive region: push a nested inactive conditional.
                    self.conditional_stack
                        .push(ConditionalState::new(false, directive.span));
                    return None;
                }
            }
            "elif" => {
                self.process_elif(directive, &rest);
                return None;
            }
            "else" => {
                self.process_else(directive);
                return None;
            }
            "endif" => {
                self.process_endif(directive);
                return None;
            }
            _ => {
                if !self.is_active() {
                    return None; // Skip all other directives in inactive regions.
                }
            }
        }

        // Active-region directives.
        match directive_name {
            "define" => {
                self.process_define(&rest);
                None
            }
            "undef" => {
                self.process_undef(&rest);
                None
            }
            "include" => {
                // process_include returns Ok(tokens) on success.
                self.process_include(&rest).ok()
            }
            "include_next" => {
                // GCC extension: search for the header starting after the
                // current file's directory in the include path.
                self.process_include_next(&rest).ok()
            }
            "if" => {
                self.process_if(&rest, directive.span);
                None
            }
            "ifdef" => {
                self.process_ifdef(&rest, directive.span);
                None
            }
            "ifndef" => {
                self.process_ifndef(&rest, directive.span);
                None
            }
            "error" => {
                self.process_error(&rest, directive.span);
                None
            }
            "warning" => {
                self.process_warning(&rest, directive.span);
                None
            }
            "line" => {
                self.process_line_directive(&rest);
                None
            }
            "pragma" => {
                self.process_pragma(&rest);
                // When in `-E` (preprocess-only) mode, preserve `#pragma`
                // directives in the output token stream, matching GCC
                // behaviour (C11 §6.10.6).
                if self.preserve_pragmas {
                    let mut pragma_tokens = Vec::new();
                    pragma_tokens.push(PPToken {
                        kind: PPTokenKind::Punctuator,
                        text: "#pragma".to_string(),
                        span: directive.span,
                        from_macro: false,
                        painted: false,
                    });
                    for t in &rest {
                        pragma_tokens.push(PPToken {
                            kind: PPTokenKind::Whitespace,
                            text: " ".to_string(),
                            span: t.span,
                            from_macro: false,
                            painted: false,
                        });
                        pragma_tokens.push(t.clone());
                    }
                    pragma_tokens.push(PPToken {
                        kind: PPTokenKind::Newline,
                        text: "\n".to_string(),
                        span: directive.span,
                        from_macro: false,
                        painted: false,
                    });
                    Some(pragma_tokens)
                } else {
                    None
                }
            }
            _ => {
                // Unknown directive — emit diagnostic.
                self.diagnostics.emit_warning(
                    directive.span,
                    format!(
                        "unknown preprocessing directive '#{}' — ignored",
                        directive_name
                    ),
                );
                None
            }
        }
    }

    // -- Directive handlers ------------------------------------------------

    /// Process `#define` directive.
    fn process_define(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected macro name after #define");
            return;
        }

        let name_token = &tokens[0];
        if name_token.kind != PPTokenKind::Identifier {
            self.diagnostics.emit_error(
                name_token.span,
                format!("macro name '{}' is not an identifier", name_token.text),
            );
            return;
        }

        let name = name_token.text.clone();
        let _ = self.interner.intern(&name);

        let mut idx = 1;

        // Check for function-like macro: `(` immediately follows name (no space).
        let kind;
        // GCC named variadic parameter: `#define FOO(a, args...)`.
        // When present, occurrences of `args` in the replacement list
        // are rewritten to `__VA_ARGS__` so the standard expansion
        // logic handles them without any further changes.
        let mut gcc_va_name: Option<String> = None;
        if idx < tokens.len()
            && tokens[idx].kind == PPTokenKind::Punctuator
            && tokens[idx].text == "("
        {
            // Function-like macro — parse parameter list.
            idx += 1; // skip `(`
            let mut params = Vec::new();
            let mut variadic = false;

            loop {
                // Skip whitespace.
                while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                    idx += 1;
                }
                if idx >= tokens.len() {
                    break;
                }

                // Check for `)`.
                if tokens[idx].kind == PPTokenKind::Punctuator && tokens[idx].text == ")" {
                    idx += 1;
                    break;
                }

                // Check for `...` (C99 variadic).
                if tokens[idx].kind == PPTokenKind::Punctuator && tokens[idx].text == "..." {
                    variadic = true;
                    idx += 1;
                    // Expect `)` next.
                    while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                        idx += 1;
                    }
                    if idx < tokens.len()
                        && tokens[idx].kind == PPTokenKind::Punctuator
                        && tokens[idx].text == ")"
                    {
                        idx += 1;
                    }
                    break;
                }

                // Parameter name.
                if tokens[idx].kind == PPTokenKind::Identifier {
                    let param_name = tokens[idx].text.clone();
                    idx += 1;

                    // Skip whitespace.
                    while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                        idx += 1;
                    }

                    // Check for GCC named variadic: `name...`
                    if idx < tokens.len()
                        && tokens[idx].kind == PPTokenKind::Punctuator
                        && tokens[idx].text == "..."
                    {
                        variadic = true;
                        gcc_va_name = Some(param_name);
                        idx += 1; // skip `...`
                                  // Expect `)` next.
                        while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                            idx += 1;
                        }
                        if idx < tokens.len()
                            && tokens[idx].kind == PPTokenKind::Punctuator
                            && tokens[idx].text == ")"
                        {
                            idx += 1;
                        }
                        break;
                    }

                    params.push(param_name);

                    // Check for `,` or `)`.
                    if idx < tokens.len()
                        && tokens[idx].kind == PPTokenKind::Punctuator
                        && tokens[idx].text == ","
                    {
                        idx += 1; // skip comma
                    }
                } else {
                    self.diagnostics.emit_error(
                        tokens[idx].span,
                        "expected parameter name or '...' in macro parameter list",
                    );
                    return;
                }
            }

            kind = MacroKind::FunctionLike { params, variadic };
        } else {
            // Object-like macro — skip whitespace before replacement.
            while idx < tokens.len() && tokens[idx].kind == PPTokenKind::Whitespace {
                idx += 1;
            }
            kind = MacroKind::ObjectLike;
        }

        // Collect replacement tokens (rest of the line).
        let replacement: Vec<PPToken> = tokens[idx..]
            .iter()
            .filter(|t| t.kind != PPTokenKind::EndOfFile)
            .cloned()
            .collect();

        // Trim trailing whitespace from replacement.
        let mut replacement: Vec<PPToken> = {
            let mut r = replacement;
            while r
                .last()
                .map_or(false, |t| t.kind == PPTokenKind::Whitespace)
            {
                r.pop();
            }
            r
        };

        // GCC named variadic: rewrite occurrences of the named parameter
        // (e.g., `ctx`) in the replacement list to `__VA_ARGS__` so the
        // standard C99 variadic expansion logic handles them transparently.
        if let Some(ref vn) = gcc_va_name {
            for tok in &mut replacement {
                if tok.kind == PPTokenKind::Identifier && tok.text == *vn {
                    tok.text = "__VA_ARGS__".to_string();
                }
            }
        }

        self.macro_defs.insert(
            name.clone(),
            MacroDef {
                name,
                kind,
                replacement,
                is_predefined: false,
                definition_span: name_token.span,
            },
        );
    }

    /// Process `#undef` directive.
    fn process_undef(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected macro name after #undef");
            return;
        }
        let name = &tokens[0].text;
        if let Some(def) = self.macro_defs.get(name) {
            if def.is_predefined {
                self.diagnostics.emit_warning(
                    tokens[0].span,
                    format!("undefining predefined macro '{}'", name),
                );
            }
        }
        self.macro_defs.remove(name.as_str());
    }

    /// Process `#include` directive.
    fn process_include(&mut self, tokens: &[PPToken]) -> Result<Vec<PPToken>, ()> {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected header name after #include");
            return Err(());
        }

        let directive_span = tokens[0].span;

        // Determine header name and type (system vs user).
        let (header_name, is_system) = match tokens[0].kind {
            PPTokenKind::HeaderName => {
                let text = &tokens[0].text;
                if text.starts_with('<') && text.ends_with('>') {
                    (text[1..text.len() - 1].to_string(), true)
                } else if text.starts_with('"') && text.ends_with('"') {
                    (text[1..text.len() - 1].to_string(), false)
                } else {
                    (text.clone(), false)
                }
            }
            PPTokenKind::StringLiteral => {
                let text = &tokens[0].text;
                let inner = if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
                    text[1..text.len() - 1].to_string()
                } else {
                    text.clone()
                };
                (inner, false)
            }
            PPTokenKind::Punctuator if tokens[0].text == "<" => {
                // Collect tokens until `>`.
                let mut header = String::new();
                let mut i = 1;
                while i < tokens.len() {
                    if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ">" {
                        break;
                    }
                    header.push_str(&tokens[i].text);
                    i += 1;
                }
                (header, true)
            }
            _ => {
                // C11 §6.10.2/4: Computed #include — macro-expand the tokens
                // and then attempt to parse the result as a header name.
                let expanded = {
                    let non_ws: Vec<PPToken> = tokens
                        .iter()
                        .filter(|t| t.kind != PPTokenKind::Whitespace)
                        .cloned()
                        .collect();
                    let mut expander = macro_expander::MacroExpander::new_with_source_map(
                        &self.macro_defs,
                        &mut *self.diagnostics,
                        self.max_recursion_depth,
                        &mut self.counter_value,
                        self.source_map,
                    );
                    expander.expand_tokens(&non_ws)
                };
                // Concatenate expanded tokens into a single string and look
                // for `"..."` or `<...>` patterns.
                let combined: String = expanded
                    .iter()
                    .filter(|t| {
                        t.kind != PPTokenKind::Whitespace && t.kind != PPTokenKind::PlacemarkerToken
                    })
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("");
                let trimmed = combined.trim();
                if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
                    (trimmed[1..trimmed.len() - 1].to_string(), false)
                } else if trimmed.starts_with('<') && trimmed.ends_with('>') && trimmed.len() >= 2 {
                    (trimmed[1..trimmed.len() - 1].to_string(), true)
                } else {
                    // Try joining with spaces between tokens for stringified result.
                    let spaced: String = expanded
                        .iter()
                        .filter(|t| {
                            t.kind != PPTokenKind::Whitespace
                                && t.kind != PPTokenKind::PlacemarkerToken
                        })
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let st = spaced.trim();
                    if st.starts_with('"') && st.ends_with('"') && st.len() >= 2 {
                        (st[1..st.len() - 1].to_string(), false)
                    } else if st.starts_with('<') && st.ends_with('>') && st.len() >= 2 {
                        (st[1..st.len() - 1].to_string(), true)
                    } else {
                        self.diagnostics.emit_error(
                            tokens[0].span,
                            "expected header name in #include directive",
                        );
                        return Err(());
                    }
                }
            }
        };

        // Determine the including file for relative path resolution.
        let including_file = self
            .source_map
            .get_filename(directive_span.file_id)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."));

        // Enforce include depth limit before recursing.  This check uses
        // the persistent `include_depth` counter on `self`, which is
        // incremented for every nested `#include` and decremented when the
        // included file finishes processing.
        if self.include_depth >= self.max_include_depth {
            self.diagnostics.emit_error(
                directive_span,
                format!(
                    "#include nested too deeply ({} levels, maximum is {})",
                    self.include_depth, self.max_include_depth
                ),
            );
            return Err(());
        }

        // Build a temporary handler for file resolution and guard checks.
        let handler = include_handler::IncludeHandler::new(
            self.include_paths.clone(),
            self.system_include_paths.clone(),
        );

        // Resolve the header to an absolute path.
        let resolved_path = match handler.resolve_include(&header_name, is_system, &including_file)
        {
            Some(path) => path,
            None => {
                self.diagnostics
                    .emit_error(directive_span, format!("'{}' file not found", header_name));
                return Err(());
            }
        };

        // Check include guards and #pragma once.
        if handler.should_skip_file(&resolved_path, &self.macro_defs) {
            return Ok(Vec::new());
        }

        // Canonicalize the resolved path for reliable circular detection.
        let canonical_path =
            std::fs::canonicalize(&resolved_path).unwrap_or_else(|_| resolved_path.clone());

        // Circular include detection: check if this file is already being
        // processed in the current include chain.  This uses the persistent
        // `include_stack` on `self` (not a per-call handler) so that the
        // stack survives across recursive `process_include` calls.
        //
        // When a circular include is detected, we silently return an empty
        // token list instead of emitting an error. This matches GCC/Clang
        // behavior: the included file's include guard (`#ifndef GUARD`)
        // would cause the re-included content to be empty anyway. The
        // kernel heavily relies on this pattern (e.g., linkage.h → export.h
        // → linkage.h), so treating circularity as a hard error would
        // break real-world code.
        if self.include_stack.contains(&canonical_path) {
            return Ok(Vec::new());
        }

        // Read the file contents using PUA-encoded reader.
        let source_content = match crate::common::encoding::read_source_file(&resolved_path) {
            Ok(content) => content,
            Err(e) => {
                self.diagnostics.emit_error(
                    directive_span,
                    format!("cannot read '{}': {}", resolved_path.display(), e),
                );
                return Err(());
            }
        };

        // Register the file in the source map.
        let file_id = self.source_map.add_file(
            resolved_path.to_string_lossy().to_string(),
            source_content.clone(),
        );

        // Apply Phase 1 transforms (trigraphs and line splicing).
        let processed = phase1_line_splice(&phase1_trigraphs(&source_content));

        // Tokenize the included source.
        let included_tokens = tokenize_preprocessing(&processed, file_id);

        // Push file onto the persistent include stack for circular detection
        // and increment depth counter, then recursively preprocess.
        self.include_stack.push(canonical_path.clone());
        self.include_depth += 1;
        let result = self.process_tokens(&included_tokens);
        self.include_depth -= 1;
        self.include_stack.pop();

        // Filter out the EOF token from included output to avoid premature
        // termination of the parent file's token stream.
        let filtered: Vec<PPToken> = result
            .into_iter()
            .filter(|t| t.kind != PPTokenKind::EndOfFile)
            .collect();

        Ok(filtered)
    }

    /// Process `#include_next` directive (GCC extension).
    ///
    /// `#include_next` searches for the header starting from the directory
    /// **after** the current file's directory in the include search path.
    /// This is used by system headers (e.g., glibc's `<limits.h>`) to chain
    /// to the compiler's built-in header of the same name.
    fn process_include_next(&mut self, tokens: &[PPToken]) -> Result<Vec<PPToken>, ()> {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected header name after #include_next");
            return Err(());
        }

        let directive_span = tokens[0].span;

        // Parse the header name (same logic as #include).
        let (header_name, _is_system) = match tokens[0].kind {
            PPTokenKind::HeaderName => {
                let text = &tokens[0].text;
                if text.starts_with('<') && text.ends_with('>') {
                    (text[1..text.len() - 1].to_string(), true)
                } else if text.starts_with('"') && text.ends_with('"') {
                    (text[1..text.len() - 1].to_string(), false)
                } else {
                    (text.clone(), false)
                }
            }
            PPTokenKind::StringLiteral => {
                let text = &tokens[0].text;
                let inner = if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
                    text[1..text.len() - 1].to_string()
                } else {
                    text.clone()
                };
                (inner, false)
            }
            PPTokenKind::Punctuator if tokens[0].text == "<" => {
                let mut header = String::new();
                let mut i = 1;
                while i < tokens.len() {
                    if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ">" {
                        break;
                    }
                    header.push_str(&tokens[i].text);
                    i += 1;
                }
                (header, true)
            }
            _ => {
                self.diagnostics.emit_error(
                    tokens[0].span,
                    "expected header name in #include_next directive",
                );
                return Err(());
            }
        };

        // Determine the including file for path resolution.
        let including_file = self
            .source_map
            .get_filename(directive_span.file_id)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."));

        if self.include_depth >= self.max_include_depth {
            self.diagnostics.emit_error(
                directive_span,
                format!(
                    "#include_next nested too deeply ({} levels, maximum is {})",
                    self.include_depth, self.max_include_depth
                ),
            );
            return Err(());
        }

        let handler = include_handler::IncludeHandler::new(
            self.include_paths.clone(),
            self.system_include_paths.clone(),
        );

        // Use #include_next resolution: skip to the directory AFTER the
        // current file's directory in the search path.
        let resolved_path = match handler.resolve_include_next(&header_name, &including_file) {
            Some(path) => path,
            None => {
                // Fall back to normal include resolution if include_next
                // doesn't find it (matches GCC behavior for edge cases).
                match handler.resolve_include(&header_name, true, &including_file) {
                    Some(path) => path,
                    None => {
                        self.diagnostics.emit_error(
                            directive_span,
                            format!("'{}' file not found", header_name),
                        );
                        return Err(());
                    }
                }
            }
        };

        // Apply include guards and circular detection identically to #include.
        if handler.should_skip_file(&resolved_path, &self.macro_defs) {
            return Ok(Vec::new());
        }

        let canonical_path =
            std::fs::canonicalize(&resolved_path).unwrap_or_else(|_| resolved_path.clone());

        if self.include_stack.contains(&canonical_path) {
            return Ok(Vec::new());
        }

        let source_content = match crate::common::encoding::read_source_file(&resolved_path) {
            Ok(content) => content,
            Err(e) => {
                self.diagnostics.emit_error(
                    directive_span,
                    format!("cannot read '{}': {}", resolved_path.display(), e),
                );
                return Err(());
            }
        };

        let file_id = self.source_map.add_file(
            resolved_path.to_string_lossy().to_string(),
            source_content.clone(),
        );

        let processed = phase1_line_splice(&phase1_trigraphs(&source_content));
        let included_tokens = tokenize_preprocessing(&processed, file_id);

        self.include_stack.push(canonical_path.clone());
        self.include_depth += 1;
        let result = self.process_tokens(&included_tokens);
        self.include_depth -= 1;
        self.include_stack.pop();

        let filtered: Vec<PPToken> = result
            .into_iter()
            .filter(|t| t.kind != PPTokenKind::EndOfFile)
            .collect();

        Ok(filtered)
    }

    /// Process `#if` directive.
    fn process_if(&mut self, tokens: &[PPToken], directive_span: Span) {
        // Evaluate the expression — for now, treat non-empty as true.
        let value = self.evaluate_condition(tokens);
        self.conditional_stack
            .push(ConditionalState::new(value, directive_span));
    }

    /// Process `#ifdef` directive.
    fn process_ifdef(&mut self, tokens: &[PPToken], directive_span: Span) {
        let is_defined = if let Some(name_tok) = tokens.first() {
            self.macro_defs.contains_key(&name_tok.text)
        } else {
            self.diagnostics
                .emit_error(directive_span, "expected macro name after #ifdef");
            false
        };
        self.conditional_stack
            .push(ConditionalState::new(is_defined, directive_span));
    }

    /// Process `#ifndef` directive.
    fn process_ifndef(&mut self, tokens: &[PPToken], directive_span: Span) {
        let is_defined = if let Some(name_tok) = tokens.first() {
            self.macro_defs.contains_key(&name_tok.text)
        } else {
            self.diagnostics
                .emit_error(directive_span, "expected macro name after #ifndef");
            false
        };
        self.conditional_stack
            .push(ConditionalState::new(!is_defined, directive_span));
    }

    /// Process `#elif` directive.
    fn process_elif(&mut self, directive_token: &PPToken, tokens: &[PPToken]) {
        if self.conditional_stack.is_empty() {
            self.diagnostics
                .emit_error(directive_token.span, "#elif without matching #if");
            return;
        }
        let stack_len = self.conditional_stack.len();

        // Check structural errors first (seen_else).
        if self.conditional_stack[stack_len - 1].seen_else {
            self.diagnostics
                .emit_error(directive_token.span, "#elif after #else");
            return;
        }

        // Check if the parent context is active.  If any enclosing
        // conditional block is inactive, this `#elif` branch is
        // unconditionally inactive — we must NOT evaluate the
        // expression (it may contain tokens that are invalid in
        // preprocessor expressions, such as `__alignof__`).
        let parent_active = if stack_len > 1 {
            self.conditional_stack[..stack_len - 1]
                .iter()
                .all(|cs| cs.active)
        } else {
            true
        };

        if !parent_active {
            // Parent is inactive — unconditionally deactivate this branch
            // without evaluating the expression.
            self.conditional_stack[stack_len - 1].active = false;
            return;
        }

        if self.conditional_stack[stack_len - 1].seen_active {
            // A previous branch was taken — deactivate.
            self.conditional_stack[stack_len - 1].active = false;
        } else {
            // No branch taken yet — evaluate condition using full evaluator.
            let value = self.evaluate_condition(tokens);
            let top = self.conditional_stack.last_mut().unwrap();
            top.active = value;
            if value {
                top.seen_active = true;
            }
        }
    }

    /// Process `#else` directive.
    fn process_else(&mut self, directive_token: &PPToken) {
        if let Some(top) = self.conditional_stack.last_mut() {
            if top.seen_else {
                self.diagnostics
                    .emit_error(directive_token.span, "duplicate #else");
                return;
            }
            top.seen_else = true;
            top.active = !top.seen_active;
        } else {
            self.diagnostics
                .emit_error(directive_token.span, "#else without matching #if");
        }
    }

    /// Process `#endif` directive.
    fn process_endif(&mut self, directive_token: &PPToken) {
        if self.conditional_stack.pop().is_none() {
            self.diagnostics
                .emit_error(directive_token.span, "#endif without matching #if");
        }
    }

    /// Process `#error` directive.
    fn process_error(&mut self, tokens: &[PPToken], directive_span: Span) {
        let msg: String = tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        self.diagnostics
            .emit_error(directive_span, format!("#error {}", msg.trim()));
    }

    /// Process `#warning` directive.
    fn process_warning(&mut self, tokens: &[PPToken], directive_span: Span) {
        let msg: String = tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        self.diagnostics
            .emit_warning(directive_span, format!("#warning {}", msg.trim()));
    }

    /// Process `#line` directive.
    fn process_line_directive(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            self.diagnostics
                .emit_error(Span::dummy(), "expected line number after #line");
            return;
        }
        // Parse the line number from the first token.
        if let Ok(line_num) = tokens[0].text.parse::<u32>() {
            // Optionally parse filename.
            let filename = tokens
                .get(1)
                .and_then(|t| {
                    if t.kind == PPTokenKind::Whitespace {
                        tokens.get(2)
                    } else {
                        Some(t)
                    }
                })
                .and_then(|t| {
                    if t.kind == PPTokenKind::StringLiteral {
                        let s = &t.text;
                        if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                            Some(s[1..s.len() - 1].to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });
            let file_id = tokens[0].span.file_id;
            let offset = tokens[0].span.start;
            self.source_map
                .add_line_directive(crate::common::source_map::LineDirective {
                    file_id,
                    directive_offset: offset,
                    new_line: line_num,
                    new_filename: filename,
                });
        } else {
            self.diagnostics.emit_error(
                tokens[0].span,
                format!(
                    "invalid line number '{}' in #line directive",
                    tokens[0].text
                ),
            );
        }
    }

    /// Process `#pragma` directive.
    fn process_pragma(&mut self, tokens: &[PPToken]) {
        if tokens.is_empty() {
            return; // Empty pragma — ignore.
        }
        match tokens[0].text.as_str() {
            "once" => {
                // Mark the current file to prevent re-inclusion.
                // (Tracked by include_handler in the full implementation.)
            }
            "pack" | "GCC" | "clang" => {
                // Recognized but not fully handled at the preprocessor level.
                // The parser/sema will handle pack push/pop semantics.
            }
            "push_macro" => {
                // #pragma push_macro("name") — save current macro definition.
                let rest = &tokens[1..];
                // Extract macro name from ( "name" ) tokens.
                if let Some(name) = Self::extract_pragma_macro_name_from_tokens(rest) {
                    let current_def = self.macro_defs.get(&name).cloned();
                    self.macro_push_stack
                        .entry(name)
                        .or_default()
                        .push(current_def);
                }
            }
            "pop_macro" => {
                // #pragma pop_macro("name") — restore previously pushed macro definition.
                let rest = &tokens[1..];
                if let Some(name) = Self::extract_pragma_macro_name_from_tokens(rest) {
                    if let Some(stack) = self.macro_push_stack.get_mut(&name) {
                        if let Some(saved_def) = stack.pop() {
                            match saved_def {
                                Some(def) => {
                                    self.macro_defs.insert(name, def);
                                }
                                None => {
                                    self.macro_defs.remove(&name);
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Unknown pragmas are silently ignored per C11 §6.10.6.
            }
        }
    }

    /// Extract macro name from push_macro/pop_macro pragma tokens.
    /// Expected format: ( "name" ) — tokens after push_macro/pop_macro keyword.
    fn extract_pragma_macro_name_from_tokens(tokens: &[PPToken]) -> Option<String> {
        // Skip whitespace tokens.
        let non_ws: Vec<&PPToken> = tokens
            .iter()
            .filter(|t| !t.text.trim().is_empty() && t.text != " ")
            .collect();
        // We expect: ( "name" )
        // The tokens might be: ["(", "\"NAME\"", ")"] or just ["\"NAME\""]
        for tok in &non_ws {
            let t = &tok.text;
            if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
                return Some(t[1..t.len() - 1].to_string());
            }
        }
        // Try unquoted name after '('.
        if non_ws.len() >= 3 && non_ws[0].text == "(" {
            let name = &non_ws[1].text;
            if name != ")" && name != "(" {
                return Some(name.clone());
            }
        }
        None
    }

    // -- Macro expansion ---------------------------------------------------

    /// Evaluate a preprocessor condition expression.
    ///
    /// This performs the full C11 §6.10.1 evaluation sequence:
    /// 1. Resolve `defined(X)` / `defined X` operators to `1` / `0` tokens
    /// 2. Macro-expand remaining tokens
    /// 3. Replace any remaining identifiers with `0` (handled by evaluator)
    /// 4. Evaluate the resulting integer constant expression
    fn evaluate_condition(&mut self, tokens: &[PPToken]) -> bool {
        // Step 1: Resolve `defined` operators BEFORE macro expansion.
        let with_defined = directives::resolve_defined_operators(tokens, &self.macro_defs);
        // Step 2: Macro-expand remaining tokens.
        let expanded = {
            let mut expander = macro_expander::MacroExpander::new_with_source_map(
                &self.macro_defs,
                &mut *self.diagnostics,
                self.max_recursion_depth,
                &mut self.counter_value,
                self.source_map,
            );
            expander.expand_tokens(&with_defined)
        };
        // Step 3+4: Evaluate the expression.
        match expression::evaluate_pp_expression(&expanded, &mut *self.diagnostics) {
            Ok(value) => value.is_nonzero(),
            Err(()) => false,
        }
    }

    /// Expand an object-like macro, returning the expanded token list.
    #[allow(dead_code)]
    fn expand_object_macro(&self, macro_def: &MacroDef, invocation_span: Span) -> Vec<PPToken> {
        // For predefined magic macros, generate dynamic values.
        if macro_def.is_predefined {
            match macro_def.name.as_str() {
                "__FILE__" => {
                    let filename = self
                        .source_map
                        .get_filename(invocation_span.file_id)
                        .unwrap_or("<unknown>");
                    return vec![PPToken::from_expansion(
                        PPTokenKind::StringLiteral,
                        format!("\"{}\"", filename),
                        invocation_span,
                    )];
                }
                "__LINE__" => {
                    if let Some(file) = self.source_map.get_file(invocation_span.file_id) {
                        let (line, _) = file.lookup_line_col(invocation_span.start);
                        return vec![PPToken::from_expansion(
                            PPTokenKind::Number,
                            line.to_string(),
                            invocation_span,
                        )];
                    }
                }
                _ => {}
            }
        }

        // Normal object-like expansion: clone replacement tokens.
        macro_def
            .replacement
            .iter()
            .map(|t| {
                let mut tok = t.clone();
                tok.from_macro = true;
                tok.span = invocation_span;
                tok
            })
            .collect()
    }

    /// Expand a function-like macro invocation, collecting arguments from the
    /// token stream starting at `paren_pos` (position of the opening `(`).
    ///
    /// Returns the expanded tokens and the position after the closing `)`.
    #[allow(dead_code)]
    fn expand_function_macro(
        &self,
        macro_def: &MacroDef,
        tokens: &[PPToken],
        name_pos: usize,
        paren_pos: usize,
    ) -> (Vec<PPToken>, usize) {
        let invocation_span = tokens[name_pos].span;
        let (params, variadic) = match &macro_def.kind {
            MacroKind::FunctionLike { params, variadic } => (params.clone(), *variadic),
            MacroKind::ObjectLike => {
                return (vec![tokens[name_pos].clone()], name_pos + 1);
            }
        };

        // Collect arguments.
        let mut args: Vec<Vec<PPToken>> = Vec::new();
        let mut current_arg: Vec<PPToken> = Vec::new();
        let mut depth = 1;
        let mut pos = paren_pos + 1; // skip `(`

        while pos < tokens.len() && depth > 0 {
            let tok = &tokens[pos];
            if tok.kind == PPTokenKind::Punctuator {
                match tok.text.as_str() {
                    "(" => {
                        depth += 1;
                        current_arg.push(tok.clone());
                    }
                    ")" => {
                        depth -= 1;
                        if depth == 0 {
                            args.push(current_arg);
                            current_arg = Vec::new();
                        } else {
                            current_arg.push(tok.clone());
                        }
                    }
                    "," if depth == 1 => {
                        args.push(current_arg);
                        current_arg = Vec::new();
                    }
                    _ => {
                        current_arg.push(tok.clone());
                    }
                }
            } else {
                current_arg.push(tok.clone());
            }
            pos += 1;
        }

        // Perform substitution.
        let mut result = Vec::new();
        for rtok in &macro_def.replacement {
            if rtok.kind == PPTokenKind::Identifier {
                // Check if this is a parameter reference.
                if let Some(param_idx) = params.iter().position(|p| p == &rtok.text) {
                    if let Some(arg_tokens) = args.get(param_idx) {
                        result.extend(arg_tokens.iter().cloned());
                    }
                    continue;
                }
                // Check for __VA_ARGS__.
                if variadic && rtok.text == "__VA_ARGS__" {
                    // Variadic arguments start after the last named parameter.
                    let va_start = params.len();
                    for (i, arg) in args.iter().enumerate().skip(va_start) {
                        if i > va_start {
                            result.push(PPToken::new(
                                PPTokenKind::Punctuator,
                                ",",
                                invocation_span,
                            ));
                        }
                        result.extend(arg.iter().cloned());
                    }
                    continue;
                }
            }
            let mut tok = rtok.clone();
            tok.from_macro = true;
            tok.span = invocation_span;
            result.push(tok);
        }

        (result, pos)
    }
}

// ---------------------------------------------------------------------------
// Helper: C99/C11 _Pragma operator stripping
// ---------------------------------------------------------------------------

/// Remove `_Pragma ( "string-literal" )` sequences from a token stream.
///
/// C99 §6.10.9 defines `_Pragma` as a unary operator on string literals that
/// behaves like `#pragma`.  BCC currently does not execute the pragma content
/// (e.g. GCC diagnostic push/pop) — we simply strip the entire expression so
/// it does not confuse the parser.
///
/// The function scans for the 4-token pattern:
///   `_Pragma` `(` `<string-literal>` `)`
/// skipping whitespace tokens between them.
fn strip_pragma_operator(tokens: &[PPToken]) -> Vec<PPToken> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    let len = tokens.len();

    while i < len {
        // Check for `_Pragma` identifier.
        if tokens[i].kind == PPTokenKind::Identifier && tokens[i].text == "_Pragma" {
            // Scan ahead for `( string-literal )`, skipping whitespace.
            let mut j = i + 1;
            // Skip whitespace.
            while j < len && tokens[j].kind == PPTokenKind::Whitespace {
                j += 1;
            }
            if j < len && tokens[j].kind == PPTokenKind::Punctuator && tokens[j].text == "(" {
                j += 1;
                // Skip whitespace.
                while j < len && tokens[j].kind == PPTokenKind::Whitespace {
                    j += 1;
                }
                if j < len && tokens[j].kind == PPTokenKind::StringLiteral {
                    j += 1;
                    // Skip whitespace.
                    while j < len && tokens[j].kind == PPTokenKind::Whitespace {
                        j += 1;
                    }
                    if j < len && tokens[j].kind == PPTokenKind::Punctuator && tokens[j].text == ")"
                    {
                        // Matched full `_Pragma("...")` — skip all tokens [i..=j].
                        i = j + 1;
                        continue;
                    }
                }
            }
        }
        // No match — emit the current token.
        result.push(tokens[i].clone());
        i += 1;
    }

    result
}

// ---------------------------------------------------------------------------
// Helper: simplified condition evaluation
// ---------------------------------------------------------------------------

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Phase 1: Trigraphs -----------------------------------------------

    #[test]
    fn trigraph_all_nine_replacements() {
        assert_eq!(phase1_trigraphs("??="), "#");
        assert_eq!(phase1_trigraphs("??/"), "\\");
        assert_eq!(phase1_trigraphs("??'"), "^");
        assert_eq!(phase1_trigraphs("??("), "[");
        assert_eq!(phase1_trigraphs("??)"), "]");
        assert_eq!(phase1_trigraphs("??!"), "|");
        assert_eq!(phase1_trigraphs("??<"), "{");
        assert_eq!(phase1_trigraphs("??>"), "}");
        assert_eq!(phase1_trigraphs("??-"), "~");
    }

    #[test]
    fn trigraph_in_context() {
        assert_eq!(phase1_trigraphs("a??=b"), "a#b");
        assert_eq!(phase1_trigraphs("x??(y??)z"), "x[y]z");
    }

    #[test]
    fn trigraph_no_match() {
        assert_eq!(phase1_trigraphs("??"), "??");
        assert_eq!(phase1_trigraphs("?x"), "?x");
        assert_eq!(phase1_trigraphs("??@"), "??@");
        assert_eq!(phase1_trigraphs("abc"), "abc");
    }

    #[test]
    fn trigraph_empty_and_short() {
        assert_eq!(phase1_trigraphs(""), "");
        assert_eq!(phase1_trigraphs("a"), "a");
        assert_eq!(phase1_trigraphs("ab"), "ab");
    }

    // -- Phase 1: Line Splicing -------------------------------------------

    #[test]
    fn line_splice_unix() {
        assert_eq!(phase1_line_splice("hello\\\nworld"), "helloworld");
    }

    #[test]
    fn line_splice_windows() {
        assert_eq!(phase1_line_splice("hello\\\r\nworld"), "helloworld");
    }

    #[test]
    fn line_splice_multiple() {
        assert_eq!(phase1_line_splice("a\\\nb\\\nc"), "abc");
    }

    #[test]
    fn line_splice_no_splice() {
        assert_eq!(phase1_line_splice("hello\nworld"), "hello\nworld");
    }

    #[test]
    fn line_splice_backslash_not_before_newline() {
        assert_eq!(phase1_line_splice("path\\to\\file"), "path\\to\\file");
    }

    // -- Tokenizer --------------------------------------------------------

    #[test]
    fn tokenize_simple_identifier() {
        let tokens = tokenize_preprocessing("hello", 0);
        assert_eq!(tokens.len(), 2); // identifier + EOF
        assert_eq!(tokens[0].kind, PPTokenKind::Identifier);
        assert_eq!(tokens[0].text, "hello");
    }

    #[test]
    fn tokenize_number() {
        let tokens = tokenize_preprocessing("42", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "42");
    }

    #[test]
    fn tokenize_hex_number() {
        let tokens = tokenize_preprocessing("0xFF", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Number);
        assert_eq!(tokens[0].text, "0xFF");
    }

    #[test]
    fn tokenize_string_literal() {
        let tokens = tokenize_preprocessing("\"hello\"", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::StringLiteral);
        assert_eq!(tokens[0].text, "\"hello\"");
    }

    #[test]
    fn tokenize_char_literal() {
        let tokens = tokenize_preprocessing("'a'", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::CharLiteral);
        assert_eq!(tokens[0].text, "'a'");
    }

    #[test]
    fn tokenize_punctuators() {
        let tokens = tokenize_preprocessing("+ - -> == !=", 0);
        let puncts: Vec<&str> = tokens
            .iter()
            .filter(|t| t.kind == PPTokenKind::Punctuator)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(puncts, vec!["+", "-", "->", "==", "!="]);
    }

    #[test]
    fn tokenize_directive_hash() {
        let tokens = tokenize_preprocessing("#define FOO 42", 0);
        assert_eq!(tokens[0].kind, PPTokenKind::Punctuator);
        assert_eq!(tokens[0].text, "#");
    }

    #[test]
    fn tokenize_comment_replaced_by_space() {
        let tokens = tokenize_preprocessing("a /* comment */ b", 0);
        let non_ws: Vec<&str> = tokens
            .iter()
            .filter(|t| !t.is_whitespace() && !t.is_eof())
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(non_ws, vec!["a", "b"]);
    }

    #[test]
    fn tokenize_line_comment() {
        let tokens = tokenize_preprocessing("a // comment\nb", 0);
        let non_ws: Vec<&str> = tokens
            .iter()
            .filter(|t| !t.is_whitespace() && t.kind != PPTokenKind::Newline && !t.is_eof())
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(non_ws, vec!["a", "b"]);
    }

    // -- PPToken constructors ---------------------------------------------

    #[test]
    fn pptoken_new() {
        let tok = PPToken::new(PPTokenKind::Identifier, "foo", Span::new(0, 0, 3));
        assert_eq!(tok.kind, PPTokenKind::Identifier);
        assert_eq!(tok.text, "foo");
        assert!(!tok.from_macro);
        assert!(!tok.painted);
    }

    #[test]
    fn pptoken_eof() {
        let tok = PPToken::eof(Span::dummy());
        assert!(tok.is_eof());
    }

    #[test]
    fn pptoken_placemarker() {
        let tok = PPToken::placemarker(Span::dummy());
        assert_eq!(tok.kind, PPTokenKind::PlacemarkerToken);
        assert!(tok.from_macro);
    }

    // -- ConditionalState -------------------------------------------------

    #[test]
    fn conditional_state_active() {
        let state = ConditionalState::new(true, Span::dummy());
        assert!(state.active);
        assert!(state.seen_active);
        assert!(!state.seen_else);
    }

    #[test]
    fn conditional_state_inactive() {
        let state = ConditionalState::new(false, Span::dummy());
        assert!(!state.active);
        assert!(!state.seen_active);
        assert!(!state.seen_else);
    }

    // -- MacroDef and MacroKind -------------------------------------------

    #[test]
    fn macro_def_object_like() {
        let def = MacroDef {
            name: "FOO".to_string(),
            kind: MacroKind::ObjectLike,
            replacement: vec![PPToken::new(PPTokenKind::Number, "42", Span::dummy())],
            is_predefined: false,
            definition_span: Span::dummy(),
        };
        assert_eq!(def.name, "FOO");
        assert!(matches!(def.kind, MacroKind::ObjectLike));
    }

    #[test]
    fn macro_def_function_like() {
        let def = MacroDef {
            name: "MAX".to_string(),
            kind: MacroKind::FunctionLike {
                params: vec!["a".to_string(), "b".to_string()],
                variadic: false,
            },
            replacement: Vec::new(),
            is_predefined: false,
            definition_span: Span::dummy(),
        };
        assert!(matches!(def.kind, MacroKind::FunctionLike { .. }));
    }

    // Condition evaluation is now tested via the full expression evaluator
    // in the `expression` submodule.  The previous `evaluate_simple_condition`
    // helper was removed in favour of the complete implementation.
}
