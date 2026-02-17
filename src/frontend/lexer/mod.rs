//! Phase 3 — Lexer module for the BCC C compiler.
//!
//! This module implements the tokenization pipeline that converts a PUA-encoded
//! source string (output of the preprocessor, Phase 1–2) into a stream of
//! [`Token`] values consumed by the recursive-descent parser (Phase 4).
//!
//! # Architecture
//!
//! The lexer is structured as a pull-based iterator: each call to
//! [`Lexer::next_token()`] returns exactly one token, advancing the internal
//! [`Scanner`] position. The main tokenization loop:
//!
//! 1. Skips whitespace and comments (single-line `//`, multi-line `/* */`).
//! 2. Records the byte-offset start position for the upcoming [`Span`].
//! 3. Dispatches to the appropriate sub-lexer based on the current character:
//!    - **Digits / dot-digit** → [`number_literal::lex_number`]
//!    - **Encoding prefix + quote** → [`string_literal::lex_string_literal`] /
//!      [`string_literal::lex_char_literal`]
//!    - **Identifier start** → internal keyword classification table
//!    - **Operator/punctuator** → multi-character operator matching
//!    - **EOF** → `TokenKind::Eof`
//! 4. Constructs a `Token` carrying the resolved `TokenKind` and byte-offset
//!    `Span`, then returns it.
//!
//! # PUA-Aware Processing
//!
//! The lexer operates on source text already processed by
//! [`crate::common::encoding::read_source_file`], which maps non-UTF-8 bytes
//! (0x80–0xFF) to Private Use Area code points (U+E080–U+E0FF). The scanner
//! transparently passes PUA code points through without interpretation, while
//! character classification helpers explicitly exclude them from identifier
//! and whitespace categories to preserve byte-exact round-tripping fidelity.
//!
//! # GCC Extension Keyword Recognition
//!
//! Per §0.4.1 of the Agent Action Plan, GCC extension keywords
//! (`__attribute__`, `__typeof__`, `__extension__`, `__builtin_*`, `asm`,
//! `__asm__`, etc.) are recognised at the lexer level. A static keyword table
//! backed by [`FxHashMap`] maps all C11 keywords, C11 special keywords, GCC
//! extension keywords, and GCC builtin identifiers to their corresponding
//! [`TokenKind`] variants.
//!
//! # Integration
//!
//! ```text
//! Preprocessor (Phase 1–2) ─── expanded source text ──→ Lexer (Phase 3)
//!                                                          │
//!                                                     Token stream
//!                                                          │
//!                                                          ▼
//!                                                    Parser (Phase 4)
//! ```

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// Token type definitions — the most foundational type in the entire frontend.
///
/// Defines [`TokenKind`], [`Token`], [`Span`] (re-exported from diagnostics),
/// and auxiliary enums for literal suffixes and prefixes. Every subsequent
/// frontend module (preprocessor, parser, sema) imports from this module.
pub mod token;

/// Character-level scanner for PUA-aware UTF-8 source reading.
///
/// Provides the foundational character-reading layer: peek/advance with
/// O(1) lookahead, byte-offset and line/column position tracking, source
/// slice extraction, and character classification that respects PUA opacity.
pub mod scanner;

/// Numeric literal lexing — decimal, hex, octal, binary integers and
/// decimal/hex floating-point literals.
///
/// Handles parsing of all C11 numeric literal forms with integer suffixes
/// (u/U, l/L, ll/LL and combinations), float suffixes (f/F, l/L),
/// hex float binary exponents (p/P), digit separator support, and
/// comprehensive error recovery with clear diagnostics.
pub mod number_literal;

/// String and character literal lexing.
///
/// Handles parsing of C11 string and character literals with full escape
/// sequence support (simple, octal, hex, universal character names),
/// wide/unicode encoding prefixes (L, u8, u, U), adjacent string literal
/// concatenation per C11 §6.4.5, and PUA transparency for non-UTF-8
/// source byte round-tripping (Section 0.7.9).
pub mod string_literal;

// ---------------------------------------------------------------------------
// Re-exports — canonical types available as `lexer::Token`, `lexer::Span`, etc.
// ---------------------------------------------------------------------------

pub use scanner::Scanner;
pub use token::{Span, Token, TokenKind};

// ---------------------------------------------------------------------------
// Internal imports
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::{FileId, SourceMap};
use crate::common::string_interner::Interner;

use token::{CharPrefix, StringPrefix};

// ---------------------------------------------------------------------------
// Keyword table — lazily initialised, shared across all Lexer instances
// ---------------------------------------------------------------------------

/// Returns a reference to the global keyword classification table.
///
/// The table maps identifier strings to their corresponding [`TokenKind`],
/// covering all C11 keywords, C11 special keywords (_Alignas, _Atomic, …),
/// GCC extension keywords (__attribute__, __typeof__, asm, …), and GCC
/// builtins (__builtin_va_list, __builtin_expect, …).
///
/// Initialised once on first access via [`OnceLock`] — subsequent calls
/// return the cached reference with zero allocation.
fn keyword_table() -> &'static FxHashMap<&'static str, TokenKind> {
    static TABLE: OnceLock<FxHashMap<&'static str, TokenKind>> = OnceLock::new();
    TABLE.get_or_init(|| {
        // Pre-size for ~110 entries (34 C11 + 10 C11 special + ~15 GCC ext + ~25 builtins + headroom).
        let mut m: FxHashMap<&'static str, TokenKind> =
            FxHashMap::with_hasher(crate::common::fx_hash::FxBuildHasher);
        m.reserve(128);

        // =================================================================
        // C11 standard keywords (34)
        // =================================================================
        m.insert("auto", TokenKind::Auto);
        m.insert("break", TokenKind::Break);
        m.insert("case", TokenKind::Case);
        m.insert("char", TokenKind::Char);
        m.insert("const", TokenKind::Const);
        m.insert("continue", TokenKind::Continue);
        m.insert("default", TokenKind::Default);
        m.insert("do", TokenKind::Do);
        m.insert("double", TokenKind::Double);
        m.insert("else", TokenKind::Else);
        m.insert("enum", TokenKind::Enum);
        m.insert("extern", TokenKind::Extern);
        m.insert("float", TokenKind::Float);
        m.insert("for", TokenKind::For);
        m.insert("goto", TokenKind::Goto);
        m.insert("if", TokenKind::If);
        m.insert("inline", TokenKind::Inline);
        m.insert("int", TokenKind::Int);
        m.insert("long", TokenKind::Long);
        m.insert("register", TokenKind::Register);
        m.insert("restrict", TokenKind::Restrict);
        m.insert("return", TokenKind::Return);
        m.insert("short", TokenKind::Short);
        m.insert("signed", TokenKind::Signed);
        m.insert("sizeof", TokenKind::Sizeof);
        m.insert("static", TokenKind::Static);
        m.insert("struct", TokenKind::Struct);
        m.insert("switch", TokenKind::Switch);
        m.insert("typedef", TokenKind::Typedef);
        m.insert("union", TokenKind::Union);
        m.insert("unsigned", TokenKind::Unsigned);
        m.insert("void", TokenKind::Void);
        m.insert("volatile", TokenKind::Volatile);
        m.insert("while", TokenKind::While);

        // =================================================================
        // C11 special keywords (10)
        // =================================================================
        m.insert("_Alignas", TokenKind::Alignas);
        m.insert("__alignas__", TokenKind::Alignas);
        m.insert("_Alignof", TokenKind::Alignof);
        m.insert("__alignof__", TokenKind::Alignof);
        m.insert("__alignof", TokenKind::Alignof);
        m.insert("_Atomic", TokenKind::Atomic);
        m.insert("_Bool", TokenKind::Bool);
        m.insert("_Complex", TokenKind::Complex);
        m.insert("_Generic", TokenKind::Generic);
        m.insert("_Imaginary", TokenKind::Imaginary);
        m.insert("_Noreturn", TokenKind::Noreturn);
        m.insert("_Static_assert", TokenKind::StaticAssert);
        m.insert("_Thread_local", TokenKind::ThreadLocal);

        // =================================================================
        // GCC extension keywords (with alternate spellings)
        // =================================================================
        m.insert("__attribute__", TokenKind::Attribute);
        m.insert("__attribute", TokenKind::Attribute);

        m.insert("typeof", TokenKind::TypeofKeyword);
        m.insert("__typeof__", TokenKind::TypeofKeyword);
        m.insert("__typeof", TokenKind::TypeofKeyword);

        m.insert("__extension__", TokenKind::Extension);

        m.insert("asm", TokenKind::AsmKeyword);
        m.insert("__asm__", TokenKind::AsmKeyword);
        m.insert("__asm", TokenKind::AsmKeyword);

        m.insert("__volatile__", TokenKind::VolatileGcc);
        m.insert("__volatile", TokenKind::VolatileGcc);

        m.insert("__inline__", TokenKind::InlineGcc);
        m.insert("__inline", TokenKind::InlineGcc);

        m.insert("__signed__", TokenKind::SignedGcc);
        m.insert("__signed", TokenKind::SignedGcc);

        // __unsigned is an alternate spelling of the standard `unsigned`.
        m.insert("__unsigned", TokenKind::Unsigned);

        m.insert("__const__", TokenKind::ConstGcc);
        m.insert("__const", TokenKind::ConstGcc);

        m.insert("__restrict__", TokenKind::RestrictGcc);
        m.insert("__restrict", TokenKind::RestrictGcc);

        m.insert("__label__", TokenKind::Label);

        // =================================================================
        // GCC builtins — variadic argument support
        // =================================================================
        m.insert("__builtin_va_list", TokenKind::BuiltinVaList);
        m.insert("__builtin_va_start", TokenKind::BuiltinVaStart);
        m.insert("__builtin_va_end", TokenKind::BuiltinVaEnd);
        m.insert("__builtin_va_arg", TokenKind::BuiltinVaArg);
        m.insert("__builtin_va_copy", TokenKind::BuiltinVaCopy);

        // =================================================================
        // GCC builtins — type introspection and compile-time evaluation
        // =================================================================
        m.insert("__builtin_offsetof", TokenKind::BuiltinOffsetof);
        m.insert(
            "__builtin_types_compatible_p",
            TokenKind::BuiltinTypesCompatibleP,
        );
        m.insert("__builtin_choose_expr", TokenKind::BuiltinChooseExpr);
        m.insert("__builtin_constant_p", TokenKind::BuiltinConstantP);

        // =================================================================
        // GCC builtins — branch prediction and control flow
        // =================================================================
        m.insert("__builtin_expect", TokenKind::BuiltinExpect);
        m.insert("__builtin_unreachable", TokenKind::BuiltinUnreachable);
        m.insert("__builtin_trap", TokenKind::BuiltinTrap);

        // =================================================================
        // GCC builtins — bit manipulation
        // =================================================================
        m.insert("__builtin_clz", TokenKind::BuiltinClz);
        m.insert("__builtin_ctz", TokenKind::BuiltinCtz);
        m.insert("__builtin_popcount", TokenKind::BuiltinPopcount);

        // =================================================================
        // GCC builtins — byte swap
        // =================================================================
        m.insert("__builtin_bswap16", TokenKind::BuiltinBswap16);
        m.insert("__builtin_bswap32", TokenKind::BuiltinBswap32);
        m.insert("__builtin_bswap64", TokenKind::BuiltinBswap64);

        // =================================================================
        // GCC builtins — miscellaneous
        // =================================================================
        m.insert("__builtin_ffs", TokenKind::BuiltinFfs);
        m.insert("__builtin_frame_address", TokenKind::BuiltinFrameAddress);
        m.insert("__builtin_return_address", TokenKind::BuiltinReturnAddress);
        m.insert("__builtin_assume_aligned", TokenKind::BuiltinAssumeAligned);

        // =================================================================
        // GCC builtins — checked arithmetic (overflow detection)
        // =================================================================
        m.insert("__builtin_add_overflow", TokenKind::BuiltinAddOverflow);
        m.insert("__builtin_sub_overflow", TokenKind::BuiltinSubOverflow);
        m.insert("__builtin_mul_overflow", TokenKind::BuiltinMulOverflow);

        m
    })
}

// ---------------------------------------------------------------------------
// Keyword Classification — Phase 3 post-processing for preprocessor output
// ---------------------------------------------------------------------------

/// Classify identifier tokens as keywords in a preprocessor-produced token
/// stream.
///
/// The preprocessor's internal tokenizer (`pp_tokenize`) produces a raw token
/// stream where all identifiers — including C keywords, GCC extension
/// keywords, and builtin names — are stored as `TokenKind::Identifier(Symbol)`.
/// This function performs a **single O(n) pass** over the token stream,
/// resolving each identifier's text via the [`Interner`] and looking it up in
/// the static keyword table. Matching identifiers are replaced in-place with
/// their corresponding `TokenKind` variant (e.g., `Int`, `Return`,
/// `Attribute`).
///
/// This step bridges the gap between the preprocessor (Phase 2) and the parser
/// (Phase 4), which expects keywords to carry their specific `TokenKind`
/// values for correct dispatch during recursive-descent parsing.
///
/// Non-identifier tokens (operators, literals, whitespace, etc.) are left
/// untouched. Identifiers that do not match any keyword remain as
/// `TokenKind::Identifier(Symbol)`.
///
/// # Arguments
///
/// * `tokens` — mutable slice of tokens to classify in-place.
/// * `interner` — string interner used to resolve `Symbol` handles back to
///   the identifier text for keyword lookup.
///
/// # Performance
///
/// Single O(n) pass over the token stream with O(1) expected-time FxHashMap
/// lookups for each identifier. No allocations are performed.
pub fn classify_keywords(tokens: &mut [Token], interner: &crate::common::Interner) {
    let table = keyword_table();
    for token in tokens.iter_mut() {
        if let TokenKind::Identifier(sym) = &token.kind {
            let text = interner.resolve(*sym);
            if let Some(kw_kind) = table.get(text) {
                token.kind = kw_kind.clone();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lexer — the Phase 3 tokenization driver
// ---------------------------------------------------------------------------

/// The BCC Phase 3 lexer — converts a PUA-encoded source string into a
/// token stream for the parser.
///
/// # Lifetime
///
/// `'src` ties the lexer to the lifetime of the source text being tokenised,
/// the string interner, the diagnostic engine, and the source map.
///
/// # Usage
///
/// ```ignore
/// let mut lexer = Lexer::new(source, file_id, &mut interner, &mut diag, &source_map);
/// loop {
///     let tok = lexer.next_token();
///     if tok.kind == TokenKind::Eof { break; }
///     // … process tok …
/// }
/// ```
pub struct Lexer<'src> {
    /// Character-level scanner providing peek, advance, and byte-offset tracking.
    scanner: Scanner<'src>,
    /// String interner — identifiers are interned here for zero-cost comparison.
    interner: &'src mut Interner,
    /// Diagnostic reporter — errors and warnings are emitted through this.
    diagnostics: &'src mut DiagnosticEngine,
    /// Source file registry — stored for deferred line/column resolution when
    /// diagnostics are rendered. Not queried during tokenisation itself for
    /// performance (byte offsets in `Span` are sufficient for the parser).
    source_map: &'src SourceMap,
    /// ID of the current source file being tokenised. Embedded into every
    /// `Token`'s `Span.file_id` for cross-file source location tracking.
    file_id: FileId,
}

// --- Public accessors for the Lexer's held resources ---
impl<'src> Lexer<'src> {
    /// Returns a reference to the source map held by this lexer.
    ///
    /// Useful for diagnostic formatting that needs line/column resolution
    /// after tokenisation is complete.
    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        self.source_map
    }

    /// Returns the [`FileId`] of the source file currently being tokenised.
    #[inline]
    pub fn file_id(&self) -> FileId {
        self.file_id
    }
}

impl<'src> Lexer<'src> {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Creates a new `Lexer` positioned at the beginning of `source`.
    ///
    /// # Arguments
    ///
    /// * `source` — PUA-encoded source text (output of the preprocessor).
    /// * `file_id` — The [`FileId`] of the file in the source map.
    /// * `interner` — Shared string interner for identifier deduplication.
    /// * `diagnostics` — Shared diagnostic engine for error/warning reporting.
    /// * `source_map` — Shared source file registry for location resolution.
    pub fn new(
        source: &'src str,
        file_id: FileId,
        interner: &'src mut Interner,
        diagnostics: &'src mut DiagnosticEngine,
        source_map: &'src SourceMap,
    ) -> Self {
        Lexer {
            scanner: Scanner::new(source, file_id.0),
            interner,
            diagnostics,
            source_map,
            file_id,
        }
    }

    // -------------------------------------------------------------------
    // Public tokenization API
    // -------------------------------------------------------------------

    /// Returns the next token from the source.
    ///
    /// This is the core lexer entry point. Each call:
    /// 1. Skips whitespace and comments.
    /// 2. Determines the token category from the current character.
    /// 3. Dispatches to the appropriate sub-lexer or internal handler.
    /// 4. Returns a [`Token`] with the resolved kind and byte-offset span.
    ///
    /// Once the end of input is reached, every subsequent call returns
    /// `TokenKind::Eof` with a zero-width span at the end of the source.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let start = self.scanner.byte_offset();
        let fid = self.scanner.file_id();

        // ----- EOF check -----
        let ch = match self.scanner.peek() {
            Some(c) => c,
            None => return Token::new(TokenKind::Eof, Span::new(fid, start, start)),
        };

        // ----- Numeric literals (decimal digit or dot-followed-by-digit) -----
        if Scanner::is_digit(ch) {
            return number_literal::lex_number(&mut self.scanner, self.diagnostics);
        }
        if ch == '.' {
            if let Some(next) = self.scanner.peek_ahead(1) {
                if Scanner::is_digit(next) {
                    return number_literal::lex_number(&mut self.scanner, self.diagnostics);
                }
            }
            // Otherwise fall through to operator handling for plain '.'
        }

        // ----- String / char literal with encoding prefix (L, u, U, u8) -----
        if matches!(ch, 'L' | 'u' | 'U') {
            if let Some(prefix) = string_literal::detect_string_prefix(&mut self.scanner) {
                // detect_string_prefix restores the scanner position after peeking.
                // Save position so we can reset on the (logically unreachable) fallback.
                let restore_mark = self.scanner.mark();

                // Advance past the prefix characters to reach the quote.
                let prefix_len: u32 = match prefix {
                    StringPrefix::U8 => 2,
                    StringPrefix::L | StringPrefix::SmallU | StringPrefix::BigU => 1,
                    StringPrefix::None => 0,
                };
                for _ in 0..prefix_len {
                    self.scanner.advance();
                }

                // Dispatch based on the opening quote character.
                match self.scanner.peek() {
                    Some('"') => {
                        let mut tok = string_literal::lex_string_literal(
                            &mut self.scanner,
                            prefix,
                            self.diagnostics,
                        );
                        // Adjust span to include the encoding prefix.
                        tok.span = Span::new(fid, start, tok.span.end);
                        return tok;
                    }
                    Some('\'') => {
                        let char_prefix = match prefix {
                            StringPrefix::L => CharPrefix::L,
                            StringPrefix::SmallU => CharPrefix::SmallU,
                            StringPrefix::BigU => CharPrefix::BigU,
                            _ => CharPrefix::None,
                        };
                        let mut tok = string_literal::lex_char_literal(
                            &mut self.scanner,
                            char_prefix,
                            self.diagnostics,
                        );
                        tok.span = Span::new(fid, start, tok.span.end);
                        return tok;
                    }
                    _ => {
                        // detect_string_prefix already confirmed a quote follows the
                        // prefix, so this branch is logically unreachable. Reset and
                        // fall through to identifier handling as a safety net.
                        self.scanner.reset_to(restore_mark);
                    }
                }
            }
            // Not a string/char prefix — fall through to identifier.
        }

        // ----- Non-prefixed string literal -----
        if ch == '"' {
            return string_literal::lex_string_literal(
                &mut self.scanner,
                StringPrefix::None,
                self.diagnostics,
            );
        }

        // ----- Non-prefixed character literal -----
        if ch == '\'' {
            return string_literal::lex_char_literal(
                &mut self.scanner,
                CharPrefix::None,
                self.diagnostics,
            );
        }

        // ----- Identifiers and keywords -----
        if Scanner::is_identifier_start(ch) {
            return self.lex_identifier_or_keyword(start);
        }

        // ----- Operators and punctuators -----
        self.lex_operator(start)
    }

    /// Tokenises the entire remaining source and collects all tokens into a
    /// `Vec`, including the final `TokenKind::Eof`.
    pub fn tokenize_all(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        tokens
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Skips all whitespace characters and comments, advancing the scanner
    /// past them.
    ///
    /// Handles:
    /// - ASCII whitespace: space, tab, vertical tab, form feed, CR, LF.
    /// - Single-line comments: `//` through end-of-line (newline NOT consumed).
    /// - Multi-line comments: `/* … */` (warns on nested `/*`, errors on unterminated).
    fn skip_whitespace_and_comments(&mut self) {
        let fid = self.scanner.file_id();
        loop {
            match self.scanner.peek() {
                // --- Whitespace ---
                Some(ch) if Scanner::is_whitespace(ch) => {
                    self.scanner.advance();
                }

                // --- Potential comment start ---
                Some('/') => {
                    match self.scanner.peek_ahead(1) {
                        // Single-line comment: // … <newline>
                        Some('/') => {
                            self.scanner.advance(); // consume first '/'
                            self.scanner.advance(); // consume second '/'
                            loop {
                                match self.scanner.peek() {
                                    None | Some('\n') | Some('\r') => break,
                                    _ => {
                                        self.scanner.advance();
                                    }
                                }
                            }
                            // Do NOT consume the newline — it is significant
                            // for preprocessor directive boundary detection.
                        }

                        // Multi-line comment: /* … */
                        Some('*') => {
                            let comment_start = self.scanner.byte_offset();
                            self.scanner.advance(); // consume '/'
                            self.scanner.advance(); // consume '*'

                            let mut terminated = false;
                            loop {
                                match self.scanner.peek() {
                                    None => break, // EOF inside comment
                                    Some('*') => {
                                        self.scanner.advance();
                                        if self.scanner.advance_if('/') {
                                            terminated = true;
                                            break;
                                        }
                                    }
                                    Some('/') => {
                                        // Warn about nested /* (non-standard).
                                        if self.scanner.peek_ahead(1) == Some('*') {
                                            let nested_start = self.scanner.byte_offset();
                                            self.diagnostics.warning(
                                                Span::new(fid, nested_start, nested_start + 2),
                                                "'/*' within block comment",
                                            );
                                        }
                                        self.scanner.advance();
                                    }
                                    _ => {
                                        self.scanner.advance();
                                    }
                                }
                            }

                            if !terminated {
                                self.diagnostics.error(
                                    Span::new(fid, comment_start, self.scanner.byte_offset()),
                                    "unterminated block comment",
                                );
                            }
                        }

                        // Not a comment — stop skipping.
                        _ => break,
                    }
                }

                // --- Any other character (or EOF) — stop skipping ---
                _ => break,
            }
        }
    }

    /// Lexes an identifier starting at byte offset `start` and classifies
    /// it as either a keyword or a plain identifier.
    ///
    /// The scanner must be positioned at the first character of the identifier
    /// (which has already been verified as [`Scanner::is_identifier_start`]).
    fn lex_identifier_or_keyword(&mut self, start: u32) -> Token {
        // Consume all identifier-continue characters (the first character
        // satisfies is_identifier_continue as well since every start char
        // is also a continue char).
        self.scanner.advance_while(Scanner::is_identifier_continue);

        let text = self.scanner.slice_from(start);
        let end = self.scanner.byte_offset();
        let span = Span::new(self.scanner.file_id(), start, end);

        // Keyword lookup — O(1) expected time via FxHashMap.
        if let Some(kind) = keyword_table().get(text) {
            return Token::new(kind.clone(), span);
        }

        // Not a keyword — intern the string and produce an Identifier token.
        let symbol = self.interner.intern(text);
        Token::new(TokenKind::Identifier(symbol), span)
    }

    /// Lexes an operator or punctuator token starting at byte offset `start`.
    ///
    /// Handles single-character, two-character, and three-character operators
    /// using one-character lookahead via [`Scanner::peek`]. Unknown characters
    /// produce `TokenKind::Error` with a diagnostic.
    fn lex_operator(&mut self, start: u32) -> Token {
        let fid = self.scanner.file_id();
        // Consume the first character — we know at least one exists.
        let ch = self.scanner.advance().unwrap();

        let kind = match ch {
            // ---- Arithmetic / assignment operators ----
            '+' => {
                if self.scanner.advance_if('+') {
                    TokenKind::PlusPlus
                } else if self.scanner.advance_if('=') {
                    TokenKind::PlusAssign
                } else {
                    TokenKind::Plus
                }
            }
            '-' => {
                if self.scanner.advance_if('-') {
                    TokenKind::MinusMinus
                } else if self.scanner.advance_if('>') {
                    TokenKind::Arrow
                } else if self.scanner.advance_if('=') {
                    TokenKind::MinusAssign
                } else {
                    TokenKind::Minus
                }
            }
            '*' => {
                if self.scanner.advance_if('=') {
                    TokenKind::StarAssign
                } else {
                    TokenKind::Star
                }
            }
            '/' => {
                // Comments are already stripped by skip_whitespace_and_comments,
                // so '/' here is always the division operator.
                if self.scanner.advance_if('=') {
                    TokenKind::SlashAssign
                } else {
                    TokenKind::Slash
                }
            }
            '%' => {
                if self.scanner.advance_if('=') {
                    TokenKind::PercentAssign
                } else {
                    TokenKind::Percent
                }
            }

            // ---- Bitwise / logical operators ----
            '&' => {
                if self.scanner.advance_if('&') {
                    TokenKind::AmpAmp
                } else if self.scanner.advance_if('=') {
                    TokenKind::AmpAssign
                } else {
                    TokenKind::Ampersand
                }
            }
            '|' => {
                if self.scanner.advance_if('|') {
                    TokenKind::PipePipe
                } else if self.scanner.advance_if('=') {
                    TokenKind::PipeAssign
                } else {
                    TokenKind::Pipe
                }
            }
            '^' => {
                if self.scanner.advance_if('=') {
                    TokenKind::CaretAssign
                } else {
                    TokenKind::Caret
                }
            }
            '~' => TokenKind::Tilde,
            '!' => {
                if self.scanner.advance_if('=') {
                    TokenKind::NotEqual
                } else {
                    TokenKind::Exclaim
                }
            }

            // ---- Comparison / shift operators ----
            '<' => {
                if self.scanner.advance_if('<') {
                    if self.scanner.advance_if('=') {
                        TokenKind::LeftShiftAssign
                    } else {
                        TokenKind::LeftShift
                    }
                } else if self.scanner.advance_if('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                }
            }
            '>' => {
                if self.scanner.advance_if('>') {
                    if self.scanner.advance_if('=') {
                        TokenKind::RightShiftAssign
                    } else {
                        TokenKind::RightShift
                    }
                } else if self.scanner.advance_if('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                }
            }

            // ---- Assignment / equality ----
            '=' => {
                if self.scanner.advance_if('=') {
                    TokenKind::EqualEqual
                } else {
                    TokenKind::Assign
                }
            }

            // ---- Dot / ellipsis ----
            '.' => {
                if self.scanner.peek() == Some('.') && self.scanner.peek_ahead(1) == Some('.') {
                    self.scanner.advance(); // second '.'
                    self.scanner.advance(); // third '.'
                    TokenKind::Ellipsis
                } else {
                    TokenKind::Dot
                }
            }

            // ---- Delimiters and separators ----
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            ':' => TokenKind::Colon,
            '?' => TokenKind::Question,
            '(' => TokenKind::LeftParen,
            ')' => TokenKind::RightParen,
            '[' => TokenKind::LeftBracket,
            ']' => TokenKind::RightBracket,
            '{' => TokenKind::LeftBrace,
            '}' => TokenKind::RightBrace,

            // ---- Preprocessor operators ----
            '#' => {
                if self.scanner.advance_if('#') {
                    TokenKind::HashHash
                } else {
                    TokenKind::Hash
                }
            }

            // ---- Unknown character — error recovery ----
            _ => {
                let end = self.scanner.byte_offset();
                self.diagnostics.error(
                    Span::new(fid, start, end),
                    format!("unexpected character '{}'", ch),
                );
                TokenKind::Error
            }
        };

        let end = self.scanner.byte_offset();
        Token::new(kind, Span::new(fid, start, end))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::source_map::SourceMap;
    use crate::common::string_interner::Interner;

    /// Helper: tokenise a source string and return all tokens (including EOF).
    fn tokenize(source: &str) -> Vec<Token> {
        let mut sm = SourceMap::new();
        let fid = sm.add_file("test.c".into(), source.into());
        let mut interner = Interner::new();
        let mut diag = DiagnosticEngine::new();
        let mut lexer = Lexer::new(source, fid, &mut interner, &mut diag, &sm);
        lexer.tokenize_all()
    }

    /// Helper: tokenise and collect only the TokenKinds (excluding EOF).
    fn token_kinds(source: &str) -> Vec<TokenKind> {
        tokenize(source)
            .into_iter()
            .filter(|t| t.kind != TokenKind::Eof)
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn test_empty_source() {
        let tokens = tokenize("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_whitespace_only() {
        let tokens = tokenize("   \t\n  \r\n  ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_single_line_comment() {
        let tokens = tokenize("// this is a comment\n42");
        assert_eq!(tokens.len(), 2);
        match &tokens[0].kind {
            TokenKind::IntegerLiteral { value, .. } => assert_eq!(*value, 42),
            other => panic!("expected IntegerLiteral, got {:?}", other),
        }
        assert_eq!(tokens[1].kind, TokenKind::Eof);
    }

    #[test]
    fn test_multi_line_comment() {
        let tokens = tokenize("/* block */ 7");
        assert_eq!(tokens.len(), 2);
        match &tokens[0].kind {
            TokenKind::IntegerLiteral { value, .. } => assert_eq!(*value, 7),
            other => panic!("expected IntegerLiteral, got {:?}", other),
        }
    }

    #[test]
    fn test_keywords() {
        let kinds = token_kinds("int return void if else while for");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Int,
                TokenKind::Return,
                TokenKind::Void,
                TokenKind::If,
                TokenKind::Else,
                TokenKind::While,
                TokenKind::For,
            ]
        );
    }

    #[test]
    fn test_gcc_extension_keywords() {
        let kinds = token_kinds("__attribute__ __typeof__ __extension__ asm __asm__");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Attribute,
                TokenKind::TypeofKeyword,
                TokenKind::Extension,
                TokenKind::AsmKeyword,
                TokenKind::AsmKeyword,
            ]
        );
    }

    #[test]
    fn test_gcc_builtins() {
        let kinds = token_kinds("__builtin_expect __builtin_clz __builtin_va_start");
        assert_eq!(
            kinds,
            vec![
                TokenKind::BuiltinExpect,
                TokenKind::BuiltinClz,
                TokenKind::BuiltinVaStart,
            ]
        );
    }

    #[test]
    fn test_identifier() {
        let kinds = token_kinds("my_variable foo123 _bar");
        assert_eq!(kinds.len(), 3);
        for kind in &kinds {
            assert!(matches!(kind, TokenKind::Identifier(_)));
        }
    }

    #[test]
    fn test_operators_single_char() {
        let kinds = token_kinds("+ - * / % & | ^ ~ ! < > = . , ; : ? ( ) [ ] { }");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::Ampersand,
                TokenKind::Pipe,
                TokenKind::Caret,
                TokenKind::Tilde,
                TokenKind::Exclaim,
                TokenKind::Less,
                TokenKind::Greater,
                TokenKind::Assign,
                TokenKind::Dot,
                TokenKind::Comma,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::Question,
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
            ]
        );
    }

    #[test]
    fn test_operators_multi_char() {
        let kinds =
            token_kinds("== != <= >= << >> -> ++ -- && || += -= *= /= %= &= |= ^= <<= >>= ...");
        assert_eq!(
            kinds,
            vec![
                TokenKind::EqualEqual,
                TokenKind::NotEqual,
                TokenKind::LessEqual,
                TokenKind::GreaterEqual,
                TokenKind::LeftShift,
                TokenKind::RightShift,
                TokenKind::Arrow,
                TokenKind::PlusPlus,
                TokenKind::MinusMinus,
                TokenKind::AmpAmp,
                TokenKind::PipePipe,
                TokenKind::PlusAssign,
                TokenKind::MinusAssign,
                TokenKind::StarAssign,
                TokenKind::SlashAssign,
                TokenKind::PercentAssign,
                TokenKind::AmpAssign,
                TokenKind::PipeAssign,
                TokenKind::CaretAssign,
                TokenKind::LeftShiftAssign,
                TokenKind::RightShiftAssign,
                TokenKind::Ellipsis,
            ]
        );
    }

    #[test]
    fn test_hash_and_hash_hash() {
        let kinds = token_kinds("# ##");
        assert_eq!(kinds, vec![TokenKind::Hash, TokenKind::HashHash]);
    }

    #[test]
    fn test_span_tracking() {
        let tokens = tokenize("int x");
        // "int" occupies bytes 0..3
        assert_eq!(tokens[0].span.start, 0);
        assert_eq!(tokens[0].span.end, 3);
        // "x" occupies bytes 4..5
        assert_eq!(tokens[1].span.start, 4);
        assert_eq!(tokens[1].span.end, 5);
    }

    #[test]
    fn test_c11_special_keywords() {
        let kinds = token_kinds(
            "_Alignas _Alignof _Atomic _Bool _Static_assert _Thread_local _Generic _Noreturn",
        );
        assert_eq!(
            kinds,
            vec![
                TokenKind::Alignas,
                TokenKind::Alignof,
                TokenKind::Atomic,
                TokenKind::Bool,
                TokenKind::StaticAssert,
                TokenKind::ThreadLocal,
                TokenKind::Generic,
                TokenKind::Noreturn,
            ]
        );
    }
}
