//! # BCC Lexer — Phase 3: Tokenization
//!
//! Tokenizes preprocessed C11 source into a typed token stream consumed by the
//! parser. Supports all C11 keywords, GCC extension keywords, numeric/string/
//! character literals, operators, and punctuators. Uses PUA-aware UTF-8 scanning
//! for transparent handling of non-UTF-8 source bytes.
//!
//! # Architecture
//!
//! The lexer is the main entry point for Phase 3 of the compilation pipeline.
//! It consists of four submodules:
//!
//! - [`token`] — Token type definitions ([`Token`], [`TokenKind`], keyword lookup)
//! - [`scanner`] — Character-level PUA-aware input scanner ([`Scanner`])
//! - [`number_literal`] — Numeric literal parsing (integers and floats)
//! - [`string_literal`] — String and character literal parsing with escape sequences
//!
//! The [`Lexer`] struct drives the tokenization loop, dispatching to specialized
//! submodule functions based on the current character. It provides a lookahead
//! interface ([`Lexer::peek`], [`Lexer::peek_nth`], [`Lexer::unget`]) for the
//! recursive-descent parser to use.
//!
//! # GCC Extension Keywords
//!
//! GCC extension keywords (`__attribute__`, `__typeof__`, `__extension__`,
//! `asm`/`__asm__`, all `__builtin_*` names) are recognized as proper
//! [`TokenKind`] variants — **not** as plain identifiers. The parser depends on
//! exact keyword dispatch for these tokens.
//!
//! # PUA Transparency
//!
//! Non-UTF-8 bytes encoded as PUA code points (U+E080–U+E0FF) by the
//! preprocessor flow through the lexer transparently within string and character
//! literals. PUA code points appearing outside literal contexts are diagnosed as
//! errors with a helpful message identifying the original byte value.
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library (`std`) and internal
//! crate modules (`crate::common::*`). No external crates are used.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

pub mod number_literal;
pub mod scanner;
pub mod string_literal;
pub mod token;

// ---------------------------------------------------------------------------
// Re-exports for convenient access by downstream modules (parser, sema)
// ---------------------------------------------------------------------------

pub use scanner::Scanner;
pub use token::{Token, TokenKind};

// ---------------------------------------------------------------------------
// Imports from crate::common
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::encoding;
use crate::common::string_interner::{Interner, Symbol};

// SourceMap is imported per the module interface contract — the Lexer's file_id
// field semantically references a SourceMap-managed file entry. The Lexer does
// not call SourceMap methods directly; file_id is passed through to Span
// constructors for downstream diagnostic resolution.
#[allow(unused_imports)]
use crate::common::source_map::SourceMap;

// ---------------------------------------------------------------------------
// Imports from lexer submodules
// ---------------------------------------------------------------------------

use token::{lookup_keyword, FloatSuffix, NumericBase, StringPrefix};

// ============================================================================
// Lexer — main tokenizer struct
// ============================================================================

/// The main lexer for BCC's Phase 3 tokenization.
///
/// Drives the tokenization loop over preprocessed C11 source, dispatching to
/// specialized functions for identifiers/keywords, numeric literals,
/// string/character literals, and operators/punctuators.
///
/// Provides a lookahead buffer for the parser's `peek()` / `unget()` needs.
/// The lexer is designed for single-pass, forward-only scanning with bounded
/// lookahead — it never backtracks over source characters.
///
/// # Lifetime
///
/// The `'src` lifetime ties the lexer to the source string and the borrowed
/// [`Interner`] and [`DiagnosticEngine`] instances.
///
/// # Example
///
/// ```ignore
/// use bcc::common::diagnostics::DiagnosticEngine;
/// use bcc::common::string_interner::Interner;
/// use bcc::frontend::lexer::Lexer;
///
/// let source = "int main(void) { return 0; }";
/// let mut interner = Interner::new();
/// let mut diags = DiagnosticEngine::new();
/// let mut lexer = Lexer::new(source, 0, &mut interner, &mut diags);
///
/// let tokens = lexer.tokenize_all();
/// // tokens: [Int, Identifier("main"), LeftParen, Void, RightParen,
/// //          LeftBrace, Return, IntegerLiteral{0}, Semicolon, RightBrace, Eof]
/// ```
pub struct Lexer<'src> {
    /// Character-level PUA-aware scanner for input consumption and position
    /// tracking. Provides advance/peek/lookahead over the source characters.
    scanner: Scanner<'src>,

    /// String interner for efficient identifier storage and O(1) comparison.
    /// Every scanned identifier is interned to produce a [`Symbol`] handle.
    interner: &'src mut Interner,

    /// Diagnostic engine for error/warning reporting without panicking.
    /// Lexer errors (invalid characters, unterminated comments) are emitted
    /// here for multi-error accumulation.
    diagnostics: &'src mut DiagnosticEngine,

    /// Source file ID for [`Span`] construction. This ID references a file
    /// entry managed by [`SourceMap`] and is embedded in every token's span.
    file_id: u32,

    /// Lookahead buffer for parser peek/unget support. Tokens are stored in
    /// order: index 0 is the next token to be consumed by [`next_token`].
    ///
    /// [`next_token`]: Lexer::next_token
    lookahead: Vec<Token>,
}

impl<'src> Lexer<'src> {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Create a new lexer for the given preprocessed source string.
    ///
    /// # Arguments
    ///
    /// * `source` — Preprocessed C11 source (may contain PUA code points for
    ///   non-UTF-8 bytes encoded by the preprocessor's source file reader).
    /// * `file_id` — Source file ID from [`SourceMap::add_file()`].
    /// * `interner` — String interner for identifier deduplication.
    /// * `diagnostics` — Diagnostic engine for error/warning emission.
    pub fn new(
        source: &'src str,
        file_id: u32,
        interner: &'src mut Interner,
        diagnostics: &'src mut DiagnosticEngine,
    ) -> Self {
        Lexer {
            scanner: Scanner::new(source),
            interner,
            diagnostics,
            file_id,
            lookahead: Vec::new(),
        }
    }

    // ===================================================================
    // Public API — token stream interface
    // ===================================================================

    /// Consume and return the next token from the input.
    ///
    /// If the lookahead buffer is non-empty, the front token is removed and
    /// returned. Otherwise, a fresh token is lexed from the scanner.
    ///
    /// Returns a token with [`TokenKind::Eof`] at end-of-input. Repeated
    /// calls after EOF continue to return `Eof` tokens.
    pub fn next_token(&mut self) -> Token {
        if !self.lookahead.is_empty() {
            return self.lookahead.remove(0);
        }
        let token = self.lex_token();
        // Span integrity: start ≤ end and file_id matches this lexer's file.
        debug_assert!(
            token.span.start <= token.span.end,
            "invalid span: start {} > end {} for token {:?}",
            token.span.start,
            token.span.end,
            token.kind
        );
        debug_assert_eq!(
            token.span.file_id, self.file_id,
            "span file_id {} does not match lexer file_id {}",
            token.span.file_id, self.file_id,
        );
        token
    }

    /// Peek at the next token without consuming it.
    ///
    /// If the lookahead buffer is empty, a token is lexed and buffered.
    /// Subsequent calls to `peek()` return the same token until it is
    /// consumed via [`next_token()`](Lexer::next_token).
    pub fn peek(&mut self) -> &Token {
        if self.lookahead.is_empty() {
            let token = self.lex_token();
            self.lookahead.push(token);
        }
        &self.lookahead[0]
    }

    /// Peek at the `n`-th token ahead (0-indexed) without consuming.
    ///
    /// `peek_nth(0)` is equivalent to [`peek()`](Lexer::peek). `peek_nth(1)`
    /// returns the token after the next one, etc. Fills the lookahead buffer
    /// as needed by lexing additional tokens.
    pub fn peek_nth(&mut self, n: usize) -> &Token {
        while self.lookahead.len() <= n {
            let token = self.lex_token();
            self.lookahead.push(token);
        }
        &self.lookahead[n]
    }

    /// Push a token back to the front of the lookahead buffer.
    ///
    /// The pushed token will be the next one returned by
    /// [`next_token()`](Lexer::next_token) or observed by
    /// [`peek()`](Lexer::peek).
    pub fn unget(&mut self, token: Token) {
        self.lookahead.insert(0, token);
    }

    /// Consume all remaining tokens until EOF and return them as a `Vec`.
    ///
    /// The returned vector includes the final [`TokenKind::Eof`] token.
    /// Useful for testing and batch processing.
    pub fn tokenize_all(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }

    // ===================================================================
    // Core lexing — main dispatch
    // ===================================================================

    /// Lex a single token from the scanner.
    ///
    /// Skips whitespace and comments, then dispatches to the appropriate
    /// sub-lexer based on the next character. Uses greedy (longest-match)
    /// operator disambiguation for multi-character operators.
    fn lex_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let start = self.scanner.offset();

        // Check for EOF after skipping whitespace/comments.
        if self.scanner.is_eof() {
            return self.make_token(TokenKind::Eof, start);
        }

        let ch = match self.scanner.peek() {
            Some(c) => c,
            None => return self.make_token(TokenKind::Eof, start),
        };

        match ch {
            // =============================================================
            // Identifiers, keywords, and prefixed string/char literals
            // =============================================================
            'a'..='z' | 'A'..='Z' | '_' => {
                // Before identifier lexing, check for string/char literal
                // prefixes: L"...", L'...', u"...", u'...', u8"...", U"...", U'...'
                if matches!(ch, 'L' | 'u' | 'U') {
                    if let Some((prefix, is_string)) =
                        string_literal::detect_prefix(&mut self.scanner)
                    {
                        // detect_prefix consumed the prefix chars; scanner is
                        // now positioned at the opening quote character.
                        let kind = if is_string {
                            string_literal::lex_string_literal(
                                &mut self.scanner,
                                prefix,
                                self.diagnostics,
                                self.file_id,
                            )
                        } else {
                            string_literal::lex_char_literal(
                                &mut self.scanner,
                                prefix,
                                self.diagnostics,
                                self.file_id,
                            )
                        };
                        return self.make_token(kind, start);
                    }
                    // detect_prefix returned None — not a prefixed literal.
                    // Fall through to standard identifier/keyword lexing.
                }
                let kind = self.lex_identifier_or_keyword();
                self.make_token(kind, start)
            }

            // =============================================================
            // Numeric literals (starting with a digit: 0-9)
            // =============================================================
            '0'..='9' => {
                let kind =
                    number_literal::lex_number(&mut self.scanner, self.diagnostics, self.file_id);
                self.make_token(kind, start)
            }

            // =============================================================
            // String literals (no prefix)
            // =============================================================
            '"' => {
                let kind = string_literal::lex_string_literal(
                    &mut self.scanner,
                    StringPrefix::None,
                    self.diagnostics,
                    self.file_id,
                );
                self.make_token(kind, start)
            }

            // =============================================================
            // Character literals (no prefix)
            // =============================================================
            '\'' => {
                let kind = string_literal::lex_char_literal(
                    &mut self.scanner,
                    StringPrefix::None,
                    self.diagnostics,
                    self.file_id,
                );
                self.make_token(kind, start)
            }

            // =============================================================
            // Operators and punctuators — greedy (longest-match) dispatch
            // =============================================================

            // --- Arithmetic / increment / compound assignment ---
            '+' => self.lex_after_plus(start),
            '-' => self.lex_after_minus(start),
            '*' => self.lex_after_star(start),
            '/' => self.lex_after_slash(start),
            '%' => self.lex_after_percent(start),

            // --- Bitwise / logical / compound assignment ---
            '&' => self.lex_after_ampersand(start),
            '|' => self.lex_after_pipe(start),
            '^' => self.lex_after_caret(start),
            '~' => {
                self.scanner.advance();
                self.make_token(TokenKind::Tilde, start)
            }

            // --- Comparison / shift / assignment ---
            '!' => self.lex_after_bang(start),
            '=' => self.lex_after_equal(start),
            '<' => self.lex_after_less(start),
            '>' => self.lex_after_greater(start),

            // --- Dot / ellipsis / float-starting-with-dot ---
            '.' => self.lex_after_dot(start),

            // --- Simple punctuators ---
            ',' => {
                self.scanner.advance();
                self.make_token(TokenKind::Comma, start)
            }
            ';' => {
                self.scanner.advance();
                self.make_token(TokenKind::Semicolon, start)
            }
            ':' => {
                self.scanner.advance();
                self.make_token(TokenKind::Colon, start)
            }
            '?' => {
                self.scanner.advance();
                self.make_token(TokenKind::Question, start)
            }

            // --- Delimiters ---
            '(' => {
                self.scanner.advance();
                self.make_token(TokenKind::LeftParen, start)
            }
            ')' => {
                self.scanner.advance();
                self.make_token(TokenKind::RightParen, start)
            }
            '[' => {
                self.scanner.advance();
                self.make_token(TokenKind::LeftBracket, start)
            }
            ']' => {
                self.scanner.advance();
                self.make_token(TokenKind::RightBracket, start)
            }
            '{' => {
                self.scanner.advance();
                self.make_token(TokenKind::LeftBrace, start)
            }
            '}' => {
                self.scanner.advance();
                self.make_token(TokenKind::RightBrace, start)
            }

            // --- Preprocessor hash (may appear for robustness) ---
            '#' => {
                self.scanner.advance();
                let kind = if self.scanner.consume_if('#') {
                    TokenKind::HashHash
                } else {
                    TokenKind::Hash
                };
                self.make_token(kind, start)
            }

            // =============================================================
            // Unknown / error characters
            // =============================================================
            _ => self.lex_error_char(ch, start),
        }
    }

    // ===================================================================
    // Operator disambiguation helpers (greedy / longest-match)
    // ===================================================================

    /// `+` → `+`, `++`, `+=`
    fn lex_after_plus(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('+') {
            TokenKind::PlusPlus
        } else if self.scanner.consume_if('=') {
            TokenKind::PlusEqual
        } else {
            TokenKind::Plus
        };
        self.make_token(kind, start)
    }

    /// `-` → `-`, `--`, `-=`, `->`
    fn lex_after_minus(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('>') {
            TokenKind::Arrow
        } else if self.scanner.consume_if('-') {
            TokenKind::MinusMinus
        } else if self.scanner.consume_if('=') {
            TokenKind::MinusEqual
        } else {
            TokenKind::Minus
        };
        self.make_token(kind, start)
    }

    /// `*` → `*`, `*=`
    fn lex_after_star(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::StarEqual
        } else {
            TokenKind::Star
        };
        self.make_token(kind, start)
    }

    /// `/` → `/`, `/=`
    ///
    /// Note: `//` and `/*` comments are handled by
    /// [`skip_whitespace_and_comments`] before this point.
    fn lex_after_slash(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::SlashEqual
        } else {
            TokenKind::Slash
        };
        self.make_token(kind, start)
    }

    /// `%` → `%`, `%=`
    fn lex_after_percent(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::PercentEqual
        } else {
            TokenKind::Percent
        };
        self.make_token(kind, start)
    }

    /// `&` → `&`, `&&`, `&=`
    fn lex_after_ampersand(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('&') {
            TokenKind::AmpAmp
        } else if self.scanner.consume_if('=') {
            TokenKind::AmpEqual
        } else {
            TokenKind::Ampersand
        };
        self.make_token(kind, start)
    }

    /// `|` → `|`, `||`, `|=`
    fn lex_after_pipe(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('|') {
            TokenKind::PipePipe
        } else if self.scanner.consume_if('=') {
            TokenKind::PipeEqual
        } else {
            TokenKind::Pipe
        };
        self.make_token(kind, start)
    }

    /// `^` → `^`, `^=`
    fn lex_after_caret(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::CaretEqual
        } else {
            TokenKind::Caret
        };
        self.make_token(kind, start)
    }

    /// `!` → `!`, `!=`
    fn lex_after_bang(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::BangEqual
        } else {
            TokenKind::Bang
        };
        self.make_token(kind, start)
    }

    /// `=` → `=`, `==`
    fn lex_after_equal(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('=') {
            TokenKind::EqualEqual
        } else {
            TokenKind::Equal
        };
        self.make_token(kind, start)
    }

    /// `<` → `<`, `<=`, `<<`, `<<=`
    fn lex_after_less(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('<') {
            // `<<` or `<<=`
            if self.scanner.consume_if('=') {
                TokenKind::LessLessEqual
            } else {
                TokenKind::LessLess
            }
        } else if self.scanner.consume_if('=') {
            TokenKind::LessEqual
        } else {
            TokenKind::Less
        };
        self.make_token(kind, start)
    }

    /// `>` → `>`, `>=`, `>>`, `>>=`
    fn lex_after_greater(&mut self, start: u32) -> Token {
        self.scanner.advance();
        let kind = if self.scanner.consume_if('>') {
            // `>>` or `>>=`
            if self.scanner.consume_if('=') {
                TokenKind::GreaterGreaterEqual
            } else {
                TokenKind::GreaterGreater
            }
        } else if self.scanner.consume_if('=') {
            TokenKind::GreaterEqual
        } else {
            TokenKind::Greater
        };
        self.make_token(kind, start)
    }

    /// `.` → `.`, `...`, or a decimal float literal starting with `.`
    ///
    /// Uses peek-based disambiguation to avoid consuming characters that
    /// would need to be ungotten:
    /// - `.5`, `.5e10` — decimal float literal
    /// - `...` — ellipsis
    /// - `.` — member access dot
    fn lex_after_dot(&mut self, start: u32) -> Token {
        // Check for float literal starting with '.' followed by digit.
        if matches!(self.scanner.peek_nth(1), Some('0'..='9')) {
            let kind = self.lex_dot_float(start);
            return self.make_token(kind, start);
        }

        // Check for ellipsis '...'
        if self.scanner.peek_nth(1) == Some('.') && self.scanner.peek_nth(2) == Some('.') {
            self.scanner.advance(); // first '.'
            self.scanner.advance(); // second '.'
            self.scanner.advance(); // third '.'
            return self.make_token(TokenKind::Ellipsis, start);
        }

        // Single dot — member access operator.
        self.scanner.advance();
        self.make_token(TokenKind::Dot, start)
    }

    // ===================================================================
    // Error character handling
    // ===================================================================

    /// Handle an unknown/unexpected character by emitting a diagnostic and
    /// returning an [`TokenKind::Error`] token.
    ///
    /// Special handling for:
    /// - **PUA code points** (U+E080–U+E0FF): Identified as non-UTF-8 bytes
    ///   outside of string/character literal context.
    /// - **Null byte** (`\0`): Reported with a specific message.
    /// - **Other characters**: Reported with the character value.
    ///
    /// The offending character is consumed and lexing continues (multi-error
    /// accumulation — the lexer never panics or aborts).
    fn lex_error_char(&mut self, ch: char, start: u32) -> Token {
        self.scanner.advance();
        let span = Span::new(self.file_id, start, self.scanner.offset());

        if encoding::is_pua_encoded(ch) {
            // PUA code point outside string/char literal context — likely a
            // non-UTF-8 byte that leaked from binary data in source.
            if let Some(byte_val) = encoding::decode_pua_to_byte(ch) {
                self.diagnostics.emit_error(
                    span,
                    format!(
                        "non-UTF-8 byte 0x{:02X} outside of string or character literal",
                        byte_val
                    ),
                );
            } else {
                self.diagnostics
                    .emit_error(span, format!("unexpected character U+{:04X}", ch as u32));
            }
        } else if ch == '\0' {
            self.diagnostics
                .emit_error(span, "unexpected null character in source");
        } else {
            self.diagnostics
                .emit_error(span, format!("unexpected character '{}'", ch));
        }

        Token::new(TokenKind::Error, span)
    }

    // ===================================================================
    // Whitespace and comment skipping
    // ===================================================================

    /// Skip whitespace characters and C-style comments.
    ///
    /// Consumes in a loop until the next non-whitespace, non-comment
    /// character (or EOF) is reached:
    ///
    /// - **Whitespace**: space, tab, newline, carriage return, form feed (`\x0C`),
    ///   vertical tab (`\x0B`).
    /// - **Line comments**: `//` through end of line (newline not consumed — it
    ///   is treated as whitespace on the next iteration).
    /// - **Block comments**: `/* ... */`. Unterminated block comments (EOF
    ///   before `*/`) are diagnosed as errors.
    ///
    /// The scanner's line/column tracking handles newlines automatically.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.scanner.peek() {
                // Standard C whitespace characters.
                Some(ch) if is_c_whitespace(ch) => {
                    self.scanner.advance();
                }
                // Potential comment start: `/` followed by `/` or `*`.
                Some('/') => {
                    match self.scanner.peek_nth(1) {
                        // Line comment: `// ... \n`
                        Some('/') => {
                            self.scanner.advance(); // consume first `/`
                            self.scanner.advance(); // consume second `/`
                                                    // Skip to end of line or EOF.
                            loop {
                                match self.scanner.peek() {
                                    None | Some('\n') => break,
                                    _ => {
                                        self.scanner.advance();
                                    }
                                }
                            }
                            // The newline (if present) is NOT consumed here; it
                            // will be handled as whitespace on the next outer
                            // loop iteration.
                        }
                        // Block comment: `/* ... */`
                        Some('*') => {
                            let comment_start = self.scanner.offset();
                            self.scanner.advance(); // consume `/`
                            self.scanner.advance(); // consume `*`
                            let mut terminated = false;
                            loop {
                                match self.scanner.advance() {
                                    None => break, // EOF — unterminated
                                    Some('/') => {
                                        // Warn on nested `/*` inside block
                                        // comment (GCC -Wcomment equivalent).
                                        if self.scanner.peek() == Some('*') {
                                            let nested_start = self.scanner.offset() - 1;
                                            let nested_span = Span::new(
                                                self.file_id,
                                                nested_start,
                                                self.scanner.offset(),
                                            );
                                            self.diagnostics.emit_warning(
                                                nested_span,
                                                "'/*' within block comment",
                                            );
                                        }
                                    }
                                    Some('*') => {
                                        if self.scanner.consume_if('/') {
                                            terminated = true;
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            if !terminated {
                                let span =
                                    Span::new(self.file_id, comment_start, self.scanner.offset());
                                self.diagnostics
                                    .emit_error(span, "unterminated block comment");
                                return;
                            }
                        }
                        // Not a comment start — stop skipping. The `/` will be
                        // handled as a division operator by lex_token.
                        _ => break,
                    }
                }
                // Not whitespace or comment — stop skipping.
                _ => break,
            }
        }
    }

    // ===================================================================
    // Identifier and keyword lexing
    // ===================================================================

    /// Lex an identifier or keyword token.
    ///
    /// Consumes `[a-zA-Z_][a-zA-Z0-9_]*` from the scanner, checks the result
    /// against the keyword table ([`lookup_keyword`]), and either returns the
    /// keyword [`TokenKind`] variant or interns the string as an
    /// [`TokenKind::Identifier`] with a [`Symbol`] handle.
    ///
    /// All C11 keywords (44) and GCC extension keywords/builtins (~40) are
    /// recognized here through `lookup_keyword`, ensuring they produce proper
    /// `TokenKind` variants rather than plain `Identifier` tokens.
    fn lex_identifier_or_keyword(&mut self) -> TokenKind {
        let start = self.scanner.offset() as usize;

        // Consume the first character (already verified: ASCII letter or `_`).
        self.scanner.advance();

        // Consume remaining identifier characters: [a-zA-Z0-9_]
        // Uses `skip_while` with a predicate for efficient bulk consumption.
        self.scanner
            .skip_while(|ch| ch.is_ascii_alphanumeric() || ch == '_');

        let end = self.scanner.offset() as usize;
        let text = self.scanner.slice(start, end);

        // Check the keyword table first (C11 keywords + GCC extension keywords
        // + GCC builtins). This dispatches to the comprehensive match in
        // token::lookup_keyword which covers all ~80 keyword strings.
        if let Some(keyword_kind) = lookup_keyword(text) {
            return keyword_kind;
        }

        // Not a keyword — intern the identifier string for O(1) comparison
        // via the Symbol handle throughout the parser and semantic analyzer.
        let symbol: Symbol = self.interner.intern(text);

        // Defensive integrity check: ensure the symbol index is valid.
        debug_assert!(
            symbol.as_u32() < u32::MAX,
            "string interner overflow: symbol index {} for identifier '{}'",
            symbol.as_u32(),
            text
        );

        TokenKind::Identifier(symbol)
    }

    // ===================================================================
    // Float literal starting with '.'
    // ===================================================================

    /// Lex a decimal float literal starting with `.` (e.g., `.5`, `.5e10f`).
    ///
    /// Called when the scanner is positioned at `.` and the next character
    /// after the dot is a digit. The `.` has NOT been consumed yet.
    ///
    /// Handles optional exponent (`e`/`E` with optional sign and digits) and
    /// optional float suffix (`f`/`F` for float, `l`/`L` for long double).
    fn lex_dot_float(&mut self, start: u32) -> TokenKind {
        // Consume the leading `.`
        self.scanner.advance();

        // Consume fractional digits (at least one, guaranteed by caller check).
        while matches!(self.scanner.peek(), Some('0'..='9')) {
            self.scanner.advance();
        }

        // Check for decimal exponent: e/E [+/-] digits
        if matches!(self.scanner.peek(), Some('e') | Some('E')) {
            self.scanner.advance(); // consume e/E

            // Optional sign.
            if matches!(self.scanner.peek(), Some('+') | Some('-')) {
                self.scanner.advance();
            }

            // Exponent digits (at least one required).
            let exp_digit_start = self.scanner.offset();
            while matches!(self.scanner.peek(), Some('0'..='9')) {
                self.scanner.advance();
            }
            if self.scanner.offset() == exp_digit_start {
                let span = Span::new(self.file_id, start, self.scanner.offset());
                self.diagnostics
                    .emit_error(span, "exponent has no digits in floating constant");
            }
        }

        // Record end of numeric value text (before any suffix character).
        let value_end = self.scanner.offset() as usize;

        // Float suffix: f/F → float, l/L → long double, i/j → imaginary, combos.
        let suffix = match self.scanner.peek() {
            Some('f') | Some('F') => {
                self.scanner.advance();
                match self.scanner.peek() {
                    Some('i') | Some('j') | Some('I') | Some('J') => {
                        self.scanner.advance();
                        FloatSuffix::FI
                    }
                    _ => FloatSuffix::F,
                }
            }
            Some('l') | Some('L') => {
                self.scanner.advance();
                match self.scanner.peek() {
                    Some('i') | Some('j') | Some('I') | Some('J') => {
                        self.scanner.advance();
                        FloatSuffix::LI
                    }
                    _ => FloatSuffix::L,
                }
            }
            Some('i') | Some('j') | Some('I') | Some('J') => {
                self.scanner.advance();
                match self.scanner.peek() {
                    Some('f') | Some('F') => {
                        self.scanner.advance();
                        FloatSuffix::FI
                    }
                    Some('l') | Some('L') => {
                        self.scanner.advance();
                        FloatSuffix::LI
                    }
                    _ => FloatSuffix::I,
                }
            }
            _ => FloatSuffix::None,
        };

        // Extract the raw numeric text (excluding the suffix character).
        let value = self.scanner.slice(start as usize, value_end).to_string();

        TokenKind::FloatLiteral {
            value,
            suffix,
            base: NumericBase::Decimal,
        }
    }

    // ===================================================================
    // Token construction helper
    // ===================================================================

    /// Construct a [`Token`] with a [`Span`] from `start` to the current
    /// scanner byte offset.
    ///
    /// Uses `self.file_id` and `self.scanner.offset()` to build the span's
    /// `[start, end)` half-open byte range.
    #[inline]
    fn make_token(&self, kind: TokenKind, start: u32) -> Token {
        Token::new(kind, Span::new(self.file_id, start, self.scanner.offset()))
    }

    /// Return the current scanner position (offset, line, column).
    ///
    /// Useful for callers (e.g. the preprocessor) that need to snapshot
    /// location state before attempting speculative lexing.
    #[inline]
    pub fn current_position(&self) -> (u32, u32, u32) {
        let pos = self.scanner.position();
        (pos.offset, pos.line, pos.column)
    }

    /// Return a reference to the full source string being lexed.
    ///
    /// Useful for error‐message formatting and test utilities that need the
    /// raw source text.
    #[inline]
    pub fn source(&self) -> &'src str {
        self.scanner.source()
    }

    // ===================================================================
    // Accessor methods for parser integration
    // ===================================================================

    /// Return a shared reference to the diagnostic engine.
    ///
    /// Used by the parser to query error state (e.g., [`DiagnosticEngine::has_errors`]).
    #[inline]
    pub fn diagnostics(&self) -> &DiagnosticEngine {
        self.diagnostics
    }

    /// Return a mutable reference to the diagnostic engine.
    ///
    /// Used by the parser to emit diagnostic errors and warnings during
    /// parsing (e.g., syntax errors, recursion overflow).
    #[inline]
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticEngine {
        self.diagnostics
    }

    /// Return a shared reference to the string interner.
    ///
    /// Used by the parser to resolve interned [`Symbol`] handles back to
    /// their string representation.
    #[inline]
    pub fn interner(&self) -> &Interner {
        self.interner
    }

    /// Return a mutable reference to the string interner.
    ///
    /// Used by the parser to intern new strings (e.g., synthesized
    /// identifiers) during parsing.
    #[inline]
    pub fn interner_mut(&mut self) -> &mut Interner {
        self.interner
    }

    /// Attempt to consume a single identifier-continuation character.
    ///
    /// Returns `true` if the next character is `[a-zA-Z0-9_]` and was
    /// consumed; `false` otherwise (scanner unchanged). Uses
    /// [`Scanner::consume_if_pred`] for predicate-based conditional advance.
    #[allow(dead_code)]
    fn try_consume_ident_char(&mut self) -> bool {
        self.scanner
            .consume_if_pred(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            .is_some()
    }
}

// ============================================================================
// Module-level helper functions
// ============================================================================

/// Check if a character is C whitespace.
///
/// The C standard defines whitespace as: space (`' '`), horizontal tab
/// (`'\t'`), newline (`'\n'`), carriage return (`'\r'`), form feed
/// (`'\x0C'`), and vertical tab (`'\x0B'`).
///
/// Note: The scanner normalizes `\r\n` and standalone `\r` to `\n`, so
/// `'\r'` is included here for robustness but may not be observed in
/// practice after scanner normalization.
#[inline]
fn is_c_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\n' | '\r' | '\x0C' | '\x0B')
}
