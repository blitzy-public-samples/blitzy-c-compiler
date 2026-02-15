//! Parser module — Phase 4 of the BCC compilation pipeline.
//!
//! Recursive-descent C11 parser with comprehensive GCC extension support.
//! Converts a token stream produced by the lexer (Phase 3) into an Abstract
//! Syntax Tree (AST) consumed by the semantic analyzer (Phase 5).
//!
//! # Architecture
//!
//! The parser is implemented as a struct [`Parser`] that holds a borrowed
//! slice of [`Token`]s and a cursor position. Parsing methods are split across
//! submodules by syntactic category:
//!
//! - [`ast`] — AST node type definitions
//! - [`declarations`] — declaration parsing (variables, functions, typedefs,
//!   structs, unions, enums, `_Static_assert`)
//! - [`expressions`] — expression parsing (Pratt / precedence-climbing)
//! - [`statements`] — statement parsing (control flow, labels, compounds)
//! - [`gcc_extensions`] — GCC language extension dispatch (statement
//!   expressions, computed gotos, case ranges, local labels, etc.)
//! - [`attributes`] — `__attribute__((...))` parsing
//! - [`inline_asm`] — `asm` / `__asm__` statement parsing
//! - [`types`] — type specifier and qualifier parsing
//!
//! # Resource Constraints
//!
//! A configurable recursion depth limit (default 512, per §0.7.3) is enforced
//! to prevent stack overflow on deeply nested kernel macro expansions. The
//! limit is checked via [`Parser::enter_recursion`].
//!
//! # Typedef Disambiguation
//!
//! C's syntax makes declarations and expressions ambiguous when typedef names
//! are involved. The parser maintains a set of known typedef names
//! ([`Parser::typedef_names`]) that is populated as `typedef` declarations
//! are parsed, enabling [`Parser::is_declaration_start`] to resolve the
//! ambiguity.

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// AST node definitions — the foundational types for the entire parser,
/// semantic analyzer, and IR lowering phases.
pub mod ast;

/// Declaration parsing — variables, functions, typedefs, struct/union/enum,
/// `_Static_assert`, storage class specifiers, alignment specifiers.
pub mod declarations;

/// Expression parsing — Pratt / precedence-climbing for all C11 operators,
/// primary expressions, casts, sizeof, `_Generic`, compound literals, and
/// GCC statement expressions.
pub mod expressions;

/// Statement parsing — compound statements, control flow (if/else, while,
/// do-while, for, switch/case), goto (including computed), break, continue,
/// return, labels, inline assembly dispatch.
pub mod statements;

/// GCC language extension dispatch — statement expressions, computed gotos,
/// case ranges, zero-length arrays, `__extension__`, transparent unions,
/// local labels, conditional omission.
pub mod gcc_extensions;

/// `__attribute__((...))` parsing for all required GCC attributes.
pub mod attributes;

/// Inline assembly (`asm` / `__asm__`) statement parsing.
pub mod inline_asm;

/// Type specifier and qualifier parsing (`typeof`, `_Atomic`, etc.).
pub mod types;

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::FxHashSet;
use crate::common::source_map::SourceMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::frontend::lexer::token::{Token, TokenKind};

// Re-export all AST types for convenience access by consumers (sema, IR lowering).
// NOTE: This includes `Span` (re-exported by ast.rs from token.rs from diagnostics.rs),
// `TranslationUnit`, `Declaration`, `Expression`, `Statement`, and all other AST nodes.
pub use ast::*;

// ===========================================================================
// ParseError
// ===========================================================================

/// A parse error with source location, human-readable message, and an
/// optional description of what was expected.
///
/// `ParseError` is the error type returned by all fallible parser functions.
/// It carries enough context for the diagnostic engine to produce a rich
/// error message.
#[derive(Clone, Debug)]
pub struct ParseError {
    /// Source location where the error was detected.
    pub span: Span,
    /// Human-readable description of the problem.
    pub message: String,
    /// Optional description of what was expected (e.g., `"identifier"`,
    /// `"';'"`) to improve the diagnostic message.
    pub expected: Option<String>,
}

/// Convenience alias for parser result types.
pub type ParseResult<T> = Result<T, ParseError>;

// ===========================================================================
// Parser
// ===========================================================================

/// Recursive-descent C11 parser with GCC extension support.
///
/// Holds a borrowed token stream and a cursor position, advancing through
/// tokens as syntactic constructs are recognised. Error recovery is performed
/// by skipping to synchronisation points (`;`, `{`, `}`, or the start of a
/// new declaration keyword).
///
/// # Lifetime `'a`
///
/// All borrowed data (tokens, diagnostic engine, source map, interner, target)
/// live at least as long as the parser.
pub struct Parser<'a> {
    /// The full token stream produced by the lexer.
    tokens: &'a [Token],

    /// Current position (index) in the token stream.
    pos: usize,

    /// Diagnostic reporting engine — errors, warnings, notes are emitted here.
    pub diagnostics: &'a mut DiagnosticEngine,

    /// Source file tracking — used for location resolution in diagnostics.
    #[allow(dead_code)]
    pub source_map: &'a SourceMap,

    /// String interner — for interning new identifiers/strings during parsing.
    pub interner: &'a mut Interner,

    /// Target architecture — for architecture-dependent size/alignment queries.
    #[allow(dead_code)]
    pub target: &'a Target,

    /// Set of known typedef names, populated as `typedef` declarations are
    /// parsed. Used by [`is_declaration_start`] for disambiguation.
    typedef_names: FxHashSet<Symbol>,

    /// Current parser recursion depth (incremented on each recursive call).
    recursion_depth: u32,

    /// Maximum allowed recursion depth (default 512 per §0.7.3).
    max_recursion_depth: u32,
}

impl<'a> Parser<'a> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new parser over the given token stream.
    ///
    /// # Arguments
    ///
    /// * `tokens` — The complete token slice from the lexer.
    /// * `diagnostics` — Mutable reference to the diagnostic engine.
    /// * `source_map` — Source file tracking.
    /// * `interner` — String interner for identifier deduplication.
    /// * `target` — Target architecture descriptor.
    pub fn new(
        tokens: &'a [Token],
        diagnostics: &'a mut DiagnosticEngine,
        source_map: &'a SourceMap,
        interner: &'a mut Interner,
        target: &'a Target,
    ) -> Self {
        Parser {
            tokens,
            pos: 0,
            diagnostics,
            source_map,
            interner,
            target,
            typedef_names: FxHashSet::default(),
            recursion_depth: 0,
            max_recursion_depth: 512,
        }
    }

    // -----------------------------------------------------------------------
    // Token stream navigation
    // -----------------------------------------------------------------------

    /// Returns a reference to the current token without consuming it.
    ///
    /// If the cursor is past the end of the token stream, returns a synthetic
    /// `Eof` token so callers never see an out-of-bounds access.
    #[inline]
    pub fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| {
            // The lexer always emits a trailing Eof token, so in practice
            // this branch is only hit if the token stream is empty.
            static EOF_TOKEN: std::sync::OnceLock<Token> = std::sync::OnceLock::new();
            EOF_TOKEN.get_or_init(Token::eof)
        })
    }

    /// Alias for [`current`] — returns the current token without consuming it.
    #[inline]
    pub fn peek(&self) -> &Token {
        self.current()
    }

    /// Returns a reference to the token `n` positions ahead of the cursor
    /// without advancing. `peek_ahead(0)` is equivalent to `current()`.
    #[inline]
    pub fn peek_ahead(&self, n: usize) -> &Token {
        self.tokens.get(self.pos + n).unwrap_or_else(|| {
            static EOF_TOKEN: std::sync::OnceLock<Token> = std::sync::OnceLock::new();
            EOF_TOKEN.get_or_init(Token::eof)
        })
    }

    /// Consumes the current token and advances the cursor. Returns the span
    /// of the consumed token.
    ///
    /// If already at EOF, this is a no-op that returns `Span::DUMMY`.
    pub fn advance(&mut self) -> Span {
        let span = self.current().span;
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        span
    }

    /// Saves the current parser position for tentative (speculative) parsing.
    /// Returns a snapshot that can be passed to [`restore`] to rewind the
    /// cursor on failure.
    #[inline]
    pub fn save_position(&self) -> usize {
        self.pos
    }

    /// Restores the parser position to a previously saved snapshot,
    /// effectively rewinding the token cursor.
    #[inline]
    pub fn restore_position(&mut self, saved: usize) {
        self.pos = saved;
    }

    /// Consumes the current token if it matches `kind`. Returns `Ok(span)`
    /// on success, or `Err(ParseError)` with a diagnostic describing the
    /// mismatch.
    ///
    /// Token matching uses discriminant comparison so data-carrying variants
    /// match regardless of payload (e.g., any `Identifier` matches
    /// `TokenKind::Identifier(Symbol::EMPTY)`).
    pub fn expect(&mut self, kind: TokenKind) -> Result<Span, ParseError> {
        let tok = self.current();
        if tok.is(kind.clone()) {
            Ok(self.advance())
        } else {
            let span = tok.span;
            let msg = format!(
                "expected '{}', found '{}'",
                token_kind_display(&kind),
                token_kind_display(&tok.kind),
            );
            self.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: Some(token_kind_display(&kind).to_string()),
            })
        }
    }

    /// Convenience wrapper for `expect(TokenKind::Semicolon)` with a
    /// human-friendly error message mentioning the missing `;`.
    pub fn expect_semicolon(&mut self) -> Result<Span, ParseError> {
        self.expect(TokenKind::Semicolon)
    }

    /// Returns `true` if the current token's kind matches `kind` (without
    /// consuming). Uses discriminant comparison.
    #[inline]
    pub fn check(&self, kind: TokenKind) -> bool {
        self.current().is(kind)
    }

    /// If the current token matches `kind`, consume it and return `true`.
    /// Otherwise, return `false` without advancing.
    pub fn eat(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Returns `true` when the cursor is at EOF.
    #[inline]
    pub fn at_end(&self) -> bool {
        self.current().kind == TokenKind::Eof
    }

    // -----------------------------------------------------------------------
    // Recursion depth guard (§0.7.3)
    // -----------------------------------------------------------------------

    /// Increments the recursion depth and checks against the limit.
    ///
    /// Must be paired with [`leave_recursion`] in a scope guard or at
    /// the end of the recursive call.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` if the depth exceeds `max_recursion_depth` (512).
    pub fn enter_recursion(&mut self) -> Result<(), ParseError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            let span = self.current().span;
            let msg = format!(
                "parser recursion depth exceeded ({} limit)",
                self.max_recursion_depth
            );
            self.diagnostics.error(span, &msg);
            Err(ParseError {
                span,
                message: msg,
                expected: None,
            })
        } else {
            Ok(())
        }
    }

    /// Decrements the recursion depth. Must be called after every successful
    /// [`enter_recursion`] to keep the counter accurate.
    #[inline]
    pub fn leave_recursion(&mut self) {
        self.recursion_depth = self.recursion_depth.saturating_sub(1);
    }

    // -----------------------------------------------------------------------
    // Typedef name tracking
    // -----------------------------------------------------------------------

    /// Registers `name` as a known typedef name.
    ///
    /// After this call, `is_typedef_name(name)` returns `true`, allowing the
    /// parser to treat the identifier as a type name in declaration contexts.
    pub fn register_typedef(&mut self, name: Symbol) {
        self.typedef_names.insert(name);
    }

    /// Returns `true` if `name` has been registered as a typedef name.
    pub fn is_typedef_name(&self, name: Symbol) -> bool {
        self.typedef_names.contains(&name)
    }

    // -----------------------------------------------------------------------
    // Declaration-start detection (typedef disambiguation)
    // -----------------------------------------------------------------------

    /// Returns `true` if the current token can begin a declaration.
    ///
    /// A declaration starts with any of:
    /// - Storage class specifier (`auto`, `register`, `static`, `extern`,
    ///   `typedef`, `_Thread_local`)
    /// - Type specifier keyword (`void`, `char`, `short`, `int`, `long`,
    ///   `float`, `double`, `signed`, `unsigned`, `_Bool`, `_Complex`,
    ///   `struct`, `union`, `enum`)
    /// - Type qualifier (`const`, `volatile`, `restrict`, `_Atomic`)
    /// - Function specifier (`inline`, `_Noreturn`)
    /// - Alignment specifier (`_Alignas`)
    /// - `typeof` / `__typeof__`
    /// - `__attribute__`
    /// - `__extension__`
    /// - `_Static_assert`
    /// - A known typedef name (identifier registered via `register_typedef`)
    pub fn is_declaration_start(&self) -> bool {
        match &self.current().kind {
            // Storage class specifiers
            TokenKind::Auto
            | TokenKind::Register
            | TokenKind::Static
            | TokenKind::Extern
            | TokenKind::Typedef
            | TokenKind::ThreadLocal => true,

            // Type specifier keywords
            TokenKind::Void
            | TokenKind::Char
            | TokenKind::Short
            | TokenKind::Int
            | TokenKind::Long
            | TokenKind::Float
            | TokenKind::Double
            | TokenKind::Signed
            | TokenKind::Unsigned
            | TokenKind::Bool
            | TokenKind::Complex => true,

            // struct / union / enum
            TokenKind::Struct | TokenKind::Union | TokenKind::Enum => true,

            // Type qualifiers
            TokenKind::Const | TokenKind::Volatile | TokenKind::Restrict | TokenKind::Atomic => {
                true
            }

            // Function specifiers
            TokenKind::Inline | TokenKind::Noreturn => true,

            // Alignment specifier
            TokenKind::Alignas => true,

            // GCC extensions
            TokenKind::TypeofKeyword | TokenKind::Attribute | TokenKind::Extension => true,

            // __builtin_va_list — GCC's opaque variadic-argument list type
            TokenKind::BuiltinVaList => true,

            // _Static_assert
            TokenKind::StaticAssert => true,

            // Typedef name (identifier that was registered as a typedef)
            TokenKind::Identifier(sym) => self.typedef_names.contains(sym),

            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Error recovery
    // -----------------------------------------------------------------------

    /// Skips tokens until reaching a synchronisation point.
    ///
    /// Synchronisation points are: `;`, `}`, `{`, EOF, or any token that
    /// could start a new declaration. After calling this, the cursor is at
    /// the synchronisation token (not past it), so the caller can decide
    /// whether to consume it.
    pub fn synchronize(&mut self) {
        while !self.at_end() {
            match self.current().kind {
                TokenKind::Semicolon => {
                    // Consume the semicolon so the next parse attempt starts
                    // cleanly after it.
                    self.advance();
                    return;
                }
                TokenKind::RightBrace | TokenKind::LeftBrace => {
                    // Stop before the brace so the caller can handle it.
                    return;
                }
                _ if self.is_declaration_start() => {
                    // Stop before the declaration start token.
                    return;
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Skips tokens until finding a token of the given `kind`.
    /// Stops with the cursor on that token (does not consume it).
    pub fn skip_to(&mut self, kind: TokenKind) {
        while !self.at_end() && !self.check(kind.clone()) {
            self.advance();
        }
    }

    /// Skips a balanced `( ... )` group, including nested parentheses.
    /// The cursor must be on or past the opening `(`.
    pub fn skip_balanced_parens(&mut self) {
        let mut depth: u32 = 0;
        loop {
            if self.at_end() {
                break;
            }
            match self.current().kind {
                TokenKind::LeftParen => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::RightParen => {
                    if depth <= 1 {
                        self.advance(); // consume closing paren
                        break;
                    }
                    depth -= 1;
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Skips a balanced `{ ... }` group, including nested braces.
    /// The cursor must be on or past the opening `{`.
    pub fn skip_balanced_braces(&mut self) {
        let mut depth: u32 = 0;
        loop {
            if self.at_end() {
                break;
            }
            match self.current().kind {
                TokenKind::LeftBrace => {
                    depth += 1;
                    self.advance();
                }
                TokenKind::RightBrace => {
                    if depth <= 1 {
                        self.advance(); // consume closing brace
                        break;
                    }
                    depth -= 1;
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Top-level entry point
    // -----------------------------------------------------------------------

    /// Parses an entire translation unit (C source file).
    ///
    /// A translation unit is a sequence of external declarations. The parser
    /// loops until EOF, calling `declarations::parse_external_declaration` for
    /// each top-level construct.
    ///
    /// # Returns
    ///
    /// A `TranslationUnit` containing all parsed declarations, or a
    /// `ParseError` if an unrecoverable error is encountered.
    pub fn parse_translation_unit(&mut self) -> Result<ast::TranslationUnit, ParseError> {
        let start_span = self.current().span;
        let mut decls: Vec<ast::Declaration> = Vec::new();

        while !self.at_end() {
            match declarations::parse_external_declaration(self) {
                Ok(decl) => {
                    decls.push(decl);
                }
                Err(e) => {
                    self.diagnostics.error(e.span, &e.message);
                    self.synchronize();
                    // If we're still not at end, keep trying.
                    if self.at_end() {
                        break;
                    }
                }
            }
        }

        let end_span = self.current().span;
        let span = if decls.is_empty() {
            start_span
        } else {
            ast::Span::merge(start_span, end_span)
        };

        Ok(ast::TranslationUnit {
            declarations: decls,
            span,
        })
    }
}

// ===========================================================================
// Helper: token kind display for diagnostics
// ===========================================================================

/// Returns a human-readable string for a `TokenKind` variant, suitable for
/// use in error messages (e.g., `"expected ';'"`).
fn token_kind_display(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::Eof => "end of file",
        TokenKind::Semicolon => ";",
        TokenKind::Colon => ":",
        TokenKind::Comma => ",",
        TokenKind::LeftParen => "(",
        TokenKind::RightParen => ")",
        TokenKind::LeftBrace => "{",
        TokenKind::RightBrace => "}",
        TokenKind::LeftBracket => "[",
        TokenKind::RightBracket => "]",
        TokenKind::Star => "*",
        TokenKind::Assign => "=",
        TokenKind::Ellipsis => "...",
        TokenKind::Dot => ".",
        TokenKind::Arrow => "->",
        TokenKind::Question => "?",
        TokenKind::Plus => "+",
        TokenKind::Minus => "-",
        TokenKind::Slash => "/",
        TokenKind::Percent => "%",
        TokenKind::Ampersand => "&",
        TokenKind::Pipe => "|",
        TokenKind::Caret => "^",
        TokenKind::Tilde => "~",
        TokenKind::Exclaim => "!",
        TokenKind::Less => "<",
        TokenKind::Greater => ">",
        TokenKind::AmpAmp => "&&",
        TokenKind::PipePipe => "||",
        TokenKind::PlusPlus => "++",
        TokenKind::MinusMinus => "--",
        TokenKind::Identifier(_) => "identifier",
        TokenKind::IntegerLiteral { .. } => "integer literal",
        TokenKind::FloatLiteral { .. } => "float literal",
        TokenKind::StringLiteral { .. } => "string literal",
        TokenKind::CharLiteral { .. } => "character literal",
        // Keywords
        TokenKind::Auto => "auto",
        TokenKind::Break => "break",
        TokenKind::Case => "case",
        TokenKind::Char => "char",
        TokenKind::Const => "const",
        TokenKind::Continue => "continue",
        TokenKind::Default => "default",
        TokenKind::Do => "do",
        TokenKind::Double => "double",
        TokenKind::Else => "else",
        TokenKind::Enum => "enum",
        TokenKind::Extern => "extern",
        TokenKind::Float => "float",
        TokenKind::For => "for",
        TokenKind::Goto => "goto",
        TokenKind::If => "if",
        TokenKind::Inline => "inline",
        TokenKind::Int => "int",
        TokenKind::Long => "long",
        TokenKind::Register => "register",
        TokenKind::Restrict => "restrict",
        TokenKind::Return => "return",
        TokenKind::Short => "short",
        TokenKind::Signed => "signed",
        TokenKind::Sizeof => "sizeof",
        TokenKind::Static => "static",
        TokenKind::Struct => "struct",
        TokenKind::Switch => "switch",
        TokenKind::Typedef => "typedef",
        TokenKind::Union => "union",
        TokenKind::Unsigned => "unsigned",
        TokenKind::Void => "void",
        TokenKind::Volatile => "volatile",
        TokenKind::While => "while",
        // C11 special
        TokenKind::Alignas => "_Alignas",
        TokenKind::Alignof => "_Alignof",
        TokenKind::Atomic => "_Atomic",
        TokenKind::Bool => "_Bool",
        TokenKind::Complex => "_Complex",
        TokenKind::Generic => "_Generic",
        TokenKind::Noreturn => "_Noreturn",
        TokenKind::StaticAssert => "_Static_assert",
        TokenKind::ThreadLocal => "_Thread_local",
        // GCC extensions
        TokenKind::Attribute => "__attribute__",
        TokenKind::Extension => "__extension__",
        TokenKind::TypeofKeyword => "typeof",
        TokenKind::AsmKeyword => "asm",
        TokenKind::Label => "__label__",
        _ => "<token>",
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::source_map::SourceMap;
    use crate::common::string_interner::Interner;
    use crate::common::target::Target;

    /// Helper to create a parser over a given set of tokens.
    fn make_parser<'a>(
        tokens: &'a [Token],
        diag: &'a mut DiagnosticEngine,
        sm: &'a SourceMap,
        interner: &'a mut Interner,
        target: &'a Target,
    ) -> Parser<'a> {
        Parser::new(tokens, diag, sm, interner, target)
    }

    #[test]
    fn test_at_end_on_empty_stream() {
        let tokens = vec![Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);
        assert!(parser.at_end());
    }

    #[test]
    fn test_advance_and_current() {
        let span = Span::new(0, 0, 3);
        let tokens = vec![
            Token::new(TokenKind::Int, span),
            Token::new(TokenKind::Semicolon, Span::new(0, 4, 5)),
            Token::eof(),
        ];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        assert!(parser.check(TokenKind::Int));
        parser.advance();
        assert!(parser.check(TokenKind::Semicolon));
        parser.advance();
        assert!(parser.at_end());
    }

    #[test]
    fn test_eat_match() {
        let span = Span::new(0, 0, 1);
        let tokens = vec![Token::new(TokenKind::Semicolon, span), Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        assert!(parser.eat(TokenKind::Semicolon));
        assert!(parser.at_end());
    }

    #[test]
    fn test_eat_no_match() {
        let span = Span::new(0, 0, 1);
        let tokens = vec![Token::new(TokenKind::Colon, span), Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        assert!(!parser.eat(TokenKind::Semicolon));
        // Cursor should not have moved
        assert!(parser.check(TokenKind::Colon));
    }

    #[test]
    fn test_expect_success() {
        let span = Span::new(0, 0, 3);
        let tokens = vec![Token::new(TokenKind::Int, span), Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        let result = parser.expect(TokenKind::Int);
        assert!(result.is_ok());
        assert!(parser.at_end());
    }

    #[test]
    fn test_expect_failure() {
        let span = Span::new(0, 0, 3);
        let tokens = vec![Token::new(TokenKind::Int, span), Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        let result = parser.expect(TokenKind::Semicolon);
        assert!(result.is_err());
        // Cursor should not have moved on failure
        assert!(parser.check(TokenKind::Int));
    }

    #[test]
    fn test_recursion_depth() {
        let tokens = vec![Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        // Should succeed up to 512
        for _ in 0..512 {
            assert!(parser.enter_recursion().is_ok());
        }
        // 513th should fail
        assert!(parser.enter_recursion().is_err());
    }

    #[test]
    fn test_typedef_tracking() {
        let tokens = vec![Token::eof()];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        let sym = parser.interner.intern("size_t");
        assert!(!parser.is_typedef_name(sym));
        parser.register_typedef(sym);
        assert!(parser.is_typedef_name(sym));
    }

    #[test]
    fn test_is_declaration_start_keywords() {
        let test_cases = vec![
            TokenKind::Auto,
            TokenKind::Register,
            TokenKind::Static,
            TokenKind::Extern,
            TokenKind::Typedef,
            TokenKind::ThreadLocal,
            TokenKind::Void,
            TokenKind::Char,
            TokenKind::Short,
            TokenKind::Int,
            TokenKind::Long,
            TokenKind::Float,
            TokenKind::Double,
            TokenKind::Signed,
            TokenKind::Unsigned,
            TokenKind::Bool,
            TokenKind::Complex,
            TokenKind::Struct,
            TokenKind::Union,
            TokenKind::Enum,
            TokenKind::Const,
            TokenKind::Volatile,
            TokenKind::Restrict,
            TokenKind::Atomic,
            TokenKind::Inline,
            TokenKind::Noreturn,
            TokenKind::Alignas,
            TokenKind::TypeofKeyword,
            TokenKind::Attribute,
            TokenKind::Extension,
            TokenKind::StaticAssert,
        ];

        for kind in test_cases {
            let tokens = vec![Token::new(kind.clone(), Span::DUMMY), Token::eof()];
            let mut diag = DiagnosticEngine::new();
            let sm = SourceMap::new();
            let mut interner = Interner::new();
            let target = Target::X86_64;
            let parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);
            assert!(
                parser.is_declaration_start(),
                "Expected {:?} to be a declaration start",
                kind
            );
        }
    }

    #[test]
    fn test_synchronize_to_semicolon() {
        let tokens = vec![
            Token::new(TokenKind::Plus, Span::new(0, 0, 1)),
            Token::new(TokenKind::Minus, Span::new(0, 1, 2)),
            Token::new(TokenKind::Semicolon, Span::new(0, 2, 3)),
            Token::new(TokenKind::Int, Span::new(0, 3, 6)),
            Token::eof(),
        ];
        let mut diag = DiagnosticEngine::new();
        let sm = SourceMap::new();
        let mut interner = Interner::new();
        let target = Target::X86_64;
        let mut parser = make_parser(&tokens, &mut diag, &sm, &mut interner, &target);

        parser.synchronize();
        // After synchronize, we should be past the semicolon (at Int)
        assert!(parser.check(TokenKind::Int));
    }
}
