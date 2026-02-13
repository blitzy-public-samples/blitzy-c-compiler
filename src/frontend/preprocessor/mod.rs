//! # BCC Preprocessor — Phase 1 & Phase 2 Driver
//!
//! This module implements the C11 preprocessor for the BCC (Blitzy C Compiler)
//! pipeline. It is responsible for:
//!
//! - **Phase 1:** Trigraph replacement and line splicing (backslash-newline removal),
//!   converting physical source lines into logical source lines.
//! - **Phase 2:** Directive processing (`#include`, `#define`, `#undef`,
//!   `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, `#pragma`, `#error`,
//!   `#warning`, `#line`) and macro expansion orchestration.
//!
//! ## Architecture
//!
//! The preprocessor emits a fully macro-expanded token stream that the lexer/parser
//! consumes downstream. Source files are read through PUA-aware encoding
//! (`crate::common::encoding::read_source_file`) so that non-UTF-8 bytes survive the
//! entire pipeline with byte-exact fidelity.
//!
//! ## Recursion Protection
//!
//! Two independent recursion protection mechanisms are enforced:
//!
//! 1. **Paint-marker system** (`paint_marker` submodule): Operates at the token level
//!    during macro expansion. When a macro name token is produced by expanding that
//!    same macro, the token is *painted* and will not be re-expanded. This prevents
//!    infinite recursion on self-referential macros like `#define A A`.
//!
//! 2. **Depth counter** (`MAX_RECURSION_DEPTH = 512`): A hard limit on nested macro
//!    expansion and `#include` depth. If the counter reaches 512, a fatal diagnostic
//!    is emitted and preprocessing halts. This guards against pathological nesting in
//!    Linux kernel macro chains.
//!
//! ## Submodules
//!
//! | Module             | Responsibility                                    |
//! |--------------------|---------------------------------------------------|
//! | `directives`       | `#include`, `#define`, `#undef`, conditionals      |
//! | `macro_expander`   | Object-like & function-like macro expansion        |
//! | `paint_marker`     | Token paint state for recursion suppression         |
//! | `include_handler`  | `#include` file resolution, guards, circular check |
//! | `token_paster`     | `##` concatenation and `#` stringification          |
//! | `expression`       | `#if`/`#elif` constant expression evaluation        |
//! | `predefined`       | `__FILE__`, `__LINE__`, arch-specific defines       |

// ── Submodule declarations ──────────────────────────────────────────────────
pub mod directives;
pub mod expression;
pub mod include_handler;
pub mod macro_expander;
pub mod paint_marker;
pub mod predefined;
pub mod token_paster;

// ── Standard library imports ────────────────────────────────────────────────
use std::path::{Path, PathBuf};

// ── Internal crate imports ──────────────────────────────────────────────────
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::encoding::read_source_file;
use crate::common::fx_hash::{fx_hash_map, FxHashMap};
use crate::common::source_map::{FileId, SourceMap};
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;

// ── Sibling / child imports ─────────────────────────────────────────────────
use self::include_handler::IncludeHandler;
use self::paint_marker::{is_painted_for, paint_tokens, PaintedToken};
use self::token_paster::{paste_tokens, stringify};
use crate::frontend::lexer::token::{Token, TokenKind};

// ── Re-exports for ergonomic access by other modules ────────────────────────
pub use self::include_handler::IncludeKind;
pub use self::paint_marker::PaintState;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Hard limit on nested macro expansion / `#include` depth.
/// Matches the requirement from Section 0.7.3 — 512 levels protects against
/// pathological nesting found in Linux kernel macro chains.
pub const MAX_RECURSION_DEPTH: u32 = 512;

// ═══════════════════════════════════════════════════════════════════════════
// MacroDef — shared across directives.rs and macro_expander.rs
// ═══════════════════════════════════════════════════════════════════════════

/// Representation of a C preprocessor macro (`#define`).
///
/// This type is shared between the directive handler (which creates definitions)
/// and the macro expander (which uses them for replacement).
#[derive(Debug, Clone)]
pub struct MacroDef {
    /// Interned macro name.
    pub name: Symbol,
    /// `None` for object-like macros, `Some(params)` for function-like macros.
    /// Each element is the interned parameter name.
    pub params: Option<Vec<Symbol>>,
    /// `true` when the last formal parameter is `...` (variadic — `__VA_ARGS__`).
    pub is_variadic: bool,
    /// Replacement token list.
    pub body: Vec<Token>,
    /// `true` for built-in macros like `__FILE__`, `__LINE__`, `__DATE__`, etc.
    pub is_predefined: bool,
    /// Source location where this macro was `#define`'d.
    pub source_span: Span,
}

impl MacroDef {
    /// Create a new object-like macro definition.
    pub fn object_like(name: Symbol, body: Vec<Token>, span: Span) -> Self {
        Self {
            name,
            params: None,
            is_variadic: false,
            body,
            is_predefined: false,
            source_span: span,
        }
    }

    /// Create a new function-like macro definition.
    pub fn function_like(
        name: Symbol,
        params: Vec<Symbol>,
        is_variadic: bool,
        body: Vec<Token>,
        span: Span,
    ) -> Self {
        Self {
            name,
            params: Some(params),
            is_variadic,
            body,
            is_predefined: false,
            source_span: span,
        }
    }

    /// Create a predefined macro whose body is computed dynamically.
    pub fn predefined(name: Symbol) -> Self {
        Self {
            name,
            params: None,
            is_variadic: false,
            body: Vec::new(),
            is_predefined: true,
            source_span: Span::DUMMY,
        }
    }

    /// Returns `true` if this is a function-like macro (has parameter list).
    pub fn is_function_like(&self) -> bool {
        self.params.is_some()
    }

    /// Returns `true` if this is an object-like macro (no parameter list).
    pub fn is_object_like(&self) -> bool {
        self.params.is_none()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Preprocessor — main driver struct
// ═══════════════════════════════════════════════════════════════════════════

/// The top-level C preprocessor driver.
///
/// Orchestrates Phase 1 (trigraph replacement and line splicing) and Phase 2
/// (directive processing and macro expansion) of the BCC compilation pipeline.
/// Holds all shared state required during preprocessing: source map, diagnostic
/// engine, target architecture info, string interner, macro tables, and include
/// search paths.
pub struct Preprocessor {
    /// Source file tracking — file IDs, line offset tables, `#line` remapping.
    pub source_map: SourceMap,
    /// Multi-error diagnostic reporting engine.
    pub diagnostics: DiagnosticEngine,
    /// Target architecture — drives predefined macro selection.
    pub target: Target,
    /// String interner for identifiers, macro names, and keywords.
    pub interner: Interner,
    /// Macro definition table: maps interned macro name → definition.
    pub macros: FxHashMap<Symbol, MacroDef>,
    /// System and user include search paths (`-I` flags).
    pub include_paths: Vec<PathBuf>,
    /// Command-line `-D` macro definitions as pre-expanded token lists.
    pub defines: FxHashMap<Symbol, Vec<Token>>,
    /// Current recursion depth (macro expansion + `#include` nesting).
    pub recursion_depth: u32,
    /// Maximum allowed recursion depth (512 per Section 0.7.3).
    pub max_recursion_depth: u32,
    /// Include handler managing include stack, guards, and circular detection.
    include_handler: IncludeHandler,
    /// Conditional compilation nesting stack.
    /// Each entry records the state of one `#if`/`#ifdef`/`#ifndef` group.
    cond_stack: Vec<CondState>,
    /// Directory of the currently-being-processed file; used for relative
    /// `#include "..."` resolution.
    current_file_dir: PathBuf,
}

/// Tracks the state of a single conditional compilation group
/// (`#if` … `#elif` … `#else` … `#endif`).
#[derive(Debug, Clone, Copy)]
struct CondState {
    /// `true` if the current branch is the active one (tokens emitted).
    active: bool,
    /// `true` once any branch in this group has been taken.
    any_branch_taken: bool,
    /// `true` once we have seen `#else` (prevents duplicate `#else`).
    seen_else: bool,
    /// Span of the opening `#if`/`#ifdef`/`#ifndef` for diagnostics.
    origin_span: Span,
}

impl CondState {
    /// Creates a new conditional group opened by `#if` / `#ifdef` / `#ifndef`.
    fn new(active: bool, span: Span) -> Self {
        Self {
            active,
            any_branch_taken: active,
            seen_else: false,
            origin_span: span,
        }
    }
}

impl Preprocessor {
    // ─── Construction ───────────────────────────────────────────────────

    /// Creates a new `Preprocessor` configured for the given target.
    ///
    /// The caller should populate `include_paths` and `defines` (from `-I`
    /// and `-D` CLI flags) before invoking [`preprocess`].
    pub fn new(
        source_map: SourceMap,
        diagnostics: DiagnosticEngine,
        target: Target,
        interner: Interner,
    ) -> Self {
        let mut pp = Self {
            source_map,
            diagnostics,
            target,
            interner,
            macros: fx_hash_map(),
            include_paths: Vec::new(),
            defines: fx_hash_map(),
            recursion_depth: 0,
            max_recursion_depth: MAX_RECURSION_DEPTH,
            include_handler: IncludeHandler::new(),
            cond_stack: Vec::new(),
            current_file_dir: PathBuf::from("."),
        };
        pp.register_predefined_macros();
        pp
    }

    // ─── Predefined macro registration ──────────────────────────────────

    /// Registers **all** predefined macros into the macro table.
    ///
    /// Delegates to the comprehensive [`predefined::register_predefined_macros`]
    /// function which populates:
    ///
    /// 1. Dynamic macros (`__FILE__`, `__LINE__`, `__DATE__`, `__TIME__`, `__COUNTER__`)
    /// 2. C11 standard macros (`__STDC__`, `__STDC_VERSION__`, `__STDC_HOSTED__`,
    ///    `__STDC_UTF_16__`, `__STDC_UTF_32__`, `__STDC_NO_VLA__`)
    /// 3. Architecture-specific macros from `Target::predefined_macros()`
    /// 4. Platform macros (`__linux__`, `__ELF__`, `__unix__`, etc.)
    /// 5. Compiler identification (`__BCC__`, `__BCC_VERSION__`, `__GNUC__`, etc.)
    /// 6. Type-size and type-limit macros (`__SIZEOF_*__`, `__INT_MAX__`, etc.)
    /// 7. GCC sync builtins indicator macros
    fn register_predefined_macros(&mut self) {
        // Target is Copy, so we can take a local copy to avoid double-borrow
        // (predefined module needs &mut self for macro insertion AND &Target
        // for architecture-dependent values).
        let target = self.target;
        predefined::register_predefined_macros(self, &target);
    }

    /// Adds a system include search path (from `-I` flag).
    pub fn add_include_path(&mut self, path: PathBuf) {
        self.include_paths.push(path.clone());
        self.include_handler.add_system_path(path);
    }

    /// Adds a user include search path.
    pub fn add_user_include_path(&mut self, path: PathBuf) {
        self.include_paths.push(path.clone());
        self.include_handler.add_user_path(path);
    }

    /// Registers a command-line `-D` definition.
    ///
    /// `name_value` is in the form `NAME` or `NAME=VALUE`. If no value is
    /// provided, the macro is defined as `1`.
    pub fn add_define(&mut self, name_value: &str) {
        let (name, value) = if let Some(eq_pos) = name_value.find('=') {
            (&name_value[..eq_pos], &name_value[eq_pos + 1..])
        } else {
            (name_value, "1")
        };

        let sym = self.interner.intern(name);
        let body = if value.is_empty() {
            Vec::new()
        } else {
            // Tokenize the value into a simple replacement list.
            self.tokenize_define_value(value)
        };

        self.defines.insert(sym, body.clone());

        let def = MacroDef::object_like(sym, body, Span::DUMMY);
        self.macros.insert(sym, def);
    }

    /// Tokenize a `-D` value string into a token vector. Uses a minimal
    /// approach: if the value parses as an integer, produce an integer literal
    /// token; otherwise produce an identifier or string token.
    fn tokenize_define_value(&mut self, value: &str) -> Vec<Token> {
        // Try integer parse first.
        if let Ok(n) = value.parse::<u128>() {
            return vec![Token::new(
                TokenKind::IntegerLiteral {
                    value: n,
                    suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                },
                Span::DUMMY,
            )];
        }
        // Fall back to a single identifier token.
        let sym = self.interner.intern(value);
        vec![Token::new(TokenKind::Identifier(sym), Span::DUMMY)]
    }

    // ─── Main entry point ───────────────────────────────────────────────

    /// Preprocess the C source file at `file_path`.
    ///
    /// This is the main orchestration method. It:
    /// 1. Reads the source file with PUA-aware encoding.
    /// 2. Applies Phase 1 (trigraph replacement + line splicing).
    /// 3. Tokenizes the result with the lightweight preprocessor tokenizer.
    /// 4. Processes Phase 2 (directives and macro expansion).
    /// 5. Returns the fully expanded token stream consumed by the parser.
    #[allow(clippy::result_unit_err)]
    pub fn preprocess(&mut self, file_path: &Path) -> Result<Vec<Token>, ()> {
        // Step 1: Read source with PUA encoding for non-UTF-8 fidelity.
        let raw_source = match read_source_file(file_path) {
            Ok(s) => s,
            Err(e) => {
                self.diagnostics.error(
                    Span::DUMMY,
                    format!("cannot open source file '{}': {}", file_path.display(), e),
                );
                return Err(());
            }
        };

        // Step 2: Phase 1 — trigraph replacement and line splicing.
        let spliced = phase1_trigraphs_and_line_splice(&raw_source);

        // Step 3: Register the file in the source map and set current dir.
        let file_name = file_path.to_str().unwrap_or("<unknown>").to_string();
        self.current_file_dir = file_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let file_id = self.source_map.add_file(file_name, spliced.clone());
        let file_id_raw = file_id.0;

        // Step 4: Tokenize into a raw preprocessing token stream.
        let raw_tokens = pp_tokenize(&spliced, file_id_raw, &mut self.interner);

        // Step 5: Phase 2 — directive processing and macro expansion.
        self.process_tokens(raw_tokens, file_id_raw)
    }

    /// Process an already-tokenized stream through Phase 2 (directive
    /// processing and macro expansion). Shared between the top-level
    /// `preprocess` and `#include` handling.
    fn process_tokens(&mut self, tokens: Vec<Token>, file_id: u32) -> Result<Vec<Token>, ()> {
        let mut output: Vec<Token> = Vec::new();
        let mut idx = 0;
        let len = tokens.len();

        // Record the conditional stack depth at entry so that the
        // unterminated-conditional check at the end of this file only
        // examines conditions opened *during this call* — not those
        // inherited from a parent file that `#include`d us.
        let cond_stack_base = self.cond_stack.len();

        // Track whether we are at the start of a logical line (for directive
        // detection). The very first token is at start-of-line.
        let mut at_line_start = true;

        while idx < len {
            let tok = &tokens[idx];

            // Skip whitespace, track newlines for line-start detection.
            if tok.kind == TokenKind::Newline {
                at_line_start = true;
                idx += 1;
                continue;
            }
            if tok.kind == TokenKind::Whitespace {
                idx += 1;
                continue;
            }

            // ── Directive detection ──────────────────────────────────
            if at_line_start && tok.kind == TokenKind::Hash {
                let directive_span = tok.span;
                idx += 1;
                // Skip whitespace after '#'.
                while idx < len && tokens[idx].kind == TokenKind::Whitespace {
                    idx += 1;
                }
                if idx >= len || tokens[idx].kind == TokenKind::Newline {
                    // Null directive `#` followed by newline — C11 §6.10.7.
                    at_line_start = true;
                    if idx < len {
                        idx += 1;
                    }
                    continue;
                }

                // Collect the directive line (up to newline or EOF).
                let dir_start = idx;
                while idx < len && tokens[idx].kind != TokenKind::Newline {
                    idx += 1;
                }
                let dir_tokens = &tokens[dir_start..idx];
                // Consume the newline.
                if idx < len {
                    idx += 1;
                }
                at_line_start = true;

                // Process the directive if conditionals allow it.
                self.handle_directive(dir_tokens, directive_span, file_id, &mut output)?;
                continue;
            }

            at_line_start = false;

            // ── Conditional skipping ─────────────────────────────────
            if !self.is_active() {
                idx += 1;
                continue;
            }

            // ── Macro expansion ──────────────────────────────────────
            // Collect tokens until the next newline to form a logical line
            // and expand macros on the batch.
            //
            // CRITICAL: Function-like macro calls can span multiple physical
            // lines (e.g., glibc's __REDIRECT macro). When collecting the
            // line, if we encounter open parentheses, we track nesting depth
            // and continue past newlines until the parentheses are balanced.
            // This matches C11 §6.10.3 which says that a preprocessing
            // directive's replacement list (and macro call arguments) can
            // contain newlines that act as whitespace.
            let line_start = idx;
            let mut paren_depth: i32 = 0;
            while idx < len {
                if tokens[idx].kind == TokenKind::Newline {
                    // Only stop at newline if parentheses are balanced.
                    if paren_depth <= 0 {
                        break;
                    }
                    // Inside a parenthesized expression — treat newline
                    // as whitespace and continue collecting tokens.
                    idx += 1;
                    continue;
                }
                if tokens[idx].kind == TokenKind::LeftParen {
                    paren_depth += 1;
                } else if tokens[idx].kind == TokenKind::RightParen {
                    paren_depth -= 1;
                }
                idx += 1;
            }
            let line_tokens = tokens[line_start..idx].to_vec();

            let expanded = self.expand_line(line_tokens)?;
            // Filter whitespace from expanded output.
            for t in expanded {
                if t.kind != TokenKind::Whitespace {
                    output.push(t);
                }
            }
        }

        // Validate conditional stack is balanced *for this file only*.
        // Only conditionals pushed after cond_stack_base are from this
        // file — any below that belong to a parent that #include'd us
        // and will be checked when that parent's process_tokens returns.
        if self.cond_stack.len() > cond_stack_base {
            let unclosed = &self.cond_stack[cond_stack_base];
            self.diagnostics
                .error(unclosed.origin_span, "unterminated #if / #ifdef / #ifndef");
            // Pop all conditionals opened in this file before returning.
            self.cond_stack.truncate(cond_stack_base);
            return Err(());
        }

        // Append EOF sentinel.
        output.push(Token::new(TokenKind::Eof, Span::DUMMY));
        Ok(output)
    }

    // ─── Conditional compilation helpers ─────────────────────────────────

    /// Returns `true` if token output is currently active (not inside a
    /// skipped conditional branch).
    fn is_active(&self) -> bool {
        self.cond_stack.iter().all(|c| c.active)
    }

    // ─── Directive dispatch ──────────────────────────────────────────────

    /// Dispatch a preprocessor directive line. `dir_tokens` starts with the
    /// directive name token (e.g., `define`, `include`, `if`, …) and extends
    /// to the end of the directive line (excluding the newline).
    fn handle_directive(
        &mut self,
        dir_tokens: &[Token],
        _hash_span: Span,
        _file_id: u32,
        output: &mut Vec<Token>,
    ) -> Result<(), ()> {
        if dir_tokens.is_empty() {
            return Ok(());
        }

        // Extract the directive keyword.
        let dir_name = self.token_text(&dir_tokens[0]);
        let dir_span = dir_tokens[0].span;

        // Some directives must be processed even inside skipped branches
        // (conditional directives that affect nesting).
        match dir_name.as_str() {
            "if" | "ifdef" | "ifndef" => {
                return self.handle_conditional_open(&dir_name, &dir_tokens[1..], dir_span);
            }
            "elif" => {
                return self.handle_elif(&dir_tokens[1..], dir_span);
            }
            "else" => {
                return self.handle_else(dir_span);
            }
            "endif" => {
                return self.handle_endif(dir_span);
            }
            _ => {}
        }

        // All other directives are only processed when active.
        if !self.is_active() {
            return Ok(());
        }

        match dir_name.as_str() {
            "define" => self.handle_define(&dir_tokens[1..], dir_span),
            "undef" => self.handle_undef(&dir_tokens[1..], dir_span),
            "include" => self.handle_include(&dir_tokens[1..], dir_span, output),
            "pragma" => self.handle_pragma(&dir_tokens[1..], dir_span),
            "error" => self.handle_error(&dir_tokens[1..], dir_span),
            "warning" => self.handle_warning(&dir_tokens[1..], dir_span),
            "line" => self.handle_line_directive(&dir_tokens[1..], dir_span),
            _ => {
                self.diagnostics.warning(
                    dir_span,
                    format!("unknown preprocessing directive '#{}' ignored", dir_name),
                );
                Ok(())
            }
        }
    }

    // ─── #define / #undef ────────────────────────────────────────────────

    /// Handle `#define NAME ...` and `#define NAME(params) ...`.
    fn handle_define(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        let tokens = skip_ws(tokens);
        if tokens.is_empty() {
            self.diagnostics
                .error(span, "expected macro name after #define");
            return Err(());
        }

        let name_sym = match tokens[0].kind {
            TokenKind::Identifier(s) => s,
            _ => {
                self.diagnostics
                    .error(tokens[0].span, "expected identifier for macro name");
                return Err(());
            }
        };

        let rest = &tokens[1..];

        // Determine if function-like: the `(` must be immediately adjacent to
        // the macro name (no whitespace).
        let is_func_like = !rest.is_empty()
            && rest[0].kind == TokenKind::LeftParen
            && tokens[0].span.end == rest[0].span.start;

        if is_func_like {
            // Parse parameter list.
            let after_paren = &rest[1..];
            let mut params: Vec<Symbol> = Vec::new();
            let mut is_variadic = false;
            let mut i = 0;
            let after_paren = skip_ws(after_paren);

            loop {
                if i >= after_paren.len() {
                    self.diagnostics
                        .error(span, "unterminated macro parameter list");
                    return Err(());
                }

                // Check for closing paren (empty param list or end of list).
                if after_paren[i].kind == TokenKind::RightParen {
                    i += 1;
                    break;
                }

                // Check for `...` (variadic).
                if after_paren[i].kind == TokenKind::Ellipsis {
                    is_variadic = true;
                    i += 1;
                    let rest_ws = skip_ws(&after_paren[i..]);
                    if rest_ws.is_empty() || rest_ws[0].kind != TokenKind::RightParen {
                        self.diagnostics
                            .error(span, "expected ')' after '...' in macro parameter list");
                        return Err(());
                    }
                    i += after_paren[i..].len() - rest_ws.len() + 1;
                    break;
                }

                // Expect identifier.
                match after_paren[i].kind {
                    TokenKind::Identifier(sym) => {
                        params.push(sym);
                        i += 1;
                    }
                    _ => {
                        self.diagnostics.error(
                            after_paren[i].span,
                            "expected parameter name in macro definition",
                        );
                        return Err(());
                    }
                }

                // Skip whitespace and expect ',' or ')'.
                while i < after_paren.len() && after_paren[i].kind == TokenKind::Whitespace {
                    i += 1;
                }
                if i < after_paren.len() && after_paren[i].kind == TokenKind::Comma {
                    i += 1;
                    // Skip whitespace after comma.
                    while i < after_paren.len() && after_paren[i].kind == TokenKind::Whitespace {
                        i += 1;
                    }
                }
            }

            // Remaining tokens after `)` form the macro body.
            let body_start = {
                let total_consumed = rest.len() - after_paren.len() + i;
                total_consumed.min(rest.len())
            };
            let body_tokens: Vec<Token> = skip_ws(&rest[body_start..]).to_vec();

            // Warn if redefining a macro (unless identical).
            if let Some(existing) = self.macros.get(&name_sym) {
                if !existing.is_predefined {
                    self.diagnostics.warning(
                        span,
                        format!("'{}' macro redefined", self.interner.resolve(name_sym)),
                    );
                }
            }

            let def = MacroDef::function_like(name_sym, params, is_variadic, body_tokens, span);
            self.macros.insert(name_sym, def);
        } else {
            // Object-like macro.
            let body_tokens: Vec<Token> = skip_ws(rest).to_vec();

            if let Some(existing) = self.macros.get(&name_sym) {
                if !existing.is_predefined {
                    self.diagnostics.warning(
                        span,
                        format!("'{}' macro redefined", self.interner.resolve(name_sym)),
                    );
                }
            }

            let def = MacroDef::object_like(name_sym, body_tokens, span);
            self.macros.insert(name_sym, def);
        }

        Ok(())
    }

    /// Handle `#undef NAME`.
    fn handle_undef(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        let tokens = skip_ws(tokens);
        if tokens.is_empty() {
            self.diagnostics
                .error(span, "expected macro name after #undef");
            return Err(());
        }
        match tokens[0].kind {
            TokenKind::Identifier(sym) => {
                self.macros.remove(&sym);
                Ok(())
            }
            _ => {
                self.diagnostics
                    .error(tokens[0].span, "expected identifier after #undef");
                Err(())
            }
        }
    }

    // ─── #include ────────────────────────────────────────────────────────

    /// Handle `#include "file"` and `#include <file>`.
    fn handle_include(
        &mut self,
        tokens: &[Token],
        span: Span,
        output: &mut Vec<Token>,
    ) -> Result<(), ()> {
        let tokens = skip_ws(tokens);
        if tokens.is_empty() {
            self.diagnostics
                .error(span, "expected file path after #include");
            return Err(());
        }

        // Determine include kind and path from the tokens.
        let (kind, path_str) = self.parse_include_path(tokens, span)?;

        // Check recursion depth (manual increment/decrement to avoid
        // holding a mutable borrow through RecursionGuard across later
        // self-method calls).
        if self.recursion_depth >= self.max_recursion_depth {
            self.diagnostics.error(
                span,
                format!(
                    "#include nesting depth exceeds limit ({})",
                    self.max_recursion_depth,
                ),
            );
            return Err(());
        }
        self.recursion_depth += 1;

        // Resolve the include path via the include handler.
        let current_dir = self.current_file_dir.clone();
        let resolved = match self
            .include_handler
            .resolve_include(&path_str, kind, &current_dir)
        {
            Some(p) => p,
            None => {
                self.diagnostics
                    .error(span, format!("'{}': file not found", path_str));
                self.recursion_depth -= 1;
                return Err(());
            }
        };

        // Read and preprocess the included file.
        let raw_source = match read_source_file(&resolved) {
            Ok(s) => s,
            Err(e) => {
                self.diagnostics
                    .error(span, format!("cannot read '{}': {}", resolved.display(), e));
                self.recursion_depth -= 1;
                return Err(());
            }
        };

        let spliced = phase1_trigraphs_and_line_splice(&raw_source);
        let file_name = resolved.to_str().unwrap_or("<unknown>").to_string();
        // Save and restore current_file_dir around the recursive include.
        let saved_dir = self.current_file_dir.clone();
        self.current_file_dir = resolved
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let inc_file_id = self.source_map.add_file(file_name, spliced.clone());
        let inc_file_id_raw = inc_file_id.0;
        let inc_tokens = pp_tokenize(&spliced, inc_file_id_raw, &mut self.interner);

        let expanded = match self.process_tokens(inc_tokens, inc_file_id_raw) {
            Ok(toks) => toks,
            Err(e) => {
                self.recursion_depth -= 1;
                self.current_file_dir = saved_dir;
                return Err(e);
            }
        };
        self.current_file_dir = saved_dir;
        self.recursion_depth -= 1;
        // Append expanded tokens (excluding EOF) to the output.
        for t in expanded {
            if t.kind != TokenKind::Eof {
                output.push(t);
            }
        }

        Ok(())
    }

    /// Parse the include path from tokens following `#include`.
    /// Returns the include kind and the path string.
    ///
    /// For `<...>` system includes, the raw source text is extracted directly
    /// from the source map using token spans, avoiding token-text reconstruction
    /// issues (e.g., `64.h` being lexed as a floating-point literal and losing
    /// the original text). This matches the C standard's requirement that the
    /// content between `<` and `>` is a "header-name" preprocessing token, not
    /// a sequence of regular C tokens.
    fn parse_include_path(
        &mut self,
        tokens: &[Token],
        span: Span,
    ) -> Result<(IncludeKind, String), ()> {
        // Check for `"path"` (user include).
        if let TokenKind::StringLiteral { ref value, .. } = tokens[0].kind {
            let path_string = String::from_utf8_lossy(value).to_string();
            return Ok((IncludeKind::User, path_string));
        }

        // Check for `<path>` (system include) — extract raw source text from
        // spans to preserve the exact filename. The content between `<` and `>`
        // can contain characters that the C tokenizer misinterprets (e.g., `64.h`
        // is lexed as a float literal), so we bypass tokenization entirely.
        if tokens[0].kind == TokenKind::Less {
            // Find the closing `>` token.
            let close_idx = tokens[1..]
                .iter()
                .position(|t| t.kind == TokenKind::Greater)
                .map(|pos| pos + 1);
            let close_idx = match close_idx {
                Some(idx) => idx,
                None => {
                    self.diagnostics
                        .error(span, "missing '>' in #include <...>");
                    return Err(());
                }
            };

            // Try to extract raw source text using spans. All tokens in a
            // non-macro-expanded #include line originate from the same file,
            // so the spans are contiguous and valid.
            let lt_span = tokens[0].span;
            let gt_span = tokens[close_idx].span;
            if lt_span.file_id == gt_span.file_id && lt_span.file_id != u32::MAX {
                let file_id = crate::common::source_map::FileId(lt_span.file_id);
                let raw = self
                    .source_map
                    .get_snippet(file_id, lt_span.end, gt_span.start);
                let path = raw.trim().to_string();
                if !path.is_empty() {
                    return Ok((IncludeKind::System, path));
                }
            }

            // Fallback: reconstruct from token text (for macro-expanded includes
            // or cases where span extraction yields an empty path).
            let mut path = String::new();
            for tok in &tokens[1..close_idx] {
                path.push_str(&self.token_text(tok));
            }
            return Ok((IncludeKind::System, path));
        }

        self.diagnostics
            .error(span, "expected \"file\" or <file> after #include");
        Err(())
    }

    // ─── Conditional compilation: #if / #ifdef / #ifndef / #elif / #else / #endif ─

    /// Handle `#if expr`, `#ifdef NAME`, `#ifndef NAME`.
    fn handle_conditional_open(
        &mut self,
        directive: &str,
        tokens: &[Token],
        span: Span,
    ) -> Result<(), ()> {
        // If we are inside a skipped branch, just push a nested inactive group.
        if !self.is_active() {
            self.cond_stack.push(CondState::new(false, span));
            return Ok(());
        }

        let active = match directive {
            "ifdef" => {
                let tokens = skip_ws(tokens);
                if tokens.is_empty() {
                    self.diagnostics
                        .error(span, "expected identifier after #ifdef");
                    return Err(());
                }
                match tokens[0].kind {
                    TokenKind::Identifier(sym) => self.macros.contains_key(&sym),
                    _ => {
                        self.diagnostics
                            .error(tokens[0].span, "expected identifier after #ifdef");
                        return Err(());
                    }
                }
            }
            "ifndef" => {
                let tokens = skip_ws(tokens);
                if tokens.is_empty() {
                    self.diagnostics
                        .error(span, "expected identifier after #ifndef");
                    return Err(());
                }
                match tokens[0].kind {
                    TokenKind::Identifier(sym) => !self.macros.contains_key(&sym),
                    _ => {
                        self.diagnostics
                            .error(tokens[0].span, "expected identifier after #ifndef");
                        return Err(());
                    }
                }
            }
            "if" => {
                let tokens = skip_ws(tokens);
                self.evaluate_condition(tokens, span)?
            }
            _ => unreachable!(),
        };

        self.cond_stack.push(CondState::new(active, span));
        Ok(())
    }

    /// Handle `#elif expr`.
    fn handle_elif(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        if self.cond_stack.is_empty() {
            self.diagnostics.error(span, "#elif without matching #if");
            return Err(());
        }

        // Read the needed state from `cond_stack` first without holding a long
        // mutable borrow, so that `evaluate_condition` can borrow `self`.
        let seen_else = self.cond_stack.last().unwrap().seen_else;
        if seen_else {
            self.diagnostics.error(span, "#elif after #else");
            return Err(());
        }

        let any_taken = self.cond_stack.last().unwrap().any_branch_taken;
        if any_taken {
            // A previous branch was taken — skip this one.
            self.cond_stack.last_mut().unwrap().active = false;
        } else {
            // Evaluate the condition (this borrows `self` mutably).
            let tokens = skip_ws(tokens);
            let val = self.evaluate_condition(tokens, span)?;
            let top = self.cond_stack.last_mut().unwrap();
            top.active = val;
            if val {
                top.any_branch_taken = true;
            }
        }

        Ok(())
    }

    /// Handle `#else`.
    fn handle_else(&mut self, span: Span) -> Result<(), ()> {
        if self.cond_stack.is_empty() {
            self.diagnostics.error(span, "#else without matching #if");
            return Err(());
        }

        let seen_else = self.cond_stack.last().unwrap().seen_else;
        if seen_else {
            self.diagnostics.error(span, "duplicate #else");
            return Err(());
        }

        let top = self.cond_stack.last_mut().unwrap();
        top.seen_else = true;
        top.active = !top.any_branch_taken;

        Ok(())
    }

    /// Handle `#endif`.
    fn handle_endif(&mut self, span: Span) -> Result<(), ()> {
        if self.cond_stack.is_empty() {
            self.diagnostics.error(span, "#endif without matching #if");
            return Err(());
        }
        self.cond_stack.pop();
        Ok(())
    }

    // ─── #pragma / #error / #warning / #line ─────────────────────────────

    /// Handle `#pragma ...`. Currently recognizes `#pragma once`.
    fn handle_pragma(&mut self, tokens: &[Token], _span: Span) -> Result<(), ()> {
        let tokens = skip_ws(tokens);
        if !tokens.is_empty() {
            let name = self.token_text(&tokens[0]);
            if name == "once" {
                // Mark the current file for include-guard optimization.
                // The include handler tracks this internally.
                // No-op here since the include handler is responsible for dedup.
                return Ok(());
            }
            // GCC-specific pragmas (push_macro, pop_macro, etc.) can be
            // added in future iterations. For now, silently ignore.
        }
        Ok(())
    }

    /// Handle `#error message`.
    fn handle_error(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        let msg = self.concat_token_text(tokens);
        self.diagnostics
            .error(span, format!("#error {}", msg.trim()));
        Err(())
    }

    /// Handle `#warning message` (GCC extension).
    fn handle_warning(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        let msg = self.concat_token_text(tokens);
        self.diagnostics
            .warning(span, format!("#warning {}", msg.trim()));
        Ok(())
    }

    /// Handle `#line NUMBER ["filename"]`.
    fn handle_line_directive(&mut self, tokens: &[Token], span: Span) -> Result<(), ()> {
        let tokens = skip_ws(tokens);
        if tokens.is_empty() {
            self.diagnostics
                .error(span, "expected line number after #line");
            return Err(());
        }
        // Parse the line number.
        if let TokenKind::IntegerLiteral { value, .. } = tokens[0].kind {
            let _line_no = value as u32;
            // Optionally parse filename.
            let _filename = if tokens.len() > 1 {
                let rest = skip_ws(&tokens[1..]);
                if !rest.is_empty() {
                    if let TokenKind::StringLiteral { ref value, .. } = rest[0].kind {
                        Some(String::from_utf8_lossy(value).to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            // Source map remapping would be applied here.
            // For now, #line is accepted but source map adjustment is deferred
            // to integration with the source_map module's remap API.
            Ok(())
        } else {
            self.diagnostics
                .error(tokens[0].span, "expected integer after #line");
            Err(())
        }
    }

    // ─── Preprocessor constant expression evaluation ─────────────────────

    /// Evaluate a preprocessor constant expression for `#if` / `#elif`.
    ///
    /// The expression is evaluated after macro expansion. Identifiers that
    /// remain after expansion (not defined as macros) are replaced with `0`
    /// per C11 §6.10.1p4, except for `defined(NAME)` / `defined NAME`.
    fn evaluate_condition(&mut self, tokens: &[Token], span: Span) -> Result<bool, ()> {
        if tokens.is_empty() {
            self.diagnostics
                .error(span, "expected expression in #if directive");
            return Err(());
        }

        // First, handle `defined` operator before macro expansion.
        let processed = self.process_defined_operator(tokens);

        // Then macro-expand the remaining tokens.
        let expanded = self.expand_line(processed)?;

        // Replace remaining identifiers with 0 (C11 §6.10.1p4).
        let replaced: Vec<Token> = expanded
            .iter()
            .map(|t| match t.kind {
                TokenKind::Identifier(_) => Token::new(
                    TokenKind::IntegerLiteral {
                        value: 0,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                    },
                    t.span,
                ),
                _ => t.clone(),
            })
            .collect();

        // Evaluate the expression.
        let mut eval = CondExprEvaluator::new(&replaced);
        let result = eval.eval_ternary();

        match result {
            Ok(val) => Ok(val != 0),
            Err(msg) => {
                self.diagnostics
                    .error(span, format!("invalid preprocessor expression: {}", msg));
                Err(())
            }
        }
    }

    /// Process the `defined` operator in a token stream. Converts
    /// `defined NAME` and `defined(NAME)` into integer literal tokens
    /// (1 if defined, 0 if not). Must run BEFORE macro expansion.
    fn process_defined_operator(&self, tokens: &[Token]) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::new();
        let mut i = 0;
        // Note: `defined` is not a keyword token in our lexer — it's an
        // identifier with special meaning only in preprocessor #if expressions.
        // We compare resolved text below rather than relying on a Symbol match.

        while i < tokens.len() {
            if let TokenKind::Identifier(sym) = tokens[i].kind {
                let text = self.interner.resolve(sym);
                if text == "defined" {
                    let span = tokens[i].span;
                    i += 1;
                    // Skip whitespace.
                    while i < tokens.len() && tokens[i].kind == TokenKind::Whitespace {
                        i += 1;
                    }
                    // `defined(NAME)` form.
                    if i < tokens.len() && tokens[i].kind == TokenKind::LeftParen {
                        i += 1;
                        while i < tokens.len() && tokens[i].kind == TokenKind::Whitespace {
                            i += 1;
                        }
                        if i < tokens.len() {
                            if let TokenKind::Identifier(name_sym) = tokens[i].kind {
                                let is_def = self.macros.contains_key(&name_sym);
                                out.push(Token::new(
                                    TokenKind::IntegerLiteral {
                                        value: if is_def { 1 } else { 0 },
                                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                                    },
                                    span,
                                ));
                                i += 1;
                                // Skip whitespace and closing paren.
                                while i < tokens.len() && tokens[i].kind == TokenKind::Whitespace {
                                    i += 1;
                                }
                                if i < tokens.len() && tokens[i].kind == TokenKind::RightParen {
                                    i += 1;
                                }
                                continue;
                            }
                        }
                        // Fallthrough: malformed defined() — treat as 0.
                        out.push(Token::new(
                            TokenKind::IntegerLiteral {
                                value: 0,
                                suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                            },
                            span,
                        ));
                        continue;
                    }
                    // `defined NAME` form (without parens).
                    if i < tokens.len() {
                        if let TokenKind::Identifier(name_sym) = tokens[i].kind {
                            let is_def = self.macros.contains_key(&name_sym);
                            out.push(Token::new(
                                TokenKind::IntegerLiteral {
                                    value: if is_def { 1 } else { 0 },
                                    suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                                },
                                span,
                            ));
                            i += 1;
                            continue;
                        }
                    }
                    // Malformed: push 0.
                    out.push(Token::new(
                        TokenKind::IntegerLiteral {
                            value: 0,
                            suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                        },
                        span,
                    ));
                    continue;
                }
            }
            out.push(tokens[i].clone());
            i += 1;
        }
        out
    }

    // ─── Macro expansion ─────────────────────────────────────────────────

    /// Expand all macros in a line of tokens. Returns the fully expanded
    /// token sequence.
    fn expand_line(&mut self, tokens: Vec<Token>) -> Result<Vec<Token>, ()> {
        let mut painted: Vec<PaintedToken> = tokens.into_iter().map(PaintedToken::new).collect();

        let mut output: Vec<Token> = Vec::new();
        let mut i = 0;

        while i < painted.len() {
            let pt = &painted[i];
            let tok = &pt.token;

            // Only try to expand identifier tokens that aren't painted for
            // their own name.
            if let TokenKind::Identifier(sym) = tok.kind {
                if !is_painted_for(&painted[i], sym) {
                    if let Some(def) = self.macros.get(&sym).cloned() {
                        // Check recursion depth (manual increment/decrement to
                        // avoid holding a mutable borrow through `RecursionGuard`).
                        if self.recursion_depth >= self.max_recursion_depth {
                            self.diagnostics.error(
                                tok.span,
                                format!(
                                    "macro expansion depth exceeds limit ({})",
                                    self.max_recursion_depth,
                                ),
                            );
                            return Err(());
                        }
                        self.recursion_depth += 1;

                        if def.is_predefined {
                            // Predefined macros: expand dynamically.
                            let expanded = self.expand_predefined(sym, tok.span);
                            output.extend(expanded);
                            self.recursion_depth -= 1;
                            i += 1;
                            continue;
                        }

                        if def.is_function_like() {
                            // Function-like macro: need to find `(` after name.
                            let after_name = i + 1;
                            let paren_pos = self.find_open_paren(&painted, after_name);
                            if let Some(paren_idx) = paren_pos {
                                // Collect arguments.
                                let (args, end_idx) = self.collect_macro_args(
                                    &painted,
                                    paren_idx + 1,
                                    &def,
                                    tok.span,
                                )?;

                                // Substitute parameters and expand.
                                let replacement = self.substitute_params(&def, &args, tok.span);

                                // Paint all output tokens for this macro.
                                let mut expanded_painted: Vec<PaintedToken> =
                                    replacement.into_iter().map(PaintedToken::new).collect();
                                paint_tokens(&mut expanded_painted, sym);

                                // Re-scan: insert expanded tokens for further expansion.
                                let rest = painted.split_off(end_idx + 1);
                                painted.truncate(i);
                                painted.extend(expanded_painted);
                                painted.extend(rest);
                                self.recursion_depth -= 1;
                                // Don't advance i — re-scan from the same position.
                                continue;
                            } else {
                                // No `(` found: not a macro invocation,
                                // just output the identifier.
                                output.push(tok.clone());
                                self.recursion_depth -= 1;
                                i += 1;
                                continue;
                            }
                        } else {
                            // Object-like macro: substitute body.
                            let mut expanded_painted: Vec<PaintedToken> =
                                def.body.iter().cloned().map(PaintedToken::new).collect();
                            paint_tokens(&mut expanded_painted, sym);

                            // Re-scan: insert expanded tokens for further expansion.
                            let rest = painted.split_off(i + 1);
                            painted.truncate(i);
                            painted.extend(expanded_painted);
                            painted.extend(rest);
                            self.recursion_depth -= 1;
                            continue;
                        }
                    }
                }
            }

            // Not a macro — pass through.
            if tok.kind != TokenKind::Whitespace {
                output.push(tok.clone());
            }
            i += 1;
        }

        Ok(output)
    }

    /// Find the index of the `(` token after a function-like macro name,
    /// skipping any whitespace between the name and the paren.
    fn find_open_paren(&self, tokens: &[PaintedToken], start: usize) -> Option<usize> {
        let mut i = start;
        while i < tokens.len() {
            match tokens[i].token.kind {
                TokenKind::Whitespace | TokenKind::Newline => {
                    i += 1;
                }
                TokenKind::LeftParen => return Some(i),
                _ => return None,
            }
        }
        None
    }

    /// Collect the argument token lists for a function-like macro invocation.
    /// Returns `(args, end_index)` where `end_index` is the index of the
    /// closing `)`.
    fn collect_macro_args(
        &mut self,
        tokens: &[PaintedToken],
        start: usize,
        def: &MacroDef,
        span: Span,
    ) -> Result<(Vec<Vec<Token>>, usize), ()> {
        let mut args: Vec<Vec<Token>> = Vec::new();
        let mut current_arg: Vec<Token> = Vec::new();
        let mut depth: u32 = 1; // Already inside the outer `(`.
        let mut i = start;

        while i < tokens.len() {
            let kind = &tokens[i].token.kind;

            match kind {
                TokenKind::LeftParen => {
                    depth += 1;
                    current_arg.push(tokens[i].token.clone());
                }
                TokenKind::RightParen => {
                    depth -= 1;
                    if depth == 0 {
                        // End of arguments.
                        args.push(current_arg);
                        return Ok((args, i));
                    }
                    current_arg.push(tokens[i].token.clone());
                }
                TokenKind::Comma if depth == 1 => {
                    // Argument separator (only at top paren level).
                    // For variadic macros, if we've collected enough non-variadic
                    // args, the rest including commas goes into __VA_ARGS__.
                    let param_count = def.params.as_ref().map_or(0, |p| p.len());
                    if def.is_variadic && args.len() >= param_count {
                        // Pack remaining tokens (including commas) into the
                        // variadic argument.
                        current_arg.push(tokens[i].token.clone());
                    } else {
                        args.push(current_arg);
                        current_arg = Vec::new();
                    }
                }
                _ => {
                    current_arg.push(tokens[i].token.clone());
                }
            }
            i += 1;
        }

        self.diagnostics
            .error(span, "unterminated macro argument list");
        Err(())
    }

    /// Substitute macro parameters with their argument values in the
    /// replacement body. Handles `#` (stringification) and `##` (pasting).
    fn substitute_params(
        &mut self,
        def: &MacroDef,
        args: &[Vec<Token>],
        _span: Span,
    ) -> Vec<Token> {
        let params = match &def.params {
            Some(p) => p,
            None => return def.body.clone(),
        };

        let va_args_sym = self.interner.intern("__VA_ARGS__");
        let mut result: Vec<Token> = Vec::new();
        let body = &def.body;
        let mut i = 0;

        while i < body.len() {
            // Check for `#` (stringification).
            if body[i].kind == TokenKind::Hash && i + 1 < body.len() {
                if let TokenKind::Identifier(param_sym) = body[i + 1].kind {
                    if let Some(idx) = params.iter().position(|p| *p == param_sym) {
                        let arg_tokens = args.get(idx).cloned().unwrap_or_default();
                        let stringified_tok = stringify(&arg_tokens, &self.interner);
                        result.push(stringified_tok);
                        i += 2;
                        continue;
                    } else if param_sym == va_args_sym && def.is_variadic {
                        let arg_tokens = args.get(params.len()).cloned().unwrap_or_default();
                        let stringified_tok = stringify(&arg_tokens, &self.interner);
                        result.push(stringified_tok);
                        i += 2;
                        continue;
                    }
                }
            }

            // Check for `##` (token pasting) — handled by collecting lhs and rhs.
            if i + 1 < body.len() && body[i + 1].kind == TokenKind::HashHash {
                let lhs_tokens = self.resolve_param_or_token(
                    &body[i],
                    params,
                    args,
                    &va_args_sym,
                    def.is_variadic,
                );
                if i + 2 < body.len() {
                    let rhs_tokens = self.resolve_param_or_token(
                        &body[i + 2],
                        params,
                        args,
                        &va_args_sym,
                        def.is_variadic,
                    );
                    // Take the last token of lhs and first token of rhs for pasting.
                    let lhs_tok = lhs_tokens
                        .last()
                        .cloned()
                        .unwrap_or_else(|| body[i].clone());
                    let rhs_tok = rhs_tokens
                        .first()
                        .cloned()
                        .unwrap_or_else(|| body[i + 2].clone());
                    // Emit any lhs tokens before the last one.
                    if lhs_tokens.len() > 1 {
                        result.extend_from_slice(&lhs_tokens[..lhs_tokens.len() - 1]);
                    }
                    let pasted = paste_tokens(
                        &lhs_tok,
                        &rhs_tok,
                        &mut self.interner,
                        &mut self.diagnostics,
                    );
                    result.push(pasted);
                    // Emit any rhs tokens after the first one.
                    if rhs_tokens.len() > 1 {
                        result.extend_from_slice(&rhs_tokens[1..]);
                    }
                    i += 3;
                    continue;
                }
            }

            // Regular parameter substitution.
            if let TokenKind::Identifier(sym) = body[i].kind {
                if let Some(idx) = params.iter().position(|p| *p == sym) {
                    let arg_tokens = args.get(idx).cloned().unwrap_or_default();
                    // Strip leading/trailing whitespace from argument tokens.
                    let trimmed: Vec<Token> = arg_tokens
                        .into_iter()
                        .filter(|t| t.kind != TokenKind::Whitespace)
                        .collect();
                    result.extend(trimmed);
                    i += 1;
                    continue;
                } else if sym == va_args_sym && def.is_variadic {
                    let arg_tokens = args.get(params.len()).cloned().unwrap_or_default();
                    let trimmed: Vec<Token> = arg_tokens
                        .into_iter()
                        .filter(|t| t.kind != TokenKind::Whitespace)
                        .collect();
                    result.extend(trimmed);
                    i += 1;
                    continue;
                }
            }

            // Pass token through unchanged.
            result.push(body[i].clone());
            i += 1;
        }

        result
    }

    /// Resolve a body token: if it's a parameter name, return the argument
    /// tokens; otherwise return a single-element vec with the token itself.
    fn resolve_param_or_token(
        &self,
        tok: &Token,
        params: &[Symbol],
        args: &[Vec<Token>],
        va_args_sym: &Symbol,
        is_variadic: bool,
    ) -> Vec<Token> {
        if let TokenKind::Identifier(sym) = tok.kind {
            if let Some(idx) = params.iter().position(|p| *p == sym) {
                return args.get(idx).cloned().unwrap_or_default();
            }
            if is_variadic && sym == *va_args_sym {
                return args.get(params.len()).cloned().unwrap_or_default();
            }
        }
        vec![tok.clone()]
    }

    /// Expand a predefined macro dynamically. Returns the replacement tokens.
    fn expand_predefined(&mut self, sym: Symbol, span: Span) -> Vec<Token> {
        let name = self.interner.resolve(sym).to_string();
        match name.as_str() {
            "__FILE__" => {
                // Look up file name from span's file_id.
                let file_name = if span != Span::DUMMY {
                    let fid = FileId(span.file_id);
                    let file = self.source_map.get_file(fid);
                    file.name.clone()
                } else {
                    "<unknown>".to_string()
                };
                vec![Token::new(
                    TokenKind::StringLiteral {
                        value: file_name.into_bytes(),
                        prefix: crate::frontend::lexer::token::StringPrefix::None,
                    },
                    span,
                )]
            }
            "__LINE__" => {
                let line = if span != Span::DUMMY {
                    let fid = FileId(span.file_id);
                    let loc = self.source_map.lookup_location(fid, span.start);
                    loc.line
                } else {
                    1
                };
                vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: line as u128,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                    },
                    span,
                )]
            }
            "__DATE__" => {
                // Static date string — compile date placeholder.
                vec![Token::new(
                    TokenKind::StringLiteral {
                        value: b"Jan  1 2025".to_vec(),
                        prefix: crate::frontend::lexer::token::StringPrefix::None,
                    },
                    span,
                )]
            }
            "__TIME__" => {
                vec![Token::new(
                    TokenKind::StringLiteral {
                        value: b"00:00:00".to_vec(),
                        prefix: crate::frontend::lexer::token::StringPrefix::None,
                    },
                    span,
                )]
            }
            "__STDC__" => {
                vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: 1,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                    },
                    span,
                )]
            }
            "__STDC_VERSION__" => {
                // C11: 201112L
                vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: 201112,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::L,
                    },
                    span,
                )]
            }
            "__STDC_HOSTED__" => {
                vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: 1,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                    },
                    span,
                )]
            }
            "__COUNTER__" => {
                // Monotonically increasing counter.
                static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let val = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: val as u128,
                        suffix: crate::frontend::lexer::token::IntegerSuffix::None,
                    },
                    span,
                )]
            }
            _ => {
                // Architecture-specific predefined: already stored as literal
                // in the macro table body.
                if let Some(def) = self.macros.get(&sym) {
                    def.body.clone()
                } else {
                    Vec::new()
                }
            }
        }
    }

    // ─── Utility helpers ─────────────────────────────────────────────────

    /// Get the textual representation of a token for identifier comparison.
    fn token_text(&self, tok: &Token) -> String {
        match &tok.kind {
            TokenKind::Identifier(sym) => self.interner.resolve(*sym).to_string(),
            TokenKind::IntegerLiteral { value, .. } => value.to_string(),
            TokenKind::StringLiteral { ref value, .. } => {
                String::from_utf8_lossy(value).to_string()
            }
            _ => format!("{}", tok.kind),
        }
    }

    /// Concatenate the textual representation of a slice of tokens into a
    /// single string (for `#error` / `#warning` messages).
    fn concat_token_text(&self, tokens: &[Token]) -> String {
        let mut result = String::new();
        for tok in tokens {
            if tok.kind == TokenKind::Whitespace {
                result.push(' ');
            } else {
                result.push_str(&self.token_text(tok));
            }
        }
        result
    }
} // end impl Preprocessor

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1 — Trigraph Replacement and Line Splicing (free function)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply C11 Phase 1 transformations to raw source text:
///
/// 1. **Trigraph replacement** — three-character sequences starting with `??`
///    are replaced with their single-character equivalents:
///
///    | Trigraph | Replacement |
///    |----------|-------------|
///    | `??=`    | `#`         |
///    | `??(`    | `[`         |
///    | `??/`    | `\`         |
///    | `??)`    | `]`         |
///    | `??'`    | `^`         |
///    | `??<`    | `{`         |
///    | `??!`    | `\|`        |
///    | `??>`    | `}`         |
///    | `??-`    | `~`         |
///
/// 2. **Line splicing** — a backslash immediately followed by a newline
///    (`\<newline>`) is removed, concatenating the next physical line into
///    the current logical line. Both `\n` and `\r\n` line endings are handled.
///
/// This function runs **before** any other processing on the raw source text.
pub fn phase1_trigraphs_and_line_splice(source: &str) -> String {
    // Pass 1: trigraph replacement.
    let after_trigraphs = replace_trigraphs(source);
    // Pass 2: line splicing.
    splice_lines(&after_trigraphs)
}

/// Replace all trigraph sequences in the source text.
fn replace_trigraphs(source: &str) -> String {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if i + 2 < len && bytes[i] == b'?' && bytes[i + 1] == b'?' {
            let replacement = match bytes[i + 2] {
                b'=' => Some('#'),
                b'(' => Some('['),
                b'/' => Some('\\'),
                b')' => Some(']'),
                b'\'' => Some('^'),
                b'<' => Some('{'),
                b'!' => Some('|'),
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
        // Safety: we only advance through valid UTF-8 chars.
        let ch = source[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }

    result
}

/// Remove backslash-newline sequences (line splicing), handling both
/// Unix (`\n`) and Windows (`\r\n`) line endings.
fn splice_lines(source: &str) -> String {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\\' {
            // Check for `\<newline>`.
            if i + 1 < len && bytes[i + 1] == b'\n' {
                // Skip `\` and `\n`.
                i += 2;
                continue;
            }
            // Check for `\<CR><LF>` (Windows).
            if i + 2 < len && bytes[i + 1] == b'\r' && bytes[i + 2] == b'\n' {
                // Skip `\`, `\r`, and `\n`.
                i += 3;
                continue;
            }
        }
        let ch = source[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Lightweight Preprocessor Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

/// Tokenize source text into a preprocessing token stream.
///
/// This is a lightweight tokenizer used internally by the preprocessor, distinct
/// from the full lexer in `src/frontend/lexer/`. It produces `Token` values
/// using `TokenKind` variants sufficient for directive processing and macro
/// expansion. It does **not** perform keyword recognition (all words are
/// `Identifier`), and it preserves `Whitespace` and `Newline` tokens so the
/// preprocessor can detect directive boundaries.
fn pp_tokenize(source: &str, file_id: u32, interner: &mut Interner) -> Vec<Token> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut tokens: Vec<Token> = Vec::new();
    let mut pos: u32 = 0;

    while (pos as usize) < len {
        let start = pos;
        let b = bytes[pos as usize];

        // ── Newline ──────────────────────────────────────────────
        if b == b'\n' {
            tokens.push(Token::new(
                TokenKind::Newline,
                Span::new(file_id, start, start + 1),
            ));
            pos += 1;
            continue;
        }
        if b == b'\r' {
            // CR or CRLF → single newline token.
            pos += 1;
            if (pos as usize) < len && bytes[pos as usize] == b'\n' {
                pos += 1;
            }
            tokens.push(Token::new(
                TokenKind::Newline,
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── Whitespace (spaces, tabs) ────────────────────────────
        if b == b' ' || b == b'\t' || b == b'\x0C' {
            pos += 1;
            while (pos as usize) < len {
                let c = bytes[pos as usize];
                if c == b' ' || c == b'\t' || c == b'\x0C' {
                    pos += 1;
                } else {
                    break;
                }
            }
            tokens.push(Token::new(
                TokenKind::Whitespace,
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── Line comments `//` ───────────────────────────────────
        if b == b'/' && (pos as usize) + 1 < len && bytes[(pos as usize) + 1] == b'/' {
            pos += 2;
            while (pos as usize) < len && bytes[pos as usize] != b'\n' {
                pos += 1;
            }
            // Emit whitespace in place of the comment.
            tokens.push(Token::new(
                TokenKind::Whitespace,
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── Block comments `/* ... */` ───────────────────────────
        if b == b'/' && (pos as usize) + 1 < len && bytes[(pos as usize) + 1] == b'*' {
            pos += 2;
            let mut found_end = false;
            while (pos as usize) + 1 < len {
                if bytes[pos as usize] == b'*' && bytes[(pos as usize) + 1] == b'/' {
                    pos += 2;
                    found_end = true;
                    break;
                }
                pos += 1;
            }
            if !found_end {
                // Unterminated block comment — consume rest.
                pos = len as u32;
            }
            tokens.push(Token::new(
                TokenKind::Whitespace,
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── String literals ──────────────────────────────────────
        if b == b'"' {
            pos += 1;
            let mut content = String::new();
            while (pos as usize) < len {
                let c = bytes[pos as usize];
                if c == b'"' {
                    pos += 1;
                    break;
                }
                if c == b'\\' && (pos as usize) + 1 < len {
                    // Escape sequence — include both chars in content.
                    content.push(source[pos as usize..].chars().next().unwrap());
                    pos += 1;
                    content.push(source[pos as usize..].chars().next().unwrap());
                    pos += 1;
                    continue;
                }
                if c == b'\n' {
                    break; // Unterminated string on this line.
                }
                let ch = source[pos as usize..].chars().next().unwrap();
                content.push(ch);
                pos += ch.len_utf8() as u32;
            }
            tokens.push(Token::new(
                TokenKind::StringLiteral {
                    value: content.into_bytes(),
                    prefix: crate::frontend::lexer::token::StringPrefix::None,
                },
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── Character literals ───────────────────────────────────
        if b == b'\'' {
            pos += 1;
            let mut value: u128 = 0;
            while (pos as usize) < len {
                let c = bytes[pos as usize];
                if c == b'\'' {
                    pos += 1;
                    break;
                }
                if c == b'\\' && (pos as usize) + 1 < len {
                    pos += 1;
                    let esc = bytes[pos as usize];
                    value = match esc {
                        b'n' => b'\n' as u128,
                        b't' => b'\t' as u128,
                        b'r' => b'\r' as u128,
                        b'0' => 0,
                        b'\\' => b'\\' as u128,
                        b'\'' => b'\'' as u128,
                        b'"' => b'"' as u128,
                        b'a' => 0x07,
                        b'b' => 0x08,
                        b'f' => 0x0C,
                        b'v' => 0x0B,
                        _ => esc as u128,
                    };
                    pos += 1;
                    continue;
                }
                if c == b'\n' {
                    break;
                }
                value = c as u128;
                pos += 1;
            }
            tokens.push(Token::new(
                TokenKind::CharLiteral {
                    value: value as u32,
                    prefix: crate::frontend::lexer::token::CharPrefix::None,
                },
                Span::new(file_id, start, pos),
            ));
            continue;
        }

        // ── Numeric literals (decimal, hex, octal, binary, float) ─
        if b.is_ascii_digit()
            || (b == b'.' && (pos as usize) + 1 < len && bytes[(pos as usize) + 1].is_ascii_digit())
        {
            let (tok, new_pos) = lex_number(source, pos, file_id, interner);
            tokens.push(tok);
            pos = new_pos;
            continue;
        }

        // ── Identifiers and keywords ─────────────────────────────
        if is_ident_start(b) || (b >= 0x80) {
            let id_start = pos;
            // Advance through identifier characters.
            let ch = source[pos as usize..].chars().next().unwrap();
            pos += ch.len_utf8() as u32;
            while (pos as usize) < len {
                let c = bytes[pos as usize];
                if is_ident_continue(c) || c >= 0x80 {
                    let ch2 = source[pos as usize..].chars().next().unwrap();
                    pos += ch2.len_utf8() as u32;
                } else {
                    break;
                }
            }
            let text = &source[id_start as usize..pos as usize];
            let sym = interner.intern(text);
            tokens.push(Token::new(
                TokenKind::Identifier(sym),
                Span::new(file_id, id_start, pos),
            ));
            continue;
        }

        // ── Punctuators / operators ──────────────────────────────
        let (tok_kind, advance) = lex_punctuator(bytes, pos as usize, len);
        tokens.push(Token::new(
            tok_kind,
            Span::new(file_id, start, start + advance as u32),
        ));
        pos += advance as u32;
    }

    // Append EOF.
    tokens.push(Token::new(TokenKind::Eof, Span::new(file_id, pos, pos)));
    tokens
}

/// Returns `true` if the byte can start a C identifier (letter or `_`).
#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Returns `true` if the byte can continue a C identifier (letter, digit, or `_`).
#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Lex a numeric literal starting at `pos`. Returns `(Token, new_pos)`.
fn lex_number(source: &str, start: u32, file_id: u32, _interner: &mut Interner) -> (Token, u32) {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut pos = start as usize;
    let mut is_float = false;

    // Detect hex, octal, or binary prefix.
    if bytes[pos] == b'0' && pos + 1 < len {
        match bytes[pos + 1] {
            b'x' | b'X' => {
                // Hex literal.
                pos += 2;
                while pos < len && (bytes[pos].is_ascii_hexdigit() || bytes[pos] == b'_') {
                    pos += 1;
                }
                // Check for hex float exponent.
                if pos < len && (bytes[pos] == b'p' || bytes[pos] == b'P') {
                    is_float = true;
                    pos += 1;
                    if pos < len && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                        pos += 1;
                    }
                    while pos < len && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
                if pos < len && bytes[pos] == b'.' {
                    is_float = true;
                    pos += 1;
                    while pos < len && bytes[pos].is_ascii_hexdigit() {
                        pos += 1;
                    }
                    if pos < len && (bytes[pos] == b'p' || bytes[pos] == b'P') {
                        pos += 1;
                        if pos < len && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                            pos += 1;
                        }
                        while pos < len && bytes[pos].is_ascii_digit() {
                            pos += 1;
                        }
                    }
                }
            }
            b'b' | b'B' => {
                // Binary literal.
                pos += 2;
                while pos < len && (bytes[pos] == b'0' || bytes[pos] == b'1' || bytes[pos] == b'_')
                {
                    pos += 1;
                }
            }
            _ => {
                // Octal or decimal starting with 0, or just `0`.
                while pos < len && (bytes[pos].is_ascii_digit() || bytes[pos] == b'_') {
                    pos += 1;
                }
                if pos < len && bytes[pos] == b'.' {
                    is_float = true;
                    pos += 1;
                    while pos < len && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
                if pos < len && (bytes[pos] == b'e' || bytes[pos] == b'E') {
                    is_float = true;
                    pos += 1;
                    if pos < len && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                        pos += 1;
                    }
                    while pos < len && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
            }
        }
    } else {
        // Decimal literal.
        while pos < len && (bytes[pos].is_ascii_digit() || bytes[pos] == b'_') {
            pos += 1;
        }
        if pos < len && bytes[pos] == b'.' {
            is_float = true;
            pos += 1;
            while pos < len && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
        }
        if pos < len && (bytes[pos] == b'e' || bytes[pos] == b'E') {
            is_float = true;
            pos += 1;
            if pos < len && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                pos += 1;
            }
            while pos < len && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
        }
    }

    // Consume integer/float suffixes.
    let suffix_start = pos;
    while pos < len && (bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_') {
        pos += 1;
    }

    let text = &source[start as usize..pos];
    let span = Span::new(file_id, start, pos as u32);

    if is_float {
        // Parse as float literal.
        let clean: String = text.chars().filter(|c| *c != '_').collect();
        let fval = clean.parse::<f64>().unwrap_or(0.0);
        let suffix_text = &source[suffix_start..pos];
        let float_suffix = match suffix_text {
            "f" | "F" => crate::frontend::lexer::token::FloatSuffix::F,
            "l" | "L" => crate::frontend::lexer::token::FloatSuffix::L,
            _ => crate::frontend::lexer::token::FloatSuffix::None,
        };
        (
            Token::new(
                TokenKind::FloatLiteral {
                    value: fval,
                    suffix: float_suffix,
                },
                span,
            ),
            pos as u32,
        )
    } else {
        // Parse as integer literal.
        let num_text = &source[start as usize..suffix_start];
        let clean: String = num_text.chars().filter(|c| *c != '_').collect();
        let value = parse_int_value(&clean);
        let suffix_text = &source[suffix_start..pos];
        let int_suffix = parse_int_suffix(suffix_text);
        (
            Token::new(
                TokenKind::IntegerLiteral {
                    value,
                    suffix: int_suffix,
                },
                span,
            ),
            pos as u32,
        )
    }
}

/// Parse an integer value from its string representation, handling hex,
/// octal, binary, and decimal bases.
fn parse_int_value(s: &str) -> u128 {
    if s.is_empty() {
        return 0;
    }
    if s.starts_with("0x") || s.starts_with("0X") {
        u128::from_str_radix(&s[2..], 16).unwrap_or(0)
    } else if s.starts_with("0b") || s.starts_with("0B") {
        u128::from_str_radix(&s[2..], 2).unwrap_or(0)
    } else if s.starts_with('0') && s.len() > 1 && s.chars().all(|c| c.is_ascii_digit()) {
        u128::from_str_radix(&s[1..], 8).unwrap_or(0)
    } else {
        s.parse::<u128>().unwrap_or(0)
    }
}

/// Parse an integer suffix string into the `IntegerSuffix` enum.
fn parse_int_suffix(s: &str) -> crate::frontend::lexer::token::IntegerSuffix {
    use crate::frontend::lexer::token::IntegerSuffix;
    let lower = s.to_ascii_lowercase();
    match lower.as_str() {
        "" => IntegerSuffix::None,
        "u" => IntegerSuffix::U,
        "l" => IntegerSuffix::L,
        "ul" | "lu" => IntegerSuffix::UL,
        "ll" => IntegerSuffix::LL,
        "ull" | "llu" => IntegerSuffix::ULL,
        _ => IntegerSuffix::None,
    }
}

/// Lex a punctuator starting at `pos` in `bytes`. Returns `(TokenKind, advance)`.
fn lex_punctuator(bytes: &[u8], pos: usize, len: usize) -> (TokenKind, usize) {
    // Try 3-character punctuators first, then 2-character, then 1-character.
    if pos + 2 < len {
        let triple = &bytes[pos..pos + 3];
        let kind = match triple {
            b"<<=" => Some(TokenKind::LeftShiftAssign),
            b">>=" => Some(TokenKind::RightShiftAssign),
            b"..." => Some(TokenKind::Ellipsis),
            _ => None,
        };
        if let Some(k) = kind {
            return (k, 3);
        }
    }

    if pos + 1 < len {
        let double = &bytes[pos..pos + 2];
        let kind = match double {
            b"+=" => Some(TokenKind::PlusAssign),
            b"-=" => Some(TokenKind::MinusAssign),
            b"*=" => Some(TokenKind::StarAssign),
            b"/=" => Some(TokenKind::SlashAssign),
            b"%=" => Some(TokenKind::PercentAssign),
            b"&=" => Some(TokenKind::AmpAssign),
            b"|=" => Some(TokenKind::PipeAssign),
            b"^=" => Some(TokenKind::CaretAssign),
            b"==" => Some(TokenKind::EqualEqual),
            b"!=" => Some(TokenKind::NotEqual),
            b"<=" => Some(TokenKind::LessEqual),
            b">=" => Some(TokenKind::GreaterEqual),
            b"&&" => Some(TokenKind::AmpAmp),
            b"||" => Some(TokenKind::PipePipe),
            b"++" => Some(TokenKind::PlusPlus),
            b"--" => Some(TokenKind::MinusMinus),
            b"->" => Some(TokenKind::Arrow),
            b"<<" => Some(TokenKind::LeftShift),
            b">>" => Some(TokenKind::RightShift),
            b"##" => Some(TokenKind::HashHash),
            _ => None,
        };
        if let Some(k) = kind {
            return (k, 2);
        }
    }

    // Single-character punctuators.
    let kind = match bytes[pos] {
        b'(' => TokenKind::LeftParen,
        b')' => TokenKind::RightParen,
        b'[' => TokenKind::LeftBracket,
        b']' => TokenKind::RightBracket,
        b'{' => TokenKind::LeftBrace,
        b'}' => TokenKind::RightBrace,
        b';' => TokenKind::Semicolon,
        b',' => TokenKind::Comma,
        b'.' => TokenKind::Dot,
        b'~' => TokenKind::Tilde,
        b'?' => TokenKind::Question,
        b':' => TokenKind::Colon,
        b'+' => TokenKind::Plus,
        b'-' => TokenKind::Minus,
        b'*' => TokenKind::Star,
        b'/' => TokenKind::Slash,
        b'%' => TokenKind::Percent,
        b'&' => TokenKind::Ampersand,
        b'|' => TokenKind::Pipe,
        b'^' => TokenKind::Caret,
        b'!' => TokenKind::Exclaim,
        b'=' => TokenKind::Assign,
        b'<' => TokenKind::Less,
        b'>' => TokenKind::Greater,
        b'#' => TokenKind::Hash,
        _ => {
            // Unknown character — emit as an error token or skip.
            // For robustness, skip unknown bytes.
            return (TokenKind::Whitespace, 1);
        }
    };
    (kind, 1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: skip leading whitespace tokens
// ═══════════════════════════════════════════════════════════════════════════

/// Return a subslice of `tokens` with leading whitespace tokens removed.
fn skip_ws(tokens: &[Token]) -> &[Token] {
    let mut i = 0;
    while i < tokens.len() && tokens[i].kind == TokenKind::Whitespace {
        i += 1;
    }
    &tokens[i..]
}

// ═══════════════════════════════════════════════════════════════════════════
// Preprocessor Expression Evaluator
// ═══════════════════════════════════════════════════════════════════════════

/// A minimal recursive-descent evaluator for preprocessor `#if`/`#elif`
/// integer constant expressions. Supports:
/// - Integer literals (decimal, hex, octal)
/// - Unary: `+`, `-`, `!`, `~`
/// - Binary: `*`, `/`, `%`, `+`, `-`, `<<`, `>>`, `<`, `<=`, `>`, `>=`,
///   `==`, `!=`, `&`, `^`, `|`, `&&`, `||`
/// - Ternary: `? :`
/// - Parenthesized sub-expressions
///
/// All arithmetic uses `i64` (matching C preprocessor semantics).
struct CondExprEvaluator<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> CondExprEvaluator<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&TokenKind> {
        let mut i = self.pos;
        while i < self.tokens.len() {
            if self.tokens[i].kind == TokenKind::Whitespace {
                i += 1;
                continue;
            }
            return Some(&self.tokens[i].kind);
        }
        None
    }

    fn advance(&mut self) -> Option<&Token> {
        while self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            self.pos += 1;
            if tok.kind == TokenKind::Whitespace {
                continue;
            }
            return Some(tok);
        }
        None
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<(), String> {
        match self.peek() {
            Some(k) if std::mem::discriminant(k) == std::mem::discriminant(kind) => {
                self.advance();
                Ok(())
            }
            other => Err(format!("expected {:?}, found {:?}", kind, other)),
        }
    }

    // ── Ternary ──────────────────────────────────────────────────────

    fn eval_ternary(&mut self) -> Result<i64, String> {
        let cond = self.eval_logical_or()?;
        if self.peek() == Some(&TokenKind::Question) {
            self.advance(); // consume '?'
            let then_val = self.eval_ternary()?;
            self.expect(&TokenKind::Colon)?;
            let else_val = self.eval_ternary()?;
            Ok(if cond != 0 { then_val } else { else_val })
        } else {
            Ok(cond)
        }
    }

    // ── Logical OR ───────────────────────────────────────────────────

    fn eval_logical_or(&mut self) -> Result<i64, String> {
        let mut val = self.eval_logical_and()?;
        while self.peek() == Some(&TokenKind::PipePipe) {
            self.advance();
            let rhs = self.eval_logical_and()?;
            val = if val != 0 || rhs != 0 { 1 } else { 0 };
        }
        Ok(val)
    }

    // ── Logical AND ──────────────────────────────────────────────────

    fn eval_logical_and(&mut self) -> Result<i64, String> {
        let mut val = self.eval_bitwise_or()?;
        while self.peek() == Some(&TokenKind::AmpAmp) {
            self.advance();
            let rhs = self.eval_bitwise_or()?;
            val = if val != 0 && rhs != 0 { 1 } else { 0 };
        }
        Ok(val)
    }

    // ── Bitwise OR ───────────────────────────────────────────────────

    fn eval_bitwise_or(&mut self) -> Result<i64, String> {
        let mut val = self.eval_bitwise_xor()?;
        while self.peek() == Some(&TokenKind::Pipe) {
            self.advance();
            let rhs = self.eval_bitwise_xor()?;
            val |= rhs;
        }
        Ok(val)
    }

    // ── Bitwise XOR ──────────────────────────────────────────────────

    fn eval_bitwise_xor(&mut self) -> Result<i64, String> {
        let mut val = self.eval_bitwise_and()?;
        while self.peek() == Some(&TokenKind::Caret) {
            self.advance();
            let rhs = self.eval_bitwise_and()?;
            val ^= rhs;
        }
        Ok(val)
    }

    // ── Bitwise AND ──────────────────────────────────────────────────

    fn eval_bitwise_and(&mut self) -> Result<i64, String> {
        let mut val = self.eval_equality()?;
        while self.peek() == Some(&TokenKind::Ampersand) {
            self.advance();
            let rhs = self.eval_equality()?;
            val &= rhs;
        }
        Ok(val)
    }

    // ── Equality ─────────────────────────────────────────────────────

    fn eval_equality(&mut self) -> Result<i64, String> {
        let mut val = self.eval_relational()?;
        loop {
            match self.peek() {
                Some(&TokenKind::EqualEqual) => {
                    self.advance();
                    let rhs = self.eval_relational()?;
                    val = if val == rhs { 1 } else { 0 };
                }
                Some(&TokenKind::NotEqual) => {
                    self.advance();
                    let rhs = self.eval_relational()?;
                    val = if val != rhs { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(val)
    }

    // ── Relational ───────────────────────────────────────────────────

    fn eval_relational(&mut self) -> Result<i64, String> {
        let mut val = self.eval_shift()?;
        loop {
            match self.peek() {
                Some(&TokenKind::Less) => {
                    self.advance();
                    let rhs = self.eval_shift()?;
                    val = if val < rhs { 1 } else { 0 };
                }
                Some(&TokenKind::LessEqual) => {
                    self.advance();
                    let rhs = self.eval_shift()?;
                    val = if val <= rhs { 1 } else { 0 };
                }
                Some(&TokenKind::Greater) => {
                    self.advance();
                    let rhs = self.eval_shift()?;
                    val = if val > rhs { 1 } else { 0 };
                }
                Some(&TokenKind::GreaterEqual) => {
                    self.advance();
                    let rhs = self.eval_shift()?;
                    val = if val >= rhs { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(val)
    }

    // ── Shift ────────────────────────────────────────────────────────

    fn eval_shift(&mut self) -> Result<i64, String> {
        let mut val = self.eval_additive()?;
        loop {
            match self.peek() {
                Some(&TokenKind::LeftShift) => {
                    self.advance();
                    let rhs = self.eval_additive()?;
                    val = val.wrapping_shl(rhs as u32);
                }
                Some(&TokenKind::RightShift) => {
                    self.advance();
                    let rhs = self.eval_additive()?;
                    val = val.wrapping_shr(rhs as u32);
                }
                _ => break,
            }
        }
        Ok(val)
    }

    // ── Additive ─────────────────────────────────────────────────────

    fn eval_additive(&mut self) -> Result<i64, String> {
        let mut val = self.eval_multiplicative()?;
        loop {
            match self.peek() {
                Some(&TokenKind::Plus) => {
                    self.advance();
                    let rhs = self.eval_multiplicative()?;
                    val = val.wrapping_add(rhs);
                }
                Some(&TokenKind::Minus) => {
                    self.advance();
                    let rhs = self.eval_multiplicative()?;
                    val = val.wrapping_sub(rhs);
                }
                _ => break,
            }
        }
        Ok(val)
    }

    // ── Multiplicative ───────────────────────────────────────────────

    fn eval_multiplicative(&mut self) -> Result<i64, String> {
        let mut val = self.eval_unary()?;
        loop {
            match self.peek() {
                Some(&TokenKind::Star) => {
                    self.advance();
                    let rhs = self.eval_unary()?;
                    val = val.wrapping_mul(rhs);
                }
                Some(&TokenKind::Slash) => {
                    self.advance();
                    let rhs = self.eval_unary()?;
                    if rhs == 0 {
                        return Err("division by zero".to_string());
                    }
                    val = val.wrapping_div(rhs);
                }
                Some(&TokenKind::Percent) => {
                    self.advance();
                    let rhs = self.eval_unary()?;
                    if rhs == 0 {
                        return Err("modulo by zero".to_string());
                    }
                    val = val.wrapping_rem(rhs);
                }
                _ => break,
            }
        }
        Ok(val)
    }

    // ── Unary ────────────────────────────────────────────────────────

    fn eval_unary(&mut self) -> Result<i64, String> {
        match self.peek() {
            Some(&TokenKind::Plus) => {
                self.advance();
                self.eval_unary()
            }
            Some(&TokenKind::Minus) => {
                self.advance();
                let val = self.eval_unary()?;
                Ok(val.wrapping_neg())
            }
            Some(&TokenKind::Exclaim) => {
                self.advance();
                let val = self.eval_unary()?;
                Ok(if val == 0 { 1 } else { 0 })
            }
            Some(&TokenKind::Tilde) => {
                self.advance();
                let val = self.eval_unary()?;
                Ok(!val)
            }
            _ => self.eval_primary(),
        }
    }

    // ── Primary ──────────────────────────────────────────────────────

    fn eval_primary(&mut self) -> Result<i64, String> {
        // Parenthesized expression.
        if self.peek() == Some(&TokenKind::LeftParen) {
            self.advance();
            let val = self.eval_ternary()?;
            self.expect(&TokenKind::RightParen)?;
            return Ok(val);
        }

        // Integer literal.
        match self.advance() {
            Some(tok) => match &tok.kind {
                TokenKind::IntegerLiteral { value, .. } => Ok(*value as i64),
                TokenKind::CharLiteral { value, .. } => Ok(*value as i64),
                TokenKind::Eof => Ok(0),
                other => Err(format!("unexpected token in expression: {:?}", other)),
            },
            None => Ok(0),
        }
    }
}
