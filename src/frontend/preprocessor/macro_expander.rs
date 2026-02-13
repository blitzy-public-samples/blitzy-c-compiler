//! Macro expansion engine for the BCC preprocessor (Phase 2).
//!
//! This module implements the core macro expansion algorithm for the C
//! preprocessor. When a macro invocation is encountered in the token stream,
//! it substitutes actual arguments into the macro body, applies `#`
//! stringification and `##` token pasting via the [`token_paster`] module,
//! then rescans the result for further macro expansion.
//!
//! # Key Features
//!
//! - **Object-like macros**: `#define PI 3.14` → `PI` expands to `3.14`
//! - **Function-like macros**: `#define MAX(a,b) ((a)>(b)?(a):(b))`
//! - **Variadic macros**: `#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)`
//! - **GCC comma deletion**: `##__VA_ARGS__` with empty variadic deletes preceding comma
//! - **Paint-marker recursion protection**: prevents infinite recursion on
//!   self-referential macros like `#define A A` (C11 §6.10.3.4)
//! - **512-depth recursion limit**: safety net for deeply nested but
//!   non-self-referential expansion chains (Section 0.7.3)
//! - **Argument pre-expansion**: arguments are macro-expanded before substitution
//!   except when used with `#` or `##` operators (C11 §6.10.3.1)
//!
//! # Architecture
//!
//! The expansion engine operates on [`PaintedToken`] vectors internally to
//! track which macros have been expanded at each token position. The public
//! API accepts and returns plain [`Token`] values — paint state is an internal
//! implementation detail of the expansion process.
//!
//! Two recursion-prevention mechanisms cooperate:
//! 1. **Paint markers** (token-level): prevent self-referential expansion.
//!    `#define A A` → paints `A` during expansion → painted `A` not re-expanded.
//! 2. **Depth limit** (global): prevents deeply nested chains from exhausting
//!    the stack. Enforced at 512 per Section 0.7.3.
//!
//! # Zero-Dependency
//!
//! Uses only internal modules. No external crates.

use crate::common::diagnostics::Span;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::{Token, TokenKind};

use super::paint_marker::{
    is_painted, is_painted_for, merge_paint, paint_tokens, PaintedToken,
};
use super::token_paster::{apply_paste_operators, apply_stringify_operators};
use super::{MacroDef, Preprocessor};

// ===========================================================================
// TokenStream — efficient token-by-token processing with lookahead
// ===========================================================================

/// A cursor-based wrapper around a sequence of [`PaintedToken`] values,
/// providing efficient sequential access with lookahead and save/restore
/// for tentative parsing of macro arguments.
///
/// `TokenStream` is the primary abstraction used by the macro expansion
/// engine to process token sequences. It supports:
///
/// - **Sequential access**: `next()` advances the cursor and returns the
///   current token.
/// - **Lookahead**: `peek()` and `peek_ahead()` inspect tokens without
///   advancing.
/// - **Backtracking**: `save()` / `restore()` checkpoints for tentative
///   parsing (e.g., checking if a function-like macro name is followed
///   by `(`).
///
/// # Internal Representation
///
/// Tokens are stored as [`PaintedToken`] values — each carrying its
/// paint state for macro recursion protection. The public constructors
/// accept plain `Token` values and wrap them as unpainted.
pub struct TokenStream {
    /// The underlying painted token buffer.
    tokens: Vec<PaintedToken>,
    /// Current read position (index into `tokens`).
    pos: usize,
    /// Stack of saved positions for backtracking.
    saved: Vec<usize>,
}

impl TokenStream {
    /// Creates a new `TokenStream` from a vector of plain tokens.
    ///
    /// Each token is wrapped with [`PaintState::Unpainted`] since no
    /// macro expansion has occurred yet.
    ///
    /// # Arguments
    ///
    /// * `tokens` — The token sequence to wrap.
    pub fn new(tokens: Vec<Token>) -> Self {
        TokenStream {
            tokens: tokens.into_iter().map(PaintedToken::new).collect(),
            pos: 0,
            saved: Vec::new(),
        }
    }

    /// Creates a `TokenStream` from pre-painted tokens.
    ///
    /// Used internally by the expansion engine when rescanning replacement
    /// lists that already carry paint state from prior expansion.
    #[allow(dead_code)]
    pub(crate) fn from_painted(tokens: Vec<PaintedToken>) -> Self {
        TokenStream {
            tokens,
            pos: 0,
            saved: Vec::new(),
        }
    }

    /// Advances the cursor by one position and returns the token at the
    /// previous position, or `None` if at end-of-stream.
    pub fn next(&mut self) -> Option<PaintedToken> {
        if self.pos < self.tokens.len() {
            let pt = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(pt)
        } else {
            None
        }
    }

    /// Returns a reference to the token at the current cursor position
    /// without advancing, or `None` if at end-of-stream.
    pub fn peek(&self) -> Option<&PaintedToken> {
        self.tokens.get(self.pos)
    }

    /// Returns a reference to the token `n` positions ahead of the
    /// current cursor, or `None` if that position is beyond the end.
    ///
    /// `peek_ahead(0)` is equivalent to `peek()`.
    pub fn peek_ahead(&self, n: usize) -> Option<&PaintedToken> {
        self.tokens.get(self.pos.checked_add(n)?)
    }

    /// Returns `true` if the cursor has reached or passed the end of the
    /// token sequence.
    pub fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Pushes the current cursor position onto the save stack.
    ///
    /// The saved position can later be restored with [`restore()`] for
    /// backtracking. Multiple saves nest as a stack (LIFO).
    pub fn save(&mut self) {
        self.saved.push(self.pos);
    }

    /// Pops the most recently saved cursor position and resets the cursor
    /// to that position.
    ///
    /// If no positions have been saved, this is a no-op.
    pub fn restore(&mut self) {
        if let Some(saved_pos) = self.saved.pop() {
            self.pos = saved_pos;
        }
    }

    /// Returns a slice of the remaining (unread) painted tokens starting
    /// from the current cursor position.
    pub fn remaining(&self) -> &[PaintedToken] {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos..]
        } else {
            &[]
        }
    }

    /// Consumes all remaining tokens from the current position onward,
    /// returning them as a `Vec<PaintedToken>` and advancing the cursor
    /// to end-of-stream.
    fn drain_remaining(&mut self) -> Vec<PaintedToken> {
        if self.pos < self.tokens.len() {
            let drained = self.tokens[self.pos..].to_vec();
            self.pos = self.tokens.len();
            drained
        } else {
            Vec::new()
        }
    }

    /// Returns the current cursor position (for diagnostic or debugging
    /// purposes).
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns the total number of tokens in the stream (including
    /// already-consumed tokens).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Replaces the remaining tokens from the current position with the
    /// given painted tokens and resets the cursor to the replacement start.
    ///
    /// This is used during rescanning: after expanding a macro, the
    /// expanded replacement is spliced into the stream for further
    /// processing.
    #[allow(dead_code)]
    pub(crate) fn splice_at_current(&mut self, replacement: Vec<PaintedToken>) {
        let before = self.tokens[..self.pos].to_vec();
        let rest_start = self.pos;
        let old_rest = if rest_start < self.tokens.len() {
            self.tokens[rest_start..].to_vec()
        } else {
            Vec::new()
        };
        self.tokens = before;
        self.tokens.extend(replacement);
        self.tokens.extend(old_rest);
        // Cursor stays at self.pos — rescan starts from here.
    }
}

// ===========================================================================
// Public API — expand_macros
// ===========================================================================

/// Expands all macro invocations in a token sequence.
///
/// This is the primary entry point for Phase 2 macro expansion. Takes a
/// slice of plain [`Token`] values (typically from a single preprocessor
/// logical line), expands all macro invocations recursively, and returns
/// the fully expanded token sequence.
///
/// # Algorithm
///
/// 1. Wraps each input token as an unpainted [`PaintedToken`].
/// 2. Scans left-to-right for identifier tokens matching defined macros.
/// 3. For each match: checks paint state, enforces recursion depth limit,
///    expands the macro (substituting arguments, applying `#`/`##`),
///    paints the result, and splices back for rescanning.
/// 4. Strips paint state and returns plain `Token` values.
///
/// # Arguments
///
/// * `pp` — Mutable reference to the preprocessor state (macro table,
///   interner, diagnostics, recursion depth).
/// * `tokens` — The token sequence to expand.
///
/// # Returns
///
/// The fully expanded token sequence with all macro invocations resolved.
/// On fatal error (recursion depth exceeded), returns the partially expanded
/// sequence up to the point of failure.
pub fn expand_macros(pp: &mut Preprocessor, tokens: &[Token]) -> Vec<Token> {
    let painted: Vec<PaintedToken> = tokens.iter().cloned().map(PaintedToken::new).collect();
    let result = expand_painted(pp, painted);
    result.into_iter().map(|pt| pt.token).collect()
}

// ===========================================================================
// Public API — expand_token_sequence
// ===========================================================================

/// Expands all macro invocations using a mutable [`TokenStream`] for
/// incremental processing.
///
/// Consumes the remaining tokens from the stream, expands all macro
/// invocations, and returns the fully expanded token sequence. The
/// stream's cursor is advanced to end-of-stream.
///
/// This function is suitable for contexts where tokens are produced
/// incrementally and expansion must interleave with token generation.
///
/// # Arguments
///
/// * `pp` — Mutable reference to the preprocessor state.
/// * `stream` — Mutable reference to the token stream to process.
///
/// # Returns
///
/// The fully expanded token sequence.
pub fn expand_token_sequence(pp: &mut Preprocessor, stream: &mut TokenStream) -> Vec<Token> {
    let remaining = stream.drain_remaining();
    let result = expand_painted(pp, remaining);
    result.into_iter().map(|pt| pt.token).collect()
}

// ===========================================================================
// Core expansion loop
// ===========================================================================

/// Core expansion loop operating on [`PaintedToken`] vectors.
///
/// This is the heart of the macro expansion engine. It implements the
/// rescanning algorithm required by C11 §6.10.3.4:
///
/// 1. Scan tokens left-to-right.
/// 2. When an identifier matches a macro:
///    a. Check paint → if painted for this macro, skip (C11 §6.10.3.4).
///    b. Check recursion depth → if exceeded, emit error and skip.
///    c. For predefined macros → expand dynamically.
///    d. For function-like → find `(`, collect args, substitute, paint, rescan.
///    e. For object-like → substitute body, paint, rescan.
/// 3. Non-macro tokens pass through to output.
///
/// Rescanning is implemented by splicing expanded tokens back into the
/// working vector at the current position and NOT advancing the index,
/// so the expanded tokens are immediately subjected to further expansion.
fn expand_painted(pp: &mut Preprocessor, mut tokens: Vec<PaintedToken>) -> Vec<PaintedToken> {
    let mut output: Vec<PaintedToken> = Vec::new();
    let mut i: usize = 0;

    while i < tokens.len() {
        // Only try to expand identifier tokens.
        let sym = match tokens[i].token.kind {
            TokenKind::Identifier(s) => s,
            _ => {
                // Not an identifier — pass through unchanged.
                output.push(tokens[i].clone());
                i += 1;
                continue;
            }
        };

        // Check paint state: if this token is painted for the macro it
        // names, do NOT expand — treat as ordinary identifier (C11 §6.10.3.4).
        if is_painted_for(&tokens[i], sym) {
            output.push(tokens[i].clone());
            i += 1;
            continue;
        }

        // Look up macro definition. Clone to release borrow on pp.macros
        // before we need &mut pp for expansion.
        let def = match pp.macros.get(&sym).cloned() {
            Some(d) => d,
            None => {
                // Not a defined macro — pass through.
                output.push(tokens[i].clone());
                i += 1;
                continue;
            }
        };

        let invocation_span = tokens[i].token.span;

        // Enforce the 512-depth recursion limit (Section 0.7.3).
        if pp.recursion_depth >= pp.max_recursion_depth {
            pp.diagnostics.error(
                invocation_span,
                format!(
                    "macro expansion depth exceeds limit ({})",
                    pp.max_recursion_depth,
                ),
            );
            // Return unexpanded tokens from this point onward.
            output.push(tokens[i].clone());
            i += 1;
            continue;
        }

        // Increment depth before expansion; decrement in every exit path.
        pp.recursion_depth += 1;

        // ── Predefined macros ──────────────────────────────────────
        if def.is_predefined {
            let expanded = pp.expand_predefined(sym, invocation_span);
            for tok in expanded {
                output.push(PaintedToken::new(tok));
            }
            pp.recursion_depth -= 1;
            i += 1;
            continue;
        }

        // ── Function-like macros ───────────────────────────────────
        if def.is_function_like() {
            let after_name = i + 1;
            match find_open_paren(&tokens, after_name) {
                Some(paren_idx) => {
                    // Collect actual arguments from the invocation.
                    match collect_arguments(pp, &tokens, paren_idx + 1, &def, invocation_span) {
                        Ok((raw_args, end_idx)) => {
                            // Validate argument count.
                            let param_count =
                                def.params.as_ref().map_or(0, |p| p.len());
                            let actual_args = raw_args.len();

                            // For non-variadic: exact match (allow 0 for 0-param macros).
                            // For variadic: at least param_count args.
                            if !def.is_variadic && actual_args != param_count {
                                // Special case: FOO() with 0 params counts as 0 args,
                                // but collect_arguments returns 1 empty arg.
                                let is_zero_arg_call =
                                    param_count == 0
                                        && actual_args == 1
                                        && raw_args[0].is_empty();
                                if !is_zero_arg_call {
                                    pp.diagnostics.error(
                                        invocation_span,
                                        format!(
                                            "macro '{}' requires {} argument{}, but {} {} given",
                                            pp.interner.resolve(sym),
                                            param_count,
                                            if param_count == 1 { "" } else { "s" },
                                            actual_args,
                                            if actual_args == 1 { "was" } else { "were" },
                                        ),
                                    );
                                    output.push(tokens[i].clone());
                                    pp.recursion_depth -= 1;
                                    i += 1;
                                    continue;
                                }
                            }

                            // Perform argument substitution into the macro body.
                            let replacement =
                                substitute_body(pp, &def, &raw_args, invocation_span);

                            // Wrap replacement tokens as painted for this macro.
                            let mut expanded_painted: Vec<PaintedToken> =
                                replacement.into_iter().map(PaintedToken::new).collect();
                            paint_tokens(&mut expanded_painted, sym);

                            // Inherit paint from the invocation token: if the
                            // macro name was already painted for other macros,
                            // propagate that paint to the replacement.
                            if is_painted(&tokens[i]) {
                                let invocation_paint = &tokens[i].paint;
                                for pt in &mut expanded_painted {
                                    pt.paint =
                                        merge_paint(&pt.paint, invocation_paint);
                                }
                            }

                            // Rescan: splice expanded tokens into the working
                            // vector at the current position.
                            let rest = tokens.split_off(end_idx + 1);
                            tokens.truncate(i);
                            tokens.extend(expanded_painted);
                            tokens.extend(rest);

                            pp.recursion_depth -= 1;
                            // Do NOT advance i — rescan from the same position.
                            continue;
                        }
                        Err(()) => {
                            // Argument collection failed (unterminated parens, etc.)
                            output.push(tokens[i].clone());
                            pp.recursion_depth -= 1;
                            i += 1;
                            continue;
                        }
                    }
                }
                None => {
                    // No `(` found after the macro name — not a function-like
                    // invocation, just output the identifier.
                    output.push(tokens[i].clone());
                    pp.recursion_depth -= 1;
                    i += 1;
                    continue;
                }
            }
        }

        // ── Object-like macros ─────────────────────────────────────
        // Substitute the replacement body directly.
        let mut expanded_painted: Vec<PaintedToken> =
            def.body.iter().cloned().map(PaintedToken::new).collect();
        paint_tokens(&mut expanded_painted, sym);

        // Inherit invocation paint.
        if is_painted(&tokens[i]) {
            let invocation_paint = &tokens[i].paint;
            for pt in &mut expanded_painted {
                pt.paint = merge_paint(&pt.paint, invocation_paint);
            }
        }

        // Rescan: splice expanded tokens into the working vector.
        let rest = tokens.split_off(i + 1);
        tokens.truncate(i);
        tokens.extend(expanded_painted);
        tokens.extend(rest);

        pp.recursion_depth -= 1;
        // Do NOT advance i — rescan from same position.
    }

    output
}

// ===========================================================================
// Argument substitution pipeline
// ===========================================================================

/// Substitutes actual arguments into a function-like macro's replacement body.
///
/// Implements the three-phase substitution algorithm from C11 §6.10.3.1:
///
/// 1. **Stringification** (`#`): Parameters preceded by `#` are replaced
///    with a string literal of the **unexpanded** argument.
/// 2. **Token pasting** (`##`): Parameters adjacent to `##` are substituted
///    with **unexpanded** argument tokens, then pasted.
/// 3. **Regular substitution**: Remaining parameter occurrences are replaced
///    with **pre-expanded** (macro-expanded) argument tokens.
///
/// Additionally handles the GCC comma-deletion extension: when `##__VA_ARGS__`
/// appears and the variadic argument is empty, the preceding comma is deleted.
fn substitute_body(
    pp: &mut Preprocessor,
    def: &MacroDef,
    raw_args: &[Vec<Token>],
    _invocation_span: Span,
) -> Vec<Token> {
    let params = def.params.as_deref().unwrap_or(&[]);
    let body = &def.body;

    if body.is_empty() {
        return Vec::new();
    }

    let va_args_sym = pp.interner.intern("__VA_ARGS__");

    // Build an extended parameter list that includes __VA_ARGS__ for variadic
    // macros so that apply_stringify_operators and apply_paste_operators can
    // resolve __VA_ARGS__ references in the body to the variadic argument.
    let extended_params: Vec<Symbol> = if def.is_variadic {
        let mut p = params.to_vec();
        if !p.contains(&va_args_sym) {
            p.push(va_args_sym);
        }
        p
    } else {
        params.to_vec()
    };

    // Step 1: GCC comma deletion for `##__VA_ARGS__` with empty variadic.
    let working_body = handle_gcc_comma_deletion(
        body,
        &extended_params,
        va_args_sym,
        def.is_variadic,
        raw_args,
    );

    // Step 2: Apply `#` stringification using **unexpanded** arguments.
    let after_stringify = apply_stringify_operators(
        &working_body,
        raw_args,
        &extended_params,
        &mut pp.interner,
        &mut pp.diagnostics,
    );

    // Step 3: Apply `##` token pasting using **unexpanded** arguments.
    let after_paste = apply_paste_operators(
        &after_stringify,
        raw_args,
        &extended_params,
        &mut pp.interner,
        &mut pp.diagnostics,
    );

    // Step 4: Pre-expand arguments for regular substitution positions.
    // Arguments used with # or ## have already been consumed by steps 2–3,
    // so the remaining parameter names will get pre-expanded substitutions.
    let expanded_args: Vec<Vec<Token>> = raw_args
        .iter()
        .map(|arg| expand_macros(pp, arg))
        .collect();

    // Step 5: Substitute remaining parameter identifiers with pre-expanded args.
    substitute_remaining_params(
        &after_paste,
        &extended_params,
        &expanded_args,
        def.is_variadic,
        va_args_sym,
    )
}

// ===========================================================================
// GCC comma-deletion extension
// ===========================================================================

/// Handles the GCC `, ## __VA_ARGS__` comma-deletion extension.
///
/// When a variadic macro is invoked with zero variadic arguments and the
/// macro body contains `, ## __VA_ARGS__`, GCC deletes the comma rather
/// than leaving it dangling. Example:
///
/// ```text
/// #define LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
/// LOG("hello")
/// // Without comma deletion: printf("hello", )  — syntax error
/// // With comma deletion:    printf("hello")     — correct
/// ```
///
/// This function pre-processes the macro body before `apply_paste_operators`
/// to remove the comma in the identified pattern.
fn handle_gcc_comma_deletion(
    body: &[Token],
    extended_params: &[Symbol],
    va_args_sym: Symbol,
    is_variadic: bool,
    raw_args: &[Vec<Token>],
) -> Vec<Token> {
    if !is_variadic {
        return body.to_vec();
    }

    // Determine the index of the variadic argument: it's always the last
    // parameter position. For standard variadic (...), __VA_ARGS__ maps
    // to index = named_param_count. For named variadic (args...), the
    // named param IS the variadic.
    let va_idx = extended_params
        .iter()
        .position(|p| *p == va_args_sym)
        .unwrap_or(extended_params.len().saturating_sub(1));

    // Check if the variadic argument is empty.
    let va_empty = raw_args.get(va_idx).map_or(true, |a| {
        a.is_empty()
            || a.iter()
                .all(|t| matches!(t.kind, TokenKind::Whitespace | TokenKind::Newline))
    });

    if !va_empty {
        // When variadic args are non-empty, we must STILL neutralise the
        // `## __VA_ARGS__` pattern by removing the `##` token.  If we leave
        // it in place, `apply_paste_operators` will later try to
        // token-paste the preceding comma with the first variadic-argument
        // token, destroying both.  The correct GCC semantics for
        // non-empty variadic is: keep the comma, substitute __VA_ARGS__
        // normally — just strip the `##` that was acting as the
        // conditional-comma-deletion marker.
        let mut result = Vec::with_capacity(body.len());
        let mut i = 0;

        while i < body.len() {
            if matches!(body[i].kind, TokenKind::Comma) {
                // Probe ahead for `## __VA_ARGS__`.
                let mut j = i + 1;
                while j < body.len()
                    && matches!(body[j].kind, TokenKind::Whitespace | TokenKind::Newline)
                {
                    j += 1;
                }
                if j < body.len() && matches!(body[j].kind, TokenKind::HashHash) {
                    let mut k = j + 1;
                    while k < body.len()
                        && matches!(body[k].kind, TokenKind::Whitespace | TokenKind::Newline)
                    {
                        k += 1;
                    }
                    if k < body.len() {
                        if let TokenKind::Identifier(s) = body[k].kind {
                            let is_va = s == va_args_sym
                                || extended_params
                                    .last()
                                    .map_or(false, |last| *last == s && s != va_args_sym);
                            if is_va {
                                // Non-empty case: keep comma, skip `##`
                                // (and interstitial whitespace), keep
                                // `__VA_ARGS__` for normal substitution.
                                result.push(body[i].clone()); // push comma
                                i = k; // advance to __VA_ARGS__; will be
                                        // pushed on the next iteration
                                continue;
                            }
                        }
                    }
                }
            }
            result.push(body[i].clone());
            i += 1;
        }

        return result;
    }

    // Scan for pattern: Comma [Whitespace…] HashHash [Whitespace…] __VA_ARGS__
    let mut result = Vec::with_capacity(body.len());
    let mut i = 0;

    while i < body.len() {
        if matches!(body[i].kind, TokenKind::Comma) {
            // Probe ahead for ## __VA_ARGS__ pattern.
            let mut j = i + 1;
            // Skip whitespace.
            while j < body.len()
                && matches!(body[j].kind, TokenKind::Whitespace | TokenKind::Newline)
            {
                j += 1;
            }
            // Check for ##.
            if j < body.len() && matches!(body[j].kind, TokenKind::HashHash) {
                let mut k = j + 1;
                // Skip whitespace after ##.
                while k < body.len()
                    && matches!(body[k].kind, TokenKind::Whitespace | TokenKind::Newline)
                {
                    k += 1;
                }
                // Check for __VA_ARGS__ (or the variadic param name).
                if k < body.len() {
                    if let TokenKind::Identifier(s) = body[k].kind {
                        let is_va = s == va_args_sym
                            || extended_params
                                .last()
                                .map_or(false, |last| *last == s && s != va_args_sym);
                        if is_va {
                            // GCC comma deletion: skip comma, ##, whitespace,
                            // and __VA_ARGS__ entirely.
                            i = k + 1;
                            continue;
                        }
                    }
                }
            }
        }
        result.push(body[i].clone());
        i += 1;
    }

    result
}

// ===========================================================================
// Remaining-parameter substitution
// ===========================================================================

/// Substitutes remaining parameter identifiers in the token list with their
/// corresponding pre-expanded argument tokens.
///
/// This runs AFTER `apply_stringify_operators` and `apply_paste_operators`
/// have consumed all `#` and `##` occurrences. The remaining parameter names
/// in the token sequence are those used in "regular" positions and should be
/// replaced with pre-expanded (macro-expanded) arguments.
fn substitute_remaining_params(
    tokens: &[Token],
    extended_params: &[Symbol],
    expanded_args: &[Vec<Token>],
    _is_variadic: bool,
    _va_args_sym: Symbol,
) -> Vec<Token> {
    let mut result = Vec::with_capacity(tokens.len());

    for tok in tokens {
        if let TokenKind::Identifier(sym) = tok.kind {
            // Check if this identifier is a parameter name.
            if let Some(idx) = extended_params
                .iter()
                .position(|p| p.as_u32() == sym.as_u32())
            {
                // Substitute with pre-expanded argument tokens.
                if let Some(arg) = expanded_args.get(idx) {
                    // Strip leading/trailing whitespace from the expanded arg.
                    let trimmed: Vec<Token> = arg
                        .iter()
                        .filter(|t| !matches!(t.kind, TokenKind::Whitespace | TokenKind::Newline))
                        .cloned()
                        .collect();
                    result.extend(trimmed);
                }
                // If the argument is missing (out-of-range), substitute nothing.
                continue;
            }
        }
        // Non-parameter token — pass through.
        result.push(tok.clone());
    }

    result
}

// ===========================================================================
// Argument collection
// ===========================================================================

/// Collects actual argument token lists for a function-like macro invocation.
///
/// Starts parsing from the first token after the opening `(` and collects
/// arguments delimited by top-level commas, respecting balanced parentheses.
/// For variadic macros, arguments beyond the named parameter count are
/// packed (including their separating commas) into the variadic argument.
///
/// # Returns
///
/// `Ok((args, end_index))` where `end_index` is the position of the
/// closing `)` in the `tokens` array, or `Err(())` on malformed input.
fn collect_arguments(
    pp: &mut Preprocessor,
    tokens: &[PaintedToken],
    start: usize,
    def: &MacroDef,
    invocation_span: Span,
) -> Result<(Vec<Vec<Token>>, usize), ()> {
    let mut args: Vec<Vec<Token>> = Vec::new();
    let mut current_arg: Vec<Token> = Vec::new();
    let mut depth: u32 = 1; // Already inside the outer `(`.
    let mut i = start;
    let param_count = def.params.as_ref().map_or(0, |p| p.len());

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
                    // End of argument list — push the last argument.
                    args.push(current_arg);
                    return Ok((args, i));
                }
                current_arg.push(tokens[i].token.clone());
            }
            TokenKind::Comma if depth == 1 => {
                // Top-level comma — argument separator.
                //
                // For variadic macros, once we've collected enough named
                // arguments, remaining tokens (including commas) are packed
                // into the variadic (__VA_ARGS__) argument.
                if def.is_variadic && args.len() >= param_count {
                    current_arg.push(tokens[i].token.clone());
                } else {
                    args.push(current_arg);
                    current_arg = Vec::new();
                }
            }
            TokenKind::Eof => {
                // Unexpected end of file inside macro arguments.
                pp.diagnostics.error(invocation_span, "unterminated macro argument list");
                return Err(());
            }
            _ => {
                current_arg.push(tokens[i].token.clone());
            }
        }
        i += 1;
    }

    // Reached end of token stream without finding closing `)`.
    pp.diagnostics
        .error(invocation_span, "unterminated macro argument list");
    Err(())
}

// ===========================================================================
// Helper: find opening parenthesis
// ===========================================================================

/// Searches for the opening `(` after a function-like macro name, skipping
/// any intervening whitespace and newline tokens.
///
/// Returns `Some(index)` if `(` is found, `None` if a non-whitespace
/// non-paren token is encountered first (meaning this is not a function-like
/// invocation, just an identifier that happens to match the macro name).
fn find_open_paren(tokens: &[PaintedToken], start: usize) -> Option<usize> {
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

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::{DiagnosticEngine, Span};
    use crate::common::source_map::SourceMap;
    use crate::common::string_interner::Interner;
    use crate::common::target::Target;

    /// Build a minimal Preprocessor for testing macro expansion.
    fn test_pp() -> Preprocessor {
        Preprocessor::new(
            SourceMap::new(),
            DiagnosticEngine::new(),
            Target::X86_64,
            Interner::new(),
        )
    }

    /// Helper: create an identifier token.
    fn ident(name: &str, interner: &mut Interner) -> Token {
        Token::new(TokenKind::Identifier(interner.intern(name)), Span::DUMMY)
    }

    /// Helper: create an integer literal token.
    fn int_tok(value: u128) -> Token {
        use crate::frontend::lexer::token::IntegerSuffix;
        Token::new(
            TokenKind::IntegerLiteral {
                value,
                suffix: IntegerSuffix::None,
            },
            Span::DUMMY,
        )
    }

    /// Helper: build a simple operator/punctuator token.
    fn punct(kind: TokenKind) -> Token {
        Token::new(kind, Span::DUMMY)
    }

    // -----------------------------------------------------------------------
    // TokenStream tests
    // -----------------------------------------------------------------------

    #[test]
    fn token_stream_basic_iteration() {
        let tokens = vec![
            Token::new(TokenKind::Identifier(Symbol::EMPTY), Span::DUMMY),
            Token::new(TokenKind::LeftParen, Span::DUMMY),
        ];
        let mut stream = TokenStream::new(tokens);

        assert!(!stream.is_at_end());
        assert!(stream.peek().is_some());
        assert!(stream.peek_ahead(1).is_some());
        assert!(stream.peek_ahead(2).is_none());

        let first = stream.next().unwrap();
        assert!(matches!(first.token.kind, TokenKind::Identifier(_)));

        let second = stream.next().unwrap();
        assert!(matches!(second.token.kind, TokenKind::LeftParen));

        assert!(stream.is_at_end());
        assert!(stream.next().is_none());
    }

    #[test]
    fn token_stream_save_restore() {
        let tokens = vec![
            Token::new(TokenKind::LeftParen, Span::DUMMY),
            Token::new(TokenKind::RightParen, Span::DUMMY),
            Token::new(TokenKind::Semicolon, Span::DUMMY),
        ];
        let mut stream = TokenStream::new(tokens);

        stream.save();
        let _ = stream.next(); // consume LeftParen
        let _ = stream.next(); // consume RightParen
        assert!(stream.peek().is_some());

        stream.restore();
        // After restore, cursor is back at position 0.
        let first_again = stream.next().unwrap();
        assert!(matches!(first_again.token.kind, TokenKind::LeftParen));
    }

    #[test]
    fn token_stream_remaining() {
        let tokens = vec![
            Token::new(TokenKind::LeftParen, Span::DUMMY),
            Token::new(TokenKind::RightParen, Span::DUMMY),
        ];
        let mut stream = TokenStream::new(tokens);

        assert_eq!(stream.remaining().len(), 2);
        let _ = stream.next();
        assert_eq!(stream.remaining().len(), 1);
        let _ = stream.next();
        assert_eq!(stream.remaining().len(), 0);
    }

    // -----------------------------------------------------------------------
    // Object-like macro expansion
    // -----------------------------------------------------------------------

    #[test]
    fn expand_object_like_simple() {
        let mut pp = test_pp();
        let one_sym = pp.interner.intern("ONE");
        let def = MacroDef::object_like(one_sym, vec![int_tok(1)], Span::DUMMY);
        pp.macros.insert(one_sym, def);

        let input = vec![ident("ONE", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0].kind,
            TokenKind::IntegerLiteral { value: 1, .. }
        ));
    }

    #[test]
    fn expand_object_like_chain() {
        // #define A B
        // #define B 42
        // A → B → 42
        let mut pp = test_pp();
        let a_sym = pp.interner.intern("A");
        let b_sym = pp.interner.intern("B");

        let b_tok = ident("B", &mut pp.interner);
        let def_a = MacroDef::object_like(a_sym, vec![b_tok], Span::DUMMY);
        let def_b = MacroDef::object_like(b_sym, vec![int_tok(42)], Span::DUMMY);
        pp.macros.insert(a_sym, def_a);
        pp.macros.insert(b_sym, def_b);

        let input = vec![ident("A", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0].kind,
            TokenKind::IntegerLiteral { value: 42, .. }
        ));
    }

    // -----------------------------------------------------------------------
    // Self-referential macro (paint marker test)
    // -----------------------------------------------------------------------

    #[test]
    fn expand_self_referential_terminates() {
        // #define A A — must terminate, not infinite loop.
        let mut pp = test_pp();
        let a_sym = pp.interner.intern("A");
        let a_tok = ident("A", &mut pp.interner);
        let def = MacroDef::object_like(a_sym, vec![a_tok], Span::DUMMY);
        pp.macros.insert(a_sym, def);

        let input = vec![ident("A", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        // A expands to [A(painted)] → painted A is not re-expanded → result: A
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].kind, TokenKind::Identifier(s) if s == a_sym));
    }

    #[test]
    fn expand_mutual_recursion_terminates() {
        // #define A B
        // #define B A
        let mut pp = test_pp();
        let a_sym = pp.interner.intern("A");
        let b_sym = pp.interner.intern("B");
        let a_tok = ident("A", &mut pp.interner);
        let b_tok = ident("B", &mut pp.interner);

        let def_a = MacroDef::object_like(a_sym, vec![b_tok], Span::DUMMY);
        let def_b = MacroDef::object_like(b_sym, vec![a_tok], Span::DUMMY);
        pp.macros.insert(a_sym, def_a);
        pp.macros.insert(b_sym, def_b);

        let input = vec![ident("A", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        // A → B(painted:A) → try B → body is [A] → paint for B →
        // [A(painted:A,B)] → A is painted for A → no re-expand → result: A
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].kind, TokenKind::Identifier(s) if s == a_sym));
    }

    // -----------------------------------------------------------------------
    // Function-like macro expansion
    // -----------------------------------------------------------------------

    #[test]
    fn expand_function_like_simple() {
        // #define F(x) x
        let mut pp = test_pp();
        let f_sym = pp.interner.intern("F");
        let x_sym = pp.interner.intern("x");
        let x_tok = ident("x", &mut pp.interner);

        let def = MacroDef::function_like(f_sym, vec![x_sym], false, vec![x_tok], Span::DUMMY);
        pp.macros.insert(f_sym, def);

        // F(42)
        let input = vec![
            ident("F", &mut pp.interner),
            punct(TokenKind::LeftParen),
            int_tok(42),
            punct(TokenKind::RightParen),
        ];
        let result = expand_macros(&mut pp, &input);

        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0].kind,
            TokenKind::IntegerLiteral { value: 42, .. }
        ));
    }

    #[test]
    fn expand_empty_macro_body() {
        // #define EMPTY
        let mut pp = test_pp();
        let sym = pp.interner.intern("EMPTY");
        let def = MacroDef::object_like(sym, Vec::new(), Span::DUMMY);
        pp.macros.insert(sym, def);

        let input = vec![ident("EMPTY", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Recursion depth limit
    // -----------------------------------------------------------------------

    #[test]
    fn expand_depth_limit_triggers_error() {
        let mut pp = test_pp();
        pp.max_recursion_depth = 2;
        pp.recursion_depth = 2; // Already at limit.

        let sym = pp.interner.intern("X");
        let def = MacroDef::object_like(sym, vec![int_tok(1)], Span::DUMMY);
        pp.macros.insert(sym, def);

        let input = vec![ident("X", &mut pp.interner)];
        let result = expand_macros(&mut pp, &input);

        // Should not expand because depth limit is reached.
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0].kind, TokenKind::Identifier(_)));
        assert!(pp.diagnostics.has_errors());
    }
}
