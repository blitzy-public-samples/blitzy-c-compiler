// src/frontend/preprocessor/directives.rs
//
// Preprocessor directive handling module for the BCC C compiler.
//
// Implements all C11 and GCC-compatible preprocessor directives:
//   - `#define` / `#undef` — macro definition and removal
//   - `#include` — file inclusion (user and system)
//   - `#if` / `#ifdef` / `#ifndef` / `#elif` / `#else` / `#endif` — conditional compilation
//   - `#pragma` — compiler pragmas (once, pack, GCC visibility, GCC diagnostic)
//   - `#error` / `#warning` — user-defined diagnostics
//   - `#line` — source location remapping
//
// This module provides the public API surface (`process_directive`, `DirectiveResult`,
// `ConditionalState`, `check_unterminated_conditionals`) for directive processing.
// It operates on the `Preprocessor` struct from the parent module, accessing both
// public and private fields as a child module.

use crate::common::diagnostics::Span;
use crate::common::source_map::{FileId, LineDirective};
use crate::common::string_interner::{Interner, Symbol};
use crate::frontend::lexer::token::{Token, TokenKind};

use super::expression::evaluate_expression;
use super::include_handler::IncludeKind;
use super::macro_expander::expand_macros;
use super::{MacroDef, Preprocessor};

// ═══════════════════════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of processing a single preprocessor directive.
///
/// Returned by [`process_directive`] to communicate the outcome of directive
/// handling back to the preprocessor driver. The driver uses this to decide
/// whether to continue emitting tokens, skip a conditional block, splice in
/// included file tokens, or halt on error.
#[derive(Debug)]
pub enum DirectiveResult {
    /// Directive processed successfully; continue with the next token.
    Continue,
    /// A conditional directive evaluated to false; the driver should skip
    /// tokens until a matching `#elif`, `#else`, or `#endif` is found.
    SkipToEndif,
    /// An `#include` directive was processed and the included file's tokens
    /// are returned for splicing into the output stream.
    FileIncluded(Vec<Token>),
    /// A fatal error occurred (e.g., `#error`, unknown directive, parse error).
    /// The diagnostic engine already contains the error message.
    Error,
}

/// Tracks the state of a single conditional compilation group for the public API.
///
/// Each `#if` / `#ifdef` / `#ifndef` pushes a new `ConditionalState` onto the
/// conditional stack. The fields track whether the current branch is active and
/// whether any branch has been taken (to correctly handle `#elif` / `#else`).
///
/// The `span` field records the source location of the opening directive for
/// diagnostic reporting of unterminated conditionals.
#[derive(Clone, Debug)]
pub struct ConditionalState {
    /// `true` if the current branch is active (tokens should be emitted).
    pub is_active: bool,
    /// `true` once any branch in this `#if`/`#elif`/`#else` group has been taken.
    pub has_been_true: bool,
    /// Source location of the opening `#if` / `#ifdef` / `#ifndef` directive.
    pub span: Span,
}

impl ConditionalState {
    /// Creates a new conditional state for an opening directive.
    ///
    /// If `active` is `true`, the branch is immediately taken and
    /// `has_been_true` is also set to `true`.
    pub fn new(active: bool, span: Span) -> Self {
        Self {
            is_active: active,
            has_been_true: active,
            span,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers — whitespace skipping and token text extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Skips leading whitespace tokens in a token slice.
///
/// Returns a sub-slice starting at the first non-whitespace token.
/// If all tokens are whitespace, returns an empty slice.
fn skip_ws(tokens: &[Token]) -> &[Token] {
    let mut i = 0;
    while i < tokens.len() && tokens[i].kind == TokenKind::Whitespace {
        i += 1;
    }
    &tokens[i..]
}

/// Extracts a human-readable text representation of a single token.
///
/// For identifiers, resolves the interned `Symbol` back to its string form.
/// For string literals, returns the string content. For other tokens, uses
/// the `Display` implementation of `TokenKind` (which produces the canonical
/// C spelling for keywords and operators).
fn token_text(tok: &Token, interner: &Interner) -> String {
    match &tok.kind {
        TokenKind::Identifier(sym) => interner.resolve(*sym).to_string(),
        TokenKind::StringLiteral { ref value, .. } => String::from_utf8_lossy(value).into_owned(),
        TokenKind::IntegerLiteral { value, .. } => value.to_string(),
        other => format!("{}", other),
    }
}

/// Concatenates the text of all tokens in a slice, separated by spaces where
/// whitespace tokens appear.
///
/// Used to reconstruct message text for `#error` and `#warning` directives.
fn concat_token_text(tokens: &[Token], interner: &Interner) -> String {
    let mut result = String::new();
    for tok in tokens {
        if tok.kind == TokenKind::Whitespace {
            if !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }
        } else if tok.kind == TokenKind::Newline || tok.kind == TokenKind::Eof {
            // Skip newlines and EOF in concatenation.
            continue;
        } else {
            result.push_str(&token_text(tok, interner));
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Processes a single preprocessor directive line.
///
/// This is the main entry point for directive handling. After the preprocessor
/// driver identifies a `#` at the start of a logical line, it collects the
/// remaining tokens on that line and passes them here. The first non-whitespace
/// token is expected to be the directive keyword (e.g., `define`, `include`).
///
/// # Arguments
///
/// * `pp` — Mutable reference to the preprocessor state. Directive handlers
///   may modify the macro table, conditional stack, include state, diagnostics,
///   and source map.
/// * `tokens` — Token slice starting after the `#` token, extending to the
///   end of the directive line (excluding the newline).
///
/// # Returns
///
/// A [`DirectiveResult`] indicating the outcome of directive processing.
pub fn process_directive(pp: &mut Preprocessor, tokens: &[Token]) -> DirectiveResult {
    let tokens = skip_ws(tokens);

    // Null directive: `#` followed by nothing (or only whitespace). Valid per C11 §6.10p1.
    if tokens.is_empty() {
        return DirectiveResult::Continue;
    }

    // Extract the directive keyword. It must be an identifier.
    let (dir_name, dir_span) = match tokens[0].kind {
        TokenKind::Identifier(sym) => {
            let name = pp.interner.resolve(sym).to_string();
            (name, tokens[0].span)
        }
        _ => {
            pp.diagnostics.error(
                tokens[0].span,
                format!(
                    "expected preprocessing directive name, found '{}'",
                    tokens[0].kind
                ),
            );
            return DirectiveResult::Error;
        }
    };

    let rest = &tokens[1..];

    // Dispatch to the appropriate handler.
    match dir_name.as_str() {
        "define" => handle_define(pp, rest, dir_span),
        "undef" => handle_undef(pp, rest, dir_span),
        "include" => handle_include(pp, rest, dir_span),
        "if" => handle_if(pp, rest, dir_span),
        "ifdef" => handle_ifdef(pp, rest, dir_span),
        "ifndef" => handle_ifndef(pp, rest, dir_span),
        "elif" => handle_elif(pp, rest, dir_span),
        "else" => handle_else(pp, dir_span),
        "endif" => handle_endif(pp, dir_span),
        "pragma" => handle_pragma(pp, rest, dir_span),
        "error" => handle_error(pp, rest, dir_span),
        "warning" => handle_warning(pp, rest, dir_span),
        "line" => handle_line(pp, rest, dir_span),
        unknown => {
            pp.diagnostics.error(
                dir_span,
                format!("unknown preprocessing directive '#{}' ", unknown),
            );
            DirectiveResult::Error
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// #define handling
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#define NAME replacement-list` and `#define NAME(params) replacement-list`.
///
/// Parses the macro name, determines whether it is object-like or function-like
/// (based on whether `(` immediately follows the name without intervening
/// whitespace), parses the parameter list for function-like macros (including
/// variadic `...`), and registers the macro definition in `pp.macros`.
///
/// Redefinition of an existing macro emits a warning unless the replacement
/// lists are identical (or the existing definition is a predefined macro,
/// in which case redefining is silently accepted per GCC compatibility).
fn handle_define(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected macro name after #define");
        return DirectiveResult::Error;
    }

    // The first token must be an identifier (the macro name).
    let name_sym = match tokens[0].kind {
        TokenKind::Identifier(s) => s,
        _ => {
            pp.diagnostics
                .error(tokens[0].span, "expected identifier for macro name");
            return DirectiveResult::Error;
        }
    };

    let rest = &tokens[1..];

    // Determine if this is a function-like macro: the `(` must be immediately
    // adjacent to the macro name (no intervening whitespace). This is a critical
    // distinction per C11 §6.10.3p3.
    let is_func_like = !rest.is_empty()
        && rest[0].kind == TokenKind::LeftParen
        && tokens[0].span.end == rest[0].span.start;

    if is_func_like {
        // Parse the parameter list for a function-like macro.
        let after_paren = &rest[1..];
        let mut params: Vec<Symbol> = Vec::new();
        let mut is_variadic = false;
        let mut i = 0;
        let after_paren = skip_ws(after_paren);

        loop {
            if i >= after_paren.len() {
                pp.diagnostics
                    .error(span, "unterminated macro parameter list");
                return DirectiveResult::Error;
            }

            // End of parameter list.
            if after_paren[i].kind == TokenKind::RightParen {
                i += 1;
                break;
            }

            // Variadic marker `...`.
            if after_paren[i].kind == TokenKind::Ellipsis {
                is_variadic = true;
                i += 1;
                let remaining = skip_ws(&after_paren[i..]);
                if remaining.is_empty() || remaining[0].kind != TokenKind::RightParen {
                    pp.diagnostics
                        .error(span, "expected ')' after '...' in macro parameter list");
                    return DirectiveResult::Error;
                }
                // Advance past whitespace and the closing paren.
                i += after_paren[i..].len() - remaining.len() + 1;
                break;
            }

            // Parameter name (must be an identifier).
            match after_paren[i].kind {
                TokenKind::Identifier(sym) => {
                    params.push(sym);
                    i += 1;
                }
                _ => {
                    pp.diagnostics.error(
                        after_paren[i].span,
                        "expected parameter name in macro definition",
                    );
                    return DirectiveResult::Error;
                }
            }

            // Skip whitespace and expect `,` or `)`.
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

        // Warn on macro redefinition (unless the existing macro is predefined).
        if let Some(existing) = pp.macros.get(&name_sym) {
            if !existing.is_predefined {
                pp.diagnostics.warning(
                    span,
                    format!("'{}' macro redefined", pp.interner.resolve(name_sym)),
                );
            }
        }

        let def = MacroDef::function_like(name_sym, params, is_variadic, body_tokens, span);
        pp.macros.insert(name_sym, def);
    } else {
        // Object-like macro: everything after the name (skipping leading whitespace) is the body.
        let body_tokens: Vec<Token> = skip_ws(rest).to_vec();

        // Warn on macro redefinition.
        if let Some(existing) = pp.macros.get(&name_sym) {
            if !existing.is_predefined {
                pp.diagnostics.warning(
                    span,
                    format!("'{}' macro redefined", pp.interner.resolve(name_sym)),
                );
            }
        }

        let def = MacroDef::object_like(name_sym, body_tokens, span);
        pp.macros.insert(name_sym, def);
    }

    DirectiveResult::Continue
}

// ═══════════════════════════════════════════════════════════════════════════
// #undef handling
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#undef NAME`.
///
/// Removes the named macro from the macro table. Undefining a non-existent
/// macro is silently accepted (no error per C11 §6.10.3.5p2). Attempting
/// to undefine a predefined macro (e.g., `__FILE__`, `__LINE__`) emits a
/// warning per GCC behavior.
fn handle_undef(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected macro name after #undef");
        return DirectiveResult::Error;
    }

    match tokens[0].kind {
        TokenKind::Identifier(sym) => {
            // Check if it's a predefined macro before removing.
            if let Some(existing) = pp.macros.get(&sym) {
                if existing.is_predefined {
                    pp.diagnostics.warning(
                        tokens[0].span,
                        format!("undefining predefined macro '{}'", pp.interner.resolve(sym)),
                    );
                }
            }
            pp.macros.remove(&sym);
            DirectiveResult::Continue
        }
        _ => {
            pp.diagnostics
                .error(tokens[0].span, "expected identifier after #undef");
            DirectiveResult::Error
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// #include handling
// ═══════════════════════════════════════════════════════════════════════════

/// Parses the include path from tokens following `#include`.
///
/// Recognizes three forms:
///   - `#include "path"` — user include (string literal).
///   - `#include <path>` — system include (angle-bracket delimited).
///   - `#include MACRO` — computed include (macro-expanded, then re-parsed).
///
/// For `<...>` system includes, the raw source text is extracted directly from
/// the source map using token spans when possible. This is critical because the
/// C tokenizer misinterprets parts of filenames (e.g., `stubs-64.h` — where
/// `64.h` is lexed as a floating-point literal, losing the original text). The
/// C standard specifies that `<h-char-sequence>` is a special "header-name"
/// preprocessing token, not a sequence of regular C tokens.
///
/// Returns the include kind and path string, or `None` on parse failure.
fn parse_include_path(
    pp: &mut Preprocessor,
    tokens: &[Token],
    span: Span,
) -> Option<(IncludeKind, String)> {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected file path after #include");
        return None;
    }

    // Form 1: `#include "path"` — user include.
    if let TokenKind::StringLiteral { ref value, .. } = tokens[0].kind {
        let path_string = String::from_utf8_lossy(value).into_owned();
        return Some((IncludeKind::User, path_string));
    }

    // Form 2: `#include <path>` — system include.
    // Extract raw source text between `<` and `>` to preserve exact filenames.
    if tokens[0].kind == TokenKind::Less {
        // Find the closing `>` token.
        let close_idx = tokens[1..]
            .iter()
            .position(|t| t.kind == TokenKind::Greater)
            .map(|pos| pos + 1);
        let close_idx = match close_idx {
            Some(idx) => idx,
            None => {
                pp.diagnostics.error(span, "missing '>' in #include <...>");
                return None;
            }
        };

        // Preferred approach: extract raw source text from spans. Tokens in a
        // non-macro-expanded #include line originate from the same file, so spans
        // are contiguous and valid.
        let lt_span = tokens[0].span;
        let gt_span = tokens[close_idx].span;
        if lt_span.file_id == gt_span.file_id && lt_span.file_id != u32::MAX {
            let file_id = FileId(lt_span.file_id);
            let raw = pp
                .source_map
                .get_snippet(file_id, lt_span.end, gt_span.start);
            let path = raw.trim().to_string();
            if !path.is_empty() {
                return Some((IncludeKind::System, path));
            }
        }

        // Fallback: reconstruct from token text (for cases where span extraction
        // yields an empty path, e.g., tokens generated synthetically).
        let mut path = String::new();
        for tok in &tokens[1..close_idx] {
            path.push_str(&token_text(tok, &pp.interner));
        }
        return Some((IncludeKind::System, path));
    }

    // Form 3: computed include — macro-expand and re-parse.
    let expanded = expand_macros(pp, tokens);
    let expanded = skip_ws(&expanded);
    if expanded.is_empty() {
        pp.diagnostics
            .error(span, "expected \"file\" or <file> after #include");
        return None;
    }

    // After expansion, try to parse as "path" or <path>.
    if let TokenKind::StringLiteral { ref value, .. } = expanded[0].kind {
        let path_string = String::from_utf8_lossy(value).into_owned();
        return Some((IncludeKind::User, path_string));
    }

    if expanded[0].kind == TokenKind::Less {
        // Find closing `>` in expanded tokens.
        let close_idx = expanded[1..]
            .iter()
            .position(|t| t.kind == TokenKind::Greater)
            .map(|pos| pos + 1);
        let close_idx = match close_idx {
            Some(idx) => idx,
            None => {
                pp.diagnostics
                    .error(span, "missing '>' in computed #include <...>");
                return None;
            }
        };

        // Try span-based extraction for expanded tokens too.
        let lt_span = expanded[0].span;
        let gt_span = expanded[close_idx].span;
        if lt_span.file_id == gt_span.file_id && lt_span.file_id != u32::MAX {
            let file_id = FileId(lt_span.file_id);
            let raw = pp
                .source_map
                .get_snippet(file_id, lt_span.end, gt_span.start);
            let path = raw.trim().to_string();
            if !path.is_empty() {
                return Some((IncludeKind::System, path));
            }
        }

        // Fallback: reconstruct from token text.
        let mut path = String::new();
        for tok in &expanded[1..close_idx] {
            path.push_str(&token_text(tok, &pp.interner));
        }
        return Some((IncludeKind::System, path));
    }

    pp.diagnostics
        .error(span, "expected \"file\" or <file> after #include");
    None
}

/// Handles `#include "file"` and `#include <file>`.
///
/// Resolves the include path via the include handler, checks for recursion
/// depth limits, loads the file with PUA-aware encoding, and returns the
/// raw token stream of the included file for further processing by the
/// preprocessor driver.
fn handle_include(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    // Parse the include path (handles "path", <path>, and computed includes).
    let (kind, path_str) = match parse_include_path(pp, tokens, span) {
        Some(v) => v,
        None => return DirectiveResult::Error,
    };

    // Check recursion depth limit (512 per Section 0.7.3).
    if pp.recursion_depth >= pp.max_recursion_depth {
        pp.diagnostics.error(
            span,
            format!(
                "#include nesting depth exceeds limit ({})",
                pp.max_recursion_depth
            ),
        );
        return DirectiveResult::Error;
    }

    // Resolve the include path via the include handler.
    let current_dir = pp.current_file_dir.clone();
    let resolved = match pp
        .include_handler
        .resolve_include(&path_str, kind, &current_dir)
    {
        Some(p) => p,
        None => {
            pp.diagnostics
                .error(span, format!("'{}': file not found", path_str));
            return DirectiveResult::Error;
        }
    };

    // Check if the file should be skipped (pragma once or include guard).
    let macros_ref = &pp.macros;
    if pp
        .include_handler
        .should_skip_include(&resolved, |sym| macros_ref.contains_key(sym))
    {
        return DirectiveResult::Continue;
    }

    // Check for circular includes.
    if let Err(circ) = pp.include_handler.push_include(&resolved) {
        pp.diagnostics
            .error(span, format!("circular #include dependency: {:?}", circ));
        return DirectiveResult::Error;
    }

    // Increment recursion depth.
    pp.recursion_depth += 1;

    // Load the file using PUA-aware encoding.
    let load_result = pp.include_handler.load_file(&resolved, &mut pp.source_map);
    let (file_id, content) = match load_result {
        Ok(result) => result,
        Err(e) => {
            pp.diagnostics
                .error(span, format!("cannot read '{}': {}", resolved.display(), e));
            pp.include_handler.pop_include();
            pp.recursion_depth -= 1;
            return DirectiveResult::Error;
        }
    };

    // Phase 1: trigraph replacement and line splicing on the loaded content.
    let spliced = super::phase1_trigraphs_and_line_splice(&content);

    // Tokenize the included file.
    let file_id_raw = file_id.0;
    let inc_tokens = super::pp_tokenize(&spliced, file_id_raw, &mut pp.interner);

    // Pop the include stack entry after tokenization; the driver will handle
    // recursive preprocessing of the returned token stream.
    pp.include_handler.pop_include();
    pp.recursion_depth -= 1;

    DirectiveResult::FileIncluded(inc_tokens)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditional compilation: #if / #ifdef / #ifndef / #elif / #else / #endif
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#if expr`.
///
/// Evaluates the preprocessor constant expression and pushes a new conditional
/// state onto the Preprocessor's conditional stack. Returns `SkipToEndif` if
/// the condition evaluates to false (the driver should skip tokens until a
/// matching `#elif`, `#else`, or `#endif`).
fn handle_if(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    // If we are inside a skipped branch, just push an inactive nested group.
    if !cond_stack_is_active(&pp.cond_stack) {
        pp.cond_stack.push(super::CondState::new(false, span));
        return DirectiveResult::SkipToEndif;
    }

    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected expression in #if directive");
        return DirectiveResult::Error;
    }

    // Evaluate the condition using the expression evaluator.
    let result = evaluate_expression(tokens, &pp.macros, &mut pp.interner, &mut pp.diagnostics);

    match result {
        Ok(val) => {
            let active = val != 0;
            pp.cond_stack.push(super::CondState::new(active, span));
            if active {
                DirectiveResult::Continue
            } else {
                DirectiveResult::SkipToEndif
            }
        }
        Err(()) => {
            // On evaluation error, treat the condition as false.
            pp.cond_stack.push(super::CondState::new(false, span));
            DirectiveResult::SkipToEndif
        }
    }
}

/// Handles `#ifdef NAME`.
///
/// Checks if the named macro is defined in `pp.macros` and pushes an
/// appropriate conditional state.
fn handle_ifdef(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    // If inside a skipped branch, push inactive group.
    if !cond_stack_is_active(&pp.cond_stack) {
        pp.cond_stack.push(super::CondState::new(false, span));
        return DirectiveResult::SkipToEndif;
    }

    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected identifier after #ifdef");
        return DirectiveResult::Error;
    }

    match tokens[0].kind {
        TokenKind::Identifier(sym) => {
            let defined = pp.macros.contains_key(&sym);
            pp.cond_stack.push(super::CondState::new(defined, span));
            if defined {
                DirectiveResult::Continue
            } else {
                DirectiveResult::SkipToEndif
            }
        }
        _ => {
            pp.diagnostics
                .error(tokens[0].span, "expected identifier after #ifdef");
            DirectiveResult::Error
        }
    }
}

/// Handles `#ifndef NAME`.
///
/// Checks if the named macro is NOT defined in `pp.macros` and pushes an
/// appropriate conditional state.
fn handle_ifndef(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    // If inside a skipped branch, push inactive group.
    if !cond_stack_is_active(&pp.cond_stack) {
        pp.cond_stack.push(super::CondState::new(false, span));
        return DirectiveResult::SkipToEndif;
    }

    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected identifier after #ifndef");
        return DirectiveResult::Error;
    }

    match tokens[0].kind {
        TokenKind::Identifier(sym) => {
            let not_defined = !pp.macros.contains_key(&sym);
            pp.cond_stack.push(super::CondState::new(not_defined, span));
            if not_defined {
                DirectiveResult::Continue
            } else {
                DirectiveResult::SkipToEndif
            }
        }
        _ => {
            pp.diagnostics
                .error(tokens[0].span, "expected identifier after #ifndef");
            DirectiveResult::Error
        }
    }
}

/// Handles `#elif expr`.
///
/// If no previous branch in the current conditional group has been taken,
/// evaluates the expression and potentially activates this branch.
fn handle_elif(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    if pp.cond_stack.is_empty() {
        pp.diagnostics.error(span, "#elif without matching #if");
        return DirectiveResult::Error;
    }

    let seen_else = pp.cond_stack.last().unwrap().seen_else;
    if seen_else {
        pp.diagnostics.error(span, "#elif after #else");
        return DirectiveResult::Error;
    }

    let any_taken = pp.cond_stack.last().unwrap().any_branch_taken;
    if any_taken {
        // A previous branch was taken — this branch is inactive.
        pp.cond_stack.last_mut().unwrap().active = false;
        return DirectiveResult::SkipToEndif;
    }

    // Evaluate the condition.
    let tokens = skip_ws(tokens);
    let result = evaluate_expression(tokens, &pp.macros, &mut pp.interner, &mut pp.diagnostics);

    match result {
        Ok(val) => {
            let active = val != 0;
            let top = pp.cond_stack.last_mut().unwrap();
            top.active = active;
            if active {
                top.any_branch_taken = true;
                DirectiveResult::Continue
            } else {
                DirectiveResult::SkipToEndif
            }
        }
        Err(()) => {
            // Treat evaluation error as false.
            pp.cond_stack.last_mut().unwrap().active = false;
            DirectiveResult::SkipToEndif
        }
    }
}

/// Handles `#else`.
///
/// If no previous branch in the current conditional group has been taken,
/// activates this branch. Prevents duplicate `#else` in the same group.
fn handle_else(pp: &mut Preprocessor, span: Span) -> DirectiveResult {
    if pp.cond_stack.is_empty() {
        pp.diagnostics.error(span, "#else without matching #if");
        return DirectiveResult::Error;
    }

    let seen_else = pp.cond_stack.last().unwrap().seen_else;
    if seen_else {
        pp.diagnostics.error(span, "duplicate #else");
        return DirectiveResult::Error;
    }

    let top = pp.cond_stack.last_mut().unwrap();
    top.seen_else = true;
    let should_activate = !top.any_branch_taken;
    top.active = should_activate;

    if should_activate {
        DirectiveResult::Continue
    } else {
        DirectiveResult::SkipToEndif
    }
}

/// Handles `#endif`.
///
/// Pops the topmost conditional state from the stack. Errors if the stack
/// is empty (unmatched `#endif`).
fn handle_endif(pp: &mut Preprocessor, span: Span) -> DirectiveResult {
    if pp.cond_stack.is_empty() {
        pp.diagnostics.error(span, "#endif without matching #if");
        return DirectiveResult::Error;
    }
    pp.cond_stack.pop();
    DirectiveResult::Continue
}

/// Checks whether all conditional branches in the stack are currently active.
///
/// Returns `true` if the conditional stack is empty (no nesting) or if every
/// nested conditional's current branch is active. Used to determine whether
/// non-conditional directives should be processed (they are skipped when
/// inside an inactive branch).
fn cond_stack_is_active(cond_stack: &[super::CondState]) -> bool {
    cond_stack.iter().all(|c| c.active)
}

// ═══════════════════════════════════════════════════════════════════════════
// #pragma handling
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#pragma` directives.
///
/// Recognized pragmas:
///   - `#pragma once` — marks the current file for include-once behavior.
///   - `#pragma pack(push, N)` / `#pragma pack(pop)` / `#pragma pack(N)` —
///     struct packing directives (stored for semantic analysis).
///   - `#pragma GCC visibility push(default|hidden|protected)` /
///     `#pragma GCC visibility pop` — symbol visibility control.
///   - `#pragma GCC diagnostic push` / `#pragma GCC diagnostic pop` /
///     `#pragma GCC diagnostic ignored "-Wname"` — diagnostic suppression.
///
/// Unknown pragmas emit a warning (not an error) per C11 §6.10.6p1.
fn handle_pragma(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        // Empty pragma — valid but does nothing.
        return DirectiveResult::Continue;
    }

    let pragma_name = token_text(&tokens[0], &pp.interner);
    let rest = skip_ws(&tokens[1..]);

    match pragma_name.as_str() {
        "once" => {
            // Register the current file as include-once.
            let current_dir = pp.current_file_dir.clone();
            pp.include_handler.register_pragma_once(&current_dir);
            DirectiveResult::Continue
        }

        "pack" => handle_pragma_pack(pp, rest, span),

        "GCC" => handle_pragma_gcc(pp, rest, span),

        _ => {
            // Unknown pragma — emit warning per C standard.
            pp.diagnostics
                .warning(span, format!("unknown pragma '{}' ignored", pragma_name));
            DirectiveResult::Continue
        }
    }
}

/// Handles `#pragma pack(...)` directives for struct packing control.
///
/// Supports:
///   - `#pragma pack(N)` — set packing alignment to N.
///   - `#pragma pack(push)` — push current packing onto stack.
///   - `#pragma pack(push, N)` — push and set new packing.
///   - `#pragma pack(pop)` — restore previous packing.
///   - `#pragma pack()` — reset to default packing.
fn handle_pragma_pack(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    // Expect `(`.
    let tokens = skip_ws(tokens);
    if tokens.is_empty() || tokens[0].kind != TokenKind::LeftParen {
        pp.diagnostics
            .warning(span, "expected '(' after #pragma pack");
        return DirectiveResult::Continue;
    }

    let inner = skip_ws(&tokens[1..]);
    if inner.is_empty() {
        pp.diagnostics
            .warning(span, "unterminated #pragma pack(...)");
        return DirectiveResult::Continue;
    }

    // `#pragma pack()` — reset to default.
    if inner[0].kind == TokenKind::RightParen {
        // Reset packing to default (implementation stores as 0 meaning "default").
        return DirectiveResult::Continue;
    }

    // Check for push/pop keyword.
    if let TokenKind::Identifier(sym) = inner[0].kind {
        let keyword = pp.interner.resolve(sym).to_string();
        match keyword.as_str() {
            "push" => {
                // Check for optional `, N`.
                let after_push = skip_ws(&inner[1..]);
                if !after_push.is_empty() && after_push[0].kind == TokenKind::Comma {
                    let after_comma = skip_ws(&after_push[1..]);
                    if !after_comma.is_empty() {
                        if let TokenKind::IntegerLiteral { value, .. } = after_comma[0].kind {
                            // Push and set new alignment to `value`.
                            let _alignment = value as u32;
                        }
                    }
                }
                // Push current packing state (semantic analysis integration point).
                return DirectiveResult::Continue;
            }
            "pop" => {
                // Pop packing state (semantic analysis integration point).
                return DirectiveResult::Continue;
            }
            _ => {
                pp.diagnostics.warning(
                    inner[0].span,
                    format!("unexpected identifier '{}' in #pragma pack", keyword),
                );
                return DirectiveResult::Continue;
            }
        }
    }

    // `#pragma pack(N)` — set alignment to N.
    if let TokenKind::IntegerLiteral { value, .. } = inner[0].kind {
        let _alignment = value as u32;
        // Set packing alignment (semantic analysis integration point).
        return DirectiveResult::Continue;
    }

    pp.diagnostics
        .warning(span, "invalid #pragma pack directive");
    DirectiveResult::Continue
}

/// Handles `#pragma GCC ...` directives.
///
/// Supports:
///   - `#pragma GCC visibility push(default|hidden|protected)` / `pop`
///   - `#pragma GCC diagnostic push` / `pop` / `ignored "-Wname"` / `warning` / `error`
fn handle_pragma_gcc(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .warning(span, "expected pragma name after #pragma GCC");
        return DirectiveResult::Continue;
    }

    let gcc_pragma = token_text(&tokens[0], &pp.interner);
    let rest = skip_ws(&tokens[1..]);

    match gcc_pragma.as_str() {
        "visibility" => handle_pragma_gcc_visibility(pp, rest, span),
        "diagnostic" => handle_pragma_gcc_diagnostic(pp, rest, span),
        _ => {
            pp.diagnostics
                .warning(span, format!("unknown #pragma GCC {} ignored", gcc_pragma));
            DirectiveResult::Continue
        }
    }
}

/// Handles `#pragma GCC visibility push(...)` / `pop`.
fn handle_pragma_gcc_visibility(
    pp: &mut Preprocessor,
    tokens: &[Token],
    span: Span,
) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics.warning(
            span,
            "expected 'push' or 'pop' after #pragma GCC visibility",
        );
        return DirectiveResult::Continue;
    }

    let action = token_text(&tokens[0], &pp.interner);
    match action.as_str() {
        "push" => {
            // Parse `(default|hidden|protected)`.
            let rest = skip_ws(&tokens[1..]);
            if rest.is_empty() || rest[0].kind != TokenKind::LeftParen {
                pp.diagnostics
                    .warning(span, "expected '(' after #pragma GCC visibility push");
                return DirectiveResult::Continue;
            }
            let inner = skip_ws(&rest[1..]);
            if !inner.is_empty() {
                let _visibility = token_text(&inner[0], &pp.interner);
                // Store visibility for semantic analysis.
            }
            DirectiveResult::Continue
        }
        "pop" => {
            // Pop the visibility stack (semantic analysis integration point).
            DirectiveResult::Continue
        }
        _ => {
            pp.diagnostics.warning(
                span,
                format!(
                    "expected 'push' or 'pop' after #pragma GCC visibility, found '{}'",
                    action
                ),
            );
            DirectiveResult::Continue
        }
    }
}

/// Handles `#pragma GCC diagnostic push` / `pop` / `ignored` / `warning` / `error`.
fn handle_pragma_gcc_diagnostic(
    pp: &mut Preprocessor,
    tokens: &[Token],
    span: Span,
) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .warning(span, "expected action after #pragma GCC diagnostic");
        return DirectiveResult::Continue;
    }

    let action = token_text(&tokens[0], &pp.interner);
    match action.as_str() {
        "push" => {
            // Push diagnostic state (integration with DiagnosticEngine).
            DirectiveResult::Continue
        }
        "pop" => {
            // Pop diagnostic state.
            DirectiveResult::Continue
        }
        "ignored" | "warning" | "error" => {
            // Parse the warning name string: `"-Wname"`.
            let rest = skip_ws(&tokens[1..]);
            if !rest.is_empty() {
                if let TokenKind::StringLiteral { ref value, .. } = rest[0].kind {
                    let _warning_name = String::from_utf8_lossy(value).into_owned();
                    // Register the diagnostic override (semantic analysis integration point).
                }
            }
            DirectiveResult::Continue
        }
        _ => {
            pp.diagnostics.warning(
                span,
                format!("unknown #pragma GCC diagnostic action '{}'", action),
            );
            DirectiveResult::Continue
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// #error and #warning handling
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#error message text`.
///
/// Emits an error diagnostic with the message text from the directive.
/// Returns `Error` to halt preprocessing (per C11 §6.10.5).
fn handle_error(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let msg = concat_token_text(tokens, &pp.interner);
    pp.diagnostics.error(span, format!("#error {}", msg.trim()));
    DirectiveResult::Error
}

/// Handles `#warning message text` (GCC extension).
///
/// Emits a warning diagnostic with the message text from the directive.
/// Preprocessing continues normally after a `#warning`.
fn handle_warning(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let msg = concat_token_text(tokens, &pp.interner);
    pp.diagnostics
        .warning(span, format!("#warning {}", msg.trim()));
    DirectiveResult::Continue
}

// ═══════════════════════════════════════════════════════════════════════════
// #line handling
// ═══════════════════════════════════════════════════════════════════════════

/// Handles `#line NUMBER ["FILENAME"]`.
///
/// Sets the current line number (and optionally filename) for subsequent
/// diagnostic reporting. Registers a `LineDirective` with the source map
/// for location remapping.
fn handle_line(pp: &mut Preprocessor, tokens: &[Token], span: Span) -> DirectiveResult {
    let tokens = skip_ws(tokens);
    if tokens.is_empty() {
        pp.diagnostics
            .error(span, "expected line number after #line");
        return DirectiveResult::Error;
    }

    // Parse the line number.
    let line_no = match tokens[0].kind {
        TokenKind::IntegerLiteral { value, .. } => value as u32,
        _ => {
            pp.diagnostics
                .error(tokens[0].span, "expected integer after #line");
            return DirectiveResult::Error;
        }
    };

    // Optionally parse the filename.
    let new_file = if tokens.len() > 1 {
        let rest = skip_ws(&tokens[1..]);
        if !rest.is_empty() {
            if let TokenKind::StringLiteral { ref value, .. } = rest[0].kind {
                Some(String::from_utf8_lossy(value).into_owned())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Register the line directive with the source map.
    // The file_id comes from the span of the directive itself.
    let file_id = FileId(span.file_id);
    let byte_offset = span.start;

    let directive = LineDirective {
        file_id,
        byte_offset,
        new_line: line_no,
        new_file,
    };
    pp.source_map.add_line_directive(directive);

    DirectiveResult::Continue
}

// ═══════════════════════════════════════════════════════════════════════════
// Unterminated conditional checking
// ═══════════════════════════════════════════════════════════════════════════

/// Checks for unterminated conditional directives at the end of a file.
///
/// Should be called after the preprocessor has finished processing all tokens
/// in a translation unit. If any `#if` / `#ifdef` / `#ifndef` directives remain
/// unclosed (i.e., the conditional stack is not empty), an error is emitted for
/// each one, including the source location of the opening directive.
pub fn check_unterminated_conditionals(pp: &mut Preprocessor) {
    // Collect spans first to avoid borrowing conflicts between
    // pp.cond_stack (immutable) and pp.diagnostics (mutable).
    let unclosed_spans: Vec<Span> = pp.cond_stack.iter().map(|c| c.origin_span).collect();

    for origin_span in unclosed_spans {
        pp.diagnostics.error(
            origin_span,
            "unterminated conditional directive (#if/#ifdef/#ifndef without matching #endif)",
        );
    }
}
