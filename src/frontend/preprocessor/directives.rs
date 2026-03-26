#![allow(clippy::result_unit_err)]
//! Preprocessor directive handling for the BCC C compiler.
//!
//! This module implements all C11 preprocessor directives:
//! - `#include` — file inclusion with user and system path resolution
//! - `#define` / `#undef` — macro definition and undefinition
//! - `#if` / `#ifdef` / `#ifndef` / `#elif` / `#else` / `#endif` — conditional compilation
//! - `#pragma` — implementation-defined behavior (`once`, `pack`, `GCC`)
//! - `#error` / `#warning` — diagnostic emission
//! - `#line` — source location remapping
//!
//! The main entry point is [`process_directive`], which dispatches to the appropriate
//! handler based on the directive name token. Conditional compilation is tracked via
//! a stack of [`ConditionalState`] on the [`Preprocessor`] struct.

use std::path::{Path, PathBuf};

use super::expression::evaluate_pp_expression;
use super::include_handler::{IncludeError, IncludeHandler};
use super::macro_expander::MacroExpander;
use super::{ConditionalState, MacroDef, MacroKind, PPToken, PPTokenKind, Preprocessor};

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Severity, Span};
use crate::common::encoding::read_source_file;
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::LineDirective;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of processing a preprocessor directive.
///
/// Directives either produce output tokens (e.g., `#include` inserts the included
/// file's expanded tokens), produce no output (most directives modify preprocessor
/// state only), or signal a fatal error that halts preprocessing.
pub enum DirectiveResult {
    /// Tokens produced by the directive (e.g., expanded content from `#include`).
    Tokens(Vec<PPToken>),
    /// No output — the directive only modifies preprocessor state.
    None,
    /// Fatal error — preprocessing must halt immediately.
    Fatal,
}

// ---------------------------------------------------------------------------
// Main dispatcher
// ---------------------------------------------------------------------------

/// Process a preprocessor directive line.
///
/// `directive_token` is the token containing the directive name (e.g., `"include"`,
/// `"define"`, `"if"`). `tokens` contains the remaining tokens on the directive line
/// after the directive name.
///
/// This function handles both active and inactive conditional regions. In inactive
/// regions, only conditional directives (`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/
/// `#endif`) are processed to maintain proper nesting; all other directives are
/// skipped.
///
/// # Returns
///
/// - `Ok(DirectiveResult::Tokens(..))` — directive produced output tokens
/// - `Ok(DirectiveResult::None)` — directive processed with no output
/// - `Ok(DirectiveResult::Fatal)` — fatal error (e.g., `#error`)
/// - `Err(())` — unrecoverable error during directive processing
pub fn process_directive(
    pp: &mut Preprocessor,
    directive_token: &PPToken,
    tokens: &[PPToken],
) -> Result<DirectiveResult, ()> {
    let directive_name = directive_token.text.as_str();
    let directive_span = directive_token.span;

    // Determine whether the current region is active — every entry on the
    // conditional stack must be active for the region to be active.
    let is_active = is_preprocessing_active(pp);

    // ------------------------------------------------------------------
    // Inactive-region handling: only conditional directives are processed
    // so that nesting depth is tracked correctly.
    // ------------------------------------------------------------------
    if !is_active {
        match directive_name {
            "if" | "ifdef" | "ifndef" => {
                // Push a nested *inactive* conditional so that the matching
                // #endif will pop the correct level.
                pp.conditional_stack
                    .push(ConditionalState::new(false, directive_span));
                return Ok(DirectiveResult::None);
            }
            "elif" => return process_elif(pp, tokens, directive_span),
            "else" => return process_else(pp, tokens, directive_span),
            "endif" => return process_endif(pp, tokens, directive_span),
            // All other directives are silently skipped in inactive regions.
            _ => return Ok(DirectiveResult::None),
        }
    }

    // ------------------------------------------------------------------
    // Active-region handling: dispatch to the appropriate handler.
    // ------------------------------------------------------------------
    match directive_name {
        "include" => match process_include(pp, tokens, directive_span) {
            Ok(toks) => Ok(DirectiveResult::Tokens(toks)),
            Err(()) => Ok(DirectiveResult::Fatal),
        },
        "define" => {
            process_define(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "undef" => {
            process_undef(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "if" => {
            process_if(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "ifdef" => {
            process_ifdef(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "ifndef" => {
            process_ifndef(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "elif" => process_elif(pp, tokens, directive_span),
        "else" => process_else(pp, tokens, directive_span),
        "endif" => process_endif(pp, tokens, directive_span),
        "pragma" => {
            process_pragma(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        "error" => {
            process_error(pp, tokens, directive_span);
            Ok(DirectiveResult::Fatal)
        }
        "warning" => {
            process_warning(pp, tokens, directive_span);
            Ok(DirectiveResult::None)
        }
        "line" => {
            process_line(pp, tokens, directive_span)?;
            Ok(DirectiveResult::None)
        }
        // Null directive — an empty `#` on a line is a valid C11 no-op.
        "" => Ok(DirectiveResult::None),
        // Unknown directive — emit a diagnostic but do not halt.
        unknown => {
            pp.diagnostics.emit(Diagnostic::error(
                directive_span,
                format!("unknown preprocessor directive '#{}' ", unknown),
            ));
            Ok(DirectiveResult::None)
        }
    }
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Returns `true` when the preprocessor is inside an active conditional region.
/// All entries on the conditional stack must be active for the region to be active.
fn is_preprocessing_active(pp: &Preprocessor) -> bool {
    pp.conditional_stack.iter().all(|cs| cs.active)
}

/// Skip leading whitespace / newline tokens and return the remaining slice.
fn skip_whitespace(tokens: &[PPToken]) -> &[PPToken] {
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i].kind {
            PPTokenKind::Whitespace | PPTokenKind::Newline => i += 1,
            _ => break,
        }
    }
    &tokens[i..]
}

/// Concatenate token text into a single message string (for `#error` / `#warning`).
fn tokens_to_message(tokens: &[PPToken]) -> String {
    let mut msg = String::new();
    for tok in tokens {
        if tok.kind == PPTokenKind::EndOfFile || tok.kind == PPTokenKind::Newline {
            break;
        }
        msg.push_str(&tok.text);
    }
    msg.trim().to_string()
}

/// Remove trailing whitespace tokens from a replacement-token list.
fn trim_trailing_whitespace(mut tokens: Vec<PPToken>) -> Vec<PPToken> {
    while let Some(last) = tokens.last() {
        if last.kind == PPTokenKind::Whitespace || last.kind == PPTokenKind::Newline {
            tokens.pop();
        } else {
            break;
        }
    }
    tokens
}

/// Warn if there are unexpected extra tokens after a directive that should
/// have consumed all meaningful input.
fn warn_extra_tokens(diagnostics: &mut DiagnosticEngine, tokens: &[PPToken]) {
    let rest = skip_whitespace(tokens);
    if let Some(first) = rest.first() {
        if first.kind != PPTokenKind::EndOfFile && first.kind != PPTokenKind::Newline {
            diagnostics.emit(Diagnostic::warning(
                first.span,
                "extra tokens at end of directive".to_string(),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// #define directive
// ---------------------------------------------------------------------------

/// Process `#define` — create a macro definition.
///
/// Handles both object-like macros (`#define FOO value`) and function-like macros
/// (`#define FOO(a, b) ...`).  Function-like macros are distinguished by an open
/// parenthesis *immediately* following the macro name (no intervening whitespace).
///
/// Variadic macros (`#define FOO(a, ...)`) use `__VA_ARGS__` in the replacement list.
fn process_define(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    // First token must be the macro name (an identifier).
    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Identifier {
        pp.diagnostics.emit_error(
            directive_span,
            "macro name missing in #define directive".to_string(),
        );
        return Err(());
    }

    let macro_name = tokens[0].text.clone();
    let name_span = tokens[0].span;
    let rest = &tokens[1..];

    // Distinguish function-like from object-like: function-like requires `(`
    // immediately after the macro name (no whitespace between).
    let (kind, replacement_start) = if !rest.is_empty()
        && rest[0].kind == PPTokenKind::Punctuator
        && rest[0].text == "("
        && name_span.end == rest[0].span.start
    {
        // Function-like macro — parse parameter list.
        parse_function_like_params(pp, rest, directive_span)?
    } else {
        // Object-like macro — skip leading whitespace before the replacement list.
        let replacement_tokens = skip_whitespace(rest);
        (MacroKind::ObjectLike, replacement_tokens)
    };

    // Collect replacement tokens up to end-of-line / EOF.
    let replacement: Vec<PPToken> = replacement_start
        .iter()
        .take_while(|t| t.kind != PPTokenKind::EndOfFile && t.kind != PPTokenKind::Newline)
        .cloned()
        .collect();

    // Remove trailing whitespace from the replacement list.
    let replacement = trim_trailing_whitespace(replacement);

    // C11 §6.10.3p2 — warn on non-identical redefinition.
    if let Some(existing) = pp.macro_defs.get(&macro_name) {
        if existing.is_predefined {
            pp.diagnostics.emit_warning(
                name_span,
                format!("redefining predefined macro '{}'", macro_name),
            );
        } else if !macro_definitions_equivalent(existing, &kind, &replacement) {
            pp.diagnostics.emit(
                Diagnostic::warning(name_span, format!("'{}' macro redefined", macro_name))
                    .with_note(
                        existing.definition_span,
                        "previous definition was here".to_string(),
                    ),
            );
        }
    }

    let macro_def = MacroDef {
        name: macro_name.clone(),
        kind,
        replacement,
        is_predefined: false,
        definition_span: name_span,
    };

    pp.macro_defs.insert(macro_name, macro_def);
    Ok(())
}

/// Parse the parameter list of a function-like macro definition.
///
/// `tokens[0]` must be the opening `(`.  Returns the parsed `MacroKind` and a
/// slice pointing at the first replacement token (after the closing `)`).
fn parse_function_like_params<'a>(
    pp: &mut Preprocessor,
    tokens: &'a [PPToken],
    directive_span: Span,
) -> Result<(MacroKind, &'a [PPToken]), ()> {
    // tokens[0] is '(' — skip it.
    let mut i: usize = 1;
    let mut params: Vec<String> = Vec::new();
    let mut variadic = false;

    loop {
        // Skip whitespace inside the parameter list.
        while i < tokens.len()
            && matches!(
                tokens[i].kind,
                PPTokenKind::Whitespace | PPTokenKind::Newline
            )
        {
            i += 1;
        }

        if i >= tokens.len() {
            pp.diagnostics.emit(Diagnostic::error(
                directive_span,
                "unterminated parameter list in function-like macro".to_string(),
            ));
            return Err(());
        }

        // Closing paren → end of parameter list (handles empty list too).
        if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ")" {
            i += 1;
            break;
        }

        // Variadic `...`
        if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == "..." {
            variadic = true;
            i += 1;
            // Expect closing paren after `...`.
            while i < tokens.len()
                && matches!(
                    tokens[i].kind,
                    PPTokenKind::Whitespace | PPTokenKind::Newline
                )
            {
                i += 1;
            }
            if i >= tokens.len()
                || tokens[i].kind != PPTokenKind::Punctuator
                || tokens[i].text != ")"
            {
                pp.diagnostics.emit(Diagnostic::error(
                    directive_span,
                    "expected ')' after '...' in macro parameter list".to_string(),
                ));
                return Err(());
            }
            i += 1;
            break;
        }

        // Expect a parameter name (identifier).
        if tokens[i].kind != PPTokenKind::Identifier {
            pp.diagnostics.emit(Diagnostic::error(
                tokens[i].span,
                format!("expected parameter name, found '{}'", tokens[i].text),
            ));
            return Err(());
        }

        let param_name = tokens[i].text.clone();

        // Reject duplicate parameter names.
        if params.contains(&param_name) {
            pp.diagnostics.emit(Diagnostic::error(
                tokens[i].span,
                format!("duplicate macro parameter name '{}'", param_name),
            ));
            return Err(());
        }
        params.push(param_name);
        i += 1;

        // Skip whitespace after parameter name.
        while i < tokens.len()
            && matches!(
                tokens[i].kind,
                PPTokenKind::Whitespace | PPTokenKind::Newline
            )
        {
            i += 1;
        }
        if i >= tokens.len() {
            pp.diagnostics.emit(Diagnostic::error(
                directive_span,
                "unterminated parameter list in function-like macro".to_string(),
            ));
            return Err(());
        }

        // After a parameter name expect ',' or ')'.
        if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == "," {
            i += 1;
            // After a comma, check for variadic `...`.
            while i < tokens.len()
                && matches!(
                    tokens[i].kind,
                    PPTokenKind::Whitespace | PPTokenKind::Newline
                )
            {
                i += 1;
            }
            if i < tokens.len()
                && tokens[i].kind == PPTokenKind::Punctuator
                && tokens[i].text == "..."
            {
                variadic = true;
                i += 1;
                while i < tokens.len()
                    && matches!(
                        tokens[i].kind,
                        PPTokenKind::Whitespace | PPTokenKind::Newline
                    )
                {
                    i += 1;
                }
                if i >= tokens.len()
                    || tokens[i].kind != PPTokenKind::Punctuator
                    || tokens[i].text != ")"
                {
                    pp.diagnostics.emit(Diagnostic::error(
                        directive_span,
                        "expected ')' after '...' in macro parameter list".to_string(),
                    ));
                    return Err(());
                }
                i += 1;
                break;
            }
            // Otherwise continue to next parameter.
        } else if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ")" {
            i += 1;
            break;
        } else {
            pp.diagnostics.emit(Diagnostic::error(
                tokens[i].span,
                format!(
                    "expected ',' or ')' in macro parameter list, found '{}'",
                    tokens[i].text
                ),
            ));
            return Err(());
        }
    }

    let kind = MacroKind::FunctionLike { params, variadic };
    // Skip whitespace before the replacement list.
    let replacement = skip_whitespace(&tokens[i..]);
    Ok((kind, replacement))
}

/// C11 §6.10.3p2 — two macro definitions are equivalent when they have the
/// same kind (object-like / function-like with identical parameters) and their
/// replacement token lists match token-by-token (ignoring whitespace differences).
fn macro_definitions_equivalent(
    existing: &MacroDef,
    new_kind: &MacroKind,
    new_replacement: &[PPToken],
) -> bool {
    match (&existing.kind, new_kind) {
        (MacroKind::ObjectLike, MacroKind::ObjectLike) => {}
        (
            MacroKind::FunctionLike {
                params: ep,
                variadic: ev,
            },
            MacroKind::FunctionLike {
                params: np,
                variadic: nv,
            },
        ) => {
            if ep != np || ev != nv {
                return false;
            }
        }
        _ => return false,
    }

    let existing_sig: Vec<_> = existing
        .replacement
        .iter()
        .filter(|t| !matches!(t.kind, PPTokenKind::Whitespace | PPTokenKind::Newline))
        .collect();
    let new_sig: Vec<_> = new_replacement
        .iter()
        .filter(|t| !matches!(t.kind, PPTokenKind::Whitespace | PPTokenKind::Newline))
        .collect();

    if existing_sig.len() != new_sig.len() {
        return false;
    }

    existing_sig
        .iter()
        .zip(new_sig.iter())
        .all(|(a, b)| a.kind == b.kind && a.text == b.text)
}

// ---------------------------------------------------------------------------
// #undef directive
// ---------------------------------------------------------------------------

/// Process `#undef` — remove a macro definition.
///
/// No error is produced if the macro was not previously defined (C11 §6.10.3.5).
/// A warning is emitted when attempting to undefine a predefined macro.
fn process_undef(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Identifier {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "macro name required after #undef".to_string(),
        ));
        return Err(());
    }

    let macro_name = &tokens[0].text;
    let name_span = tokens[0].span;

    // Warn when attempting to undefine a predefined macro.
    if let Some(existing) = pp.macro_defs.get(macro_name.as_str()) {
        if existing.is_predefined {
            pp.diagnostics.emit(Diagnostic::warning(
                name_span,
                format!("undefining predefined macro '{}'", macro_name),
            ));
        }
    }

    // Remove — no error if not previously defined.
    pp.macro_defs.remove(macro_name.as_str());

    // Warn about trailing junk.
    warn_extra_tokens(&mut *pp.diagnostics, &tokens[1..]);

    Ok(())
}

// ---------------------------------------------------------------------------
// #include directive
// ---------------------------------------------------------------------------

/// Process `#include` — include the contents of another source file.
///
/// Supports `#include "header.h"` (user paths), `#include <header.h>` (system
/// paths), and computed includes (macro-expanded header names).  The included
/// file is read with PUA encoding for non-UTF-8 byte fidelity, registered in
/// the source map, and recursively preprocessed.
fn process_include(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<Vec<PPToken>, ()> {
    let tokens = skip_whitespace(tokens);

    if tokens.is_empty() || tokens[0].kind == PPTokenKind::EndOfFile {
        pp.diagnostics.emit_error(
            directive_span,
            "expected header name after #include".to_string(),
        );
        return Err(());
    }

    // Parse the header specification.
    let (header_name, is_system) = parse_include_header(pp, tokens, directive_span)?;

    // Determine the file that contains the #include so relative paths resolve
    // correctly.
    let including_file = pp
        .source_map
        .get_filename(directive_span.file_id)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    // Create an include handler for path resolution and include-stack management.
    let mut handler =
        IncludeHandler::new(pp.include_paths.clone(), pp.system_include_paths.clone());

    // Resolve to an absolute path.
    let resolved_path = match handler.resolve_include(&header_name, is_system, &including_file) {
        Some(path) => path,
        None => {
            pp.diagnostics.emit(Diagnostic::error(
                directive_span,
                format!("'{}' file not found", header_name),
            ));
            return Err(());
        }
    };

    // Check include guards and #pragma once.
    if handler.should_skip_file(&resolved_path, &pp.macro_defs) {
        return Ok(Vec::new());
    }

    // Push the file onto the include stack (checks circular includes and depth).
    if let Err(inc_err) = handler.push_include(&resolved_path) {
        match inc_err {
            IncludeError::Circular(_path) => {
                pp.diagnostics.emit_error(
                    directive_span,
                    format!("circular #include of '{}'", resolved_path.display()),
                );
            }
            IncludeError::TooDeep(_depth) => {
                pp.diagnostics.emit_error(
                    directive_span,
                    format!(
                        "#include nested too deeply (depth {} exceeds maximum {})",
                        pp.include_depth, pp.max_recursion_depth
                    ),
                );
            }
            IncludeError::NotFound(ref name) => {
                pp.diagnostics.emit(Diagnostic::error(
                    directive_span,
                    format!("'{}' file not found", name),
                ));
            }
            IncludeError::IoError(ref io_err) => {
                pp.diagnostics.emit(Diagnostic::error(
                    directive_span,
                    format!("cannot open '{}': {}", resolved_path.display(), io_err),
                ));
            }
        }
        return Err(());
    }

    // Read the file contents.  Try the handler first; fall back to the raw
    // PUA-encoded reader.
    let source_content = match handler.read_include_file(&resolved_path) {
        Ok(content) => content,
        Err(_) => match read_source_file(&resolved_path) {
            Ok(content) => content,
            Err(e) => {
                handler.pop_include();
                pp.diagnostics.emit_error(
                    directive_span,
                    format!("cannot read '{}': {}", resolved_path.display(), e),
                );
                return Err(());
            }
        },
    };

    // Register the new file in the source map.
    let file_id = pp.source_map.add_file(
        resolved_path.to_string_lossy().to_string(),
        source_content.clone(),
    );

    // Apply Phase 1 transforms (trigraphs and line splicing).
    let processed = super::phase1_line_splice(&super::phase1_trigraphs(&source_content));

    // Tokenize the included source.
    let included_tokens = super::tokenize_preprocessing(&processed, file_id);

    // Recursively preprocess the included file.
    pp.include_depth += 1;
    let result = pp.process_tokens(&included_tokens);
    pp.include_depth -= 1;

    // Pop the include stack entry.
    handler.pop_include();

    // Attempt to detect and register include guards for future optimisation.
    detect_and_register_guard(&mut handler, &resolved_path, &pp.macro_defs);

    Ok(result)
}

/// Parse the header name from the tokens following `#include`.
///
/// Recognises three forms:
/// 1. `<header.h>` — `PPTokenKind::HeaderName` starting with `<`
/// 2. `"header.h"` — `PPTokenKind::StringLiteral` or `HeaderName` starting with `"`
/// 3. Computed include — macro-expand the tokens and re-parse.
fn parse_include_header(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(String, bool), ()> {
    if tokens.is_empty() {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "expected header name after #include".to_string(),
        ));
        return Err(());
    }

    match tokens[0].kind {
        PPTokenKind::HeaderName => {
            let text = &tokens[0].text;
            let is_system = text.starts_with('<');
            // Strip the delimiters (< > or " ").
            let header = text[1..text.len().saturating_sub(1)].to_string();
            Ok((header, is_system))
        }
        PPTokenKind::StringLiteral => {
            let text = &tokens[0].text;
            let header = text[1..text.len().saturating_sub(1)].to_string();
            Ok((header, false))
        }
        _ => {
            // Computed include — macro-expand and re-parse.
            let expanded = {
                let mut expander = MacroExpander::new(
                    &pp.macro_defs,
                    &mut *pp.diagnostics,
                    pp.max_recursion_depth,
                    &mut pp.counter_value,
                );
                expander.expand_tokens(tokens)
            };
            let expanded = skip_whitespace(&expanded);
            if expanded.is_empty() {
                pp.diagnostics.emit(Diagnostic::error(
                    directive_span,
                    "empty expansion in computed #include".to_string(),
                ));
                return Err(());
            }
            match expanded[0].kind {
                PPTokenKind::HeaderName => {
                    let text = &expanded[0].text;
                    let is_system = text.starts_with('<');
                    let header = text[1..text.len().saturating_sub(1)].to_string();
                    Ok((header, is_system))
                }
                PPTokenKind::StringLiteral => {
                    let text = &expanded[0].text;
                    let header = text[1..text.len().saturating_sub(1)].to_string();
                    Ok((header, false))
                }
                PPTokenKind::Punctuator if expanded[0].text == "<" => {
                    // Reconstruct `<header.h>` from individual tokens.
                    let mut header = String::new();
                    for tok in &expanded[1..] {
                        if tok.kind == PPTokenKind::Punctuator && tok.text == ">" {
                            return Ok((header, true));
                        }
                        header.push_str(&tok.text);
                    }
                    pp.diagnostics.emit(Diagnostic::error(
                        directive_span,
                        "unterminated '<' in computed #include".to_string(),
                    ));
                    Err(())
                }
                _ => {
                    pp.diagnostics.emit(Diagnostic::error(
                        directive_span,
                        "computed #include did not produce a valid header name".to_string(),
                    ));
                    Err(())
                }
            }
        }
    }
}

/// Heuristic include-guard registration.
///
/// When a file's name maps to a plausible guard macro (`FOO_H` for `foo.h`)
/// that is currently defined, register the guard with the include handler so
/// that subsequent inclusions of the same file can be short-circuited.
fn detect_and_register_guard(
    handler: &mut IncludeHandler,
    path: &Path,
    macro_defs: &FxHashMap<String, MacroDef>,
) {
    let guard_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|name| name.replace('.', "_").replace('-', "_").to_uppercase());

    if let Some(guard) = guard_name {
        if macro_defs.contains_key(&guard) {
            handler.register_guard(path, guard);
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional compilation directives
// ---------------------------------------------------------------------------

/// Resolve `defined(X)` and `defined X` operators in a preprocessor
/// expression token stream.  Replaces each `defined` operator sequence
/// with a `1` or `0` pp-number token.  This MUST run before macro
/// expansion so that `#define defined ...` cannot alter semantics
/// (C11 §6.10.1p4).
pub fn resolve_defined_operators(
    tokens: &[PPToken],
    macro_defs: &FxHashMap<String, MacroDef>,
) -> Vec<PPToken> {
    let mut result: Vec<PPToken> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        if tok.kind == PPTokenKind::Identifier && tok.text == "defined" {
            let span = tok.span;
            // Skip any whitespace after `defined`.
            let mut j = i + 1;
            while j < tokens.len() && tokens[j].kind == PPTokenKind::Whitespace {
                j += 1;
            }
            if j < tokens.len()
                && tokens[j].kind == PPTokenKind::Punctuator
                && tokens[j].text == "("
            {
                // `defined ( IDENT )`  form.
                let mut k = j + 1;
                while k < tokens.len() && tokens[k].kind == PPTokenKind::Whitespace {
                    k += 1;
                }
                if k < tokens.len() && tokens[k].kind == PPTokenKind::Identifier {
                    let name = &tokens[k].text;
                    let is_def = macro_defs.contains_key(name.as_str());
                    // Skip past closing `)`.
                    let mut m = k + 1;
                    while m < tokens.len() && tokens[m].kind == PPTokenKind::Whitespace {
                        m += 1;
                    }
                    if m < tokens.len()
                        && tokens[m].kind == PPTokenKind::Punctuator
                        && tokens[m].text == ")"
                    {
                        result.push(PPToken {
                            kind: PPTokenKind::Number,
                            text: if is_def {
                                "1".to_string()
                            } else {
                                "0".to_string()
                            },
                            span,
                            from_macro: false,
                            painted: false,
                        });
                        i = m + 1;
                        continue;
                    }
                }
                // Malformed `defined(` — fall through and keep tokens as-is.
            } else if j < tokens.len() && tokens[j].kind == PPTokenKind::Identifier {
                // `defined IDENT` form (no parentheses).
                let name = &tokens[j].text;
                let is_def = macro_defs.contains_key(name.as_str());
                result.push(PPToken {
                    kind: PPTokenKind::Number,
                    text: if is_def {
                        "1".to_string()
                    } else {
                        "0".to_string()
                    },
                    span,
                    from_macro: false,
                    painted: false,
                });
                i = j + 1;
                continue;
            }
            // Malformed `defined` at end of tokens — keep as-is.
        }
        result.push(tok.clone());
        i += 1;
    }
    result
}

/// Process `#if` — begin a conditional block whose activity depends on the
/// value of a preprocessor constant expression.
fn process_if(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    // Step 1: Resolve `defined(X)` / `defined X` operators BEFORE macro expansion.
    let with_defined = resolve_defined_operators(tokens, &pp.macro_defs);

    // Step 2: Macro-expand the remaining expression tokens.
    let expanded = {
        let mut expander = MacroExpander::new(
            &pp.macro_defs,
            &mut *pp.diagnostics,
            pp.max_recursion_depth,
            &mut pp.counter_value,
        );
        expander.expand_tokens(&with_defined)
    };

    // Step 3: Evaluate the preprocessor constant expression.
    let result = match evaluate_pp_expression(&expanded, &mut *pp.diagnostics) {
        Ok(value) => value.is_nonzero(),
        Err(()) => {
            // On evaluation failure treat as false so that nesting is still
            // tracked correctly.
            false
        }
    };

    pp.conditional_stack
        .push(ConditionalState::new(result, directive_span));
    Ok(())
}

/// Process `#ifdef` — begin a conditional block that is active when a macro is
/// defined.
fn process_ifdef(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Identifier {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "macro name required after #ifdef".to_string(),
        ));
        return Err(());
    }

    let macro_name = &tokens[0].text;
    let is_defined = pp.macro_defs.contains_key(macro_name.as_str());

    pp.conditional_stack
        .push(ConditionalState::new(is_defined, directive_span));

    warn_extra_tokens(&mut *pp.diagnostics, &tokens[1..]);
    Ok(())
}

/// Process `#ifndef` — begin a conditional block that is active when a macro
/// is *not* defined.  This is the canonical include-guard pattern.
fn process_ifndef(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Identifier {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "macro name required after #ifndef".to_string(),
        ));
        return Err(());
    }

    let macro_name = &tokens[0].text;
    let is_defined = pp.macro_defs.contains_key(macro_name.as_str());

    pp.conditional_stack
        .push(ConditionalState::new(!is_defined, directive_span));

    warn_extra_tokens(&mut *pp.diagnostics, &tokens[1..]);
    Ok(())
}

/// Process `#elif` — transition to an alternative conditional branch.
///
/// Rules:
/// - There must be an open conditional block on the stack.
/// - `#elif` must not appear after `#else`.
/// - If no prior branch has been active *and* the parent is active, evaluate
///   the expression.  If it is non-zero, activate this branch.
fn process_elif(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<DirectiveResult, ()> {
    let stack_len = pp.conditional_stack.len();
    if stack_len == 0 {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "#elif without matching #if".to_string(),
        ));
        return Err(());
    }

    // Check for #elif after #else — that is an error.
    if pp.conditional_stack[stack_len - 1].seen_else {
        let opening = pp.conditional_stack[stack_len - 1].opening_span;
        pp.diagnostics.emit(
            Diagnostic::error(directive_span, "#elif after #else".to_string())
                .with_note(opening, "conditional block started here".to_string()),
        );
        return Err(());
    }

    // Determine whether the parent context is active.
    let parent_active = if stack_len > 1 {
        pp.conditional_stack[..stack_len - 1]
            .iter()
            .all(|cs| cs.active)
    } else {
        true
    };

    if !parent_active {
        // Parent is inactive → this branch is unconditionally inactive.
        pp.conditional_stack[stack_len - 1].active = false;
        return Ok(DirectiveResult::None);
    }

    // If a previous branch was already active, deactivate.
    if pp.conditional_stack[stack_len - 1].seen_active {
        pp.conditional_stack[stack_len - 1].active = false;
        return Ok(DirectiveResult::None);
    }

    // Evaluate the condition.
    let tokens = skip_whitespace(tokens);
    // Step 1: Resolve `defined(X)` / `defined X` operators.
    let with_defined = resolve_defined_operators(tokens, &pp.macro_defs);
    // Step 2: Macro-expand remaining tokens.
    let expanded = {
        let mut expander = MacroExpander::new(
            &pp.macro_defs,
            &mut *pp.diagnostics,
            pp.max_recursion_depth,
            &mut pp.counter_value,
        );
        expander.expand_tokens(&with_defined)
    };

    let result = match evaluate_pp_expression(&expanded, &mut *pp.diagnostics) {
        Ok(value) => value.is_nonzero(),
        Err(()) => false,
    };

    let state = &mut pp.conditional_stack[stack_len - 1];
    state.active = result;
    if result {
        state.seen_active = true;
    }

    Ok(DirectiveResult::None)
}

/// Process `#else` — activate the fallback branch if no prior branch was active.
fn process_else(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<DirectiveResult, ()> {
    let stack_len = pp.conditional_stack.len();
    if stack_len == 0 {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "#else without matching #if".to_string(),
        ));
        return Err(());
    }

    if pp.conditional_stack[stack_len - 1].seen_else {
        let opening = pp.conditional_stack[stack_len - 1].opening_span;
        pp.diagnostics.emit(
            Diagnostic::error(directive_span, "duplicate #else".to_string())
                .with_note(opening, "conditional block started here".to_string()),
        );
        return Err(());
    }

    pp.conditional_stack[stack_len - 1].seen_else = true;

    // Check parent context.
    let parent_active = if stack_len > 1 {
        pp.conditional_stack[..stack_len - 1]
            .iter()
            .all(|cs| cs.active)
    } else {
        true
    };

    if !parent_active {
        pp.conditional_stack[stack_len - 1].active = false;
    } else {
        let already_taken = pp.conditional_stack[stack_len - 1].seen_active;
        pp.conditional_stack[stack_len - 1].active = !already_taken;
        if !already_taken {
            pp.conditional_stack[stack_len - 1].seen_active = true;
        }
    }

    // Warn about trailing tokens.
    warn_extra_tokens(&mut *pp.diagnostics, tokens);

    Ok(DirectiveResult::None)
}

/// Process `#endif` — close the most recent conditional block.
fn process_endif(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<DirectiveResult, ()> {
    if pp.conditional_stack.is_empty() {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "#endif without matching #if".to_string(),
        ));
        return Err(());
    }

    pp.conditional_stack.pop();

    // Warn about trailing tokens.
    warn_extra_tokens(&mut *pp.diagnostics, tokens);

    Ok(DirectiveResult::None)
}

// ---------------------------------------------------------------------------
// #pragma directive
// ---------------------------------------------------------------------------

/// Process `#pragma` — implementation-defined behaviour.
///
/// Recognised pragmas:
/// - `#pragma once` — prevent redundant re-inclusion of the current file.
/// - `#pragma pack(...)` — struct packing control (push/pop/set).
/// - `#pragma GCC ...` / `#pragma clang ...` — silently accepted.
///
/// Unknown pragmas are silently ignored per C11 §6.10.6.
fn process_pragma(
    pp: &mut Preprocessor,
    tokens: &[PPToken],
    directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    // Empty pragma — silently ignore.
    if tokens.is_empty()
        || tokens[0].kind == PPTokenKind::EndOfFile
        || tokens[0].kind == PPTokenKind::Newline
    {
        return Ok(());
    }

    match tokens[0].text.as_str() {
        "once" => {
            // Mark the current file as "include once" via the include handler.
            let filename = pp
                .source_map
                .get_filename(directive_span.file_id)
                .unwrap_or("")
                .to_string();
            if !filename.is_empty() {
                let path = PathBuf::from(&filename);
                let mut handler =
                    IncludeHandler::new(pp.include_paths.clone(), pp.system_include_paths.clone());
                handler.mark_pragma_once(&path);
            }
            Ok(())
        }
        "pack" => process_pragma_pack(pp, &tokens[1..], directive_span),
        "push_macro" => process_pragma_push_macro(pp, &tokens[1..]),
        "pop_macro" => process_pragma_pop_macro(pp, &tokens[1..]),
        // GCC / Clang pragmas — accept silently.
        "GCC" | "clang" => Ok(()),
        // All other pragmas — silently ignore (C11 §6.10.6).
        _ => Ok(()),
    }
}

/// Process `#pragma pack(...)` — struct alignment control.
///
/// Accepted forms:
/// - `#pragma pack(push, N)` — push current alignment and set to `N`
/// - `#pragma pack(pop)` — restore previous alignment
/// - `#pragma pack(N)` — set alignment to `N`
/// - `#pragma pack()` — reset to default alignment
fn process_pragma_pack(
    _pp: &mut Preprocessor,
    tokens: &[PPToken],
    _directive_span: Span,
) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    // Expect '(' after 'pack'.
    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Punctuator || tokens[0].text != "(" {
        // Bare `#pragma pack` without arguments — reset to default.
        return Ok(());
    }

    let mut i: usize = 1; // skip '('

    // Skip whitespace.
    while i < tokens.len()
        && matches!(
            tokens[i].kind,
            PPTokenKind::Whitespace | PPTokenKind::Newline
        )
    {
        i += 1;
    }

    if i >= tokens.len() {
        return Ok(());
    }

    // Closing paren immediately → `#pragma pack()` (reset).
    if tokens[i].kind == PPTokenKind::Punctuator && tokens[i].text == ")" {
        return Ok(());
    }

    match tokens[i].text.as_str() {
        "push" => {
            // `#pragma pack(push [, N])`
            i += 1;
            while i < tokens.len()
                && matches!(
                    tokens[i].kind,
                    PPTokenKind::Whitespace | PPTokenKind::Newline
                )
            {
                i += 1;
            }
            // Consume optional `, N`
            if i < tokens.len()
                && tokens[i].kind == PPTokenKind::Punctuator
                && tokens[i].text == ","
            {
                i += 1;
                // The alignment value follows — consume but do not act on it
                // (packing state is managed at a higher level).
                while i < tokens.len()
                    && matches!(
                        tokens[i].kind,
                        PPTokenKind::Whitespace | PPTokenKind::Newline
                    )
                {
                    i += 1;
                }
                // Alignment number (consumed, not stored locally).
                if i < tokens.len() && tokens[i].kind == PPTokenKind::Number {
                    let _alignment = &tokens[i].text;
                    let _ = i; // consumed
                }
            }
        }
        "pop" => { /* `#pragma pack(pop)` — handled at higher level. */ }
        _ => {
            // `#pragma pack(N)` — set alignment to N.
            if tokens[i].kind == PPTokenKind::Number {
                let _alignment = &tokens[i].text;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// #pragma push_macro / pop_macro
// ---------------------------------------------------------------------------

/// Extract the macro name from `("name")` token sequence.
fn extract_pragma_macro_name(tokens: &[PPToken]) -> Option<String> {
    let tokens = skip_whitespace(tokens);
    if tokens.is_empty() || tokens[0].text != "(" {
        return None;
    }
    let inner = skip_whitespace(&tokens[1..]);
    if inner.is_empty() {
        return None;
    }
    // Accept both "name" (quoted) and name (unquoted).
    let name = if inner[0].text.starts_with('"')
        && inner[0].text.ends_with('"')
        && inner[0].text.len() >= 2
    {
        inner[0].text[1..inner[0].text.len() - 1].to_string()
    } else {
        inner[0].text.clone()
    };
    Some(name)
}

/// Process `#pragma push_macro("name")` — save the current definition of
/// the named macro onto a per-name stack.
fn process_pragma_push_macro(pp: &mut Preprocessor, tokens: &[PPToken]) -> Result<(), ()> {
    if let Some(name) = extract_pragma_macro_name(tokens) {
        let current_def = pp.macro_defs.get(&name).cloned();
        pp.macro_push_stack
            .entry(name)
            .or_default()
            .push(current_def);
    }
    Ok(())
}

/// Process `#pragma pop_macro("name")` — restore the most recently pushed
/// definition of the named macro.
fn process_pragma_pop_macro(pp: &mut Preprocessor, tokens: &[PPToken]) -> Result<(), ()> {
    if let Some(name) = extract_pragma_macro_name(tokens) {
        if let Some(stack) = pp.macro_push_stack.get_mut(&name) {
            if let Some(saved_def) = stack.pop() {
                match saved_def {
                    Some(def) => {
                        pp.macro_defs.insert(name, def);
                    }
                    None => {
                        pp.macro_defs.remove(&name);
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// #error and #warning directives
// ---------------------------------------------------------------------------

/// Process `#error` — emit a fatal diagnostic constructed from the remaining
/// tokens on the directive line.
fn process_error(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) {
    let message = tokens_to_message(tokens);
    let diag_span = if !tokens.is_empty() && tokens[0].kind != PPTokenKind::EndOfFile {
        Span::new(
            directive_span.file_id,
            directive_span.start,
            tokens
                .iter()
                .rev()
                .find(|t| t.kind != PPTokenKind::EndOfFile && t.kind != PPTokenKind::Newline)
                .map_or(directive_span.end, |t| t.span.end),
        )
    } else {
        directive_span
    };
    pp.diagnostics
        .emit(Diagnostic::error(diag_span, format!("#error {}", message)));
}

/// Process `#warning` — emit a non-fatal warning diagnostic.
fn process_warning(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) {
    let message = tokens_to_message(tokens);
    let diag_span = if !tokens.is_empty() && tokens[0].kind != PPTokenKind::EndOfFile {
        Span::new(
            directive_span.file_id,
            directive_span.start,
            tokens
                .iter()
                .rev()
                .find(|t| t.kind != PPTokenKind::EndOfFile && t.kind != PPTokenKind::Newline)
                .map_or(directive_span.end, |t| t.span.end),
        )
    } else {
        directive_span
    };
    pp.diagnostics.emit(Diagnostic::warning(
        diag_span,
        format!("#warning {}", message),
    ));
}

// ---------------------------------------------------------------------------
// #line directive
// ---------------------------------------------------------------------------

/// Process `#line` — remap subsequent source locations.
///
/// Accepted forms:
/// - `#line 42` — set line number
/// - `#line 42 "file.c"` — set line number and filename
fn process_line(pp: &mut Preprocessor, tokens: &[PPToken], directive_span: Span) -> Result<(), ()> {
    let tokens = skip_whitespace(tokens);

    if tokens.is_empty() || tokens[0].kind != PPTokenKind::Number {
        pp.diagnostics.emit(Diagnostic::error(
            directive_span,
            "expected line number after #line".to_string(),
        ));
        return Err(());
    }

    // Parse the line number.
    let line_num: u32 = match tokens[0].text.parse::<u32>() {
        Ok(n) if n > 0 => n,
        _ => {
            pp.diagnostics.emit(Diagnostic::error(
                tokens[0].span,
                format!(
                    "invalid line number '{}' in #line directive",
                    tokens[0].text
                ),
            ));
            return Err(());
        }
    };

    // Optional filename argument.
    let rest = skip_whitespace(&tokens[1..]);
    let new_filename = if !rest.is_empty() && rest[0].kind == PPTokenKind::StringLiteral {
        let text = &rest[0].text;
        Some(text[1..text.len().saturating_sub(1)].to_string())
    } else {
        None
    };

    // Build and register the line directive.
    let line_directive = LineDirective {
        file_id: directive_span.file_id,
        directive_offset: directive_span.start,
        new_line: line_num,
        new_filename,
    };
    pp.source_map.add_line_directive(line_directive);

    Ok(())
}

// ---------------------------------------------------------------------------
// Public utilities
// ---------------------------------------------------------------------------

/// Verify that the conditional stack is empty at end-of-file.
///
/// Call this after preprocessing is complete.  Any remaining entries on the
/// stack indicate unterminated `#if` / `#ifdef` / `#ifndef` blocks.
pub fn verify_no_unterminated_conditionals(pp: &mut Preprocessor) {
    // Collect the opening spans first to avoid holding an immutable borrow
    // on the conditional stack while calling the diagnostic engine.
    let unterminated_spans: Vec<Span> = pp
        .conditional_stack
        .iter()
        .map(|state| state.opening_span)
        .collect();

    for span in unterminated_spans {
        pp.diagnostics.emit(Diagnostic::error(
            span,
            "unterminated conditional directive".to_string(),
        ));
        // Attach a note pointing at where the conditional was opened.
        pp.diagnostics.emit(Diagnostic::note(
            span,
            "conditional block opened here was never closed with #endif".to_string(),
        ));
    }
}

/// Emit a diagnostic at a chosen severity level.
///
/// This is a convenience wrapper that dispatches to the appropriate
/// `Diagnostic` constructor based on the `Severity` value.
#[allow(dead_code)]
fn emit_at_severity(
    diagnostics: &mut DiagnosticEngine,
    severity: Severity,
    span: Span,
    message: String,
) {
    match severity {
        Severity::Error => diagnostics.emit(Diagnostic::error(span, message)),
        Severity::Warning => diagnostics.emit(Diagnostic::warning(span, message)),
        Severity::Note => diagnostics.emit(Diagnostic::note(span, message)),
    }
}
