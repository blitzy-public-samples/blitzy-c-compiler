//! Macro expansion engine with paint-marker recursion protection and
//! 512-depth recursion limit enforcement.
//!
//! Implements the core macro expansion engine for both function-like and
//! object-like macros as specified by C11 §6.10.3.  Handles:
//!
//! - **Object-like macros:** `#define FOO 42` — simple token replacement.
//! - **Function-like macros:** `#define MAX(a, b) ((a) > (b) ? (a) : (b))` —
//!   parameterised replacement with argument substitution.
//! - **Variadic macros:** `__VA_ARGS__` and `__VA_OPT__` support for macros
//!   declared with `...`.
//! - **`#` stringification:** `#param` → string literal (C11 §6.10.3.2).
//! - **`##` token pasting:** `a ## b` → concatenated token (C11 §6.10.3.3).
//! - **Paint-marker recursion protection:** Prevents infinite expansion of
//!   self-referential macros like `#define A A` (C11 §6.10.3.4).
//! - **512-depth recursion limit:** Hard cap on expansion nesting to prevent
//!   stack overflow from deeply nested Linux kernel macro chains.
//!
//! # C11 Expansion Algorithm
//!
//! 1. Identify macro invocation (object-like or function-like with arguments).
//! 2. For function-like: perform argument substitution in the replacement list:
//!    - Arguments adjacent to `#` or `##` use the **unexpanded** argument.
//!    - Arguments in normal context use the **pre-expanded** argument.
//! 3. Process `##` concatenation on the substituted replacement list.
//! 4. Remove placemarker tokens.
//! 5. **Rescan** the result concatenated with remaining source tokens for
//!    further macro expansion (C11 §6.10.3.4), with paint markers preventing
//!    re-expansion of the macro being replaced.
//!
//! # Zero-Dependency
//!
//! This module depends only on `std` and `crate::` references — no external
//! crates.  Integrates with sibling modules `paint_marker` and `token_paster`
//! for recursion protection and `#`/`##` processing respectively.

use super::paint_marker::{PaintMarker, PaintState};
use super::token_paster::{paste_tokens, stringify_tokens, PasteError};
use super::{MacroDef, MacroKind, PPToken, PPTokenKind};
use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sentinel index used to represent `__VA_ARGS__` in parameter lookups.
/// Distinguished from real parameter indices by being `usize::MAX`.
const VA_ARGS_INDEX: usize = usize::MAX;

// ---------------------------------------------------------------------------
// MacroExpander — the public expansion engine
// ---------------------------------------------------------------------------

/// Macro expansion engine with paint-marker recursion protection
/// and 512-depth limit enforcement.
///
/// Holds references to the macro definition table and diagnostic engine,
/// along with internal state for tracking expansion depth and paint markers.
///
/// # Lifetime
///
/// The `'a` lifetime ties the expander to the macro definition map and
/// diagnostic engine provided by the preprocessor.
///
/// # Thread Safety
///
/// `MacroExpander` is **not** thread-safe — it is designed to be used within
/// a single compilation worker thread (the 64 MiB stack thread spawned by
/// `main.rs`).
pub struct MacroExpander<'a> {
    /// Reference to macro definitions from the preprocessor's `#define` table.
    /// Keyed by macro name, values are complete [`MacroDef`] entries.
    macro_defs: &'a FxHashMap<String, MacroDef>,

    /// Paint marker tracker — records which macro names are currently "in
    /// flight" (being expanded) to suppress re-expansion per C11 §6.10.3.4.
    paint_marker: PaintMarker,

    /// Current expansion nesting depth.  Incremented before each recursive
    /// expansion call and decremented after.
    depth: usize,

    /// Maximum allowed expansion depth.  Defaults to 512 per the AAP
    /// requirement.  If `depth >= max_depth`, further expansion is refused
    /// and a diagnostic error is emitted.
    max_depth: usize,

    /// Mutable reference to the diagnostic engine for error/warning emission.
    diagnostics: &'a mut DiagnosticEngine,

    /// Mutable reference to the `__COUNTER__` value.  Each expansion of
    /// `__COUNTER__` returns the current value and increments it.
    counter: &'a mut u64,

    /// Optional reference to the source map for resolving `__LINE__` and
    /// `__FILE__` magic macros during nested macro expansion.  When `None`,
    /// these macros fall back to their predefined static values.
    source_map: Option<&'a SourceMap>,
}

// ===========================================================================
// Public API
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Create a new macro expander.
    ///
    /// # Parameters
    ///
    /// - `macro_defs`: Shared reference to the preprocessor's macro definition
    ///   table (an [`FxHashMap`] for fast Fibonacci-hashed lookups).
    /// - `diagnostics`: Mutable reference to the diagnostic engine for
    ///   emitting errors and warnings during expansion.
    /// - `max_depth`: Maximum expansion nesting depth.  The AAP mandates 512.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut diag = DiagnosticEngine::new();
    /// let defs: FxHashMap<String, MacroDef> = FxHashMap::default();
    /// let mut counter_val: u64 = 0;
    /// let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);
    /// ```
    pub fn new(
        macro_defs: &'a FxHashMap<String, MacroDef>,
        diagnostics: &'a mut DiagnosticEngine,
        max_depth: usize,
        counter: &'a mut u64,
    ) -> Self {
        MacroExpander {
            macro_defs,
            paint_marker: PaintMarker::new(),
            depth: 0,
            max_depth,
            diagnostics,
            counter,
            source_map: None,
        }
    }

    /// Create a new macro expander with source map access for `__LINE__`
    /// and `__FILE__` magic macro resolution during nested expansion.
    pub fn new_with_source_map(
        macro_defs: &'a FxHashMap<String, MacroDef>,
        diagnostics: &'a mut DiagnosticEngine,
        max_depth: usize,
        counter: &'a mut u64,
        source_map: &'a SourceMap,
    ) -> Self {
        MacroExpander {
            macro_defs,
            paint_marker: PaintMarker::new(),
            depth: 0,
            max_depth,
            diagnostics,
            counter,
            source_map: Some(source_map),
        }
    }

    /// Expand all macros in a token sequence.
    ///
    /// This is the main entry point called by the preprocessor after directive
    /// processing.  Iterates through `tokens` left-to-right, expanding macros
    /// as they are encountered.  Expansion results are rescanned together with
    /// remaining source tokens per C11 §6.10.3.4.
    ///
    /// # Algorithm
    ///
    /// For each identifier token:
    /// 1. Check paint state — if painted, skip (treat as ordinary identifier).
    /// 2. Look up in `macro_defs` — if not a macro, pass through.
    /// 3. Check depth limit — if exceeded, emit error and pass through.
    /// 4. Object-like: replace with expansion, prepend to remaining tokens,
    ///    recursively rescan the combined list.
    /// 5. Function-like: look ahead for `(`, collect arguments, substitute,
    ///    process `##`, prepend to remaining tokens, recursively rescan.
    ///
    /// Non-identifier tokens pass through unchanged.
    ///
    /// # Returns
    ///
    /// A fully macro-expanded token sequence with all identifiable macros
    /// replaced according to their definitions.
    pub fn expand_tokens(&mut self, tokens: &[PPToken]) -> Vec<PPToken> {
        let mut result: Vec<PPToken> = Vec::with_capacity(tokens.len());
        let mut i: usize = 0;

        while i < tokens.len() {
            let token = tokens[i].clone();

            // ----- Non-identifier tokens pass through unchanged -----
            if token.kind != PPTokenKind::Identifier {
                result.push(token);
                i += 1;
                continue;
            }

            // ----- Check paint state on the token itself -----
            if token.painted {
                result.push(token);
                i += 1;
                continue;
            }

            // ----- Check paint marker for active expansions (C11 §6.10.3.4) -----
            match self.paint_marker.check_token_paint(&token.text) {
                PaintState::Painted => {
                    let mut painted_tok = token;
                    painted_tok.painted = true;
                    result.push(painted_tok);
                    i += 1;
                    continue;
                }
                PaintState::Unpainted => {
                    // Eligible for expansion — continue below.
                }
            }

            // ----- Look up macro definition -----
            if !self.macro_defs.contains_key(&token.text) {
                result.push(token);
                i += 1;
                continue;
            }
            let macro_def = self.macro_defs.get(&token.text).unwrap().clone();

            // ----- Depth limit enforcement -----
            if self.depth >= self.max_depth {
                self.diagnostics.emit(Diagnostic::error(
                    token.span,
                    format!(
                        "macro expansion depth exceeds maximum of {}",
                        self.max_depth
                    ),
                ));
                result.push(token);
                i += 1;
                continue;
            }

            // ----- Dispatch by macro kind -----
            match &macro_def.kind {
                MacroKind::ObjectLike => {
                    // Prepare replacement tokens (with ## processing).
                    let replacement = self.prepare_object_replacement(&macro_def, token.span);

                    // C11 §6.10.3.4: rescan replacement tokens with paint
                    // active (suppress self-referential re-expansion), then
                    // unpaint before processing remaining source tokens.
                    self.paint_marker.paint(&macro_def.name);
                    self.depth += 1;

                    let rescanned_sub = self.expand_tokens(&replacement);

                    self.depth -= 1;
                    self.paint_marker.unpaint(&macro_def.name);

                    let remaining = &tokens[i + 1..];

                    // C11 §6.10.3.4 cross-boundary check: if the last
                    // token of the rescanned replacement is a function-like
                    // macro name (unexpanded because its argument '(' is
                    // in the remaining source tokens), combine them so the
                    // invocation spans the replacement/source boundary.
                    if !remaining.is_empty() {
                        if let Some(last) = rescanned_sub.last() {
                            if last.kind == PPTokenKind::Identifier && !last.painted {
                                let cross = self
                                    .macro_defs
                                    .get(&last.text)
                                    .map(|md| matches!(md.kind, MacroKind::FunctionLike { .. }))
                                    .unwrap_or(false);
                                if cross && find_lparen(remaining, 0).is_some() {
                                    let mut tail = rescanned_sub;
                                    let func_tok = tail.pop().unwrap();
                                    result.extend(tail);
                                    let mut combined = vec![func_tok];
                                    combined.extend_from_slice(remaining);
                                    let expanded = self.expand_tokens(&combined);
                                    result.extend(expanded);
                                    return result;
                                }
                            }
                        }
                    }

                    // No cross-boundary invocation — process normally.
                    result.extend(rescanned_sub);
                    let rest = self.expand_tokens(remaining);
                    result.extend(rest);
                    return result;
                }

                MacroKind::FunctionLike { .. } => {
                    // Look ahead for '(' — skip whitespace/newlines.
                    match find_lparen(tokens, i + 1) {
                        Some(lparen_idx) => {
                            // Collect arguments starting after '('.
                            match self.collect_arguments(tokens, lparen_idx + 1) {
                                Ok((args, end_idx)) => {
                                    let substituted = self.perform_function_substitution(
                                        &macro_def, args, token.span,
                                    );

                                    // C11 §6.10.3.4: Rescan the replacement
                                    // tokens with paint active to suppress
                                    // self-referential re-expansion.  Then
                                    // UNPAINT before processing remaining
                                    // source tokens so that subsequent
                                    // invocations of the same macro (e.g.
                                    // `C(1) || C(2)`) are expanded normally.
                                    self.paint_marker.paint(&macro_def.name);
                                    self.depth += 1;

                                    let rescanned_sub = self.expand_tokens(&substituted);

                                    self.depth -= 1;
                                    self.paint_marker.unpaint(&macro_def.name);

                                    let remaining = &tokens[end_idx..];

                                    // C11 §6.10.3.4 cross-boundary check
                                    // (same as object-like case above).
                                    if !remaining.is_empty() {
                                        if let Some(last) = rescanned_sub.last() {
                                            if last.kind == PPTokenKind::Identifier && !last.painted
                                            {
                                                let cross = self
                                                    .macro_defs
                                                    .get(&last.text)
                                                    .map(|md| {
                                                        matches!(
                                                            md.kind,
                                                            MacroKind::FunctionLike { .. }
                                                        )
                                                    })
                                                    .unwrap_or(false);
                                                if cross && find_lparen(remaining, 0).is_some() {
                                                    let mut tail = rescanned_sub;
                                                    let func_tok = tail.pop().unwrap();
                                                    result.extend(tail);
                                                    let mut combined = vec![func_tok];
                                                    combined.extend_from_slice(remaining);
                                                    let expanded = self.expand_tokens(&combined);
                                                    result.extend(expanded);
                                                    return result;
                                                }
                                            }
                                        }
                                    }

                                    result.extend(rescanned_sub);

                                    // Continue expanding remaining source
                                    // tokens from after the macro invocation.
                                    let rest = self.expand_tokens(remaining);
                                    result.extend(rest);
                                    return result;
                                }
                                Err(()) => {
                                    // Argument collection failed — pass through.
                                    result.push(token);
                                    i += 1;
                                }
                            }
                        }
                        None => {
                            // No '(' follows — not a function-like invocation.
                            // Pass the identifier through unchanged.
                            result.push(token);
                            i += 1;
                        }
                    }
                }
            }
        }

        result
    }
}

// ===========================================================================
// Private implementation — object-like expansion
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Prepare the replacement tokens for an object-like macro.
    ///
    /// Clones the replacement list, marks tokens as `from_macro = true`,
    /// updates spans to the invocation site, and processes any `##`
    /// concatenation operators present in the replacement list.
    fn prepare_object_replacement(
        &mut self,
        macro_def: &MacroDef,
        invocation_span: Span,
    ) -> Vec<PPToken> {
        // Handle __COUNTER__ — auto-incrementing integer per expansion.
        if macro_def.is_predefined && macro_def.name == "__COUNTER__" {
            let val = *self.counter;
            *self.counter += 1;
            return vec![PPToken::from_expansion(
                PPTokenKind::Number,
                val.to_string(),
                invocation_span,
            )];
        }

        // Handle __LINE__ — expands to current source line number.
        // This must be handled here (not just in the top-level preprocessor)
        // because __LINE__ can appear inside nested macro expansions where
        // the MacroExpander drives the expansion, not expand_object_macro().
        if macro_def.is_predefined && macro_def.name == "__LINE__" {
            if let Some(source_map) = self.source_map {
                if let Some(file) = source_map.get_file(invocation_span.file_id) {
                    let (line, _) = file.lookup_line_col(invocation_span.start);
                    return vec![PPToken::from_expansion(
                        PPTokenKind::Number,
                        line.to_string(),
                        invocation_span,
                    )];
                }
            }
        }

        // Handle __FILE__ — expands to current source filename as string.
        if macro_def.is_predefined && macro_def.name == "__FILE__" {
            if let Some(source_map) = self.source_map {
                let filename = source_map
                    .get_filename(invocation_span.file_id)
                    .unwrap_or("<unknown>");
                return vec![PPToken::from_expansion(
                    PPTokenKind::StringLiteral,
                    format!("\"{}\"", filename),
                    invocation_span,
                )];
            }
        }

        if macro_def.replacement.is_empty() {
            return Vec::new();
        }

        let marked: Vec<PPToken> = macro_def
            .replacement
            .iter()
            .map(|t| {
                let mut tok = t.clone();
                tok.from_macro = true;
                tok.span = invocation_span;
                tok
            })
            .collect();

        // C11 §6.10.3.3: ## concatenation applies to both object-like and
        // function-like macro replacement lists.
        self.process_paste(&marked, invocation_span)
    }
}

// ===========================================================================
// Private implementation — function-like expansion
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Perform complete function-like macro substitution.
    ///
    /// 1. Validates argument count against parameter count.
    /// 2. Pre-expands arguments for normal-context substitution.
    /// 3. Substitutes parameters in the replacement list (Pass 1).
    /// 4. Processes `##` concatenation on the result (Pass 2).
    /// 5. Removes placemarker tokens.
    ///
    /// The result is ready for the rescan phase (handled by the caller).
    fn perform_function_substitution(
        &mut self,
        macro_def: &MacroDef,
        args: Vec<Vec<PPToken>>,
        invocation_span: Span,
    ) -> Vec<PPToken> {
        let (params, variadic) = match &macro_def.kind {
            MacroKind::FunctionLike { params, variadic } => (params, *variadic),
            MacroKind::ObjectLike => return Vec::new(),
        };

        // ---- Validate argument count ----
        if !self.validate_arg_count(
            &macro_def.name,
            macro_def.is_predefined,
            params,
            variadic,
            &args,
            invocation_span,
        ) {
            return Vec::new();
        }

        if macro_def.replacement.is_empty() {
            return Vec::new();
        }

        // Handle 0-param macro called as FOO() → treat as 0 args.
        let effective_args = if params.is_empty() && args.len() == 1 && is_arg_empty(&args[0]) {
            Vec::new()
        } else {
            args
        };

        // ---- Pre-expand arguments for normal-context substitution ----
        // C11 §6.10.3.1: arguments NOT adjacent to # or ## are fully
        // macro-expanded before substitution.
        let expanded_args: Vec<Vec<PPToken>> = effective_args
            .iter()
            .map(|a| self.expand_tokens(a))
            .collect();

        // ---- Pass 1: parameter substitution ----
        let substituted = self.substitute_params(
            &macro_def.replacement,
            params,
            &effective_args,
            &expanded_args,
            variadic,
            invocation_span,
        );

        // ---- Pass 2: ## concatenation ----
        let pasted = self.process_paste(&substituted, invocation_span);

        // ---- Remove placemarker tokens ----
        pasted
            .into_iter()
            .filter(|t| t.kind != PPTokenKind::PlacemarkerToken)
            .collect()
    }

    /// Validate that the provided argument count matches the macro's parameter
    /// count, emitting a diagnostic on mismatch.
    ///
    /// Returns `true` if the count is valid, `false` otherwise.
    fn validate_arg_count(
        &mut self,
        macro_name: &str,
        is_predefined: bool,
        params: &[String],
        variadic: bool,
        args: &[Vec<PPToken>],
        span: Span,
    ) -> bool {
        let expected = params.len();
        let provided = args.len();
        let kind_str = if is_predefined {
            "predefined macro"
        } else {
            "macro"
        };

        // Handle 0-param macro called as FOO() — single empty arg is OK.
        let effective_provided = if expected == 0 && provided == 1 && is_arg_empty(&args[0]) {
            0
        } else {
            provided
        };

        if variadic {
            // Variadic: need at least `expected` args (named params); excess
            // go to __VA_ARGS__.  For `#define FOO(...)`, expected == 0.
            if effective_provided < expected {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    format!(
                        "{} '{}' requires at least {} argument(s), but {} provided",
                        kind_str, macro_name, expected, effective_provided,
                    ),
                ));
                return false;
            }
        } else {
            // Non-variadic: exact match required.
            if effective_provided != expected {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    format!(
                        "{} '{}' requires {} argument(s), but {} provided",
                        kind_str, macro_name, expected, effective_provided,
                    ),
                ));
                return false;
            }
        }
        true
    }
}

// ===========================================================================
// Private implementation — parameter substitution (Pass 1)
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Replace parameters in a function-like macro's replacement list with
    /// the corresponding argument tokens.
    ///
    /// Handles `#` (stringification), `##` adjacency (uses unexpanded args),
    /// `__VA_ARGS__`, `__VA_OPT__`, and normal parameter substitution (uses
    /// pre-expanded args).
    ///
    /// `##` tokens are preserved in the output for Pass 2 processing.
    fn substitute_params(
        &mut self,
        replacement: &[PPToken],
        params: &[String],
        unexpanded_args: &[Vec<PPToken>],
        expanded_args: &[Vec<PPToken>],
        variadic: bool,
        invocation_span: Span,
    ) -> Vec<PPToken> {
        let mut result: Vec<PPToken> = Vec::with_capacity(replacement.len() * 2);
        let mut i: usize = 0;

        while i < replacement.len() {
            let tok = &replacement[i];

            // === Case 1: `#` stringification operator ===
            if tok.kind == PPTokenKind::Punctuator && tok.text == "#" {
                // Find the next non-whitespace token.
                let mut j = i + 1;
                while j < replacement.len() && replacement[j].is_whitespace() {
                    j += 1;
                }

                if j < replacement.len() && replacement[j].kind == PPTokenKind::Identifier {
                    if let Some(param_idx) =
                        self.find_param_or_va_index(&replacement[j].text, params, variadic)
                    {
                        // C11 §6.10.3.2: stringify the UNEXPANDED argument.
                        let arg_tokens =
                            self.get_argument_tokens(param_idx, unexpanded_args, params, variadic);
                        let mut stringified = stringify_tokens(&arg_tokens);
                        stringified.from_macro = true;
                        stringified.span = invocation_span;
                        result.push(stringified);
                        i = j + 1;
                        continue;
                    }
                }

                // Not followed by a parameter — treat `#` as a regular token.
                let mut t = tok.clone();
                t.from_macro = true;
                result.push(t);
                i += 1;
                continue;
            }

            // === Case 2: `##` concatenation — preserve for Pass 2 ===
            if is_hashhash_token(tok) {
                let mut t = tok.clone();
                t.from_macro = true;
                result.push(t);
                i += 1;
                continue;
            }

            // === Case 3: __VA_OPT__(content) ===
            if tok.kind == PPTokenKind::Identifier && tok.text == "__VA_OPT__" && variadic {
                let (va_opt_content, end_i) = collect_va_opt_content(replacement, i + 1);

                let va_args = self.get_va_args_tokens(unexpanded_args, params);
                let has_va_args = va_args
                    .iter()
                    .any(|t| !t.is_whitespace() && t.kind != PPTokenKind::PlacemarkerToken);

                if has_va_args {
                    // Recursively substitute params within __VA_OPT__ content.
                    let inner = self.substitute_params(
                        &va_opt_content,
                        params,
                        unexpanded_args,
                        expanded_args,
                        variadic,
                        invocation_span,
                    );
                    result.extend(inner);
                }
                // If __VA_ARGS__ is empty, __VA_OPT__ expands to nothing.
                i = end_i;
                continue;
            }

            // === Case 4: identifier that may be a parameter ===
            if tok.kind == PPTokenKind::Identifier {
                // Check __VA_ARGS__ first.
                if tok.text == "__VA_ARGS__" && variadic {
                    let adjacent = followed_by_hashhash(replacement, i)
                        || preceded_by_hashhash(replacement, i);
                    if adjacent {
                        // Adjacent to ## → use unexpanded variadic args.
                        let va = self.get_va_args_tokens(unexpanded_args, params);
                        let is_va_empty = va.is_empty()
                            || va.iter().all(|t| {
                                t.is_whitespace() || t.kind == PPTokenKind::PlacemarkerToken
                            });
                        if is_va_empty {
                            result.push(PPToken::placemarker(invocation_span));
                        } else {
                            for t in &va {
                                let mut c = t.clone();
                                c.from_macro = true;
                                result.push(c);
                            }
                        }
                    } else {
                        // Normal context → use expanded variadic args.
                        let va = self.get_va_args_tokens(expanded_args, params);
                        for t in &va {
                            let mut c = t.clone();
                            c.from_macro = true;
                            result.push(c);
                        }
                    }
                    i += 1;
                    continue;
                }

                // Check named parameters.
                if let Some(param_idx) = find_named_param_index(&tok.text, params) {
                    let adjacent = followed_by_hashhash(replacement, i)
                        || preceded_by_hashhash(replacement, i);

                    if adjacent {
                        // C11 §6.10.3.3: parameter adjacent to ## uses the
                        // UNEXPANDED argument.
                        let arg = if param_idx < unexpanded_args.len() {
                            &unexpanded_args[param_idx]
                        } else {
                            &[][..]
                        };
                        // An argument is considered empty (placemarker) if it
                        // has no tokens or contains only whitespace tokens.
                        let is_arg_empty = arg.is_empty()
                            || arg.iter().all(|t| {
                                t.is_whitespace() || t.kind == PPTokenKind::PlacemarkerToken
                            });
                        if is_arg_empty {
                            result.push(PPToken::placemarker(invocation_span));
                        } else {
                            for t in arg {
                                let mut c = t.clone();
                                c.from_macro = true;
                                result.push(c);
                            }
                        }
                    } else {
                        // C11 §6.10.3.1: normal context uses the
                        // PRE-EXPANDED argument.
                        let arg = if param_idx < expanded_args.len() {
                            &expanded_args[param_idx]
                        } else {
                            &[][..]
                        };
                        for t in arg {
                            let mut c = t.clone();
                            c.from_macro = true;
                            result.push(c);
                        }
                    }
                    i += 1;
                    continue;
                }
            }

            // === Default: pass through unchanged ===
            // Set span to invocation_span so that magic macros like
            // __LINE__ and __FILE__ that appear in the replacement list
            // (but are not parameters) resolve to the invocation site,
            // not the macro definition site.
            let mut t = tok.clone();
            t.from_macro = true;
            t.span = invocation_span;
            result.push(t);
            i += 1;
        }

        result
    }
}

// ===========================================================================
// Private implementation — ## concatenation processing (Pass 2)
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Process all `##` concatenation operators in a token list.
    ///
    /// Walks through the tokens, and when a `##` is found, pops the last
    /// non-whitespace token from the result, skips whitespace after `##`,
    /// takes the next token, and pastes them using [`paste_tokens`].
    ///
    /// Invalid paste results (e.g., `@` `##` `!` → `@!`) are diagnosed
    /// as warnings and the original tokens are preserved.
    fn process_paste(&mut self, tokens: &[PPToken], invocation_span: Span) -> Vec<PPToken> {
        // Quick check: if no ## operator is present, return as-is.
        if !tokens.iter().any(is_hashhash_token) {
            return tokens.to_vec();
        }

        let mut result: Vec<PPToken> = Vec::with_capacity(tokens.len());
        let mut i: usize = 0;

        while i < tokens.len() {
            let tok = &tokens[i];

            if !is_hashhash_token(tok) {
                result.push(tok.clone());
                i += 1;
                continue;
            }

            // ## found — pop left operand (skip trailing whitespace in result).
            while result.last().map_or(false, |t| t.is_whitespace()) {
                result.pop();
            }

            let left = match result.pop() {
                Some(l) => l,
                None => {
                    self.diagnostics.emit(Diagnostic::error(
                        tok.span,
                        "'##' cannot appear at the start of a macro replacement list",
                    ));
                    i += 1;
                    continue;
                }
            };

            // Skip ## and any whitespace to find the right operand.
            i += 1;
            while i < tokens.len() && tokens[i].is_whitespace() {
                i += 1;
            }

            if i >= tokens.len() {
                self.diagnostics.emit(Diagnostic::error(
                    tok.span,
                    "'##' cannot appear at the end of a macro replacement list",
                ));
                result.push(left);
                continue;
            }

            let right = &tokens[i];

            // GCC comma elision extension: `, ## __VA_ARGS__` where
            // __VA_ARGS__ is empty (placemarker) removes the preceding comma.
            if left.kind == PPTokenKind::Punctuator
                && left.text == ","
                && right.kind == PPTokenKind::PlacemarkerToken
            {
                // Drop both the comma and the placemarker.
                i += 1;
                continue;
            }

            match paste_tokens(&left, right) {
                Ok(pasted) => {
                    result.push(pasted);
                }
                Err(PasteError::InvalidToken(ref bad_text)) => {
                    self.diagnostics.emit(Diagnostic::warning(
                        invocation_span,
                        format!(
                            "pasting \"{}\" and \"{}\" does not give a valid \
                             preprocessing token: \"{}\"",
                            left.text, right.text, bad_text,
                        ),
                    ));
                    // Preserve original tokens on invalid paste.
                    result.push(left);
                    result.push(right.clone());
                }
            }

            i += 1;
        }

        result
    }
}

// ===========================================================================
// Private implementation — argument collection
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Collect the arguments of a function-like macro invocation.
    ///
    /// `start` is the index of the first token **after** the opening `(`.
    /// Arguments are separated by `,` at the top-level parenthesis depth.
    /// Commas inside nested `()` are part of the argument, not separators.
    ///
    /// # Returns
    ///
    /// - `Ok((args, end_idx))` where `args` is the list of argument token
    ///   sequences and `end_idx` is the index of the first token **after**
    ///   the closing `)`.
    /// - `Err(())` if the argument list is unterminated (missing `)`).
    fn collect_arguments(
        &mut self,
        tokens: &[PPToken],
        start: usize,
    ) -> Result<(Vec<Vec<PPToken>>, usize), ()> {
        let mut args: Vec<Vec<PPToken>> = vec![Vec::new()];
        let mut paren_depth: usize = 1;
        let mut i = start;

        while i < tokens.len() {
            let tok = &tokens[i];

            match &tok.kind {
                PPTokenKind::Punctuator if tok.text == "(" => {
                    paren_depth += 1;
                    args.last_mut().unwrap().push(tok.clone());
                }
                PPTokenKind::Punctuator if tok.text == ")" => {
                    paren_depth -= 1;
                    if paren_depth == 0 {
                        // Closing paren found — return arguments.
                        return Ok((args, i + 1));
                    }
                    args.last_mut().unwrap().push(tok.clone());
                }
                PPTokenKind::Punctuator if tok.text == "," && paren_depth == 1 => {
                    // Top-level comma: start a new argument.
                    args.push(Vec::new());
                }
                PPTokenKind::EndOfFile => {
                    self.diagnostics.emit(Diagnostic::error(
                        tok.span,
                        "unterminated argument list for macro invocation",
                    ));
                    return Err(());
                }
                _ => {
                    args.last_mut().unwrap().push(tok.clone());
                }
            }

            i += 1;
        }

        // Ran out of tokens without finding ')'.
        let err_span = if start > 0 {
            tokens[start - 1].span
        } else {
            Span::dummy()
        };
        self.diagnostics.emit(Diagnostic::error(
            err_span,
            "unterminated argument list for macro invocation",
        ));
        Err(())
    }
}

// ===========================================================================
// Private implementation — parameter index helpers
// ===========================================================================

impl<'a> MacroExpander<'a> {
    /// Find the index of a token name in the parameter list, or identify it
    /// as `__VA_ARGS__` for a variadic macro.
    ///
    /// Returns `Some(index)` for named parameters, `Some(VA_ARGS_INDEX)` for
    /// `__VA_ARGS__` in a variadic macro, or `None` if not a parameter.
    fn find_param_or_va_index(
        &self,
        name: &str,
        params: &[String],
        variadic: bool,
    ) -> Option<usize> {
        // Check named parameters first.
        if let Some(idx) = find_named_param_index(name, params) {
            return Some(idx);
        }
        // Check __VA_ARGS__ for variadic macros.
        if variadic && name == "__VA_ARGS__" {
            return Some(VA_ARGS_INDEX);
        }
        None
    }

    /// Retrieve the argument tokens for a given parameter index.
    ///
    /// For named parameters (`idx < params.len()`), returns the corresponding
    /// argument.  For `VA_ARGS_INDEX`, returns the comma-joined variadic
    /// arguments.
    fn get_argument_tokens(
        &self,
        param_idx: usize,
        args: &[Vec<PPToken>],
        params: &[String],
        variadic: bool,
    ) -> Vec<PPToken> {
        if param_idx == VA_ARGS_INDEX && variadic {
            return self.get_va_args_tokens(args, params);
        }
        if param_idx < args.len() {
            args[param_idx].clone()
        } else {
            Vec::new()
        }
    }

    /// Build the `__VA_ARGS__` token sequence from the variadic arguments.
    ///
    /// Variadic arguments start at index `params.len()` in the argument list.
    /// Multiple variadic arguments are joined with `,` ` ` separators to
    /// reproduce the original argument list structure.
    fn get_va_args_tokens(&self, args: &[Vec<PPToken>], params: &[String]) -> Vec<PPToken> {
        let va_start = params.len();
        if va_start >= args.len() {
            return Vec::new();
        }

        let mut result: Vec<PPToken> = Vec::new();
        for (idx, arg) in args[va_start..].iter().enumerate() {
            if idx > 0 {
                // Insert comma separator between variadic arguments.
                result.push(PPToken::from_expansion(
                    PPTokenKind::Punctuator,
                    ",",
                    Span::dummy(),
                ));
                result.push(PPToken::from_expansion(
                    PPTokenKind::Whitespace,
                    " ",
                    Span::dummy(),
                ));
            }
            result.extend(arg.iter().cloned());
        }
        result
    }
}

// ===========================================================================
// Free helper functions
// ===========================================================================

/// Find the index of the opening `(` in `tokens` starting from `start`,
/// skipping whitespace and newlines.
///
/// Returns `Some(index)` if `(` is found, `None` if a non-whitespace
/// non-`(` token is encountered first, or if the token slice is exhausted.
fn find_lparen(tokens: &[PPToken], start: usize) -> Option<usize> {
    let mut j = start;
    while j < tokens.len() {
        match &tokens[j].kind {
            PPTokenKind::Whitespace | PPTokenKind::Newline => {
                j += 1;
            }
            PPTokenKind::Punctuator if tokens[j].text == "(" => {
                return Some(j);
            }
            _ => {
                return None;
            }
        }
    }
    None
}

/// Find the index of a named parameter in the parameter list.
///
/// Returns `Some(index)` if found, `None` otherwise.  This does NOT check
/// for `__VA_ARGS__` — use [`MacroExpander::find_param_or_va_index`] for
/// that.
fn find_named_param_index(name: &str, params: &[String]) -> Option<usize> {
    params.iter().position(|p| p == name)
}

/// Check whether a token is the `##` concatenation operator.
#[inline]
fn is_hashhash_token(tok: &PPToken) -> bool {
    tok.kind == PPTokenKind::Punctuator && tok.text == "##"
}

/// Check whether position `pos` in `tokens` is followed by a `##` token
/// (skipping intervening whitespace and newlines).
fn followed_by_hashhash(tokens: &[PPToken], pos: usize) -> bool {
    let mut j = pos + 1;
    while j < tokens.len() {
        if tokens[j].is_whitespace() {
            j += 1;
            continue;
        }
        return is_hashhash_token(&tokens[j]);
    }
    false
}

/// Check whether position `pos` in `tokens` is preceded by a `##` token
/// (skipping intervening whitespace and newlines).
fn preceded_by_hashhash(tokens: &[PPToken], pos: usize) -> bool {
    if pos == 0 {
        return false;
    }
    let mut j = pos - 1;
    loop {
        if tokens[j].is_whitespace() {
            if j == 0 {
                return false;
            }
            j -= 1;
            continue;
        }
        return is_hashhash_token(&tokens[j]);
    }
}

/// Check whether an argument token list is effectively empty (contains no
/// non-whitespace tokens).
fn is_arg_empty(arg: &[PPToken]) -> bool {
    arg.iter()
        .all(|t| t.is_whitespace() || t.kind == PPTokenKind::PlacemarkerToken)
}

/// Collect the content tokens inside a `__VA_OPT__(...)` construct.
///
/// `start` is the index immediately after the `__VA_OPT__` identifier.
/// Finds the `(`, collects tokens tracking parenthesis depth, and returns
/// the inner content and the index of the first token after the closing `)`.
fn collect_va_opt_content(tokens: &[PPToken], start: usize) -> (Vec<PPToken>, usize) {
    let mut i = start;

    // Skip whitespace to find '('.
    while i < tokens.len() && tokens[i].is_whitespace() {
        i += 1;
    }

    if i >= tokens.len() || tokens[i].kind != PPTokenKind::Punctuator || tokens[i].text != "(" {
        // No opening paren — return empty content, consume nothing extra.
        return (Vec::new(), start);
    }

    // Skip the '('.
    i += 1;
    let mut content: Vec<PPToken> = Vec::new();
    let mut depth: usize = 1;

    while i < tokens.len() {
        let tok = &tokens[i];
        if tok.kind == PPTokenKind::Punctuator && tok.text == "(" {
            depth += 1;
            content.push(tok.clone());
        } else if tok.kind == PPTokenKind::Punctuator && tok.text == ")" {
            depth -= 1;
            if depth == 0 {
                // Found matching ')'.
                return (content, i + 1);
            }
            content.push(tok.clone());
        } else {
            content.push(tok.clone());
        }
        i += 1;
    }

    // Unterminated __VA_OPT__ — return what we have.
    (content, i)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::fx_hash::FxHashMap;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Create a simple identifier token.
    fn ident(name: &str) -> PPToken {
        PPToken::new(PPTokenKind::Identifier, name, Span::dummy())
    }

    /// Create a simple number token.
    fn num(val: &str) -> PPToken {
        PPToken::new(PPTokenKind::Number, val, Span::dummy())
    }

    /// Create a punctuator token.
    fn punct(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Punctuator, text, Span::dummy())
    }

    /// Create a whitespace token.
    fn ws() -> PPToken {
        PPToken::new(PPTokenKind::Whitespace, " ", Span::dummy())
    }

    /// Build a simple object-like macro definition.
    fn object_macro(name: &str, replacement: Vec<PPToken>) -> MacroDef {
        MacroDef {
            name: name.to_string(),
            kind: MacroKind::ObjectLike,
            replacement,
            is_predefined: false,
            definition_span: Span::dummy(),
        }
    }

    /// Build a simple function-like macro definition.
    fn function_macro(
        name: &str,
        params: Vec<&str>,
        variadic: bool,
        replacement: Vec<PPToken>,
    ) -> MacroDef {
        MacroDef {
            name: name.to_string(),
            kind: MacroKind::FunctionLike {
                params: params.into_iter().map(String::from).collect(),
                variadic,
            },
            replacement,
            is_predefined: false,
            definition_span: Span::dummy(),
        }
    }

    // -----------------------------------------------------------------------
    // Object-like macro tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_object_like_expansion() {
        // #define FOO 42
        let mut defs = FxHashMap::default();
        defs.insert("FOO".to_string(), object_macro("FOO", vec![num("42")]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("FOO")];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "42");
        assert!(result[0].from_macro);
    }

    #[test]
    fn test_self_referential_macro_terminates() {
        // #define A A
        // Must terminate quickly — paint marker prevents infinite recursion.
        let mut defs = FxHashMap::default();
        defs.insert("A".to_string(), object_macro("A", vec![ident("A")]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("A")];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "A");
        assert!(result[0].painted); // Should be painted.
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_mutual_recursion_terminates() {
        // #define A B
        // #define B A
        // A → B → A(painted) → stops at token A.
        let mut defs = FxHashMap::default();
        defs.insert("A".to_string(), object_macro("A", vec![ident("B")]));
        defs.insert("B".to_string(), object_macro("B", vec![ident("A")]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("A")];
        let result = expander.expand_tokens(&tokens);

        // A → B → expansion of B → A (painted) → stops
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "A");
        assert!(result[0].painted);
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_chain_expansion() {
        // #define X Y
        // #define Y 42
        // X → Y → 42
        let mut defs = FxHashMap::default();
        defs.insert("X".to_string(), object_macro("X", vec![ident("Y")]));
        defs.insert("Y".to_string(), object_macro("Y", vec![num("42")]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("X")];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "42");
    }

    #[test]
    fn test_empty_replacement() {
        // #define EMPTY
        let mut defs = FxHashMap::default();
        defs.insert("EMPTY".to_string(), object_macro("EMPTY", vec![]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("EMPTY"), ws(), ident("x")];
        let result = expander.expand_tokens(&tokens);

        // EMPTY expands to nothing, then remaining tokens are: ws, x
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, " ");
        assert_eq!(result[1].text, "x");
    }

    // -----------------------------------------------------------------------
    // Depth limit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_depth_limit_enforced() {
        // Test with max_depth = 2
        // #define A B
        // #define B C
        // #define C D
        let mut defs = FxHashMap::default();
        defs.insert("A".to_string(), object_macro("A", vec![ident("B")]));
        defs.insert("B".to_string(), object_macro("B", vec![ident("C")]));
        defs.insert("C".to_string(), object_macro("C", vec![ident("D")]));

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 2, &mut counter_val);

        let tokens = vec![ident("A")];
        let result = expander.expand_tokens(&tokens);

        // A→B (depth 1), B→C (depth 2), C exceeds depth 2 → stops
        assert!(diag.has_errors());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "C");
    }

    // -----------------------------------------------------------------------
    // Function-like macro tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_function_like_basic() {
        // #define ADD(a, b) a + b
        let mut defs = FxHashMap::default();
        defs.insert(
            "ADD".to_string(),
            function_macro(
                "ADD",
                vec!["a", "b"],
                false,
                vec![ident("a"), ws(), punct("+"), ws(), ident("b")],
            ),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // ADD(1,2) — no whitespace around args to avoid extra ws in output
        let tokens = vec![
            ident("ADD"),
            punct("("),
            num("1"),
            punct(","),
            num("2"),
            punct(")"),
        ];
        let result = expander.expand_tokens(&tokens);

        let texts: Vec<&str> = result.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["1", " ", "+", " ", "2"]);
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_function_like_no_lparen() {
        // #define FOO(a) a
        // Usage: FOO without parens — should NOT expand
        let mut defs = FxHashMap::default();
        defs.insert(
            "FOO".to_string(),
            function_macro("FOO", vec!["a"], false, vec![ident("a")]),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("FOO"), ws(), ident("bar")];
        let result = expander.expand_tokens(&tokens);

        let texts: Vec<&str> = result.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["FOO", " ", "bar"]);
    }

    #[test]
    fn test_function_like_arg_count_mismatch() {
        // #define TWO(a, b) a b
        let mut defs = FxHashMap::default();
        defs.insert(
            "TWO".to_string(),
            function_macro(
                "TWO",
                vec!["a", "b"],
                false,
                vec![ident("a"), ws(), ident("b")],
            ),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // TWO(1) — too few arguments
        let tokens = vec![ident("TWO"), punct("("), num("1"), punct(")")];
        let result = expander.expand_tokens(&tokens);
        assert!(diag.has_errors());
        assert!(result.is_empty()); // Expansion failed.
    }

    #[test]
    fn test_zero_param_macro() {
        // #define NOP() 42
        let mut defs = FxHashMap::default();
        defs.insert(
            "NOP".to_string(),
            function_macro("NOP", vec![], false, vec![num("42")]),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // NOP()
        let tokens = vec![ident("NOP"), punct("("), punct(")")];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "42");
        assert!(!diag.has_errors());
    }

    // -----------------------------------------------------------------------
    // Variadic macro tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_variadic_va_args() {
        // #define LOG(...) __VA_ARGS__
        let mut defs = FxHashMap::default();
        defs.insert(
            "LOG".to_string(),
            function_macro("LOG", vec![], true, vec![ident("__VA_ARGS__")]),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // LOG(1,2,3) — no whitespace around args
        let tokens = vec![
            ident("LOG"),
            punct("("),
            num("1"),
            punct(","),
            num("2"),
            punct(","),
            num("3"),
            punct(")"),
        ];
        let result = expander.expand_tokens(&tokens);

        // VA_ARGS inserts ", " between variadic args
        let texts: Vec<&str> = result.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["1", ",", " ", "2", ",", " ", "3"]);
    }

    #[test]
    fn test_variadic_empty_va_args() {
        // #define LOG(...) __VA_ARGS__
        // LOG() → empty
        let mut defs = FxHashMap::default();
        defs.insert(
            "LOG".to_string(),
            function_macro("LOG", vec![], true, vec![ident("__VA_ARGS__")]),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("LOG"), punct("("), punct(")")];
        let result = expander.expand_tokens(&tokens);

        assert!(result.is_empty() || result.iter().all(|t| t.is_whitespace()));
    }

    // -----------------------------------------------------------------------
    // Stringification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stringify_operator() {
        // #define STR(x) #x
        let mut defs = FxHashMap::default();
        defs.insert(
            "STR".to_string(),
            function_macro("STR", vec!["x"], false, vec![punct("#"), ident("x")]),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // STR(hello)
        let tokens = vec![ident("STR"), punct("("), ident("hello"), punct(")")];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].kind, PPTokenKind::StringLiteral);
        assert_eq!(result[0].text, "\"hello\"");
    }

    // -----------------------------------------------------------------------
    // Token paste tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_paste() {
        // #define PASTE(a, b) a ## b
        let mut defs = FxHashMap::default();
        defs.insert(
            "PASTE".to_string(),
            function_macro(
                "PASTE",
                vec!["a", "b"],
                false,
                vec![ident("a"), ws(), punct("##"), ws(), ident("b")],
            ),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        // PASTE(foo, bar) → foobar
        let tokens = vec![
            ident("PASTE"),
            punct("("),
            ident("foo"),
            punct(","),
            ws(),
            ident("bar"),
            punct(")"),
        ];
        let result = expander.expand_tokens(&tokens);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "foobar");
        assert_eq!(result[0].kind, PPTokenKind::Identifier);
    }

    // -----------------------------------------------------------------------
    // Rescan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rescan_with_remaining_tokens() {
        // #define CALL FUNC
        // #define FUNC(x) x + 1
        // CALL(42) → should expand CALL → FUNC, then FUNC(42) → 42 + 1
        let mut defs = FxHashMap::default();
        defs.insert(
            "CALL".to_string(),
            object_macro("CALL", vec![ident("FUNC")]),
        );
        defs.insert(
            "FUNC".to_string(),
            function_macro(
                "FUNC",
                vec!["x"],
                false,
                vec![ident("x"), ws(), punct("+"), ws(), num("1")],
            ),
        );

        let mut diag = DiagnosticEngine::new();
        let mut counter_val: u64 = 0;
        let mut expander = MacroExpander::new(&defs, &mut diag, 512, &mut counter_val);

        let tokens = vec![ident("CALL"), punct("("), num("42"), punct(")")];
        let result = expander.expand_tokens(&tokens);

        let texts: Vec<&str> = result.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["42", " ", "+", " ", "1"]);
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_lparen_skips_whitespace() {
        let tokens = vec![ws(), ws(), punct("(")];
        assert_eq!(find_lparen(&tokens, 0), Some(2));
    }

    #[test]
    fn test_find_lparen_no_paren() {
        let tokens = vec![ws(), ident("x")];
        assert_eq!(find_lparen(&tokens, 0), None);
    }

    #[test]
    fn test_find_lparen_empty() {
        let tokens: Vec<PPToken> = vec![];
        assert_eq!(find_lparen(&tokens, 0), None);
    }

    #[test]
    fn test_is_arg_empty_true() {
        assert!(is_arg_empty(&[]));
        assert!(is_arg_empty(&[ws()]));
    }

    #[test]
    fn test_is_arg_empty_false() {
        assert!(!is_arg_empty(&[num("1")]));
    }

    #[test]
    fn test_collect_va_opt_content_basic() {
        // __VA_OPT__( hello )
        let tokens = vec![punct("("), ident("hello"), punct(")")];
        let (content, end) = collect_va_opt_content(&tokens, 0);
        assert_eq!(content.len(), 1);
        assert_eq!(content[0].text, "hello");
        assert_eq!(end, 3);
    }

    #[test]
    fn test_collect_va_opt_content_nested_parens() {
        // __VA_OPT__( (a, b) )
        let tokens = vec![
            punct("("),
            punct("("),
            ident("a"),
            punct(","),
            ident("b"),
            punct(")"),
            punct(")"),
        ];
        let (content, end) = collect_va_opt_content(&tokens, 0);
        assert_eq!(content.len(), 5); // ( a , b )
        assert_eq!(end, 7);
    }

    #[test]
    fn test_paint_state_check_in_expansion() {
        // Verify that check_token_paint returns Painted/Unpainted correctly.
        let marker = PaintMarker::new();
        assert_eq!(marker.check_token_paint("X"), PaintState::Unpainted);
    }

    #[test]
    fn test_preceded_by_hashhash_basic() {
        let tokens = vec![ident("a"), punct("##"), ident("b")];
        assert!(!preceded_by_hashhash(&tokens, 0));
        assert!(preceded_by_hashhash(&tokens, 2));
    }

    #[test]
    fn test_followed_by_hashhash_basic() {
        let tokens = vec![ident("a"), punct("##"), ident("b")];
        assert!(followed_by_hashhash(&tokens, 0));
        assert!(!followed_by_hashhash(&tokens, 2));
    }
}
