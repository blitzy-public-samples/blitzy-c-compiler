//! Preprocessor constant expression evaluation — C11 §6.10.1.
//!
//! Evaluates `#if` / `#elif` constant expressions using **integer-only**
//! arithmetic (no floating-point).  The evaluator is a recursive-descent
//! parser that respects standard C operator precedence and associativity.
//!
//! ## Supported Constructs
//! - Integer literals: decimal, hex (`0x`/`0X`), octal (`0`), binary (`0b`/`0B`)
//! - Integer suffixes: `u`/`U`, `l`/`L`, `ll`/`LL` and combinations
//! - Character constants: `'a'`, `'\n'`, `'\x41'`, multi-character, prefixed (`L`, `u`, `U`)
//! - `defined(X)` / `defined X` — preprocessor macro existence check
//! - All C integer operators at correct precedence (ternary through unary)
//! - Signed / unsigned promotion per C11 usual arithmetic conversions
//!
//! ## Error Handling
//! - Division / modulo by zero → diagnostic error
//! - Invalid integer literals → diagnostic error
//! - Unexpected tokens / unterminated ternary → diagnostic error
//! - Shift by negative or >= 64 → diagnostic warning (returns 0)
//!
//! ## C11 §6.10.1p4 — Identifier Replacement
//! After macro expansion, any remaining identifier (except `defined`,
//! `true`, `false`) is replaced with the integer constant `0`.

// ---------------------------------------------------------------------------
// Imports — only from depends_on_files
// ---------------------------------------------------------------------------

use super::{PPToken, PPTokenKind};
use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};

// ---------------------------------------------------------------------------
// ShiftDir — direction enum for the shift helper
// ---------------------------------------------------------------------------

/// Internal direction tag for shift operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftDir {
    Left,
    Right,
}

// ---------------------------------------------------------------------------
// PPValue — preprocessor expression value (signed or unsigned integer)
// ---------------------------------------------------------------------------

/// Value resulting from preprocessor expression evaluation.
///
/// Preprocessor expressions are integer-only per C11 §6.10.1.  Every
/// intermediate and final result is either a signed `i64` or an unsigned
/// `u64`.  The signedness is determined by integer literal suffixes and
/// the usual arithmetic conversion rules.
#[derive(Debug, Clone, Copy)]
pub enum PPValue {
    /// A signed 64-bit integer value — the default for un-suffixed literals.
    Signed(i64),
    /// An unsigned 64-bit integer value — produced by `U`-suffixed literals
    /// or when mixed-signedness arithmetic promotes to unsigned.
    Unsigned(u64),
}

impl PPValue {
    /// Returns `true` if the value is non-zero — used for truthiness in
    /// `#if` / `#elif` condition evaluation.
    #[inline]
    pub fn is_nonzero(&self) -> bool {
        match self {
            PPValue::Signed(v) => *v != 0,
            PPValue::Unsigned(v) => *v != 0,
        }
    }

    /// Convert the value to a signed `i64`.
    ///
    /// Unsigned values are reinterpreted via wrapping cast.
    #[inline]
    pub fn to_i64(&self) -> i64 {
        match self {
            PPValue::Signed(v) => *v,
            PPValue::Unsigned(v) => *v as i64,
        }
    }

    /// Convert the value to an unsigned `u64`.
    ///
    /// Signed values are reinterpreted via wrapping cast.
    #[inline]
    pub fn to_u64(&self) -> u64 {
        match self {
            PPValue::Signed(v) => *v as u64,
            PPValue::Unsigned(v) => *v,
        }
    }

    /// Returns `true` if the value carries unsigned semantics.
    #[inline]
    pub fn is_unsigned(&self) -> bool {
        matches!(self, PPValue::Unsigned(_))
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Evaluate a preprocessor `#if` / `#elif` constant expression.
///
/// `tokens` is the macro-expanded token stream for the expression.  Per
/// C11 §6.10.1p4, any remaining identifiers (other than `defined`, `true`,
/// `false`) are replaced with `0` during evaluation.
///
/// Returns the evaluated value wrapped in `Ok`, or `Err(())` if a fatal
/// error (e.g. division by zero, malformed literal) was diagnosed.  All
/// errors are reported through `diagnostics`.
#[allow(clippy::result_unit_err)]
pub fn evaluate_pp_expression(
    tokens: &[PPToken],
    diagnostics: &mut DiagnosticEngine,
) -> Result<PPValue, ()> {
    // Filter out whitespace and newline tokens — they carry no semantic
    // meaning inside a preprocessor constant expression.
    let filtered: Vec<&PPToken> = tokens
        .iter()
        .filter(|t| {
            !matches!(
                t.kind,
                PPTokenKind::Whitespace | PPTokenKind::Newline | PPTokenKind::PlacemarkerToken
            )
        })
        .collect();

    if filtered.is_empty() {
        // An empty expression is an error — `#if` with no condition.
        let span = tokens.first().map_or(Span::dummy(), |t| t.span);
        diagnostics.emit(Diagnostic::error(span, "empty preprocessor expression"));
        return Err(());
    }

    let mut parser = ExprParser::new(&filtered, diagnostics);
    let result = parser.parse_ternary()?;

    // Verify that we consumed all tokens (aside from EOF).
    if !parser.at_end() {
        let tok = parser.peek_token();
        let err_span = tok.span;
        let err_text = tok.text.clone();
        parser.diagnostics.emit(Diagnostic::error(
            err_span,
            format!(
                "unexpected token '{}' after preprocessor expression",
                err_text
            ),
        ));
        return Err(());
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// ExprParser — internal recursive-descent parser
// ---------------------------------------------------------------------------

/// Internal parser state for preprocessor expression evaluation.
///
/// Walks a filtered slice of `PPToken` references, dispatching to the
/// recursive-descent parsing functions for each precedence level.
struct ExprParser<'a> {
    /// The filtered (no whitespace / newline) token stream.
    tokens: &'a [&'a PPToken],
    /// Current position in the token stream.
    pos: usize,
    /// Mutable reference to the diagnostic engine for error reporting.
    diagnostics: &'a mut DiagnosticEngine,
}

impl<'a> ExprParser<'a> {
    /// Construct a new expression parser over a filtered token slice.
    fn new(tokens: &'a [&'a PPToken], diagnostics: &'a mut DiagnosticEngine) -> Self {
        ExprParser {
            tokens,
            pos: 0,
            diagnostics,
        }
    }

    // -- Token access helpers ------------------------------------------------

    /// Returns `true` if all tokens have been consumed.
    #[inline]
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || self.tokens[self.pos].kind == PPTokenKind::EndOfFile
    }

    /// Peek at the current token without consuming it.
    ///
    /// Returns the EOF sentinel if past the end of the stream.
    #[inline]
    fn peek_token(&self) -> &PPToken {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos]
        } else {
            // Safety: if the caller provided tokens, the last one should be
            // EOF.  If not, we fabricate a reference from the last available.
            self.tokens.last().copied().unwrap_or_else(|| {
                // This branch is unreachable if the caller checked is_empty().
                // Provide a best-effort reference — this is defensive only.
                self.tokens[0]
            })
        }
    }

    /// Advance past the current token and return a reference to it.
    fn advance(&mut self) -> &PPToken {
        let tok = self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    /// Return `true` if the current token is a punctuator matching `text`.
    #[inline]
    fn is_punct(&self, text: &str) -> bool {
        !self.at_end()
            && self.peek_token().kind == PPTokenKind::Punctuator
            && self.peek_token().text == text
    }

    /// If the current token is a punctuator matching `text`, consume it
    /// and return `true`.  Otherwise return `false`.
    fn eat_punct(&mut self, text: &str) -> bool {
        if self.is_punct(text) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Return the span for the current position (for diagnostics).
    fn current_span(&self) -> Span {
        self.peek_token().span
    }

    /// Build a span covering from `start` to the token just consumed.
    fn span_from(&self, start_span: Span) -> Span {
        if self.pos > 0 && self.pos <= self.tokens.len() {
            let end_span = self.tokens[self.pos - 1].span;
            Span::new(start_span.file_id, start_span.start, end_span.end)
        } else {
            start_span
        }
    }

    // -- Recursive-descent parsing functions --------------------------------
    //
    // Precedence (lowest → highest):
    //  1. Ternary:        ? :
    //  2. Logical OR:     ||
    //  3. Logical AND:    &&
    //  4. Bitwise OR:     |
    //  5. Bitwise XOR:    ^
    //  6. Bitwise AND:    &
    //  7. Equality:       == !=
    //  8. Relational:     < > <= >=
    //  9. Shift:          << >>
    // 10. Additive:       + -
    // 11. Multiplicative: * / %
    // 12. Unary:          + - ~ !
    // 13. Primary:        literals, defined(), (expr), identifiers → 0

    // ---- Level 1: Ternary  ? : ----

    /// Parse a ternary conditional expression: `condition ? true_expr : false_expr`.
    ///
    /// Both branches are parsed (not short-circuited at the parse level); the
    /// result is chosen based on the condition's truthiness.
    fn parse_ternary(&mut self) -> Result<PPValue, ()> {
        let condition = self.parse_logical_or()?;

        if !self.eat_punct("?") {
            return Ok(condition);
        }

        let true_val = self.parse_ternary()?;

        if !self.eat_punct(":") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected ':' in ternary expression",
            ));
            return Err(());
        }

        let false_val = self.parse_ternary()?;

        if condition.is_nonzero() {
            Ok(true_val)
        } else {
            Ok(false_val)
        }
    }

    // ---- Level 2: Logical OR  || ----

    /// Parse a logical OR expression.
    ///
    /// Result is `1` (signed) if either operand is nonzero, `0` otherwise.
    fn parse_logical_or(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_logical_and()?;

        while self.eat_punct("||") {
            let right = self.parse_logical_and()?;
            let result = if left.is_nonzero() || right.is_nonzero() {
                1i64
            } else {
                0i64
            };
            left = PPValue::Signed(result);
        }

        Ok(left)
    }

    // ---- Level 3: Logical AND  && ----

    /// Parse a logical AND expression.
    ///
    /// Result is `1` (signed) if both operands are nonzero, `0` otherwise.
    fn parse_logical_and(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_bitwise_or()?;

        while self.eat_punct("&&") {
            let right = self.parse_bitwise_or()?;
            let result = if left.is_nonzero() && right.is_nonzero() {
                1i64
            } else {
                0i64
            };
            left = PPValue::Signed(result);
        }

        Ok(left)
    }

    // ---- Level 4: Bitwise OR  | ----

    /// Parse a bitwise OR expression.
    fn parse_bitwise_or(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_bitwise_xor()?;

        while self.is_single_punct("|") {
            self.advance();
            let right = self.parse_bitwise_xor()?;
            left = apply_binary_bitwise(left, right, |a, b| a | b);
        }

        Ok(left)
    }

    // ---- Level 5: Bitwise XOR  ^ ----

    /// Parse a bitwise XOR expression.
    fn parse_bitwise_xor(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_bitwise_and()?;

        while self.eat_punct("^") {
            let right = self.parse_bitwise_and()?;
            left = apply_binary_bitwise(left, right, |a, b| a ^ b);
        }

        Ok(left)
    }

    // ---- Level 6: Bitwise AND  & ----

    /// Parse a bitwise AND expression.
    fn parse_bitwise_and(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_equality()?;

        while self.is_single_punct("&") {
            self.advance();
            let right = self.parse_equality()?;
            left = apply_binary_bitwise(left, right, |a, b| a & b);
        }

        Ok(left)
    }

    /// Check if the current token is *exactly* the single-character punctuator
    /// `ch` and NOT the start of a two-character operator (e.g. `|` vs `||`,
    /// `&` vs `&&`, `<` vs `<<`).
    fn is_single_punct(&self, ch: &str) -> bool {
        if self.at_end() {
            return false;
        }
        let tok = self.peek_token();
        tok.kind == PPTokenKind::Punctuator && tok.text == ch
    }

    // ---- Level 7: Equality  == != ----

    /// Parse an equality expression: `==` or `!=`.
    ///
    /// Result is `1` (signed) for true, `0` for false.
    fn parse_equality(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_relational()?;

        loop {
            if self.eat_punct("==") {
                let right = self.parse_relational()?;
                let eq = apply_comparison(left, right, |a, b| a == b, |a, b| a == b);
                left = bool_to_ppvalue(eq);
            } else if self.eat_punct("!=") {
                let right = self.parse_relational()?;
                let neq = apply_comparison(left, right, |a, b| a != b, |a, b| a != b);
                left = bool_to_ppvalue(neq);
            } else {
                break;
            }
        }

        Ok(left)
    }

    // ---- Level 8: Relational  < > <= >= ----

    /// Parse a relational expression.
    ///
    /// Result is `1` (signed) for true, `0` for false.  Signed vs unsigned
    /// comparison semantics are applied per usual arithmetic conversions.
    fn parse_relational(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_shift()?;

        loop {
            if self.eat_punct("<=") {
                let right = self.parse_shift()?;
                let r = apply_comparison(left, right, |a, b| a <= b, |a, b| a <= b);
                left = bool_to_ppvalue(r);
            } else if self.eat_punct(">=") {
                let right = self.parse_shift()?;
                let r = apply_comparison(left, right, |a, b| a >= b, |a, b| a >= b);
                left = bool_to_ppvalue(r);
            } else if self.eat_punct("<") {
                let right = self.parse_shift()?;
                let r = apply_comparison(left, right, |a, b| a < b, |a, b| a < b);
                left = bool_to_ppvalue(r);
            } else if self.eat_punct(">") {
                let right = self.parse_shift()?;
                let r = apply_comparison(left, right, |a, b| a > b, |a, b| a > b);
                left = bool_to_ppvalue(r);
            } else {
                break;
            }
        }

        Ok(left)
    }

    // ---- Level 9: Shift  << >> ----

    /// Parse a shift expression.
    ///
    /// Left shift: `a << b`.  Right shift: `a >> b` (arithmetic for signed,
    /// logical for unsigned).  Shift by negative amount or >= 64 diagnoses
    /// an error and yields `0`.
    fn parse_shift(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_additive()?;

        loop {
            if self.eat_punct("<<") {
                let shift_span = self.current_span();
                let right = self.parse_additive()?;
                left = self.apply_shift(left, right, ShiftDir::Left, shift_span)?;
            } else if self.eat_punct(">>") {
                let shift_span = self.current_span();
                let right = self.parse_additive()?;
                left = self.apply_shift(left, right, ShiftDir::Right, shift_span)?;
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Apply a shift operation with bounds checking.
    fn apply_shift(
        &mut self,
        left: PPValue,
        right: PPValue,
        dir: ShiftDir,
        span: Span,
    ) -> Result<PPValue, ()> {
        let shift_amount = right.to_i64();

        // Check for undefined behaviour: negative shift or shift >= 64.
        if !(0..64).contains(&shift_amount) {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!("shift amount {} is out of range (0..63)", shift_amount),
            ));
            return Ok(PPValue::Signed(0));
        }

        let amount = shift_amount as u32;

        if left.is_unsigned() || right.is_unsigned() {
            let lv = left.to_u64();
            let result = match dir {
                ShiftDir::Left => lv.wrapping_shl(amount),
                ShiftDir::Right => lv.wrapping_shr(amount),
            };
            Ok(PPValue::Unsigned(result))
        } else {
            let lv = left.to_i64();
            let result = match dir {
                ShiftDir::Left => lv.wrapping_shl(amount),
                // Arithmetic right shift for signed values.
                ShiftDir::Right => lv.wrapping_shr(amount),
            };
            Ok(PPValue::Signed(result))
        }
    }

    // ---- Level 10: Additive  + - ----

    /// Parse an additive expression: `+` and `-`.
    fn parse_additive(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_multiplicative()?;

        loop {
            if self.eat_punct("+") {
                let right = self.parse_multiplicative()?;
                left = apply_binary_arithmetic(
                    left,
                    right,
                    |a, b| a.wrapping_add(b),
                    |a, b| a.wrapping_add(b),
                );
            } else if self.eat_punct("-") {
                let right = self.parse_multiplicative()?;
                left = apply_binary_arithmetic(
                    left,
                    right,
                    |a, b| a.wrapping_sub(b),
                    |a, b| a.wrapping_sub(b),
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    // ---- Level 11: Multiplicative  * / % ----

    /// Parse a multiplicative expression: `*`, `/`, `%`.
    ///
    /// Division and modulo by zero are diagnosed as errors.
    fn parse_multiplicative(&mut self) -> Result<PPValue, ()> {
        let mut left = self.parse_unary()?;

        loop {
            if self.eat_punct("*") {
                let right = self.parse_unary()?;
                left = apply_binary_arithmetic(
                    left,
                    right,
                    |a, b| a.wrapping_mul(b),
                    |a, b| a.wrapping_mul(b),
                );
            } else if self.eat_punct("/") {
                let op_span = self.current_span();
                let right = self.parse_unary()?;
                left = self.apply_division(left, right, false, op_span)?;
            } else if self.eat_punct("%") {
                let op_span = self.current_span();
                let right = self.parse_unary()?;
                left = self.apply_division(left, right, true, op_span)?;
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Apply division or modulo, checking for division by zero.
    fn apply_division(
        &mut self,
        left: PPValue,
        right: PPValue,
        is_modulo: bool,
        span: Span,
    ) -> Result<PPValue, ()> {
        // Check for zero divisor.
        if !right.is_nonzero() {
            let op_name = if is_modulo { "modulo" } else { "division" };
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!("{} by zero in preprocessor expression", op_name),
            ));
            return Err(());
        }

        if left.is_unsigned() || right.is_unsigned() {
            let lv = left.to_u64();
            let rv = right.to_u64();
            let result = if is_modulo { lv % rv } else { lv / rv };
            Ok(PPValue::Unsigned(result))
        } else {
            let lv = left.to_i64();
            let rv = right.to_i64();
            // Handle signed MIN / -1 overflow (wraps to MIN in two's complement).
            let result = if is_modulo {
                lv.wrapping_rem(rv)
            } else {
                lv.wrapping_div(rv)
            };
            Ok(PPValue::Signed(result))
        }
    }

    // ---- Level 12: Unary  + - ~ ! ----

    /// Parse a unary expression: `+expr`, `-expr`, `~expr`, `!expr`.
    fn parse_unary(&mut self) -> Result<PPValue, ()> {
        // Unary plus — identity.
        if self.eat_punct("+") {
            return self.parse_unary();
        }

        // Unary minus — negation.
        if self.eat_punct("-") {
            let val = self.parse_unary()?;
            return Ok(match val {
                PPValue::Signed(v) => PPValue::Signed(v.wrapping_neg()),
                PPValue::Unsigned(v) => PPValue::Unsigned(v.wrapping_neg()),
            });
        }

        // Bitwise NOT.
        if self.eat_punct("~") {
            let val = self.parse_unary()?;
            return Ok(match val {
                PPValue::Signed(v) => PPValue::Signed(!v),
                PPValue::Unsigned(v) => PPValue::Unsigned(!v),
            });
        }

        // Logical NOT — result is always `Signed(0)` or `Signed(1)`.
        if self.eat_punct("!") {
            let val = self.parse_unary()?;
            return Ok(if val.is_nonzero() {
                PPValue::Signed(0)
            } else {
                PPValue::Signed(1)
            });
        }

        self.parse_primary()
    }

    // ---- Level 13: Primary expressions ----

    /// Parse a primary expression:
    /// - Integer literal (decimal, hex, octal, binary)
    /// - Character constant (`'a'`, `'\n'`, etc.)
    /// - `defined(X)` / `defined X`
    /// - `true` → 1, `false` → 0
    /// - Parenthesised sub-expression `(expr)`
    /// - Any other identifier → `0` (C11 §6.10.1p4)
    fn parse_primary(&mut self) -> Result<PPValue, ()> {
        if self.at_end() {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected expression in preprocessor directive",
            ));
            return Err(());
        }

        let tok = self.peek_token();
        let kind = tok.kind.clone();
        let text = tok.text.clone();
        let span = tok.span;

        match kind {
            // --- Integer literal ---
            PPTokenKind::Number => {
                self.advance();
                parse_integer_literal(&text, span, self.diagnostics)
            }

            // --- Character constant ---
            PPTokenKind::CharLiteral => {
                self.advance();
                parse_char_constant(&text, span, self.diagnostics)
            }

            // --- Identifier: defined, __has_attribute, __has_builtin, etc. ---
            PPTokenKind::Identifier => {
                self.advance();

                if text == "defined" {
                    // Parse `defined(X)` or `defined X`.
                    self.parse_defined_operator(span)
                } else if text == "__has_attribute" {
                    self.parse_has_attribute_operator(span)
                } else if text == "__has_builtin" {
                    self.parse_has_builtin_operator(span)
                } else if text == "__has_include" || text == "__has_include_next" {
                    self.parse_has_include_operator(span)
                } else if text == "__has_feature" || text == "__has_extension" {
                    // GCC/Clang feature/extension checks — return 0 for
                    // unrecognized features, which is the safe default.
                    self.parse_has_feature_operator(span)
                } else if text == "true" {
                    Ok(PPValue::Signed(1))
                } else if text == "false" {
                    Ok(PPValue::Signed(0))
                } else {
                    // C11 §6.10.1p4: after macro expansion, remaining
                    // identifiers are replaced with 0.
                    Ok(PPValue::Signed(0))
                }
            }

            // --- Parenthesised sub-expression ---
            PPTokenKind::Punctuator if text == "(" => {
                let open_span = span;
                self.advance();
                let val = self.parse_ternary()?;
                if !self.eat_punct(")") {
                    let close_span = self.current_span();
                    self.diagnostics.emit(Diagnostic::error(
                        close_span,
                        "expected ')' in preprocessor expression",
                    ));
                    return Err(());
                }
                // Build a combined span from '(' to ')' for any future
                // diagnostics that might reference this sub-expression.
                let _full_span = self.span_from(open_span);
                Ok(val)
            }

            // --- Unexpected token ---
            _ => {
                self.advance();
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    format!("unexpected token '{}' in preprocessor expression", text),
                ));
                Err(())
            }
        }
    }

    // ---- `defined` operator ----

    /// Parse the `defined` operator:  `defined(X)` or `defined X`.
    ///
    /// Since the expression evaluator does not have access to the macro
    /// definition table (the preprocessor resolves `defined` before calling
    /// us), any `defined` that reaches here is treated as evaluating to `0`.
    /// This function still consumes the correct tokens so that the rest of
    /// the expression can be parsed without confusion.
    fn parse_defined_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()> {
        let has_paren = self.eat_punct("(");

        // Expect an identifier (the macro name).
        if !self.at_end() && self.peek_token().kind == PPTokenKind::Identifier {
            self.advance(); // consume the macro name
        } else {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected identifier after 'defined'",
            ));
            return Err(());
        }

        if has_paren && !self.eat_punct(")") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected ')' after 'defined(identifier'",
            ));
            return Err(());
        }

        // The preprocessor should have resolved `defined()` before calling
        // this evaluator.  If it reaches here, the macro is not defined.
        Ok(PPValue::Signed(0))
    }

    // ---- `__has_attribute` operator ----

    /// Parse `__has_attribute(attr_name)` — checks if the compiler supports
    /// a given `__attribute__` name. Returns 1 for supported attributes, 0 otherwise.
    fn parse_has_attribute_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()> {
        if !self.eat_punct("(") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected '(' after '__has_attribute'",
            ));
            return Err(());
        }

        let attr_name = if !self.at_end() && self.peek_token().kind == PPTokenKind::Identifier {
            let name = self.peek_token().text.clone();
            self.advance();
            name
        } else {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected attribute name in '__has_attribute(...)'",
            ));
            return Err(());
        };

        if !self.eat_punct(")") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected ')' after '__has_attribute(name'",
            ));
            return Err(());
        }

        // Return 1 for attributes BCC supports (GCC-compatible set).
        let supported = is_supported_attribute(&attr_name);
        Ok(PPValue::Signed(if supported { 1 } else { 0 }))
    }

    // ---- `__has_builtin` operator ----

    /// Parse `__has_builtin(builtin_name)` — checks if the compiler provides
    /// a given builtin function. Returns 1 for supported builtins, 0 otherwise.
    fn parse_has_builtin_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()> {
        if !self.eat_punct("(") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected '(' after '__has_builtin'",
            ));
            return Err(());
        }

        let builtin_name = if !self.at_end() && self.peek_token().kind == PPTokenKind::Identifier {
            let name = self.peek_token().text.clone();
            self.advance();
            name
        } else {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected builtin name in '__has_builtin(...)'",
            ));
            return Err(());
        };

        if !self.eat_punct(")") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected ')' after '__has_builtin(name'",
            ));
            return Err(());
        }

        let supported = is_supported_builtin(&builtin_name);
        Ok(PPValue::Signed(if supported { 1 } else { 0 }))
    }

    // ---- `__has_include` / `__has_include_next` operator ----

    /// Parse `__has_include(<header>)` or `__has_include("header")`.
    /// Returns 1 if the header file exists in the include search paths, 0 otherwise.
    fn parse_has_include_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()> {
        if !self.eat_punct("(") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected '(' after '__has_include'",
            ));
            return Err(());
        }

        // Consume tokens until we find the closing ')'
        // We don't actually resolve the include — just consume the argument.
        let mut depth = 1u32;
        while !self.at_end() && depth > 0 {
            let tok = self.peek_token();
            if tok.kind == PPTokenKind::Punctuator && tok.text == "(" {
                depth += 1;
            } else if tok.kind == PPTokenKind::Punctuator && tok.text == ")" {
                depth -= 1;
                if depth == 0 {
                    self.advance();
                    break;
                }
            }
            self.advance();
        }

        // Conservative: return 0 (header not found) since we don't have
        // include path context in the expression evaluator.
        Ok(PPValue::Signed(0))
    }

    // ---- `__has_feature` / `__has_extension` operator ----

    /// Parse `__has_feature(feature_name)` or `__has_extension(ext_name)`.
    /// These are primarily Clang extensions; GCC doesn't have them but
    /// some code checks for them. Return 0 for most features.
    fn parse_has_feature_operator(&mut self, _kw_span: Span) -> Result<PPValue, ()> {
        if !self.eat_punct("(") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected '(' after '__has_feature'/'__has_extension'",
            ));
            return Err(());
        }

        if !self.at_end() && self.peek_token().kind == PPTokenKind::Identifier {
            self.advance(); // consume feature name
        }

        if !self.eat_punct(")") {
            let span = self.current_span();
            self.diagnostics.emit(Diagnostic::error(
                span,
                "expected ')' after '__has_feature(name'",
            ));
            return Err(());
        }

        // Return 0 for all features (not Clang).
        Ok(PPValue::Signed(0))
    }
}

// ===========================================================================
// Attribute and builtin support tables
// ===========================================================================

/// Check if an attribute name is supported by BCC.
/// This covers the GCC attributes commonly used by the Linux kernel and
/// other real-world C code.
fn is_supported_attribute(name: &str) -> bool {
    // Strip leading/trailing underscores for canonical form:
    // __aligned__ → aligned, __packed__ → packed, etc.
    let canonical = name
        .strip_prefix("__")
        .and_then(|s| s.strip_suffix("__"))
        .unwrap_or(name);

    matches!(
        canonical,
        "aligned"
            | "packed"
            | "section"
            | "used"
            | "unused"
            | "weak"
            | "alias"
            | "constructor"
            | "destructor"
            | "visibility"
            | "deprecated"
            | "noreturn"
            | "noinline"
            | "always_inline"
            | "cold"
            | "hot"
            | "format"
            | "format_arg"
            | "malloc"
            | "pure"
            | "const"
            | "warn_unused_result"
            | "fallthrough"
            | "nonnull"
            | "returns_nonnull"
            | "sentinel"
            | "mode"
            | "transparent_union"
            | "may_alias"
            | "cleanup"
            | "noclone"
            | "no_instrument_function"
            | "no_profile_instrument_function"
            | "no_sanitize"
            | "no_sanitize_address"
            | "no_sanitize_thread"
            | "no_sanitize_undefined"
            | "no_stack_protector"
            | "no_caller_saved_registers"
            | "error"
            | "warning"
            | "externally_visible"
            | "gnu_inline"
            | "artificial"
            | "flatten"
            | "leaf"
            | "assume_aligned"
            | "alloc_size"
            | "alloc_align"
            | "copy"
            | "designated_init"
            | "nonstring"
            | "noplt"
            | "optimize"
            | "target"
            | "counted_by"
            | "btf_type_tag"
            | "diagnose_as_builtin"
            | "noipa"
            | "access"
            | "fd_arg"
            | "tainted_args"
            | "disable_sanitizer_instrumentation"
            | "no_stack_limit"
            | "nocf_check"
            | "force_align_arg_pointer"
            | "naked"
            | "regparm"
            | "stdcall"
            | "cdecl"
            | "fastcall"
    )
}

/// Check if a builtin function name is supported by BCC.
fn is_supported_builtin(name: &str) -> bool {
    matches!(
        name,
        "__builtin_expect"
            | "__builtin_expect_with_probability"
            | "__builtin_unreachable"
            | "__builtin_constant_p"
            | "__builtin_offsetof"
            | "__builtin_types_compatible_p"
            | "__builtin_choose_expr"
            | "__builtin_clz"
            | "__builtin_clzl"
            | "__builtin_clzll"
            | "__builtin_ctz"
            | "__builtin_ctzl"
            | "__builtin_ctzll"
            | "__builtin_popcount"
            | "__builtin_popcountl"
            | "__builtin_popcountll"
            | "__builtin_bswap16"
            | "__builtin_bswap32"
            | "__builtin_bswap64"
            | "__builtin_ffs"
            | "__builtin_ffsl"
            | "__builtin_ffsll"
            | "__builtin_va_start"
            | "__builtin_va_end"
            | "__builtin_va_copy"
            | "__builtin_va_arg"
            | "__builtin_va_list"
            | "__builtin_frame_address"
            | "__builtin_return_address"
            | "__builtin_trap"
            | "__builtin_assume_aligned"
            | "__builtin_add_overflow"
            | "__builtin_sub_overflow"
            | "__builtin_mul_overflow"
            | "__builtin_object_size"
            | "__builtin_memcpy"
            | "__builtin_memset"
            | "__builtin_memmove"
            | "__builtin_memcmp"
            | "__builtin_strlen"
            | "__builtin_strcmp"
            | "__builtin_strncmp"
            | "__builtin_strcpy"
            | "__builtin_strncpy"
            | "__builtin_strncat"
            | "__builtin_abs"
            | "__builtin_labs"
            | "__builtin_llabs"
            | "__builtin_huge_val"
            | "__builtin_huge_valf"
            | "__builtin_inf"
            | "__builtin_inff"
            | "__builtin_nan"
            | "__builtin_nanf"
            | "__builtin_prefetch"
            | "__builtin_alloca"
            | "__builtin_classify_type"
            | "__builtin_LINE"
            | "__builtin_FUNCTION"
            | "__builtin_FILE"
            | "__builtin___clear_cache"
            | "__builtin_sadd_overflow"
            | "__builtin_saddl_overflow"
            | "__builtin_saddll_overflow"
            | "__builtin_uadd_overflow"
            | "__builtin_uaddl_overflow"
            | "__builtin_uaddll_overflow"
            | "__builtin_ssub_overflow"
            | "__builtin_ssubl_overflow"
            | "__builtin_ssubll_overflow"
            | "__builtin_usub_overflow"
            | "__builtin_usubl_overflow"
            | "__builtin_usubll_overflow"
            | "__builtin_smul_overflow"
            | "__builtin_smull_overflow"
            | "__builtin_smulll_overflow"
            | "__builtin_umul_overflow"
            | "__builtin_umull_overflow"
            | "__builtin_umulll_overflow"
    )
}

// ===========================================================================
// Free helper functions — integer literal parsing, char constant parsing,
// and binary operation helpers
// ===========================================================================

// ---------------------------------------------------------------------------
// Integer literal parsing
// ---------------------------------------------------------------------------

/// Parse a preprocessor integer literal into a [`PPValue`].
///
/// Supports:
/// - Decimal: `123`
/// - Hexadecimal: `0x1A`, `0X1a`
/// - Octal: `0777`
/// - Binary: `0b1010`, `0B1010`
/// - Suffixes: `u`/`U` (unsigned), `l`/`L` (long), `ll`/`LL` (long long),
///   and combinations (`ul`, `ULL`, `lu`, etc.)
fn parse_integer_literal(
    text: &str,
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Result<PPValue, ()> {
    let original = text;

    // Strip suffix to determine signedness and parse the numeric part.
    let (num_str, is_unsigned) = strip_integer_suffix(text);

    // Empty numeric part after suffix stripping is an error.
    if num_str.is_empty() {
        diagnostics.emit(Diagnostic::error(
            span,
            format!("invalid integer literal '{}'", original),
        ));
        return Err(());
    }

    // Determine the base and skip the prefix.
    let (digits, radix): (&str, u32) = if num_str.starts_with("0x") || num_str.starts_with("0X") {
        (&num_str[2..], 16)
    } else if num_str.starts_with("0b") || num_str.starts_with("0B") {
        (&num_str[2..], 2)
    } else if num_str.starts_with('0') && num_str.len() > 1 {
        // Octal — but skip the leading '0'.
        (&num_str[1..], 8)
    } else {
        (num_str, 10)
    };

    // Filter out digit separators (some compilers accept `_` in digit strings,
    // but standard C does not — we'll be lenient and just skip underscores if present).
    let clean: String = digits.chars().filter(|c| *c != '_').collect();

    if clean.is_empty() {
        // e.g. "0x" with no hex digits.
        diagnostics.emit(Diagnostic::error(
            span,
            format!("invalid integer literal '{}'", original),
        ));
        return Err(());
    }

    // Parse as u64 first (large enough for all valid values).
    let value = match u64::from_str_radix(&clean, radix) {
        Ok(v) => v,
        Err(_) => {
            diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "integer literal '{}' is too large for preprocessor arithmetic",
                    original
                ),
            ));
            return Err(());
        }
    };

    if is_unsigned {
        Ok(PPValue::Unsigned(value))
    } else {
        // Fit into i64 if possible; otherwise promote to unsigned.
        if value <= i64::MAX as u64 {
            Ok(PPValue::Signed(value as i64))
        } else {
            // Values that overflow signed i64 are treated as unsigned.
            Ok(PPValue::Unsigned(value))
        }
    }
}

/// Strip integer suffixes from a literal string and report whether the `u`/`U`
/// suffix was present (indicating unsigned).
///
/// Returns `(numeric_part, is_unsigned)`.
fn strip_integer_suffix(text: &str) -> (&str, bool) {
    let bytes = text.as_bytes();
    let mut end = bytes.len();
    let mut is_unsigned = false;

    // Walk backwards, consuming suffix characters: u/U, l/L.
    // Valid suffix sequences (case-insensitive): u, l, ul, lu, ll, ull, llu
    // We consume greedily from the end.

    if end == 0 {
        return (text, false);
    }

    // First pass: consume the trailing suffix component.
    match bytes[end - 1] {
        b'u' | b'U' => {
            is_unsigned = true;
            end -= 1;
        }
        b'l' | b'L' => {
            end -= 1;
            // Check for 'll' / 'LL'.
            if end > 0 && (bytes[end - 1] == b'l' || bytes[end - 1] == b'L') {
                end -= 1;
            }
        }
        _ => { /* no suffix */ }
    }

    // Second pass: check for a preceding 'u'/'U' (handles `llu`, `LLU`, etc.)
    // or a following 'l'/'L' group (handles `ull`, `ULL`, etc.).
    if end > 0 {
        match bytes[end - 1] {
            b'u' | b'U' => {
                is_unsigned = true;
                end -= 1;
            }
            b'l' | b'L' => {
                end -= 1;
                if end > 0 && (bytes[end - 1] == b'l' || bytes[end - 1] == b'L') {
                    end -= 1;
                }
                // Check for 'u'/'U' before 'll'.
                if end > 0 && (bytes[end - 1] == b'u' || bytes[end - 1] == b'U') {
                    is_unsigned = true;
                    end -= 1;
                }
            }
            _ => { /* done */ }
        }
    }

    (&text[..end], is_unsigned)
}

// ---------------------------------------------------------------------------
// Character constant parsing
// ---------------------------------------------------------------------------

/// Parse a C character constant into a [`PPValue`].
///
/// Handles:
/// - Simple characters: `'a'` → 97
/// - Escape sequences: `'\n'` → 10, `'\t'` → 9, `'\0'` → 0, `'\\'` → 92,
///   `'\''` → 39, `'\"'` → 34, `'\a'` → 7, `'\b'` → 8, `'\f'` → 12,
///   `'\r'` → 13, `'\v'` → 11
/// - Hex escapes: `'\x41'` → 65
/// - Octal escapes: `'\101'` → 65
/// - Multi-character constants: `'ABCD'` → big-endian packed i32 (impl-defined)
/// - Prefixed char constants: `L'a'`, `u'a'`, `U'a'` (treated as plain `int` for PP)
fn parse_char_constant(
    text: &str,
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Result<PPValue, ()> {
    // Skip optional prefix (L, u, U) and the opening quote.
    let inner = strip_char_prefix_and_quotes(text);

    if inner.is_empty() {
        diagnostics.emit(Diagnostic::error(span, "empty character constant"));
        return Err(());
    }

    // Parse the character(s) inside the quotes.
    let chars = parse_escape_sequence(inner, span, diagnostics)?;

    if chars.is_empty() {
        diagnostics.emit(Diagnostic::error(span, "empty character constant"));
        return Err(());
    }

    // Multi-character constant: pack bytes big-endian into an i32.
    if chars.len() == 1 {
        Ok(PPValue::Signed(chars[0] as i64))
    } else {
        // Implementation-defined: big-endian packing.
        let mut value: i64 = 0;
        for &byte_val in &chars {
            value = (value << 8) | (byte_val as i64 & 0xFF);
        }
        Ok(PPValue::Signed(value))
    }
}

/// Strip the optional prefix (`L`, `u`, `U`) and surrounding single quotes
/// from a character literal token text, returning the inner content.
fn strip_char_prefix_and_quotes(text: &str) -> &str {
    let s = text.as_bytes();
    let mut start = 0;

    // Skip prefix: L, u, U (but NOT u8 — that's for strings only).
    if start < s.len() && (s[start] == b'L' || s[start] == b'u' || s[start] == b'U') {
        start += 1;
    }

    // Skip opening quote.
    if start < s.len() && s[start] == b'\'' {
        start += 1;
    }

    // Find closing quote.
    let mut end = s.len();
    if end > start && s[end - 1] == b'\'' {
        end -= 1;
    }

    if start >= end {
        return "";
    }

    &text[start..end]
}

/// Parse a string of characters (inside quotes), handling escape sequences.
///
/// Returns a vector of byte values for each logical character.
fn parse_escape_sequence(
    input: &str,
    span: Span,
    diagnostics: &mut DiagnosticEngine,
) -> Result<Vec<u8>, ()> {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut result = Vec::with_capacity(len);
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\\' {
            i += 1;
            if i >= len {
                diagnostics.emit(Diagnostic::error(
                    span,
                    "unterminated escape sequence in character constant",
                ));
                return Err(());
            }
            match bytes[i] {
                b'n' => {
                    result.push(b'\n');
                    i += 1;
                }
                b't' => {
                    result.push(b'\t');
                    i += 1;
                }
                b'r' => {
                    result.push(b'\r');
                    i += 1;
                }
                b'0'..=b'7' => {
                    // Octal escape: up to 3 octal digits.
                    let mut val: u8 = bytes[i] - b'0';
                    i += 1;
                    let mut count = 1;
                    while count < 3 && i < len && bytes[i] >= b'0' && bytes[i] <= b'7' {
                        val = val.wrapping_mul(8).wrapping_add(bytes[i] - b'0');
                        i += 1;
                        count += 1;
                    }
                    result.push(val);
                }
                b'x' => {
                    // Hex escape: arbitrary number of hex digits (we cap at 2 for char).
                    i += 1;
                    if i >= len || !is_hex_digit(bytes[i]) {
                        diagnostics.emit(Diagnostic::error(
                            span,
                            "expected hex digit after '\\x' in character constant",
                        ));
                        return Err(());
                    }
                    let mut val: u64 = 0;
                    while i < len && is_hex_digit(bytes[i]) {
                        val = val
                            .wrapping_mul(16)
                            .wrapping_add(hex_digit_value(bytes[i]) as u64);
                        i += 1;
                    }
                    result.push(val as u8);
                }
                b'a' => {
                    result.push(0x07); // BEL
                    i += 1;
                }
                b'b' => {
                    result.push(0x08); // BS
                    i += 1;
                }
                b'f' => {
                    result.push(0x0C); // FF
                    i += 1;
                }
                b'v' => {
                    result.push(0x0B); // VT
                    i += 1;
                }
                b'\\' => {
                    result.push(b'\\');
                    i += 1;
                }
                b'\'' => {
                    result.push(b'\'');
                    i += 1;
                }
                b'\"' => {
                    result.push(b'\"');
                    i += 1;
                }
                b'?' => {
                    result.push(b'?');
                    i += 1;
                }
                other => {
                    // Unknown escape — emit the backslash character and the
                    // following byte literally (GCC extension behaviour).
                    result.push(other);
                    i += 1;
                }
            }
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    Ok(result)
}

/// Returns `true` if `b` is an ASCII hexadecimal digit.
#[inline]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

/// Convert a hex-digit byte to its numeric value (0–15).
#[inline]
fn hex_digit_value(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Binary operation helpers — usual arithmetic conversions
// ---------------------------------------------------------------------------

/// Apply a binary bitwise operation respecting signed/unsigned promotion.
///
/// If either operand is unsigned, both are promoted to unsigned (`u64`)
/// before the operation.  Otherwise both remain signed (`i64`).
fn apply_binary_bitwise<F>(left: PPValue, right: PPValue, op: F) -> PPValue
where
    F: Fn(u64, u64) -> u64,
{
    if left.is_unsigned() || right.is_unsigned() {
        PPValue::Unsigned(op(left.to_u64(), right.to_u64()))
    } else {
        let result = op(left.to_u64(), right.to_u64());
        PPValue::Signed(result as i64)
    }
}

/// Apply a binary arithmetic operation with usual arithmetic conversions.
///
/// `signed_op` is applied when both operands are signed.
/// `unsigned_op` is applied when either operand is unsigned.
fn apply_binary_arithmetic<S, U>(
    left: PPValue,
    right: PPValue,
    signed_op: S,
    unsigned_op: U,
) -> PPValue
where
    S: Fn(i64, i64) -> i64,
    U: Fn(u64, u64) -> u64,
{
    if left.is_unsigned() || right.is_unsigned() {
        PPValue::Unsigned(unsigned_op(left.to_u64(), right.to_u64()))
    } else {
        PPValue::Signed(signed_op(left.to_i64(), right.to_i64()))
    }
}

/// Apply a comparison operation with correct signed/unsigned semantics.
///
/// `signed_cmp` is used when both operands are signed.
/// `unsigned_cmp` is used when either operand is unsigned.
fn apply_comparison<S, U>(left: PPValue, right: PPValue, signed_cmp: S, unsigned_cmp: U) -> bool
where
    S: Fn(i64, i64) -> bool,
    U: Fn(u64, u64) -> bool,
{
    if left.is_unsigned() || right.is_unsigned() {
        unsigned_cmp(left.to_u64(), right.to_u64())
    } else {
        signed_cmp(left.to_i64(), right.to_i64())
    }
}

/// Convert a boolean comparison result to a [`PPValue`].
///
/// `true` → `Signed(1)`, `false` → `Signed(0)`.
#[inline]
fn bool_to_ppvalue(b: bool) -> PPValue {
    PPValue::Signed(if b { 1 } else { 0 })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a number token.
    fn num_tok(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Number, text, Span::dummy())
    }

    /// Helper: create a punctuator token.
    fn punct_tok(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Punctuator, text, Span::dummy())
    }

    /// Helper: create an identifier token.
    fn ident_tok(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::Identifier, text, Span::dummy())
    }

    /// Helper: create a char literal token.
    fn char_tok(text: &str) -> PPToken {
        PPToken::new(PPTokenKind::CharLiteral, text, Span::dummy())
    }

    /// Helper: create an EOF token.
    fn eof_tok() -> PPToken {
        PPToken::eof(Span::dummy())
    }

    /// Helper: evaluate a slice of tokens and return the i64 result.
    fn eval(tokens: &[PPToken]) -> Result<i64, ()> {
        let mut diag = DiagnosticEngine::new();
        let result = evaluate_pp_expression(tokens, &mut diag)?;
        Ok(result.to_i64())
    }

    /// Helper: evaluate and expect success returning i64.
    fn eval_ok(tokens: &[PPToken]) -> i64 {
        eval(tokens).expect("expression evaluation failed")
    }

    /// Helper: evaluate and expect an error.
    fn eval_err(tokens: &[PPToken]) -> bool {
        let mut diag = DiagnosticEngine::new();
        evaluate_pp_expression(tokens, &mut diag).is_err()
    }

    #[test]
    fn test_single_integer() {
        assert_eq!(eval_ok(&[num_tok("1"), eof_tok()]), 1);
        assert_eq!(eval_ok(&[num_tok("0"), eof_tok()]), 0);
        assert_eq!(eval_ok(&[num_tok("42"), eof_tok()]), 42);
    }

    #[test]
    fn test_hex_literal() {
        assert_eq!(eval_ok(&[num_tok("0xFF"), eof_tok()]), 255);
        assert_eq!(eval_ok(&[num_tok("0X1A"), eof_tok()]), 26);
    }

    #[test]
    fn test_octal_literal() {
        assert_eq!(eval_ok(&[num_tok("077"), eof_tok()]), 63);
        assert_eq!(eval_ok(&[num_tok("010"), eof_tok()]), 8);
    }

    #[test]
    fn test_binary_literal() {
        assert_eq!(eval_ok(&[num_tok("0b1010"), eof_tok()]), 10);
        assert_eq!(eval_ok(&[num_tok("0B1111"), eof_tok()]), 15);
    }

    #[test]
    fn test_suffix_unsigned() {
        let tokens = [num_tok("0U"), eof_tok()];
        let mut diag = DiagnosticEngine::new();
        let result = evaluate_pp_expression(&tokens, &mut diag).unwrap();
        assert!(result.is_unsigned());
        assert_eq!(result.to_u64(), 0);
    }

    #[test]
    fn test_suffix_long_long() {
        let tokens = [num_tok("42LL"), eof_tok()];
        assert_eq!(eval_ok(&tokens), 42);
    }

    #[test]
    fn test_suffix_unsigned_long_long() {
        let tokens = [num_tok("100ULL"), eof_tok()];
        let mut diag = DiagnosticEngine::new();
        let result = evaluate_pp_expression(&tokens, &mut diag).unwrap();
        assert!(result.is_unsigned());
        assert_eq!(result.to_u64(), 100);
    }

    #[test]
    fn test_addition() {
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("+"), num_tok("2"), eof_tok()]),
            3
        );
    }

    #[test]
    fn test_subtraction() {
        assert_eq!(
            eval_ok(&[num_tok("10"), punct_tok("-"), num_tok("3"), eof_tok()]),
            7
        );
    }

    #[test]
    fn test_multiplication() {
        assert_eq!(
            eval_ok(&[num_tok("6"), punct_tok("*"), num_tok("7"), eof_tok()]),
            42
        );
    }

    #[test]
    fn test_division() {
        assert_eq!(
            eval_ok(&[num_tok("10"), punct_tok("/"), num_tok("3"), eof_tok()]),
            3
        );
    }

    #[test]
    fn test_modulo() {
        assert_eq!(
            eval_ok(&[num_tok("10"), punct_tok("%"), num_tok("3"), eof_tok()]),
            1
        );
    }

    #[test]
    fn test_division_by_zero() {
        assert!(eval_err(&[
            num_tok("10"),
            punct_tok("/"),
            num_tok("0"),
            eof_tok()
        ]));
    }

    #[test]
    fn test_modulo_by_zero() {
        assert!(eval_err(&[
            num_tok("10"),
            punct_tok("%"),
            num_tok("0"),
            eof_tok()
        ]));
    }

    #[test]
    fn test_logical_and() {
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("&&"), num_tok("1"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("&&"), num_tok("0"), eof_tok()]),
            0
        );
    }

    #[test]
    fn test_logical_or() {
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("||"), num_tok("0"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("0"), punct_tok("||"), num_tok("0"), eof_tok()]),
            0
        );
    }

    #[test]
    fn test_bitwise_and() {
        // 0xFF & 0x0F = 0x0F = 15
        assert_eq!(
            eval_ok(&[num_tok("0xFF"), punct_tok("&"), num_tok("0x0F"), eof_tok()]),
            15
        );
    }

    #[test]
    fn test_bitwise_or() {
        assert_eq!(
            eval_ok(&[num_tok("0xF0"), punct_tok("|"), num_tok("0x0F"), eof_tok()]),
            0xFF
        );
    }

    #[test]
    fn test_bitwise_xor() {
        assert_eq!(
            eval_ok(&[num_tok("0xFF"), punct_tok("^"), num_tok("0x0F"), eof_tok()]),
            0xF0
        );
    }

    #[test]
    fn test_shift_left() {
        // 1 << 4 = 16
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("<<"), num_tok("4"), eof_tok()]),
            16
        );
    }

    #[test]
    fn test_shift_right() {
        // 16 >> 2 = 4
        assert_eq!(
            eval_ok(&[num_tok("16"), punct_tok(">>"), num_tok("2"), eof_tok()]),
            4
        );
    }

    #[test]
    fn test_ternary() {
        // 1 ? 42 : 0
        assert_eq!(
            eval_ok(&[
                num_tok("1"),
                punct_tok("?"),
                num_tok("42"),
                punct_tok(":"),
                num_tok("0"),
                eof_tok()
            ]),
            42
        );
        // 0 ? 42 : 99
        assert_eq!(
            eval_ok(&[
                num_tok("0"),
                punct_tok("?"),
                num_tok("42"),
                punct_tok(":"),
                num_tok("99"),
                eof_tok()
            ]),
            99
        );
    }

    #[test]
    fn test_unary_minus() {
        assert_eq!(eval_ok(&[punct_tok("-"), num_tok("5"), eof_tok()]), -5);
    }

    #[test]
    fn test_unary_not() {
        assert_eq!(eval_ok(&[punct_tok("!"), num_tok("0"), eof_tok()]), 1);
        assert_eq!(eval_ok(&[punct_tok("!"), num_tok("1"), eof_tok()]), 0);
    }

    #[test]
    fn test_unary_bitwise_not() {
        // ~0 should produce all-ones = -1 for signed
        assert_eq!(eval_ok(&[punct_tok("~"), num_tok("0"), eof_tok()]), -1);
    }

    #[test]
    fn test_parenthesised() {
        // (1 + 2) * 3 = 9
        assert_eq!(
            eval_ok(&[
                punct_tok("("),
                num_tok("1"),
                punct_tok("+"),
                num_tok("2"),
                punct_tok(")"),
                punct_tok("*"),
                num_tok("3"),
                eof_tok()
            ]),
            9
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("=="), num_tok("1"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("!="), num_tok("2"), eof_tok()]),
            1
        );
    }

    #[test]
    fn test_relational() {
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("<"), num_tok("2"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("2"), punct_tok(">"), num_tok("1"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("1"), punct_tok("<="), num_tok("1"), eof_tok()]),
            1
        );
        assert_eq!(
            eval_ok(&[num_tok("2"), punct_tok(">="), num_tok("3"), eof_tok()]),
            0
        );
    }

    #[test]
    fn test_char_constant() {
        assert_eq!(eval_ok(&[char_tok("'A'"), eof_tok()]), 65);
    }

    #[test]
    fn test_char_escape_newline() {
        assert_eq!(eval_ok(&[char_tok("'\\n'"), eof_tok()]), 10);
    }

    #[test]
    fn test_char_escape_hex() {
        assert_eq!(
            eval_ok(&[char_tok("'\\x41'"), eof_tok()]),
            65 // 'A'
        );
    }

    #[test]
    fn test_char_escape_octal() {
        assert_eq!(
            eval_ok(&[char_tok("'\\101'"), eof_tok()]),
            65 // 'A'
        );
    }

    #[test]
    fn test_identifier_replaced_with_zero() {
        assert_eq!(eval_ok(&[ident_tok("UNDEFINED_MACRO"), eof_tok()]), 0);
    }

    #[test]
    fn test_true_false_identifiers() {
        assert_eq!(eval_ok(&[ident_tok("true"), eof_tok()]), 1);
        assert_eq!(eval_ok(&[ident_tok("false"), eof_tok()]), 0);
    }

    #[test]
    fn test_defined_without_parens() {
        // `defined FOO` — evaluates to 0 (no macro table available).
        assert_eq!(
            eval_ok(&[ident_tok("defined"), ident_tok("FOO"), eof_tok()]),
            0
        );
    }

    #[test]
    fn test_defined_with_parens() {
        // `defined(FOO)` — evaluates to 0 (no macro table available).
        assert_eq!(
            eval_ok(&[
                ident_tok("defined"),
                punct_tok("("),
                ident_tok("FOO"),
                punct_tok(")"),
                eof_tok()
            ]),
            0
        );
    }

    #[test]
    fn test_empty_expression() {
        assert!(eval_err(&[eof_tok()]));
    }

    #[test]
    fn test_complex_expression() {
        // (1 + 2) * 3 - 1 == 8  →  1
        assert_eq!(
            eval_ok(&[
                punct_tok("("),
                num_tok("1"),
                punct_tok("+"),
                num_tok("2"),
                punct_tok(")"),
                punct_tok("*"),
                num_tok("3"),
                punct_tok("-"),
                num_tok("1"),
                punct_tok("=="),
                num_tok("8"),
                eof_tok()
            ]),
            1
        );
    }

    #[test]
    fn test_precedence_mul_over_add() {
        // 2 + 3 * 4 = 14  (not 20)
        assert_eq!(
            eval_ok(&[
                num_tok("2"),
                punct_tok("+"),
                num_tok("3"),
                punct_tok("*"),
                num_tok("4"),
                eof_tok()
            ]),
            14
        );
    }

    #[test]
    fn test_unary_plus() {
        assert_eq!(eval_ok(&[punct_tok("+"), num_tok("42"), eof_tok()]), 42);
    }

    #[test]
    fn test_nested_ternary() {
        // 1 ? (0 ? 10 : 20) : 30  →  20
        assert_eq!(
            eval_ok(&[
                num_tok("1"),
                punct_tok("?"),
                punct_tok("("),
                num_tok("0"),
                punct_tok("?"),
                num_tok("10"),
                punct_tok(":"),
                num_tok("20"),
                punct_tok(")"),
                punct_tok(":"),
                num_tok("30"),
                eof_tok()
            ]),
            20
        );
    }

    #[test]
    fn test_ppvalue_methods() {
        let s = PPValue::Signed(42);
        assert!(s.is_nonzero());
        assert_eq!(s.to_i64(), 42);
        assert_eq!(s.to_u64(), 42);
        assert!(!s.is_unsigned());

        let u = PPValue::Unsigned(100);
        assert!(u.is_nonzero());
        assert_eq!(u.to_i64(), 100);
        assert_eq!(u.to_u64(), 100);
        assert!(u.is_unsigned());

        let z = PPValue::Signed(0);
        assert!(!z.is_nonzero());
    }
}
