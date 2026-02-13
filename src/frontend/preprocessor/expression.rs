// src/frontend/preprocessor/expression.rs
//
// Preprocessor #if / #elif constant expression evaluator.
//
// Implements the full C11 §6.10.1 preprocessor constant expression grammar
// using a precedence-climbing parser. All arithmetic is performed on `i64`
// values, with unsigned signedness tracked alongside the value to correctly
// implement the "usual arithmetic conversions" for preprocessor expressions:
// when one operand is unsigned the result is unsigned.
//
// Processing order for a `#if` expression token sequence:
//   1. Resolve `defined(NAME)` / `defined NAME` operators (before any macro
//      expansion, since `defined` must not have its operand expanded).
//   2. Expand remaining object-like macros in the token stream.
//   3. Replace any surviving identifiers with integer literal `0` (per C11
//      §6.10.1p4 — undefined identifiers evaluate to zero with no diagnostic).
//   4. Parse and evaluate the resulting arithmetic expression.
//
// Operator precedence (lowest → highest):
//   1  —  `? :`         (right-associative ternary conditional)
//   2  —  `||`          (logical OR, short-circuiting)
//   3  —  `&&`          (logical AND, short-circuiting)
//   4  —  `|`           (bitwise OR)
//   5  —  `^`           (bitwise XOR)
//   6  —  `&`           (bitwise AND)
//   7  —  `==`  `!=`    (equality)
//   8  —  `<` `>` `<=` `>=`  (relational)
//   9  —  `<<`  `>>`    (shift)
//  10  —  `+`  `-`      (additive)
//  11  —  `*`  `/`  `%` (multiplicative)
//  12  —  unary `+` `-` `~` `!` (prefix operators)
//
// Short-circuit semantics are fully honoured: the right operand of `&&`,
// `||`, and the unselected branch of `? :` are parsed but not evaluated,
// suppressing diagnostics such as division-by-zero in dead branches.

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::frontend::lexer::token::{CharPrefix, IntegerSuffix, Token, TokenKind};

use super::MacroDef;

// ═══════════════════════════════════════════════════════════════════════════
// PPValue — preprocessor expression value with signedness tracking
// ═══════════════════════════════════════════════════════════════════════════

/// An integer value produced during preprocessor expression evaluation.
///
/// The C preprocessor operates on `intmax_t` (signed) or `uintmax_t`
/// (unsigned) values. We store both as `i64` and track whether the value
/// should be interpreted as unsigned for comparison and division purposes.
/// Unsigned overflow wraps via two's complement as required by C semantics.
#[derive(Clone, Copy, Debug)]
struct PPValue {
    /// The raw 64-bit integer value (two's-complement).
    val: i64,
    /// `true` when the value originated from an unsigned literal or was
    /// promoted to unsigned via the usual arithmetic conversions.
    is_unsigned: bool,
}

impl PPValue {
    /// Create a signed zero — the default result for error recovery and
    /// undefined-identifier replacement.
    #[inline]
    fn zero() -> Self {
        PPValue {
            val: 0,
            is_unsigned: false,
        }
    }

    /// Create a signed value.
    #[inline]
    fn signed(val: i64) -> Self {
        PPValue {
            val,
            is_unsigned: false,
        }
    }

    /// Create an unsigned value.
    #[inline]
    fn unsigned(val: i64) -> Self {
        PPValue {
            val,
            is_unsigned: true,
        }
    }

    /// Boolean truth test (non-zero → true).
    #[inline]
    fn is_true(self) -> bool {
        self.val != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluates a preprocessor `#if` or `#elif` constant expression.
///
/// The `tokens` slice contains the token sequence following the directive
/// keyword (e.g. everything after `#if` / `#elif` up to the end of the
/// logical line). The caller must not include the directive keyword itself.
///
/// # Processing Steps
///
/// 1. **`defined` handling** — `defined(NAME)` and `defined NAME` forms
///    are resolved to integer literal tokens `1` or `0` before macro expansion,
///    since the operand of `defined` must not be macro-expanded.
/// 2. **Macro expansion** — remaining object-like macros are expanded
///    inline (function-like macros are left unexpanded per simplification).
/// 3. **Identifier replacement** — any identifier tokens surviving
///    expansion are replaced with `0` (C11 §6.10.1p4).
/// 4. **Expression evaluation** — the resulting token stream is parsed
///    via precedence-climbing and evaluated to a single `i64`.
///
/// # Returns
///
/// `Ok(value)` where `value != 0` means "condition is true" and
/// `value == 0` means "condition is false".
///
/// `Err(())` on unrecoverable parse failure (a diagnostic will already have
/// been emitted).
#[allow(clippy::result_unit_err)]
pub fn evaluate_expression(
    tokens: &[Token],
    macros: &FxHashMap<Symbol, MacroDef>,
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Result<i64, ()> {
    // Fast-path: empty expression.
    if tokens.is_empty() || tokens.iter().all(|t| matches!(t.kind, TokenKind::Eof)) {
        diag.error(Span::DUMMY, "expected expression in preprocessor directive");
        return Err(());
    }

    // ── Phase 1: resolve `defined` operators ────────────────────────────
    let after_defined = resolve_defined_operators(tokens, macros, interner, diag);

    // ── Phase 2: expand object-like macros ──────────────────────────────
    let after_expand = expand_macros_in_expr(&after_defined, macros, interner);

    // ── Phase 3: replace surviving identifiers with 0 ───────────────────
    let final_tokens = replace_identifiers_with_zero(&after_expand);

    // ── Phase 4: parse and evaluate ─────────────────────────────────────
    let mut parser = ExprParser::new(&final_tokens, diag);
    let result = parser.parse_ternary(false);

    // If there are unconsumed tokens (besides Eof), warn but still succeed.
    if !parser.at_end() {
        let span = parser.peek_span();
        diag.warning(span, "extra tokens after end of preprocessor expression");
    }

    Ok(result.val)
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1 — resolve `defined` operators
// ═══════════════════════════════════════════════════════════════════════════

/// Process `defined(NAME)` and `defined NAME` operators, replacing them
/// with integer literal tokens `1` (defined) or `0` (not defined).
///
/// This MUST occur before macro expansion so that the operand of `defined`
/// is never expanded (C11 §6.10.1p4).
fn resolve_defined_operators(
    tokens: &[Token],
    macros: &FxHashMap<Symbol, MacroDef>,
    interner: &mut Interner,
    diag: &mut DiagnosticEngine,
) -> Vec<Token> {
    let defined_str = "defined";
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        // Check for `defined` keyword (it appears as an Identifier token).
        if let TokenKind::Identifier(sym) = &tokens[i].kind {
            let name = interner.resolve(*sym);
            if name == defined_str {
                let span = tokens[i].span;
                i += 1;
                // Skip whitespace tokens if present.
                while i < tokens.len() && matches!(tokens[i].kind, TokenKind::Whitespace) {
                    i += 1;
                }
                // Two forms: `defined(IDENT)` or `defined IDENT`.
                if i < tokens.len() && matches!(tokens[i].kind, TokenKind::LeftParen) {
                    // `defined ( IDENT )`
                    i += 1; // skip '('
                    // Skip whitespace.
                    while i < tokens.len() && matches!(tokens[i].kind, TokenKind::Whitespace) {
                        i += 1;
                    }
                    if i < tokens.len() {
                        if let TokenKind::Identifier(macro_sym) = &tokens[i].kind {
                            let is_def = macros.contains_key(macro_sym);
                            result.push(make_int_token(if is_def { 1 } else { 0 }, span));
                            i += 1; // skip identifier
                            // Skip whitespace.
                            while i < tokens.len()
                                && matches!(tokens[i].kind, TokenKind::Whitespace)
                            {
                                i += 1;
                            }
                            // Expect ')'.
                            if i < tokens.len()
                                && matches!(tokens[i].kind, TokenKind::RightParen)
                            {
                                i += 1; // skip ')'
                            } else {
                                diag.error(
                                    span,
                                    "missing ')' after \"defined\" operator",
                                );
                            }
                        } else {
                            diag.error(
                                tokens[i].span,
                                "expected identifier after \"defined(\"",
                            );
                            result.push(make_int_token(0, span));
                            i += 1;
                        }
                    } else {
                        diag.error(span, "expected identifier after \"defined(\"");
                        result.push(make_int_token(0, span));
                    }
                } else if i < tokens.len() {
                    // `defined IDENT` (no parentheses).
                    if let TokenKind::Identifier(macro_sym) = &tokens[i].kind {
                        let is_def = macros.contains_key(macro_sym);
                        result.push(make_int_token(if is_def { 1 } else { 0 }, span));
                        i += 1; // skip identifier
                    } else {
                        diag.error(
                            tokens[i].span,
                            "expected identifier after \"defined\"",
                        );
                        result.push(make_int_token(0, span));
                    }
                } else {
                    diag.error(span, "expected identifier after \"defined\"");
                    result.push(make_int_token(0, span));
                }
                continue;
            }
        }
        // Not `defined` — pass through.
        result.push(tokens[i].clone());
        i += 1;
    }
    result
}

/// Helper to create an `IntegerLiteral` token with a given value and span.
#[inline]
fn make_int_token(val: u128, span: Span) -> Token {
    Token::new(
        TokenKind::IntegerLiteral {
            value: val,
            suffix: IntegerSuffix::None,
        },
        span,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 — simplified macro expansion for #if expressions
// ═══════════════════════════════════════════════════════════════════════════

/// Expand object-like macros in the token stream.
///
/// This is a simplified expansion that handles object-like macros only.
/// Function-like macros are left unexpanded (their name becomes an
/// identifier that will later be replaced with `0`). Expansion is
/// iterative with a recursion guard to prevent infinite loops on
/// self-referential macros (e.g. `#define A A`).
fn expand_macros_in_expr(
    tokens: &[Token],
    macros: &FxHashMap<Symbol, MacroDef>,
    _interner: &mut Interner,
) -> Vec<Token> {
    let max_iterations = 256;
    let mut current = tokens.to_vec();
    for _ in 0..max_iterations {
        let mut changed = false;
        let mut next = Vec::with_capacity(current.len());
        for tok in &current {
            if let TokenKind::Identifier(sym) = &tok.kind {
                if let Some(def) = macros.get(sym) {
                    // Only expand object-like macros (params == None) with a
                    // non-empty body. Skip predefined macros that require
                    // special context (they will be resolved to 0 later).
                    if def.params.is_none() && !def.body.is_empty() && !def.is_predefined {
                        // Recursion guard: do not expand a macro to itself.
                        // This catches `#define A A` and similar trivial
                        // self-references.
                        let is_self_ref = def.body.len() == 1
                            && matches!(&def.body[0].kind, TokenKind::Identifier(s) if s == sym);
                        if !is_self_ref {
                            for body_tok in &def.body {
                                next.push(Token::new(body_tok.kind.clone(), tok.span));
                            }
                            changed = true;
                            continue;
                        }
                    }
                }
            }
            next.push(tok.clone());
        }
        current = next;
        if !changed {
            break;
        }
    }
    current
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 3 — replace surviving identifiers with zero
// ═══════════════════════════════════════════════════════════════════════════

/// After `defined` resolution and macro expansion, any remaining identifier
/// tokens represent undefined macros. Per C11 §6.10.1p4 they evaluate to `0`
/// with no diagnostic emitted.
fn replace_identifiers_with_zero(tokens: &[Token]) -> Vec<Token> {
    tokens
        .iter()
        .filter(|t| !matches!(t.kind, TokenKind::Whitespace | TokenKind::Newline))
        .map(|tok| match &tok.kind {
            TokenKind::Identifier(_) => make_int_token(0, tok.span),
            _ => tok.clone(),
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// ExprParser — precedence-climbing expression parser
// ═══════════════════════════════════════════════════════════════════════════

/// A cursor-based parser that consumes preprocessed tokens and evaluates
/// the preprocessor constant expression via precedence climbing.
///
/// The `suppress` flag is threaded through evaluation to implement
/// short-circuit semantics: when `suppress` is `true`, we still parse
/// the tokens (to advance the cursor correctly) but do not emit
/// diagnostics for runtime errors like division-by-zero.
struct ExprParser<'a> {
    /// The token sequence to parse (whitespace already stripped).
    tokens: &'a [Token],
    /// Current position in `tokens`.
    pos: usize,
    /// Diagnostic engine for error/warning reporting.
    diag: &'a mut DiagnosticEngine,
}

impl<'a> ExprParser<'a> {
    /// Construct a new parser over the given token slice.
    fn new(tokens: &'a [Token], diag: &'a mut DiagnosticEngine) -> Self {
        ExprParser {
            tokens,
            pos: 0,
            diag,
        }
    }

    // ── Token access helpers ────────────────────────────────────────────

    /// Peek at the current token kind without consuming it.
    #[inline]
    fn peek(&self) -> &TokenKind {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos].kind
        } else {
            &TokenKind::Eof
        }
    }

    /// Return the span of the current token.
    #[inline]
    fn peek_span(&self) -> Span {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].span
        } else {
            Span::DUMMY
        }
    }

    /// Advance the cursor by one token.
    #[inline]
    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    /// Consume the current token if it matches `kind`, returning `true`.
    /// Otherwise leave the cursor untouched and return `false`.
    #[inline]
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Check whether we have consumed all meaningful tokens.
    #[inline]
    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.peek(), TokenKind::Eof)
    }

    // ── Precedence levels ───────────────────────────────────────────────
    // We use explicit functions for each precedence level rather than a
    // table-driven approach because the ternary operator and short-circuit
    // operators require special control flow.

    /// Entry point: parse a ternary conditional expression (lowest precedence).
    ///
    /// ```text
    /// conditional = logical_or ( '?' expression ':' conditional )?
    /// ```
    ///
    /// The ternary is right-associative: `a ? b : c ? d : e` groups as
    /// `a ? b : (c ? d : e)`.
    fn parse_ternary(&mut self, suppress: bool) -> PPValue {
        let cond = self.parse_logical_or(suppress);
        if self.eat(&TokenKind::Question) {
            // True branch — suppress only if the condition is false OR outer suppress.
            let true_val = self.parse_ternary(suppress || !cond.is_true());
            if !self.eat(&TokenKind::Colon) {
                if !suppress {
                    let span = self.peek_span();
                    self.diag.error(span, "expected ':' in ternary expression");
                }
                return PPValue::zero();
            }
            // False branch — suppress only if the condition is true OR outer suppress.
            let false_val = self.parse_ternary(suppress || cond.is_true());
            if suppress {
                return PPValue::zero();
            }
            if cond.is_true() {
                true_val
            } else {
                false_val
            }
        } else {
            cond
        }
    }

    /// `logical_or = logical_and ( '||' logical_and )*`
    ///
    /// Short-circuits: once the left operand is non-zero, remaining
    /// operands are parsed but not evaluated.
    fn parse_logical_or(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_logical_and(suppress);
        while self.eat(&TokenKind::PipePipe) {
            // If left is already true, suppress evaluation of the right side.
            let rhs_suppress = suppress || left.is_true();
            let right = self.parse_logical_and(rhs_suppress);
            if !suppress {
                left = PPValue::signed(if left.is_true() || right.is_true() {
                    1
                } else {
                    0
                });
            }
        }
        left
    }

    /// `logical_and = bitwise_or ( '&&' bitwise_or )*`
    ///
    /// Short-circuits: once the left operand is zero, remaining operands
    /// are parsed but not evaluated.
    fn parse_logical_and(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_bitwise_or(suppress);
        while self.eat(&TokenKind::AmpAmp) {
            // If left is false, suppress evaluation of the right side.
            let rhs_suppress = suppress || !left.is_true();
            let right = self.parse_bitwise_or(rhs_suppress);
            if !suppress {
                left = PPValue::signed(if left.is_true() && right.is_true() {
                    1
                } else {
                    0
                });
            }
        }
        left
    }

    /// `bitwise_or = bitwise_xor ( '|' bitwise_xor )*`
    fn parse_bitwise_or(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_bitwise_xor(suppress);
        while self.eat(&TokenKind::Pipe) {
            let right = self.parse_bitwise_xor(suppress);
            if !suppress {
                left = apply_binary(left, right, BinOp::BitOr);
            }
        }
        left
    }

    /// `bitwise_xor = bitwise_and ( '^' bitwise_and )*`
    fn parse_bitwise_xor(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_bitwise_and(suppress);
        while self.eat(&TokenKind::Caret) {
            let right = self.parse_bitwise_and(suppress);
            if !suppress {
                left = apply_binary(left, right, BinOp::BitXor);
            }
        }
        left
    }

    /// `bitwise_and = equality ( '&' equality )*`
    fn parse_bitwise_and(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_equality(suppress);
        while self.eat(&TokenKind::Ampersand) {
            let right = self.parse_equality(suppress);
            if !suppress {
                left = apply_binary(left, right, BinOp::BitAnd);
            }
        }
        left
    }

    /// `equality = relational ( ('==' | '!=') relational )*`
    fn parse_equality(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_relational(suppress);
        loop {
            let op = match self.peek() {
                TokenKind::EqualEqual => BinOp::Eq,
                TokenKind::NotEqual => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let right = self.parse_relational(suppress);
            if !suppress {
                left = apply_comparison(left, right, op);
            }
        }
        left
    }

    /// `relational = shift ( ('<' | '>' | '<=' | '>=') shift )*`
    fn parse_relational(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_shift(suppress);
        loop {
            let op = match self.peek() {
                TokenKind::Less => BinOp::Lt,
                TokenKind::Greater => BinOp::Gt,
                TokenKind::LessEqual => BinOp::Le,
                TokenKind::GreaterEqual => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_shift(suppress);
            if !suppress {
                left = apply_comparison(left, right, op);
            }
        }
        left
    }

    /// `shift = additive ( ('<<' | '>>') additive )*`
    fn parse_shift(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_additive(suppress);
        loop {
            let op = match self.peek() {
                TokenKind::LeftShift => BinOp::Shl,
                TokenKind::RightShift => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive(suppress);
            if !suppress {
                left = apply_shift(left, right, op, self.diag, self.peek_span());
            }
        }
        left
    }

    /// `additive = multiplicative ( ('+' | '-') multiplicative )*`
    fn parse_additive(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_multiplicative(suppress);
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative(suppress);
            if !suppress {
                left = apply_additive(left, right, op, self.diag, self.peek_span());
            }
        }
        left
    }

    /// `multiplicative = unary ( ('*' | '/' | '%') unary )*`
    fn parse_multiplicative(&mut self, suppress: bool) -> PPValue {
        let mut left = self.parse_unary(suppress);
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            let op_span = self.peek_span();
            self.advance();
            let right = self.parse_unary(suppress);
            if !suppress {
                left = apply_multiplicative(left, right, op, self.diag, op_span);
            }
        }
        left
    }

    /// `unary = ('+' | '-' | '~' | '!') unary | primary`
    fn parse_unary(&mut self, suppress: bool) -> PPValue {
        match self.peek().clone() {
            TokenKind::Plus => {
                self.advance();
                self.parse_unary(suppress) // unary + is identity
            }
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary(suppress);
                if suppress {
                    return PPValue::zero();
                }
                // Wrapping negate — for unsigned, this is two's-complement.
                PPValue {
                    val: operand.val.wrapping_neg(),
                    is_unsigned: operand.is_unsigned,
                }
            }
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_unary(suppress);
                if suppress {
                    return PPValue::zero();
                }
                PPValue {
                    val: !operand.val,
                    is_unsigned: operand.is_unsigned,
                }
            }
            TokenKind::Exclaim => {
                self.advance();
                let operand = self.parse_unary(suppress);
                if suppress {
                    return PPValue::zero();
                }
                PPValue::signed(if operand.val == 0 { 1 } else { 0 })
            }
            _ => self.parse_primary(suppress),
        }
    }

    /// `primary = integer_literal | char_literal | '(' expression ')' | error`
    ///
    /// By this point all identifiers have been replaced with `0` and all
    /// `defined` operators have been resolved, so the only valid primary
    /// expressions are literals and parenthesised sub-expressions.
    fn parse_primary(&mut self, suppress: bool) -> PPValue {
        match self.peek().clone() {
            TokenKind::IntegerLiteral { value, suffix } => {
                self.advance();
                if suppress {
                    return PPValue::zero();
                }
                let raw = value as i64;
                // Determine signedness from suffix per C11 §6.4.4.1:
                // U/UL/ULL → unsigned; None/L/LL → signed.
                match suffix {
                    IntegerSuffix::U | IntegerSuffix::UL | IntegerSuffix::ULL => {
                        PPValue::unsigned(raw)
                    }
                    IntegerSuffix::None | IntegerSuffix::L | IntegerSuffix::LL => {
                        PPValue::signed(raw)
                    }
                }
            }
            TokenKind::CharLiteral { value, prefix } => {
                self.advance();
                if suppress {
                    return PPValue::zero();
                }
                // Character constants have type `int` (signed) in preprocessor
                // expressions for CharPrefix::None (ordinary chars). Wide and
                // unicode prefixes (L, u, U) also yield int-width values in
                // preprocessor context. We use the prefix to determine the
                // range of the value but all map to signed results.
                let char_val = match prefix {
                    CharPrefix::None => value as i64,
                    // Wide/unicode prefixes — the stored u32 is the code point,
                    // which maps directly to an i64 value.
                    _ => value as i64,
                };
                PPValue::signed(char_val)
            }
            TokenKind::LeftParen => {
                self.advance();
                let inner = self.parse_ternary(suppress);
                if !self.eat(&TokenKind::RightParen) && !suppress {
                    let span = self.peek_span();
                    self.diag
                        .error(span, "expected ')' in preprocessor expression");
                }
                inner
            }
            TokenKind::Eof => {
                if !suppress {
                    self.diag.error(
                        self.peek_span(),
                        "unexpected end of expression in preprocessor directive",
                    );
                }
                PPValue::zero()
            }
            _ => {
                if !suppress {
                    let span = self.peek_span();
                    self.diag.error(
                        span,
                        "unexpected token in preprocessor expression",
                    );
                }
                // Skip the offending token to attempt recovery.
                self.advance();
                PPValue::zero()
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Binary operation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Internal enumeration of binary operators supported in preprocessor
/// constant expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    // Shift
    Shl,
    Shr,
    // Comparison
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// Apply "usual arithmetic conversions" for preprocessor values.
/// If either operand is unsigned, both are treated as unsigned.
#[inline]
fn promote(a: PPValue, b: PPValue) -> (PPValue, PPValue, bool) {
    let unsigned = a.is_unsigned || b.is_unsigned;
    (
        PPValue {
            val: a.val,
            is_unsigned: unsigned,
        },
        PPValue {
            val: b.val,
            is_unsigned: unsigned,
        },
        unsigned,
    )
}

/// Evaluate a bitwise binary operation (|, ^, &).
#[inline]
fn apply_binary(a: PPValue, b: PPValue, op: BinOp) -> PPValue {
    let (a, b, unsigned) = promote(a, b);
    let val = match op {
        BinOp::BitOr => a.val | b.val,
        BinOp::BitXor => a.val ^ b.val,
        BinOp::BitAnd => a.val & b.val,
        _ => unreachable!(),
    };
    PPValue {
        val,
        is_unsigned: unsigned,
    }
}

/// Evaluate a comparison operation, returning a signed `1` or `0`.
///
/// When both operands are unsigned, comparison uses `u64` semantics.
#[inline]
fn apply_comparison(a: PPValue, b: PPValue, op: BinOp) -> PPValue {
    let (a, b, unsigned) = promote(a, b);
    let result = if unsigned {
        let ua = a.val as u64;
        let ub = b.val as u64;
        match op {
            BinOp::Eq => ua == ub,
            BinOp::Ne => ua != ub,
            BinOp::Lt => ua < ub,
            BinOp::Gt => ua > ub,
            BinOp::Le => ua <= ub,
            BinOp::Ge => ua >= ub,
            _ => unreachable!(),
        }
    } else {
        match op {
            BinOp::Eq => a.val == b.val,
            BinOp::Ne => a.val != b.val,
            BinOp::Lt => a.val < b.val,
            BinOp::Gt => a.val > b.val,
            BinOp::Le => a.val <= b.val,
            BinOp::Ge => a.val >= b.val,
            _ => unreachable!(),
        }
    };
    PPValue::signed(if result { 1 } else { 0 })
}

/// Evaluate a shift operation.
///
/// Negative or oversized shift amounts produce a warning. For unsigned
/// values, left-shift wraps modulo 64. For signed values, left-shift of
/// a negative number is undefined but we wrap to match GCC behaviour.
fn apply_shift(
    a: PPValue,
    b: PPValue,
    op: BinOp,
    diag: &mut DiagnosticEngine,
    span: Span,
) -> PPValue {
    let (a, _b, unsigned) = promote(a, b);
    let shift = b.val;
    // Validate shift amount.
    if !(0..64).contains(&shift) {
        diag.warning(span, "shift amount out of range for preprocessor expression");
        return PPValue {
            val: 0,
            is_unsigned: unsigned,
        };
    }
    let shift = shift as u32;
    let val = if unsigned {
        let ua = a.val as u64;
        match op {
            BinOp::Shl => (ua.wrapping_shl(shift)) as i64,
            BinOp::Shr => (ua >> shift) as i64,
            _ => unreachable!(),
        }
    } else {
        match op {
            BinOp::Shl => a.val.wrapping_shl(shift),
            // Arithmetic right shift for signed values (sign-extending).
            BinOp::Shr => a.val >> shift,
            _ => unreachable!(),
        }
    };
    PPValue {
        val,
        is_unsigned: unsigned,
    }
}

/// Evaluate an additive operation (+, -) with overflow warning.
fn apply_additive(
    a: PPValue,
    b: PPValue,
    op: BinOp,
    diag: &mut DiagnosticEngine,
    span: Span,
) -> PPValue {
    let (a, b, unsigned) = promote(a, b);
    let (val, overflowed) = match op {
        BinOp::Add => {
            if unsigned {
                let (r, ov) = (a.val as u64).overflowing_add(b.val as u64);
                (r as i64, ov)
            } else {
                a.val.overflowing_add(b.val)
            }
        }
        BinOp::Sub => {
            if unsigned {
                let (r, ov) = (a.val as u64).overflowing_sub(b.val as u64);
                (r as i64, ov)
            } else {
                a.val.overflowing_sub(b.val)
            }
        }
        _ => unreachable!(),
    };
    if overflowed && !unsigned {
        diag.warning(span, "integer overflow in preprocessor expression");
    }
    PPValue {
        val,
        is_unsigned: unsigned,
    }
}

/// Evaluate a multiplicative operation (*, /, %) with division-by-zero
/// error and overflow warning.
fn apply_multiplicative(
    a: PPValue,
    b: PPValue,
    op: BinOp,
    diag: &mut DiagnosticEngine,
    span: Span,
) -> PPValue {
    let (a, b, unsigned) = promote(a, b);
    match op {
        BinOp::Mul => {
            let (val, overflowed) = if unsigned {
                let (r, ov) = (a.val as u64).overflowing_mul(b.val as u64);
                (r as i64, ov)
            } else {
                a.val.overflowing_mul(b.val)
            };
            if overflowed && !unsigned {
                diag.warning(span, "integer overflow in preprocessor expression");
            }
            PPValue {
                val,
                is_unsigned: unsigned,
            }
        }
        BinOp::Div => {
            if b.val == 0 {
                diag.error(span, "division by zero in preprocessor expression");
                return PPValue {
                    val: 0,
                    is_unsigned: unsigned,
                };
            }
            let val = if unsigned {
                ((a.val as u64) / (b.val as u64)) as i64
            } else {
                // Handle i64::MIN / -1 overflow (undefined in C, wraps here).
                a.val.wrapping_div(b.val)
            };
            PPValue {
                val,
                is_unsigned: unsigned,
            }
        }
        BinOp::Mod => {
            if b.val == 0 {
                diag.error(span, "division by zero in preprocessor expression");
                return PPValue {
                    val: 0,
                    is_unsigned: unsigned,
                };
            }
            let val = if unsigned {
                ((a.val as u64) % (b.val as u64)) as i64
            } else {
                a.val.wrapping_rem(b.val)
            };
            PPValue {
                val,
                is_unsigned: unsigned,
            }
        }
        _ => unreachable!(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::fx_hash::FxHashMap;
    use crate::common::string_interner::Interner;
    use crate::frontend::lexer::token::{IntegerSuffix, Token, TokenKind};

    /// Helper: build tokens from a sequence of TokenKind values.
    fn make_tokens(kinds: &[TokenKind]) -> Vec<Token> {
        kinds
            .iter()
            .map(|k| Token::new(k.clone(), Span::DUMMY))
            .collect()
    }

    /// Helper: evaluate a token sequence and return the i64 result.
    fn eval(kinds: &[TokenKind]) -> Result<i64, ()> {
        let tokens = make_tokens(kinds);
        let macros: FxHashMap<Symbol, MacroDef> = FxHashMap::default();
        let mut interner = Interner::new();
        let mut diag = DiagnosticEngine::new();
        evaluate_expression(&tokens, &macros, &mut interner, &mut diag)
    }

    #[test]
    fn test_simple_integer() {
        let result = eval(&[TokenKind::IntegerLiteral {
            value: 42,
            suffix: IntegerSuffix::None,
        }]);
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_zero_is_false() {
        let result = eval(&[TokenKind::IntegerLiteral {
            value: 0,
            suffix: IntegerSuffix::None,
        }]);
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn test_addition() {
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Plus,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(5));
    }

    #[test]
    fn test_subtraction() {
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 10,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Minus,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(7));
    }

    #[test]
    fn test_multiplication() {
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 4,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Star,
            TokenKind::IntegerLiteral {
                value: 5,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(20));
    }

    #[test]
    fn test_division() {
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 15,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Slash,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(5));
    }

    #[test]
    fn test_modulo() {
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 17,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Percent,
            TokenKind::IntegerLiteral {
                value: 5,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(2));
    }

    #[test]
    fn test_precedence_mul_add() {
        // 2 + 3 * 4 == 14  (not 20)
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Plus,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Star,
            TokenKind::IntegerLiteral {
                value: 4,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(14));
    }

    #[test]
    fn test_parenthesised() {
        // (2 + 3) * 4 == 20
        let result = eval(&[
            TokenKind::LeftParen,
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Plus,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
            TokenKind::RightParen,
            TokenKind::Star,
            TokenKind::IntegerLiteral {
                value: 4,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(20));
    }

    #[test]
    fn test_logical_and() {
        // 1 && 1 == 1
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::AmpAmp,
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(1));

        // 1 && 0 == 0
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::AmpAmp,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(0));
    }

    #[test]
    fn test_logical_or() {
        // 0 || 1 == 1
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::PipePipe,
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(1));

        // 0 || 0 == 0
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::PipePipe,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(0));
    }

    #[test]
    fn test_logical_not() {
        // !0 == 1
        let result = eval(&[
            TokenKind::Exclaim,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(1));

        // !5 == 0
        let result2 = eval(&[
            TokenKind::Exclaim,
            TokenKind::IntegerLiteral {
                value: 5,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(0));
    }

    #[test]
    fn test_bitwise_ops() {
        // 0xFF & 0x0F == 0x0F == 15
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 0xFF,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Ampersand,
            TokenKind::IntegerLiteral {
                value: 0x0F,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(15));

        // 0xF0 | 0x0F == 0xFF == 255
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 0xF0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Pipe,
            TokenKind::IntegerLiteral {
                value: 0x0F,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(255));

        // 0xFF ^ 0x0F == 0xF0 == 240
        let result3 = eval(&[
            TokenKind::IntegerLiteral {
                value: 0xFF,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Caret,
            TokenKind::IntegerLiteral {
                value: 0x0F,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result3, Ok(240));
    }

    #[test]
    fn test_shift() {
        // 1 << 4 == 16
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::LeftShift,
            TokenKind::IntegerLiteral {
                value: 4,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(16));

        // 32 >> 2 == 8
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 32,
                suffix: IntegerSuffix::None,
            },
            TokenKind::RightShift,
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(8));
    }

    #[test]
    fn test_comparison() {
        // 5 > 3 == 1
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 5,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Greater,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(1));

        // 3 >= 3 == 1
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
            TokenKind::GreaterEqual,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(1));

        // 2 == 2 == 1
        let result3 = eval(&[
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
            TokenKind::EqualEqual,
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result3, Ok(1));

        // 2 != 3 == 1
        let result4 = eval(&[
            TokenKind::IntegerLiteral {
                value: 2,
                suffix: IntegerSuffix::None,
            },
            TokenKind::NotEqual,
            TokenKind::IntegerLiteral {
                value: 3,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result4, Ok(1));
    }

    #[test]
    fn test_ternary() {
        // 1 ? 42 : 99 == 42
        let result = eval(&[
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Question,
            TokenKind::IntegerLiteral {
                value: 42,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Colon,
            TokenKind::IntegerLiteral {
                value: 99,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(42));

        // 0 ? 42 : 99 == 99
        let result2 = eval(&[
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Question,
            TokenKind::IntegerLiteral {
                value: 42,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Colon,
            TokenKind::IntegerLiteral {
                value: 99,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result2, Ok(99));
    }

    #[test]
    fn test_short_circuit_div_by_zero_suppressed() {
        // 0 && (1 / 0) should NOT produce an error because the right
        // operand is short-circuited.
        let tokens = make_tokens(&[
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::AmpAmp,
            TokenKind::LeftParen,
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Slash,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
            TokenKind::RightParen,
        ]);
        let macros: FxHashMap<Symbol, MacroDef> = FxHashMap::default();
        let mut interner = Interner::new();
        let mut diag = DiagnosticEngine::new();
        let result = evaluate_expression(&tokens, &macros, &mut interner, &mut diag);
        assert_eq!(result, Ok(0));
        assert!(!diag.has_errors());
    }

    #[test]
    fn test_unary_minus() {
        // -5 == -5
        let result = eval(&[
            TokenKind::Minus,
            TokenKind::IntegerLiteral {
                value: 5,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(-5));
    }

    #[test]
    fn test_bitwise_not() {
        // ~0 == -1 (all bits set in two's complement)
        let result = eval(&[
            TokenKind::Tilde,
            TokenKind::IntegerLiteral {
                value: 0,
                suffix: IntegerSuffix::None,
            },
        ]);
        assert_eq!(result, Ok(-1));
    }

    #[test]
    fn test_empty_expression_is_error() {
        let tokens: Vec<Token> = vec![];
        let macros: FxHashMap<Symbol, MacroDef> = FxHashMap::default();
        let mut interner = Interner::new();
        let mut diag = DiagnosticEngine::new();
        let result = evaluate_expression(&tokens, &macros, &mut interner, &mut diag);
        assert!(result.is_err());
    }

    #[test]
    fn test_defined_operator() {
        let mut interner = Interner::new();
        let mut diag = DiagnosticEngine::new();
        let mut macros: FxHashMap<Symbol, MacroDef> = FxHashMap::default();
        let foo_sym = interner.intern("FOO");
        macros.insert(
            foo_sym,
            MacroDef {
                name: foo_sym,
                params: None,
                is_variadic: false,
                body: vec![Token::new(
                    TokenKind::IntegerLiteral {
                        value: 1,
                        suffix: IntegerSuffix::None,
                    },
                    Span::DUMMY,
                )],
                is_predefined: false,
                source_span: Span::DUMMY,
            },
        );

        let defined_sym = interner.intern("defined");

        // `defined(FOO)` should be 1.
        let tokens = vec![
            Token::new(TokenKind::Identifier(defined_sym), Span::DUMMY),
            Token::new(TokenKind::LeftParen, Span::DUMMY),
            Token::new(TokenKind::Identifier(foo_sym), Span::DUMMY),
            Token::new(TokenKind::RightParen, Span::DUMMY),
        ];
        let result = evaluate_expression(&tokens, &macros, &mut interner, &mut diag);
        assert_eq!(result, Ok(1));

        // `defined BAR` (not defined) should be 0.
        let bar_sym = interner.intern("BAR");
        let tokens2 = vec![
            Token::new(TokenKind::Identifier(defined_sym), Span::DUMMY),
            Token::new(TokenKind::Identifier(bar_sym), Span::DUMMY),
        ];
        let result2 = evaluate_expression(&tokens2, &macros, &mut interner, &mut diag);
        assert_eq!(result2, Ok(0));
    }

    #[test]
    fn test_char_literal() {
        let result = eval(&[TokenKind::CharLiteral {
            value: b'A' as u32,
            prefix: CharPrefix::None,
        }]);
        assert_eq!(result, Ok(65)); // ASCII 'A' = 65
    }

    #[test]
    fn test_unsigned_comparison() {
        // When one operand is unsigned, comparison uses unsigned semantics.
        // -1 (signed) vs 1U (unsigned): as unsigned, -1 becomes a large number > 1.
        let result = eval(&[
            TokenKind::Minus,
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::None,
            },
            TokenKind::Greater,
            TokenKind::IntegerLiteral {
                value: 1,
                suffix: IntegerSuffix::U,
            },
        ]);
        // -1 as u64 is u64::MAX which is > 1, so result is 1.
        assert_eq!(result, Ok(1));
    }
}
