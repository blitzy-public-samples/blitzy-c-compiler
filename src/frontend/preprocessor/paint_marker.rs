//! Token-level paint marker implementation for preprocessor macro expansion
//! recursion protection.
//!
//! This module implements the C standard's rule (C11 §6.10.3.4) that prevents
//! self-referential macro expansion from entering infinite recursion. When a
//! macro is expanded, its name is "painted" onto all tokens in the replacement
//! list; during subsequent rescanning, any painted token that matches the
//! painted macro name is treated as an ordinary identifier and is **not**
//! re-expanded.
//!
//! # Architecture
//!
//! Paint markers operate at the **token level** during Phase 2 (macro expansion)
//! of the preprocessor. This is architecturally distinct from two other
//! recursion-prevention mechanisms in BCC:
//!
//! - **Circular `#include` detection** — operates at the file/include level
//!   during `#include` directive processing.
//! - **512-depth recursion limit** — a global safety net that prevents deeply
//!   nested (but non-self-referential) macro chains from exhausting the stack.
//!
//! Paint markers specifically handle the self-referential case where a macro's
//! replacement list contains its own name (directly or through a chain of other
//! macros).
//!
//! # Correctness Examples
//!
//! ```text
//! #define A A
//! // Expand A: body is [A], paint A for macro 'A' → [A(painted:A)]
//! // Rescan: A is painted for 'A' → no expansion → result: A
//! // Terminates correctly in O(1).
//!
//! #define A B
//! #define B A
//! // Expand A: body [B], paint for 'A' → [B(painted:A)]
//! // Rescan: B is a macro → expand B: body [A], paint for 'B'
//! //   → [A(painted:A,B)]
//! // Rescan: A is painted for 'A' → no expansion → result: A
//!
//! #define F(x) x
//! // F(F(1)): inner F(1) pre-expanded to 1, outer F substitutes 1 → result: 1
//! // Paint does not interfere because inner expansion completes fully.
//! ```
//!
//! # Performance
//!
//! Most tokens have 0 or 1 paint entries. The `PaintState` enum is optimised
//! for this common case with a `Single(Symbol)` variant that avoids allocating
//! an `FxHashSet`. The `Multi` variant is used only when a token accumulates
//! paint from two or more distinct macros (rare in practice).
//!
//! # Zero-Dependency Implementation
//!
//! This module uses only the internal [`FxHashSet`] (from `crate::common::fx_hash`)
//! and [`Symbol`] (from `crate::common::string_interner`). No external crates
//! are required.

use crate::common::fx_hash::FxHashSet;
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::Token;

// ===========================================================================
// PaintState — tracks which macros have painted a token
// ===========================================================================

/// Represents the paint state of a preprocessor token.
///
/// A token's paint state records which macro expansions have "claimed" it.
/// During rescanning after macro expansion, if a token is an identifier that
/// matches a macro name AND that macro name is present in the token's paint
/// set, the token is treated as an ordinary identifier — the macro is **not**
/// re-expanded. This implements C11 §6.10.3.4.
///
/// # Variants
///
/// - [`Unpainted`](PaintState::Unpainted) — The token has never been produced
///   by macro expansion (or all paint has been explicitly removed). It is
///   eligible for expansion as any macro.
///
/// - [`Painted`](PaintState::Painted) — The token was produced during the
///   expansion of one or more macros. The inner representation tracks which
///   macro names are suppressed. Internally optimised into `Single` (one macro)
///   and `Multi` (two or more macros) sub-variants to avoid heap allocation in
///   the common single-paint case.
///
/// # Examples
///
/// ```text
/// // Unpainted token — eligible for any expansion:
/// PaintState::Unpainted
///
/// // Painted for a single macro 'A':
/// PaintState::Painted  (internally: Single(sym_a))
///
/// // Painted for two macros 'A' and 'B':
/// PaintState::Painted  (internally: Multi({sym_a, sym_b}))
/// ```
#[derive(Clone, Debug)]
pub enum PaintState {
    /// The token has not been produced by any macro expansion and is eligible
    /// for expansion as any macro.
    Unpainted,

    /// The token was produced during expansion of one or more macros. The
    /// inner [`PaintSet`] records which macro names are suppressed for
    /// re-expansion.
    Painted(PaintSet),
}

// ===========================================================================
// PaintSet — optimised storage for paint entries
// ===========================================================================

/// Internal storage for paint entries, optimised for the common case.
///
/// Most tokens in real-world C code are painted by at most one macro during
/// expansion. The `Single` variant stores exactly one `Symbol` inline without
/// any heap allocation. When a second distinct macro paints the same token,
/// the representation promotes to `Multi` with an `FxHashSet`.
///
/// This type is an implementation detail of [`PaintState`] and is not intended
/// for direct use outside this module. External code interacts exclusively
/// through the public functions [`paint_tokens`], [`is_painted_for`],
/// [`is_painted`], [`merge_paint`], [`unpaint`], and [`unpaint_for`].
#[derive(Clone, Debug)]
pub enum PaintSet {
    /// Exactly one macro has painted this token. Stored inline without heap
    /// allocation. This is the overwhelmingly common case.
    Single(Symbol),

    /// Two or more macros have painted this token. Uses an [`FxHashSet`] for
    /// efficient membership testing and union operations.
    Multi(FxHashSet<Symbol>),
}

impl PaintSet {
    /// Returns `true` if the given macro name is present in this paint set.
    #[inline]
    fn contains(&self, macro_name: Symbol) -> bool {
        match self {
            PaintSet::Single(sym) => *sym == macro_name,
            PaintSet::Multi(set) => set.contains(&macro_name),
        }
    }

    /// Returns `true` if this paint set is empty.
    ///
    /// A `PaintSet` should never actually be empty in normal operation (it is
    /// always created with at least one entry), but this method is provided
    /// for defensive programming after removal operations.
    #[inline]
    fn is_empty(&self) -> bool {
        match self {
            PaintSet::Single(_) => false,
            PaintSet::Multi(set) => set.is_empty(),
        }
    }

    /// Inserts a macro name into the paint set.
    ///
    /// If the set is `Single` and the new name is different from the existing
    /// entry, promotes to `Multi`. If the name is already present, this is a
    /// no-op.
    fn insert(&mut self, macro_name: Symbol) {
        match self {
            PaintSet::Single(existing) => {
                if *existing != macro_name {
                    // Promote to Multi: create a set containing both symbols.
                    let mut set = FxHashSet::default();
                    set.insert(*existing);
                    set.insert(macro_name);
                    *self = PaintSet::Multi(set);
                }
                // If existing == macro_name, it's already painted — no-op.
            }
            PaintSet::Multi(set) => {
                set.insert(macro_name);
            }
        }
    }

    /// Removes a macro name from the paint set.
    ///
    /// Returns `true` if the set becomes empty after removal (indicating the
    /// caller should transition the token to `Unpainted`).
    fn remove(&mut self, macro_name: Symbol) -> bool {
        match self {
            PaintSet::Single(existing) => {
                // If this single entry matches, the set becomes empty.
                *existing == macro_name
            }
            PaintSet::Multi(set) => {
                set.remove(&macro_name);
                if set.is_empty() {
                    return true;
                }
                // Optionally demote to Single if only one entry remains.
                if set.len() == 1 {
                    // Extract the sole remaining symbol.
                    let remaining = *set.iter().next().unwrap();
                    *self = PaintSet::Single(remaining);
                }
                false
            }
        }
    }
}

// ===========================================================================
// PaintState — implementation
// ===========================================================================

impl PaintState {
    /// Creates a new `Unpainted` state.
    #[inline]
    pub fn new_unpainted() -> Self {
        PaintState::Unpainted
    }

    /// Creates a new `Painted` state for a single macro name.
    #[inline]
    pub fn new_painted(macro_name: Symbol) -> Self {
        PaintState::Painted(PaintSet::Single(macro_name))
    }
}

impl Default for PaintState {
    /// The default paint state is `Unpainted`.
    #[inline]
    fn default() -> Self {
        PaintState::Unpainted
    }
}

impl PartialEq for PaintState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PaintState::Unpainted, PaintState::Unpainted) => true,
            (PaintState::Painted(a), PaintState::Painted(b)) => {
                // Compare paint sets for equality by checking mutual containment.
                match (a, b) {
                    (PaintSet::Single(sa), PaintSet::Single(sb)) => sa == sb,
                    (PaintSet::Single(s), PaintSet::Multi(set))
                    | (PaintSet::Multi(set), PaintSet::Single(s)) => {
                        set.len() == 1 && set.contains(s)
                    }
                    (PaintSet::Multi(a_set), PaintSet::Multi(b_set)) => {
                        a_set.len() == b_set.len()
                            && a_set.iter().all(|sym| b_set.contains(sym))
                    }
                }
            }
            _ => false,
        }
    }
}

impl Eq for PaintState {}

// ===========================================================================
// PaintedToken — wrapper associating paint state with a Token
// ===========================================================================

/// A token paired with its paint state for preprocessor macro expansion.
///
/// `PaintedToken` wraps the core [`Token`] struct (which contains `TokenKind`
/// and `Span`) with an associated [`PaintState`], without modifying the `Token`
/// type itself. This keeps `Token` clean for use across the entire compiler
/// pipeline — only the preprocessor's macro expansion engine needs paint state.
///
/// # Fields
///
/// - `token` — The underlying lexer token with its kind and source span.
/// - `paint` — The paint state tracking which macros have claimed this token.
///
/// # Construction
///
/// Use [`PaintedToken::new`] to wrap a token with the default `Unpainted` state,
/// or construct directly with a specific paint state:
///
/// ```text
/// let pt = PaintedToken::new(token);           // Unpainted
/// let pt = PaintedToken { token, paint: PaintState::new_painted(sym) }; // Painted
/// ```
#[derive(Clone, Debug)]
pub struct PaintedToken {
    /// The underlying lexer token (kind + source span).
    pub token: Token,

    /// The paint state of this token, tracking which macros have painted it
    /// during expansion. Defaults to `Unpainted` for freshly created tokens.
    pub paint: PaintState,
}

impl PaintedToken {
    /// Creates a new `PaintedToken` wrapping the given `Token` with an
    /// `Unpainted` state.
    ///
    /// This is the standard constructor for tokens entering the preprocessor
    /// before any macro expansion has occurred.
    ///
    /// # Arguments
    ///
    /// * `token` — The lexer token to wrap.
    ///
    /// # Returns
    ///
    /// A `PaintedToken` with `PaintState::Unpainted`.
    #[inline]
    pub fn new(token: Token) -> Self {
        PaintedToken {
            token,
            paint: PaintState::Unpainted,
        }
    }

    /// Creates a new `PaintedToken` with a specific initial paint state.
    ///
    /// Useful when cloning a token with an inherited paint state, or when
    /// constructing a token that is already known to be painted.
    ///
    /// # Arguments
    ///
    /// * `token` — The lexer token to wrap.
    /// * `paint` — The initial paint state.
    #[inline]
    pub fn with_paint(token: Token, paint: PaintState) -> Self {
        PaintedToken { token, paint }
    }

    /// Returns a reference to the underlying `Token`.
    #[inline]
    pub fn token(&self) -> &Token {
        &self.token
    }

    /// Returns a reference to this token's paint state.
    #[inline]
    pub fn paint(&self) -> &PaintState {
        &self.paint
    }
}

// ===========================================================================
// Paint Operations — core functions for the macro expansion engine
// ===========================================================================

/// Paints all tokens in a replacement list with the expanding macro's name.
///
/// This function is called by the macro expander **after** argument substitution
/// but **before** rescanning. It marks every token in the replacement list so
/// that during rescanning, any occurrence of `macro_name` that was produced by
/// this expansion will not be re-expanded.
///
/// # Algorithm
///
/// For each token in the slice:
/// - If currently `Unpainted` → set to `Painted({macro_name})`
/// - If currently `Painted(set)` → add `macro_name` to set: `Painted(set ∪ {macro_name})`
///
/// # Arguments
///
/// * `tokens` — Mutable slice of painted tokens to mark.
/// * `macro_name` — The `Symbol` handle of the macro currently being expanded.
///
/// # Performance
///
/// This function operates in O(n) time where n is the number of tokens in the
/// replacement list. Each individual paint insertion is O(1) amortised for the
/// common single-entry case.
///
/// # Example
///
/// ```text
/// // Given: #define A A
/// // Replacement list for A: [Token("A")]
/// // After paint_tokens(&mut tokens, sym_a):
/// //   tokens[0].paint == Painted({sym_a})
/// ```
pub fn paint_tokens(tokens: &mut [PaintedToken], macro_name: Symbol) {
    for pt in tokens.iter_mut() {
        match &mut pt.paint {
            PaintState::Unpainted => {
                // First paint — create a Single-entry paint set.
                pt.paint = PaintState::Painted(PaintSet::Single(macro_name));
            }
            PaintState::Painted(set) => {
                // Already painted — add this macro to the existing set.
                set.insert(macro_name);
            }
        }
    }
}

/// Checks whether a token is painted for a specific macro name.
///
/// This is the primary query used by the macro expander during rescanning.
/// When scanning the replacement list, if a token is an identifier matching
/// a known macro name, the expander calls this function to determine whether
/// that macro should be expanded or treated as an ordinary identifier.
///
/// # Arguments
///
/// * `token` — The painted token to check.
/// * `macro_name` — The `Symbol` handle of the macro being considered for expansion.
///
/// # Returns
///
/// `true` if the token has been painted for `macro_name` (suppressing expansion),
/// `false` otherwise (allowing expansion to proceed).
///
/// # Example
///
/// ```text
/// // After expanding #define A A:
/// // token "A" is painted for sym_a.
/// is_painted_for(&token, sym_a) == true   // → do NOT expand
/// is_painted_for(&token, sym_b) == false  // → may expand if B is defined
/// ```
#[inline]
pub fn is_painted_for(token: &PaintedToken, macro_name: Symbol) -> bool {
    match &token.paint {
        PaintState::Unpainted => false,
        PaintState::Painted(set) => set.contains(macro_name),
    }
}

/// Checks whether a token has any paint at all.
///
/// Returns `true` if the token was produced by at least one macro expansion
/// and has not been fully unpainted. This is a coarser check than
/// [`is_painted_for`] — it does not specify which macro painted the token.
///
/// # Use Cases
///
/// - Quick pre-filter before more expensive paint-specific checks.
/// - Diagnostic output: reporting whether a token originated from macro expansion.
/// - Debug assertions verifying paint propagation correctness.
///
/// # Arguments
///
/// * `token` — The painted token to check.
///
/// # Returns
///
/// `true` if the token has any paint entries, `false` if `Unpainted`.
#[inline]
pub fn is_painted(token: &PaintedToken) -> bool {
    match &token.paint {
        PaintState::Unpainted => false,
        PaintState::Painted(set) => !set.is_empty(),
    }
}

/// Merges two paint states into a single combined state.
///
/// The result contains the **union** of both input paint sets. This function
/// is called by the `##` (token pasting) operator: when two tokens are
/// concatenated, the resulting token inherits paint from both operands. If
/// either operand was painted for a given macro, the pasted result is also
/// painted for that macro.
///
/// # Arguments
///
/// * `a` — Paint state of the left operand.
/// * `b` — Paint state of the right operand.
///
/// # Returns
///
/// A new `PaintState` representing the union of `a` and `b`.
///
/// # Examples
///
/// ```text
/// merge_paint(Unpainted, Unpainted)       → Unpainted
/// merge_paint(Painted({A}), Unpainted)    → Painted({A})
/// merge_paint(Unpainted, Painted({B}))    → Painted({B})
/// merge_paint(Painted({A}), Painted({B})) → Painted({A, B})
/// merge_paint(Painted({A}), Painted({A})) → Painted({A})
/// ```
pub fn merge_paint(a: &PaintState, b: &PaintState) -> PaintState {
    match (a, b) {
        // Both unpainted — result is unpainted.
        (PaintState::Unpainted, PaintState::Unpainted) => PaintState::Unpainted,

        // One side painted, the other unpainted — clone the painted side.
        (PaintState::Painted(set), PaintState::Unpainted)
        | (PaintState::Unpainted, PaintState::Painted(set)) => {
            PaintState::Painted(set.clone())
        }

        // Both sides painted — compute the union.
        (PaintState::Painted(set_a), PaintState::Painted(set_b)) => {
            PaintState::Painted(union_paint_sets(set_a, set_b))
        }
    }
}

/// Computes the union of two `PaintSet` values.
///
/// Handles all combinations of `Single` and `Multi` variants efficiently,
/// avoiding unnecessary heap allocations when possible.
fn union_paint_sets(a: &PaintSet, b: &PaintSet) -> PaintSet {
    match (a, b) {
        // Single + Single
        (PaintSet::Single(sa), PaintSet::Single(sb)) => {
            if *sa == *sb {
                // Same symbol — result is still single.
                PaintSet::Single(*sa)
            } else {
                // Different symbols — promote to Multi.
                let mut set = FxHashSet::default();
                set.insert(*sa);
                set.insert(*sb);
                PaintSet::Multi(set)
            }
        }

        // Single + Multi — insert the single into a clone of the multi.
        (PaintSet::Single(s), PaintSet::Multi(set)) => {
            let mut result = set.clone();
            result.insert(*s);
            if result.len() == 1 {
                PaintSet::Single(*result.iter().next().unwrap())
            } else {
                PaintSet::Multi(result)
            }
        }

        // Multi + Single — symmetric to above.
        (PaintSet::Multi(set), PaintSet::Single(s)) => {
            let mut result = set.clone();
            result.insert(*s);
            if result.len() == 1 {
                PaintSet::Single(*result.iter().next().unwrap())
            } else {
                PaintSet::Multi(result)
            }
        }

        // Multi + Multi — union of both sets.
        (PaintSet::Multi(set_a), PaintSet::Multi(set_b)) => {
            // Clone the larger set and extend with the smaller for efficiency.
            let (base, extend_from) = if set_a.len() >= set_b.len() {
                (set_a, set_b)
            } else {
                (set_b, set_a)
            };
            let mut result = base.clone();
            result.extend(extend_from.iter().copied());
            if result.len() == 1 {
                PaintSet::Single(*result.iter().next().unwrap())
            } else {
                PaintSet::Multi(result)
            }
        }
    }
}

/// Removes all paint from a token, resetting it to `Unpainted`.
///
/// This is rarely needed in normal preprocessing but is provided for:
/// - Resetting tokens during error recovery.
/// - Test fixtures that need to clear paint state.
/// - Edge cases where the preprocessor needs to "forget" expansion history.
///
/// After calling this, the token is eligible for expansion as any macro,
/// just as if it had never been produced by macro expansion.
///
/// # Arguments
///
/// * `token` — Mutable reference to the painted token to unpaint.
#[inline]
pub fn unpaint(token: &mut PaintedToken) {
    token.paint = PaintState::Unpainted;
}

/// Removes paint for a specific macro from a token.
///
/// If the token is painted for `macro_name`, that entry is removed. If no
/// paint entries remain after removal, the token transitions to `Unpainted`.
/// If the token was not painted for `macro_name`, this is a no-op.
///
/// # Arguments
///
/// * `token` — Mutable reference to the painted token.
/// * `macro_name` — The `Symbol` handle of the macro to unpaint.
///
/// # Examples
///
/// ```text
/// // Token painted for {A, B}
/// unpaint_for(&mut token, sym_a);
/// // Token now painted for {B}
///
/// unpaint_for(&mut token, sym_b);
/// // Token now Unpainted
/// ```
pub fn unpaint_for(token: &mut PaintedToken, macro_name: Symbol) {
    match &mut token.paint {
        PaintState::Unpainted => {
            // Already unpainted — nothing to do.
        }
        PaintState::Painted(set) => {
            let became_empty = set.remove(macro_name);
            if became_empty {
                // The set is now empty — transition to Unpainted.
                token.paint = PaintState::Unpainted;
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::frontend::lexer::token::TokenKind;

    /// Helper: create a Symbol from a raw u32 index.
    /// In tests we bypass the Interner and construct Symbols directly.
    fn sym(index: u32) -> Symbol {
        // Symbol is a newtype around u32 with no public constructor,
        // but we can access it indirectly via the EMPTY constant and
        // use distinct values by relying on the fact that Symbol
        // is Copy + Eq + Hash and its identity is its u32 index.
        //
        // For test purposes, we construct symbols by casting.
        // This is safe because Symbol(u32) is repr-transparent.
        unsafe { std::mem::transmute::<u32, Symbol>(index) }
    }

    /// Helper: create a simple identifier token with a given symbol.
    fn make_token(symbol: Symbol) -> Token {
        Token::new(TokenKind::Identifier(symbol), Span::DUMMY)
    }

    /// Helper: create a PaintedToken wrapping an identifier.
    fn make_painted_token(symbol: Symbol) -> PaintedToken {
        PaintedToken::new(make_token(symbol))
    }

    // -----------------------------------------------------------------------
    // PaintState basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_paint_state_is_unpainted() {
        let state = PaintState::default();
        assert!(matches!(state, PaintState::Unpainted));
    }

    #[test]
    fn test_new_painted_creates_single() {
        let s = sym(42);
        let state = PaintState::new_painted(s);
        match &state {
            PaintState::Painted(PaintSet::Single(inner)) => assert_eq!(*inner, s),
            other => panic!("Expected Painted(Single), got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // PaintedToken construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_painted_token_new_is_unpainted() {
        let s = sym(1);
        let pt = make_painted_token(s);
        assert!(matches!(pt.paint, PaintState::Unpainted));
        // Verify we can access token.kind and token.span
        assert!(pt.token.kind.is_identifier());
        assert_eq!(pt.token.span, Span::DUMMY);
    }

    #[test]
    fn test_painted_token_with_paint() {
        let s = sym(1);
        let tok = make_token(s);
        let pt = PaintedToken::with_paint(tok, PaintState::new_painted(s));
        assert!(is_painted(&pt));
        assert!(is_painted_for(&pt, s));
    }

    // -----------------------------------------------------------------------
    // paint_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_paint_tokens_marks_unpainted() {
        let s1 = sym(1);
        let s2 = sym(2);
        let macro_sym = sym(100);

        let mut tokens = vec![make_painted_token(s1), make_painted_token(s2)];

        paint_tokens(&mut tokens, macro_sym);

        assert!(is_painted_for(&tokens[0], macro_sym));
        assert!(is_painted_for(&tokens[1], macro_sym));
    }

    #[test]
    fn test_paint_tokens_accumulates() {
        let s1 = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s1)];

        // First paint
        paint_tokens(&mut tokens, macro_a);
        assert!(is_painted_for(&tokens[0], macro_a));
        assert!(!is_painted_for(&tokens[0], macro_b));

        // Second paint
        paint_tokens(&mut tokens, macro_b);
        assert!(is_painted_for(&tokens[0], macro_a));
        assert!(is_painted_for(&tokens[0], macro_b));
    }

    #[test]
    fn test_paint_same_macro_twice_is_idempotent() {
        let s1 = sym(1);
        let macro_a = sym(10);

        let mut tokens = vec![make_painted_token(s1)];

        paint_tokens(&mut tokens, macro_a);
        paint_tokens(&mut tokens, macro_a);

        assert!(is_painted_for(&tokens[0], macro_a));
        // Should still be Single internally
        match &tokens[0].paint {
            PaintState::Painted(PaintSet::Single(s)) => assert_eq!(*s, macro_a),
            other => panic!("Expected Single after idempotent paint, got {:?}", other),
        }
    }

    #[test]
    fn test_paint_empty_slice() {
        let macro_a = sym(10);
        let mut tokens: Vec<PaintedToken> = vec![];
        // Should not panic on empty slice.
        paint_tokens(&mut tokens, macro_a);
        assert!(tokens.is_empty());
    }

    // -----------------------------------------------------------------------
    // is_painted / is_painted_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_painted_unpainted_token() {
        let pt = make_painted_token(sym(1));
        assert!(!is_painted(&pt));
        assert!(!is_painted_for(&pt, sym(1)));
    }

    #[test]
    fn test_is_painted_for_correct_macro() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);

        assert!(is_painted(&tokens[0]));
        assert!(is_painted_for(&tokens[0], macro_a));
        assert!(!is_painted_for(&tokens[0], macro_b));
    }

    // -----------------------------------------------------------------------
    // merge_paint
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_both_unpainted() {
        let result = merge_paint(&PaintState::Unpainted, &PaintState::Unpainted);
        assert!(matches!(result, PaintState::Unpainted));
    }

    #[test]
    fn test_merge_one_painted_one_unpainted() {
        let macro_a = sym(10);
        let painted = PaintState::new_painted(macro_a);

        let r1 = merge_paint(&painted, &PaintState::Unpainted);
        let r2 = merge_paint(&PaintState::Unpainted, &painted);

        match (&r1, &r2) {
            (PaintState::Painted(s1), PaintState::Painted(s2)) => {
                assert!(s1.contains(macro_a));
                assert!(s2.contains(macro_a));
            }
            _ => panic!("Expected both to be Painted"),
        }
    }

    #[test]
    fn test_merge_both_painted_different() {
        let macro_a = sym(10);
        let macro_b = sym(20);

        let pa = PaintState::new_painted(macro_a);
        let pb = PaintState::new_painted(macro_b);

        let result = merge_paint(&pa, &pb);

        match &result {
            PaintState::Painted(set) => {
                assert!(set.contains(macro_a));
                assert!(set.contains(macro_b));
            }
            _ => panic!("Expected Painted with both macros"),
        }
    }

    #[test]
    fn test_merge_both_painted_same() {
        let macro_a = sym(10);

        let pa = PaintState::new_painted(macro_a);
        let pb = PaintState::new_painted(macro_a);

        let result = merge_paint(&pa, &pb);

        match &result {
            PaintState::Painted(PaintSet::Single(s)) => assert_eq!(*s, macro_a),
            _ => panic!("Expected Painted(Single) for identical merge"),
        }
    }

    // -----------------------------------------------------------------------
    // unpaint / unpaint_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_unpaint_removes_all_paint() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);
        paint_tokens(&mut tokens, macro_b);

        assert!(is_painted(&tokens[0]));

        unpaint(&mut tokens[0]);

        assert!(!is_painted(&tokens[0]));
        assert!(!is_painted_for(&tokens[0], macro_a));
        assert!(!is_painted_for(&tokens[0], macro_b));
    }

    #[test]
    fn test_unpaint_for_specific_macro() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);
        paint_tokens(&mut tokens, macro_b);

        // Remove paint for macro_a only.
        unpaint_for(&mut tokens[0], macro_a);

        assert!(is_painted(&tokens[0]));
        assert!(!is_painted_for(&tokens[0], macro_a));
        assert!(is_painted_for(&tokens[0], macro_b));
    }

    #[test]
    fn test_unpaint_for_last_macro_transitions_to_unpainted() {
        let s = sym(1);
        let macro_a = sym(10);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);

        unpaint_for(&mut tokens[0], macro_a);

        assert!(!is_painted(&tokens[0]));
        assert!(matches!(tokens[0].paint, PaintState::Unpainted));
    }

    #[test]
    fn test_unpaint_for_nonexistent_macro_is_noop() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);

        // Unpaint for a macro that was never painted — should be no-op.
        unpaint_for(&mut tokens[0], macro_b);

        assert!(is_painted(&tokens[0]));
        assert!(is_painted_for(&tokens[0], macro_a));
    }

    #[test]
    fn test_unpaint_for_on_unpainted_is_noop() {
        let s = sym(1);
        let macro_a = sym(10);

        let mut pt = make_painted_token(s);

        // Unpaint on an already-unpainted token — no-op.
        unpaint_for(&mut pt, macro_a);
        assert!(!is_painted(&pt));
    }

    // -----------------------------------------------------------------------
    // Self-referential macro simulation: #define A A
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_referential_macro_terminates() {
        // Simulates the expansion of: #define A A
        //
        // Step 1: Expand A → replacement is [A]
        // Step 2: Paint for macro 'A' → [A(painted:A)]
        // Step 3: Rescan: encounter A, check is_painted_for(A) → true → stop.
        //
        // This MUST terminate and not loop.

        let sym_a = sym(1);
        let macro_a = sym_a; // macro name is the same symbol as the identifier

        // Replacement list for #define A A: one token "A"
        let mut replacement = vec![make_painted_token(sym_a)];

        // Paint the replacement list with the expanding macro's name.
        paint_tokens(&mut replacement, macro_a);

        // During rescan, check if the token is painted for macro A.
        let should_suppress = is_painted_for(&replacement[0], macro_a);
        assert!(
            should_suppress,
            "#define A A must suppress re-expansion of A"
        );
    }

    // -----------------------------------------------------------------------
    // Mutual-recursion simulation: #define A B / #define B A
    // -----------------------------------------------------------------------

    #[test]
    fn test_mutual_recursion_terminates() {
        // Simulates:
        //   #define A B
        //   #define B A
        //
        // Step 1: Expand A → replacement [B], paint for 'A' → [B(painted:A)]
        // Step 2: Rescan: B is a macro (not painted for B) → expand B
        // Step 3: B's replacement [A], paint for 'B' → [A(painted:A,B)]
        //         (A inherits paint from the context: painted:A from step 1,
        //          plus painted:B from step 3)
        // Step 4: Rescan: A is painted for 'A' → stop.

        let sym_a = sym(1);
        let sym_b = sym(2);

        // Step 1: Expand A → [B]
        let mut replacement_a = vec![make_painted_token(sym_b)];
        paint_tokens(&mut replacement_a, sym_a); // Paint for macro A

        // Verify B is not painted for B (can still expand)
        assert!(!is_painted_for(&replacement_a[0], sym_b));

        // Step 2-3: Expand B → [A], but A inherits paint from the outer context.
        // The macro expander would propagate the paint from the context token.
        let mut replacement_b = vec![make_painted_token(sym_a)];

        // Paint for macro B (current expansion)
        paint_tokens(&mut replacement_b, sym_b);

        // Also propagate paint from the context (the B token was painted for A)
        paint_tokens(&mut replacement_b, sym_a);

        // Step 4: A is painted for both A and B.
        assert!(is_painted_for(&replacement_b[0], sym_a));
        assert!(is_painted_for(&replacement_b[0], sym_b));

        // The macro expander would see A painted for 'A' → stop. Terminates.
    }

    // -----------------------------------------------------------------------
    // PaintState equality
    // -----------------------------------------------------------------------

    #[test]
    fn test_paint_state_equality() {
        let a = sym(1);
        let b = sym(2);

        assert_eq!(PaintState::Unpainted, PaintState::Unpainted);
        assert_eq!(PaintState::new_painted(a), PaintState::new_painted(a));
        assert_ne!(PaintState::new_painted(a), PaintState::new_painted(b));
        assert_ne!(PaintState::Unpainted, PaintState::new_painted(a));
    }

    // -----------------------------------------------------------------------
    // Token field access through PaintedToken
    // -----------------------------------------------------------------------

    #[test]
    fn test_painted_token_field_access() {
        let s = sym(42);
        let tok = make_token(s);
        let pt = PaintedToken::new(tok);

        // Access token.kind
        match &pt.token.kind {
            TokenKind::Identifier(sym) => assert_eq!(sym.as_u32(), 42),
            other => panic!("Expected Identifier, got {:?}", other),
        }

        // Access token.span
        assert_eq!(pt.token.span, Span::DUMMY);
    }

    // -----------------------------------------------------------------------
    // Multi-paint set operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_three_way_paint_accumulation() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);
        let macro_c = sym(30);

        let mut tokens = vec![make_painted_token(s)];

        paint_tokens(&mut tokens, macro_a);
        paint_tokens(&mut tokens, macro_b);
        paint_tokens(&mut tokens, macro_c);

        assert!(is_painted_for(&tokens[0], macro_a));
        assert!(is_painted_for(&tokens[0], macro_b));
        assert!(is_painted_for(&tokens[0], macro_c));
        assert!(!is_painted_for(&tokens[0], sym(99)));
    }

    #[test]
    fn test_unpaint_for_demotes_multi_to_single() {
        let s = sym(1);
        let macro_a = sym(10);
        let macro_b = sym(20);

        let mut tokens = vec![make_painted_token(s)];
        paint_tokens(&mut tokens, macro_a);
        paint_tokens(&mut tokens, macro_b);

        // Should be Multi now.
        assert!(matches!(
            tokens[0].paint,
            PaintState::Painted(PaintSet::Multi(_))
        ));

        // Remove one — should demote to Single.
        unpaint_for(&mut tokens[0], macro_a);
        match &tokens[0].paint {
            PaintState::Painted(PaintSet::Single(sym)) => assert_eq!(*sym, macro_b),
            other => panic!("Expected Single after demotion, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // merge_paint with multi-entry sets
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_multi_with_multi() {
        let a = sym(1);
        let b = sym(2);
        let c = sym(3);

        // Build multi-set {a, b}
        let mut set_ab = FxHashSet::default();
        set_ab.insert(a);
        set_ab.insert(b);
        let paint_ab = PaintState::Painted(PaintSet::Multi(set_ab));

        // Build multi-set {b, c}
        let mut set_bc = FxHashSet::default();
        set_bc.insert(b);
        set_bc.insert(c);
        let paint_bc = PaintState::Painted(PaintSet::Multi(set_bc));

        let merged = merge_paint(&paint_ab, &paint_bc);

        match &merged {
            PaintState::Painted(set) => {
                assert!(set.contains(a));
                assert!(set.contains(b));
                assert!(set.contains(c));
            }
            _ => panic!("Expected Painted after merge"),
        }
    }
}
