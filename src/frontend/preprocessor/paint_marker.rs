//! Token-level paint marker implementation for C preprocessor recursion protection.
//!
//! The paint-marker system prevents infinite recursion during macro expansion by
//! tracking which macro names are currently "in flight" (being expanded). When a
//! macro name is encountered during its own expansion, the token is marked as
//! "painted" and treated as an ordinary identifier — it is **NOT** re-expanded.
//!
//! # C11 §6.10.3.4 — Rescanning and Further Replacement
//!
//! > If the name of the macro being replaced is found during [the] scan of the
//! > replacement list [...], it is not replaced. Furthermore, if any nested
//! > replacements encounter the name of the macro being replaced, it is not
//! > replaced.
//!
//! # Architecture
//!
//! Paint markers are architecturally **distinct** from circular `#include` detection:
//!
//! - **Paint markers** operate at the **token/macro level** during Phase 2 expansion.
//!   They prevent re-expansion of self-referential macros like `#define A A`.
//! - **Circular include detection** (in `include_handler.rs`) operates at the
//!   **file level** to detect `#include` cycles.
//!
//! These are completely separate mechanisms solving different problems.
//!
//! # Examples
//!
//! ```ignore
//! use bcc::frontend::preprocessor::paint_marker::{PaintMarker, PaintState};
//!
//! let mut marker = PaintMarker::new();
//! assert_eq!(marker.check_token_paint("A"), PaintState::Unpainted);
//!
//! marker.paint("A");
//! assert_eq!(marker.check_token_paint("A"), PaintState::Painted);
//!
//! marker.unpaint("A");
//! assert_eq!(marker.check_token_paint("A"), PaintState::Unpainted);
//! ```
//!
//! # Self-Referential Macro Example
//!
//! Given `#define A A`:
//!
//! 1. Expansion of `A` begins — `paint("A")` → active = `{"A"}`
//! 2. Replacement is token `A`
//! 3. Rescan: `check_token_paint("A")` → `Painted` → stop (do NOT re-expand)
//! 4. Expansion ends — `unpaint("A")` → active = `{}`
//!
//! Result: token `A` remains as-is. **Terminates immediately** (no infinite recursion).

use crate::common::fx_hash::FxHashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// PaintState — per-token expansion eligibility
// ---------------------------------------------------------------------------

/// Paint state for a preprocessing token.
///
/// Determines whether the token can be further macro-expanded during
/// the rescan phase of macro expansion (C11 §6.10.3.4).
///
/// - [`Unpainted`](PaintState::Unpainted) tokens are eligible for macro expansion.
/// - [`Painted`](PaintState::Painted) tokens appeared during expansion of the macro
///   they name and must **NOT** be re-expanded; they are treated as ordinary identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaintState {
    /// Token is unpainted — eligible for macro expansion.
    Unpainted,
    /// Token is painted — it appeared during expansion of the macro it names,
    /// so it must NOT be re-expanded (treated as ordinary identifier).
    Painted,
}

impl Default for PaintState {
    /// Returns [`PaintState::Unpainted`] — tokens are eligible for expansion by default.
    #[inline]
    fn default() -> Self {
        PaintState::Unpainted
    }
}

impl fmt::Display for PaintState {
    /// Formats the paint state as a human-readable string.
    ///
    /// - `PaintState::Painted` → `"painted"`
    /// - `PaintState::Unpainted` → `"unpainted"`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaintState::Painted => write!(f, "painted"),
            PaintState::Unpainted => write!(f, "unpainted"),
        }
    }
}

// ---------------------------------------------------------------------------
// PaintMarker — macro expansion recursion tracker
// ---------------------------------------------------------------------------

/// Tracks which macro names are currently "in flight" (being expanded).
///
/// Used to prevent infinite recursion in self-referential macros per
/// C11 §6.10.3.4. The paint-marker system works as follows:
///
/// 1. When a macro begins expansion, its name is **painted** (added to the
///    active expansion set) via [`paint()`](PaintMarker::paint).
/// 2. During the rescan of the replacement token list, if a token matches
///    an active (painted) macro name, it is marked as [`PaintState::Painted`]
///    and is **NOT** re-expanded.
/// 3. When expansion completes, the macro name is **unpainted** (removed from
///    the active set) via [`unpaint()`](PaintMarker::unpaint).
///
/// This mechanism naturally supports nested expansion. For example, with
/// `#define A B` and `#define B A`:
///
/// - Expanding `A`: paint `"A"` → replacement `B` → rescan, expand `B` →
///   paint `"B"` → replacement `A` → rescan, `A` IS painted → stop →
///   unpaint `"B"` → unpaint `"A"`.
/// - Result: `A` (the inner token remains unexpanded because it is painted).
///
/// ## Thread Safety
///
/// `PaintMarker` is **NOT** thread-safe and does not use any synchronization
/// primitives. It is designed to be used within a single compilation worker
/// thread (the 64 MiB stack worker thread spawned by `main.rs`).
///
/// ## Relationship to Recursion Depth Limit
///
/// The paint-marker system is **conceptually independent** from the 512-depth
/// recursion limit enforced by `macro_expander.rs`. The paint marker prevents
/// re-expansion of self-referential tokens; the depth limit prevents stack
/// overflow from deeply nested (non-self-referential) macro chains.
pub struct PaintMarker {
    /// Set of macro names currently being expanded (the "active expansion stack").
    ///
    /// Uses [`FxHashSet`] (Fibonacci hashing) instead of the standard library's
    /// `HashSet` (SipHash) for faster lookups in performance-critical macro
    /// expansion paths.
    active_expansions: FxHashSet<String>,
}

impl Default for PaintMarker {
    /// Returns a new, empty `PaintMarker` with no active expansions.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PaintMarker {
    /// Creates a new, empty `PaintMarker` with no active expansions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let marker = PaintMarker::new();
    /// assert!(marker.is_empty());
    /// assert_eq!(marker.active_count(), 0);
    /// ```
    #[inline]
    pub fn new() -> Self {
        PaintMarker {
            active_expansions: FxHashSet::default(),
        }
    }

    /// Marks a macro name as "painted" (currently being expanded).
    ///
    /// Called at the **start** of expanding a macro. After this call, any token
    /// matching `macro_name` encountered during the rescan of the replacement
    /// list will be considered painted and will NOT be re-expanded.
    ///
    /// # Parameters
    ///
    /// - `macro_name`: The name of the macro beginning expansion.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut marker = PaintMarker::new();
    /// marker.paint("FOO");
    /// assert!(marker.is_painted("FOO"));
    /// ```
    #[inline]
    pub fn paint(&mut self, macro_name: &str) {
        self.active_expansions.insert(macro_name.to_string());
    }

    /// Removes the "painted" mark from a macro name.
    ///
    /// Called at the **end** of expanding a macro (after the rescan of the
    /// replacement list completes). After this call, tokens matching
    /// `macro_name` are once again eligible for expansion.
    ///
    /// If `macro_name` is not currently painted, this is a no-op.
    ///
    /// # Parameters
    ///
    /// - `macro_name`: The name of the macro whose expansion has completed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut marker = PaintMarker::new();
    /// marker.paint("FOO");
    /// assert!(marker.is_painted("FOO"));
    /// marker.unpaint("FOO");
    /// assert!(!marker.is_painted("FOO"));
    /// ```
    #[inline]
    pub fn unpaint(&mut self, macro_name: &str) {
        self.active_expansions.remove(macro_name);
    }

    /// Checks whether a macro name is currently painted (being expanded).
    ///
    /// Returns `true` if `macro_name` is in the active expansion set, meaning
    /// any token matching this name must NOT be re-expanded during rescan.
    ///
    /// This is the primary query used by `macro_expander.rs` during the
    /// rescan phase of macro expansion.
    ///
    /// # Parameters
    ///
    /// - `macro_name`: The name to check against active expansions.
    ///
    /// # Returns
    ///
    /// `true` if the macro name is currently painted (do NOT expand).
    /// `false` if the macro name is eligible for expansion.
    #[inline]
    pub fn is_painted(&self, macro_name: &str) -> bool {
        self.active_expansions.contains(macro_name)
    }

    /// Returns `true` if no macros are currently being expanded.
    ///
    /// Useful for debugging and assertions — the paint marker should be
    /// empty when macro expansion is not in progress.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active_expansions.is_empty()
    }

    /// Returns the number of macros currently being expanded.
    ///
    /// This corresponds to the depth of nested macro expansion currently
    /// tracked by the paint marker. For example, if macro `A` is being
    /// expanded and its replacement contains macro `B` which is also being
    /// expanded, `active_count()` returns 2.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_expansions.len()
    }

    /// Checks a token against the paint marker and returns its paint state.
    ///
    /// If `token_text` matches a macro name currently in the active expansion
    /// set, returns [`PaintState::Painted`] (the token must NOT be re-expanded).
    /// Otherwise, returns [`PaintState::Unpainted`] (the token is eligible for
    /// expansion).
    ///
    /// This is a convenience method that combines [`is_painted()`](PaintMarker::is_painted)
    /// with a [`PaintState`] return value for cleaner integration with the
    /// macro expander's token processing loop.
    ///
    /// # Parameters
    ///
    /// - `token_text`: The text of the token to check (typically an identifier).
    ///
    /// # Returns
    ///
    /// [`PaintState::Painted`] if the token matches an active expansion,
    /// [`PaintState::Unpainted`] otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut marker = PaintMarker::new();
    /// assert_eq!(marker.check_token_paint("X"), PaintState::Unpainted);
    ///
    /// marker.paint("X");
    /// assert_eq!(marker.check_token_paint("X"), PaintState::Painted);
    /// assert_eq!(marker.check_token_paint("Y"), PaintState::Unpainted);
    /// ```
    #[inline]
    pub fn check_token_paint(&self, token_text: &str) -> PaintState {
        if self.active_expansions.contains(token_text) {
            PaintState::Painted
        } else {
            PaintState::Unpainted
        }
    }
}

impl fmt::Debug for PaintMarker {
    /// Formats the `PaintMarker` for debugging, showing the set of currently
    /// active macro names.
    ///
    /// Output format: `PaintMarker { active: {"A", "B"} }` (when macros A and
    /// B are being expanded) or `PaintMarker { active: {} }` (when no macros
    /// are being expanded).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PaintMarker {{ active: {{")?;
        let mut first = true;
        for name in &self.active_expansions {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "\"{}\"", name)?;
            first = false;
        }
        write!(f, "}} }}")
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PaintState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_paint_state_default_is_unpainted() {
        let state = PaintState::default();
        assert_eq!(state, PaintState::Unpainted);
    }

    #[test]
    fn test_paint_state_equality() {
        assert_eq!(PaintState::Painted, PaintState::Painted);
        assert_eq!(PaintState::Unpainted, PaintState::Unpainted);
        assert_ne!(PaintState::Painted, PaintState::Unpainted);
    }

    #[test]
    fn test_paint_state_clone_copy() {
        let state = PaintState::Painted;
        let cloned = state.clone();
        let copied = state;
        assert_eq!(state, cloned);
        assert_eq!(state, copied);
    }

    #[test]
    fn test_paint_state_debug() {
        let painted = format!("{:?}", PaintState::Painted);
        let unpainted = format!("{:?}", PaintState::Unpainted);
        assert_eq!(painted, "Painted");
        assert_eq!(unpainted, "Unpainted");
    }

    #[test]
    fn test_paint_state_display_painted() {
        assert_eq!(format!("{}", PaintState::Painted), "painted");
    }

    #[test]
    fn test_paint_state_display_unpainted() {
        assert_eq!(format!("{}", PaintState::Unpainted), "unpainted");
    }

    // -----------------------------------------------------------------------
    // PaintMarker::new() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_creates_empty_marker() {
        let marker = PaintMarker::new();
        assert!(marker.is_empty());
        assert_eq!(marker.active_count(), 0);
    }

    // -----------------------------------------------------------------------
    // PaintMarker::paint() / is_painted() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_paint_adds_macro_to_active() {
        let mut marker = PaintMarker::new();
        marker.paint("A");
        assert!(marker.is_painted("A"));
        assert!(!marker.is_empty());
        assert_eq!(marker.active_count(), 1);
    }

    #[test]
    fn test_paint_multiple_macros() {
        let mut marker = PaintMarker::new();
        marker.paint("A");
        marker.paint("B");
        marker.paint("C");
        assert!(marker.is_painted("A"));
        assert!(marker.is_painted("B"));
        assert!(marker.is_painted("C"));
        assert_eq!(marker.active_count(), 3);
    }

    #[test]
    fn test_paint_same_macro_twice_is_idempotent() {
        let mut marker = PaintMarker::new();
        marker.paint("A");
        marker.paint("A");
        assert!(marker.is_painted("A"));
        assert_eq!(marker.active_count(), 1);
    }

    #[test]
    fn test_is_painted_returns_false_for_unpainted_macro() {
        let marker = PaintMarker::new();
        assert!(!marker.is_painted("A"));
        assert!(!marker.is_painted("SOME_MACRO"));
    }

    // -----------------------------------------------------------------------
    // PaintMarker::unpaint() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unpaint_removes_macro_from_active() {
        let mut marker = PaintMarker::new();
        marker.paint("A");
        assert!(marker.is_painted("A"));
        marker.unpaint("A");
        assert!(!marker.is_painted("A"));
        assert!(marker.is_empty());
    }

    #[test]
    fn test_unpaint_nonexistent_is_noop() {
        let mut marker = PaintMarker::new();
        marker.unpaint("NONEXISTENT");
        assert!(marker.is_empty());
    }

    #[test]
    fn test_unpaint_only_removes_specified_macro() {
        let mut marker = PaintMarker::new();
        marker.paint("A");
        marker.paint("B");
        marker.unpaint("A");
        assert!(!marker.is_painted("A"));
        assert!(marker.is_painted("B"));
        assert_eq!(marker.active_count(), 1);
    }

    // -----------------------------------------------------------------------
    // PaintMarker::is_empty() / active_count() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_empty_after_paint_and_unpaint() {
        let mut marker = PaintMarker::new();
        assert!(marker.is_empty());
        marker.paint("X");
        assert!(!marker.is_empty());
        marker.unpaint("X");
        assert!(marker.is_empty());
    }

    #[test]
    fn test_active_count_tracks_correctly() {
        let mut marker = PaintMarker::new();
        assert_eq!(marker.active_count(), 0);
        marker.paint("A");
        assert_eq!(marker.active_count(), 1);
        marker.paint("B");
        assert_eq!(marker.active_count(), 2);
        marker.paint("C");
        assert_eq!(marker.active_count(), 3);
        marker.unpaint("B");
        assert_eq!(marker.active_count(), 2);
        marker.unpaint("A");
        assert_eq!(marker.active_count(), 1);
        marker.unpaint("C");
        assert_eq!(marker.active_count(), 0);
    }

    // -----------------------------------------------------------------------
    // PaintMarker::check_token_paint() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_token_paint_unpainted_by_default() {
        let marker = PaintMarker::new();
        assert_eq!(marker.check_token_paint("FOO"), PaintState::Unpainted);
    }

    #[test]
    fn test_check_token_paint_painted_when_active() {
        let mut marker = PaintMarker::new();
        marker.paint("FOO");
        assert_eq!(marker.check_token_paint("FOO"), PaintState::Painted);
    }

    #[test]
    fn test_check_token_paint_other_tokens_unaffected() {
        let mut marker = PaintMarker::new();
        marker.paint("FOO");
        assert_eq!(marker.check_token_paint("BAR"), PaintState::Unpainted);
        assert_eq!(marker.check_token_paint("BAZ"), PaintState::Unpainted);
    }

    #[test]
    fn test_check_token_paint_after_unpaint() {
        let mut marker = PaintMarker::new();
        marker.paint("FOO");
        assert_eq!(marker.check_token_paint("FOO"), PaintState::Painted);
        marker.unpaint("FOO");
        assert_eq!(marker.check_token_paint("FOO"), PaintState::Unpainted);
    }

    // -----------------------------------------------------------------------
    // Nested expansion scenario tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_expansion_a_b_a() {
        // Simulates: #define A B and #define B A
        // Expanding A: paint A → replacement B → rescan → expand B →
        //   paint B → replacement A → rescan → A IS painted → stop
        //   → unpaint B → unpaint A
        let mut marker = PaintMarker::new();

        // Step 1: Start expanding A
        marker.paint("A");
        assert!(marker.is_painted("A"));
        assert!(!marker.is_painted("B"));
        assert_eq!(marker.active_count(), 1);

        // Step 2: A's replacement is "B". B is not painted, so expand B.
        assert_eq!(marker.check_token_paint("B"), PaintState::Unpainted);

        // Step 3: Start expanding B (nested)
        marker.paint("B");
        assert!(marker.is_painted("A"));
        assert!(marker.is_painted("B"));
        assert_eq!(marker.active_count(), 2);

        // Step 4: B's replacement is "A". A IS painted → stop.
        assert_eq!(marker.check_token_paint("A"), PaintState::Painted);

        // Step 5: Finish expanding B
        marker.unpaint("B");
        assert!(marker.is_painted("A"));
        assert!(!marker.is_painted("B"));
        assert_eq!(marker.active_count(), 1);

        // Step 6: Finish expanding A
        marker.unpaint("A");
        assert!(!marker.is_painted("A"));
        assert!(!marker.is_painted("B"));
        assert!(marker.is_empty());
    }

    #[test]
    fn test_self_referential_define_a_a() {
        // Simulates: #define A A
        // Expanding A: paint A → replacement A → rescan →
        //   A IS painted → stop → unpaint A
        let mut marker = PaintMarker::new();

        // Step 1: Start expanding A
        marker.paint("A");
        assert!(marker.is_painted("A"));

        // Step 2: A's replacement is "A". A IS painted → stop. No hang.
        assert_eq!(marker.check_token_paint("A"), PaintState::Painted);

        // Step 3: Finish expanding A
        marker.unpaint("A");
        assert!(marker.is_empty());
    }

    #[test]
    fn test_deeply_nested_expansion() {
        // Simulates a chain: paint A, then B, then C, then D.
        // Each should be painted when active.
        let mut marker = PaintMarker::new();
        let names = ["A", "B", "C", "D"];

        for (i, name) in names.iter().enumerate() {
            marker.paint(name);
            assert_eq!(marker.active_count(), i + 1);
            for painted_name in &names[..=i] {
                assert!(marker.is_painted(painted_name));
            }
        }

        // Unpaint in reverse order (stack-like behavior)
        for (i, name) in names.iter().rev().enumerate() {
            marker.unpaint(name);
            assert_eq!(marker.active_count(), names.len() - 1 - i);
            assert!(!marker.is_painted(name));
        }

        assert!(marker.is_empty());
    }

    // -----------------------------------------------------------------------
    // Debug formatting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_empty_marker() {
        let marker = PaintMarker::new();
        let debug_str = format!("{:?}", marker);
        assert!(debug_str.contains("PaintMarker"));
        assert!(debug_str.contains("active"));
    }

    #[test]
    fn test_debug_with_active_macros() {
        let mut marker = PaintMarker::new();
        marker.paint("FOO");
        let debug_str = format!("{:?}", marker);
        assert!(debug_str.contains("FOO"));
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_string_macro_name() {
        let mut marker = PaintMarker::new();
        marker.paint("");
        assert!(marker.is_painted(""));
        assert_eq!(marker.check_token_paint(""), PaintState::Painted);
        marker.unpaint("");
        assert!(!marker.is_painted(""));
    }

    #[test]
    fn test_long_macro_name() {
        let mut marker = PaintMarker::new();
        let long_name = "A".repeat(1024);
        marker.paint(&long_name);
        assert!(marker.is_painted(&long_name));
        marker.unpaint(&long_name);
        assert!(!marker.is_painted(&long_name));
    }

    #[test]
    fn test_case_sensitivity() {
        let mut marker = PaintMarker::new();
        marker.paint("Foo");
        assert!(marker.is_painted("Foo"));
        assert!(!marker.is_painted("foo"));
        assert!(!marker.is_painted("FOO"));
    }

    #[test]
    fn test_special_characters_in_names() {
        // C macros are identifiers, but the paint marker doesn't validate names.
        // It works with any string.
        let mut marker = PaintMarker::new();
        marker.paint("__GUARD_H__");
        assert!(marker.is_painted("__GUARD_H__"));
        marker.paint("_Complex");
        assert!(marker.is_painted("_Complex"));
    }

    #[test]
    fn test_paint_unpaint_cycle_many_times() {
        let mut marker = PaintMarker::new();
        for _ in 0..1000 {
            marker.paint("CYCLE_TEST");
            assert!(marker.is_painted("CYCLE_TEST"));
            marker.unpaint("CYCLE_TEST");
            assert!(!marker.is_painted("CYCLE_TEST"));
        }
        assert!(marker.is_empty());
    }
}
