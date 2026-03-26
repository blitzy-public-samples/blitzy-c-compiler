//! Multi-error diagnostic reporting engine for the BCC compiler.
//!
//! Provides [`Diagnostic`] with severity levels ([`Severity`]), source spans
//! ([`Span`]), human-readable messages, optional sub-notes ([`SubDiagnostic`]),
//! and optional fix suggestions ([`FixSuggestion`]). [`DiagnosticEngine`]
//! accumulates multiple diagnostics across all pipeline stages and renders
//! them to stderr in GCC-compatible format:
//!
//! ```text
//! filename:line:column: severity: message
//! ```
//!
//! This module replaces the external `codespan-reporting` crate — it is
//! feature-complete for compiler diagnostics with zero external dependencies.
//!
//! # Integration
//!
//! Every pipeline stage integrates with this module:
//! - **Preprocessor** — unterminated `#if`, circular includes, macro errors
//! - **Lexer** — invalid tokens, unterminated strings, illegal characters
//! - **Parser** — syntax errors, unsupported GCC extensions
//! - **Sema** — type errors, undeclared identifiers, constraint violations
//! - **IR lowering** — unsupported constructs, IR generation failures
//! - **Backend** — inline asm constraints, relocation overflows
//!
//! # Zero-Dependency
//!
//! This module depends only on the Rust standard library (`std`) and
//! `crate::common::source_map::SourceMap` for span resolution.

use std::fmt;

use crate::common::source_map::SourceMap;

// ---------------------------------------------------------------------------
// Severity — diagnostic severity level
// ---------------------------------------------------------------------------

/// The severity level of a compiler diagnostic.
///
/// Ordered by decreasing severity: [`Error`](Severity::Error) >
/// [`Warning`](Severity::Warning) > [`Note`](Severity::Note).
///
/// - **Error**: A hard failure that prevents compilation from producing
///   valid output.  The diagnostic engine will report `has_errors() == true`.
/// - **Warning**: A potential issue that does not block compilation but
///   should be addressed.
/// - **Note**: Supplementary information attached to a preceding error or
///   warning (e.g. "previous declaration was here").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// A hard compilation error.
    Error,
    /// A non-fatal warning.
    Warning,
    /// An informational note.
    Note,
}

impl fmt::Display for Severity {
    /// Formats the severity as its GCC-compatible lowercase name.
    ///
    /// - `Error`   → `"error"`
    /// - `Warning` → `"warning"`
    /// - `Note`    → `"note"`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
        }
    }
}

// ---------------------------------------------------------------------------
// Span — source location span
// ---------------------------------------------------------------------------

/// A contiguous byte range within a source file tracked by [`SourceMap`].
///
/// `start` is inclusive and `end` is exclusive, following the standard
/// half-open interval convention `[start, end)`.  Both are byte offsets
/// relative to the beginning of the file identified by `file_id`.
///
/// A **dummy span** (all fields zero) is used for compiler-generated
/// constructs that have no corresponding source location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Span {
    /// Source file ID from [`SourceMap`].
    pub file_id: u32,
    /// Start byte offset in the source (inclusive).
    pub start: u32,
    /// End byte offset in the source (exclusive).
    pub end: u32,
}

impl Span {
    /// Create a new span covering `[start, end)` in file `file_id`.
    #[inline]
    pub fn new(file_id: u32, start: u32, end: u32) -> Self {
        Span {
            file_id,
            start,
            end,
        }
    }

    /// Create a **dummy span** — all fields zero — for compiler-generated
    /// constructs that have no real source location.
    #[inline]
    pub fn dummy() -> Self {
        Span {
            file_id: 0,
            start: 0,
            end: 0,
        }
    }

    /// Merge two spans into the smallest span that covers both.
    ///
    /// Both spans must belong to the same file.  If the file IDs differ
    /// the current span is returned unchanged (a defensive choice that
    /// avoids panicking on malformed input).
    #[inline]
    pub fn merge(self, other: Span) -> Span {
        if self.file_id != other.file_id {
            // Cannot merge spans from different files — return self.
            return self;
        }
        Span {
            file_id: self.file_id,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Returns `true` if this is a dummy span (all fields zero).
    #[inline]
    pub fn is_dummy(&self) -> bool {
        self.file_id == 0 && self.start == 0 && self.end == 0
    }
}

// ---------------------------------------------------------------------------
// SubDiagnostic — secondary note attached to a primary diagnostic
// ---------------------------------------------------------------------------

/// A secondary note attached to a primary [`Diagnostic`].
///
/// Sub-diagnostics carry their own [`Span`] so the reader can see exactly
/// where the related information originates.  Common uses:
/// - "previous declaration was here" (points to first declaration)
/// - "candidate function not viable" (points to overload)
#[derive(Debug, Clone)]
pub struct SubDiagnostic {
    /// Source location of this note.
    pub span: Span,
    /// Human-readable message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// FixSuggestion — machine-applicable replacement suggestion
// ---------------------------------------------------------------------------

/// A machine-applicable fix suggestion that proposes replacing a source
/// range with new text.
///
/// Fix suggestions enable future IDE integration and `--fix` modes by
/// providing both the replacement text and a human-readable explanation.
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    /// The source range to be replaced.
    pub span: Span,
    /// The replacement text.
    pub replacement: String,
    /// Human-readable description of the fix (e.g. "insert semicolon").
    pub message: String,
}

// ---------------------------------------------------------------------------
// Diagnostic — a single compiler diagnostic
// ---------------------------------------------------------------------------

/// A single compiler diagnostic with a severity, source location, message,
/// optional secondary notes, and an optional machine-applicable fix.
///
/// Diagnostics are typically created via the convenience constructors
/// [`error`](Diagnostic::error), [`warning`](Diagnostic::warning), and
/// [`note`](Diagnostic::note), then enriched with the builder methods
/// [`with_note`](Diagnostic::with_note) and [`with_fix`](Diagnostic::with_fix).
///
/// # Example
///
/// ```ignore
/// let diag = Diagnostic::error(span, "undeclared identifier 'foo'")
///     .with_note(prev_span, "did you mean 'bar'?")
///     .with_fix(span, "bar", "replace with 'bar'");
/// engine.emit(diag);
/// ```
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level.
    pub severity: Severity,
    /// Source location of the primary message.
    pub span: Span,
    /// Human-readable error/warning/note message.
    pub message: String,
    /// Optional secondary notes (e.g. "previous declaration was here").
    pub notes: Vec<SubDiagnostic>,
    /// Optional machine-applicable fix suggestion.
    pub fix_suggestion: Option<FixSuggestion>,
}

impl Diagnostic {
    // -- Convenience constructors -----------------------------------------

    /// Create an **error** diagnostic at the given span.
    pub fn error(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
            notes: Vec::new(),
            fix_suggestion: None,
        }
    }

    /// Create a **warning** diagnostic at the given span.
    pub fn warning(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            span,
            message: message.into(),
            notes: Vec::new(),
            fix_suggestion: None,
        }
    }

    /// Create a **note** diagnostic at the given span.
    pub fn note(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            span,
            message: message.into(),
            notes: Vec::new(),
            fix_suggestion: None,
        }
    }

    // -- Builder methods --------------------------------------------------

    /// Attach a secondary note to this diagnostic.
    ///
    /// Returns `self` for fluent chaining.
    pub fn with_note(mut self, span: Span, message: impl Into<String>) -> Self {
        self.notes.push(SubDiagnostic {
            span,
            message: message.into(),
        });
        self
    }

    /// Attach a machine-applicable fix suggestion to this diagnostic.
    ///
    /// Returns `self` for fluent chaining.
    pub fn with_fix(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        self.fix_suggestion = Some(FixSuggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
        });
        self
    }
}

// ---------------------------------------------------------------------------
// DiagnosticEngine — accumulator and renderer
// ---------------------------------------------------------------------------

/// Accumulates compiler diagnostics and renders them to stderr in
/// GCC-compatible format.
///
/// The engine maintains separate error and warning counters so the driver
/// can decide whether to halt compilation after parsing or semantic analysis
/// without scanning the entire diagnostic list.
///
/// # Thread Safety
///
/// `DiagnosticEngine` is **not** `Sync` — it is designed to be owned by a
/// single compilation context.  The compilation driver should collect
/// diagnostics from parallel compilation units and merge them after
/// joining worker threads.
pub struct DiagnosticEngine {
    /// All accumulated diagnostics, in emission order.
    diagnostics: Vec<Diagnostic>,
    /// Number of diagnostics with [`Severity::Error`].
    error_count: usize,
    /// Number of diagnostics with [`Severity::Warning`].
    warning_count: usize,
    /// When nonzero, diagnostics are silently discarded.
    /// Incremented by [`begin_suppress`], decremented by [`end_suppress`].
    suppress_depth: usize,
}

impl DiagnosticEngine {
    // -- Construction -----------------------------------------------------

    /// Create a new, empty diagnostic engine.
    pub fn new() -> Self {
        DiagnosticEngine {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
            suppress_depth: 0,
        }
    }

    // -- Suppression ------------------------------------------------------

    /// Begin suppressing diagnostics.  While suppressed, [`emit`] silently
    /// drops all diagnostics without recording them or incrementing counters.
    /// Supports nesting: call [`end_suppress`] exactly once per
    /// `begin_suppress`.
    pub fn begin_suppress(&mut self) {
        self.suppress_depth += 1;
    }

    /// End one level of diagnostic suppression.
    pub fn end_suppress(&mut self) {
        self.suppress_depth = self.suppress_depth.saturating_sub(1);
    }

    // -- Emission ---------------------------------------------------------

    /// Emit a diagnostic into the engine.
    ///
    /// The appropriate severity counter is incremented automatically.
    /// If suppression is active (via [`begin_suppress`]), the diagnostic
    /// is silently discarded.
    pub fn emit(&mut self, diag: Diagnostic) {
        if self.suppress_depth > 0 {
            return;
        }
        match diag.severity {
            Severity::Error => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            Severity::Note => { /* Notes don't increment counters */ }
        }
        self.diagnostics.push(diag);
    }

    // -- Convenience emission helpers ------------------------------------

    /// Shorthand: emit an error diagnostic.
    pub fn emit_error(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::error(span, msg));
    }

    /// Shorthand: emit a warning diagnostic.
    pub fn emit_warning(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::warning(span, msg));
    }

    /// Shorthand: emit a note diagnostic.
    pub fn emit_note(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::note(span, msg));
    }

    // -- Queries ----------------------------------------------------------

    /// Return the number of error diagnostics emitted so far.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Return the number of warning diagnostics emitted so far.
    #[inline]
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Returns `true` if at least one error has been emitted.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Access all accumulated diagnostics in emission order.
    #[inline]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Clear all diagnostics and reset counters to zero.
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }

    // -- Formatted output -------------------------------------------------

    /// Print all diagnostics to stderr in GCC-compatible format.
    ///
    /// For each diagnostic the primary line is formatted as:
    ///
    /// ```text
    /// filename:line:column: severity: message
    /// ```
    ///
    /// followed by the source line with a caret (`^`) indicator, any
    /// attached sub-notes (indented), and any fix suggestion.
    ///
    /// A summary line is printed at the end:
    ///
    /// ```text
    /// N error(s), M warning(s) generated.
    /// ```
    ///
    /// If there are zero diagnostics nothing is printed.
    pub fn print_all(&self, source_map: &SourceMap) {
        if self.diagnostics.is_empty() {
            return;
        }

        for diag in &self.diagnostics {
            self.print_diagnostic(diag, source_map);
        }

        // Summary line.
        eprintln!(
            "{} error(s), {} warning(s) generated.",
            self.error_count, self.warning_count,
        );
    }

    /// Format and print a single diagnostic to stderr.
    fn print_diagnostic(&self, diag: &Diagnostic, source_map: &SourceMap) {
        // Resolve primary location.
        if diag.span.is_dummy() {
            // Dummy span — no filename/line/column context available.
            eprintln!("<unknown>: {}: {}", diag.severity, diag.message);
        } else {
            // Use resolve_location for full #line-aware location resolution.
            let loc = source_map.resolve_location(diag.span.file_id, diag.span.start);
            eprintln!(
                "{}:{}:{}: {}: {}",
                loc.filename, loc.line, loc.column, diag.severity, diag.message,
            );

            // Show the source line and a caret indicator if the file is
            // available in the source map via get_file().
            if let Some(file) = source_map.get_file(diag.span.file_id) {
                let line_content = file.get_line_content(loc.line);
                if !line_content.is_empty() {
                    eprintln!(" {} | {}", loc.line, line_content);

                    // Compute the caret position.  `loc.column` is 1-indexed
                    // so we need `column - 1` spaces after the gutter.
                    let gutter_width = format!(" {} | ", loc.line).len();
                    let caret_col = if loc.column > 0 {
                        (loc.column - 1) as usize
                    } else {
                        0
                    };

                    // Build the caret indicator with tildes showing the extent.
                    let underline_len = if diag.span.end > diag.span.start {
                        let extent = (diag.span.end - diag.span.start) as usize;
                        // Clamp to the remaining line length after the caret.
                        let remaining = line_content.len().saturating_sub(caret_col);
                        extent.min(remaining).max(1)
                    } else {
                        1
                    };

                    let mut indicator =
                        String::with_capacity(gutter_width + caret_col + underline_len);
                    // Fill the gutter area with spaces.
                    for _ in 0..gutter_width {
                        indicator.push(' ');
                    }
                    // Spaces to reach the caret column.
                    for _ in 0..caret_col {
                        indicator.push(' ');
                    }
                    // The caret itself.
                    indicator.push('^');
                    // Tildes for the rest of the underline extent.
                    for _ in 1..underline_len {
                        indicator.push('~');
                    }
                    eprintln!("{}", indicator);
                }
            }
        }

        // Print attached sub-notes using format_span for concise location strings.
        for sub in &diag.notes {
            if sub.span.is_dummy() {
                eprintln!("  note: {}", sub.message);
            } else {
                let location_str =
                    source_map.format_span(sub.span.file_id, sub.span.start, sub.span.end);
                eprintln!("{}: note: {}", location_str, sub.message);
            }
        }

        // Print fix suggestion if present.
        if let Some(ref fix) = diag.fix_suggestion {
            if fix.span.is_dummy() {
                eprintln!(
                    "  fix: {} (replace with '{}')",
                    fix.message, fix.replacement
                );
            } else {
                let location_str =
                    source_map.format_span(fix.span.file_id, fix.span.start, fix.span.end);
                eprintln!(
                    "{}: fix: {} (replace with '{}')",
                    location_str, fix.message, fix.replacement,
                );
            }
        }
    }

    /// Return the filename associated with a span, if it can be resolved.
    ///
    /// Uses [`SourceMap::get_filename`] for a lightweight lookup that avoids
    /// the full `resolve_location` computation when only the filename is
    /// needed (e.g. for diagnostic filtering or grouping).
    pub fn span_filename<'a>(&self, span: &Span, source_map: &'a SourceMap) -> Option<&'a str> {
        if span.is_dummy() {
            return None;
        }
        source_map.get_filename(span.file_id)
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DiagnosticEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiagnosticEngine")
            .field("error_count", &self.error_count)
            .field("warning_count", &self.warning_count)
            .field("diagnostics_len", &self.diagnostics.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::source_map::SourceMap;

    // -- Severity tests ---------------------------------------------------

    #[test]
    fn test_severity_display_error() {
        assert_eq!(format!("{}", Severity::Error), "error");
    }

    #[test]
    fn test_severity_display_warning() {
        assert_eq!(format!("{}", Severity::Warning), "warning");
    }

    #[test]
    fn test_severity_display_note() {
        assert_eq!(format!("{}", Severity::Note), "note");
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
        assert_ne!(Severity::Warning, Severity::Note);
    }

    #[test]
    fn test_severity_clone_copy() {
        let s = Severity::Error;
        let s2 = s; // Copy
        let s3 = s.clone(); // Clone
        assert_eq!(s2, Severity::Error);
        assert_eq!(s3, Severity::Error);
    }

    // -- Span tests -------------------------------------------------------

    #[test]
    fn test_span_new() {
        let span = Span::new(1, 10, 20);
        assert_eq!(span.file_id, 1);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
    }

    #[test]
    fn test_span_dummy() {
        let span = Span::dummy();
        assert_eq!(span.file_id, 0);
        assert_eq!(span.start, 0);
        assert_eq!(span.end, 0);
        assert!(span.is_dummy());
    }

    #[test]
    fn test_span_is_dummy_false() {
        let span = Span::new(0, 0, 1);
        assert!(!span.is_dummy());
    }

    #[test]
    fn test_span_merge_same_file() {
        let a = Span::new(1, 5, 10);
        let b = Span::new(1, 8, 15);
        let merged = a.merge(b);
        assert_eq!(merged.file_id, 1);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 15);
    }

    #[test]
    fn test_span_merge_disjoint() {
        let a = Span::new(1, 0, 3);
        let b = Span::new(1, 10, 20);
        let merged = a.merge(b);
        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_span_merge_different_files() {
        let a = Span::new(1, 0, 10);
        let b = Span::new(2, 5, 15);
        let merged = a.merge(b);
        // Should return `a` unchanged when files differ.
        assert_eq!(merged, a);
    }

    #[test]
    fn test_span_default() {
        let span = Span::default();
        assert!(span.is_dummy());
    }

    #[test]
    fn test_span_clone_copy() {
        let s1 = Span::new(3, 10, 20);
        let s2 = s1; // Copy
        let s3 = s1.clone(); // Clone
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
    }

    // -- Diagnostic tests -------------------------------------------------

    #[test]
    fn test_diagnostic_error_constructor() {
        let d = Diagnostic::error(Span::new(0, 5, 10), "bad thing");
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.span, Span::new(0, 5, 10));
        assert_eq!(d.message, "bad thing");
        assert!(d.notes.is_empty());
        assert!(d.fix_suggestion.is_none());
    }

    #[test]
    fn test_diagnostic_warning_constructor() {
        let d = Diagnostic::warning(Span::new(1, 0, 1), "suspicious");
        assert_eq!(d.severity, Severity::Warning);
        assert_eq!(d.message, "suspicious");
    }

    #[test]
    fn test_diagnostic_note_constructor() {
        let d = Diagnostic::note(Span::new(2, 0, 3), "info here");
        assert_eq!(d.severity, Severity::Note);
        assert_eq!(d.message, "info here");
    }

    #[test]
    fn test_diagnostic_with_note() {
        let d = Diagnostic::error(Span::new(0, 0, 5), "error msg")
            .with_note(Span::new(0, 10, 15), "see also");
        assert_eq!(d.notes.len(), 1);
        assert_eq!(d.notes[0].message, "see also");
        assert_eq!(d.notes[0].span, Span::new(0, 10, 15));
    }

    #[test]
    fn test_diagnostic_with_multiple_notes() {
        let d = Diagnostic::error(Span::new(0, 0, 1), "err")
            .with_note(Span::new(0, 5, 6), "note 1")
            .with_note(Span::new(0, 10, 11), "note 2");
        assert_eq!(d.notes.len(), 2);
        assert_eq!(d.notes[0].message, "note 1");
        assert_eq!(d.notes[1].message, "note 2");
    }

    #[test]
    fn test_diagnostic_with_fix() {
        let d = Diagnostic::error(Span::new(0, 0, 3), "typo").with_fix(
            Span::new(0, 0, 3),
            "foo",
            "replace with 'foo'",
        );
        assert!(d.fix_suggestion.is_some());
        let fix = d.fix_suggestion.unwrap();
        assert_eq!(fix.replacement, "foo");
        assert_eq!(fix.message, "replace with 'foo'");
        assert_eq!(fix.span, Span::new(0, 0, 3));
    }

    #[test]
    fn test_diagnostic_with_note_and_fix() {
        let d = Diagnostic::warning(Span::new(0, 0, 5), "warning msg")
            .with_note(Span::new(0, 10, 15), "see also")
            .with_fix(Span::new(0, 0, 5), "fixed", "apply fix");
        assert_eq!(d.notes.len(), 1);
        assert!(d.fix_suggestion.is_some());
    }

    #[test]
    fn test_diagnostic_message_into_string() {
        // Verify impl Into<String> works with &str and String.
        let d1 = Diagnostic::error(Span::dummy(), "from &str");
        let d2 = Diagnostic::error(Span::dummy(), String::from("from String"));
        assert_eq!(d1.message, "from &str");
        assert_eq!(d2.message, "from String");
    }

    // -- DiagnosticEngine tests -------------------------------------------

    #[test]
    fn test_engine_new_is_empty() {
        let engine = DiagnosticEngine::new();
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert!(!engine.has_errors());
        assert!(engine.diagnostics().is_empty());
    }

    #[test]
    fn test_engine_default() {
        let engine = DiagnosticEngine::default();
        assert_eq!(engine.error_count(), 0);
        assert!(!engine.has_errors());
    }

    #[test]
    fn test_engine_emit_error() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::error(Span::dummy(), "test error"));
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.warning_count(), 0);
        assert!(engine.has_errors());
        assert_eq!(engine.diagnostics().len(), 1);
    }

    #[test]
    fn test_engine_emit_warning() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::warning(Span::dummy(), "test warning"));
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 1);
        assert!(!engine.has_errors());
    }

    #[test]
    fn test_engine_emit_note_no_counter() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::note(Span::dummy(), "test note"));
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert!(!engine.has_errors());
        assert_eq!(engine.diagnostics().len(), 1);
    }

    #[test]
    fn test_engine_emit_mixed() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::error(Span::dummy(), "err 1"));
        engine.emit(Diagnostic::warning(Span::dummy(), "warn 1"));
        engine.emit(Diagnostic::error(Span::dummy(), "err 2"));
        engine.emit(Diagnostic::note(Span::dummy(), "note 1"));
        engine.emit(Diagnostic::warning(Span::dummy(), "warn 2"));
        assert_eq!(engine.error_count(), 2);
        assert_eq!(engine.warning_count(), 2);
        assert!(engine.has_errors());
        assert_eq!(engine.diagnostics().len(), 5);
    }

    #[test]
    fn test_engine_clear() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::error(Span::dummy(), "err"));
        engine.emit(Diagnostic::warning(Span::dummy(), "warn"));
        assert!(engine.has_errors());

        engine.clear();
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert!(!engine.has_errors());
        assert!(engine.diagnostics().is_empty());
    }

    #[test]
    fn test_engine_emit_error_shorthand() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_error(Span::new(0, 0, 5), "shorthand error");
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Error);
        assert_eq!(engine.diagnostics()[0].message, "shorthand error");
    }

    #[test]
    fn test_engine_emit_warning_shorthand() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_warning(Span::new(0, 0, 5), "shorthand warning");
        assert_eq!(engine.warning_count(), 1);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Warning);
    }

    #[test]
    fn test_engine_emit_note_shorthand() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_note(Span::new(0, 0, 5), "shorthand note");
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Note);
    }

    #[test]
    fn test_engine_diagnostics_order() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_error(Span::dummy(), "first");
        engine.emit_warning(Span::dummy(), "second");
        engine.emit_note(Span::dummy(), "third");
        let diags = engine.diagnostics();
        assert_eq!(diags[0].message, "first");
        assert_eq!(diags[1].message, "second");
        assert_eq!(diags[2].message, "third");
    }

    #[test]
    fn test_engine_clear_then_reuse() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_error(Span::dummy(), "first round");
        assert_eq!(engine.error_count(), 1);

        engine.clear();
        assert_eq!(engine.error_count(), 0);

        engine.emit_warning(Span::dummy(), "second round");
        assert_eq!(engine.warning_count(), 1);
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.diagnostics().len(), 1);
    }

    // -- print_all integration test ---------------------------------------

    #[test]
    fn test_engine_print_all_with_source_map() {
        let mut sm = SourceMap::new();
        let fid = sm.add_file("hello.c".into(), "int main() {\n    return 0;\n}\n".into());

        let mut engine = DiagnosticEngine::new();
        // Error on "main" (byte offset 4..8 on line 1).
        engine.emit(
            Diagnostic::error(Span::new(fid, 4, 8), "undeclared identifier 'main'")
                .with_note(Span::new(fid, 18, 19), "did you mean 'mane'?"),
        );
        // Warning on "return" (byte offset 18..24 on line 2).
        engine.emit_warning(Span::new(fid, 18, 24), "unreachable code");

        // This test just verifies print_all doesn't panic. Actual output
        // goes to stderr which is not captured in standard test output.
        engine.print_all(&sm);

        // Verify state.
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.warning_count(), 1);
    }

    #[test]
    fn test_engine_print_all_empty() {
        let sm = SourceMap::new();
        let engine = DiagnosticEngine::new();
        // Should be a no-op.
        engine.print_all(&sm);
    }

    #[test]
    fn test_engine_print_all_dummy_spans() {
        let sm = SourceMap::new();
        let mut engine = DiagnosticEngine::new();
        engine.emit_error(Span::dummy(), "internal compiler error");
        engine.print_all(&sm);
    }

    #[test]
    fn test_engine_print_all_with_fix_suggestion() {
        let mut sm = SourceMap::new();
        let fid = sm.add_file("test.c".into(), "int x = ;\n".into());

        let mut engine = DiagnosticEngine::new();
        engine.emit(
            Diagnostic::error(Span::new(fid, 8, 9), "expected expression").with_fix(
                Span::new(fid, 8, 8),
                "0",
                "insert '0'",
            ),
        );
        engine.print_all(&sm);
        assert_eq!(engine.error_count(), 1);
    }

    // -- SubDiagnostic tests ----------------------------------------------

    #[test]
    fn test_subdiagnostic_fields() {
        let sub = SubDiagnostic {
            span: Span::new(1, 5, 10),
            message: "related note".into(),
        };
        assert_eq!(sub.span.file_id, 1);
        assert_eq!(sub.message, "related note");
    }

    // -- FixSuggestion tests ----------------------------------------------

    #[test]
    fn test_fix_suggestion_fields() {
        let fix = FixSuggestion {
            span: Span::new(0, 0, 3),
            replacement: "bar".into(),
            message: "replace foo with bar".into(),
        };
        assert_eq!(fix.span, Span::new(0, 0, 3));
        assert_eq!(fix.replacement, "bar");
        assert_eq!(fix.message, "replace foo with bar");
    }

    // -- Debug formatting -------------------------------------------------

    #[test]
    fn test_engine_debug_format() {
        let mut engine = DiagnosticEngine::new();
        engine.emit_error(Span::dummy(), "e1");
        engine.emit_warning(Span::dummy(), "w1");
        let debug_str = format!("{:?}", engine);
        assert!(debug_str.contains("error_count: 1"));
        assert!(debug_str.contains("warning_count: 1"));
        assert!(debug_str.contains("diagnostics_len: 2"));
    }
}
