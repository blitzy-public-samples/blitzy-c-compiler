// src/common/diagnostics.rs
//
// Multi-error diagnostic reporting engine for the BCC compiler.
//
// Provides a complete diagnostic infrastructure used by every pipeline stage
// (preprocessor, lexer, parser, semantic analyzer, IR lowering, backend)
// to report errors, warnings, notes, and fix suggestions to the user.
//
// Key design decisions:
// - Lazy formatting: source locations are only resolved from byte offsets
//   when diagnostics are printed, not when they are emitted. This avoids
//   coupling the emission site to the SourceMap and allows diagnostics
//   to be collected during any compilation phase.
// - GCC-compatible output format: `filename:line:column: severity: message`
//   with source line display and caret/tilde underline indicators.
// - Multi-error collection: the engine accumulates all diagnostics rather
//   than aborting on the first error, enabling batch error reporting.
// - Thread-local usage: DiagnosticEngine is designed for single-thread use
//   within the 64 MiB worker thread. No interior mutability or synchronization.
//
// Integration points:
// - `crate::common::source_map::SourceMap` — resolves Span byte offsets
//   to file names, line numbers, and column numbers for formatted output.
// - All pipeline stages call `DiagnosticEngine::error()`, `warning()`,
//   `note()`, or `emit()` to report issues.
// - The CLI driver calls `DiagnosticEngine::print_all()` after compilation
//   (or on fatal error) to flush all collected diagnostics to stderr.

use std::fmt;
use std::io::{self, BufWriter, Write};

use crate::common::source_map::{FileId, SourceLocation, SourceMap};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a diagnostic message.
///
/// Ordered from most severe to least severe. Only `Error` severity prevents
/// successful compilation output; warnings, notes, and help messages are
/// informational.
///
/// # Variants
///
/// - `Error` — A compilation failure. The compiler will not produce output.
/// - `Warning` — A potential issue that does not prevent compilation.
/// - `Note` — Supplementary context attached to a preceding error or warning.
/// - `Help` — A suggested fix or alternative approach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Compilation failure — prevents output generation.
    Error,
    /// Potential issue — does not prevent compilation.
    Warning,
    /// Supplementary context for a preceding diagnostic.
    Note,
    /// Suggested fix or alternative.
    Help,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
            Severity::Help => write!(f, "help"),
        }
    }
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// A byte range within a source file, identifying a contiguous region of text.
///
/// Spans are the primary mechanism for associating diagnostics, AST nodes,
/// and IR instructions with their source locations. They are lightweight
/// (12 bytes, Copy) and carried throughout the entire pipeline.
///
/// # Fields
///
/// - `file_id` — Index of the source file in the `SourceMap` (corresponds to `FileId.0`).
/// - `start` — Inclusive start byte offset within the file content.
/// - `end` — Exclusive end byte offset within the file content.
///
/// # Conventions
///
/// - `start <= end` for valid spans.
/// - `start == end` represents a zero-width insertion point (cursor position).
/// - `Span::DUMMY` is used for compiler-generated constructs with no source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Source file identifier (index into SourceMap).
    pub file_id: u32,
    /// Inclusive start byte offset within the file.
    pub start: u32,
    /// Exclusive end byte offset within the file.
    pub end: u32,
}

impl Span {
    /// Sentinel span for compiler-generated diagnostics without source locations.
    ///
    /// The dummy span uses `u32::MAX` for all fields, making it trivially
    /// distinguishable from any valid span. Formatting routines detect this
    /// sentinel and omit source location display.
    pub const DUMMY: Span = Span {
        file_id: u32::MAX,
        start: u32::MAX,
        end: u32::MAX,
    };

    /// Creates a new span from its constituent parts.
    ///
    /// # Arguments
    ///
    /// * `file_id` — The source file index.
    /// * `start` — Inclusive start byte offset.
    /// * `end` — Exclusive end byte offset.
    #[inline]
    pub fn new(file_id: u32, start: u32, end: u32) -> Self {
        Span {
            file_id,
            start,
            end,
        }
    }

    /// Merges two spans into a single span covering both ranges.
    ///
    /// The resulting span extends from the minimum start to the maximum end
    /// of the two input spans. Both spans must refer to the same file;
    /// if they refer to different files, the first span's file_id is used.
    ///
    /// If either span is `DUMMY`, the other span is returned unchanged.
    ///
    /// # Arguments
    ///
    /// * `a` — First span.
    /// * `b` — Second span.
    ///
    /// # Returns
    ///
    /// A span covering the union of `a` and `b`.
    pub fn merge(a: Span, b: Span) -> Span {
        // If either is dummy, return the other
        if a == Span::DUMMY {
            return b;
        }
        if b == Span::DUMMY {
            return a;
        }

        Span {
            file_id: a.file_id,
            start: a.start.min(b.start),
            end: a.end.max(b.end),
        }
    }

    /// Returns `true` if this span is the dummy sentinel.
    #[inline]
    pub fn is_dummy(&self) -> bool {
        *self == Span::DUMMY
    }

    /// Returns the byte length of this span.
    #[inline]
    pub fn len(&self) -> u32 {
        if self.is_dummy() {
            0
        } else {
            self.end.saturating_sub(self.start)
        }
    }

    /// Returns `true` if this span has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dummy() {
            write!(f, "<no location>")
        } else {
            write!(
                f,
                "file_id={}, bytes {}..{}",
                self.file_id, self.start, self.end
            )
        }
    }
}

// ---------------------------------------------------------------------------
// FixSuggestion
// ---------------------------------------------------------------------------

/// An optional fix suggestion attached to a diagnostic.
///
/// Contains a source span to replace and the replacement text. Used by
/// the diagnostic formatter to display "did you mean..." style suggestions
/// with concrete replacement text.
///
/// # Example Output
///
/// ```text
/// test.c:5:10: error: use of undeclared identifier 'prntf'
///     prntf("hello");
///     ^~~~~
///     help: did you mean 'printf'?
/// ```
#[derive(Clone, Debug)]
pub struct FixSuggestion {
    /// The source span to be replaced.
    pub span: Span,
    /// The suggested replacement text.
    pub replacement: String,
}

// ---------------------------------------------------------------------------
// Diagnostic
// ---------------------------------------------------------------------------

/// A single diagnostic message with source location, severity, and optional
/// supplementary notes and fix suggestions.
///
/// Diagnostics are the primary communication channel between the compiler
/// and the user. They are collected by the `DiagnosticEngine` during
/// compilation and formatted for display after processing completes
/// (or on fatal error).
///
/// # Fields
///
/// - `severity` — Error, Warning, Note, or Help level.
/// - `span` — Source location of the primary issue.
/// - `message` — Human-readable description of the issue.
/// - `notes` — Additional context messages with their own spans.
/// - `fix` — Optional automated fix suggestion.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// The severity level of this diagnostic.
    pub severity: Severity,
    /// The source span where the issue was detected.
    pub span: Span,
    /// Human-readable message describing the issue.
    pub message: String,
    /// Supplementary notes providing additional context.
    /// Each note has its own span (which may differ from the primary span)
    /// and a message string.
    pub notes: Vec<(Span, String)>,
    /// Optional fix suggestion with replacement text.
    pub fix: Option<FixSuggestion>,
}

impl Diagnostic {
    /// Creates a new diagnostic with the given severity, span, and message.
    ///
    /// The diagnostic starts with no notes and no fix suggestion. Use
    /// `with_note()` and `with_fix()` to add supplementary information.
    pub fn new(severity: Severity, span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            span,
            message: message.into(),
            notes: Vec::new(),
            fix: None,
        }
    }

    /// Adds a note to this diagnostic and returns it for chaining.
    pub fn with_note(mut self, span: Span, message: impl Into<String>) -> Self {
        self.notes.push((span, message.into()));
        self
    }

    /// Attaches a fix suggestion to this diagnostic and returns it for chaining.
    pub fn with_fix(mut self, span: Span, replacement: impl Into<String>) -> Self {
        self.fix = Some(FixSuggestion {
            span,
            replacement: replacement.into(),
        });
        self
    }
}

// ---------------------------------------------------------------------------
// DiagnosticEngine
// ---------------------------------------------------------------------------

/// Central diagnostic collection and reporting engine for the BCC compiler.
///
/// All pipeline stages (preprocessor, lexer, parser, semantic analyzer,
/// IR lowering, backend) emit diagnostics through this engine. Diagnostics
/// are accumulated in a vector and formatted for display when `print_all()`
/// is called, typically after the compilation pipeline completes or when
/// a fatal error requires early termination.
///
/// # Usage Pattern
///
/// ```ignore
/// let mut engine = DiagnosticEngine::new();
///
/// // During compilation:
/// engine.error(span, "undeclared identifier 'foo'");
/// engine.warning(another_span, "unused variable 'bar'");
///
/// // After compilation:
/// if engine.has_errors() {
///     engine.print_all(&source_map);
///     std::process::exit(1);
/// }
/// ```
///
/// # Thread Safety
///
/// `DiagnosticEngine` is not thread-safe. It is designed for use within
/// a single compilation thread (the 64 MiB worker thread).
pub struct DiagnosticEngine {
    /// Collected diagnostics in emission order.
    diagnostics: Vec<Diagnostic>,
    /// Running count of Error-severity diagnostics.
    error_count: usize,
    /// Running count of Warning-severity diagnostics.
    warning_count: usize,
}

impl DiagnosticEngine {
    /// Creates a new, empty diagnostic engine with no collected diagnostics.
    pub fn new() -> Self {
        DiagnosticEngine {
            diagnostics: Vec::new(),
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Emits (records) a diagnostic into the engine.
    ///
    /// The diagnostic is appended to the collection and the appropriate
    /// counter (error or warning) is incremented based on severity.
    ///
    /// # Arguments
    ///
    /// * `diag` — The diagnostic to record.
    pub fn emit(&mut self, diag: Diagnostic) {
        match diag.severity {
            Severity::Error => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            Severity::Note | Severity::Help => {}
        }
        self.diagnostics.push(diag);
    }

    /// Convenience method to emit an error diagnostic.
    ///
    /// Creates and emits a `Diagnostic` with `Severity::Error`, the given
    /// span and message, and no notes or fix suggestion.
    ///
    /// # Arguments
    ///
    /// * `span` — Source location of the error.
    /// * `msg` — Human-readable error message.
    pub fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::new(Severity::Error, span, msg));
    }

    /// Convenience method to emit a warning diagnostic.
    ///
    /// Creates and emits a `Diagnostic` with `Severity::Warning`.
    ///
    /// # Arguments
    ///
    /// * `span` — Source location of the warning.
    /// * `msg` — Human-readable warning message.
    pub fn warning(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::new(Severity::Warning, span, msg));
    }

    /// Convenience method to emit a note diagnostic.
    ///
    /// Creates and emits a `Diagnostic` with `Severity::Note`.
    ///
    /// # Arguments
    ///
    /// * `span` — Source location for the note.
    /// * `msg` — Human-readable note message.
    pub fn note(&mut self, span: Span, msg: impl Into<String>) {
        self.emit(Diagnostic::new(Severity::Note, span, msg));
    }

    /// Returns `true` if any error-severity diagnostics have been emitted.
    ///
    /// Used by the pipeline driver to decide whether to proceed to the
    /// next compilation phase or abort.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Returns the total number of error-severity diagnostics emitted.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Returns the total number of warning-severity diagnostics emitted.
    #[inline]
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Returns a slice of all collected diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Formats and prints all collected diagnostics to stderr in
    /// GCC-compatible format.
    ///
    /// Output format for each diagnostic:
    /// ```text
    /// filename:line:column: severity: message
    ///     source line content
    ///     ^~~~ underline pointing to the span
    ///     note: supplementary context message
    ///     help: suggested fix replacement
    /// ```
    ///
    /// Uses the provided `SourceMap` to resolve `Span` byte offsets into
    /// human-readable file:line:column locations. Dummy spans produce
    /// `<unknown>:0:0` locations.
    ///
    /// # Arguments
    ///
    /// * `source_map` — The source file registry for location resolution.
    pub fn print_all(&self, source_map: &SourceMap) {
        // Use a BufWriter around stderr for efficient batched I/O.
        // Diagnostic output can be substantial for large compilations.
        let stderr = io::stderr();
        let mut writer = BufWriter::new(stderr.lock());

        for diag in &self.diagnostics {
            Self::format_diagnostic(&mut writer, diag, source_map);
        }

        // Print summary line if there were errors or warnings
        if self.error_count > 0 || self.warning_count > 0 {
            let _ = write!(
                writer,
                "{}",
                Self::summary_line(self.error_count, self.warning_count)
            );
        }

        // Flush to ensure all output is written
        let _ = writer.flush();
    }

    /// Formats a single diagnostic and writes it to the given writer.
    ///
    /// Handles the primary diagnostic message, source line display with
    /// caret/tilde underline, attached notes, and fix suggestions.
    fn format_diagnostic<W: Write>(writer: &mut W, diag: &Diagnostic, source_map: &SourceMap) {
        // --- Primary diagnostic header ---
        let location = Self::resolve_location(source_map, &diag.span);
        let _ = writeln!(writer, "{}: {}: {}", location, diag.severity, diag.message);

        // --- Source line with caret/tilde underline ---
        if !diag.span.is_dummy() {
            Self::print_source_line(writer, source_map, &diag.span);
        }

        // --- Attached notes ---
        for (note_span, note_msg) in &diag.notes {
            let note_location = Self::resolve_location(source_map, note_span);
            let _ = writeln!(writer, "{}: note: {}", note_location, note_msg);
            if !note_span.is_dummy() {
                Self::print_source_line(writer, source_map, note_span);
            }
        }

        // --- Fix suggestion ---
        if let Some(ref fix) = diag.fix {
            let fix_location = Self::resolve_location(source_map, &fix.span);
            let _ = writeln!(
                writer,
                "{}: help: replace with '{}'",
                fix_location, fix.replacement
            );
            if !fix.span.is_dummy() {
                Self::print_source_line(writer, source_map, &fix.span);
            }
        }
    }

    /// Resolves a Span to a GCC-compatible location string.
    ///
    /// For valid spans, produces `filename:line:column`.
    /// For dummy spans, produces `<unknown>:0:0`.
    fn resolve_location(source_map: &SourceMap, span: &Span) -> String {
        if span.is_dummy() {
            return "<unknown>:0:0".to_string();
        }

        let file_id = FileId(span.file_id);
        let loc: SourceLocation = source_map.lookup_location(file_id, span.start);
        format!("{}:{}:{}", loc.file_name, loc.line, loc.column)
    }

    /// Prints a source line with caret (^) and tilde (~) underline indicators.
    ///
    /// Displays the source line containing the start of the span, followed by
    /// a line with a caret at the span start position and tildes extending
    /// to cover the span width (up to the end of the line).
    ///
    /// For multi-line spans, only the first line is displayed with the caret
    /// and tildes extending to the end of that line.
    ///
    /// # Output Format
    ///
    /// ```text
    ///     int x = unknwon_func();
    ///             ^~~~~~~~~~~~~~
    /// ```
    fn print_source_line<W: Write>(writer: &mut W, source_map: &SourceMap, span: &Span) {
        let file_id = FileId(span.file_id);
        let loc: SourceLocation = source_map.lookup_location(file_id, span.start);

        // Retrieve the source line content (1-indexed line number)
        let line_content = source_map.get_line_content(file_id, loc.line);

        if line_content.is_empty() {
            return;
        }

        // Print the source line with a leading indent for visual separation
        let _ = writeln!(writer, " {}", line_content);

        // Build the caret/tilde underline
        // Column is 1-indexed, so the caret position is (column - 1) characters in
        let caret_col = loc.column.saturating_sub(1) as usize;

        // Determine underline width: span length clamped to the remaining line length
        let span_byte_len = span.end.saturating_sub(span.start) as usize;

        // For multi-line spans, clamp to end of this line
        let line_remaining = if caret_col < line_content.len() {
            line_content.len() - caret_col
        } else {
            1 // At least the caret itself
        };

        let underline_width = span_byte_len.min(line_remaining).max(1);

        // Build the underline: spaces for indent + leading whitespace, then ^ followed by ~~~
        let mut underline = String::with_capacity(caret_col + underline_width + 2);
        // Leading space matching the " " prefix on the source line
        underline.push(' ');

        // Preserve whitespace alignment: for tabs in the source line, use tabs;
        // for other characters, use spaces
        for (i, ch) in line_content.chars().enumerate() {
            if i >= caret_col {
                break;
            }
            if ch == '\t' {
                underline.push('\t');
            } else {
                underline.push(' ');
            }
        }

        // Caret at the start of the span
        underline.push('^');

        // Tildes for the rest of the underline width
        for _ in 1..underline_width {
            underline.push('~');
        }

        let _ = writeln!(writer, "{}", underline);
    }

    /// Generates a summary line reporting the total number of errors and warnings.
    ///
    /// # Examples
    ///
    /// - `"1 error generated.\n"`
    /// - `"3 errors generated.\n"`
    /// - `"2 warnings generated.\n"`
    /// - `"1 error and 2 warnings generated.\n"`
    fn summary_line(errors: usize, warnings: usize) -> String {
        let error_part = match errors {
            0 => String::new(),
            1 => "1 error".to_string(),
            n => format!("{} errors", n),
        };

        let warning_part = match warnings {
            0 => String::new(),
            1 => "1 warning".to_string(),
            n => format!("{} warnings", n),
        };

        if !error_part.is_empty() && !warning_part.is_empty() {
            format!("{} and {} generated.\n", error_part, warning_part)
        } else if !error_part.is_empty() {
            format!("{} generated.\n", error_part)
        } else if !warning_part.is_empty() {
            format!("{} generated.\n", warning_part)
        } else {
            String::new()
        }
    }
}

impl Default for DiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Severity tests ---

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Error), "error");
        assert_eq!(format!("{}", Severity::Warning), "warning");
        assert_eq!(format!("{}", Severity::Note), "note");
        assert_eq!(format!("{}", Severity::Help), "help");
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
    }

    // --- Span tests ---

    #[test]
    fn test_span_new() {
        let span = Span::new(0, 10, 20);
        assert_eq!(span.file_id, 0);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
    }

    #[test]
    fn test_span_dummy() {
        assert!(Span::DUMMY.is_dummy());
        assert_eq!(Span::DUMMY.file_id, u32::MAX);
        assert_eq!(Span::DUMMY.start, u32::MAX);
        assert_eq!(Span::DUMMY.end, u32::MAX);
    }

    #[test]
    fn test_span_not_dummy() {
        let span = Span::new(0, 0, 5);
        assert!(!span.is_dummy());
    }

    #[test]
    fn test_span_len() {
        let span = Span::new(0, 5, 15);
        assert_eq!(span.len(), 10);
    }

    #[test]
    fn test_span_len_dummy() {
        assert_eq!(Span::DUMMY.len(), 0);
    }

    #[test]
    fn test_span_is_empty() {
        let empty = Span::new(0, 5, 5);
        assert!(empty.is_empty());

        let nonempty = Span::new(0, 5, 6);
        assert!(!nonempty.is_empty());
    }

    #[test]
    fn test_span_merge_normal() {
        let a = Span::new(0, 5, 10);
        let b = Span::new(0, 15, 20);
        let merged = Span::merge(a, b);
        assert_eq!(merged.file_id, 0);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_span_merge_overlapping() {
        let a = Span::new(0, 5, 15);
        let b = Span::new(0, 10, 20);
        let merged = Span::merge(a, b);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_span_merge_with_dummy_left() {
        let a = Span::DUMMY;
        let b = Span::new(0, 10, 20);
        let merged = Span::merge(a, b);
        assert_eq!(merged, b);
    }

    #[test]
    fn test_span_merge_with_dummy_right() {
        let a = Span::new(0, 10, 20);
        let b = Span::DUMMY;
        let merged = Span::merge(a, b);
        assert_eq!(merged, a);
    }

    #[test]
    fn test_span_merge_both_dummy() {
        let merged = Span::merge(Span::DUMMY, Span::DUMMY);
        assert!(merged.is_dummy());
    }

    #[test]
    fn test_span_display_normal() {
        let span = Span::new(1, 10, 20);
        let display = format!("{}", span);
        assert!(display.contains("file_id=1"));
        assert!(display.contains("10..20"));
    }

    #[test]
    fn test_span_display_dummy() {
        let display = format!("{}", Span::DUMMY);
        assert_eq!(display, "<no location>");
    }

    // --- FixSuggestion tests ---

    #[test]
    fn test_fix_suggestion_creation() {
        let fix = FixSuggestion {
            span: Span::new(0, 5, 10),
            replacement: "printf".to_string(),
        };
        assert_eq!(fix.replacement, "printf");
        assert_eq!(fix.span.start, 5);
    }

    // --- Diagnostic tests ---

    #[test]
    fn test_diagnostic_new() {
        let diag = Diagnostic::new(
            Severity::Error,
            Span::new(0, 10, 15),
            "undeclared identifier",
        );
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.span.start, 10);
        assert_eq!(diag.message, "undeclared identifier");
        assert!(diag.notes.is_empty());
        assert!(diag.fix.is_none());
    }

    #[test]
    fn test_diagnostic_with_note() {
        let diag = Diagnostic::new(Severity::Error, Span::new(0, 10, 15), "type mismatch")
            .with_note(Span::new(0, 0, 5), "expected 'int'");

        assert_eq!(diag.notes.len(), 1);
        assert_eq!(diag.notes[0].1, "expected 'int'");
    }

    #[test]
    fn test_diagnostic_with_fix() {
        let diag = Diagnostic::new(
            Severity::Error,
            Span::new(0, 10, 15),
            "use of undeclared identifier 'prntf'",
        )
        .with_fix(Span::new(0, 10, 15), "printf");

        assert!(diag.fix.is_some());
        assert_eq!(diag.fix.as_ref().unwrap().replacement, "printf");
    }

    #[test]
    fn test_diagnostic_chaining() {
        let diag = Diagnostic::new(Severity::Error, Span::new(0, 10, 15), "error msg")
            .with_note(Span::new(0, 0, 5), "note 1")
            .with_note(Span::new(0, 20, 25), "note 2")
            .with_fix(Span::new(0, 10, 15), "fix text");

        assert_eq!(diag.notes.len(), 2);
        assert!(diag.fix.is_some());
    }

    // --- DiagnosticEngine tests ---

    #[test]
    fn test_engine_new() {
        let engine = DiagnosticEngine::new();
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert!(engine.diagnostics().is_empty());
    }

    #[test]
    fn test_engine_default() {
        let engine = DiagnosticEngine::default();
        assert!(!engine.has_errors());
    }

    #[test]
    fn test_engine_emit_error() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::new(
            Severity::Error,
            Span::new(0, 0, 5),
            "test error",
        ));
        assert!(engine.has_errors());
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.warning_count(), 0);
        assert_eq!(engine.diagnostics().len(), 1);
    }

    #[test]
    fn test_engine_emit_warning() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::new(
            Severity::Warning,
            Span::new(0, 0, 5),
            "test warning",
        ));
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 1);
    }

    #[test]
    fn test_engine_emit_note_no_count() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::new(
            Severity::Note,
            Span::new(0, 0, 5),
            "test note",
        ));
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert_eq!(engine.diagnostics().len(), 1);
    }

    #[test]
    fn test_engine_error_convenience() {
        let mut engine = DiagnosticEngine::new();
        engine.error(Span::new(0, 0, 5), "convenience error");
        assert!(engine.has_errors());
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Error);
        assert_eq!(engine.diagnostics()[0].message, "convenience error");
    }

    #[test]
    fn test_engine_warning_convenience() {
        let mut engine = DiagnosticEngine::new();
        engine.warning(Span::new(0, 0, 5), "convenience warning");
        assert_eq!(engine.warning_count(), 1);
        assert_eq!(engine.diagnostics()[0].severity, Severity::Warning);
    }

    #[test]
    fn test_engine_note_convenience() {
        let mut engine = DiagnosticEngine::new();
        engine.note(Span::new(0, 0, 5), "convenience note");
        assert_eq!(engine.diagnostics()[0].severity, Severity::Note);
    }

    #[test]
    fn test_engine_multiple_diagnostics() {
        let mut engine = DiagnosticEngine::new();
        engine.error(Span::new(0, 0, 5), "error 1");
        engine.warning(Span::new(0, 10, 15), "warning 1");
        engine.error(Span::new(0, 20, 25), "error 2");
        engine.note(Span::new(0, 30, 35), "note 1");
        engine.warning(Span::new(0, 40, 45), "warning 2");

        assert_eq!(engine.error_count(), 2);
        assert_eq!(engine.warning_count(), 2);
        assert_eq!(engine.diagnostics().len(), 5);
    }

    // --- Summary line tests ---

    #[test]
    fn test_summary_line_errors_only() {
        let line = DiagnosticEngine::summary_line(1, 0);
        assert_eq!(line, "1 error generated.\n");

        let line = DiagnosticEngine::summary_line(3, 0);
        assert_eq!(line, "3 errors generated.\n");
    }

    #[test]
    fn test_summary_line_warnings_only() {
        let line = DiagnosticEngine::summary_line(0, 1);
        assert_eq!(line, "1 warning generated.\n");

        let line = DiagnosticEngine::summary_line(0, 5);
        assert_eq!(line, "5 warnings generated.\n");
    }

    #[test]
    fn test_summary_line_both() {
        let line = DiagnosticEngine::summary_line(1, 2);
        assert_eq!(line, "1 error and 2 warnings generated.\n");
    }

    #[test]
    fn test_summary_line_none() {
        let line = DiagnosticEngine::summary_line(0, 0);
        assert_eq!(line, "");
    }

    // --- Formatting tests (using a buffer instead of stderr) ---

    #[test]
    fn test_format_diagnostic_basic() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file("test.c".to_string(), "int x = 0;\n".to_string());

        let diag = Diagnostic::new(Severity::Error, Span::new(0, 4, 5), "expected ';'");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("test.c:1:5: error: expected ';'"));
        assert!(output.contains("int x = 0;"));
        assert!(output.contains("^"));
    }

    #[test]
    fn test_format_diagnostic_with_note() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file("test.c".to_string(), "int x = y;\nint y = 0;\n".to_string());

        let diag = Diagnostic::new(
            Severity::Error,
            Span::new(0, 8, 9),
            "use of undeclared identifier 'y'",
        )
        .with_note(Span::new(0, 15, 16), "'y' declared here");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("error: use of undeclared identifier 'y'"));
        assert!(output.contains("note: 'y' declared here"));
    }

    #[test]
    fn test_format_diagnostic_with_fix() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file("test.c".to_string(), "prntf(\"hello\");\n".to_string());

        let diag = Diagnostic::new(
            Severity::Error,
            Span::new(0, 0, 5),
            "use of undeclared identifier 'prntf'",
        )
        .with_fix(Span::new(0, 0, 5), "printf");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("error: use of undeclared identifier 'prntf'"));
        assert!(output.contains("help: replace with 'printf'"));
    }

    #[test]
    fn test_format_diagnostic_dummy_span() {
        let sm = SourceMap::new();
        let diag = Diagnostic::new(Severity::Error, Span::DUMMY, "internal compiler error");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("<unknown>:0:0: error: internal compiler error"));
    }

    #[test]
    fn test_format_underline_width() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file("test.c".to_string(), "unknown_func();\n".to_string());

        // Span covering "unknown_func" (bytes 0..12)
        let diag = Diagnostic::new(Severity::Error, Span::new(0, 0, 12), "undeclared");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        // Should have ^ followed by 11 tildes (total underline width = 12)
        assert!(output.contains("^~~~~~~~~~~~"));
    }

    #[test]
    fn test_print_all_with_source_map() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file(
            "main.c".to_string(),
            "int main() {\n  return x;\n}\n".to_string(),
        );

        let mut engine = DiagnosticEngine::new();
        engine.error(Span::new(0, 23, 24), "use of undeclared identifier 'x'");

        // print_all writes to stderr; we verify it doesn't panic
        // (actual stderr output is not captured in unit tests)
        engine.print_all(&sm);
    }

    #[test]
    fn test_resolve_location_valid_span() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file("foo.c".to_string(), "abc\ndef\n".to_string());

        let location = DiagnosticEngine::resolve_location(&sm, &Span::new(0, 4, 7));
        assert_eq!(location, "foo.c:2:1");
    }

    #[test]
    fn test_resolve_location_dummy_span() {
        let sm = SourceMap::new();
        let location = DiagnosticEngine::resolve_location(&sm, &Span::DUMMY);
        assert_eq!(location, "<unknown>:0:0");
    }

    #[test]
    fn test_engine_help_no_count_increment() {
        let mut engine = DiagnosticEngine::new();
        engine.emit(Diagnostic::new(
            Severity::Help,
            Span::DUMMY,
            "try using --verbose",
        ));
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.warning_count(), 0);
        assert_eq!(engine.diagnostics().len(), 1);
    }

    #[test]
    fn test_format_second_line_span() {
        let mut sm = SourceMap::new();
        let _fid = sm.add_file(
            "test.c".to_string(),
            "int x = 0;\nint y = bad;\n".to_string(),
        );

        // "bad" is at byte offset 19..22 (line 2)
        let diag = Diagnostic::new(Severity::Error, Span::new(0, 19, 22), "undeclared 'bad'");

        let mut buf = Vec::new();
        DiagnosticEngine::format_diagnostic(&mut buf, &diag, &sm);
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("test.c:2:"));
        assert!(output.contains("int y = bad;"));
        assert!(output.contains("^~~"));
    }
}
