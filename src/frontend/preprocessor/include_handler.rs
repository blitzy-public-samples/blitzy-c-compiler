//! `#include` file resolution, include guard optimization, and circular detection.
//!
//! This module implements the include file handling for the BCC preprocessor:
//!
//! - **File Resolution**: Resolves `#include "..."` (user paths) and `#include <...>`
//!   (system paths) by searching configured include directories following GCC conventions.
//! - **Include Guard Optimization**: Detects the `#ifndef GUARD / #define GUARD / ... / #endif`
//!   pattern and skips re-inclusion when the guard macro is already defined.
//! - **Circular Include Detection**: Tracks the current include stack at the file level,
//!   architecturally distinct from paint-marker recursion protection which operates at the
//!   token/macro level.
//! - **`#pragma once` Support**: Tracks files marked with `#pragma once` to prevent
//!   re-inclusion.
//! - **PUA-Encoded Reading**: Uses [`crate::common::encoding::read_source_file`] for
//!   byte-exact round-tripping of non-UTF-8 bytes through PUA encoding (U+E080–U+E0FF).
//!
//! # Architectural Note
//!
//! Circular include detection operates at the **file level** — it tracks which files are
//! currently being processed in the include stack. This is architecturally separate from
//! the **paint-marker** system in [`super::paint_marker`], which operates at the
//! **token/macro level** to suppress recursive macro expansion per C11 §6.10.3.4.
//!
//! # Search Order
//!
//! For `#include "header.h"` (user include):
//! 1. Directory of the including file
//! 2. User include paths (`-I` directories) in order
//! 3. System include paths as fallback
//!
//! For `#include <header.h>` (system include):
//! 1. System include paths in order
//! 2. User include paths as fallback (GCC behavior)
//!
//! # Zero-Dependency
//!
//! This module uses only the Rust standard library and `crate::` internal modules.
//! No external crates are imported, in compliance with the zero-dependency mandate.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::common::diagnostics::{Diagnostic, Span};
use crate::common::encoding::read_source_file;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::source_map::SourceMap;

use super::{MacroDef, PPToken, PPTokenKind};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum include nesting depth, matching GCC's default (`-fmax-include-depth`).
///
/// This prevents stack overflow and resource exhaustion from extremely deep
/// include chains, which can occur in pathological or misconfigured builds.
const DEFAULT_MAX_INCLUDE_DEPTH: usize = 200;

// ---------------------------------------------------------------------------
// IncludeError — error types for include processing
// ---------------------------------------------------------------------------

/// Errors that can occur during `#include` file processing.
///
/// Each variant represents a distinct failure mode in include resolution,
/// circular detection, or file reading. These errors are converted to
/// compiler diagnostics via [`IncludeError::to_diagnostic`] for user-facing
/// error reporting.
#[derive(Debug)]
pub enum IncludeError {
    /// Circular include detected — the file at the given path is already
    /// being processed in the current include chain.
    ///
    /// Example: `a.h` includes `b.h` which includes `a.h` → Circular.
    Circular(PathBuf),

    /// Include depth exceeded the maximum limit (default: 200).
    ///
    /// The `usize` value is the current depth at the point of failure.
    TooDeep(usize),

    /// The specified header file was not found in any search path.
    ///
    /// The `String` contains the header name as it appeared in the
    /// `#include` directive (e.g., `"missing.h"` or `<noexist.h>`).
    NotFound(String),

    /// An I/O error occurred while reading the include file.
    ///
    /// Wraps the underlying [`std::io::Error`] from filesystem operations.
    IoError(std::io::Error),
}

impl fmt::Display for IncludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncludeError::Circular(path) => {
                write!(f, "circular #include detected: '{}'", path.display())
            }
            IncludeError::TooDeep(depth) => {
                write!(
                    f,
                    "#include nested too deeply ({} levels, maximum is {})",
                    depth, DEFAULT_MAX_INCLUDE_DEPTH
                )
            }
            IncludeError::NotFound(header) => {
                write!(f, "'{}': file not found", header)
            }
            IncludeError::IoError(err) => {
                write!(f, "error reading include file: {}", err)
            }
        }
    }
}

impl std::error::Error for IncludeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IncludeError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IncludeError {
    fn from(err: std::io::Error) -> Self {
        IncludeError::IoError(err)
    }
}

impl IncludeError {
    /// Convert this include error into a compiler [`Diagnostic`] at the given
    /// source span.
    ///
    /// When a [`SourceMap`] reference is provided, the diagnostic is enriched
    /// with contextual notes (e.g., the name of the file that triggered the
    /// include). This exercises the diagnostic builder pattern:
    /// [`Diagnostic::error`], [`Diagnostic::warning`], and
    /// [`Diagnostic::with_note`].
    ///
    /// # Arguments
    ///
    /// - `span`: The source location of the `#include` directive that caused
    ///   the error.  Uses [`Span::file_id`], [`Span::start`], and [`Span::end`]
    ///   for positioning.
    /// - `source_map`: Optional source map for enriching the diagnostic with
    ///   filename context.
    pub fn to_diagnostic(&self, span: Span, source_map: Option<&SourceMap>) -> Diagnostic {
        let msg = self.to_string();
        match self {
            IncludeError::Circular(path) => {
                let mut diag = Diagnostic::error(span, &msg);
                // Add a note showing which file triggered the circular include.
                // Use SourceMap::get_filename to retrieve the including file name.
                if let Some(sm) = source_map {
                    if let Some(filename) = sm.get_filename(span.file_id) {
                        diag = diag.with_note(
                            Span::new(span.file_id, span.start, span.end),
                            format!(
                                "'{}' is already being included from '{}'",
                                path.display(),
                                filename
                            ),
                        );
                    }
                }
                diag
            }
            IncludeError::TooDeep(_) => Diagnostic::error(span, msg),
            IncludeError::NotFound(_) => Diagnostic::error(span, msg),
            IncludeError::IoError(_) => Diagnostic::error(span, msg),
        }
    }

    /// Create a warning diagnostic for include-related issues that do not
    /// prevent compilation (e.g., deprecated header names, redundant includes).
    pub fn as_warning(&self, span: Span) -> Diagnostic {
        Diagnostic::warning(span, self.to_string())
    }
}

// ---------------------------------------------------------------------------
// IncludeHandler — the core include resolution and tracking engine
// ---------------------------------------------------------------------------

/// Handles `#include` file resolution, include guard optimization, and circular
/// include detection.
///
/// The handler maintains:
/// - **Search paths**: User (`-I`) and system include directories, searched in
///   the order specified by GCC conventions.
/// - **Include stack**: For circular include detection at the file level.
/// - **Guard file map**: Files with detected include guards (`#ifndef`/`#define`/`#endif`
///   pattern), mapped to their guard macro name.
/// - **Pragma-once set**: Files marked with `#pragma once` that must never be
///   re-included.
///
/// # Usage
///
/// ```ignore
/// let mut handler = IncludeHandler::new(user_paths, system_paths);
///
/// // Resolve and process an include
/// if let Some(path) = handler.resolve_include("header.h", false, &current_file) {
///     if !handler.should_skip_file(&path, &defined_macros) {
///         handler.push_include(&path)?;
///         let content = handler.read_include_file(&path)?;
///         // ... process content ...
///         handler.pop_include();
///     }
/// }
/// ```
pub struct IncludeHandler {
    /// User include paths from `-I` flags (searched for `#include "..."`).
    user_paths: Vec<PathBuf>,

    /// System include paths (searched for `#include <...>`).
    system_paths: Vec<PathBuf>,

    /// Current include stack for circular detection — file paths currently
    /// being processed. A file appears in this stack from [`push_include`]
    /// until [`pop_include`] is called.
    include_stack: Vec<PathBuf>,

    /// Files that have been fully included and have detected include guards.
    /// Maps the canonical file path to the guard macro name.
    guarded_files: FxHashMap<PathBuf, String>,

    /// Files marked with `#pragma once` — these are never re-included.
    pragma_once_files: FxHashSet<PathBuf>,

    /// Maximum include depth (prevents very deep include chains).
    /// Default: 200 (matching GCC's default `-fmax-include-depth`).
    max_include_depth: usize,
}

impl IncludeHandler {
    // -- Construction -------------------------------------------------------

    /// Create a new `IncludeHandler` with the given user and system include paths.
    ///
    /// - `user_paths`: Directories from `-I` flags, searched for `#include "..."`.
    /// - `system_paths`: System include directories, searched for `#include <...>`.
    ///
    /// The include stack, guarded files, and pragma-once set are initialized empty.
    /// The maximum include depth defaults to 200 (GCC's default).
    pub fn new(user_paths: Vec<PathBuf>, system_paths: Vec<PathBuf>) -> Self {
        IncludeHandler {
            user_paths,
            system_paths,
            include_stack: Vec::new(),
            guarded_files: FxHashMap::default(),
            pragma_once_files: FxHashSet::default(),
            max_include_depth: DEFAULT_MAX_INCLUDE_DEPTH,
        }
    }

    // -- Path Configuration ------------------------------------------------

    /// Add a user include path (from a `-I` flag).
    ///
    /// User paths are searched for `#include "..."` directives after the
    /// directory of the including file. They are also searched as a fallback
    /// for `#include <...>` directives after system paths.
    pub fn add_user_path(&mut self, path: PathBuf) {
        self.user_paths.push(path);
    }

    /// Add a system include path.
    ///
    /// System paths are the primary search location for `#include <...>`
    /// directives. They are also searched as a fallback for `#include "..."`
    /// directives after user paths.
    pub fn add_system_path(&mut self, path: PathBuf) {
        self.system_paths.push(path);
    }

    // -- File Resolution ---------------------------------------------------

    /// Resolve an `#include` directive to a file path.
    ///
    /// # Arguments
    ///
    /// - `header`: The header name (e.g., `"stdio.h"` or `"my/header.h"`).
    /// - `is_system`: `true` for `#include <...>`, `false` for `#include "..."`.
    /// - `including_file`: Path of the file containing the `#include` directive.
    ///
    /// # Search Order
    ///
    /// For `#include "..."` (`is_system = false`):
    /// 1. Relative to the directory of `including_file`
    /// 2. Each directory in `user_paths` (from `-I` flags) in order
    /// 3. Each directory in `system_paths` as fallback
    ///
    /// For `#include <...>` (`is_system = true`):
    /// 1. Each directory in `system_paths` in order
    /// 2. Each directory in `user_paths` as fallback (GCC behavior)
    ///
    /// Returns the canonical path of the first matching file, or `None` if
    /// the header cannot be found in any search path.
    pub fn resolve_include(
        &self,
        header: &str,
        is_system: bool,
        including_file: &Path,
    ) -> Option<PathBuf> {
        if is_system {
            // System include: search system paths first, then user paths
            if let Some(found) = self.search_paths(&self.system_paths, header) {
                return Some(found);
            }
            if let Some(found) = self.search_paths(&self.user_paths, header) {
                return Some(found);
            }
        } else {
            // User include: search relative to the including file's directory first
            if let Some(parent) = including_file.parent() {
                let candidate = parent.join(header);
                if candidate.exists() && candidate.is_file() {
                    return Some(canonicalize_path(&candidate));
                }
            }
            // Then user paths (from -I flags)
            if let Some(found) = self.search_paths(&self.user_paths, header) {
                return Some(found);
            }
            // Then system paths as fallback
            if let Some(found) = self.search_paths(&self.system_paths, header) {
                return Some(found);
            }
        }

        // Cross-compilation fallback: when targeting a non-native arch,
        // the host system may not have cross-compilation headers installed.
        // `gnu/stubs-{32,x32}.h` files are architecture-variant stub
        // manifests that only define `__stub_FUNCTION` macros — they are
        // functionally interchangeable across architectures.  Fall back to
        // any existing stubs variant file from the system include paths.
        if header.starts_with("gnu/stubs-") && header.ends_with(".h") {
            for variant in &["gnu/stubs-64.h", "gnu/stubs-32.h", "gnu/stubs-x32.h"] {
                if let Some(found) = self.search_paths(&self.system_paths, variant) {
                    return Some(found);
                }
            }
        }

        None
    }

    /// Resolve a `#include_next` directive.
    ///
    /// `#include_next` searches for the header starting from the directory
    /// **after** the directory that contains `including_file` in the
    /// include-path list.  This is a GCC extension used by system headers
    /// (e.g., `/usr/include/limits.h` chains to the compiler's built-in
    /// `limits.h` via `#include_next`).
    pub fn resolve_include_next(&self, header: &str, including_file: &Path) -> Option<PathBuf> {
        // Determine which directory the current file lives in.
        let current_dir = including_file
            .parent()
            .and_then(|p| std::fs::canonicalize(p).ok());

        // Build a combined search list: user paths, then system paths.
        let all_paths: Vec<&PathBuf> = self
            .user_paths
            .iter()
            .chain(self.system_paths.iter())
            .collect();

        // Find the index of the current directory in the search list.
        let mut skip_until = 0;
        if let Some(ref cur) = current_dir {
            for (i, p) in all_paths.iter().enumerate() {
                if let Ok(canon) = std::fs::canonicalize(p) {
                    if &canon == cur {
                        skip_until = i + 1; // start searching AFTER this entry
                        break;
                    }
                }
            }
        }

        // Search from skip_until onwards.
        for p in &all_paths[skip_until..] {
            let candidate = p.join(header);
            if candidate.exists() && candidate.is_file() {
                return Some(canonicalize_path(&candidate));
            }
        }
        None
    }

    /// Search a list of directories for a header file.
    ///
    /// Returns the canonical path of the first directory/header combination
    /// that exists as a regular file, or `None` if not found.
    fn search_paths(&self, paths: &[PathBuf], header: &str) -> Option<PathBuf> {
        for dir in paths {
            let candidate = dir.join(header);
            if candidate.exists() && candidate.is_file() {
                return Some(canonicalize_path(&candidate));
            }
        }
        None
    }

    // -- Circular Include Detection ----------------------------------------

    /// Push a file onto the include stack, checking for circular includes
    /// and depth limits.
    ///
    /// Must be called before processing an included file's content. Call
    /// [`pop_include`](IncludeHandler::pop_include) when the file is fully
    /// processed.
    ///
    /// # Errors
    ///
    /// - [`IncludeError::Circular`] — if `path` is already in the include stack
    ///   (i.e., a circular include chain has been detected).
    /// - [`IncludeError::TooDeep`] — if the current stack depth would exceed
    ///   `max_include_depth` (default: 200).
    pub fn push_include(&mut self, path: &Path) -> Result<(), IncludeError> {
        let canonical = canonicalize_path(path);

        // Check for circular includes by scanning the current stack.
        // This is O(n) in stack depth, which is acceptable since the maximum
        // depth is bounded (200 by default).
        for entry in &self.include_stack {
            if *entry == canonical {
                return Err(IncludeError::Circular(canonical));
            }
        }

        // Check depth limit before pushing.
        if self.include_stack.len() >= self.max_include_depth {
            return Err(IncludeError::TooDeep(self.include_stack.len()));
        }

        self.include_stack.push(canonical);
        Ok(())
    }

    /// Pop the current file from the include stack.
    ///
    /// Called when processing of an included file is complete. Must be paired
    /// with a preceding [`push_include`](IncludeHandler::push_include) call.
    ///
    /// If the stack is empty (defensive), this is a no-op rather than panicking.
    pub fn pop_include(&mut self) {
        self.include_stack.pop();
    }

    // -- Include Guard Optimization ----------------------------------------

    /// Register a file's include guard macro name.
    ///
    /// Called by the preprocessor when it detects the canonical include guard
    /// pattern in a fully-processed file:
    ///
    /// ```text
    /// #ifndef GUARD_MACRO
    /// #define GUARD_MACRO
    /// ... (file content) ...
    /// #endif
    /// ```
    ///
    /// Once registered, subsequent includes of this file will be skipped
    /// if the guard macro is already defined (see [`should_skip_guarded`]).
    ///
    /// The file path is canonicalized for consistent matching regardless of
    /// how the file is referenced in different `#include` directives.
    pub fn register_guard(&mut self, path: &Path, guard_macro: String) {
        let canonical = canonicalize_path(path);
        self.guarded_files.insert(canonical, guard_macro);
    }

    /// Check if a file should be skipped due to its include guard macro
    /// already being defined.
    ///
    /// Returns `true` if **both** of the following conditions hold:
    /// 1. The file has a registered include guard (via [`register_guard`]).
    /// 2. The guard macro name is present as a key in `defined_macros`.
    ///
    /// The check accesses [`MacroDef::name`] to verify the macro definition
    /// matches the expected guard name.
    pub fn should_skip_guarded(
        &self,
        path: &Path,
        defined_macros: &FxHashMap<String, MacroDef>,
    ) -> bool {
        let canonical = canonicalize_path(path);
        if let Some(guard_name) = self.guarded_files.get(&canonical) {
            // Check if the guard macro is defined. Access MacroDef.name
            // through the HashMap to verify the macro identity matches.
            if let Some(macro_def) = defined_macros.get(guard_name) {
                return macro_def.name == *guard_name;
            }
        }
        false
    }

    // -- #pragma once Tracking ---------------------------------------------

    /// Mark a file with `#pragma once` — it will never be re-included.
    ///
    /// The canonical path is stored to ensure consistent matching regardless
    /// of how the file is referenced in different `#include` directives.
    pub fn mark_pragma_once(&mut self, path: &Path) {
        let canonical = canonicalize_path(path);
        self.pragma_once_files.insert(canonical);
    }

    /// Check if a file has been marked with `#pragma once`.
    ///
    /// Returns `true` if the canonical form of `path` has been registered
    /// via [`mark_pragma_once`].
    pub fn is_pragma_once(&self, path: &Path) -> bool {
        let canonical = canonicalize_path(path);
        self.pragma_once_files.contains(&canonical)
    }

    /// Check if a file should be skipped entirely — either due to `#pragma once`
    /// or an include guard with a currently-defined macro.
    ///
    /// This is the combined check that the preprocessor should call before
    /// processing a `#include` directive. It short-circuits on `#pragma once`
    /// (cheapest check) before falling through to the guard check.
    pub fn should_skip_file(
        &self,
        path: &Path,
        defined_macros: &FxHashMap<String, MacroDef>,
    ) -> bool {
        self.is_pragma_once(path) || self.should_skip_guarded(path, defined_macros)
    }

    // -- File Reading Integration ------------------------------------------

    /// Read an include file with PUA encoding for non-UTF-8 byte preservation.
    ///
    /// Uses [`crate::common::encoding::read_source_file`] to ensure that
    /// non-UTF-8 bytes (0x80–0xFF) in C header files are mapped to Private
    /// Use Area code points (U+E080–U+E0FF) for round-trip fidelity through
    /// the compilation pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`IncludeError::IoError`] if the file cannot be read due to
    /// filesystem errors (file not found, permission denied, etc.).
    pub fn read_include_file(&self, path: &Path) -> Result<String, IncludeError> {
        read_source_file(path).map_err(IncludeError::IoError)
    }

    // -- Source Map Integration (utility) -----------------------------------

    /// Read an include file and register it with the [`SourceMap`].
    ///
    /// This convenience method combines file reading with source map
    /// registration, returning both the assigned file ID and the file
    /// contents. The file ID can then be used in [`Span`] values for
    /// diagnostics originating from this included file.
    ///
    /// Uses [`SourceMap::add_file`] for file registration and
    /// [`read_source_file`] for PUA-encoded reading.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the include file.
    /// - `source_map`: The source map to register the file with.
    ///
    /// # Returns
    ///
    /// A tuple of `(file_id, contents)` where `file_id` is the stable ID
    /// assigned by the source map for subsequent span and diagnostic
    /// references.
    pub fn read_and_register(
        &self,
        path: &Path,
        source_map: &mut SourceMap,
    ) -> Result<(u32, String), IncludeError> {
        let contents = self.read_include_file(path)?;
        let filename = path.to_string_lossy().into_owned();
        let file_id = source_map.add_file(filename, contents.clone());
        Ok((file_id, contents))
    }

    /// Look up the filename for a file ID from the source map.
    ///
    /// Utility method that delegates to [`SourceMap::get_filename`] for
    /// constructing diagnostic messages about included files.
    pub fn get_filename_from_source_map<'a>(
        &self,
        source_map: &'a SourceMap,
        file_id: u32,
    ) -> Option<&'a str> {
        source_map.get_filename(file_id)
    }

    // -- Include stack introspection (utility) ------------------------------

    /// Returns the current include stack depth.
    ///
    /// A depth of 0 means no files are currently being processed via
    /// `#include`. Each [`push_include`] increments this, each
    /// [`pop_include`] decrements it.
    #[inline]
    pub fn depth(&self) -> usize {
        self.include_stack.len()
    }

    /// Returns the current include stack as a slice of canonical paths.
    ///
    /// Useful for diagnostic messages that need to show the full include
    /// chain leading to an error.
    #[inline]
    pub fn include_stack(&self) -> &[PathBuf] {
        &self.include_stack
    }
}

// ---------------------------------------------------------------------------
// Path Canonicalization
// ---------------------------------------------------------------------------

/// Normalize a file path for consistent comparison.
///
/// Uses [`std::fs::canonicalize`] if the file exists on disk (resolves
/// symlinks, removes `.` and `..` components, produces an absolute path).
/// Falls back to manual normalization for paths that don't exist yet,
/// which resolves `.` and `..` components without filesystem access.
///
/// This ensures the same physical file is recognized regardless of how it
/// is referenced in different `#include` directives (e.g.,
/// `"../include/foo.h"` vs `"include/foo.h"` from different directories).
fn canonicalize_path(path: &Path) -> PathBuf {
    // Try filesystem-level canonicalization first — this resolves symlinks,
    // removes . and .., and produces an absolute path.
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }

    // Fallback: manual normalization for paths that don't exist on disk yet
    // or when canonicalize fails (e.g., broken symlinks, permission issues).
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                // ".." — pop the last normal component if possible.
                // If nothing to pop, keep the ".." (e.g., for relative paths
                // that start with "../").
                if !result
                    .components()
                    .next_back()
                    .map_or(false, |c| matches!(c, std::path::Component::Normal(_)))
                {
                    result.push(component);
                } else {
                    result.pop();
                }
            }
            std::path::Component::CurDir => {
                // "." — skip entirely, it's a no-op in path resolution.
            }
            _ => {
                // RootDir, Prefix, or Normal — keep as-is.
                result.push(component);
            }
        }
    }

    // If the result is empty (e.g., input was just "."), use "." as the path.
    if result.as_os_str().is_empty() {
        result.push(".");
    }

    result
}

// ---------------------------------------------------------------------------
// Include Guard Detection
// ---------------------------------------------------------------------------

/// Analyze a file's preprocessing token stream to detect an include guard pattern.
///
/// Returns `Some(guard_name)` if the file follows the canonical include guard
/// pattern:
///
/// ```text
/// #ifndef GUARD_MACRO
/// #define GUARD_MACRO
/// ... (arbitrary content, possibly including nested #if blocks) ...
/// #endif
/// ```
///
/// The detection algorithm requires:
/// 1. The **first directive** in the file is `#ifndef <identifier>`.
/// 2. The **second directive** is `#define <same identifier>`.
/// 3. The **last directive** is `#endif` that closes the outer `#ifndef`.
/// 4. No `#else` or `#elif` at the **top nesting level** (depth 1).
///
/// Nested `#if`/`#ifdef`/`#ifndef` blocks inside the guard are allowed and
/// correctly tracked via depth counting.
///
/// # Arguments
///
/// - `tokens`: The full preprocessing token stream of the file. Each token's
///   [`PPToken::kind`] and [`PPToken::text`] fields are inspected to identify
///   preprocessor directives.
///
/// # Returns
///
/// `Some(guard_name)` if a valid include guard is detected, `None` otherwise.
pub fn detect_include_guard(tokens: &[PPToken]) -> Option<String> {
    if tokens.is_empty() {
        return None;
    }

    // Phase 1: Extract all preprocessor directives from the token stream.
    //
    // A directive is a '#' punctuator token at the start of a logical line
    // (i.e., preceded only by whitespace/newlines or at the very beginning
    // of the token stream), followed by an identifier (the directive name),
    // and optionally followed by another identifier (the directive argument).
    //
    // We collect: (directive_name, optional_argument)
    let mut directives: Vec<(String, Option<String>)> = Vec::new();
    let mut i: usize = 0;
    let len = tokens.len();

    // Track whether we are at the start of a logical line.
    let mut at_line_start = true;

    while i < len {
        let token = &tokens[i];

        match token.kind {
            PPTokenKind::Newline => {
                at_line_start = true;
                i += 1;
            }
            PPTokenKind::Whitespace => {
                // Whitespace does not change line-start status.
                i += 1;
            }
            PPTokenKind::EndOfFile => {
                break;
            }
            PPTokenKind::Punctuator if token.text == "#" && at_line_start => {
                // Found a directive: '#' at the start of a line.
                i += 1;

                // Skip whitespace between '#' and the directive name.
                while i < len && tokens[i].kind == PPTokenKind::Whitespace {
                    i += 1;
                }

                if i < len && tokens[i].kind == PPTokenKind::Identifier {
                    let directive_name = tokens[i].text.clone();
                    i += 1;

                    // Skip whitespace between directive name and argument.
                    while i < len && tokens[i].kind == PPTokenKind::Whitespace {
                        i += 1;
                    }

                    // Capture the argument if it is an identifier.
                    let arg = if i < len && tokens[i].kind == PPTokenKind::Identifier {
                        let a = tokens[i].text.clone();
                        i += 1;
                        Some(a)
                    } else {
                        None
                    };

                    directives.push((directive_name, arg));
                }
                // After processing a directive, skip to end of line.
                while i < len
                    && tokens[i].kind != PPTokenKind::Newline
                    && tokens[i].kind != PPTokenKind::EndOfFile
                {
                    i += 1;
                }
                at_line_start = false;
            }
            _ => {
                at_line_start = false;
                i += 1;
            }
        }
    }

    // Phase 2: Validate the include guard pattern.
    //
    // Need at least 3 directives: #ifndef, #define, #endif
    if directives.len() < 3 {
        return None;
    }

    // Check first directive: must be #ifndef <IDENTIFIER>
    let (ref first_name, ref first_arg) = directives[0];
    if first_name != "ifndef" {
        return None;
    }
    let guard_name = match first_arg {
        Some(name) => name.clone(),
        None => return None,
    };

    // Check second directive: must be #define <same IDENTIFIER>
    let (ref second_name, ref second_arg) = directives[1];
    if second_name != "define" {
        return None;
    }
    match second_arg {
        Some(name) if *name == guard_name => { /* Match — continue */ }
        _ => return None,
    }

    // Check last directive: must be #endif
    let (ref last_name, _) = directives[directives.len() - 1];
    if last_name != "endif" {
        return None;
    }

    // Phase 3: Verify nesting correctness.
    //
    // The last #endif must close the outer #ifndef (depth returns to 0).
    // Also verify that there is no #else or #elif at the top nesting level
    // (depth == 1), which would indicate a conditional split rather than a
    // simple include guard.
    let mut depth: i32 = 0;
    for (idx, (name, _)) in directives.iter().enumerate() {
        match name.as_str() {
            "if" | "ifdef" | "ifndef" => {
                depth += 1;
            }
            "endif" => {
                depth -= 1;
                if depth == 0 {
                    // The outer #ifndef is now closed. This must be the
                    // very last directive for a valid include guard.
                    if idx != directives.len() - 1 {
                        return None;
                    }
                }
                if depth < 0 {
                    // More #endif than #if — malformed, not a valid guard.
                    return None;
                }
            }
            "else" | "elif" => {
                // #else or #elif at the outermost nesting level (depth 1)
                // means the file has conditional branches at the guard level.
                // This is NOT a simple include guard pattern.
                if depth == 1 {
                    return None;
                }
            }
            _ => { /* Other directives (#define, #include, etc.) — no nesting effect */ }
        }
    }

    // Verify balanced nesting (depth should be exactly 0).
    if depth != 0 {
        return None;
    }

    Some(guard_name)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;

    // Helper: create a PPToken with dummy span for testing.
    fn tok(kind: PPTokenKind, text: &str) -> PPToken {
        PPToken::new(kind, text, Span::dummy())
    }

    // -----------------------------------------------------------------------
    // IncludeHandler construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_creates_empty_handler() {
        let handler = IncludeHandler::new(vec![], vec![]);
        assert_eq!(handler.depth(), 0);
        assert_eq!(handler.max_include_depth, DEFAULT_MAX_INCLUDE_DEPTH);
        assert!(handler.include_stack().is_empty());
    }

    #[test]
    fn test_new_with_paths() {
        let user = vec![PathBuf::from("/usr/local/include")];
        let sys = vec![PathBuf::from("/usr/include")];
        let handler = IncludeHandler::new(user.clone(), sys.clone());
        assert_eq!(handler.user_paths, user);
        assert_eq!(handler.system_paths, sys);
    }

    #[test]
    fn test_add_user_path() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        handler.add_user_path(PathBuf::from("/my/include"));
        assert_eq!(handler.user_paths.len(), 1);
        assert_eq!(handler.user_paths[0], PathBuf::from("/my/include"));
    }

    #[test]
    fn test_add_system_path() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        handler.add_system_path(PathBuf::from("/sys/include"));
        assert_eq!(handler.system_paths.len(), 1);
        assert_eq!(handler.system_paths[0], PathBuf::from("/sys/include"));
    }

    // -----------------------------------------------------------------------
    // Circular include detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_pop_include() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        assert!(handler.push_include(Path::new("/tmp/a.h")).is_ok());
        assert_eq!(handler.depth(), 1);
        handler.pop_include();
        assert_eq!(handler.depth(), 0);
    }

    #[test]
    fn test_circular_include_detected() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path_a = Path::new("/tmp/test_a.h");
        let path_b = Path::new("/tmp/test_b.h");

        assert!(handler.push_include(path_a).is_ok());
        assert!(handler.push_include(path_b).is_ok());
        // Pushing A again should fail — circular.
        let result = handler.push_include(path_a);
        assert!(matches!(result, Err(IncludeError::Circular(_))));
    }

    #[test]
    fn test_circular_after_pop_allows_reentry() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path_a = Path::new("/tmp/test_reentry.h");

        assert!(handler.push_include(path_a).is_ok());
        handler.pop_include();
        // After popping, the same file can be pushed again.
        assert!(handler.push_include(path_a).is_ok());
    }

    #[test]
    fn test_include_depth_limit() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        handler.max_include_depth = 3;

        assert!(handler.push_include(Path::new("/tmp/d1.h")).is_ok());
        assert!(handler.push_include(Path::new("/tmp/d2.h")).is_ok());
        assert!(handler.push_include(Path::new("/tmp/d3.h")).is_ok());
        let result = handler.push_include(Path::new("/tmp/d4.h"));
        assert!(matches!(result, Err(IncludeError::TooDeep(3))));
    }

    #[test]
    fn test_pop_on_empty_stack_is_noop() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        handler.pop_include(); // Should not panic.
        assert_eq!(handler.depth(), 0);
    }

    // -----------------------------------------------------------------------
    // Include guard tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_and_skip_guarded() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path = PathBuf::from("/tmp/guarded.h");

        // No macros defined — should not skip.
        let empty_macros: FxHashMap<String, MacroDef> = FxHashMap::default();
        handler.register_guard(&path, "GUARDED_H".to_string());
        assert!(!handler.should_skip_guarded(&path, &empty_macros));

        // Define the guard macro — should now skip.
        let mut macros: FxHashMap<String, MacroDef> = FxHashMap::default();
        macros.insert(
            "GUARDED_H".to_string(),
            MacroDef {
                name: "GUARDED_H".to_string(),
                kind: super::super::MacroKind::ObjectLike,
                replacement: vec![],
                is_predefined: false,
                definition_span: Span::dummy(),
            },
        );
        assert!(handler.should_skip_guarded(&path, &macros));
    }

    #[test]
    fn test_unregistered_file_not_skipped() {
        let handler = IncludeHandler::new(vec![], vec![]);
        let path = PathBuf::from("/tmp/not_registered.h");
        let macros: FxHashMap<String, MacroDef> = FxHashMap::default();
        assert!(!handler.should_skip_guarded(&path, &macros));
    }

    // -----------------------------------------------------------------------
    // #pragma once tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pragma_once_mark_and_check() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path = PathBuf::from("/tmp/once.h");

        assert!(!handler.is_pragma_once(&path));
        handler.mark_pragma_once(&path);
        assert!(handler.is_pragma_once(&path));
    }

    #[test]
    fn test_should_skip_file_pragma_once() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path = PathBuf::from("/tmp/skip_once.h");
        let macros: FxHashMap<String, MacroDef> = FxHashMap::default();

        assert!(!handler.should_skip_file(&path, &macros));
        handler.mark_pragma_once(&path);
        assert!(handler.should_skip_file(&path, &macros));
    }

    #[test]
    fn test_should_skip_file_guard_defined() {
        let mut handler = IncludeHandler::new(vec![], vec![]);
        let path = PathBuf::from("/tmp/skip_guard.h");

        handler.register_guard(&path, "SKIP_GUARD_H".to_string());

        let mut macros: FxHashMap<String, MacroDef> = FxHashMap::default();
        macros.insert(
            "SKIP_GUARD_H".to_string(),
            MacroDef {
                name: "SKIP_GUARD_H".to_string(),
                kind: super::super::MacroKind::ObjectLike,
                replacement: vec![],
                is_predefined: false,
                definition_span: Span::dummy(),
            },
        );
        assert!(handler.should_skip_file(&path, &macros));
    }

    // -----------------------------------------------------------------------
    // detect_include_guard tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_guard_valid_pattern() {
        // #ifndef MY_HEADER_H
        // #define MY_HEADER_H
        // int x;
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "MY_HEADER_H"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "MY_HEADER_H"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Identifier, "int"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "x"),
            tok(PPTokenKind::Punctuator, ";"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(
            detect_include_guard(&tokens),
            Some("MY_HEADER_H".to_string())
        );
    }

    #[test]
    fn test_detect_guard_with_nested_if() {
        // #ifndef GUARD
        // #define GUARD
        // #ifdef INNER
        // int y;
        // #endif
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifdef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "INNER"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Identifier, "int"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "y"),
            tok(PPTokenKind::Punctuator, ";"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), Some("GUARD".to_string()));
    }

    #[test]
    fn test_detect_guard_rejects_else_at_top_level() {
        // #ifndef GUARD
        // #define GUARD
        // #else
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "else"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), None);
    }

    #[test]
    fn test_detect_guard_rejects_elif_at_top_level() {
        // #ifndef GUARD
        // #define GUARD
        // #elif defined(OTHER)
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "elif"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "OTHER"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), None);
    }

    #[test]
    fn test_detect_guard_allows_nested_else() {
        // #ifndef GUARD
        // #define GUARD
        // #ifdef INNER
        // #else
        // #endif
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifdef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "INNER"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "else"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), Some("GUARD".to_string()));
    }

    #[test]
    fn test_detect_guard_mismatched_names() {
        // #ifndef FOO
        // #define BAR  <-- different name
        // #endif
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "FOO"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "BAR"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), None);
    }

    #[test]
    fn test_detect_guard_empty_tokens() {
        assert_eq!(detect_include_guard(&[]), None);
    }

    #[test]
    fn test_detect_guard_no_directives() {
        let tokens = vec![
            tok(PPTokenKind::Identifier, "int"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "x"),
            tok(PPTokenKind::Punctuator, ";"),
        ];
        assert_eq!(detect_include_guard(&tokens), None);
    }

    #[test]
    fn test_detect_guard_first_directive_not_ifndef() {
        // #define FOO — first directive is not #ifndef
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "FOO"),
            tok(PPTokenKind::Newline, "\n"),
        ];
        assert_eq!(detect_include_guard(&tokens), None);
    }

    #[test]
    fn test_detect_guard_content_after_endif() {
        // #ifndef GUARD
        // #define GUARD
        // #endif
        // int extra;  <-- content after #endif
        // But only directives after #endif matter — non-directive content is OK
        // Wait, actually content after the closing #endif should be fine
        // as long as there are no additional DIRECTIVES after it.
        let tokens = vec![
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "ifndef"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "define"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "GUARD"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Punctuator, "#"),
            tok(PPTokenKind::Identifier, "endif"),
            tok(PPTokenKind::Newline, "\n"),
            tok(PPTokenKind::Identifier, "int"),
            tok(PPTokenKind::Whitespace, " "),
            tok(PPTokenKind::Identifier, "extra"),
            tok(PPTokenKind::Punctuator, ";"),
        ];
        // This IS a valid guard — the extra content is outside the #endif
        // but there are no additional directives. The file's guard is valid.
        assert_eq!(detect_include_guard(&tokens), Some("GUARD".to_string()));
    }

    // -----------------------------------------------------------------------
    // IncludeError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_include_error_display() {
        let err = IncludeError::NotFound("missing.h".to_string());
        let msg = err.to_string();
        assert!(msg.contains("missing.h"));
        assert!(msg.contains("file not found"));

        let err = IncludeError::Circular(PathBuf::from("/tmp/circ.h"));
        assert!(err.to_string().contains("circular"));

        let err = IncludeError::TooDeep(200);
        assert!(err.to_string().contains("200"));

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let err = IncludeError::IoError(io_err);
        assert!(err.to_string().contains("error reading"));
    }

    #[test]
    fn test_include_error_to_diagnostic() {
        let err = IncludeError::NotFound("missing.h".to_string());
        let span = Span::new(1, 10, 25);
        let diag = err.to_diagnostic(span, None);
        assert_eq!(diag.span.file_id, 1);
        assert_eq!(diag.span.start, 10);
        assert_eq!(diag.span.end, 25);
        assert!(diag.message.contains("missing.h"));
    }

    #[test]
    fn test_include_error_to_diagnostic_with_source_map() {
        let err = IncludeError::Circular(PathBuf::from("/tmp/loop.h"));
        let mut source_map = SourceMap::new();
        let _fid = source_map.add_file("main.c".to_string(), "content".to_string());
        let span = Span::new(0, 5, 15);
        let diag = err.to_diagnostic(span, Some(&source_map));
        // Should have a note about the including file.
        assert!(!diag.notes.is_empty());
        assert!(diag.notes[0].message.contains("main.c"));
    }

    #[test]
    fn test_include_error_as_warning() {
        let err = IncludeError::NotFound("optional.h".to_string());
        let span = Span::new(0, 0, 10);
        let diag = err.as_warning(span);
        assert_eq!(diag.severity, crate::common::diagnostics::Severity::Warning);
    }

    #[test]
    fn test_include_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let include_err: IncludeError = io_err.into();
        assert!(matches!(include_err, IncludeError::IoError(_)));
    }

    // -----------------------------------------------------------------------
    // Path canonicalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_canonicalize_removes_dot() {
        let result = canonicalize_path(Path::new("/a/./b/c"));
        assert_eq!(result, PathBuf::from("/a/b/c"));
    }

    #[test]
    fn test_canonicalize_resolves_dotdot() {
        let result = canonicalize_path(Path::new("/a/b/../c"));
        assert_eq!(result, PathBuf::from("/a/c"));
    }

    #[test]
    fn test_canonicalize_preserves_leading_dotdot() {
        // For relative paths that start with "..", the ".." should be preserved
        // since there's nothing to pop.
        let result = canonicalize_path(Path::new("../relative/path"));
        assert_eq!(result, PathBuf::from("../relative/path"));
    }

    #[test]
    fn test_canonicalize_dot_resolves_to_absolute() {
        // "." exists on disk, so `fs::canonicalize` resolves it to an absolute path.
        let result = canonicalize_path(Path::new("."));
        assert!(
            result.is_absolute(),
            "Expected absolute path, got: {:?}",
            result
        );
    }

    #[test]
    fn test_canonicalize_nonexistent_dot_dot_sequence() {
        // A non-existent path forces the manual fallback.
        let result = canonicalize_path(Path::new("/nonexistent_xyz/./sub/../file.h"));
        assert_eq!(result, PathBuf::from("/nonexistent_xyz/file.h"));
    }

    // -----------------------------------------------------------------------
    // Source map integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_filename_from_source_map() {
        let handler = IncludeHandler::new(vec![], vec![]);
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("test.h".to_string(), "int x;".to_string());
        let name = handler.get_filename_from_source_map(&source_map, file_id);
        assert_eq!(name, Some("test.h"));
    }

    #[test]
    fn test_get_filename_invalid_id() {
        let handler = IncludeHandler::new(vec![], vec![]);
        let source_map = SourceMap::new();
        let name = handler.get_filename_from_source_map(&source_map, 999);
        assert_eq!(name, None);
    }

    // -----------------------------------------------------------------------
    // File resolution tests (filesystem-dependent)
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_nonexistent_header() {
        let handler = IncludeHandler::new(vec![], vec![]);
        let result =
            handler.resolve_include("nonexistent_header_xyz.h", false, Path::new("/tmp/test.c"));
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_system_nonexistent() {
        let handler = IncludeHandler::new(vec![], vec![]);
        let result =
            handler.resolve_include("nonexistent_system_xyz.h", true, Path::new("/tmp/test.c"));
        assert!(result.is_none());
    }
}
