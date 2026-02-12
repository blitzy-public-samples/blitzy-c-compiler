//! `#include` file resolution module for the BCC C compiler.
//!
//! Implements the complete `#include` directive handling pipeline:
//!
//! - **Path Resolution:** Distinguishes between user includes (`#include "file"`)
//!   and system includes (`#include <file>`), searching directories in the correct
//!   order per C11 §6.10.2.
//!
//! - **Include Guard Optimization:** Detects the canonical `#ifndef GUARD` /
//!   `#define GUARD` / `#endif` pattern and tracks guard macro names per file.
//!   On subsequent includes, if the guard macro is already defined, the file is
//!   skipped without re-reading.
//!
//! - **`#pragma once` Support:** Tracks files marked with `#pragma once` and skips
//!   them on subsequent includes.
//!
//! - **Circular Include Detection:** Maintains an include stack and detects when a
//!   file is about to be included while it is already being processed, producing a
//!   diagnostic chain showing the full circular dependency.
//!
//! - **PUA-Aware File Loading:** Reads source files through
//!   [`crate::common::encoding::read_source_file`] for byte-exact non-UTF-8
//!   round-tripping (U+E080–U+E0FF mapping per Section 0.7.9).
//!
//! # Architecture
//!
//! The [`IncludeHandler`] is owned by the preprocessor driver and consulted on
//! every `#include` directive. The typical flow is:
//!
//! 1. [`IncludeHandler::resolve_include()`] — find the file on disk.
//! 2. [`IncludeHandler::should_skip_include()`] — check guards and `#pragma once`.
//! 3. [`IncludeHandler::push_include()`] — detect circular includes and push onto stack.
//! 4. [`IncludeHandler::load_file()`] — read the file with PUA encoding.
//! 5. *(preprocess the file's tokens)*
//! 6. [`IncludeHandler::detect_include_guard()`] + [`IncludeHandler::register_include_guard()`]
//! 7. [`IncludeHandler::pop_include()`] — pop the include stack.
//!
//! # Zero-Dependency Implementation
//!
//! This module uses only the Rust standard library and internal BCC modules,
//! adhering to the project's zero-dependency mandate.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::common::encoding::read_source_file;
use crate::common::fx_hash::{FxHashMap, FxHashSet, fx_hash_map, fx_hash_set};
use crate::common::source_map::{FileId, SourceMap};
use crate::common::string_interner::Symbol;
use crate::frontend::lexer::token::{Token, TokenKind};

// ---------------------------------------------------------------------------
// IncludeKind — distinguishes user includes from system includes
// ---------------------------------------------------------------------------

/// Distinguishes `#include "..."` (user) from `#include <...>` (system) includes.
///
/// The include kind determines the search order for resolving the file path:
///
/// - **User** (`"..."`) — searches the current file's directory first, then `-I`
///   paths, then system paths.
/// - **System** (`<...>`) — searches `-I` paths first, then system paths. The
///   current file's directory is **not** searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IncludeKind {
    /// User include: `#include "file.h"`.
    /// Search order: current dir → `-I` paths → system paths.
    User,
    /// System include: `#include <file.h>`.
    /// Search order: `-I` paths → system paths.
    System,
}

impl fmt::Display for IncludeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncludeKind::User => write!(f, "user"),
            IncludeKind::System => write!(f, "system"),
        }
    }
}

// ---------------------------------------------------------------------------
// CircularIncludeError
// ---------------------------------------------------------------------------

/// Error produced when a circular `#include` chain is detected.
///
/// Contains the full include chain leading to the circular dependency, enabling
/// diagnostic messages that show the user exactly which files form the cycle.
///
/// # Example
///
/// For the cycle `a.h → b.h → c.h → a.h`, the chain would be
/// `[a.h, b.h, c.h, a.h]` and the [`Display`] output:
///
/// ```text
/// a.h → b.h → c.h → a.h (circular)
/// ```
#[derive(Clone, Debug)]
pub struct CircularIncludeError {
    /// The full include chain forming the cycle.
    ///
    /// The last element is always a duplicate of an earlier element,
    /// demonstrating the circular dependency.  For example, if `a.h`
    /// includes `b.h` which includes `a.h`, the chain is
    /// `[path/to/a.h, path/to/b.h, path/to/a.h]`.
    pub chain: Vec<PathBuf>,
}

impl fmt::Display for CircularIncludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.chain.is_empty() {
            return write!(f, "circular include detected");
        }
        for (idx, path) in self.chain.iter().enumerate() {
            if idx > 0 {
                write!(f, " → ")?;
            }
            write!(f, "{}", path.display())?;
        }
        write!(f, " (circular)")
    }
}

impl std::error::Error for CircularIncludeError {}

// ---------------------------------------------------------------------------
// IncludeHandler — central handler for #include directive processing
// ---------------------------------------------------------------------------

/// Central handler for all `#include` directive processing in the BCC
/// preprocessor.
///
/// Manages include-path resolution, guard optimization, `#pragma once`
/// tracking, circular-dependency detection, and PUA-aware file loading.
/// This struct is owned by the preprocessor driver and consulted on every
/// `#include` directive encountered during preprocessing.
///
/// # Fields
///
/// | Field | Visibility | Purpose |
/// |-------|-----------|---------|
/// | `system_paths` | `pub` | System include directories (e.g. `/usr/include`). |
/// | `user_paths` | `pub` | User include directories from `-I` flags. |
/// | `include_stack` | `pub` | Current include chain for circular detection. |
/// | `include_guards` | private | Maps canonical paths → guard macro [`Symbol`]. |
/// | `pragma_once_files` | private | Set of `#pragma once` files. |
/// | `included_files` | private | Set of all files included at least once. |
pub struct IncludeHandler {
    /// System include directories (e.g. `/usr/include`, `/usr/local/include`).
    /// Searched last in the include-resolution order for both user and system
    /// includes.
    pub system_paths: Vec<PathBuf>,

    /// User include directories provided via `-I` flags on the command line.
    /// Searched before system paths.  For user includes, searched after the
    /// current file's directory.
    pub user_paths: Vec<PathBuf>,

    /// Stack of canonical file paths currently being processed.
    /// The top of the stack is the file currently being preprocessed.
    /// Used for circular-include detection: if a file to be included is
    /// already on this stack, a circular dependency exists.
    pub include_stack: Vec<PathBuf>,

    /// Maps canonical file paths to their detected include-guard macro
    /// [`Symbol`].  Populated by [`IncludeHandler::register_include_guard`]
    /// after [`IncludeHandler::detect_include_guard`] succeeds.  Consulted by
    /// [`IncludeHandler::should_skip_include`] to avoid re-processing files
    /// whose guard macro is already defined.
    include_guards: FxHashMap<PathBuf, Symbol>,

    /// Set of canonical file paths that have been marked with `#pragma once`.
    /// Once a file is in this set, subsequent `#include` directives targeting
    /// the same file are silently skipped.
    pragma_once_files: FxHashSet<PathBuf>,

    /// Set of all canonical file paths that have been included at least once
    /// during this compilation unit.  Used for tracking and guard
    /// optimisation.
    included_files: FxHashSet<PathBuf>,
}

impl Default for IncludeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl IncludeHandler {
    // -------------------------------------------------------------------
    // Construction & configuration
    // -------------------------------------------------------------------

    /// Creates a new [`IncludeHandler`] with empty path lists, an empty
    /// include stack, and no tracked include guards or pragma-once files.
    ///
    /// Include paths should be configured via [`Self::add_system_path`] and
    /// [`Self::add_user_path`] before the first `#include` directive is
    /// processed.
    pub fn new() -> Self {
        IncludeHandler {
            system_paths: Vec::new(),
            user_paths: Vec::new(),
            include_stack: Vec::new(),
            include_guards: fx_hash_map(),
            pragma_once_files: fx_hash_set(),
            included_files: fx_hash_set(),
        }
    }

    /// Appends a system include directory to the search path.
    ///
    /// System paths are searched last in the include-resolution order, after
    /// the current file's directory (for user includes) and all `-I` paths.
    pub fn add_system_path(&mut self, path: PathBuf) {
        self.system_paths.push(path);
    }

    /// Appends a user include directory (from a `-I` flag) to the search
    /// path.
    ///
    /// User paths are searched before system paths.  For user includes
    /// (`#include "file"`), they are searched after the current file's
    /// directory.  For system includes (`#include <file>`), they are the
    /// first directories searched.
    pub fn add_user_path(&mut self, path: PathBuf) {
        self.user_paths.push(path);
    }

    // -------------------------------------------------------------------
    // Path resolution
    // -------------------------------------------------------------------

    /// Resolves an `#include` directive to a canonical file path on disk.
    ///
    /// Search order follows the C standard and GCC conventions:
    ///
    /// - **User include** (`#include "file"`):
    ///   1. Directory of the file containing the `#include` directive.
    ///   2. Each `-I` directory in order.
    ///   3. Each system directory in order.
    ///
    /// - **System include** (`#include <file>`):
    ///   1. Each `-I` directory in order.
    ///   2. Each system directory in order.
    ///
    /// Returns the canonicalized path of the first existing file found, or
    /// `None` if no matching file exists in any search directory.
    ///
    /// # Arguments
    ///
    /// * `path` — The include file name from the directive (e.g. `"stdio.h"`,
    ///   `"sys/types.h"`).
    /// * `kind` — Whether this is a user (`"..."`) or system (`<...>`)
    ///   include.
    /// * `current_file_dir` — The directory of the file that contains the
    ///   `#include` directive.  Used only for user includes.
    pub fn resolve_include(
        &self,
        path: &str,
        kind: IncludeKind,
        current_file_dir: &Path,
    ) -> Option<PathBuf> {
        match kind {
            IncludeKind::User => {
                // 1. Directory of the current file
                let candidate = current_file_dir.join(path);
                if candidate.exists() {
                    return Some(canonicalize_path(&candidate));
                }

                // 2. -I directories
                for dir in &self.user_paths {
                    let candidate = dir.join(path);
                    if candidate.exists() {
                        return Some(canonicalize_path(&candidate));
                    }
                }

                // 3. System directories
                for dir in &self.system_paths {
                    let candidate = dir.join(path);
                    if candidate.exists() {
                        return Some(canonicalize_path(&candidate));
                    }
                }
            }
            IncludeKind::System => {
                // 1. -I directories (no current-dir search for system includes)
                for dir in &self.user_paths {
                    let candidate = dir.join(path);
                    if candidate.exists() {
                        return Some(canonicalize_path(&candidate));
                    }
                }

                // 2. System directories
                for dir in &self.system_paths {
                    let candidate = dir.join(path);
                    if candidate.exists() {
                        return Some(canonicalize_path(&candidate));
                    }
                }
            }
        }

        None
    }

    // -------------------------------------------------------------------
    // Include guard detection
    // -------------------------------------------------------------------

    /// Detects the canonical include-guard pattern in a file's token stream.
    ///
    /// Analyses the structural token pattern:
    /// ```c
    /// #ifndef GUARD_NAME
    /// #define GUARD_NAME
    /// /* … file content … */
    /// #endif
    /// ```
    ///
    /// Detection is performed by **structural pattern matching** on the token
    /// sequence.  The algorithm verifies:
    ///
    /// 1. The first non-whitespace tokens form `# <dir1> <guard>`.
    /// 2. The second directive is `# <dir2> <guard>` with the **same** guard
    ///    name symbol.
    /// 3. The two directive keywords differ from each other and from the
    ///    guard name (i.e. `ifndef` ≠ `define` ≠ `GUARD`).
    /// 4. The last directive in the file is `# <dir3>` (the closing
    ///    `#endif`).
    ///
    /// Returns `Some(guard_symbol)` if the pattern is detected, or `None`
    /// otherwise.
    ///
    /// # Arguments
    ///
    /// * `tokens` — The complete token sequence of the file to analyse.
    pub fn detect_include_guard(tokens: &[Token]) -> Option<Symbol> {
        if tokens.is_empty() {
            return None;
        }

        let len = tokens.len();
        let mut pos: usize = 0;

        // ---- Helper: advance past Whitespace tokens only ----
        let skip_ws = |p: &mut usize| {
            while *p < len {
                if let TokenKind::Whitespace = tokens[*p].kind {
                    *p += 1;
                } else {
                    break;
                }
            }
        };

        // ---- Helper: advance past Whitespace and Newline tokens ----
        let skip_ws_nl = |p: &mut usize| {
            while *p < len {
                match tokens[*p].kind {
                    TokenKind::Whitespace | TokenKind::Newline => *p += 1,
                    _ => break,
                }
            }
        };

        // ---- Forward scan: first directive (#ifndef GUARD) ----

        // Skip leading whitespace / newlines before the first #
        skip_ws_nl(&mut pos);

        // Expect '#'
        if pos >= len {
            return None;
        }
        if !matches!(tokens[pos].kind, TokenKind::Hash) {
            return None;
        }
        pos += 1;

        // Skip whitespace between '#' and directive name
        skip_ws(&mut pos);

        // Expect an identifier (the directive name, e.g. "ifndef")
        let first_directive_sym = match tokens.get(pos).map(|t| &t.kind) {
            Some(TokenKind::Identifier(sym)) => *sym,
            _ => return None,
        };
        pos += 1;

        // Skip whitespace between directive name and guard name
        skip_ws(&mut pos);

        // Expect an identifier (the guard macro name)
        let guard_sym = match tokens.get(pos).map(|t| &t.kind) {
            Some(TokenKind::Identifier(sym)) => *sym,
            _ => return None,
        };
        pos += 1;

        // The directive keyword must differ from the guard name
        if first_directive_sym == guard_sym {
            return None;
        }

        // Advance past the remainder of this line (to the next Newline)
        while pos < len && !matches!(tokens[pos].kind, TokenKind::Newline) {
            pos += 1;
        }
        // Skip the Newline itself
        if pos < len && matches!(tokens[pos].kind, TokenKind::Newline) {
            pos += 1;
        }

        // ---- Forward scan: second directive (#define GUARD) ----

        // Allow whitespace / blank lines between the two directives
        skip_ws_nl(&mut pos);

        // Expect '#'
        if pos >= len || !matches!(tokens[pos].kind, TokenKind::Hash) {
            return None;
        }
        pos += 1;

        skip_ws(&mut pos);

        // Expect an identifier (the directive name, e.g. "define")
        let second_directive_sym = match tokens.get(pos).map(|t| &t.kind) {
            Some(TokenKind::Identifier(sym)) => *sym,
            _ => return None,
        };
        pos += 1;

        // The two directives must name *different* keywords (ifndef ≠ define)
        if second_directive_sym == first_directive_sym {
            return None;
        }
        // The second directive keyword must also differ from the guard
        if second_directive_sym == guard_sym {
            return None;
        }

        skip_ws(&mut pos);

        // Expect the guard name again — must match the first directive's guard
        let guard_sym_2 = match tokens.get(pos).map(|t| &t.kind) {
            Some(TokenKind::Identifier(sym)) => *sym,
            _ => return None,
        };

        if guard_sym != guard_sym_2 {
            return None;
        }

        // ---- Backward scan: last directive must be #endif ----

        let mut end = len;

        // Skip trailing Eof
        if end > 0 && matches!(tokens[end - 1].kind, TokenKind::Eof) {
            end -= 1;
        }

        // Skip trailing whitespace / newlines
        while end > 0 {
            match tokens[end - 1].kind {
                TokenKind::Whitespace | TokenKind::Newline => end -= 1,
                _ => break,
            }
        }

        if end == 0 {
            return None;
        }

        // The last significant token should be an Identifier (the "endif"
        // directive keyword).
        let endif_directive_sym = match &tokens[end - 1].kind {
            TokenKind::Identifier(sym) => *sym,
            _ => return None,
        };
        end -= 1;

        // Skip whitespace between '#' and "endif"
        while end > 0 {
            if let TokenKind::Whitespace = tokens[end - 1].kind {
                end -= 1;
            } else {
                break;
            }
        }

        // Expect '#' immediately before the endif identifier
        if end == 0 || !matches!(tokens[end - 1].kind, TokenKind::Hash) {
            return None;
        }

        // The endif directive keyword must differ from both previous
        // directive keywords and from the guard name itself.
        if endif_directive_sym == guard_sym
            || endif_directive_sym == first_directive_sym
            || endif_directive_sym == second_directive_sym
        {
            return None;
        }

        // All structural checks passed — this file uses an include guard.
        Some(guard_sym)
    }

    // -------------------------------------------------------------------
    // Circular include detection
    // -------------------------------------------------------------------

    /// Pushes a file onto the include stack for circular-dependency detection.
    ///
    /// Before pushing, the canonical form of `path` is checked against the
    /// current stack.  If the file is already on the stack a
    /// [`CircularIncludeError`] is returned with the full chain for
    /// diagnostic reporting.
    ///
    /// The caller **must** call [`Self::pop_include`] after the file has been
    /// fully processed.
    ///
    /// # Errors
    ///
    /// Returns [`CircularIncludeError`] if the file's canonical path is
    /// already present on the include stack.
    pub fn push_include(&mut self, path: &Path) -> Result<(), CircularIncludeError> {
        let canonical = canonicalize_path(path);

        // Check for circular dependency by comparing canonical paths.
        // Use .as_path() for explicit &Path comparison against stack entries.
        let canonical_ref = canonical.as_path();
        if self.include_stack.iter().any(|p| p.as_path() == canonical_ref) {
            let mut chain: Vec<PathBuf> = self
                .include_stack
                .iter()
                .skip_while(|p| p.as_path() != canonical_ref)
                .map(|p| p.to_path_buf())
                .collect();
            chain.push(canonical);
            return Err(CircularIncludeError { chain });
        }

        self.include_stack.push(canonical);
        Ok(())
    }

    /// Pops the topmost entry from the include stack after the included file
    /// has been fully processed.
    ///
    /// Must be called exactly once for each successful [`Self::push_include`].
    /// Panics in debug mode if the stack is empty.
    pub fn pop_include(&mut self) {
        debug_assert!(
            !self.include_stack.is_empty(),
            "pop_include called on an empty include stack"
        );
        self.include_stack.pop();
    }

    // -------------------------------------------------------------------
    // Include stack queries
    // -------------------------------------------------------------------

    /// Returns the directory containing the file at the top of the include
    /// stack, or `None` if the stack is empty or the path has no parent.
    ///
    /// This is a convenience method that the preprocessor driver can use to
    /// obtain the `current_file_dir` argument required by
    /// [`Self::resolve_include`] without tracking it separately.
    pub fn current_include_dir(&self) -> Option<PathBuf> {
        self.include_stack
            .last()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
    }

    // -------------------------------------------------------------------
    // File loading (PUA-aware)
    // -------------------------------------------------------------------

    /// Loads a source file with PUA-aware encoding and registers it in the
    /// [`SourceMap`].
    ///
    /// Reads the file via [`crate::common::encoding::read_source_file`],
    /// which encodes non-UTF-8 bytes (0x80–0xFF) as Unicode Private Use Area
    /// code points (U+E080–U+E0FF) for byte-exact round-tripping — critical
    /// for Linux kernel source files that embed binary data in string
    /// literals and inline assembly.
    ///
    /// The file is also registered in `included_files` so that subsequent
    /// guard checks can function correctly.
    ///
    /// # Arguments
    ///
    /// * `path` — The file path to load (should already be resolved).
    /// * `source_map` — The source map in which to register the file.
    ///
    /// # Returns
    ///
    /// A `(FileId, String)` tuple containing the file's unique source-map
    /// identifier and its PUA-encoded content.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from the underlying file read (e.g. file not
    /// found, permission denied).
    pub fn load_file(
        &mut self,
        path: &Path,
        source_map: &mut SourceMap,
    ) -> Result<(FileId, String), io::Error> {
        // Read the file with PUA encoding for non-UTF-8 byte preservation
        let content = read_source_file(path)?;

        // Derive a display name from the path for diagnostic messages
        let display_name = path.to_string_lossy().into_owned();

        // Register the file in the source map to obtain a FileId
        let file_id = source_map.add_file(display_name, content.clone());

        // Record this file as having been included (canonical form)
        let canonical = canonicalize_path(path);
        self.included_files.insert(canonical);

        Ok((file_id, content))
    }

    // -------------------------------------------------------------------
    // #pragma once
    // -------------------------------------------------------------------

    /// Registers a file as `#pragma once`, preventing future re-includes.
    ///
    /// After this call, [`Self::is_pragma_once`] and
    /// [`Self::should_skip_include`] will return `true` for the canonical
    /// form of `path`.
    pub fn register_pragma_once(&mut self, path: &Path) {
        let canonical = canonicalize_path(path);
        self.pragma_once_files.insert(canonical);
    }

    /// Returns `true` if `path` has been marked with `#pragma once`.
    pub fn is_pragma_once(&self, path: &Path) -> bool {
        let canonical = canonicalize_path(path);
        self.pragma_once_files.contains(&canonical)
    }

    // -------------------------------------------------------------------
    // Include-skip decision
    // -------------------------------------------------------------------

    /// Determines whether an `#include` of the given file should be skipped.
    ///
    /// A file is skipped if **either** of the following is true:
    ///
    /// 1. It has been marked with `#pragma once` (unconditional skip).
    /// 2. It has a detected include guard **and** the guard macro is
    ///    currently defined in the preprocessor's macro table.
    ///
    /// The `is_macro_defined` callback decouples the include handler from
    /// the preprocessor's macro table — the preprocessor provides its own
    /// check.
    ///
    /// # Arguments
    ///
    /// * `path` — The resolved path of the file to check.
    /// * `is_macro_defined` — A callback returning `true` if the given
    ///   [`Symbol`] is currently defined as a macro.
    pub fn should_skip_include(
        &self,
        path: &Path,
        is_macro_defined: impl Fn(&Symbol) -> bool,
    ) -> bool {
        let canonical = canonicalize_path(path);

        // Fast path: #pragma once — unconditional skip
        if self.pragma_once_files.contains(&canonical) {
            return true;
        }

        // Include-guard check: skip if the guard macro is defined
        if self.include_guards.contains_key(&canonical) {
            if let Some(guard_sym) = self.include_guards.get(&canonical) {
                if is_macro_defined(guard_sym) {
                    return true;
                }
            }
        }

        false
    }

    // -------------------------------------------------------------------
    // Include-guard registration
    // -------------------------------------------------------------------

    /// Registers a detected include guard for a file.
    ///
    /// Associates the canonical form of `path` with the guard macro
    /// [`Symbol`], enabling future [`Self::should_skip_include`] checks to
    /// avoid re-processing the file when its guard macro is already defined.
    ///
    /// Typically called after [`Self::detect_include_guard`] returns
    /// `Some(guard_symbol)`.
    pub fn register_include_guard(&mut self, path: &Path, guard_symbol: Symbol) {
        let canonical = canonicalize_path(path);
        self.include_guards.insert(canonical, guard_symbol);
    }
}

// ---------------------------------------------------------------------------
// Path normalisation helpers
// ---------------------------------------------------------------------------

/// Canonicalizes `path` for consistent map look-ups.
///
/// Attempts [`fs::canonicalize`] first (resolves symlinks and `.`/`..`
/// components).  If that fails (e.g. the file does not exist on disk yet or
/// permissions are insufficient), falls back to a manual normalisation that
/// collapses `.` and `..` components without touching the file system.
fn canonicalize_path(path: &Path) -> PathBuf {
    // Try the OS-level canonicalize first — this resolves symlinks and
    // produces an absolute path.
    match fs::canonicalize(path) {
        Ok(canonical) => canonical,
        Err(_) => normalize_path(path),
    }
}

/// Manual path normalisation: collapses `.` and `..` without filesystem
/// access.
///
/// This is used as a fallback when [`fs::canonicalize`] fails.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut parts: Vec<Component<'_>> = Vec::new();

    for component in path.components() {
        match component {
            // Skip current-directory markers entirely
            Component::CurDir => {}

            // Parent-directory: pop the last Normal component if possible
            Component::ParentDir => {
                if let Some(Component::Normal(_)) = parts.last() {
                    parts.pop();
                } else {
                    // Cannot collapse further (root-relative `..` or
                    // already at a ParentDir) — keep it
                    parts.push(component);
                }
            }

            // RootDir, Prefix, Normal — keep as-is
            other => parts.push(other),
        }
    }

    // Rebuild the PathBuf from the collapsed components
    let mut result = PathBuf::new();
    for comp in &parts {
        result.push(comp.as_os_str());
    }

    // If nothing remains, return "." so the path is never empty
    if result.as_os_str().is_empty() {
        result.push(".");
    }

    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- IncludeKind Display -------------------------------------------------

    #[test]
    fn include_kind_display_user() {
        assert_eq!(format!("{}", IncludeKind::User), "user");
    }

    #[test]
    fn include_kind_display_system() {
        assert_eq!(format!("{}", IncludeKind::System), "system");
    }

    // -- CircularIncludeError Display ----------------------------------------

    #[test]
    fn circular_error_display_empty_chain() {
        let err = CircularIncludeError { chain: vec![] };
        assert_eq!(format!("{}", err), "circular include detected");
    }

    #[test]
    fn circular_error_display_with_chain() {
        let err = CircularIncludeError {
            chain: vec![
                PathBuf::from("a.h"),
                PathBuf::from("b.h"),
                PathBuf::from("a.h"),
            ],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("a.h"));
        assert!(msg.contains("b.h"));
        assert!(msg.ends_with("(circular)"));
    }

    // -- IncludeHandler construction -----------------------------------------

    #[test]
    fn new_handler_has_empty_state() {
        let handler = IncludeHandler::new();
        assert!(handler.system_paths.is_empty());
        assert!(handler.user_paths.is_empty());
        assert!(handler.include_stack.is_empty());
    }

    // -- Path configuration --------------------------------------------------

    #[test]
    fn add_system_and_user_paths() {
        let mut handler = IncludeHandler::new();
        handler.add_system_path(PathBuf::from("/usr/include"));
        handler.add_user_path(PathBuf::from("/my/project/include"));
        assert_eq!(handler.system_paths.len(), 1);
        assert_eq!(handler.user_paths.len(), 1);
    }

    // -- Circular include detection ------------------------------------------

    #[test]
    fn push_include_detects_cycle() {
        let mut handler = IncludeHandler::new();

        // Create a temporary directory structure for canonicalize_path
        let tmp_dir = std::env::temp_dir().join("bcc_test_circular");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let file_a = tmp_dir.join("a.h");
        let file_b = tmp_dir.join("b.h");

        // Create files so canonicalize_path can succeed
        let _ = std::fs::write(&file_a, "");
        let _ = std::fs::write(&file_b, "");

        assert!(handler.push_include(&file_a).is_ok());
        assert!(handler.push_include(&file_b).is_ok());
        // Circular: a.h is already on the stack
        let result = handler.push_include(&file_a);
        assert!(result.is_err());

        let err = result.unwrap_err();
        // The chain should include a.h → b.h → a.h
        assert!(err.chain.len() >= 2);

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn push_pop_include_stack_management() {
        let mut handler = IncludeHandler::new();

        let tmp_dir = std::env::temp_dir().join("bcc_test_pushpop");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let file_a = tmp_dir.join("header.h");
        let _ = std::fs::write(&file_a, "");

        assert!(handler.push_include(&file_a).is_ok());
        assert_eq!(handler.include_stack.len(), 1);

        handler.pop_include();
        assert!(handler.include_stack.is_empty());

        // After popping, we can push the same file again (no cycle)
        assert!(handler.push_include(&file_a).is_ok());
        handler.pop_include();

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // -- #pragma once --------------------------------------------------------

    #[test]
    fn pragma_once_registration_and_check() {
        let mut handler = IncludeHandler::new();

        let tmp_dir = std::env::temp_dir().join("bcc_test_pragma");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let header = tmp_dir.join("once.h");
        let _ = std::fs::write(&header, "");

        assert!(!handler.is_pragma_once(&header));
        handler.register_pragma_once(&header);
        assert!(handler.is_pragma_once(&header));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // -- should_skip_include -------------------------------------------------

    #[test]
    fn should_skip_pragma_once_file() {
        let mut handler = IncludeHandler::new();

        let tmp_dir = std::env::temp_dir().join("bcc_test_skip_pragma");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let header = tmp_dir.join("skip_me.h");
        let _ = std::fs::write(&header, "");

        handler.register_pragma_once(&header);

        // should_skip returns true even if the macro callback says false
        assert!(handler.should_skip_include(&header, |_sym| false));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn should_skip_include_guard_when_defined() {
        let mut handler = IncludeHandler::new();

        let tmp_dir = std::env::temp_dir().join("bcc_test_skip_guard");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let header = tmp_dir.join("guarded.h");
        let _ = std::fs::write(&header, "");

        let guard = Symbol::new(42);
        handler.register_include_guard(&header, guard);

        // Guard is defined → skip
        assert!(handler.should_skip_include(&header, |sym| *sym == guard));

        // Guard is NOT defined → do not skip
        assert!(!handler.should_skip_include(&header, |_sym| false));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // -- normalize_path ------------------------------------------------------

    #[test]
    fn normalize_removes_current_dir() {
        let p = Path::new("a/./b/./c");
        let n = normalize_path(p);
        assert_eq!(n, PathBuf::from("a/b/c"));
    }

    #[test]
    fn normalize_collapses_parent_dir() {
        let p = Path::new("a/b/../c");
        let n = normalize_path(p);
        assert_eq!(n, PathBuf::from("a/c"));
    }

    #[test]
    fn normalize_empty_yields_dot() {
        let p = Path::new("a/..");
        let n = normalize_path(p);
        assert_eq!(n, PathBuf::from("."));
    }
}
