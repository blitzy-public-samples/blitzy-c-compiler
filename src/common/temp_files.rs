//! RAII-based temporary file and directory management for BCC.
//!
//! During multi-file compilation, BCC compiles each `.c` source file to
//! an intermediate `.o` object file, then links them together. These
//! intermediate files need automatic cleanup to avoid littering the
//! filesystem. This module provides [`TempFile`] and [`TempDir`] types
//! that automatically delete their underlying file or directory when
//! dropped, implementing the RAII (Resource Acquisition Is Initialization)
//! pattern.
//!
//! This module replaces the external `tempfile` crate and uses only the
//! Rust standard library — no external dependencies permitted per the
//! zero-dependency mandate.
//!
//! # Thread Safety
//!
//! Unique name generation uses an [`AtomicU64`] counter, making it safe
//! to create temporary files from multiple threads simultaneously without
//! name collisions.
//!
//! # Error Handling
//!
//! Cleanup failures during `Drop` are silently ignored to prevent panics
//! during unwinding. The creation functions (`new`, `new_in`) return
//! `io::Result` so callers can handle errors properly.
//!
//! # Examples
//!
//! ```rust,no_run
//! use bcc::common::temp_files::{TempFile, TempDir, create_temp_object_file};
//!
//! // Create a temporary object file — automatically cleaned up when dropped
//! let obj = create_temp_object_file().expect("failed to create temp .o file");
//! println!("Temp object at: {}", obj.path().display());
//! // File is deleted when `obj` goes out of scope
//!
//! // Use keep() to prevent automatic deletion
//! let kept_path = {
//!     let obj = create_temp_object_file().expect("failed to create temp .o file");
//!     obj.keep()
//! };
//! // `kept_path` still exists on disk — caller is responsible for cleanup
//!
//! // Create a temporary directory with files inside it
//! let dir = TempDir::new().expect("failed to create temp dir");
//! let file_in_dir = dir.create_file("output.o").expect("failed to create file in dir");
//! // Both the file and directory are cleaned up when dropped
//! ```

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Unique name generation
// ---------------------------------------------------------------------------

/// Global atomic counter for generating unique temporary file/directory names.
/// Uses `Ordering::Relaxed` because we only need uniqueness, not ordering
/// guarantees — each `fetch_add` returns a distinct value regardless of
/// memory ordering.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique name for a temporary file or directory.
///
/// The name is composed of:
/// - A caller-supplied `prefix` (e.g., `"bcc_tmp"` or `"bcc_dir"`)
/// - The current process ID (for uniqueness across processes)
/// - A monotonically increasing atomic counter (for uniqueness within a process)
/// - A caller-supplied `suffix` (e.g., `".o"`, `".s"`, `""`)
///
/// The resulting format is: `{prefix}_{pid}_{counter}{suffix}`
///
/// # Thread Safety
///
/// The atomic counter ensures that concurrent calls from different threads
/// always produce distinct names.
fn unique_name(prefix: &str, suffix: &str) -> String {
    let pid = process::id();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}_{}{}", prefix, pid, count, suffix)
}

// ---------------------------------------------------------------------------
// TempFile
// ---------------------------------------------------------------------------

/// RAII temporary file — automatically deleted when dropped.
///
/// When a `TempFile` is created, an empty file is physically created on disk
/// to reserve the name. When the `TempFile` value is dropped, the file is
/// deleted unless [`keep`](TempFile::keep) or [`into_path`](TempFile::into_path)
/// was called to transfer ownership of the path to the caller.
///
/// Deletion failures during `Drop` are silently ignored to prevent panics
/// during stack unwinding.
pub struct TempFile {
    /// The absolute path to the temporary file on disk.
    path: PathBuf,
    /// Whether to delete the file when this struct is dropped.
    /// Set to `false` by [`keep`](TempFile::keep) or
    /// [`into_path`](TempFile::into_path).
    delete_on_drop: bool,
}

impl TempFile {
    /// Create a new temporary file in the system temporary directory.
    ///
    /// The file is created with a unique name using the pattern
    /// `bcc_tmp_{pid}_{counter}{suffix}` inside the directory returned by
    /// [`std::env::temp_dir()`].
    ///
    /// # Arguments
    ///
    /// * `suffix` — File extension including the dot, e.g. `".o"`, `".s"`, `".i"`.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file cannot be created (e.g., the temp
    /// directory does not exist or is not writable).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempFile;
    ///
    /// let tmp = TempFile::new(".o").expect("failed to create temp file");
    /// assert!(tmp.path().exists());
    /// ```
    pub fn new(suffix: &str) -> io::Result<Self> {
        let dir = env::temp_dir();
        Self::new_in(&dir, suffix)
    }

    /// Create a new temporary file in a specific directory.
    ///
    /// Behaves like [`TempFile::new`] but places the file in `dir` instead
    /// of the system temporary directory.
    ///
    /// # Arguments
    ///
    /// * `dir` — The directory in which to create the temporary file. Must
    ///   already exist.
    /// * `suffix` — File extension including the dot, e.g. `".o"`.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if:
    /// - `dir` does not exist or is not a directory
    /// - The file cannot be created (permissions, disk full, etc.)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempFile;
    /// use std::path::Path;
    ///
    /// let tmp = TempFile::new_in(Path::new("/tmp"), ".s")
    ///     .expect("failed to create temp file");
    /// assert!(tmp.path().starts_with("/tmp"));
    /// ```
    pub fn new_in(dir: &Path, suffix: &str) -> io::Result<Self> {
        let name = unique_name("bcc_tmp", suffix);
        let path = dir.join(name);

        // Create the file on disk to reserve the name. We use `File::create`
        // which creates or truncates, then immediately drop the file handle.
        // The file remains on disk for later writing by the compilation pipeline.
        fs::File::create(&path)?;

        Ok(TempFile {
            path,
            delete_on_drop: true,
        })
    }

    /// Returns the path to the temporary file.
    ///
    /// The returned path is valid as long as the `TempFile` has not been
    /// dropped (or [`keep`](TempFile::keep) / [`into_path`](TempFile::into_path)
    /// was called, in which case the file persists).
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Prevent deletion of the temporary file on drop and return its path.
    ///
    /// After calling `keep`, the caller assumes responsibility for managing
    /// the file's lifetime. The file will NOT be deleted when the `TempFile`
    /// is dropped.
    ///
    /// This consumes the `TempFile` and returns the owned [`PathBuf`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempFile;
    ///
    /// let tmp = TempFile::new(".o").expect("failed to create temp file");
    /// let path = tmp.keep();
    /// // File at `path` still exists — caller must clean it up manually
    /// ```
    pub fn keep(mut self) -> PathBuf {
        self.delete_on_drop = false;
        // We need to extract the path before self is dropped.
        // Since Drop will see delete_on_drop = false, it won't delete.
        self.path.clone()
    }

    /// Consume this `TempFile` and return its path, preventing automatic cleanup.
    ///
    /// This is semantically identical to [`keep`](TempFile::keep) — it
    /// transfers ownership of the file path to the caller and suppresses
    /// the automatic deletion on drop.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempFile;
    ///
    /// let tmp = TempFile::new(".o").expect("failed to create temp file");
    /// let path = tmp.into_path();
    /// // File at `path` still exists — caller manages lifetime
    /// ```
    #[inline]
    pub fn into_path(self) -> PathBuf {
        self.keep()
    }
}

impl Drop for TempFile {
    /// Automatically delete the temporary file when the `TempFile` is dropped.
    ///
    /// If `delete_on_drop` is `true` (the default), the file is removed from
    /// disk. Errors during deletion are silently ignored — this is intentional
    /// to prevent panics during stack unwinding (e.g., if the file was already
    /// deleted by the user or a concurrent process, or if the filesystem
    /// became read-only).
    fn drop(&mut self) {
        if self.delete_on_drop {
            // Intentionally ignore errors: the file may already be gone, the
            // filesystem may be read-only, or we may be unwinding from a panic.
            let _ = fs::remove_file(&self.path);
        }
    }
}

// ---------------------------------------------------------------------------
// TempDir
// ---------------------------------------------------------------------------

/// RAII temporary directory — recursively deleted when dropped.
///
/// When a `TempDir` is created, a directory is physically created on disk.
/// When the `TempDir` value is dropped, the directory and all of its
/// contents are recursively deleted unless [`keep`](TempDir::keep) was
/// called to transfer ownership to the caller.
///
/// Deletion failures during `Drop` are silently ignored.
///
/// # Examples
///
/// ```rust,no_run
/// use bcc::common::temp_files::TempDir;
///
/// let dir = TempDir::new().expect("failed to create temp dir");
/// let file = dir.create_file("intermediate.o").expect("failed to create file");
/// // Both `file` and `dir` will be cleaned up on drop (dir recursively)
/// ```
pub struct TempDir {
    /// The absolute path to the temporary directory on disk.
    path: PathBuf,
    /// Whether to recursively delete the directory when this struct is dropped.
    /// Set to `false` by [`keep`](TempDir::keep).
    delete_on_drop: bool,
}

impl TempDir {
    /// Create a new temporary directory in the system temporary directory.
    ///
    /// The directory is created with a unique name using the pattern
    /// `bcc_dir_{pid}_{counter}` inside the directory returned by
    /// [`std::env::temp_dir()`].
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the directory cannot be created.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempDir;
    ///
    /// let dir = TempDir::new().expect("failed to create temp dir");
    /// assert!(dir.path().is_dir());
    /// ```
    pub fn new() -> io::Result<Self> {
        let parent = env::temp_dir();
        Self::new_in(&parent)
    }

    /// Create a new temporary directory inside a specific parent directory.
    ///
    /// # Arguments
    ///
    /// * `parent` — The parent directory in which to create the temp
    ///   directory. Must already exist.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if:
    /// - `parent` does not exist or is not a directory
    /// - The directory cannot be created (permissions, disk full, etc.)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempDir;
    /// use std::path::Path;
    ///
    /// let dir = TempDir::new_in(Path::new("/tmp"))
    ///     .expect("failed to create temp dir");
    /// assert!(dir.path().starts_with("/tmp"));
    /// ```
    pub fn new_in(parent: &Path) -> io::Result<Self> {
        let name = unique_name("bcc_dir", "");
        let path = parent.join(name);

        fs::create_dir(&path)?;

        Ok(TempDir {
            path,
            delete_on_drop: true,
        })
    }

    /// Returns the path to the temporary directory.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create a file with the given name inside this temporary directory.
    ///
    /// The created file is wrapped in a [`TempFile`] with `delete_on_drop`
    /// set to `false` because the parent `TempDir` will recursively delete
    /// all contents when it is dropped. This avoids double-delete attempts.
    ///
    /// # Arguments
    ///
    /// * `name` — The filename (not a path) to create inside this directory.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file cannot be created.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempDir;
    ///
    /// let dir = TempDir::new().expect("failed to create temp dir");
    /// let file = dir.create_file("output.o").expect("failed to create file");
    /// assert!(file.path().exists());
    /// assert!(file.path().starts_with(dir.path()));
    /// ```
    pub fn create_file(&self, name: &str) -> io::Result<TempFile> {
        let file_path = self.path.join(name);

        // Create the file on disk
        fs::File::create(&file_path)?;

        // The file's delete_on_drop is false because the parent TempDir's
        // recursive deletion will handle it. This prevents double-delete
        // attempts and avoids errors if the file is dropped after the
        // directory has already been removed.
        Ok(TempFile {
            path: file_path,
            delete_on_drop: false,
        })
    }

    /// Prevent recursive deletion of the temporary directory on drop.
    ///
    /// After calling `keep`, the caller assumes responsibility for the
    /// directory's lifetime. The directory and all its contents will NOT
    /// be deleted when the `TempDir` is dropped.
    ///
    /// This consumes the `TempDir` and returns the owned [`PathBuf`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use bcc::common::temp_files::TempDir;
    ///
    /// let dir = TempDir::new().expect("failed to create temp dir");
    /// let path = dir.keep();
    /// // Directory at `path` still exists — caller must clean it up
    /// ```
    pub fn keep(mut self) -> PathBuf {
        self.delete_on_drop = false;
        self.path.clone()
    }
}

impl Drop for TempDir {
    /// Recursively delete the temporary directory and all its contents.
    ///
    /// Errors are silently ignored to prevent panics during stack unwinding.
    fn drop(&mut self) {
        if self.delete_on_drop {
            // Intentionally ignore errors: same rationale as TempFile::drop.
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience functions for common temporary file types
// ---------------------------------------------------------------------------

/// Create a temporary object file (`.o` suffix) in the system temp directory.
///
/// This is a shorthand for `TempFile::new(".o")`, commonly used during
/// multi-file compilation when each `.c` source is compiled to an
/// intermediate `.o` file before linking.
///
/// # Errors
///
/// Returns an `io::Error` if the file cannot be created.
///
/// # Examples
///
/// ```rust,no_run
/// use bcc::common::temp_files::create_temp_object_file;
///
/// let obj = create_temp_object_file().expect("failed to create temp .o");
/// assert!(obj.path().extension().unwrap() == "o");
/// ```
pub fn create_temp_object_file() -> io::Result<TempFile> {
    TempFile::new(".o")
}

/// Create a temporary assembly file (`.s` suffix) in the system temp directory.
///
/// Used when the `-S` flag requests assembly output to a temporary location
/// before the final output path is determined.
///
/// # Errors
///
/// Returns an `io::Error` if the file cannot be created.
///
/// # Examples
///
/// ```rust,no_run
/// use bcc::common::temp_files::create_temp_assembly_file;
///
/// let asm = create_temp_assembly_file().expect("failed to create temp .s");
/// assert!(asm.path().extension().unwrap() == "s");
/// ```
pub fn create_temp_assembly_file() -> io::Result<TempFile> {
    TempFile::new(".s")
}

/// Create a temporary preprocessed file (`.i` suffix) in the system temp directory.
///
/// Used when the `-E` flag requests preprocessed output to a temporary
/// location, or as an intermediate step in the compilation pipeline.
///
/// # Errors
///
/// Returns an `io::Error` if the file cannot be created.
///
/// # Examples
///
/// ```rust,no_run
/// use bcc::common::temp_files::create_temp_preprocessed_file;
///
/// let pp = create_temp_preprocessed_file().expect("failed to create temp .i");
/// assert!(pp.path().extension().unwrap() == "i");
/// ```
pub fn create_temp_preprocessed_file() -> io::Result<TempFile> {
    TempFile::new(".i")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_name_produces_distinct_names() {
        let name1 = unique_name("test", ".o");
        let name2 = unique_name("test", ".o");
        let name3 = unique_name("test", ".o");
        assert_ne!(name1, name2);
        assert_ne!(name2, name3);
        assert_ne!(name1, name3);
    }

    #[test]
    fn test_unique_name_includes_prefix_and_suffix() {
        let name = unique_name("bcc_tmp", ".o");
        assert!(name.starts_with("bcc_tmp_"));
        assert!(name.ends_with(".o"));
    }

    #[test]
    fn test_unique_name_includes_pid() {
        let name = unique_name("bcc_tmp", ".o");
        let pid = process::id().to_string();
        assert!(name.contains(&pid), "Name should contain PID: {}", name);
    }

    #[test]
    fn test_temp_file_new_creates_file() {
        let tmp = TempFile::new(".o").expect("failed to create TempFile");
        assert!(tmp.path().exists(), "Temp file should exist on disk");
        assert!(tmp.path().is_file(), "Path should be a file");
    }

    #[test]
    fn test_temp_file_new_suffix() {
        let tmp = TempFile::new(".o").expect("failed to create TempFile");
        let ext = tmp.path().extension().unwrap().to_str().unwrap();
        assert_eq!(ext, "o");
    }

    #[test]
    fn test_temp_file_dropped_deletes_file() {
        let path;
        {
            let tmp = TempFile::new(".o").expect("failed to create TempFile");
            path = tmp.path().to_path_buf();
            assert!(path.exists(), "File should exist before drop");
        }
        // After drop, the file should be deleted
        assert!(!path.exists(), "File should be deleted after drop");
    }

    #[test]
    fn test_temp_file_keep_prevents_deletion() {
        let path;
        {
            let tmp = TempFile::new(".o").expect("failed to create TempFile");
            path = tmp.keep();
        }
        // After drop with keep(), the file should still exist
        assert!(path.exists(), "File should still exist after keep()");
        // Clean up manually
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_temp_file_into_path_prevents_deletion() {
        let path;
        {
            let tmp = TempFile::new(".o").expect("failed to create TempFile");
            path = tmp.into_path();
        }
        // After drop with into_path(), the file should still exist
        assert!(path.exists(), "File should still exist after into_path()");
        // Clean up manually
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_temp_file_new_in_specific_directory() {
        let dir = env::temp_dir();
        let tmp = TempFile::new_in(&dir, ".s").expect("failed to create TempFile");
        assert!(tmp.path().starts_with(&dir));
        assert!(tmp.path().exists());
    }

    #[test]
    fn test_temp_dir_new_creates_directory() {
        let dir = TempDir::new().expect("failed to create TempDir");
        assert!(dir.path().exists(), "Temp dir should exist on disk");
        assert!(dir.path().is_dir(), "Path should be a directory");
    }

    #[test]
    fn test_temp_dir_dropped_deletes_directory() {
        let path;
        {
            let dir = TempDir::new().expect("failed to create TempDir");
            path = dir.path().to_path_buf();
            assert!(path.exists(), "Dir should exist before drop");
        }
        assert!(!path.exists(), "Dir should be deleted after drop");
    }

    #[test]
    fn test_temp_dir_recursive_deletion() {
        let dir_path;
        {
            let dir = TempDir::new().expect("failed to create TempDir");
            dir_path = dir.path().to_path_buf();

            // Create some files inside
            fs::File::create(dir.path().join("a.o")).expect("create a.o");
            fs::File::create(dir.path().join("b.o")).expect("create b.o");

            // Create a subdirectory with a file
            let sub = dir.path().join("subdir");
            fs::create_dir(&sub).expect("create subdir");
            fs::File::create(sub.join("c.o")).expect("create c.o");

            assert!(dir.path().join("a.o").exists());
            assert!(dir.path().join("subdir").join("c.o").exists());
        }
        // Everything should be gone after drop
        assert!(!dir_path.exists(), "Dir and contents should be deleted");
    }

    #[test]
    fn test_temp_dir_keep_prevents_deletion() {
        let path;
        {
            let dir = TempDir::new().expect("failed to create TempDir");
            path = dir.keep();
        }
        assert!(path.exists(), "Dir should still exist after keep()");
        // Clean up manually
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn test_temp_dir_create_file() {
        let dir = TempDir::new().expect("failed to create TempDir");
        let file = dir.create_file("output.o").expect("failed to create file");
        assert!(file.path().exists(), "Created file should exist");
        assert!(
            file.path().starts_with(dir.path()),
            "File should be inside the directory"
        );
        assert_eq!(
            file.path().file_name().unwrap().to_str().unwrap(),
            "output.o"
        );
    }

    #[test]
    fn test_temp_dir_new_in_specific_parent() {
        let parent = env::temp_dir();
        let dir = TempDir::new_in(&parent).expect("failed to create TempDir");
        assert!(dir.path().starts_with(&parent));
        assert!(dir.path().is_dir());
    }

    #[test]
    fn test_create_temp_object_file() {
        let tmp = create_temp_object_file().expect("failed");
        assert!(tmp.path().exists());
        let ext = tmp.path().extension().unwrap().to_str().unwrap();
        assert_eq!(ext, "o");
    }

    #[test]
    fn test_create_temp_assembly_file() {
        let tmp = create_temp_assembly_file().expect("failed");
        assert!(tmp.path().exists());
        let ext = tmp.path().extension().unwrap().to_str().unwrap();
        assert_eq!(ext, "s");
    }

    #[test]
    fn test_create_temp_preprocessed_file() {
        let tmp = create_temp_preprocessed_file().expect("failed");
        assert!(tmp.path().exists());
        let ext = tmp.path().extension().unwrap().to_str().unwrap();
        assert_eq!(ext, "i");
    }

    #[test]
    fn test_cleanup_failure_does_not_panic() {
        // Create a temp file, delete it manually, then let TempFile drop.
        // The Drop impl should NOT panic.
        let tmp = TempFile::new(".o").expect("failed to create TempFile");
        let path = tmp.path().to_path_buf();
        // Manually delete the file before the TempFile is dropped
        fs::remove_file(&path).expect("manual delete");
        // tmp drops here — should not panic even though file is already gone
    }

    #[test]
    fn test_temp_dir_cleanup_failure_does_not_panic() {
        // Create a temp dir, delete it manually, then let TempDir drop.
        let dir = TempDir::new().expect("failed to create TempDir");
        let path = dir.path().to_path_buf();
        fs::remove_dir_all(&path).expect("manual delete");
        // dir drops here — should not panic
    }

    #[test]
    fn test_multiple_temp_files_unique_paths() {
        let f1 = TempFile::new(".o").expect("f1");
        let f2 = TempFile::new(".o").expect("f2");
        let f3 = TempFile::new(".o").expect("f3");
        assert_ne!(f1.path(), f2.path());
        assert_ne!(f2.path(), f3.path());
        assert_ne!(f1.path(), f3.path());
    }

    #[test]
    fn test_temp_dir_create_file_does_not_double_delete() {
        // Files created via TempDir::create_file() should have
        // delete_on_drop = false to avoid double-delete issues.
        let dir_path;
        let file_path;
        {
            let dir = TempDir::new().expect("dir");
            let file = dir.create_file("test.o").expect("file");
            dir_path = dir.path().to_path_buf();
            file_path = file.path().to_path_buf();
            assert!(file_path.exists());
            // file drops first (it should NOT delete since delete_on_drop = false)
            // dir drops second (it deletes everything recursively)
            drop(file);
            assert!(
                file_path.exists(),
                "File should still exist after TempFile drop (delete_on_drop=false)"
            );
        }
        // After dir drops, everything is cleaned up
        assert!(!dir_path.exists());
    }
}
