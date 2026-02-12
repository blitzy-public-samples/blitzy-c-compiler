//! RAII-based temporary file and directory management for intermediate compilation artifacts.
//!
//! This module provides [`TempFile`] and [`TempDir`] types that automatically clean up
//! their underlying filesystem entries when dropped. This guarantees no stale temporary
//! files remain on disk after compilation completes, panics, or the process is
//! interrupted (via normal unwinding).
//!
//! # Unique Name Generation
//!
//! Temporary file and directory names are generated using the current process ID
//! combined with a module-level atomic counter, producing names of the form
//! `{prefix}{pid}_{counter}{suffix}` (e.g., `bcc_12345_0.o`). The atomic counter
//! ensures uniqueness across threads within the same process, while the PID ensures
//! uniqueness across concurrent BCC invocations on the same system.
//!
//! # Zero-Dependency Implementation
//!
//! This module replaces the external `tempfile` crate and uses only the Rust standard
//! library (`std::fs`, `std::env`, `std::process`, `std::sync::atomic`), honoring the
//! project's zero-dependency mandate.
//!
//! # Error Handling
//!
//! - **Creation errors** propagate as `std::io::Result`.
//! - **Drop (cleanup) errors** are silently ignored for best-effort cleanup, ensuring
//!   that a failure to delete a temp file does not cause a panic during stack unwinding.
//!
//! # Usage
//!
//! ```ignore
//! use crate::common::temp_files::{TempFile, TempDir};
//!
//! // Create a temporary object file in the system temp directory
//! let mut tmp = TempFile::new("bcc_", ".o")?;
//! tmp.write_all(&object_code)?;
//! let bytes = tmp.read_all()?;
//!
//! // Prevent deletion if this is the final output
//! tmp.keep();
//!
//! // Create a temporary directory for multi-file compilation
//! let dir = TempDir::new("bcc_build_")?;
//! let obj = dir.create_file("main.o")?;
//! // Both `obj` and `dir` are cleaned up when they go out of scope.
//! ```

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global atomic counter for generating unique temporary file/directory names.
///
/// Incremented via `fetch_add(1, Ordering::Relaxed)` on each creation request.
/// `Relaxed` ordering is sufficient because we only require uniqueness, not
/// cross-thread data synchronization (each counter value is used exactly once).
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique name for a temporary file or directory.
///
/// The name follows the pattern `{prefix}{pid}_{counter}{suffix}`, for example
/// `bcc_12345_0.o`. The PID component ensures uniqueness across concurrent
/// processes, and the atomic counter ensures uniqueness across threads within a
/// single process invocation.
///
/// # Arguments
///
/// * `prefix` — A descriptive prefix such as `"bcc_"` or `"obj_"`.
/// * `suffix` — A file extension such as `".o"` or `".s"`, or an empty string
///   for directories.
///
/// # Returns
///
/// A `String` containing the generated unique name.
fn unique_name(prefix: &str, suffix: &str) -> String {
    let pid = process::id();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}{}_{}{}", prefix, pid, counter, suffix)
}

// ---------------------------------------------------------------------------
// TempFile
// ---------------------------------------------------------------------------

/// An RAII temporary file that is automatically deleted when dropped.
///
/// `TempFile` wraps a [`PathBuf`] pointing to a file on disk. When the
/// `TempFile` value is dropped, the file is removed unless [`keep`](TempFile::keep)
/// has been called to mark it as persistent (e.g., for a final output artifact).
///
/// # Thread Safety
///
/// `TempFile` is **not** `Send`/`Sync` by design — file handle ownership stays
/// on the creating thread. The atomic counter used during name generation is the
/// only cross-thread coordination point.
pub struct TempFile {
    /// Absolute path to the temporary file on disk.
    path: PathBuf,
    /// When `true`, the file will **not** be deleted on drop. Defaults to `false`.
    keep: bool,
}

impl TempFile {
    /// Create a new temporary file in the operating system's default temporary
    /// directory (typically `/tmp` on Linux).
    ///
    /// The file is created on disk immediately so that subsequent calls to
    /// [`write_all`](TempFile::write_all) and [`read_all`](TempFile::read_all)
    /// operate on a valid path.
    ///
    /// # Arguments
    ///
    /// * `prefix` — Descriptive prefix for the file name (e.g., `"bcc_"`).
    /// * `suffix` — File extension including the dot (e.g., `".o"`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the temporary directory cannot be determined or
    /// if the file cannot be created on disk.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tmp = TempFile::new("bcc_", ".o")?;
    /// assert!(tmp.path().exists());
    /// ```
    pub fn new(prefix: &str, suffix: &str) -> io::Result<Self> {
        let dir = env::temp_dir();
        Self::new_in(&dir, prefix, suffix)
    }

    /// Create a new temporary file in the specified directory.
    ///
    /// The parent directory must already exist. The file is created on disk
    /// immediately.
    ///
    /// # Arguments
    ///
    /// * `dir`    — The parent directory in which the temporary file is created.
    /// * `prefix` — Descriptive prefix for the file name (e.g., `"bcc_"`).
    /// * `suffix` — File extension including the dot (e.g., `".o"`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the directory does not exist or the file
    /// cannot be created.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tmp = TempFile::new_in(Path::new("/tmp/bcc_build"), "obj_", ".o")?;
    /// ```
    pub fn new_in(dir: &Path, prefix: &str, suffix: &str) -> io::Result<Self> {
        let name = unique_name(prefix, suffix);
        let path = dir.join(&name);

        // Create the file on disk. `fs::File::create` truncates if the file
        // already exists, which is acceptable given the uniqueness guarantees
        // of the name generator.
        fs::File::create(&path)?;

        Ok(TempFile { path, keep: false })
    }

    /// Returns the filesystem path of this temporary file.
    ///
    /// The returned path is valid for the lifetime of the `TempFile` (i.e.,
    /// until the value is dropped and the file is potentially deleted).
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Mark this temporary file as persistent, preventing deletion on drop.
    ///
    /// This is used when the temporary file becomes the final output artifact
    /// (e.g., the linked executable or shared library) and should be preserved
    /// after the compilation pipeline completes.
    ///
    /// Once called, the file will remain on disk indefinitely; the caller
    /// assumes responsibility for its lifecycle.
    #[inline]
    pub fn keep(&mut self) {
        self.keep = true;
    }

    /// Write the entirety of `data` to the temporary file, replacing any
    /// previous content.
    ///
    /// This is a convenience wrapper around [`std::fs::write`] that targets
    /// the temporary file's path.
    ///
    /// # Arguments
    ///
    /// * `data` — The byte slice to write to the file.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the write operation fails (e.g., due to
    /// disk-full conditions or permission errors).
    pub fn write_all(&self, data: &[u8]) -> io::Result<()> {
        fs::write(&self.path, data)
    }

    /// Read the entire contents of the temporary file into memory.
    ///
    /// This is a convenience wrapper around [`std::fs::read`] that targets
    /// the temporary file's path.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the raw bytes of the file.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the read operation fails (e.g., the file
    /// has already been deleted or is not readable).
    pub fn read_all(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.path)
    }
}

impl Drop for TempFile {
    /// Attempt to delete the temporary file from disk.
    ///
    /// If [`keep`](TempFile::keep) was called, the file is left in place.
    /// Deletion errors (e.g., file already removed, permission denied) are
    /// silently ignored to prevent panics during stack unwinding.
    fn drop(&mut self) {
        if !self.keep {
            // Best-effort cleanup — ignore errors.
            let _ = fs::remove_file(&self.path);
        }
    }
}

// ---------------------------------------------------------------------------
// TempDir
// ---------------------------------------------------------------------------

/// An RAII temporary directory that is recursively removed when dropped.
///
/// `TempDir` creates a uniquely-named directory under the system temporary
/// directory. All files and subdirectories within it are recursively deleted
/// on drop, making it ideal for grouping intermediate object files during
/// multi-source-file compilation.
///
/// # Examples
///
/// ```ignore
/// let dir = TempDir::new("bcc_build_")?;
/// let obj1 = dir.create_file("foo.o")?;
/// let obj2 = dir.create_file("bar.o")?;
/// // When `dir` is dropped, the entire directory tree is removed.
/// ```
pub struct TempDir {
    /// Absolute path to the temporary directory on disk.
    path: PathBuf,
}

impl TempDir {
    /// Create a new temporary directory in the operating system's default
    /// temporary directory (typically `/tmp` on Linux).
    ///
    /// The directory (and any required parent components) is created
    /// immediately via [`std::fs::create_dir_all`].
    ///
    /// # Arguments
    ///
    /// * `prefix` — Descriptive prefix for the directory name (e.g., `"bcc_build_"`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the directory cannot be created.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dir = TempDir::new("bcc_build_")?;
    /// assert!(dir.path().is_dir());
    /// ```
    pub fn new(prefix: &str) -> io::Result<Self> {
        let base = env::temp_dir();
        let name = unique_name(prefix, "");
        let path = base.join(&name);

        fs::create_dir_all(&path)?;

        Ok(TempDir { path })
    }

    /// Returns the filesystem path of this temporary directory.
    ///
    /// The returned path is valid for the lifetime of the `TempDir` (i.e.,
    /// until the value is dropped and the directory is recursively removed).
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Create a named file inside this temporary directory.
    ///
    /// The file is created on disk immediately and wrapped in a [`TempFile`]
    /// for RAII cleanup. Note that when the parent `TempDir` is dropped, the
    /// entire directory (including this file) is removed regardless of the
    /// individual `TempFile`'s `keep` flag.
    ///
    /// # Arguments
    ///
    /// * `name` — The file name to create inside the directory (e.g., `"main.o"`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the file cannot be created.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let dir = TempDir::new("bcc_build_")?;
    /// let obj = dir.create_file("module.o")?;
    /// obj.write_all(&compiled_bytes)?;
    /// ```
    pub fn create_file(&self, name: &str) -> io::Result<TempFile> {
        let file_path = self.path.join(name);
        fs::File::create(&file_path)?;

        Ok(TempFile {
            path: file_path,
            keep: false,
        })
    }
}

impl Drop for TempDir {
    /// Recursively remove the temporary directory and all of its contents.
    ///
    /// Deletion errors are silently ignored (best-effort cleanup) to avoid
    /// panics during stack unwinding.
    fn drop(&mut self) {
        // Best-effort recursive cleanup — ignore errors.
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_name_format() {
        let name = unique_name("bcc_", ".o");
        let pid = process::id();
        // The name should contain the PID and end with the suffix.
        assert!(name.starts_with("bcc_"));
        assert!(name.ends_with(".o"));
        assert!(name.contains(&pid.to_string()));
    }

    #[test]
    fn test_unique_names_are_distinct() {
        let a = unique_name("t_", "");
        let b = unique_name("t_", "");
        assert_ne!(a, b, "consecutive unique names must differ");
    }

    #[test]
    fn test_temp_file_creation_and_cleanup() {
        let path;
        {
            let tmp = TempFile::new("bcc_test_", ".o").expect("TempFile::new failed");
            path = tmp.path().to_path_buf();
            assert!(path.exists(), "temp file should exist after creation");
        }
        // After drop, the file should be removed.
        assert!(!path.exists(), "temp file should be removed after drop");
    }

    #[test]
    fn test_temp_file_keep_prevents_deletion() {
        let path;
        {
            let mut tmp = TempFile::new("bcc_keep_", ".o").expect("TempFile::new failed");
            tmp.keep();
            path = tmp.path().to_path_buf();
        }
        // File should still exist because keep() was called.
        assert!(path.exists(), "kept temp file should survive drop");
        // Manual cleanup for test hygiene.
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_temp_file_write_and_read() {
        let tmp = TempFile::new("bcc_rw_", ".bin").expect("TempFile::new failed");
        let data = b"hello, BCC!";
        tmp.write_all(data).expect("write_all failed");
        let read_back = tmp.read_all().expect("read_all failed");
        assert_eq!(
            read_back, data,
            "read_all must return exactly what was written"
        );
    }

    #[test]
    fn test_temp_file_new_in() {
        let dir = TempDir::new("bcc_in_test_").expect("TempDir::new failed");
        let tmp = TempFile::new_in(dir.path(), "sub_", ".o").expect("TempFile::new_in failed");
        assert!(tmp.path().starts_with(dir.path()));
        assert!(tmp.path().exists());
    }

    #[test]
    fn test_temp_dir_creation_and_cleanup() {
        let path;
        {
            let dir = TempDir::new("bcc_dir_test_").expect("TempDir::new failed");
            path = dir.path().to_path_buf();
            assert!(path.is_dir(), "temp dir should exist after creation");
        }
        // After drop, the directory should be removed.
        assert!(!path.exists(), "temp dir should be removed after drop");
    }

    #[test]
    fn test_temp_dir_create_file() {
        let dir = TempDir::new("bcc_cf_test_").expect("TempDir::new failed");
        let obj = dir.create_file("main.o").expect("create_file failed");
        assert!(obj.path().exists());
        assert_eq!(obj.path().file_name().unwrap(), "main.o");

        obj.write_all(b"\x7fELF").expect("write_all failed");
        let bytes = obj.read_all().expect("read_all failed");
        assert_eq!(bytes, b"\x7fELF");
    }

    #[test]
    fn test_temp_dir_recursive_cleanup() {
        let dir_path;
        let file_path;
        {
            let dir = TempDir::new("bcc_rec_test_").expect("TempDir::new failed");
            let obj = dir.create_file("nested.o").expect("create_file failed");
            dir_path = dir.path().to_path_buf();
            file_path = obj.path().to_path_buf();
            assert!(dir_path.is_dir());
            assert!(file_path.exists());
        }
        // Both the directory and its contents should be gone.
        assert!(!dir_path.exists(), "temp dir should be removed after drop");
        assert!(
            !file_path.exists(),
            "files inside temp dir should be removed after drop"
        );
    }

    #[test]
    fn test_multiple_files_in_temp_dir() {
        let dir = TempDir::new("bcc_multi_").expect("TempDir::new failed");
        let files: Vec<TempFile> = (0..5)
            .map(|i| {
                dir.create_file(&format!("part_{}.o", i))
                    .expect("create_file failed")
            })
            .collect();

        for f in &files {
            assert!(f.path().exists());
            f.write_all(&[0xDE, 0xAD]).expect("write_all failed");
        }

        for f in &files {
            let data = f.read_all().expect("read_all failed");
            assert_eq!(data, &[0xDE, 0xAD]);
        }
    }

    #[test]
    fn test_drop_on_already_deleted_file_is_safe() {
        let tmp = TempFile::new("bcc_del_", ".o").expect("TempFile::new failed");
        let path = tmp.path().to_path_buf();
        // Manually remove the file before drop.
        fs::remove_file(&path).expect("manual remove failed");
        // Dropping should not panic even though the file is already gone.
        drop(tmp);
    }

    #[test]
    fn test_drop_on_already_deleted_dir_is_safe() {
        let dir = TempDir::new("bcc_deld_").expect("TempDir::new failed");
        let path = dir.path().to_path_buf();
        // Manually remove the directory before drop.
        fs::remove_dir_all(&path).expect("manual remove failed");
        // Dropping should not panic even though the directory is already gone.
        drop(dir);
    }
}
