//! Shared test harness for BCC (Blitzy's C Compiler) checkpoint validation.
//!
//! This module provides reusable helper functions used by all 7 checkpoint test suites
//! (`tests/checkpoint1_hello_world.rs` through `tests/checkpoint7_stretch.rs`).
//!
//! Capabilities:
//! - BCC binary invocation via `std::process::Command`
//! - Output assertion (stdout, exit code, compilation success)
//! - ELF inspection via `readelf` / `objdump` subprocesses
//! - Cross-architecture execution via QEMU user-mode emulation
//! - Temporary file management with RAII cleanup
//! - Fixture path resolution for C test sources
//!
//! # Design Notes
//! - All BCC invocation uses subprocess execution — NO library-level BCC integration.
//! - Functions return `std::process::Output` rather than parsed results, giving
//!   checkpoint tests full flexibility in what they assert.
//! - `readelf` and `objdump` functions delegate to system binutils (v2.42 on Ubuntu 24.04).
//! - Cross-architecture testing uses QEMU user-mode emulation (v8.2.2).
//! - All functions are synchronous (tests run sequentially per checkpoint ordering).
//! - Zero external crate dependencies — ONLY `std::` imports.

#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global atomic counter for generating unique temporary file paths.
/// Avoids collisions between concurrent test executions or sequential calls
/// within the same test run.
static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// BCC Binary Location
// ---------------------------------------------------------------------------

/// Locates the BCC compiler binary in the build output directory.
///
/// Resolution order:
/// 1. `CARGO_BIN_EXE_bcc` environment variable (set by Cargo in integration tests)
/// 2. `target/release/bcc` relative to `CARGO_MANIFEST_DIR`
/// 3. `target/debug/bcc` relative to `CARGO_MANIFEST_DIR`
///
/// # Panics
/// Panics with a helpful message if the BCC binary is not found anywhere.
pub fn bcc_path() -> PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for binary targets in integration tests
    if let Ok(path) = env::var("CARGO_BIN_EXE_bcc") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
    }

    // Resolve project root from CARGO_MANIFEST_DIR
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let root = Path::new(&manifest_dir);

    // Prefer release build for test performance
    let release_path = root.join("target").join("release").join("bcc");
    if release_path.exists() {
        return release_path;
    }

    // Fall back to debug build
    let debug_path = root.join("target").join("debug").join("bcc");
    if debug_path.exists() {
        return debug_path;
    }

    panic!(
        "BCC binary not found. Run `cargo build --release` first.\n\
         Searched:\n  \
         - CARGO_BIN_EXE_bcc env var\n  \
         - {}\n  \
         - {}",
        release_path.display(),
        debug_path.display()
    );
}

// ---------------------------------------------------------------------------
// BCC Compilation Helpers
// ---------------------------------------------------------------------------

/// Invokes the BCC compiler with the given flags on the specified source file.
///
/// The source file path is appended as the last argument. This function does
/// **not** assert success — callers decide whether to expect success or failure.
///
/// # Arguments
/// * `source` — Path to the C source file.
/// * `flags` — Slice of additional flags (e.g., `["--target=x86-64", "-o", "out"]`).
///
/// # Returns
/// The raw `std::process::Output` containing stdout, stderr, and exit status.
///
/// # Panics
/// Panics if the BCC binary cannot be executed (e.g., permission denied).
pub fn compile(source: &str, flags: &[&str]) -> Output {
    let bcc = bcc_path();
    Command::new(&bcc)
        .args(flags)
        .arg(source)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute BCC binary at '{}': {}", bcc.display(), e))
}

/// Compiles a C source file for the specified target architecture, then runs
/// the resulting binary (dispatching through QEMU for cross-architecture targets).
///
/// # Arguments
/// * `source` — Path to the C source file.
/// * `target` — Target architecture string (e.g., `"x86-64"`, `"aarch64"`, `"riscv64"`).
///
/// # Returns
/// The execution `Output` of the compiled binary.
///
/// # Panics
/// Panics if compilation fails (includes stderr in the panic message) or if
/// the resulting binary cannot be executed.
pub fn compile_and_run(source: &str, target: &str) -> Output {
    let output_path = temp_output_path("test_binary");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile
    let target_flag = format!("--target={}", target);
    let compile_output = compile(source, &[&target_flag, "-o", output_str]);

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr);
        let stdout = String::from_utf8_lossy(&compile_output.stdout);
        panic!(
            "Compilation failed with exit code {:?}\n\
             source: {}\ntarget: {}\n\
             stdout: {}\nstderr: {}",
            compile_output.status.code(),
            source,
            target,
            stdout,
            stderr
        );
    }

    // Run
    let run_output = run_binary(output_str, target);

    // Clean up
    let _ = fs::remove_file(&output_path);

    run_output
}

/// Links one or more object files into an output binary using BCC.
///
/// # Arguments
/// * `objects` — Slice of object file paths to link.
/// * `output`  — Output file path.
/// * `flags`   — Additional linker flags (e.g., `["-shared", "--target=x86-64"]`).
///
/// # Returns
/// The raw `std::process::Output` from the BCC invocation.
///
/// # Panics
/// Panics if the BCC binary cannot be executed.
pub fn link(objects: &[&str], output: &str, flags: &[&str]) -> Output {
    let bcc = bcc_path();
    let mut cmd = Command::new(&bcc);
    cmd.args(flags);
    cmd.arg("-o").arg(output);
    for obj in objects {
        cmd.arg(obj);
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("Failed to execute BCC linker at '{}': {}", bcc.display(), e))
}

// ---------------------------------------------------------------------------
// Output Assertion Helpers
// ---------------------------------------------------------------------------

/// Asserts that the process stdout matches the expected string exactly.
///
/// Uses lossy UTF-8 conversion on the raw stdout bytes. On failure, includes
/// both expected and actual values in the panic message for easy diagnosis.
pub fn assert_stdout_eq(output: &Output, expected: &str) {
    let actual = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        actual, expected,
        "stdout mismatch:\nexpected: {:?}\nactual:   {:?}",
        expected, actual
    );
}

/// Asserts that the process exited successfully (exit code 0).
///
/// On failure, includes the exit status and stderr content in the panic
/// message to aid debugging.
pub fn assert_exit_success(output: &Output) {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Process exited with {}\nstdout: {}\nstderr: {}",
            output.status, stdout, stderr
        );
    }
}

/// Asserts that a BCC compilation succeeded (exit code 0).
///
/// Provides compiler-specific diagnostic context including both stdout and
/// stderr on failure.
pub fn assert_compilation_succeeds(output: &Output) {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "Compilation failed with exit code {}\nstdout: {}\nstderr: {}",
            output.status, stdout, stderr
        );
    }
}

// ---------------------------------------------------------------------------
// ELF Inspection via readelf
// ---------------------------------------------------------------------------

/// Runs `readelf -h <path>` and returns the ELF header output as a string.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_header(path: &str) -> String {
    let output = Command::new("readelf")
        .args(["-h", path])
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `readelf -S <path>` and returns the section headers output.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_sections(path: &str) -> String {
    let output = Command::new("readelf")
        .args(["-S", path])
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `readelf -s <path>` and returns the symbol table output.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_symbols(path: &str) -> String {
    let output = Command::new("readelf")
        .args(["-s", path])
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `readelf -r <path>` and returns the relocation entries output.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_relocations(path: &str) -> String {
    let output = Command::new("readelf")
        .args(["-r", path])
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `readelf -l <path>` and returns the program headers output.
///
/// Used by Checkpoint 4 to verify PT_DYNAMIC, PT_INTERP for shared libraries.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_program_headers(path: &str) -> String {
    let output = Command::new("readelf")
        .args(["-l", path])
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `readelf --debug-dump=<section> <path>` and returns the DWARF dump output.
///
/// # Arguments
/// * `path`    — Path to the ELF file.
/// * `section` — DWARF section name (e.g., `"info"`, `"line"`, `"abbrev"`, `"str"`).
///
/// Used by Checkpoint 4 for DWARF v4 debug information validation.
///
/// # Panics
/// Panics if `readelf` is not found or the command fails.
pub fn readelf_debug_dump(path: &str, section: &str) -> String {
    let flag = format!("--debug-dump={}", section);
    let output = Command::new("readelf")
        .arg(&flag)
        .arg(path)
        .output()
        .expect("readelf not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

// ---------------------------------------------------------------------------
// Disassembly / Section Inspection via objdump
// ---------------------------------------------------------------------------

/// Runs `objdump -d <path>` and returns the disassembly output.
///
/// Used heavily by Checkpoint 5 for security mitigation validation
/// (retpoline thunks, CET/IBT `endbr64`, stack probe loops).
///
/// # Panics
/// Panics if `objdump` is not found or the command fails.
pub fn objdump_disassemble(path: &str) -> String {
    let output = Command::new("objdump")
        .args(["-d", path])
        .output()
        .expect("objdump not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Runs `objdump -s -j <section> <path>` and returns the hex dump of a specific
/// ELF section.
///
/// Used by Checkpoint 2 for PUA encoding round-trip verification: inspecting
/// `.rodata` for exact bytes `80 ff`.
///
/// # Arguments
/// * `path`    — Path to the ELF file.
/// * `section` — Section name (e.g., `".rodata"`).
///
/// # Panics
/// Panics if `objdump` is not found or the command fails.
pub fn objdump_section_contents(path: &str, section: &str) -> String {
    let output = Command::new("objdump")
        .args(["-s", "-j", section, path])
        .output()
        .expect("objdump not found or failed to execute. Is binutils installed?");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

// ---------------------------------------------------------------------------
// Cross-Architecture Execution via QEMU User-Mode
// ---------------------------------------------------------------------------

/// Executes a compiled binary, dispatching through QEMU for cross-architecture
/// targets.
///
/// # Architecture Dispatch
/// - `"x86-64"` / `"x86_64"` → native execution
/// - `"i686"` / `"i386"` → `qemu-i386` (or native if 32-bit support available)
/// - `"aarch64"` / `"arm64"` → `qemu-aarch64`
/// - `"riscv64"` → `qemu-riscv64`
///
/// # Returns
/// The `Output` from running the binary.
///
/// # Panics
/// Panics if the binary cannot be executed or if the required QEMU emulator
/// is not installed for cross-architecture targets.
pub fn run_binary(path: &str, target: &str) -> Output {
    match target {
        "x86-64" | "x86_64" => Command::new(path)
            .output()
            .unwrap_or_else(|e| panic!("Failed to run binary '{}': {}", path, e)),
        "i686" | "i386" => {
            // Try direct execution first (works on x86-64 hosts with 32-bit support),
            // fall back to qemu-i386 if direct execution fails
            if let Ok(output) = Command::new(path).output() {
                if output.status.success() {
                    output
                } else {
                    // Direct execution failed — retry with QEMU + LD prefix
                    Command::new("qemu-i386")
                        .env("QEMU_LD_PREFIX", "/usr/i686-linux-gnu")
                        .arg(path)
                        .output()
                        .unwrap_or_else(|e| {
                            panic!("Failed to run i686 binary '{}' via qemu-i386: {}", path, e)
                        })
                }
            } else {
                Command::new("qemu-i386")
                    .env("QEMU_LD_PREFIX", "/usr/i686-linux-gnu")
                    .arg(path)
                    .output()
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to run i686 binary '{}' (neither direct nor via qemu-i386): {}",
                            path, e
                        )
                    })
            }
        }
        "aarch64" | "arm64" => Command::new("qemu-aarch64")
            .env("QEMU_LD_PREFIX", "/usr/aarch64-linux-gnu")
            .arg(path)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run aarch64 binary '{}' via qemu-aarch64: {}.\n\
                         Is qemu-user installed?",
                    path, e
                )
            }),
        "riscv64" => Command::new("qemu-riscv64")
            .env("QEMU_LD_PREFIX", "/usr/riscv64-linux-gnu")
            .arg(path)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run riscv64 binary '{}' via qemu-riscv64: {}.\n\
                         Is qemu-user installed?",
                    path, e
                )
            }),
        other => {
            panic!(
                "Unsupported target architecture for binary execution: '{}'",
                other
            );
        }
    }
}

/// Checks whether a QEMU user-mode emulator is available for the given
/// architecture.
///
/// # Returns
/// `true` if the required emulator binary is in PATH and responds to
/// `--version`, `false` otherwise.
///
/// # Architecture Mapping
/// - `"x86-64"` / `"x86_64"` → always `true` (native execution)
/// - `"i686"` / `"i386"` → checks `qemu-i386 --version`
/// - `"aarch64"` / `"arm64"` → checks `qemu-aarch64 --version`
/// - `"riscv64"` → checks `qemu-riscv64 --version`
pub fn qemu_available(arch: &str) -> bool {
    let qemu_binary = match arch {
        "x86-64" | "x86_64" => return true, // native execution
        "i686" | "i386" => "qemu-i386",
        "aarch64" | "arm64" => "qemu-aarch64",
        "riscv64" => "qemu-riscv64",
        _ => return false,
    };

    Command::new(qemu_binary)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Temporary File Management
// ---------------------------------------------------------------------------

/// Generates a unique temporary file path for test output.
///
/// Uses the system temp directory combined with a process-ID and atomic counter
/// suffix to guarantee uniqueness even under concurrent test execution.
///
/// The file at this path does **not** need to exist yet — the returned path is
/// intended as a `-o` argument to BCC compilation commands.
///
/// # Arguments
/// * `name` — A descriptive name embedded in the filename for identification.
///
/// # Returns
/// A `PathBuf` like `/tmp/bcc_test_<name>_<pid>_<counter>`.
pub fn temp_output_path(name: &str) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let filename = format!("bcc_test_{}_{pid}_{counter}", name);
    env::temp_dir().join(filename)
}

/// Silently removes a list of temporary files.
///
/// Errors (e.g., file does not exist because compilation failed) are
/// deliberately ignored — this is a best-effort cleanup helper.
///
/// # Arguments
/// * `paths` — Slice of file path strings to remove.
pub fn cleanup_temp_files(paths: &[&str]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

/// RAII guard for a temporary test file. Automatically removes the file when
/// the guard is dropped, ensuring cleanup even on test panics.
///
/// # Example
/// ```rust,ignore
/// let tmp = TempTestFile::new("my_test_output");
/// // Use tmp.path() as the -o target for BCC
/// // File is automatically cleaned up when `tmp` goes out of scope
/// ```
pub struct TempTestFile {
    /// The filesystem path of the temporary file.
    pub path: PathBuf,
}

impl TempTestFile {
    /// Creates a new `TempTestFile` with a unique path derived from `name`.
    ///
    /// The underlying file is **not** created — only the path is reserved.
    /// BCC or other tools write to this path during tests.
    pub fn new(name: &str) -> Self {
        TempTestFile {
            path: temp_output_path(name),
        }
    }

    /// Returns a reference to the temporary file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempTestFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Fixture Path Resolution
// ---------------------------------------------------------------------------

/// Resolves the path to a test fixture file under `tests/fixtures/`.
///
/// Uses `CARGO_MANIFEST_DIR` for absolute path resolution when available,
/// falling back to a relative path from the current directory.
///
/// # Arguments
/// * `name` — Relative path within the fixtures directory (e.g., `"hello.c"`,
///   `"shared_lib/foo.c"`, `"security/retpoline.c"`).
///
/// # Returns
/// Absolute `PathBuf` to the fixture file.
pub fn fixture_path(name: &str) -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest_dir)
        .join("tests")
        .join("fixtures")
        .join(name)
}
