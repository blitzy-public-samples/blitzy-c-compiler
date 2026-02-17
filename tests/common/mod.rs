//! Shared test utilities module for the BCC integration test suite.
//!
//! Provides common test infrastructure imported by all seven checkpoint test files
//! via `mod common;`. Contains:
//!
//! - BCC binary invocation helpers (compile, compile_to_binary, compile_to_object, etc.)
//! - stdout/stderr capture utilities for verifying compiler output
//! - ELF section inspection wrappers (readelf/objdump subprocess invocations)
//! - Architecture-specific QEMU invocation helpers for cross-architecture execution
//! - Assertion helpers for ELF validation (header checks, section existence, disassembly patterns)
//! - RAII temporary directory management for test artifact cleanup
//! - Wall-clock timing helpers for performance ceiling validation
//!
//! No external dependencies — uses only the Rust standard library.

// Each checkpoint test file includes this module via `mod common;` but only
// uses a subset of the utilities. Suppress dead-code warnings for the functions
// that are not used by a particular test file.
#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// BccOutput — captured result of a subprocess invocation
// ---------------------------------------------------------------------------

/// Captured output from invoking the BCC compiler or executing a compiled binary.
///
/// Wraps the process exit status together with the full stdout and stderr output
/// decoded as UTF-8 (lossy). Every compilation and execution helper in this
/// module returns a `BccOutput`.
#[derive(Debug)]
pub struct BccOutput {
    /// Process exit status (success / failure / signal).
    pub status: ExitStatus,
    /// Captured standard output as a UTF-8 string.
    pub stdout: String,
    /// Captured standard error as a UTF-8 string.
    pub stderr: String,
}

impl BccOutput {
    /// Returns `true` if the subprocess exited successfully (exit code 0).
    ///
    /// Convenience wrapper around `ExitStatus::success()`.
    pub fn success(&self) -> bool {
        self.status.success()
    }

    /// Returns the exit code of the process, or `None` if terminated by signal.
    ///
    /// Convenience wrapper around `ExitStatus::code()`.
    pub fn exit_code(&self) -> Option<i32> {
        self.status.code()
    }

    /// Assert that the compilation or execution succeeded (exit code 0).\
    ///
    /// # Panics
    ///
    /// Panics with a diagnostic message including stderr if the process did
    /// not exit successfully.
    pub fn assert_success(&self) {
        assert!(
            self.status.success(),
            "Process exited with code {:?} (expected success).\nstderr:\n{}",
            self.status.code(),
            self.stderr
        );
    }

    /// Assert that the compilation or execution failed (non-zero exit code).
    ///
    /// # Panics
    ///
    /// Panics if the process exited with code 0.
    pub fn assert_failure(&self) {
        assert!(
            !self.status.success(),
            "Process exited successfully (expected failure).\nstdout:\n{}",
            self.stdout
        );
    }
}

// ---------------------------------------------------------------------------
// BCC binary location
// ---------------------------------------------------------------------------

/// Locate the compiled `bcc` binary within the Cargo target directory.
///
/// Search strategy:
/// 1. `$CARGO_MANIFEST_DIR/target/release/bcc` (tests normally run in `--release`)
/// 2. `$CARGO_MANIFEST_DIR/target/debug/bcc` (fallback for debug builds)
/// 3. Relative `target/release/bcc` from the current working directory
/// 4. Relative `target/debug/bcc` from the current working directory
///
/// # Panics
///
/// Panics if the `bcc` binary cannot be found at any of the searched locations.
pub fn bcc_binary_path() -> PathBuf {
    // Try resolving from CARGO_MANIFEST_DIR first (set by Cargo during test runs).
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let base = PathBuf::from(&manifest_dir);
        let release_path = base.join("target").join("release").join("bcc");
        if release_path.exists() {
            return release_path;
        }
        let debug_path = base.join("target").join("debug").join("bcc");
        if debug_path.exists() {
            return debug_path;
        }
    }

    // Fallback: resolve relative to the current working directory.
    if let Ok(cwd) = env::current_dir() {
        let release_path = cwd.join("target").join("release").join("bcc");
        if release_path.exists() {
            return release_path;
        }
        let debug_path = cwd.join("target").join("debug").join("bcc");
        if debug_path.exists() {
            return debug_path;
        }
    }

    panic!(
        "Could not locate the `bcc` binary. Ensure the project has been built \
         (`cargo build --release`) before running integration tests."
    );
}

// ---------------------------------------------------------------------------
// BCC compilation helpers
// ---------------------------------------------------------------------------

/// Invoke the BCC compiler with a source file and additional CLI arguments.
///
/// Runs `<bcc_binary> <source> <args...>`, captures stdout/stderr, and returns
/// a `BccOutput` with the full result.
///
/// # Arguments
///
/// * `source` — Path to the C source file to compile.
/// * `args`   — Additional CLI flags (e.g., `["-c", "-o", "out.o"]`).
///
/// # Panics
///
/// Panics if the subprocess cannot be spawned (e.g., binary not found).
pub fn compile(source: &str, args: &[&str]) -> BccOutput {
    let bcc = bcc_binary_path();
    let output = Command::new(&bcc)
        .arg(source)
        .args(args)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to execute BCC compiler at '{}': {}",
                bcc.display(),
                e
            )
        });

    BccOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Compile a C source file to a native binary for the specified target architecture.
///
/// Equivalent to: `bcc --target=<target> -o <output> <extra_args...> <source>`
///
/// # Arguments
///
/// * `source`     — Path to the C source file.
/// * `output`     — Path for the output binary.
/// * `target`     — Target architecture string (`"x86-64"`, `"i686"`, `"aarch64"`, `"riscv64"`).
/// * `extra_args` — Any additional CLI flags (e.g., `["-g"]`).
pub fn compile_to_binary(
    source: &str,
    output: &str,
    target: &str,
    extra_args: &[&str],
) -> BccOutput {
    let bcc = bcc_binary_path();
    let target_flag = format!("--target={}", target);

    let mut cmd = Command::new(&bcc);
    cmd.arg(source).arg(&target_flag).arg("-o").arg(output);
    cmd.args(extra_args);

    let result = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to execute BCC compiler at '{}': {}",
            bcc.display(),
            e
        )
    });

    BccOutput {
        status: result.status,
        stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
    }
}

/// Compile a C source file to a relocatable object file (`.o`) with the `-c` flag.
///
/// Equivalent to: `bcc -c --target=<target> -o <output> <extra_args...> <source>`
///
/// # Arguments
///
/// * `source`     — Path to the C source file.
/// * `output`     — Path for the output object file.
/// * `target`     — Target architecture string.
/// * `extra_args` — Any additional CLI flags.
pub fn compile_to_object(
    source: &str,
    output: &str,
    target: &str,
    extra_args: &[&str],
) -> BccOutput {
    let bcc = bcc_binary_path();
    let target_flag = format!("--target={}", target);

    let mut cmd = Command::new(&bcc);
    cmd.arg(source)
        .arg("-c")
        .arg(&target_flag)
        .arg("-o")
        .arg(output);
    cmd.args(extra_args);

    let result = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to execute BCC compiler at '{}': {}",
            bcc.display(),
            e
        )
    });

    BccOutput {
        status: result.status,
        stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
    }
}

/// Run the preprocessor only (`-E` flag) on a C source file.
///
/// Equivalent to: `bcc -E <extra_args...> <source>`
///
/// Returns the preprocessed token stream on stdout.
///
/// # Arguments
///
/// * `source`     — Path to the C source file.
/// * `extra_args` — Any additional CLI flags (e.g., `["-DFOO=1", "-I/usr/include"]`).
pub fn preprocess_only(source: &str, extra_args: &[&str]) -> BccOutput {
    let bcc = bcc_binary_path();

    let mut cmd = Command::new(&bcc);
    cmd.arg("-E").arg(source);
    cmd.args(extra_args);

    let result = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to execute BCC compiler at '{}': {}",
            bcc.display(),
            e
        )
    });

    BccOutput {
        status: result.status,
        stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
    }
}

/// Compile a C source file to a shared library with `-fPIC -shared` flags.
///
/// Equivalent to: `bcc -fPIC -shared --target=<target> -o <output> <extra_args...> <source>`
///
/// # Arguments
///
/// * `source`     — Path to the C source file.
/// * `output`     — Path for the output shared object (e.g., `"libfoo.so"`).
/// * `target`     — Target architecture string.
/// * `extra_args` — Any additional CLI flags.
pub fn compile_shared_lib(
    source: &str,
    output: &str,
    target: &str,
    extra_args: &[&str],
) -> BccOutput {
    let bcc = bcc_binary_path();
    let target_flag = format!("--target={}", target);

    let mut cmd = Command::new(&bcc);
    cmd.arg(source)
        .arg("-fPIC")
        .arg("-shared")
        .arg(&target_flag)
        .arg("-o")
        .arg(output);
    cmd.args(extra_args);

    let result = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to execute BCC compiler at '{}': {}",
            bcc.display(),
            e
        )
    });

    BccOutput {
        status: result.status,
        stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
    }
}

// ---------------------------------------------------------------------------
// Binary execution helpers
// ---------------------------------------------------------------------------

/// Execute a native binary, capturing stdout, stderr, and exit code.
///
/// # Arguments
///
/// * `path` — Path to the ELF binary to execute.
///
/// # Panics
///
/// Panics if the subprocess cannot be spawned.
pub fn run_binary(path: &str) -> BccOutput {
    let output = Command::new(path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute binary '{}': {}", path, e));

    BccOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Execute a compiled binary through the appropriate QEMU user-mode emulator
/// for the given target architecture.
///
/// Mapping:
/// - `"aarch64"` → `qemu-aarch64 <path>`
/// - `"riscv64"` → `qemu-riscv64 <path>`
/// - `"i686"`    → `qemu-i386 <path>`
/// - `"x86-64"`  → direct execution (native host assumed)
///
/// # Panics
///
/// Panics if the target is unrecognised or the subprocess cannot be spawned.
pub fn run_binary_with_qemu(path: &str, target: &str) -> BccOutput {
    match target {
        "x86-64" => run_binary(path),
        "aarch64" => {
            let output = Command::new("qemu-aarch64")
                .env("QEMU_LD_PREFIX", "/usr/aarch64-linux-gnu")
                .arg(path)
                .output()
                .unwrap_or_else(|e| panic!("Failed to execute '{}' via qemu-aarch64: {}", path, e));
            BccOutput {
                status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }
        }
        "riscv64" => {
            let output = Command::new("qemu-riscv64")
                .env("QEMU_LD_PREFIX", "/usr/riscv64-linux-gnu")
                .arg(path)
                .output()
                .unwrap_or_else(|e| panic!("Failed to execute '{}' via qemu-riscv64: {}", path, e));
            BccOutput {
                status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }
        }
        "i686" => {
            let output = Command::new("qemu-i386")
                .arg(path)
                .output()
                .unwrap_or_else(|e| panic!("Failed to execute '{}' via qemu-i386: {}", path, e));
            BccOutput {
                status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            }
        }
        other => panic!(
            "Unsupported target architecture for QEMU execution: '{}'. \
             Expected one of: x86-64, i686, aarch64, riscv64",
            other
        ),
    }
}

/// Dispatch binary execution to either direct execution (for native x86-64)
/// or the appropriate QEMU user-mode emulator based on the target architecture.
///
/// This is the primary entry point for running compiled test binaries across
/// all four supported architectures.
///
/// # Arguments
///
/// * `path`   — Path to the compiled ELF binary.
/// * `target` — Target architecture string.
pub fn run_binary_for_target(path: &str, target: &str) -> BccOutput {
    run_binary_with_qemu(path, target)
}

// ---------------------------------------------------------------------------
// ELF inspection helpers — readelf wrappers
// ---------------------------------------------------------------------------

/// Internal helper: run a command and return its stdout as a String.
///
/// Panics with a descriptive message if the command fails to execute.
fn run_tool(program: &str, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute '{}': {}", program, e));

    // Return stdout even if the exit code is non-zero; some readelf invocations
    // return warnings on stderr while still producing useful stdout output.
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Invoke `readelf -h <path>` and return the ELF header summary.
pub fn readelf_header(path: &str) -> String {
    run_tool("readelf", &["-h", path])
}

/// Invoke `readelf -S <path>` and return the section header table.
pub fn readelf_sections(path: &str) -> String {
    run_tool("readelf", &["-S", path])
}

/// Invoke `readelf -l <path>` and return the program header table.
pub fn readelf_program_headers(path: &str) -> String {
    run_tool("readelf", &["-l", path])
}

/// Invoke `readelf -s <path>` and return the symbol table.
pub fn readelf_symbols(path: &str) -> String {
    run_tool("readelf", &["-sW", path])
}

/// Invoke `readelf -d <path>` and return the `.dynamic` section entries.
pub fn readelf_dynamic(path: &str) -> String {
    run_tool("readelf", &["-d", path])
}

/// Invoke `readelf --dyn-syms <path>` and return the dynamic symbol table.
pub fn readelf_dyn_symbols(path: &str) -> String {
    run_tool("readelf", &["--dyn-syms", path])
}

/// Invoke `readelf -r <path>` and return the relocation entries.
pub fn readelf_relocations(path: &str) -> String {
    run_tool("readelf", &["-r", path])
}

/// Invoke `readelf --debug-dump=info <path>` and return DWARF `.debug_info`.
pub fn readelf_debug_info(path: &str) -> String {
    run_tool("readelf", &["--debug-dump=info", path])
}

/// Invoke `readelf --debug-dump=line <path>` and return DWARF `.debug_line`.
pub fn readelf_debug_line(path: &str) -> String {
    run_tool("readelf", &["--debug-dump=line", path])
}

// ---------------------------------------------------------------------------
// Disassembly helpers — objdump wrappers
// ---------------------------------------------------------------------------

/// Invoke `objdump -d -r <path>` and return the disassembly output
/// with interleaved relocation entries.  The `-r` flag causes `objdump`
/// to annotate call/jump targets with their relocation symbol names,
/// which is essential for validating retpoline thunk references in
/// relocatable object files where targets are not yet resolved.
pub fn objdump_disassemble(path: &str) -> String {
    run_tool("objdump", &["-d", "-r", path])
}

/// Invoke `objdump -s -j <section> <path>` and return the hex dump of a
/// specific ELF section.
///
/// # Arguments
///
/// * `path`    — Path to the ELF binary or object file.
/// * `section` — Section name (e.g., `".rodata"`, `".text"`).
pub fn objdump_section_contents(path: &str, section: &str) -> String {
    run_tool("objdump", &["-s", "-j", section, path])
}

/// Invoke `objdump -d -r -t <path>` and return the full disassembly with
/// relocation entries and symbol table.
pub fn objdump_full(path: &str) -> String {
    run_tool("objdump", &["-d", "-r", "-t", path])
}

// ---------------------------------------------------------------------------
// ELF assertion helpers
// ---------------------------------------------------------------------------

/// Assert that the `e_machine` field in the ELF header matches the expected value.
///
/// Reads the ELF header via `readelf -h` and searches for the `Machine:` line.
///
/// # Arguments
///
/// * `path`     — Path to the ELF file.
/// * `expected` — Expected machine string (e.g., `"Advanced Micro Devices X86-64"`,
///   `"Intel 80386"`, `"AArch64"`, `"RISC-V"`).
///
/// # Panics
///
/// Panics if the Machine field is not found or does not contain `expected`.
pub fn assert_elf_machine(path: &str, expected: &str) {
    let header = readelf_header(path);
    let machine_line = header
        .lines()
        .find(|line| line.contains("Machine:"))
        .unwrap_or_else(|| {
            panic!(
                "ELF header for '{}' does not contain a 'Machine:' field.\nHeader:\n{}",
                path, header
            )
        });

    assert!(
        machine_line.contains(expected),
        "ELF machine mismatch for '{}'.\n  Expected to contain: '{}'\n  Actual Machine line: '{}'",
        path,
        expected,
        machine_line.trim()
    );
}

/// Assert that the ELF class matches the expected value (`"ELF64"` or `"ELF32"`).
///
/// # Panics
///
/// Panics if the Class field is not found or does not contain `expected`.
pub fn assert_elf_class(path: &str, expected: &str) {
    let header = readelf_header(path);
    let class_line = header
        .lines()
        .find(|line| line.contains("Class:"))
        .unwrap_or_else(|| {
            panic!(
                "ELF header for '{}' does not contain a 'Class:' field.\nHeader:\n{}",
                path, header
            )
        });

    assert!(
        class_line.contains(expected),
        "ELF class mismatch for '{}'.\n  Expected to contain: '{}'\n  Actual Class line: '{}'",
        path,
        expected,
        class_line.trim()
    );
}

/// Assert that the ELF type matches the expected value (`"EXEC"`, `"DYN"`, or `"REL"`).
///
/// # Panics
///
/// Panics if the Type field is not found or does not contain `expected`.
pub fn assert_elf_type(path: &str, expected: &str) {
    let header = readelf_header(path);
    let type_line = header
        .lines()
        .find(|line| line.contains("Type:"))
        .unwrap_or_else(|| {
            panic!(
                "ELF header for '{}' does not contain a 'Type:' field.\nHeader:\n{}",
                path, header
            )
        });

    assert!(
        type_line.contains(expected),
        "ELF type mismatch for '{}'.\n  Expected to contain: '{}'\n  Actual Type line: '{}'",
        path,
        expected,
        type_line.trim()
    );
}

/// Assert that a named section exists in the ELF section header table.
///
/// # Arguments
///
/// * `path`         — Path to the ELF file.
/// * `section_name` — Expected section name (e.g., `".text"`, `".dynamic"`).
///
/// # Panics
///
/// Panics if the section is not found.
pub fn assert_section_exists(path: &str, section_name: &str) {
    let sections = readelf_sections(path);
    assert!(
        sections.contains(section_name),
        "Expected section '{}' to exist in '{}'.\nSection headers:\n{}",
        section_name,
        path,
        sections
    );
}

/// Assert that a named section does **not** exist in the ELF section header table.
///
/// # Arguments
///
/// * `path`         — Path to the ELF file.
/// * `section_name` — Section name that must be absent.
///
/// # Panics
///
/// Panics if the section is found.
pub fn assert_section_absent(path: &str, section_name: &str) {
    let sections = readelf_sections(path);
    assert!(
        !sections.contains(section_name),
        "Expected section '{}' to be ABSENT from '{}'.\nSection headers:\n{}",
        section_name,
        path,
        sections
    );
}

/// Assert that no `.debug_*` sections exist in the ELF file.
///
/// Per Section 0.7.10: a binary compiled without `-g` must not contain any
/// debug sections — zero debug section leakage.
///
/// # Panics
///
/// Panics if any `.debug_` section is found in the section headers.
pub fn assert_no_debug_sections(path: &str) {
    let sections = readelf_sections(path);
    for line in sections.lines() {
        assert!(
            !line.contains(".debug_"),
            "Found unexpected debug section in '{}' (compiled without -g).\n  Line: '{}'\nFull section headers:\n{}",
            path,
            line.trim(),
            sections
        );
    }
}

/// Assert that the four required DWARF v4 debug sections are all present.
///
/// Checks for: `.debug_info`, `.debug_abbrev`, `.debug_line`, `.debug_str`.
///
/// # Panics
///
/// Panics if any of the four sections is missing.
pub fn assert_has_debug_sections(path: &str) {
    let required = [".debug_info", ".debug_abbrev", ".debug_line", ".debug_str"];
    let sections = readelf_sections(path);

    for section_name in &required {
        assert!(
            sections.contains(section_name),
            "Expected DWARF section '{}' to exist in '{}' (compiled with -g).\nSection headers:\n{}",
            section_name,
            path,
            sections
        );
    }
}

/// Assert that the disassembly output of an ELF file contains a given pattern.
///
/// Uses `objdump -d` to disassemble and searches the output for `pattern`.
///
/// # Arguments
///
/// * `path`    — Path to the ELF binary or object file.
/// * `pattern` — Substring expected in the disassembly (e.g., `"endbr64"`, `"__x86_indirect_thunk"`).
///
/// # Panics
///
/// Panics if the pattern is not found.
pub fn assert_disassembly_contains(path: &str, pattern: &str) {
    let disasm = objdump_disassemble(path);
    assert!(
        disasm.contains(pattern),
        "Expected disassembly of '{}' to contain '{}'.\nDisassembly (first 2000 chars):\n{}",
        path,
        pattern,
        &disasm[..disasm.len().min(2000)]
    );
}

/// Assert that the disassembly output of an ELF file does **not** contain a
/// given pattern.
///
/// # Panics
///
/// Panics if the pattern is found.
pub fn assert_disassembly_not_contains(path: &str, pattern: &str) {
    let disasm = objdump_disassemble(path);
    assert!(
        !disasm.contains(pattern),
        "Expected disassembly of '{}' to NOT contain '{}'.\nDisassembly (first 2000 chars):\n{}",
        path,
        pattern,
        &disasm[..disasm.len().min(2000)]
    );
}

/// Assert that a symbol with the given name exists in the ELF symbol table.
///
/// Uses `readelf -s` and searches for the symbol name.
///
/// # Panics
///
/// Panics if the symbol is not found.
pub fn assert_symbol_exists(path: &str, symbol_name: &str) {
    let symbols = readelf_symbols(path);
    assert!(
        symbols.contains(symbol_name),
        "Expected symbol '{}' to exist in '{}'.\nSymbol table:\n{}",
        symbol_name,
        path,
        symbols
    );
}

/// Assert that a specific entry exists in the `.dynamic` section.
///
/// Uses `readelf -d` and searches for the entry string.
///
/// # Arguments
///
/// * `path`  — Path to the ELF file.
/// * `entry` — Expected dynamic entry string (e.g., `"NEEDED"`, `"SONAME"`).
///
/// # Panics
///
/// Panics if the entry is not found.
pub fn assert_dynamic_entry_exists(path: &str, entry: &str) {
    let dynamic = readelf_dynamic(path);
    assert!(
        dynamic.contains(entry),
        "Expected dynamic entry '{}' to exist in '{}'.\nDynamic section:\n{}",
        entry,
        path,
        dynamic
    );
}

/// Assert that a specific program header type exists in the ELF program header
/// table.
///
/// Uses `readelf -l` and searches for the header type string.
///
/// # Arguments
///
/// * `path`        — Path to the ELF file.
/// * `header_type` — Expected program header type (e.g., `"DYNAMIC"`, `"INTERP"`, `"LOAD"`).
///
/// # Panics
///
/// Panics if the header type is not found.
pub fn assert_program_header_exists(path: &str, header_type: &str) {
    let headers = readelf_program_headers(path);
    assert!(
        headers.contains(header_type),
        "Expected program header type '{}' to exist in '{}'.\nProgram headers:\n{}",
        header_type,
        path,
        headers
    );
}

// ---------------------------------------------------------------------------
// Temporary directory management (RAII)
// ---------------------------------------------------------------------------

/// RAII temporary directory for test artifacts.
///
/// Creates a uniquely-named directory under the system temp directory on
/// construction, and recursively removes it (including all contents) when
/// dropped. This ensures compiled objects, ELF binaries, and shared libraries
/// produced during a test are cleaned up automatically.
pub struct TestDir {
    /// Absolute path to the temporary directory.
    path: PathBuf,
}

impl TestDir {
    /// Create a new temporary directory for the given test.
    ///
    /// The directory is created under `std::env::temp_dir()` with a name
    /// derived from `test_name` and the current process ID for uniqueness.
    ///
    /// # Panics
    ///
    /// Panics if the directory cannot be created.
    pub fn new(test_name: &str) -> Self {
        let mut dir = env::temp_dir();
        // Include PID and a monotonic counter fragment for uniqueness when
        // tests run in parallel.
        let unique_name = format!(
            "bcc_test_{}_{}_{}",
            test_name,
            std::process::id(),
            Instant::now().elapsed().subsec_nanos()
        );
        dir.push(unique_name);

        fs::create_dir_all(&dir).unwrap_or_else(|e| {
            panic!("Failed to create test directory '{}': {}", dir.display(), e)
        });

        TestDir { path: dir }
    }

    /// Get a reference to the temporary directory path.
    pub fn path(&self) -> &Path {
        self.path.as_path()
    }

    /// Construct the full path for a file within this temporary directory.
    ///
    /// # Arguments
    ///
    /// * `name` — File name (e.g., `"hello.o"`, `"libfoo.so"`).
    pub fn file_path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDir {
    /// Recursively remove the temporary directory and all its contents.
    ///
    /// Errors during cleanup are silently ignored to avoid masking test
    /// failures with cleanup panics.
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Wall-clock timing helper
// ---------------------------------------------------------------------------

/// Execute a closure and measure wall-clock time using a monotonic clock.
///
/// Returns a tuple of `(result, duration)` where `result` is the closure's
/// return value and `duration` is the elapsed time.
///
/// Used by checkpoint tests for the 5× GCC wall-clock ceiling validation
/// (Section 0.7.8) and the 5-second recursive macro timeout (Section 0.1.2).
///
/// # Example
///
/// ```ignore
/// let (output, elapsed) = timed_execution(|| compile("test.c", &["-o", "test"]));
/// assert!(elapsed < Duration::from_secs(5), "compilation took too long");
/// ```
pub fn timed_execution<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Default timeout duration for operations that should complete quickly
/// (e.g., recursive macro expansion must terminate within 5 seconds per
/// Section 0.1.2).
///
/// # Returns
///
/// A `Duration` of 5 seconds.
pub fn default_timeout() -> Duration {
    Duration::from_secs(5)
}

/// Assert that a timed operation completed within the given number of seconds.
///
/// Uses `Duration::as_secs()` for comparison against wall-clock ceilings.
///
/// # Arguments
///
/// * `elapsed`     — Measured duration of the operation.
/// * `max_seconds` — Maximum allowed wall-clock seconds.
/// * `label`       — Human-readable label for error messages.
///
/// # Panics
///
/// Panics if `elapsed` exceeds `max_seconds`.
pub fn assert_within_timeout(elapsed: Duration, max_seconds: u64, label: &str) {
    let limit = Duration::from_secs(max_seconds);
    assert!(
        elapsed.as_secs() <= limit.as_secs(),
        "{} took {} seconds, exceeding the {}-second ceiling.",
        label,
        elapsed.as_secs(),
        max_seconds
    );
}

// ---------------------------------------------------------------------------
// Fixtures path helper
// ---------------------------------------------------------------------------

/// Resolve a path relative to the `tests/fixtures/` directory.
///
/// Handles both running from the project root (via `cargo test`) and running
/// from a test binary location by trying multiple resolution strategies:
///
/// 1. `$CARGO_MANIFEST_DIR/tests/fixtures/<relative>`
/// 2. `<cwd>/tests/fixtures/<relative>`
///
/// # Arguments
///
/// * `relative` — Relative path within the fixtures directory
///   (e.g., `"hello.c"`, `"shared_lib/foo.c"`, `"security/retpoline.c"`).
///
/// # Panics
///
/// Panics if the fixture file cannot be located.
pub fn fixture_path(relative: &str) -> PathBuf {
    // Strategy 1: Use CARGO_MANIFEST_DIR (set by Cargo during test runs).
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let path = Path::new(&manifest_dir)
            .join("tests")
            .join("fixtures")
            .join(relative);
        if path.exists() {
            return path;
        }
    }

    // Strategy 2: Resolve from the current working directory.
    if let Ok(cwd) = env::current_dir() {
        let path = cwd.join("tests").join("fixtures").join(relative);
        if path.exists() {
            return path;
        }
    }

    // Return the CARGO_MANIFEST_DIR-based path even if it doesn't exist yet,
    // so that test code gets a meaningful path in error messages. The fixture
    // file may be created by a parallel agent or a setup step.
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        return Path::new(&manifest_dir)
            .join("tests")
            .join("fixtures")
            .join(relative);
    }

    // Absolute fallback.
    PathBuf::from("tests").join("fixtures").join(relative)
}
