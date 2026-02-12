//! Checkpoint 1 — Hello World Integration Test Suite
//!
//! This is the **first sequential hard gate** in the BCC validation protocol.
//! All subsequent checkpoints (2–7) depend on this checkpoint passing.
//!
//! Tests the fundamental end-to-end compilation pipeline for all four target
//! architectures: x86-64, i686, AArch64, and RISC-V 64.
//!
//! For each architecture, the test:
//! 1. Compiles `tests/fixtures/hello.c` to a native ELF executable
//! 2. Validates the ELF headers (e_machine, EI_CLASS, ELF type)
//! 3. Executes the binary (natively or via QEMU user-mode emulation)
//! 4. Asserts stdout == "Hello, World!\n" and exit code == 0
//!
//! Additionally validates:
//! - The `-c` flag produces a relocatable object file (ET_REL)
//! - The `-E` flag produces preprocessed source output (no object code)
//!
//! Backend validation order per Section 0.1.2:
//!   x86-64 → i686 → AArch64 → RISC-V 64
//!
//! Cross-architecture execution uses QEMU user-mode emulation:
//! - `qemu-aarch64` for AArch64 binaries
//! - `qemu-riscv64` for RISC-V 64 binaries
//! - `qemu-i386` for i686 binaries
//! - Direct execution for x86-64 (native host assumed)

mod common;

use std::path::Path;
// Command is imported per the schema specification — all direct subprocess
// invocations in this module are delegated to the `common` test utilities,
// but Command remains available for any ad-hoc subprocess needs during
// test development and debugging.
#[allow(unused_imports)]
use std::process::Command;

// ---------------------------------------------------------------------------
// Constants — expected output and ELF header values
// ---------------------------------------------------------------------------

/// Expected stdout from the Hello World program across all architectures.
const EXPECTED_STDOUT: &str = "Hello, World!\n";

/// Expected exit code for a successful Hello World execution.
const EXPECTED_EXIT_CODE: i32 = 0;

/// ELF machine string for x86-64 as reported by `readelf -h`.
const ELF_MACHINE_X86_64: &str = "Advanced Micro Devices X86-64";

/// ELF machine string for i686 (IA-32) as reported by `readelf -h`.
const ELF_MACHINE_I686: &str = "Intel 80386";

/// ELF machine string for AArch64 as reported by `readelf -h`.
const ELF_MACHINE_AARCH64: &str = "AArch64";

/// ELF machine string for RISC-V as reported by `readelf -h`.
const ELF_MACHINE_RISCV: &str = "RISC-V";

/// ELF class for 64-bit binaries.
const ELF_CLASS_64: &str = "ELF64";

/// ELF class for 32-bit binaries.
const ELF_CLASS_32: &str = "ELF32";

/// ELF type for executable binaries.
const ELF_TYPE_EXEC: &str = "EXEC";

/// ELF type for relocatable object files.
const ELF_TYPE_REL: &str = "REL";

// ---------------------------------------------------------------------------
// Helper: run a full Hello World test for a given target architecture
// ---------------------------------------------------------------------------

/// Execute the complete Hello World validation sequence for one target
/// architecture:
///
/// 1. Resolve the fixture path for `hello.c`
/// 2. Compile to a native ELF binary using `compile_to_binary()`
/// 3. Assert compilation succeeded (exit code 0)
/// 4. Verify the output binary exists on disk
/// 5. Validate ELF headers (machine, class, type)
/// 6. Execute the binary via `run_binary_for_target()` (with QEMU dispatch)
/// 7. Assert stdout matches `EXPECTED_STDOUT`
/// 8. Assert exit code matches `EXPECTED_EXIT_CODE`
///
/// # Arguments
///
/// * `target`           — Target architecture string (`"x86-64"`, `"i686"`,
///                        `"aarch64"`, `"riscv64"`).
/// * `binary_name`      — Output binary filename (e.g., `"hello_x86_64"`).
/// * `expected_machine` — Expected ELF `Machine:` field substring.
/// * `expected_class`   — Expected ELF `Class:` field value (`"ELF64"` or `"ELF32"`).
fn run_hello_world_test(
    target: &str,
    binary_name: &str,
    expected_machine: &str,
    expected_class: &str,
) {
    // --- Setup: create a temporary directory for test artifacts ---
    let test_dir = common::TestDir::new(&format!("checkpoint1_{}", target));
    let output_path = test_dir.file_path(binary_name);
    let output_str = output_path.to_str().expect("output path must be valid UTF-8");

    // --- Resolve the hello.c fixture path ---
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path must be valid UTF-8");

    // Verify the fixture file exists before attempting compilation.
    assert!(
        Path::new(source_str).exists(),
        "Test fixture 'hello.c' not found at '{}'. \
         Ensure tests/fixtures/hello.c has been created.",
        source_str
    );

    // --- Step 1: Compile hello.c to a native binary ---
    let compile_result = common::compile_to_binary(source_str, output_str, target, &[]);

    // Assert compilation succeeded.
    compile_result.assert_success();

    // Verify the output binary was actually produced on disk.
    assert!(
        Path::new(output_str).exists(),
        "Compiled binary '{}' was not produced despite successful compilation.\n\
         Compiler stdout:\n{}\nCompiler stderr:\n{}",
        output_str,
        compile_result.stdout,
        compile_result.stderr
    );

    // --- Step 2: Validate ELF headers ---
    // Verify e_machine matches the expected architecture.
    common::assert_elf_machine(output_str, expected_machine);

    // Verify EI_CLASS matches the expected bitness (ELF64 or ELF32).
    common::assert_elf_class(output_str, expected_class);

    // Verify ELF type is EXEC (executable), not DYN or REL.
    common::assert_elf_type(output_str, ELF_TYPE_EXEC);

    // --- Step 3: Execute the compiled binary ---
    let run_result = common::run_binary_for_target(output_str, target);

    // Assert execution succeeded with exit code 0.
    run_result.assert_success();

    // Verify the exit code is exactly 0 (not just "success").
    assert_eq!(
        run_result.exit_code(),
        Some(EXPECTED_EXIT_CODE),
        "Hello World ({}) exited with code {:?}, expected {}.\nstderr:\n{}",
        target,
        run_result.exit_code(),
        EXPECTED_EXIT_CODE,
        run_result.stderr
    );

    // --- Step 4: Validate stdout output ---
    assert_eq!(
        run_result.stdout, EXPECTED_STDOUT,
        "Hello World ({}) produced unexpected stdout.\n\
         Expected: {:?}\n\
         Actual:   {:?}\n\
         stderr:\n{}",
        target, EXPECTED_STDOUT, run_result.stdout, run_result.stderr
    );
}

// ---------------------------------------------------------------------------
// Checkpoint 1 Tests — Architecture-Specific Hello World
// ---------------------------------------------------------------------------
// Backend validation order per Section 0.1.2: x86-64 → i686 → AArch64 → RISC-V 64

/// **Checkpoint 1.1 — x86-64 Hello World**
///
/// Validates the fundamental compilation pipeline for the x86-64 architecture:
/// - Compiles `hello.c` targeting x86-64
/// - Verifies ELF header: `EM_X86_64`, `ELFCLASS64`, `ET_EXEC`
/// - Executes natively (no QEMU required)
/// - Asserts stdout == "Hello, World!\n" and exit code == 0
///
/// This is the **primary validation target** — x86-64 is tested first.
///
/// Per Section 0.1.2 User Example:
///   `./bcc -o hello hello.c && ./hello` → stdout: "Hello, World!\n", exit code 0
#[test]
fn test_hello_world_x86_64() {
    run_hello_world_test("x86-64", "hello_x86_64", ELF_MACHINE_X86_64, ELF_CLASS_64);
}

/// **Checkpoint 1.2 — i686 Hello World**
///
/// Validates the compilation pipeline for the i686 (32-bit x86) architecture:
/// - Compiles `hello.c` targeting i686
/// - Verifies ELF header: `EM_386`, `ELFCLASS32`, `ET_EXEC`
/// - Executes via `qemu-i386` user-mode emulation
/// - Asserts stdout == "Hello, World!\n" and exit code == 0
#[test]
fn test_hello_world_i686() {
    run_hello_world_test("i686", "hello_i686", ELF_MACHINE_I686, ELF_CLASS_32);
}

/// **Checkpoint 1.3 — AArch64 Hello World**
///
/// Validates the compilation pipeline for the AArch64 architecture:
/// - Compiles `hello.c` targeting AArch64
/// - Verifies ELF header: `EM_AARCH64`, `ELFCLASS64`, `ET_EXEC`
/// - Executes via `qemu-aarch64` user-mode emulation
/// - Asserts stdout == "Hello, World!\n" and exit code == 0
#[test]
fn test_hello_world_aarch64() {
    run_hello_world_test(
        "aarch64",
        "hello_aarch64",
        ELF_MACHINE_AARCH64,
        ELF_CLASS_64,
    );
}

/// **Checkpoint 1.4 — RISC-V 64 Hello World**
///
/// Validates the compilation pipeline for the RISC-V 64-bit architecture:
/// - Compiles `hello.c` targeting RISC-V 64
/// - Verifies ELF header: `EM_RISCV`, `ELFCLASS64`, `ET_EXEC`
/// - Executes via `qemu-riscv64` user-mode emulation
/// - Asserts stdout == "Hello, World!\n" and exit code == 0
///
/// RISC-V 64 is the final architecture in the validation order and the
/// target for the Linux kernel 6.9 boot test (Checkpoint 6).
#[test]
fn test_hello_world_riscv64() {
    run_hello_world_test("riscv64", "hello_riscv64", ELF_MACHINE_RISCV, ELF_CLASS_64);
}

// ---------------------------------------------------------------------------
// Checkpoint 1 Tests — Compilation Mode Flags
// ---------------------------------------------------------------------------

/// **Checkpoint 1.5 — Compile-Only (`-c` flag)**
///
/// Validates the `-c` flag produces a relocatable object file (`.o`)
/// without linking:
/// - Invokes `bcc -c tests/fixtures/hello.c -o hello.o`
/// - Asserts compilation succeeds (exit code 0)
/// - Verifies the output file exists on disk
/// - Validates ELF type is `ET_REL` (relocatable), confirming the linker
///   was correctly bypassed
///
/// Uses the default (native) target architecture since this test validates
/// the `-c` flag behavior rather than cross-architecture output.
#[test]
fn test_compile_only_c_flag() {
    // Setup: create a temporary directory for the object file.
    let test_dir = common::TestDir::new("checkpoint1_compile_only");
    let output_path = test_dir.file_path("hello.o");
    let output_str = output_path.to_str().expect("output path must be valid UTF-8");

    // Resolve the hello.c fixture path.
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path must be valid UTF-8");

    // Verify fixture exists.
    assert!(
        Path::new(source_str).exists(),
        "Test fixture 'hello.c' not found at '{}'.",
        source_str
    );

    // Compile with -c flag to produce a relocatable object file.
    // Use x86-64 as the default target for this test.
    let compile_result = common::compile_to_object(source_str, output_str, "x86-64", &[]);

    // Assert compilation succeeded.
    compile_result.assert_success();

    // Verify the object file was produced.
    assert!(
        Path::new(output_str).exists(),
        "Object file '{}' was not produced despite successful compilation.\n\
         Compiler stdout:\n{}\nCompiler stderr:\n{}",
        output_str,
        compile_result.stdout,
        compile_result.stderr
    );

    // Validate ELF type is REL (relocatable), not EXEC or DYN.
    // This confirms the linker was properly bypassed with the -c flag.
    common::assert_elf_type(output_str, ELF_TYPE_REL);

    // Also verify the ELF machine matches x86-64 (the default target).
    common::assert_elf_machine(output_str, ELF_MACHINE_X86_64);

    // Verify ELF class is 64-bit.
    common::assert_elf_class(output_str, ELF_CLASS_64);
}

/// **Checkpoint 1.6 — Preprocess-Only (`-E` flag)**
///
/// Validates the `-E` flag produces preprocessed C source output on stdout
/// without compilation or linking:
/// - Invokes `bcc -E tests/fixtures/hello.c`
/// - Asserts the preprocessor succeeded (exit code 0)
/// - Verifies stdout is non-empty (preprocessing produced output)
/// - Verifies stdout contains evidence of macro expansion / `#include`
///   processing (e.g., the string `"Hello, World!"` or function declarations
///   from `<stdio.h>`)
/// - Verifies stdout does NOT contain ELF binary signatures (confirming
///   no object code was produced)
///
/// This test validates the preprocessor pipeline in isolation, confirming
/// the `-E` flag correctly halts the pipeline after Phase 2 (preprocessing).
#[test]
fn test_preprocess_only_e_flag() {
    // Resolve the hello.c fixture path.
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path must be valid UTF-8");

    // Verify fixture exists.
    assert!(
        Path::new(source_str).exists(),
        "Test fixture 'hello.c' not found at '{}'.",
        source_str
    );

    // Run the preprocessor only with -E flag.
    let preprocess_result = common::preprocess_only(source_str, &[]);

    // Assert preprocessing succeeded.
    preprocess_result.assert_success();

    // Verify stdout is non-empty — preprocessing must produce output.
    assert!(
        !preprocess_result.stdout.is_empty(),
        "Preprocessor (-E) produced empty stdout for 'hello.c'.\nstderr:\n{}",
        preprocess_result.stderr
    );

    // Verify the preprocessed output contains the original string literal
    // from main(). The #include <stdio.h> expansion should be present, and
    // the original source code (including the printf call) should appear in
    // the preprocessed output.
    assert!(
        preprocess_result.stdout.contains("Hello, World!"),
        "Preprocessed output does not contain 'Hello, World!' string literal.\n\
         This indicates the preprocessor is not emitting the original source tokens.\n\
         First 500 chars of stdout:\n{}",
        &preprocess_result.stdout[..preprocess_result.stdout.len().min(500)]
    );

    // Verify the preprocessed output contains evidence of stdio.h expansion.
    // After #include <stdio.h> processing, declarations like 'printf' or
    // 'int' (from function prototypes) should be present.
    assert!(
        preprocess_result.stdout.contains("printf")
            || preprocess_result.stdout.contains("int main"),
        "Preprocessed output does not contain expected tokens from stdio.h \
         expansion or from the main function.\n\
         First 500 chars of stdout:\n{}",
        &preprocess_result.stdout[..preprocess_result.stdout.len().min(500)]
    );

    // Verify the output does NOT contain ELF magic bytes, confirming no
    // binary object code was emitted. The ELF magic is "\x7fELF" — if this
    // appears in the preprocessed output, something went wrong.
    assert!(
        !preprocess_result.stdout.contains("\x7fELF"),
        "Preprocessed output (-E) contains ELF magic bytes, indicating \
         object code was emitted instead of preprocessed source.\n\
         This suggests the -E flag did not properly halt the pipeline."
    );

    // The preprocessed output should be valid text (no binary garbage).
    // Check that the output is predominantly ASCII/UTF-8 text by verifying
    // it does not contain null bytes (common in binary output).
    assert!(
        !preprocess_result.stdout.contains('\0'),
        "Preprocessed output (-E) contains null bytes, suggesting binary \
         output was produced instead of preprocessed text."
    );
}
