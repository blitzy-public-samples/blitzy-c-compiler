//! Checkpoint 1: Hello World Validation
//!
//! This is the foundational validation gate for BCC (Blitzy's C Compiler).
//! If Hello World doesn't compile and run correctly on all four target
//! architectures, no subsequent checkpoint can proceed.
//!
//! **Backend validation order is FIXED:** x86-64 → i686 → AArch64 → RISC-V 64
//!
//! **Checkpoints 1–6 are strictly sequential hard gates.** Failure at any
//! gate halts forward progress.
//!
//! Tests validate the complete compilation pipeline:
//! preprocessing → lexing → parsing → sema → IR lowering → mem2reg →
//! optimization → phi elimination → codegen → assembly → linking → ELF output
//!
//! # Test Organization
//! - Phase 3: x86-64 Hello World (primary validation target)
//! - Phase 4: i686 Hello World
//! - Phase 5: AArch64 Hello World (QEMU user-mode)
//! - Phase 6: RISC-V 64 Hello World (QEMU user-mode, kernel target arch)
//! - Phase 7: ELF structure verification for all four architectures
//! - Phase 8: Compile-only mode (`-c` flag) producing relocatable objects
//! - Phase 9: User example validation (exact AAP invocation pattern)
//!
//! # Zero-Dependency Mandate
//! This file uses only `std` library types and the shared `common` test
//! harness module. No external crates are imported.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Test Helper Functions
// ---------------------------------------------------------------------------

/// Compiles the Hello World test fixture (`tests/fixtures/hello.c`) for the
/// specified target architecture and returns the compilation output together
/// with the path to the produced binary.
///
/// The caller is responsible for cleaning up the output file after use.
///
/// # Arguments
/// * `target` — Target architecture string: `"x86-64"`, `"i686"`,
///   `"aarch64"`, or `"riscv64"`.
///
/// # Returns
/// A tuple of `(compilation Output, PathBuf to the output binary)`.
fn compile_hello_world(target: &str) -> (std::process::Output, PathBuf) {
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path(&format!("hello_{}", target));
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let target_flag = format!("--target={}", target);
    let compile_output = common::compile(source_str, &[&target_flag, "-o", output_str]);

    (compile_output, output_path)
}

/// Compiles `tests/fixtures/hello.c` for the specified target, asserts that
/// compilation succeeds, and returns the path to the output binary.
///
/// # Panics
/// Panics with diagnostic output if compilation fails or the binary does not
/// exist on disk after a successful exit code.
fn compile_hello_for_target(target: &str) -> PathBuf {
    let (output, path) = compile_hello_world(target);
    common::assert_compilation_succeeds(&output);
    assert!(
        path.exists(),
        "Compiled binary does not exist at '{}' after successful compilation",
        path.display()
    );
    path
}

// ===========================================================================
// Phase 3: x86-64 Hello World Test (FIRST — primary validation target)
// ===========================================================================

/// Validates the complete compilation pipeline for x86-64 (native execution):
/// preprocessing → lexing → parsing → sema → IR → codegen → assembly →
/// linking → ELF output.
///
/// This is the **primary validation target** and MUST pass before any other
/// architecture test is considered meaningful.
///
/// Corresponds to AAP §0.5.1 Group 7, Checkpoint 1 and AAP §0.7.5 sequential
/// gate ordering with x86-64 first.
#[test]
fn test_hello_world_x86_64() {
    let output_path = compile_hello_for_target("x86-64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Run the compiled binary natively on the host
    let run_output = common::run_binary(output_str, "x86-64");

    // Validate correct execution
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");

    // Clean up temporary binary
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 4: i686 Hello World Test (SECOND)
// ===========================================================================

/// Validates Hello World compilation and execution for the i686 (32-bit x86)
/// target architecture.
///
/// On an x86-64 host, the 32-bit binary may run natively with multilib
/// support or fall back to `qemu-i386` user-mode emulation.
#[test]
fn test_hello_world_i686() {
    let output_path = compile_hello_for_target("i686");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // run_binary handles i686 execution dispatch (native or qemu-i386)
    let run_output = common::run_binary(output_str, "i686");

    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");

    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 5: AArch64 Hello World Test (THIRD)
// ===========================================================================

/// Validates Hello World compilation and execution for AArch64 (ARM 64-bit).
///
/// Execution uses `qemu-aarch64` user-mode emulation (QEMU v8.2.2 on Ubuntu
/// 24.04). The test gracefully skips if the QEMU emulator is not available.
#[test]
fn test_hello_world_aarch64() {
    if !common::qemu_available("aarch64") {
        eprintln!(
            "SKIP: qemu-aarch64 not available — \
             install qemu-user to run AArch64 tests"
        );
        return;
    }

    let output_path = compile_hello_for_target("aarch64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let run_output = common::run_binary(output_str, "aarch64");

    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");

    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 6: RISC-V 64 Hello World Test (FOURTH — Linux kernel target arch)
// ===========================================================================

/// Validates Hello World compilation and execution for RISC-V 64.
///
/// This is the **Linux kernel target architecture** — the ultimate validation
/// target for Checkpoint 6. Execution uses `qemu-riscv64` user-mode
/// emulation. The test gracefully skips if the QEMU emulator is not available.
#[test]
fn test_hello_world_riscv64() {
    if !common::qemu_available("riscv64") {
        eprintln!(
            "SKIP: qemu-riscv64 not available — \
             install qemu-user to run RISC-V 64 tests"
        );
        return;
    }

    let output_path = compile_hello_for_target("riscv64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let run_output = common::run_binary(output_str, "riscv64");

    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");

    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 7: ELF Structure Verification Tests
// ===========================================================================

/// Verifies the ELF structure of an x86-64 Hello World binary:
/// - ELF class: ELFCLASS64
/// - Data encoding: 2's complement, little endian
/// - Machine type: Advanced Micro Devices X86-64
/// - ELF type: EXEC (Executable file) or ET_EXEC
/// - Entry point address is present
/// - `.text` section exists
#[test]
fn test_elf_structure_x86_64() {
    let output_path = compile_hello_for_target("x86-64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // ---- ELF Header Verification via readelf -h ----
    let header = common::readelf_header(output_str);

    // ELF class must be 64-bit
    assert!(
        header.contains("ELF64") || header.contains("ELFCLASS64"),
        "Expected ELFCLASS64 for x86-64 target.\nreadelf -h output:\n{}",
        header
    );

    // Machine type must be x86-64
    assert!(
        header.contains("Advanced Micro Devices X86-64") || header.contains("X86-64"),
        "Expected x86-64 (EM_X86_64) machine type.\nreadelf -h output:\n{}",
        header
    );

    // ELF type must be executable
    assert!(
        header.contains("EXEC") || header.contains("ET_EXEC"),
        "Expected executable ELF type (ET_EXEC).\nreadelf -h output:\n{}",
        header
    );

    // Data encoding: little endian
    assert!(
        header.contains("2's complement, little endian") || header.contains("little endian"),
        "Expected little-endian data encoding.\nreadelf -h output:\n{}",
        header
    );

    // Entry point address must be present and non-zero
    assert!(
        header.contains("Entry point address:"),
        "Expected entry point address in ELF header.\nreadelf -h output:\n{}",
        header
    );
    // Verify entry point is not 0x0
    let entry_line_has_nonzero = header.lines().any(|line| {
        line.contains("Entry point address:")
            && !line.contains("0x0\n")
            && !line.trim().ends_with("0x0")
    });
    assert!(
        entry_line_has_nonzero,
        "Entry point address appears to be zero.\nreadelf -h output:\n{}",
        header
    );

    // ---- Section Verification via readelf -S ----
    let sections = common::readelf_sections(output_str);
    assert!(
        sections.contains(".text"),
        "Expected .text section in x86-64 ELF.\nreadelf -S output:\n{}",
        sections
    );

    let _ = fs::remove_file(&output_path);
}

/// Verifies the ELF structure of an i686 Hello World binary:
/// - ELF class: ELFCLASS32
/// - Machine type: Intel 80386 (EM_386)
/// - ELF type: EXEC (ET_EXEC)
/// - `.text` section exists
#[test]
fn test_elf_structure_i686() {
    let output_path = compile_hello_for_target("i686");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let header = common::readelf_header(output_str);

    assert!(
        header.contains("ELF32") || header.contains("ELFCLASS32"),
        "Expected ELFCLASS32 for i686 target.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("Intel 80386") || header.contains("EM_386") || header.contains("80386"),
        "Expected Intel 80386 (EM_386) machine type.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("EXEC") || header.contains("ET_EXEC"),
        "Expected executable ELF type (ET_EXEC).\nreadelf -h output:\n{}",
        header
    );

    let sections = common::readelf_sections(output_str);
    assert!(
        sections.contains(".text"),
        "Expected .text section in i686 ELF.\nreadelf -S output:\n{}",
        sections
    );

    let _ = fs::remove_file(&output_path);
}

/// Verifies the ELF structure of an AArch64 Hello World binary:
/// - ELF class: ELFCLASS64
/// - Machine type: AArch64 (EM_AARCH64)
/// - ELF type: EXEC (ET_EXEC)
/// - `.text` section exists
#[test]
fn test_elf_structure_aarch64() {
    let output_path = compile_hello_for_target("aarch64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let header = common::readelf_header(output_str);

    assert!(
        header.contains("ELF64") || header.contains("ELFCLASS64"),
        "Expected ELFCLASS64 for AArch64 target.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("AArch64") || header.contains("EM_AARCH64"),
        "Expected AArch64 (EM_AARCH64) machine type.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("EXEC") || header.contains("ET_EXEC"),
        "Expected executable ELF type (ET_EXEC).\nreadelf -h output:\n{}",
        header
    );

    let sections = common::readelf_sections(output_str);
    assert!(
        sections.contains(".text"),
        "Expected .text section in AArch64 ELF.\nreadelf -S output:\n{}",
        sections
    );

    let _ = fs::remove_file(&output_path);
}

/// Verifies the ELF structure of a RISC-V 64 Hello World binary:
/// - ELF class: ELFCLASS64
/// - Machine type: RISC-V (EM_RISCV)
/// - ELF type: EXEC (ET_EXEC)
/// - `.text` section exists
#[test]
fn test_elf_structure_riscv64() {
    let output_path = compile_hello_for_target("riscv64");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let header = common::readelf_header(output_str);

    assert!(
        header.contains("ELF64") || header.contains("ELFCLASS64"),
        "Expected ELFCLASS64 for RISC-V 64 target.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("RISC-V") || header.contains("EM_RISCV"),
        "Expected RISC-V (EM_RISCV) machine type.\nreadelf -h output:\n{}",
        header
    );
    assert!(
        header.contains("EXEC") || header.contains("ET_EXEC"),
        "Expected executable ELF type (ET_EXEC).\nreadelf -h output:\n{}",
        header
    );

    let sections = common::readelf_sections(output_str);
    assert!(
        sections.contains(".text"),
        "Expected .text section in RISC-V 64 ELF.\nreadelf -S output:\n{}",
        sections
    );

    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 8: Compile-Only Mode Test
// ===========================================================================

/// Validates that the `-c` (compile-only) flag produces a relocatable object
/// file (ELF type ET_REL) rather than a linked executable.
///
/// This exercises the assembler's object file emission path without invoking
/// the built-in linker.
#[test]
fn test_compile_only_produces_object() {
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("hello_compile_only");
    let obj_path = PathBuf::from(format!(
        "{}.o",
        output_path.to_str().expect("temp path is valid UTF-8")
    ));
    let obj_str = obj_path.to_str().expect("obj path is valid UTF-8");

    // Invoke BCC with -c to produce a relocatable object file
    let compile_output = common::compile(source_str, &["-c", "-o", obj_str]);
    common::assert_compilation_succeeds(&compile_output);

    // Verify the .o file was produced on disk
    assert!(
        Path::new(obj_str).exists(),
        "Object file was not produced at '{}'",
        obj_str
    );

    // Verify it has non-zero size
    let metadata = fs::metadata(obj_str).expect("Could not read object file metadata");
    assert!(
        metadata.len() > 0,
        "Object file at '{}' is empty (0 bytes)",
        obj_str
    );

    // Verify it's a relocatable ELF (ET_REL) via readelf
    let header = common::readelf_header(obj_str);
    assert!(
        header.contains("REL (Relocatable file)") || header.contains("REL"),
        "Expected relocatable ELF type (ET_REL) for object file.\nreadelf -h output:\n{}",
        header
    );

    // Clean up both files
    common::cleanup_temp_files(&[obj_str]);
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Phase 9: User Example Validation
// ===========================================================================

/// Replicates the **exact user example** from the Agent Action Plan:
///
/// ```text
/// ./bcc -o hello hello.c && ./hello
/// ```
///
/// Expected result: stdout `Hello, World!\n`, exit code 0.
///
/// This test uses `Command` directly (not the `common::compile` helper) to
/// match the invocation pattern as closely as possible to the AAP specification.
#[test]
fn test_user_example_hello_world() {
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("hello_user_example");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Step 1: Compile — mimics `./bcc -o hello hello.c`
    // Uses both Command::arg() and Command::args() to match AAP external import members.
    let bcc = common::bcc_path();
    let compile_output = Command::new(&bcc)
        .arg("-o")
        .arg(output_str)
        .arg(source_str)
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute BCC at '{}': {}", bcc.display(), e));

    common::assert_compilation_succeeds(&compile_output);
    assert!(
        output_path.exists(),
        "Compiled binary does not exist at '{}' after user-example compilation",
        output_path.display()
    );

    // Step 2: Run — mimics `./hello`
    let run_status = Command::new(output_str)
        .status()
        .unwrap_or_else(|e| panic!("Failed to run compiled binary '{}': {}", output_str, e));
    assert!(
        run_status.success(),
        "User example binary exited with non-zero status: {}",
        run_status
    );

    // Step 3: Run again capturing stdout for content verification
    let run_output = Command::new(output_str)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run compiled binary '{}': {}", output_str, e));
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Additional Integration: compile_and_run shortcut validation
// ===========================================================================

/// Validates the `common::compile_and_run` convenience function by compiling
/// and running Hello World in a single call for the native x86-64 target.
///
/// This exercises the high-level helper that other checkpoint tests rely on,
/// ensuring it correctly orchestrates compilation, execution, and cleanup.
#[test]
fn test_compile_and_run_hello_x86_64() {
    let source = common::fixture_path("hello.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let run_output = common::compile_and_run(source_str, "x86-64");

    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "Hello, World!\n");
}
