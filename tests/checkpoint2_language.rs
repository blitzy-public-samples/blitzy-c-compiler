//! Checkpoint 2 Integration Test Suite — Language and Preprocessor Correctness
//!
//! This module validates that the BCC compiler correctly handles C11 language
//! features, GCC extensions, and preprocessor semantics. It is a sequential
//! hard gate per Section 0.7.5: all tests must pass before subsequent
//! checkpoints can proceed.
//!
//! Test coverage:
//!
//! - **PUA encoding round-trip:** Non-UTF-8 bytes (0x80–0xFF) survive the
//!   entire compilation pipeline with byte-exact fidelity (Section 0.7.9).
//! - **Recursive macro termination:** Self-referential macros (`#define A A`)
//!   terminate within 5 seconds via paint-marker recursion protection
//!   (Section 0.1.2).
//! - **GCC statement expressions:** `({ ... })` syntax evaluates correctly.
//! - **typeof / __typeof__:** Type inference from expressions works correctly.
//! - **Designated initializers:** Out-of-order, nested, and array index
//!   designation with brace elision and implicit zero-initialization.
//! - **Inline assembly:** AT&T syntax, constraints, clobber lists, named
//!   operands (x86-64 specific).
//! - **Computed gotos:** `goto *ptr` dispatch with label addresses (`&&label`).
//! - **Zero-length arrays:** GCC extension for trailing flexible members.
//! - **GCC builtins:** `__builtin_constant_p`, `__builtin_offsetof`,
//!   `__builtin_clz`, `__builtin_bswap*`, etc.
//! - **_Static_assert:** C11 compile-time assertion (valid and invalid cases).
//! - **_Generic:** C11 type-based selection expression.
//! - **Multi-architecture:** Key tests run across x86-64, i686, AArch64,
//!   and RISC-V 64.

mod common;

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use common::{
    compile, compile_to_binary, compile_to_object, fixture_path, objdump_section_contents,
    run_binary_for_target, timed_execution, BccOutput, TestDir,
};

// ---------------------------------------------------------------------------
// Target architecture constants used throughout the test suite.
// ---------------------------------------------------------------------------
const TARGET_X86_64: &str = "x86-64";
const TARGET_I686: &str = "i686";
const TARGET_AARCH64: &str = "aarch64";
const TARGET_RISCV64: &str = "riscv64";

/// All four supported target architectures for multi-arch test iterations.
const ALL_TARGETS: &[&str] = &[TARGET_X86_64, TARGET_I686, TARGET_AARCH64, TARGET_RISCV64];

// ---------------------------------------------------------------------------
// Helper: compile a fixture to a binary, assert compilation success, run it,
// and assert execution success + exit code 0.
// ---------------------------------------------------------------------------

/// Compile a C test fixture to a native binary for the given target, assert
/// successful compilation, execute the binary (via QEMU if cross-compiling),
/// and assert the program exits with code 0.
///
/// Returns the captured execution output for further assertions.
fn compile_and_run_fixture(fixture_name: &str, test_name: &str, target: &str) -> BccOutput {
    let source = fixture_path(fixture_name);
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let tmp = TestDir::new(&format!("{}_{}", test_name, target));
    let binary = tmp.file_path(&format!("{}_{}", test_name, target));
    let binary_str = binary.to_str().expect("binary path is valid UTF-8");

    // Compile
    let compile_result = compile_to_binary(source_str, binary_str, target, &[]);
    compile_result.assert_success();

    // Verify the binary was actually produced on disk.
    assert!(
        Path::new(binary_str).exists(),
        "Compiled binary '{}' does not exist after successful compilation",
        binary_str
    );

    // Execute
    let run_result = run_binary_for_target(binary_str, target);
    assert!(
        run_result.success(),
        "Test binary '{}' for fixture '{}' (target {}) exited with code {:?}.\nstdout:\n{}\nstderr:\n{}",
        binary_str,
        fixture_name,
        target,
        run_result.exit_code(),
        run_result.stdout,
        run_result.stderr
    );

    run_result
}

/// Compile a C fixture to a relocatable object file for the given target,
/// assert success, and return the path plus temporary directory.
fn compile_fixture_to_object(
    fixture_name: &str,
    test_name: &str,
    target: &str,
) -> (String, TestDir) {
    let source = fixture_path(fixture_name);
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let tmp = TestDir::new(&format!("{}_obj_{}", test_name, target));
    let object = tmp.file_path(&format!("{}.o", test_name));
    let object_str = object.to_str().expect("object path is valid UTF-8");

    let result = compile_to_object(source_str, object_str, target, &[]);
    result.assert_success();

    assert!(
        Path::new(object_str).exists(),
        "Object file '{}' does not exist after successful compilation",
        object_str
    );

    (object_str.to_string(), tmp)
}

// ===========================================================================
// Test 1: PUA Encoding Round-Trip
//
// Per Section 0.7.9: Non-UTF-8 bytes (0x80–0xFF) in C source files must
// survive the entire pipeline with byte-exact fidelity. The PUA code-point
// mapping (U+E080–U+E0FF) encodes on source read and decodes on output.
//
// Validation protocol:
//   1. Compile pua_roundtrip.c to a binary
//   2. Inspect .rodata via `objdump -s` and confirm bytes 80 ff are present
//   3. Execute the binary and assert exit code 0 (all runtime checks pass)
// ===========================================================================

#[test]
fn test_pua_roundtrip() {
    let source = fixture_path("pua_roundtrip.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let tmp = TestDir::new("pua_roundtrip");
    let binary = tmp.file_path("pua_roundtrip");
    let binary_str = binary.to_str().expect("binary path is valid UTF-8");

    // Compile to native x86-64 binary (primary validation target).
    let compile_result = compile_to_binary(source_str, binary_str, TARGET_X86_64, &[]);
    compile_result.assert_success();

    // --- .rodata inspection ---
    // Per Section 0.7.9: inspect .rodata with objdump -s, confirm exact
    // bytes 80 ff are present.
    let rodata_hex = objdump_section_contents(binary_str, ".rodata");

    // The hex dump from `objdump -s -j .rodata` produces lines like:
    //   <addr>  80ff0000 ...
    // We search for the canonical byte pair "80" and "ff" appearing in the
    // hex output. The pua_core[] string literal is "\x80\xFF" which must
    // appear as contiguous bytes 80 ff in the .rodata section.
    let hex_lower = rodata_hex.to_lowercase();
    assert!(
        hex_lower.contains("80") && hex_lower.contains("ff"),
        "Expected .rodata section to contain bytes 0x80 and 0xFF for PUA \
         round-trip validation.\nobjdump output:\n{}",
        rodata_hex
    );

    // Stronger check: look for the "80ff" pair somewhere in the hex dump,
    // which corresponds to the pua_core[] = "\x80\xFF" string literal.
    // objdump -s prints hex words (groups of 4 bytes), so "80ff" could
    // appear within a word or split across words. We normalise whitespace
    // and search for the contiguous pair.
    let hex_no_space: String = hex_lower
        .lines()
        .flat_map(|line| {
            // Each objdump line looks like:
            //  <addr>  XXXXXXXX XXXXXXXX XXXXXXXX XXXXXXXX  <ascii>
            // Extract the hex portion between the address and the ASCII.
            if let Some(hex_start) = line.find("  ") {
                let after_addr = &line[hex_start..];
                // Take characters up to the next double-space or ASCII column
                let hex_part = after_addr.split("  ").next().unwrap_or("").replace(' ', "");
                Some(hex_part)
            } else {
                None
            }
        })
        .collect::<String>();

    assert!(
        hex_no_space.contains("80ff"),
        "Expected contiguous bytes 0x80 0xFF (PUA core test) in .rodata hex dump.\n\
         Normalised hex: {}\nRaw objdump output:\n{}",
        &hex_no_space[..hex_no_space.len().min(500)],
        rodata_hex
    );

    // --- Binary format verification ---
    // Use Command directly to verify the compiled binary is a valid ELF.
    let file_check = Command::new("file")
        .arg(binary_str)
        .output()
        .expect("failed to execute 'file' command");
    let file_output = String::from_utf8_lossy(&file_check.stdout);
    assert!(
        file_output.contains("ELF"),
        "Compiled PUA test binary is not a valid ELF file.\nfile output: {}",
        file_output
    );

    // --- Runtime verification ---
    // Execute the binary and confirm all runtime PUA checks pass.
    let run_result = run_binary_for_target(binary_str, TARGET_X86_64);
    assert!(
        run_result.success(),
        "PUA round-trip binary exited with code {:?}.\nstdout:\n{}\nstderr:\n{}",
        run_result.exit_code(),
        run_result.stdout,
        run_result.stderr
    );

    // The fixture prints "PUA round-trip: ALL TESTS PASSED" on success.
    assert!(
        run_result.stdout.contains("ALL TESTS PASSED"),
        "Expected PUA round-trip output to contain 'ALL TESTS PASSED'.\nstdout:\n{}",
        run_result.stdout
    );
}

// ===========================================================================
// Test 2: Recursive Macro Termination
//
// Per Section 0.1.2:
//   "#define A A" and "int x = A;" must terminate in <5 seconds, no hang.
//
// The preprocessor's paint-marker system prevents infinite expansion of
// self-referential and mutually recursive macros. This test validates that
// the compilation completes within the timeout and produces a correct binary.
// ===========================================================================

#[test]
fn test_recursive_macro() {
    let source = fixture_path("recursive_macro.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let tmp = TestDir::new("recursive_macro");
    let binary = tmp.file_path("recursive_macro");
    let binary_str = binary.to_str().expect("binary path is valid UTF-8");

    // Compile with wall-clock timing. Must complete within 5 seconds.
    let (compile_result, elapsed) =
        timed_execution(|| compile_to_binary(source_str, binary_str, TARGET_X86_64, &[]));

    // Assert compilation terminated within the 5-second ceiling.
    let timeout = Duration::from_secs(5);
    assert!(
        elapsed < timeout,
        "Recursive macro compilation took {:?}, exceeding the 5-second timeout.\n\
         This indicates the paint-marker recursion protection failed.\nstderr:\n{}",
        elapsed,
        compile_result.stderr
    );

    // Assert compilation succeeded.
    compile_result.assert_success();

    // Execute the compiled binary and verify correctness (exit code 0 means
    // all recursive macro expansion results were correct).
    let run_result = run_binary_for_target(binary_str, TARGET_X86_64);
    assert!(
        run_result.success(),
        "Recursive macro test binary exited with code {:?} (expected 0).\n\
         Non-zero exit code indicates incorrect macro expansion results.\n\
         stdout:\n{}\nstderr:\n{}",
        run_result.exit_code(),
        run_result.stdout,
        run_result.stderr
    );
}

// ===========================================================================
// Test 3: GCC Statement Expressions
//
// Tests the GCC `({ ... })` extension for statement expressions including:
// - Nested expressions
// - Macro-wrapped min/max patterns
// - Statement expressions as function arguments
// ===========================================================================

#[test]
fn test_stmt_expr() {
    let output = compile_and_run_fixture("stmt_expr.c", "stmt_expr", TARGET_X86_64);

    // The fixture prints "All statement expression tests passed." on success.
    assert!(
        output.stdout.contains("passed"),
        "Expected statement expression test output to indicate success.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 4: typeof / __typeof__ GCC Extension
//
// Tests type inference from integer, pointer, struct, and array expressions
// using both `typeof` and `__typeof__` keyword forms.
// ===========================================================================

#[test]
fn test_typeof() {
    let output = compile_and_run_fixture("typeof_test.c", "typeof", TARGET_X86_64);

    // The fixture prints "typeof_test: PASSED" on success.
    assert!(
        output.stdout.contains("PASSED"),
        "Expected typeof test output to contain 'PASSED'.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 5: Designated Initializers
//
// Exercises C11 and GCC extension designated initializers:
// - Out-of-order field designation (.field = value)
// - Nested struct designation (.field.subfield = value)
// - Array index designation ([idx] = value)
// - Range designators ([low ... high] = value)
// - Brace elision
// - Implicit zero-initialization of unspecified members
// ===========================================================================

#[test]
fn test_designated_init() {
    let output = compile_and_run_fixture("designated_init.c", "designated_init", TARGET_X86_64);

    // The fixture returns the failure count as its exit code.
    // A successful run means exit code 0 (no failures), already asserted by
    // compile_and_run_fixture.
    //
    // Additionally verify no FAIL markers appear in stdout.
    assert!(
        !output.stdout.contains("FAIL"),
        "Designated initializer test output contains FAIL markers.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 6: Basic Inline Assembly
//
// Tests GCC asm/asm volatile statements using AT&T syntax on x86-64:
// - Basic asm with no operands (nop)
// - asm volatile with output operand ("=r")
// - asm with both input and output operands
// - Memory and cc clobbers
// - __asm__ / __volatile__ alternate keywords
//
// NOTE: Inline assembly fixtures use x86-64 AT&T syntax, so this test
// targets x86-64 exclusively.
// ===========================================================================

#[test]
fn test_inline_asm_basic() {
    let output = compile_and_run_fixture("inline_asm_basic.c", "inline_asm_basic", TARGET_X86_64);

    // The fixture prints "ALL TESTS PASSED" on success.
    assert!(
        output.stdout.contains("ALL TESTS PASSED"),
        "Expected inline asm basic test output to contain 'ALL TESTS PASSED'.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 7: Advanced Inline Assembly Constraints
//
// Tests advanced inline assembly features on x86-64:
// - Named operands ([name] "constraint" (expr))
// - Multiple constraint types ("=r", "=m", "+r", "r", "i", "n")
// - Complex clobber lists ("memory", "cc")
// - asm goto with jump labels
// - .pushsection / .popsection directives
// - Early clobber ("=&r")
//
// NOTE: x86-64 target only (AT&T syntax).
// ===========================================================================

#[test]
fn test_inline_asm_constraints() {
    let output = compile_and_run_fixture(
        "inline_asm_constraints.c",
        "inline_asm_constraints",
        TARGET_X86_64,
    );

    // The fixture prints "All advanced inline assembly constraint tests passed."
    // on success.
    assert!(
        output.stdout.contains("passed"),
        "Expected inline asm constraints test output to indicate success.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 8: Computed Gotos
//
// Tests the GCC `goto *ptr` extension with label addresses (`&&label`)
// for computed dispatch tables — the common interpreter-loop pattern used
// extensively in the Linux kernel and language runtimes.
// ===========================================================================

#[test]
fn test_computed_goto() {
    let output = compile_and_run_fixture("computed_goto.c", "computed_goto", TARGET_X86_64);

    // The fixture prints "PASS: computed_goto" on success.
    assert!(
        output.stdout.contains("PASS"),
        "Expected computed goto test output to contain 'PASS'.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 9: Zero-Length Arrays
//
// Tests the GCC zero-length array extension (`int data[0]` as a trailing
// struct member). This is a common pattern in the Linux kernel for
// variable-length structures. Validates that:
// - The compiler accepts the zero-length array declaration
// - sizeof(struct) computes correctly (excluding the flex member)
// - Dynamic allocation and access patterns work at runtime
// ===========================================================================

#[test]
fn test_zero_length_array() {
    // First, verify compilation to object succeeds (compilation-acceptance
    // check for the zero-length array GCC extension).
    let (object_path, _obj_tmp) =
        compile_fixture_to_object("zero_length_array.c", "zero_length_array", TARGET_X86_64);
    assert!(
        Path::new(&object_path).exists(),
        "Object file for zero-length array fixture was not produced"
    );

    // Then compile to a full binary and run to validate runtime behaviour.
    let output = compile_and_run_fixture("zero_length_array.c", "zero_length_array", TARGET_X86_64);

    // The fixture returns 0 on success (no failures).
    // Additional check: ensure no FAIL markers in output.
    assert!(
        !output.stdout.contains("FAIL"),
        "Zero-length array test output contains FAIL markers.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 10: GCC Builtins
//
// Exercises both compile-time and runtime GCC builtins:
//
// Compile-time:
//   __builtin_constant_p, __builtin_types_compatible_p,
//   __builtin_choose_expr, __builtin_offsetof
//
// Runtime:
//   __builtin_clz, __builtin_ctz, __builtin_popcount,
//   __builtin_bswap32, __builtin_bswap64, __builtin_ffs,
//   __builtin_expect, __builtin_frame_address
// ===========================================================================

#[test]
fn test_builtins() {
    let output = compile_and_run_fixture("builtins.c", "builtins", TARGET_X86_64);

    // The fixture prints "All builtin tests PASSED." on success.
    assert!(
        output.stdout.contains("PASSED"),
        "Expected builtins test output to contain 'PASSED'.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Test 11: _Static_assert
//
// Tests C11 `_Static_assert` at file scope, block scope, and struct scope.
// Two scenarios are validated:
//   a) Valid static assertions compile and run successfully.
//   b) An invalid assertion (false condition) produces a compilation error.
// ===========================================================================

#[test]
fn test_static_assert() {
    // Part A: Compile and run the valid _Static_assert fixture.
    let output = compile_and_run_fixture("static_assert.c", "static_assert", TARGET_X86_64);

    // The fixture prints "PASS: all _Static_assert tests compiled and ran successfully"
    assert!(
        output.stdout.contains("PASS"),
        "Expected _Static_assert test output to contain 'PASS'.\nstdout:\n{}",
        output.stdout
    );

    // Part B: Compile an inline snippet with an invalid _Static_assert.
    // This must fail at compile time. We write a temporary C file with a
    // false assertion and verify that BCC rejects it.
    let tmp = TestDir::new("static_assert_invalid");
    let invalid_source = tmp.file_path("invalid_static_assert.c");
    let invalid_source_str = invalid_source.to_str().expect("path is valid UTF-8");

    // Write a minimal C source with a failing _Static_assert.
    std::fs::write(
        &invalid_source,
        b"_Static_assert(0, \"this assertion must fail\");\nint main(void) { return 0; }\n",
    )
    .expect("failed to write invalid _Static_assert fixture");

    // Use compile() directly (compilation-only check, no binary needed).
    let invalid_result = compile(invalid_source_str, &["--target=x86-64", "-c"]);

    // The compiler must reject the invalid assertion.
    invalid_result.assert_failure();

    // Verify the error message references the assertion failure.
    let combined_output = format!("{}{}", invalid_result.stdout, invalid_result.stderr);
    assert!(
        combined_output.contains("static_assert")
            || combined_output.contains("Static_assert")
            || combined_output.contains("assertion")
            || combined_output.contains("fail"),
        "Expected error output to reference static assertion failure.\nstdout:\n{}\nstderr:\n{}",
        invalid_result.stdout,
        invalid_result.stderr
    );
}

// ===========================================================================
// Test 12: _Generic Selection
//
// Tests C11 `_Generic` keyword for type-based compile-time dispatch across
// int, float, double, char*, and default association cases.
// ===========================================================================

#[test]
fn test_generic() {
    let output = compile_and_run_fixture("generic.c", "generic", TARGET_X86_64);

    // The fixture returns 0 on success. Verify no FAIL markers in output.
    assert!(
        !output.stdout.contains("FAIL"),
        "Expected _Generic test output to contain no FAIL markers.\nstdout:\n{}",
        output.stdout
    );
}

// ===========================================================================
// Multi-Architecture Coverage Tests
//
// Per Section 0.7.5 and Section 0.2.1: Key language correctness tests are
// run across all four architectures (x86-64, i686, AArch64, RISC-V 64).
//
// Tests that are inherently architecture-specific (inline assembly) are
// excluded from multi-arch coverage since they use x86-64 AT&T syntax.
// ===========================================================================

/// Run the recursive macro test across all four architectures.
///
/// This validates that the preprocessor's paint-marker system operates
/// correctly regardless of the target architecture (the preprocessor is
/// architecture-independent, but we verify end-to-end correctness).
#[test]
fn test_recursive_macro_multi_arch() {
    for &target in ALL_TARGETS {
        let source = fixture_path("recursive_macro.c");
        let source_str = source.to_str().expect("fixture path is valid UTF-8");

        let tmp = TestDir::new(&format!("recursive_macro_{}", target));
        let binary = tmp.file_path("recursive_macro");
        let binary_str = binary.to_str().expect("binary path is valid UTF-8");

        // Compile with timing check (5-second ceiling).
        let (compile_result, elapsed) =
            timed_execution(|| compile_to_binary(source_str, binary_str, target, &[]));

        let timeout = Duration::from_secs(5);
        assert!(
            elapsed < timeout,
            "Recursive macro compilation for target '{}' took {:?}, exceeding 5s ceiling.",
            target,
            elapsed
        );

        compile_result.assert_success();

        // Execute and verify.
        let run_result = run_binary_for_target(binary_str, target);
        assert!(
            run_result.success(),
            "Recursive macro test binary failed for target '{}' (exit code {:?}).\n\
             stdout:\n{}\nstderr:\n{}",
            target,
            run_result.exit_code(),
            run_result.stdout,
            run_result.stderr
        );
    }
}

/// Run the statement expression test across all four architectures.
#[test]
fn test_stmt_expr_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("stmt_expr.c", "stmt_expr_ma", target);
        assert!(
            output.stdout.contains("passed"),
            "Statement expression test failed for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the typeof test across all four architectures.
#[test]
fn test_typeof_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("typeof_test.c", "typeof_ma", target);
        assert!(
            output.stdout.contains("PASSED"),
            "typeof test failed for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the designated initializer test across all four architectures.
#[test]
fn test_designated_init_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("designated_init.c", "desinit_ma", target);
        assert!(
            !output.stdout.contains("FAIL"),
            "Designated initializer test has failures for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the computed goto test across all four architectures.
#[test]
fn test_computed_goto_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("computed_goto.c", "computed_goto_ma", target);
        assert!(
            output.stdout.contains("PASS"),
            "Computed goto test failed for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the GCC builtins test across all four architectures.
#[test]
fn test_builtins_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("builtins.c", "builtins_ma", target);
        assert!(
            output.stdout.contains("PASSED"),
            "Builtins test failed for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the _Generic selection test across all four architectures.
#[test]
fn test_generic_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("generic.c", "generic_ma", target);
        assert!(
            !output.stdout.contains("FAIL"),
            "_Generic test has failures for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the PUA round-trip test on all four architectures to verify
/// byte-exact fidelity is maintained across all code generation backends.
#[test]
fn test_pua_roundtrip_multi_arch() {
    for &target in ALL_TARGETS {
        let source = fixture_path("pua_roundtrip.c");
        let source_str = source.to_str().expect("fixture path is valid UTF-8");

        let tmp = TestDir::new(&format!("pua_roundtrip_{}", target));
        let binary = tmp.file_path("pua_roundtrip");
        let binary_str = binary.to_str().expect("binary path is valid UTF-8");

        let compile_result = compile_to_binary(source_str, binary_str, target, &[]);
        compile_result.assert_success();

        let run_result = run_binary_for_target(binary_str, target);
        assert!(
            run_result.success(),
            "PUA round-trip binary failed for target '{}' (exit code {:?}).\n\
             stdout:\n{}\nstderr:\n{}",
            target,
            run_result.exit_code(),
            run_result.stdout,
            run_result.stderr
        );

        assert!(
            run_result.stdout.contains("ALL TESTS PASSED"),
            "PUA round-trip output missing 'ALL TESTS PASSED' for target '{}'.\nstdout:\n{}",
            target,
            run_result.stdout
        );
    }
}

/// Run the _Static_assert test across all four architectures.
#[test]
fn test_static_assert_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("static_assert.c", "static_assert_ma", target);
        assert!(
            output.stdout.contains("PASS"),
            "_Static_assert test failed for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}

/// Run the zero-length array test across all four architectures.
#[test]
fn test_zero_length_array_multi_arch() {
    for &target in ALL_TARGETS {
        let output = compile_and_run_fixture("zero_length_array.c", "zla_ma", target);
        assert!(
            !output.stdout.contains("FAIL"),
            "Zero-length array test has failures for target '{}'. stdout:\n{}",
            target,
            output.stdout
        );
    }
}
