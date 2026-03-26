//! Checkpoint 2 — Language and Preprocessor Correctness Validation
//!
//! This test suite validates C11 language features, GCC extensions, and
//! preprocessor correctness within the BCC compiler. It is a **sequential
//! hard gate** — Checkpoint 1 (Hello World) must pass before Checkpoint 2
//! tests are meaningful.
//!
//! # Test Coverage
//!
//! 1. **PUA Encoding Round-Trip (AAP §0.7.9):** Non-UTF-8 bytes (0x80–0xFF)
//!    must survive the entire pipeline with byte-exact fidelity via the Private
//!    Use Area (U+E080–U+E0FF) encoding system.
//!
//! 2. **Recursive Macro Termination (AAP User Example):** `#define A A` and
//!    `int x = A;` MUST terminate in <5 seconds via paint-marker protection.
//!
//! 3. **GCC Statement Expressions:** `({ ... })` syntax used in kernel
//!    min/max macros.
//!
//! 4. **typeof / __typeof__:** Type inference extension for type-safe macros.
//!
//! 5. **Designated Initializers:** Out-of-order, nested, array-index designation
//!    and brace elision.
//!
//! 6. **Inline Assembly:** Basic `asm volatile` and constraint validation
//!    (`=r`, `=m`, `+r`, `r`, `i`, clobbers, named operands).
//!
//! 7. **Computed Gotos:** `&&label` and `goto *ptr` dispatch tables.
//!
//! 8. **Zero-Length Arrays:** GCC zero-length array extension (`type data[0]`).
//!
//! 9. **GCC Builtins:** `__builtin_constant_p`, `__builtin_offsetof`,
//!    `__builtin_clz`, `__builtin_bswap*`, etc.
//!
//! 10. **C11 `_Static_assert`:** Compile-time assertion validation.
//!
//! 11. **C11 `_Generic`:** Type-based dispatch at compile time.
//!
//! 12. **Cross-Architecture:** Critical language features verified across
//!     multiple architectures (x86-64 primary, plus cross-compiled targets).
//!
//! # AAP Compliance
//! - **Sequential hard gate:** Checkpoint 1 must pass first (AAP §0.7.5).
//! - **Backend validation order:** x86-64 → i686 → AArch64 → RISC-V 64.
//! - **Zero external crate dependencies** — only `std::` imports are used.
//! - **PUA fidelity (AAP §0.7.9):** Byte-exact round-trip for 0x80–0xFF.
//! - **Recursive macro timeout (AAP User Example):** <5 seconds, no hang.
//!
//! # Fixture Files
//! - `tests/fixtures/pua_roundtrip.c`          — PUA encoding byte fidelity
//! - `tests/fixtures/recursive_macro.c`        — Self-referential macro test
//! - `tests/fixtures/stmt_expr.c`              — GCC statement expressions
//! - `tests/fixtures/typeof_test.c`            — typeof/__typeof__ extension
//! - `tests/fixtures/designated_init.c`        — Designated initializers
//! - `tests/fixtures/inline_asm_basic.c`       — Basic inline assembly
//! - `tests/fixtures/inline_asm_constraints.c` — Inline asm constraints
//! - `tests/fixtures/computed_goto.c`          — Computed goto dispatch
//! - `tests/fixtures/zero_length_array.c`      — Zero-length array extension
//! - `tests/fixtures/builtins.c`               — GCC builtins coverage
//! - `tests/fixtures/static_assert.c`          — C11 _Static_assert
//! - `tests/fixtures/generic.c`                — C11 _Generic selection
//!
//! # Prerequisites
//! - Release build of BCC: `cargo build --release`
//! - GNU binutils: `objdump` (v2.42) for PUA `.rodata` inspection
//! - QEMU user-mode emulation for cross-architecture tests
//!
//! # Zero Dependencies
//! Only `std::` imports are used — no external crates.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

// ===========================================================================
// BCC Binary Availability Check
// ===========================================================================

/// Verifies the BCC binary is available and executable before running tests.
/// Uses `bcc_path()` directly to locate the compiler binary and `Command`
/// to verify it can be invoked.
fn ensure_bcc_available() {
    let bcc = common::bcc_path();
    let meta = fs::metadata(&bcc);
    assert!(
        meta.is_ok(),
        "BCC binary not found at '{}'. Run `cargo build --release` first.",
        bcc.display()
    );

    // Verify the binary is actually executable by invoking it with --help
    // or a harmless flag. We use Command directly here.
    let status = Command::new(&bcc).arg("--help").status();
    // We only check the binary is invocable, not the exit code —
    // --help might not be implemented yet, but the binary should start.
    assert!(
        status.is_ok(),
        "BCC binary at '{}' could not be executed: {:?}",
        bcc.display(),
        status.err()
    );
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

/// Compiles a fixture file, asserts compilation success, runs the resulting
/// binary, asserts successful execution, and validates expected stdout output.
///
/// Cleans up the temporary binary after execution.
///
/// # Arguments
/// * `fixture`       — Relative path within `tests/fixtures/`.
/// * `target`        — Target architecture string.
/// * `test_name`     — Name suffix for the temporary output file.
/// * `expected_stdout` — Expected exact stdout string from the binary.
fn compile_run_and_assert(fixture: &str, target: &str, test_name: &str, expected_stdout: &str) {
    let source = common::fixture_path(fixture);
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path(test_name);
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");
    let target_flag = format!("--target={}", target);

    // Compile
    let compile_output = common::compile(source_str, &[&target_flag, "-o", output_str]);
    common::assert_compilation_succeeds(&compile_output);

    // Verify binary was produced
    assert!(
        Path::new(output_str).exists(),
        "Compiled binary not found at '{}'",
        output_str
    );

    // Run
    let run_output = common::run_binary(output_str, target);
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, expected_stdout);

    // Clean up
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Checkpoint 2 Tests — PUA Encoding Round-Trip
// ===========================================================================

/// Validates PUA (Private Use Area) encoding byte-exact fidelity.
///
/// Per AAP §0.7.9: "Non-UTF-8 bytes (0x80–0xFF) in C source files MUST
/// survive the entire pipeline with byte-exact fidelity."
///
/// Workflow:
/// 1. Compile `pua_roundtrip.c` (contains `\x80\xFF` in string literals).
/// 2. Inspect `.rodata` section with `objdump -s`.
/// 3. Assert exact bytes `80 ff` are present.
/// 4. Run the binary and verify "PUA round-trip OK\n" output.
#[test]
fn test_pua_encoding_roundtrip() {
    let source = common::fixture_path("pua_roundtrip.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("pua_roundtrip");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Compile for x86-64 (primary validation architecture)
    let compile_output = common::compile(source_str, &["--target=x86-64", "-o", output_str]);
    common::assert_compilation_succeeds(&compile_output);

    // Verify the binary was produced
    assert!(
        Path::new(output_str).exists(),
        "Compiled binary not found at '{}'",
        output_str
    );

    // Inspect .rodata section using objdump for byte-exact verification
    let rodata = common::objdump_section_contents(output_str, ".rodata");

    // The PUA round-trip test requires bytes 0x80 and 0xFF to be present
    // in the .rodata section. objdump -s outputs hex bytes in groups of 4
    // with spaces between. We search for the characteristic byte pattern.
    //
    // The fixture `pua_roundtrip.c` defines:
    //   static const char core_data[] = "\x80\xFF";
    // which should produce bytes 80 ff 00 in .rodata.
    let rodata_lower = rodata.to_lowercase();
    let has_pua_bytes = rodata_lower.contains("80ff")
        || rodata_lower.contains("80 ff")
        || contains_hex_bytes(&rodata_lower, &[0x80, 0xFF]);

    assert!(
        has_pua_bytes,
        "PUA round-trip FAILED: bytes 0x80 0xFF not found in .rodata section.\n\
         objdump output:\n{}",
        rodata
    );

    // Run the binary and verify correct execution
    let run_output = common::run_binary(output_str, "x86-64");
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "PUA round-trip OK\n");

    // Clean up
    let _ = fs::remove_file(&output_path);
}

/// Searches the objdump hex output for a sequence of bytes.
///
/// objdump -s outputs hex dumps in a format like:
///   401000 48656c6c 6f2c2057 6f726c64 210a0080  Hello, World!...
///
/// The hex portion is space-separated groups of 8 hex characters (4 bytes).
/// This function flattens the hex portion and searches for the byte sequence.
fn contains_hex_bytes(objdump_output: &str, bytes: &[u8]) -> bool {
    // Build the target hex string from the byte sequence
    let target: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();

    // Extract hex content from each line of objdump output.
    // Lines with hex data have the format:
    //   <address> <hex1> <hex2> <hex3> <hex4>  <ascii>
    for line in objdump_output.lines() {
        let trimmed = line.trim();
        // Skip non-data lines (headers, section names, etc.)
        if trimmed.is_empty() || !trimmed.starts_with(|c: char| c.is_ascii_hexdigit()) {
            continue;
        }

        // Extract hex data: split by whitespace, skip address (first field),
        // collect hex groups until we hit the ASCII representation column.
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        let mut hex_content = String::new();
        for part in &parts[1..] {
            // Stop when we hit non-hex characters (ASCII representation column)
            if part.chars().all(|c| c.is_ascii_hexdigit()) && part.len() <= 8 {
                hex_content.push_str(part);
            } else {
                break;
            }
        }

        if hex_content.contains(&target) {
            return true;
        }
    }
    false
}

// ===========================================================================
// Checkpoint 2 Tests — Recursive Macro Termination
// ===========================================================================

/// Validates that self-referential macros terminate via paint-marker protection.
///
/// Per AAP User Example: `#define A A` and `int x = A;` → terminates in
/// <5 seconds, no hang. The paint-marker system marks expanded macro tokens
/// to suppress re-expansion, preventing infinite recursion.
///
/// This test measures compilation wall-clock time and asserts it completes
/// within the 5-second budget.
#[test]
fn test_recursive_macro_terminates() {
    let source = common::fixture_path("recursive_macro.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("recursive_macro");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    // Start timing — compilation MUST complete within 5 seconds
    let start = Instant::now();

    let compile_output = common::compile(source_str, &["--target=x86-64", "-o", output_str]);

    let elapsed = start.elapsed();
    let timeout = Duration::from_secs(5);

    // Assert compilation completed within the timeout
    assert!(
        elapsed < timeout,
        "Recursive macro compilation timed out!\n\
         Elapsed: {:.2}s (limit: {}s)\n\
         Paint-marker recursion protection may not be functioning correctly.\n\
         stderr: {}",
        elapsed.as_secs_f64(),
        timeout.as_secs(),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    // Assert compilation succeeded
    common::assert_compilation_succeeds(&compile_output);

    // Verify the binary was produced
    assert!(
        Path::new(output_str).exists(),
        "Compiled binary not found at '{}' after recursive macro compilation",
        output_str
    );

    // Run the binary — should exit cleanly
    let run_output = common::run_binary(output_str, "x86-64");
    common::assert_exit_success(&run_output);

    // Log timing for informational purposes
    eprintln!(
        "Recursive macro compilation completed in {:.3}s (limit: {}s)",
        elapsed.as_secs_f64(),
        timeout.as_secs()
    );

    // Clean up
    let _ = fs::remove_file(&output_path);
}

// ===========================================================================
// Checkpoint 2 Tests — GCC Statement Expressions
// ===========================================================================

/// Validates GCC statement expression `({ ... })` support.
///
/// Statement expressions are a GCC extension heavily used in the Linux kernel
/// for type-safe min/max macros. The value of a statement expression is the
/// value of the last expression statement within the braces.
///
/// Tests: basic, multi-statement, macro-based (MAX/MIN), nested, control flow,
///        loop within statement expressions.
///
/// Uses `compile_and_run()` for end-to-end compilation + execution in a single
/// call (demonstrating the simpler workflow for standard test patterns).
#[test]
fn test_statement_expressions() {
    ensure_bcc_available();

    let source = common::fixture_path("stmt_expr.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");

    let run_output = common::compile_and_run(source_str, "x86-64");
    common::assert_exit_success(&run_output);
    common::assert_stdout_eq(&run_output, "stmt_expr OK\n");
}

// ===========================================================================
// Checkpoint 2 Tests — typeof / __typeof__
// ===========================================================================

/// Validates `typeof`/`__typeof__` GCC extension support.
///
/// typeof infers types at compile time and is heavily used in Linux kernel
/// macros for type-safe generic programming. Both `typeof(expression)` and
/// `typeof(type-name)` forms must be supported, along with `__typeof__`.
///
/// Tests: typeof with expressions, type names, pointers, array elements,
///        structs, const qualifiers, macro usage (kernel-style max/min/swap).
#[test]
fn test_typeof() {
    compile_run_and_assert("typeof_test.c", "x86-64", "typeof_test", "typeof OK\n");
}

// ===========================================================================
// Checkpoint 2 Tests — Designated Initializers
// ===========================================================================

/// Validates designated initializer support.
///
/// Tests out-of-order field designation, nested designation (`.field.subfield`),
/// array index designation, brace elision, and implicit zero-initialization
/// of unspecified members.
///
/// Exercises `src/frontend/sema/initializer.rs`.
#[test]
fn test_designated_initializers() {
    compile_run_and_assert(
        "designated_init.c",
        "x86-64",
        "designated_init",
        "designated_init OK\n",
    );
}

// ===========================================================================
// Checkpoint 2 Tests — Inline Assembly (Basic)
// ===========================================================================

/// Validates basic inline assembly support (`asm volatile`, AT&T syntax).
///
/// Inline assembly is critical for Linux kernel compilation. This test covers:
/// - No-operand asm (`asm volatile("nop")`)
/// - `__asm__ __volatile__` alternate spelling
/// - Output operands (`"=r"`)
/// - Input + output operands
/// - Memory clobber (`"memory"`)
///
/// Targets x86-64 as the primary validation architecture.
#[test]
fn test_inline_asm_basic() {
    compile_run_and_assert(
        "inline_asm_basic.c",
        "x86-64",
        "inline_asm_basic",
        "inline_asm_basic OK\n",
    );
}

// ===========================================================================
// Checkpoint 2 Tests — Inline Assembly (Constraints)
// ===========================================================================

/// Validates inline assembly constraint support.
///
/// Tests various constraint types:
/// - `"=r"` — output to register
/// - `"=m"` — output to memory
/// - `"+r"` — read-write register
/// - `"r"`  — input from register
/// - `"i"`  — immediate integer operand
/// - `"cc"` — condition code clobber
/// - Named operands `[name]`
/// - Combined multiple inputs, outputs, and clobbers
///
/// Targets x86-64 as the primary validation architecture.
#[test]
fn test_inline_asm_constraints() {
    compile_run_and_assert(
        "inline_asm_constraints.c",
        "x86-64",
        "inline_asm_constraints",
        "inline_asm_constraints OK\n",
    );
}

// ===========================================================================
// Checkpoint 2 Tests — Computed Goto
// ===========================================================================

/// Validates GCC computed goto extension (`goto *ptr`).
///
/// Computed gotos use `&&label` (address-of-label operator) and `goto *expr`
/// (indirect goto) for dispatch tables — a pattern used in performance-critical
/// kernel bytecode interpreters and threaded interpreters.
///
/// Tests: dispatch table with multiple targets, direct address-of-label usage,
///        correct execution path verification.
#[test]
fn test_computed_goto() {
    compile_run_and_assert(
        "computed_goto.c",
        "x86-64",
        "computed_goto",
        "computed_goto OK\n",
    );
}

// ===========================================================================
// Checkpoint 2 Tests — Zero-Length Array
// ===========================================================================

/// Validates GCC zero-length array extension (`type data[0]`).
///
/// Zero-length arrays are used as flexible array-like members in pre-C99 code
/// and are still present in the Linux kernel for variable-length data patterns.
///
/// Tests: sizeof excludes the zero-length array, dynamic allocation with extra
///        trailing space, read/write through the array tail.
#[test]
fn test_zero_length_array() {
    compile_run_and_assert(
        "zero_length_array.c",
        "x86-64",
        "zero_length_array",
        "zero_length_array OK\n",
    );
}

// ===========================================================================
// Checkpoint 2 Tests — GCC Builtins
// ===========================================================================

/// Validates GCC builtin function support.
///
/// Tests compile-time builtins:
/// - `__builtin_constant_p` — constant detection
/// - `__builtin_offsetof`   — struct member offset
/// - `__builtin_types_compatible_p` — type compatibility check
/// - `__builtin_choose_expr` — compile-time ternary
///
/// Tests runtime builtins:
/// - `__builtin_clz` / `__builtin_ctz` — leading/trailing zero count
/// - `__builtin_popcount` — population count
/// - `__builtin_bswap32` / `__builtin_bswap64` — byte swap
/// - `__builtin_ffs` — find first set bit
/// - `__builtin_expect` — branch prediction hint
#[test]
fn test_builtins() {
    compile_run_and_assert("builtins.c", "x86-64", "builtins", "builtins OK\n");
}

// ===========================================================================
// Checkpoint 2 Tests — C11 _Static_assert
// ===========================================================================

/// Validates C11 `_Static_assert` support.
///
/// Tests `_Static_assert` at file scope and block scope with true conditions.
/// All assertions in the fixture use true conditions so the program compiles
/// and runs successfully.
///
/// Exercises `src/frontend/parser/declarations.rs` (_Static_assert parsing)
/// and `src/frontend/sema/constant_eval.rs` (constant expression evaluation).
#[test]
fn test_static_assert() {
    compile_run_and_assert(
        "static_assert.c",
        "x86-64",
        "static_assert",
        "static_assert OK\n",
    );
}

/// Validates that `_Static_assert` with a false condition produces a
/// compilation error with the expected diagnostic message.
///
/// This is a negative test — we expect compilation to **fail** and check
/// that the error message contains the assertion string.
#[test]
fn test_static_assert_failure() {
    // Create a temporary source file with a failing _Static_assert
    let tmp_source = common::temp_output_path("static_assert_fail_src");
    let tmp_source_str = tmp_source.to_str().expect("temp path is valid UTF-8");
    let source_with_ext = format!("{}.c", tmp_source_str);

    fs::write(
        &source_with_ext,
        b"_Static_assert(0, \"intentional failure for testing\");\nint main(void) { return 0; }\n",
    )
    .expect("Failed to write temporary source file");

    let output_path = common::temp_output_path("static_assert_fail_bin");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let compile_output = common::compile(&source_with_ext, &["--target=x86-64", "-o", output_str]);

    // Compilation MUST fail for a false _Static_assert
    assert!(
        !compile_output.status.success(),
        "_Static_assert(0, ...) should cause compilation failure, but it succeeded"
    );

    // Check that the error message references the assertion
    let stderr = String::from_utf8_lossy(&compile_output.stderr);
    let stdout = String::from_utf8_lossy(&compile_output.stdout);
    let combined = format!("{}{}", stderr, stdout);
    let has_diagnostic = combined.contains("intentional failure")
        || combined.contains("static_assert")
        || combined.contains("Static_assert")
        || combined.contains("assertion");
    assert!(
        has_diagnostic,
        "_Static_assert failure should produce a diagnostic containing the assertion message.\n\
         stdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Clean up using batch cleanup helper and PathBuf::from for path construction
    let source_path_str = source_with_ext.clone();
    let output_path_str = PathBuf::from(output_str);
    common::cleanup_temp_files(&[&source_path_str, output_path_str.to_str().unwrap_or("")]);
}

// ===========================================================================
// Checkpoint 2 Tests — C11 _Generic Selection
// ===========================================================================

/// Validates C11 `_Generic` selection expression support.
///
/// `_Generic` provides compile-time type-based dispatch, used in C11
/// type-generic math macros. The controlling expression is NOT evaluated —
/// only its type matters.
///
/// Tests: type_id macro with int/double/char*/default, type_name macro,
///        direct _Generic expression with float literal.
#[test]
fn test_generic_selection() {
    compile_run_and_assert("generic.c", "x86-64", "generic", "generic OK\n");
}

// ===========================================================================
// Checkpoint 2 Tests — Cross-Architecture Language Verification
// ===========================================================================

/// Cross-architecture PUA encoding round-trip test for i686.
///
/// Verifies that PUA encoding fidelity is maintained on the i686 target.
/// Per AAP §0.7.5 backend validation order: x86-64 → i686 → AArch64 → RISC-V 64.
#[test]
fn test_pua_roundtrip_i686() {
    if !common::qemu_available("i686") {
        eprintln!("SKIP: qemu-i386 not available for i686 cross-arch PUA test");
        return;
    }
    compile_run_and_assert(
        "pua_roundtrip.c",
        "i686",
        "pua_roundtrip_i686",
        "PUA round-trip OK\n",
    );
}

/// Cross-architecture PUA encoding round-trip test for AArch64.
///
/// Verifies that PUA encoding fidelity is maintained on the AArch64 target.
#[test]
fn test_pua_roundtrip_aarch64() {
    if !common::qemu_available("aarch64") {
        eprintln!("SKIP: qemu-aarch64 not available for aarch64 cross-arch PUA test");
        return;
    }
    compile_run_and_assert(
        "pua_roundtrip.c",
        "aarch64",
        "pua_roundtrip_aarch64",
        "PUA round-trip OK\n",
    );
}

/// Cross-architecture PUA encoding round-trip test for RISC-V 64.
///
/// Verifies that PUA encoding fidelity is maintained on the RISC-V 64 target.
#[test]
fn test_pua_roundtrip_riscv64() {
    if !common::qemu_available("riscv64") {
        eprintln!("SKIP: qemu-riscv64 not available for riscv64 cross-arch PUA test");
        return;
    }
    compile_run_and_assert(
        "pua_roundtrip.c",
        "riscv64",
        "pua_roundtrip_riscv64",
        "PUA round-trip OK\n",
    );
}

/// Cross-architecture recursive macro termination test for i686.
///
/// Verifies that paint-marker recursion protection works on i686 target.
/// Must complete within 5 seconds per AAP User Example.
#[test]
fn test_recursive_macro_i686() {
    if !common::qemu_available("i686") {
        eprintln!("SKIP: qemu-i386 not available for i686 recursive macro test");
        return;
    }

    let source = common::fixture_path("recursive_macro.c");
    let source_str = source.to_str().expect("fixture path is valid UTF-8");
    let output_path = common::temp_output_path("recursive_macro_i686");
    let output_str = output_path.to_str().expect("temp path is valid UTF-8");

    let start = Instant::now();

    let compile_output = common::compile(source_str, &["--target=i686", "-o", output_str]);

    let elapsed = start.elapsed();
    let timeout = Duration::from_secs(5);

    assert!(
        elapsed < timeout,
        "Recursive macro compilation for i686 timed out! Elapsed: {:.2}s (limit: {}s)",
        elapsed.as_secs_f64(),
        timeout.as_secs()
    );

    common::assert_compilation_succeeds(&compile_output);

    let run_output = common::run_binary(output_str, "i686");
    common::assert_exit_success(&run_output);

    let _ = fs::remove_file(&output_path);
}

/// Cross-architecture statement expression test for AArch64.
///
/// Verifies GCC statement expressions work on AArch64 target.
#[test]
fn test_stmt_expr_aarch64() {
    if !common::qemu_available("aarch64") {
        eprintln!("SKIP: qemu-aarch64 not available for aarch64 stmt_expr test");
        return;
    }
    compile_run_and_assert(
        "stmt_expr.c",
        "aarch64",
        "stmt_expr_aarch64",
        "stmt_expr OK\n",
    );
}

/// Cross-architecture typeof test for RISC-V 64.
///
/// Verifies typeof/__typeof__ works on RISC-V 64 target.
#[test]
fn test_typeof_riscv64() {
    if !common::qemu_available("riscv64") {
        eprintln!("SKIP: qemu-riscv64 not available for riscv64 typeof test");
        return;
    }
    compile_run_and_assert(
        "typeof_test.c",
        "riscv64",
        "typeof_test_riscv64",
        "typeof OK\n",
    );
}

/// Cross-architecture designated initializer test for i686.
///
/// Verifies designated initializers work correctly on i686 target,
/// including out-of-order, nested, and array-index designation.
#[test]
fn test_designated_init_i686() {
    if !common::qemu_available("i686") {
        eprintln!("SKIP: qemu-i386 not available for i686 designated init test");
        return;
    }
    compile_run_and_assert(
        "designated_init.c",
        "i686",
        "designated_init_i686",
        "designated_init OK\n",
    );
}

/// Cross-architecture builtins test for AArch64.
///
/// Verifies GCC builtins work correctly on AArch64 target.
/// Note: Some builtins like __builtin_clz may have architecture-specific
/// instruction selection but must produce the same logical result.
#[test]
fn test_builtins_aarch64() {
    if !common::qemu_available("aarch64") {
        eprintln!("SKIP: qemu-aarch64 not available for aarch64 builtins test");
        return;
    }
    compile_run_and_assert("builtins.c", "aarch64", "builtins_aarch64", "builtins OK\n");
}

/// Cross-architecture _Generic selection test for RISC-V 64.
///
/// Verifies C11 _Generic type dispatch works correctly on RISC-V 64.
#[test]
fn test_generic_riscv64() {
    if !common::qemu_available("riscv64") {
        eprintln!("SKIP: qemu-riscv64 not available for riscv64 generic test");
        return;
    }
    compile_run_and_assert("generic.c", "riscv64", "generic_riscv64", "generic OK\n");
}

/// Cross-architecture computed goto test for i686.
///
/// Verifies computed goto dispatch table works correctly on i686 target.
#[test]
fn test_computed_goto_i686() {
    if !common::qemu_available("i686") {
        eprintln!("SKIP: qemu-i386 not available for i686 computed goto test");
        return;
    }
    compile_run_and_assert(
        "computed_goto.c",
        "i686",
        "computed_goto_i686",
        "computed_goto OK\n",
    );
}

/// Cross-architecture zero-length array test for AArch64.
///
/// Verifies the GCC zero-length array extension works on AArch64 target.
#[test]
fn test_zero_length_array_aarch64() {
    if !common::qemu_available("aarch64") {
        eprintln!("SKIP: qemu-aarch64 not available for aarch64 zero-length array test");
        return;
    }
    compile_run_and_assert(
        "zero_length_array.c",
        "aarch64",
        "zero_length_array_aarch64",
        "zero_length_array OK\n",
    );
}

/// Cross-architecture _Static_assert test for RISC-V 64.
///
/// Verifies _Static_assert compile-time assertions work on RISC-V 64.
#[test]
fn test_static_assert_riscv64() {
    if !common::qemu_available("riscv64") {
        eprintln!("SKIP: qemu-riscv64 not available for riscv64 static_assert test");
        return;
    }
    compile_run_and_assert(
        "static_assert.c",
        "riscv64",
        "static_assert_riscv64",
        "static_assert OK\n",
    );
}
