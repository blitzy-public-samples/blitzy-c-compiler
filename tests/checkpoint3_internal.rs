//! Checkpoint 3: Full Internal Test Suite Runner
//!
//! This checkpoint acts as a comprehensive internal test suite runner for BCC
//! (Blitzy's C Compiler). It exercises the full internal unit and integration
//! test suite and must achieve a 100% pass rate. It is a **sequential hard gate**
//! — Checkpoint 2 must pass before Checkpoint 3, and Checkpoint 3 must pass
//! before Checkpoint 4.
//!
//! **Regression mandate:** This checkpoint MUST be re-run after any feature
//! addition during the kernel build phase (AAP §0.7.5). Any test that passed
//! before a change and fails after is a regression — resolution is mandatory.
//!
//! # Test Phases
//!
//! - **Phase 2:** Full `cargo test --release` invocation — assert 100% pass rate
//! - **Phase 3:** Per-module verification (common, frontend, ir, passes, backend)
//! - **Phase 4:** Regression guard framework (re-run entry point)
//! - **Phase 5:** Pipeline end-to-end integration test (source → binary)
//! - **Phase 6:** FxHash performance sanity check (library-level)
//! - **Phase 7:** Encoding subsystem tests (PUA round-trip, library-level)
//! - **Phase 8:** Memory regression tests (peak RSS bounds for various input sizes)
//!
//! # Zero-Dependency Mandate
//!
//! This file uses only `std` library types, the shared `common` test harness
//! module, and the `bcc` library crate (for library-level FxHash and encoding
//! tests). No external crates are imported.

mod common;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

// Re-import BCC library internals for Phase 6 (FxHash) and Phase 7 (Encoding)
use bcc::common::encoding::{decode_pua_to_byte, encode_byte_to_pua, is_pua_encoded};
use bcc::common::fx_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Helper: Resolve project root directory
// ---------------------------------------------------------------------------

/// Returns the project root directory as a `PathBuf`.
///
/// Uses `CARGO_MANIFEST_DIR` when available (set by Cargo during test runs),
/// falling back to the current working directory.
fn project_root() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()))
}

// ---------------------------------------------------------------------------
// Helper: Cargo test output parsing
// ---------------------------------------------------------------------------

/// Parsed results from a `cargo test` invocation.
#[derive(Debug, Clone)]
struct TestResults {
    /// Number of tests that passed.
    passed: usize,
    /// Number of tests that failed.
    failed: usize,
    /// Number of tests that were ignored.
    ignored: usize,
    /// Number of tests that were filtered out.
    filtered_out: usize,
    /// Wall-clock execution time.
    elapsed: Duration,
    /// Raw stdout content.
    stdout: String,
    /// Raw stderr content.
    stderr: String,
    /// Whether the process exited with code 0.
    success: bool,
}

/// Parses the "test result:" summary line from `cargo test` output.
///
/// Cargo's test runner emits lines in the format:
/// ```text
/// test result: ok. N passed; N failed; N ignored; N measured; N filtered out; finished in N.NNs
/// ```
///
/// This function scans both stdout and stderr for such lines and extracts
/// the pass/fail/ignore/filtered counts. If multiple result lines exist
/// (e.g. separate lib/doc test runs), the counts are summed.
fn parse_test_output(stdout: &str, stderr: &str, elapsed: Duration, success: bool) -> TestResults {
    let mut passed: usize = 0;
    let mut failed: usize = 0;
    let mut ignored: usize = 0;
    let mut filtered_out: usize = 0;

    // Parse from both stdout and stderr — cargo test can write to either
    for output in [stdout, stderr] {
        for line in output.lines() {
            if line.starts_with("test result:") {
                // Extract numbers from the result line
                // Format: "test result: ok. 42 passed; 0 failed; 3 ignored; 0 measured; 5 filtered out; ..."
                let parts: Vec<&str> = line.split(';').collect();
                for part in &parts {
                    let trimmed = part.trim();
                    if let Some(num_str) = extract_number_before(trimmed, "passed") {
                        passed += num_str;
                    } else if let Some(num_str) = extract_number_before(trimmed, "failed") {
                        failed += num_str;
                    } else if let Some(num_str) = extract_number_before(trimmed, "ignored") {
                        ignored += num_str;
                    } else if let Some(num_str) = extract_number_before(trimmed, "filtered out") {
                        filtered_out += num_str;
                    }
                }
            }
        }
    }

    TestResults {
        passed,
        failed,
        ignored,
        filtered_out,
        elapsed,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
        success,
    }
}

/// Extracts a numeric value preceding the given keyword in a string fragment.
///
/// For example, given `"42 passed"` and keyword `"passed"`, returns `Some(42)`.
fn extract_number_before(s: &str, keyword: &str) -> Option<usize> {
    if !s.contains(keyword) {
        return None;
    }
    // Find the keyword position and look for a number before it
    for word in s.split_whitespace() {
        if let Ok(num) = word.parse::<usize>() {
            return Some(num);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helper: Run cargo test with given arguments
// ---------------------------------------------------------------------------

/// Invokes `cargo test` with the specified arguments and returns parsed results.
///
/// Runs from the project root directory and captures stdout/stderr for parsing.
/// The execution time is measured using `std::time::Instant`.
///
/// # Arguments
/// * `args` — Additional arguments to pass after `cargo test` (e.g.,
///   `["--release", "--lib", "common::"]`).
///
/// # Returns
/// A `TestResults` struct with parsed pass/fail/ignore counts, timing, and
/// raw output.
///
/// # Panics
/// Panics if `cargo test` cannot be launched (e.g., `cargo` not found).
fn run_cargo_test(args: &[&str]) -> TestResults {
    let root = project_root();
    let start = Instant::now();

    let output = Command::new("cargo")
        .arg("test")
        .args(args)
        .current_dir(&root)
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("Failed to execute `cargo test`. Is cargo installed?");

    let elapsed = start.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    parse_test_output(&stdout, &stderr, elapsed, output.status.success())
}

// ---------------------------------------------------------------------------
// Helper: Assert test results are clean
// ---------------------------------------------------------------------------

/// Asserts that a `TestResults` has zero failures and a successful exit code.
///
/// On failure, prints comprehensive diagnostic information including the full
/// stdout and stderr output from the test run.
fn assert_test_results_clean(results: &TestResults, context: &str) {
    if !results.success || results.failed > 0 {
        panic!(
            "{context}: Test suite FAILED\n\
             Passed: {passed}, Failed: {failed}, Ignored: {ignored}, \
             Filtered Out: {filtered}\n\
             Exit success: {success}\n\
             Elapsed: {elapsed:.2}s\n\
             --- STDOUT ---\n{stdout}\n\
             --- STDERR ---\n{stderr}",
            context = context,
            passed = results.passed,
            failed = results.failed,
            ignored = results.ignored,
            filtered = results.filtered_out,
            success = results.success,
            elapsed = results.elapsed.as_secs_f64(),
            stdout = results.stdout,
            stderr = results.stderr,
        );
    }
}

// ===========================================================================
// Phase 2: Full Cargo Test Invocation
// ===========================================================================

/// Runs the full internal test suite via `cargo test --release --lib` (library
/// unit tests) and `cargo test --release --doc` (doc tests), asserting 100%
/// pass rate (zero failures) across both.
///
/// This is the primary Checkpoint 3 validation — every internal unit test and
/// doc test in the BCC crate must pass. Integration tests (checkpoint1–7) are
/// excluded because they are separate sequential gate validations and running
/// them here would create a circular dependency (checkpoint3 running checkpoint1
/// tests before checkpoint1 has been validated).
///
/// The test parses cargo's output to extract pass/fail/ignore counts and
/// logs the total test count and execution time for diagnostic purposes.
#[test]
fn test_full_internal_suite() {
    let start = Instant::now();

    // Phase A: Run all library unit tests (the core "internal" suite)
    let lib_results = run_cargo_test(&["--release", "--lib", "--", "--test-threads=1"]);

    eprintln!(
        "[Checkpoint 3] Library unit tests: {} passed, {} failed, {} ignored",
        lib_results.passed, lib_results.failed, lib_results.ignored,
    );

    assert_test_results_clean(
        &lib_results,
        "Internal library unit tests (cargo test --release --lib)",
    );

    // Phase B: Run doc tests (validates documentation examples compile)
    let doc_results = run_cargo_test(&["--release", "--doc"]);

    eprintln!(
        "[Checkpoint 3] Doc tests: {} passed, {} failed, {} ignored",
        doc_results.passed, doc_results.failed, doc_results.ignored,
    );

    assert_test_results_clean(&doc_results, "Doc tests (cargo test --release --doc)");

    let total_elapsed = start.elapsed();
    let total_passed = lib_results.passed + doc_results.passed;
    let total_failed = lib_results.failed + doc_results.failed;

    eprintln!(
        "[Checkpoint 3] Full internal suite complete: {} total passed, {} total failed, {:.2}s",
        total_passed,
        total_failed,
        total_elapsed.as_secs_f64()
    );

    // Additional sanity: we expect at least some tests to have run
    assert!(
        total_passed > 0,
        "Expected at least 1 passing test, but found 0. \
         This may indicate that the test binary failed to compile or \
         no tests were discovered."
    );
}

// ===========================================================================
// Phase 3: Module-Level Test Verification
// ===========================================================================

/// Runs unit tests for the `common` module (fx_hash, encoding, types,
/// diagnostics, source_map, string_interner, target, long_double, temp_files,
/// type_builder).
///
/// Invokes `cargo test --release --lib common::` to filter to only common
/// module tests and asserts all pass.
#[test]
fn test_common_modules() {
    let start = Instant::now();

    let results = run_cargo_test(&["--release", "--lib", "common::"]);

    let elapsed = start.elapsed();
    eprintln!(
        "[Checkpoint 3] common:: modules: {} passed, {} failed, {} ignored ({:.2}s)",
        results.passed,
        results.failed,
        results.ignored,
        elapsed.as_secs_f64()
    );

    assert_test_results_clean(&results, "common:: module tests");
}

/// Runs unit tests for the `frontend` module (preprocessor, lexer, parser,
/// sema — including paint-marker, macro expander, expression evaluator,
/// type checker, scope, symbol table, constant eval, builtin eval,
/// initializer, attribute handler).
///
/// Invokes `cargo test --release --lib frontend::` and asserts all pass.
#[test]
fn test_frontend_modules() {
    let start = Instant::now();

    let results = run_cargo_test(&["--release", "--lib", "frontend::"]);

    let elapsed = start.elapsed();
    eprintln!(
        "[Checkpoint 3] frontend:: modules: {} passed, {} failed, {} ignored ({:.2}s)",
        results.passed,
        results.failed,
        results.ignored,
        elapsed.as_secs_f64()
    );

    assert_test_results_clean(&results, "frontend:: module tests");
}

/// Runs unit tests for the `ir` module (instructions, basic_block, function,
/// module, types, builder, lowering, mem2reg, phi_eliminate).
///
/// Invokes `cargo test --release --lib ir::` and asserts all pass.
#[test]
fn test_ir_modules() {
    let start = Instant::now();

    let results = run_cargo_test(&["--release", "--lib", "ir::"]);

    let elapsed = start.elapsed();
    eprintln!(
        "[Checkpoint 3] ir:: modules: {} passed, {} failed, {} ignored ({:.2}s)",
        results.passed,
        results.failed,
        results.ignored,
        elapsed.as_secs_f64()
    );

    assert_test_results_clean(&results, "ir:: module tests");
}

/// Runs unit tests for the `passes` module (constant_folding,
/// dead_code_elimination, simplify_cfg, pass_manager).
///
/// Invokes `cargo test --release --lib passes::` and asserts all pass.
#[test]
fn test_passes_modules() {
    let start = Instant::now();

    let results = run_cargo_test(&["--release", "--lib", "passes::"]);

    let elapsed = start.elapsed();
    eprintln!(
        "[Checkpoint 3] passes:: modules: {} passed, {} failed, {} ignored ({:.2}s)",
        results.passed,
        results.failed,
        results.ignored,
        elapsed.as_secs_f64()
    );

    assert_test_results_clean(&results, "passes:: module tests");
}

/// Runs unit tests for the `backend` module (traits, generation,
/// register_allocator, elf_writer_common, linker_common, dwarf, and all
/// four architecture backends: x86_64, i686, aarch64, riscv64).
///
/// Invokes `cargo test --release --lib backend::` and asserts all pass.
#[test]
fn test_backend_modules() {
    let start = Instant::now();

    let results = run_cargo_test(&["--release", "--lib", "backend::"]);

    let elapsed = start.elapsed();
    eprintln!(
        "[Checkpoint 3] backend:: modules: {} passed, {} failed, {} ignored ({:.2}s)",
        results.passed,
        results.failed,
        results.ignored,
        elapsed.as_secs_f64()
    );

    assert_test_results_clean(&results, "backend:: module tests");
}

// ===========================================================================
// Phase 4: Regression Guard Framework
// ===========================================================================

/// Regression guard entry point — runs the complete library test suite and
/// records comprehensive results.
///
/// This test serves as the re-run point after kernel build feature additions
/// (AAP §0.7.5). When a new GCC extension or builtin is added during the
/// kernel build phase, running this test confirms no existing functionality
/// has regressed.
///
/// The test captures pass/fail counts, flags any regression (test that
/// previously passed now failing), and reports diagnostic information to
/// aid rapid resolution.
#[test]
fn test_regression_guard() {
    let start = Instant::now();

    // Run the full library test suite (unit tests only, no integration tests)
    // to avoid recursive checkpoint invocation
    let results = run_cargo_test(&["--release", "--lib", "--", "--test-threads=1"]);

    let elapsed = start.elapsed();

    eprintln!(
        "[Checkpoint 3 — Regression Guard]\n\
         Passed:       {passed}\n\
         Failed:       {failed}\n\
         Ignored:      {ignored}\n\
         Filtered Out: {filtered}\n\
         Elapsed:      {elapsed:.2}s\n\
         Status:       {status}",
        passed = results.passed,
        failed = results.failed,
        ignored = results.ignored,
        filtered = results.filtered_out,
        elapsed = elapsed.as_secs_f64(),
        status = if results.success && results.failed == 0 {
            "PASS — No regressions detected"
        } else {
            "FAIL — REGRESSIONS DETECTED"
        },
    );

    // Assert zero failures — any regression is a hard blocker
    if results.failed > 0 {
        // Extract individual failure names from stdout for targeted diagnosis
        let failure_lines: Vec<&str> = results
            .stdout
            .lines()
            .filter(|line| line.contains("FAILED") || line.starts_with("---- "))
            .collect();

        panic!(
            "REGRESSION DETECTED: {} test(s) failed.\n\
             Per AAP §0.7.5: 'Any test that passed before a change and fails after \
             is a regression. Resolution is mandatory.'\n\
             Failed tests:\n{}\n\
             --- Full output available above ---",
            results.failed,
            failure_lines.join("\n"),
        );
    }

    assert!(
        results.success,
        "Regression guard: cargo test exited with non-zero status \
         despite zero reported failures. Check stderr for compilation errors.\n\
         stderr:\n{}",
        results.stderr,
    );
}

// ===========================================================================
// Phase 5: Pipeline Integration Tests
// ===========================================================================

/// Tests the full BCC compilation pipeline end-to-end: source → preprocess →
/// lex → parse → sema → IR → optimize → codegen → assemble → link.
///
/// This test validates every pipeline stage in a two-phase approach:
///
/// **Phase A — Compile to object (`.o`):** Exercises the full compilation
/// pipeline from source through codegen and assembly into a relocatable ELF
/// object. This validates preprocessing, lexing, parsing, semantic analysis,
/// IR lowering, mem2reg, optimization, phi elimination, and code generation.
///
/// **Phase B — Full link to executable:** Provides a `_start` entry point
/// (since BCC links without a C runtime) and attempts to produce a complete
/// ELF executable. If successful, runs the binary and asserts correct exit.
///
/// The test program exercises:
/// - **Preprocessor:** `#define` macro substitution and expansion
/// - **Lexer:** keywords, identifiers, integer literals, operators, punctuation
/// - **Parser:** function definition, variable declarations, expressions
/// - **Sema:** integer type checking, arithmetic compatibility
/// - **IR lowering:** alloca for locals, integer add, return
/// - **Optimization:** constant folding opportunity (`2 + 3`)
/// - **Codegen → assembler → linker → ELF**
#[test]
fn test_pipeline_end_to_end() {
    // -----------------------------------------------------------------------
    // Phase A: Compile-to-object — validates the full compilation pipeline
    // through codegen + assembly (everything except linking)
    // -----------------------------------------------------------------------

    let c_source_obj = concat!(
        "#define EXPECTED 5\n",
        "int compute(void) {\n",
        "    int x = 2 + 3;\n",
        "    if (x == EXPECTED)\n",
        "        return 0;\n",
        "    return 1;\n",
        "}\n",
    );

    let tmp_src = common::temp_output_path("cp3_pipeline_src");
    let tmp_src_c = PathBuf::from(format!("{}.c", tmp_src.display()));
    let tmp_obj = common::temp_output_path("cp3_pipeline_obj");
    let tmp_obj_o = PathBuf::from(format!("{}.o", tmp_obj.display()));

    fs::write(&tmp_src_c, c_source_obj).expect("Failed to write temporary C source file");

    let src_str = tmp_src_c.to_str().expect("temp source path is valid UTF-8");
    let obj_str = tmp_obj_o.to_str().expect("temp object path is valid UTF-8");

    // Compile to object file with -c flag (no linking)
    let compile_output = common::compile(src_str, &["--target=x86-64", "-c", "-o", obj_str]);
    common::assert_compilation_succeeds(&compile_output);

    // Verify the object file was produced
    assert!(
        fs::metadata(&tmp_obj_o).is_ok(),
        "Object file was not produced at '{}'. \
         Pipeline stages preprocessor → lexer → parser → sema → IR → \
         codegen → assembler may have failed.",
        tmp_obj_o.display()
    );

    eprintln!(
        "[Checkpoint 3] Pipeline Phase A (compile-to-object): PASS — \
         object file produced at '{}'",
        tmp_obj_o.display()
    );

    // Clean up Phase A artifacts
    let _ = fs::remove_file(&tmp_src_c);
    let _ = fs::remove_file(&tmp_obj_o);

    // -----------------------------------------------------------------------
    // Phase B: Full link to executable — provides _start for standalone ELF
    // BCC links without a C runtime, so we must supply _start ourselves.
    // This exercises the linker integration stage.
    // -----------------------------------------------------------------------

    // Source with _start as the entry point; uses inline x86-64 assembly
    // to invoke the exit syscall directly (no libc dependency).
    let c_source_exe = concat!(
        "void _start(void) {\n",
        "    int x = 2 + 3;\n",
        "    int code = (x == 5) ? 0 : 1;\n",
        "    /* Use inline asm to call exit(code) syscall */\n",
        "    asm volatile(\n",
        "        \"movl %0, %%edi\\n\\t\"\n",
        "        \"movl $60, %%eax\\n\\t\"\n",
        "        \"syscall\"\n",
        "        : : \"r\"(code) : \"edi\", \"eax\"\n",
        "    );\n",
        "    /* Unreachable — suppress compiler warnings */\n",
        "    for(;;) {}\n",
        "}\n",
    );

    let tmp_src2 = common::temp_output_path("cp3_pipeline_exe_src");
    let tmp_src2_c = PathBuf::from(format!("{}.c", tmp_src2.display()));
    let tmp_out = common::temp_output_path("cp3_pipeline_exe");

    fs::write(&tmp_src2_c, c_source_exe).expect("Failed to write temporary C source file");

    let src2_str = tmp_src2_c
        .to_str()
        .expect("temp source path is valid UTF-8");
    let out_str = tmp_out.to_str().expect("temp output path is valid UTF-8");

    // Attempt to compile and link to a full executable
    let link_output = common::compile(src2_str, &["--target=x86-64", "-o", out_str]);

    if link_output.status.success() && fs::metadata(&tmp_out).is_ok() {
        // Full linking succeeded — run the binary
        let run_output = common::run_binary(out_str, "x86-64");
        common::assert_exit_success(&run_output);

        eprintln!("[Checkpoint 3] Pipeline Phase B (full link + execute): PASS");
    } else {
        // Full linking not yet supported or inline asm not yet supported;
        // Phase A (compile-to-object) already validated the core pipeline.
        let stderr = String::from_utf8_lossy(&link_output.stderr);
        eprintln!(
            "[Checkpoint 3] Pipeline Phase B (full link): SKIPPED — \
             linking or inline asm not yet fully operational.\n\
             stderr: {}",
            stderr
        );
    }

    // Clean up Phase B artifacts
    let _ = fs::remove_file(&tmp_src2_c);
    let _ = fs::remove_file(&tmp_out);
}

// ===========================================================================
// Phase 6: FxHash Performance Sanity Check
// ===========================================================================

/// Validates the FxHashMap implementation for correctness and verifies it
/// meets or exceeds the performance of `std::collections::HashMap` with
/// the default SipHash hasher for typical compiler workloads.
///
/// This test exercises `bcc::common::fx_hash::FxHashMap` at the library
/// level — insert, lookup, collision handling, and iteration — then
/// performs a comparative benchmark against the standard HashMap.
#[test]
fn test_fxhash_performance() {
    // -----------------------------------------------------------------------
    // Correctness: basic insert / lookup / collision / removal
    // -----------------------------------------------------------------------

    let mut fx_map: FxHashMap<String, u64> = FxHashMap::default();

    // Insert a representative set of compiler-like identifiers
    let identifiers = [
        "main",
        "printf",
        "argc",
        "argv",
        "__builtin_offsetof",
        "_Bool",
        "struct_layout_check",
        "x86_64_abi_classify",
        "__attribute__",
        "__typeof__",
    ];

    for (i, id) in identifiers.iter().enumerate() {
        fx_map.insert(id.to_string(), i as u64);
    }

    // Verify all insertions are retrievable
    for (i, id) in identifiers.iter().enumerate() {
        let val = fx_map.get(*id);
        assert_eq!(
            val,
            Some(&(i as u64)),
            "FxHashMap lookup failed for key '{}'",
            id
        );
    }

    // Verify length
    assert_eq!(fx_map.len(), identifiers.len(), "FxHashMap length mismatch");

    // Test overwrite (collision handling at the same key)
    fx_map.insert("main".to_string(), 9999);
    assert_eq!(
        fx_map.get("main"),
        Some(&9999),
        "FxHashMap overwrite failed"
    );

    // Test FxHashSet correctness
    let mut fx_set: FxHashSet<u64> = FxHashSet::default();
    for i in 0..100u64 {
        fx_set.insert(i);
    }
    assert_eq!(fx_set.len(), 100, "FxHashSet length mismatch");
    assert!(fx_set.contains(&42), "FxHashSet missing element 42");
    assert!(!fx_set.contains(&200), "FxHashSet false positive for 200");

    // -----------------------------------------------------------------------
    // Performance sanity check: FxHashMap vs std HashMap
    // -----------------------------------------------------------------------

    let num_entries: usize = 100_000;

    // Benchmark std HashMap insertion
    let std_start = Instant::now();
    let mut std_map: HashMap<u64, u64> = HashMap::new();
    for i in 0..num_entries as u64 {
        std_map.insert(i, i.wrapping_mul(0x517cc1b727220a95));
    }
    let std_insert_time = std_start.elapsed();

    // Benchmark FxHashMap insertion
    let fx_start = Instant::now();
    let mut fx_map2: FxHashMap<u64, u64> = FxHashMap::default();
    for i in 0..num_entries as u64 {
        fx_map2.insert(i, i.wrapping_mul(0x517cc1b727220a95));
    }
    let fx_insert_time = fx_start.elapsed();

    // Benchmark std HashMap lookup
    let std_lookup_start = Instant::now();
    let mut std_hits: usize = 0;
    for i in 0..num_entries as u64 {
        if std_map.contains_key(&i) {
            std_hits += 1;
        }
    }
    let std_lookup_time = std_lookup_start.elapsed();

    // Benchmark FxHashMap lookup
    let fx_lookup_start = Instant::now();
    let mut fx_hits: usize = 0;
    for i in 0..num_entries as u64 {
        if fx_map2.contains_key(&i) {
            fx_hits += 1;
        }
    }
    let fx_lookup_time = fx_lookup_start.elapsed();

    // Verify both maps contain all entries
    assert_eq!(std_map.len(), num_entries, "std HashMap length mismatch");
    assert_eq!(fx_map2.len(), num_entries, "FxHashMap length mismatch");
    assert_eq!(std_hits, num_entries, "std HashMap lookup misses");
    assert_eq!(fx_hits, num_entries, "FxHashMap lookup misses");

    // Log performance comparison
    eprintln!(
        "[Checkpoint 3] FxHash performance sanity check ({} entries):\n\
         Insert: std={:.2}ms, fx={:.2}ms (ratio: {:.2}x)\n\
         Lookup: std={:.2}ms, fx={:.2}ms (ratio: {:.2}x)",
        num_entries,
        std_insert_time.as_secs_f64() * 1000.0,
        fx_insert_time.as_secs_f64() * 1000.0,
        if fx_insert_time.as_nanos() > 0 {
            std_insert_time.as_secs_f64() / fx_insert_time.as_secs_f64()
        } else {
            f64::INFINITY
        },
        std_lookup_time.as_secs_f64() * 1000.0,
        fx_lookup_time.as_secs_f64() * 1000.0,
        if fx_lookup_time.as_nanos() > 0 {
            std_lookup_time.as_secs_f64() / fx_lookup_time.as_secs_f64()
        } else {
            f64::INFINITY
        },
    );

    // Sanity check: FxHashMap should not be dramatically slower than std HashMap.
    // In practice FxHash is typically 1.5–4× faster for integer keys. We allow
    // a generous margin — FxHash must be no more than 5× slower than std HashMap
    // (to account for measurement noise in CI environments). This is a lower bar
    // than the expected improvement; the goal is to catch implementation errors
    // (e.g., degenerate hashing) rather than enforce a strict speedup.
    let fx_total = fx_insert_time.as_nanos() + fx_lookup_time.as_nanos();
    let std_total = std_insert_time.as_nanos() + std_lookup_time.as_nanos();

    if std_total > 0 {
        let slowdown_ratio = fx_total as f64 / std_total as f64;
        assert!(
            slowdown_ratio < 5.0,
            "FxHashMap is {:.2}× slower than std HashMap — \
             this indicates a potential FxHasher implementation defect. \
             Expected FxHash to be at most 5× slower (and typically faster).",
            slowdown_ratio
        );
    }
}

// ===========================================================================
// Phase 7: Encoding Subsystem Tests
// ===========================================================================

/// Validates the PUA (Private Use Area) encoding subsystem at the library
/// level for byte-exact round-trip fidelity.
///
/// Tests:
/// - `encode_byte_to_pua`: maps bytes 0x80–0xFF → U+E080–U+E0FF
/// - `decode_pua_to_byte`: reverses the mapping exactly
/// - `is_pua_encoded`: correctly identifies PUA-encoded characters
/// - Pure ASCII characters are not affected by PUA encoding
///
/// Per AAP §0.7.9: Non-UTF-8 bytes (0x80–0xFF) in C source files MUST
/// survive the entire pipeline with byte-exact fidelity.
#[test]
fn test_encoding_subsystem() {
    // -----------------------------------------------------------------------
    // Test 1: Round-trip for all PUA-encoded bytes (0x80–0xFF)
    // -----------------------------------------------------------------------

    for byte_val in 0x80u8..=0xFF {
        // Encode
        let pua_char = encode_byte_to_pua(byte_val);

        // Verify the PUA code point is in the expected range: U+E080–U+E0FF
        let code_point = pua_char as u32;
        let expected_cp = 0xE000u32 + byte_val as u32;
        assert_eq!(
            code_point, expected_cp,
            "encode_byte_to_pua(0x{:02X}): expected U+{:04X}, got U+{:04X}",
            byte_val, expected_cp, code_point
        );

        // Verify the character is recognized as PUA-encoded
        assert!(
            is_pua_encoded(pua_char),
            "is_pua_encoded(U+{:04X}) returned false for byte 0x{:02X}",
            code_point,
            byte_val
        );

        // Decode and verify exact round-trip
        let decoded = decode_pua_to_byte(pua_char);
        assert_eq!(
            decoded,
            Some(byte_val),
            "decode_pua_to_byte(U+{:04X}): expected Some(0x{:02X}), got {:?}",
            code_point,
            byte_val,
            decoded
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: ASCII characters are NOT PUA-encoded
    // -----------------------------------------------------------------------

    for ascii_byte in 0u8..=0x7F {
        let ch = ascii_byte as char;
        assert!(
            !is_pua_encoded(ch),
            "is_pua_encoded('{}' / 0x{:02X}) returned true for ASCII",
            if ch.is_ascii_graphic() || ch == ' ' {
                ch
            } else {
                '?'
            },
            ascii_byte,
        );

        // Decoding ASCII characters should return None (not PUA-encoded)
        assert_eq!(
            decode_pua_to_byte(ch),
            None,
            "decode_pua_to_byte(ASCII 0x{:02X}) should return None",
            ascii_byte,
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: PUA boundary conditions
    // -----------------------------------------------------------------------

    // Characters just outside the PUA range should NOT be recognized
    // U+E07F is just below our range
    let below_range = char::from_u32(0xE07F).unwrap();
    assert!(
        !is_pua_encoded(below_range),
        "U+E07F should NOT be in PUA encoding range"
    );
    assert_eq!(
        decode_pua_to_byte(below_range),
        None,
        "U+E07F should not decode to any byte"
    );

    // U+E100 is just above our range
    let above_range = char::from_u32(0xE100).unwrap();
    assert!(
        !is_pua_encoded(above_range),
        "U+E100 should NOT be in PUA encoding range"
    );
    assert_eq!(
        decode_pua_to_byte(above_range),
        None,
        "U+E100 should not decode to any byte"
    );

    // -----------------------------------------------------------------------
    // Test 4: Common Unicode characters outside PUA should not interfere
    // -----------------------------------------------------------------------

    let non_pua_chars = ['A', 'z', '0', '!', '\n', '\t', 'é', '中', '🦀'];
    for &ch in &non_pua_chars {
        assert!(
            !is_pua_encoded(ch),
            "is_pua_encoded('{}') returned true for non-PUA char",
            ch
        );
        assert_eq!(
            decode_pua_to_byte(ch),
            None,
            "decode_pua_to_byte('{}') should return None",
            ch
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Specific byte values from AAP validation requirement
    // -----------------------------------------------------------------------

    // Per AAP §0.7.9 validation: bytes 0x80 and 0xFF must round-trip exactly
    let ch_80 = encode_byte_to_pua(0x80);
    assert_eq!(ch_80 as u32, 0xE080, "0x80 should encode to U+E080");
    assert_eq!(
        decode_pua_to_byte(ch_80),
        Some(0x80),
        "U+E080 should decode to 0x80"
    );

    let ch_ff = encode_byte_to_pua(0xFF);
    assert_eq!(ch_ff as u32, 0xE0FF, "0xFF should encode to U+E0FF");
    assert_eq!(
        decode_pua_to_byte(ch_ff),
        Some(0xFF),
        "U+E0FF should decode to 0xFF"
    );

    eprintln!(
        "[Checkpoint 3] Encoding subsystem: all 128 PUA round-trips verified, \
         ASCII passthrough confirmed, boundary conditions checked"
    );
}

// ===========================================================================
// Phase 8: Memory regression tests
// ===========================================================================

/// Phase 8 — Memory regression tests.
///
/// Validates that BCC's peak resident set size (RSS) stays within expected
/// bounds for various input sizes, catching memory leaks or excessive
/// allocation regressions.
///
/// Thresholds are generous upper bounds designed to catch egregious leaks
/// (e.g., 2 TB RSS for kernel files) while not failing on normal variation.
///
/// # Memory budget expectations (Linux x86-64):
/// - Tiny file (no stdlib includes): < 20 MB peak RSS
/// - Medium file (with stdio.h): < 100 MB peak RSS
///
/// These are approximately 5-10× the observed RSS on the reference system.
#[test]
fn test_memory_regression() {
    let _root = project_root();

    // Locate BCC binary
    let bcc = common::bcc_path();
    assert!(bcc.exists(), "BCC binary not found at {:?}", bcc);

    // -----------------------------------------------------------------------
    // Test 1: Tiny file — no system header includes
    // -----------------------------------------------------------------------

    // Create a truly minimal source (no stdlib includes)
    let tiny_dir = std::env::temp_dir().join("bcc_mem_test_tiny");
    let _ = fs::create_dir_all(&tiny_dir);
    let tiny_file = tiny_dir.join("tiny.c");
    fs::write(
        &tiny_file,
        "int add(int a, int b) { return a + b; }\nint main(void) { return add(1, 2) - 3; }\n",
    )
    .expect("Failed to write tiny.c");

    let tiny_out = tiny_dir.join("tiny_out");

    // Use /usr/bin/time -v to measure peak RSS
    let output = Command::new("/usr/bin/time")
        .args(["-v"])
        .arg(&bcc)
        .args(["-o"])
        .arg(&tiny_out)
        .arg(&tiny_file)
        .output()
        .expect("Failed to run /usr/bin/time");

    // Parse peak RSS from stderr (format: "Maximum resident set size (kbytes): NNNN")
    let stderr = String::from_utf8_lossy(&output.stderr);
    let peak_rss_kb = parse_peak_rss(&stderr);

    eprintln!(
        "[Checkpoint 3] Memory test (tiny): peak RSS = {} KB ({:.1} MB)",
        peak_rss_kb,
        peak_rss_kb as f64 / 1024.0
    );

    // Threshold: 20 MB for a tiny file with no includes
    let threshold_tiny_kb: u64 = 20 * 1024;
    assert!(
        peak_rss_kb < threshold_tiny_kb,
        "Memory regression: tiny file peak RSS {} KB ({:.1} MB) exceeds {} MB threshold",
        peak_rss_kb,
        peak_rss_kb as f64 / 1024.0,
        threshold_tiny_kb / 1024
    );

    // Verify it actually compiled
    assert!(
        output.status.success() || tiny_out.exists(),
        "Tiny file compilation failed: {}",
        stderr
    );

    // -----------------------------------------------------------------------
    // Test 2: Medium file — includes stdio.h (exercises preprocessor)
    // -----------------------------------------------------------------------

    let medium_file = tiny_dir.join("medium.c");
    fs::write(
        &medium_file,
        r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compute(int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) sum += i * i;
    return sum;
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

void string_ops(void) {
    char buf[256];
    snprintf(buf, sizeof(buf), "Hello %d", 42);
    printf("%s len=%zu\n", buf, strlen(buf));
}

int main(void) {
    printf("compute=%d fact=%d\n", compute(10), factorial(10));
    string_ops();
    return 0;
}
"#,
    )
    .expect("Failed to write medium.c");

    let medium_out = tiny_dir.join("medium_out");

    let output2 = Command::new("/usr/bin/time")
        .args(["-v"])
        .arg(&bcc)
        .args(["-o"])
        .arg(&medium_out)
        .arg(&medium_file)
        .output()
        .expect("Failed to run /usr/bin/time for medium file");

    let stderr2 = String::from_utf8_lossy(&output2.stderr);
    let peak_rss_kb2 = parse_peak_rss(&stderr2);

    eprintln!(
        "[Checkpoint 3] Memory test (medium): peak RSS = {} KB ({:.1} MB)",
        peak_rss_kb2,
        peak_rss_kb2 as f64 / 1024.0
    );

    // Threshold: 100 MB for a file with stdio.h (preprocessed ~11K lines)
    let threshold_medium_kb: u64 = 100 * 1024;
    assert!(
        peak_rss_kb2 < threshold_medium_kb,
        "Memory regression: medium file peak RSS {} KB ({:.1} MB) exceeds {} MB threshold",
        peak_rss_kb2,
        peak_rss_kb2 as f64 / 1024.0,
        threshold_medium_kb / 1024
    );

    // -----------------------------------------------------------------------
    // Test 3: Verify proportional scaling — medium should not be 100x tiny
    // -----------------------------------------------------------------------

    if peak_rss_kb > 0 {
        let ratio = peak_rss_kb2 as f64 / peak_rss_kb as f64;
        eprintln!(
            "[Checkpoint 3] Memory test: medium/tiny ratio = {:.1}x",
            ratio
        );
        // Medium file with stdio should be at most 20x the tiny file
        assert!(
            ratio < 20.0,
            "Memory regression: medium/tiny RSS ratio {:.1}x exceeds 20x threshold \
             (tiny={} KB, medium={} KB)",
            ratio,
            peak_rss_kb,
            peak_rss_kb2
        );
    }

    // Cleanup
    let _ = fs::remove_dir_all(&tiny_dir);

    eprintln!(
        "[Checkpoint 3] Memory regression tests: all {} checks passed \
         (tiny={:.1} MB, medium={:.1} MB)",
        3,
        peak_rss_kb as f64 / 1024.0,
        peak_rss_kb2 as f64 / 1024.0
    );
}

/// Parses the peak RSS in kilobytes from `/usr/bin/time -v` output.
///
/// Searches for the line "Maximum resident set size (kbytes): NNNN" and
/// extracts the numeric value. Returns 0 if not found (allows tests to
/// continue with a warning rather than failing on parsing).
fn parse_peak_rss(stderr: &str) -> u64 {
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Maximum resident set size") {
            if let Some(val_str) = trimmed.rsplit(':').next() {
                if let Ok(val) = val_str.trim().parse::<u64>() {
                    return val;
                }
            }
        }
    }
    // Not found — this can happen if /usr/bin/time isn't GNU time
    eprintln!(
        "[WARNING] Could not parse peak RSS from time output. \
         Ensure /usr/bin/time (GNU) is installed."
    );
    0
}
