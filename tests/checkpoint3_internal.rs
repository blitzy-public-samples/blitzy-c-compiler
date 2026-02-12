//! Checkpoint 3 — Full Internal Unit and Integration Test Suite Runner
//!
//! This checkpoint integration test validates that the complete BCC compiler
//! internal test suite passes at 100%. It serves as a hard regression gate
//! per Section 0.7.5 of the Agent Action Plan: this checkpoint MUST be re-run
//! after any feature addition during the kernel build phase to confirm no
//! regressions have been introduced.
//!
//! # Test Organization
//!
//! - [`test_cargo_test_all_pass`]: Runs the full library test suite via
//!   `cargo test --release --lib` and asserts zero failures.
//! - [`test_common_module_tests`] through [`test_backend_module_tests`]: Run
//!   tests for individual modules (common, frontend, ir, passes, backend)
//!   using name-filtered `cargo test` invocations.
//! - [`test_regression_after_feature_addition`]: Regression gate mechanism
//!   designed for post-feature-addition validation during the kernel build
//!   phase, with optional enhanced reporting via environment variables.
//!
//! # Sequential Gate
//!
//! This is Checkpoint 3 in the validation pipeline. It must pass before
//! proceeding to Checkpoint 4 (shared library and DWARF validation).
//! Checkpoints 1–6 are strictly sequential hard gates per Section 0.7.5.
//!
//! # Regression Definition (Section 0.7.5)
//!
//! Any test that passed before the current change and fails after is a
//! regression. Resolution is mandatory before proceeding.
//!
//! # Test Count Validation
//!
//! Each test parses `cargo test` output to extract pass/fail/ignore counts.
//! If the `BCC_EXPECTED_TEST_COUNT` environment variable is set, the suite
//! warns when the discovered test count decreases — a signal that tests may
//! have been deleted, potentially masking regressions.

mod common;

use std::env;
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Test result parsing infrastructure
// ---------------------------------------------------------------------------

/// Aggregated results parsed from one or more `cargo test` output summary lines.
///
/// A single `cargo test` invocation may produce multiple `test result:` lines
/// (one per test target — lib tests, doc tests, etc.). This struct accumulates
/// counts across all summary lines encountered in the output.
struct TestResults {
    /// Total number of tests that passed across all test targets.
    passed: u64,
    /// Total number of tests that failed across all test targets.
    failed: u64,
    /// Total number of tests skipped via `#[ignore]`.
    ignored: u64,
    /// Total number of benchmark measurements reported.
    measured: u64,
    /// Total number of tests excluded by the name filter.
    filtered_out: u64,
    /// Total number of tests discovered (sum of all `running N tests` lines).
    total_discovered: u64,
}

impl TestResults {
    /// Create a zeroed test result accumulator.
    fn new() -> Self {
        TestResults {
            passed: 0,
            failed: 0,
            ignored: 0,
            measured: 0,
            filtered_out: 0,
            total_discovered: 0,
        }
    }

    /// Total tests that actually executed (passed + failed, excluding ignored).
    fn total_run(&self) -> u64 {
        self.passed + self.failed
    }

    /// Returns `true` if zero tests failed — the 100% pass rate requirement.
    fn is_all_pass(&self) -> bool {
        self.failed == 0
    }
}

impl std::fmt::Display for TestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} passed; {} failed; {} ignored; {} measured; {} filtered out \
             (discovered: {})",
            self.passed,
            self.failed,
            self.ignored,
            self.measured,
            self.filtered_out,
            self.total_discovered
        )
    }
}

// ---------------------------------------------------------------------------
// Project root resolution
// ---------------------------------------------------------------------------

/// Resolve the BCC project root directory.
///
/// Uses `CARGO_MANIFEST_DIR` (set by Cargo during `cargo test` runs) as the
/// authoritative source, falling back to `std::env::current_dir()` when
/// running outside the Cargo test harness.
///
/// # Panics
///
/// Panics if neither `CARGO_MANIFEST_DIR` is set nor the current working
/// directory can be determined.
fn project_root() -> PathBuf {
    // Primary: CARGO_MANIFEST_DIR is set by Cargo during test execution.
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest_dir);
    }

    // Fallback: resolve from the current working directory.
    env::current_dir().expect(
        "Failed to determine project root: CARGO_MANIFEST_DIR is not set \
         and the current directory cannot be resolved.",
    )
}

// ---------------------------------------------------------------------------
// Cargo test subprocess execution
// ---------------------------------------------------------------------------

/// Execute `cargo test --release` with additional arguments, returning
/// captured output wrapped in [`common::BccOutput`].
///
/// The command is executed from the project root directory with the following
/// environment variables set to ensure deterministic, non-interactive behavior:
///
/// - `CI=true` — prevents interactive prompts and enables CI-friendly output.
/// - `RUST_BACKTRACE=1` — provides backtraces on test panics for diagnostics.
///
/// # Arguments
///
/// * `extra_args` — Additional CLI arguments appended after `cargo test --release`.
///   For example, `["--lib"]` to restrict to library unit tests,
///   or `["--lib", "common::"]` to filter by module.
///
/// # Panics
///
/// Panics if the `cargo` subprocess cannot be spawned (e.g., `cargo` not in PATH).
fn run_cargo_test(extra_args: &[&str]) -> common::BccOutput {
    let root = project_root();

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&root)
        .arg("test")
        .arg("--release")
        .args(extra_args)
        .env("CI", "true")
        .env("RUST_BACKTRACE", "1");

    let output = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Failed to spawn `cargo test --release {}` in '{}': {}",
            extra_args.join(" "),
            root.display(),
            e
        )
    });

    common::BccOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Execute library-only unit tests with an optional test name filter.
///
/// Invokes `cargo test --release --lib [filter]`, restricting execution to
/// unit tests defined within `src/`. The `--lib` flag is critical to prevent
/// recursive invocation of integration tests (which reside in `tests/` and
/// would include this very file, causing infinite recursion).
///
/// # Arguments
///
/// * `filter` — Optional test name substring filter. For example:
///   - `Some("common::")` runs only tests in the `common` module.
///   - `Some("frontend::")` runs only tests in the `frontend` module.
///   - `None` runs all library unit tests.
fn run_cargo_test_lib(filter: Option<&str>) -> common::BccOutput {
    let mut args: Vec<&str> = vec!["--lib"];

    if let Some(f) = filter {
        args.push(f);
    }

    run_cargo_test(&args)
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// Parse `cargo test` stdout/stderr into aggregated [`TestResults`].
///
/// Recognizes two line formats in the output:
///
/// 1. **Discovery lines:** `running N tests` or `running N test`
///    — contributes to [`TestResults::total_discovered`].
///
/// 2. **Summary lines:**
///    `test result: ok. N passed; N failed; N ignored; N measured; N filtered out; finished in X.XXs`
///    or `test result: FAILED. N passed; N failed; ...`
///    — contributes to the individual pass/fail/ignore/measured/filtered counters.
///
/// Multiple summary lines (from multiple test targets) are aggregated into a
/// single [`TestResults`] instance.
///
/// # Arguments
///
/// * `output` — The combined stdout and stderr text from a `cargo test` run.
fn parse_test_results(output: &str) -> TestResults {
    let mut results = TestResults::new();

    for line in output.lines() {
        let trimmed = line.trim();

        // ---------------------------------------------------------------
        // Parse "running N tests" or "running 1 test" discovery lines.
        // ---------------------------------------------------------------
        if trimmed.starts_with("running ")
            && (trimmed.ends_with(" tests") || trimmed.ends_with(" test"))
        {
            let count_str = trimmed
                .strip_prefix("running ")
                .unwrap_or("")
                .trim_end_matches(" tests")
                .trim_end_matches(" test");

            if let Ok(count) = count_str.parse::<u64>() {
                results.total_discovered += count;
            }
        }

        // ---------------------------------------------------------------
        // Parse "test result:" summary lines.
        //
        // Format examples:
        //   test result: ok. 42 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 1.23s
        //   test result: FAILED. 40 passed; 2 failed; 3 ignored; 0 measured; 0 filtered out; finished in 1.23s
        // ---------------------------------------------------------------
        if trimmed.starts_with("test result:") {
            // Skip past "ok. " or "FAILED. " to reach the counter fields.
            let after_status = match trimmed.find(". ") {
                Some(pos) => &trimmed[pos + 2..],
                None => continue,
            };

            // Each counter field is semicolon-separated: "N keyword"
            for part in after_status.split(';') {
                let part = part.trim();
                let tokens: Vec<&str> = part.split_whitespace().collect();

                if tokens.len() >= 2 {
                    if let Ok(count) = tokens[0].parse::<u64>() {
                        match tokens[1] {
                            "passed" => results.passed += count,
                            "failed" => results.failed += count,
                            "ignored" => results.ignored += count,
                            "measured" => results.measured += count,
                            keyword if keyword.starts_with("filtered") => {
                                results.filtered_out += count;
                            }
                            _ => {
                                // Unrecognized counter field (e.g., "finished")
                                // — silently skip.
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Assertion and validation helpers
// ---------------------------------------------------------------------------

/// Assert that all tests in the parsed results passed (zero failures).
///
/// Emits a summary line to stderr for CI log visibility, then panics with
/// a detailed diagnostic message if any tests failed. The raw `BccOutput`
/// is included in the panic message for debugging.
///
/// # Arguments
///
/// * `results`    — Parsed test results to validate.
/// * `label`      — Human-readable label for diagnostic messages (e.g.,
///   "Full test suite" or "common module tests").
/// * `bcc_output` — The raw subprocess output for inclusion in failure messages.
fn assert_all_tests_pass(
    results: &TestResults,
    label: &str,
    bcc_output: &common::BccOutput,
) {
    // Emit summary for CI log visibility (always printed, even on success).
    eprintln!(
        "[Checkpoint 3] {} — Test results: {}",
        label, results
    );

    // Hard gate: zero failures required (Section 0.7.5).
    assert!(
        results.is_all_pass(),
        "[Checkpoint 3] {} FAILED — {} test(s) failed out of {} run.\n\
         100% pass rate is a hard gate (Section 0.7.5).\n\
         \nstdout (tail):\n{}\n\nstderr (tail):\n{}",
        label,
        results.failed,
        results.total_run(),
        tail_str(&bcc_output.stdout, 3000),
        tail_str(&bcc_output.stderr, 3000)
    );
}

/// Validate the discovered test count and warn if it decreased.
///
/// If the `BCC_EXPECTED_TEST_COUNT` environment variable is set to a numeric
/// value, compares the actual discovered count against it and emits a warning
/// when the count is lower. A decrease may indicate that tests were deleted,
/// which could mask regressions.
///
/// Regardless of the environment variable, the discovered count is always
/// logged to stderr for traceability.
///
/// # Arguments
///
/// * `results` — Parsed test results containing the discovered count.
/// * `label`   — Human-readable label for diagnostic messages.
fn validate_test_count(results: &TestResults, label: &str) {
    eprintln!(
        "[Checkpoint 3] {} — Discovered {} test(s) ({} ran, {} filtered out)",
        label,
        results.total_discovered,
        results.total_run(),
        results.filtered_out
    );

    // Compare against an externally supplied baseline test count.
    if let Ok(expected_str) = env::var("BCC_EXPECTED_TEST_COUNT") {
        if let Ok(expected) = expected_str.parse::<u64>() {
            if results.total_discovered < expected {
                eprintln!(
                    "[Checkpoint 3] WARNING: {} — Test count decreased! \
                     Expected at least {} test(s), but only {} discovered. \
                     Possible test deletion — this could mask regressions \
                     (Section 0.7.5).",
                    label, expected, results.total_discovered
                );
            } else {
                eprintln!(
                    "[Checkpoint 3] {} — Test count OK ({} >= expected {})",
                    label, results.total_discovered, expected
                );
            }
        }
    }
}

/// Return the trailing portion of a string, capped at `max_bytes` bytes.
///
/// Handles UTF-8 boundary safety by walking forward to the nearest valid
/// character boundary when the byte offset falls inside a multi-byte sequence.
///
/// # Arguments
///
/// * `s`         — The source string.
/// * `max_bytes` — Maximum number of trailing bytes to return.
fn tail_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let start = s.len() - max_bytes;
    // Walk forward from the raw byte offset to the nearest char boundary.
    let mut boundary = start;
    while boundary < s.len() && !s.is_char_boundary(boundary) {
        boundary += 1;
    }
    &s[boundary..]
}

// ---------------------------------------------------------------------------
// Module-specific test runner
// ---------------------------------------------------------------------------

/// Run and validate unit tests for a specific BCC module.
///
/// Invokes `cargo test --release --lib <module_name>::` to execute only the
/// unit tests whose fully-qualified name contains the module prefix. For
/// example, a filter of `"common::"` matches test names such as
/// `common::fx_hash::tests::test_basic_hash`.
///
/// Validates:
/// 1. Test count is within expected range (if `BCC_EXPECTED_TEST_COUNT` is set).
/// 2. Zero test failures (100% pass rate hard gate).
///
/// # Arguments
///
/// * `module_name` — Module name used as the test filter prefix. Must be one
///   of: `"common"`, `"frontend"`, `"ir"`, `"passes"`, `"backend"`.
fn run_and_validate_module_tests(module_name: &str) {
    let filter = format!("{}::", module_name);
    let label = format!("{} module tests", module_name);

    eprintln!(
        "[Checkpoint 3] Running {} (filter: '{}')",
        label, filter
    );

    // Execute module-filtered library tests.
    let output = run_cargo_test_lib(Some(&filter));

    // Combine stdout and stderr for parsing (cargo may emit test results
    // on either stream depending on version and configuration).
    let combined_output = format!("{}\n{}", output.stdout, output.stderr);
    let results = parse_test_results(&combined_output);

    // Validate counts and assert pass rate.
    validate_test_count(&results, &label);
    assert_all_tests_pass(&results, &label, &output);

    eprintln!(
        "[Checkpoint 3] {} — PASSED ({} test(s) passed)",
        label, results.passed
    );
}

// ===========================================================================
// CHECKPOINT 3 — TEST FUNCTIONS
// ===========================================================================

/// Checkpoint 3 — Full library test suite execution.
///
/// Invokes `cargo test --release --lib` to run ALL unit tests across every
/// BCC module (common, frontend, ir, passes, backend). This is the primary
/// hard gate for Checkpoint 3.
///
/// # Assertions
///
/// 1. `cargo test` exits with code 0 (compilation succeeded and all tests pass).
/// 2. Parsed output reports zero failures.
/// 3. Test discovery count is validated against `BCC_EXPECTED_TEST_COUNT` if set.
///
/// # Sequential Gate (Section 0.7.5)
///
/// Failure of this test halts all forward progress — Checkpoint 4 cannot
/// begin until this test passes. This is a non-negotiable hard gate.
#[test]
fn test_cargo_test_all_pass() {
    eprintln!("[Checkpoint 3] ========================================");
    eprintln!("[Checkpoint 3] Full Internal Test Suite Execution");
    eprintln!("[Checkpoint 3] ========================================");

    // Use timed_execution from common utilities to measure wall-clock duration
    // of the complete test suite run.
    let (output, elapsed) = common::timed_execution(|| run_cargo_test_lib(None));

    eprintln!(
        "[Checkpoint 3] Full test suite completed in {:.2}s",
        elapsed.as_secs_f64()
    );

    // Gate 1: Assert the cargo test process exited successfully (exit code 0).
    // A non-zero exit indicates either compilation failure or test failures.
    assert!(
        output.success(),
        "[Checkpoint 3] `cargo test --release --lib` exited with code {:?}.\n\
         This is a hard gate — 100% pass rate required (Section 0.7.5).\n\
         \nstdout (tail):\n{}\n\nstderr (tail):\n{}",
        output.exit_code(),
        tail_str(&output.stdout, 3000),
        tail_str(&output.stderr, 3000)
    );

    // Gate 2: Parse test output and validate individual counters.
    let combined = format!("{}\n{}", output.stdout, output.stderr);
    let results = parse_test_results(&combined);

    validate_test_count(&results, "Full test suite");
    assert_all_tests_pass(&results, "Full test suite", &output);

    // Gate 3: Explicit zero-failure assertion (belt-and-suspenders check
    // alongside the exit code, in case cargo returns 0 with test failures
    // in unusual configurations).
    assert_eq!(
        results.failed, 0,
        "[Checkpoint 3] Full test suite reported {} failure(s). \
         100% pass rate is mandatory (Section 0.7.5).",
        results.failed
    );

    eprintln!("[Checkpoint 3] ========================================");
    eprintln!(
        "[Checkpoint 3] PASSED — {} test(s) in {:.2}s",
        results.passed,
        elapsed.as_secs_f64()
    );
    eprintln!("[Checkpoint 3] ========================================");
}

/// Checkpoint 3 — Common module tests (`src/common/`).
///
/// Runs unit tests for the infrastructure layer: FxHash, encoding (PUA/UTF-8),
/// types (dual type system), diagnostics, source map, string interning, target
/// definitions, temp files, long-double software math, and type builder.
#[test]
fn test_common_module_tests() {
    run_and_validate_module_tests("common");
}

/// Checkpoint 3 — Frontend module tests (`src/frontend/`).
///
/// Runs unit tests for the frontend pipeline: preprocessor (with paint-marker
/// recursion protection), lexer (PUA-aware scanning), parser (recursive descent
/// with GCC extension support), and semantic analyzer (type checking, builtins,
/// attribute handling, initializer analysis).
#[test]
fn test_frontend_module_tests() {
    run_and_validate_module_tests("frontend");
}

/// Checkpoint 3 — IR module tests (`src/ir/`).
///
/// Runs unit tests for the middle-end: IR instruction definitions, basic block
/// construction, function representation, module structure, IR lowering (alloca
/// insertion for all locals), mem2reg (SSA construction via Lengauer-Tarjan
/// dominator tree and dominance frontiers), and phi-node elimination.
#[test]
fn test_ir_module_tests() {
    run_and_validate_module_tests("ir");
}

/// Checkpoint 3 — Optimization passes module tests (`src/passes/`).
///
/// Runs unit tests for the optimization pipeline: constant folding and
/// propagation, dead code elimination, control-flow graph simplification
/// (unreachable block removal, branch threading), and the pass manager
/// scheduling framework.
#[test]
fn test_passes_module_tests() {
    run_and_validate_module_tests("passes");
}

/// Checkpoint 3 — Backend module tests (`src/backend/`).
///
/// Runs unit tests for the code generation layer: `ArchCodegen` trait
/// implementations, linear scan register allocator, common ELF writer,
/// linker infrastructure (symbol resolution, section merging, relocation
/// processing, dynamic linking), DWARF v4 generation, and all four
/// architecture-specific backends (x86-64, i686, AArch64, RISC-V 64)
/// including their built-in assemblers and linkers.
#[test]
fn test_backend_module_tests() {
    run_and_validate_module_tests("backend");
}

/// Checkpoint 3 — Regression gate for feature additions during kernel build.
///
/// This test serves as the regression detection mechanism mandated by
/// Section 0.7.5. It is designed to be re-run after ANY feature addition
/// during the kernel build phase (Section 5.4) to confirm that no
/// previously-passing test now fails.
///
/// # Regression Definition (Section 0.7.5)
///
/// Any test that passed before the current change and fails after is a
/// regression. Resolution is mandatory before proceeding to the next
/// checkpoint or continuing the kernel build.
///
/// # Environment Variables
///
/// - `BCC_REGRESSION_CHECK`: When set to `"1"`, activates enhanced
///   regression reporting with a detailed per-module breakdown. Each of
///   the five modules (common, frontend, ir, passes, backend) is tested
///   individually and failures are attributed to the specific module.
///
/// - `BCC_EXPECTED_TEST_COUNT`: When set to a numeric value, the test
///   warns if the discovered test count has decreased since the baseline
///   was established. A decrease may indicate test deletion that could
///   mask regressions.
///
/// # Standard vs Enhanced Mode
///
/// - **Standard mode** (default): Runs the full library test suite once
///   and asserts zero failures.
/// - **Enhanced mode** (`BCC_REGRESSION_CHECK=1`): Additionally runs
///   per-module test suites and reports individual module results for
///   faster regression localisation.
#[test]
fn test_regression_after_feature_addition() {
    let is_regression_mode = env::var("BCC_REGRESSION_CHECK")
        .map(|v| v == "1")
        .unwrap_or(false);

    if is_regression_mode {
        eprintln!("[Checkpoint 3] ========================================");
        eprintln!("[Checkpoint 3] REGRESSION CHECK MODE (post-feature-addition)");
        eprintln!("[Checkpoint 3] ========================================");
    } else {
        eprintln!("[Checkpoint 3] ========================================");
        eprintln!("[Checkpoint 3] Regression Gate (standard mode)");
        eprintln!("[Checkpoint 3] ========================================");
    }

    // Run the full library test suite with timing measurement.
    let (output, elapsed) = common::timed_execution(|| run_cargo_test_lib(None));

    eprintln!(
        "[Checkpoint 3] Regression gate completed in {:.2}s",
        elapsed.as_secs_f64()
    );

    // Parse results from combined output streams.
    let combined = format!("{}\n{}", output.stdout, output.stderr);
    let results = parse_test_results(&combined);

    // Validate test discovery count (detect possible test deletion).
    validate_test_count(&results, "Regression gate");

    // In enhanced regression mode, run per-module breakdown for precise
    // regression localisation.
    if is_regression_mode {
        eprintln!("[Checkpoint 3] --- Per-module regression breakdown ---");

        let modules = ["common", "frontend", "ir", "passes", "backend"];
        for module_name in &modules {
            let filter = format!("{}::", module_name);
            let module_output = run_cargo_test_lib(Some(&filter));

            let module_combined =
                format!("{}\n{}", module_output.stdout, module_output.stderr);
            let module_results = parse_test_results(&module_combined);

            eprintln!(
                "[Checkpoint 3]   {} — {}",
                module_name, module_results
            );

            // Each module must individually pass — attribute regressions to
            // the specific module for actionable diagnostics.
            assert!(
                module_results.is_all_pass(),
                "[Checkpoint 3] REGRESSION DETECTED in '{}' module!\n\
                 {} test(s) failed. Per Section 0.7.5, this regression must \
                 be resolved before proceeding.\n\
                 \nstdout (tail):\n{}\n\nstderr (tail):\n{}",
                module_name,
                module_results.failed,
                tail_str(&module_output.stdout, 3000),
                tail_str(&module_output.stderr, 3000)
            );
        }

        eprintln!("[Checkpoint 3] --- All modules passed individually ---");
    }

    // Assert the overall suite process exited successfully.
    assert!(
        output.success(),
        "[Checkpoint 3] REGRESSION DETECTED — `cargo test --release --lib` \
         exited with code {:?}.\n\
         Per Section 0.7.5: any test that previously passed and now fails \
         is a regression. Resolution is mandatory before proceeding.\n\
         \nstdout (tail):\n{}\n\nstderr (tail):\n{}",
        output.exit_code(),
        tail_str(&output.stdout, 3000),
        tail_str(&output.stderr, 3000)
    );

    // Explicit zero-failure count assertion.
    assert_eq!(
        results.failed, 0,
        "[Checkpoint 3] REGRESSION DETECTED — {} test(s) failed.\n\
         Per Section 0.7.5: resolution is mandatory before proceeding.",
        results.failed
    );

    eprintln!("[Checkpoint 3] ========================================");
    eprintln!(
        "[Checkpoint 3] Regression Gate: PASSED ({} test(s), {:.2}s)",
        results.passed,
        elapsed.as_secs_f64()
    );
    eprintln!("[Checkpoint 3] ========================================");
}
