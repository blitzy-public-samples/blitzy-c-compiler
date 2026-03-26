//! Regression tests for chibicc-pattern bugs (CCC GitHub Issue #232)
//! and Regehr fuzzing bug classes found via Csmith/YARPGen.
//!
//! These tests compile C fixtures with BCC, run the resulting binaries,
//! and verify correct output — ensuring BCC does not suffer from the
//! same classes of bugs found in other C compiler implementations.

mod common;

use common::{assert_exit_success, compile_and_run};

// ===========================================================================
// chibicc-pattern bug tests (Task 1)
// ===========================================================================

#[test]
fn test_chibicc_bugs_x86_64() {
    let output = compile_and_run("tests/fixtures/chibicc_bugs/all_bugs.c", "x86-64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("chibicc_bugs OK"),
        "chibicc bugs test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_chibicc_bugs_aarch64() {
    let output = compile_and_run("tests/fixtures/chibicc_bugs/all_bugs.c", "aarch64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("chibicc_bugs OK"),
        "chibicc bugs test failed (aarch64). stdout: {}",
        stdout
    );
}

#[test]
fn test_chibicc_bugs_riscv64() {
    let output = compile_and_run("tests/fixtures/chibicc_bugs/all_bugs.c", "riscv64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("chibicc_bugs OK"),
        "chibicc bugs test failed (riscv64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Regehr fuzzing bug class tests (Task 2)
// ===========================================================================

#[test]
fn test_regehr_bugs_x86_64() {
    let output = compile_and_run("tests/fixtures/regehr_bugs/all_bugs.c", "x86-64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("regehr_bugs OK"),
        "Regehr bugs test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_regehr_bugs_aarch64() {
    let output = compile_and_run("tests/fixtures/regehr_bugs/all_bugs.c", "aarch64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("regehr_bugs OK"),
        "Regehr bugs test failed (aarch64). stdout: {}",
        stdout
    );
}

#[test]
fn test_regehr_bugs_riscv64() {
    let output = compile_and_run("tests/fixtures/regehr_bugs/all_bugs.c", "riscv64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("regehr_bugs OK"),
        "Regehr bugs test failed (riscv64). stdout: {}",
        stdout
    );
}
