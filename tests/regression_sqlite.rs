//! Regression tests for the SQLite runtime segfault (Task 3).
//!
//! Two root-cause bugs were found and fixed:
//!
//!  1. **Stack alignment** — an odd number of callee-saved register pushes
//!     left RSP mis-aligned for SSE instructions in libc (e.g. `movaps`).
//!     Fixed in `src/backend/generation.rs`: after register allocation
//!     finalises the callee-saved list, pad the frame to restore 16-byte
//!     alignment.
//!
//!  2. **Static initialiser: `&struct.array_member[idx]`** — the IR lowering
//!     emitted NULL for addresses such as `&g.items[1]` inside static
//!     initialisers because `evaluate_address_of_subscript` did not handle
//!     `MemberAccess` base expressions.
//!     Fixed in `src/ir/lowering/decl_lowering.rs`: added MemberAccess
//!     handling in both `evaluate_address_of_subscript` and
//!     `evaluate_constant_expr`.
//!
//! The fixture `tests/fixtures/sqlite_regression.c` exercises both patterns
//! with minimal reproducers.

mod common;

use common::{assert_exit_success, compile_and_run};

/// Test SQLite regression bugs on x86-64 (primary target).
#[test]
fn test_sqlite_regression_x86_64() {
    let output = compile_and_run("tests/fixtures/sqlite_regression.c", "x86-64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("sqlite_regression OK"),
        "SQLite regression test failed (x86-64).\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Test SQLite regression bugs on AArch64 (cross-architecture validation).
#[test]
fn test_sqlite_regression_aarch64() {
    let output = compile_and_run("tests/fixtures/sqlite_regression.c", "aarch64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("sqlite_regression OK"),
        "SQLite regression test failed (aarch64).\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Test SQLite regression bugs on RISC-V 64 (cross-architecture validation).
#[test]
fn test_sqlite_regression_riscv64() {
    let output = compile_and_run("tests/fixtures/sqlite_regression.c", "riscv64");
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("sqlite_regression OK"),
        "SQLite regression test failed (riscv64).\nstdout: {}\nstderr: {}",
        stdout,
        String::from_utf8_lossy(&output.stderr),
    );
}
