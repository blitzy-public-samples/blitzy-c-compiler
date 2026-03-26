//! Regression tests for bugs found via Csmith/YARPGen fuzzing campaigns (Task 7).
//!
//! Each test corresponds to a specific bug class discovered during fuzzing.
//! Tests compile C fixtures with BCC, run the resulting binaries, and verify
//! correct output matches GCC behavior.
//!
//! Bug classes discovered:
//! - Bug #1: Empty struct member (bare ';' inside struct body)
//! - Bug #2: typeof on array subscript expression
//! - Bug #3: typeof on statement expressions
//! - Bug E: struct_load_source Store R10 clobber for 2-eightbyte structs
//! - Bug F: usual_arithmetic_conversion LongLong vs ULong on LP64
//! - Bug G: aggregate rvalue pointer-vs-data confusion in assignment/init

mod common;

use common::{assert_exit_success, compile_and_run};

// ===========================================================================
// Bug #1: Empty struct member (bare ';' inside struct body)
// ===========================================================================

#[test]
fn test_fuzz_empty_struct_member_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/empty_struct_member.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("empty_struct_member OK"),
        "empty_struct_member test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_empty_struct_member_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/empty_struct_member.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("empty_struct_member OK"),
        "empty_struct_member test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug #2: typeof on array subscript expression
// ===========================================================================

#[test]
fn test_fuzz_typeof_array_subscript_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_array_subscript.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_array_subscript OK"),
        "typeof_array_subscript test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_typeof_array_subscript_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_array_subscript.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_array_subscript OK"),
        "typeof_array_subscript test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug #3: typeof on statement expressions
// ===========================================================================

#[test]
fn test_fuzz_typeof_stmt_expr_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_stmt_expr.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_stmt_expr OK"),
        "typeof_stmt_expr test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_typeof_stmt_expr_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/typeof_stmt_expr.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("typeof_stmt_expr OK"),
        "typeof_stmt_expr test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug E: struct_load_source Store R10 clobber for 2-eightbyte structs
// When storing a 2-eightbyte struct loaded from a global, the Store handler
// used R10 as scratch but also loaded into R10, clobbering the second
// eightbyte with stale data.
// ===========================================================================

#[test]
fn test_fuzz_struct_store_r10_clobber_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/struct_store_r10_clobber.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("struct_store_r10_clobber OK"),
        "struct_store_r10_clobber test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_struct_store_r10_clobber_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/struct_store_r10_clobber.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("struct_store_r10_clobber OK"),
        "struct_store_r10_clobber test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug F: usual_arithmetic_conversion LongLong vs ULong on LP64
// On LP64 targets where unsigned long and signed long long have the same size
// (64-bit), the usual arithmetic conversions must convert both to unsigned
// long long per C11 6.3.1.8 step 4d.
// ===========================================================================

#[test]
fn test_fuzz_arith_conv_longlong_ulong_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/arith_conv_longlong_ulong.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("arith_conv_longlong_ulong OK"),
        "arith_conv_longlong_ulong test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_arith_conv_longlong_ulong_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/arith_conv_longlong_ulong.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("arith_conv_longlong_ulong OK"),
        "arith_conv_longlong_ulong test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===========================================================================
// Bug G: Aggregate rvalue pointer-vs-data confusion
// BCC's IR has a dual representation for aggregate TypedValues: assignment
// expressions return a pointer to the LHS (an address), while function calls
// and va_arg return the struct DATA itself (held in backend temp slots).
// Both cases use struct CType (not Pointer CType). The lowering must use
// AST-based checks (not CType-based) to distinguish the two cases.
// ===========================================================================

#[test]
fn test_fuzz_aggregate_rvalue_assign_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/aggregate_rvalue_assign.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("aggregate_rvalue_assign OK"),
        "aggregate_rvalue_assign test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_aggregate_rvalue_assign_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/aggregate_rvalue_assign.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("aggregate_rvalue_assign OK"),
        "aggregate_rvalue_assign test failed (aarch64). stdout: {}",
        stdout
    );
}

// ==========================================================================
// Bug H: Global pointer initializer through union→struct→field chain
// --------------------------------------------------------------------------
// When a global pointer is initialized to &union_var[i].struct_field.member,
// BCC must emit a relocation addend that includes the byte offset of the
// member within the inner struct.  Previously, infer_struct_type_from_expr()
// in decl_lowering.rs only handled CType::Struct in the MemberAccess branch,
// not CType::Union, so it returned None for union→struct member chains.
// This caused evaluate_member_field_offset() to return None, which was
// silently converted to offset 0 via unwrap_or(0).
//
// Found via Csmith fuzzing: mismatch_44, mismatch_112, mismatch_216 all
// contained global pointer initializers through union→struct→field chains.
// ==========================================================================

#[test]
fn test_fuzz_global_ptr_union_member_offset_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/global_ptr_union_member_offset.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("OK: global pointer union member offset correct"),
        "global_ptr_union_member_offset test failed (x86-64). stdout: {}",
        stdout
    );
}

#[test]
fn test_fuzz_global_ptr_union_member_offset_aarch64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/global_ptr_union_member_offset.c",
        "aarch64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("OK: global pointer union member offset correct"),
        "global_ptr_union_member_offset test failed (aarch64). stdout: {}",
        stdout
    );
}

// ===================================================================
// Bug I: Multi-dim array global initializer brace-elision symbol size
// ===================================================================
#[test]
fn test_bug_i_array_brace_elision_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/bug_i_array_brace_elision.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("bug_i_array_brace_elision OK"),
        "bug_i test failed (x86-64). stdout: {}",
        stdout
    );
}

// ===================================================================
// Bug J: Sub-register truncation for promoted I8/I16 arithmetic
// ===================================================================
#[test]
fn test_bug_j_uint8_shortcircuit_x86_64() {
    let output = compile_and_run(
        "tests/fixtures/fuzz_regressions/bug_j_uint8_shortcircuit.c",
        "x86-64",
    );
    assert_exit_success(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("bug_j_uint8_shortcircuit OK"),
        "bug_j test failed (x86-64). stdout: {}",
        stdout
    );
}
