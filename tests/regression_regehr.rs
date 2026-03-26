//! Regression tests for Regehr fuzzing bug classes found in CCC.
//! These test classes of miscompilation bugs discovered by Prof. John Regehr
//! via Csmith/YARPGen fuzzing of CCC (Claude's C Compiler).
//! BCC must produce correct output for all of these patterns.

use std::path::Path;
use std::process::Command;

fn bcc_path() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    format!("{}/target/release/bcc", manifest)
}

fn compile_and_run(c_code: &str, test_name: &str) -> String {
    let src = format!("/tmp/regehr_{}.c", test_name);
    let bin = format!("/tmp/regehr_{}", test_name);
    std::fs::write(&src, c_code).unwrap();

    let compile = Command::new(bcc_path())
        .args(&["-o", &bin, &src])
        .output()
        .expect("Failed to run bcc");
    assert!(
        compile.status.success(),
        "Compilation failed for {}: {}",
        test_name,
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin).output().expect("Failed to run binary");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&bin);

    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(
        run.status.success(),
        "Runtime failure for {}: exit={}, stdout={}, stderr={}",
        test_name,
        run.status,
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );
    stdout
}

// === Bug 2.1: IR narrowing for 64-bit bitwise ops ===

#[test]
fn regehr_2_1_64bit_bitwise_or() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned long long f(void) { return 0xFFFFFFFF00000000ULL; }
__attribute__((noinline)) unsigned long long g(void) { return 0x00000000FFFFFFFFULL; }
int main(void) {
    unsigned long long r = f() | g();
    printf("%llu\n", r == 0xFFFFFFFFFFFFFFFFULL);
    return 0;
}
"#,
        "2_1_or",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_1_64bit_narrow_trap() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    unsigned long long a = 0x0000000100000000ULL;
    unsigned long long b = 0x0000000200000000ULL;
    unsigned long long r = a | b;
    printf("%llu\n", r == 0x0000000300000000ULL);
    return 0;
}
"#,
        "2_1_narrow",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_1_64bit_rotate() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline))
unsigned long long rot64(unsigned long long x, int n) {
    return (x << n) | (x >> (64 - n));
}
int main(void) {
    unsigned long long r = rot64(0x0123456789ABCDEFULL, 8);
    printf("%llu\n", r == 0x23456789ABCDEF01ULL);
    return 0;
}
"#,
        "2_1_rot",
    );
    assert_eq!(out.trim(), "1");
}

// === Bug 2.2: Unsigned negation in constant folding ===

#[test]
fn regehr_2_2_unsigned_negation_const() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    unsigned int x = -8u;
    printf("%u\n", x);
    return 0;
}
"#,
        "2_2_const",
    );
    assert_eq!(out.trim(), "4294967288"); // 0xFFFFFFF8
}

#[test]
fn regehr_2_2_unsigned_negation_runtime() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int get8(void) { return 8u; }
int main(void) {
    unsigned int x = -get8();
    printf("%u\n", x);
    return 0;
}
"#,
        "2_2_rt",
    );
    assert_eq!(out.trim(), "4294967288");
}

#[test]
fn regehr_2_2_neg_uint_max() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    unsigned int x = -(0xFFFFFFFFu);
    printf("%u\n", x);
    return 0;
}
"#,
        "2_2_max",
    );
    assert_eq!(out.trim(), "1");
}

// === Bug 2.3: Peephole cmp+branch fusion ===

#[test]
fn regehr_2_3_cmp_across_call() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
volatile int g = 0;
__attribute__((noinline)) int clobber(int a, int b) { g = 1; return a - b; }
int main(void) {
    volatile int x = 10, y = 20;
    int result = (x < y);
    clobber(100, 200);
    printf("%d\n", result);
    return 0;
}
"#,
        "2_3_call",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_3_interleaved_cmps() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    volatile int a=1, b=2, c=3, d=4;
    int r1 = (a < b);
    int r2 = (c < d);
    printf("%d %d\n", r1, r2);
    return 0;
}
"#,
        "2_3_interleaved",
    );
    assert_eq!(out.trim(), "1 1");
}

// === Bug 2.4: narrow_cmps cast stripping ===

#[test]
fn regehr_2_4_sign_change_cmp() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned char get_uc(unsigned char x) { return x; }
int main(void) {
    unsigned char x = get_uc(200);
    signed char y = (signed char)x;
    printf("%d\n", y < 0);
    return 0;
}
"#,
        "2_4_sign",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_4_sub_int_width() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    int a = 300, b = 200;
    printf("%d\n", a > b);
    return 0;
}
"#,
        "2_4_subint",
    );
    assert_eq!(out.trim(), "1");
}

// === Bug 2.5: Shift narrowing ===

#[test]
fn regehr_2_5_shift_left_gt32() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned long long get1(void) { return 1ULL; }
int main(void) {
    unsigned long long r = get1() << 48;
    printf("%llu\n", r == 0x1000000000000ULL);
    return 0;
}
"#,
        "2_5_gt32",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_5_signed_right_shift() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    long long x = -0x100000000LL;
    long long r = x >> 16;
    printf("%lld\n", r);
    return 0;
}
"#,
        "2_5_signed",
    );
    assert_eq!(out.trim(), "-65536");
}

// === Bug 2.6: Usual arithmetic conversions ===

#[test]
fn regehr_2_6_mixed_sign_compare() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
__attribute__((noinline)) int gets(int x) { return x; }
int main(void) {
    unsigned int u = getu(1);
    int s = gets(-1);
    // u < s is unsigned comparison: 1 < 0xFFFFFFFF → true
    printf("%d\n", u < s);
    return 0;
}
"#,
        "2_6_cmp",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_6_mixed_add() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
__attribute__((noinline)) int gets(int x) { return x; }
int main(void) {
    unsigned int u = getu(1);
    int s = gets(-2);
    unsigned int r = u + s;
    printf("%u\n", r);
    return 0;
}
"#,
        "2_6_add",
    );
    assert_eq!(out.trim(), "4294967295"); // 0xFFFFFFFF
}

// === Bug 2.7: Explicit cast sign-extension ===

#[test]
fn regehr_2_7_int_to_long_sign_ext() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) int get_intmin(void) { return (int)0x80000000; }
int main(void) {
    int x = get_intmin();
    long long y = (long long)x;
    printf("%lld\n", y);
    return 0;
}
"#,
        "2_7_intmin",
    );
    assert_eq!(out.trim(), "-2147483648");
}

#[test]
fn regehr_2_7_uint_to_ll_zero_ext() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    unsigned int x = 0x80000000u;
    long long y = (long long)x;
    printf("%lld\n", y);
    return 0;
}
"#,
        "2_7_uint",
    );
    assert_eq!(out.trim(), "2147483648");
}

// === Bug 2.8: Narrowing optimization for And/Shl ===

#[test]
fn regehr_2_8_and_high_bits() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) long long getll(long long x) { return x; }
int main(void) {
    long long x = getll(0x100000000LL);
    long long m = getll(0x1FFFFFFFFLL);
    long long r = x & m;
    printf("%lld\n", r == 0x100000000LL);
    return 0;
}
"#,
        "2_8_and",
    );
    assert_eq!(out.trim(), "1");
}

#[test]
fn regehr_2_8_shl_cast() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
int main(void) {
    int x = -1;
    long long r = ((long long)x) << 1;
    printf("%lld\n", r);
    return 0;
}
"#,
        "2_8_shl",
    );
    assert_eq!(out.trim(), "-2");
}

// === Bug 2.9: U32→I32 same-width cast ===

#[test]
fn regehr_2_9_u32_to_i32() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
int main(void) {
    unsigned int u = getu(0x80000000u);
    int s = (int)u;
    printf("%d\n", s < 0);
    printf("%d\n", s);
    return 0;
}
"#,
        "2_9_cast",
    );
    let lines: Vec<&str> = out.trim().lines().collect();
    assert_eq!(lines[0], "1");
    assert_eq!(lines[1], "-2147483648");
}

#[test]
fn regehr_2_9_sign_ext_after_cast() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
int main(void) {
    unsigned int u = getu(0xFFFF0000u);
    int s = (int)u;
    long long ll = (long long)s;
    printf("%lld\n", ll);
    return 0;
}
"#,
        "2_9_ext",
    );
    assert_eq!(out.trim(), "-65536");
}

// === Bug 2.10: div_by_const range analysis ===

#[test]
fn regehr_2_10_large_unsigned_div() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
int main(void) {
    unsigned int r = getu(0xFFFFFFFFu) / 3;
    printf("%u\n", r);
    return 0;
}
"#,
        "2_10_large",
    );
    assert_eq!(out.trim(), "1431655765");
}

#[test]
fn regehr_2_10_above_intmax() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
int main(void) {
    unsigned int r = getu(0x80000000u) / 7;
    printf("%u\n", r);
    return 0;
}
"#,
        "2_10_intmax",
    );
    assert_eq!(out.trim(), "306783378");
}

#[test]
fn regehr_2_10_unsigned_mod() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) unsigned int getu(unsigned int x) { return x; }
int main(void) {
    unsigned int r = getu(0xFFFFFFFFu) % 5;
    printf("%u\n", r);
    return 0;
}
"#,
        "2_10_mod",
    );
    assert_eq!(out.trim(), "0");
}

// === Bug 2.11: cfg_simplify constant propagation through Cast ===

#[test]
fn regehr_2_11_narrow_widen() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) int geti(int x) { return x; }
int main(void) {
    int x = geti(128);
    signed char n = (signed char)x;
    int w = (int)n;
    printf("%d\n", w);
    return 0;
}
"#,
        "2_11_narrow",
    );
    assert_eq!(out.trim(), "-128");
}

#[test]
fn regehr_2_11_truncation_chain() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) int geti(int x) { return x; }
int main(void) {
    int x = geti(300);
    unsigned char y = (unsigned char)x;
    int z = y + 1;
    printf("%d\n", z);
    return 0;
}
"#,
        "2_11_trunc",
    );
    assert_eq!(out.trim(), "45");
}

#[test]
fn regehr_2_11_cast_chain() {
    let out = compile_and_run(
        r#"
#include <stdio.h>
__attribute__((noinline)) int cast_chain(int x) {
    unsigned char a = (unsigned char)x;
    signed char b = (signed char)a;
    unsigned short c = (unsigned short)b;
    int d = (int)c;
    return d;
}
int main(void) {
    printf("%d\n", cast_chain(255));
    printf("%d\n", cast_chain(128));
    printf("%d\n", cast_chain(127));
    return 0;
}
"#,
        "2_11_chain",
    );
    let lines: Vec<&str> = out.trim().lines().collect();
    assert_eq!(lines[0], "65535");
    assert_eq!(lines[1], "65408");
    assert_eq!(lines[2], "127");
}

// === Full combined test ===

#[test]
fn regehr_full_regression_suite() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = format!("{}/tests/fixtures/regehr_regression.c", manifest);
    if !Path::new(&src).exists() {
        panic!("Missing test fixture: {}", src);
    }

    let bin = "/tmp/regehr_full_suite";
    let compile = Command::new(bcc_path())
        .args(&["-o", bin, &src])
        .output()
        .expect("Failed to run bcc");
    assert!(
        compile.status.success(),
        "Compilation failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(bin).output().expect("Failed to run");
    let _ = std::fs::remove_file(bin);
    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    assert!(
        run.status.success(),
        "Runtime failure: exit={}, stdout={}, stderr={}",
        run.status,
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        stdout.contains("ALL 58 Regehr regression tests PASSED"),
        "Not all tests passed: {}",
        stdout
    );
}
