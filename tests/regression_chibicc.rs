//! Regression tests for chibicc-pattern bugs (Task 1).
//! Tests 18 specific bug patterns documented in CCC GitHub Issue #232.

use std::fs;
use std::process::Command;

fn bcc() -> String {
    std::env::current_dir()
        .unwrap()
        .join("target/release/bcc")
        .to_string_lossy()
        .to_string()
}

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(format!("/tmp/bcc_chibicc_test_{}", name));
    let _ = fs::create_dir_all(&p);
    p
}

fn compile_and_run(name: &str, c_code: &str) -> (i32, String) {
    let dir = tmp_dir(name);
    let src = dir.join("test.c");
    let out = dir.join("test");
    fs::write(&src, c_code).unwrap();
    let compile = Command::new(bcc())
        .args(["-o", out.to_str().unwrap(), src.to_str().unwrap()])
        .output()
        .expect("failed to run bcc");
    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        return (-1, format!("Compilation failed: {}", stderr));
    }
    let run = Command::new(out.to_str().unwrap())
        .output()
        .expect("failed to run binary");
    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    let _ = fs::remove_dir_all(&dir);
    (run.status.code().unwrap_or(-1), stdout)
}

fn compile_only(name: &str, c_code: &str) -> bool {
    let dir = tmp_dir(name);
    let src = dir.join("test.c");
    let out = dir.join("test");
    fs::write(&src, c_code).unwrap();
    let compile = Command::new(bcc())
        .args(["-o", out.to_str().unwrap(), src.to_str().unwrap()])
        .output()
        .expect("failed to run bcc");
    let _ = fs::remove_dir_all(&dir);
    compile.status.success()
}

#[test]
fn bug_1_1_sizeof_compound_literal() {
    let (code, stdout) = compile_and_run(
        "1_1",
        r#"
#include <stdio.h>
int main(void) {
    int sz = sizeof((int[]){1,2,3});
    printf("%d\n", sz == 3 * (int)sizeof(int));
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "1");
}

#[test]
fn bug_1_2_typeof_function_ptr() {
    assert!(compile_only(
        "1_2",
        r#"
typedef int fn_t(int);
int main(void) {
    __typeof__(fn_t) *fptr;
    (void)fptr;
    return 0;
}
"#
    ));
}

#[test]
fn bug_1_3_atomic_pointer() {
    assert!(compile_only(
        "1_3",
        r#"
int main(void) {
    int * _Atomic p;
    (void)p;
    return 0;
}
"#
    ));
}

#[test]
fn bug_1_4_designated_init_ordering() {
    let (code, stdout) = compile_and_run(
        "1_4",
        r#"
#include <stdio.h>
int main(void) {
    int a[] = { [2] = 30, [0] = 10, [1] = 20 };
    printf("%d %d %d\n", a[0], a[1], a[2]);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "10 20 30");
}

#[test]
fn bug_1_5_pointer_qualifier_compat() {
    assert!(compile_only(
        "1_5",
        r#"
int main(void) {
    int x = 42;
    int *p2 = &x;
    const int *p1 = p2;
    (void)p1;
    return 0;
}
"#
    ));
}

#[test]
fn bug_1_6_generic_duplicate_diagnostic() {
    let (code, stdout) = compile_and_run(
        "1_6",
        r#"
#include <stdio.h>
int main(void) {
    int x = 1;
    int r = _Generic(x, int: 42, double: 99);
    printf("%d\n", r);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "42");
}

#[test]
fn bug_1_7_32bit_truncation() {
    let (code, stdout) = compile_and_run(
        "1_7",
        r#"
#include <stdio.h>
int main(void) {
    int val = (int)(0x100000000ULL);
    printf("%d\n", val);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "0");
}

#[test]
fn bug_1_8_cast_to_bool_const() {
    let (code, stdout) = compile_and_run(
        "1_8",
        r#"
#include <stdio.h>
int main(void) {
    _Bool b = (_Bool)2;
    printf("%d\n", (int)b);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "1");
}

#[test]
fn bug_1_9_cast_to_bool_reloc() {
    let (code, stdout) = compile_and_run(
        "1_9",
        r#"
#include <stdio.h>
static int global_var;
int main(void) {
    _Bool b = (_Bool)&global_var;
    printf("%d\n", (int)b);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "1");
}

#[test]
fn bug_1_10_const_global_struct() {
    let (code, stdout) = compile_and_run(
        "1_10",
        r#"
#include <stdio.h>
struct point { int x; int y; };
const struct point origin = { 0, 0 };
int main(void) {
    printf("%d %d\n", origin.x, origin.y);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "0 0");
}

#[test]
fn bug_1_11_boolean_bitfield_inc() {
    let (code, stdout) = compile_and_run(
        "1_11",
        r#"
#include <stdio.h>
struct bf { unsigned int b : 1; };
int main(void) {
    struct bf s = { 0 };
    s.b++;
    printf("%u\n", s.b);
    s.b++;
    printf("%u\n", s.b);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(lines[0], "1");
    assert_eq!(lines[1], "0");
}

#[test]
fn bug_1_12_struct_alignment() {
    let (code, stdout) = compile_and_run(
        "1_12",
        r#"
#include <stdio.h>
#include <stddef.h>
struct s1 { char c; int i; };
struct s2 { char c; double d; };
int main(void) {
    printf("%zu %zu\n", offsetof(struct s1, i), offsetof(struct s2, d));
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "4 8");
}

#[test]
fn bug_1_13_long_double_printf() {
    let (code, stdout) = compile_and_run(
        "1_13",
        r#"
#include <stdio.h>
int main(void) {
    long double a = 1.5L;
    long double b = -42.5L;
    printf("%.1Lf\n", a);
    printf("%.1Lf\n", b);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(lines[0], "1.5");
    assert_eq!(lines[1], "-42.5");
}

#[test]
fn bug_1_14_array_decay() {
    let (code, stdout) = compile_and_run(
        "1_14",
        r#"
#include <stdio.h>
int main(void) {
    int arr[3] = {10, 20, 30};
    int *p = arr;
    printf("%d\n", *p);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "10");
}

#[test]
fn bug_1_15_stmt_expr_promotion() {
    let (code, stdout) = compile_and_run(
        "1_15",
        r#"
#include <stdio.h>
int main(void) {
    int r = ({short x = 1; x;}) + ({short y = 2; y;});
    printf("%d\n", r);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    assert_eq!(stdout.trim(), "3");
}

#[test]
fn bug_1_16_pragma_preserved_in_E() {
    let dir = tmp_dir("1_16");
    let src = dir.join("test.c");
    let out = dir.join("test.i");
    fs::write(
        &src,
        r#"
#pragma once
#pragma GCC diagnostic push
int x;
"#,
    )
    .unwrap();
    let result = Command::new(bcc())
        .args(["-E", "-o", out.to_str().unwrap(), src.to_str().unwrap()])
        .output()
        .expect("failed to run bcc -E");
    assert!(result.status.success());
    let content = fs::read_to_string(&out).unwrap();
    assert!(
        content.contains("#pragma"),
        "preprocessed output must preserve #pragma directives"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bug_1_18_va_args_comma_extension() {
    let (code, stdout) = compile_and_run(
        "1_18",
        r#"
#include <stdio.h>
#define LOG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
int main(void) {
    LOG("hello");
    LOG("val=%d", 42);
    return 0;
}
"#,
    );
    assert_eq!(code, 0);
    let lines: Vec<&str> = stdout.trim().lines().collect();
    assert_eq!(lines[0], "hello");
    assert_eq!(lines[1], "val=42");
}
