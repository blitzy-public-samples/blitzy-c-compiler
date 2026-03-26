/*
 * tests/fixtures/dwarf/debug_test.c — DWARF Debug Information Test Source
 *
 * This file is the primary fixture for Checkpoint 4 (DWARF validation).
 * It is designed to exercise all aspects of DWARF v4 debug information
 * generation in BCC:
 *
 *   - DW_TAG_compile_unit  (compilation unit with producer, language, file)
 *   - DW_TAG_subprogram    (multiple functions: main, add, make_point, compute)
 *   - DW_TAG_formal_parameter (function parameters of various types)
 *   - DW_TAG_variable      (local variables: int, double, char*, struct)
 *   - DW_TAG_structure_type / DW_TAG_member (struct Point)
 *   - DW_TAG_base_type     (int, double, char)
 *   - .debug_line           (rich line-number program from multiple statements)
 *
 * Compilation:
 *   bcc -g -o debug_test debug_test.c        (with DWARF sections)
 *   bcc    -o debug_test_nodbg debug_test.c  (without DWARF — zero leakage)
 *
 * Validation (readelf):
 *   readelf -S debug_test | grep .debug_info
 *   readelf -S debug_test | grep .debug_abbrev
 *   readelf -S debug_test | grep .debug_line
 *   readelf -S debug_test | grep .debug_str
 *   readelf --debug-dump=info debug_test
 *   readelf --debug-dump=line debug_test
 *
 * GDB compatibility:
 *   gdb -batch -ex "break main" -ex "run" -ex "info locals" ./debug_test
 *
 * Runtime:
 *   ./debug_test  →  exit code 0, stdout contains "debug_test OK"
 */

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Struct definition: exercises DW_TAG_structure_type and DW_TAG_member */
/* ------------------------------------------------------------------ */
struct Point {
    int x;
    int y;
};

/* ------------------------------------------------------------------ */
/* add() — helper function                                             */
/*   DW_TAG_subprogram  for a non-main function                        */
/*   DW_TAG_formal_parameter for (int a, int b)                        */
/*   DW_TAG_variable for local 'result'                                */
/* ------------------------------------------------------------------ */
int add(int a, int b) {
    int result = a + b;
    return result;
}

/* ------------------------------------------------------------------ */
/* make_point() — struct-returning function                            */
/*   DW_TAG_subprogram  with struct return type                        */
/*   DW_TAG_formal_parameter for (int x, int y)                        */
/*   DW_TAG_variable for local 'p' of struct type                      */
/* ------------------------------------------------------------------ */
struct Point make_point(int x, int y) {
    struct Point p;
    p.x = x;
    p.y = y;
    return p;
}

/* ------------------------------------------------------------------ */
/* compute() — loop-bearing function                                   */
/*   Generates many .debug_line entries (for-loop iterations)          */
/*   DW_TAG_formal_parameter for (int n)                               */
/*   DW_TAG_variable for locals 'sum' and 'i'                         */
/* ------------------------------------------------------------------ */
int compute(int n) {
    int sum = 0;
    int i;
    for (i = 1; i <= n; i++) {
        sum += i;
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/* main() — program entry point                                        */
/*   DW_TAG_subprogram with DW_AT_name "main"                         */
/*   DW_TAG_variable for various typed locals:                         */
/*     int x, int y, double pi, const char *msg,                      */
/*     int sum, struct Point pt, int series                            */
/*   Calls to add/make_point/compute anchor DW_TAG_subprogram refs    */
/*   printf calls prevent dead-code elimination at -O0                 */
/*   Correctness assertions validate runtime behaviour                 */
/* ------------------------------------------------------------------ */
int main(void) {
    /* Local variables of different types for DW_TAG_variable testing */
    int x = 10;
    int y = 20;
    double pi = 3.14159;
    const char *msg = "DWARF test";

    /* Call helper functions to generate DW_TAG_subprogram references */
    int sum = add(x, y);
    struct Point pt = make_point(x, y);
    int series = compute(5);

    /* Use values to prevent dead code elimination */
    printf("sum=%d pt=(%d,%d) series=%d pi=%.2f msg=%s\n",
           sum, pt.x, pt.y, series, pi, msg);

    /* Verify correctness */
    if (sum != 30) {
        printf("FAIL: sum\n");
        return 1;
    }
    if (pt.x != 10 || pt.y != 20) {
        printf("FAIL: point\n");
        return 1;
    }
    if (series != 15) {
        printf("FAIL: series\n");
        return 1;
    }

    printf("debug_test OK\n");
    return 0;
}
