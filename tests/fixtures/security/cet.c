/*
 * tests/fixtures/security/cet.c — CET/IBT Test Fixture
 *
 * Validates Intel CET (Control-flow Enforcement Technology) / IBT (Indirect
 * Branch Tracking) code generation in BCC.
 *
 * When compiled with:   -fcf-protection --target=x86-64
 *   - Every function entry point MUST begin with an endbr64 instruction
 *     (machine encoding: f3 0f 1e fa, 4 bytes).
 *   - Indirect branch targets must also have endbr64.
 *
 * When compiled WITHOUT -fcf-protection:
 *   - endbr64 MUST NOT appear at function entries.
 *
 * Consumed by: tests/checkpoint5_security.rs
 *   which inspects objdump -d disassembly output.
 *
 * All functions are non-static so they appear as visible symbols in the
 * binary, making objdump disassembly inspection straightforward.
 */

#include <stdio.h>

/* Multiple functions to verify endbr64 at each entry point */

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int negate(int x) {
    return -x;
}

/* Function called through a pointer — its entry must also have endbr64
   as it may be an indirect branch target */
int apply(int (*op)(int, int), int x, int y) {
    return op(x, y);
}

int main(void) {
    /* Direct calls */
    int sum = add(3, 4);
    if (sum != 7) { printf("FAIL: add=%d\n", sum); return 1; }

    int prod = multiply(5, 6);
    if (prod != 30) { printf("FAIL: multiply=%d\n", prod); return 1; }

    int neg = negate(42);
    if (neg != -42) { printf("FAIL: negate=%d\n", neg); return 1; }

    /* Indirect call through function pointer */
    int result = apply(add, 10, 20);
    if (result != 30) { printf("FAIL: apply(add)=%d\n", result); return 1; }

    result = apply(multiply, 3, 7);
    if (result != 21) { printf("FAIL: apply(multiply)=%d\n", result); return 1; }

    printf("cet OK\n");
    return 0;
}
