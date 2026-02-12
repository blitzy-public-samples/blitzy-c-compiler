/*
 * cet.c — Intel CET/IBT (Control-flow Enforcement Technology / Indirect Branch
 *          Tracking) test fixture for BCC Checkpoint 5 security validation.
 *
 * Purpose:
 *   Validates that BCC emits the `endbr64` instruction (opcode 0xf3 0x0f 0x1e 0xfa)
 *   at every function entry point and indirect branch target when compiled with
 *   the -fcf-protection flag targeting x86-64.
 *
 * Compilation:
 *   ./bcc -fcf-protection --target=x86-64 -c -o cet.o cet.c
 *
 * Verification:
 *   objdump -d cet.o | grep endbr64
 *   — Every function (handler_a, handler_b, handler_c, dispatch_handler,
 *     call_via_local, main) must begin with an endbr64 instruction.
 *
 * Indirect call patterns exercised:
 *   1. Call through a global function pointer array element
 *   2. Call through a local function pointer variable
 *   3. Call through a function pointer passed as a parameter
 *   4. Call through a function pointer returned from another function
 *
 * Security mitigations are x86-64 only (Section 0.6.2).
 */

#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Function pointer type used throughout the test.
 * ------------------------------------------------------------------------- */
typedef int (*handler_fn)(int);

/* ---------------------------------------------------------------------------
 * Indirect branch target functions.
 * Each of these MUST start with `endbr64` when -fcf-protection is active,
 * because they are reachable through indirect calls (function pointers).
 * ------------------------------------------------------------------------- */

int handler_a(int x) {
    return x + 1;
}

int handler_b(int x) {
    return x * 2;
}

int handler_c(int x) {
    return x - 3;
}

/* ---------------------------------------------------------------------------
 * Global function pointer array — exercises indirect call via array indexing.
 * ------------------------------------------------------------------------- */
handler_fn handlers[] = { handler_a, handler_b, handler_c };

/* ---------------------------------------------------------------------------
 * dispatch_handler — indirect call through the global function pointer array.
 *
 * The call `handlers[index](value)` is an indirect branch; the target function
 * must contain `endbr64` for CET/IBT validation.
 * ------------------------------------------------------------------------- */
int dispatch_handler(int index, int value) {
    if (index < 0 || index >= 3) {
        return -1;
    }
    int result = handlers[index](value);
    return result;
}

/* ---------------------------------------------------------------------------
 * call_via_local — indirect call through a local function pointer variable.
 *
 * Assigns `handler_b` to a local `handler_fn` variable, then calls through it.
 * This pattern exercises a different code-generation path for indirect calls.
 * ------------------------------------------------------------------------- */
int call_via_local(void) {
    handler_fn fn = handler_b;
    int r = fn(42);
    return r;
}

/* ---------------------------------------------------------------------------
 * call_via_param — indirect call through a function pointer parameter.
 *
 * The caller passes a function pointer; this function invokes it indirectly.
 * ------------------------------------------------------------------------- */
int call_via_param(handler_fn fp, int arg) {
    return fp(arg);
}

/* ---------------------------------------------------------------------------
 * select_handler — returns a function pointer, exercising indirect call
 * through a returned pointer at the call site.
 * ------------------------------------------------------------------------- */
handler_fn select_handler(int choice) {
    if (choice == 0) {
        return handler_a;
    } else if (choice == 1) {
        return handler_b;
    }
    return handler_c;
}

/* ---------------------------------------------------------------------------
 * main — exercises every indirect call pattern and verifies correctness.
 *
 * Returns 0 on success (all assertions pass), non-zero on failure.
 * ------------------------------------------------------------------------- */
int main(void) {
    int failed = 0;

    /* Pattern 1: Indirect call through global function pointer array. */
    int r1 = dispatch_handler(0, 10);  /* handler_a(10) => 11 */
    if (r1 != 11) {
        printf("FAIL: dispatch_handler(0, 10) = %d, expected 11\n", r1);
        failed = 1;
    }

    int r2 = dispatch_handler(1, 10);  /* handler_b(10) => 20 */
    if (r2 != 20) {
        printf("FAIL: dispatch_handler(1, 10) = %d, expected 20\n", r2);
        failed = 1;
    }

    int r3 = dispatch_handler(2, 10);  /* handler_c(10) => 7 */
    if (r3 != 7) {
        printf("FAIL: dispatch_handler(2, 10) = %d, expected 7\n", r3);
        failed = 1;
    }

    /* Pattern 2: Indirect call through a local function pointer variable. */
    int r4 = call_via_local();  /* handler_b(42) => 84 */
    if (r4 != 84) {
        printf("FAIL: call_via_local() = %d, expected 84\n", r4);
        failed = 1;
    }

    /* Pattern 3: Indirect call through a function pointer parameter. */
    int r5 = call_via_param(handler_a, 100);  /* handler_a(100) => 101 */
    if (r5 != 101) {
        printf("FAIL: call_via_param(handler_a, 100) = %d, expected 101\n", r5);
        failed = 1;
    }

    int r6 = call_via_param(handler_c, 5);  /* handler_c(5) => 2 */
    if (r6 != 2) {
        printf("FAIL: call_via_param(handler_c, 5) = %d, expected 2\n", r6);
        failed = 1;
    }

    /* Pattern 4: Indirect call through a returned function pointer. */
    handler_fn selected = select_handler(1);
    int r7 = selected(7);  /* handler_b(7) => 14 */
    if (r7 != 14) {
        printf("FAIL: select_handler(1) -> fn(7) = %d, expected 14\n", r7);
        failed = 1;
    }

    selected = select_handler(2);
    int r8 = selected(10);  /* handler_c(10) => 7 */
    if (r8 != 7) {
        printf("FAIL: select_handler(2) -> fn(10) = %d, expected 7\n", r8);
        failed = 1;
    }

    /* Boundary: out-of-range index returns -1. */
    int r9 = dispatch_handler(5, 10);
    if (r9 != -1) {
        printf("FAIL: dispatch_handler(5, 10) = %d, expected -1\n", r9);
        failed = 1;
    }

    if (!failed) {
        printf("CET/IBT test passed\n");
    }

    return failed;
}
