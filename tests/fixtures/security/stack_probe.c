/*
 * stack_probe.c — Stack guard page probe test fixture for BCC Checkpoint 5
 *
 * Purpose:
 *   Validates that the BCC x86-64 backend emits a probe loop before the
 *   stack pointer adjustment for any function whose stack frame exceeds
 *   4096 bytes (one page). The probe loop iterates in page-sized (4096-byte)
 *   increments, touching each page to trigger guard page faults and prevent
 *   stack clash attacks.
 *
 * Usage:
 *   ./bcc --target=x86-64 -c -o stack_probe.o tests/fixtures/security/stack_probe.c
 *   objdump -d stack_probe.o   # Inspect for probe loops
 *
 * Expected behaviour (objdump -d):
 *   Functions f(), large_frame_4097(), large_frame_16384(), and use_stack()
 *   MUST contain a probe loop (page-sized decrement + store) BEFORE the
 *   final stack pointer adjustment.
 *
 *   Function large_frame_exact_page() allocates exactly 4096 bytes, which
 *   is the borderline case — it does NOT exceed the threshold and therefore
 *   a probe loop is NOT strictly required.
 *
 * Reference:
 *   Section 0.1.2 User Example — "disassembly MUST show a probe loop before
 *   the stack pointer adjustment"
 *   Section 0.5.1 Group 6 — src/backend/x86_64/security.rs implements stack
 *   probe loop for frames exceeding 4096 bytes.
 *   Section 0.6.2 — Security mitigations are x86-64 only.
 */

#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Canonical test case from Section 0.1.2 User Example.
 * 8192 bytes > 4096 byte page threshold  =>  probe loop REQUIRED
 * ------------------------------------------------------------------------- */
void f(void) {
    char buf[8192];
    buf[0] = 1;
}

/* ---------------------------------------------------------------------------
 * Just over the threshold: 4097 > 4096  =>  probe loop REQUIRED
 * ------------------------------------------------------------------------- */
void large_frame_4097(void) {
    char buf[4097];
    buf[0] = 'a';
}

/* ---------------------------------------------------------------------------
 * Four full pages: 16384 > 4096  =>  probe loop REQUIRED
 * The loop must iterate at least 4 times (16384 / 4096 = 4 pages).
 * ------------------------------------------------------------------------- */
void large_frame_16384(void) {
    char buf[16384];
    buf[0] = 'b';
}

/* ---------------------------------------------------------------------------
 * Exactly at the page boundary: 4096 == 4096  =>  probe loop NOT required
 * This is the borderline case; the frame does NOT exceed the threshold.
 * Included to verify that the compiler does NOT emit an unnecessary probe.
 * ------------------------------------------------------------------------- */
void large_frame_exact_page(void) {
    char buf[4096];
    buf[0] = 'c';
}

/* ---------------------------------------------------------------------------
 * Uses a runtime index into the large buffer to prevent the compiler from
 * optimising the allocation away entirely. Returns a value derived from
 * the buffer contents so that the call in main() is not dead code.
 * 8192 bytes > 4096 byte page threshold  =>  probe loop REQUIRED
 * ------------------------------------------------------------------------- */
int use_stack(int n) {
    char buf[8192];
    buf[n] = (char)n;
    return buf[0] + buf[n];
}

/* ---------------------------------------------------------------------------
 * main — calls every test function to prevent dead code elimination by
 * the compiler or linker. Returns 0 on success.
 * ------------------------------------------------------------------------- */
int main(void) {
    f();
    large_frame_4097();
    large_frame_16384();
    large_frame_exact_page();

    int result = use_stack(42);

    printf("stack_probe: all functions called, use_stack returned %d\n", result);

    return 0;
}
