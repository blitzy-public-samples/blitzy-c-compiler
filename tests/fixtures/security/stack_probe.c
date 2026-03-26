/*
 * tests/fixtures/security/stack_probe.c
 *
 * Stack Probe Test Fixture — Checkpoint 5 (x86-64 Only)
 *
 * Validates that BCC emits stack guard page probing for large stack frames
 * (>4096 bytes). On Linux, each thread's stack is bounded by a guard page.
 * If a function allocates more than one page of stack space in a single
 * adjustment (sub rsp, N where N > 4096), the guard page may be silently
 * skipped, enabling stack clash attacks. The compiler MUST emit a probe
 * loop that touches each 4096-byte page BEFORE the final stack pointer
 * adjustment.
 *
 * Consumed by: tests/checkpoint5_security.rs
 * Target: --target=x86-64 only (security mitigations are x86-64 specific)
 *
 * Validation method:
 *   1. Compile: ./bcc --target=x86-64 -o stack_probe tests/fixtures/security/stack_probe.c
 *   2. Disassemble: objdump -d stack_probe
 *   3. Verify:
 *      - Function 'f': probe loop PRESENT (8192 > 4096)
 *      - Function 'g': probe loop ABSENT  (64 < 4096)
 *      - Function 'h': probe loop PRESENT (16384 > 4096, multi-page)
 *   4. Run: ./stack_probe → prints "stack_probe OK", exit code 0
 */

#include <stdio.h>

/* Large stack frame (8192 bytes > 4096 page size).
   Per AAP User Example, this is the exact test case:
   disassembly MUST show a probe loop before the stack pointer adjustment.
   The probe loop touches each 4096-byte page to ensure the guard page
   is not silently skipped. */
void f(void) {
    char buf[8192];
    buf[0] = 1;
    /* Prevent optimization from eliminating the buffer */
    __asm__ volatile("" : : "r"(buf) : "memory");
}

/* Small stack frame (64 bytes < 4096 page size).
   No probe loop should be generated for this function.
   This serves as a negative control for the checkpoint test. */
void g(void) {
    char buf[64];
    buf[0] = 1;
    __asm__ volatile("" : : "r"(buf) : "memory");
}

/* Even larger frame to test multi-page probing */
void h(void) {
    char buf[16384];  /* 4 pages — probe loop must touch each page */
    buf[0] = 1;
    __asm__ volatile("" : : "r"(buf) : "memory");
}

int main(void) {
    f();
    g();
    h();
    printf("stack_probe OK\n");
    return 0;
}
