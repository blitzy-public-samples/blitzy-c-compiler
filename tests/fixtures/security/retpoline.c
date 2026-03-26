/*
 * tests/fixtures/security/retpoline.c — Retpoline Code Generation Test
 *
 * Checkpoint 5 — x86-64 Security Mitigation Validation
 *
 * Purpose:
 *   This test validates that BCC correctly implements retpoline code
 *   generation when compiled with the -mretpoline flag on x86-64.
 *
 *   Retpolines are a Spectre v2 (Branch Target Injection) mitigation
 *   technique. Instead of emitting a direct indirect branch instruction
 *   such as `call *%rax` or `jmp *%rax`, the compiler must redirect
 *   all indirect calls/jumps through a special thunk function named
 *   `__x86_indirect_thunk_<reg>` (e.g., __x86_indirect_thunk_rax).
 *   The thunk uses a return-trampoline ("retpoline") sequence that
 *   prevents the CPU's speculative execution engine from predicting
 *   the indirect branch target.
 *
 * Validation protocol (performed by tests/checkpoint5_security.rs):
 *
 *   1. Compile WITH -mretpoline --target=x86-64:
 *      - `objdump -d` of call_indirect MUST show:
 *          call __x86_indirect_thunk_rax  (or similar register thunk)
 *      - `objdump -d` MUST NOT show:
 *          call *%rax   or   jmp *%rax
 *        for the indirect call site inside call_indirect.
 *
 *   2. Compile WITHOUT -mretpoline:
 *      - `objdump -d` of call_indirect MUST show a normal:
 *          call *%rax   (standard indirect call)
 *
 *   3. Runtime correctness (both modes):
 *      - Program prints "retpoline OK" to stdout and exits with code 0.
 *
 * Design notes:
 *   - call_indirect is intentionally non-static and non-inline so the
 *     compiler emits a real function body visible in disassembly.
 *   - The function pointer fptr is received as a parameter (not a global)
 *     to prevent the compiler from devirtualizing the call at -O0.
 *   - target_function is static to keep the test self-contained while
 *     still being callable through the pointer.
 */

#include <stdio.h>

/* Target function for the indirect call.
   Static linkage keeps it out of the global symbol table, but its
   address can still be taken and passed through a function pointer. */
static int target_function(int x) {
    return x + 1;
}

/*
 * call_indirect — Performs an indirect call through a function pointer.
 *
 * When compiled with -mretpoline on x86-64, the indirect call
 *     fptr(arg)
 * MUST be lowered to:
 *     call __x86_indirect_thunk_rax
 * (or the thunk for whichever register holds the function pointer)
 * instead of a bare:
 *     call *%rax
 *
 * This function is NOT static and NOT inline so it appears as a
 * distinct symbol in the disassembly output, making it easy for the
 * test harness to locate and inspect the generated instructions.
 */
int call_indirect(int (*fptr)(int), int arg) {
    return fptr(arg);
}

int main(void) {
    int result = call_indirect(target_function, 41);
    if (result != 42) {
        printf("FAIL: retpoline result=%d, expected 42\n", result);
        return 1;
    }
    printf("retpoline OK\n");
    return 0;
}
