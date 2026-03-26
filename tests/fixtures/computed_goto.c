/*
 * tests/fixtures/computed_goto.c — Computed Goto Dispatch Test (Checkpoint 2)
 *
 * Validates GCC computed goto extension:
 *   - &&label (address-of-label operator) to obtain label pointers
 *   - goto *expr (indirect goto / computed goto) for indirect jumps
 *   - Dispatch table pattern used in threaded interpreters and the Linux kernel
 *
 * Expected: prints "computed_goto OK" and exits with code 0.
 */
#include <stdio.h>

int main(void) {
    /* Computed goto dispatch table — local array of label addresses */
    void *dispatch[4];
    dispatch[0] = &&step1;
    dispatch[1] = &&step2;
    dispatch[2] = &&step3;
    dispatch[3] = &&done;

    int result = 0;
    int idx = 0;

    goto *dispatch[idx];

step1:
    result += 1;
    idx = 1;
    goto *dispatch[idx];

step2:
    result += 10;
    idx = 2;
    goto *dispatch[idx];

step3:
    result += 100;
    idx = 3;
    goto *dispatch[idx];

done:
    if (result != 111) {
        printf("FAIL: computed goto result=%d, expected 111\n", result);
        return 1;
    }

    /* Test address-of-label directly */
    void *target = &&success;
    goto *target;

    printf("FAIL: should not reach here\n");
    return 1;

success:
    printf("computed_goto OK\n");
    return 0;
}
