/*
 * tests/fixtures/shared_lib/main.c — Dynamic Linking Consumer
 *
 * Checkpoint 4 validation source file.
 * Compile: ./bcc -o main main.c -L. -lfoo
 * Run:     LD_LIBRARY_PATH=. ./main
 *
 * This file links against libfoo.so (produced from foo.c) and validates
 * that dynamic symbol resolution and PLT-based function calls work
 * correctly at runtime across all four target architectures:
 *   x86-64, i686, AArch64, RISC-V 64
 *
 * Test coverage:
 *   - PLT-based function calls with integer arguments and return values
 *   - PLT-based function calls returning pointers (string from .rodata)
 *   - GOT-based global data access through an exported accessor function
 *   - Dynamic symbol resolution via .dynsym / .gnu.hash
 *   - Relocation processing via .rela.dyn / .rela.plt
 *
 * Expected result:
 *   stdout: "shared_lib OK\n"
 *   exit code: 0
 *
 * On any failure, a diagnostic FAIL message is printed to stdout
 * identifying the failing test, and the program exits with code 1.
 */

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Extern declarations for functions provided by libfoo.so.           */
/* These prototypes must exactly match the exported symbols in foo.c.  */
/* At link time, these resolve through the PLT/GOT mechanism.         */
/* ------------------------------------------------------------------ */

/* Simple integer addition — validates basic PLT call with args */
extern int foo_add(int a, int b);

/* Integer multiplication (via internal helper) — another PLT pattern */
extern int foo_multiply(int a, int b);

/* Returns a string literal pointer from .rodata — tests pointer return
   through PLT/GOT and data section relocations in PIC mode */
extern const char *foo_greeting(void);

/* Returns value of exported global variable — tests GOT-based global
   data access across shared object boundary */
extern int foo_get_global(void);

int main(void) {
    /* -------------------------------------------------------------- */
    /* Test 1: PLT-based function call — foo_add                      */
    /* Validates: integer argument passing, integer return, PLT stub   */
    /* Expected: foo_add(3, 4) == 7                                   */
    /* -------------------------------------------------------------- */
    int sum = foo_add(3, 4);
    if (sum != 7) {
        printf("FAIL: foo_add(3, 4) = %d, expected 7\n", sum);
        return 1;
    }

    /* -------------------------------------------------------------- */
    /* Test 2: PLT-based function call — foo_multiply                 */
    /* Validates: another PLT call pattern, intra-library helper call  */
    /* Expected: foo_multiply(5, 6) == 30                             */
    /*   (internally: foo_internal_helper(5) * 6 / 2 = 10*6/2 = 30)  */
    /* -------------------------------------------------------------- */
    int product = foo_multiply(5, 6);
    if (product != 30) {
        printf("FAIL: foo_multiply(5, 6) = %d, expected 30\n", product);
        return 1;
    }

    /* -------------------------------------------------------------- */
    /* Test 3: Pointer return through PLT/GOT — foo_greeting          */
    /* Validates: pointer return value, .rodata access in PIC mode,   */
    /*            GOT-based data reference across .so boundary         */
    /* Expected: non-NULL pointer starting with 'H' ("Hello from...")  */
    /* -------------------------------------------------------------- */
    const char *msg = foo_greeting();
    if (msg == 0 || msg[0] != 'H') {
        printf("FAIL: foo_greeting() returned unexpected value\n");
        return 1;
    }

    /* -------------------------------------------------------------- */
    /* Test 4: GOT-based global data access — foo_get_global          */
    /* Validates: global variable access through GOT, data section    */
    /*            relocation across shared object boundary             */
    /* Expected: foo_get_global() == 42 (value of foo_global_value)   */
    /* -------------------------------------------------------------- */
    int gval = foo_get_global();
    if (gval != 42) {
        printf("FAIL: foo_get_global() = %d, expected 42\n", gval);
        return 1;
    }

    /* All tests passed */
    printf("shared_lib OK\n");
    return 0;
}
