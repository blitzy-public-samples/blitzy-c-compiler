/*
 * main.c — Dynamic linking consumer test fixture for BCC Checkpoint 4.
 *
 * Compilation:
 *   ./bcc -fPIC -shared -o libfoo.so foo.c
 *   ./bcc -o main main.c -L. -lfoo
 *
 * Execution:
 *   LD_LIBRARY_PATH=. ./main
 *
 * This program links against libfoo.so (produced from foo.c) and calls its
 * exported functions to exercise the complete dynamic linking pipeline:
 *
 *   - DT_NEEDED entries: the executable's .dynamic section must reference
 *     libfoo.so so that the runtime dynamic linker loads it automatically.
 *
 *   - PLT stubs: each call to an external function (add, multiply,
 *     get_library_name, get_shared_value) must go through a PLT entry that
 *     performs lazy symbol resolution via the GOT.
 *
 *   - GOT entries: the global data accessed by get_shared_value() inside the
 *     shared library is resolved at load time through GOT relocations
 *     (R_*_GLOB_DAT or R_*_COPY depending on architecture and access style).
 *
 *   - PT_INTERP program header: the resulting executable must contain a
 *     PT_INTERP segment referencing the correct dynamic linker path for the
 *     target architecture:
 *       x86-64:   /lib64/ld-linux-x86-64.so.2
 *       i686:     /lib/ld-linux.so.2
 *       AArch64:  /lib/ld-linux-aarch64.so.1
 *       RISC-V64: /lib/ld-linux-riscv64-lp64d.so.1
 *
 *   - Dynamic symbol resolution: all extern-declared functions must be
 *     resolved at runtime by the dynamic linker against libfoo.so's .dynsym.
 *
 * Expected output on success:
 *   add(3, 4) = 7
 *   multiply(5, 6) = 30
 *   library name: libfoo
 *   shared value: 42
 *   SHARED_LIB_OK
 *
 * Exit code:
 *   0 on success, 1 on any verification failure.
 *
 * Used by: tests/checkpoint4_shared_lib.rs
 * Depends on: tests/fixtures/shared_lib/foo.c (compiled into libfoo.so)
 */

#include <stdio.h>

/*
 * External function declarations matching the exported symbols from foo.c.
 *
 * These symbols are resolved at runtime by the dynamic linker against
 * libfoo.so's .dynsym table. Each declaration generates a PLT stub and a
 * corresponding GOT entry in the linked executable.
 */

/* Basic integer addition — exercises PLT call and return value passing. */
extern int add(int a, int b);

/* Integer multiplication — validates multi-symbol .dynsym resolution. */
extern int multiply(int a, int b);

/*
 * Returns a pointer to a string literal in the shared library's .rodata.
 * Tests GOT-relative data access for string constants under PIC.
 */
extern const char *get_library_name(void);

/*
 * Returns the value of the global variable 'shared_value' (initialized to 42
 * in foo.c). Tests GOT access for global data symbols across shared object
 * boundaries.
 */
extern int get_shared_value(void);

/*
 * main — Dynamic linking consumer entry point.
 *
 * Calls each exported function from libfoo.so, verifies the returned values
 * against expected results, and prints diagnostic output. On complete success
 * the sentinel string "SHARED_LIB_OK" is printed, which the checkpoint test
 * harness (checkpoint4_shared_lib.rs) scans for to confirm a passing test.
 *
 * Returns 0 on success, 1 on any verification failure.
 */
int main(void)
{
    int result;
    const char *name;
    int failures = 0;

    /* -----------------------------------------------------------------------
     * Test 1: add(3, 4) — expected result is 7.
     *
     * This call exercises a basic PLT-dispatched function call into the shared
     * library. The dynamic linker resolves 'add' from libfoo.so's .dynsym on
     * the first invocation (lazy binding) or at load time (eager binding with
     * LD_BIND_NOW).
     * -----------------------------------------------------------------------
     */
    result = add(3, 4);
    printf("add(3, 4) = %d\n", result);
    if (result != 7) {
        printf("FAIL: add(3, 4) expected 7, got %d\n", result);
        failures++;
    }

    /* -----------------------------------------------------------------------
     * Test 2: multiply(5, 6) — expected result is 30.
     *
     * Validates that the dynamic symbol table correctly handles multiple
     * exported function symbols and that PLT stubs are generated for each
     * distinct external call target.
     * -----------------------------------------------------------------------
     */
    result = multiply(5, 6);
    printf("multiply(5, 6) = %d\n", result);
    if (result != 30) {
        printf("FAIL: multiply(5, 6) expected 30, got %d\n", result);
        failures++;
    }

    /* -----------------------------------------------------------------------
     * Test 3: get_library_name() — expected return value is "libfoo".
     *
     * The returned pointer references a string literal in the shared library's
     * .rodata section. Under PIC, the shared library accesses this address via
     * a GOT-relative load. The pointer itself is returned via register (per
     * ABI) and dereferenced in this executable's address space — validating
     * that the shared library's text and data segments are correctly mapped.
     * -----------------------------------------------------------------------
     */
    name = get_library_name();
    printf("library name: %s\n", name);
    if (name[0] != 'l' || name[1] != 'i' || name[2] != 'b' ||
        name[3] != 'f' || name[4] != 'o' || name[5] != 'o' ||
        name[6] != '\0') {
        printf("FAIL: get_library_name() expected \"libfoo\", got \"%s\"\n",
               name);
        failures++;
    }

    /* -----------------------------------------------------------------------
     * Test 4: get_shared_value() — expected result is 42.
     *
     * The function accesses the global variable 'shared_value' (initialized
     * to 42 in foo.c). Within the shared library, this access goes through
     * the GOT because 'shared_value' has default visibility and could be
     * interposed. This test validates GOT entry generation for global data
     * symbols and proper relocation processing at load time.
     * -----------------------------------------------------------------------
     */
    result = get_shared_value();
    printf("shared value: %d\n", result);
    if (result != 42) {
        printf("FAIL: get_shared_value() expected 42, got %d\n", result);
        failures++;
    }

    /* -----------------------------------------------------------------------
     * Final verdict: print the sentinel string only when every test passed.
     * The checkpoint test harness scans stdout for "SHARED_LIB_OK" to
     * determine success.
     * -----------------------------------------------------------------------
     */
    if (failures == 0) {
        printf("SHARED_LIB_OK\n");
        return 0;
    }

    printf("FAIL: %d test(s) failed\n", failures);
    return 1;
}
