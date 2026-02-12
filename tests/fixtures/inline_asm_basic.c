/*
 * tests/fixtures/inline_asm_basic.c
 *
 * Basic inline assembly test fixture for Checkpoint 2 validation.
 *
 * Tests GCC-style asm / asm volatile statements using AT&T syntax,
 * exercising the parser's inline_asm.rs module on fundamental constructs:
 *
 *   1. Basic asm with no operands (nop)
 *   2. asm volatile with a single output operand ("=r")
 *   3. asm with both input and output operands ("=r", "r")
 *   4. asm volatile with memory clobber ("memory")
 *   5. asm volatile with condition-codes clobber ("cc")
 *   6. __asm__ / __volatile__ alternate keyword forms
 *   7. Simple integer addition performed entirely in inline asm
 *
 * All inline assembly uses x86-64 AT&T syntax. This fixture validates
 * parsing correctness; advanced constraints (named operands, asm goto,
 * .pushsection/.popsection) are covered in inline_asm_constraints.c.
 *
 * Expected result: compiles, runs, prints "ALL TESTS PASSED", exits 0.
 */

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Test 1: Basic asm statement with no operands.                      */
/*         The simplest possible inline assembly — a single nop.      */
/* ------------------------------------------------------------------ */
static int test_basic_nop(void) {
    asm("nop");
    /* If the parser accepted the asm statement and code generation
       emitted the NOP opcode, reaching this point means success. */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 2: asm volatile with a single output operand.                 */
/*         Moves the immediate value 42 into a general-purpose        */
/*         register selected by the compiler ("=r" constraint).       */
/* ------------------------------------------------------------------ */
static int test_volatile_output(void) {
    int result;
    asm volatile("mov $42, %0" : "=r"(result));
    if (result != 42) {
        printf("FAIL: test_volatile_output: expected 42, got %d\n", result);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 3: asm with input and output operands.                        */
/*         Copies the value of variable 'a' into variable 'b' via    */
/*         a mov instruction.  Output: "=r"(b), Input: "r"(a).       */
/* ------------------------------------------------------------------ */
static int test_input_output(void) {
    int a = 5, b;
    asm("mov %1, %0" : "=r"(b) : "r"(a));
    if (b != 5) {
        printf("FAIL: test_input_output: expected 5, got %d\n", b);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 4: asm volatile with memory clobber ("memory").               */
/*         An empty template with the "memory" clobber acts as a      */
/*         compiler memory barrier, forcing all pending stores to be  */
/*         flushed and preventing load/store reordering across this   */
/*         point.                                                     */
/* ------------------------------------------------------------------ */
static int test_memory_clobber(void) {
    volatile int x = 10;
    asm volatile("" ::: "memory");
    /* The volatile qualifier and memory barrier ensure 'x' is not
       optimized away and retains its value across the barrier. */
    if (x != 10) {
        printf("FAIL: test_memory_clobber: x changed unexpectedly to %d\n", x);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 5: asm volatile with condition-codes clobber ("cc").          */
/*         Informs the compiler that the assembly may modify the      */
/*         processor flags register (EFLAGS/RFLAGS on x86).           */
/* ------------------------------------------------------------------ */
static int test_cc_clobber(void) {
    asm volatile("" ::: "cc");
    /* Success: the "cc" clobber was parsed and accepted. */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 6: __asm__ / __volatile__ alternate keyword variants.         */
/*         These are the ISO C-compatible forms of the asm and        */
/*         volatile keywords, usable even with -std=c11 strict mode.  */
/* ------------------------------------------------------------------ */
static int test_asm_keyword_variant(void) {
    /* Alternate keyword with volatile qualifier */
    __asm__ __volatile__("nop");

    /* Alternate keyword without volatile (plain __asm__) */
    __asm__("nop");

    /* Mix: __asm__ with explicit empty operands and no clobbers */
    __asm__ __volatile__("" : : : );

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 7: Simple integer addition performed via inline assembly.     */
/*         Adds two C integers using the x86 ADD instruction.         */
/*         Uses "+r" (read-write) constraint on the accumulator.      */
/* ------------------------------------------------------------------ */
static int test_asm_add(void) {
    int x = 10;
    int y = 25;
    int sum = x;
    /* addl: 32-bit integer addition in AT&T syntax.
       "+r"(sum): sum is both an input (initial value = x) and output.
       "r"(y):   y is a read-only input in any GPR. */
    asm("addl %1, %0" : "+r"(sum) : "r"(y));
    if (sum != 35) {
        printf("FAIL: test_asm_add: expected 35, got %d\n", sum);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 8: Combined memory and cc clobbers with operands.             */
/*         Demonstrates that the parser handles a clobber list with   */
/*         multiple entries alongside input/output operands.          */
/* ------------------------------------------------------------------ */
static int test_combined_clobbers(void) {
    int val = 100;
    int out;
    asm volatile("mov %1, %0"
                 : "=r"(out)
                 : "r"(val)
                 : "memory", "cc");
    if (out != 100) {
        printf("FAIL: test_combined_clobbers: expected 100, got %d\n", out);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* main — run every test case, report results, return 0 on success.   */
/* ------------------------------------------------------------------ */
int main(void) {
    int failures = 0;

    /* Test 1: basic asm nop, no operands */
    failures += test_basic_nop();

    /* Test 2: asm volatile with output operand ("=r") */
    failures += test_volatile_output();

    /* Test 3: asm with input ("r") and output ("=r") */
    failures += test_input_output();

    /* Test 4: asm volatile with memory clobber */
    failures += test_memory_clobber();

    /* Test 5: asm volatile with cc clobber */
    failures += test_cc_clobber();

    /* Test 6: __asm__ / __volatile__ alternate keywords */
    failures += test_asm_keyword_variant();

    /* Test 7: addition of two integers via inline asm */
    failures += test_asm_add();

    /* Test 8: combined memory + cc clobbers with operands */
    failures += test_combined_clobbers();

    if (failures != 0) {
        printf("FAILED: %d test(s) failed\n", failures);
        return 1;
    }

    printf("ALL TESTS PASSED\n");
    return 0;
}
