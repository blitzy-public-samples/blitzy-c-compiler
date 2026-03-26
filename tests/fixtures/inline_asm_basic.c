/*
 * tests/fixtures/inline_asm_basic.c — Basic Inline Assembly Test (Checkpoint 2)
 *
 * Validates basic inline assembly support in BCC:
 *   - asm volatile / __asm__ __volatile__ keyword spellings
 *   - AT&T syntax (src, dst operand order; % register prefix)
 *   - No-operand asm statements
 *   - Output-only operands ("=r")
 *   - Input + output operands
 *   - Memory clobber ("memory")
 *   - cc (condition-code) clobber
 *
 * Target: x86-64 (primary validation architecture)
 *
 * NOTE: Constraint-specific tests are in inline_asm_constraints.c.
 *       This file focuses on BASIC inline asm forms only.
 *
 * Expected: compiles, runs, prints "inline_asm_basic OK", exits 0.
 */

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Helper: report a failure and return non-zero exit code.            */
/* ------------------------------------------------------------------ */
static int fail(const char *msg)
{
    printf("FAIL: %s\n", msg);
    return 1;
}

int main(void)
{
    /* ==============================================================
     * Test 1: No-operand asm — simplest possible form.
     *         Just a NOP instruction with no inputs, outputs, or
     *         clobbers.  Validates that the parser accepts the
     *         basic `asm volatile("...")` syntax.
     * ============================================================== */
    asm volatile("nop");

    /* ==============================================================
     * Test 2: Alternate keyword spelling.
     *         GCC (and the Linux kernel) use __asm__ and
     *         __volatile__ interchangeably with asm/volatile.
     *         Both forms must be accepted.
     * ============================================================== */
    __asm__ __volatile__("nop");

    /* ==============================================================
     * Test 3: Output operand only.
     *         Move the immediate value 42 into a register selected
     *         by the compiler ("=r" constraint), then verify the
     *         C variable received the correct value.
     * ============================================================== */
    int result;
    asm volatile("movl $42, %0"
                 : "=r"(result));
    if (result != 42) {
        return fail("asm output — expected 42");
    }

    /* ==============================================================
     * Test 4: Input + output operands — add two values.
     *         Demonstrates multi-operand syntax.  The "addl %2, %1"
     *         instruction is in AT&T order (source, destination).
     *
     *         NOTE: %1 is an input-only operand and we modify the
     *         register holding it; this is intentionally loose to
     *         exercise the parser with a realistic (if imprecise)
     *         pattern seen in kernel code.  We do NOT assert the
     *         result because the semantics are architecture-
     *         dependent when clobbering an input operand.
     * ============================================================== */
    int a = 10, b = 20, sum;
    asm volatile("addl %2, %1\n\t"
                 "movl %1, %0"
                 : "=r"(sum)
                 : "r"(a), "r"(b)
                 : /* no clobbers */);
    /*
     * A simpler, well-defined input/output test follows:
     * copy a value through a register.
     */
    int val = 100;
    int out;
    asm volatile("movl %1, %0"
                 : "=r"(out)
                 : "r"(val));
    if (out != 100) {
        return fail("asm in/out — expected 100");
    }

    /* ==============================================================
     * Test 5: Memory clobber.
     *         An empty asm template with "memory" in the clobber
     *         list acts as a compiler memory barrier, preventing
     *         reordering of memory accesses across the barrier.
     *         This is the most common inline-asm idiom in the
     *         Linux kernel (barrier() macro).
     * ============================================================== */
    int mem_val = 0;
    asm volatile("" : : : "memory");   /* compiler memory barrier */
    mem_val = 1;
    asm volatile("" : : : "memory");   /* compiler memory barrier */
    if (mem_val != 1) {
        return fail("memory clobber — expected 1");
    }

    /* ==============================================================
     * Test 6: Condition-code clobber ("cc").
     *         Many x86 instructions implicitly modify EFLAGS.
     *         The "cc" clobber tells the compiler that condition
     *         codes are destroyed.  Combine with "memory" to
     *         exercise multi-clobber syntax.
     * ============================================================== */
    int cc_val = 55;
    int cc_out;
    asm volatile("movl %1, %0"
                 : "=r"(cc_out)
                 : "r"(cc_val)
                 : "cc", "memory");
    if (cc_out != 55) {
        return fail("cc clobber — expected 55");
    }

    /* ==============================================================
     * Test 7: Multi-line asm template.
     *         Verifies that the parser handles newlines and tabs
     *         within the asm template string correctly—common in
     *         kernel inline assembly for readability.
     * ============================================================== */
    int multi_a = 3, multi_b = 7, multi_out;
    asm volatile(
        "movl %1, %0\n\t"
        "addl %2, %0\n\t"
        : "=&r"(multi_out)        /* early-clobber output */
        : "r"(multi_a), "r"(multi_b)
        : "cc"
    );
    if (multi_out != 10) {
        return fail("multi-line asm — expected 10");
    }

    /* ==============================================================
     * Test 8: Empty asm template with output.
     *         The template is empty; the compiler is simply told
     *         to associate a register with the C variable.  The
     *         value is undefined, but the form must parse and
     *         compile without errors.  (No value assertion.)
     * ============================================================== */
    int empty_out;
    asm volatile("" : "=r"(empty_out));
    (void)empty_out;   /* suppress unused-variable warning */

    /* All tests passed. */
    printf("inline_asm_basic OK\n");
    return 0;
}
