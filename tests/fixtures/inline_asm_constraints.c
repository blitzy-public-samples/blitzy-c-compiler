/*
 * inline_asm_constraints.c — Advanced GCC Inline Assembly Constraint Test Fixture
 *
 * Checkpoint 2 test fixture for the BCC (Blitzy's C Compiler) project.
 * Exercises the full range of GCC inline assembly features required for
 * Linux kernel compilation:
 *
 *   - Named operands ([name] syntax)
 *   - Multiple output operands
 *   - Read-write operand constraint "+r"
 *   - Immediate constraint "i"
 *   - Memory operand constraint "m"
 *   - Complex clobber lists ("memory", "cc", register clobbers)
 *   - asm goto with jump labels
 *   - .pushsection/.popsection assembler directives
 *   - Numeric constraint references ("0" tying input to output)
 *   - asm volatile semantics
 *
 * Target: x86-64 (AT&T syntax)
 *
 * Expected: Compiles cleanly, all runtime checks pass, returns 0.
 */

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* 1. Named operands: [name] syntax for readable asm templates        */
/* ------------------------------------------------------------------ */
static int test_named_operands(void)
{
    int value = 42;
    int result = 0;

    /*
     * AT&T syntax: mov src, dst
     * Named operands: %[input] and %[output] replace positional %0, %1
     */
    __asm__ __volatile__(
        "movl %[input], %[output]"
        : [output] "=r"(result)
        : [input] "r"(value)
    );

    if (result != 42) {
        printf("FAIL: named operands: expected 42, got %d\n", result);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 2. Multiple output operands                                        */
/* ------------------------------------------------------------------ */
static int test_multiple_outputs(void)
{
    int a = 0;
    int b = 0;
    int c = 100;

    /*
     * Two output operands: split the input into two copies
     * %0 = a (first output), %1 = b (second output), %2 = c (input)
     */
    __asm__ __volatile__(
        "movl %2, %0\n\t"
        "movl %2, %1\n\t"
        "addl $1, %1"
        : "=r"(a), "=r"(b)
        : "r"(c)
    );

    if (a != 100) {
        printf("FAIL: multiple outputs: a expected 100, got %d\n", a);
        return 1;
    }
    if (b != 101) {
        printf("FAIL: multiple outputs: b expected 101, got %d\n", b);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. Read-write operand "+r"                                         */
/* ------------------------------------------------------------------ */
static int test_read_write_operand(void)
{
    int val = 5;

    /*
     * "+r" constraint: val is both input and output.
     * The compiler loads val into a register, the asm adds 1, then
     * the result is stored back to val.
     */
    __asm__ __volatile__(
        "addl $1, %0"
        : "+r"(val)
    );

    if (val != 6) {
        printf("FAIL: read-write +r: expected 6, got %d\n", val);
        return 1;
    }

    /* Chain multiple read-write operations */
    __asm__ __volatile__(
        "addl $10, %0"
        : "+r"(val)
    );

    if (val != 16) {
        printf("FAIL: read-write +r chained: expected 16, got %d\n", val);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 4. Immediate constraint "i"                                        */
/* ------------------------------------------------------------------ */
static int test_immediate_constraint(void)
{
    int result = 0;

    /*
     * "i" constraint: the operand is a compile-time integer constant
     * embedded directly into the instruction encoding.
     */
    __asm__ __volatile__(
        "movl %1, %0"
        : "=r"(result)
        : "i"(42)
    );

    if (result != 42) {
        printf("FAIL: immediate constraint: expected 42, got %d\n", result);
        return 1;
    }

    /* Test "n" constraint (numeric constant, similar to "i") */
    int result2 = 0;
    __asm__ __volatile__(
        "movl %1, %0"
        : "=r"(result2)
        : "n"(99)
    );

    if (result2 != 99) {
        printf("FAIL: numeric constraint n: expected 99, got %d\n", result2);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 5. Memory operand "m"                                              */
/* ------------------------------------------------------------------ */
static int test_memory_operand(void)
{
    int var = 77;
    int dest = 0;

    /*
     * "m" constraint: the operand is a memory reference.
     * The compiler provides the memory address directly; no register
     * load is performed by the compiler — the asm accesses memory.
     */
    __asm__ __volatile__(
        "movl %1, %%eax\n\t"
        "movl %%eax, %0"
        : "=m"(dest)
        : "m"(var)
        : "eax"
    );

    if (dest != 77) {
        printf("FAIL: memory operand: expected 77, got %d\n", dest);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 6. Complex clobber list                                            */
/* ------------------------------------------------------------------ */
static int test_complex_clobbers(void)
{
    int before = 123;
    int after = 0;

    /*
     * Complex clobber list: informs the compiler that "memory", "cc"
     * (condition codes), and "rax" are modified by the asm block.
     * "memory" forces the compiler to reload values from memory after
     * the asm statement (compiler barrier).
     */
    after = before;
    __asm__ __volatile__(
        ""
        :
        :
        : "memory", "cc", "rax"
    );

    if (after != 123) {
        printf("FAIL: complex clobbers: expected 123, got %d\n", after);
        return 1;
    }

    /* Clobber with actual register usage */
    int result = 0;
    __asm__ __volatile__(
        "movl $55, %%eax\n\t"
        "movl %%eax, %0"
        : "=r"(result)
        :
        : "rax", "memory", "cc"
    );

    if (result != 55) {
        printf("FAIL: clobber with register: expected 55, got %d\n", result);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 7. asm goto with jump labels                                       */
/* ------------------------------------------------------------------ */
static int test_asm_goto(void)
{
    int val = 0;
    int reached_done = 0;

    /*
     * asm goto: the assembly template may branch to one of the
     * C labels listed after the fourth colon. The compiler must
     * wire these labels as potential jump targets.
     *
     * Syntax: asm goto("..." : : inputs : clobbers : labels)
     * Note: asm goto cannot have output operands.
     */
    __asm__ goto(
        "testl %0, %0\n\t"
        "jz %l[done]"
        :
        : "r"(val)
        : "cc"
        : done
    );

    /* If val != 0, we fall through here */
    printf("FAIL: asm goto: should have jumped to done\n");
    return 1;

done:
    reached_done = 1;

    if (!reached_done) {
        printf("FAIL: asm goto: did not reach done label\n");
        return 1;
    }

    /* Test asm goto with non-zero value (should NOT jump) */
    int val2 = 1;
    int fell_through = 0;

    __asm__ goto(
        "testl %0, %0\n\t"
        "jz %l[skip]"
        :
        : "r"(val2)
        : "cc"
        : skip
    );

    fell_through = 1;
    goto asm_goto_end;

skip:
    printf("FAIL: asm goto: should not have jumped to skip\n");
    return 1;

asm_goto_end:
    if (!fell_through) {
        printf("FAIL: asm goto: did not fall through as expected\n");
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 8. .pushsection/.popsection directives                             */
/* ------------------------------------------------------------------ */
static int test_pushsection_popsection(void)
{
    /*
     * .pushsection/.popsection: temporarily switch the assembler's
     * output section without disturbing the current section. Heavily
     * used in the Linux kernel for exception tables, alternative
     * instructions, and static keys.
     *
     * This asm places a byte 0x42 into the .data section while
     * the surrounding code remains in .text.
     */
    __asm__ __volatile__(
        ".pushsection .data\n\t"
        ".byte 0x42\n\t"
        ".popsection"
    );

    /*
     * More complex pushsection: place a string into a named section
     * (commonly used for kernel annotations).
     */
    __asm__ __volatile__(
        ".pushsection .rodata.str, \"aMS\", @progbits, 1\n\t"
        ".asciz \"bcc_test_marker\"\n\t"
        ".popsection"
    );

    /* If we reach here without assembly errors, the test passes */
    return 0;
}

/* ------------------------------------------------------------------ */
/* 9. Numeric constraint reference (tying input to output)            */
/* ------------------------------------------------------------------ */
static int test_matching_constraint(void)
{
    int val = 30;

    /*
     * The "0" constraint on the input operand ties it to output
     * operand %0, meaning both share the same register. This is
     * the traditional way to express read-modify-write before
     * the "+r" shorthand existed.
     */
    __asm__ __volatile__(
        "addl $12, %0"
        : "=r"(val)
        : "0"(val)
    );

    if (val != 42) {
        printf("FAIL: matching constraint: expected 42, got %d\n", val);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 10. Combined complex test: named + clobbers + volatile             */
/* ------------------------------------------------------------------ */
static int test_combined_complex(void)
{
    int x = 10;
    int y = 20;
    int sum = 0;

    /*
     * Combines named operands, multiple inputs, output, and clobbers
     * in a single asm statement — a pattern common in kernel code.
     */
    __asm__ __volatile__(
        "movl %[lhs], %[result]\n\t"
        "addl %[rhs], %[result]"
        : [result] "=r"(sum)
        : [lhs] "r"(x), [rhs] "r"(y)
        : "cc"
    );

    if (sum != 30) {
        printf("FAIL: combined complex: expected 30, got %d\n", sum);
        return 1;
    }

    /* Named operands with memory constraint */
    int mem_val = 0;
    int src_val = 99;

    __asm__ __volatile__(
        "movl %[src], %%eax\n\t"
        "movl %%eax, %[dst]"
        : [dst] "=m"(mem_val)
        : [src] "m"(src_val)
        : "eax"
    );

    if (mem_val != 99) {
        printf("FAIL: named memory: expected 99, got %d\n", mem_val);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 11. Early clobber "=&r" constraint                                 */
/* ------------------------------------------------------------------ */
static int test_early_clobber(void)
{
    int in1 = 7;
    int in2 = 8;
    int out = 0;

    /*
     * "=&r" (early clobber): tells the compiler that this output
     * operand is written before all input operands are consumed.
     * The compiler must NOT assign it the same register as any input.
     */
    __asm__ __volatile__(
        "movl %1, %0\n\t"
        "addl %2, %0"
        : "=&r"(out)
        : "r"(in1), "r"(in2)
    );

    if (out != 15) {
        printf("FAIL: early clobber: expected 15, got %d\n", out);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* main: exercise all constraint types, verify results                */
/* ------------------------------------------------------------------ */
int main(void)
{
    int failures = 0;

    printf("Testing advanced inline assembly constraints...\n");

    /* 1. Named operands */
    failures += test_named_operands();
    if (!failures)
        printf("  [PASS] Named operands ([name] syntax)\n");

    /* 2. Multiple output operands */
    failures += test_multiple_outputs();
    if (!failures)
        printf("  [PASS] Multiple output operands\n");

    /* 3. Read-write operand "+r" */
    failures += test_read_write_operand();
    if (!failures)
        printf("  [PASS] Read-write operand (+r)\n");

    /* 4. Immediate constraint "i" and "n" */
    failures += test_immediate_constraint();
    if (!failures)
        printf("  [PASS] Immediate constraints (i, n)\n");

    /* 5. Memory operand "m" */
    failures += test_memory_operand();
    if (!failures)
        printf("  [PASS] Memory operand (m)\n");

    /* 6. Complex clobber list */
    failures += test_complex_clobbers();
    if (!failures)
        printf("  [PASS] Complex clobber lists\n");

    /* 7. asm goto with labels */
    failures += test_asm_goto();
    if (!failures)
        printf("  [PASS] asm goto with jump labels\n");

    /* 8. .pushsection/.popsection */
    failures += test_pushsection_popsection();
    if (!failures)
        printf("  [PASS] .pushsection/.popsection directives\n");

    /* 9. Numeric matching constraint "0" */
    failures += test_matching_constraint();
    if (!failures)
        printf("  [PASS] Matching constraint (0)\n");

    /* 10. Combined complex test */
    failures += test_combined_complex();
    if (!failures)
        printf("  [PASS] Combined complex (named + clobbers + volatile)\n");

    /* 11. Early clobber "=&r" */
    failures += test_early_clobber();
    if (!failures)
        printf("  [PASS] Early clobber (=&r)\n");

    /* Summary */
    if (failures) {
        printf("\nFAILED: %d test(s) failed\n", failures);
        return 1;
    }

    printf("\nAll advanced inline assembly constraint tests passed.\n");
    return 0;
}
