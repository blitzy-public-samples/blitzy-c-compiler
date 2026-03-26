/*
 * tests/fixtures/inline_asm_constraints.c
 *
 * Inline assembly constraint test for BCC Checkpoint 2.
 * Validates support for various inline asm constraint types:
 *   Output:  "=r" (register), "=m" (memory), "+r" (read-write register)
 *   Input:   "r" (register), "i" (immediate), "n" (compile-time immediate)
 *   Clobber: "cc" (condition codes), "memory" (memory barrier)
 *   Named:   [name] operand syntax
 *
 * All tests use x86-64 AT&T syntax.
 * Exercises:
 *   - src/frontend/parser/inline_asm.rs   (constraint parsing)
 *   - src/ir/lowering/asm_lowering.rs     (constraint validation)
 */

#include <stdio.h>

int main(void) {
    /* ------------------------------------------------------------------ */
    /* "=r" output constraint: result written to a general-purpose register */
    /* ------------------------------------------------------------------ */
    int r_out;
    asm volatile("movl $10, %0" : "=r"(r_out));
    if (r_out != 10) {
        printf("FAIL: =r constraint\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "=m" output constraint: result written directly to memory           */
    /* ------------------------------------------------------------------ */
    int m_out;
    asm volatile("movl $20, %0" : "=m"(m_out));
    if (m_out != 20) {
        printf("FAIL: =m constraint\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "+r" read-write constraint: operand is both input and output        */
    /* ------------------------------------------------------------------ */
    int rw = 5;
    asm volatile("addl $10, %0" : "+r"(rw));
    if (rw != 15) {
        printf("FAIL: +r constraint\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "r" input constraint: value provided in a register                  */
    /* ------------------------------------------------------------------ */
    int in_val = 77;
    int out_val;
    asm volatile("movl %1, %0" : "=r"(out_val) : "r"(in_val));
    if (out_val != 77) {
        printf("FAIL: r input\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "i" immediate constraint: integer constant as immediate operand     */
    /* ------------------------------------------------------------------ */
    int imm_out;
    asm volatile("movl %1, %0" : "=r"(imm_out) : "i"(99));
    if (imm_out != 99) {
        printf("FAIL: i constraint\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "n" immediate constraint: compile-time known numeric constant       */
    /* Similar to "i" but guarantees a known numeric value (not a symbol)  */
    /* ------------------------------------------------------------------ */
    int n_out;
    asm volatile("movl %1, %0" : "=r"(n_out) : "n"(42));
    if (n_out != 42) {
        printf("FAIL: n constraint\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "cc" clobber: tells compiler that condition flags are modified       */
    /* ------------------------------------------------------------------ */
    int cc_val = 1;
    asm volatile("addl $1, %0" : "+r"(cc_val) : : "cc");
    if (cc_val != 2) {
        printf("FAIL: cc clobber\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* "memory" clobber: acts as a compiler memory barrier                 */
    /* Ensures memory writes before the asm are visible after it           */
    /* ------------------------------------------------------------------ */
    volatile int mem_val = 100;
    asm volatile("" : : : "memory");
    if (mem_val != 100) {
        printf("FAIL: memory clobber\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Named operands [name]: use symbolic names instead of %0, %1, etc.  */
    /* ------------------------------------------------------------------ */
    int named_in = 50;
    int named_out;
    asm volatile("movl %[src], %[dst]"
                 : [dst] "=r"(named_out)
                 : [src] "r"(named_in));
    if (named_out != 50) {
        printf("FAIL: named operands\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Combined: multiple inputs, outputs, named operands, and clobbers   */
    /* Uses "=&r" (early-clobber) to prevent output aliasing with inputs  */
    /* ------------------------------------------------------------------ */
    int a = 3, b = 7, sum;
    asm volatile("movl %[val_a], %[result]\n\t"
                 "addl %[val_b], %[result]"
                 : [result] "=&r"(sum)
                 : [val_a] "r"(a), [val_b] "r"(b)
                 : "cc");
    if (sum != 10) {
        printf("FAIL: combined\n");
        return 1;
    }

    printf("inline_asm_constraints OK\n");
    return 0;
}
