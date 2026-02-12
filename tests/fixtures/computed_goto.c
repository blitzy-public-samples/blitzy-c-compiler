/*
 * computed_goto.c — GCC computed goto dispatch test fixture (Checkpoint 2)
 *
 * Tests the GCC extension `goto *ptr` with label addresses (`&&label`) for
 * computed goto dispatch tables. This pattern is widely used in the Linux
 * kernel and bytecode interpreters (e.g., Python, Lua, CPython eval loop).
 *
 * Validates:
 *   - Label address-of operator: &&label
 *   - Computed goto dispatch: goto *ptr
 *   - Dispatch table pattern (array of label addresses + indexed jump)
 *   - Threaded dispatch loop (interpreter-style opcode fetch + dispatch)
 *   - Nested/multiple dispatch tables in a single function
 *
 * Requirements:
 *   - Parser statements.rs: computed goto `goto *ptr`
 *   - Parser gcc_extensions.rs: computed goto extension dispatch
 *   - IR lowering stmt_lowering.rs: indirect branch emission
 */

#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Test 1: Basic computed goto dispatch via index
 *
 * Creates a label address table, dispatches to a label by index,
 * and verifies the correct target executed.
 * ---------------------------------------------------------------------------*/
static int test_basic_dispatch(void) {
    int result = 0;
    int index = 1; /* select label_b */

    void *labels[] = { &&label_a, &&label_b, &&label_c, &&label_end };

    goto *labels[index];

label_a:
    result += 1;
    goto *labels[3]; /* jump to label_end */
label_b:
    result += 2;
    goto *labels[3];
label_c:
    result += 4;
    goto *labels[3];
label_end:
    ;

    /* index == 1 → label_b → result should be 2 */
    if (result != 2) {
        printf("FAIL: test_basic_dispatch: expected 2, got %d\n", result);
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 2: Dispatch each target individually
 *
 * Exercises every label target in the dispatch table to confirm that each
 * one is reachable and produces the correct side effect.
 * ---------------------------------------------------------------------------*/
static int test_all_targets(void) {
    int results[3];
    int i;

    for (i = 0; i < 3; i++) {
        int val = 0;
        void *tbl[] = { &&tgt_0, &&tgt_1, &&tgt_2, &&tgt_done };

        goto *tbl[i];

    tgt_0:
        val = 10;
        goto *tbl[3];
    tgt_1:
        val = 20;
        goto *tbl[3];
    tgt_2:
        val = 30;
        goto *tbl[3];
    tgt_done:
        results[i] = val;
    }

    if (results[0] != 10 || results[1] != 20 || results[2] != 30) {
        printf("FAIL: test_all_targets: got %d %d %d\n",
               results[0], results[1], results[2]);
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 3: Threaded dispatch loop (bytecode interpreter pattern)
 *
 * Simulates a simple bytecode interpreter:
 *   Opcode 0 → ADD 1
 *   Opcode 1 → ADD 2
 *   Opcode 2 → ADD 4
 *   Opcode 3 → HALT
 *
 * The opcodes array { 0, 1, 2, 0, 3 } should produce:
 *   accumulator = 0 + 1 + 2 + 4 + 1 = 8, then HALT.
 * ---------------------------------------------------------------------------*/
static int test_threaded_dispatch(void) {
    int opcodes[] = { 0, 1, 2, 0, 3 };
    int num_ops = (int)(sizeof(opcodes) / sizeof(opcodes[0]));
    int accumulator = 0;
    int pc = 0;

    /* Dispatch table — common threaded-code pattern */
    void *dispatch[] = { &&op_add1, &&op_add2, &&op_add4, &&op_halt };

    /* Initial dispatch */
    goto *dispatch[opcodes[pc++]];

op_add1:
    accumulator += 1;
    if (pc >= num_ops) goto *dispatch[3]; /* safety: halt */
    goto *dispatch[opcodes[pc++]];

op_add2:
    accumulator += 2;
    if (pc >= num_ops) goto *dispatch[3];
    goto *dispatch[opcodes[pc++]];

op_add4:
    accumulator += 4;
    if (pc >= num_ops) goto *dispatch[3];
    goto *dispatch[opcodes[pc++]];

op_halt:
    ;

    /* Expected: 1 + 2 + 4 + 1 = 8, then halt on opcode 3 */
    if (accumulator != 8) {
        printf("FAIL: test_threaded_dispatch: expected 8, got %d\n",
               accumulator);
        return 1;
    }
    if (pc != 5) {
        printf("FAIL: test_threaded_dispatch: expected pc=5, got %d\n", pc);
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 4: Computed goto with a variable pointer (not array subscript)
 *
 * Ensures `goto *ptr` works when the target address is stored in a plain
 * void* variable rather than directly indexed from an array.
 * ---------------------------------------------------------------------------*/
static int test_variable_pointer(void) {
    int reached = 0;
    void *target = &&my_target;

    goto *target;

    /* This code must not execute */
    reached = -1;
    return 1;

my_target:
    reached = 1;

    if (reached != 1) {
        printf("FAIL: test_variable_pointer: reached=%d\n", reached);
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 5: Computed goto dispatch table with fall-through accumulation
 *
 * Each label adds to the result and jumps to the next label, forming a
 * chain. Dispatching to an earlier label accumulates more additions.
 * This tests that multiple consecutive computed gotos resolve correctly.
 *
 *   Entry at step 0 → +1 → step 1 → +2 → step 2 → +4 → done  = 7
 *   Entry at step 1 →        +2 → step 2 → +4 → done          = 6
 *   Entry at step 2 →                       +4 → done          = 4
 * ---------------------------------------------------------------------------*/
static int test_chain_dispatch(void) {
    int result;
    int entry;

    int expected[] = { 7, 6, 4 };

    for (entry = 0; entry < 3; entry++) {
        result = 0;
        void *chain[] = { &&step_0, &&step_1, &&step_2, &&chain_done };

        goto *chain[entry];

    step_0:
        result += 1;
        goto *chain[1];
    step_1:
        result += 2;
        goto *chain[2];
    step_2:
        result += 4;
        goto *chain[3];
    chain_done:
        ;

        if (result != expected[entry]) {
            printf("FAIL: test_chain_dispatch[%d]: expected %d, got %d\n",
                   entry, expected[entry], result);
            return 1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 6: Larger dispatch table (8 entries) — stress test
 *
 * Ensures the compiler handles dispatch tables beyond a trivial size.
 * Each handler sets `value` to a distinct power of 2.
 * ---------------------------------------------------------------------------*/
static int test_large_table(void) {
    int value = 0;
    int idx;

    for (idx = 0; idx < 8; idx++) {
        void *big_table[] = {
            &&ent_0, &&ent_1, &&ent_2, &&ent_3,
            &&ent_4, &&ent_5, &&ent_6, &&ent_7
        };

        goto *big_table[idx];

    ent_0: value = 1;   goto *&&ent_check;
    ent_1: value = 2;   goto *&&ent_check;
    ent_2: value = 4;   goto *&&ent_check;
    ent_3: value = 8;   goto *&&ent_check;
    ent_4: value = 16;  goto *&&ent_check;
    ent_5: value = 32;  goto *&&ent_check;
    ent_6: value = 64;  goto *&&ent_check;
    ent_7: value = 128; goto *&&ent_check;
    ent_check:
        ;

        if (value != (1 << idx)) {
            printf("FAIL: test_large_table[%d]: expected %d, got %d\n",
                   idx, 1 << idx, value);
            return 1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Test 7: Nested function with its own dispatch table
 *
 * Ensures that computed goto tables work correctly in helper functions,
 * not just in main. This mirrors how the Linux kernel uses computed gotos
 * in various subsystem functions.
 * ---------------------------------------------------------------------------*/
static int interpreter_run(const int *program, int len) {
    /*
     * Simple accumulator machine:
     *   Opcode 0: NOP   (no operation)
     *   Opcode 1: INC   (accumulator += 1)
     *   Opcode 2: DEC   (accumulator -= 1)
     *   Opcode 3: DBL   (accumulator *= 2)
     *   Opcode 4: HALT  (stop execution)
     */
    int acc = 0;
    int pc = 0;

    void *handlers[] = { &&h_nop, &&h_inc, &&h_dec, &&h_dbl, &&h_halt };

    if (pc >= len) return acc;
    goto *handlers[program[pc++]];

h_nop:
    if (pc >= len) return acc;
    goto *handlers[program[pc++]];

h_inc:
    acc += 1;
    if (pc >= len) return acc;
    goto *handlers[program[pc++]];

h_dec:
    acc -= 1;
    if (pc >= len) return acc;
    goto *handlers[program[pc++]];

h_dbl:
    acc *= 2;
    if (pc >= len) return acc;
    goto *handlers[program[pc++]];

h_halt:
    return acc;
}

static int test_interpreter(void) {
    /* Program: INC, INC, INC, DBL, DEC, HALT → ((0+1+1+1)*2)-1 = 5 */
    int program[] = { 1, 1, 1, 3, 2, 4 };
    int len = (int)(sizeof(program) / sizeof(program[0]));

    int result = interpreter_run(program, len);
    if (result != 5) {
        printf("FAIL: test_interpreter: expected 5, got %d\n", result);
        return 1;
    }

    /* Program: NOP, INC, NOP, INC, DBL, HALT → ((0+1+1)*2) = 4 */
    int program2[] = { 0, 1, 0, 1, 3, 4 };
    int len2 = (int)(sizeof(program2) / sizeof(program2[0]));

    int result2 = interpreter_run(program2, len2);
    if (result2 != 4) {
        printf("FAIL: test_interpreter(2): expected 4, got %d\n", result2);
        return 1;
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * main — exercise all computed goto tests
 * ---------------------------------------------------------------------------*/
int main(void) {
    int failures = 0;

    failures += test_basic_dispatch();
    failures += test_all_targets();
    failures += test_threaded_dispatch();
    failures += test_variable_pointer();
    failures += test_chain_dispatch();
    failures += test_large_table();
    failures += test_interpreter();

    if (failures != 0) {
        printf("FAIL: %d test(s) failed\n", failures);
        return 1;
    }

    printf("PASS: computed_goto\n");
    return 0;
}
