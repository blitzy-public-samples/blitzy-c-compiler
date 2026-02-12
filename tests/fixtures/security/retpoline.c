/*
 * retpoline.c — Retpoline (Spectre v2 mitigation) test fixture
 *
 * Checkpoint 5 validation: When compiled with -mretpoline targeting x86-64,
 * every indirect function pointer call in this file MUST be routed through
 * __x86_indirect_thunk_* instead of using a direct `call *%reg` instruction.
 *
 * Compilation command:
 *   ./bcc -mretpoline --target=x86-64 -c -o retpoline.o retpoline.c
 *
 * Verification (objdump):
 *   objdump -d retpoline.o | grep -E 'call.*__x86_indirect_thunk'
 *   # Must find thunk calls; must NOT find 'call *%rax' or similar
 *
 * Per Section 0.1.2 User Example:
 *   "function containing (*fptr)() call → call instruction targets
 *    __x86_indirect_thunk_*, not the pointer directly"
 *
 * Per Section 0.5.1 Group 6:
 *   src/backend/x86_64/security.rs implements retpoline thunks
 *   (__x86_indirect_thunk_rax, etc.) when -mretpoline flag is active.
 */

#include <stdio.h>

/* ---------- Function pointer type declaration ---------- */
typedef void (*func_ptr)(void);

/* ---------- Target functions called via indirect pointers ---------- */

/* First indirect call target */
void target_a(void) {
    printf("target_a\n");
}

/* Second indirect call target */
void target_b(void) {
    printf("target_b\n");
}

/* Third indirect call target for array-based dispatch */
void target_c(void) {
    printf("target_c\n");
}

/* Fourth indirect call target for additional coverage */
void target_d(void) {
    printf("target_d\n");
}

/* ---------- Global function pointer variable ---------- */

/*
 * Call through a global function pointer variable.
 * The assignment and call site are separated so the compiler cannot
 * resolve the target statically — the indirect call must go through
 * __x86_indirect_thunk_*.
 */
func_ptr global_fptr;

/* ---------- Function pointer array ---------- */

/*
 * Call through a function pointer array element.
 * Array indexing produces an indirect call that must be retpoline-protected.
 */
func_ptr dispatch_table[4] = { target_a, target_b, target_c, target_d };

/* ---------- Canonical indirect call pattern (Section 0.1.2) ---------- */

/*
 * This is the exact pattern from the user example in Section 0.1.2:
 *   "function containing (*fptr)() call"
 *
 * When compiled with -mretpoline, the (*fptr)() indirect call MUST
 * generate:  call __x86_indirect_thunk_rax  (or similar register thunk)
 * and MUST NOT generate:  call *%rax  (or any direct indirect call)
 */
void call_indirect(func_ptr fptr) {
    (*fptr)();  /* This indirect call MUST go through __x86_indirect_thunk_* */
}

/* ---------- Call through global function pointer ---------- */

/*
 * Exercises the retpoline path for a call through a globally-visible
 * function pointer variable.  The global is written externally (by main),
 * so the compiler cannot devirtualize the call.
 */
void call_global_fptr(void) {
    global_fptr();  /* Indirect call via global — must use retpoline thunk */
}

/* ---------- Call through function pointer array element ---------- */

/*
 * Exercises the retpoline path for an indexed dispatch through a function
 * pointer array.  The index is a runtime parameter, preventing constant
 * folding of the target address.
 */
void call_from_array(int index) {
    if (index >= 0 && index < 4) {
        dispatch_table[index]();  /* Indirect call via array — must use retpoline thunk */
    }
}

/* ---------- Call through parameter (alternative syntax) ---------- */

/*
 * Exercises the retpoline path using the alternative call-through-parameter
 * syntax (fptr(arg) rather than (*fptr)(arg)).  Both syntactic forms
 * produce identical indirect call IR and must both be retpoline-protected.
 */
int call_with_return(int (*compute)(int, int), int a, int b) {
    return compute(a, b);  /* Indirect call — must use retpoline thunk */
}

/* Helper functions used as targets for call_with_return */
int add_values(int a, int b) {
    return a + b;
}

int mul_values(int a, int b) {
    return a * b;
}

/* ---------- Struct containing a function pointer ---------- */

/*
 * Exercises the retpoline path for an indirect call through a function
 * pointer embedded in a struct (vtable-like pattern common in Linux kernel).
 */
struct operations {
    void (*execute)(void);
    int  (*compute)(int, int);
};

void call_via_struct(struct operations *ops) {
    ops->execute();            /* Indirect call via struct member — must use retpoline thunk */
    ops->compute(10, 20);     /* Another indirect call via struct member */
}

/* ---------- main — exercises all indirect call patterns ---------- */

int main(void) {
    int result;

    /* Pattern 1: Canonical indirect call through parameter (Section 0.1.2) */
    printf("=== Pattern 1: call_indirect (canonical) ===\n");
    call_indirect(target_a);
    call_indirect(target_b);

    /* Pattern 2: Indirect call through global function pointer */
    printf("=== Pattern 2: global function pointer ===\n");
    global_fptr = target_c;
    call_global_fptr();
    global_fptr = target_d;
    call_global_fptr();

    /* Pattern 3: Indirect call through function pointer array */
    printf("=== Pattern 3: function pointer array ===\n");
    call_from_array(0);
    call_from_array(1);
    call_from_array(2);
    call_from_array(3);

    /* Pattern 4: Indirect call with return value */
    printf("=== Pattern 4: indirect call with return ===\n");
    result = call_with_return(add_values, 3, 4);
    printf("add_values(3, 4) = %d\n", result);
    result = call_with_return(mul_values, 5, 6);
    printf("mul_values(5, 6) = %d\n", result);

    /* Pattern 5: Indirect call through struct member (vtable pattern) */
    printf("=== Pattern 5: struct function pointer (vtable) ===\n");
    struct operations ops;
    ops.execute = target_a;
    ops.compute = add_values;
    call_via_struct(&ops);

    printf("All retpoline tests passed.\n");
    return 0;
}
