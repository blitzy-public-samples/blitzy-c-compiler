/*
 * recursive_macro.c — Preprocessor paint-marker recursion protection test
 *
 * Checkpoint 2 fixture for BCC (Blitzy's C Compiler).
 *
 * This file validates that the preprocessor's paint-marker system correctly
 * prevents infinite expansion of self-referential and mutually recursive
 * macros. Per Section 0.1.2 of the Agent Action Plan:
 *
 *   "#define A A" and "int x = A;" must terminate in <5 seconds, no hang.
 *
 * The paint-marker mechanism works at the token level during Phase 2 macro
 * expansion: when a macro name token is encountered during its own expansion,
 * it is "painted" (marked) and treated as an ordinary identifier — it is NOT
 * re-expanded. This is architecturally distinct from circular #include
 * detection, which operates at the file/directive level.
 *
 * Test cases exercised:
 *   1. Simple self-referential object-like macro:  #define A A
 *   2. Mutually recursive object-like macros:      #define B C / #define C B
 *   3. Function-like self-referential macro:        #define FOO(x) FOO(x)
 *   4. Nested macros that eventually bottom out (non-recursive convergence)
 *   5. Self-referential macro used in complex expressions
 *   6. Chained mutual recursion (three-way cycle: P->Q->R->P)
 *   7. Function-like macro with two-argument self-reference
 *   8. Indirect mutual recursion via function-like macros
 *
 * Expected behavior: The entire compilation must complete in under 5 seconds.
 * All macros must resolve to finite token sequences. The resulting program
 * must execute and return 0.
 */

/* ===================================================================
 * Test 1: Simple self-referential object-like macro
 *
 * Per Section 0.1.2 User Example:
 *   #define A A
 *   int x = A;
 *
 * Expansion trace:
 *   1. Preprocessor encounters token A
 *   2. A is an object-like macro: replacement list is { A }
 *   3. During expansion of A, the token A in the replacement is painted
 *   4. Painted A is treated as an ordinary identifier, NOT re-expanded
 *   5. Result: identifier 'A'
 *
 * The declaration 'int x = A;' therefore refers to whatever 'A' is
 * in the C scope — we provide a local variable named A.
 * =================================================================== */

#define A A

/* ===================================================================
 * Test 2: Mutually recursive object-like macros
 *
 *   #define B C
 *   #define C B
 *
 * Expansion trace for B:
 *   1. B is a macro, replacement: { C }. Hide set: {B}
 *   2. Rescan: C is a macro, replacement: { B }. Hide set: {B, C}
 *   3. Rescan: B is in hide set — painted, not expanded
 *   4. Result: identifier 'B'
 *
 * Expansion trace for C:
 *   1. C is a macro, replacement: { B }. Hide set: {C}
 *   2. Rescan: B is a macro, replacement: { C }. Hide set: {C, B}
 *   3. Rescan: C is in hide set — painted, not expanded
 *   4. Result: identifier 'C'
 * =================================================================== */

#define B C
#define C B

/* ===================================================================
 * Test 3: Function-like self-referential macro
 *
 *   #define FOO(val) FOO(val)
 *
 * Expansion trace for FOO(5):
 *   1. FOO(5) matches function-like macro, argument val = 5
 *   2. Substitute: FOO(5). Hide set: {FOO}
 *   3. Rescan: FOO is in hide set — painted, not expanded
 *   4. Result: tokens FOO ( 5 ) — a call to the real function FOO
 *
 * We define a real function FOO so the result is a valid function call.
 * When the preprocessor sees the function definition 'int FOO(int val)',
 * FOO(int val) is matched as a macro invocation (arg val = "int val"),
 * expanded to FOO(int val) with FOO painted, resulting in the same
 * declaration text — the function definition is preserved.
 * =================================================================== */

#define FOO(val) FOO(val)

/* Real function — preprocessor expansion is idempotent for single-param
 * self-referential macros: FOO(int val) -> FOO(int val) (painted) */
static int FOO(int val)
{
    return val + 1;
}

/* ===================================================================
 * Test 4: Nested macros that eventually bottom out
 *
 * Normal (non-recursive) multi-level expansion. No paint markers
 * are needed — this validates that the paint-marker system does NOT
 * interfere with legitimate convergent macro expansion.
 *
 *   OUTER -> (MIDDLE + 2) -> ((INNER + 1) + 2) -> ((100 + 1) + 2)
 * =================================================================== */

#define INNER 100
#define MIDDLE (INNER + 1)
#define OUTER (MIDDLE + 2)

/* ===================================================================
 * Test 5: Another self-referential macro used in expressions
 *
 *   #define SELF SELF
 *
 * Identical to Test 1 but used in a more complex expression context:
 * arithmetic, comparison, and conditional logic.
 * =================================================================== */

#define SELF SELF

/* ===================================================================
 * Test 6: Three-way mutual recursion cycle
 *
 *   #define P Q
 *   #define Q R
 *   #define R P
 *
 * Expansion trace for P:
 *   1. P -> Q. Hide set: {P}
 *   2. Q -> R. Hide set: {P, Q}
 *   3. R -> P. Hide set: {P, Q, R}
 *   4. P is in hide set — painted, not expanded
 *   5. Result: identifier 'P'
 * =================================================================== */

#define P Q
#define Q R
#define R P

/* ===================================================================
 * Test 7: Function-like macro with two arguments, self-referential
 *
 *   #define BAR(a, b) BAR(b, a)
 *
 * This swaps arguments in the replacement list and self-references.
 * The paint marker prevents infinite expansion.
 *
 * We use a commutative operation (addition) in the real function
 * so the result is independent of the macro's argument reordering
 * in the function declaration.
 * =================================================================== */

#define BAR(a, b) BAR(b, a)

/* Note: BAR(int a, int b) is preprocessed as:
 *   macro args: a="int a", b="int b"
 *   substitution: BAR(int b, int a) — args swapped, BAR painted
 * The C function ends up with swapped parameter names.
 * We use a+b (commutative) to avoid depending on name ordering. */
static int BAR(int a, int b)
{
    return a + b;
}

/* ===================================================================
 * Test 8: Indirect mutual recursion via function-like macros
 *
 *   #define WRAP(z) INDIRECT(z)
 *   #define INDIRECT(z) WRAP(z)
 *
 * Expansion trace for WRAP(5):
 *   1. WRAP(5) -> INDIRECT(5). Hide set: {WRAP}
 *   2. INDIRECT(5) -> WRAP(5). Hide set: {WRAP, INDIRECT}
 *   3. WRAP is in hide set — painted, not expanded
 *   4. Result: WRAP(5) — calls the real function WRAP
 * =================================================================== */

#define WRAP(z) INDIRECT(z)
#define INDIRECT(z) WRAP(z)

/* Real function — WRAP(int z) preprocesses through the cycle:
 *   WRAP(int z) -> INDIRECT(int z) -> WRAP(int z) (painted)
 * Single parameter, so no name reordering issue. */
static int WRAP(int z)
{
    return z + 100;
}

/* ===================================================================
 * main — Exercise all test cases and verify correctness
 *
 * All variable declarations are block-scoped to avoid C's restriction
 * that file-scope initializers must be constant expressions.
 * =================================================================== */

int main(void)
{
    int errors = 0;

    /* ----------------------------------------------------------
     * Test 1: Simple self-referential macro (#define A A)
     * Per Section 0.1.2: #define A A and int x = A; must terminate
     * in <5 seconds with no hang.
     * ---------------------------------------------------------- */
    {
        int A = 42;
        int x = A;  /* A (macro) -> A (painted) -> identifier A -> 42 */
        if (x != 42) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 2: Mutually recursive macros (#define B C / #define C B)
     * ---------------------------------------------------------- */
    {
        int B = 10;
        int C = 20;
        int mutual_b = B;  /* B -> C -> B(painted) => identifier B => 10 */
        int mutual_c = C;  /* C -> B -> C(painted) => identifier C => 20 */
        if (mutual_b != 10) {
            errors++;
        }
        if (mutual_c != 20) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 3: Function-like self-referential macro
     *   #define FOO(val) FOO(val)
     * FOO(5) -> FOO(5) (painted) => calls real FOO(5) => 6
     * ---------------------------------------------------------- */
    {
        int foo_result = FOO(5);
        if (foo_result != 6) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 4: Nested macros that bottom out (convergent expansion)
     *   OUTER -> (MIDDLE + 2) -> ((INNER + 1) + 2) -> ((100+1)+2) = 103
     * No recursion here — pure convergent multi-level expansion.
     * ---------------------------------------------------------- */
    {
        int nested_result = OUTER;
        if (nested_result != 103) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 5: Self-referential macro in complex expressions
     *   #define SELF SELF
     * ---------------------------------------------------------- */
    {
        int SELF = 7;
        int complex_expr = SELF * 2 + 1;  /* SELF(painted) => 7; 7*2+1 = 15 */
        if (complex_expr != 15) {
            errors++;
        }
        /* Also test in a conditional context */
        int cond_result = (SELF > 3) ? SELF : 0;  /* 7 > 3 ? 7 : 0 = 7 */
        if (cond_result != 7) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 6: Three-way mutual recursion cycle
     *   #define P Q / #define Q R / #define R P
     * ---------------------------------------------------------- */
    {
        int P = 111;
        int Q = 222;
        int R = 333;
        int three_way_p = P;  /* P -> Q -> R -> P(painted) => 111 */
        int three_way_q = Q;  /* Q -> R -> P -> Q(painted) => 222 */
        int three_way_r = R;  /* R -> P -> Q -> R(painted) => 333 */
        if (three_way_p != 111) {
            errors++;
        }
        if (three_way_q != 222) {
            errors++;
        }
        if (three_way_r != 333) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 7: Two-argument function-like self-referential macro
     *   #define BAR(a, b) BAR(b, a)
     * BAR(1, 2) -> BAR(2, 1) (painted) => calls real BAR(2, 1)
     * BAR uses addition (commutative), so 2+1 = 3
     * ---------------------------------------------------------- */
    {
        int bar_result = BAR(1, 2);
        if (bar_result != 3) {
            errors++;
        }
    }

    /* ----------------------------------------------------------
     * Test 8: Indirect mutual recursion via function-like macros
     *   #define WRAP(z) INDIRECT(z)
     *   #define INDIRECT(z) WRAP(z)
     * WRAP(5) -> INDIRECT(5) -> WRAP(5)(painted) => calls WRAP(5) => 105
     * ---------------------------------------------------------- */
    {
        int wrap_result = WRAP(5);
        if (wrap_result != 105) {
            errors++;
        }
    }

    return errors;
}
