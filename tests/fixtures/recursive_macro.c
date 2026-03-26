/*
 * Recursive macro termination test.
 *
 * Per BCC spec: #define A A and usage must terminate in <5 seconds, no hang.
 * The paint-marker system (src/frontend/preprocessor/paint_marker.rs)
 * suppresses re-expansion of self-referential macros.
 *
 * C standard reference (ISO/IEC 9899:2011 §6.10.3.4):
 *   "If the name of the macro being replaced is found during [rescanning],
 *    it is not replaced."
 *   This is the paint-marker invariant that BCC must implement.
 *
 * Test strategy:
 *   1. Define several self-referential macros (#define X X patterns)
 *   2. Use them in code to force the preprocessor to attempt expansion
 *   3. Verify the preprocessor terminates (no infinite loop)
 *   4. Verify the program compiles, links, and runs with exit code 0
 *
 * This file exercises:
 *   - src/frontend/preprocessor/paint_marker.rs (paint-marker protection)
 *   - src/frontend/preprocessor/macro_expander.rs (expansion termination)
 *   - The 512-depth recursion limit enforcement (should not be reached here)
 */

/* ===================================================================
 * Self-referential macro definitions
 * ===================================================================
 *
 * #define A A
 *
 * When the preprocessor encounters token A:
 *   1. It recognizes A as a macro and begins expansion.
 *   2. The replacement list is: A
 *   3. During rescanning, the resulting token A is "painted"
 *      (marked as having been produced by expansion of macro A).
 *   4. The painted token A is NOT eligible for re-expansion.
 *   5. Result: the identifier A (an ordinary token, not a macro invocation).
 *
 * Without paint-marker protection, this would loop infinitely:
 *   A -> A -> A -> A -> ... (never terminates)
 */
#define A A

/*
 * Indirect self-reference: B's replacement list mentions B.
 * Same paint-marker logic applies.
 */
#define B B

/*
 * Self-referential with additional tokens.
 * SELF expands to SELF + 0, where SELF is painted and not re-expanded.
 * Result after preprocessing: SELF + 0  (SELF is an identifier)
 */
#define SELF SELF + 0

/*
 * Mutually-referential macros.
 * #define X Y
 * #define Y X
 *
 * Expansion of X:
 *   X -> Y  (X is now in the "hide set" / painted)
 *   Y -> X  (Y is now in the hide set, X is still painted from outer)
 *   X is painted -> not re-expanded
 *   Result: X
 *
 * This tests that the paint-marker system handles transitive
 * self-reference correctly.
 */
#define X Y
#define Y X

/*
 * More complex chain: P -> Q -> R -> P
 * Expansion of P:
 *   P -> Q  (P painted)
 *   Q -> R  (Q painted)
 *   R -> P  (R painted, P still painted from outermost)
 *   P is painted -> not re-expanded
 *   Result: P
 */
#define P Q
#define Q R
#define R P

/* ===================================================================
 * Global variable using the self-referential macro A
 * ===================================================================
 *
 * After preprocessing, this line becomes:
 *   int x = A;
 * where A is now just an identifier (not a macro — it was painted).
 *
 * To make this compilable and linkable, we need A to resolve to
 * something at compile time. We handle this in main() below by
 * declaring a local variable named A that shadows the global usage.
 */
int x = 0;  /* Initialize to 0; we will set it in main() */

int main(void) {
    /*
     * Declare a local variable named 'A'.
     * This shadows any identifier 'A' produced by macro expansion.
     * After preprocessing, references to macro A in this scope
     * produce the painted identifier A, which resolves to this local.
     */
    int A = 42;

    /*
     * Test 1: Direct use of self-referential macro A.
     * Preprocessor expands A -> A (painted, stops).
     * Compiler sees: x = A; where A is the local variable (42).
     */
    x = A;
    if (x != 42) {
        return 1;
    }

    /*
     * Test 2: Use of self-referential macro B.
     * Preprocessor expands B -> B (painted, stops).
     * Compiler sees: int B = 100; (declaration of local B)
     * Then: int b_val = B; where B is the local variable.
     */
    int B = 100;
    int b_val = B;
    if (b_val != 100) {
        return 2;
    }

    /*
     * Test 3: Mutually-referential macros X and Y.
     * Expansion of X: X -> Y -> X (painted, stops) => X
     * Expansion of Y: Y -> X -> Y (painted, stops) => Y
     * We declare locals to make them resolve.
     */
    int X = 10;
    int Y = 20;
    int x_val = X;  /* Preprocessor: X -> Y -> X (painted) => X => local X = 10 */
    int y_val = Y;  /* Preprocessor: Y -> X -> Y (painted) => Y => local Y = 20 */
    if (x_val != 10) {
        return 3;
    }
    if (y_val != 20) {
        return 4;
    }

    /*
     * Test 4: Transitive chain P -> Q -> R -> P.
     * Expansion of P: P -> Q -> R -> P (painted, stops) => P
     * We declare a local P to make it resolve.
     */
    int P = 77;
    int p_val = P;  /* After preprocessing: P (painted identifier) => local P */
    if (p_val != 77) {
        return 5;
    }

    /*
     * Test 5: Self-referential macro with extra tokens (SELF + 0).
     * SELF expands to: SELF + 0  (SELF is painted, not re-expanded)
     *
     * We must #undef SELF first, declare the variable, then re-define,
     * because the multi-token expansion would break declaration syntax.
     * This pattern tests that the painted SELF identifier in the
     * expansion result correctly resolves to our local variable.
     */
#undef SELF
    int SELF = 200;
#define SELF SELF + 0
    /* SELF expands to: SELF + 0 (SELF is painted identifier, resolves to local = 200) */
    int self_val = SELF;  /* After preprocessing: SELF + 0 => 200 + 0 => 200 */
    if (self_val != 200) {
        return 6;
    }

    /*
     * All tests passed.
     * The critical validation is that we reached this point at all —
     * if the paint-marker system were broken, the preprocessor would
     * have looped infinitely on one of the macro expansions above,
     * and compilation would never complete.
     *
     * The checkpoint harness (tests/checkpoint2_language.rs) enforces
     * a 5-second timeout to catch any such infinite loops.
     */
    return 0;
}
