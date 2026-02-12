/*
 * tests/fixtures/stmt_expr.c
 *
 * GCC Statement Expression Test Fixture — BCC Checkpoint 2
 *
 * Tests the GCC extension ({ ... }) — compound statements used as expressions.
 * The value of a statement expression is the value of the last expression
 * statement in the compound block.
 *
 * Statement expressions are heavily used in the Linux kernel's macro
 * definitions (min, max, container_of, clamp, etc.) to achieve type-safe,
 * single-evaluation semantics that cannot be accomplished with plain macros.
 *
 * Reference: GCC Manual — Statement Exprs
 *            Section 0.5.1 Group 3 (gcc_extensions.rs, expressions.rs)
 *            Section 0.6.1 (GCC language extension coverage)
 */

#include <stdio.h>

/* ======================================================================
 * Macros using statement expressions — Linux kernel patterns
 * ====================================================================== */

/*
 * Type-safe max macro using statement expressions and __typeof__.
 * Each argument is evaluated exactly once, avoiding double-evaluation
 * pitfalls of the naive #define max(a,b) ((a)>(b)?(a):(b)) form.
 */
#define max(a, b) ({            \
    __typeof__(a) _a = (a);     \
    __typeof__(b) _b = (b);     \
    _a > _b ? _a : _b;         \
})

/*
 * Type-safe min macro — same single-evaluation guarantee.
 */
#define min(a, b) ({            \
    __typeof__(a) _a = (a);     \
    __typeof__(b) _b = (b);     \
    _a < _b ? _a : _b;         \
})

/*
 * Swap macro — demonstrates statement expression with side effects
 * on the caller's variables through argument references.
 */
#define swap(a, b) ({           \
    __typeof__(a) _tmp = (a);   \
    (a) = (b);                  \
    (b) = _tmp;                 \
})

/*
 * Clamp macro — combines nested __typeof__ and ternary in a single
 * statement expression. Constrains val to [lo, hi].
 */
#define clamp(val, lo, hi) ({       \
    __typeof__(val) _val = (val);   \
    __typeof__(lo)  _lo  = (lo);    \
    __typeof__(hi)  _hi  = (hi);    \
    _val < _lo ? _lo :              \
        (_val > _hi ? _hi : _val);  \
})

/* ======================================================================
 * Test infrastructure
 * ====================================================================== */

static int test_failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        printf("FAIL: %s (line %d)\n", (msg), __LINE__);            \
        test_failures++;                                            \
    }                                                               \
} while (0)

/* ======================================================================
 * main — exercises every statement expression pattern
 * ====================================================================== */

int main(void) {

    /* ------------------------------------------------------------------
     * Test 1: Basic statement expression
     *   The value of the block is the value of the last expression
     *   statement (tmp * 2 == 10).
     * ------------------------------------------------------------------ */
    int a = ({ int tmp = 5; tmp * 2; });
    CHECK(a == 10, "basic stmt expr: ({ int tmp=5; tmp*2; }) should be 10");

    /* ------------------------------------------------------------------
     * Test 2: Statement expression in a macro — max / min
     *   Validates the Linux kernel's most common statement expression
     *   pattern: type-safe comparison macros.
     * ------------------------------------------------------------------ */
    {
        int x = 17, y = 42;
        int m = max(x, y);
        CHECK(m == 42, "max(17, 42) should be 42");

        int n = min(x, y);
        CHECK(n == 17, "min(17, 42) should be 17");

        /* Equal operands */
        CHECK(max(10, 10) == 10, "max(10, 10) should be 10");

        /* Negative operands */
        CHECK(min(-5, -3) == -5, "min(-5, -3) should be -5");
        CHECK(max(-5, -3) == -3, "max(-5, -3) should be -3");
    }

    /* ------------------------------------------------------------------
     * Test 3: Nested statement expressions
     *   Inner: ({ 1 + 2; }) == 3
     *   Outer: inner * 3 == 9
     * ------------------------------------------------------------------ */
    int c = ({ int inner = ({ 1 + 2; }); inner * 3; });
    CHECK(c == 9, "nested stmt expr: ({ ({ 1+2; }) * 3; }) should be 9");

    /* Three levels of nesting */
    int d = ({
        int level1 = ({
            int level2 = ({
                int level3 = 7;
                level3 + 3;     /* 10 */
            });
            level2 * 2;         /* 20 */
        });
        level1 + 1;             /* 21 */
    });
    CHECK(d == 21, "3-level nested stmt expr should be 21");

    /* ------------------------------------------------------------------
     * Test 4: Statement expression as a function argument
     *   printf receives the result of the statement expression directly.
     * ------------------------------------------------------------------ */
    printf("%d\n", ({ 42; }));

    /* Verify programmatically as well */
    {
        int arg_val = ({ 42; });
        CHECK(arg_val == 42, "stmt expr as func arg: expected 42");
    }

    /* ------------------------------------------------------------------
     * Test 5: Statement expression with side effects
     *   Variable declarations, increments, and reads inside the block.
     * ------------------------------------------------------------------ */
    {
        int counter = 0;
        int result = ({
            counter++;
            counter++;
            counter++;
            counter;            /* value is 3 */
        });
        CHECK(result == 3, "side effects: result should be 3");
        CHECK(counter == 3, "side effects: counter should be 3");
    }

    /* ------------------------------------------------------------------
     * Test 6: Statement expression with if/else control flow
     *   Exercises branching within the compound statement.
     * ------------------------------------------------------------------ */
    {
        int val = 15;
        int clamped = ({
            int r;
            if (val < 0)
                r = 0;
            else if (val > 10)
                r = 10;
            else
                r = val;
            r;
        });
        CHECK(clamped == 10,
              "control flow stmt expr: clamp 15 to [0,10] should be 10");

        val = -3;
        int clamped2 = ({
            int r;
            if (val < 0)
                r = 0;
            else if (val > 10)
                r = 10;
            else
                r = val;
            r;
        });
        CHECK(clamped2 == 0,
              "control flow stmt expr: clamp -3 to [0,10] should be 0");
    }

    /* ------------------------------------------------------------------
     * Test 7: Statement expression with a for-loop
     *   Computes the sum 1+2+...+10 = 55 inside the expression.
     * ------------------------------------------------------------------ */
    {
        int sum = ({
            int s = 0;
            int i;
            for (i = 1; i <= 10; i++)
                s += i;
            s;
        });
        CHECK(sum == 55, "loop in stmt expr: sum 1..10 should be 55");
    }

    /* ------------------------------------------------------------------
     * Test 8: Statement expression with while-loop
     *   Factorial of 6 = 720 computed via while loop.
     * ------------------------------------------------------------------ */
    {
        int fact = ({
            int result = 1;
            int n = 6;
            while (n > 1) {
                result *= n;
                n--;
            }
            result;
        });
        CHECK(fact == 720, "while loop in stmt expr: 6! should be 720");
    }

    /* ------------------------------------------------------------------
     * Test 9: Swap macro — side effects on the caller's variables
     *   Tests that the statement expression can mutate external state
     *   through macro argument references.
     * ------------------------------------------------------------------ */
    {
        int p = 100, q = 200;
        swap(p, q);
        CHECK(p == 200, "swap: p should be 200 after swap");
        CHECK(q == 100, "swap: q should be 100 after swap");
    }

    /* ------------------------------------------------------------------
     * Test 10: Clamp macro — complex nested ternary in stmt expr
     * ------------------------------------------------------------------ */
    {
        CHECK(clamp(5,  0, 10) == 5,
              "clamp(5, 0, 10) should be 5");
        CHECK(clamp(-1, 0, 10) == 0,
              "clamp(-1, 0, 10) should be 0");
        CHECK(clamp(15, 0, 10) == 10,
              "clamp(15, 0, 10) should be 10");
        CHECK(clamp(0,  0, 10) == 0,
              "clamp(0, 0, 10) should be 0");
        CHECK(clamp(10, 0, 10) == 10,
              "clamp(10, 0, 10) should be 10");
    }

    /* ------------------------------------------------------------------
     * Test 11: Variable shadowing inside statement expression
     *   An inner variable with the same name as an outer one must shadow
     *   it within the statement expression scope, without modifying the
     *   outer variable.
     * ------------------------------------------------------------------ */
    {
        int x = 100;
        int result = ({
            int x = 200;        /* shadows outer x */
            x + 1;              /* 201 */
        });
        CHECK(result == 201,
              "shadowing: inner x=200, result should be 201");
        CHECK(x == 100,
              "shadowing: outer x should remain 100");
    }

    /* ------------------------------------------------------------------
     * Test 12: Statement expression returning a pointer
     *   The last expression produces a pointer value.
     * ------------------------------------------------------------------ */
    {
        int arr[3] = {10, 20, 30};
        int *ptr = ({
            int *p = &arr[1];
            p;                  /* returns pointer to arr[1] */
        });
        CHECK(*ptr == 20, "pointer from stmt expr: *ptr should be 20");
    }

    /* ------------------------------------------------------------------
     * Test 13: Single-evaluation guarantee of max macro
     *   Ensures that each macro argument is evaluated exactly once,
     *   verifiable by post-increment side effects.
     * ------------------------------------------------------------------ */
    {
        int arr2[4] = {5, 15, 10, 25};
        int idx = 0;
        /*
         * max(arr2[idx++], 10):
         *   _a = arr2[0] (==5), idx becomes 1 — evaluated once
         *   _b = 10
         *   _a > _b => 5 > 10 => false => result is 10
         */
        int max_val = max(arr2[idx++], 10);
        CHECK(max_val == 10,
              "single-eval max: max(arr2[0]=5, 10) should be 10");
        CHECK(idx == 1,
              "single-eval max: idx should be 1 (incremented exactly once)");
    }

    /* ==================================================================
     * Summary
     * ================================================================== */
    if (test_failures == 0) {
        printf("All statement expression tests passed.\n");
        return 0;
    } else {
        printf("%d statement expression test(s) FAILED.\n", test_failures);
        return 1;
    }
}
