/*
 * tests/fixtures/stmt_expr.c — GCC Statement Expression Test (Checkpoint 2)
 *
 * Validates GCC statement expression support ({ ... }). Statement expressions
 * are a GCC extension heavily used in the Linux kernel (e.g., min(), max()
 * macros). The value of a statement expression is the value of the last
 * expression statement within the braces.
 *
 * This test exercises:
 *   - Basic statement expressions with a single expression
 *   - Multi-statement expressions with intermediate variables
 *   - Classic kernel-style MIN/MAX macros using __typeof__ and stmt exprs
 *   - Nested statement expressions (stmt expr inside stmt expr)
 *   - Statement expressions containing control flow (if/else)
 *   - Statement expressions used as function arguments
 *   - Statement expressions with side effects
 *   - Correct value propagation from the last expression
 *
 * Compile: gcc -std=gnu11 -o stmt_expr stmt_expr.c && ./stmt_expr
 *      or: ./bcc -o stmt_expr stmt_expr.c && ./stmt_expr
 * Expected: prints "stmt_expr OK\n" and exits with code 0
 */

#include <stdio.h>

/* Classic Linux kernel-style min/max macros using statement expressions */
#define MAX(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b; \
})

#define MIN(a, b) ({ \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b; \
})

/* Clamp macro — nested statement expression usage */
#define CLAMP(val, lo, hi) ({ \
    __typeof__(val) _val = (val); \
    __typeof__(lo) _lo = (lo); \
    __typeof__(hi) _hi = (hi); \
    _val < _lo ? _lo : (_val > _hi ? _hi : _val); \
})

/* Swap macro using statement expression with side effects */
#define SWAP(a, b) ({ \
    __typeof__(a) _tmp = (a); \
    (a) = (b); \
    (b) = _tmp; \
})

int main(void) {
    /* Test 1: Basic statement expression — single expression yields value */
    int x = ({ 1 + 2; });
    if (x != 3) { printf("FAIL: basic stmt expr, got %d expected 3\n", x); return 1; }

    /* Test 2: Statement expression with multiple statements */
    int y = ({
        int tmp = 10;
        tmp *= 2;
        tmp + 5;
    });
    if (y != 25) { printf("FAIL: multi-stmt expr, got %d expected 25\n", y); return 1; }

    /* Test 3: MAX macro using statement expressions and __typeof__ */
    int a = 42, b = 17;
    int m = MAX(a, b);
    if (m != 42) { printf("FAIL: MAX macro, got %d expected 42\n", m); return 1; }

    /* Test 4: MIN macro using statement expressions and __typeof__ */
    int n = MIN(a, b);
    if (n != 17) { printf("FAIL: MIN macro, got %d expected 17\n", n); return 1; }

    /* Test 5: Nested statement expressions */
    int z = ({ int inner = ({ 5 + 3; }); inner * 2; });
    if (z != 16) { printf("FAIL: nested stmt expr, got %d expected 16\n", z); return 1; }

    /* Test 6: Statement expression with control flow (if/else) */
    int cf = ({
        int result;
        int val = 100;
        if (val > 50) {
            result = val - 50;
        } else {
            result = val + 50;
        }
        result;
    });
    if (cf != 50) { printf("FAIL: control flow stmt expr, got %d expected 50\n", cf); return 1; }

    /* Test 7: Statement expression used as a function argument */
    int arg_test = ({
        int v1 = 10;
        int v2 = 20;
        v1 + v2;
    });
    char buf[64];
    int printed = sprintf(buf, "%d", arg_test);
    if (printed != 2) { printf("FAIL: stmt expr as arg, sprintf returned %d\n", printed); return 1; }
    if (buf[0] != '3' || buf[1] != '0') {
        printf("FAIL: stmt expr as arg, got '%s' expected '30'\n", buf);
        return 1;
    }

    /* Test 8: CLAMP macro — compound statement expression usage */
    int clamped_low = CLAMP(-5, 0, 100);
    if (clamped_low != 0) { printf("FAIL: CLAMP low, got %d expected 0\n", clamped_low); return 1; }

    int clamped_high = CLAMP(200, 0, 100);
    if (clamped_high != 100) { printf("FAIL: CLAMP high, got %d expected 100\n", clamped_high); return 1; }

    int clamped_mid = CLAMP(42, 0, 100);
    if (clamped_mid != 42) { printf("FAIL: CLAMP mid, got %d expected 42\n", clamped_mid); return 1; }

    /* Test 9: SWAP macro — statement expression with side effects */
    int s1 = 111, s2 = 222;
    SWAP(s1, s2);
    if (s1 != 222) { printf("FAIL: SWAP s1, got %d expected 222\n", s1); return 1; }
    if (s2 != 111) { printf("FAIL: SWAP s2, got %d expected 111\n", s2); return 1; }

    /* Test 10: Statement expression evaluating to void (no value used) */
    ({
        int dummy = 999;
        (void)dummy;
    });

    /* Test 11: Deeply nested statement expressions */
    int deep = ({
        int level1 = ({
            int level2 = ({
                int level3 = 7;
                level3 * 3;
            });
            level2 + 1;
        });
        level1 * 2;
    });
    if (deep != 44) { printf("FAIL: deep nested, got %d expected 44\n", deep); return 1; }

    /* Test 12: Statement expression with a loop */
    int loop_result = ({
        int sum = 0;
        int i;
        for (i = 1; i <= 10; i++) {
            sum += i;
        }
        sum;
    });
    if (loop_result != 55) { printf("FAIL: loop stmt expr, got %d expected 55\n", loop_result); return 1; }

    /* Test 13: MAX/MIN with expressions as arguments (no double evaluation) */
    int counter = 0;
    int inc_val = ({
        counter++;
        42;
    });
    if (counter != 1) { printf("FAIL: side-effect count, got %d expected 1\n", counter); return 1; }
    if (inc_val != 42) { printf("FAIL: side-effect value, got %d expected 42\n", inc_val); return 1; }

    printf("stmt_expr OK\n");
    return 0;
}
