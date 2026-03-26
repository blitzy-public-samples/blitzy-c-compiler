// Regression test: typeof on statement expressions
// Bug: __typeof__(({type var = val; var;})) resolved to int instead of the
// actual type of the last expression in the compound block.
#include <stdio.h>

#define safe_abs(x) ({                \
    __typeof__(x) _val = (x);        \
    _val < 0 ? -_val : _val;         \
})

int main(void) {
    int ok = 1;

    // Test 1: typeof on statement expression with long long
    long long v1 = -8945932830116527018LL;
    long long a1 = safe_abs(v1);
    // The absolute value should fit in long long
    if (a1 != 8945932830116527018LL) ok = 0;

    // Test 2: typeof on nested statement expression
    long long v2 = -100LL;
    long long r2 = ({
        __typeof__(v2) _a = v2;
        __typeof__(_a) _b = _a;
        _b < 0 ? -_b : _b;
    });
    if (r2 != 100LL) ok = 0;

    // Test 3: sizeof verification (typeof must produce correct size)
    long long v3 = 42LL;
    if (sizeof(({__typeof__(v3) _t = v3; _t;})) != sizeof(long long)) ok = 0;

    if (ok) printf("typeof_stmt_expr OK\n");
    else printf("FAIL\n");
    return !ok;
}
