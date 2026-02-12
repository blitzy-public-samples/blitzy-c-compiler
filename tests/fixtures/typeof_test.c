/*
 * typeof_test.c - GCC typeof/__typeof__ extension test fixture
 *
 * Checkpoint 2 validation: Tests the typeof and __typeof__ GCC extensions
 * for type inference from variables, expressions, pointers, structs,
 * arrays, function return types, and common kernel macro patterns.
 *
 * The parser types.rs module handles typeof/__typeof__ as part of
 * GCC extension type specifier/qualifier parsing.
 *
 * Requirements covered:
 *   - typeof/__typeof__ GCC extension validation (Section 0.6.1)
 *   - Checkpoint 2 fixture — typeof test source (Section 0.2.1)
 *   - Parser type specifier extension support (Section 0.5.1 Group 3)
 */

#include <stdio.h>

/*
 * Requirement 7: typeof in macro — common Linux kernel container_of pattern.
 * Uses __builtin_offsetof (compiler builtin, no header needed) and typeof
 * to recover the enclosing struct pointer from a member pointer.
 */
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - __builtin_offsetof(type, member)))

/*
 * typeof-based safe max macro — common kernel pattern.
 * Uses typeof to evaluate each argument exactly once, preventing
 * double-evaluation bugs with side-effecting expressions.
 * Also exercises GCC statement expressions ({ ... }).
 */
#define safe_max(a, b) ({               \
    typeof(a) _max_a = (a);             \
    typeof(b) _max_b = (b);             \
    _max_a > _max_b ? _max_a : _max_b;  \
})

/*
 * typeof-based swap macro — another common kernel pattern.
 * Uses typeof to create a temporary of the correct type.
 */
#define swap(a, b) do {                 \
    typeof(a) _swap_tmp = (a);          \
    (a) = (b);                          \
    (b) = _swap_tmp;                    \
} while (0)

/* Test struct for container_of macro validation */
struct container {
    int id;
    float value;
    char name[32];
};

/* Function to test typeof with function return types */
static double compute_ratio(int numerator, int denominator) {
    if (denominator == 0) return 0.0;
    return (double)numerator / (double)denominator;
}

/* Helper function returning a pointer, for typeof testing */
static int *get_int_ptr(int *src) {
    return src;
}

int main(void) {
    int failed = 0;

    /*
     * Requirement 2: Basic typeof usage.
     * typeof(x) infers int from x, so y is declared as int with value x + 1 = 43.
     */
    int x = 42;
    typeof(x) y = x + 1;
    if (y != 43) {
        printf("FAIL: basic typeof: typeof(x) y = x + 1: expected 43, got %d\n", y);
        failed = 1;
    }
    if (sizeof(y) != sizeof(int)) {
        printf("FAIL: basic typeof: sizeof(typeof(x)) expected %zu, got %zu\n",
               sizeof(int), sizeof(y));
        failed = 1;
    }

    /*
     * Requirement 3: __typeof__ variant — functionally equivalent to typeof.
     * __typeof__ is the double-underscore form that avoids namespace pollution
     * and is usable even when strict ISO mode is active.
     */
    __typeof__(x) z = 100;
    if (z != 100) {
        printf("FAIL: __typeof__: __typeof__(x) z = 100: expected 100, got %d\n", z);
        failed = 1;
    }
    if (sizeof(z) != sizeof(int)) {
        printf("FAIL: __typeof__: sizeof(__typeof__(x)) expected %zu, got %zu\n",
               sizeof(int), sizeof(z));
        failed = 1;
    }

    /*
     * Requirement 4: typeof from expression.
     * The expression (1 + 2.0) has type double due to usual arithmetic
     * conversions (int promoted to double), so result is declared as double.
     */
    typeof(1 + 2.0) result = 3.14;
    if (sizeof(result) != sizeof(double)) {
        printf("FAIL: typeof(expr): typeof(1 + 2.0) sizeof = %zu, expected %zu\n",
               sizeof(result), sizeof(double));
        failed = 1;
    }
    if (result < 3.13 || result > 3.15) {
        printf("FAIL: typeof(expr): result = 3.14 value check failed\n");
        failed = 1;
    }

    /* Additional typeof from expression: float arithmetic */
    typeof(1.0f + 2.0f) float_result = 5.5f;
    if (sizeof(float_result) != sizeof(float)) {
        printf("FAIL: typeof(float expr): sizeof = %zu, expected %zu\n",
               sizeof(float_result), sizeof(float));
        failed = 1;
    }

    /*
     * Requirement 5: typeof with pointers.
     * typeof(p) where p is int* gives int*, so q is declared as int*.
     */
    int val = 99;
    int *p = &val;
    typeof(p) q = &val;
    if (sizeof(q) != sizeof(int *)) {
        printf("FAIL: typeof(ptr): sizeof(typeof(int*)) = %zu, expected %zu\n",
               sizeof(q), sizeof(int *));
        failed = 1;
    }
    if (*q != 99) {
        printf("FAIL: typeof(ptr): *q expected 99, got %d\n", *q);
        failed = 1;
    }

    /* Additional: typeof on pointer dereference gives the pointed-to type */
    typeof(*p) deref_val = 55;
    if (sizeof(deref_val) != sizeof(int)) {
        printf("FAIL: typeof(*ptr): sizeof = %zu, expected %zu\n",
               sizeof(deref_val), sizeof(int));
        failed = 1;
    }
    if (deref_val != 55) {
        printf("FAIL: typeof(*ptr): deref_val expected 55, got %d\n", deref_val);
        failed = 1;
    }

    /* typeof with double pointer */
    int **pp = &p;
    typeof(pp) pp2 = &p;
    if (sizeof(pp2) != sizeof(int **)) {
        printf("FAIL: typeof(double ptr): sizeof mismatch\n");
        failed = 1;
    }
    if (**pp2 != 99) {
        printf("FAIL: typeof(double ptr): **pp2 expected 99, got %d\n", **pp2);
        failed = 1;
    }

    /*
     * Requirement 6: typeof with struct.
     * Uses an anonymous struct type; typeof(s) creates s2 with the same
     * anonymous struct type, allowing access to the same members.
     */
    struct { int a; float b; } s;
    s.a = 10;
    s.b = 20.5f;
    typeof(s) s2;
    s2.a = s.a + 5;
    s2.b = s.b * 2.0f;
    if (s2.a != 15) {
        printf("FAIL: typeof(struct): s2.a expected 15, got %d\n", s2.a);
        failed = 1;
    }
    if (s2.b < 40.9f || s2.b > 41.1f) {
        printf("FAIL: typeof(struct): s2.b expected ~41.0\n");
        failed = 1;
    }
    if (sizeof(s2) != sizeof(s)) {
        printf("FAIL: typeof(struct): sizeof mismatch: %zu vs %zu\n",
               sizeof(s2), sizeof(s));
        failed = 1;
    }

    /* typeof with named struct */
    struct container c1;
    c1.id = 100;
    c1.value = 2.5f;
    typeof(c1) c1_copy;
    c1_copy.id = c1.id;
    c1_copy.value = c1.value;
    if (c1_copy.id != 100 || c1_copy.value < 2.4f || c1_copy.value > 2.6f) {
        printf("FAIL: typeof(named struct): copy mismatch\n");
        failed = 1;
    }

    /* typeof with struct member access — infers member type */
    typeof(c1.id) member_int = 42;
    if (sizeof(member_int) != sizeof(int)) {
        printf("FAIL: typeof(struct.member): int member sizeof = %zu, expected %zu\n",
               sizeof(member_int), sizeof(int));
        failed = 1;
    }
    typeof(c1.value) member_float = 1.5f;
    if (sizeof(member_float) != sizeof(float)) {
        printf("FAIL: typeof(struct.member): float member sizeof = %zu, expected %zu\n",
               sizeof(member_float), sizeof(float));
        failed = 1;
    }

    /*
     * Requirement 7: typeof in container_of macro.
     * Recovers a struct container pointer from a pointer to its value member.
     */
    struct container c;
    c.id = 42;
    c.value = 3.14f;
    c.name[0] = 'X';
    c.name[1] = '\0';

    float *vptr = &c.value;
    struct container *recovered = container_of(vptr, struct container, value);
    if (recovered->id != 42) {
        printf("FAIL: container_of: recovered->id = %d, expected 42\n", recovered->id);
        failed = 1;
    }
    if (recovered->name[0] != 'X') {
        printf("FAIL: container_of: recovered->name[0] = '%c', expected 'X'\n",
               recovered->name[0]);
        failed = 1;
    }

    /* container_of from a different member */
    char *nptr = c.name;
    struct container *recovered2 = container_of(nptr, struct container, name);
    if (recovered2->id != 42) {
        printf("FAIL: container_of(name): recovered2->id = %d, expected 42\n",
               recovered2->id);
        failed = 1;
    }

    /*
     * Requirement 8: typeof with array element.
     * typeof(arr[0]) infers the element type (int), not the array type.
     */
    int arr[10];
    arr[0] = 777;
    arr[5] = 555;
    typeof(arr[0]) elem = arr[0];
    if (sizeof(elem) != sizeof(int)) {
        printf("FAIL: typeof(arr[0]): sizeof = %zu, expected %zu\n",
               sizeof(elem), sizeof(int));
        failed = 1;
    }
    if (elem != 777) {
        printf("FAIL: typeof(arr[0]): elem expected 777, got %d\n", elem);
        failed = 1;
    }

    /* typeof of array itself preserves array type */
    typeof(arr) arr2;
    arr2[0] = 111;
    arr2[9] = 999;
    if (sizeof(arr2) != sizeof(arr)) {
        printf("FAIL: typeof(arr): sizeof = %zu, expected %zu\n",
               sizeof(arr2), sizeof(arr));
        failed = 1;
    }
    if (sizeof(arr2) != 10 * sizeof(int)) {
        printf("FAIL: typeof(arr): not 10 ints: sizeof = %zu\n", sizeof(arr2));
        failed = 1;
    }
    if (arr2[0] != 111 || arr2[9] != 999) {
        printf("FAIL: typeof(arr): arr2 element access failed\n");
        failed = 1;
    }

    /*
     * Additional: typeof with function return type.
     * typeof(compute_ratio(1, 2)) infers double from the function's return type.
     * The expression inside typeof is not evaluated; only its type matters.
     */
    typeof(compute_ratio(1, 2)) ratio = compute_ratio(355, 113);
    if (sizeof(ratio) != sizeof(double)) {
        printf("FAIL: typeof(func()): sizeof = %zu, expected %zu\n",
               sizeof(ratio), sizeof(double));
        failed = 1;
    }
    /* 355/113 ≈ 3.14159... */
    if (ratio < 3.14 || ratio > 3.15) {
        printf("FAIL: typeof(func()): ratio value check failed\n");
        failed = 1;
    }

    /* typeof with function returning pointer */
    typeof(get_int_ptr(&val)) func_ptr_result = &val;
    if (*func_ptr_result != 99) {
        printf("FAIL: typeof(func_returning_ptr): expected 99, got %d\n",
               *func_ptr_result);
        failed = 1;
    }

    /*
     * Additional: __typeof__ with long long expression.
     * Ensures large integer types are correctly inferred.
     */
    long long big = 1000000000LL;
    __typeof__(big * 2) big2 = big * 3;
    if (sizeof(big2) != sizeof(long long)) {
        printf("FAIL: __typeof__(long long expr): sizeof = %zu, expected %zu\n",
               sizeof(big2), sizeof(long long));
        failed = 1;
    }
    if (big2 != 3000000000LL) {
        printf("FAIL: __typeof__(long long expr): value mismatch\n");
        failed = 1;
    }

    /*
     * Additional: typeof-based swap macro.
     * Demonstrates typeof creating a correctly-typed temporary variable.
     */
    int swap_a = 10, swap_b = 20;
    swap(swap_a, swap_b);
    if (swap_a != 20 || swap_b != 10) {
        printf("FAIL: swap: expected (20, 10), got (%d, %d)\n", swap_a, swap_b);
        failed = 1;
    }

    /* Swap with different type — typeof adapts */
    double swap_da = 1.5, swap_db = 2.5;
    swap(swap_da, swap_db);
    if (swap_da < 2.4 || swap_da > 2.6 || swap_db < 1.4 || swap_db > 1.6) {
        printf("FAIL: swap(double): type-adapted swap failed\n");
        failed = 1;
    }

    /*
     * Additional: typeof-based safe_max macro.
     * Uses typeof + statement expression for type-safe comparison.
     */
    int max_result = safe_max(42, 17);
    if (max_result != 42) {
        printf("FAIL: safe_max(42, 17): expected 42, got %d\n", max_result);
        failed = 1;
    }
    int max_result2 = safe_max(-5, 10);
    if (max_result2 != 10) {
        printf("FAIL: safe_max(-5, 10): expected 10, got %d\n", max_result2);
        failed = 1;
    }

    /*
     * Additional: typeof with char types.
     * Ensures single-byte types are correctly inferred.
     */
    char ch = 'A';
    typeof(ch) ch2 = 'Z';
    if (sizeof(ch2) != sizeof(char)) {
        printf("FAIL: typeof(char): sizeof = %zu, expected %zu\n",
               sizeof(ch2), sizeof(char));
        failed = 1;
    }
    if (ch2 != 'Z') {
        printf("FAIL: typeof(char): ch2 expected 'Z', got '%c'\n", ch2);
        failed = 1;
    }

    /*
     * Additional: typeof with unsigned types.
     * Ensures signedness is preserved through typeof.
     */
    unsigned int ux = 0xFFFFFFFFu;
    typeof(ux) uy = 100u;
    if (sizeof(uy) != sizeof(unsigned int)) {
        printf("FAIL: typeof(unsigned int): sizeof mismatch\n");
        failed = 1;
    }
    /* If typeof correctly preserves unsigned, ux should be positive (large) */
    if (ux < uy) {
        printf("FAIL: typeof(unsigned): signedness not preserved\n");
        failed = 1;
    }

    /*
     * Additional: nested typeof — typeof of a typeof-declared variable.
     * typeof(typeof(x) *) should produce int* (pointer to the type of x).
     */
    typeof(typeof(x) *) nested_ptr = &x;
    if (*nested_ptr != 42) {
        printf("FAIL: nested typeof: *nested_ptr expected 42, got %d\n",
               *nested_ptr);
        failed = 1;
    }
    if (sizeof(nested_ptr) != sizeof(int *)) {
        printf("FAIL: nested typeof: sizeof mismatch\n");
        failed = 1;
    }

    /* ---- Final result ---- */
    if (failed) {
        printf("typeof_test: FAILED\n");
        return 1;
    }

    printf("typeof_test: PASSED\n");
    return 0;
}
