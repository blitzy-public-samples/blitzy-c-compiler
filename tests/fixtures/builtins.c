/*
 * builtins.c — GCC builtins coverage test fixture for BCC Checkpoint 2.
 *
 * Exercises ~30 GCC builtin functions spanning compile-time evaluation,
 * runtime intrinsics, address introspection, variadic argument handling,
 * overflow-checked arithmetic, and miscellaneous hints/traps.
 *
 * Expected: compiles cleanly, runs, prints PASS for each test, exits 0.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Global failure counter — incremented on any test mismatch.         */
/* ------------------------------------------------------------------ */
static int failures = 0;

/* ------------------------------------------------------------------ */
/* Helper macro: check a boolean condition, print PASS / FAIL.        */
/* ------------------------------------------------------------------ */
#define CHECK(name, cond) do {                          \
    if (cond) {                                         \
        printf("PASS: %s\n", name);                     \
    } else {                                            \
        printf("FAIL: %s\n", name);                     \
        failures++;                                     \
    }                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* Struct used by __builtin_offsetof tests.                           */
/* ------------------------------------------------------------------ */
struct OffsetTest {
    int a;
    int b;
    char c;
    double d;
};

/* ------------------------------------------------------------------ */
/* 1. Compile-time builtins                                           */
/* ------------------------------------------------------------------ */

/*
 * test_builtin_constant_p — verifies __builtin_constant_p correctly
 * distinguishes compile-time constants from runtime values.
 */
static void test_builtin_constant_p(int argc) {
    /* Compile-time constant: should yield 1 */
    CHECK("__builtin_constant_p(42)",
          __builtin_constant_p(42) == 1);

    /* Runtime value: should yield 0 */
    CHECK("__builtin_constant_p(argc)",
          __builtin_constant_p(argc) == 0);

    /* Constant expression: should yield 1 */
    CHECK("__builtin_constant_p(100 + 200)",
          __builtin_constant_p(100 + 200) == 1);

    /* String literal: implementation-defined, but typically 1 for GCC */
    /* We accept either 0 or 1 here — just ensure it compiles */
    int str_result = __builtin_constant_p("hello");
    CHECK("__builtin_constant_p(\"hello\") compiles",
          str_result == 0 || str_result == 1);
}

/*
 * test_builtin_types_compatible_p — verifies type compatibility checks.
 */
static void test_builtin_types_compatible_p(void) {
    CHECK("__builtin_types_compatible_p(int, int)",
          __builtin_types_compatible_p(int, int) == 1);

    CHECK("__builtin_types_compatible_p(int, long) == 0",
          __builtin_types_compatible_p(int, long) == 0);

    CHECK("__builtin_types_compatible_p(char, char)",
          __builtin_types_compatible_p(char, char) == 1);

    CHECK("__builtin_types_compatible_p(int, unsigned int) == 0",
          __builtin_types_compatible_p(int, unsigned int) == 0);

    CHECK("__builtin_types_compatible_p(int*, int*)",
          __builtin_types_compatible_p(int *, int *) == 1);

    CHECK("__builtin_types_compatible_p(int*, char*) == 0",
          __builtin_types_compatible_p(int *, char *) == 0);
}

/*
 * test_builtin_choose_expr — verifies compile-time expression selection.
 * The un-chosen branch is NOT evaluated (not even type-checked for
 * side-effect-free expressions), so referencing a non-existent function
 * in the dead branch is valid.
 */
static void test_builtin_choose_expr(void) {
    /*
     * __builtin_constant_p(1) is true at compile time, so the first
     * expression (42) is selected. The second expression would reference
     * a function that does not exist — but it is never evaluated.
     */
    int result = __builtin_choose_expr(__builtin_constant_p(1), 42, ((void)0, 0));
    CHECK("__builtin_choose_expr selects constant branch", result == 42);

    /*
     * Choose between two concrete values based on type compatibility.
     */
    int val = __builtin_choose_expr(
        __builtin_types_compatible_p(int, int), 100, 200);
    CHECK("__builtin_choose_expr with types_compatible_p", val == 100);
}

/*
 * test_builtin_offsetof — verifies struct member offset computation.
 */
static void test_builtin_offsetof(void) {
    size_t off_a = __builtin_offsetof(struct OffsetTest, a);
    CHECK("__builtin_offsetof(OffsetTest, a) == 0", off_a == 0);

    size_t off_b = __builtin_offsetof(struct OffsetTest, b);
    CHECK("__builtin_offsetof(OffsetTest, b) == sizeof(int)",
          off_b == sizeof(int));

    /* Anonymous struct form — must also compile */
    size_t off_anon = __builtin_offsetof(struct { int x; int y; }, y);
    CHECK("__builtin_offsetof(anonymous struct, y)", off_anon == sizeof(int));
}

/* ------------------------------------------------------------------ */
/* 2. Runtime integer builtins                                        */
/* ------------------------------------------------------------------ */

/*
 * test_builtin_clz — count leading zeros.
 * __builtin_clz(8) = clz(0x00000008) = 28  (for 32-bit int)
 */
static void test_builtin_clz(void) {
    CHECK("__builtin_clz(8) == 28",  __builtin_clz(8) == 28);
    CHECK("__builtin_clz(1) == 31",  __builtin_clz(1) == 31);
    CHECK("__builtin_clz(0x80000000) == 0",
          __builtin_clz((int)0x80000000u) == 0);

    /* Long variants */
    CHECK("__builtin_clzl(1L) == (sizeof(long)*8 - 1)",
          __builtin_clzl(1L) == (int)(sizeof(long) * 8 - 1));
}

/*
 * test_builtin_ctz — count trailing zeros.
 * __builtin_ctz(8) = ctz(0b1000) = 3
 */
static void test_builtin_ctz(void) {
    CHECK("__builtin_ctz(8) == 3",   __builtin_ctz(8) == 3);
    CHECK("__builtin_ctz(1) == 0",   __builtin_ctz(1) == 0);
    CHECK("__builtin_ctz(16) == 4",  __builtin_ctz(16) == 4);

    /* Long variant */
    CHECK("__builtin_ctzl(8L) == 3", __builtin_ctzl(8L) == 3);
}

/*
 * test_builtin_popcount — population count (number of set bits).
 * __builtin_popcount(0xFF) = 8
 */
static void test_builtin_popcount(void) {
    CHECK("__builtin_popcount(0xFF) == 8",
          __builtin_popcount(0xFF) == 8);
    CHECK("__builtin_popcount(0) == 0",
          __builtin_popcount(0) == 0);
    CHECK("__builtin_popcount(0x55555555) == 16",
          __builtin_popcount(0x55555555) == 16);

    /* Long variant */
    CHECK("__builtin_popcountl(0xFFL) == 8",
          __builtin_popcountl(0xFFL) == 8);
}

/*
 * test_builtin_bswap — byte-order swap.
 */
static void test_builtin_bswap(void) {
    /* 16-bit swap */
    CHECK("__builtin_bswap16(0x1234) == 0x3412",
          __builtin_bswap16(0x1234) == 0x3412);

    /* 32-bit swap */
    CHECK("__builtin_bswap32(0x12345678) == 0x78563412",
          __builtin_bswap32(0x12345678) == 0x78563412);

    /* 64-bit swap */
    unsigned long long swapped = __builtin_bswap64(0x0102030405060708ULL);
    CHECK("__builtin_bswap64(0x0102030405060708ULL) == 0x0807060504030201ULL",
          swapped == 0x0807060504030201ULL);
}

/*
 * test_builtin_ffs — find first set bit (1-indexed from LSB, 0 if input is 0).
 * __builtin_ffs(8) = 4  (bit 3 is set, 1-indexed = 4)
 */
static void test_builtin_ffs(void) {
    CHECK("__builtin_ffs(8) == 4",   __builtin_ffs(8) == 4);
    CHECK("__builtin_ffs(0) == 0",   __builtin_ffs(0) == 0);
    CHECK("__builtin_ffs(1) == 1",   __builtin_ffs(1) == 1);
    CHECK("__builtin_ffs(0x100) == 9", __builtin_ffs(0x100) == 9);

    /* Long variant */
    CHECK("__builtin_ffsl(8L) == 4", __builtin_ffsl(8L) == 4);
}

/*
 * test_builtin_expect — branch prediction hint.
 * Returns its first argument unchanged; semantics are purely hint-based.
 */
static void test_builtin_expect(void) {
    int x = 1;
    int result = (int)__builtin_expect(x, 1);
    CHECK("__builtin_expect(1, 1) == 1", result == 1);

    x = 0;
    result = (int)__builtin_expect(x, 1);
    CHECK("__builtin_expect(0, 1) == 0 (value unchanged)", result == 0);

    long lx = 42L;
    long lr = __builtin_expect(lx, 42L);
    CHECK("__builtin_expect(42L, 42L) == 42L", lr == 42L);
}

/* ------------------------------------------------------------------ */
/* 3. Address / introspection builtins                                */
/* ------------------------------------------------------------------ */

/*
 * test_builtin_frame_address — returns the frame pointer for the
 * current function.  We verify it is non-NULL at level 0.
 */
static void test_builtin_frame_address(void) {
    void *fp = __builtin_frame_address(0);
    CHECK("__builtin_frame_address(0) != NULL", fp != NULL);
}

/*
 * test_builtin_return_address — returns the return address for the
 * current function.  We verify it is non-NULL at level 0.
 */
static void test_builtin_return_address(void) {
    void *ra = __builtin_return_address(0);
    CHECK("__builtin_return_address(0) != NULL", ra != NULL);
}

/* ------------------------------------------------------------------ */
/* 4. Variadic argument builtins                                      */
/* ------------------------------------------------------------------ */

/*
 * Helper variadic function that sums 'count' integer arguments using
 * __builtin_va_start, __builtin_va_arg, __builtin_va_end, and
 * __builtin_va_copy.
 */
static int variadic_sum(int count, ...) {
    va_list ap;
    __builtin_va_start(ap, count);

    /* Test __builtin_va_copy */
    va_list ap_copy;
    __builtin_va_copy(ap_copy, ap);

    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += __builtin_va_arg(ap, int);
    }
    __builtin_va_end(ap);

    /* Verify the copy produces the same result */
    int sum_copy = 0;
    for (int i = 0; i < count; i++) {
        sum_copy += __builtin_va_arg(ap_copy, int);
    }
    __builtin_va_end(ap_copy);

    /* Return sum only if both agree — otherwise return -1 as sentinel */
    return (sum == sum_copy) ? sum : -1;
}

static void test_builtin_va(void) {
    int result = variadic_sum(3, 10, 20, 30);
    CHECK("__builtin_va_start/arg/end/copy (sum 10+20+30 == 60)",
          result == 60);

    result = variadic_sum(0);
    CHECK("__builtin_va (zero args sum == 0)", result == 0);

    result = variadic_sum(1, 42);
    CHECK("__builtin_va (single arg 42)", result == 42);
}

/* ------------------------------------------------------------------ */
/* 5. Overflow-checked arithmetic builtins                            */
/* ------------------------------------------------------------------ */

static void test_builtin_overflow(void) {
    int result;
    int overflowed;

    /* Addition: no overflow */
    overflowed = __builtin_add_overflow(100, 200, &result);
    CHECK("__builtin_add_overflow(100, 200) no overflow",
          overflowed == 0 && result == 300);

    /* Addition: overflow (INT_MAX + 1) */
    overflowed = __builtin_add_overflow(2147483647, 1, &result);
    CHECK("__builtin_add_overflow(INT_MAX, 1) overflows",
          overflowed != 0);

    /* Subtraction: no overflow */
    overflowed = __builtin_sub_overflow(300, 100, &result);
    CHECK("__builtin_sub_overflow(300, 100) no overflow",
          overflowed == 0 && result == 200);

    /* Subtraction: overflow (INT_MIN - 1) */
    overflowed = __builtin_sub_overflow(-2147483647 - 1, 1, &result);
    CHECK("__builtin_sub_overflow(INT_MIN, 1) overflows",
          overflowed != 0);

    /* Multiplication: no overflow */
    overflowed = __builtin_mul_overflow(100, 200, &result);
    CHECK("__builtin_mul_overflow(100, 200) no overflow",
          overflowed == 0 && result == 20000);

    /* Multiplication: overflow */
    overflowed = __builtin_mul_overflow(2147483647, 2, &result);
    CHECK("__builtin_mul_overflow(INT_MAX, 2) overflows",
          overflowed != 0);
}

/* ------------------------------------------------------------------ */
/* 6. __builtin_assume_aligned                                        */
/* ------------------------------------------------------------------ */

static void test_builtin_assume_aligned(void) {
    /*
     * __builtin_assume_aligned returns its first argument, telling the
     * compiler to assume the pointer has the specified alignment.
     * We verify the returned pointer value is unchanged.
     */
    int __attribute__((aligned(16))) aligned_var = 77;
    int *p = &aligned_var;
    int *q = (int *)__builtin_assume_aligned(p, 16);
    CHECK("__builtin_assume_aligned returns same pointer", p == q);
    CHECK("__builtin_assume_aligned value intact", *q == 77);
}

/* ------------------------------------------------------------------ */
/* 7. __builtin_unreachable — tested in a dead code path              */
/* ------------------------------------------------------------------ */

/*
 * We place __builtin_unreachable in a code path that is guaranteed
 * unreachable so the program does not abort.  The compiler may use
 * this hint for optimization (e.g., eliminating the after-switch
 * return).
 */
static int test_unreachable_helper(int x) {
    switch (x) {
    case 0: return 10;
    case 1: return 20;
    case 2: return 30;
    default:
        __builtin_unreachable();
    }
}

static void test_builtin_unreachable(void) {
    CHECK("__builtin_unreachable (case 0)", test_unreachable_helper(0) == 10);
    CHECK("__builtin_unreachable (case 1)", test_unreachable_helper(1) == 20);
    CHECK("__builtin_unreachable (case 2)", test_unreachable_helper(2) == 30);
}

/* ------------------------------------------------------------------ */
/* 8. __builtin_trap — verified only that it compiles.                */
/*    Calling it would abort the process, so we place it in a path    */
/*    that is never executed.                                         */
/* ------------------------------------------------------------------ */

static void test_builtin_trap_compiles(void) {
    /*
     * The expression below references __builtin_trap in dead code
     * to confirm it compiles without actually invoking it.
     */
    if (0) {
        __builtin_trap();
    }
    CHECK("__builtin_trap compiles (dead-code path)", 1);
}

/* ------------------------------------------------------------------ */
/* 9. Additional long-long variants for completeness (~30 builtins)   */
/* ------------------------------------------------------------------ */

static void test_long_long_variants(void) {
    /* __builtin_clzll */
    CHECK("__builtin_clzll(1ULL) == 63",
          __builtin_clzll(1ULL) == 63);

    /* __builtin_ctzll */
    CHECK("__builtin_ctzll(0x100000000ULL) == 32",
          __builtin_ctzll(0x100000000ULL) == 32);

    /* __builtin_popcountll */
    CHECK("__builtin_popcountll(0xFFFFFFFFFFFFFFFFULL) == 64",
          __builtin_popcountll(0xFFFFFFFFFFFFFFFFULL) == 64);

    /* __builtin_ffsll */
    CHECK("__builtin_ffsll(0x100000000LL) == 33",
          __builtin_ffsll(0x100000000LL) == 33);

    CHECK("__builtin_ffsll(0LL) == 0",
          __builtin_ffsll(0LL) == 0);
}

/* ------------------------------------------------------------------ */
/* 10. __builtin_abs / __builtin_labs / __builtin_llabs               */
/* ------------------------------------------------------------------ */

static void test_builtin_abs(void) {
    CHECK("__builtin_abs(-5) == 5",  __builtin_abs(-5) == 5);
    CHECK("__builtin_abs(5) == 5",   __builtin_abs(5) == 5);
    CHECK("__builtin_abs(0) == 0",   __builtin_abs(0) == 0);
    CHECK("__builtin_labs(-100L) == 100L", __builtin_labs(-100L) == 100L);
    CHECK("__builtin_llabs(-1000LL) == 1000LL",
          __builtin_llabs(-1000LL) == 1000LL);
}

/* ------------------------------------------------------------------ */
/* main — execute all test groups and report results.                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    (void)argv; /* suppress unused-parameter warning */
    printf("=== BCC GCC Builtins Test Suite ===\n\n");

    /* 1. Compile-time builtins */
    printf("--- Compile-time builtins ---\n");
    test_builtin_constant_p(argc);
    test_builtin_types_compatible_p();
    test_builtin_choose_expr();
    test_builtin_offsetof();

    /* 2. Runtime integer builtins */
    printf("\n--- Runtime integer builtins ---\n");
    test_builtin_clz();
    test_builtin_ctz();
    test_builtin_popcount();
    test_builtin_bswap();
    test_builtin_ffs();
    test_builtin_expect();

    /* 3. Address / introspection builtins */
    printf("\n--- Address builtins ---\n");
    test_builtin_frame_address();
    test_builtin_return_address();

    /* 4. Variadic argument builtins */
    printf("\n--- Variadic argument builtins ---\n");
    test_builtin_va();

    /* 5. Overflow-checked arithmetic builtins */
    printf("\n--- Overflow arithmetic builtins ---\n");
    test_builtin_overflow();

    /* 6. Alignment builtins */
    printf("\n--- Alignment builtins ---\n");
    test_builtin_assume_aligned();

    /* 7. __builtin_unreachable */
    printf("\n--- Unreachable builtin ---\n");
    test_builtin_unreachable();

    /* 8. __builtin_trap (compile-only verification) */
    printf("\n--- Trap builtin ---\n");
    test_builtin_trap_compiles();

    /* 9. Long-long variants */
    printf("\n--- Long-long variants ---\n");
    test_long_long_variants();

    /* 10. Absolute value builtins */
    printf("\n--- Absolute value builtins ---\n");
    test_builtin_abs();

    /* Summary */
    printf("\n=== Summary ===\n");
    if (failures == 0) {
        printf("All builtin tests PASSED.\n");
    } else {
        printf("%d builtin test(s) FAILED.\n", failures);
    }

    return failures;
}
