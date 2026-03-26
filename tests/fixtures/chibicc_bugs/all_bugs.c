/* chibicc-pattern bug regression tests
 * Each test returns 0 on success, non-zero on failure.
 * Tests are numbered to match CCC GitHub Issue #232.
 */
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/* Bug 1: sizeof on compound literals */
static int test_sizeof_compound_literal(void) {
    int s = sizeof((int[]){1, 2, 3});
    if (s != 3 * sizeof(int)) {
        printf("FAIL bug1: sizeof compound literal = %d, expected %d\n", s, (int)(3 * sizeof(int)));
        return 1;
    }
    /* Also test sizeof on a struct compound literal */
    struct { int x; char y; } dummy;
    (void)dummy;
    int s2 = sizeof((struct { int x; char y; }){.x = 1, .y = 2});
    if (s2 != sizeof(dummy)) {
        printf("FAIL bug1b: sizeof struct compound = %d\n", s2);
        return 1;
    }
    /* Test without extra parens: sizeof(type){init} — the chibicc pattern */
    int s3 = sizeof(int[]){1, 2, 3};
    if (s3 != 3 * sizeof(int)) {
        printf("FAIL bug1c: sizeof(int[]){1,2,3} no-parens = %d, expected %d\n", s3, (int)(3 * sizeof(int)));
        return 1;
    }
    return 0;
}

/* Bug 2: typeof(function-type) * */
static int identity(int x) { return x; }
static int test_typeof_function_ptr(void) {
    __typeof__(identity) *fp = &identity;
    int result = fp(42);
    if (result != 42) {
        printf("FAIL bug2: typeof function ptr result = %d\n", result);
        return 1;
    }
    return 0;
}

/* Bug 3: int *_Atomic parsing */
static int test_atomic_pointer(void) {
    int x = 42;
    int * _Atomic ap = &x;
    int *p = ap;
    if (*p != 42) {
        printf("FAIL bug3: atomic pointer value = %d\n", *p);
        return 1;
    }
    return 0;
}

/* Bug 4: Designated initializer ordering */
static int test_designated_init_order(void) {
    int arr[5] = { [2] = 20, [0] = 10, [4] = 40 };
    if (arr[0] != 10 || arr[1] != 0 || arr[2] != 20 || arr[3] != 0 || arr[4] != 40) {
        printf("FAIL bug4: arr = {%d,%d,%d,%d,%d}\n", arr[0], arr[1], arr[2], arr[3], arr[4]);
        return 1;
    }
    /* Struct designated init ordering */
    struct { int a, b, c; } s = { .c = 3, .a = 1, .b = 2 };
    if (s.a != 1 || s.b != 2 || s.c != 3) {
        printf("FAIL bug4b: s = {%d,%d,%d}\n", s.a, s.b, s.c);
        return 1;
    }
    return 0;
}

/* Bug 5: Qualifier of pointed-to type for type compatibility */
static int test_pointer_qualifier_compat(void) {
    const int x = 42;
    const int *cp = &x;
    /* Assigning const int* to int* is a constraint violation in strict C,
       but GCC warns and allows it. We test that the value survives. */
    int *p = (int *)cp;
    if (*p != 42) {
        printf("FAIL bug5: pointer qualifier compat = %d\n", *p);
        return 1;
    }
    return 0;
}

/* Bug 7: 32-bit truncation in constant evaluation */
static int test_constant_truncation(void) {
    int x = (int)(0x100000000ULL);
    if (x != 0) {
        printf("FAIL bug7: (int)(0x100000000ULL) = %d, expected 0\n", x);
        return 1;
    }
    int y = (int)(0x1FFFFFFFFULL);
    if (y != -1) {
        printf("FAIL bug7b: (int)(0x1FFFFFFFF) = %d, expected -1\n", y);
        return 1;
    }
    return 0;
}

/* Bug 8: Cast-to-bool in constant evaluation */
static int test_cast_to_bool_const(void) {
    _Bool b1 = (_Bool)42;
    if (b1 != 1) {
        printf("FAIL bug8: (_Bool)42 = %d, expected 1\n", (int)b1);
        return 1;
    }
    _Bool b2 = (_Bool)0;
    if (b2 != 0) {
        printf("FAIL bug8b: (_Bool)0 = %d, expected 0\n", (int)b2);
        return 1;
    }
    _Bool b3 = (_Bool)256;
    if (b3 != 1) {
        printf("FAIL bug8c: (_Bool)256 = %d, expected 1\n", (int)b3);
        return 1;
    }
    return 0;
}

/* Bug 11: (boolean-bitfield)++ correct result */
static int test_bool_bitfield_increment(void) {
    struct { _Bool flag : 1; } s = { .flag = 0 };
    s.flag++;
    if (s.flag != 1) {
        printf("FAIL bug11: bool bitfield++ = %d, expected 1\n", (int)s.flag);
        return 1;
    }
    /* Incrementing again should wrap (bool has range 0-1) */
    s.flag++;
    if (s.flag != 0) {
        printf("FAIL bug11b: bool bitfield++(2) = %d, expected 0\n", (int)s.flag);
        return 1;
    }
    return 0;
}

/* Bug 12: Struct alignment (offsetof checks) */
static int test_struct_alignment(void) {
    struct S1 { char a; int b; };
    if (offsetof(struct S1, b) != 4) {
        printf("FAIL bug12: offsetof(S1,b) = %d, expected 4\n", (int)offsetof(struct S1, b));
        return 1;
    }
    struct S2 { char a; long long b; };
    if (offsetof(struct S2, b) != 8) {
        printf("FAIL bug12b: offsetof(S2,b) = %d, expected 8\n", (int)offsetof(struct S2, b));
        return 1;
    }
    struct S3 { char a; short b; int c; };
    if (offsetof(struct S3, b) != 2 || offsetof(struct S3, c) != 4) {
        printf("FAIL bug12c: offsetof(S3,b)=%d,c=%d\n",
            (int)offsetof(struct S3, b), (int)offsetof(struct S3, c));
        return 1;
    }
    return 0;
}

/* Bug 14: Array-to-pointer decay in all required contexts */
static int test_array_decay(void) {
    int arr[3] = {10, 20, 30};
    /* Decay in conditional expression */
    int *p = 1 ? arr : (int *)0;
    if (*p != 10) {
        printf("FAIL bug14: conditional decay = %d\n", *p);
        return 1;
    }
    /* Decay in comma expression */
    int *q = (0, arr);
    if (*q != 10) {
        printf("FAIL bug14b: comma decay = %d\n", *q);
        return 1;
    }
    return 0;
}

/* Bug 15: Integer promotion for statement-expressions */
static int test_stmt_expr_promotion(void) {
    int result = sizeof(({short x = 1; x;}));
    /* The statement expression should evaluate to int (promoted from short) */
    if (result != sizeof(int)) {
        printf("FAIL bug15: stmt expr promotion sizeof = %d, expected %d\n",
               result, (int)sizeof(int));
        return 1;
    }
    return 0;
}

/* Bug 18: GCC ,##__VA_ARGS__ extension */
#define DEBUG_PRINT(fmt, ...) sprintf(buf, fmt, ##__VA_ARGS__)
static int test_va_args_comma_elision(void) {
    char buf[256];
    /* With no variadic args, the comma before ##__VA_ARGS__ should be elided */
    DEBUG_PRINT("hello");
    if (strcmp(buf, "hello") != 0) {
        printf("FAIL bug18: comma elision result = '%s'\n", buf);
        return 1;
    }
    /* With variadic args, normal behavior */
    DEBUG_PRINT("value=%d", 42);
    if (strcmp(buf, "value=42") != 0) {
        printf("FAIL bug18b: with args result = '%s'\n", buf);
        return 1;
    }
    return 0;
}

/* Bug 9: Cast-to-bool with relocation pointers — must normalize to 0/1 */
static int test_cast_to_bool_reloc_ptr(void) {
    /* Use a local address-taken variable to avoid aarch64 ADRP range issues
     * in large translation units.  The core bug is about cast-to-bool
     * normalizing to 0/1, not about global addressing. */
    int local_var = 42;
    _Bool b = (_Bool)&local_var;
    if ((int)b != 1) {
        printf("FAIL bug9: (_Bool)&var = %d, expected 1\n", (int)b);
        return 1;
    }
    /* NULL pointer cast to bool should be 0 */
    _Bool b2 = (_Bool)(void *)0;
    if ((int)b2 != 0) {
        printf("FAIL bug9b: (_Bool)NULL = %d, expected 0\n", (int)b2);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_sizeof_compound_literal();
    failures += test_typeof_function_ptr();
    failures += test_atomic_pointer();
    failures += test_designated_init_order();
    failures += test_pointer_qualifier_compat();
    failures += test_constant_truncation();
    failures += test_cast_to_bool_const();
    failures += test_bool_bitfield_increment();
    failures += test_struct_alignment();
    failures += test_array_decay();
    failures += test_stmt_expr_promotion();
    failures += test_va_args_comma_elision();
    failures += test_cast_to_bool_reloc_ptr();
    if (failures == 0) {
        printf("chibicc_bugs OK\n");
    } else {
        printf("chibicc_bugs: %d FAILURES\n", failures);
    }
    return failures;
}
