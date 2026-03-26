/* Regehr fuzzing bug regression tests.
 * Each test targets a specific class of miscompilation bugs found by
 * Prof. Regehr via Csmith/YARPGen fuzzing of CCC.
 * All tests return 0 on success.
 */
#include <stdio.h>
#include <limits.h>
#include <string.h>

/* Bug 1: IR narrowing for 64-bit bitwise ops */
static int test_64bit_bitwise_narrowing(void) {
    unsigned long long a = 0xFFFFFFFF00000000ULL;
    unsigned long long b = 0x00000000FFFFFFFFULL;
    unsigned long long c = a | b;
    if (c != 0xFFFFFFFFFFFFFFFFULL) {
        printf("FAIL regehr1: 64-bit OR = 0x%llx\n", c);
        return 1;
    }
    unsigned long long d = a & 0xFFFF0000FFFF0000ULL;
    if (d != 0xFFFF000000000000ULL) {
        printf("FAIL regehr1b: 64-bit AND = 0x%llx\n", d);
        return 1;
    }
    return 0;
}

/* Bug 2: Unsigned negation in constant folding */
static int test_unsigned_negation(void) {
    unsigned int x = -8u;
    if (x != 0xFFFFFFF8u) {
        printf("FAIL regehr2: -8u = 0x%x, expected 0xFFFFFFF8\n", x);
        return 1;
    }
    unsigned int y = -(unsigned int)1;
    if (y != 0xFFFFFFFFu) {
        printf("FAIL regehr2b: -(unsigned)1 = 0x%x\n", y);
        return 1;
    }
    return 0;
}

/* Bug 3: Peephole cmp+branch fusion */
static int test_cmp_branch_fusion(void) {
    volatile int a = 5, b = 10;
    int result;
    if (a < b) {
        result = 1;
    } else {
        result = 0;
    }
    if (result != 1) {
        printf("FAIL regehr3: cmp+branch = %d\n", result);
        return 1;
    }
    return 0;
}

/* Bug 4: narrow_cmps cast stripping */
static int test_narrow_cmps(void) {
    signed char a = -1;
    signed char b = 1;
    int result = (a < b);
    if (result != 1) {
        printf("FAIL regehr4: narrow cmp = %d\n", result);
        return 1;
    }
    unsigned char ua = 255;
    unsigned char ub = 1;
    int result2 = (ua > ub);
    if (result2 != 1) {
        printf("FAIL regehr4b: unsigned narrow cmp = %d\n", result2);
        return 1;
    }
    return 0;
}

/* Bug 5: Shift narrowing */
static int test_shift_narrowing(void) {
    unsigned long long val = 1ULL << 32;
    if (val != 0x100000000ULL) {
        printf("FAIL regehr5: 1ULL<<32 = 0x%llx\n", val);
        return 1;
    }
    unsigned long long val2 = 0x8000000000000000ULL >> 63;
    if (val2 != 1) {
        printf("FAIL regehr5b: >>63 = 0x%llx\n", val2);
        return 1;
    }
    return 0;
}

/* Bug 6: Usual arithmetic conversions */
static int test_usual_arith_conv(void) {
    unsigned int a = 0xFFFFFFFFu;
    int b = 1;
    unsigned int c = a + b;
    if (c != 0) {
        printf("FAIL regehr6: arith conv = %u\n", c);
        return 1;
    }
    return 0;
}

/* Bug 7: Explicit cast sign-extension */
static int test_explicit_cast_sext(void) {
    int x = -1;
    long y = (long)x;
    if (y != -1L) {
        printf("FAIL regehr7: (long)(int)-1 = %ld\n", y);
        return 1;
    }
    unsigned int ux = 0xFFFFFFFF;
    long sy = (long)(int)ux;
    if (sy != -1L) {
        printf("FAIL regehr7b: (long)(int)0xFFFFFFFF = %ld\n", sy);
        return 1;
    }
    return 0;
}

/* Bug 8: Narrowing optimization for And/Shl */
static int test_and_shl_narrowing(void) {
    long long x = 0x100000000LL;
    long long y = x & 0x1FFFFFFFFLL;
    if (y != 0x100000000LL) {
        printf("FAIL regehr8: and narrowing = 0x%llx\n", y);
        return 1;
    }
    unsigned long long z = 1ULL;
    unsigned long long shifted = z << 33;
    unsigned long long anded = shifted & 0x3FFFFFFFFULL;
    if (anded != 0x200000000ULL) {
        printf("FAIL regehr8b: shl+and = 0x%llx\n", anded);
        return 1;
    }
    return 0;
}

/* Bug 9: U32 to I32 same-width cast */
static int test_u32_i32_cast(void) {
    unsigned int u = 0x80000000u;
    int s = (int)u;
    if (s != (int)0x80000000) {
        printf("FAIL regehr9: u32->i32 = %d\n", s);
        return 1;
    }
    long l = (long)(int)(unsigned int)0xFFFFFFFF;
    if (l != -1L) {
        printf("FAIL regehr9b: u32->i32->i64 = %ld\n", l);
        return 1;
    }
    return 0;
}

/* Bug 10: div_by_const range analysis */
static int test_div_const_range(void) {
    unsigned int x = 0xFFFFFFFF;
    unsigned int result = x / 3;
    if (result != 1431655765) {
        printf("FAIL regehr10: 0xFFFFFFFF/3 = %u, expected 1431655765\n", result);
        return 1;
    }
    return 0;
}

/* Bug 11: cfg_simplify constant propagation through Cast */
static int test_cfg_simplify_cast(void) {
    int x = 256;
    signed char c = (signed char)x;
    if (c != 0) {
        printf("FAIL regehr11: (signed char)256 = %d, expected 0\n", (int)c);
        return 1;
    }
    int y = -129;
    signed char d = (signed char)y;
    if (d != 127) {
        printf("FAIL regehr11b: (signed char)-129 = %d, expected 127\n", (int)d);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_64bit_bitwise_narrowing();
    failures += test_unsigned_negation();
    failures += test_cmp_branch_fusion();
    failures += test_narrow_cmps();
    failures += test_shift_narrowing();
    failures += test_usual_arith_conv();
    failures += test_explicit_cast_sext();
    failures += test_and_shl_narrowing();
    failures += test_u32_i32_cast();
    failures += test_div_const_range();
    failures += test_cfg_simplify_cast();
    if (failures == 0) {
        printf("regehr_bugs OK\n");
    } else {
        printf("regehr_bugs: %d FAILURES\n", failures);
    }
    return failures;
}
