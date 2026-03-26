/*
 * tests/fixtures/builtins.c — GCC Builtins Coverage Test (Checkpoint 2)
 *
 * Validates GCC builtin function support across compile-time and runtime
 * builtins. Covers ~30 builtins referenced in AAP §0.6.1 / §4.3:
 *
 *   Compile-time: __builtin_constant_p, __builtin_offsetof,
 *                 __builtin_types_compatible_p, __builtin_choose_expr
 *
 *   Bit manipulation: __builtin_clz, __builtin_ctz, __builtin_popcount,
 *                     __builtin_clzll, __builtin_ctzll, __builtin_popcountll,
 *                     __builtin_ffs, __builtin_ffsll
 *
 *   Byte swap: __builtin_bswap16, __builtin_bswap32, __builtin_bswap64
 *
 *   Branch/control: __builtin_expect, __builtin_unreachable
 *
 *   Variadic: __builtin_va_start, __builtin_va_end, __builtin_va_arg,
 *             __builtin_va_copy
 *
 *   Address introspection: __builtin_frame_address, __builtin_return_address
 *
 *   Alignment: __builtin_assume_aligned
 *
 *   Overflow arithmetic: __builtin_add_overflow, __builtin_sub_overflow,
 *                        __builtin_mul_overflow
 *
 *   Trap: __builtin_trap (compiled but not executed)
 *
 * Expected output: "builtins OK\n" with exit code 0.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdarg.h>
#include <limits.h>

/* ---------- struct for offsetof testing ---------- */
struct TestStruct {
    char a;
    int b;
    short c;
};

/* ---------- nested struct for deep offsetof testing ---------- */
struct Outer {
    int x;
    struct TestStruct inner;
};

/* ---------- variadic helpers for __builtin_va_* testing ---------- */
static int sum_ints(int count, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += __builtin_va_arg(ap, int);
    }

    __builtin_va_end(ap);
    return total;
}

static int sum_ints_copy(int count, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, count);

    /* Test __builtin_va_copy */
    __builtin_va_list ap_copy;
    __builtin_va_copy(ap_copy, ap);

    /* Consume the original — just to advance it */
    int total_orig = 0;
    for (int i = 0; i < count; i++) {
        total_orig += __builtin_va_arg(ap, int);
    }
    __builtin_va_end(ap);

    /* Now consume the copy — should yield identical results */
    int total_copy = 0;
    for (int i = 0; i < count; i++) {
        total_copy += __builtin_va_arg(ap_copy, int);
    }
    __builtin_va_end(ap_copy);

    if (total_orig != total_copy) {
        return -1; /* mismatch indicates broken va_copy */
    }
    return total_copy;
}

/* ---------- function that uses __builtin_unreachable ---------- */
static int always_returns_42(int n) {
    if (n == 0) {
        return 42;
    }
    if (n != 0) {
        return 42;
    }
    /* Control never reaches here — inform the compiler */
    __builtin_unreachable();
}

/* ---------- function whose address is taken to test __builtin_trap compilation ---------- */
static void trap_function(void) {
    /*
     * __builtin_trap terminates the process immediately.
     * We only need to verify it compiles; we never call this function.
     */
    __builtin_trap();
}

int main(void) {
    /* ===================================================================
     * 1. __builtin_constant_p — compile-time constant detection
     * =================================================================== */
    if (!__builtin_constant_p(42)) {
        printf("FAIL: constant_p(42)\n");
        return 1;
    }

    /* Compile-time string literal is also a constant expression */
    if (!__builtin_constant_p(0 + 1)) {
        printf("FAIL: constant_p(0+1)\n");
        return 1;
    }

    /* Note: __builtin_constant_p on a variable is implementation-defined,
       so we only test the definite-positive case (literals). */

    /* ===================================================================
     * 2. __builtin_offsetof — struct member offset
     * =================================================================== */
    size_t off_b = __builtin_offsetof(struct TestStruct, b);
    /* 'b' is an int after 'char a'; with typical alignment off_b == 4 */
    if (off_b < 1) {
        printf("FAIL: offsetof b=%zu\n", off_b);
        return 1;
    }

    size_t off_c = __builtin_offsetof(struct TestStruct, c);
    /* 'c' must come after 'b' */
    if (off_c <= off_b) {
        printf("FAIL: offsetof c=%zu <= b=%zu\n", off_c, off_b);
        return 1;
    }

    /* Nested struct offsetof */
    size_t off_inner_b = __builtin_offsetof(struct Outer, inner.b);
    if (off_inner_b < sizeof(int)) {
        printf("FAIL: offsetof Outer.inner.b=%zu\n", off_inner_b);
        return 1;
    }

    /* ===================================================================
     * 3. __builtin_types_compatible_p — type compatibility
     * =================================================================== */
    if (!__builtin_types_compatible_p(int, int)) {
        printf("FAIL: types_compatible int,int\n");
        return 1;
    }
    if (!__builtin_types_compatible_p(unsigned int, unsigned int)) {
        printf("FAIL: types_compatible uint,uint\n");
        return 1;
    }
    /* int and float are never compatible */
    if (__builtin_types_compatible_p(int, float)) {
        printf("FAIL: types_compatible int,float\n");
        return 1;
    }
    /* signed char and unsigned char are distinct */
    if (__builtin_types_compatible_p(signed char, unsigned char)) {
        printf("FAIL: types_compatible schar,uchar\n");
        return 1;
    }

    /* ===================================================================
     * 4. __builtin_choose_expr — compile-time selection
     * =================================================================== */
    int chosen = __builtin_choose_expr(1, 42, "not this");
    if (chosen != 42) {
        printf("FAIL: choose_expr got %d\n", chosen);
        return 1;
    }

    /* The false branch is never type-checked, so mismatched type is fine */
    double chosen_d = __builtin_choose_expr(0, "ignored", 3.14);
    if (chosen_d < 3.13 || chosen_d > 3.15) {
        printf("FAIL: choose_expr(0) got %f\n", chosen_d);
        return 1;
    }

    /* ===================================================================
     * 5. __builtin_clz — count leading zeros (32-bit int)
     * =================================================================== */
    int clz_val = __builtin_clz(1); /* 0x00000001 → 31 leading zeros */
    if (clz_val != 31) {
        printf("FAIL: clz(1)=%d\n", clz_val);
        return 1;
    }

    if (__builtin_clz(0x80000000U) != 0) {
        printf("FAIL: clz(0x80000000)\n");
        return 1;
    }

    if (__builtin_clz(16) != 27) { /* 0b10000 → 27 leading zeros */
        printf("FAIL: clz(16)=%d\n", __builtin_clz(16));
        return 1;
    }

    /* ===================================================================
     * 6. __builtin_clzll — count leading zeros (64-bit)
     * =================================================================== */
    if (__builtin_clzll(1ULL) != 63) {
        printf("FAIL: clzll(1)=%d\n", __builtin_clzll(1ULL));
        return 1;
    }

    /* ===================================================================
     * 7. __builtin_ctz — count trailing zeros (32-bit)
     * =================================================================== */
    int ctz_val = __builtin_ctz(8); /* 0b1000 → 3 trailing zeros */
    if (ctz_val != 3) {
        printf("FAIL: ctz(8)=%d\n", ctz_val);
        return 1;
    }

    if (__builtin_ctz(1) != 0) {
        printf("FAIL: ctz(1)=%d\n", __builtin_ctz(1));
        return 1;
    }

    /* ===================================================================
     * 8. __builtin_ctzll — count trailing zeros (64-bit)
     * =================================================================== */
    if (__builtin_ctzll(0x100000000ULL) != 32) {
        printf("FAIL: ctzll(0x100000000)=%d\n", __builtin_ctzll(0x100000000ULL));
        return 1;
    }

    /* ===================================================================
     * 9. __builtin_popcount — count set bits (32-bit)
     * =================================================================== */
    int pop_val = __builtin_popcount(0xFF); /* 8 bits set */
    if (pop_val != 8) {
        printf("FAIL: popcount(0xFF)=%d\n", pop_val);
        return 1;
    }

    if (__builtin_popcount(0) != 0) {
        printf("FAIL: popcount(0)\n");
        return 1;
    }

    if (__builtin_popcount(0xAAAAAAAA) != 16) {
        printf("FAIL: popcount(0xAAAAAAAA)=%d\n", __builtin_popcount(0xAAAAAAAA));
        return 1;
    }

    /* ===================================================================
     * 10. __builtin_popcountll — count set bits (64-bit)
     * =================================================================== */
    if (__builtin_popcountll(0xFFFFFFFFFFFFFFFFULL) != 64) {
        printf("FAIL: popcountll(all 1s)=%d\n",
               __builtin_popcountll(0xFFFFFFFFFFFFFFFFULL));
        return 1;
    }

    /* ===================================================================
     * 11. __builtin_bswap16 — byte swap 16-bit
     * =================================================================== */
    unsigned short bswap16 = __builtin_bswap16(0x1234);
    if (bswap16 != 0x3412) {
        printf("FAIL: bswap16=0x%04x\n", bswap16);
        return 1;
    }

    /* ===================================================================
     * 12. __builtin_bswap32 — byte swap 32-bit
     * =================================================================== */
    unsigned int bswap32 = __builtin_bswap32(0x12345678);
    if (bswap32 != 0x78563412) {
        printf("FAIL: bswap32=0x%08x\n", bswap32);
        return 1;
    }

    /* ===================================================================
     * 13. __builtin_bswap64 — byte swap 64-bit
     * =================================================================== */
    unsigned long long bswap64 = __builtin_bswap64(0x0102030405060708ULL);
    if (bswap64 != 0x0807060504030201ULL) {
        printf("FAIL: bswap64\n");
        return 1;
    }

    /* ===================================================================
     * 14. __builtin_ffs — find first set bit (1-indexed, 0 if none)
     * =================================================================== */
    int ffs_val = __builtin_ffs(0x80); /* bit 7 is lowest set → position 8 */
    if (ffs_val != 8) {
        printf("FAIL: ffs(0x80)=%d\n", ffs_val);
        return 1;
    }

    if (__builtin_ffs(0) != 0) {
        printf("FAIL: ffs(0)=%d\n", __builtin_ffs(0));
        return 1;
    }

    if (__builtin_ffs(1) != 1) {
        printf("FAIL: ffs(1)=%d\n", __builtin_ffs(1));
        return 1;
    }

    /* ===================================================================
     * 15. __builtin_ffsll — find first set bit (64-bit)
     * =================================================================== */
    if (__builtin_ffsll(0x100000000LL) != 33) {
        printf("FAIL: ffsll(0x100000000)=%d\n", __builtin_ffsll(0x100000000LL));
        return 1;
    }

    /* ===================================================================
     * 16. __builtin_expect — branch prediction hint
     * =================================================================== */
    if (__builtin_expect(1, 1)) {
        /* expected path — no observable correctness effect */
    } else {
        printf("FAIL: expect(1,1) took else branch\n");
        return 1;
    }

    if (__builtin_expect(0, 0)) {
        printf("FAIL: expect(0,0) took if branch\n");
        return 1;
    }

    /* Value passthrough: __builtin_expect returns its first argument */
    int expect_val = __builtin_expect(7, 7);
    if (expect_val != 7) {
        printf("FAIL: expect passthrough=%d\n", expect_val);
        return 1;
    }

    /* ===================================================================
     * 17. __builtin_unreachable — control flow hint
     * =================================================================== */
    int r42 = always_returns_42(0);
    if (r42 != 42) {
        printf("FAIL: unreachable helper=%d\n", r42);
        return 1;
    }

    /* ===================================================================
     * 18. __builtin_va_start / __builtin_va_arg / __builtin_va_end
     * =================================================================== */
    int va_sum = sum_ints(4, 10, 20, 30, 40);
    if (va_sum != 100) {
        printf("FAIL: va sum=%d\n", va_sum);
        return 1;
    }

    /* ===================================================================
     * 19. __builtin_va_copy
     * =================================================================== */
    int va_copy_sum = sum_ints_copy(3, 5, 10, 15);
    if (va_copy_sum != 30) {
        printf("FAIL: va_copy sum=%d\n", va_copy_sum);
        return 1;
    }

    /* ===================================================================
     * 20. __builtin_frame_address — returns frame pointer (level 0)
     * =================================================================== */
    void *frame_addr = __builtin_frame_address(0);
    if (frame_addr == (void *)0) {
        printf("FAIL: frame_address(0) is NULL\n");
        return 1;
    }

    /* ===================================================================
     * 21. __builtin_return_address — returns return address (level 0)
     * =================================================================== */
    void *ret_addr = __builtin_return_address(0);
    if (ret_addr == (void *)0) {
        printf("FAIL: return_address(0) is NULL\n");
        return 1;
    }

    /* ===================================================================
     * 22. __builtin_assume_aligned — alignment hint, returns pointer
     * =================================================================== */
    int aligned_buf[4] __attribute__((aligned(16)));
    aligned_buf[0] = 99;
    int *aptr = (int *)__builtin_assume_aligned(aligned_buf, 16);
    if (*aptr != 99) {
        printf("FAIL: assume_aligned value=%d\n", *aptr);
        return 1;
    }

    /* ===================================================================
     * 23. __builtin_add_overflow — checked addition
     * =================================================================== */
    {
        int result;
        int overflow = __builtin_add_overflow(INT_MAX, 1, &result);
        if (!overflow) {
            printf("FAIL: add_overflow(INT_MAX,1) no overflow\n");
            return 1;
        }

        overflow = __builtin_add_overflow(10, 20, &result);
        if (overflow || result != 30) {
            printf("FAIL: add_overflow(10,20)=%d ovf=%d\n", result, overflow);
            return 1;
        }
    }

    /* ===================================================================
     * 24. __builtin_sub_overflow — checked subtraction
     * =================================================================== */
    {
        int result;
        int overflow = __builtin_sub_overflow(INT_MIN, 1, &result);
        if (!overflow) {
            printf("FAIL: sub_overflow(INT_MIN,1) no overflow\n");
            return 1;
        }

        overflow = __builtin_sub_overflow(50, 30, &result);
        if (overflow || result != 20) {
            printf("FAIL: sub_overflow(50,30)=%d ovf=%d\n", result, overflow);
            return 1;
        }
    }

    /* ===================================================================
     * 25. __builtin_mul_overflow — checked multiplication
     * =================================================================== */
    {
        int result;
        int overflow = __builtin_mul_overflow(INT_MAX, 2, &result);
        if (!overflow) {
            printf("FAIL: mul_overflow(INT_MAX,2) no overflow\n");
            return 1;
        }

        overflow = __builtin_mul_overflow(6, 7, &result);
        if (overflow || result != 42) {
            printf("FAIL: mul_overflow(6,7)=%d ovf=%d\n", result, overflow);
            return 1;
        }
    }

    /* ===================================================================
     * 26. __builtin_trap — verify compilation only (never called)
     * =================================================================== */
    {
        /*
         * Take the address of trap_function so the compiler keeps it,
         * proving __builtin_trap() compiles. We never invoke it.
         */
        void (*volatile tp)(void) = trap_function;
        (void)tp;
    }

    /* ===================================================================
     * All tests passed
     * =================================================================== */
    printf("builtins OK\n");
    return 0;
}
