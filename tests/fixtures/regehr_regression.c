// Regression tests for Regehr fuzzing bug classes found in CCC
// Each test_N_M function tests bug class N, sub-case M
// Return 0 = pass, nonzero = fail
#include <stdio.h>
#include <stdlib.h>

// === Bug 2.1: IR narrowing for 64-bit bitwise ops ===
// Sign-extend vs zero-extend mismatch after widening back

__attribute__((noinline))
unsigned long long get_mask64(void) { return 0xFFFFFFFF00000000ULL; }

__attribute__((noinline))
unsigned long long get_val64(void) { return 0x00000000FFFFFFFFULL; }

int test_2_1_or(void) {
    unsigned long long a = get_val64();
    unsigned long long b = get_mask64();
    unsigned long long r = a | b;
    if (r != 0xFFFFFFFFFFFFFFFFULL) { printf("FAIL 2.1a: or = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_1_xor(void) {
    unsigned long long a = 0x1234567800000000ULL;
    unsigned long long b = 0x1234567800000000ULL;
    unsigned long long r = a ^ b;
    if (r != 0) { printf("FAIL 2.1b: xor = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_1_and_high(void) {
    unsigned long long v = 0xABCD1234DEADBEEFULL;
    unsigned long long m = 0xFFFF0000FFFF0000ULL;
    unsigned long long r = v & m;
    if (r != 0xABCD0000DEAD0000ULL) { printf("FAIL 2.1c: and = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_1_not(void) {
    unsigned long long v = get_mask64();
    unsigned long long r = ~v;
    if (r != 0x00000000FFFFFFFFULL) { printf("FAIL 2.1d: not = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_1_narrow_trap(void) {
    unsigned long long a = 0x0000000100000000ULL;
    unsigned long long b = 0x0000000200000000ULL;
    unsigned long long r = a | b;
    if (r != 0x0000000300000000ULL) { printf("FAIL 2.1e: narrow = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

__attribute__((noinline))
unsigned long long rot64(unsigned long long x, int n) {
    return (x << n) | (x >> (64 - n));
}

int test_2_1_rotate(void) {
    unsigned long long r = rot64(0x0123456789ABCDEFULL, 8);
    if (r != 0x23456789ABCDEF01ULL) { printf("FAIL 2.1f: rot = 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

// === Bug 2.2: Unsigned negation in constant folding ===

__attribute__((noinline)) unsigned int get_8u(void) { return 8u; }

int test_2_2_const(void) {
    unsigned int a = -8u;
    if (a != 0xFFFFFFF8u) { printf("FAIL 2.2a: -8u = 0x%x\n", a); return 1; }
    return 0;
}

int test_2_2_runtime(void) {
    unsigned int b = -get_8u();
    if (b != 0xFFFFFFF8u) { printf("FAIL 2.2b: -8u rt = 0x%x\n", b); return 1; }
    return 0;
}

int test_2_2_zero(void) {
    unsigned int c = -0u;
    if (c != 0) { printf("FAIL 2.2c: -0u = 0x%x\n", c); return 1; }
    return 0;
}

int test_2_2_max(void) {
    unsigned int d = -(0xFFFFFFFFu);
    if (d != 1) { printf("FAIL 2.2d: -UINT_MAX = 0x%x\n", d); return 1; }
    return 0;
}

int test_2_2_64(void) {
    unsigned long long e = -1ULL;
    if (e != 0xFFFFFFFFFFFFFFFFULL) { printf("FAIL 2.2e: -1ULL = 0x%llx\n", (unsigned long long)e); return 1; }
    return 0;
}

int test_2_2_compare(void) {
    unsigned int f = -1u;
    if (f < 100) { printf("FAIL 2.2f: -1u < 100\n"); return 1; }
    return 0;
}

__attribute__((noinline)) unsigned int neg_mod_fn(unsigned int a, unsigned int b) { return (-a) % b; }

int test_2_2_neg_mod(void) {
    unsigned int r = neg_mod_fn(8, 5);
    if (r != 3) { printf("FAIL 2.2g: neg_mod = %u\n", r); return 1; }
    return 0;
}

// === Bug 2.3: Peephole cmp+branch fusion ===

volatile int g_side_effect = 0;
__attribute__((noinline)) int compute_fn(int a, int b) { g_side_effect = 1; return a - b; }

__attribute__((noinline))
int reload_regs(int x) {
    volatile int v1=x, v2=x+1, v3=x+2, v4=x+3, v5=x+4, v6=x+5, v7=x+6, v8=x+7;
    return v1+v2+v3+v4+v5+v6+v7+v8;
}

int test_2_3_across_reload(void) {
    int a = 5, b = 3;
    int cmp = (a > b);
    int dummy = reload_regs(10);
    if (cmp) return (dummy > 0) ? 0 : 1;
    return 1;
}

int test_2_3_across_call(void) {
    volatile int x = 10, y = 20;
    int result = (x < y);
    compute_fn(100, 200);
    if (result) return 0;
    return 1;
}

int test_2_3_interleaved(void) {
    volatile int a=1, b=2, c=3, d=4;
    int r1 = (a < b);
    int r2 = (c < d);
    if (r1 && r2) return 0;
    return 1;
}

// === Bug 2.4: narrow_cmps cast stripping ===

__attribute__((noinline)) signed char get_sc(signed char x) { return x; }
__attribute__((noinline)) unsigned char get_uc(unsigned char x) { return x; }

int test_2_4_schar_cmp(void) {
    signed char a = get_sc(-1);
    signed char b = get_sc(1);
    if (a < b) return 0;
    return 1;
}

int test_2_4_uchar_vs_schar(void) {
    unsigned char a = get_uc(255);
    signed char b = get_sc(-1);
    if (a > b) return 0;
    return 1;
}

int test_2_4_sign_change(void) {
    unsigned char x = get_uc(200);
    signed char y = (signed char)x;
    if (y < 0) return 0;
    return 1;
}

int test_2_4_sub_int_narrow(void) {
    int a = 300, b = 200;
    if (a > b) return 0;
    return 1;
}

// === Bug 2.5: Shift narrowing ===

__attribute__((noinline)) unsigned long long get_ull(unsigned long long x) { return x; }

int test_2_5_left32(void) {
    unsigned long long x = get_ull(1ULL);
    unsigned long long r = x << 32;
    if (r != 0x100000000ULL) { printf("FAIL 2.5a\n"); return 1; }
    return 0;
}

int test_2_5_left48(void) {
    unsigned long long x = get_ull(1ULL);
    unsigned long long r = x << 48;
    if (r != 0x1000000000000ULL) { printf("FAIL 2.5b\n"); return 1; }
    return 0;
}

int test_2_5_right32(void) {
    unsigned long long x = get_ull(0xFFFFFFFF00000000ULL);
    unsigned long long r = x >> 32;
    if (r != 0xFFFFFFFFULL) { printf("FAIL 2.5c\n"); return 1; }
    return 0;
}

int test_2_5_signed_right(void) {
    long long x = -0x100000000LL;
    long long r = x >> 16;
    if (r != -0x10000LL) { printf("FAIL 2.5d: %lld\n", r); return 1; }
    return 0;
}

__attribute__((noinline)) unsigned long long shift_chain_fn(unsigned long long x) {
    unsigned long long a = x << 16;
    unsigned long long b = a << 16;
    unsigned long long c = b >> 16;
    return c;
}

int test_2_5_chain(void) {
    unsigned long long r = shift_chain_fn(0x12345678ULL);
    if (r != 0x123456780000ULL) { printf("FAIL 2.5e: 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

// === Bug 2.6: Usual arithmetic conversions ===

__attribute__((noinline)) unsigned int get_u32(unsigned int x) { return x; }
__attribute__((noinline)) int get_i32(int x) { return x; }

int test_2_6_mixed_add(void) {
    unsigned int u = get_u32(1);
    int s = get_i32(-2);
    unsigned int r = u + s;
    if (r != 0xFFFFFFFF) { printf("FAIL 2.6a: 0x%x\n", r); return 1; }
    return 0;
}

int test_2_6_mixed_cmp(void) {
    unsigned int u = get_u32(1);
    int s = get_i32(-1);
    if (u < s) return 0;
    printf("FAIL 2.6b\n"); return 1;
}

int test_2_6_sub(void) {
    unsigned int a = get_u32(5);
    int b = get_i32(10);
    unsigned int r = a - b;
    if (r != 0xFFFFFFFB) { printf("FAIL 2.6c: 0x%x\n", r); return 1; }
    return 0;
}

int test_2_6_char_mul(void) {
    signed char a = 100, b = 100;
    int r = a * b;
    if (r != 10000) { printf("FAIL 2.6d: %d\n", r); return 1; }
    return 0;
}

__attribute__((noinline)) int mixed_cmp_fn(unsigned int u, int s) { return (u < s) ? 1 : 0; }

int test_2_6_mixed_fn(void) {
    if (mixed_cmp_fn(5, -1) != 1) { printf("FAIL 2.6e\n"); return 1; }
    if (mixed_cmp_fn(5, 10) != 1) { printf("FAIL 2.6f\n"); return 1; }
    if (mixed_cmp_fn(10, 5) != 0) { printf("FAIL 2.6g\n"); return 1; }
    return 0;
}

// === Bug 2.7: Explicit cast sign-extension ===

__attribute__((noinline)) int get_neg(void) { return -1; }
__attribute__((noinline)) int get_intmin(void) { return (int)0x80000000; }

int test_2_7_basic(void) {
    int x = get_neg();
    long y = (long)x;
    if (y != -1L) { printf("FAIL 2.7a: %ld\n", y); return 1; }
    return 0;
}

int test_2_7_intmin(void) {
    int x = get_intmin();
    long long y = (long long)x;
    if (y != -2147483648LL) { printf("FAIL 2.7b: %lld\n", y); return 1; }
    return 0;
}

int test_2_7_uint_to_ll(void) {
    unsigned int x = 0x80000000u;
    long long y = (long long)x;
    if (y != 2147483648LL) { printf("FAIL 2.7c: %lld\n", y); return 1; }
    return 0;
}

int test_2_7_chain(void) {
    short s = -100;
    int i = (int)s;
    long long ll = (long long)i;
    if (ll != -100LL) { printf("FAIL 2.7d: %lld\n", ll); return 1; }
    return 0;
}

__attribute__((noinline)) long long sign_ext_chain_fn(int x) {
    long long a = (long long)x;
    long long b = a * 2;
    long long c = b / 2;
    return c;
}

int test_2_7_chain_fn(void) {
    long long r = sign_ext_chain_fn(-1);
    if (r != -1LL) { printf("FAIL 2.7e: %lld\n", r); return 1; }
    return 0;
}

// === Bug 2.8: Narrowing optimization for And/Shl ===

__attribute__((noinline)) long long get_ll(long long x) { return x; }

int test_2_8_and_high(void) {
    long long x = get_ll(0x100000000LL);
    long long m = get_ll(0x1FFFFFFFFLL);
    long long r = x & m;
    if (r != 0x100000000LL) { printf("FAIL 2.8a: 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_8_shl(void) {
    long long x = get_ll(0x1LL);
    long long r = x << 33;
    if (r != 0x200000000LL) { printf("FAIL 2.8b: 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_8_sign_cast(void) {
    unsigned int u = 0x80000000u;
    long long s = (long long)(int)u;
    long long m = 0xFFFFFFFFLL;
    long long r = s & m;
    if (r != 0x80000000LL) { printf("FAIL 2.8c: 0x%llx\n", (unsigned long long)r); return 1; }
    return 0;
}

int test_2_8_shl_cast(void) {
    int x = -1;
    long long r = ((long long)x) << 1;
    if (r != -2LL) { printf("FAIL 2.8d: %lld\n", r); return 1; }
    return 0;
}

// === Bug 2.9: U32→I32 same-width cast ===

int test_2_9_high_bit(void) {
    unsigned int u = get_u32(0x80000000u);
    int s = (int)u;
    if (s >= 0) { printf("FAIL 2.9a: %d\n", s); return 1; }
    if (s != -2147483647 - 1) { printf("FAIL 2.9b: %d\n", s); return 1; }
    return 0;
}

int test_2_9_max(void) {
    unsigned int u = get_u32(0xFFFFFFFFu);
    int s = (int)u;
    if (s != -1) { printf("FAIL 2.9c: %d\n", s); return 1; }
    return 0;
}

int test_2_9_ext(void) {
    unsigned int u = get_u32(0xFFFF0000u);
    int s = (int)u;
    long long ll = (long long)s;
    if (ll != -65536LL) { printf("FAIL 2.9d: %lld\n", ll); return 1; }
    return 0;
}

__attribute__((noinline)) int u32_cmp_zero_fn(unsigned int x) { int s = (int)x; return s < 0; }

int test_2_9_cmp(void) {
    if (u32_cmp_zero_fn(0x7FFFFFFF) != 0) { printf("FAIL 2.9e\n"); return 1; }
    if (u32_cmp_zero_fn(0x80000000) != 1) { printf("FAIL 2.9f\n"); return 1; }
    if (u32_cmp_zero_fn(0xFFFFFFFF) != 1) { printf("FAIL 2.9g\n"); return 1; }
    return 0;
}

// === Bug 2.10: div_by_const range analysis ===

__attribute__((noinline)) unsigned int udiv_fn(unsigned int x, unsigned int d) { return x / d; }

int test_2_10_large(void) {
    unsigned int r = udiv_fn(0xFFFFFFFF, 3);
    if (r != 1431655765u) { printf("FAIL 2.10a: %u\n", r); return 1; }
    return 0;
}

int test_2_10_above_intmax(void) {
    unsigned int r = udiv_fn(0x80000000, 7);
    if (r != 306783378u) { printf("FAIL 2.10b: %u\n", r); return 1; }
    return 0;
}

int test_2_10_mod(void) {
    unsigned int x = get_u32(0xFFFFFFFF);
    unsigned int r = x % 5;
    if (r != 0) { printf("FAIL 2.10c: %u\n", r); return 1; }
    return 0;
}

int test_2_10_signed(void) {
    int x = -100;
    int r = x / 3;
    if (r != -33) { printf("FAIL 2.10d: %d\n", r); return 1; }
    return 0;
}

int test_2_10_boundary(void) {
    if (udiv_fn(0x7FFFFFFF, 1) != 0x7FFFFFFF) { printf("FAIL 2.10e\n"); return 1; }
    if (udiv_fn(0x80000000, 1) != 0x80000000) { printf("FAIL 2.10f\n"); return 1; }
    if (udiv_fn(0x80000000, 2) != 0x40000000) { printf("FAIL 2.10g\n"); return 1; }
    if (udiv_fn(0xFFFFFFFF, 0xFFFFFFFF) != 1) { printf("FAIL 2.10h\n"); return 1; }
    return 0;
}

// === Bug 2.11: cfg_simplify constant propagation through Cast ===

int test_2_11_uchar_schar(void) {
    unsigned char x = get_uc(200);
    signed char y = (signed char)x;
    if (y != -56) { printf("FAIL 2.11a: %d\n", (int)y); return 1; }
    return 0;
}

int test_2_11_int_ushort(void) {
    int x = get_i32(0x12345678);
    unsigned short y = (unsigned short)x;
    if (y != 0x5678) { printf("FAIL 2.11b: 0x%x\n", (unsigned)y); return 1; }
    return 0;
}

int test_2_11_narrow_widen(void) {
    int x = get_i32(128);
    signed char n = (signed char)x;
    int w = (int)n;
    if (w != -128) { printf("FAIL 2.11c: %d\n", w); return 1; }
    return 0;
}

int test_2_11_const_through(void) {
    int x = get_i32(300);
    unsigned char y = (unsigned char)x;
    int z = y + 1;
    if (z != 45) { printf("FAIL 2.11d: %d\n", z); return 1; }
    return 0;
}

int test_2_11_conditional(void) {
    unsigned int x = get_i32(0xFF);
    signed char c = (signed char)x;
    if (c < 0) return 0;
    printf("FAIL 2.11e\n"); return 1;
}

__attribute__((noinline)) int cast_chain_fn(int x) {
    unsigned char a = (unsigned char)x;
    signed char b = (signed char)a;
    unsigned short c = (unsigned short)b;
    int d = (int)c;
    return d;
}

int test_2_11_chain(void) {
    if (cast_chain_fn(255) != 65535) { printf("FAIL 2.11f: %d\n", cast_chain_fn(255)); return 1; }
    if (cast_chain_fn(128) != 65408) { printf("FAIL 2.11g: %d\n", cast_chain_fn(128)); return 1; }
    if (cast_chain_fn(127) != 127)   { printf("FAIL 2.11h: %d\n", cast_chain_fn(127)); return 1; }
    return 0;
}

// === Main ===

int main(void) {
    int fail = 0;
    
    // 2.1: 64-bit bitwise
    fail += test_2_1_or();
    fail += test_2_1_xor();
    fail += test_2_1_and_high();
    fail += test_2_1_not();
    fail += test_2_1_narrow_trap();
    fail += test_2_1_rotate();
    
    // 2.2: unsigned negation
    fail += test_2_2_const();
    fail += test_2_2_runtime();
    fail += test_2_2_zero();
    fail += test_2_2_max();
    fail += test_2_2_64();
    fail += test_2_2_compare();
    fail += test_2_2_neg_mod();
    
    // 2.3: cmp+branch fusion
    fail += test_2_3_across_reload();
    fail += test_2_3_across_call();
    fail += test_2_3_interleaved();
    
    // 2.4: narrow_cmps
    fail += test_2_4_schar_cmp();
    fail += test_2_4_uchar_vs_schar();
    fail += test_2_4_sign_change();
    fail += test_2_4_sub_int_narrow();
    
    // 2.5: shift narrowing
    fail += test_2_5_left32();
    fail += test_2_5_left48();
    fail += test_2_5_right32();
    fail += test_2_5_signed_right();
    fail += test_2_5_chain();
    
    // 2.6: usual arithmetic conversions
    fail += test_2_6_mixed_add();
    fail += test_2_6_mixed_cmp();
    fail += test_2_6_sub();
    fail += test_2_6_char_mul();
    fail += test_2_6_mixed_fn();
    
    // 2.7: cast sign-extension
    fail += test_2_7_basic();
    fail += test_2_7_intmin();
    fail += test_2_7_uint_to_ll();
    fail += test_2_7_chain();
    fail += test_2_7_chain_fn();
    
    // 2.8: And/Shl narrowing
    fail += test_2_8_and_high();
    fail += test_2_8_shl();
    fail += test_2_8_sign_cast();
    fail += test_2_8_shl_cast();
    
    // 2.9: U32->I32 cast
    fail += test_2_9_high_bit();
    fail += test_2_9_max();
    fail += test_2_9_ext();
    fail += test_2_9_cmp();
    
    // 2.10: div_by_const
    fail += test_2_10_large();
    fail += test_2_10_above_intmax();
    fail += test_2_10_mod();
    fail += test_2_10_signed();
    fail += test_2_10_boundary();
    
    // 2.11: cfg_simplify cast propagation
    fail += test_2_11_uchar_schar();
    fail += test_2_11_int_ushort();
    fail += test_2_11_narrow_widen();
    fail += test_2_11_const_through();
    fail += test_2_11_conditional();
    fail += test_2_11_chain();

    if (fail == 0) printf("ALL 58 Regehr regression tests PASSED\n");
    else printf("%d Regehr test(s) FAILED\n", fail);
    return fail;
}
