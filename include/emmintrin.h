/* BCC bundled emmintrin.h — SSE2 intrinsics (struct-based fallback) */
#ifndef _EMMINTRIN_H_INCLUDED
#define _EMMINTRIN_H_INCLUDED
#include <xmmintrin.h>

/* === Double-precision and integer SIMD types === */
typedef struct __m128d_struct {
    double __d[2];
} __attribute__((aligned(16))) __m128d;

typedef union __m128i_union {
    long long      __i64[2];
    unsigned long long __u64[2];
    int            __i32[4];
    unsigned int   __u32[4];
    short          __i16[8];
    unsigned short __u16[8];
    signed char    __i8[16];
    unsigned char  __u8[16];
} __attribute__((aligned(16))) __m128i;

/* ========================
 * Double-precision (pd/sd)
 * ======================== */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_set_pd(double __e1, double __e0)
{
    __m128d __r; __r.__d[0] = __e0; __r.__d[1] = __e1; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_set1_pd(double __w)
{
    __m128d __r; __r.__d[0] = __w; __r.__d[1] = __w; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_set_pd1(double __w) { return _mm_set1_pd(__w); }

static __inline__ __m128d __attribute__((__always_inline__))
_mm_setr_pd(double __e0, double __e1)
{
    __m128d __r; __r.__d[0] = __e0; __r.__d[1] = __e1; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_setzero_pd(void)
{
    __m128d __r; __r.__d[0] = 0; __r.__d[1] = 0; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_set_sd(double __w)
{
    __m128d __r; __r.__d[0] = __w; __r.__d[1] = 0; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_load_pd(const double *__p)
{
    __m128d __r; __r.__d[0] = __p[0]; __r.__d[1] = __p[1]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_loadu_pd(const double *__p) { return _mm_load_pd(__p); }

static __inline__ __m128d __attribute__((__always_inline__))
_mm_load_sd(const double *__p) { return _mm_set_sd(*__p); }

static __inline__ __m128d __attribute__((__always_inline__))
_mm_load1_pd(const double *__p) { return _mm_set1_pd(*__p); }

static __inline__ __m128d __attribute__((__always_inline__))
_mm_loadr_pd(const double *__p) { return _mm_set_pd(__p[0], __p[1]); }

static __inline__ void __attribute__((__always_inline__))
_mm_store_pd(double *__p, __m128d __a) { __p[0] = __a.__d[0]; __p[1] = __a.__d[1]; }

static __inline__ void __attribute__((__always_inline__))
_mm_storeu_pd(double *__p, __m128d __a) { _mm_store_pd(__p, __a); }

static __inline__ void __attribute__((__always_inline__))
_mm_store_sd(double *__p, __m128d __a) { *__p = __a.__d[0]; }

static __inline__ void __attribute__((__always_inline__))
_mm_store1_pd(double *__p, __m128d __a) { __p[0] = __a.__d[0]; __p[1] = __a.__d[0]; }

static __inline__ void __attribute__((__always_inline__))
_mm_storer_pd(double *__p, __m128d __a) { __p[0] = __a.__d[1]; __p[1] = __a.__d[0]; }

static __inline__ void __attribute__((__always_inline__))
_mm_stream_pd(double *__p, __m128d __a) { _mm_store_pd(__p, __a); }

/* Double arithmetic */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_add_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[0]+__b.__d[0]; __r.__d[1] = __a.__d[1]+__b.__d[1]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_sub_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[0]-__b.__d[0]; __r.__d[1] = __a.__d[1]-__b.__d[1]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_mul_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[0]*__b.__d[0]; __r.__d[1] = __a.__d[1]*__b.__d[1]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_div_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[0]/__b.__d[0]; __r.__d[1] = __a.__d[1]/__b.__d[1]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_add_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a; __r.__d[0] = __a.__d[0]+__b.__d[0]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_sub_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a; __r.__d[0] = __a.__d[0]-__b.__d[0]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_mul_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a; __r.__d[0] = __a.__d[0]*__b.__d[0]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_div_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a; __r.__d[0] = __a.__d[0]/__b.__d[0]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_min_pd(__m128d __a, __m128d __b)
{
    __m128d __r;
    __r.__d[0] = __a.__d[0] < __b.__d[0] ? __a.__d[0] : __b.__d[0];
    __r.__d[1] = __a.__d[1] < __b.__d[1] ? __a.__d[1] : __b.__d[1];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_max_pd(__m128d __a, __m128d __b)
{
    __m128d __r;
    __r.__d[0] = __a.__d[0] > __b.__d[0] ? __a.__d[0] : __b.__d[0];
    __r.__d[1] = __a.__d[1] > __b.__d[1] ? __a.__d[1] : __b.__d[1];
    return __r;
}

/* Double comparison */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_cmpeq_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __t;
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __t.u = __a.__d[__i] == __b.__d[__i] ? 0xFFFFFFFFFFFFFFFFULL : 0;
        __r.__d[__i] = __t.d;
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_cmplt_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __t;
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __t.u = __a.__d[__i] < __b.__d[__i] ? 0xFFFFFFFFFFFFFFFFULL : 0;
        __r.__d[__i] = __t.d;
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_cmple_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __t;
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __t.u = __a.__d[__i] <= __b.__d[__i] ? 0xFFFFFFFFFFFFFFFFULL : 0;
        __r.__d[__i] = __t.d;
    }
    return __r;
}

static __inline__ int __attribute__((__always_inline__))
_mm_comieq_sd(__m128d __a, __m128d __b) { return __a.__d[0] == __b.__d[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comilt_sd(__m128d __a, __m128d __b) { return __a.__d[0] < __b.__d[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comile_sd(__m128d __a, __m128d __b) { return __a.__d[0] <= __b.__d[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comigt_sd(__m128d __a, __m128d __b) { return __a.__d[0] > __b.__d[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comige_sd(__m128d __a, __m128d __b) { return __a.__d[0] >= __b.__d[0]; }

/* Double logical */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_and_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __ua[2], __ub[2];
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __ua[__i].d = __a.__d[__i]; __ub[__i].d = __b.__d[__i];
        __ua[__i].u &= __ub[__i].u;
        __r.__d[__i] = __ua[__i].d;
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_or_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __ua[2], __ub[2];
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __ua[__i].d = __a.__d[__i]; __ub[__i].d = __b.__d[__i];
        __ua[__i].u |= __ub[__i].u;
        __r.__d[__i] = __ua[__i].d;
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_xor_pd(__m128d __a, __m128d __b)
{
    union { double d; unsigned long long u; } __ua[2], __ub[2];
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        __ua[__i].d = __a.__d[__i]; __ub[__i].d = __b.__d[__i];
        __ua[__i].u ^= __ub[__i].u;
        __r.__d[__i] = __ua[__i].d;
    }
    return __r;
}

/* Double unpack / shuffle */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_unpacklo_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[0]; __r.__d[1] = __b.__d[0]; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_unpackhi_pd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __a.__d[1]; __r.__d[1] = __b.__d[1]; return __r;
}

#define _mm_shuffle_pd(__a, __b, __imm8) __extension__({ \
    __m128d __r; \
    __r.__d[0] = (__a).__d[(__imm8) & 1]; \
    __r.__d[1] = (__b).__d[((__imm8)>>1) & 1]; \
    __r; \
})

static __inline__ __m128d __attribute__((__always_inline__))
_mm_move_sd(__m128d __a, __m128d __b)
{
    __m128d __r; __r.__d[0] = __b.__d[0]; __r.__d[1] = __a.__d[1]; return __r;
}

/* Double movemask */
static __inline__ int __attribute__((__always_inline__))
_mm_movemask_pd(__m128d __a)
{
    union { double d; unsigned long long u; } __t;
    int __r = 0;
    for (int __i = 0; __i < 2; __i++) {
        __t.d = __a.__d[__i];
        if (__t.u & 0x8000000000000000ULL) __r |= (1 << __i);
    }
    return __r;
}

/* Double conversion */
static __inline__ int __attribute__((__always_inline__))
_mm_cvtsd_si32(__m128d __a) { return (int)__a.__d[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_cvttsd_si32(__m128d __a) { return (int)__a.__d[0]; }

static __inline__ double __attribute__((__always_inline__))
_mm_cvtsd_f64(__m128d __a) { return __a.__d[0]; }

static __inline__ __m128d __attribute__((__always_inline__))
_mm_cvtsi32_sd(__m128d __a, int __b)
{
    __m128d __r = __a; __r.__d[0] = (double)__b; return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_cvtss_sd(__m128d __a, __m128 __b)
{
    __m128d __r = __a; __r.__d[0] = (double)__b.__f[0]; return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cvtsd_ss(__m128 __a, __m128d __b)
{
    __m128 __r = __a; __r.__f[0] = (float)__b.__d[0]; return __r;
}

/* ========================
 * Integer (epi/epu)
 * ======================== */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_setzero_si128(void)
{
    __m128i __r; __r.__i64[0] = 0; __r.__i64[1] = 0; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set_epi32(int __e3, int __e2, int __e1, int __e0)
{
    __m128i __r; __r.__i32[0] = __e0; __r.__i32[1] = __e1; __r.__i32[2] = __e2; __r.__i32[3] = __e3;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi32(int __w)
{
    __m128i __r; __r.__i32[0] = __w; __r.__i32[1] = __w; __r.__i32[2] = __w; __r.__i32[3] = __w;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set_epi16(short __e7, short __e6, short __e5, short __e4,
              short __e3, short __e2, short __e1, short __e0)
{
    __m128i __r;
    __r.__i16[0]=__e0; __r.__i16[1]=__e1; __r.__i16[2]=__e2; __r.__i16[3]=__e3;
    __r.__i16[4]=__e4; __r.__i16[5]=__e5; __r.__i16[6]=__e6; __r.__i16[7]=__e7;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi16(short __w)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = __w;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi8(char __w)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++) __r.__i8[__i] = __w;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set_epi64x(long long __e1, long long __e0)
{
    __m128i __r; __r.__i64[0] = __e0; __r.__i64[1] = __e1; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_set1_epi64x(long long __w)
{
    __m128i __r; __r.__i64[0] = __w; __r.__i64[1] = __w; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_setr_epi32(int __e0, int __e1, int __e2, int __e3)
{
    return _mm_set_epi32(__e3, __e2, __e1, __e0);
}

/* Integer load/store */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_load_si128(const __m128i *__p) { return *__p; }

static __inline__ __m128i __attribute__((__always_inline__))
_mm_loadu_si128(const __m128i *__p)
{
    __m128i __r;
    __builtin_memcpy(&__r, __p, 16);
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_si128(__m128i *__p, __m128i __a) { *__p = __a; }

static __inline__ void __attribute__((__always_inline__))
_mm_storeu_si128(__m128i *__p, __m128i __a) { __builtin_memcpy(__p, &__a, 16); }

static __inline__ void __attribute__((__always_inline__))
_mm_stream_si128(__m128i *__p, __m128i __a) { _mm_store_si128(__p, __a); }

/* Integer arithmetic */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_add_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++) __r.__i8[__i] = __a.__i8[__i] + __b.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_add_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = __a.__i16[__i] + __b.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_add_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = __a.__i32[__i] + __b.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_add_epi64(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i64[0] = __a.__i64[0] + __b.__i64[0]; __r.__i64[1] = __a.__i64[1] + __b.__i64[1];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sub_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++) __r.__i8[__i] = __a.__i8[__i] - __b.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sub_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = __a.__i16[__i] - __b.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sub_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = __a.__i32[__i] - __b.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sub_epi64(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i64[0] = __a.__i64[0] - __b.__i64[0]; __r.__i64[1] = __a.__i64[1] - __b.__i64[1];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_mullo_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = __a.__i16[__i] * __b.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_mulhi_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = (short)(((int)__a.__i16[__i] * (int)__b.__i16[__i]) >> 16);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_madd_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = (int)__a.__i16[2*__i] * (int)__b.__i16[2*__i]
                       + (int)__a.__i16[2*__i+1] * (int)__b.__i16[2*__i+1];
    return __r;
}

/* Integer logical */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_and_si128(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = __a.__i64[0] & __b.__i64[0]; __r.__i64[1] = __a.__i64[1] & __b.__i64[1]; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_andnot_si128(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = ~__a.__i64[0] & __b.__i64[0]; __r.__i64[1] = ~__a.__i64[1] & __b.__i64[1]; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_or_si128(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = __a.__i64[0] | __b.__i64[0]; __r.__i64[1] = __a.__i64[1] | __b.__i64[1]; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_xor_si128(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = __a.__i64[0] ^ __b.__i64[0]; __r.__i64[1] = __a.__i64[1] ^ __b.__i64[1]; return __r;
}

/* Integer shift */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_slli_epi16(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = (unsigned short)(__cnt >= 16 ? 0 : __a.__u16[__i] << __cnt);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_slli_epi32(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__u32[__i] = __cnt >= 32 ? 0 : __a.__u32[__i] << __cnt;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_slli_epi64(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++)
        __r.__u64[__i] = __cnt >= 64 ? 0 : __a.__u64[__i] << __cnt;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_srli_epi16(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = (unsigned short)(__cnt >= 16 ? 0 : __a.__u16[__i] >> __cnt);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_srli_epi32(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__u32[__i] = __cnt >= 32 ? 0 : __a.__u32[__i] >> __cnt;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_srli_epi64(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++)
        __r.__u64[__i] = __cnt >= 64 ? 0 : __a.__u64[__i] >> __cnt;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_srai_epi16(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __cnt >= 16 ? (__a.__i16[__i] >> 15) : (__a.__i16[__i] >> __cnt);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_srai_epi32(__m128i __a, int __cnt)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __cnt >= 32 ? (__a.__i32[__i] >> 31) : (__a.__i32[__i] >> __cnt);
    return __r;
}

/* Byte shift of 128-bit register */
#define _mm_slli_si128(__a, __n) __extension__({ \
    __m128i __src = (__a), __r; \
    __r.__i64[0] = 0; __r.__i64[1] = 0; \
    for (int __i = (__n); __i < 16; __i++) __r.__u8[__i] = __src.__u8[__i - (__n)]; \
    __r; \
})

#define _mm_srli_si128(__a, __n) __extension__({ \
    __m128i __src = (__a), __r; \
    __r.__i64[0] = 0; __r.__i64[1] = 0; \
    for (int __i = 0; __i + (__n) < 16; __i++) __r.__u8[__i] = __src.__u8[__i + (__n)]; \
    __r; \
})

#define _mm_bslli_si128 _mm_slli_si128
#define _mm_bsrli_si128 _mm_srli_si128

/* Integer compare */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__i8[__i] = __a.__i8[__i] == __b.__i8[__i] ? (signed char)-1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __a.__i16[__i] == __b.__i16[__i] ? (short)-1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] == __b.__i32[__i] ? -1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpgt_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__i8[__i] = __a.__i8[__i] > __b.__i8[__i] ? (signed char)-1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpgt_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __a.__i16[__i] > __b.__i16[__i] ? (short)-1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpgt_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] > __b.__i32[__i] ? -1 : 0;
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmplt_epi8(__m128i __a, __m128i __b) { return _mm_cmpgt_epi8(__b, __a); }

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmplt_epi16(__m128i __a, __m128i __b) { return _mm_cmpgt_epi16(__b, __a); }

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmplt_epi32(__m128i __a, __m128i __b) { return _mm_cmpgt_epi32(__b, __a); }

/* Min/Max */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epu8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__u8[__i] = __a.__u8[__i] > __b.__u8[__i] ? __a.__u8[__i] : __b.__u8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epu8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__u8[__i] = __a.__u8[__i] < __b.__u8[__i] ? __a.__u8[__i] : __b.__u8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __a.__i16[__i] > __b.__i16[__i] ? __a.__i16[__i] : __b.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __a.__i16[__i] < __b.__i16[__i] ? __a.__i16[__i] : __b.__i16[__i];
    return __r;
}

/* Movemask / extract */
static __inline__ int __attribute__((__always_inline__))
_mm_movemask_epi8(__m128i __a)
{
    int __r = 0;
    for (int __i = 0; __i < 16; __i++)
        if (__a.__i8[__i] < 0) __r |= (1 << __i);
    return __r;
}

#define _mm_extract_epi16(__a, __imm8) ((int)(unsigned short)((__a).__i16[(__imm8) & 7]))

#define _mm_insert_epi16(__a, __i, __imm8) __extension__({ \
    __m128i __r = (__a); \
    __r.__i16[(__imm8) & 7] = (short)(__i); \
    __r; \
})

/* Shuffle epi32 */
#define _mm_shuffle_epi32(__a, __imm8) __extension__({ \
    __m128i __r; \
    __r.__i32[0] = (__a).__i32[(__imm8) & 3]; \
    __r.__i32[1] = (__a).__i32[((__imm8)>>2) & 3]; \
    __r.__i32[2] = (__a).__i32[((__imm8)>>4) & 3]; \
    __r.__i32[3] = (__a).__i32[((__imm8)>>6) & 3]; \
    __r; \
})

#define _mm_shufflelo_epi16(__a, __imm8) __extension__({ \
    __m128i __r = (__a); \
    __r.__i16[0] = (__a).__i16[(__imm8) & 3]; \
    __r.__i16[1] = (__a).__i16[((__imm8)>>2) & 3]; \
    __r.__i16[2] = (__a).__i16[((__imm8)>>4) & 3]; \
    __r.__i16[3] = (__a).__i16[((__imm8)>>6) & 3]; \
    __r; \
})

#define _mm_shufflehi_epi16(__a, __imm8) __extension__({ \
    __m128i __r = (__a); \
    __r.__i16[4] = (__a).__i16[4 + ((__imm8) & 3)]; \
    __r.__i16[5] = (__a).__i16[4 + (((__imm8)>>2) & 3)]; \
    __r.__i16[6] = (__a).__i16[4 + (((__imm8)>>4) & 3)]; \
    __r.__i16[7] = (__a).__i16[4 + (((__imm8)>>6) & 3)]; \
    __r; \
})

/* Unpack integers */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpacklo_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        __r.__i8[2*__i] = __a.__i8[__i]; __r.__i8[2*__i+1] = __b.__i8[__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpackhi_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        __r.__i8[2*__i] = __a.__i8[8+__i]; __r.__i8[2*__i+1] = __b.__i8[8+__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpacklo_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        __r.__i16[2*__i] = __a.__i16[__i]; __r.__i16[2*__i+1] = __b.__i16[__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpackhi_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        __r.__i16[2*__i] = __a.__i16[4+__i]; __r.__i16[2*__i+1] = __b.__i16[4+__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpacklo_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i32[0] = __a.__i32[0]; __r.__i32[1] = __b.__i32[0];
    __r.__i32[2] = __a.__i32[1]; __r.__i32[3] = __b.__i32[1];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpackhi_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i32[0] = __a.__i32[2]; __r.__i32[1] = __b.__i32[2];
    __r.__i32[2] = __a.__i32[3]; __r.__i32[3] = __b.__i32[3];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpacklo_epi64(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = __a.__i64[0]; __r.__i64[1] = __b.__i64[0]; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_unpackhi_epi64(__m128i __a, __m128i __b)
{
    __m128i __r; __r.__i64[0] = __a.__i64[1]; __r.__i64[1] = __b.__i64[1]; return __r;
}

/* Pack with saturation */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_packs_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        int __v = __a.__i16[__i];
        __r.__i8[__i] = (signed char)(__v < -128 ? -128 : (__v > 127 ? 127 : __v));
    }
    for (int __i = 0; __i < 8; __i++) {
        int __v = __b.__i16[__i];
        __r.__i8[8+__i] = (signed char)(__v < -128 ? -128 : (__v > 127 ? 127 : __v));
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_packs_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        int __v = __a.__i32[__i];
        __r.__i16[__i] = (short)(__v < -32768 ? -32768 : (__v > 32767 ? 32767 : __v));
    }
    for (int __i = 0; __i < 4; __i++) {
        int __v = __b.__i32[__i];
        __r.__i16[4+__i] = (short)(__v < -32768 ? -32768 : (__v > 32767 ? 32767 : __v));
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_packus_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        int __v = __a.__i16[__i];
        __r.__u8[__i] = (unsigned char)(__v < 0 ? 0 : (__v > 255 ? 255 : __v));
    }
    for (int __i = 0; __i < 8; __i++) {
        int __v = __b.__i16[__i];
        __r.__u8[8+__i] = (unsigned char)(__v < 0 ? 0 : (__v > 255 ? 255 : __v));
    }
    return __r;
}

/* Conversion integer <-> float */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtps_epi32(__m128 __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = (int)__a.__f[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvttps_epi32(__m128 __a) { return _mm_cvtps_epi32(__a); }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cvtepi32_ps(__m128i __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = (float)__a.__i32[__i];
    return __r;
}

/* Conversion double <-> float */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_cvtpd_ps(__m128d __a)
{
    __m128 __r;
    __r.__f[0] = (float)__a.__d[0]; __r.__f[1] = (float)__a.__d[1];
    __r.__f[2] = 0; __r.__f[3] = 0;
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_cvtps_pd(__m128 __a)
{
    __m128d __r;
    __r.__d[0] = (double)__a.__f[0]; __r.__d[1] = (double)__a.__f[1];
    return __r;
}

/* Scalar extract */
static __inline__ int __attribute__((__always_inline__))
_mm_cvtsi128_si32(__m128i __a) { return __a.__i32[0]; }

#ifdef __x86_64__
static __inline__ long long __attribute__((__always_inline__))
_mm_cvtsi128_si64(__m128i __a) { return __a.__i64[0]; }
#endif

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtsi32_si128(int __a)
{
    __m128i __r = _mm_setzero_si128(); __r.__i32[0] = __a; return __r;
}

#ifdef __x86_64__
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtsi64_si128(long long __a)
{
    __m128i __r = _mm_setzero_si128(); __r.__i64[0] = __a; return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtsi64x_si128(long long __a) { return _mm_cvtsi64_si128(__a); }
#endif

/* Cast between types (reinterpret) */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_castsi128_ps(__m128i __a)
{
    __m128 __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_castps_si128(__m128 __a)
{
    __m128i __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_castsi128_pd(__m128i __a)
{
    __m128d __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_castpd_si128(__m128d __a)
{
    __m128i __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_castpd_ps(__m128d __a)
{
    __m128 __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_castps_pd(__m128 __a)
{
    __m128d __r;
    __builtin_memcpy(&__r, &__a, 16);
    return __r;
}

/* Fences */
static __inline__ void __attribute__((__always_inline__)) _mm_lfence(void) {}
static __inline__ void __attribute__((__always_inline__)) _mm_mfence(void) {}

/* Average */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_avg_epu8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__u8[__i] = (unsigned char)(((unsigned)__a.__u8[__i] + (unsigned)__b.__u8[__i] + 1) >> 1);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_avg_epu16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = (unsigned short)(((unsigned)__a.__u16[__i] + (unsigned)__b.__u16[__i] + 1) >> 1);
    return __r;
}

/* SAD */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_sad_epu8(__m128i __a, __m128i __b)
{
    __m128i __r = _mm_setzero_si128();
    int __s0 = 0, __s1 = 0;
    for (int __i = 0; __i < 8; __i++) {
        int __d = (int)__a.__u8[__i] - (int)__b.__u8[__i];
        __s0 += __d < 0 ? -__d : __d;
    }
    for (int __i = 8; __i < 16; __i++) {
        int __d = (int)__a.__u8[__i] - (int)__b.__u8[__i];
        __s1 += __d < 0 ? -__d : __d;
    }
    __r.__u16[0] = (unsigned short)__s0;
    __r.__u16[4] = (unsigned short)__s1;
    return __r;
}

/* Multiply unsigned */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_mul_epu32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__u64[0] = (unsigned long long)__a.__u32[0] * (unsigned long long)__b.__u32[0];
    __r.__u64[1] = (unsigned long long)__a.__u32[2] * (unsigned long long)__b.__u32[2];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_mulhi_epu16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = (unsigned short)(((unsigned)__a.__u16[__i] * (unsigned)__b.__u16[__i]) >> 16);
    return __r;
}

#endif /* _EMMINTRIN_H_INCLUDED */
