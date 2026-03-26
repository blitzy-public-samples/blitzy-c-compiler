/* BCC bundled immintrin.h — umbrella header for all x86 SIMD intrinsics (struct-based) */
#ifndef _IMMINTRIN_H_INCLUDED
#define _IMMINTRIN_H_INCLUDED
#include <nmmintrin.h>

/* ===================================================================
   AVX 256-bit types — struct/union based (no vector_size)
   =================================================================== */
typedef struct __m256_struct {
    float __f[8];
} __attribute__((aligned(32))) __m256;

typedef struct __m256d_struct {
    double __d[4];
} __attribute__((aligned(32))) __m256d;

typedef union __m256i_union {
    long long          __i64[4];
    unsigned long long __u64[4];
    int                __i32[8];
    unsigned int       __u32[8];
    short              __i16[16];
    unsigned short     __u16[16];
    signed char        __i8[32];
    unsigned char      __u8[32];
} __attribute__((aligned(32))) __m256i;

/* ===================================================================
   AVX: Set / Zero
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_setzero_ps(void)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = 0.0f;
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_setzero_pd(void)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = 0.0;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_setzero_si256(void)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = 0;
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_set_ps(float __e7, float __e6, float __e5, float __e4,
              float __e3, float __e2, float __e1, float __e0)
{
    __m256 __r;
    __r.__f[0] = __e0; __r.__f[1] = __e1; __r.__f[2] = __e2; __r.__f[3] = __e3;
    __r.__f[4] = __e4; __r.__f[5] = __e5; __r.__f[6] = __e6; __r.__f[7] = __e7;
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_set_pd(double __e3, double __e2, double __e1, double __e0)
{
    __m256d __r;
    __r.__d[0] = __e0; __r.__d[1] = __e1; __r.__d[2] = __e2; __r.__d[3] = __e3;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_set_epi32(int __e7, int __e6, int __e5, int __e4,
                 int __e3, int __e2, int __e1, int __e0)
{
    __m256i __r;
    __r.__i32[0] = __e0; __r.__i32[1] = __e1; __r.__i32[2] = __e2; __r.__i32[3] = __e3;
    __r.__i32[4] = __e4; __r.__i32[5] = __e5; __r.__i32[6] = __e6; __r.__i32[7] = __e7;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_set_epi64x(long long __e3, long long __e2, long long __e1, long long __e0)
{
    __m256i __r;
    __r.__i64[0] = __e0; __r.__i64[1] = __e1; __r.__i64[2] = __e2; __r.__i64[3] = __e3;
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_set1_ps(float __w)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __w;
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_set1_pd(double __w)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __w;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_set1_epi32(int __w)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = __w;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_set1_epi64x(long long __w)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = __w;
    return __r;
}

/* ===================================================================
   AVX: Load / Store
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_load_ps(const float *__p)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __p[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_loadu_ps(const float *__p)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __p[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_load_pd(const double *__p)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __p[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_loadu_pd(const double *__p)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __p[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_load_si256(const __m256i *__p) { return *__p; }

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_loadu_si256(const __m256i *__p)
{
    __m256i __r;
    const unsigned char *__src = (const unsigned char *)__p;
    unsigned char *__dst = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __dst[__i] = __src[__i];
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm256_store_ps(float *__p, __m256 __a)
{
    for (int __i = 0; __i < 8; __i++) __p[__i] = __a.__f[__i];
}

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_ps(float *__p, __m256 __a)
{
    for (int __i = 0; __i < 8; __i++) __p[__i] = __a.__f[__i];
}

static __inline__ void __attribute__((__always_inline__))
_mm256_store_pd(double *__p, __m256d __a)
{
    for (int __i = 0; __i < 4; __i++) __p[__i] = __a.__d[__i];
}

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_pd(double *__p, __m256d __a)
{
    for (int __i = 0; __i < 4; __i++) __p[__i] = __a.__d[__i];
}

static __inline__ void __attribute__((__always_inline__))
_mm256_store_si256(__m256i *__p, __m256i __a) { *__p = __a; }

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_si256(__m256i *__p, __m256i __a)
{
    unsigned char *__dst = (unsigned char *)__p;
    const unsigned char *__src = (const unsigned char *)&__a;
    for (int __i = 0; __i < 32; __i++) __dst[__i] = __src[__i];
}

/* ===================================================================
   AVX: Float arithmetic
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_add_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] + __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_sub_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] - __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_mul_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] * __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_div_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] / __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_min_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] < __b.__f[__i] ? __a.__f[__i] : __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_max_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] > __b.__f[__i] ? __a.__f[__i] : __b.__f[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_sqrt_ps(__m256 __a)
{
    __m256 __r;
    /* software sqrt via Newton-Raphson: sufficient for fallback use */
    for (int __i = 0; __i < 8; __i++) {
        float __v = __a.__f[__i];
        if (__v <= 0.0f) { __r.__f[__i] = 0.0f; continue; }
        union { float f; unsigned int u; } __u; __u.f = __v;
        __u.u = ((__u.u - 0x3f800000u) >> 1) + 0x3f800000u; /* initial guess */
        float __x = __u.f;
        __x = 0.5f * (__x + __v / __x); __x = 0.5f * (__x + __v / __x);
        __x = 0.5f * (__x + __v / __x);
        __r.__f[__i] = __x;
    }
    return __r;
}

/* ===================================================================
   AVX: Double arithmetic
   =================================================================== */
static __inline__ __m256d __attribute__((__always_inline__))
_mm256_add_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] + __b.__d[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_sub_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] - __b.__d[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_mul_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] * __b.__d[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_div_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] / __b.__d[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_min_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] < __b.__d[__i] ? __a.__d[__i] : __b.__d[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_max_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __a.__d[__i] > __b.__d[__i] ? __a.__d[__i] : __b.__d[__i];
    return __r;
}

/* ===================================================================
   AVX: Logical (float/double via int union trick)
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_and_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    union { float f; unsigned int u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 8; __i++) {
        __ua.f = __a.__f[__i]; __ub.f = __b.__f[__i];
        __ur.u = __ua.u & __ub.u; __r.__f[__i] = __ur.f;
    }
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_andnot_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    union { float f; unsigned int u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 8; __i++) {
        __ua.f = __a.__f[__i]; __ub.f = __b.__f[__i];
        __ur.u = (~__ua.u) & __ub.u; __r.__f[__i] = __ur.f;
    }
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_or_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    union { float f; unsigned int u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 8; __i++) {
        __ua.f = __a.__f[__i]; __ub.f = __b.__f[__i];
        __ur.u = __ua.u | __ub.u; __r.__f[__i] = __ur.f;
    }
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_xor_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    union { float f; unsigned int u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 8; __i++) {
        __ua.f = __a.__f[__i]; __ub.f = __b.__f[__i];
        __ur.u = __ua.u ^ __ub.u; __r.__f[__i] = __ur.f;
    }
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_and_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    union { double d; unsigned long long u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 4; __i++) {
        __ua.d = __a.__d[__i]; __ub.d = __b.__d[__i];
        __ur.u = __ua.u & __ub.u; __r.__d[__i] = __ur.d;
    }
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_or_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    union { double d; unsigned long long u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 4; __i++) {
        __ua.d = __a.__d[__i]; __ub.d = __b.__d[__i];
        __ur.u = __ua.u | __ub.u; __r.__d[__i] = __ur.d;
    }
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_xor_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    union { double d; unsigned long long u; } __ua, __ub, __ur;
    for (int __i = 0; __i < 4; __i++) {
        __ua.d = __a.__d[__i]; __ub.d = __b.__d[__i];
        __ur.u = __ua.u ^ __ub.u; __r.__d[__i] = __ur.d;
    }
    return __r;
}

/* ===================================================================
   AVX: Compare
   =================================================================== */
#define _CMP_EQ_OQ    0x00
#define _CMP_LT_OS    0x01
#define _CMP_LE_OS    0x02
#define _CMP_UNORD_Q  0x03
#define _CMP_NEQ_UQ   0x04
#define _CMP_NLT_US   0x05
#define _CMP_NLE_US   0x06
#define _CMP_ORD_Q    0x07
#define _CMP_EQ_UQ    0x08
#define _CMP_NGE_US   0x09
#define _CMP_NGT_US   0x0a
#define _CMP_FALSE_OQ 0x0b
#define _CMP_NEQ_OQ   0x0c
#define _CMP_GE_OS    0x0d
#define _CMP_GT_OS    0x0e
#define _CMP_TRUE_UQ  0x0f

#define _mm256_cmp_ps(__a, __b, __p) __extension__({ \
    __m256 __result; \
    union { float f; unsigned int u; } __tu; \
    for (int __i = 0; __i < 8; __i++) { \
        int __c = 0; \
        switch ((__p) & 0x0f) { \
            case 0x00: __c = (__a).__f[__i] == (__b).__f[__i]; break; \
            case 0x01: __c = (__a).__f[__i] < (__b).__f[__i]; break; \
            case 0x02: __c = (__a).__f[__i] <= (__b).__f[__i]; break; \
            case 0x04: __c = (__a).__f[__i] != (__b).__f[__i]; break; \
            case 0x05: __c = !((__a).__f[__i] < (__b).__f[__i]); break; \
            case 0x06: __c = !((__a).__f[__i] <= (__b).__f[__i]); break; \
            case 0x0d: __c = (__a).__f[__i] >= (__b).__f[__i]; break; \
            case 0x0e: __c = (__a).__f[__i] > (__b).__f[__i]; break; \
            default: __c = 0; break; \
        } \
        __tu.u = __c ? 0xFFFFFFFFu : 0u; \
        __result.__f[__i] = __tu.f; \
    } \
    __result; })

/* ===================================================================
   AVX: Shuffle / Permute / Broadcast
   =================================================================== */
#define _mm256_shuffle_ps(__a, __b, __imm) __extension__({ \
    __m256 __result; \
    __result.__f[0] = (__a).__f[((__imm) >> 0) & 3]; \
    __result.__f[1] = (__a).__f[((__imm) >> 2) & 3]; \
    __result.__f[2] = (__b).__f[((__imm) >> 4) & 3]; \
    __result.__f[3] = (__b).__f[((__imm) >> 6) & 3]; \
    __result.__f[4] = (__a).__f[4 + (((__imm) >> 0) & 3)]; \
    __result.__f[5] = (__a).__f[4 + (((__imm) >> 2) & 3)]; \
    __result.__f[6] = (__b).__f[4 + (((__imm) >> 4) & 3)]; \
    __result.__f[7] = (__b).__f[4 + (((__imm) >> 6) & 3)]; \
    __result; })

#define _mm256_permute_ps(__a, __imm) __extension__({ \
    __m256 __result; \
    __result.__f[0] = (__a).__f[((__imm) >> 0) & 3]; \
    __result.__f[1] = (__a).__f[((__imm) >> 2) & 3]; \
    __result.__f[2] = (__a).__f[((__imm) >> 4) & 3]; \
    __result.__f[3] = (__a).__f[((__imm) >> 6) & 3]; \
    __result.__f[4] = (__a).__f[4 + (((__imm) >> 0) & 3)]; \
    __result.__f[5] = (__a).__f[4 + (((__imm) >> 2) & 3)]; \
    __result.__f[6] = (__a).__f[4 + (((__imm) >> 4) & 3)]; \
    __result.__f[7] = (__a).__f[4 + (((__imm) >> 6) & 3)]; \
    __result; })

#define _mm256_permute2f128_ps(__a, __b, __imm) __extension__({ \
    __m256 __result; \
    const float *__sa = ((__imm) & 2) ? (__b).__f : (__a).__f; \
    const float *__sb = ((__imm) & 0x20) ? (__b).__f : (__a).__f; \
    int __oa = ((__imm) & 1) ? 4 : 0, __ob = ((__imm) & 0x10) ? 4 : 0; \
    if ((__imm) & 8) { for (int __i=0;__i<4;__i++) __result.__f[__i] = 0; } \
    else { for (int __i=0;__i<4;__i++) __result.__f[__i] = __sa[__oa+__i]; } \
    if ((__imm) & 0x80) { for (int __i=0;__i<4;__i++) __result.__f[4+__i] = 0; } \
    else { for (int __i=0;__i<4;__i++) __result.__f[4+__i] = __sb[__ob+__i]; } \
    __result; })

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_broadcast_ss(const float *__p)
{
    __m256 __r;
    float __v = *__p;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __v;
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_broadcast_sd(const double *__p)
{
    __m256d __r;
    double __v = *__p;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = __v;
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_broadcast_ps(const __m128 *__p)
{
    __m256 __r;
    for (int __i = 0; __i < 4; __i++) { __r.__f[__i] = __p->__f[__i]; __r.__f[__i+4] = __p->__f[__i]; }
    return __r;
}

/* ===================================================================
   AVX: Unpack
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_unpackhi_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    __r.__f[0]=__a.__f[2]; __r.__f[1]=__b.__f[2]; __r.__f[2]=__a.__f[3]; __r.__f[3]=__b.__f[3];
    __r.__f[4]=__a.__f[6]; __r.__f[5]=__b.__f[6]; __r.__f[6]=__a.__f[7]; __r.__f[7]=__b.__f[7];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_unpacklo_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    __r.__f[0]=__a.__f[0]; __r.__f[1]=__b.__f[0]; __r.__f[2]=__a.__f[1]; __r.__f[3]=__b.__f[1];
    __r.__f[4]=__a.__f[4]; __r.__f[5]=__b.__f[4]; __r.__f[6]=__a.__f[5]; __r.__f[7]=__b.__f[5];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_unpackhi_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    __r.__d[0]=__a.__d[1]; __r.__d[1]=__b.__d[1]; __r.__d[2]=__a.__d[3]; __r.__d[3]=__b.__d[3];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_unpacklo_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    __r.__d[0]=__a.__d[0]; __r.__d[1]=__b.__d[0]; __r.__d[2]=__a.__d[2]; __r.__d[3]=__b.__d[2];
    return __r;
}

/* ===================================================================
   AVX: Movemask
   =================================================================== */
static __inline__ int __attribute__((__always_inline__))
_mm256_movemask_ps(__m256 __a)
{
    int __r = 0;
    union { float f; unsigned int u; } __t;
    for (int __i = 0; __i < 8; __i++) {
        __t.f = __a.__f[__i];
        if (__t.u & 0x80000000u) __r |= (1 << __i);
    }
    return __r;
}

static __inline__ int __attribute__((__always_inline__))
_mm256_movemask_pd(__m256d __a)
{
    int __r = 0;
    union { double d; unsigned long long u; } __t;
    for (int __i = 0; __i < 4; __i++) {
        __t.d = __a.__d[__i];
        if (__t.u & 0x8000000000000000ULL) __r |= (1 << __i);
    }
    return __r;
}

/* ===================================================================
   AVX: Conversion
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_cvtepi32_ps(__m256i __a)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = (float)__a.__i32[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_cvtps_epi32(__m256 __a)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (int)__a.__f[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_cvttps_epi32(__m256 __a)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (int)__a.__f[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_cvtps_pd(__m128 __a)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++) __r.__d[__i] = (double)__a.__f[__i];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm256_cvtpd_ps(__m256d __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = (float)__a.__d[__i];
    return __r;
}

/* ===================================================================
   AVX: Extract / Insert 128-bit
   =================================================================== */
#define _mm256_extractf128_ps(__a, __imm) __extension__({ \
    __m128 __result; \
    int __off = ((__imm) & 1) ? 4 : 0; \
    for (int __i = 0; __i < 4; __i++) __result.__f[__i] = (__a).__f[__off + __i]; \
    __result; })

#define _mm256_extractf128_pd(__a, __imm) __extension__({ \
    __m128d __result; \
    int __off = ((__imm) & 1) ? 2 : 0; \
    for (int __i = 0; __i < 2; __i++) __result.__d[__i] = (__a).__d[__off + __i]; \
    __result; })

#define _mm256_extractf128_si256(__a, __imm) __extension__({ \
    __m128i __result; \
    int __off = ((__imm) & 1) ? 2 : 0; \
    for (int __i = 0; __i < 2; __i++) __result.__i64[__i] = (__a).__i64[__off + __i]; \
    __result; })

#define _mm256_insertf128_ps(__a, __b, __imm) __extension__({ \
    __m256 __result = (__a); \
    int __off = ((__imm) & 1) ? 4 : 0; \
    for (int __i = 0; __i < 4; __i++) __result.__f[__off + __i] = (__b).__f[__i]; \
    __result; })

#define _mm256_insertf128_pd(__a, __b, __imm) __extension__({ \
    __m256d __result = (__a); \
    int __off = ((__imm) & 1) ? 2 : 0; \
    for (int __i = 0; __i < 2; __i++) __result.__d[__off + __i] = (__b).__d[__i]; \
    __result; })

#define _mm256_insertf128_si256(__a, __b, __imm) __extension__({ \
    __m256i __result = (__a); \
    int __off = ((__imm) & 1) ? 2 : 0; \
    for (int __i = 0; __i < 2; __i++) __result.__i64[__off + __i] = (__b).__i64[__i]; \
    __result; })

/* ===================================================================
   AVX: Cast between types (reinterpret, zero-cost)
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_castpd_ps(__m256d __a)
{
    __m256 __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_castps_pd(__m256 __a)
{
    __m256d __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_castps_si256(__m256 __a)
{
    __m256i __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_castsi256_ps(__m256i __a)
{
    __m256 __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_castpd_si256(__m256d __a)
{
    __m256i __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_castsi256_pd(__m256i __a)
{
    __m256d __r;
    const unsigned char *__s = (const unsigned char *)&__a;
    unsigned char *__d = (unsigned char *)&__r;
    for (int __i = 0; __i < 32; __i++) __d[__i] = __s[__i];
    return __r;
}

/* 256 <-> 128 casts */
static __inline__ __m128 __attribute__((__always_inline__))
_mm256_castps256_ps128(__m256 __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = __a.__f[__i];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm256_castpd256_pd128(__m256d __a)
{
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) __r.__d[__i] = __a.__d[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm256_castsi256_si128(__m256i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = __a.__i64[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_castps128_ps256(__m128 __a)
{
    __m256 __r = _mm256_setzero_ps();
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = __a.__f[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_castpd128_pd256(__m128d __a)
{
    __m256d __r = _mm256_setzero_pd();
    for (int __i = 0; __i < 2; __i++) __r.__d[__i] = __a.__d[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_castsi128_si256(__m128i __a)
{
    __m256i __r = _mm256_setzero_si256();
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = __a.__i64[__i];
    return __r;
}

/* ===================================================================
   AVX: Horizontal add
   =================================================================== */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_hadd_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    __r.__f[0] = __a.__f[0]+__a.__f[1]; __r.__f[1] = __a.__f[2]+__a.__f[3];
    __r.__f[2] = __b.__f[0]+__b.__f[1]; __r.__f[3] = __b.__f[2]+__b.__f[3];
    __r.__f[4] = __a.__f[4]+__a.__f[5]; __r.__f[5] = __a.__f[6]+__a.__f[7];
    __r.__f[6] = __b.__f[4]+__b.__f[5]; __r.__f[7] = __b.__f[6]+__b.__f[7];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_hadd_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    __r.__d[0] = __a.__d[0]+__a.__d[1]; __r.__d[1] = __b.__d[0]+__b.__d[1];
    __r.__d[2] = __a.__d[2]+__a.__d[3]; __r.__d[3] = __b.__d[2]+__b.__d[3];
    return __r;
}

/* ===================================================================
   AVX: Zeroupper (no-op in software fallback)
   =================================================================== */
static __inline__ void __attribute__((__always_inline__))
_mm256_zeroupper(void) { /* no-op */ }

static __inline__ void __attribute__((__always_inline__))
_mm256_zeroall(void) { /* no-op */ }

/* ===================================================================
   AVX2: Integer operations on 256-bit
   =================================================================== */
static __inline__ __m256i __attribute__((__always_inline__))
_mm256_add_epi8(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 32; __i++) __r.__i8[__i] = __a.__i8[__i] + __b.__i8[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_add_epi16(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 16; __i++) __r.__i16[__i] = __a.__i16[__i] + __b.__i16[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_add_epi32(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = __a.__i32[__i] + __b.__i32[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_add_epi64(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = __a.__i64[__i] + __b.__i64[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_sub_epi32(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = __a.__i32[__i] - __b.__i32[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_and_si256(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = __a.__i64[__i] & __b.__i64[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_or_si256(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = __a.__i64[__i] | __b.__i64[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_xor_si256(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i64[__i] = __a.__i64[__i] ^ __b.__i64[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_cmpeq_epi32(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (__a.__i32[__i] == __b.__i32[__i]) ? -1 : 0;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_cmpgt_epi32(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (__a.__i32[__i] > __b.__i32[__i]) ? -1 : 0;
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_mullo_epi32(__m256i __a, __m256i __b)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = __a.__i32[__i] * __b.__i32[__i];
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_slli_epi32(__m256i __a, int __count)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (__count >= 32) ? 0 : (__a.__i32[__i] << __count);
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_srli_epi32(__m256i __a, int __count)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__u32[__i] = (__count >= 32) ? 0 : (__a.__u32[__i] >> __count);
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_srai_epi32(__m256i __a, int __count)
{
    __m256i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i32[__i] = (__count >= 32) ? (__a.__i32[__i] >> 31) : (__a.__i32[__i] >> __count);
    return __r;
}

/* AVX2 permute */
#define _mm256_permute4x64_epi64(__a, __imm) __extension__({ \
    __m256i __result; \
    __result.__i64[0] = (__a).__i64[((__imm) >> 0) & 3]; \
    __result.__i64[1] = (__a).__i64[((__imm) >> 2) & 3]; \
    __result.__i64[2] = (__a).__i64[((__imm) >> 4) & 3]; \
    __result.__i64[3] = (__a).__i64[((__imm) >> 6) & 3]; \
    __result; })

static __inline__ int __attribute__((__always_inline__))
_mm256_movemask_epi8(__m256i __a)
{
    int __r = 0;
    for (int __i = 0; __i < 32; __i++)
        if (__a.__i8[__i] < 0) __r |= (1 << __i);
    return __r;
}

/* ===================================================================
   BMI / BMI2 / LZCNT / POPCNT builtins (scalar, using C fallback)
   =================================================================== */
static __inline__ unsigned int __attribute__((__always_inline__))
_bzhi_u32(unsigned int __x, unsigned int __idx)
{
    return __x & ((1u << (__idx & 31)) - 1u);
}

static __inline__ unsigned int __attribute__((__always_inline__))
_pdep_u32(unsigned int __x, unsigned int __mask)
{
    unsigned int __res = 0;
    for (unsigned int __b = 1; __mask; __b <<= 1) {
        if (__x & __b) __res |= __mask & (unsigned int)(-(int)__mask);
        __mask &= __mask - 1;
    }
    return __res;
}

static __inline__ unsigned int __attribute__((__always_inline__))
_pext_u32(unsigned int __x, unsigned int __mask)
{
    unsigned int __res = 0, __b = 1;
    while (__mask) {
        if (__x & (__mask & (unsigned int)(-(int)__mask))) __res |= __b;
        __mask &= __mask - 1; __b <<= 1;
    }
    return __res;
}

static __inline__ unsigned int __attribute__((__always_inline__))
_blsi_u32(unsigned int __x) { return __x & (unsigned int)(-(int)__x); }

static __inline__ unsigned int __attribute__((__always_inline__))
_blsmsk_u32(unsigned int __x) { return __x ^ (__x - 1); }

static __inline__ unsigned int __attribute__((__always_inline__))
_blsr_u32(unsigned int __x) { return __x & (__x - 1); }

static __inline__ unsigned int __attribute__((__always_inline__))
_tzcnt_u32(unsigned int __x)
{
    if (__x == 0) return 32;
    unsigned int __n = 0;
    while (!(__x & 1)) { __n++; __x >>= 1; }
    return __n;
}

static __inline__ unsigned int __attribute__((__always_inline__))
_lzcnt_u32(unsigned int __x)
{
    if (__x == 0) return 32;
    unsigned int __n = 0;
    while (!(__x & 0x80000000u)) { __n++; __x <<= 1; }
    return __n;
}

/* FMA (software fallback) */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_fmadd_ps(__m128 __a, __m128 __b, __m128 __c)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = __a.__f[__i] * __b.__f[__i] + __c.__f[__i];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_fmadd_pd(__m128d __a, __m128d __b, __m128d __c)
{
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) __r.__d[__i] = __a.__d[__i] * __b.__d[__i] + __c.__d[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_fmadd_ps(__m256 __a, __m256 __b, __m256 __c)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++) __r.__f[__i] = __a.__f[__i] * __b.__f[__i] + __c.__f[__i];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_fmsub_ps(__m128 __a, __m128 __b, __m128 __c)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = __a.__f[__i] * __b.__f[__i] - __c.__f[__i];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_fnmadd_ps(__m128 __a, __m128 __b, __m128 __c)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) __r.__f[__i] = -__a.__f[__i] * __b.__f[__i] + __c.__f[__i];
    return __r;
}

#endif /* _IMMINTRIN_H_INCLUDED */
