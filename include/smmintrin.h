/* BCC bundled smmintrin.h — SSE4.1 intrinsics (struct-based fallback) */
#ifndef _SMMINTRIN_H_INCLUDED
#define _SMMINTRIN_H_INCLUDED
#include <tmmintrin.h>

/* Rounding mode constants */
#define _MM_FROUND_TO_NEAREST_INT  0x00
#define _MM_FROUND_TO_NEG_INF     0x01
#define _MM_FROUND_TO_POS_INF     0x02
#define _MM_FROUND_TO_ZERO        0x03
#define _MM_FROUND_CUR_DIRECTION  0x04
#define _MM_FROUND_RAISE_EXC      0x00
#define _MM_FROUND_NO_EXC         0x08
#define _MM_FROUND_NINT    (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_FLOOR   (_MM_FROUND_TO_NEG_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_CEIL    (_MM_FROUND_TO_POS_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_TRUNC   (_MM_FROUND_TO_ZERO | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_RINT    (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_NEARBYINT (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC)

/* --- Blend --- */
#define _mm_blend_ps(__a, __b, __imm) __extension__({ \
    __m128 __result; \
    __result.__f[0] = ((__imm) & 1) ? (__b).__f[0] : (__a).__f[0]; \
    __result.__f[1] = ((__imm) & 2) ? (__b).__f[1] : (__a).__f[1]; \
    __result.__f[2] = ((__imm) & 4) ? (__b).__f[2] : (__a).__f[2]; \
    __result.__f[3] = ((__imm) & 8) ? (__b).__f[3] : (__a).__f[3]; \
    __result; })

#define _mm_blend_pd(__a, __b, __imm) __extension__({ \
    __m128d __result; \
    __result.__d[0] = ((__imm) & 1) ? (__b).__d[0] : (__a).__d[0]; \
    __result.__d[1] = ((__imm) & 2) ? (__b).__d[1] : (__a).__d[1]; \
    __result; })

#define _mm_blend_epi16(__a, __b, __imm) __extension__({ \
    __m128i __result; \
    for (int __i = 0; __i < 8; __i++) \
        __result.__i16[__i] = ((__imm) & (1 << __i)) ? (__b).__i16[__i] : (__a).__i16[__i]; \
    __result; })

static __inline__ __m128 __attribute__((__always_inline__))
_mm_blendv_ps(__m128 __a, __m128 __b, __m128 __mask)
{
    __m128 __r;
    union { float f; unsigned int u; } __mu;
    for (int __i = 0; __i < 4; __i++) {
        __mu.f = __mask.__f[__i];
        __r.__f[__i] = (__mu.u & 0x80000000u) ? __b.__f[__i] : __a.__f[__i];
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_blendv_pd(__m128d __a, __m128d __b, __m128d __mask)
{
    __m128d __r;
    union { double d; unsigned long long u; } __mu;
    for (int __i = 0; __i < 2; __i++) {
        __mu.d = __mask.__d[__i];
        __r.__d[__i] = (__mu.u & 0x8000000000000000ULL) ? __b.__d[__i] : __a.__d[__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_blendv_epi8(__m128i __a, __m128i __b, __m128i __mask)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__u8[__i] = (__mask.__i8[__i] < 0) ? __b.__u8[__i] : __a.__u8[__i];
    return __r;
}

/* --- Dot product --- */
#define _mm_dp_ps(__a, __b, __imm) __extension__({ \
    __m128 __result = _mm_setzero_ps(); \
    float __sum = 0.0f; \
    for (int __i = 0; __i < 4; __i++) \
        if ((__imm) & (0x10 << __i)) __sum += (__a).__f[__i] * (__b).__f[__i]; \
    for (int __i = 0; __i < 4; __i++) \
        __result.__f[__i] = ((__imm) & (1 << __i)) ? __sum : 0.0f; \
    __result; })

#define _mm_dp_pd(__a, __b, __imm) __extension__({ \
    __m128d __result; \
    double __sum = 0.0; \
    if ((__imm) & 0x10) __sum += (__a).__d[0] * (__b).__d[0]; \
    if ((__imm) & 0x20) __sum += (__a).__d[1] * (__b).__d[1]; \
    __result.__d[0] = ((__imm) & 1) ? __sum : 0.0; \
    __result.__d[1] = ((__imm) & 2) ? __sum : 0.0; \
    __result; })

/* --- Min/Max integer --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__i8[__i] = __a.__i8[__i] < __b.__i8[__i] ? __a.__i8[__i] : __b.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__i8[__i] = __a.__i8[__i] > __b.__i8[__i] ? __a.__i8[__i] : __b.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epu16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = __a.__u16[__i] < __b.__u16[__i] ? __a.__u16[__i] : __b.__u16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epu16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__u16[__i] = __a.__u16[__i] > __b.__u16[__i] ? __a.__u16[__i] : __b.__u16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] < __b.__i32[__i] ? __a.__i32[__i] : __b.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] > __b.__i32[__i] ? __a.__i32[__i] : __b.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_min_epu32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__u32[__i] = __a.__u32[__i] < __b.__u32[__i] ? __a.__u32[__i] : __b.__u32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_max_epu32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__u32[__i] = __a.__u32[__i] > __b.__u32[__i] ? __a.__u32[__i] : __b.__u32[__i];
    return __r;
}

/* --- Integer extend (sign/zero) --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi8_epi16(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = (short)__a.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi8_epi32(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = (int)__a.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi8_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi16_epi32(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = (int)__a.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi16_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepi32_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu8_epi16(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) __r.__i16[__i] = (short)__a.__u8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu8_epi32(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = (int)__a.__u8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu8_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__u8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu16_epi32(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) __r.__i32[__i] = (int)__a.__u16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu16_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__u16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_cvtepu32_epi64(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++) __r.__i64[__i] = (long long)__a.__u32[__i];
    return __r;
}

/* --- Multiply 32-bit integers --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_mullo_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] * __b.__i32[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_mul_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i64[0] = (long long)__a.__i32[0] * (long long)__b.__i32[0];
    __r.__i64[1] = (long long)__a.__i32[2] * (long long)__b.__i32[2];
    return __r;
}

/* --- Extract / Insert --- */
#define _mm_extract_epi8(__a, __imm)  ((int)(unsigned char)((__a).__u8[(__imm) & 15]))
#define _mm_extract_epi32(__a, __imm) ((__a).__i32[(__imm) & 3])
#define _mm_extract_epi64(__a, __imm) ((__a).__i64[(__imm) & 1])

#define _mm_extract_ps(__a, __imm) __extension__({ \
    union { float __f; int __i; } __u; \
    __u.__f = (__a).__f[(__imm) & 3]; \
    __u.__i; })

#define _mm_insert_epi8(__a, __val, __imm) __extension__({ \
    __m128i __result = (__a); \
    __result.__u8[(__imm) & 15] = (unsigned char)(__val); \
    __result; })

#define _mm_insert_epi32(__a, __val, __imm) __extension__({ \
    __m128i __result = (__a); \
    __result.__i32[(__imm) & 3] = (__val); \
    __result; })

#define _mm_insert_epi64(__a, __val, __imm) __extension__({ \
    __m128i __result = (__a); \
    __result.__i64[(__imm) & 1] = (__val); \
    __result; })

#define _mm_insert_ps(__a, __b, __imm) __extension__({ \
    __m128 __result; \
    float __srcval = (__b).__f[((__imm) >> 6) & 3]; \
    __result = (__a); \
    __result.__f[((__imm) >> 4) & 3] = __srcval; \
    if ((__imm) & 1) __result.__f[0] = 0.0f; \
    if ((__imm) & 2) __result.__f[1] = 0.0f; \
    if ((__imm) & 4) __result.__f[2] = 0.0f; \
    if ((__imm) & 8) __result.__f[3] = 0.0f; \
    __result; })

/* --- Rounding (ceil/floor simplified) --- */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_ceil_ps(__m128 __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        float __v = __a.__f[__i];
        int __iv = (int)__v;
        __r.__f[__i] = (float)(__v > (float)__iv ? __iv + 1 : __iv);
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_floor_ps(__m128 __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        float __v = __a.__f[__i];
        int __iv = (int)__v;
        __r.__f[__i] = (float)(__v < (float)__iv ? __iv - 1 : __iv);
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_ceil_pd(__m128d __a)
{
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        double __v = __a.__d[__i];
        long long __iv = (long long)__v;
        __r.__d[__i] = (double)(__v > (double)__iv ? __iv + 1 : __iv);
    }
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_floor_pd(__m128d __a)
{
    __m128d __r;
    for (int __i = 0; __i < 2; __i++) {
        double __v = __a.__d[__i];
        long long __iv = (long long)__v;
        __r.__d[__i] = (double)(__v < (double)__iv ? __iv - 1 : __iv);
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_ceil_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a;
    float __v = __b.__f[0]; int __iv = (int)__v;
    __r.__f[0] = (float)(__v > (float)__iv ? __iv + 1 : __iv);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_floor_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a;
    float __v = __b.__f[0]; int __iv = (int)__v;
    __r.__f[0] = (float)(__v < (float)__iv ? __iv - 1 : __iv);
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_ceil_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a;
    double __v = __b.__d[0]; long long __iv = (long long)__v;
    __r.__d[0] = (double)(__v > (double)__iv ? __iv + 1 : __iv);
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_floor_sd(__m128d __a, __m128d __b)
{
    __m128d __r = __a;
    double __v = __b.__d[0]; long long __iv = (long long)__v;
    __r.__d[0] = (double)(__v < (double)__iv ? __iv - 1 : __iv);
    return __r;
}

/* round macros redirect to floor/ceil/trunc at compile time */
#define _mm_round_ps(__a, __r) __extension__({ \
    __m128 __rr; \
    if (((__r) & 3) == 1) __rr = _mm_floor_ps(__a); \
    else if (((__r) & 3) == 2) __rr = _mm_ceil_ps(__a); \
    else __rr = (__a); \
    __rr; })
#define _mm_round_pd(__a, __r) __extension__({ \
    __m128d __rr; \
    if (((__r) & 3) == 1) __rr = _mm_floor_pd(__a); \
    else if (((__r) & 3) == 2) __rr = _mm_ceil_pd(__a); \
    else __rr = (__a); \
    __rr; })

/* --- Compare (64-bit) --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpeq_epi64(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++)
        __r.__i64[__i] = (__a.__i64[__i] == __b.__i64[__i]) ? (long long)-1 : 0;
    return __r;
}

/* --- Pack --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_packus_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        int __va = __a.__i32[__i]; int __vb = __b.__i32[__i];
        __r.__u16[__i] = (unsigned short)(__va < 0 ? 0 : __va > 65535 ? 65535 : __va);
        __r.__u16[__i+4] = (unsigned short)(__vb < 0 ? 0 : __vb > 65535 ? 65535 : __vb);
    }
    return __r;
}

/* --- Test --- */
static __inline__ int __attribute__((__always_inline__))
_mm_testz_si128(__m128i __a, __m128i __b)
{
    for (int __i = 0; __i < 2; __i++)
        if (__a.__i64[__i] & __b.__i64[__i]) return 0;
    return 1;
}

static __inline__ int __attribute__((__always_inline__))
_mm_testc_si128(__m128i __a, __m128i __b)
{
    for (int __i = 0; __i < 2; __i++)
        if ((~__a.__i64[__i]) & __b.__i64[__i]) return 0;
    return 1;
}

static __inline__ int __attribute__((__always_inline__))
_mm_testnzc_si128(__m128i __a, __m128i __b)
{
    return !_mm_testz_si128(__a, __b) && !_mm_testc_si128(__a, __b);
}

/* --- Stream load --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_stream_load_si128(const __m128i *__p) { return *__p; }

/* --- Min horizontal unsigned 16 --- */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_minpos_epu16(__m128i __a)
{
    __m128i __r = _mm_setzero_si128();
    unsigned short __min = __a.__u16[0];
    int __idx = 0;
    for (int __i = 1; __i < 8; __i++) {
        if (__a.__u16[__i] < __min) { __min = __a.__u16[__i]; __idx = __i; }
    }
    __r.__u16[0] = __min;
    __r.__u16[1] = (unsigned short)__idx;
    return __r;
}

/* --- Multiprocessor synchronization --- */
#define _mm_mpsadbw_epu8(__a, __b, __imm) __extension__({ \
    __m128i __result = _mm_setzero_si128(); \
    int __aoff = (((__imm) >> 2) & 1) * 4; \
    int __boff = ((__imm) & 3) * 4; \
    for (int __i = 0; __i < 8; __i++) { \
        int __sum = 0; \
        for (int __j = 0; __j < 4; __j++) { \
            int __diff = (int)(__a).__u8[__aoff + __i + __j] - (int)(__b).__u8[__boff + __j]; \
            __sum += __diff < 0 ? -__diff : __diff; \
        } \
        __result.__u16[__i] = (unsigned short)__sum; \
    } \
    __result; })

#endif /* _SMMINTRIN_H_INCLUDED */
