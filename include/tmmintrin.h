/* BCC bundled tmmintrin.h — SSSE3 intrinsics (struct-based fallback) */
#ifndef _TMMINTRIN_H_INCLUDED
#define _TMMINTRIN_H_INCLUDED
#include <pmmintrin.h>

/* Absolute value (integer 8/16/32) */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_abs_epi8(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__i8[__i] = __a.__i8[__i] < 0 ? (signed char)(-__a.__i8[__i]) : __a.__i8[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_abs_epi16(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__i16[__i] = __a.__i16[__i] < 0 ? (short)(-__a.__i16[__i]) : __a.__i16[__i];
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_abs_epi32(__m128i __a)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__i32[__i] = __a.__i32[__i] < 0 ? -__a.__i32[__i] : __a.__i32[__i];
    return __r;
}

/* Horizontal add 16/32 */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_hadd_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i16[0] = (short)(__a.__i16[0] + __a.__i16[1]);
    __r.__i16[1] = (short)(__a.__i16[2] + __a.__i16[3]);
    __r.__i16[2] = (short)(__a.__i16[4] + __a.__i16[5]);
    __r.__i16[3] = (short)(__a.__i16[6] + __a.__i16[7]);
    __r.__i16[4] = (short)(__b.__i16[0] + __b.__i16[1]);
    __r.__i16[5] = (short)(__b.__i16[2] + __b.__i16[3]);
    __r.__i16[6] = (short)(__b.__i16[4] + __b.__i16[5]);
    __r.__i16[7] = (short)(__b.__i16[6] + __b.__i16[7]);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_hadd_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i32[0] = __a.__i32[0] + __a.__i32[1];
    __r.__i32[1] = __a.__i32[2] + __a.__i32[3];
    __r.__i32[2] = __b.__i32[0] + __b.__i32[1];
    __r.__i32[3] = __b.__i32[2] + __b.__i32[3];
    return __r;
}

/* Horizontal add saturate 16 */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_hadds_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        int __va = (int)__a.__i16[__i*2] + (int)__a.__i16[__i*2+1];
        int __vb = (int)__b.__i16[__i*2] + (int)__b.__i16[__i*2+1];
        __r.__i16[__i] = (short)(__va > 32767 ? 32767 : __va < -32768 ? -32768 : __va);
        __r.__i16[__i+4] = (short)(__vb > 32767 ? 32767 : __vb < -32768 ? -32768 : __vb);
    }
    return __r;
}

/* Horizontal sub 16/32 */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_hsub_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i16[0] = (short)(__a.__i16[0] - __a.__i16[1]);
    __r.__i16[1] = (short)(__a.__i16[2] - __a.__i16[3]);
    __r.__i16[2] = (short)(__a.__i16[4] - __a.__i16[5]);
    __r.__i16[3] = (short)(__a.__i16[6] - __a.__i16[7]);
    __r.__i16[4] = (short)(__b.__i16[0] - __b.__i16[1]);
    __r.__i16[5] = (short)(__b.__i16[2] - __b.__i16[3]);
    __r.__i16[6] = (short)(__b.__i16[4] - __b.__i16[5]);
    __r.__i16[7] = (short)(__b.__i16[6] - __b.__i16[7]);
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_hsub_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    __r.__i32[0] = __a.__i32[0] - __a.__i32[1];
    __r.__i32[1] = __a.__i32[2] - __a.__i32[3];
    __r.__i32[2] = __b.__i32[0] - __b.__i32[1];
    __r.__i32[3] = __b.__i32[2] - __b.__i32[3];
    return __r;
}

/* Horizontal sub saturate 16 */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_hsubs_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        int __va = (int)__a.__i16[__i*2] - (int)__a.__i16[__i*2+1];
        int __vb = (int)__b.__i16[__i*2] - (int)__b.__i16[__i*2+1];
        __r.__i16[__i] = (short)(__va > 32767 ? 32767 : __va < -32768 ? -32768 : __va);
        __r.__i16[__i+4] = (short)(__vb > 32767 ? 32767 : __vb < -32768 ? -32768 : __vb);
    }
    return __r;
}

/* Shuffle bytes */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_shuffle_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++) {
        if (__b.__u8[__i] & 0x80)
            __r.__u8[__i] = 0;
        else
            __r.__u8[__i] = __a.__u8[__b.__u8[__i] & 0x0f];
    }
    return __r;
}

/* Sign (negate/zero/keep based on sign of b) */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_sign_epi8(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 16; __i++) {
        if (__b.__i8[__i] < 0) __r.__i8[__i] = (signed char)(-__a.__i8[__i]);
        else if (__b.__i8[__i] == 0) __r.__i8[__i] = 0;
        else __r.__i8[__i] = __a.__i8[__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sign_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        if (__b.__i16[__i] < 0) __r.__i16[__i] = (short)(-__a.__i16[__i]);
        else if (__b.__i16[__i] == 0) __r.__i16[__i] = 0;
        else __r.__i16[__i] = __a.__i16[__i];
    }
    return __r;
}

static __inline__ __m128i __attribute__((__always_inline__))
_mm_sign_epi32(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 4; __i++) {
        if (__b.__i32[__i] < 0) __r.__i32[__i] = -__a.__i32[__i];
        else if (__b.__i32[__i] == 0) __r.__i32[__i] = 0;
        else __r.__i32[__i] = __a.__i32[__i];
    }
    return __r;
}

/* Multiply-add unsigned/signed byte pairs to 16-bit accumulate */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_maddubs_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        int __val = (int)__a.__u8[__i*2] * (int)__b.__i8[__i*2]
                  + (int)__a.__u8[__i*2+1] * (int)__b.__i8[__i*2+1];
        if (__val > 32767) __val = 32767;
        if (__val < -32768) __val = -32768;
        __r.__i16[__i] = (short)__val;
    }
    return __r;
}

/* Multiply high-order 16 bits with rounding and shift */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_mulhrs_epi16(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 8; __i++) {
        int __prod = (int)__a.__i16[__i] * (int)__b.__i16[__i];
        __r.__i16[__i] = (short)((__prod + 0x4000) >> 15);
    }
    return __r;
}

/* Byte align (concatenate and extract) */
#define _mm_alignr_epi8(__a, __b, __n) __extension__({ \
    __m128i __result; \
    unsigned char __tmp[32]; \
    for (int __i = 0; __i < 16; __i++) { __tmp[__i] = (__b).__u8[__i]; __tmp[16+__i] = (__a).__u8[__i]; } \
    if ((__n) >= 32) { for (int __i = 0; __i < 16; __i++) __result.__u8[__i] = 0; } \
    else { for (int __i = 0; __i < 16; __i++) { int __idx = __i + (__n); __result.__u8[__i] = __idx < 32 ? __tmp[__idx] : 0; } } \
    __result; })

#endif /* _TMMINTRIN_H_INCLUDED */
