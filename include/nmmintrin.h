/* BCC bundled nmmintrin.h — SSE4.2 intrinsics (struct-based fallback) */
#ifndef _NMMINTRIN_H_INCLUDED
#define _NMMINTRIN_H_INCLUDED
#include <smmintrin.h>

/* String comparison immediates */
#define _SIDD_UBYTE_OPS          0x00
#define _SIDD_UWORD_OPS          0x01
#define _SIDD_SBYTE_OPS          0x02
#define _SIDD_SWORD_OPS          0x03
#define _SIDD_CMP_EQUAL_ANY      0x00
#define _SIDD_CMP_RANGES         0x04
#define _SIDD_CMP_EQUAL_EACH     0x08
#define _SIDD_CMP_EQUAL_ORDERED  0x0c
#define _SIDD_POSITIVE_POLARITY  0x00
#define _SIDD_NEGATIVE_POLARITY  0x10
#define _SIDD_MASKED_POSITIVE_POLARITY 0x20
#define _SIDD_MASKED_NEGATIVE_POLARITY 0x30
#define _SIDD_LEAST_SIGNIFICANT  0x00
#define _SIDD_MOST_SIGNIFICANT   0x40
#define _SIDD_BIT_MASK           0x00
#define _SIDD_UNIT_MASK          0x40

/* CRC32 (software fallback using polynomial 0x1EDC6F41) */
static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u8(unsigned int __crc, unsigned char __v)
{
    __crc ^= __v;
    for (int __i = 0; __i < 8; __i++)
        __crc = (__crc >> 1) ^ ((__crc & 1) ? 0x82F63B78u : 0u);
    return __crc;
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u16(unsigned int __crc, unsigned short __v)
{
    __crc = _mm_crc32_u8(__crc, (unsigned char)(__v & 0xFF));
    __crc = _mm_crc32_u8(__crc, (unsigned char)((__v >> 8) & 0xFF));
    return __crc;
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u32(unsigned int __crc, unsigned int __v)
{
    __crc = _mm_crc32_u16(__crc, (unsigned short)(__v & 0xFFFF));
    __crc = _mm_crc32_u16(__crc, (unsigned short)((__v >> 16) & 0xFFFF));
    return __crc;
}

static __inline__ unsigned long long __attribute__((__always_inline__))
_mm_crc32_u64(unsigned long long __crc, unsigned long long __v)
{
    unsigned int __lo = _mm_crc32_u32((unsigned int)__crc, (unsigned int)(__v & 0xFFFFFFFFu));
    return (unsigned long long)_mm_crc32_u32(__lo, (unsigned int)((__v >> 32) & 0xFFFFFFFFu));
}

/* Population count */
static __inline__ int __attribute__((__always_inline__))
_mm_popcnt_u32(unsigned int __v)
{
    __v = __v - ((__v >> 1) & 0x55555555u);
    __v = (__v & 0x33333333u) + ((__v >> 2) & 0x33333333u);
    return (int)(((__v + (__v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24);
}

static __inline__ long long __attribute__((__always_inline__))
_mm_popcnt_u64(unsigned long long __v)
{
    return (long long)_mm_popcnt_u32((unsigned int)__v)
         + (long long)_mm_popcnt_u32((unsigned int)(__v >> 32));
}

/* Compare gt 64-bit */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_cmpgt_epi64(__m128i __a, __m128i __b)
{
    __m128i __r;
    for (int __i = 0; __i < 2; __i++)
        __r.__i64[__i] = (__a.__i64[__i] > __b.__i64[__i]) ? (long long)-1 : 0;
    return __r;
}

#endif /* _NMMINTRIN_H_INCLUDED */
