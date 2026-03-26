/* BCC bundled pmmintrin.h — SSE3 intrinsics (struct-based fallback) */
#ifndef _PMMINTRIN_H_INCLUDED
#define _PMMINTRIN_H_INCLUDED
#include <emmintrin.h>

/* Horizontal add / sub for float */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_hadd_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0] + __a.__f[1];
    __r.__f[1] = __a.__f[2] + __a.__f[3];
    __r.__f[2] = __b.__f[0] + __b.__f[1];
    __r.__f[3] = __b.__f[2] + __b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_hsub_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0] - __a.__f[1];
    __r.__f[1] = __a.__f[2] - __a.__f[3];
    __r.__f[2] = __b.__f[0] - __b.__f[1];
    __r.__f[3] = __b.__f[2] - __b.__f[3];
    return __r;
}

/* Horizontal add / sub for double */
static __inline__ __m128d __attribute__((__always_inline__))
_mm_hadd_pd(__m128d __a, __m128d __b)
{
    __m128d __r;
    __r.__d[0] = __a.__d[0] + __a.__d[1];
    __r.__d[1] = __b.__d[0] + __b.__d[1];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_hsub_pd(__m128d __a, __m128d __b)
{
    __m128d __r;
    __r.__d[0] = __a.__d[0] - __a.__d[1];
    __r.__d[1] = __b.__d[0] - __b.__d[1];
    return __r;
}

/* Duplicate / move */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_movehdup_ps(__m128 __a)
{
    __m128 __r;
    __r.__f[0] = __a.__f[1]; __r.__f[1] = __a.__f[1];
    __r.__f[2] = __a.__f[3]; __r.__f[3] = __a.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_moveldup_ps(__m128 __a)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]; __r.__f[1] = __a.__f[0];
    __r.__f[2] = __a.__f[2]; __r.__f[3] = __a.__f[2];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_movedup_pd(__m128d __a)
{
    __m128d __r; __r.__d[0] = __a.__d[0]; __r.__d[1] = __a.__d[0]; return __r;
}

/* Addsub */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_addsub_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0] - __b.__f[0];
    __r.__f[1] = __a.__f[1] + __b.__f[1];
    __r.__f[2] = __a.__f[2] - __b.__f[2];
    __r.__f[3] = __a.__f[3] + __b.__f[3];
    return __r;
}

static __inline__ __m128d __attribute__((__always_inline__))
_mm_addsub_pd(__m128d __a, __m128d __b)
{
    __m128d __r;
    __r.__d[0] = __a.__d[0] - __b.__d[0];
    __r.__d[1] = __a.__d[1] + __b.__d[1];
    return __r;
}

/* Unaligned load */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_lddqu_si128(const __m128i *__p) { return _mm_loadu_si128(__p); }

/* Denormals-are-zero macros */
#define _MM_DENORMALS_ZERO_MASK  0x0040
#define _MM_DENORMALS_ZERO_ON   0x0040
#define _MM_DENORMALS_ZERO_OFF  0x0000
#define _MM_SET_DENORMALS_ZERO_MODE(x)  ((void)(x))
#define _MM_GET_DENORMALS_ZERO_MODE()   _MM_DENORMALS_ZERO_OFF

#endif /* _PMMINTRIN_H_INCLUDED */
