/* BCC bundled xmmintrin.h — SSE intrinsics (struct-based fallback) */
#ifndef _XMMINTRIN_H_INCLUDED
#define _XMMINTRIN_H_INCLUDED

/* === Core SIMD types as aligned structs === */
typedef struct __m128_struct {
    float __f[4];
} __attribute__((aligned(16))) __m128;

/* Rounding mode constants */
#define _MM_ROUND_NEAREST     0x0000
#define _MM_ROUND_DOWN        0x2000
#define _MM_ROUND_UP          0x4000
#define _MM_ROUND_TOWARD_ZERO 0x6000
#define _MM_ROUND_MASK        0x6000
#define _MM_FLUSH_ZERO_MASK   0x8000
#define _MM_FLUSH_ZERO_ON     0x8000
#define _MM_FLUSH_ZERO_OFF    0x0000

#define _MM_SHUFFLE(z, y, x, w) (((z)<<6)|((y)<<4)|((x)<<2)|(w))

/* === Set intrinsics === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_set_ps(float __e3, float __e2, float __e1, float __e0)
{
    __m128 __r;
    __r.__f[0] = __e0; __r.__f[1] = __e1; __r.__f[2] = __e2; __r.__f[3] = __e3;
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set1_ps(float __w)
{
    __m128 __r;
    __r.__f[0] = __w; __r.__f[1] = __w; __r.__f[2] = __w; __r.__f[3] = __w;
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set_ps1(float __w) { return _mm_set1_ps(__w); }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_setr_ps(float __e0, float __e1, float __e2, float __e3)
{
    return _mm_set_ps(__e3, __e2, __e1, __e0);
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_setzero_ps(void)
{
    __m128 __r;
    __r.__f[0] = 0; __r.__f[1] = 0; __r.__f[2] = 0; __r.__f[3] = 0;
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set_ss(float __w)
{
    __m128 __r;
    __r.__f[0] = __w; __r.__f[1] = 0; __r.__f[2] = 0; __r.__f[3] = 0;
    return __r;
}

/* === Load / store === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_load_ps(const float *__p)
{
    __m128 __r;
    __r.__f[0] = __p[0]; __r.__f[1] = __p[1]; __r.__f[2] = __p[2]; __r.__f[3] = __p[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_loadu_ps(const float *__p) { return _mm_load_ps(__p); }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load_ss(const float *__p)
{
    return _mm_set_ss(*__p);
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load1_ps(const float *__p)
{
    return _mm_set1_ps(*__p);
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load_ps1(const float *__p) { return _mm_load1_ps(__p); }

static __inline__ void __attribute__((__always_inline__))
_mm_store_ps(float *__p, __m128 __a)
{
    __p[0] = __a.__f[0]; __p[1] = __a.__f[1]; __p[2] = __a.__f[2]; __p[3] = __a.__f[3];
}

static __inline__ void __attribute__((__always_inline__))
_mm_storeu_ps(float *__p, __m128 __a) { _mm_store_ps(__p, __a); }

static __inline__ void __attribute__((__always_inline__))
_mm_store_ss(float *__p, __m128 __a) { *__p = __a.__f[0]; }

static __inline__ void __attribute__((__always_inline__))
_mm_store1_ps(float *__p, __m128 __a)
{
    __p[0] = __a.__f[0]; __p[1] = __a.__f[0]; __p[2] = __a.__f[0]; __p[3] = __a.__f[0];
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_ps1(float *__p, __m128 __a) { _mm_store1_ps(__p, __a); }

static __inline__ void __attribute__((__always_inline__))
_mm_storer_ps(float *__p, __m128 __a)
{
    __p[0] = __a.__f[3]; __p[1] = __a.__f[2]; __p[2] = __a.__f[1]; __p[3] = __a.__f[0];
}

static __inline__ void __attribute__((__always_inline__))
_mm_stream_ps(float *__p, __m128 __a) { _mm_store_ps(__p, __a); }

/* === Arithmetic === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_add_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]+__b.__f[0]; __r.__f[1] = __a.__f[1]+__b.__f[1];
    __r.__f[2] = __a.__f[2]+__b.__f[2]; __r.__f[3] = __a.__f[3]+__b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sub_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]-__b.__f[0]; __r.__f[1] = __a.__f[1]-__b.__f[1];
    __r.__f[2] = __a.__f[2]-__b.__f[2]; __r.__f[3] = __a.__f[3]-__b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_mul_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]*__b.__f[0]; __r.__f[1] = __a.__f[1]*__b.__f[1];
    __r.__f[2] = __a.__f[2]*__b.__f[2]; __r.__f[3] = __a.__f[3]*__b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_div_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]/__b.__f[0]; __r.__f[1] = __a.__f[1]/__b.__f[1];
    __r.__f[2] = __a.__f[2]/__b.__f[2]; __r.__f[3] = __a.__f[3]/__b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_add_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a; __r.__f[0] = __a.__f[0]+__b.__f[0]; return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sub_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a; __r.__f[0] = __a.__f[0]-__b.__f[0]; return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_mul_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a; __r.__f[0] = __a.__f[0]*__b.__f[0]; return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_div_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a; __r.__f[0] = __a.__f[0]/__b.__f[0]; return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_min_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__f[__i] = __a.__f[__i] < __b.__f[__i] ? __a.__f[__i] : __b.__f[__i];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_max_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__f[__i] = __a.__f[__i] > __b.__f[__i] ? __a.__f[__i] : __b.__f[__i];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_min_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a;
    __r.__f[0] = __a.__f[0] < __b.__f[0] ? __a.__f[0] : __b.__f[0];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_max_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a;
    __r.__f[0] = __a.__f[0] > __b.__f[0] ? __a.__f[0] : __b.__f[0];
    return __r;
}

/* === Logical (via unsigned int reinterpret) === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_and_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __ua[4], __ub[4];
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __ua[__i].f = __a.__f[__i]; __ub[__i].f = __b.__f[__i];
        __ua[__i].u &= __ub[__i].u;
        __r.__f[__i] = __ua[__i].f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_andnot_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __ua[4], __ub[4];
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __ua[__i].f = __a.__f[__i]; __ub[__i].f = __b.__f[__i];
        __ua[__i].u = (~__ua[__i].u) & __ub[__i].u;
        __r.__f[__i] = __ua[__i].f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_or_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __ua[4], __ub[4];
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __ua[__i].f = __a.__f[__i]; __ub[__i].f = __b.__f[__i];
        __ua[__i].u |= __ub[__i].u;
        __r.__f[__i] = __ua[__i].f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_xor_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __ua[4], __ub[4];
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __ua[__i].f = __a.__f[__i]; __ub[__i].f = __b.__f[__i];
        __ua[__i].u ^= __ub[__i].u;
        __r.__f[__i] = __ua[__i].f;
    }
    return __r;
}

/* === Comparison (return __m128 with all-ones/-zero lanes) === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpeq_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __t;
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __t.u = __a.__f[__i] == __b.__f[__i] ? 0xFFFFFFFFU : 0;
        __r.__f[__i] = __t.f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmplt_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __t;
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __t.u = __a.__f[__i] < __b.__f[__i] ? 0xFFFFFFFFU : 0;
        __r.__f[__i] = __t.f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmple_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __t;
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __t.u = __a.__f[__i] <= __b.__f[__i] ? 0xFFFFFFFFU : 0;
        __r.__f[__i] = __t.f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpgt_ps(__m128 __a, __m128 __b) { return _mm_cmplt_ps(__b, __a); }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpge_ps(__m128 __a, __m128 __b) { return _mm_cmple_ps(__b, __a); }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpneq_ps(__m128 __a, __m128 __b)
{
    union { float f; unsigned int u; } __t;
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        __t.u = __a.__f[__i] != __b.__f[__i] ? 0xFFFFFFFFU : 0;
        __r.__f[__i] = __t.f;
    }
    return __r;
}

/* === Scalar compare intrinsics === */
static __inline__ int __attribute__((__always_inline__))
_mm_comieq_ss(__m128 __a, __m128 __b) { return __a.__f[0] == __b.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comilt_ss(__m128 __a, __m128 __b) { return __a.__f[0] < __b.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comile_ss(__m128 __a, __m128 __b) { return __a.__f[0] <= __b.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comigt_ss(__m128 __a, __m128 __b) { return __a.__f[0] > __b.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comige_ss(__m128 __a, __m128 __b) { return __a.__f[0] >= __b.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_comineq_ss(__m128 __a, __m128 __b) { return __a.__f[0] != __b.__f[0]; }

/* === Conversion === */
static __inline__ int __attribute__((__always_inline__))
_mm_cvtss_si32(__m128 __a) { return (int)__a.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_cvtt_ss2si(__m128 __a) { return (int)__a.__f[0]; }

static __inline__ int __attribute__((__always_inline__))
_mm_cvttss_si32(__m128 __a) { return (int)__a.__f[0]; }

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cvtsi32_ss(__m128 __a, int __b)
{
    __m128 __r = __a; __r.__f[0] = (float)__b; return __r;
}

static __inline__ float __attribute__((__always_inline__))
_mm_cvtss_f32(__m128 __a) { return __a.__f[0]; }

/* === Shuffle / unpack === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_unpacklo_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]; __r.__f[1] = __b.__f[0];
    __r.__f[2] = __a.__f[1]; __r.__f[3] = __b.__f[1];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_unpackhi_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[2]; __r.__f[1] = __b.__f[2];
    __r.__f[2] = __a.__f[3]; __r.__f[3] = __b.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_movelh_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __a.__f[0]; __r.__f[1] = __a.__f[1];
    __r.__f[2] = __b.__f[0]; __r.__f[3] = __b.__f[1];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_movehl_ps(__m128 __a, __m128 __b)
{
    __m128 __r;
    __r.__f[0] = __b.__f[2]; __r.__f[1] = __b.__f[3];
    __r.__f[2] = __a.__f[2]; __r.__f[3] = __a.__f[3];
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_move_ss(__m128 __a, __m128 __b)
{
    __m128 __r = __a; __r.__f[0] = __b.__f[0]; return __r;
}

/* _mm_shuffle_ps as a macro using immediates */
#define _mm_shuffle_ps(__a, __b, __imm8) __extension__({ \
    __m128 __r; \
    __r.__f[0] = (__a).__f[(__imm8) & 3]; \
    __r.__f[1] = (__a).__f[((__imm8)>>2) & 3]; \
    __r.__f[2] = (__b).__f[((__imm8)>>4) & 3]; \
    __r.__f[3] = (__b).__f[((__imm8)>>6) & 3]; \
    __r; \
})

/* === Movemask === */
static __inline__ int __attribute__((__always_inline__))
_mm_movemask_ps(__m128 __a)
{
    union { float f; unsigned int u; } __t;
    int __r = 0;
    for (int __i = 0; __i < 4; __i++) {
        __t.f = __a.__f[__i];
        if (__t.u & 0x80000000U) __r |= (1 << __i);
    }
    return __r;
}

/* === Sqrt === */
static __inline__ __m128 __attribute__((__always_inline__))
_mm_sqrt_ps(__m128 __a)
{
    __m128 __r;
    /* Using Newton-Raphson approximation for software sqrt */
    for (int __i = 0; __i < 4; __i++) {
        float __x = __a.__f[__i];
        if (__x <= 0.0f) { __r.__f[__i] = 0.0f; continue; }
        union { float f; unsigned int u; } __conv;
        __conv.f = __x;
        __conv.u = (0x5F3759DFU - (__conv.u >> 1));
        float __rsqrt = __conv.f;
        __rsqrt *= 1.5f - 0.5f * __x * __rsqrt * __rsqrt;
        __r.__f[__i] = __x * __rsqrt;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rsqrt_ps(__m128 __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++) {
        float __x = __a.__f[__i];
        if (__x <= 0.0f) { __r.__f[__i] = 0.0f; continue; }
        union { float f; unsigned int u; } __conv;
        __conv.f = __x;
        __conv.u = 0x5F3759DFU - (__conv.u >> 1);
        __r.__f[__i] = __conv.f;
    }
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rcp_ps(__m128 __a)
{
    __m128 __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__f[__i] = 1.0f / __a.__f[__i];
    return __r;
}

/* === Prefetch / fence (no-ops) === */
#define _mm_prefetch(__p, __i) ((void)0)
static __inline__ void __attribute__((__always_inline__)) _mm_sfence(void) {}

/* === Transpose macro === */
#define _MM_TRANSPOSE4_PS(__r0, __r1, __r2, __r3) do { \
    __m128 __t0 = _mm_unpacklo_ps(__r0, __r1); \
    __m128 __t1 = _mm_unpackhi_ps(__r0, __r1); \
    __m128 __t2 = _mm_unpacklo_ps(__r2, __r3); \
    __m128 __t3 = _mm_unpackhi_ps(__r2, __r3); \
    __r0 = _mm_movelh_ps(__t0, __t2); \
    __r1 = _mm_movehl_ps(__t2, __t0); \
    __r2 = _mm_movelh_ps(__t1, __t3); \
    __r3 = _mm_movehl_ps(__t3, __t1); \
} while (0)

/* === MXCSR stubs === */
static __inline__ unsigned int __attribute__((__always_inline__))
_mm_getcsr(void) { return 0; }

static __inline__ void __attribute__((__always_inline__))
_mm_setcsr(unsigned int __val) { (void)__val; }

#endif /* _XMMINTRIN_H_INCLUDED */
