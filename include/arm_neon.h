/* BCC bundled arm_neon.h — ARM NEON intrinsics (struct-based fallback) */
#ifndef _ARM_NEON_H_INCLUDED
#define _ARM_NEON_H_INCLUDED

#include <stdint.h>

/* ===================================================================
   NEON vector types — struct-based (no vector_size)
   64-bit (D-register) types
   =================================================================== */
typedef struct { int8_t   __v[8]; }  __attribute__((aligned(8)))  int8x8_t;
typedef struct { int16_t  __v[4]; }  __attribute__((aligned(8)))  int16x4_t;
typedef struct { int32_t  __v[2]; }  __attribute__((aligned(8)))  int32x2_t;
typedef struct { int64_t  __v[1]; }  __attribute__((aligned(8)))  int64x1_t;
typedef struct { uint8_t  __v[8]; }  __attribute__((aligned(8)))  uint8x8_t;
typedef struct { uint16_t __v[4]; }  __attribute__((aligned(8)))  uint16x4_t;
typedef struct { uint32_t __v[2]; }  __attribute__((aligned(8)))  uint32x2_t;
typedef struct { uint64_t __v[1]; }  __attribute__((aligned(8)))  uint64x1_t;
typedef struct { float    __v[2]; }  __attribute__((aligned(8)))  float32x2_t;
typedef struct { double   __v[1]; }  __attribute__((aligned(8)))  float64x1_t;

/* 128-bit (Q-register) types */
typedef struct { int8_t   __v[16]; } __attribute__((aligned(16))) int8x16_t;
typedef struct { int16_t  __v[8]; }  __attribute__((aligned(16))) int16x8_t;
typedef struct { int32_t  __v[4]; }  __attribute__((aligned(16))) int32x4_t;
typedef struct { int64_t  __v[2]; }  __attribute__((aligned(16))) int64x2_t;
typedef struct { uint8_t  __v[16]; } __attribute__((aligned(16))) uint8x16_t;
typedef struct { uint16_t __v[8]; }  __attribute__((aligned(16))) uint16x8_t;
typedef struct { uint32_t __v[4]; }  __attribute__((aligned(16))) uint32x4_t;
typedef struct { uint64_t __v[2]; }  __attribute__((aligned(16))) uint64x2_t;
typedef struct { float    __v[4]; }  __attribute__((aligned(16))) float32x4_t;
typedef struct { double   __v[2]; }  __attribute__((aligned(16))) float64x2_t;

/* Paired types (x2) */
typedef struct { int8x8_t   val[2]; } int8x8x2_t;
typedef struct { int16x4_t  val[2]; } int16x4x2_t;
typedef struct { int32x2_t  val[2]; } int32x2x2_t;
typedef struct { uint8x8_t  val[2]; } uint8x8x2_t;
typedef struct { uint16x4_t val[2]; } uint16x4x2_t;
typedef struct { uint32x2_t val[2]; } uint32x2x2_t;
typedef struct { float32x2_t val[2]; } float32x2x2_t;
typedef struct { int8x16_t  val[2]; } int8x16x2_t;
typedef struct { int16x8_t  val[2]; } int16x8x2_t;
typedef struct { int32x4_t  val[2]; } int32x4x2_t;
typedef struct { uint8x16_t val[2]; } uint8x16x2_t;
typedef struct { uint16x8_t val[2]; } uint16x8x2_t;
typedef struct { uint32x4_t val[2]; } uint32x4x2_t;
typedef struct { float32x4_t val[2]; } float32x4x2_t;

/* ===================================================================
   Dup (broadcast scalar to vector)
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vdupq_n_s32(int32_t __v)
{
    int32x4_t __r; for (int __i=0;__i<4;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ uint32x4_t __attribute__((__always_inline__))
vdupq_n_u32(uint32_t __v)
{
    uint32x4_t __r; for (int __i=0;__i<4;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ float32x4_t __attribute__((__always_inline__))
vdupq_n_f32(float __v)
{
    float32x4_t __r; for (int __i=0;__i<4;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ int8x16_t __attribute__((__always_inline__))
vdupq_n_s8(int8_t __v)
{
    int8x16_t __r; for (int __i=0;__i<16;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ uint8x16_t __attribute__((__always_inline__))
vdupq_n_u8(uint8_t __v)
{
    uint8x16_t __r; for (int __i=0;__i<16;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ int16x8_t __attribute__((__always_inline__))
vdupq_n_s16(int16_t __v)
{
    int16x8_t __r; for (int __i=0;__i<8;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ uint16x8_t __attribute__((__always_inline__))
vdupq_n_u16(uint16_t __v)
{
    uint16x8_t __r; for (int __i=0;__i<8;__i++) __r.__v[__i]=__v; return __r;
}
static __inline__ int64x2_t __attribute__((__always_inline__))
vdupq_n_s64(int64_t __v)
{
    int64x2_t __r; __r.__v[0]=__v; __r.__v[1]=__v; return __r;
}
static __inline__ uint64x2_t __attribute__((__always_inline__))
vdupq_n_u64(uint64_t __v)
{
    uint64x2_t __r; __r.__v[0]=__v; __r.__v[1]=__v; return __r;
}

/* 64-bit dup */
static __inline__ int32x2_t __attribute__((__always_inline__))
vdup_n_s32(int32_t __v) { int32x2_t __r; __r.__v[0]=__v; __r.__v[1]=__v; return __r; }
static __inline__ float32x2_t __attribute__((__always_inline__))
vdup_n_f32(float __v) { float32x2_t __r; __r.__v[0]=__v; __r.__v[1]=__v; return __r; }
static __inline__ uint8x8_t __attribute__((__always_inline__))
vdup_n_u8(uint8_t __v) { uint8x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=__v; return __r; }

/* ===================================================================
   Load / Store
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vld1q_s32(const int32_t *__p) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vld1q_u32(const uint32_t *__p) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vld1q_f32(const float *__p) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ int8x16_t __attribute__((__always_inline__))
vld1q_s8(const int8_t *__p) { int8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ uint8x16_t __attribute__((__always_inline__))
vld1q_u8(const uint8_t *__p) { uint8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ int16x8_t __attribute__((__always_inline__))
vld1q_s16(const int16_t *__p) { int16x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=__p[__i]; return __r; }
static __inline__ uint16x8_t __attribute__((__always_inline__))
vld1q_u16(const uint16_t *__p) { uint16x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=__p[__i]; return __r; }

static __inline__ void __attribute__((__always_inline__))
vst1q_s32(int32_t *__p, int32x4_t __v) { for(int __i=0;__i<4;__i++) __p[__i]=__v.__v[__i]; }
static __inline__ void __attribute__((__always_inline__))
vst1q_u32(uint32_t *__p, uint32x4_t __v) { for(int __i=0;__i<4;__i++) __p[__i]=__v.__v[__i]; }
static __inline__ void __attribute__((__always_inline__))
vst1q_f32(float *__p, float32x4_t __v) { for(int __i=0;__i<4;__i++) __p[__i]=__v.__v[__i]; }
static __inline__ void __attribute__((__always_inline__))
vst1q_s8(int8_t *__p, int8x16_t __v) { for(int __i=0;__i<16;__i++) __p[__i]=__v.__v[__i]; }
static __inline__ void __attribute__((__always_inline__))
vst1q_u8(uint8_t *__p, uint8x16_t __v) { for(int __i=0;__i<16;__i++) __p[__i]=__v.__v[__i]; }

/* ===================================================================
   Arithmetic (add, sub, mul)
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vaddq_s32(int32x4_t __a, int32x4_t __b) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vaddq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vaddq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]; return __r; }
static __inline__ int8x16_t __attribute__((__always_inline__))
vaddq_s8(int8x16_t __a, int8x16_t __b) { int8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]; return __r; }
static __inline__ int16x8_t __attribute__((__always_inline__))
vaddq_s16(int16x8_t __a, int16x8_t __b) { int16x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]; return __r; }

static __inline__ int32x4_t __attribute__((__always_inline__))
vsubq_s32(int32x4_t __a, int32x4_t __b) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]-__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vsubq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]-__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vsubq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]-__b.__v[__i]; return __r; }

static __inline__ int32x4_t __attribute__((__always_inline__))
vmulq_s32(int32x4_t __a, int32x4_t __b) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]*__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vmulq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]*__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vmulq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]*__b.__v[__i]; return __r; }

/* Negate / Abs */
static __inline__ int32x4_t __attribute__((__always_inline__))
vnegq_s32(int32x4_t __a) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=-__a.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vnegq_f32(float32x4_t __a) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=-__a.__v[__i]; return __r; }
static __inline__ int32x4_t __attribute__((__always_inline__))
vabsq_s32(int32x4_t __a) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<0?-__a.__v[__i]:__a.__v[__i]; return __r; }

/* ===================================================================
   Logical
   =================================================================== */
static __inline__ uint32x4_t __attribute__((__always_inline__))
vandq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]&__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vorrq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]|__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
veorq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]^__b.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vmvnq_u32(uint32x4_t __a) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=~__a.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vbicq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]&~__b.__v[__i]; return __r; }

static __inline__ uint8x16_t __attribute__((__always_inline__))
vandq_u8(uint8x16_t __a, uint8x16_t __b) { uint8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__a.__v[__i]&__b.__v[__i]; return __r; }
static __inline__ uint8x16_t __attribute__((__always_inline__))
vorrq_u8(uint8x16_t __a, uint8x16_t __b) { uint8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__a.__v[__i]|__b.__v[__i]; return __r; }
static __inline__ uint8x16_t __attribute__((__always_inline__))
veorq_u8(uint8x16_t __a, uint8x16_t __b) { uint8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=__a.__v[__i]^__b.__v[__i]; return __r; }

/* ===================================================================
   Shift
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vshlq_n_s32(int32x4_t __a, int __n) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<<__n; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vshlq_n_u32(uint32x4_t __a, int __n) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<<__n; return __r; }
static __inline__ int32x4_t __attribute__((__always_inline__))
vshrq_n_s32(int32x4_t __a, int __n) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>>__n; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vshrq_n_u32(uint32x4_t __a, int __n) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>>__n; return __r; }

/* ===================================================================
   Compare
   =================================================================== */
static __inline__ uint32x4_t __attribute__((__always_inline__))
vceqq_s32(int32x4_t __a, int32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]==__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vceqq_u32(uint32x4_t __a, uint32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]==__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vceqq_f32(float32x4_t __a, float32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]==__b.__v[__i]?0xFFFFFFFFu:0; return __r; }

static __inline__ uint32x4_t __attribute__((__always_inline__))
vcgtq_s32(int32x4_t __a, int32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vcgtq_f32(float32x4_t __a, float32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vcltq_s32(int32x4_t __a, int32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vcgeq_s32(int32x4_t __a, int32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>=__b.__v[__i]?0xFFFFFFFFu:0; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vcleq_s32(int32x4_t __a, int32x4_t __b) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<=__b.__v[__i]?0xFFFFFFFFu:0; return __r; }

/* ===================================================================
   Min / Max
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vminq_s32(int32x4_t __a, int32x4_t __b) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<__b.__v[__i]?__a.__v[__i]:__b.__v[__i]; return __r; }
static __inline__ int32x4_t __attribute__((__always_inline__))
vmaxq_s32(int32x4_t __a, int32x4_t __b) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>__b.__v[__i]?__a.__v[__i]:__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vminq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]<__b.__v[__i]?__a.__v[__i]:__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vmaxq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]>__b.__v[__i]?__a.__v[__i]:__b.__v[__i]; return __r; }

/* ===================================================================
   Bitwise select
   =================================================================== */
static __inline__ uint32x4_t __attribute__((__always_inline__))
vbslq_u32(uint32x4_t __mask, uint32x4_t __a, uint32x4_t __b)
{
    uint32x4_t __r;
    for (int __i=0; __i<4; __i++)
        __r.__v[__i] = (__mask.__v[__i] & __a.__v[__i]) | (~__mask.__v[__i] & __b.__v[__i]);
    return __r;
}
static __inline__ float32x4_t __attribute__((__always_inline__))
vbslq_f32(uint32x4_t __mask, float32x4_t __a, float32x4_t __b)
{
    float32x4_t __r;
    union { float f; uint32_t u; } __ua, __ub, __ur;
    for (int __i=0; __i<4; __i++) {
        __ua.f = __a.__v[__i]; __ub.f = __b.__v[__i];
        __ur.u = (__mask.__v[__i] & __ua.u) | (~__mask.__v[__i] & __ub.u);
        __r.__v[__i] = __ur.f;
    }
    return __r;
}

/* ===================================================================
   Lane extract / set
   =================================================================== */
#define vgetq_lane_s32(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_u32(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_f32(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_s8(__v, __lane)   ((__v).__v[(__lane)])
#define vgetq_lane_u8(__v, __lane)   ((__v).__v[(__lane)])
#define vgetq_lane_s16(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_u16(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_s64(__v, __lane)  ((__v).__v[(__lane)])
#define vgetq_lane_u64(__v, __lane)  ((__v).__v[(__lane)])

#define vget_lane_s32(__v, __lane)  ((__v).__v[(__lane)])
#define vget_lane_f32(__v, __lane)  ((__v).__v[(__lane)])

#define vsetq_lane_s32(__val, __v, __lane) __extension__({ \
    int32x4_t __r = (__v); __r.__v[(__lane)] = (__val); __r; })
#define vsetq_lane_f32(__val, __v, __lane) __extension__({ \
    float32x4_t __r = (__v); __r.__v[(__lane)] = (__val); __r; })

/* ===================================================================
   Combine / Get low / Get high
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vcombine_s32(int32x2_t __lo, int32x2_t __hi) { int32x4_t __r; __r.__v[0]=__lo.__v[0]; __r.__v[1]=__lo.__v[1]; __r.__v[2]=__hi.__v[0]; __r.__v[3]=__hi.__v[1]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vcombine_f32(float32x2_t __lo, float32x2_t __hi) { float32x4_t __r; __r.__v[0]=__lo.__v[0]; __r.__v[1]=__lo.__v[1]; __r.__v[2]=__hi.__v[0]; __r.__v[3]=__hi.__v[1]; return __r; }

static __inline__ int32x2_t __attribute__((__always_inline__))
vget_low_s32(int32x4_t __a) { int32x2_t __r; __r.__v[0]=__a.__v[0]; __r.__v[1]=__a.__v[1]; return __r; }
static __inline__ int32x2_t __attribute__((__always_inline__))
vget_high_s32(int32x4_t __a) { int32x2_t __r; __r.__v[0]=__a.__v[2]; __r.__v[1]=__a.__v[3]; return __r; }
static __inline__ float32x2_t __attribute__((__always_inline__))
vget_low_f32(float32x4_t __a) { float32x2_t __r; __r.__v[0]=__a.__v[0]; __r.__v[1]=__a.__v[1]; return __r; }
static __inline__ float32x2_t __attribute__((__always_inline__))
vget_high_f32(float32x4_t __a) { float32x2_t __r; __r.__v[0]=__a.__v[2]; __r.__v[1]=__a.__v[3]; return __r; }

/* ===================================================================
   Reciprocal estimate / step
   =================================================================== */
static __inline__ float32x4_t __attribute__((__always_inline__))
vrecpeq_f32(float32x4_t __a) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=1.0f/__a.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vrecpsq_f32(float32x4_t __a, float32x4_t __b) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=2.0f-__a.__v[__i]*__b.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vrsqrteq_f32(float32x4_t __a)
{
    float32x4_t __r;
    for(int __i=0;__i<4;__i++) {
        float __v = __a.__v[__i];
        if (__v <= 0.0f) { __r.__v[__i] = 0.0f; continue; }
        union { float f; unsigned int u; } __u; __u.f = __v;
        __u.u = 0x5f3759dfu - (__u.u >> 1);
        __r.__v[__i] = __u.f;
    }
    return __r;
}

/* ===================================================================
   Multiply-accumulate
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vmlaq_s32(int32x4_t __a, int32x4_t __b, int32x4_t __c) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]*__c.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vmlaq_f32(float32x4_t __a, float32x4_t __b, float32x4_t __c) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]+__b.__v[__i]*__c.__v[__i]; return __r; }
static __inline__ int32x4_t __attribute__((__always_inline__))
vmlsq_s32(int32x4_t __a, int32x4_t __b, int32x4_t __c) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]-__b.__v[__i]*__c.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vmlsq_f32(float32x4_t __a, float32x4_t __b, float32x4_t __c) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=__a.__v[__i]-__b.__v[__i]*__c.__v[__i]; return __r; }

/* ===================================================================
   Conversion
   =================================================================== */
static __inline__ int32x4_t __attribute__((__always_inline__))
vcvtq_s32_f32(float32x4_t __a) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(int32_t)__a.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vcvtq_f32_s32(int32x4_t __a) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(float)__a.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vcvtq_u32_f32(float32x4_t __a) { uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(uint32_t)__a.__v[__i]; return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vcvtq_f32_u32(uint32x4_t __a) { float32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(float)__a.__v[__i]; return __r; }

/* ===================================================================
   Reinterpret casts
   =================================================================== */
static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_s8(int8x16_t __a)
{ uint8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=(uint8_t)__a.__v[__i]; return __r; }
static __inline__ int8x16_t __attribute__((__always_inline__))
vreinterpretq_s8_u8(uint8x16_t __a)
{ int8x16_t __r; for(int __i=0;__i<16;__i++) __r.__v[__i]=(int8_t)__a.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vreinterpretq_u32_s32(int32x4_t __a)
{ uint32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(uint32_t)__a.__v[__i]; return __r; }
static __inline__ int32x4_t __attribute__((__always_inline__))
vreinterpretq_s32_u32(uint32x4_t __a)
{ int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(int32_t)__a.__v[__i]; return __r; }
static __inline__ uint32x4_t __attribute__((__always_inline__))
vreinterpretq_u32_f32(float32x4_t __a)
{ uint32x4_t __r; for(int __i=0;__i<4;__i++) { union{float f;uint32_t u;} __t; __t.f=__a.__v[__i]; __r.__v[__i]=__t.u; } return __r; }
static __inline__ float32x4_t __attribute__((__always_inline__))
vreinterpretq_f32_u32(uint32x4_t __a)
{ float32x4_t __r; for(int __i=0;__i<4;__i++) { union{float f;uint32_t u;} __t; __t.u=__a.__v[__i]; __r.__v[__i]=__t.f; } return __r; }

/* ===================================================================
   Pairwise add
   =================================================================== */
static __inline__ int32x2_t __attribute__((__always_inline__))
vpadd_s32(int32x2_t __a, int32x2_t __b)
{ int32x2_t __r; __r.__v[0]=__a.__v[0]+__a.__v[1]; __r.__v[1]=__b.__v[0]+__b.__v[1]; return __r; }
static __inline__ float32x2_t __attribute__((__always_inline__))
vpadd_f32(float32x2_t __a, float32x2_t __b)
{ float32x2_t __r; __r.__v[0]=__a.__v[0]+__a.__v[1]; __r.__v[1]=__b.__v[0]+__b.__v[1]; return __r; }

/* ===================================================================
   Multiply by lane
   =================================================================== */
#define vmulq_lane_s32(__a, __b, __lane) __extension__({ \
    int32x4_t __r; int32_t __s = (__b).__v[(__lane)]; \
    for(int __i=0;__i<4;__i++) __r.__v[__i]=(__a).__v[__i]*__s; __r; })
#define vmulq_lane_f32(__a, __b, __lane) __extension__({ \
    float32x4_t __r; float __s = (__b).__v[(__lane)]; \
    for(int __i=0;__i<4;__i++) __r.__v[__i]=(__a).__v[__i]*__s; __r; })

/* ===================================================================
   Zip / Unzip
   =================================================================== */
static __inline__ int32x4x2_t __attribute__((__always_inline__))
vzipq_s32(int32x4_t __a, int32x4_t __b)
{
    int32x4x2_t __r;
    __r.val[0].__v[0]=__a.__v[0]; __r.val[0].__v[1]=__b.__v[0];
    __r.val[0].__v[2]=__a.__v[1]; __r.val[0].__v[3]=__b.__v[1];
    __r.val[1].__v[0]=__a.__v[2]; __r.val[1].__v[1]=__b.__v[2];
    __r.val[1].__v[2]=__a.__v[3]; __r.val[1].__v[3]=__b.__v[3];
    return __r;
}

static __inline__ int32x4x2_t __attribute__((__always_inline__))
vuzpq_s32(int32x4_t __a, int32x4_t __b)
{
    int32x4x2_t __r;
    __r.val[0].__v[0]=__a.__v[0]; __r.val[0].__v[1]=__a.__v[2];
    __r.val[0].__v[2]=__b.__v[0]; __r.val[0].__v[3]=__b.__v[2];
    __r.val[1].__v[0]=__a.__v[1]; __r.val[1].__v[1]=__a.__v[3];
    __r.val[1].__v[2]=__b.__v[1]; __r.val[1].__v[3]=__b.__v[3];
    return __r;
}

/* Create / move */
static __inline__ int32x2_t __attribute__((__always_inline__))
vmov_n_s32(int32_t __v) { return vdup_n_s32(__v); }
static __inline__ int32x4_t __attribute__((__always_inline__))
vmovq_n_s32(int32_t __v) { return vdupq_n_s32(__v); }

/* Zero */
static __inline__ int32x4_t __attribute__((__always_inline__))
vdupq_n_s32_zero(void) { return vdupq_n_s32(0); }

/* Count leading zeros (32-bit) */
static __inline__ int32x4_t __attribute__((__always_inline__))
vclzq_s32(int32x4_t __a)
{
    int32x4_t __r;
    for (int __i=0; __i<4; __i++) {
        uint32_t __x = (uint32_t)__a.__v[__i];
        if (__x == 0) { __r.__v[__i] = 32; continue; }
        int __n = 0;
        while (!(__x & 0x80000000u)) { __n++; __x <<= 1; }
        __r.__v[__i] = __n;
    }
    return __r;
}

/* Widen: 16→32, 8→16 */
static __inline__ int32x4_t __attribute__((__always_inline__))
vmovl_s16(int16x4_t __a) { int32x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(int32_t)__a.__v[__i]; return __r; }
static __inline__ int16x8_t __attribute__((__always_inline__))
vmovl_s8(int8x8_t __a) { int16x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=(int16_t)__a.__v[__i]; return __r; }

/* Narrow: 32→16, 16→8 */
static __inline__ int16x4_t __attribute__((__always_inline__))
vmovn_s32(int32x4_t __a) { int16x4_t __r; for(int __i=0;__i<4;__i++) __r.__v[__i]=(int16_t)__a.__v[__i]; return __r; }
static __inline__ int8x8_t __attribute__((__always_inline__))
vmovn_s16(int16x8_t __a) { int8x8_t __r; for(int __i=0;__i<8;__i++) __r.__v[__i]=(int8_t)__a.__v[__i]; return __r; }

/* Reverse elements */
static __inline__ int32x4_t __attribute__((__always_inline__))
vrev64q_s32(int32x4_t __a)
{
    int32x4_t __r;
    __r.__v[0]=__a.__v[1]; __r.__v[1]=__a.__v[0];
    __r.__v[2]=__a.__v[3]; __r.__v[3]=__a.__v[2];
    return __r;
}

/* Table lookup */
static __inline__ uint8x8_t __attribute__((__always_inline__))
vtbl1_u8(uint8x8_t __a, uint8x8_t __idx)
{
    uint8x8_t __r;
    for (int __i=0; __i<8; __i++)
        __r.__v[__i] = __idx.__v[__i] < 8 ? __a.__v[__idx.__v[__i]] : 0;
    return __r;
}

#endif /* _ARM_NEON_H_INCLUDED */
