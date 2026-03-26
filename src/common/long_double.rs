//! Software implementation of 80-bit extended precision (x87 format) arithmetic.
//!
//! This module provides the [`LongDouble`] type for `long double` support in the
//! C11 type system and compile-time constant folding of long double expressions.
//! It replaces external math libraries (`num`, `rug`) to comply with the
//! zero-dependency mandate.
//!
//! # x87 Extended Precision Format
//!
//! Unlike IEEE 754 binary64 (`f64`) which has an implicit leading 1 bit,
//! the x87 80-bit format stores the integer part **explicitly** in bit 63
//! of the 64-bit significand:
//!
//! - **1** sign bit
//! - **15** exponent bits (bias = 16383)
//! - **64** significand bits (bit 63 = explicit integer bit, bits 62:0 = fraction)
//!
//! Special value encoding:
//! - Zero: exponent = 0, significand = 0
//! - Denormal: exponent = 0, significand ≠ 0, integer bit = 0
//! - Normal: exponent ∈ [1, 32766], integer bit = 1
//! - Infinity: exponent = 32767, significand = `1 << 63` (integer bit set, fraction = 0)
//! - NaN: exponent = 32767, fraction ≠ 0

use std::cmp::Ordering;
use std::fmt;

/// Software implementation of 80-bit extended precision (x87 format).
///
/// Stored internally as decomposed fields: sign, biased exponent, and significand
/// with explicit integer bit. All arithmetic uses round-to-nearest-even semantics.
#[derive(Clone, Copy)]
pub struct LongDouble {
    /// Sign bit: `false` = positive, `true` = negative.
    sign: bool,
    /// Biased exponent (15 bits). Bias = 16383.
    /// Range: 0 (denormal/zero) to 32767 (infinity/NaN).
    exponent: u16,
    /// Significand with explicit integer bit at bit 63.
    /// For normalized numbers bit 63 is always 1.
    significand: u64,
}

// ============================================================================
// Constants
// ============================================================================

impl LongDouble {
    /// Exponent bias for 80-bit extended precision.
    const EXPONENT_BIAS: u16 = 16383;
    /// Maximum biased exponent value (reserved for infinity and NaN).
    const MAX_EXPONENT: u16 = 32767;
    /// Mask for the explicit integer bit (bit 63).
    const INTEGER_BIT: u64 = 1u64 << 63;
    /// Mask for the fraction portion (bits 62:0).
    const FRACTION_MASK: u64 = (1u64 << 63) - 1;

    /// Positive zero (`+0.0`).
    pub const ZERO: LongDouble = LongDouble {
        sign: false,
        exponent: 0,
        significand: 0,
    };

    /// Negative zero (`-0.0`).
    pub const NEG_ZERO: LongDouble = LongDouble {
        sign: true,
        exponent: 0,
        significand: 0,
    };

    /// Positive one (`1.0`).
    pub const ONE: LongDouble = LongDouble {
        sign: false,
        exponent: 16383,
        significand: 1u64 << 63,
    };

    /// Positive infinity (`+∞`).
    pub const INFINITY: LongDouble = LongDouble {
        sign: false,
        exponent: 32767,
        significand: 1u64 << 63,
    };

    /// Negative infinity (`-∞`).
    pub const NEG_INFINITY: LongDouble = LongDouble {
        sign: true,
        exponent: 32767,
        significand: 1u64 << 63,
    };

    /// Quiet NaN (Not a Number).
    pub const NAN: LongDouble = LongDouble {
        sign: false,
        exponent: 32767,
        significand: (1u64 << 63) | (1u64 << 62),
    };
}

// ============================================================================
// Classification and Predicates
// ============================================================================

impl LongDouble {
    /// Returns `true` if the value is positive or negative zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.significand == 0
    }

    /// Returns `true` if the value is positive or negative infinity.
    ///
    /// Infinity is encoded as exponent = 32767, integer bit set, fraction = 0.
    #[inline]
    pub fn is_infinity(&self) -> bool {
        self.exponent == Self::MAX_EXPONENT
            && (self.significand & Self::INTEGER_BIT) != 0
            && (self.significand & Self::FRACTION_MASK) == 0
    }

    /// Returns `true` if the value is NaN (Not a Number).
    ///
    /// NaN is encoded as exponent = 32767 with a non-zero fraction.
    #[inline]
    pub fn is_nan(&self) -> bool {
        if self.exponent != Self::MAX_EXPONENT {
            return false;
        }
        // For x87: NaN has integer bit set and non-zero fraction,
        // or integer bit clear with non-zero significand.
        let fraction = self.significand & Self::FRACTION_MASK;
        if (self.significand & Self::INTEGER_BIT) != 0 {
            fraction != 0
        } else {
            // Pseudo-NaN: integer bit clear, any non-zero significand
            self.significand != 0
        }
    }

    /// Returns `true` if the sign bit is set (value is negative).
    ///
    /// Note: NaN and zero can also have the sign bit set.
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.sign
    }

    /// Returns `true` if the value is a denormalized number.
    ///
    /// Denormals have exponent = 0 and a non-zero significand.
    #[inline]
    pub fn is_denormal(&self) -> bool {
        self.exponent == 0 && self.significand != 0
    }

    /// Returns `true` if this is a normal finite number (not zero, denormal,
    /// infinity, or NaN).
    #[inline]
    #[allow(dead_code)]
    fn is_normal(&self) -> bool {
        self.exponent > 0 && self.exponent < Self::MAX_EXPONENT
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

impl LongDouble {
    /// Returns the unbiased (true) exponent.
    ///
    /// For denormals the true exponent is `1 - BIAS = -16382` (same as biased
    /// exponent 1), matching x87 semantics.
    #[inline]
    fn true_exponent(&self) -> i32 {
        if self.exponent == 0 {
            1 - Self::EXPONENT_BIAS as i32
        } else {
            self.exponent as i32 - Self::EXPONENT_BIAS as i32
        }
    }

    /// Compare absolute magnitude of two `LongDouble` values.
    ///
    /// Returns `true` if `|self| > |other|`.  Used by addition to determine
    /// which operand has the larger magnitude.
    #[inline]
    fn abs_greater_than(&self, other: &LongDouble) -> bool {
        if self.exponent != other.exponent {
            self.exponent > other.exponent
        } else {
            self.significand > other.significand
        }
    }

    /// Normalize a 128-bit significand and round to 64 bits.
    ///
    /// `exp` is the unbiased exponent corresponding to bit 127 of `sig` being
    /// the integer-bit position.  The function shifts `sig` so that its leading
    /// 1 lands at bit 127, extracts the top 64 bits as the result significand,
    /// and uses the bottom 64 bits for round-to-nearest-even.
    fn normalize_round(sign: bool, mut exp: i32, mut sig: u128) -> LongDouble {
        if sig == 0 {
            return if sign { Self::NEG_ZERO } else { Self::ZERO };
        }

        // Shift the leading 1 to bit 127.
        let lz = sig.leading_zeros();
        if lz > 0 {
            sig <<= lz;
            exp -= lz as i32;
        }

        let sig_hi = (sig >> 64) as u64;
        let sig_lo = sig as u64;
        Self::round_and_pack(sign, exp, sig_hi, sig_lo)
    }

    /// Pack a result with rounding.
    ///
    /// `exp` is the unbiased exponent when `sig_hi` has its integer bit at
    /// bit 63.  `sig_lo` carries guard/round/sticky bits for
    /// round-to-nearest-even.
    fn round_and_pack(sign: bool, mut exp: i32, mut sig_hi: u64, sig_lo: u64) -> LongDouble {
        let half: u64 = 1u64 << 63;
        let round_up = if sig_lo > half {
            true
        } else if sig_lo == half {
            // Tie — round to even (round up if LSB of sig_hi is 1).
            (sig_hi & 1) != 0
        } else {
            false
        };

        if round_up {
            sig_hi = sig_hi.wrapping_add(1);
            if sig_hi == 0 {
                // Significand overflowed from 0xFFFF…FFFF to 0.
                sig_hi = Self::INTEGER_BIT;
                exp += 1;
            }
        }

        let biased = exp + Self::EXPONENT_BIAS as i32;

        // Overflow → ±infinity.
        if biased >= Self::MAX_EXPONENT as i32 {
            return if sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // Underflow → denormal or zero.
        if biased <= 0 {
            let shift = (1 - biased) as u32;
            if shift >= 64 {
                return if sign { Self::NEG_ZERO } else { Self::ZERO };
            }
            // Create denormal by shifting significand right.
            sig_hi >>= shift;
            return LongDouble {
                sign,
                exponent: 0,
                significand: sig_hi,
            };
        }

        LongDouble {
            sign,
            exponent: biased as u16,
            significand: sig_hi,
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl LongDouble {
    /// Add two `LongDouble` values: `self + other`.
    ///
    /// Handles all IEEE special cases (NaN propagation, infinity arithmetic,
    /// signed zeros) and uses round-to-nearest-even.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: LongDouble) -> LongDouble {
        // NaN propagation.
        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        // Infinity cases.
        if self.is_infinity() {
            if other.is_infinity() {
                return if self.sign == other.sign {
                    self
                } else {
                    Self::NAN
                };
            }
            return self;
        }
        if other.is_infinity() {
            return other;
        }

        // Zero cases.
        if self.is_zero() {
            if other.is_zero() {
                // +0 + −0 = +0 in round-to-nearest-even.
                return if self.sign && other.sign {
                    Self::NEG_ZERO
                } else {
                    Self::ZERO
                };
            }
            return other;
        }
        if other.is_zero() {
            return self;
        }

        // Order operands so that `a` has the larger (or equal) magnitude.
        let (a, b) = if self.abs_greater_than(&other) {
            (self, other)
        } else {
            (other, self)
        };

        let a_exp = a.true_exponent();
        let b_exp = b.true_exponent();
        let exp_diff = (a_exp - b_exp) as u32;

        // Widen significands to 128 bits, placing the integer bit at bit 127
        // so that the low 64 bits capture rounding information after alignment.
        let a_wide: u128 = (a.significand as u128) << 64;
        let mut b_wide: u128 = (b.significand as u128) << 64;

        // Align `b` to `a` by shifting right.
        if exp_diff > 0 {
            if exp_diff < 128 {
                let sticky = (b_wide & ((1u128 << exp_diff) - 1)) != 0;
                b_wide >>= exp_diff;
                if sticky {
                    b_wide |= 1;
                }
            } else {
                b_wide = u128::from(b_wide != 0); // preserve sticky
            }
        }

        let result_exp = a_exp; // base exponent for the result

        if a.sign == b.sign {
            // Effective addition — significands may carry.
            let (sum, carry) = a_wide.overflowing_add(b_wide);
            if carry {
                // The 129-bit result has its integer bit at position 128.
                // Shift right by 1 and bump the exponent.
                let sticky = (sum & 1) != 0;
                let shifted = (1u128 << 127) | (sum >> 1);
                let shifted = if sticky { shifted | 1 } else { shifted };
                return Self::normalize_round(a.sign, result_exp + 1, shifted);
            }
            Self::normalize_round(a.sign, result_exp, sum)
        } else {
            // Effective subtraction — `a` ≥ `b` in magnitude so no underflow.
            let diff = a_wide - b_wide;
            if diff == 0 {
                return Self::ZERO; // +0 in round-to-nearest-even
            }
            Self::normalize_round(a.sign, result_exp, diff)
        }
    }

    /// Subtract: `self − other`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: LongDouble) -> LongDouble {
        self.add(other.neg())
    }

    /// Multiply two `LongDouble` values: `self × other`.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: LongDouble) -> LongDouble {
        let result_sign = self.sign != other.sign;

        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        if self.is_infinity() {
            if other.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }
        if other.is_infinity() {
            if self.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        if self.is_zero() || other.is_zero() {
            return if result_sign {
                Self::NEG_ZERO
            } else {
                Self::ZERO
            };
        }

        // Result unbiased exponent.
        let exp = self.true_exponent() + other.true_exponent();

        // 64 × 64 → 128 bit product.
        let product = (self.significand as u128) * (other.significand as u128);

        // The product of two significands (each with integer bit at position 63)
        // yields a 128-bit value whose "integer bit squared" position is 126.
        // `normalize_round` interprets bit 127 as the integer-bit position at
        // exponent `exp_param`, so we pass `exp + 1` to account for the
        // one-bit offset:  product / 2^126 × 2^exp = product / 2^127 × 2^(exp+1).
        Self::normalize_round(result_sign, exp + 1, product)
    }

    /// Divide: `self / other`.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: LongDouble) -> LongDouble {
        let result_sign = self.sign != other.sign;

        if self.is_nan() {
            return Self::NAN;
        }
        if other.is_nan() {
            return Self::NAN;
        }

        // inf / inf → NaN; inf / finite → inf.
        if self.is_infinity() {
            if other.is_infinity() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // finite / 0 → inf (or NaN for 0/0).
        if other.is_zero() {
            if self.is_zero() {
                return Self::NAN;
            }
            return if result_sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // 0 / finite → ±0.
        if self.is_zero() {
            return if result_sign {
                Self::NEG_ZERO
            } else {
                Self::ZERO
            };
        }

        // Both operands are finite and non-zero.
        let exp = self.true_exponent() - other.true_exponent();

        // Divide significands using 128-bit arithmetic.
        // dividend = self.sig << 64  (gives 128 bits of precision in quotient)
        let dividend = (self.significand as u128) << 64;
        let divisor = other.significand as u128;

        let quotient = dividend / divisor;
        let remainder = dividend % divisor;

        // The mathematical value is:
        //   (self.sig / other.sig) × 2^exp
        //   = (quotient / 2^64) × 2^exp
        //   = quotient × 2^(exp − 64)
        //
        // `normalize_round` interprets sig / 2^127 × 2^exp_param, so:
        //   quotient × 2^(exp − 64) = quotient / 2^127 × 2^(exp + 63)
        let mut sig = quotient;
        if remainder != 0 {
            sig |= 1; // sticky bit for rounding
        }

        Self::normalize_round(result_sign, exp + 63, sig)
    }

    /// Negate: flip the sign bit.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> LongDouble {
        LongDouble {
            sign: !self.sign,
            exponent: self.exponent,
            significand: self.significand,
        }
    }
}

// ============================================================================
// Comparison
// ============================================================================

impl PartialEq for LongDouble {
    /// IEEE 754 equality: NaN ≠ everything (including itself), −0 == +0.
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        // −0 == +0
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.sign == other.sign
            && self.exponent == other.exponent
            && self.significand == other.significand
    }
}

impl PartialOrd for LongDouble {
    /// IEEE 754 ordering: NaN is unordered, −0 == +0.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        // Both zeros are equal regardless of sign.
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }

        // Handle sign differences.
        match (self.sign, other.sign) {
            (false, true) => {
                // self positive, other negative → self > other
                // (both-zero case already handled above)
                return Some(Ordering::Greater);
            }
            (true, false) => {
                return Some(Ordering::Less);
            }
            _ => {}
        }

        // Same sign — compare magnitudes.
        let mag_cmp = if self.exponent != other.exponent {
            self.exponent.cmp(&other.exponent)
        } else {
            self.significand.cmp(&other.significand)
        };

        // For negative numbers the larger magnitude is the *smaller* value.
        Some(if self.sign {
            mag_cmp.reverse()
        } else {
            mag_cmp
        })
    }
}

impl LongDouble {
    /// Total ordering that is reflexive, antisymmetric, and transitive even
    /// for NaN values.
    ///
    /// Order: −NaN < −∞ < −normals < −denormals < −0 < +0 < +denormals <
    ///        +normals < +∞ < +NaN.
    pub fn total_cmp(&self, other: &LongDouble) -> Ordering {
        // Convert to a sortable integer representation.
        //
        // Order: −NaN < −∞ < −finite < −0 < +0 < +finite < +∞ < +NaN
        //
        // For positive values the magnitude ((exponent << 64) | significand)
        // already sorts correctly.  For negative values we invert: larger
        // magnitudes map to *smaller* sort keys.  We use `−mag − 1` so that
        // −0 (mag = 0 → key = −1) sorts strictly below +0 (mag = 0 → key = 0).
        fn sort_key(v: &LongDouble) -> i128 {
            let mag = ((v.exponent as u128) << 64) | (v.significand as u128);
            if v.sign {
                -(mag as i128) - 1
            } else {
                mag as i128
            }
        }
        sort_key(self).cmp(&sort_key(other))
    }
}

// ============================================================================
// Conversion — f64
// ============================================================================

impl LongDouble {
    /// Create a `LongDouble` from an `f64` value.
    ///
    /// The conversion is exact for all finite `f64` values because the 80-bit
    /// format has strictly more precision and exponent range than binary64.
    pub fn from_f64(val: f64) -> LongDouble {
        let bits = val.to_bits();
        let sign = (bits >> 63) != 0;
        let exp_bits = ((bits >> 52) & 0x7FF) as u16;
        let fraction = bits & ((1u64 << 52) - 1);

        // ±0
        if exp_bits == 0 && fraction == 0 {
            return LongDouble {
                sign,
                exponent: 0,
                significand: 0,
            };
        }

        // Denormal f64 → normalized long double.
        if exp_bits == 0 {
            // value = ±fraction × 2^(−1074)
            let lz = fraction.leading_zeros(); // ≥ 12 since fraction fits in 52 bits
            let significand = fraction << lz; // leading 1 now at bit 63
                                              // Derive biased exponent:
                                              //   value = significand × 2^(−1074 − lz)
                                              //   LD form: 2^(biased − 16383) × significand × 2^(−63)
                                              //   ⇒ biased = 16446 − 1074 − lz = 15372 − lz
            let biased = 15372i32 - lz as i32;
            if biased < 1 {
                // Extremely small — still denormal in long double format.
                // Shouldn't happen for valid f64 denormals but handle gracefully.
                let shift = (1 - biased) as u32;
                let sig = if shift < 64 { significand >> shift } else { 0 };
                return LongDouble {
                    sign,
                    exponent: 0,
                    significand: sig,
                };
            }
            return LongDouble {
                sign,
                exponent: biased as u16,
                significand,
            };
        }

        // ±infinity
        if exp_bits == 0x7FF && fraction == 0 {
            return if sign {
                Self::NEG_INFINITY
            } else {
                Self::INFINITY
            };
        }

        // NaN — preserve quiet bit in the long-double significand.
        if exp_bits == 0x7FF {
            let significand = Self::INTEGER_BIT | (fraction << 11);
            return LongDouble {
                sign,
                exponent: Self::MAX_EXPONENT,
                significand,
            };
        }

        // Normal f64.
        // f64 biased exponent → LD biased exponent:
        //   unbiased = exp_bits − 1023
        //   LD biased = unbiased + 16383 = exp_bits + 15360
        let exponent = exp_bits + 15360;
        // Significand: set explicit integer bit and extend 52 → 63 fraction bits.
        let significand = Self::INTEGER_BIT | (fraction << 11);

        LongDouble {
            sign,
            exponent,
            significand,
        }
    }

    /// Convert to `f64`, rounding to nearest even.
    ///
    /// Precision loss is expected (64-bit significand → 52-bit fraction).
    /// Overflow saturates to `±infinity`; underflow flushes toward `±0`.
    pub fn to_f64(&self) -> f64 {
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_infinity() {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if self.is_zero() {
            return if self.sign { -0.0_f64 } else { 0.0_f64 };
        }

        let true_exp = self.true_exponent();
        let f64_biased = true_exp + 1023;

        // Overflow → ±infinity.
        if f64_biased >= 2047 {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        // Underflow → f64 denormal or zero.
        if f64_biased <= 0 {
            // Number of positions to shift to form a denormal.
            let shift = (1 - f64_biased) as u32;
            if shift > 63 {
                return if self.sign { -0.0_f64 } else { 0.0_f64 };
            }
            // Shift significand right to create a denormal fraction.
            //   LD value = sig × 2^(true_exp − 63)
            //   f64 denormal value = frac × 2^(−1074)
            //   frac = sig × 2^(true_exp − 63 + 1074) = sig × 2^(true_exp + 1011)
            //   Since true_exp + 1011 < 0 for underflow: frac = sig >> (−true_exp − 1011)
            let sa = (0i32 - true_exp - 1011) as u32;
            if sa >= 64 {
                return if self.sign { -0.0_f64 } else { 0.0_f64 };
            }
            let frac = (self.significand >> sa) & ((1u64 << 52) - 1);
            let bits = ((self.sign as u64) << 63) | frac;
            return f64::from_bits(bits);
        }

        // Normal f64.  Extract 52 fraction bits from 63 fraction bits.
        let fraction_full = self.significand & Self::FRACTION_MASK; // 63 fraction bits
        let frac52 = fraction_full >> 11;
        let round_bits = fraction_full & 0x7FF; // bottom 11 bits
        let mut fraction = frac52;

        // Round to nearest even.
        let tie: u64 = 1u64 << 10; // 0x400
        if round_bits > tie || (round_bits == tie && (fraction & 1) != 0) {
            fraction += 1;
            if fraction >= (1u64 << 52) {
                // Carry into exponent field.
                fraction = 0;
                let new_biased = f64_biased + 1;
                if new_biased >= 2047 {
                    return if self.sign {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                }
                let bits = ((self.sign as u64) << 63) | ((new_biased as u64) << 52) | fraction;
                return f64::from_bits(bits);
            }
        }

        let bits = ((self.sign as u64) << 63) | ((f64_biased as u64) << 52) | fraction;
        f64::from_bits(bits)
    }
}

// ============================================================================
// Conversion — Integers
// ============================================================================

impl LongDouble {
    /// Create a `LongDouble` from an `i64`.
    pub fn from_i64(val: i64) -> LongDouble {
        if val == 0 {
            return Self::ZERO;
        }
        let sign = val < 0;
        // Handle i64::MIN carefully (its absolute value doesn't fit in i64).
        let abs_val: u64 = if val == i64::MIN {
            (i64::MAX as u64) + 1
        } else if val < 0 {
            (-val) as u64
        } else {
            val as u64
        };
        let mut result = Self::from_u64(abs_val);
        result.sign = sign;
        result
    }

    /// Create a `LongDouble` from a `u64`.
    pub fn from_u64(val: u64) -> LongDouble {
        if val == 0 {
            return Self::ZERO;
        }
        let lz = val.leading_zeros();
        let significand = val << lz; // leading 1 at bit 63
                                     // Unbiased exponent: the leading 1 was at bit position (63 − lz).
                                     // That means the value is significand × 2^(−lz) × ... let's derive:
                                     //   val = significand >> lz   (undo the shift we applied)
                                     //   LD value = 2^(biased − 16383) × significand / 2^63
                                     //            = significand × 2^(biased − 16446)
                                     //   We need significand × 2^(biased − 16446) = val = significand >> lz
                                     //                                              = significand × 2^(−lz)
                                     //   ⇒ biased − 16446 = −lz  ⇒  biased = 16446 − lz
        let biased = 16446u32 - lz;
        LongDouble {
            sign: false,
            exponent: biased as u16,
            significand,
        }
    }

    /// Convert to `i64`, truncating toward zero.
    ///
    /// Values outside the representable range saturate to `i64::MIN` / `i64::MAX`.
    /// NaN returns 0.
    pub fn to_i64(&self) -> i64 {
        if self.is_nan() || self.is_zero() {
            return 0;
        }
        if self.is_infinity() {
            return if self.sign { i64::MIN } else { i64::MAX };
        }

        let true_exp = self.true_exponent();
        if true_exp < 0 {
            return 0; // |value| < 1
        }

        // bit_pos = true_exp is the position of the integer bit (0-indexed).
        // We need the integer part: significand >> (63 − true_exp).
        let shift = 63 - true_exp;
        if shift < 0 {
            // Value is larger than 2^63 — overflow.
            if self.sign {
                // Check for exactly −2^63 = i64::MIN.
                if true_exp == 63 && self.significand == Self::INTEGER_BIT {
                    return i64::MIN;
                }
                return i64::MIN;
            }
            return i64::MAX;
        }

        let magnitude = self.significand >> (shift as u32);

        if self.sign {
            let max_neg = (i64::MAX as u64) + 1; // 2^63
            if magnitude > max_neg {
                i64::MIN // saturate
            } else {
                // Safe for all magnitude ∈ [0, 2^63]:
                // when magnitude == 2^63, magnitude as i64 is i64::MIN,
                // and 0i64.wrapping_sub(i64::MIN) == i64::MIN (correct).
                0i64.wrapping_sub(magnitude as i64)
            }
        } else if magnitude > i64::MAX as u64 {
            i64::MAX
        } else {
            magnitude as i64
        }
    }

    /// Convert to `u64`, truncating toward zero.
    ///
    /// Negative values and NaN return 0. Overflow saturates to `u64::MAX`.
    pub fn to_u64(&self) -> u64 {
        if self.is_nan() || self.is_zero() {
            return 0;
        }
        if self.sign {
            return 0; // negative → 0 for unsigned
        }
        if self.is_infinity() {
            return u64::MAX;
        }

        let true_exp = self.true_exponent();
        if true_exp < 0 {
            return 0; // |value| < 1
        }

        let shift = 63 - true_exp;
        if shift < 0 {
            // Value ≥ 2^64 — overflow to u64::MAX.
            return u64::MAX;
        }

        self.significand >> (shift as u32)
    }
}

// ============================================================================
// Raw Byte Serialization (x87 10-byte format, little-endian)
// ============================================================================

impl LongDouble {
    /// Serialize to the 10-byte little-endian x87 memory format.
    ///
    /// Layout:
    /// - Bytes 0–7: significand (little-endian `u64`)
    /// - Bytes 8–9: `sign_bit | exponent` (little-endian `u16`, sign is MSB)
    ///
    /// Used for emitting `long double` constants into `.rodata`.
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut out = [0u8; 10];
        let sig_bytes = self.significand.to_le_bytes();
        out[..8].copy_from_slice(&sig_bytes);
        let exp_sign: u16 = self.exponent | ((self.sign as u16) << 15);
        let es_bytes = exp_sign.to_le_bytes();
        out[8] = es_bytes[0];
        out[9] = es_bytes[1];
        out
    }

    /// Deserialize from the 10-byte little-endian x87 memory format.
    pub fn from_bytes(bytes: &[u8; 10]) -> LongDouble {
        let mut sig_buf = [0u8; 8];
        sig_buf.copy_from_slice(&bytes[..8]);
        let significand = u64::from_le_bytes(sig_buf);
        let mut es_buf = [0u8; 2];
        es_buf.copy_from_slice(&bytes[8..10]);
        let exp_sign = u16::from_le_bytes(es_buf);
        let sign = (exp_sign >> 15) != 0;
        let exponent = exp_sign & 0x7FFF;
        LongDouble {
            sign,
            exponent,
            significand,
        }
    }
}

// ============================================================================
// Display / Debug
// ============================================================================

impl fmt::Display for LongDouble {
    /// Displays the value by converting to `f64` first (acceptable precision
    /// loss for diagnostic / display purposes).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            return write!(f, "NaN");
        }
        if self.is_infinity() {
            return if self.sign {
                write!(f, "-inf")
            } else {
                write!(f, "inf")
            };
        }
        // Delegate to f64 formatting for finite values.
        let val = self.to_f64();
        fmt::Display::fmt(&val, f)
    }
}

impl fmt::Debug for LongDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LongDouble {{ sign: {}, exp: {}, sig: {:#018x} }}",
            self.sign, self.exponent, self.significand
        )
    }
}

// ============================================================================
// Standard trait impls required by the rest of the compiler
// ============================================================================

impl Default for LongDouble {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// std::ops trait implementations (delegates to named methods)
// ============================================================================

impl std::ops::Add for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn add(self, rhs: LongDouble) -> LongDouble {
        LongDouble::add(self, rhs)
    }
}

impl std::ops::Sub for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn sub(self, rhs: LongDouble) -> LongDouble {
        LongDouble::sub(self, rhs)
    }
}

impl std::ops::Mul for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn mul(self, rhs: LongDouble) -> LongDouble {
        LongDouble::mul(self, rhs)
    }
}

impl std::ops::Div for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn div(self, rhs: LongDouble) -> LongDouble {
        LongDouble::div(self, rhs)
    }
}

impl std::ops::Neg for LongDouble {
    type Output = LongDouble;
    #[inline]
    fn neg(self) -> LongDouble {
        LongDouble::neg(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // Constants and Classification
    // ====================================================================

    #[test]
    fn test_zero_constant_is_zero() {
        assert!(LongDouble::ZERO.is_zero());
        assert!(!LongDouble::ZERO.is_negative());
    }

    #[test]
    fn test_neg_zero_constant_is_zero_and_negative() {
        assert!(LongDouble::NEG_ZERO.is_zero());
        assert!(LongDouble::NEG_ZERO.is_negative());
    }

    #[test]
    fn test_one_constant_is_finite_and_positive() {
        let one = LongDouble::ONE;
        assert!(!one.is_zero());
        assert!(!one.is_infinity());
        assert!(!one.is_nan());
        assert!(!one.is_negative());
        assert!(!one.is_denormal());
    }

    #[test]
    fn test_infinity_constant_classification() {
        assert!(LongDouble::INFINITY.is_infinity());
        assert!(!LongDouble::INFINITY.is_nan());
        assert!(!LongDouble::INFINITY.is_zero());
        assert!(!LongDouble::INFINITY.is_negative());
    }

    #[test]
    fn test_neg_infinity_constant_classification() {
        assert!(LongDouble::NEG_INFINITY.is_infinity());
        assert!(LongDouble::NEG_INFINITY.is_negative());
        assert!(!LongDouble::NEG_INFINITY.is_nan());
    }

    #[test]
    fn test_nan_constant_classification() {
        assert!(LongDouble::NAN.is_nan());
        assert!(!LongDouble::NAN.is_infinity());
        assert!(!LongDouble::NAN.is_zero());
    }

    #[test]
    fn test_exponent_bias_is_16383() {
        assert_eq!(LongDouble::EXPONENT_BIAS, 16383);
    }

    #[test]
    fn test_default_is_zero() {
        let d: LongDouble = Default::default();
        assert!(d.is_zero());
        assert!(!d.is_negative());
    }

    // ====================================================================
    // f64 Conversion — from_f64
    // ====================================================================

    #[test]
    fn test_from_f64_positive_zero() {
        let ld = LongDouble::from_f64(0.0);
        assert!(ld.is_zero());
        assert!(!ld.is_negative());
    }

    #[test]
    fn test_from_f64_negative_zero() {
        let ld = LongDouble::from_f64(-0.0);
        assert!(ld.is_zero());
        assert!(ld.is_negative());
    }

    #[test]
    fn test_from_f64_positive_infinity() {
        let ld = LongDouble::from_f64(f64::INFINITY);
        assert!(ld.is_infinity());
        assert!(!ld.is_negative());
    }

    #[test]
    fn test_from_f64_negative_infinity() {
        let ld = LongDouble::from_f64(f64::NEG_INFINITY);
        assert!(ld.is_infinity());
        assert!(ld.is_negative());
    }

    #[test]
    fn test_from_f64_nan() {
        let ld = LongDouble::from_f64(f64::NAN);
        assert!(ld.is_nan());
    }

    #[test]
    fn test_from_f64_one() {
        let ld = LongDouble::from_f64(1.0);
        // Should produce exponent = 16383 (bias + 0) and integer bit set
        assert_eq!(ld.to_f64(), 1.0);
    }

    #[test]
    fn test_from_f64_negative_one() {
        let ld = LongDouble::from_f64(-1.0);
        assert!(ld.is_negative());
        assert_eq!(ld.to_f64(), -1.0);
    }

    #[test]
    fn test_from_f64_small_positive() {
        let ld = LongDouble::from_f64(0.5);
        assert_eq!(ld.to_f64(), 0.5);
    }

    #[test]
    fn test_from_f64_large_value() {
        let ld = LongDouble::from_f64(1.0e18);
        assert_eq!(ld.to_f64(), 1.0e18);
    }

    #[test]
    fn test_from_f64_subnormal() {
        // f64 minimum positive subnormal: 5e-324
        let val = f64::from_bits(1); // smallest positive subnormal
        let ld = LongDouble::from_f64(val);
        assert!(!ld.is_zero());
        // The round-trip should recover the same f64 (exact conversion)
        assert_eq!(ld.to_f64(), val);
    }

    // ====================================================================
    // f64 Conversion — to_f64
    // ====================================================================

    #[test]
    fn test_to_f64_zero() {
        assert_eq!(LongDouble::ZERO.to_f64(), 0.0);
        // Check negative zero using bit pattern
        let neg_z = LongDouble::NEG_ZERO.to_f64();
        assert_eq!(neg_z.to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn test_to_f64_infinity() {
        assert_eq!(LongDouble::INFINITY.to_f64(), f64::INFINITY);
        assert_eq!(LongDouble::NEG_INFINITY.to_f64(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_to_f64_nan() {
        assert!(LongDouble::NAN.to_f64().is_nan());
    }

    #[test]
    fn test_f64_roundtrip_various_values() {
        let values = [
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            100.0,
            -100.0,
            1.23456789,
            -9.876543e10,
            1.0e-100,
            1.0e100,
            core::f64::consts::PI,
            core::f64::consts::E,
        ];
        for &v in &values {
            let ld = LongDouble::from_f64(v);
            let back = ld.to_f64();
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "Round-trip failed for f64 value {}",
                v
            );
        }
    }

    // ====================================================================
    // Integer Conversion — from_u64 / from_i64 / to_u64 / to_i64
    // ====================================================================

    #[test]
    fn test_from_u64_zero() {
        let ld = LongDouble::from_u64(0);
        assert!(ld.is_zero());
    }

    #[test]
    fn test_from_u64_one() {
        let ld = LongDouble::from_u64(1);
        assert_eq!(ld.to_f64(), 1.0);
    }

    #[test]
    fn test_from_u64_large() {
        let ld = LongDouble::from_u64(1_000_000);
        assert_eq!(ld.to_f64(), 1_000_000.0);
    }

    #[test]
    fn test_from_u64_max() {
        let ld = LongDouble::from_u64(u64::MAX);
        assert!(!ld.is_zero());
        assert!(!ld.is_infinity());
        // u64::MAX = 2^64 - 1, to_u64 should recover it
        assert_eq!(ld.to_u64(), u64::MAX);
    }

    #[test]
    fn test_from_i64_zero() {
        let ld = LongDouble::from_i64(0);
        assert!(ld.is_zero());
    }

    #[test]
    fn test_from_i64_positive() {
        let ld = LongDouble::from_i64(42);
        assert_eq!(ld.to_i64(), 42);
    }

    #[test]
    fn test_from_i64_negative() {
        let ld = LongDouble::from_i64(-42);
        assert!(ld.is_negative());
        assert_eq!(ld.to_i64(), -42);
    }

    #[test]
    fn test_from_i64_min() {
        let ld = LongDouble::from_i64(i64::MIN);
        assert!(ld.is_negative());
        assert_eq!(ld.to_i64(), i64::MIN);
    }

    #[test]
    fn test_from_i64_max() {
        let ld = LongDouble::from_i64(i64::MAX);
        assert_eq!(ld.to_i64(), i64::MAX);
    }

    #[test]
    fn test_to_i64_nan_returns_zero() {
        assert_eq!(LongDouble::NAN.to_i64(), 0);
    }

    #[test]
    fn test_to_i64_infinity_saturates() {
        assert_eq!(LongDouble::INFINITY.to_i64(), i64::MAX);
        assert_eq!(LongDouble::NEG_INFINITY.to_i64(), i64::MIN);
    }

    #[test]
    fn test_to_u64_negative_returns_zero() {
        let ld = LongDouble::from_i64(-1);
        assert_eq!(ld.to_u64(), 0);
    }

    #[test]
    fn test_to_u64_nan_returns_zero() {
        assert_eq!(LongDouble::NAN.to_u64(), 0);
    }

    #[test]
    fn test_to_u64_infinity_saturates() {
        assert_eq!(LongDouble::INFINITY.to_u64(), u64::MAX);
    }

    #[test]
    fn test_to_i64_fractional_truncates() {
        let ld = LongDouble::from_f64(3.9);
        assert_eq!(ld.to_i64(), 3);
    }

    // ====================================================================
    // Basic Arithmetic — Addition
    // ====================================================================

    #[test]
    fn test_add_positive_values() {
        let a = LongDouble::from_f64(1.5);
        let b = LongDouble::from_f64(2.5);
        let result = a.add(b);
        assert_eq!(result.to_f64(), 4.0);
    }

    #[test]
    fn test_add_negative_values() {
        let a = LongDouble::from_f64(-1.0);
        let b = LongDouble::from_f64(-2.0);
        let result = a.add(b);
        assert_eq!(result.to_f64(), -3.0);
    }

    #[test]
    fn test_add_opposite_signs_positive_result() {
        let a = LongDouble::from_f64(5.0);
        let b = LongDouble::from_f64(-3.0);
        let result = a.add(b);
        assert_eq!(result.to_f64(), 2.0);
    }

    #[test]
    fn test_add_opposite_signs_negative_result() {
        let a = LongDouble::from_f64(3.0);
        let b = LongDouble::from_f64(-5.0);
        let result = a.add(b);
        assert_eq!(result.to_f64(), -2.0);
    }

    #[test]
    fn test_add_cancel_to_zero() {
        let a = LongDouble::from_f64(3.0);
        let b = LongDouble::from_f64(-3.0);
        let result = a.add(b);
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_zero_identity() {
        let a = LongDouble::from_f64(42.0);
        let result = a.add(LongDouble::ZERO);
        assert_eq!(result.to_f64(), 42.0);
    }

    #[test]
    fn test_add_nan_propagation() {
        let a = LongDouble::from_f64(1.0);
        let result = a.add(LongDouble::NAN);
        assert!(result.is_nan());
        let result2 = LongDouble::NAN.add(a);
        assert!(result2.is_nan());
    }

    #[test]
    fn test_add_infinity_plus_finite() {
        let result = LongDouble::INFINITY.add(LongDouble::from_f64(1.0));
        assert!(result.is_infinity());
        assert!(!result.is_negative());
    }

    #[test]
    fn test_add_infinity_plus_neg_infinity_is_nan() {
        let result = LongDouble::INFINITY.add(LongDouble::NEG_INFINITY);
        assert!(result.is_nan());
    }

    #[test]
    fn test_add_neg_zero_plus_neg_zero() {
        let result = LongDouble::NEG_ZERO.add(LongDouble::NEG_ZERO);
        assert!(result.is_zero());
        assert!(result.is_negative());
    }

    #[test]
    fn test_add_pos_zero_plus_neg_zero() {
        let result = LongDouble::ZERO.add(LongDouble::NEG_ZERO);
        assert!(result.is_zero());
        // Per round-to-nearest-even: +0 + -0 = +0
        assert!(!result.is_negative());
    }

    // ====================================================================
    // Basic Arithmetic — Subtraction
    // ====================================================================

    #[test]
    fn test_sub_positive_values() {
        let a = LongDouble::from_f64(5.0);
        let b = LongDouble::from_f64(3.0);
        let result = a.sub(b);
        assert_eq!(result.to_f64(), 2.0);
    }

    #[test]
    fn test_sub_nan_propagation() {
        let result = LongDouble::NAN.sub(LongDouble::from_f64(1.0));
        assert!(result.is_nan());
    }

    // ====================================================================
    // Basic Arithmetic — Multiplication
    // ====================================================================

    #[test]
    fn test_mul_positive_values() {
        let a = LongDouble::from_f64(3.0);
        let b = LongDouble::from_f64(4.0);
        let result = a.mul(b);
        assert_eq!(result.to_f64(), 12.0);
    }

    #[test]
    fn test_mul_negative_result() {
        let a = LongDouble::from_f64(3.0);
        let b = LongDouble::from_f64(-4.0);
        let result = a.mul(b);
        assert_eq!(result.to_f64(), -12.0);
    }

    #[test]
    fn test_mul_both_negative() {
        let a = LongDouble::from_f64(-3.0);
        let b = LongDouble::from_f64(-4.0);
        let result = a.mul(b);
        assert_eq!(result.to_f64(), 12.0);
    }

    #[test]
    fn test_mul_by_zero() {
        let a = LongDouble::from_f64(42.0);
        let result = a.mul(LongDouble::ZERO);
        assert!(result.is_zero());
    }

    #[test]
    fn test_mul_infinity_by_zero_is_nan() {
        let result = LongDouble::INFINITY.mul(LongDouble::ZERO);
        assert!(result.is_nan());
    }

    #[test]
    fn test_mul_infinity_by_finite() {
        let result = LongDouble::INFINITY.mul(LongDouble::from_f64(2.0));
        assert!(result.is_infinity());
        assert!(!result.is_negative());
    }

    #[test]
    fn test_mul_infinity_by_negative() {
        let result = LongDouble::INFINITY.mul(LongDouble::from_f64(-2.0));
        assert!(result.is_infinity());
        assert!(result.is_negative());
    }

    #[test]
    fn test_mul_nan_propagation() {
        let result = LongDouble::NAN.mul(LongDouble::from_f64(1.0));
        assert!(result.is_nan());
    }

    #[test]
    fn test_mul_by_one_identity() {
        let a = LongDouble::from_f64(42.0);
        let result = a.mul(LongDouble::ONE);
        assert_eq!(result.to_f64(), 42.0);
    }

    // ====================================================================
    // Basic Arithmetic — Division
    // ====================================================================

    #[test]
    fn test_div_positive_values() {
        let a = LongDouble::from_f64(12.0);
        let b = LongDouble::from_f64(4.0);
        let result = a.div(b);
        assert_eq!(result.to_f64(), 3.0);
    }

    #[test]
    fn test_div_negative_result() {
        let a = LongDouble::from_f64(12.0);
        let b = LongDouble::from_f64(-4.0);
        let result = a.div(b);
        assert_eq!(result.to_f64(), -3.0);
    }

    #[test]
    fn test_div_by_zero_finite_is_infinity() {
        let a = LongDouble::from_f64(1.0);
        let result = a.div(LongDouble::ZERO);
        assert!(result.is_infinity());
        assert!(!result.is_negative());
    }

    #[test]
    fn test_div_negative_by_zero_is_neg_infinity() {
        let a = LongDouble::from_f64(-1.0);
        let result = a.div(LongDouble::ZERO);
        assert!(result.is_infinity());
        assert!(result.is_negative());
    }

    #[test]
    fn test_div_zero_by_zero_is_nan() {
        let result = LongDouble::ZERO.div(LongDouble::ZERO);
        assert!(result.is_nan());
    }

    #[test]
    fn test_div_infinity_by_infinity_is_nan() {
        let result = LongDouble::INFINITY.div(LongDouble::INFINITY);
        assert!(result.is_nan());
    }

    #[test]
    fn test_div_zero_by_finite() {
        let result = LongDouble::ZERO.div(LongDouble::from_f64(5.0));
        assert!(result.is_zero());
    }

    #[test]
    fn test_div_nan_propagation() {
        let result = LongDouble::NAN.div(LongDouble::from_f64(1.0));
        assert!(result.is_nan());
    }

    // ====================================================================
    // Negation
    // ====================================================================

    #[test]
    fn test_neg_positive_to_negative() {
        let a = LongDouble::from_f64(5.0);
        let result = a.neg();
        assert!(result.is_negative());
        assert_eq!(result.to_f64(), -5.0);
    }

    #[test]
    fn test_neg_negative_to_positive() {
        let a = LongDouble::from_f64(-5.0);
        let result = a.neg();
        assert!(!result.is_negative());
        assert_eq!(result.to_f64(), 5.0);
    }

    #[test]
    fn test_neg_zero() {
        let result = LongDouble::ZERO.neg();
        assert!(result.is_zero());
        assert!(result.is_negative());
    }

    // ====================================================================
    // std::ops trait implementations
    // ====================================================================

    #[test]
    fn test_ops_add_trait() {
        let a = LongDouble::from_f64(1.0);
        let b = LongDouble::from_f64(2.0);
        let result = a + b;
        assert_eq!(result.to_f64(), 3.0);
    }

    #[test]
    fn test_ops_sub_trait() {
        let a = LongDouble::from_f64(5.0);
        let b = LongDouble::from_f64(3.0);
        let result = a - b;
        assert_eq!(result.to_f64(), 2.0);
    }

    #[test]
    fn test_ops_mul_trait() {
        let a = LongDouble::from_f64(3.0);
        let b = LongDouble::from_f64(7.0);
        let result = a * b;
        assert_eq!(result.to_f64(), 21.0);
    }

    #[test]
    fn test_ops_div_trait() {
        let a = LongDouble::from_f64(10.0);
        let b = LongDouble::from_f64(2.0);
        let result = a / b;
        assert_eq!(result.to_f64(), 5.0);
    }

    #[test]
    fn test_ops_neg_trait() {
        let a = LongDouble::from_f64(3.0);
        let result = -a;
        assert_eq!(result.to_f64(), -3.0);
    }

    // ====================================================================
    // Comparison — PartialEq
    // ====================================================================

    #[test]
    fn test_eq_same_value() {
        let a = LongDouble::from_f64(1.0);
        let b = LongDouble::from_f64(1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_eq_different_values() {
        let a = LongDouble::from_f64(1.0);
        let b = LongDouble::from_f64(2.0);
        assert_ne!(a, b);
    }

    #[test]
    fn test_eq_nan_not_equal_to_itself() {
        assert_ne!(LongDouble::NAN, LongDouble::NAN);
    }

    #[test]
    fn test_eq_pos_zero_equals_neg_zero() {
        assert_eq!(LongDouble::ZERO, LongDouble::NEG_ZERO);
    }

    // ====================================================================
    // Comparison — PartialOrd
    // ====================================================================

    #[test]
    fn test_ord_positive_ordering() {
        let a = LongDouble::from_f64(1.0);
        let b = LongDouble::from_f64(2.0);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_ord_negative_ordering() {
        let a = LongDouble::from_f64(-2.0);
        let b = LongDouble::from_f64(-1.0);
        assert!(a < b);
    }

    #[test]
    fn test_ord_mixed_sign() {
        let a = LongDouble::from_f64(-1.0);
        let b = LongDouble::from_f64(1.0);
        assert!(a < b);
    }

    #[test]
    fn test_ord_nan_is_unordered() {
        let a = LongDouble::from_f64(1.0);
        assert_eq!(a.partial_cmp(&LongDouble::NAN), None);
        assert_eq!(LongDouble::NAN.partial_cmp(&a), None);
    }

    #[test]
    fn test_ord_zeros_are_equal() {
        assert_eq!(
            LongDouble::ZERO.partial_cmp(&LongDouble::NEG_ZERO),
            Some(Ordering::Equal)
        );
    }

    // ====================================================================
    // Comparison — total_cmp
    // ====================================================================

    #[test]
    fn test_total_cmp_basic_ordering() {
        let neg_inf = LongDouble::NEG_INFINITY;
        let neg_one = LongDouble::from_f64(-1.0);
        let zero = LongDouble::ZERO;
        let pos_one = LongDouble::from_f64(1.0);
        let pos_inf = LongDouble::INFINITY;

        assert_eq!(neg_inf.total_cmp(&neg_one), Ordering::Less);
        assert_eq!(neg_one.total_cmp(&zero), Ordering::Less);
        assert_eq!(zero.total_cmp(&pos_one), Ordering::Less);
        assert_eq!(pos_one.total_cmp(&pos_inf), Ordering::Less);
    }

    #[test]
    fn test_total_cmp_neg_zero_less_than_pos_zero() {
        assert_eq!(
            LongDouble::NEG_ZERO.total_cmp(&LongDouble::ZERO),
            Ordering::Less
        );
    }

    #[test]
    fn test_total_cmp_nan_is_ordered() {
        // total_cmp places +NaN above +infinity
        assert_eq!(
            LongDouble::INFINITY.total_cmp(&LongDouble::NAN),
            Ordering::Less
        );
    }

    // ====================================================================
    // 10-byte Serialization — to_bytes / from_bytes Round-Trip
    // ====================================================================

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_zero() {
        let original = LongDouble::ZERO;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert!(recovered.is_zero());
        assert!(!recovered.is_negative());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_neg_zero() {
        let original = LongDouble::NEG_ZERO;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert!(recovered.is_zero());
        assert!(recovered.is_negative());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_one() {
        let original = LongDouble::ONE;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert_eq!(recovered.to_f64(), 1.0);
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_infinity() {
        let original = LongDouble::INFINITY;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert!(recovered.is_infinity());
        assert!(!recovered.is_negative());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_neg_infinity() {
        let original = LongDouble::NEG_INFINITY;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert!(recovered.is_infinity());
        assert!(recovered.is_negative());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_nan() {
        let original = LongDouble::NAN;
        let bytes = original.to_bytes();
        let recovered = LongDouble::from_bytes(&bytes);
        assert!(recovered.is_nan());
    }

    #[test]
    fn test_to_bytes_from_bytes_roundtrip_various() {
        let values = [1.0, -1.0, 0.5, 42.0, -42.0, 3.14159, 1.0e10, -1.0e10];
        for &v in &values {
            let original = LongDouble::from_f64(v);
            let bytes = original.to_bytes();
            let recovered = LongDouble::from_bytes(&bytes);
            assert_eq!(
                recovered.to_f64(),
                original.to_f64(),
                "Round-trip failed for f64 value {}",
                v
            );
        }
    }

    #[test]
    fn test_to_bytes_layout_zero() {
        let bytes = LongDouble::ZERO.to_bytes();
        // Significand = 0, exponent = 0, sign = 0
        assert_eq!(bytes, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ====================================================================
    // Display and Debug
    // ====================================================================

    #[test]
    fn test_display_nan() {
        let s = format!("{}", LongDouble::NAN);
        assert_eq!(s, "NaN");
    }

    #[test]
    fn test_display_infinity() {
        assert_eq!(format!("{}", LongDouble::INFINITY), "inf");
        assert_eq!(format!("{}", LongDouble::NEG_INFINITY), "-inf");
    }

    #[test]
    fn test_display_finite() {
        let ld = LongDouble::from_f64(42.0);
        let s = format!("{}", ld);
        assert_eq!(s, "42");
    }

    #[test]
    fn test_debug_format() {
        let ld = LongDouble::ZERO;
        let s = format!("{:?}", ld);
        assert!(s.starts_with("LongDouble {"));
        assert!(s.contains("sign:"));
        assert!(s.contains("exp:"));
        assert!(s.contains("sig:"));
    }

    // ====================================================================
    // Denormal Numbers
    // ====================================================================

    #[test]
    fn test_denormal_classification() {
        let denormal = LongDouble {
            sign: false,
            exponent: 0,
            significand: 1,
        };
        assert!(denormal.is_denormal());
        assert!(!denormal.is_zero());
        assert!(!denormal.is_nan());
        assert!(!denormal.is_infinity());
    }

    // ====================================================================
    // Overflow and Underflow
    // ====================================================================

    #[test]
    fn test_mul_overflow_to_infinity() {
        // Construct a value near the maximum representable long double.
        // Biased exponent 32766 (max normal), integer bit set = largest normal.
        // Squaring this produces unbiased exponent 16383 + 16383 = 32766,
        // biased = 32766 + 16383 = 49149 > 32767, which overflows to infinity.
        let big = LongDouble {
            sign: false,
            exponent: 32766,
            significand: LongDouble::INTEGER_BIT,
        };
        let result = big.mul(big);
        assert!(result.is_infinity());
    }

    #[test]
    fn test_div_very_small_underflow() {
        // Divide a small number by a very large number
        let tiny = LongDouble::from_f64(f64::MIN_POSITIVE);
        let huge = LongDouble::from_f64(f64::MAX);
        let result = tiny.div(huge);
        // Result should be very small (possibly zero/denormal)
        assert!(!result.is_nan());
        assert!(!result.is_infinity());
    }

    // ====================================================================
    // Mixed Arithmetic Verification
    // ====================================================================

    #[test]
    fn test_arithmetic_chain() {
        // (2 + 3) * 4 - 10 / 2 = 20 - 5 = 15
        let two = LongDouble::from_f64(2.0);
        let three = LongDouble::from_f64(3.0);
        let four = LongDouble::from_f64(4.0);
        let ten = LongDouble::from_f64(10.0);

        let sum = two.add(three);
        let product = sum.mul(four);
        let quotient = ten.div(two);
        let result = product.sub(quotient);
        assert_eq!(result.to_f64(), 15.0);
    }

    #[test]
    fn test_clone_copy() {
        let a = LongDouble::from_f64(1.0);
        let b = a; // copy
        let c = a.clone();
        assert_eq!(a.to_f64(), b.to_f64());
        assert_eq!(a.to_f64(), c.to_f64());
    }
}
