//! Software implementation of 80-bit IEEE 754 extended-precision (long double) arithmetic.
//!
//! Provides the [`LongDouble`] type representing the x86-64 extended precision format:
//! 1 sign bit, 15 exponent bits, 64 significand bits (with explicit integer bit).
//!
//! All operations are implemented in software with zero external dependencies,
//! as required by the project's zero-dependency mandate. This module replaces what
//! would normally require the `num` or `rug` crate.
//!
//! # Usage
//! Used by:
//! - Constant expression evaluation in the frontend (`constant_eval.rs`)
//! - Floating-point code generation in the backend for compile-time long double computation
//!
//! # IEEE 754 Extended Precision Format
//! ```text
//! Bit layout (80 bits total):
//!   [79]       Sign (1 bit)
//!   [78:64]    Biased exponent (15 bits, bias = 16383)
//!   [63:0]     Significand (64 bits, with explicit integer bit at position 63)
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Exponent bias for 80-bit extended precision: 2^14 - 1 = 16383.
const BIAS: i32 = 16383;

/// Maximum biased exponent value (reserved for infinity and NaN).
const MAX_EXPONENT: u16 = 0x7FFF;

/// The explicit integer bit at position 63 of the 64-bit significand.
/// Normal numbers always have this bit set.
const INTEGER_BIT: u64 = 1u64 << 63;

/// Number of guard bits used when widening significands to u128 for
/// intermediate arithmetic. The 64-bit significand occupies bits [126:63]
/// of the u128, leaving 63 lower bits for rounding precision (guard, round,
/// sticky information).
const WIDE_SHIFT: u32 = 63;

// ═══════════════════════════════════════════════════════════════════════════════
// LongDouble struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Software-emulated 80-bit IEEE 754 extended-precision floating-point number.
///
/// # Format
/// Matches the x86-64 `long double` memory layout:
/// - `sign`: `false` = positive, `true` = negative
/// - `exponent`: 15-bit biased exponent (bias = 16383)
/// - `significand`: 64-bit significand with explicit integer bit at bit 63
///
/// # Value Classification
/// | Exponent    | Integer bit | Fraction bits | Class      |
/// |-------------|-------------|---------------|------------|
/// | 0           | 0           | 0             | Zero       |
/// | 0           | 0           | non-zero      | Subnormal  |
/// | 1..0x7FFE   | 1           | any           | Normal     |
/// | 0x7FFF      | 1           | 0             | Infinity   |
/// | 0x7FFF      | 1           | non-zero      | NaN        |
#[derive(Clone, Copy)]
pub struct LongDouble {
    /// Sign bit: `false` = positive, `true` = negative.
    pub sign: bool,
    /// 15-bit biased exponent. Bias = 16383.
    /// Special values: 0 = zero/subnormal, 0x7FFF = infinity/NaN.
    pub exponent: u16,
    /// 64-bit significand with explicit integer bit at position 63.
    /// For normal numbers, bit 63 is always 1.
    pub significand: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Special value constants
// ═══════════════════════════════════════════════════════════════════════════════

impl LongDouble {
    /// Positive zero: +0.0
    pub const ZERO: LongDouble = LongDouble {
        sign: false,
        exponent: 0,
        significand: 0,
    };

    /// Negative zero: -0.0
    pub const NEG_ZERO: LongDouble = LongDouble {
        sign: true,
        exponent: 0,
        significand: 0,
    };

    /// Positive one: 1.0
    /// True exponent = 0, so biased exponent = BIAS = 16383 = 0x3FFF.
    /// Significand = 1.0 in binary = just the integer bit.
    pub const ONE: LongDouble = LongDouble {
        sign: false,
        exponent: 0x3FFF,
        significand: INTEGER_BIT,
    };

    /// Positive infinity: +∞
    pub const INFINITY: LongDouble = LongDouble {
        sign: false,
        exponent: MAX_EXPONENT,
        significand: INTEGER_BIT,
    };

    /// Negative infinity: -∞
    pub const NEG_INFINITY: LongDouble = LongDouble {
        sign: true,
        exponent: MAX_EXPONENT,
        significand: INTEGER_BIT,
    };

    /// Quiet NaN (Not a Number).
    /// Exponent = 0x7FFF, integer bit (63) and quiet-NaN bit (62) both set.
    pub const NAN: LongDouble = LongDouble {
        sign: false,
        exponent: MAX_EXPONENT,
        significand: 0xC000_0000_0000_0000,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Classification methods
// ═══════════════════════════════════════════════════════════════════════════════

impl LongDouble {
    /// Returns `true` if this value is NaN (Not a Number).
    ///
    /// NaN is indicated by exponent = 0x7FFF with a significand that is
    /// neither the infinity pattern (just the integer bit) nor zero.
    pub fn is_nan(&self) -> bool {
        if self.exponent != MAX_EXPONENT {
            return false;
        }
        // Infinity: significand == INTEGER_BIT exactly
        // Pseudo-infinity: significand == 0 (treated as NaN for safety)
        // NaN: anything else
        self.significand != INTEGER_BIT && self.significand != 0
    }

    /// Returns `true` if this value is positive or negative infinity.
    pub fn is_infinity(&self) -> bool {
        self.exponent == MAX_EXPONENT && self.significand == INTEGER_BIT
    }

    /// Returns `true` if this value is positive or negative zero.
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.significand == 0
    }

    /// Returns `true` if this value is subnormal (denormalized).
    ///
    /// A subnormal (denormalized) number has a zero exponent field but
    /// a non-zero significand, representing values very close to zero that
    /// sacrifice precision for extended range.
    pub fn is_subnormal(&self) -> bool {
        self.exponent == 0 && self.significand != 0
    }

    /// Returns `true` if this value is finite (not infinity, not NaN).
    ///
    /// Finite values include normal numbers, subnormal numbers, and zero.
    /// Returns `false` for infinities and NaN values.
    pub fn is_finite(&self) -> bool {
        self.exponent < MAX_EXPONENT
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helper functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply IEEE 754 round-to-nearest-even (banker's rounding).
///
/// Given a 64-bit significand and two rounding indicators:
/// - `guard`: the most significant discarded bit (the "halfway" indicator)
/// - `sticky`: logical OR of all remaining discarded bits
///
/// Returns `(rounded_significand, overflowed)`. When `overflowed` is `true`,
/// the significand wrapped around (was `u64::MAX`, became 0), and the caller
/// must increment the exponent and set significand to `INTEGER_BIT`.
fn round_to_nearest_even(sig: u64, guard: bool, sticky: bool) -> (u64, bool) {
    // Round up when: guard AND (sticky OR significand-LSB-is-odd)
    // This ensures ties (guard=1, sticky=0) break to even (round up only if odd).
    if guard && (sticky || (sig & 1) != 0) {
        let (result, overflow) = sig.overflowing_add(1);
        (result, overflow)
    } else {
        (sig, false)
    }
}

/// Logical right-shift a u128 by `shift` bits, returning the shifted value
/// and a sticky bit that is `true` if any discarded bits were non-zero.
fn shift_right_sticky_u128(val: u128, shift: u32) -> (u128, bool) {
    if shift == 0 {
        (val, false)
    } else if shift < 128 {
        let mask = (1u128 << shift) - 1;
        let sticky = (val & mask) != 0;
        (val >> shift, sticky)
    } else {
        (0, val != 0)
    }
}

/// Construct a properly rounded LongDouble from intermediate arithmetic results.
///
/// Takes the result sign, a biased exponent (which may be out of range), a 64-bit
/// significand, and guard/sticky rounding bits. Handles:
/// - Overflow → ±Infinity
/// - Underflow → subnormal or ±zero
/// - Round-to-nearest-even
/// - Rounding-induced exponent increment
fn assemble_float(
    sign: bool,
    mut biased_exp: i32,
    mut sig: u64,
    mut guard: bool,
    mut sticky: bool,
) -> LongDouble {
    // Overflow: exponent too large → infinity
    if biased_exp >= MAX_EXPONENT as i32 {
        return if sign {
            LongDouble::NEG_INFINITY
        } else {
            LongDouble::INFINITY
        };
    }

    // Underflow: result is subnormal or zero
    if biased_exp <= 0 {
        let shift = (1 - biased_exp) as u32;
        if shift > 64 {
            // Complete underflow: all significand bits shifted away
            return if sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }
        if shift > 0 {
            // Denormalize: shift right, merge lost bits into guard/sticky
            let new_guard = (sig >> (shift - 1)) & 1 != 0;
            let new_sticky = if shift > 1 {
                (sig & ((1u64 << (shift - 1)) - 1)) != 0 || guard || sticky
            } else {
                guard || sticky
            };
            sig >>= shift;
            guard = new_guard;
            sticky = new_sticky;
        }
        biased_exp = 0;
    }

    // Apply rounding
    let (sig, overflow) = round_to_nearest_even(sig, guard, sticky);
    if overflow {
        // Significand rolled over from 0xFFFF...F to 0: carry into exponent
        let new_exp = biased_exp + 1;
        if new_exp >= MAX_EXPONENT as i32 {
            return if sign {
                LongDouble::NEG_INFINITY
            } else {
                LongDouble::INFINITY
            };
        }
        return LongDouble {
            sign,
            exponent: new_exp as u16,
            significand: INTEGER_BIT,
        };
    }

    // Rounding a subnormal may have produced a normal (integer bit now set)
    if biased_exp == 0 && (sig & INTEGER_BIT) != 0 {
        biased_exp = 1;
    }

    LongDouble {
        sign,
        exponent: biased_exp as u16,
        significand: sig,
    }
}

/// Compare magnitudes of two LongDouble values (ignoring sign).
fn compare_magnitude(a: &LongDouble, b: &LongDouble) -> Ordering {
    match a.exponent.cmp(&b.exponent) {
        Ordering::Equal => a.significand.cmp(&b.significand),
        other => other,
    }
}

/// Get the effective biased exponent for arithmetic purposes.
/// For subnormals (biased exponent = 0), the effective exponent is 1 because
/// the significand lacks the implicit integer bit and represents
/// `0.fraction × 2^(1 − BIAS)` rather than `1.fraction × 2^(0 − BIAS)`.
fn effective_exponent(ld: &LongDouble) -> i32 {
    if ld.exponent == 0 {
        1
    } else {
        ld.exponent as i32
    }
}

/// Add magnitudes of two finite, non-zero, same-sign operands.
/// Returns a LongDouble with the given `sign`.
fn add_magnitudes(sign: bool, a: &LongDouble, b: &LongDouble) -> LongDouble {
    // Ensure `larger` has the greater-or-equal magnitude
    let (larger, smaller) = if compare_magnitude(a, b) != Ordering::Less {
        (a, b)
    } else {
        (b, a)
    };

    let exp_l = effective_exponent(larger);
    let exp_s = effective_exponent(smaller);
    let d = (exp_l - exp_s) as u32;

    // Widen significands to u128 with WIDE_SHIFT (63) guard bits below.
    // Each widened value occupies bits [126:63] of the u128 at most.
    let a_wide: u128 = (larger.significand as u128) << WIDE_SHIFT;
    let (b_wide, shift_sticky) =
        shift_right_sticky_u128((smaller.significand as u128) << WIDE_SHIFT, d);
    // Set sticky bit in LSB if any bits were lost during alignment
    let b_wide = b_wide | (shift_sticky as u128);

    let sum = a_wide + b_wide;
    // Maximum sum: 2 × ((2^64 − 1) << 63) = 2^128 − 2^64, fits in u128.
    // The MSB can be at position 127 (overflow from addition).

    let result_exp: i32;
    let sig: u64;
    let guard: bool;
    let sticky: bool;

    if sum & (1u128 << 127) != 0 {
        // Carry out: MSB at bit 127. Shift right by 1 to normalize.
        let extra_sticky = (sum & 1) != 0;
        let normalized = sum >> 1;
        sig = (normalized >> WIDE_SHIFT) as u64;
        let round_bits = (normalized & ((1u128 << WIDE_SHIFT) - 1)) as u64;
        guard = (round_bits >> 62) & 1 != 0;
        sticky = (round_bits & ((1u64 << 62) - 1)) != 0 || extra_sticky;
        result_exp = exp_l + 1;
    } else {
        // No carry: MSB at bit 126 or below.
        sig = (sum >> WIDE_SHIFT) as u64;
        let round_bits = (sum & ((1u128 << WIDE_SHIFT) - 1)) as u64;
        guard = (round_bits >> 62) & 1 != 0;
        sticky = (round_bits & ((1u64 << 62) - 1)) != 0;
        result_exp = exp_l;
    }

    assemble_float(sign, result_exp, sig, guard, sticky)
}

/// Subtract magnitudes: compute |larger| − |smaller|.
/// `sign` is the sign of the result (the sign of the operand with greater magnitude).
/// Caller guarantees |larger| >= |smaller|.
fn sub_magnitudes(sign: bool, larger: &LongDouble, smaller: &LongDouble) -> LongDouble {
    let exp_l = effective_exponent(larger);
    let exp_s = effective_exponent(smaller);
    let d = (exp_l - exp_s) as u32;

    let a_wide: u128 = (larger.significand as u128) << WIDE_SHIFT;
    let (b_wide, shift_sticky) =
        shift_right_sticky_u128((smaller.significand as u128) << WIDE_SHIFT, d);
    // For subtraction, sticky bits from alignment mean b_wide is slightly smaller
    // than the exact shifted value. We add the sticky to b_wide so the subtraction
    // is correctly rounded (the borrow is accounted for).
    let b_wide = b_wide | (shift_sticky as u128);

    // a_wide >= b_wide since |larger| >= |smaller|
    let diff = a_wide.wrapping_sub(b_wide);

    if diff == 0 {
        return LongDouble::ZERO;
    }

    // Normalize: place the MSB at position 126 (= WIDE_SHIFT + 63)
    let lz = diff.leading_zeros();
    let msb_pos = 127 - lz;
    let target_pos: u32 = WIDE_SHIFT + 63; // = 126

    let result_exp: i32;
    let normalized: u128;

    if msb_pos > target_pos {
        // MSB above target: shift right (rare in subtraction)
        let shift = msb_pos - target_pos;
        let extra_sticky = (diff & ((1u128 << shift) - 1)) != 0;
        normalized = (diff >> shift) | (extra_sticky as u128);
        result_exp = exp_l + shift as i32;
    } else if msb_pos < target_pos {
        // MSB below target: massive cancellation, shift left
        let shift = target_pos - msb_pos;
        normalized = diff << shift;
        result_exp = exp_l - shift as i32;
    } else {
        normalized = diff;
        result_exp = exp_l;
    }

    let sig = (normalized >> WIDE_SHIFT) as u64;
    let round_bits = (normalized & ((1u128 << WIDE_SHIFT) - 1)) as u64;
    let guard = (round_bits >> 62) & 1 != 0;
    let sticky = (round_bits & ((1u64 << 62) - 1)) != 0;

    assemble_float(sign, result_exp, sig, guard, sticky)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Arithmetic operations (public API)
// ═══════════════════════════════════════════════════════════════════════════════

impl LongDouble {
    /// Add two extended-precision values with IEEE 754 round-to-nearest-even.
    pub fn add(a: &LongDouble, b: &LongDouble) -> LongDouble {
        // NaN propagation (NaN is "sticky")
        if a.is_nan() {
            return *a;
        }
        if b.is_nan() {
            return *b;
        }

        // Infinity arithmetic
        if a.is_infinity() {
            if b.is_infinity() && a.sign != b.sign {
                return LongDouble::NAN; // (+∞) + (−∞) = NaN
            }
            return *a;
        }
        if b.is_infinity() {
            return *b;
        }

        // Zero arithmetic
        if a.is_zero() {
            if b.is_zero() {
                // IEEE 754: +0 + −0 = +0 in round-to-nearest; −0 + −0 = −0
                return if a.sign && b.sign {
                    LongDouble::NEG_ZERO
                } else {
                    LongDouble::ZERO
                };
            }
            return *b;
        }
        if b.is_zero() {
            return *a;
        }

        // Both operands are finite and non-zero
        if a.sign == b.sign {
            // Same sign: add magnitudes, keep the common sign
            add_magnitudes(a.sign, a, b)
        } else {
            // Different sign: subtract the smaller magnitude from the larger
            match compare_magnitude(a, b) {
                Ordering::Greater => sub_magnitudes(a.sign, a, b),
                Ordering::Less => sub_magnitudes(b.sign, b, a),
                Ordering::Equal => LongDouble::ZERO, // IEEE 754: result is +0 in RNE mode
            }
        }
    }

    /// Subtract: `a − b`. Implemented as `a + (−b)`.
    pub fn sub(a: &LongDouble, b: &LongDouble) -> LongDouble {
        let neg_b = LongDouble {
            sign: !b.sign,
            ..*b
        };
        LongDouble::add(a, &neg_b)
    }

    /// Multiply two extended-precision values.
    pub fn mul(a: &LongDouble, b: &LongDouble) -> LongDouble {
        let result_sign = a.sign ^ b.sign;

        // NaN propagation
        if a.is_nan() {
            return *a;
        }
        if b.is_nan() {
            return *b;
        }

        // Infinity × 0 = NaN; Infinity × finite = Infinity
        if a.is_infinity() {
            return if b.is_zero() {
                LongDouble::NAN
            } else if result_sign {
                LongDouble::NEG_INFINITY
            } else {
                LongDouble::INFINITY
            };
        }
        if b.is_infinity() {
            return if a.is_zero() {
                LongDouble::NAN
            } else if result_sign {
                LongDouble::NEG_INFINITY
            } else {
                LongDouble::INFINITY
            };
        }

        // Zero × anything-finite = zero
        if a.is_zero() || b.is_zero() {
            return if result_sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }

        let exp_a = effective_exponent(a);
        let exp_b = effective_exponent(b);

        // Full 128-bit product of the two 64-bit significands.
        let product: u128 = (a.significand as u128) * (b.significand as u128);

        // Result biased exponent before normalization:
        //   true_result = (exp_a − BIAS) + (exp_b − BIAS)
        //   biased_result = true_result + BIAS = exp_a + exp_b − BIAS
        let mut result_exp: i32 = exp_a + exp_b - BIAS;

        if product == 0 {
            return if result_sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }

        // Product MSB analysis:
        //   min (both = 2^63): product = 2^126 → MSB at bit 126
        //   max (both ≈ 2^64): product ≈ 2^128 → MSB at bit 127
        // For subnormal inputs, MSB can be lower.
        let msb_pos = 127 - product.leading_zeros();

        let sig: u64;
        let guard: bool;
        let sticky: bool;

        if msb_pos >= 127 {
            // MSB at bit 127: significand in bits [127:64], rounding in [63:0]
            sig = (product >> 64) as u64;
            guard = (product >> 63) & 1 != 0;
            sticky = (product & ((1u128 << 63) - 1)) != 0;
            result_exp += 1;
        } else if msb_pos >= 126 {
            // MSB at bit 126: significand in bits [126:63], rounding in [62:0]
            sig = (product >> 63) as u64;
            guard = (product >> 62) & 1 != 0;
            sticky = (product & ((1u128 << 62) - 1)) != 0;
        } else {
            // MSB below 126 (subnormal inputs): shift left to normalize
            let shift = 126 - msb_pos;
            let shifted = product << shift;
            sig = (shifted >> 63) as u64;
            guard = (shifted >> 62) & 1 != 0;
            sticky = (shifted & ((1u128 << 62) - 1)) != 0;
            result_exp -= shift as i32;
        }

        assemble_float(result_sign, result_exp, sig, guard, sticky)
    }

    /// Divide: `a / b`.
    pub fn div(a: &LongDouble, b: &LongDouble) -> LongDouble {
        let result_sign = a.sign ^ b.sign;

        // NaN propagation
        if a.is_nan() {
            return *a;
        }
        if b.is_nan() {
            return *b;
        }

        // ∞ / ∞ = NaN
        if a.is_infinity() && b.is_infinity() {
            return LongDouble::NAN;
        }
        // ∞ / finite = ∞
        if a.is_infinity() {
            return if result_sign {
                LongDouble::NEG_INFINITY
            } else {
                LongDouble::INFINITY
            };
        }
        // finite / ∞ = 0
        if b.is_infinity() {
            return if result_sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }
        // x / 0: 0/0 = NaN, nonzero/0 = ∞
        if b.is_zero() {
            return if a.is_zero() {
                LongDouble::NAN
            } else if result_sign {
                LongDouble::NEG_INFINITY
            } else {
                LongDouble::INFINITY
            };
        }
        // 0 / x = 0
        if a.is_zero() {
            return if result_sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }

        let exp_a = effective_exponent(a);
        let exp_b = effective_exponent(b);

        // Result biased exponent derivation:
        //   value(a) = a_sig × 2^(exp_a − BIAS − 63)
        //   value(b) = b_sig × 2^(exp_b − BIAS − 63)
        //   a/b = (a_sig/b_sig) × 2^(exp_a − exp_b)
        //   We compute quotient = (a_sig << 64) / b_sig = (a_sig/b_sig) × 2^64
        //   Result: result_sig × 2^(result_exp − BIAS − 63) = (a_sig/b_sig) × 2^(exp_a − exp_b)
        //   When quotient MSB is at bit N (shift right by N−63 to normalize):
        //     result_exp = exp_a − exp_b + BIAS + N − 64
        //   Base for N=63 (no shift): result_exp = exp_a − exp_b + BIAS − 1
        let mut result_exp: i32 = exp_a - exp_b + BIAS - 1;

        // Compute quotient with extra precision:
        //   dividend = a_sig << 64 (fits in u128 since a_sig < 2^64)
        //   quotient ≈ (a_sig/b_sig) × 2^64 → MSB typically at bit 63 or 64
        //   remainder for IEEE 754 rounding
        let dividend: u128 = (a.significand as u128) << 64;
        let divisor: u128 = b.significand as u128;
        let quotient: u128 = dividend / divisor;
        let remainder: u128 = dividend % divisor;

        if quotient == 0 {
            return if result_sign {
                LongDouble::NEG_ZERO
            } else {
                LongDouble::ZERO
            };
        }

        let msb_pos = 127 - quotient.leading_zeros();
        let sig: u64;
        let guard: bool;
        let sticky: bool;

        if msb_pos > 63 {
            // MSB above bit 63: shift right by (msb_pos − 63) to place
            // the integer bit at position 63.
            let shift = msb_pos - 63;
            sig = (quotient >> shift) as u64;
            guard = (quotient >> (shift - 1)) & 1 != 0;
            let shift_lost = if shift > 1 {
                (quotient & ((1u128 << (shift - 1)) - 1)) != 0
            } else {
                false
            };
            sticky = shift_lost || remainder != 0;
            result_exp += shift as i32;
        } else if msb_pos == 63 {
            // MSB at bit 63: perfect alignment, no shift needed.
            sig = quotient as u64;
            // Rounding comes from the remainder: compare 2 × remainder vs divisor
            let double_rem = remainder << 1;
            guard = double_rem >= divisor;
            sticky = if guard {
                (double_rem - divisor) != 0
            } else {
                remainder != 0
            };
        } else {
            // MSB below bit 63 (subnormal dividend): extract extra quotient
            // bits from the remainder to fill the significand up to bit 63.
            let shift = 63 - msb_pos;
            // Compute extra quotient bits: (remainder << shift) / divisor
            // Safe: remainder < divisor < 2^64, shift ≤ 63,
            // so remainder << 63 < 2^127, fits in u128.
            let extra_dividend = remainder << shift;
            let extra_quotient = extra_dividend / divisor;
            let extra_remainder = extra_dividend % divisor;
            sig = ((quotient << shift) | extra_quotient) as u64;
            let double_extra_rem = extra_remainder << 1;
            guard = double_extra_rem >= divisor;
            sticky = if guard {
                (double_extra_rem - divisor) != 0
            } else {
                extra_remainder != 0
            };
            result_exp -= shift as i32;
        }

        assemble_float(result_sign, result_exp, sig, guard, sticky)
    }

    /// Negate: flip the sign bit. Works correctly for all values including NaN and zero.
    pub fn neg(a: &LongDouble) -> LongDouble {
        LongDouble {
            sign: !a.sign,
            ..*a
        }
    }

    /// Absolute value: clear the sign bit.
    pub fn abs(a: &LongDouble) -> LongDouble {
        LongDouble { sign: false, ..*a }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Comparison: PartialEq and PartialOrd
// ═══════════════════════════════════════════════════════════════════════════════

impl PartialEq for LongDouble {
    fn eq(&self, other: &Self) -> bool {
        // IEEE 754: NaN ≠ NaN
        if self.is_nan() || other.is_nan() {
            return false;
        }
        // IEEE 754: +0 == −0
        if self.is_zero() && other.is_zero() {
            return true;
        }
        // Otherwise compare all fields
        self.sign == other.sign
            && self.exponent == other.exponent
            && self.significand == other.significand
    }
}

impl PartialOrd for LongDouble {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // IEEE 754: NaN is unordered (returns None)
        if self.is_nan() || other.is_nan() {
            return None;
        }
        // +0 == −0
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }

        // Different signs: positive > negative (we already handled both-zero)
        if self.sign != other.sign {
            return if self.sign {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            };
        }

        // Same sign: compare magnitudes, then account for sign
        let mag_cmp = compare_magnitude(self, other);

        if self.sign {
            // Both negative: larger magnitude means smaller value
            Some(mag_cmp.reverse())
        } else {
            // Both positive: larger magnitude means larger value
            Some(mag_cmp)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversions
// ═══════════════════════════════════════════════════════════════════════════════

impl LongDouble {
    /// Convert an `f64` (IEEE 754 binary64) to extended precision.
    ///
    /// The conversion is exact (binary64 is a strict subset of 80-bit extended).
    pub fn from_f64(val: f64) -> LongDouble {
        let bits = val.to_bits();
        let sign = (bits >> 63) != 0;
        let f64_exp = ((bits >> 52) & 0x7FF) as u16;
        let f64_frac = bits & ((1u64 << 52) - 1);

        // Zero
        if f64_exp == 0 && f64_frac == 0 {
            return LongDouble {
                sign,
                exponent: 0,
                significand: 0,
            };
        }

        // NaN
        if f64_exp == 0x7FF && f64_frac != 0 {
            return LongDouble {
                sign,
                exponent: MAX_EXPONENT,
                // Preserve quiet/signaling status: shift fraction to 80-bit position
                significand: INTEGER_BIT | (f64_frac << 11),
            };
        }

        // Infinity
        if f64_exp == 0x7FF {
            return LongDouble {
                sign,
                exponent: MAX_EXPONENT,
                significand: INTEGER_BIT,
            };
        }

        // Subnormal f64
        if f64_exp == 0 {
            // true exponent = 1 − 1023 = −1022
            // significand = 0.fraction (no implicit 1)
            let lz = f64_frac.leading_zeros() - 12; // leading zeros among the 52-bit fraction
            let normalized_frac = f64_frac << (lz + 1); // shift to make MSB implicit
            let sig = normalized_frac << 11; // widen from 52 to 63 bits (bit 63 is integer bit)
                                             // Adjusted true exponent: −1022 − lz
                                             // Biased 80-bit exponent: (−1022 − lz) + 16383 = 15361 − lz
            let biased = 15361i32 - lz as i32;
            if biased <= 0 {
                // Extremely small subnormal: becomes subnormal in 80-bit too
                // This practically never happens since 80-bit has a much wider exponent range
                return LongDouble {
                    sign,
                    exponent: 0,
                    significand: f64_frac << 11,
                };
            }
            return LongDouble {
                sign,
                exponent: biased as u16,
                significand: sig,
            };
        }

        // Normal f64
        // f64 true exponent: f64_exp − 1023
        // 80-bit biased exponent: (f64_exp − 1023) + 16383 = f64_exp + 15360
        let biased_exp = f64_exp + 15360;
        // f64 significand: 1.fraction → 53 bits. 80-bit: explicit integer bit + shift
        let sig = INTEGER_BIT | (f64_frac << 11);

        LongDouble {
            sign,
            exponent: biased_exp,
            significand: sig,
        }
    }

    /// Convert to `f64` with potential precision loss (round-to-nearest-even).
    pub fn to_f64(&self) -> f64 {
        // Zero
        if self.is_zero() {
            return if self.sign { -0.0f64 } else { 0.0f64 };
        }

        // NaN
        if self.is_nan() {
            return f64::NAN;
        }

        // Infinity
        if self.is_infinity() {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        // Subnormal 80-bit value
        if self.is_subnormal() {
            // True exponent = 1 − 16383 = −16382
            // This is way below f64's minimum (−1022), so it rounds to ±0
            return if self.sign { -0.0f64 } else { 0.0f64 };
        }

        // Normal 80-bit value
        // true exponent = self.exponent − 16383
        // f64 biased exponent = true_exp + 1023 = self.exponent − 15360
        let f64_biased_exp = self.exponent as i32 - 15360;

        if f64_biased_exp >= 0x7FF {
            // Overflow: too large for f64 → infinity
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        if f64_biased_exp <= 0 {
            // Underflow: becomes f64 subnormal or zero
            // f64 subnormal: exponent field = 0, fraction = shifted significand
            let shift = 1 - f64_biased_exp; // how much to shift right
            if shift > 63 {
                return if self.sign { -0.0f64 } else { 0.0f64 };
            }
            // The f64 fraction is derived from the 80-bit significand
            // Normal 80-bit sig: bit 63 = integer bit, bits [62:0] = fraction
            // For f64: we need to map to 52-bit fraction field
            // Shift: first drop the integer bit and upper bits, then shift for subnormal
            let full_sig = self.significand;
            let total_shift = 11 + shift as u32; // 11 for 63→52 bit reduction + shift for subnormal
            if total_shift >= 64 {
                return if self.sign { -0.0f64 } else { 0.0f64 };
            }
            let frac = full_sig >> total_shift;
            // Rounding
            let guard_bit = if total_shift > 0 {
                (full_sig >> (total_shift - 1)) & 1 != 0
            } else {
                false
            };
            let sticky_bits = if total_shift > 1 {
                (full_sig & ((1u64 << (total_shift - 1)) - 1)) != 0
            } else {
                false
            };
            let round_up = guard_bit && (sticky_bits || (frac & 1) != 0);
            let frac = if round_up { frac + 1 } else { frac };

            // If rounding promoted to normal, handle overflow
            if frac >= (1u64 << 52) {
                // Became smallest normal f64
                let bits = ((self.sign as u64) << 63) | (1u64 << 52);
                return f64::from_bits(bits);
            }

            let bits = ((self.sign as u64) << 63) | frac;
            return f64::from_bits(bits);
        }

        // Normal f64 result
        // Significand conversion: 80-bit has 64-bit sig with explicit integer bit
        // f64 has 52-bit fraction with implicit integer bit
        // Drop integer bit, take upper 52 bits of remaining 63 bits → shift right by 11
        let frac = (self.significand >> 11) & ((1u64 << 52) - 1);

        // Rounding: the 11 low bits are discarded
        let discarded = self.significand & 0x7FF;
        let guard = (discarded >> 10) & 1 != 0;
        let sticky = (discarded & 0x3FF) != 0;
        let round_up = guard && (sticky || (frac & 1) != 0);
        let frac = if round_up { frac + 1 } else { frac };

        // Check if rounding overflowed the fraction field (52 bits)
        let (frac, f64_biased_exp) = if frac >= (1u64 << 52) {
            // Carry into exponent: fraction becomes 0, exponent increments
            (0u64, f64_biased_exp + 1)
        } else {
            (frac, f64_biased_exp)
        };

        if f64_biased_exp >= 0x7FF {
            return if self.sign {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        let bits = ((self.sign as u64) << 63) | ((f64_biased_exp as u64) << 52) | frac;
        f64::from_bits(bits)
    }

    /// Convert a signed 64-bit integer to extended precision. The conversion is exact.
    pub fn from_i64(val: i64) -> LongDouble {
        if val == 0 {
            return LongDouble::ZERO;
        }
        let sign = val < 0;
        // Handle i64::MIN carefully: its absolute value (2^63) does not fit in i64
        let abs_val: u64 = if val == i64::MIN {
            1u64 << 63
        } else if val < 0 {
            (-val) as u64
        } else {
            val as u64
        };
        from_u64_inner(sign, abs_val)
    }

    /// Convert to a signed 64-bit integer by truncating toward zero.
    /// Returns `None` if the value is NaN, infinite, or out of i64 range.
    pub fn to_i64(&self) -> Option<i64> {
        if self.is_nan() || self.is_infinity() {
            return None;
        }
        if self.is_zero() {
            return Some(0);
        }

        // True exponent
        let true_exp = if self.is_subnormal() {
            1 - BIAS
        } else {
            self.exponent as i32 - BIAS
        };

        // If true exponent is negative, value is < 1, truncates to 0
        if true_exp < 0 {
            return Some(0);
        }

        // The significand represents: sig × 2^(true_exp − 63)
        // Integer part = sig >> (63 − true_exp) if true_exp <= 63
        if true_exp > 63 {
            return None; // Value ≥ 2^64, definitely out of range
        }

        let shift = (63 - true_exp) as u32;
        let magnitude = self.significand >> shift;

        if self.sign {
            // Negative: valid range is −2^63
            if magnitude > (1u64 << 63) {
                return None;
            }
            if magnitude == (1u64 << 63) {
                Some(i64::MIN)
            } else {
                Some(-(magnitude as i64))
            }
        } else {
            // Positive: valid range is 0..=2^63 − 1
            if magnitude > i64::MAX as u64 {
                return None;
            }
            Some(magnitude as i64)
        }
    }

    /// Convert an unsigned 64-bit integer to extended precision. The conversion is exact.
    pub fn from_u64(val: u64) -> LongDouble {
        if val == 0 {
            return LongDouble::ZERO;
        }
        from_u64_inner(false, val)
    }

    /// Construct a LongDouble from its raw 80-bit (10-byte) little-endian representation.
    ///
    /// Byte layout:
    /// - `bytes[0..8]`: 64-bit significand (little-endian)
    /// - `bytes[8..10]`: sign (bit 15) | exponent (bits 14:0) (little-endian)
    pub fn from_bytes(bytes: [u8; 10]) -> LongDouble {
        let significand = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let exp_sign = u16::from_le_bytes([bytes[8], bytes[9]]);
        let sign = (exp_sign >> 15) != 0;
        let exponent = exp_sign & 0x7FFF;

        LongDouble {
            sign,
            exponent,
            significand,
        }
    }

    /// Serialize to 10-byte little-endian representation (raw 80-bit format).
    pub fn to_bytes(&self) -> [u8; 10] {
        let mut bytes = [0u8; 10];
        let sig_bytes = self.significand.to_le_bytes();
        bytes[..8].copy_from_slice(&sig_bytes);
        let exp_sign: u16 = ((self.sign as u16) << 15) | self.exponent;
        let es_bytes = exp_sign.to_le_bytes();
        bytes[8] = es_bytes[0];
        bytes[9] = es_bytes[1];
        bytes
    }

    /// Serialize to 16-byte padded representation (80-bit value + 6 zero-padding bytes).
    ///
    /// This matches the x86-64 ABI layout where `long double` is stored in 16 bytes
    /// (10 bytes of data followed by 6 bytes of padding) for 16-byte alignment.
    pub fn to_bytes_padded(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        let raw = self.to_bytes();
        bytes[..10].copy_from_slice(&raw);
        // Bytes 10..16 remain zero (padding)
        bytes
    }
}

/// Internal helper: convert an unsigned magnitude with a given sign to LongDouble.
/// The conversion is always exact since u64 fits within the 64-bit significand.
fn from_u64_inner(sign: bool, val: u64) -> LongDouble {
    debug_assert!(val != 0, "caller must handle zero");

    let lz = val.leading_zeros();
    // Shift left so the MSB of val lands at bit 63 (the integer bit position)
    let significand = val << lz;
    // True exponent: the MSB of `val` is at position (63 − lz), so the value is
    // significand × 2^((63 − lz) − 63) = significand × 2^(−lz).
    // We need: significand × 2^(biased_exp − BIAS − 63) = val
    //   → biased_exp = BIAS + 63 − lz
    let biased_exp = (BIAS + 63 - lz as i32) as u16;

    LongDouble {
        sign,
        exponent: biased_exp,
        significand,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Operator trait implementations: Add, Sub, Mul, Div, Neg
// ═══════════════════════════════════════════════════════════════════════════════

// --- Owned value implementations ---

impl Add for LongDouble {
    type Output = LongDouble;

    fn add(self, rhs: LongDouble) -> LongDouble {
        LongDouble::add(&self, &rhs)
    }
}

impl Sub for LongDouble {
    type Output = LongDouble;

    fn sub(self, rhs: LongDouble) -> LongDouble {
        LongDouble::sub(&self, &rhs)
    }
}

impl Mul for LongDouble {
    type Output = LongDouble;

    fn mul(self, rhs: LongDouble) -> LongDouble {
        LongDouble::mul(&self, &rhs)
    }
}

impl Div for LongDouble {
    type Output = LongDouble;

    fn div(self, rhs: LongDouble) -> LongDouble {
        LongDouble::div(&self, &rhs)
    }
}

impl Neg for LongDouble {
    type Output = LongDouble;

    fn neg(self) -> LongDouble {
        LongDouble::neg(&self)
    }
}

// --- Reference implementations for ergonomic use ---

impl<'a> Add for &'a LongDouble {
    type Output = LongDouble;

    fn add(self, rhs: &'a LongDouble) -> LongDouble {
        LongDouble::add(self, rhs)
    }
}

impl<'a> Sub for &'a LongDouble {
    type Output = LongDouble;

    fn sub(self, rhs: &'a LongDouble) -> LongDouble {
        LongDouble::sub(self, rhs)
    }
}

impl<'a> Mul for &'a LongDouble {
    type Output = LongDouble;

    fn mul(self, rhs: &'a LongDouble) -> LongDouble {
        LongDouble::mul(self, rhs)
    }
}

impl<'a> Div for &'a LongDouble {
    type Output = LongDouble;

    fn div(self, rhs: &'a LongDouble) -> LongDouble {
        LongDouble::div(self, rhs)
    }
}

impl Neg for &LongDouble {
    type Output = LongDouble;

    fn neg(self) -> LongDouble {
        LongDouble::neg(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Display and Debug
// ═══════════════════════════════════════════════════════════════════════════════

impl fmt::Debug for LongDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LongDouble {{ sign: {}, exp: 0x{:04X}, sig: 0x{:016X} }}",
            self.sign as u8, self.exponent, self.significand
        )
    }
}

impl fmt::Display for LongDouble {
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
        if self.is_zero() {
            return if self.sign {
                write!(f, "-0.0")
            } else {
                write!(f, "0.0")
            };
        }
        // Approximate decimal display via f64 conversion.
        // For exact display, a full decimal conversion algorithm would be needed,
        // but for compiler diagnostics this approximation is sufficient.
        // The sign is already encoded in the f64 value returned by to_f64().
        let approx = self.to_f64();
        write!(f, "{}", approx)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_constants() {
        assert!(LongDouble::ZERO.is_zero());
        assert!(!LongDouble::ZERO.sign);
        assert!(LongDouble::NEG_ZERO.is_zero());
        assert!(LongDouble::NEG_ZERO.sign);
    }

    #[test]
    fn test_infinity_constants() {
        assert!(LongDouble::INFINITY.is_infinity());
        assert!(!LongDouble::INFINITY.sign);
        assert!(LongDouble::NEG_INFINITY.is_infinity());
        assert!(LongDouble::NEG_INFINITY.sign);
    }

    #[test]
    fn test_nan_constant() {
        assert!(LongDouble::NAN.is_nan());
        assert!(!LongDouble::NAN.is_infinity());
        assert!(!LongDouble::NAN.is_zero());
    }

    #[test]
    fn test_one_constant() {
        assert!(!LongDouble::ONE.is_zero());
        assert!(!LongDouble::ONE.is_nan());
        assert!(!LongDouble::ONE.is_infinity());
        assert_eq!(LongDouble::ONE.exponent, 0x3FFF);
        assert_eq!(LongDouble::ONE.significand, INTEGER_BIT);
    }

    // ── Classification ──────────────────────────────────────────────────────

    #[test]
    fn test_classification_normal() {
        let val = LongDouble::from_f64(3.14);
        assert!(!val.is_nan());
        assert!(!val.is_infinity());
        assert!(!val.is_zero());
        assert!(val.is_finite());
    }

    // ── Integer conversions ─────────────────────────────────────────────────

    #[test]
    fn test_from_i64_basic() {
        let one = LongDouble::from_i64(1);
        assert_eq!(one.exponent, 0x3FFF);
        assert_eq!(one.significand, INTEGER_BIT);
        assert!(!one.sign);

        let neg_one = LongDouble::from_i64(-1);
        assert_eq!(neg_one.exponent, 0x3FFF);
        assert_eq!(neg_one.significand, INTEGER_BIT);
        assert!(neg_one.sign);
    }

    #[test]
    fn test_from_i64_zero() {
        let z = LongDouble::from_i64(0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_from_i64_large() {
        let val = LongDouble::from_i64(i64::MAX);
        let back = val.to_i64();
        assert_eq!(back, Some(i64::MAX));

        let val = LongDouble::from_i64(i64::MIN);
        let back = val.to_i64();
        assert_eq!(back, Some(i64::MIN));
    }

    #[test]
    fn test_from_u64() {
        let val = LongDouble::from_u64(42);
        assert!(!val.sign);
        assert_eq!(val.to_i64(), Some(42));

        let val = LongDouble::from_u64(u64::MAX);
        // u64::MAX doesn't fit in i64
        assert_eq!(val.to_i64(), None);
    }

    #[test]
    fn test_to_i64_nan_inf() {
        assert_eq!(LongDouble::NAN.to_i64(), None);
        assert_eq!(LongDouble::INFINITY.to_i64(), None);
        assert_eq!(LongDouble::NEG_INFINITY.to_i64(), None);
    }

    #[test]
    fn test_to_i64_fractional() {
        let val = LongDouble::from_f64(3.7);
        assert_eq!(val.to_i64(), Some(3)); // truncate toward zero

        let val = LongDouble::from_f64(-3.7);
        assert_eq!(val.to_i64(), Some(-3));
    }

    // ── f64 conversions ─────────────────────────────────────────────────────

    #[test]
    fn test_from_f64_special() {
        let z = LongDouble::from_f64(0.0);
        assert!(z.is_zero());
        assert!(!z.sign);

        let nz = LongDouble::from_f64(-0.0);
        assert!(nz.is_zero());
        assert!(nz.sign);

        let inf = LongDouble::from_f64(f64::INFINITY);
        assert!(inf.is_infinity());
        assert!(!inf.sign);

        let ninf = LongDouble::from_f64(f64::NEG_INFINITY);
        assert!(ninf.is_infinity());
        assert!(ninf.sign);

        let nan = LongDouble::from_f64(f64::NAN);
        assert!(nan.is_nan());
    }

    #[test]
    fn test_f64_roundtrip() {
        let values = [
            1.0f64,
            -1.0,
            0.5,
            -0.5,
            3.14159265358979,
            1e100,
            1e-100,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];
        for &v in &values {
            let ld = LongDouble::from_f64(v);
            let back = ld.to_f64();
            assert_eq!(
                v.to_bits(),
                back.to_bits(),
                "f64 roundtrip failed for {}",
                v
            );
        }
    }

    // ── Byte serialization ──────────────────────────────────────────────────

    #[test]
    fn test_bytes_roundtrip() {
        let values = [
            LongDouble::ZERO,
            LongDouble::NEG_ZERO,
            LongDouble::ONE,
            LongDouble::INFINITY,
            LongDouble::NEG_INFINITY,
            LongDouble::NAN,
            LongDouble::from_f64(3.14),
            LongDouble::from_i64(-42),
        ];
        for &val in &values {
            let bytes = val.to_bytes();
            let restored = LongDouble::from_bytes(bytes);
            assert_eq!(val.sign, restored.sign, "sign mismatch in bytes roundtrip");
            assert_eq!(
                val.exponent, restored.exponent,
                "exponent mismatch in bytes roundtrip"
            );
            assert_eq!(
                val.significand, restored.significand,
                "significand mismatch in bytes roundtrip"
            );
        }
    }

    #[test]
    fn test_to_bytes_padded_length() {
        let padded = LongDouble::ONE.to_bytes_padded();
        assert_eq!(padded.len(), 16);
        // Bytes 10-15 must be zero
        for &b in &padded[10..] {
            assert_eq!(b, 0);
        }
    }

    // ── Addition ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let one = LongDouble::ONE;
        let two = LongDouble::add(&one, &one);
        let expected = LongDouble::from_i64(2);
        assert_eq!(two.exponent, expected.exponent);
        assert_eq!(two.significand, expected.significand);
    }

    #[test]
    fn test_add_negative() {
        let one = LongDouble::ONE;
        let neg_one = LongDouble::neg(&one);
        let result = LongDouble::add(&one, &neg_one);
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_infinity() {
        let result = LongDouble::add(&LongDouble::INFINITY, &LongDouble::ONE);
        assert!(result.is_infinity());
        assert!(!result.sign);

        let result = LongDouble::add(&LongDouble::INFINITY, &LongDouble::NEG_INFINITY);
        assert!(result.is_nan());
    }

    // ── Subtraction ─────────────────────────────────────────────────────────

    #[test]
    fn test_sub_basic() {
        let three = LongDouble::from_i64(3);
        let two = LongDouble::from_i64(2);
        let result = LongDouble::sub(&three, &two);
        assert_eq!(result.to_i64(), Some(1));
    }

    // ── Multiplication ──────────────────────────────────────────────────────

    #[test]
    fn test_mul_basic() {
        let three = LongDouble::from_i64(3);
        let four = LongDouble::from_i64(4);
        let result = LongDouble::mul(&three, &four);
        assert_eq!(result.to_i64(), Some(12));
    }

    #[test]
    fn test_mul_by_zero() {
        let val = LongDouble::from_i64(42);
        let result = LongDouble::mul(&val, &LongDouble::ZERO);
        assert!(result.is_zero());
    }

    #[test]
    fn test_mul_infinity_zero() {
        let result = LongDouble::mul(&LongDouble::INFINITY, &LongDouble::ZERO);
        assert!(result.is_nan());
    }

    #[test]
    fn test_mul_signs() {
        let pos = LongDouble::from_i64(3);
        let neg = LongDouble::from_i64(-4);
        let result = LongDouble::mul(&pos, &neg);
        assert_eq!(result.to_i64(), Some(-12));
        assert!(result.sign);
    }

    // ── Division ────────────────────────────────────────────────────────────

    #[test]
    fn test_div_basic() {
        let twelve = LongDouble::from_i64(12);
        let four = LongDouble::from_i64(4);
        let result = LongDouble::div(&twelve, &four);
        assert_eq!(result.to_i64(), Some(3));
    }

    #[test]
    fn test_div_by_zero() {
        let val = LongDouble::from_i64(1);
        let result = LongDouble::div(&val, &LongDouble::ZERO);
        assert!(result.is_infinity());
        assert!(!result.sign);
    }

    #[test]
    fn test_div_zero_by_zero() {
        let result = LongDouble::div(&LongDouble::ZERO, &LongDouble::ZERO);
        assert!(result.is_nan());
    }

    #[test]
    fn test_div_fractional() {
        let one = LongDouble::from_i64(1);
        let three = LongDouble::from_i64(3);
        let result = LongDouble::div(&one, &three);
        // 1/3 ≈ 0.333... — should not be zero, should be positive
        assert!(!result.is_zero());
        assert!(!result.sign);
        let f = result.to_f64();
        assert!((f - 1.0 / 3.0).abs() < 1e-15);
    }

    // ── Negation and Abs ────────────────────────────────────────────────────

    #[test]
    fn test_neg() {
        let val = LongDouble::from_i64(42);
        let negated = LongDouble::neg(&val);
        assert!(negated.sign);
        assert_eq!(negated.to_i64(), Some(-42));
    }

    #[test]
    fn test_abs() {
        let neg = LongDouble::from_i64(-42);
        let a = LongDouble::abs(&neg);
        assert!(!a.sign);
        assert_eq!(a.to_i64(), Some(42));
    }

    // ── Comparison ──────────────────────────────────────────────────────────

    #[test]
    fn test_partial_eq_basic() {
        let a = LongDouble::from_i64(42);
        let b = LongDouble::from_i64(42);
        assert_eq!(a, b);

        let c = LongDouble::from_i64(43);
        assert_ne!(a, c);
    }

    #[test]
    fn test_nan_not_equal_to_self() {
        let nan = LongDouble::NAN;
        assert_ne!(nan, nan);
    }

    #[test]
    fn test_zero_equality() {
        assert_eq!(LongDouble::ZERO, LongDouble::NEG_ZERO);
    }

    #[test]
    fn test_partial_ord() {
        let one = LongDouble::ONE;
        let two = LongDouble::from_i64(2);
        assert!(one < two);
        assert!(two > one);

        let neg = LongDouble::from_i64(-5);
        assert!(neg < one);
    }

    #[test]
    fn test_nan_unordered() {
        let nan = LongDouble::NAN;
        let one = LongDouble::ONE;
        assert_eq!(nan.partial_cmp(&one), None);
        assert_eq!(one.partial_cmp(&nan), None);
        assert_eq!(nan.partial_cmp(&nan), None);
    }

    // ── Operator traits ─────────────────────────────────────────────────────

    #[test]
    fn test_operator_add() {
        let a = LongDouble::from_i64(10);
        let b = LongDouble::from_i64(20);
        let result = a + b;
        assert_eq!(result.to_i64(), Some(30));
    }

    #[test]
    fn test_operator_sub() {
        let a = LongDouble::from_i64(30);
        let b = LongDouble::from_i64(20);
        let result = a - b;
        assert_eq!(result.to_i64(), Some(10));
    }

    #[test]
    fn test_operator_mul() {
        let a = LongDouble::from_i64(6);
        let b = LongDouble::from_i64(7);
        let result = a * b;
        assert_eq!(result.to_i64(), Some(42));
    }

    #[test]
    fn test_operator_div() {
        let a = LongDouble::from_i64(100);
        let b = LongDouble::from_i64(5);
        let result = a / b;
        assert_eq!(result.to_i64(), Some(20));
    }

    #[test]
    fn test_operator_neg() {
        let a = LongDouble::from_i64(42);
        let result = -a;
        assert!(result.sign);
        assert_eq!(result.to_i64(), Some(-42));
    }

    #[test]
    fn test_ref_operators() {
        let a = LongDouble::from_i64(5);
        let b = LongDouble::from_i64(3);
        let sum = &a + &b;
        assert_eq!(sum.to_i64(), Some(8));
        let diff = &a - &b;
        assert_eq!(diff.to_i64(), Some(2));
        let prod = &a * &b;
        assert_eq!(prod.to_i64(), Some(15));
        let quot = &a / &b;
        assert!(!quot.is_zero());
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_large_integer_roundtrip() {
        // Test powers of 2 that fit exactly
        for shift in 0..63u32 {
            let val: i64 = 1i64 << shift;
            let ld = LongDouble::from_i64(val);
            assert_eq!(ld.to_i64(), Some(val), "roundtrip failed for 2^{}", shift);
        }
    }

    #[test]
    fn test_mul_large_values() {
        let a = LongDouble::from_i64(1_000_000);
        let b = LongDouble::from_i64(1_000_000);
        let result = LongDouble::mul(&a, &b);
        assert_eq!(result.to_i64(), Some(1_000_000_000_000));
    }

    #[test]
    fn test_display_special() {
        assert_eq!(format!("{}", LongDouble::NAN), "NaN");
        assert_eq!(format!("{}", LongDouble::INFINITY), "inf");
        assert_eq!(format!("{}", LongDouble::NEG_INFINITY), "-inf");
        assert_eq!(format!("{}", LongDouble::ZERO), "0.0");
        assert_eq!(format!("{}", LongDouble::NEG_ZERO), "-0.0");
    }

    #[test]
    fn test_debug_format() {
        let dbg = format!("{:?}", LongDouble::ONE);
        assert!(dbg.contains("LongDouble"));
        assert!(dbg.contains("3FFF"));
    }

    #[test]
    fn test_sub_equal_values() {
        let a = LongDouble::from_f64(1.5);
        let b = LongDouble::from_f64(1.5);
        let result = LongDouble::sub(&a, &b);
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_different_magnitudes() {
        // Add a very large and very small number
        let big = LongDouble::from_f64(1e18);
        let small = LongDouble::from_f64(1.0);
        let result = LongDouble::add(&big, &small);
        // The small value may be lost due to precision, result ≈ big
        let back = result.to_f64();
        assert!((back - 1e18).abs() <= 1.0);
    }
}
