//! Builder-pattern API for constructing complex C and machine types.
//!
//! This module provides [`TypeBuilder`], a target-aware factory for
//! constructing `CType` values (pointers, arrays, functions, qualified types),
//! computing struct/union memory layouts with `__attribute__((packed))` and
//! `__attribute__((aligned(N)))` support, and querying type sizes and
//! alignments for any of the four supported architectures.
//!
//! Additionally, free functions are provided for common type classification
//! tasks required by the semantic analyser and IR lowering phases:
//! [`is_integer_type`], [`is_arithmetic_type`], [`is_scalar_type`],
//! [`is_complete_type`], [`integer_rank`], and [`usual_arithmetic_conversion`].
//!
//! # Zero-Dependency Mandate
//!
//! This module uses only `std` and `crate::` references. No external crates.
//!
//! # Architecture-Dependent Behaviour
//!
//! All size and alignment computations are parameterised by the stored
//! [`Target`] so that pointer widths (4 vs 8), `long` sizes (4 vs 8),
//! and `long double` sizes/alignments are correct for both ILP32 (i686)
//! and LP64 (x86-64, AArch64, RISC-V 64) data models.

use crate::common::target::Target;
use crate::common::types::{alignof_ctype, sizeof_ctype, CType, StructField, TypeQualifiers};

// ===========================================================================
// FieldLayout
// ===========================================================================

/// Layout information for a single field within a struct or union.
///
/// Computed by [`TypeBuilder::compute_struct_layout`] or
/// [`TypeBuilder::compute_union_layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout {
    /// Byte offset from the beginning of the containing aggregate.
    pub offset: usize,
    /// Size of the field in bytes.
    pub size: usize,
    /// Natural alignment of the field in bytes.
    pub alignment: usize,
    /// For bitfield members, contains `Some((bit_offset_in_unit, bit_width))`
    /// where `bit_offset_in_unit` is the bit position of this field within
    /// its storage unit (starting from bit 0 = LSB on little-endian),
    /// and `bit_width` is the number of bits in the bitfield.
    /// `None` for regular (non-bitfield) fields.
    pub bitfield_info: Option<(usize, usize)>,
}

// ===========================================================================
// StructLayout
// ===========================================================================

/// Complete layout information for a struct or union.
///
/// Contains per-field layout entries plus the overall size and alignment of
/// the aggregate. The `has_flexible_array` flag indicates whether the last
/// field is a flexible array member (zero-length array).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    /// Per-field layout entries in declaration order.
    pub fields: Vec<FieldLayout>,
    /// Total size of the aggregate in bytes (including tail padding).
    pub size: usize,
    /// Overall alignment requirement of the aggregate in bytes.
    pub alignment: usize,
    /// `true` if the last field is a flexible array member.
    pub has_flexible_array: bool,
}

// ===========================================================================
// TypeBuilder
// ===========================================================================

/// Target-aware builder for constructing C types and computing layouts.
///
/// The builder stores a [`Target`] reference to provide architecture-dependent
/// type sizes and alignments. All construction methods return new `CType`
/// values without mutating the builder, allowing it to be reused across the
/// entire compilation of a translation unit.
///
/// # Examples (conceptual)
///
/// ```ignore
/// let tb = TypeBuilder::new(Target::X86_64);
/// let ptr_int = tb.pointer_to(CType::Int);
/// let arr = tb.array_of(CType::Double, Some(10));
/// ```
pub struct TypeBuilder {
    /// The target architecture for size/alignment computations.
    target: Target,
}

impl TypeBuilder {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new `TypeBuilder` bound to the given target architecture.
    ///
    /// The target determines pointer widths, `long` sizes, `long double`
    /// sizes, and alignment rules for all subsequent operations.
    #[inline]
    pub fn new(target: Target) -> Self {
        TypeBuilder { target }
    }

    /// Return a reference to the stored target.
    #[inline]
    pub fn target(&self) -> &Target {
        &self.target
    }

    // -----------------------------------------------------------------------
    // Type construction helpers
    // -----------------------------------------------------------------------

    /// Construct a pointer type pointing to `pointee`.
    ///
    /// The resulting pointer has no qualifiers. Use [`const_qualified`] or
    /// [`volatile_qualified`] on the result to add pointer-level qualifiers.
    #[inline]
    pub fn pointer_to(&self, pointee: CType) -> CType {
        CType::Pointer(Box::new(pointee), TypeQualifiers::default())
    }

    /// Construct an array type of `element` with the given `size`.
    ///
    /// - `Some(n)` — fixed-size array of `n` elements.
    /// - `None` — incomplete array / flexible array member.
    #[inline]
    pub fn array_of(&self, element: CType, size: Option<usize>) -> CType {
        CType::Array(Box::new(element), size)
    }

    /// Construct a function type with the given return type, parameter types,
    /// and variadic flag.
    #[inline]
    pub fn function_type(&self, return_type: CType, params: Vec<CType>, variadic: bool) -> CType {
        CType::Function {
            return_type: Box::new(return_type),
            params,
            variadic,
        }
    }

    /// Wrap `ty` with a `const` qualifier.
    ///
    /// If `ty` is already `Qualified`, the `const` flag is merged into the
    /// existing qualifier set rather than double-wrapping.
    pub fn const_qualified(&self, ty: CType) -> CType {
        match ty {
            CType::Qualified(inner, mut quals) => {
                quals.is_const = true;
                CType::Qualified(inner, quals)
            }
            other => {
                let quals = TypeQualifiers {
                    is_const: true,
                    ..TypeQualifiers::default()
                };
                CType::Qualified(Box::new(other), quals)
            }
        }
    }

    /// Wrap `ty` with a `volatile` qualifier.
    ///
    /// If `ty` is already `Qualified`, the `volatile` flag is merged into
    /// the existing qualifier set rather than double-wrapping.
    pub fn volatile_qualified(&self, ty: CType) -> CType {
        match ty {
            CType::Qualified(inner, mut quals) => {
                quals.is_volatile = true;
                CType::Qualified(inner, quals)
            }
            other => {
                let quals = TypeQualifiers {
                    is_volatile: true,
                    ..TypeQualifiers::default()
                };
                CType::Qualified(Box::new(other), quals)
            }
        }
    }

    /// Wrap `ty` with `_Atomic` semantics.
    ///
    /// This produces a `CType::Atomic` wrapper which indicates atomic
    /// storage/access semantics at the type system level.
    #[inline]
    pub fn atomic_type(&self, ty: CType) -> CType {
        CType::Atomic(Box::new(ty))
    }

    // -----------------------------------------------------------------------
    // Struct layout computation
    // -----------------------------------------------------------------------

    /// Compute the memory layout of a struct from its field types.
    ///
    /// Fields are laid out sequentially with alignment padding inserted
    /// between fields (unless `packed` is true). After all fields are placed,
    /// tail padding is added to make the total size a multiple of the struct's
    /// alignment.
    ///
    /// # Packed structs
    ///
    /// When `packed` is `true`, every field is placed at alignment 1 — no
    /// inter-field padding is inserted. This matches the behaviour of
    /// `__attribute__((packed))`.
    ///
    /// # Explicit alignment
    ///
    /// When `explicit_align` is `Some(N)`, the overall struct alignment is
    /// raised to at least `N` bytes (matching `__attribute__((aligned(N)))`).
    /// The natural alignment (max of field alignments) is still computed and
    /// the larger of the two is used.
    ///
    /// # Flexible array members
    ///
    /// If the last field is an array with `None` size (i.e., a
    /// `CType::Array(_, None)`), it is treated as a flexible array member:
    /// it contributes 0 bytes to the struct size but is recorded in the
    /// layout. The returned `StructLayout::has_flexible_array` is set to
    /// `true`.
    pub fn compute_struct_layout(
        &self,
        fields: &[CType],
        packed: bool,
        explicit_align: Option<usize>,
    ) -> StructLayout {
        let mut field_layouts = Vec::with_capacity(fields.len());
        let mut offset: usize = 0;
        let mut max_field_align: usize = 1;
        let mut has_flexible_array = false;

        for (i, field_ty) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;

            // Detect flexible array member: last field is an incomplete array.
            let is_flex = is_last && matches!(field_ty, CType::Array(_, None));
            if is_flex {
                has_flexible_array = true;
            }

            let field_size = if is_flex {
                0 // Flexible array members contribute 0 to struct size.
            } else {
                sizeof_ctype(field_ty, &self.target)
            };

            let field_align = if packed {
                1
            } else {
                alignof_ctype(field_ty, &self.target)
            };

            // Align current offset to the field's alignment requirement.
            offset = align_up(offset, field_align);

            field_layouts.push(FieldLayout {
                offset,
                size: field_size,
                alignment: field_align,
                bitfield_info: None,
            });

            // Advance past this field (flex contributes 0).
            offset += field_size;

            if field_align > max_field_align {
                max_field_align = field_align;
            }
        }

        // Determine overall struct alignment.
        let struct_align = compute_aggregate_alignment(max_field_align, packed, explicit_align);

        // Pad total size to a multiple of the struct alignment.
        let total_size = align_up(offset, struct_align);

        StructLayout {
            fields: field_layouts,
            size: total_size,
            alignment: struct_align,
            has_flexible_array,
        }
    }

    /// Compute the memory layout of a struct from [`StructField`] entries,
    /// including bitfield support.
    ///
    /// This is the full-featured layout computation that handles:
    /// - Regular fields with natural alignment.
    /// - Bitfields with bit-level offset tracking within allocation units.
    /// - Zero-width bitfields that force alignment to the next allocation
    ///   unit boundary.
    /// - Packed structs (`__attribute__((packed))`).
    /// - Explicit alignment overrides (`__attribute__((aligned(N)))`).
    /// - Flexible array members (zero-length array as the last field).
    pub fn compute_struct_layout_with_fields(
        &self,
        fields: &[StructField],
        packed: bool,
        explicit_align: Option<usize>,
    ) -> StructLayout {
        let mut field_layouts = Vec::with_capacity(fields.len());
        // Track position as absolute bit offset from the struct start.
        // Use u128 to avoid overflow on huge structs (e.g. arrays with
        // (1<<62) elements where byte_size * 8 would overflow usize).
        let mut abs_bit: u128 = 0;
        let mut max_field_align: usize = 1;
        let mut has_flexible_array = false;

        for (i, field) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;
            let is_flex = is_last && matches!(field.ty, CType::Array(_, None));
            if is_flex {
                has_flexible_array = true;
            }

            // Helper: align a u128 value up to an alignment boundary.
            let align_u128 = |val: u128, a: usize| -> u128 {
                if a <= 1 {
                    return val;
                }
                let a = a as u128;
                (val + a - 1) / a * a
            };

            if let Some(bits) = field.bit_width {
                let bits = bits as u128;
                // ----- Bitfield handling -----
                let unit_type_size = sizeof_ctype(&field.ty, &self.target);
                let unit_type_align = if packed {
                    1
                } else {
                    alignof_ctype(&field.ty, &self.target)
                };
                let unit_size_bits = (unit_type_size as u128) * 8;

                if bits == 0 {
                    // Zero-width bitfield: pad to next alignment boundary
                    // of the underlying type, then close the current
                    // storage unit.
                    let align_bits = unit_type_align * 8;
                    if align_bits > 0 {
                        abs_bit = align_u128(abs_bit, align_bits);
                    }
                    if unit_type_align > max_field_align {
                        max_field_align = unit_type_align;
                    }
                    let byte_off = (abs_bit / 8) as usize;
                    field_layouts.push(FieldLayout {
                        offset: byte_off,
                        size: 0,
                        alignment: unit_type_align,
                        bitfield_info: Some((0, 0)),
                    });
                    continue;
                }

                // Determine the natural-alignment region that contains the
                // current bit position.
                if packed {
                    // Packed structs: bitfields are laid out contiguously
                    // with no padding, even when spanning storage unit
                    // boundaries.  Use byte-level alignment (align=1).
                    let bf_byte = (abs_bit / 8) as usize;
                    let bit_offset_in_unit = (abs_bit % 8) as usize;

                    // Determine the access size: must cover all bits from
                    // the start bit within the byte through the last bit.
                    let needed_bytes = ((bit_offset_in_unit + bits as usize) + 7) / 8;
                    // Round up to the next power-of-two load size (1/2/4/8).
                    let access_size = if needed_bytes <= 1 {
                        1
                    } else if needed_bytes <= 2 {
                        2
                    } else if needed_bytes <= 4 {
                        4
                    } else {
                        8
                    };

                    field_layouts.push(FieldLayout {
                        offset: bf_byte,
                        size: access_size,
                        alignment: 1,
                        bitfield_info: Some((bit_offset_in_unit, bits as usize)),
                    });

                    abs_bit += bits;
                } else {
                    let unit_align_bits: u128 = (unit_type_align as u128) * 8;
                    let unit_start = if unit_align_bits > 0 {
                        (abs_bit / unit_align_bits) * unit_align_bits
                    } else {
                        abs_bit
                    };
                    let bits_used_in_unit = abs_bit - unit_start;

                    if bits_used_in_unit + bits > unit_size_bits {
                        // Doesn't fit in the current storage unit — advance to
                        // the next unit boundary.
                        abs_bit = unit_start + unit_size_bits;
                        // Re-align for the new unit.
                        let byte_offset = (abs_bit + 7) / 8;
                        let aligned_byte = align_u128(byte_offset, unit_type_align);
                        abs_bit = aligned_byte * 8;
                    }

                    // The byte offset of the storage unit containing this
                    // bitfield.  Compute as the aligned-down byte position.
                    let mask = if unit_type_align > 0 {
                        !((unit_type_align as u128) - 1)
                    } else {
                        !0u128
                    };
                    let bf_byte = ((abs_bit / 8) & mask) as usize;

                    // Compute the bit offset within the storage unit.
                    let bit_offset_in_unit = (abs_bit - (bf_byte as u128) * 8) as usize;

                    field_layouts.push(FieldLayout {
                        offset: bf_byte,
                        size: unit_type_size,
                        alignment: unit_type_align,
                        bitfield_info: Some((bit_offset_in_unit, bits as usize)),
                    });

                    abs_bit += bits;
                }

                if unit_type_align > max_field_align {
                    max_field_align = unit_type_align;
                }
            } else {
                // ----- Regular (non-bitfield) field -----
                // Round up to byte boundary (close any partial bitfield
                // byte).
                let byte_offset = (abs_bit + 7) / 8;

                let field_size = if is_flex {
                    0
                } else {
                    sizeof_ctype(&field.ty, &self.target)
                };

                let field_align = if packed {
                    1
                } else {
                    alignof_ctype(&field.ty, &self.target)
                };

                let aligned_byte = align_u128(byte_offset, field_align) as usize;

                field_layouts.push(FieldLayout {
                    offset: aligned_byte,
                    size: field_size,
                    alignment: field_align,
                    bitfield_info: None,
                });

                abs_bit = ((aligned_byte as u128) + (field_size as u128)) * 8;

                if field_align > max_field_align {
                    max_field_align = field_align;
                }
            }
        }

        // Convert final bit offset to bytes (round up).
        let final_byte = ((abs_bit + 7) / 8) as usize;

        let struct_align = compute_aggregate_alignment(max_field_align, packed, explicit_align);
        let total_size = align_up(final_byte, struct_align);

        StructLayout {
            fields: field_layouts,
            size: total_size,
            alignment: struct_align,
            has_flexible_array,
        }
    }

    // -----------------------------------------------------------------------
    // Union layout computation
    // -----------------------------------------------------------------------

    /// Compute the memory layout of a union from its field types.
    ///
    /// In a union, all fields start at offset 0. The union's size is the
    /// size of the largest field, padded to the union's alignment.
    ///
    /// # Packed unions
    ///
    /// When `packed` is `true`, field alignment is forced to 1. The union
    /// size is simply the max field size (no alignment padding).
    ///
    /// # Explicit alignment
    ///
    /// `explicit_align` (from `__attribute__((aligned(N)))`) raises the
    /// overall union alignment.
    pub fn compute_union_layout(
        &self,
        fields: &[CType],
        packed: bool,
        explicit_align: Option<usize>,
    ) -> StructLayout {
        let mut field_layouts = Vec::with_capacity(fields.len());
        let mut max_size: usize = 0;
        let mut max_field_align: usize = 1;
        let has_flexible_array = false;

        for field_ty in fields.iter() {
            let field_size = sizeof_ctype(field_ty, &self.target);
            let field_align = if packed {
                1
            } else {
                alignof_ctype(field_ty, &self.target)
            };

            field_layouts.push(FieldLayout {
                offset: 0, // All union fields start at offset 0.
                size: field_size,
                alignment: field_align,
                bitfield_info: None,
            });

            if field_size > max_size {
                max_size = field_size;
            }
            if field_align > max_field_align {
                max_field_align = field_align;
            }
        }

        let union_align = compute_aggregate_alignment(max_field_align, packed, explicit_align);
        let total_size = align_up(max_size, union_align);

        StructLayout {
            fields: field_layouts,
            size: total_size,
            alignment: union_align,
            has_flexible_array,
        }
    }

    // -----------------------------------------------------------------------
    // Size and alignment queries
    // -----------------------------------------------------------------------

    /// Return the size in bytes of `ty` on this builder's target architecture.
    ///
    /// Delegates to [`sizeof_ctype`] with the stored target. This is
    /// the primary entry point for size queries throughout the pipeline.
    #[inline]
    pub fn sizeof_type(&self, ty: &CType) -> usize {
        sizeof_ctype(ty, &self.target)
    }

    /// Return the alignment in bytes of `ty` on this builder's target
    /// architecture.
    ///
    /// Delegates to [`alignof_ctype`] with the stored target.
    #[inline]
    pub fn alignof_type(&self, ty: &CType) -> usize {
        alignof_ctype(ty, &self.target)
    }
}

// ===========================================================================
// Helper functions (private)
// ===========================================================================

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1). If `align` is 0, returns `value`
/// unchanged.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

/// Compute the overall alignment for a struct or union given the maximum
/// natural field alignment, packed flag, and optional explicit alignment.
///
/// - If `packed` is true and no explicit alignment is given, alignment is 1.
/// - If `explicit_align` is `Some(N)`, the alignment is `max(natural, N)`.
/// - Otherwise, the alignment equals the natural maximum field alignment.
#[inline]
fn compute_aggregate_alignment(
    max_natural_field_align: usize,
    packed: bool,
    explicit_align: Option<usize>,
) -> usize {
    let natural = if packed { 1 } else { max_natural_field_align };
    match explicit_align {
        Some(a) => a.max(natural),
        None => natural,
    }
}

/// Recursively strip `Qualified`, `Typedef`, and `Atomic` wrappers to
/// reach the underlying undecorated type. Used by type predicate functions.
fn resolve_and_strip(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => resolve_and_strip(inner),
        CType::Typedef { underlying, .. } => resolve_and_strip(underlying),
        CType::Atomic(inner) => resolve_and_strip(inner),
        other => other,
    }
}

// ===========================================================================
// Type predicate free functions
// ===========================================================================

/// Returns `true` if `ty` is an integer type (including `_Bool`, `char`,
/// and `enum`), after stripping qualifiers, typedefs, and atomic wrappers.
///
/// This follows C11 §6.2.5 ¶17 which defines the integer types.
pub fn is_integer_type(ty: &CType) -> bool {
    matches!(
        resolve_and_strip(ty),
        CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong
            | CType::Enum { .. }
    )
}

/// Returns `true` if `ty` is an arithmetic type (integer or floating-point).
///
/// C11 §6.2.5 ¶18: "Integer and floating types are collectively called
/// arithmetic types."
pub fn is_arithmetic_type(ty: &CType) -> bool {
    let resolved = resolve_and_strip(ty);
    is_integer_type_inner(resolved)
        || matches!(resolved, CType::Float | CType::Double | CType::LongDouble)
}

/// Returns `true` if `ty` is a scalar type (arithmetic or pointer).
///
/// C11 §6.2.5 ¶21: "Arithmetic types and pointer types are collectively
/// called scalar types."
pub fn is_scalar_type(ty: &CType) -> bool {
    let resolved = resolve_and_strip(ty);
    is_integer_type_inner(resolved)
        || matches!(
            resolved,
            CType::Float | CType::Double | CType::LongDouble | CType::Pointer(_, _)
        )
}

/// Returns `true` if `ty` is a complete type — i.e. it has a known,
/// finite size.
///
/// Incomplete types include:
/// - `void`
/// - Arrays with unknown element count (`CType::Array(_, None)`)
/// - Forward-declared structs/unions (empty fields list used as heuristic)
pub fn is_complete_type(ty: &CType) -> bool {
    match resolve_and_strip(ty) {
        CType::Void => false,
        CType::Array(_, None) => false,
        // Forward-declared structs/unions with empty fields are incomplete.
        // However, GCC extension allows empty structs, so we treat them as
        // complete and rely on the semantic analyser to distinguish.
        CType::Struct { .. } | CType::Union { .. } => true,
        CType::Function { .. } => true,
        _ => true,
    }
}

/// Return the integer conversion rank of `ty` per C11 §6.3.1.1.
///
/// Higher rank means higher precedence in the usual arithmetic conversions.
///
/// | Rank | Types                                       |
/// |------|---------------------------------------------|
/// | 1    | `_Bool`                                     |
/// | 2    | `char`, `signed char`, `unsigned char`       |
/// | 3    | `short`, `unsigned short`                   |
/// | 4    | `int`, `unsigned int`                       |
/// | 5    | `long`, `unsigned long`                     |
/// | 6    | `long long`, `unsigned long long`           |
///
/// Returns `0` for non-integer types.
/// Return the bit-width of a promoted integer type on the LP64 data model
/// used by x86-64, aarch64, and riscv64 (and ILP32 for i686 where long=32).
///
/// This is used by `usual_arithmetic_conversion` step 4d to determine whether
/// a signed type can actually represent all values of an unsigned type — which
/// requires the signed type to have strictly more bits, not just higher rank.
fn integer_bit_width(ty: &CType) -> u32 {
    let resolved = resolve_and_strip(ty);
    match resolved {
        CType::Bool => 1,
        CType::Char | CType::SChar | CType::UChar => 8,
        CType::Short | CType::UShort => 16,
        CType::Int | CType::UInt => 32,
        // LP64: long = 64-bit, long long = 64-bit (both have same width)
        CType::Long | CType::ULong => 64,
        CType::LongLong | CType::ULongLong => 64,
        CType::Enum {
            underlying_type, ..
        } => integer_bit_width(underlying_type),
        _ => 0,
    }
}

pub fn integer_rank(ty: &CType) -> u8 {
    let resolved = resolve_and_strip(ty);
    match resolved {
        CType::Bool => 1,
        CType::Char | CType::SChar | CType::UChar => 2,
        CType::Short | CType::UShort => 3,
        CType::Int | CType::UInt => 4,
        CType::Long | CType::ULong => 5,
        CType::LongLong | CType::ULongLong => 6,
        CType::Enum {
            underlying_type, ..
        } => integer_rank(underlying_type),
        _ => 0,
    }
}

/// Perform the C11 §6.3.1.8 "usual arithmetic conversions" to determine
/// the common type resulting from a binary operation on `lhs` and `rhs`.
///
/// The algorithm:
/// 1. If either operand is `long double`, the result is `long double`.
/// 2. Else if either is `double`, the result is `double`.
/// 3. Else if either is `float`, the result is `float`.
/// 4. Otherwise, integer promotions are applied to both operands, then:
///    a. If both have the same type after promotion, that type is the result.
///    b. If both are signed or both are unsigned, the one with lower rank
///    is converted to the type with higher rank.
///    c. If the unsigned type has rank ≥ the signed type's rank, the signed
///    is converted to unsigned.
///    d. If the signed type can represent all values of the unsigned type,
///    the unsigned is converted to signed.
///    e. Otherwise, both are converted to the unsigned counterpart of the
///    signed type.
pub fn usual_arithmetic_conversion(lhs: &CType, rhs: &CType) -> CType {
    let left = resolve_and_strip(lhs);
    let right = resolve_and_strip(rhs);

    // Step 1: long double dominates.
    if matches!(left, CType::LongDouble) || matches!(right, CType::LongDouble) {
        return CType::LongDouble;
    }

    // Step 2: double dominates.
    if matches!(left, CType::Double) || matches!(right, CType::Double) {
        return CType::Double;
    }

    // Step 3: float dominates.
    if matches!(left, CType::Float) || matches!(right, CType::Float) {
        return CType::Float;
    }

    // Step 4: integer promotions.
    let promoted_l = integer_promote(left);
    let promoted_r = integer_promote(right);

    // 4a. If both have the same type, done.
    if types_equal_for_conversion(&promoted_l, &promoted_r) {
        return promoted_l;
    }

    let l_unsigned = is_unsigned_type(&promoted_l);
    let r_unsigned = is_unsigned_type(&promoted_r);
    let l_rank = integer_rank(&promoted_l);
    let r_rank = integer_rank(&promoted_r);

    // 4b. Both signed or both unsigned — convert to the one with higher rank.
    if l_unsigned == r_unsigned {
        return if l_rank >= r_rank {
            promoted_l
        } else {
            promoted_r
        };
    }

    // One is signed, the other unsigned.
    let (signed_ty, signed_rank, unsigned_ty, unsigned_rank) = if l_unsigned {
        (&promoted_r, r_rank, &promoted_l, l_rank)
    } else {
        (&promoted_l, l_rank, &promoted_r, r_rank)
    };

    // 4c. If unsigned rank >= signed rank, convert to unsigned.
    if unsigned_rank >= signed_rank {
        return unsigned_ty.clone();
    }

    // 4d. If signed type can represent all values of unsigned type, use signed.
    // This is only true when the signed type has strictly MORE bits than the
    // unsigned type.  On LP64, `long` and `long long` are both 64-bit, so
    // `long long` (rank 6) vs `unsigned long` (rank 5): rank differs but
    // bit-width is the same → signed long long CANNOT represent all unsigned
    // long values.  We must fall through to 4e in that case.
    if signed_rank > unsigned_rank && integer_bit_width(signed_ty) > integer_bit_width(unsigned_ty)
    {
        return signed_ty.clone();
    }

    // 4e. Otherwise, convert both to the unsigned counterpart of the signed type.
    to_unsigned(signed_ty)
}

// ===========================================================================
// Private helpers for usual_arithmetic_conversion
// ===========================================================================

/// Apply C11 §6.3.1.1 integer promotions.
///
/// Types with rank below `int` are promoted to `int`. All other types
/// are returned unchanged.
pub fn integer_promote(ty: &CType) -> CType {
    // Resolve through typedefs, qualifiers, and _Atomic before matching,
    // so that e.g. `typedef unsigned char u8` is correctly promoted to int.
    let resolved = resolve_and_strip(ty);
    match resolved {
        CType::Bool | CType::Char | CType::SChar | CType::Short => CType::Int,
        CType::UChar | CType::UShort => {
            // unsigned char and unsigned short fit in int on all targets
            // (int is always 32-bit).
            CType::Int
        }
        CType::Enum {
            underlying_type, ..
        } => integer_promote(underlying_type),
        other => other.clone(),
    }
}

/// Check if two types are the "same type" for conversion purposes.
///
/// This performs a shallow structural comparison after resolution, which is
/// sufficient for the usual arithmetic conversion algorithm.
fn types_equal_for_conversion(a: &CType, b: &CType) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

/// Returns `true` if `ty` is an unsigned integer type.
fn is_unsigned_type(ty: &CType) -> bool {
    let resolved = resolve_and_strip(ty);
    matches!(
        resolved,
        CType::Bool | CType::UChar | CType::UShort | CType::UInt | CType::ULong | CType::ULongLong
    )
}

/// Convert a signed integer type to its unsigned counterpart.
///
/// Used in step 4e of the usual arithmetic conversions.
fn to_unsigned(ty: &CType) -> CType {
    match ty {
        CType::Char => CType::UChar,
        CType::Short => CType::UShort,
        CType::Int => CType::UInt,
        CType::Long => CType::ULong,
        CType::LongLong => CType::ULongLong,
        // Already unsigned or non-integer — return as-is.
        other => other.clone(),
    }
}

/// Inner integer type check (operates on already-resolved type).
fn is_integer_type_inner(ty: &CType) -> bool {
    matches!(
        ty,
        CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort
            | CType::Int
            | CType::UInt
            | CType::Long
            | CType::ULong
            | CType::LongLong
            | CType::ULongLong
            | CType::Enum { .. }
    )
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::target::Target;
    use crate::common::types::{CType, StructField, TypeQualifiers};

    // -----------------------------------------------------------------------
    // TypeBuilder construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_all_targets() {
        for target in &[
            Target::X86_64,
            Target::I686,
            Target::AArch64,
            Target::RiscV64,
        ] {
            let tb = TypeBuilder::new(*target);
            assert_eq!(*tb.target(), *target);
        }
    }

    // -----------------------------------------------------------------------
    // pointer_to
    // -----------------------------------------------------------------------

    #[test]
    fn test_pointer_to_int() {
        let tb = TypeBuilder::new(Target::X86_64);
        let ptr = tb.pointer_to(CType::Int);
        match &ptr {
            CType::Pointer(inner, quals) => {
                assert_eq!(**inner, CType::Int);
                assert!(quals.is_empty());
            }
            _ => panic!("Expected Pointer type"),
        }
    }

    #[test]
    fn test_pointer_to_void() {
        let tb = TypeBuilder::new(Target::I686);
        let ptr = tb.pointer_to(CType::Void);
        match &ptr {
            CType::Pointer(inner, _) => assert_eq!(**inner, CType::Void),
            _ => panic!("Expected Pointer type"),
        }
    }

    // -----------------------------------------------------------------------
    // array_of
    // -----------------------------------------------------------------------

    #[test]
    fn test_array_of_fixed() {
        let tb = TypeBuilder::new(Target::X86_64);
        let arr = tb.array_of(CType::Int, Some(10));
        match &arr {
            CType::Array(elem, Some(10)) => assert_eq!(**elem, CType::Int),
            _ => panic!("Expected Array type with size 10"),
        }
    }

    #[test]
    fn test_array_of_flexible() {
        let tb = TypeBuilder::new(Target::X86_64);
        let arr = tb.array_of(CType::Char, None);
        match &arr {
            CType::Array(elem, None) => assert_eq!(**elem, CType::Char),
            _ => panic!("Expected incomplete Array type"),
        }
    }

    // -----------------------------------------------------------------------
    // function_type
    // -----------------------------------------------------------------------

    #[test]
    fn test_function_type_basic() {
        let tb = TypeBuilder::new(Target::X86_64);
        let ft = tb.function_type(CType::Int, vec![CType::Int, CType::Double], false);
        match &ft {
            CType::Function {
                return_type,
                params,
                variadic,
            } => {
                assert_eq!(**return_type, CType::Int);
                assert_eq!(params.len(), 2);
                assert_eq!(params[0], CType::Int);
                assert_eq!(params[1], CType::Double);
                assert!(!variadic);
            }
            _ => panic!("Expected Function type"),
        }
    }

    #[test]
    fn test_function_type_variadic() {
        let tb = TypeBuilder::new(Target::X86_64);
        let ft = tb.function_type(CType::Void, vec![CType::Int], true);
        match &ft {
            CType::Function { variadic, .. } => assert!(*variadic),
            _ => panic!("Expected Function type"),
        }
    }

    // -----------------------------------------------------------------------
    // Qualifiers
    // -----------------------------------------------------------------------

    #[test]
    fn test_const_qualified() {
        let tb = TypeBuilder::new(Target::X86_64);
        let cq = tb.const_qualified(CType::Int);
        match &cq {
            CType::Qualified(inner, quals) => {
                assert_eq!(**inner, CType::Int);
                assert!(quals.is_const);
                assert!(!quals.is_volatile);
            }
            _ => panic!("Expected Qualified type"),
        }
    }

    #[test]
    fn test_volatile_qualified() {
        let tb = TypeBuilder::new(Target::X86_64);
        let vq = tb.volatile_qualified(CType::Double);
        match &vq {
            CType::Qualified(inner, quals) => {
                assert_eq!(**inner, CType::Double);
                assert!(quals.is_volatile);
                assert!(!quals.is_const);
            }
            _ => panic!("Expected Qualified type"),
        }
    }

    #[test]
    fn test_const_volatile_merged() {
        let tb = TypeBuilder::new(Target::X86_64);
        let cv = tb.volatile_qualified(tb.const_qualified(CType::Int));
        match &cv {
            CType::Qualified(inner, quals) => {
                assert_eq!(**inner, CType::Int);
                assert!(quals.is_const);
                assert!(quals.is_volatile);
            }
            _ => panic!("Expected Qualified type"),
        }
    }

    #[test]
    fn test_atomic_type() {
        let tb = TypeBuilder::new(Target::X86_64);
        let at = tb.atomic_type(CType::Int);
        match &at {
            CType::Atomic(inner) => assert_eq!(**inner, CType::Int),
            _ => panic!("Expected Atomic type"),
        }
    }

    #[test]
    fn test_qualifier_is_atomic_field() {
        // Verify that TypeQualifiers.is_atomic is correctly used
        // when constructing qualified types with the _Atomic qualifier.
        let quals = TypeQualifiers {
            is_atomic: true,
            ..TypeQualifiers::default()
        };
        assert!(quals.is_atomic);
        assert!(!quals.is_const);
        assert!(!quals.is_volatile);
        let qualified = CType::Qualified(Box::new(CType::Int), quals);
        // Qualified with atomic should still be an integer type
        // (resolve_and_strip strips the Qualified wrapper).
        assert!(is_integer_type(&qualified));
    }

    // -----------------------------------------------------------------------
    // Struct layout — basic
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_layout_single_int() {
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_struct_layout(&[CType::Int], false, None);
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.size, 4);
        assert_eq!(layout.alignment, 4);
        assert!(!layout.has_flexible_array);
    }

    #[test]
    fn test_struct_layout_with_padding() {
        // struct { char a; int b; } — expects 3 bytes padding after `a`.
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_struct_layout(&[CType::Char, CType::Int], false, None);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].offset, 0); // char at 0
        assert_eq!(layout.fields[1].offset, 4); // int at 4 (aligned)
        assert_eq!(layout.size, 8); // 4 + 4, padded to alignment 4
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn test_struct_layout_packed() {
        // packed struct { char a; int b; } — no padding.
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_struct_layout(&[CType::Char, CType::Int], true, None);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 1); // no padding!
        assert_eq!(layout.size, 5);
        assert_eq!(layout.alignment, 1); // packed → alignment 1
    }

    #[test]
    fn test_struct_layout_explicit_align() {
        // struct __attribute__((aligned(16))) { int a; } — alignment 16.
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_struct_layout(&[CType::Int], false, Some(16));
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.size, 16); // padded to 16
        assert_eq!(layout.alignment, 16);
    }

    #[test]
    fn test_struct_layout_flexible_array() {
        // struct { int len; char data[]; }
        let tb = TypeBuilder::new(Target::X86_64);
        let flex = CType::Array(Box::new(CType::Char), None);
        let layout = tb.compute_struct_layout(&[CType::Int, flex], false, None);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.fields[1].offset, 4);
        assert_eq!(layout.fields[1].size, 0); // flex member
        assert_eq!(layout.size, 4); // flex does not contribute
        assert!(layout.has_flexible_array);
    }

    #[test]
    fn test_struct_layout_empty() {
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_struct_layout(&[], false, None);
        assert_eq!(layout.fields.len(), 0);
        assert_eq!(layout.size, 0);
        assert_eq!(layout.alignment, 1);
    }

    // -----------------------------------------------------------------------
    // Struct layout — target-dependent
    // -----------------------------------------------------------------------

    #[test]
    fn test_struct_layout_pointer_size_x86_64() {
        // struct { char a; void *p; } on x86-64 — ptr is 8 bytes, align 8.
        let tb = TypeBuilder::new(Target::X86_64);
        let ptr_void = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        let layout = tb.compute_struct_layout(&[CType::Char, ptr_void], false, None);
        assert_eq!(layout.fields[0].offset, 0); // char at 0
        assert_eq!(layout.fields[1].offset, 8); // ptr at 8 (aligned to 8)
        assert_eq!(layout.size, 16); // 8 + 8 = 16
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_struct_layout_pointer_size_i686() {
        // struct { char a; void *p; } on i686 — ptr is 4 bytes, align 4.
        let tb = TypeBuilder::new(Target::I686);
        let ptr_void = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        let layout = tb.compute_struct_layout(&[CType::Char, ptr_void], false, None);
        assert_eq!(layout.fields[0].offset, 0); // char at 0
        assert_eq!(layout.fields[1].offset, 4); // ptr at 4 (aligned to 4)
        assert_eq!(layout.size, 8); // 4 + 4 = 8
        assert_eq!(layout.alignment, 4);
    }

    // -----------------------------------------------------------------------
    // Union layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_union_layout_basic() {
        // union { int a; double b; }
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_union_layout(&[CType::Int, CType::Double], false, None);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 0);
        assert_eq!(layout.size, 8); // max(4, 8) = 8
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_union_layout_packed() {
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_union_layout(&[CType::Int, CType::Double], true, None);
        assert_eq!(layout.size, 8); // max(4, 8) = 8, packed → align 1
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn test_union_layout_explicit_align() {
        let tb = TypeBuilder::new(Target::X86_64);
        let layout = tb.compute_union_layout(&[CType::Int], false, Some(16));
        assert_eq!(layout.size, 16); // 4 padded to alignment 16
        assert_eq!(layout.alignment, 16);
    }

    // -----------------------------------------------------------------------
    // sizeof_type / alignof_type
    // -----------------------------------------------------------------------

    #[test]
    fn test_sizeof_type_primitives() {
        let tb = TypeBuilder::new(Target::X86_64);
        assert_eq!(tb.sizeof_type(&CType::Void), 1);
        assert_eq!(tb.sizeof_type(&CType::Bool), 1);
        assert_eq!(tb.sizeof_type(&CType::Char), 1);
        assert_eq!(tb.sizeof_type(&CType::Short), 2);
        assert_eq!(tb.sizeof_type(&CType::UShort), 2);
        assert_eq!(tb.sizeof_type(&CType::Int), 4);
        assert_eq!(tb.sizeof_type(&CType::UInt), 4);
        assert_eq!(tb.sizeof_type(&CType::Long), 8);
        assert_eq!(tb.sizeof_type(&CType::ULong), 8);
        assert_eq!(tb.sizeof_type(&CType::LongLong), 8);
        assert_eq!(tb.sizeof_type(&CType::ULongLong), 8);
        assert_eq!(tb.sizeof_type(&CType::Float), 4);
        assert_eq!(tb.sizeof_type(&CType::Double), 8);
        assert_eq!(tb.sizeof_type(&CType::LongDouble), 16);
    }

    #[test]
    fn test_sizeof_type_i686_differences() {
        let tb = TypeBuilder::new(Target::I686);
        assert_eq!(tb.sizeof_type(&CType::Long), 4); // ILP32
        assert_eq!(tb.sizeof_type(&CType::ULong), 4);
        assert_eq!(tb.sizeof_type(&CType::LongDouble), 12);
        let ptr = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        assert_eq!(tb.sizeof_type(&ptr), 4);
    }

    #[test]
    fn test_sizeof_pointer_lp64() {
        for target in &[Target::X86_64, Target::AArch64, Target::RiscV64] {
            let tb = TypeBuilder::new(*target);
            let ptr = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
            assert_eq!(tb.sizeof_type(&ptr), 8);
        }
    }

    #[test]
    fn test_alignof_type_primitives() {
        let tb = TypeBuilder::new(Target::X86_64);
        assert_eq!(tb.alignof_type(&CType::Char), 1);
        assert_eq!(tb.alignof_type(&CType::Short), 2);
        assert_eq!(tb.alignof_type(&CType::Int), 4);
        assert_eq!(tb.alignof_type(&CType::Long), 8);
        assert_eq!(tb.alignof_type(&CType::Float), 4);
        assert_eq!(tb.alignof_type(&CType::Double), 8);
        assert_eq!(tb.alignof_type(&CType::LongDouble), 16);
    }

    #[test]
    fn test_alignof_type_i686_differences() {
        let tb = TypeBuilder::new(Target::I686);
        assert_eq!(tb.alignof_type(&CType::Long), 4); // ILP32
        assert_eq!(tb.alignof_type(&CType::LongDouble), 4); // i686 System V ABI
    }

    // -----------------------------------------------------------------------
    // Type predicate functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_integer_type() {
        assert!(is_integer_type(&CType::Bool));
        assert!(is_integer_type(&CType::Char));
        assert!(is_integer_type(&CType::Int));
        assert!(is_integer_type(&CType::UInt));
        assert!(is_integer_type(&CType::Short));
        assert!(is_integer_type(&CType::Long));
        assert!(is_integer_type(&CType::ULong));
        assert!(is_integer_type(&CType::LongLong));
        assert!(is_integer_type(&CType::ULongLong));
        assert!(is_integer_type(&CType::Enum {
            name: Some("E".to_string()),
            underlying_type: Box::new(CType::Int),
        }));
        assert!(!is_integer_type(&CType::Float));
        assert!(!is_integer_type(&CType::Double));
        assert!(!is_integer_type(&CType::Void));
        assert!(!is_integer_type(&CType::Pointer(
            Box::new(CType::Int),
            TypeQualifiers::default()
        )));
    }

    #[test]
    fn test_is_integer_type_through_typedef() {
        let td = CType::Typedef {
            name: "myint".to_string(),
            underlying: Box::new(CType::Int),
        };
        assert!(is_integer_type(&td));
    }

    #[test]
    fn test_is_arithmetic_type() {
        assert!(is_arithmetic_type(&CType::Int));
        assert!(is_arithmetic_type(&CType::Float));
        assert!(is_arithmetic_type(&CType::Double));
        assert!(is_arithmetic_type(&CType::LongDouble));
        assert!(!is_arithmetic_type(&CType::Void));
        assert!(!is_arithmetic_type(&CType::Pointer(
            Box::new(CType::Int),
            TypeQualifiers::default()
        )));
    }

    #[test]
    fn test_is_scalar_type() {
        assert!(is_scalar_type(&CType::Int));
        assert!(is_scalar_type(&CType::Float));
        assert!(is_scalar_type(&CType::Pointer(
            Box::new(CType::Void),
            TypeQualifiers::default()
        )));
        assert!(!is_scalar_type(&CType::Struct {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        }));
    }

    #[test]
    fn test_is_complete_type() {
        assert!(!is_complete_type(&CType::Void));
        assert!(is_complete_type(&CType::Int));
        assert!(is_complete_type(&CType::Array(
            Box::new(CType::Int),
            Some(5)
        )));
        assert!(!is_complete_type(&CType::Array(Box::new(CType::Int), None)));
        assert!(is_complete_type(&CType::Struct {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        }));
    }

    // -----------------------------------------------------------------------
    // integer_rank
    // -----------------------------------------------------------------------

    #[test]
    fn test_integer_rank() {
        assert_eq!(integer_rank(&CType::Bool), 1);
        assert_eq!(integer_rank(&CType::Char), 2);
        assert_eq!(integer_rank(&CType::Short), 3);
        assert_eq!(integer_rank(&CType::UShort), 3);
        assert_eq!(integer_rank(&CType::Int), 4);
        assert_eq!(integer_rank(&CType::UInt), 4);
        assert_eq!(integer_rank(&CType::Long), 5);
        assert_eq!(integer_rank(&CType::ULong), 5);
        assert_eq!(integer_rank(&CType::LongLong), 6);
        assert_eq!(integer_rank(&CType::ULongLong), 6);
        assert_eq!(integer_rank(&CType::Float), 0); // non-integer
    }

    #[test]
    fn test_integer_rank_enum() {
        let e = CType::Enum {
            name: Some("Color".to_string()),
            underlying_type: Box::new(CType::Int),
        };
        assert_eq!(integer_rank(&e), 4); // same as int
    }

    // -----------------------------------------------------------------------
    // usual_arithmetic_conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_uac_long_double_dominates() {
        assert_eq!(
            usual_arithmetic_conversion(&CType::LongDouble, &CType::Int),
            CType::LongDouble
        );
        assert_eq!(
            usual_arithmetic_conversion(&CType::Float, &CType::LongDouble),
            CType::LongDouble
        );
    }

    #[test]
    fn test_uac_double_dominates() {
        assert_eq!(
            usual_arithmetic_conversion(&CType::Double, &CType::Int),
            CType::Double
        );
        assert_eq!(
            usual_arithmetic_conversion(&CType::Float, &CType::Double),
            CType::Double
        );
    }

    #[test]
    fn test_uac_float_dominates() {
        assert_eq!(
            usual_arithmetic_conversion(&CType::Float, &CType::Int),
            CType::Float
        );
    }

    #[test]
    fn test_uac_same_type() {
        assert_eq!(
            usual_arithmetic_conversion(&CType::Int, &CType::Int),
            CType::Int
        );
        assert_eq!(
            usual_arithmetic_conversion(&CType::ULong, &CType::ULong),
            CType::ULong
        );
    }

    #[test]
    fn test_uac_same_sign_higher_rank() {
        // int vs long long — both signed, long long has higher rank.
        assert_eq!(
            usual_arithmetic_conversion(&CType::Int, &CType::LongLong),
            CType::LongLong
        );
    }

    #[test]
    fn test_uac_unsigned_higher_rank() {
        // unsigned long vs int — unsigned has higher rank.
        assert_eq!(
            usual_arithmetic_conversion(&CType::ULong, &CType::Int),
            CType::ULong
        );
    }

    #[test]
    fn test_uac_signed_wider_than_unsigned() {
        // long (signed) vs unsigned int — signed has higher rank and can
        // represent all unsigned int values (on LP64 where long is 64-bit).
        assert_eq!(
            usual_arithmetic_conversion(&CType::Long, &CType::UInt),
            CType::Long
        );
    }

    #[test]
    fn test_uac_char_promoted() {
        // char vs short — both promote to int.
        assert_eq!(
            usual_arithmetic_conversion(&CType::Char, &CType::Short),
            CType::Int
        );
    }

    // -----------------------------------------------------------------------
    // Bitfield layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitfield_layout_basic() {
        let tb = TypeBuilder::new(Target::X86_64);
        let fields = vec![
            StructField {
                name: Some("a".to_string()),
                ty: CType::UInt,
                bit_width: Some(3),
            },
            StructField {
                name: Some("b".to_string()),
                ty: CType::UInt,
                bit_width: Some(5),
            },
        ];
        let layout = tb.compute_struct_layout_with_fields(&fields, false, None);
        assert_eq!(layout.fields.len(), 2);
        // Both should fit in a single unsigned int allocation unit.
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 0);
        assert_eq!(layout.size, 4); // one unsigned int
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn test_bitfield_zero_width_forces_alignment() {
        let tb = TypeBuilder::new(Target::X86_64);
        let fields = vec![
            StructField {
                name: Some("a".to_string()),
                ty: CType::UInt,
                bit_width: Some(3),
            },
            StructField {
                name: None,
                ty: CType::UInt,
                bit_width: Some(0), // zero-width: force alignment
            },
            StructField {
                name: Some("b".to_string()),
                ty: CType::UInt,
                bit_width: Some(5),
            },
        ];
        let layout = tb.compute_struct_layout_with_fields(&fields, false, None);
        assert_eq!(layout.fields.len(), 3);
        // After zero-width, 'b' starts in a new allocation unit.
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[2].offset, 4); // new unit
        assert_eq!(layout.size, 8);
    }

    #[test]
    fn test_struct_layout_with_fields_regular() {
        // Same as compute_struct_layout but with StructField wrappers.
        let tb = TypeBuilder::new(Target::X86_64);
        let fields = vec![
            StructField {
                name: Some("x".to_string()),
                ty: CType::Char,
                bit_width: None,
            },
            StructField {
                name: Some("y".to_string()),
                ty: CType::Int,
                bit_width: None,
            },
        ];
        let layout = tb.compute_struct_layout_with_fields(&fields, false, None);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 4);
        assert_eq!(layout.size, 8);
    }

    // -----------------------------------------------------------------------
    // Complex nested type construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_pointer_to_array() {
        let tb = TypeBuilder::new(Target::X86_64);
        let arr = tb.array_of(CType::Int, Some(5));
        let ptr = tb.pointer_to(arr);
        assert_eq!(tb.sizeof_type(&ptr), 8); // pointer on x86-64
    }

    #[test]
    fn test_function_pointer() {
        let tb = TypeBuilder::new(Target::X86_64);
        let ft = tb.function_type(CType::Int, vec![CType::Int], false);
        let fptr = tb.pointer_to(ft);
        assert_eq!(tb.sizeof_type(&fptr), 8);
    }

    #[test]
    fn test_sizeof_array() {
        let tb = TypeBuilder::new(Target::X86_64);
        let arr = tb.array_of(CType::Int, Some(10));
        assert_eq!(tb.sizeof_type(&arr), 40); // 10 × 4
    }

    #[test]
    fn test_sizeof_struct_type() {
        let tb = TypeBuilder::new(Target::X86_64);
        let s = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        // int (4) + padding (4) + double (8) = 16
        assert_eq!(tb.sizeof_type(&s), 16);
    }

    #[test]
    fn test_sizeof_union_type() {
        let tb = TypeBuilder::new(Target::X86_64);
        let u = CType::Union {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".to_string()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        assert_eq!(tb.sizeof_type(&u), 8); // max(4, 8)
    }

    #[test]
    fn test_sizeof_enum_type() {
        let tb = TypeBuilder::new(Target::X86_64);
        let e = CType::Enum {
            name: Some("Color".to_string()),
            underlying_type: Box::new(CType::Int),
        };
        assert_eq!(tb.sizeof_type(&e), 4);
    }

    #[test]
    fn test_sizeof_typedef() {
        let tb = TypeBuilder::new(Target::X86_64);
        let td = CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::ULong),
        };
        assert_eq!(tb.sizeof_type(&td), 8);
    }

    #[test]
    fn test_sizeof_qualified() {
        let tb = TypeBuilder::new(Target::X86_64);
        let cq = tb.const_qualified(CType::Int);
        assert_eq!(tb.sizeof_type(&cq), 4);
    }

    #[test]
    fn test_sizeof_atomic() {
        let tb = TypeBuilder::new(Target::X86_64);
        let at = tb.atomic_type(CType::Int);
        assert_eq!(tb.sizeof_type(&at), 4);
    }
}
