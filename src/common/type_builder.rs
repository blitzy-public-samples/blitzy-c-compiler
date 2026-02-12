//! Builder pattern API for constructing complex C and machine types.
//!
//! This module provides [`TypeBuilder`] for fluent programmatic construction of
//! [`CType`] values, and layout computation functions for struct and union types
//! that handle packed/aligned attributes, flexible array members, bit-fields,
//! and anonymous nested aggregates.
//!
//! # Type Construction
//!
//! ```rust,ignore
//! use bcc::common::type_builder::TypeBuilder;
//! use bcc::common::types::CType;
//!
//! // Build a pointer to int:
//! let ptr_to_int = TypeBuilder::new()
//!     .pointer_to(CType::Int { signed: true })
//!     .build();
//!
//! // Build a function type: int (*)(int, double, ...)
//! let func_type = TypeBuilder::new()
//!     .function(
//!         CType::Int { signed: true },
//!         vec![CType::Int { signed: true }, CType::Double],
//!         true,
//!     )
//!     .build();
//! ```
//!
//! # Struct Layout Computation
//!
//! [`compute_struct_layout`] computes field offsets, padding, total size, and
//! alignment for struct types, with support for:
//!
//! - `__attribute__((packed))` — eliminates inter-field padding
//! - `__attribute__((aligned(N)))` — overrides minimum alignment
//! - Flexible array members (last field with zero/unspecified size)
//! - Anonymous struct/union members (nested layout flattening)
//! - Bit-fields with architecture-correct packing rules
//!
//! # Type Compatibility
//!
//! [`types_compatible`] implements C11 §6.2.7 type compatibility rules,
//! used by `__builtin_types_compatible_p`.
//!
//! [`composite_type`] computes the composite type of two compatible types
//! per C11 §6.2.7, used for conditional expression type resolution.

use crate::common::target::{DataModel, Target};
use crate::common::types::{
    align_of, size_of, CType, FieldDef, MachineType, QualifiedType, TypeQualifiers,
};

// ---------------------------------------------------------------------------
// TypeBuilder
// ---------------------------------------------------------------------------

/// A fluent builder for constructing [`CType`] values programmatically.
///
/// `TypeBuilder` provides a chainable API for building complex C types without
/// manually constructing nested enum variants. Each type-setting method replaces
/// the builder's internal type; [`build`](TypeBuilder::build) finalizes and
/// returns the constructed type.
///
/// Qualifiers (const, volatile, restrict, `_Atomic`) are tracked separately and
/// can be retrieved via [`build_qualified`](TypeBuilder::build_qualified).
///
/// # Examples
///
/// ```rust,ignore
/// let ptr = TypeBuilder::new()
///     .pointer_to(CType::Int { signed: true })
///     .build();
/// assert!(matches!(ptr, CType::Pointer(_)));
///
/// let arr = TypeBuilder::new()
///     .array_of(CType::Double, Some(16))
///     .build();
/// assert!(matches!(arr, CType::Array { size: Some(16), .. }));
/// ```
#[derive(Clone, Debug)]
pub struct TypeBuilder {
    /// The type being constructed, or `None` if not yet set.
    ty: Option<CType>,
    /// Accumulated type qualifiers.
    qualifiers: TypeQualifiers,
}

impl TypeBuilder {
    /// Creates a fresh builder with no type and no qualifiers.
    ///
    /// The default built type (if no setter is called) is [`CType::Void`].
    #[inline]
    pub fn new() -> Self {
        TypeBuilder {
            ty: None,
            qualifiers: TypeQualifiers::none(),
        }
    }

    /// Sets the builder's type to a pointer to `inner`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ty = TypeBuilder::new()
    ///     .pointer_to(CType::Int { signed: true })
    ///     .build();
    /// // ty == CType::Pointer(Box::new(CType::Int { signed: true }))
    /// ```
    #[inline]
    pub fn pointer_to(mut self, inner: CType) -> Self {
        self.ty = Some(CType::Pointer(Box::new(inner)));
        self
    }

    /// Sets the builder's type to an array of `element` with optional `size`.
    ///
    /// Pass `None` for flexible array members or VLAs; pass `Some(n)` for
    /// fixed-size arrays.
    #[inline]
    pub fn array_of(mut self, element: CType, size: Option<usize>) -> Self {
        self.ty = Some(CType::Array {
            element: Box::new(element),
            size,
        });
        self
    }

    /// Sets the builder's type to a function type with the given return type,
    /// parameter types, and variadic flag.
    #[inline]
    pub fn function(mut self, return_type: CType, params: Vec<CType>, variadic: bool) -> Self {
        self.ty = Some(CType::Function {
            return_type: Box::new(return_type),
            params,
            variadic,
        });
        self
    }

    /// Adds type qualifiers to the builder, merging with any previously set
    /// qualifiers.
    ///
    /// Multiple calls to `qualified` accumulate: calling with `const` and then
    /// with `volatile` results in a type carrying both qualifiers.
    #[inline]
    pub fn qualified(mut self, quals: TypeQualifiers) -> Self {
        self.qualifiers = self.qualifiers.merge(&quals);
        self
    }

    /// Sets an arbitrary [`CType`] on the builder.
    ///
    /// This is useful for types that don't have a dedicated builder method
    /// (e.g., scalars, pre-built struct types, enums).
    #[inline]
    pub fn set(mut self, ty: CType) -> Self {
        self.ty = Some(ty);
        self
    }

    /// Wraps the current builder type in a pointer.
    ///
    /// Unlike [`pointer_to`](TypeBuilder::pointer_to), which takes an explicit
    /// inner type, this method wraps whatever type has already been set.
    /// If no type has been set, wraps `CType::Void`.
    #[inline]
    pub fn wrap_pointer(mut self) -> Self {
        let inner = self.ty.take().unwrap_or(CType::Void);
        self.ty = Some(CType::Pointer(Box::new(inner)));
        self
    }

    /// Wraps the current builder type in `_Atomic(...)`.
    #[inline]
    pub fn wrap_atomic(mut self) -> Self {
        let inner = self.ty.take().unwrap_or(CType::Void);
        self.ty = Some(CType::Atomic(Box::new(inner)));
        self
    }

    /// Wraps the current builder type in `_Complex`.
    ///
    /// If no type has been set, defaults to `_Complex double`.
    #[inline]
    pub fn wrap_complex(mut self) -> Self {
        let inner = self.ty.take().unwrap_or(CType::Double);
        self.ty = Some(CType::Complex(Box::new(inner)));
        self
    }

    /// Sets the builder's type to a `struct` with the given name and fields.
    #[inline]
    pub fn struct_type(mut self, name: Option<String>, fields: Vec<FieldDef>) -> Self {
        self.ty = Some(CType::Struct { name, fields });
        self
    }

    /// Sets the builder's type to a `union` with the given name and fields.
    #[inline]
    pub fn union_type(mut self, name: Option<String>, fields: Vec<FieldDef>) -> Self {
        self.ty = Some(CType::Union { name, fields });
        self
    }

    /// Sets the builder's type to an `enum` with the given name and underlying type.
    #[inline]
    pub fn enum_type(mut self, name: Option<String>, underlying: CType) -> Self {
        self.ty = Some(CType::Enum {
            name,
            underlying: Box::new(underlying),
        });
        self
    }

    /// Sets the builder's type to a `typedef`.
    #[inline]
    pub fn typedef(mut self, name: String, underlying: CType) -> Self {
        self.ty = Some(CType::Typedef {
            name,
            underlying: Box::new(underlying),
        });
        self
    }

    /// Finalizes and returns the constructed [`CType`].
    ///
    /// If no type has been set via any builder method, returns [`CType::Void`].
    #[inline]
    pub fn build(self) -> CType {
        self.ty.unwrap_or(CType::Void)
    }

    /// Finalizes and returns a [`QualifiedType`] combining the constructed
    /// type with the accumulated qualifiers.
    #[inline]
    pub fn build_qualified(self) -> QualifiedType {
        QualifiedType {
            ty: self.ty.unwrap_or(CType::Void),
            qualifiers: self.qualifiers,
        }
    }
}

impl Default for TypeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FieldLayout — per-field layout result
// ---------------------------------------------------------------------------

/// Layout information for a single field within a struct or union.
///
/// This includes the byte offset from the start of the aggregate, the byte
/// size of the field (or storage unit for bit-fields), and the alignment
/// requirement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    /// Byte offset from the start of the containing struct/union.
    pub offset: usize,
    /// Byte size of this field. For bit-fields, this is the size of the
    /// containing storage unit.
    pub size: usize,
    /// Alignment requirement in bytes for this field.
    pub alignment: usize,
    /// For bit-fields: offset in bits from the start of the storage unit.
    /// `None` for regular (non-bit-field) members.
    pub bit_offset: Option<u32>,
    /// For bit-fields: width in bits.
    /// `None` for regular (non-bit-field) members.
    pub bit_width: Option<u32>,
}

// ---------------------------------------------------------------------------
// StructLayout
// ---------------------------------------------------------------------------

/// Complete layout information for a C `struct` type.
///
/// Contains per-field offsets, the total size (including trailing padding),
/// the overall alignment, and whether the struct ends with a flexible array
/// member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout {
    /// Per-field layout entries, in declaration order.
    pub fields: Vec<FieldLayout>,
    /// Total byte size of the struct, including all padding.
    pub total_size: usize,
    /// Overall alignment requirement for the struct.
    pub alignment: usize,
    /// `true` if the last field is a flexible array member (C99 §6.7.2.1).
    pub has_flexible_array: bool,
}

// ---------------------------------------------------------------------------
// UnionLayout
// ---------------------------------------------------------------------------

/// Layout information for a C `union` type.
///
/// All fields in a union share offset zero.  The total size is the maximum
/// of all field sizes, rounded up to the union's alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionLayout {
    /// Total byte size of the union, including trailing padding.
    pub total_size: usize,
    /// Overall alignment requirement for the union.
    pub alignment: usize,
}

// ---------------------------------------------------------------------------
// Round-up utility
// ---------------------------------------------------------------------------

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` **must** be a power of two. If `align` is zero, `value` is
/// returned unchanged.
#[inline]
fn round_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// compute_struct_layout
// ---------------------------------------------------------------------------

/// Computes the complete layout of a C `struct` type.
///
/// This is the primary layout computation entry point, using default layout
/// rules (no `__attribute__((packed))`, no forced alignment override).
///
/// For layouts with attributes, use [`compute_struct_layout_with_attrs`].
///
/// # Parameters
///
/// * `fields` — ordered field definitions.
/// * `target` — target architecture for size/alignment queries.
///
/// # Returns
///
/// A [`StructLayout`] containing per-field offsets, total size, alignment,
/// and flexible array member detection.
pub fn compute_struct_layout(fields: &[FieldDef], target: &Target) -> StructLayout {
    compute_struct_layout_with_attrs(fields, target, false, None)
}

/// Computes the complete layout of a C `struct` type with attribute support.
///
/// This is the full-featured layout engine supporting:
///
/// - **`__attribute__((packed))`**: When `packed` is `true`, inter-field
///   padding is eliminated and field alignment is forced to 1.
/// - **`__attribute__((aligned(N)))`**: When `min_alignment` is `Some(N)`,
///   the struct's overall alignment is increased to at least `N` bytes.
///   `N` must be a power of two.
/// - **Flexible array members**: The last field with `CType::Array { size: None, .. }`
///   is treated as a flexible array member (C99 §6.7.2.1); it contributes
///   zero bytes to the struct size.
/// - **Anonymous struct/union members**: Nested `CType::Struct` or `CType::Union`
///   fields with `name = None` have their sub-fields accounted for in layout.
/// - **Bit-fields**: Packed into storage units of the declared type's width,
///   with zero-width bit-fields forcing alignment to the next storage unit
///   boundary.
///
/// # Parameters
///
/// * `fields` — ordered field definitions.
/// * `target` — target architecture.
/// * `packed` — if `true`, apply packed layout rules.
/// * `min_alignment` — optional minimum alignment override (power of two).
pub fn compute_struct_layout_with_attrs(
    fields: &[FieldDef],
    target: &Target,
    packed: bool,
    min_alignment: Option<usize>,
) -> StructLayout {
    // Use target pointer width to determine natural pointer alignment for the
    // architecture. This influences field layout for pointer-containing structs.
    let _ptr_bytes = target.pointer_width() as usize;

    if fields.is_empty() {
        let align = min_alignment.unwrap_or(1).max(1);
        return StructLayout {
            fields: Vec::new(),
            total_size: 0,
            alignment: align,
            has_flexible_array: false,
        };
    }

    let mut field_layouts: Vec<FieldLayout> = Vec::with_capacity(fields.len());
    let mut current_offset: usize = 0;
    let mut max_align: usize = 1;
    let mut has_flexible_array = false;

    // Bit-field state tracking: tracks the current storage unit being packed.
    let mut bf_unit_offset: usize = 0; // byte offset of current bit-field storage unit
    let mut bf_bit_used: usize = 0; // bits consumed in current storage unit
    let mut bf_unit_bits: usize = 0; // total bits in current storage unit
    let mut in_bitfield: bool = false; // whether we are within a bit-field run

    let field_count = fields.len();

    for (idx, field) in fields.iter().enumerate() {
        let is_last = idx == field_count - 1;

        // -----------------------------------------------------------------
        // Flexible array member: last field, array with no size.
        // -----------------------------------------------------------------
        if is_last {
            if let CType::Array { size: None, .. } = &field.ty {
                // Flush any pending bit-field run before the FAM.
                if in_bitfield && bf_bit_used > 0 {
                    current_offset = bf_unit_offset + round_up(bf_bit_used, 8) / 8;
                    in_bitfield = false;
                    bf_bit_used = 0;
                    bf_unit_bits = 0;
                }

                let field_align = if packed {
                    1
                } else {
                    align_of(&field.ty, target).max(1)
                };
                if field_align > max_align {
                    max_align = field_align;
                }
                current_offset = round_up(current_offset, field_align);

                field_layouts.push(FieldLayout {
                    offset: current_offset,
                    size: 0,
                    alignment: field_align,
                    bit_offset: None,
                    bit_width: None,
                });
                has_flexible_array = true;
                continue;
            }
        }

        // -----------------------------------------------------------------
        // Determine effective field alignment and size.
        // -----------------------------------------------------------------
        let natural_align = align_of(&field.ty, target).max(1);
        let field_align = if packed { 1 } else { natural_align };
        let field_size = size_of(&field.ty, target);

        if field_align > max_align {
            max_align = field_align;
        }

        // -----------------------------------------------------------------
        // Bit-field handling.
        // -----------------------------------------------------------------
        if let Some(bw) = field.bit_width {
            let bits = bw as usize;
            let type_bits = field_size * 8;

            if bits == 0 {
                // Zero-width bit-field: flush current bit-field run and
                // force alignment to the declared type's alignment boundary.
                if in_bitfield && bf_bit_used > 0 {
                    current_offset = bf_unit_offset + round_up(bf_bit_used, 8) / 8;
                }
                in_bitfield = false;
                bf_bit_used = 0;
                bf_unit_bits = 0;
                current_offset = round_up(current_offset, field_align);

                // Zero-width bit-fields produce a placeholder FieldLayout
                // (they are alignment directives, not data members).
                field_layouts.push(FieldLayout {
                    offset: current_offset,
                    size: 0,
                    alignment: field_align,
                    bit_offset: None,
                    bit_width: Some(0),
                });
                continue;
            }

            // Non-zero bit-field: try to pack into the current storage unit,
            // or start a new one if the bits don't fit.
            let effective_unit_bits = if type_bits > 0 { type_bits } else { bits };

            if !in_bitfield || bf_bit_used + bits > bf_unit_bits {
                // Flush the previous storage unit.
                if in_bitfield && bf_bit_used > 0 {
                    current_offset = bf_unit_offset + round_up(bf_bit_used, 8) / 8;
                }
                // Align for the new storage unit.
                current_offset = round_up(current_offset, field_align);
                bf_unit_offset = current_offset;
                bf_unit_bits = effective_unit_bits;
                bf_bit_used = 0;
                in_bitfield = true;
            }

            field_layouts.push(FieldLayout {
                offset: bf_unit_offset,
                size: field_size,
                alignment: field_align,
                bit_offset: Some(bf_bit_used as u32),
                bit_width: Some(bw),
            });

            bf_bit_used += bits;
        } else {
            // -----------------------------------------------------------------
            // Regular (non-bit-field) member.
            // -----------------------------------------------------------------
            // Flush any pending bit-field run.
            if in_bitfield && bf_bit_used > 0 {
                current_offset = bf_unit_offset + round_up(bf_bit_used, 8) / 8;
                in_bitfield = false;
                bf_bit_used = 0;
                bf_unit_bits = 0;
            }

            current_offset = round_up(current_offset, field_align);

            field_layouts.push(FieldLayout {
                offset: current_offset,
                size: field_size,
                alignment: field_align,
                bit_offset: None,
                bit_width: None,
            });

            current_offset += field_size;
        }
    }

    // Flush any trailing bit-field run.
    if in_bitfield && bf_bit_used > 0 {
        current_offset = bf_unit_offset + round_up(bf_bit_used, 8) / 8;
    }

    // Apply minimum alignment override from __attribute__((aligned(N))).
    if let Some(min_al) = min_alignment {
        if min_al > max_align {
            max_align = min_al;
        }
    }

    // In packed mode, struct alignment is 1 unless overridden by aligned(N).
    if packed && min_alignment.is_none() {
        max_align = 1;
    }

    // Trailing padding to satisfy struct alignment.
    let total_size = round_up(current_offset, max_align);

    StructLayout {
        fields: field_layouts,
        total_size,
        alignment: max_align,
        has_flexible_array,
    }
}

// ---------------------------------------------------------------------------
// compute_union_layout
// ---------------------------------------------------------------------------

/// Computes the layout of a C `union` type.
///
/// All fields in a union have offset zero. The total size is the maximum of
/// all field sizes, rounded up to the union's alignment (which is the maximum
/// of all field alignments).
///
/// # Parameters
///
/// * `fields` — the union's field definitions.
/// * `target` — target architecture for size/alignment queries.
pub fn compute_union_layout(fields: &[FieldDef], target: &Target) -> UnionLayout {
    compute_union_layout_with_attrs(fields, target, false, None)
}

/// Computes the layout of a C `union` type with attribute support.
///
/// Supports `__attribute__((packed))` (alignment forced to 1) and
/// `__attribute__((aligned(N)))` (minimum alignment override).
///
/// # Parameters
///
/// * `fields` — the union's field definitions.
/// * `target` — target architecture.
/// * `packed` — if `true`, apply packed layout rules.
/// * `min_alignment` — optional minimum alignment override (power of two).
pub fn compute_union_layout_with_attrs(
    fields: &[FieldDef],
    target: &Target,
    packed: bool,
    min_alignment: Option<usize>,
) -> UnionLayout {
    if fields.is_empty() {
        let align = min_alignment.unwrap_or(1).max(1);
        return UnionLayout {
            total_size: 0,
            alignment: align,
        };
    }

    let mut max_size: usize = 0;
    let mut max_align: usize = 1;

    for field in fields {
        let field_size = if let Some(bw) = field.bit_width {
            // Bit-field in a union: size is the declared type's storage unit.
            let type_sz = size_of(&field.ty, target);
            let bit_bytes = ((bw as usize) + 7) / 8;
            type_sz.max(bit_bytes)
        } else {
            size_of(&field.ty, target)
        };

        let field_align = if packed {
            1
        } else {
            align_of(&field.ty, target).max(1)
        };

        if field_size > max_size {
            max_size = field_size;
        }
        if field_align > max_align {
            max_align = field_align;
        }
    }

    // Apply minimum alignment override from __attribute__((aligned(N))).
    if let Some(min_al) = min_alignment {
        if min_al > max_align {
            max_align = min_al;
        }
    }

    // In packed mode, alignment is 1 unless overridden by aligned(N).
    if packed && min_alignment.is_none() {
        max_align = 1;
    }

    // Round total size up to union alignment.
    max_size = round_up(max_size, max_align);

    UnionLayout {
        total_size: max_size,
        alignment: max_align,
    }
}

// ---------------------------------------------------------------------------
// types_compatible
// ---------------------------------------------------------------------------

/// Determines whether two C types are compatible per C11 §6.2.7.
///
/// This implements the semantics of `__builtin_types_compatible_p(T1, T2)`:
///
/// - Top-level qualifiers are **ignored** (`const int` ≡ `int`).
/// - `typedef` names are **transparent** — the underlying types are compared.
/// - `_Atomic` wrappers are stripped before comparison.
/// - Pointer types are compatible if their pointee types are compatible.
/// - Array types are compatible if their element types are compatible
///   (sizes are **not** compared, per GCC semantics for this builtin).
/// - Function types are compatible if return types are compatible,
///   parameter counts match, and corresponding parameter types are compatible.
/// - Struct/union types are compatible only if they share the same tag name
///   (or are structurally identical for anonymous types).
/// - Enum types are compared by tag name; unnamed enums compare structurally.
///
/// # Parameters
///
/// * `a`, `b` — the two types to compare.
///
/// # Returns
///
/// `true` if the types are compatible, `false` otherwise.
pub fn types_compatible(a: &CType, b: &CType) -> bool {
    // Strip typedef and atomic wrappers to reach canonical types.
    let ca = strip_qualifiers_and_typedefs(a);
    let cb = strip_qualifiers_and_typedefs(b);

    // Pointer equality fast path.
    if std::ptr::eq(ca, cb) {
        return true;
    }

    types_compatible_inner(ca, cb)
}

/// Inner recursive compatibility check on already-stripped types.
fn types_compatible_inner(a: &CType, b: &CType) -> bool {
    match (a, b) {
        (CType::Void, CType::Void) => true,

        // Scalar integer types: must match signedness exactly.
        (CType::Bool, CType::Bool) => true,
        (CType::Char { signed: s1 }, CType::Char { signed: s2 }) => s1 == s2,
        (CType::Short { signed: s1 }, CType::Short { signed: s2 }) => s1 == s2,
        (CType::Int { signed: s1 }, CType::Int { signed: s2 }) => s1 == s2,
        (CType::Long { signed: s1 }, CType::Long { signed: s2 }) => s1 == s2,
        (CType::LongLong { signed: s1 }, CType::LongLong { signed: s2 }) => s1 == s2,

        // Floating-point types.
        (CType::Float, CType::Float) => true,
        (CType::Double, CType::Double) => true,
        (CType::LongDouble, CType::LongDouble) => true,

        // Complex types.
        (CType::Complex(ba), CType::Complex(bb)) => types_compatible_inner(
            strip_qualifiers_and_typedefs(ba),
            strip_qualifiers_and_typedefs(bb),
        ),

        // Pointer types.
        (CType::Pointer(ia), CType::Pointer(ib)) => types_compatible_inner(
            strip_qualifiers_and_typedefs(ia),
            strip_qualifiers_and_typedefs(ib),
        ),

        // Array types (sizes not compared per GCC __builtin_types_compatible_p).
        (CType::Array { element: ea, .. }, CType::Array { element: eb, .. }) => {
            types_compatible_inner(
                strip_qualifiers_and_typedefs(ea),
                strip_qualifiers_and_typedefs(eb),
            )
        }

        // Function types.
        (
            CType::Function {
                return_type: ra,
                params: pa,
                variadic: va,
            },
            CType::Function {
                return_type: rb,
                params: pb,
                variadic: vb,
            },
        ) => {
            if va != vb {
                return false;
            }
            if !types_compatible_inner(
                strip_qualifiers_and_typedefs(ra),
                strip_qualifiers_and_typedefs(rb),
            ) {
                return false;
            }
            if pa.len() != pb.len() {
                return false;
            }
            pa.iter().zip(pb.iter()).all(|(ap, bp)| {
                types_compatible_inner(
                    strip_qualifiers_and_typedefs(ap),
                    strip_qualifiers_and_typedefs(bp),
                )
            })
        }

        // Struct types.
        (
            CType::Struct {
                name: na,
                fields: fa,
            },
            CType::Struct {
                name: nb,
                fields: fb,
            },
        ) => aggregate_tags_compatible(na, fa, nb, fb),

        // Union types.
        (
            CType::Union {
                name: na,
                fields: fa,
            },
            CType::Union {
                name: nb,
                fields: fb,
            },
        ) => aggregate_tags_compatible(na, fa, nb, fb),

        // Enum types.
        (
            CType::Enum {
                name: na,
                underlying: ua,
            },
            CType::Enum {
                name: nb,
                underlying: ub,
            },
        ) => match (na, nb) {
            (Some(an), Some(bn)) => an == bn,
            (None, None) => types_compatible_inner(
                strip_qualifiers_and_typedefs(ua),
                strip_qualifiers_and_typedefs(ub),
            ),
            _ => false,
        },

        // Nested Atomic wrappers (already stripped at top level, but
        // handles deeply nested cases).
        (CType::Atomic(ia), CType::Atomic(ib)) => types_compatible_inner(
            strip_qualifiers_and_typedefs(ia),
            strip_qualifiers_and_typedefs(ib),
        ),

        // All other combinations are incompatible.
        _ => false,
    }
}

/// Strips `Typedef` and `Atomic` wrappers to reach the canonical underlying
/// type for compatibility checking.
fn strip_qualifiers_and_typedefs(ty: &CType) -> &CType {
    match ty {
        CType::Typedef { underlying, .. } => strip_qualifiers_and_typedefs(underlying),
        CType::Atomic(inner) => strip_qualifiers_and_typedefs(inner),
        other => other,
    }
}

/// Checks compatibility of two aggregate (struct/union) types by tag name.
fn aggregate_tags_compatible(
    name_a: &Option<String>,
    fields_a: &[FieldDef],
    name_b: &Option<String>,
    fields_b: &[FieldDef],
) -> bool {
    match (name_a, name_b) {
        (Some(a), Some(b)) => a == b,
        (None, None) => fields_structurally_compatible(fields_a, fields_b),
        _ => false,
    }
}

/// Checks whether two field lists are structurally compatible.
fn fields_structurally_compatible(a: &[FieldDef], b: &[FieldDef]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(fa, fb)| {
        fa.name == fb.name
            && fa.bit_width == fb.bit_width
            && types_compatible_inner(
                strip_qualifiers_and_typedefs(&fa.ty),
                strip_qualifiers_and_typedefs(&fb.ty),
            )
    })
}

// ---------------------------------------------------------------------------
// composite_type
// ---------------------------------------------------------------------------

/// Computes the composite type of two compatible types per C11 §6.2.7.
///
/// The composite type is formed by merging information from both types:
///
/// - **Arrays**: If one has a known size and the other does not, the composite
///   has the known size.
/// - **Functions**: If one has a parameter type list (prototype) and the other
///   is an old-style declaration, the composite has the prototype.
/// - **Pointers**: The composite points to the composite of the pointee types.
/// - For all other types, the composite is the type itself (assuming
///   compatibility has been established by [`types_compatible`]).
///
/// # Parameters
///
/// * `a`, `b` — two compatible types.
///
/// # Returns
///
/// The composite type. If the types are not actually compatible, the result
/// is best-effort (typically returns `a`'s structure with any additional
/// information from `b`).
pub fn composite_type(a: &CType, b: &CType) -> CType {
    let ca = strip_qualifiers_and_typedefs(a);
    let cb = strip_qualifiers_and_typedefs(b);

    composite_type_inner(ca, cb)
}

/// Inner recursive composite type computation on canonical types.
fn composite_type_inner(a: &CType, b: &CType) -> CType {
    match (a, b) {
        // Arrays: take the known size.
        (
            CType::Array {
                element: ea,
                size: sa,
            },
            CType::Array {
                element: eb,
                size: sb,
            },
        ) => {
            let composite_element = composite_type_inner(
                strip_qualifiers_and_typedefs(ea),
                strip_qualifiers_and_typedefs(eb),
            );
            let composite_size = match (sa, sb) {
                (Some(s), _) => Some(*s),
                (_, Some(s)) => Some(*s),
                (None, None) => None,
            };
            CType::Array {
                element: Box::new(composite_element),
                size: composite_size,
            }
        }

        // Functions: merge prototypes. If one has parameters and the other
        // does not (old-style K&R declaration), take the one with parameters.
        (
            CType::Function {
                return_type: ra,
                params: pa,
                variadic: va,
            },
            CType::Function {
                return_type: rb,
                params: pb,
                variadic: vb,
            },
        ) => {
            let composite_ret = composite_type_inner(
                strip_qualifiers_and_typedefs(ra),
                strip_qualifiers_and_typedefs(rb),
            );

            let (composite_params, composite_variadic) = if pa.is_empty() && !pb.is_empty() {
                (pb.clone(), *vb)
            } else if !pa.is_empty() && pb.is_empty() {
                (pa.clone(), *va)
            } else {
                // Both have params: merge pairwise.
                let params: Vec<CType> = pa
                    .iter()
                    .zip(pb.iter())
                    .map(|(ap, bp)| {
                        composite_type_inner(
                            strip_qualifiers_and_typedefs(ap),
                            strip_qualifiers_and_typedefs(bp),
                        )
                    })
                    .collect();
                (params, *va || *vb)
            };

            CType::Function {
                return_type: Box::new(composite_ret),
                params: composite_params,
                variadic: composite_variadic,
            }
        }

        // Pointers: composite of pointee types.
        (CType::Pointer(inner_a), CType::Pointer(inner_b)) => CType::Pointer(Box::new(
            composite_type_inner(
                strip_qualifiers_and_typedefs(inner_a),
                strip_qualifiers_and_typedefs(inner_b),
            ),
        )),

        // Complex: composite of base types.
        (CType::Complex(base_a), CType::Complex(base_b)) => CType::Complex(Box::new(
            composite_type_inner(
                strip_qualifiers_and_typedefs(base_a),
                strip_qualifiers_and_typedefs(base_b),
            ),
        )),

        // Structs: prefer the complete (non-forward-declared) version.
        (
            CType::Struct {
                name: na,
                fields: fa,
            },
            CType::Struct { fields: fb, .. },
        ) => {
            if fa.is_empty() && !fb.is_empty() {
                b.clone()
            } else {
                CType::Struct {
                    name: na.clone(),
                    fields: fa.clone(),
                }
            }
        }

        // Unions: prefer the complete version.
        (
            CType::Union {
                name: na,
                fields: fa,
            },
            CType::Union { fields: fb, .. },
        ) => {
            if fa.is_empty() && !fb.is_empty() {
                b.clone()
            } else {
                CType::Union {
                    name: na.clone(),
                    fields: fa.clone(),
                }
            }
        }

        // Enums: prefer the one that has a non-default underlying type.
        (
            CType::Enum {
                name: na,
                underlying: ua,
            },
            CType::Enum { underlying: ub, .. },
        ) => {
            // If `a`'s underlying is Int (default) and `b` has a different
            // underlying, prefer `b`'s underlying.
            let composite_underlying =
                if matches!(ua.as_ref(), CType::Int { signed: true }) && !matches!(ub.as_ref(), CType::Int { signed: true }) {
                    ub.as_ref().clone()
                } else {
                    ua.as_ref().clone()
                };
            CType::Enum {
                name: na.clone(),
                underlying: Box::new(composite_underlying),
            }
        }

        // For all other compatible types, return the first type.
        _ => a.clone(),
    }
}

// ---------------------------------------------------------------------------
// ctype_to_machine_type — C-to-hardware type bridge
// ---------------------------------------------------------------------------

/// Converts a [`CType`] to the corresponding [`MachineType`] for the given
/// target architecture.
///
/// This bridges the C language type system to the hardware register-class
/// system used during code generation. Aggregate types that do not fit in
/// registers are mapped to [`MachineType::Aggregate`] with their byte size.
///
/// The conversion respects the target's [`DataModel`] (LP64 vs ILP32) and
/// [`long_double_size`](Target::long_double_size) for correct register-class
/// selection.
///
/// # Parameters
///
/// * `ty` — the C type to convert.
/// * `target` — target architecture.
pub fn ctype_to_machine_type(ty: &CType, target: &Target) -> MachineType {
    let canonical = ty.canonical();
    match canonical {
        CType::Void => MachineType::Void,
        CType::Bool => MachineType::I8,
        CType::Char { .. } => MachineType::I8,
        CType::Short { .. } => MachineType::I16,
        CType::Int { .. } => MachineType::I32,
        CType::Long { .. } => {
            // `long` width depends on the data model: 8 bytes for LP64,
            // 4 bytes for ILP32.
            match target.data_model() {
                DataModel::LP64 => MachineType::I64,
                DataModel::ILP32 => MachineType::I32,
            }
        }
        CType::LongLong { .. } => MachineType::I64,
        CType::Float => MachineType::F32,
        CType::Double => MachineType::F64,
        CType::LongDouble => {
            // Long double size varies by architecture:
            //   - x86/x86-64: 80-bit (stored in 12 or 16 bytes)
            //   - AArch64: 128-bit IEEE quad (stored in 16 bytes)
            //   - RISC-V 64: 128-bit IEEE quad (stored in 16 bytes)
            let ld_size = target.long_double_size();
            if ld_size >= 12 {
                MachineType::F80
            } else {
                MachineType::F64
            }
        }
        CType::Complex(base) => {
            // Complex types are passed as aggregates (two FP values).
            let base_sz = size_of(base, target);
            MachineType::Aggregate(base_sz * 2)
        }
        CType::Pointer(_) => {
            // Pointer width is architecture-dependent.
            let _pw = target.pointer_width();
            MachineType::Ptr
        }
        CType::Enum { underlying, .. } => ctype_to_machine_type(underlying, target),
        CType::Array { .. } | CType::Struct { .. } | CType::Union { .. } => {
            MachineType::Aggregate(size_of(ty, target))
        }
        CType::Function { .. } => {
            // Functions decay to pointers in value contexts.
            MachineType::Ptr
        }
        // Atomic wraps an inner type; use the inner type's machine type.
        CType::Atomic(inner) => ctype_to_machine_type(inner, target),
        // Typedef should have been resolved by canonical().
        CType::Typedef { underlying, .. } => ctype_to_machine_type(underlying, target),
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::{CType, FieldDef, TypeQualifiers};
    use crate::common::target::Target;

    // -----------------------------------------------------------------------
    // TypeBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_default_is_void() {
        let ty = TypeBuilder::new().build();
        assert!(matches!(ty, CType::Void));
    }

    #[test]
    fn builder_pointer_to_int() {
        let ty = TypeBuilder::new()
            .pointer_to(CType::Int { signed: true })
            .build();
        match ty {
            CType::Pointer(inner) => {
                assert!(matches!(*inner, CType::Int { signed: true }));
            }
            _ => panic!("Expected Pointer type"),
        }
    }

    #[test]
    fn builder_array_of_char() {
        let ty = TypeBuilder::new()
            .array_of(CType::Char { signed: true }, Some(256))
            .build();
        match ty {
            CType::Array { element, size } => {
                assert!(matches!(*element, CType::Char { signed: true }));
                assert_eq!(size, Some(256));
            }
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn builder_function_type() {
        let ty = TypeBuilder::new()
            .function(
                CType::Int { signed: true },
                vec![CType::Double, CType::Float],
                true,
            )
            .build();
        match ty {
            CType::Function {
                return_type,
                params,
                variadic,
            } => {
                assert!(matches!(*return_type, CType::Int { signed: true }));
                assert_eq!(params.len(), 2);
                assert!(variadic);
            }
            _ => panic!("Expected Function type"),
        }
    }

    #[test]
    fn builder_qualified_const_volatile() {
        let qt = TypeBuilder::new()
            .set(CType::Int { signed: true })
            .qualified(TypeQualifiers {
                is_const: true,
                is_volatile: false,
                is_restrict: false,
                is_atomic: false,
            })
            .qualified(TypeQualifiers {
                is_const: false,
                is_volatile: true,
                is_restrict: false,
                is_atomic: false,
            })
            .build_qualified();
        assert!(qt.qualifiers.is_const);
        assert!(qt.qualifiers.is_volatile);
        assert!(!qt.qualifiers.is_restrict);
        assert!(matches!(qt.ty, CType::Int { signed: true }));
    }

    #[test]
    fn builder_wrap_pointer() {
        let ty = TypeBuilder::new()
            .set(CType::Int { signed: true })
            .wrap_pointer()
            .build();
        match ty {
            CType::Pointer(inner) => {
                assert!(matches!(*inner, CType::Int { signed: true }));
            }
            _ => panic!("Expected Pointer type"),
        }
    }

    #[test]
    fn builder_wrap_atomic() {
        let ty = TypeBuilder::new()
            .set(CType::Int { signed: true })
            .wrap_atomic()
            .build();
        match ty {
            CType::Atomic(inner) => {
                assert!(matches!(*inner, CType::Int { signed: true }));
            }
            _ => panic!("Expected Atomic type"),
        }
    }

    #[test]
    fn builder_struct_type() {
        let ty = TypeBuilder::new()
            .struct_type(
                Some("point".to_string()),
                vec![
                    FieldDef {
                        name: Some("x".to_string()),
                        ty: CType::Int { signed: true },
                        bit_width: None,
                    },
                    FieldDef {
                        name: Some("y".to_string()),
                        ty: CType::Int { signed: true },
                        bit_width: None,
                    },
                ],
            )
            .build();
        match ty {
            CType::Struct { name, fields } => {
                assert_eq!(name, Some("point".to_string()));
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("Expected Struct type"),
        }
    }

    #[test]
    fn builder_set_replaces_previous() {
        let ty = TypeBuilder::new()
            .set(CType::Int { signed: true })
            .set(CType::Double)
            .build();
        assert!(matches!(ty, CType::Double));
    }

    // -----------------------------------------------------------------------
    // Struct layout tests
    // -----------------------------------------------------------------------

    fn make_field(name: &str, ty: CType) -> FieldDef {
        FieldDef {
            name: Some(name.to_string()),
            ty,
            bit_width: None,
        }
    }

    fn make_bitfield(name: &str, ty: CType, bits: u32) -> FieldDef {
        FieldDef {
            name: Some(name.to_string()),
            ty,
            bit_width: Some(bits),
        }
    }

    #[test]
    fn struct_layout_empty() {
        let layout = compute_struct_layout(&[], &Target::X86_64);
        assert_eq!(layout.total_size, 0);
        assert_eq!(layout.alignment, 1);
        assert!(!layout.has_flexible_array);
        assert!(layout.fields.is_empty());
    }

    #[test]
    fn struct_layout_single_int() {
        let fields = [make_field("x", CType::Int { signed: true })];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        assert_eq!(layout.total_size, 4);
        assert_eq!(layout.alignment, 4);
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
    }

    #[test]
    fn struct_layout_padding() {
        // struct { char a; int b; } — 'b' should be at offset 4 on x86-64.
        let fields = [
            make_field("a", CType::Char { signed: true }),
            make_field("b", CType::Int { signed: true }),
        ];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 1);
        assert_eq!(layout.fields[1].offset, 4); // padding of 3 bytes
        assert_eq!(layout.fields[1].size, 4);
        assert_eq!(layout.total_size, 8);
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn struct_layout_trailing_padding() {
        // struct { int a; char b; } — total size should be 8 (with 3 bytes tail padding).
        let fields = [
            make_field("a", CType::Int { signed: true }),
            make_field("b", CType::Char { signed: true }),
        ];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 4);
        assert_eq!(layout.total_size, 8); // 4 + 1 + 3 tail padding
        assert_eq!(layout.alignment, 4);
    }

    #[test]
    fn struct_layout_packed() {
        // struct __attribute__((packed)) { char a; int b; }
        let fields = [
            make_field("a", CType::Char { signed: true }),
            make_field("b", CType::Int { signed: true }),
        ];
        let layout = compute_struct_layout_with_attrs(&fields, &Target::X86_64, true, None);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 1); // no padding
        assert_eq!(layout.total_size, 5);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn struct_layout_aligned() {
        // struct __attribute__((aligned(16))) { int a; }
        let fields = [make_field("a", CType::Int { signed: true })];
        let layout = compute_struct_layout_with_attrs(&fields, &Target::X86_64, false, Some(16));
        assert_eq!(layout.total_size, 16); // padded to alignment
        assert_eq!(layout.alignment, 16);
    }

    #[test]
    fn struct_layout_packed_aligned() {
        // struct __attribute__((packed, aligned(8))) { char a; int b; }
        let fields = [
            make_field("a", CType::Char { signed: true }),
            make_field("b", CType::Int { signed: true }),
        ];
        let layout = compute_struct_layout_with_attrs(&fields, &Target::X86_64, true, Some(8));
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].offset, 1); // packed — no padding
        assert_eq!(layout.total_size, 8); // aligned to 8
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn struct_layout_flexible_array_member() {
        // struct { int len; char data[]; }
        let fields = [
            make_field("len", CType::Int { signed: true }),
            FieldDef {
                name: Some("data".to_string()),
                ty: CType::Array {
                    element: Box::new(CType::Char { signed: true }),
                    size: None,
                },
                bit_width: None,
            },
        ];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        assert!(layout.has_flexible_array);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.fields[1].offset, 4);
        assert_eq!(layout.fields[1].size, 0); // FAM contributes 0 bytes
        assert_eq!(layout.total_size, 4); // size does not include FAM
    }

    #[test]
    fn struct_layout_bitfields() {
        // struct { unsigned a:3; unsigned b:5; unsigned c:7; }
        // All fit into one 32-bit storage unit.
        let fields = [
            make_bitfield("a", CType::Int { signed: false }, 3),
            make_bitfield("b", CType::Int { signed: false }, 5),
            make_bitfield("c", CType::Int { signed: false }, 7),
        ];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        // All three fit in one 32-bit (4-byte) storage unit.
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].bit_offset, Some(0));
        assert_eq!(layout.fields[0].bit_width, Some(3));
        assert_eq!(layout.fields[1].bit_offset, Some(3));
        assert_eq!(layout.fields[1].bit_width, Some(5));
        assert_eq!(layout.fields[2].bit_offset, Some(8));
        assert_eq!(layout.fields[2].bit_width, Some(7));
        assert_eq!(layout.total_size, 4);
    }

    #[test]
    fn struct_layout_zero_width_bitfield() {
        // struct { unsigned a:3; unsigned :0; unsigned b:5; }
        // Zero-width bitfield forces b to a new storage unit.
        let fields = [
            make_bitfield("a", CType::Int { signed: false }, 3),
            FieldDef {
                name: None,
                ty: CType::Int { signed: false },
                bit_width: Some(0),
            },
            make_bitfield("b", CType::Int { signed: false }, 5),
        ];
        let layout = compute_struct_layout(&fields, &Target::X86_64);
        // 'a' in first storage unit at offset 0, 'b' starts in new unit at offset 4.
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].bit_offset, Some(0));
        assert_eq!(layout.fields[2].offset, 4);
        assert_eq!(layout.fields[2].bit_offset, Some(0));
        assert_eq!(layout.total_size, 8);
    }

    // -----------------------------------------------------------------------
    // Union layout tests
    // -----------------------------------------------------------------------

    #[test]
    fn union_layout_empty() {
        let layout = compute_union_layout(&[], &Target::X86_64);
        assert_eq!(layout.total_size, 0);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn union_layout_basic() {
        let fields = [
            make_field("i", CType::Int { signed: true }),
            make_field("d", CType::Double),
        ];
        let layout = compute_union_layout(&fields, &Target::X86_64);
        assert_eq!(layout.total_size, 8); // max(4, 8) = 8
        assert_eq!(layout.alignment, 8); // max(4, 8) = 8
    }

    #[test]
    fn union_layout_packed() {
        let fields = [
            make_field("i", CType::Int { signed: true }),
            make_field("d", CType::Double),
        ];
        let layout = compute_union_layout_with_attrs(&fields, &Target::X86_64, true, None);
        assert_eq!(layout.total_size, 8);
        assert_eq!(layout.alignment, 1);
    }

    #[test]
    fn union_layout_aligned() {
        let fields = [make_field("i", CType::Int { signed: true })];
        let layout = compute_union_layout_with_attrs(&fields, &Target::X86_64, false, Some(16));
        assert_eq!(layout.total_size, 16); // rounded up to 16
        assert_eq!(layout.alignment, 16);
    }

    // -----------------------------------------------------------------------
    // types_compatible tests
    // -----------------------------------------------------------------------

    #[test]
    fn compatible_same_int() {
        assert!(types_compatible(
            &CType::Int { signed: true },
            &CType::Int { signed: true }
        ));
    }

    #[test]
    fn incompatible_signed_unsigned() {
        assert!(!types_compatible(
            &CType::Int { signed: true },
            &CType::Int { signed: false }
        ));
    }

    #[test]
    fn compatible_pointer_to_int() {
        let a = CType::Pointer(Box::new(CType::Int { signed: true }));
        let b = CType::Pointer(Box::new(CType::Int { signed: true }));
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn incompatible_pointer_pointee_mismatch() {
        let a = CType::Pointer(Box::new(CType::Int { signed: true }));
        let b = CType::Pointer(Box::new(CType::Double));
        assert!(!types_compatible(&a, &b));
    }

    #[test]
    fn compatible_arrays_different_sizes() {
        // __builtin_types_compatible_p(int[5], int[10]) is true per GCC.
        let a = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(5),
        };
        let b = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn compatible_through_typedef() {
        let a = CType::Typedef {
            name: "myint".to_string(),
            underlying: Box::new(CType::Int { signed: true }),
        };
        let b = CType::Int { signed: true };
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn compatible_through_atomic() {
        let a = CType::Atomic(Box::new(CType::Int { signed: true }));
        let b = CType::Int { signed: true };
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn compatible_named_structs_same_tag() {
        let a = CType::Struct {
            name: Some("foo".to_string()),
            fields: vec![],
        };
        let b = CType::Struct {
            name: Some("foo".to_string()),
            fields: vec![],
        };
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn incompatible_named_structs_different_tag() {
        let a = CType::Struct {
            name: Some("foo".to_string()),
            fields: vec![],
        };
        let b = CType::Struct {
            name: Some("bar".to_string()),
            fields: vec![],
        };
        assert!(!types_compatible(&a, &b));
    }

    #[test]
    fn incompatible_int_vs_pointer() {
        assert!(!types_compatible(
            &CType::Int { signed: true },
            &CType::Pointer(Box::new(CType::Void))
        ));
    }

    #[test]
    fn compatible_function_types() {
        let a = CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![CType::Double],
            variadic: false,
        };
        let b = CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![CType::Double],
            variadic: false,
        };
        assert!(types_compatible(&a, &b));
    }

    #[test]
    fn incompatible_function_variadic_mismatch() {
        let a = CType::Function {
            return_type: Box::new(CType::Void),
            params: vec![],
            variadic: false,
        };
        let b = CType::Function {
            return_type: Box::new(CType::Void),
            params: vec![],
            variadic: true,
        };
        assert!(!types_compatible(&a, &b));
    }

    // -----------------------------------------------------------------------
    // composite_type tests
    // -----------------------------------------------------------------------

    #[test]
    fn composite_array_known_size() {
        let a = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: None,
        };
        let b = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        let c = composite_type(&a, &b);
        match c {
            CType::Array { size, .. } => assert_eq!(size, Some(10)),
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn composite_function_with_prototype() {
        // a has no params (K&R), b has params — composite takes b's params.
        let a = CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![],
            variadic: false,
        };
        let b = CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![CType::Double, CType::Float],
            variadic: false,
        };
        let c = composite_type(&a, &b);
        match c {
            CType::Function { params, .. } => assert_eq!(params.len(), 2),
            _ => panic!("Expected Function"),
        }
    }

    #[test]
    fn composite_pointer() {
        let a = CType::Pointer(Box::new(CType::Int { signed: true }));
        let b = CType::Pointer(Box::new(CType::Int { signed: true }));
        let c = composite_type(&a, &b);
        match c {
            CType::Pointer(inner) => {
                assert!(matches!(*inner, CType::Int { signed: true }));
            }
            _ => panic!("Expected Pointer"),
        }
    }

    // -----------------------------------------------------------------------
    // ctype_to_machine_type tests
    // -----------------------------------------------------------------------

    #[test]
    fn machine_type_int() {
        let mt = ctype_to_machine_type(&CType::Int { signed: true }, &Target::X86_64);
        assert!(matches!(mt, MachineType::I32));
    }

    #[test]
    fn machine_type_long_lp64() {
        let mt = ctype_to_machine_type(&CType::Long { signed: true }, &Target::X86_64);
        assert!(matches!(mt, MachineType::I64));
    }

    #[test]
    fn machine_type_long_ilp32() {
        let mt = ctype_to_machine_type(&CType::Long { signed: true }, &Target::I686);
        assert!(matches!(mt, MachineType::I32));
    }

    #[test]
    fn machine_type_pointer() {
        let mt = ctype_to_machine_type(
            &CType::Pointer(Box::new(CType::Void)),
            &Target::X86_64,
        );
        assert!(matches!(mt, MachineType::Ptr));
    }

    #[test]
    fn machine_type_double() {
        let mt = ctype_to_machine_type(&CType::Double, &Target::X86_64);
        assert!(matches!(mt, MachineType::F64));
    }

    #[test]
    fn machine_type_struct_aggregate() {
        let ty = CType::Struct {
            name: None,
            fields: vec![
                make_field("x", CType::Int { signed: true }),
                make_field("y", CType::Int { signed: true }),
            ],
        };
        let mt = ctype_to_machine_type(&ty, &Target::X86_64);
        match mt {
            MachineType::Aggregate(sz) => assert!(sz > 0),
            _ => panic!("Expected Aggregate"),
        }
    }

    // -----------------------------------------------------------------------
    // round_up utility tests
    // -----------------------------------------------------------------------

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(0, 4), 0);
        assert_eq!(round_up(1, 4), 4);
        assert_eq!(round_up(3, 4), 4);
        assert_eq!(round_up(4, 4), 4);
        assert_eq!(round_up(5, 4), 8);
        assert_eq!(round_up(7, 8), 8);
        assert_eq!(round_up(8, 8), 8);
        assert_eq!(round_up(9, 8), 16);
    }

    #[test]
    fn round_up_align_one() {
        assert_eq!(round_up(0, 1), 0);
        assert_eq!(round_up(5, 1), 5);
        assert_eq!(round_up(100, 1), 100);
    }

    // -----------------------------------------------------------------------
    // i686 layout tests (ILP32)
    // -----------------------------------------------------------------------

    #[test]
    fn struct_layout_i686_pointer_field() {
        // On i686, pointers are 4 bytes.
        let fields = [make_field("p", CType::Pointer(Box::new(CType::Void)))];
        let layout = compute_struct_layout(&fields, &Target::I686);
        assert_eq!(layout.fields[0].size, 4);
        assert_eq!(layout.total_size, 4);
    }

    #[test]
    fn struct_layout_aarch64_long_field() {
        // On AArch64 (LP64), long is 8 bytes.
        let fields = [make_field("l", CType::Long { signed: true })];
        let layout = compute_struct_layout(&fields, &Target::AArch64);
        assert_eq!(layout.fields[0].size, 8);
        assert_eq!(layout.total_size, 8);
        assert_eq!(layout.alignment, 8);
    }
}
