//! # IR Type System
//!
//! This module defines BCC's intermediate representation type system — the
//! [`IrType`] enum and supporting [`StructType`] struct.
//!
//! ## Purpose
//!
//! `IrType` bridges C language types (from [`crate::common::types::CType`])
//! to the machine-level types consumed by the backend during code generation.
//! IR types are deliberately simpler and more uniform than C types:
//!
//! - **No qualifiers** — `const`, `volatile`, `restrict`, `_Atomic` are
//!   stripped during lowering; they have no effect on data layout.
//! - **No typedef indirection** — typedefs are resolved to their underlying
//!   type before conversion.
//! - **Opaque pointers** — all pointer types collapse to a single [`IrType::Ptr`]
//!   variant (LLVM-style opaque pointer), eliminating pointee-type tracking
//!   at the IR level.
//! - **Fixed-width integers** — C's `int`, `long`, etc. are resolved to
//!   explicit bit widths (I8, I16, I32, I64) based on the target architecture.
//!
//! ## Supported Types
//!
//! | Category    | Variants                                      |
//! |-------------|-----------------------------------------------|
//! | Void        | `Void`                                        |
//! | Integer     | `I1`, `I8`, `I16`, `I32`, `I64`, `I128`       |
//! | Float       | `F32`, `F64`, `F80`                           |
//! | Pointer     | `Ptr` (opaque)                                |
//! | Composite   | `Array(elem, count)`, `Struct(StructType)`    |
//! | Callable    | `Function(ret, params)`                       |
//!
//! ## Usage
//!
//! - **Phase 6 (AST-to-IR lowering):** Use [`IrType::from_ctype`] to convert
//!   C types to IR types during IR construction.
//! - **Middle-end passes:** IR types annotate every instruction, basic block,
//!   and function in the IR module.
//! - **Phase 10 (code generation):** Backend reads IR types to select
//!   instructions, allocate registers, and compute ABI-correct layouts.
//! - **[`IrType::ptr_int`]** returns the pointer-width integer type for the
//!   current target (I32 on i686, I64 on 64-bit targets).

use std::fmt;

use crate::common::target::Target;
use crate::common::types::{alignof_ctype, sizeof_ctype, CType};

// ===========================================================================
// StructType — Aggregate field layout
// ===========================================================================

/// Representation of an IR-level struct type.
///
/// Structs are ordered sequences of typed fields, optionally packed (no
/// inter-field alignment padding). An optional `name` is carried for
/// debug output and DWARF generation but has no semantic effect on layout.
///
/// # Packed structs
///
/// When `packed` is `true`, all fields are laid out contiguously with
/// alignment 1, mirroring C's `__attribute__((packed))`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    /// Ordered list of field types.
    pub fields: Vec<IrType>,
    /// Whether this struct uses packed layout (`__attribute__((packed))`).
    pub packed: bool,
    /// Optional human-readable name (for debug/display, not for identity).
    pub name: Option<String>,
}

impl StructType {
    /// Create a new struct type with the given fields and packing mode.
    #[inline]
    pub fn new(fields: Vec<IrType>, packed: bool) -> Self {
        Self {
            fields,
            packed,
            name: None,
        }
    }

    /// Create a named struct type.
    #[inline]
    pub fn with_name(fields: Vec<IrType>, packed: bool, name: String) -> Self {
        Self {
            fields,
            packed,
            name: Some(name),
        }
    }
}

impl fmt::Display for StructType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.packed {
            write!(f, "packed {{ ")?;
        } else {
            write!(f, "{{ ")?;
        }
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", field)?;
        }
        write!(f, " }}")
    }
}

// ===========================================================================
// IrType — Core IR type enum
// ===========================================================================

/// The IR-level type system for BCC.
///
/// Every IR instruction, value, and function is annotated with an `IrType`.
/// These types are simpler than their C-language counterparts: qualifiers
/// are stripped, typedefs are resolved, and all pointers are opaque.
///
/// # Integer types
///
/// - [`I1`](IrType::I1) is a 1-bit boolean (comparison results, branch
///   conditions). Stored as a full byte in memory.
/// - [`I8`](IrType::I8) through [`I128`](IrType::I128) are fixed-width
///   integers.
///
/// # Floating-point types
///
/// - [`F32`](IrType::F32) — IEEE 754 single-precision.
/// - [`F64`](IrType::F64) — IEEE 754 double-precision.
/// - [`F80`](IrType::F80) — 80-bit x87 extended precision (`long double`
///   on x86 platforms). Stored in 12 bytes on i686, 16 bytes on x86-64.
///
/// # Opaque pointer
///
/// [`Ptr`](IrType::Ptr) is a single, type-less pointer (like LLVM's opaque
/// `ptr`). Pointer arithmetic and dereference widths are determined by the
/// instruction context, not by the pointer type itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// Void type — used for function return types and `void*` conversions.
    Void,
    /// 1-bit integer — boolean, produced by comparison instructions.
    I1,
    /// 8-bit integer (`char`, `signed char`, `unsigned char`).
    I8,
    /// 16-bit integer (`short`, `unsigned short`).
    I16,
    /// 32-bit integer (`int`, `unsigned int`, `long` on ILP32).
    I32,
    /// 64-bit integer (`long long`, `long` on LP64).
    I64,
    /// 128-bit integer (`__int128`).
    I128,
    /// 32-bit IEEE 754 float (`float`).
    F32,
    /// 64-bit IEEE 754 double (`double`).
    F64,
    /// 80-bit extended precision (`long double` on x86).
    F80,
    /// Opaque pointer type (LLVM-style; no pointee type tracking).
    Ptr,
    /// Fixed-size array: element type and element count.
    Array(Box<IrType>, usize),
    /// Struct: ordered list of member types, optionally packed.
    Struct(StructType),
    /// Function type: return type and parameter types.
    Function(Box<IrType>, Vec<IrType>),
}

// ===========================================================================
// Size and Alignment Queries
// ===========================================================================

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1). If `align` is 0, returns `value`
/// unchanged to avoid division by zero.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

impl IrType {
    /// Returns the storage size of this type in bytes for the given target.
    ///
    /// # Storage semantics
    ///
    /// - [`I1`](IrType::I1) is stored as 1 byte (not 1 bit).
    /// - [`F80`](IrType::F80) occupies 16 bytes on 64-bit targets and
    ///   12 bytes on 32-bit targets (matching ABI padding requirements).
    /// - [`Ptr`](IrType::Ptr) size is target-dependent (4 on ILP32, 8 on LP64).
    /// - [`Void`](IrType::Void) and [`Function`](IrType::Function) types
    ///   have size 0 (they cannot be instantiated as values).
    /// - Struct sizes include inter-field alignment padding and trailing
    ///   padding to the struct's overall alignment.
    pub fn size_bytes(&self, target: &Target) -> usize {
        match self {
            IrType::Void => 0,
            IrType::I1 => 1,
            IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => 8,
            IrType::I128 => 16,
            IrType::F32 => 4,
            IrType::F64 => 8,
            IrType::F80 => {
                // 80-bit extended precision: padded to 16 on 64-bit, 12 on 32-bit.
                if target.is_64bit() {
                    16
                } else {
                    12
                }
            }
            IrType::Ptr => target.pointer_width(),
            IrType::Array(elem, count) => elem.size_bytes(target) * count,
            IrType::Struct(st) => compute_struct_size(st, target),
            IrType::Function(_, _) => 0,
        }
    }

    /// Returns the required alignment of this type in bytes for the given target.
    ///
    /// The alignment is always a power of two and ≥ 1.
    ///
    /// # Target-dependent alignment
    ///
    /// - On i686 (ILP32), `I64`/`F64` have 4-byte alignment (System V i386 ABI).
    /// - On 64-bit targets (LP64), `I64`/`F64` have 8-byte alignment.
    /// - `F80` aligns to 16 bytes on 64-bit, 4 bytes on i686.
    /// - `I128` always aligns to 16 bytes.
    /// - Packed structs have alignment 1 regardless of field types.
    pub fn align_bytes(&self, target: &Target) -> usize {
        match self {
            IrType::Void => 1,
            IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 => {
                if target.is_64bit() {
                    8
                } else {
                    4
                }
            }
            IrType::I128 => 16,
            IrType::F80 => {
                if target.is_64bit() {
                    16
                } else {
                    4
                }
            }
            IrType::Ptr => target.pointer_width(),
            IrType::Array(elem, _) => elem.align_bytes(target),
            IrType::Struct(st) => {
                if st.packed {
                    1
                } else {
                    st.fields
                        .iter()
                        .map(|f| f.align_bytes(target))
                        .max()
                        .unwrap_or(1)
                }
            }
            IrType::Function(_, _) => 1,
        }
    }
}

/// Compute the total size of a struct including inter-field alignment
/// padding and trailing padding to the struct's overall alignment.
fn compute_struct_size(st: &StructType, target: &Target) -> usize {
    if st.fields.is_empty() {
        return 0;
    }

    let mut offset: usize = 0;
    let mut max_align: usize = 1;

    for field in &st.fields {
        let field_size = field.size_bytes(target);
        let field_align = if st.packed {
            1
        } else {
            field.align_bytes(target)
        };

        // Align the current offset to the field's alignment.
        offset = align_up(offset, field_align);
        // Advance past this field.
        offset += field_size;

        if field_align > max_align {
            max_align = field_align;
        }
    }

    // Pad the struct to a multiple of its overall alignment.
    align_up(offset, max_align)
}

// ===========================================================================
// Type Classification Predicates
// ===========================================================================

impl IrType {
    /// Returns `true` if this is the void type.
    #[inline]
    pub fn is_void(&self) -> bool {
        matches!(self, IrType::Void)
    }

    /// Returns `true` if this is an integer type (I1 through I128).
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128
        )
    }

    /// Returns `true` if this is a floating-point type (F32, F64, or F80).
    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64 | IrType::F80)
    }

    /// Returns `true` if this is the opaque pointer type.
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(self, IrType::Ptr)
    }

    /// Returns `true` if this is an array type.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, IrType::Array(..))
    }

    /// Returns `true` if this is a struct type.
    #[inline]
    pub fn is_struct(&self) -> bool {
        matches!(self, IrType::Struct(..))
    }

    /// Returns `true` if this is a function type.
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self, IrType::Function(..))
    }

    /// Returns `true` if this is a scalar type (integer, float, or pointer).
    ///
    /// Scalar types fit in a single register and are the primary operands
    /// of arithmetic and comparison instructions.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_integer() || self.is_float() || self.is_pointer()
    }

    /// Returns `true` if this is an aggregate type (array or struct).
    ///
    /// Aggregate types are composed of multiple elements and are typically
    /// accessed via `GetElementPtr` or decomposed during lowering.
    #[inline]
    pub fn is_aggregate(&self) -> bool {
        self.is_array() || self.is_struct()
    }

    /// Returns the bit width of an integer type.
    ///
    /// # Panics
    ///
    /// Panics if called on a non-integer type.
    pub fn int_width(&self) -> u32 {
        match self {
            IrType::I1 => 1,
            IrType::I8 => 8,
            IrType::I16 => 16,
            IrType::I32 => 32,
            IrType::I64 => 64,
            IrType::I128 => 128,
            other => panic!("int_width called on non-integer type: {:?}", other),
        }
    }
}

// ===========================================================================
// C Type → IR Type Conversion
// ===========================================================================

impl IrType {
    /// Convert a C language type ([`CType`]) to its IR representation.
    ///
    /// This conversion is the primary bridge from the frontend type system
    /// to the middle-end IR, invoked during Phase 6 (AST-to-IR lowering).
    ///
    /// # Conversion rules
    ///
    /// - **Qualifiers** (`const`, `volatile`, `restrict`, `_Atomic`) are
    ///   stripped — they have no effect on IR data layout.
    /// - **Typedefs** are resolved to their underlying type.
    /// - **Enums** resolve to their underlying integer type.
    /// - **Pointers** all become opaque [`IrType::Ptr`].
    /// - **Unions** are represented as byte arrays (`[N x i8]`) where N is
    ///   the union's total size including alignment padding.
    /// - **`_Complex`** types become arrays of two base-type elements.
    /// - **`long`** maps to I32 on ILP32 (i686) or I64 on LP64 (64-bit).
    ///
    /// # Arguments
    ///
    /// * `ctype` — The C type to convert.
    /// * `target` — The target architecture (for size/alignment resolution).
    pub fn from_ctype(ctype: &CType, target: &Target) -> IrType {
        match ctype {
            // ----- Void -----
            CType::Void => IrType::Void,

            // ----- Boolean -----
            CType::Bool => IrType::I1,

            // ----- Character types (all 8-bit) -----
            CType::Char | CType::SChar | CType::UChar => IrType::I8,

            // ----- Short (16-bit) -----
            CType::Short | CType::UShort => IrType::I16,

            // ----- Int (32-bit on all targets) -----
            CType::Int | CType::UInt => IrType::I32,

            // ----- Long (target-dependent: 32-bit on ILP32, 64-bit on LP64) -----
            CType::Long | CType::ULong => {
                if target.is_64bit() {
                    IrType::I64
                } else {
                    IrType::I32
                }
            }

            // ----- Long long (always 64-bit) -----
            CType::LongLong | CType::ULongLong => IrType::I64,

            // ----- __int128 (always 128-bit) -----
            CType::Int128 | CType::UInt128 => IrType::I128,

            // ----- Floating-point -----
            CType::Float => IrType::F32,
            CType::Double => IrType::F64,
            CType::LongDouble => IrType::F80,

            // ----- Complex: pair of base floating-point type -----
            CType::Complex(base) => {
                let base_ir = IrType::from_ctype(base, target);
                IrType::Array(Box::new(base_ir), 2)
            }

            // ----- Pointer: all pointers collapse to opaque Ptr -----
            CType::Pointer(_, _) => IrType::Ptr,

            // ----- Array: element type + count -----
            CType::Array(elem, size) => {
                let elem_ir = IrType::from_ctype(elem, target);
                IrType::Array(Box::new(elem_ir), size.unwrap_or(0))
            }

            // ----- Function: return type + parameter types -----
            CType::Function {
                return_type,
                params,
                ..
            } => {
                let ret = IrType::from_ctype(return_type, target);
                let param_types: Vec<IrType> = params
                    .iter()
                    .map(|p| IrType::from_ctype(p, target))
                    .collect();
                IrType::Function(Box::new(ret), param_types)
            }

            // ----- Struct: convert each field type -----
            CType::Struct {
                name,
                fields,
                packed,
                ..
            } => {
                // For structs with bitfields, the IR field types don't
                // match the actual byte layout because multiple bitfields
                // pack into one allocation unit.  Compute the actual sizeof
                // from the C type system and build an IR struct with the
                // correct number of allocation-unit-sized fields.  This
                // preserves `is_struct() == true` for ABI classification
                // while ensuring `size_bytes()` returns the correct value.
                let has_bitfield = fields.iter().any(|f| f.bit_width.is_some());
                if has_bitfield {
                    let struct_size = sizeof_ctype(ctype, target);
                    let struct_align = alignof_ctype(ctype, target);
                    let (unit_ty, unit_size) = if struct_align >= 8 && struct_size % 8 == 0 {
                        (IrType::I64, 8)
                    } else if struct_align >= 4 && struct_size % 4 == 0 {
                        (IrType::I32, 4)
                    } else if struct_align >= 2 && struct_size % 2 == 0 {
                        (IrType::I16, 2)
                    } else {
                        (IrType::I8, 1)
                    };
                    let unit_count = if unit_size > 0 && struct_size > 0 {
                        struct_size / unit_size
                    } else {
                        0
                    };
                    IrType::Struct(StructType {
                        fields: vec![unit_ty; unit_count],
                        packed: *packed,
                        name: name.clone(),
                    })
                } else {
                    let field_types: Vec<IrType> = fields
                        .iter()
                        .map(|f| IrType::from_ctype(&f.ty, target))
                        .collect();
                    let mut st = StructType {
                        fields: field_types,
                        packed: *packed,
                        name: name.clone(),
                    };
                    // Check if C-level alignment (e.g. __attribute__((aligned(N))))
                    // makes the C sizeof larger than the IR struct's natural size.
                    // If so, add a tail-padding array to match the C sizeof.
                    // This prevents stack corruption when the alloca uses the IR
                    // type's size_bytes() — it must be at least as large as the
                    // C type's sizeof.
                    let ir_struct_tmp = IrType::Struct(st.clone());
                    let ir_sz = ir_struct_tmp.size_bytes(target);
                    let c_sz = sizeof_ctype(ctype, target);
                    if c_sz > ir_sz {
                        let pad = c_sz - ir_sz;
                        st.fields.push(IrType::Array(Box::new(IrType::I8), pad));
                    }
                    IrType::Struct(st)
                }
            }

            // ----- Union: represented as array of properly-aligned elements -----
            CType::Union { .. } => {
                // Compute the union's total size (including alignment padding)
                // from the C type system for accuracy with bitfields and
                // packed/aligned attributes.
                let union_size = sizeof_ctype(ctype, target);
                // Use the union's natural alignment to choose the IR base
                // element type.  `Array(I8, N)` would lose alignment info,
                // causing enclosing structs to lay out fields at wrong offsets.
                let union_align = alignof_ctype(ctype, target);
                if union_align >= 8 && union_size % 8 == 0 {
                    IrType::Array(Box::new(IrType::I64), union_size / 8)
                } else if union_align >= 4 && union_size % 4 == 0 {
                    IrType::Array(Box::new(IrType::I32), union_size / 4)
                } else if union_align >= 2 && union_size % 2 == 0 {
                    IrType::Array(Box::new(IrType::I16), union_size / 2)
                } else {
                    IrType::Array(Box::new(IrType::I8), union_size)
                }
            }

            // ----- Enum: resolve to underlying integer type -----
            CType::Enum {
                underlying_type, ..
            } => IrType::from_ctype(underlying_type, target),

            // ----- Atomic: strip atomic qualifier, convert inner type -----
            CType::Atomic(inner) => IrType::from_ctype(inner, target),

            // ----- Typedef: resolve to underlying type -----
            CType::Typedef { underlying, .. } => IrType::from_ctype(underlying, target),

            // ----- Qualified: strip qualifiers, convert inner type -----
            CType::Qualified(inner, _) => IrType::from_ctype(inner, target),
        }
    }

    /// Returns the pointer-width integer type for the given target.
    ///
    /// - 64-bit targets (x86-64, AArch64, RISC-V 64): [`IrType::I64`]
    /// - 32-bit targets (i686): [`IrType::I32`]
    ///
    /// This is commonly used for `intptr_t`, `uintptr_t`, `size_t`, and
    /// `ptrdiff_t` representation in the IR.
    #[inline]
    pub fn ptr_int(target: &Target) -> IrType {
        if target.is_64bit() {
            IrType::I64
        } else {
            IrType::I32
        }
    }
}

// ===========================================================================
// Display Implementation
// ===========================================================================

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::I1 => write!(f, "i1"),
            IrType::I8 => write!(f, "i8"),
            IrType::I16 => write!(f, "i16"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::I128 => write!(f, "i128"),
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::F80 => write!(f, "f80"),
            IrType::Ptr => write!(f, "ptr"),
            IrType::Array(elem, count) => write!(f, "[{} x {}]", count, elem),
            IrType::Struct(st) => write!(f, "{}", st),
            IrType::Function(ret, params) => {
                write!(f, "{} (", ret)?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::target::Target;
    use crate::common::types::{CType, StructField, TypeQualifiers};

    // -- IrType variant existence -------------------------------------------

    #[test]
    fn all_variants_constructible() {
        let _ = IrType::Void;
        let _ = IrType::I1;
        let _ = IrType::I8;
        let _ = IrType::I16;
        let _ = IrType::I32;
        let _ = IrType::I64;
        let _ = IrType::I128;
        let _ = IrType::F32;
        let _ = IrType::F64;
        let _ = IrType::F80;
        let _ = IrType::Ptr;
        let _ = IrType::Array(Box::new(IrType::I32), 10);
        let _ = IrType::Struct(StructType::new(vec![IrType::I32, IrType::I64], false));
        let _ = IrType::Function(Box::new(IrType::I32), vec![IrType::Ptr, IrType::I32]);
    }

    // -- StructType ---------------------------------------------------------

    #[test]
    fn struct_type_construction() {
        let st = StructType::new(vec![IrType::I32, IrType::I8], true);
        assert!(st.packed);
        assert!(st.name.is_none());
        assert_eq!(st.fields.len(), 2);

        let named = StructType::with_name(vec![IrType::Ptr], false, "point".to_string());
        assert!(!named.packed);
        assert_eq!(named.name.as_deref(), Some("point"));
    }

    // -- size_bytes ---------------------------------------------------------

    #[test]
    fn size_bytes_scalars_64bit() {
        let t = Target::X86_64;
        assert_eq!(IrType::Void.size_bytes(&t), 0);
        assert_eq!(IrType::I1.size_bytes(&t), 1);
        assert_eq!(IrType::I8.size_bytes(&t), 1);
        assert_eq!(IrType::I16.size_bytes(&t), 2);
        assert_eq!(IrType::I32.size_bytes(&t), 4);
        assert_eq!(IrType::I64.size_bytes(&t), 8);
        assert_eq!(IrType::I128.size_bytes(&t), 16);
        assert_eq!(IrType::F32.size_bytes(&t), 4);
        assert_eq!(IrType::F64.size_bytes(&t), 8);
        assert_eq!(IrType::F80.size_bytes(&t), 16);
        assert_eq!(IrType::Ptr.size_bytes(&t), 8);
    }

    #[test]
    fn size_bytes_scalars_32bit() {
        let t = Target::I686;
        assert_eq!(IrType::F80.size_bytes(&t), 12);
        assert_eq!(IrType::Ptr.size_bytes(&t), 4);
        assert_eq!(IrType::I64.size_bytes(&t), 8);
    }

    #[test]
    fn size_bytes_array() {
        let t = Target::X86_64;
        let arr = IrType::Array(Box::new(IrType::I32), 10);
        assert_eq!(arr.size_bytes(&t), 40);
    }

    #[test]
    fn size_bytes_empty_array() {
        let t = Target::X86_64;
        let arr = IrType::Array(Box::new(IrType::I32), 0);
        assert_eq!(arr.size_bytes(&t), 0);
    }

    #[test]
    fn size_bytes_struct_with_padding() {
        let t = Target::X86_64;
        // struct { i8, i32 } — padding after i8 to align i32
        let st = StructType::new(vec![IrType::I8, IrType::I32], false);
        let ty = IrType::Struct(st);
        // i8 at offset 0, padding 3 bytes, i32 at offset 4, total = 8
        assert_eq!(ty.size_bytes(&t), 8);
    }

    #[test]
    fn size_bytes_packed_struct() {
        let t = Target::X86_64;
        // packed struct { i8, i32 } — no padding
        let st = StructType::new(vec![IrType::I8, IrType::I32], true);
        let ty = IrType::Struct(st);
        assert_eq!(ty.size_bytes(&t), 5);
    }

    #[test]
    fn size_bytes_empty_struct() {
        let t = Target::X86_64;
        let st = StructType::new(vec![], false);
        let ty = IrType::Struct(st);
        assert_eq!(ty.size_bytes(&t), 0);
    }

    #[test]
    fn size_bytes_function_is_zero() {
        let t = Target::X86_64;
        let func = IrType::Function(Box::new(IrType::I32), vec![IrType::Ptr]);
        assert_eq!(func.size_bytes(&t), 0);
    }

    // -- align_bytes --------------------------------------------------------

    #[test]
    fn align_bytes_scalars_64bit() {
        let t = Target::X86_64;
        assert_eq!(IrType::Void.align_bytes(&t), 1);
        assert_eq!(IrType::I1.align_bytes(&t), 1);
        assert_eq!(IrType::I8.align_bytes(&t), 1);
        assert_eq!(IrType::I16.align_bytes(&t), 2);
        assert_eq!(IrType::I32.align_bytes(&t), 4);
        assert_eq!(IrType::I64.align_bytes(&t), 8);
        assert_eq!(IrType::I128.align_bytes(&t), 16);
        assert_eq!(IrType::F32.align_bytes(&t), 4);
        assert_eq!(IrType::F64.align_bytes(&t), 8);
        assert_eq!(IrType::F80.align_bytes(&t), 16);
        assert_eq!(IrType::Ptr.align_bytes(&t), 8);
    }

    #[test]
    fn align_bytes_scalars_32bit() {
        let t = Target::I686;
        assert_eq!(IrType::I64.align_bytes(&t), 4);
        assert_eq!(IrType::F64.align_bytes(&t), 4);
        assert_eq!(IrType::F80.align_bytes(&t), 4);
        assert_eq!(IrType::Ptr.align_bytes(&t), 4);
    }

    #[test]
    fn align_bytes_array_inherits_element() {
        let t = Target::X86_64;
        let arr = IrType::Array(Box::new(IrType::I64), 3);
        assert_eq!(arr.align_bytes(&t), 8);
    }

    #[test]
    fn align_bytes_struct() {
        let t = Target::X86_64;
        let st = StructType::new(vec![IrType::I8, IrType::I64], false);
        let ty = IrType::Struct(st);
        assert_eq!(ty.align_bytes(&t), 8); // max(1, 8) = 8
    }

    #[test]
    fn align_bytes_packed_struct() {
        let t = Target::X86_64;
        let st = StructType::new(vec![IrType::I8, IrType::I64], true);
        let ty = IrType::Struct(st);
        assert_eq!(ty.align_bytes(&t), 1); // packed → 1
    }

    #[test]
    fn align_bytes_empty_struct() {
        let t = Target::X86_64;
        let st = StructType::new(vec![], false);
        let ty = IrType::Struct(st);
        assert_eq!(ty.align_bytes(&t), 1); // max of empty = 1
    }

    // -- Type predicates ----------------------------------------------------

    #[test]
    fn predicate_is_void() {
        assert!(IrType::Void.is_void());
        assert!(!IrType::I32.is_void());
    }

    #[test]
    fn predicate_is_integer() {
        assert!(IrType::I1.is_integer());
        assert!(IrType::I8.is_integer());
        assert!(IrType::I16.is_integer());
        assert!(IrType::I32.is_integer());
        assert!(IrType::I64.is_integer());
        assert!(IrType::I128.is_integer());
        assert!(!IrType::F32.is_integer());
        assert!(!IrType::Ptr.is_integer());
        assert!(!IrType::Void.is_integer());
    }

    #[test]
    fn predicate_is_float() {
        assert!(IrType::F32.is_float());
        assert!(IrType::F64.is_float());
        assert!(IrType::F80.is_float());
        assert!(!IrType::I32.is_float());
        assert!(!IrType::Ptr.is_float());
    }

    #[test]
    fn predicate_is_pointer() {
        assert!(IrType::Ptr.is_pointer());
        assert!(!IrType::I64.is_pointer());
    }

    #[test]
    fn predicate_is_array() {
        assert!(IrType::Array(Box::new(IrType::I8), 5).is_array());
        assert!(!IrType::Ptr.is_array());
    }

    #[test]
    fn predicate_is_struct() {
        let st = StructType::new(vec![IrType::I32], false);
        assert!(IrType::Struct(st).is_struct());
        assert!(!IrType::I32.is_struct());
    }

    #[test]
    fn predicate_is_function() {
        assert!(IrType::Function(Box::new(IrType::Void), vec![]).is_function());
        assert!(!IrType::Ptr.is_function());
    }

    #[test]
    fn predicate_is_scalar() {
        assert!(IrType::I32.is_scalar());
        assert!(IrType::F64.is_scalar());
        assert!(IrType::Ptr.is_scalar());
        assert!(!IrType::Void.is_scalar());
        assert!(!IrType::Array(Box::new(IrType::I8), 1).is_scalar());
    }

    #[test]
    fn predicate_is_aggregate() {
        assert!(IrType::Array(Box::new(IrType::I8), 4).is_aggregate());
        let st = StructType::new(vec![IrType::I32], false);
        assert!(IrType::Struct(st).is_aggregate());
        assert!(!IrType::I32.is_aggregate());
        assert!(!IrType::Ptr.is_aggregate());
    }

    // -- int_width ----------------------------------------------------------

    #[test]
    fn int_width_values() {
        assert_eq!(IrType::I1.int_width(), 1);
        assert_eq!(IrType::I8.int_width(), 8);
        assert_eq!(IrType::I16.int_width(), 16);
        assert_eq!(IrType::I32.int_width(), 32);
        assert_eq!(IrType::I64.int_width(), 64);
        assert_eq!(IrType::I128.int_width(), 128);
    }

    #[test]
    #[should_panic(expected = "int_width called on non-integer type")]
    fn int_width_panics_on_float() {
        IrType::F64.int_width();
    }

    #[test]
    #[should_panic(expected = "int_width called on non-integer type")]
    fn int_width_panics_on_ptr() {
        IrType::Ptr.int_width();
    }

    // -- from_ctype ---------------------------------------------------------

    #[test]
    fn from_ctype_basic_types() {
        let t64 = Target::X86_64;
        assert_eq!(IrType::from_ctype(&CType::Void, &t64), IrType::Void);
        assert_eq!(IrType::from_ctype(&CType::Bool, &t64), IrType::I1);
        assert_eq!(IrType::from_ctype(&CType::Char, &t64), IrType::I8);
        assert_eq!(IrType::from_ctype(&CType::SChar, &t64), IrType::I8);
        assert_eq!(IrType::from_ctype(&CType::UChar, &t64), IrType::I8);
        assert_eq!(IrType::from_ctype(&CType::Short, &t64), IrType::I16);
        assert_eq!(IrType::from_ctype(&CType::UShort, &t64), IrType::I16);
        assert_eq!(IrType::from_ctype(&CType::Int, &t64), IrType::I32);
        assert_eq!(IrType::from_ctype(&CType::UInt, &t64), IrType::I32);
        assert_eq!(IrType::from_ctype(&CType::Float, &t64), IrType::F32);
        assert_eq!(IrType::from_ctype(&CType::Double, &t64), IrType::F64);
        assert_eq!(IrType::from_ctype(&CType::LongDouble, &t64), IrType::F80);
    }

    #[test]
    fn from_ctype_long_target_dependent() {
        // 64-bit: Long -> I64
        assert_eq!(
            IrType::from_ctype(&CType::Long, &Target::X86_64),
            IrType::I64
        );
        assert_eq!(
            IrType::from_ctype(&CType::ULong, &Target::AArch64),
            IrType::I64
        );
        assert_eq!(
            IrType::from_ctype(&CType::Long, &Target::RiscV64),
            IrType::I64
        );
        // 32-bit: Long -> I32
        assert_eq!(IrType::from_ctype(&CType::Long, &Target::I686), IrType::I32);
        assert_eq!(
            IrType::from_ctype(&CType::ULong, &Target::I686),
            IrType::I32
        );
    }

    #[test]
    fn from_ctype_long_long() {
        assert_eq!(
            IrType::from_ctype(&CType::LongLong, &Target::X86_64),
            IrType::I64
        );
        assert_eq!(
            IrType::from_ctype(&CType::ULongLong, &Target::I686),
            IrType::I64
        );
    }

    #[test]
    fn from_ctype_pointer_is_opaque() {
        let ptr_to_int = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        assert_eq!(
            IrType::from_ctype(&ptr_to_int, &Target::X86_64),
            IrType::Ptr
        );

        let ptr_to_ptr = CType::Pointer(
            Box::new(CType::Pointer(
                Box::new(CType::Void),
                TypeQualifiers::default(),
            )),
            TypeQualifiers::default(),
        );
        assert_eq!(
            IrType::from_ctype(&ptr_to_ptr, &Target::X86_64),
            IrType::Ptr
        );
    }

    #[test]
    fn from_ctype_array() {
        let arr = CType::Array(Box::new(CType::Int), Some(10));
        let ir = IrType::from_ctype(&arr, &Target::X86_64);
        assert_eq!(ir, IrType::Array(Box::new(IrType::I32), 10));
    }

    #[test]
    fn from_ctype_incomplete_array() {
        let arr = CType::Array(Box::new(CType::Char), None);
        let ir = IrType::from_ctype(&arr, &Target::X86_64);
        assert_eq!(ir, IrType::Array(Box::new(IrType::I8), 0));
    }

    #[test]
    fn from_ctype_function() {
        let func = CType::Function {
            return_type: Box::new(CType::Int),
            params: vec![
                CType::Int,
                CType::Pointer(Box::new(CType::Char), TypeQualifiers::default()),
            ],
            variadic: true,
        };
        let ir = IrType::from_ctype(&func, &Target::X86_64);
        assert_eq!(
            ir,
            IrType::Function(Box::new(IrType::I32), vec![IrType::I32, IrType::Ptr])
        );
    }

    #[test]
    fn from_ctype_struct() {
        let cstruct = CType::Struct {
            name: Some("point".to_string()),
            fields: vec![
                StructField {
                    name: Some("x".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("y".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let ir = IrType::from_ctype(&cstruct, &Target::X86_64);
        match &ir {
            IrType::Struct(st) => {
                assert_eq!(st.fields, vec![IrType::I32, IrType::I32]);
                assert!(!st.packed);
                assert_eq!(st.name.as_deref(), Some("point"));
            }
            other => panic!("Expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn from_ctype_packed_struct() {
        let cstruct = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: None,
                    ty: CType::Char,
                    bit_width: None,
                },
                StructField {
                    name: None,
                    ty: CType::Int,
                    bit_width: None,
                },
            ],
            packed: true,
            aligned: None,
        };
        let ir = IrType::from_ctype(&cstruct, &Target::X86_64);
        match &ir {
            IrType::Struct(st) => {
                assert!(st.packed);
                assert_eq!(st.fields, vec![IrType::I8, IrType::I32]);
            }
            other => panic!("Expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn from_ctype_union_becomes_byte_array() {
        let cunion = CType::Union {
            name: Some("data".to_string()),
            fields: vec![
                StructField {
                    name: Some("i".to_string()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("d".to_string()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let ir = IrType::from_ctype(&cunion, &Target::X86_64);
        // Union size = max(sizeof(int)=4, sizeof(double)=8) = 8, aligned to 8.
        // On 64-bit targets, the union has 8-byte alignment, so it becomes
        // Array(I64, 1) to preserve that alignment in enclosing struct layouts.
        assert_eq!(ir, IrType::Array(Box::new(IrType::I64), 1));
    }

    #[test]
    fn from_ctype_complex() {
        let complex_float = CType::Complex(Box::new(CType::Float));
        let ir = IrType::from_ctype(&complex_float, &Target::X86_64);
        assert_eq!(ir, IrType::Array(Box::new(IrType::F32), 2));

        let complex_double = CType::Complex(Box::new(CType::Double));
        let ir = IrType::from_ctype(&complex_double, &Target::X86_64);
        assert_eq!(ir, IrType::Array(Box::new(IrType::F64), 2));
    }

    #[test]
    fn from_ctype_enum() {
        let cenum = CType::Enum {
            name: Some("color".to_string()),
            underlying_type: Box::new(CType::Int),
        };
        let ir = IrType::from_ctype(&cenum, &Target::X86_64);
        assert_eq!(ir, IrType::I32);
    }

    #[test]
    fn from_ctype_atomic() {
        let atomic_int = CType::Atomic(Box::new(CType::Int));
        let ir = IrType::from_ctype(&atomic_int, &Target::X86_64);
        assert_eq!(ir, IrType::I32);
    }

    #[test]
    fn from_ctype_typedef() {
        let typedef = CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::ULong),
        };
        assert_eq!(IrType::from_ctype(&typedef, &Target::X86_64), IrType::I64);
        assert_eq!(IrType::from_ctype(&typedef, &Target::I686), IrType::I32);
    }

    #[test]
    fn from_ctype_qualified_strips_qualifiers() {
        let const_int = CType::Qualified(
            Box::new(CType::Int),
            TypeQualifiers {
                is_const: true,
                is_volatile: false,
                is_restrict: false,
                is_atomic: false,
            },
        );
        assert_eq!(IrType::from_ctype(&const_int, &Target::X86_64), IrType::I32);
    }

    // -- ptr_int ------------------------------------------------------------

    #[test]
    fn ptr_int_64bit() {
        assert_eq!(IrType::ptr_int(&Target::X86_64), IrType::I64);
        assert_eq!(IrType::ptr_int(&Target::AArch64), IrType::I64);
        assert_eq!(IrType::ptr_int(&Target::RiscV64), IrType::I64);
    }

    #[test]
    fn ptr_int_32bit() {
        assert_eq!(IrType::ptr_int(&Target::I686), IrType::I32);
    }

    // -- Display ------------------------------------------------------------

    #[test]
    fn display_scalar_types() {
        assert_eq!(format!("{}", IrType::Void), "void");
        assert_eq!(format!("{}", IrType::I1), "i1");
        assert_eq!(format!("{}", IrType::I8), "i8");
        assert_eq!(format!("{}", IrType::I16), "i16");
        assert_eq!(format!("{}", IrType::I32), "i32");
        assert_eq!(format!("{}", IrType::I64), "i64");
        assert_eq!(format!("{}", IrType::I128), "i128");
        assert_eq!(format!("{}", IrType::F32), "f32");
        assert_eq!(format!("{}", IrType::F64), "f64");
        assert_eq!(format!("{}", IrType::F80), "f80");
        assert_eq!(format!("{}", IrType::Ptr), "ptr");
    }

    #[test]
    fn display_array() {
        let arr = IrType::Array(Box::new(IrType::I32), 10);
        assert_eq!(format!("{}", arr), "[10 x i32]");
    }

    #[test]
    fn display_struct() {
        let st = StructType::new(vec![IrType::I32, IrType::I64, IrType::Ptr], false);
        let ty = IrType::Struct(st);
        assert_eq!(format!("{}", ty), "{ i32, i64, ptr }");
    }

    #[test]
    fn display_packed_struct() {
        let st = StructType::new(vec![IrType::I8, IrType::I8], true);
        let ty = IrType::Struct(st);
        assert_eq!(format!("{}", ty), "packed { i8, i8 }");
    }

    #[test]
    fn display_function() {
        let func = IrType::Function(Box::new(IrType::I32), vec![IrType::I32, IrType::Ptr]);
        assert_eq!(format!("{}", func), "i32 (i32, ptr)");
    }

    #[test]
    fn display_void_function() {
        let func = IrType::Function(Box::new(IrType::Void), vec![]);
        assert_eq!(format!("{}", func), "void ()");
    }

    #[test]
    fn display_nested_array() {
        let inner = IrType::Array(Box::new(IrType::I32), 4);
        let outer = IrType::Array(Box::new(inner), 3);
        assert_eq!(format!("{}", outer), "[3 x [4 x i32]]");
    }

    // -- Clone and PartialEq -------------------------------------------------

    #[test]
    fn clone_and_eq() {
        let ty = IrType::Struct(StructType::new(vec![IrType::I32, IrType::Ptr], false));
        let cloned = ty.clone();
        assert_eq!(ty, cloned);
    }

    #[test]
    fn ne_different_types() {
        assert_ne!(IrType::I32, IrType::I64);
        assert_ne!(IrType::Ptr, IrType::I64);
        assert_ne!(
            IrType::Array(Box::new(IrType::I32), 5),
            IrType::Array(Box::new(IrType::I32), 6)
        );
    }

    // -- Hash ---------------------------------------------------------------

    #[test]
    fn hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(IrType::I32);
        set.insert(IrType::I32);
        assert_eq!(set.len(), 1);

        set.insert(IrType::I64);
        assert_eq!(set.len(), 2);
    }
}
