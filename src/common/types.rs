//! Dual type system for the BCC compiler.
//!
//! This module defines both **C language types** ([`CType`]) and **target-machine
//! types** ([`MachineType`]), forming the backbone of the entire compilation
//! pipeline.  Every stage—frontend type checking, IR lowering, and backend
//! register allocation—depends on the types declared here.
//!
//! # Architecture
//!
//! * [`CType`] captures the full C11 type universe: scalars, pointers, arrays,
//!   functions, aggregates (struct/union/enum), `_Atomic`, `_Complex`, and
//!   typedefs.
//! * [`MachineType`] maps C types to hardware register classes and memory
//!   operand sizes, consumed by the code-generation backends.
//! * [`QualifiedType`] pairs a [`CType`] with its [`TypeQualifiers`] (const,
//!   volatile, restrict, _Atomic).
//! * [`FieldDef`] describes individual struct/union members, including
//!   bit-field width and anonymous fields.
//!
//! # Target-Dependent Sizing
//!
//! The free functions [`size_of`] and [`align_of`] compute the byte size and
//! alignment of a [`CType`] for a given [`Target`] architecture.  Results
//! differ between LP64 (x86-64, AArch64, RISC-V 64) and ILP32 (i686) data
//! models.
//!
//! # Integer Promotion & Usual Arithmetic Conversions
//!
//! [`integer_promote`] implements C11 §6.3.1.1 integer promotion rules.
//! [`usual_arithmetic_conversion`] implements C11 §6.3.1.8 for determining the
//! common type of two arithmetic operands.

use std::fmt;

use crate::common::target::{DataModel, Target};

// ---------------------------------------------------------------------------
// FieldDef — struct / union member
// ---------------------------------------------------------------------------

/// A single field within a `struct` or `union` type.
///
/// Anonymous fields (`name = None`) arise from anonymous structs/unions nested
/// inside an outer aggregate (a C11 feature).  Bit-fields carry an explicit
/// `bit_width`; a zero-width bit-field forces alignment to the next storage
/// unit boundary of the declared type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldDef {
    /// Field name, or `None` for anonymous struct/union members.
    pub name: Option<String>,
    /// The C type of this field.
    pub ty: CType,
    /// If this field is a bit-field, the width in bits; otherwise `None`.
    pub bit_width: Option<u32>,
}

// ---------------------------------------------------------------------------
// TypeQualifiers
// ---------------------------------------------------------------------------

/// C type qualifiers that refine the semantics of a base type.
///
/// Qualifiers are orthogonal to the base type and are tracked as a separate
/// bitmask so that qualified and unqualified versions of the same type share
/// the same [`CType`] representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TypeQualifiers {
    /// `const` — object may not be modified after initialisation.
    pub is_const: bool,
    /// `volatile` — every access must be observed by the abstract machine.
    pub is_volatile: bool,
    /// `restrict` — pointer is the sole means of accessing the object (C99).
    pub is_restrict: bool,
    /// `_Atomic` — accesses have sequentially-consistent or explicit memory
    /// ordering semantics (C11).
    pub is_atomic: bool,
}

impl TypeQualifiers {
    /// Returns a qualifier set with no qualifiers active.
    #[inline]
    pub fn none() -> Self {
        Self::default()
    }

    /// Returns `true` when no qualifier is set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.is_const && !self.is_volatile && !self.is_restrict && !self.is_atomic
    }

    /// Merges two qualifier sets, producing a set where each qualifier is
    /// active if it is active in *either* operand.
    #[inline]
    pub fn merge(&self, other: &TypeQualifiers) -> TypeQualifiers {
        TypeQualifiers {
            is_const: self.is_const || other.is_const,
            is_volatile: self.is_volatile || other.is_volatile,
            is_restrict: self.is_restrict || other.is_restrict,
            is_atomic: self.is_atomic || other.is_atomic,
        }
    }
}

impl fmt::Display for TypeQualifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut space = false;
        if self.is_const {
            write!(f, "const")?;
            space = true;
        }
        if self.is_volatile {
            if space {
                write!(f, " ")?;
            }
            write!(f, "volatile")?;
            space = true;
        }
        if self.is_restrict {
            if space {
                write!(f, " ")?;
            }
            write!(f, "restrict")?;
            space = true;
        }
        if self.is_atomic {
            if space {
                write!(f, " ")?;
            }
            write!(f, "_Atomic")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CType — the C language type universe
// ---------------------------------------------------------------------------

/// Represents the full C11 type system including GCC extensions.
///
/// Every node in the AST, every IR value, and every symbol-table entry
/// ultimately refers to a `CType`.  Recursive types (pointers, arrays,
/// functions, complex, atomic, typedef) use `Box<CType>` to allow arbitrary
/// nesting without infinite-size enum variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CType {
    /// `void` — the incomplete type with no values.
    Void,

    /// `_Bool` (C99).
    Bool,

    /// `char` / `signed char` / `unsigned char`.
    Char { signed: bool },

    /// `short` / `unsigned short`.
    Short { signed: bool },

    /// `int` / `unsigned int`.
    Int { signed: bool },

    /// `long` / `unsigned long`.  Size is target-dependent (LP64 = 8, ILP32 = 4).
    Long { signed: bool },

    /// `long long` / `unsigned long long`.  Always 8 bytes.
    LongLong { signed: bool },

    /// `float` — IEEE 754 single precision (32-bit).
    Float,

    /// `double` — IEEE 754 double precision (64-bit).
    Double,

    /// `long double` — size is target-dependent:
    /// x86-64: 16 bytes (80-bit x87 extended, padded).
    /// i686:   12 bytes (80-bit x87 extended, padded).
    /// AArch64/RISC-V 64: 8 bytes (maps to double).
    LongDouble,

    /// `_Complex <base>` (C99).  The base type is one of `Float`, `Double`,
    /// or `LongDouble`.  Storage is 2× the base type size.
    Complex(Box<CType>),

    /// Pointer to another type.
    Pointer(Box<CType>),

    /// Array with optional compile-time-known element count.
    /// `size = None` for VLAs, flexible array members, or `extern T[]`.
    Array {
        element: Box<CType>,
        size: Option<usize>,
    },

    /// Function type: return type, parameter types, and variadic flag.
    Function {
        return_type: Box<CType>,
        params: Vec<CType>,
        variadic: bool,
    },

    /// `struct` type with optional tag name and field list.
    Struct {
        name: Option<String>,
        fields: Vec<FieldDef>,
    },

    /// `union` type with optional tag name and field list.
    Union {
        name: Option<String>,
        fields: Vec<FieldDef>,
    },

    /// `enum` type with optional tag name and underlying integer type.
    Enum {
        name: Option<String>,
        underlying: Box<CType>,
    },

    /// `_Atomic(T)` — atomic-qualified type (C11 §6.7.2.4).
    Atomic(Box<CType>),

    /// `typedef` name alias wrapping an underlying type.
    Typedef {
        name: String,
        underlying: Box<CType>,
    },
}

// ---------------------------------------------------------------------------
// QualifiedType
// ---------------------------------------------------------------------------

/// A [`CType`] paired with its [`TypeQualifiers`].
///
/// Most compiler stages pass `QualifiedType` rather than bare `CType` so that
/// qualifier information (const, volatile, restrict, _Atomic) is preserved
/// throughout the pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedType {
    /// The underlying C type.
    pub ty: CType,
    /// The set of type qualifiers applied to `ty`.
    pub qualifiers: TypeQualifiers,
}

impl QualifiedType {
    /// Creates an unqualified type.
    #[inline]
    pub fn unqualified(ty: CType) -> Self {
        Self {
            ty,
            qualifiers: TypeQualifiers::none(),
        }
    }

    /// Creates a type with the given qualifiers.
    #[inline]
    pub fn with_qualifiers(ty: CType, qualifiers: TypeQualifiers) -> Self {
        Self { ty, qualifiers }
    }
}

impl fmt::Display for QualifiedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.qualifiers.is_empty() {
            write!(f, "{} {}", self.qualifiers, self.ty)
        } else {
            write!(f, "{}", self.ty)
        }
    }
}

// ---------------------------------------------------------------------------
// MachineType — backend register-class mapping
// ---------------------------------------------------------------------------

/// Target-machine type used during code generation for register-class mapping.
///
/// While [`CType`] represents the C-language view of a value, `MachineType`
/// represents the *hardware* view: each variant corresponds to a register class
/// (integer, floating-point, vector) or a memory operand of a specific byte
/// width.  The backend's `ArchCodegen` trait converts `CType` into
/// `MachineType` during instruction selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachineType {
    /// 8-bit integer (byte register, e.g. AL on x86).
    I8,
    /// 16-bit integer (e.g. AX on x86).
    I16,
    /// 32-bit integer (e.g. EAX on x86, W-registers on AArch64).
    I32,
    /// 64-bit integer (e.g. RAX on x86-64, X-registers on AArch64).
    I64,
    /// 128-bit integer (e.g. for `__int128` or CMPXCHG16B on x86-64).
    I128,
    /// 32-bit IEEE 754 single precision (SSE on x86, S-registers on AArch64).
    F32,
    /// 64-bit IEEE 754 double precision (SSE on x86, D-registers on AArch64).
    F64,
    /// 80-bit x87 extended precision (x86-only, ST(n) stack).
    F80,
    /// Pointer-width integer (maps to I32 on ILP32, I64 on LP64).
    Ptr,
    /// Zero-sized (for `void` return types).
    Void,
    /// Aggregate passed in memory; `usize` is the total byte size.
    Aggregate(usize),
}

impl fmt::Display for MachineType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MachineType::I8 => write!(f, "i8"),
            MachineType::I16 => write!(f, "i16"),
            MachineType::I32 => write!(f, "i32"),
            MachineType::I64 => write!(f, "i64"),
            MachineType::I128 => write!(f, "i128"),
            MachineType::F32 => write!(f, "f32"),
            MachineType::F64 => write!(f, "f64"),
            MachineType::F80 => write!(f, "f80"),
            MachineType::Ptr => write!(f, "ptr"),
            MachineType::Void => write!(f, "void"),
            MachineType::Aggregate(sz) => write!(f, "agg({})", sz),
        }
    }
}

// ---------------------------------------------------------------------------
// CType — type predicate methods
// ---------------------------------------------------------------------------

impl CType {
    /// Returns `true` for integer types: `_Bool`, `char`, `short`, `int`,
    /// `long`, `long long`, and `enum`.
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            CType::Bool
                | CType::Char { .. }
                | CType::Short { .. }
                | CType::Int { .. }
                | CType::Long { .. }
                | CType::LongLong { .. }
                | CType::Enum { .. }
        )
    }

    /// Returns `true` for floating-point types: `float`, `double`, and
    /// `long double`.
    #[inline]
    pub fn is_floating(&self) -> bool {
        matches!(self, CType::Float | CType::Double | CType::LongDouble)
    }

    /// Returns `true` for arithmetic types: integers, floating-point, and
    /// `_Complex` types.
    #[inline]
    pub fn is_arithmetic(&self) -> bool {
        self.is_integer() || self.is_floating() || matches!(self, CType::Complex(_))
    }

    /// Returns `true` for scalar types: arithmetic types and pointers.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_arithmetic() || self.is_pointer()
    }

    /// Returns `true` for aggregate types: `struct`, `union`, and arrays.
    #[inline]
    pub fn is_aggregate(&self) -> bool {
        matches!(
            self,
            CType::Struct { .. } | CType::Union { .. } | CType::Array { .. }
        )
    }

    /// Returns `true` for pointer types.
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(self, CType::Pointer(_))
    }

    /// Returns `true` for the `void` type.
    #[inline]
    pub fn is_void(&self) -> bool {
        matches!(self, CType::Void)
    }

    /// Returns `true` for function types.
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self, CType::Function { .. })
    }

    /// Returns `true` for array types (both sized and unsized).
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, CType::Array { .. })
    }

    /// Returns `true` if the type is *complete* — that is, its size is known
    /// at the point of use.
    ///
    /// Incomplete types include:
    /// - `void`
    /// - Arrays with unknown size (`size = None`)
    /// - Forward-declared structs/unions (tag name present, no fields)
    pub fn is_complete(&self) -> bool {
        match self {
            CType::Void => false,
            CType::Array { size: None, .. } => false,
            // A named struct/union with zero fields is treated as a forward
            // declaration (incomplete).  Anonymous aggregates with zero fields
            // are considered complete (degenerate but valid GCC extension).
            CType::Struct { name: Some(_), fields } if fields.is_empty() => false,
            CType::Union { name: Some(_), fields } if fields.is_empty() => false,
            CType::Typedef { underlying, .. } => underlying.is_complete(),
            CType::Atomic(inner) => inner.is_complete(),
            _ => true,
        }
    }

    /// Returns `true` for signed integer types.
    ///
    /// `_Bool` is unsigned.  `enum` types are considered signed (their
    /// underlying type is `int` by default).
    pub fn is_signed(&self) -> bool {
        match self {
            CType::Char { signed } => *signed,
            CType::Short { signed } => *signed,
            CType::Int { signed } => *signed,
            CType::Long { signed } => *signed,
            CType::LongLong { signed } => *signed,
            CType::Enum { underlying, .. } => underlying.is_signed(),
            CType::Typedef { underlying, .. } => underlying.is_signed(),
            CType::Atomic(inner) => inner.is_signed(),
            _ => false,
        }
    }

    /// Returns `true` for unsigned integer types.
    ///
    /// `_Bool` is considered unsigned.
    pub fn is_unsigned(&self) -> bool {
        match self {
            CType::Bool => true,
            CType::Char { signed } => !*signed,
            CType::Short { signed } => !*signed,
            CType::Int { signed } => !*signed,
            CType::Long { signed } => !*signed,
            CType::LongLong { signed } => !*signed,
            CType::Enum { underlying, .. } => underlying.is_unsigned(),
            CType::Typedef { underlying, .. } => underlying.is_unsigned(),
            CType::Atomic(inner) => inner.is_unsigned(),
            _ => false,
        }
    }

    /// Strips any `Typedef` and `Atomic` wrappers, returning the canonical
    /// underlying type.
    pub fn canonical(&self) -> &CType {
        match self {
            CType::Typedef { underlying, .. } => underlying.canonical(),
            CType::Atomic(inner) => inner.canonical(),
            other => other,
        }
    }

    /// Returns the integer conversion rank per C11 §6.3.1.1.
    ///
    /// Higher rank means wider type.  Returns `None` for non-integer types.
    pub fn integer_rank(&self) -> Option<u32> {
        match self.canonical() {
            CType::Bool => Some(0),
            CType::Char { .. } => Some(1),
            CType::Short { .. } => Some(2),
            CType::Int { .. } => Some(3),
            CType::Long { .. } => Some(4),
            CType::LongLong { .. } => Some(5),
            CType::Enum { .. } => Some(3), // enum rank == int rank
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display for CType — human-readable C-style formatting
// ---------------------------------------------------------------------------

impl fmt::Display for CType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CType::Void => write!(f, "void"),
            CType::Bool => write!(f, "_Bool"),
            CType::Char { signed: true } => write!(f, "char"),
            CType::Char { signed: false } => write!(f, "unsigned char"),
            CType::Short { signed: true } => write!(f, "short"),
            CType::Short { signed: false } => write!(f, "unsigned short"),
            CType::Int { signed: true } => write!(f, "int"),
            CType::Int { signed: false } => write!(f, "unsigned int"),
            CType::Long { signed: true } => write!(f, "long"),
            CType::Long { signed: false } => write!(f, "unsigned long"),
            CType::LongLong { signed: true } => write!(f, "long long"),
            CType::LongLong { signed: false } => write!(f, "unsigned long long"),
            CType::Float => write!(f, "float"),
            CType::Double => write!(f, "double"),
            CType::LongDouble => write!(f, "long double"),
            CType::Complex(base) => write!(f, "_Complex {}", base),
            CType::Pointer(pointee) => {
                if let CType::Function {
                    return_type,
                    params,
                    variadic,
                } = pointee.as_ref()
                {
                    // Function pointer: void (*)(int, int)
                    write!(f, "{} (*)(", return_type)?;
                    for (i, p) in params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", p)?;
                    }
                    if *variadic {
                        if !params.is_empty() {
                            write!(f, ", ")?;
                        }
                        write!(f, "...")?;
                    }
                    write!(f, ")")
                } else {
                    write!(f, "{} *", pointee)
                }
            }
            CType::Array { element, size } => match size {
                Some(n) => write!(f, "{}[{}]", element, n),
                None => write!(f, "{}[]", element),
            },
            CType::Function {
                return_type,
                params,
                variadic,
            } => {
                write!(f, "{}(", return_type)?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                if *variadic {
                    if !params.is_empty() {
                        write!(f, ", ")?;
                    }
                    write!(f, "...")?;
                }
                write!(f, ")")
            }
            CType::Struct { name, .. } => match name {
                Some(n) => write!(f, "struct {}", n),
                None => write!(f, "struct <anonymous>"),
            },
            CType::Union { name, .. } => match name {
                Some(n) => write!(f, "union {}", n),
                None => write!(f, "union <anonymous>"),
            },
            CType::Enum { name, .. } => match name {
                Some(n) => write!(f, "enum {}", n),
                None => write!(f, "enum <anonymous>"),
            },
            CType::Atomic(inner) => write!(f, "_Atomic({})", inner),
            CType::Typedef { name, .. } => write!(f, "{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// size_of — target-dependent byte size computation
// ---------------------------------------------------------------------------

/// Computes the byte size of `ty` on the given `target` architecture.
///
/// For aggregate types, this includes all internal padding and trailing
/// padding required to satisfy the aggregate's alignment constraint.  For
/// incomplete types (`void`, unsized arrays, forward-declared structs),
/// returns `0`.
///
/// # Parameters
///
/// * `ty` — the C type to measure.
/// * `target` — the compilation target, providing data-model and
///   architecture-specific sizing.
pub fn size_of(ty: &CType, target: &Target) -> usize {
    match ty {
        CType::Void => 0,
        CType::Bool => 1,
        CType::Char { .. } => 1,
        CType::Short { .. } => 2,
        CType::Int { .. } => 4,
        CType::Long { .. } => target.long_size() as usize,
        CType::LongLong { .. } => 8,
        CType::Float => 4,
        CType::Double => 8,
        CType::LongDouble => target.long_double_size() as usize,
        CType::Complex(base) => {
            // _Complex is two copies of the base floating-point type.
            2 * size_of(base, target)
        }
        CType::Pointer(_) => target.pointer_width() as usize,
        CType::Array { element, size } => match size {
            Some(count) => size_of(element, target) * count,
            None => 0, // Incomplete array type.
        },
        CType::Function { .. } => {
            // Function types are not object types and have no size.
            // GCC as an extension treats sizeof(function) as 1.
            1
        }
        CType::Struct { fields, .. } => {
            let (sz, _) = compute_struct_layout(fields, target);
            sz
        }
        CType::Union { fields, .. } => {
            let (sz, _) = compute_union_layout(fields, target);
            sz
        }
        CType::Enum { underlying, .. } => size_of(underlying, target),
        CType::Atomic(inner) => size_of(inner, target),
        CType::Typedef { underlying, .. } => size_of(underlying, target),
    }
}

// ---------------------------------------------------------------------------
// align_of — target-dependent alignment computation
// ---------------------------------------------------------------------------

/// Computes the required alignment (in bytes) of `ty` on the given `target`.
///
/// For aggregate types, the alignment is the maximum alignment of any member.
/// For `_Atomic` types, alignment may be increased to the type's size (when a
/// power of two) to guarantee lock-free atomic operations.
///
/// # Parameters
///
/// * `ty`     — the C type to query.
/// * `target` — the compilation target, providing ABI alignment rules.
pub fn align_of(ty: &CType, target: &Target) -> usize {
    match ty {
        CType::Void => 1,
        CType::Bool => 1,
        CType::Char { .. } => 1,
        CType::Short { .. } => 2,
        CType::Int { .. } => 4,
        CType::Long { .. } => target.long_size() as usize,
        // On ILP32 (i686), the System V i386 ABI specifies 4-byte alignment
        // for 8-byte types (long long, double) within aggregates.  On LP64
        // targets these are naturally 8-byte aligned.
        CType::LongLong { .. } => match target.data_model() {
            DataModel::LP64 => 8,
            DataModel::ILP32 => 4,
        },
        CType::Float => 4,
        CType::Double => match target.data_model() {
            DataModel::LP64 => 8,
            DataModel::ILP32 => 4,
        },
        CType::LongDouble => long_double_align(target),
        CType::Complex(base) => align_of(base, target),
        CType::Pointer(_) => target.pointer_align() as usize,
        CType::Array { element, .. } => align_of(element, target),
        CType::Function { .. } => 1,
        CType::Struct { fields, .. } => {
            let (_, al) = compute_struct_layout(fields, target);
            al
        }
        CType::Union { fields, .. } => {
            let (_, al) = compute_union_layout(fields, target);
            al
        }
        CType::Enum { underlying, .. } => align_of(underlying, target),
        CType::Atomic(inner) => {
            let base_align = align_of(inner, target);
            let inner_sz = size_of(inner, target);
            // _Atomic types increase alignment to their size when the size
            // is a power of two and does not exceed the target's stack
            // alignment boundary, ensuring lock-free access patterns.
            let stack_align = target.stack_alignment() as usize;
            if inner_sz > 0 && inner_sz.is_power_of_two() && inner_sz <= stack_align {
                inner_sz.max(base_align)
            } else {
                base_align
            }
        }
        CType::Typedef { underlying, .. } => align_of(underlying, target),
    }
}

// ---------------------------------------------------------------------------
// Struct / Union layout helpers
// ---------------------------------------------------------------------------

/// Computes the total byte size and alignment of a `struct` type, including
/// inter-field padding and trailing padding.
///
/// Bit-fields are packed into storage units of the declared type's width.
/// A zero-width bit-field forces alignment to the next storage unit boundary.
fn compute_struct_layout(fields: &[FieldDef], target: &Target) -> (usize, usize) {
    if fields.is_empty() {
        return (0, 1);
    }

    let mut total_size: usize = 0;
    let mut max_align: usize = 1;
    let mut bit_offset: usize = 0; // bits used in the current allocation unit
    let mut unit_bits: usize = 0; // total bits in the current allocation unit

    for field in fields {
        let field_align = align_of(&field.ty, target);
        let field_size = size_of(&field.ty, target);

        if field_align > max_align {
            max_align = field_align;
        }

        if let Some(bw) = field.bit_width {
            let bits = bw as usize;
            let type_bits = field_size * 8;

            if bits == 0 {
                // Zero-width bit-field: flush current bit run and align to
                // the declared type's alignment.
                total_size += (bit_offset + 7) / 8;
                bit_offset = 0;
                unit_bits = 0;
                total_size = round_up(total_size, field_align);
                continue;
            }

            // Determine whether the bit-field fits in the current storage unit.
            if unit_bits == 0 || bit_offset + bits > unit_bits {
                // Start a new storage unit — flush pending bits first.
                total_size += (bit_offset + 7) / 8;
                total_size = round_up(total_size, field_align);
                unit_bits = type_bits;
                bit_offset = 0;
            }

            bit_offset += bits;
        } else {
            // Regular (non-bit-field) member.
            if bit_offset > 0 {
                total_size += (bit_offset + 7) / 8;
                bit_offset = 0;
                unit_bits = 0;
            }
            total_size = round_up(total_size, field_align);
            total_size += field_size;
        }
    }

    // Flush any trailing bit-field bits.
    if bit_offset > 0 {
        total_size += (bit_offset + 7) / 8;
    }

    // Trailing padding to satisfy the overall struct alignment.
    total_size = round_up(total_size, max_align);

    (total_size, max_align)
}

/// Computes the total byte size and alignment of a `union` type.
///
/// The size is the maximum of all member sizes, rounded up to the union's
/// alignment (which is the maximum of all member alignments).
fn compute_union_layout(fields: &[FieldDef], target: &Target) -> (usize, usize) {
    if fields.is_empty() {
        return (0, 1);
    }

    let mut max_size: usize = 0;
    let mut max_align: usize = 1;

    for field in fields {
        let field_size = if let Some(bw) = field.bit_width {
            // Bit-field in a union: size is that of the declared type.
            size_of(&field.ty, target)
                .max(((bw as usize) + 7) / 8)
        } else {
            size_of(&field.ty, target)
        };
        let field_align = align_of(&field.ty, target);

        if field_size > max_size {
            max_size = field_size;
        }
        if field_align > max_align {
            max_align = field_align;
        }
    }

    // Pad to union alignment.
    max_size = round_up(max_size, max_align);

    (max_size, max_align)
}

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` **must** be a power of two.  Behaviour is undefined (in the Rust
/// sense) if `align` is zero; this function will return `value` unchanged.
#[inline]
fn round_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

/// Returns the ABI alignment for `long double` on the given `target`.
///
/// | Target     | Size | Alignment |
/// |------------|------|-----------|
/// | x86-64     | 16   | 16        |
/// | i686       | 12   | 4         |
/// | AArch64    | 8    | 8         |
/// | RISC-V 64  | 8    | 8         |
fn long_double_align(target: &Target) -> usize {
    match target.long_double_size() {
        16 => 16, // x86-64: 80-bit padded to 16 bytes, 16-byte aligned
        12 => 4,  // i686: 80-bit padded to 12 bytes, 4-byte aligned (SysV i386 ABI)
        _ => 8,   // AArch64 / RISC-V 64: maps to double, 8-byte aligned
    }
}

// ---------------------------------------------------------------------------
// integer_promote — C11 §6.3.1.1
// ---------------------------------------------------------------------------

/// Performs integer promotion per C11 §6.3.1.1.
///
/// Types narrower than `int` are promoted to `int` if `int` can represent all
/// values of the original type; otherwise they are promoted to `unsigned int`.
/// Types of rank `int` or higher are returned unchanged.
///
/// # Rules applied
///
/// | Original type              | Promoted to       |
/// |---------------------------|-------------------|
/// | `_Bool`                   | `int`             |
/// | `char` (signed/unsigned)  | `int`             |
/// | `short` (signed/unsigned) | `int`             |
/// | `int` and wider           | unchanged         |
/// | `enum`                    | underlying type (typically `int`) |
/// | non-integer types         | returned unchanged |
pub fn integer_promote(ty: &CType) -> CType {
    match ty.canonical() {
        CType::Bool => CType::Int { signed: true },
        CType::Char { .. } => {
            // Both signed and unsigned char fit within int (since int is at
            // least 16 bits and char is exactly 8 bits on all targets).
            CType::Int { signed: true }
        }
        CType::Short { .. } => {
            // Both signed and unsigned short fit within int (since int is 32
            // bits and short is 16 bits on all targets).
            CType::Int { signed: true }
        }
        CType::Enum { underlying, .. } => {
            // Promote the enum's underlying type.
            integer_promote(underlying)
        }
        // Int, Long, LongLong — already at or above int rank; no promotion.
        // Non-integer types — returned as-is (caller should not pass them).
        _ => ty.clone(),
    }
}

// ---------------------------------------------------------------------------
// usual_arithmetic_conversion — C11 §6.3.1.8
// ---------------------------------------------------------------------------

/// Determines the common type of two arithmetic operands per C11 §6.3.1.8
/// (the "usual arithmetic conversions").
///
/// Both operands are first subjected to integer promotion.  The rules then
/// select the wider or higher-ranked type, with careful handling of mixed
/// signed/unsigned operands.
///
/// # Parameters
///
/// * `a`, `b` — the two operand types (should be arithmetic types).
///
/// # Returns
///
/// The common type to which both operands will be implicitly converted.
pub fn usual_arithmetic_conversion(a: &CType, b: &CType) -> CType {
    let a = a.canonical();
    let b = b.canonical();

    // 1. If either operand is `long double`, result is `long double`.
    if matches!(a, CType::LongDouble) || matches!(b, CType::LongDouble) {
        return CType::LongDouble;
    }
    // 2. If either operand is `double`, result is `double`.
    if matches!(a, CType::Double) || matches!(b, CType::Double) {
        return CType::Double;
    }
    // 3. If either operand is `float`, result is `float`.
    if matches!(a, CType::Float) || matches!(b, CType::Float) {
        return CType::Float;
    }

    // 4. Integer promotions are performed on both operands.
    let pa = integer_promote(a);
    let pb = integer_promote(b);

    // 5. If both operands are the same type after promotion, done.
    if pa == pb {
        return pa;
    }

    let rank_a = pa.integer_rank().unwrap_or(3);
    let rank_b = pb.integer_rank().unwrap_or(3);
    let signed_a = pa.is_signed();
    let signed_b = pb.is_signed();

    // Both same signedness → convert to the higher rank.
    if signed_a == signed_b {
        return if rank_a >= rank_b {
            pa
        } else {
            pb
        };
    }

    // Mixed signedness.  Identify which is unsigned and which is signed.
    let (unsigned_ty, unsigned_rank, signed_ty, signed_rank) = if !signed_a {
        (&pa, rank_a, &pb, rank_b)
    } else {
        (&pb, rank_b, &pa, rank_a)
    };

    // 6. If the unsigned operand's rank >= signed operand's rank,
    //    convert to the unsigned type.
    if unsigned_rank >= signed_rank {
        return unsigned_ty.clone();
    }

    // 7. If the signed type can represent all values of the unsigned type
    //    (i.e. the signed type is strictly wider), convert to the signed type.
    //    Since we use standard widths (short=16, int=32, long=32/64, llong=64),
    //    a higher-ranked signed type is always strictly wider.
    if signed_rank > unsigned_rank {
        return signed_ty.clone();
    }

    // 8. Otherwise, convert both to the unsigned counterpart of the signed type.
    //    This case arises when signed and unsigned types have the same rank
    //    (handled above) so in practice this is unreachable with the standard
    //    C integer widths, but we include it for correctness.
    make_unsigned(signed_ty)
}

/// Converts an integer type to its unsigned counterpart.  Non-integer types
/// are returned unchanged.
fn make_unsigned(ty: &CType) -> CType {
    match ty {
        CType::Char { .. } => CType::Char { signed: false },
        CType::Short { .. } => CType::Short { signed: false },
        CType::Int { .. } => CType::Int { signed: false },
        CType::Long { .. } => CType::Long { signed: false },
        CType::LongLong { .. } => CType::LongLong { signed: false },
        other => other.clone(),
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience aliases for all four targets.
    fn x86_64() -> Target {
        Target::from_str("x86-64").unwrap()
    }
    fn i686() -> Target {
        Target::from_str("i686").unwrap()
    }
    fn aarch64() -> Target {
        Target::from_str("aarch64").unwrap()
    }
    fn riscv64() -> Target {
        Target::from_str("riscv64").unwrap()
    }

    // ----- Type predicates -----------------------------------------------

    #[test]
    fn test_is_integer() {
        assert!(CType::Bool.is_integer());
        assert!(CType::Char { signed: true }.is_integer());
        assert!(CType::Short { signed: false }.is_integer());
        assert!(CType::Int { signed: true }.is_integer());
        assert!(CType::Long { signed: true }.is_integer());
        assert!(CType::LongLong { signed: false }.is_integer());
        assert!(!CType::Float.is_integer());
        assert!(!CType::Double.is_integer());
        assert!(!CType::Pointer(Box::new(CType::Void)).is_integer());
    }

    #[test]
    fn test_is_floating() {
        assert!(CType::Float.is_floating());
        assert!(CType::Double.is_floating());
        assert!(CType::LongDouble.is_floating());
        assert!(!CType::Int { signed: true }.is_floating());
    }

    #[test]
    fn test_is_arithmetic() {
        assert!(CType::Int { signed: true }.is_arithmetic());
        assert!(CType::Double.is_arithmetic());
        assert!(CType::Complex(Box::new(CType::Float)).is_arithmetic());
        assert!(!CType::Pointer(Box::new(CType::Int { signed: true })).is_arithmetic());
    }

    #[test]
    fn test_is_scalar() {
        assert!(CType::Int { signed: true }.is_scalar());
        assert!(CType::Pointer(Box::new(CType::Void)).is_scalar());
        assert!(!CType::Struct {
            name: Some("foo".into()),
            fields: vec![],
        }
        .is_scalar());
    }

    #[test]
    fn test_is_aggregate() {
        assert!(CType::Struct {
            name: None,
            fields: vec![FieldDef {
                name: Some("x".into()),
                ty: CType::Int { signed: true },
                bit_width: None,
            }],
        }
        .is_aggregate());
        assert!(CType::Union {
            name: None,
            fields: vec![],
        }
        .is_aggregate());
        assert!(CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        }
        .is_aggregate());
        assert!(!CType::Int { signed: true }.is_aggregate());
    }

    #[test]
    fn test_is_complete() {
        assert!(!CType::Void.is_complete());
        assert!(CType::Int { signed: true }.is_complete());
        // Forward-declared struct is incomplete.
        assert!(!CType::Struct {
            name: Some("forward".into()),
            fields: vec![],
        }
        .is_complete());
        // Anonymous struct with fields is complete.
        assert!(CType::Struct {
            name: None,
            fields: vec![FieldDef {
                name: Some("a".into()),
                ty: CType::Int { signed: true },
                bit_width: None,
            }],
        }
        .is_complete());
        // Unsized array is incomplete.
        assert!(!CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: None,
        }
        .is_complete());
    }

    #[test]
    fn test_is_signed_unsigned() {
        assert!(CType::Int { signed: true }.is_signed());
        assert!(!CType::Int { signed: true }.is_unsigned());
        assert!(CType::Int { signed: false }.is_unsigned());
        assert!(!CType::Int { signed: false }.is_signed());
        assert!(CType::Bool.is_unsigned());
        assert!(!CType::Bool.is_signed());
    }

    // ----- size_of -------------------------------------------------------

    #[test]
    fn size_of_basic_types_x86_64() {
        let t = x86_64();
        assert_eq!(size_of(&CType::Void, &t), 0);
        assert_eq!(size_of(&CType::Bool, &t), 1);
        assert_eq!(size_of(&CType::Char { signed: true }, &t), 1);
        assert_eq!(size_of(&CType::Short { signed: true }, &t), 2);
        assert_eq!(size_of(&CType::Int { signed: true }, &t), 4);
        assert_eq!(size_of(&CType::Long { signed: true }, &t), 8);
        assert_eq!(size_of(&CType::LongLong { signed: true }, &t), 8);
        assert_eq!(size_of(&CType::Float, &t), 4);
        assert_eq!(size_of(&CType::Double, &t), 8);
        assert_eq!(size_of(&CType::LongDouble, &t), 16);
        assert_eq!(
            size_of(&CType::Pointer(Box::new(CType::Void)), &t),
            8
        );
    }

    #[test]
    fn size_of_basic_types_i686() {
        let t = i686();
        assert_eq!(size_of(&CType::Long { signed: true }, &t), 4);
        assert_eq!(size_of(&CType::LongDouble, &t), 12);
        assert_eq!(
            size_of(&CType::Pointer(Box::new(CType::Void)), &t),
            4
        );
    }

    #[test]
    fn size_of_complex() {
        let t = x86_64();
        assert_eq!(
            size_of(&CType::Complex(Box::new(CType::Float)), &t),
            8
        );
        assert_eq!(
            size_of(&CType::Complex(Box::new(CType::Double)), &t),
            16
        );
    }

    #[test]
    fn size_of_array() {
        let t = x86_64();
        let arr = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        assert_eq!(size_of(&arr, &t), 40);
    }

    #[test]
    fn size_of_struct_with_padding() {
        let t = x86_64();
        // struct { char c; int i; } → size=8 (1+3pad+4), align=4
        let s = CType::Struct {
            name: None,
            fields: vec![
                FieldDef {
                    name: Some("c".into()),
                    ty: CType::Char { signed: true },
                    bit_width: None,
                },
                FieldDef {
                    name: Some("i".into()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
            ],
        };
        assert_eq!(size_of(&s, &t), 8);
        assert_eq!(align_of(&s, &t), 4);
    }

    #[test]
    fn size_of_union() {
        let t = x86_64();
        let u = CType::Union {
            name: None,
            fields: vec![
                FieldDef {
                    name: Some("i".into()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
                FieldDef {
                    name: Some("d".into()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
        };
        assert_eq!(size_of(&u, &t), 8); // max(4, 8) padded to align=8
        assert_eq!(align_of(&u, &t), 8);
    }

    // ----- align_of ------------------------------------------------------

    #[test]
    fn align_of_pointer() {
        assert_eq!(
            align_of(&CType::Pointer(Box::new(CType::Void)), &x86_64()),
            8
        );
        assert_eq!(
            align_of(&CType::Pointer(Box::new(CType::Void)), &i686()),
            4
        );
    }

    #[test]
    fn align_of_long_double() {
        assert_eq!(align_of(&CType::LongDouble, &x86_64()), 16);
        assert_eq!(align_of(&CType::LongDouble, &i686()), 4);
        assert_eq!(align_of(&CType::LongDouble, &aarch64()), 8);
        assert_eq!(align_of(&CType::LongDouble, &riscv64()), 8);
    }

    #[test]
    fn align_of_double_ilp32_vs_lp64() {
        // On i686 (ILP32), double has 4-byte ABI alignment.
        assert_eq!(align_of(&CType::Double, &i686()), 4);
        // On LP64 targets, double has 8-byte alignment.
        assert_eq!(align_of(&CType::Double, &x86_64()), 8);
    }

    #[test]
    fn align_of_atomic() {
        let t = x86_64();
        // _Atomic(int): size=4, power of 2, within stack_alignment → align=4
        let atomic_int = CType::Atomic(Box::new(CType::Int { signed: true }));
        assert_eq!(align_of(&atomic_int, &t), 4);
        // _Atomic(long long) on x86-64: size=8, align=8
        let atomic_ll = CType::Atomic(Box::new(CType::LongLong { signed: true }));
        assert_eq!(align_of(&atomic_ll, &t), 8);
    }

    // ----- integer_promote -----------------------------------------------

    #[test]
    fn promote_narrow_types() {
        assert_eq!(
            integer_promote(&CType::Bool),
            CType::Int { signed: true }
        );
        assert_eq!(
            integer_promote(&CType::Char { signed: true }),
            CType::Int { signed: true }
        );
        assert_eq!(
            integer_promote(&CType::Char { signed: false }),
            CType::Int { signed: true }
        );
        assert_eq!(
            integer_promote(&CType::Short { signed: false }),
            CType::Int { signed: true }
        );
    }

    #[test]
    fn promote_int_unchanged() {
        let int_ty = CType::Int { signed: true };
        assert_eq!(integer_promote(&int_ty), int_ty);
        let uint_ty = CType::Int { signed: false };
        assert_eq!(integer_promote(&uint_ty), uint_ty);
    }

    #[test]
    fn promote_long_unchanged() {
        let long_ty = CType::Long { signed: true };
        assert_eq!(integer_promote(&long_ty), long_ty);
    }

    // ----- usual_arithmetic_conversion -----------------------------------

    #[test]
    fn uac_floating_dominance() {
        // long double wins over everything.
        assert_eq!(
            usual_arithmetic_conversion(&CType::Int { signed: true }, &CType::LongDouble),
            CType::LongDouble,
        );
        // double wins over int.
        assert_eq!(
            usual_arithmetic_conversion(&CType::Int { signed: true }, &CType::Double),
            CType::Double,
        );
        // float wins over int.
        assert_eq!(
            usual_arithmetic_conversion(&CType::Int { signed: true }, &CType::Float),
            CType::Float,
        );
    }

    #[test]
    fn uac_same_type() {
        assert_eq!(
            usual_arithmetic_conversion(
                &CType::Int { signed: true },
                &CType::Int { signed: true },
            ),
            CType::Int { signed: true },
        );
    }

    #[test]
    fn uac_mixed_sign_same_rank() {
        // unsigned int vs signed int → unsigned int (unsigned rank >= signed rank).
        assert_eq!(
            usual_arithmetic_conversion(
                &CType::Int { signed: false },
                &CType::Int { signed: true },
            ),
            CType::Int { signed: false },
        );
    }

    #[test]
    fn uac_signed_wider_than_unsigned() {
        // signed long vs unsigned int → signed long (long is wider).
        assert_eq!(
            usual_arithmetic_conversion(
                &CType::Long { signed: true },
                &CType::Int { signed: false },
            ),
            CType::Long { signed: true },
        );
    }

    // ----- Display -------------------------------------------------------

    #[test]
    fn display_basic_types() {
        assert_eq!(format!("{}", CType::Void), "void");
        assert_eq!(format!("{}", CType::Bool), "_Bool");
        assert_eq!(format!("{}", CType::Int { signed: true }), "int");
        assert_eq!(
            format!("{}", CType::Int { signed: false }),
            "unsigned int"
        );
        assert_eq!(format!("{}", CType::LongDouble), "long double");
    }

    #[test]
    fn display_pointer() {
        let ptr = CType::Pointer(Box::new(CType::Int { signed: true }));
        assert_eq!(format!("{}", ptr), "int *");
    }

    #[test]
    fn display_function_pointer() {
        let fptr = CType::Pointer(Box::new(CType::Function {
            return_type: Box::new(CType::Void),
            params: vec![CType::Int { signed: true }, CType::Int { signed: true }],
            variadic: false,
        }));
        assert_eq!(format!("{}", fptr), "void (*)(int, int)");
    }

    #[test]
    fn display_struct_and_union() {
        let s = CType::Struct {
            name: Some("point".into()),
            fields: vec![],
        };
        assert_eq!(format!("{}", s), "struct point");
        let u = CType::Union {
            name: None,
            fields: vec![],
        };
        assert_eq!(format!("{}", u), "union <anonymous>");
    }

    #[test]
    fn display_atomic() {
        let a = CType::Atomic(Box::new(CType::Int { signed: true }));
        assert_eq!(format!("{}", a), "_Atomic(int)");
    }

    #[test]
    fn display_qualified_type() {
        let qt = QualifiedType {
            ty: CType::Int { signed: true },
            qualifiers: TypeQualifiers {
                is_const: true,
                is_volatile: true,
                is_restrict: false,
                is_atomic: false,
            },
        };
        assert_eq!(format!("{}", qt), "const volatile int");
    }

    #[test]
    fn display_machine_type() {
        assert_eq!(format!("{}", MachineType::I32), "i32");
        assert_eq!(format!("{}", MachineType::F64), "f64");
        assert_eq!(format!("{}", MachineType::Aggregate(24)), "agg(24)");
        assert_eq!(format!("{}", MachineType::Ptr), "ptr");
    }

    // ----- TypeQualifiers ------------------------------------------------

    #[test]
    fn qualifiers_default_empty() {
        let q = TypeQualifiers::default();
        assert!(q.is_empty());
    }

    #[test]
    fn qualifiers_merge() {
        let a = TypeQualifiers {
            is_const: true,
            is_volatile: false,
            is_restrict: false,
            is_atomic: false,
        };
        let b = TypeQualifiers {
            is_const: false,
            is_volatile: true,
            is_restrict: false,
            is_atomic: true,
        };
        let merged = a.merge(&b);
        assert!(merged.is_const);
        assert!(merged.is_volatile);
        assert!(!merged.is_restrict);
        assert!(merged.is_atomic);
    }

    // ----- canonical / integer_rank --------------------------------------

    #[test]
    fn canonical_strips_typedef() {
        let typedef_int = CType::Typedef {
            name: "myint".into(),
            underlying: Box::new(CType::Int { signed: true }),
        };
        assert_eq!(typedef_int.canonical(), &CType::Int { signed: true });
    }

    #[test]
    fn integer_rank_ordering() {
        assert!(CType::Bool.integer_rank().unwrap() < CType::Char { signed: true }.integer_rank().unwrap());
        assert!(CType::Char { signed: true }.integer_rank().unwrap() < CType::Short { signed: true }.integer_rank().unwrap());
        assert!(CType::Short { signed: true }.integer_rank().unwrap() < CType::Int { signed: true }.integer_rank().unwrap());
        assert!(CType::Int { signed: true }.integer_rank().unwrap() < CType::Long { signed: true }.integer_rank().unwrap());
        assert!(CType::Long { signed: true }.integer_rank().unwrap() < CType::LongLong { signed: true }.integer_rank().unwrap());
    }

    #[test]
    fn float_has_no_integer_rank() {
        assert!(CType::Float.integer_rank().is_none());
        assert!(CType::Double.integer_rank().is_none());
    }
}
