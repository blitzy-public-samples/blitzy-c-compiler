//! IR type system for the BCC compiler.
//!
//! This module defines [`IrType`], the machine-level type representation used
//! throughout the intermediate representation (IR).  IR types serve as the
//! bridge between C language types ([`CType`]) used in the frontend and
//! machine register classes used in the backend code generators.
//!
//! # Design Philosophy
//!
//! IR types are deliberately simpler and more regular than C types:
//!
//! - **No signedness tracking**: Integer signedness is captured by the
//!   *operations* (e.g., `sdiv` vs `udiv`, `sext` vs `zext`), not the types.
//! - **No typedef/atomic wrappers**: These are stripped during the
//!   [`from_c_type`](IrType::from_c_type) conversion.
//! - **Opaque pointers**: [`Ptr`](IrType::Ptr) carries no pointee type,
//!   following the modern LLVM opaque-pointer convention.
//! - **Explicit aggregate layout**: Struct field padding and alignment are
//!   computed from target properties, not carried implicitly.
//!
//! # Target-Dependent Sizing
//!
//! Several types have target-dependent sizes and alignments:
//!
//! | IR Type | x86-64    | i686      | AArch64   | RISC-V 64 |
//! |---------|-----------|-----------|-----------|-----------|
//! | Ptr     | 8B / 8A   | 4B / 4A   | 8B / 8A   | 8B / 8A   |
//! | I64     | 8B / 8A   | 8B / 4A   | 8B / 8A   | 8B / 8A   |
//! | F64     | 8B / 8A   | 8B / 4A   | 8B / 8A   | 8B / 8A   |
//! | F80     | 16B / 16A | 12B / 4A  | (n/a)     | (n/a)     |
//!
//! F80 should not normally appear on AArch64/RISC-V targets;
//! `CType::LongDouble` maps to `F64` on those architectures.

use std::fmt;

use crate::common::target::{DataModel, Target};
use crate::common::types::{CType, FieldDef};

// ---------------------------------------------------------------------------
// IrType — IR type system
// ---------------------------------------------------------------------------

/// Machine-level type representation for the BCC intermediate representation.
///
/// Every IR value, instruction operand, and function signature references an
/// `IrType`.  The enum covers all types needed for C11 code generation:
/// booleans, integers of various widths, IEEE floating-point types, the x87
/// extended-precision type, opaque pointers, arrays, structs, and function
/// signatures.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IrType {
    /// `void` — zero-sized type used for function return types and sentinels.
    Void,

    /// 1-bit boolean.  Stored as 1 byte in memory; produced by comparison
    /// instructions and used for `_Bool` representation.
    I1,

    /// 8-bit integer — `char`, `signed char`, `unsigned char`.
    I8,

    /// 16-bit integer — `short`, `unsigned short`.
    I16,

    /// 32-bit integer — `int`, `unsigned int`, `long` on ILP32.
    I32,

    /// 64-bit integer — `long` on LP64, `long long`, `unsigned long long`.
    I64,

    /// 128-bit integer — `__int128` (GCC extension).
    I128,

    /// IEEE 754 single-precision (32-bit) floating point — `float`.
    F32,

    /// IEEE 754 double-precision (64-bit) floating point — `double`.
    F64,

    /// x87 extended-precision (80-bit) floating point — `long double` on
    /// x86-64 and i686.  Storage size is target-dependent:
    /// - x86-64: 16 bytes (padded for 16-byte alignment)
    /// - i686:   12 bytes (padded per SysV i386 ABI)
    F80,

    /// Opaque pointer — target-width (32-bit on i686, 64-bit on LP64 targets).
    /// Carries no pointee type information (opaque pointer model).
    Ptr,

    /// Fixed-size array: `count` elements of type `element`.
    Array {
        /// The element type of the array.
        element: Box<IrType>,
        /// Number of elements (0 for incomplete/flexible arrays).
        count: usize,
    },

    /// Struct type: ordered list of field types with optional packing.
    ///
    /// When `packed` is `true`, fields are laid out without inter-field or
    /// trailing alignment padding (equivalent to `__attribute__((packed))`).
    Struct {
        /// Field types in declaration order.
        fields: Vec<IrType>,
        /// If `true`, no alignment padding is inserted between fields.
        packed: bool,
    },

    /// Function type: return type, parameter types, and variadic flag.
    Function {
        /// Return type of the function.
        return_type: Box<IrType>,
        /// Parameter types in declaration order.
        param_types: Vec<IrType>,
        /// `true` if the function accepts variadic arguments (`...`).
        is_variadic: bool,
    },
}

// ---------------------------------------------------------------------------
// Type predicate methods
// ---------------------------------------------------------------------------

impl IrType {
    /// Returns `true` if this type is [`Void`](IrType::Void).
    #[inline]
    pub fn is_void(&self) -> bool {
        matches!(self, IrType::Void)
    }

    /// Returns `true` for any integer type: `I1`, `I8`, `I16`, `I32`, `I64`,
    /// or `I128`.
    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128
        )
    }

    /// Returns `true` for any floating-point type: `F32`, `F64`, or `F80`.
    #[inline]
    pub fn is_floating(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64 | IrType::F80)
    }

    /// Returns `true` for the opaque pointer type [`Ptr`](IrType::Ptr).
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(self, IrType::Ptr)
    }

    /// Returns `true` for aggregate types: [`Array`](IrType::Array) and
    /// [`Struct`](IrType::Struct).
    #[inline]
    pub fn is_aggregate(&self) -> bool {
        matches!(self, IrType::Array { .. } | IrType::Struct { .. })
    }

    /// Returns `true` for scalar types: integers, floats, and pointers.
    ///
    /// Scalars can fit in a single register (or register pair) and are directly
    /// manipulated by arithmetic/comparison instructions.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.is_integer() || self.is_floating() || self.is_pointer()
    }

    /// Returns `true` for [`Function`](IrType::Function) types.
    #[inline]
    pub fn is_function(&self) -> bool {
        matches!(self, IrType::Function { .. })
    }

    /// Returns `true` for [`Array`](IrType::Array) types.
    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self, IrType::Array { .. })
    }

    /// Returns `true` for [`Struct`](IrType::Struct) types.
    #[inline]
    pub fn is_struct(&self) -> bool {
        matches!(self, IrType::Struct { .. })
    }

    // -----------------------------------------------------------------------
    // Size and alignment
    // -----------------------------------------------------------------------

    /// Returns the logical data width of this type in bits.
    ///
    /// For aggregate types (arrays, structs), this equals `size_bytes * 8`
    /// since the byte size already accounts for alignment padding.  For `F80`,
    /// the logical width is 80 bits even though storage is 12 or 16 bytes.
    ///
    /// # Parameters
    ///
    /// * `target` — the target architecture, needed for target-dependent types
    ///   like `Ptr` and aggregate layout.
    pub fn size_bits(&self, target: &Target) -> u64 {
        match self {
            IrType::Void => 0,
            IrType::I1 => 1,
            IrType::I8 => 8,
            IrType::I16 => 16,
            IrType::I32 => 32,
            IrType::I64 => 64,
            IrType::I128 => 128,
            IrType::F32 => 32,
            IrType::F64 => 64,
            IrType::F80 => 80,
            IrType::Ptr => u64::from(target.pointer_width()) * 8,
            // Aggregate/function types return their storage size in bits.
            IrType::Array { .. } | IrType::Struct { .. } => self.size_bytes(target) * 8,
            IrType::Function { .. } => 0,
        }
    }

    /// Returns the storage size of this type in bytes.
    ///
    /// This is the number of bytes the type occupies in memory, *including*
    /// any alignment padding.  For `F80`, the storage size is
    /// target-dependent: 16 bytes on x86-64 (padded for 16-byte alignment),
    /// 12 bytes on i686 (padded per SysV i386 ABI).
    ///
    /// For structs, the result includes both inter-field and trailing padding
    /// to satisfy the struct's overall alignment.  For arrays, the result is
    /// simply `element.size_bytes * count`.
    pub fn size_bytes(&self, target: &Target) -> u64 {
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
                // x86-64: 80-bit data padded to 16 bytes (128 bits) for
                //         16-byte natural alignment.
                // i686:   80-bit data padded to 12 bytes (96 bits) per
                //         SysV i386 ABI.
                // Other:  F80 is not native; use 16 bytes as fallback.
                match *target {
                    Target::X86_64 => 16,
                    Target::I686 => 12,
                    Target::AArch64 | Target::RiscV64 => 16,
                }
            }
            IrType::Ptr => u64::from(target.pointer_width()),
            IrType::Array { element, count } => element.size_bytes(target) * (*count as u64),
            IrType::Struct { fields, packed } => Self::compute_struct_size(fields, *packed, target),
            IrType::Function { .. } => 0,
        }
    }

    /// Returns the ABI alignment requirement of this type in bytes.
    ///
    /// On i686, 64-bit and wider scalar types have only 4-byte alignment per
    /// the SysV i386 ABI.  On 64-bit targets, alignment matches the natural
    /// type width.
    ///
    /// For packed structs, alignment is always 1.  For unpacked structs,
    /// alignment is the maximum alignment among all fields.
    pub fn alignment(&self, target: &Target) -> u64 {
        match self {
            IrType::Void | IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 => 4,
            IrType::I64 => {
                // SysV i386 ABI: `long long` has 4-byte alignment.
                match *target {
                    Target::I686 => 4,
                    Target::X86_64 | Target::AArch64 | Target::RiscV64 => 8,
                }
            }
            IrType::I128 => match *target {
                Target::I686 => 4,
                Target::X86_64 | Target::AArch64 | Target::RiscV64 => 16,
            },
            IrType::F32 => 4,
            IrType::F64 => {
                // SysV i386 ABI: `double` has 4-byte alignment.
                match *target {
                    Target::I686 => 4,
                    Target::X86_64 | Target::AArch64 | Target::RiscV64 => 8,
                }
            }
            IrType::F80 => {
                // x86-64: 16-byte alignment for long double.
                // i686:   4-byte alignment per SysV i386 ABI.
                // Other:  F80 is non-native; use 16-byte alignment as fallback.
                match *target {
                    Target::X86_64 => 16,
                    Target::I686 => 4,
                    Target::AArch64 | Target::RiscV64 => 16,
                }
            }
            IrType::Ptr => u64::from(target.pointer_width()),
            IrType::Array { element, .. } => {
                // Array alignment equals its element alignment.
                element.alignment(target)
            }
            IrType::Struct { fields, packed } => {
                if *packed {
                    // Packed structs have byte alignment.
                    return 1;
                }
                // Struct alignment is the maximum field alignment.
                fields
                    .iter()
                    .map(|f| f.alignment(target))
                    .max()
                    .unwrap_or(1)
            }
            IrType::Function { .. } => {
                // Function types are not stored in memory as values;
                // use 1-byte alignment as a sentinel.
                1
            }
        }
    }

    // -----------------------------------------------------------------------
    // Integer width helpers
    // -----------------------------------------------------------------------

    /// Returns the bit width for integer types, or `None` for non-integer types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(IrType::I32.integer_width(), Some(32));
    /// assert_eq!(IrType::F64.integer_width(), None);
    /// ```
    #[inline]
    pub fn integer_width(&self) -> Option<u32> {
        match self {
            IrType::I1 => Some(1),
            IrType::I8 => Some(8),
            IrType::I16 => Some(16),
            IrType::I32 => Some(32),
            IrType::I64 => Some(64),
            IrType::I128 => Some(128),
            _ => None,
        }
    }

    /// Returns the integer IR type for a given bit width.
    ///
    /// Supported widths are 1, 8, 16, 32, 64, and 128.  Other widths are
    /// rounded up to the nearest supported width, saturating at `I128`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(IrType::int_type_for_width(32), IrType::I32);
    /// assert_eq!(IrType::int_type_for_width(24), IrType::I32); // rounded up
    /// ```
    pub fn int_type_for_width(width: u32) -> IrType {
        match width {
            0 | 1 => IrType::I1,
            2..=8 => IrType::I8,
            9..=16 => IrType::I16,
            17..=32 => IrType::I32,
            33..=64 => IrType::I64,
            _ => IrType::I128,
        }
    }

    /// Returns a pointer-sized integer type for the given target.
    ///
    /// * i686: [`I32`](IrType::I32)
    /// * x86-64, AArch64, RISC-V 64: [`I64`](IrType::I64)
    #[inline]
    pub fn ptr_sized_int(target: &Target) -> IrType {
        match target.data_model() {
            DataModel::ILP32 => IrType::I32,
            DataModel::LP64 => IrType::I64,
        }
    }

    // -----------------------------------------------------------------------
    // Element type access
    // -----------------------------------------------------------------------

    /// Returns the element type for [`Array`](IrType::Array) types.
    ///
    /// Returns `None` for all other types including [`Ptr`](IrType::Ptr),
    /// which is opaque and carries no pointee type.
    pub fn element_type(&self) -> Option<&IrType> {
        match self {
            IrType::Array { element, .. } => Some(element),
            _ => None,
        }
    }

    /// Returns the field type slice for [`Struct`](IrType::Struct) types.
    ///
    /// Returns `None` for non-struct types.
    pub fn struct_fields(&self) -> Option<&[IrType]> {
        match self {
            IrType::Struct { fields, .. } => Some(fields),
            _ => None,
        }
    }

    /// Returns the return type for [`Function`](IrType::Function) types.
    ///
    /// Returns `None` for non-function types.
    pub fn function_return_type(&self) -> Option<&IrType> {
        match self {
            IrType::Function { return_type, .. } => Some(return_type),
            _ => None,
        }
    }

    /// Returns the parameter type slice for [`Function`](IrType::Function) types.
    ///
    /// Returns `None` for non-function types.
    pub fn function_param_types(&self) -> Option<&[IrType]> {
        match self {
            IrType::Function { param_types, .. } => Some(param_types),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Struct layout computation
    // -----------------------------------------------------------------------

    /// Computes the byte offset of a field within a struct.
    ///
    /// Fields are laid out sequentially with each field aligned to its natural
    /// alignment (or byte-packed if `packed` is `true`).  If `index` is out of
    /// bounds, the returned offset corresponds to the first byte after the
    /// last field (i.e., the raw struct size before trailing padding).
    ///
    /// # Parameters
    ///
    /// * `fields` — the field types of the struct.
    /// * `index` — the zero-based field index to compute the offset for.
    /// * `packed` — if `true`, no alignment padding is inserted.
    /// * `target` — the target architecture for alignment computation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // struct { i32, i8, i32 } on x86-64:
    /// //   field 0 at offset 0
    /// //   field 1 at offset 4
    /// //   field 2 at offset 8 (aligned from 5 to 8)
    /// let fields = vec![IrType::I32, IrType::I8, IrType::I32];
    /// assert_eq!(IrType::struct_field_offset(&fields, 0, false, &target), 0);
    /// assert_eq!(IrType::struct_field_offset(&fields, 1, false, &target), 4);
    /// assert_eq!(IrType::struct_field_offset(&fields, 2, false, &target), 8);
    /// ```
    pub fn struct_field_offset(
        fields: &[IrType],
        index: usize,
        packed: bool,
        target: &Target,
    ) -> u64 {
        let mut offset: u64 = 0;

        for (i, field) in fields.iter().enumerate() {
            // Align the current offset to this field's alignment requirement.
            if !packed {
                let align = field.alignment(target);
                offset = align_up(offset, align);
            }

            // If we have reached the target field, return its offset.
            if i == index {
                return offset;
            }

            // Advance past this field's storage.
            offset += field.size_bytes(target);
        }

        // `index` is at or beyond the end — return the offset at the end of
        // all fields (before trailing struct padding).
        offset
    }

    // -----------------------------------------------------------------------
    // C type → IR type conversion
    // -----------------------------------------------------------------------

    /// Converts a C language type ([`CType`]) to an IR type.
    ///
    /// This is the primary bridge between the frontend type system and the
    /// middle-end IR.  The conversion strips signedness (carried by IR
    /// instructions instead), resolves typedef chains, unwraps `_Atomic`
    /// wrappers, and maps target-dependent types (e.g., `long`, `long double`)
    /// to the appropriate IR integer/float width.
    ///
    /// # Union Representation
    ///
    /// C unions are represented as an IR struct containing the most-aligned
    /// field followed by a byte-array pad to reach the full union size.  This
    /// preserves both the size and alignment of the original union.
    ///
    /// # Bit-Fields
    ///
    /// Bit-field layout is not captured in the IR type system; bit-field access
    /// is lowered to load/shift/mask operations during AST→IR lowering.  The
    /// IR struct contains the *underlying* type of each bit-field member.
    pub fn from_c_type(ctype: &CType, target: &Target) -> IrType {
        match ctype {
            CType::Void => IrType::Void,
            CType::Bool => IrType::I1,

            // Integer types — signedness is not tracked in IR.
            CType::Char { .. } => IrType::I8,
            CType::Short { .. } => IrType::I16,
            CType::Int { .. } => IrType::I32,
            CType::Long { .. } => {
                // LP64 targets (long_size == 8): long is 64-bit.
                // ILP32 targets (long_size == 4): long is 32-bit.
                if target.long_size() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                }
            }
            CType::LongLong { .. } => IrType::I64,

            // Floating-point types.
            CType::Float => IrType::F32,
            CType::Double => IrType::F64,
            CType::LongDouble => {
                // long_double_size() returns:
                //   16 (x86-64) → F80 (80-bit extended, 16-byte padded storage)
                //   12 (i686)   → F80 (80-bit extended, 12-byte padded storage)
                //    8 (AArch64/RISC-V) → F64 (IEEE double)
                if target.long_double_size() == 8 {
                    IrType::F64
                } else {
                    IrType::F80
                }
            }

            // _Complex T is represented as a two-element struct { T, T }.
            CType::Complex(inner) => {
                let elem = IrType::from_c_type(inner, target);
                IrType::Struct {
                    fields: vec![elem.clone(), elem],
                    packed: false,
                }
            }

            // Pointers — opaque in the IR.
            CType::Pointer(_) => IrType::Ptr,

            // Arrays — convert element type, use 0 for unknown size.
            CType::Array { element, size } => IrType::Array {
                element: Box::new(IrType::from_c_type(element, target)),
                count: size.unwrap_or(0),
            },

            // Function types.
            CType::Function {
                return_type,
                params,
                variadic,
            } => IrType::Function {
                return_type: Box::new(IrType::from_c_type(return_type, target)),
                param_types: params
                    .iter()
                    .map(|p| IrType::from_c_type(p, target))
                    .collect(),
                is_variadic: *variadic,
            },

            // Struct — convert each field's underlying type.
            CType::Struct { fields, .. } => {
                let ir_fields = Self::convert_field_defs(fields, target);
                IrType::Struct {
                    fields: ir_fields,
                    packed: false,
                }
            }

            // Union — represented as a struct preserving size and alignment.
            CType::Union { fields, .. } => Self::convert_union(fields, target),

            // Enum — lower to the underlying integer type.
            CType::Enum { underlying, .. } => IrType::from_c_type(underlying, target),

            // _Atomic(T) — strip the atomic wrapper; atomic semantics are
            // handled by instruction-level annotations.
            CType::Atomic(inner) => IrType::from_c_type(inner, target),

            // typedef — resolve to the underlying type.
            CType::Typedef { underlying, .. } => IrType::from_c_type(underlying, target),
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Computes the total size of a struct in bytes, including inter-field
    /// alignment padding and trailing padding to the struct's alignment.
    fn compute_struct_size(fields: &[IrType], packed: bool, target: &Target) -> u64 {
        if fields.is_empty() {
            return 0;
        }

        let mut offset: u64 = 0;
        for field in fields {
            if !packed {
                let align = field.alignment(target);
                offset = align_up(offset, align);
            }
            offset += field.size_bytes(target);
        }

        // Add trailing padding to satisfy the struct's overall alignment.
        if !packed {
            let struct_align = fields
                .iter()
                .map(|f| f.alignment(target))
                .max()
                .unwrap_or(1);
            offset = align_up(offset, struct_align);
        }

        offset
    }

    /// Converts a slice of [`FieldDef`]s (struct/union fields) to a vector of
    /// IR types.  Bit-field widths are ignored; the underlying type is used
    /// directly.
    fn convert_field_defs(fields: &[FieldDef], target: &Target) -> Vec<IrType> {
        fields
            .iter()
            .map(|f| IrType::from_c_type(&f.ty, target))
            .collect()
    }

    /// Converts a C `union` into an IR `Struct` that preserves the union's
    /// size and alignment.
    ///
    /// The resulting struct contains the most-aligned field as its first
    /// element, followed by a byte-array padding element (if needed) to reach
    /// the full union size.
    fn convert_union(fields: &[FieldDef], target: &Target) -> IrType {
        if fields.is_empty() {
            return IrType::Struct {
                fields: vec![],
                packed: false,
            };
        }

        // Compute the union's total size and maximum alignment, and find the
        // single field that has the highest alignment.  In case of ties,
        // prefer the larger field so we minimise padding.
        let mut max_size: u64 = 0;
        let mut max_align: u64 = 1;
        let mut best_field = IrType::I8;
        let mut best_field_align: u64 = 0;
        let mut best_field_size: u64 = 0;

        for field_def in fields {
            let ir_ty = IrType::from_c_type(&field_def.ty, target);
            let field_size = ir_ty.size_bytes(target);
            let field_align = ir_ty.alignment(target);

            if field_size > max_size {
                max_size = field_size;
            }
            if field_align > max_align {
                max_align = field_align;
            }

            // Choose the field with the highest alignment; break ties by size.
            if field_align > best_field_align
                || (field_align == best_field_align && field_size > best_field_size)
            {
                best_field = ir_ty;
                best_field_align = field_align;
                best_field_size = field_size;
            }
        }

        let mut ir_fields = vec![best_field];

        // If the chosen field is smaller than the full union size, add a
        // byte-array pad so that the struct's total raw size equals the union
        // size.  The struct's trailing padding (from alignment) will be added
        // automatically by `compute_struct_size`.
        let remaining = max_size.saturating_sub(best_field_size);
        if remaining > 0 {
            ir_fields.push(IrType::Array {
                element: Box::new(IrType::I8),
                count: remaining as usize,
            });
        }

        IrType::Struct {
            fields: ir_fields,
            packed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Display implementation
// ---------------------------------------------------------------------------

impl fmt::Display for IrType {
    /// Formats the IR type in a human-readable textual representation:
    ///
    /// | Type      | Format                             |
    /// |-----------|------------------------------------|
    /// | Void      | `void`                             |
    /// | I1…I128   | `i1`, `i8`, `i16`, `i32`, …       |
    /// | F32…F80   | `f32`, `f64`, `f80`                |
    /// | Ptr       | `ptr`                              |
    /// | Array     | `[N x <ty>]`                       |
    /// | Struct    | `{<ty>, <ty>, …}` or `<{…}>` packed|
    /// | Function  | `fn(<ty>, <ty>) -> <ty>`           |
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => f.write_str("void"),
            IrType::I1 => f.write_str("i1"),
            IrType::I8 => f.write_str("i8"),
            IrType::I16 => f.write_str("i16"),
            IrType::I32 => f.write_str("i32"),
            IrType::I64 => f.write_str("i64"),
            IrType::I128 => f.write_str("i128"),
            IrType::F32 => f.write_str("f32"),
            IrType::F64 => f.write_str("f64"),
            IrType::F80 => f.write_str("f80"),
            IrType::Ptr => f.write_str("ptr"),

            IrType::Array { element, count } => {
                write!(f, "[{} x {}]", count, element)
            }

            IrType::Struct { fields, packed } => {
                if *packed {
                    f.write_str("<")?;
                }
                f.write_str("{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    fmt::Display::fmt(field, f)?;
                }
                f.write_str("}")?;
                if *packed {
                    f.write_str(">")?;
                }
                Ok(())
            }

            IrType::Function {
                return_type,
                param_types,
                is_variadic,
            } => {
                f.write_str("fn(")?;
                for (i, param) in param_types.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    fmt::Display::fmt(param, f)?;
                }
                if *is_variadic {
                    if !param_types.is_empty() {
                        f.write_str(", ")?;
                    }
                    f.write_str("...")?;
                }
                write!(f, ") -> {}", return_type)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing helper functions
// ---------------------------------------------------------------------------

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` must be a power of two.  If `align` is zero or one, `value` is
/// returned unchanged.
#[inline]
fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    // align is a power of two, so (align - 1) is a bitmask.
    (value + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Predicate tests ----------------------------------------------------

    #[test]
    fn test_is_void() {
        assert!(IrType::Void.is_void());
        assert!(!IrType::I32.is_void());
    }

    #[test]
    fn test_is_integer() {
        assert!(IrType::I1.is_integer());
        assert!(IrType::I8.is_integer());
        assert!(IrType::I16.is_integer());
        assert!(IrType::I32.is_integer());
        assert!(IrType::I64.is_integer());
        assert!(IrType::I128.is_integer());
        assert!(!IrType::F32.is_integer());
        assert!(!IrType::Void.is_integer());
        assert!(!IrType::Ptr.is_integer());
    }

    #[test]
    fn test_is_floating() {
        assert!(IrType::F32.is_floating());
        assert!(IrType::F64.is_floating());
        assert!(IrType::F80.is_floating());
        assert!(!IrType::I32.is_floating());
        assert!(!IrType::Ptr.is_floating());
    }

    #[test]
    fn test_is_pointer() {
        assert!(IrType::Ptr.is_pointer());
        assert!(!IrType::I64.is_pointer());
    }

    #[test]
    fn test_is_aggregate() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 4,
        };
        let st = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(arr.is_aggregate());
        assert!(st.is_aggregate());
        assert!(!IrType::I32.is_aggregate());
        assert!(!IrType::Ptr.is_aggregate());
    }

    #[test]
    fn test_is_scalar() {
        assert!(IrType::I32.is_scalar());
        assert!(IrType::F64.is_scalar());
        assert!(IrType::Ptr.is_scalar());
        assert!(!IrType::Void.is_scalar());
        let st = IrType::Struct {
            fields: vec![IrType::I32],
            packed: false,
        };
        assert!(!st.is_scalar());
    }

    #[test]
    fn test_is_function() {
        let func = IrType::Function {
            return_type: Box::new(IrType::Void),
            param_types: vec![IrType::I32],
            is_variadic: false,
        };
        assert!(func.is_function());
        assert!(!IrType::I32.is_function());
    }

    // -- Size tests ---------------------------------------------------------

    #[test]
    fn test_scalar_sizes_x86_64() {
        let t = Target::X86_64;
        assert_eq!(IrType::Void.size_bits(&t), 0);
        assert_eq!(IrType::I1.size_bits(&t), 1);
        assert_eq!(IrType::I8.size_bits(&t), 8);
        assert_eq!(IrType::I16.size_bits(&t), 16);
        assert_eq!(IrType::I32.size_bits(&t), 32);
        assert_eq!(IrType::I64.size_bits(&t), 64);
        assert_eq!(IrType::I128.size_bits(&t), 128);
        assert_eq!(IrType::F32.size_bits(&t), 32);
        assert_eq!(IrType::F64.size_bits(&t), 64);
        assert_eq!(IrType::F80.size_bits(&t), 80);
        assert_eq!(IrType::Ptr.size_bits(&t), 64);
    }

    #[test]
    fn test_scalar_sizes_bytes_x86_64() {
        let t = Target::X86_64;
        assert_eq!(IrType::I1.size_bytes(&t), 1);
        assert_eq!(IrType::I32.size_bytes(&t), 4);
        assert_eq!(IrType::I64.size_bytes(&t), 8);
        assert_eq!(IrType::F80.size_bytes(&t), 16);
        assert_eq!(IrType::Ptr.size_bytes(&t), 8);
    }

    #[test]
    fn test_scalar_sizes_i686() {
        let t = Target::I686;
        assert_eq!(IrType::Ptr.size_bits(&t), 32);
        assert_eq!(IrType::Ptr.size_bytes(&t), 4);
        assert_eq!(IrType::F80.size_bytes(&t), 12);
    }

    #[test]
    fn test_array_size() {
        let t = Target::X86_64;
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(arr.size_bytes(&t), 40);
        assert_eq!(arr.size_bits(&t), 320);
        assert_eq!(arr.alignment(&t), 4);
    }

    #[test]
    fn test_struct_size_x86_64() {
        let t = Target::X86_64;

        // struct { i32, i8, i32 } → offset 0, 4, 8; size = 12
        let st = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8, IrType::I32],
            packed: false,
        };
        assert_eq!(st.size_bytes(&t), 12);
        assert_eq!(st.alignment(&t), 4);

        // struct { i8, i64 } → offset 0, 8; size = 16
        let st2 = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: false,
        };
        assert_eq!(st2.size_bytes(&t), 16);
        assert_eq!(st2.alignment(&t), 8);
    }

    #[test]
    fn test_packed_struct() {
        let t = Target::X86_64;
        let st = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: true,
        };
        // Packed: no padding → 1 + 8 = 9
        assert_eq!(st.size_bytes(&t), 9);
        assert_eq!(st.alignment(&t), 1);
    }

    #[test]
    fn test_empty_struct_and_array() {
        let t = Target::X86_64;
        let empty_struct = IrType::Struct {
            fields: vec![],
            packed: false,
        };
        assert_eq!(empty_struct.size_bytes(&t), 0);

        let empty_arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 0,
        };
        assert_eq!(empty_arr.size_bytes(&t), 0);
    }

    // -- Alignment tests ----------------------------------------------------

    #[test]
    fn test_alignment_x86_64() {
        let t = Target::X86_64;
        assert_eq!(IrType::I1.alignment(&t), 1);
        assert_eq!(IrType::I8.alignment(&t), 1);
        assert_eq!(IrType::I16.alignment(&t), 2);
        assert_eq!(IrType::I32.alignment(&t), 4);
        assert_eq!(IrType::I64.alignment(&t), 8);
        assert_eq!(IrType::I128.alignment(&t), 16);
        assert_eq!(IrType::F32.alignment(&t), 4);
        assert_eq!(IrType::F64.alignment(&t), 8);
        assert_eq!(IrType::F80.alignment(&t), 16);
        assert_eq!(IrType::Ptr.alignment(&t), 8);
    }

    #[test]
    fn test_alignment_i686() {
        let t = Target::I686;
        assert_eq!(IrType::I64.alignment(&t), 4);
        assert_eq!(IrType::F64.alignment(&t), 4);
        assert_eq!(IrType::F80.alignment(&t), 4);
        assert_eq!(IrType::Ptr.alignment(&t), 4);
        assert_eq!(IrType::I128.alignment(&t), 4);
    }

    // -- Integer width tests ------------------------------------------------

    #[test]
    fn test_integer_width() {
        assert_eq!(IrType::I1.integer_width(), Some(1));
        assert_eq!(IrType::I8.integer_width(), Some(8));
        assert_eq!(IrType::I32.integer_width(), Some(32));
        assert_eq!(IrType::I128.integer_width(), Some(128));
        assert_eq!(IrType::F64.integer_width(), None);
        assert_eq!(IrType::Ptr.integer_width(), None);
    }

    #[test]
    fn test_int_type_for_width() {
        assert_eq!(IrType::int_type_for_width(1), IrType::I1);
        assert_eq!(IrType::int_type_for_width(8), IrType::I8);
        assert_eq!(IrType::int_type_for_width(16), IrType::I16);
        assert_eq!(IrType::int_type_for_width(32), IrType::I32);
        assert_eq!(IrType::int_type_for_width(64), IrType::I64);
        assert_eq!(IrType::int_type_for_width(128), IrType::I128);
        // Rounding up
        assert_eq!(IrType::int_type_for_width(3), IrType::I8);
        assert_eq!(IrType::int_type_for_width(24), IrType::I32);
    }

    // -- Pointer-sized integer tests ----------------------------------------

    #[test]
    fn test_ptr_sized_int() {
        assert_eq!(IrType::ptr_sized_int(&Target::X86_64), IrType::I64);
        assert_eq!(IrType::ptr_sized_int(&Target::AArch64), IrType::I64);
        assert_eq!(IrType::ptr_sized_int(&Target::RiscV64), IrType::I64);
        assert_eq!(IrType::ptr_sized_int(&Target::I686), IrType::I32);
    }

    // -- Element type accessor tests ----------------------------------------

    #[test]
    fn test_element_type() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 5,
        };
        assert_eq!(arr.element_type(), Some(&IrType::I32));
        assert_eq!(IrType::Ptr.element_type(), None);
        assert_eq!(IrType::I32.element_type(), None);
    }

    #[test]
    fn test_struct_fields_accessor() {
        let st = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8],
            packed: false,
        };
        assert_eq!(st.struct_fields(), Some(&[IrType::I32, IrType::I8][..]));
        assert_eq!(IrType::I32.struct_fields(), None);
    }

    #[test]
    fn test_function_accessors() {
        let func = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::Ptr, IrType::I64],
            is_variadic: true,
        };
        assert_eq!(func.function_return_type(), Some(&IrType::I32));
        assert_eq!(
            func.function_param_types(),
            Some(&[IrType::Ptr, IrType::I64][..])
        );
        assert_eq!(IrType::I32.function_return_type(), None);
    }

    // -- Struct field offset tests ------------------------------------------

    #[test]
    fn test_struct_field_offset_basic() {
        let t = Target::X86_64;
        let fields = vec![IrType::I32, IrType::I8, IrType::I32];

        // field 0: offset 0, aligned to 4 → 0
        assert_eq!(IrType::struct_field_offset(&fields, 0, false, &t), 0);
        // field 1: after field 0 (offset 4), aligned to 1 → 4
        assert_eq!(IrType::struct_field_offset(&fields, 1, false, &t), 4);
        // field 2: after field 1 (offset 5), aligned to 4 → 8
        assert_eq!(IrType::struct_field_offset(&fields, 2, false, &t), 8);
    }

    #[test]
    fn test_struct_field_offset_packed() {
        let t = Target::X86_64;
        let fields = vec![IrType::I8, IrType::I64, IrType::I8];

        assert_eq!(IrType::struct_field_offset(&fields, 0, true, &t), 0);
        assert_eq!(IrType::struct_field_offset(&fields, 1, true, &t), 1);
        assert_eq!(IrType::struct_field_offset(&fields, 2, true, &t), 9);
    }

    #[test]
    fn test_struct_field_offset_with_i64_alignment() {
        let t = Target::X86_64;
        let fields = vec![IrType::I8, IrType::I64];

        assert_eq!(IrType::struct_field_offset(&fields, 0, false, &t), 0);
        // I8 at 0, then I64 aligned to 8 → offset 8
        assert_eq!(IrType::struct_field_offset(&fields, 1, false, &t), 8);
    }

    // -- from_c_type conversion tests ---------------------------------------

    #[test]
    fn test_from_c_type_basic() {
        let t = Target::X86_64;
        assert_eq!(IrType::from_c_type(&CType::Void, &t), IrType::Void);
        assert_eq!(IrType::from_c_type(&CType::Bool, &t), IrType::I1);
        assert_eq!(
            IrType::from_c_type(&CType::Char { signed: true }, &t),
            IrType::I8
        );
        assert_eq!(
            IrType::from_c_type(&CType::Char { signed: false }, &t),
            IrType::I8
        );
        assert_eq!(
            IrType::from_c_type(&CType::Short { signed: true }, &t),
            IrType::I16
        );
        assert_eq!(
            IrType::from_c_type(&CType::Int { signed: true }, &t),
            IrType::I32
        );
        assert_eq!(
            IrType::from_c_type(&CType::Int { signed: false }, &t),
            IrType::I32
        );
        assert_eq!(IrType::from_c_type(&CType::Float, &t), IrType::F32);
        assert_eq!(IrType::from_c_type(&CType::Double, &t), IrType::F64);
    }

    #[test]
    fn test_from_c_type_long_lp64() {
        let t = Target::X86_64;
        assert_eq!(
            IrType::from_c_type(&CType::Long { signed: true }, &t),
            IrType::I64
        );
        assert_eq!(
            IrType::from_c_type(&CType::LongLong { signed: true }, &t),
            IrType::I64
        );
    }

    #[test]
    fn test_from_c_type_long_ilp32() {
        let t = Target::I686;
        assert_eq!(
            IrType::from_c_type(&CType::Long { signed: true }, &t),
            IrType::I32
        );
    }

    #[test]
    fn test_from_c_type_long_double_x86() {
        assert_eq!(
            IrType::from_c_type(&CType::LongDouble, &Target::X86_64),
            IrType::F80
        );
        assert_eq!(
            IrType::from_c_type(&CType::LongDouble, &Target::I686),
            IrType::F80
        );
    }

    #[test]
    fn test_from_c_type_long_double_arm_riscv() {
        assert_eq!(
            IrType::from_c_type(&CType::LongDouble, &Target::AArch64),
            IrType::F64
        );
        assert_eq!(
            IrType::from_c_type(&CType::LongDouble, &Target::RiscV64),
            IrType::F64
        );
    }

    #[test]
    fn test_from_c_type_pointer() {
        let t = Target::X86_64;
        let ptr = CType::Pointer(Box::new(CType::Int { signed: true }));
        assert_eq!(IrType::from_c_type(&ptr, &t), IrType::Ptr);
    }

    #[test]
    fn test_from_c_type_array() {
        let t = Target::X86_64;
        let arr = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        assert_eq!(
            IrType::from_c_type(&arr, &t),
            IrType::Array {
                element: Box::new(IrType::I32),
                count: 10,
            }
        );
    }

    #[test]
    fn test_from_c_type_function() {
        let t = Target::X86_64;
        let func = CType::Function {
            return_type: Box::new(CType::Int { signed: true }),
            params: vec![CType::Pointer(Box::new(CType::Char { signed: true }))],
            variadic: true,
        };
        assert_eq!(
            IrType::from_c_type(&func, &t),
            IrType::Function {
                return_type: Box::new(IrType::I32),
                param_types: vec![IrType::Ptr],
                is_variadic: true,
            }
        );
    }

    #[test]
    fn test_from_c_type_complex() {
        let t = Target::X86_64;
        let cplx = CType::Complex(Box::new(CType::Double));
        assert_eq!(
            IrType::from_c_type(&cplx, &t),
            IrType::Struct {
                fields: vec![IrType::F64, IrType::F64],
                packed: false,
            }
        );
    }

    #[test]
    fn test_from_c_type_struct() {
        let t = Target::X86_64;
        let st = CType::Struct {
            name: Some("point".to_string()),
            fields: vec![
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
        };
        assert_eq!(
            IrType::from_c_type(&st, &t),
            IrType::Struct {
                fields: vec![IrType::I32, IrType::I32],
                packed: false,
            }
        );
    }

    #[test]
    fn test_from_c_type_union() {
        let t = Target::X86_64;
        let u = CType::Union {
            name: None,
            fields: vec![
                FieldDef {
                    name: Some("i".to_string()),
                    ty: CType::Int { signed: true },
                    bit_width: None,
                },
                FieldDef {
                    name: Some("d".to_string()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
        };
        let ir = IrType::from_c_type(&u, &t);

        // Union { int, double } on x86-64: size = 8, align = 8.
        // Represented as Struct { F64 } (F64 is the most-aligned and largest).
        assert_eq!(ir.size_bytes(&t), 8);
        assert_eq!(ir.alignment(&t), 8);
    }

    #[test]
    fn test_from_c_type_typedef() {
        let t = Target::X86_64;
        let td = CType::Typedef {
            name: "size_t".to_string(),
            underlying: Box::new(CType::Long { signed: false }),
        };
        assert_eq!(IrType::from_c_type(&td, &t), IrType::I64);
    }

    #[test]
    fn test_from_c_type_atomic() {
        let t = Target::X86_64;
        let at = CType::Atomic(Box::new(CType::Int { signed: true }));
        assert_eq!(IrType::from_c_type(&at, &t), IrType::I32);
    }

    #[test]
    fn test_from_c_type_enum() {
        let t = Target::X86_64;
        let en = CType::Enum {
            name: Some("color".to_string()),
            underlying: Box::new(CType::Int { signed: false }),
        };
        assert_eq!(IrType::from_c_type(&en, &t), IrType::I32);
    }

    // -- Display tests ------------------------------------------------------

    #[test]
    fn test_display_scalars() {
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
    fn test_display_array() {
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(format!("{}", arr), "[10 x i32]");
    }

    #[test]
    fn test_display_struct() {
        let st = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8],
            packed: false,
        };
        assert_eq!(format!("{}", st), "{i32, i8}");

        let packed = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: true,
        };
        assert_eq!(format!("{}", packed), "<{i8, i64}>");
    }

    #[test]
    fn test_display_function() {
        let func = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::Ptr, IrType::I64],
            is_variadic: false,
        };
        assert_eq!(format!("{}", func), "fn(ptr, i64) -> i32");

        let variadic = IrType::Function {
            return_type: Box::new(IrType::I32),
            param_types: vec![IrType::Ptr],
            is_variadic: true,
        };
        assert_eq!(format!("{}", variadic), "fn(ptr, ...) -> i32");

        let void_void = IrType::Function {
            return_type: Box::new(IrType::Void),
            param_types: vec![],
            is_variadic: false,
        };
        assert_eq!(format!("{}", void_void), "fn() -> void");
    }

    #[test]
    fn test_display_nested() {
        let nested = IrType::Array {
            element: Box::new(IrType::Struct {
                fields: vec![IrType::I32, IrType::Ptr],
                packed: false,
            }),
            count: 3,
        };
        assert_eq!(format!("{}", nested), "[3 x {i32, ptr}]");
    }

    // -- align_up helper tests ----------------------------------------------

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(10, 1), 10);
        assert_eq!(align_up(10, 0), 10);
    }

    // -- Struct trailing padding tests --------------------------------------

    #[test]
    fn test_struct_trailing_padding() {
        let t = Target::X86_64;

        // struct { i32, i8 } — raw=5, align=4, padded=8
        let st = IrType::Struct {
            fields: vec![IrType::I32, IrType::I8],
            packed: false,
        };
        assert_eq!(st.size_bytes(&t), 8);
    }

    #[test]
    fn test_struct_complex_layout() {
        let t = Target::X86_64;

        // struct { i8, i16, i32, i64 }
        // offsets: 0, 2, 4, 8; raw_end = 16; align=8; padded=16
        let st = IrType::Struct {
            fields: vec![IrType::I8, IrType::I16, IrType::I32, IrType::I64],
            packed: false,
        };
        assert_eq!(
            IrType::struct_field_offset(st.struct_fields().unwrap(), 0, false, &t),
            0
        );
        assert_eq!(
            IrType::struct_field_offset(st.struct_fields().unwrap(), 1, false, &t),
            2
        );
        assert_eq!(
            IrType::struct_field_offset(st.struct_fields().unwrap(), 2, false, &t),
            4
        );
        assert_eq!(
            IrType::struct_field_offset(st.struct_fields().unwrap(), 3, false, &t),
            8
        );
        assert_eq!(st.size_bytes(&t), 16);
    }

    // -- Clone, Eq, Hash derive tests ---------------------------------------

    #[test]
    fn test_clone_and_eq() {
        let a = IrType::Struct {
            fields: vec![IrType::I32, IrType::F64],
            packed: false,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_of(t: &IrType) -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        }

        let a = IrType::I32;
        let b = IrType::I32;
        assert_eq!(hash_of(&a), hash_of(&b));
    }
}
