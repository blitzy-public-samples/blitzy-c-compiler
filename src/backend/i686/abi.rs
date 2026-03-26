//! cdecl / System V i386 ABI implementation for the i686 backend.
//!
//! This module implements the **cdecl** calling convention as specified by the
//! System V Application Binary Interface — Intel386 Architecture Processor
//! Supplement.  It is fundamentally different from the x86-64 System V AMD64
//! ABI:
//!
//! - **ALL function arguments are passed on the stack** — there are **no**
//!   register-based arguments (unlike x86-64's RDI/RSI/RDX/RCX/R8/R9).
//! - Arguments are pushed **right-to-left** by the caller.
//! - **Integer return values** ≤ 32 bits are returned in **EAX**.
//! - **64-bit integer return values** (`long long`) use the **EDX:EAX** pair
//!   (high 32 bits in EDX, low 32 bits in EAX).
//! - **Floating-point return values** (`float`, `double`, `long double`) are
//!   returned in **ST(0)** — the top of the x87 FPU stack.
//! - **Struct/union return** uses a hidden first pointer parameter: the caller
//!   allocates a buffer, passes its address as the first (hidden) argument at
//!   `[EBP+8]`, and the callee writes the result there.
//! - **Caller cleans up the stack** after a call (`ADD ESP, N`).
//! - Stack must be **4-byte aligned**, with GCC requiring **16-byte alignment**
//!   at the CALL instruction point for ABI compatibility.
//!
//! ## ILP32 Data Model
//!
//! On i686, the ILP32 data model applies:
//! - `int`: 4 bytes
//! - `long`: **4 bytes** (not 8 as on LP64)
//! - `long long`: 8 bytes
//! - Pointer: **4 bytes** (32-bit addressing)
//! - `long double`: **12 bytes** (80-bit x87 + 2 bytes padding, 4-byte aligned)
//!
//! ## Integration with `crate::backend::traits::ArgLocation`
//!
//! This module defines its **own** [`ArgLocation`] enum with i686-specific
//! variants (`StackSlot`, `RegisterReturn`, `RegisterPairReturn`,
//! `X87Return`, `StructReturn`) that are richer than the trait-level
//! `crate::backend::traits::ArgLocation` (which only has `Register`,
//! `RegisterPair`, `Stack`).  This is intentional:
//!
//! - The i686-specific `ArgLocation` captures the full ABI detail needed for
//!   accurate codegen (hidden struct-return pointer, x87 FPU returns, etc.).
//! - The `ArchCodegen` trait methods `classify_argument()` and
//!   `classify_return()` in `i686/codegen.rs` bridge between this
//!   module's detailed classification and the trait-level enum, translating
//!   i686-specific variants into the corresponding `traits::ArgLocation`
//!   value.  This keeps the trait interface uniform across all four
//!   architectures while preserving i686-specific ABI detail internally.
//!
//! ## Exports
//!
//! | Type             | Kind   | Purpose                                      |
//! |------------------|--------|----------------------------------------------|
//! | [`I686Abi`]      | struct | ABI classification handler                   |
//! | [`ArgLocation`]  | enum   | Argument placement descriptor (always Stack)  |
//! | [`ReturnLocation`]| enum  | Return value placement descriptor             |
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::backend::i686::registers`,
//! `crate::common::types`, `crate::common::target`, `crate::ir::types`,
//! and the Rust standard library.  No external crates are used.

use crate::backend::i686::registers;
use crate::common::target::Target;
use crate::common::types::{alignof_ctype, sizeof_ctype, CType, TypeQualifiers};
use crate::ir::types::IrType;

// ===========================================================================
// ABI Constants
// ===========================================================================

/// Offset from EBP to the first function argument in the cdecl stack frame.
///
/// After the standard prologue (`push ebp; mov ebp, esp`), the stack layout
/// relative to EBP is:
///
/// ```text
/// [EBP + 8 + N]  →  Nth argument (higher addresses, right-to-left push)
/// [EBP + 12]     →  second 4-byte argument
/// [EBP + 8]      →  first argument (or hidden struct return pointer)
/// [EBP + 4]      →  return address (pushed by CALL)
/// [EBP + 0]      →  saved EBP (pushed by prologue)
/// [EBP - 4]      →  first local variable
/// ```
pub const FIRST_ARG_OFFSET: i32 = 8;

/// Minimum stack slot size for argument passing (4 bytes on i686).
///
/// Arguments smaller than 4 bytes (e.g., `char`, `short`, `_Bool`) are
/// zero-extended or sign-extended to fill a full 4-byte stack slot.
pub const MIN_STACK_SLOT_SIZE: usize = 4;

/// Stack alignment requirement at the CALL instruction boundary.
///
/// The System V i386 ABI requires 4-byte stack alignment, but modern GCC
/// on Linux enforces **16-byte alignment** at the point of a CALL instruction
/// for SSE compatibility and ABI interoperability.
pub const STACK_ALIGNMENT: usize = 16;

/// The register used for 32-bit integer/pointer return values in cdecl.
///
/// Scalar integer and pointer return values that fit in 32 bits are placed
/// in EAX by the callee before returning.
pub const INTEGER_RETURN_REG: u16 = registers::EAX;

/// The high register for 64-bit integer return values (EDX:EAX pair).
///
/// When a function returns a `long long` (64-bit integer), the low 32 bits
/// are placed in EAX and the high 32 bits are placed in EDX.
pub const INTEGER_RETURN_REG_HI: u16 = registers::EDX;

/// The frame pointer register — base for argument offset calculation.
///
/// In the standard cdecl prologue, EBP is set to the current stack pointer
/// value.  Arguments are then accessed as positive offsets from EBP:
/// `[EBP+8]` = first argument, `[EBP+12]` = second argument (for 4-byte args).
pub const FRAME_POINTER: u16 = registers::EBP;

/// The stack pointer register — used for stack space allocation and
/// argument area sizing during function calls.
pub const STACK_POINTER: u16 = registers::ESP;

/// The x87 FPU top-of-stack register for floating-point return values.
///
/// In cdecl, `float`, `double`, and `long double` return values are placed
/// in ST(0) by the callee.
pub const FLOAT_RETURN_REG: u16 = registers::ST0;

// ===========================================================================
// ArgLocation — Argument Placement Descriptor
// ===========================================================================

/// Describes where a function argument is placed in the cdecl calling
/// convention.
///
/// In the i686 cdecl ABI, **all** arguments are passed on the stack — there
/// is no register variant.  This is fundamentally different from x86-64
/// where the first 6 integer and first 8 floating-point arguments use
/// registers.
///
/// # Fields
///
/// - `offset`: Byte offset from EBP where this argument is located.
///   The first argument is at `[EBP+8]`, the second at `[EBP+8+size_of_first]`,
///   and so on.
/// - `size`: Number of bytes this argument occupies on the stack, rounded
///   up to a 4-byte boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgLocation {
    /// Argument passed on the stack at the given offset from EBP.
    Stack {
        /// Byte offset from EBP (always positive, ≥ 8 for the first argument).
        offset: i32,
        /// Size in bytes on the stack (always a multiple of 4).
        size: usize,
    },
}

// ===========================================================================
// ReturnLocation — Return Value Placement Descriptor
// ===========================================================================

/// Describes where a function's return value is placed in the cdecl calling
/// convention.
///
/// The i686 cdecl ABI uses a simpler return value scheme than x86-64:
///
/// - Scalar integers and pointers ≤ 32 bits → EAX
/// - 64-bit integers (`long long`) → EDX:EAX register pair
/// - All floating-point types → ST(0) (x87 FPU top-of-stack)
/// - Structs and unions → hidden first pointer parameter (struct return)
/// - `void` → no return value
///
/// **Design Decision — Struct Return:**
/// All struct and union types are unconditionally returned via hidden pointer
/// (StructReturn), even for small structs (e.g., 4-byte structs that could
/// theoretically fit in EAX on some System V i386 ABI implementations).
/// This is an intentional simplification for the initial implementation that
/// matches GCC's default behavior for the `-m32` target.  Optimizing small
/// struct returns to EAX is a future enhancement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnLocation {
    /// Integer return value in EAX (≤ 32 bits).
    ///
    /// Applies to: `_Bool`, `char`, `short`, `int`, `unsigned int`, `long`,
    /// `unsigned long`, pointers, and `enum` types.
    InEAX,

    /// 64-bit integer return value in the EDX:EAX register pair.
    ///
    /// Low 32 bits in EAX, high 32 bits in EDX.
    /// Applies to: `long long`, `unsigned long long`.
    InEDXEAX,

    /// Floating-point return value in ST(0) — the x87 FPU stack top.
    ///
    /// Applies to: `float`, `double`, `long double`.
    InST0,

    /// Struct or union return via a hidden first pointer parameter.
    ///
    /// The caller allocates a buffer of sufficient size and alignment,
    /// passes its address as a hidden first argument at `[EBP+8]`, and
    /// the callee writes the return value to that buffer.  All explicit
    /// arguments are shifted to higher stack offsets.
    StructReturn,

    /// Void return — no value is returned.
    Void,
}

// ===========================================================================
// Private Helpers
// ===========================================================================

/// Round `value` up to the nearest multiple of `align`.
///
/// `align` must be a power of two (or 1).  If `align` is 0, returns
/// `value` unchanged to prevent division by zero.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

/// Compute the stack slot size for an argument of the given raw byte size.
///
/// Each argument on the i686 stack occupies at least [`MIN_STACK_SLOT_SIZE`]
/// (4) bytes, and the total is rounded up to a 4-byte boundary.
#[inline]
fn arg_slot_size(raw_size: usize) -> usize {
    let aligned = align_up(raw_size, MIN_STACK_SLOT_SIZE);
    // Guarantee at least MIN_STACK_SLOT_SIZE even for zero-sized types.
    if aligned < MIN_STACK_SLOT_SIZE {
        MIN_STACK_SLOT_SIZE
    } else {
        aligned
    }
}

/// Recursively strip type qualifiers (`const`, `volatile`, `restrict`,
/// `_Atomic`), typedef indirection, and `_Atomic(T)` wrappers to reach
/// the underlying concrete type.
///
/// This is necessary for ABI classification because qualifiers and typedefs
/// do not affect argument/return value placement.
fn strip_qualifiers(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => strip_qualifiers(inner),
        CType::Atomic(inner) => strip_qualifiers(inner),
        CType::Typedef { underlying, .. } => strip_qualifiers(underlying),
        _ => ty,
    }
}

// ===========================================================================
// I686Abi — ABI Classification Handler
// ===========================================================================

/// The cdecl / System V i386 ABI handler for the i686 backend.
///
/// Provides argument classification, return value classification, stack
/// layout computation, and type size/alignment queries parameterised for
/// the ILP32 data model.
///
/// # Usage
///
/// ```ignore
/// let abi = I686Abi::new();
/// let arg_locs = abi.classify_arguments(&param_types);
/// let ret_loc  = abi.classify_return(&return_type);
/// let stack_sz = abi.compute_arg_stack_size(&param_types);
/// ```
pub struct I686Abi {
    /// The target architecture — always [`Target::I686`] for this ABI handler.
    target: Target,
}

impl Default for I686Abi {
    fn default() -> Self {
        Self::new()
    }
}

impl I686Abi {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// Create a new i686 cdecl ABI handler.
    ///
    /// Initialises the handler with [`Target::I686`], ensuring all size and
    /// alignment queries use the ILP32 data model (pointer = 4 bytes,
    /// `long` = 4 bytes, `long double` = 12 bytes).
    #[inline]
    pub fn new() -> Self {
        I686Abi {
            target: Target::I686,
        }
    }

    // -----------------------------------------------------------------------
    // Argument Classification
    // -----------------------------------------------------------------------

    /// Classify function arguments for the cdecl calling convention.
    ///
    /// In cdecl, **every** argument is passed on the stack.  Arguments are
    /// pushed right-to-left by the caller, so the first argument ends up
    /// at the lowest address (`[EBP+8]`).
    ///
    /// Each argument occupies at least 4 bytes on the stack (smaller types
    /// are zero/sign-extended).  Struct and `long long` arguments occupy
    /// their natural size rounded up to a 4-byte boundary.
    ///
    /// # Struct Return
    ///
    /// If the function returns a struct/union (detected via [`classify_return`]),
    /// the caller should prepend a pointer-to-return-buffer as the first
    /// element of `param_types` before calling this method.  The hidden
    /// pointer then occupies `[EBP+8]`, and all explicit arguments shift
    /// to higher offsets.
    ///
    /// # Arguments
    ///
    /// * `param_types` — Slice of C types for each declared parameter
    ///   (plus the hidden struct return pointer, if applicable).
    ///
    /// # Returns
    ///
    /// A `Vec<ArgLocation>` with one entry per parameter, each variant
    /// being `ArgLocation::Stack { offset, size }`.
    pub fn classify_arguments(&self, param_types: &[CType]) -> Vec<ArgLocation> {
        let mut result = Vec::with_capacity(param_types.len());
        let mut current_offset: i32 = FIRST_ARG_OFFSET;

        for param_ty in param_types {
            let ty = strip_qualifiers(param_ty);
            let raw_size = sizeof_ctype(ty, &self.target);
            let slot_size = arg_slot_size(raw_size);

            result.push(ArgLocation::Stack {
                offset: current_offset,
                size: slot_size,
            });

            current_offset += slot_size as i32;
        }

        result
    }

    // -----------------------------------------------------------------------
    // Return Value Classification
    // -----------------------------------------------------------------------

    /// Classify a function's return type for the cdecl calling convention.
    ///
    /// # Classification Rules
    ///
    /// | C Type                          | Location          |
    /// |--------------------------------|-------------------|
    /// | `void`                          | `Void`            |
    /// | `_Bool`, `char`, `short`, `int` | `InEAX`           |
    /// | `unsigned int`, `long`, pointer | `InEAX`           |
    /// | `enum`                          | `InEAX`           |
    /// | `long long`, `unsigned long long`| `InEDXEAX`       |
    /// | `float`, `double`, `long double`| `InST0`           |
    /// | `struct`, `union`               | `StructReturn`    |
    /// | `_Complex`                      | `StructReturn`    |
    /// | `_Atomic(T)`, `typedef`, qualified | recurse on inner |
    ///
    /// # Arguments
    ///
    /// * `return_type` — The C type of the function's return value.
    pub fn classify_return(&self, return_type: &CType) -> ReturnLocation {
        let ty = strip_qualifiers(return_type);

        match ty {
            // Void — no return value.
            CType::Void => ReturnLocation::Void,

            // Boolean — fits in EAX (zero-extended to 32 bits).
            CType::Bool => ReturnLocation::InEAX,

            // Character types — all fit in EAX (8-bit, zero/sign-extended).
            CType::Char | CType::SChar | CType::UChar => ReturnLocation::InEAX,

            // Short — fits in EAX (16-bit, zero/sign-extended to 32).
            CType::Short | CType::UShort => ReturnLocation::InEAX,

            // Int — exactly 32 bits, returned in EAX.
            CType::Int | CType::UInt => ReturnLocation::InEAX,

            // Long — 4 bytes on ILP32, returned in EAX.
            CType::Long | CType::ULong => ReturnLocation::InEAX,

            // Long long — 8 bytes, returned in EDX:EAX pair.
            CType::LongLong | CType::ULongLong => ReturnLocation::InEDXEAX,

            // __int128 — 16 bytes, returned via struct return on i686.
            CType::Int128 | CType::UInt128 => ReturnLocation::StructReturn,

            // Float — returned in ST(0).
            CType::Float => ReturnLocation::InST0,

            // Double — returned in ST(0).
            CType::Double => ReturnLocation::InST0,

            // Long double — 80-bit extended precision, returned in ST(0).
            CType::LongDouble => ReturnLocation::InST0,

            // Complex — returned via hidden struct return pointer on i686.
            // GCC on i386 returns all _Complex types via struct return.
            CType::Complex(_) => ReturnLocation::StructReturn,

            // Pointer — 4 bytes on ILP32, returned in EAX.
            CType::Pointer(_, _) => ReturnLocation::InEAX,

            // Array — should not normally appear as a return type (arrays
            // decay to pointers), but classify as InEAX (pointer return).
            CType::Array(_, _) => ReturnLocation::InEAX,

            // Function — should not appear as a return type (functions
            // decay to pointers), but classify as InEAX (pointer return).
            CType::Function { .. } => ReturnLocation::InEAX,

            // Struct — returned via hidden first pointer parameter.
            CType::Struct { .. } => ReturnLocation::StructReturn,

            // Union — returned via hidden first pointer parameter.
            CType::Union { .. } => ReturnLocation::StructReturn,

            // Enum — returned in EAX (underlying type is typically int).
            CType::Enum {
                underlying_type, ..
            } => self.classify_return(underlying_type),

            // Atomic, Typedef, Qualified — already stripped by
            // strip_qualifiers above, but handle defensively.
            CType::Atomic(inner) => self.classify_return(inner),
            CType::Typedef { underlying, .. } => self.classify_return(underlying),
            CType::Qualified(inner, _) => self.classify_return(inner),
        }
    }

    // -----------------------------------------------------------------------
    // Calling Convention Helpers
    // -----------------------------------------------------------------------

    /// Compute the total stack space (in bytes) required for all function
    /// arguments.
    ///
    /// Each argument's size is rounded up to a 4-byte boundary.  The total
    /// represents the number of bytes the caller must reserve on the stack
    /// before the CALL instruction, and subsequently clean up with
    /// `ADD ESP, N` after the call returns (cdecl caller-cleanup).
    ///
    /// # Arguments
    ///
    /// * `param_types` — Slice of C types for each parameter (including
    ///   the hidden struct return pointer, if applicable).
    pub fn compute_arg_stack_size(&self, param_types: &[CType]) -> usize {
        let mut total: usize = 0;

        for param_ty in param_types {
            let ty = strip_qualifiers(param_ty);
            let raw_size = sizeof_ctype(ty, &self.target);
            total += arg_slot_size(raw_size);
        }

        total
    }

    /// Returns the EBP-relative offset of the hidden struct return pointer.
    ///
    /// When a function returns a struct or union, the caller inserts a
    /// hidden pointer as the **first** stack argument at `[EBP+8]`.  All
    /// explicit arguments are shifted to higher offsets (`[EBP+12]`,
    /// `[EBP+16]`, etc.).
    ///
    /// # Returns
    ///
    /// `8` — the constant offset `[EBP+8]` where the hidden struct return
    /// pointer is located.
    #[inline]
    pub fn get_struct_return_pointer_offset(&self) -> i32 {
        FIRST_ARG_OFFSET
    }

    /// Returns the stack alignment requirement at the CALL instruction
    /// boundary.
    ///
    /// The basic System V i386 ABI mandates 4-byte stack alignment, but
    /// modern GCC on Linux enforces **16-byte alignment** at the point of
    /// a CALL instruction for SSE compatibility and ABI interoperability.
    ///
    /// # Returns
    ///
    /// `16` — the alignment requirement in bytes.
    #[inline]
    pub fn stack_alignment_requirement(&self) -> usize {
        STACK_ALIGNMENT
    }

    /// Returns `true` because cdecl uses **caller-cleanup** semantics.
    ///
    /// After a CALL instruction, the **caller** is responsible for removing
    /// the arguments from the stack, typically with `ADD ESP, N` where N
    /// is the total argument stack size.  This is in contrast to `stdcall`
    /// where the callee cleans up via `RET N`.
    #[inline]
    pub fn caller_cleanup(&self) -> bool {
        true
    }

    // -----------------------------------------------------------------------
    // Variadic Function Support
    // -----------------------------------------------------------------------

    /// Classify arguments for a variadic function call.
    ///
    /// In the cdecl ABI, variadic arguments are treated identically to
    /// regular arguments — **all** are placed on the stack.  The only
    /// difference is that the C **default argument promotions** apply to
    /// the variadic (non-fixed) arguments:
    ///
    /// - Integer types narrower than `int` are promoted to `int`.
    /// - `float` is promoted to `double`.
    ///
    /// This is much simpler than x86-64, where variadic calls require the
    /// AL register to hold the count of SSE registers used.
    ///
    /// # Arguments
    ///
    /// * `fixed_count` — Number of fixed (non-variadic) parameters.
    /// * `all_types` — Types of all arguments (fixed + variadic).
    ///
    /// # Returns
    ///
    /// A `Vec<ArgLocation>` with one entry per argument.
    pub fn classify_variadic_args(
        &self,
        fixed_count: usize,
        all_types: &[CType],
    ) -> Vec<ArgLocation> {
        let mut result = Vec::with_capacity(all_types.len());
        let mut current_offset: i32 = FIRST_ARG_OFFSET;

        for (i, param_ty) in all_types.iter().enumerate() {
            let ty = strip_qualifiers(param_ty);

            // For variadic arguments (beyond the fixed parameters), apply
            // the C default argument promotions.
            let effective_ty = if i >= fixed_count {
                self.promote_argument_type(ty)
            } else {
                ty.clone()
            };

            let raw_size = sizeof_ctype(&effective_ty, &self.target);
            let slot_size = arg_slot_size(raw_size);

            result.push(ArgLocation::Stack {
                offset: current_offset,
                size: slot_size,
            });

            current_offset += slot_size as i32;
        }

        result
    }

    // -----------------------------------------------------------------------
    // Type Size and Alignment Helpers (i686-specific)
    // -----------------------------------------------------------------------

    /// Returns the size in bytes of a C type under the i686 ILP32 data model.
    ///
    /// Delegates to [`sizeof_ctype`] with [`Target::I686`].
    ///
    /// # Key i686 Size Differences from x86-64
    ///
    /// | Type          | i686 (ILP32) | x86-64 (LP64) |
    /// |---------------|-------------|---------------|
    /// | Pointer       | 4 bytes     | 8 bytes       |
    /// | `long`        | 4 bytes     | 8 bytes       |
    /// | `long double` | 12 bytes    | 16 bytes      |
    #[inline]
    pub fn type_size(&self, ty: &CType) -> usize {
        sizeof_ctype(ty, &self.target)
    }

    /// Returns the alignment in bytes of a C type under the i686 ILP32 ABI.
    ///
    /// Delegates to [`alignof_ctype`] with [`Target::I686`].
    ///
    /// # Key i686 Alignment Differences from x86-64
    ///
    /// | Type          | i686 Align | x86-64 Align |
    /// |---------------|-----------|-------------|
    /// | `double`      | 4 bytes   | 8 bytes     |
    /// | `long long`   | 4 bytes   | 8 bytes     |
    /// | `long double` | 4 bytes   | 16 bytes    |
    #[inline]
    pub fn type_alignment(&self, ty: &CType) -> usize {
        alignof_ctype(ty, &self.target)
    }

    // -----------------------------------------------------------------------
    // Stack Argument Promotion Rules
    // -----------------------------------------------------------------------

    /// Apply the C **default argument promotions** to a type.
    ///
    /// The default argument promotions (C11 §6.5.2.2 ¶6) are applied to:
    /// - Variadic function arguments (beyond the last fixed parameter).
    /// - Arguments in calls to unprototyped (K&R-style) functions.
    ///
    /// # Promotion Rules
    ///
    /// | Input Type                         | Promoted Type            |
    /// |-----------------------------------|--------------------------|
    /// | `_Bool`, `char`, `signed char`     | `int`                    |
    /// | `unsigned char`, `short`           | `int`                    |
    /// | `unsigned short`                   | `int`                    |
    /// | `float`                            | `double`                 |
    /// | `T[]` (array)                      | `T*` (pointer to T)      |
    /// | `T(...)` (function)                | `T(*)(...)` (func ptr)   |
    /// | All other types                    | unchanged                |
    pub fn promote_argument_type(&self, ty: &CType) -> CType {
        let stripped = strip_qualifiers(ty);

        match stripped {
            // Integer types narrower than `int` are promoted to `int`.
            CType::Bool
            | CType::Char
            | CType::SChar
            | CType::UChar
            | CType::Short
            | CType::UShort => CType::Int,

            // `float` is promoted to `double` (C default arg promotions).
            CType::Float => CType::Double,

            // Array decays to pointer-to-element.
            CType::Array(elem, _) => CType::Pointer(elem.clone(), TypeQualifiers::default()),

            // Function type decays to pointer-to-function.
            CType::Function { .. } => {
                CType::Pointer(Box::new(stripped.clone()), TypeQualifiers::default())
            }

            // Enum: promote the underlying type if it is narrower than int.
            CType::Enum {
                underlying_type, ..
            } => self.promote_argument_type(underlying_type),

            // Atomic, Typedef, Qualified: already stripped above; handle
            // defensively by recursing.
            CType::Atomic(inner) => self.promote_argument_type(inner),
            CType::Typedef { underlying, .. } => self.promote_argument_type(underlying),
            CType::Qualified(inner, _) => self.promote_argument_type(inner),

            // All other types (Int, UInt, Long, ULong, LongLong, ULongLong,
            // Double, LongDouble, Complex, Pointer, Struct, Union) are
            // returned unchanged — they are already at least `int`-width
            // or are aggregate/composite types not subject to promotion.
            _ => stripped.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // IR Type Bridging Utilities
    // -----------------------------------------------------------------------

    /// Classify an IR-level type for return value placement.
    ///
    /// This bridges between the code generation layer's [`IrType`] and the
    /// ABI's [`ReturnLocation`], enabling the code generator to query ABI
    /// classification directly from IR types without first converting back
    /// to [`CType`].
    ///
    /// # Arguments
    ///
    /// * `ir_ty` — The IR type to classify.
    ///
    /// # Returns
    ///
    /// The corresponding [`ReturnLocation`] for this IR type.
    pub fn classify_ir_return(&self, ir_ty: &IrType) -> ReturnLocation {
        // Void → no return value.
        if ir_ty.is_void() {
            return ReturnLocation::Void;
        }

        // Integer or pointer types → EAX (≤32 bits) or EDX:EAX (64 bits).
        if ir_ty.is_integer() || ir_ty.is_pointer() {
            let size = ir_ty.size_bytes(&self.target);
            return if size <= 4 {
                ReturnLocation::InEAX
            } else if size <= 8 {
                ReturnLocation::InEDXEAX
            } else {
                // Integers wider than 64 bits (e.g., I128) use struct return.
                ReturnLocation::StructReturn
            };
        }

        // Floating-point types → ST(0).
        if ir_ty.is_float() {
            return ReturnLocation::InST0;
        }

        // Struct types → always via hidden struct return pointer on i686.
        if ir_ty.is_struct() {
            return ReturnLocation::StructReturn;
        }

        // Arrays and other aggregate types → struct return.
        ReturnLocation::StructReturn
    }

    /// Compute the stack slot size for a value of the given IR type.
    ///
    /// Returns the number of bytes this value would occupy on the argument
    /// stack, rounded up to a 4-byte boundary with a minimum of 4 bytes.
    ///
    /// # Arguments
    ///
    /// * `ir_ty` — The IR type whose stack slot size to compute.
    pub fn ir_type_stack_size(&self, ir_ty: &IrType) -> usize {
        let raw_size = ir_ty.size_bytes(&self.target);
        arg_slot_size(raw_size)
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::TypeQualifiers;

    #[test]
    fn test_new_creates_i686_target() {
        let abi = I686Abi::new();
        assert_eq!(abi.target, Target::I686);
    }

    #[test]
    fn test_classify_return_void() {
        let abi = I686Abi::new();
        assert_eq!(abi.classify_return(&CType::Void), ReturnLocation::Void);
    }

    #[test]
    fn test_classify_return_integer_types_in_eax() {
        let abi = I686Abi::new();
        let eax_types = [
            CType::Bool,
            CType::Char,
            CType::SChar,
            CType::UChar,
            CType::Short,
            CType::UShort,
            CType::Int,
            CType::UInt,
            CType::Long,
            CType::ULong,
        ];
        for ty in &eax_types {
            assert_eq!(
                abi.classify_return(ty),
                ReturnLocation::InEAX,
                "Expected InEAX for {:?}",
                ty
            );
        }
    }

    #[test]
    fn test_classify_return_pointer_in_eax() {
        let abi = I686Abi::new();
        let ptr_ty = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        assert_eq!(abi.classify_return(&ptr_ty), ReturnLocation::InEAX);
    }

    #[test]
    fn test_classify_return_longlong_in_edx_eax() {
        let abi = I686Abi::new();
        assert_eq!(
            abi.classify_return(&CType::LongLong),
            ReturnLocation::InEDXEAX
        );
        assert_eq!(
            abi.classify_return(&CType::ULongLong),
            ReturnLocation::InEDXEAX
        );
    }

    #[test]
    fn test_classify_return_float_in_st0() {
        let abi = I686Abi::new();
        assert_eq!(abi.classify_return(&CType::Float), ReturnLocation::InST0);
        assert_eq!(abi.classify_return(&CType::Double), ReturnLocation::InST0);
        assert_eq!(
            abi.classify_return(&CType::LongDouble),
            ReturnLocation::InST0
        );
    }

    #[test]
    fn test_classify_return_struct_return() {
        let abi = I686Abi::new();
        let struct_ty = CType::Struct {
            name: Some("point".to_string()),
            fields: vec![],
            packed: false,
            aligned: None,
        };
        assert_eq!(
            abi.classify_return(&struct_ty),
            ReturnLocation::StructReturn
        );

        let union_ty = CType::Union {
            name: None,
            fields: vec![],
            packed: false,
            aligned: None,
        };
        assert_eq!(abi.classify_return(&union_ty), ReturnLocation::StructReturn);
    }

    #[test]
    fn test_classify_return_complex_struct_return() {
        let abi = I686Abi::new();
        let complex_float = CType::Complex(Box::new(CType::Float));
        assert_eq!(
            abi.classify_return(&complex_float),
            ReturnLocation::StructReturn
        );
    }

    #[test]
    fn test_classify_arguments_all_on_stack() {
        let abi = I686Abi::new();
        let params = vec![CType::Int, CType::Int, CType::Int];
        let locs = abi.classify_arguments(&params);

        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0], ArgLocation::Stack { offset: 8, size: 4 });
        assert_eq!(
            locs[1],
            ArgLocation::Stack {
                offset: 12,
                size: 4
            }
        );
        assert_eq!(
            locs[2],
            ArgLocation::Stack {
                offset: 16,
                size: 4
            }
        );
    }

    #[test]
    fn test_classify_arguments_mixed_sizes() {
        let abi = I686Abi::new();
        let params = vec![CType::Char, CType::LongLong, CType::Short];
        let locs = abi.classify_arguments(&params);

        assert_eq!(locs.len(), 3);
        // char → 1 byte → padded to 4 on stack
        assert_eq!(locs[0], ArgLocation::Stack { offset: 8, size: 4 });
        // long long → 8 bytes
        assert_eq!(
            locs[1],
            ArgLocation::Stack {
                offset: 12,
                size: 8
            }
        );
        // short → 2 bytes → padded to 4
        assert_eq!(
            locs[2],
            ArgLocation::Stack {
                offset: 20,
                size: 4
            }
        );
    }

    #[test]
    fn test_classify_arguments_pointer_is_4_bytes() {
        let abi = I686Abi::new();
        let ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        let locs = abi.classify_arguments(&[ptr]);

        assert_eq!(locs.len(), 1);
        // Pointer = 4 bytes on i686
        assert_eq!(locs[0], ArgLocation::Stack { offset: 8, size: 4 });
    }

    #[test]
    fn test_compute_arg_stack_size() {
        let abi = I686Abi::new();
        let params = vec![CType::Int, CType::LongLong, CType::Char];
        // int=4 + long_long=8 + char=4(padded) = 16
        assert_eq!(abi.compute_arg_stack_size(&params), 16);
    }

    #[test]
    fn test_struct_return_pointer_offset() {
        let abi = I686Abi::new();
        assert_eq!(abi.get_struct_return_pointer_offset(), 8);
    }

    #[test]
    fn test_stack_alignment() {
        let abi = I686Abi::new();
        assert_eq!(abi.stack_alignment_requirement(), 16);
    }

    #[test]
    fn test_caller_cleanup() {
        let abi = I686Abi::new();
        assert!(abi.caller_cleanup());
    }

    #[test]
    fn test_type_size_i686() {
        let abi = I686Abi::new();
        // Pointer: 4 bytes on ILP32
        let ptr = CType::Pointer(Box::new(CType::Int), TypeQualifiers::default());
        assert_eq!(abi.type_size(&ptr), 4);
        // Long: 4 bytes on ILP32
        assert_eq!(abi.type_size(&CType::Long), 4);
        // Long double: 12 bytes on i686
        assert_eq!(abi.type_size(&CType::LongDouble), 12);
        // Int: 4 bytes
        assert_eq!(abi.type_size(&CType::Int), 4);
        // Long long: 8 bytes
        assert_eq!(abi.type_size(&CType::LongLong), 8);
    }

    #[test]
    fn test_type_alignment_i686() {
        let abi = I686Abi::new();
        // Double: 4-byte aligned on i686
        assert_eq!(abi.type_alignment(&CType::Double), 4);
        // Long long: 4-byte aligned on i686
        assert_eq!(abi.type_alignment(&CType::LongLong), 4);
        // Long double: 4-byte aligned on i686
        assert_eq!(abi.type_alignment(&CType::LongDouble), 4);
    }

    #[test]
    fn test_promote_argument_type() {
        let abi = I686Abi::new();
        // Narrow integers → int
        assert_eq!(abi.promote_argument_type(&CType::Bool), CType::Int);
        assert_eq!(abi.promote_argument_type(&CType::Char), CType::Int);
        assert_eq!(abi.promote_argument_type(&CType::SChar), CType::Int);
        assert_eq!(abi.promote_argument_type(&CType::UChar), CType::Int);
        assert_eq!(abi.promote_argument_type(&CType::Short), CType::Int);
        assert_eq!(abi.promote_argument_type(&CType::UShort), CType::Int);
        // Float → double
        assert_eq!(abi.promote_argument_type(&CType::Float), CType::Double);
        // Int → unchanged
        assert_eq!(abi.promote_argument_type(&CType::Int), CType::Int);
        // Double → unchanged
        assert_eq!(abi.promote_argument_type(&CType::Double), CType::Double);
    }

    #[test]
    fn test_promote_array_to_pointer() {
        let abi = I686Abi::new();
        let arr = CType::Array(Box::new(CType::Int), Some(10));
        let promoted = abi.promote_argument_type(&arr);
        match promoted {
            CType::Pointer(inner, quals) => {
                assert_eq!(*inner, CType::Int);
                assert!(quals.is_empty());
            }
            _ => panic!("Expected Pointer, got {:?}", promoted),
        }
    }

    #[test]
    fn test_promote_function_to_function_pointer() {
        let abi = I686Abi::new();
        let func = CType::Function {
            return_type: Box::new(CType::Int),
            params: vec![CType::Int],
            variadic: false,
        };
        let promoted = abi.promote_argument_type(&func);
        match promoted {
            CType::Pointer(_, _) => {} // OK — function decayed to pointer
            _ => panic!("Expected Pointer, got {:?}", promoted),
        }
    }

    #[test]
    fn test_classify_variadic_args() {
        let abi = I686Abi::new();
        // printf("fmt", char_val, float_val)
        // Fixed: 1 (the format string pointer)
        // Variadic: char → promoted to int, float → promoted to double
        let ptr = CType::Pointer(Box::new(CType::Char), TypeQualifiers::default());
        let all_types = vec![ptr, CType::Char, CType::Float];
        let locs = abi.classify_variadic_args(1, &all_types);

        assert_eq!(locs.len(), 3);
        // First (fixed): pointer → 4 bytes
        assert_eq!(locs[0], ArgLocation::Stack { offset: 8, size: 4 });
        // Second (variadic): char promoted to int → 4 bytes
        assert_eq!(
            locs[1],
            ArgLocation::Stack {
                offset: 12,
                size: 4
            }
        );
        // Third (variadic): float promoted to double → 8 bytes
        assert_eq!(
            locs[2],
            ArgLocation::Stack {
                offset: 16,
                size: 8
            }
        );
    }

    #[test]
    fn test_classify_ir_return() {
        let abi = I686Abi::new();
        assert_eq!(abi.classify_ir_return(&IrType::Void), ReturnLocation::Void);
        assert_eq!(abi.classify_ir_return(&IrType::I32), ReturnLocation::InEAX);
        assert_eq!(abi.classify_ir_return(&IrType::I8), ReturnLocation::InEAX);
        assert_eq!(
            abi.classify_ir_return(&IrType::I64),
            ReturnLocation::InEDXEAX
        );
        assert_eq!(abi.classify_ir_return(&IrType::F32), ReturnLocation::InST0);
        assert_eq!(abi.classify_ir_return(&IrType::Ptr), ReturnLocation::InEAX);
    }

    #[test]
    fn test_ir_type_stack_size() {
        let abi = I686Abi::new();
        // I8 → 1 byte → padded to 4
        assert_eq!(abi.ir_type_stack_size(&IrType::I8), 4);
        // I32 → 4 bytes
        assert_eq!(abi.ir_type_stack_size(&IrType::I32), 4);
        // I64 → 8 bytes
        assert_eq!(abi.ir_type_stack_size(&IrType::I64), 8);
        // F80 → 12 bytes on i686
        assert_eq!(abi.ir_type_stack_size(&IrType::F80), 12);
        // Ptr → 4 bytes on i686
        assert_eq!(abi.ir_type_stack_size(&IrType::Ptr), 4);
    }

    #[test]
    fn test_register_constants() {
        // Verify the ABI constants reference the correct register values.
        assert_eq!(INTEGER_RETURN_REG, registers::EAX);
        assert_eq!(INTEGER_RETURN_REG_HI, registers::EDX);
        assert_eq!(FRAME_POINTER, registers::EBP);
        assert_eq!(STACK_POINTER, registers::ESP);
        assert_eq!(FLOAT_RETURN_REG, registers::ST0);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4), 0);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(9, 4), 12);
        assert_eq!(align_up(7, 16), 16);
    }

    #[test]
    fn test_classify_return_through_typedef() {
        let abi = I686Abi::new();
        let typedef_int = CType::Typedef {
            name: "myint".to_string(),
            underlying: Box::new(CType::Int),
        };
        assert_eq!(abi.classify_return(&typedef_int), ReturnLocation::InEAX);
    }

    #[test]
    fn test_classify_return_through_qualified() {
        let abi = I686Abi::new();
        let const_int = CType::Qualified(
            Box::new(CType::Int),
            TypeQualifiers {
                is_const: true,
                is_volatile: false,
                is_restrict: false,
                is_atomic: false,
            },
        );
        assert_eq!(abi.classify_return(&const_int), ReturnLocation::InEAX);
    }

    #[test]
    fn test_classify_return_enum() {
        let abi = I686Abi::new();
        let enum_ty = CType::Enum {
            name: Some("color".to_string()),
            underlying_type: Box::new(CType::Int),
        };
        assert_eq!(abi.classify_return(&enum_ty), ReturnLocation::InEAX);
    }

    #[test]
    fn test_empty_arguments() {
        let abi = I686Abi::new();
        let locs = abi.classify_arguments(&[]);
        assert!(locs.is_empty());
        assert_eq!(abi.compute_arg_stack_size(&[]), 0);
    }

    #[test]
    fn test_long_double_arg_size() {
        let abi = I686Abi::new();
        let locs = abi.classify_arguments(&[CType::LongDouble]);
        assert_eq!(locs.len(), 1);
        // long double = 12 bytes on i686, already 4-aligned
        assert_eq!(
            locs[0],
            ArgLocation::Stack {
                offset: 8,
                size: 12
            }
        );
    }

    #[test]
    fn test_ir_struct_return() {
        let abi = I686Abi::new();
        let st = IrType::Struct(crate::ir::types::StructType::new(
            vec![IrType::I32, IrType::I32],
            false,
        ));
        assert_eq!(abi.classify_ir_return(&st), ReturnLocation::StructReturn);
        assert!(st.is_struct());
    }
}
