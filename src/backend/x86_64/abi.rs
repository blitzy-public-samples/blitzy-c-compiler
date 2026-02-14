//! System V AMD64 ABI — parameter classification, call/return conventions,
//! frame layout, and red zone support for the x86-64 target.
//!
//! This module implements the type classification algorithm from the
//! System V AMD64 ABI specification (§3.2.3). Every C type is decomposed
//! into 8-byte "eightbytes" and each eightbyte is assigned a class:
//!
//! | Class     | Meaning                                      |
//! |-----------|----------------------------------------------|
//! | INTEGER   | Passed in a general-purpose register (GPR)   |
//! | SSE       | Passed in an XMM register                    |
//! | MEMORY    | Passed on the stack                          |
//! | X87       | Passed on the x87 FPU stack                  |
//! | X87UP     | Upper portion of an x87 long double value    |
//! | NO_CLASS  | Padding or void (ignored for passing)        |
//!
//! # Register Assignment Order
//!
//! - **Integer arguments:** RDI, RSI, RDX, RCX, R8, R9 (6 registers)
//! - **Float arguments:** XMM0–XMM7 (8 registers)
//! - **Return values:** RAX (integer), XMM0 (float), RAX:RDX (128-bit int),
//!   XMM0:XMM1 (complex float)
//!
//! # Red Zone
//!
//! Leaf functions may use the 128-byte area below RSP without adjusting
//! the stack pointer (System V AMD64 ABI §3.2.2). This optimisation is
//! only valid when the function contains no calls and the frame fits
//! within 128 bytes.

use crate::backend::traits::{ParamClass, PhysReg};
use crate::backend::x86_64::registers;
use crate::common::target::Target;
use crate::common::types::CType;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum aggregate size (in bytes) that the System V AMD64 ABI passes
/// in registers. Structs larger than 16 bytes are always passed in MEMORY.
const MAX_REGISTER_AGGREGATE_SIZE: usize = 16;

/// Number of bytes in one "eightbyte" — the fundamental classification
/// unit in the System V AMD64 ABI.
const EIGHTBYTE_SIZE: usize = 8;

/// Size of the red zone in bytes below RSP that leaf functions may use
/// without adjusting the stack pointer.
#[allow(dead_code)]
const RED_ZONE_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// ParamLocation — describes where a parameter or return value is placed
// ---------------------------------------------------------------------------

/// Describes the physical location(s) assigned to a single function
/// parameter or return value after ABI classification and register
/// assignment.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamLocation {
    /// Value is passed in a single general-purpose register.
    IntReg(PhysReg),

    /// Value is passed in a single SSE register.
    SseReg(PhysReg),

    /// Value is split across two general-purpose registers (e.g. 128-bit
    /// integers or structs with two INTEGER eightbytes).
    IntRegPair(PhysReg, PhysReg),

    /// Value is split across two SSE registers (e.g. complex float).
    SseRegPair(PhysReg, PhysReg),

    /// Value is split across an integer register and an SSE register
    /// (e.g. a struct with one INTEGER and one SSE eightbyte).
    MixedRegPair {
        /// Register for the first eightbyte.
        first: PhysReg,
        /// Register for the second eightbyte.
        second: PhysReg,
        /// `true` if the first eightbyte is INTEGER and the second is SSE.
        first_is_int: bool,
    },

    /// Value is passed on the stack at the given byte offset from the
    /// caller's stack pointer (RSP at the point of the CALL instruction,
    /// before the return address push).
    Stack {
        /// Byte offset from RSP at the call site.
        offset: u32,
        /// Size of the value on the stack in bytes.
        size: u32,
    },

    /// Value is passed via the x87 FPU stack (long double).
    X87Stack,
}

// ---------------------------------------------------------------------------
// classify_type — core ABI classification
// ---------------------------------------------------------------------------

/// Classifies a C type into per-eightbyte [`ParamClass`] assignments
/// according to the System V AMD64 ABI specification (§3.2.3).
///
/// # Arguments
///
/// * `ty` — the C language type to classify
/// * `target` — the target architecture (must be [`Target::X86_64`])
///
/// # Returns
///
/// A `Vec<ParamClass>` with one entry per eightbyte. Scalar types
/// typically produce a single-element vector; structs ≤ 16 bytes may
/// produce one or two entries; large structs produce a single MEMORY
/// entry.
///
/// # Examples
///
/// ```ignore
/// let classes = classify_type(&CType::Int { signed: true }, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::Integer]);
///
/// let classes = classify_type(&CType::Double, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::SSE]);
///
/// let classes = classify_type(&CType::LongDouble, &Target::X86_64);
/// assert_eq!(classes, vec![ParamClass::X87, ParamClass::X87Up]);
/// ```
pub fn classify_type(ty: &CType, target: &Target) -> Vec<ParamClass> {
    match ty {
        // -- Void -------------------------------------------------------
        CType::Void => vec![ParamClass::NoClass],

        // -- Boolean ----------------------------------------------------
        CType::Bool => vec![ParamClass::Integer],

        // -- Integer types — all fit in GPRs ----------------------------
        CType::Char { .. }
        | CType::Short { .. }
        | CType::Int { .. }
        | CType::Long { .. }
        | CType::LongLong { .. } => vec![ParamClass::Integer],

        // -- Enum — underlying type is integer --------------------------
        CType::Enum { .. } => vec![ParamClass::Integer],

        // -- Float / Double — SSE registers -----------------------------
        CType::Float | CType::Double => vec![ParamClass::SSE],

        // -- Long double — x87 extended precision on x86-64 -------------
        CType::LongDouble => vec![ParamClass::X87, ParamClass::X87Up],

        // -- Complex types — two SSE registers for float/double,
        //    MEMORY for complex long double.
        CType::Complex(base) => match base.as_ref() {
            CType::Float | CType::Double => vec![ParamClass::SSE, ParamClass::SSE],
            CType::LongDouble => vec![ParamClass::Memory],
            _ => vec![ParamClass::SSE, ParamClass::SSE],
        },

        // -- Pointer — always INTEGER (8 bytes on x86-64) ---------------
        CType::Pointer(_) => vec![ParamClass::Integer],

        // -- Function type — treated as pointer to function -------------
        CType::Function { .. } => vec![ParamClass::Integer],

        // -- Array — treat as aggregate for ABI purposes ----------------
        CType::Array { element, size } => {
            let total_size = size
                .map(|n| n * type_size_bytes(element, target))
                .unwrap_or(0);
            if total_size > MAX_REGISTER_AGGREGATE_SIZE {
                vec![ParamClass::Memory]
            } else {
                classify_aggregate_fields_fallback(total_size)
            }
        }

        // -- Struct — full eightbyte decomposition ----------------------
        CType::Struct { fields, .. } => {
            classify_struct(fields, target)
        }

        // -- Union — classified based on the largest member -------------
        CType::Union { fields, .. } => {
            classify_union(fields, target)
        }

        // -- Atomic — classify the underlying type ----------------------
        CType::Atomic(inner) => classify_type(inner, target),

        // -- Typedef — classify the underlying type ---------------------
        CType::Typedef { underlying, .. } => classify_type(underlying, target),
    }
}

/// Computes the size of a [`CType`] in bytes for the given target.
///
/// This is a simplified helper for ABI classification. It uses
/// target-dependent sizes for pointer-width and `long` types.
fn type_size_bytes(ty: &CType, target: &Target) -> usize {
    match ty {
        CType::Void => 0,
        CType::Bool => 1,
        CType::Char { .. } => 1,
        CType::Short { .. } => 2,
        CType::Int { .. } => 4,
        CType::Long { .. } => {
            if *target == Target::I686 { 4 } else { 8 }
        }
        CType::LongLong { .. } => 8,
        CType::Float => 4,
        CType::Double => 8,
        CType::LongDouble => {
            match target {
                Target::X86_64 => 16,
                Target::I686 => 12,
                _ => 8,
            }
        }
        CType::Complex(base) => 2 * type_size_bytes(base, target),
        CType::Pointer(_) | CType::Function { .. } => {
            if *target == Target::I686 { 4 } else { 8 }
        }
        CType::Array { element, size } => {
            size.unwrap_or(0) * type_size_bytes(element, target)
        }
        CType::Struct { fields, .. } => {
            compute_struct_size(fields, target)
        }
        CType::Union { fields, .. } => {
            fields.iter()
                .map(|f| type_size_bytes(&f.ty, target))
                .max()
                .unwrap_or(0)
        }
        CType::Enum { .. } => 4, // enums default to int size
        CType::Atomic(inner) => type_size_bytes(inner, target),
        CType::Typedef { underlying, .. } => type_size_bytes(underlying, target),
    }
}

/// Computes the alignment of a [`CType`] in bytes for the given target.
fn type_align_bytes(ty: &CType, target: &Target) -> usize {
    match ty {
        CType::Void => 1,
        CType::Bool => 1,
        CType::Char { .. } => 1,
        CType::Short { .. } => 2,
        CType::Int { .. } => 4,
        CType::Long { .. } => {
            if *target == Target::I686 { 4 } else { 8 }
        }
        CType::LongLong { .. } => 8,
        CType::Float => 4,
        CType::Double => 8,
        CType::LongDouble => {
            match target {
                Target::X86_64 => 16,
                Target::I686 => 4,
                _ => 8,
            }
        }
        CType::Complex(base) => type_align_bytes(base, target),
        CType::Pointer(_) | CType::Function { .. } => {
            if *target == Target::I686 { 4 } else { 8 }
        }
        CType::Array { element, .. } => type_align_bytes(element, target),
        CType::Struct { fields, .. } => {
            fields.iter()
                .map(|f| type_align_bytes(&f.ty, target))
                .max()
                .unwrap_or(1)
        }
        CType::Union { fields, .. } => {
            fields.iter()
                .map(|f| type_align_bytes(&f.ty, target))
                .max()
                .unwrap_or(1)
        }
        CType::Enum { .. } => 4,
        CType::Atomic(inner) => type_align_bytes(inner, target),
        CType::Typedef { underlying, .. } => type_align_bytes(underlying, target),
    }
}

/// Computes the total size of a struct including alignment padding.
fn compute_struct_size(
    fields: &[crate::common::types::FieldDef],
    target: &Target,
) -> usize {
    let mut offset: usize = 0;
    let mut max_align: usize = 1;
    for field in fields {
        let align = type_align_bytes(&field.ty, target);
        if align > max_align {
            max_align = align;
        }
        // Align the offset to the field's alignment requirement
        offset = (offset + align - 1) & !(align - 1);
        offset += type_size_bytes(&field.ty, target);
    }
    // Pad to struct alignment
    if max_align > 0 {
        offset = (offset + max_align - 1) & !(max_align - 1);
    }
    offset
}

/// Classifies a struct type field-by-field using the eightbyte decomposition
/// algorithm from the System V AMD64 ABI §3.2.3.
fn classify_struct(
    fields: &[crate::common::types::FieldDef],
    target: &Target,
) -> Vec<ParamClass> {
    let total_size = compute_struct_size(fields, target);

    // Rule: structs > 16 bytes are always passed in MEMORY.
    if total_size > MAX_REGISTER_AGGREGATE_SIZE {
        return vec![ParamClass::Memory];
    }

    // Empty structs get NoClass.
    if total_size == 0 {
        return vec![ParamClass::NoClass];
    }

    // Number of eightbytes needed.
    let num_eightbytes = (total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    let mut classes = vec![ParamClass::NoClass; num_eightbytes];

    // Walk fields and classify each eightbyte they overlap with.
    let mut offset: usize = 0;
    for field in fields {
        let align = type_align_bytes(&field.ty, target);
        offset = (offset + align - 1) & !(align - 1);

        let field_size = type_size_bytes(&field.ty, target);
        let field_class = scalar_class(&field.ty);

        // Determine which eightbyte(s) this field overlaps.
        let start_eb = offset / EIGHTBYTE_SIZE;
        let end_eb = if field_size > 0 {
            (offset + field_size - 1) / EIGHTBYTE_SIZE
        } else {
            start_eb
        };

        for (idx, class) in classes.iter_mut().enumerate().take(num_eightbytes).skip(start_eb) {
            if idx <= end_eb {
                *class = class.merge(field_class);
            }
        }

        offset += field_size;
    }

    // Post-merge rules from the ABI:
    // If any eightbyte is MEMORY, the entire struct is MEMORY.
    if classes.contains(&ParamClass::Memory) {
        return vec![ParamClass::Memory];
    }

    // If the first eightbyte is X87 and the second isn't X87UP (or vice versa),
    // the whole struct is MEMORY.
    if num_eightbytes == 2
        && classes[0] == ParamClass::X87
        && classes[1] != ParamClass::X87Up
    {
        return vec![ParamClass::Memory];
    }

    classes
}

/// Classifies a union type — the classification is the merge of all
/// member classifications across the eightbytes they occupy.
fn classify_union(
    fields: &[crate::common::types::FieldDef],
    target: &Target,
) -> Vec<ParamClass> {
    let total_size = fields.iter()
        .map(|f| type_size_bytes(&f.ty, target))
        .max()
        .unwrap_or(0);

    // Rule: unions > 16 bytes are always passed in MEMORY.
    if total_size > MAX_REGISTER_AGGREGATE_SIZE {
        return vec![ParamClass::Memory];
    }

    if total_size == 0 {
        return vec![ParamClass::NoClass];
    }

    let num_eightbytes = (total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    let mut classes = vec![ParamClass::NoClass; num_eightbytes];

    for field in fields {
        let field_classes = classify_type(&field.ty, target);
        for (i, &fc) in field_classes.iter().enumerate() {
            if i < num_eightbytes {
                classes[i] = classes[i].merge(fc);
            }
        }
    }

    // If any eightbyte is MEMORY, the entire union is MEMORY.
    if classes.contains(&ParamClass::Memory) {
        return vec![ParamClass::Memory];
    }

    classes
}

/// Returns the base class for a scalar type (non-aggregate).
fn scalar_class(ty: &CType) -> ParamClass {
    match ty {
        CType::Void => ParamClass::NoClass,
        CType::Bool
        | CType::Char { .. }
        | CType::Short { .. }
        | CType::Int { .. }
        | CType::Long { .. }
        | CType::LongLong { .. }
        | CType::Enum { .. }
        | CType::Pointer(_)
        | CType::Function { .. } => ParamClass::Integer,
        CType::Float | CType::Double => ParamClass::SSE,
        CType::LongDouble => ParamClass::X87,
        CType::Complex(base) => {
            match base.as_ref() {
                CType::Float | CType::Double => ParamClass::SSE,
                _ => ParamClass::Memory,
            }
        }
        CType::Array { .. }
        | CType::Struct { .. }
        | CType::Union { .. } => ParamClass::Memory,
        CType::Atomic(inner) => scalar_class(inner),
        CType::Typedef { underlying, .. } => scalar_class(underlying),
    }
}

/// Fallback classification for aggregate types based on total size,
/// when detailed field information is not available.
fn classify_aggregate_fields_fallback(total_size: usize) -> Vec<ParamClass> {
    if total_size == 0 {
        return vec![ParamClass::NoClass];
    }
    if total_size > MAX_REGISTER_AGGREGATE_SIZE {
        return vec![ParamClass::Memory];
    }
    let num_eightbytes = (total_size + EIGHTBYTE_SIZE - 1) / EIGHTBYTE_SIZE;
    vec![ParamClass::Integer; num_eightbytes]
}

// ---------------------------------------------------------------------------
// compute_param_locations — assign parameters to registers or stack
// ---------------------------------------------------------------------------

/// Assigns each function parameter to a physical location (register or
/// stack slot) based on the System V AMD64 calling convention.
///
/// # Arguments
///
/// * `param_types` — the C types of the function's formal parameters,
///   in declaration order
///
/// # Returns
///
/// A `Vec<ParamLocation>` with the same length as `param_types`, where
/// each entry describes where the corresponding parameter is passed.
///
/// # Algorithm
///
/// 1. Classify each parameter using [`classify_type`].
/// 2. For each classified parameter, try to assign it to the next
///    available register(s) in the integer or SSE sequence.
/// 3. If no register is available, assign the parameter to the stack.
pub fn compute_param_locations(param_types: &[CType]) -> Vec<ParamLocation> {
    let target = Target::X86_64;
    let mut locations = Vec::with_capacity(param_types.len());

    let mut int_reg_idx: usize = 0;
    let mut sse_reg_idx: usize = 0;
    let mut stack_offset: u32 = 0;

    for ty in param_types {
        let classes = classify_type(ty, &target);

        // Count how many integer and SSE registers this parameter needs.
        let int_needed = classes.iter()
            .filter(|c| **c == ParamClass::Integer)
            .count();
        let sse_needed = classes.iter()
            .filter(|c| **c == ParamClass::SSE)
            .count();
        let has_memory = classes.contains(&ParamClass::Memory);
        let has_x87 = classes.contains(&ParamClass::X87);

        // X87 or Memory types always go on the stack.
        if has_memory || has_x87 {
            let size = type_size_bytes(ty, &target) as u32;
            let aligned_size = align_up(size, 8);
            locations.push(ParamLocation::Stack {
                offset: stack_offset,
                size: aligned_size,
            });
            stack_offset += aligned_size;
            continue;
        }

        // Check if we have enough registers for this parameter.
        let int_avail = registers::ARG_REGS_INT.len() - int_reg_idx;
        let sse_avail = registers::ARG_REGS_FLOAT.len() - sse_reg_idx;

        if int_needed > int_avail || sse_needed > sse_avail {
            // Not enough registers — pass on the stack.
            let size = type_size_bytes(ty, &target) as u32;
            let aligned_size = align_up(size, 8);
            locations.push(ParamLocation::Stack {
                offset: stack_offset,
                size: aligned_size,
            });
            stack_offset += aligned_size;
            continue;
        }

        // Assign registers based on the classification.
        match (int_needed, sse_needed) {
            (1, 0) => {
                locations.push(ParamLocation::IntReg(
                    registers::ARG_REGS_INT[int_reg_idx],
                ));
                int_reg_idx += 1;
            }
            (0, 1) => {
                locations.push(ParamLocation::SseReg(
                    registers::ARG_REGS_FLOAT[sse_reg_idx],
                ));
                sse_reg_idx += 1;
            }
            (2, 0) => {
                locations.push(ParamLocation::IntRegPair(
                    registers::ARG_REGS_INT[int_reg_idx],
                    registers::ARG_REGS_INT[int_reg_idx + 1],
                ));
                int_reg_idx += 2;
            }
            (0, 2) => {
                locations.push(ParamLocation::SseRegPair(
                    registers::ARG_REGS_FLOAT[sse_reg_idx],
                    registers::ARG_REGS_FLOAT[sse_reg_idx + 1],
                ));
                sse_reg_idx += 2;
            }
            (1, 1) => {
                // Mixed: first eightbyte INTEGER, second SSE (or vice versa).
                // Determine order from the classification vector.
                let first_is_int = classes.first() == Some(&ParamClass::Integer);
                if first_is_int {
                    locations.push(ParamLocation::MixedRegPair {
                        first: registers::ARG_REGS_INT[int_reg_idx],
                        second: registers::ARG_REGS_FLOAT[sse_reg_idx],
                        first_is_int: true,
                    });
                } else {
                    locations.push(ParamLocation::MixedRegPair {
                        first: registers::ARG_REGS_FLOAT[sse_reg_idx],
                        second: registers::ARG_REGS_INT[int_reg_idx],
                        first_is_int: false,
                    });
                }
                int_reg_idx += 1;
                sse_reg_idx += 1;
            }
            _ => {
                // Complex classification — fall back to MEMORY.
                let size = type_size_bytes(ty, &target) as u32;
                let aligned_size = align_up(size, 8);
                locations.push(ParamLocation::Stack {
                    offset: stack_offset,
                    size: aligned_size,
                });
                stack_offset += aligned_size;
            }
        }
    }

    locations
}

// ---------------------------------------------------------------------------
// compute_return_location — assign return value to register(s) or memory
// ---------------------------------------------------------------------------

/// Determines the physical location for a function's return value based
/// on the System V AMD64 calling convention.
///
/// # Arguments
///
/// * `return_type` — the C type of the function's return value
///
/// # Returns
///
/// A [`ParamLocation`] describing where the return value is placed:
/// - INTEGER types: `IntReg(RAX)` or `IntRegPair(RAX, RDX)` for 128-bit
/// - SSE types: `SseReg(XMM0)` or `SseRegPair(XMM0, XMM1)` for complex
/// - X87 types: `X87Stack` (ST(0) / ST(1))
/// - MEMORY: `Stack` (caller provides hidden pointer in RDI)
/// - Void: `IntReg(RAX)` (unused, convention only)
pub fn compute_return_location(return_type: &CType) -> ParamLocation {
    let target = Target::X86_64;
    let classes = classify_type(return_type, &target);

    // Void functions: return value is unused, but RAX is the convention.
    if return_type.is_void() {
        return ParamLocation::IntReg(registers::RAX);
    }

    // Count class occurrences.
    let int_count = classes.iter()
        .filter(|c| **c == ParamClass::Integer)
        .count();
    let sse_count = classes.iter()
        .filter(|c| **c == ParamClass::SSE)
        .count();
    let has_memory = classes.contains(&ParamClass::Memory);
    let has_x87 = classes.contains(&ParamClass::X87);

    // MEMORY class: returned via hidden first argument (RDI points to
    // caller-allocated space, RAX returns the same pointer).
    if has_memory {
        return ParamLocation::Stack {
            offset: 0,
            size: type_size_bytes(return_type, &target) as u32,
        };
    }

    // X87 class: returned on the x87 FPU stack (ST(0)).
    if has_x87 {
        return ParamLocation::X87Stack;
    }

    match (int_count, sse_count) {
        (1, 0) => ParamLocation::IntReg(registers::RAX),
        (2, 0) => ParamLocation::IntRegPair(registers::RAX, registers::RDX),
        (0, 1) => ParamLocation::SseReg(registers::XMM0),
        (0, 2) => ParamLocation::SseRegPair(registers::XMM0, registers::XMM1),
        (1, 1) => {
            let first_is_int = classes.first() == Some(&ParamClass::Integer);
            if first_is_int {
                ParamLocation::MixedRegPair {
                    first: registers::RAX,
                    second: registers::XMM0,
                    first_is_int: true,
                }
            } else {
                ParamLocation::MixedRegPair {
                    first: registers::XMM0,
                    second: registers::RAX,
                    first_is_int: false,
                }
            }
        }
        _ => {
            // Fallback: treat as MEMORY return.
            ParamLocation::Stack {
                offset: 0,
                size: type_size_bytes(return_type, &target) as u32,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Aligns `value` up to the next multiple of `alignment`.
#[inline]
fn align_up(value: u32, alignment: u32) -> u32 {
    (value.wrapping_add(alignment - 1)) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_void() {
        let classes = classify_type(&CType::Void, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::NoClass]);
    }

    #[test]
    fn classify_bool() {
        let classes = classify_type(&CType::Bool, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_int() {
        let classes = classify_type(&CType::Int { signed: true }, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_unsigned_long() {
        let classes = classify_type(&CType::Long { signed: false }, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn classify_float() {
        let classes = classify_type(&CType::Float, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::SSE]);
    }

    #[test]
    fn classify_double() {
        let classes = classify_type(&CType::Double, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::SSE]);
    }

    #[test]
    fn classify_long_double() {
        let classes = classify_type(&CType::LongDouble, &Target::X86_64);
        assert_eq!(classes, vec![ParamClass::X87, ParamClass::X87Up]);
    }

    #[test]
    fn classify_pointer() {
        let classes = classify_type(
            &CType::Pointer(Box::new(CType::Int { signed: true })),
            &Target::X86_64,
        );
        assert_eq!(classes, vec![ParamClass::Integer]);
    }

    #[test]
    fn compute_return_int() {
        let loc = compute_return_location(&CType::Int { signed: true });
        assert_eq!(loc, ParamLocation::IntReg(registers::RAX));
    }

    #[test]
    fn compute_return_double() {
        let loc = compute_return_location(&CType::Double);
        assert_eq!(loc, ParamLocation::SseReg(registers::XMM0));
    }

    #[test]
    fn compute_return_void() {
        let loc = compute_return_location(&CType::Void);
        assert_eq!(loc, ParamLocation::IntReg(registers::RAX));
    }

    #[test]
    fn compute_return_long_double() {
        let loc = compute_return_location(&CType::LongDouble);
        assert_eq!(loc, ParamLocation::X87Stack);
    }

    #[test]
    fn param_locations_basic() {
        let params = vec![
            CType::Int { signed: true },          // → RDI
            CType::Pointer(Box::new(CType::Void)), // → RSI
            CType::Double,                          // → XMM0
        ];
        let locs = compute_param_locations(&params);
        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0], ParamLocation::IntReg(registers::ARG_REGS_INT[0]));
        assert_eq!(locs[1], ParamLocation::IntReg(registers::ARG_REGS_INT[1]));
        assert_eq!(locs[2], ParamLocation::SseReg(registers::ARG_REGS_FLOAT[0]));
    }

    #[test]
    fn param_locations_overflow_to_stack() {
        // 7 integer args — first 6 go in registers, 7th goes on stack.
        let params: Vec<CType> = (0..7)
            .map(|_| CType::Int { signed: true })
            .collect();
        let locs = compute_param_locations(&params);
        assert_eq!(locs.len(), 7);
        for (loc, &reg) in locs.iter().zip(registers::ARG_REGS_INT.iter()).take(6) {
            assert_eq!(*loc, ParamLocation::IntReg(reg));
        }
        assert!(matches!(locs[6], ParamLocation::Stack { .. }));
    }
}
