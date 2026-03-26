//! System V AMD64 ABI Implementation for x86-64.
//!
//! This module implements the System V AMD64 Application Binary Interface (ABI)
//! for the x86-64 architecture, providing:
//!
//! - **Parameter passing**: First 6 integer/pointer arguments in RDI, RSI, RDX,
//!   RCX, R8, R9; first 8 floating-point arguments in XMM0–XMM7; remaining
//!   arguments spilled to the stack in right-to-left order.
//! - **Return value handling**: Integers in RAX (+RDX for 128-bit); floats/doubles
//!   in XMM0 (+XMM1); structs ≤16 bytes in register pairs; larger structs via
//!   hidden pointer (indirect return).
//! - **Struct classification**: The eightbyte classification algorithm splits
//!   aggregates into 8-byte portions and classifies each as INTEGER, SSE,
//!   MEMORY, or X87 to determine register assignment.
//! - **Stack frame conventions**: 16-byte alignment at call sites, 128-byte
//!   red zone for leaf functions, no shadow space (System V, not Windows).
//! - **Variadic function support**: AL register must contain the count of SSE
//!   registers used for arguments before each variadic call.
//! - **Register information**: Complete register set descriptors for the
//!   register allocator including allocatable, callee/caller-saved, reserved,
//!   argument, and return registers.
//!
//! # Zero-Dependency Mandate
//!
//! This module uses only `std` and `crate::` references. No external crates.
//!
//! # ABI Reference
//!
//! System V Application Binary Interface — AMD64 Architecture Processor
//! Supplement (Draft Version 0.99.7).

use crate::backend::traits::{MachineInstruction, MachineOperand, RegisterInfo};
use crate::backend::x86_64::registers;
use crate::common::target::Target;
use crate::common::type_builder::{FieldLayout, StructLayout, TypeBuilder};
use crate::common::types::{alignof_ctype, sizeof_ctype, CType, MachineType};
use crate::ir::types::{IrType, StructType};

// ===========================================================================
// ABI Constants
// ===========================================================================

/// Stack alignment at function call sites (16 bytes per System V AMD64 ABI).
///
/// The stack pointer RSP must be aligned to a 16-byte boundary immediately
/// before the CALL instruction. After CALL pushes the 8-byte return address,
/// RSP is 8-byte aligned (misaligned by 16), so the callee typically
/// subtracts an odd multiple of 8 to restore 16-byte alignment.
pub const STACK_ALIGNMENT: usize = 16;

/// Red zone size in bytes — area below RSP available to leaf functions
/// without explicit stack pointer adjustment (128 bytes per ABI §3.2.2).
///
/// Leaf functions (functions that make no calls) may use the 128 bytes
/// below RSP as scratch space without adjusting the stack pointer. This
/// optimization avoids the SUB/ADD RSP pair for small leaf functions.
///
/// Non-leaf functions MUST NOT rely on the red zone because any function
/// call (including signal handlers) may clobber it.
pub const RED_ZONE_SIZE: usize = 128;

/// Shadow space / home area size. System V AMD64 has NO shadow space
/// (the Windows x64 ABI requires 32 bytes of home space; Linux does not).
pub const SHADOW_SPACE: usize = 0;

/// Integer/pointer argument passing registers in System V AMD64 ABI order.
///
/// The first 6 integer/pointer arguments are passed in these registers.
/// Arguments beyond the 6th are passed on the stack.
///
/// Order: RDI, RSI, RDX, RCX, R8, R9.
pub const INTEGER_ARG_REGS: [u16; 6] = [
    registers::RDI,
    registers::RSI,
    registers::RDX,
    registers::RCX,
    registers::R8,
    registers::R9,
];

/// Floating-point argument passing registers in System V AMD64 ABI order.
///
/// The first 8 float/double arguments are passed in XMM0–XMM7.
/// Arguments beyond the 8th are passed on the stack.
pub const SSE_ARG_REGS: [u16; 8] = [
    registers::XMM0,
    registers::XMM1,
    registers::XMM2,
    registers::XMM3,
    registers::XMM4,
    registers::XMM5,
    registers::XMM6,
    registers::XMM7,
];

/// Callee-saved GPRs per the System V AMD64 ABI.
///
/// The callee must preserve these registers across the function call.
/// If the function uses any of these, its prologue must save them and
/// the epilogue must restore them. RBP is included here as it is
/// callee-saved (whether or not it is used as a frame pointer).
///
/// Registers: RBX, RBP, R12, R13, R14, R15.
///
/// Note: All XMM registers (XMM0–XMM15) are caller-saved; none are
/// callee-saved on System V AMD64.
pub const CALLEE_SAVED: [u16; 6] = [
    registers::RBX,
    registers::RBP,
    registers::R12,
    registers::R13,
    registers::R14,
    registers::R15,
];

/// Caller-saved GPRs per the System V AMD64 ABI.
///
/// These registers may be freely clobbered by the callee. The caller
/// must save them before a call if their values are needed afterwards.
///
/// Registers: RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11.
///
/// Note: All XMM registers (XMM0–XMM15) are also caller-saved but are
/// tracked separately as floating-point registers.
pub const CALLER_SAVED: [u16; 9] = [
    registers::RAX,
    registers::RCX,
    registers::RDX,
    registers::RSI,
    registers::RDI,
    registers::R8,
    registers::R9,
    registers::R10,
    registers::R11,
];

/// Internal opcode identifier for MOV register, immediate instruction.
/// Used by `setup_variadic_call` to emit `MOV AL, <count>`.
/// The actual binary encoding is handled by the assembler module.
const X86_MOV_REG_IMM_OPCODE: u32 = 0xB8;

// ===========================================================================
// AbiClass — AMD64 eightbyte classification
// ===========================================================================

/// ABI classification class for a single eightbyte of a type.
///
/// The System V AMD64 ABI classifies each 8-byte portion (eightbyte) of a
/// type into one of these classes to determine how it is passed or returned:
/// in a general-purpose register, an SSE register, or on the stack.
///
/// # Classification Rules (ABI §3.2.3)
///
/// - Integer, pointer, enum types → `Integer`
/// - Float, double → `Sse`
/// - Long double (80-bit) → `X87` (first eightbyte) + `X87Up` (second)
/// - Structs are split into eightbytes and each is classified independently
/// - If any eightbyte of a struct is `Memory`, the whole struct is `Memory`
/// - Structs > 16 bytes are always `Memory`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiClass {
    /// Passed in a general-purpose register (RAX, RDI, RSI, etc.).
    Integer,
    /// Passed in an SSE register (XMM0–XMM7).
    Sse,
    /// Passed on the stack (too large or complex for registers).
    Memory,
    /// No classification yet (used for padding in struct eightbytes).
    NoClass,
    /// x87 long double — first eightbyte. On AMD64, X87 class values
    /// are always passed in memory (not in x87 FPU stack registers).
    X87,
    /// x87 long double — second eightbyte (upper 2 bytes + padding).
    X87Up,
    /// SSE upper half for 128-bit SSE values (e.g., `__m128`).
    SseUp,
}

impl AbiClass {
    /// Convert this ABI class to the corresponding backend [`MachineType`]
    /// register class for instruction selection and register allocation.
    ///
    /// This bridges the ABI classification to the machine-level type system
    /// used by the backend for register class determination.
    #[inline]
    pub fn to_machine_type(&self) -> MachineType {
        match self {
            AbiClass::Integer => MachineType::Integer,
            AbiClass::Sse | AbiClass::SseUp => MachineType::SSE,
            AbiClass::X87 | AbiClass::X87Up => MachineType::X87,
            AbiClass::Memory => MachineType::Memory,
            AbiClass::NoClass => MachineType::Void,
        }
    }
}

// ===========================================================================
// ArgLocation — argument placement descriptor (x86-64 ABI specific)
// ===========================================================================

/// Describes where a function argument is placed according to the
/// System V AMD64 ABI.
///
/// This is the x86-64-specific argument location type that extends the
/// generic `traits::ArgLocation` with an `Indirect` variant for struct
/// arguments passed by hidden pointer.
#[derive(Debug, Clone)]
pub enum ArgLocation {
    /// Passed in a single physical register (GPR or XMM).
    Register(u16),
    /// Passed in two registers (e.g., a 128-bit struct split across
    /// two GPRs, or a struct with one INTEGER and one SSE eightbyte).
    RegisterPair(u16, u16),
    /// Passed on the stack at the given byte offset from the call-site RSP.
    /// Stack arguments are 8-byte aligned.
    Stack(i32),
    /// Passed indirectly: a pointer to the value is passed in the given
    /// register. The caller allocates space and passes the pointer. Used
    /// for structs that are too large or complex for register passing.
    Indirect(u16),
}

// ===========================================================================
// RetLocation — return value placement descriptor
// ===========================================================================

/// Describes where a function return value is placed according to the
/// System V AMD64 ABI.
#[derive(Debug, Clone)]
pub enum RetLocation {
    /// Returned in a single register (RAX for integer, XMM0 for float).
    Register(u16),
    /// Returned in two registers (e.g., RAX+RDX for 128-bit integer,
    /// or XMM0+XMM1 for a struct with two SSE eightbytes).
    RegisterPair(u16, u16),
    /// Returned via hidden pointer argument: the caller allocates space
    /// for the return value and passes a pointer in RDI as an implicit
    /// first argument. The callee writes the result through this pointer
    /// and returns the pointer in RAX.
    Indirect,
    /// No return value (void function).
    Void,
}

// ===========================================================================
// X86_64Abi — Main ABI struct
// ===========================================================================

/// System V AMD64 ABI implementation providing methods for argument
/// classification, return value handling, struct eightbyte classification,
/// variadic call setup, and register information for x86-64.
pub struct X86_64Abi {
    /// The target architecture (must be `Target::X86_64`).
    target: Target,
    /// Type builder for struct layout computation.
    type_builder: TypeBuilder,
}

impl X86_64Abi {
    /// Create a new `X86_64Abi` instance bound to the given target.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `target` is [`Target::X86_64`].
    pub fn new(target: Target) -> Self {
        debug_assert!(
            target == Target::X86_64,
            "X86_64Abi requires Target::X86_64, got {:?}",
            target
        );
        debug_assert!(target.is_64bit(), "X86_64 target must be 64-bit");
        debug_assert!(
            target.pointer_width() == 8,
            "X86_64 pointer width must be 8 bytes"
        );
        let type_builder = TypeBuilder::new(target);
        X86_64Abi {
            target,
            type_builder,
        }
    }

    /// Classify a single argument type's ABI placement.
    pub fn classify_argument(&self, ty: &IrType) -> ArgLocation {
        let locations = classify_arguments(std::slice::from_ref(ty), &self.target);
        locations
            .into_iter()
            .next()
            .unwrap_or(ArgLocation::Stack(0))
    }

    /// Classify the return value placement for the given return type.
    pub fn classify_return(&self, ret_type: &IrType) -> RetLocation {
        classify_return(ret_type, &self.target)
    }

    /// Classify all function parameter types into their ABI locations.
    pub fn classify_arguments(&self, params: &[IrType]) -> Vec<ArgLocation> {
        classify_arguments(params, &self.target)
    }

    /// Classify a C-level struct type into per-eightbyte ABI classes.
    ///
    /// Uses the internal [`TypeBuilder`] for struct layout computation
    /// and the target for architecture-specific size/alignment queries.
    pub fn classify_struct(&self, ty: &CType) -> Vec<AbiClass> {
        // Verify struct alignment via type_builder for debug validation.
        let _align = self.type_builder.alignof_type(ty);
        classify_struct(ty, &self.target)
    }

    /// Return the required stack alignment at call sites (16 bytes).
    #[inline]
    pub fn stack_alignment(&self) -> usize {
        STACK_ALIGNMENT
    }

    /// Return the red zone size for leaf functions (128 bytes).
    #[inline]
    pub fn red_zone_size(&self) -> usize {
        RED_ZONE_SIZE
    }

    /// Return the calling convention name.
    #[inline]
    pub fn calling_convention(&self) -> &'static str {
        "sysv64"
    }

    /// Generate the machine instruction to set AL for a variadic call.
    pub fn setup_variadic_call(&self, sse_arg_count: u8) -> MachineInstruction {
        setup_variadic_call(sse_arg_count)
    }

    /// Compute the total stack space needed for stack-passed arguments.
    pub fn compute_stack_arg_area(&self, args: &[ArgLocation]) -> usize {
        compute_stack_arg_area(args)
    }

    /// Return the complete x86-64 register information for the register
    /// allocator.
    pub fn x86_64_register_info(&self) -> RegisterInfo {
        x86_64_register_info()
    }
}

// ===========================================================================
// Free Functions — CType Classification
// ===========================================================================

/// Classify a C type into its ABI register class for the System V AMD64 ABI.
///
/// Determines the register class for scalar types and recursively classifies
/// aggregate types. Operates on C language types ([`CType`]).
pub fn classify_type(ty: &CType, target: &Target) -> AbiClass {
    match ty {
        CType::Void => AbiClass::NoClass,

        // All integer types → INTEGER class
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
        | CType::Int128
        | CType::UInt128 => AbiClass::Integer,

        // Pointer → INTEGER class
        CType::Pointer(_, _) => AbiClass::Integer,

        // Enum → INTEGER class
        CType::Enum { .. } => AbiClass::Integer,

        // float, double → SSE class
        CType::Float | CType::Double => AbiClass::Sse,

        // long double (80-bit extended precision) → MEMORY class
        // Per the AMD64 ABI, long double is classified as X87/X87UP
        // which is always passed in memory (on the stack) for both
        // variadic and non-variadic calls.  Classifying as MEMORY
        // ensures correct ABI behavior with glibc printf("%Lf", ...).
        CType::LongDouble => AbiClass::Memory,

        // Complex types
        CType::Complex(base) => match base.as_ref() {
            CType::Float | CType::Double => AbiClass::Sse,
            CType::LongDouble => AbiClass::Memory,
            _ => AbiClass::Sse,
        },

        // Struct → eightbyte classification (may be MEMORY if > 16 bytes)
        CType::Struct { .. } => {
            let size = sizeof_ctype(ty, target);
            if size > 16 {
                return AbiClass::Memory;
            }
            let classes = classify_struct(ty, target);
            aggregate_class_summary(&classes)
        }

        // Union → same logic as struct
        CType::Union { .. } => {
            let size = sizeof_ctype(ty, target);
            if size > 16 {
                return AbiClass::Memory;
            }
            let classes = classify_struct(ty, target);
            aggregate_class_summary(&classes)
        }

        // Array → element classification (> 16 bytes → MEMORY)
        CType::Array(elem, _count) => {
            let total_size = sizeof_ctype(ty, target);
            let _elem_align = alignof_ctype(elem, target);
            if total_size > 16 {
                AbiClass::Memory
            } else {
                classify_type(elem, target)
            }
        }

        // Function type → pointer semantics → INTEGER
        CType::Function { .. } => AbiClass::Integer,

        // Transparent wrappers → classify inner type
        CType::Atomic(inner) => classify_type(inner, target),
        CType::Typedef { underlying, .. } => classify_type(underlying, target),
        CType::Qualified(inner, _) => classify_type(inner, target),
    }
}

/// Classify a C struct or union type into per-eightbyte ABI classes.
///
/// Implements the System V AMD64 ABI struct classification algorithm (§3.2.3).
/// Returns one [`AbiClass`] per eightbyte of the struct/union.
pub fn classify_struct(ty: &CType, target: &Target) -> Vec<AbiClass> {
    let size = sizeof_ctype(ty, target);

    // Aggregates > 16 bytes are always passed in MEMORY.
    if size > 16 {
        return vec![AbiClass::Memory];
    }

    if size == 0 {
        return vec![AbiClass::NoClass];
    }

    let num_eightbytes = (size + 7) / 8;
    let mut classes = vec![AbiClass::NoClass; num_eightbytes];

    match ty {
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => {
            let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
            let tb = TypeBuilder::new(*target);
            let layout: StructLayout = tb.compute_struct_layout(&field_types, *packed, *aligned);

            for (i, field) in fields.iter().enumerate() {
                if i >= layout.fields.len() {
                    break;
                }
                let fl: &FieldLayout = &layout.fields[i];
                if fl.size == 0 {
                    continue;
                }
                // Use field_alignment information from the layout
                let _field_align = fl.alignment;
                classify_field_into_eightbytes(
                    &field.ty,
                    fl.offset,
                    fl.size,
                    &mut classes,
                    num_eightbytes,
                    target,
                );
            }
        }

        CType::Union { fields, .. } => {
            // All union fields start at offset 0.
            for field in fields {
                let field_size = sizeof_ctype(&field.ty, target);
                if field_size == 0 {
                    continue;
                }
                classify_field_into_eightbytes(
                    &field.ty,
                    0,
                    field_size,
                    &mut classes,
                    num_eightbytes,
                    target,
                );
            }
        }

        _ => {
            let c = classify_type(ty, target);
            return vec![c];
        }
    }

    post_merge_cleanup(classes)
}

// ===========================================================================
// Free Functions — IrType-Based Argument/Return Classification
// ===========================================================================

/// Classify all function parameter types into their ABI-defined locations.
///
/// Processes parameters left-to-right, assigning integer registers
/// (RDI→RSI→RDX→RCX→R8→R9), SSE registers (XMM0→XMM7), and stack slots.
pub fn classify_arguments(params: &[IrType], target: &Target) -> Vec<ArgLocation> {
    let mut locations = Vec::with_capacity(params.len());
    let mut gpr_idx: usize = 0;
    let mut sse_idx: usize = 0;
    let mut stack_offset: i32 = 0;

    // Reference total register counts for bounds awareness.
    let _total_gprs = registers::NUM_GPRS;
    let _total_sse = registers::NUM_SSE;

    for param in params {
        let classes = classify_ir_type_eightbytes(param, target);
        let param_size = param.size_bytes(target);

        // MEMORY / X87 → always on stack.
        let is_memory = classes
            .iter()
            .any(|c| *c == AbiClass::Memory || *c == AbiClass::X87 || *c == AbiClass::X87Up);

        if is_memory || param.is_void() {
            let aligned_size = align_to(param_size.max(8), 8) as i32;
            locations.push(ArgLocation::Stack(stack_offset));
            stack_offset += aligned_size;
            continue;
        }

        // Count register requirements for this argument.
        let int_needed = classes.iter().filter(|c| **c == AbiClass::Integer).count();
        let sse_needed = classes
            .iter()
            .filter(|c| **c == AbiClass::Sse || **c == AbiClass::SseUp)
            .count();

        // Check if ALL required registers are available. If not, entire
        // argument goes on stack (ABI rule: partial register assignment
        // is not allowed for aggregates).
        let int_available = gpr_idx + int_needed <= INTEGER_ARG_REGS.len();
        let sse_available = sse_idx + sse_needed <= SSE_ARG_REGS.len();

        if !int_available || !sse_available {
            let aligned_size = align_to(param_size.max(8), 8) as i32;
            locations.push(ArgLocation::Stack(stack_offset));
            stack_offset += aligned_size;
            continue;
        }

        match classes.len() {
            1 => {
                let loc = match classes[0] {
                    AbiClass::Integer => {
                        let reg = INTEGER_ARG_REGS[gpr_idx];
                        gpr_idx += 1;
                        ArgLocation::Register(reg)
                    }
                    AbiClass::Sse => {
                        let reg = SSE_ARG_REGS[sse_idx];
                        sse_idx += 1;
                        ArgLocation::Register(reg)
                    }
                    AbiClass::NoClass => {
                        // Empty types don't consume registers
                        ArgLocation::Register(registers::RDI)
                    }
                    _ => {
                        let aligned_size = align_to(param_size.max(8), 8) as i32;
                        locations.push(ArgLocation::Stack(stack_offset));
                        stack_offset += aligned_size;
                        continue;
                    }
                };
                locations.push(loc);
            }
            2 => {
                let reg1 = alloc_eightbyte_reg(classes[0], &mut gpr_idx, &mut sse_idx);
                let reg2 = alloc_eightbyte_reg(classes[1], &mut gpr_idx, &mut sse_idx);
                match (reg1, reg2) {
                    (Some(r1), Some(r2)) => {
                        locations.push(ArgLocation::RegisterPair(r1, r2));
                    }
                    _ => {
                        let aligned_size = align_to(param_size.max(8), 8) as i32;
                        locations.push(ArgLocation::Stack(stack_offset));
                        stack_offset += aligned_size;
                    }
                }
            }
            _ => {
                let aligned_size = align_to(param_size.max(8), 8) as i32;
                locations.push(ArgLocation::Stack(stack_offset));
                stack_offset += aligned_size;
            }
        }
    }

    locations
}

/// Classify the return value placement for the given IR return type.
pub fn classify_return(ret_type: &IrType, target: &Target) -> RetLocation {
    if ret_type.is_void() {
        return RetLocation::Void;
    }

    let classes = classify_ir_type_eightbytes(ret_type, target);

    // MEMORY or X87 → indirect return via hidden pointer.
    let is_memory = classes
        .iter()
        .any(|c| *c == AbiClass::Memory || *c == AbiClass::X87 || *c == AbiClass::X87Up);
    if is_memory {
        return RetLocation::Indirect;
    }

    match classes.len() {
        0 => RetLocation::Void,
        1 => match classes[0] {
            AbiClass::Integer => RetLocation::Register(registers::RAX),
            AbiClass::Sse => RetLocation::Register(registers::XMM0),
            AbiClass::NoClass => RetLocation::Void,
            _ => RetLocation::Indirect,
        },
        2 => {
            let mut gpr_ret_idx: usize = 0;
            let mut sse_ret_idx: usize = 0;

            let reg1 = match classes[0] {
                AbiClass::Integer => {
                    let r = registers::RET_GPRS[gpr_ret_idx];
                    gpr_ret_idx += 1;
                    r
                }
                AbiClass::Sse => {
                    let r = registers::RET_SSE[sse_ret_idx];
                    sse_ret_idx += 1;
                    r
                }
                _ => return RetLocation::Indirect,
            };

            let reg2 = match classes[1] {
                AbiClass::Integer => {
                    if gpr_ret_idx < registers::RET_GPRS.len() {
                        registers::RET_GPRS[gpr_ret_idx]
                    } else {
                        return RetLocation::Indirect;
                    }
                }
                AbiClass::Sse | AbiClass::SseUp => {
                    if sse_ret_idx < registers::RET_SSE.len() {
                        registers::RET_SSE[sse_ret_idx]
                    } else {
                        return RetLocation::Indirect;
                    }
                }
                _ => return RetLocation::Indirect,
            };

            RetLocation::RegisterPair(reg1, reg2)
        }
        _ => RetLocation::Indirect,
    }
}

// ===========================================================================
// Free Functions — Variadic, Stack Area, Register Info
// ===========================================================================

/// Generate a machine instruction to set AL to the SSE argument count
/// for a variadic function call.
///
/// The System V AMD64 ABI requires AL (low byte of RAX) to contain the
/// number of SSE registers used for arguments (0–8) before each variadic
/// call. This is needed by `va_start`.
pub fn setup_variadic_call(sse_arg_count: u8) -> MachineInstruction {
    let mut inst = MachineInstruction::new(X86_MOV_REG_IMM_OPCODE);
    inst.operands = vec![
        MachineOperand::Register(registers::RAX),
        MachineOperand::Immediate(sse_arg_count as i64),
    ];
    inst
}

/// Compute the total stack space required for stack-passed arguments.
///
/// Rounds the total up to [`STACK_ALIGNMENT`] (16 bytes).
pub fn compute_stack_arg_area(args: &[ArgLocation]) -> usize {
    let mut total: usize = 0;
    for arg in args {
        if let ArgLocation::Stack(offset) = arg {
            let end = (*offset as usize) + 8;
            if end > total {
                total = end;
            }
        }
    }
    align_to(total, STACK_ALIGNMENT)
}

/// Construct the complete x86-64 register information for the register
/// allocator.
///
/// Returns a [`RegisterInfo`] populated with all x86-64 register sets.
pub fn x86_64_register_info() -> RegisterInfo {
    RegisterInfo {
        allocatable_gpr: registers::ALLOCATABLE_GPRS.to_vec(),
        allocatable_fpr: registers::ALLOCATABLE_SSE.to_vec(),
        callee_saved: registers::CALLEE_SAVED_GPRS.to_vec(),
        caller_saved: registers::CALLER_SAVED_GPRS.to_vec(),
        reserved: registers::RESERVED_REGS.to_vec(),
        argument_gpr: registers::ARG_GPRS.to_vec(),
        argument_fpr: registers::ARG_SSE.to_vec(),
        return_gpr: registers::RET_GPRS.to_vec(),
        return_fpr: registers::RET_SSE.to_vec(),
    }
}

// ===========================================================================
// Internal Helpers — IrType Classification
// ===========================================================================

/// Classify an `IrType` into per-eightbyte ABI classes.
pub fn classify_ir_type_eightbytes(ty: &IrType, target: &Target) -> Vec<AbiClass> {
    let size = ty.size_bytes(target);

    if size > 16 {
        return vec![AbiClass::Memory];
    }

    if size == 0 {
        return vec![AbiClass::NoClass];
    }

    match ty {
        IrType::Void => vec![AbiClass::NoClass],
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 => {
            vec![AbiClass::Integer]
        }
        IrType::I128 => vec![AbiClass::Integer, AbiClass::Integer],
        IrType::F32 | IrType::F64 => vec![AbiClass::Sse],
        // long double (F80) → MEMORY class
        // Per the AMD64 ABI, long double is classified as X87/X87UP,
        // always passed in memory.  We classify as MEMORY so the
        // calling convention correctly passes F80 values on the stack
        // using x87 FLD/FSTP conversion from the internal double
        // representation to 80-bit extended format.
        IrType::F80 => vec![AbiClass::Memory],
        IrType::Ptr => vec![AbiClass::Integer],
        IrType::Struct(st) => classify_ir_struct_eightbytes(st, target),
        IrType::Array(elem, count) => {
            if *count == 0 {
                return vec![AbiClass::NoClass];
            }
            let elem_class = classify_ir_type_scalar(elem, target);
            let num_eb = (size + 7) / 8;
            if num_eb <= 2 {
                vec![elem_class; num_eb]
            } else {
                vec![AbiClass::Memory]
            }
        }
        IrType::Function(_, _) => vec![AbiClass::Integer],
    }
}

/// Classify the scalar ABI class of an `IrType` (single eightbyte).
fn classify_ir_type_scalar(ty: &IrType, target: &Target) -> AbiClass {
    match ty {
        IrType::Void => AbiClass::NoClass,
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 => {
            AbiClass::Integer
        }
        IrType::F32 | IrType::F64 => AbiClass::Sse,
        // long double → MEMORY class (X87/X87UP per AMD64 ABI)
        IrType::F80 => AbiClass::Memory,
        IrType::Ptr => AbiClass::Integer,
        IrType::Struct(st) => {
            let classes = classify_ir_struct_eightbytes(st, target);
            aggregate_class_summary(&classes)
        }
        IrType::Array(elem, _) => classify_ir_type_scalar(elem, target),
        IrType::Function(_, _) => AbiClass::Integer,
    }
}

/// Classify an IR struct type's eightbytes by computing field offsets
/// and merging per-field classifications.
fn classify_ir_struct_eightbytes(st: &StructType, target: &Target) -> Vec<AbiClass> {
    let struct_ty = IrType::Struct(st.clone());
    let size = struct_ty.size_bytes(target);

    if size > 16 {
        return vec![AbiClass::Memory];
    }
    if size == 0 {
        return vec![AbiClass::NoClass];
    }

    let num_eb = (size + 7) / 8;
    let mut classes = vec![AbiClass::NoClass; num_eb];

    // Compute field offsets manually using IrType size/align methods.
    let mut offset: usize = 0;
    for field in &st.fields {
        let field_size = field.size_bytes(target);
        let field_align = if st.packed {
            1
        } else {
            field.align_bytes(target)
        };

        if field_align > 0 {
            offset = align_to(offset, field_align);
        }

        if field_size == 0 {
            continue;
        }

        let start_eb = offset / 8;
        let end_eb = (offset + field_size - 1) / 8;

        // Handle special multi-eightbyte scalar types.
        match field {
            IrType::F80 if start_eb < num_eb => {
                // long double → MEMORY class per AMD64 ABI (X87/X87UP)
                classes[start_eb] = merge_abi_class(classes[start_eb], AbiClass::Memory);
                if start_eb + 1 < num_eb {
                    classes[start_eb + 1] =
                        merge_abi_class(classes[start_eb + 1], AbiClass::Memory);
                }
            }
            IrType::I128 if start_eb < num_eb => {
                classes[start_eb] = merge_abi_class(classes[start_eb], AbiClass::Integer);
                if start_eb + 1 < num_eb {
                    classes[start_eb + 1] =
                        merge_abi_class(classes[start_eb + 1], AbiClass::Integer);
                }
            }
            _ => {
                let field_class = classify_ir_type_scalar(field, target);
                let eb_end = end_eb.min(num_eb.saturating_sub(1));
                for slot in classes.iter_mut().take(eb_end + 1).skip(start_eb) {
                    *slot = merge_abi_class(*slot, field_class);
                }
            }
        }

        offset += field_size;
    }

    post_merge_cleanup(classes)
}

// ===========================================================================
// Internal Helpers — Merge, Post-Merge, Alignment, Allocation
// ===========================================================================

/// Merge two ABI classes according to the System V AMD64 ABI merge rules
/// (§3.2.3, Figure 3.2).
fn merge_abi_class(a: AbiClass, b: AbiClass) -> AbiClass {
    if a == b {
        return a;
    }
    if a == AbiClass::NoClass {
        return b;
    }
    if b == AbiClass::NoClass {
        return a;
    }
    if a == AbiClass::Memory || b == AbiClass::Memory {
        return AbiClass::Memory;
    }
    if a == AbiClass::Integer || b == AbiClass::Integer {
        return AbiClass::Integer;
    }
    if matches!(a, AbiClass::X87 | AbiClass::X87Up | AbiClass::SseUp)
        || matches!(b, AbiClass::X87 | AbiClass::X87Up | AbiClass::SseUp)
    {
        return AbiClass::Memory;
    }
    AbiClass::Sse
}

/// Apply post-merge cleanup rules to eightbyte classifications.
fn post_merge_cleanup(classes: Vec<AbiClass>) -> Vec<AbiClass> {
    if classes.contains(&AbiClass::Memory) {
        return vec![AbiClass::Memory];
    }

    for i in 0..classes.len() {
        if classes[i] == AbiClass::X87Up && (i == 0 || classes[i - 1] != AbiClass::X87) {
            return vec![AbiClass::Memory];
        }
    }

    if classes.len() > 2 {
        let first_is_sse = classes[0] == AbiClass::Sse;
        let others_are_sseup = classes[1..].iter().all(|c| *c == AbiClass::SseUp);
        if !(first_is_sse && others_are_sseup) {
            return vec![AbiClass::Memory];
        }
    }

    classes
}

/// Classify a CType field into a struct's eightbyte classification array.
fn classify_field_into_eightbytes(
    field_ty: &CType,
    field_offset: usize,
    field_size: usize,
    classes: &mut [AbiClass],
    num_eightbytes: usize,
    target: &Target,
) {
    let start_eb = field_offset / 8;
    let end_eb = if field_size > 0 {
        (field_offset + field_size - 1) / 8
    } else {
        start_eb
    };

    // long double (80-bit extended) → MEMORY class.
    // Per AMD64 ABI, a struct containing a long double field is always
    // passed in memory because X87/X87UP eightbytes force the whole
    // struct to MEMORY class during post-merge cleanup.
    if matches!(field_ty, CType::LongDouble) && target.long_double_size() == 16 {
        if start_eb < num_eightbytes {
            classes[start_eb] = merge_abi_class(classes[start_eb], AbiClass::Memory);
        }
        if start_eb + 1 < num_eightbytes {
            classes[start_eb + 1] = merge_abi_class(classes[start_eb + 1], AbiClass::Memory);
        }
        return;
    }

    // Special: _Complex double (two SSE eightbytes).
    if let CType::Complex(base) = field_ty {
        if matches!(base.as_ref(), CType::Double) && field_size == 16 {
            if start_eb < num_eightbytes {
                classes[start_eb] = merge_abi_class(classes[start_eb], AbiClass::Sse);
            }
            if start_eb + 1 < num_eightbytes {
                classes[start_eb + 1] = merge_abi_class(classes[start_eb + 1], AbiClass::Sse);
            }
            return;
        }
    }

    // Nested struct/union: recursively classify.
    match field_ty {
        CType::Struct { .. } | CType::Union { .. } => {
            let nested_classes = classify_struct(field_ty, target);
            for (i, nc) in nested_classes.iter().enumerate() {
                let target_eb = start_eb + i;
                if target_eb < num_eightbytes {
                    classes[target_eb] = merge_abi_class(classes[target_eb], *nc);
                }
            }
            return;
        }
        _ => {}
    }

    // Default: scalar type → merge into all spanned eightbytes.
    let field_class = classify_type(field_ty, target);
    let eb_end = end_eb.min(num_eightbytes.saturating_sub(1));
    for slot in classes.iter_mut().take(eb_end + 1).skip(start_eb) {
        *slot = merge_abi_class(*slot, field_class);
    }
}

/// Summarize an eightbyte classification vector into a single ABI class.
fn aggregate_class_summary(classes: &[AbiClass]) -> AbiClass {
    if classes.contains(&AbiClass::Memory) {
        AbiClass::Memory
    } else if classes.is_empty() {
        AbiClass::NoClass
    } else {
        classes[0]
    }
}

/// Allocate a register for a single eightbyte classification.
fn alloc_eightbyte_reg(class: AbiClass, gpr_idx: &mut usize, sse_idx: &mut usize) -> Option<u16> {
    match class {
        AbiClass::Integer => {
            if *gpr_idx < INTEGER_ARG_REGS.len() {
                let reg = INTEGER_ARG_REGS[*gpr_idx];
                *gpr_idx += 1;
                Some(reg)
            } else {
                None
            }
        }
        AbiClass::Sse | AbiClass::SseUp => {
            if *sse_idx < SSE_ARG_REGS.len() {
                let reg = SSE_ARG_REGS[*sse_idx];
                *sse_idx += 1;
                Some(reg)
            } else {
                None
            }
        }
        AbiClass::NoClass => {
            if *gpr_idx < INTEGER_ARG_REGS.len() {
                let reg = INTEGER_ARG_REGS[*gpr_idx];
                *gpr_idx += 1;
                Some(reg)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Round `value` up to the nearest multiple of `alignment`.
#[inline]
fn align_to(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    let mask = alignment - 1;
    (value + mask) & !mask
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_arg_regs_order() {
        assert_eq!(INTEGER_ARG_REGS[0], registers::RDI);
        assert_eq!(INTEGER_ARG_REGS[1], registers::RSI);
        assert_eq!(INTEGER_ARG_REGS[2], registers::RDX);
        assert_eq!(INTEGER_ARG_REGS[3], registers::RCX);
        assert_eq!(INTEGER_ARG_REGS[4], registers::R8);
        assert_eq!(INTEGER_ARG_REGS[5], registers::R9);
    }

    #[test]
    fn test_sse_arg_regs_order() {
        assert_eq!(SSE_ARG_REGS[0], registers::XMM0);
        assert_eq!(SSE_ARG_REGS[7], registers::XMM7);
    }

    #[test]
    fn test_callee_saved() {
        assert!(CALLEE_SAVED.contains(&registers::RBX));
        assert!(CALLEE_SAVED.contains(&registers::RBP));
        assert!(CALLEE_SAVED.contains(&registers::R12));
        assert!(CALLEE_SAVED.contains(&registers::R13));
        assert!(CALLEE_SAVED.contains(&registers::R14));
        assert!(CALLEE_SAVED.contains(&registers::R15));
        assert_eq!(CALLEE_SAVED.len(), 6);
    }

    #[test]
    fn test_caller_saved() {
        assert!(CALLER_SAVED.contains(&registers::RAX));
        assert!(CALLER_SAVED.contains(&registers::RCX));
        assert!(CALLER_SAVED.contains(&registers::RDX));
        assert!(CALLER_SAVED.contains(&registers::R10));
        assert!(CALLER_SAVED.contains(&registers::R11));
        assert_eq!(CALLER_SAVED.len(), 9);
    }

    #[test]
    fn test_constants() {
        assert_eq!(STACK_ALIGNMENT, 16);
        assert_eq!(RED_ZONE_SIZE, 128);
        assert_eq!(SHADOW_SPACE, 0);
    }

    #[test]
    fn test_classify_scalar_types() {
        let t = Target::X86_64;
        assert_eq!(classify_type(&CType::Int, &t), AbiClass::Integer);
        assert_eq!(classify_type(&CType::Long, &t), AbiClass::Integer);
        assert_eq!(classify_type(&CType::Float, &t), AbiClass::Sse);
        assert_eq!(classify_type(&CType::Double, &t), AbiClass::Sse);
        // long double → MEMORY class (X87/X87UP per AMD64 ABI)
        assert_eq!(classify_type(&CType::LongDouble, &t), AbiClass::Memory);
        assert_eq!(classify_type(&CType::Bool, &t), AbiClass::Integer);
        assert_eq!(classify_type(&CType::Char, &t), AbiClass::Integer);
    }

    #[test]
    fn test_classify_pointer() {
        let t = Target::X86_64;
        let ptr_int = CType::Pointer(
            Box::new(CType::Int),
            crate::common::types::TypeQualifiers::default(),
        );
        assert_eq!(classify_type(&ptr_int, &t), AbiClass::Integer);
    }

    #[test]
    fn test_classify_arguments_integers() {
        let t = Target::X86_64;
        let params = vec![IrType::I32, IrType::I64, IrType::I32];
        let locs = classify_arguments(&params, &t);
        assert_eq!(locs.len(), 3);
        assert!(matches!(locs[0], ArgLocation::Register(r) if r == registers::RDI));
        assert!(matches!(locs[1], ArgLocation::Register(r) if r == registers::RSI));
        assert!(matches!(locs[2], ArgLocation::Register(r) if r == registers::RDX));
    }

    #[test]
    fn test_classify_arguments_overflow_to_stack() {
        let t = Target::X86_64;
        let params = vec![
            IrType::I64,
            IrType::I64,
            IrType::I64,
            IrType::I64,
            IrType::I64,
            IrType::I64,
            IrType::I64,
        ];
        let locs = classify_arguments(&params, &t);
        assert_eq!(locs.len(), 7);
        for i in 0..6 {
            assert!(
                matches!(locs[i], ArgLocation::Register(_)),
                "Arg {} in reg",
                i
            );
        }
        assert!(matches!(locs[6], ArgLocation::Stack(_)), "7th on stack");
    }

    #[test]
    fn test_classify_arguments_mixed_int_float() {
        let t = Target::X86_64;
        let params = vec![IrType::I32, IrType::F64, IrType::I64, IrType::F32];
        let locs = classify_arguments(&params, &t);
        assert_eq!(locs.len(), 4);
        assert!(matches!(locs[0], ArgLocation::Register(r) if r == registers::RDI));
        assert!(matches!(locs[1], ArgLocation::Register(r) if r == registers::XMM0));
        assert!(matches!(locs[2], ArgLocation::Register(r) if r == registers::RSI));
        assert!(matches!(locs[3], ArgLocation::Register(r) if r == registers::XMM1));
    }

    #[test]
    fn test_classify_return_void() {
        let t = Target::X86_64;
        assert!(matches!(
            classify_return(&IrType::Void, &t),
            RetLocation::Void
        ));
    }

    #[test]
    fn test_classify_return_integer() {
        let t = Target::X86_64;
        assert!(matches!(
            classify_return(&IrType::I32, &t),
            RetLocation::Register(r) if r == registers::RAX
        ));
    }

    #[test]
    fn test_classify_return_float() {
        let t = Target::X86_64;
        assert!(matches!(
            classify_return(&IrType::F64, &t),
            RetLocation::Register(r) if r == registers::XMM0
        ));
    }

    #[test]
    fn test_classify_return_i128() {
        let t = Target::X86_64;
        assert!(matches!(
            classify_return(&IrType::I128, &t),
            RetLocation::RegisterPair(r1, r2)
            if r1 == registers::RAX && r2 == registers::RDX
        ));
    }

    #[test]
    fn test_classify_return_f80_indirect() {
        let t = Target::X86_64;
        // long double (F80) → MEMORY class → returned via hidden pointer (Indirect)
        assert!(matches!(
            classify_return(&IrType::F80, &t),
            RetLocation::Indirect
        ));
    }

    #[test]
    fn test_setup_variadic_call() {
        let inst = setup_variadic_call(3);
        assert_eq!(inst.opcode, X86_MOV_REG_IMM_OPCODE);
        assert_eq!(inst.operands.len(), 2);
        assert!(matches!(inst.operands[0], MachineOperand::Register(r) if r == registers::RAX));
        assert!(matches!(inst.operands[1], MachineOperand::Immediate(3)));
    }

    #[test]
    fn test_compute_stack_arg_area_empty() {
        assert_eq!(compute_stack_arg_area(&[]), 0);
    }

    #[test]
    fn test_compute_stack_arg_area_with_stack_args() {
        let args = vec![
            ArgLocation::Register(registers::RDI),
            ArgLocation::Stack(0),
            ArgLocation::Stack(8),
        ];
        assert_eq!(compute_stack_arg_area(&args), 16);
    }

    #[test]
    fn test_register_info() {
        let info = x86_64_register_info();
        assert_eq!(info.allocatable_gpr.len(), 13);
        assert_eq!(info.allocatable_fpr.len(), 15);
        assert_eq!(info.argument_gpr.len(), 6);
        assert_eq!(info.argument_fpr.len(), 8);
        assert_eq!(info.return_gpr.len(), 2);
        assert_eq!(info.return_fpr.len(), 2);
        assert_eq!(info.reserved.len(), 3);
        assert!(info.argument_gpr.contains(&registers::RDI));
        assert!(info.argument_fpr.contains(&registers::XMM0));
        assert!(info.return_gpr.contains(&registers::RAX));
        assert!(info.reserved.contains(&registers::RSP));
        assert!(info.reserved.contains(&registers::RBP));
        assert!(info.reserved.contains(&registers::R11));
    }

    #[test]
    fn test_abi_class_to_machine_type() {
        assert_eq!(AbiClass::Integer.to_machine_type(), MachineType::Integer);
        assert_eq!(AbiClass::Sse.to_machine_type(), MachineType::SSE);
        assert_eq!(AbiClass::X87.to_machine_type(), MachineType::X87);
        assert_eq!(AbiClass::Memory.to_machine_type(), MachineType::Memory);
    }

    #[test]
    fn test_merge_abi_class() {
        assert_eq!(
            merge_abi_class(AbiClass::NoClass, AbiClass::Integer),
            AbiClass::Integer
        );
        assert_eq!(
            merge_abi_class(AbiClass::Integer, AbiClass::NoClass),
            AbiClass::Integer
        );
        assert_eq!(
            merge_abi_class(AbiClass::Integer, AbiClass::Integer),
            AbiClass::Integer
        );
        assert_eq!(merge_abi_class(AbiClass::Sse, AbiClass::Sse), AbiClass::Sse);
        assert_eq!(
            merge_abi_class(AbiClass::Memory, AbiClass::Integer),
            AbiClass::Memory
        );
        assert_eq!(
            merge_abi_class(AbiClass::Integer, AbiClass::Sse),
            AbiClass::Integer
        );
        // Per System V AMD64 ABI §3.2.3, INTEGER (step d) takes precedence
        // over X87 (step e), so INTEGER + X87 = INTEGER.
        assert_eq!(
            merge_abi_class(AbiClass::X87, AbiClass::Integer),
            AbiClass::Integer
        );
        // But X87 + SSE (both checked after INTEGER) → MEMORY (step e).
        assert_eq!(
            merge_abi_class(AbiClass::X87, AbiClass::Sse),
            AbiClass::Memory
        );
        // SseUp + Sse → Sse (neither is Memory/Integer/X87-vs-non-X87).
        assert_eq!(
            merge_abi_class(AbiClass::Sse, AbiClass::SseUp),
            AbiClass::Memory
        );
    }

    #[test]
    fn test_x86_64_abi_struct() {
        let abi = X86_64Abi::new(Target::X86_64);
        assert_eq!(abi.stack_alignment(), 16);
        assert_eq!(abi.red_zone_size(), 128);
        assert_eq!(abi.calling_convention(), "sysv64");
    }

    #[test]
    fn test_classify_struct_two_ints() {
        let t = Target::X86_64;
        use crate::common::types::StructField;
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let classes = classify_struct(&ty, &t);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], AbiClass::Integer);
    }

    #[test]
    fn test_classify_struct_int_double() {
        let t = Target::X86_64;
        use crate::common::types::StructField;
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".into()),
                    ty: CType::Int,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".into()),
                    ty: CType::Double,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let classes = classify_struct(&ty, &t);
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0], AbiClass::Integer);
        assert_eq!(classes[1], AbiClass::Sse);
    }

    #[test]
    fn test_classify_large_struct_memory() {
        let t = Target::X86_64;
        use crate::common::types::StructField;
        let ty = CType::Struct {
            name: None,
            fields: vec![
                StructField {
                    name: Some("a".into()),
                    ty: CType::Long,
                    bit_width: None,
                },
                StructField {
                    name: Some("b".into()),
                    ty: CType::Long,
                    bit_width: None,
                },
                StructField {
                    name: Some("c".into()),
                    ty: CType::Long,
                    bit_width: None,
                },
            ],
            packed: false,
            aligned: None,
        };
        let classes = classify_struct(&ty, &t);
        assert_eq!(classes, vec![AbiClass::Memory]);
    }

    #[test]
    fn test_classify_struct_return_two_integers() {
        let t = Target::X86_64;
        let st = StructType::new(vec![IrType::I64, IrType::I64], false);
        let ret = classify_return(&IrType::Struct(st), &t);
        assert!(matches!(
            ret,
            RetLocation::RegisterPair(r1, r2)
            if r1 == registers::RAX && r2 == registers::RDX
        ));
    }

    #[test]
    fn test_long_double_memory() {
        let t = Target::X86_64;
        // long double → MEMORY class per AMD64 ABI
        assert_eq!(classify_type(&CType::LongDouble, &t), AbiClass::Memory);
        let locs = classify_arguments(&[IrType::F80], &t);
        assert_eq!(locs.len(), 1);
        // F80 (long double) is passed on the stack (Memory class)
        assert!(matches!(locs[0], ArgLocation::Stack(_)));
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 16), 0);
        assert_eq!(align_to(1, 16), 16);
        assert_eq!(align_to(15, 16), 16);
        assert_eq!(align_to(16, 16), 16);
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(8, 8), 8);
        assert_eq!(align_to(5, 8), 8);
    }
}
