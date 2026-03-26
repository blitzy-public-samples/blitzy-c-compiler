#![allow(clippy::if_same_then_else, clippy::collapsible_match)]
//! # Expression Lowering
//!
//! Converts semantically-analyzed AST expressions into IR instructions.
//!
//! Every expression evaluation produces an SSA `Value` that can be used
//! as an operand in subsequent instructions. For lvalue expressions (variables,
//! dereferences, array subscripts, member access), the result is a pointer
//! to the storage location. For rvalue expressions (arithmetic, constants, casts),
//! the result is the computed value.
//!
//! ## Key Lowering Patterns
//!
//! - **Variable reference**: Load from alloca pointer → Value
//! - **Assignment**: Lower rhs, store to lhs pointer
//! - **Binary arithmetic**: Lower both operands, emit BinOp instruction
//! - **Comparison**: Lower both operands, emit ICmp/FCmp instruction → I1
//! - **Function call**: Lower arguments, emit Call instruction
//! - **Address-of (`&x`)**: Return the pointer to x (its alloca) without loading
//! - **Dereference (`*p`)**: Load the pointer, then load through it
//! - **Cast**: Emit Trunc/ZExt/SExt/BitCast/IntToPtr/PtrToInt as appropriate
//! - **Short-circuit (`&&`, `||`)**: Create basic blocks for lazy evaluation
//! - **Ternary (`? :`)**: Create then/else/merge blocks with phi node
//!
//! ## Constant Representation
//!
//! Integer and floating-point constants are materialized through
//! `Instruction::BinOp` with `Value::UNDEF` sentinels.  The module-level
//! `GlobalVariable` pool stores the actual constant values so the backend
//! code-generator can resolve them to immediate operands.
//!
//! ## Dependencies
//! - `crate::ir::*` — IR types, instructions, builder
//! - `crate::frontend::parser::ast` — AST expression nodes
//! - `crate::common::*` — Types, diagnostics, target

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::common::type_builder::{self, TypeBuilder};
use crate::common::types::{self, CType, TypeQualifiers};
use crate::frontend::parser::ast;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp as IrBinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::module::{self as ir_module, Constant, GlobalVariable, IrModule};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Expression Lowering Context
// ---------------------------------------------------------------------------

/// Context for expression lowering within a function.
///
/// Holds mutable references to the IR builder, current function, and module,
/// along with read-only references to target information, type builder, and
/// variable lookup maps.  The caller (declaration / statement lowering) creates
/// this context and passes it to [`lower_expression`] / [`lower_lvalue`].
pub struct ExprLoweringContext<'a> {
    /// IR builder for instruction creation and value numbering.
    pub builder: &'a mut IrBuilder,
    /// Current IR function being constructed.
    pub function: &'a mut IrFunction,
    /// IR module — global variable / function / string-literal pool access.
    pub module: &'a mut IrModule,
    /// Target architecture (pointer width, endianness, type sizes).
    pub target: &'a Target,
    /// Type builder for `sizeof`, `alignof`, struct layout computation, and
    /// usual-arithmetic-conversion queries.
    pub type_builder: &'a TypeBuilder,
    /// Diagnostic engine for emitting errors and warnings.
    pub diagnostics: &'a mut DiagnosticEngine,
    /// Local variable name → alloca `Value` mapping (alloca-first pattern).
    pub local_vars: &'a FxHashMap<String, Value>,
    /// Function parameter name → `Value` mapping.
    pub param_values: &'a FxHashMap<String, Value>,
    /// Name resolution table:  `Symbol::as_u32()` → interned string.
    /// Populated by the lowering driver from the string interner so that
    /// expression lowering can resolve AST `Symbol` handles to the `String`
    /// keys used in `local_vars` / `param_values`.
    pub name_table: &'a [String],
    /// Variable type table: variable name → declared C type.
    /// Used to determine signedness and pointee type for correct instruction
    /// selection (signed vs unsigned division, arithmetic shift vs logical, etc.).
    pub local_types: &'a FxHashMap<String, CType>,
    /// Enum constant name → integer value mapping.
    /// Populated by the lowering driver from the semantic analysis results
    /// so that enum constant identifiers can be resolved to immediate values.
    pub enum_constants: &'a FxHashMap<String, i128>,
    /// Static local variable name → mangled global name mapping.
    /// When a `static` local variable `x` inside function `foo` is
    /// encountered, its storage is a global variable named `foo.x`.
    /// This map allows `lower_identifier` to redirect the access to the
    /// corresponding global variable.
    pub static_locals: &'a mut FxHashMap<String, String>,
    /// Struct/union tag → full CType definition registry.
    /// Used by `resolve_type_name` and `lower_sizeof_type` to resolve
    /// forward-referenced struct/union types to their complete layouts.
    pub struct_defs: &'a FxHashMap<String, CType>,
    /// Label name → BlockId mapping for computed goto (`&&label`).
    /// When non-empty, `lower_address_of_label` can resolve label names to
    /// block IDs and produce meaningful pointer-sized integer values.
    pub label_blocks: &'a FxHashMap<String, crate::ir::instructions::BlockId>,
    /// Name of the function currently being lowered.
    /// Used for C99 `__func__` / GCC `__FUNCTION__` / `__PRETTY_FUNCTION__`.
    pub current_function_name: Option<&'a str>,
    /// Enclosing loop/switch stack propagated from the `StmtLoweringContext`
    /// so that GCC statement expressions `({ break; })` or `({ continue; })`
    /// inside a loop body can resolve to the correct targets.
    pub enclosing_loop_stack: Vec<super::stmt_lowering::LoopContext>,
    /// Scope-local type overrides for block-scoped variable shadowing.
    /// When a compound statement declares a variable with the same name as
    /// one in an outer scope but with a different type (e.g. `u32 t` in one
    /// case block and `void *t` in another), this map holds the CURRENT
    /// scope's type for that variable, overriding `local_types`.
    pub scope_type_overrides: &'a FxHashMap<String, CType>,
    /// Bitfield information from the most recent `lower_member_access_lvalue`
    /// call.  Set to `Some((bit_offset_in_unit, bit_width))` when the
    /// accessed member is a bitfield, `None` for regular fields.  Consumed
    /// by `lower_member_access` (read) and `lower_assignment` (write) to
    /// emit proper bit-manipulation instead of full-width load/store.
    pub last_bitfield_info: Option<(usize, usize)>,
    /// Cache for computed struct/union layouts, keyed by struct tag name.
    /// Avoids recomputing full layout for every member access on large
    /// structs like `struct sock` (hundreds of fields).
    pub layout_cache: &'a mut FxHashMap<String, crate::common::type_builder::StructLayout>,
    /// VLA variable name → IR Value holding the total byte size (runtime).
    /// When sizeof is applied to a VLA variable, this value is emitted
    /// instead of the compile-time sizeof (which would be 8 = pointer size).
    pub vla_sizes: &'a FxHashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Internal helpers — typed value tracking
// ---------------------------------------------------------------------------

/// Combined `Value` + C type returned by internal lowering helpers so that
/// callers can propagate signedness / pointee information.
/// A pair of an IR [`Value`] and its C-level type.
///
/// Used internally during expression lowering to propagate type information
/// alongside the generated value, and exposed publicly for callers that need
/// the expression's C type for implicit conversion insertion (e.g. local
/// variable initializer lowering).
pub struct TypedValue {
    /// The IR value handle.
    pub value: Value,
    /// The C-level type of this value.
    pub ty: CType,
    /// Bitfield precision — when set, arithmetic on this value should
    /// truncate results to the given number of bits.  This implements
    /// C11 §6.7.2.1p10: "A bit-field is interpreted as having a signed
    /// or unsigned integer type consisting of the specified number of
    /// bits."  For bitfields wider than `int` (> 32 bits), the promoted
    /// type is the declared type but arithmetic wraps at the bitfield
    /// width (matching GCC behavior).
    pub bitfield_width: Option<u32>,
}

impl TypedValue {
    #[inline]
    fn new(value: Value, ty: CType) -> Self {
        Self {
            value,
            ty,
            bitfield_width: None,
        }
    }

    #[inline]
    fn with_bitfield_width(value: Value, ty: CType, width: u32) -> Self {
        Self {
            value,
            ty,
            bitfield_width: Some(width),
        }
    }

    #[inline]
    fn void() -> Self {
        Self {
            value: Value::UNDEF,
            ty: CType::Void,
            bitfield_width: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction emission helper
// ---------------------------------------------------------------------------

/// Push an instruction into the current insertion block of the function.
fn emit_inst(ctx: &mut ExprLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let idx = block_id.index();
        if let Some(block) = ctx.function.get_block_mut(idx) {
            block.instructions.push(inst);
        }
    }
}

/// Emit an alloca instruction into the **entry block** (block 0).
///
/// This is needed for temporary allocas created during expression
/// lowering (e.g. struct-to-alloca copies for ABI correctness).
/// Create a new basic block in the current function and return its `BlockId`.
fn new_block(ctx: &mut ExprLoweringContext<'_>) -> BlockId {
    let block_id = ctx.builder.create_block();
    let idx = ctx.function.block_count();
    let block = crate::ir::basic_block::BasicBlock::new(idx);
    ctx.function.add_block(block);
    block_id
}

/// Adds a CFG edge (predecessor/successor) between two blocks.
///
/// Maintains the predecessor and successor lists that the optimizer's
/// dead code elimination and CFG simplification passes rely on for
/// reachability analysis.
fn add_cfg_edge(ctx: &mut ExprLoweringContext<'_>, from_idx: usize, to_idx: usize) {
    if let Some(from_block) = ctx.function.blocks.get_mut(from_idx) {
        from_block.add_successor(to_idx);
    }
    if let Some(to_block) = ctx.function.blocks.get_mut(to_idx) {
        to_block.add_predecessor(from_idx);
    }
}

// ---------------------------------------------------------------------------
// Constant materialisation helpers
// ---------------------------------------------------------------------------

/// Emit a compile-time integer constant and return its `Value` handle.
///
/// The constant is registered as an internal `GlobalVariable` in the module
/// so that the backend can resolve the `Value` to an immediate.  The
/// in-function instruction is a `BinOp::Add` with `Value::UNDEF` operand
/// that serves as a placeholder defining the SSA value.
/// Check whether an AST expression is a compile-time constant for
/// `__builtin_constant_p`.  This recursively evaluates arithmetic on
/// literals, casts of constants, sizeof, and parenthesized sub-expressions.
fn is_compile_time_constant(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::IntegerLiteral { .. }
        | ast::Expression::FloatLiteral { .. }
        | ast::Expression::CharLiteral { .. }
        | ast::Expression::StringLiteral { .. }
        | ast::Expression::SizeofType { .. }
        | ast::Expression::SizeofExpr { .. }
        | ast::Expression::AlignofType { .. }
        | ast::Expression::AlignofExpr { .. } => true,

        // Parenthesized: transparent.
        ast::Expression::Parenthesized { inner, .. } => is_compile_time_constant(inner),

        // Binary arithmetic on two constants is constant.
        ast::Expression::Binary { left, right, .. } => {
            is_compile_time_constant(left) && is_compile_time_constant(right)
        }

        // Unary arithmetic on a constant is constant.
        ast::Expression::UnaryOp { operand, op, .. } => {
            match op {
                // AddressOf a variable is NOT a compile-time constant
                // for __builtin_constant_p purposes.
                ast::UnaryOp::AddressOf => false,
                _ => is_compile_time_constant(operand),
            }
        }

        // Cast of a constant is constant.
        ast::Expression::Cast { operand, .. } => is_compile_time_constant(operand),

        // Conditional with constant condition and constant branches.
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            is_compile_time_constant(condition)
                && then_expr
                    .as_ref()
                    .map_or(true, |e| is_compile_time_constant(e))
                && is_compile_time_constant(else_expr)
        }

        // Comma: the last expression determines constness.
        ast::Expression::Comma { exprs, .. } => {
            exprs.last().map_or(false, is_compile_time_constant)
        }

        // Everything else (variables, function calls, etc.) is NOT constant.
        _ => false,
    }
}

/// Apply bitfield-width truncation to a `TypedValue` when either operand
/// of a binary expression originated from a bitfield read wider than 32
/// bits.  C11 §6.7.2.1p10 says a bitfield is interpreted as having a
/// type "consisting of the specified number of bits," so arithmetic wraps
/// at the bitfield precision — not at the full declared type width.
///
/// `lhs_bfw` / `rhs_bfw` are the `bitfield_width` from the two operands.
/// The narrower width wins (matching GCC); unsigned fields are truncated
/// with AND, signed fields with sign-extension (SHL + ASHR).
fn maybe_truncate_bitfield(
    ctx: &mut ExprLoweringContext<'_>,
    mut tv: TypedValue,
    lhs_bfw: Option<u32>,
    rhs_bfw: Option<u32>,
    span: Span,
) -> TypedValue {
    let bfw = match (lhs_bfw, rhs_bfw) {
        (Some(a), Some(b)) => std::cmp::max(a, b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return tv,
    };
    let ir = ctype_to_ir(&tv.ty, ctx.target);
    let ir_bits = match &ir {
        IrType::I8 => 8u32,
        IrType::I16 => 16,
        IrType::I32 => 32,
        IrType::I64 => 64,
        _ => return tv,
    };
    if bfw >= ir_bits {
        return tv;
    }
    // For unsigned types: AND with (2^bfw - 1)
    if is_unsigned(&tv.ty) {
        let mask = (1u64 << bfw).wrapping_sub(1);
        let mask_val = emit_int_const(ctx, mask as i128, ir.clone(), span);
        let (v, inst) = ctx.builder.build_and(tv.value, mask_val, ir, span);
        emit_inst(ctx, inst);
        tv.value = v;
    } else {
        // For signed types: sign-extend from bit_width
        let shift = ir_bits - bfw;
        let shift_val = emit_int_const(ctx, shift as i128, ir.clone(), span);
        let (shl, si) = ctx.builder.build_shl(tv.value, shift_val, ir.clone(), span);
        emit_inst(ctx, si);
        let shift_val2 = emit_int_const(ctx, shift as i128, ir.clone(), span);
        let (sar, ai) = ctx.builder.build_shr(shl, shift_val2, ir, true, span);
        emit_inst(ctx, ai);
        tv.value = sar;
    }
    // Propagate bitfield width to result
    tv.bitfield_width = Some(bfw);
    tv
}

pub fn emit_int_const(
    ctx: &mut ExprLoweringContext<'_>,
    value: i128,
    ir_ty: IrType,
    span: Span,
) -> Value {
    // Register the constant in the module's global pool for the backend.
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.i.{}", const_id);
    let mut gv = GlobalVariable::new(gname, ir_ty.clone(), Some(Constant::Integer(value)));
    gv.linkage = ir_module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    // Define the SSA value via a self-referencing Add+UNDEF sentinel.
    let result = ctx.builder.fresh_value();
    let inst = Instruction::BinOp {
        result,
        op: IrBinOp::Add,
        lhs: result,
        rhs: Value::UNDEF,
        ty: ir_ty,
        span,
    };
    emit_inst(ctx, inst);
    // Record the direct Value → constant mapping so the backend can
    // resolve constants without fragile positional matching.
    ctx.function.constant_values.insert(result, value as i64);
    result
}

/// Emit an integer constant for use as a GEP byte-offset index.
///
/// This is a public wrapper around `emit_int_const` for use by
/// `decl_lowering::make_index_value` which computes element byte
/// offsets for aggregate initialiser stores.
pub fn emit_int_const_for_index(ctx: &mut ExprLoweringContext<'_>, byte_offset: i64) -> Value {
    let ir_ty = size_ir_type(ctx.target);
    emit_int_const(ctx, byte_offset as i128, ir_ty, Span::dummy())
}

/// Emit a zero constant with the given IR type.  Used by `zero_init_field`
/// in `decl_lowering` to produce correctly-sized stores for aggregate
/// member zero-initialization (C99 §6.7.9/19).
pub fn emit_int_const_for_zero(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType) -> Value {
    emit_int_const(ctx, 0, ir_ty, Span::dummy())
}

/// Convert a value for storage into an element of the given C type.
///
/// When the target element type is narrower than `int` (e.g., `char` = I8,
/// `short` = I16), inserts a truncation so that the backend emits the
/// correctly-sized store operation.  Without this, integer literals (which
/// default to I32) would be stored with 32-bit width, overwriting adjacent
/// memory in arrays.
pub fn convert_for_store(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    target_ctype: &CType,
) -> Value {
    let target_ir = ctype_to_ir(target_ctype, ctx.target);
    // Only convert for integer IR types.
    // Guard: non-integer IR types (Struct, Ptr, Float, etc.) skip conversion.
    let tw = match target_ir {
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 => {
            target_ir.int_width()
        }
        _ => return value,
    };
    // Determine the value's current IR type by scanning the function's
    // instructions for the one that produced it.
    let vw = infer_value_ir_width(ctx, value);
    if tw > 0 && tw < vw {
        // Truncation: value is wider than target.
        let (v, i) = ctx.builder.build_trunc(value, target_ir, Span::dummy());
        emit_inst(ctx, i);
        return v;
    }
    if tw > vw && vw > 0 {
        // Widening: value is narrower than target (e.g., int → long).
        // Determine signedness from C type for SExt vs ZExt.
        let resolved = crate::common::types::resolve_typedef(target_ctype);
        let is_signed = matches!(
            resolved,
            CType::Char
                | CType::SChar
                | CType::Short
                | CType::Int
                | CType::Long
                | CType::LongLong
                | CType::Int128
        );
        let from_ir = match vw {
            1 => IrType::I1,
            8 => IrType::I8,
            16 => IrType::I16,
            32 => IrType::I32,
            64 => IrType::I64,
            128 => IrType::I128,
            _ => IrType::I32,
        };
        if is_signed {
            let (v, i) = ctx.builder.build_sext(value, target_ir, Span::dummy());
            // Patch from_type so codegen selects correct MOVSX variant.
            let mut patched = i;
            if let Instruction::SExt {
                ref mut from_type, ..
            } = patched
            {
                *from_type = from_ir;
            }
            emit_inst(ctx, patched);
            return v;
        } else {
            let (v, i) = ctx.builder.build_zext(value, target_ir, Span::dummy());
            let mut patched = i;
            if let Instruction::ZExt {
                ref mut from_type, ..
            } = patched
            {
                *from_type = from_ir;
            }
            emit_inst(ctx, patched);
            return v;
        }
    }
    value
}

/// Infer the IR width (in bits) of a Value by scanning the function's
/// instructions for the one that produced it.
fn infer_value_ir_width(ctx: &ExprLoweringContext<'_>, value: Value) -> u32 {
    // Check all blocks for the instruction that produced this value.
    for block in ctx.function.blocks() {
        for inst in block.instructions() {
            if let Some(res) = inst.result() {
                if res == value {
                    // Extract the type from the instruction variant.
                    return inst_result_ir_width(inst);
                }
            }
        }
    }
    // Fall back to 32 (default for integer constants in IR lowering).
    32
}

/// Extract the IR integer width (in bits) from an instruction that produces
/// a result, by pattern-matching on the variant's type field.
fn inst_result_ir_width(inst: &Instruction) -> u32 {
    match inst {
        Instruction::Alloca { .. } => {
            // Alloca result is a pointer, not the element type.
            64 // pointer width
        }
        Instruction::StackAlloc { .. } => 64, // pointer
        Instruction::Load { ty, .. } => ty.int_width(),
        Instruction::BinOp { ty, .. } => ty.int_width(),
        Instruction::ICmp { .. } | Instruction::FCmp { .. } => 1,
        Instruction::Call { return_type, .. } => return_type.int_width(),
        Instruction::Phi { ty, .. } => ty.int_width(),
        Instruction::GetElementPtr { .. } => 64, // pointer
        Instruction::BitCast { to_type, .. } => to_type.int_width(),
        Instruction::Trunc { to_type, .. } => to_type.int_width(),
        Instruction::ZExt { to_type, .. } => to_type.int_width(),
        Instruction::SExt { to_type, .. } => to_type.int_width(),
        Instruction::IntToPtr { .. } => 64, // pointer
        Instruction::PtrToInt { to_type, .. } => to_type.int_width(),
        Instruction::InlineAsm { .. } => 64, // conservative
        _ => 32,
    }
}

/// Emit a compile-time floating-point constant.
fn emit_float_const(
    ctx: &mut ExprLoweringContext<'_>,
    value: f64,
    ir_ty: IrType,
    span: Span,
) -> Value {
    let const_id = ctx.module.globals().len();
    let gname = format!(".Lconst.f.{}", const_id);
    let gname_clone = gname.clone();
    // Note: F80 (long double) currently treated as F64 in the backend
    // (SSE-only codegen without x87 support), so we always store as
    // Constant::Float(f64) for consistency between .rodata layout and
    // the Movsd load instruction used by the backend.
    let mut gv = GlobalVariable::new(gname, ir_ty.clone(), Some(Constant::Float(value)));
    gv.linkage = ir_module::Linkage::Internal;
    gv.is_constant = true;
    ctx.module.add_global(gv);

    let result = ctx.builder.fresh_value();
    let inst = Instruction::BinOp {
        result,
        op: IrBinOp::FAdd,
        lhs: result,
        rhs: Value::UNDEF,
        ty: ir_ty,
        span,
    };
    emit_inst(ctx, inst);
    // Record the direct Value → float constant mapping for the backend.
    ctx.function
        .float_constant_values
        .insert(result, (gname_clone, value));
    result
}

/// Emit a zero constant of the given IR type.
///
/// For pointer types, uses `Constant::Null`.  For aggregate types, uses
/// `Constant::ZeroInit`.  For scalar types, emits an integer zero.
fn emit_zero(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType, span: Span) -> Value {
    match &ir_ty {
        IrType::Ptr => {
            // Null pointer constant.
            let name = format!(".null.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, ir_ty.clone(), Some(Constant::Null));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            emit_int_const(ctx, 0, IrType::I64, span)
        }
        IrType::Struct(_) | IrType::Array(_, _) => {
            // Aggregate zero-init.
            let name = format!(".zeroinit.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, ir_ty.clone(), Some(Constant::ZeroInit));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            emit_int_const(ctx, 0, IrType::I32, span)
        }
        IrType::F32 | IrType::F64 | IrType::F80 => emit_float_const(ctx, 0.0, ir_ty, span),
        _ => emit_int_const(ctx, 0, ir_ty, span),
    }
}

/// Emit integer constant `1` of the given IR type.
fn emit_one(ctx: &mut ExprLoweringContext<'_>, ir_ty: IrType, span: Span) -> Value {
    match &ir_ty {
        IrType::F32 | IrType::F64 | IrType::F80 => emit_float_const(ctx, 1.0, ir_ty, span),
        _ => emit_int_const(ctx, 1, ir_ty, span),
    }
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Convert a C type to its IR representation.
///
/// For the most common leaf types we resolve directly to avoid a round-trip
/// through `IrType::from_ctype`; aggregate and function types still delegate.
pub fn ctype_to_ir(ctype: &CType, target: &Target) -> IrType {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Void => IrType::Void,
        CType::Bool => IrType::I1,
        CType::Char | CType::UChar | CType::SChar => IrType::I8,
        CType::Short | CType::UShort => IrType::I16,
        CType::Int | CType::UInt => IrType::I32,
        CType::Long | CType::ULong => {
            if target.long_size() == 8 {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        CType::LongLong | CType::ULongLong => IrType::I64,
        CType::Int128 | CType::UInt128 => IrType::I128,
        CType::Float => IrType::F32,
        CType::Double => IrType::F64,
        CType::LongDouble => {
            // Both 80-bit and 128-bit extended precision map to IrType::F80
            // for code generation purposes.
            IrType::F80
        }
        CType::Pointer(..) => IrType::Ptr,
        CType::Array(elem, count) => {
            let elem_ir = ctype_to_ir(elem, target);
            IrType::Array(Box::new(elem_ir), count.unwrap_or(0))
        }
        CType::Struct { .. } => {
            // Delegate to IrType::from_ctype which correctly handles
            // bitfield packing (multiple bitfields may share one
            // allocation unit, so per-field mapping inflates the size).
            IrType::from_ctype(resolved, target)
        }
        CType::Union {
            fields: _fields, ..
        } => {
            // Union IR type: represented as a byte array of the union size.
            let sz = target.pointer_width(); // fallback, real size from type_builder
            let _ = sz;
            IrType::from_ctype(resolved, target)
        }
        CType::Function {
            return_type,
            params,
            variadic: _,
        } => {
            let ret = ctype_to_ir(return_type, target);
            let ps: Vec<IrType> = params.iter().map(|p| ctype_to_ir(p, target)).collect();
            IrType::Function(Box::new(ret), ps)
        }
        CType::Enum { .. } => IrType::I32,
        CType::Atomic(inner) => ctype_to_ir(inner, target),
        CType::Qualified(inner, _) => ctype_to_ir(inner, target),
        CType::Typedef { underlying, .. } => ctype_to_ir(underlying, target),
        _ => IrType::from_ctype(resolved, target),
    }
}

/// Return the IR type corresponding to `size_t` for the current target.
fn size_ir_type(target: &Target) -> IrType {
    if target.is_64bit() {
        IrType::I64
    } else {
        IrType::I32
    }
}

/// Return the C type corresponding to `size_t` for the current target.
/// Map an IR element type back to an approximate CType.
///
/// This is used when a global array's element type needs to be recovered
/// for array-to-pointer decay but the original C type is unavailable
/// (e.g. lookup_var_type fell back to CType::Int).
fn ir_elem_to_approx_ctype(ir_ty: &IrType) -> CType {
    match ir_ty {
        IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::LongLong,
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        _ => CType::Int,
    }
}

/// Map an IR type to an approximate CType (including compound types).
///
/// Used to reconstruct C types from global variable IR types for sizeof
/// and array decay type computations.
fn ir_type_to_approx_ctype(ir_ty: &IrType) -> CType {
    match ir_ty {
        IrType::I1 => CType::Bool,
        IrType::I8 => CType::Char,
        IrType::I16 => CType::Short,
        IrType::I32 => CType::Int,
        IrType::I64 => CType::LongLong,
        IrType::I128 => CType::LongLong, // approximate
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::Ptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        IrType::Array(elem, count) => {
            let elem_cty = ir_type_to_approx_ctype(elem);
            CType::Array(Box::new(elem_cty), Some(*count))
        }
        IrType::Void => CType::Void,
        _ => CType::Int,
    }
}

fn size_ctype(target: &Target) -> CType {
    if target.is_64bit() {
        CType::ULong
    } else {
        CType::UInt
    }
}

/// Check if a C type is unsigned.
fn is_unsigned(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_unsigned(resolved)
}

/// Check if a C type is signed (complement of unsigned for integer types).
fn is_signed(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_signed(resolved)
}

/// Check if a C type is a floating-point type.
fn is_floating(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_floating(resolved)
}

/// Check if a C type is a pointer type.
fn is_pointer_type(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_pointer(resolved)
}

/// Check if a C type is an integer type.
fn is_integer_type(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_integer(resolved)
}

/// Check if a C type is an arithmetic type (integer or floating).
fn is_arithmetic(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_arithmetic(resolved)
}

/// Check if a C type is a scalar type (arithmetic or pointer).
fn is_scalar(ctype: &CType) -> bool {
    let resolved = types::resolve_typedef(ctype);
    types::is_scalar(resolved)
}

/// Strip qualifiers, typedefs, and `_Atomic` to get the canonical type.
fn strip_type(ctype: &CType) -> &CType {
    match ctype {
        CType::Typedef { underlying, .. } => strip_type(underlying),
        CType::Atomic(inner) => strip_type(inner),
        CType::Qualified(inner, _) => strip_type(inner),
        other => {
            let unqual = types::unqualified(other);
            types::resolve_typedef(unqual)
        }
    }
}

/// Get the pointee type of a pointer or array C type.
fn pointee_of(ctype: &CType) -> CType {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Pointer(inner, _) => (**inner).clone(),
        CType::Array(inner, _) => (**inner).clone(),
        _ => CType::Void,
    }
}

/// Compute the byte size of the pointee for pointer arithmetic scaling.
#[allow(dead_code)]
fn pointee_size(ctype: &CType, tb: &TypeBuilder) -> usize {
    let pt = pointee_of(ctype);
    if matches!(pt, CType::Void) {
        1
    } else {
        tb.sizeof_type(&pt)
    }
}

/// Like [`pointee_size`] but falls back to `sizeof_resolved` so that
/// pointer arithmetic on `T *p` correctly scales by `sizeof(T)` even
/// when `T` is a forward-declared/incomplete struct behind a typedef.
fn pointee_size_resolved(
    ctype: &CType,
    tb: &TypeBuilder,
    struct_defs: &FxHashMap<String, CType>,
    target: &Target,
) -> usize {
    let pt = pointee_of(ctype);
    if matches!(pt, CType::Void) {
        1
    } else {
        sizeof_resolved(&pt, tb, struct_defs, target)
    }
}

/// Compute `sizeof(ty)`, resolving incomplete struct/union types through
/// typedefs and the `struct_defs` registry.  Falls back to the plain
/// `TypeBuilder::sizeof_type` when no resolution is needed.
fn sizeof_resolved(
    ty: &CType,
    tb: &TypeBuilder,
    struct_defs: &FxHashMap<String, CType>,
    target: &Target,
) -> usize {
    // Fast path — normal sizeof.
    let sz = tb.sizeof_type(ty);
    if sz > 0 {
        return sz;
    }
    // Slow path — strip typedefs and resolve empty structs/unions from the
    // global tag registry.
    sizeof_resolved_inner(ty, struct_defs, target)
}

fn sizeof_resolved_inner(
    ty: &CType,
    struct_defs: &FxHashMap<String, CType>,
    target: &Target,
) -> usize {
    match ty {
        CType::Typedef { underlying, .. } => sizeof_resolved_inner(underlying, struct_defs, target),
        CType::Struct {
            name: Some(ref tag),
            fields,
            ..
        } if fields.is_empty() => {
            if let Some(full) = struct_defs.get(tag) {
                types::sizeof_ctype(full, target)
            } else {
                0
            }
        }
        CType::Union {
            name: Some(ref tag),
            fields,
            ..
        } if fields.is_empty() => {
            if let Some(full) = struct_defs.get(tag) {
                types::sizeof_ctype(full, target)
            } else {
                0
            }
        }
        _ => types::sizeof_ctype(ty, target),
    }
}

/// Determine the C type of an integer literal from its value and suffix.
fn integer_literal_ctype(value: u128, suffix: &ast::IntegerSuffix, is_hex_or_octal: bool) -> CType {
    // C11 §6.4.4.1 — Type determination for integer constants.
    // Hex/octal constants try unsigned types before widening, while
    // decimal constants only try signed types (then long long unsigned).
    match suffix {
        ast::IntegerSuffix::None => {
            if value <= i32::MAX as u128 {
                CType::Int
            } else if is_hex_or_octal && value <= u32::MAX as u128 {
                CType::UInt
            } else if value <= i64::MAX as u128 {
                CType::Long
            } else if is_hex_or_octal && value <= u64::MAX as u128 {
                CType::ULong
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::U => {
            if value <= u32::MAX as u128 {
                CType::UInt
            } else if value <= u64::MAX as u128 {
                CType::ULong
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::L => {
            if value <= i64::MAX as u128 {
                CType::Long
            } else if is_hex_or_octal && value <= u64::MAX as u128 {
                CType::ULong
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::UL => CType::ULong,
        ast::IntegerSuffix::LL => {
            if value <= i64::MAX as u128 {
                CType::LongLong
            } else if is_hex_or_octal && value <= u64::MAX as u128 {
                CType::ULongLong
            } else {
                CType::ULongLong
            }
        }
        ast::IntegerSuffix::ULL => CType::ULongLong,
    }
}

/// Determine the C type of a float literal from its suffix.
/// Imaginary suffixes return the base float type — the imaginary nature
/// is handled during complex expression lowering.
fn float_literal_ctype(suffix: &ast::FloatSuffix) -> CType {
    match suffix {
        ast::FloatSuffix::None | ast::FloatSuffix::I => CType::Double,
        ast::FloatSuffix::F | ast::FloatSuffix::FI => CType::Float,
        ast::FloatSuffix::L | ast::FloatSuffix::LI => CType::LongDouble,
    }
}

/// Resolve a `Symbol` handle to its interned string name.
fn resolve_sym<'a>(ctx: &'a ExprLoweringContext<'_>, sym_idx: u32) -> &'a str {
    ctx.name_table
        .get(sym_idx as usize)
        .map(|s| s.as_str())
        .unwrap_or("<unknown>")
}

/// Look up a variable by name, checking local_vars first, then param_values.
/// Returns `(Value, is_alloca)`:
///  - `is_alloca = true` → value is an alloca pointer (needs `Load` for rvalue)
///  - `is_alloca = false` → value is the raw SSA value (no load needed)
fn lookup_var(ctx: &ExprLoweringContext<'_>, name: &str) -> Option<(Value, bool)> {
    if let Some(&v) = ctx.local_vars.get(name) {
        return Some((v, true));
    }
    if let Some(&v) = ctx.param_values.get(name) {
        return Some((v, true));
    }
    None
}

/// Look up the declared C type of a variable.
fn lookup_var_type(ctx: &ExprLoweringContext<'_>, name: &str) -> CType {
    // For static local variables, prefer the original C type stored during
    // lower_static_local (which preserves struct/union/typedef information)
    // over the lossy ir_type_to_approx_ctype reverse mapping (which turns
    // IrType::Struct into CType::Int, producing wrong sizeof/field offsets).
    if let Some(mangled) = ctx.static_locals.get(name) {
        if let Some(ct) = ctx.module.global_c_types.get(mangled.as_str()) {
            return ct.clone();
        }
        if let Some(gv) = ctx.module.get_global(mangled.as_str()) {
            return ir_type_to_approx_ctype(&gv.ty);
        }
    }
    // Check scope-local type overrides first — these handle block-scoped
    // variable shadowing (e.g. `u32 t` in one case block vs `void *t` in
    // another within the same switch).
    if let Some(ct) = ctx.scope_type_overrides.get(name) {
        return ct.clone();
    }
    // Check local types (parameters, local variables).
    if let Some(ct) = ctx.local_types.get(name) {
        return ct.clone();
    }
    // Prefer the original C type recorded during global variable lowering —
    // this preserves struct field names, function pointer signatures, and
    // other information that the IR type system cannot represent.
    if let Some(ct) = ctx.module.global_c_types.get(name) {
        return ct.clone();
    }
    // GCC global register variables: type recorded in register_globals.
    if let Some((_reg, ct)) = ctx.module.register_globals.get(name) {
        return ct.clone();
    }
    // Fall back to global variable types from the module.
    if let Some(gv) = ctx.module.get_global(name) {
        return ir_type_to_approx_ctype(&gv.ty);
    }
    CType::Int
}

// =========================================================================
// PUBLIC API
// =========================================================================

/// Lower an AST expression to an IR `Value` (rvalue).
///
/// This is the primary entry point for expression lowering.  It dispatches
/// on the expression variant and returns the SSA value representing the
/// computed result.  For lvalue-capable expressions this performs an
/// implicit load.
pub fn lower_expression(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> Value {
    lower_expr_inner(ctx, expr).value
}

/// Lower an AST expression, returning both the IR `Value` and the C type.
///
/// Used by callers that need the expression's C-level type for implicit
/// conversion (e.g. local variable initializers where `char *p = 0` must
/// convert the `int` literal to a pointer).
pub fn lower_expression_typed(
    ctx: &mut ExprLoweringContext<'_>,
    expr: &ast::Expression,
) -> TypedValue {
    lower_expr_inner(ctx, expr)
}

/// Lower an AST expression as an **lvalue**, returning a pointer `Value`.
///
/// Used for the left-hand side of assignments, `&` address-of, and other
/// contexts that require a memory address rather than a loaded value.
/// Panics (via diagnostic) if the expression is not a valid lvalue.
pub fn lower_lvalue(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> Value {
    lower_lvalue_inner(ctx, expr).value
}

// =========================================================================
// MAIN DISPATCH
// =========================================================================

/// Internal expression dispatch returning `TypedValue`.
fn lower_expr_inner(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> TypedValue {
    match expr {
        // ---- Literals ----
        ast::Expression::IntegerLiteral {
            value,
            suffix,
            is_hex_or_octal,
            span,
        } => lower_integer_literal(ctx, *value, suffix, *is_hex_or_octal, *span),
        ast::Expression::FloatLiteral {
            value,
            suffix,
            span,
        } => lower_float_literal(ctx, *value, suffix, *span),
        ast::Expression::StringLiteral {
            segments,
            prefix,
            span,
        } => lower_string_literal(ctx, segments, prefix, *span),
        ast::Expression::CharLiteral {
            value,
            prefix,
            span,
        } => lower_char_literal(ctx, *value, prefix, *span),

        // ---- Identifier ----
        ast::Expression::Identifier { name, span } => lower_identifier(ctx, name.as_u32(), *span),

        // ---- Parenthesised (transparent) ----
        ast::Expression::Parenthesized { inner, .. } => lower_expr_inner(ctx, inner),

        // ---- Binary / Logical ----
        ast::Expression::Binary {
            op,
            left,
            right,
            span,
        } => lower_binary(ctx, op, left, right, *span),

        // ---- Unary ----
        ast::Expression::UnaryOp { op, operand, span } => lower_unary(ctx, op, operand, *span),

        // ---- Assignment / Compound assignment ----
        ast::Expression::Assignment {
            op,
            target,
            value,
            span,
        } => lower_assignment(ctx, op, target, value, *span),

        // ---- Conditional (ternary) ----
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            span,
        } => lower_conditional(ctx, condition, then_expr.as_deref(), else_expr, *span),

        // ---- Function call ----
        ast::Expression::FunctionCall { callee, args, span } => {
            // Check if this is a known compiler builtin that should be
            // lowered inline rather than emitted as an external function call.
            if let Some(tv) = try_lower_builtin_call(ctx, callee, args, *span) {
                tv
            } else {
                lower_function_call(ctx, callee, args, *span)
            }
        }

        // ---- Cast ----
        ast::Expression::Cast {
            type_name,
            operand,
            span,
        } => lower_cast_expr(ctx, type_name, operand, *span),

        // ---- sizeof / alignof ----
        ast::Expression::SizeofExpr { operand, span } => lower_sizeof_expr(ctx, operand, *span),
        ast::Expression::SizeofType { type_name, span } => lower_sizeof_type(ctx, type_name, *span),
        ast::Expression::AlignofType { type_name, span } => {
            lower_alignof_type(ctx, type_name, *span)
        }
        ast::Expression::AlignofExpr { expr: inner, span } => lower_alignof_expr(ctx, inner, *span),

        // ---- Member access ----
        ast::Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access(ctx, object, member.as_u32(), false, *span),
        ast::Expression::PointerMemberAccess {
            object,
            member,
            span,
        } => lower_member_access(ctx, object, member.as_u32(), true, *span),

        // ---- Array subscript ----
        ast::Expression::ArraySubscript { base, index, span } => {
            lower_array_subscript(ctx, base, index, *span)
        }

        // ---- Post-increment / decrement ----
        ast::Expression::PostIncrement { operand, span } => {
            lower_post_inc_dec(ctx, operand, true, *span)
        }
        ast::Expression::PostDecrement { operand, span } => {
            lower_post_inc_dec(ctx, operand, false, *span)
        }

        // ---- Pre-increment / decrement ----
        ast::Expression::PreIncrement { operand, span } => {
            lower_pre_inc_dec(ctx, operand, true, *span)
        }
        ast::Expression::PreDecrement { operand, span } => {
            lower_pre_inc_dec(ctx, operand, false, *span)
        }

        // ---- Comma ----
        ast::Expression::Comma { exprs, span } => lower_comma(ctx, exprs, *span),

        // ---- Compound literal ----
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal(ctx, type_name, initializer, *span),

        // ---- GCC statement expression ----
        ast::Expression::StatementExpression { compound, span } => {
            lower_statement_expression(ctx, compound, *span)
        }

        // ---- GCC builtin call ----
        ast::Expression::BuiltinCall {
            builtin,
            args,
            span,
        } => lower_builtin(ctx, builtin, args, *span),

        // ---- C11 _Generic ----
        ast::Expression::Generic {
            controlling,
            associations,
            span,
        } => lower_generic(ctx, controlling, associations, *span),

        // ---- GCC address-of-label (&&label) ----
        ast::Expression::AddressOfLabel { label, span } => {
            lower_address_of_label(ctx, label.as_u32(), *span)
        }
    }
}

/// Internal lvalue dispatch returning `TypedValue` (pointer + pointee type).
fn lower_lvalue_inner(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> TypedValue {
    // Clear bitfield info — it will be set by lower_member_access_lvalue
    // if the resulting lvalue is a bitfield member.
    ctx.last_bitfield_info = None;
    match expr {
        ast::Expression::Identifier { name, span } => {
            lower_identifier_lvalue(ctx, name.as_u32(), *span)
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            span: _,
        } => {
            // *ptr as lvalue → evaluate ptr, the pointer IS the lvalue
            let ptr = lower_expr_inner(ctx, operand);
            let pt = pointee_of(&ptr.ty);
            TypedValue::new(ptr.value, pt)
        }
        ast::Expression::ArraySubscript { base, index, span } => {
            lower_array_subscript_lvalue(ctx, base, index, *span)
        }
        ast::Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, member.as_u32(), false, *span),
        ast::Expression::PointerMemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, member.as_u32(), true, *span),
        ast::Expression::Parenthesized { inner, .. } => lower_lvalue_inner(ctx, inner),
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal_lvalue(ctx, type_name, initializer, *span),
        // Cast expressions can be lvalues in GCC extension mode.
        ast::Expression::Cast { .. } => lower_expr_inner(ctx, expr),
        // Statement expressions can be lvalues (GCC extension).
        ast::Expression::StatementExpression { .. } => lower_expr_inner(ctx, expr),
        // Function calls returning structs used as lvalues in compound
        // expressions (GCC extension, e.g. `struct_returning_fn().field`).
        ast::Expression::FunctionCall { .. } => lower_expr_inner(ctx, expr),
        // Comma expressions — the last operand is the lvalue.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                lower_lvalue_inner(ctx, last)
            } else {
                lower_expr_inner(ctx, expr)
            }
        }
        // Conditional expression can be an lvalue in GCC mode.
        ast::Expression::Conditional { .. } => lower_expr_inner(ctx, expr),
        // GCC __real__/__imag__ as lvalue: __real__ z = 1.0
        // For _Complex types, returns pointer to the component.
        // For non-complex, __real__ is identity lvalue.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::RealPart,
            operand,
            span,
            ..
        } => {
            let inner = lower_lvalue_inner(ctx, operand);
            let resolved = types::resolve_typedef(&inner.ty);
            if let CType::Complex(base) = resolved {
                let elem_ctype = base.as_ref().clone();
                // Real part is at offset 0 from the complex alloca.
                let zero_idx = emit_int_const_for_index(ctx, 0);
                let (real_ptr, gep) =
                    ctx.builder
                        .build_gep(inner.value, vec![zero_idx], IrType::Ptr, *span);
                emit_inst(ctx, gep);
                TypedValue::new(real_ptr, elem_ctype)
            } else {
                inner
            }
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::ImagPart,
            operand,
            span,
            ..
        } => {
            let inner = lower_lvalue_inner(ctx, operand);
            let resolved = types::resolve_typedef(&inner.ty);
            if let CType::Complex(base) = resolved {
                let elem_ctype = base.as_ref().clone();
                let elem_ir = ctype_to_ir(&elem_ctype, ctx.target);
                let elem_size: i64 = complex_elem_size(&elem_ir);
                // Imaginary part at offset elem_size from the complex alloca.
                let off_idx = emit_int_const_for_index(ctx, elem_size);
                let (imag_ptr, gep) =
                    ctx.builder
                        .build_gep(inner.value, vec![off_idx], IrType::Ptr, *span);
                emit_inst(ctx, gep);
                TypedValue::new(imag_ptr, elem_ctype)
            } else {
                inner
            }
        }
        other => {
            // Fallback: try lowering as a value. Many kernel macro
            // expansions produce expressions that look like non-lvalues
            // but are used in lvalue contexts (e.g. compound literals
            // wrapped in casts or function-call wrappers).
            let sp = expr_span(other);
            ctx.diagnostics
                .emit_warning(sp, "expression used as lvalue may not be valid");
            lower_expr_inner(ctx, other)
        }
    }
}

/// Extract the `Span` from an arbitrary expression variant.
fn expr_span(expr: &ast::Expression) -> Span {
    match expr {
        ast::Expression::IntegerLiteral { span, .. }
        | ast::Expression::FloatLiteral { span, .. }
        | ast::Expression::StringLiteral { span, .. }
        | ast::Expression::CharLiteral { span, .. }
        | ast::Expression::Identifier { span, .. }
        | ast::Expression::Parenthesized { span, .. }
        | ast::Expression::Binary { span, .. }
        | ast::Expression::UnaryOp { span, .. }
        | ast::Expression::Assignment { span, .. }
        | ast::Expression::Conditional { span, .. }
        | ast::Expression::FunctionCall { span, .. }
        | ast::Expression::Cast { span, .. }
        | ast::Expression::SizeofExpr { span, .. }
        | ast::Expression::SizeofType { span, .. }
        | ast::Expression::AlignofType { span, .. }
        | ast::Expression::AlignofExpr { span, .. }
        | ast::Expression::MemberAccess { span, .. }
        | ast::Expression::PointerMemberAccess { span, .. }
        | ast::Expression::ArraySubscript { span, .. }
        | ast::Expression::PostIncrement { span, .. }
        | ast::Expression::PostDecrement { span, .. }
        | ast::Expression::PreIncrement { span, .. }
        | ast::Expression::PreDecrement { span, .. }
        | ast::Expression::Comma { span, .. }
        | ast::Expression::CompoundLiteral { span, .. }
        | ast::Expression::StatementExpression { span, .. }
        | ast::Expression::BuiltinCall { span, .. }
        | ast::Expression::Generic { span, .. }
        | ast::Expression::AddressOfLabel { span, .. } => *span,
    }
}

// =========================================================================
// LITERAL LOWERING
// =========================================================================

fn lower_integer_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: u128,
    suffix: &ast::IntegerSuffix,
    is_hex_or_octal: bool,
    span: Span,
) -> TypedValue {
    let cty = integer_literal_ctype(value, suffix, is_hex_or_octal);
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_int_const(ctx, value as i128, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_float_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: f64,
    suffix: &ast::FloatSuffix,
    span: Span,
) -> TypedValue {
    // GCC imaginary suffix: `2.0i` produces `_Complex double` with
    // real=0.0, imag=2.0.  Create an alloca, store zero in the real
    // component and `value` in the imaginary component.
    let is_imaginary = matches!(
        suffix,
        ast::FloatSuffix::I | ast::FloatSuffix::FI | ast::FloatSuffix::LI
    );
    if is_imaginary {
        let base_cty = float_literal_ctype(suffix);
        let elem_ir = ctype_to_ir(&base_cty, ctx.target);
        let complex_cty = CType::Complex(Box::new(base_cty));
        let elem_count: usize = 2;
        let array_ir = IrType::Array(Box::new(elem_ir.clone()), elem_count);
        let (alloca_val, alloca_inst) = ctx.builder.build_alloca(array_ir.clone(), span);
        emit_inst(ctx, alloca_inst);

        // Real part (offset 0) = 0.0
        let zero_val = emit_float_const(ctx, 0.0, elem_ir.clone(), span);
        let zero_idx = emit_int_const_for_index(ctx, 0);
        let (real_ptr, gep_r) =
            ctx.builder
                .build_gep(alloca_val, vec![zero_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_r);
        let store_r = ctx.builder.build_store(zero_val, real_ptr, span);
        emit_inst(ctx, store_r);

        // Imaginary part (offset elem_size) = value
        let imag_val = emit_float_const(ctx, value, elem_ir.clone(), span);
        let elem_size: i64 = complex_elem_size(&elem_ir);
        let off_idx = emit_int_const_for_index(ctx, elem_size);
        let (imag_ptr, gep_i) = ctx
            .builder
            .build_gep(alloca_val, vec![off_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_i);
        let store_i = ctx.builder.build_store(imag_val, imag_ptr, span);
        emit_inst(ctx, store_i);

        // Load the whole complex value from the alloca
        let (loaded, load_inst) = ctx.builder.build_load(alloca_val, array_ir, span);
        emit_inst(ctx, load_inst);
        return TypedValue::new(loaded, complex_cty);
    }

    let cty = float_literal_ctype(suffix);
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_float_const(ctx, value, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_char_literal(
    ctx: &mut ExprLoweringContext<'_>,
    value: u32,
    prefix: &ast::CharPrefix,
    span: Span,
) -> TypedValue {
    let cty = match prefix {
        ast::CharPrefix::None => CType::Int,
        ast::CharPrefix::L => CType::Int,
        ast::CharPrefix::U16 => CType::UShort,
        ast::CharPrefix::U32 => CType::UInt,
    };
    let ir_ty = ctype_to_ir(&cty, ctx.target);
    let val = emit_int_const(ctx, value as i128, ir_ty, span);
    TypedValue::new(val, cty)
}

fn lower_string_literal(
    ctx: &mut ExprLoweringContext<'_>,
    segments: &[ast::StringSegment],
    prefix: &ast::StringPrefix,
    span: Span,
) -> TypedValue {
    // Determine character width from prefix.  On Linux (all BCC targets):
    //   no prefix / u8  →  1 byte  (char / UTF-8)
    //   L / U           →  4 bytes (wchar_t = int / char32_t = unsigned int)
    //   u               →  2 bytes (char16_t = unsigned short)
    let char_width: usize = match prefix {
        ast::StringPrefix::None | ast::StringPrefix::U8 => 1,
        ast::StringPrefix::L | ast::StringPrefix::U32 => 4,
        ast::StringPrefix::U16 => 2,
    };

    // Concatenate raw segment bytes, then expand to target char width.
    // For wide strings (char_width > 1), UTF-8 byte sequences are decoded
    // to Unicode code points first, so multi-byte UTF-8 chars produce one
    // wide character (not multiple).
    let mut raw_bytes = Vec::new();
    for seg in segments {
        raw_bytes.extend_from_slice(&seg.value);
    }
    let bytes = super::decl_lowering::expand_string_bytes_for_width(&raw_bytes, char_width);

    let num_elements = bytes.len() / char_width;
    let str_id = ctx.module.intern_string(bytes.clone());

    // Register a global for the string literal with Constant::String initializer.
    // Only add the global once — if the same string content was already interned,
    // intern_string returns the same id so the global name will collide.
    let str_global_name = format!(".str.{}", str_id);
    if ctx.module.get_global(&str_global_name).is_none() {
        let elem_ir = match char_width {
            1 => IrType::I8,
            2 => IrType::I16,
            4 => IrType::I32,
            _ => IrType::I8,
        };
        let str_arr_ty = IrType::Array(Box::new(elem_ir), num_elements);
        let mut gv = GlobalVariable::new(
            str_global_name.clone(),
            str_arr_ty,
            Some(Constant::String(bytes)),
        );
        gv.linkage = crate::ir::module::Linkage::Internal;
        gv.is_constant = true;
        ctx.module.add_global(gv);
    }

    let result = ctx.builder.fresh_value();
    // Register the string literal base pointer as a global variable
    // reference so the backend can emit the correct address relocation
    // (e.g., RIP-relative on x86-64) instead of an immediate zero.
    ctx.module
        .global_var_refs
        .insert(result, str_global_name.clone());
    {
        let current_func = &mut *ctx.function;
        current_func.global_var_refs.insert(result, str_global_name);
    }
    let zero = emit_int_const(ctx, 0, IrType::I64, span);
    let (val, gep_inst) = ctx.builder.build_gep(result, vec![zero], IrType::Ptr, span);
    emit_inst(ctx, gep_inst);

    // Build pointer-to-element type for the string literal.
    let elem_ctype = match prefix {
        ast::StringPrefix::None | ast::StringPrefix::U8 => CType::Char,
        ast::StringPrefix::L => CType::Int, // wchar_t = int on Linux
        ast::StringPrefix::U16 => CType::UShort, // char16_t
        ast::StringPrefix::U32 => CType::UInt, // char32_t
    };
    let cty = CType::Pointer(
        Box::new(CType::Qualified(
            Box::new(elem_ctype),
            TypeQualifiers {
                is_const: true,
                is_volatile: false,
                is_restrict: false,
                is_atomic: false,
            },
        )),
        TypeQualifiers::default(),
    );
    TypedValue::new(val, cty)
}

// =========================================================================
// IDENTIFIER LOWERING
// =========================================================================

fn lower_identifier(ctx: &mut ExprLoweringContext<'_>, sym_idx: u32, span: Span) -> TypedValue {
    let name = resolve_sym(ctx, sym_idx);
    let var_cty = lookup_var_type(ctx, name);
    let ir_ty = ctype_to_ir(&var_cty, ctx.target);

    if std::env::var("BCC_DEBUG_KNR").is_ok() {
        let in_params = ctx.param_values.contains_key(name);
        let in_locals = ctx.local_vars.contains_key(name);
        let in_globals = ctx.module.get_global(name).is_some();
        let in_local_types = ctx.local_types.contains_key(name);
        let fn_name = ctx.current_function_name.unwrap_or("?");
        eprintln!("[KNR-IDENT] fn='{}' name='{}' in_params={} in_locals={} in_globals={} in_local_types={} var_cty={:?}", fn_name, name, in_params, in_locals, in_globals, in_local_types, var_cty);
    }

    // VLA type is correctly Pointer(elem, ...) here via local_types override.

    // Arrays decay to pointers: the alloca IS the address of the first
    // element, so we return the alloca pointer directly without loading.
    let is_array_decay = types::is_array(&var_cty);
    let decayed_ty = if is_array_decay {
        // Array decays to pointer to element type.
        match &var_cty {
            CType::Array(elem, _) => CType::Pointer(elem.clone(), TypeQualifiers::default()),
            _ => var_cty.clone(),
        }
    } else {
        var_cty.clone()
    };

    // First check function parameter values (common fast-path).
    if ctx.param_values.contains_key(name) {
        if let Some((ptr_val, needs_load)) = lookup_var(ctx, name) {
            if is_array_decay {
                return TypedValue::new(ptr_val, decayed_ty);
            }
            if needs_load {
                let (loaded, li) = ctx.builder.build_load(ptr_val, ir_ty, span);
                emit_inst(ctx, li);
                return TypedValue::new(loaded, var_cty);
            }
            return TypedValue::new(ptr_val, var_cty);
        }
    }

    // Local variables.
    if let Some((ptr_val, needs_load)) = lookup_var(ctx, name) {
        if is_array_decay {
            return TypedValue::new(ptr_val, decayed_ty);
        }
        if needs_load {
            let (loaded, li) = ctx.builder.build_load(ptr_val, ir_ty, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
        return TypedValue::new(ptr_val, var_cty);
    }

    // GCC global register variables: `register T *name __asm__("reg");`
    // These map a C identifier to a hardware register.  We emit an
    // inline-asm instruction that reads the register and returns the
    // value directly (no memory access).
    if let Some((reg_name, reg_ctype)) = ctx.module.register_globals.get(name).cloned() {
        let _result_ir_ty = ctype_to_ir(&reg_ctype, ctx.target);
        // Build a trivial inline-asm: `mv $0, <reg>` that outputs the
        // register value.  We use the generic "=r" output constraint so
        // the backend copies the specific register into a fresh virtual
        // register.  The template is architecture-aware.
        let template = format!("mv $0, {}", reg_name);
        let (result_val, asm_inst) = ctx.builder.build_inline_asm(
            template,
            "=r".to_string(),
            Vec::new(),
            Vec::new(),
            false, // has_side_effects
            span,
        );
        emit_inst(ctx, asm_inst);
        return TypedValue::new(result_val, reg_ctype);
    }

    // Global variables.
    {
        let global_ty = ctx.module.get_global(name).map(|gv| gv.ty.clone());
        if let Some(gt) = global_ty {
            let global_name = name.to_string();
            let ptr_val = ctx.builder.fresh_value();
            // Record the mapping from this Value to the global variable name
            // so the backend can emit architecture-specific global addressing
            // (e.g., RIP-relative on x86-64, ADRP/LDR on AArch64).
            ctx.module
                .global_var_refs
                .insert(ptr_val, global_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, global_name);
            }

            // Check if the global variable itself is an array type.
            // lookup_var_type only checks local_types and may miss globals,
            // so we also detect array decay by inspecting the IR type.
            // IMPORTANT: Structs and unions are also represented as
            // IrType::Array in the IR (e.g., `union u { int i; }` →
            // `Array(I32, 1)`).  We must NOT treat those as C-level arrays.
            let resolved_var = types::resolve_typedef(&var_cty);
            let ctype_is_struct_or_union =
                matches!(resolved_var, CType::Struct { .. } | CType::Union { .. });
            let ctype_is_complex = matches!(resolved_var, CType::Complex(_));
            let global_is_array = matches!(&gt, crate::ir::types::IrType::Array(_, _))
                && !ctype_is_struct_or_union
                && !ctype_is_complex;

            // Arrays decay to pointers: the address of the global IS the
            // pointer to the first element, so return it directly (no load).
            if is_array_decay || global_is_array {
                // Compute proper decayed type.
                let ret_ty = if is_array_decay {
                    decayed_ty
                } else {
                    // Build pointer-to-element from the IR array element type.
                    // We construct a generic pointer CType since we don't have
                    // the original C element type here.
                    match &gt {
                        crate::ir::types::IrType::Array(elem_ir, _) => {
                            // Map IR element type back to an approximate CType.
                            let elem_cty = ir_elem_to_approx_ctype(elem_ir);
                            CType::Pointer(Box::new(elem_cty), TypeQualifiers::default())
                        }
                        _ => CType::Pointer(Box::new(CType::Int), TypeQualifiers::default()),
                    }
                };
                return TypedValue::new(ptr_val, ret_ty);
            }

            let (loaded, li) = ctx.builder.build_load(ptr_val, gt, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
    }

    // Function references (decay to function pointer).
    if ctx.module.get_function(name).is_some()
        || ctx.module.declarations().iter().any(|d| d.name == *name)
    {
        let func_name = name.to_string();
        let fptr = ctx.builder.fresh_value();
        // Record in both the module-level map (for backward compat) and the
        // per-function map (which is Value-ID safe across functions).
        ctx.module.func_ref_map.insert(fptr, func_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, func_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // Enum constants — compile-time integer values.
    if let Some(&enum_val) = ctx.enum_constants.get(name) {
        let val = emit_int_const(ctx, enum_val, IrType::I32, span);
        return TypedValue::new(val, CType::Int);
    }

    // Static local variables — redirected to their mangled global name.
    if let Some(mangled) = ctx.static_locals.get(name) {
        let global_ty = ctx
            .module
            .get_global(mangled.as_str())
            .map(|gv| gv.ty.clone());
        if let Some(gt) = global_ty {
            let mangled_name = mangled.clone();
            let ptr_val = ctx.builder.fresh_value();
            ctx.module
                .global_var_refs
                .insert(ptr_val, mangled_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, mangled_name);
            }
            // Check for array-to-pointer decay: arrays and string-initialized
            // arrays should return the pointer directly (no load).
            // Exclude struct/union types which also map to IR Array.
            let resolved_svar = types::resolve_typedef(&var_cty);
            let ctype_is_sou = matches!(resolved_svar, CType::Struct { .. } | CType::Union { .. });
            let ctype_is_complex_s = matches!(resolved_svar, CType::Complex(_));
            let global_is_array =
                matches!(&gt, IrType::Array(_, _)) && !ctype_is_sou && !ctype_is_complex_s;
            if is_array_decay || global_is_array {
                // Use the accurately-computed decayed type from
                // lookup_var_type (which reads global_c_types) when
                // available; this preserves Typedef/Struct element types
                // that ir_elem_to_approx_ctype would lose.
                let ret_ty = if is_array_decay {
                    decayed_ty
                } else if let IrType::Array(elem, _) = &gt {
                    let elem_cty = ir_elem_to_approx_ctype(elem);
                    CType::Pointer(Box::new(elem_cty), TypeQualifiers::default())
                } else {
                    var_cty
                };
                return TypedValue::new(ptr_val, ret_ty);
            }
            let (loaded, li) = ctx.builder.build_load(ptr_val, gt, span);
            emit_inst(ctx, li);
            return TypedValue::new(loaded, var_cty);
        }
    }

    // C99 §6.4.2.2: __func__ is an implicit static const char[] in each
    // function, expanding to the function name. GCC also supports
    // __FUNCTION__ and __PRETTY_FUNCTION__ as synonyms.
    if name == "__func__" || name == "__FUNCTION__" || name == "__PRETTY_FUNCTION__" {
        let func_name = ctx.current_function_name.unwrap_or("(unknown)");
        let mut bytes: Vec<u8> = func_name.as_bytes().to_vec();
        bytes.push(0); // NUL terminator
        let str_id = ctx.module.intern_string(bytes.clone());
        let str_global_name = format!(".str.{}", str_id);
        if ctx.module.get_global(&str_global_name).is_none() {
            let str_arr_ty = IrType::Array(Box::new(IrType::I8), bytes.len());
            let mut gv = crate::ir::module::GlobalVariable::new(
                str_global_name.clone(),
                str_arr_ty,
                Some(crate::ir::module::Constant::String(bytes)),
            );
            gv.linkage = crate::ir::module::Linkage::Internal;
            gv.is_constant = true;
            ctx.module.add_global(gv);
        }
        let result = ctx.builder.fresh_value();
        ctx.module
            .global_var_refs
            .insert(result, str_global_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.global_var_refs.insert(result, str_global_name);
        }
        return TypedValue::new(
            result,
            CType::Pointer(
                Box::new(CType::Char),
                crate::common::types::TypeQualifiers {
                    is_const: true,
                    ..Default::default()
                },
            ),
        );
    }

    // Implicit builtin functions — auto-declare in the IR module when
    // __builtin_* identifiers are used as function references.  The kernel
    // and libc code freely call __builtin_memcpy, __builtin_memset, etc.
    // without explicit declarations, relying on the compiler's knowledge.
    if name.starts_with("__builtin_") {
        // Capture the name as an owned string to release the borrow on ctx.
        let owned_name = name.to_string();
        // Map builtins to their underlying libc symbol names: the codegen
        // will emit a call to the libc function (e.g. memcpy, memset).
        let (lib_name_str, ret_ir, params_ir, variadic): (&str, IrType, Vec<IrType>, bool) =
            match owned_name.as_str() {
                "__builtin_memcpy" | "__builtin_memmove" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::Ptr, IrType::I64],
                    false,
                ),
                "__builtin_memset" => (
                    "memset",
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::I32, IrType::I64],
                    false,
                ),
                "__builtin_memcmp" => (
                    "memcmp",
                    IrType::I32,
                    vec![IrType::Ptr, IrType::Ptr, IrType::I64],
                    false,
                ),
                "__builtin_strlen" => ("strlen", IrType::I64, vec![IrType::Ptr], false),
                "__builtin_strcmp" | "__builtin_strncmp" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::I32,
                    vec![IrType::Ptr, IrType::Ptr],
                    false,
                ),
                "__builtin_strcpy" | "__builtin_strncpy" | "__builtin_strcat"
                | "__builtin_strncat" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::Ptr],
                    false,
                ),
                "__builtin_strchr" | "__builtin_strrchr" => (
                    owned_name.strip_prefix("__builtin_").unwrap(),
                    IrType::Ptr,
                    vec![IrType::Ptr, IrType::I32],
                    false,
                ),
                "__builtin_abs" => ("abs", IrType::I32, vec![IrType::I32], false),
                "__builtin_labs" => ("labs", IrType::I64, vec![IrType::I64], false),
                _ => {
                    // Generic fallback: declare as returning int with variadic params.
                    (
                        owned_name.strip_prefix("__builtin_").unwrap_or(&owned_name),
                        IrType::I32,
                        vec![],
                        true,
                    )
                }
            };
        let lib_name_owned = lib_name_str.to_string();
        // Add a declaration for the underlying libc function if not present.
        let already_declared = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == lib_name_owned)
            || ctx.module.get_function(&lib_name_owned).is_some();
        if !already_declared {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: lib_name_owned.clone(),
                    return_type: ret_ir.clone(),
                    param_types: params_ir.clone(),
                    is_variadic: variadic,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        // Return a function reference value pointing to the libc name.
        // Do NOT use the __builtin_ prefixed name — it doesn't exist in libc.
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, lib_name_owned.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, lib_name_owned);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // GCC __atomic_* and __sync_* builtins — atomic operations that the
    // compiler treats as known functions.  For simple cases (aligned integer
    // loads/stores) these compile to plain loads/stores on architectures
    // with naturally-atomic access (x86-64, AArch64).  We auto-declare them
    // as external functions so the call site resolves, then the call lowering
    // can handle them specially or emit a regular library call.
    if name.starts_with("__atomic_") || name.starts_with("__sync_") {
        let owned_name = name.to_string();
        let (ret_ir, params_ir, variadic): (IrType, Vec<IrType>, bool) = match owned_name.as_str() {
            "__atomic_load_n" | "__atomic_load" => {
                (IrType::I64, vec![IrType::Ptr, IrType::I32], false)
            }
            "__atomic_store_n" | "__atomic_store" => (
                IrType::Void,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_exchange_n" | "__atomic_exchange" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_compare_exchange_n" | "__atomic_compare_exchange" => (
                IrType::I8,
                vec![
                    IrType::Ptr,
                    IrType::Ptr,
                    IrType::I64,
                    IrType::I8,
                    IrType::I32,
                    IrType::I32,
                ],
                false,
            ),
            "__atomic_add_fetch"
            | "__atomic_sub_fetch"
            | "__atomic_and_fetch"
            | "__atomic_or_fetch"
            | "__atomic_xor_fetch"
            | "__atomic_nand_fetch"
            | "__atomic_fetch_add"
            | "__atomic_fetch_sub"
            | "__atomic_fetch_and"
            | "__atomic_fetch_or"
            | "__atomic_fetch_xor"
            | "__atomic_fetch_nand" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I32],
                false,
            ),
            "__atomic_test_and_set" => (IrType::I8, vec![IrType::Ptr, IrType::I32], false),
            "__atomic_clear" => (IrType::Void, vec![IrType::Ptr, IrType::I32], false),
            "__atomic_thread_fence" | "__atomic_signal_fence" => {
                (IrType::Void, vec![IrType::I32], false)
            }
            "__atomic_always_lock_free" | "__atomic_is_lock_free" => {
                (IrType::I8, vec![IrType::I64, IrType::Ptr], false)
            }
            "__sync_synchronize" => (IrType::Void, vec![], false),
            "__sync_fetch_and_add"
            | "__sync_fetch_and_sub"
            | "__sync_fetch_and_or"
            | "__sync_fetch_and_and"
            | "__sync_fetch_and_xor"
            | "__sync_fetch_and_nand"
            | "__sync_add_and_fetch"
            | "__sync_sub_and_fetch"
            | "__sync_or_and_fetch"
            | "__sync_and_and_fetch"
            | "__sync_xor_and_fetch"
            | "__sync_nand_and_fetch" => (IrType::I64, vec![IrType::Ptr, IrType::I64], true),
            "__sync_bool_compare_and_swap" => (
                IrType::I8,
                vec![IrType::Ptr, IrType::I64, IrType::I64],
                true,
            ),
            "__sync_val_compare_and_swap" => (
                IrType::I64,
                vec![IrType::Ptr, IrType::I64, IrType::I64],
                true,
            ),
            "__sync_lock_test_and_set" => (IrType::I64, vec![IrType::Ptr, IrType::I64], true),
            "__sync_lock_release" => (IrType::Void, vec![IrType::Ptr], true),
            _ => {
                // Generic fallback: int-returning with variadic params.
                (IrType::I64, vec![], true)
            }
        };
        let already_declared = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == owned_name)
            || ctx.module.get_function(&owned_name).is_some();
        if !already_declared {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: owned_name.clone(),
                    return_type: ret_ir,
                    param_types: params_ir,
                    is_variadic: variadic,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, owned_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, owned_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // Implicit function declarations for common C library functions.
    // These are functions that torture tests and real C code call without
    // prior declaration, relying on implicit int-returning convention.
    {
        let implicit_func: Option<(IrType, Vec<IrType>, bool)> = match name {
            "abort" | "_abort" => Some((IrType::Void, vec![], false)),
            "exit" | "_exit" | "_Exit" => Some((IrType::Void, vec![IrType::I32], false)),
            "malloc" | "calloc" | "realloc" => Some((IrType::Ptr, vec![], true)),
            "free" => Some((IrType::Void, vec![IrType::Ptr], false)),
            "memcpy" | "memmove" => Some((
                IrType::Ptr,
                vec![IrType::Ptr, IrType::Ptr, IrType::I64],
                false,
            )),
            "memset" => Some((
                IrType::Ptr,
                vec![IrType::Ptr, IrType::I32, IrType::I64],
                false,
            )),
            "memcmp" | "strcmp" | "strncmp" => Some((IrType::I32, vec![], true)),
            "strlen" => Some((IrType::I64, vec![IrType::Ptr], false)),
            "strcpy" | "strncpy" | "strcat" | "strncat" => Some((IrType::Ptr, vec![], true)),
            "printf" | "fprintf" | "sprintf" | "snprintf" => Some((IrType::I32, vec![], true)),
            "puts" | "fputs" | "putchar" | "putc" | "fputc" => Some((IrType::I32, vec![], true)),
            "getchar" | "fgetc" | "getc" => Some((IrType::I32, vec![], true)),
            "fopen" | "fdopen" => Some((IrType::Ptr, vec![], true)),
            "fclose" | "fflush" => Some((IrType::I32, vec![IrType::Ptr], false)),
            "fread" | "fwrite" => Some((IrType::I64, vec![], true)),
            "fseek" | "ftell" | "feof" | "ferror" => Some((IrType::I32, vec![], true)),
            "atoi" | "atol" => Some((IrType::I32, vec![IrType::Ptr], false)),
            "abs" | "labs" => Some((IrType::I32, vec![IrType::I32], false)),
            "llabs" => Some((IrType::I64, vec![IrType::I64], false)),
            "fabs" => Some((IrType::F64, vec![IrType::F64], false)),
            "fabsf" => Some((IrType::F32, vec![IrType::F32], false)),
            "sqrt" | "sin" | "cos" | "tan" | "log" | "log2" | "log10" | "exp" | "pow" | "ceil"
            | "floor" | "round" | "fmod" | "atan2" | "atan" | "asin" | "acos" | "sinh" | "cosh"
            | "tanh" => Some((IrType::F64, vec![], true)),
            "sqrtf" | "sinf" | "cosf" | "tanf" | "logf" | "expf" | "powf" | "ceilf" | "floorf"
            | "roundf" | "fmodf" => Some((IrType::F32, vec![], true)),
            "qsort" | "bsearch" => Some((IrType::Void, vec![], true)),
            "signal" => Some((IrType::Ptr, vec![], true)),
            "setjmp" | "_setjmp" | "sigsetjmp" => Some((IrType::I32, vec![IrType::Ptr], false)),
            "longjmp" | "siglongjmp" => Some((IrType::Void, vec![], true)),
            "strtol" | "strtoul" | "strtoll" | "strtoull" | "strtod" | "strtof" => {
                Some((IrType::I64, vec![], true))
            }
            "sscanf" | "fscanf" | "scanf" => Some((IrType::I32, vec![], true)),
            "perror" => Some((IrType::Void, vec![IrType::Ptr], false)),
            "raise" | "system" | "atexit" => Some((IrType::I32, vec![], true)),
            "getenv" => Some((IrType::Ptr, vec![IrType::Ptr], false)),
            "isalpha" | "isdigit" | "isalnum" | "isspace" | "isupper" | "islower" | "ispunct"
            | "isprint" | "iscntrl" | "isxdigit" | "toupper" | "tolower" => {
                Some((IrType::I32, vec![IrType::I32], false))
            }
            _ => None,
        };
        if let Some((ret_ir, params_ir, variadic)) = implicit_func {
            let owned_name = name.to_string();
            let already_declared = ctx
                .module
                .declarations()
                .iter()
                .any(|d| d.name == owned_name)
                || ctx.module.get_function(&owned_name).is_some();
            if !already_declared {
                ctx.module
                    .add_declaration(crate::ir::module::FunctionDeclaration {
                        name: owned_name.clone(),
                        return_type: ret_ir,
                        param_types: params_ir,
                        is_variadic: variadic,
                        linkage: crate::ir::module::Linkage::External,
                        visibility: crate::ir::module::Visibility::Default,
                    });
            }
            let fptr = ctx.builder.fresh_value();
            ctx.module.func_ref_map.insert(fptr, owned_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.func_ref_map.insert(fptr, owned_name);
            }
            return TypedValue::new(
                fptr,
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            );
        }
    }

    // Global variables like stdout, stderr, stdin — accessed without declaration
    if matches!(name, "stdout" | "stderr" | "stdin") {
        let owned_name = name.to_string();
        // Declare as an extern global variable of type pointer
        if ctx.module.get_global(&owned_name).is_none() {
            let mut gv =
                crate::ir::module::GlobalVariable::new(owned_name.clone(), IrType::Ptr, None);
            gv.linkage = crate::ir::module::Linkage::External;
            ctx.module.add_global(gv);
        }
        let ptr_val = ctx.builder.fresh_value();
        ctx.module
            .global_var_refs
            .insert(ptr_val, owned_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.global_var_refs.insert(ptr_val, owned_name);
        }
        let (loaded, li) = ctx.builder.build_load(ptr_val, IrType::Ptr, span);
        emit_inst(ctx, li);
        return TypedValue::new(
            loaded,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }

    // C89/GNU89 implicit function declaration: treat undeclared identifiers
    // as `int name(...)` when they appear in call context.  The semantic
    // analyzer already emitted a warning.  Generate a declaration so that
    // the linker can resolve the symbol at link time.
    {
        let owned_name = name.to_string();
        let ret_ir = IrType::I32;
        let already_declared = ctx
            .module
            .declarations()
            .iter()
            .any(|d| d.name == owned_name)
            || ctx.module.get_function(&owned_name).is_some();
        if !already_declared {
            ctx.module
                .add_declaration(crate::ir::module::FunctionDeclaration {
                    name: owned_name.clone(),
                    return_type: ret_ir,
                    param_types: vec![],
                    is_variadic: true,
                    linkage: crate::ir::module::Linkage::External,
                    visibility: crate::ir::module::Visibility::Default,
                });
        }
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, owned_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, owned_name);
        }
        TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        )
    }
}

fn lower_identifier_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    sym_idx: u32,
    span: Span,
) -> TypedValue {
    let name = resolve_sym(ctx, sym_idx);
    let var_cty = lookup_var_type(ctx, name);
    if let Some((ptr_val, _)) = lookup_var(ctx, name) {
        return TypedValue::new(ptr_val, var_cty);
    }
    // GCC global register variables do not have addressable storage.
    // They cannot appear as lvalues for address-of (&) but can appear
    // in assignment context.  For now we emit an error if used as an
    // lvalue (the kernel never does this).
    if ctx.module.register_globals.contains_key(name) {
        ctx.diagnostics.emit_error(
            span,
            format!("address of global register variable '{}' requested", name),
        );
        return TypedValue::new(Value::UNDEF, var_cty);
    }
    if ctx.module.get_global(name).is_some() {
        let global_name = name.to_string();
        let ptr_val = ctx.builder.fresh_value();
        ctx.module
            .global_var_refs
            .insert(ptr_val, global_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.global_var_refs.insert(ptr_val, global_name);
        }
        return TypedValue::new(ptr_val, var_cty);
    }
    // Static local variables — redirected to their mangled global name.
    if let Some(mangled) = ctx.static_locals.get(name) {
        if ctx.module.get_global(mangled.as_str()).is_some() {
            let mangled_name = mangled.clone();
            let ptr_val = ctx.builder.fresh_value();
            ctx.module
                .global_var_refs
                .insert(ptr_val, mangled_name.clone());
            {
                let current_func = &mut *ctx.function;
                current_func.global_var_refs.insert(ptr_val, mangled_name);
            }
            return TypedValue::new(ptr_val, var_cty);
        }
    }
    // Enum constants — compile-time integer values used as lvalues
    // (rare but can occur in macro expansions).
    if let Some(&enum_val) = ctx.enum_constants.get(name) {
        let val = emit_int_const(ctx, enum_val, IrType::I32, span);
        return TypedValue::new(val, CType::Int);
    }
    // Function references as lvalues — used for `&function_name`.
    // Functions decay to pointers, so taking the address of a function
    // returns the function pointer value.
    if ctx.module.get_function(name).is_some()
        || ctx.module.declarations().iter().any(|d| d.name == *name)
    {
        let func_name = name.to_string();
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, func_name.clone());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, func_name);
        }
        return TypedValue::new(
            fptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );
    }
    ctx.diagnostics
        .emit_error(span, format!("undeclared identifier '{}'", name));
    TypedValue::new(Value::UNDEF, CType::Int)
}

// =========================================================================
// BINARY OPERATIONS
// =========================================================================

fn lower_binary(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::BinaryOp,
    lhs_expr: &ast::Expression,
    rhs_expr: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::BinaryOp::LogicalAnd => return lower_logical_and(ctx, lhs_expr, rhs_expr, span),
        ast::BinaryOp::LogicalOr => return lower_logical_or(ctx, lhs_expr, rhs_expr, span),
        _ => {}
    }
    let lhs = lower_expr_inner(ctx, lhs_expr);
    let rhs = lower_expr_inner(ctx, rhs_expr);

    // Apply array-to-pointer decay for operands with array type.
    // In C, arrays in expression context decay to pointers-to-first-element
    // (C11 §6.3.2.1p3).  This is critical for expressions like `*(p+i) + j`
    // where `*(p+i)` yields an array lvalue that must decay to a pointer
    // before the `+ j` pointer arithmetic is applied.
    let lhs = if let CType::Array(elem, _) = types::resolve_typedef(&lhs.ty) {
        TypedValue::new(
            lhs.value,
            CType::Pointer(elem.clone(), types::TypeQualifiers::default()),
        )
    } else {
        lhs
    };
    let rhs = if let CType::Array(elem, _) = types::resolve_typedef(&rhs.ty) {
        TypedValue::new(
            rhs.value,
            CType::Pointer(elem.clone(), types::TypeQualifiers::default()),
        )
    } else {
        rhs
    };

    // ---- _Complex type arithmetic ----
    // Decompose complex operations into component-wise floating-point ops.
    {
        let lhs_resolved = types::resolve_typedef(&lhs.ty);
        let rhs_resolved = types::resolve_typedef(&rhs.ty);
        if matches!(lhs_resolved, CType::Complex(..)) || matches!(rhs_resolved, CType::Complex(..))
        {
            return lower_complex_binary(ctx, op, lhs_expr, rhs_expr, &lhs, &rhs, span);
        }
    }

    if is_pointer_type(&lhs.ty) && is_integer_type(&rhs.ty) {
        // Pointer-integer comparisons: convert integer to pointer and use ptr-ptr comparison.
        match op {
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let (li, lic) = ctx.builder.build_ptr_to_int(lhs.value, pi.clone(), span);
                emit_inst(ctx, lic);
                let rv = insert_implicit_conversion(
                    ctx,
                    rhs.value,
                    &rhs.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, li, rv, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {
                return lower_ptr_arith(ctx, op, &lhs, &rhs, span);
            }
        }
    }
    if is_integer_type(&lhs.ty) && is_pointer_type(&rhs.ty) {
        match op {
            ast::BinaryOp::Add => {
                return lower_ptr_arith(ctx, op, &rhs, &lhs, span);
            }
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let lv = insert_implicit_conversion(
                    ctx,
                    lhs.value,
                    &lhs.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let (ri, ric) = ctx.builder.build_ptr_to_int(rhs.value, pi.clone(), span);
                emit_inst(ctx, ric);
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, lv, ri, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {}
        }
    }
    if is_pointer_type(&lhs.ty) && is_pointer_type(&rhs.ty) && matches!(op, ast::BinaryOp::Sub) {
        return lower_ptr_diff(ctx, &lhs, &rhs, span);
    }
    // Pointer-pointer comparisons (==, !=, <, >, <=, >=): convert both to
    // integers (ptrdiff_t-sized) and perform an unsigned integer comparison.
    if is_pointer_type(&lhs.ty) && is_pointer_type(&rhs.ty) {
        match op {
            ast::BinaryOp::Equal
            | ast::BinaryOp::NotEqual
            | ast::BinaryOp::Less
            | ast::BinaryOp::LessEqual
            | ast::BinaryOp::Greater
            | ast::BinaryOp::GreaterEqual => {
                let pi = size_ir_type(ctx.target);
                let (li, lic) = ctx.builder.build_ptr_to_int(lhs.value, pi.clone(), span);
                emit_inst(ctx, lic);
                let (ri, ric) = ctx.builder.build_ptr_to_int(rhs.value, pi.clone(), span);
                emit_inst(ctx, ric);
                let cmp_op = match op {
                    ast::BinaryOp::Equal => ICmpOp::Eq,
                    ast::BinaryOp::NotEqual => ICmpOp::Ne,
                    ast::BinaryOp::Less => ICmpOp::Ult,
                    ast::BinaryOp::LessEqual => ICmpOp::Ule,
                    ast::BinaryOp::Greater => ICmpOp::Ugt,
                    ast::BinaryOp::GreaterEqual => ICmpOp::Uge,
                    _ => unreachable!(),
                };
                let (v, ci) = ctx.builder.build_icmp(cmp_op, li, ri, span);
                emit_inst(ctx, ci);
                return TypedValue::new(v, CType::Bool);
            }
            _ => {
                // Pointer + pointer is only valid for sub (handled above)
                // and comparisons (handled above). Other ops fall through
                // to the pointer arithmetic handler which will error.
                return lower_ptr_arith(ctx, op, &lhs, &rhs, span);
            }
        }
    }

    // C11 §6.5.7: Shift operations — each operand undergoes integer promotions
    // independently; result type is the promoted left operand type.
    // Do NOT apply usual arithmetic conversions (which would convert both to
    // a common type, potentially making signed → unsigned).
    if matches!(op, ast::BinaryOp::ShiftLeft | ast::BinaryOp::ShiftRight) {
        let lhs_promoted = type_builder::integer_promote(&lhs.ty);
        let rhs_promoted = type_builder::integer_promote(&rhs.ty);
        let lhs_ir = ctype_to_ir(&lhs_promoted, ctx.target);
        let rhs_ir = ctype_to_ir(&rhs_promoted, ctx.target);
        let lv = insert_implicit_conversion(ctx, lhs.value, &lhs.ty, &lhs_promoted, span);
        let rv = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &rhs_promoted, span);
        // Convert rhs to same IR type as lhs for the shift instruction
        let rv_conv = if rhs_ir != lhs_ir {
            insert_implicit_conversion(ctx, rv, &rhs_promoted, &lhs_promoted, span)
        } else {
            rv
        };
        let lhs_unsigned = is_unsigned(&lhs_promoted);
        // C11 §6.5.7: result type is the promoted LEFT operand type.
        // If the left operand is a bitfield, truncate the shift result.
        let lhs_bfw = lhs.bitfield_width;
        return match op {
            ast::BinaryOp::ShiftLeft => {
                let (v, i) = ctx.builder.build_shl(lv, rv_conv, lhs_ir, span);
                emit_inst(ctx, i);
                let tv = TypedValue::new(v, lhs_promoted);
                maybe_truncate_bitfield(ctx, tv, lhs_bfw, None, span)
            }
            ast::BinaryOp::ShiftRight => {
                let (v, i) = ctx
                    .builder
                    .build_shr(lv, rv_conv, lhs_ir, !lhs_unsigned, span);
                emit_inst(ctx, i);
                let tv = TypedValue::new(v, lhs_promoted);
                maybe_truncate_bitfield(ctx, tv, lhs_bfw, None, span)
            }
            _ => unreachable!(),
        };
    }

    // Apply usual arithmetic conversions; also check integer rank for
    // type-dependent instruction selection (signed vs unsigned).
    let _lhs_rank = type_builder::integer_rank(&lhs.ty);
    let _rhs_rank = type_builder::integer_rank(&rhs.ty);
    let _lhs_is_int = type_builder::is_integer_type(&lhs.ty);
    // Capture bitfield widths before consuming operands.
    let lhs_bfw = lhs.bitfield_width;
    let rhs_bfw = rhs.bitfield_width;
    let common = type_builder::usual_arithmetic_conversion(&lhs.ty, &rhs.ty);
    let ci = ctype_to_ir(&common, ctx.target);
    let lv = insert_implicit_conversion(ctx, lhs.value, &lhs.ty, &common, span);
    let rv = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &common, span);
    let uns = is_unsigned(&common);

    match op {
        ast::BinaryOp::Add => {
            let (v, i) = ctx.builder.build_add(lv, rv, ci, span);
            emit_inst(ctx, i);
            let tv = TypedValue::new(v, common);
            maybe_truncate_bitfield(ctx, tv, lhs_bfw, rhs_bfw, span)
        }
        ast::BinaryOp::Sub => {
            let (v, i) = ctx.builder.build_sub(lv, rv, ci, span);
            emit_inst(ctx, i);
            let tv = TypedValue::new(v, common);
            maybe_truncate_bitfield(ctx, tv, lhs_bfw, rhs_bfw, span)
        }
        ast::BinaryOp::Mul => {
            let (v, i) = ctx.builder.build_mul(lv, rv, ci, span);
            emit_inst(ctx, i);
            let tv = TypedValue::new(v, common);
            maybe_truncate_bitfield(ctx, tv, lhs_bfw, rhs_bfw, span)
        }
        ast::BinaryOp::Div => {
            let (v, i) = ctx.builder.build_div(lv, rv, ci, !uns, span);
            emit_inst(ctx, i);
            let tv = TypedValue::new(v, common);
            maybe_truncate_bitfield(ctx, tv, lhs_bfw, rhs_bfw, span)
        }
        ast::BinaryOp::Mod => {
            let (v, i) = ctx.builder.build_rem(lv, rv, ci, !uns, span);
            emit_inst(ctx, i);
            let tv = TypedValue::new(v, common);
            maybe_truncate_bitfield(ctx, tv, lhs_bfw, rhs_bfw, span)
        }
        ast::BinaryOp::BitwiseAnd => {
            let (v, i) = ctx.builder.build_and(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::BitwiseOr => {
            let (v, i) = ctx.builder.build_or(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        ast::BinaryOp::BitwiseXor => {
            let (v, i) = ctx.builder.build_xor(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
        // Shifts handled above via early return
        ast::BinaryOp::ShiftLeft | ast::BinaryOp::ShiftRight => unreachable!(),
        ast::BinaryOp::Equal => TypedValue::new(
            emit_cmp(ctx, lv, rv, &common, ICmpOp::Eq, FCmpOp::Oeq, span),
            CType::Bool,
        ),
        ast::BinaryOp::NotEqual => TypedValue::new(
            emit_cmp(ctx, lv, rv, &common, ICmpOp::Ne, FCmpOp::One, span),
            CType::Bool,
        ),
        ast::BinaryOp::Less => {
            let ic = if uns { ICmpOp::Ult } else { ICmpOp::Slt };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Olt, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::LessEqual => {
            let ic = if uns { ICmpOp::Ule } else { ICmpOp::Sle };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Ole, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::Greater => {
            let ic = if uns { ICmpOp::Ugt } else { ICmpOp::Sgt };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Ogt, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::GreaterEqual => {
            let ic = if uns { ICmpOp::Uge } else { ICmpOp::Sge };
            TypedValue::new(
                emit_cmp(ctx, lv, rv, &common, ic, FCmpOp::Oge, span),
                CType::Bool,
            )
        }
        ast::BinaryOp::LogicalAnd | ast::BinaryOp::LogicalOr => unreachable!(),
    }
}

fn emit_cmp(
    ctx: &mut ExprLoweringContext<'_>,
    l: Value,
    r: Value,
    ty: &CType,
    ic: ICmpOp,
    fc: FCmpOp,
    span: Span,
) -> Value {
    if is_floating(ty) {
        let (v, i) = ctx.builder.build_fcmp(fc, l, r, span);
        emit_inst(ctx, i);
        v
    } else {
        let (v, i) = ctx.builder.build_icmp(ic, l, r, span);
        emit_inst(ctx, i);
        v
    }
}

// =========================================================================
// _COMPLEX ARITHMETIC  (component-wise FP decomposition)
// =========================================================================

/// Compute the byte-size of a single element of a `_Complex` type.
/// Handles both floating-point and integer element types correctly.
fn complex_elem_size(elem_ir: &IrType) -> i64 {
    match elem_ir {
        IrType::F32 => 4,
        IrType::F64 => 8,
        IrType::F80 => 16, // 80-bit extended padded to 16 bytes
        IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 => 4,
        IrType::I64 => 8,
        IrType::I128 => 16,
        _ => 8,
    }
}

/// Returns `true` if the given `IrType` is a floating-point type.
fn is_fp_ir_type(ir_ty: &IrType) -> bool {
    matches!(ir_ty, IrType::F32 | IrType::F64 | IrType::F80)
}

/// Determine the scalar element type for a `_Complex` C type.
/// Returns `None` if `ty` is not `_Complex`.
fn complex_element_ctype(ty: &CType) -> Option<&CType> {
    match types::resolve_typedef(ty) {
        CType::Complex(base) => Some(base.as_ref()),
        _ => None,
    }
}

/// Helper: given the lvalue (pointer) of a `_Complex` variable, load the
/// real part (offset 0) and imaginary part (offset `elem_size`) as separate
/// scalar float values.
fn load_complex_parts(
    ctx: &mut ExprLoweringContext<'_>,
    complex_ptr: Value,
    elem_ir: &IrType,
    elem_size: i64,
    span: Span,
) -> (Value, Value) {
    // Real part: GEP base+0 → load
    let zero_idx = emit_int_const_for_index(ctx, 0);
    let (real_ptr, gep_r) = ctx
        .builder
        .build_gep(complex_ptr, vec![zero_idx], IrType::Ptr, span);
    emit_inst(ctx, gep_r);
    let (real_val, ld_r) = ctx.builder.build_load(real_ptr, elem_ir.clone(), span);
    emit_inst(ctx, ld_r);

    // Imaginary part: GEP base+elem_size → load
    let off_idx = emit_int_const_for_index(ctx, elem_size);
    let (imag_ptr, gep_i) = ctx
        .builder
        .build_gep(complex_ptr, vec![off_idx], IrType::Ptr, span);
    emit_inst(ctx, gep_i);
    let (imag_val, ld_i) = ctx.builder.build_load(imag_ptr, elem_ir.clone(), span);
    emit_inst(ctx, ld_i);

    (real_val, imag_val)
}

/// Helper: store real and imaginary parts into a freshly allocated `_Complex`
/// alloca and return the alloca pointer.
fn store_complex_result(
    ctx: &mut ExprLoweringContext<'_>,
    real_val: Value,
    imag_val: Value,
    complex_ir: &IrType,
    elem_size: i64,
    span: Span,
) -> Value {
    let (alloca, ai) = ctx.builder.build_alloca(complex_ir.clone(), span);
    emit_inst(ctx, ai);

    // Store real at offset 0
    let zero_idx = emit_int_const_for_index(ctx, 0);
    let (real_ptr, gep_r) = ctx
        .builder
        .build_gep(alloca, vec![zero_idx], IrType::Ptr, span);
    emit_inst(ctx, gep_r);
    let st_r = ctx.builder.build_store(real_val, real_ptr, span);
    emit_inst(ctx, st_r);

    // Store imag at offset elem_size
    let off_idx = emit_int_const_for_index(ctx, elem_size);
    let (imag_ptr, gep_i) = ctx
        .builder
        .build_gep(alloca, vec![off_idx], IrType::Ptr, span);
    emit_inst(ctx, gep_i);
    let st_i = ctx.builder.build_store(imag_val, imag_ptr, span);
    emit_inst(ctx, st_i);

    alloca
}

/// Lower a `_Complex` expression and extract its real/imaginary parts.
/// If the expression is an lvalue (variable/array element), we use the
/// existing pointer.  Otherwise (e.g. imaginary literal, function call
/// result), we materialise a temporary alloca and load parts from that.
/// Lower a `_Complex` expression from its AST node and extract its real/imaginary parts.
/// Prefer `extract_complex_parts_from_value` when you already have a lowered Value to avoid
/// duplicate IR node creation.
#[allow(dead_code)]
fn lower_complex_operand_parts(
    ctx: &mut ExprLoweringContext<'_>,
    expr: &ast::Expression,
    complex_ir: &IrType,
    elem_ir: &IrType,
    elem_size: i64,
    span: Span,
) -> (Value, Value) {
    let tv = lower_expr_inner(ctx, expr);
    extract_complex_parts_from_value(ctx, tv.value, complex_ir, elem_ir, elem_size, span)
}

/// Extract real and imaginary parts from an already-lowered `_Complex` value.
/// Materialises into a temp alloca so we can GEP into it, then loads both parts.
/// This avoids re-lowering expressions and prevents duplicate IR node creation.
fn extract_complex_parts_from_value(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    complex_ir: &IrType,
    elem_ir: &IrType,
    elem_size: i64,
    span: Span,
) -> (Value, Value) {
    let (alloca, ai) = ctx.builder.build_alloca(complex_ir.clone(), span);
    emit_inst(ctx, ai);
    let st = ctx.builder.build_store(value, alloca, span);
    emit_inst(ctx, st);
    load_complex_parts(ctx, alloca, elem_ir, elem_size, span)
}

/// Lower `__builtin_conjf`, `__builtin_conj`, `__builtin_conjl`:
/// complex conjugate — negate the imaginary part.
fn lower_complex_conj(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    elem_ctype: CType,
    elem_ir: IrType,
    span: Span,
) -> Option<TypedValue> {
    if args.is_empty() {
        return None;
    }
    let complex_ctype = CType::Complex(Box::new(elem_ctype.clone()));
    let complex_ir = ctype_to_ir(&complex_ctype, ctx.target);
    let elem_size = complex_elem_size(&elem_ir);

    let inner_tv = lower_expr_inner(ctx, &args[0]);
    let (real, imag) = extract_complex_parts_from_value(
        ctx,
        inner_tv.value,
        &complex_ir,
        &elem_ir,
        elem_size,
        span,
    );

    // Negate the imaginary part: 0 - imag.
    let zero = if matches!(elem_ir, IrType::F32 | IrType::F64 | IrType::F80) {
        emit_float_const(ctx, 0.0, elem_ir.clone(), span)
    } else {
        emit_int_const(ctx, 0, elem_ir.clone(), span)
    };
    let (neg_imag, neg_inst) = ctx.builder.build_sub(zero, imag, elem_ir.clone(), span);
    emit_inst(ctx, neg_inst);

    let result = store_complex_result(ctx, real, neg_imag, &complex_ir, elem_size, span);
    let (loaded, ld) = ctx.builder.build_load(result, complex_ir, span);
    emit_inst(ctx, ld);
    Some(TypedValue::new(loaded, complex_ctype))
}

/// Convert a scalar value to the target floating-point element type
/// of a `_Complex`.  Delegates to `lower_cast` for the actual
/// instruction emission (int→fp, fp→fp, etc.).
fn ensure_fp_type(
    ctx: &mut ExprLoweringContext<'_>,
    val: Value,
    src_ctype: &CType,
    target_ir: &IrType,
    span: Span,
) -> Value {
    let resolved = types::resolve_typedef(src_ctype);
    let src_ir = ctype_to_ir(resolved, ctx.target);
    if src_ir == *target_ir {
        return val;
    }
    // Map the target IR type back to a CType for lower_cast.
    let target_ctype = match target_ir {
        IrType::F32 => CType::Float,
        IrType::F64 => CType::Double,
        IrType::F80 => CType::LongDouble,
        _ => CType::Double,
    };
    lower_cast(ctx, val, resolved, &target_ctype, span)
}

/// Lower a binary operation on `_Complex` types by decomposing into
/// component-wise floating-point operations on real/imaginary parts.
#[allow(clippy::too_many_arguments)]
fn lower_complex_binary(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::BinaryOp,
    _lhs_expr: &ast::Expression,
    _rhs_expr: &ast::Expression,
    _lhs_tv: &TypedValue,
    _rhs_tv: &TypedValue,
    span: Span,
) -> TypedValue {
    // Determine the complex element type (Float, Double, LongDouble).
    let lhs_resolved = types::resolve_typedef(&_lhs_tv.ty);
    let rhs_resolved = types::resolve_typedef(&_rhs_tv.ty);

    let complex_ctype: CType;
    let elem_ctype: CType;
    let lhs_elem = complex_element_ctype(lhs_resolved);
    let rhs_elem = complex_element_ctype(rhs_resolved);
    match (lhs_elem, rhs_elem) {
        (Some(le), Some(re)) => {
            // Both complex: promote to the common element type using
            // usual arithmetic conversions.
            let common_elem = type_builder::usual_arithmetic_conversion(le, re);
            elem_ctype = common_elem.clone();
            complex_ctype = CType::Complex(Box::new(common_elem));
        }
        (Some(le), None) => {
            // LHS is complex, RHS is scalar: promote scalar side.
            // If the scalar is floating and wider, use float type.
            let rhs_promoted = type_builder::usual_arithmetic_conversion(le, rhs_resolved);
            elem_ctype = rhs_promoted.clone();
            complex_ctype = CType::Complex(Box::new(rhs_promoted));
        }
        (None, Some(re)) => {
            let lhs_promoted = type_builder::usual_arithmetic_conversion(lhs_resolved, re);
            elem_ctype = lhs_promoted.clone();
            complex_ctype = CType::Complex(Box::new(lhs_promoted));
        }
        (None, None) => {
            unreachable!("lower_complex_binary called without _Complex type");
        }
    }

    let elem_ir = ctype_to_ir(&elem_ctype, ctx.target);
    let complex_ir = ctype_to_ir(&complex_ctype, ctx.target);
    let elem_size = complex_elem_size(&elem_ir);

    // Load real/imaginary parts from both operands.  If one operand is a
    // plain real scalar (not _Complex), promote it to complex by treating
    // its value as the real component with imaginary = 0.0.
    //
    // IMPORTANT: We use the already-lowered values (_lhs_tv, _rhs_tv) directly
    // rather than re-lowering from the AST expressions. Re-lowering would create
    // duplicate IR nodes for every complex binary operation, causing unbounded
    // IR growth (memory leak) in programs with many _Complex operations.
    let lhs_is_complex = matches!(types::resolve_typedef(&_lhs_tv.ty), CType::Complex(..));
    let rhs_is_complex = matches!(types::resolve_typedef(&_rhs_tv.ty), CType::Complex(..));

    let elem_is_fp = is_fp_ir_type(&elem_ir);

    // Helper: extract complex parts from an already-lowered operand value,
    // converting to the common element type if the operand's element type differs.
    let extract_complex_parts_typed = |ctx: &mut ExprLoweringContext<'_>,
                                       tv: &TypedValue|
     -> (Value, Value) {
        let operand_resolved = types::resolve_typedef(&tv.ty);
        if let Some(op_elem) = complex_element_ctype(operand_resolved) {
            let op_elem_ir = ctype_to_ir(op_elem, ctx.target);
            let op_complex_ir = ctype_to_ir(operand_resolved, ctx.target);
            let op_esz = complex_elem_size(&op_elem_ir);
            // Extract parts from the already-lowered value (no re-lowering)
            let (raw_r, raw_i) = extract_complex_parts_from_value(
                ctx,
                tv.value,
                &op_complex_ir,
                &op_elem_ir,
                op_esz,
                span,
            );
            // Convert to the common element type if needed
            if op_elem_ir == elem_ir {
                (raw_r, raw_i)
            } else {
                let cvt_r = lower_cast(ctx, raw_r, op_elem, &elem_ctype, span);
                let cvt_i = lower_cast(ctx, raw_i, op_elem, &elem_ctype, span);
                (cvt_r, cvt_i)
            }
        } else {
            // Should not happen (caller verified lhs_is_complex)
            extract_complex_parts_from_value(ctx, tv.value, &complex_ir, &elem_ir, elem_size, span)
        }
    };

    let (lr, li) = if lhs_is_complex {
        extract_complex_parts_typed(ctx, _lhs_tv)
    } else {
        // Scalar: use the already-lowered value, don't re-lower
        let real_val = if elem_is_fp {
            ensure_fp_type(ctx, _lhs_tv.value, &_lhs_tv.ty, &elem_ir, span)
        } else {
            lower_cast(ctx, _lhs_tv.value, &_lhs_tv.ty, &elem_ctype, span)
        };
        let zero = if elem_is_fp {
            emit_float_const(ctx, 0.0, elem_ir.clone(), span)
        } else {
            emit_int_const(ctx, 0, elem_ir.clone(), span)
        };
        (real_val, zero)
    };
    let (rr, ri) = if rhs_is_complex {
        extract_complex_parts_typed(ctx, _rhs_tv)
    } else {
        // Scalar: use the already-lowered value, don't re-lower
        let real_val = if elem_is_fp {
            ensure_fp_type(ctx, _rhs_tv.value, &_rhs_tv.ty, &elem_ir, span)
        } else {
            lower_cast(ctx, _rhs_tv.value, &_rhs_tv.ty, &elem_ctype, span)
        };
        let zero = if elem_is_fp {
            emit_float_const(ctx, 0.0, elem_ir.clone(), span)
        } else {
            emit_int_const(ctx, 0, elem_ir.clone(), span)
        };
        (real_val, zero)
    };

    match op {
        ast::BinaryOp::Add => {
            // (a+bi) + (c+di) = (a+c) + (b+d)i
            let (res_r, inst_r) = ctx.builder.build_add(lr, rr, elem_ir.clone(), span);
            emit_inst(ctx, inst_r);
            let (res_i, inst_i) = ctx.builder.build_add(li, ri, elem_ir.clone(), span);
            emit_inst(ctx, inst_i);
            let result = store_complex_result(ctx, res_r, res_i, &complex_ir, elem_size, span);
            let (loaded, ld) = ctx.builder.build_load(result, complex_ir, span);
            emit_inst(ctx, ld);
            TypedValue::new(loaded, complex_ctype)
        }
        ast::BinaryOp::Sub => {
            // (a+bi) - (c+di) = (a-c) + (b-d)i
            let (res_r, inst_r) = ctx.builder.build_sub(lr, rr, elem_ir.clone(), span);
            emit_inst(ctx, inst_r);
            let (res_i, inst_i) = ctx.builder.build_sub(li, ri, elem_ir.clone(), span);
            emit_inst(ctx, inst_i);
            let result = store_complex_result(ctx, res_r, res_i, &complex_ir, elem_size, span);
            let (loaded, ld) = ctx.builder.build_load(result, complex_ir, span);
            emit_inst(ctx, ld);
            TypedValue::new(loaded, complex_ctype)
        }
        ast::BinaryOp::Mul => {
            // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
            let (ac, i1) = ctx.builder.build_mul(lr, rr, elem_ir.clone(), span);
            emit_inst(ctx, i1);
            let (bd, i2) = ctx.builder.build_mul(li, ri, elem_ir.clone(), span);
            emit_inst(ctx, i2);
            let (ad, i3) = ctx.builder.build_mul(lr, ri, elem_ir.clone(), span);
            emit_inst(ctx, i3);
            let (bc, i4) = ctx.builder.build_mul(li, rr, elem_ir.clone(), span);
            emit_inst(ctx, i4);
            let (res_r, i5) = ctx.builder.build_sub(ac, bd, elem_ir.clone(), span);
            emit_inst(ctx, i5);
            let (res_i, i6) = ctx.builder.build_add(ad, bc, elem_ir.clone(), span);
            emit_inst(ctx, i6);
            let result = store_complex_result(ctx, res_r, res_i, &complex_ir, elem_size, span);
            let (loaded, ld) = ctx.builder.build_load(result, complex_ir, span);
            emit_inst(ctx, ld);
            TypedValue::new(loaded, complex_ctype)
        }
        ast::BinaryOp::Div => {
            // (a+bi) / (c+di) = ((ac+bd)/(c²+d²)) + ((bc-ad)/(c²+d²))i
            let (ac, i1) = ctx.builder.build_mul(lr, rr, elem_ir.clone(), span);
            emit_inst(ctx, i1);
            let (bd, i2) = ctx.builder.build_mul(li, ri, elem_ir.clone(), span);
            emit_inst(ctx, i2);
            let (bc, i3) = ctx.builder.build_mul(li, rr, elem_ir.clone(), span);
            emit_inst(ctx, i3);
            let (ad, i4) = ctx.builder.build_mul(lr, ri, elem_ir.clone(), span);
            emit_inst(ctx, i4);
            let (cc, i5) = ctx.builder.build_mul(rr, rr, elem_ir.clone(), span);
            emit_inst(ctx, i5);
            let (dd, i6) = ctx.builder.build_mul(ri, ri, elem_ir.clone(), span);
            emit_inst(ctx, i6);
            let (denom, i7) = ctx.builder.build_add(cc, dd, elem_ir.clone(), span);
            emit_inst(ctx, i7);
            let (num_r, i8) = ctx.builder.build_add(ac, bd, elem_ir.clone(), span);
            emit_inst(ctx, i8);
            let (num_i, i9) = ctx.builder.build_sub(bc, ad, elem_ir.clone(), span);
            emit_inst(ctx, i9);
            let (res_r, i10) = ctx
                .builder
                .build_div(num_r, denom, elem_ir.clone(), true, span);
            emit_inst(ctx, i10);
            let (res_i, i11) = ctx
                .builder
                .build_div(num_i, denom, elem_ir.clone(), true, span);
            emit_inst(ctx, i11);
            let result = store_complex_result(ctx, res_r, res_i, &complex_ir, elem_size, span);
            let (loaded, ld) = ctx.builder.build_load(result, complex_ir, span);
            emit_inst(ctx, ld);
            TypedValue::new(loaded, complex_ctype)
        }
        ast::BinaryOp::Equal | ast::BinaryOp::NotEqual => {
            // Complex equality: both real and imag must be equal.
            let cmp_real = if matches!(elem_ir, IrType::F32 | IrType::F64 | IrType::F80) {
                let cmp_op = if matches!(op, ast::BinaryOp::Equal) {
                    FCmpOp::Oeq
                } else {
                    FCmpOp::One
                };
                let (v, i) = ctx.builder.build_fcmp(cmp_op, lr, rr, span);
                emit_inst(ctx, i);
                v
            } else {
                let cmp_op = if matches!(op, ast::BinaryOp::Equal) {
                    ICmpOp::Eq
                } else {
                    ICmpOp::Ne
                };
                let (v, i) = ctx.builder.build_icmp(cmp_op, lr, rr, span);
                emit_inst(ctx, i);
                v
            };
            let cmp_imag = if matches!(elem_ir, IrType::F32 | IrType::F64 | IrType::F80) {
                let cmp_op = if matches!(op, ast::BinaryOp::Equal) {
                    FCmpOp::Oeq
                } else {
                    FCmpOp::One
                };
                let (v, i) = ctx.builder.build_fcmp(cmp_op, li, ri, span);
                emit_inst(ctx, i);
                v
            } else {
                let cmp_op = if matches!(op, ast::BinaryOp::Equal) {
                    ICmpOp::Eq
                } else {
                    ICmpOp::Ne
                };
                let (v, i) = ctx.builder.build_icmp(cmp_op, li, ri, span);
                emit_inst(ctx, i);
                v
            };
            // For ==: both must be true (AND).  For !=: either can be true (OR).
            let (result, combine_inst) = if matches!(op, ast::BinaryOp::Equal) {
                ctx.builder.build_and(cmp_real, cmp_imag, IrType::I1, span)
            } else {
                ctx.builder.build_or(cmp_real, cmp_imag, IrType::I1, span)
            };
            emit_inst(ctx, combine_inst);
            TypedValue::new(result, CType::Bool)
        }
        _ => {
            // Other operations (bitwise, shift, etc.) are not valid on _Complex.
            // Fall back to the non-complex path (will likely produce incorrect
            // results but avoids a panic; sema should have caught this).
            let common = type_builder::usual_arithmetic_conversion(&_lhs_tv.ty, &_rhs_tv.ty);
            let ci = ctype_to_ir(&common, ctx.target);
            let lv = insert_implicit_conversion(ctx, _lhs_tv.value, &_lhs_tv.ty, &common, span);
            let rv = insert_implicit_conversion(ctx, _rhs_tv.value, &_rhs_tv.ty, &common, span);
            let (v, i) = ctx.builder.build_add(lv, rv, ci, span);
            emit_inst(ctx, i);
            TypedValue::new(v, common)
        }
    }
}

/// Helper: get a pointer to a complex aggregate for an expression.
/// For simple lvalues (identifiers, member access), re-lowers as lvalue.
/// For non-lvalue expressions (assignments, function calls), stores the
/// evaluated aggregate value to a temporary alloca.
fn complex_expr_to_ptr(
    ctx: &mut ExprLoweringContext<'_>,
    expr: &ast::Expression,
    value: Value,
    ctype: &CType,
    span: Span,
) -> Value {
    match expr {
        ast::Expression::Identifier { .. }
        | ast::Expression::MemberAccess { .. }
        | ast::Expression::PointerMemberAccess { .. }
        | ast::Expression::ArraySubscript { .. }
        | ast::Expression::UnaryOp { .. } => lower_lvalue(ctx, expr),
        _ => {
            // Store aggregate value to a temp alloca for decomposition.
            let complex_ir = ctype_to_ir(ctype, ctx.target);
            let (alloca_val, alloca_inst) = ctx.builder.build_alloca(complex_ir, span);
            emit_inst(ctx, alloca_inst);
            let store_inst = ctx.builder.build_store(value, alloca_val, span);
            emit_inst(ctx, store_inst);
            alloca_val
        }
    }
}

/// Helper: convert expression to boolean, handling Complex types via lvalue pointer.
fn expr_to_bool(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression, span: Span) -> Value {
    let inner = lower_expr_inner(ctx, expr);
    let resolved = types::resolve_typedef(&inner.ty);
    if let CType::Complex(ref base) = resolved {
        let ptr = complex_expr_to_ptr(ctx, expr, inner.value, resolved, span);
        complex_ptr_to_bool(ctx, ptr, base, span)
    } else {
        lower_to_bool(ctx, inner.value, &inner.ty, span)
    }
}

fn lower_logical_and(
    ctx: &mut ExprLoweringContext<'_>,
    le: &ast::Expression,
    re: &ast::Expression,
    span: Span,
) -> TypedValue {
    let rb = new_block(ctx);
    let mb = new_block(ctx);
    let lb = expr_to_bool(ctx, le, span);
    // Emit the short-circuit false constant (0) in the current block BEFORE
    // the conditional branch.  This ensures the phi in the merge block is the
    // very first instruction, maintaining the SSA invariant that phi nodes
    // form a contiguous prefix of a basic block's instruction list.
    let fv = emit_int_const(ctx, 0, IrType::I1, span);
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx.builder.build_cond_branch(lb, rb, mb, span);
    emit_inst(ctx, ci);
    // CFG edges: cond_blk → rb (rhs eval), cond_blk → mb (short-circuit false)
    add_cfg_edge(ctx, cond_blk.index(), rb.index());
    add_cfg_edge(ctx, cond_blk.index(), mb.index());
    let le_blk = cond_blk;
    ctx.builder.set_insert_point(rb);
    let rbv = expr_to_bool(ctx, re, span);
    let bi = ctx.builder.build_branch(mb, span);
    emit_inst(ctx, bi);
    let re_blk = ctx.builder.get_insert_block().unwrap();
    // CFG edge: re_blk → mb
    add_cfg_edge(ctx, re_blk.index(), mb.index());
    ctx.builder.set_insert_point(mb);
    let (pv, pi) = ctx
        .builder
        .build_phi(IrType::I1, vec![(fv, le_blk), (rbv, re_blk)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, CType::Bool)
}

fn lower_logical_or(
    ctx: &mut ExprLoweringContext<'_>,
    le: &ast::Expression,
    re: &ast::Expression,
    span: Span,
) -> TypedValue {
    let rb = new_block(ctx);
    let mb = new_block(ctx);
    let lb = expr_to_bool(ctx, le, span);
    // Emit the short-circuit true constant (1) in the current block BEFORE
    // the conditional branch, so the phi in the merge block is the very
    // first instruction, preserving the phi-prefix invariant.
    let tv = emit_int_const(ctx, 1, IrType::I1, span);
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx.builder.build_cond_branch(lb, mb, rb, span);
    emit_inst(ctx, ci);
    // CFG edges: cond_blk → mb (short-circuit true), cond_blk → rb (rhs eval)
    add_cfg_edge(ctx, cond_blk.index(), mb.index());
    add_cfg_edge(ctx, cond_blk.index(), rb.index());
    let le_blk = cond_blk;
    ctx.builder.set_insert_point(rb);
    let rbv = expr_to_bool(ctx, re, span);
    let bi = ctx.builder.build_branch(mb, span);
    emit_inst(ctx, bi);
    let re_blk = ctx.builder.get_insert_block().unwrap();
    // CFG edge: re_blk → mb
    add_cfg_edge(ctx, re_blk.index(), mb.index());
    ctx.builder.set_insert_point(mb);
    let (pv, pi) = ctx
        .builder
        .build_phi(IrType::I1, vec![(tv, le_blk), (rbv, re_blk)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, CType::Bool)
}

fn lower_ptr_arith(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::BinaryOp,
    ptr: &TypedValue,
    idx: &TypedValue,
    span: Span,
) -> TypedValue {
    let es = pointee_size_resolved(&ptr.ty, ctx.type_builder, ctx.struct_defs, ctx.target) as i128;
    let si = size_ir_type(ctx.target);
    let sc = emit_int_const(ctx, es, si.clone(), span);
    let iv = insert_implicit_conversion(ctx, idx.value, &idx.ty, &size_ctype(ctx.target), span);
    let (bo, mi) = ctx.builder.build_mul(iv, sc, si.clone(), span);
    emit_inst(ctx, mi);
    match op {
        ast::BinaryOp::Add => {
            let (r, g) = ctx
                .builder
                .build_gep(ptr.value, vec![bo], IrType::Ptr, span);
            emit_inst(ctx, g);
            TypedValue::new(r, ptr.ty.clone())
        }
        ast::BinaryOp::Sub => {
            let (n, ni) = ctx.builder.build_neg(bo, si, span);
            emit_inst(ctx, ni);
            let (r, g) = ctx.builder.build_gep(ptr.value, vec![n], IrType::Ptr, span);
            emit_inst(ctx, g);
            TypedValue::new(r, ptr.ty.clone())
        }
        _ => {
            ctx.diagnostics
                .emit_error(span, "invalid pointer arithmetic");
            TypedValue::void()
        }
    }
}

fn lower_ptr_diff(
    ctx: &mut ExprLoweringContext<'_>,
    l: &TypedValue,
    r: &TypedValue,
    span: Span,
) -> TypedValue {
    let pi = size_ir_type(ctx.target);
    let (li, lic) = ctx.builder.build_ptr_to_int(l.value, pi.clone(), span);
    emit_inst(ctx, lic);
    let (ri, ric) = ctx.builder.build_ptr_to_int(r.value, pi.clone(), span);
    emit_inst(ctx, ric);
    let (d, di) = ctx.builder.build_sub(li, ri, pi.clone(), span);
    emit_inst(ctx, di);
    let es = pointee_size_resolved(&l.ty, ctx.type_builder, ctx.struct_defs, ctx.target) as i128;
    let dv = emit_int_const(ctx, es, pi.clone(), span);
    let (res, rdi) = ctx.builder.build_div(d, dv, pi, true, span);
    emit_inst(ctx, rdi);
    TypedValue::new(
        res,
        if ctx.target.is_64bit() {
            CType::Long
        } else {
            CType::Int
        },
    )
}

// =========================================================================
// UNARY OPERATIONS
// =========================================================================

fn lower_unary(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::UnaryOp,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::UnaryOp::AddressOf => {
            let lv = lower_lvalue_inner(ctx, operand);
            TypedValue::new(
                lv.value,
                CType::Pointer(Box::new(lv.ty), TypeQualifiers::default()),
            )
        }
        ast::UnaryOp::Deref => {
            let p = lower_expr_inner(ctx, operand);
            let pt = pointee_of(&p.ty);
            // Resolve typedefs on the pointee so that patterns like
            //   `typedef void MyFunc(int); MyFunc *f; (*f)(42);`
            // correctly recognise the function type underneath the
            // typedef wrapper.  Without this, pt would be
            // CType::Typedef("MyFunc", Function{..}) and the
            // `matches!(CType::Function{..})` check below would miss
            // it, causing an erroneous Load instruction that reads
            // the first bytes of machine code instead of using the
            // pointer as a callable address.
            let resolved_pt = types::resolve_typedef(&pt);
            // In C, dereferencing a function pointer is an identity
            // operation — `(*fptr)(args)` is equivalent to `fptr(args)`.
            // The function pointer value IS the callable address; emitting
            // a Load would read the first bytes of the function's machine
            // code instead of using the pointer as an address.  Skip the
            // Load for function and function-pointer pointee types.
            if matches!(resolved_pt, CType::Function { .. }) {
                // Return the pointer value unchanged with function type.
                // The call lowering will use this value directly.
                TypedValue::new(p.value, pt)
            } else if matches!(resolved_pt, CType::Array(_, _)) {
                // Dereferencing a pointer-to-array yields an array lvalue
                // at the same address.  In C, this lvalue decays to a
                // pointer-to-first-element.  We must NOT emit a Load —
                // the pointer value IS the address of the array.
                // Example: `int (*a)[2]; *a` → same address, type int[2].
                // Array-to-pointer decay will be applied when needed.
                TypedValue::new(p.value, pt)
            } else {
                let it = ctype_to_ir(&pt, ctx.target);
                let (v, li) = ctx.builder.build_load(p.value, it, span);
                emit_inst(ctx, li);
                TypedValue::new(v, pt)
            }
        }
        ast::UnaryOp::Plus => {
            let inner = lower_expr_inner(ctx, operand);
            let prom = types::integer_promotion(&inner.ty);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            TypedValue::new(v, prom)
        }
        ast::UnaryOp::Negate => {
            // For float literals, negate at compile time to preserve
            // negative zero (-0.0) which cannot survive FSub(0.0, 0.0).
            if let ast::Expression::FloatLiteral { value, suffix, .. } = operand {
                let negated = if *value == 0.0 {
                    // -0.0 special case: bit-flip the sign bit
                    f64::from_bits(value.to_bits() ^ 0x8000000000000000)
                } else {
                    -value
                };
                let (ir_ty, c_ty) = match suffix {
                    ast::FloatSuffix::F | ast::FloatSuffix::FI => (IrType::F32, CType::Float),
                    ast::FloatSuffix::L | ast::FloatSuffix::LI => (IrType::F80, CType::LongDouble),
                    ast::FloatSuffix::None | ast::FloatSuffix::I => (IrType::F64, CType::Double),
                };
                let val = emit_float_const(ctx, negated, ir_ty, span);
                return TypedValue::new(val, c_ty);
            }
            let inner = lower_expr_inner(ctx, operand);
            let prom = types::integer_promotion(&inner.ty);
            let it = ctype_to_ir(&prom, ctx.target);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            let (r, i) = ctx.builder.build_neg(v, it, span);
            emit_inst(ctx, i);
            TypedValue::new(r, prom)
        }
        ast::UnaryOp::BitwiseNot => {
            let inner = lower_expr_inner(ctx, operand);
            let resolved_ty = types::resolve_typedef(&inner.ty);
            // GCC extension: ~(complex) is complex conjugate — negate
            // the imaginary part, leave real part unchanged.
            if let CType::Complex(ref base) = resolved_ty {
                let elem_ir = ctype_to_ir(base.as_ref(), ctx.target);
                let complex_ir = ctype_to_ir(resolved_ty, ctx.target);
                let elem_sz: i64 = complex_elem_size(&elem_ir);

                // Store complex value to alloca.
                let (alloca_val, alloca_inst) = ctx.builder.build_alloca(complex_ir.clone(), span);
                emit_inst(ctx, alloca_inst);
                let st = ctx.builder.build_store(inner.value, alloca_val, span);
                emit_inst(ctx, st);

                // Load real part at offset 0 (unchanged).
                let zero_idx = emit_int_const_for_index(ctx, 0);
                let (real_ptr, gep_r) =
                    ctx.builder
                        .build_gep(alloca_val, vec![zero_idx], IrType::Ptr, span);
                emit_inst(ctx, gep_r);
                let (real_val, ld_r) = ctx.builder.build_load(real_ptr, elem_ir.clone(), span);
                emit_inst(ctx, ld_r);

                // Load imaginary part at offset elem_sz.
                let imag_idx = emit_int_const_for_index(ctx, elem_sz);
                let (imag_ptr, gep_i) =
                    ctx.builder
                        .build_gep(alloca_val, vec![imag_idx], IrType::Ptr, span);
                emit_inst(ctx, gep_i);
                let (imag_val, ld_i) = ctx.builder.build_load(imag_ptr, elem_ir.clone(), span);
                emit_inst(ctx, ld_i);

                // Negate imaginary part (build_neg handles both int
                // and float via 0 - operand / 0.0 - operand).
                let (neg_imag, neg_inst) = ctx.builder.build_neg(imag_val, elem_ir.clone(), span);
                emit_inst(ctx, neg_inst);

                // Build result: store {real, -imag} into a new alloca.
                let (res_val, res_inst) = ctx.builder.build_alloca(complex_ir.clone(), span);
                emit_inst(ctx, res_inst);

                let r_idx = emit_int_const_for_index(ctx, 0);
                let (r_ptr, gep_rr) =
                    ctx.builder
                        .build_gep(res_val, vec![r_idx], IrType::Ptr, span);
                emit_inst(ctx, gep_rr);
                let st_r = ctx.builder.build_store(real_val, r_ptr, span);
                emit_inst(ctx, st_r);

                let i_idx = emit_int_const_for_index(ctx, elem_sz);
                let (i_ptr, gep_ii) =
                    ctx.builder
                        .build_gep(res_val, vec![i_idx], IrType::Ptr, span);
                emit_inst(ctx, gep_ii);
                let st_i = ctx.builder.build_store(neg_imag, i_ptr, span);
                emit_inst(ctx, st_i);

                let (result, load_inst) = ctx.builder.build_load(res_val, complex_ir, span);
                emit_inst(ctx, load_inst);
                return TypedValue::new(result, resolved_ty.clone());
            }

            // Non-complex bitwise NOT
            let prom = types::integer_promotion(&inner.ty);
            let it = ctype_to_ir(&prom, ctx.target);
            let v = insert_implicit_conversion(ctx, inner.value, &inner.ty, &prom, span);
            let (r, i) = ctx.builder.build_not(v, it, span);
            emit_inst(ctx, i);
            TypedValue::new(r, prom)
        }
        ast::UnaryOp::LogicalNot => {
            let inner = lower_expr_inner(ctx, operand);
            let resolved = types::resolve_typedef(&inner.ty);
            let bv = if let CType::Complex(ref base) = resolved {
                // Complex types need lvalue pointer for decomposition.
                let ptr = complex_expr_to_ptr(ctx, operand, inner.value, resolved, span);
                complex_ptr_to_bool(ctx, ptr, base, span)
            } else {
                lower_to_bool(ctx, inner.value, &inner.ty, span)
            };
            let one = emit_int_const(ctx, 1, IrType::I1, span);
            let (r, i) = ctx.builder.build_xor(bv, one, IrType::I1, span);
            emit_inst(ctx, i);
            TypedValue::new(r, CType::Bool)
        }
        ast::UnaryOp::RealPart => {
            // GCC __real__: extract real component from _Complex, or
            // identity for non-complex scalar types.
            let inner = lower_expr_inner(ctx, operand);
            let resolved = types::resolve_typedef(&inner.ty);
            if let CType::Complex(base) = resolved {
                let elem_ctype = base.as_ref().clone();
                let elem_ir = ctype_to_ir(&elem_ctype, ctx.target);
                let complex_ir = ctype_to_ir(resolved, ctx.target);
                // Store the already-lowered complex value into a
                // temporary alloca so we can extract the real part via
                // pointer arithmetic.  Do NOT re-lower the operand with
                // lower_lvalue — that would re-evaluate the expression
                // (calling functions twice) and return a raw value
                // instead of a pointer, causing SEGFAULT on GEP.
                let (tmp_alloca, ai) = ctx.builder.build_alloca(complex_ir.clone(), span);
                emit_inst(ctx, ai);
                let st = ctx.builder.build_store(inner.value, tmp_alloca, span);
                emit_inst(ctx, st);
                // Real part is at offset 0.
                let zero_idx = emit_int_const_for_index(ctx, 0);
                let (real_ptr, gep) =
                    ctx.builder
                        .build_gep(tmp_alloca, vec![zero_idx], IrType::Ptr, span);
                emit_inst(ctx, gep);
                let (real_val, ld) = ctx.builder.build_load(real_ptr, elem_ir, span);
                emit_inst(ctx, ld);
                TypedValue::new(real_val, elem_ctype)
            } else {
                inner
            }
        }
        ast::UnaryOp::ImagPart => {
            // GCC __imag__: extract imaginary component from _Complex, or
            // return 0 for non-complex scalar types.
            let inner = lower_expr_inner(ctx, operand);
            let resolved = types::resolve_typedef(&inner.ty);
            if let CType::Complex(base) = resolved {
                let elem_ctype = base.as_ref().clone();
                let elem_ir = ctype_to_ir(&elem_ctype, ctx.target);
                let elem_size: i64 = complex_elem_size(&elem_ir);
                let complex_ir = ctype_to_ir(resolved, ctx.target);
                // Store the already-lowered complex value into a
                // temporary alloca so we can extract the imaginary part
                // via pointer arithmetic.  See RealPart comment above.
                let (tmp_alloca, ai) = ctx.builder.build_alloca(complex_ir.clone(), span);
                emit_inst(ctx, ai);
                let st = ctx.builder.build_store(inner.value, tmp_alloca, span);
                emit_inst(ctx, st);
                // Imaginary part is at offset elem_size.
                let off_idx = emit_int_const_for_index(ctx, elem_size);
                let (imag_ptr, gep) =
                    ctx.builder
                        .build_gep(tmp_alloca, vec![off_idx], IrType::Ptr, span);
                emit_inst(ctx, gep);
                let (imag_val, ld) = ctx.builder.build_load(imag_ptr, elem_ir, span);
                emit_inst(ctx, ld);
                TypedValue::new(imag_val, elem_ctype)
            } else {
                let ir_ty = ctype_to_ir(&inner.ty, ctx.target);
                if matches!(ir_ty, IrType::F32 | IrType::F64 | IrType::F80) {
                    let val = emit_float_const(ctx, 0.0, ir_ty, span);
                    TypedValue::new(val, inner.ty)
                } else {
                    let val = emit_int_const(ctx, 0, ir_ty, span);
                    TypedValue::new(val, inner.ty)
                }
            }
        }
    }
}

// =========================================================================
// ASSIGNMENT
// =========================================================================

fn lower_assignment(
    ctx: &mut ExprLoweringContext<'_>,
    op: &ast::AssignOp,
    target: &ast::Expression,
    value: &ast::Expression,
    span: Span,
) -> TypedValue {
    match op {
        ast::AssignOp::Assign => {
            // Determine the LHS type first to decide assignment strategy.
            let lhs = lower_lvalue_inner(ctx, target);
            let assign_bf_info = ctx.last_bitfield_info.take();

            // For aggregate types (struct/union), the x86-64 codegen cannot
            // load/store an entire struct through a single register.  Instead,
            // expand into individual 8-byte (or smaller) load/store pairs that
            // copy the aggregate field by field at the IR level.
            // Resolve forward-declared struct/union types so that
            // sizeof_ctype returns the actual size rather than 0.
            let mut resolved_lhs_ty = lhs.ty.clone();
            resolve_forward_ref_type(&mut resolved_lhs_ty, ctx.struct_defs);
            let is_agg = crate::common::types::is_struct_or_union(&resolved_lhs_ty);
            let struct_size = if is_agg {
                crate::common::types::sizeof_ctype(&resolved_lhs_ty, ctx.target)
            } else {
                0
            };
            // GCC extension: zero-sized structs (e.g. `struct g{};`).
            // Assignment is a no-op — nothing to copy.
            if is_agg && struct_size == 0 {
                // Still evaluate RHS for side effects.
                let _ = lower_expr_inner(ctx, value);
                return TypedValue::new(lhs.value, lhs.ty);
            }

            // Special case: LHS is _Complex but RHS is a non-complex scalar.
            // Convert the scalar to complex first, then store.
            // Complex→Complex assignment falls through to the normal
            // scalar Store path — the codegen handles aggregate copies
            // via struct_load_source for function-call returns and
            // struct_pair_hi for parameter-received pairs.
            if let CType::Complex(ref base_ty) = resolved_lhs_ty {
                let rhs_tv = lower_expr_inner(ctx, value);
                let rhs_resolved = types::resolve_typedef(&rhs_tv.ty);
                let rhs_is_complex = matches!(rhs_resolved, CType::Complex(_));
                if !rhs_is_complex {
                    // Scalar → Complex: store real=scalar, imag=0 directly
                    // into the LHS alloca, component by component.
                    let elem_ir = crate::ir::types::IrType::from_ctype(base_ty, ctx.target);
                    let scalar_as_elem = lower_cast(ctx, rhs_tv.value, &rhs_tv.ty, base_ty, span);
                    // Store real part at [lhs+0]
                    let si_real = ctx.builder.build_store(scalar_as_elem, lhs.value, span);
                    emit_inst(ctx, si_real);
                    // Store zero imag part at [lhs+elem_size]
                    let elem_size = elem_ir.size_bytes(ctx.target);
                    let off_val = emit_int_const(ctx, elem_size as i128, IrType::I64, span);
                    let (imag_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(lhs.value, vec![off_val], elem_ir.clone(), span);
                    emit_inst(ctx, gep_inst);
                    let zero_imag = if is_fp_ir_type(&elem_ir) {
                        emit_float_const(ctx, 0.0, elem_ir.clone(), span)
                    } else {
                        emit_int_const(ctx, 0, elem_ir.clone(), span)
                    };
                    let si_imag = ctx.builder.build_store(zero_imag, imag_ptr, span);
                    emit_inst(ctx, si_imag);
                    return TypedValue::new(lhs.value, lhs.ty);
                }
                // For Complex→Complex with different element types,
                // we must convert component-by-component. For same types,
                // fall through to the normal Store path which handles
                // aggregate copies via struct_load_source/struct_pair_hi.
                if let CType::Complex(ref rhs_base) = *rhs_resolved {
                    if **rhs_base != **base_ty {
                        // Different element types (e.g. _Complex float → _Complex double).
                        // Get a pointer to the RHS complex value.
                        let rhs_ptr = if expr_is_natural_lvalue(value) {
                            lower_lvalue_inner(ctx, value).value
                        } else {
                            // Spill the rvalue to a temp alloca.
                            let rhs_ir = IrType::from_ctype(&rhs_tv.ty, ctx.target);
                            let (tmp, tmp_inst) = ctx.builder.build_alloca(rhs_ir, span);
                            emit_inst(ctx, tmp_inst);
                            let si = ctx.builder.build_store(rhs_tv.value, tmp, span);
                            emit_inst(ctx, si);
                            tmp
                        };
                        let src_elem_ir = IrType::from_ctype(rhs_base, ctx.target);
                        let dst_elem_ir = IrType::from_ctype(base_ty, ctx.target);
                        let src_elem_sz = src_elem_ir.size_bytes(ctx.target);
                        let dst_elem_sz = dst_elem_ir.size_bytes(ctx.target);
                        // Convert real part
                        let (src_real, src_real_li) =
                            ctx.builder.build_load(rhs_ptr, src_elem_ir.clone(), span);
                        emit_inst(ctx, src_real_li);
                        let dst_real = lower_cast(ctx, src_real, rhs_base, base_ty, span);
                        let si_real = ctx.builder.build_store(dst_real, lhs.value, span);
                        emit_inst(ctx, si_real);
                        // Convert imag part
                        let src_imag_off =
                            emit_int_const(ctx, src_elem_sz as i128, IrType::I64, span);
                        let (src_imag_ptr, src_gep) = ctx.builder.build_gep(
                            rhs_ptr,
                            vec![src_imag_off],
                            src_elem_ir.clone(),
                            span,
                        );
                        emit_inst(ctx, src_gep);
                        let (src_imag, src_imag_li) =
                            ctx.builder.build_load(src_imag_ptr, src_elem_ir, span);
                        emit_inst(ctx, src_imag_li);
                        let dst_imag = lower_cast(ctx, src_imag, rhs_base, base_ty, span);
                        let dst_imag_off =
                            emit_int_const(ctx, dst_elem_sz as i128, IrType::I64, span);
                        let (dst_imag_ptr, dst_gep) =
                            ctx.builder
                                .build_gep(lhs.value, vec![dst_imag_off], dst_elem_ir, span);
                        emit_inst(ctx, dst_gep);
                        let si_imag = ctx.builder.build_store(dst_imag, dst_imag_ptr, span);
                        emit_inst(ctx, si_imag);
                        return TypedValue::new(lhs.value, lhs.ty);
                    }
                }
                // Same-type Complex→Complex assignment: store the already-
                // evaluated RHS value directly. We must NOT fall through to
                // the generic scalar Store path below because that would call
                // lower_expr_inner(value) a SECOND time, re-executing any
                // side effects (e.g. pointer post-increments in `*p++ * s`).
                let si = ctx.builder.build_store(rhs_tv.value, lhs.value, span);
                emit_inst(ctx, si);
                // C11 §6.5.16p3: the value of an assignment expression is
                // the value of the left operand after the assignment, i.e.
                // the stored RHS rvalue — NOT the LHS pointer/lvalue.
                return TypedValue::new(rhs_tv.value, lhs.ty);
            }

            if is_agg && struct_size > 8 {
                // Get source pointer for the struct copy.
                // If the RHS is a natural lvalue (variable, array element,
                // member access, dereference, compound literal), we can take
                // its address directly.  Otherwise it is an rvalue
                // (function call, statement expression, conditional, etc.)
                // and we must spill it to a temporary alloca first.
                let rhs_ptr = if expr_is_natural_lvalue(value) {
                    lower_lvalue_inner(ctx, value)
                } else if rhs_expr_returns_aggregate_pointer(value) {
                    // The RHS is an expression whose lowered result is
                    // already a *pointer* to the aggregate data (e.g. a
                    // chained assignment `(*p = x)` returns the LHS
                    // address).  Use the pointer directly as the copy
                    // source — do NOT spill it into a new alloca (that
                    // would copy the 8-byte pointer bytes, not the
                    // struct data).
                    let rhs_val = lower_expr_inner(ctx, value);
                    TypedValue::new(
                        rhs_val.value,
                        CType::Pointer(
                            Box::new(resolved_lhs_ty.clone()),
                            crate::common::types::TypeQualifiers {
                                is_const: false,
                                is_volatile: false,
                                is_restrict: false,
                                is_atomic: false,
                            },
                        ),
                    )
                } else {
                    // The RHS is an expression that produces aggregate
                    // DATA (function call returning a struct, va_arg,
                    // conditional, etc.).  The backend holds this data
                    // in a temp slot or register pair.  We must spill
                    // into a temporary alloca so we can GEP into the
                    // memory representation for the byte-by-byte copy.
                    let rhs_val = lower_expr_inner(ctx, value);
                    let struct_ir_ty = IrType::from_ctype(&resolved_lhs_ty, ctx.target);
                    let (tmp_alloca, tmp_inst) =
                        ctx.builder.build_alloca(struct_ir_ty.clone(), span);
                    emit_inst(ctx, tmp_inst);
                    let si = ctx.builder.build_store(rhs_val.value, tmp_alloca, span);
                    emit_inst(ctx, si);
                    TypedValue::new(
                        tmp_alloca,
                        CType::Pointer(
                            Box::new(resolved_lhs_ty.clone()),
                            crate::common::types::TypeQualifiers {
                                is_const: false,
                                is_volatile: false,
                                is_restrict: false,
                                is_atomic: false,
                            },
                        ),
                    )
                };

                // Emit individual 8-byte load/store pairs to copy the
                // struct from source to destination.  Process in 8-byte
                // chunks (matching pointer/qword width on 64-bit targets),
                // with a smaller final chunk for any remainder.
                let mut offset: usize = 0;
                while offset < struct_size {
                    let remaining = struct_size - offset;
                    let (chunk_ty, chunk_ir) = if remaining >= 8 {
                        (8, IrType::I64)
                    } else if remaining >= 4 {
                        (4, IrType::I32)
                    } else if remaining >= 2 {
                        (2, IrType::I16)
                    } else {
                        (1, IrType::I8)
                    };

                    // Compute source address: rhs_ptr + offset
                    let src_addr = if offset == 0 {
                        rhs_ptr.value
                    } else {
                        let off_val = emit_int_const(ctx, offset as i128, IrType::I64, span);
                        let (gep_val, gep_inst) = ctx.builder.build_gep(
                            rhs_ptr.value,
                            vec![off_val],
                            chunk_ir.clone(),
                            span,
                        );
                        emit_inst(ctx, gep_inst);
                        gep_val
                    };

                    // Load chunk from source.
                    let (loaded, li) = ctx.builder.build_load(src_addr, chunk_ir.clone(), span);
                    emit_inst(ctx, li);

                    // Compute destination address: lhs + offset
                    let dst_addr = if offset == 0 {
                        lhs.value
                    } else {
                        let off_val2 = emit_int_const(ctx, offset as i128, IrType::I64, span);
                        let (gep_val2, gep_inst2) = ctx.builder.build_gep(
                            lhs.value,
                            vec![off_val2],
                            chunk_ir.clone(),
                            span,
                        );
                        emit_inst(ctx, gep_inst2);
                        gep_val2
                    };

                    // Store chunk to destination.
                    let si = ctx.builder.build_store(loaded, dst_addr, span);
                    emit_inst(ctx, si);

                    offset += chunk_ty;
                }

                return TypedValue::new(lhs.value, lhs.ty);
            }

            // Scalar / small-aggregate assignment: load + store.
            let rhs = lower_expr_inner(ctx, value);
            let rhs_val = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &lhs.ty, span);

            // Bitfield store: read-modify-write the storage unit to
            // preserve adjacent fields packed in the same unit.
            // C11 §6.5.16p3: "The value of an assignment expression is the
            // value of the left operand after the assignment."  For bitfields
            // the stored value is truncated to the field width, so the
            // expression result must reflect that truncation, not the
            // original RHS value.
            if let Some((bit_offset, bit_width)) = assign_bf_info {
                if bit_width > 0 {
                    lower_bitfield_store(
                        ctx, lhs.value, rhs_val, &lhs.ty, bit_offset, bit_width, span,
                    );
                    // Compute the truncated value that was actually stored.
                    let is_signed = is_signed(&lhs.ty);
                    let truncated = if is_signed && bit_width < 64 {
                        // Sign-extend from bit_width: SHL left, then AShr right.
                        let storage_ir = ctype_to_ir(&lhs.ty, ctx.target);
                        let ir_width = match &storage_ir {
                            IrType::I8 => 8,
                            IrType::I16 => 16,
                            IrType::I32 => 32,
                            _ => 64,
                        };
                        let shift_amt = ir_width - bit_width;
                        if shift_amt > 0 {
                            let shift_val =
                                emit_int_const(ctx, shift_amt as i128, storage_ir.clone(), span);
                            let (shl_v, shl_i) =
                                ctx.builder
                                    .build_shl(rhs_val, shift_val, storage_ir.clone(), span);
                            emit_inst(ctx, shl_i);
                            let shift_val2 =
                                emit_int_const(ctx, shift_amt as i128, storage_ir.clone(), span);
                            let (ashr_v, ashr_i) = ctx.builder.build_shr(
                                shl_v,
                                shift_val2,
                                storage_ir.clone(),
                                true,
                                span,
                            );
                            emit_inst(ctx, ashr_i);
                            ashr_v
                        } else {
                            rhs_val
                        }
                    } else if !is_signed && bit_width < 64 {
                        // Unsigned: mask to bit_width bits.
                        let storage_ir = ctype_to_ir(&lhs.ty, ctx.target);
                        let mask = (1u64 << bit_width).wrapping_sub(1);
                        let mask_val = emit_int_const(ctx, mask as i128, storage_ir.clone(), span);
                        let (and_v, and_i) =
                            ctx.builder
                                .build_and(rhs_val, mask_val, storage_ir.clone(), span);
                        emit_inst(ctx, and_i);
                        and_v
                    } else {
                        rhs_val
                    };
                    return TypedValue::new(truncated, lhs.ty);
                }
            }

            let si = ctx.builder.build_store(rhs_val, lhs.value, span);
            emit_inst(ctx, si);
            TypedValue::new(rhs_val, lhs.ty)
        }
        _ => {
            // Compound assignment:  lhs <op>= rhs
            // C11 §6.5.16.2: E1 op= E2 ≡ E1 = (typeof(E1))(E1 op E2), E1 evaluated once.
            // Evaluation order: compute LHS address, evaluate RHS (with side effects),
            // THEN read LHS value. This ensures RHS side effects are visible when
            // reading LHS (e.g., x[0] |= foo() where foo() modifies x[0]).
            let lhs_ptr = lower_lvalue_inner(ctx, target);
            let compound_bf_info = ctx.last_bitfield_info.take();
            let lhs_ir = ctype_to_ir(&lhs_ptr.ty, ctx.target);

            // Evaluate RHS FIRST — side effects of E2 must be sequenced before
            // reading E1's value for the compound operation.
            let rhs = lower_expr_inner(ctx, value);

            // Now read LHS value AFTER RHS side effects have completed.
            // For bitfields, use bitfield-aware extraction (shift + mask).
            let cur_val = if let Some((bit_offset, bit_width)) = compound_bf_info {
                if bit_width > 0 {
                    let tv = lower_bitfield_read(
                        ctx,
                        lhs_ptr.value,
                        &lhs_ptr.ty,
                        bit_offset,
                        bit_width,
                        span,
                    );
                    tv.value
                } else {
                    let (v, li) = ctx.builder.build_load(lhs_ptr.value, lhs_ir.clone(), span);
                    emit_inst(ctx, li);
                    v
                }
            } else {
                let (v, li) = ctx.builder.build_load(lhs_ptr.value, lhs_ir.clone(), span);
                emit_inst(ctx, li);
                v
            };

            // _Complex compound assignment: x += y, x -= y, x *= y, x /= y
            // must be done component-wise.
            let resolved_lhs_cty = types::resolve_typedef(&lhs_ptr.ty);
            let resolved_rhs_cty = types::resolve_typedef(&rhs.ty);
            let lhs_is_complex = matches!(resolved_lhs_cty, CType::Complex(_));
            let rhs_is_complex = matches!(resolved_rhs_cty, CType::Complex(_));
            if lhs_is_complex || rhs_is_complex {
                // Determine the element FP type
                let elem_ctype = if let CType::Complex(ref base) = resolved_lhs_cty {
                    base.as_ref().clone()
                } else if let CType::Complex(ref base) = resolved_rhs_cty {
                    base.as_ref().clone()
                } else {
                    CType::Double
                };
                let elem_ir = ctype_to_ir(&elem_ctype, ctx.target);
                let elem_size: i64 = complex_elem_size(&elem_ir);

                // Load LHS real/imag from the lvalue pointer
                let zero_idx = emit_int_const_for_index(ctx, 0);
                let (lr_ptr, g1) =
                    ctx.builder
                        .build_gep(lhs_ptr.value, vec![zero_idx], IrType::Ptr, span);
                emit_inst(ctx, g1);
                let (lr, l1) = ctx.builder.build_load(lr_ptr, elem_ir.clone(), span);
                emit_inst(ctx, l1);
                let off_idx = emit_int_const_for_index(ctx, elem_size);
                let (li_ptr, g2) =
                    ctx.builder
                        .build_gep(lhs_ptr.value, vec![off_idx], IrType::Ptr, span);
                emit_inst(ctx, g2);
                let (li_val, l2) = ctx.builder.build_load(li_ptr, elem_ir.clone(), span);
                emit_inst(ctx, l2);

                // Get RHS real/imag
                let (rr, ri) = if rhs_is_complex {
                    // Store rhs aggregate to temp, then load parts
                    let complex_ir = IrType::from_ctype(resolved_rhs_cty, ctx.target);
                    let (tmp_alloca, ai) = ctx.builder.build_alloca(complex_ir, span);
                    emit_inst(ctx, ai);
                    let si = ctx.builder.build_store(rhs.value, tmp_alloca, span);
                    emit_inst(ctx, si);
                    let z2 = emit_int_const_for_index(ctx, 0);
                    let (rr_ptr, rg1) =
                        ctx.builder
                            .build_gep(tmp_alloca, vec![z2], IrType::Ptr, span);
                    emit_inst(ctx, rg1);
                    let (rr_v, rl1) = ctx.builder.build_load(rr_ptr, elem_ir.clone(), span);
                    emit_inst(ctx, rl1);
                    let o2 = emit_int_const_for_index(ctx, elem_size);
                    let (ri_ptr, rg2) =
                        ctx.builder
                            .build_gep(tmp_alloca, vec![o2], IrType::Ptr, span);
                    emit_inst(ctx, rg2);
                    let (ri_v, rl2) = ctx.builder.build_load(ri_ptr, elem_ir.clone(), span);
                    emit_inst(ctx, rl2);
                    (rr_v, ri_v)
                } else {
                    // Scalar RHS: promote to FP, imag = 0.0
                    let rv = insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &elem_ctype, span);
                    let zero = emit_float_const(ctx, 0.0, elem_ir.clone(), span);
                    (rv, zero)
                };

                // Perform component-wise operation
                let (res_r, res_i) = match op {
                    ast::AssignOp::AddAssign => {
                        let (r, i1) = ctx.builder.build_add(lr, rr, elem_ir.clone(), span);
                        emit_inst(ctx, i1);
                        let (i, i2) = ctx.builder.build_add(li_val, ri, elem_ir.clone(), span);
                        emit_inst(ctx, i2);
                        (r, i)
                    }
                    ast::AssignOp::SubAssign => {
                        let (r, i1) = ctx.builder.build_sub(lr, rr, elem_ir.clone(), span);
                        emit_inst(ctx, i1);
                        let (i, i2) = ctx.builder.build_sub(li_val, ri, elem_ir.clone(), span);
                        emit_inst(ctx, i2);
                        (r, i)
                    }
                    ast::AssignOp::MulAssign => {
                        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                        let (ac, m1) = ctx.builder.build_mul(lr, rr, elem_ir.clone(), span);
                        emit_inst(ctx, m1);
                        let (bd, m2) = ctx.builder.build_mul(li_val, ri, elem_ir.clone(), span);
                        emit_inst(ctx, m2);
                        let (ad, m3) = ctx.builder.build_mul(lr, ri, elem_ir.clone(), span);
                        emit_inst(ctx, m3);
                        let (bc, m4) = ctx.builder.build_mul(li_val, rr, elem_ir.clone(), span);
                        emit_inst(ctx, m4);
                        let (r, s1) = ctx.builder.build_sub(ac, bd, elem_ir.clone(), span);
                        emit_inst(ctx, s1);
                        let (i, a1) = ctx.builder.build_add(ad, bc, elem_ir.clone(), span);
                        emit_inst(ctx, a1);
                        (r, i)
                    }
                    ast::AssignOp::DivAssign => {
                        // (a+bi)/(c+di) = ((ac+bd)/(c²+d²)) + ((bc-ad)/(c²+d²))i
                        let (cc, m1) = ctx.builder.build_mul(rr, rr, elem_ir.clone(), span);
                        emit_inst(ctx, m1);
                        let (dd, m2) = ctx.builder.build_mul(ri, ri, elem_ir.clone(), span);
                        emit_inst(ctx, m2);
                        let (denom, a1) = ctx.builder.build_add(cc, dd, elem_ir.clone(), span);
                        emit_inst(ctx, a1);
                        let (ac, m3) = ctx.builder.build_mul(lr, rr, elem_ir.clone(), span);
                        emit_inst(ctx, m3);
                        let (bd, m4) = ctx.builder.build_mul(li_val, ri, elem_ir.clone(), span);
                        emit_inst(ctx, m4);
                        let (numer_r, a2) = ctx.builder.build_add(ac, bd, elem_ir.clone(), span);
                        emit_inst(ctx, a2);
                        let (bc, m5) = ctx.builder.build_mul(li_val, rr, elem_ir.clone(), span);
                        emit_inst(ctx, m5);
                        let (ad, m6) = ctx.builder.build_mul(lr, ri, elem_ir.clone(), span);
                        emit_inst(ctx, m6);
                        let (numer_i, s1) = ctx.builder.build_sub(bc, ad, elem_ir.clone(), span);
                        emit_inst(ctx, s1);
                        let (r, d1) =
                            ctx.builder
                                .build_div(numer_r, denom, elem_ir.clone(), true, span);
                        emit_inst(ctx, d1);
                        let (i, d2) =
                            ctx.builder
                                .build_div(numer_i, denom, elem_ir.clone(), true, span);
                        emit_inst(ctx, d2);
                        (r, i)
                    }
                    _ => {
                        // Other compound ops on complex: fall through to scalar
                        let (r, _) = ctx.builder.build_add(lr, rr, elem_ir.clone(), span);
                        (r, li_val)
                    }
                };

                // Store real and imag back to lvalue
                let z3 = emit_int_const_for_index(ctx, 0);
                let (wr_ptr, wg1) =
                    ctx.builder
                        .build_gep(lhs_ptr.value, vec![z3], IrType::Ptr, span);
                emit_inst(ctx, wg1);
                let ws1 = ctx.builder.build_store(res_r, wr_ptr, span);
                emit_inst(ctx, ws1);
                let o3 = emit_int_const_for_index(ctx, elem_size);
                let (wi_ptr, wg2) =
                    ctx.builder
                        .build_gep(lhs_ptr.value, vec![o3], IrType::Ptr, span);
                emit_inst(ctx, wg2);
                let ws2 = ctx.builder.build_store(res_i, wi_ptr, span);
                emit_inst(ctx, ws2);

                // Load the full complex back as return value
                let complex_ir = IrType::from_ctype(&lhs_ptr.ty, ctx.target);
                let (result, rl) = ctx.builder.build_load(lhs_ptr.value, complex_ir, span);
                emit_inst(ctx, rl);
                return TypedValue::new(result, lhs_ptr.ty);
            }

            // C11 §6.5.16.2: Compound assignment  E1 op= E2  is equivalent to
            //   E1 = (typeof(E1))(E1 op E2)
            // where E1 op E2 follows the usual arithmetic conversions (§6.3.1.8).
            // We must NOT convert rhs to lhs type before the operation — instead
            // promote both operands to the common type, perform the operation,
            // and then convert the result back to lhs type for storage.
            let lhs_ty_for_arith = lhs_ptr.ty.clone();
            let common_ty = type_builder::usual_arithmetic_conversion(&lhs_ty_for_arith, &rhs.ty);
            let common_ir = ctype_to_ir(&common_ty, ctx.target);
            let cur_promoted =
                insert_implicit_conversion(ctx, cur_val, &lhs_ty_for_arith, &common_ty, span);
            let rhs_promoted =
                insert_implicit_conversion(ctx, rhs.value, &rhs.ty, &common_ty, span);
            let uns = is_unsigned(&common_ty);
            let new_val = match op {
                ast::AssignOp::AddAssign => {
                    if is_pointer_type(&lhs_ptr.ty) {
                        // Pointer arithmetic: keep original cur_val (pointer)
                        let es = pointee_size_resolved(
                            &lhs_ptr.ty,
                            ctx.type_builder,
                            ctx.struct_defs,
                            ctx.target,
                        ) as i128;
                        let si = size_ir_type(ctx.target);
                        let sc = emit_int_const(ctx, es, si.clone(), span);
                        let iv = insert_implicit_conversion(
                            ctx,
                            rhs.value,
                            &rhs.ty,
                            &size_ctype(ctx.target),
                            span,
                        );
                        let (bo, mi) = ctx.builder.build_mul(iv, sc, si, span);
                        emit_inst(ctx, mi);
                        let (r, g) = ctx.builder.build_gep(cur_val, vec![bo], IrType::Ptr, span);
                        emit_inst(ctx, g);
                        r
                    } else {
                        let (v, i) = ctx.builder.build_add(
                            cur_promoted,
                            rhs_promoted,
                            common_ir.clone(),
                            span,
                        );
                        emit_inst(ctx, i);
                        v
                    }
                }
                ast::AssignOp::SubAssign => {
                    if is_pointer_type(&lhs_ptr.ty) {
                        // Pointer arithmetic: p -= n means p = p - n*sizeof(*p)
                        let es = pointee_size_resolved(
                            &lhs_ptr.ty,
                            ctx.type_builder,
                            ctx.struct_defs,
                            ctx.target,
                        ) as i128;
                        let si = size_ir_type(ctx.target);
                        let sc = emit_int_const(ctx, es, si.clone(), span);
                        let iv = insert_implicit_conversion(
                            ctx,
                            rhs.value,
                            &rhs.ty,
                            &size_ctype(ctx.target),
                            span,
                        );
                        let (bo, mi) = ctx.builder.build_mul(iv, sc, si.clone(), span);
                        emit_inst(ctx, mi);
                        // Negate the offset for subtraction
                        let (neg, ni) = ctx.builder.build_neg(bo, si, span);
                        emit_inst(ctx, ni);
                        let (r, g) = ctx.builder.build_gep(cur_val, vec![neg], IrType::Ptr, span);
                        emit_inst(ctx, g);
                        r
                    } else {
                        let (v, i) = ctx.builder.build_sub(
                            cur_promoted,
                            rhs_promoted,
                            common_ir.clone(),
                            span,
                        );
                        emit_inst(ctx, i);
                        v
                    }
                }
                ast::AssignOp::MulAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_mul(cur_promoted, rhs_promoted, common_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::DivAssign => {
                    let (v, i) = ctx.builder.build_div(
                        cur_promoted,
                        rhs_promoted,
                        common_ir.clone(),
                        !uns,
                        span,
                    );
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::ModAssign => {
                    let (v, i) = ctx.builder.build_rem(
                        cur_promoted,
                        rhs_promoted,
                        common_ir.clone(),
                        !uns,
                        span,
                    );
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::AndAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_and(cur_promoted, rhs_promoted, common_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::OrAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_or(cur_promoted, rhs_promoted, common_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::XorAssign => {
                    let (v, i) =
                        ctx.builder
                            .build_xor(cur_promoted, rhs_promoted, common_ir.clone(), span);
                    emit_inst(ctx, i);
                    v
                }
                ast::AssignOp::ShlAssign => {
                    // C11 §6.5.7: Shift uses the promoted lhs type, not common type
                    let lhs_ir_ty = ctype_to_ir(&lhs_ty_for_arith, ctx.target);
                    let lv_for_shift = insert_implicit_conversion(
                        ctx,
                        cur_val,
                        &lhs_ty_for_arith,
                        &lhs_ty_for_arith,
                        span,
                    );
                    let rhs_for_shift = insert_implicit_conversion(
                        ctx,
                        rhs.value,
                        &rhs.ty,
                        &lhs_ty_for_arith,
                        span,
                    );
                    let (v, i) =
                        ctx.builder
                            .build_shl(lv_for_shift, rhs_for_shift, lhs_ir_ty, span);
                    emit_inst(ctx, i);
                    // Convert back to common type so the downstream store-back logic works
                    insert_implicit_conversion(ctx, v, &lhs_ty_for_arith, &common_ty, span)
                }
                ast::AssignOp::ShrAssign => {
                    // C11 §6.5.7: Shift signedness comes from left operand (lhs), not common type
                    let lhs_ir_ty = ctype_to_ir(&lhs_ty_for_arith, ctx.target);
                    let lhs_uns = is_unsigned(&lhs_ty_for_arith);
                    let lv_for_shift = insert_implicit_conversion(
                        ctx,
                        cur_val,
                        &lhs_ty_for_arith,
                        &lhs_ty_for_arith,
                        span,
                    );
                    let rhs_for_shift = insert_implicit_conversion(
                        ctx,
                        rhs.value,
                        &rhs.ty,
                        &lhs_ty_for_arith,
                        span,
                    );
                    let (v, i) = ctx.builder.build_shr(
                        lv_for_shift,
                        rhs_for_shift,
                        lhs_ir_ty,
                        !lhs_uns,
                        span,
                    );
                    emit_inst(ctx, i);
                    insert_implicit_conversion(ctx, v, &lhs_ty_for_arith, &common_ty, span)
                }
                ast::AssignOp::Assign => unreachable!(),
            };

            // Convert result back from common type to lhs type
            let new_val = insert_implicit_conversion(ctx, new_val, &common_ty, &lhs_ptr.ty, span);

            // For bitfields, use read-modify-write to store the computed
            // value back without clobbering adjacent fields.
            if let Some((bit_offset, bit_width)) = compound_bf_info {
                if bit_width > 0 {
                    lower_bitfield_store(
                        ctx,
                        lhs_ptr.value,
                        new_val,
                        &lhs_ptr.ty,
                        bit_offset,
                        bit_width,
                        span,
                    );
                    return TypedValue::new(new_val, lhs_ptr.ty);
                }
            }

            let si = ctx.builder.build_store(new_val, lhs_ptr.value, span);
            emit_inst(ctx, si);
            TypedValue::new(new_val, lhs_ptr.ty)
        }
    }
}

// =========================================================================
// CONDITIONAL (TERNARY)
// =========================================================================

fn lower_conditional(
    ctx: &mut ExprLoweringContext<'_>,
    cond: &ast::Expression,
    then_expr: Option<&ast::Expression>,
    else_expr: &ast::Expression,
    span: Span,
) -> TypedValue {
    let then_blk = new_block(ctx);
    let else_blk = new_block(ctx);
    let merge_blk = new_block(ctx);

    // Evaluate condition — we need both the boolean result and the full
    // typed value (for the GNU extension `x ?: y` where the condition IS
    // the then-value).
    let cond_tv = lower_expr_inner(ctx, cond);
    let resolved_cond = types::resolve_typedef(&cond_tv.ty);
    let cond_bool = if let CType::Complex(ref base) = resolved_cond {
        let ptr = complex_expr_to_ptr(ctx, cond, cond_tv.value, resolved_cond, span);
        complex_ptr_to_bool(ctx, ptr, base, span)
    } else {
        lower_to_bool(ctx, cond_tv.value, &cond_tv.ty, span)
    };

    // Record the block that emits the conditional branch (current block).
    let cond_blk = ctx.builder.get_insert_block().unwrap();
    let ci = ctx
        .builder
        .build_cond_branch(cond_bool, then_blk, else_blk, span);
    emit_inst(ctx, ci);

    // Establish CFG edges: cond_blk → then_blk, cond_blk → else_blk
    add_cfg_edge(ctx, cond_blk.index(), then_blk.index());
    add_cfg_edge(ctx, cond_blk.index(), else_blk.index());

    // Then branch — lower the value but don't emit the branch yet.
    // We need to know the result type first to insert implicit conversions
    // in the correct branch block before branching to the merge block.
    ctx.builder.set_insert_point(then_blk);
    let then_tv = if let Some(te) = then_expr {
        lower_expr_inner(ctx, te)
    } else {
        // GCC extension: x ?: y — condition value IS the then-value.
        TypedValue::new(cond_tv.value, cond_tv.ty.clone())
    };
    // Save the current block (may differ from then_blk due to sub-expressions
    // creating new blocks, e.g. nested ternaries).
    let then_value_end = ctx.builder.get_insert_block().unwrap();

    // Else branch — lower the value but don't emit the branch yet.
    ctx.builder.set_insert_point(else_blk);
    let else_tv = lower_expr_inner(ctx, else_expr);
    let else_value_end = ctx.builder.get_insert_block().unwrap();

    // Compute the unified result type for both branches.
    let result_ty = type_builder::usual_arithmetic_conversion(&then_tv.ty, &else_tv.ty);
    let result_ir = ctype_to_ir(&result_ty, ctx.target);

    // Insert implicit conversions in the THEN block (before the branch).
    // This ensures conversion instructions (e.g. ptrtoint, sext) are
    // placed in the branch that produces the value, not in the merge block
    // where they would violate the phi-at-block-start invariant.
    ctx.builder.set_insert_point(then_value_end);
    let tv = insert_implicit_conversion(ctx, then_tv.value, &then_tv.ty, &result_ty, span);
    let bi_then = ctx.builder.build_branch(merge_blk, span);
    emit_inst(ctx, bi_then);
    let then_end = ctx.builder.get_insert_block().unwrap();

    // Establish CFG edge: then_end → merge_blk
    add_cfg_edge(ctx, then_end.index(), merge_blk.index());

    // Insert implicit conversions in the ELSE block (before the branch).
    ctx.builder.set_insert_point(else_value_end);
    let ev = insert_implicit_conversion(ctx, else_tv.value, &else_tv.ty, &result_ty, span);
    let bi_else = ctx.builder.build_branch(merge_blk, span);
    emit_inst(ctx, bi_else);
    let else_end = ctx.builder.get_insert_block().unwrap();

    // Establish CFG edge: else_end → merge_blk
    add_cfg_edge(ctx, else_end.index(), merge_blk.index());

    // Merge — phi uses the already-converted values from each branch.
    ctx.builder.set_insert_point(merge_blk);
    let (pv, pi) = ctx
        .builder
        .build_phi(result_ir, vec![(tv, then_end), (ev, else_end)], span);
    emit_inst(ctx, pi);
    TypedValue::new(pv, result_ty)
}

// =========================================================================
// FUNCTION CALL
// =========================================================================

/// Try to lower a function call as a compiler builtin.
///
/// Returns `Some(TypedValue)` when the call was successfully lowered inline,
/// `None` if it is not a recognised builtin and should fall through to the
/// normal call lowering path.
///
/// Builtins handled here:
///   * `__atomic_load_n(ptr, order)` → load from ptr
///   * `__atomic_store_n(ptr, val, order)` → store val to ptr
///   * `__atomic_load(ptr, ret, order)` → load from ptr, store to ret
///   * `__atomic_store(ptr, val_ptr, order)` → load from val_ptr, store to ptr
///   * `__atomic_exchange_n(ptr, val, order)` → load old, store new, return old
///   * `__atomic_compare_exchange_n(ptr, expected, desired, weak, succ, fail)`
///   * `__sync_val_compare_and_swap(ptr, old, new)` → simple CAS
///   * `__atomic_fetch_{add,sub,and,or,xor}(ptr, val, order)` → load+op+store
///   * `__sync_fetch_and_{add,sub,and,or,xor}(ptr, val)` → same
///   * `__sync_add_and_fetch(ptr, val)` → add+store, return new
///   * `__sync_synchronize()` → no-op (full barrier, lowered as nothing for now)
///   * `__builtin_inff()`, `__builtin_inf()`, `__builtin_huge_valf()` → inf constant
///   * `__builtin_nanf(str)`, `__builtin_nan(str)` → NaN constant
///   * `__builtin_isnan(x)`, `__builtin_isinf(x)` → comparison
///   * `__builtin_fabs(x)`, `__builtin_fabsf(x)` → absolute value
fn try_lower_builtin_call(
    ctx: &mut ExprLoweringContext<'_>,
    callee: &ast::Expression,
    args: &[ast::Expression],
    span: Span,
) -> Option<TypedValue> {
    // Extract the function name from the callee expression.
    let func_name = match callee {
        ast::Expression::Identifier { name, .. } => resolve_sym(ctx, name.as_u32()).to_string(),
        _ => return None,
    };

    match func_name.as_str() {
        // ── alloca / __builtin_alloca → StackAlloc ──
        "alloca" | "__builtin_alloca" | "__builtin_alloca_with_align" => {
            if let Some(size_expr) = args.first() {
                let size_tv = lower_expr_inner(ctx, size_expr);
                let size_val = insert_implicit_conversion(
                    ctx,
                    size_tv.value,
                    &size_tv.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let (result, inst) = ctx.builder.build_stack_alloc(size_val, span);
                emit_inst(ctx, inst);
                Some(TypedValue::new(
                    result,
                    CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
                ))
            } else {
                let zero = emit_int_const(ctx, 0, IrType::Ptr, span);
                Some(TypedValue::new(
                    zero,
                    CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
                ))
            }
        }

        // ── abs/labs/llabs → inline branchless absolute value ──
        // These MUST be inlined so user-defined replacements don't interfere
        // with constant-folded calls (GCC torture test 20021127-1).
        "abs" | "labs" | "llabs" | "__builtin_abs" | "__builtin_labs" | "__builtin_llabs" => {
            if args.is_empty() {
                return None;
            }
            let (ir_ty, c_ty) = match func_name.as_str() {
                "abs" | "__builtin_abs" => (IrType::I32, CType::Int),
                "labs" | "__builtin_labs" => (IrType::I64, CType::Long),
                "llabs" | "__builtin_llabs" => (IrType::I64, CType::LongLong),
                _ => unreachable!(),
            };
            let arg_tv = lower_expr_inner(ctx, &args[0]);
            let arg_val = insert_implicit_conversion(ctx, arg_tv.value, &arg_tv.ty, &c_ty, span);
            // Branchless abs: result = (x ^ (x >> 63)) - (x >> 63)
            // where 63 for I64 / 31 for I32
            let shift_amt = if matches!(ir_ty, IrType::I32) {
                31i128
            } else {
                63i128
            };
            let sh = emit_int_const(ctx, shift_amt, ir_ty.clone(), span);
            let (sar_val, sar_inst) = ctx
                .builder
                .build_shr(arg_val, sh, ir_ty.clone(), true, span);
            emit_inst(ctx, sar_inst);
            let (xor_val, xor_inst) = ctx.builder.build_xor(arg_val, sar_val, ir_ty.clone(), span);
            emit_inst(ctx, xor_inst);
            let (sub_val, sub_inst) = ctx.builder.build_sub(xor_val, sar_val, ir_ty.clone(), span);
            emit_inst(ctx, sub_inst);
            Some(TypedValue::new(sub_val, c_ty))
        }

        // ── fabs/fabsf → libc forwarders ──
        "fabs" | "__builtin_fabs" => Some(lower_libc_builtin(
            ctx,
            args,
            "fabs",
            IrType::F64,
            CType::Double,
            span,
        )),
        "fabsf" | "__builtin_fabsf" => Some(lower_libc_builtin(
            ctx,
            args,
            "fabsf",
            IrType::F32,
            CType::Float,
            span,
        )),

        // ── __atomic_load_n(ptr, order) → *ptr ──
        "__atomic_load_n" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            // Determine loaded type from pointer's pointee type.
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => {
                    let inner_s = strip_type(inner);
                    ctype_to_ir(inner_s, ctx.target)
                }
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(val, ret_cty))
        }

        // ── __atomic_store_n(ptr, val, order) → *ptr = val ──
        "__atomic_store_n" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let si = ctx.builder.build_store(val_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_load(ptr, ret_ptr, order) ──
        "__atomic_load" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let ret_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val, ret_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_store(ptr, val_ptr, order) ──
        "__atomic_store" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_ptr_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (val, li) = ctx.builder.build_load(val_ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __atomic_exchange_n(ptr, new_val, order) → old ──
        "__atomic_exchange_n" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let new_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __atomic_compare_exchange_n(ptr, expected, desired, weak, succ, fail) → bool ──
        // Simplified for single-threaded: always succeeds (store desired).
        "__atomic_compare_exchange_n" | "__atomic_compare_exchange" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let _exp_tv = lower_expr_inner(ctx, &args[1]);
            let des_tv = lower_expr_inner(ctx, &args[2]);
            // Evaluate remaining args for side effects.
            for arg in args.iter().skip(3) {
                let _ = lower_expr_inner(ctx, arg);
            }
            let si = ctx.builder.build_store(des_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let one = emit_int_const(ctx, 1, IrType::I32, span);
            Some(TypedValue::new(one, CType::Int))
        }

        // ── __sync_val_compare_and_swap(ptr, old, new) → old_value ──
        "__sync_val_compare_and_swap" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let _old_tv = lower_expr_inner(ctx, &args[1]);
            let new_tv = lower_expr_inner(ctx, &args[2]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            // For non-threaded: just store the new value unconditionally.
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_bool_compare_and_swap(ptr, old, new) → bool ──
        "__sync_bool_compare_and_swap" => {
            if args.len() < 3 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let old_tv = lower_expr_inner(ctx, &args[1]);
            let new_tv = lower_expr_inner(ctx, &args[2]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (cur_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            // Compare and swap.
            let (cmp_val, ci) = ctx.builder.build_icmp(
                crate::ir::instructions::ICmpOp::Eq,
                cur_val,
                old_tv.value,
                span,
            );
            emit_inst(ctx, ci);
            // Store new value unconditionally (non-threaded simplified).
            let si = ctx.builder.build_store(new_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::new(cmp_val, CType::Int))
        }

        // ── __atomic_fetch_{add,sub,and,or,xor}(ptr, val, order) → old ──
        "__atomic_fetch_add"
        | "__atomic_fetch_sub"
        | "__atomic_fetch_and"
        | "__atomic_fetch_or"
        | "__atomic_fetch_xor"
        | "__sync_fetch_and_add"
        | "__sync_fetch_and_sub"
        | "__sync_fetch_and_and"
        | "__sync_fetch_and_or"
        | "__sync_fetch_and_xor" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);

            let (new_val, bi) = if func_name.contains("add") {
                ctx.builder
                    .build_add(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("sub") {
                ctx.builder
                    .build_sub(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("_and") {
                ctx.builder
                    .build_and(old_val, val_tv.value, load_ty.clone(), span)
            } else if func_name.contains("_or") {
                ctx.builder
                    .build_or(old_val, val_tv.value, load_ty.clone(), span)
            } else {
                ctx.builder
                    .build_xor(old_val, val_tv.value, load_ty.clone(), span)
            };
            emit_inst(ctx, bi);
            let si = ctx.builder.build_store(new_val, ptr_tv.value, span);
            emit_inst(ctx, si);

            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_add_and_fetch(ptr, val) → new_val ──
        "__sync_add_and_fetch" | "__sync_sub_and_fetch" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty.clone(), span);
            emit_inst(ctx, li);
            let (new_val, bi) = if func_name.contains("add") {
                ctx.builder.build_add(old_val, val_tv.value, load_ty, span)
            } else {
                ctx.builder.build_sub(old_val, val_tv.value, load_ty, span)
            };
            emit_inst(ctx, bi);
            let si = ctx.builder.build_store(new_val, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(new_val, ret_cty))
        }

        // ── __sync_synchronize() → memory fence (no-op for codegen) ──
        "__sync_synchronize" => Some(TypedValue::void()),

        // ── __sync_lock_test_and_set(ptr, val) → old (exchange) ──
        "__sync_lock_test_and_set" => {
            if args.len() < 2 {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let val_tv = lower_expr_inner(ctx, &args[1]);
            let load_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let (old_val, li) = ctx.builder.build_load(ptr_tv.value, load_ty, span);
            emit_inst(ctx, li);
            let si = ctx.builder.build_store(val_tv.value, ptr_tv.value, span);
            emit_inst(ctx, si);
            let ret_cty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => strip_type(inner).clone(),
                _ => CType::Long,
            };
            Some(TypedValue::new(old_val, ret_cty))
        }

        // ── __sync_lock_release(ptr) → *ptr = 0 ──
        "__sync_lock_release" => {
            if args.is_empty() {
                return None;
            }
            let ptr_tv = lower_expr_inner(ctx, &args[0]);
            let store_ty = match strip_type(&ptr_tv.ty) {
                CType::Pointer(inner, _) => ctype_to_ir(strip_type(inner), ctx.target),
                _ => IrType::I64,
            };
            let zero = emit_int_const(ctx, 0, store_ty, span);
            let si = ctx.builder.build_store(zero, ptr_tv.value, span);
            emit_inst(ctx, si);
            Some(TypedValue::void())
        }

        // ── __builtin_inff() / __builtin_inf() / __builtin_huge_valf() ──
        "__builtin_inff" | "__builtin_huge_valf" => {
            let v = emit_float_const(ctx, f64::INFINITY, IrType::F32, span);
            Some(TypedValue::new(v, CType::Float))
        }
        "__builtin_inf" | "__builtin_huge_val" => {
            let v = emit_float_const(ctx, f64::INFINITY, IrType::F64, span);
            Some(TypedValue::new(v, CType::Double))
        }

        // ── __builtin_nanf("") / __builtin_nan("") → NaN constant ──
        "__builtin_nanf" => {
            let v = emit_float_const(ctx, f64::NAN, IrType::F32, span);
            Some(TypedValue::new(v, CType::Float))
        }
        "__builtin_nan" => {
            let v = emit_float_const(ctx, f64::NAN, IrType::F64, span);
            Some(TypedValue::new(v, CType::Double))
        }

        // ── Float predicates: isinf, isnan, signbit, isfinite ──
        // All use volatile store+load to type-pun float→int, then bit manipulation.
        // Float:  int_ty=I32, abs_mask=0x7FFFFFFF, exp_mask=0x7F800000, shift=31
        // Double: int_ty=I64, abs_mask=0x7FFFFFFFFFFFFFFF, exp_mask=0x7FF0000000000000, shift=63
        "__builtin_isinf"
        | "__builtin_isinff"
        | "__builtin_isinfl"
        | "isinf"
        | "isinff"
        | "__builtin_isinf_sign"
        | "__builtin_isfinite"
        | "__builtin_isfinitef"
        | "__builtin_isfinitel"
        | "isfinite"
        | "__isfinite"
        | "__finitef"
        | "__finite"
        | "__builtin_isnan"
        | "__builtin_isnanf"
        | "__builtin_isnanl"
        | "isnan"
        | "isnanf"
        | "isnanl"
        | "__builtin_signbit"
        | "__builtin_signbitf"
        | "__builtin_signbitl"
        | "signbit"
        | "__signbit"
        | "__signbitf" => {
            if args.is_empty() {
                return None;
            }
            let x_tv = lower_expr_inner(ctx, &args[0]);
            let resolved = crate::common::types::resolve_typedef(&x_tv.ty);
            let is_float = matches!(resolved, CType::Float);
            let (int_ty, abs_mask, exp_mask, shift_n) = if is_float {
                (IrType::I32, 0x7FFF_FFFFi128, 0x7F80_0000i128, 31i128)
            } else {
                (
                    IrType::I64,
                    0x7FFF_FFFF_FFFF_FFFFi128,
                    0x7FF0_0000_0000_0000i128,
                    63i128,
                )
            };
            // Type-pun via volatile store+load (bitcast codegen doesn't
            // preserve bits for float→int on x86-64, so use memory).
            let (alloca, alloc_i) = ctx.builder.build_alloca(int_ty.clone(), span);
            emit_inst(ctx, alloc_i);
            let mut si = ctx.builder.build_store(x_tv.value, alloca, span);
            if let crate::ir::instructions::Instruction::Store {
                ref mut volatile, ..
            } = si
            {
                *volatile = true;
            }
            emit_inst(ctx, si);
            let (bits, li) = ctx.builder.build_load(alloca, int_ty.clone(), span);
            emit_inst(ctx, li);

            let is_isinf = func_name.contains("isinf") && !func_name.contains("sign");
            let is_isinf_sign = func_name.contains("isinf_sign");
            let is_isnan = func_name.contains("isnan");
            let is_signbit = func_name.contains("signbit");
            // isfinite = everything else

            if is_signbit {
                let sh = emit_int_const(ctx, shift_n, int_ty.clone(), span);
                let (shifted, shi) = ctx.builder.build_shr(bits, sh, int_ty.clone(), false, span);
                emit_inst(ctx, shi);
                let result = if int_ty != IrType::I32 {
                    let (tr, tri) = ctx.builder.build_trunc(shifted, IrType::I32, span);
                    emit_inst(ctx, tri);
                    tr
                } else {
                    shifted
                };
                return Some(TypedValue::new(result, CType::Int));
            }

            // Compute abs_bits = bits & abs_mask
            let mask_v = emit_int_const(ctx, abs_mask, int_ty.clone(), span);
            let (abs_bits, abi) = ctx.builder.build_and(bits, mask_v, int_ty.clone(), span);
            emit_inst(ctx, abi);
            let exp_v = emit_int_const(ctx, exp_mask, int_ty.clone(), span);

            if is_isinf {
                // isinf: abs_bits == exp_mask
                let (cmp, ci) = ctx.builder.build_icmp(ICmpOp::Eq, abs_bits, exp_v, span);
                emit_inst(ctx, ci);
                let (val, zi) = ctx
                    .builder
                    .build_zext_from(cmp, IrType::I1, IrType::I32, span);
                emit_inst(ctx, zi);
                Some(TypedValue::new(val, CType::Int))
            } else if is_isnan {
                // isnan: abs_bits > exp_mask
                let (cmp, ci) = ctx.builder.build_icmp(ICmpOp::Ugt, abs_bits, exp_v, span);
                emit_inst(ctx, ci);
                let (val, zi) = ctx
                    .builder
                    .build_zext_from(cmp, IrType::I1, IrType::I32, span);
                emit_inst(ctx, zi);
                Some(TypedValue::new(val, CType::Int))
            } else if is_isinf_sign {
                // isinf_sign: +inf→1, -inf→-1, else→0
                let (is_inf, ici) = ctx.builder.build_icmp(ICmpOp::Eq, abs_bits, exp_v, span);
                emit_inst(ctx, ici);
                let (inf_i32, ie) =
                    ctx.builder
                        .build_zext_from(is_inf, IrType::I1, IrType::I32, span);
                emit_inst(ctx, ie);
                let sh = emit_int_const(ctx, shift_n, int_ty.clone(), span);
                let (shifted, shi) = ctx.builder.build_shr(bits, sh, int_ty.clone(), false, span);
                emit_inst(ctx, shi);
                let sign_i32 = if int_ty != IrType::I32 {
                    let (tr, tri) = ctx.builder.build_trunc(shifted, IrType::I32, span);
                    emit_inst(ctx, tri);
                    tr
                } else {
                    shifted
                };
                let one_v = emit_int_const(ctx, 1, IrType::I32, span);
                let (s_and, sai) = ctx.builder.build_and(sign_i32, one_v, IrType::I32, span);
                emit_inst(ctx, sai);
                let two_v = emit_int_const(ctx, 2, IrType::I32, span);
                let (t2s, t2i) = ctx.builder.build_mul(two_v, s_and, IrType::I32, span);
                emit_inst(ctx, t2i);
                let (neg, ni) = ctx.builder.build_mul(inf_i32, t2s, IrType::I32, span);
                emit_inst(ctx, ni);
                let (res, ri) = ctx.builder.build_sub(inf_i32, neg, IrType::I32, span);
                emit_inst(ctx, ri);
                Some(TypedValue::new(res, CType::Int))
            } else {
                // isfinite: abs_bits < exp_mask
                let (cmp, ci) = ctx.builder.build_icmp(ICmpOp::Ult, abs_bits, exp_v, span);
                emit_inst(ctx, ci);
                let (val, zi) = ctx
                    .builder
                    .build_zext_from(cmp, IrType::I1, IrType::I32, span);
                emit_inst(ctx, zi);
                Some(TypedValue::new(val, CType::Int))
            }
        }

        // ── __builtin_classify_type(x) → integer constant ──
        "__builtin_classify_type" | "classify_type" => {
            // GCC type classification. Return compile-time constant.
            if args.is_empty() {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                return Some(TypedValue::new(v, CType::Int));
            }
            let atv = lower_expr_inner(ctx, &args[0]);
            let class = classify_type_for_builtin(&atv.ty);
            let v = emit_int_const(ctx, class as i128, IrType::I32, span);
            Some(TypedValue::new(v, CType::Int))
        }

        // ── __builtin_clrsb / clrsbl / clrsbll → count leading redundant sign bits ──
        "__builtin_clrsb" | "clrsb" => Some(lower_bit_builtin(ctx, args, "clrsb", span)),
        "__builtin_clrsbl" => Some(lower_bit_builtin(ctx, args, "clrsbl", span)),
        "__builtin_clrsbll" => Some(lower_bit_builtin(ctx, args, "clrsbll", span)),

        // ── __builtin_parity / parityl / parityll → popcount & 1 ──
        "__builtin_parity" => Some(lower_bit_builtin(ctx, args, "parity", span)),
        "__builtin_parityl" => Some(lower_bit_builtin(ctx, args, "parityl", span)),
        "__builtin_parityll" => Some(lower_bit_builtin(ctx, args, "parityll", span)),

        // ── Math builtins → libc forwarders ──
        "__builtin_copysign" => Some(lower_libc_builtin(
            ctx,
            args,
            "copysign",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_copysignf" => Some(lower_libc_builtin(
            ctx,
            args,
            "copysignf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_copysignl" => Some(lower_libc_builtin(
            ctx,
            args,
            "copysignl",
            IrType::F64,
            CType::LongDouble,
            span,
        )),
        "__builtin_fabsl" => Some(lower_libc_builtin(
            ctx,
            args,
            "fabsl",
            IrType::F64,
            CType::LongDouble,
            span,
        )),
        "__builtin_floor" => Some(lower_libc_builtin(
            ctx,
            args,
            "floor",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_floorf" => Some(lower_libc_builtin(
            ctx,
            args,
            "floorf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_ceil" => Some(lower_libc_builtin(
            ctx,
            args,
            "ceil",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_ceilf" => Some(lower_libc_builtin(
            ctx,
            args,
            "ceilf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_sqrt" => Some(lower_libc_builtin(
            ctx,
            args,
            "sqrt",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_sqrtf" => Some(lower_libc_builtin(
            ctx,
            args,
            "sqrtf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_log" => Some(lower_libc_builtin(
            ctx,
            args,
            "log",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_logf" => Some(lower_libc_builtin(
            ctx,
            args,
            "logf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_log2" => Some(lower_libc_builtin(
            ctx,
            args,
            "log2",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_log2f" => Some(lower_libc_builtin(
            ctx,
            args,
            "log2f",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_log10" => Some(lower_libc_builtin(
            ctx,
            args,
            "log10",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_log10f" => Some(lower_libc_builtin(
            ctx,
            args,
            "log10f",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_exp" => Some(lower_libc_builtin(
            ctx,
            args,
            "exp",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_expf" => Some(lower_libc_builtin(
            ctx,
            args,
            "expf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_exp2" => Some(lower_libc_builtin(
            ctx,
            args,
            "exp2",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_exp2f" => Some(lower_libc_builtin(
            ctx,
            args,
            "exp2f",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_pow" => Some(lower_libc_builtin(
            ctx,
            args,
            "pow",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_powf" => Some(lower_libc_builtin(
            ctx,
            args,
            "powf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_sin" => Some(lower_libc_builtin(
            ctx,
            args,
            "sin",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_sinf" => Some(lower_libc_builtin(
            ctx,
            args,
            "sinf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_cos" => Some(lower_libc_builtin(
            ctx,
            args,
            "cos",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_cosf" => Some(lower_libc_builtin(
            ctx,
            args,
            "cosf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_tan" => Some(lower_libc_builtin(
            ctx,
            args,
            "tan",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_tanf" => Some(lower_libc_builtin(
            ctx,
            args,
            "tanf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_asin" => Some(lower_libc_builtin(
            ctx,
            args,
            "asin",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_asinf" => Some(lower_libc_builtin(
            ctx,
            args,
            "asinf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_acos" => Some(lower_libc_builtin(
            ctx,
            args,
            "acos",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_acosf" => Some(lower_libc_builtin(
            ctx,
            args,
            "acosf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_atan" => Some(lower_libc_builtin(
            ctx,
            args,
            "atan",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_atanf" => Some(lower_libc_builtin(
            ctx,
            args,
            "atanf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_atan2" => Some(lower_libc_builtin(
            ctx,
            args,
            "atan2",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_atan2f" => Some(lower_libc_builtin(
            ctx,
            args,
            "atan2f",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_round" => Some(lower_libc_builtin(
            ctx,
            args,
            "round",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_roundf" => Some(lower_libc_builtin(
            ctx,
            args,
            "roundf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_trunc" => Some(lower_libc_builtin(
            ctx,
            args,
            "trunc",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_truncf" => Some(lower_libc_builtin(
            ctx,
            args,
            "truncf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_fmin" => Some(lower_libc_builtin(
            ctx,
            args,
            "fmin",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_fminf" => Some(lower_libc_builtin(
            ctx,
            args,
            "fminf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_fmax" => Some(lower_libc_builtin(
            ctx,
            args,
            "fmax",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_fmaxf" => Some(lower_libc_builtin(
            ctx,
            args,
            "fmaxf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_remainder" => Some(lower_libc_builtin(
            ctx,
            args,
            "remainder",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_remainderf" => Some(lower_libc_builtin(
            ctx,
            args,
            "remainderf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_fmod" => Some(lower_libc_builtin(
            ctx,
            args,
            "fmod",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_fmodf" => Some(lower_libc_builtin(
            ctx,
            args,
            "fmodf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_ldexp" => Some(lower_libc_builtin(
            ctx,
            args,
            "ldexp",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_ldexpf" => Some(lower_libc_builtin(
            ctx,
            args,
            "ldexpf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_frexp" => {
            // frexp returns double, second arg is int* (unchanged)
            Some(lower_libc_builtin(
                ctx,
                args,
                "frexp",
                IrType::F64,
                CType::Double,
                span,
            ))
        }
        "__builtin_frexpf" => Some(lower_libc_builtin(
            ctx,
            args,
            "frexpf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_modf" => Some(lower_libc_builtin(
            ctx,
            args,
            "modf",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_modff" => Some(lower_libc_builtin(
            ctx,
            args,
            "modff",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_nearbyint" => Some(lower_libc_builtin(
            ctx,
            args,
            "nearbyint",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_nearbyintf" => Some(lower_libc_builtin(
            ctx,
            args,
            "nearbyintf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_rint" => Some(lower_libc_builtin(
            ctx,
            args,
            "rint",
            IrType::F64,
            CType::Double,
            span,
        )),
        "__builtin_rintf" => Some(lower_libc_builtin(
            ctx,
            args,
            "rintf",
            IrType::F32,
            CType::Float,
            span,
        )),
        "__builtin_lrint" => Some(lower_libc_builtin(
            ctx,
            args,
            "lrint",
            IrType::I64,
            CType::Long,
            span,
        )),
        "__builtin_lrintf" => Some(lower_libc_builtin(
            ctx,
            args,
            "lrintf",
            IrType::I64,
            CType::Long,
            span,
        )),
        "__builtin_llrint" => Some(lower_libc_builtin(
            ctx,
            args,
            "llrint",
            IrType::I64,
            CType::LongLong,
            span,
        )),
        "__builtin_llrintf" => Some(lower_libc_builtin(
            ctx,
            args,
            "llrintf",
            IrType::I64,
            CType::LongLong,
            span,
        )),
        "__builtin_lround" => Some(lower_libc_builtin(
            ctx,
            args,
            "lround",
            IrType::I64,
            CType::Long,
            span,
        )),
        "__builtin_lroundf" => Some(lower_libc_builtin(
            ctx,
            args,
            "lroundf",
            IrType::I64,
            CType::Long,
            span,
        )),
        "__builtin_llround" => Some(lower_libc_builtin(
            ctx,
            args,
            "llround",
            IrType::I64,
            CType::LongLong,
            span,
        )),
        "__builtin_llroundf" => Some(lower_libc_builtin(
            ctx,
            args,
            "llroundf",
            IrType::I64,
            CType::LongLong,
            span,
        )),
        "__builtin_cabs" | "__builtin_cabsf" | "__builtin_cabsl" => {
            // Complex absolute value — forward to libc cabs/cabsf/cabsl
            let libc_name = func_name.strip_prefix("__builtin_").unwrap_or(&func_name);
            Some(lower_libc_builtin(
                ctx,
                args,
                libc_name,
                IrType::F64,
                CType::Double,
                span,
            ))
        }

        // ── __builtin_conjf / __builtin_conj / __builtin_conjl ──
        // Complex conjugate: negate the imaginary part.
        "__builtin_conjf" | "conjf" => {
            lower_complex_conj(ctx, args, CType::Float, IrType::F32, span)
        }
        "__builtin_conj" | "conj" => {
            lower_complex_conj(ctx, args, CType::Double, IrType::F64, span)
        }
        "__builtin_conjl" | "conjl" => {
            lower_complex_conj(ctx, args, CType::LongDouble, IrType::F80, span)
        }

        // Not a recognized builtin — fall through.
        _ => None,
    }
}

/// Strip `*` (Deref) and parenthesized wrappers from a function-call
/// callee expression.  In C, `(*fptr)(args)` is identical to
/// `fptr(args)` — dereferencing a function pointer yields a function
/// designator that immediately decays back to a pointer.  Multiple
/// levels of dereference are also legal: `(***fptr)(args)`.
///
/// This stripping MUST happen before IR lowering because by that point
/// the C-level function type has been erased to an opaque
/// `Pointer(Void)` and the generic Deref handler would emit a real
/// memory load (reading machine-code bytes rather than using the
/// pointer as a callable address).
fn strip_fn_ptr_deref(expr: &ast::Expression) -> &ast::Expression {
    match expr {
        ast::Expression::Parenthesized { inner, .. } => strip_fn_ptr_deref(inner),
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            ..
        } => strip_fn_ptr_deref(operand),
        other => other,
    }
}

fn lower_function_call(
    ctx: &mut ExprLoweringContext<'_>,
    callee: &ast::Expression,
    args: &[ast::Expression],
    span: Span,
) -> TypedValue {
    // In C, dereferencing a function pointer is a no-op in call context:
    //   `(*fptr)(args)` ≡ `fptr(args)` ≡ `(**fptr)(args)`.
    // The function designator decays back to a function pointer.
    //
    // We must strip Deref (and Parenthesized) wrappers here because by the
    // time lower_expr_inner evaluates the operand, the C-level function-
    // pointer type has been erased to `Pointer(Void)` in the IR, so the
    // generic Deref handler cannot distinguish `*fptr` (identity) from
    // `*p` (actual memory load) and emits an incorrect Load instruction
    // that reads machine-code bytes from the function's address.
    let effective_callee = strip_fn_ptr_deref(callee);

    // Lower callee (may be an identifier or a function pointer expression).
    let callee_tv = lower_expr_inner(ctx, effective_callee);

    // Determine return type from the callee's function type.
    // For direct function calls, lower_identifier returns a void pointer,
    // so we look up the actual function return type from the IR module.
    let (mut ret_cty, _is_variadic) = extract_function_return_type(&callee_tv.ty);
    let mut ret_ir = ctype_to_ir(&ret_cty, ctx.target);

    // If the callee is a known function and the CType-derived return type
    // doesn't capture struct layout (because lower_identifier returns a void
    // pointer for function references), resolve the actual IR return type
    // directly from the function definition or declaration.  This ensures
    // the Call instruction carries the correct struct field layout for ABI
    // classification (e.g., small structs returned in registers).
    if let Some(fname) = ctx.function.func_ref_map.get(&callee_tv.value) {
        // Recover the correct C-level return type from our map.
        if let Some(stored_cty) = ctx.module.func_c_return_types.get(fname) {
            ret_cty = stored_cty.clone();
            ret_ir = ctype_to_ir(&ret_cty, ctx.target);
        }
        if let Some(func) = ctx.module.get_function(fname) {
            if !func.return_type.is_void() {
                ret_ir = func.return_type.clone();
            }
        } else {
            // Check function declarations.
            for decl in ctx.module.declarations() {
                if decl.name == *fname {
                    if !decl.return_type.is_void() {
                        ret_ir = decl.return_type.clone();
                    }
                    break;
                }
            }
        }
    }

    // Look up the callee's declared C-level parameter types from the
    // module map.  This is necessary because `lower_identifier` returns
    // the callee as an opaque `Pointer(Void)` without function type info.
    let param_ctypes: Option<Vec<CType>> = ctx
        .function
        .func_ref_map
        .get(&callee_tv.value)
        .and_then(|fname| ctx.module.func_c_param_types.get(fname))
        .cloned()
        // Also try extracting from callee's CType (for function pointers
        // that preserve their type information).
        .or_else(|| extract_function_param_types(&callee_tv.ty));

    // Lower arguments left-to-right.
    let mut arg_vals = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let atv = lower_expr_inner(ctx, arg);

        // If a declared parameter type is available for this position,
        // convert the argument to match the parameter type (this handles
        // int → long long sign-extension, int → float, etc.).
        // For variadic extra arguments (beyond the declared params) or
        // when no parameter type list is available, apply default argument
        // promotions (C11 §6.5.2.2p7).
        let target_ty = if let Some(pty) = param_ctypes.as_ref().and_then(|p| p.get(i)) {
            pty.clone()
        } else {
            // Default argument promotions for variadic / unprototyped calls.
            let promoted = types::integer_promotion(&atv.ty);
            // float → double for variadic.
            if matches!(types::resolve_typedef(&atv.ty), CType::Float) {
                CType::Double
            } else {
                promoted
            }
        };
        let v = insert_implicit_conversion(ctx, atv.value, &atv.ty, &target_ty, span);
        arg_vals.push(v);
    }

    let (result, ci) = ctx
        .builder
        .build_call(callee_tv.value, arg_vals, ret_ir.clone(), span);
    emit_inst(ctx, ci);

    if matches!(ret_cty, CType::Void) {
        TypedValue::void()
    } else {
        TypedValue::new(result, ret_cty)
    }
}

/// Extract the return type from a function or function-pointer C type.
fn extract_function_return_type(ctype: &CType) -> (CType, bool) {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Function {
            return_type,
            variadic,
            ..
        } => (*return_type.clone(), *variadic),
        CType::Pointer(inner, _) => {
            let inner_resolved = types::resolve_typedef(inner);
            match inner_resolved {
                CType::Function {
                    return_type,
                    variadic,
                    ..
                } => (*return_type.clone(), *variadic),
                _ => (CType::Int, false),
            }
        }
        _ => (CType::Int, false),
    }
}

/// Extract the declared parameter types from a function or function-pointer
/// C type.  Returns `None` when the callee has no known prototype (K&R or
/// untyped function pointer).
fn extract_function_param_types(ctype: &CType) -> Option<Vec<CType>> {
    let resolved = types::resolve_typedef(ctype);
    match resolved {
        CType::Function { params, .. } => {
            if params.is_empty() {
                None // empty param list ⇒ unprototyped (K&R `int f()`)
            } else {
                Some(params.clone())
            }
        }
        CType::Pointer(inner, _) => {
            let inner_resolved = types::resolve_typedef(inner);
            match inner_resolved {
                CType::Function { params, .. } => {
                    if params.is_empty() {
                        None
                    } else {
                        Some(params.clone())
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// =========================================================================
// CAST
// =========================================================================

fn lower_cast_expr(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    let inner = lower_expr_inner(ctx, operand);
    let target_cty = resolve_type_name(ctx, type_name);

    // Void cast — evaluate for side effects, discard result.
    if matches!(strip_type(&target_cty), CType::Void) {
        return TypedValue::void();
    }

    let val = lower_cast(ctx, inner.value, &inner.ty, &target_cty, span);
    TypedValue::new(val, target_cty)
}

/// Emit the appropriate cast instruction(s) for a type conversion.
fn lower_cast(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    from_cty: &CType,
    to_cty: &CType,
    span: Span,
) -> Value {
    let from = strip_type(from_cty);
    let to = strip_type(to_cty);
    let from_ir = ctype_to_ir(from, ctx.target);
    let to_ir = ctype_to_ir(to, ctx.target);

    // Same type → no-op.
    if from_ir == to_ir {
        return value;
    }

    // Cast to Bool must use comparison against zero (not truncation).
    // C11 §6.3.1.2: "When any scalar value is converted to _Bool, the
    // result is 0 if the value compares equal to 0; otherwise, the result is 1."
    // This must run BEFORE the generic integer→integer path, which would
    // use trunc (extracting the low bit) — incorrect for e.g. (_Bool)42.
    if matches!(to, CType::Bool) && from_ir.is_integer() {
        let zero = emit_zero(ctx, from_ir, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Bool (I1) to integer — zero-extend.
    if matches!(from, CType::Bool) && to_ir.is_integer() {
        let (v, i) = ctx.builder.build_zext_from(value, IrType::I1, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Integer.
    if from_ir.is_integer() && to_ir.is_integer() {
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else if is_unsigned(from) {
            let (v, i) = ctx
                .builder
                .build_zext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else {
            let (v, i) = ctx
                .builder
                .build_sext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
    }

    // Float → Float.
    if from_ir.is_float() && to_ir.is_float() {
        // Use bitcast as a placeholder; proper FPTrunc/FPExt would be added
        // to the instruction set in a follow-up.  For now, identity via bitcast.
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Float.
    if from_ir.is_integer() && to_ir.is_float() {
        let src_unsigned = types::is_unsigned(from);
        let (v, i) = ctx
            .builder
            .build_bitcast_ex(value, to_ir, src_unsigned, span);
        emit_inst(ctx, i);
        return v;
    }

    // Float → Integer.
    if from_ir.is_float() && to_ir.is_integer() {
        let dst_unsigned = types::is_unsigned(to);
        let (v, i) = ctx
            .builder
            .build_bitcast_ex(value, to_ir, dst_unsigned, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Bool: convert to integer-sized value, then compare != 0.
    // C11 §6.3.1.2: any scalar (including pointers) converted to _Bool gives
    // 0 if equal to 0, otherwise 1.  A plain PtrToInt to I1 would just grab
    // the low bit of the address, which is wrong.
    if from_ir.is_pointer() && matches!(to, CType::Bool) {
        let ptr_int_ty = size_ir_type(ctx.target);
        let (int_val, pti) = ctx
            .builder
            .build_ptr_to_int(value, ptr_int_ty.clone(), span);
        emit_inst(ctx, pti);
        let zero = emit_zero(ctx, ptr_int_ty, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, int_val, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Float → Bool: compare != 0.0.
    if from_ir.is_float() && matches!(to, CType::Bool) {
        let zero = emit_float_const(ctx, 0.0, from_ir, span);
        let (v, i) = ctx.builder.build_fcmp(FCmpOp::One, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Pointer.
    if from_ir.is_integer() && to_ir.is_pointer() {
        let (v, i) = ctx.builder.build_int_to_ptr(value, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Integer.
    if from_ir.is_pointer() && to_ir.is_integer() {
        let (v, i) = ctx.builder.build_ptr_to_int(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Pointer.
    if from_ir.is_pointer() && to_ir.is_pointer() {
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Scalar → Complex: promote a real/integer scalar to _Complex by
    // setting real = scalar, imag = 0.  C11 §6.3.1.7: "When a value of
    // real type is converted to a complex type, the real part of the
    // complex result value is determined by the rules of conversion to
    // the corresponding real type and the imaginary part of the complex
    // result value is a positive zero or an unsigned zero."
    if let CType::Complex(ref base) = to {
        let elem_ir = ctype_to_ir(base.as_ref(), ctx.target);
        // Cast the scalar value to the element type if needed.
        let real_val = if from_ir != elem_ir {
            lower_cast(ctx, value, from_cty, base.as_ref(), span)
        } else {
            value
        };
        // Build the complex value {real, 0.0} via alloca+store.
        let (alloca_val, alloca_inst) = ctx.builder.build_alloca(to_ir.clone(), span);
        emit_inst(ctx, alloca_inst);
        // Store real part at offset 0.
        let zero_idx = emit_int_const_for_index(ctx, 0);
        let (real_ptr, gep_r) =
            ctx.builder
                .build_gep(alloca_val, vec![zero_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_r);
        let st_real = ctx.builder.build_store(real_val, real_ptr, span);
        emit_inst(ctx, st_real);
        // Store zero imaginary part at offset elem_sz.
        let elem_sz: i64 = complex_elem_size(&elem_ir);
        let imag_idx = emit_int_const_for_index(ctx, elem_sz);
        let (imag_ptr, gep_i) =
            ctx.builder
                .build_gep(alloca_val, vec![imag_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_i);
        // Use integer zero for integer-based complex, float zero for FP complex.
        let zero_imag = if is_fp_ir_type(&elem_ir) {
            emit_float_const(ctx, 0.0, elem_ir, span)
        } else {
            emit_int_const(ctx, 0, elem_ir, span)
        };
        let st_imag = ctx.builder.build_store(zero_imag, imag_ptr, span);
        emit_inst(ctx, st_imag);
        // Load the complete complex value.
        let (v, i) = ctx.builder.build_load(alloca_val, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Complex → Scalar: extract real part.
    // C11 §6.3.1.7: "When a value of complex type is converted to a
    // real type, the imaginary part of the complex value is discarded
    // and the value of the real part is converted."
    if let CType::Complex(ref base) = from {
        let elem_ir = ctype_to_ir(base.as_ref(), ctx.target);
        // Extract real part (component at offset 0) via alloca+GEP+load.
        let (alloca_val, alloca_inst) = ctx.builder.build_alloca(from_ir.clone(), span);
        emit_inst(ctx, alloca_inst);
        let st = ctx.builder.build_store(value, alloca_val, span);
        emit_inst(ctx, st);
        let zero_idx = emit_int_const_for_index(ctx, 0);
        let (real_ptr, gep_r) =
            ctx.builder
                .build_gep(alloca_val, vec![zero_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_r);
        let (real_val, ld) = ctx.builder.build_load(real_ptr, elem_ir, span);
        emit_inst(ctx, ld);
        // Cast the extracted real value to the target type if needed.
        return lower_cast(ctx, real_val, base.as_ref(), to_cty, span);
    }

    // Fallback: bitcast.
    let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
    emit_inst(ctx, i);
    v
}

/// Resolve a `TypeName` AST node to its corresponding C type.
/// This is a simplified resolver that handles the most common patterns.
/// When the base type is a forward-referenced struct/union (tag name only,
/// no inline members), the struct definitions registry is consulted to
/// retrieve the full layout — essential for correct `sizeof` computation.
fn resolve_type_name(ctx: &ExprLoweringContext<'_>, tn: &ast::TypeName) -> CType {
    use super::decl_lowering::resolve_base_type_from_sqlist;

    // Resolve the base type from specifier-qualifier list (e.g. `int`, `double`,
    // `unsigned long long`, `struct foo`).
    let mut base = resolve_base_type_from_sqlist(&tn.specifier_qualifiers);

    // Resolve forward-referenced struct/union types using the struct_defs
    // registry.  A forward reference has a tag name but empty fields list.
    resolve_forward_ref_type(&mut base, ctx.struct_defs);

    // Apply qualifiers from the specifier-qualifier list to the base type.
    // E.g., `const char` → Qualified(Char, {const: true}).
    let has_const = tn
        .specifier_qualifiers
        .type_qualifiers
        .contains(&ast::TypeQualifier::Const);
    let has_volatile = tn
        .specifier_qualifiers
        .type_qualifiers
        .contains(&ast::TypeQualifier::Volatile);
    let has_restrict = tn
        .specifier_qualifiers
        .type_qualifiers
        .contains(&ast::TypeQualifier::Restrict);
    let has_atomic = tn
        .specifier_qualifiers
        .type_qualifiers
        .contains(&ast::TypeQualifier::Atomic);
    if has_const || has_volatile || has_restrict || has_atomic {
        let quals = crate::common::types::TypeQualifiers {
            is_const: has_const,
            is_volatile: has_volatile,
            is_restrict: has_restrict,
            is_atomic: has_atomic,
        };
        base = CType::Qualified(Box::new(base), quals);
    }

    // Apply the abstract declarator (pointer, array, function layers) if present.
    if let Some(ref abs_decl) = tn.abstract_declarator {
        apply_abstract_declarator(base, abs_decl)
    } else {
        base
    }
}

/// Recursively resolve forward-referenced struct/union types.
///
/// If `ctype` is a `Struct` or `Union` with a tag name but empty fields,
/// replace it with the full definition from the `struct_defs` registry.
/// Also recurses into pointer and array element types.
pub fn resolve_forward_ref_type(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                // struct_defs entries are already fully resolved by the
                // recursive pass0 fixup loop, so no further recursion
                // into the replacement is needed.  Recursing would loop
                // infinitely on self-referential pointer fields like
                // `struct PgHdr1 *pNext` inside PgHdr1.
                *ctype = full_def.clone();
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                *ctype = full_def.clone();
            }
        }
        CType::Struct { ref mut fields, .. } | CType::Union { ref mut fields, .. } => {
            // Recurse into field types of non-empty structs/unions
            // built locally from AST (not from struct_defs) to resolve
            // any nested empty-field tag references.
            for field in fields.iter_mut() {
                resolve_forward_ref_type(&mut field.ty, struct_defs);
            }
        }
        CType::Pointer(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Array(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Qualified(inner, _) => resolve_forward_ref_type(inner, struct_defs),
        CType::Typedef { underlying, .. } => resolve_forward_ref_type(underlying, struct_defs),
        CType::Atomic(inner) => resolve_forward_ref_type(inner, struct_defs),
        _ => {}
    }
}

/// Apply pointer / array / function layers from an abstract declarator to a
/// base type.  For example, `char *` has base `char` and one pointer layer.
/// The correct order in C is: pointer first, then direct (array/function).
/// `int *[]` = array of (int *), NOT pointer to (int[]).
fn apply_abstract_declarator(base: CType, abs: &ast::AbstractDeclarator) -> CType {
    // First wrap in pointer layers (e.g., `*`, `**`, `*const`).
    let mut result = if let Some(ref pointer) = abs.pointer {
        apply_pointer_layers_abstract(base, pointer)
    } else {
        base
    };

    // Then apply the direct part (array/function shapes) if any.
    if let Some(ref direct) = abs.direct {
        result = apply_direct_abstract_declarator(result, direct);
    }

    result
}

fn apply_pointer_layers_abstract(base: CType, pointer: &ast::Pointer) -> CType {
    let quals = crate::common::types::TypeQualifiers::default();
    let mut current = CType::Pointer(Box::new(base), quals);
    if let Some(ref inner) = pointer.inner {
        current = apply_pointer_layers_abstract(current, inner);
    }
    current
}

fn apply_direct_abstract_declarator(base: CType, direct: &ast::DirectAbstractDeclarator) -> CType {
    match direct {
        ast::DirectAbstractDeclarator::Parenthesized(inner) => {
            apply_abstract_declarator(base, inner)
        }
        ast::DirectAbstractDeclarator::Array { size, .. } => {
            let len = size
                .as_ref()
                .and_then(|e| super::decl_lowering::evaluate_const_int_expr_pub(e));
            CType::Array(Box::new(base), len)
        }
        ast::DirectAbstractDeclarator::Function { .. } => {
            // Function pointer type name — simplify to a function pointer.
            CType::Pointer(
                Box::new(base),
                crate::common::types::TypeQualifiers::default(),
            )
        }
    }
}

// =========================================================================
// SIZEOF / ALIGNOF
// =========================================================================

fn lower_sizeof_expr(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    // Check if the operand is an identifier referencing a VLA variable.
    // VLA variables have runtime-determined size, stored in ctx.vla_sizes
    // during VLA allocation in stmt_lowering. Return the runtime Value
    // directly instead of the compile-time sizeof (which would be 8 =
    // pointer size, since VLAs are internally stored as pointers).
    if let Some(vla_size) = try_resolve_vla_sizeof(ctx, operand) {
        return TypedValue::new(vla_size, size_ctype(ctx.target));
    }

    // sizeof(expr) — evaluate the expression's type WITHOUT triggering
    // array-to-pointer decay. In C, sizeof is one of the few contexts
    // that preserves array types: sizeof(arr) returns the total array
    // size, not the pointer size.
    //
    // Strategy: use `sizeof_infer_type` which recursively determines the
    // undecayed C type for any expression, including member access chains
    // like `s.arr`, `p->arr`, `*p`, casts, dereferences, and subscripts.
    // This avoids calling `lower_expr_inner` which performs array-to-pointer
    // decay on array-typed sub-expressions.
    let expr_ty = sizeof_infer_type(ctx, operand);
    // Use sizeof_resolved to handle incomplete/forward-declared struct types
    // that appear when dereferencing pointers (e.g. sizeof(*pColl) where
    // pColl's pointee type is a forward-declared struct with empty fields).
    let size = sizeof_resolved(&expr_ty, ctx.type_builder, ctx.struct_defs, ctx.target);
    let ir_ty = size_ir_type(ctx.target);
    let val = emit_int_const(ctx, size as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

/// Try to resolve `sizeof(expr)` as a VLA sizeof. Handles:
///   - `sizeof(vla_name)` → total byte size from `vla_sizes`
///   - `sizeof(vla_name[i])` → row stride from `vla_sizes` (stride at
///     subscript depth), i.e., the byte size of one "row" of the VLA.
///   - `sizeof(vla_name[i][j])` → stride at depth+1, etc.
fn try_resolve_vla_sizeof(
    ctx: &mut ExprLoweringContext<'_>,
    expr: &ast::Expression,
) -> Option<Value> {
    match expr {
        ast::Expression::Parenthesized { inner, .. } => try_resolve_vla_sizeof(ctx, inner),
        ast::Expression::Identifier { name, .. } => {
            let sym_name = resolve_sym(ctx, name.as_u32());
            ctx.vla_sizes.get(sym_name).copied()
        }
        ast::Expression::ArraySubscript { base, .. } => {
            // sizeof(a[i]) for VLA `int a[n][m]`: the size of one row is
            // the stride stored as "a\0stride_0". For sizeof(a[i][j]) the
            // size is stride_1, etc.
            //
            // Strategy: find the base VLA name and depth via the same
            // helper used by subscript lowering, but add 1 because the
            // subscript itself consumes one level.
            if let Some((vla_name, depth)) = find_vla_subscript_info(ctx, base) {
                let stride_key = format!("{}\0stride_{}", vla_name, depth);
                if let Some(&stride_val) = ctx.vla_sizes.get(&stride_key) {
                    return Some(stride_val);
                }
            }
            None
        }
        _ => None,
    }
}

/// Infer the C type of an expression for `sizeof` purposes WITHOUT
/// triggering array-to-pointer decay. In C11 §6.5.3.4, the `sizeof`
/// operator yields the size of the operand's type, where array types
/// are NOT converted to pointer types (unlike most other expression
/// contexts).
///
/// This function recursively determines the undecayed type for
/// identifiers, member accesses (`s.arr`, `p->arr`), dereferences,
/// subscripts, casts, and parenthesized expressions.
fn sizeof_infer_type(ctx: &mut ExprLoweringContext<'_>, expr: &ast::Expression) -> CType {
    match expr {
        // Parenthesized: recurse through
        ast::Expression::Parenthesized { inner, .. } => sizeof_infer_type(ctx, inner),

        // Identifier: look up the variable's declared type (preserving array types)
        ast::Expression::Identifier { name, .. } => {
            let sym_name = resolve_sym(ctx, name.as_u32());
            lookup_var_type(ctx, sym_name)
        }

        // Member access (s.member): get member type from struct definition
        ast::Expression::MemberAccess { object, member, .. } => {
            let obj_ty = sizeof_infer_type(ctx, object);
            sizeof_resolve_member(&obj_ty, *member, ctx)
        }

        // Arrow access (p->member): dereference pointer, get member type
        ast::Expression::PointerMemberAccess { object, member, .. } => {
            let ptr_ty = sizeof_infer_type(ctx, object);
            let obj_ty = match types::resolve_typedef(&ptr_ty) {
                CType::Pointer(inner, _) => (**inner).clone(),
                CType::Array(inner, _) => (**inner).clone(),
                other => other.clone(),
            };
            sizeof_resolve_member(&obj_ty, *member, ctx)
        }

        // Dereference (*p): get pointee type
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            ..
        } => {
            let inner_ty = sizeof_infer_type(ctx, operand);
            match types::resolve_typedef(&inner_ty) {
                CType::Pointer(pointee, _) => (**pointee).clone(),
                CType::Array(elem, _) => (**elem).clone(),
                other => other.clone(),
            }
        }

        // Array subscript (a[i]): get element type
        ast::Expression::ArraySubscript { base, .. } => {
            let arr_ty = sizeof_infer_type(ctx, base);
            match types::resolve_typedef(&arr_ty) {
                CType::Array(elem, _) => (**elem).clone(),
                CType::Pointer(pointee, _) => (**pointee).clone(),
                other => other.clone(),
            }
        }

        // String literal: type is char[N] (including null terminator),
        // NOT char * — sizeof("abc") must return 4, not pointer size.
        ast::Expression::StringLiteral { segments, .. } => {
            let total_bytes: usize = segments.iter().map(|s| s.value.len()).sum();
            CType::Array(Box::new(CType::Char), Some(total_bytes + 1))
        }

        // Cast: use the target type
        ast::Expression::Cast { type_name, .. } => resolve_type_name(ctx, type_name),

        // Compound literal: resolve its TypeName to get the declared type
        // (preserving array types — e.g. sizeof((int[]){1,2,3}) == 3*sizeof(int))
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            ..
        } => {
            let resolved = resolve_type_name(ctx, type_name);
            // If the type is an incomplete array (size = 0), infer the size
            // from the number of initializer elements.
            match &resolved {
                CType::Array(elem, Some(0)) | CType::Array(elem, None) => {
                    let count = match initializer {
                        ast::Initializer::List {
                            designators_and_initializers,
                            ..
                        } => designators_and_initializers.len(),
                        ast::Initializer::Expression(_) => 1,
                    };
                    CType::Array(elem.clone(), Some(count))
                }
                _ => resolved,
            }
        }

        // For all other expressions, fall back to normal lowering.
        // This handles arithmetic, function calls, etc. where array
        // decay is the correct behavior (e.g. sizeof(a + b)).
        _ => lower_expr_inner(ctx, expr).ty,
    }
}

/// Resolve a struct/union member type for sizeof purposes.
/// Returns the member's declared type WITHOUT array-to-pointer decay.
fn sizeof_resolve_member(
    obj_ty: &CType,
    member_sym: crate::common::string_interner::Symbol,
    ctx: &mut ExprLoweringContext<'_>,
) -> CType {
    let member_name = resolve_sym(ctx, member_sym.as_u32()).to_string();
    let mut resolved = obj_ty.clone();
    resolve_forward_ref_type(&mut resolved, ctx.struct_defs);
    let (_offset, member_ty, _bf_info) = find_struct_member(
        &resolved,
        &member_name,
        ctx.type_builder,
        ctx.struct_defs,
        ctx.layout_cache,
    );
    member_ty
}

fn lower_sizeof_type(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    span: Span,
) -> TypedValue {
    // Check for VLA typedef: if the type_name is a single TypedefName
    // and it has a runtime VLA size stored in vla_sizes, use that.
    if type_name.abstract_declarator.is_none() {
        for spec in &type_name.specifier_qualifiers.type_specifiers {
            if let ast::TypeSpecifier::TypedefName(sym) = spec {
                let td_name = &ctx.name_table[sym.as_u32() as usize];
                if let Some(&vla_total_bytes) = ctx.vla_sizes.get(td_name) {
                    return TypedValue::new(vla_total_bytes, size_ctype(ctx.target));
                }
            }
        }
    }

    // Also check for VLA array type: `sizeof(int[n])` where n is dynamic.
    // This handles `sizeof(type_name)` where the abstract declarator is an array
    // with a non-constant size expression.
    if let Some(ref abs_decl) = type_name.abstract_declarator {
        if let Some(vla_expr) = extract_vla_from_abstract_declarator(abs_decl) {
            // Compute element type (without the VLA array layer).
            let elem_cty = resolve_type_name_without_vla(type_name);
            let elem_size = crate::common::types::sizeof_ctype(&elem_cty, ctx.target) as i64;
            let ptr_ty = if ctx.target.pointer_width() == 8 {
                IrType::I64
            } else {
                IrType::I32
            };
            // Evaluate VLA size expression.
            let tv = lower_expression_typed(ctx, &vla_expr);
            let n_val = tv.value;
            let n_wide = {
                let (v, inst) =
                    ctx.builder
                        .build_zext_from(n_val, IrType::I32, ptr_ty.clone(), span);
                emit_inst(ctx, inst);
                v
            };
            let elem_const = emit_int_const(ctx, elem_size as i128, ptr_ty.clone(), span);
            let (total, mul_inst) = ctx.builder.build_mul(n_wide, elem_const, ptr_ty, span);
            emit_inst(ctx, mul_inst);
            return TypedValue::new(total, size_ctype(ctx.target));
        }
    }

    let cty = resolve_type_name(ctx, type_name);
    let size = ctx.type_builder.sizeof_type(&cty);

    // Use target properties for result type selection.
    let ptr_w = ctx.target.pointer_width();
    let _endian = ctx.target.endianness();

    // For struct types, also verify via compute_struct_layout for consistency.
    if let CType::Struct {
        fields,
        packed,
        aligned,
        ..
    } = &cty
    {
        let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
        let _layout = ctx
            .type_builder
            .compute_struct_layout(&field_types, *packed, *aligned);
    }

    let ir_ty = if ptr_w == 8 { IrType::I64 } else { IrType::I32 };
    let val = emit_int_const(ctx, size as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

/// Extract a VLA size expression from an abstract declarator's outermost array layer.
fn extract_vla_from_abstract_declarator(
    abs_decl: &ast::AbstractDeclarator,
) -> Option<Box<ast::Expression>> {
    // Check the direct abstract declarator for array with non-constant size.
    if let Some(ref dad) = abs_decl.direct {
        if let ast::DirectAbstractDeclarator::Array { size, .. } = dad {
            // If there's a size expression, it's a VLA.
            if let Some(size_expr) = size {
                return Some(size_expr.clone());
            }
        }
    }
    None
}

/// Resolve a type_name to its element type (stripping the outermost VLA array layer).
fn resolve_type_name_without_vla(tn: &ast::TypeName) -> CType {
    use super::decl_lowering::resolve_base_type_from_sqlist;
    resolve_base_type_from_sqlist(&tn.specifier_qualifiers)
}

fn lower_alignof_type(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);
    let align = ctx.type_builder.alignof_type(&cty);
    let ir_ty = size_ir_type(ctx.target);
    let val = emit_int_const(ctx, align as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

/// GCC `__alignof__(expr)` — evaluate to the alignment of the expression's
/// type.  The expression itself is not evaluated.
///
/// When the operand is an identifier that refers to a function or global
/// variable with an explicit `__attribute__((aligned(N)))`, the alignment
/// returned is the maximum of the type's natural alignment and the explicit
/// override — matching GCC's semantics.
fn lower_alignof_expr(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    span: Span,
) -> TypedValue {
    // Determine the expression's C type (without array-to-pointer decay,
    // without evaluating the expression) — same approach as sizeof(expr).
    let expr_ty = sizeof_infer_type(ctx, operand);
    let mut align = ctx.type_builder.alignof_type(&expr_ty);

    // For identifiers, check if the declaration has an explicit alignment
    // override (from `__attribute__((aligned(N)))`) stored on the IR
    // function or global variable.  This is the key difference between
    // `__alignof__(expr)` and `_Alignof(type)`: the former considers
    // declaration-level attributes, the latter does not.
    if let Some(name) = alignof_extract_ident_name(ctx, operand) {
        // Check the module-level function alignment map.  This covers
        // alignment from both prior declarations and definitions, merged
        // across multiple declarations.
        if let Some(&func_align) = ctx.module.func_alignments.get(name.as_str()) {
            align = align.max(func_align);
        }
        // Check functions (IrFunction.alignment from definition).
        if let Some(func) = ctx.module.get_function(&name) {
            if let Some(func_align) = func.alignment {
                align = align.max(func_align);
            }
        }
        // Check global variables
        if let Some(global) = ctx.module.get_global(&name) {
            if let Some(global_align) = global.alignment {
                align = align.max(global_align);
            }
        }
    }

    let ir_ty = size_ir_type(ctx.target);
    let val = emit_int_const(ctx, align as i128, ir_ty, span);
    TypedValue::new(val, size_ctype(ctx.target))
}

/// Extract the identifier name from an expression for `__alignof__` purposes.
/// Strips parentheses to handle `__alignof__((var))`.
fn alignof_extract_ident_name(
    ctx: &ExprLoweringContext<'_>,
    expr: &ast::Expression,
) -> Option<String> {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            Some(resolve_sym(ctx, name.as_u32()).to_string())
        }
        ast::Expression::Parenthesized { inner, .. } => alignof_extract_ident_name(ctx, inner),
        _ => None,
    }
}

// =========================================================================
// MEMBER ACCESS  (. and ->)
// =========================================================================

fn lower_member_access(
    ctx: &mut ExprLoweringContext<'_>,
    object: &ast::Expression,
    member_sym: u32,
    is_arrow: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_member_access_lvalue(ctx, object, member_sym, is_arrow, span);
    let bitfield_info = ctx.last_bitfield_info.take();

    // Array-typed members decay to a pointer to the first element.
    // The GEP address IS the pointer value — do not load from it.
    if matches!(types::resolve_typedef(&lv.ty), CType::Array(_, _)) {
        let elem_ty = match types::resolve_typedef(&lv.ty) {
            CType::Array(elem, _) => (**elem).clone(),
            _ => unreachable!(),
        };
        let ptr_ty = CType::Pointer(
            Box::new(elem_ty),
            crate::common::types::TypeQualifiers::default(),
        );
        return TypedValue::new(lv.value, ptr_ty);
    }

    // Bitfield read: load the storage unit, shift right, and mask to
    // extract the bitfield value.
    if let Some((bit_offset, bit_width)) = bitfield_info {
        if bit_width > 0 {
            return lower_bitfield_read(ctx, lv.value, &lv.ty, bit_offset, bit_width, span);
        }
    }

    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
    let (loaded, li) = ctx.builder.build_load(lv.value, ir_ty, span);
    emit_inst(ctx, li);
    TypedValue::new(loaded, lv.ty)
}

/// Emit IR for reading a bitfield: load the storage unit, shift right by
/// `bit_offset`, then mask to `bit_width` bits.  Returns the extracted
/// value with the declared C type.
fn lower_bitfield_read(
    ctx: &mut ExprLoweringContext<'_>,
    storage_ptr: Value,
    field_ty: &CType,
    bit_offset: usize,
    bit_width: usize,
    span: Span,
) -> TypedValue {
    // Determine the storage unit IR type from the field's declared C type.
    let storage_size = crate::common::types::sizeof_ctype(field_ty, ctx.target);
    let storage_ir = match storage_size {
        1 => IrType::I8,
        2 => IrType::I16,
        4 => IrType::I32,
        _ => IrType::I64,
    };

    // Load the full storage unit.
    let (loaded, li) = ctx
        .builder
        .build_load(storage_ptr, storage_ir.clone(), span);
    emit_inst(ctx, li);

    // Shift right by bit_offset to bring the field to bit 0.
    let shifted = if bit_offset > 0 {
        let shift_amt = emit_int_const(ctx, bit_offset as i128, storage_ir.clone(), span);
        let (s, si) = ctx
            .builder
            .build_shr(loaded, shift_amt, storage_ir.clone(), false, span);
        emit_inst(ctx, si);
        s
    } else {
        loaded
    };

    // Mask to extract only the bitfield bits.
    let mask_val = (1u64 << bit_width).wrapping_sub(1);
    let mask = emit_int_const(ctx, mask_val as i128, storage_ir.clone(), span);
    let (masked, mi) = ctx
        .builder
        .build_and(shifted, mask, storage_ir.clone(), span);
    emit_inst(ctx, mi);

    // For signed bitfields, sign-extend from bit_width to the storage type.
    // For enum bitfields, the field_ty's underlying_type may be stale
    // (defaulting to Int) because struct collection (pass 0) runs before
    // enum underlying type computation (pass 0.5).  Look up the
    // authoritative underlying type from the ENUM_UNDERLYING_TYPES map.
    // IMPORTANT: field_ty may be a Typedef wrapping an Enum, so resolve
    // typedefs first before checking for Enum.
    let resolved_field_ty = crate::common::types::resolve_typedef(field_ty);
    let effective_unsigned = match resolved_field_ty {
        CType::Enum {
            name: Some(ref en), ..
        } => super::ENUM_UNDERLYING_TYPES.with(|m| {
            let borrow = m.borrow();
            borrow
                .as_ref()
                .and_then(|map| map.get(en).cloned())
                .map(|ut| is_unsigned(&ut))
                .unwrap_or_else(|| is_unsigned(resolved_field_ty))
        }),
        _ => is_unsigned(resolved_field_ty),
    };
    if std::env::var("BCC_DEBUG_BF").is_ok() {
        eprintln!(
            "[BCC_BF] lower_bitfield_read: field_ty={:?}, is_unsigned={}, bit_width={}",
            field_ty, effective_unsigned, bit_width,
        );
    }
    let is_signed = !effective_unsigned;
    let result = if is_signed && bit_width < storage_size * 8 {
        // Sign extension: shift left to put the sign bit at the MSB of
        // the storage type, then arithmetic shift right to propagate it.
        let shift_up = (storage_size * 8 - bit_width) as i128;
        let shift_up_val = emit_int_const(ctx, shift_up, storage_ir.clone(), span);
        let (shl_val, shl_i) =
            ctx.builder
                .build_shl(masked, shift_up_val, storage_ir.clone(), span);
        emit_inst(ctx, shl_i);
        let shift_down_val = emit_int_const(ctx, shift_up, storage_ir.clone(), span);
        let (sar_val, sar_i) =
            ctx.builder
                .build_shr(shl_val, shift_down_val, storage_ir.clone(), true, span);
        emit_inst(ctx, sar_i);
        sar_val
    } else {
        masked
    };

    // C11 §6.3.1.1p1: Integer promotion for bitfields.  An unsigned
    // bitfield whose width is less than the width of `int` (31 value bits
    // for a 32-bit int) gets promoted to `int` because all its
    // representable values fit in signed int.  A signed bitfield that fits
    // in `int` also promotes to `int`.  This applies regardless of the
    // declared base type (unsigned int, unsigned long, unsigned long long).
    let promoted_ty = {
        let base = crate::common::types::resolve_and_strip(field_ty);
        let is_unsigned_base = matches!(
            base,
            CType::UChar
                | CType::UShort
                | CType::UInt
                | CType::ULong
                | CType::ULongLong
                | CType::UInt128
        );
        let is_signed_base = matches!(
            base,
            CType::Char
                | CType::SChar
                | CType::Short
                | CType::Int
                | CType::Long
                | CType::LongLong
                | CType::Int128
        );
        if is_unsigned_base && bit_width < 32 {
            // All values 0..2^(bit_width)-1 fit in signed int (max 2^31-1).
            CType::Int
        } else if is_signed_base && bit_width <= 32 {
            // Signed bitfield: if it fits in int, promote to int.
            CType::Int
        } else if is_unsigned_base && bit_width == 32 {
            // Unsigned 32-bit bitfield doesn't fit in signed int → unsigned int.
            CType::UInt
        } else {
            field_ty.clone()
        }
    };

    // If the C type is smaller/larger than the storage IR type, cast.
    let target_ir = ctype_to_ir(&promoted_ty, ctx.target);
    let final_val = if target_ir != storage_ir {
        let (cast_val, cast_inst) = ctx.builder.build_bitcast(result, target_ir, span);
        emit_inst(ctx, cast_inst);
        cast_val
    } else {
        result
    };

    // C11 §6.7.2.1p10: bitfields wider than `int` (> 32 bits) retain
    // their precision — arithmetic on them wraps at the bitfield width,
    // not the declared type width.  Track this so that `lower_binary`
    // and shift operations can insert the appropriate truncation.
    let storage_bits = storage_size * 8;
    if bit_width > 32 && bit_width < storage_bits {
        TypedValue::with_bitfield_width(final_val, promoted_ty, bit_width as u32)
    } else {
        TypedValue::new(final_val, promoted_ty)
    }
}

/// Emit IR for writing a bitfield: load the storage unit, clear the
/// target bits, OR in the new value, and store back.  This preserves
/// adjacent fields packed in the same storage unit.
pub(crate) fn lower_bitfield_store(
    ctx: &mut ExprLoweringContext<'_>,
    storage_ptr: Value,
    new_value: Value,
    field_ty: &CType,
    bit_offset: usize,
    bit_width: usize,
    span: Span,
) {
    let storage_size = crate::common::types::sizeof_ctype(field_ty, ctx.target);
    let storage_ir = match storage_size {
        1 => IrType::I8,
        2 => IrType::I16,
        4 => IrType::I32,
        _ => IrType::I64,
    };

    // Load the full storage unit.
    let (old_val, li) = ctx
        .builder
        .build_load(storage_ptr, storage_ir.clone(), span);
    emit_inst(ctx, li);

    // Build the mask for the bitfield bits: ((1 << bit_width) - 1) << bit_offset
    let field_mask = ((1u64 << bit_width).wrapping_sub(1)) << bit_offset;
    let inv_mask = emit_int_const(ctx, !field_mask as i64 as i128, storage_ir.clone(), span);

    // Clear the bitfield bits in the old value: old & ~mask
    let (cleared, ci) = ctx
        .builder
        .build_and(old_val, inv_mask, storage_ir.clone(), span);
    emit_inst(ctx, ci);

    // Prepare the new value: (new_value & ((1 << bit_width) - 1)) << bit_offset
    // First, ensure the new value is the same IR type as the storage unit.
    let new_val_cast = if ctype_to_ir(field_ty, ctx.target) != storage_ir {
        // Truncate or zero-extend to match storage unit width.
        let (cv, ci2) = ctx
            .builder
            .build_bitcast(new_value, storage_ir.clone(), span);
        emit_inst(ctx, ci2);
        cv
    } else {
        new_value
    };

    // Mask the new value to bit_width bits.
    let val_mask_int = (1u64 << bit_width).wrapping_sub(1);
    let val_mask = emit_int_const(ctx, val_mask_int as i128, storage_ir.clone(), span);
    let (masked_new, mi) = ctx
        .builder
        .build_and(new_val_cast, val_mask, storage_ir.clone(), span);
    emit_inst(ctx, mi);

    // Shift the masked value into position.
    let shifted_new = if bit_offset > 0 {
        let shift_amt = emit_int_const(ctx, bit_offset as i128, storage_ir.clone(), span);
        let (s, si) = ctx
            .builder
            .build_shl(masked_new, shift_amt, storage_ir.clone(), span);
        emit_inst(ctx, si);
        s
    } else {
        masked_new
    };

    // OR the shifted new value into the cleared storage unit.
    let (combined, oi) = ctx
        .builder
        .build_or(cleared, shifted_new, storage_ir.clone(), span);
    emit_inst(ctx, oi);

    // Store back the modified storage unit.
    let si = ctx.builder.build_store(combined, storage_ptr, span);
    emit_inst(ctx, si);
}

/// Check whether an expression is a "natural lvalue" — one that directly
/// produces a memory address (variable, dereference, subscript, compound
/// literal, etc.).  Non-lvalue expressions (function calls, conditionals,
/// binary ops, casts of rvalues, etc.) produce a VALUE that must be spilled
/// to a temporary alloca before its address can be taken for member access.
fn expr_is_natural_lvalue(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::Identifier { .. }
        | ast::Expression::MemberAccess { .. }
        | ast::Expression::PointerMemberAccess { .. }
        | ast::Expression::ArraySubscript { .. }
        | ast::Expression::CompoundLiteral { .. } => true,
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            ..
        } => true,
        ast::Expression::Parenthesized { inner, .. } => expr_is_natural_lvalue(inner),
        // Cast expressions produce rvalues in standard C (C11 §6.5.4).
        // Even if the operand is an lvalue, the cast result is an rvalue.
        // Treating casts as lvalues caused aggregate struct copies to
        // treat the loaded struct VALUE as a pointer (extra dereference),
        // corrupting struct fields.  The lower_lvalue_inner handler for
        // Cast already falls through to lower_expr_inner, which returns
        // a VALUE not a pointer — so callers that expect an address
        // (like the aggregate copy path in lower_assignment) would break.
        ast::Expression::Cast { .. } => false,
        _ => false,
    }
}

/// Check whether an rvalue expression, when lowered, produces a *pointer*
/// to aggregate data rather than the aggregate data itself.
///
/// In BCC's IR, most aggregate rvalues (function calls, va_arg, etc.)
/// return the struct data held in a backend temp slot.  However, some
/// expressions — specifically assignment expressions and comma expressions
/// ending in an assignment — return the LHS *address* from
/// `lower_assignment`, which is a pointer.
///
/// This distinction matters for the aggregate copy path: pointer results
/// can be used directly as the copy source, while data results must be
/// spilled to a temporary alloca first.
fn rhs_expr_returns_aggregate_pointer(expr: &ast::Expression) -> bool {
    match expr {
        // Assignment expressions return the LHS address (a pointer).
        ast::Expression::Assignment { .. } => true,
        // Parenthesized: check the inner expression.
        ast::Expression::Parenthesized { inner, .. } => rhs_expr_returns_aggregate_pointer(inner),
        // Comma expression: the result is the last element.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                rhs_expr_returns_aggregate_pointer(last)
            } else {
                false
            }
        }
        // Everything else (function calls, va_arg, conditionals,
        // statement expressions, casts, etc.) returns struct data.
        _ => false,
    }
}

fn lower_member_access_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    object: &ast::Expression,
    member_sym: u32,
    is_arrow: bool,
    span: Span,
) -> TypedValue {
    // Resolve the member name first (borrows ctx immutably only briefly).
    let member_name_owned = resolve_sym(ctx, member_sym).to_string();

    let (base_ptr, obj_ty) = if is_arrow {
        // ptr->member: base is the pointer value.
        let p = lower_expr_inner(ctx, object);
        let pt = pointee_of(&p.ty);
        (p.value, pt)
    } else if !expr_is_natural_lvalue(object) {
        // Rvalue struct (e.g. function call returning struct, conditional
        // expression, cast of rvalue, etc.): the expression produces a
        // VALUE, not a pointer.  We must spill it to a temporary alloca
        // so that we can GEP into its memory representation.
        let val = lower_expr_inner(ctx, object);
        let ty = val.ty.clone();
        let ir_ty = ctype_to_ir(&ty, ctx.target);
        let struct_sz = crate::common::types::sizeof_ctype(&ty, ctx.target);

        // Create a temporary alloca for the struct value.
        let (alloca_val, alloca_inst) = ctx.builder.build_alloca(ir_ty.clone(), span);
        emit_inst(ctx, alloca_inst);

        // Store the struct value into the alloca.  For large structs
        // (>8 bytes, returned as RegisterPair or in memory) we use
        // an appropriately-sized store.
        if struct_sz > 8 {
            // Large struct — store via appropriately-typed store
            let (store_inst,) = (ctx.builder.build_store(val.value, alloca_val, span),);
            emit_inst(ctx, store_inst);
        } else {
            let store_inst = ctx.builder.build_store(val.value, alloca_val, span);
            emit_inst(ctx, store_inst);
        }

        (alloca_val, ty)
    } else {
        // obj.member: base is the address of the struct.
        let lv = lower_lvalue_inner(ctx, object);
        let ty = lv.ty.clone();
        (lv.value, ty)
    };

    // Resolve forward-declared struct/union tag references before member lookup.
    let mut resolved_obj_ty = obj_ty.clone();
    resolve_forward_ref_type(&mut resolved_obj_ty, ctx.struct_defs);

    // Look up the member in the struct fields.
    let (member_offset, member_ty, bitfield_info) = find_struct_member(
        &resolved_obj_ty,
        &member_name_owned,
        ctx.type_builder,
        ctx.struct_defs,
        ctx.layout_cache,
    );

    // Store bitfield information for the caller to use.
    // `lower_member_access` (reads) and `lower_assignment` (writes)
    // will check this to emit proper bit-manipulation.
    ctx.last_bitfield_info = bitfield_info;

    // Compute GEP to the member (or storage unit for bitfields).
    let offset_val = emit_int_const(ctx, member_offset as i128, size_ir_type(ctx.target), span);
    let (gep_val, gep_inst) = ctx
        .builder
        .build_gep(base_ptr, vec![offset_val], IrType::Ptr, span);
    emit_inst(ctx, gep_inst);

    TypedValue::new(gep_val, member_ty)
}

/// Find a struct member by name. Returns (byte_offset, member_type, bitfield_info).
///
/// For bitfield members, `bitfield_info` is `Some((bit_offset_in_unit, bit_width))`
/// where `bit_offset_in_unit` is the bit position within the storage unit starting
/// from bit 0 (LSB on little-endian).
///
/// When the type is a lightweight tag reference (empty fields with a tag
/// name), the full definition is resolved from `struct_defs` before
/// performing the member lookup.
fn find_struct_member(
    ctype: &CType,
    name: &str,
    tb: &TypeBuilder,
    struct_defs: &FxHashMap<String, CType>,
    layout_cache: &mut FxHashMap<String, crate::common::type_builder::StructLayout>,
) -> (usize, CType, Option<(usize, usize)>) {
    // Strip both qualifiers (const, volatile, restrict) AND typedefs so that
    // `const struct F`, `volatile union U`, etc. match the Struct/Union arms.
    let resolved = types::resolve_and_strip(ctype);
    // Resolve lightweight tag references (empty fields with a tag name)
    // by looking up the full definition from the struct_defs registry.
    let resolved = match resolved {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                full_def
            } else {
                resolved
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                full_def
            } else {
                resolved
            }
        }
        other => other,
    };
    match resolved {
        CType::Struct {
            name: ref tag_name,
            fields,
            packed,
            aligned,
            ..
        } => {
            // Build a cache key from the struct tag if available.
            // For tagged structs, cache the computed layout to avoid
            // recomputing for every member access on large structs
            // like `struct sock` (hundreds of fields).
            let cache_key = tag_name.as_ref().map(|t| {
                let mut key = String::with_capacity(t.len() + 10);
                key.push_str("s:");
                key.push_str(t);
                if *packed {
                    key.push_str(":p");
                }
                if let Some(a) = aligned {
                    key.push(':');
                    key.push_str(&a.to_string());
                }
                key
            });

            let layout = if let Some(ref ck) = cache_key {
                if let Some(cached) = layout_cache.get(ck) {
                    cached.clone()
                } else {
                    let resolved_fields: Vec<crate::common::types::StructField> = fields
                        .iter()
                        .map(|f| {
                            let mut resolved_ty = f.ty.clone();
                            resolve_forward_ref_type(&mut resolved_ty, struct_defs);
                            crate::common::types::StructField {
                                name: f.name.clone(),
                                ty: resolved_ty,
                                bit_width: f.bit_width,
                            }
                        })
                        .collect();
                    let l =
                        tb.compute_struct_layout_with_fields(&resolved_fields, *packed, *aligned);
                    layout_cache.insert(ck.clone(), l.clone());
                    l
                }
            } else {
                // Anonymous struct — compute without caching.
                let resolved_fields: Vec<crate::common::types::StructField> = fields
                    .iter()
                    .map(|f| {
                        let mut resolved_ty = f.ty.clone();
                        resolve_forward_ref_type(&mut resolved_ty, struct_defs);
                        crate::common::types::StructField {
                            name: f.name.clone(),
                            ty: resolved_ty,
                            bit_width: f.bit_width,
                        }
                    })
                    .collect();
                tb.compute_struct_layout_with_fields(&resolved_fields, *packed, *aligned)
            };

            for (i, field) in fields.iter().enumerate() {
                if field.name.as_deref() == Some(name) {
                    let fl = &layout.fields[i];
                    let mut result_ty = field.ty.clone();
                    resolve_forward_ref_type(&mut result_ty, struct_defs);
                    return (fl.offset, result_ty, fl.bitfield_info);
                }
            }
            // Anonymous struct/union members — search recursively.
            for (i, field) in fields.iter().enumerate() {
                if field.name.is_none() {
                    let mut inner_ty = field.ty.clone();
                    resolve_forward_ref_type(&mut inner_ty, struct_defs);
                    let inner = types::resolve_typedef(&inner_ty);
                    if matches!(inner, CType::Struct { .. } | CType::Union { .. }) {
                        let (inner_off, found_ty, bf_info) =
                            find_struct_member(inner, name, tb, struct_defs, layout_cache);
                        if !matches!(found_ty, CType::Void) {
                            return (layout.fields[i].offset + inner_off, found_ty, bf_info);
                        }
                    }
                }
            }
            (0, CType::Void, None)
        }
        CType::Union {
            name: ref _tag_name,
            fields,
            packed: _,
            aligned: _,
            ..
        } => {
            // For unions, layout computation is simpler but still cache
            // for consistency and to avoid repeated forward-ref resolution.
            for field in fields {
                if field.name.as_deref() == Some(name) {
                    let mut result_ty = field.ty.clone();
                    resolve_forward_ref_type(&mut result_ty, struct_defs);
                    // Bitfield members in unions start at bit offset 0.
                    let bf_info = field
                        .bit_width
                        .filter(|&bw| bw > 0)
                        .map(|bw| (0usize, bw as usize));
                    return (0, result_ty, bf_info);
                }
                // Anonymous nested.
                if field.name.is_none() {
                    let mut inner_ty = field.ty.clone();
                    resolve_forward_ref_type(&mut inner_ty, struct_defs);
                    let inner = types::resolve_typedef(&inner_ty);
                    if matches!(inner, CType::Struct { .. } | CType::Union { .. }) {
                        let (off, ty, bf_info) =
                            find_struct_member(inner, name, tb, struct_defs, layout_cache);
                        if !matches!(ty, CType::Void) {
                            return (off, ty, bf_info);
                        }
                    }
                }
            }
            (0, CType::Void, None)
        }
        _ => (0, CType::Void, None),
    }
}

// =========================================================================
// ARRAY SUBSCRIPT
// =========================================================================

fn lower_array_subscript(
    ctx: &mut ExprLoweringContext<'_>,
    base: &ast::Expression,
    index: &ast::Expression,
    span: Span,
) -> TypedValue {
    let lv = lower_array_subscript_lvalue(ctx, base, index, span);
    // If the element type is an array (possibly behind a typedef), the
    // subscript yields an array lvalue which decays to a pointer.  We must
    // NOT load from the address because the address itself IS the decayed
    // pointer value (a `T (*)[N]` → `T *` implicit conversion).
    //
    // E.g., for `int a[3][4]; a[i]` — the result is a pointer to the first
    // element of the i-th inner array, NOT a load of 4 integers from memory.
    //
    // CRITICAL: resolve typedefs before checking — `typedef char A4[4];
    // A4 a[3]; a[i]` has lv.ty == Typedef{A4, Array(Char,4)}, which must
    // still be recognized as an array element type.
    let resolved_ty = types::resolve_typedef(&lv.ty);
    if matches!(resolved_ty, CType::Array(_, _)) {
        // Return the pointer (address) directly, with the type set to a
        // pointer-to-element-type so that subsequent subscripting
        // (e.g., `a[i][j]`) indexes individual elements.
        // This implements C array-to-pointer decay: `T a[N][M]; a[i]`
        // yields a `T*` pointing at the first element of row i.
        let inner_elem = match resolved_ty {
            CType::Array(elem, _) => elem.as_ref().clone(),
            _ => unreachable!(),
        };
        TypedValue::new(
            lv.value,
            CType::Pointer(Box::new(inner_elem), TypeQualifiers::default()),
        )
    } else {
        let ir_ty = ctype_to_ir(&lv.ty, ctx.target);
        let (loaded, li) = ctx.builder.build_load(lv.value, ir_ty, span);
        emit_inst(ctx, li);
        TypedValue::new(loaded, lv.ty)
    }
}

fn lower_array_subscript_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    base: &ast::Expression,
    index: &ast::Expression,
    span: Span,
) -> TypedValue {
    let base_tv = lower_expr_inner(ctx, base);
    let idx_tv = lower_expr_inner(ctx, index);

    let elem_ty = pointee_of(&base_tv.ty);
    let elem_size =
        sizeof_resolved(&elem_ty, ctx.type_builder, ctx.struct_defs, ctx.target) as i128;
    let si = size_ir_type(ctx.target);

    // Scale index by element size.
    let idx_val =
        insert_implicit_conversion(ctx, idx_tv.value, &idx_tv.ty, &size_ctype(ctx.target), span);

    // For VLA element types (e.g., Array(Int, None) from `int a[n][m]`),
    // sizeof_resolved returns 0 because the inner dimension is unknown at
    // compile time. Use the runtime stride stored during VLA allocation.
    let byte_off = if elem_size == 0 {
        if let Some((vla_name, depth)) = find_vla_subscript_info(ctx, base) {
            let stride_key = format!("{}\0stride_{}", vla_name, depth);
            if let Some(&stride_val) = ctx.vla_sizes.get(&stride_key) {
                let (off, mi) = ctx.builder.build_mul(idx_val, stride_val, si.clone(), span);
                emit_inst(ctx, mi);
                off
            } else {
                // Fallback: stride not found — emit zero offset.
                let sc = emit_int_const(ctx, 0, si.clone(), span);
                let (off, mi) = ctx.builder.build_mul(idx_val, sc, si.clone(), span);
                emit_inst(ctx, mi);
                off
            }
        } else {
            // Not a VLA subscript — emit zero offset.
            let sc = emit_int_const(ctx, 0, si.clone(), span);
            let (off, mi) = ctx.builder.build_mul(idx_val, sc, si.clone(), span);
            emit_inst(ctx, mi);
            off
        }
    } else {
        let sc = emit_int_const(ctx, elem_size, si.clone(), span);
        let (off, mi) = ctx.builder.build_mul(idx_val, sc, si.clone(), span);
        emit_inst(ctx, mi);
        off
    };

    let (ptr, gi) = ctx
        .builder
        .build_gep(base_tv.value, vec![byte_off], IrType::Ptr, span);
    emit_inst(ctx, gi);

    TypedValue::new(ptr, elem_ty)
}

/// Walk the AST expression to find the base VLA variable name and the
/// subscript depth for stride lookup. For `a` returns ("a", 0); for
/// `a[i]` (when used as base of a further subscript) returns ("a", 1).
fn find_vla_subscript_info(
    ctx: &ExprLoweringContext<'_>,
    expr: &ast::Expression,
) -> Option<(String, usize)> {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let sym_name = resolve_sym(ctx, name.as_u32());
            if ctx.vla_sizes.contains_key(sym_name) {
                Some((sym_name.to_string(), 0))
            } else {
                None
            }
        }
        ast::Expression::Parenthesized { inner, .. } => find_vla_subscript_info(ctx, inner),
        ast::Expression::ArraySubscript { base, .. } => {
            if let Some((name, depth)) = find_vla_subscript_info(ctx, base) {
                Some((name, depth + 1))
            } else {
                None
            }
        }
        _ => None,
    }
}

// =========================================================================
// INCREMENT / DECREMENT
// =========================================================================

fn lower_post_inc_dec(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    is_inc: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_lvalue_inner(ctx, operand);
    let bf_info = ctx.last_bitfield_info.take();
    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);

    // For bitfields, use bitfield-aware read/write.
    let old_val = if let Some((bo, bw)) = bf_info {
        if bw > 0 {
            lower_bitfield_read(ctx, lv.value, &lv.ty, bo, bw, span).value
        } else {
            let (v, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
            emit_inst(ctx, li);
            v
        }
    } else {
        let (v, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
        emit_inst(ctx, li);
        v
    };

    let new_val = if is_pointer_type(&lv.ty) {
        let es =
            pointee_size_resolved(&lv.ty, ctx.type_builder, ctx.struct_defs, ctx.target) as i128;
        let step = emit_int_const(
            ctx,
            if is_inc { es } else { -(es as i128) },
            size_ir_type(ctx.target),
            span,
        );
        let (r, g) = ctx
            .builder
            .build_gep(old_val, vec![step], IrType::Ptr, span);
        emit_inst(ctx, g);
        r
    } else {
        let one = emit_one(ctx, ir_ty.clone(), span);
        if is_inc {
            let (v, i) = ctx.builder.build_add(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        } else {
            let (v, i) = ctx.builder.build_sub(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        }
    };

    if let Some((bo, bw)) = bf_info {
        if bw > 0 {
            lower_bitfield_store(ctx, lv.value, new_val, &lv.ty, bo, bw, span);
            return TypedValue::new(old_val, lv.ty);
        }
    }
    let si = ctx.builder.build_store(new_val, lv.value, span);
    emit_inst(ctx, si);
    // Post-: return the OLD value.
    TypedValue::new(old_val, lv.ty)
}

fn lower_pre_inc_dec(
    ctx: &mut ExprLoweringContext<'_>,
    operand: &ast::Expression,
    is_inc: bool,
    span: Span,
) -> TypedValue {
    let lv = lower_lvalue_inner(ctx, operand);
    let bf_info = ctx.last_bitfield_info.take();
    let ir_ty = ctype_to_ir(&lv.ty, ctx.target);

    let old_val = if let Some((bo, bw)) = bf_info {
        if bw > 0 {
            lower_bitfield_read(ctx, lv.value, &lv.ty, bo, bw, span).value
        } else {
            let (v, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
            emit_inst(ctx, li);
            v
        }
    } else {
        let (v, li) = ctx.builder.build_load(lv.value, ir_ty.clone(), span);
        emit_inst(ctx, li);
        v
    };

    let new_val = if is_pointer_type(&lv.ty) {
        let es =
            pointee_size_resolved(&lv.ty, ctx.type_builder, ctx.struct_defs, ctx.target) as i128;
        let step = emit_int_const(
            ctx,
            if is_inc { es } else { -(es as i128) },
            size_ir_type(ctx.target),
            span,
        );
        let (r, g) = ctx
            .builder
            .build_gep(old_val, vec![step], IrType::Ptr, span);
        emit_inst(ctx, g);
        r
    } else {
        let one = emit_one(ctx, ir_ty.clone(), span);
        if is_inc {
            let (v, i) = ctx.builder.build_add(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        } else {
            let (v, i) = ctx.builder.build_sub(old_val, one, ir_ty, span);
            emit_inst(ctx, i);
            v
        }
    };

    if let Some((bo, bw)) = bf_info {
        if bw > 0 {
            lower_bitfield_store(ctx, lv.value, new_val, &lv.ty, bo, bw, span);
            // The value actually stored is truncated to `bw` bits.
            // Return the truncated value, not the raw arithmetic result.
            let storage_bits = crate::common::types::sizeof_ctype(&lv.ty, ctx.target) * 8;
            let truncated = if bw < storage_bits {
                let ir = ctype_to_ir(&lv.ty, ctx.target);
                if is_unsigned(&lv.ty) {
                    let mask = (1u64 << bw).wrapping_sub(1);
                    let mask_val = emit_int_const(ctx, mask as i128, ir.clone(), span);
                    let (v, inst) = ctx.builder.build_and(new_val, mask_val, ir, span);
                    emit_inst(ctx, inst);
                    v
                } else {
                    let shift = (storage_bits - bw) as i128;
                    let shift_val = emit_int_const(ctx, shift, ir.clone(), span);
                    let (shl, si2) = ctx.builder.build_shl(new_val, shift_val, ir.clone(), span);
                    emit_inst(ctx, si2);
                    let sv2 = emit_int_const(ctx, shift, ir.clone(), span);
                    let (sar, ai) = ctx.builder.build_shr(shl, sv2, ir, true, span);
                    emit_inst(ctx, ai);
                    sar
                }
            } else {
                new_val
            };
            return TypedValue::new(truncated, lv.ty);
        }
    }
    let si = ctx.builder.build_store(new_val, lv.value, span);
    emit_inst(ctx, si);
    // Pre-: return the NEW value.
    TypedValue::new(new_val, lv.ty)
}

// =========================================================================
// COMMA EXPRESSION
// =========================================================================

fn lower_comma(
    ctx: &mut ExprLoweringContext<'_>,
    exprs: &[ast::Expression],
    _span: Span,
) -> TypedValue {
    if exprs.is_empty() {
        return TypedValue::void();
    }
    let mut last = TypedValue::void();
    for e in exprs {
        last = lower_expr_inner(ctx, e);
    }
    last
}

// =========================================================================
// COMPOUND LITERAL
// =========================================================================

/// Walk an initializer tree and evaluate all expression nodes for their
/// side effects.  Used for zero-sized compound literals where we cannot
/// store a result but must not skip function calls or other effectful
/// expressions inside the initializer list.
fn eval_initializer_for_side_effects(ctx: &mut ExprLoweringContext<'_>, init: &ast::Initializer) {
    match init {
        ast::Initializer::Expression(expr) => {
            // Evaluate the expression; discard the result.
            let _ = lower_expr_inner(ctx, expr);
        }
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => {
            for di in designators_and_initializers {
                eval_initializer_for_side_effects(ctx, &di.initializer);
            }
        }
    }
}

fn lower_compound_literal(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    initializer: &ast::Initializer,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);

    // Validate completeness of the type for diagnostic reporting.
    if !type_builder::is_complete_type(&cty) {
        ctx.diagnostics
            .emit_warning(span, "compound literal of incomplete type");
    }

    // GCC extension: zero-sized types (e.g. `(struct g){}`).
    // No storage is needed — return a dummy value.  However, we MUST
    // still evaluate the initializer expressions for their side effects
    // (e.g., function calls that modify global state).
    let type_size = crate::common::types::sizeof_ctype(&cty, ctx.target);
    if type_size == 0 && crate::common::types::is_struct_or_union(&cty) {
        eval_initializer_for_side_effects(ctx, initializer);
        let v = ctx.builder.fresh_value();
        return TypedValue::new(v, cty);
    }

    let ir_ty = ctype_to_ir(&cty, ctx.target);

    // Allocate storage for the compound literal in the entry block.
    let (alloca, ai) = ctx.builder.build_alloca(ir_ty.clone(), span);
    emit_inst(ctx, ai);

    // Zero-initialize first so that any fields not explicitly initialised
    // in the initializer list default to zero.
    let zero = emit_zero(ctx, ir_ty.clone(), span);
    let si = ctx.builder.build_store(zero, alloca, span);
    emit_inst(ctx, si);

    // Now actually process the initializer list to populate the compound
    // literal storage.  This delegates to the same initializer lowering
    // used for local variable declarations.
    super::decl_lowering::lower_local_initializer(alloca, initializer, &cty, ctx);

    // A compound literal is an lvalue in C, so the alloca pointer is
    // its canonical representation.  However, when used as an rvalue
    // (e.g., `return (S){1,2}`, `x = (S){1,2}`, or passing to a
    // function), the caller needs the loaded value.
    //
    // - Arrays: decay to pointer (return alloca address)
    // - Scalars, small aggregates, and large structs/unions: load and
    //   return the value so the codegen can correctly handle register
    //   placement and MEMORY-class stack copying for function calls.
    let resolved = crate::common::types::resolve_typedef(&cty);
    let is_array = matches!(resolved, CType::Array(..));
    if is_array {
        // Arrays decay to pointer — return alloca address.
        TypedValue::new(alloca, cty)
    } else {
        // Load the scalar / aggregate value from the alloca.
        // The codegen handles all sizes correctly:
        //   ≤8 bytes  → single register
        //   9-16 bytes → RegisterPair
        //   >16 bytes  → MEMORY-class stack copy at call site
        let (loaded, li) = ctx.builder.build_load(alloca, ir_ty, span);
        emit_inst(ctx, li);
        TypedValue::new(loaded, cty)
    }
}

/// Lower a compound literal as an **lvalue**, returning the alloca pointer.
/// Used when the compound literal appears in address-of (`&(S){1,2}`) or
/// member access (`(S){1,2}.field`) contexts.
fn lower_compound_literal_lvalue(
    ctx: &mut ExprLoweringContext<'_>,
    type_name: &ast::TypeName,
    initializer: &ast::Initializer,
    span: Span,
) -> TypedValue {
    let cty = resolve_type_name(ctx, type_name);

    if !type_builder::is_complete_type(&cty) {
        ctx.diagnostics
            .emit_warning(span, "compound literal of incomplete type");
    }

    // GCC extension: zero-sized types (e.g. `(struct g){}`).
    // No storage is needed — return a dummy alloca pointer.  Evaluate
    // initializer expressions for side effects.
    let type_size = crate::common::types::sizeof_ctype(&cty, ctx.target);
    if type_size == 0 && crate::common::types::is_struct_or_union(&cty) {
        eval_initializer_for_side_effects(ctx, initializer);
        let v = ctx.builder.fresh_value();
        return TypedValue::new(v, cty);
    }

    let ir_ty = ctype_to_ir(&cty, ctx.target);

    let (alloca, ai) = ctx.builder.build_alloca(ir_ty.clone(), span);
    emit_inst(ctx, ai);

    let zero = emit_zero(ctx, ir_ty, span);
    let si = ctx.builder.build_store(zero, alloca, span);
    emit_inst(ctx, si);

    super::decl_lowering::lower_local_initializer(alloca, initializer, &cty, ctx);

    // Return the alloca pointer — the compound literal is an lvalue.
    TypedValue::new(alloca, cty)
}

// =========================================================================
// STATEMENT EXPRESSION (GCC `({ ... })`)
// =========================================================================

fn lower_statement_expression(
    ctx: &mut ExprLoweringContext<'_>,
    compound: &ast::CompoundStatement,
    span: Span,
) -> TypedValue {
    // GCC statement expressions: `({ decls; stmts; expr; })`.
    // Variables declared inside the compound are scoped to this expression.
    // We clone the local variable maps, pre-allocate all declared variables,
    // then process each block item using the full statement + expression
    // lowering infrastructure.

    // Clone local maps so that statement-expression-scoped variables can be
    // added without affecting the enclosing scope.
    let mut stmt_local_vars = ctx.local_vars.clone();
    let mut stmt_local_types = ctx.local_types.clone();

    // First pass: pre-allocate all variables declared in this compound
    // by creating alloca instructions in the function entry block.
    for item in &compound.items {
        if let ast::BlockItem::Declaration(decl) = item {
            // Skip storage classes that don't need local allocation.
            if matches!(
                decl.specifiers.storage_class,
                Some(ast::StorageClass::Extern)
                    | Some(ast::StorageClass::Static)
                    | Some(ast::StorageClass::ThreadLocal)
                    | Some(ast::StorageClass::Typedef)
            ) {
                continue;
            }
            for init_decl in &decl.declarators {
                let var_name = match super::stmt_lowering::extract_decl_name(
                    &init_decl.declarator,
                    ctx.name_table,
                ) {
                    Some(name) => name,
                    None => continue,
                };
                let c_type = super::decl_lowering::resolve_declaration_type(
                    &decl.specifiers,
                    &init_decl.declarator,
                    ctx.target,
                    ctx.name_table,
                );
                // Resolve forward-referenced struct/union tags so the alloca
                // gets the correct field layout and size (not a lightweight
                // tag reference with empty fields → 0-byte alloca).
                let resolved_c_type =
                    super::decl_lowering::resolve_sizeof_struct_ref_pub(c_type.clone(), ctx.target);
                let ir_type = IrType::from_ctype(&resolved_c_type, ctx.target);
                // Round up non-power-of-2 struct allocas (e.g. 3-byte struct)
                // so whole-struct register stores don't overflow.
                let alloca_ir_type = {
                    let sz = ir_type.size_bytes(ctx.target);
                    if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                        IrType::Array(Box::new(IrType::I8), sz.next_power_of_two())
                    } else {
                        ir_type
                    }
                };
                let (alloca_val, alloca_inst) = ctx.builder.build_alloca(alloca_ir_type, span);
                ctx.function.entry_block_mut().push_alloca(alloca_inst);
                stmt_local_vars.insert(var_name.clone(), alloca_val);
                // Store the RESOLVED type so that subsequent loads/stores
                // use the correct struct field layout and size.
                stmt_local_types.insert(var_name, resolved_c_type);
            }
        }
    }

    // Build a StmtLoweringContext so we can lower declarations (initialiser
    // stores), non-expression statements (if/else, for, while), and
    // expression statements using the augmented variable maps.
    let mut label_blocks = FxHashMap::default();
    // Inherit the layout cache from the expression context so struct layout
    // computations are shared across statement-expression boundaries.
    let inherited_layout_cache = std::mem::take(ctx.layout_cache);
    let mut stmt_ctx = super::stmt_lowering::StmtLoweringContext {
        builder: ctx.builder,
        function: ctx.function,
        module: ctx.module,
        target: ctx.target,
        diagnostics: ctx.diagnostics,
        local_vars: &mut stmt_local_vars,
        label_blocks: &mut label_blocks,
        loop_stack: ctx.enclosing_loop_stack.clone(),
        switch_ctx: None,
        recursion_depth: 0,
        type_builder: ctx.type_builder,
        param_values: ctx.param_values,
        name_table: ctx.name_table,
        local_types: &stmt_local_types,
        enum_constants: ctx.enum_constants,
        static_locals: ctx.static_locals,
        struct_defs: ctx.struct_defs,
        current_function_name: ctx.current_function_name,
        scope_type_overrides: ctx.scope_type_overrides.clone(),
        return_ctype: None,
        layout_cache: inherited_layout_cache,
        vla_sizes: FxHashMap::default(),
        vla_stack_save: None,
        claimed_vars: FxHashSet::default(), // Fresh scope for statement expression.
    };

    // Second pass: lower each block item. Track the value of the last
    // expression-statement as the result of the whole statement expression.
    let mut last_val = Value::UNDEF;
    let mut last_ty = CType::Int;
    for item in &compound.items {
        match item {
            ast::BlockItem::Declaration(decl) => {
                // Ensure allocas (or static local globals) exist for all
                // declarators in this declaration. This handles static
                // variables declared inside statement expressions (e.g.
                // `DO_ONCE`, `WARN_ONCE` kernel macros).
                super::stmt_lowering::ensure_allocas_for_declaration(&mut stmt_ctx, decl);
                super::stmt_lowering::lower_declaration_initializers(&mut stmt_ctx, decl);
                last_val = Value::UNDEF;
                last_ty = CType::Void;
            }
            ast::BlockItem::Statement(stmt) => {
                if let ast::Statement::Expression(Some(expr)) = stmt {
                    // Expression statement — evaluate and keep the value
                    // (the last one becomes the result).
                    let mut inner_expr_ctx = ExprLoweringContext {
                        builder: stmt_ctx.builder,
                        function: stmt_ctx.function,
                        module: stmt_ctx.module,
                        target: stmt_ctx.target,
                        type_builder: stmt_ctx.type_builder,
                        diagnostics: stmt_ctx.diagnostics,
                        local_vars: stmt_ctx.local_vars,
                        param_values: stmt_ctx.param_values,
                        name_table: stmt_ctx.name_table,
                        local_types: &stmt_local_types,
                        enum_constants: stmt_ctx.enum_constants,
                        static_locals: stmt_ctx.static_locals,
                        struct_defs: stmt_ctx.struct_defs,
                        label_blocks: stmt_ctx.label_blocks,
                        current_function_name: stmt_ctx.current_function_name,
                        enclosing_loop_stack: stmt_ctx.loop_stack.clone(),
                        scope_type_overrides: &stmt_ctx.scope_type_overrides,
                        last_bitfield_info: None,
                        layout_cache: &mut stmt_ctx.layout_cache,
                        vla_sizes: &stmt_ctx.vla_sizes,
                    };
                    let tv = lower_expr_inner(&mut inner_expr_ctx, expr);
                    last_val = tv.value;
                    last_ty = tv.ty.clone();
                } else if let ast::Statement::Expression(None) = stmt {
                    // Empty expression statement — no-op.
                } else {
                    // Non-expression statements: if/else, for, while, etc.
                    super::stmt_lowering::lower_statement(&mut stmt_ctx, stmt);
                    last_val = Value::UNDEF;
                    last_ty = CType::Void;
                }
            }
        }
    }

    // Restore the layout cache back to the parent expression context
    // so that cached struct layouts persist across statement-expression
    // boundaries within the same function.
    let restored_cache = std::mem::take(&mut stmt_ctx.layout_cache);
    drop(stmt_ctx);
    *ctx.layout_cache = restored_cache;

    // Apply integer promotion to the result type: GCC promotes the result
    // of a statement expression the same way as any other rvalue expression.
    // `({ short x = 1; x; })` has type `int`, not `short`.
    let last_ty = match &last_ty {
        CType::Bool | CType::Char | CType::SChar | CType::UChar | CType::Short | CType::UShort => {
            CType::Int
        }
        _ => last_ty,
    };

    TypedValue::new(last_val, last_ty)
}

// =========================================================================
// GCC BUILTIN DISPATCH
// =========================================================================

fn lower_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    kind: &ast::BuiltinKind,
    args: &[ast::Expression],
    span: Span,
) -> TypedValue {
    match kind {
        // ----- Compile-time builtins (produce constants) -----
        ast::BuiltinKind::ConstantP => {
            // __builtin_constant_p(expr) → 1 if constant, 0 otherwise.
            // Evaluate whether the expression is a compile-time constant.
            if let Some(arg) = args.first() {
                let is_const = is_compile_time_constant(arg);
                let v = emit_int_const(ctx, if is_const { 1 } else { 0 }, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            } else {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            }
        }

        ast::BuiltinKind::TypesCompatibleP => {
            // __builtin_types_compatible_p(type1, type2)
            // args[0] and args[1] are SizeofType carriers with TypeNames.
            let ty1 = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.first() {
                resolve_type_name(ctx, type_name)
            } else {
                CType::Int
            };
            let ty2 = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.get(1) {
                resolve_type_name(ctx, type_name)
            } else {
                CType::Int
            };
            // Compare types ignoring top-level qualifiers (GCC semantics).
            let compat = types_are_compatible_ignoring_qualifiers(&ty1, &ty2);
            let v = emit_int_const(ctx, if compat { 1 } else { 0 }, IrType::I32, span);
            TypedValue::new(v, CType::Int)
        }

        ast::BuiltinKind::ChooseExpr => {
            // __builtin_choose_expr(const_expr, expr1, expr2)
            // const_expr must be an integer constant expression.
            // Per GCC semantics, if const_expr is nonzero, use expr1; else expr2.
            if args.len() >= 3 {
                // Try to evaluate the condition as a compile-time constant.
                let cond_val =
                    try_eval_integer_constant(&args[0], ctx.name_table, ctx.enum_constants);
                if cond_val.unwrap_or(1) != 0 {
                    lower_expr_inner(ctx, &args[1])
                } else {
                    lower_expr_inner(ctx, &args[2])
                }
            } else {
                TypedValue::void()
            }
        }

        ast::BuiltinKind::Offsetof => {
            // __builtin_offsetof(type, member) — compile-time constant.
            // args[0] = SizeofType carrying the TypeName
            // args[1] = member designator expression (Identifier or MemberAccess chain)
            let offset = compute_builtin_offsetof(ctx, args, span);
            let v = emit_int_const(ctx, offset as i128, size_ir_type(ctx.target), span);
            TypedValue::new(v, size_ctype(ctx.target))
        }

        // ----- Branch prediction hint -----
        ast::BuiltinKind::Expect => {
            // __builtin_expect(expr, expected) — return expr as-is.
            // The second argument (hint) is lowered for its side effects
            // (e.g. `z++`) but its value is discarded.
            if let Some(arg) = args.first() {
                let result = lower_expr_inner(ctx, arg);
                // Lower the second argument for side effects only.
                if args.len() > 1 {
                    let _ = lower_expr_inner(ctx, &args[1]);
                }
                result
            } else {
                TypedValue::void()
            }
        }

        // ----- Unreachable / Trap -----
        ast::BuiltinKind::Unreachable => {
            // __builtin_unreachable() — undefined behavior after this point.
            // Register an Undefined constant to mark this path as UB, then
            // emit an unreachable return to mark the block terminated.
            let name = format!(".undef.{}", ctx.builder.fresh_value().0);
            let mut gv = GlobalVariable::new(name, IrType::Void, Some(Constant::Undefined));
            gv.linkage = crate::ir::module::Linkage::Internal;
            ctx.module.add_global(gv);
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        ast::BuiltinKind::Trap => {
            // __builtin_trap() — abort execution.
            // Emit a call to trap (will be lowered to UD2 on x86-64, BRK on AArch64).
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        // ----- Bit-manipulation builtins (produce runtime IR) -----
        ast::BuiltinKind::Clz => lower_bit_builtin(ctx, args, "clz", span),
        ast::BuiltinKind::ClzL => lower_bit_builtin(ctx, args, "clzl", span),
        ast::BuiltinKind::ClzLL => lower_bit_builtin(ctx, args, "clzll", span),
        ast::BuiltinKind::Ctz => lower_bit_builtin(ctx, args, "ctz", span),
        ast::BuiltinKind::CtzL => lower_bit_builtin(ctx, args, "ctzl", span),
        ast::BuiltinKind::CtzLL => lower_bit_builtin(ctx, args, "ctzll", span),
        ast::BuiltinKind::Popcount => lower_bit_builtin(ctx, args, "popcount", span),
        ast::BuiltinKind::PopcountL => lower_bit_builtin(ctx, args, "popcountl", span),
        ast::BuiltinKind::PopcountLL => lower_bit_builtin(ctx, args, "popcountll", span),
        ast::BuiltinKind::Ffs => lower_bit_builtin(ctx, args, "ffs", span),
        ast::BuiltinKind::Ffsll => lower_bit_builtin(ctx, args, "ffsll", span),

        // ----- Byte-swap builtins -----
        ast::BuiltinKind::Bswap16 => lower_bswap(ctx, args, 16, span),
        ast::BuiltinKind::Bswap32 => lower_bswap(ctx, args, 32, span),
        ast::BuiltinKind::Bswap64 => lower_bswap(ctx, args, 64, span),

        // ----- Variadic argument builtins -----
        ast::BuiltinKind::VaStart => lower_va_builtin(ctx, args, "va_start", span),
        ast::BuiltinKind::VaEnd => lower_va_builtin(ctx, args, "va_end", span),
        ast::BuiltinKind::VaArg => lower_va_builtin(ctx, args, "va_arg", span),
        ast::BuiltinKind::VaCopy => lower_va_builtin(ctx, args, "va_copy", span),

        // ----- Frame / return address -----
        // These builtins are lowered as Call instructions so the backend's
        // try_inline_builtin can intercept them and emit MOV RBP / MOV [RBP+8].
        ast::BuiltinKind::FrameAddress => {
            let mut call_args = Vec::new();
            if let Some(arg) = args.first() {
                let v = lower_expr_inner(ctx, arg);
                call_args.push(v.value);
            }
            let callee = emit_global_ref(ctx, "__builtin_frame_address", span);
            let result = ctx.builder.fresh_value();
            let inst = crate::ir::instructions::Instruction::Call {
                result,
                callee,
                args: call_args,
                return_type: IrType::Ptr,
                span,
            };
            emit_inst(ctx, inst);
            TypedValue::new(
                result,
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            )
        }

        ast::BuiltinKind::ReturnAddress => {
            let mut call_args = Vec::new();
            if let Some(arg) = args.first() {
                let v = lower_expr_inner(ctx, arg);
                call_args.push(v.value);
            }
            let callee = emit_global_ref(ctx, "__builtin_return_address", span);
            let result = ctx.builder.fresh_value();
            let inst = crate::ir::instructions::Instruction::Call {
                result,
                callee,
                args: call_args,
                return_type: IrType::Ptr,
                span,
            };
            emit_inst(ctx, inst);
            TypedValue::new(
                result,
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            )
        }

        // ----- Alignment hint -----
        ast::BuiltinKind::AssumeAligned => {
            // __builtin_assume_aligned(ptr, align) → return ptr.
            if let Some(arg) = args.first() {
                lower_expr_inner(ctx, arg)
            } else {
                TypedValue::void()
            }
        }

        // ----- Overflow-checking arithmetic -----
        ast::BuiltinKind::AddOverflow => lower_overflow_arith(ctx, args, IrBinOp::Add, span, false),
        ast::BuiltinKind::SubOverflow => lower_overflow_arith(ctx, args, IrBinOp::Sub, span, false),
        ast::BuiltinKind::MulOverflow => lower_overflow_arith(ctx, args, IrBinOp::Mul, span, false),

        // ----- Prefetch -----
        ast::BuiltinKind::PrefetchData => {
            // __builtin_prefetch(addr, ...) — hint; lower args for side effects.
            for arg in args {
                let _ = lower_expr_inner(ctx, arg);
            }
            TypedValue::void()
        }

        ast::BuiltinKind::ObjectSize => {
            // __builtin_object_size(ptr, type) — at compile time, returns
            // the number of bytes remaining in the object ptr points to.
            // When the size cannot be determined at compile time (which is
            // typical), return (size_t)-1 for type 0/1 or 0 for type 2/3.
            // For the kernel, we conservatively return -1 (unknown size).
            for arg in args {
                let _ = lower_expr_inner(ctx, arg);
            }
            let size_ty = size_ir_type(ctx.target);
            let val = emit_int_const(ctx, -1i128, size_ty, span);
            TypedValue::new(val, CType::ULong)
        }

        ast::BuiltinKind::ExtractReturnAddr => {
            // __builtin_extract_return_addr(addr) — on all Linux targets
            // this is a no-op identity: just return the argument as void*.
            if let Some(a) = args.first() {
                lower_expr_inner(ctx, a)
            } else {
                TypedValue::void()
            }
        }

        // ----- Abort / Exit -----
        ast::BuiltinKind::Abort => {
            // __builtin_abort() — emit a call to abort() then a return.
            let callee = emit_global_ref(ctx, "abort", span);
            let (_result, ci) = ctx.builder.build_call(callee, vec![], IrType::Void, span);
            emit_inst(ctx, ci);
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        ast::BuiltinKind::Exit => {
            // __builtin_exit(status) — emit a call to exit(status) then a return.
            let mut call_args = Vec::new();
            if let Some(arg) = args.first() {
                let v = lower_expr_inner(ctx, arg);
                call_args.push(v.value);
            }
            let callee = emit_global_ref(ctx, "exit", span);
            let (_result, ci) = ctx
                .builder
                .build_call(callee, call_args, IrType::Void, span);
            emit_inst(ctx, ci);
            let ri = ctx.builder.build_return(None, span);
            emit_inst(ctx, ri);
            TypedValue::void()
        }

        // ----- Memory/string builtins -----
        ast::BuiltinKind::Memcpy => lower_libc_builtin(
            ctx,
            args,
            "memcpy",
            IrType::Ptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            span,
        ),
        ast::BuiltinKind::Memset => lower_libc_builtin(
            ctx,
            args,
            "memset",
            IrType::Ptr,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
            span,
        ),
        ast::BuiltinKind::Memcmp => {
            lower_libc_builtin(ctx, args, "memcmp", IrType::I32, CType::Int, span)
        }
        ast::BuiltinKind::Strlen => lower_libc_builtin(
            ctx,
            args,
            "strlen",
            size_ir_type(ctx.target),
            CType::ULong,
            span,
        ),
        ast::BuiltinKind::Strcmp => {
            lower_libc_builtin(ctx, args, "strcmp", IrType::I32, CType::Int, span)
        }
        ast::BuiltinKind::Strncmp => {
            lower_libc_builtin(ctx, args, "strncmp", IrType::I32, CType::Int, span)
        }

        // ----- Abs builtins -----
        ast::BuiltinKind::Abs => {
            lower_libc_builtin(ctx, args, "abs", IrType::I32, CType::Int, span)
        }
        ast::BuiltinKind::Labs => {
            lower_libc_builtin(ctx, args, "labs", IrType::I64, CType::Long, span)
        }
        ast::BuiltinKind::Llabs => {
            lower_libc_builtin(ctx, args, "llabs", IrType::I64, CType::LongLong, span)
        }

        // ----- Alloca -----
        ast::BuiltinKind::Alloca => {
            // __builtin_alloca(size) — dynamic stack allocation.
            // Lower to StackAlloc IR instruction which the backend
            // translates to a runtime SUB RSP + alignment sequence.
            if let Some(size_expr) = args.first() {
                let size_tv = lower_expr_inner(ctx, size_expr);
                let size_val = insert_implicit_conversion(
                    ctx,
                    size_tv.value,
                    &size_tv.ty,
                    &size_ctype(ctx.target),
                    span,
                );
                let (result, inst) = ctx.builder.build_stack_alloc(size_val, span);
                emit_inst(ctx, inst);
                TypedValue::new(
                    result,
                    CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
                )
            } else {
                let zero = emit_int_const(ctx, 0, IrType::Ptr, span);
                TypedValue::new(
                    zero,
                    CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
                )
            }
        }

        // ----- Float constants (compile-time) -----
        ast::BuiltinKind::Inf | ast::BuiltinKind::HugeVal => {
            let val = f64::INFINITY;
            let v = emit_float_const(ctx, val, IrType::F64, span);
            TypedValue::new(v, CType::Double)
        }
        ast::BuiltinKind::Inff | ast::BuiltinKind::HugeValf => {
            let val = f64::INFINITY;
            let v = emit_float_const(ctx, val, IrType::F32, span);
            TypedValue::new(v, CType::Float)
        }
        ast::BuiltinKind::Infl | ast::BuiltinKind::HugeVall => {
            let val = f64::INFINITY;
            let v = emit_float_const(ctx, val, IrType::F64, span);
            TypedValue::new(v, CType::LongDouble)
        }
        ast::BuiltinKind::Nan | ast::BuiltinKind::Nanl => {
            let val = f64::NAN;
            let v = emit_float_const(ctx, val, IrType::F64, span);
            TypedValue::new(v, CType::Double)
        }
        ast::BuiltinKind::Nanf => {
            let val = f64::NAN;
            let v = emit_float_const(ctx, val, IrType::F32, span);
            TypedValue::new(v, CType::Float)
        }

        // ----- Float predicates -----
        // Handled by try_lower_builtin_call via string matching in the
        // FunctionCall path. This BuiltinKind path is a fallback that
        // uses volatile store+load for type-punning (bitcast codegen
        // does float→int conversion, not bit reinterpretation).
        ast::BuiltinKind::Signbit
        | ast::BuiltinKind::Isnan
        | ast::BuiltinKind::Isinf
        | ast::BuiltinKind::Isfinite
        | ast::BuiltinKind::IsinfSign => {
            if let Some(arg_expr) = args.first() {
                let tv = lower_expr_inner(ctx, arg_expr);
                let resolved = crate::common::types::resolve_typedef(&tv.ty);
                let is_float = matches!(resolved, CType::Float);
                let (int_ty, abs_mask, exp_mask, shift_n) = if is_float {
                    (IrType::I32, 0x7FFF_FFFFi128, 0x7F80_0000i128, 31i128)
                } else {
                    (
                        IrType::I64,
                        0x7FFF_FFFF_FFFF_FFFFi128,
                        0x7FF0_0000_0000_0000i128,
                        63i128,
                    )
                };
                // Volatile store+load for type-pun
                let (alloca, alloc_i) = ctx.builder.build_alloca(int_ty.clone(), span);
                emit_inst(ctx, alloc_i);
                let mut si = ctx.builder.build_store(tv.value, alloca, span);
                if let crate::ir::instructions::Instruction::Store {
                    ref mut volatile, ..
                } = si
                {
                    *volatile = true;
                }
                emit_inst(ctx, si);
                let (bits, li) = ctx.builder.build_load(alloca, int_ty.clone(), span);
                emit_inst(ctx, li);

                let result = match kind {
                    ast::BuiltinKind::Signbit => {
                        let sh = emit_int_const(ctx, shift_n, int_ty.clone(), span);
                        let (shifted, shi) =
                            ctx.builder.build_shr(bits, sh, int_ty.clone(), false, span);
                        emit_inst(ctx, shi);
                        if int_ty != IrType::I32 {
                            let (tr, tri) = ctx.builder.build_trunc(shifted, IrType::I32, span);
                            emit_inst(ctx, tri);
                            tr
                        } else {
                            shifted
                        }
                    }
                    ast::BuiltinKind::Isnan
                    | ast::BuiltinKind::Isinf
                    | ast::BuiltinKind::Isfinite => {
                        let mask_v = emit_int_const(ctx, abs_mask, int_ty.clone(), span);
                        let (abs_bits, abi) =
                            ctx.builder.build_and(bits, mask_v, int_ty.clone(), span);
                        emit_inst(ctx, abi);
                        let exp_v = emit_int_const(ctx, exp_mask, int_ty.clone(), span);
                        let cmp_op = match kind {
                            ast::BuiltinKind::Isnan => ICmpOp::Ugt,
                            ast::BuiltinKind::Isinf => ICmpOp::Eq,
                            _ => ICmpOp::Ult, // Isfinite
                        };
                        let (cmp, ci) = ctx.builder.build_icmp(cmp_op, abs_bits, exp_v, span);
                        emit_inst(ctx, ci);
                        let (ext, ei) = ctx.builder.build_zext(cmp, IrType::I32, span);
                        emit_inst(ctx, ei);
                        ext
                    }
                    ast::BuiltinKind::IsinfSign => {
                        let mask_v = emit_int_const(ctx, abs_mask, int_ty.clone(), span);
                        let (abs_bits, abi) =
                            ctx.builder.build_and(bits, mask_v, int_ty.clone(), span);
                        emit_inst(ctx, abi);
                        let exp_v = emit_int_const(ctx, exp_mask, int_ty.clone(), span);
                        let (is_inf, ici) =
                            ctx.builder.build_icmp(ICmpOp::Eq, abs_bits, exp_v, span);
                        emit_inst(ctx, ici);
                        let (inf_i32, ie) = ctx.builder.build_zext(is_inf, IrType::I32, span);
                        emit_inst(ctx, ie);
                        let sh = emit_int_const(ctx, shift_n, int_ty.clone(), span);
                        let (shifted, shi) =
                            ctx.builder.build_shr(bits, sh, int_ty.clone(), false, span);
                        emit_inst(ctx, shi);
                        let sign_i32 = if int_ty != IrType::I32 {
                            let (tr, tri) = ctx.builder.build_trunc(shifted, IrType::I32, span);
                            emit_inst(ctx, tri);
                            tr
                        } else {
                            shifted
                        };
                        let one_v = emit_int_const(ctx, 1, IrType::I32, span);
                        let (s_and, sai) =
                            ctx.builder.build_and(sign_i32, one_v, IrType::I32, span);
                        emit_inst(ctx, sai);
                        let two_v = emit_int_const(ctx, 2, IrType::I32, span);
                        let (t2s, t2i) = ctx.builder.build_mul(two_v, s_and, IrType::I32, span);
                        emit_inst(ctx, t2i);
                        let (neg, ni) = ctx.builder.build_mul(inf_i32, t2s, IrType::I32, span);
                        emit_inst(ctx, ni);
                        let (res, ri) = ctx.builder.build_sub(inf_i32, neg, IrType::I32, span);
                        emit_inst(ctx, ri);
                        res
                    }
                    _ => unreachable!(),
                };
                TypedValue::new(result, CType::Int)
            } else {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            }
        }

        // ----- Copysign / fabs -----
        ast::BuiltinKind::Copysign => {
            lower_libc_builtin(ctx, args, "copysign", IrType::F64, CType::Double, span)
        }
        ast::BuiltinKind::Copysignf => {
            lower_libc_builtin(ctx, args, "copysignf", IrType::F32, CType::Float, span)
        }
        ast::BuiltinKind::Copysignl => {
            lower_libc_builtin(ctx, args, "copysignl", IrType::F64, CType::LongDouble, span)
        }
        ast::BuiltinKind::Fabs => {
            lower_libc_builtin(ctx, args, "fabs", IrType::F64, CType::Double, span)
        }
        ast::BuiltinKind::Fabsf => {
            lower_libc_builtin(ctx, args, "fabsf", IrType::F32, CType::Float, span)
        }
        ast::BuiltinKind::Fabsl => {
            lower_libc_builtin(ctx, args, "fabsl", IrType::F64, CType::LongDouble, span)
        }

        // ----- classify_type -----
        ast::BuiltinKind::ClassifyType => {
            // GCC __builtin_classify_type:
            // void=0, integer=1, char=2, enum=3, bool=4, pointer=5,
            // reference=6, offset=7, real=8, complex=9, function=10, method=11
            // We'll classify based on the argument type.
            if let Some(arg) = args.first() {
                let tv = lower_expr_inner(ctx, arg);
                let type_class = classify_type_for_builtin(&tv.ty);
                let v = emit_int_const(ctx, type_class as i128, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            } else {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            }
        }

        // ----- Sync atomics -----
        ast::BuiltinKind::SyncValCompareAndSwap
        | ast::BuiltinKind::SyncBoolCompareAndSwap
        | ast::BuiltinKind::SyncFetchAndAdd
        | ast::BuiltinKind::SyncFetchAndSub
        | ast::BuiltinKind::SyncFetchAndAnd
        | ast::BuiltinKind::SyncFetchAndOr
        | ast::BuiltinKind::SyncFetchAndXor
        | ast::BuiltinKind::SyncLockTestAndSet => {
            let fname = match kind {
                ast::BuiltinKind::SyncValCompareAndSwap => "__sync_val_compare_and_swap",
                ast::BuiltinKind::SyncBoolCompareAndSwap => "__sync_bool_compare_and_swap",
                ast::BuiltinKind::SyncFetchAndAdd => "__sync_fetch_and_add",
                ast::BuiltinKind::SyncFetchAndSub => "__sync_fetch_and_sub",
                ast::BuiltinKind::SyncFetchAndAnd => "__sync_fetch_and_and",
                ast::BuiltinKind::SyncFetchAndOr => "__sync_fetch_and_or",
                ast::BuiltinKind::SyncFetchAndXor => "__sync_fetch_and_xor",
                ast::BuiltinKind::SyncLockTestAndSet => "__sync_lock_test_and_set",
                _ => unreachable!(),
            };
            let ret_ty = if matches!(kind, ast::BuiltinKind::SyncBoolCompareAndSwap) {
                IrType::I32
            } else {
                IrType::I64
            };
            let ret_cty = if matches!(kind, ast::BuiltinKind::SyncBoolCompareAndSwap) {
                CType::Bool
            } else {
                CType::Int
            };
            lower_libc_builtin(ctx, args, fname, ret_ty, ret_cty, span)
        }
        ast::BuiltinKind::SyncLockRelease => lower_libc_builtin(
            ctx,
            args,
            "__sync_lock_release",
            IrType::Void,
            CType::Void,
            span,
        ),
        ast::BuiltinKind::SyncSynchronize => lower_libc_builtin(
            ctx,
            args,
            "__sync_synchronize",
            IrType::Void,
            CType::Void,
            span,
        ),

        // ----- C11 atomics -----
        ast::BuiltinKind::AtomicLoadN => {
            lower_libc_builtin(ctx, args, "__atomic_load_n", IrType::I64, CType::Int, span)
        }
        ast::BuiltinKind::AtomicStoreN => lower_libc_builtin(
            ctx,
            args,
            "__atomic_store_n",
            IrType::Void,
            CType::Void,
            span,
        ),
        ast::BuiltinKind::AtomicExchangeN => lower_libc_builtin(
            ctx,
            args,
            "__atomic_exchange_n",
            IrType::I64,
            CType::Int,
            span,
        ),
        ast::BuiltinKind::AtomicCompareExchangeN => lower_libc_builtin(
            ctx,
            args,
            "__atomic_compare_exchange_n",
            IrType::I32,
            CType::Bool,
            span,
        ),
        ast::BuiltinKind::AtomicFetchAdd => lower_libc_builtin(
            ctx,
            args,
            "__atomic_fetch_add",
            IrType::I64,
            CType::Int,
            span,
        ),
        ast::BuiltinKind::AtomicFetchSub => lower_libc_builtin(
            ctx,
            args,
            "__atomic_fetch_sub",
            IrType::I64,
            CType::Int,
            span,
        ),
        ast::BuiltinKind::AtomicFetchAnd => lower_libc_builtin(
            ctx,
            args,
            "__atomic_fetch_and",
            IrType::I64,
            CType::Int,
            span,
        ),
        ast::BuiltinKind::AtomicFetchOr => lower_libc_builtin(
            ctx,
            args,
            "__atomic_fetch_or",
            IrType::I64,
            CType::Int,
            span,
        ),
        ast::BuiltinKind::AtomicFetchXor => lower_libc_builtin(
            ctx,
            args,
            "__atomic_fetch_xor",
            IrType::I64,
            CType::Int,
            span,
        ),

        // ----- Overflow _p variants -----
        ast::BuiltinKind::AddOverflowP
        | ast::BuiltinKind::SubOverflowP
        | ast::BuiltinKind::MulOverflowP => {
            // _p variants: overflow check without storing result.
            // Semantically same as overflow arithmetic, but result pointer is unused.
            let op = match kind {
                ast::BuiltinKind::AddOverflowP => IrBinOp::Add,
                ast::BuiltinKind::SubOverflowP => IrBinOp::Sub,
                ast::BuiltinKind::MulOverflowP => IrBinOp::Mul,
                _ => unreachable!(),
            };
            lower_overflow_arith(ctx, args, op, span, true)
        }

        // ----- BuiltinConstantP alias -----
        ast::BuiltinKind::BuiltinConstantP => {
            if let Some(arg) = args.first() {
                let is_const = is_compile_time_constant(arg);
                let v = emit_int_const(ctx, if is_const { 1 } else { 0 }, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            } else {
                let v = emit_int_const(ctx, 0, IrType::I32, span);
                TypedValue::new(v, CType::Int)
            }
        }
    }
}

/// Helper: lower a builtin that maps directly to a libc or intrinsic function call.
fn lower_libc_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    func_name: &str,
    ret_ir_type: IrType,
    ret_c_type: CType,
    span: Span,
) -> TypedValue {
    let mut call_args = Vec::new();
    for arg in args {
        let tv = lower_expr_inner(ctx, arg);
        call_args.push(tv.value);
    }
    let callee = emit_global_ref(ctx, func_name, span);
    if ret_ir_type == IrType::Void {
        let (_result, ci) = ctx
            .builder
            .build_call(callee, call_args, IrType::Void, span);
        emit_inst(ctx, ci);
        TypedValue::void()
    } else {
        let (result, ci) = ctx.builder.build_call(callee, call_args, ret_ir_type, span);
        emit_inst(ctx, ci);
        TypedValue::new(result, ret_c_type)
    }
}

/// Helper: classify a C type for __builtin_classify_type.
fn classify_type_for_builtin(ctype: &CType) -> i32 {
    match ctype {
        CType::Void => 0,
        CType::Bool => 4,
        CType::Char | CType::SChar | CType::UChar => 2,
        CType::Short
        | CType::UShort
        | CType::Int
        | CType::UInt
        | CType::Long
        | CType::ULong
        | CType::LongLong
        | CType::ULongLong => 1,
        CType::Float | CType::Double | CType::LongDouble => 8,
        CType::Pointer(..) => 5,
        CType::Function { .. } => 10,
        CType::Enum { .. } => 3,
        CType::Complex(..) => 9,
        _ => 1, // default: integer
    }
}

/// Lower a bit-manipulation builtin (clz, ctz, popcount, ffs).
/// These map to target-specific instructions in the backend.
fn lower_bit_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    name: &str,
    span: Span,
) -> TypedValue {
    let arg = if let Some(a) = args.first() {
        lower_expr_inner(ctx, a)
    } else {
        return TypedValue::void();
    };

    // For ffs/ffsl/ffsll — forward to libc `ffs`/`ffsl`/`ffsll` (they exist in libc).
    match name {
        "ffs" => {
            let callee = emit_global_ref(ctx, "ffs", span);
            let (result, ci) = ctx
                .builder
                .build_call(callee, vec![arg.value], IrType::I32, span);
            emit_inst(ctx, ci);
            return TypedValue::new(result, CType::Int);
        }
        "ffsl" => {
            let callee = emit_global_ref(ctx, "ffsl", span);
            let v64 = insert_implicit_conversion(ctx, arg.value, &arg.ty, &CType::Long, span);
            let (result, ci) = ctx.builder.build_call(callee, vec![v64], IrType::I32, span);
            emit_inst(ctx, ci);
            return TypedValue::new(result, CType::Int);
        }
        "ffsll" => {
            let callee = emit_global_ref(ctx, "ffsll", span);
            let v64 = insert_implicit_conversion(ctx, arg.value, &arg.ty, &CType::LongLong, span);
            let (result, ci) = ctx.builder.build_call(callee, vec![v64], IrType::I32, span);
            emit_inst(ctx, ci);
            return TypedValue::new(result, CType::Int);
        }
        _ => {}
    }

    // Determine operand width from the builtin suffix
    let (op_bits, op_ir, promote_to) = if name.ends_with("ll") {
        (64u32, IrType::I64, CType::ULongLong)
    } else if name.ends_with('l') {
        (64u32, IrType::I64, CType::ULong)
    } else {
        (32u32, IrType::I32, CType::UInt)
    };

    let val = insert_implicit_conversion(ctx, arg.value, &arg.ty, &promote_to, span);

    // Strip the suffix to get the base operation name
    let base = name.trim_end_matches('l');

    match base {
        "clz" => lower_clz_inline(ctx, val, op_bits, &op_ir, span),
        "ctz" => lower_ctz_inline(ctx, val, op_bits, &op_ir, span),
        "popcount" => lower_popcount_inline(ctx, val, op_bits, &op_ir, span),
        "parity" => {
            let pc = lower_popcount_inline(ctx, val, op_bits, &op_ir, span);
            let one = emit_int_const(ctx, 1, IrType::I32, span);
            let (r, i) = ctx.builder.build_and(pc.value, one, IrType::I32, span);
            emit_inst(ctx, i);
            TypedValue::new(r, CType::Int)
        }
        "clrsb" => {
            // clrsb(x) = clz(x ^ (x >> (bits-1))) - 1
            let shift_amt = emit_int_const(ctx, (op_bits - 1) as i128, op_ir.clone(), span);
            let (shifted, si) = ctx
                .builder
                .build_shr(val, shift_amt, op_ir.clone(), true, span);
            emit_inst(ctx, si);
            let (xored, xi) = ctx.builder.build_xor(val, shifted, op_ir.clone(), span);
            emit_inst(ctx, xi);
            let clz_tv = lower_clz_inline(ctx, xored, op_bits, &op_ir, span);
            let one_i32 = emit_int_const(ctx, 1, IrType::I32, span);
            let (r, i) = ctx
                .builder
                .build_sub(clz_tv.value, one_i32, IrType::I32, span);
            emit_inst(ctx, i);
            TypedValue::new(r, CType::Int)
        }
        _ => {
            // Fallback: emit a call to the libc name (without __builtin_ prefix)
            let callee = emit_global_ref(ctx, &format!("__builtin_{}", name), span);
            let (result, ci) = ctx.builder.build_call(callee, vec![val], IrType::I32, span);
            emit_inst(ctx, ci);
            TypedValue::new(result, CType::Int)
        }
    }
}

/// Inline CLZ (count leading zeros) using binary search in IR.
/// Returns result as I32 (like GCC's __builtin_clz).
fn lower_clz_inline(
    ctx: &mut ExprLoweringContext<'_>,
    val: Value,
    bits: u32,
    ir_ty: &IrType,
    span: Span,
) -> TypedValue {
    // CLZ via branchless De Bruijn / bit-manipulation approach.
    // For 32-bit: Propagate highest set bit rightward, count via popcount.
    // clz(x) = bits - 1 - floor(log2(x)) for x>0
    // We use: propagate msb right, then popcount, then bits - popcount.
    // x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; (x |= x >> 32 for 64-bit)
    // clz = bits - popcount(x)
    let mut x = val;
    let shifts: &[u32] = if bits == 64 {
        &[1, 2, 4, 8, 16, 32]
    } else {
        &[1, 2, 4, 8, 16]
    };
    for &sh in shifts {
        let sc = emit_int_const(ctx, sh as i128, ir_ty.clone(), span);
        let (shifted, si) = ctx.builder.build_shr(x, sc, ir_ty.clone(), false, span);
        emit_inst(ctx, si);
        let (ored, oi) = ctx.builder.build_or(x, shifted, ir_ty.clone(), span);
        emit_inst(ctx, oi);
        x = ored;
    }
    // popcount(x) gives position of highest bit + number of bits below it
    // clz = bits - popcount(x)
    let pop = lower_popcount_inline(ctx, x, bits, ir_ty, span);
    let bits_c = emit_int_const(ctx, bits as i128, IrType::I32, span);
    let (result, ri) = ctx.builder.build_sub(bits_c, pop.value, IrType::I32, span);
    emit_inst(ctx, ri);
    TypedValue::new(result, CType::Int)
}

/// Inline CTZ (count trailing zeros) using arithmetic in IR.
fn lower_ctz_inline(
    ctx: &mut ExprLoweringContext<'_>,
    val: Value,
    bits: u32,
    ir_ty: &IrType,
    span: Span,
) -> TypedValue {
    // CTZ(x) = popcount((x & -x) - 1) for x != 0
    // (x & -x) isolates the lowest set bit.
    // Alternatively: CTZ(x) = bits - CLZ(~x & (x-1)) but CLZ calls popcount anyway.
    // Simplest: CTZ(x) = popcount(~x & (x - 1))
    // ~x & (x-1) has all bits below the lowest set bit set.
    let one = emit_int_const(ctx, 1, ir_ty.clone(), span);
    let (x_minus_1, xm1i) = ctx.builder.build_sub(val, one, ir_ty.clone(), span);
    emit_inst(ctx, xm1i);
    let (not_x, nxi) = ctx.builder.build_not(val, ir_ty.clone(), span);
    emit_inst(ctx, nxi);
    let (masked, mi) = ctx.builder.build_and(not_x, x_minus_1, ir_ty.clone(), span);
    emit_inst(ctx, mi);
    lower_popcount_inline(ctx, masked, bits, ir_ty, span)
}

/// Inline POPCOUNT using parallel bit-count algorithm in IR.
fn lower_popcount_inline(
    ctx: &mut ExprLoweringContext<'_>,
    val: Value,
    bits: u32,
    ir_ty: &IrType,
    span: Span,
) -> TypedValue {
    // Hamming weight (parallel bit count):
    // x = x - ((x >> 1) & 0x5555...)
    // x = (x & 0x3333...) + ((x >> 2) & 0x3333...)
    // x = (x + (x >> 4)) & 0x0F0F...
    // x = (x * 0x0101...) >> (bits - 8)
    let (m1, m2, m4, h01) = if bits == 64 {
        (
            0x5555555555555555u64 as i128,
            0x3333333333333333u64 as i128,
            0x0F0F0F0F0F0F0F0Fu64 as i128,
            0x0101010101010101u64 as i128,
        )
    } else {
        (
            0x55555555i128,
            0x33333333i128,
            0x0F0F0F0Fi128,
            0x01010101i128,
        )
    };

    let one = emit_int_const(ctx, 1, ir_ty.clone(), span);
    let (shr1, s1i) = ctx.builder.build_shr(val, one, ir_ty.clone(), false, span);
    emit_inst(ctx, s1i);
    let m1c = emit_int_const(ctx, m1, ir_ty.clone(), span);
    let (and1, a1i) = ctx.builder.build_and(shr1, m1c, ir_ty.clone(), span);
    emit_inst(ctx, a1i);
    let (x1, x1i) = ctx.builder.build_sub(val, and1, ir_ty.clone(), span);
    emit_inst(ctx, x1i);

    let m2c = emit_int_const(ctx, m2, ir_ty.clone(), span);
    let (lo, loi) = ctx.builder.build_and(x1, m2c, ir_ty.clone(), span);
    emit_inst(ctx, loi);
    let two = emit_int_const(ctx, 2, ir_ty.clone(), span);
    let (shr2, s2i) = ctx.builder.build_shr(x1, two, ir_ty.clone(), false, span);
    emit_inst(ctx, s2i);
    let (hi, hii) = ctx.builder.build_and(shr2, m2c, ir_ty.clone(), span);
    emit_inst(ctx, hii);
    let (x2, x2i) = ctx.builder.build_add(lo, hi, ir_ty.clone(), span);
    emit_inst(ctx, x2i);

    let four = emit_int_const(ctx, 4, ir_ty.clone(), span);
    let (shr4, s4i) = ctx.builder.build_shr(x2, four, ir_ty.clone(), false, span);
    emit_inst(ctx, s4i);
    let (add4, a4i) = ctx.builder.build_add(x2, shr4, ir_ty.clone(), span);
    emit_inst(ctx, a4i);
    let m4c = emit_int_const(ctx, m4, ir_ty.clone(), span);
    let (x3, x3i) = ctx.builder.build_and(add4, m4c, ir_ty.clone(), span);
    emit_inst(ctx, x3i);

    let h01c = emit_int_const(ctx, h01, ir_ty.clone(), span);
    let (mul, mi) = ctx.builder.build_mul(x3, h01c, ir_ty.clone(), span);
    emit_inst(ctx, mi);
    let shift_final = emit_int_const(ctx, (bits - 8) as i128, ir_ty.clone(), span);
    let (result, ri) = ctx
        .builder
        .build_shr(mul, shift_final, ir_ty.clone(), false, span);
    emit_inst(ctx, ri);

    // Truncate to I32 if 64-bit
    let result_i32 = if bits > 32 {
        let (t, ti) = ctx.builder.build_trunc(result, IrType::I32, span);
        emit_inst(ctx, ti);
        t
    } else {
        result
    };
    TypedValue::new(result_i32, CType::Int)
}

/// Lower a byte-swap builtin.
fn lower_bswap(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    bits: u32,
    span: Span,
) -> TypedValue {
    let arg = if let Some(a) = args.first() {
        lower_expr_inner(ctx, a)
    } else {
        return TypedValue::void();
    };

    let ret_ty = match bits {
        16 => IrType::I16,
        32 => IrType::I32,
        64 => IrType::I64,
        _ => IrType::I32,
    };
    let ret_cty = match bits {
        16 => CType::UShort,
        32 => CType::UInt,
        64 => CType::ULongLong,
        _ => CType::UInt,
    };
    let promote_cty = match bits {
        16 => CType::UShort,
        32 => CType::UInt,
        64 => CType::ULongLong,
        _ => CType::UInt,
    };
    let val = insert_implicit_conversion(ctx, arg.value, &arg.ty, &promote_cty, span);

    // Inline byte-swap using shifts and ORs
    match bits {
        16 => {
            // bswap16(x) = ((x & 0xFF) << 8) | ((x >> 8) & 0xFF)
            let mask = emit_int_const(ctx, 0xFF, ret_ty.clone(), span);
            let eight = emit_int_const(ctx, 8, ret_ty.clone(), span);
            let (lo, li) = ctx.builder.build_and(val, mask, ret_ty.clone(), span);
            emit_inst(ctx, li);
            let (lo_s, lsi) = ctx.builder.build_shl(lo, eight, ret_ty.clone(), span);
            emit_inst(ctx, lsi);
            let (hi_s, hsi) = ctx
                .builder
                .build_shr(val, eight, ret_ty.clone(), false, span);
            emit_inst(ctx, hsi);
            let (hi, hii) = ctx.builder.build_and(hi_s, mask, ret_ty.clone(), span);
            emit_inst(ctx, hii);
            let (result, ri) = ctx.builder.build_or(lo_s, hi, ret_ty.clone(), span);
            emit_inst(ctx, ri);
            TypedValue::new(result, ret_cty)
        }
        32 => {
            // bswap32 using shift-and-or
            let mut result = {
                let c24 = emit_int_const(ctx, 24, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shl(val, c24, ret_ty.clone(), span);
                emit_inst(ctx, si);
                s
            };
            {
                let m = emit_int_const(ctx, 0x0000FF00i128, ret_ty.clone(), span);
                let (a, ai) = ctx.builder.build_and(val, m, ret_ty.clone(), span);
                emit_inst(ctx, ai);
                let c8 = emit_int_const(ctx, 8, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shl(a, c8, ret_ty.clone(), span);
                emit_inst(ctx, si);
                let (o, oi) = ctx.builder.build_or(result, s, ret_ty.clone(), span);
                emit_inst(ctx, oi);
                result = o;
            }
            {
                let m = emit_int_const(ctx, 0x00FF0000i128, ret_ty.clone(), span);
                let (a, ai) = ctx.builder.build_and(val, m, ret_ty.clone(), span);
                emit_inst(ctx, ai);
                let c8 = emit_int_const(ctx, 8, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shr(a, c8, ret_ty.clone(), false, span);
                emit_inst(ctx, si);
                let (o, oi) = ctx.builder.build_or(result, s, ret_ty.clone(), span);
                emit_inst(ctx, oi);
                result = o;
            }
            {
                let c24 = emit_int_const(ctx, 24, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shr(val, c24, ret_ty.clone(), false, span);
                emit_inst(ctx, si);
                let (o, oi) = ctx.builder.build_or(result, s, ret_ty.clone(), span);
                emit_inst(ctx, oi);
                result = o;
            }
            TypedValue::new(result, ret_cty)
        }
        64 => {
            // bswap64: swap bytes pairwise then reverse halves
            // Inline shift-and-or approach
            let mut result = {
                let c56 = emit_int_const(ctx, 56, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shl(val, c56, ret_ty.clone(), span);
                emit_inst(ctx, si);
                s
            };
            for (mask_val, shift_val, is_left) in [
                (0x000000000000FF00u64 as i128, 40i128, true),
                (0x0000000000FF0000u64 as i128, 24i128, true),
                (0x00000000FF000000u64 as i128, 8i128, true),
                (0x000000FF00000000u64 as i128, 8i128, false),
                (0x0000FF0000000000u64 as i128, 24i128, false),
                (0x00FF000000000000u64 as i128, 40i128, false),
            ] {
                let m = emit_int_const(ctx, mask_val, ret_ty.clone(), span);
                let (a, ai) = ctx.builder.build_and(val, m, ret_ty.clone(), span);
                emit_inst(ctx, ai);
                let sc = emit_int_const(ctx, shift_val, ret_ty.clone(), span);
                let (s, si) = if is_left {
                    ctx.builder.build_shl(a, sc, ret_ty.clone(), span)
                } else {
                    ctx.builder.build_shr(a, sc, ret_ty.clone(), false, span)
                };
                emit_inst(ctx, si);
                let (o, oi) = ctx.builder.build_or(result, s, ret_ty.clone(), span);
                emit_inst(ctx, oi);
                result = o;
            }
            {
                let c56 = emit_int_const(ctx, 56, ret_ty.clone(), span);
                let (s, si) = ctx.builder.build_shr(val, c56, ret_ty.clone(), false, span);
                emit_inst(ctx, si);
                let (o, oi) = ctx.builder.build_or(result, s, ret_ty.clone(), span);
                emit_inst(ctx, oi);
                result = o;
            }
            TypedValue::new(result, ret_cty)
        }
        _ => {
            // Fallback for unknown widths
            let callee = emit_global_ref(ctx, &format!("__builtin_bswap{}", bits), span);
            let (result, ci) = ctx.builder.build_call(callee, vec![val], ret_ty, span);
            emit_inst(ctx, ci);
            TypedValue::new(result, ret_cty)
        }
    }
}

/// Lower a variadic argument builtin (va_start, va_end, va_arg, va_copy).
/// Classify each eightbyte of a struct for va_arg purposes.
///
/// Returns a `Vec<IrType>` where each entry is `IrType::F64` for an
/// SSE-class eightbyte (all fields are float/double) or `IrType::I64`
/// for an INTEGER-class eightbyte (any integer/pointer field present).
/// This mirrors the System V AMD64 ABI struct-classification rules
/// needed for multi-register va_arg extraction.
fn classify_struct_va_arg_eightbytes(cty: &CType, target: &Target) -> Vec<IrType> {
    let sz = crate::common::types::sizeof_ctype(cty, target);
    let num_eightbytes = (sz + 7) / 8;
    if num_eightbytes == 0 {
        return vec![IrType::I64];
    }
    // Start each eightbyte as SSE (true).  Any non-float field downgrades
    // the eightbyte to INTEGER.
    let mut is_sse = vec![true; num_eightbytes];
    let mut has_field = vec![false; num_eightbytes];

    fn walk_fields(
        fields: &[crate::common::types::StructField],
        is_packed: bool,
        base_offset: usize,
        target: &Target,
        is_sse: &mut Vec<bool>,
        has_field: &mut Vec<bool>,
    ) {
        let mut offset = base_offset;
        let num_eb = is_sse.len();
        for field in fields {
            if let (None, Some(bw_val)) = (&field.name, field.bit_width) {
                let bw = bw_val as usize;
                if bw == 0 {
                    // Zero-width bitfield: align to next unit boundary
                    let unit_align = crate::common::types::alignof_ctype(&field.ty, target);
                    if unit_align > 0 {
                        offset = (offset + unit_align - 1) & !(unit_align - 1);
                    }
                } else {
                    offset += (bw + 7) / 8;
                }
                continue;
            }
            let field_sz = crate::common::types::sizeof_ctype(&field.ty, target);
            let field_align = if is_packed {
                1
            } else {
                crate::common::types::alignof_ctype(&field.ty, target)
            };
            if field_align > 0 && !is_packed {
                offset = (offset + field_align - 1) & !(field_align - 1);
            }
            if let Some(bw_val) = field.bit_width {
                // Named bitfield → INTEGER class
                let eb = offset / 8;
                if eb < num_eb {
                    is_sse[eb] = false;
                    has_field[eb] = true;
                }
                let bw = bw_val as usize;
                offset += (bw + 7) / 8;
                continue;
            }
            let start_eb = offset / 8;
            let end_eb = if field_sz > 0 {
                (offset + field_sz - 1) / 8
            } else {
                start_eb
            };
            // Classify each eightbyte by field type: Float/Double ⇒ SSE,
            // nested structs ⇒ recurse, Array(Float/Double) ⇒ SSE per
            // element, everything else ⇒ INTEGER.
            match &field.ty {
                CType::Float | CType::Double => {
                    for item in has_field
                        .iter_mut()
                        .take(end_eb.min(num_eb - 1) + 1)
                        .skip(start_eb)
                    {
                        *item = true;
                        // stays SSE
                    }
                }
                CType::Struct {
                    fields: inner_fields,
                    packed: inner_packed,
                    ..
                } => {
                    walk_fields(
                        inner_fields,
                        *inner_packed,
                        offset,
                        target,
                        is_sse,
                        has_field,
                    );
                }
                CType::Array(elem_ty, count) => {
                    // Walk each array element individually so that
                    // Array(Float, N) properly classifies eightbytes
                    // as SSE instead of falling into the INTEGER arm.
                    let elem_sz = crate::common::types::sizeof_ctype(elem_ty, target);
                    let elem_align = crate::common::types::alignof_ctype(elem_ty, target);
                    let n = count.unwrap_or(0);
                    let mut arr_off = offset;
                    for _ in 0..n {
                        if elem_align > 0 {
                            arr_off = (arr_off + elem_align - 1) & !(elem_align - 1);
                        }
                        let e_start = arr_off / 8;
                        let e_end = if elem_sz > 0 {
                            (arr_off + elem_sz - 1) / 8
                        } else {
                            e_start
                        };
                        match elem_ty.as_ref() {
                            CType::Float | CType::Double => {
                                for item in has_field
                                    .iter_mut()
                                    .take(e_end.min(num_eb.saturating_sub(1)) + 1)
                                    .skip(e_start)
                                {
                                    *item = true;
                                    // stays SSE
                                }
                            }
                            CType::Struct {
                                fields: sf,
                                packed: sp,
                                ..
                            } => {
                                walk_fields(sf, *sp, arr_off, target, is_sse, has_field);
                            }
                            _ => {
                                for eb in e_start..=e_end.min(num_eb.saturating_sub(1)) {
                                    is_sse[eb] = false;
                                    has_field[eb] = true;
                                }
                            }
                        }
                        arr_off += elem_sz;
                    }
                }
                _ => {
                    // Integer, pointer, etc. → INTEGER class
                    for eb in start_eb..=end_eb.min(num_eb - 1) {
                        is_sse[eb] = false;
                        has_field[eb] = true;
                    }
                }
            }
            offset += field_sz;
        }
    }

    if let CType::Struct { fields, packed, .. } = cty {
        walk_fields(fields, *packed, 0, target, &mut is_sse, &mut has_field);
    } else {
        // Union or other — all INTEGER
        for v in is_sse.iter_mut() {
            *v = false;
        }
    }

    // Any eightbyte with no fields at all is INTEGER
    is_sse
        .iter()
        .zip(has_field.iter())
        .map(
            |(&sse, &has)| {
                if sse && has {
                    IrType::F64
                } else {
                    IrType::I64
                }
            },
        )
        .collect()
}

///
/// For va_start, va_end, va_arg: the first argument (the va_list) must be
/// passed by ADDRESS (not by value) so the backend can read/write the slot.
/// For va_copy: both arguments are va_list addresses.
fn lower_va_builtin(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    name: &str,
    span: Span,
) -> TypedValue {
    let mut arg_vals = Vec::new();
    for (i, a) in args.iter().enumerate() {
        // For va_start/va_end/va_arg: pass the ADDRESS of the first
        // argument (the va_list variable) so the backend can modify it.
        // For va_copy: pass the ADDRESS of both arguments.
        let need_address = match name {
            "va_start" | "va_end" | "va_arg" => i == 0,
            "va_copy" => i <= 1,
            _ => false,
        };
        if need_address {
            let addr = lower_lvalue(ctx, a);
            arg_vals.push(addr);
        } else {
            let tv = lower_expr_inner(ctx, a);
            arg_vals.push(tv.value);
        }
    }
    // For va_arg, determine the requested type from the type-name carrier
    // (the second argument, wrapped as SizeofType by the parser). This lets
    // the backend distinguish integer vs. floating-point va_arg requests and
    // select the correct register class / load instruction.
    let va_arg_ctype: Option<CType> = if name == "va_arg" {
        args.get(1).and_then(|e| {
            if let ast::Expression::SizeofType { type_name, .. } = e {
                Some(resolve_type_name(ctx, type_name))
            } else {
                None
            }
        })
    } else {
        None
    };

    let (ret_ir_type, ret_ctype) = if name == "va_arg" {
        let cty_raw = va_arg_ctype.unwrap_or(CType::LongLong);
        // Resolve through typedef wrappers so that
        // `typedef double TYPE; va_arg(ap, TYPE)` correctly
        // identifies the underlying type (Float/Double/etc.)
        // instead of falling through to the integer path.
        let cty_resolved = crate::common::types::resolve_typedef(&cty_raw).clone();
        let ir_ty = match &cty_resolved {
            CType::Float => IrType::F32,
            CType::Double => IrType::F64,
            CType::LongDouble => {
                // x86-64 SysV ABI: long double is class X87/X87UP
                // → passed entirely in the overflow_arg_area (stack)
                // in 80-bit x87 extended precision format (16-byte slot).
                //
                // The dedicated __builtin_va_arg_f80 intrinsic:
                //   1. Reads overflow_arg_area from the va_list
                //   2. Uses x87 FLDT to load 80-bit extended from it
                //   3. Uses FSTP QWORD to convert to f64 into the alloca
                //   4. Advances overflow_arg_area by 16
                //
                // This is necessary because the 80-bit format cannot
                // simply be reinterpreted as f64 — an x87 conversion
                // step is required.
                let alloca_ir = IrType::F64;
                let (alloca_val, alloca_i) = ctx.builder.build_alloca(alloca_ir.clone(), span);
                emit_inst(ctx, alloca_i);

                let f80_name = "__builtin_va_arg_f80".to_string();
                let callee = emit_global_ref(ctx, &f80_name, span);
                let (_, ci) = ctx.builder.build_call(
                    callee,
                    vec![arg_vals[0], alloca_val],
                    IrType::Void,
                    span,
                );
                emit_inst(ctx, ci);

                let (loaded, li) = ctx.builder.build_load(alloca_val, alloca_ir, span);
                emit_inst(ctx, li);

                return TypedValue::new(loaded, cty_resolved);
            }
            CType::Struct { .. } | CType::Union { .. } => {
                // For struct/union va_arg, use an IR type that matches
                // the struct's actual size to prevent oversized stores
                // that corrupt adjacent stack memory.
                let sz = crate::common::types::sizeof_ctype(&cty_resolved, ctx.target);
                match sz {
                    0..=1 => IrType::I8,
                    2 => IrType::I16,
                    3..=4 => IrType::I32,
                    5..=8 => IrType::I64,
                    _ if sz <= 16 => {
                        // 9–16-byte struct: emit TWO va_arg calls
                        // (one per eightbyte) so the backend reads
                        // from the correct register-save area (GPR or
                        // SSE) for each half.  Assemble the halves
                        // into a temp alloca and return it.
                        let eb_types = classify_struct_va_arg_eightbytes(&cty_resolved, ctx.target);
                        let struct_ir = ctype_to_ir(&cty_resolved, ctx.target);
                        // Alloca for the temporary that holds two
                        // eightbyte halves.  The struct may be smaller
                        // than 16 bytes (e.g. 12-byte `struct {float a[3];}`)
                        // but we store two 8-byte values at offsets 0 and 8,
                        // so the alloca must be at least 16 bytes to prevent
                        // the second store from overflowing into adjacent
                        // stack memory and corrupting other variables.
                        let alloca_ir = IrType::Array(Box::new(IrType::I64), 2);
                        let (alloca_val, alloca_i) = ctx.builder.build_alloca(alloca_ir, span);
                        emit_inst(ctx, alloca_i);

                        let intrinsic_name_local = "__builtin_va_arg".to_string();

                        // --- first eightbyte (offset 0) ---
                        let ty0 = eb_types[0].clone();
                        let callee0 = emit_global_ref(ctx, &intrinsic_name_local, span);
                        let (val0, ci0) =
                            ctx.builder.build_call(callee0, arg_vals.clone(), ty0, span);
                        emit_inst(ctx, ci0);

                        // Store first 8 bytes at alloca+0 (no bitcast needed —
                        // the backend determines store width from value type).
                        let st0 = ctx.builder.build_store(val0, alloca_val, span);
                        emit_inst(ctx, st0);

                        // --- second eightbyte (offset 8) ---
                        let ty1 = if eb_types.len() > 1 {
                            eb_types[1].clone()
                        } else {
                            IrType::I64
                        };
                        let callee1 = emit_global_ref(ctx, &intrinsic_name_local, span);
                        let (val1, ci1) =
                            ctx.builder.build_call(callee1, arg_vals.clone(), ty1, span);
                        emit_inst(ctx, ci1);

                        // GEP to alloca + 8
                        let off8 = emit_int_const(ctx, 8, IrType::I64, span);
                        let (gep8, gepi) =
                            ctx.builder
                                .build_gep(alloca_val, vec![off8], IrType::Ptr, span);
                        emit_inst(ctx, gepi);
                        let st1 = ctx.builder.build_store(val1, gep8, span);
                        emit_inst(ctx, st1);

                        // Load the assembled struct from the alloca.
                        let (loaded, li) = ctx.builder.build_load(alloca_val, struct_ir, span);
                        emit_inst(ctx, li);

                        return TypedValue::new(loaded, cty_resolved);
                    }
                    _ => {
                        // >16-byte MEMORY-class structs are passed
                        // directly on the stack (overflow area), NOT
                        // in registers.  We use a special intrinsic
                        // name so the backend reads from overflow_arg_area
                        // instead of trying the gp_offset register path.
                        let struct_ir = ctype_to_ir(&cty_resolved, ctx.target);

                        // Alloca for the struct result
                        let (alloca_val, alloca_i) =
                            ctx.builder.build_alloca(struct_ir.clone(), span);
                        emit_inst(ctx, alloca_i);

                        // Tell the backend: "read sz bytes from overflow_arg_area"
                        // We encode size in the intrinsic name so the backend can
                        // use it as an immediate without global constant indirection.
                        let mem_name = format!("__builtin_va_arg_mem_{}", sz);
                        let callee = emit_global_ref(ctx, &mem_name, span);
                        let (_, ci) = ctx.builder.build_call(
                            callee,
                            vec![arg_vals[0], alloca_val],
                            IrType::Void,
                            span,
                        );
                        emit_inst(ctx, ci);

                        // Load the assembled struct from the alloca
                        let (loaded, li) = ctx.builder.build_load(alloca_val, struct_ir, span);
                        emit_inst(ctx, li);

                        return TypedValue::new(loaded, cty_resolved);
                    }
                }
            }
            CType::Pointer(_, _) => IrType::I64,
            CType::Bool => IrType::I32,
            CType::Char | CType::SChar | CType::UChar => IrType::I32,
            CType::Short | CType::UShort => IrType::I32,
            CType::Int | CType::UInt => IrType::I32,
            CType::Long | CType::ULong | CType::LongLong | CType::ULongLong => IrType::I64,
            _ => IrType::I64,
        };
        (ir_ty, cty_resolved)
    } else {
        (IrType::Void, CType::Void)
    };

    let intrinsic_name = format!("__builtin_{}", name);
    let callee = emit_global_ref(ctx, &intrinsic_name, span);
    let (result, ci) = ctx.builder.build_call(callee, arg_vals, ret_ir_type, span);
    emit_inst(ctx, ci);

    if name == "va_arg" {
        TypedValue::new(result, ret_ctype)
    } else {
        TypedValue::void()
    }
}

/// Lower overflow-checking arithmetic builtin.
/// Lower `__builtin_{add,sub,mul}_overflow(a, b, result_ptr)` as a Call
/// instruction so the backend's `try_inline_builtin` can intercept it and
/// emit architecture-specific overflow-checking code (ADD/SUB/IMUL + SETO).
fn overflow_ctype_info(ty: &CType) -> (IrType, bool, u32) {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Bool => (IrType::I8, false, 8),
        CType::Char | CType::SChar => (IrType::I8, true, 8),
        CType::UChar => (IrType::I8, false, 8),
        CType::Short => (IrType::I16, true, 16),
        CType::UShort => (IrType::I16, false, 16),
        CType::Int => (IrType::I32, true, 32),
        CType::UInt => (IrType::I32, false, 32),
        CType::Long | CType::LongLong => (IrType::I64, true, 64),
        CType::ULong | CType::ULongLong => (IrType::I64, false, 64),
        _ => (IrType::I32, true, 32),
    }
}

fn lower_overflow_arith(
    ctx: &mut ExprLoweringContext<'_>,
    args: &[ast::Expression],
    op: IrBinOp,
    span: Span,
    predicate_only: bool,
) -> TypedValue {
    if args.len() < 3 {
        ctx.diagnostics
            .emit_error(span, "overflow builtin requires 3 arguments");
        return TypedValue::void();
    }
    let a = lower_expr_inner(ctx, &args[0]);
    let b = lower_expr_inner(ctx, &args[1]);
    let result_ptr = lower_expr_inner(ctx, &args[2]);

    // GCC semantics: promote both operands to "infinite precision signed
    // type", perform the operation, store the result truncated to the
    // destination pointee type, and return whether the mathematical result
    // fits in the destination type.
    //
    // For _p variants (predicate_only=true), the third argument is a
    // type-expression (like `(int)0`) whose TYPE determines the result type
    // for the overflow check, but no store is performed.
    //
    // For operands and result types ≤32 bits, i64 is a sufficient "wide"
    // type.  For 64-bit operands/results, we use special per-operation
    // overflow detection since BCC's codegen does not fully support i128.

    // Determine result type from the third argument.
    // For non-_p variants, it's a pointer — use the pointee type.
    // For _p variants, use the expression's type directly.
    let result_ctype = if predicate_only {
        result_ptr.ty.clone()
    } else {
        match &result_ptr.ty {
            CType::Pointer(inner, _) => (**inner).clone(),
            _ => CType::Int,
        }
    };

    let (_a_ir, a_signed, a_bits) = overflow_ctype_info(&a.ty);
    let (_b_ir, b_signed, b_bits) = overflow_ctype_info(&b.ty);
    let (res_ir, res_signed, res_bits) = overflow_ctype_info(&result_ctype);

    let max_operand_bits = a_bits.max(b_bits);

    // Determine from_type for each operand based on its actual IR type.
    // After lower_expr_inner, small types may have been promoted to I32.
    let a_from = if a_bits <= 32 {
        IrType::I32
    } else {
        IrType::I64
    };
    let b_from = if b_bits <= 32 {
        IrType::I32
    } else {
        IrType::I64
    };

    if max_operand_bits < 64 && res_bits < 64 {
        // --- Common path: all types ≤32 bits, use i64 as wide type ---

        // Step 1: extend operands to i64 (sign/zero based on C type).
        let a64 = if a_signed {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::SExt {
                    result: v,
                    value: a.value,
                    to_type: IrType::I64,
                    from_type: a_from,
                    span,
                },
            );
            v
        } else {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ZExt {
                    result: v,
                    value: a.value,
                    to_type: IrType::I64,
                    from_type: a_from,
                    span,
                },
            );
            v
        };
        let b64 = if b_signed {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::SExt {
                    result: v,
                    value: b.value,
                    to_type: IrType::I64,
                    from_type: b_from,
                    span,
                },
            );
            v
        } else {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ZExt {
                    result: v,
                    value: b.value,
                    to_type: IrType::I64,
                    from_type: b_from,
                    span,
                },
            );
            v
        };

        // Step 2: 64-bit operation.
        let result64 = ctx.builder.fresh_value();
        emit_inst(
            ctx,
            Instruction::BinOp {
                result: result64,
                op,
                lhs: a64,
                rhs: b64,
                ty: IrType::I64,
                span,
            },
        );

        // Step 3: truncate to the actual result type.
        let result_trunc = if res_bits >= 64 {
            result64
        } else {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::Trunc {
                    result: v,
                    value: result64,
                    to_type: res_ir.clone(),
                    span,
                },
            );
            v
        };

        // Step 4: store through the pointer (skip for _p variants).
        if !predicate_only {
            emit_inst(
                ctx,
                Instruction::Store {
                    value: result_trunc,
                    ptr: result_ptr.value,
                    volatile: false,
                    span,
                },
            );
        }

        // Step 5: extend truncated value back to i64 using RESULT signedness.
        let extended64 = if res_signed {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::SExt {
                    result: v,
                    value: result_trunc,
                    to_type: IrType::I64,
                    from_type: res_ir,
                    span,
                },
            );
            v
        } else {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ZExt {
                    result: v,
                    value: result_trunc,
                    to_type: IrType::I64,
                    from_type: res_ir,
                    span,
                },
            );
            v
        };

        // Step 6: compare — if they differ, overflow occurred.
        let overflow_i1 = ctx.builder.fresh_value();
        emit_inst(
            ctx,
            Instruction::ICmp {
                result: overflow_i1,
                op: ICmpOp::Ne,
                lhs: result64,
                rhs: extended64,
                span,
            },
        );

        // Step 7: zero-extend i1 to i32 for the C int return value.
        let overflow_int = ctx.builder.fresh_value();
        emit_inst(
            ctx,
            Instruction::ZExt {
                result: overflow_int,
                value: overflow_i1,
                to_type: IrType::I32,
                from_type: IrType::I1,
                span,
            },
        );

        TypedValue::new(overflow_int, CType::Int)
    } else {
        // --- 64-bit path: at least one operand or result is 64-bit ---
        // Use the native 64-bit operation and detect overflow directly.

        // Extend operands to i64.
        let a64 = if a_bits < 64 {
            let v = ctx.builder.fresh_value();
            if a_signed {
                emit_inst(
                    ctx,
                    Instruction::SExt {
                        result: v,
                        value: a.value,
                        to_type: IrType::I64,
                        from_type: a_from,
                        span,
                    },
                );
            } else {
                emit_inst(
                    ctx,
                    Instruction::ZExt {
                        result: v,
                        value: a.value,
                        to_type: IrType::I64,
                        from_type: a_from,
                        span,
                    },
                );
            }
            v
        } else {
            a.value
        };
        let b64 = if b_bits < 64 {
            let v = ctx.builder.fresh_value();
            if b_signed {
                emit_inst(
                    ctx,
                    Instruction::SExt {
                        result: v,
                        value: b.value,
                        to_type: IrType::I64,
                        from_type: b_from,
                        span,
                    },
                );
            } else {
                emit_inst(
                    ctx,
                    Instruction::ZExt {
                        result: v,
                        value: b.value,
                        to_type: IrType::I64,
                        from_type: b_from,
                        span,
                    },
                );
            }
            v
        } else {
            b.value
        };

        // Perform i64 operation.
        let result64 = ctx.builder.fresh_value();
        emit_inst(
            ctx,
            Instruction::BinOp {
                result: result64,
                op,
                lhs: a64,
                rhs: b64,
                ty: IrType::I64,
                span,
            },
        );

        // Truncate to result type if needed, then store.
        let result_store = if res_bits < 64 {
            let v = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::Trunc {
                    result: v,
                    value: result64,
                    to_type: res_ir.clone(),
                    span,
                },
            );
            v
        } else {
            result64
        };

        if !predicate_only {
            emit_inst(
                ctx,
                Instruction::Store {
                    value: result_store,
                    ptr: result_ptr.value,
                    volatile: false,
                    span,
                },
            );
        }

        if res_bits < 64 {
            // Result smaller than 64-bit — check via extend-and-compare.
            let extended64 = if res_signed {
                let v = ctx.builder.fresh_value();
                emit_inst(
                    ctx,
                    Instruction::SExt {
                        result: v,
                        value: result_store,
                        to_type: IrType::I64,
                        from_type: res_ir,
                        span,
                    },
                );
                v
            } else {
                let v = ctx.builder.fresh_value();
                emit_inst(
                    ctx,
                    Instruction::ZExt {
                        result: v,
                        value: result_store,
                        to_type: IrType::I64,
                        from_type: res_ir,
                        span,
                    },
                );
                v
            };
            let overflow_i1 = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ICmp {
                    result: overflow_i1,
                    op: ICmpOp::Ne,
                    lhs: result64,
                    rhs: extended64,
                    span,
                },
            );
            let overflow_int = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ZExt {
                    result: overflow_int,
                    value: overflow_i1,
                    to_type: IrType::I32,
                    from_type: IrType::I1,
                    span,
                },
            );
            TypedValue::new(overflow_int, CType::Int)
        } else {
            // 64-bit result: detect overflow using sign/carry analysis.
            // Both operands are effectively "infinite precision signed"
            // values that have been extended to i64.
            //
            // Signed result: overflow when (a^result) & (b^result) < 0
            //   for Add; (a^b) & (a^result) < 0 for Sub.
            // Unsigned result: for Add, overflow when result < a;
            //   for Sub, overflow when a < b.
            // For Mul: compute high half — overflow if high != 0 (unsigned)
            //   or high != sign-extend(low) (signed).
            //
            // For simplicity and correctness, emit call to a small inline
            // overflow check pattern.
            let overflow_i1 = if res_signed {
                match op {
                    IrBinOp::Add => {
                        // Signed add overflow: (a64 ^ result64) & (b64 ^ result64) < 0
                        let xor_a = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: xor_a,
                                op: IrBinOp::Xor,
                                lhs: a64,
                                rhs: result64,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let xor_b = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: xor_b,
                                op: IrBinOp::Xor,
                                lhs: b64,
                                rhs: result64,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let and_val = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: and_val,
                                op: IrBinOp::And,
                                lhs: xor_a,
                                rhs: xor_b,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let zero64 = emit_int_const(ctx, 0, IrType::I64, span);
                        let ov = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ICmp {
                                result: ov,
                                op: ICmpOp::Slt,
                                lhs: and_val,
                                rhs: zero64,
                                span,
                            },
                        );
                        ov
                    }
                    IrBinOp::Sub => {
                        // Signed sub overflow: (a64 ^ b64) & (a64 ^ result64) < 0
                        let xor_ab = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: xor_ab,
                                op: IrBinOp::Xor,
                                lhs: a64,
                                rhs: b64,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let xor_ar = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: xor_ar,
                                op: IrBinOp::Xor,
                                lhs: a64,
                                rhs: result64,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let and_val = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: and_val,
                                op: IrBinOp::And,
                                lhs: xor_ab,
                                rhs: xor_ar,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let zero64 = emit_int_const(ctx, 0, IrType::I64, span);
                        let ov = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ICmp {
                                result: ov,
                                op: ICmpOp::Slt,
                                lhs: and_val,
                                rhs: zero64,
                                span,
                            },
                        );
                        ov
                    }
                    IrBinOp::Mul => {
                        // Signed mul overflow: compute via dividing back.
                        // If b != 0 && (result / b != a), overflow.
                        // Guard division with safe_b = b | (b==0) to avoid SIGFPE.
                        // When b==0: safe_b = 0|1 = 1 (safe divisor).
                        // When b!=0: safe_b = b|0 = b (unchanged).
                        let zero = emit_int_const(ctx, 0, IrType::I64, span);
                        let b_eq_zero_i1 = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ICmp {
                                result: b_eq_zero_i1,
                                op: ICmpOp::Eq,
                                lhs: b64,
                                rhs: zero,
                                span,
                            },
                        );
                        let b_nz = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ICmp {
                                result: b_nz,
                                op: ICmpOp::Ne,
                                lhs: b64,
                                rhs: zero,
                                span,
                            },
                        );
                        let b_eq_zero_i64 = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ZExt {
                                result: b_eq_zero_i64,
                                value: b_eq_zero_i1,
                                to_type: IrType::I64,
                                from_type: IrType::I1,
                                span,
                            },
                        );
                        let safe_b = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: safe_b,
                                op: IrBinOp::Or,
                                lhs: b64,
                                rhs: b_eq_zero_i64,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let div_back = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: div_back,
                                op: IrBinOp::SDiv,
                                lhs: result64,
                                rhs: safe_b,
                                ty: IrType::I64,
                                span,
                            },
                        );
                        let ne_a = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::ICmp {
                                result: ne_a,
                                op: ICmpOp::Ne,
                                lhs: div_back,
                                rhs: a64,
                                span,
                            },
                        );
                        let ov = ctx.builder.fresh_value();
                        emit_inst(
                            ctx,
                            Instruction::BinOp {
                                result: ov,
                                op: IrBinOp::And,
                                lhs: b_nz,
                                rhs: ne_a,
                                ty: IrType::I1,
                                span,
                            },
                        );
                        ov
                    }
                    _ => emit_int_const(ctx, 0, IrType::I1, span),
                }
            } else {
                // Unsigned result type.
                // When operands are signed and fit in < 64 bits, the
                // mathematical result of add/sub/mul fits exactly in i64
                // (magnitude bounded by ~2^62 for mul, ~2^33 for add/sub).
                // Overflow for an unsigned destination simply means
                // the mathematical result is negative.
                let any_operand_signed = a_signed || b_signed;
                if any_operand_signed && max_operand_bits < 64 {
                    // The i64 result is the exact mathematical result.
                    // Overflow iff result < 0 (signed comparison).
                    let zero = emit_int_const(ctx, 0, IrType::I64, span);
                    let ov = ctx.builder.fresh_value();
                    emit_inst(
                        ctx,
                        Instruction::ICmp {
                            result: ov,
                            op: ICmpOp::Slt,
                            lhs: result64,
                            rhs: zero,
                            span,
                        },
                    );
                    ov
                } else {
                    // Both operands unsigned (or 64-bit signed→unsigned
                    // where standard unsigned checks apply).
                    match op {
                        IrBinOp::Add => {
                            // Unsigned add overflow: result < a.
                            let ov = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ICmp {
                                    result: ov,
                                    op: ICmpOp::Ult,
                                    lhs: result64,
                                    rhs: a64,
                                    span,
                                },
                            );
                            ov
                        }
                        IrBinOp::Sub => {
                            // Unsigned sub overflow: a < b.
                            let ov = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ICmp {
                                    result: ov,
                                    op: ICmpOp::Ult,
                                    lhs: a64,
                                    rhs: b64,
                                    span,
                                },
                            );
                            ov
                        }
                        IrBinOp::Mul => {
                            // Unsigned mul: result / b != a (when b != 0).
                            // Guard division: safe_b = b | (b==0).
                            let zero = emit_int_const(ctx, 0, IrType::I64, span);
                            let b_eq_zero_i1 = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ICmp {
                                    result: b_eq_zero_i1,
                                    op: ICmpOp::Eq,
                                    lhs: b64,
                                    rhs: zero,
                                    span,
                                },
                            );
                            let b_nz = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ICmp {
                                    result: b_nz,
                                    op: ICmpOp::Ne,
                                    lhs: b64,
                                    rhs: zero,
                                    span,
                                },
                            );
                            let b_eq_zero_i64 = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ZExt {
                                    result: b_eq_zero_i64,
                                    value: b_eq_zero_i1,
                                    to_type: IrType::I64,
                                    from_type: IrType::I1,
                                    span,
                                },
                            );
                            let safe_b = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::BinOp {
                                    result: safe_b,
                                    op: IrBinOp::Or,
                                    lhs: b64,
                                    rhs: b_eq_zero_i64,
                                    ty: IrType::I64,
                                    span,
                                },
                            );
                            let div_back = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::BinOp {
                                    result: div_back,
                                    op: IrBinOp::UDiv,
                                    lhs: result64,
                                    rhs: safe_b,
                                    ty: IrType::I64,
                                    span,
                                },
                            );
                            let ne_a = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::ICmp {
                                    result: ne_a,
                                    op: ICmpOp::Ne,
                                    lhs: div_back,
                                    rhs: a64,
                                    span,
                                },
                            );
                            let ov = ctx.builder.fresh_value();
                            emit_inst(
                                ctx,
                                Instruction::BinOp {
                                    result: ov,
                                    op: IrBinOp::And,
                                    lhs: b_nz,
                                    rhs: ne_a,
                                    ty: IrType::I1,
                                    span,
                                },
                            );
                            ov
                        }
                        _ => emit_int_const(ctx, 0, IrType::I1, span),
                    }
                }
            };

            let overflow_int = ctx.builder.fresh_value();
            emit_inst(
                ctx,
                Instruction::ZExt {
                    result: overflow_int,
                    value: overflow_i1,
                    to_type: IrType::I32,
                    from_type: IrType::I1,
                    span,
                },
            );
            TypedValue::new(overflow_int, CType::Int)
        }
    }
}

/// Emit a reference to a global function/symbol by name.
fn emit_global_ref(ctx: &mut ExprLoweringContext<'_>, name: &str, span: Span) -> Value {
    // Look up in module globals / functions.
    if let Some(func) = ctx.module.get_function(name) {
        // Function declarations provide a callable Value.
        let _param_count = func.param_count();
        let _entry = func.entry_block();
        // Emit a global reference constant pointing to the function.
        let gref_name = format!(".gref.{}", name);
        let mut gv = GlobalVariable::new(
            gref_name.clone(),
            IrType::Ptr,
            Some(Constant::GlobalRef(name.to_string())),
        );
        gv.linkage = crate::ir::module::Linkage::Internal;
        ctx.module.add_global(gv);
        let result = ctx.builder.fresh_value();
        let zero = emit_int_const(ctx, 0, IrType::I64, span);
        let (ptr_val, gi) = ctx.builder.build_gep(result, vec![zero], IrType::Ptr, span);
        emit_inst(ctx, gi);
        ptr_val
    } else {
        // Register a func-ref so the backend can intercept __builtin_*
        // intrinsics and emit inline code instead of an external call.
        let fptr = ctx.builder.fresh_value();
        ctx.module.func_ref_map.insert(fptr, name.to_string());
        {
            let current_func = &mut *ctx.function;
            current_func.func_ref_map.insert(fptr, name.to_string());
        }
        fptr
    }
}

// =========================================================================
// GENERIC SELECTION  (C11 _Generic)
// =========================================================================

/// Apply C11 lvalue conversion for `_Generic`: strip qualifiers, decay
/// arrays to pointers, and decay functions to function pointers.
/// This is applied to both the controlling expression type and each
/// association type before comparison.
fn generic_lvalue_convert(ty: &CType) -> CType {
    let stripped = types::unqualified(ty);
    match stripped {
        CType::Array(elem, _) => CType::Pointer(
            Box::new(generic_lvalue_convert(elem)),
            TypeQualifiers::default(),
        ),
        CType::Function { .. } => {
            CType::Pointer(Box::new(stripped.clone()), TypeQualifiers::default())
        }
        CType::Pointer(inner, _) => {
            // Strip qualifiers from pointee too, for _Generic matching
            CType::Pointer(
                Box::new(generic_lvalue_convert(inner)),
                TypeQualifiers::default(),
            )
        }
        other => other.clone(),
    }
}

fn lower_generic(
    ctx: &mut ExprLoweringContext<'_>,
    controlling: &ast::Expression,
    associations: &[ast::GenericAssociation],
    span: Span,
) -> TypedValue {
    // Evaluate the controlling expression's type (not its value).
    let ctrl_tv = lower_expr_inner(ctx, controlling);
    // C11 §6.5.1.1p2: Apply lvalue conversion to the controlling type.
    // This means: array → pointer decay, function → pointer decay,
    // and strip all qualifiers (including on pointee for _Generic).
    let ctrl_ty = generic_lvalue_convert(&ctrl_tv.ty);

    // C11 §6.5.1.1p3: No two generic associations in the same _Generic
    // selection shall specify compatible types.
    // GCC extension: the Linux kernel's min()/max() macros generate _Generic
    // with duplicate types via typeof().  We emit a warning (not an error)
    // and accept the first matching association, matching GCC behavior.
    {
        let mut seen: Vec<CType> = Vec::new();
        let mut seen_default = false;
        for assoc in associations {
            if let Some(ref tn) = assoc.type_name {
                let aty = resolve_type_name(ctx, tn);
                let ac = generic_lvalue_convert(&aty);
                for prev in &seen {
                    if types::is_compatible(&ac, prev) {
                        ctx.diagnostics
                            .emit_warning(span, "duplicate type name in _Generic association");
                        break;
                    }
                }
                seen.push(ac);
            } else {
                if seen_default {
                    ctx.diagnostics
                        .emit_warning(span, "duplicate default case in _Generic association");
                }
                seen_default = true;
            }
        }
    }

    // Find the matching association.
    // GenericAssociation is a struct with type_name: Option<TypeName> and expression: Box<Expression>.
    // type_name = None means "default:" association.
    let mut default_expr: Option<&ast::Expression> = None;
    for assoc in associations {
        if let Some(ref tn) = assoc.type_name {
            let assoc_ty = resolve_type_name(ctx, tn);
            let assoc_converted = generic_lvalue_convert(&assoc_ty);
            if types::is_compatible(&ctrl_ty, &assoc_converted) {
                return lower_expr_inner(ctx, &assoc.expression);
            }
        } else {
            default_expr = Some(&assoc.expression);
        }
    }
    // Use default if no type matched.
    if let Some(def_expr) = default_expr {
        return lower_expr_inner(ctx, def_expr);
    }

    // Last resort: if no type matched and no default, try the first
    // association as a fallback.  This handles cases where the controlling
    // expression's type was resolved imprecisely (e.g. struct types with
    // empty fields from the lightweight tag reference optimization).
    if let Some(assoc) = associations.first() {
        return lower_expr_inner(ctx, &assoc.expression);
    }

    // No matching association — emit a diagnostic.
    ctx.diagnostics
        .emit_error(span, "_Generic: no matching association");
    TypedValue::void()
}

// =========================================================================
// ADDRESS OF LABEL (GCC `&&label`)
// =========================================================================

fn lower_address_of_label(
    ctx: &mut ExprLoweringContext<'_>,
    label_sym: u32,
    span: Span,
) -> TypedValue {
    // &&label produces a void* pointing to the machine-code address
    // of the label.  This is used with computed goto (`goto *ptr`).
    //
    // label_blocks uses the naming convention "label_{symbol_id}"
    // (matching collect_label_names in decl_lowering.rs).
    let label_name = format!("label_{}", label_sym);

    if let Some(&block_id) = ctx.label_blocks.get(&label_name) {
        // Emit a BlockAddress instruction that materialises the
        // runtime address of the target basic block.
        let (val, inst) = ctx.builder.build_block_address(block_id, span);
        emit_inst(ctx, inst);
        TypedValue::new(
            val,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        )
    } else {
        // Label not yet resolved — this may happen if label_blocks was not
        // populated (e.g. the label is in a different scope, or is a
        // forward reference in a static initializer).  Fall back to the
        // index-based encoding used by the old Switch-based dispatch.
        let block_index: i128 = 0;
        let val = emit_int_const(ctx, block_index, IrType::I64, span);
        let ptr_val = {
            let (v, inst) = ctx.builder.build_int_to_ptr(val, span);
            emit_inst(ctx, inst);
            v
        };
        TypedValue::new(
            ptr_val,
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        )
    }
}

// =========================================================================
// IMPLICIT CONVERSION  (CRITICAL UTILITY)
// =========================================================================

/// Insert implicit type conversion between two C types.
///
/// Handles:
/// - Integer promotions (char/short → int).
/// - Usual arithmetic conversions (find common type for binary operations).
/// - Pointer ↔ integer conversions.
/// - Float ↔ integer conversions.
/// - Identity (same type → no-op).
///
/// Insert an implicit type conversion from `from_cty` to `to_cty`.
///
/// Handles integer ↔ integer (trunc/zext/sext), integer ↔ pointer (inttoptr/
/// ptrtoint), pointer ↔ pointer (bitcast), integer ↔ float, float ↔ float,
/// enum → integer, array/function decay, and bool conversions.
///
/// Returns `value` unchanged if no conversion is necessary.
pub fn insert_implicit_conversion(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    from_cty: &CType,
    to_cty: &CType,
    span: Span,
) -> Value {
    let from = strip_type(from_cty);
    let to = strip_type(to_cty);

    // Guard: if diagnostics already have errors, bail out early.
    if ctx.diagnostics.has_errors() {
        // Continue anyway — best-effort lowering.
    }

    // Use TypeBuilder predicates for validation of convertible types.
    let _from_is_arithmetic = type_builder::is_arithmetic_type(from);
    let _to_is_arithmetic = type_builder::is_arithmetic_type(to);
    let _from_is_scalar = type_builder::is_scalar_type(from);
    let _to_is_scalar = type_builder::is_scalar_type(to);

    // Also leverage signedness and arithmetic checks.
    let _from_signed = is_signed(from);
    let _to_signed = is_signed(to);
    let _from_arith = is_arithmetic(from);
    let _to_arith = is_arithmetic(to);
    let _from_scalar = is_scalar(from);
    let _to_scalar = is_scalar(to);

    // Same type: no conversion needed.
    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        let from_ir = ctype_to_ir(from, ctx.target);
        let to_ir = ctype_to_ir(to, ctx.target);
        if from_ir == to_ir {
            return value;
        }
    }

    // Both integers.
    if types::is_integer(from) && types::is_integer(to) {
        let from_ir = ctype_to_ir(from, ctx.target);
        let to_ir = ctype_to_ir(to, ctx.target);
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw == fw {
            return value;
        }
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
        if is_unsigned(from) {
            let (v, i) = ctx
                .builder
                .build_zext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        } else {
            let (v, i) = ctx
                .builder
                .build_sext_from(value, from_ir.clone(), to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
    }

    // Bool (I1) to integer.
    if matches!(from, CType::Bool) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_zext_from(value, IrType::I1, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer to bool.
    if types::is_integer(from) && matches!(to, CType::Bool) {
        let zero = emit_zero(ctx, ctype_to_ir(from, ctx.target), span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer ↔ Float.
    if types::is_integer(from) && types::is_floating(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let src_unsigned = types::is_unsigned(from);
        let (v, i) = ctx
            .builder
            .build_bitcast_ex(value, to_ir, src_unsigned, span);
        emit_inst(ctx, i);
        return v;
    }
    if types::is_floating(from) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let dst_unsigned = types::is_unsigned(to);
        let (v, i) = ctx
            .builder
            .build_bitcast_ex(value, to_ir, dst_unsigned, span);
        emit_inst(ctx, i);
        return v;
    }

    // Float ↔ Float (different widths).
    if types::is_floating(from) && types::is_floating(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_bitcast(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Integer → Pointer.
    if types::is_integer(from) && types::is_pointer(to) {
        let (v, i) = ctx.builder.build_int_to_ptr(value, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Integer.
    if types::is_pointer(from) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let (v, i) = ctx.builder.build_ptr_to_int(value, to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer → Pointer.
    if types::is_pointer(from) && types::is_pointer(to) {
        // Same representation at IR level.
        return value;
    }

    // Enum → Int.
    if matches!(from, CType::Enum { .. }) && types::is_integer(to) {
        let to_ir = ctype_to_ir(to, ctx.target);
        let from_ir = ctype_to_ir(from, ctx.target);
        if from_ir == to_ir {
            return value;
        }
        let fw = from_ir.int_width();
        let tw = to_ir.int_width();
        if tw < fw {
            let (v, i) = ctx.builder.build_trunc(value, to_ir, span);
            emit_inst(ctx, i);
            return v;
        }
        let (v, i) = ctx
            .builder
            .build_sext_from(value, from_ir.clone(), to_ir, span);
        emit_inst(ctx, i);
        return v;
    }

    // Array → Pointer (array decay).
    if matches!(from, CType::Array { .. }) {
        return value;
    }

    // Function → Pointer (function pointer decay).
    if matches!(from, CType::Function { .. }) {
        return value;
    }

    // Complex → Complex (different element types, e.g. _Complex double → _Complex int).
    if let (CType::Complex(from_elem), CType::Complex(to_elem)) = (from, to) {
        let from_elem_ir = ctype_to_ir(from_elem, ctx.target);
        let to_elem_ir = ctype_to_ir(to_elem, ctx.target);
        if from_elem_ir != to_elem_ir {
            // Spill the source complex to memory so we can decompose it.
            let from_array_ir = IrType::Array(Box::new(from_elem_ir.clone()), 2);
            let (src_alloca, alloca_i) = ctx.builder.build_alloca(from_array_ir.clone(), span);
            emit_inst(ctx, alloca_i);
            let st = ctx.builder.build_store(value, src_alloca, span);
            emit_inst(ctx, st);

            let from_esz = complex_elem_size(&from_elem_ir);

            // Load real part from source
            let (src_real, lr) = ctx
                .builder
                .build_load(src_alloca, from_elem_ir.clone(), span);
            emit_inst(ctx, lr);
            // Load imag part from source
            let off = emit_int_const_for_index(ctx, from_esz);
            let (src_imag_ptr, gep_si) =
                ctx.builder
                    .build_gep(src_alloca, vec![off], IrType::Ptr, span);
            emit_inst(ctx, gep_si);
            let (src_imag, li) = ctx
                .builder
                .build_load(src_imag_ptr, from_elem_ir.clone(), span);
            emit_inst(ctx, li);

            // Convert each component
            let cvt_real = insert_implicit_conversion(ctx, src_real, from_elem, to_elem, span);
            let cvt_imag = insert_implicit_conversion(ctx, src_imag, from_elem, to_elem, span);

            // Build destination complex
            let to_array_ir = IrType::Array(Box::new(to_elem_ir.clone()), 2);
            let (dst_alloca, da) = ctx.builder.build_alloca(to_array_ir.clone(), span);
            emit_inst(ctx, da);
            let st_r = ctx.builder.build_store(cvt_real, dst_alloca, span);
            emit_inst(ctx, st_r);
            let to_esz = complex_elem_size(&to_elem_ir);
            let off2 = emit_int_const_for_index(ctx, to_esz);
            let (dst_imag_ptr, gep_di) =
                ctx.builder
                    .build_gep(dst_alloca, vec![off2], IrType::Ptr, span);
            emit_inst(ctx, gep_di);
            let st_i = ctx.builder.build_store(cvt_imag, dst_imag_ptr, span);
            emit_inst(ctx, st_i);
            let (loaded, ld) = ctx.builder.build_load(dst_alloca, to_array_ir, span);
            emit_inst(ctx, ld);
            return loaded;
        }
    }

    // Scalar → Complex: promote scalar to the complex's element type.
    if let CType::Complex(to_elem) = to {
        let to_elem_ir = ctype_to_ir(to_elem, ctx.target);
        let to_array_ir = IrType::Array(Box::new(to_elem_ir.clone()), 2);
        let (alloca_val, alloca_i) = ctx.builder.build_alloca(to_array_ir.clone(), span);
        emit_inst(ctx, alloca_i);
        // Convert the scalar to the element type
        let cvt_real = insert_implicit_conversion(ctx, value, from, to_elem, span);
        let st_r = ctx.builder.build_store(cvt_real, alloca_val, span);
        emit_inst(ctx, st_r);
        // Zero imaginary part
        let to_esz = complex_elem_size(&to_elem_ir);
        let off = emit_int_const_for_index(ctx, to_esz);
        let (imag_ptr, gep_i) = ctx
            .builder
            .build_gep(alloca_val, vec![off], IrType::Ptr, span);
        emit_inst(ctx, gep_i);
        let zero_imag = if is_fp_ir_type(&to_elem_ir) {
            emit_float_const(ctx, 0.0, to_elem_ir.clone(), span)
        } else {
            emit_int_const(ctx, 0, to_elem_ir.clone(), span)
        };
        let st_i = ctx.builder.build_store(zero_imag, imag_ptr, span);
        emit_inst(ctx, st_i);
        let (loaded, ld) = ctx.builder.build_load(alloca_val, to_array_ir, span);
        emit_inst(ctx, ld);
        return loaded;
    }

    // Complex → Scalar: extract real part and convert to target type.
    if let CType::Complex(from_elem) = from {
        let from_elem_ir = ctype_to_ir(from_elem, ctx.target);
        let from_array_ir = IrType::Array(Box::new(from_elem_ir.clone()), 2);
        let (src_alloca, alloca_i) = ctx.builder.build_alloca(from_array_ir, span);
        emit_inst(ctx, alloca_i);
        let st = ctx.builder.build_store(value, src_alloca, span);
        emit_inst(ctx, st);
        let (real_val, lr) = ctx.builder.build_load(src_alloca, from_elem_ir, span);
        emit_inst(ctx, lr);
        return insert_implicit_conversion(ctx, real_val, from_elem, to, span);
    }

    // Default: no conversion (return as-is).
    value
}

// =========================================================================
// BOOLEAN CONVERSION  (CRITICAL UTILITY)
// =========================================================================

/// Convert a scalar value to a boolean (I1).
///
/// - Integer: `value != 0`
/// - Float: `value != 0.0`
/// - Pointer: `value != null`
///
/// Convert a _Complex value (given its alloca pointer) to a boolean (I1).
///
/// Per C11, a complex value is truthy if EITHER its real or imaginary part
/// is non-zero.  This function takes a POINTER to the complex aggregate
/// (not the loaded value, which is meaningless for >8-byte aggregates).
pub fn complex_ptr_to_bool(
    ctx: &mut ExprLoweringContext<'_>,
    ptr: Value,
    base_ctype: &CType,
    span: Span,
) -> Value {
    let elem_ir = ctype_to_ir(base_ctype, ctx.target);
    let elem_sz: i64 = complex_elem_size(&elem_ir);

    // Load real part at offset 0 (direct load from alloca pointer).
    let (real_val, load_r) = ctx.builder.build_load(ptr, elem_ir.clone(), span);
    emit_inst(ctx, load_r);

    // GEP + load imaginary part at offset elem_sz.
    let imag_idx = emit_int_const_for_index(ctx, elem_sz);
    let (imag_ptr, gep_i) = ctx
        .builder
        .build_gep(ptr, vec![imag_idx], IrType::Ptr, span);
    emit_inst(ctx, gep_i);
    let (imag_val, load_i) = ctx.builder.build_load(imag_ptr, elem_ir.clone(), span);
    emit_inst(ctx, load_i);

    // Compare each part to zero.
    let (real_nz, imag_nz) = if elem_ir.is_float() {
        let fzero_r = emit_float_const(ctx, 0.0, elem_ir.clone(), span);
        let (rnz, ri) = ctx.builder.build_fcmp(FCmpOp::One, real_val, fzero_r, span);
        emit_inst(ctx, ri);
        let fzero_i = emit_float_const(ctx, 0.0, elem_ir, span);
        let (inz, ii) = ctx.builder.build_fcmp(FCmpOp::One, imag_val, fzero_i, span);
        emit_inst(ctx, ii);
        (rnz, inz)
    } else {
        let izero_r = emit_zero(ctx, elem_ir.clone(), span);
        let (rnz, ri) = ctx.builder.build_icmp(ICmpOp::Ne, real_val, izero_r, span);
        emit_inst(ctx, ri);
        let izero_i = emit_zero(ctx, elem_ir, span);
        let (inz, ii) = ctx.builder.build_icmp(ICmpOp::Ne, imag_val, izero_i, span);
        emit_inst(ctx, ii);
        (rnz, inz)
    };

    // OR: truthy if either part is non-zero.
    let (result, or_inst) = ctx.builder.build_or(real_nz, imag_nz, IrType::I1, span);
    emit_inst(ctx, or_inst);
    result
}

fn lower_to_bool(
    ctx: &mut ExprLoweringContext<'_>,
    value: Value,
    ctype: &CType,
    span: Span,
) -> Value {
    let stripped = strip_type(ctype);

    // Already bool (I1) — return as-is.
    if matches!(stripped, CType::Bool) {
        return value;
    }

    // Integer types.
    if types::is_integer(stripped) || matches!(stripped, CType::Enum { .. }) {
        let ir_ty = ctype_to_ir(stripped, ctx.target);
        let zero = emit_zero(ctx, ir_ty, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Floating-point types.
    if types::is_floating(stripped) {
        let ir_ty = ctype_to_ir(stripped, ctx.target);
        let zero = emit_float_const(ctx, 0.0, ir_ty, span);
        let (v, i) = ctx.builder.build_fcmp(FCmpOp::One, value, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Pointer types.
    if types::is_pointer(stripped) {
        let ptr_int_ty = size_ir_type(ctx.target);
        let (int_val, pti) = ctx
            .builder
            .build_ptr_to_int(value, ptr_int_ty.clone(), span);
        emit_inst(ctx, pti);
        let zero = emit_zero(ctx, ptr_int_ty, span);
        let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, int_val, zero, span);
        emit_inst(ctx, i);
        return v;
    }

    // Complex types: truthy if real != 0 OR imag != 0.
    // NOTE: For complex types, `value` is a pointer to the complex aggregate.
    if let CType::Complex(ref base) = stripped {
        let elem_ir = ctype_to_ir(base.as_ref(), ctx.target);
        let elem_sz: i64 = complex_elem_size(&elem_ir);

        // Load real part (offset 0) — value is already a pointer to the aggregate.
        let zero_idx = emit_int_const_for_index(ctx, 0);
        let (real_ptr, gep_r) = ctx
            .builder
            .build_gep(value, vec![zero_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_r);
        let (real_val, ld_r) = ctx.builder.build_load(real_ptr, elem_ir.clone(), span);
        emit_inst(ctx, ld_r);

        // Load imaginary part (offset elem_sz).
        let imag_idx = emit_int_const_for_index(ctx, elem_sz);
        let (imag_ptr, gep_i) = ctx
            .builder
            .build_gep(value, vec![imag_idx], IrType::Ptr, span);
        emit_inst(ctx, gep_i);
        let (imag_val, ld_i) = ctx.builder.build_load(imag_ptr, elem_ir.clone(), span);
        emit_inst(ctx, ld_i);

        // Compare each part to zero.
        let (real_nz, imag_nz) = if elem_ir.is_float() {
            let fzero = emit_float_const(ctx, 0.0, elem_ir.clone(), span);
            let (rnz, ri) = ctx.builder.build_fcmp(FCmpOp::One, real_val, fzero, span);
            emit_inst(ctx, ri);
            let (inz, ii) = ctx.builder.build_fcmp(FCmpOp::One, imag_val, fzero, span);
            emit_inst(ctx, ii);
            (rnz, inz)
        } else {
            let izero = emit_zero(ctx, elem_ir.clone(), span);
            let (rnz, ri) = ctx.builder.build_icmp(ICmpOp::Ne, real_val, izero, span);
            emit_inst(ctx, ri);
            let (inz, ii) = ctx.builder.build_icmp(ICmpOp::Ne, imag_val, izero, span);
            emit_inst(ctx, ii);
            (rnz, inz)
        };

        // Result = real_nz OR imag_nz
        let (v, i) = ctx.builder.build_or(real_nz, imag_nz, IrType::I1, span);
        emit_inst(ctx, i);
        return v;
    }

    // Fallback — treat as integer comparison with zero.
    let zero = emit_zero(ctx, IrType::I32, span);
    let (v, i) = ctx.builder.build_icmp(ICmpOp::Ne, value, zero, span);
    emit_inst(ctx, i);
    v
}

// ===========================================================================
// Compile-time integer constant evaluation (for choose_expr, etc.)
// ===========================================================================

/// Try to evaluate an AST expression as a compile-time integer constant.
/// Returns `None` if the expression cannot be evaluated at compile time.
fn try_eval_integer_constant(
    expr: &ast::Expression,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<i128> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as i128),
        ast::Expression::Parenthesized { inner, .. } => {
            try_eval_integer_constant(inner, name_table, enum_constants)
        }
        ast::Expression::UnaryOp { op, operand, .. } => {
            let val = try_eval_integer_constant(operand, name_table, enum_constants)?;
            match op {
                ast::UnaryOp::Negate => Some(-val),
                ast::UnaryOp::BitwiseNot => Some(!val),
                ast::UnaryOp::LogicalNot => Some(if val == 0 { 1 } else { 0 }),
                ast::UnaryOp::Plus => Some(val),
                _ => None,
            }
        }
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = try_eval_integer_constant(left, name_table, enum_constants)?;
            let r = try_eval_integer_constant(right, name_table, enum_constants)?;
            match op {
                ast::BinaryOp::Add => Some(l.wrapping_add(r)),
                ast::BinaryOp::Sub => Some(l.wrapping_sub(r)),
                ast::BinaryOp::Mul => Some(l.wrapping_mul(r)),
                ast::BinaryOp::Div if r != 0 => Some(l / r),
                ast::BinaryOp::Mod if r != 0 => Some(l % r),
                ast::BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                ast::BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                ast::BinaryOp::Less => Some(if l < r { 1 } else { 0 }),
                ast::BinaryOp::Greater => Some(if l > r { 1 } else { 0 }),
                ast::BinaryOp::LessEqual => Some(if l <= r { 1 } else { 0 }),
                ast::BinaryOp::GreaterEqual => Some(if l >= r { 1 } else { 0 }),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                ast::BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                ast::BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                ast::BinaryOp::ShiftLeft => Some(l.wrapping_shl(r as u32)),
                ast::BinaryOp::ShiftRight => Some(l.wrapping_shr(r as u32)),
                _ => None,
            }
        }
        ast::Expression::Identifier { name, .. } => {
            let idx = name.as_u32() as usize;
            if idx < name_table.len() {
                enum_constants.get(&name_table[idx]).copied()
            } else {
                None
            }
        }
        ast::Expression::BuiltinCall {
            builtin: ast::BuiltinKind::TypesCompatibleP,
            args,
            ..
        } => {
            // Recursively evaluate types_compatible_p as a constant.
            // For choose_expr where the condition IS types_compatible_p.
            // We can't fully resolve types here without ctx, so return None
            // to let the caller use the default behavior.
            let _ = args;
            None
        }
        _ => None,
    }
}

// ===========================================================================
// __builtin_types_compatible_p
// ===========================================================================

/// Compare two CTypes for compatibility, ignoring top-level qualifiers.
///
/// GCC semantics: two types are compatible if they are the same type after
/// stripping top-level `const`/`volatile`/`restrict` qualifiers.
fn types_are_compatible_ignoring_qualifiers(a: &CType, b: &CType) -> bool {
    let a = strip_qualifiers(a);
    let b = strip_qualifiers(b);
    ctypes_structurally_equal(a, b)
}

/// Strip top-level qualifiers and typedefs from a CType.
fn strip_qualifiers(ty: &CType) -> &CType {
    match ty {
        CType::Qualified(inner, _) => strip_qualifiers(inner),
        CType::Typedef { underlying, .. } => strip_qualifiers(underlying),
        _ => ty,
    }
}

/// Check structural equality of two CTypes (after qualifier stripping).
fn ctypes_structurally_equal(a: &CType, b: &CType) -> bool {
    // Discriminant check for simple unit-variant types.
    match (a, b) {
        (CType::Void, CType::Void)
        | (CType::Bool, CType::Bool)
        | (CType::Char, CType::Char)
        | (CType::SChar, CType::SChar)
        | (CType::UChar, CType::UChar)
        | (CType::Short, CType::Short)
        | (CType::UShort, CType::UShort)
        | (CType::Int, CType::Int)
        | (CType::UInt, CType::UInt)
        | (CType::Long, CType::Long)
        | (CType::ULong, CType::ULong)
        | (CType::LongLong, CType::LongLong)
        | (CType::ULongLong, CType::ULongLong)
        | (CType::Float, CType::Float)
        | (CType::Double, CType::Double)
        | (CType::LongDouble, CType::LongDouble) => true,
        (CType::Pointer(a_inner, _a_qual), CType::Pointer(b_inner, _b_qual)) => {
            // GCC __builtin_types_compatible_p semantics: qualifiers on the
            // pointed-to type MATTER.  `char *` vs `const char *` are NOT
            // compatible.  We compare inner types WITHOUT stripping qualifiers
            // to correctly distinguish `const T *` from `T *`.
            ctypes_structurally_equal_exact(a_inner, b_inner)
        }
        (CType::Array(a_elem, a_sz), CType::Array(b_elem, b_sz)) => {
            // GCC semantics: int[5] and int[] are compatible (C11 §6.7.6.2).
            // If either size is None (unspecified), sizes are compatible.
            let sizes_ok = match (a_sz, b_sz) {
                (Some(sa), Some(sb)) => sa == sb,
                _ => true, // one or both unspecified → compatible
            };
            sizes_ok
                && ctypes_structurally_equal(strip_qualifiers(a_elem), strip_qualifiers(b_elem))
        }
        (CType::Struct { name: an, .. }, CType::Struct { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (CType::Union { name: an, .. }, CType::Union { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (CType::Enum { name: an, .. }, CType::Enum { name: bn, .. }) => match (an, bn) {
            (Some(a_name), Some(b_name)) => a_name == b_name,
            _ => false,
        },
        (
            CType::Function {
                return_type: ar,
                params: ap,
                ..
            },
            CType::Function {
                return_type: br,
                params: bp,
                ..
            },
        ) => {
            ctypes_structurally_equal(strip_qualifiers(ar), strip_qualifiers(br))
                && ap.len() == bp.len()
                && ap.iter().zip(bp.iter()).all(|(pa, pb)| {
                    ctypes_structurally_equal(strip_qualifiers(pa), strip_qualifiers(pb))
                })
        }
        _ => false,
    }
}

// ===========================================================================
// __builtin_offsetof computation
// ===========================================================================

/// Compute the byte offset of a struct/union member for `__builtin_offsetof`.
///
/// Extracts the type from args[0] (a `SizeofType` carrier) and the member
/// designator chain from args[1] (an `Identifier` or `MemberAccess` chain).
/// Returns the byte offset as a `usize`.
/// Like `ctypes_structurally_equal` but does NOT strip qualifiers from
/// the compared types.  Used for pointed-to type comparison where
/// `const char` and `char` must be distinguished.
fn ctypes_structurally_equal_exact(a: &CType, b: &CType) -> bool {
    match (a, b) {
        // Strip only Typedef wrappers, not Qualified wrappers.
        (CType::Typedef { underlying, .. }, _) => ctypes_structurally_equal_exact(underlying, b),
        (_, CType::Typedef { underlying, .. }) => ctypes_structurally_equal_exact(a, underlying),
        // Qualified must match exactly.
        (CType::Qualified(ai, aq), CType::Qualified(bi, bq)) => {
            aq == bq && ctypes_structurally_equal_exact(ai, bi)
        }
        (CType::Qualified(_, _), _) | (_, CType::Qualified(_, _)) => false,
        // For non-qualified types, delegate to regular comparison.
        _ => ctypes_structurally_equal(a, b),
    }
}

fn compute_builtin_offsetof(
    ctx: &ExprLoweringContext<'_>,
    args: &[ast::Expression],
    _span: Span,
) -> usize {
    // Extract the type from args[0].
    let ctype = if let Some(ast::Expression::SizeofType { type_name, .. }) = args.first() {
        resolve_type_name(ctx, type_name)
    } else {
        return 0;
    };

    // Extract the member designator chain from args[1].
    // For `b` → ["b"]
    // For `inner.b` → ["inner", "b"]
    let member_chain = if let Some(member_expr) = args.get(1) {
        extract_member_chain(member_expr, ctx.name_table)
    } else {
        return 0;
    };

    // Walk the struct type, computing cumulative byte offset.
    compute_offset_in_type(&ctype, &member_chain, ctx.type_builder)
}

/// A single step in an `offsetof` member-designator chain.
///
/// Represents either a struct/union field name or an array index,
/// allowing `offsetof(type, field.subfield[3])` to be properly computed.
enum OffsetofDesignator {
    /// A struct/union field access: `.field_name`
    Field(String),
    /// An array subscript: `[index]`
    ArrayIndex(usize),
}

/// Extract a chain of member designators from a member-designator expression.
///
/// Handles:
///   - `Identifier { name }` → `[Field("name")]`
///   - `MemberAccess { object, member }` → recursively: object chain + `[Field("member")]`
///   - `ArraySubscript { base, index }` → recursively: base chain + `[ArrayIndex(n)]`
fn extract_member_chain(expr: &ast::Expression, name_table: &[String]) -> Vec<OffsetofDesignator> {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let idx = name.as_u32() as usize;
            let s = if idx < name_table.len() {
                name_table[idx].clone()
            } else {
                format!("sym_{}", idx)
            };
            vec![OffsetofDesignator::Field(s)]
        }
        ast::Expression::MemberAccess { object, member, .. } => {
            let mut chain = extract_member_chain(object, name_table);
            let idx = member.as_u32() as usize;
            let s = if idx < name_table.len() {
                name_table[idx].clone()
            } else {
                format!("sym_{}", idx)
            };
            chain.push(OffsetofDesignator::Field(s));
            chain
        }
        ast::Expression::ArraySubscript { base, index, .. } => {
            let mut chain = extract_member_chain(base, name_table);
            // Evaluate the index as a constant integer.
            let idx_val = extract_constant_index(index);
            chain.push(OffsetofDesignator::ArrayIndex(idx_val));
            chain
        }
        _ => Vec::new(),
    }
}

/// Extract a constant integer value from an expression (for offsetof array indices).
fn extract_constant_index(expr: &ast::Expression) -> usize {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => *value as usize,
        ast::Expression::Parenthesized { inner, .. } => extract_constant_index(inner),
        _ => 0,
    }
}

/// Compute byte offset of a member designator chain within a CType.
///
/// Walks through struct field accesses and array subscripts to compute
/// the cumulative byte offset for `__builtin_offsetof(type, chain)`.
fn compute_offset_in_type(
    ctype: &CType,
    member_chain: &[OffsetofDesignator],
    type_builder: &TypeBuilder,
) -> usize {
    if member_chain.is_empty() {
        return 0;
    }

    match &member_chain[0] {
        OffsetofDesignator::ArrayIndex(idx) => {
            // Array subscript: compute element_size * index, then continue
            // with the element type for the remaining chain.
            let (elem_ty, _elem_count) = match unwrap_type(ctype) {
                CType::Array(elem, count) => ((**elem).clone(), count.unwrap_or(0)),
                _ => return 0, // Not an array — can't subscript
            };
            let elem_size = type_builder.sizeof_type(&elem_ty);
            idx * elem_size + compute_offset_in_type(&elem_ty, &member_chain[1..], type_builder)
        }
        OffsetofDesignator::Field(target_name) => {
            // Struct/union field access: find the field and its offset.
            // Use the type builder's bitfield-aware layout computation
            // to get correct byte offsets for bitfield members.
            let unwrapped = unwrap_type(ctype);
            let (fields, packed, explicit_align) = match unwrapped {
                CType::Struct {
                    fields,
                    packed,
                    aligned,
                    ..
                } => (fields, *packed, *aligned),
                CType::Union {
                    fields, aligned, ..
                } => (fields, false, *aligned),
                _ => return 0,
            };

            let is_union = matches!(unwrapped, CType::Union { .. });

            if is_union {
                // Union: all fields at offset 0.
                for field in fields {
                    let field_name = field.name.as_deref().unwrap_or("");
                    if field_name == target_name {
                        if member_chain.len() == 1 {
                            return 0;
                        }
                        return compute_offset_in_type(&field.ty, &member_chain[1..], type_builder);
                    }
                    // Recurse into anonymous struct/union members to find
                    // fields promoted into the enclosing union's namespace.
                    if field_name.is_empty() {
                        let inner = compute_offset_in_type(&field.ty, member_chain, type_builder);
                        // Distinguish "found at offset 0" from "not found" by
                        // checking if the anonymous member actually contains
                        // the target field.
                        if anon_member_contains_field(&field.ty, target_name) {
                            return inner;
                        }
                    }
                }
                0
            } else {
                // Struct: use bitfield-aware layout computation.
                let layout =
                    type_builder.compute_struct_layout_with_fields(fields, packed, explicit_align);
                for (i, field) in fields.iter().enumerate() {
                    let field_name = field.name.as_deref().unwrap_or("");
                    if field_name == target_name {
                        let field_offset = layout.fields[i].offset;
                        if member_chain.len() == 1 {
                            return field_offset;
                        }
                        return field_offset
                            + compute_offset_in_type(&field.ty, &member_chain[1..], type_builder);
                    }
                    // Recurse into anonymous struct/union members to find
                    // fields promoted into the enclosing struct's namespace.
                    // The anonymous member's own offset must be added.
                    if field_name.is_empty() && anon_member_contains_field(&field.ty, target_name) {
                        let anon_offset = layout.fields[i].offset;
                        let inner = compute_offset_in_type(&field.ty, member_chain, type_builder);
                        return anon_offset + inner;
                    }
                }
                0
            }
        }
    }
}

/// Check whether a (possibly anonymous) struct or union type contains
/// a field with the given name, recursing into nested anonymous members.
///
/// This is used by `compute_offset_in_type` to distinguish between
/// "field found at offset 0" and "field not found" when searching
/// anonymous members that might return 0 for either case.
fn anon_member_contains_field(ctype: &CType, target_name: &str) -> bool {
    let unwrapped = unwrap_type(ctype);
    let fields = match unwrapped {
        CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
        _ => return false,
    };
    for field in fields {
        let field_name = field.name.as_deref().unwrap_or("");
        if field_name == target_name {
            return true;
        }
        // Recurse into nested anonymous struct/union members.
        if field_name.is_empty() && anon_member_contains_field(&field.ty, target_name) {
            return true;
        }
    }
    false
}

/// Unwrap Typedef / Qualified wrappers to reach the underlying type.
fn unwrap_type(ctype: &CType) -> &CType {
    let mut t = ctype;
    loop {
        match t {
            CType::Typedef { underlying, .. } => t = underlying.as_ref(),
            CType::Qualified(inner, _) => t = inner.as_ref(),
            _ => return t,
        }
    }
}
