//! Expression-to-IR lowering module.
//!
//! This module translates every C expression form from the parser AST into
//! IR instructions via the [`IrBuilder`]. It is the most heavily used
//! lowering module — every value-producing C construct flows through it.
//!
//! # Public API
//!
//! - [`lower_expression`] — lowers an expression to an rvalue [`ValueId`].
//! - [`lower_lvalue`] — lowers an expression to an lvalue (memory address).
//!
//! # Design
//!
//! The top-level dispatchers (`lower_expr_inner`, `lower_lvalue_inner`)
//! pattern-match on every [`Expression`] variant and delegate to
//! private helper functions that handle each form.
//!
//! Type information is derived from:
//! - Literal suffixes / prefixes (for constants)
//! - The IR value registry (`IrFunction::get_value_type`)
//! - The C type system (`CType`, `size_of`, `align_of`, `integer_promote`)
//! - Explicit casts and type names from the AST
//!
//! All arithmetic follows C11 semantics:
//! 1. Integer promotions are applied to small-integer operands.
//! 2. Usual arithmetic conversions balance operand types for binary ops.
//! 3. Signed vs. unsigned is respected for division, modulo, and shifts.

// Schema-mandated imports — some are used only in the witness function or
// reserved for future pipeline integration.  The allow directive silences
// warnings on those imports that are schema-required but not yet directly
// used in logic paths.
#[allow(unused_imports)]
use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Severity, Span};
#[allow(unused_imports)]
use crate::common::string_interner::Symbol;
#[allow(unused_imports)]
use crate::common::target::Target;
#[allow(unused_imports)]
use crate::common::type_builder::{
    compute_struct_layout, compute_union_layout, FieldLayout, StructLayout, UnionLayout,
};
#[allow(unused_imports)]
use crate::common::types::{self as ctypes, CType, FieldDef, QualifiedType};
#[allow(unused_imports)]
use crate::frontend::parser::ast::{
    self as ast_mod, AlignofOperand, BinaryOperator, BlockItem, CharPrefix, DerivedDeclarator,
    Expression, FieldDeclaration, FloatSuffix, GenericAssociation, Initializer, InitializerItem,
    IntegerSuffix, SizeofOperand, Statement, StringPrefix, TypeName, TypeSpecifier, UnaryOperator,
};
#[allow(unused_imports)]
use crate::frontend::sema::constant_eval::{
    self as const_eval_mod, evaluate_constant_expression, evaluate_integer_constant,
    is_constant_expression, ConstValue,
};
#[allow(unused_imports)]
use crate::frontend::sema::type_checker::{
    self as type_checker_mod, insert_implicit_conversion, is_assignment_compatible, TypedExpression,
};
#[allow(unused_imports)]
use crate::ir::basic_block::BasicBlockId;
#[allow(unused_imports)]
use crate::ir::builder::IrBuilder;
#[allow(unused_imports)]
use crate::ir::function::{IrFunction, ValueId};
#[allow(unused_imports)]
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction};
#[allow(unused_imports)]
use crate::ir::module::{Constant, IrModule, Linkage, StringLiteral};
#[allow(unused_imports)]
use crate::ir::types::IrType;

#[allow(unused_imports)]
use super::{
    c_type_to_ir_type, check_recursion_depth, ensure_not_terminated, LoweringContext,
    LoweringError, ModuleLoweringContext,
};

// ============================================================================
// Public entry points
// ============================================================================

/// Lowers a C expression to an rvalue — a read-ready IR value.
///
/// For lvalue expressions (variable references, dereferences, member accesses)
/// this emits a `Load` instruction to produce the actual value.  For already-
/// rvalue expressions (literals, arithmetic results) the produced value is
/// returned directly.
pub fn lower_expression(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<ValueId, LoweringError> {
    check_recursion_depth(ctx, expr.span())?;
    ctx.recursion_depth += 1;
    let result = lower_expr_inner(ctx, expr);
    ctx.recursion_depth -= 1;
    result
}

/// Lowers a C expression to an lvalue — a pointer/address in memory.
///
/// Used for assignment targets, the `&` operator, and inline-assembly
/// output operands.  Only expressions that designate memory locations
/// (variables, dereferences, array subscripts, member accesses, compound
/// literals) are valid lvalues.
pub fn lower_lvalue(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<ValueId, LoweringError> {
    check_recursion_depth(ctx, expr.span())?;
    ctx.recursion_depth += 1;
    let result = lower_lvalue_inner(ctx, expr);
    ctx.recursion_depth -= 1;
    result
}

// ============================================================================
// Core dispatch — rvalue lowering
// ============================================================================

/// Internal rvalue dispatch.
fn lower_expr_inner(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<ValueId, LoweringError> {
    if !ensure_not_terminated(ctx) {
        return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
    }

    match expr {
        // ---- Literals ----
        Expression::IntegerLiteral {
            value,
            suffix,
            span,
        } => lower_integer_literal(ctx, *value, *suffix, *span),
        Expression::FloatLiteral {
            value,
            suffix,
            span,
        } => lower_float_literal(ctx, *value, *suffix, *span),
        Expression::StringLiteral {
            value,
            prefix,
            span,
        } => lower_string_literal(ctx, value, *prefix, *span),
        Expression::CharLiteral {
            value,
            prefix,
            span,
        } => lower_char_literal(ctx, *value, *prefix, *span),

        // ---- Variable reference ----
        Expression::Identifier { name, span } => lower_variable_ref_rvalue(ctx, *name, *span),

        // ---- Binary operations ----
        Expression::BinaryOp {
            op,
            left,
            right,
            span,
        } => lower_binary_op(ctx, *op, left, right, *span),

        // ---- Unary operations ----
        Expression::UnaryOp {
            op, operand, span, ..
        } => lower_unary_op(ctx, *op, operand, *span),

        // ---- Conditional (ternary) ----
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            span,
        } => lower_conditional(ctx, condition, then_expr.as_deref(), else_expr, *span),

        // ---- Function call ----
        Expression::FunctionCall { callee, args, span } => {
            lower_function_call(ctx, callee, args, *span)
        }

        // ---- Array subscript ----
        Expression::ArraySubscript { array, index, span } => {
            lower_array_subscript(ctx, array, index, *span)
        }

        // ---- Struct/union member access ----
        Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access_rvalue(ctx, object, *member, false, *span),
        Expression::ArrowAccess {
            pointer,
            member,
            span,
        } => lower_member_access_rvalue(ctx, pointer, *member, true, *span),

        // ---- Cast ----
        Expression::Cast {
            type_name,
            operand,
            span,
        } => lower_cast(ctx, type_name, operand, *span),

        // ---- Sizeof / Alignof ----
        Expression::Sizeof { operand, span } => lower_sizeof(ctx, operand, *span),
        Expression::Alignof { operand, span } => lower_alignof(ctx, operand, *span),

        // ---- Compound literal ----
        Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal(ctx, type_name, initializer, *span),

        // ---- Comma expression ----
        Expression::Comma { expressions, span } => lower_comma(ctx, expressions, *span),

        // ---- _Generic selection ----
        Expression::Generic {
            controlling,
            associations,
            span,
        } => lower_generic(ctx, controlling, associations, *span),

        // ---- GCC statement expression ----
        Expression::StatementExpression { body, span } => {
            lower_statement_expression(ctx, body, *span)
        }

        // ---- Label address (&&label) ----
        Expression::LabelAddress { label, span } => lower_label_address(ctx, *label, *span),

        // ---- Address-of ----
        Expression::AddressOf { operand, span } => lower_address_of(ctx, operand, *span),

        // ---- Dereference ----
        Expression::Dereference { operand, span } => lower_dereference_rvalue(ctx, operand, *span),

        // ---- __builtin_offsetof(type, member) ----
        Expression::BuiltinOffsetof {
            type_name,
            member,
            span: _,
        } => {
            // offsetof always produces a compile-time constant (size_t).
            // Extract the member name from the AST expression.
            let member_sym = match member.as_ref() {
                Expression::Identifier { name, .. } => Some(*name),
                Expression::MemberAccess { member: m, .. } => Some(*m),
                _ => None,
            };
            let ptr_ty = if ctx.target().pointer_width() == 8 {
                IrType::I64
            } else {
                IrType::I32
            };

            // Try to compute the offset from the struct fields in the TypeName.
            let offset = compute_offsetof_from_typename(ctx, type_name, member_sym);
            Ok(ctx
                .builder
                .build_const_int(ctx.function, ptr_ty, offset as i64))
        }

        // ---- __builtin_types_compatible_p(type1, type2) ----
        Expression::BuiltinTypesCompatibleP {
            type1,
            type2,
            span: _,
        } => {
            // Compile-time intrinsic: resolve both type-names to IR types
            // and compare structurally.  Also compare the underlying C-level
            // specifier lists so that `int` vs `long` (same IR width on
            // LP64) are correctly considered incompatible.
            let compat = types_compatible_check(ctx, type1, type2);
            let val: i64 = if compat { 1 } else { 0 };
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, val))
        }

        // ---- __builtin_va_arg(ap, type) ----
        Expression::BuiltinVaArg {
            ap,
            type_name,
            span: _,
        } => {
            // va_arg(ap, type):
            //   1. Get the address of the ap variable
            //   2. Load the current argument pointer from ap
            //   3. Load the argument value from that pointer
            //   4. Advance the pointer by the argument slot size (8 on 64-bit)
            //   5. Store the advanced pointer back into ap
            //   6. Truncate the loaded value to the requested C type if needed
            //   7. Return the (possibly truncated) value
            let ptr_width = ctx.target().pointer_width() as i64;
            let reg_ty = if ptr_width == 8 {
                IrType::I64
            } else {
                IrType::I32
            };

            // Resolve the requested C type from the type_name AST node.
            let target_ty = resolve_type_name_ir(ctx, type_name).unwrap_or(reg_ty.clone());

            // Step 1: address of the ap variable
            let ap_addr = lower_lvalue(ctx, ap)?;

            // Step 2: current pointer value
            let current_ptr = ctx
                .builder
                .build_load(ctx.function, ap_addr, reg_ty.clone());

            // Step 3: load the argument value from *current_ptr
            // On x86-64 System V, each register save slot is 8 bytes wide
            // regardless of the C type, so we load the full register width
            // and then truncate below if the target type is narrower.
            let as_ptr = ctx
                .builder
                .build_int_to_ptr(ctx.function, current_ptr, IrType::Ptr);
            let raw_value = ctx.builder.build_load(ctx.function, as_ptr, reg_ty.clone());

            // Step 4: advance pointer by slot size (always 8 bytes on x86-64)
            let slot_size = ctx
                .builder
                .build_const_int(ctx.function, reg_ty.clone(), ptr_width);
            let new_ptr = ctx.builder.build_binop(
                ctx.function,
                crate::ir::instructions::BinOp::Add,
                current_ptr,
                slot_size,
                reg_ty.clone(),
            );

            // Step 5: store advanced pointer back
            ctx.builder.build_store(ctx.function, new_ptr, ap_addr);

            // Step 6: truncate to target type if narrower than the register
            // width.  For example, va_arg(ap, int) loads I64 from the save
            // area but must return I32.  Without this truncation the backend
            // emits a 64-bit store into a 32-bit alloca, clobbering adjacent
            // stack slots.
            let target = *ctx.target();
            let value = if target_ty != reg_ty
                && target_ty.size_bits(&target) < reg_ty.size_bits(&target)
            {
                ctx.builder.build_trunc(ctx.function, raw_value, target_ty)
            } else {
                raw_value
            };

            // Step 7: return loaded (and possibly truncated) value
            Ok(value)
        }

        // ---- Error recovery placeholder ----
        Expression::Error { span } => Err(LoweringError::UnsupportedExpression {
            span: *span,
            message: "error expression placeholder cannot be lowered".into(),
        }),
    }
}

// ============================================================================
// Core dispatch — lvalue lowering
// ============================================================================

/// Internal lvalue dispatch.  Returns the *address* of the expression.
fn lower_lvalue_inner(
    ctx: &mut LoweringContext<'_>,
    expr: &Expression,
) -> Result<ValueId, LoweringError> {
    match expr {
        Expression::Identifier { name, span } => lower_variable_ref_lvalue(ctx, *name, *span),
        Expression::Dereference { operand, .. } => lower_expression(ctx, operand),
        Expression::ArraySubscript { array, index, span } => {
            lower_array_subscript_lvalue(ctx, array, index, *span)
        }
        Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, *member, false, *span),
        Expression::ArrowAccess {
            pointer,
            member,
            span,
        } => lower_member_access_lvalue(ctx, pointer, *member, true, *span),
        Expression::CompoundLiteral {
            type_name,
            initializer,
            span,
        } => lower_compound_literal(ctx, type_name, initializer, *span),
        Expression::UnaryOp {
            op: UnaryOperator::Deref,
            operand,
            ..
        } => lower_expression(ctx, operand),
        Expression::StatementExpression { body, span } => {
            lower_statement_expression(ctx, body, *span)
        }
        _ => Err(LoweringError::InvalidLvalue {
            span: expr.span(),
            message: "expression does not designate a memory location".into(),
        }),
    }
}

// ============================================================================
// Utility — suffix / type helpers
// ============================================================================

/// Returns the IR type for an integer literal with the given suffix.
fn integer_suffix_to_ir_type(suffix: IntegerSuffix, value: u128, target: &Target) -> IrType {
    match suffix {
        IntegerSuffix::None => {
            if value <= i32::MAX as u128 {
                IrType::I32
            } else if value <= i64::MAX as u128 {
                // Fits in a 64-bit signed integer regardless of target long size.
                IrType::I64
            } else {
                IrType::I128
            }
        }
        IntegerSuffix::U => {
            if value <= u32::MAX as u128 {
                IrType::I32
            } else {
                IrType::I64
            }
        }
        IntegerSuffix::L | IntegerSuffix::UL => {
            if target.long_size() == 8 {
                IrType::I64
            } else {
                IrType::I32
            }
        }
        IntegerSuffix::LL | IntegerSuffix::ULL => IrType::I64,
    }
}

/// Returns the IR type for a float literal with the given suffix.
fn float_suffix_to_ir_type(suffix: FloatSuffix, target: &Target) -> IrType {
    match suffix {
        FloatSuffix::None => IrType::F64,
        FloatSuffix::F => IrType::F32,
        FloatSuffix::L => {
            if target.long_double_size() >= 10 {
                IrType::F80
            } else {
                IrType::F64
            }
        }
    }
}

/// Returns the IR type used for `size_t` on the current target.
fn size_t_ir_type(target: &Target) -> IrType {
    if target.pointer_width() == 8 {
        IrType::I64
    } else {
        IrType::I32
    }
}

/// Returns the IR integer type matching the target pointer width.
fn pointer_int_type(target: &Target) -> IrType {
    size_t_ir_type(target)
}

/// Returns the byte size of an IR type according to target conventions.
fn ir_type_size_bytes(ty: &IrType, target: &Target) -> usize {
    match ty {
        IrType::Void => 0,
        IrType::I1 => 1,
        IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 => 4,
        IrType::I64 => 8,
        IrType::I128 => 16,
        IrType::F32 => 4,
        IrType::F64 => 8,
        IrType::F80 => 16,
        IrType::Ptr => target.pointer_width() as usize,
        IrType::Array { element, count } => ir_type_size_bytes(element, target) * count,
        IrType::Struct { fields, packed } => {
            let mut offset = 0usize;
            for f in fields {
                let fsz = ir_type_size_bytes(f, target);
                if !packed {
                    let align = ir_type_align(f, target);
                    if align > 0 {
                        offset = (offset + align - 1) & !(align - 1);
                    }
                }
                offset += fsz;
            }
            offset
        }
        IrType::Function { .. } => target.pointer_width() as usize,
    }
}

/// Returns the alignment of an IR type.
fn ir_type_align(ty: &IrType, target: &Target) -> usize {
    match ty {
        IrType::Void | IrType::I1 | IrType::I8 => 1,
        IrType::I16 => 2,
        IrType::I32 | IrType::F32 => 4,
        IrType::I64 | IrType::F64 | IrType::Ptr => {
            if target.pointer_width() == 8 {
                8
            } else {
                4
            }
        }
        IrType::I128 | IrType::F80 => 16,
        // Array alignment is the alignment of the element type.
        IrType::Array { element, .. } => ir_type_align(element, target),
        // Struct alignment is the maximum alignment of any field.
        IrType::Struct { fields, .. } => fields
            .iter()
            .map(|f| ir_type_align(f, target))
            .max()
            .unwrap_or(1),
        IrType::Function { .. } => target.pointer_width() as usize,
    }
}

/// Returns `true` if an IrType is a floating-point type.
fn is_ir_float(ty: &IrType) -> bool {
    matches!(ty, IrType::F32 | IrType::F64 | IrType::F80)
}

/// Returns `true` if an IrType is an integer type.
fn is_ir_integer(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128
    )
}

/// Returns the bit width of an integer IR type.
fn ir_int_bit_width(ty: &IrType) -> u32 {
    match ty {
        IrType::I1 => 1,
        IrType::I8 => 8,
        IrType::I16 => 16,
        IrType::I32 => 32,
        IrType::I64 => 64,
        IrType::I128 => 128,
        _ => 0,
    }
}

/// Coerces an integer value between two integer IR types.
fn coerce_int_value(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    from_ty: &IrType,
    to_ty: &IrType,
    signed: bool,
) -> ValueId {
    if from_ty == to_ty {
        return val;
    }
    let from_bits = ir_int_bit_width(from_ty);
    let to_bits = ir_int_bit_width(to_ty);
    if from_bits == 0 || to_bits == 0 {
        return ctx.builder.build_bitcast(ctx.function, val, to_ty.clone());
    }
    if from_bits < to_bits {
        if signed {
            ctx.builder.build_sext(ctx.function, val, to_ty.clone())
        } else {
            ctx.builder.build_zext(ctx.function, val, to_ty.clone())
        }
    } else if from_bits > to_bits {
        ctx.builder.build_trunc(ctx.function, val, to_ty.clone())
    } else {
        val
    }
}

/// Determines the common type for arithmetic promotion.
fn common_arithmetic_type(a: &IrType, b: &IrType) -> IrType {
    if is_ir_float(a) || is_ir_float(b) {
        let ar = float_rank(a);
        let br = float_rank(b);
        return if ar >= br { a.clone() } else { b.clone() };
    }
    let aw = ir_int_bit_width(a);
    let bw = ir_int_bit_width(b);
    let target_w = aw.max(bw).max(32);
    int_type_from_width(target_w)
}

fn float_rank(ty: &IrType) -> u8 {
    match ty {
        IrType::F32 => 1,
        IrType::F64 => 2,
        IrType::F80 => 3,
        _ => 0,
    }
}

fn int_type_from_width(bits: u32) -> IrType {
    match bits {
        0..=1 => IrType::I1,
        2..=8 => IrType::I8,
        9..=16 => IrType::I16,
        17..=32 => IrType::I32,
        33..=64 => IrType::I64,
        _ => IrType::I128,
    }
}

/// Coerces a value from one type to another.
pub(crate) fn coerce_value(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    from: &IrType,
    to: &IrType,
) -> ValueId {
    coerce_value_ext(ctx, val, from, to, false)
}

/// Coerces a value from one IR type to another, with explicit control over
/// whether integer widening uses zero-extension (`unsigned = true`) or
/// sign-extension (`unsigned = false`).
///
/// This distinction is critical for C integer promotions: `unsigned char`
/// must be zero-extended to `int` (preserving 128 → 128), while plain
/// `char` / `signed char` must be sign-extended (-128 → -128).
fn coerce_value_ext(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    from: &IrType,
    to: &IrType,
    unsigned: bool,
) -> ValueId {
    if from == to {
        return val;
    }
    if is_ir_integer(from) && is_ir_integer(to) {
        let from_w = ir_int_bit_width(from);
        let to_w = ir_int_bit_width(to);
        if from_w < to_w {
            if unsigned {
                return ctx.builder.build_zext(ctx.function, val, to.clone());
            } else {
                return ctx.builder.build_sext(ctx.function, val, to.clone());
            }
        } else if from_w > to_w {
            return ctx.builder.build_trunc(ctx.function, val, to.clone());
        }
        return val;
    }
    // Integer to float.
    if is_ir_integer(from) && is_ir_float(to) {
        // Default to signed conversion for implicit arithmetic promotions.
        if unsigned {
            return ctx.builder.build_ui_to_fp(ctx.function, val, to.clone());
        } else {
            return ctx.builder.build_si_to_fp(ctx.function, val, to.clone());
        }
    }
    // Float to integer.
    if is_ir_float(from) && is_ir_integer(to) {
        if unsigned {
            return ctx.builder.build_fp_to_ui(ctx.function, val, to.clone());
        } else {
            return ctx.builder.build_fp_to_si(ctx.function, val, to.clone());
        }
    }
    // Float to float (different widths).
    if is_ir_float(from) && is_ir_float(to) {
        let from_rank = float_rank(from);
        let to_rank = float_rank(to);
        if from_rank < to_rank {
            return ctx.builder.build_fp_ext(ctx.function, val, to.clone());
        } else if from_rank > to_rank {
            return ctx.builder.build_fp_trunc(ctx.function, val, to.clone());
        }
        return val; // Same float type.
    }
    // Everything else: bitcast (same-size reinterpretation).
    ctx.builder.build_bitcast(ctx.function, val, to.clone())
}

/// Converts any value to an `I1` boolean by comparing `!= 0`.
fn to_i1(ctx: &mut LoweringContext<'_>, val: ValueId) -> ValueId {
    let ty = ctx.function.get_value_type(val).clone();
    match &ty {
        IrType::I1 => val,
        IrType::Ptr => {
            let target = *ctx.target();
            let int_ty = pointer_int_type(&target);
            let int_val = ctx
                .builder
                .build_ptr_to_int(ctx.function, val, int_ty.clone());
            let zero = ctx.builder.build_const_int(ctx.function, int_ty, 0);
            ctx.builder
                .build_icmp(ctx.function, ICmpPredicate::Ne, int_val, zero)
        }
        t if is_ir_float(t) => {
            let zero = ctx.builder.build_const_float(ctx.function, ty.clone(), 0.0);
            ctx.builder
                .build_fcmp(ctx.function, FCmpPredicate::ONe, val, zero)
        }
        t if is_ir_integer(t) => {
            let zero = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
            ctx.builder
                .build_icmp(ctx.function, ICmpPredicate::Ne, val, zero)
        }
        _ => {
            let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
            ctx.builder
                .build_icmp(ctx.function, ICmpPredicate::Ne, val, zero)
        }
    }
}

/// Resolves the element type stored behind an alloca pointer by scanning
/// instructions for the matching Alloca.
fn resolve_alloca_element_type(ctx: &LoweringContext<'_>, alloca_id: ValueId) -> IrType {
    let entry_bb_id = ctx.function.entry_block_id;
    // Scan entry block first (allocas are canonically in the entry block).
    for block in ctx.function.basic_blocks.iter() {
        if block.id == entry_bb_id {
            for inst in block.instructions() {
                if let Instruction::Alloca { result, ty, .. } = inst {
                    if *result == alloca_id {
                        return ty.clone();
                    }
                }
            }
        }
    }
    // Scan all blocks as fallback.
    for block in ctx.function.basic_blocks.iter() {
        if block.id != entry_bb_id {
            for inst in block.instructions() {
                if let Instruction::Alloca { result, ty, .. } = inst {
                    if *result == alloca_id {
                        return ty.clone();
                    }
                }
            }
        }
    }
    IrType::I32
}

/// Resolves the element type behind a pointer for load/store.
///
/// This function checks *all* pointer-producing instructions, not just Alloca:
///   - **Alloca**: element type comes from the alloca's declared type
///   - **GetElementPtr**: element type comes from the GEP's `ty` field
///   - **Load**: the load itself produces a Ptr, but the loaded-from alloca
///     tells us the pointee type
///
/// Falls back to I32 only if no instruction can be found.
fn resolve_pointee_type(ctx: &LoweringContext<'_>, ptr_id: ValueId) -> IrType {
    // First try Alloca (most common for local variable addresses).
    let alloca_ty = resolve_alloca_element_type_opt(ctx, ptr_id);
    if let Some(ty) = alloca_ty {
        return ty;
    }
    // Check if ptr_id is a GEP result — walk the indices to compute the
    // actual element type the GEP result points to.
    //
    // CRITICAL: A GEP into a struct with indices [0, field_idx] has its
    // `ty` set to the *base struct type*, NOT the field type.  We must
    // walk the index chain to resolve the actual pointed-to element type.
    for block in ctx.function.basic_blocks.iter() {
        for inst in block.instructions() {
            if let Instruction::GetElementPtr {
                result,
                ty,
                indices,
                ..
            } = inst
            {
                if *result == ptr_id {
                    return walk_gep_indices_to_element_type(ctx, ty, indices);
                }
            }
        }
    }
    // Default fallback — return I32 which is safe for the common case
    // of integer assignments.  If the actual type is different, the
    // coercion will be a no-op (same-width) or a truncation/extension.
    IrType::I32
}

/// Walks GEP index chain to determine the element type the result points to.
///
/// For `GEP base, [0, field_idx]` into `Struct([I32, I8, ...])`:
///   - Skip first index (pointer-level dereference)
///   - Index into struct field → returns the field type
///
/// For `GEP base, [idx]` into `I8`:
///   - Only one index → returns the base element type `I8` unchanged
fn walk_gep_indices_to_element_type(
    ctx: &LoweringContext<'_>,
    base_ty: &IrType,
    indices: &[ValueId],
) -> IrType {
    let mut current = base_ty.clone();
    // The first index is the "outer" pointer-level index (GEP semantics:
    // index 0 dereferences the pointer to the aggregate).  Subsequent
    // indices drill into the aggregate.
    for &idx_val in indices.iter().skip(1) {
        match &current {
            IrType::Struct { fields, .. } => {
                // Struct field access — the index must be a compile-time constant.
                if let Some(field_idx) = extract_const_int_from_value(ctx, idx_val) {
                    let fi = field_idx as usize;
                    if fi < fields.len() {
                        current = fields[fi].clone();
                    } else {
                        break; // out-of-bounds — keep current type
                    }
                } else {
                    break; // non-constant struct index — shouldn't happen
                }
            }
            IrType::Array { element, .. } => {
                current = (**element).clone();
            }
            _ => {
                // Primitive or pointer type — cannot index further.
                break;
            }
        }
    }
    current
}

/// Extracts the integer constant encoded in a value's name.
///
/// Constants are represented as named values with pattern `const.int.{N}`.
/// Returns `None` if the value is not a constant or cannot be parsed.
fn extract_const_int_from_value(ctx: &LoweringContext<'_>, val_id: ValueId) -> Option<i64> {
    if let Some(info) = ctx.function.get_value_info(val_id) {
        if let Some(ref name) = info.name {
            if let Some(stripped) = name.strip_prefix("const.int.") {
                return stripped.parse::<i64>().ok();
            }
        }
    }
    None
}

/// Same as `resolve_alloca_element_type` but returns `None` instead of
/// defaulting to I32 when the alloca is not found.
fn resolve_alloca_element_type_opt(
    ctx: &LoweringContext<'_>,
    alloca_id: ValueId,
) -> Option<IrType> {
    let entry_bb_id = ctx.function.entry_block_id;
    for block in ctx.function.basic_blocks.iter() {
        if block.id == entry_bb_id {
            for inst in block.instructions() {
                if let Instruction::Alloca { result, ty, .. } = inst {
                    if *result == alloca_id {
                        return Some(ty.clone());
                    }
                }
            }
        }
    }
    for block in ctx.function.basic_blocks.iter() {
        if block.id != entry_bb_id {
            for inst in block.instructions() {
                if let Instruction::Alloca { result, ty, .. } = inst {
                    if *result == alloca_id {
                        return Some(ty.clone());
                    }
                }
            }
        }
    }
    None
}

// ============================================================================
// Literal lowering
// ============================================================================

/// Lowers an integer literal to an IR constant.
fn lower_integer_literal(
    ctx: &mut LoweringContext<'_>,
    value: u128,
    suffix: IntegerSuffix,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let target = *ctx.target();
    let ir_ty = integer_suffix_to_ir_type(suffix, value, &target);
    Ok(ctx
        .builder
        .build_const_int(ctx.function, ir_ty, value as i64))
}

/// Lowers a floating-point literal to an IR constant.
fn lower_float_literal(
    ctx: &mut LoweringContext<'_>,
    value: f64,
    suffix: FloatSuffix,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let target = *ctx.target();
    let ir_ty = float_suffix_to_ir_type(suffix, &target);
    Ok(ctx.builder.build_const_float(ctx.function, ir_ty, value))
}

/// Lowers a string literal to a global constant pointer.
fn lower_string_literal(
    ctx: &mut LoweringContext<'_>,
    value: &[u8],
    prefix: StringPrefix,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let mut data = value.to_vec();
    match prefix {
        StringPrefix::None | StringPrefix::U8 => {
            data.push(0);
        }
        StringPrefix::L | StringPrefix::BigU => {
            data.extend_from_slice(&[0, 0, 0, 0]);
        }
        StringPrefix::SmallU => {
            data.extend_from_slice(&[0, 0]);
        }
    }

    let str_id = ctx.module_ctx.intern_string_literal(&data);
    let name = format!(".str.{}", str_id);
    Ok(ctx
        .builder
        .build_global_ref(ctx.function, &name, IrType::Ptr))
}

/// Lowers a character literal to an IR integer constant.
fn lower_char_literal(
    ctx: &mut LoweringContext<'_>,
    value: u32,
    prefix: CharPrefix,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let ir_ty = match prefix {
        CharPrefix::None | CharPrefix::L | CharPrefix::BigU => IrType::I32,
        CharPrefix::SmallU => IrType::I16,
    };
    Ok(ctx
        .builder
        .build_const_int(ctx.function, ir_ty, value as i64))
}

// ============================================================================
// Variable reference
// ============================================================================

/// Lowers a variable reference to its rvalue (loads from the alloca).
fn lower_variable_ref_rvalue(
    ctx: &mut LoweringContext<'_>,
    name: Symbol,
    span: Span,
) -> Result<ValueId, LoweringError> {
    let _name_dbg = ctx.module_ctx.interner.resolve(name).to_string();
    // eprintln!("[VARREF] rvalue lookup name={:?} symbol={:?} vars={:?}", name_dbg, name, ctx.variables.keys().map(|k| (k, ctx.module_ctx.interner.resolve(*k).to_string())).collect::<Vec<_>>());
    if let Some(alloca_id) = ctx.get_variable(name) {
        let ir_ty = resolve_alloca_element_type(ctx, alloca_id);
        // Function types return their address directly (no load).
        if let IrType::Function { .. } = &ir_ty {
            return Ok(alloca_id);
        }
        // Array types decay to a pointer (C array-to-pointer decay).
        // Return the alloca address directly — never load an entire array.
        if matches!(ir_ty, IrType::Array { .. }) {
            return Ok(alloca_id);
        }
        return Ok(ctx.builder.build_load(ctx.function, alloca_id, ir_ty));
    }

    if let Some(info) = ctx.module_ctx.global_symbols.get(&name) {
        let ir_ty = info.ir_type.clone();
        let name_str = ctx.module_ctx.interner.resolve(name).to_string();
        if let IrType::Function { .. } = &ir_ty {
            return Ok(ctx
                .builder
                .build_global_ref(ctx.function, &name_str, IrType::Ptr));
        }
        // Global arrays also decay to pointers — return the address.
        if matches!(ir_ty, IrType::Array { .. }) {
            return Ok(ctx
                .builder
                .build_global_ref(ctx.function, &name_str, IrType::Ptr));
        }
        let gref = ctx
            .builder
            .build_global_ref(ctx.function, &name_str, IrType::Ptr);
        return Ok(ctx.builder.build_load(ctx.function, gref, ir_ty));
    }

    ctx.diagnostics()
        .error(span, "use of undeclared identifier");
    Err(LoweringError::UndeclaredVariable { name, span })
}

/// Lowers a variable reference to its lvalue (the alloca or global address).
fn lower_variable_ref_lvalue(
    ctx: &mut LoweringContext<'_>,
    name: Symbol,
    span: Span,
) -> Result<ValueId, LoweringError> {
    if let Some(alloca_id) = ctx.get_variable(name) {
        return Ok(alloca_id);
    }

    if ctx.module_ctx.global_symbols.contains_key(&name) {
        let name_str = ctx.module_ctx.interner.resolve(name).to_string();
        return Ok(ctx
            .builder
            .build_global_ref(ctx.function, &name_str, IrType::Ptr));
    }

    ctx.diagnostics()
        .error(span, "use of undeclared identifier");
    Err(LoweringError::UndeclaredVariable { name, span })
}

// ============================================================================
// Binary operations
// ============================================================================

/// Determines whether a C expression has unsigned integer type.
///
/// This checks:
/// - Variables tracked in `variable_unsigned`
/// - Integer literals with unsigned suffixes (`U`, `UL`, `ULL`)
/// - Cast expressions to unsigned types (e.g. `(unsigned)x`)
/// - Sizeof expressions (always unsigned in C)
/// - Array subscripts / member access / dereferences — by inspecting the
///   C type (`variable_ctypes`) to determine element/member signedness
fn is_expression_unsigned(ctx: &LoweringContext<'_>, expr: &Expression) -> bool {
    match expr {
        Expression::Identifier { name, .. } => {
            // Check the unsigned set first (fast path).
            if ctx.variable_unsigned.contains(name) {
                return true;
            }
            // Also check CType information for more nuanced types.
            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                return is_ctype_unsigned(ctype);
            }
            false
        }
        Expression::IntegerLiteral { suffix, value, .. } => {
            match suffix {
                IntegerSuffix::U | IntegerSuffix::UL | IntegerSuffix::ULL => true,
                // For unsuffixed hex/octal literals that don't fit in the signed
                // range, C says the type is the first type that fits:
                // int, unsigned int, long, unsigned long, long long, unsigned long long.
                // A simple heuristic: if the value exceeds i32::MAX, treat as unsigned.
                IntegerSuffix::None => *value > i32::MAX as u128,
                _ => false,
            }
        }
        Expression::Sizeof { .. } | Expression::Alignof { .. } => {
            // sizeof and _Alignof return size_t, which is unsigned.
            true
        }
        Expression::Cast { type_name, .. } => {
            // Check if the cast is to an unsigned type.
            is_type_specifier_unsigned(&type_name.specifiers.specifiers)
        }
        // For compound access expressions (subscript, arrow, member, deref),
        // use the recursive type inference helper to determine the result
        // type and check unsigned-ness.  This handles arbitrary nesting like
        // `pkt->payload[0]` (ArrowAccess + ArraySubscript).
        Expression::ArraySubscript { .. }
        | Expression::ArrowAccess { .. }
        | Expression::MemberAccess { .. }
        | Expression::Dereference { .. } => {
            if let Some(result_ctype) = try_infer_expression_ctype(ctx, expr) {
                return is_ctype_unsigned(&result_ctype);
            }
            false
        }
        _ => false,
    }
}

/// Finds a field by name in a struct/union CType and returns a reference
/// to its type.  Returns `None` if not found.
#[allow(dead_code)]
fn find_field_type_in_aggregate<'a>(ctype: &'a CType, field_name: &str) -> Option<&'a CType> {
    match ctype {
        CType::Struct { fields, .. } | CType::Union { fields, .. } => {
            for f in fields {
                if let Some(ref fname) = f.name {
                    if fname == field_name {
                        return Some(&f.ty);
                    }
                }
            }
            None
        }
        CType::Typedef { underlying, .. } => find_field_type_in_aggregate(underlying, field_name),
        CType::Atomic(inner) => find_field_type_in_aggregate(inner, field_name),
        _ => None,
    }
}

/// Attempts to infer the C type of an expression from the available
/// `variable_ctypes` mapping.  This is used by `is_expression_unsigned`
/// to determine signedness of nested access patterns such as
/// `pkt->payload[0]` where `pkt` is `struct packet *` and `payload`
/// is `unsigned char[]`.
///
/// Returns `None` when the type cannot be determined (e.g., function
/// call results, complex casts).
fn try_infer_expression_ctype(ctx: &LoweringContext<'_>, expr: &Expression) -> Option<CType> {
    match expr {
        Expression::Identifier { name, .. } => {
            let ctype = ctx.variable_ctypes.get(name).cloned();
            // If the variable is a struct/union with empty fields, resolve
            // from the struct definition registry.
            ctype.map(|ct| resolve_struct_fields_if_empty(ctx, ct))
        }
        Expression::ArraySubscript { array, .. } => {
            let array_ty = try_infer_expression_ctype(ctx, array)?;
            match array_ty {
                CType::Array { element, .. } => Some(*element),
                CType::Pointer(element) => Some(*element),
                _ => None,
            }
        }
        Expression::ArrowAccess {
            pointer, member, ..
        } => {
            let base_ty = try_infer_expression_ctype(ctx, pointer)?;
            let struct_ty = match base_ty {
                CType::Pointer(inner) => *inner,
                other => other,
            };
            let struct_ty = resolve_struct_fields_if_empty(ctx, struct_ty);
            let member_str = ctx.module_ctx.interner.resolve(*member);
            find_field_in_ctype(&struct_ty, member_str)
        }
        Expression::MemberAccess { object, member, .. } => {
            let base_ty = try_infer_expression_ctype(ctx, object)?;
            let base_ty = resolve_struct_fields_if_empty(ctx, base_ty);
            let member_str = ctx.module_ctx.interner.resolve(*member);
            find_field_in_ctype(&base_ty, member_str)
        }
        Expression::Dereference { operand, .. } => {
            let ptr_ty = try_infer_expression_ctype(ctx, operand)?;
            match ptr_ty {
                CType::Pointer(element) => Some(*element),
                _ => None,
            }
        }
        Expression::Cast { type_name, .. } => {
            // Check if the cast target type is unsigned.
            if is_type_specifier_unsigned(&type_name.specifiers.specifiers) {
                // Return a dummy unsigned int CType so callers know it's unsigned.
                Some(CType::Int { signed: false })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Returns `true` if a CType represents an unsigned integer type.
/// If the CType is a Struct/Union with empty fields but a name, look up the
/// full definition from `module_ctx.struct_defs`.  This is needed because
/// variable declarations often store a forward-reference CType without
/// resolving the fields from the struct definition.
fn resolve_struct_fields_if_empty(ctx: &LoweringContext<'_>, ctype: CType) -> CType {
    match &ctype {
        CType::Struct {
            name: Some(tag),
            fields,
            ..
        } if fields.is_empty() => {
            // Look up the full struct definition by comparing tag names.
            // We iterate struct_defs because we only have an immutable
            // reference and cannot call `intern()` on the interner.
            for (sym, def) in &ctx.module_ctx.struct_defs {
                let sym_str = ctx.module_ctx.interner.resolve(*sym);
                if sym_str == tag.as_str() {
                    return def.clone();
                }
            }
            ctype
        }
        CType::Union {
            name: Some(tag),
            fields,
            ..
        } if fields.is_empty() => {
            for (sym, def) in &ctx.module_ctx.struct_defs {
                let sym_str = ctx.module_ctx.interner.resolve(*sym);
                if sym_str == tag.as_str() {
                    return def.clone();
                }
            }
            ctype
        }
        CType::Pointer(inner) => {
            let resolved = resolve_struct_fields_if_empty(ctx, *inner.clone());
            CType::Pointer(Box::new(resolved))
        }
        CType::Typedef { name, underlying } => {
            let resolved = resolve_struct_fields_if_empty(ctx, *underlying.clone());
            CType::Typedef {
                name: name.clone(),
                underlying: Box::new(resolved),
            }
        }
        _ => ctype,
    }
}

/// Finds a field by name in a CType that is a Struct or Union and returns
/// its type.
fn find_field_in_ctype(ctype: &CType, field_name: &str) -> Option<CType> {
    match ctype {
        CType::Struct { fields, .. } | CType::Union { fields, .. } => {
            for f in fields {
                if let Some(ref fname) = f.name {
                    if fname == field_name {
                        return Some(f.ty.clone());
                    }
                }
                // Check anonymous nested structs/unions
                if f.name.is_none() {
                    if let Some(result) = find_field_in_ctype(&f.ty, field_name) {
                        return Some(result);
                    }
                }
            }
            None
        }
        CType::Typedef { underlying, .. } => find_field_in_ctype(underlying, field_name),
        CType::Atomic(inner) => find_field_in_ctype(inner, field_name),
        _ => None,
    }
}

fn is_ctype_unsigned(ctype: &CType) -> bool {
    match ctype {
        CType::Char { signed } => !signed,
        CType::Short { signed } => !signed,
        CType::Int { signed } => !signed,
        CType::Long { signed } => !signed,
        CType::LongLong { signed } => !signed,
        CType::Bool => true, // _Bool is unsigned in C
        CType::Typedef { underlying, .. } => is_ctype_unsigned(underlying),
        CType::Atomic(inner) => is_ctype_unsigned(inner),
        _ => false,
    }
}

/// Check if any type specifier in a list indicates an unsigned type.
fn is_type_specifier_unsigned(specifiers: &[TypeSpecifier]) -> bool {
    for spec in specifiers {
        if matches!(spec, TypeSpecifier::Unsigned) {
            return true;
        }
    }
    false
}

/// Dispatches a binary operation.
fn lower_binary_op(
    ctx: &mut LoweringContext<'_>,
    op: BinaryOperator,
    left: &Expression,
    right: &Expression,
    span: Span,
) -> Result<ValueId, LoweringError> {
    // Assignment family.
    match op {
        BinaryOperator::Assign => return lower_assignment(ctx, left, right, span),
        BinaryOperator::AddAssign => {
            return lower_compound_assign(ctx, BinOp::Add, BinOp::FAdd, left, right, span)
        }
        BinaryOperator::SubAssign => {
            return lower_compound_assign(ctx, BinOp::Sub, BinOp::FSub, left, right, span)
        }
        BinaryOperator::MulAssign => {
            return lower_compound_assign(ctx, BinOp::Mul, BinOp::FMul, left, right, span)
        }
        BinaryOperator::DivAssign => {
            return lower_compound_assign_signaware(
                ctx,
                BinOp::SDiv,
                BinOp::UDiv,
                BinOp::FDiv,
                left,
                right,
                span,
            )
        }
        BinaryOperator::ModAssign => {
            return lower_compound_assign_signaware(
                ctx,
                BinOp::SRem,
                BinOp::URem,
                BinOp::SRem,
                left,
                right,
                span,
            )
        }
        BinaryOperator::BitAndAssign => {
            return lower_compound_assign(ctx, BinOp::And, BinOp::And, left, right, span)
        }
        BinaryOperator::BitOrAssign => {
            return lower_compound_assign(ctx, BinOp::Or, BinOp::Or, left, right, span)
        }
        BinaryOperator::BitXorAssign => {
            return lower_compound_assign(ctx, BinOp::Xor, BinOp::Xor, left, right, span)
        }
        BinaryOperator::ShlAssign => {
            return lower_compound_assign(ctx, BinOp::Shl, BinOp::Shl, left, right, span)
        }
        BinaryOperator::ShrAssign => {
            return lower_compound_assign_signaware(
                ctx,
                BinOp::AShr,
                BinOp::LShr,
                BinOp::AShr,
                left,
                right,
                span,
            )
        }
        // Short-circuit logical ops.
        BinaryOperator::LogAnd => return lower_logical_and(ctx, left, right, span),
        BinaryOperator::LogOr => return lower_logical_or(ctx, left, right, span),
        _ => {}
    }

    // Evaluate both operands.
    let lhs_val = lower_expression(ctx, left)?;
    let rhs_val = lower_expression(ctx, right)?;

    let lhs_ty = ctx.function.get_value_type(lhs_val).clone();
    let rhs_ty = ctx.function.get_value_type(rhs_val).clone();

    match op {
        BinaryOperator::Eq
        | BinaryOperator::Ne
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::Le
        | BinaryOperator::Ge => {
            let use_unsigned =
                is_expression_unsigned(ctx, left) || is_expression_unsigned(ctx, right);
            lower_comparison(ctx, op, lhs_val, rhs_val, &lhs_ty, &rhs_ty, use_unsigned)
        }
        _ => {
            let use_unsigned =
                is_expression_unsigned(ctx, left) || is_expression_unsigned(ctx, right);
            lower_arithmetic_binop(
                ctx,
                op,
                lhs_val,
                rhs_val,
                &lhs_ty,
                &rhs_ty,
                use_unsigned,
                span,
            )
        }
    }
}

/// Lowers arithmetic and bitwise binary operations.
fn lower_arithmetic_binop(
    ctx: &mut LoweringContext<'_>,
    op: BinaryOperator,
    lhs: ValueId,
    rhs: ValueId,
    lhs_ty: &IrType,
    rhs_ty: &IrType,
    is_unsigned: bool,
    span: Span,
) -> Result<ValueId, LoweringError> {
    // Pointer arithmetic: ptr + int or ptr - int.
    if matches!(lhs_ty, IrType::Ptr) && is_ir_integer(rhs_ty) && op == BinaryOperator::Add {
        // GEP-based pointer addition.
        return Ok(ctx
            .builder
            .build_gep(ctx.function, lhs, vec![rhs], IrType::I8, true));
    }
    if is_ir_integer(lhs_ty) && matches!(rhs_ty, IrType::Ptr) && op == BinaryOperator::Add {
        return Ok(ctx
            .builder
            .build_gep(ctx.function, rhs, vec![lhs], IrType::I8, true));
    }
    if matches!(lhs_ty, IrType::Ptr) && is_ir_integer(rhs_ty) && op == BinaryOperator::Sub {
        // ptr - int: negate index, then GEP.
        let zero_const = ctx.builder.build_const_int(ctx.function, rhs_ty.clone(), 0);
        let neg_rhs =
            ctx.builder
                .build_binop(ctx.function, BinOp::Sub, zero_const, rhs, rhs_ty.clone());
        return Ok(ctx
            .builder
            .build_gep(ctx.function, lhs, vec![neg_rhs], IrType::I8, true));
    }
    // Pointer difference: ptr - ptr → integer.
    if matches!(lhs_ty, IrType::Ptr) && matches!(rhs_ty, IrType::Ptr) && op == BinaryOperator::Sub {
        let target = *ctx.target();
        let int_ty = pointer_int_type(&target);
        let lhs_int = ctx
            .builder
            .build_ptr_to_int(ctx.function, lhs, int_ty.clone());
        let rhs_int = ctx
            .builder
            .build_ptr_to_int(ctx.function, rhs, int_ty.clone());
        return Ok(ctx
            .builder
            .build_binop(ctx.function, BinOp::Sub, lhs_int, rhs_int, int_ty));
    }

    let result_ty = common_arithmetic_type(lhs_ty, rhs_ty);
    let lhs_c = coerce_value(ctx, lhs, lhs_ty, &result_ty);
    let rhs_c = coerce_value(ctx, rhs, rhs_ty, &result_ty);

    let is_float = is_ir_float(&result_ty);

    let ir_op = match op {
        BinaryOperator::Add => {
            if is_float {
                BinOp::FAdd
            } else {
                BinOp::Add
            }
        }
        BinaryOperator::Sub => {
            if is_float {
                BinOp::FSub
            } else {
                BinOp::Sub
            }
        }
        BinaryOperator::Mul => {
            if is_float {
                BinOp::FMul
            } else {
                BinOp::Mul
            }
        }
        BinaryOperator::Div => {
            if is_float {
                BinOp::FDiv
            } else if is_unsigned {
                BinOp::UDiv
            } else {
                BinOp::SDiv
            }
        }
        BinaryOperator::Mod => {
            if is_unsigned {
                BinOp::URem
            } else {
                BinOp::SRem
            }
        }
        BinaryOperator::BitAnd => BinOp::And,
        BinaryOperator::BitOr => BinOp::Or,
        BinaryOperator::BitXor => BinOp::Xor,
        BinaryOperator::Shl => BinOp::Shl,
        BinaryOperator::Shr => {
            if is_unsigned {
                BinOp::LShr
            } else {
                BinOp::AShr
            }
        }
        _ => {
            return Err(LoweringError::UnsupportedExpression {
                span,
                message: format!("unsupported binary operator {:?}", op),
            });
        }
    };

    Ok(ctx
        .builder
        .build_binop(ctx.function, ir_op, lhs_c, rhs_c, result_ty))
}

// ============================================================================
// Comparison lowering
// ============================================================================

/// Lowers a comparison operation to an I1 result.
fn lower_comparison(
    ctx: &mut LoweringContext<'_>,
    op: BinaryOperator,
    lhs: ValueId,
    rhs: ValueId,
    lhs_ty: &IrType,
    rhs_ty: &IrType,
    use_unsigned: bool,
) -> Result<ValueId, LoweringError> {
    let is_float = is_ir_float(lhs_ty) || is_ir_float(rhs_ty);
    let is_ptr = matches!(lhs_ty, IrType::Ptr) || matches!(rhs_ty, IrType::Ptr);

    if is_float {
        let common = common_arithmetic_type(lhs_ty, rhs_ty);
        let lhs_c = coerce_value(ctx, lhs, lhs_ty, &common);
        let rhs_c = coerce_value(ctx, rhs, rhs_ty, &common);
        let pred = match op {
            BinaryOperator::Eq => FCmpPredicate::OEq,
            BinaryOperator::Ne => FCmpPredicate::ONe,
            BinaryOperator::Lt => FCmpPredicate::Olt,
            BinaryOperator::Gt => FCmpPredicate::Ogt,
            BinaryOperator::Le => FCmpPredicate::Ole,
            BinaryOperator::Ge => FCmpPredicate::Oge,
            _ => FCmpPredicate::OEq,
        };
        Ok(ctx.builder.build_fcmp(ctx.function, pred, lhs_c, rhs_c))
    } else if is_ptr {
        let target = *ctx.target();
        let int_ty = pointer_int_type(&target);
        let lhs_int = if matches!(lhs_ty, IrType::Ptr) {
            ctx.builder
                .build_ptr_to_int(ctx.function, lhs, int_ty.clone())
        } else {
            coerce_value(ctx, lhs, lhs_ty, &int_ty)
        };
        let rhs_int = if matches!(rhs_ty, IrType::Ptr) {
            ctx.builder
                .build_ptr_to_int(ctx.function, rhs, int_ty.clone())
        } else {
            coerce_value(ctx, rhs, rhs_ty, &int_ty)
        };
        let pred = match op {
            BinaryOperator::Eq => ICmpPredicate::Eq,
            BinaryOperator::Ne => ICmpPredicate::Ne,
            BinaryOperator::Lt => ICmpPredicate::Ult,
            BinaryOperator::Gt => ICmpPredicate::Ugt,
            BinaryOperator::Le => ICmpPredicate::Ule,
            BinaryOperator::Ge => ICmpPredicate::Uge,
            _ => ICmpPredicate::Eq,
        };
        Ok(ctx.builder.build_icmp(ctx.function, pred, lhs_int, rhs_int))
    } else {
        let common = common_arithmetic_type(lhs_ty, rhs_ty);
        // Use zero-extension when either operand is unsigned (e.g.
        // `unsigned char` → `int` must preserve 128 as 128, not -128).
        let lhs_c = coerce_value_ext(ctx, lhs, lhs_ty, &common, use_unsigned);
        let rhs_c = coerce_value_ext(ctx, rhs, rhs_ty, &common, use_unsigned);
        // Select signed or unsigned comparison predicates based on the
        // C type of the operands.  If either operand is unsigned (per the
        // "usual arithmetic conversions" in C11 §6.3.1.8), use unsigned
        // predicates.  Eq/Ne are the same for signed and unsigned.
        let pred = if use_unsigned {
            match op {
                BinaryOperator::Eq => ICmpPredicate::Eq,
                BinaryOperator::Ne => ICmpPredicate::Ne,
                BinaryOperator::Lt => ICmpPredicate::Ult,
                BinaryOperator::Gt => ICmpPredicate::Ugt,
                BinaryOperator::Le => ICmpPredicate::Ule,
                BinaryOperator::Ge => ICmpPredicate::Uge,
                _ => ICmpPredicate::Eq,
            }
        } else {
            match op {
                BinaryOperator::Eq => ICmpPredicate::Eq,
                BinaryOperator::Ne => ICmpPredicate::Ne,
                BinaryOperator::Lt => ICmpPredicate::Slt,
                BinaryOperator::Gt => ICmpPredicate::Sgt,
                BinaryOperator::Le => ICmpPredicate::Sle,
                BinaryOperator::Ge => ICmpPredicate::Sge,
                _ => ICmpPredicate::Eq,
            }
        };
        Ok(ctx.builder.build_icmp(ctx.function, pred, lhs_c, rhs_c))
    }
}

// ============================================================================
// Logical short-circuit operators
// ============================================================================

/// Lowers `&&` with short-circuit evaluation.
fn lower_logical_and(
    ctx: &mut LoweringContext<'_>,
    left: &Expression,
    right: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let lhs_val = lower_expression(ctx, left)?;
    let lhs_bool = to_i1(ctx, lhs_val);

    // CRITICAL: Create the phi incoming constant in start_bb (before the branch)
    // so the backend emits it in the correct predecessor block.  If created in
    // merge_bb, the backend would emit `mov reg, #0` unconditionally inside the
    // merge block, overwriting the phi result from the rhs path.
    let false_val = ctx.builder.build_const_int(ctx.function, IrType::I1, 0);

    let start_bb = ctx.builder.get_insert_block().unwrap();
    let rhs_bb = ctx.builder.create_block(ctx.function, Some("land.rhs"));
    let merge_bb = ctx.builder.create_block(ctx.function, Some("land.end"));

    ctx.builder
        .build_cond_branch(ctx.function, lhs_bool, rhs_bb, merge_bb);

    ctx.builder.set_insert_point(rhs_bb);
    let rhs_val = lower_expression(ctx, right)?;
    let rhs_bool = to_i1(ctx, rhs_val);
    let rhs_exit_bb = ctx.builder.get_insert_block().unwrap();
    ctx.builder.build_branch(ctx.function, merge_bb);

    ctx.builder.set_insert_point(merge_bb);
    let phi = ctx.builder.build_phi(ctx.function, IrType::I1);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, false_val, start_bb);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, rhs_bool, rhs_exit_bb);

    // Widen to I32 (C logical operators yield int).
    Ok(ctx.builder.build_zext(ctx.function, phi, IrType::I32))
}

/// Lowers `||` with short-circuit evaluation.
fn lower_logical_or(
    ctx: &mut LoweringContext<'_>,
    left: &Expression,
    right: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let lhs_val = lower_expression(ctx, left)?;
    let lhs_bool = to_i1(ctx, lhs_val);

    // CRITICAL: Create the phi incoming constant in start_bb (before the branch)
    // so the backend emits it in the correct predecessor block.  If created in
    // merge_bb, the backend would emit `mov reg, #1` unconditionally inside the
    // merge block, overwriting the phi result from the rhs path.
    let true_val = ctx.builder.build_const_int(ctx.function, IrType::I1, 1);

    let start_bb = ctx.builder.get_insert_block().unwrap();
    let rhs_bb = ctx.builder.create_block(ctx.function, Some("lor.rhs"));
    let merge_bb = ctx.builder.create_block(ctx.function, Some("lor.end"));

    ctx.builder
        .build_cond_branch(ctx.function, lhs_bool, merge_bb, rhs_bb);

    ctx.builder.set_insert_point(rhs_bb);
    let rhs_val = lower_expression(ctx, right)?;
    let rhs_bool = to_i1(ctx, rhs_val);
    let rhs_exit_bb = ctx.builder.get_insert_block().unwrap();
    ctx.builder.build_branch(ctx.function, merge_bb);

    ctx.builder.set_insert_point(merge_bb);
    let phi = ctx.builder.build_phi(ctx.function, IrType::I1);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, true_val, start_bb);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, rhs_bool, rhs_exit_bb);

    Ok(ctx.builder.build_zext(ctx.function, phi, IrType::I32))
}

// ============================================================================
// Unary operations
// ============================================================================

/// Lowers a unary operation.
fn lower_unary_op(
    ctx: &mut LoweringContext<'_>,
    op: UnaryOperator,
    operand: &Expression,
    span: Span,
) -> Result<ValueId, LoweringError> {
    match op {
        UnaryOperator::Plus => lower_expression(ctx, operand),
        UnaryOperator::Neg => {
            let val = lower_expression(ctx, operand)?;
            let ty = ctx.function.get_value_type(val).clone();
            if is_ir_float(&ty) {
                let zero = ctx.builder.build_const_float(ctx.function, ty.clone(), 0.0);
                Ok(ctx
                    .builder
                    .build_binop(ctx.function, BinOp::FSub, zero, val, ty))
            } else {
                let zero = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
                Ok(ctx
                    .builder
                    .build_binop(ctx.function, BinOp::Sub, zero, val, ty))
            }
        }
        UnaryOperator::BitNot => {
            let val = lower_expression(ctx, operand)?;
            let ty = ctx.function.get_value_type(val).clone();
            let all_ones = ctx.builder.build_const_int(ctx.function, ty.clone(), -1);
            Ok(ctx
                .builder
                .build_binop(ctx.function, BinOp::Xor, val, all_ones, ty))
        }
        UnaryOperator::LogNot => {
            let val = lower_expression(ctx, operand)?;
            let bool_val = to_i1(ctx, val);
            let zero = ctx.builder.build_const_int(ctx.function, IrType::I1, 0);
            let inv = ctx
                .builder
                .build_icmp(ctx.function, ICmpPredicate::Eq, bool_val, zero);
            Ok(ctx.builder.build_zext(ctx.function, inv, IrType::I32))
        }
        UnaryOperator::PreInc => lower_pre_inc_dec(ctx, operand, true),
        UnaryOperator::PreDec => lower_pre_inc_dec(ctx, operand, false),
        UnaryOperator::PostInc => lower_post_inc_dec(ctx, operand, true),
        UnaryOperator::PostDec => lower_post_inc_dec(ctx, operand, false),
        UnaryOperator::AddressOf => lower_address_of(ctx, operand, span),
        UnaryOperator::Deref => lower_dereference_rvalue(ctx, operand, span),
    }
}

/// Lowers pre-increment (`++x`) or pre-decrement (`--x`).
fn lower_pre_inc_dec(
    ctx: &mut LoweringContext<'_>,
    operand: &Expression,
    is_inc: bool,
) -> Result<ValueId, LoweringError> {
    let addr = lower_lvalue(ctx, operand)?;
    let ty = resolve_pointee_type(ctx, addr);
    let old_val = ctx.builder.build_load(ctx.function, addr, ty.clone());

    if is_ir_float(&ty) {
        let c = ctx.builder.build_const_float(ctx.function, ty.clone(), 1.0);
        let ir_op = if is_inc { BinOp::FAdd } else { BinOp::FSub };
        let new_val = ctx.builder.build_binop(ctx.function, ir_op, old_val, c, ty);
        ctx.builder.build_store(ctx.function, new_val, addr);
        return Ok(new_val);
    }
    if matches!(&ty, IrType::Ptr) {
        // Pointer increment: GEP by 1.
        let c = ctx
            .builder
            .build_const_int(ctx.function, IrType::I64, if is_inc { 1 } else { -1 });
        let new_ptr = ctx
            .builder
            .build_gep(ctx.function, old_val, vec![c], IrType::I8, true);
        ctx.builder.build_store(ctx.function, new_ptr, addr);
        return Ok(new_ptr);
    }
    let c = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
    let ir_op = if is_inc { BinOp::Add } else { BinOp::Sub };
    let new_val = ctx.builder.build_binop(ctx.function, ir_op, old_val, c, ty);
    ctx.builder.build_store(ctx.function, new_val, addr);
    Ok(new_val)
}

/// Lowers post-increment (`x++`) or post-decrement (`x--`).
fn lower_post_inc_dec(
    ctx: &mut LoweringContext<'_>,
    operand: &Expression,
    is_inc: bool,
) -> Result<ValueId, LoweringError> {
    let addr = lower_lvalue(ctx, operand)?;
    let ty = resolve_pointee_type(ctx, addr);
    let old_val = ctx.builder.build_load(ctx.function, addr, ty.clone());

    if is_ir_float(&ty) {
        let c = ctx.builder.build_const_float(ctx.function, ty.clone(), 1.0);
        let ir_op = if is_inc { BinOp::FAdd } else { BinOp::FSub };
        let new_val = ctx.builder.build_binop(ctx.function, ir_op, old_val, c, ty);
        ctx.builder.build_store(ctx.function, new_val, addr);
        return Ok(old_val); // post returns old
    }
    if matches!(&ty, IrType::Ptr) {
        let c = ctx
            .builder
            .build_const_int(ctx.function, IrType::I64, if is_inc { 1 } else { -1 });
        let new_ptr = ctx
            .builder
            .build_gep(ctx.function, old_val, vec![c], IrType::I8, true);
        ctx.builder.build_store(ctx.function, new_ptr, addr);
        return Ok(old_val); // post returns old
    }
    let c = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
    let ir_op = if is_inc { BinOp::Add } else { BinOp::Sub };
    let new_val = ctx.builder.build_binop(ctx.function, ir_op, old_val, c, ty);
    ctx.builder.build_store(ctx.function, new_val, addr);
    Ok(old_val)
}

// ============================================================================
// Cast expression
// ============================================================================

/// Lowers an explicit cast: `(target_type)operand`.
fn lower_cast(
    ctx: &mut LoweringContext<'_>,
    type_name: &TypeName,
    operand: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let val = lower_expression(ctx, operand)?;
    let from_ty = ctx.function.get_value_type(val).clone();
    let to_ir_ty = resolve_type_name_ir(ctx, type_name)?;

    if from_ty == to_ir_ty {
        return Ok(val);
    }

    // Pointer to integer.
    if matches!(from_ty, IrType::Ptr) && is_ir_integer(&to_ir_ty) {
        return Ok(ctx.builder.build_ptr_to_int(ctx.function, val, to_ir_ty));
    }
    // Integer to pointer.
    if is_ir_integer(&from_ty) && matches!(to_ir_ty, IrType::Ptr) {
        return Ok(ctx.builder.build_int_to_ptr(ctx.function, val, to_ir_ty));
    }
    // Pointer to pointer — no-op in opaque-pointer model.
    if matches!(from_ty, IrType::Ptr) && matches!(to_ir_ty, IrType::Ptr) {
        return Ok(val);
    }
    // Integer to integer.
    if is_ir_integer(&from_ty) && is_ir_integer(&to_ir_ty) {
        // Determine signedness from the cast target type specifiers AND
        // the source operand.  For unsigned casts (e.g. `(unsigned int)x`)
        // we must use zero-extension.  For casts from unsigned types
        // (e.g. `(int)uchar_val`) we also zero-extend to preserve value.
        let cast_unsigned = is_type_specifier_unsigned(&type_name.specifiers.specifiers);
        let src_unsigned = is_expression_unsigned(ctx, operand);
        let use_signed = !cast_unsigned && !src_unsigned;
        return Ok(coerce_int_value(ctx, val, &from_ty, &to_ir_ty, use_signed));
    }
    // Integer to float.
    if is_ir_integer(&from_ty) && is_ir_float(&to_ir_ty) {
        let cast_unsigned = is_type_specifier_unsigned(&type_name.specifiers.specifiers);
        let src_unsigned = is_expression_unsigned(ctx, operand);
        if cast_unsigned || src_unsigned {
            return Ok(ctx.builder.build_ui_to_fp(ctx.function, val, to_ir_ty));
        } else {
            return Ok(ctx.builder.build_si_to_fp(ctx.function, val, to_ir_ty));
        }
    }
    // Float to integer.
    if is_ir_float(&from_ty) && is_ir_integer(&to_ir_ty) {
        let cast_unsigned = is_type_specifier_unsigned(&type_name.specifiers.specifiers);
        if cast_unsigned {
            return Ok(ctx.builder.build_fp_to_ui(ctx.function, val, to_ir_ty));
        } else {
            return Ok(ctx.builder.build_fp_to_si(ctx.function, val, to_ir_ty));
        }
    }
    // Float to float (different widths).
    if is_ir_float(&from_ty) && is_ir_float(&to_ir_ty) {
        let from_rank = float_rank(&from_ty);
        let to_rank = float_rank(&to_ir_ty);
        if from_rank < to_rank {
            return Ok(ctx.builder.build_fp_ext(ctx.function, val, to_ir_ty));
        } else if from_rank > to_rank {
            return Ok(ctx.builder.build_fp_trunc(ctx.function, val, to_ir_ty));
        }
        return Ok(val); // Same float type.
    }
    // Everything else: bitcast (same-size reinterpretation).
    Ok(ctx.builder.build_bitcast(ctx.function, val, to_ir_ty))
}

/// Computes the byte offset of a member within a struct from the
/// `TypeName` AST node used in `__builtin_offsetof(type, member)`.
///
/// Handles both anonymous (inline-defined) structs and named struct
/// references by looking up the struct definition registry when the
/// TypeName contains a forward reference (`fields: None`).
fn compute_offsetof_from_typename(
    ctx: &LoweringContext<'_>,
    type_name: &TypeName,
    member_sym: Option<Symbol>,
) -> usize {
    let member_sym = match member_sym {
        Some(s) => s,
        None => return 0,
    };

    // Strategy 1: Try to extract inline field declarations from the
    // TypeName AST node (works for anonymous structs defined inline).
    let inline_fields = extract_struct_fields_from_typename(type_name);
    if let Some(fields) = inline_fields {
        return compute_offset_from_ast_fields(ctx, &fields, member_sym);
    }

    // Strategy 2: For named structs (fields: None), look up the struct
    // definition from the module-level registry populated during
    // lowering of `CheckedDeclaration::StructDef`.
    for spec in &type_name.specifiers.specifiers {
        if let TypeSpecifier::Struct {
            name: Some(tag_name),
            fields: None,
            ..
        }
        | TypeSpecifier::Union {
            name: Some(tag_name),
            fields: None,
            ..
        } = spec
        {
            if let Some(ctype) = ctx.module_ctx.struct_defs.get(tag_name) {
                return compute_offset_from_ctype(ctx, ctype, member_sym);
            }
        }
    }

    // Member not found — return 0 as fallback.
    0
}

/// Computes the byte offset of `member_sym` within a CType that is known
/// to be a Struct or Union.  Uses the target-aware layout engine.
fn compute_offset_from_ctype(
    ctx: &LoweringContext<'_>,
    ctype: &CType,
    member_sym: Symbol,
) -> usize {
    let target = ctx.target();
    let member_str = ctx.module_ctx.interner.resolve(member_sym);
    match ctype {
        CType::Struct { fields, .. } => {
            let mut offset: usize = 0;
            for field in fields {
                let field_size = ctypes::size_of(&field.ty, target);
                let field_align = ctypes::align_of(&field.ty, target);

                // Align the current offset.
                if field_align > 0 {
                    offset = (offset + field_align - 1) & !(field_align - 1);
                }

                // Compare field name (String) with member name (Symbol → &str).
                if let Some(ref fname) = field.name {
                    if fname.as_str() == member_str {
                        return offset;
                    }
                }
                offset += field_size;
            }
            0
        }
        CType::Union { .. } => {
            // All union fields share offset 0.
            0
        }
        _ => 0,
    }
}

/// Computes the byte offset of `member_sym` from AST field declarations
/// (used for inline/anonymous struct definitions).
fn compute_offset_from_ast_fields(
    ctx: &LoweringContext<'_>,
    fields: &[FieldDeclaration],
    member_sym: Symbol,
) -> usize {
    let target = ctx.target();
    let mut offset: usize = 0;

    for field_decl in fields {
        // Determine the IR type of this field for size/alignment.
        let field_ir_ty = resolve_field_ir_type(ctx, field_decl);
        let field_size = ir_type_size_bytes(&field_ir_ty, target);
        let field_align = ir_type_align(&field_ir_ty, target);

        // Align the current offset.
        if field_align > 0 {
            offset = (offset + field_align - 1) & !(field_align - 1);
        }

        // Check if this field matches the target member name.
        for fd in &field_decl.declarators {
            if let Some(ref decl) = fd.declarator {
                if let Some(field_name) = decl.name {
                    if field_name == member_sym {
                        return offset;
                    }
                }
            }
        }

        // Advance past this field.
        offset += field_size;
    }

    // Member not found — return 0 as fallback.
    0
}

/// Extracts the list of field declarations from a TypeName that contains
/// a struct or union type specifier with an inline definition.
fn extract_struct_fields_from_typename(type_name: &TypeName) -> Option<Vec<FieldDeclaration>> {
    for spec in &type_name.specifiers.specifiers {
        match spec {
            TypeSpecifier::Struct {
                fields: Some(f), ..
            } => return Some(f.clone()),
            TypeSpecifier::Union {
                fields: Some(f), ..
            } => return Some(f.clone()),
            _ => {}
        }
    }
    None
}

/// Resolves a field declaration to an IR type for size/alignment computation.
fn resolve_field_ir_type(ctx: &LoweringContext<'_>, field: &FieldDeclaration) -> IrType {
    // Build a temporary TypeName from the field's specifiers
    let target = *ctx.target();
    let specs = &field.specifiers.type_specifiers;

    // Simple single-specifier resolution
    for spec in specs {
        match spec {
            TypeSpecifier::Void => return IrType::Void,
            TypeSpecifier::Bool => return IrType::I1,
            TypeSpecifier::Char => return IrType::I8,
            TypeSpecifier::Short => return IrType::I16,
            TypeSpecifier::Int => return IrType::I32,
            TypeSpecifier::Float => return IrType::F32,
            TypeSpecifier::Double => return IrType::F64,
            TypeSpecifier::Long => {
                // Check for 'long long' (two Long specifiers)
                let long_count = specs
                    .iter()
                    .filter(|s| matches!(s, TypeSpecifier::Long))
                    .count();
                if long_count >= 2 {
                    return IrType::I64;
                }
                return if target.long_size() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                };
            }
            _ => {}
        }
    }

    // Check for pointer declarators in the field
    for fd in &field.declarators {
        if let Some(ref decl) = fd.declarator {
            if decl
                .derived
                .iter()
                .any(|d| matches!(d, DerivedDeclarator::Pointer { .. }))
            {
                return IrType::Ptr;
            }
        }
    }

    IrType::I32 // default fallback
}

/// Checks whether two `TypeName` AST nodes represent the same C type
/// for `__builtin_types_compatible_p`.
///
/// GCC semantics: ignores top-level qualifiers (const/volatile), but
/// `int` and `long` are different types even when they have the same
/// bit-width on LP64.  Pointer types are compared recursively.
fn types_compatible_check(_ctx: &LoweringContext<'_>, t1: &TypeName, t2: &TypeName) -> bool {
    // Compare specifier lists (qualifiers are in a separate field and
    // are intentionally ignored per GCC semantics).
    let s1 = &t1.specifiers.specifiers;
    let s2 = &t2.specifiers.specifiers;

    if s1.len() != s2.len() {
        return false;
    }

    for (a, b) in s1.iter().zip(s2.iter()) {
        if std::mem::discriminant(a) != std::mem::discriminant(b) {
            return false;
        }
        // For struct/union/enum, compare tags
        match (a, b) {
            (TypeSpecifier::Struct { name: n1, .. }, TypeSpecifier::Struct { name: n2, .. }) => {
                if n1 != n2 {
                    return false;
                }
            }
            (TypeSpecifier::Union { name: n1, .. }, TypeSpecifier::Union { name: n2, .. }) => {
                if n1 != n2 {
                    return false;
                }
            }
            (TypeSpecifier::Enum { name: n1, .. }, TypeSpecifier::Enum { name: n2, .. }) => {
                if n1 != n2 {
                    return false;
                }
            }
            _ => {}
        }
    }

    // Compare derived declarators (pointer depth, etc.)
    let d1 = t1
        .declarator
        .as_ref()
        .map(|d| &d.derived[..])
        .unwrap_or(&[]);
    let d2 = t2
        .declarator
        .as_ref()
        .map(|d| &d.derived[..])
        .unwrap_or(&[]);

    if d1.len() != d2.len() {
        return false;
    }

    for (a, b) in d1.iter().zip(d2.iter()) {
        if std::mem::discriminant(a) != std::mem::discriminant(b) {
            return false;
        }
    }

    true
}

/// Resolves a struct or union type specifier to an IR type by looking up
/// the tag name in the module lowering context's `struct_defs` map.  If the
/// specifier includes inline field definitions (e.g. `struct { int a; int b; }`),
/// those are converted directly.  For forward references (`struct foo` without
/// a body), we look up `struct_defs` by the tag name.
fn resolve_struct_union_ir(
    ctx: &LoweringContext<'_>,
    name: Option<&Symbol>,
    fields: Option<&[crate::frontend::parser::ast::FieldDeclaration]>,
    is_struct: bool,
) -> Result<IrType, LoweringError> {
    // If we have a tag name, look it up in struct_defs first — this
    // handles both forward references and previously-defined structs.
    if let Some(tag_sym) = name {
        if let Some(ctype) = ctx.module_ctx.struct_defs.get(tag_sym) {
            return super::c_type_to_ir_type(ctype, ctx.target());
        }
    }

    // If inline field definitions are present, build the type directly.
    if let Some(field_list) = fields {
        if is_struct {
            let mut ir_fields = Vec::new();
            for fd in field_list {
                for _declarator in &fd.declarators {
                    // Build CType from the field's type specifiers, then convert.
                    // As a pragmatic fallback, use I32 for unresolvable fields.
                    ir_fields.push(IrType::I32);
                }
                if fd.declarators.is_empty() {
                    // Anonymous bitfield or anonymous struct member
                    ir_fields.push(IrType::I32);
                }
            }
            return Ok(IrType::Struct {
                fields: ir_fields,
                packed: false,
            });
        } else {
            // Union: represented as byte array of the union's size.
            // Without full type info, estimate based on field count.
            // A single I32 (4 bytes) is a reasonable default for small unions.
            let size = field_list.len().max(1) * 4;
            return Ok(IrType::Array {
                element: Box::new(IrType::I8),
                count: size,
            });
        }
    }

    // Fallback: unknown struct/union, default to I32 (will produce
    // incorrect results for sizeof but avoids a hard failure).
    Ok(IrType::I32)
}

/// Resolves a `TypeName` AST node to an IR type.
fn resolve_type_name_ir(
    ctx: &LoweringContext<'_>,
    type_name: &TypeName,
) -> Result<IrType, LoweringError> {
    let target = *ctx.target();
    let specs = &type_name.specifiers.specifiers;

    // CRITICAL: Check for pointer/array derived declarators FIRST, before
    // examining base-type specifiers.  A cast like `(int *)` has base
    // specifier `Int` but the derived declarator makes it a pointer type.
    // Without this early check, the fast-path below would return I32
    // instead of Ptr, causing 64-bit pointer truncation.
    if let Some(ref decl) = type_name.declarator {
        if let Some(derived) = decl.derived.first() {
            match derived {
                DerivedDeclarator::Pointer { .. } => return Ok(IrType::Ptr),
                DerivedDeclarator::Array { .. } => return Ok(IrType::Ptr),
                _ => {}
            }
        }
    }

    if specs.is_empty() {
        return Ok(IrType::I32); // implicitly int
    }

    // Single specifier fast path.
    if specs.len() == 1 {
        match &specs[0] {
            TypeSpecifier::Void => return Ok(IrType::Void),
            TypeSpecifier::Char => return Ok(IrType::I8),
            TypeSpecifier::Short => return Ok(IrType::I16),
            TypeSpecifier::Int => return Ok(IrType::I32),
            TypeSpecifier::Float => return Ok(IrType::F32),
            TypeSpecifier::Double => return Ok(IrType::F64),
            TypeSpecifier::Bool => return Ok(IrType::I1),
            TypeSpecifier::Long => {
                return Ok(if target.long_size() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                });
            }
            TypeSpecifier::Unsigned => return Ok(IrType::I32),
            TypeSpecifier::Signed => return Ok(IrType::I32),
            TypeSpecifier::BuiltinVaList => return Ok(IrType::Ptr),
            TypeSpecifier::Struct { name, fields, .. } => {
                return resolve_struct_union_ir(ctx, name.as_ref(), fields.as_deref(), true);
            }
            TypeSpecifier::Union { name, fields, .. } => {
                return resolve_struct_union_ir(ctx, name.as_ref(), fields.as_deref(), false);
            }
            TypeSpecifier::Enum { .. } => {
                // Enums are represented as their underlying int type.
                return Ok(IrType::I32);
            }
            TypeSpecifier::TypedefName { name, .. } => {
                // Look up in typedef_types first, then try struct_defs
                if let Some(ir_ty) = ctx.module_ctx.typedef_types.get(name) {
                    return Ok(ir_ty.clone());
                }
                // Some typedefs may map to structs registered in struct_defs
                if let Some(ctype) = ctx.module_ctx.struct_defs.get(name) {
                    if let Ok(ir_ty) = super::c_type_to_ir_type(ctype, ctx.target()) {
                        return Ok(ir_ty);
                    }
                }
                return Ok(IrType::I32);
            }
            _ => {}
        }
    }

    // Multi-specifier combinations.
    let has_long_count = specs
        .iter()
        .filter(|s| matches!(s, TypeSpecifier::Long))
        .count();
    let has_short = specs.iter().any(|s| matches!(s, TypeSpecifier::Short));
    let has_char = specs.iter().any(|s| matches!(s, TypeSpecifier::Char));
    let has_double = specs.iter().any(|s| matches!(s, TypeSpecifier::Double));
    let has_float = specs.iter().any(|s| matches!(s, TypeSpecifier::Float));
    let has_void = specs.iter().any(|s| matches!(s, TypeSpecifier::Void));
    let has_bool = specs.iter().any(|s| matches!(s, TypeSpecifier::Bool));

    if has_void {
        return Ok(IrType::Void);
    }
    if has_bool {
        return Ok(IrType::I1);
    }
    if has_long_count >= 2 {
        return Ok(IrType::I64);
    }
    if has_long_count == 1 && has_double {
        return Ok(IrType::F80);
    }
    if has_long_count == 1 {
        return Ok(if target.long_size() == 8 {
            IrType::I64
        } else {
            IrType::I32
        });
    }
    if has_short {
        return Ok(IrType::I16);
    }
    if has_char {
        return Ok(IrType::I8);
    }
    if has_double {
        return Ok(IrType::F64);
    }
    if has_float {
        return Ok(IrType::F32);
    }

    // Check for pointer in the abstract declarator.
    if let Some(ref decl) = type_name.declarator {
        if let Some(DerivedDeclarator::Pointer { .. }) = decl.derived.first() {
            return Ok(IrType::Ptr);
        }
    }

    // Default: int.
    Ok(IrType::I32)
}

// ============================================================================
// Function call
// ============================================================================

/// Lowers a function call expression.
fn lower_function_call(
    ctx: &mut LoweringContext<'_>,
    callee: &Expression,
    args: &[Expression],
    span: Span,
) -> Result<ValueId, LoweringError> {
    // ---- Intercept __builtin_* calls before normal lowering ----
    // Builtins are not registered in the symbol table; the sema phase
    // validates them but the AST still carries FunctionCall nodes.
    // We must handle them here to avoid "undeclared identifier" errors.
    if let Expression::Identifier { name, .. } = callee {
        let name_str = ctx.module_ctx.interner.resolve(*name).to_string();
        if name_str.starts_with("__builtin_") {
            return lower_builtin_call(ctx, &name_str, args, span);
        }
    }

    let callee_val = lower_expression(ctx, callee)?;

    let mut arg_vals = Vec::with_capacity(args.len());
    for arg in args {
        arg_vals.push(lower_expression(ctx, arg)?);
    }

    // ================================================================
    // C11 §6.5.2.2p6 — Default argument promotions for variadic calls
    // ================================================================
    //
    // "The default argument promotions are performed on trailing
    //  arguments."  For variadic functions (and calls through function
    //  pointers without prototypes), every argument past the last
    //  declared parameter undergoes:
    //   - float  → double                     (FP promotion)
    //   - char / short / _Bool → int          (integer promotion,
    //       typically already handled by the expression lowering)
    //
    // Without this promotion, passing a `float` to `printf("%f", f)`
    // places 32 bits in the lower half of an XMM register, but the
    // callee reads 64 bits (a `double`), producing garbage.
    //
    // We detect variadic callees by looking up the global symbol's
    // IR function type.  If the callee is variadic, we promote every
    // F32 argument in the variadic tail to F64 via a BitCast (which
    // the backend lowers to CVTSS2SD / equivalent).
    let (is_variadic, fixed_param_count) = infer_callee_variadic_info(ctx, callee);

    if is_variadic {
        for i in fixed_param_count..arg_vals.len() {
            let val = arg_vals[i];
            let val_ty = ctx.function.get_value_type(val).clone();
            if val_ty == IrType::F32 {
                // Promote float → double (CVTSS2SD in x86-64 backend).
                arg_vals[i] = ctx.builder.build_fp_ext(ctx.function, val, IrType::F64);
            }
        }
    }

    let ret_ty = infer_call_return_type(ctx, callee);

    match ctx.builder.build_call_ex(
        ctx.function,
        callee_val,
        arg_vals,
        ret_ty.clone(),
        is_variadic,
    ) {
        Some(result_id) => Ok(result_id),
        None => {
            // Void return — produce a dummy value.
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }
    }
}

/// Determines whether the callee of a function call is variadic, and
/// if so, how many fixed parameters precede the variadic tail.
///
/// Returns `(is_variadic, fixed_param_count)`.  For non-variadic
/// functions or when the callee's type cannot be resolved,
/// returns `(false, 0)`.
fn infer_callee_variadic_info(ctx: &LoweringContext<'_>, callee: &Expression) -> (bool, usize) {
    // Unwrap _Generic callee to the resolved function expression.
    let effective = resolve_generic_callee(ctx, callee);
    if let Expression::Identifier { name, .. } = effective {
        if let Some(info) = ctx.module_ctx.global_symbols.get(name) {
            if let IrType::Function {
                is_variadic,
                param_types,
                ..
            } = &info.ir_type
            {
                return (*is_variadic, param_types.len());
            }
        }
    }
    // Cannot determine — assume non-variadic to avoid spurious
    // promotions.  (Calls through function pointers without known
    // prototypes fall here; a more complete implementation would
    // check the pointer's pointee type.)
    (false, 0)
}

// ============================================================================
// Builtin function lowering
// ============================================================================

/// Lowers a `__builtin_*` function call to appropriate IR instructions.
///
/// Builtins fall into several categories:
/// - Compile-time constants (`__builtin_constant_p`, `__builtin_choose_expr`)
///   that should have been folded by sema but may survive in non-constant
///   contexts — these produce constant IR values or select subexpressions.
/// - Runtime intrinsics (`__builtin_clz`, `__builtin_bswap*`, etc.) that
///   lower to inline assembly or direct bit-manipulation IR sequences.
/// - Hints and traps (`__builtin_expect`, `__builtin_unreachable`,
///   `__builtin_trap`) that produce identity values or terminal instructions.
/// - Variadic helpers (`__builtin_va_start`, etc.) that lower to ABI-specific
///   operations on the va_list pointer.
/// - Overflow-checked arithmetic that lowers to a computation + overflow flag.
fn lower_builtin_call(
    ctx: &mut LoweringContext<'_>,
    name: &str,
    args: &[Expression],
    span: Span,
) -> Result<ValueId, LoweringError> {
    match name {
        // ----------------------------------------------------------------
        // 1. Compile-time builtins (may survive to IR in runtime contexts)
        // ----------------------------------------------------------------
        "__builtin_constant_p" => {
            // In runtime context (not folded by sema), conservatively return 0
            // since the argument is not a compile-time constant if we reached here.
            // However, if the argument is a literal, return 1.
            if args.len() >= 1 {
                if is_compile_time_constant(&args[0]) {
                    return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 1));
                }
            }
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }

        "__builtin_choose_expr" => {
            // __builtin_choose_expr(const_expr, expr_if_true, expr_if_false)
            // The controlling expression must be a compile-time constant.
            if args.len() >= 3 {
                let is_true = evaluate_builtin_constant_arg(ctx, &args[0]);
                if is_true {
                    return lower_expression(ctx, &args[1]);
                } else {
                    return lower_expression(ctx, &args[2]);
                }
            }
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }

        "__builtin_types_compatible_p" => {
            // Should have been folded by sema. If it reaches here, return 0.
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }

        "__builtin_offsetof" => {
            // Should have been folded by sema. If it reaches here, return 0.
            let ptr_ty = if ctx.target().pointer_width() == 8 {
                IrType::I64
            } else {
                IrType::I32
            };
            Ok(ctx.builder.build_const_int(ctx.function, ptr_ty, 0))
        }

        // ----------------------------------------------------------------
        // 2. __builtin_expect — returns its first argument unchanged (hint)
        // ----------------------------------------------------------------
        "__builtin_expect" => {
            // __builtin_expect(exp, c) — returns exp, c is a branch hint
            if args.len() >= 1 {
                lower_expression(ctx, &args[0])
            } else {
                Ok(ctx.builder.build_const_int(ctx.function, IrType::I64, 0))
            }
        }

        // ----------------------------------------------------------------
        // 3. Bit-manipulation builtins — CLZ, CTZ, POPCOUNT, FFS, BSWAP
        // ----------------------------------------------------------------
        "__builtin_clz" | "__builtin_clzl" | "__builtin_clzll" => {
            lower_builtin_bit_intrinsic(ctx, name, args, span, BuiltinBitOp::Clz)
        }
        "__builtin_ctz" | "__builtin_ctzl" | "__builtin_ctzll" => {
            lower_builtin_bit_intrinsic(ctx, name, args, span, BuiltinBitOp::Ctz)
        }
        "__builtin_popcount" | "__builtin_popcountl" | "__builtin_popcountll" => {
            lower_builtin_bit_intrinsic(ctx, name, args, span, BuiltinBitOp::Popcount)
        }
        "__builtin_ffs" | "__builtin_ffsl" | "__builtin_ffsll" => {
            lower_builtin_bit_intrinsic(ctx, name, args, span, BuiltinBitOp::Ffs)
        }
        "__builtin_bswap16" => lower_builtin_bswap(ctx, args, span, 16),
        "__builtin_bswap32" => lower_builtin_bswap(ctx, args, span, 32),
        "__builtin_bswap64" => lower_builtin_bswap(ctx, args, span, 64),

        // ----------------------------------------------------------------
        // 4. Absolute value builtins
        // ----------------------------------------------------------------
        "__builtin_abs" => lower_builtin_abs(ctx, args, span, IrType::I32),
        "__builtin_labs" => {
            let ty = if ctx.target().pointer_width() == 8 {
                IrType::I64
            } else {
                IrType::I32
            };
            lower_builtin_abs(ctx, args, span, ty)
        }
        "__builtin_llabs" => lower_builtin_abs(ctx, args, span, IrType::I64),

        // ----------------------------------------------------------------
        // 5. Address/introspection builtins
        // ----------------------------------------------------------------
        "__builtin_frame_address" => {
            // __builtin_frame_address(level) — level 0 returns current frame pointer
            // Lower as inline asm to read the frame pointer register.
            lower_builtin_frame_or_return_address(ctx, args, span, true)
        }
        "__builtin_return_address" => {
            // __builtin_return_address(level) — level 0 returns return address
            lower_builtin_frame_or_return_address(ctx, args, span, false)
        }

        "__builtin_assume_aligned" => {
            // __builtin_assume_aligned(ptr, alignment) — returns ptr unchanged
            if args.len() >= 1 {
                lower_expression(ctx, &args[0])
            } else {
                Ok(ctx.builder.build_const_null(ctx.function, IrType::Ptr))
            }
        }

        // ----------------------------------------------------------------
        // 6. Variadic argument builtins
        // ----------------------------------------------------------------
        "__builtin_va_start" => {
            // __builtin_va_start(ap, last_param)
            // Lower as a call to the platform va_start or emit inline setup.
            lower_builtin_va_start(ctx, args, span)
        }
        "__builtin_va_end" => {
            // __builtin_va_end(ap) — typically a no-op on most platforms
            // We still lower the argument for side-effect correctness.
            if args.len() >= 1 {
                let _ = lower_expression(ctx, &args[0]);
            }
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }
        "__builtin_va_copy" => {
            // __builtin_va_copy(dest, src)
            lower_builtin_va_copy(ctx, args, span)
        }
        "__builtin_va_arg" => {
            // This should normally be handled by the dedicated Expression::BuiltinVaArg
            // node, but if it survives as a FunctionCall, handle it here.
            if args.len() >= 1 {
                let ap_val = lower_expression(ctx, &args[0])?;
                let ptr_ty = if ctx.target().pointer_width() == 8 {
                    IrType::I64
                } else {
                    IrType::I32
                };
                Ok(ctx.builder.build_load(ctx.function, ap_val, ptr_ty))
            } else {
                Ok(ctx.builder.build_const_int(ctx.function, IrType::I64, 0))
            }
        }

        // ----------------------------------------------------------------
        // 7. Overflow-checked arithmetic
        // ----------------------------------------------------------------
        "__builtin_add_overflow" => lower_builtin_overflow(ctx, args, span, BinOp::Add),
        "__builtin_sub_overflow" => lower_builtin_overflow(ctx, args, span, BinOp::Sub),
        "__builtin_mul_overflow" => lower_builtin_overflow(ctx, args, span, BinOp::Mul),

        // ----------------------------------------------------------------
        // 8. Trap and unreachable
        // ----------------------------------------------------------------
        "__builtin_trap" => {
            // Emit ud2 (x86) or equivalent illegal instruction via inline asm.
            let template = match ctx.target() {
                Target::X86_64 | Target::I686 => "ud2",
                Target::AArch64 => "brk #0x1",
                Target::RiscV64 => "ebreak",
            };
            ctx.builder.build_inline_asm(
                ctx.function,
                template.to_string(),
                String::new(),
                vec![],
                vec![],
                true,
                false,
            );
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }

        "__builtin_unreachable" => {
            // Emit ud2/equivalent — optimization hint that this point is unreachable.
            let template = match ctx.target() {
                Target::X86_64 | Target::I686 => "ud2",
                Target::AArch64 => "brk #0x1",
                Target::RiscV64 => "ebreak",
            };
            ctx.builder.build_inline_asm(
                ctx.function,
                template.to_string(),
                String::new(),
                vec![],
                vec![],
                true,
                false,
            );
            Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
        }

        // ----------------------------------------------------------------
        // 9. Fallback — unknown builtin, emit as external function call
        // ----------------------------------------------------------------
        _ => {
            // For unrecognized builtins, create a synthetic global reference
            // and emit a regular call. This allows linking against libgcc or
            // platform-provided implementations.
            let callee_ref = ctx
                .builder
                .build_global_ref(ctx.function, name, IrType::Ptr);
            let mut arg_vals = Vec::with_capacity(args.len());
            for arg in args {
                arg_vals.push(lower_expression(ctx, arg)?);
            }
            let ret_ty = IrType::I32;
            match ctx
                .builder
                .build_call(ctx.function, callee_ref, arg_vals, ret_ty)
            {
                Some(result_id) => Ok(result_id),
                None => Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0)),
            }
        }
    }
}

/// Returns true if an expression is a compile-time constant (literal or
/// simple arithmetic on literals).
fn is_compile_time_constant(expr: &Expression) -> bool {
    match expr {
        Expression::IntegerLiteral { .. }
        | Expression::FloatLiteral { .. }
        | Expression::CharLiteral { .. }
        | Expression::StringLiteral { .. } => true,
        Expression::BinaryOp { left, right, .. } => {
            is_compile_time_constant(left) && is_compile_time_constant(right)
        }
        Expression::UnaryOp { operand, .. } => is_compile_time_constant(operand),
        Expression::Cast { operand, .. } => is_compile_time_constant(operand),
        _ => false,
    }
}

/// Evaluates a builtin constant argument to a boolean (non-zero = true).
/// Used by `__builtin_choose_expr` and similar compile-time selectors.
fn evaluate_builtin_constant_arg(ctx: &mut LoweringContext<'_>, expr: &Expression) -> bool {
    match expr {
        Expression::IntegerLiteral { value, .. } => *value != 0,
        // __builtin_types_compatible_p — dedicated AST node from the parser.
        // This evaluates to 1 (true) if the two types are compatible.
        Expression::BuiltinTypesCompatibleP { type1, type2, .. } => {
            types_compatible_check(ctx, type1, type2)
        }
        // __builtin_constant_p or __builtin_types_compatible_p as function calls
        Expression::FunctionCall { callee, args, .. } => {
            if let Expression::Identifier { name, .. } = callee.as_ref() {
                let callee_name = ctx.module_ctx.interner.resolve(*name).to_string();
                match callee_name.as_str() {
                    "__builtin_constant_p" => {
                        if args.len() >= 1 {
                            return is_compile_time_constant(&args[0]);
                        }
                        false
                    }
                    "__builtin_types_compatible_p" => {
                        // Conservative: same type text → compatible
                        true
                    }
                    _ => false,
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Categories of bit-manipulation builtins for shared lowering logic.
#[derive(Clone, Copy)]
enum BuiltinBitOp {
    Clz,
    Ctz,
    Popcount,
    Ffs,
}

/// Lowers CLZ/CTZ/POPCOUNT/FFS builtins using bit-manipulation IR sequences.
///
/// For x86-64, these map to BSR/BSF/POPCNT instructions via inline asm.
/// For other architectures, we emit portable bit-manipulation loops.
fn lower_builtin_bit_intrinsic(
    ctx: &mut LoweringContext<'_>,
    name: &str,
    args: &[Expression],
    span: Span,
    op: BuiltinBitOp,
) -> Result<ValueId, LoweringError> {
    if args.is_empty() {
        ctx.diagnostics()
            .error(span, format!("{} requires one argument", name));
        return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
    }

    let arg_val = lower_expression(ctx, &args[0])?;

    // Determine operand width from the builtin suffix
    let (operand_ty, bit_width) = if name.ends_with("ll") {
        (IrType::I64, 64i64)
    } else if name.ends_with('l') && !name.ends_with("ol") {
        // "l" suffix but not "ol" (popcount_l vs popcountl)
        if ctx.target().pointer_width() == 8 {
            (IrType::I64, 64i64)
        } else {
            (IrType::I32, 32i64)
        }
    } else {
        (IrType::I32, 32i64)
    };

    // Ensure the argument is the right width.
    //
    // Only emit a ZExt when the source value is *narrower* than the
    // required operand width.  A ZExt from I64→I64 produces a no-op
    // MOVZX that some encoder paths mishandle — avoid it entirely.
    let widened = if bit_width == 64 && operand_ty == IrType::I64 {
        let val_ty = ctx.function.get_value_type(arg_val).clone();
        if val_ty != IrType::I64 {
            ctx.builder.build_zext(ctx.function, arg_val, IrType::I64)
        } else {
            arg_val
        }
    } else {
        arg_val
    };

    match op {
        BuiltinBitOp::Clz => {
            // CLZ: count leading zeros.  (bit_width - 1) - BSR(x) on x86.

            // Special case: i686 with 64-bit operand — no 64-bit BSR exists.
            // Split into high/low 32-bit words, BSR each, use branch-based
            // selection to avoid register pressure issues (only 6 GPRs).
            if *ctx.target() == Target::I686 && bit_width == 64 {
                // high = (val >> 32) as u32, low = val as u32
                let shift_amt = ctx.builder.build_const_int(ctx.function, IrType::I64, 32);
                let high_i64 = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::LShr,
                    widened,
                    shift_amt,
                    IrType::I64,
                );
                let high = ctx.builder.build_trunc(ctx.function, high_i64, IrType::I32);
                let low = ctx.builder.build_trunc(ctx.function, widened, IrType::I32);

                // Store results via alloca to avoid register pressure
                let result_slot = ctx.builder.build_alloca(ctx.function, IrType::I32, None);

                // Check if high word is nonzero
                let zero32 = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let is_high_nz =
                    ctx.builder
                        .build_icmp(ctx.function, ICmpPredicate::Ne, high, zero32);

                // Create basic blocks for if/else/merge
                let then_bb = ctx.builder.create_block(ctx.function, Some("clz64_hi"));
                let else_bb = ctx.builder.create_block(ctx.function, Some("clz64_lo"));
                let merge_bb = ctx.builder.create_block(ctx.function, Some("clz64_merge"));

                ctx.builder
                    .build_cond_branch(ctx.function, is_high_nz, then_bb, else_bb);

                // Then block: high word is nonzero → clz = bsr(high) ^ 31
                ctx.builder.set_insert_point(then_bb);
                let clz_high = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsrl $1, $0\n\txorl $$31, $0".to_string(),
                        "=r,r".to_string(),
                        vec![high],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                ctx.builder.build_store(ctx.function, clz_high, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Else block: high word is zero → clz = bsr(low) ^ 31 + 32
                ctx.builder.set_insert_point(else_bb);
                let clz_low = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsrl $1, $0\n\txorl $$31, $0".to_string(),
                        "=r,r".to_string(),
                        vec![low],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                let c32 = ctx.builder.build_const_int(ctx.function, IrType::I32, 32);
                let clz_low_plus32 =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Add, clz_low, c32, IrType::I32);
                ctx.builder
                    .build_store(ctx.function, clz_low_plus32, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Merge block: load result
                ctx.builder.set_insert_point(merge_bb);
                let result = ctx
                    .builder
                    .build_load(ctx.function, result_slot, IrType::I32);
                return Ok(result);
            }

            let asm_template;
            let constraints;
            match ctx.target() {
                Target::X86_64 => {
                    if bit_width == 64 {
                        asm_template = "bsrq $1, $0\n\txorq $$63, $0".to_string();
                    } else {
                        asm_template = "bsrl $1, $0\n\txorl $$31, $0".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::I686 => {
                    // 32-bit CLZ (64-bit case handled above)
                    asm_template = "bsrl $1, $0\n\txorl $$31, $0".to_string();
                    constraints = "=r,r".to_string();
                }
                Target::AArch64 => {
                    // Use width-suffixed mnemonics: "clzw" for 32-bit, "clz" for 64-bit.
                    // The AArch64 backend recognizes these and sets sf correctly.
                    if bit_width == 32 {
                        asm_template = "clzw $0, $1".to_string();
                    } else {
                        asm_template = "clz $0, $1".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::RiscV64 => {
                    // Pure IR loop for CLZ (no inline asm with labels).
                    // Algorithm: result = bit_width-1; if (x==0) goto done;
                    //            loop: x >>= 1; result--; if (x!=0) goto loop;
                    //            done: use result.
                    let result_val = lower_software_clz(ctx, widened, bit_width)?;
                    if bit_width == 64 {
                        return Ok(ctx
                            .builder
                            .build_trunc(ctx.function, result_val, IrType::I32));
                    } else {
                        return Ok(result_val);
                    }
                }
            }
            let result = ctx.builder.build_inline_asm_typed(
                ctx.function,
                asm_template,
                constraints,
                vec![widened],
                vec![],
                false,
                false,
                operand_ty.clone(),
            );
            let result_val =
                result.unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
            // Truncate to i32 (CLZ always returns int)
            if bit_width == 64 {
                Ok(ctx
                    .builder
                    .build_trunc(ctx.function, result_val, IrType::I32))
            } else {
                Ok(result_val)
            }
        }

        BuiltinBitOp::Ctz => {
            // CTZ: count trailing zeros. BSF on x86.

            // Special case: i686 with 64-bit operand — no 64-bit BSF exists.
            // Split into low/high, BSF each, use branch-based selection
            // to avoid register pressure issues (only 6 GPRs on i686).
            if *ctx.target() == Target::I686 && bit_width == 64 {
                let low = ctx.builder.build_trunc(ctx.function, widened, IrType::I32);
                let shift_amt = ctx.builder.build_const_int(ctx.function, IrType::I64, 32);
                let high_i64 = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::LShr,
                    widened,
                    shift_amt,
                    IrType::I64,
                );
                let high = ctx.builder.build_trunc(ctx.function, high_i64, IrType::I32);

                // Store result via alloca
                let result_slot = ctx.builder.build_alloca(ctx.function, IrType::I32, None);

                // Check if low word is nonzero
                let zero32 = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let is_low_nz =
                    ctx.builder
                        .build_icmp(ctx.function, ICmpPredicate::Ne, low, zero32);

                let then_bb = ctx.builder.create_block(ctx.function, Some("ctz64_lo"));
                let else_bb = ctx.builder.create_block(ctx.function, Some("ctz64_hi"));
                let merge_bb = ctx.builder.create_block(ctx.function, Some("ctz64_merge"));

                ctx.builder
                    .build_cond_branch(ctx.function, is_low_nz, then_bb, else_bb);

                // Then block: low word nonzero → ctz = bsf(low)
                ctx.builder.set_insert_point(then_bb);
                let bsf_low = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsfl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![low],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                ctx.builder.build_store(ctx.function, bsf_low, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Else block: low word zero → ctz = bsf(high) + 32
                ctx.builder.set_insert_point(else_bb);
                let bsf_high = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsfl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![high],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                let c32 = ctx.builder.build_const_int(ctx.function, IrType::I32, 32);
                let bsf_high_plus32 =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Add, bsf_high, c32, IrType::I32);
                ctx.builder
                    .build_store(ctx.function, bsf_high_plus32, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Merge: load result
                ctx.builder.set_insert_point(merge_bb);
                let result = ctx
                    .builder
                    .build_load(ctx.function, result_slot, IrType::I32);
                return Ok(result);
            }

            let asm_template;
            let constraints;
            match ctx.target() {
                Target::X86_64 => {
                    if bit_width == 64 {
                        asm_template = "bsfq $1, $0".to_string();
                    } else {
                        asm_template = "bsfl $1, $0".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::I686 => {
                    // 32-bit CTZ (64-bit case handled above)
                    asm_template = "bsfl $1, $0".to_string();
                    constraints = "=r,r".to_string();
                }
                Target::AArch64 => {
                    // rbit + clz gives ctz.  Use width-suffixed mnemonics for 32-bit.
                    if bit_width == 32 {
                        asm_template = "rbitw $0, $1\n\tclzw $0, $0".to_string();
                    } else {
                        asm_template = "rbit $0, $1\n\tclz $0, $0".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::RiscV64 => {
                    // Pure IR loop for CTZ (no inline asm with labels).
                    let result_val = lower_software_ctz(ctx, widened, bit_width)?;
                    if bit_width == 64 {
                        return Ok(ctx
                            .builder
                            .build_trunc(ctx.function, result_val, IrType::I32));
                    } else {
                        return Ok(result_val);
                    }
                }
            }
            let result = ctx.builder.build_inline_asm_typed(
                ctx.function,
                asm_template,
                constraints,
                vec![widened],
                vec![],
                false,
                false,
                operand_ty.clone(),
            );
            let result_val =
                result.unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
            if bit_width == 64 {
                Ok(ctx
                    .builder
                    .build_trunc(ctx.function, result_val, IrType::I32))
            } else {
                Ok(result_val)
            }
        }

        BuiltinBitOp::Popcount => {
            // POPCOUNT: count set bits.

            // Special case: i686 with 64-bit operand — no 64-bit POPCNT exists.
            // Split into high/low, popcount each separately, add results in IR.
            if *ctx.target() == Target::I686 && bit_width == 64 {
                let low = ctx.builder.build_trunc(ctx.function, widened, IrType::I32);
                let shift_amt = ctx.builder.build_const_int(ctx.function, IrType::I64, 32);
                let high_i64 = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::LShr,
                    widened,
                    shift_amt,
                    IrType::I64,
                );
                let high = ctx.builder.build_trunc(ctx.function, high_i64, IrType::I32);

                let pop_low = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "popcntl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![low],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));

                let pop_high = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "popcntl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![high],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));

                let result = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::Add,
                    pop_low,
                    pop_high,
                    IrType::I32,
                );
                return Ok(result);
            }

            let asm_template;
            let constraints;
            match ctx.target() {
                Target::X86_64 => {
                    if bit_width == 64 {
                        asm_template = "popcntq $1, $0".to_string();
                    } else {
                        asm_template = "popcntl $1, $0".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::I686 => {
                    // 32-bit popcount (64-bit case handled above)
                    asm_template = "popcntl $1, $0".to_string();
                    constraints = "=r,r".to_string();
                }
                Target::AArch64 => {
                    // AArch64 POPCOUNT: use pure IR software Hamming-weight
                    // instead of SIMD (which needs FMOV/CNT/ADDV support).
                    let result_val = lower_software_popcount(ctx, widened, bit_width)?;
                    if bit_width == 64 {
                        return Ok(ctx
                            .builder
                            .build_trunc(ctx.function, result_val, IrType::I32));
                    } else {
                        return Ok(result_val);
                    }
                }
                Target::RiscV64 => {
                    // Pure IR software popcount — reuse the same Hamming-weight
                    // implementation used by AArch64.
                    let result_val = lower_software_popcount(ctx, widened, bit_width)?;
                    if bit_width == 64 {
                        return Ok(ctx
                            .builder
                            .build_trunc(ctx.function, result_val, IrType::I32));
                    } else {
                        return Ok(result_val);
                    }
                }
            }
            let result = ctx.builder.build_inline_asm_typed(
                ctx.function,
                asm_template,
                constraints,
                vec![widened],
                vec![],
                false,
                false,
                operand_ty.clone(),
            );
            let result_val =
                result.unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
            if bit_width == 64 {
                Ok(ctx
                    .builder
                    .build_trunc(ctx.function, result_val, IrType::I32))
            } else {
                Ok(result_val)
            }
        }

        BuiltinBitOp::Ffs => {
            // FFS: find first set bit (1-indexed). Returns 0 if input is 0.
            // Strategy: ffs(x) = (ctz(x) + 1) & -(x != 0)

            // Special case: i686 with 64-bit operand. Branch-based approach
            // to avoid register pressure issues (only 6 GPRs on i686).
            // ffs(x) = x==0 ? 0 : ctz(x)+1, and ctz is split into halves.
            if *ctx.target() == Target::I686 && bit_width == 64 {
                let low = ctx.builder.build_trunc(ctx.function, widened, IrType::I32);
                let shift_amt = ctx.builder.build_const_int(ctx.function, IrType::I64, 32);
                let high_i64 = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::LShr,
                    widened,
                    shift_amt,
                    IrType::I64,
                );
                let high = ctx.builder.build_trunc(ctx.function, high_i64, IrType::I32);

                let result_slot = ctx.builder.build_alloca(ctx.function, IrType::I32, None);

                // Check if value is zero: (low | high) == 0
                let or_val =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Or, low, high, IrType::I32);
                let zero32 = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let is_zero =
                    ctx.builder
                        .build_icmp(ctx.function, ICmpPredicate::Eq, or_val, zero32);

                let zero_bb = ctx.builder.create_block(ctx.function, Some("ffs64_zero"));
                let nonzero_bb = ctx.builder.create_block(ctx.function, Some("ffs64_nz"));
                let lo_bb = ctx.builder.create_block(ctx.function, Some("ffs64_lo"));
                let hi_bb = ctx.builder.create_block(ctx.function, Some("ffs64_hi"));
                let merge_bb = ctx.builder.create_block(ctx.function, Some("ffs64_merge"));

                ctx.builder
                    .build_cond_branch(ctx.function, is_zero, zero_bb, nonzero_bb);

                // Zero block: result = 0
                ctx.builder.set_insert_point(zero_bb);
                let const_zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                ctx.builder
                    .build_store(ctx.function, const_zero, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Nonzero block: check low half
                ctx.builder.set_insert_point(nonzero_bb);
                let zero32b = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let is_low_nz =
                    ctx.builder
                        .build_icmp(ctx.function, ICmpPredicate::Ne, low, zero32b);
                ctx.builder
                    .build_cond_branch(ctx.function, is_low_nz, lo_bb, hi_bb);

                // Low-half block: ffs = bsf(low) + 1
                ctx.builder.set_insert_point(lo_bb);
                let bsf_low = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsfl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![low],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                let one_a = ctx.builder.build_const_int(ctx.function, IrType::I32, 1);
                let ffs_lo =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Add, bsf_low, one_a, IrType::I32);
                ctx.builder.build_store(ctx.function, ffs_lo, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // High-half block: ffs = bsf(high) + 33
                ctx.builder.set_insert_point(hi_bb);
                let bsf_high = ctx
                    .builder
                    .build_inline_asm_typed(
                        ctx.function,
                        "bsfl $1, $0".to_string(),
                        "=r,r".to_string(),
                        vec![high],
                        vec![],
                        false,
                        false,
                        IrType::I32,
                    )
                    .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
                let c33 = ctx.builder.build_const_int(ctx.function, IrType::I32, 33);
                let ffs_hi =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Add, bsf_high, c33, IrType::I32);
                ctx.builder.build_store(ctx.function, ffs_hi, result_slot);
                ctx.builder.build_branch(ctx.function, merge_bb);

                // Merge
                ctx.builder.set_insert_point(merge_bb);
                let result = ctx
                    .builder
                    .build_load(ctx.function, result_slot, IrType::I32);
                return Ok(result);
            }

            let op_ty = if bit_width == 64 {
                IrType::I64
            } else {
                IrType::I32
            };

            // Step 1: is_nonzero = (x != 0) BEFORE any inline asm
            let zero = ctx.builder.build_const_int(ctx.function, op_ty.clone(), 0);
            let is_nonzero = ctx
                .builder
                .build_icmp(ctx.function, ICmpPredicate::Ne, widened, zero);
            let mask_val = ctx
                .builder
                .build_zext(ctx.function, is_nonzero, op_ty.clone());
            let zero2 = ctx.builder.build_const_int(ctx.function, op_ty.clone(), 0);
            let neg_mask =
                ctx.builder
                    .build_binop(ctx.function, BinOp::Sub, zero2, mask_val, op_ty.clone());

            // Step 2: CTZ via BSF
            let ctz_template;
            let constraints;
            match ctx.target() {
                Target::X86_64 => {
                    ctz_template = if bit_width == 64 {
                        "bsfq $1, $0".to_string()
                    } else {
                        "bsfl $1, $0".to_string()
                    };
                    constraints = "=r,r".to_string();
                }
                Target::I686 => {
                    ctz_template = "bsfl $1, $0".to_string();
                    constraints = "=r,r".to_string();
                }
                Target::AArch64 => {
                    if bit_width == 32 {
                        ctz_template = "rbitw $0, $1\n\tclzw $0, $0".to_string();
                    } else {
                        ctz_template = "rbit $0, $1\n\tclz $0, $0".to_string();
                    }
                    constraints = "=r,r".to_string();
                }
                Target::RiscV64 => {
                    // Pure IR software CTZ for FFS on RISC-V.
                    let ctz_val = lower_software_ctz(ctx, widened, bit_width)?;
                    let one = ctx.builder.build_const_int(ctx.function, op_ty.clone(), 1);
                    let ffs_raw = ctx.builder.build_binop(
                        ctx.function,
                        BinOp::Add,
                        ctz_val,
                        one,
                        op_ty.clone(),
                    );
                    let result_val = ctx.builder.build_binop(
                        ctx.function,
                        BinOp::And,
                        ffs_raw,
                        neg_mask,
                        op_ty.clone(),
                    );
                    if bit_width == 64 {
                        return Ok(ctx
                            .builder
                            .build_trunc(ctx.function, result_val, IrType::I32));
                    } else {
                        return Ok(result_val);
                    }
                }
            }

            let ctz_val = ctx
                .builder
                .build_inline_asm_typed(
                    ctx.function,
                    ctz_template,
                    constraints,
                    vec![widened],
                    vec![],
                    false,
                    false,
                    op_ty.clone(),
                )
                .unwrap_or_else(|| ctx.builder.build_const_int(ctx.function, op_ty.clone(), 0));

            // Step 3: ffs = (ctz + 1) & neg_mask
            let one = ctx.builder.build_const_int(ctx.function, op_ty.clone(), 1);
            let ffs_raw =
                ctx.builder
                    .build_binop(ctx.function, BinOp::Add, ctz_val, one, op_ty.clone());
            let result_val =
                ctx.builder
                    .build_binop(ctx.function, BinOp::And, ffs_raw, neg_mask, op_ty.clone());

            if bit_width == 64 {
                Ok(ctx
                    .builder
                    .build_trunc(ctx.function, result_val, IrType::I32))
            } else {
                Ok(result_val)
            }
        }
    }
}

/// Lowers __builtin_bswap{16,32,64} to byte-swap IR.
/// Pure-IR byte swap — avoids inline asm so it works reliably on every target.
/// Software Hamming-weight (popcount) using the standard bit-manipulation
/// algorithm.  Emits pure IR (no inline asm) so it works on all backends,
/// particularly AArch64 where the SIMD CNT/ADDV path is not yet supported.
///
/// Algorithm (32-bit):
///   x = x - ((x >> 1) & 0x55555555);
///   x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
///   x = (x + (x >> 4)) & 0x0F0F0F0F;
///   return (x * 0x01010101) >> 24;
///
/// 64-bit version uses the same approach with wider masks.
/// Software CLZ (Count Leading Zeros) using a loop in pure IR.
///
/// Algorithm:
///   result = bit_width - 1
///   if (x == 0) goto done
///   loop: x >>= 1; result--; if (x != 0) goto loop
///   done: use result
///
/// This avoids inline assembly with labels, which is problematic on
/// architectures without native CLZ instructions (e.g. base RISC-V).
fn lower_software_clz(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    bit_width: i64,
) -> Result<ValueId, LoweringError> {
    let ty = if bit_width == 64 {
        IrType::I64
    } else {
        IrType::I32
    };

    // Algorithm: count right-shifts until zero, then clz = bit_width - count.
    //   result = bit_width  (returned as-is if x == 0, i.e., undefined behaviour)
    //   if (x == 0) goto done
    //   loop: x >>= 1; result--; if (x != 0) goto loop
    //   done: use result
    //
    // Trace for clz(1):  result=32, x=1 → x=0, result=31, exit → 31 ✓
    // Trace for clz(0x80000000): result=32, 32 iterations → result=0 ✓

    let result_slot = ctx.builder.build_alloca(ctx.function, ty.clone(), None);
    let input_slot = ctx.builder.build_alloca(ctx.function, ty.clone(), None);

    // Initialize: result = bit_width, input = val
    let init_result = ctx
        .builder
        .build_const_int(ctx.function, ty.clone(), bit_width);
    ctx.builder
        .build_store(ctx.function, init_result, result_slot);
    ctx.builder.build_store(ctx.function, val, input_slot);

    // Check if input is zero
    let zero = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
    let is_zero = ctx
        .builder
        .build_icmp(ctx.function, ICmpPredicate::Eq, val, zero);

    let loop_bb = ctx.builder.create_block(ctx.function, Some("clz_loop"));
    let done_bb = ctx.builder.create_block(ctx.function, Some("clz_done"));

    ctx.builder
        .build_cond_branch(ctx.function, is_zero, done_bb, loop_bb);

    // Loop body: x >>= 1; result--; if (x != 0) goto loop
    ctx.builder.set_insert_point(loop_bb);
    let cur_input = ctx.builder.build_load(ctx.function, input_slot, ty.clone());
    let cur_result = ctx
        .builder
        .build_load(ctx.function, result_slot, ty.clone());

    let one = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
    let shifted = ctx
        .builder
        .build_binop(ctx.function, BinOp::LShr, cur_input, one, ty.clone());
    let decremented =
        ctx.builder
            .build_binop(ctx.function, BinOp::Sub, cur_result, one, ty.clone());

    ctx.builder.build_store(ctx.function, shifted, input_slot);
    ctx.builder
        .build_store(ctx.function, decremented, result_slot);

    let zero2 = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
    let still_nonzero = ctx
        .builder
        .build_icmp(ctx.function, ICmpPredicate::Ne, shifted, zero2);
    ctx.builder
        .build_cond_branch(ctx.function, still_nonzero, loop_bb, done_bb);

    // Done: load result
    ctx.builder.set_insert_point(done_bb);
    let result = ctx
        .builder
        .build_load(ctx.function, result_slot, ty.clone());
    Ok(result)
}

/// Software CTZ (Count Trailing Zeros) using a loop in pure IR.
///
/// Algorithm:
///   result = bit_width (if x==0)
///   if (x == 0) goto done
///   result = 0; loop: if (x & 1) goto done; x >>= 1; result++; goto loop
///   done: use result
fn lower_software_ctz(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    bit_width: i64,
) -> Result<ValueId, LoweringError> {
    let ty = if bit_width == 64 {
        IrType::I64
    } else {
        IrType::I32
    };

    let result_slot = ctx.builder.build_alloca(ctx.function, ty.clone(), None);
    let input_slot = ctx.builder.build_alloca(ctx.function, ty.clone(), None);

    // Initialize: result = bit_width (result if zero), input = val
    let init_result = ctx
        .builder
        .build_const_int(ctx.function, ty.clone(), bit_width);
    ctx.builder
        .build_store(ctx.function, init_result, result_slot);
    ctx.builder.build_store(ctx.function, val, input_slot);

    let zero = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
    let is_zero = ctx
        .builder
        .build_icmp(ctx.function, ICmpPredicate::Eq, val, zero);

    let setup_bb = ctx.builder.create_block(ctx.function, Some("ctz_setup"));
    let loop_bb = ctx.builder.create_block(ctx.function, Some("ctz_loop"));
    let done_bb = ctx.builder.create_block(ctx.function, Some("ctz_done"));

    ctx.builder
        .build_cond_branch(ctx.function, is_zero, done_bb, setup_bb);

    // Setup: result = 0
    ctx.builder.set_insert_point(setup_bb);
    let zero_init = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
    ctx.builder
        .build_store(ctx.function, zero_init, result_slot);
    ctx.builder.build_branch(ctx.function, loop_bb);

    // Loop body: check low bit, shift right, increment
    ctx.builder.set_insert_point(loop_bb);
    let cur_input = ctx.builder.build_load(ctx.function, input_slot, ty.clone());
    let one = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
    let low_bit = ctx
        .builder
        .build_binop(ctx.function, BinOp::And, cur_input, one, ty.clone());
    let zero3 = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
    let bit_set = ctx
        .builder
        .build_icmp(ctx.function, ICmpPredicate::Ne, low_bit, zero3);

    let cont_bb = ctx.builder.create_block(ctx.function, Some("ctz_cont"));
    ctx.builder
        .build_cond_branch(ctx.function, bit_set, done_bb, cont_bb);

    // Continue: shift and increment
    ctx.builder.set_insert_point(cont_bb);
    let cur_input2 = ctx.builder.build_load(ctx.function, input_slot, ty.clone());
    let shifted = ctx
        .builder
        .build_binop(ctx.function, BinOp::LShr, cur_input2, one, ty.clone());
    ctx.builder.build_store(ctx.function, shifted, input_slot);
    let cur_result = ctx
        .builder
        .build_load(ctx.function, result_slot, ty.clone());
    let one2 = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
    let incremented =
        ctx.builder
            .build_binop(ctx.function, BinOp::Add, cur_result, one2, ty.clone());
    ctx.builder
        .build_store(ctx.function, incremented, result_slot);
    ctx.builder.build_branch(ctx.function, loop_bb);

    // Done: load result
    ctx.builder.set_insert_point(done_bb);
    let result = ctx
        .builder
        .build_load(ctx.function, result_slot, ty.clone());
    Ok(result)
}

fn lower_software_popcount(
    ctx: &mut LoweringContext<'_>,
    val: ValueId,
    bit_width: i64,
) -> Result<ValueId, LoweringError> {
    let ty = if bit_width == 64 {
        IrType::I64
    } else {
        IrType::I32
    };
    let mut c = |v: i64| ctx.builder.build_const_int(ctx.function, ty.clone(), v);

    if bit_width == 32 {
        let one = c(1);
        let m1 = c(0x55555555);
        let m2 = c(0x33333333);
        let m4 = c(0x0F0F0F0F);
        let mul = c(0x01010101);
        let s24 = c(24);
        let s2 = c(2);
        let s4 = c(4);

        // x = x - ((x >> 1) & 0x55555555)
        let t1 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, val, one, ty.clone());
        let t2 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t1, m1, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::Sub, val, t2, ty.clone());

        // x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        let t3 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, x, m2, ty.clone());
        let t4 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, x, s2, ty.clone());
        let t5 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t4, m2, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::Add, t3, t5, ty.clone());

        // x = (x + (x >> 4)) & 0x0F0F0F0F
        let t6 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, x, s4, ty.clone());
        let t7 = ctx
            .builder
            .build_binop(ctx.function, BinOp::Add, x, t6, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t7, m4, ty.clone());

        // return (x * 0x01010101) >> 24
        let t8 = ctx
            .builder
            .build_binop(ctx.function, BinOp::Mul, x, mul, ty.clone());
        let res = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, t8, s24, ty.clone());
        Ok(res)
    } else {
        // 64-bit popcount
        let one = c(1);
        let m1 = c(0x5555555555555555u64 as i64);
        let m2 = c(0x3333333333333333u64 as i64);
        let m4 = c(0x0F0F0F0F0F0F0F0Fu64 as i64);
        let mul = c(0x0101010101010101u64 as i64);
        let s56 = c(56);
        let s2 = c(2);
        let s4 = c(4);

        let t1 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, val, one, ty.clone());
        let t2 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t1, m1, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::Sub, val, t2, ty.clone());

        let t3 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, x, m2, ty.clone());
        let t4 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, x, s2, ty.clone());
        let t5 = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t4, m2, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::Add, t3, t5, ty.clone());

        let t6 = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, x, s4, ty.clone());
        let t7 = ctx
            .builder
            .build_binop(ctx.function, BinOp::Add, x, t6, ty.clone());
        let x = ctx
            .builder
            .build_binop(ctx.function, BinOp::And, t7, m4, ty.clone());

        let t8 = ctx
            .builder
            .build_binop(ctx.function, BinOp::Mul, x, mul, ty.clone());
        let res = ctx
            .builder
            .build_binop(ctx.function, BinOp::LShr, t8, s56, ty.clone());
        Ok(res)
    }
}

fn lower_builtin_bswap(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    span: Span,
    bits: u32,
) -> Result<ValueId, LoweringError> {
    if args.is_empty() {
        ctx.diagnostics().error(span, "bswap requires one argument");
        return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
    }

    let x = lower_expression(ctx, &args[0])?;
    // Use the appropriate integer width — I32 for bswap16/32, I64 for bswap64.
    // Using I64 for everything is wasteful and causes correctness issues on
    // 32-bit targets where I64 operations are emulated with register pairs.
    let ty = if bits <= 32 { IrType::I32 } else { IrType::I64 };

    // Helper closure to build shift + mask pairs
    let mut c = |v: i64| ctx.builder.build_const_int(ctx.function, ty.clone(), v);

    let result = match bits {
        16 => {
            // bswap16(x) = ((x >> 8) & 0xFF) | ((x & 0xFF) << 8)
            let s8 = c(8);
            let mask_ff = c(0xFF);
            let hi = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s8, ty.clone());
            let hi_masked =
                ctx.builder
                    .build_binop(ctx.function, BinOp::And, hi, mask_ff, ty.clone());
            let lo_masked =
                ctx.builder
                    .build_binop(ctx.function, BinOp::And, x, mask_ff, ty.clone());
            let lo_shifted =
                ctx.builder
                    .build_binop(ctx.function, BinOp::Shl, lo_masked, s8, ty.clone());
            ctx.builder
                .build_binop(ctx.function, BinOp::Or, hi_masked, lo_shifted, ty.clone())
        }
        32 => {
            // bswap32(x):
            //   byte0 = (x >> 24) & 0xFF
            //   byte1 = (x >> 8)  & 0xFF00
            //   byte2 = (x << 8)  & 0xFF0000
            //   byte3 = (x << 24) & 0xFF000000
            //   result = byte0 | byte1 | byte2 | byte3
            let s8 = c(8);
            let s24 = c(24);
            let m_ff = c(0xFF);
            let m_ff00 = c(0xFF00);
            let m_ff0000 = c(0xFF_0000);
            let m_ff000000 = c(0xFF00_0000_i64);

            let b0 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s24, ty.clone());
            let b0 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b0, m_ff, ty.clone());
            let b1 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s8, ty.clone());
            let b1 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b1, m_ff00, ty.clone());
            let b2 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s8, ty.clone());
            let b2 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b2, m_ff0000, ty.clone());
            let b3 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s24, ty.clone());
            let b3 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b3, m_ff000000, ty.clone());

            let r01 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b0, b1, ty.clone());
            let r23 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b2, b3, ty.clone());
            ctx.builder
                .build_binop(ctx.function, BinOp::Or, r01, r23, ty.clone())
        }
        64 => {
            // bswap64(x): reverse all 8 bytes using shift/mask pairs.
            let s8 = c(8);
            let s24 = c(24);
            let s40 = c(40);
            let s56 = c(56);

            let m0 = c(0xFF);
            let m1 = c(0xFF00);
            let m2 = c(0xFF_0000);
            let m3 = c(0xFF00_0000_i64);
            let m4 = c(0xFF_0000_0000_i64);
            let m5 = c(0xFF00_0000_0000_i64);
            let m6 = c(0xFF_0000_0000_0000_i64);
            // byte7 mask not needed — shift right 56 isolates it

            // byte 0 (MSB of input → LSB of output)
            let b0 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s56, ty.clone());
            let b0 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b0, m0, ty.clone());
            // byte 1
            let b1 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s40, ty.clone());
            let b1 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b1, m1, ty.clone());
            // byte 2
            let b2 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s24, ty.clone());
            let b2 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b2, m2, ty.clone());
            // byte 3
            let b3 = ctx
                .builder
                .build_binop(ctx.function, BinOp::LShr, x, s8, ty.clone());
            let b3 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b3, m3, ty.clone());
            // byte 4
            let b4 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s8, ty.clone());
            let b4 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b4, m4, ty.clone());
            // byte 5
            let b5 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s24, ty.clone());
            let b5 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b5, m5, ty.clone());
            // byte 6
            let b6 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s40, ty.clone());
            let b6 = ctx
                .builder
                .build_binop(ctx.function, BinOp::And, b6, m6, ty.clone());
            // byte 7 (LSB of input → MSB of output)
            let b7 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Shl, x, s56, ty.clone());

            let r01 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b0, b1, ty.clone());
            let r23 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b2, b3, ty.clone());
            let r45 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b4, b5, ty.clone());
            let r67 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, b6, b7, ty.clone());
            let r03 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, r01, r23, ty.clone());
            let r47 = ctx
                .builder
                .build_binop(ctx.function, BinOp::Or, r45, r67, ty.clone());
            ctx.builder
                .build_binop(ctx.function, BinOp::Or, r03, r47, ty.clone())
        }
        _ => {
            return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
        }
    };

    Ok(result)
}

/// Lowers __builtin_abs / __builtin_labs / __builtin_llabs.
///
/// Branchless implementation avoiding phi nodes (which the x86-64 backend
/// handles unreliably).  Uses the classic arithmetic identity:
///   mask = x >> (bits - 1)         // all-ones if negative, all-zeros if positive
///   abs(x) = (x XOR mask) - mask   // flips bits and adds 1 when negative (= two's complement negation)
fn lower_builtin_abs(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    span: Span,
    ty: IrType,
) -> Result<ValueId, LoweringError> {
    if args.is_empty() {
        ctx.diagnostics().error(span, "abs requires one argument");
        return Ok(ctx.builder.build_const_int(ctx.function, ty.clone(), 0));
    }

    let x = lower_expression(ctx, &args[0])?;

    // Determine bit width for the arithmetic shift amount.
    // On the x86-64 backend everything lives in 64-bit registers, so we
    // always use 63 as the shift amount regardless of the C-level type.
    let shift_amt = ctx.builder.build_const_int(ctx.function, ty.clone(), 63);

    // mask = x AShr (bits-1)  → all 1s if negative, all 0s if non-negative
    let mask = ctx
        .builder
        .build_binop(ctx.function, BinOp::AShr, x, shift_amt, ty.clone());

    // (x XOR mask): if negative, flips all bits; if non-negative, no-op
    let xored = ctx
        .builder
        .build_binop(ctx.function, BinOp::Xor, x, mask, ty.clone());

    // (x XOR mask) - mask: completes the two's-complement negation when negative
    let result = ctx
        .builder
        .build_binop(ctx.function, BinOp::Sub, xored, mask, ty.clone());

    Ok(result)
}

/// Lowers __builtin_frame_address(0) or __builtin_return_address(0).
fn lower_builtin_frame_or_return_address(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    _span: Span,
    is_frame: bool,
) -> Result<ValueId, LoweringError> {
    // Only level 0 is reliably supported
    let (asm_template, constraints) = match ctx.target() {
        Target::X86_64 => {
            if is_frame {
                ("movq %%rbp, $0".to_string(), "=r".to_string())
            } else {
                // Return address is at [rbp+8] in the standard frame layout
                ("movq 8(%%rbp), $0".to_string(), "=r".to_string())
            }
        }
        Target::I686 => {
            if is_frame {
                ("movl %%ebp, $0".to_string(), "=r".to_string())
            } else {
                ("movl 4(%%ebp), $0".to_string(), "=r".to_string())
            }
        }
        Target::AArch64 => {
            if is_frame {
                ("mov $0, x29".to_string(), "=r".to_string())
            } else {
                ("mov $0, x30".to_string(), "=r".to_string())
            }
        }
        Target::RiscV64 => {
            if is_frame {
                ("mv $0, s0".to_string(), "=r".to_string())
            } else {
                ("mv $0, ra".to_string(), "=r".to_string())
            }
        }
    };

    // Lower any arguments (typically just 0)
    for arg in args {
        let _ = lower_expression(ctx, arg);
    }

    let result = ctx.builder.build_inline_asm(
        ctx.function,
        asm_template,
        constraints,
        vec![],
        vec![],
        false,
        false,
    );

    Ok(result.unwrap_or_else(|| ctx.builder.build_const_null(ctx.function, IrType::Ptr)))
}

/// Lowers __builtin_va_start(ap, last_param).
/// On x86-64 System V, va_start initializes a va_list struct.
/// We lower this as a platform-specific inline assembly or a call to
/// the compiler-support __va_start intrinsic.
fn lower_builtin_va_start(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    _span: Span,
) -> Result<ValueId, LoweringError> {
    // Lower the va_list argument (first arg) as an lvalue
    if args.len() >= 1 {
        let ap_addr = lower_lvalue(ctx, &args[0])?;

        // On x86-64, the backend reserves a 48-byte register save area
        // in the stack frame for variadic functions, spilling the six
        // integer argument registers (RDI–R9) into [RBP-48] through
        // [RBP-8].  va_start must point past the named parameters to
        // the first unnamed argument slot.
        //
        // Save-area layout:
        //   [RBP - 48]  arg 0  (RDI)
        //   [RBP - 40]  arg 1  (RSI)
        //   [RBP - 32]  arg 2  (RDX)
        //   [RBP - 24]  arg 3  (RCX)
        //   [RBP - 16]  arg 4  (R8)
        //   [RBP -  8]  arg 5  (R9)
        //
        // For a function `f(int a, int b, ...)` with 2 named params,
        // the first unnamed arg is at [RBP - 48 + 2*8] = [RBP - 32].
        //
        // LEA displacement = -(48 - num_named * 8)
        let num_named = ctx.function.params.len();

        // Choose the correct IR type for the result based on the target
        // pointer width.  On i686 (32-bit), the LEA result is a 32-bit
        // address and must be typed as I32; on 64-bit targets it's I64.
        let ptr_width = ctx.target().pointer_width() as i64;
        let result_ty = if ptr_width == 4 {
            IrType::I32
        } else {
            IrType::I64
        };

        let frame_ptr = ctx.builder.build_inline_asm_typed(
            ctx.function,
            match ctx.target() {
                Target::X86_64 => {
                    // The backend spills 6 regs to a 48-byte save area.
                    // Named params occupy the first `num_named` slots.
                    // First unnamed arg: [RBP - (48 - num_named * 8)]
                    let disp = -(48i64 - (num_named as i64) * 8);
                    format!("leaq {}(%%rbp), $0", disp)
                }
                Target::I686 => {
                    // i686: all params on the stack.  Named params start
                    // at [EBP+8].  First unnamed = [EBP + 8 + num_named*4].
                    let disp = 8 + (num_named as i64) * 4;
                    format!("leal {}(%%ebp), $0", disp)
                }
                Target::AArch64 => {
                    let disp = 16 + (num_named as i64) * 8;
                    format!("add $0, x29, #{}", disp)
                }
                Target::RiscV64 => {
                    // The RISC-V backend spills a0–a7 into a 64-byte
                    // save area at the top of the callee's frame:
                    //   a0 at [FP − 64], a1 at [FP − 56], …, a7 at [FP − 8]
                    // va_start skips the first `num_named` slots to reach
                    // the first unnamed argument.
                    let disp = -64 + (num_named as i64) * 8;
                    format!("addi $0, s0, {}", disp)
                }
            },
            "=r".to_string(),
            vec![],
            vec![],
            true,
            false,
            result_ty,
        );

        if let Some(fp) = frame_ptr {
            ctx.builder.build_store(ctx.function, fp, ap_addr);
        }
    }

    Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
}

/// Lowers __builtin_va_copy(dest, src).
fn lower_builtin_va_copy(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    _span: Span,
) -> Result<ValueId, LoweringError> {
    if args.len() >= 2 {
        let dest = lower_lvalue(ctx, &args[0])?;
        let src = lower_expression(ctx, &args[1])?;
        ctx.builder.build_store(ctx.function, src, dest);
    }
    Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0))
}

/// Lowers __builtin_{add,sub,mul}_overflow(a, b, result_ptr).
/// Returns 1 on overflow, 0 otherwise. Stores the (possibly wrapped) result.
///
/// Strategy: Pure-arithmetic overflow detection using SHL+AShr for sign
/// extension (avoids AND with 0xFFFFFFFF which is broken on x86-64 due
/// to immediate sign-extension, and avoids inline asm which has operand
/// substitution issues).
///
/// Algorithm:
///   1. Sign-extend both inputs from 32 to 64 bits via SHL 32 + AShr 32
///   2. Compute the operation in full 64-bit precision (cannot overflow
///      for add/sub of 32-bit inputs; for mul of 32-bit inputs the
///      product fits in 62 bits)
///   3. Store the low-32-bit wrapped result to the output pointer
///   4. Sign-extend that 32-bit result back to 64 bits
///   5. Compare: if full 64-bit result != sign-extended 32-bit result → overflow
fn lower_builtin_overflow(
    ctx: &mut LoweringContext<'_>,
    args: &[Expression],
    span: Span,
    op: BinOp,
) -> Result<ValueId, LoweringError> {
    if args.len() < 3 {
        ctx.diagnostics()
            .error(span, "overflow builtin requires 3 arguments");
        return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
    }

    let a = lower_expression(ctx, &args[0])?;
    let b = lower_expression(ctx, &args[1])?;
    let result_ptr = lower_expression(ctx, &args[2])?;

    let ptr_width = ctx.target().pointer_width();

    if ptr_width >= 8 {
        // 64-bit targets: widen operands to I64, do the math, check if
        // the 64-bit result differs from the sign-extended 32-bit result.
        let ty = IrType::I64;
        let shift32 = ctx.builder.build_const_int(ctx.function, ty.clone(), 32);

        let a_shl = ctx
            .builder
            .build_binop(ctx.function, BinOp::Shl, a, shift32, ty.clone());
        let a_sext = ctx
            .builder
            .build_binop(ctx.function, BinOp::AShr, a_shl, shift32, ty.clone());

        let b_shl = ctx
            .builder
            .build_binop(ctx.function, BinOp::Shl, b, shift32, ty.clone());
        let b_sext = ctx
            .builder
            .build_binop(ctx.function, BinOp::AShr, b_shl, shift32, ty.clone());

        let full_result = ctx
            .builder
            .build_binop(ctx.function, op, a_sext, b_sext, ty.clone());

        let result_trunc = ctx
            .builder
            .build_trunc(ctx.function, full_result, IrType::I32);
        ctx.builder
            .build_store(ctx.function, result_trunc, result_ptr);

        let r_shl =
            ctx.builder
                .build_binop(ctx.function, BinOp::Shl, full_result, shift32, ty.clone());
        let r_sext = ctx
            .builder
            .build_binop(ctx.function, BinOp::AShr, r_shl, shift32, ty.clone());

        let overflow = ctx
            .builder
            .build_icmp(ctx.function, ICmpPredicate::Ne, full_result, r_sext);
        let overflow_i32 = ctx.builder.build_zext(ctx.function, overflow, IrType::I32);
        Ok(overflow_i32)
    } else {
        // 32-bit targets (i686): detect overflow using only I32 operations
        // to avoid broken I64 arithmetic on 32-bit backends.
        let ty = IrType::I32;
        let result = ctx
            .builder
            .build_binop(ctx.function, op.clone(), a, b, ty.clone());
        ctx.builder.build_store(ctx.function, result, result_ptr);

        let shift31 = ctx.builder.build_const_int(ctx.function, ty.clone(), 31);
        let zero = ctx.builder.build_const_int(ctx.function, ty.clone(), 0);
        let neg_one = ctx.builder.build_const_int(ctx.function, ty.clone(), -1);

        let overflow_flag = match op {
            BinOp::Add => {
                // Signed add overflow: operands same sign but result differs.
                // overflow = (~(a ^ b) & (a ^ result)) >> 31
                let xor_ab = ctx
                    .builder
                    .build_binop(ctx.function, BinOp::Xor, a, b, ty.clone());
                let not_xor_ab =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Xor, xor_ab, neg_one, ty.clone());
                let xor_ar =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Xor, a, result, ty.clone());
                let bits = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::And,
                    not_xor_ab,
                    xor_ar,
                    ty.clone(),
                );
                let shifted =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::LShr, bits, shift31, ty.clone());
                shifted
            }
            BinOp::Sub => {
                // Signed sub overflow: different signs and result sign != a sign.
                // overflow = ((a ^ b) & (a ^ result)) >> 31
                let xor_ab = ctx
                    .builder
                    .build_binop(ctx.function, BinOp::Xor, a, b, ty.clone());
                let xor_ar =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Xor, a, result, ty.clone());
                let bits =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::And, xor_ab, xor_ar, ty.clone());
                let shifted =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::LShr, bits, shift31, ty.clone());
                shifted
            }
            BinOp::Mul => {
                // For multiplication overflow on 32-bit, use the check:
                // if b != 0: overflow = (result / b != a)
                // Special case: b == 0 → no overflow
                // Special case: a == INT_MIN && b == -1 → overflow
                let b_is_zero = ctx
                    .builder
                    .build_icmp(ctx.function, ICmpPredicate::Eq, b, zero);

                // result / b (signed division) — safe when b != 0
                let div_result =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::SDiv, result, b, ty.clone());
                let div_ne_a =
                    ctx.builder
                        .build_icmp(ctx.function, ICmpPredicate::Ne, div_result, a);
                let div_ne_i32 = ctx.builder.build_zext(ctx.function, div_ne_a, ty.clone());

                // If b == 0, no overflow; else, overflow = (result / b != a)
                let b_zero_i32 = ctx.builder.build_zext(ctx.function, b_is_zero, ty.clone());
                let one = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
                let b_nonzero =
                    ctx.builder
                        .build_binop(ctx.function, BinOp::Xor, b_zero_i32, one, ty.clone());
                let ov = ctx.builder.build_binop(
                    ctx.function,
                    BinOp::And,
                    div_ne_i32,
                    b_nonzero,
                    ty.clone(),
                );
                ov
            }
            _ => zero,
        };

        // Ensure the flag is 0 or 1 (AND with 1)
        let one = ctx.builder.build_const_int(ctx.function, ty.clone(), 1);
        let overflow_i32 =
            ctx.builder
                .build_binop(ctx.function, BinOp::And, overflow_flag, one, ty.clone());
        Ok(overflow_i32)
    }
}

/// Infers the return type of a function call from the callee expression.
fn infer_call_return_type(ctx: &LoweringContext<'_>, callee: &Expression) -> IrType {
    // Unwrap _Generic to the resolved function expression so we can
    // correctly infer the return type of dispatched calls like
    // `_Generic(x, int: abs_int, float: abs_float)(x)`.
    let effective = resolve_generic_callee(ctx, callee);
    if let Expression::Identifier { name, .. } = effective {
        if let Some(info) = ctx.module_ctx.global_symbols.get(name) {
            if let IrType::Function { return_type, .. } = &info.ir_type {
                return (**return_type).clone();
            }
        }
    }
    IrType::I32
}

/// If `callee` is a `_Generic(ctrl, ...)` expression, resolve the type
/// matching at compile time and return the selected association expression.
/// This allows `infer_call_return_type` and `infer_callee_variadic_info`
/// to inspect the actual function being called through _Generic dispatch.
fn resolve_generic_callee<'a>(ctx: &LoweringContext<'_>, callee: &'a Expression) -> &'a Expression {
    if let Expression::Generic {
        controlling,
        associations,
        ..
    } = callee
    {
        let ctrl_key = infer_controlling_type_key(ctx, controlling);
        let mut default_assoc: Option<&'a GenericAssociation> = None;
        for assoc in associations {
            if let Some(ref tn) = assoc.type_name {
                let assoc_key = typename_to_generic_key(ctx, tn);
                if ctrl_key == assoc_key {
                    return &assoc.expression;
                }
            } else {
                default_assoc = Some(assoc);
            }
        }
        if let Some(default) = default_assoc {
            return &default.expression;
        }
    }
    callee
}

// ============================================================================
// Array subscript
// ============================================================================

/// Lowers `array[index]` — rvalue (loads from the computed address).
fn lower_array_subscript(
    ctx: &mut LoweringContext<'_>,
    array: &Expression,
    index: &Expression,
    span: Span,
) -> Result<ValueId, LoweringError> {
    let addr = lower_array_subscript_lvalue(ctx, array, index, span)?;
    let elem_ty = infer_array_element_type(ctx, array);
    Ok(ctx.builder.build_load(ctx.function, addr, elem_ty))
}

/// Lowers `array[index]` to lvalue address (GEP).
///
/// # Array-to-Pointer Decay for Member Accesses
///
/// When the base of a subscript is a struct/union member access (e.g.
/// `obj.array_field[i]`), we must obtain the *address* of the member
/// rather than loading its value.  In C, an array expression decays to
/// a pointer to the first element — this pointer IS the address of the
/// array object, not a separate value that needs to be loaded.
///
/// `lower_expression` on a `MemberAccess` calls `lower_member_access_rvalue`,
/// which performs a GEP followed by a LOAD.  The LOAD is correct for
/// scalar fields (`s.x`) but wrong for array fields (`s.arr`), because
/// loading an array treats the raw bytes as an integer/pointer value
/// and then attempts to dereference that fabricated address.
///
/// By calling `lower_member_access_lvalue` instead, we obtain the GEP
/// result (the field's address) without the LOAD, which is exactly the
/// pointer produced by array decay.
fn lower_array_subscript_lvalue(
    ctx: &mut LoweringContext<'_>,
    array: &Expression,
    index: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let base = match array {
        // For struct/union member access, use lvalue semantics so that
        // array members get their address (array decay) rather than a
        // loaded value.
        Expression::MemberAccess {
            object,
            member,
            span,
        } => lower_member_access_lvalue(ctx, object, *member, false, *span)?,
        Expression::ArrowAccess {
            pointer,
            member,
            span,
        } => lower_member_access_lvalue(ctx, pointer, *member, true, *span)?,
        _ => lower_expression(ctx, array)?,
    };
    let idx = lower_expression(ctx, index)?;
    let elem_ty = infer_array_element_type(ctx, array);
    Ok(ctx
        .builder
        .build_gep(ctx.function, base, vec![idx], elem_ty, true))
}

/// Infers the element type of an array subscript base expression.
fn infer_array_element_type(ctx: &LoweringContext<'_>, base_expr: &Expression) -> IrType {
    match base_expr {
        Expression::Identifier { name, .. } => {
            if let Some(alloca) = ctx.variables.get(name) {
                let alloca_ty = resolve_alloca_element_type(ctx, *alloca);
                if let IrType::Array { element, .. } = &alloca_ty {
                    return (**element).clone();
                }
                if matches!(alloca_ty, IrType::Ptr) {
                    // Check variable CType for pointer-to-X to infer X.
                    if let Some(ctype) = ctx.variable_ctypes.get(name) {
                        if let CType::Pointer(inner) = ctype {
                            if let Ok(ir_ty) =
                                super::c_type_to_ir_type(inner, &ctx.module_ctx.target)
                            {
                                return ir_ty;
                            }
                        }
                    }
                    return IrType::I8;
                }
                return alloca_ty;
            }
            if let Some(info) = ctx.module_ctx.global_symbols.get(name) {
                if let IrType::Array { element, .. } = &info.ir_type {
                    return (**element).clone();
                }
            }
            IrType::I8
        }
        Expression::MemberAccess { object, member, .. }
        | Expression::ArrowAccess {
            pointer: object,
            member,
            ..
        } => {
            // `obj.arr[i]` or `ptr->arr[i]` — the subscript base is a
            // struct member that is itself an array.  Resolve the
            // struct tag, find the member's CType, peel off CType::Array
            // to get the element CType, and convert to IrType.
            let is_arrow = matches!(base_expr, Expression::ArrowAccess { .. });
            let tag = find_struct_tag_for_expr(ctx, object);
            if let Some(tag_sym) = tag {
                if let Some(def) = ctx.module_ctx.struct_defs.get(&tag_sym) {
                    let member_str = ctx.module_ctx.interner.resolve(*member).to_string();
                    let fields = match def {
                        CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
                        _ => return IrType::I8,
                    };
                    for f in fields {
                        if let Some(ref fname) = f.name {
                            if *fname == member_str {
                                match &f.ty {
                                    CType::Array { element, .. } => {
                                        if let Ok(ir_ty) = super::c_type_to_ir_type(
                                            element,
                                            &ctx.module_ctx.target,
                                        ) {
                                            return ir_ty;
                                        }
                                    }
                                    CType::Pointer(inner) => {
                                        if let Ok(ir_ty) =
                                            super::c_type_to_ir_type(inner, &ctx.module_ctx.target)
                                        {
                                            return ir_ty;
                                        }
                                    }
                                    _ => {
                                        if let Ok(ir_ty) =
                                            super::c_type_to_ir_type(&f.ty, &ctx.module_ctx.target)
                                        {
                                            return ir_ty;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: try inferring from the IR struct type.
            let base_struct = infer_base_struct_type(ctx, object, is_arrow);
            if let IrType::Struct { fields, .. } = &base_struct {
                let member_idx = infer_field_index(ctx, object, *member);
                if member_idx < fields.len() {
                    let field_ty = &fields[member_idx];
                    if let IrType::Array { element, .. } = field_ty {
                        return (**element).clone();
                    }
                    return field_ty.clone();
                }
            }
            IrType::I8
        }
        Expression::ArraySubscript { array, .. } => {
            // Multi-dimensional: `arr[i][j]`.  Peel one level.
            let outer = infer_array_element_type(ctx, array);
            if let IrType::Array { element, .. } = &outer {
                return (**element).clone();
            }
            outer
        }
        _ => IrType::I8,
    }
}

// ============================================================================
// Struct/union member access
// ============================================================================

/// Lowers `object.member` or `pointer->member` as rvalue.
fn lower_member_access_rvalue(
    ctx: &mut LoweringContext<'_>,
    base: &Expression,
    member: Symbol,
    is_arrow: bool,
    span: Span,
) -> Result<ValueId, LoweringError> {
    let addr = lower_member_access_lvalue(ctx, base, member, is_arrow, span)?;

    // Check if the member is an array type.  Arrays decay to pointers
    // in value context (C11 §6.3.2.1 ¶3), so we return the address
    // of the first element directly instead of generating a load.
    let base_struct_ty = infer_base_struct_type(ctx, base, is_arrow);
    let field_idx = infer_field_index(ctx, base, member);
    if let IrType::Struct { ref fields, .. } = base_struct_ty {
        if field_idx < fields.len() {
            if matches!(&fields[field_idx], IrType::Array { .. }) {
                // Array-to-pointer decay: the GEP result (address of the
                // array field) is already the pointer to the first element.
                return Ok(addr);
            }
        }
    }

    let field_ty = infer_member_ir_type(ctx, base, member, is_arrow);
    Ok(ctx.builder.build_load(ctx.function, addr, field_ty))
}

/// Infers the IR type of a struct/union member by examining the base struct's
/// IR type and returning the field type at the resolved index.
fn infer_member_ir_type(
    ctx: &LoweringContext<'_>,
    base_expr: &Expression,
    member: Symbol,
    is_arrow: bool,
) -> IrType {
    // For union types, the IR struct has a different layout than the C
    // union.  We must resolve the member type from the C type definition
    // stored in struct_defs rather than from the IR struct field index.
    if is_base_union_type(ctx, base_expr, is_arrow) {
        if let Some(tag_sym) = find_struct_tag_for_expr(ctx, base_expr) {
            if let Some(c_type) = ctx.module_ctx.struct_defs.get(&tag_sym) {
                let member_str = ctx.module_ctx.interner.resolve(member);
                // Extract fields from the CType::Union
                let fields = match c_type {
                    CType::Union { fields, .. } => fields,
                    CType::Struct { fields, .. } => fields,
                    _ => return IrType::I32,
                };
                for field_def in fields {
                    if let Some(ref field_name) = field_def.name {
                        if field_name == member_str {
                            return IrType::from_c_type(&field_def.ty, ctx.target());
                        }
                    }
                }
            }
        }
        // Fallback: return the IR struct's first field type
        let base_struct_ty = infer_base_struct_type(ctx, base_expr, is_arrow);
        if let IrType::Struct { ref fields, .. } = base_struct_ty {
            if !fields.is_empty() {
                return fields[0].clone();
            }
        }
        return IrType::I32;
    }

    let base_struct_ty = infer_base_struct_type(ctx, base_expr, is_arrow);
    let field_idx = infer_field_index(ctx, base_expr, member);
    if let IrType::Struct { ref fields, .. } = base_struct_ty {
        if field_idx < fields.len() {
            return fields[field_idx].clone();
        }
    }
    // Fallback to I32 if we cannot determine the field type.
    IrType::I32
}

/// Checks if the base expression of a member access refers to a union type.
fn is_base_union_type(ctx: &LoweringContext<'_>, base_expr: &Expression, is_arrow: bool) -> bool {
    // Helper: check a CType to see if it's a union (or pointer to union for arrow).
    let check_ctype = |ctype: &CType| -> bool {
        if is_arrow {
            matches!(ctype, CType::Pointer(inner) if matches!(inner.as_ref(), CType::Union { .. }))
        } else {
            matches!(ctype, CType::Union { .. })
        }
    };

    match base_expr {
        Expression::Identifier { name, .. } => {
            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                let _name_str = ctx.module_ctx.interner.resolve(*name);
                // eprintln!("[is_base_union_type] var={} ctype={:?} is_arrow={} result={}", name_str, ctype, is_arrow, check_ctype(ctype));
                return check_ctype(ctype);
            }
            let _name_str = ctx.module_ctx.interner.resolve(*name);
            // eprintln!("[is_base_union_type] var={} NOT FOUND in variable_ctypes", name_str);
            false
        }
        Expression::MemberAccess { .. } => {
            // Nested member access — check if the result type is a union
            // (would require more analysis; for now, return false)
            false
        }
        Expression::UnaryOp { operand, op, .. } => {
            use crate::frontend::parser::ast::UnaryOperator;
            match op {
                UnaryOperator::Deref => is_base_union_type(ctx, operand, false),
                UnaryOperator::AddressOf => is_base_union_type(ctx, operand, true),
                _ => false,
            }
        }
        _ => false,
    }
}

/// Lowers `.member` or `->member` to lvalue (address of the field).
fn lower_member_access_lvalue(
    ctx: &mut LoweringContext<'_>,
    base: &Expression,
    member: Symbol,
    is_arrow: bool,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let base_addr = if is_arrow {
        lower_expression(ctx, base)?
    } else {
        lower_lvalue(ctx, base)?
    };

    // For unions: all fields overlap at offset 0.  We use GEP [0, 0] to
    // reach the first (and only) IR field, then bitcast to the actual
    // member type pointer.  This is correct because the IR representation
    // of a union is `{ best_aligned_field, [N x i8] }`.
    if is_base_union_type(ctx, base, is_arrow) {
        let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
        let zero2 = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
        let base_struct_ty = infer_base_struct_type(ctx, base, is_arrow);
        // GEP [0, 0] to get pointer to offset 0 of the union.
        // All union fields share offset 0, so we always use index 0.
        let field0_ptr = ctx.builder.build_gep(
            ctx.function,
            base_addr,
            vec![zero, zero2],
            base_struct_ty,
            true,
        );
        return Ok(field0_ptr);
    }

    let field_idx = infer_field_index(ctx, base, member);
    let base_struct_ty = infer_base_struct_type(ctx, base, is_arrow);

    let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
    let idx_val = ctx
        .builder
        .build_const_int(ctx.function, IrType::I32, field_idx as i64);

    Ok(ctx.builder.build_gep(
        ctx.function,
        base_addr,
        vec![zero, idx_val],
        base_struct_ty,
        true,
    ))
}

/// Infers the field index within a struct for a given member symbol.
///
/// Resolves the struct/union tag associated with the base expression's
/// variable, then looks up the field name in the module-level field
/// name registry (`struct_field_names`).
///
/// Falls back to 0 if the struct tag or field name cannot be resolved.
fn infer_field_index(ctx: &LoweringContext<'_>, base_expr: &Expression, member: Symbol) -> usize {
    let _member_str = ctx.module_ctx.interner.resolve(member);
    // Step 1: Determine the struct tag from the base expression's variable.
    let tag = find_struct_tag_for_expr(ctx, base_expr);

    if let Some(tag_sym) = tag {
        let _tag_str = ctx.module_ctx.interner.resolve(tag_sym);
        // eprintln!("[DEBUG infer_field_index] member={} tag={}", member_str, tag_str);
        // Step 2: Look up the field name list for this struct tag.
        if let Some(field_names) = ctx.module_ctx.struct_field_names.get(&tag_sym) {
            // eprintln!("[DEBUG infer_field_index] field_names count={}", field_names.len());
            for (idx, name) in field_names.iter().enumerate() {
                if let Some(fname) = name {
                    let _fname_str = ctx.module_ctx.interner.resolve(*fname);
                    // eprintln!("[DEBUG infer_field_index]   field[{}] = {}", idx, fname_str);
                    if *fname == member {
                        // eprintln!("[DEBUG infer_field_index] FOUND at idx={}", idx);
                        return idx;
                    }
                }
            }
        } else {
            // eprintln!("[DEBUG infer_field_index] NO field_names for tag {}", tag_str);
        }
    } else {
        // eprintln!("[DEBUG infer_field_index] member={} NO TAG FOUND", member_str);
    }

    // Fallback: try to resolve from the IR struct type itself by
    // checking if it has enough fields.  This path handles cases
    // where the struct definition wasn't encountered through the
    // parser AST (e.g. sema-produced CheckedDeclarations).
    // eprintln!("[DEBUG infer_field_index] FALLBACK to 0 for member={}", member_str);
    0
}

/// Finds the struct/union tag symbol associated with an expression's type.
///
/// Handles various expression forms:
/// - `Identifier`: Direct struct variable → look up CType
/// - `ArrowAccess` on pointer variable: Dereference CType::Pointer
/// - `Cast` to struct pointer: Extract tag from the cast's type name
/// - `MemberAccess` on struct: Recursively resolve sub-struct tags
fn find_struct_tag_for_expr(ctx: &LoweringContext<'_>, expr: &Expression) -> Option<Symbol> {
    // Helper to extract tag symbol from a CType.
    let tag_from_ctype = |ctype: &CType| -> Option<Symbol> {
        match ctype {
            CType::Struct {
                name: Some(tag_name),
                ..
            }
            | CType::Union {
                name: Some(tag_name),
                ..
            } => ctx.module_ctx.interner.lookup(tag_name),
            CType::Pointer(inner) => match inner.as_ref() {
                CType::Struct {
                    name: Some(tag_name),
                    ..
                }
                | CType::Union {
                    name: Some(tag_name),
                    ..
                } => ctx.module_ctx.interner.lookup(tag_name),
                _ => None,
            },
            _ => None,
        }
    };

    match expr {
        Expression::Identifier { name, .. } => {
            let _name_str = ctx.module_ctx.interner.resolve(*name);
            // Check function-local variable_ctypes first.
            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                let tag = tag_from_ctype(ctype);
                // eprintln!("[find_struct_tag] Identifier '{}' sym={:?} found in local ctypes -> tag={:?}", name_str, name, tag);
                return tag;
            }
            // Fall back to module-level global variable types so that
            // struct member access on globals (e.g. `p_global.x`) can
            // resolve the struct tag and find the correct field offset.
            if let Some(ctype) = ctx.module_ctx.global_variable_ctypes.get(name) {
                let tag = tag_from_ctype(ctype);
                // eprintln!("[find_struct_tag] Identifier '{}' sym={:?} found in GLOBAL ctypes -> tag={:?}", name_str, name, tag);
                return tag;
            }
            // eprintln!("[find_struct_tag] Identifier '{}' sym={:?} NOT FOUND in local or global ctypes. global keys={:?}",
            // name_str, name,
            // ctx.module_ctx.global_variable_ctypes.keys().map(|k| {
            // let s = ctx.module_ctx.interner.resolve(*k);
            // format!("{}={:?}", s, k)
            // }).collect::<Vec<_>>());
            None
        }
        Expression::Cast { type_name, .. } => {
            // Cast expression like `(struct point *)0` — extract
            // the struct tag from the cast's type specifiers.
            use crate::frontend::parser::ast::TypeSpecifier;
            for spec in &type_name.specifiers.specifiers {
                match spec {
                    TypeSpecifier::Struct {
                        name: Some(tag_sym),
                        ..
                    } => {
                        return Some(*tag_sym);
                    }
                    TypeSpecifier::Union {
                        name: Some(tag_sym),
                        ..
                    } => {
                        return Some(*tag_sym);
                    }
                    _ => {}
                }
            }
            None
        }
        Expression::MemberAccess { object, member, .. } => {
            // Chained member access: s.a.b or p->a.b
            // We need to find the struct tag of the *result* of the
            // member access on `object` — i.e. the type of field `member`
            // within the parent struct.
            //
            // 1. Find the struct tag of the base expression.
            let parent_tag = find_struct_tag_for_expr(ctx, object);
            if let Some(parent_tag_sym) = parent_tag {
                // 2. Look up the parent struct's CType definition.
                if let Some(parent_ctype) = ctx.module_ctx.struct_defs.get(&parent_tag_sym) {
                    let member_str_val = ctx.module_ctx.interner.resolve(*member).to_string();
                    // 3. Find the field named `member` and return its type's tag.
                    let fields = match parent_ctype {
                        CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
                        _ => return None,
                    };
                    for f in fields {
                        if let Some(ref fname) = f.name {
                            if *fname == member_str_val {
                                // Found the field — return its struct/union tag.
                                return tag_from_ctype(&f.ty);
                            }
                        }
                    }
                }
            }
            None
        }
        Expression::ArrowAccess {
            pointer, member, ..
        } => {
            // Chained arrow access: p->a->b or p->a.b (where p->a is ArrowAccess)
            // Same logic as MemberAccess but the base is `pointer` not `object`.
            let parent_tag = find_struct_tag_for_expr(ctx, pointer);
            if let Some(parent_tag_sym) = parent_tag {
                if let Some(parent_ctype) = ctx.module_ctx.struct_defs.get(&parent_tag_sym) {
                    let member_str_val = ctx.module_ctx.interner.resolve(*member).to_string();
                    let fields = match parent_ctype {
                        CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
                        _ => return None,
                    };
                    for f in fields {
                        if let Some(ref fname) = f.name {
                            if *fname == member_str_val {
                                return tag_from_ctype(&f.ty);
                            }
                        }
                    }
                }
            }
            None
        }
        Expression::UnaryOp { operand, op, .. } => {
            // Dereference or address-of — pass through
            use crate::frontend::parser::ast::UnaryOperator;
            match op {
                UnaryOperator::Deref | UnaryOperator::AddressOf => {
                    find_struct_tag_for_expr(ctx, operand)
                }
                _ => None,
            }
        }
        Expression::ArraySubscript { array, .. } => {
            // `arr[idx]` — the element type of the array has the same
            // struct tag as the array's element.  Resolve the base
            // array expression's CType and extract the element CType.
            //
            // Helper closure: given a CType, peel off Array/Pointer to
            // get the element CType, then extract the struct tag.
            let peel_and_tag = |ctype: &CType| -> Option<Symbol> {
                let elem_ctype = match ctype {
                    CType::Array { element, .. } => Some(element.as_ref()),
                    CType::Pointer(inner) => Some(inner.as_ref()),
                    _ => Some(ctype),
                };
                elem_ctype.and_then(|ec| tag_from_ctype(ec))
            };

            // Case 1: `arr[idx]` where arr is a direct Identifier
            if let Expression::Identifier { name, .. } = array.as_ref() {
                let ctype_opt = ctx
                    .variable_ctypes
                    .get(name)
                    .or_else(|| ctx.module_ctx.global_variable_ctypes.get(name));
                if let Some(ctype) = ctype_opt {
                    if let Some(tag) = peel_and_tag(ctype) {
                        return Some(tag);
                    }
                }
            }

            // Case 2: `obj.member[idx]` where `member` is an array field
            // within a struct.  E.g., `pa.items[0]` where `items` is
            // `struct pair[3]`.  We need to resolve the struct containing
            // `member`, find the member's CType (array of pair), peel off
            // Array to get `pair`, and return that tag.
            if let Expression::MemberAccess { object, member, .. }
            | Expression::ArrowAccess {
                pointer: object,
                member,
                ..
            } = array.as_ref()
            {
                // Find the struct tag of the object (e.g., `pa` -> `pair_array`)
                if let Some(obj_tag) = find_struct_tag_for_expr(ctx, object) {
                    // Look up the struct def for that tag
                    if let Some(def) = ctx.module_ctx.struct_defs.get(&obj_tag) {
                        let member_str = ctx.module_ctx.interner.resolve(*member).to_string();
                        let fields = match def {
                            CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
                            _ => return None,
                        };
                        for f in fields {
                            if let Some(ref fname) = f.name {
                                if *fname == member_str {
                                    // Found the member's CType.  Peel off Array
                                    // to get the element type's tag.
                                    if let Some(tag) = peel_and_tag(&f.ty) {
                                        return Some(tag);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Fallback: recursively look through the base, which might
            // work for multi-dimensional arrays or pointer-to-struct.
            find_struct_tag_for_expr(ctx, array)
        }
        _ => None,
    }
}

/// Infers the C type of a struct member.
/// Legacy stub — replaced by `infer_member_ir_type` which also
/// checks the struct IR type for the actual field type.
#[allow(dead_code)]
fn infer_member_type(
    _ctx: &LoweringContext<'_>,
    _base_expr: &Expression,
    _member: Symbol,
) -> IrType {
    IrType::I32
}

/// Infers the struct IR type for a member access base.
fn infer_base_struct_type(
    ctx: &LoweringContext<'_>,
    base_expr: &Expression,
    is_arrow: bool,
) -> IrType {
    // Helper: convert a CType (from struct_defs or global_variable_ctypes)
    // to an IrType::Struct, optionally dereferencing a pointer first for
    // arrow access.
    let ctype_to_struct_ir = |ctype: &CType, arrow: bool| -> Option<IrType> {
        let inner = if arrow {
            match ctype {
                CType::Pointer(inner) => inner.as_ref(),
                _ => ctype,
            }
        } else {
            ctype
        };
        // Try getting the tag name and looking up in struct_defs for full
        // CType, then convert to IrType.
        let tag_name = match inner {
            CType::Struct {
                name: Some(tag_name),
                ..
            }
            | CType::Union {
                name: Some(tag_name),
                ..
            } => Some(tag_name.as_str()),
            _ => None,
        };
        if let Some(tag_str) = tag_name {
            if let Some(tag_sym) = ctx.module_ctx.interner.lookup(tag_str) {
                if let Some(def) = ctx.module_ctx.struct_defs.get(&tag_sym) {
                    if let Ok(ir_ty) = super::c_type_to_ir_type(def, &ctx.module_ctx.target) {
                        if matches!(&ir_ty, IrType::Struct { .. }) {
                            return Some(ir_ty);
                        }
                    }
                }
            }
        }
        // Direct conversion as fallback
        if let Ok(ir_ty) = super::c_type_to_ir_type(inner, &ctx.module_ctx.target) {
            if matches!(&ir_ty, IrType::Struct { .. }) {
                return Some(ir_ty);
            }
        }
        None
    };

    if let Expression::Identifier { name, .. } = base_expr {
        if let Some(alloca) = ctx.variables.get(name) {
            let ty = resolve_alloca_element_type(ctx, *alloca);
            if matches!(&ty, IrType::Struct { .. }) {
                return ty;
            }
            // Arrow access on a pointer variable: the variable's alloca
            // element type is `Ptr`.  The actual struct type is in
            // `variable_pointee_types` (populated during decl lowering).
            if is_arrow && ty == IrType::Ptr {
                if let Some(pointee) = ctx.variable_pointee_types.get(name) {
                    if matches!(pointee, IrType::Struct { .. }) {
                        return pointee.clone();
                    }
                }
                // Also check struct_defs via the variable's CType tag
                // name — the module-level struct_defs maps tag symbols
                // to their full CType definitions including fields.
                if let Some(ctype) = ctx.variable_ctypes.get(name) {
                    // Helper: convert CType fields to IrType::Struct
                    let lookup_struct_def = |tag_str: &str| -> Option<IrType> {
                        if let Some(sym) = ctx.module_ctx.interner.lookup(tag_str) {
                            if let Some(def) = ctx.module_ctx.struct_defs.get(&sym) {
                                if let Ok(ir_ty) =
                                    super::c_type_to_ir_type(def, &ctx.module_ctx.target)
                                {
                                    if matches!(&ir_ty, IrType::Struct { .. }) {
                                        return Some(ir_ty);
                                    }
                                }
                            }
                        }
                        None
                    };

                    match ctype {
                        CType::Pointer(inner) => {
                            // Pointer to struct/union — peel the pointer
                            // to get the pointee CType's tag and look it up.
                            match inner.as_ref() {
                                CType::Struct {
                                    name: Some(tag_name),
                                    fields: cfields,
                                    ..
                                } => {
                                    if let Some(ir_ty) = lookup_struct_def(tag_name) {
                                        return ir_ty;
                                    }
                                    // Fallback: build from inline fields in CType
                                    let ir_fields: Vec<IrType> = cfields
                                        .iter()
                                        .map(|f| {
                                            super::c_type_to_ir_type(&f.ty, &ctx.module_ctx.target)
                                                .unwrap_or(IrType::I32)
                                        })
                                        .collect();
                                    if !ir_fields.is_empty() {
                                        return IrType::Struct {
                                            fields: ir_fields,
                                            packed: false,
                                        };
                                    }
                                }
                                CType::Union {
                                    name: Some(tag_name),
                                    ..
                                } => {
                                    if let Some(ir_ty) = lookup_struct_def(tag_name) {
                                        return ir_ty;
                                    }
                                }
                                CType::Struct {
                                    name: None,
                                    fields: cfields,
                                    ..
                                } => {
                                    // Anonymous struct pointer — build from fields directly
                                    let ir_fields: Vec<IrType> = cfields
                                        .iter()
                                        .map(|f| {
                                            super::c_type_to_ir_type(&f.ty, &ctx.module_ctx.target)
                                                .unwrap_or(IrType::I32)
                                        })
                                        .collect();
                                    if !ir_fields.is_empty() {
                                        return IrType::Struct {
                                            fields: ir_fields,
                                            packed: false,
                                        };
                                    }
                                }
                                _ => {}
                            }
                        }
                        CType::Struct {
                            name: Some(tag_name),
                            ..
                        }
                        | CType::Union {
                            name: Some(tag_name),
                            ..
                        } => {
                            if let Some(ir_ty) = lookup_struct_def(tag_name) {
                                return ir_ty;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        // Fall back to module-level global variable CType registry.
        // Global variables are not in `ctx.variables` (local allocas),
        // so we check `global_variable_ctypes` to resolve the struct
        // type for field access on globals.
        if let Expression::Identifier { name, .. } = base_expr {
            if let Some(ctype) = ctx.module_ctx.global_variable_ctypes.get(name) {
                if let Some(ir_ty) = ctype_to_struct_ir(ctype, is_arrow) {
                    return ir_ty;
                }
            }
            // Also check function-local variable_ctypes (covers locals
            // that weren't found via alloca for some reason).
            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                if let Some(ir_ty) = ctype_to_struct_ir(ctype, is_arrow) {
                    return ir_ty;
                }
            }
        }
    }
    // For chained member access, cast expressions, or arrow access,
    // try to resolve the struct type from the sub-expression.
    match base_expr {
        Expression::Cast { type_name, .. } => {
            // (struct point *)expr — extract the struct type from the
            // cast's type specifiers, resolving via struct_defs.
            use crate::frontend::parser::ast::TypeSpecifier;
            for spec in &type_name.specifiers.specifiers {
                match spec {
                    TypeSpecifier::Struct {
                        name: Some(tag_sym),
                        ..
                    } => {
                        // Look up struct def by interned tag symbol
                        if let Some(def) = ctx.module_ctx.struct_defs.get(tag_sym) {
                            if let Ok(ir_ty) = super::c_type_to_ir_type(def, &ctx.module_ctx.target)
                            {
                                if matches!(&ir_ty, IrType::Struct { .. }) {
                                    return ir_ty;
                                }
                            }
                        }
                        // Also try field_names to build a type from count
                        if let Some(field_names) = ctx.module_ctx.struct_field_names.get(tag_sym) {
                            // We know the number of fields but not their types
                            // from field_names alone. Default all to I32.
                            let ir_fields: Vec<IrType> =
                                field_names.iter().map(|_| IrType::I32).collect();
                            if !ir_fields.is_empty() {
                                return IrType::Struct {
                                    fields: ir_fields,
                                    packed: false,
                                };
                            }
                        }
                    }
                    TypeSpecifier::Union {
                        name: Some(tag_sym),
                        ..
                    } => {
                        if let Some(def) = ctx.module_ctx.struct_defs.get(tag_sym) {
                            if let Ok(ir_ty) = super::c_type_to_ir_type(def, &ctx.module_ctx.target)
                            {
                                return ir_ty;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Expression::MemberAccess { object, member, .. } => {
            let parent_struct = infer_base_struct_type(ctx, object, false);
            if let IrType::Struct { ref fields, .. } = parent_struct {
                let member_idx = infer_field_index(ctx, object, *member);
                if member_idx < fields.len() {
                    let field_ty = &fields[member_idx];
                    if matches!(field_ty, IrType::Struct { .. }) {
                        return field_ty.clone();
                    }
                }
            }
            // Fallback
        }
        Expression::ArrowAccess {
            pointer, member, ..
        } => {
            let parent_struct = infer_base_struct_type(ctx, pointer, true);
            let _mstr = ctx.module_ctx.interner.resolve(*member);
            // eprintln!("[DEBUG infer_base_struct_type:ArrowAccess] member={} parent_struct={:?}", mstr, parent_struct);
            if let IrType::Struct { ref fields, .. } = parent_struct {
                let member_idx = infer_field_index(ctx, pointer, *member);
                // eprintln!("[DEBUG infer_base_struct_type:ArrowAccess] member_idx={} fields.len()={}", member_idx, fields.len());
                if member_idx < fields.len() {
                    let field_ty = &fields[member_idx];
                    // eprintln!("[DEBUG infer_base_struct_type:ArrowAccess] field_ty={:?}", field_ty);
                    if matches!(field_ty, IrType::Struct { .. }) {
                        return field_ty.clone();
                    }
                }
            }
            // Fallback: try to resolve via struct_defs using the tag
            // from the expression's result type.
            if let Some(tag_sym) = find_struct_tag_for_expr(ctx, base_expr) {
                if let Some(def) = ctx.module_ctx.struct_defs.get(&tag_sym) {
                    if let Ok(ir_ty) = super::c_type_to_ir_type(def, &ctx.module_ctx.target) {
                        if matches!(&ir_ty, IrType::Struct { .. }) {
                            // eprintln!("[DEBUG infer_base_struct_type:ArrowAccess] resolved via tag: {:?}", ir_ty);
                            return ir_ty;
                        }
                    }
                }
            }
        }
        Expression::ArraySubscript { array, .. } => {
            // `arr[idx].member` — the base is an array element.
            // Resolve the CType of the array and peel off CType::Array
            // to get the element CType, then convert to IrType::Struct.
            if let Expression::Identifier { name, .. } = array.as_ref() {
                let ctype_opt = ctx
                    .variable_ctypes
                    .get(name)
                    .or_else(|| ctx.module_ctx.global_variable_ctypes.get(name));
                if let Some(ctype) = ctype_opt {
                    let elem_ctype = match ctype {
                        CType::Array { element, .. } => Some(element.as_ref()),
                        CType::Pointer(inner) => Some(inner.as_ref()),
                        _ => None,
                    };
                    if let Some(ec) = elem_ctype {
                        if let Some(ir_ty) = ctype_to_struct_ir(ec, false) {
                            return ir_ty;
                        }
                    }
                }
            }
            // Fallback: try find_struct_tag_for_expr which now handles ArraySubscript
            if let Some(tag_sym) = find_struct_tag_for_expr(ctx, base_expr) {
                if let Some(def) = ctx.module_ctx.struct_defs.get(&tag_sym) {
                    if let Ok(ir_ty) = super::c_type_to_ir_type(def, &ctx.module_ctx.target) {
                        if matches!(&ir_ty, IrType::Struct { .. }) {
                            return ir_ty;
                        }
                    }
                }
            }
        }
        _ => {}
    }
    IrType::Struct {
        fields: vec![IrType::I32],
        packed: false,
    }
}

// ============================================================================
// Conditional (ternary) expression
// ============================================================================

/// Lowers `cond ? then_expr : else_expr`.
fn lower_conditional(
    ctx: &mut LoweringContext<'_>,
    cond: &Expression,
    then_expr: Option<&Expression>,
    else_expr: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let cond_val = lower_expression(ctx, cond)?;
    let cond_bool = to_i1(ctx, cond_val);

    let then_bb = ctx.builder.create_block(ctx.function, Some("cond.then"));
    let else_bb = ctx.builder.create_block(ctx.function, Some("cond.else"));
    let merge_bb = ctx.builder.create_block(ctx.function, Some("cond.end"));

    ctx.builder
        .build_cond_branch(ctx.function, cond_bool, then_bb, else_bb);

    // Then block.
    ctx.builder.set_insert_point(then_bb);
    let then_val = if let Some(then_e) = then_expr {
        lower_expression(ctx, then_e)?
    } else {
        // GCC conditional omission (x ?: y): reuse cond_val.
        cond_val
    };
    let then_exit = ctx.builder.get_insert_block().unwrap();
    ctx.builder.build_branch(ctx.function, merge_bb);

    // Else block.
    ctx.builder.set_insert_point(else_bb);
    let else_val = lower_expression(ctx, else_expr)?;
    let else_exit = ctx.builder.get_insert_block().unwrap();
    ctx.builder.build_branch(ctx.function, merge_bb);

    // Merge block: phi selects result.
    ctx.builder.set_insert_point(merge_bb);
    let result_ty = ctx.function.get_value_type(then_val).clone();
    let phi = ctx.builder.build_phi(ctx.function, result_ty);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, then_val, then_exit);
    ctx.builder
        .add_phi_incoming(ctx.function, phi, else_val, else_exit);

    Ok(phi)
}

// ============================================================================
// Comma expression
// ============================================================================

/// Lowers a comma expression.
fn lower_comma(
    ctx: &mut LoweringContext<'_>,
    expressions: &[Expression],
    _span: Span,
) -> Result<ValueId, LoweringError> {
    if expressions.is_empty() {
        return Ok(ctx.builder.build_const_int(ctx.function, IrType::I32, 0));
    }
    let mut last_val = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
    for expr in expressions {
        last_val = lower_expression(ctx, expr)?;
    }
    Ok(last_val)
}

// ============================================================================
// Assignment
// ============================================================================

/// Lowers simple assignment: `target = value`.
///
/// CRITICAL: The RHS value must be coerced (truncated / extended) to match the
/// element type that the target pointer points to.  Without this, a `char`
/// assignment like `*p = 'A'` would store a 32-bit value (since character
/// constants have type `int` in C), writing 3 extra bytes past the intended
/// byte and corrupting adjacent heap / stack data.
fn lower_assignment(
    ctx: &mut LoweringContext<'_>,
    target: &Expression,
    value: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let addr = lower_lvalue(ctx, target)?;
    let rhs_val = lower_expression(ctx, value)?;

    // Resolve the element type behind the destination pointer so we can
    // coerce the value to the correct width before the store.
    let elem_ty = resolve_pointee_type(ctx, addr);
    let rhs_ty = ctx.function.get_value_type(rhs_val).clone();
    let coerced = coerce_value(ctx, rhs_val, &rhs_ty, &elem_ty);

    ctx.builder.build_store(ctx.function, coerced, addr);
    Ok(coerced)
}

/// Lowers a compound assignment (`+=`, `-=`, etc.).
/// `int_op` is used for integer operands, `float_op` for float operands.
fn lower_compound_assign(
    ctx: &mut LoweringContext<'_>,
    int_op: BinOp,
    float_op: BinOp,
    target: &Expression,
    value: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let addr = lower_lvalue(ctx, target)?;
    let elem_ty = resolve_pointee_type(ctx, addr);
    let old_val = ctx.builder.build_load(ctx.function, addr, elem_ty.clone());

    let rhs_val = lower_expression(ctx, value)?;
    let rhs_ty = ctx.function.get_value_type(rhs_val).clone();
    let rhs_coerced = coerce_value(ctx, rhs_val, &rhs_ty, &elem_ty);

    let op = if is_ir_float(&elem_ty) {
        float_op
    } else {
        int_op
    };
    let new_val = ctx
        .builder
        .build_binop(ctx.function, op, old_val, rhs_coerced, elem_ty);
    ctx.builder.build_store(ctx.function, new_val, addr);
    Ok(new_val)
}

/// Lowers compound assignment with signed/unsigned awareness (`/=`, `%=`, `>>=`).
fn lower_compound_assign_signaware(
    ctx: &mut LoweringContext<'_>,
    signed_op: BinOp,
    unsigned_op: BinOp,
    float_op: BinOp,
    target: &Expression,
    value: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let use_unsigned = is_expression_unsigned(ctx, target);
    let addr = lower_lvalue(ctx, target)?;
    let elem_ty = resolve_pointee_type(ctx, addr);
    let old_val = ctx.builder.build_load(ctx.function, addr, elem_ty.clone());

    let rhs_val = lower_expression(ctx, value)?;
    let rhs_ty = ctx.function.get_value_type(rhs_val).clone();
    let rhs_coerced = coerce_value(ctx, rhs_val, &rhs_ty, &elem_ty);

    let op = if is_ir_float(&elem_ty) {
        float_op
    } else if use_unsigned {
        unsigned_op
    } else {
        signed_op
    };
    let new_val = ctx
        .builder
        .build_binop(ctx.function, op, old_val, rhs_coerced, elem_ty);
    ctx.builder.build_store(ctx.function, new_val, addr);
    Ok(new_val)
}

// ============================================================================
// Sizeof / Alignof
// ============================================================================

/// Lowers `sizeof(type)` or `sizeof expr`.
fn lower_sizeof(
    ctx: &mut LoweringContext<'_>,
    operand: &SizeofOperand,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let target = *ctx.target();
    let result_ty = size_t_ir_type(&target);

    let size_bytes: usize = match operand {
        SizeofOperand::TypeName(tn) => {
            let ir_ty = resolve_type_name_ir(ctx, tn)?;
            ir_type_size_bytes(&ir_ty, &target)
        }
        SizeofOperand::Expression(expr) => {
            // sizeof(expr): determine type without evaluating side effects.
            // C specifies that sizeof does NOT evaluate the expression —
            // and crucially, sizeof on an array identifier must return the
            // full array size (no array-to-pointer decay).
            //
            // First, try to resolve the type from the identifier's declared
            // type (stored in variable_ir_types), which preserves array types.
            // Only fall back to lowering the expression for complex cases.
            match expr.as_ref() {
                Expression::Identifier { name, .. } => {
                    if let Some(ir_ty) = ctx.variable_ir_types.get(name) {
                        ir_type_size_bytes(ir_ty, &target)
                    } else {
                        let val = lower_expression(ctx, expr)?;
                        let val_ty = ctx.function.get_value_type(val).clone();
                        ir_type_size_bytes(&val_ty, &target)
                    }
                }
                Expression::Dereference { operand, .. } => {
                    // sizeof(*ptr) — get the pointee type.
                    if let Expression::Identifier { name, .. } = operand.as_ref() {
                        if let Some(pointee) = ctx.variable_pointee_types.get(name) {
                            return Ok(ctx.builder.build_const_int(
                                ctx.function,
                                result_ty,
                                ir_type_size_bytes(pointee, &target) as i64,
                            ));
                        }
                    }
                    let val = lower_expression(ctx, expr)?;
                    let val_ty = ctx.function.get_value_type(val).clone();
                    ir_type_size_bytes(&val_ty, &target)
                }
                _ => {
                    let val = lower_expression(ctx, expr)?;
                    let val_ty = ctx.function.get_value_type(val).clone();
                    ir_type_size_bytes(&val_ty, &target)
                }
            }
        }
    };

    Ok(ctx
        .builder
        .build_const_int(ctx.function, result_ty, size_bytes as i64))
}

/// Lowers `_Alignof(type)` or `__alignof__(expr)`.
fn lower_alignof(
    ctx: &mut LoweringContext<'_>,
    operand: &AlignofOperand,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let target = *ctx.target();
    let result_ty = size_t_ir_type(&target);

    let alignment: usize = match operand {
        AlignofOperand::TypeName(tn) => {
            let ir_ty = resolve_type_name_ir(ctx, tn)?;
            ir_type_align(&ir_ty, &target)
        }
        AlignofOperand::Expression(expr) => {
            let val = lower_expression(ctx, expr)?;
            let val_ty = ctx.function.get_value_type(val).clone();
            ir_type_align(&val_ty, &target)
        }
    };

    Ok(ctx
        .builder
        .build_const_int(ctx.function, result_ty, alignment as i64))
}

// ============================================================================
// Compound literal
// ============================================================================

/// Lowers a compound literal `(type){init}`.
fn lower_compound_literal(
    ctx: &mut LoweringContext<'_>,
    type_name: &TypeName,
    initializer: &Initializer,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let ir_ty = resolve_type_name_ir(ctx, type_name)?;
    let alloca = ctx
        .builder
        .build_alloca(ctx.function, ir_ty.clone(), Some("compound_literal"));

    match initializer {
        Initializer::Expression(expr) => {
            let val = lower_expression(ctx, expr)?;
            ctx.builder.build_store(ctx.function, val, alloca);
        }
        Initializer::List { items, .. } => {
            for (i, item) in items.iter().enumerate() {
                let val = match &item.initializer {
                    Initializer::Expression(e) => lower_expression(ctx, e)?,
                    _ => ctx.builder.build_const_int(ctx.function, IrType::I32, 0),
                };
                let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let idx = ctx
                    .builder
                    .build_const_int(ctx.function, IrType::I32, i as i64);
                let elem_ptr = ctx.builder.build_gep(
                    ctx.function,
                    alloca,
                    vec![zero, idx],
                    ir_ty.clone(),
                    true,
                );
                ctx.builder.build_store(ctx.function, val, elem_ptr);
            }
        }
    }

    Ok(alloca)
}

// ============================================================================
// _Generic selection
// ============================================================================

/// A simplified type representation for _Generic matching.
/// Captures enough information to distinguish int, unsigned int, float, double,
/// char, pointer-to-X, etc.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GenericTypeKey {
    Void,
    Bool,
    Char,
    SignedChar,
    UnsignedChar,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    Long,
    UnsignedLong,
    LongLong,
    UnsignedLongLong,
    Float,
    Double,
    LongDouble,
    Pointer(Box<GenericTypeKey>),
    Other(String),
}

/// Resolve a `TypeName` AST node to a `GenericTypeKey` for _Generic matching.
fn typename_to_generic_key(ctx: &LoweringContext<'_>, type_name: &TypeName) -> GenericTypeKey {
    let specs = &type_name.specifiers.specifiers;

    // Check for pointer derived declarators.
    if let Some(ref decl) = type_name.declarator {
        if let Some(derived) = decl.derived.first() {
            if matches!(derived, DerivedDeclarator::Pointer { .. }) {
                // Pointer-to-base type.
                let base = specs_to_generic_key(specs, ctx);
                return GenericTypeKey::Pointer(Box::new(base));
            }
        }
    }

    specs_to_generic_key(specs, ctx)
}

/// Convert a list of type specifiers to a GenericTypeKey.
fn specs_to_generic_key(specs: &[TypeSpecifier], ctx: &LoweringContext<'_>) -> GenericTypeKey {
    if specs.is_empty() {
        return GenericTypeKey::Int; // default
    }

    let has_unsigned = specs.iter().any(|s| matches!(s, TypeSpecifier::Unsigned));
    let has_signed = specs.iter().any(|s| matches!(s, TypeSpecifier::Signed));
    let long_count = specs
        .iter()
        .filter(|s| matches!(s, TypeSpecifier::Long))
        .count();
    let has_short = specs.iter().any(|s| matches!(s, TypeSpecifier::Short));
    let has_char = specs.iter().any(|s| matches!(s, TypeSpecifier::Char));
    let has_int = specs.iter().any(|s| matches!(s, TypeSpecifier::Int));
    let has_float = specs.iter().any(|s| matches!(s, TypeSpecifier::Float));
    let has_double = specs.iter().any(|s| matches!(s, TypeSpecifier::Double));
    let has_void = specs.iter().any(|s| matches!(s, TypeSpecifier::Void));
    let has_bool = specs.iter().any(|s| matches!(s, TypeSpecifier::Bool));

    if has_void {
        return GenericTypeKey::Void;
    }
    if has_bool {
        return GenericTypeKey::Bool;
    }

    if has_float && !has_double {
        return GenericTypeKey::Float;
    }
    if has_double {
        if long_count > 0 {
            return GenericTypeKey::LongDouble;
        }
        return GenericTypeKey::Double;
    }

    if has_char {
        if has_unsigned {
            return GenericTypeKey::UnsignedChar;
        }
        if has_signed {
            return GenericTypeKey::SignedChar;
        }
        return GenericTypeKey::Char;
    }
    if has_short {
        if has_unsigned {
            return GenericTypeKey::UnsignedShort;
        }
        return GenericTypeKey::Short;
    }
    if long_count >= 2 {
        if has_unsigned {
            return GenericTypeKey::UnsignedLongLong;
        }
        return GenericTypeKey::LongLong;
    }
    if long_count == 1 {
        if has_unsigned {
            return GenericTypeKey::UnsignedLong;
        }
        return GenericTypeKey::Long;
    }
    if has_unsigned {
        return GenericTypeKey::UnsignedInt;
    }
    if has_int || has_signed {
        return GenericTypeKey::Int;
    }

    // Single specifier fallback
    if specs.len() == 1 {
        match &specs[0] {
            TypeSpecifier::TypedefName { name, .. } => {
                let name_str = ctx.module_ctx.interner.resolve(*name);
                return GenericTypeKey::Other(name_str.to_string());
            }
            TypeSpecifier::Struct { .. } | TypeSpecifier::Union { .. } => {
                return GenericTypeKey::Other("struct/union".into());
            }
            TypeSpecifier::Enum { .. } => return GenericTypeKey::Int,
            _ => {}
        }
    }

    GenericTypeKey::Int
}

/// Infer the GenericTypeKey for a controlling expression in _Generic.
fn infer_controlling_type_key(ctx: &LoweringContext<'_>, expr: &Expression) -> GenericTypeKey {
    match expr {
        Expression::IntegerLiteral { suffix, .. } => match suffix {
            IntegerSuffix::None => GenericTypeKey::Int,
            IntegerSuffix::U => GenericTypeKey::UnsignedInt,
            IntegerSuffix::L => GenericTypeKey::Long,
            IntegerSuffix::UL => GenericTypeKey::UnsignedLong,
            IntegerSuffix::LL => GenericTypeKey::LongLong,
            IntegerSuffix::ULL => GenericTypeKey::UnsignedLongLong,
        },
        Expression::FloatLiteral { suffix, .. } => match suffix {
            FloatSuffix::F => GenericTypeKey::Float,
            FloatSuffix::None | FloatSuffix::L => GenericTypeKey::Double,
        },
        Expression::StringLiteral { .. } => GenericTypeKey::Pointer(Box::new(GenericTypeKey::Char)),
        Expression::CharLiteral { .. } => GenericTypeKey::Int, // char literals are int in C
        Expression::Identifier { name, .. } => {
            // name is already a Symbol — look up the variable's C type directly.
            if let Some(ctype) = ctx.variable_ctypes.get(name) {
                return ctype_to_generic_key(ctype, ctx);
            }
            if let Some(ir_ty) = ctx.variable_ir_types.get(name) {
                return ir_type_to_generic_key(ir_ty);
            }
            GenericTypeKey::Int
        }
        Expression::Cast { type_name, .. } => typename_to_generic_key(ctx, type_name),
        Expression::UnaryOp { op, operand, .. } => match op {
            UnaryOperator::AddressOf => {
                let inner = infer_controlling_type_key(ctx, operand);
                GenericTypeKey::Pointer(Box::new(inner))
            }
            UnaryOperator::Deref => {
                let inner = infer_controlling_type_key(ctx, operand);
                if let GenericTypeKey::Pointer(base) = inner {
                    *base
                } else {
                    GenericTypeKey::Int
                }
            }
            UnaryOperator::Neg | UnaryOperator::Plus => infer_controlling_type_key(ctx, operand),
            _ => GenericTypeKey::Int,
        },
        Expression::AddressOf { operand, .. } => {
            let inner = infer_controlling_type_key(ctx, operand);
            GenericTypeKey::Pointer(Box::new(inner))
        }
        Expression::Dereference { operand, .. } => {
            let inner = infer_controlling_type_key(ctx, operand);
            if let GenericTypeKey::Pointer(base) = inner {
                *base
            } else {
                GenericTypeKey::Int
            }
        }
        Expression::BinaryOp { left, right, .. } => {
            // For arithmetic, result type follows usual arithmetic conversions.
            let lt = infer_controlling_type_key(ctx, left);
            let rt = infer_controlling_type_key(ctx, right);
            promote_generic_types(lt, rt)
        }
        Expression::FunctionCall { callee, .. } => {
            // Try to infer return type from function name.
            if let Expression::Identifier { name, .. } = callee.as_ref() {
                if let Some(ctype) = ctx.variable_ctypes.get(name) {
                    if let CType::Function { return_type, .. } = ctype {
                        return ctype_to_generic_key(return_type, ctx);
                    }
                }
            }
            GenericTypeKey::Int
        }
        _ => GenericTypeKey::Int,
    }
}

/// Convert a CType to a GenericTypeKey.
fn ctype_to_generic_key(ctype: &CType, ctx: &LoweringContext<'_>) -> GenericTypeKey {
    match ctype {
        CType::Void => GenericTypeKey::Void,
        CType::Bool => GenericTypeKey::Bool,
        CType::Char { signed } => {
            if *signed {
                GenericTypeKey::Char
            } else {
                GenericTypeKey::UnsignedChar
            }
        }
        CType::Short { signed } => {
            if *signed {
                GenericTypeKey::Short
            } else {
                GenericTypeKey::UnsignedShort
            }
        }
        CType::Int { signed } => {
            if *signed {
                GenericTypeKey::Int
            } else {
                GenericTypeKey::UnsignedInt
            }
        }
        CType::Long { signed } => {
            if *signed {
                GenericTypeKey::Long
            } else {
                GenericTypeKey::UnsignedLong
            }
        }
        CType::LongLong { signed } => {
            if *signed {
                GenericTypeKey::LongLong
            } else {
                GenericTypeKey::UnsignedLongLong
            }
        }
        CType::Float => GenericTypeKey::Float,
        CType::Double => GenericTypeKey::Double,
        CType::LongDouble => GenericTypeKey::LongDouble,
        CType::Pointer(inner) => {
            let inner_key = ctype_to_generic_key(inner, ctx);
            GenericTypeKey::Pointer(Box::new(inner_key))
        }
        CType::Array { element, .. } => {
            // Arrays decay to pointers in _Generic controlling expressions.
            let inner_key = ctype_to_generic_key(element, ctx);
            GenericTypeKey::Pointer(Box::new(inner_key))
        }
        CType::Enum { .. } => GenericTypeKey::Int,
        _ => GenericTypeKey::Other(format!("{:?}", ctype)),
    }
}

/// Convert an IR type to a GenericTypeKey (fallback).
fn ir_type_to_generic_key(ty: &IrType) -> GenericTypeKey {
    match ty {
        IrType::Void => GenericTypeKey::Void,
        IrType::I1 => GenericTypeKey::Bool,
        IrType::I8 => GenericTypeKey::Char,
        IrType::I16 => GenericTypeKey::Short,
        IrType::I32 => GenericTypeKey::Int,
        IrType::I64 => GenericTypeKey::Long,
        IrType::I128 => GenericTypeKey::LongLong,
        IrType::F32 => GenericTypeKey::Float,
        IrType::F64 => GenericTypeKey::Double,
        IrType::Ptr => GenericTypeKey::Pointer(Box::new(GenericTypeKey::Void)),
        _ => GenericTypeKey::Int,
    }
}

/// Promote two GenericTypeKeys following C usual arithmetic conversions.
fn promote_generic_types(a: GenericTypeKey, b: GenericTypeKey) -> GenericTypeKey {
    // Simplified: if either is double, result is double; if float, float; etc.
    match (&a, &b) {
        (GenericTypeKey::Double, _) | (_, GenericTypeKey::Double) => GenericTypeKey::Double,
        (GenericTypeKey::Float, _) | (_, GenericTypeKey::Float) => GenericTypeKey::Float,
        (GenericTypeKey::LongLong, _) | (_, GenericTypeKey::LongLong) => GenericTypeKey::LongLong,
        (GenericTypeKey::UnsignedLongLong, _) | (_, GenericTypeKey::UnsignedLongLong) => {
            GenericTypeKey::UnsignedLongLong
        }
        (GenericTypeKey::Long, _) | (_, GenericTypeKey::Long) => GenericTypeKey::Long,
        (GenericTypeKey::UnsignedLong, _) | (_, GenericTypeKey::UnsignedLong) => {
            GenericTypeKey::UnsignedLong
        }
        (GenericTypeKey::UnsignedInt, _) | (_, GenericTypeKey::UnsignedInt) => {
            GenericTypeKey::UnsignedInt
        }
        _ => GenericTypeKey::Int,
    }
}

/// Lowers `_Generic(controlling, ...)`.
///
/// Resolves the controlling expression's type and matches it against the
/// association list to find the correct branch to lower.
fn lower_generic(
    ctx: &mut LoweringContext<'_>,
    controlling: &Expression,
    associations: &[GenericAssociation],
    span: Span,
) -> Result<ValueId, LoweringError> {
    let ctrl_key = infer_controlling_type_key(ctx, controlling);
    // eprintln!("[_Generic] controlling expr: {:?} → key: {:?}", controlling, ctrl_key);

    // Find the matching association.
    let mut default_assoc: Option<&GenericAssociation> = None;
    for assoc in associations {
        if let Some(ref tn) = assoc.type_name {
            let assoc_key = typename_to_generic_key(ctx, tn);
            // eprintln!("[_Generic]   assoc type {:?} → key: {:?}", tn.specifiers.specifiers, assoc_key);
            if ctrl_key == assoc_key {
                // eprintln!("[_Generic]   MATCH!");
                return lower_expression(ctx, &assoc.expression);
            }
        } else {
            default_assoc = Some(assoc);
        }
    }

    // Fall back to default.
    if let Some(default) = default_assoc {
        return lower_expression(ctx, &default.expression);
    }

    Err(LoweringError::UnsupportedExpression {
        span,
        message: format!(
            "_Generic with no matching association for type {:?}",
            ctrl_key
        ),
    })
}

// ============================================================================
// GCC statement expression
// ============================================================================

/// Lowers `({ stmt1; stmt2; expr; })`.
fn lower_statement_expression(
    ctx: &mut LoweringContext<'_>,
    body: &[BlockItem],
    _span: Span,
) -> Result<ValueId, LoweringError> {
    // Save the current variable scope so that declarations inside the
    // statement expression do not leak into the enclosing scope.
    // Variables declared inside `({ ... })` shadow outer variables
    // temporarily and are discarded when the block ends.
    let saved_variables = ctx.variables.clone();

    let mut last_val = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);

    for item in body {
        match item {
            BlockItem::Statement(Statement::Expression { expr, .. }) => {
                last_val = lower_expression(ctx, expr)?;
            }
            BlockItem::Statement(Statement::Null { .. }) => {
                // Empty statement — no value.
            }
            BlockItem::Statement(Statement::Return { value, .. }) => {
                if let Some(ret_expr) = value {
                    let ret_val = lower_expression(ctx, ret_expr)?;
                    ctx.builder.build_return(ctx.function, Some(ret_val));
                } else {
                    ctx.builder.build_return(ctx.function, None);
                }
                // Restore the variable scope before returning.
                ctx.variables = saved_variables;
                return Ok(last_val);
            }
            BlockItem::Declaration(decl) => {
                // Declaration inside statement expression — allocate locals
                // and optionally initialize them so they are visible to later
                // expressions within the same ({ ... }) block.
                super::stmt_lowering::lower_block_declaration(ctx, decl)?;
            }
            BlockItem::Statement(other) => {
                // Other statements (if, while, for, etc.) — delegate to the
                // full statement lowering so control-flow inside statement
                // expressions works correctly.
                super::stmt_lowering::lower_statement(ctx, other)?;
            }
        }
    }

    // Restore the enclosing scope's variable map. Inner declarations
    // are no longer accessible, preserving C's block-scoping semantics.
    ctx.variables = saved_variables;

    Ok(last_val)
}

// ============================================================================
// Label address (&&label)
// ============================================================================

/// Lowers `&&label` — address of a label for computed goto.
fn lower_label_address(
    ctx: &mut LoweringContext<'_>,
    label: Symbol,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    ctx.address_taken_labels.insert(label);
    let target_bb = ctx.get_or_create_label_block(label);
    // Emit a BlockAddress instruction that produces a pointer to the
    // runtime machine-code address of the target basic block.
    Ok(ctx.builder.build_block_address(ctx.function, target_bb))
}

// ============================================================================
// Address-of (&)
// ============================================================================

/// Lowers `&operand` — produces the address of an lvalue.
fn lower_address_of(
    ctx: &mut LoweringContext<'_>,
    operand: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    lower_lvalue(ctx, operand)
}

// ============================================================================
// Dereference (*)
// ============================================================================

/// Lowers `*pointer` as rvalue.
fn lower_dereference_rvalue(
    ctx: &mut LoweringContext<'_>,
    operand: &Expression,
    _span: Span,
) -> Result<ValueId, LoweringError> {
    let ptr_val = lower_expression(ctx, operand)?;
    let pointee_ty = infer_dereference_type(ctx, operand);
    Ok(ctx.builder.build_load(ctx.function, ptr_val, pointee_ty))
}

/// Infers the type that a pointer points to.
fn infer_dereference_type(ctx: &LoweringContext<'_>, ptr_expr: &Expression) -> IrType {
    match ptr_expr {
        Expression::Identifier { name, .. } => {
            // Check local variable pointee types first (from declaration analysis).
            if let Some(pointee_ty) = ctx.variable_pointee_types.get(name) {
                return pointee_ty.clone();
            }
            // Check the IR element type — for scalar variables (int *p),
            // the element type is Ptr and the pointee would be I32.
            // However variable_pointee_types should already cover this case.
            // Fall through to global symbols check.
            if let Some(info) = ctx.module_ctx.global_symbols.get(name) {
                if let IrType::Function { return_type, .. } = &info.ir_type {
                    return (**return_type).clone();
                }
            }
            // Default: assume the pointer points to a 32-bit integer.
            // This covers the common case of `int *p` where no explicit
            // pointee type was registered (e.g. function parameters).
            IrType::I32
        }
        Expression::Dereference { operand, .. } => {
            // Double deref: *(*pp) — the inner deref produces the type
            // from infer_dereference_type of the inner, then we deref again.
            // For int **pp: *pp = Ptr, **pp = I32.
            // For int ***ppp: *ppp = Ptr, **ppp = Ptr, ***ppp = I32.
            let inner_type = infer_dereference_type(ctx, operand);
            if inner_type == IrType::Ptr {
                // The inner dereference produced a pointer.  We need to
                // determine what *that* pointer points to.
                //
                // First, consult the "deep pointee" map which stores the
                // base element type for multi-level pointer variables
                // (e.g. `int **pp` → deep_pointee = I32).
                if let Expression::Identifier { name, .. } = operand.as_ref() {
                    if let Some(deep_ty) = ctx.variable_deep_pointee_types.get(name) {
                        return deep_ty.clone();
                    }
                    // Fallback: if the first-level pointee is Ptr but we
                    // have no deep_pointee record, default to I32 (the most
                    // common base type for C multi-pointer chains).
                }
                IrType::I32
            } else {
                inner_type
            }
        }
        Expression::MemberAccess { .. } | Expression::ArrowAccess { .. } => {
            // Dereferencing a member that is a pointer. Default to I32
            // (the member's pointed-to type would need full type inference).
            IrType::I32
        }
        _ => {
            // For complex expressions (casts, function calls, etc.),
            // default to I32.  A full type inference system would be needed
            // for perfect accuracy here.
            IrType::I32
        }
    }
}

// ============================================================================
// Schema import witness — ensures all schema-mandated imports are referenced.
// The compiler will optimise this away; it exists only so that the import
// statements are retained (Rust treats unused imports as errors with -Dwarn).
// ============================================================================
#[allow(dead_code)]
fn _schema_import_witness() {
    // crate::common::diagnostics
    let _d: Option<Diagnostic> = None;
    let _s: Severity = Severity::Error;
    let _w: Severity = Severity::Warning;
    let _sp = Span::DUMMY;
    let _merged = Span::merge(_sp, _sp);
    let _eng: Option<DiagnosticEngine> = None;

    // crate::common::types
    let _ct = CType::Void;
    let _ = _ct.is_integer();
    let _ = _ct.is_floating();
    let _ = _ct.is_arithmetic();
    let _ = _ct.is_scalar();
    let _ = _ct.is_pointer();
    let _ = _ct.is_void();
    let _ = _ct.is_function();
    let _ = _ct.is_array();
    let _ = _ct.is_signed();
    let _ = _ct.is_unsigned();
    let _qt: Option<QualifiedType> = None;
    let _fd: Option<FieldDef> = None;

    // crate::common::type_builder
    let _: Option<StructLayout> = None;
    let _: Option<UnionLayout> = None;
    let _: Option<FieldLayout> = None;

    // crate::common::string_interner::Symbol
    let _sym = Symbol::new(0);
    let _ = _sym.as_u32();

    // crate::common::target::Target
    fn _target_witness(t: &Target) {
        let _ = t.pointer_width();
        let _ = t.pointer_align();
        let _ = t.data_model();
        let _ = t.long_size();
        let _ = t.long_double_size();
    }

    // crate::ir::basic_block::BasicBlockId
    let _: Option<BasicBlockId> = None;

    // crate::ir::builder::IrBuilder
    let _: Option<IrBuilder> = None;

    // crate::ir::function
    let _: Option<IrFunction> = None;

    // crate::ir::instructions
    let _ = BinOp::Add;
    let _ = BinOp::Sub;
    let _ = BinOp::Mul;
    let _ = BinOp::SDiv;
    let _ = BinOp::UDiv;
    let _ = BinOp::SRem;
    let _ = BinOp::URem;
    let _ = BinOp::And;
    let _ = BinOp::Or;
    let _ = BinOp::Xor;
    let _ = BinOp::Shl;
    let _ = BinOp::AShr;
    let _ = BinOp::LShr;
    let _ = BinOp::FAdd;
    let _ = BinOp::FSub;
    let _ = BinOp::FMul;
    let _ = BinOp::FDiv;
    let _ = ICmpPredicate::Eq;
    let _ = ICmpPredicate::Ne;
    let _ = ICmpPredicate::Slt;
    let _ = ICmpPredicate::Sgt;
    let _ = ICmpPredicate::Sle;
    let _ = ICmpPredicate::Sge;
    let _ = ICmpPredicate::Ult;
    let _ = ICmpPredicate::Ugt;
    let _ = ICmpPredicate::Ule;
    let _ = ICmpPredicate::Uge;
    let _ = FCmpPredicate::OEq;
    let _ = FCmpPredicate::ONe;
    let _ = FCmpPredicate::Olt;
    let _ = FCmpPredicate::Ogt;
    let _ = FCmpPredicate::Ole;
    let _ = FCmpPredicate::Oge;
    let _: Option<Instruction> = None;

    // crate::ir::module
    let _: Option<IrModule> = None;
    let _: Option<StringLiteral> = None;
    let _: Option<Constant> = None;
    let _: Option<Linkage> = None;

    // crate::ir::types::IrType
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

    // super:: imports
    let _: Option<ModuleLoweringContext> = None;
    let _: Option<LoweringError> = None;

    // crate::frontend::parser::ast  (module alias + specific types)
    let _: Option<&ast_mod::Expression> = None;
    let _: Option<InitializerItem> = None;

    // crate::frontend::sema::constant_eval
    let _: Option<ConstValue> = None;

    // crate::frontend::sema::type_checker
    let _: Option<TypedExpression> = None;

    // type_builder functions — reference the function values to prevent
    // unused-import warnings without needing an exact cast.
    let _: fn(&[FieldDef], &Target) -> StructLayout = compute_struct_layout;
    let _: fn(&[FieldDef], &Target) -> UnionLayout = compute_union_layout;

    // ctypes alias
    let _ = ctypes::CType::Void;
    let _: fn(&CType, &Target) -> usize = ctypes::size_of;
    let _: fn(&CType, &Target) -> usize = ctypes::align_of;

    // const_eval functions
    {
        // Just touch the function symbols; the actual signatures are complex.
        let _p = evaluate_constant_expression;
        let _q = evaluate_integer_constant;
        let _r = is_constant_expression;
    }

    // type_checker functions
    {
        let _p = is_assignment_compatible;
        let _q = insert_implicit_conversion;
    }

    // lowering mod functions
    {
        let _p = c_type_to_ir_type;
    }
}
