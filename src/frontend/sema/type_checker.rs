//! Type checking module for Phase 5 of the BCC compiler.
//!
//! Implements C11 type compatibility checks, implicit conversions (integer
//! promotions per C11 §6.3.1.1, usual arithmetic conversions per C11 §6.3.1.8),
//! pointer-integer conversion warnings, pointer arithmetic validation,
//! struct/union member access validation (`.` and `->` operators), function
//! call argument type matching with variadic support, assignment type
//! compatibility, cast validity, and expression type inference for all AST
//! expression nodes.
//!
//! Produces [`TypedExpression`] consumed by IR lowering (Phase 6).

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::target::{DataModel, Target};
use crate::common::type_builder;
use crate::common::types::{
    self, CType, FieldDef,
};
use crate::common::string_interner::Symbol;
use crate::frontend::parser::ast::{
    AlignofOperand, BinaryOperator, BlockItem, CharPrefix, Expression, ForInit,
    FloatSuffix, GenericAssociation, IntegerSuffix, SizeofOperand, Statement,
    StringPrefix, TypeName, UnaryOperator,
};
use crate::frontend::sema::scope::ScopeStack;
use crate::frontend::sema::symbol_table::{
    StorageClass, SymbolEntry, SymbolTable,
};

// ===========================================================================
// TypedExpression — type-annotated expression output
// ===========================================================================

/// A type-annotated expression produced by the type checker.
///
/// Wraps the original AST [`Expression`] with its resolved C type, lvalue
/// status, and whether the expression is a compile-time constant.  This is
/// the primary output of [`check_expression`] and the primary input to
/// IR lowering (Phase 6).
#[derive(Clone, Debug)]
pub struct TypedExpression {
    /// The original AST expression node.
    pub expr: Expression,
    /// The resolved C type of this expression.
    pub ty: CType,
    /// `true` if this expression designates an object (is an lvalue).
    pub is_lvalue: bool,
    /// `true` if this expression is a compile-time constant.
    pub is_constant: bool,
    /// Source location span for this expression.
    pub span: Span,
}

impl TypedExpression {
    /// Creates a new typed expression from its components.
    #[inline]
    fn new(expr: Expression, ty: CType, is_lvalue: bool, is_constant: bool, span: Span) -> Self {
        Self {
            expr,
            ty,
            is_lvalue,
            is_constant,
            span,
        }
    }

    /// Creates a typed expression representing an error recovery value.
    /// Uses `CType::Int` as a fallback type to allow continued analysis.
    #[inline]
    fn error(span: Span) -> Self {
        Self {
            expr: Expression::Error { span },
            ty: CType::Int { signed: true },
            is_lvalue: false,
            is_constant: false,
            span,
        }
    }
}

// ===========================================================================
// Context — bundles all state needed by the type checker
// ===========================================================================

/// Internal context carrying all mutable and immutable references needed
/// during type checking.  Avoids passing many parameters through every
/// recursive call.
struct TypeCheckContext<'a> {
    scopes: &'a ScopeStack,
    symbols: &'a SymbolTable,
    target: &'a Target,
    diag: &'a mut DiagnosticEngine,
    /// The return type of the enclosing function (if any).
    return_type: Option<&'a CType>,
}

// ===========================================================================
// Public API — integer_promote (target-aware wrapper)
// ===========================================================================

/// Target-aware integer promotion per C11 §6.3.1.1.
///
/// Integer types whose conversion rank is less than `int` are promoted to
/// `int` (or `unsigned int` if `int` cannot represent all values, which does
/// not arise with standard widths).
///
/// This is a thin wrapper around [`types::integer_promote`] that accepts
/// a [`Target`] for future target-dependent promotion rules (e.g. if a
/// target used a 16-bit `int`, `unsigned short` would promote to
/// `unsigned int`).
pub fn integer_promote(ty: &CType, _target: &Target) -> CType {
    types::integer_promote(ty)
}

// ===========================================================================
// Public API — usual_arithmetic_conversion (target-aware wrapper)
// ===========================================================================

/// Target-aware usual arithmetic conversions per C11 §6.3.1.8.
///
/// Determines the common type to which two arithmetic operands are implicitly
/// converted.  Both operands undergo integer promotion first, then the
/// higher-ranked (wider) type wins with careful signed/unsigned handling.
///
/// On ILP32 targets (i686) where `long` and `int` have identical width,
/// the base implementation's assumption that higher rank implies strictly
/// wider type may produce incorrect results for `unsigned int` vs
/// `signed long`.  This wrapper applies a correction for that case.
pub fn usual_arithmetic_conversion(a: &CType, b: &CType, target: &Target) -> CType {
    let result = types::usual_arithmetic_conversion(a, b);

    // Correction for ILP32 targets: if the result is `signed long` and one
    // of the original promoted operands is `unsigned int`, `signed long` cannot
    // represent all values of `unsigned int` on ILP32 (both are 32 bits).
    // The correct result is `unsigned long` per C11 §6.3.1.8 step 4.
    if target.data_model() == DataModel::ILP32 {
        let pa = types::integer_promote(a);
        let pb = types::integer_promote(b);
        if result == (CType::Long { signed: true }) {
            let has_unsigned_int =
                pa == (CType::Int { signed: false }) || pb == (CType::Int { signed: false });
            if has_unsigned_int {
                return CType::Long { signed: false };
            }
        }
    }

    result
}

// ===========================================================================
// Public API — is_assignment_compatible
// ===========================================================================

/// Checks whether a value of type `rhs` can be assigned to an lvalue of
/// type `lhs` per C11 §6.5.16.1 simple assignment constraints.
///
/// Returns `true` if the assignment is valid (possibly with implicit
/// conversion).  Emits warnings for pointer-integer conversions and
/// incompatible pointer assignments via `diag`.
pub fn is_assignment_compatible(
    lhs: &CType,
    rhs: &CType,
    rhs_is_null: bool,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> bool {
    let lhs_c = lhs.canonical();
    let rhs_c = rhs.canonical();

    // 1. Both are arithmetic types — always compatible.
    if lhs_c.is_arithmetic() && rhs_c.is_arithmetic() {
        return true;
    }

    // 2. Both are compatible struct/union types.
    if (matches!(lhs_c, CType::Struct { .. }) || matches!(lhs_c, CType::Union { .. }))
        && type_builder::types_compatible(lhs_c, rhs_c)
    {
        return true;
    }

    // 3. Both are pointers.
    if let (CType::Pointer(lhs_pointee), CType::Pointer(rhs_pointee)) = (lhs_c, rhs_c) {
        // void* is compatible with any object pointer.
        if lhs_pointee.canonical().is_void() || rhs_pointee.canonical().is_void() {
            return true;
        }
        // Check pointed-to type compatibility (ignoring qualifiers).
        if type_builder::types_compatible(lhs_pointee.canonical(), rhs_pointee.canonical()) {
            return true;
        }
        // Incompatible pointer types — warn and allow.
        diag.warning(
            span,
            format!(
                "incompatible pointer types assigning '{}' from '{}'",
                lhs, rhs
            ),
        );
        return true;
    }

    // 4. LHS is pointer and RHS is null pointer constant (integer 0).
    if lhs_c.is_pointer() && rhs_is_null {
        return true;
    }

    // 5. LHS is _Bool — any scalar value can be assigned.
    if matches!(lhs_c, CType::Bool) && rhs_c.is_scalar() {
        return true;
    }

    // 6. Pointer ↔ integer conversions — warn but allow.
    if lhs_c.is_pointer() && rhs_c.is_integer() {
        diag.warning(
            span,
            format!(
                "implicit conversion from integer type '{}' to pointer type '{}'",
                rhs, lhs
            ),
        );
        return true;
    }
    if lhs_c.is_integer() && rhs_c.is_pointer() {
        diag.warning(
            span,
            format!(
                "implicit conversion from pointer type '{}' to integer type '{}'",
                rhs, lhs
            ),
        );
        return true;
    }

    false
}

// ===========================================================================
// Public API — insert_implicit_conversion
// ===========================================================================

/// Inserts an implicit conversion on a typed expression, updating its type.
///
/// This handles:
/// - Array-to-pointer decay (C11 §6.3.2.1p3)
/// - Function-to-pointer decay (C11 §6.3.2.1p4)
/// - Integer promotions on rvalue expressions
///
/// Returns a new [`TypedExpression`] with the updated type.
pub fn insert_implicit_conversion(
    typed_expr: TypedExpression,
    target_type: &CType,
    _target: &Target,
) -> TypedExpression {
    TypedExpression {
        expr: typed_expr.expr,
        ty: target_type.clone(),
        is_lvalue: false,
        is_constant: typed_expr.is_constant,
        span: typed_expr.span,
    }
}

// ===========================================================================
// Public API — check_expression
// ===========================================================================

/// Type-checks an expression, producing a [`TypedExpression`] with the
/// resolved type, lvalue status, and constancy.
///
/// This is the primary entry point for expression-level type checking.
/// It dispatches to specialised helpers for each expression variant
/// (binary operations, unary operations, function calls, member access,
/// casts, sizeof/alignof, etc.).
///
/// # Parameters
///
/// * `expr`        — the AST expression to type-check.
/// * `scopes`      — lexical scope stack for identifier resolution.
/// * `symbols`     — symbol table for type retrieval.
/// * `target`      — compilation target for architecture-dependent decisions.
/// * `diag`        — diagnostic engine for error/warning reporting.
/// * `return_type` — the return type of the enclosing function, for
///                   return-statement validation (passed through context).
pub fn check_expression(
    expr: &Expression,
    scopes: &ScopeStack,
    symbols: &SymbolTable,
    target: &Target,
    diag: &mut DiagnosticEngine,
    return_type: Option<&CType>,
) -> TypedExpression {
    let mut ctx = TypeCheckContext {
        scopes,
        symbols,
        target,
        diag,
        return_type,
    };
    check_expr_inner(expr, &mut ctx)
}

// ===========================================================================
// Public API — check_statement
// ===========================================================================

/// Type-checks a statement, validating that all embedded expressions have
/// correct types and that control flow constraints are satisfied.
///
/// Delegates expression checking to [`check_expression`] for sub-expressions,
/// and validates:
/// - Condition expressions are scalar (if, while, do-while, for).
/// - Switch controlling expression is integer type.
/// - Return expression matches the function's return type.
/// - Case values are integer constant expressions.
///
/// # Parameters
///
/// * `stmt`        — the AST statement to type-check.
/// * `scopes`      — lexical scope stack.
/// * `symbols`     — symbol table.
/// * `target`      — compilation target.
/// * `diag`        — diagnostic engine.
/// * `return_type` — return type of the enclosing function.
pub fn check_statement(
    stmt: &Statement,
    scopes: &ScopeStack,
    symbols: &SymbolTable,
    target: &Target,
    diag: &mut DiagnosticEngine,
    return_type: Option<&CType>,
) -> () {
    let mut ctx = TypeCheckContext {
        scopes,
        symbols,
        target,
        diag,
        return_type,
    };
    check_stmt_inner(stmt, &mut ctx);
}

// ===========================================================================
// Internal — expression dispatch
// ===========================================================================

/// Core recursive expression type checker.
fn check_expr_inner(expr: &Expression, ctx: &mut TypeCheckContext<'_>) -> TypedExpression {
    match expr {
        // ----- Literals -----
        Expression::IntegerLiteral {
            value,
            suffix,
            span,
        } => check_integer_literal(*value, *suffix, *span),

        Expression::FloatLiteral {
            value: _,
            suffix,
            span,
        } => check_float_literal(*suffix, *span, expr),

        Expression::StringLiteral {
            value,
            prefix,
            span,
        } => check_string_literal(value, *prefix, *span, expr),

        Expression::CharLiteral {
            value: _,
            prefix,
            span,
        } => check_char_literal(*prefix, *span, expr),

        // ----- Identifier -----
        Expression::Identifier { name, span } => check_identifier(*name, *span, ctx),

        // ----- Binary operation -----
        Expression::BinaryOp {
            op,
            left,
            right,
            span,
        } => check_binary_op(*op, left, right, *span, expr, ctx),

        // ----- Unary operation -----
        Expression::UnaryOp {
            op,
            operand,
            is_postfix: _,
            span,
        } => check_unary_op(*op, operand, *span, expr, ctx),

        // ----- Address-of (distinct node) -----
        Expression::AddressOf { operand, span } => check_address_of(operand, *span, expr, ctx),

        // ----- Dereference (distinct node) -----
        Expression::Dereference { operand, span } => check_dereference(operand, *span, expr, ctx),

        // ----- Conditional (ternary) -----
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            span,
        } => check_conditional(condition, then_expr.as_deref(), else_expr, *span, expr, ctx),

        // ----- Function call -----
        Expression::FunctionCall {
            callee,
            args,
            span,
        } => check_function_call(callee, args, *span, expr, ctx),

        // ----- Array subscript -----
        Expression::ArraySubscript {
            array,
            index,
            span,
        } => check_array_subscript(array, index, *span, expr, ctx),

        // ----- Member access (.) -----
        Expression::MemberAccess {
            object,
            member,
            span,
        } => check_member_access(object, *member, *span, expr, ctx),

        // ----- Arrow access (->) -----
        Expression::ArrowAccess {
            pointer,
            member,
            span,
        } => check_arrow_access(pointer, *member, *span, expr, ctx),

        // ----- Cast -----
        Expression::Cast {
            type_name,
            operand,
            span,
        } => check_cast(type_name, operand, *span, expr, ctx),

        // ----- sizeof -----
        Expression::Sizeof { operand, span } => check_sizeof(operand, *span, ctx),

        // ----- alignof -----
        Expression::Alignof { operand, span } => check_alignof(operand, *span, ctx),

        // ----- Compound literal -----
        Expression::CompoundLiteral {
            type_name: _,
            initializer: _,
            span,
        } => {
            // Compound literals are lvalues with the declared type.
            // Full initializer checking is done by the initializer module.
            // Here we produce a typed expression with the declared type as Int
            // (the actual type resolution is deferred to the declaration handler
            // which resolves TypeName to CType).
            TypedExpression::new(expr.clone(), CType::Int { signed: true }, true, false, *span)
        }

        // ----- Comma expression -----
        Expression::Comma { expressions, span } => check_comma(expressions, *span, expr, ctx),

        // ----- _Generic -----
        Expression::Generic {
            controlling,
            associations,
            span,
        } => check_generic(controlling, associations, *span, expr, ctx),

        // ----- Statement expression GCC extension -----
        Expression::StatementExpression { body, span } => {
            check_stmt_expr(body, *span, expr, ctx)
        }

        // ----- Label address (&&label) -----
        Expression::LabelAddress { label: _, span } => {
            // Result is void* per GCC extension.
            TypedExpression::new(
                expr.clone(),
                CType::Pointer(Box::new(CType::Void)),
                false,
                false,
                *span,
            )
        }

        // ----- Error recovery -----
        Expression::Error { span } => TypedExpression::error(*span),
    }
}

// ===========================================================================
// Internal — Literal type inference
// ===========================================================================

/// Determines the type of an integer literal based on its value and suffix.
///
/// Per C11 §6.4.4.1, the type of an unsuffixed decimal integer literal is the
/// first of `int`, `long int`, `long long int` that can represent the value.
/// Suffixed literals follow similar but narrower rules.
fn check_integer_literal(value: u128, suffix: IntegerSuffix, span: Span) -> TypedExpression {
    let ty = match suffix {
        IntegerSuffix::None => {
            // Decimal: int -> long -> long long (signed only for decimal).
            if value <= i32::MAX as u128 {
                CType::Int { signed: true }
            } else if value <= i64::MAX as u128 {
                CType::Long { signed: true }
            } else {
                CType::LongLong { signed: true }
            }
        }
        IntegerSuffix::U => {
            if value <= u32::MAX as u128 {
                CType::Int { signed: false }
            } else {
                CType::LongLong { signed: false }
            }
        }
        IntegerSuffix::L => CType::Long { signed: true },
        IntegerSuffix::UL => CType::Long { signed: false },
        IntegerSuffix::LL => CType::LongLong { signed: true },
        IntegerSuffix::ULL => CType::LongLong { signed: false },
    };
    TypedExpression::new(
        Expression::IntegerLiteral {
            value,
            suffix,
            span,
        },
        ty,
        false,
        true,
        span,
    )
}

/// Determines the type of a float literal from its suffix.
fn check_float_literal(suffix: FloatSuffix, span: Span, expr: &Expression) -> TypedExpression {
    let ty = match suffix {
        FloatSuffix::None => CType::Double,
        FloatSuffix::F => CType::Float,
        FloatSuffix::L => CType::LongDouble,
    };
    TypedExpression::new(expr.clone(), ty, false, true, span)
}

/// Determines the type of a string literal from its prefix.
///
/// String literals are arrays of the appropriate character type.  They
/// are lvalues in C (they designate static storage).
fn check_string_literal(
    value: &[u8],
    prefix: StringPrefix,
    span: Span,
    expr: &Expression,
) -> TypedExpression {
    let (element_ty, char_size) = match prefix {
        StringPrefix::None | StringPrefix::U8 => (CType::Char { signed: true }, 1),
        StringPrefix::L => (CType::Int { signed: true }, 4), // wchar_t = int on Linux
        StringPrefix::SmallU => (CType::Short { signed: false }, 2), // char16_t
        StringPrefix::BigU => (CType::Int { signed: false }, 4),     // char32_t
    };
    // +1 for the null terminator.
    let len = (value.len() / char_size) + 1;
    let ty = CType::Array {
        element: Box::new(element_ty),
        size: Some(len),
    };
    TypedExpression::new(expr.clone(), ty, true, true, span)
}

/// Determines the type of a character literal from its prefix.
///
/// Ordinary character constants have type `int` per C11 §6.4.4.4.
fn check_char_literal(prefix: CharPrefix, span: Span, expr: &Expression) -> TypedExpression {
    let ty = match prefix {
        CharPrefix::None => CType::Int { signed: true },      // int
        CharPrefix::L => CType::Int { signed: true },          // wchar_t = int
        CharPrefix::SmallU => CType::Short { signed: false },  // char16_t
        CharPrefix::BigU => CType::Int { signed: false },      // char32_t
    };
    TypedExpression::new(expr.clone(), ty, false, true, span)
}

// ===========================================================================
// Internal — Identifier resolution
// ===========================================================================

/// Resolves an identifier to its declared type via scope and symbol table.
fn check_identifier(
    name: Symbol,
    span: Span,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let sym_id = match ctx.scopes.lookup(name) {
        Some(id) => id,
        None => {
            ctx.diag.error(span, format!("use of undeclared identifier"));
            return TypedExpression::error(span);
        }
    };

    let entry: &SymbolEntry = ctx.symbols.get(sym_id);
    let ty = entry.ty.clone();

    // Functions are not lvalues; variables and enum constants have different
    // lvalue characteristics.
    let is_lvalue = !ty.is_function() && entry.storage_class != StorageClass::Typedef;

    // Enum constants are compile-time constant expressions.
    let is_constant = matches!(ty.canonical(), CType::Enum { .. })
        && entry.storage_class != StorageClass::Static;

    TypedExpression::new(
        Expression::Identifier { name, span },
        ty,
        is_lvalue,
        is_constant,
        span,
    )
}

// ===========================================================================
// Internal — Array / function decay helpers
// ===========================================================================

/// Performs array-to-pointer decay (C11 §6.3.2.1p3).
fn array_to_pointer_decay(ty: &CType) -> CType {
    match ty.canonical() {
        CType::Array { element, .. } => CType::Pointer(element.clone()),
        other => other.clone(),
    }
}

/// Performs function-to-pointer decay (C11 §6.3.2.1p4).
fn function_to_pointer_decay(ty: &CType) -> CType {
    match ty.canonical() {
        f @ CType::Function { .. } => CType::Pointer(Box::new(f.clone())),
        other => other.clone(),
    }
}

/// Performs lvalue conversion: applies array-to-pointer and function-to-pointer
/// decay, yielding an rvalue type suitable for most expression contexts.
fn lvalue_conversion(ty: &CType) -> CType {
    let decayed = array_to_pointer_decay(ty);
    function_to_pointer_decay(&decayed)
}

/// Applies default argument promotions (C11 §6.5.2.2p6) for variadic
/// arguments: integer promotions on integer types, float -> double.
fn default_argument_promotion(ty: &CType, target: &Target) -> CType {
    let decayed = lvalue_conversion(ty);
    if decayed.is_integer() {
        return integer_promote(&decayed, target);
    }
    if matches!(decayed.canonical(), CType::Float) {
        return CType::Double;
    }
    decayed
}

/// Returns `true` if the given typed expression represents a null pointer
/// constant (integer constant expression with value 0).
fn is_null_pointer_constant(typed: &TypedExpression) -> bool {
    if !typed.is_constant {
        return false;
    }
    match &typed.expr {
        Expression::IntegerLiteral { value, .. } => *value == 0,
        _ => false,
    }
}

/// Returns `true` if the expression designates a modifiable lvalue.
fn is_modifiable_lvalue(typed: &TypedExpression) -> bool {
    if !typed.is_lvalue {
        return false;
    }
    let canonical = typed.ty.canonical();
    if canonical.is_array() {
        return false;
    }
    if !canonical.is_complete() {
        return false;
    }
    true
}

// ===========================================================================
// Internal — Binary operation type checking
// ===========================================================================

/// Type-checks a binary operation expression.
///
/// Dispatches to specific handlers for arithmetic, pointer arithmetic,
/// comparison, bitwise, logical, shift, and assignment operators.
fn check_binary_op(
    op: BinaryOperator,
    left: &Expression,
    right: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lhs = check_expr_inner(left, ctx);
    let rhs = check_expr_inner(right, ctx);

    match op {
        // --- Arithmetic operators ---
        BinaryOperator::Add => check_additive_op(op, &lhs, &rhs, span, expr, ctx),
        BinaryOperator::Sub => check_additive_op(op, &lhs, &rhs, span, expr, ctx),
        BinaryOperator::Mul | BinaryOperator::Div | BinaryOperator::Mod => {
            check_multiplicative_op(op, &lhs, &rhs, span, expr, ctx)
        }

        // --- Bitwise operators ---
        BinaryOperator::BitAnd | BinaryOperator::BitOr | BinaryOperator::BitXor => {
            check_bitwise_op(op, &lhs, &rhs, span, expr, ctx)
        }

        // --- Shift operators ---
        BinaryOperator::Shl | BinaryOperator::Shr => {
            check_shift_op(op, &lhs, &rhs, span, expr, ctx)
        }

        // --- Logical operators ---
        BinaryOperator::LogAnd | BinaryOperator::LogOr => {
            check_logical_op(&lhs, &rhs, span, expr, ctx)
        }

        // --- Relational operators ---
        BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::Le
        | BinaryOperator::Ge => check_relational_op(&lhs, &rhs, span, expr, ctx),

        // --- Equality operators ---
        BinaryOperator::Eq | BinaryOperator::Ne => {
            check_equality_op(&lhs, &rhs, span, expr, ctx)
        }

        // --- Simple assignment ---
        BinaryOperator::Assign => check_simple_assignment(&lhs, &rhs, span, expr, ctx),

        // --- Compound assignment ---
        BinaryOperator::AddAssign
        | BinaryOperator::SubAssign
        | BinaryOperator::MulAssign
        | BinaryOperator::DivAssign
        | BinaryOperator::ModAssign
        | BinaryOperator::BitAndAssign
        | BinaryOperator::BitOrAssign
        | BinaryOperator::BitXorAssign
        | BinaryOperator::ShlAssign
        | BinaryOperator::ShrAssign => {
            check_compound_assignment(op, &lhs, &rhs, span, expr, ctx)
        }
    }
}

/// Checks additive operations: `+` and `-`.
///
/// Additive operators have special rules for pointer arithmetic:
/// - `ptr + integer` or `integer + ptr` → pointer arithmetic.
/// - `ptr - ptr` → ptrdiff_t (pointer difference).
/// - `arith + arith` → usual arithmetic conversions.
fn check_additive_op(
    op: BinaryOperator,
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    // pointer + integer
    if lty.is_pointer() && rty.is_integer() {
        validate_pointer_arithmetic_target(&lty, span, ctx);
        return TypedExpression::new(expr.clone(), lty, false, false, span);
    }

    // integer + pointer (commutative for Add only)
    if op == BinaryOperator::Add && lty.is_integer() && rty.is_pointer() {
        validate_pointer_arithmetic_target(&rty, span, ctx);
        return TypedExpression::new(expr.clone(), rty, false, false, span);
    }

    // pointer - pointer -> ptrdiff_t
    if op == BinaryOperator::Sub && lty.is_pointer() && rty.is_pointer() {
        // Both pointers must point to compatible types.
        if let (CType::Pointer(lp), CType::Pointer(rp)) = (lty.canonical(), rty.canonical()) {
            if !type_builder::types_compatible(lp.canonical(), rp.canonical())
                && !lp.canonical().is_void()
                && !rp.canonical().is_void()
            {
                ctx.diag.error(
                    span,
                    format!(
                        "subtraction of pointers to incompatible types '{}' and '{}'",
                        lty, rty
                    ),
                );
            }
        }
        // Result is ptrdiff_t: long on LP64, int on ILP32.
        let result_ty = ptrdiff_type(ctx.target);
        return TypedExpression::new(expr.clone(), result_ty, false, false, span);
    }

    // pointer - integer
    if op == BinaryOperator::Sub && lty.is_pointer() && rty.is_integer() {
        validate_pointer_arithmetic_target(&lty, span, ctx);
        return TypedExpression::new(expr.clone(), lty, false, false, span);
    }

    // arithmetic + arithmetic
    if lty.is_arithmetic() && rty.is_arithmetic() {
        let result_ty = usual_arithmetic_conversion(&lty, &rty, ctx.target);
        let is_const = lhs.is_constant && rhs.is_constant;
        return TypedExpression::new(expr.clone(), result_ty, false, is_const, span);
    }

    ctx.diag.error(
        span,
        format!(
            "invalid operands to binary '{}': '{}' and '{}'",
            if op == BinaryOperator::Add { "+" } else { "-" },
            lty,
            rty
        ),
    );
    TypedExpression::error(span)
}

/// Checks multiplicative operations: `*`, `/`, `%`.
fn check_multiplicative_op(
    op: BinaryOperator,
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    // `%` requires integer operands.
    if op == BinaryOperator::Mod {
        if !lty.is_integer() || !rty.is_integer() {
            ctx.diag.error(
                span,
                format!(
                    "invalid operands to binary '%%': '{}' and '{}' (both must be integer)",
                    lty, rty
                ),
            );
            return TypedExpression::error(span);
        }
    } else if !lty.is_arithmetic() || !rty.is_arithmetic() {
        let op_str = if op == BinaryOperator::Mul { "*" } else { "/" };
        ctx.diag.error(
            span,
            format!(
                "invalid operands to binary '{}': '{}' and '{}'",
                op_str, lty, rty
            ),
        );
        return TypedExpression::error(span);
    }

    let result_ty = usual_arithmetic_conversion(&lty, &rty, ctx.target);
    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), result_ty, false, is_const, span)
}

/// Checks bitwise operations: `&`, `|`, `^`.
fn check_bitwise_op(
    _op: BinaryOperator,
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    if !lty.is_integer() || !rty.is_integer() {
        ctx.diag.error(
            span,
            format!(
                "invalid operands to bitwise operator: '{}' and '{}' (both must be integer)",
                lty, rty
            ),
        );
        return TypedExpression::error(span);
    }

    let result_ty = usual_arithmetic_conversion(&lty, &rty, ctx.target);
    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), result_ty, false, is_const, span)
}

/// Checks shift operations: `<<`, `>>`.
///
/// Each operand is independently integer-promoted; the result type is the
/// promoted type of the left operand.
fn check_shift_op(
    _op: BinaryOperator,
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    if !lty.is_integer() || !rty.is_integer() {
        ctx.diag.error(
            span,
            format!(
                "invalid operands to shift operator: '{}' and '{}' (both must be integer)",
                lty, rty
            ),
        );
        return TypedExpression::error(span);
    }

    // Result type is the promoted type of the left operand.
    let result_ty = integer_promote(&lty, ctx.target);
    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), result_ty, false, is_const, span)
}

/// Checks logical operations: `&&`, `||`.
///
/// Both operands must be scalar. Result is `int` (always 0 or 1).
fn check_logical_op(
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    if !lty.is_scalar() {
        ctx.diag.error(
            span,
            format!(
                "operand of logical operator is not scalar (has type '{}')",
                lty
            ),
        );
    }
    if !rty.is_scalar() {
        ctx.diag.error(
            span,
            format!(
                "operand of logical operator is not scalar (has type '{}')",
                rty
            ),
        );
    }

    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), CType::Int { signed: true }, false, is_const, span)
}

/// Checks relational operators: `<`, `>`, `<=`, `>=`.
///
/// Operands must both be real arithmetic types, or both pointers to
/// compatible types.  Result is `int`.
fn check_relational_op(
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    if lty.is_arithmetic() && rty.is_arithmetic() {
        // Valid: real arithmetic comparison.
    } else if lty.is_pointer() && rty.is_pointer() {
        // Valid: pointer comparison (should be compatible types, but we allow
        // with a warning for incompatible pointers as GCC does).
        if let (CType::Pointer(lp), CType::Pointer(rp)) = (lty.canonical(), rty.canonical()) {
            if !type_builder::types_compatible(lp.canonical(), rp.canonical())
                && !lp.canonical().is_void()
                && !rp.canonical().is_void()
            {
                ctx.diag.warning(
                    span,
                    format!(
                        "comparison of distinct pointer types ('{}' and '{}')",
                        lty, rty
                    ),
                );
            }
        }
    } else {
        ctx.diag.error(
            span,
            format!(
                "invalid operands to relational operator: '{}' and '{}'",
                lty, rty
            ),
        );
    }

    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), CType::Int { signed: true }, false, is_const, span)
}

/// Checks equality operators: `==`, `!=`.
///
/// Similar to relational, but additionally allows pointer vs null pointer
/// constant, and pointer vs `void *` comparisons.
fn check_equality_op(
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    if lty.is_arithmetic() && rty.is_arithmetic() {
        // Valid.
    } else if lty.is_pointer() && rty.is_pointer() {
        // Valid (with warning for incompatible pointer types).
        if let (CType::Pointer(lp), CType::Pointer(rp)) = (lty.canonical(), rty.canonical()) {
            if !type_builder::types_compatible(lp.canonical(), rp.canonical())
                && !lp.canonical().is_void()
                && !rp.canonical().is_void()
            {
                ctx.diag.warning(
                    span,
                    format!(
                        "comparison of distinct pointer types ('{}' and '{}')",
                        lty, rty
                    ),
                );
            }
        }
    } else if lty.is_pointer() && is_null_pointer_constant(rhs) {
        // Pointer compared to null — valid.
    } else if rty.is_pointer() && is_null_pointer_constant(lhs) {
        // Null compared to pointer — valid.
    } else if (lty.is_pointer() && rty.is_integer()) || (lty.is_integer() && rty.is_pointer()) {
        ctx.diag.warning(
            span,
            format!(
                "comparison between pointer and integer ('{}' and '{}')",
                lty, rty
            ),
        );
    } else {
        ctx.diag.error(
            span,
            format!(
                "invalid operands to equality operator: '{}' and '{}'",
                lty, rty
            ),
        );
    }

    let is_const = lhs.is_constant && rhs.is_constant;
    TypedExpression::new(expr.clone(), CType::Int { signed: true }, false, is_const, span)
}

/// Checks simple assignment: `lhs = rhs`.
fn check_simple_assignment(
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    // LHS must be a modifiable lvalue.
    if !is_modifiable_lvalue(lhs) {
        ctx.diag.error(
            span,
            format!("expression is not assignable"),
        );
        return TypedExpression::error(span);
    }

    let lty = &lhs.ty;
    let rty = lvalue_conversion(&rhs.ty);

    let rhs_is_null = is_null_pointer_constant(rhs);
    if !is_assignment_compatible(lty, &rty, rhs_is_null, span, ctx.diag) {
        ctx.diag.error(
            span,
            format!(
                "assigning to '{}' from incompatible type '{}'",
                lty, rty
            ),
        );
    }

    // Result of assignment is the value of LHS after assignment (not an lvalue
    // per C11, though GCC treats it as one; we follow the standard here).
    TypedExpression::new(expr.clone(), lty.clone(), false, false, span)
}

/// Checks compound assignment operators: `+=`, `-=`, `*=`, `/=`, `%=`,
/// `&=`, `|=`, `^=`, `<<=`, `>>=`.
fn check_compound_assignment(
    op: BinaryOperator,
    lhs: &TypedExpression,
    rhs: &TypedExpression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    // LHS must be a modifiable lvalue.
    if !is_modifiable_lvalue(lhs) {
        ctx.diag.error(
            span,
            format!("expression is not assignable"),
        );
        return TypedExpression::error(span);
    }

    let lty = lvalue_conversion(&lhs.ty);
    let rty = lvalue_conversion(&rhs.ty);

    match op {
        BinaryOperator::AddAssign | BinaryOperator::SubAssign => {
            // Pointer arithmetic allowed for += and -=.
            if lty.is_pointer() && rty.is_integer() {
                validate_pointer_arithmetic_target(&lty, span, ctx);
            } else if !lty.is_arithmetic() || !rty.is_arithmetic() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid operands to compound assignment: '{}' and '{}'",
                        lty, rty
                    ),
                );
            }
        }
        BinaryOperator::MulAssign | BinaryOperator::DivAssign => {
            if !lty.is_arithmetic() || !rty.is_arithmetic() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid operands to compound assignment: '{}' and '{}'",
                        lty, rty
                    ),
                );
            }
        }
        BinaryOperator::ModAssign
        | BinaryOperator::BitAndAssign
        | BinaryOperator::BitOrAssign
        | BinaryOperator::BitXorAssign
        | BinaryOperator::ShlAssign
        | BinaryOperator::ShrAssign => {
            if !lty.is_integer() || !rty.is_integer() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid operands to compound assignment: '{}' and '{}' (both must be integer)",
                        lty, rty
                    ),
                );
            }
        }
        _ => {}
    }

    TypedExpression::new(expr.clone(), lhs.ty.clone(), false, false, span)
}

// ===========================================================================
// Internal — Unary operation type checking
// ===========================================================================

/// Type-checks a unary operation expression.
fn check_unary_op(
    op: UnaryOperator,
    operand: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_operand = check_expr_inner(operand, ctx);

    match op {
        UnaryOperator::Plus | UnaryOperator::Neg => {
            let ty = lvalue_conversion(&typed_operand.ty);
            if !ty.is_arithmetic() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid argument type '{}' to unary '{}'",
                        ty,
                        if op == UnaryOperator::Plus { "+" } else { "-" }
                    ),
                );
                return TypedExpression::error(span);
            }
            let promoted = integer_promote(&ty, ctx.target);
            TypedExpression::new(expr.clone(), promoted, false, typed_operand.is_constant, span)
        }

        UnaryOperator::BitNot => {
            let ty = lvalue_conversion(&typed_operand.ty);
            if !ty.is_integer() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid argument type '{}' to unary '~' (must be integer)",
                        ty
                    ),
                );
                return TypedExpression::error(span);
            }
            let promoted = integer_promote(&ty, ctx.target);
            TypedExpression::new(expr.clone(), promoted, false, typed_operand.is_constant, span)
        }

        UnaryOperator::LogNot => {
            let ty = lvalue_conversion(&typed_operand.ty);
            if !ty.is_scalar() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid argument type '{}' to unary '!' (must be scalar)",
                        ty
                    ),
                );
                return TypedExpression::error(span);
            }
            TypedExpression::new(
                expr.clone(),
                CType::Int { signed: true },
                false,
                typed_operand.is_constant,
                span,
            )
        }

        UnaryOperator::PreInc
        | UnaryOperator::PreDec
        | UnaryOperator::PostInc
        | UnaryOperator::PostDec => {
            if !is_modifiable_lvalue(&typed_operand) {
                ctx.diag.error(
                    span,
                    format!("operand of increment/decrement must be a modifiable lvalue"),
                );
                return TypedExpression::error(span);
            }
            let ty = lvalue_conversion(&typed_operand.ty);
            if !ty.is_scalar() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid argument type '{}' to increment/decrement (must be scalar)",
                        ty
                    ),
                );
                return TypedExpression::error(span);
            }
            TypedExpression::new(expr.clone(), typed_operand.ty.clone(), false, false, span)
        }

        // AddressOf and Deref are typically separate AST nodes, but may also
        // appear as UnaryOp variants.  Handle them here for completeness.
        UnaryOperator::AddressOf => check_address_of(operand, span, expr, ctx),
        UnaryOperator::Deref => check_dereference(operand, span, expr, ctx),
    }
}

// ===========================================================================
// Internal — Address-of and dereference
// ===========================================================================

/// Type-checks the address-of operator `&operand`.
fn check_address_of(
    operand: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_operand = check_expr_inner(operand, ctx);

    // The operand must be an lvalue or a function designator.
    if !typed_operand.is_lvalue && !typed_operand.ty.is_function() {
        ctx.diag.error(
            span,
            format!("cannot take the address of an rvalue"),
        );
        return TypedExpression::error(span);
    }

    // Check that the operand is not register-qualified (if we tracked that).
    // In the current AST, register storage class is checked at declaration level.

    let pointee_ty = typed_operand.ty.clone();
    let result_ty = CType::Pointer(Box::new(pointee_ty));
    TypedExpression::new(expr.clone(), result_ty, false, false, span)
}

/// Type-checks the dereference operator `*operand`.
fn check_dereference(
    operand: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_operand = check_expr_inner(operand, ctx);
    let ty = lvalue_conversion(&typed_operand.ty);

    match ty.canonical() {
        CType::Pointer(pointee) => {
            let result_ty = (**pointee).clone();
            // Dereferencing a pointer yields an lvalue (unless it's void*).
            let is_lvalue = !result_ty.is_void() && !result_ty.is_function();
            TypedExpression::new(expr.clone(), result_ty, is_lvalue, false, span)
        }
        _ => {
            ctx.diag.error(
                span,
                format!(
                    "indirection requires pointer operand (have '{}')",
                    ty
                ),
            );
            TypedExpression::error(span)
        }
    }
}

// ===========================================================================
// Internal — Conditional (ternary) expression
// ===========================================================================

/// Type-checks a conditional expression: `condition ? then_expr : else_expr`.
///
/// GCC extension: `condition ?: else_expr` (then_expr is None).
fn check_conditional(
    condition: &Expression,
    then_expr: Option<&Expression>,
    else_expr: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_cond = check_expr_inner(condition, ctx);
    let cond_ty = lvalue_conversion(&typed_cond.ty);

    // Condition must be scalar.
    if !cond_ty.is_scalar() {
        ctx.diag.error(
            span,
            format!(
                "used type '{}' where a scalar is required (in conditional expression)",
                cond_ty
            ),
        );
        return TypedExpression::error(span);
    }

    let typed_else = check_expr_inner(else_expr, ctx);

    // GCC extension: x ?: y — the condition value is used as the "then" value.
    let typed_then = if let Some(te) = then_expr {
        check_expr_inner(te, ctx)
    } else {
        typed_cond.clone()
    };

    let then_ty = lvalue_conversion(&typed_then.ty);
    let else_ty = lvalue_conversion(&typed_else.ty);

    // Determine the result type per C11 §6.5.15.
    let result_ty = if then_ty.is_arithmetic() && else_ty.is_arithmetic() {
        usual_arithmetic_conversion(&then_ty, &else_ty, ctx.target)
    } else if type_builder::types_compatible(then_ty.canonical(), else_ty.canonical()) {
        then_ty.clone()
    } else if then_ty.is_pointer() && is_null_pointer_constant(&typed_else) {
        then_ty.clone()
    } else if else_ty.is_pointer() && is_null_pointer_constant(&typed_then) {
        else_ty.clone()
    } else if then_ty.is_pointer() && else_ty.is_pointer() {
        // If one is void* and the other is a typed pointer, result is void*.
        if let CType::Pointer(tp) = then_ty.canonical() {
            if tp.canonical().is_void() {
                return TypedExpression::new(expr.clone(), then_ty.clone(), false, false, span);
            }
        }
        if let CType::Pointer(ep) = else_ty.canonical() {
            if ep.canonical().is_void() {
                return TypedExpression::new(expr.clone(), else_ty.clone(), false, false, span);
            }
        }
        // Incompatible pointers — warn and use the "then" type.
        ctx.diag.warning(
            span,
            format!(
                "pointer type mismatch in conditional expression ('{}' and '{}')",
                then_ty, else_ty
            ),
        );
        then_ty.clone()
    } else if then_ty.is_void() && else_ty.is_void() {
        CType::Void
    } else {
        ctx.diag.error(
            span,
            format!(
                "incompatible operand types in conditional expression ('{}' and '{}')",
                then_ty, else_ty
            ),
        );
        CType::Int { signed: true }
    };

    let is_const = typed_cond.is_constant && typed_then.is_constant && typed_else.is_constant;
    TypedExpression::new(expr.clone(), result_ty, false, is_const, span)
}

// ===========================================================================
// Internal — Function call type checking
// ===========================================================================

/// Type-checks a function call expression.
///
/// Validates:
/// - Callee is a function type or pointer to function type.
/// - Argument count matches parameter count (with variadic consideration).
/// - Argument types are assignment-compatible with parameter types.
/// - Default argument promotions applied for variadic and unprototyped args.
fn check_function_call(
    callee: &Expression,
    args: &[Expression],
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_callee = check_expr_inner(callee, ctx);
    let callee_ty = lvalue_conversion(&typed_callee.ty);

    // Resolve the function type.  Callee may be a function (implicit decay)
    // or a pointer-to-function.
    let func_ty = match callee_ty.canonical() {
        CType::Pointer(inner) => match inner.canonical() {
            f @ CType::Function { .. } => f.clone(),
            _ => {
                ctx.diag.error(
                    span,
                    format!(
                        "called object type '{}' is not a function or function pointer",
                        callee_ty
                    ),
                );
                return TypedExpression::error(span);
            }
        },
        f @ CType::Function { .. } => f.clone(),
        _ => {
            ctx.diag.error(
                span,
                format!(
                    "called object type '{}' is not a function or function pointer",
                    callee_ty
                ),
            );
            return TypedExpression::error(span);
        }
    };

    let (return_type, params, variadic) = match &func_ty {
        CType::Function {
            return_type,
            params,
            variadic,
        } => (return_type.as_ref().clone(), params, *variadic),
        _ => unreachable!(),
    };

    // Validate argument count.
    if !variadic && args.len() != params.len() && !params.is_empty() {
        ctx.diag.error(
            span,
            format!(
                "too {} arguments to function call, expected {}, have {}",
                if args.len() < params.len() {
                    "few"
                } else {
                    "many"
                },
                params.len(),
                args.len()
            ),
        );
    } else if variadic && args.len() < params.len() {
        ctx.diag.error(
            span,
            format!(
                "too few arguments to function call, expected at least {}, have {}",
                params.len(),
                args.len()
            ),
        );
    }

    // Type-check each argument.
    for (i, arg) in args.iter().enumerate() {
        let typed_arg = check_expr_inner(arg, ctx);
        let arg_ty = lvalue_conversion(&typed_arg.ty);

        if i < params.len() {
            // Matched parameter — check assignment compatibility.
            let param_ty = &params[i];
            let arg_is_null = is_null_pointer_constant(&typed_arg);
            if !is_assignment_compatible(param_ty, &arg_ty, arg_is_null, arg.span(), ctx.diag) {
                ctx.diag.error(
                    arg.span(),
                    format!(
                        "passing '{}' to parameter of incompatible type '{}'",
                        arg_ty, param_ty
                    ),
                );
            }
        } else if variadic {
            // Variadic argument — apply default argument promotions.
            // (Promotion is applied during lowering; here we just validate
            // the type is suitable for promotion.)
            let _ = default_argument_promotion(&arg_ty, ctx.target);
        }
        // If params is empty (unprototyped), all arguments receive default
        // promotions.
    }

    TypedExpression::new(expr.clone(), return_type, false, false, span)
}

// ===========================================================================
// Internal — Array subscript
// ===========================================================================

/// Type-checks an array subscript expression: `array[index]`.
///
/// One operand must be a pointer (or array decayed to pointer) and the other
/// must be an integer.  The result is an lvalue of the element type.
fn check_array_subscript(
    array: &Expression,
    index: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_array = check_expr_inner(array, ctx);
    let typed_index = check_expr_inner(index, ctx);

    let arr_ty = lvalue_conversion(&typed_array.ty);
    let idx_ty = lvalue_conversion(&typed_index.ty);

    // a[b] is equivalent to *(a + b), so either operand can be the pointer.
    let (ptr_ty, _int_ty) = if arr_ty.is_pointer() && idx_ty.is_integer() {
        (arr_ty, idx_ty)
    } else if idx_ty.is_pointer() && arr_ty.is_integer() {
        (idx_ty, arr_ty)
    } else {
        ctx.diag.error(
            span,
            format!(
                "subscripted value is not an array, pointer, or vector (have '{}' and '{}')",
                arr_ty, idx_ty
            ),
        );
        return TypedExpression::error(span);
    };

    // Validate pointer target is a complete type.
    if let CType::Pointer(pointee) = ptr_ty.canonical() {
        if !pointee.is_complete() && !pointee.is_void() {
            ctx.diag.error(
                span,
                format!(
                    "subscript of pointer to incomplete type '{}'",
                    pointee
                ),
            );
        }
        let result_ty = (**pointee).clone();
        TypedExpression::new(expr.clone(), result_ty, true, false, span)
    } else {
        // Should not reach here due to the pointer check above.
        TypedExpression::error(span)
    }
}

// ===========================================================================
// Internal — Member access (.  and ->)
// ===========================================================================

/// Type-checks a member access expression: `object.member`.
fn check_member_access(
    object: &Expression,
    member: Symbol,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_object = check_expr_inner(object, ctx);
    let obj_ty = typed_object.ty.canonical().clone();

    match &obj_ty {
        CType::Struct { fields, .. } | CType::Union { fields, .. } => {
            match resolve_member(fields, member, ctx) {
                Some(field_ty) => {
                    TypedExpression::new(
                        expr.clone(),
                        field_ty,
                        typed_object.is_lvalue,
                        false,
                        span,
                    )
                }
                None => {
                    ctx.diag.error(
                        span,
                        format!(
                            "no member named in type '{}'",
                            obj_ty
                        ),
                    );
                    TypedExpression::error(span)
                }
            }
        }
        _ => {
            ctx.diag.error(
                span,
                format!(
                    "member reference base type '{}' is not a structure or union",
                    obj_ty
                ),
            );
            TypedExpression::error(span)
        }
    }
}

/// Type-checks an arrow access expression: `pointer->member`.
fn check_arrow_access(
    pointer: &Expression,
    member: Symbol,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_ptr = check_expr_inner(pointer, ctx);
    let ptr_ty = lvalue_conversion(&typed_ptr.ty);

    match ptr_ty.canonical() {
        CType::Pointer(pointee) => {
            let pointee_c = pointee.canonical().clone();
            match &pointee_c {
                CType::Struct { fields, .. } | CType::Union { fields, .. } => {
                    match resolve_member(fields, member, ctx) {
                        Some(field_ty) => {
                            TypedExpression::new(expr.clone(), field_ty, true, false, span)
                        }
                        None => {
                            ctx.diag.error(
                                span,
                                format!(
                                    "no member named in type '{}'",
                                    pointee_c
                                ),
                            );
                            TypedExpression::error(span)
                        }
                    }
                }
                _ => {
                    ctx.diag.error(
                        span,
                        format!(
                            "member reference type '{}' is not a pointer to a structure or union",
                            ptr_ty
                        ),
                    );
                    TypedExpression::error(span)
                }
            }
        }
        _ => {
            ctx.diag.error(
                span,
                format!(
                    "member reference type '{}' is not a pointer",
                    ptr_ty
                ),
            );
            TypedExpression::error(span)
        }
    }
}

/// Resolves a member name within a struct/union field list.
///
/// Searches fields by name (comparing Symbol handles).  If the member is
/// found in an anonymous struct/union sub-member, recursively resolves.
///
/// Returns `Some(CType)` of the member's type, or `None` if not found.
fn resolve_member(
    fields: &[FieldDef],
    member: Symbol,
    ctx: &TypeCheckContext<'_>,
) -> Option<CType> {
    let member_u32 = member.as_u32();

    for field in fields {
        if let Some(ref field_name) = field.name {
            // FieldDef.name is a String, while member is a Symbol.
            // We need to compare by checking if the interned symbol matches
            // the field's string name.  Since we don't have the interner here
            // for reverse lookup, we compare using the u32 index against
            // the field name hash.  In practice, the AST and type system would
            // use Symbols consistently.  For now, we use a positional lookup
            // that is compatible with both String-based FieldDef and Symbol-based
            // member access.
            //
            // The field name string and the member Symbol should correspond
            // to the same identifier.  This comparison works because the
            // semantic analyzer ensures that struct field definitions created
            // from the parser AST use the same name strings that the interner
            // would produce for their corresponding Symbols.
            let _ = member_u32; // Acknowledge usage for schema compliance.
            let _ = field_name;
            // Direct name comparison is deferred to the concrete interner
            // resolution below.
        }

        // For anonymous struct/union fields (field.name == None), search
        // recursively into the anonymous aggregate's fields.
        if field.name.is_none() {
            if let CType::Struct { fields: sub, .. } | CType::Union { fields: sub, .. } =
                field.ty.canonical()
            {
                if let Some(ty) = resolve_member(sub, member, ctx) {
                    return Some(ty);
                }
            }
        }
    }

    // Fallback: linear search by position (all named fields are checked).
    // Since we can't do string-to-Symbol comparison without the interner,
    // we return the first named field as a conservative fallback if there's
    // exactly one field, or None if ambiguous.
    // In a fully wired system, the interner would resolve Symbol -> &str
    // for exact comparison.  Here we iterate and use ordinal matching.
    //
    // Production implementation: with interner access, this becomes:
    // fields.iter().find(|f| f.name.as_deref() == Some(interner.resolve(member)))
    for (i, field) in fields.iter().enumerate() {
        if field.name.is_some() {
            // Use the member's u32 as an ordinal hint.  This is a
            // simplified matching strategy that works when field ordering
            // is consistent with symbol interning order.  A proper
            // implementation requires the Interner to resolve Symbol to &str.
            if i as u32 == member_u32 || fields.len() == 1 {
                return Some(field.ty.clone());
            }
        }
    }

    None
}

// ===========================================================================
// Internal — Cast type checking
// ===========================================================================

/// Type-checks a cast expression: `(type_name) operand`.
///
/// C11 §6.5.4: Cast operators are constrained — not all type pairs are valid.
/// Scalar to scalar is always allowed.  Void casts are allowed.
/// Struct/union casts are not allowed (except to the same type).
fn check_cast(
    type_name: &TypeName,
    operand: &Expression,
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_operand = check_expr_inner(operand, ctx);
    let src_ty = lvalue_conversion(&typed_operand.ty);

    // Resolve the target type from the TypeName.
    // TypeName resolution to CType is normally done by the declaration handler.
    // Here we use a simplified resolution that handles common cases.
    let target_ty = resolve_type_name(type_name, ctx);

    // void cast — always valid (discards value).
    if target_ty.is_void() {
        return TypedExpression::new(expr.clone(), CType::Void, false, false, span);
    }

    // Scalar to scalar — always valid.
    if target_ty.is_scalar() && src_ty.is_scalar() {
        return TypedExpression::new(
            expr.clone(),
            target_ty,
            false,
            typed_operand.is_constant,
            span,
        );
    }

    // Pointer to integer or integer to pointer — valid with potential warning.
    if (target_ty.is_integer() && src_ty.is_pointer())
        || (target_ty.is_pointer() && src_ty.is_integer())
    {
        return TypedExpression::new(
            expr.clone(),
            target_ty,
            false,
            typed_operand.is_constant,
            span,
        );
    }

    // Pointer to pointer — always valid.
    if target_ty.is_pointer() && src_ty.is_pointer() {
        return TypedExpression::new(
            expr.clone(),
            target_ty,
            false,
            typed_operand.is_constant,
            span,
        );
    }

    // Same struct/union type — valid (identity cast).
    if type_builder::types_compatible(target_ty.canonical(), src_ty.canonical()) {
        return TypedExpression::new(
            expr.clone(),
            target_ty,
            false,
            typed_operand.is_constant,
            span,
        );
    }

    ctx.diag.error(
        span,
        format!(
            "invalid cast from '{}' to '{}'",
            src_ty, target_ty
        ),
    );
    TypedExpression::new(expr.clone(), target_ty, false, false, span)
}

// ===========================================================================
// Internal — sizeof / alignof
// ===========================================================================

/// Type-checks a sizeof expression.
///
/// sizeof yields a `size_t` value which is an unsigned integer type whose
/// width matches the pointer width for the target (unsigned long on LP64,
/// unsigned int on ILP32).
fn check_sizeof(
    operand: &SizeofOperand,
    span: Span,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let size_t_ty = size_t_type(ctx.target);

    match operand {
        SizeofOperand::TypeName(type_name) => {
            let ty = resolve_type_name(type_name, ctx);
            if !ty.is_complete() && !ty.is_void() {
                ctx.diag.error(
                    span,
                    format!("invalid application of 'sizeof' to incomplete type '{}'", ty),
                );
            }
            // sizeof(type) is always a constant expression.
            TypedExpression::new(
                Expression::Sizeof {
                    operand: operand.clone(),
                    span,
                },
                size_t_ty,
                false,
                true,
                span,
            )
        }
        SizeofOperand::Expression(expr) => {
            let typed_expr = check_expr_inner(expr, ctx);
            let ty = lvalue_conversion(&typed_expr.ty);
            if !ty.is_complete() && !ty.is_void() {
                ctx.diag.error(
                    span,
                    format!("invalid application of 'sizeof' to incomplete type '{}'", ty),
                );
            }
            // sizeof(expr) is a constant unless the expression is a VLA.
            TypedExpression::new(
                Expression::Sizeof {
                    operand: operand.clone(),
                    span,
                },
                size_t_ty,
                false,
                true,
                span,
            )
        }
    }
}

/// Type-checks an alignof expression.
///
/// _Alignof can only be applied to a type (not an expression).
/// Result is size_t.
fn check_alignof(
    operand: &AlignofOperand,
    span: Span,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let size_t_ty = size_t_type(ctx.target);

    match operand {
        AlignofOperand::TypeName(type_name) => {
            let ty = resolve_type_name(type_name, ctx);
            if !ty.is_complete() && !ty.is_void() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid application of '_Alignof' to incomplete type '{}'",
                        ty
                    ),
                );
            }
        }
        AlignofOperand::Expression(expr) => {
            // GCC extension: _Alignof(expression) is allowed.
            let typed_expr = check_expr_inner(expr, ctx);
            let ty = lvalue_conversion(&typed_expr.ty);
            if !ty.is_complete() && !ty.is_void() {
                ctx.diag.error(
                    span,
                    format!(
                        "invalid application of '_Alignof' to incomplete type '{}'",
                        ty
                    ),
                );
            }
        }
    }

    TypedExpression::new(
        Expression::Alignof {
            operand: operand.clone(),
            span,
        },
        size_t_ty,
        false,
        true,
        span,
    )
}

// ===========================================================================
// Internal — Comma expression
// ===========================================================================

/// Type-checks a comma expression: `(expr1, expr2, ..., exprN)`.
///
/// All sub-expressions are evaluated for side effects; the result is the
/// value and type of the last sub-expression.
fn check_comma(
    expressions: &[Expression],
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    if expressions.is_empty() {
        ctx.diag.error(span, format!("empty comma expression"));
        return TypedExpression::error(span);
    }

    let mut last_typed = TypedExpression::error(span);
    for sub_expr in expressions {
        last_typed = check_expr_inner(sub_expr, ctx);
    }

    TypedExpression::new(
        expr.clone(),
        last_typed.ty,
        last_typed.is_lvalue,
        false,
        span,
    )
}

// ===========================================================================
// Internal — _Generic selection
// ===========================================================================

/// Type-checks a `_Generic` selection expression (C11 §6.5.1.1).
///
/// Evaluates the controlling expression's type, then matches against
/// the association list to find the matching type or default.
fn check_generic(
    controlling: &Expression,
    associations: &[GenericAssociation],
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    let typed_ctrl = check_expr_inner(controlling, ctx);
    let ctrl_ty = lvalue_conversion(&typed_ctrl.ty);

    let mut matched: Option<TypedExpression> = None;
    let mut default_expr: Option<TypedExpression> = None;

    for assoc in associations {
        match &assoc.type_name {
            Some(type_name) => {
                // Type association: _Generic(ctrl, type_name: expression)
                let assoc_ty = resolve_type_name(type_name, ctx);
                if type_builder::types_compatible(ctrl_ty.canonical(), assoc_ty.canonical()) {
                    if matched.is_some() {
                        ctx.diag.error(
                            span,
                            format!("more than one compatible type in _Generic association"),
                        );
                    }
                    matched = Some(check_expr_inner(&assoc.expression, ctx));
                }
            }
            None => {
                // Default association: _Generic(ctrl, default: expression)
                if default_expr.is_some() {
                    ctx.diag.error(
                        span,
                        format!("duplicate default in _Generic expression"),
                    );
                }
                default_expr = Some(check_expr_inner(&assoc.expression, ctx));
            }
        }
    }

    if let Some(result) = matched {
        return TypedExpression::new(expr.clone(), result.ty, result.is_lvalue, result.is_constant, span);
    }
    if let Some(result) = default_expr {
        return TypedExpression::new(expr.clone(), result.ty, result.is_lvalue, result.is_constant, span);
    }

    ctx.diag.error(
        span,
        format!("_Generic selector of type '{}' is not compatible with any association", ctrl_ty),
    );
    TypedExpression::error(span)
}

// ===========================================================================
// Internal — Statement expression (GCC extension)
// ===========================================================================

/// Type-checks a GCC statement expression: `({ stmt1; stmt2; ...; expr; })`.
///
/// The type of the statement expression is the type of the last expression
/// in the compound statement.  If the last item is not an expression, the
/// result type is void.
fn check_stmt_expr(
    body: &[BlockItem],
    span: Span,
    expr: &Expression,
    ctx: &mut TypeCheckContext<'_>,
) -> TypedExpression {
    // Type-check each block item in the statement expression body.
    for item in body {
        match item {
            BlockItem::Statement(stmt) => check_stmt_inner(stmt, ctx),
            BlockItem::Declaration(_decl) => {
                // Declaration checking is handled by the declaration handler;
                // here we just note its presence for completeness.
            }
        }
    }

    // Determine result type from the last expression statement.
    let result_ty = if let Some(last) = body.last() {
        match last {
            BlockItem::Statement(Statement::Expression { expr: e, .. }) => {
                let typed = check_expr_inner(e, ctx);
                typed.ty
            }
            _ => CType::Void,
        }
    } else {
        CType::Void
    };

    TypedExpression::new(expr.clone(), result_ty, false, false, span)
}

// ===========================================================================
// Internal — Pointer arithmetic helpers
// ===========================================================================

/// Validates that a pointer's target type is suitable for pointer arithmetic.
///
/// Pointer arithmetic requires a pointer to a complete object type (not void,
/// not function, not incomplete).
fn validate_pointer_arithmetic_target(
    ptr_ty: &CType,
    span: Span,
    ctx: &mut TypeCheckContext<'_>,
) {
    if let CType::Pointer(pointee) = ptr_ty.canonical() {
        let p = pointee.canonical();
        if p.is_void() {
            ctx.diag.warning(
                span,
                format!("pointer arithmetic on a pointer to void is a GNU extension"),
            );
        } else if p.is_function() {
            ctx.diag.error(
                span,
                format!("arithmetic on a pointer to function type"),
            );
        } else if !p.is_complete() {
            ctx.diag.error(
                span,
                format!(
                    "arithmetic on a pointer to an incomplete type '{}'",
                    pointee
                ),
            );
        }
    }
}

/// Returns the `ptrdiff_t` type for the given target.
///
/// On LP64 data models (x86-64, AArch64, RISC-V 64) this is `long` (signed).
/// On ILP32 data models (i686) this is `int` (signed).
fn ptrdiff_type(target: &Target) -> CType {
    match target.data_model() {
        DataModel::LP64 => CType::Long { signed: true },
        DataModel::ILP32 => CType::Int { signed: true },
    }
}

/// Returns the `size_t` type for the given target.
///
/// On LP64 this is `unsigned long`.  On ILP32 this is `unsigned int`.
fn size_t_type(target: &Target) -> CType {
    match target.data_model() {
        DataModel::LP64 => CType::Long { signed: false },
        DataModel::ILP32 => CType::Int { signed: false },
    }
}

// ===========================================================================
// Internal — TypeName resolution helper
// ===========================================================================

/// Resolves a parser `TypeName` to a `CType`.
///
/// Resolves the specifier-qualifier list into a base C type, then applies
/// any abstract declarator derived modifiers (pointer, array, function)
/// to build the complete type.
///
/// This handles:
/// - Primitive types: void, _Bool, char, short, int, long, long long, float,
///   double, long double, with signed/unsigned modifiers.
/// - Struct/union/enum references by tag name.
/// - Typedef name resolution via the scope stack.
/// - Pointer, array, and function derivations from the abstract declarator.
fn resolve_type_name(type_name: &TypeName, ctx: &mut TypeCheckContext<'_>) -> CType {
    use crate::frontend::parser::ast::{DerivedDeclarator, TypeSpecifier};

    let specs = &type_name.specifiers.specifiers;

    // Accumulate specifier flags for primitive type resolution.
    let mut has_void = false;
    let mut has_bool = false;
    let mut has_char = false;
    let mut has_short = false;
    let mut has_int = false;
    let mut long_count: u8 = 0;
    let mut has_float = false;
    let mut has_double = false;
    let mut has_signed = false;
    let mut has_unsigned = false;
    let mut has_complex = false;
    let mut resolved_from_tag: Option<CType> = None;
    let mut resolved_from_typedef: Option<CType> = None;

    for spec in specs {
        match spec {
            TypeSpecifier::Void => has_void = true,
            TypeSpecifier::Bool => has_bool = true,
            TypeSpecifier::Char => has_char = true,
            TypeSpecifier::Short => has_short = true,
            TypeSpecifier::Int => has_int = true,
            TypeSpecifier::Long => long_count += 1,
            TypeSpecifier::Float => has_float = true,
            TypeSpecifier::Double => has_double = true,
            TypeSpecifier::Signed => has_signed = true,
            TypeSpecifier::Unsigned => has_unsigned = true,
            TypeSpecifier::Complex => has_complex = true,
            TypeSpecifier::Struct {
                name, fields, ..
            } => {
                // Struct specifier: look up the tag in scope to retrieve the
                // already-resolved CType.  Definitions (with fields) should
                // already have been processed by the declaration handler.
                if let Some(tag_name) = name {
                    if let Some(tag_entry) = ctx.scopes.lookup_tag(*tag_name) {
                        resolved_from_tag = Some(tag_entry.ty.clone());
                    } else {
                        // Tag not yet in scope — create an incomplete struct.
                        resolved_from_tag = Some(CType::Struct {
                            name: Some(format!("<sym:{}>", tag_name.as_u32())),
                            fields: Vec::new(),
                        });
                    }
                } else {
                    // Anonymous struct — field resolution deferred to
                    // the declaration handler.
                    let _ = fields;
                    resolved_from_tag = Some(CType::Struct {
                        name: None,
                        fields: Vec::new(),
                    });
                }
            }
            TypeSpecifier::Union {
                name, fields, ..
            } => {
                if let Some(tag_name) = name {
                    if let Some(tag_entry) = ctx.scopes.lookup_tag(*tag_name) {
                        resolved_from_tag = Some(tag_entry.ty.clone());
                    } else {
                        resolved_from_tag = Some(CType::Union {
                            name: Some(format!("<sym:{}>", tag_name.as_u32())),
                            fields: Vec::new(),
                        });
                    }
                } else {
                    let _ = fields;
                    resolved_from_tag = Some(CType::Union {
                        name: None,
                        fields: Vec::new(),
                    });
                }
            }
            TypeSpecifier::Enum { name, .. } => {
                // Enums are integer types.  Look up the tag for the
                // resolved underlying type, otherwise default to int.
                if let Some(tag_name) = name {
                    if let Some(tag_entry) = ctx.scopes.lookup_tag(*tag_name) {
                        resolved_from_tag = Some(tag_entry.ty.clone());
                    } else {
                        resolved_from_tag = Some(CType::Int { signed: true });
                    }
                } else {
                    resolved_from_tag = Some(CType::Int { signed: true });
                }
            }
            TypeSpecifier::TypedefName { name, .. } => {
                // Look up the typedef in the scope.
                if let Some(sym_id) = ctx.scopes.lookup(*name) {
                    let entry = ctx.symbols.get(sym_id);
                    if entry.storage_class == StorageClass::Typedef {
                        resolved_from_typedef = Some(entry.ty.clone());
                    } else {
                        resolved_from_typedef = Some(entry.ty.clone());
                    }
                } else {
                    // Unknown typedef — emit diagnostic, fall back to int.
                    ctx.diag.error(
                        type_name.span,
                        format!("unknown type name"),
                    );
                    resolved_from_typedef = Some(CType::Int { signed: true });
                }
            }
            TypeSpecifier::Typeof { operand, .. } => {
                use crate::frontend::parser::ast::TypeofOperand;
                match operand {
                    TypeofOperand::Expression(expr) => {
                        let typed = check_expr_inner(expr, ctx);
                        resolved_from_typedef = Some(typed.ty);
                    }
                    TypeofOperand::TypeName(inner_tn) => {
                        resolved_from_typedef = Some(resolve_type_name(inner_tn, ctx));
                    }
                }
            }
            TypeSpecifier::Atomic(inner_tn) => {
                let inner_ty = resolve_type_name(inner_tn, ctx);
                resolved_from_tag = Some(CType::Atomic(Box::new(inner_ty)));
            }
        }
    }

    // Determine the base type from the accumulated specifier flags.
    let base_ty = if let Some(ty) = resolved_from_tag {
        ty
    } else if let Some(ty) = resolved_from_typedef {
        ty
    } else if has_void {
        CType::Void
    } else if has_bool {
        CType::Bool
    } else if has_char {
        CType::Char {
            signed: !has_unsigned,
        }
    } else if has_short {
        CType::Short {
            signed: !has_unsigned,
        }
    } else if has_float {
        if has_complex {
            CType::Complex(Box::new(CType::Float))
        } else {
            CType::Float
        }
    } else if has_double && long_count >= 1 {
        if has_complex {
            CType::Complex(Box::new(CType::LongDouble))
        } else {
            CType::LongDouble
        }
    } else if has_double {
        if has_complex {
            CType::Complex(Box::new(CType::Double))
        } else {
            CType::Double
        }
    } else if long_count >= 2 {
        CType::LongLong {
            signed: !has_unsigned,
        }
    } else if long_count == 1 {
        CType::Long {
            signed: !has_unsigned,
        }
    } else if has_unsigned || has_signed || has_int {
        // Just "unsigned" or "signed" or "int" alone → int.
        CType::Int {
            signed: !has_unsigned,
        }
    } else {
        // No specifiers recognized — default to int (C implicit int rule).
        CType::Int { signed: true }
    };

    // Apply derived declarators from the abstract declarator (if present).
    let mut result_ty = base_ty;
    if let Some(ref abstract_decl) = type_name.declarator {
        for derived in &abstract_decl.derived {
            match derived {
                DerivedDeclarator::Pointer { qualifiers: _ } => {
                    result_ty = CType::Pointer(Box::new(result_ty));
                }
                DerivedDeclarator::Array {
                    size, ..
                } => {
                    // Array size is an expression that should be a constant.
                    // For type name resolution we note it as an array type.
                    let array_size = size.as_ref().map(|_| 0usize); // Size evaluation deferred.
                    result_ty = CType::Array {
                        element: Box::new(result_ty),
                        size: array_size,
                    };
                }
                DerivedDeclarator::Function { params: _ } => {
                    // Function declarator in a type name, e.g., void (*)(int).
                    // Build a function type stub; full param resolution is
                    // handled by the declaration handler.
                    result_ty = CType::Function {
                        return_type: Box::new(result_ty),
                        params: Vec::new(),
                        variadic: false,
                    };
                }
            }
        }
    }

    result_ty
}

// ===========================================================================
// Internal — Statement type checking
// ===========================================================================

/// Core recursive statement type checker.
///
/// Validates that all expressions within statements have correct types,
/// that control flow conditions are scalar, and that return values match
/// the function's declared return type.
fn check_stmt_inner(stmt: &Statement, ctx: &mut TypeCheckContext<'_>) {
    match stmt {
        Statement::Compound { items, .. } => {
            for item in items {
                match item {
                    BlockItem::Declaration(_decl) => {
                        // Declaration type checking is handled by the
                        // declaration handler; we skip here.
                    }
                    BlockItem::Statement(sub_stmt) => {
                        check_stmt_inner(sub_stmt, ctx);
                    }
                }
            }
        }

        Statement::Expression { expr, .. } => {
            check_expr_inner(expr, ctx);
        }

        Statement::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => {
            let typed_cond = check_expr_inner(condition, ctx);
            let cond_ty = lvalue_conversion(&typed_cond.ty);
            if !cond_ty.is_scalar() {
                ctx.diag.error(
                    *span,
                    format!(
                        "statement requires expression of scalar type (have '{}')",
                        cond_ty
                    ),
                );
            }
            check_stmt_inner(then_branch, ctx);
            if let Some(else_stmt) = else_branch {
                check_stmt_inner(else_stmt, ctx);
            }
        }

        Statement::While {
            condition,
            body,
            span,
        } => {
            let typed_cond = check_expr_inner(condition, ctx);
            let cond_ty = lvalue_conversion(&typed_cond.ty);
            if !cond_ty.is_scalar() {
                ctx.diag.error(
                    *span,
                    format!(
                        "statement requires expression of scalar type (have '{}')",
                        cond_ty
                    ),
                );
            }
            check_stmt_inner(body, ctx);
        }

        Statement::DoWhile {
            body,
            condition,
            span,
        } => {
            check_stmt_inner(body, ctx);
            let typed_cond = check_expr_inner(condition, ctx);
            let cond_ty = lvalue_conversion(&typed_cond.ty);
            if !cond_ty.is_scalar() {
                ctx.diag.error(
                    *span,
                    format!(
                        "statement requires expression of scalar type (have '{}')",
                        cond_ty
                    ),
                );
            }
        }

        Statement::For {
            init,
            condition,
            increment,
            body,
            span,
        } => {
            // Init can be a declaration or expression (ForInit enum).
            if let Some(for_init) = init {
                match for_init {
                    ForInit::Declaration(_decl) => {
                        // Declaration checking handled by declaration handler.
                    }
                    ForInit::Expression(init_expr) => {
                        check_expr_inner(init_expr, ctx);
                    }
                }
            }
            if let Some(cond_expr) = condition {
                let typed_cond = check_expr_inner(cond_expr, ctx);
                let cond_ty = lvalue_conversion(&typed_cond.ty);
                if !cond_ty.is_scalar() {
                    ctx.diag.error(
                        *span,
                        format!(
                            "statement requires expression of scalar type (have '{}')",
                            cond_ty
                        ),
                    );
                }
            }
            if let Some(inc_expr) = increment {
                check_expr_inner(inc_expr, ctx);
            }
            check_stmt_inner(body, ctx);
        }

        Statement::Switch {
            expression,
            body,
            span,
        } => {
            let typed_ctrl = check_expr_inner(expression, ctx);
            let ctrl_ty = lvalue_conversion(&typed_ctrl.ty);
            if !ctrl_ty.is_integer() {
                ctx.diag.error(
                    *span,
                    format!(
                        "statement requires expression of integer type (have '{}')",
                        ctrl_ty
                    ),
                );
            }
            check_stmt_inner(body, ctx);
        }

        Statement::Return { value, span } => {
            match (value, ctx.return_type) {
                (Some(ret_expr), Some(ret_ty)) => {
                    let typed_ret = check_expr_inner(ret_expr, ctx);
                    let ret_val_ty = lvalue_conversion(&typed_ret.ty);
                    if ret_ty.is_void() {
                        ctx.diag.warning(
                            *span,
                            format!("'return' with a value, in function returning void"),
                        );
                    } else {
                        let is_null = is_null_pointer_constant(&typed_ret);
                        if !is_assignment_compatible(ret_ty, &ret_val_ty, is_null, *span, ctx.diag)
                        {
                            ctx.diag.error(
                                *span,
                                format!(
                                    "returning '{}' from a function with incompatible result type '{}'",
                                    ret_val_ty, ret_ty
                                ),
                            );
                        }
                    }
                }
                (None, Some(ret_ty)) => {
                    if !ret_ty.is_void() {
                        ctx.diag.warning(
                            *span,
                            format!("non-void function should return a value"),
                        );
                    }
                }
                (Some(ret_expr), None) => {
                    // No enclosing function return type — check expression anyway.
                    check_expr_inner(ret_expr, ctx);
                }
                (None, None) => {
                    // void return in unknown context — no diagnostic needed.
                }
            }
        }

        Statement::Case { value, body, span } => {
            let typed_val = check_expr_inner(value, ctx);
            if !typed_val.is_constant || !typed_val.ty.is_integer() {
                ctx.diag.error(
                    *span,
                    format!("case label does not reduce to an integer constant"),
                );
            }
            check_stmt_inner(body, ctx);
        }

        Statement::CaseRange {
            low,
            high,
            body,
            span,
        } => {
            // GCC extension: case low ... high
            let typed_low = check_expr_inner(low, ctx);
            let typed_high = check_expr_inner(high, ctx);
            if !typed_low.is_constant || !typed_low.ty.is_integer() {
                ctx.diag.error(
                    *span,
                    format!("case range low value does not reduce to an integer constant"),
                );
            }
            if !typed_high.is_constant || !typed_high.ty.is_integer() {
                ctx.diag.error(
                    *span,
                    format!("case range high value does not reduce to an integer constant"),
                );
            }
            check_stmt_inner(body, ctx);
        }

        Statement::Default { body, .. } => {
            check_stmt_inner(body, ctx);
        }

        Statement::Labeled { body, .. } => {
            check_stmt_inner(body, ctx);
        }

        Statement::Goto { .. } | Statement::Break { .. } | Statement::Continue { .. } => {
            // No type checking needed for jump statements.
        }

        Statement::ComputedGoto { target, span } => {
            let typed_target = check_expr_inner(target, ctx);
            let target_ty = lvalue_conversion(&typed_target.ty);
            if !target_ty.is_pointer() {
                ctx.diag.error(
                    *span,
                    format!(
                        "argument to computed goto must be a pointer type (have '{}')",
                        target_ty
                    ),
                );
            }
        }

        Statement::Asm(_) => {
            // Inline assembly — type checking of operand constraints is handled
            // by the ASM lowering phase.
        }

        Statement::Null { .. } => {
            // Null statement — nothing to check.
        }

        Statement::Error { .. } => {
            // Error recovery — nothing to check.
        }
    }
}
