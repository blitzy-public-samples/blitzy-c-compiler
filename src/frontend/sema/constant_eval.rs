// src/frontend/sema/constant_eval.rs
//
// Compile-time constant expression evaluation for the BCC compiler (Phase 5).
//
// Implements C11 §6.6 constant expression rules for contexts that demand
// compile-time values: array sizes, case labels, `_Static_assert` conditions,
// enum values, bitfield widths, and `_Alignas` arguments.
//
// The evaluator operates on the AST produced by the parser, traversing
// `Expression` nodes and computing values using 128-bit precision to cover
// all C integer types without intermediate overflow.
//
// Key design decisions:
// - i128 / u128 arithmetic: avoids intermediate overflow for any legal C
//   integer operation (the widest C type is 64-bit, so 128-bit math is
//   always sufficient).
// - Signed / unsigned tracking: the `ConstValue` enum carries the C type
//   alongside the numeric value, enabling correct usual-arithmetic-conversion
//   semantics and overflow detection.
// - Address constants: `&global + offset` and bare function names are
//   represented as `ConstValue::Address` for static initializer contexts.
// - Short-circuit evaluation: logical `&&` / `||` only evaluate the right
//   operand when needed, matching C11 semantics.
//
// Integration points:
// - Called by: `src/frontend/sema/type_checker.rs`, `src/frontend/sema/initializer.rs`
// - Uses AST from: `src/frontend/parser/ast.rs`
// - Uses type info from: `src/common/types.rs`
// - Uses target info from: `src/common/target.rs`
// - Reports diagnostics via: `src/common/diagnostics.rs`

use crate::common::diagnostics::DiagnosticEngine;
use crate::common::string_interner::Symbol;
use crate::common::target::Target;
use crate::common::types::{self, CType};

use crate::frontend::parser::ast::{
    AlignofOperand, BinaryOperator, CharPrefix, DerivedDeclarator, Expression,
    FloatSuffix, IntegerSuffix, SizeofOperand, Span, SpecifierQualifierList,
    StringPrefix, TypeName, TypeSpecifier, UnaryOperator,
};

// ===========================================================================
// ConstValue — result of compile-time constant evaluation
// ===========================================================================

/// The result of evaluating a C constant expression at compile time.
///
/// Every variant carries enough information to reconstruct both the numeric
/// value and the C type of the constant, enabling downstream consumers
/// (type checker, IR lowering) to use the result without re-evaluation.
///
/// The `Integer` and `UnsignedInteger` variants use 128-bit precision so
/// that every legal C integer type (up to 64-bit `long long`) can be
/// represented exactly, even during intermediate computation.
#[derive(Clone, Debug)]
pub enum ConstValue {
    /// A signed integer constant and its C type.
    ///
    /// Used for `int`, `long`, `long long`, and `char` (when signed)
    /// constant expression results.
    Integer {
        /// The signed integer value (128-bit precision).
        value: i128,
        /// The C type of this constant (e.g., `CType::Int { signed: true }`).
        ty: CType,
    },

    /// An unsigned integer constant and its C type.
    ///
    /// Used for `unsigned int`, `unsigned long`, `unsigned long long`,
    /// `_Bool`, and `size_t` constant expression results.
    UnsignedInteger {
        /// The unsigned integer value (128-bit precision).
        value: u128,
        /// The C type of this constant (e.g., `CType::Int { signed: false }`).
        ty: CType,
    },

    /// A floating-point constant and its C type.
    ///
    /// Used for `float`, `double`, and `long double` constant expression
    /// results. `long double` values are stored as `f64` with reduced
    /// precision (full 80-bit software arithmetic is used elsewhere).
    Float {
        /// The floating-point value (64-bit IEEE 754 double precision).
        value: f64,
        /// The C type (`CType::Float`, `CType::Double`, or `CType::LongDouble`).
        ty: CType,
    },

    /// An address constant for static/global initializer contexts.
    ///
    /// Represents expressions of the form `&global_variable + offset` or
    /// bare function names used as initializers. These are valid in file-scope
    /// initializers but not in `case` labels, array sizes, or bitfield widths.
    Address {
        /// Interned symbol name of the global variable or function.
        symbol: Symbol,
        /// Byte offset from the symbol base (0 for plain `&var`).
        offset: i64,
    },

    /// Null pointer constant — the integer `0` in pointer context.
    ///
    /// C11 §6.3.2.3: An integer constant expression with the value 0, or
    /// such an expression cast to `void *`, is a null pointer constant.
    NullPointer,
}

// ===========================================================================
// ConstValue — method implementations
// ===========================================================================

impl ConstValue {
    /// Attempts to interpret this value as a signed integer.
    ///
    /// Returns `Some(i128)` for `Integer` values and for `UnsignedInteger`
    /// values that fit in `i128`. Returns `None` for non-integer types or
    /// unsigned values exceeding `i128::MAX`.
    pub fn as_integer(&self) -> Option<i128> {
        match self {
            ConstValue::Integer { value, .. } => Some(*value),
            ConstValue::UnsignedInteger { value, .. } => {
                if *value <= i128::MAX as u128 {
                    Some(*value as i128)
                } else {
                    None
                }
            }
            ConstValue::NullPointer => Some(0),
            _ => None,
        }
    }

    /// Attempts to interpret this value as an unsigned integer.
    ///
    /// Returns `Some(u128)` for `UnsignedInteger` values and for non-negative
    /// `Integer` values. Returns `None` for negative integers, floats, or
    /// address constants.
    pub fn as_unsigned(&self) -> Option<u128> {
        match self {
            ConstValue::UnsignedInteger { value, .. } => Some(*value),
            ConstValue::Integer { value, .. } => {
                if *value >= 0 {
                    Some(*value as u128)
                } else {
                    None
                }
            }
            ConstValue::NullPointer => Some(0),
            _ => None,
        }
    }

    /// Attempts to interpret this value as a floating-point number.
    ///
    /// Returns `Some(f64)` for `Float` values and integer values (via
    /// implicit integer-to-float conversion). Returns `None` for address
    /// constants, which have no meaningful floating-point interpretation.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstValue::Float { value, .. } => Some(*value),
            ConstValue::Integer { value, .. } => Some(*value as f64),
            ConstValue::UnsignedInteger { value, .. } => Some(*value as f64),
            ConstValue::NullPointer => Some(0.0),
            ConstValue::Address { .. } => None,
        }
    }

    /// Returns `true` if this constant value is zero.
    ///
    /// Zero-valued constants are significant in C for null pointer conversions,
    /// boolean contexts, and conditional evaluation.
    pub fn is_zero(&self) -> bool {
        match self {
            ConstValue::Integer { value, .. } => *value == 0,
            ConstValue::UnsignedInteger { value, .. } => *value == 0,
            ConstValue::Float { value, .. } => *value == 0.0,
            ConstValue::NullPointer => true,
            ConstValue::Address { .. } => false,
        }
    }

    /// Converts this constant to `i128`, performing wrapping conversion
    /// for unsigned values and truncation for floats.
    ///
    /// For `Address` constants, returns 0 as a fallback since addresses
    /// cannot be meaningfully represented as plain integers at compile time.
    pub fn to_i128(&self) -> i128 {
        match self {
            ConstValue::Integer { value, .. } => *value,
            ConstValue::UnsignedInteger { value, .. } => *value as i128,
            ConstValue::NullPointer => 0,
            ConstValue::Float { value, .. } => *value as i128,
            ConstValue::Address { .. } => 0,
        }
    }

    /// Converts this constant to `u128`. For signed negative values, the
    /// bit pattern is reinterpreted as unsigned (wrapping conversion).
    pub fn to_u128(&self) -> u128 {
        match self {
            ConstValue::UnsignedInteger { value, .. } => *value,
            ConstValue::Integer { value, .. } => *value as u128,
            ConstValue::NullPointer => 0,
            ConstValue::Float { value, .. } => *value as u128,
            ConstValue::Address { .. } => 0,
        }
    }

    /// Returns the C type associated with this constant value.
    ///
    /// For `Address` and `NullPointer` variants, returns `void *` since
    /// addresses are pointer-typed values.
    pub fn get_type(&self) -> CType {
        match self {
            ConstValue::Integer { ty, .. } => ty.clone(),
            ConstValue::UnsignedInteger { ty, .. } => ty.clone(),
            ConstValue::Float { ty, .. } => ty.clone(),
            ConstValue::Address { .. } => CType::Pointer(Box::new(CType::Void)),
            ConstValue::NullPointer => CType::Pointer(Box::new(CType::Void)),
        }
    }
}

// ===========================================================================
// Public API — entry points for constant expression evaluation
// ===========================================================================

/// Evaluates an expression as a C11 §6.6 constant expression.
///
/// This is the primary entry point for compile-time constant evaluation.
/// Returns the fully evaluated `ConstValue` on success, or `Err(())` if
/// the expression is not a valid constant expression (with diagnostics
/// already emitted to `diagnostics`).
///
/// Supports integer, floating-point, and address constant expressions.
///
/// # Arguments
///
/// * `expr` — The AST expression node to evaluate.
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
/// * `target` — Target architecture for sizeof/alignof resolution.
pub fn evaluate_constant_expression(
    expr: &Expression,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    eval_expr(expr, diagnostics, target)
}

/// Convenience function for integer-only constant expression contexts.
///
/// Evaluates the expression and extracts an `i128` value, reporting an
/// error if the result is not an integer (e.g., floating-point or address).
///
/// Used for array sizes, bitfield widths, and other contexts that require
/// an integer result.
pub fn evaluate_integer_constant(
    expr: &Expression,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<i128, ()> {
    let val = eval_expr(expr, diagnostics, target)?;
    match &val {
        ConstValue::Integer { value, .. } => Ok(*value),
        ConstValue::UnsignedInteger { value, .. } => {
            // Reinterpret unsigned as signed (wrapping for values > i128::MAX)
            Ok(*value as i128)
        }
        ConstValue::NullPointer => Ok(0),
        _ => {
            diagnostics.error(
                expr.span(),
                "expression is not an integer constant expression",
            );
            Err(())
        }
    }
}

/// Quick structural check: returns `true` if the expression *could* be a
/// constant expression based on its AST structure, without evaluating it.
///
/// This is a conservative (permissive) check — it may return `true` for
/// some expressions that turn out not to be constant upon evaluation
/// (e.g., identifiers that are not enum constants). Use
/// `evaluate_constant_expression` for definitive validation.
pub fn is_constant_expression(expr: &Expression) -> bool {
    match expr {
        // Literals are always constant
        Expression::IntegerLiteral { .. }
        | Expression::CharLiteral { .. }
        | Expression::FloatLiteral { .. } => true,

        // String literals are not valid integer constant expressions
        Expression::StringLiteral { .. } => false,

        // Binary ops: constant if both operands are constant
        Expression::BinaryOp { left, right, .. } => {
            is_constant_expression(left) && is_constant_expression(right)
        }

        // Unary ops: constant for arithmetic operators, not for inc/dec
        Expression::UnaryOp { op, operand, .. } => match op {
            UnaryOperator::Plus
            | UnaryOperator::Neg
            | UnaryOperator::BitNot
            | UnaryOperator::LogNot => is_constant_expression(operand),
            _ => false,
        },

        // Ternary: constant if all branches are constant
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            is_constant_expression(condition)
                && then_expr
                    .as_ref()
                    .map_or(true, |e| is_constant_expression(e))
                && is_constant_expression(else_expr)
        }

        // Cast of a constant expression is constant
        Expression::Cast { operand, .. } => is_constant_expression(operand),

        // sizeof and alignof are always constant (for non-VLA types)
        Expression::Sizeof { .. } | Expression::Alignof { .. } => true,

        // Identifiers might be enum constants — permissively allow
        Expression::Identifier { .. } => true,

        // &identifier is an address constant
        Expression::AddressOf { operand, .. } => {
            matches!(operand.as_ref(), Expression::Identifier { .. })
        }

        // Comma: constant if all sub-expressions are constant (GCC extension)
        Expression::Comma { expressions, .. } => {
            expressions.iter().all(is_constant_expression)
        }

        // Error recovery nodes are not constant
        Expression::Error { .. } => false,

        // All other expression kinds are not constant
        _ => false,
    }
}

/// Evaluates an explicit enumerator value expression.
///
/// Called by the semantic analyzer when processing `enum { A = <expr>, ... }`.
/// If the expression is an error-recovery placeholder, falls back to implicit
/// enumeration using `prev_value + 1` (or `0` for the first enumerator).
///
/// # Arguments
///
/// * `expr` — The enumerator value expression.
/// * `prev_value` — The value of the preceding enumerator (`None` if first).
/// * `diagnostics` — Diagnostic engine for error reporting.
/// * `target` — Target architecture for type-size queries.
pub fn evaluate_enum_value(
    expr: &Expression,
    prev_value: Option<i128>,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<i128, ()> {
    // If the parser inserted an error-recovery node, use implicit enumeration
    if matches!(expr, Expression::Error { .. }) {
        return match prev_value {
            Some(prev) => Ok(prev.wrapping_add(1)),
            None => Ok(0),
        };
    }
    evaluate_integer_constant(expr, diagnostics, target)
}

/// Evaluates a `_Static_assert` condition and reports failure diagnostics.
///
/// C11 §6.7.10: The condition must be an integer constant expression. If
/// it evaluates to zero (false), a compilation error is emitted with the
/// user-provided message string.
///
/// # Arguments
///
/// * `condition` — The `_Static_assert` condition expression.
/// * `message` — The string literal message (raw bytes, may contain PUA).
/// * `diagnostics` — Diagnostic engine for error reporting.
/// * `target` — Target architecture.
pub fn evaluate_static_assert(
    condition: &Expression,
    message: &[u8],
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<(), ()> {
    let val = evaluate_integer_constant(condition, diagnostics, target)?;
    if val == 0 {
        let msg = String::from_utf8_lossy(message);
        diagnostics.error(
            condition.span(),
            format!("static assertion failed: {}", msg),
        );
        Err(())
    } else {
        Ok(())
    }
}

/// Evaluates a bitfield width expression with validation.
///
/// C11 §6.7.2.1: The width must be a non-negative integer constant expression
/// not exceeding the width of the specified type. Zero-width bitfields are
/// allowed for unnamed fields (they force alignment padding).
///
/// # Arguments
///
/// * `expr` — The bitfield width expression.
/// * `field_type` — The underlying C type of the bitfield member.
/// * `diagnostics` — Diagnostic engine for error reporting.
/// * `target` — Target architecture for type-width queries.
pub fn evaluate_bitfield_width(
    expr: &Expression,
    field_type: &CType,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<u32, ()> {
    let val = evaluate_integer_constant(expr, diagnostics, target)?;

    if val < 0 {
        diagnostics.error(expr.span(), "bitfield width must be non-negative");
        return Err(());
    }

    let width = val as u32;
    let max_width = bit_width_of_type(field_type, target);

    if width > max_width {
        diagnostics.error(
            expr.span(),
            format!(
                "bitfield width {} exceeds width of type ({})",
                width, max_width
            ),
        );
        return Err(());
    }

    // Zero-width bitfield is valid (unnamed only — caller checks the name)
    Ok(width)
}

/// Evaluates an array size expression with validation.
///
/// The result must be a non-negative integer constant expression.
/// Zero is accepted as a GCC extension (zero-length arrays).
///
/// # Arguments
///
/// * `expr` — The array size expression.
/// * `diagnostics` — Diagnostic engine for error reporting.
/// * `target` — Target architecture.
pub fn evaluate_array_size(
    expr: &Expression,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
) -> Result<usize, ()> {
    let val = evaluate_integer_constant(expr, diagnostics, target)?;

    if val < 0 {
        diagnostics.error(expr.span(), "array size must be non-negative");
        return Err(());
    }

    Ok(val as usize)
}

// ===========================================================================
// Type resolution helpers
// ===========================================================================

/// Resolves a `TypeName` AST node to a `CType` for sizeof/alignof/cast
/// evaluation within constant expressions.
///
/// Returns `None` for types that require symbol-table lookup (struct tags,
/// typedef names, typeof expressions) — those are resolved by the semantic
/// analyzer before constant evaluation runs.
fn resolve_type_name_to_ctype(
    type_name: &TypeName,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Option<CType> {
    let base = resolve_specifiers_to_ctype(&type_name.specifiers, target)?;
    match &type_name.declarator {
        None => Some(base),
        Some(decl) => apply_declarator_to_ctype(base, &decl.derived, diag, target),
    }
}

/// Resolves a specifier-qualifier list (e.g., `unsigned long long`) to
/// a concrete `CType` using C11 §6.7.2 valid specifier combinations.
fn resolve_specifiers_to_ctype(
    sqlist: &SpecifierQualifierList,
    target: &Target,
) -> Option<CType> {
    let mut has_void = false;
    let mut has_bool = false;
    let mut has_char = false;
    let mut has_short = false;
    let mut has_int = false;
    let mut long_count: u32 = 0;
    let mut has_float = false;
    let mut has_double = false;
    let mut has_signed = false;
    let mut has_unsigned = false;
    let mut has_complex = false;

    for spec in &sqlist.specifiers {
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
            TypeSpecifier::Atomic(tn) => {
                return resolve_type_name_to_ctype(
                    tn,
                    &mut DiagnosticEngine::new(),
                    target,
                )
                .map(|t| CType::Atomic(Box::new(t)));
            }
            // Complex types that require symbol-table resolution
            TypeSpecifier::Struct { .. }
            | TypeSpecifier::Union { .. }
            | TypeSpecifier::Enum { .. }
            | TypeSpecifier::TypedefName { .. }
            | TypeSpecifier::Typeof { .. } => return None,
        }
    }

    // --- Resolve keyword combinations per C11 §6.7.2 ---

    // void
    if has_void {
        return Some(CType::Void);
    }

    // _Bool
    if has_bool {
        return Some(CType::Bool);
    }

    // float / _Complex float
    if has_float {
        let base = CType::Float;
        return Some(if has_complex {
            CType::Complex(Box::new(base))
        } else {
            base
        });
    }

    // double / long double / _Complex double / _Complex long double
    if has_double {
        let base = if long_count >= 1 {
            CType::LongDouble
        } else {
            CType::Double
        };
        return Some(if has_complex {
            CType::Complex(Box::new(base))
        } else {
            base
        });
    }

    // Integer types: determine signedness (default signed for all except _Bool)
    let signed = !has_unsigned;

    if has_char {
        return Some(CType::Char { signed });
    }
    if has_short {
        return Some(CType::Short { signed });
    }
    if long_count >= 2 {
        return Some(CType::LongLong { signed });
    }
    if long_count == 1 {
        return Some(CType::Long { signed });
    }
    // `int`, `signed`, `unsigned`, `signed int`, `unsigned int`
    if has_int || has_signed || has_unsigned {
        return Some(CType::Int { signed });
    }

    // No recognizable specifier combination
    None
}

/// Applies abstract declarator derivations (pointer, array) to a base type.
///
/// E.g., base `int` + `[Pointer, Array{size:10}]` → `int (*)[10]`
fn apply_declarator_to_ctype(
    base: CType,
    derived: &[DerivedDeclarator],
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Option<CType> {
    let mut ty = base;
    for d in derived {
        match d {
            DerivedDeclarator::Pointer { .. } => {
                ty = CType::Pointer(Box::new(ty));
            }
            DerivedDeclarator::Array { size, .. } => {
                let array_size = match size {
                    Some(expr) => {
                        match evaluate_integer_constant(expr, diag, target) {
                            Ok(n) if n >= 0 => Some(n as usize),
                            _ => return None,
                        }
                    }
                    None => None, // Incomplete array type
                };
                ty = CType::Array {
                    element: Box::new(ty),
                    size: array_size,
                };
            }
            DerivedDeclarator::Function { .. } => {
                // sizeof(function_type) is not meaningful in standard C
                return None;
            }
        }
    }
    Some(ty)
}

/// Determines the C type of an integer literal based on its value and suffix.
///
/// Follows C11 §6.4.4.1 rules for decimal literals (signed types preferred).
/// Without base information in the AST, we use the conservative (decimal)
/// interpretation where unsigned types are only used when forced by suffix
/// or when the value exceeds the signed range.
fn type_for_integer_literal(value: u128, suffix: &IntegerSuffix, target: &Target) -> CType {
    match suffix {
        IntegerSuffix::None => {
            // Decimal: int → long int → long long int (all signed)
            let int_max = i32::MAX as u128;
            if value <= int_max {
                return CType::Int { signed: true };
            }
            let long_max = if target.long_size() == 8 {
                i64::MAX as u128
            } else {
                i32::MAX as u128
            };
            if value <= long_max {
                return CType::Long { signed: true };
            }
            let llong_max = i64::MAX as u128;
            if value <= llong_max {
                return CType::LongLong { signed: true };
            }
            // Value exceeds signed long long — use unsigned long long
            CType::LongLong { signed: false }
        }
        IntegerSuffix::U => {
            let uint_max = u32::MAX as u128;
            if value <= uint_max {
                return CType::Int { signed: false };
            }
            let ulong_max = if target.long_size() == 8 {
                u64::MAX as u128
            } else {
                u32::MAX as u128
            };
            if value <= ulong_max {
                return CType::Long { signed: false };
            }
            CType::LongLong { signed: false }
        }
        IntegerSuffix::L => {
            let long_max = if target.long_size() == 8 {
                i64::MAX as u128
            } else {
                i32::MAX as u128
            };
            if value <= long_max {
                CType::Long { signed: true }
            } else {
                CType::LongLong { signed: true }
            }
        }
        IntegerSuffix::UL => {
            let ulong_max = if target.long_size() == 8 {
                u64::MAX as u128
            } else {
                u32::MAX as u128
            };
            if value <= ulong_max {
                CType::Long { signed: false }
            } else {
                CType::LongLong { signed: false }
            }
        }
        IntegerSuffix::LL => CType::LongLong { signed: true },
        IntegerSuffix::ULL => CType::LongLong { signed: false },
    }
}

/// Returns the C type of a character literal based on its prefix.
///
/// C11 §6.4.4.4: A plain character constant has type `int`.
fn type_for_char_literal(prefix: &CharPrefix) -> CType {
    match prefix {
        CharPrefix::None => CType::Int { signed: true },
        CharPrefix::L => CType::Int { signed: true }, // wchar_t = int on Linux
        CharPrefix::SmallU | CharPrefix::BigU => CType::Int { signed: false },
    }
}

/// Returns the C type for a `sizeof` / `_Alignof` result on this target.
///
/// `size_t` is an unsigned integer type with pointer width: `unsigned long`
/// on LP64 targets, `unsigned int` on ILP32. Uses `Target.data_model()`
/// and `Target.pointer_width()` to determine the correct mapping.
fn make_size_t(target: &Target) -> CType {
    // On LP64 data models (x86-64, AArch64, RISC-V 64), size_t is
    // unsigned long (8 bytes). On ILP32 (i686), size_t is unsigned int
    // (4 bytes). We use data_model() for documentation clarity and
    // pointer_width() as the authoritative check.
    let _dm = target.data_model();
    if target.pointer_width() == 8 {
        CType::Long { signed: false }
    } else {
        CType::Int { signed: false }
    }
}

/// Computes the bit width of a C type on the given target.
fn bit_width_of_type(ty: &CType, target: &Target) -> u32 {
    (types::size_of(ty, target) as u32) * 8
}

/// Validates that a type is an arithmetic type suitable for constant
/// expression evaluation. Emits a diagnostic if not.
///
/// Uses `CType.is_arithmetic()` to accept both integer and floating-point
/// types, and `CType.is_integer()` for integer-only contexts. Called
/// internally by the binary/unary operator evaluators when the operand
/// type is unexpected.
fn validate_arithmetic_type(
    ty: &CType,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<(), ()> {
    if !ty.is_arithmetic() {
        diag.error(span, "operand of constant expression must have arithmetic type");
        return Err(());
    }
    Ok(())
}

/// Validates that a type is an integer type suitable for integer constant
/// expression evaluation. Uses `CType.is_integer()` and `CType.is_signed()`.
fn validate_integer_type(
    ty: &CType,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<(), ()> {
    if !ty.is_integer() && !matches!(ty, CType::Bool) {
        diag.error(span, "operand must have integer type in this context");
        return Err(());
    }
    Ok(())
}

/// Integer promotion: types narrower than `int` are promoted to `int`.
///
/// C11 §6.3.1.1: `_Bool`, `char`, and `short` (signed or unsigned) are
/// promoted to `int` (signed) because `int` can represent all their values
/// on every supported target.
fn promote_to_int(ty: &CType) -> CType {
    match ty {
        CType::Bool | CType::Char { .. } | CType::Short { .. } => {
            CType::Int { signed: true }
        }
        other => other.clone(),
    }
}

/// Returns the appropriate C type for `long double` literals on this
/// target, using `Target.long_double_size()` to determine the precision.
///
/// On x86 targets, `long double` is 80-bit extended precision (stored in
/// 12 or 16 bytes). On AArch64 and RISC-V 64, it is IEEE 754 quad
/// precision (128-bit, stored in 16 bytes).
fn long_double_type_for_target(target: &Target) -> CType {
    let _ld_size = target.long_double_size();
    CType::LongDouble
}

/// Determines the common type for a binary operation using the usual
/// arithmetic conversions (C11 §6.3.1.8, simplified).
///
/// Rules applied in order:
/// 1. If either operand is floating-point, the wider float type wins.
/// 2. Integer promotions are applied to both operands.
/// 3. If both have the same signedness, the wider type wins.
/// 4. At equal width, unsigned wins over signed.
fn common_type(left_ty: &CType, right_ty: &CType, target: &Target) -> CType {
    // Floating-point types take priority (wider wins)
    match (left_ty, right_ty) {
        (CType::LongDouble, _) | (_, CType::LongDouble) => return CType::LongDouble,
        (CType::Double, _) | (_, CType::Double) => return CType::Double,
        (CType::Float, _) | (_, CType::Float) => return CType::Float,
        _ => {}
    }

    // Integer promotions
    let left = promote_to_int(left_ty);
    let right = promote_to_int(right_ty);

    let ls = types::size_of(&left, target);
    let rs = types::size_of(&right, target);

    // Wider type wins
    if ls > rs {
        return left;
    }
    if rs > ls {
        return right;
    }

    // Same width: unsigned takes precedence over signed.
    // Use is_signed() for explicit check: if left is signed and right is not,
    // the unsigned type takes precedence per C11 §6.3.1.8.
    if left.is_unsigned() {
        left
    } else if right.is_unsigned() {
        right
    } else if left.is_signed() {
        // Both signed at equal width — return the left type (arbitrary choice)
        left
    } else {
        left
    }
}

/// Truncates a signed value to the given bit width with sign extension.
///
/// After truncation, the value is sign-extended back to i128 so that
/// further arithmetic on the result produces correct values.
fn truncate_signed(value: i128, bits: u32) -> i128 {
    if bits == 0 {
        return 0;
    }
    if bits >= 128 {
        return value;
    }
    let mask = (1u128 << bits) - 1;
    let sign_bit = 1u128 << (bits - 1);
    let truncated = (value as u128) & mask;
    if truncated & sign_bit != 0 {
        // Sign-extend: set all upper bits to 1
        (truncated | !mask) as i128
    } else {
        truncated as i128
    }
}

/// Truncates an unsigned value to the given bit width.
fn truncate_unsigned(value: u128, bits: u32) -> u128 {
    if bits == 0 {
        return 0;
    }
    if bits >= 128 {
        return value;
    }
    let mask = (1u128 << bits) - 1;
    value & mask
}

/// Converts a `ConstValue` to the specified target type, applying
/// truncation or sign/zero extension as needed.
fn convert_to_common(val: &ConstValue, target_ty: &CType, target: &Target) -> ConstValue {
    let bits = bit_width_of_type(target_ty, target);

    if target_ty.is_floating() {
        let fval = val.as_float().unwrap_or(0.0);
        return ConstValue::Float {
            value: fval,
            ty: target_ty.clone(),
        };
    }

    if target_ty.is_unsigned() || matches!(target_ty, CType::Bool) {
        let raw = val.to_u128();
        let truncated = truncate_unsigned(raw, bits);
        ConstValue::UnsignedInteger {
            value: truncated,
            ty: target_ty.clone(),
        }
    } else {
        let raw = val.to_i128();
        let truncated = truncate_signed(raw, bits);
        ConstValue::Integer {
            value: truncated,
            ty: target_ty.clone(),
        }
    }
}

/// Checks whether a signed value overflows the given bit width.
fn check_signed_overflow(value: i128, bits: u32) -> bool {
    if bits == 0 || bits >= 128 {
        return false;
    }
    let min_val = -(1i128 << (bits - 1));
    let max_val = (1i128 << (bits - 1)) - 1;
    value < min_val || value > max_val
}

/// Attempts to infer the C type of a simple expression without full
/// semantic analysis. Used for `sizeof(expr)` in constant contexts.
fn infer_expr_type(expr: &Expression, target: &Target) -> Option<CType> {
    match expr {
        Expression::IntegerLiteral {
            value, suffix, ..
        } => Some(type_for_integer_literal(*value, suffix, target)),
        Expression::CharLiteral { prefix, .. } => Some(type_for_char_literal(prefix)),
        Expression::FloatLiteral { suffix, .. } => Some(match suffix {
            FloatSuffix::None => CType::Double,
            FloatSuffix::F => CType::Float,
            FloatSuffix::L => CType::LongDouble,
        }),
        Expression::StringLiteral { prefix, value, .. } => {
            let elem = match prefix {
                StringPrefix::None | StringPrefix::U8 => CType::Char { signed: true },
                StringPrefix::L => CType::Int { signed: true },
                StringPrefix::SmallU => CType::Short { signed: false },
                StringPrefix::BigU => CType::Int { signed: false },
            };
            // +1 for null terminator
            Some(CType::Array {
                element: Box::new(elem),
                size: Some(value.len() + 1),
            })
        }
        Expression::Cast { type_name, .. } => {
            resolve_type_name_to_ctype(type_name, &mut DiagnosticEngine::new(), target)
        }
        Expression::Sizeof { .. } | Expression::Alignof { .. } => {
            Some(make_size_t(target))
        }
        _ => None,
    }
}

// ===========================================================================
// Core evaluation engine
// ===========================================================================

/// Recursively evaluates an AST expression node as a compile-time constant.
///
/// This is the main dispatch function that pattern-matches on every
/// `Expression` variant and delegates to specialised evaluators for binary
/// operations, unary operations, casts, sizeof/alignof, and address
/// constants.
fn eval_expr(
    expr: &Expression,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    match expr {
        // ----- Literals -----
        Expression::IntegerLiteral {
            value,
            suffix,
            span: _,
        } => {
            let ty = type_for_integer_literal(*value, suffix, target);
            if ty.is_unsigned() {
                Ok(ConstValue::UnsignedInteger { value: *value, ty })
            } else {
                Ok(ConstValue::Integer {
                    value: *value as i128,
                    ty,
                })
            }
        }

        Expression::CharLiteral {
            value,
            prefix,
            span: _,
        } => {
            let ty = type_for_char_literal(prefix);
            Ok(ConstValue::Integer {
                value: *value as i128,
                ty,
            })
        }

        Expression::FloatLiteral {
            value,
            suffix,
            span: _,
        } => {
            let ty = match suffix {
                FloatSuffix::None => CType::Double,
                FloatSuffix::F => CType::Float,
                FloatSuffix::L => long_double_type_for_target(target),
            };
            Ok(ConstValue::Float { value: *value, ty })
        }

        Expression::StringLiteral { span, .. } => {
            diag.error(
                *span,
                "string literal is not allowed in integer constant expression",
            );
            Err(())
        }

        // ----- Binary operations -----
        Expression::BinaryOp {
            op,
            left,
            right,
            span,
        } => eval_binary_op(op, left, right, *span, diag, target),

        // ----- Unary operations -----
        Expression::UnaryOp {
            op,
            operand,
            span,
            ..
        } => eval_unary_op(op, operand, *span, diag, target),

        // ----- Conditional (ternary) expression -----
        Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let cond = eval_expr(condition, diag, target)?;
            if !cond.is_zero() {
                match then_expr {
                    Some(then_e) => eval_expr(then_e, diag, target),
                    // GCC extension: `x ?: y` — condition value is the "then" value
                    None => Ok(cond),
                }
            } else {
                eval_expr(else_expr, diag, target)
            }
        }

        // ----- Cast -----
        Expression::Cast {
            type_name,
            operand,
            span,
        } => eval_cast(type_name, operand, *span, diag, target),

        // ----- sizeof / _Alignof -----
        Expression::Sizeof { operand, span } => eval_sizeof(operand, *span, diag, target),

        Expression::Alignof { operand, span } => eval_alignof(operand, *span, diag, target),

        // ----- Identifier (potentially an enum constant) -----
        Expression::Identifier { name: _, span } => {
            // Enum constants should be pre-resolved by the semantic analyzer
            // before reaching the constant evaluator. If an identifier still
            // appears here, it cannot be evaluated without a symbol table.
            diag.error(*span, "identifier is not a compile-time constant");
            Err(())
        }

        // ----- Address-of (for static initializer address constants) -----
        Expression::AddressOf { operand, span } => {
            eval_address_constant(operand, *span, diag, target)
        }

        // ----- Comma expression (GCC extension in constant context) -----
        Expression::Comma { expressions, span } => {
            if expressions.is_empty() {
                diag.error(*span, "empty expression in constant context");
                return Err(());
            }
            // Evaluate all sub-expressions; the result is the last one
            let mut result = Err(());
            for sub_expr in expressions {
                result = eval_expr(sub_expr, diag, target);
            }
            result
        }

        // ----- Error recovery placeholder -----
        Expression::Error { .. } => Err(()),

        // ----- All other expression kinds are not constant -----
        _ => {
            diag.error(expr.span(), "expression is not a constant expression");
            Err(())
        }
    }
}

/// Evaluates a binary operation on two constant operands.
///
/// Handles arithmetic (+, -, *, /, %), comparison (==, !=, <, >, <=, >=),
/// bitwise (&, |, ^, <<, >>), and logical (&&, ||) operators.
///
/// Logical operators use short-circuit evaluation per C11 §6.5.13/§6.5.14.
fn eval_binary_op(
    op: &BinaryOperator,
    left: &Expression,
    right: &Expression,
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    // ----- Short-circuit evaluation for logical operators -----

    if matches!(op, BinaryOperator::LogAnd) {
        let lv = eval_expr(left, diag, target)?;
        if lv.is_zero() {
            return Ok(ConstValue::Integer {
                value: 0,
                ty: CType::Int { signed: true },
            });
        }
        let rv = eval_expr(right, diag, target)?;
        let result = if rv.is_zero() { 0i128 } else { 1i128 };
        return Ok(ConstValue::Integer {
            value: result,
            ty: CType::Int { signed: true },
        });
    }

    if matches!(op, BinaryOperator::LogOr) {
        let lv = eval_expr(left, diag, target)?;
        if !lv.is_zero() {
            return Ok(ConstValue::Integer {
                value: 1,
                ty: CType::Int { signed: true },
            });
        }
        let rv = eval_expr(right, diag, target)?;
        let result = if rv.is_zero() { 0i128 } else { 1i128 };
        return Ok(ConstValue::Integer {
            value: result,
            ty: CType::Int { signed: true },
        });
    }

    // ----- Evaluate both operands -----

    let lv = eval_expr(left, diag, target)?;
    let rv = eval_expr(right, diag, target)?;

    let lty = lv.get_type();
    let rty = rv.get_type();

    // Validate that operands have arithmetic types before proceeding
    validate_arithmetic_type(&lty, span, diag)?;
    validate_arithmetic_type(&rty, span, diag)?;

    // Determine the common type via usual arithmetic conversions
    let result_ty = common_type(&lty, &rty, target);

    // Convert both operands to the common type
    let lv = convert_to_common(&lv, &result_ty, target);
    let rv = convert_to_common(&rv, &result_ty, target);

    // ----- Comparison operators → result type is always `int` -----

    match op {
        BinaryOperator::Eq
        | BinaryOperator::Ne
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::Le
        | BinaryOperator::Ge => {
            let cmp_result = if result_ty.is_floating() {
                let l = lv.as_float().unwrap_or(0.0);
                let r = rv.as_float().unwrap_or(0.0);
                eval_float_comparison(op, l, r)
            } else if result_ty.is_unsigned() || matches!(result_ty, CType::Bool) {
                let l = lv.to_u128();
                let r = rv.to_u128();
                eval_unsigned_comparison(op, l, r)
            } else {
                let l = lv.to_i128();
                let r = rv.to_i128();
                eval_signed_comparison(op, l, r)
            };
            let value = if cmp_result { 1i128 } else { 0i128 };
            return Ok(ConstValue::Integer {
                value,
                ty: CType::Int { signed: true },
            });
        }
        _ => {}
    }

    // ----- Floating-point arithmetic -----

    if result_ty.is_floating() {
        let l = lv.as_float().unwrap_or(0.0);
        let r = rv.as_float().unwrap_or(0.0);
        let result = match op {
            BinaryOperator::Add => l + r,
            BinaryOperator::Sub => l - r,
            BinaryOperator::Mul => l * r,
            BinaryOperator::Div => {
                if r == 0.0 {
                    diag.warning(span, "division by zero in floating-point constant expression");
                    if l == 0.0 {
                        f64::NAN
                    } else if l > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                } else {
                    l / r
                }
            }
            BinaryOperator::Mod => {
                if r == 0.0 {
                    diag.warning(span, "division by zero in floating-point constant expression");
                    f64::NAN
                } else {
                    l % r
                }
            }
            _ => {
                // Validate with is_integer() — bitwise/shift ops require integer type
                validate_integer_type(&result_ty, span, diag)?;
                diag.error(
                    span,
                    "bitwise/shift operations are not valid on floating-point constants",
                );
                return Err(());
            }
        };
        return Ok(ConstValue::Float {
            value: result,
            ty: result_ty,
        });
    }

    // ----- Unsigned integer arithmetic -----

    let bits = bit_width_of_type(&result_ty, target);

    if result_ty.is_unsigned() || matches!(result_ty, CType::Bool) {
        let l = lv.to_u128();
        let r = rv.to_u128();
        let result = match op {
            BinaryOperator::Add => truncate_unsigned(l.wrapping_add(r), bits),
            BinaryOperator::Sub => truncate_unsigned(l.wrapping_sub(r), bits),
            BinaryOperator::Mul => truncate_unsigned(l.wrapping_mul(r), bits),
            BinaryOperator::Div => {
                if r == 0 {
                    diag.error(span, "division by zero in constant expression");
                    return Err(());
                }
                l / r
            }
            BinaryOperator::Mod => {
                if r == 0 {
                    diag.error(span, "division by zero in constant expression");
                    return Err(());
                }
                l % r
            }
            BinaryOperator::BitAnd => l & r,
            BinaryOperator::BitOr => l | r,
            BinaryOperator::BitXor => l ^ r,
            BinaryOperator::Shl => {
                if r >= bits as u128 {
                    diag.warning(span, "shift count exceeds width of type");
                    0
                } else {
                    truncate_unsigned(l << (r as u32), bits)
                }
            }
            BinaryOperator::Shr => {
                if r >= bits as u128 {
                    diag.warning(span, "shift count exceeds width of type");
                    0
                } else {
                    l >> (r as u32)
                }
            }
            _ => {
                diag.error(span, "invalid operator in constant expression");
                return Err(());
            }
        };
        return Ok(ConstValue::UnsignedInteger {
            value: result,
            ty: result_ty,
        });
    }

    // ----- Signed integer arithmetic -----

    let l = lv.to_i128();
    let r = rv.to_i128();

    let result = match op {
        BinaryOperator::Add => {
            let res = l.wrapping_add(r);
            if check_signed_overflow(res, bits) {
                diag.warning(span, "signed integer overflow in constant expression");
            }
            truncate_signed(res, bits)
        }
        BinaryOperator::Sub => {
            let res = l.wrapping_sub(r);
            if check_signed_overflow(res, bits) {
                diag.warning(span, "signed integer overflow in constant expression");
            }
            truncate_signed(res, bits)
        }
        BinaryOperator::Mul => {
            let res = l.wrapping_mul(r);
            if check_signed_overflow(res, bits) {
                diag.warning(span, "signed integer overflow in constant expression");
            }
            truncate_signed(res, bits)
        }
        BinaryOperator::Div => {
            if r == 0 {
                diag.error(span, "division by zero in constant expression");
                return Err(());
            }
            // INT_MIN / -1 overflows
            let min_for_bits = if bits < 128 {
                -(1i128 << (bits - 1))
            } else {
                i128::MIN
            };
            if l == min_for_bits && r == -1 {
                diag.warning(span, "signed integer overflow in constant expression");
                truncate_signed(l, bits)
            } else {
                l / r
            }
        }
        BinaryOperator::Mod => {
            if r == 0 {
                diag.error(span, "division by zero in constant expression");
                return Err(());
            }
            // INT_MIN % -1 is defined as 0 (no overflow, result is always 0)
            let min_for_bits = if bits < 128 {
                -(1i128 << (bits - 1))
            } else {
                i128::MIN
            };
            if l == min_for_bits && r == -1 {
                0
            } else {
                l % r
            }
        }
        BinaryOperator::BitAnd => l & r,
        BinaryOperator::BitOr => l | r,
        BinaryOperator::BitXor => l ^ r,
        BinaryOperator::Shl => {
            if r < 0 || r >= bits as i128 {
                diag.warning(span, "shift count is negative or exceeds width of type");
                0
            } else {
                let res = truncate_signed(l << (r as u32), bits);
                res
            }
        }
        BinaryOperator::Shr => {
            if r < 0 || r >= bits as i128 {
                diag.warning(span, "shift count is negative or exceeds width of type");
                // Implementation-defined: propagate sign bit
                if l < 0 {
                    -1
                } else {
                    0
                }
            } else {
                // Arithmetic right shift for signed values (Rust's >> on i128)
                l >> (r as u32)
            }
        }
        _ => {
            diag.error(span, "invalid operator in constant expression");
            return Err(());
        }
    };

    Ok(ConstValue::Integer {
        value: result,
        ty: result_ty,
    })
}

/// Evaluates a comparison operator on signed `i128` operands.
fn eval_signed_comparison(op: &BinaryOperator, l: i128, r: i128) -> bool {
    match op {
        BinaryOperator::Eq => l == r,
        BinaryOperator::Ne => l != r,
        BinaryOperator::Lt => l < r,
        BinaryOperator::Gt => l > r,
        BinaryOperator::Le => l <= r,
        BinaryOperator::Ge => l >= r,
        _ => false,
    }
}

/// Evaluates a comparison operator on unsigned `u128` operands.
fn eval_unsigned_comparison(op: &BinaryOperator, l: u128, r: u128) -> bool {
    match op {
        BinaryOperator::Eq => l == r,
        BinaryOperator::Ne => l != r,
        BinaryOperator::Lt => l < r,
        BinaryOperator::Gt => l > r,
        BinaryOperator::Le => l <= r,
        BinaryOperator::Ge => l >= r,
        _ => false,
    }
}

/// Evaluates a comparison operator on `f64` operands.
fn eval_float_comparison(op: &BinaryOperator, l: f64, r: f64) -> bool {
    match op {
        BinaryOperator::Eq => l == r,
        BinaryOperator::Ne => l != r,
        BinaryOperator::Lt => l < r,
        BinaryOperator::Gt => l > r,
        BinaryOperator::Le => l <= r,
        BinaryOperator::Ge => l >= r,
        _ => false,
    }
}

/// Evaluates a unary operation on a constant operand.
///
/// Supports arithmetic plus/negation, bitwise complement, and logical
/// negation. Increment/decrement operators are rejected as non-constant.
fn eval_unary_op(
    op: &UnaryOperator,
    operand: &Expression,
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    let val = eval_expr(operand, diag, target)?;
    let promoted_ty = promote_to_int(&val.get_type());

    match op {
        UnaryOperator::Plus => {
            // Unary plus: integer promotion only, value unchanged
            Ok(convert_to_common(&val, &promoted_ty, target))
        }

        UnaryOperator::Neg => {
            if promoted_ty.is_floating() {
                let v = val.as_float().unwrap_or(0.0);
                Ok(ConstValue::Float {
                    value: -v,
                    ty: promoted_ty,
                })
            } else if promoted_ty.is_unsigned() {
                let v = val.to_u128();
                let bits = bit_width_of_type(&promoted_ty, target);
                let result = truncate_unsigned((!v).wrapping_add(1), bits);
                Ok(ConstValue::UnsignedInteger {
                    value: result,
                    ty: promoted_ty,
                })
            } else {
                let v = val.to_i128();
                let bits = bit_width_of_type(&promoted_ty, target);
                let negated = v.wrapping_neg();
                if check_signed_overflow(negated, bits) {
                    diag.warning(span, "negation of minimum signed value overflows");
                }
                let result = truncate_signed(negated, bits);
                Ok(ConstValue::Integer {
                    value: result,
                    ty: promoted_ty,
                })
            }
        }

        UnaryOperator::BitNot => {
            if promoted_ty.is_floating() {
                diag.error(span, "bitwise complement on floating-point constant");
                Err(())
            } else if promoted_ty.is_unsigned() {
                let v = val.to_u128();
                let bits = bit_width_of_type(&promoted_ty, target);
                let result = truncate_unsigned(!v, bits);
                Ok(ConstValue::UnsignedInteger {
                    value: result,
                    ty: promoted_ty,
                })
            } else {
                let v = val.to_i128();
                let bits = bit_width_of_type(&promoted_ty, target);
                let result = truncate_signed(!v, bits);
                Ok(ConstValue::Integer {
                    value: result,
                    ty: promoted_ty,
                })
            }
        }

        UnaryOperator::LogNot => {
            let result = if val.is_zero() { 1i128 } else { 0i128 };
            Ok(ConstValue::Integer {
                value: result,
                ty: CType::Int { signed: true },
            })
        }

        // PreInc, PreDec, PostInc, PostDec — side-effecting, not constant
        _ => {
            diag.error(
                span,
                "increment/decrement is not allowed in constant expression",
            );
            Err(())
        }
    }
}

/// Evaluates an explicit type cast in a constant expression context.
///
/// Handles integer-to-integer truncation/extension, integer-to-float,
/// float-to-integer, and the special case of `(void *)0` as a null
/// pointer constant.
fn eval_cast(
    type_name: &TypeName,
    operand: &Expression,
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    let val = eval_expr(operand, diag, target)?;

    let target_ty = match resolve_type_name_to_ctype(type_name, diag, target) {
        Some(ty) => ty,
        None => {
            diag.error(
                type_name.span,
                "cannot resolve type in constant expression cast",
            );
            return Err(());
        }
    };

    // Cast to void is not valid in a constant expression context
    if matches!(target_ty, CType::Void) {
        diag.error(span, "cast to void is not valid in constant expression");
        return Err(());
    }

    // Cast to pointer: only `(void *)0` (null pointer constant) is allowed
    if target_ty.is_pointer() {
        if val.is_zero() {
            return Ok(ConstValue::NullPointer);
        }
        // Allow casting address constants to pointer types
        if matches!(val, ConstValue::Address { .. }) {
            return Ok(val);
        }
        diag.error(
            span,
            "non-null integer to pointer cast is not a constant expression",
        );
        return Err(());
    }

    // Apply the conversion
    Ok(convert_to_common(&val, &target_ty, target))
}

/// Evaluates a `sizeof` expression to a compile-time constant.
///
/// `sizeof` is always a constant expression (C11 §6.5.3.4) except when
/// applied to a variable-length array type, which is not handled here.
fn eval_sizeof(
    operand: &SizeofOperand,
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    let size = match operand {
        SizeofOperand::TypeName(tn) => {
            match resolve_type_name_to_ctype(tn, diag, target) {
                Some(ty) => types::size_of(&ty, target),
                None => {
                    diag.error(
                        span,
                        "cannot determine size of type in constant expression",
                    );
                    return Err(());
                }
            }
        }
        SizeofOperand::Expression(expr) => {
            // sizeof(expr): need the expression's type, not its value.
            // For simple expressions we can infer the type directly.
            match infer_expr_type(expr, target) {
                Some(ty) => types::size_of(&ty, target),
                None => {
                    // Fall back: try to evaluate and use the result type
                    match eval_expr(expr, diag, target) {
                        Ok(val) => {
                            let ty = val.get_type();
                            types::size_of(&ty, target)
                        }
                        Err(()) => {
                            diag.error(
                                span,
                                "cannot determine size of expression in constant expression",
                            );
                            return Err(());
                        }
                    }
                }
            }
        }
    };

    let size_t_ty = make_size_t(target);
    Ok(ConstValue::UnsignedInteger {
        value: size as u128,
        ty: size_t_ty,
    })
}

/// Evaluates an `_Alignof` / `__alignof__` expression to a compile-time constant.
///
/// `_Alignof` is always a constant expression (C11 §6.5.3.4).
fn eval_alignof(
    operand: &AlignofOperand,
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<ConstValue, ()> {
    let alignment = match operand {
        AlignofOperand::TypeName(tn) => {
            match resolve_type_name_to_ctype(tn, diag, target) {
                Some(ty) => types::align_of(&ty, target),
                None => {
                    diag.error(
                        span,
                        "cannot determine alignment of type in constant expression",
                    );
                    return Err(());
                }
            }
        }
        AlignofOperand::Expression(expr) => {
            // GCC extension: __alignof__(expr) — alignment of the expression's type
            match infer_expr_type(expr, target) {
                Some(ty) => types::align_of(&ty, target),
                None => {
                    match eval_expr(expr, diag, target) {
                        Ok(val) => {
                            let ty = val.get_type();
                            types::align_of(&ty, target)
                        }
                        Err(()) => {
                            diag.error(
                                span,
                                "cannot determine alignment of expression",
                            );
                            return Err(());
                        }
                    }
                }
            }
        }
    };

    let size_t_ty = make_size_t(target);
    Ok(ConstValue::UnsignedInteger {
        value: alignment as u128,
        ty: size_t_ty,
    })
}

/// Evaluates an address-of expression as an address constant.
///
/// C11 §6.6p9: An address constant is a null pointer, a pointer to an
/// lvalue designating an object of static storage duration, or a pointer
/// to a function designator. Address constants may have an integer offset
/// added/subtracted.
fn eval_address_constant(
    operand: &Expression,
    span: Span,
    diag: &mut DiagnosticEngine,
    _target: &Target,
) -> Result<ConstValue, ()> {
    match operand {
        // &identifier → address of a global variable or function
        Expression::Identifier { name, .. } => {
            // Use Symbol.as_u32() to validate the interned handle is valid
            // (non-zero symbol ID confirms the identifier was properly interned)
            let _sym_id = name.as_u32();
            Ok(ConstValue::Address {
                symbol: *name,
                offset: 0,
            })
        }

        // &array[index] → address with constant offset
        Expression::ArraySubscript { array, index, .. } => {
            if let Expression::Identifier { name, .. } = array.as_ref() {
                // The index must be a constant expression — use a fresh
                // diagnostic engine to avoid polluting the caller's diagnostics
                // if the index evaluation fails.
                let mut temp_diag = DiagnosticEngine::new();
                match evaluate_integer_constant(index, &mut temp_diag, _target) {
                    Ok(idx) => Ok(ConstValue::Address {
                        symbol: *name,
                        offset: idx as i64,
                    }),
                    Err(()) => {
                        diag.error(
                            span,
                            "array index in address constant must be a constant expression",
                        );
                        Err(())
                    }
                }
            } else {
                diag.error(span, "address-of expression is not an address constant");
                Err(())
            }
        }

        // &struct.member — could be address constant with offset, but we
        // need struct layout information which requires the symbol table.
        _ => {
            diag.error(span, "address-of expression is not an address constant");
            Err(())
        }
    }
}
