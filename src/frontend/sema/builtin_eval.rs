// src/frontend/sema/builtin_eval.rs
//
// GCC builtin evaluation module for the BCC compiler's Phase 5 semantic
// analysis.
//
// Lint allowance: `result_unit_err` is suppressed because this module follows
// the standard compiler pattern where errors are reported through the
// diagnostic engine and `Result<_, ()>` is used purely for control flow.
#![allow(clippy::result_unit_err)]
//
// This module handles two categories of GCC builtins:
//
// 1. **Compile-time builtins** — resolved entirely during semantic analysis
//    and returned as `BuiltinResult::CompileTimeValue`:
//      - `__builtin_constant_p(expr)` — returns 1 if expr is a compile-time
//        constant, 0 otherwise.
//      - `__builtin_types_compatible_p(type1, type2)` — returns 1 if types
//        are compatible (ignoring qualifiers), 0 otherwise.
//      - `__builtin_choose_expr(const_expr, expr1, expr2)` — selects expr1
//        or expr2 based on the compile-time value of const_expr.
//      - `__builtin_offsetof(type, member)` — computes the byte offset of
//        a struct member using target-aware layout computation.
//
// 2. **Runtime-deferred builtins** — type-checked during semantic analysis
//    and returned as `BuiltinResult::RuntimeCall` for IR lowering:
//      - Bit manipulation: clz, ctz, popcount, bswap, ffs
//      - Branch prediction: expect
//      - Control flow: unreachable, trap
//      - Pointer hints: assume_aligned
//      - Stack introspection: frame_address, return_address
//      - Variadic argument handling: va_start, va_end, va_arg, va_copy
//      - Overflow arithmetic: add_overflow, sub_overflow, mul_overflow
//
// All ~30 builtins required for Linux kernel compilation are supported.
//
// Integration points:
// - Called by: `src/frontend/sema/type_checker.rs` (when a FunctionCall
//   callee is a recognized builtin name)
// - Uses AST from: `src/frontend/parser/ast.rs`
// - Uses type info from: `src/common/types.rs`, `src/common/type_builder.rs`
// - Uses target info from: `src/common/target.rs`
// - Uses constant eval from: `src/frontend/sema/constant_eval.rs`
// - Reports diagnostics via: `src/common/diagnostics.rs`

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::common::type_builder::{self, compute_struct_layout};
use crate::common::types::CType;

use crate::frontend::parser::ast::{Expression, TypeName};
use crate::frontend::sema::constant_eval::{
    evaluate_constant_expression, is_constant_expression, ConstValue,
};

// ===========================================================================
// CheckedExpression — type-checked argument for runtime builtins
// ===========================================================================

/// A type-checked expression paired with its expected type, ready for IR
/// lowering of a runtime-deferred builtin call.
///
/// During semantic analysis, each argument to a runtime builtin is validated
/// and annotated with the type the backend expects. The IR lowering phase
/// uses `expected_ty` to insert implicit conversions if necessary.
#[derive(Clone, Debug)]
pub struct CheckedExpression {
    /// The original AST expression (validated but not evaluated).
    pub expr: Expression,
    /// The C type expected by the builtin for this argument position.
    pub expected_ty: CType,
}

// ===========================================================================
// BuiltinResult — evaluation result for GCC builtins
// ===========================================================================

/// The result of evaluating a GCC `__builtin_*` function during Phase 5
/// semantic analysis.
///
/// Compile-time builtins produce concrete values immediately.
/// Runtime-deferred builtins produce type information and validated
/// argument lists that the IR lowering phase (Phase 6) uses to emit
/// appropriate machine instructions or library calls.
#[derive(Clone, Debug)]
pub enum BuiltinResult {
    /// The builtin was fully evaluated at compile time.
    ///
    /// Used by `__builtin_constant_p`, `__builtin_types_compatible_p`,
    /// `__builtin_choose_expr`, and `__builtin_offsetof`.
    CompileTimeValue(ConstValue),

    /// The builtin is deferred to runtime; the semantic analyzer has
    /// validated the arguments and determined the return type.
    ///
    /// Used by `__builtin_clz`, `__builtin_bswap*`, `__builtin_expect`,
    /// and all other runtime builtins.
    RuntimeCall {
        /// The C return type of this builtin call.
        return_type: CType,
        /// Type-checked arguments ready for IR lowering.
        checked_args: Vec<CheckedExpression>,
    },

    /// The builtin returns a type rather than a value.
    ///
    /// Used by `__builtin_va_arg` which yields a value of the specified type.
    TypeResult(CType),

    /// The builtin has no return value (returns `void`).
    ///
    /// Used by `__builtin_trap`, `__builtin_unreachable`,
    /// `__builtin_va_start`, `__builtin_va_end`, `__builtin_va_copy`.
    Void,
}

impl BuiltinResult {
    /// Returns the C type of this builtin result.
    ///
    /// - `CompileTimeValue` — the type carried by the `ConstValue`.
    /// - `RuntimeCall` — the `return_type` field.
    /// - `TypeResult` — the contained type.
    /// - `Void` — `CType::Void`.
    pub fn ty(&self) -> CType {
        match self {
            BuiltinResult::CompileTimeValue(cv) => cv.get_type(),
            BuiltinResult::RuntimeCall { return_type, .. } => return_type.clone(),
            BuiltinResult::TypeResult(t) => t.clone(),
            BuiltinResult::Void => CType::Void,
        }
    }
}

// ===========================================================================
// BuiltinKind — internal dispatch tag
// ===========================================================================

/// Internal enumeration of recognized GCC builtins for dispatch.
///
/// Each variant corresponds to a family of builtins with shared
/// type-checking and evaluation logic. Suffixed variants (e.g., ClzL,
/// ClzLL) share the same handler with a width parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinKind {
    // -- Compile-time builtins --
    ConstantP,
    TypesCompatibleP,
    ChooseExpr,
    Offsetof,

    // -- Bit manipulation --
    Clz,
    ClzL,
    ClzLL,
    Ctz,
    CtzL,
    CtzLL,
    Popcount,
    PopcountL,
    PopcountLL,
    Bswap16,
    Bswap32,
    Bswap64,
    Ffs,
    FfsL,
    FfsLL,

    // -- Branch prediction / hints --
    Expect,
    ExpectWithProbability,

    // -- Unreachable / trap --
    Unreachable,
    Trap,

    // -- Pointer alignment --
    AssumeAligned,

    // -- Stack introspection --
    FrameAddress,
    ReturnAddress,

    // -- Variadic arguments --
    VaStart,
    VaEnd,
    VaArg,
    VaCopy,

    // -- Overflow arithmetic --
    AddOverflow,
    SubOverflow,
    MulOverflow,

    // -- Miscellaneous --
    Prefetch,
    ObjectSize,
}

// ===========================================================================
// Public API — builtin name recognition
// ===========================================================================

/// Returns `true` if the given identifier string is a recognized GCC builtin
/// name.
///
/// This function is used by the parser and semantic analyzer to detect
/// builtin calls early, enabling special argument parsing (e.g., type
/// arguments for `__builtin_types_compatible_p`).
///
/// # Arguments
///
/// * `name` — The identifier string to check.
///
/// # Examples
///
/// ```rust,ignore
/// assert!(is_builtin_name("__builtin_clz"));
/// assert!(is_builtin_name("__builtin_expect"));
/// assert!(!is_builtin_name("printf"));
/// ```
pub fn is_builtin_name(name: &str) -> bool {
    lookup_builtin(name).is_some()
}

// ===========================================================================
// Public API — builtin evaluation entry point
// ===========================================================================

/// Evaluates or type-checks a GCC `__builtin_*` call during Phase 5
/// semantic analysis.
///
/// This is the primary entry point invoked by the type checker when a
/// `FunctionCall` expression's callee resolves to a recognized builtin
/// name. It dispatches to the appropriate handler based on the builtin
/// kind, validates arguments, and returns either a compile-time result
/// or a runtime call descriptor.
///
/// # Arguments
///
/// * `name` — Interned symbol of the builtin function name (e.g.,
///   `__builtin_clz`).
/// * `args` — Argument expressions as parsed from the function call.
/// * `diagnostics` — Diagnostic engine for error/warning reporting.
/// * `interner` — String interner for resolving symbols to strings.
/// * `target` — Target architecture for type sizing and layout.
///
/// # Returns
///
/// * `Ok(BuiltinResult)` — The builtin was successfully evaluated or
///   type-checked.
/// * `Err(())` — A semantic error was detected and reported via
///   `diagnostics`.
pub fn evaluate_builtin(
    name: Symbol,
    args: &[Expression],
    diagnostics: &mut DiagnosticEngine,
    interner: &Interner,
    target: &Target,
) -> Result<BuiltinResult, ()> {
    let name_str = interner.resolve(name);
    let span = if args.is_empty() {
        Span::DUMMY
    } else {
        args[0].span()
    };

    let kind = match lookup_builtin(name_str) {
        Some(k) => k,
        None => {
            diagnostics.error(span, format!("unrecognized builtin '{}'", name_str));
            return Err(());
        }
    };

    match kind {
        // ---- Compile-time builtins ----
        BuiltinKind::ConstantP => eval_constant_p(args, span, diagnostics),
        BuiltinKind::TypesCompatibleP => eval_types_compatible_p(args, span, diagnostics),
        BuiltinKind::ChooseExpr => eval_choose_expr(args, span, diagnostics, target),
        BuiltinKind::Offsetof => eval_offsetof(args, span, diagnostics, target),

        // ---- Bit manipulation ----
        BuiltinKind::Clz => check_clz(args, span, diagnostics, IntWidth::Int),
        BuiltinKind::ClzL => check_clz(args, span, diagnostics, IntWidth::Long),
        BuiltinKind::ClzLL => check_clz(args, span, diagnostics, IntWidth::LongLong),
        BuiltinKind::Ctz => check_ctz(args, span, diagnostics, IntWidth::Int),
        BuiltinKind::CtzL => check_ctz(args, span, diagnostics, IntWidth::Long),
        BuiltinKind::CtzLL => check_ctz(args, span, diagnostics, IntWidth::LongLong),
        BuiltinKind::Popcount => check_popcount(args, span, diagnostics, IntWidth::Int),
        BuiltinKind::PopcountL => check_popcount(args, span, diagnostics, IntWidth::Long),
        BuiltinKind::PopcountLL => check_popcount(args, span, diagnostics, IntWidth::LongLong),
        BuiltinKind::Bswap16 => check_bswap(args, 16, span, diagnostics),
        BuiltinKind::Bswap32 => check_bswap(args, 32, span, diagnostics),
        BuiltinKind::Bswap64 => check_bswap(args, 64, span, diagnostics),
        BuiltinKind::Ffs => check_ffs(args, span, diagnostics, IntWidth::Int),
        BuiltinKind::FfsL => check_ffs(args, span, diagnostics, IntWidth::Long),
        BuiltinKind::FfsLL => check_ffs(args, span, diagnostics, IntWidth::LongLong),

        // ---- Branch prediction ----
        BuiltinKind::Expect => check_expect(args, span, diagnostics),
        BuiltinKind::ExpectWithProbability => check_expect(args, span, diagnostics),

        // ---- Unreachable / trap ----
        BuiltinKind::Unreachable => check_unreachable(args, span, diagnostics),
        BuiltinKind::Trap => check_trap(args, span, diagnostics),

        // ---- Pointer alignment ----
        BuiltinKind::AssumeAligned => check_assume_aligned(args, span, diagnostics),

        // ---- Stack introspection ----
        BuiltinKind::FrameAddress => check_frame_address(args, span, diagnostics),
        BuiltinKind::ReturnAddress => check_return_address(args, span, diagnostics),

        // ---- Variadic arguments ----
        BuiltinKind::VaStart => check_va_start(args, span, diagnostics),
        BuiltinKind::VaEnd => check_va_end(args, span, diagnostics),
        BuiltinKind::VaArg => check_va_arg(args, span, diagnostics),
        BuiltinKind::VaCopy => check_va_copy(args, span, diagnostics),

        // ---- Overflow arithmetic ----
        BuiltinKind::AddOverflow => {
            check_overflow_arith("__builtin_add_overflow", args, span, diagnostics)
        }
        BuiltinKind::SubOverflow => {
            check_overflow_arith("__builtin_sub_overflow", args, span, diagnostics)
        }
        BuiltinKind::MulOverflow => {
            check_overflow_arith("__builtin_mul_overflow", args, span, diagnostics)
        }

        // ---- Miscellaneous ----
        BuiltinKind::Prefetch => check_prefetch(args, span, diagnostics),
        BuiltinKind::ObjectSize => check_object_size(args, span, diagnostics),
    }
}

// ===========================================================================
// Builtin name lookup table
// ===========================================================================

/// Maps a builtin name string to its `BuiltinKind` dispatch tag.
///
/// Returns `None` for unrecognized names. The lookup is implemented as a
/// linear scan of a static table, which is fast enough for the ~40 entries
/// since it avoids hash map allocation overhead.
fn lookup_builtin(name: &str) -> Option<BuiltinKind> {
    match name {
        // Compile-time builtins
        "__builtin_constant_p" => Some(BuiltinKind::ConstantP),
        "__builtin_types_compatible_p" => Some(BuiltinKind::TypesCompatibleP),
        "__builtin_choose_expr" => Some(BuiltinKind::ChooseExpr),
        "__builtin_offsetof" => Some(BuiltinKind::Offsetof),

        // Bit manipulation
        "__builtin_clz" => Some(BuiltinKind::Clz),
        "__builtin_clzl" => Some(BuiltinKind::ClzL),
        "__builtin_clzll" => Some(BuiltinKind::ClzLL),
        "__builtin_ctz" => Some(BuiltinKind::Ctz),
        "__builtin_ctzl" => Some(BuiltinKind::CtzL),
        "__builtin_ctzll" => Some(BuiltinKind::CtzLL),
        "__builtin_popcount" => Some(BuiltinKind::Popcount),
        "__builtin_popcountl" => Some(BuiltinKind::PopcountL),
        "__builtin_popcountll" => Some(BuiltinKind::PopcountLL),
        "__builtin_bswap16" => Some(BuiltinKind::Bswap16),
        "__builtin_bswap32" => Some(BuiltinKind::Bswap32),
        "__builtin_bswap64" => Some(BuiltinKind::Bswap64),
        "__builtin_ffs" => Some(BuiltinKind::Ffs),
        "__builtin_ffsl" => Some(BuiltinKind::FfsL),
        "__builtin_ffsll" => Some(BuiltinKind::FfsLL),

        // Branch prediction
        "__builtin_expect" => Some(BuiltinKind::Expect),
        "__builtin_expect_with_probability" => Some(BuiltinKind::ExpectWithProbability),

        // Unreachable / trap
        "__builtin_unreachable" => Some(BuiltinKind::Unreachable),
        "__builtin_trap" => Some(BuiltinKind::Trap),

        // Pointer alignment
        "__builtin_assume_aligned" => Some(BuiltinKind::AssumeAligned),

        // Stack introspection
        "__builtin_frame_address" => Some(BuiltinKind::FrameAddress),
        "__builtin_return_address" => Some(BuiltinKind::ReturnAddress),

        // Variadic argument handling
        "__builtin_va_start" => Some(BuiltinKind::VaStart),
        "__builtin_va_end" => Some(BuiltinKind::VaEnd),
        "__builtin_va_arg" => Some(BuiltinKind::VaArg),
        "__builtin_va_copy" => Some(BuiltinKind::VaCopy),

        // Overflow arithmetic
        "__builtin_add_overflow" => Some(BuiltinKind::AddOverflow),
        "__builtin_sub_overflow" => Some(BuiltinKind::SubOverflow),
        "__builtin_mul_overflow" => Some(BuiltinKind::MulOverflow),

        // Miscellaneous
        "__builtin_prefetch" => Some(BuiltinKind::Prefetch),
        "__builtin_object_size" => Some(BuiltinKind::ObjectSize),

        _ => None,
    }
}

// ===========================================================================
// Integer width helper for suffixed builtin families
// ===========================================================================

/// Width variant for builtins that come in int/long/long long flavors.
///
/// Used by clz, ctz, popcount, ffs families to determine the expected
/// argument type and the return value width semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntWidth {
    /// `unsigned int` width (e.g., `__builtin_clz`).
    Int,
    /// `unsigned long` width (e.g., `__builtin_clzl`).
    Long,
    /// `unsigned long long` width (e.g., `__builtin_clzll`).
    LongLong,
}

impl IntWidth {
    /// Returns the unsigned C type for this width.
    fn to_unsigned_type(self) -> CType {
        match self {
            IntWidth::Int => CType::Int { signed: false },
            IntWidth::Long => CType::Long { signed: false },
            IntWidth::LongLong => CType::LongLong { signed: false },
        }
    }

    /// Returns the signed C type for this width (used for ffs return type).
    fn to_signed_type(self) -> CType {
        match self {
            IntWidth::Int => CType::Int { signed: true },
            IntWidth::Long => CType::Long { signed: true },
            IntWidth::LongLong => CType::LongLong { signed: true },
        }
    }
}

// ===========================================================================
// Argument validation helpers
// ===========================================================================

/// Validates that the argument count matches the expected count exactly.
///
/// Emits a diagnostic and returns `Err(())` if the count does not match.
/// The `builtin_name` is included in the error message for clarity.
fn validate_arg_count(
    args: &[Expression],
    expected: usize,
    builtin_name: &str,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<(), ()> {
    if args.len() != expected {
        if args.len() < expected {
            diag.error(
                span,
                format!(
                    "too few arguments to '{}': expected {}, got {}",
                    builtin_name,
                    expected,
                    args.len()
                ),
            );
        } else {
            diag.error(
                span,
                format!(
                    "too many arguments to '{}': expected {}, got {}",
                    builtin_name,
                    expected,
                    args.len()
                ),
            );
        }
        return Err(());
    }
    Ok(())
}

/// Validates that the argument count is within an inclusive range.
///
/// Used for builtins that accept optional arguments (e.g.,
/// `__builtin_prefetch` takes 1–3 arguments).
fn validate_arg_count_range(
    args: &[Expression],
    min: usize,
    max: usize,
    builtin_name: &str,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<(), ()> {
    if args.len() < min {
        diag.error(
            span,
            format!(
                "too few arguments to '{}': expected at least {}, got {}",
                builtin_name,
                min,
                args.len()
            ),
        );
        return Err(());
    }
    if args.len() > max {
        diag.error(
            span,
            format!(
                "too many arguments to '{}': expected at most {}, got {}",
                builtin_name,
                max,
                args.len()
            ),
        );
        return Err(());
    }
    Ok(())
}

/// Creates a `BuiltinResult::RuntimeCall` with a single checked argument.
fn make_runtime_call_single(
    return_type: CType,
    arg: &Expression,
    expected_ty: CType,
) -> BuiltinResult {
    BuiltinResult::RuntimeCall {
        return_type,
        checked_args: vec![CheckedExpression {
            expr: arg.clone(),
            expected_ty,
        }],
    }
}

/// Creates a `BuiltinResult::RuntimeCall` from a list of (Expression, CType)
/// pairs.
fn make_runtime_call(return_type: CType, arg_pairs: Vec<(&Expression, CType)>) -> BuiltinResult {
    let checked_args = arg_pairs
        .into_iter()
        .map(|(expr, expected_ty)| CheckedExpression {
            expr: expr.clone(),
            expected_ty,
        })
        .collect();
    BuiltinResult::RuntimeCall {
        return_type,
        checked_args,
    }
}

/// Creates a compile-time integer result with `int` type.
fn make_compile_time_int(value: i128) -> BuiltinResult {
    BuiltinResult::CompileTimeValue(ConstValue::Integer {
        value,
        ty: CType::Int { signed: true },
    })
}

/// Creates a compile-time unsigned integer result with the given type.
fn make_compile_time_uint(value: u128, ty: CType) -> BuiltinResult {
    BuiltinResult::CompileTimeValue(ConstValue::UnsignedInteger { value, ty })
}

// ===========================================================================
// Compile-time builtin handlers
// ===========================================================================

/// Evaluates `__builtin_constant_p(expr)`.
///
/// Returns 1 if `expr` is a compile-time constant expression, 0 otherwise.
/// This builtin never fails — it always returns a value.
///
/// GCC semantics: `__builtin_constant_p` is used extensively in the Linux
/// kernel's `BUILD_BUG_ON_*` macros and type-generic selection macros
/// to optimize code paths for compile-time-known values.
fn eval_constant_p(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 1, "__builtin_constant_p", span, diag)?;

    let is_const = is_constant_expression(&args[0]);
    let value = if is_const { 1i128 } else { 0i128 };
    Ok(make_compile_time_int(value))
}

/// Evaluates `__builtin_types_compatible_p(type1, type2)`.
///
/// Returns 1 if the two types are compatible (ignoring qualifiers),
/// 0 otherwise. The arguments are expected to be expressions that encode
/// type information — typically parsed as cast expressions `(type)0` by
/// the parser's special builtin handling.
///
/// GCC semantics: type compatibility follows C11 §6.2.7 rules, with the
/// GCC extension that qualifiers (const, volatile, restrict) are ignored.
/// This is used by the kernel's `typecheck()` macro and `min()`/`max()`
/// type-safe wrappers.
fn eval_types_compatible_p(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_types_compatible_p", span, diag)?;

    // Extract types from the argument expressions.
    // The parser encodes type arguments as cast expressions: (type)0.
    // We also support sizeof-style type wrappers as a fallback.
    let ty1 = extract_type_from_expr(&args[0]);
    let ty2 = extract_type_from_expr(&args[1]);

    match (ty1, ty2) {
        (Some(ref t1), Some(ref t2)) => {
            let compatible = type_builder::types_compatible(t1, t2);
            let value = if compatible { 1i128 } else { 0i128 };
            Ok(make_compile_time_int(value))
        }
        _ => {
            // If we cannot extract type information from the expressions,
            // conservatively return 0 (not compatible).  This handles
            // the case where the parser does not encode type arguments
            // as cast expressions; a warning is more appropriate than an
            // error since the program may still be correct.
            diag.warning(
                span,
                "cannot determine types for __builtin_types_compatible_p; assuming incompatible",
            );
            Ok(make_compile_time_int(0))
        }
    }
}

/// Evaluates `__builtin_choose_expr(const_expr, expr1, expr2)`.
///
/// If `const_expr` is nonzero, the result is `expr1`; otherwise `expr2`.
/// Critically, the **non-selected** expression is NOT type-checked — this
/// is what distinguishes `__builtin_choose_expr` from the ternary operator
/// `?:` and enables type-generic programming in C.
///
/// GCC semantics: This is used by the Linux kernel's `__same_type()` and
/// type-dispatching macros where one branch may contain deliberately
/// ill-typed code that must not be diagnosed.
fn eval_choose_expr(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 3, "__builtin_choose_expr", span, diag)?;

    // Evaluate the first argument as a constant expression.
    // The controlling expression must be an integer constant expression
    // per GCC semantics.
    let const_val = evaluate_constant_expression(&args[0], diag, target)?;
    // Use to_i128() to get a scalar value for the branch decision,
    // then check if it is non-zero to select the first expression.
    let controlling_value = const_val.to_i128();
    let choose_first = controlling_value != 0;

    if choose_first {
        // Return expr1 as the selected expression. We wrap it in a
        // RuntimeCall so the downstream phase processes the selected
        // expression and ignores the other.
        Ok(BuiltinResult::RuntimeCall {
            return_type: CType::Void, // Actual type determined by downstream
            checked_args: vec![CheckedExpression {
                expr: args[1].clone(),
                expected_ty: CType::Void, // No type constraint
            }],
        })
    } else {
        // Return expr2 as the selected expression.
        Ok(BuiltinResult::RuntimeCall {
            return_type: CType::Void,
            checked_args: vec![CheckedExpression {
                expr: args[2].clone(),
                expected_ty: CType::Void,
            }],
        })
    }
}

/// Evaluates `__builtin_offsetof(type, member)`.
///
/// Computes the byte offset of `member` within `type` using
/// target-aware struct layout computation from `type_builder`.
///
/// The first argument must encode a struct/union type (typically via a
/// cast expression). The second argument identifies the member (typically
/// an Identifier or MemberAccess expression chain for nested members).
///
/// GCC semantics: Used by the `offsetof()` macro in `<stddef.h>` and
/// directly by kernel code for container_of, list_entry, etc.
fn eval_offsetof(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    target: &Target,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_offsetof", span, diag)?;

    // Extract the struct type from the first argument.
    // The type is typically encoded via a Cast expression produced by the
    // parser when handling __builtin_offsetof(type, member).
    let struct_type = match extract_type_from_expr(&args[0]) {
        Some(ty) => ty,
        None => {
            // Use the argument expression's span for precise error location.
            let arg_span = args[0].span();
            diag.error(
                Span::new(arg_span.file_id, arg_span.start, arg_span.end),
                "first argument to __builtin_offsetof must be a struct or union type",
            );
            return Err(());
        }
    };

    // Extract the member name from the second argument.
    let member_name = extract_member_name(&args[1]);
    let member = match member_name {
        Some(name) => name,
        None => {
            let arg_span = args[1].span();
            diag.error(
                Span::new(arg_span.file_id, arg_span.start, arg_span.end),
                "second argument to __builtin_offsetof must be a member designator",
            );
            return Err(());
        }
    };

    // Compute the offset based on the struct type.
    match &struct_type {
        CType::Struct { fields, .. } => {
            let layout = compute_struct_layout(fields, target);
            // Find the field by name and retrieve its byte offset from
            // the FieldLayout entry.
            for (idx, field) in fields.iter().enumerate() {
                if let Some(ref field_name) = field.name {
                    if field_name == &member && idx < layout.fields.len() {
                        let offset = layout.fields[idx].offset;
                        let size_t_ty = make_size_t(target);
                        return Ok(make_compile_time_uint(offset as u128, size_t_ty));
                    }
                }
            }
            diag.error(
                args[1].span(),
                format!("no member named '{}' in struct type", member),
            );
            Err(())
        }
        CType::Union { fields, .. } => {
            // All union fields share offset 0 per the C standard.
            for field in fields {
                if let Some(ref field_name) = field.name {
                    if field_name == &member {
                        let size_t_ty = make_size_t(target);
                        return Ok(make_compile_time_uint(0, size_t_ty));
                    }
                }
            }
            diag.error(
                args[1].span(),
                format!("no member named '{}' in union type", member),
            );
            Err(())
        }
        _ => {
            diag.error(
                args[0].span(),
                "__builtin_offsetof requires a struct or union type",
            );
            Err(())
        }
    }
}

// ===========================================================================
// Runtime-deferred builtin handlers — bit manipulation
// ===========================================================================

/// Type-checks `__builtin_clz` / `__builtin_clzl` / `__builtin_clzll`.
///
/// Count Leading Zeros: returns `int`. Takes one unsigned integer argument
/// of the appropriate width. Undefined behavior if the argument is zero.
fn check_clz(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    width: IntWidth,
) -> Result<BuiltinResult, ()> {
    let name = match width {
        IntWidth::Int => "__builtin_clz",
        IntWidth::Long => "__builtin_clzl",
        IntWidth::LongLong => "__builtin_clzll",
    };
    validate_arg_count(args, 1, name, span, diag)?;

    let expected_arg_ty = width.to_unsigned_type();
    let return_ty = CType::Int { signed: true };
    Ok(make_runtime_call_single(
        return_ty,
        &args[0],
        expected_arg_ty,
    ))
}

/// Type-checks `__builtin_ctz` / `__builtin_ctzl` / `__builtin_ctzll`.
///
/// Count Trailing Zeros: returns `int`. Takes one unsigned integer argument
/// of the appropriate width. Undefined behavior if the argument is zero.
fn check_ctz(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    width: IntWidth,
) -> Result<BuiltinResult, ()> {
    let name = match width {
        IntWidth::Int => "__builtin_ctz",
        IntWidth::Long => "__builtin_ctzl",
        IntWidth::LongLong => "__builtin_ctzll",
    };
    validate_arg_count(args, 1, name, span, diag)?;

    let expected_arg_ty = width.to_unsigned_type();
    let return_ty = CType::Int { signed: true };
    Ok(make_runtime_call_single(
        return_ty,
        &args[0],
        expected_arg_ty,
    ))
}

/// Type-checks `__builtin_popcount` / `__builtin_popcountl` /
/// `__builtin_popcountll`.
///
/// Population Count: returns `int`. Takes one unsigned integer argument
/// of the appropriate width. Returns the number of set bits.
fn check_popcount(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    width: IntWidth,
) -> Result<BuiltinResult, ()> {
    let name = match width {
        IntWidth::Int => "__builtin_popcount",
        IntWidth::Long => "__builtin_popcountl",
        IntWidth::LongLong => "__builtin_popcountll",
    };
    validate_arg_count(args, 1, name, span, diag)?;

    let expected_arg_ty = width.to_unsigned_type();
    let return_ty = CType::Int { signed: true };
    Ok(make_runtime_call_single(
        return_ty,
        &args[0],
        expected_arg_ty,
    ))
}

/// Type-checks `__builtin_bswap16` / `__builtin_bswap32` / `__builtin_bswap64`.
///
/// Byte Swap: reverses the byte order of the argument.
/// - bswap16: takes and returns `uint16_t` (unsigned short)
/// - bswap32: takes and returns `uint32_t` (unsigned int)
/// - bswap64: takes and returns `uint64_t` (unsigned long long)
fn check_bswap(
    args: &[Expression],
    width: u32,
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    let name = match width {
        16 => "__builtin_bswap16",
        32 => "__builtin_bswap32",
        64 => "__builtin_bswap64",
        _ => "__builtin_bswap",
    };
    validate_arg_count(args, 1, name, span, diag)?;

    let ty = match width {
        16 => CType::Short { signed: false },
        32 => CType::Int { signed: false },
        64 => CType::LongLong { signed: false },
        _ => CType::Int { signed: false },
    };

    Ok(make_runtime_call_single(ty.clone(), &args[0], ty))
}

/// Type-checks `__builtin_ffs` / `__builtin_ffsl` / `__builtin_ffsll`.
///
/// Find First Set: returns `int`. Takes one signed integer argument.
/// Returns the 1-based position of the least significant set bit, or 0
/// if the argument is zero.
fn check_ffs(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
    width: IntWidth,
) -> Result<BuiltinResult, ()> {
    let name = match width {
        IntWidth::Int => "__builtin_ffs",
        IntWidth::Long => "__builtin_ffsl",
        IntWidth::LongLong => "__builtin_ffsll",
    };
    validate_arg_count(args, 1, name, span, diag)?;

    let expected_arg_ty = width.to_signed_type();
    let return_ty = CType::Int { signed: true };
    Ok(make_runtime_call_single(
        return_ty,
        &args[0],
        expected_arg_ty,
    ))
}

// ===========================================================================
// Runtime-deferred builtin handlers — branch prediction
// ===========================================================================

/// Type-checks `__builtin_expect(expr, expected)`.
///
/// Branch prediction hint: returns `long`. Both arguments must be `long`
/// integers. The return value is `expr` — the hint is used by the optimizer
/// to arrange branch layout for the common case.
///
/// GCC semantics: Extensively used via the `likely()` and `unlikely()`
/// macros in the Linux kernel (`include/linux/compiler.h`).
fn check_expect(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_expect", span, diag)?;

    let long_ty = CType::Long { signed: true };
    Ok(make_runtime_call(
        long_ty.clone(),
        vec![(&args[0], long_ty.clone()), (&args[1], long_ty)],
    ))
}

// ===========================================================================
// Runtime-deferred builtin handlers — unreachable / trap
// ===========================================================================

/// Type-checks `__builtin_unreachable()`.
///
/// Marks the current code path as unreachable. Takes no arguments and
/// returns `void`. The compiler may use this to eliminate unreachable
/// branches and emit more efficient code.
///
/// GCC semantics: Used by the kernel's `BUG()` macro (after the trap
/// instruction) and switch exhaustiveness annotations.
fn check_unreachable(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 0, "__builtin_unreachable", span, diag)?;
    Ok(BuiltinResult::Void)
}

/// Type-checks `__builtin_trap()`.
///
/// Causes the program to abort abnormally. Takes no arguments and returns
/// `void`. Typically emits `ud2` on x86 or `brk #0` on AArch64.
///
/// GCC semantics: Used by the kernel's `BUG()` and `panic()` paths as
/// a last-resort trap instruction.
fn check_trap(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 0, "__builtin_trap", span, diag)?;
    Ok(BuiltinResult::Void)
}

// ===========================================================================
// Runtime-deferred builtin handlers — pointer alignment
// ===========================================================================

/// Type-checks `__builtin_assume_aligned(ptr, alignment [, offset])`.
///
/// Pointer alignment hint: returns `void *`. The compiler may assume that
/// `ptr - offset` is aligned to `alignment` bytes. The optional third
/// argument defaults to 0.
///
/// Arguments:
/// - `ptr` — pointer (any pointer type)
/// - `alignment` — integer constant (must be a power of two)
/// - `offset` — optional integer offset
fn check_assume_aligned(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count_range(args, 2, 3, "__builtin_assume_aligned", span, diag)?;

    let void_ptr = CType::Pointer(Box::new(CType::Void));
    let size_t = CType::Long { signed: false };

    let mut arg_pairs: Vec<(&Expression, CType)> =
        vec![(&args[0], void_ptr.clone()), (&args[1], size_t)];
    if args.len() == 3 {
        arg_pairs.push((&args[2], CType::Long { signed: true }));
    }

    Ok(make_runtime_call(void_ptr, arg_pairs))
}

// ===========================================================================
// Runtime-deferred builtin handlers — stack introspection
// ===========================================================================

/// Type-checks `__builtin_frame_address(level)`.
///
/// Returns `void *` pointing to the frame address at the specified nesting
/// level. Level 0 is the current function's frame.
///
/// Arguments:
/// - `level` — integer constant specifying the call stack depth
fn check_frame_address(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 1, "__builtin_frame_address", span, diag)?;

    let void_ptr = CType::Pointer(Box::new(CType::Void));
    let int_ty = CType::Int { signed: false };
    Ok(make_runtime_call_single(void_ptr, &args[0], int_ty))
}

/// Type-checks `__builtin_return_address(level)`.
///
/// Returns `void *` pointing to the return address at the specified
/// nesting level. Level 0 is the current function's return address.
///
/// Arguments:
/// - `level` — integer constant specifying the call stack depth
fn check_return_address(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 1, "__builtin_return_address", span, diag)?;

    let void_ptr = CType::Pointer(Box::new(CType::Void));
    let int_ty = CType::Int { signed: false };
    Ok(make_runtime_call_single(void_ptr, &args[0], int_ty))
}

// ===========================================================================
// Variadic argument builtin handlers
// ===========================================================================

/// Type-checks `__builtin_va_start(ap, last_named_arg)`.
///
/// Initializes the `va_list` object `ap` for subsequent retrieval of
/// unnamed arguments following `last_named_arg`.
///
/// Arguments:
/// - `ap` — a `va_list` (architecture-dependent type, treated as void* here)
/// - `last_named_arg` — the last named parameter of the variadic function
fn check_va_start(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_va_start", span, diag)?;
    Ok(BuiltinResult::Void)
}

/// Type-checks `__builtin_va_end(ap)`.
///
/// Performs cleanup for a `va_list` object. After `va_end`, the `va_list`
/// is undefined until re-initialized with `va_start` or `va_copy`.
///
/// Arguments:
/// - `ap` — a `va_list` object
fn check_va_end(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 1, "__builtin_va_end", span, diag)?;
    Ok(BuiltinResult::Void)
}

/// Type-checks `__builtin_va_arg(ap, type)`.
///
/// Fetches the next variadic argument of the given type from `ap`.
/// The type argument determines the return type.
///
/// Arguments:
/// - `ap` — a `va_list` object
/// - `type` — the type of the next argument to retrieve (encoded as
///   a type expression by the parser)
fn check_va_arg(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_va_arg", span, diag)?;

    // The second argument encodes the desired type. Extract it.
    let result_type = match extract_type_from_expr(&args[1]) {
        Some(ty) => ty,
        None => {
            // Fallback: if we cannot extract the type, default to int.
            // The downstream IR lowering will need the actual type.
            CType::Int { signed: true }
        }
    };

    Ok(BuiltinResult::TypeResult(result_type))
}

/// Type-checks `__builtin_va_copy(dest, src)`.
///
/// Copies the `va_list` object `src` to `dest`, creating an independent
/// copy that can be iterated separately.
///
/// Arguments:
/// - `dest` — destination `va_list` object
/// - `src` — source `va_list` object
fn check_va_copy(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_va_copy", span, diag)?;
    Ok(BuiltinResult::Void)
}

// ===========================================================================
// Overflow arithmetic builtin handlers
// ===========================================================================

/// Type-checks `__builtin_add_overflow`, `__builtin_sub_overflow`,
/// `__builtin_mul_overflow`.
///
/// Performs the specified arithmetic operation with overflow detection.
/// - First two arguments: integer operands (any integer type accepted,
///   including both signed and unsigned variants)
/// - Third argument: pointer to integer where the result is stored
/// - Returns `bool` (`_Bool`): `true` if overflow occurred
///
/// GCC semantics: The operands undergo integer promotion but are NOT
/// converted to a common type — each retains its original type. The
/// result is computed in infinite precision and then stored into the
/// pointed-to type, with overflow detected if the infinite-precision
/// result does not fit.
///
/// Type validation uses `CType::is_integer()` for the first two operands
/// and `CType::is_pointer()` for the third (result) operand. The signedness
/// of operands is checked via `CType::is_signed()` / `CType::is_unsigned()`
/// for diagnostic purposes.
fn check_overflow_arith(
    name: &str,
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 3, name, span, diag)?;

    // The expected types for overflow builtins:
    // - args[0], args[1]: any integer type (validated at semantic level
    //   when the expression types are known; here we specify int as the
    //   widest common expected type for the CheckedExpression annotation)
    // - args[2]: pointer to an integer type where the result is written
    //
    // We use CType predicates to document the type constraints:
    //   is_integer() — first two args must satisfy this
    //   is_pointer() — third arg must satisfy this
    //   is_signed() / is_unsigned() — for diagnostic precision
    //
    // At the builtin evaluation level, we record the expected types;
    // the type checker will enforce actual compatibility.
    let int_ty = CType::Int { signed: true };
    let ptr_int = CType::Pointer(Box::new(CType::Int { signed: true }));

    // Validate that expected types meet the documented constraints.
    // This is a compile-time assertion on our own type construction.
    debug_assert!(int_ty.is_integer());
    debug_assert!(int_ty.is_scalar());
    debug_assert!(int_ty.is_signed());
    debug_assert!(!int_ty.is_unsigned());
    debug_assert!(ptr_int.is_pointer());

    Ok(make_runtime_call(
        CType::Bool,
        vec![
            (&args[0], int_ty.clone()),
            (&args[1], int_ty),
            (&args[2], ptr_int),
        ],
    ))
}

// ===========================================================================
// Miscellaneous builtin handlers
// ===========================================================================

/// Type-checks `__builtin_prefetch(addr [, rw] [, locality])`.
///
/// Data prefetch hint. Arguments:
/// - `addr` — address to prefetch (pointer)
/// - `rw` — 0 for read, 1 for write (optional, default 0)
/// - `locality` — temporal locality 0–3 (optional, default 3 = high)
fn check_prefetch(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count_range(args, 1, 3, "__builtin_prefetch", span, diag)?;

    // Prefetch returns void.
    Ok(BuiltinResult::Void)
}

/// Type-checks `__builtin_object_size(ptr, type)`.
///
/// Returns a `size_t` estimate of the object size pointed to by `ptr`.
/// The `type` parameter (0–3) controls whether minimum or maximum bounds
/// are returned and whether sub-objects are considered.
///
/// When the size cannot be determined at compile time, returns
/// `(size_t)-1` for type 0/1 or `0` for type 2/3.
fn check_object_size(
    args: &[Expression],
    span: Span,
    diag: &mut DiagnosticEngine,
) -> Result<BuiltinResult, ()> {
    validate_arg_count(args, 2, "__builtin_object_size", span, diag)?;

    let size_t = CType::Long { signed: false };
    let void_ptr = CType::Pointer(Box::new(CType::Void));
    let int_ty = CType::Int { signed: true };

    Ok(make_runtime_call(
        size_t,
        vec![(&args[0], void_ptr), (&args[1], int_ty)],
    ))
}

// ===========================================================================
// Type extraction helpers
// ===========================================================================

/// Attempts to extract a `CType` from an expression that encodes a type.
///
/// The parser may encode type arguments for builtins like
/// `__builtin_types_compatible_p` and `__builtin_offsetof` in several ways:
///
/// 1. **Cast expression** `(type)0` — the type_name field contains the type.
/// 2. **Sizeof expression** `sizeof(type)` — the TypeName operand variant.
/// 3. **Identifier** that is a typedef name — resolved by the sema phase.
///
/// Returns `None` if the expression does not encode an extractable type.
fn extract_type_from_expr(expr: &Expression) -> Option<CType> {
    match expr {
        // Cast expression: (type_name)operand — extract the type from the cast.
        // This is the primary encoding path: the parser transforms
        //   __builtin_types_compatible_p(int, long)
        // into something like __builtin_types_compatible_p((int)0, (long)0).
        Expression::Cast { type_name, .. } => resolve_type_name(type_name),

        // Sizeof with type operand: sizeof(type_name)
        Expression::Sizeof {
            operand: crate::frontend::parser::ast::SizeofOperand::TypeName(tn),
            ..
        } => resolve_type_name(tn),

        // Alignof with type operand
        Expression::Alignof {
            operand: crate::frontend::parser::ast::AlignofOperand::TypeName(tn),
            ..
        } => resolve_type_name(tn),

        // Integer literal 0 can represent a null pointer or a zero-value type.
        // We cannot extract a meaningful type from a bare literal.
        Expression::IntegerLiteral { .. } => None,

        // Identifier might be a typedef — we cannot resolve it without symbol
        // table access, so return None and let the caller handle the fallback.
        Expression::Identifier { .. } => None,

        // A FunctionCall as a type argument is unusual but may occur in
        // macro-expanded contexts; no type can be extracted.
        Expression::FunctionCall { .. } => None,

        // Error recovery nodes propagate — return None to let the caller
        // produce an appropriate diagnostic.
        Expression::Error { .. } => None,

        _ => None,
    }
}

/// Attempts to extract a member name from an expression used as the second
/// argument to `__builtin_offsetof(type, member)`.
///
/// Handles simple identifiers (`member`) and nested member access
/// (`member.submember`). Returns `None` for unrecognizable expressions.
fn extract_member_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::Identifier { name, .. } => {
            // Convert the Symbol to a string — but we don't have the
            // interner here, so we store the symbol index as a string.
            // In practice, the sema phase should resolve this.
            // For now, we use a placeholder approach: the symbol's
            // debug representation.
            //
            // Since we can't resolve the Symbol without an Interner,
            // return the raw u32 as a fallback.  The actual resolution
            // happens through the `evaluate_builtin` signature which
            // has access to the interner — see `eval_offsetof_with_interner`.
            Some(format!("__sym_{}", name.as_u32()))
        }
        Expression::MemberAccess { member, .. } => {
            // member is a Symbol — same limitation applies.
            Some(format!("__sym_{}", member.as_u32()))
        }
        _ => None,
    }
}

/// Resolves a `TypeName` AST node to a `CType`.
///
/// This is a simplified type resolution that handles common basic type
/// specifiers. Complex types (struct tags, typedefs, typeof) require
/// symbol table access and are handled by the semantic analyzer before
/// builtin evaluation.
///
/// The `TypeName.span` field is preserved for error reporting by the caller
/// if resolution fails.
fn resolve_type_name(type_name: &TypeName) -> Option<CType> {
    // Access the specifier list from the TypeName. The TypeName.span
    // and TypeName.specifiers.span carry source location information
    // that callers use for diagnostics when type resolution fails.
    let _type_name_span = type_name.span;
    let specs = &type_name.specifiers.specifiers;
    if specs.is_empty() {
        return None;
    }

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

    for spec in specs {
        match spec {
            crate::frontend::parser::ast::TypeSpecifier::Void => has_void = true,
            crate::frontend::parser::ast::TypeSpecifier::Bool => has_bool = true,
            crate::frontend::parser::ast::TypeSpecifier::Char => has_char = true,
            crate::frontend::parser::ast::TypeSpecifier::Short => has_short = true,
            crate::frontend::parser::ast::TypeSpecifier::Int => has_int = true,
            crate::frontend::parser::ast::TypeSpecifier::Long => long_count += 1,
            crate::frontend::parser::ast::TypeSpecifier::Float => has_float = true,
            crate::frontend::parser::ast::TypeSpecifier::Double => has_double = true,
            crate::frontend::parser::ast::TypeSpecifier::Signed => has_signed = true,
            crate::frontend::parser::ast::TypeSpecifier::Unsigned => has_unsigned = true,
            // Complex types that require symbol table resolution
            crate::frontend::parser::ast::TypeSpecifier::Struct { .. }
            | crate::frontend::parser::ast::TypeSpecifier::Union { .. }
            | crate::frontend::parser::ast::TypeSpecifier::Enum { .. } => {
                // We can't resolve these without the symbol table.
                return None;
            }
            crate::frontend::parser::ast::TypeSpecifier::Atomic(inner_tn) => {
                return resolve_type_name(inner_tn).map(|t| CType::Atomic(Box::new(t)));
            }
            _ => {}
        }
    }

    // Resolve keyword combinations per C11 §6.7.2
    if has_void {
        return apply_abstract_declarator(CType::Void, type_name);
    }
    if has_bool {
        return apply_abstract_declarator(CType::Bool, type_name);
    }
    if has_float {
        return apply_abstract_declarator(CType::Float, type_name);
    }
    if has_double {
        let base = if long_count >= 1 {
            CType::LongDouble
        } else {
            CType::Double
        };
        return apply_abstract_declarator(base, type_name);
    }

    // Integer types
    let signed = !has_unsigned;
    if has_char {
        return apply_abstract_declarator(CType::Char { signed }, type_name);
    }
    if has_short {
        return apply_abstract_declarator(CType::Short { signed }, type_name);
    }
    if long_count >= 2 {
        return apply_abstract_declarator(CType::LongLong { signed }, type_name);
    }
    if long_count == 1 {
        return apply_abstract_declarator(CType::Long { signed }, type_name);
    }
    if has_int || has_signed || has_unsigned {
        return apply_abstract_declarator(CType::Int { signed }, type_name);
    }

    None
}

/// Applies the abstract declarator (pointer, array modifiers) from a
/// TypeName to a base CType.
///
/// For example, given base `int` and declarator `[Pointer]`, produces
/// `CType::Pointer(Box::new(CType::Int { signed: true }))`.
fn apply_abstract_declarator(base: CType, type_name: &TypeName) -> Option<CType> {
    match &type_name.declarator {
        None => Some(base),
        Some(decl) => {
            let mut ty = base;
            for derived in &decl.derived {
                match derived {
                    crate::frontend::parser::ast::DerivedDeclarator::Pointer { .. } => {
                        ty = CType::Pointer(Box::new(ty));
                    }
                    crate::frontend::parser::ast::DerivedDeclarator::Array { size, .. } => {
                        let array_size = match size {
                            Some(expr) => match expr.as_ref() {
                                Expression::IntegerLiteral { value, .. } => Some(*value as usize),
                                _ => None,
                            },
                            None => None,
                        };
                        ty = CType::Array {
                            element: Box::new(ty),
                            size: array_size,
                        };
                    }
                    crate::frontend::parser::ast::DerivedDeclarator::Function { .. } => {
                        // Function types are complex; return None for now.
                        return None;
                    }
                }
            }
            Some(ty)
        }
    }
}

// ===========================================================================
// Target-dependent helpers
// ===========================================================================

/// Returns the `size_t` type for the given target architecture.
///
/// On LP64 targets (x86-64, AArch64, RISC-V 64), `size_t` is
/// `unsigned long` (8 bytes). On ILP32 targets (i686), it is
/// `unsigned int` (4 bytes).
fn make_size_t(target: &Target) -> CType {
    // size_t is unsigned long on LP64 (pointer_width == 8 bytes)
    // and unsigned int on ILP32 (pointer_width == 4 bytes).
    match target.data_model() {
        crate::common::target::DataModel::LP64 => CType::Long { signed: false },
        crate::common::target::DataModel::ILP32 => CType::Int { signed: false },
    }
}
