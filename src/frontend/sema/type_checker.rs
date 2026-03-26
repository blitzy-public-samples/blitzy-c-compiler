// Suppress result_unit_err: TypeChecker deliberately returns Result<_, ()>
// because error details flow through DiagnosticEngine, not through the Err variant.
#![allow(clippy::result_unit_err, clippy::if_same_then_else)]

//! Type checking engine for Phase 5 (semantic analysis) of the BCC C11 compiler.
//!
//! Implements C11 type conversion rules:
//! - Integer promotions (§6.3.1.1)
//! - Usual arithmetic conversions (§6.3.1.8)
//! - Type compatibility (§6.2.7)
//! - Implicit conversions
//! - Expression type inference for all AST expression variants
//! - Operator type checking (binary, unary)
//! - Assignment compatibility
//! - Cast validation
//! - Pointer arithmetic validation
//! - Function call argument/parameter matching
//! - Struct/union member access validation
//! - Lvalue checking
//! - Default argument promotions

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::target::Target;
use crate::common::type_builder::{self, TypeBuilder};
use crate::common::types::{self, CType, StructField, TypeQualifiers};
use crate::frontend::parser::ast::*;

/// Empty type qualifiers constant — no const/volatile/restrict/atomic.
const EMPTY_QUALS: TypeQualifiers = TypeQualifiers {
    is_const: false,
    is_volatile: false,
    is_restrict: false,
    is_atomic: false,
};

/// Const-qualified type qualifiers constant.
const CONST_QUALS: TypeQualifiers = TypeQualifiers {
    is_const: true,
    is_volatile: false,
    is_restrict: false,
    is_atomic: false,
};

/// Type checking engine for C11 semantic analysis.
///
/// Performs type inference, implicit conversions, operator type checking,
/// and type compatibility validation. Integrates with the diagnostic engine
/// for error and warning reporting. Uses the target architecture for
/// platform-dependent type sizes and alignments.
pub struct TypeChecker<'a> {
    /// Diagnostic engine for error/warning accumulation
    diagnostics: &'a mut DiagnosticEngine,
    /// Type builder for struct layout computation and sizeof/alignof queries
    type_builder: &'a TypeBuilder,
    /// Target architecture for size/alignment decisions
    target: Target,
}

impl<'a> TypeChecker<'a> {
    // ═══════════════════════════════════════════════════════════════════
    // Constructor
    // ═══════════════════════════════════════════════════════════════════

    /// Creates a new TypeChecker with the given diagnostic engine, type builder,
    /// and target architecture.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
    ) -> Self {
        TypeChecker {
            diagnostics,
            type_builder,
            target,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Private Helper Methods
    // ═══════════════════════════════════════════════════════════════════

    /// Returns the C type representing `size_t` for the current target.
    /// On all Linux targets, `size_t` is `unsigned long`.
    fn size_t_type(&self) -> CType {
        CType::ULong
    }

    /// Returns the C type representing `ptrdiff_t` for the current target.
    /// On all Linux targets, `ptrdiff_t` is `long` (signed).
    fn ptrdiff_t_type(&self) -> CType {
        CType::Long
    }

    /// Strips typedefs and qualifiers from a type to get the underlying base type.
    fn strip_type(ty: &CType) -> CType {
        types::resolve_and_strip(ty).clone()
    }

    /// Performs lvalue-to-rvalue conversion, array-to-pointer decay,
    /// and function-to-pointer decay per C11 §6.3.2.1.
    fn decay_type(ty: &CType) -> CType {
        let resolved = types::resolve_typedef(ty);
        match &resolved {
            // Array decays to pointer to first element
            CType::Array(elem, _) => CType::Pointer(elem.clone(), EMPTY_QUALS),
            // Function decays to pointer to function
            CType::Function { .. } => CType::Pointer(Box::new(resolved.clone()), EMPTY_QUALS),
            // Strip qualifiers for rvalue conversion
            CType::Qualified(inner, _) => Self::decay_type(inner),
            // Other types: just strip top-level qualifiers
            _ => types::unqualified(resolved).clone(),
        }
    }

    /// Extracts qualifiers from a type, handling Qualified wrapper.
    fn get_qualifiers(ty: &CType) -> TypeQualifiers {
        match types::resolve_typedef(ty) {
            CType::Qualified(_, quals) => *quals,
            CType::Pointer(_, quals) => *quals,
            _ => EMPTY_QUALS,
        }
    }

    /// Determines the C type for an integer literal based on value and suffix.
    /// Follows C11 §6.4.4.1 integer literal type determination rules.
    fn integer_literal_type(
        &self,
        value: u128,
        suffix: &IntegerSuffix,
        is_hex_or_octal: bool,
    ) -> CType {
        // C11 §6.4.4.1: hex/octal unsuffixed literals may have unsigned types.
        let long_sz = self.target.long_size();
        match suffix {
            IntegerSuffix::None => {
                if value <= i32::MAX as u128 {
                    CType::Int
                } else if is_hex_or_octal && value <= u32::MAX as u128 {
                    CType::UInt
                } else if long_sz == 8 && value <= i64::MAX as u128 {
                    CType::Long
                } else if is_hex_or_octal && long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else if long_sz != 8 && value <= i32::MAX as u128 {
                    CType::Long
                } else if is_hex_or_octal && long_sz != 8 && value <= u32::MAX as u128 {
                    CType::ULong
                } else if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::U => {
                if value <= u32::MAX as u128 {
                    CType::UInt
                } else if long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::L => {
                if long_sz == 8 {
                    if value <= i64::MAX as u128 {
                        CType::Long
                    } else if is_hex_or_octal && value <= u64::MAX as u128 {
                        CType::ULong
                    } else {
                        CType::LongLong
                    }
                } else if value <= i32::MAX as u128 {
                    CType::Long
                } else if is_hex_or_octal && value <= u32::MAX as u128 {
                    CType::ULong
                } else if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::UL => {
                if long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::LL => {
                if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::ULL => CType::ULongLong,
        }
    }

    /// Determines the C type for a float literal based on suffix.
    /// Imaginary suffixes (i/j) produce the base float type (complex lowering
    /// constructs the full {0.0, imag} pair).
    fn float_literal_type(&self, suffix: &FloatSuffix) -> CType {
        match suffix {
            FloatSuffix::None => CType::Double,
            FloatSuffix::F => CType::Float,
            FloatSuffix::L => CType::LongDouble,
            FloatSuffix::I => CType::Complex(Box::new(CType::Double)),
            FloatSuffix::FI => CType::Complex(Box::new(CType::Float)),
            FloatSuffix::LI => CType::Complex(Box::new(CType::LongDouble)),
        }
    }

    /// Determines the C type for a string literal based on prefix.
    /// Returns a pointer to const-qualified char (or wide char variant).
    fn string_literal_type(&self, prefix: &StringPrefix) -> CType {
        let char_type = match prefix {
            StringPrefix::None | StringPrefix::U8 => CType::Char,
            StringPrefix::L => {
                // wchar_t is typically int on Linux
                CType::Int
            }
            StringPrefix::U16 => CType::UShort,
            StringPrefix::U32 => CType::UInt,
        };
        // String literals are arrays of const char, which decay to pointer to const char
        CType::Pointer(Box::new(char_type), CONST_QUALS)
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.3.1.1 — Integer Promotions
    // ═══════════════════════════════════════════════════════════════════

    /// Performs integer promotion on a type per C11 §6.3.1.1.
    ///
    /// Types with rank less than `int` are promoted to `int` (or `unsigned int`
    /// if `int` cannot represent all values of the original type).
    /// Types `Bool`, `Char`, `SChar`, `UChar`, `Short`, `UShort` are promoted to `int`.
    /// `Int` and higher-ranked integer types are unchanged.
    pub fn integer_promotion(&self, ty: &CType) -> CType {
        types::integer_promotion(ty)
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.3.1.8 — Usual Arithmetic Conversions
    // ═══════════════════════════════════════════════════════════════════

    /// Performs usual arithmetic conversions on two operand types per C11 §6.3.1.8.
    ///
    /// Determines the common type for a binary arithmetic operation:
    /// 1. If either operand is `long double` → result is `long double`
    /// 2. If either operand is `double` → result is `double`
    /// 3. If either operand is `float` → result is `float`
    /// 4. Otherwise, integer promotions applied to both, then rank-based selection:
    ///    - Same signedness: use the higher rank
    ///    - Unsigned rank ≥ signed rank: use unsigned
    ///    - Signed can represent all unsigned values: use signed
    ///    - Otherwise: use unsigned version of the signed type
    pub fn usual_arithmetic_conversion(&self, lhs: &CType, rhs: &CType) -> CType {
        type_builder::usual_arithmetic_conversion(lhs, rhs)
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.2.7 — Type Compatibility
    // ═══════════════════════════════════════════════════════════════════

    /// Checks whether two types are compatible per C11 §6.2.7.
    ///
    /// Compatible types include:
    /// - Same base type after removing qualifiers and resolving typedefs
    /// - Pointers to compatible types
    /// - Arrays with same element type and compatible sizes
    /// - Functions with same return type and compatible parameters
    /// - Struct/union with same tag in same scope
    pub fn are_types_compatible(&self, a: &CType, b: &CType) -> bool {
        types::is_compatible(a, b)
    }

    // ═══════════════════════════════════════════════════════════════════
    // Implicit Conversion Rules
    // ═══════════════════════════════════════════════════════════════════

    /// Checks whether a value of type `from` can be implicitly converted to type `to`.
    ///
    /// Implicit conversions per C11 §6.3:
    /// - All arithmetic types convert to each other
    /// - Any pointer converts to `void *`
    /// - `void *` converts to any pointer type
    /// - Null pointer constant (integer 0) converts to any pointer
    /// - Array decays to pointer to first element
    /// - Function decays to pointer to function
    /// - Enum converts to/from integer types
    pub fn is_implicitly_convertible(&self, from: &CType, to: &CType) -> bool {
        let from_stripped = Self::strip_type(from);
        let to_stripped = Self::strip_type(to);

        // Same type (after stripping) is always convertible
        if types::is_compatible(&from_stripped, &to_stripped) {
            return true;
        }

        // Arithmetic types convert to each other
        if types::is_arithmetic(&from_stripped) && types::is_arithmetic(&to_stripped) {
            return true;
        }

        // Pointer-to-pointer conversions
        if types::is_pointer(&from_stripped) && types::is_pointer(&to_stripped) {
            // Any pointer converts to void*
            if let CType::Pointer(ref inner, _) = to_stripped {
                if matches!(**inner, CType::Void) {
                    return true;
                }
            }
            // void* converts to any pointer
            if let CType::Pointer(ref inner, _) = from_stripped {
                if matches!(**inner, CType::Void) {
                    return true;
                }
            }
            // Pointers to compatible types
            if let (CType::Pointer(ref a_inner, _), CType::Pointer(ref b_inner, _)) =
                (&from_stripped, &to_stripped)
            {
                if types::is_compatible(a_inner, b_inner) {
                    return true;
                }
            }
            // Allow any pointer-to-pointer conversion (with warnings handled by caller)
            return true;
        }

        // Integer to pointer (null pointer constant or with warning)
        if types::is_integer(&from_stripped) && types::is_pointer(&to_stripped) {
            return true;
        }

        // Pointer to integer (with warning)
        if types::is_pointer(&from_stripped) && types::is_integer(&to_stripped) {
            return true;
        }

        // Array decays to pointer to first element
        if let CType::Array(ref elem, _) = from_stripped {
            if let CType::Pointer(ref target_elem, _) = to_stripped {
                return types::is_compatible(elem, target_elem);
            }
        }

        // Function decays to pointer to function
        if types::is_function(&from_stripped) {
            if let CType::Pointer(ref inner, _) = to_stripped {
                return types::is_compatible(&from_stripped, inner);
            }
        }

        // Enum to integer and vice versa
        if matches!(from_stripped, CType::Enum { .. }) && types::is_integer(&to_stripped) {
            return true;
        }
        if types::is_integer(&from_stripped) && matches!(to_stripped, CType::Enum { .. }) {
            return true;
        }

        // Bool converts to/from any scalar
        if matches!(from_stripped, CType::Bool) && types::is_scalar(&to_stripped) {
            return true;
        }
        if types::is_scalar(&from_stripped) && matches!(to_stripped, CType::Bool) {
            return true;
        }

        // Void is not convertible to or from anything (except void)
        if matches!(from_stripped, CType::Void) || matches!(to_stripped, CType::Void) {
            return matches!(from_stripped, CType::Void) && matches!(to_stripped, CType::Void);
        }

        false
    }

    // ═══════════════════════════════════════════════════════════════════
    // Expression Type Inference
    // ═══════════════════════════════════════════════════════════════════

    /// Infers the type of an expression by recursively analyzing sub-expressions.
    ///
    /// Handles all 28+ Expression variants including literals, operators,
    /// function calls, member access, casts, sizeof/alignof, and GCC extensions.
    /// Returns `Err(())` for expression variants that require symbol table access
    /// (e.g., `Identifier`) — the SemanticAnalyzer should handle those.
    pub fn check_expression_type(&mut self, expr: &Expression) -> Result<CType, ()> {
        match expr {
            // ── Literals ────────────────────────────────────────────
            Expression::IntegerLiteral {
                value,
                suffix,
                is_hex_or_octal,
                ..
            } => Ok(self.integer_literal_type(*value, suffix, *is_hex_or_octal)),
            Expression::FloatLiteral { suffix, .. } => Ok(self.float_literal_type(suffix)),
            Expression::StringLiteral { prefix, .. } => Ok(self.string_literal_type(prefix)),
            Expression::CharLiteral { .. } => {
                // C11 §6.4.4.4: character constants have type int
                Ok(CType::Int)
            }

            // ── Identifiers (require symbol table — handled by caller) ──
            Expression::Identifier { span, .. } => {
                self.diagnostics.emit_error(
                    *span,
                    "identifier type cannot be resolved by type checker alone",
                );
                Err(())
            }

            // ── Parenthesized expressions ───────────────────────────
            Expression::Parenthesized { inner, .. } => self.check_expression_type(inner),

            // ── Array subscript: base[index] ────────────────────────
            Expression::ArraySubscript { base, index, span } => {
                let base_ty = self.check_expression_type(base)?;
                let index_ty = self.check_expression_type(index)?;
                self.check_array_subscript(&base_ty, &index_ty, *span)
            }

            // ── Function call: callee(args...) ──────────────────────
            Expression::FunctionCall { callee, args, span } => {
                let callee_ty = self.check_expression_type(callee)?;
                let mut arg_types = Vec::with_capacity(args.len());
                for arg in args.iter() {
                    arg_types.push(self.check_expression_type(arg)?);
                }
                self.check_function_call(&callee_ty, &arg_types, *span)
            }

            // ── Member access: object.member (requires name resolution) ──
            Expression::MemberAccess { object, span, .. } => {
                let obj_ty = self.check_expression_type(object)?;
                let resolved = Self::strip_type(&obj_ty);
                if !types::is_struct_or_union(&resolved) {
                    self.diagnostics
                        .emit_error(*span, "member access on non-struct/union type");
                    return Err(());
                }
                // Cannot resolve member Symbol without interner; caller handles full flow
                Err(())
            }

            // ── Pointer member access: object->member ───────────────
            Expression::PointerMemberAccess { object, span, .. } => {
                let obj_ty = self.check_expression_type(object)?;
                let resolved = Self::strip_type(&obj_ty);
                if let CType::Pointer(ref inner, _) = resolved {
                    let inner_resolved = Self::strip_type(inner);
                    if !types::is_struct_or_union(&inner_resolved) {
                        self.diagnostics.emit_error(
                            *span,
                            "member access through '->' on non-struct/union pointer",
                        );
                        return Err(());
                    }
                } else {
                    self.diagnostics
                        .emit_error(*span, "'->' operator requires pointer to struct/union");
                    return Err(());
                }
                Err(())
            }

            // ── Post-increment/decrement ────────────────────────────
            Expression::PostIncrement { operand, span }
            | Expression::PostDecrement { operand, span } => {
                let op_ty = self.check_expression_type(operand)?;
                let resolved = Self::strip_type(&op_ty);
                if !types::is_arithmetic(&resolved) && !types::is_pointer(&resolved) {
                    self.diagnostics.emit_error(
                        *span,
                        "increment/decrement requires arithmetic or pointer operand",
                    );
                    return Err(());
                }
                Ok(types::unqualified(types::resolve_typedef(&op_ty)).clone())
            }

            // ── Pre-increment/decrement ─────────────────────────────
            Expression::PreIncrement { operand, span }
            | Expression::PreDecrement { operand, span } => {
                let op_ty = self.check_expression_type(operand)?;
                let resolved = Self::strip_type(&op_ty);
                if !types::is_arithmetic(&resolved) && !types::is_pointer(&resolved) {
                    self.diagnostics.emit_error(
                        *span,
                        "increment/decrement requires arithmetic or pointer operand",
                    );
                    return Err(());
                }
                Ok(types::unqualified(types::resolve_typedef(&op_ty)).clone())
            }

            // ── Unary operators ─────────────────────────────────────
            Expression::UnaryOp { op, operand, span } => {
                let op_ty = self.check_expression_type(operand)?;
                self.check_unary_op(op, &op_ty, *span)
            }

            // ── sizeof(expr), sizeof(type), _Alignof(type) ─────────
            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. }
            | Expression::AlignofExpr { .. } => Ok(self.size_t_type()),

            // ── Cast expression: (type)expr (needs TypeName resolution) ──
            Expression::Cast { operand, .. } => {
                let _operand_ty = self.check_expression_type(operand)?;
                // TypeName → CType conversion requires semantic context
                Err(())
            }

            // ── Binary expression: left op right ────────────────────
            Expression::Binary {
                op,
                left,
                right,
                span,
            } => {
                let lhs_ty = self.check_expression_type(left)?;
                let rhs_ty = self.check_expression_type(right)?;
                let lhs_decayed = Self::decay_type(&lhs_ty);
                let rhs_decayed = Self::decay_type(&rhs_ty);
                self.check_binary_op(op, &lhs_decayed, &rhs_decayed, *span)
            }

            // ── Conditional: cond ? then : else ─────────────────────
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                span,
            } => {
                let cond_ty = self.check_expression_type(condition)?;
                let cond_stripped = Self::strip_type(&cond_ty);
                if !types::is_scalar(&cond_stripped) {
                    self.diagnostics.emit_error(
                        *span,
                        "condition of conditional expression must be scalar type",
                    );
                    return Err(());
                }
                let then_ty = if let Some(ref then_e) = then_expr {
                    self.check_expression_type(then_e)?
                } else {
                    // GCC extension: omitted middle operand (cond ?: else)
                    Self::decay_type(&cond_ty)
                };
                let else_ty = self.check_expression_type(else_expr)?;
                self.check_conditional(&then_ty, &else_ty, *span)
            }

            // ── Assignment: target op= value ────────────────────────
            Expression::Assignment {
                target,
                value,
                span,
                op,
            } => {
                let target_ty = self.check_expression_type(target)?;
                let value_ty = self.check_expression_type(value)?;
                match op {
                    AssignOp::Assign => {
                        self.check_assignment(&target_ty, &value_ty, *span)?;
                    }
                    _ => {
                        let bin_op = Self::assign_op_to_binary_op(op);
                        let target_decayed = Self::decay_type(&target_ty);
                        let value_decayed = Self::decay_type(&value_ty);
                        let _result_ty =
                            self.check_binary_op(&bin_op, &target_decayed, &value_decayed, *span)?;
                    }
                }
                Ok(types::unqualified(types::resolve_typedef(&target_ty)).clone())
            }

            // ── Comma expression ────────────────────────────────────
            Expression::Comma { exprs, span } => {
                if exprs.is_empty() {
                    self.diagnostics.emit_error(*span, "empty comma expression");
                    return Err(());
                }
                let mut last_ty = CType::Void;
                for e in exprs.iter() {
                    last_ty = self.check_expression_type(e)?;
                }
                Ok(last_ty)
            }

            // ── Compound literal (needs TypeName resolution) ────────
            Expression::CompoundLiteral { .. } => Err(()),
            // ── Statement expression (needs scope/symbols) ──────────
            Expression::StatementExpression { .. } => Err(()),

            // ── Builtin call: __builtin_xxx(args...) ────────────────
            Expression::BuiltinCall {
                builtin,
                args,
                span,
            } => self.builtin_result_type(builtin, args, *span),

            // ── _Generic selection (needs TypeName resolution) ──────
            Expression::Generic { controlling, .. } => {
                let _ctrl_ty = self.check_expression_type(controlling)?;
                Err(())
            }

            // ── Address-of label: &&label (GCC extension) ───────────
            Expression::AddressOfLabel { .. } => {
                Ok(CType::Pointer(Box::new(CType::Void), EMPTY_QUALS))
            }
        }
    }

    /// Maps an AssignOp to the corresponding BinaryOp for compound assignment.
    fn assign_op_to_binary_op(op: &AssignOp) -> BinaryOp {
        match op {
            AssignOp::Assign => BinaryOp::Add, // unused path
            AssignOp::AddAssign => BinaryOp::Add,
            AssignOp::SubAssign => BinaryOp::Sub,
            AssignOp::MulAssign => BinaryOp::Mul,
            AssignOp::DivAssign => BinaryOp::Div,
            AssignOp::ModAssign => BinaryOp::Mod,
            AssignOp::AndAssign => BinaryOp::BitwiseAnd,
            AssignOp::OrAssign => BinaryOp::BitwiseOr,
            AssignOp::XorAssign => BinaryOp::BitwiseXor,
            AssignOp::ShlAssign => BinaryOp::ShiftLeft,
            AssignOp::ShrAssign => BinaryOp::ShiftRight,
        }
    }

    /// Determines the result type of a GCC builtin call based on the builtin kind.
    fn builtin_result_type(
        &mut self,
        builtin: &BuiltinKind,
        _args: &[Expression],
        _span: Span,
    ) -> Result<CType, ()> {
        match builtin {
            BuiltinKind::Expect => Ok(CType::Long),
            BuiltinKind::Unreachable | BuiltinKind::Trap => Ok(CType::Void),
            BuiltinKind::ConstantP | BuiltinKind::TypesCompatibleP => Ok(CType::Int),
            BuiltinKind::Offsetof => Ok(self.size_t_type()),
            BuiltinKind::ChooseExpr => Err(()),
            BuiltinKind::Clz
            | BuiltinKind::ClzL
            | BuiltinKind::ClzLL
            | BuiltinKind::Ctz
            | BuiltinKind::CtzL
            | BuiltinKind::CtzLL
            | BuiltinKind::Popcount
            | BuiltinKind::PopcountL
            | BuiltinKind::PopcountLL
            | BuiltinKind::Ffs
            | BuiltinKind::Ffsll => Ok(CType::Int),
            BuiltinKind::Bswap16 => Ok(CType::UShort),
            BuiltinKind::Bswap32 => Ok(CType::UInt),
            BuiltinKind::Bswap64 => Ok(CType::ULongLong),
            BuiltinKind::VaStart | BuiltinKind::VaEnd | BuiltinKind::VaCopy => Ok(CType::Void),
            BuiltinKind::VaArg => Err(()),
            BuiltinKind::FrameAddress | BuiltinKind::ReturnAddress => {
                Ok(CType::Pointer(Box::new(CType::Void), EMPTY_QUALS))
            }
            BuiltinKind::AssumeAligned => Ok(CType::Pointer(Box::new(CType::Void), EMPTY_QUALS)),
            BuiltinKind::AddOverflow | BuiltinKind::SubOverflow | BuiltinKind::MulOverflow => {
                Ok(CType::Bool)
            }
            BuiltinKind::PrefetchData => Ok(CType::Void),
            BuiltinKind::ObjectSize => Ok(self.size_t_type()),
            BuiltinKind::ExtractReturnAddr => {
                Ok(CType::Pointer(Box::new(CType::Void), EMPTY_QUALS))
            }
            // All other builtins — use BuiltinEvaluator for comprehensive return type.
            other => {
                let eval = crate::frontend::sema::builtin_eval::BuiltinEvaluator::new(
                    self.diagnostics,
                    self.type_builder,
                    self.target,
                );
                Ok(eval.get_builtin_return_type(other))
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5 — Binary Operator Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Checks binary operator type rules and returns the result type.
    ///
    /// Handles arithmetic operators, pointer arithmetic, comparison operators,
    /// logical operators, bitwise operators, and shift operators.
    pub fn check_binary_op(
        &mut self,
        op: &BinaryOp,
        lhs_ty: &CType,
        rhs_ty: &CType,
        span: Span,
    ) -> Result<CType, ()> {
        let lhs = Self::strip_type(lhs_ty);
        let rhs = Self::strip_type(rhs_ty);

        match op {
            // ── Arithmetic: +, -, *, /, % ───────────────────────────
            BinaryOp::Add => {
                // pointer + integer
                if types::is_pointer(&lhs) && types::is_integer(&rhs) {
                    return self.check_pointer_arithmetic(lhs_ty, rhs_ty, op, span);
                }
                // integer + pointer
                if types::is_integer(&lhs) && types::is_pointer(&rhs) {
                    return self.check_pointer_arithmetic(rhs_ty, lhs_ty, op, span);
                }
                // arithmetic + arithmetic
                if types::is_arithmetic(&lhs) && types::is_arithmetic(&rhs) {
                    return Ok(self.usual_arithmetic_conversion(lhs_ty, rhs_ty));
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to binary '+'");
                Err(())
            }

            BinaryOp::Sub => {
                // pointer - integer
                if types::is_pointer(&lhs) && types::is_integer(&rhs) {
                    return self.check_pointer_arithmetic(lhs_ty, rhs_ty, op, span);
                }
                // pointer - pointer → ptrdiff_t
                if types::is_pointer(&lhs) && types::is_pointer(&rhs) {
                    // Both must point to compatible types
                    if let (CType::Pointer(ref a, _), CType::Pointer(ref b, _)) = (&lhs, &rhs) {
                        let a_stripped = Self::strip_type(a);
                        let b_stripped = Self::strip_type(b);
                        if !types::is_compatible(&a_stripped, &b_stripped)
                            && !matches!(a_stripped, CType::Void)
                            && !matches!(b_stripped, CType::Void)
                        {
                            self.diagnostics.emit_warning(
                                span,
                                "subtraction of pointers to incompatible types",
                            );
                        }
                    }
                    return Ok(self.ptrdiff_t_type());
                }
                // arithmetic - arithmetic
                if types::is_arithmetic(&lhs) && types::is_arithmetic(&rhs) {
                    return Ok(self.usual_arithmetic_conversion(lhs_ty, rhs_ty));
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to binary '-'");
                Err(())
            }

            BinaryOp::Mul | BinaryOp::Div => {
                if types::is_arithmetic(&lhs) && types::is_arithmetic(&rhs) {
                    return Ok(self.usual_arithmetic_conversion(lhs_ty, rhs_ty));
                }
                self.diagnostics.emit_error(
                    span,
                    format!(
                        "invalid operands to binary '{}'",
                        if matches!(op, BinaryOp::Mul) {
                            "*"
                        } else {
                            "/"
                        }
                    ),
                );
                Err(())
            }

            BinaryOp::Mod => {
                if types::is_integer(&lhs) && types::is_integer(&rhs) {
                    return Ok(self.usual_arithmetic_conversion(lhs_ty, rhs_ty));
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to binary '%'");
                Err(())
            }

            // ── Shift: <<, >> ───────────────────────────────────────
            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
                if types::is_integer(&lhs) && types::is_integer(&rhs) {
                    // Result type is the promoted type of the left operand
                    return Ok(self.integer_promotion(lhs_ty));
                }
                self.diagnostics.emit_error(
                    span,
                    format!(
                        "invalid operands to binary '{}'",
                        if matches!(op, BinaryOp::ShiftLeft) {
                            "<<"
                        } else {
                            ">>"
                        }
                    ),
                );
                Err(())
            }

            // ── Comparison: ==, !=, <, >, <=, >= ────────────────────
            BinaryOp::Equal | BinaryOp::NotEqual => {
                // arithmetic == arithmetic
                if types::is_arithmetic(&lhs) && types::is_arithmetic(&rhs) {
                    return Ok(CType::Int);
                }
                // pointer == pointer
                if types::is_pointer(&lhs) && types::is_pointer(&rhs) {
                    return Ok(CType::Int);
                }
                // pointer == integer (null pointer check)
                if (types::is_pointer(&lhs) && types::is_integer(&rhs))
                    || (types::is_integer(&lhs) && types::is_pointer(&rhs))
                {
                    self.diagnostics
                        .emit_warning(span, "comparison between pointer and integer");
                    return Ok(CType::Int);
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to comparison operator");
                Err(())
            }

            BinaryOp::Less | BinaryOp::Greater | BinaryOp::LessEqual | BinaryOp::GreaterEqual => {
                // arithmetic < arithmetic
                if types::is_arithmetic(&lhs) && types::is_arithmetic(&rhs) {
                    return Ok(CType::Int);
                }
                // pointer < pointer (must be compatible types)
                if types::is_pointer(&lhs) && types::is_pointer(&rhs) {
                    return Ok(CType::Int);
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to relational operator");
                Err(())
            }

            // ── Bitwise: &, |, ^ ───────────────────────────────────
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                if types::is_integer(&lhs) && types::is_integer(&rhs) {
                    return Ok(self.usual_arithmetic_conversion(lhs_ty, rhs_ty));
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to bitwise operator");
                Err(())
            }

            // ── Logical: &&, || ─────────────────────────────────────
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                if types::is_scalar(&lhs) && types::is_scalar(&rhs) {
                    return Ok(CType::Int);
                }
                self.diagnostics
                    .emit_error(span, "invalid operands to logical operator");
                Err(())
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.3 — Unary Operator Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Checks unary operator type rules and returns the result type.
    ///
    /// Handles address-of (&), dereference (*), unary plus (+), negate (-),
    /// bitwise NOT (~), and logical NOT (!).
    pub fn check_unary_op(
        &mut self,
        op: &UnaryOp,
        operand_ty: &CType,
        span: Span,
    ) -> Result<CType, ()> {
        let resolved = Self::strip_type(operand_ty);

        match op {
            // ── Address-of: &operand → pointer to operand type ──────
            UnaryOp::AddressOf => Ok(CType::Pointer(
                Box::new(types::resolve_typedef(operand_ty).clone()),
                EMPTY_QUALS,
            )),

            // ── Dereference: *operand → pointee type ────────────────
            UnaryOp::Deref => {
                if let CType::Pointer(ref inner, _) = resolved {
                    Ok((**inner).clone())
                } else {
                    self.diagnostics
                        .emit_error(span, "indirection requires pointer operand");
                    Err(())
                }
            }

            // ── Unary plus: +operand (arithmetic, integer promotion) ─
            UnaryOp::Plus => {
                if types::is_arithmetic(&resolved) {
                    Ok(self.integer_promotion(operand_ty))
                } else {
                    self.diagnostics
                        .emit_error(span, "unary '+' requires arithmetic operand");
                    Err(())
                }
            }

            // ── Unary negate: -operand (arithmetic, integer promotion) ─
            UnaryOp::Negate => {
                if types::is_arithmetic(&resolved) {
                    Ok(self.integer_promotion(operand_ty))
                } else {
                    self.diagnostics
                        .emit_error(span, "unary '-' requires arithmetic operand");
                    Err(())
                }
            }

            // ── Bitwise NOT: ~operand (integer, integer promotion) ──
            UnaryOp::BitwiseNot => {
                if types::is_integer(&resolved) {
                    Ok(self.integer_promotion(operand_ty))
                } else {
                    self.diagnostics
                        .emit_error(span, "bitwise '~' requires integer operand");
                    Err(())
                }
            }

            // ── Logical NOT: !operand (scalar → int) ────────────────
            UnaryOp::LogicalNot => {
                if types::is_scalar(&resolved) {
                    Ok(CType::Int)
                } else {
                    self.diagnostics
                        .emit_error(span, "logical '!' requires scalar operand");
                    Err(())
                }
            }

            // ── GCC __real__/__imag__: extract component of _Complex ──
            UnaryOp::RealPart | UnaryOp::ImagPart => {
                if let CType::Complex(inner) = &resolved {
                    Ok((**inner).clone())
                } else {
                    // For non-complex types, __real__ returns the type itself,
                    // __imag__ returns 0 of the same type.
                    Ok(operand_ty.clone())
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.2.2 — Function Call Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Validates a function call and returns the function's return type.
    ///
    /// Checks that the callee is a function type (or pointer to function),
    /// validates argument count against parameter count (respecting variadic),
    /// and checks assignment compatibility for each argument/parameter pair.
    /// Applies default argument promotions for variadic arguments.
    pub fn check_function_call(
        &mut self,
        callee_type: &CType,
        args: &[CType],
        span: Span,
    ) -> Result<CType, ()> {
        let resolved = Self::strip_type(callee_type);

        // Dereference pointer to function
        let func_type = if let CType::Pointer(ref inner, _) = resolved {
            Self::strip_type(inner)
        } else {
            resolved.clone()
        };

        match &func_type {
            CType::Function {
                return_type,
                params,
                variadic,
            } => {
                // Check argument count
                if !variadic && args.len() != params.len() {
                    self.diagnostics.emit(Diagnostic::error(
                        span,
                        format!("expected {} arguments, got {}", params.len(), args.len()),
                    ));
                    return Err(());
                }
                if *variadic && args.len() < params.len() {
                    self.diagnostics.emit(Diagnostic::error(
                        span,
                        format!(
                            "too few arguments: expected at least {}, got {}",
                            params.len(),
                            args.len()
                        ),
                    ));
                    return Err(());
                }

                // Check each argument against its parameter type
                for (i, (arg_ty, param_ty)) in args.iter().zip(params.iter()).enumerate() {
                    let arg_decayed = Self::decay_type(arg_ty);
                    if !self.is_implicitly_convertible(&arg_decayed, param_ty) {
                        self.diagnostics.emit(Diagnostic::warning(
                            span,
                            format!("incompatible type for argument {}", i + 1),
                        ));
                    }
                }

                // For variadic arguments, apply default promotions
                for arg_ty in args.iter().skip(params.len()) {
                    let _promoted = self.default_argument_promotion(arg_ty);
                }

                Ok((**return_type).clone())
            }
            _ => {
                self.diagnostics
                    .emit_error(span, "called object is not a function or function pointer");
                Err(())
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.2.3 — Struct/Union Member Access Validation
    // ═══════════════════════════════════════════════════════════════════

    /// Validates struct/union member access and returns the member's type.
    ///
    /// For dot access (`.`), the operand must be a struct/union.
    /// For arrow access (`->`), the operand must be a pointer to struct/union.
    /// Looks up the member name in the struct/union field list.
    /// Handles anonymous struct/union members by searching nested fields.
    ///
    /// The `member_name` parameter is a `&str` because `StructField.name` is
    /// `Option<String>`. The caller (SemanticAnalyzer) resolves the `Symbol`
    /// to a string via the `Interner` before calling this method.
    pub fn check_member_access(
        &mut self,
        struct_type: &CType,
        member_name: &str,
        is_arrow: bool,
        span: Span,
    ) -> Result<CType, ()> {
        let resolved = Self::strip_type(struct_type);

        // For arrow access, dereference the pointer first
        let actual_type = if is_arrow {
            if let CType::Pointer(ref inner, _) = resolved {
                Self::strip_type(inner)
            } else {
                self.diagnostics
                    .emit_error(span, "member reference through '->' requires pointer type");
                return Err(());
            }
        } else {
            resolved
        };

        // Must be a struct or union
        let fields = match &actual_type {
            CType::Struct { fields, .. } => fields,
            CType::Union { fields, .. } => fields,
            _ => {
                let op_str = if is_arrow { "->" } else { "." };
                self.diagnostics.emit_error(
                    span,
                    format!(
                        "member reference base type is not a struct or union (operator '{}')",
                        op_str
                    ),
                );
                return Err(());
            }
        };

        // Search for the member in the field list
        if let Some(field_ty) = Self::find_member_in_fields(fields, member_name) {
            return Ok(field_ty);
        }

        self.diagnostics.emit_error(
            span,
            format!("no member named '{}' in struct/union", member_name),
        );
        Err(())
    }

    /// Recursively searches for a member name in struct/union fields,
    /// including anonymous struct/union members.
    /// Returns the field's type. For bitfield members (`bit_width.is_some()`),
    /// the type is the declared integer type — the bit width is used during
    /// IR lowering for load/store mask generation.
    fn find_member_in_fields(fields: &[StructField], member_name: &str) -> Option<CType> {
        for field in fields {
            if let Some(ref name) = field.name {
                if name == member_name {
                    // C11 §6.3.1.1p1: If an int can represent all values of the original
                    // type, the value is converted to an int; otherwise, it is converted to
                    // an unsigned int.  For a bitfield of type `unsigned int : N` where
                    // N < 32, all values (0 .. 2^N-1) fit in a signed int (2^31-1 max),
                    // so the expression type after integer promotion is `int`, NOT
                    // `unsigned int`.  We apply this adjustment at the point of member
                    // access so that downstream arithmetic uses the correct signedness.
                    if let Some(width) = field.bit_width {
                        let base = crate::common::types::resolve_and_strip(&field.ty);
                        match base {
                            CType::UInt if width < 32 => return Some(CType::Int),
                            CType::ULong | CType::ULongLong if width < 64 => {
                                // For ULong/ULongLong bitfields narrower than 64 bits
                                // that fit in int, promote to int; otherwise keep original.
                                if width < 32 {
                                    return Some(CType::Int);
                                }
                                return Some(field.ty.clone());
                            }
                            _ => return Some(field.ty.clone()),
                        }
                    }
                    return Some(field.ty.clone());
                }
            } else {
                // Anonymous struct/union member — search nested fields
                let inner_type = Self::strip_type(&field.ty);
                match &inner_type {
                    CType::Struct {
                        fields: inner_fields,
                        ..
                    }
                    | CType::Union {
                        fields: inner_fields,
                        ..
                    } => {
                        if let Some(ty) = Self::find_member_in_fields(inner_fields, member_name) {
                            return Some(ty);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.16 — Assignment Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Validates that a value can be assigned to a target type.
    ///
    /// Checks:
    /// - Target must be modifiable (not const-qualified, not array, not incomplete)
    /// - Value must be implicitly convertible to target type
    /// - Warns on implicit pointer-integer conversions
    /// - Warns on incompatible pointer type assignments
    /// - Errors on struct/union assignment with incompatible types
    pub fn check_assignment(
        &mut self,
        target_type: &CType,
        value_type: &CType,
        span: Span,
    ) -> Result<(), ()> {
        let target_resolved = types::resolve_typedef(target_type);
        let target_stripped = Self::strip_type(target_type);
        let value_stripped = Self::strip_type(value_type);

        // Check for const-qualified target (not modifiable)
        let quals = Self::get_qualifiers(target_resolved);
        if quals.is_const {
            self.diagnostics
                .emit_error(span, "assignment to const-qualified variable");
            return Err(());
        }

        // Arrays are not assignable
        if types::is_array(&target_stripped) {
            self.diagnostics
                .emit_error(span, "assignment to expression with array type");
            return Err(());
        }

        // Void is not assignable
        if matches!(target_stripped, CType::Void) {
            self.diagnostics.emit_error(span, "assignment to void type");
            return Err(());
        }

        // Check type compatibility
        if self.is_implicitly_convertible(&value_stripped, &target_stripped) {
            // Warn on pointer-to-integer or integer-to-pointer implicit conversion
            if types::is_pointer(&target_stripped) && types::is_integer(&value_stripped) {
                self.diagnostics
                    .emit_warning(span, "implicit conversion from integer to pointer type");
            } else if types::is_integer(&target_stripped) && types::is_pointer(&value_stripped) {
                self.diagnostics
                    .emit_warning(span, "implicit conversion from pointer to integer type");
            }

            // Warn on incompatible pointer types
            if types::is_pointer(&target_stripped) && types::is_pointer(&value_stripped) {
                if let (CType::Pointer(ref t_inner, _), CType::Pointer(ref v_inner, _)) =
                    (&target_stripped, &value_stripped)
                {
                    let t_base = Self::strip_type(t_inner);
                    let v_base = Self::strip_type(v_inner);
                    if !types::is_compatible(&t_base, &v_base)
                        && !matches!(t_base, CType::Void)
                        && !matches!(v_base, CType::Void)
                    {
                        self.diagnostics
                            .emit_warning(span, "assignment from incompatible pointer type");
                    }
                }
            }

            return Ok(());
        }

        // Struct/union assignment requires compatible types
        if types::is_struct_or_union(&target_stripped) && types::is_struct_or_union(&value_stripped)
        {
            if !types::is_compatible(&target_stripped, &value_stripped) {
                self.diagnostics
                    .emit_error(span, "assignment between incompatible struct/union types");
                return Err(());
            }
            return Ok(());
        }

        self.diagnostics
            .emit_error(span, "incompatible types in assignment");
        Err(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.3.2.1 — Lvalue Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Determines whether an expression is an lvalue (designates an object).
    ///
    /// Lvalues include:
    /// - Identifiers (variables, not functions/arrays after decay)
    /// - Dereference expressions (`*ptr`)
    /// - Array subscript expressions (`a[i]`)
    /// - Member access expressions (`s.field`, `p->field`)
    /// - Compound literals (C11 §6.5.2.5)
    /// - String literals (they designate an array object)
    ///
    /// Non-lvalues:
    /// - Function call results, casts, arithmetic results, literals (non-string)
    pub fn is_lvalue(expr: &Expression) -> bool {
        match expr {
            // Identifiers are lvalues (the SemanticAnalyzer filters functions/arrays)
            Expression::Identifier { .. } => true,
            // Dereference of a pointer is an lvalue
            Expression::UnaryOp {
                op: UnaryOp::Deref, ..
            } => true,
            // Array subscript is an lvalue (equivalent to *(base + index))
            Expression::ArraySubscript { .. } => true,
            // Member access is an lvalue if the object is an lvalue
            Expression::MemberAccess { .. } => true,
            Expression::PointerMemberAccess { .. } => true,
            // Compound literals are lvalues in C11
            Expression::CompoundLiteral { .. } => true,
            // String literals designate array objects
            Expression::StringLiteral { .. } => true,
            // Parenthesized lvalue is still an lvalue
            Expression::Parenthesized { inner, .. } => Self::is_lvalue(inner),
            // Pre-increment/decrement results are lvalues
            Expression::PreIncrement { .. } | Expression::PreDecrement { .. } => true,
            // GCC __real__/__imag__ are lvalues if their operand is an lvalue
            Expression::UnaryOp {
                op: UnaryOp::RealPart | UnaryOp::ImagPart,
                operand,
                ..
            } => Self::is_lvalue(operand),
            // Everything else is not an lvalue
            _ => false,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.6 — Pointer Arithmetic Validation
    // ═══════════════════════════════════════════════════════════════════

    /// Validates pointer arithmetic and returns the result type.
    ///
    /// Only addition and subtraction are valid with pointers.
    /// The pointer must point to a complete object type (not void*, not function*).
    /// Result type: same pointer type for ptr+int/ptr-int, `ptrdiff_t` for ptr-ptr.
    /// GCC extension: pointer-to-void arithmetic treats void as char (sizeof 1).
    pub fn check_pointer_arithmetic(
        &mut self,
        ptr_ty: &CType,
        int_ty: &CType,
        op: &BinaryOp,
        span: Span,
    ) -> Result<CType, ()> {
        // Only + and - are valid for pointer arithmetic
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            self.diagnostics
                .emit_error(span, "invalid pointer arithmetic operation");
            return Err(());
        }

        let ptr_resolved = Self::strip_type(ptr_ty);

        // Verify the pointer target type
        if let CType::Pointer(ref pointee, quals) = ptr_resolved {
            let pointee_stripped = Self::strip_type(pointee);

            // Warn on void pointer arithmetic (GCC extension: sizeof(void) = 1)
            if matches!(pointee_stripped, CType::Void) {
                self.diagnostics.emit_warning(
                    span,
                    "pointer arithmetic on void pointer (GCC extension: sizeof(void) = 1)",
                );
                return Ok(ptr_resolved.clone());
            }

            // Function pointer arithmetic is not allowed
            if types::is_function(&pointee_stripped) {
                self.diagnostics.emit_error(
                    span,
                    "pointer arithmetic on function pointer is not allowed",
                );
                return Err(());
            }

            // Pointer must point to complete type
            if !types::is_complete(&pointee_stripped) {
                self.diagnostics
                    .emit_error(span, "arithmetic on pointer to incomplete type");
                return Err(());
            }

            // Verify the pointee has a known nonzero size for stride computation
            let pointee_size = types::sizeof_ctype(&pointee_stripped, &self.target);
            let _pointee_align = types::alignof_ctype(&pointee_stripped, &self.target);
            if pointee_size == 0 {
                self.diagnostics
                    .emit_warning(span, "pointer arithmetic on zero-sized type");
            }

            // The integer operand must be an integer type
            let int_resolved = Self::strip_type(int_ty);
            if !types::is_integer(&int_resolved) {
                self.diagnostics
                    .emit_error(span, "pointer arithmetic requires integer operand");
                return Err(());
            }

            // Result is the same pointer type
            Ok(CType::Pointer(pointee.clone(), quals))
        } else {
            self.diagnostics
                .emit_error(span, "expected pointer type in pointer arithmetic");
            Err(())
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.2.1 — Array Subscript Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Validates array subscript `base[index]` and returns the element type.
    ///
    /// The base must be a pointer or array type (arrays decay to pointers),
    /// and the index must be an integer type. Alternatively, `index[base]`
    /// is also valid (C11 §6.5.2.1: a[b] is equivalent to *(a+b)).
    pub fn check_array_subscript(
        &mut self,
        base_ty: &CType,
        index_ty: &CType,
        span: Span,
    ) -> Result<CType, ()> {
        let base_stripped = Self::strip_type(base_ty);
        let index_stripped = Self::strip_type(index_ty);

        // Standard case: base is pointer/array, index is integer
        if let Some(element_ty) = Self::extract_element_type(&base_stripped) {
            if types::is_integer(&index_stripped) {
                return Ok(element_ty);
            }
        }

        // Reversed case: index is pointer/array, base is integer (a[b] == b[a])
        if let Some(element_ty) = Self::extract_element_type(&index_stripped) {
            if types::is_integer(&base_stripped) {
                return Ok(element_ty);
            }
        }

        self.diagnostics.emit_error(
            span,
            "subscripted value is not an array or pointer, or index is not an integer",
        );
        Err(())
    }

    /// Extracts the element type from an array or pointer type.
    fn extract_element_type(ty: &CType) -> Option<CType> {
        match ty {
            CType::Array(elem, _) => Some((**elem).clone()),
            CType::Pointer(pointee, _) => Some((**pointee).clone()),
            _ => None,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.15 — Conditional Expression Type Checking
    // ═══════════════════════════════════════════════════════════════════

    /// Determines the result type of a conditional expression `cond ? a : b`.
    ///
    /// Rules per C11 §6.5.15:
    /// 1. Both arithmetic → usual arithmetic conversions
    /// 2. Both the same struct/union → that type
    /// 3. Both void → void
    /// 4. One pointer, one null constant → pointer type
    /// 5. Both pointers to compatible types → composite pointer type
    /// 6. One void*, one other pointer → void* (with merged qualifiers)
    pub fn check_conditional(
        &mut self,
        then_ty: &CType,
        else_ty: &CType,
        span: Span,
    ) -> Result<CType, ()> {
        let then_stripped = Self::strip_type(then_ty);
        let else_stripped = Self::strip_type(else_ty);

        // Both void
        if matches!(then_stripped, CType::Void) && matches!(else_stripped, CType::Void) {
            return Ok(CType::Void);
        }

        // Both arithmetic
        if types::is_arithmetic(&then_stripped) && types::is_arithmetic(&else_stripped) {
            return Ok(self.usual_arithmetic_conversion(then_ty, else_ty));
        }

        // Both compatible struct/union
        if types::is_struct_or_union(&then_stripped) && types::is_struct_or_union(&else_stripped) {
            if types::is_compatible(&then_stripped, &else_stripped) {
                return Ok(then_stripped);
            }
            self.diagnostics.emit_error(
                span,
                "incompatible struct/union types in conditional expression",
            );
            return Err(());
        }

        // Both pointers
        if types::is_pointer(&then_stripped) && types::is_pointer(&else_stripped) {
            // One is void* → result is void* with merged qualifiers
            if let (CType::Pointer(ref a, ref aq), CType::Pointer(ref b, ref bq)) =
                (&then_stripped, &else_stripped)
            {
                if matches!(**a, CType::Void) || matches!(**b, CType::Void) {
                    let merged = aq.merge(*bq);
                    return Ok(CType::Pointer(Box::new(CType::Void), merged));
                }
                // Both point to compatible types
                if types::is_compatible(a, b) {
                    return Ok(then_stripped);
                }
            }
            // Incompatible pointers — warn but allow
            self.diagnostics
                .emit_warning(span, "pointer type mismatch in conditional expression");
            return Ok(then_stripped);
        }

        // One pointer, one null constant (integer zero)
        if types::is_pointer(&then_stripped) && types::is_integer(&else_stripped) {
            return Ok(then_stripped);
        }
        if types::is_integer(&then_stripped) && types::is_pointer(&else_stripped) {
            return Ok(else_stripped);
        }

        self.diagnostics
            .emit_error(span, "incompatible types in conditional expression");
        Err(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.4 — Cast Validation
    // ═══════════════════════════════════════════════════════════════════

    /// Validates a cast from one type to another.
    ///
    /// Allowed casts:
    /// - Arithmetic → arithmetic: always allowed
    /// - Pointer → integer: allowed with warning
    /// - Integer → pointer: allowed with warning
    /// - Pointer → pointer: allowed (may warn for object type incompatibility)
    /// - Void → void: allowed
    ///
    /// Disallowed:
    /// - Struct/union casts (not allowed in C)
    /// - Void → non-void (except void → void)
    /// - Function pointer casts: allowed with warning
    pub fn check_cast(&mut self, from: &CType, to: &CType, span: Span) -> Result<(), ()> {
        let from_stripped = Self::strip_type(from);
        let to_stripped = Self::strip_type(to);

        // void → void: allowed
        if matches!(from_stripped, CType::Void) && matches!(to_stripped, CType::Void) {
            return Ok(());
        }

        // Casting anything to void: allowed (discards value)
        if matches!(to_stripped, CType::Void) {
            return Ok(());
        }

        // Cannot cast FROM void (except to void, handled above)
        if matches!(from_stripped, CType::Void) {
            self.diagnostics
                .emit_error(span, "cannot cast from void type");
            return Err(());
        }

        // Arithmetic → arithmetic: always allowed
        if types::is_arithmetic(&from_stripped) && types::is_arithmetic(&to_stripped) {
            return Ok(());
        }

        // Pointer → integer: allowed with warning
        if types::is_pointer(&from_stripped) && types::is_integer(&to_stripped) {
            let ptr_size = self.target.pointer_width();
            let int_size = self.type_builder.sizeof_type(&to_stripped);
            if int_size < ptr_size {
                self.diagnostics.emit_warning(
                    span,
                    "cast from pointer to smaller integer type loses information",
                );
            }
            return Ok(());
        }

        // Integer → pointer: allowed with warning
        if types::is_integer(&from_stripped) && types::is_pointer(&to_stripped) {
            self.diagnostics
                .emit_warning(span, "cast to pointer from integer of different size");
            return Ok(());
        }

        // Pointer → pointer: allowed
        if types::is_pointer(&from_stripped) && types::is_pointer(&to_stripped) {
            return Ok(());
        }

        // Struct/union casts: not allowed
        if types::is_struct_or_union(&from_stripped) || types::is_struct_or_union(&to_stripped) {
            self.diagnostics
                .emit_error(span, "invalid cast involving struct or union type");
            return Err(());
        }

        // Array and function types (should have decayed already, but handle gracefully)
        if types::is_array(&from_stripped) || types::is_function(&from_stripped) {
            // Array/function should have decayed to pointer
            self.diagnostics
                .emit_warning(span, "unusual cast from array/function type");
            return Ok(());
        }

        self.diagnostics.emit_error(span, "invalid cast");
        Err(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // §6.5.2.2p6 — Default Argument Promotions
    // ═══════════════════════════════════════════════════════════════════

    /// Applies default argument promotions for variadic function arguments
    /// and K&R-style function calls per C11 §6.5.2.2p6.
    ///
    /// - Integer promotions are applied to integer types
    /// - `float` is promoted to `double`
    /// - Arrays decay to pointers to first element
    /// - Functions decay to pointers to function
    pub fn default_argument_promotion(&self, ty: &CType) -> CType {
        let resolved = types::resolve_typedef(ty);
        let stripped = Self::strip_type(resolved);

        // Array → pointer to first element
        if let CType::Array(ref elem, _) = stripped {
            return CType::Pointer(elem.clone(), EMPTY_QUALS);
        }

        // Function → pointer to function
        if types::is_function(&stripped) {
            return CType::Pointer(Box::new(resolved.clone()), EMPTY_QUALS);
        }

        // Float → double
        if matches!(stripped, CType::Float) {
            return CType::Double;
        }

        // Integer promotions for small integer types
        if types::is_integer(&stripped) {
            return self.integer_promotion(ty);
        }

        // All other types pass through unchanged
        types::unqualified(resolved).clone()
    }
}
