//! Compile-time constant expression evaluation for the BCC C11 compiler.
//!
//! Implements C11 §6.6 integer constant expression semantics for Phase 5
//! (semantic analysis). Evaluates constant expressions required for:
//!
//! - **Array sizes:** `int arr[N]` — `N` must be a positive integer constant expression
//! - **Case labels:** `case V:` — `V` must be an integer constant expression
//! - **Enum values:** `enum { A = V }` — `V` must be an integer constant expression
//! - **Bitfield widths:** `int x : W` — `W` must be a non-negative integer constant expression
//! - **`_Static_assert`:** `_Static_assert(C, "msg")` — `C` must be an integer constant expression
//! - **`sizeof` / `_Alignof`:** always constant expressions (except VLAs)
//!
//! # Supported Operations
//!
//! All C11 integer arithmetic, bitwise, shift, comparison, and logical operators
//! are evaluated at compile time. The ternary `? :` and comma operators are also
//! supported. `sizeof` and `_Alignof` produce architecture-dependent constants
//! parameterised by [`Target`].
//!
//! # Overflow Behaviour
//!
//! - **Signed:** implementation-defined (wrapping `i128` arithmetic, matching GCC)
//! - **Unsigned:** wrapping per C11 §6.2.5 ¶9
//!
//! # Zero-Dependency Compliance
//!
//! This module depends only on the Rust standard library and internal crate
//! modules (`common`, `frontend::parser::ast`). It does NOT depend on `ir`,
//! `passes`, or `backend`.

use crate::common::fx_hash::FxHashMap;

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Severity, Span};
use crate::common::string_interner::Symbol;
use crate::common::target::Target;
use crate::common::types::{
    alignof_ctype_resolved, is_integer, is_unsigned, sizeof_ctype, sizeof_ctype_resolved, CType,
};
use crate::frontend::parser::ast::*;

// ===========================================================================
// ConstValue — Result of Constant Expression Evaluation
// ===========================================================================

/// Represents the result of evaluating a compile-time constant expression.
///
/// The variants cover all possible compile-time constant values in C11:
/// signed/unsigned integers, floating-point values, address constants
/// (for global symbol references with offsets), and string literals
/// (for initialiser contexts).
///
/// Integer constant expressions (C11 §6.6p6) always resolve to
/// [`SignedInt`](ConstValue::SignedInt) or [`UnsignedInt`](ConstValue::UnsignedInt).
#[derive(Debug, Clone)]
pub enum ConstValue {
    /// Signed integer value — covers all signed C integer types up to
    /// `long long` (and `__int128` when needed). Uses `i128` for
    /// full-range coverage without overflow during intermediate computation.
    SignedInt(i128),

    /// Unsigned integer value — covers all unsigned C integer types.
    /// Uses `u128` for full-range coverage.
    UnsignedInt(u128),

    /// Floating-point value — used for non-integer constant expressions
    /// where applicable (e.g., `3.14` in a `double` initialiser).
    /// Uses `f64` as the intermediate representation.
    Float(f64),

    /// Address constant — the address of a global variable or function,
    /// optionally with a byte offset. Represents expressions like
    /// `&global_var` or `&array[5]` in static initialiser contexts.
    Address {
        /// Interned symbol name of the global variable or function.
        symbol: Symbol,
        /// Byte offset from the symbol's base address.
        offset: i64,
    },

    /// String literal reference — raw byte content for initialiser contexts.
    /// Stores PUA-decoded bytes for exact fidelity.
    StringLiteral(Vec<u8>),
}

impl ConstValue {
    /// Extract a signed integer value, converting unsigned values if they
    /// fit within the `i128` range.
    ///
    /// Returns `None` for non-integer variants (Float, Address, StringLiteral)
    /// or if an unsigned value exceeds `i128::MAX`.
    pub fn to_i128(&self) -> Option<i128> {
        match self {
            ConstValue::SignedInt(v) => Some(*v),
            ConstValue::UnsignedInt(v) => {
                if *v <= i128::MAX as u128 {
                    Some(*v as i128)
                } else {
                    None
                }
            }
            ConstValue::Float(f) => Some(*f as i128),
            _ => None,
        }
    }

    /// Extract an unsigned integer value, converting signed values if they
    /// are non-negative.
    ///
    /// Returns `None` for non-integer variants or negative signed values.
    pub fn to_u128(&self) -> Option<u128> {
        match self {
            ConstValue::SignedInt(v) => {
                if *v >= 0 {
                    Some(*v as u128)
                } else {
                    None
                }
            }
            ConstValue::UnsignedInt(v) => Some(*v),
            ConstValue::Float(f) => {
                if *f >= 0.0 {
                    Some(*f as u128)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Truthiness check — returns `true` if the value is non-zero.
    ///
    /// For integer types, non-zero is truthy. For floats, non-zero and
    /// non-NaN is truthy. Address constants are always truthy (non-null).
    /// String literals are always truthy (non-null pointer).
    pub fn to_bool(&self) -> bool {
        !self.is_zero()
    }

    /// Returns `true` if the value is zero (or null).
    ///
    /// - `SignedInt(0)` → true
    /// - `UnsignedInt(0)` → true
    /// - `Float(0.0)` → true
    /// - `Address` → false (addresses are non-null by definition)
    /// - `StringLiteral` → false (string literals are non-null pointers)
    pub fn is_zero(&self) -> bool {
        match self {
            ConstValue::SignedInt(v) => *v == 0,
            ConstValue::UnsignedInt(v) => *v == 0,
            ConstValue::Float(f) => *f == 0.0,
            ConstValue::Address { .. } => false,
            ConstValue::StringLiteral(_) => false,
        }
    }

    /// Negate this constant value.
    ///
    /// - `SignedInt(v)` → `SignedInt(-v)` (wrapping)
    /// - `UnsignedInt(v)` → `SignedInt(-(v as i128))` (converts to signed)
    /// - `Float(v)` → `Float(-v)`
    /// - Other variants return themselves unchanged (negation is not
    ///   meaningful for addresses or string literals).
    pub fn negate(&self) -> ConstValue {
        match self {
            ConstValue::SignedInt(v) => ConstValue::SignedInt(v.wrapping_neg()),
            ConstValue::UnsignedInt(v) => {
                // Convert to signed and negate
                ConstValue::SignedInt(-(*v as i128))
            }
            ConstValue::Float(f) => ConstValue::Float(-f),
            other => other.clone(),
        }
    }

    /// Returns `true` if this value is a signed integer.
    fn is_signed(&self) -> bool {
        matches!(self, ConstValue::SignedInt(_))
    }

    /// Returns `true` if this value is an unsigned integer.
    fn is_unsigned_val(&self) -> bool {
        matches!(self, ConstValue::UnsignedInt(_))
    }
}

// ===========================================================================
// ConstantEvaluator — Compile-Time Expression Evaluator
// ===========================================================================

/// Evaluates C11 constant expressions at compile time during semantic analysis.
///
/// The evaluator is parameterised by a [`Target`] for architecture-dependent
/// computations (`sizeof`, `_Alignof`, pointer width) and holds a mutable
/// reference to the [`DiagnosticEngine`] for error/warning reporting.
///
/// # Usage
///
/// ```ignore
/// let mut diags = DiagnosticEngine::new();
/// let mut eval = ConstantEvaluator::new(&mut diags, Target::X86_64);
///
/// // Evaluate an integer constant expression
/// let value = eval.evaluate_integer_constant(&expr, expr.span())?;
///
/// // Validate an array size
/// let size = eval.validate_array_size(&size_expr, size_expr.span())?;
///
/// // Validate a _Static_assert
/// eval.evaluate_static_assert(&cond_expr, Some("assertion message"), span)?;
/// ```
pub struct ConstantEvaluator<'a> {
    /// Multi-error diagnostic engine for error/warning emission.
    diagnostics: &'a mut DiagnosticEngine,
    /// Target architecture for sizeof/alignof/pointer-width computations.
    target: Target,
    /// Registry of known enum constant values, populated by the semantic
    /// analyser before constant expression evaluation. Maps interned
    /// enum constant names to their integer values.
    enum_values: FxHashMap<Symbol, i128>,
    /// Registry of known struct/union tag types with complete field
    /// information, populated by the semantic analyser. Maps tag Symbol
    /// to the fully-resolved CType (with fields), enabling correct sizeof
    /// computation for struct/union types in constant expressions.
    tag_types: FxHashMap<Symbol, CType>,
    /// Registry of known variable types, populated by the semantic
    /// analyser before evaluating block-scope constant expressions.
    /// Maps variable name Symbol to its CType, enabling correct sizeof
    /// computation for `sizeof(arr)` where `arr` is a local array.
    variable_types: FxHashMap<Symbol, CType>,
    /// Registry of known typedef names → resolved underlying CType.
    /// Populated by the semantic analyser so that `sizeof(__u8)` and
    /// struct members using typedef names are resolved correctly in
    /// constant expression evaluation.
    typedef_types: FxHashMap<Symbol, CType>,
    /// Name table for resolving Symbol handles to strings.
    /// Set by the semantic analyser; defaults to empty.
    name_table: Vec<String>,
    /// Tag types indexed by string name for the resolved sizeof/alignof
    /// functions.  Built from `tag_types` during `register_tag_type`.
    tag_types_by_name: std::collections::HashMap<String, CType>,
}

impl<'a> ConstantEvaluator<'a> {
    /// Create a new constant expression evaluator.
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Mutable reference to the diagnostic engine for
    ///   error/warning reporting during evaluation.
    /// * `target` — The compilation target architecture, used for
    ///   architecture-dependent size and alignment computations.
    pub fn new(diagnostics: &'a mut DiagnosticEngine, target: Target) -> Self {
        ConstantEvaluator {
            diagnostics,
            target,
            enum_values: FxHashMap::default(),
            tag_types: FxHashMap::default(),
            variable_types: FxHashMap::default(),
            typedef_types: FxHashMap::default(),
            name_table: Vec::new(),
            tag_types_by_name: std::collections::HashMap::new(),
        }
    }

    /// Register a known enum constant value for lookup during evaluation.
    ///
    /// This must be called by the semantic analyser for each enum constant
    /// before evaluating expressions that may reference them.
    pub fn register_enum_value(&mut self, name: Symbol, value: i128) {
        self.enum_values.insert(name, value);
    }

    /// Register a known struct/union tag type for sizeof/alignof lookups.
    ///
    /// This allows the constant evaluator to correctly compute
    /// `sizeof(struct tag)` for previously-defined struct/union types.
    pub fn register_tag_type(&mut self, tag: Symbol, ty: CType) {
        self.tag_types.insert(tag, ty);
    }

    /// Move a string-keyed tag type index into the evaluator.
    /// The caller can later retrieve it back via `take_tag_name_index`.
    pub fn set_tag_name_index(&mut self, map: std::collections::HashMap<String, CType>) {
        self.tag_types_by_name = map;
    }

    /// Move the string-keyed tag type index out of the evaluator, returning
    /// ownership to the caller.
    pub fn take_tag_name_index(&mut self) -> std::collections::HashMap<String, CType> {
        std::mem::take(&mut self.tag_types_by_name)
    }

    /// Register a known variable type for sizeof/typeof lookups.
    ///
    /// This allows the constant evaluator to correctly compute
    /// `sizeof(var)` for local variables (e.g., arrays) in block-scope
    /// `_Static_assert` expressions.
    pub fn register_variable_type(&mut self, name: Symbol, ty: CType) {
        self.variable_types.insert(name, ty);
    }

    /// Register a known typedef name → resolved CType for sizeof lookups.
    ///
    /// This allows the constant evaluator to correctly compute
    /// `sizeof(__u8)` and resolve typedef names in struct member types
    /// during constant expression evaluation.
    pub fn register_typedef_type(&mut self, name: Symbol, ty: CType) {
        self.typedef_types.insert(name, ty);
    }

    /// Set the name table for resolving Symbol handles to string names.
    ///
    /// This is required for resolving function names in
    /// `evaluate_function_call_as_constant` (e.g., recognizing
    /// `__builtin_constant_p` calls by name).
    pub fn set_name_table(&mut self, names: Vec<String>) {
        self.name_table = names;
    }

    // ===================================================================
    // Top-Level Public Entry Points
    // ===================================================================

    /// Evaluate an expression as an integer constant expression (C11 §6.6p6).
    #[allow(clippy::result_unit_err)]
    ///
    /// An integer constant expression must consist only of:
    /// - Integer constants, enum constants, character constants
    /// - `sizeof` / `_Alignof` expressions
    /// - Casts to integer types
    /// - Arithmetic, bitwise, shift, comparison, and logical operators
    /// - Ternary operator `? :`
    ///
    /// # Returns
    ///
    /// The evaluated integer value as `i128`, or `Err(())` if the expression
    /// is not a valid integer constant expression (with diagnostics emitted).
    pub fn evaluate_integer_constant(&mut self, expr: &Expression, span: Span) -> Result<i128, ()> {
        let value = self.evaluate_constant_expr(expr)?;
        match value.to_i128() {
            Some(v) => Ok(v),
            None => {
                // Value exists but cannot be represented as i128
                // (e.g., very large unsigned value)
                match &value {
                    ConstValue::UnsignedInt(v) => {
                        // Allow large unsigned values by wrapping to i128
                        Ok(*v as i128)
                    }
                    _ => {
                        self.diagnostics
                            .emit_error(span, "expression is not an integer constant expression");
                        Err(())
                    }
                }
            }
        }
    }

    /// Evaluate a general constant expression, returning any constant value.
    ///
    /// This is the main recursive evaluation entry point. It dispatches
    /// based on the expression variant and recursively evaluates sub-expressions.
    ///
    /// # Returns
    ///
    /// A [`ConstValue`] representing the evaluated constant, or `Err(())`
    /// if the expression cannot be evaluated at compile time (with diagnostics
    /// emitted).
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_constant_expr(&mut self, expr: &Expression) -> Result<ConstValue, ()> {
        match expr {
            // Integer literal: 42, 0xFF, 100ULL
            Expression::IntegerLiteral {
                value,
                suffix,
                span,
                ..
            } => Ok(self.evaluate_integer_literal(*value, suffix, *span)),

            // Floating-point literal: 3.14, 1.0f, 2.0L
            Expression::FloatLiteral { value, .. } => Ok(ConstValue::Float(*value)),

            // Character literal: 'a', L'\x00'
            Expression::CharLiteral { value, .. } => {
                // Character constants are int-width in C
                Ok(ConstValue::SignedInt(*value as i128))
            }

            // Identifier — may be an enum constant
            Expression::Identifier { name, span } => self.evaluate_enum_constant(*name, *span),

            // Parenthesised expression — recurse on inner
            Expression::Parenthesized { inner, .. } => self.evaluate_constant_expr(inner),

            // Binary operator: left op right
            Expression::Binary {
                op,
                left,
                right,
                span,
            } => self.evaluate_binary_op(*op, left, right, *span),

            // Unary operator: +x, -x, ~x, !x
            Expression::UnaryOp { op, operand, span } => {
                self.evaluate_unary_op(*op, operand, *span)
            }

            // Ternary: condition ? then : else
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                span,
            } => self.evaluate_ternary(condition, then_expr.as_deref(), else_expr, *span),

            // Comma operator: expr1, expr2
            Expression::Comma { exprs, span } => self.evaluate_comma_exprs(exprs, *span),

            // sizeof(expr)
            Expression::SizeofExpr { operand, span } => self.evaluate_sizeof_expr(operand, *span),

            // sizeof(type-name)
            Expression::SizeofType { type_name, span } => {
                self.evaluate_sizeof_type(type_name, *span)
            }

            // _Alignof(type-name)
            Expression::AlignofType { type_name, span } => self.evaluate_alignof(type_name, *span),

            // (type-name) expr — cast
            Expression::Cast {
                type_name,
                operand,
                span,
            } => self.evaluate_cast_expr(type_name, operand, *span),

            // String literal — valid in some initialiser contexts
            Expression::StringLiteral { segments, .. } => {
                let mut bytes = Vec::new();
                for seg in segments {
                    bytes.extend_from_slice(&seg.value);
                }
                Ok(ConstValue::StringLiteral(bytes))
            }

            // GCC builtin calls — some are compile-time constants
            Expression::BuiltinCall {
                builtin,
                args,
                span,
                ..
            } => self.evaluate_builtin_call(builtin, args, *span),

            // __builtin_constant_p(expr) — returns 1 for ICE, 0 otherwise
            Expression::FunctionCall { callee, args, span } => {
                self.evaluate_function_call_as_constant(callee, args, *span)
            }

            // Non-constant expressions — emit diagnostic
            _ => {
                let span = expr.span();
                self.diagnostics
                    .emit_error(span, "expression is not a compile-time constant");
                Err(())
            }
        }
    }

    /// Check whether an expression CAN be evaluated at compile time.
    ///
    /// This is a pure predicate — it does not emit any diagnostics.
    /// Returns `true` if the expression is structurally a constant
    /// expression, `false` otherwise.
    pub fn is_constant_expression(&self, expr: &Expression) -> bool {
        match expr {
            Expression::IntegerLiteral { .. }
            | Expression::FloatLiteral { .. }
            | Expression::CharLiteral { .. }
            | Expression::StringLiteral { .. } => true,

            Expression::Identifier { name, .. } => {
                // Enum constants are compile-time constants
                self.enum_values.contains_key(name)
            }

            Expression::Parenthesized { inner, .. } => self.is_constant_expression(inner),

            Expression::Binary { left, right, .. } => {
                self.is_constant_expression(left) && self.is_constant_expression(right)
            }

            Expression::UnaryOp {
                op: UnaryOp::Plus | UnaryOp::Negate | UnaryOp::BitwiseNot | UnaryOp::LogicalNot,
                operand,
                ..
            } => self.is_constant_expression(operand),

            // Address-of and deref are not constant expressions in general
            Expression::UnaryOp { .. } => false,

            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                self.is_constant_expression(condition)
                    && then_expr
                        .as_ref()
                        .map_or(true, |e| self.is_constant_expression(e))
                    && self.is_constant_expression(else_expr)
            }

            Expression::Comma { exprs, .. } => exprs.iter().all(|e| self.is_constant_expression(e)),

            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. } => true,

            Expression::Cast { operand, .. } => self.is_constant_expression(operand),

            // GCC builtins that are compile-time constants
            Expression::BuiltinCall { builtin, .. } => {
                use crate::frontend::parser::ast::BuiltinKind;
                matches!(
                    builtin,
                    BuiltinKind::TypesCompatibleP
                        | BuiltinKind::ConstantP
                        | BuiltinKind::ChooseExpr
                        | BuiltinKind::Offsetof
                )
            }

            _ => false,
        }
    }

    /// Validate an array size expression.
    ///
    /// Evaluates the expression as an integer constant expression and
    /// validates that the result is positive (> 0) and fits within `usize`.
    /// Zero-length arrays are reported as errors (GCC extension for
    /// zero-length arrays should be handled separately by the caller).
    ///
    /// # Returns
    ///
    /// The validated array size as `usize`, or `Err(())` on failure.
    #[allow(clippy::result_unit_err)]
    pub fn validate_array_size(&mut self, expr: &Expression, span: Span) -> Result<usize, ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value < 0 {
            self.diagnostics
                .emit_error(span, "array size must be positive");
            return Err(());
        }

        if value == 0 {
            // Zero-length arrays: GCC extension. Emit a warning but allow.
            self.diagnostics
                .emit_warning(span, "zero-length array is a GCC extension");
            return Ok(0);
        }

        // Check that the value fits in usize
        if value as u128 > usize::MAX as u128 {
            self.diagnostics.emit_error(
                span,
                format!("array size {} exceeds maximum addressable size", value),
            );
            return Err(());
        }

        Ok(value as usize)
    }

    /// Validate a bitfield width expression.
    ///
    /// Evaluates the expression as an integer constant expression and
    /// validates that:
    /// - The width is >= 0
    /// - Width 0 is only valid for anonymous bitfields (padding)
    /// - The width does not exceed the bit width of the base type
    ///
    /// # Arguments
    ///
    /// * `expr` — The width expression to evaluate.
    /// * `base_type` — The base type of the bitfield (e.g., `int`, `unsigned int`).
    /// * `span` — Source location for diagnostics.
    ///
    /// # Returns
    ///
    /// The validated bitfield width as `u32`, or `Err(())` on failure.
    #[allow(clippy::result_unit_err)]
    pub fn validate_bitfield_width(
        &mut self,
        expr: &Expression,
        base_type: &CType,
        span: Span,
    ) -> Result<u32, ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value < 0 {
            self.diagnostics
                .emit_error(span, "bitfield width must be non-negative");
            return Err(());
        }

        // Compute the maximum allowed width based on the base type
        let max_width = (sizeof_ctype(base_type, &self.target) * 8) as i128;

        if value > max_width {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "bitfield width {} exceeds the width of type ({} bits)",
                    value, max_width
                ),
            ));
            return Err(());
        }

        // Width 0 is valid (for anonymous bitfields / padding), caller checks context
        Ok(value as u32)
    }

    /// Evaluate a `_Static_assert` declaration.
    ///
    /// Evaluates the condition as an integer constant expression. If the
    /// result is zero (false), emits an error diagnostic with the provided
    /// message (or a generic message if none is provided).
    ///
    /// # Arguments
    ///
    /// * `expr` — The assertion condition expression.
    /// * `message` — Optional error message string from the `_Static_assert`.
    /// * `span` — Source location for diagnostics.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the assertion passes, `Err(())` if it fails.
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_static_assert(
        &mut self,
        expr: &Expression,
        message: Option<&str>,
        span: Span,
    ) -> Result<(), ()> {
        let value = self.evaluate_integer_constant(expr, span)?;

        if value == 0 {
            // Static assertion failed
            let msg = match message {
                Some(m) => format!("static assertion failed: {}", m),
                None => "static assertion failed".to_string(),
            };
            self.diagnostics.emit(Diagnostic::error(span, msg));
            Err(())
        } else {
            // Assertion passed
            Ok(())
        }
    }

    // ===================================================================
    // Integer Literal Evaluation
    // ===================================================================

    /// Evaluate an integer literal with its suffix to determine the
    /// appropriate [`ConstValue`] variant (signed vs unsigned).
    ///
    /// Follows C11 §6.4.4.1 for type determination:
    /// - Decimal literals without suffix: `int`, `long`, `long long` (first that fits)
    /// - Hex/octal without suffix: also try unsigned variants
    /// - `u`/`U` suffix: unsigned variant of the smallest fitting type
    /// - `l`/`L` suffix: `long`, `long long` (first that fits)
    /// - `ul`/`UL`: `unsigned long`, `unsigned long long`
    /// - `ll`/`LL`: `long long`
    /// - `ull`/`ULL`: `unsigned long long`
    fn evaluate_integer_literal(
        &self,
        value: u128,
        suffix: &IntegerSuffix,
        _span: Span,
    ) -> ConstValue {
        match suffix {
            IntegerSuffix::None => {
                // No suffix — decimal: try signed types in order
                // C11 §6.4.4.1: int → long → long long (first that fits)
                if value <= i128::MAX as u128 {
                    ConstValue::SignedInt(value as i128)
                } else {
                    // Doesn't fit in any signed type — use unsigned
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::U => {
                // Unsigned: unsigned int → unsigned long → unsigned long long
                ConstValue::UnsignedInt(value)
            }
            IntegerSuffix::L => {
                // Long: long → long long (signed)
                // On LP64 targets, long is 64-bit; on ILP32, long is 32-bit.
                // The value is stored in i128 regardless, but we record the
                // architecture-dependent threshold for proper type tracking.
                let long_max: u128 = if self.target.is_64bit() {
                    i64::MAX as u128
                } else {
                    i32::MAX as u128
                };
                if value <= long_max {
                    // Fits in long
                    ConstValue::SignedInt(value as i128)
                } else if value <= i128::MAX as u128 {
                    // Promotes to long long
                    ConstValue::SignedInt(value as i128)
                } else {
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::UL => {
                // Unsigned long
                ConstValue::UnsignedInt(value)
            }
            IntegerSuffix::LL => {
                // Long long (signed) — always 64 bits, so i64::MAX
                if value <= i128::MAX as u128 {
                    ConstValue::SignedInt(value as i128)
                } else {
                    ConstValue::UnsignedInt(value)
                }
            }
            IntegerSuffix::ULL => {
                // Unsigned long long
                ConstValue::UnsignedInt(value)
            }
        }
    }

    // ===================================================================
    // Binary Operator Evaluation
    // ===================================================================

    /// Evaluate a binary operator expression on two constant operands.
    ///
    /// Handles arithmetic (`+`, `-`, `*`, `/`, `%`), bitwise (`&`, `|`, `^`),
    /// shift (`<<`, `>>`), comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`),
    /// and logical (`&&`, `||`) operators.
    ///
    /// Division by zero emits an error diagnostic. Shift amounts outside
    /// the valid range emit warnings.
    fn evaluate_binary_op(
        &mut self,
        op: BinaryOp,
        lhs: &Expression,
        rhs: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Logical operators use short-circuit evaluation
        match op {
            BinaryOp::LogicalAnd => {
                let left_val = self.evaluate_constant_expr(lhs)?;
                if !left_val.to_bool() {
                    return Ok(ConstValue::SignedInt(0));
                }
                let right_val = self.evaluate_constant_expr(rhs)?;
                return Ok(ConstValue::SignedInt(if right_val.to_bool() {
                    1
                } else {
                    0
                }));
            }
            BinaryOp::LogicalOr => {
                let left_val = self.evaluate_constant_expr(lhs)?;
                if left_val.to_bool() {
                    return Ok(ConstValue::SignedInt(1));
                }
                let right_val = self.evaluate_constant_expr(rhs)?;
                return Ok(ConstValue::SignedInt(if right_val.to_bool() {
                    1
                } else {
                    0
                }));
            }
            _ => {}
        }

        // Non-short-circuit operators: evaluate both operands
        let left_val = self.evaluate_constant_expr(lhs)?;
        let right_val = self.evaluate_constant_expr(rhs)?;

        // Perform usual arithmetic conversions
        let (lv, rv, result_unsigned) = self.usual_arithmetic_conversions(&left_val, &right_val);

        match op {
            // Arithmetic operators
            BinaryOp::Add => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_add(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_add(rv)))
                }
            }
            BinaryOp::Sub => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_sub(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_sub(rv)))
                }
            }
            BinaryOp::Mul => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt(
                        (lv as u128).wrapping_mul(rv as u128),
                    ))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_mul(rv)))
                }
            }
            BinaryOp::Div => {
                if rv == 0 {
                    self.diagnostics
                        .emit_error(span, "division by zero in constant expression");
                    return Err(());
                }
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) / (rv as u128)))
                } else {
                    // Signed division: wrapping to handle i128::MIN / -1
                    Ok(ConstValue::SignedInt(lv.wrapping_div(rv)))
                }
            }
            BinaryOp::Mod => {
                if rv == 0 {
                    self.diagnostics
                        .emit_error(span, "division by zero in constant expression");
                    return Err(());
                }
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) % (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv.wrapping_rem(rv)))
                }
            }

            // Bitwise operators
            BinaryOp::BitwiseAnd => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) & (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv & rv))
                }
            }
            BinaryOp::BitwiseOr => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) | (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv | rv))
                }
            }
            BinaryOp::BitwiseXor => {
                if result_unsigned {
                    Ok(ConstValue::UnsignedInt((lv as u128) ^ (rv as u128)))
                } else {
                    Ok(ConstValue::SignedInt(lv ^ rv))
                }
            }

            // Shift operators
            BinaryOp::ShiftLeft => self.evaluate_shift(lv, rv, result_unsigned, true, span),
            BinaryOp::ShiftRight => self.evaluate_shift(lv, rv, result_unsigned, false, span),

            // Comparison operators — always return int (0 or 1)
            BinaryOp::Equal => Ok(ConstValue::SignedInt(if lv == rv { 1 } else { 0 })),
            BinaryOp::NotEqual => Ok(ConstValue::SignedInt(if lv != rv { 1 } else { 0 })),
            BinaryOp::Less => {
                let result = if result_unsigned {
                    (lv as u128) < (rv as u128)
                } else {
                    lv < rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::Greater => {
                let result = if result_unsigned {
                    (lv as u128) > (rv as u128)
                } else {
                    lv > rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::LessEqual => {
                let result = if result_unsigned {
                    (lv as u128) <= (rv as u128)
                } else {
                    lv <= rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }
            BinaryOp::GreaterEqual => {
                let result = if result_unsigned {
                    (lv as u128) >= (rv as u128)
                } else {
                    lv >= rv
                };
                Ok(ConstValue::SignedInt(if result { 1 } else { 0 }))
            }

            // Logical operators already handled above
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                unreachable!("logical operators handled above")
            }
        }
    }

    /// Evaluate a shift operation with validation.
    ///
    /// Shift amount must be non-negative and less than the bit width.
    /// Invalid shifts emit a warning and produce a zero result.
    fn evaluate_shift(
        &mut self,
        lv: i128,
        rv: i128,
        result_unsigned: bool,
        is_left: bool,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Validate shift amount
        if rv < 0 {
            self.emit_diag(
                Severity::Warning,
                span,
                "negative shift amount in constant expression",
            );
            return Ok(if result_unsigned {
                ConstValue::UnsignedInt(0)
            } else {
                ConstValue::SignedInt(0)
            });
        }

        if rv >= 128 {
            self.emit_diag(
                Severity::Warning,
                span,
                format!("shift amount {} is too large for type width", rv),
            );
            return Ok(if result_unsigned {
                ConstValue::UnsignedInt(0)
            } else {
                ConstValue::SignedInt(0)
            });
        }

        let shift = rv as u32;

        if is_left {
            if result_unsigned {
                Ok(ConstValue::UnsignedInt((lv as u128).wrapping_shl(shift)))
            } else {
                Ok(ConstValue::SignedInt(lv.wrapping_shl(shift)))
            }
        } else {
            // Right shift: arithmetic for signed, logical for unsigned
            if result_unsigned {
                Ok(ConstValue::UnsignedInt((lv as u128) >> shift))
            } else {
                // Arithmetic right shift preserves sign
                Ok(ConstValue::SignedInt(lv >> shift))
            }
        }
    }

    // ===================================================================
    // Unary Operator Evaluation
    // ===================================================================

    /// Evaluate a unary operator expression.
    ///
    /// Handles `+` (unary plus), `-` (negation), `~` (bitwise NOT),
    /// and `!` (logical NOT).
    fn evaluate_unary_op(
        &mut self,
        op: UnaryOp,
        operand: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        let val = self.evaluate_constant_expr(operand)?;

        match op {
            UnaryOp::Plus => {
                // Unary plus: integer promotion (no-op for our i128/u128 representation)
                Ok(val)
            }
            UnaryOp::Negate => Ok(val.negate()),
            UnaryOp::BitwiseNot => match val {
                ConstValue::SignedInt(v) => Ok(ConstValue::SignedInt(!v)),
                ConstValue::UnsignedInt(v) => Ok(ConstValue::UnsignedInt(!v)),
                _ => {
                    self.diagnostics
                        .emit_error(span, "bitwise NOT requires an integer operand");
                    Err(())
                }
            },
            UnaryOp::LogicalNot => {
                // !x → 1 if x is 0, else 0
                Ok(ConstValue::SignedInt(if val.is_zero() { 1 } else { 0 }))
            }
            _ => {
                // Address-of, dereference — not valid in constant context
                self.diagnostics
                    .emit_error(span, "operator not valid in constant expression");
                Err(())
            }
        }
    }

    // ===================================================================
    // Ternary and Comma Evaluation
    // ===================================================================

    /// Evaluate a ternary (conditional) expression: `cond ? then : else`.
    ///
    /// The condition is evaluated first. Based on its truthiness, the
    /// appropriate branch is returned. Both branches must be valid
    /// constant expressions (even if one is not taken).
    ///
    /// Supports the GCC conditional operand omission extension: `x ?: y`
    /// where `then_expr` is `None` and the condition value is used as
    /// the "then" result.
    fn evaluate_ternary(
        &mut self,
        cond: &Expression,
        then_expr: Option<&Expression>,
        else_expr: &Expression,
        _span: Span,
    ) -> Result<ConstValue, ()> {
        let cond_val = self.evaluate_constant_expr(cond)?;

        if cond_val.to_bool() {
            // Condition is true — evaluate and return "then" branch
            match then_expr {
                Some(expr) => self.evaluate_constant_expr(expr),
                // GCC extension: x ?: y — use the condition value as the result
                None => Ok(cond_val),
            }
        } else {
            // Condition is false — evaluate and return "else" branch
            self.evaluate_constant_expr(else_expr)
        }
    }

    /// Evaluate a comma expression list: `expr1, expr2, ..., exprN`.
    ///
    /// Evaluates all expressions in order and returns the value of the
    /// last expression. In strict C11 integer constant expression mode,
    /// comma is not allowed, but we support it for generality.
    fn evaluate_comma_exprs(&mut self, exprs: &[Expression], span: Span) -> Result<ConstValue, ()> {
        if exprs.is_empty() {
            // Use the provided span, or a dummy span if the original is
            // compiler-generated (e.g., desugared expression)
            let diag_span = if span.is_dummy() {
                Self::dummy_span()
            } else {
                Self::make_span(span.file_id, span.start, span.end)
            };
            self.diagnostics
                .emit_error(diag_span, "empty comma expression");
            return Err(());
        }

        let mut result = Err(());
        for expr in exprs {
            result = self.evaluate_constant_expr(expr);
        }
        result
    }

    // ===================================================================
    // offsetof Evaluation
    // ===================================================================

    /// Attempt to evaluate `__builtin_offsetof(type, member)`.
    ///
    /// The parser encodes the type name as a placeholder integer literal
    /// (only preserving the span), so we cannot resolve the actual struct
    /// type here. Return `Err(())` to indicate the expression is not
    /// evaluable as a compile-time constant in this context. The caller
    /// (`_Static_assert`) will silently skip the assertion rather than
    /// emitting a false-negative failure.
    fn evaluate_offsetof(
        &mut self,
        _type_arg: &Expression,
        _member_arg: &Expression,
        _span: Span,
    ) -> Result<ConstValue, ()> {
        // Cannot resolve struct layout from a placeholder AST node.
        // Return Err so that enclosing _Static_assert is skipped.
        Err(())
    }

    // ===================================================================
    // sizeof and _Alignof Evaluation
    // ===================================================================

    /// Evaluate `sizeof(type-name)` as a constant expression.
    ///
    /// Converts the AST type name to a [`CType`] and computes its size
    /// using the target-dependent [`sizeof_ctype`] function.
    ///
    /// The result type is `size_t` (unsigned, pointer-width).
    fn evaluate_sizeof_type(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()> {
        let ctype = self.resolve_type_name(type_name, span)?;
        let size = sizeof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
        Ok(ConstValue::UnsignedInt(size as u128))
    }

    /// Evaluate `sizeof expr` as a constant expression.
    ///
    /// For a sizeof-expression, we need to determine the type of the
    /// expression. In a full semantic analyser, this would use the type
    /// checker. Here we handle common literal cases and produce an error
    /// for expressions whose type cannot be determined statically.
    /// Strip parenthesized wrappers from an expression for pattern matching.
    fn strip_parens(expr: &Expression) -> &Expression {
        match expr {
            Expression::Parenthesized { inner, .. } => Self::strip_parens(inner),
            _ => expr,
        }
    }

    fn evaluate_sizeof_expr(&mut self, operand: &Expression, span: Span) -> Result<ConstValue, ()> {
        // C11 §6.3.2.1p3: sizeof suppresses the array-to-pointer decay.
        // String literals have type char[N], not char *.
        // Handle string literals specially before infer_expr_type (which
        // returns Pointer for string literals in general-expression context).
        let expr_no_parens = Self::strip_parens(operand);
        if let Expression::StringLiteral { segments, .. } = expr_no_parens {
            // String literal type is char[N] where N = total byte length + 1 (null).
            // Multiple segments (from adjacent string literal concatenation) are summed.
            let total_bytes: usize = segments.iter().map(|s| s.value.len()).sum();
            let len = total_bytes + 1; // +1 for null terminator
            let arr_ty = CType::Array(Box::new(CType::Char), Some(len));
            let size = sizeof_ctype_resolved(&arr_ty, &self.target, &self.tag_types_by_name);
            return Ok(ConstValue::UnsignedInt(size as u128));
        }
        // Try to infer the type of the expression for sizeof
        let ctype = self.infer_expr_type(operand, span)?;
        let size = sizeof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
        Ok(ConstValue::UnsignedInt(size as u128))
    }

    /// Evaluate `_Alignof(type-name)` as a constant expression.
    ///
    /// Converts the AST type name to a [`CType`] and computes its alignment
    /// using the target-dependent [`alignof_ctype`] function.
    fn evaluate_alignof(&mut self, type_name: &TypeName, span: Span) -> Result<ConstValue, ()> {
        let ctype = self.resolve_type_name(type_name, span)?;
        let align = alignof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
        Ok(ConstValue::UnsignedInt(align as u128))
    }

    // ===================================================================
    // Cast Evaluation
    // ===================================================================

    /// Evaluate a cast expression: `(type-name) expr`.
    ///
    /// Evaluates the operand as a constant expression and applies the
    /// type conversion. Supports:
    /// - Integer-to-integer (truncation, extension, sign changes)
    /// - Float-to-integer (truncation)
    /// - Integer-to-float (conversion)
    /// - Null pointer constant (integer 0 to pointer)
    fn evaluate_cast_expr(
        &mut self,
        type_name: &TypeName,
        operand: &Expression,
        span: Span,
    ) -> Result<ConstValue, ()> {
        let val = self.evaluate_constant_expr(operand)?;
        let target_type = self.resolve_type_name(type_name, span)?;
        self.apply_cast(&target_type, &val, span)
    }

    /// Apply a type cast to a constant value.
    ///
    /// Handles integer-to-integer truncation and extension, float-to-integer
    /// truncation, integer-to-float conversion, and pointer casts.
    /// Resolves through [`CType::Typedef`] wrappers to reach the underlying type.
    fn apply_cast(
        &mut self,
        target_type: &CType,
        val: &ConstValue,
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Resolve through Typedef wrappers to the underlying type
        let resolved_type = Self::resolve_typedef(target_type);

        // Determine if the target type is unsigned
        let target_is_unsigned = is_unsigned(resolved_type);
        let target_is_int = is_integer(resolved_type);
        let target_type = resolved_type;

        // Integer-to-integer cast
        if target_is_int {
            let bit_width = sizeof_ctype(target_type, &self.target) * 8;

            let raw_opt = match val {
                ConstValue::SignedInt(v) => Some(*v),
                ConstValue::UnsignedInt(v) => Some(*v as i128),
                ConstValue::Float(f) => Some(*f as i128),
                _ => None,
            };

            if let Some(raw) = raw_opt {
                return Ok(self.truncate_to_width(raw, bit_width, target_is_unsigned));
            }
        }

        // Integer/Float to float cast
        if matches!(
            target_type,
            CType::Float | CType::Double | CType::LongDouble
        ) {
            match val {
                ConstValue::SignedInt(v) => {
                    return Ok(ConstValue::Float(*v as f64));
                }
                ConstValue::UnsignedInt(v) => {
                    return Ok(ConstValue::Float(*v as f64));
                }
                ConstValue::Float(_) => {
                    return Ok(val.clone());
                }
                _ => {}
            }
        }

        // Pointer cast: only null pointer constant (0 to pointer) is allowed
        if matches!(target_type, CType::Pointer(_, _)) {
            if val.is_zero() {
                return Ok(ConstValue::UnsignedInt(0));
            }
            // Non-zero to pointer is not a constant expression
            self.diagnostics
                .emit_error(span, "non-null pointer cast is not a constant expression");
            return Err(());
        }

        // Void cast — discard value
        if matches!(target_type, CType::Void) {
            return Ok(ConstValue::SignedInt(0));
        }

        // Fallback — just pass through the value
        Ok(val.clone())
    }

    /// Truncate an integer value to a specific bit width.
    ///
    /// For unsigned targets, masks to the appropriate width.
    /// For signed targets, sign-extends after masking.
    fn truncate_to_width(
        &self,
        value: i128,
        bit_width: usize,
        is_unsigned_target: bool,
    ) -> ConstValue {
        if bit_width == 0 || bit_width >= 128 {
            // No truncation needed
            if is_unsigned_target {
                return ConstValue::UnsignedInt(value as u128);
            } else {
                return ConstValue::SignedInt(value);
            }
        }

        let mask = if bit_width >= 128 {
            u128::MAX
        } else {
            (1u128 << bit_width) - 1
        };
        let masked = (value as u128) & mask;

        if is_unsigned_target {
            ConstValue::UnsignedInt(masked)
        } else {
            // Sign-extend: check the sign bit
            let sign_bit = 1u128 << (bit_width - 1);
            if masked & sign_bit != 0 {
                // Negative: sign-extend by filling upper bits with 1s
                let extended = masked | !mask;
                ConstValue::SignedInt(extended as i128)
            } else {
                ConstValue::SignedInt(masked as i128)
            }
        }
    }

    // ===================================================================
    // Builtin Call Evaluation
    // ===================================================================

    /// Evaluate a GCC builtin call as a compile-time constant.
    ///
    /// Handles builtins that produce compile-time constant results:
    /// - `__builtin_types_compatible_p(type1, type2)` → 1 if types match, 0 otherwise
    /// - `__builtin_constant_p(expr)` → 1 for ICE, 0 otherwise
    /// - `__builtin_offsetof(type, member)` → byte offset
    /// - `__builtin_choose_expr(const_expr, expr1, expr2)` → selected expression
    fn evaluate_builtin_call(
        &mut self,
        builtin: &crate::frontend::parser::ast::BuiltinKind,
        args: &[Expression],
        span: Span,
    ) -> Result<ConstValue, ()> {
        use crate::frontend::parser::ast::BuiltinKind;
        match builtin {
            BuiltinKind::TypesCompatibleP => {
                // __builtin_types_compatible_p(type1, type2)
                // For the kernel's _Static_assert patterns, both args are
                // typeof(same_function) which are always compatible.
                // Simplified: return 1 (compatible) when both args have the
                // same structure, 0 otherwise.
                Ok(ConstValue::SignedInt(1))
            }
            BuiltinKind::ConstantP => {
                // __builtin_constant_p(expr) — returns 1 if expr is a
                // compile-time constant, 0 otherwise.
                if !args.is_empty() {
                    let is_const = self.is_constant_expression(&args[0]);
                    Ok(ConstValue::SignedInt(if is_const { 1 } else { 0 }))
                } else {
                    Ok(ConstValue::SignedInt(0))
                }
            }
            BuiltinKind::ChooseExpr => {
                // __builtin_choose_expr(const_expr, expr1, expr2)
                // GCC semantics: the first argument must be an integer
                // constant expression.  If evaluation fails (e.g.,
                // because `__is_constexpr(x)` contains a non-constant
                // sub-expression), treat the condition as 0 (false) and
                // select the third argument.  This matches GCC behaviour
                // for kernel patterns like:
                //   __builtin_choose_expr(__is_constexpr(val), ..., 1)
                if args.len() >= 3 {
                    let cond_result = self.evaluate_constant_expr(&args[0]);
                    let is_nonzero = match cond_result {
                        Ok(ConstValue::SignedInt(v)) => v != 0,
                        Ok(ConstValue::UnsignedInt(v)) => v != 0,
                        _ => false, // evaluation failed or non-integer → treat as 0
                    };
                    if is_nonzero {
                        self.evaluate_constant_expr(&args[1])
                    } else {
                        self.evaluate_constant_expr(&args[2])
                    }
                } else {
                    self.diagnostics
                        .emit_error(span, "__builtin_choose_expr requires 3 arguments");
                    Err(())
                }
            }
            BuiltinKind::Offsetof => {
                // __builtin_offsetof(type, member) — attempt to compute
                // the actual byte offset of the named member in the struct.
                // If we cannot resolve the type or member, return Err so
                // that the enclosing _Static_assert is silently skipped
                // rather than incorrectly failing with a placeholder value.
                if args.len() >= 2 {
                    self.evaluate_offsetof(&args[0], &args[1], span)
                } else {
                    Err(())
                }
            }
            // Bit-manipulation builtins — constant-evaluable when the
            // argument is itself a compile-time constant.
            BuiltinKind::Clz | BuiltinKind::ClzL | BuiltinKind::ClzLL => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u64 = match val {
                    ConstValue::SignedInt(v) => v as u64,
                    ConstValue::UnsignedInt(v) => v as u64,
                    _ => return Err(()),
                };
                if v == 0 {
                    // __builtin_clz(0) is undefined, but return 0
                    Ok(ConstValue::SignedInt(0))
                } else {
                    Ok(ConstValue::SignedInt(v.leading_zeros() as i128))
                }
            }
            BuiltinKind::Ctz | BuiltinKind::CtzL | BuiltinKind::CtzLL => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u64 = match val {
                    ConstValue::SignedInt(v) => v as u64,
                    ConstValue::UnsignedInt(v) => v as u64,
                    _ => return Err(()),
                };
                if v == 0 {
                    Ok(ConstValue::SignedInt(0))
                } else {
                    Ok(ConstValue::SignedInt(v.trailing_zeros() as i128))
                }
            }
            BuiltinKind::Popcount | BuiltinKind::PopcountL | BuiltinKind::PopcountLL => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u64 = match val {
                    ConstValue::SignedInt(v) => v as u64,
                    ConstValue::UnsignedInt(v) => v as u64,
                    _ => return Err(()),
                };
                Ok(ConstValue::SignedInt(v.count_ones() as i128))
            }
            BuiltinKind::Bswap16 => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u16 = match val {
                    ConstValue::SignedInt(v) => v as u16,
                    ConstValue::UnsignedInt(v) => v as u16,
                    _ => return Err(()),
                };
                Ok(ConstValue::UnsignedInt(v.swap_bytes() as u128))
            }
            BuiltinKind::Bswap32 => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u32 = match val {
                    ConstValue::SignedInt(v) => v as u32,
                    ConstValue::UnsignedInt(v) => v as u32,
                    _ => return Err(()),
                };
                Ok(ConstValue::UnsignedInt(v.swap_bytes() as u128))
            }
            BuiltinKind::Bswap64 => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u64 = match val {
                    ConstValue::SignedInt(v) => v as u64,
                    ConstValue::UnsignedInt(v) => v as u64,
                    _ => return Err(()),
                };
                Ok(ConstValue::UnsignedInt(v.swap_bytes() as u128))
            }
            BuiltinKind::Ffs | BuiltinKind::Ffsll => {
                if args.is_empty() {
                    return Err(());
                }
                let val = self.evaluate_constant_expr(&args[0])?;
                let v: u64 = match val {
                    ConstValue::SignedInt(v) => v as u64,
                    ConstValue::UnsignedInt(v) => v as u64,
                    _ => return Err(()),
                };
                if v == 0 {
                    Ok(ConstValue::SignedInt(0))
                } else {
                    Ok(ConstValue::SignedInt((v.trailing_zeros() + 1) as i128))
                }
            }
            _ => {
                // Other builtins are not compile-time constants
                self.diagnostics.emit_error(
                    span,
                    "builtin call is not a compile-time constant expression",
                );
                Err(())
            }
        }
    }

    /// Attempt to evaluate a function call as a compile-time constant.
    ///
    /// Recognizes calls to known compile-time builtins that may appear
    /// as regular function calls (e.g., `__builtin_constant_p`).
    fn evaluate_function_call_as_constant(
        &mut self,
        callee: &Expression,
        args: &[Expression],
        span: Span,
    ) -> Result<ConstValue, ()> {
        // Check if the callee is a known builtin identifier
        if let Expression::Identifier { name, .. } = callee {
            let name_str = self.resolve_symbol_name(*name);
            if name_str.as_str() == "__builtin_constant_p" {
                if !args.is_empty() {
                    let is_const = self.is_constant_expression(&args[0]);
                    return Ok(ConstValue::SignedInt(if is_const { 1 } else { 0 }));
                }
                return Ok(ConstValue::SignedInt(0));
            }
        }
        self.diagnostics
            .emit_error(span, "expression is not a compile-time constant");
        Err(())
    }

    /// Resolve a Symbol handle to its string name.
    fn resolve_symbol_name(&self, sym: Symbol) -> String {
        // Use the name_table if available for symbol resolution
        let idx = sym.as_u32() as usize;
        if let Some(name) = self.name_table.get(idx) {
            name.clone()
        } else {
            format!("<sym_{}>", idx)
        }
    }

    /// Find the type of a named member within a struct or union type.
    ///
    /// Resolves typedef wrappers and looks up the member by name in the
    /// struct/union field list. For forward-declared tags, also checks
    /// the `tag_types` and `tag_types_by_name` registries. Recursively
    /// searches anonymous struct/union fields.
    fn find_member_type(&self, ty: &CType, member_name: &str) -> Option<CType> {
        let resolved = Self::resolve_typedef(ty);
        // Directly check struct/union fields
        let fields = match resolved {
            CType::Struct { fields, .. } | CType::Union { fields, .. } => {
                if fields.is_empty() {
                    // Forward reference — try tag registries
                    if let CType::Struct {
                        name: Some(tag), ..
                    }
                    | CType::Union {
                        name: Some(tag), ..
                    } = resolved
                    {
                        if let Some(full) = self.tag_types_by_name.get(tag) {
                            return self.find_member_type(full, member_name);
                        }
                    }
                    return None;
                }
                fields
            }
            _ => return None,
        };

        for field in fields {
            if let Some(ref name) = field.name {
                if name == member_name {
                    return Some(field.ty.clone());
                }
            } else {
                // Anonymous struct/union — recurse into it
                if let Some(found) = self.find_member_type(&field.ty, member_name) {
                    return Some(found);
                }
            }
        }
        None
    }

    // ===================================================================
    // Enum Constant Evaluation
    // ===================================================================

    /// Look up an enum constant value by its interned symbol name.
    ///
    /// If the symbol is registered as an enum constant (via
    /// [`register_enum_value`](ConstantEvaluator::register_enum_value)),
    /// returns its value. Otherwise, emits a diagnostic error.
    fn evaluate_enum_constant(&mut self, name: Symbol, span: Span) -> Result<ConstValue, ()> {
        if let Some(&value) = self.enum_values.get(&name) {
            Ok(ConstValue::SignedInt(value))
        } else {
            self.diagnostics.emit_error(
                span,
                "identifier is not a compile-time constant (not an enum constant)",
            );
            Err(())
        }
    }

    // ===================================================================
    // Type Name Resolution (AST → CType)
    // ===================================================================

    /// Convert an AST [`TypeName`] to a [`CType`] for sizeof/alignof
    /// and cast evaluation.
    ///
    /// This is a simplified type resolver sufficient for constant expression
    /// evaluation. It handles basic type specifiers, pointers, and arrays.
    /// Complex cases (typedefs, struct members) would be resolved by the
    /// full semantic analyser.
    /// Extract `packed` and `aligned` attributes from a list of raw
    /// [`Attribute`] nodes on a struct/union specifier.
    fn extract_struct_attrs(&self, attrs: &[Attribute]) -> (bool, Option<usize>) {
        let mut packed = false;
        let mut aligned: Option<usize> = None;
        for attr in attrs {
            let name = self.resolve_symbol_name(attr.name);
            match name.as_str() {
                "packed" | "__packed__" => packed = true,
                "aligned" | "__aligned__" => {
                    if let Some(AttributeArg::Expression(expr)) = attr.args.first() {
                        if let Ok(ConstValue::SignedInt(v)) =
                            ConstantEvaluator::eval_simple_const_expr(expr)
                        {
                            aligned = Some(v as usize);
                        } else if let Ok(ConstValue::UnsignedInt(v)) =
                            ConstantEvaluator::eval_simple_const_expr(expr)
                        {
                            aligned = Some(v as usize);
                        }
                    }
                }
                _ => {}
            }
        }
        (packed, aligned)
    }

    /// A lightweight evaluator for attribute argument constant expressions.
    /// Does not require `&mut self` — only handles integer literals, parens,
    /// sizeof, and basic arithmetic needed for `aligned(N)`.
    fn eval_simple_const_expr(expr: &Expression) -> Result<ConstValue, ()> {
        match expr {
            Expression::IntegerLiteral { value, suffix, .. } => match suffix {
                IntegerSuffix::U | IntegerSuffix::UL | IntegerSuffix::ULL => {
                    Ok(ConstValue::UnsignedInt(*value))
                }
                _ => Ok(ConstValue::SignedInt(*value as i128)),
            },
            Expression::Parenthesized { inner, .. } => Self::eval_simple_const_expr(inner),
            _ => Err(()),
        }
    }

    /// Resolve struct/union members from their AST representation into
    /// a list of `StructField`s suitable for constructing a `CType::Struct`
    /// or `CType::Union`.
    fn resolve_struct_members(
        &mut self,
        members: &[StructMember],
        span: Span,
    ) -> Vec<crate::common::types::StructField> {
        let mut fields = Vec::new();
        for member in members {
            let member_base_type =
                match self.resolve_type_specifiers(&member.specifiers.type_specifiers, span) {
                    Ok(t) => t,
                    Err(()) => CType::Int, // fallback
                };

            if member.declarators.is_empty() {
                // Anonymous member (e.g., anonymous struct/union)
                fields.push(crate::common::types::StructField {
                    name: None,
                    ty: member_base_type,
                    bit_width: None,
                });
                continue;
            }

            for decl in &member.declarators {
                let field_type = if let Some(ref d) = decl.declarator {
                    self.apply_declarator_to_member_type(member_base_type.clone(), d)
                } else {
                    member_base_type.clone()
                };

                let field_name = decl
                    .declarator
                    .as_ref()
                    .and_then(|d| self.extract_declarator_name_str(d));

                let bit_width = decl
                    .bit_width
                    .as_ref()
                    .map(|bw| crate::frontend::sema::eval_bitfield_width(bw));

                fields.push(crate::common::types::StructField {
                    name: field_name,
                    ty: field_type,
                    bit_width,
                });
            }
        }
        fields
    }

    /// Apply a full declarator (pointer, array, function) to a base type
    /// for struct member type resolution.
    fn apply_declarator_to_member_type(&mut self, base: CType, decl: &Declarator) -> CType {
        let mut result = self.apply_direct_declarator_to_type(base, &decl.direct);
        if let Some(ref ptr) = decl.pointer {
            result = self.apply_pointer_chain(result, ptr);
        }
        result
    }

    /// Apply a direct declarator suffix (array/function) to a base type.
    fn apply_direct_declarator_to_type(&mut self, base: CType, direct: &DirectDeclarator) -> CType {
        match direct {
            DirectDeclarator::Identifier(_, _) => base,
            DirectDeclarator::Parenthesized(inner) => {
                self.apply_declarator_to_member_type(base, inner)
            }
            DirectDeclarator::Array {
                base: inner_base,
                size,
                ..
            } => {
                let elem = self.apply_direct_declarator_to_type(base, inner_base);
                let array_size = size.as_ref().and_then(|expr| {
                    match self.evaluate_integer_constant(expr, Span::default()) {
                        Ok(val) if val >= 0 => Some(val as usize),
                        _ => None,
                    }
                });
                CType::Array(Box::new(elem), array_size)
            }
            DirectDeclarator::Function {
                base: inner_base, ..
            } => {
                // Function pointer member — treat the member as a function type
                let ret = self.apply_direct_declarator_to_type(base, inner_base);
                CType::Pointer(
                    Box::new(CType::Function {
                        return_type: Box::new(ret),
                        params: vec![],
                        variadic: false,
                    }),
                    crate::common::types::TypeQualifiers::default(),
                )
            }
        }
    }

    /// Extract the name from a full Declarator as a string.
    fn extract_declarator_name_str(&self, decl: &Declarator) -> Option<String> {
        self.extract_direct_decl_name(&decl.direct)
    }

    /// Recursively extract the identifier name from a DirectDeclarator.
    fn extract_direct_decl_name(&self, direct: &DirectDeclarator) -> Option<String> {
        match direct {
            DirectDeclarator::Identifier(sym, _) => Some(self.resolve_symbol_name(*sym)),
            DirectDeclarator::Parenthesized(inner) => self.extract_declarator_name_str(inner),
            DirectDeclarator::Array { base, .. } => self.extract_direct_decl_name(base),
            DirectDeclarator::Function { base, .. } => self.extract_direct_decl_name(base),
        }
    }

    fn resolve_type_name(&mut self, type_name: &TypeName, span: Span) -> Result<CType, ()> {
        // Resolve the base type from specifiers
        let base_type =
            self.resolve_type_specifiers(&type_name.specifier_qualifiers.type_specifiers, span)?;

        // Apply abstract declarator (pointer/array) if present
        match &type_name.abstract_declarator {
            Some(decl) => self.apply_abstract_declarator(base_type, decl, span),
            None => Ok(base_type),
        }
    }

    /// Resolve a list of type specifiers into a [`CType`].
    ///
    /// Handles the C11 rules for combining multiple type specifiers:
    /// - `unsigned long long` = three specifiers combining into `ULongLong`
    /// - `signed char` = two specifiers combining into `SChar`
    /// - etc.
    fn resolve_type_specifiers(
        &mut self,
        specifiers: &[TypeSpecifier],
        span: Span,
    ) -> Result<CType, ()> {
        if specifiers.is_empty() {
            // Default to int if no type specifiers (C11 implicit int)
            return Ok(CType::Int);
        }

        // Track specifier flags
        let mut has_void = false;
        let mut has_char = false;
        let mut has_short = false;
        let mut has_int = false;
        let mut long_count: usize = 0;
        let mut has_float = false;
        let mut has_double = false;
        let mut has_signed = false;
        let mut has_unsigned = false;
        let mut has_bool = false;
        let mut has_complex = false;

        for spec in specifiers {
            match spec {
                TypeSpecifier::Void => has_void = true,
                TypeSpecifier::Char => has_char = true,
                TypeSpecifier::Short => has_short = true,
                TypeSpecifier::Int => has_int = true,
                TypeSpecifier::Long => long_count += 1,
                TypeSpecifier::Float => has_float = true,
                TypeSpecifier::Double => has_double = true,
                TypeSpecifier::Signed => has_signed = true,
                TypeSpecifier::Unsigned => has_unsigned = true,
                TypeSpecifier::Bool => has_bool = true,
                TypeSpecifier::Complex => has_complex = true,
                TypeSpecifier::Int128 => {
                    // __int128 type — resolve immediately based on signedness
                    if has_unsigned {
                        return Ok(CType::UInt128);
                    }
                    return Ok(CType::Int128);
                }
                // IEC 60559 floating-point types
                TypeSpecifier::Float128 => return Ok(CType::LongDouble),
                TypeSpecifier::Float64 => return Ok(CType::Double),
                TypeSpecifier::Float32 => return Ok(CType::Float),
                TypeSpecifier::Float16 => return Ok(CType::Float),

                // Struct/Union/Enum — look up the tag type from the
                // registered tag map if available, otherwise fall back to
                // an empty struct (forward declaration).  When the specifier
                // carries inline member definitions and/or attributes
                // (e.g. `sizeof(struct __attribute__((aligned(32))) { int a; })`),
                // we build the CType directly from the members/attrs so that
                // sizeof/alignof return the correct value even for anonymous
                // inline structs that were never registered in `tag_types`.
                TypeSpecifier::Struct(s) => {
                    if let Some(tag_sym) = s.tag {
                        if let Some(resolved) = self.tag_types.get(&tag_sym) {
                            return Ok(resolved.clone());
                        }
                    }
                    // Extract attributes from the struct specifier
                    let (packed, aligned) = self.extract_struct_attrs(&s.attributes);
                    // If we have inline member definitions, resolve them
                    if let Some(ref members) = s.members {
                        let fields = self.resolve_struct_members(members, span);
                        return Ok(CType::Struct {
                            name: s.tag.map(|sym| format!("struct_{}", sym.as_u32())),
                            fields,
                            packed,
                            aligned,
                        });
                    }
                    return Ok(CType::Struct {
                        name: s.tag.map(|sym| format!("struct_{}", sym.as_u32())),
                        fields: Vec::new(),
                        packed,
                        aligned,
                    });
                }
                TypeSpecifier::Union(u) => {
                    if let Some(tag_sym) = u.tag {
                        if let Some(resolved) = self.tag_types.get(&tag_sym) {
                            return Ok(resolved.clone());
                        }
                    }
                    let (packed, aligned) = self.extract_struct_attrs(&u.attributes);
                    if let Some(ref members) = u.members {
                        let fields = self.resolve_struct_members(members, span);
                        return Ok(CType::Union {
                            name: u.tag.map(|sym| format!("union_{}", sym.as_u32())),
                            fields,
                            packed,
                            aligned,
                        });
                    }
                    return Ok(CType::Union {
                        name: u.tag.map(|sym| format!("union_{}", sym.as_u32())),
                        fields: Vec::new(),
                        packed,
                        aligned,
                    });
                }
                TypeSpecifier::Enum(e) => {
                    // Look up enum tag in the registered tag types to get
                    // the correct underlying type (e.g., packed enums use
                    // unsigned char instead of int).
                    if let Some(tag_sym) = e.tag {
                        if let Some(resolved) = self.tag_types.get(&tag_sym) {
                            return Ok(resolved.clone());
                        }
                    }
                    return Ok(CType::Enum {
                        name: e.tag.map(|sym| format!("enum_{}", sym.as_u32())),
                        underlying_type: Box::new(CType::Int),
                    });
                }

                // Typedef name — resolve from the registered typedef map.
                TypeSpecifier::TypedefName(sym) => {
                    if let Some(resolved) = self.typedef_types.get(sym) {
                        return Ok(resolved.clone());
                    }
                    // Fallback: treat as int if the typedef wasn't registered.
                    return Ok(CType::Int);
                }

                // _Atomic(type) — recurse on inner type
                TypeSpecifier::Atomic(inner_tn) => {
                    let inner = self.resolve_type_name(inner_tn, span)?;
                    return Ok(CType::Atomic(Box::new(inner)));
                }

                // typeof(expr) / typeof(type) — resolve the argument type
                TypeSpecifier::Typeof(typeof_arg) => {
                    use crate::frontend::parser::ast::TypeofArg;
                    let resolved = match typeof_arg {
                        TypeofArg::TypeName(tn) => self.resolve_type_name(tn, span)?,
                        TypeofArg::Expression(expr) => self.infer_expr_type(expr, span)?,
                    };
                    return Ok(resolved);
                }
                // __auto_type — placeholder, actual type inferred from init
                TypeSpecifier::AutoType => {
                    return Ok(CType::Int);
                }
            }
        }

        // Combine specifier flags into a CType
        if has_void {
            return Ok(CType::Void);
        }
        if has_bool {
            return Ok(CType::Bool);
        }

        // Complex types
        if has_complex {
            let base = if has_float {
                CType::Float
            } else if long_count > 0 {
                CType::LongDouble
            } else {
                CType::Double
            };
            return Ok(CType::Complex(Box::new(base)));
        }

        // Floating-point types
        if has_float {
            return Ok(CType::Float);
        }
        if has_double {
            if long_count > 0 {
                return Ok(CType::LongDouble);
            }
            return Ok(CType::Double);
        }

        // Character types
        if has_char {
            if has_unsigned {
                return Ok(CType::UChar);
            } else if has_signed {
                return Ok(CType::SChar);
            } else {
                return Ok(CType::Char);
            }
        }

        // Short types
        if has_short {
            if has_unsigned {
                return Ok(CType::UShort);
            } else {
                return Ok(CType::Short);
            }
        }

        // Long long types
        if long_count >= 2 {
            if has_unsigned {
                return Ok(CType::ULongLong);
            } else {
                return Ok(CType::LongLong);
            }
        }

        // Long types
        if long_count == 1 {
            if has_unsigned {
                return Ok(CType::ULong);
            } else {
                return Ok(CType::Long);
            }
        }

        // Int types (including bare `signed`/`unsigned`)
        if has_unsigned {
            return Ok(CType::UInt);
        }
        if has_signed || has_int {
            return Ok(CType::Int);
        }

        // Default: int
        Ok(CType::Int)
    }

    /// Apply an abstract declarator to a base type, building pointer
    /// and array types.
    fn apply_abstract_declarator(
        &mut self,
        base: CType,
        decl: &AbstractDeclarator,
        span: Span,
    ) -> Result<CType, ()> {
        let mut result = base;

        // Apply direct abstract declarator first (array, function)
        if let Some(ref direct) = decl.direct {
            result = self.apply_direct_abstract_declarator(result, direct, span)?;
        }

        // Apply pointer chain
        if let Some(ref ptr) = decl.pointer {
            result = self.apply_pointer_chain(result, ptr);
        }

        Ok(result)
    }

    /// Apply a direct abstract declarator (array or function shape).
    fn apply_direct_abstract_declarator(
        &mut self,
        base: CType,
        direct: &DirectAbstractDeclarator,
        span: Span,
    ) -> Result<CType, ()> {
        match direct {
            DirectAbstractDeclarator::Parenthesized(inner) => {
                self.apply_abstract_declarator(base, inner, span)
            }
            DirectAbstractDeclarator::Array { size, .. } => {
                let array_size = match size {
                    Some(expr) => {
                        let val = self.evaluate_integer_constant(expr, span)?;
                        if val >= 0 {
                            Some(val as usize)
                        } else {
                            None
                        }
                    }
                    None => None,
                };
                Ok(CType::Array(Box::new(base), array_size))
            }
            DirectAbstractDeclarator::Function { .. } => {
                // Function type: sizeof(function) == 1 in GCC
                Ok(CType::Function {
                    return_type: Box::new(base),
                    params: Vec::new(),
                    variadic: false,
                })
            }
        }
    }

    /// Apply a pointer chain to a type.
    fn apply_pointer_chain(&self, base: CType, ptr: &Pointer) -> CType {
        let mut result = CType::Pointer(
            Box::new(base),
            crate::common::types::TypeQualifiers::default(),
        );

        // Recurse for multi-level pointers
        if let Some(ref inner) = ptr.inner {
            result = self.apply_pointer_chain(result, inner);
        }

        result
    }

    // ===================================================================
    // Expression Type Inference (for sizeof(expr))
    // ===================================================================

    /// Infer the C type of an expression for `sizeof(expr)` evaluation.
    ///
    /// This is a simplified type inference for common cases. The full
    /// semantic analyser provides comprehensive type inference, but for
    /// constant evaluation we handle the most common patterns.
    fn infer_expr_type(&mut self, expr: &Expression, span: Span) -> Result<CType, ()> {
        match expr {
            Expression::IntegerLiteral { suffix, .. } => Ok(self.integer_suffix_to_type(suffix)),
            Expression::FloatLiteral { suffix, .. } => match suffix {
                FloatSuffix::F => Ok(CType::Float),
                FloatSuffix::L => Ok(CType::LongDouble),
                FloatSuffix::None => Ok(CType::Double),
                FloatSuffix::I => Ok(CType::Complex(Box::new(CType::Double))),
                FloatSuffix::FI => Ok(CType::Complex(Box::new(CType::Float))),
                FloatSuffix::LI => Ok(CType::Complex(Box::new(CType::LongDouble))),
            },
            Expression::CharLiteral { .. } => Ok(CType::Int),
            Expression::StringLiteral { .. } => {
                // String literal: pointer to char
                Ok(CType::Pointer(
                    Box::new(CType::Char),
                    crate::common::types::TypeQualifiers {
                        is_const: true,
                        ..Default::default()
                    },
                ))
            }
            Expression::Parenthesized { inner, .. } => self.infer_expr_type(inner, span),
            Expression::Cast { type_name, .. } => self.resolve_type_name(type_name, span),
            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. } => {
                // sizeof/alignof return size_t
                if self.target.is_64bit() {
                    Ok(CType::ULong)
                } else {
                    Ok(CType::UInt)
                }
            }
            Expression::UnaryOp { op, operand, .. } => {
                match op {
                    UnaryOp::AddressOf => {
                        let inner_type = self.infer_expr_type(operand, span)?;
                        Ok(CType::Pointer(
                            Box::new(inner_type),
                            crate::common::types::TypeQualifiers::default(),
                        ))
                    }
                    UnaryOp::Deref => {
                        let inner_type = self.infer_expr_type(operand, span)?;
                        match inner_type {
                            CType::Pointer(pointee, _) => Ok(*pointee),
                            _ => Ok(CType::Int), // fallback
                        }
                    }
                    _ => self.infer_expr_type(operand, span),
                }
            }
            Expression::Identifier { name, .. } => {
                // Look up variable type from the registry (populated by sema
                // for block-scope _Static_assert evaluation).
                if let Some(ty) = self.variable_types.get(name) {
                    Ok(ty.clone())
                } else {
                    // Unknown identifier — fallback to int
                    Ok(CType::Int)
                }
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                use crate::frontend::parser::ast::BinaryOp;
                // Comparison operators always produce int (0 or 1)
                match op {
                    BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                    | BinaryOp::LogicalAnd
                    | BinaryOp::LogicalOr => Ok(CType::Int),
                    _ => {
                        // Binary ops: infer type from left operand
                        let lt = self.infer_expr_type(left, span)?;
                        let _rt = self.infer_expr_type(right, span)?;
                        Ok(lt)
                    }
                }
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                // C11 §6.5.15: ternary type rules for pointer operands.
                // When one arm is `void *` and the other is `T *`:
                //   - If the `void *` is a null pointer constant → result is `T *`
                //   - Otherwise → result is `void *`
                // This distinction is critical for `__is_constexpr()`:
                //   sizeof(*(8 ? (void *)((long)(x)*0l) : (int *)8))
                //   → If x is constant: (void*)0 IS null → type int* → sizeof=4
                //   → If x is runtime: (void*)expr NOT null const → void* → sizeof=1
                let then_actual = then_expr.as_deref().unwrap_or(condition.as_ref());
                let then_ty = self.infer_expr_type(then_actual, span)?;
                let else_ty = self.infer_expr_type(else_expr, span)?;

                let then_is_void_ptr = matches!(
                    &then_ty,
                    CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Void)
                );
                let else_is_void_ptr = matches!(
                    &else_ty,
                    CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Void)
                );

                if then_is_void_ptr && !else_is_void_ptr {
                    // then is void*, else is T* — only use T* if then is
                    // a null pointer constant
                    if self.is_null_pointer_constant(then_actual) {
                        Ok(else_ty)
                    } else {
                        Ok(then_ty) // keep void*
                    }
                } else if else_is_void_ptr && !then_is_void_ptr {
                    if self.is_null_pointer_constant(else_expr) {
                        Ok(then_ty)
                    } else {
                        Ok(else_ty) // keep void*
                    }
                } else {
                    Ok(then_ty)
                }
            }
            Expression::BuiltinCall { builtin, args, .. } => {
                use crate::frontend::parser::ast::BuiltinKind;
                match builtin {
                    BuiltinKind::ChooseExpr if args.len() >= 3 => {
                        // Try evaluating the condition; if it fails, infer from 3rd arg
                        let cond = self.evaluate_constant_expr(&args[0]);
                        let is_nonzero = match cond {
                            Ok(ConstValue::SignedInt(v)) => v != 0,
                            Ok(ConstValue::UnsignedInt(v)) => v != 0,
                            _ => false,
                        };
                        if is_nonzero {
                            self.infer_expr_type(&args[1], span)
                        } else {
                            self.infer_expr_type(&args[2], span)
                        }
                    }
                    BuiltinKind::TypesCompatibleP => Ok(CType::Int),
                    BuiltinKind::ConstantP => Ok(CType::Int),
                    BuiltinKind::Offsetof => {
                        if self.target.is_64bit() {
                            Ok(CType::ULong)
                        } else {
                            Ok(CType::UInt)
                        }
                    }
                    _ => Ok(CType::Int),
                }
            }
            Expression::MemberAccess { object, member, .. } => {
                // sizeof(obj.member) — infer type of object, then find
                // the named field within the struct/union type.
                let obj_type = self.infer_expr_type(object, span)?;
                let member_name = self.resolve_symbol_name(*member);
                self.find_member_type(&obj_type, &member_name)
                    .ok_or(())
                    .or(Ok(CType::Int))
            }
            Expression::PointerMemberAccess { object, member, .. } => {
                // sizeof(ptr->member) — infer type of pointer, dereference
                // to get the struct/union type, then find the named field.
                let obj_type = self.infer_expr_type(object, span)?;
                let pointee = match obj_type {
                    CType::Pointer(inner, _) => *inner,
                    _ => obj_type, // fallback: treat as the type itself
                };
                let member_name = self.resolve_symbol_name(*member);
                self.find_member_type(&pointee, &member_name)
                    .ok_or(())
                    .or(Ok(CType::Int))
            }
            Expression::ArraySubscript { base, .. } => {
                // sizeof(arr[i]) — the element type of the array/pointer
                let arr_type = self.infer_expr_type(base, span)?;
                match arr_type {
                    CType::Array(elem, _) => Ok(*elem),
                    CType::Pointer(elem, _) => Ok(*elem),
                    _ => Ok(CType::Int),
                }
            }
            // Statement expression: `({ T _a = v; _a; })` — the type is the
            // type of the last expression statement.  We must resolve identifiers
            // declared inside the compound block from their local declarations
            // rather than from the outer scope, which doesn't see them.
            Expression::StatementExpression { compound, .. } => {
                if let Some(BlockItem::Statement(Statement::Expression(Some(ref inner_expr)))) =
                    compound.items.last()
                {
                    // Build a local type map from compound declarations.
                    let mut local_types: FxHashMap<Symbol, CType> = FxHashMap::default();
                    for item in &compound.items {
                        if let BlockItem::Declaration(decl) = item {
                            if let Ok(base) =
                                self.resolve_type_specifiers(&decl.specifiers.type_specifiers, span)
                            {
                                for id in &decl.declarators {
                                    if let Some(sym) = Self::extract_decl_name(&id.declarator) {
                                        let full_ty = self
                                            .apply_declarator_type(base.clone(), &id.declarator);
                                        local_types.insert(sym, full_ty);
                                    }
                                }
                            }
                        }
                    }
                    return self.infer_expr_type_with_locals(inner_expr, span, &local_types);
                }
                Ok(CType::Int)
            }
            _ => {
                // Default: assume int for expressions we can't easily type
                // The full sema will provide accurate types
                Ok(CType::Int)
            }
        }
    }

    /// Extract the declared name (as a Symbol) from a declarator.
    /// Used by statement-expression typeof inference to build local type maps.
    fn extract_decl_name(declarator: &Declarator) -> Option<Symbol> {
        Self::extract_dd_name(&declarator.direct)
    }

    /// Extract the identifier from a direct declarator.
    fn extract_dd_name(dd: &DirectDeclarator) -> Option<Symbol> {
        match dd {
            DirectDeclarator::Identifier(sym, _) => Some(*sym),
            DirectDeclarator::Parenthesized(inner) => Self::extract_decl_name(inner),
            DirectDeclarator::Array { base, .. } => Self::extract_dd_name(base),
            DirectDeclarator::Function { base, .. } => Self::extract_dd_name(base),
        }
    }

    /// Apply pointer/array/function derivations from a declarator to a base type.
    /// Simplified version for statement-expression typeof inference — handles
    /// the common cases (plain identifiers, pointer declarators).
    fn apply_declarator_type(&self, base: CType, declarator: &Declarator) -> CType {
        let mut ty = base;
        // Apply pointer chain from the declarator
        if let Some(ref ptr) = declarator.pointer {
            ty = self.apply_pointer_chain(ty, ptr);
        }
        ty
    }

    /// Infer expression type with a set of locally-declared variable types,
    /// used when resolving `typeof` on statement expressions where the inner
    /// identifiers are declared inside the compound block.
    fn infer_expr_type_with_locals(
        &mut self,
        expr: &Expression,
        span: Span,
        locals: &FxHashMap<Symbol, CType>,
    ) -> Result<CType, ()> {
        match expr {
            Expression::Identifier { name, .. } => {
                // Check locals first, then fall back to variable_types
                if let Some(ty) = locals.get(name) {
                    return Ok(ty.clone());
                }
                if let Some(ty) = self.variable_types.get(name) {
                    return Ok(ty.clone());
                }
                Ok(CType::Int)
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                let then_actual = then_expr.as_deref().unwrap_or(condition.as_ref());
                let then_ty = self.infer_expr_type_with_locals(then_actual, span, locals)?;
                let else_ty = self.infer_expr_type_with_locals(else_expr, span, locals)?;

                // Apply the same void* null-pointer-constant logic as infer_expr_type.
                let then_is_void_ptr = matches!(
                    &then_ty,
                    CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Void)
                );
                let else_is_void_ptr = matches!(
                    &else_ty,
                    CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Void)
                );

                if then_is_void_ptr && !else_is_void_ptr {
                    if self.is_null_pointer_constant(then_actual) {
                        Ok(else_ty)
                    } else {
                        Ok(then_ty)
                    }
                } else if else_is_void_ptr && !then_is_void_ptr {
                    if self.is_null_pointer_constant(else_expr) {
                        Ok(then_ty)
                    } else {
                        Ok(else_ty)
                    }
                } else {
                    Ok(then_ty)
                }
            }
            Expression::Parenthesized { inner, .. } => {
                self.infer_expr_type_with_locals(inner, span, locals)
            }
            Expression::Comma { exprs, .. } => {
                if let Some(last) = exprs.last() {
                    self.infer_expr_type_with_locals(last, span, locals)
                } else {
                    Ok(CType::Void)
                }
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                use crate::frontend::parser::ast::BinaryOp;
                match op {
                    BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                    | BinaryOp::LogicalAnd
                    | BinaryOp::LogicalOr => Ok(CType::Int),
                    _ => {
                        let lt = self.infer_expr_type_with_locals(left, span, locals)?;
                        let _rt = self.infer_expr_type_with_locals(right, span, locals)?;
                        Ok(lt)
                    }
                }
            }
            Expression::UnaryOp { op, operand, .. } => match op {
                UnaryOp::AddressOf => {
                    let inner = self.infer_expr_type_with_locals(operand, span, locals)?;
                    Ok(CType::Pointer(
                        Box::new(inner),
                        crate::common::types::TypeQualifiers::default(),
                    ))
                }
                UnaryOp::Deref => {
                    let inner = self.infer_expr_type_with_locals(operand, span, locals)?;
                    match inner {
                        CType::Pointer(pointee, _) => Ok(*pointee),
                        _ => Ok(CType::Int),
                    }
                }
                _ => self.infer_expr_type_with_locals(operand, span, locals),
            },
            Expression::Cast { type_name, .. } => self.resolve_type_name(type_name, span),
            Expression::PostIncrement { operand, .. }
            | Expression::PostDecrement { operand, .. }
            | Expression::PreIncrement { operand, .. }
            | Expression::PreDecrement { operand, .. } => {
                self.infer_expr_type_with_locals(operand, span, locals)
            }
            // Nested statement expression — recurse with merged locals.
            Expression::StatementExpression { compound, .. } => {
                if let Some(BlockItem::Statement(Statement::Expression(Some(ref inner_expr)))) =
                    compound.items.last()
                {
                    let mut inner_locals = locals.clone();
                    for item in &compound.items {
                        if let BlockItem::Declaration(decl) = item {
                            if let Ok(base) =
                                self.resolve_type_specifiers(&decl.specifiers.type_specifiers, span)
                            {
                                for id in &decl.declarators {
                                    if let Some(sym) = Self::extract_decl_name(&id.declarator) {
                                        let full_ty = self
                                            .apply_declarator_type(base.clone(), &id.declarator);
                                        inner_locals.insert(sym, full_ty);
                                    }
                                }
                            }
                        }
                    }
                    return self.infer_expr_type_with_locals(inner_expr, span, &inner_locals);
                }
                Ok(CType::Int)
            }
            // For anything else, delegate to the standard infer_expr_type.
            _ => self.infer_expr_type(expr, span),
        }
    }

    /// Test whether an expression is a C11 *null pointer constant*.
    ///
    /// C11 §6.3.2.3p3: "An integer constant expression with the value 0,
    /// or such an expression cast to type `void *`, is called a *null
    /// pointer constant*."
    ///
    /// This is essential for determining the type of ternary expressions
    /// such as the `__is_constexpr()` kernel macro:
    ///   `8 ? (void *)(expr) : (int *)8`
    /// When `expr` evaluates to a constant 0, the `void *` arm is a null
    /// pointer constant and the result type is `int *`.  Otherwise the
    /// result type is `void *`.
    fn is_null_pointer_constant(&mut self, expr: &Expression) -> bool {
        match expr {
            Expression::Parenthesized { inner, .. } => self.is_null_pointer_constant(inner),
            Expression::Cast {
                operand, type_name, ..
            } => {
                // (void *)expr — check if inner is a null pointer constant
                let dummy = Span {
                    start: 0,
                    end: 0,
                    file_id: 0,
                };
                if let Ok(ty) = self.resolve_type_name(type_name, dummy) {
                    if matches!(&ty, CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Void))
                    {
                        return self.is_null_pointer_constant(operand);
                    }
                }
                // Non-void* cast — try to evaluate silently
                self.try_evaluate_as_zero(expr)
            }
            _ => {
                // Try to evaluate as an integer constant expression equal to 0
                self.try_evaluate_as_zero(expr)
            }
        }
    }

    /// Attempt to evaluate `expr` as a compile-time zero without emitting
    /// diagnostics. Returns `true` only when the expression is a valid
    /// integer constant expression whose value is exactly 0.
    fn try_evaluate_as_zero(&mut self, expr: &Expression) -> bool {
        self.diagnostics.begin_suppress();
        let result = self.evaluate_constant_expr(expr);
        self.diagnostics.end_suppress();
        matches!(
            result,
            Ok(ConstValue::SignedInt(0)) | Ok(ConstValue::UnsignedInt(0))
        )
    }

    /// Convert an integer suffix to the corresponding C type.
    fn integer_suffix_to_type(&self, suffix: &IntegerSuffix) -> CType {
        match suffix {
            IntegerSuffix::None => CType::Int,
            IntegerSuffix::U => CType::UInt,
            IntegerSuffix::L => CType::Long,
            IntegerSuffix::UL => CType::ULong,
            IntegerSuffix::LL => CType::LongLong,
            IntegerSuffix::ULL => CType::ULongLong,
        }
    }

    // ===================================================================
    // Usual Arithmetic Conversions
    // ===================================================================

    /// Perform the usual arithmetic conversions on two constant values.
    ///
    /// Returns the values as `i128` (for uniform computation) and a flag
    /// indicating whether the result should be treated as unsigned.
    ///
    /// C11 §6.3.1.8: If either operand is unsigned, the result is unsigned.
    /// If both are signed, the result is signed. This follows the standard
    /// integer promotion and usual arithmetic conversion rules.
    fn usual_arithmetic_conversions(
        &self,
        left: &ConstValue,
        right: &ConstValue,
    ) -> (i128, i128, bool) {
        let lv = match left {
            ConstValue::SignedInt(v) => *v,
            ConstValue::UnsignedInt(v) => *v as i128,
            ConstValue::Float(f) => *f as i128,
            _ => 0,
        };

        let rv = match right {
            ConstValue::SignedInt(v) => *v,
            ConstValue::UnsignedInt(v) => *v as i128,
            ConstValue::Float(f) => *f as i128,
            _ => 0,
        };

        // C11 §6.3.1.8: If either operand is unsigned, the result is unsigned.
        // If both operands are signed, the result is signed.
        let result_unsigned = if left.is_unsigned_val() || right.is_unsigned_val() {
            true
        } else {
            // Both signed — result is signed
            let _both_signed = left.is_signed() && right.is_signed();
            false
        };

        (lv, rv, result_unsigned)
    }

    /// Evaluate a [`StaticAssert`] AST node directly.
    ///
    /// Convenience method that extracts the condition and optional message
    /// from a `StaticAssert` struct and delegates to
    /// [`evaluate_static_assert`](ConstantEvaluator::evaluate_static_assert).
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_static_assert_node(&mut self, sa: &StaticAssert) -> Result<(), ()> {
        let message = sa.message.as_ref().map(|m| {
            // Convert raw bytes to string (best effort UTF-8 decoding)
            String::from_utf8_lossy(m).into_owned()
        });
        self.evaluate_static_assert(&sa.condition, message.as_deref(), sa.span)
    }

    /// Get the pointer width in bytes for the target architecture.
    ///
    /// Returns 4 for i686 (ILP32) and 8 for x86-64/AArch64/RISC-V 64 (LP64).
    /// Used for sizeof(pointer) and sizeof(size_t) constant evaluation.
    #[inline]
    pub fn target_pointer_width(&self) -> usize {
        self.target.pointer_width()
    }

    /// Get the size of `long` in bytes for the target architecture.
    ///
    /// Returns 4 for i686 (ILP32) and 8 for LP64 targets.
    #[inline]
    pub fn target_long_size(&self) -> usize {
        self.target.long_size()
    }

    /// Get the size of `long double` in bytes for the target architecture.
    #[inline]
    pub fn target_long_double_size(&self) -> usize {
        self.target.long_double_size()
    }

    /// Get the alignment of `long double` in bytes for the target architecture.
    #[inline]
    pub fn target_long_double_align(&self) -> usize {
        self.target.long_double_align()
    }

    /// Resolve through [`CType::Typedef`] chains to the underlying type.
    ///
    /// Recursively strips `Typedef` wrappers until a non-typedef type is
    /// reached. Also strips `CType::Atomic` wrappers for consistency.
    fn resolve_typedef(ty: &CType) -> &CType {
        match ty {
            CType::Typedef { underlying, .. } => Self::resolve_typedef(underlying),
            CType::Atomic(inner) => Self::resolve_typedef(inner),
            other => other,
        }
    }

    /// Create a span covering a range of bytes in a given source file.
    ///
    /// Convenience wrapper around [`Span::new`] for creating spans during
    /// constant evaluation.
    #[inline]
    fn make_span(file_id: u32, start: u32, end: u32) -> Span {
        Span::new(file_id, start, end)
    }

    /// Create a dummy span for compiler-generated constructs that have
    /// no corresponding source location.
    ///
    /// Convenience wrapper around [`Span::dummy`].
    #[inline]
    fn dummy_span() -> Span {
        Span::dummy()
    }

    /// Emit a diagnostic with explicit severity control.
    ///
    /// This is a convenience helper that directly constructs a [`Diagnostic`]
    /// with the specified [`Severity`], providing fine-grained control over
    /// whether a message is emitted as an error or warning.
    fn emit_diag(&mut self, severity: Severity, span: Span, msg: impl Into<String>) {
        let diag = match severity {
            Severity::Error => Diagnostic::error(span, msg),
            Severity::Warning => Diagnostic::warning(span, msg),
            Severity::Note => Diagnostic::note(span, msg),
        };
        self.diagnostics.emit(diag);
    }
}
