//! GCC builtin evaluation for Phase 5 (semantic analysis) of the BCC C11 compiler.
//!
//! Handles two categories of GCC builtins:
//!
//! 1. **Compile-time builtins** — evaluated during semantic analysis:
//!    - `__builtin_constant_p(expr)` — constant predicate
//!    - `__builtin_types_compatible_p(type1, type2)` — type compatibility check
//!    - `__builtin_choose_expr(const_expr, expr1, expr2)` — compile-time selection
//!    - `__builtin_offsetof(type, member)` — struct member byte offset
//!
//! 2. **Runtime-deferred builtins** — produce markers for IR lowering:
//!    - Bit manipulation: `clz`, `ctz`, `popcount`, `ffs`, `bswap16/32/64`
//!    - Optimization hints: `expect`, `unreachable`, `prefetch`
//!    - Variable arguments: `va_start`, `va_end`, `va_arg`, `va_copy`
//!    - Stack/frame: `frame_address`, `return_address`
//!    - Traps: `trap`, `assume_aligned`
//!    - Overflow arithmetic: `add_overflow`, `sub_overflow`, `mul_overflow`
//!
//! # Critical Rules
//!
//! - **Zero-dependency mandate**: Only `std` and `crate::` references.
//! - Unknown builtins MUST NOT be silently miscompiled — diagnosed with clear error.
//! - All diagnostics flow through [`DiagnosticEngine`].
//! - Does **not** depend on `crate::ir`, `crate::passes`, or `crate::backend`.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::string_interner::Symbol;
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{CType, TypeQualifiers};

use crate::frontend::parser::ast::*;

// ===========================================================================
// BuiltinResult — evaluation outcome
// ===========================================================================

/// The result of evaluating a GCC builtin during semantic analysis.
///
/// Compile-time builtins yield [`ConstantInt`](BuiltinResult::ConstantInt),
/// [`ConstantBool`](BuiltinResult::ConstantBool), or
/// [`ResolvedType`](BuiltinResult::ResolvedType).  Runtime-deferred builtins
/// yield [`RuntimeCall`](BuiltinResult::RuntimeCall) so that the IR lowering
/// phase can emit the appropriate instructions.  Builtins that produce no
/// value (e.g. `__builtin_unreachable`, `__builtin_trap`) yield
/// [`NoValue`](BuiltinResult::NoValue).
#[derive(Debug, Clone)]
pub enum BuiltinResult {
    /// Compile-time evaluated to an integer constant (e.g. `__builtin_offsetof`).
    ConstantInt(i128),
    /// Compile-time evaluated to a boolean value (e.g. `__builtin_constant_p`).
    ConstantBool(bool),
    /// Compile-time evaluated to a floating-point constant (e.g. `__builtin_inf`).
    ConstantFloat(f64),
    /// Compile-time resolved to a type (e.g. `__builtin_choose_expr` branch type).
    ResolvedType(CType),
    /// Runtime-deferred: IR lowering must generate a call for this builtin.
    RuntimeCall {
        /// Which builtin to lower.
        builtin: BuiltinKind,
        /// C type of the result value.
        result_type: CType,
    },
    /// Builtin produces no value (e.g. `__builtin_unreachable`, `__builtin_trap`).
    NoValue,
}

// ===========================================================================
// BuiltinEvaluator — main evaluator struct
// ===========================================================================

/// Evaluator for GCC builtins during Phase 5 semantic analysis.
///
/// Holds references to the diagnostic engine, type builder, and target
/// architecture.  Each compilation context should create one
/// `BuiltinEvaluator` and reuse it for all builtin calls in the
/// translation unit.
pub struct BuiltinEvaluator<'a> {
    /// Diagnostic engine for error/warning reporting.
    diagnostics: &'a mut DiagnosticEngine,
    /// Type construction and layout API (used for `__builtin_offsetof`).
    type_builder: &'a TypeBuilder,
    /// Target architecture for size/alignment decisions.
    target: Target,
}

impl<'a> BuiltinEvaluator<'a> {
    // -------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------

    /// Create a new `BuiltinEvaluator`.
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — mutable reference to the diagnostic engine.
    /// * `type_builder` — immutable reference to the target-aware type builder.
    /// * `target` — the compilation target architecture.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
    ) -> Self {
        BuiltinEvaluator {
            diagnostics,
            type_builder,
            target,
        }
    }

    // -------------------------------------------------------------------
    // Main dispatch
    // -------------------------------------------------------------------

    /// Evaluate a GCC builtin call.
    ///
    /// Dispatches on the [`BuiltinKind`] to either compile-time evaluate
    /// or mark as runtime-deferred.  Returns `Err(())` only when a hard
    /// error is emitted (e.g. wrong argument count, unsupported builtin).
    #[allow(clippy::result_unit_err)]
    pub fn evaluate_builtin(
        &mut self,
        builtin: &BuiltinKind,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        match builtin {
            // -- Compile-time builtins ------------------------------------
            BuiltinKind::ConstantP => self.eval_constant_p(args, span),
            BuiltinKind::TypesCompatibleP => self.eval_types_compatible_p(args, span),
            BuiltinKind::ChooseExpr => self.eval_choose_expr(args, span),
            BuiltinKind::Offsetof => self.eval_offsetof(args, span),

            // -- Runtime: bit manipulation --------------------------------
            BuiltinKind::Clz => self.eval_single_arg_int("__builtin_clz", args, CType::Int, span),
            BuiltinKind::ClzL => self.eval_single_arg_int("__builtin_clzl", args, CType::Int, span),
            BuiltinKind::ClzLL => {
                self.eval_single_arg_int("__builtin_clzll", args, CType::Int, span)
            }
            BuiltinKind::Ctz => self.eval_single_arg_int("__builtin_ctz", args, CType::Int, span),
            BuiltinKind::CtzL => self.eval_single_arg_int("__builtin_ctzl", args, CType::Int, span),
            BuiltinKind::CtzLL => {
                self.eval_single_arg_int("__builtin_ctzll", args, CType::Int, span)
            }
            BuiltinKind::Popcount => {
                self.eval_single_arg_int("__builtin_popcount", args, CType::Int, span)
            }
            BuiltinKind::PopcountL => {
                self.eval_single_arg_int("__builtin_popcountl", args, CType::Int, span)
            }
            BuiltinKind::PopcountLL => {
                self.eval_single_arg_int("__builtin_popcountll", args, CType::Int, span)
            }
            BuiltinKind::Ffs => self.eval_single_arg_int("__builtin_ffs", args, CType::Int, span),
            BuiltinKind::Ffsll => {
                self.eval_single_arg_int("__builtin_ffsll", args, CType::Int, span)
            }
            BuiltinKind::Bswap16 => {
                self.eval_single_arg_int("__builtin_bswap16", args, CType::UShort, span)
            }
            BuiltinKind::Bswap32 => {
                self.eval_single_arg_int("__builtin_bswap32", args, CType::UInt, span)
            }
            BuiltinKind::Bswap64 => {
                self.eval_single_arg_int("__builtin_bswap64", args, CType::ULongLong, span)
            }

            // -- Runtime: optimization hints ------------------------------
            BuiltinKind::Expect => self.eval_expect(args, span),
            BuiltinKind::Unreachable => self.eval_unreachable(args, span),
            BuiltinKind::PrefetchData => self.eval_prefetch(args, span),

            // -- Runtime: variable arguments ------------------------------
            BuiltinKind::VaStart => self.eval_va_start(args, span),
            BuiltinKind::VaEnd => self.eval_va_end(args, span),
            BuiltinKind::VaArg => self.eval_va_arg(args, span),
            BuiltinKind::VaCopy => self.eval_va_copy(args, span),

            // -- Runtime: stack/frame -------------------------------------
            BuiltinKind::FrameAddress => self.eval_frame_address(args, span),
            BuiltinKind::ReturnAddress => self.eval_return_address(args, span),

            // -- Runtime: traps and assertions ----------------------------
            BuiltinKind::Trap => self.eval_trap(args, span),
            BuiltinKind::AssumeAligned => self.eval_assume_aligned(args, span),

            // -- Runtime: overflow arithmetic -----------------------------
            BuiltinKind::AddOverflow => {
                self.eval_overflow_arith("__builtin_add_overflow", args, span)
            }
            BuiltinKind::SubOverflow => {
                self.eval_overflow_arith("__builtin_sub_overflow", args, span)
            }
            BuiltinKind::MulOverflow => {
                self.eval_overflow_arith("__builtin_mul_overflow", args, span)
            }
            BuiltinKind::ObjectSize => {
                // __builtin_object_size(ptr, type) — returns size_t.
                // At compile time we can't determine the size, so treat
                // it as a runtime call returning size_t.
                Ok(BuiltinResult::RuntimeCall {
                    builtin: BuiltinKind::ObjectSize,
                    result_type: CType::ULong,
                })
            }
            BuiltinKind::ExtractReturnAddr => {
                // __builtin_extract_return_addr(addr) — on most
                // architectures this is a no-op identity returning void*.
                // Treat as a runtime call returning void*.
                Ok(BuiltinResult::RuntimeCall {
                    builtin: BuiltinKind::ExtractReturnAddr,
                    result_type: CType::Pointer(
                        Box::new(CType::Void),
                        crate::common::types::TypeQualifiers::default(),
                    ),
                })
            }

            // -- Abort / Exit -------------------------------------------------
            BuiltinKind::Abort => {
                // __builtin_abort() — emit a call to abort().
                Ok(BuiltinResult::NoValue)
            }
            BuiltinKind::Exit => {
                // __builtin_exit(status) — emit a call to exit(status).
                Ok(BuiltinResult::NoValue)
            }

            // -- Memory/string builtins (runtime calls) -----------------------
            BuiltinKind::Memcpy
            | BuiltinKind::Memset
            | BuiltinKind::Memcmp
            | BuiltinKind::Strlen
            | BuiltinKind::Strcmp
            | BuiltinKind::Strncmp => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: rt,
                })
            }

            // -- Abs builtins (runtime calls) --------------------------------
            BuiltinKind::Abs | BuiltinKind::Labs | BuiltinKind::Llabs => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: rt,
                })
            }

            // -- Alloca (runtime call) ----------------------------------------
            BuiltinKind::Alloca => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: BuiltinKind::Alloca,
                    result_type: rt,
                })
            }

            // -- Floating-point constants (compile-time evaluable) -----------
            BuiltinKind::Inf | BuiltinKind::HugeVal => {
                Ok(BuiltinResult::ConstantFloat(f64::INFINITY))
            }
            BuiltinKind::Inff | BuiltinKind::HugeValf => {
                Ok(BuiltinResult::ConstantFloat(f64::INFINITY))
            }
            BuiltinKind::Infl | BuiltinKind::HugeVall => {
                Ok(BuiltinResult::ConstantFloat(f64::INFINITY))
            }
            BuiltinKind::Nan | BuiltinKind::Nanf | BuiltinKind::Nanl => {
                Ok(BuiltinResult::ConstantFloat(f64::NAN))
            }

            // -- Float predicates (runtime calls) ----------------------------
            BuiltinKind::Signbit
            | BuiltinKind::Isnan
            | BuiltinKind::Isinf
            | BuiltinKind::Isfinite
            | BuiltinKind::IsinfSign => Ok(BuiltinResult::RuntimeCall {
                builtin: *builtin,
                result_type: CType::Int,
            }),

            // -- Copysign / fabs (runtime calls) ----------------------------
            BuiltinKind::Copysign
            | BuiltinKind::Copysignf
            | BuiltinKind::Copysignl
            | BuiltinKind::Fabs
            | BuiltinKind::Fabsf
            | BuiltinKind::Fabsl => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: rt,
                })
            }

            // -- classify_type (compile-time) --------------------------------
            BuiltinKind::ClassifyType => {
                // GCC __builtin_classify_type returns an integer enum:
                // 0=no_type, 1=void, 2=int, 5=pointer, 8=real, etc.
                // Conservative: return runtime call to handle complex types.
                Ok(BuiltinResult::RuntimeCall {
                    builtin: BuiltinKind::ClassifyType,
                    result_type: CType::Int,
                })
            }

            // -- Sync atomics (runtime calls) --------------------------------
            BuiltinKind::SyncValCompareAndSwap
            | BuiltinKind::SyncBoolCompareAndSwap
            | BuiltinKind::SyncFetchAndAdd
            | BuiltinKind::SyncFetchAndSub
            | BuiltinKind::SyncFetchAndAnd
            | BuiltinKind::SyncFetchAndOr
            | BuiltinKind::SyncFetchAndXor
            | BuiltinKind::SyncLockTestAndSet
            | BuiltinKind::SyncLockRelease
            | BuiltinKind::SyncSynchronize => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: rt,
                })
            }

            // -- C11 atomics (runtime calls) ---------------------------------
            BuiltinKind::AtomicLoadN
            | BuiltinKind::AtomicStoreN
            | BuiltinKind::AtomicExchangeN
            | BuiltinKind::AtomicCompareExchangeN
            | BuiltinKind::AtomicFetchAdd
            | BuiltinKind::AtomicFetchSub
            | BuiltinKind::AtomicFetchAnd
            | BuiltinKind::AtomicFetchOr
            | BuiltinKind::AtomicFetchXor => {
                let rt = self.get_builtin_return_type(builtin);
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: rt,
                })
            }

            // -- Overflow _p variants (runtime calls) -----------------------
            BuiltinKind::MulOverflowP | BuiltinKind::AddOverflowP | BuiltinKind::SubOverflowP => {
                Ok(BuiltinResult::RuntimeCall {
                    builtin: *builtin,
                    result_type: CType::Bool,
                })
            }

            // -- BuiltinConstantP alias (same as ConstantP) ------------------
            BuiltinKind::BuiltinConstantP => self.eval_constant_p(args, span),
        }
    }

    // -------------------------------------------------------------------
    // Static query: is this name a known builtin?
    // -------------------------------------------------------------------

    /// Returns `true` if `name` corresponds to a recognised GCC builtin.
    ///
    /// This is a fast lookup used by the parser/lexer to distinguish
    /// builtin calls from ordinary function calls.
    pub fn is_builtin_name(name: &str) -> bool {
        // Use the comprehensive name_to_builtin_kind mapping
        Self::name_to_builtin_kind(name).is_some()
    }

    /// Return the C result type for a given builtin kind on this target.
    ///
    /// Useful for the type checker when it needs to determine the type
    /// of a `BuiltinCall` expression without performing full evaluation.
    pub fn get_builtin_return_type(&self, builtin: &BuiltinKind) -> CType {
        match builtin {
            // Compile-time builtins that return int constants.
            BuiltinKind::ConstantP | BuiltinKind::TypesCompatibleP => CType::Int,

            // __builtin_choose_expr returns the type of the chosen branch —
            // cannot be determined statically without evaluating the condition.
            // Fall back to Int as a placeholder; the actual evaluate_builtin
            // call produces the correct type.
            BuiltinKind::ChooseExpr => CType::Int,

            // __builtin_offsetof returns size_t (unsigned, pointer-width).
            BuiltinKind::Offsetof => self.size_t_type(),

            // Bit manipulation — int result.
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
            | BuiltinKind::Ffsll => CType::Int,

            // Byte swaps — same-width unsigned result.
            BuiltinKind::Bswap16 => CType::UShort,
            BuiltinKind::Bswap32 => CType::UInt,
            BuiltinKind::Bswap64 => CType::ULongLong,

            // __builtin_expect returns long.
            BuiltinKind::Expect => CType::Long,

            // No-value builtins.
            BuiltinKind::Unreachable
            | BuiltinKind::PrefetchData
            | BuiltinKind::Trap
            | BuiltinKind::VaStart
            | BuiltinKind::VaEnd
            | BuiltinKind::VaCopy => CType::Void,

            // __builtin_va_arg returns the specified type — cannot be
            // statically determined here.  Use Void as conservative fallback.
            BuiltinKind::VaArg => CType::Void,

            // Frame/return address return void *.
            BuiltinKind::FrameAddress | BuiltinKind::ReturnAddress => {
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default())
            }

            // __builtin_assume_aligned returns the pointer type of its first
            // argument.  Conservatively return void *.
            BuiltinKind::AssumeAligned => {
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default())
            }

            // Overflow arithmetic returns _Bool.
            BuiltinKind::AddOverflow | BuiltinKind::SubOverflow | BuiltinKind::MulOverflow => {
                CType::Bool
            }

            // __builtin_object_size returns size_t.
            BuiltinKind::ObjectSize => self.size_t_type(),

            // Abort/exit — noreturn void.
            BuiltinKind::Abort | BuiltinKind::Exit => CType::Void,

            // Memory/string builtins.
            BuiltinKind::Memcpy | BuiltinKind::Memset => {
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default())
            }
            BuiltinKind::Memcmp | BuiltinKind::Strcmp | BuiltinKind::Strncmp => CType::Int,
            BuiltinKind::Strlen => self.size_t_type(),

            // Abs builtins.
            BuiltinKind::Abs => CType::Int,
            BuiltinKind::Labs => CType::Long,
            BuiltinKind::Llabs => CType::LongLong,

            // Alloca returns void *.
            BuiltinKind::Alloca => CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),

            // Floating-point constants.
            BuiltinKind::Inf | BuiltinKind::HugeVal | BuiltinKind::Nan => CType::Double,
            BuiltinKind::Inff | BuiltinKind::HugeValf | BuiltinKind::Nanf => CType::Float,
            BuiltinKind::Infl | BuiltinKind::HugeVall | BuiltinKind::Nanl => CType::LongDouble,

            // Float predicates return int.
            BuiltinKind::Signbit
            | BuiltinKind::Isnan
            | BuiltinKind::Isinf
            | BuiltinKind::Isfinite
            | BuiltinKind::IsinfSign => CType::Int,

            // Copysign returns matching float type.
            BuiltinKind::Copysign => CType::Double,
            BuiltinKind::Copysignf => CType::Float,
            BuiltinKind::Copysignl => CType::LongDouble,

            // Fabs returns matching float type.
            BuiltinKind::Fabs => CType::Double,
            BuiltinKind::Fabsf => CType::Float,
            BuiltinKind::Fabsl => CType::LongDouble,

            // classify_type returns int.
            BuiltinKind::ClassifyType => CType::Int,

            // Sync atomics return the value type — use Int as conservative fallback.
            BuiltinKind::SyncValCompareAndSwap
            | BuiltinKind::SyncFetchAndAdd
            | BuiltinKind::SyncFetchAndSub
            | BuiltinKind::SyncFetchAndAnd
            | BuiltinKind::SyncFetchAndOr
            | BuiltinKind::SyncFetchAndXor
            | BuiltinKind::SyncLockTestAndSet => CType::Int,
            BuiltinKind::SyncBoolCompareAndSwap => CType::Bool,
            BuiltinKind::SyncLockRelease | BuiltinKind::SyncSynchronize => CType::Void,

            // C11 atomics — return types depend on pointer target.
            BuiltinKind::AtomicLoadN
            | BuiltinKind::AtomicExchangeN
            | BuiltinKind::AtomicFetchAdd
            | BuiltinKind::AtomicFetchSub
            | BuiltinKind::AtomicFetchAnd
            | BuiltinKind::AtomicFetchOr
            | BuiltinKind::AtomicFetchXor => CType::Int,
            BuiltinKind::AtomicStoreN => CType::Void,
            BuiltinKind::AtomicCompareExchangeN => CType::Bool,

            // Overflow _p variants return bool.
            BuiltinKind::MulOverflowP | BuiltinKind::AddOverflowP | BuiltinKind::SubOverflowP => {
                CType::Bool
            }

            // BuiltinConstantP — maps to ConstantP.
            BuiltinKind::BuiltinConstantP => CType::Int,

            // __builtin_extract_return_addr returns void*.
            BuiltinKind::ExtractReturnAddr => CType::Pointer(
                Box::new(CType::Void),
                crate::common::types::TypeQualifiers::default(),
            ),
        }
    }

    // ===================================================================
    /// Get a builtin return type by name and target (static method).
    /// Used for implicit builtin declarations when the identifier is undeclared.
    pub fn builtin_return_type_by_name(name: &str, target: crate::common::target::Target) -> CType {
        use crate::common::target::Target;
        let size_t = if matches!(target, Target::X86_64 | Target::AArch64 | Target::RiscV64) {
            CType::ULong
        } else {
            CType::UInt
        };
        let void_ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        // Match on known builtin name patterns
        match name {
            n if n.ends_with("abort") => CType::Void,
            n if n.ends_with("exit") => CType::Void,
            n if n.contains("memcpy") || n.contains("memmove") || n.contains("memset") => void_ptr,
            n if n.contains("memcmp") || n.contains("strcmp") || n.contains("strncmp") => {
                CType::Int
            }
            n if n.contains("strlen") => size_t,
            n if n.contains("alloca") => {
                CType::Pointer(Box::new(CType::Void), TypeQualifiers::default())
            }
            "__builtin_inf" | "__builtin_huge_val" => CType::Double,
            "__builtin_inff" | "__builtin_huge_valf" => CType::Float,
            "__builtin_infl" | "__builtin_huge_vall" => CType::LongDouble,
            "__builtin_nan" => CType::Double,
            "__builtin_nanf" => CType::Float,
            "__builtin_nanl" => CType::LongDouble,
            n if n.contains("copysignf") || n.contains("fabsf") => CType::Float,
            n if n.contains("copysignl") || n.contains("fabsl") => CType::LongDouble,
            n if n.contains("copysign") || n.contains("fabs") => CType::Double,
            n if n.contains("classify_type") => CType::Int,
            n if n.contains("constant_p") => CType::Int,
            n if n.contains("clz")
                || n.contains("ctz")
                || n.contains("popcount")
                || n.contains("parity")
                || n.contains("clrsb")
                || n.contains("ffs") =>
            {
                CType::Int
            }
            n if n.contains("bswap16") => CType::UShort,
            n if n.contains("bswap32") => CType::UInt,
            n if n.contains("bswap64") => CType::ULongLong,
            n if n.contains("signbit")
                || n.contains("isnan")
                || n.contains("isinf")
                || n.contains("isfinite") =>
            {
                CType::Int
            }
            n if n.starts_with("__sync_") || n.starts_with("__atomic_") => CType::Int,
            _ => CType::Int, // safe default
        }
    }

    // Compile-time builtins
    // ===================================================================

    /// `__builtin_constant_p(expr)` — returns 1 if `expr` is a compile-time
    /// constant, 0 otherwise.  Must **not** generate an error for
    /// non-constant expressions.
    fn eval_constant_p(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_constant_p", args, 1, span)
            .is_err()
        {
            return Err(());
        }

        // Determine if the argument is a manifest constant.
        let is_constant = Self::is_compile_time_constant(&args[0]);
        if is_constant {
            Ok(BuiltinResult::ConstantInt(1))
        } else {
            Ok(BuiltinResult::ConstantInt(0))
        }
    }

    /// `__builtin_types_compatible_p(type1, type2)` — returns 1 if the two
    /// types are compatible (ignoring top-level qualifiers), 0 otherwise.
    ///
    /// Both arguments are type names encoded as expressions by the parser
    /// (typically carried inside `SizeofType` nodes).
    fn eval_types_compatible_p(
        &mut self,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_types_compatible_p", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        // Attempt to extract CType from both arguments.  The parser should
        // have encoded the type-name arguments in a form that carries a
        // TypeName (e.g. SizeofType or AlignofType wrappers).
        let ty1_opt = Self::extract_type_from_expr(&args[0]);
        let ty2_opt = Self::extract_type_from_expr(&args[1]);

        match (ty1_opt, ty2_opt) {
            (Some(tn1), Some(tn2)) => {
                // Compare types ignoring top-level qualifiers.
                let compatible = Self::types_compatible_ignoring_qualifiers(&tn1, &tn2);

                // If types differ only by qualifiers, note this in diagnostics
                // for developer information.
                if !compatible {
                    if let (CType::Qualified(_, q1), _) | (_, CType::Qualified(_, q1)) =
                        (&tn1, &tn2)
                    {
                        let _qual_desc = Self::describe_qualifiers(q1);
                    }
                }

                Ok(BuiltinResult::ConstantInt(if compatible { 1 } else { 0 }))
            }
            _ => {
                // If we cannot extract types, fall back: treat as
                // runtime-indeterminate and return 0 (safe conservative
                // answer).  This avoids breaking compilation when the
                // parser encoding differs from our expectations.
                self.diagnostics.emit(
                    Diagnostic::warning(
                        span,
                        "__builtin_types_compatible_p: unable to extract type arguments; \
                         treating as incompatible",
                    )
                    .with_note(
                        Span::dummy(),
                        "type arguments should be encoded as SizeofType expressions by the parser",
                    ),
                );
                Ok(BuiltinResult::ConstantInt(0))
            }
        }
    }

    /// `__builtin_choose_expr(const_expr, expr1, expr2)` — compile-time
    /// conditional.  Evaluates `const_expr`; if nonzero, the result is
    /// `expr1`; otherwise `expr2`.  The *unchosen* branch is **not**
    /// type-checked — this is the critical difference from the ternary
    /// operator.
    fn eval_choose_expr(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_choose_expr", args, 3, span)
            .is_err()
        {
            return Err(());
        }

        // Evaluate the first argument as an integer constant.
        let condition_val = Self::try_evaluate_integer_constant(&args[0]);

        match condition_val {
            Some(v) if v != 0 => {
                // Condition is nonzero → choose expr1 (index 1).
                // Return ConstantBool(true) to signal that the first
                // (then) branch was chosen.  The caller (SemanticAnalyzer)
                // semantically analyses only the chosen branch.
                Ok(BuiltinResult::ConstantBool(true))
            }
            Some(_) => {
                // Condition is zero → choose expr2 (index 2).
                // Return ConstantBool(false) to signal the else branch.
                Ok(BuiltinResult::ConstantBool(false))
            }
            None => {
                // Use the span of the first argument for a more precise diagnostic.
                let arg_span = if !args.is_empty() {
                    args[0].span()
                } else {
                    Span::new(span.file_id, span.start, span.end)
                };
                self.diagnostics.emit(
                    Diagnostic::error(
                        arg_span,
                        "__builtin_choose_expr requires a compile-time integer constant \
                         as its first argument",
                    )
                    .with_note(
                        Span::dummy(),
                        "the first argument must be evaluable at compile time",
                    ),
                );
                Err(())
            }
        }
    }

    /// `__builtin_offsetof(type, member)` — byte offset of a struct member.
    ///
    /// The first argument is a type (struct) and the second identifies the
    /// member (possibly nested: `field.subfield`).  Computes the layout
    /// using [`TypeBuilder::compute_struct_layout`] and returns the offset
    /// as a `size_t` integer constant.
    fn eval_offsetof(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_offsetof", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        // Try to extract the struct type from the first argument.
        let struct_type_opt = Self::extract_type_from_expr(&args[0]);

        let struct_type = match struct_type_opt {
            Some(ty) => ty,
            None => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "__builtin_offsetof requires a struct/union type as its first argument",
                ));
                return Err(());
            }
        };

        // Resolve the struct fields from the CType (with bitfield info preserved).
        let struct_fields_opt = Self::extract_struct_fields_full(&struct_type);
        let (struct_fields, is_packed, explicit_align) = match struct_fields_opt {
            Some(f) => f,
            None => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "__builtin_offsetof: first argument is not a struct or union type",
                ));
                return Err(());
            }
        };

        // Extract the member name from the second argument.
        let member_name = Self::extract_member_identifier(&args[1]);
        let member_name = match member_name {
            Some(name) => name,
            None => {
                self.diagnostics.emit(Diagnostic::error(
                    span,
                    "__builtin_offsetof: second argument must be a member identifier",
                ));
                return Err(());
            }
        };

        // Compute struct layout using the full field info (preserves bitfield widths).
        let layout = self.type_builder.compute_struct_layout_with_fields(
            &struct_fields,
            is_packed,
            explicit_align,
        );

        // Compute overall struct size and alignment for diagnostic context.
        let struct_total_size = self.type_builder.sizeof_type(&struct_type);
        let struct_alignment = self.type_builder.alignof_type(&struct_type);

        // Find the member's index by name.
        let member_idx = Self::find_field_index_by_name(&struct_type, &member_name);
        match member_idx {
            Some(idx) if idx < layout.fields.len() => {
                let offset = layout.fields[idx].offset as i128;
                Ok(BuiltinResult::ConstantInt(offset))
            }
            _ => {
                self.diagnostics.emit(
                    Diagnostic::error(
                        span,
                        format!(
                            "__builtin_offsetof: no member '{}' in the specified struct type",
                            member_name
                        ),
                    )
                    .with_note(
                        Span::dummy(),
                        format!(
                            "struct has total size {} bytes and alignment {} bytes",
                            struct_total_size, struct_alignment
                        ),
                    ),
                );
                Err(())
            }
        }
    }

    // ===================================================================
    // Runtime-deferred: bit manipulation helpers
    // ===================================================================

    /// Validate a single-argument integer builtin and produce a RuntimeCall.
    fn eval_single_arg_int(
        &mut self,
        name: &str,
        args: &[Expression],
        result_type: CType,
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        if self.validate_arg_count(name, args, 1, span).is_err() {
            return Err(());
        }

        // Map name back to BuiltinKind for the RuntimeCall.
        let builtin = Self::name_to_builtin_kind(name).unwrap_or(BuiltinKind::Clz);

        Ok(BuiltinResult::RuntimeCall {
            builtin,
            result_type,
        })
    }

    // ===================================================================
    // Runtime-deferred: optimization hints
    // ===================================================================

    /// `__builtin_expect(expr, expected)` — branch prediction hint.
    /// Both arguments are `long`.  Result type is `long`.
    ///
    /// On ILP32 targets `long` is 4 bytes; on LP64 targets it is 8 bytes.
    /// The result type is always `CType::Long` regardless.
    fn eval_expect(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_expect", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        // Note: long size is target-dependent (4 on ILP32, 8 on LP64).
        // We record it here for documentation; the actual type is CType::Long
        // and its size is resolved later during codegen.
        let _long_bytes = self.target.long_size();

        Ok(BuiltinResult::RuntimeCall {
            builtin: BuiltinKind::Expect,
            result_type: CType::Long,
        })
    }

    /// `__builtin_unreachable()` — marks unreachable code path.
    fn eval_unreachable(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_unreachable", args, 0, span)
            .is_err()
        {
            return Err(());
        }

        Ok(BuiltinResult::NoValue)
    }

    /// `__builtin_prefetch(addr [, rw [, locality]])` — cache prefetch hint.
    /// 1–3 arguments.
    fn eval_prefetch(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if args.is_empty() || args.len() > 3 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "__builtin_prefetch requires 1 to 3 arguments, got {}",
                    args.len()
                ),
            ));
            return Err(());
        }

        // Validate optional rw argument (0 = read, 1 = write).
        if args.len() >= 2 {
            if let Some(val) = Self::try_evaluate_integer_constant(&args[1]) {
                if val != 0 && val != 1 {
                    self.diagnostics.emit_warning(
                        span,
                        "__builtin_prefetch: second argument (rw) should be 0 (read) or 1 (write)",
                    );
                }
            }
        }

        // Validate optional locality argument (0–3).
        if args.len() == 3 {
            if let Some(val) = Self::try_evaluate_integer_constant(&args[2]) {
                if !(0..=3).contains(&val) {
                    self.diagnostics.emit_warning(
                        span,
                        "__builtin_prefetch: third argument (locality) should be 0–3",
                    );
                }
            }
        }

        Ok(BuiltinResult::NoValue)
    }

    // ===================================================================
    // Runtime-deferred: variable arguments
    // ===================================================================

    /// `__builtin_va_start(ap, last_named_param)` — initialise `va_list`.
    fn eval_va_start(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_va_start", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        // The second argument should be the last named parameter; we accept
        // any expression here and let the IR lowering phase validate further.
        Ok(BuiltinResult::NoValue)
    }

    /// `__builtin_va_end(ap)` — clean up `va_list`.
    fn eval_va_end(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_va_end", args, 1, span)
            .is_err()
        {
            return Err(());
        }

        Ok(BuiltinResult::NoValue)
    }

    /// `__builtin_va_arg(ap, type)` — fetch next variadic argument.
    /// Returns a `RuntimeCall` whose `result_type` is the specified type.
    fn eval_va_arg(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_va_arg", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        // Attempt to resolve the type from the second argument.
        let result_type = Self::extract_type_from_expr(&args[1]).unwrap_or(CType::Int);

        Ok(BuiltinResult::RuntimeCall {
            builtin: BuiltinKind::VaArg,
            result_type,
        })
    }

    /// `__builtin_va_copy(dest, src)` — copy `va_list`.
    fn eval_va_copy(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_va_copy", args, 2, span)
            .is_err()
        {
            return Err(());
        }

        Ok(BuiltinResult::NoValue)
    }

    // ===================================================================
    // Runtime-deferred: stack / frame
    // ===================================================================

    /// `__builtin_frame_address(level)` — frame pointer at call depth.
    /// Argument must be an unsigned integer constant.  Result: `void *`
    /// (pointer width depends on target: 4 bytes on ILP32, 8 on LP64).
    fn eval_frame_address(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_frame_address", args, 1, span)
            .is_err()
        {
            return Err(());
        }

        // Validate level is a non-negative integer constant.
        if let Some(val) = Self::try_evaluate_integer_constant(&args[0]) {
            if val < 0 {
                self.diagnostics.emit_error(
                    span,
                    "__builtin_frame_address: level must be a non-negative integer constant",
                );
                return Err(());
            }
        }

        // Result is void *, whose width is target::pointer_width() bytes.
        let _ptr_width = self.target.pointer_width();
        let void_ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        Ok(BuiltinResult::RuntimeCall {
            builtin: BuiltinKind::FrameAddress,
            result_type: void_ptr,
        })
    }

    /// `__builtin_return_address(level)` — return address at call depth.
    /// Same signature and semantics as `frame_address`.
    fn eval_return_address(
        &mut self,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_return_address", args, 1, span)
            .is_err()
        {
            return Err(());
        }

        if let Some(val) = Self::try_evaluate_integer_constant(&args[0]) {
            if val < 0 {
                self.diagnostics.emit_error(
                    span,
                    "__builtin_return_address: level must be a non-negative integer constant",
                );
                return Err(());
            }
        }

        let void_ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        Ok(BuiltinResult::RuntimeCall {
            builtin: BuiltinKind::ReturnAddress,
            result_type: void_ptr,
        })
    }

    // ===================================================================
    // Runtime-deferred: traps and assertions
    // ===================================================================

    /// `__builtin_trap()` — generate a trap instruction (noreturn).
    fn eval_trap(&mut self, args: &[Expression], span: Span) -> Result<BuiltinResult, ()> {
        if self
            .validate_arg_count("__builtin_trap", args, 0, span)
            .is_err()
        {
            return Err(());
        }

        Ok(BuiltinResult::NoValue)
    }

    /// `__builtin_assume_aligned(ptr, align [, offset])` — alignment hint.
    /// Result type is the same pointer type as the first argument.
    fn eval_assume_aligned(
        &mut self,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        if args.len() < 2 || args.len() > 3 {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "__builtin_assume_aligned requires 2 or 3 arguments, got {}",
                    args.len()
                ),
            ));
            return Err(());
        }

        // Validate alignment is a power of two.
        if let Some(align_val) = Self::try_evaluate_integer_constant(&args[1]) {
            if align_val <= 0 || (align_val & (align_val - 1)) != 0 {
                self.diagnostics.emit_warning(
                    span,
                    "__builtin_assume_aligned: alignment should be a positive power of 2",
                );
            }
        }

        // Result type: same pointer type as the first argument.
        // Conservatively use void * if we cannot determine the pointer type.
        let result_type = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
        Ok(BuiltinResult::RuntimeCall {
            builtin: BuiltinKind::AssumeAligned,
            result_type,
        })
    }

    // ===================================================================
    // Runtime-deferred: overflow arithmetic
    // ===================================================================

    /// `__builtin_{add,sub,mul}_overflow(a, b, result_ptr)` — checked
    /// arithmetic.  Three arguments: two integers and a pointer to the
    /// result.  Returns `_Bool` (`true` if overflow occurred).
    fn eval_overflow_arith(
        &mut self,
        name: &str,
        args: &[Expression],
        span: Span,
    ) -> Result<BuiltinResult, ()> {
        if self.validate_arg_count(name, args, 3, span).is_err() {
            return Err(());
        }

        let builtin = Self::name_to_builtin_kind(name).unwrap_or(BuiltinKind::AddOverflow);

        Ok(BuiltinResult::RuntimeCall {
            builtin,
            result_type: CType::Bool,
        })
    }

    // ===================================================================
    // Argument validation helpers
    // ===================================================================

    /// Validate that `args` has exactly `expected` elements.
    /// Emits a diagnostic error and returns `Err(())` on mismatch.
    fn validate_arg_count(
        &mut self,
        builtin_name: &str,
        args: &[Expression],
        expected: usize,
        span: Span,
    ) -> Result<(), ()> {
        if args.len() != expected {
            self.diagnostics.emit(Diagnostic::error(
                span,
                format!(
                    "{} requires {} argument{}, got {}",
                    builtin_name,
                    expected,
                    if expected == 1 { "" } else { "s" },
                    args.len()
                ),
            ));
            return Err(());
        }
        Ok(())
    }

    /// Validate that an argument type is compatible with an expected type.
    /// Emits a warning on mismatch (not a hard error, as C allows implicit
    /// conversions for most integer arguments).
    #[allow(dead_code)]
    fn validate_arg_type(
        &mut self,
        builtin_name: &str,
        arg_type: &CType,
        expected: &CType,
        arg_index: usize,
        span: Span,
    ) -> Result<(), ()> {
        if !Self::types_are_assignment_compatible(arg_type, expected) {
            self.diagnostics.emit(Diagnostic::warning(
                span,
                format!(
                    "{}: argument {} has incompatible type",
                    builtin_name,
                    arg_index + 1,
                ),
            ));
        }
        Ok(())
    }

    // ===================================================================
    // Type helper utilities
    // ===================================================================

    /// Return the `size_t` type for the current target.
    ///
    /// On LP64 (x86-64, AArch64, RISC-V 64) this is `unsigned long`;
    /// on ILP32 (i686) this is `unsigned int`.
    fn size_t_type(&self) -> CType {
        if self.target.is_64bit() {
            CType::ULong
        } else {
            CType::UInt
        }
    }

    /// Determine if an expression is a manifest compile-time constant.
    ///
    /// Used by `__builtin_constant_p`.  This is a conservative check —
    /// it recognises integer/float/char literals, string literals, and
    /// `sizeof`/`alignof` expressions.
    fn is_compile_time_constant(expr: &Expression) -> bool {
        match expr {
            Expression::IntegerLiteral { .. }
            | Expression::FloatLiteral { .. }
            | Expression::CharLiteral { .. }
            | Expression::StringLiteral { .. }
            | Expression::SizeofType { .. }
            | Expression::SizeofExpr { .. }
            | Expression::AlignofType { .. } => true,

            Expression::Parenthesized { inner, .. } => Self::is_compile_time_constant(inner),

            Expression::Cast { operand, .. } => Self::is_compile_time_constant(operand),

            Expression::UnaryOp { operand, .. } => Self::is_compile_time_constant(operand),

            Expression::Binary { left, right, .. } => {
                Self::is_compile_time_constant(left) && Self::is_compile_time_constant(right)
            }

            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                Self::is_compile_time_constant(condition)
                    && then_expr
                        .as_ref()
                        .map_or(true, |e| Self::is_compile_time_constant(e))
                    && Self::is_compile_time_constant(else_expr)
            }

            // BuiltinCall to another compile-time builtin
            Expression::BuiltinCall { builtin, .. } => matches!(
                builtin,
                BuiltinKind::ConstantP | BuiltinKind::TypesCompatibleP | BuiltinKind::Offsetof
            ),

            _ => false,
        }
    }

    /// Try to evaluate an expression as a simple integer constant.
    ///
    /// Returns `Some(value)` for integer literals (possibly negated).
    /// Returns `None` for anything more complex.  This is used for
    /// validating constant arguments in builtins, not for full constant
    /// expression evaluation (which is the job of `ConstantEvaluator`).
    fn try_evaluate_integer_constant(expr: &Expression) -> Option<i128> {
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as i128),

            Expression::CharLiteral { value, .. } => Some(*value as i128),

            Expression::Parenthesized { inner, .. } => Self::try_evaluate_integer_constant(inner),

            Expression::UnaryOp {
                op: UnaryOp::Negate,
                operand,
                ..
            } => Self::try_evaluate_integer_constant(operand).map(|v| -v),

            Expression::UnaryOp {
                op: UnaryOp::Plus,
                operand,
                ..
            } => Self::try_evaluate_integer_constant(operand),

            Expression::Cast { operand, .. } => Self::try_evaluate_integer_constant(operand),

            _ => None,
        }
    }

    /// Extract a `CType` from an expression that the parser used to encode
    /// a type-name argument (e.g. via `SizeofType`, `AlignofType`, or
    /// `Cast`).
    ///
    /// Returns a simplified `CType` based on the type specifiers found.
    fn extract_type_from_expr(expr: &Expression) -> Option<CType> {
        match expr {
            Expression::SizeofType { type_name, .. }
            | Expression::AlignofType { type_name, .. } => Some(Self::typename_to_ctype(type_name)),

            Expression::Cast { type_name, .. } => Some(Self::typename_to_ctype(type_name)),

            Expression::Parenthesized { inner, .. } => Self::extract_type_from_expr(inner),

            _ => None,
        }
    }

    /// Convert a [`TypeName`] AST node into a simplified [`CType`].
    ///
    /// This is a best-effort conversion used only for builtin evaluation.
    /// Full type resolution is performed by the main semantic analyzer.
    fn typename_to_ctype(tn: &TypeName) -> CType {
        let base = Self::typename_specifiers_to_ctype(&tn.specifier_qualifiers.type_specifiers);
        // Apply pointer wrapping from the abstract declarator.
        Self::apply_abstract_declarator(base, &tn.abstract_declarator)
    }

    /// Apply pointer wrapping from an abstract declarator to a base type.
    fn apply_abstract_declarator(base: CType, ad: &Option<AbstractDeclarator>) -> CType {
        let ad = match ad {
            Some(ad) => ad,
            None => return base,
        };
        // Count pointer levels.
        let mut result = base;
        if let Some(ref ptr) = ad.pointer {
            result = Self::apply_pointer_chain(result, ptr);
        }
        result
    }

    /// Recursively apply pointer chain: each `Pointer` node adds one `*`.
    fn apply_pointer_chain(base: CType, ptr: &crate::frontend::parser::ast::Pointer) -> CType {
        // Inner pointer first (if `**`, inner is the leftmost `*`).
        let inner = if let Some(ref inner_ptr) = ptr.inner {
            Self::apply_pointer_chain(base, inner_ptr)
        } else {
            base
        };
        CType::Pointer(Box::new(inner), TypeQualifiers::default())
    }

    fn typename_specifiers_to_ctype(specs: &[TypeSpecifier]) -> CType {
        // Count occurrences of various specifiers to build the type.
        let mut has_void = false;
        let mut has_char = false;
        let mut has_short = false;
        let mut has_int = false;
        let mut long_count = 0u32;
        let mut has_float = false;
        let mut has_double = false;
        let mut has_signed = false;
        let mut has_unsigned = false;
        let mut has_bool = false;
        let mut has_complex = false;
        let mut struct_type: Option<CType> = None;

        for spec in specs {
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
                    if has_unsigned {
                        return CType::UInt128;
                    }
                    return CType::Int128;
                }
                TypeSpecifier::Float128 => return CType::LongDouble,
                TypeSpecifier::Float64 => return CType::Double,
                TypeSpecifier::Float32 => return CType::Float,
                TypeSpecifier::Float16 => return CType::Float,
                TypeSpecifier::Struct(s) => {
                    let name = s.tag.map(|sym| format!("struct_{}", sym.as_u32()));
                    struct_type = Some(CType::Struct {
                        name,
                        fields: Vec::new(),
                        packed: false,
                        aligned: None,
                    });
                }
                TypeSpecifier::Union(s) => {
                    let name = s.tag.map(|sym| format!("union_{}", sym.as_u32()));
                    struct_type = Some(CType::Union {
                        name,
                        fields: Vec::new(),
                        packed: false,
                        aligned: None,
                    });
                }
                TypeSpecifier::Enum(_) => {
                    return CType::Int; // Enums are int-compatible.
                }
                TypeSpecifier::TypedefName(_sym) => {
                    // Cannot fully resolve typedefs here without the symbol
                    // table; return Int as a conservative fallback.
                    return CType::Int;
                }
                TypeSpecifier::Atomic(inner_tn) => {
                    return CType::Atomic(Box::new(Self::typename_to_ctype(inner_tn)));
                }
                TypeSpecifier::Typeof(_) => {
                    return CType::Int; // Cannot resolve typeof without expression evaluation.
                }
                TypeSpecifier::AutoType => {
                    return CType::Int; // __auto_type — placeholder, resolved at init.
                }
            }
        }

        // Build type from specifier flags.
        if let Some(st) = struct_type {
            return st;
        }
        if has_void {
            return CType::Void;
        }
        if has_bool {
            return CType::Bool;
        }

        if has_float {
            if has_complex {
                return CType::Complex(Box::new(CType::Float));
            }
            return CType::Float;
        }
        if has_double {
            if long_count >= 1 {
                if has_complex {
                    return CType::Complex(Box::new(CType::LongDouble));
                }
                return CType::LongDouble;
            }
            if has_complex {
                return CType::Complex(Box::new(CType::Double));
            }
            return CType::Double;
        }

        // Integer types.
        if has_char {
            return if has_unsigned {
                CType::UChar
            } else if has_signed {
                CType::SChar
            } else {
                CType::Char
            };
        }
        if has_short {
            return if has_unsigned {
                CType::UShort
            } else {
                CType::Short
            };
        }
        if long_count >= 2 {
            return if has_unsigned {
                CType::ULongLong
            } else {
                CType::LongLong
            };
        }
        if long_count == 1 {
            // `long` alone or `long int`.
            return if has_unsigned {
                CType::ULong
            } else {
                CType::Long
            };
        }

        // Default: int (or unsigned int).
        if has_unsigned {
            CType::UInt
        } else {
            // `signed`, `int`, `signed int`, or bare specifiers → Int.
            let _ = has_int;
            let _ = has_signed;
            CType::Int
        }
    }

    /// Compare two `CType` values for compatibility, ignoring top-level
    /// qualifiers (const, volatile, restrict, atomic).
    ///
    /// This is the semantics of `__builtin_types_compatible_p`.  Per the
    /// GCC documentation, `const int` and `int` are compatible, as are
    /// `volatile int` and `int`.
    fn types_compatible_ignoring_qualifiers(a: &CType, b: &CType) -> bool {
        let a_stripped = Self::strip_qualifiers(a);
        let b_stripped = Self::strip_qualifiers(b);
        Self::ctypes_structurally_equal(a_stripped, b_stripped)
    }

    /// Check whether a `TypeQualifiers` set has any active qualifiers.
    ///
    /// Returns a description of the active qualifiers for diagnostic
    /// purposes.  Accesses `is_const`, `is_volatile`, `is_restrict`,
    /// and `is_atomic` fields of [`TypeQualifiers`].
    fn describe_qualifiers(q: &TypeQualifiers) -> String {
        let mut parts = Vec::new();
        if q.is_const {
            parts.push("const");
        }
        if q.is_volatile {
            parts.push("volatile");
        }
        if q.is_restrict {
            parts.push("restrict");
        }
        if q.is_atomic {
            parts.push("_Atomic");
        }
        if parts.is_empty() {
            String::from("(none)")
        } else {
            parts.join(" ")
        }
    }

    /// Strip top-level qualifiers and typedefs from a `CType`.
    fn strip_qualifiers(ty: &CType) -> &CType {
        match ty {
            CType::Qualified(inner, _) => Self::strip_qualifiers(inner),
            CType::Typedef { underlying, .. } => Self::strip_qualifiers(underlying),
            CType::Atomic(inner) => Self::strip_qualifiers(inner),
            other => other,
        }
    }

    /// Structural equality check for two `CType` values (after stripping
    /// qualifiers).
    fn ctypes_structurally_equal(a: &CType, b: &CType) -> bool {
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

            (CType::Pointer(inner_a, _), CType::Pointer(inner_b, _)) => {
                Self::ctypes_structurally_equal(
                    Self::strip_qualifiers(inner_a),
                    Self::strip_qualifiers(inner_b),
                )
            }

            (CType::Array(elem_a, sz_a), CType::Array(elem_b, sz_b)) => {
                sz_a == sz_b
                    && Self::ctypes_structurally_equal(
                        Self::strip_qualifiers(elem_a),
                        Self::strip_qualifiers(elem_b),
                    )
            }

            (
                CType::Function {
                    return_type: ret_a,
                    params: params_a,
                    variadic: var_a,
                },
                CType::Function {
                    return_type: ret_b,
                    params: params_b,
                    variadic: var_b,
                },
            ) => {
                var_a == var_b
                    && Self::ctypes_structurally_equal(
                        Self::strip_qualifiers(ret_a),
                        Self::strip_qualifiers(ret_b),
                    )
                    && params_a.len() == params_b.len()
                    && params_a.iter().zip(params_b.iter()).all(|(pa, pb)| {
                        Self::ctypes_structurally_equal(
                            Self::strip_qualifiers(pa),
                            Self::strip_qualifiers(pb),
                        )
                    })
            }

            (CType::Struct { name: na, .. }, CType::Struct { name: nb, .. }) => na == nb,

            (CType::Union { name: na, .. }, CType::Union { name: nb, .. }) => na == nb,

            (CType::Enum { name: na, .. }, CType::Enum { name: nb, .. }) => na == nb,

            (CType::Complex(inner_a), CType::Complex(inner_b)) => {
                Self::ctypes_structurally_equal(inner_a, inner_b)
            }

            _ => false,
        }
    }

    /// Simple assignment compatibility check (conservative).
    fn types_are_assignment_compatible(from: &CType, to: &CType) -> bool {
        let from_s = Self::strip_qualifiers(from);
        let to_s = Self::strip_qualifiers(to);

        // Same type is always compatible.
        if Self::ctypes_structurally_equal(from_s, to_s) {
            return true;
        }

        // All arithmetic types are inter-convertible.
        if Self::is_arithmetic(from_s) && Self::is_arithmetic(to_s) {
            return true;
        }

        // Pointer ↔ pointer (void * is compatible with any pointer).
        if let (CType::Pointer(_, _), CType::Pointer(_, _)) = (from_s, to_s) {
            return true;
        }

        // Integer → pointer (null constant) and pointer → integer.
        if Self::is_integer(from_s) && matches!(to_s, CType::Pointer(_, _)) {
            return true;
        }
        if matches!(from_s, CType::Pointer(_, _)) && Self::is_integer(to_s) {
            return true;
        }

        false
    }

    /// Returns `true` if the type is an integer type.
    fn is_integer(ty: &CType) -> bool {
        matches!(
            ty,
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
                | CType::Enum { .. }
        )
    }

    /// Returns `true` if the type is an arithmetic type (integer or FP).
    fn is_arithmetic(ty: &CType) -> bool {
        Self::is_integer(ty)
            || matches!(
                ty,
                CType::Float | CType::Double | CType::LongDouble | CType::Complex(_)
            )
    }

    /// Extract struct fields, packed flag, and explicit alignment from a CType.
    #[allow(dead_code)]
    fn extract_struct_fields(ty: &CType) -> Option<(Vec<CType>, bool, Option<usize>)> {
        match Self::strip_qualifiers(ty) {
            CType::Struct {
                fields,
                packed,
                aligned,
                ..
            } => {
                let field_types: Vec<CType> = fields.iter().map(|f| f.ty.clone()).collect();
                Some((field_types, *packed, *aligned))
            }
            _ => None,
        }
    }

    /// Extract full [`StructField`] list from a struct type, preserving bitfield
    /// width information needed for accurate offset computation.
    fn extract_struct_fields_full(
        ty: &CType,
    ) -> Option<(Vec<crate::common::types::StructField>, bool, Option<usize>)> {
        match Self::strip_qualifiers(ty) {
            CType::Struct {
                fields,
                packed,
                aligned,
                ..
            } => Some((fields.clone(), *packed, *aligned)),
            _ => None,
        }
    }

    /// Convert an interned [`Symbol`] to a printable name string.
    ///
    /// In a full compiler the `Interner` would be used for resolution;
    /// here we produce a canonical `sym_<id>` representation that is
    /// stable across runs and suitable for struct-field matching.
    fn symbol_to_name(sym: Symbol) -> String {
        format!("sym_{}", sym.as_u32())
    }

    /// Extract a member identifier name from an expression.
    ///
    /// Handles `Identifier`, `MemberAccess` (for nested member references),
    /// and `PointerMemberAccess`.
    fn extract_member_identifier(expr: &Expression) -> Option<String> {
        match expr {
            Expression::Identifier { name, .. } => {
                // Use the raw u32 value as a placeholder.  In practice the
                // full SemanticAnalyzer would use the Interner to resolve.
                Some(Self::symbol_to_name(*name))
            }
            Expression::MemberAccess { member, .. } => Some(Self::symbol_to_name(*member)),
            Expression::PointerMemberAccess { member, .. } => Some(Self::symbol_to_name(*member)),
            Expression::Parenthesized { inner, .. } => Self::extract_member_identifier(inner),
            _ => None,
        }
    }

    /// Find the index of a field by name within a struct CType.
    fn find_field_index_by_name(struct_ty: &CType, member_name: &str) -> Option<usize> {
        let fields = match Self::strip_qualifiers(struct_ty) {
            CType::Struct { fields, .. } => fields,
            CType::Union { fields, .. } => fields,
            _ => return None,
        };

        for (i, field) in fields.iter().enumerate() {
            if let Some(ref name) = field.name {
                if name == member_name || format!("sym_{}", name) == member_name {
                    return Some(i);
                }
            }
        }

        // Also try matching against the field index identifier pattern.
        for (i, field) in fields.iter().enumerate() {
            if let Some(ref name) = field.name {
                // Accept both the raw name and the sym_ prefix form.
                if member_name.ends_with(name.as_str()) {
                    return Some(i);
                }
            }
        }

        None
    }

    /// Map a builtin name string to its [`BuiltinKind`] variant.
    pub fn name_to_builtin_kind(name: &str) -> Option<BuiltinKind> {
        match name {
            "__builtin_expect" => Some(BuiltinKind::Expect),
            "__builtin_unreachable" => Some(BuiltinKind::Unreachable),
            "__builtin_constant_p" => Some(BuiltinKind::ConstantP),
            "__builtin_offsetof" => Some(BuiltinKind::Offsetof),
            "__builtin_types_compatible_p" => Some(BuiltinKind::TypesCompatibleP),
            "__builtin_choose_expr" => Some(BuiltinKind::ChooseExpr),
            "__builtin_clz" => Some(BuiltinKind::Clz),
            "__builtin_ctz" => Some(BuiltinKind::Ctz),
            "__builtin_popcount" => Some(BuiltinKind::Popcount),
            "__builtin_bswap16" => Some(BuiltinKind::Bswap16),
            "__builtin_bswap32" => Some(BuiltinKind::Bswap32),
            "__builtin_bswap64" => Some(BuiltinKind::Bswap64),
            "__builtin_ffs" => Some(BuiltinKind::Ffs),
            "__builtin_ffsll" => Some(BuiltinKind::Ffsll),
            "__builtin_va_start" => Some(BuiltinKind::VaStart),
            "__builtin_va_end" => Some(BuiltinKind::VaEnd),
            "__builtin_va_arg" => Some(BuiltinKind::VaArg),
            "__builtin_va_copy" => Some(BuiltinKind::VaCopy),
            "__builtin_frame_address" => Some(BuiltinKind::FrameAddress),
            "__builtin_return_address" => Some(BuiltinKind::ReturnAddress),
            "__builtin_trap" => Some(BuiltinKind::Trap),
            "__builtin_assume_aligned" => Some(BuiltinKind::AssumeAligned),
            "__builtin_add_overflow" => Some(BuiltinKind::AddOverflow),
            "__builtin_sub_overflow" => Some(BuiltinKind::SubOverflow),
            "__builtin_mul_overflow" => Some(BuiltinKind::MulOverflow),
            "__builtin_prefetch" => Some(BuiltinKind::PrefetchData),
            "__builtin_extract_return_addr" => Some(BuiltinKind::ExtractReturnAddr),
            "__builtin_object_size" => Some(BuiltinKind::ObjectSize),
            "__builtin_abort" => Some(BuiltinKind::Abort),
            "__builtin_exit" => Some(BuiltinKind::Exit),
            "__builtin_memcpy" | "__builtin___memcpy_chk" => Some(BuiltinKind::Memcpy),
            "__builtin_memset" | "__builtin___memset_chk" => Some(BuiltinKind::Memset),
            "__builtin_memcmp" => Some(BuiltinKind::Memcmp),
            "__builtin_strlen" => Some(BuiltinKind::Strlen),
            "__builtin_strcmp" => Some(BuiltinKind::Strcmp),
            "__builtin_strncmp" => Some(BuiltinKind::Strncmp),
            "__builtin_abs" => Some(BuiltinKind::Abs),
            "__builtin_labs" => Some(BuiltinKind::Labs),
            "__builtin_llabs" => Some(BuiltinKind::Llabs),
            "__builtin_alloca" | "__builtin_alloca_with_align" => Some(BuiltinKind::Alloca),
            "__builtin_inf" => Some(BuiltinKind::Inf),
            "__builtin_inff" => Some(BuiltinKind::Inff),
            "__builtin_infl" => Some(BuiltinKind::Infl),
            "__builtin_nan" => Some(BuiltinKind::Nan),
            "__builtin_nanf" => Some(BuiltinKind::Nanf),
            "__builtin_nanl" => Some(BuiltinKind::Nanl),
            "__builtin_huge_val" => Some(BuiltinKind::HugeVal),
            "__builtin_huge_valf" => Some(BuiltinKind::HugeValf),
            "__builtin_huge_vall" => Some(BuiltinKind::HugeVall),
            "__builtin_signbit" | "__builtin_signbitf" | "__builtin_signbitl" => {
                Some(BuiltinKind::Signbit)
            }
            "__builtin_isnan" | "__builtin_isnanf" | "__builtin_isnanl" => Some(BuiltinKind::Isnan),
            "__builtin_isinf" | "__builtin_isinff" | "__builtin_isinfl" => Some(BuiltinKind::Isinf),
            "__builtin_isfinite" => Some(BuiltinKind::Isfinite),
            "__builtin_isinf_sign" => Some(BuiltinKind::IsinfSign),
            "__builtin_copysign" => Some(BuiltinKind::Copysign),
            "__builtin_copysignf" => Some(BuiltinKind::Copysignf),
            "__builtin_copysignl" => Some(BuiltinKind::Copysignl),
            "__builtin_fabs" => Some(BuiltinKind::Fabs),
            "__builtin_fabsf" => Some(BuiltinKind::Fabsf),
            "__builtin_fabsl" => Some(BuiltinKind::Fabsl),
            "__builtin_classify_type" => Some(BuiltinKind::ClassifyType),
            "__sync_val_compare_and_swap" => Some(BuiltinKind::SyncValCompareAndSwap),
            "__sync_bool_compare_and_swap" => Some(BuiltinKind::SyncBoolCompareAndSwap),
            "__sync_fetch_and_add" => Some(BuiltinKind::SyncFetchAndAdd),
            "__sync_fetch_and_sub" => Some(BuiltinKind::SyncFetchAndSub),
            "__sync_fetch_and_and" => Some(BuiltinKind::SyncFetchAndAnd),
            "__sync_fetch_and_or" => Some(BuiltinKind::SyncFetchAndOr),
            "__sync_fetch_and_xor" => Some(BuiltinKind::SyncFetchAndXor),
            "__sync_lock_test_and_set" => Some(BuiltinKind::SyncLockTestAndSet),
            "__sync_lock_release" => Some(BuiltinKind::SyncLockRelease),
            "__sync_synchronize" => Some(BuiltinKind::SyncSynchronize),
            "__atomic_load_n" => Some(BuiltinKind::AtomicLoadN),
            "__atomic_store_n" => Some(BuiltinKind::AtomicStoreN),
            "__atomic_exchange_n" => Some(BuiltinKind::AtomicExchangeN),
            "__atomic_compare_exchange_n" => Some(BuiltinKind::AtomicCompareExchangeN),
            "__atomic_fetch_add" => Some(BuiltinKind::AtomicFetchAdd),
            "__atomic_fetch_sub" => Some(BuiltinKind::AtomicFetchSub),
            "__atomic_fetch_and" => Some(BuiltinKind::AtomicFetchAnd),
            "__atomic_fetch_or" => Some(BuiltinKind::AtomicFetchOr),
            "__atomic_fetch_xor" => Some(BuiltinKind::AtomicFetchXor),
            "__builtin_mul_overflow_p" => Some(BuiltinKind::MulOverflowP),
            "__builtin_add_overflow_p" => Some(BuiltinKind::AddOverflowP),
            "__builtin_sub_overflow_p" => Some(BuiltinKind::SubOverflowP),
            _ => None,
        }
    }
}
