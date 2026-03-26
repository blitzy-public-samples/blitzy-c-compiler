//! # IR Builder API
//!
//! This module provides [`IrBuilder`] — the primary interface for constructing
//! BCC's intermediate representation during AST-to-IR lowering (Phase 6).
//!
//! ## Purpose
//!
//! `IrBuilder` is a lightweight instruction factory that:
//! - Tracks the **current insertion point** (which basic block instructions
//!   target).
//! - Handles **automatic SSA value numbering** — each new result-producing
//!   instruction receives a unique, sequential [`Value`] number.
//! - Allocates **basic block IDs** for control-flow graph construction.
//! - Produces fully-formed [`Instruction`] values that the caller inserts
//!   into [`BasicBlock`]s and [`IrFunction`]s.
//!
//! ## Usage Pattern
//!
//! ```ignore
//! use bcc::ir::builder::IrBuilder;
//! use bcc::ir::types::IrType;
//! use bcc::common::diagnostics::Span;
//!
//! let mut builder = IrBuilder::new();
//! let entry = builder.create_block();
//! builder.set_insert_point(entry);
//!
//! // Build an alloca for a local variable (alloca-then-promote pattern)
//! let (ptr, alloca_inst) = builder.build_alloca(IrType::I32, Span::dummy());
//! // ... caller inserts alloca_inst into the entry block ...
//!
//! // Build a store
//! let store_inst = builder.build_store(some_value, ptr, Span::dummy());
//! // ... caller inserts store_inst into the current block ...
//! ```
//!
//! ## Alloca-Then-Promote Architecture
//!
//! During Phase 6 (AST-to-IR lowering), every local variable is emitted as an
//! [`Instruction::Alloca`] in the function's entry block.  The subsequent
//! mem2reg pass (Phase 7) promotes eligible allocas — scalar, non-address-taken
//! — to SSA virtual registers by inserting phi nodes at dominance frontiers.
//! The builder supports this by providing [`build_alloca`](IrBuilder::build_alloca)
//! for variable allocation and [`build_phi`](IrBuilder::build_phi) for SSA
//! merge construction.
//!
//! ## Ownership Model
//!
//! The builder does **NOT** own or manage block/function storage.  It produces
//! [`Instruction`] values and [`BlockId`] / [`Value`] handles.  The caller
//! (typically the lowering driver in `src/ir/lowering/`) is responsible for
//! inserting instructions into the appropriate [`BasicBlock`] within an
//! [`IrFunction`].
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::ir::*` and `crate::common::diagnostics`
//! — no external crates are used.

use crate::common::diagnostics::Span;
use crate::ir::instructions::{BinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// IrType classification helpers (private)
// ===========================================================================

/// Returns `true` if the given IR type is a floating-point type.
///
/// Used internally by arithmetic builder methods to select between integer
/// and floating-point binary operation variants (e.g. `BinOp::Add` vs
/// `BinOp::FAdd`).
#[inline]
fn is_float_type(ty: &IrType) -> bool {
    matches!(ty, IrType::F32 | IrType::F64 | IrType::F80)
}

// ===========================================================================
// IrBuilder — IR instruction factory with SSA value numbering
// ===========================================================================

/// The IR builder — primary interface for constructing IR instructions
/// during AST-to-IR lowering (Phase 6).
///
/// The builder tracks the current insertion point (a [`BlockId`]) and
/// assigns sequential SSA [`Value`] numbers to each result-producing
/// instruction.  It does **not** own or manage block/function storage;
/// it produces instructions that the caller inserts into [`BasicBlock`]s
/// within an [`IrFunction`].
///
/// # SSA Value Numbering
///
/// Each call to a `build_*` method that produces a result allocates the
/// next sequential [`Value`] via [`fresh_value`](IrBuilder::fresh_value).
/// Values are numbered per-function starting from 0 (or from the number
/// of function parameters if values are pre-allocated for params).
///
/// # Block Management
///
/// [`create_block`](IrBuilder::create_block) allocates a new [`BlockId`]
/// without creating any storage.  The caller creates a corresponding
/// [`BasicBlock`] and adds it to the function's block list.
///
/// # Thread Safety
///
/// `IrBuilder` is **not** `Sync` — it is intended for single-threaded use
/// within one function's lowering context.
pub struct IrBuilder {
    /// Index of the current basic block we're inserting into.
    ///
    /// `None` until [`set_insert_point`](IrBuilder::set_insert_point) is
    /// called.  All `build_*` methods conceptually target this block.
    current_block: Option<BlockId>,

    /// Next available SSA value number.
    ///
    /// Incremented by [`fresh_value`](IrBuilder::fresh_value) each time a
    /// result-producing instruction is created.
    next_value: u32,

    /// Next available basic block ID.
    ///
    /// Incremented by [`create_block`](IrBuilder::create_block) each time
    /// a new block is allocated.
    next_block: u32,
}

// ===========================================================================
// IrBuilder — construction and lifecycle
// ===========================================================================

impl IrBuilder {
    /// Creates a new IR builder with no insertion point set.
    ///
    /// The builder starts with `next_value = 0` and `next_block = 0`.
    /// Call [`set_insert_point`](IrBuilder::set_insert_point) to set the
    /// current block before building instructions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut builder = IrBuilder::new();
    /// assert!(builder.get_insert_block().is_none());
    /// ```
    #[inline]
    pub fn new() -> Self {
        IrBuilder {
            current_block: None,
            next_value: 0,
            next_block: 0,
        }
    }

    /// Sets the current insertion block.
    ///
    /// All subsequent `build_*` calls conceptually target this block.  The
    /// caller is responsible for actually inserting instructions into the
    /// corresponding [`BasicBlock`] in the function.
    ///
    /// # Parameters
    ///
    /// - `block`: The [`BlockId`] of the block to insert into.
    #[inline]
    pub fn set_insert_point(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Returns the current insertion block, or `None` if no insertion point
    /// has been set.
    ///
    /// # Returns
    ///
    /// `Some(BlockId)` if an insertion point is active, `None` otherwise.
    #[inline]
    pub fn get_insert_block(&self) -> Option<BlockId> {
        self.current_block
    }

    /// Allocates a new basic block ID.
    ///
    /// This method only allocates an ID — it does **not** create a
    /// [`BasicBlock`] struct or add it to any function.  The caller must
    /// create the block and register it with the function's block list.
    ///
    /// Block IDs are assigned sequentially starting from 0.
    ///
    /// # Returns
    ///
    /// A fresh [`BlockId`] that has not been used before in this builder.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut builder = IrBuilder::new();
    /// let bb0 = builder.create_block();
    /// let bb1 = builder.create_block();
    /// assert_eq!(bb0, BlockId(0));
    /// assert_eq!(bb1, BlockId(1));
    /// ```
    #[inline]
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    /// Allocates the next sequential SSA value number.
    ///
    /// Called internally by `build_*` methods that produce results, but
    /// also available publicly for cases where the lowering phase needs
    /// to pre-allocate values (e.g. for function parameters).
    ///
    /// # Returns
    ///
    /// A fresh [`Value`] with a unique, monotonically increasing index.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut builder = IrBuilder::new();
    /// let v0 = builder.fresh_value();
    /// let v1 = builder.fresh_value();
    /// assert_eq!(v0, Value(0));
    /// assert_eq!(v1, Value(1));
    /// ```
    #[inline]
    pub fn fresh_value(&mut self) -> Value {
        let val = Value(self.next_value);
        self.next_value += 1;
        val
    }
}

// ===========================================================================
// IrBuilder — memory instructions
// ===========================================================================

impl IrBuilder {
    /// Builds an [`Instruction::Alloca`] — stack allocation for a local variable.
    ///
    /// In the alloca-then-promote architecture, every local variable is
    /// initially emitted as an alloca in the function's entry block during
    /// Phase 6.  The mem2reg pass (Phase 7) later promotes eligible allocas
    /// to SSA virtual registers.
    ///
    /// # Parameters
    ///
    /// - `ty`: The type of the value to allocate (the pointee type).
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where `Value` is the SSA result
    /// (a pointer to the allocated memory) and `Instruction` is the
    /// alloca instruction ready for insertion into a basic block.
    pub fn build_alloca(&mut self, ty: IrType, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::Alloca {
            result,
            ty,
            alignment: None,
            span,
        };
        (result, inst)
    }

    /// Builds an [`Instruction::StackAlloc`] — dynamic stack allocation.
    ///
    /// Implements `alloca(size)` / `__builtin_alloca(size)` by adjusting
    /// the stack pointer at runtime.
    pub fn build_stack_alloc(&mut self, size: Value, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::StackAlloc { result, size, span };
        (result, inst)
    }

    /// Builds an [`Instruction::StackSave`] — captures the current stack
    /// pointer into an opaque value.  Used with `StackRestore` to
    /// deallocate VLA stack allocations.
    pub fn build_stack_save(&mut self, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::StackSave { result, span };
        (result, inst)
    }

    /// Builds an [`Instruction::StackRestore`] — restores the stack pointer
    /// to a value previously captured by `StackSave`, deallocating any
    /// dynamic stack allocations made since the save point.
    pub fn build_stack_restore(&mut self, ptr: Value, span: Span) -> Instruction {
        Instruction::StackRestore { ptr, span }
    }

    /// Builds an [`Instruction::Load`] — reads a value from a memory address.
    ///
    /// # Parameters
    ///
    /// - `ptr`: The pointer value to load from.
    /// - `ty`: The type of the value being loaded.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where `Value` is the SSA result
    /// (the loaded data) and `Instruction` is the load instruction.
    pub fn build_load(&mut self, ptr: Value, ty: IrType, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::Load {
            result,
            ptr,
            ty,
            volatile: false,
            span,
        };
        (result, inst)
    }

    /// Builds an [`Instruction::Store`] — writes a value to a memory address.
    ///
    /// Store instructions do **not** produce an SSA result value (they are
    /// void operations).
    ///
    /// # Parameters
    ///
    /// - `val`: The value to store.
    /// - `ptr`: The pointer to the destination memory address.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// The store [`Instruction`] ready for insertion into a basic block.
    pub fn build_store(&mut self, val: Value, ptr: Value, span: Span) -> Instruction {
        Instruction::Store {
            value: val,
            ptr,
            volatile: false,
            span,
        }
    }
}

// ===========================================================================
// IrBuilder — arithmetic and logic instructions
// ===========================================================================

impl IrBuilder {
    /// Builds an addition instruction (integer or floating-point).
    ///
    /// Automatically selects [`BinOp::Add`] for integer types or
    /// [`BinOp::FAdd`] for floating-point types based on `ty`.
    ///
    /// # Parameters
    ///
    /// - `lhs`: Left-hand operand.
    /// - `rhs`: Right-hand operand.
    /// - `ty`: Type of operands and result.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair with the result value and instruction.
    pub fn build_add(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FAdd
        } else {
            BinOp::Add
        };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a subtraction instruction (integer or floating-point).
    ///
    /// Automatically selects [`BinOp::Sub`] or [`BinOp::FSub`] based on `ty`.
    pub fn build_sub(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FSub
        } else {
            BinOp::Sub
        };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a multiplication instruction (integer or floating-point).
    ///
    /// Automatically selects [`BinOp::Mul`] or [`BinOp::FMul`] based on `ty`.
    pub fn build_mul(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FMul
        } else {
            BinOp::Mul
        };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a division instruction (integer or floating-point).
    ///
    /// For integer types, `signed` selects between [`BinOp::SDiv`] (signed)
    /// and [`BinOp::UDiv`] (unsigned).  For floating-point types, `signed`
    /// is ignored and [`BinOp::FDiv`] is always used.
    ///
    /// # Parameters
    ///
    /// - `lhs`: Dividend.
    /// - `rhs`: Divisor.
    /// - `ty`: Type of operands and result.
    /// - `signed`: If `true`, use signed integer division; if `false`, unsigned.
    /// - `span`: Source location for diagnostic reporting.
    pub fn build_div(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        signed: bool,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FDiv
        } else if signed {
            BinOp::SDiv
        } else {
            BinOp::UDiv
        };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a remainder/modulo instruction (integer or floating-point).
    ///
    /// For integer types, `signed` selects between [`BinOp::SRem`] (signed)
    /// and [`BinOp::URem`] (unsigned).  For floating-point types, `signed`
    /// is ignored and [`BinOp::FRem`] is always used.
    pub fn build_rem(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        signed: bool,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FRem
        } else if signed {
            BinOp::SRem
        } else {
            BinOp::URem
        };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a bitwise AND instruction.
    pub fn build_and(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        self.build_binop(BinOp::And, lhs, rhs, ty, span)
    }

    /// Builds a bitwise OR instruction.
    pub fn build_or(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        self.build_binop(BinOp::Or, lhs, rhs, ty, span)
    }

    /// Builds a bitwise XOR instruction.
    pub fn build_xor(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        self.build_binop(BinOp::Xor, lhs, rhs, ty, span)
    }

    /// Builds a shift-left instruction.
    pub fn build_shl(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        self.build_binop(BinOp::Shl, lhs, rhs, ty, span)
    }

    /// Builds a shift-right instruction (arithmetic or logical).
    ///
    /// When `signed` is `true`, uses [`BinOp::AShr`] (arithmetic shift right,
    /// sign-extending).  When `false`, uses [`BinOp::LShr`] (logical shift
    /// right, zero-extending).
    pub fn build_shr(
        &mut self,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        signed: bool,
        span: Span,
    ) -> (Value, Instruction) {
        let op = if signed { BinOp::AShr } else { BinOp::LShr };
        self.build_binop(op, lhs, rhs, ty, span)
    }

    /// Builds a unary negation instruction.
    ///
    /// Implemented as `0 - operand` for integer types (`BinOp::Sub`) or
    /// `0.0 - operand` for floating-point types (`BinOp::FSub`).  The
    /// zero constant is represented by [`Value::UNDEF`] as a sentinel —
    /// the lowering phase is expected to materialize the actual zero
    /// constant and use it as the lhs operand.
    ///
    /// # Design Note
    ///
    /// In practice, the lowering phase should materialize a zero constant
    /// and call `build_sub` directly.  This method provides a convenience
    /// API that encodes the negation pattern for callers that prefer it.
    pub fn build_neg(&mut self, operand: Value, ty: IrType, span: Span) -> (Value, Instruction) {
        let op = if is_float_type(&ty) {
            BinOp::FSub
        } else {
            BinOp::Sub
        };
        // Negation is encoded as `0 - operand`.  The zero operand is
        // represented by Value::UNDEF as a placeholder — the lowering
        // driver or a subsequent canonicalization pass materializes the
        // actual zero constant.
        self.build_binop(op, Value::UNDEF, operand, ty, span)
    }

    /// Builds a bitwise NOT instruction.
    ///
    /// Implemented as `operand XOR all_ones`.  The all-ones constant is
    /// represented by [`Value::UNDEF`] as a sentinel — the lowering phase
    /// materializes the actual all-ones constant.
    ///
    /// # Design Note
    ///
    /// As with `build_neg`, the lowering phase should ideally materialize
    /// the constant and call `build_xor` directly.  This method provides
    /// convenience for callers that prefer the higher-level API.
    pub fn build_not(&mut self, operand: Value, ty: IrType, span: Span) -> (Value, Instruction) {
        // NOT is encoded as `operand XOR all_ones`.  The all-ones
        // sentinel (Value::UNDEF) is the rhs.
        self.build_binop(BinOp::Xor, operand, Value::UNDEF, ty, span)
    }

    /// Internal helper: builds a generic [`Instruction::BinOp`].
    ///
    /// All arithmetic/logic builder methods delegate to this function.
    #[inline]
    fn build_binop(
        &mut self,
        op: BinOp,
        lhs: Value,
        rhs: Value,
        ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — comparison instructions
// ===========================================================================

impl IrBuilder {
    /// Builds an integer comparison instruction.
    ///
    /// Compares `lhs` and `rhs` using the specified [`ICmpOp`] predicate
    /// and produces an [`IrType::I1`] boolean result.
    ///
    /// # Parameters
    ///
    /// - `op`: The comparison predicate (Eq, Ne, Slt, Sle, Sgt, Sge,
    ///   Ult, Ule, Ugt, Uge).
    /// - `lhs`: Left-hand operand.
    /// - `rhs`: Right-hand operand.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is always `I1`.
    pub fn build_icmp(
        &mut self,
        op: ICmpOp,
        lhs: Value,
        rhs: Value,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::ICmp {
            result,
            op,
            lhs,
            rhs,
            span,
        };
        (result, inst)
    }

    /// Builds a floating-point comparison instruction.
    ///
    /// Compares `lhs` and `rhs` using the specified [`FCmpOp`] predicate
    /// (with IEEE 754 NaN semantics) and produces an [`IrType::I1`] boolean
    /// result.
    ///
    /// # Parameters
    ///
    /// - `op`: The comparison predicate (Oeq, One, Olt, Ole, Ogt, Oge,
    ///   Uno, Ord).
    /// - `lhs`: Left-hand operand.
    /// - `rhs`: Right-hand operand.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is always `I1`.
    pub fn build_fcmp(
        &mut self,
        op: FCmpOp,
        lhs: Value,
        rhs: Value,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::FCmp {
            result,
            op,
            lhs,
            rhs,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — control flow (terminator) instructions
// ===========================================================================

impl IrBuilder {
    /// Builds an unconditional branch (terminator).
    ///
    /// Transfers control to the specified `target` block.  This is a
    /// terminator instruction — no instructions should follow it in the
    /// containing basic block.
    ///
    /// # Parameters
    ///
    /// - `target`: The [`BlockId`] of the branch target.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// The branch [`Instruction`] (no result value — terminators are void).
    pub fn build_branch(&mut self, target: BlockId, span: Span) -> Instruction {
        Instruction::Branch { target, span }
    }

    /// Builds a conditional branch (terminator).
    ///
    /// Transfers control to `then_block` if `cond` is true (non-zero I1),
    /// or to `else_block` if false.
    ///
    /// # Parameters
    ///
    /// - `cond`: The branch condition ([`IrType::I1`] value).
    /// - `then_block`: Block to branch to when condition is true.
    /// - `else_block`: Block to branch to when condition is false.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// The conditional branch [`Instruction`] (no result value).
    pub fn build_cond_branch(
        &mut self,
        cond: Value,
        then_block: BlockId,
        else_block: BlockId,
        span: Span,
    ) -> Instruction {
        Instruction::CondBranch {
            condition: cond,
            then_block,
            else_block,
            span,
        }
    }

    /// Builds a switch / jump table (terminator).
    ///
    /// Multi-way branch based on an integer value.  If `value` matches a
    /// case, control transfers to the corresponding block; otherwise,
    /// control transfers to `default`.
    ///
    /// # Parameters
    ///
    /// - `value`: The integer value to switch on.
    /// - `default`: Default target when no case matches.
    /// - `cases`: `(case_value, target_block)` pairs.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// The switch [`Instruction`] (no result value).
    pub fn build_switch(
        &mut self,
        value: Value,
        default: BlockId,
        cases: Vec<(i64, BlockId)>,
        span: Span,
    ) -> Instruction {
        Instruction::Switch {
            value,
            default,
            cases,
            span,
        }
    }

    /// Builds an indirect branch (terminator) for GCC computed-goto.
    ///
    /// Jumps to the code address contained in `target`.  The `possible_targets`
    /// list enumerates all blocks that could be reached (for CFG construction /
    /// phi-node placement).
    pub fn build_indirect_branch(
        &mut self,
        target: Value,
        possible_targets: Vec<BlockId>,
        span: Span,
    ) -> Instruction {
        Instruction::IndirectBranch {
            target,
            possible_targets,
            span,
        }
    }

    /// Builds a block address instruction for computed goto.
    ///
    /// Materializes the runtime address of a labeled basic block, producing
    /// a pointer value that can be stored in a `void *` and later used as
    /// the target of an `IndirectBranch`.
    ///
    /// # Parameters
    ///
    /// - `block`: The target basic block whose address is taken.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair. The `Value` is a pointer to the block.
    pub fn build_block_address(&mut self, block: BlockId, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        (
            result,
            Instruction::BlockAddress {
                result,
                block,
                span,
            },
        )
    }

    /// Builds a function return (terminator).
    ///
    /// Returns from the current function, optionally with a return value.
    /// For void functions, pass `None` as the value.
    ///
    /// # Parameters
    ///
    /// - `value`: The return value, or `None` for void returns.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// The return [`Instruction`] (no result value).
    pub fn build_return(&mut self, value: Option<Value>, span: Span) -> Instruction {
        Instruction::Return { value, span }
    }
}

// ===========================================================================
// IrBuilder — call instructions
// ===========================================================================

impl IrBuilder {
    /// Builds a function call instruction.
    ///
    /// Calls the function referenced by `callee` with the given arguments.
    /// A result [`Value`] is always allocated; for void functions, the
    /// result is unused but still assigned for SSA numbering consistency.
    ///
    /// # Parameters
    ///
    /// - `callee`: The function to call (direct reference or indirect pointer).
    /// - `args`: Call arguments in order.
    /// - `ret_ty`: Return type of the called function.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where `Value` is the call's return
    /// value (unused for void functions).
    pub fn build_call(
        &mut self,
        callee: Value,
        args: Vec<Value>,
        ret_ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::Call {
            result,
            callee,
            args,
            return_type: ret_ty,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — cast instructions
// ===========================================================================

impl IrBuilder {
    /// Builds a bitcast instruction — reinterpret bits as a different type.
    ///
    /// Both source and target types must have the same storage size.
    ///
    /// # Parameters
    ///
    /// - `val`: The source value.
    /// - `to_ty`: The target type.
    /// - `span`: Source location for diagnostic reporting.
    pub fn build_bitcast(&mut self, val: Value, to_ty: IrType, span: Span) -> (Value, Instruction) {
        self.build_bitcast_ex(val, to_ty, false, span)
    }

    /// Builds a bitcast instruction with explicit unsigned flag.
    ///
    /// When `source_unsigned` is true and the conversion is int→float,
    /// the backend emits an unsigned-to-float conversion instead of
    /// the default signed conversion.
    pub fn build_bitcast_ex(
        &mut self,
        val: Value,
        to_ty: IrType,
        source_unsigned: bool,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::BitCast {
            result,
            value: val,
            to_type: to_ty,
            source_unsigned,
            span,
        };
        (result, inst)
    }

    /// Builds an integer truncation instruction.
    ///
    /// Truncates `val` to a narrower integer type by discarding high-order bits.
    ///
    /// # Parameters
    ///
    /// - `val`: The source value (wider integer).
    /// - `to_ty`: The target type (narrower integer).
    /// - `span`: Source location for diagnostic reporting.
    pub fn build_trunc(&mut self, val: Value, to_ty: IrType, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::Trunc {
            result,
            value: val,
            to_type: to_ty,
            span,
        };
        (result, inst)
    }

    /// Builds a zero-extension instruction.
    ///
    /// Extends `val` to a wider integer type by filling high-order bits
    /// with zeros.  Used for unsigned integer widening.
    ///
    /// # Parameters
    ///
    /// - `val`: The source value (narrower integer).
    /// - `to_ty`: The target type (wider integer).
    /// - `span`: Source location for diagnostic reporting.
    pub fn build_zext(&mut self, val: Value, to_ty: IrType, span: Span) -> (Value, Instruction) {
        self.build_zext_from(val, IrType::I8, to_ty, span)
    }

    /// Builds a zero-extension instruction with an explicit source type.
    ///
    /// This variant is used when the source type width matters for
    /// correct code generation (e.g., MOVZX byte vs word on x86-64).
    pub fn build_zext_from(
        &mut self,
        val: Value,
        from_ty: IrType,
        to_ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::ZExt {
            result,
            value: val,
            to_type: to_ty,
            from_type: from_ty,
            span,
        };
        (result, inst)
    }

    /// Builds a sign-extension instruction.
    ///
    /// Extends `val` to a wider integer type by replicating the sign bit.
    /// Used for signed integer widening.
    ///
    /// # Parameters
    ///
    /// - `val`: The source value (narrower integer).
    /// - `to_ty`: The target type (wider integer).
    /// - `span`: Source location for diagnostic reporting.
    pub fn build_sext(&mut self, val: Value, to_ty: IrType, span: Span) -> (Value, Instruction) {
        self.build_sext_from(val, IrType::I8, to_ty, span)
    }

    /// Builds a sign-extension instruction with an explicit source type.
    pub fn build_sext_from(
        &mut self,
        val: Value,
        from_ty: IrType,
        to_ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::SExt {
            result,
            value: val,
            to_type: to_ty,
            from_type: from_ty,
            span,
        };
        (result, inst)
    }

    /// Builds an integer-to-pointer conversion instruction.
    ///
    /// Reinterprets an integer value as a pointer.  The integer width
    /// should match the target's pointer width for well-defined behavior.
    ///
    /// # Parameters
    ///
    /// - `val`: The source integer value.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is a pointer value.
    pub fn build_int_to_ptr(&mut self, val: Value, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::IntToPtr {
            result,
            value: val,
            span,
        };
        (result, inst)
    }

    /// Builds a pointer-to-integer conversion instruction.
    ///
    /// Reinterprets a pointer value as an integer of the specified type.
    /// The target integer width should match the target's pointer width.
    ///
    /// # Parameters
    ///
    /// - `val`: The source pointer value.
    /// - `ty`: The target integer type.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is an integer value.
    pub fn build_ptr_to_int(&mut self, val: Value, ty: IrType, span: Span) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::PtrToInt {
            result,
            value: val,
            to_type: ty,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — pointer/aggregate instructions
// ===========================================================================

impl IrBuilder {
    /// Builds a GetElementPtr instruction — address computation for
    /// structs and arrays.
    ///
    /// Starting from `base` (a pointer), applies each index in `indices`
    /// to compute a derived pointer into a composite type, similar to
    /// LLVM's GEP instruction.
    ///
    /// # Parameters
    ///
    /// - `base`: The base pointer to index into.
    /// - `indices`: Index chain (struct field indices and/or array offsets).
    /// - `result_ty`: The type of the resulting pointer target.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is the computed
    /// derived pointer.
    pub fn build_gep(
        &mut self,
        base: Value,
        indices: Vec<Value>,
        result_ty: IrType,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::GetElementPtr {
            result,
            base,
            indices,
            result_type: result_ty,
            in_bounds: true,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — SSA phi nodes
// ===========================================================================

impl IrBuilder {
    /// Builds a phi node for SSA merge points.
    ///
    /// Phi nodes select a value based on which predecessor block transferred
    /// control to the current block.  They are typically inserted by the
    /// mem2reg pass (Phase 7) at dominance frontiers during SSA construction,
    /// rather than by the initial lowering phase (Phase 6).
    ///
    /// # Parameters
    ///
    /// - `ty`: The type of the phi result and all incoming values.
    /// - `incoming`: `(value, predecessor_block)` pairs.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result is the SSA merge value.
    pub fn build_phi(
        &mut self,
        ty: IrType,
        incoming: Vec<(Value, BlockId)>,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::Phi {
            result,
            ty,
            incoming,
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// IrBuilder — inline assembly
// ===========================================================================

impl IrBuilder {
    /// Builds an inline assembly instruction.
    ///
    /// Represents an `asm` / `__asm__` statement from C source, preserving
    /// the AT&T-syntax template string, constraint descriptors, operand
    /// bindings, and clobber declarations.
    ///
    /// Inline assembly is opaque to optimization passes — all declared
    /// side effects and clobbers must be respected.
    ///
    /// # Parameters
    ///
    /// - `template`: The assembly template string (AT&T syntax with `%0`, `%1`, etc.).
    /// - `constraints`: Constraint string describing operand bindings (e.g. `"=r,r,0"`).
    /// - `operands`: SSA values bound to template placeholders.
    /// - `clobbers`: Clobber list (e.g. `"memory"`, `"cc"`, register names).
    /// - `has_side_effects`: Whether the asm has side effects beyond its operands.
    /// - `span`: Source location for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A `(Value, Instruction)` pair where the result value corresponds to
    /// the output operand (or [`Value::UNDEF`] if there are no outputs).
    pub fn build_inline_asm(
        &mut self,
        template: String,
        constraints: String,
        operands: Vec<Value>,
        clobbers: Vec<String>,
        has_side_effects: bool,
        span: Span,
    ) -> (Value, Instruction) {
        let result = self.fresh_value();
        let inst = Instruction::InlineAsm {
            result,
            template,
            constraints,
            operands,
            clobbers,
            has_side_effects,
            is_volatile: has_side_effects,
            goto_targets: Vec::new(),
            span,
        };
        (result, inst)
    }
}

// ===========================================================================
// Default implementation
// ===========================================================================

impl Default for IrBuilder {
    /// Creates a default `IrBuilder` — equivalent to [`IrBuilder::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::instructions::{BinOp, BlockId, FCmpOp, ICmpOp, Instruction, Value};
    use crate::ir::types::IrType;

    /// Helper: create a dummy span for tests.
    fn dummy_span() -> Span {
        Span::dummy()
    }

    // -- Lifecycle tests --------------------------------------------------

    #[test]
    fn test_new_builder_has_no_insert_point() {
        let builder = IrBuilder::new();
        assert!(builder.get_insert_block().is_none());
    }

    #[test]
    fn test_set_and_get_insert_point() {
        let mut builder = IrBuilder::new();
        let block = BlockId(42);
        builder.set_insert_point(block);
        assert_eq!(builder.get_insert_block(), Some(BlockId(42)));
    }

    #[test]
    fn test_fresh_value_sequential() {
        let mut builder = IrBuilder::new();
        assert_eq!(builder.fresh_value(), Value(0));
        assert_eq!(builder.fresh_value(), Value(1));
        assert_eq!(builder.fresh_value(), Value(2));
    }

    #[test]
    fn test_create_block_sequential() {
        let mut builder = IrBuilder::new();
        assert_eq!(builder.create_block(), BlockId(0));
        assert_eq!(builder.create_block(), BlockId(1));
        assert_eq!(builder.create_block(), BlockId(2));
    }

    // -- Memory instruction tests -----------------------------------------

    #[test]
    fn test_build_alloca() {
        let mut builder = IrBuilder::new();
        let (val, inst) = builder.build_alloca(IrType::I32, dummy_span());
        assert_eq!(val, Value(0));
        match inst {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                ..
            } => {
                assert_eq!(result, Value(0));
                assert_eq!(ty, IrType::I32);
                assert!(alignment.is_none());
            }
            _ => panic!("Expected Alloca instruction"),
        }
    }

    #[test]
    fn test_build_load() {
        let mut builder = IrBuilder::new();
        let ptr = builder.fresh_value();
        let (val, inst) = builder.build_load(ptr, IrType::I64, dummy_span());
        assert_eq!(val, Value(1));
        match inst {
            Instruction::Load {
                result,
                ptr: p,
                ty,
                volatile,
                ..
            } => {
                assert_eq!(result, Value(1));
                assert_eq!(p, Value(0));
                assert_eq!(ty, IrType::I64);
                assert!(!volatile);
            }
            _ => panic!("Expected Load instruction"),
        }
    }

    #[test]
    fn test_build_store_no_result() {
        let mut builder = IrBuilder::new();
        let v = builder.fresh_value();
        let p = builder.fresh_value();
        let inst = builder.build_store(v, p, dummy_span());
        match &inst {
            Instruction::Store {
                value,
                ptr,
                volatile,
                ..
            } => {
                assert_eq!(*value, Value(0));
                assert_eq!(*ptr, Value(1));
                assert!(!volatile);
            }
            _ => panic!("Expected Store instruction"),
        }
        // Store has no result
        assert!(inst.result().is_none());
    }

    // -- Arithmetic/logic instruction tests -------------------------------

    #[test]
    fn test_build_add_integer() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (val, inst) = builder.build_add(lhs, rhs, IrType::I32, dummy_span());
        assert_eq!(val, Value(2));
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Add),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_add_float() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_add(lhs, rhs, IrType::F64, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::FAdd),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_sub_integer() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_sub(lhs, rhs, IrType::I32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Sub),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_sub_float() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_sub(lhs, rhs, IrType::F32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::FSub),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_mul_integer() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_mul(lhs, rhs, IrType::I64, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Mul),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_mul_float() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_mul(lhs, rhs, IrType::F80, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::FMul),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_div_signed() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_div(lhs, rhs, IrType::I32, true, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::SDiv),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_div_unsigned() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_div(lhs, rhs, IrType::I32, false, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::UDiv),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_div_float() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_div(lhs, rhs, IrType::F64, true, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::FDiv),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_rem_signed() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_rem(lhs, rhs, IrType::I32, true, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::SRem),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_rem_unsigned() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_rem(lhs, rhs, IrType::I32, false, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::URem),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_rem_float() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_rem(lhs, rhs, IrType::F64, false, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::FRem),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_and() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_and(lhs, rhs, IrType::I32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::And),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_or() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_or(lhs, rhs, IrType::I32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Or),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_xor() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_xor(lhs, rhs, IrType::I32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Xor),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_shl() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_shl(lhs, rhs, IrType::I32, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::Shl),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_shr_arithmetic() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_shr(lhs, rhs, IrType::I32, true, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::AShr),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_shr_logical() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (_, inst) = builder.build_shr(lhs, rhs, IrType::I32, false, dummy_span());
        match inst {
            Instruction::BinOp { op, .. } => assert_eq!(op, BinOp::LShr),
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_neg_integer() {
        let mut builder = IrBuilder::new();
        let operand = builder.fresh_value();
        let (val, inst) = builder.build_neg(operand, IrType::I32, dummy_span());
        assert_eq!(val, Value(1));
        match inst {
            Instruction::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, BinOp::Sub);
                assert_eq!(lhs, Value::UNDEF);
                assert_eq!(rhs, Value(0));
            }
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_neg_float() {
        let mut builder = IrBuilder::new();
        let operand = builder.fresh_value();
        let (_, inst) = builder.build_neg(operand, IrType::F64, dummy_span());
        match inst {
            Instruction::BinOp { op, lhs, .. } => {
                assert_eq!(op, BinOp::FSub);
                assert_eq!(lhs, Value::UNDEF);
            }
            _ => panic!("Expected BinOp instruction"),
        }
    }

    #[test]
    fn test_build_not() {
        let mut builder = IrBuilder::new();
        let operand = builder.fresh_value();
        let (val, inst) = builder.build_not(operand, IrType::I32, dummy_span());
        assert_eq!(val, Value(1));
        match inst {
            Instruction::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, BinOp::Xor);
                assert_eq!(lhs, Value(0));
                assert_eq!(rhs, Value::UNDEF);
            }
            _ => panic!("Expected BinOp instruction"),
        }
    }

    // -- Comparison instruction tests -------------------------------------

    #[test]
    fn test_build_icmp() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (val, inst) = builder.build_icmp(ICmpOp::Eq, lhs, rhs, dummy_span());
        assert_eq!(val, Value(2));
        match inst {
            Instruction::ICmp { op, .. } => assert_eq!(op, ICmpOp::Eq),
            _ => panic!("Expected ICmp instruction"),
        }
    }

    #[test]
    fn test_build_icmp_all_predicates() {
        let predicates = [
            ICmpOp::Eq,
            ICmpOp::Ne,
            ICmpOp::Slt,
            ICmpOp::Sle,
            ICmpOp::Sgt,
            ICmpOp::Sge,
            ICmpOp::Ult,
            ICmpOp::Ule,
            ICmpOp::Ugt,
            ICmpOp::Uge,
        ];
        for pred in &predicates {
            let mut builder = IrBuilder::new();
            let lhs = builder.fresh_value();
            let rhs = builder.fresh_value();
            let (_, inst) = builder.build_icmp(*pred, lhs, rhs, dummy_span());
            match inst {
                Instruction::ICmp { op, .. } => assert_eq!(op, *pred),
                _ => panic!("Expected ICmp instruction"),
            }
        }
    }

    #[test]
    fn test_build_fcmp() {
        let mut builder = IrBuilder::new();
        let lhs = builder.fresh_value();
        let rhs = builder.fresh_value();
        let (val, inst) = builder.build_fcmp(FCmpOp::Olt, lhs, rhs, dummy_span());
        assert_eq!(val, Value(2));
        match inst {
            Instruction::FCmp { op, .. } => assert_eq!(op, FCmpOp::Olt),
            _ => panic!("Expected FCmp instruction"),
        }
    }

    #[test]
    fn test_build_fcmp_all_predicates() {
        let predicates = [
            FCmpOp::Oeq,
            FCmpOp::One,
            FCmpOp::Olt,
            FCmpOp::Ole,
            FCmpOp::Ogt,
            FCmpOp::Oge,
            FCmpOp::Uno,
            FCmpOp::Ord,
        ];
        for pred in &predicates {
            let mut builder = IrBuilder::new();
            let lhs = builder.fresh_value();
            let rhs = builder.fresh_value();
            let (_, inst) = builder.build_fcmp(*pred, lhs, rhs, dummy_span());
            match inst {
                Instruction::FCmp { op, .. } => assert_eq!(op, *pred),
                _ => panic!("Expected FCmp instruction"),
            }
        }
    }

    // -- Control flow instruction tests -----------------------------------

    #[test]
    fn test_build_branch() {
        let mut builder = IrBuilder::new();
        let target = builder.create_block();
        let inst = builder.build_branch(target, dummy_span());
        match inst {
            Instruction::Branch { target: t, .. } => assert_eq!(t, BlockId(0)),
            _ => panic!("Expected Branch instruction"),
        }
        assert!(inst.is_terminator());
    }

    #[test]
    fn test_build_cond_branch() {
        let mut builder = IrBuilder::new();
        let cond = builder.fresh_value();
        let then_bb = builder.create_block();
        let else_bb = builder.create_block();
        let inst = builder.build_cond_branch(cond, then_bb, else_bb, dummy_span());
        match inst {
            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                assert_eq!(condition, Value(0));
                assert_eq!(then_block, BlockId(0));
                assert_eq!(else_block, BlockId(1));
            }
            _ => panic!("Expected CondBranch instruction"),
        }
        assert!(inst.is_terminator());
    }

    #[test]
    fn test_build_switch() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let default = builder.create_block();
        let case1 = builder.create_block();
        let case2 = builder.create_block();
        let cases = vec![(1, case1), (2, case2)];
        let inst = builder.build_switch(val, default, cases, dummy_span());
        match &inst {
            Instruction::Switch {
                value,
                default: d,
                cases: c,
                ..
            } => {
                assert_eq!(*value, Value(0));
                assert_eq!(*d, BlockId(0));
                assert_eq!(c.len(), 2);
                assert_eq!(c[0], (1, BlockId(1)));
                assert_eq!(c[1], (2, BlockId(2)));
            }
            _ => panic!("Expected Switch instruction"),
        }
        assert!(inst.is_terminator());
    }

    #[test]
    fn test_build_return_with_value() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let inst = builder.build_return(Some(val), dummy_span());
        match inst {
            Instruction::Return { value, .. } => assert_eq!(value, Some(Value(0))),
            _ => panic!("Expected Return instruction"),
        }
        assert!(inst.is_terminator());
    }

    #[test]
    fn test_build_return_void() {
        let builder = IrBuilder::new();
        // build_return with None takes &mut self, but we're not actually
        // allocating any values, so we can use a fresh builder.
        let mut builder = builder;
        let inst = builder.build_return(None, dummy_span());
        match inst {
            Instruction::Return { value, .. } => assert_eq!(value, None),
            _ => panic!("Expected Return instruction"),
        }
    }

    // -- Call instruction tests -------------------------------------------

    #[test]
    fn test_build_call() {
        let mut builder = IrBuilder::new();
        let callee = builder.fresh_value();
        let arg1 = builder.fresh_value();
        let arg2 = builder.fresh_value();
        let (val, inst) = builder.build_call(callee, vec![arg1, arg2], IrType::I32, dummy_span());
        assert_eq!(val, Value(3));
        match inst {
            Instruction::Call {
                result,
                callee: c,
                args,
                return_type,
                ..
            } => {
                assert_eq!(result, Value(3));
                assert_eq!(c, Value(0));
                assert_eq!(args.len(), 2);
                assert_eq!(return_type, IrType::I32);
            }
            _ => panic!("Expected Call instruction"),
        }
    }

    #[test]
    fn test_build_call_void() {
        let mut builder = IrBuilder::new();
        let callee = builder.fresh_value();
        let (val, inst) = builder.build_call(callee, vec![], IrType::Void, dummy_span());
        // Even void calls allocate a Value for SSA consistency
        assert_eq!(val, Value(1));
        match inst {
            Instruction::Call {
                return_type, args, ..
            } => {
                assert_eq!(return_type, IrType::Void);
                assert!(args.is_empty());
            }
            _ => panic!("Expected Call instruction"),
        }
    }

    // -- Cast instruction tests -------------------------------------------

    #[test]
    fn test_build_bitcast() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_bitcast(val, IrType::Ptr, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::BitCast { value, to_type, .. } => {
                assert_eq!(value, Value(0));
                assert_eq!(to_type, IrType::Ptr);
            }
            _ => panic!("Expected BitCast instruction"),
        }
    }

    #[test]
    fn test_build_trunc() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_trunc(val, IrType::I8, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::Trunc { value, to_type, .. } => {
                assert_eq!(value, Value(0));
                assert_eq!(to_type, IrType::I8);
            }
            _ => panic!("Expected Trunc instruction"),
        }
    }

    #[test]
    fn test_build_zext() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_zext(val, IrType::I64, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::ZExt { value, to_type, .. } => {
                assert_eq!(value, Value(0));
                assert_eq!(to_type, IrType::I64);
            }
            _ => panic!("Expected ZExt instruction"),
        }
    }

    #[test]
    fn test_build_sext() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_sext(val, IrType::I32, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::SExt { value, to_type, .. } => {
                assert_eq!(value, Value(0));
                assert_eq!(to_type, IrType::I32);
            }
            _ => panic!("Expected SExt instruction"),
        }
    }

    #[test]
    fn test_build_int_to_ptr() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_int_to_ptr(val, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::IntToPtr { value, .. } => {
                assert_eq!(value, Value(0));
            }
            _ => panic!("Expected IntToPtr instruction"),
        }
    }

    #[test]
    fn test_build_ptr_to_int() {
        let mut builder = IrBuilder::new();
        let val = builder.fresh_value();
        let (result, inst) = builder.build_ptr_to_int(val, IrType::I64, dummy_span());
        assert_eq!(result, Value(1));
        match inst {
            Instruction::PtrToInt { value, to_type, .. } => {
                assert_eq!(value, Value(0));
                assert_eq!(to_type, IrType::I64);
            }
            _ => panic!("Expected PtrToInt instruction"),
        }
    }

    // -- GEP instruction tests --------------------------------------------

    #[test]
    fn test_build_gep() {
        let mut builder = IrBuilder::new();
        let base = builder.fresh_value();
        let idx = builder.fresh_value();
        let (result, inst) = builder.build_gep(base, vec![idx], IrType::I32, dummy_span());
        assert_eq!(result, Value(2));
        match inst {
            Instruction::GetElementPtr {
                base: b,
                indices,
                result_type,
                in_bounds,
                ..
            } => {
                assert_eq!(b, Value(0));
                assert_eq!(indices.len(), 1);
                assert_eq!(indices[0], Value(1));
                assert_eq!(result_type, IrType::I32);
                assert!(in_bounds);
            }
            _ => panic!("Expected GetElementPtr instruction"),
        }
    }

    // -- Phi instruction tests --------------------------------------------

    #[test]
    fn test_build_phi() {
        let mut builder = IrBuilder::new();
        let v1 = builder.fresh_value();
        let v2 = builder.fresh_value();
        let bb0 = builder.create_block();
        let bb1 = builder.create_block();
        let incoming = vec![(v1, bb0), (v2, bb1)];
        let (result, inst) = builder.build_phi(IrType::I32, incoming, dummy_span());
        assert_eq!(result, Value(2));
        match inst {
            Instruction::Phi {
                ty, incoming: inc, ..
            } => {
                assert_eq!(ty, IrType::I32);
                assert_eq!(inc.len(), 2);
                assert_eq!(inc[0], (Value(0), BlockId(0)));
                assert_eq!(inc[1], (Value(1), BlockId(1)));
            }
            _ => panic!("Expected Phi instruction"),
        }
    }

    #[test]
    fn test_build_phi_empty_incoming() {
        let mut builder = IrBuilder::new();
        let (result, inst) = builder.build_phi(IrType::Ptr, vec![], dummy_span());
        assert_eq!(result, Value(0));
        match inst {
            Instruction::Phi { incoming, ty, .. } => {
                assert!(incoming.is_empty());
                assert_eq!(ty, IrType::Ptr);
            }
            _ => panic!("Expected Phi instruction"),
        }
    }

    // -- Inline assembly instruction tests --------------------------------

    #[test]
    fn test_build_inline_asm() {
        let mut builder = IrBuilder::new();
        let operand = builder.fresh_value();
        let (result, inst) = builder.build_inline_asm(
            "mov %0, %1".to_string(),
            "=r,r".to_string(),
            vec![operand],
            vec!["memory".to_string(), "cc".to_string()],
            true,
            dummy_span(),
        );
        assert_eq!(result, Value(1));
        match inst {
            Instruction::InlineAsm {
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_volatile,
                goto_targets,
                ..
            } => {
                assert_eq!(template, "mov %0, %1");
                assert_eq!(constraints, "=r,r");
                assert_eq!(operands.len(), 1);
                assert_eq!(clobbers.len(), 2);
                assert!(has_side_effects);
                assert!(is_volatile);
                assert!(goto_targets.is_empty());
            }
            _ => panic!("Expected InlineAsm instruction"),
        }
    }

    // -- Integration test: sequential value numbering across methods ------

    #[test]
    fn test_value_numbering_across_instructions() {
        let mut builder = IrBuilder::new();

        // Allocate values for params manually
        let _p0 = builder.fresh_value(); // Value(0)
        let _p1 = builder.fresh_value(); // Value(1)

        // Alloca
        let (alloca_val, _) = builder.build_alloca(IrType::I32, dummy_span());
        assert_eq!(alloca_val, Value(2));

        // Load
        let (load_val, _) = builder.build_load(alloca_val, IrType::I32, dummy_span());
        assert_eq!(load_val, Value(3));

        // Store does not consume a value number
        let _store = builder.build_store(load_val, alloca_val, dummy_span());

        // Add
        let (add_val, _) = builder.build_add(load_val, load_val, IrType::I32, dummy_span());
        assert_eq!(add_val, Value(4));

        // ICmp
        let (cmp_val, _) = builder.build_icmp(ICmpOp::Eq, add_val, load_val, dummy_span());
        assert_eq!(cmp_val, Value(5));

        // Call
        let (call_val, _) = builder.build_call(Value(0), vec![add_val], IrType::I32, dummy_span());
        assert_eq!(call_val, Value(6));
    }

    // -- Default trait test -----------------------------------------------

    #[test]
    fn test_default_builder() {
        let builder = IrBuilder::default();
        assert!(builder.get_insert_block().is_none());
    }

    // -- Span propagation test --------------------------------------------

    #[test]
    fn test_span_propagation() {
        let mut builder = IrBuilder::new();
        let span = Span::new(1, 10, 20);
        let (_, inst) = builder.build_alloca(IrType::I32, span);
        assert_eq!(inst.span().file_id, 1);
        assert_eq!(inst.span().start, 10);
        assert_eq!(inst.span().end, 20);
    }
}
