//! # IR Instruction Definitions
//!
//! This module defines all intermediate representation (IR) instruction types
//! for BCC.  Instructions are the fundamental building blocks of the IR —
//! every pipeline phase from AST-to-IR lowering (Phase 6) through optimization
//! passes (Phase 8) to code generation (Phase 10) operates on them.
//!
//! ## SSA Value Numbering
//!
//! Each instruction that produces a result carries a unique [`Value`] (an SSA
//! virtual register number).  Values are numbered sequentially per function
//! and are immutable once assigned — this is the core SSA property.
//!
//! ## Terminator Instructions
//!
//! [`Branch`](Instruction::Branch), [`CondBranch`](Instruction::CondBranch),
//! [`Switch`](Instruction::Switch), and [`Return`](Instruction::Return) are
//! *terminator* instructions that end their containing basic block.  Every
//! well-formed basic block has exactly one terminator as its last instruction.
//!
//! ## Phi Nodes
//!
//! [`Phi`](Instruction::Phi) instructions implement SSA merge semantics at
//! control-flow join points.  Each phi carries a list of `(Value, BlockId)`
//! pairs — one per incoming control-flow edge — selecting the appropriate
//! reaching definition at runtime.  Phi nodes always appear at the beginning
//! of a basic block, before any non-phi instructions.
//!
//! ## Alloca-Then-Promote Pattern
//!
//! The IR lowering phase (Phase 6) emits [`Alloca`](Instruction::Alloca)
//! instructions for every local variable in the entry basic block.  The
//! subsequent mem2reg pass (Phase 7) promotes eligible allocas to SSA
//! virtual registers, inserting phi nodes at dominance frontiers.  This
//! mirrors the LLVM approach to SSA construction.
//!
//! ## Instruction Categories
//!
//! | Category     | Variants                                                  |
//! |-------------|-----------------------------------------------------------|
//! | Memory       | `Alloca`, `Load`, `Store`                                 |
//! | Arithmetic   | `BinOp`                                                   |
//! | Comparison   | `ICmp`, `FCmp`                                            |
//! | Control flow | `Branch`, `CondBranch`, `Switch`, `Return`                |
//! | Call         | `Call`                                                    |
//! | SSA          | `Phi`                                                     |
//! | Pointer      | `GetElementPtr`                                           |
//! | Casts        | `BitCast`, `Trunc`, `ZExt`, `SExt`, `IntToPtr`, `PtrToInt`|
//! | Inline asm   | `InlineAsm`                                               |
//!
//! ## Zero-Dependency
//!
//! This module depends only on `std` (formatting traits) and internal crate
//! modules — no external crates are used.

use std::fmt;

use crate::common::diagnostics::Span;
use crate::ir::types::IrType;

// ===========================================================================
// Value — SSA value reference
// ===========================================================================

/// An SSA value reference — represents the result of an instruction.
///
/// Values are numbered sequentially within a function starting from 0.
/// Each result-producing instruction receives a unique `Value`, and the
/// SSA property guarantees that each `Value` is defined exactly once.
///
/// The special sentinel [`Value::UNDEF`] represents an undefined value,
/// used during IR construction before reaching definitions are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub u32);

impl Value {
    /// Sentinel value representing an undefined SSA value.
    ///
    /// Used as a placeholder during IR construction (e.g., before
    /// mem2reg resolves reaching definitions) and for instructions
    /// whose result is unused.
    pub const UNDEF: Value = Value(u32::MAX);

    /// Returns the numeric index of this value.
    ///
    /// Value indices are used for array-based storage in the IR builder,
    /// register allocator, and use-def chain data structures.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Value {
    /// Formats the value as `%N` where N is the numeric index,
    /// or `undef` for the [`UNDEF`](Value::UNDEF) sentinel.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Value::UNDEF {
            write!(f, "undef")
        } else {
            write!(f, "%{}", self.0)
        }
    }
}

// ===========================================================================
// BlockId — basic block reference
// ===========================================================================

/// Reference to a basic block by its index within its parent function.
///
/// `BlockId` is a lightweight handle used by terminator instructions
/// (branch, conditional branch, switch) and phi nodes to reference
/// target or predecessor basic blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Returns the numeric index of this block within its parent function.
    ///
    /// Block indices correspond to positions in the function's basic block
    /// list and are used for direct array indexing.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for BlockId {
    /// Formats the block reference as `label %bb<N>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label %bb{}", self.0)
    }
}

// ===========================================================================
// BinOp — binary operation kinds
// ===========================================================================

/// Binary arithmetic and logic operation kinds.
///
/// These are the operations available for [`Instruction::BinOp`].  Integer
/// and floating-point variants are kept separate (e.g. `Add` vs `FAdd`)
/// because they have different semantics — integer overflow wraps in two's
/// complement, while floating-point follows IEEE 754 rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Integer addition (wrapping two's-complement).
    Add,
    /// Integer subtraction (wrapping two's-complement).
    Sub,
    /// Integer multiplication (wrapping two's-complement).
    Mul,
    /// Signed integer division (rounds toward zero).
    SDiv,
    /// Unsigned integer division.
    UDiv,
    /// Signed integer remainder (sign follows dividend).
    SRem,
    /// Unsigned integer remainder.
    URem,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Shift left (zero-fill).
    Shl,
    /// Arithmetic shift right (sign-extending).
    AShr,
    /// Logical shift right (zero-extending).
    LShr,
    /// IEEE 754 floating-point addition.
    FAdd,
    /// IEEE 754 floating-point subtraction.
    FSub,
    /// IEEE 754 floating-point multiplication.
    FMul,
    /// IEEE 754 floating-point division.
    FDiv,
    /// IEEE 754 floating-point remainder.
    FRem,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::SDiv => "sdiv",
            BinOp::UDiv => "udiv",
            BinOp::SRem => "srem",
            BinOp::URem => "urem",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::AShr => "ashr",
            BinOp::LShr => "lshr",
            BinOp::FAdd => "fadd",
            BinOp::FSub => "fsub",
            BinOp::FMul => "fmul",
            BinOp::FDiv => "fdiv",
            BinOp::FRem => "frem",
        };
        write!(f, "{}", name)
    }
}

// ===========================================================================
// ICmpOp — integer comparison predicates
// ===========================================================================

/// Integer comparison predicates for [`Instruction::ICmp`].
///
/// Signed and unsigned comparisons are distinguished because the same
/// bit pattern can represent different numeric values depending on
/// interpretation.  All comparisons produce an `I1` (boolean) result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ICmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
}

impl fmt::Display for ICmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ICmpOp::Eq => "eq",
            ICmpOp::Ne => "ne",
            ICmpOp::Slt => "slt",
            ICmpOp::Sle => "sle",
            ICmpOp::Sgt => "sgt",
            ICmpOp::Sge => "sge",
            ICmpOp::Ult => "ult",
            ICmpOp::Ule => "ule",
            ICmpOp::Ugt => "ugt",
            ICmpOp::Uge => "uge",
        };
        write!(f, "{}", name)
    }
}

// ===========================================================================
// FCmpOp — floating-point comparison predicates
// ===========================================================================

/// Floating-point comparison predicates for [`Instruction::FCmp`].
///
/// Floating-point comparisons must account for NaN — any comparison
/// involving NaN returns `false` for ordered predicates and `true`
/// for the unordered predicate.  The `Ord` predicate returns `true`
/// when neither operand is NaN.
///
/// All comparisons produce an `I1` (boolean) result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FCmpOp {
    /// Ordered and equal.
    Oeq,
    /// Ordered and not equal.
    One,
    /// Ordered and less than.
    Olt,
    /// Ordered and less than or equal.
    Ole,
    /// Ordered and greater than.
    Ogt,
    /// Ordered and greater than or equal.
    Oge,
    /// Unordered (at least one operand is NaN).
    Uno,
    /// Ordered (neither operand is NaN).
    Ord,
}

impl fmt::Display for FCmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            FCmpOp::Oeq => "oeq",
            FCmpOp::One => "one",
            FCmpOp::Olt => "olt",
            FCmpOp::Ole => "ole",
            FCmpOp::Ogt => "ogt",
            FCmpOp::Oge => "oge",
            FCmpOp::Uno => "uno",
            FCmpOp::Ord => "ord",
        };
        write!(f, "{}", name)
    }
}

// ===========================================================================
// Instruction — the core IR instruction enum
// ===========================================================================

/// The core IR instruction enum — every IR operation is represented by one
/// of these variants.
///
/// # Result values
///
/// Instructions that produce a result carry a `result: Value` field.
/// Instructions without results ([`Store`](Instruction::Store),
/// [`Branch`](Instruction::Branch), [`CondBranch`](Instruction::CondBranch),
/// [`Switch`](Instruction::Switch), [`Return`](Instruction::Return)) have
/// no result field.
///
/// # Source spans
///
/// Every instruction carries a [`Span`] for diagnostic reporting, enabling
/// precise error messages during optimization and code generation phases.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Allocate stack memory for a local variable.
    ///
    /// In the alloca-then-promote pattern, the IR lowering phase (Phase 6)
    /// emits an `Alloca` for every local variable in the function's entry
    /// block.  The mem2reg pass (Phase 7) then promotes eligible allocas
    /// to SSA virtual registers.
    ///
    /// The result is a pointer to the allocated memory; `ty` specifies the
    /// type of the value stored at that address (not the pointer type).
    Alloca {
        /// SSA result value (pointer to allocated memory).
        result: Value,
        /// Type of the value to be allocated (the pointee type).
        ty: IrType,
        /// Optional explicit alignment override (from `_Alignas` or `aligned` attribute).
        alignment: Option<usize>,
        /// Source location.
        span: Span,
    },

    /// Dynamic stack allocation (`alloca(size)` / `__builtin_alloca`).
    ///
    /// Adjusts the stack pointer at runtime by `size` bytes (with proper
    /// alignment) and returns a pointer to the allocated region.  Unlike
    /// `Alloca`, the size is a runtime-computed value, not a compile-time
    /// type.
    StackAlloc {
        /// SSA result value (pointer to allocated memory).
        result: Value,
        /// Runtime size value (in bytes).
        size: Value,
        /// Source location.
        span: Span,
    },

    /// Save the current stack pointer.  Used before VLA allocations so that
    /// the stack can be restored when the VLA goes out of scope or when a
    /// backwards goto re-enters the VLA's scope.
    StackSave {
        /// SSA result value (opaque pointer representing saved SP).
        result: Value,
        /// Source location.
        span: Span,
    },

    /// Restore the stack pointer to a previously saved value.  Deallocates
    /// any stack allocations made after the corresponding `StackSave`.
    StackRestore {
        /// The value returned by a prior `StackSave`.
        ptr: Value,
        /// Source location.
        span: Span,
    },

    /// Load a value from a memory address.
    ///
    /// Reads `ty`-sized data from the pointer `ptr`.  When `volatile` is
    /// `true`, the load must not be eliminated, reordered, or duplicated
    /// by optimization passes.
    Load {
        /// SSA result value (the loaded data).
        result: Value,
        /// Pointer to load from.
        ptr: Value,
        /// Type of the value being loaded.
        ty: IrType,
        /// Whether this load has volatile semantics.
        volatile: bool,
        /// Source location.
        span: Span,
    },

    /// Store a value to a memory address.
    ///
    /// Writes `value` to the memory location pointed to by `ptr`.  Store
    /// does not produce a result value.  When `volatile` is `true`, the
    /// store must not be eliminated, reordered, or duplicated.
    Store {
        /// Value to store.
        value: Value,
        /// Pointer to store to.
        ptr: Value,
        /// Whether this store has volatile semantics.
        volatile: bool,
        /// Source location.
        span: Span,
    },

    /// Binary arithmetic or logic operation.
    ///
    /// Computes `lhs <op> rhs` where both operands have type `ty`.  The
    /// result has the same type.  Integer and floating-point operations
    /// use distinct [`BinOp`] variants.
    BinOp {
        /// SSA result value.
        result: Value,
        /// The specific binary operation.
        op: BinOp,
        /// Left-hand operand.
        lhs: Value,
        /// Right-hand operand.
        rhs: Value,
        /// Type of operands and result.
        ty: IrType,
        /// Source location.
        span: Span,
    },

    /// Integer comparison.
    ///
    /// Compares `lhs` and `rhs` using the specified predicate.  Always
    /// produces an [`IrType::I1`] boolean result.
    ICmp {
        /// SSA result value (I1).
        result: Value,
        /// Comparison predicate.
        op: ICmpOp,
        /// Left-hand operand.
        lhs: Value,
        /// Right-hand operand.
        rhs: Value,
        /// Source location.
        span: Span,
    },

    /// Floating-point comparison.
    ///
    /// Compares `lhs` and `rhs` using the specified predicate, accounting
    /// for IEEE 754 NaN semantics.  Always produces an [`IrType::I1`]
    /// boolean result.
    FCmp {
        /// SSA result value (I1).
        result: Value,
        /// Comparison predicate.
        op: FCmpOp,
        /// Left-hand operand.
        lhs: Value,
        /// Right-hand operand.
        rhs: Value,
        /// Source location.
        span: Span,
    },

    /// Unconditional branch (terminator).
    ///
    /// Transfers control to `target`.  This is a terminator instruction
    /// and must be the last instruction in its basic block.
    Branch {
        /// Target basic block.
        target: BlockId,
        /// Source location.
        span: Span,
    },

    /// Conditional branch (terminator).
    ///
    /// Transfers control to `then_block` if `condition` is true (non-zero),
    /// or `else_block` if false.  The condition must be an `I1` value.
    CondBranch {
        /// Branch condition (I1 value).
        condition: Value,
        /// Block to branch to when condition is true.
        then_block: BlockId,
        /// Block to branch to when condition is false.
        else_block: BlockId,
        /// Source location.
        span: Span,
    },

    /// Switch / jump table (terminator).
    ///
    /// Multi-way branch based on an integer `value`.  If `value` matches
    /// a case, control transfers to the corresponding block; otherwise,
    /// control transfers to `default`.
    Switch {
        /// Value to switch on.
        value: Value,
        /// Default target when no case matches.
        default: BlockId,
        /// `(case_value, target_block)` pairs.
        cases: Vec<(i64, BlockId)>,
        /// Source location.
        span: Span,
    },

    /// Indirect branch (terminator).
    ///
    /// Jumps to the code address contained in `target`.  Used for GCC
    /// computed-goto (`goto *expr`).  `possible_targets` lists all blocks
    /// that could be reached — these are the blocks corresponding to labels
    /// whose address was taken via `&&label`.  The list is used for CFG
    /// construction only; at runtime the actual branch target is the
    /// pointer value in `target`.
    IndirectBranch {
        /// The code-address value (a `void*` produced by `&&label`).
        target: Value,
        /// All basic blocks that could be reached (for CFG / phi edges).
        possible_targets: Vec<BlockId>,
        /// Source location.
        span: Span,
    },

    /// Produces the runtime address of a basic block as a `void*`.
    /// Used by the GCC `&&label` extension.  At code-generation time,
    /// this lowers to a `LEA` of the block's assembly label (RIP-relative
    /// on x86-64).
    BlockAddress {
        /// SSA result value — the address as a pointer.
        result: Value,
        /// The target block whose address is taken.
        block: BlockId,
        /// Source location.
        span: Span,
    },

    /// Function call.
    ///
    /// Calls the function referenced by `callee` with the given `args`.
    /// The result is the return value of the call; for void functions,
    /// the result `Value` is allocated but unused.
    Call {
        /// SSA result value (the return value).
        result: Value,
        /// Callee — either a direct function reference or an indirect pointer.
        callee: Value,
        /// Call arguments.
        args: Vec<Value>,
        /// Return type of the called function.
        return_type: IrType,
        /// Source location.
        span: Span,
    },

    /// Function return (terminator).
    ///
    /// Returns from the current function.  `value` is `Some(v)` for
    /// non-void functions and `None` for void returns.
    Return {
        /// Return value, or `None` for void functions.
        value: Option<Value>,
        /// Source location.
        span: Span,
    },

    /// SSA Phi node.
    ///
    /// Selects a value based on which predecessor block transferred
    /// control to the current block.  Phi nodes are inserted by the
    /// mem2reg pass at dominance frontiers and must appear at the
    /// beginning of their block, before any non-phi instructions.
    ///
    /// Each entry in `incoming` is a `(Value, BlockId)` pair mapping
    /// a predecessor block to the value that reaches along that edge.
    Phi {
        /// SSA result value.
        result: Value,
        /// Type of the phi result and all incoming values.
        ty: IrType,
        /// `(value, predecessor_block)` pairs.
        incoming: Vec<(Value, BlockId)>,
        /// Source location.
        span: Span,
    },

    /// Get Element Pointer — computes an address within a composite type.
    ///
    /// Starting from `base` (a pointer), applies each index in `indices`
    /// to compute a derived pointer into a struct, array, or nested
    /// aggregate.  The `in_bounds` flag indicates that all intermediate
    /// addresses are within the allocated object (enabling optimizations).
    GetElementPtr {
        /// SSA result value (the computed pointer).
        result: Value,
        /// Base pointer to index into.
        base: Value,
        /// Index chain (may include struct field indices and array offsets).
        indices: Vec<Value>,
        /// Type of the resulting pointer target.
        result_type: IrType,
        /// Whether all indices are known to be in-bounds.
        in_bounds: bool,
        /// Source location.
        span: Span,
    },

    /// Bit-level type reinterpretation.
    ///
    /// Reinterprets the bits of `value` as `to_type` without changing
    /// the underlying data.  Both types must have the same storage size.
    BitCast {
        /// SSA result value.
        result: Value,
        /// Source value.
        value: Value,
        /// Target type (same size as source type).
        to_type: IrType,
        /// Whether the source integer is unsigned (for int→float / float→int).
        source_unsigned: bool,
        /// Source location.
        span: Span,
    },

    /// Integer truncation.
    ///
    /// Truncates `value` to a narrower integer type by discarding
    /// high-order bits.  The source must be wider than `to_type`.
    Trunc {
        /// SSA result value.
        result: Value,
        /// Source value (wider integer).
        value: Value,
        /// Target type (narrower integer).
        to_type: IrType,
        /// Source location.
        span: Span,
    },

    /// Zero extension.
    ///
    /// Extends `value` to a wider integer type by filling high-order
    /// bits with zeros.  Used for unsigned integer widening.
    ZExt {
        /// SSA result value.
        result: Value,
        /// Source value (narrower integer).
        value: Value,
        /// Target type (wider integer).
        to_type: IrType,
        /// Source type (narrower integer) — used by codegen to select
        /// the correct MOVZX variant (byte vs word source).
        from_type: IrType,
        /// Source location.
        span: Span,
    },

    /// Sign extension.
    ///
    /// Extends `value` to a wider integer type by replicating the sign
    /// bit.  Used for signed integer widening.
    SExt {
        /// SSA result value.
        result: Value,
        /// Source value (narrower integer).
        value: Value,
        /// Target type (wider integer).
        to_type: IrType,
        /// Source type (narrower integer) — used by codegen to select
        /// the correct MOVSX variant (byte vs word source).
        from_type: IrType,
        /// Source location.
        span: Span,
    },

    /// Integer to pointer conversion.
    ///
    /// Reinterprets an integer value as a pointer.  The integer width
    /// should match the target's pointer width for well-defined behavior.
    IntToPtr {
        /// SSA result value (pointer).
        result: Value,
        /// Source integer value.
        value: Value,
        /// Source location.
        span: Span,
    },

    /// Pointer to integer conversion.
    ///
    /// Reinterprets a pointer value as an integer of type `to_type`.
    /// The target integer width should match the target's pointer width.
    PtrToInt {
        /// SSA result value (integer).
        result: Value,
        /// Source pointer value.
        value: Value,
        /// Target integer type.
        to_type: IrType,
        /// Source location.
        span: Span,
    },

    /// Inline assembly statement.
    ///
    /// Represents an `asm` / `__asm__` statement with its template string,
    /// constraints, operands, clobber list, and optional `goto` targets.
    /// Inline assembly is opaque to optimization passes — side effects
    /// and clobbers must be respected.
    InlineAsm {
        /// SSA result value (for output operands; UNDEF if no output).
        result: Value,
        /// Assembly template string (AT&T syntax with `%0`, `%1`, etc.).
        template: String,
        /// Constraint string (e.g. `"=r,r,0"`).
        constraints: String,
        /// Operand values bound to template placeholders.
        operands: Vec<Value>,
        /// Clobber list (e.g. `"memory"`, `"cc"`, register names).
        clobbers: Vec<String>,
        /// Whether the inline asm has side effects beyond its operands.
        has_side_effects: bool,
        /// Whether this asm statement is volatile (must not be eliminated).
        is_volatile: bool,
        /// Target blocks for `asm goto` (empty for normal asm).
        goto_targets: Vec<BlockId>,
        /// Source location.
        span: Span,
    },
}

// ===========================================================================
// Instruction — utility methods
// ===========================================================================

impl Instruction {
    /// Returns the SSA result value produced by this instruction, or `None`
    /// for instructions that do not produce a result.
    ///
    /// Instructions without results: `Store`, `Branch`, `CondBranch`,
    /// `Switch`, `Return`.
    pub fn result(&self) -> Option<Value> {
        match self {
            Instruction::Alloca { result, .. }
            | Instruction::StackAlloc { result, .. }
            | Instruction::StackSave { result, .. }
            | Instruction::Load { result, .. }
            | Instruction::BinOp { result, .. }
            | Instruction::ICmp { result, .. }
            | Instruction::FCmp { result, .. }
            | Instruction::Call { result, .. }
            | Instruction::Phi { result, .. }
            | Instruction::GetElementPtr { result, .. }
            | Instruction::BitCast { result, .. }
            | Instruction::Trunc { result, .. }
            | Instruction::ZExt { result, .. }
            | Instruction::SExt { result, .. }
            | Instruction::IntToPtr { result, .. }
            | Instruction::PtrToInt { result, .. }
            | Instruction::InlineAsm { result, .. }
            | Instruction::BlockAddress { result, .. } => Some(*result),

            Instruction::Store { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Switch { .. }
            | Instruction::IndirectBranch { .. }
            | Instruction::StackRestore { .. }
            | Instruction::Return { .. } => None,
        }
    }

    /// Returns the source location span for this instruction.
    ///
    /// Every instruction carries a span for diagnostic reporting, enabling
    /// precise error messages from any pipeline phase.
    pub fn span(&self) -> Span {
        match self {
            Instruction::Alloca { span, .. }
            | Instruction::StackAlloc { span, .. }
            | Instruction::StackSave { span, .. }
            | Instruction::StackRestore { span, .. }
            | Instruction::Load { span, .. }
            | Instruction::Store { span, .. }
            | Instruction::BinOp { span, .. }
            | Instruction::ICmp { span, .. }
            | Instruction::FCmp { span, .. }
            | Instruction::Branch { span, .. }
            | Instruction::CondBranch { span, .. }
            | Instruction::Switch { span, .. }
            | Instruction::IndirectBranch { span, .. }
            | Instruction::Call { span, .. }
            | Instruction::Return { span, .. }
            | Instruction::Phi { span, .. }
            | Instruction::GetElementPtr { span, .. }
            | Instruction::BitCast { span, .. }
            | Instruction::Trunc { span, .. }
            | Instruction::ZExt { span, .. }
            | Instruction::SExt { span, .. }
            | Instruction::IntToPtr { span, .. }
            | Instruction::PtrToInt { span, .. }
            | Instruction::InlineAsm { span, .. }
            | Instruction::BlockAddress { span, .. } => *span,
        }
    }

    /// Returns `true` if this instruction is a terminator.
    ///
    /// Terminator instructions end a basic block and transfer control
    /// flow to successor blocks (or return from the function).
    /// Every well-formed basic block has exactly one terminator as its
    /// last instruction.
    #[inline]
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Branch { .. }
                | Instruction::CondBranch { .. }
                | Instruction::Switch { .. }
                | Instruction::IndirectBranch { .. }
                | Instruction::Return { .. }
        )
    }

    /// Returns `true` if this is a Phi instruction.
    ///
    /// Phi nodes must appear at the start of a basic block, before
    /// any non-phi instructions.
    #[inline]
    pub fn is_phi(&self) -> bool {
        matches!(self, Instruction::Phi { .. })
    }

    /// Returns `true` if this is an Alloca instruction.
    ///
    /// Alloca instructions allocate stack memory and are the starting
    /// point of the alloca-then-promote SSA construction pattern.
    #[inline]
    pub fn is_alloca(&self) -> bool {
        matches!(self, Instruction::Alloca { .. })
    }

    /// Returns `true` if this instruction has volatile semantics.
    ///
    /// Volatile instructions must not be eliminated, reordered, or
    /// duplicated by optimization passes.  Applies to volatile loads,
    /// volatile stores, and volatile inline assembly.
    pub fn is_volatile(&self) -> bool {
        match self {
            Instruction::Load { volatile, .. } => *volatile,
            Instruction::Store { volatile, .. } => *volatile,
            Instruction::InlineAsm { is_volatile, .. } => *is_volatile,
            _ => false,
        }
    }

    /// Returns all [`Value`] operands used (read) by this instruction.
    ///
    /// This is used by optimization passes for use-def analysis — every
    /// value returned here is a *use* of the corresponding definition.
    /// The result value (if any) is NOT included; it is a *definition*.
    pub fn operands(&self) -> Vec<Value> {
        match self {
            Instruction::Alloca { .. } => vec![],
            Instruction::StackAlloc { size, .. } => vec![*size],
            Instruction::StackSave { .. } => vec![],
            Instruction::StackRestore { ptr, .. } => vec![*ptr],

            Instruction::Load { ptr, .. } => vec![*ptr],

            Instruction::Store { value, ptr, .. } => vec![*value, *ptr],

            Instruction::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::FCmp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::Branch { .. } => vec![],

            Instruction::CondBranch { condition, .. } => vec![*condition],

            Instruction::Switch { value, .. } => vec![*value],

            Instruction::IndirectBranch { target, .. } => vec![*target],

            Instruction::Call { callee, args, .. } => {
                let mut ops = Vec::with_capacity(1 + args.len());
                ops.push(*callee);
                ops.extend_from_slice(args);
                ops
            }

            Instruction::Return { value, .. } => value.map_or_else(Vec::new, |v| vec![v]),

            Instruction::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),

            Instruction::GetElementPtr { base, indices, .. } => {
                let mut ops = Vec::with_capacity(1 + indices.len());
                ops.push(*base);
                ops.extend_from_slice(indices);
                ops
            }

            Instruction::BitCast { value, .. }
            | Instruction::Trunc { value, .. }
            | Instruction::ZExt { value, .. }
            | Instruction::SExt { value, .. }
            | Instruction::IntToPtr { value, .. }
            | Instruction::PtrToInt { value, .. } => vec![*value],

            Instruction::InlineAsm { operands, .. } => operands.clone(),

            Instruction::BlockAddress { .. } => vec![],
        }
    }

    /// Returns mutable references to all [`Value`] operands used by this
    /// instruction.
    ///
    /// This is critical for the SSA renaming pass (mem2reg) which rewrites
    /// operand references in-place as it walks the dominator tree.
    pub fn operands_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Instruction::Alloca { .. } => vec![],
            Instruction::StackAlloc { size, .. } => vec![size],
            Instruction::StackSave { .. } => vec![],
            Instruction::StackRestore { ptr, .. } => vec![ptr],

            Instruction::Load { ptr, .. } => vec![ptr],

            Instruction::Store { value, ptr, .. } => vec![value, ptr],

            Instruction::BinOp { lhs, rhs, .. } => vec![lhs, rhs],

            Instruction::ICmp { lhs, rhs, .. } => vec![lhs, rhs],

            Instruction::FCmp { lhs, rhs, .. } => vec![lhs, rhs],

            Instruction::Branch { .. } => vec![],

            Instruction::CondBranch { condition, .. } => vec![condition],

            Instruction::Switch { value, .. } => vec![value],

            Instruction::IndirectBranch { target, .. } => vec![target],

            Instruction::Call { callee, args, .. } => {
                let mut ops: Vec<&mut Value> = Vec::with_capacity(1 + args.len());
                ops.push(callee);
                for arg in args.iter_mut() {
                    ops.push(arg);
                }
                ops
            }

            Instruction::Return { value, .. } => value.as_mut().map_or_else(Vec::new, |v| vec![v]),

            Instruction::Phi { incoming, .. } => incoming.iter_mut().map(|(v, _)| v).collect(),

            Instruction::GetElementPtr { base, indices, .. } => {
                let mut ops: Vec<&mut Value> = Vec::with_capacity(1 + indices.len());
                ops.push(base);
                for idx in indices.iter_mut() {
                    ops.push(idx);
                }
                ops
            }

            Instruction::BitCast { value, .. }
            | Instruction::Trunc { value, .. }
            | Instruction::ZExt { value, .. }
            | Instruction::SExt { value, .. }
            | Instruction::IntToPtr { value, .. }
            | Instruction::PtrToInt { value, .. } => vec![value],

            Instruction::InlineAsm { operands, .. } => operands.iter_mut().collect(),

            Instruction::BlockAddress { .. } => vec![],
        }
    }

    /// Returns the successor [`BlockId`]s for terminator instructions.
    ///
    /// Non-terminator instructions return an empty vector.  For terminators:
    /// - `Branch` → `[target]`
    /// - `CondBranch` → `[then_block, else_block]`
    /// - `Switch` → `[default, ...case targets]`
    /// - `Return` → `[]` (exits the function)
    ///
    /// For `InlineAsm` with `asm goto`, the `goto_targets` are also
    /// returned as potential successors.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Instruction::Branch { target, .. } => vec![*target],

            Instruction::CondBranch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],

            Instruction::Switch { default, cases, .. } => {
                let mut succs = Vec::with_capacity(1 + cases.len());
                succs.push(*default);
                for (_, blk) in cases {
                    succs.push(*blk);
                }
                succs
            }

            Instruction::IndirectBranch {
                possible_targets, ..
            } => possible_targets.clone(),

            Instruction::InlineAsm { goto_targets, .. } if !goto_targets.is_empty() => {
                goto_targets.clone()
            }

            _ => vec![],
        }
    }
}

// ===========================================================================
// Display implementation for Instruction
// ===========================================================================

impl fmt::Display for Instruction {
    /// Formats the instruction in a human-readable IR text form.
    ///
    /// # Examples
    ///
    /// ```text
    /// %3 = alloca i32
    /// %5 = load i32, ptr %3
    /// store i32 %4, ptr %3
    /// %6 = add i32 %4, %5
    /// %7 = icmp eq %4, %5
    /// br label %bb7
    /// br i1 %2, label %bb3, label %bb4
    /// ret i32 %5
    /// %8 = phi i32 [%1, %bb0], [%5, %bb1]
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Alloca {
                result,
                ty,
                alignment,
                ..
            } => {
                write!(f, "{} = alloca {}", result, ty)?;
                if let Some(align) = alignment {
                    write!(f, ", align {}", align)?;
                }
                Ok(())
            }

            Instruction::StackAlloc { result, size, .. } => {
                write!(f, "{} = stackalloc {}", result, size)
            }

            Instruction::StackSave { result, .. } => {
                write!(f, "{} = stacksave", result)
            }

            Instruction::StackRestore { ptr, .. } => {
                write!(f, "stackrestore {}", ptr)
            }

            Instruction::Load {
                result,
                ptr,
                ty,
                volatile,
                ..
            } => {
                if *volatile {
                    write!(f, "{} = volatile load {}, ptr {}", result, ty, ptr)
                } else {
                    write!(f, "{} = load {}, ptr {}", result, ty, ptr)
                }
            }

            Instruction::Store {
                value,
                ptr,
                volatile,
                ..
            } => {
                if *volatile {
                    write!(f, "volatile store {}, ptr {}", value, ptr)
                } else {
                    write!(f, "store {}, ptr {}", value, ptr)
                }
            }

            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                ..
            } => {
                write!(f, "{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => {
                write!(f, "{} = icmp {} {}, {}", result, op, lhs, rhs)
            }

            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => {
                write!(f, "{} = fcmp {} {}, {}", result, op, lhs, rhs)
            }

            Instruction::Branch { target, .. } => {
                write!(f, "br {}", target)
            }

            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                write!(f, "br i1 {}, {}, {}", condition, then_block, else_block)
            }

            Instruction::Switch {
                value,
                default,
                cases,
                ..
            } => {
                write!(f, "switch {} default {} [", value, default)?;
                for (i, (case_val, target)) in cases.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", case_val, target)?;
                }
                write!(f, "]")
            }

            Instruction::IndirectBranch {
                target,
                possible_targets,
                ..
            } => {
                write!(f, "indirectbr {}, [", target)?;
                for (i, blk) in possible_targets.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", blk)?;
                }
                write!(f, "]")
            }

            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                ..
            } => {
                write!(f, "{} = call {} {}(", result, return_type, callee)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }

            Instruction::Return { value, .. } => match value {
                Some(v) => write!(f, "ret {}", v),
                None => write!(f, "ret void"),
            },

            Instruction::Phi {
                result,
                ty,
                incoming,
                ..
            } => {
                write!(f, "{} = phi {}", result, ty)?;
                for (i, (val, blk)) in incoming.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " [{}, %bb{}]", val, blk.0)?;
                }
                Ok(())
            }

            Instruction::GetElementPtr {
                result,
                base,
                indices,
                result_type,
                in_bounds,
                ..
            } => {
                if *in_bounds {
                    write!(
                        f,
                        "{} = getelementptr inbounds {}, ptr {}",
                        result, result_type, base
                    )?;
                } else {
                    write!(
                        f,
                        "{} = getelementptr {}, ptr {}",
                        result, result_type, base
                    )?;
                }
                for idx in indices {
                    write!(f, ", {}", idx)?;
                }
                Ok(())
            }

            Instruction::BitCast {
                result,
                value,
                to_type,
                ..
            } => {
                write!(f, "{} = bitcast {} to {}", result, value, to_type)
            }

            Instruction::Trunc {
                result,
                value,
                to_type,
                ..
            } => {
                write!(f, "{} = trunc {} to {}", result, value, to_type)
            }

            Instruction::ZExt {
                result,
                value,
                to_type,
                ..
            } => {
                write!(f, "{} = zext {} to {}", result, value, to_type)
            }

            Instruction::SExt {
                result,
                value,
                to_type,
                ..
            } => {
                write!(f, "{} = sext {} to {}", result, value, to_type)
            }

            Instruction::IntToPtr { result, value, .. } => {
                write!(f, "{} = inttoptr {} to ptr", result, value)
            }

            Instruction::PtrToInt {
                result,
                value,
                to_type,
                ..
            } => {
                write!(f, "{} = ptrtoint {} to {}", result, value, to_type)
            }

            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_volatile,
                goto_targets,
                ..
            } => {
                write!(f, "{} = asm", result)?;
                if *is_volatile {
                    write!(f, " volatile")?;
                }
                if *has_side_effects {
                    write!(f, " sideeffect")?;
                }
                if !goto_targets.is_empty() {
                    write!(f, " goto")?;
                }
                write!(f, " \"{}\"", template)?;
                write!(f, ", \"{}\"", constraints)?;
                write!(f, " (")?;
                for (i, op) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", op)?;
                }
                write!(f, ")")?;
                if !clobbers.is_empty() {
                    write!(f, " clobber(")?;
                    for (i, c) in clobbers.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "\"{}\"", c)?;
                    }
                    write!(f, ")")?;
                }
                if !goto_targets.is_empty() {
                    write!(f, " [")?;
                    for (i, tgt) in goto_targets.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", tgt)?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }

            Instruction::BlockAddress { result, block, .. } => {
                write!(f, "{} = blockaddress({})", result, block)
            }
        }
    }
}
