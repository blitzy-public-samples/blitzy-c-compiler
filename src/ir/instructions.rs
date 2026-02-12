//! IR instruction definitions for the BCC compiler.
//!
//! This module defines [`Instruction`], the central enum representing all
//! intermediate representation operations. Every phase of the compilation
//! pipeline after AST lowering works with these instruction types:
//!
//! - **IR lowering** (`src/ir/lowering/`) creates instructions from AST nodes
//! - **SSA construction** (`src/ir/mem2reg/`) inserts and manipulates [`Phi`] nodes
//! - **Optimization passes** (`src/passes/`) analyze and transform instructions
//! - **Phi elimination** (`src/ir/mem2reg/phi_eliminate.rs`) removes phi nodes
//! - **Code generation** (`src/backend/`) lowers instructions to machine code
//!
//! # Instruction Categories
//!
//! | Category        | Variants                                                    |
//! |-----------------|-------------------------------------------------------------|
//! | Memory          | [`Alloca`], [`Load`], [`Store`]                             |
//! | Arithmetic      | [`BinOp`](Instruction::BinOp) (18 binary operations)        |
//! | Comparison      | [`ICmp`] (10 predicates), [`FCmp`] (14 predicates)          |
//! | Control flow    | [`Branch`], [`CondBranch`], [`Switch`], [`Return`]          |
//! | Invocation      | [`Call`]                                                    |
//! | SSA             | [`Phi`]                                                     |
//! | Pointer arith   | [`GetElementPtr`]                                           |
//! | Type conversion | [`BitCast`], [`Trunc`], [`ZExt`], [`SExt`],                 |
//! |                 | [`IntToPtr`], [`PtrToInt`]                                  |
//! | Inline assembly | [`InlineAsm`]                                               |
//!
//! # Alloca-Then-Promote Architecture
//!
//! The [`Alloca`](Instruction::Alloca) instruction is central to the
//! "alloca-then-promote" SSA construction strategy. During Phase 6 (IR
//! lowering), every local variable is initially emitted as an `Alloca`.
//! Phase 7 (`mem2reg`) then promotes eligible allocas — those that are
//! scalar and never have their address taken — to SSA virtual registers,
//! inserting [`Phi`](Instruction::Phi) nodes at dominance frontiers.
//!
//! # Value IDs and Block IDs
//!
//! [`ValueId`] and [`BasicBlockId`] are lightweight handle types defined
//! in this module. They serve as unique identifiers for SSA values and
//! basic blocks respectively, and are used throughout the entire IR layer.
//!
//! # Type Integration
//!
//! Instructions reference [`IrType`] from `crate::ir::types` to annotate
//! operand and result types. The IR type system bridges C language types
//! (from the frontend) to machine types (for the backend). Type variants
//! that flow through instruction fields include:
//!
//! - **Integer types:** [`IrType::I1`], [`IrType::I8`], [`IrType::I16`],
//!   [`IrType::I32`], [`IrType::I64`], [`IrType::I128`]
//! - **Floating-point types:** [`IrType::F32`], [`IrType::F64`], [`IrType::F80`]
//! - **Pointer type:** [`IrType::Ptr`] (opaque, target-width)
//! - **Aggregate types:** [`IrType::Array`], [`IrType::Struct`]
//! - **Special types:** [`IrType::Void`], [`IrType::Function`]

use std::fmt;

use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// ValueId — SSA value identifier
// ---------------------------------------------------------------------------

/// Unique identifier for an SSA value within a function.
///
/// `ValueId` is a lightweight, `Copy`-able handle used to reference virtual
/// registers, constants, and parameters in the IR. Each function maintains
/// its own value numbering namespace — IDs are only meaningful within the
/// function that allocated them.
///
/// The inner `u32` allows up to ~4 billion values per function, which is
/// more than sufficient even for the largest kernel translation units.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Returns the raw numeric index of this value ID.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// BasicBlockId — basic block identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a basic block within a function.
///
/// Like [`ValueId`], this is a lightweight handle. Block IDs are assigned
/// sequentially starting from 0 (the entry block).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
pub struct BasicBlockId(pub u32);

impl BasicBlockId {
    /// Returns the raw numeric index of this block ID.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for BasicBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// BinOp — binary arithmetic/bitwise operations
// ---------------------------------------------------------------------------

/// Binary operation kind for [`Instruction::BinOp`].
///
/// Integer operations (`Add` through `Xor`) expect integer-typed operands
/// (e.g., [`IrType::I32`], [`IrType::I64`]). Floating-point operations
/// (`FAdd` through `FRem`) expect float-typed operands (e.g., [`IrType::F32`],
/// [`IrType::F64`], [`IrType::F80`]). The distinction between signed and
/// unsigned semantics is captured in the operation — e.g., `SDiv` vs `UDiv`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum BinOp {
    // -- Integer arithmetic --
    /// Integer addition (wrapping on overflow).
    Add,
    /// Integer subtraction (wrapping on overflow).
    Sub,
    /// Integer multiplication (wrapping on overflow).
    Mul,
    /// Unsigned integer division. Undefined behavior if divisor is zero.
    UDiv,
    /// Signed integer division. Undefined behavior if divisor is zero
    /// or if the result overflows (e.g., `INT_MIN / -1`).
    SDiv,
    /// Unsigned integer remainder. Undefined behavior if divisor is zero.
    URem,
    /// Signed integer remainder. Undefined behavior if divisor is zero.
    SRem,

    // -- Bitwise shifts --
    /// Left shift. The shift amount must be less than the bit width.
    Shl,
    /// Logical (unsigned) right shift — fills with zeros.
    LShr,
    /// Arithmetic (signed) right shift — fills with sign bit.
    AShr,

    // -- Bitwise logical --
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,

    // -- Floating-point arithmetic --
    /// Floating-point addition.
    FAdd,
    /// Floating-point subtraction.
    FSub,
    /// Floating-point multiplication.
    FMul,
    /// Floating-point division.
    FDiv,
    /// Floating-point remainder (C `fmod` semantics).
    FRem,
}

impl BinOp {
    /// Returns `true` if this is a floating-point operation.
    ///
    /// Floating-point operations require operands of type [`IrType::F32`],
    /// [`IrType::F64`], or [`IrType::F80`].
    #[inline]
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem
        )
    }

    /// Returns `true` if this is an integer operation (including shifts and bitwise).
    ///
    /// Integer operations require operands of type [`IrType::I1`] through
    /// [`IrType::I128`].
    #[inline]
    pub fn is_integer(&self) -> bool {
        !self.is_floating_point()
    }

    /// Returns `true` if the operation is commutative (operand order irrelevant).
    ///
    /// Commutative operations: `Add`, `Mul`, `And`, `Or`, `Xor`, `FAdd`, `FMul`.
    /// Note: floating-point commutativity holds under IEEE 754 but not under
    /// strict associativity (which is not claimed here).
    #[inline]
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            BinOp::Add
                | BinOp::Mul
                | BinOp::And
                | BinOp::Or
                | BinOp::Xor
                | BinOp::FAdd
                | BinOp::FMul
        )
    }

    /// Returns `true` if this is a shift operation (`Shl`, `LShr`, `AShr`).
    #[inline]
    pub fn is_shift(&self) -> bool {
        matches!(self, BinOp::Shl | BinOp::LShr | BinOp::AShr)
    }

    /// Returns `true` if this is a division or remainder operation.
    ///
    /// These operations may trap on division by zero and are not safely
    /// removable without proving the divisor is non-zero.
    #[inline]
    pub fn is_division(&self) -> bool {
        matches!(
            self,
            BinOp::UDiv
                | BinOp::SDiv
                | BinOp::URem
                | BinOp::SRem
                | BinOp::FDiv
                | BinOp::FRem
        )
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::UDiv => "udiv",
            BinOp::SDiv => "sdiv",
            BinOp::URem => "urem",
            BinOp::SRem => "srem",
            BinOp::Shl => "shl",
            BinOp::LShr => "lshr",
            BinOp::AShr => "ashr",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::FAdd => "fadd",
            BinOp::FSub => "fsub",
            BinOp::FMul => "fmul",
            BinOp::FDiv => "fdiv",
            BinOp::FRem => "frem",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// ICmpPredicate — integer comparison predicates
// ---------------------------------------------------------------------------

/// Integer comparison predicate for [`Instruction::ICmp`].
///
/// Equality predicates (`Eq`, `Ne`) work on both signed and unsigned operands.
/// Relational predicates are split into unsigned (`Ugt`, `Uge`, `Ult`, `Ule`)
/// and signed (`Sgt`, `Sge`, `Slt`, `Sle`) variants. The result type of an
/// integer comparison is always [`IrType::I1`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ICmpPredicate {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Unsigned greater than.
    Ugt,
    /// Unsigned greater than or equal.
    Uge,
    /// Unsigned less than.
    Ult,
    /// Unsigned less than or equal.
    Ule,
    /// Signed greater than.
    Sgt,
    /// Signed greater than or equal.
    Sge,
    /// Signed less than.
    Slt,
    /// Signed less than or equal.
    Sle,
}

impl ICmpPredicate {
    /// Returns `true` if this is a signed relational comparison.
    #[inline]
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            ICmpPredicate::Sgt | ICmpPredicate::Sge | ICmpPredicate::Slt | ICmpPredicate::Sle
        )
    }

    /// Returns `true` if this is an unsigned relational comparison.
    #[inline]
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            ICmpPredicate::Ugt | ICmpPredicate::Uge | ICmpPredicate::Ult | ICmpPredicate::Ule
        )
    }

    /// Returns `true` if this is an equality predicate (`Eq` or `Ne`).
    #[inline]
    pub fn is_equality(&self) -> bool {
        matches!(self, ICmpPredicate::Eq | ICmpPredicate::Ne)
    }

    /// Returns the logically negated predicate.
    ///
    /// For example, `Eq` negates to `Ne`, `Slt` negates to `Sge`.
    pub fn negate(&self) -> Self {
        match self {
            ICmpPredicate::Eq => ICmpPredicate::Ne,
            ICmpPredicate::Ne => ICmpPredicate::Eq,
            ICmpPredicate::Ugt => ICmpPredicate::Ule,
            ICmpPredicate::Uge => ICmpPredicate::Ult,
            ICmpPredicate::Ult => ICmpPredicate::Uge,
            ICmpPredicate::Ule => ICmpPredicate::Ugt,
            ICmpPredicate::Sgt => ICmpPredicate::Sle,
            ICmpPredicate::Sge => ICmpPredicate::Slt,
            ICmpPredicate::Slt => ICmpPredicate::Sge,
            ICmpPredicate::Sle => ICmpPredicate::Sgt,
        }
    }

    /// Returns the predicate with operand order swapped.
    ///
    /// `a < b` becomes `b > a`, etc. Equality predicates are unchanged.
    pub fn swap_operands(&self) -> Self {
        match self {
            ICmpPredicate::Eq => ICmpPredicate::Eq,
            ICmpPredicate::Ne => ICmpPredicate::Ne,
            ICmpPredicate::Ugt => ICmpPredicate::Ult,
            ICmpPredicate::Uge => ICmpPredicate::Ule,
            ICmpPredicate::Ult => ICmpPredicate::Ugt,
            ICmpPredicate::Ule => ICmpPredicate::Uge,
            ICmpPredicate::Sgt => ICmpPredicate::Slt,
            ICmpPredicate::Sge => ICmpPredicate::Sle,
            ICmpPredicate::Slt => ICmpPredicate::Sgt,
            ICmpPredicate::Sle => ICmpPredicate::Sge,
        }
    }

    /// Returns the unsigned equivalent of a signed predicate, or self for
    /// equality/unsigned predicates.
    pub fn to_unsigned(&self) -> Self {
        match self {
            ICmpPredicate::Sgt => ICmpPredicate::Ugt,
            ICmpPredicate::Sge => ICmpPredicate::Uge,
            ICmpPredicate::Slt => ICmpPredicate::Ult,
            ICmpPredicate::Sle => ICmpPredicate::Ule,
            other => *other,
        }
    }

    /// Returns the signed equivalent of an unsigned predicate, or self for
    /// equality/signed predicates.
    pub fn to_signed(&self) -> Self {
        match self {
            ICmpPredicate::Ugt => ICmpPredicate::Sgt,
            ICmpPredicate::Uge => ICmpPredicate::Sge,
            ICmpPredicate::Ult => ICmpPredicate::Slt,
            ICmpPredicate::Ule => ICmpPredicate::Sle,
            other => *other,
        }
    }
}

impl fmt::Display for ICmpPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ICmpPredicate::Eq => "eq",
            ICmpPredicate::Ne => "ne",
            ICmpPredicate::Ugt => "ugt",
            ICmpPredicate::Uge => "uge",
            ICmpPredicate::Ult => "ult",
            ICmpPredicate::Ule => "ule",
            ICmpPredicate::Sgt => "sgt",
            ICmpPredicate::Sge => "sge",
            ICmpPredicate::Slt => "slt",
            ICmpPredicate::Sle => "sle",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// FCmpPredicate — floating-point comparison predicates
// ---------------------------------------------------------------------------

/// Floating-point comparison predicate for [`Instruction::FCmp`].
///
/// IEEE 754 floating-point comparisons must account for NaN values:
///
/// - **Ordered** (O-prefix): Returns `false` if either operand is NaN.
/// - **Unordered** (U-prefix): Returns `true` if either operand is NaN.
/// - [`Ord`](FCmpPredicate::Ord): True if neither operand is NaN.
/// - [`Uno`](FCmpPredicate::Uno): True if either operand is NaN.
///
/// The result type is always [`IrType::I1`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum FCmpPredicate {
    /// Ordered and equal.
    OEq,
    /// Ordered and not equal.
    ONe,
    /// Ordered and greater than.
    Ogt,
    /// Ordered and greater than or equal.
    Oge,
    /// Ordered and less than.
    Olt,
    /// Ordered and less than or equal.
    Ole,
    /// Ordered (neither operand is NaN).
    Ord,
    /// Unordered (at least one operand is NaN).
    Uno,
    /// Unordered or equal.
    UEq,
    /// Unordered or not equal.
    UNe,
    /// Unordered or greater than.
    Ugt,
    /// Unordered or greater than or equal.
    Uge,
    /// Unordered or less than.
    Ult,
    /// Unordered or less than or equal.
    Ule,
}

impl FCmpPredicate {
    /// Returns `true` if this is an ordered predicate.
    ///
    /// Ordered predicates return `false` when either operand is NaN.
    #[inline]
    pub fn is_ordered(&self) -> bool {
        matches!(
            self,
            FCmpPredicate::OEq
                | FCmpPredicate::ONe
                | FCmpPredicate::Ogt
                | FCmpPredicate::Oge
                | FCmpPredicate::Olt
                | FCmpPredicate::Ole
                | FCmpPredicate::Ord
        )
    }

    /// Returns `true` if this is an unordered predicate.
    ///
    /// Unordered predicates return `true` when either operand is NaN.
    #[inline]
    pub fn is_unordered(&self) -> bool {
        !self.is_ordered()
    }

    /// Returns the logically negated predicate.
    ///
    /// Negation swaps ordered↔unordered: `OEq` negates to `UNe`, etc.
    /// This correctly accounts for NaN behavior under IEEE 754.
    pub fn negate(&self) -> Self {
        match self {
            FCmpPredicate::OEq => FCmpPredicate::UNe,
            FCmpPredicate::ONe => FCmpPredicate::UEq,
            FCmpPredicate::Ogt => FCmpPredicate::Ule,
            FCmpPredicate::Oge => FCmpPredicate::Ult,
            FCmpPredicate::Olt => FCmpPredicate::Uge,
            FCmpPredicate::Ole => FCmpPredicate::Ugt,
            FCmpPredicate::Ord => FCmpPredicate::Uno,
            FCmpPredicate::Uno => FCmpPredicate::Ord,
            FCmpPredicate::UEq => FCmpPredicate::ONe,
            FCmpPredicate::UNe => FCmpPredicate::OEq,
            FCmpPredicate::Ugt => FCmpPredicate::Ole,
            FCmpPredicate::Uge => FCmpPredicate::Olt,
            FCmpPredicate::Ult => FCmpPredicate::Oge,
            FCmpPredicate::Ule => FCmpPredicate::Ogt,
        }
    }

    /// Returns the predicate with operand order swapped.
    ///
    /// `a < b` becomes `b > a`. Equality and ordering predicates are
    /// symmetric and return themselves.
    pub fn swap_operands(&self) -> Self {
        match self {
            FCmpPredicate::OEq => FCmpPredicate::OEq,
            FCmpPredicate::ONe => FCmpPredicate::ONe,
            FCmpPredicate::Ogt => FCmpPredicate::Olt,
            FCmpPredicate::Oge => FCmpPredicate::Ole,
            FCmpPredicate::Olt => FCmpPredicate::Ogt,
            FCmpPredicate::Ole => FCmpPredicate::Oge,
            FCmpPredicate::Ord => FCmpPredicate::Ord,
            FCmpPredicate::Uno => FCmpPredicate::Uno,
            FCmpPredicate::UEq => FCmpPredicate::UEq,
            FCmpPredicate::UNe => FCmpPredicate::UNe,
            FCmpPredicate::Ugt => FCmpPredicate::Ult,
            FCmpPredicate::Uge => FCmpPredicate::Ule,
            FCmpPredicate::Ult => FCmpPredicate::Ugt,
            FCmpPredicate::Ule => FCmpPredicate::Uge,
        }
    }
}

impl fmt::Display for FCmpPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            FCmpPredicate::OEq => "oeq",
            FCmpPredicate::ONe => "one",
            FCmpPredicate::Ogt => "ogt",
            FCmpPredicate::Oge => "oge",
            FCmpPredicate::Olt => "olt",
            FCmpPredicate::Ole => "ole",
            FCmpPredicate::Ord => "ord",
            FCmpPredicate::Uno => "uno",
            FCmpPredicate::UEq => "ueq",
            FCmpPredicate::UNe => "une",
            FCmpPredicate::Ugt => "ugt",
            FCmpPredicate::Uge => "uge",
            FCmpPredicate::Ult => "ult",
            FCmpPredicate::Ule => "ule",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Instruction — core IR instruction enum
// ---------------------------------------------------------------------------

/// A single IR instruction in the BCC intermediate representation.
///
/// Every basic block contains an ordered list of `Instruction` values.
/// Terminal instructions ([`Branch`](Instruction::Branch),
/// [`CondBranch`](Instruction::CondBranch), [`Switch`](Instruction::Switch),
/// [`Return`](Instruction::Return)) appear only as the last instruction
/// in a block.
///
/// [`Phi`](Instruction::Phi) nodes appear only at the beginning of a block,
/// before any non-phi instructions.
///
/// Most instructions produce a result value identified by [`ValueId`].
/// Instructions that do not produce results (e.g., [`Store`](Instruction::Store),
/// [`Branch`](Instruction::Branch)) return `None` from
/// [`result()`](Instruction::result).
#[derive(Clone, Debug)]
pub enum Instruction {
    /// Allocate stack space for a local variable.
    ///
    /// Central to the "alloca-then-promote" SSA architecture: during
    /// Phase 6 lowering, every local variable is initially an `Alloca`.
    /// Phase 7 (mem2reg) promotes eligible allocas to SSA registers.
    /// The result is always a pointer ([`IrType::Ptr`]) to the allocated
    /// memory of the specified type.
    Alloca {
        /// Result value — a pointer to the allocated memory.
        result: ValueId,
        /// The type of the value to allocate (not the pointer type).
        /// Can be any type: [`IrType::I32`], [`IrType::Struct`], etc.
        ty: IrType,
        /// Alignment requirement in bytes (must be a power of two).
        alignment: u32,
    },

    /// Load a value from memory through a pointer.
    ///
    /// The pointer operand must be of type [`IrType::Ptr`]. The loaded
    /// value will have the type specified in `ty`.
    Load {
        /// Result value — the loaded value.
        result: ValueId,
        /// Pointer operand to load from (must be [`IrType::Ptr`]).
        ptr: ValueId,
        /// Type of the loaded value.
        ty: IrType,
        /// If `true`, this load has volatile semantics (C11 `volatile`).
        /// Volatile loads must not be reordered, removed, or duplicated.
        volatile: bool,
    },

    /// Store a value to memory through a pointer.
    ///
    /// The pointer operand must be of type [`IrType::Ptr`].
    Store {
        /// Value to store.
        value: ValueId,
        /// Pointer to store to (must be [`IrType::Ptr`]).
        ptr: ValueId,
        /// If `true`, this store has volatile semantics (C11 `volatile`).
        /// Volatile stores must not be reordered, removed, or duplicated.
        volatile: bool,
    },

    /// Binary arithmetic or bitwise operation.
    ///
    /// Both operands and the result share the same type.
    BinOp {
        /// Result value.
        result: ValueId,
        /// The specific binary operation.
        op: BinOp,
        /// Left-hand operand.
        lhs: ValueId,
        /// Right-hand operand.
        rhs: ValueId,
        /// Type of operands and result (e.g., [`IrType::I32`], [`IrType::F64`]).
        ty: IrType,
    },

    /// Integer comparison.
    ///
    /// The result type is always [`IrType::I1`] (boolean).
    ICmp {
        /// Result value — always [`IrType::I1`].
        result: ValueId,
        /// Comparison predicate (signed, unsigned, or equality).
        pred: ICmpPredicate,
        /// Left-hand operand.
        lhs: ValueId,
        /// Right-hand operand.
        rhs: ValueId,
    },

    /// Floating-point comparison.
    ///
    /// The result type is always [`IrType::I1`] (boolean). Ordered
    /// predicates return `false` when either operand is NaN; unordered
    /// predicates return `true` when either operand is NaN.
    FCmp {
        /// Result value — always [`IrType::I1`].
        result: ValueId,
        /// Comparison predicate (ordered or unordered).
        pred: FCmpPredicate,
        /// Left-hand operand.
        lhs: ValueId,
        /// Right-hand operand.
        rhs: ValueId,
    },

    /// Unconditional branch (terminator).
    ///
    /// Transfers control to the specified target block. Every basic block
    /// must end with exactly one terminator instruction.
    Branch {
        /// Target basic block.
        target: BasicBlockId,
    },

    /// Conditional branch (terminator).
    ///
    /// Transfers control to `true_target` if `condition` is non-zero,
    /// or to `false_target` if `condition` is zero. The condition must
    /// be of type [`IrType::I1`].
    CondBranch {
        /// Condition value — must be [`IrType::I1`].
        condition: ValueId,
        /// Block to branch to when condition is true (non-zero).
        true_target: BasicBlockId,
        /// Block to branch to when condition is false (zero).
        false_target: BasicBlockId,
    },

    /// Multi-way branch on an integer value (terminator).
    ///
    /// Used to lower C `switch` statements. If the `value` matches a
    /// case constant, control transfers to the corresponding block;
    /// otherwise, control transfers to the `default` block.
    Switch {
        /// Value to switch on (integer type).
        value: ValueId,
        /// Default target when no case matches.
        default: BasicBlockId,
        /// Case value → target block mappings.
        cases: Vec<(i64, BasicBlockId)>,
    },

    /// Function call.
    ///
    /// Invokes the function at `callee` with the given arguments.
    /// If the function returns [`IrType::Void`], `result` is `None`.
    Call {
        /// Result value (`None` for void calls).
        result: Option<ValueId>,
        /// The function to call (a pointer or direct reference).
        callee: ValueId,
        /// Arguments in parameter order.
        args: Vec<ValueId>,
        /// If `true`, this is a tail call eligible for tail-call optimization.
        is_tail: bool,
    },

    /// Return from the current function (terminator).
    ///
    /// Returns the specified value (or void if `value` is `None`).
    /// This is always the final instruction in a function's exit block.
    Return {
        /// Return value (`None` for void return).
        value: Option<ValueId>,
    },

    /// SSA phi node — selects a value based on which predecessor block
    /// transferred control.
    ///
    /// Phi nodes are inserted during SSA construction (mem2reg Phase 7)
    /// and eliminated during Phase 9 (phi elimination). They must appear
    /// at the beginning of a basic block, before any non-phi instruction.
    ///
    /// Each incoming edge associates a value with a predecessor block.
    /// At runtime, the phi evaluates to the value corresponding to the
    /// predecessor that actually transferred control.
    Phi {
        /// Result value.
        result: ValueId,
        /// Type of the phi result and all incoming values.
        ty: IrType,
        /// List of (value, predecessor block) pairs.
        incoming: Vec<(ValueId, BasicBlockId)>,
    },

    /// Compute the address of a sub-element within an aggregate.
    ///
    /// Analogous to LLVM's `getelementptr` — performs pointer arithmetic
    /// to reach array elements and struct fields without loading.
    /// The result is always a pointer ([`IrType::Ptr`]).
    ///
    /// The `ty` field specifies the base element type used for stride
    /// computation (e.g., the element type of an [`IrType::Array`] or
    /// the field layout of an [`IrType::Struct`]).
    GetElementPtr {
        /// Result value — a pointer to the indexed element.
        result: ValueId,
        /// Base pointer.
        base: ValueId,
        /// Index values for each level of nesting.
        indices: Vec<ValueId>,
        /// The base element type (needed for stride computation).
        ty: IrType,
        /// If `true`, undefined behavior if the address is out of bounds.
        in_bounds: bool,
    },

    /// Bitwise reinterpretation cast (no value change, only type change).
    ///
    /// Source and target types must have the same bit-width.
    BitCast {
        /// Result value.
        result: ValueId,
        /// Input value.
        value: ValueId,
        /// Target type (must have the same bit-width as the source).
        to_ty: IrType,
    },

    /// Truncate an integer to a narrower type.
    ///
    /// The target type must be narrower than the source type.
    /// For example, [`IrType::I64`] → [`IrType::I32`].
    Trunc {
        /// Result value.
        result: ValueId,
        /// Input value.
        value: ValueId,
        /// Target type (must be narrower than the source).
        to_ty: IrType,
    },

    /// Zero-extend an integer to a wider type.
    ///
    /// The target type must be wider than the source type. High bits are
    /// filled with zeros. For example, [`IrType::I8`] → [`IrType::I32`].
    ZExt {
        /// Result value.
        result: ValueId,
        /// Input value.
        value: ValueId,
        /// Target type (must be wider than the source).
        to_ty: IrType,
    },

    /// Sign-extend an integer to a wider type.
    ///
    /// The target type must be wider than the source type. High bits are
    /// filled with copies of the sign bit. For example,
    /// [`IrType::I16`] → [`IrType::I64`].
    SExt {
        /// Result value.
        result: ValueId,
        /// Input value.
        value: ValueId,
        /// Target type (must be wider than the source).
        to_ty: IrType,
    },

    /// Convert an integer to a pointer.
    ///
    /// The target type is always [`IrType::Ptr`].
    IntToPtr {
        /// Result value.
        result: ValueId,
        /// Integer input value.
        value: ValueId,
        /// Target pointer type (always [`IrType::Ptr`]).
        to_ty: IrType,
    },

    /// Convert a pointer to an integer.
    ///
    /// The source type must be [`IrType::Ptr`]. The target type determines
    /// the integer width (e.g., [`IrType::I64`] on a 64-bit target).
    PtrToInt {
        /// Result value.
        result: ValueId,
        /// Pointer input value.
        value: ValueId,
        /// Target integer type (e.g., [`IrType::I64`]).
        to_ty: IrType,
    },

    /// Inline assembly statement.
    ///
    /// Encapsulates an entire `asm volatile(...)` construct from the C source.
    /// The assembler template and constraint strings are stored verbatim;
    /// operand binding happens during code generation.
    InlineAsm {
        /// Result value (`None` if no output operand).
        result: Option<ValueId>,
        /// Assembly template string (AT&T syntax).
        template: String,
        /// Constraint string (comma-separated output/input constraints).
        constraints: String,
        /// Operand values bound to constraints.
        operands: Vec<ValueId>,
        /// Clobber registers/flags (e.g., `"memory"`, `"cc"`, `"rax"`).
        clobbers: Vec<String>,
        /// If `true`, the inline assembly has side effects and must not
        /// be removed or reordered by optimizers.
        has_side_effects: bool,
        /// If `true`, the stack must be aligned before execution.
        is_align_stack: bool,
    },
}

// ---------------------------------------------------------------------------
// Instruction — core query methods
// ---------------------------------------------------------------------------

impl Instruction {
    /// Returns the [`ValueId`] produced by this instruction, if any.
    ///
    /// Returns `None` for instructions that do not produce a result:
    /// [`Store`](Instruction::Store), [`Branch`](Instruction::Branch),
    /// [`CondBranch`](Instruction::CondBranch), [`Switch`](Instruction::Switch)
    /// (as a terminator), [`Return`](Instruction::Return), void
    /// [`Call`](Instruction::Call)s, and void [`InlineAsm`](Instruction::InlineAsm).
    pub fn result(&self) -> Option<ValueId> {
        match self {
            Instruction::Alloca { result, .. }
            | Instruction::Load { result, .. }
            | Instruction::BinOp { result, .. }
            | Instruction::ICmp { result, .. }
            | Instruction::FCmp { result, .. }
            | Instruction::Phi { result, .. }
            | Instruction::GetElementPtr { result, .. }
            | Instruction::BitCast { result, .. }
            | Instruction::Trunc { result, .. }
            | Instruction::ZExt { result, .. }
            | Instruction::SExt { result, .. }
            | Instruction::IntToPtr { result, .. }
            | Instruction::PtrToInt { result, .. } => Some(*result),

            Instruction::Call { result, .. } | Instruction::InlineAsm { result, .. } => *result,

            Instruction::Store { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Switch { .. }
            | Instruction::Return { .. } => None,
        }
    }

    /// Returns all [`ValueId`]s read (used) by this instruction.
    ///
    /// This is the set of values that the instruction depends on. The
    /// result value (if any) is NOT included — it is produced, not consumed.
    ///
    /// For [`Phi`](Instruction::Phi), only the incoming values are returned
    /// (not the associated block IDs).
    pub fn uses(&self) -> Vec<ValueId> {
        match self {
            Instruction::Alloca { .. } => Vec::new(),

            Instruction::Load { ptr, .. } => vec![*ptr],

            Instruction::Store { value, ptr, .. } => vec![*value, *ptr],

            Instruction::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::FCmp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::Branch { .. } => Vec::new(),

            Instruction::CondBranch { condition, .. } => vec![*condition],

            Instruction::Switch { value, .. } => vec![*value],

            Instruction::Call { callee, args, .. } => {
                let mut uses = Vec::with_capacity(1 + args.len());
                uses.push(*callee);
                uses.extend_from_slice(args);
                uses
            }

            Instruction::Return { value } => {
                value.map_or_else(Vec::new, |v| vec![v])
            }

            Instruction::Phi { incoming, .. } => {
                incoming.iter().map(|(v, _)| *v).collect()
            }

            Instruction::GetElementPtr {
                base, indices, ..
            } => {
                let mut uses = Vec::with_capacity(1 + indices.len());
                uses.push(*base);
                uses.extend_from_slice(indices);
                uses
            }

            Instruction::BitCast { value, .. }
            | Instruction::Trunc { value, .. }
            | Instruction::ZExt { value, .. }
            | Instruction::SExt { value, .. }
            | Instruction::IntToPtr { value, .. }
            | Instruction::PtrToInt { value, .. } => vec![*value],

            Instruction::InlineAsm { operands, .. } => operands.clone(),
        }
    }

    /// Returns `true` if this instruction is a block terminator.
    ///
    /// Terminators transfer control to another block or exit the function.
    /// Every well-formed basic block must end with exactly one terminator.
    #[inline]
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Branch { .. }
                | Instruction::CondBranch { .. }
                | Instruction::Switch { .. }
                | Instruction::Return { .. }
        )
    }

    /// Returns `true` if this is a [`Phi`](Instruction::Phi) node.
    ///
    /// Phi nodes must appear at the beginning of a basic block, before
    /// any non-phi instructions.
    #[inline]
    pub fn is_phi(&self) -> bool {
        matches!(self, Instruction::Phi { .. })
    }

    /// Returns `true` if this is an [`Alloca`](Instruction::Alloca) instruction.
    ///
    /// Alloca instructions are the initial representation of local variables
    /// in the "alloca-then-promote" SSA construction strategy. The mem2reg
    /// pass identifies promotable allocas and converts them to SSA registers.
    #[inline]
    pub fn is_alloca(&self) -> bool {
        matches!(self, Instruction::Alloca { .. })
    }

    /// Returns `true` if this instruction has volatile semantics.
    ///
    /// Volatile operations must not be reordered, removed, or duplicated
    /// by optimization passes. Returns `true` for:
    /// - Volatile [`Load`](Instruction::Load) operations
    /// - Volatile [`Store`](Instruction::Store) operations
    /// - [`InlineAsm`](Instruction::InlineAsm) with side effects
    #[inline]
    pub fn is_volatile(&self) -> bool {
        match self {
            Instruction::Load { volatile, .. } => *volatile,
            Instruction::Store { volatile, .. } => *volatile,
            Instruction::InlineAsm {
                has_side_effects, ..
            } => *has_side_effects,
            _ => false,
        }
    }

    /// Returns `true` if this instruction has observable side effects.
    ///
    /// Side-effecting instructions cannot be removed by dead code
    /// elimination even if their result is unused. This includes:
    /// - All [`Store`](Instruction::Store) operations
    /// - All [`Call`](Instruction::Call) operations (may modify global state)
    /// - All [`InlineAsm`](Instruction::InlineAsm) operations
    /// - Volatile [`Load`](Instruction::Load) operations
    /// - All terminator instructions (affect control flow)
    pub fn has_side_effects(&self) -> bool {
        match self {
            Instruction::Store { .. } => true,
            Instruction::Call { .. } => true,
            Instruction::InlineAsm { .. } => true,
            Instruction::Load { volatile, .. } => *volatile,
            _ if self.is_terminator() => true,
            _ => false,
        }
    }

    /// Returns the target basic blocks for terminator instructions.
    ///
    /// For non-terminator instructions, returns an empty vector.
    ///
    /// | Terminator    | Successor blocks                          |
    /// |---------------|-------------------------------------------|
    /// | Branch        | `[target]`                                |
    /// | CondBranch    | `[true_target, false_target]`             |
    /// | Switch        | `[default]` ∪ all case targets            |
    /// | Return        | `[]` (exits the function)                 |
    pub fn successor_blocks(&self) -> Vec<BasicBlockId> {
        match self {
            Instruction::Branch { target } => vec![*target],

            Instruction::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],

            Instruction::Switch {
                default, cases, ..
            } => {
                let mut blocks = Vec::with_capacity(1 + cases.len());
                blocks.push(*default);
                for (_, target) in cases {
                    blocks.push(*target);
                }
                blocks
            }

            // Return exits the function — no successor blocks.
            Instruction::Return { .. } => Vec::new(),

            // Non-terminators have no successor blocks.
            _ => Vec::new(),
        }
    }

    /// Returns the IR type associated with this instruction's result, if any.
    ///
    /// This returns the type annotation stored directly in the instruction
    /// variant. For instructions without an explicit type field (e.g., `ICmp`
    /// which always produces [`IrType::I1`]), returns `None`.
    pub fn result_type(&self) -> Option<&IrType> {
        match self {
            Instruction::Alloca { ty, .. } => Some(ty),
            Instruction::Load { ty, .. } => Some(ty),
            Instruction::BinOp { ty, .. } => Some(ty),
            Instruction::Phi { ty, .. } => Some(ty),
            Instruction::GetElementPtr { ty, .. } => Some(ty),
            Instruction::BitCast { to_ty, .. }
            | Instruction::Trunc { to_ty, .. }
            | Instruction::ZExt { to_ty, .. }
            | Instruction::SExt { to_ty, .. }
            | Instruction::IntToPtr { to_ty, .. }
            | Instruction::PtrToInt { to_ty, .. } => Some(to_ty),
            // ICmp and FCmp always produce I1, but don't carry a type field.
            // Call/InlineAsm result types are determined by the callee signature.
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction — SSA manipulation methods
// ---------------------------------------------------------------------------

impl Instruction {
    /// Replaces all occurrences of `old` with `new` in value operands.
    ///
    /// Used during SSA renaming (mem2reg Phase 7) and optimization passes
    /// to rewrite value references when a definition is replaced. Only
    /// _use_ positions are updated — the result position is unchanged.
    pub fn replace_use(&mut self, old: ValueId, new: ValueId) {
        // Helper: replace in-place if the value matches.
        #[inline]
        fn sub(v: &mut ValueId, old: ValueId, new: ValueId) {
            if *v == old {
                *v = new;
            }
        }

        match self {
            Instruction::Alloca { .. } => {
                // Alloca has no use operands.
            }

            Instruction::Load { ptr, .. } => {
                sub(ptr, old, new);
            }

            Instruction::Store { value, ptr, .. } => {
                sub(value, old, new);
                sub(ptr, old, new);
            }

            Instruction::BinOp { lhs, rhs, .. } => {
                sub(lhs, old, new);
                sub(rhs, old, new);
            }

            Instruction::ICmp { lhs, rhs, .. } => {
                sub(lhs, old, new);
                sub(rhs, old, new);
            }

            Instruction::FCmp { lhs, rhs, .. } => {
                sub(lhs, old, new);
                sub(rhs, old, new);
            }

            Instruction::Branch { .. } => {
                // No value operands.
            }

            Instruction::CondBranch { condition, .. } => {
                sub(condition, old, new);
            }

            Instruction::Switch { value, .. } => {
                sub(value, old, new);
            }

            Instruction::Call { callee, args, .. } => {
                sub(callee, old, new);
                for arg in args.iter_mut() {
                    sub(arg, old, new);
                }
            }

            Instruction::Return { value } => {
                if let Some(v) = value.as_mut() {
                    sub(v, old, new);
                }
            }

            Instruction::Phi { incoming, .. } => {
                for (v, _) in incoming.iter_mut() {
                    sub(v, old, new);
                }
            }

            Instruction::GetElementPtr {
                base, indices, ..
            } => {
                sub(base, old, new);
                for idx in indices.iter_mut() {
                    sub(idx, old, new);
                }
            }

            Instruction::BitCast { value, .. }
            | Instruction::Trunc { value, .. }
            | Instruction::ZExt { value, .. }
            | Instruction::SExt { value, .. }
            | Instruction::IntToPtr { value, .. }
            | Instruction::PtrToInt { value, .. } => {
                sub(value, old, new);
            }

            Instruction::InlineAsm { operands, .. } => {
                for op in operands.iter_mut() {
                    sub(op, old, new);
                }
            }
        }
    }

    /// Replaces all occurrences of `old` with `new` in block references.
    ///
    /// Used when splitting, merging, or redirecting basic blocks during
    /// CFG transformations and phi-node updates.
    pub fn replace_block(&mut self, old: BasicBlockId, new: BasicBlockId) {
        // Helper: replace in-place if the block ID matches.
        #[inline]
        fn sub(b: &mut BasicBlockId, old: BasicBlockId, new: BasicBlockId) {
            if *b == old {
                *b = new;
            }
        }

        match self {
            Instruction::Branch { target } => {
                sub(target, old, new);
            }

            Instruction::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                sub(true_target, old, new);
                sub(false_target, old, new);
            }

            Instruction::Switch {
                default, cases, ..
            } => {
                sub(default, old, new);
                for (_, target) in cases.iter_mut() {
                    sub(target, old, new);
                }
            }

            Instruction::Phi { incoming, .. } => {
                for (_, block) in incoming.iter_mut() {
                    sub(block, old, new);
                }
            }

            // All other instruction variants have no block references.
            _ => {}
        }
    }

    /// Updates the result [`ValueId`] of this instruction.
    ///
    /// Used during SSA renaming to assign fresh value numbers. For
    /// [`Call`](Instruction::Call) and [`InlineAsm`](Instruction::InlineAsm),
    /// this sets the `result` field to `Some(new_id)`.
    ///
    /// # Panics
    ///
    /// Panics if the instruction does not produce a result (e.g.,
    /// [`Store`](Instruction::Store), [`Branch`](Instruction::Branch)).
    pub fn set_result(&mut self, new_id: ValueId) {
        match self {
            Instruction::Alloca { result, .. }
            | Instruction::Load { result, .. }
            | Instruction::BinOp { result, .. }
            | Instruction::ICmp { result, .. }
            | Instruction::FCmp { result, .. }
            | Instruction::Phi { result, .. }
            | Instruction::GetElementPtr { result, .. }
            | Instruction::BitCast { result, .. }
            | Instruction::Trunc { result, .. }
            | Instruction::ZExt { result, .. }
            | Instruction::SExt { result, .. }
            | Instruction::IntToPtr { result, .. }
            | Instruction::PtrToInt { result, .. } => {
                *result = new_id;
            }

            Instruction::Call { result, .. } | Instruction::InlineAsm { result, .. } => {
                *result = Some(new_id);
            }

            Instruction::Store { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Switch { .. }
            | Instruction::Return { .. } => {
                panic!(
                    "set_result called on instruction that produces no result: {:?}",
                    std::mem::discriminant(self)
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction — phi node specific methods
// ---------------------------------------------------------------------------

impl Instruction {
    /// Adds an incoming (value, predecessor block) pair to a [`Phi`] node.
    ///
    /// This is used during SSA construction when a new predecessor edge
    /// is discovered or created.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a `Phi` instruction.
    pub fn add_phi_operand(&mut self, value: ValueId, block: BasicBlockId) {
        match self {
            Instruction::Phi { incoming, .. } => {
                incoming.push((value, block));
            }
            _ => panic!(
                "add_phi_operand called on non-Phi instruction: {:?}",
                std::mem::discriminant(self)
            ),
        }
    }

    /// Returns the incoming edge list for a [`Phi`] node, or `None` if
    /// this instruction is not a `Phi`.
    ///
    /// Each entry is a `(value, predecessor_block)` pair indicating what
    /// value the phi evaluates to when control arrives from the given block.
    pub fn phi_operands(&self) -> Option<&[(ValueId, BasicBlockId)]> {
        match self {
            Instruction::Phi { incoming, .. } => Some(incoming.as_slice()),
            _ => None,
        }
    }

    /// Removes all incoming edges from the specified predecessor block.
    ///
    /// This is used when a predecessor block is deleted or redirected.
    /// If no edge from `block` exists, this is a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a `Phi` instruction.
    pub fn remove_phi_operand(&mut self, block: BasicBlockId) {
        match self {
            Instruction::Phi { incoming, .. } => {
                incoming.retain(|(_, b)| *b != block);
            }
            _ => panic!(
                "remove_phi_operand called on non-Phi instruction: {:?}",
                std::mem::discriminant(self)
            ),
        }
    }

    /// Returns a mutable reference to the incoming edge list for a [`Phi`]
    /// node, or `None` if this instruction is not a `Phi`.
    pub fn phi_operands_mut(&mut self) -> Option<&mut Vec<(ValueId, BasicBlockId)>> {
        match self {
            Instruction::Phi { incoming, .. } => Some(incoming),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display implementation for Instruction
// ---------------------------------------------------------------------------

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Alloca {
                result,
                ty,
                alignment,
            } => {
                write!(f, "{} = alloca {}, align {}", result, ty, alignment)
            }

            Instruction::Load {
                result,
                ptr,
                ty,
                volatile,
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
            } => {
                write!(f, "{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            Instruction::ICmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                write!(f, "{} = icmp {} {}, {}", result, pred, lhs, rhs)
            }

            Instruction::FCmp {
                result,
                pred,
                lhs,
                rhs,
            } => {
                write!(f, "{} = fcmp {} {}, {}", result, pred, lhs, rhs)
            }

            Instruction::Branch { target } => {
                write!(f, "br label {}", target)
            }

            Instruction::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                write!(
                    f,
                    "br i1 {}, label {}, label {}",
                    condition, true_target, false_target
                )
            }

            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                write!(f, "switch {} [default: {}", value, default)?;
                for (val, target) in cases {
                    write!(f, ", {}: {}", val, target)?;
                }
                write!(f, "]")
            }

            Instruction::Call {
                result,
                callee,
                args,
                is_tail,
            } => {
                if let Some(r) = result {
                    write!(f, "{} = ", r)?;
                }
                if *is_tail {
                    write!(f, "tail ")?;
                }
                write!(f, "call {}(", callee)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }

            Instruction::Return { value } => {
                if let Some(v) = value {
                    write!(f, "ret {}", v)
                } else {
                    write!(f, "ret void")
                }
            }

            Instruction::Phi {
                result,
                ty,
                incoming,
            } => {
                write!(f, "{} = phi {}", result, ty)?;
                for (i, (val, block)) in incoming.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " [{}, {}]", val, block)?;
                }
                Ok(())
            }

            Instruction::GetElementPtr {
                result,
                base,
                indices,
                ty,
                in_bounds,
            } => {
                write!(f, "{} = getelementptr ", result)?;
                if *in_bounds {
                    write!(f, "inbounds ")?;
                }
                write!(f, "{}, ptr {}", ty, base)?;
                for idx in indices {
                    write!(f, ", {}", idx)?;
                }
                Ok(())
            }

            Instruction::BitCast {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = bitcast {} to {}", result, value, to_ty)
            }

            Instruction::Trunc {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = trunc {} to {}", result, value, to_ty)
            }

            Instruction::ZExt {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = zext {} to {}", result, value, to_ty)
            }

            Instruction::SExt {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = sext {} to {}", result, value, to_ty)
            }

            Instruction::IntToPtr {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = inttoptr {} to {}", result, value, to_ty)
            }

            Instruction::PtrToInt {
                result,
                value,
                to_ty,
            } => {
                write!(f, "{} = ptrtoint {} to {}", result, value, to_ty)
            }

            Instruction::InlineAsm {
                result,
                template,
                constraints,
                operands,
                clobbers,
                has_side_effects,
                is_align_stack,
            } => {
                if let Some(r) = result {
                    write!(f, "{} = ", r)?;
                }
                write!(f, "asm")?;
                if *has_side_effects {
                    write!(f, " volatile")?;
                }
                if *is_align_stack {
                    write!(f, " alignstack")?;
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
                    write!(f, " ~{{")?;
                    for (i, c) in clobbers.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", c)?;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ValueId tests --

    #[test]
    fn test_value_id_equality() {
        let v0 = ValueId(0);
        let v1 = ValueId(1);
        let v0b = ValueId(0);
        assert_eq!(v0, v0b);
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_value_id_display() {
        assert_eq!(format!("{}", ValueId(0)), "%v0");
        assert_eq!(format!("{}", ValueId(42)), "%v42");
    }

    #[test]
    fn test_value_id_ordering() {
        let v0 = ValueId(0);
        let v1 = ValueId(1);
        assert!(v0 < v1);
    }

    #[test]
    fn test_value_id_index() {
        assert_eq!(ValueId(99).index(), 99);
    }

    // -- BasicBlockId tests --

    #[test]
    fn test_basic_block_id_display() {
        assert_eq!(format!("{}", BasicBlockId(0)), "bb0");
        assert_eq!(format!("{}", BasicBlockId(7)), "bb7");
    }

    #[test]
    fn test_basic_block_id_index() {
        assert_eq!(BasicBlockId(5).index(), 5);
    }

    // -- BinOp tests --

    #[test]
    fn test_binop_is_floating_point() {
        assert!(BinOp::FAdd.is_floating_point());
        assert!(BinOp::FSub.is_floating_point());
        assert!(BinOp::FMul.is_floating_point());
        assert!(BinOp::FDiv.is_floating_point());
        assert!(BinOp::FRem.is_floating_point());
        assert!(!BinOp::Add.is_floating_point());
        assert!(!BinOp::Shl.is_floating_point());
        assert!(!BinOp::And.is_floating_point());
    }

    #[test]
    fn test_binop_is_integer() {
        assert!(BinOp::Add.is_integer());
        assert!(BinOp::Sub.is_integer());
        assert!(BinOp::Shl.is_integer());
        assert!(BinOp::And.is_integer());
        assert!(!BinOp::FAdd.is_integer());
    }

    #[test]
    fn test_binop_is_commutative() {
        assert!(BinOp::Add.is_commutative());
        assert!(BinOp::Mul.is_commutative());
        assert!(BinOp::And.is_commutative());
        assert!(BinOp::Or.is_commutative());
        assert!(BinOp::Xor.is_commutative());
        assert!(BinOp::FAdd.is_commutative());
        assert!(BinOp::FMul.is_commutative());
        assert!(!BinOp::Sub.is_commutative());
        assert!(!BinOp::SDiv.is_commutative());
        assert!(!BinOp::Shl.is_commutative());
    }

    #[test]
    fn test_binop_is_shift() {
        assert!(BinOp::Shl.is_shift());
        assert!(BinOp::LShr.is_shift());
        assert!(BinOp::AShr.is_shift());
        assert!(!BinOp::Add.is_shift());
    }

    #[test]
    fn test_binop_is_division() {
        assert!(BinOp::UDiv.is_division());
        assert!(BinOp::SDiv.is_division());
        assert!(BinOp::URem.is_division());
        assert!(BinOp::SRem.is_division());
        assert!(BinOp::FDiv.is_division());
        assert!(BinOp::FRem.is_division());
        assert!(!BinOp::Add.is_division());
        assert!(!BinOp::Mul.is_division());
    }

    #[test]
    fn test_binop_display() {
        assert_eq!(format!("{}", BinOp::Add), "add");
        assert_eq!(format!("{}", BinOp::FMul), "fmul");
        assert_eq!(format!("{}", BinOp::AShr), "ashr");
    }

    // -- ICmpPredicate tests --

    #[test]
    fn test_icmp_is_signed() {
        assert!(ICmpPredicate::Sgt.is_signed());
        assert!(ICmpPredicate::Sge.is_signed());
        assert!(ICmpPredicate::Slt.is_signed());
        assert!(ICmpPredicate::Sle.is_signed());
        assert!(!ICmpPredicate::Eq.is_signed());
        assert!(!ICmpPredicate::Ugt.is_signed());
    }

    #[test]
    fn test_icmp_is_unsigned() {
        assert!(ICmpPredicate::Ugt.is_unsigned());
        assert!(ICmpPredicate::Uge.is_unsigned());
        assert!(ICmpPredicate::Ult.is_unsigned());
        assert!(ICmpPredicate::Ule.is_unsigned());
        assert!(!ICmpPredicate::Eq.is_unsigned());
        assert!(!ICmpPredicate::Sgt.is_unsigned());
    }

    #[test]
    fn test_icmp_negate() {
        assert_eq!(ICmpPredicate::Eq.negate(), ICmpPredicate::Ne);
        assert_eq!(ICmpPredicate::Ne.negate(), ICmpPredicate::Eq);
        assert_eq!(ICmpPredicate::Slt.negate(), ICmpPredicate::Sge);
        assert_eq!(ICmpPredicate::Uge.negate(), ICmpPredicate::Ult);
        // Double negation is identity.
        for pred in &[
            ICmpPredicate::Eq,
            ICmpPredicate::Ne,
            ICmpPredicate::Ugt,
            ICmpPredicate::Uge,
            ICmpPredicate::Ult,
            ICmpPredicate::Ule,
            ICmpPredicate::Sgt,
            ICmpPredicate::Sge,
            ICmpPredicate::Slt,
            ICmpPredicate::Sle,
        ] {
            assert_eq!(pred.negate().negate(), *pred);
        }
    }

    #[test]
    fn test_icmp_swap_operands() {
        assert_eq!(ICmpPredicate::Eq.swap_operands(), ICmpPredicate::Eq);
        assert_eq!(ICmpPredicate::Slt.swap_operands(), ICmpPredicate::Sgt);
        assert_eq!(ICmpPredicate::Ugt.swap_operands(), ICmpPredicate::Ult);
        // Double swap is identity.
        for pred in &[
            ICmpPredicate::Eq,
            ICmpPredicate::Ne,
            ICmpPredicate::Ugt,
            ICmpPredicate::Uge,
            ICmpPredicate::Ult,
            ICmpPredicate::Ule,
            ICmpPredicate::Sgt,
            ICmpPredicate::Sge,
            ICmpPredicate::Slt,
            ICmpPredicate::Sle,
        ] {
            assert_eq!(pred.swap_operands().swap_operands(), *pred);
        }
    }

    #[test]
    fn test_icmp_to_unsigned_signed() {
        assert_eq!(ICmpPredicate::Sgt.to_unsigned(), ICmpPredicate::Ugt);
        assert_eq!(ICmpPredicate::Ugt.to_signed(), ICmpPredicate::Sgt);
        assert_eq!(ICmpPredicate::Eq.to_unsigned(), ICmpPredicate::Eq);
        assert_eq!(ICmpPredicate::Eq.to_signed(), ICmpPredicate::Eq);
    }

    #[test]
    fn test_icmp_display() {
        assert_eq!(format!("{}", ICmpPredicate::Eq), "eq");
        assert_eq!(format!("{}", ICmpPredicate::Slt), "slt");
    }

    // -- FCmpPredicate tests --

    #[test]
    fn test_fcmp_is_ordered() {
        assert!(FCmpPredicate::OEq.is_ordered());
        assert!(FCmpPredicate::ONe.is_ordered());
        assert!(FCmpPredicate::Ord.is_ordered());
        assert!(!FCmpPredicate::Uno.is_ordered());
        assert!(!FCmpPredicate::UEq.is_ordered());
    }

    #[test]
    fn test_fcmp_is_unordered() {
        assert!(FCmpPredicate::Uno.is_unordered());
        assert!(FCmpPredicate::UEq.is_unordered());
        assert!(FCmpPredicate::Ugt.is_unordered());
        assert!(!FCmpPredicate::OEq.is_unordered());
    }

    #[test]
    fn test_fcmp_negate() {
        assert_eq!(FCmpPredicate::OEq.negate(), FCmpPredicate::UNe);
        assert_eq!(FCmpPredicate::UNe.negate(), FCmpPredicate::OEq);
        assert_eq!(FCmpPredicate::Ord.negate(), FCmpPredicate::Uno);
        // Double negation is identity.
        for pred in &[
            FCmpPredicate::OEq,
            FCmpPredicate::ONe,
            FCmpPredicate::Ogt,
            FCmpPredicate::Oge,
            FCmpPredicate::Olt,
            FCmpPredicate::Ole,
            FCmpPredicate::Ord,
            FCmpPredicate::Uno,
            FCmpPredicate::UEq,
            FCmpPredicate::UNe,
            FCmpPredicate::Ugt,
            FCmpPredicate::Uge,
            FCmpPredicate::Ult,
            FCmpPredicate::Ule,
        ] {
            assert_eq!(pred.negate().negate(), *pred);
        }
    }

    #[test]
    fn test_fcmp_swap_operands() {
        assert_eq!(FCmpPredicate::OEq.swap_operands(), FCmpPredicate::OEq);
        assert_eq!(FCmpPredicate::Ogt.swap_operands(), FCmpPredicate::Olt);
        assert_eq!(FCmpPredicate::Uge.swap_operands(), FCmpPredicate::Ule);
        // Double swap is identity.
        for pred in &[
            FCmpPredicate::OEq,
            FCmpPredicate::ONe,
            FCmpPredicate::Ogt,
            FCmpPredicate::Oge,
            FCmpPredicate::Olt,
            FCmpPredicate::Ole,
            FCmpPredicate::Ord,
            FCmpPredicate::Uno,
            FCmpPredicate::UEq,
            FCmpPredicate::UNe,
            FCmpPredicate::Ugt,
            FCmpPredicate::Uge,
            FCmpPredicate::Ult,
            FCmpPredicate::Ule,
        ] {
            assert_eq!(pred.swap_operands().swap_operands(), *pred);
        }
    }

    #[test]
    fn test_fcmp_display() {
        assert_eq!(format!("{}", FCmpPredicate::OEq), "oeq");
        assert_eq!(format!("{}", FCmpPredicate::Uno), "uno");
    }

    // -- Instruction query method tests --

    #[test]
    fn test_alloca_result() {
        let instr = Instruction::Alloca {
            result: ValueId(0),
            ty: IrType::I32,
            alignment: 4,
        };
        assert_eq!(instr.result(), Some(ValueId(0)));
        assert!(instr.uses().is_empty());
        assert!(instr.is_alloca());
        assert!(!instr.is_terminator());
        assert!(!instr.is_phi());
    }

    #[test]
    fn test_load_volatile() {
        let instr = Instruction::Load {
            result: ValueId(1),
            ptr: ValueId(0),
            ty: IrType::I32,
            volatile: true,
        };
        assert!(instr.is_volatile());
        assert!(instr.has_side_effects());
        assert_eq!(instr.uses(), vec![ValueId(0)]);
    }

    #[test]
    fn test_load_non_volatile() {
        let instr = Instruction::Load {
            result: ValueId(1),
            ptr: ValueId(0),
            ty: IrType::I32,
            volatile: false,
        };
        assert!(!instr.is_volatile());
        assert!(!instr.has_side_effects());
    }

    #[test]
    fn test_store_uses() {
        let instr = Instruction::Store {
            value: ValueId(1),
            ptr: ValueId(0),
            volatile: false,
        };
        assert_eq!(instr.result(), None);
        assert_eq!(instr.uses(), vec![ValueId(1), ValueId(0)]);
        assert!(instr.has_side_effects());
    }

    #[test]
    fn test_binop_uses() {
        let instr = Instruction::BinOp {
            result: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        assert_eq!(instr.result(), Some(ValueId(2)));
        assert_eq!(instr.uses(), vec![ValueId(0), ValueId(1)]);
        assert!(!instr.has_side_effects());
    }

    #[test]
    fn test_icmp_result() {
        let instr = Instruction::ICmp {
            result: ValueId(3),
            pred: ICmpPredicate::Slt,
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        assert_eq!(instr.result(), Some(ValueId(3)));
        assert_eq!(instr.uses(), vec![ValueId(0), ValueId(1)]);
    }

    #[test]
    fn test_branch_is_terminator() {
        let instr = Instruction::Branch {
            target: BasicBlockId(1),
        };
        assert!(instr.is_terminator());
        assert_eq!(instr.result(), None);
        assert!(instr.uses().is_empty());
        assert_eq!(instr.successor_blocks(), vec![BasicBlockId(1)]);
    }

    #[test]
    fn test_cond_branch_successors() {
        let instr = Instruction::CondBranch {
            condition: ValueId(0),
            true_target: BasicBlockId(1),
            false_target: BasicBlockId(2),
        };
        assert!(instr.is_terminator());
        assert_eq!(instr.uses(), vec![ValueId(0)]);
        assert_eq!(
            instr.successor_blocks(),
            vec![BasicBlockId(1), BasicBlockId(2)]
        );
    }

    #[test]
    fn test_switch_successors() {
        let instr = Instruction::Switch {
            value: ValueId(0),
            default: BasicBlockId(3),
            cases: vec![
                (0, BasicBlockId(0)),
                (1, BasicBlockId(1)),
                (2, BasicBlockId(2)),
            ],
        };
        assert!(instr.is_terminator());
        assert_eq!(
            instr.successor_blocks(),
            vec![
                BasicBlockId(3),
                BasicBlockId(0),
                BasicBlockId(1),
                BasicBlockId(2)
            ]
        );
    }

    #[test]
    fn test_call_with_result() {
        let instr = Instruction::Call {
            result: Some(ValueId(5)),
            callee: ValueId(0),
            args: vec![ValueId(1), ValueId(2)],
            is_tail: false,
        };
        assert_eq!(instr.result(), Some(ValueId(5)));
        assert_eq!(
            instr.uses(),
            vec![ValueId(0), ValueId(1), ValueId(2)]
        );
        assert!(instr.has_side_effects());
    }

    #[test]
    fn test_call_void() {
        let instr = Instruction::Call {
            result: None,
            callee: ValueId(0),
            args: vec![ValueId(1)],
            is_tail: false,
        };
        assert_eq!(instr.result(), None);
    }

    #[test]
    fn test_return_with_value() {
        let instr = Instruction::Return {
            value: Some(ValueId(0)),
        };
        assert!(instr.is_terminator());
        assert_eq!(instr.uses(), vec![ValueId(0)]);
        assert!(instr.successor_blocks().is_empty());
    }

    #[test]
    fn test_return_void() {
        let instr = Instruction::Return { value: None };
        assert!(instr.is_terminator());
        assert!(instr.uses().is_empty());
    }

    #[test]
    fn test_phi_operations() {
        let mut instr = Instruction::Phi {
            result: ValueId(3),
            ty: IrType::I32,
            incoming: vec![
                (ValueId(0), BasicBlockId(0)),
                (ValueId(1), BasicBlockId(1)),
            ],
        };
        assert!(instr.is_phi());
        assert_eq!(instr.result(), Some(ValueId(3)));
        assert_eq!(instr.uses(), vec![ValueId(0), ValueId(1)]);

        // phi_operands
        let ops = instr.phi_operands().unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0], (ValueId(0), BasicBlockId(0)));

        // add_phi_operand
        instr.add_phi_operand(ValueId(2), BasicBlockId(2));
        assert_eq!(instr.phi_operands().unwrap().len(), 3);

        // remove_phi_operand
        instr.remove_phi_operand(BasicBlockId(1));
        assert_eq!(instr.phi_operands().unwrap().len(), 2);
        assert_eq!(
            instr.phi_operands().unwrap()[0],
            (ValueId(0), BasicBlockId(0))
        );
        assert_eq!(
            instr.phi_operands().unwrap()[1],
            (ValueId(2), BasicBlockId(2))
        );
    }

    #[test]
    fn test_phi_operands_on_non_phi() {
        let instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        assert_eq!(instr.phi_operands(), None);
    }

    #[test]
    fn test_gep_uses() {
        let instr = Instruction::GetElementPtr {
            result: ValueId(5),
            base: ValueId(0),
            indices: vec![ValueId(1), ValueId(2)],
            ty: IrType::I32,
            in_bounds: true,
        };
        assert_eq!(
            instr.uses(),
            vec![ValueId(0), ValueId(1), ValueId(2)]
        );
    }

    #[test]
    fn test_cast_instructions() {
        let casts = vec![
            Instruction::BitCast {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::Ptr,
            },
            Instruction::Trunc {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::I16,
            },
            Instruction::ZExt {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::I64,
            },
            Instruction::SExt {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::I64,
            },
            Instruction::IntToPtr {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::Ptr,
            },
            Instruction::PtrToInt {
                result: ValueId(1),
                value: ValueId(0),
                to_ty: IrType::I64,
            },
        ];
        for instr in &casts {
            assert_eq!(instr.result(), Some(ValueId(1)));
            assert_eq!(instr.uses(), vec![ValueId(0)]);
            assert!(!instr.is_terminator());
            assert!(!instr.has_side_effects());
        }
    }

    #[test]
    fn test_inline_asm() {
        let instr = Instruction::InlineAsm {
            result: Some(ValueId(5)),
            template: "mov $0, $1".to_string(),
            constraints: "=r,r".to_string(),
            operands: vec![ValueId(0)],
            clobbers: vec!["memory".to_string(), "cc".to_string()],
            has_side_effects: true,
            is_align_stack: false,
        };
        assert_eq!(instr.result(), Some(ValueId(5)));
        assert_eq!(instr.uses(), vec![ValueId(0)]);
        assert!(instr.is_volatile());
        assert!(instr.has_side_effects());
    }

    // -- SSA manipulation tests --

    #[test]
    fn test_replace_use_in_binop() {
        let mut instr = Instruction::BinOp {
            result: ValueId(2),
            op: BinOp::Add,
            lhs: ValueId(0),
            rhs: ValueId(1),
            ty: IrType::I32,
        };
        instr.replace_use(ValueId(0), ValueId(10));
        assert_eq!(instr.uses(), vec![ValueId(10), ValueId(1)]);
    }

    #[test]
    fn test_replace_use_in_phi() {
        let mut instr = Instruction::Phi {
            result: ValueId(3),
            ty: IrType::I32,
            incoming: vec![
                (ValueId(0), BasicBlockId(0)),
                (ValueId(1), BasicBlockId(1)),
            ],
        };
        instr.replace_use(ValueId(0), ValueId(10));
        let ops = instr.phi_operands().unwrap();
        assert_eq!(ops[0].0, ValueId(10));
        assert_eq!(ops[1].0, ValueId(1));
    }

    #[test]
    fn test_replace_use_in_call() {
        let mut instr = Instruction::Call {
            result: Some(ValueId(5)),
            callee: ValueId(0),
            args: vec![ValueId(1), ValueId(0), ValueId(2)],
            is_tail: false,
        };
        instr.replace_use(ValueId(0), ValueId(99));
        assert_eq!(
            instr.uses(),
            vec![ValueId(99), ValueId(1), ValueId(99), ValueId(2)]
        );
    }

    #[test]
    fn test_replace_use_in_gep() {
        let mut instr = Instruction::GetElementPtr {
            result: ValueId(5),
            base: ValueId(0),
            indices: vec![ValueId(1), ValueId(0)],
            ty: IrType::I32,
            in_bounds: true,
        };
        instr.replace_use(ValueId(0), ValueId(10));
        assert_eq!(
            instr.uses(),
            vec![ValueId(10), ValueId(1), ValueId(10)]
        );
    }

    #[test]
    fn test_replace_use_in_return() {
        let mut instr = Instruction::Return {
            value: Some(ValueId(0)),
        };
        instr.replace_use(ValueId(0), ValueId(5));
        assert_eq!(instr.uses(), vec![ValueId(5)]);
    }

    #[test]
    fn test_replace_use_in_store() {
        let mut instr = Instruction::Store {
            value: ValueId(0),
            ptr: ValueId(1),
            volatile: false,
        };
        instr.replace_use(ValueId(1), ValueId(10));
        assert_eq!(instr.uses(), vec![ValueId(0), ValueId(10)]);
    }

    #[test]
    fn test_replace_block_in_branch() {
        let mut instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        instr.replace_block(BasicBlockId(0), BasicBlockId(5));
        assert_eq!(instr.successor_blocks(), vec![BasicBlockId(5)]);
    }

    #[test]
    fn test_replace_block_in_cond_branch() {
        let mut instr = Instruction::CondBranch {
            condition: ValueId(0),
            true_target: BasicBlockId(1),
            false_target: BasicBlockId(2),
        };
        instr.replace_block(BasicBlockId(2), BasicBlockId(5));
        assert_eq!(
            instr.successor_blocks(),
            vec![BasicBlockId(1), BasicBlockId(5)]
        );
    }

    #[test]
    fn test_replace_block_in_switch() {
        let mut instr = Instruction::Switch {
            value: ValueId(0),
            default: BasicBlockId(0),
            cases: vec![(1, BasicBlockId(1)), (2, BasicBlockId(0))],
        };
        instr.replace_block(BasicBlockId(0), BasicBlockId(9));
        let succs = instr.successor_blocks();
        assert_eq!(succs[0], BasicBlockId(9));
        assert_eq!(succs[1], BasicBlockId(1));
        assert_eq!(succs[2], BasicBlockId(9));
    }

    #[test]
    fn test_replace_block_in_phi() {
        let mut instr = Instruction::Phi {
            result: ValueId(3),
            ty: IrType::I32,
            incoming: vec![
                (ValueId(0), BasicBlockId(0)),
                (ValueId(1), BasicBlockId(1)),
            ],
        };
        instr.replace_block(BasicBlockId(0), BasicBlockId(5));
        let ops = instr.phi_operands().unwrap();
        assert_eq!(ops[0].1, BasicBlockId(5));
        assert_eq!(ops[1].1, BasicBlockId(1));
    }

    #[test]
    fn test_set_result() {
        let mut instr = Instruction::BinOp {
            result: ValueId(0),
            op: BinOp::Add,
            lhs: ValueId(1),
            rhs: ValueId(2),
            ty: IrType::I32,
        };
        instr.set_result(ValueId(99));
        assert_eq!(instr.result(), Some(ValueId(99)));
    }

    #[test]
    fn test_set_result_on_call() {
        let mut instr = Instruction::Call {
            result: None,
            callee: ValueId(0),
            args: vec![],
            is_tail: false,
        };
        instr.set_result(ValueId(10));
        assert_eq!(instr.result(), Some(ValueId(10)));
    }

    #[test]
    #[should_panic(expected = "set_result called on instruction that produces no result")]
    fn test_set_result_panics_on_store() {
        let mut instr = Instruction::Store {
            value: ValueId(0),
            ptr: ValueId(1),
            volatile: false,
        };
        instr.set_result(ValueId(99));
    }

    #[test]
    #[should_panic(expected = "add_phi_operand called on non-Phi instruction")]
    fn test_add_phi_operand_panics_on_non_phi() {
        let mut instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        instr.add_phi_operand(ValueId(0), BasicBlockId(0));
    }

    #[test]
    #[should_panic(expected = "remove_phi_operand called on non-Phi instruction")]
    fn test_remove_phi_operand_panics_on_non_phi() {
        let mut instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        instr.remove_phi_operand(BasicBlockId(0));
    }

    // -- Display tests --

    #[test]
    fn test_alloca_display() {
        let instr = Instruction::Alloca {
            result: ValueId(0),
            ty: IrType::I32,
            alignment: 4,
        };
        let s = format!("{}", instr);
        assert!(s.contains("alloca"));
        assert!(s.contains("align 4"));
    }

    #[test]
    fn test_load_display() {
        let instr = Instruction::Load {
            result: ValueId(1),
            ptr: ValueId(0),
            ty: IrType::I32,
            volatile: false,
        };
        let s = format!("{}", instr);
        assert!(s.contains("load"));
        assert!(!s.contains("volatile"));

        let volatile_instr = Instruction::Load {
            result: ValueId(1),
            ptr: ValueId(0),
            ty: IrType::I32,
            volatile: true,
        };
        let vs = format!("{}", volatile_instr);
        assert!(vs.contains("volatile load"));
    }

    #[test]
    fn test_branch_display() {
        let instr = Instruction::Branch {
            target: BasicBlockId(1),
        };
        assert_eq!(format!("{}", instr), "br label bb1");
    }

    #[test]
    fn test_return_void_display() {
        let instr = Instruction::Return { value: None };
        assert_eq!(format!("{}", instr), "ret void");
    }

    #[test]
    fn test_return_value_display() {
        let instr = Instruction::Return {
            value: Some(ValueId(0)),
        };
        assert_eq!(format!("{}", instr), "ret %v0");
    }

    #[test]
    fn test_phi_display() {
        let instr = Instruction::Phi {
            result: ValueId(3),
            ty: IrType::I32,
            incoming: vec![
                (ValueId(0), BasicBlockId(0)),
                (ValueId(1), BasicBlockId(1)),
            ],
        };
        let s = format!("{}", instr);
        assert!(s.contains("phi"));
        assert!(s.contains("[%v0, bb0]"));
        assert!(s.contains("[%v1, bb1]"));
    }

    #[test]
    fn test_gep_display() {
        let instr = Instruction::GetElementPtr {
            result: ValueId(5),
            base: ValueId(0),
            indices: vec![ValueId(1)],
            ty: IrType::I32,
            in_bounds: true,
        };
        let s = format!("{}", instr);
        assert!(s.contains("getelementptr"));
        assert!(s.contains("inbounds"));
    }

    #[test]
    fn test_call_display() {
        let instr = Instruction::Call {
            result: Some(ValueId(5)),
            callee: ValueId(0),
            args: vec![ValueId(1), ValueId(2)],
            is_tail: true,
        };
        let s = format!("{}", instr);
        assert!(s.contains("tail"));
        assert!(s.contains("call"));
    }

    // -- result_type tests --

    #[test]
    fn test_result_type_alloca() {
        let instr = Instruction::Alloca {
            result: ValueId(0),
            ty: IrType::I32,
            alignment: 4,
        };
        assert_eq!(instr.result_type(), Some(&IrType::I32));
    }

    #[test]
    fn test_result_type_cast() {
        let instr = Instruction::ZExt {
            result: ValueId(1),
            value: ValueId(0),
            to_ty: IrType::I64,
        };
        assert_eq!(instr.result_type(), Some(&IrType::I64));
    }

    #[test]
    fn test_result_type_none_for_branch() {
        let instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        assert_eq!(instr.result_type(), None);
    }

    // -- Non-terminator successor_blocks returns empty --

    #[test]
    fn test_non_terminator_successor_blocks() {
        let instr = Instruction::BinOp {
            result: ValueId(0),
            op: BinOp::Add,
            lhs: ValueId(1),
            rhs: ValueId(2),
            ty: IrType::I32,
        };
        assert!(instr.successor_blocks().is_empty());
    }

    // -- replace_use no-op on alloca --

    #[test]
    fn test_replace_use_noop_on_alloca() {
        let mut instr = Instruction::Alloca {
            result: ValueId(0),
            ty: IrType::I32,
            alignment: 4,
        };
        instr.replace_use(ValueId(0), ValueId(1));
        // Result is unchanged since replace_use only touches use positions.
        assert_eq!(instr.result(), Some(ValueId(0)));
    }

    // -- replace_block no-op on non-branching instruction --

    #[test]
    fn test_replace_block_noop_on_binop() {
        let mut instr = Instruction::BinOp {
            result: ValueId(0),
            op: BinOp::Add,
            lhs: ValueId(1),
            rhs: ValueId(2),
            ty: IrType::I32,
        };
        // Should be a safe no-op.
        instr.replace_block(BasicBlockId(0), BasicBlockId(1));
        assert_eq!(instr.result(), Some(ValueId(0)));
    }

    // -- InlineAsm replace_use --

    #[test]
    fn test_replace_use_in_inline_asm() {
        let mut instr = Instruction::InlineAsm {
            result: None,
            template: "nop".to_string(),
            constraints: String::new(),
            operands: vec![ValueId(0), ValueId(1), ValueId(0)],
            clobbers: vec![],
            has_side_effects: false,
            is_align_stack: false,
        };
        instr.replace_use(ValueId(0), ValueId(99));
        assert_eq!(
            instr.uses(),
            vec![ValueId(99), ValueId(1), ValueId(99)]
        );
    }

    // -- FCmp uses --

    #[test]
    fn test_fcmp_uses() {
        let instr = Instruction::FCmp {
            result: ValueId(3),
            pred: FCmpPredicate::OEq,
            lhs: ValueId(0),
            rhs: ValueId(1),
        };
        assert_eq!(instr.result(), Some(ValueId(3)));
        assert_eq!(instr.uses(), vec![ValueId(0), ValueId(1)]);
    }

    // -- Switch uses --

    #[test]
    fn test_switch_uses() {
        let instr = Instruction::Switch {
            value: ValueId(7),
            default: BasicBlockId(0),
            cases: vec![(0, BasicBlockId(1))],
        };
        assert_eq!(instr.uses(), vec![ValueId(7)]);
    }

    // -- Volatile store --

    #[test]
    fn test_volatile_store() {
        let instr = Instruction::Store {
            value: ValueId(0),
            ptr: ValueId(1),
            volatile: true,
        };
        assert!(instr.is_volatile());
        assert!(instr.has_side_effects());
    }

    // -- phi_operands_mut --

    #[test]
    fn test_phi_operands_mut() {
        let mut instr = Instruction::Phi {
            result: ValueId(0),
            ty: IrType::I32,
            incoming: vec![(ValueId(1), BasicBlockId(0))],
        };
        if let Some(ops) = instr.phi_operands_mut() {
            ops.push((ValueId(2), BasicBlockId(1)));
        }
        assert_eq!(instr.phi_operands().unwrap().len(), 2);
    }

    #[test]
    fn test_phi_operands_mut_on_non_phi() {
        let mut instr = Instruction::Branch {
            target: BasicBlockId(0),
        };
        assert!(instr.phi_operands_mut().is_none());
    }

    // -- CondBranch replace_use --

    #[test]
    fn test_replace_use_in_cond_branch() {
        let mut instr = Instruction::CondBranch {
            condition: ValueId(0),
            true_target: BasicBlockId(1),
            false_target: BasicBlockId(2),
        };
        instr.replace_use(ValueId(0), ValueId(5));
        assert_eq!(instr.uses(), vec![ValueId(5)]);
    }

    // -- Switch replace_use --

    #[test]
    fn test_replace_use_in_switch() {
        let mut instr = Instruction::Switch {
            value: ValueId(0),
            default: BasicBlockId(0),
            cases: vec![],
        };
        instr.replace_use(ValueId(0), ValueId(5));
        assert_eq!(instr.uses(), vec![ValueId(5)]);
    }

    // -- replace_use in cast instructions --

    #[test]
    fn test_replace_use_in_casts() {
        let mut instr = Instruction::BitCast {
            result: ValueId(1),
            value: ValueId(0),
            to_ty: IrType::Ptr,
        };
        instr.replace_use(ValueId(0), ValueId(10));
        assert_eq!(instr.uses(), vec![ValueId(10)]);
    }

    // -- InlineAsm no side effects --

    #[test]
    fn test_inline_asm_no_side_effects_not_volatile() {
        let instr = Instruction::InlineAsm {
            result: None,
            template: "nop".to_string(),
            constraints: String::new(),
            operands: vec![],
            clobbers: vec![],
            has_side_effects: false,
            is_align_stack: false,
        };
        assert!(!instr.is_volatile());
        // InlineAsm always has_side_effects from the perspective of DCE.
        assert!(instr.has_side_effects());
    }

    // -- ICmp equality check --

    #[test]
    fn test_icmp_is_equality() {
        assert!(ICmpPredicate::Eq.is_equality());
        assert!(ICmpPredicate::Ne.is_equality());
        assert!(!ICmpPredicate::Slt.is_equality());
        assert!(!ICmpPredicate::Ugt.is_equality());
    }

    // -- Store display --

    #[test]
    fn test_store_display() {
        let instr = Instruction::Store {
            value: ValueId(0),
            ptr: ValueId(1),
            volatile: false,
        };
        let s = format!("{}", instr);
        assert!(s.contains("store"));
        assert!(!s.contains("volatile"));

        let v_instr = Instruction::Store {
            value: ValueId(0),
            ptr: ValueId(1),
            volatile: true,
        };
        let vs = format!("{}", v_instr);
        assert!(vs.contains("volatile store"));
    }
}
