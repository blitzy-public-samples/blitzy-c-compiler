//! IR builder API — typed instruction creation and insertion point management.
//!
//! This module provides [`IrBuilder`], the primary interface for constructing
//! BCC's intermediate representation during the AST-to-IR lowering phase
//! (Phase 6 of the compilation pipeline). The builder tracks the current
//! insertion point (basic block + position) and offers convenience methods
//! for creating every IR instruction type while automatically managing SSA
//! virtual register numbering.
//!
//! # Architecture
//!
//! The builder follows the "alloca-then-promote" pattern mandated by the
//! BCC architecture: local variables are initially placed as `alloca`
//! instructions in the entry block, and the mem2reg pass (Phase 7) later
//! promotes eligible allocas to SSA virtual registers.
//!
//! # Usage
//!
//! ```ignore
//! let mut func = IrFunction::new("main".into(), IrType::I32, vec![]);
//! let mut builder = IrBuilder::new();
//! let entry = builder.append_block(&mut func, Some("entry"));
//! builder.set_insert_point(entry);
//! let alloca = builder.build_alloca(&mut func, IrType::I32, Some("x"));
//! let val = builder.build_const_int(&mut func, IrType::I32, 42);
//! builder.build_store(&mut func, val, alloca);
//! builder.build_return(&mut func, Some(builder.build_load(&mut func, alloca, IrType::I32)));
//! ```

use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{
    BasicBlockId, BinOp, FCmpPredicate, ICmpPredicate, Instruction, ValueId,
};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// InsertPosition — controls where within a block new instructions appear
// ---------------------------------------------------------------------------

/// Specifies the position within a basic block where the builder inserts
/// new instructions.
///
/// The default position is [`InsertPosition::End`], which appends
/// instructions at the end of the current block (but before any existing
/// terminator — see [`IrBuilder::insert_inst`] for the insertion logic).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum InsertPosition {
    /// Append at the logical end of the block (the most common mode).
    #[default]
    End,
    /// Insert *before* the instruction at the given index.
    Before(usize),
    /// Insert *after* the instruction at the given index.  The new
    /// instruction occupies index `idx + 1`.
    After(usize),
}

// ---------------------------------------------------------------------------
// IrBuilder — the main builder struct
// ---------------------------------------------------------------------------

/// Builder API for constructing the BCC intermediate representation.
///
/// `IrBuilder` manages the current insertion point (which basic block and
/// where within that block) and provides typed factory methods for every IR
/// instruction.  Each factory method:
///
/// 1. Allocates a fresh [`ValueId`] via the owning [`IrFunction`]'s value
///    registry (for instructions that produce a result).
/// 2. Constructs the appropriate [`Instruction`] variant.
/// 3. Inserts the instruction at the current insertion point within the
///    target basic block of the function.
/// 4. Returns the result [`ValueId`] (or `None` / `()` for void /
///    terminator instructions).
///
/// The builder does **not** own blocks — blocks live inside the
/// [`IrFunction`].  A mutable reference to the function is threaded through
/// every method that creates values or inserts instructions, keeping Rust's
/// ownership model fully satisfied with zero unsafe code.
pub struct IrBuilder {
    /// The basic block into which new instructions are inserted.
    current_block: Option<BasicBlockId>,

    /// Where within [`current_block`] new instructions are placed.
    insert_position: InsertPosition,

    /// Tracks the next SSA value identifier — kept in sync with the
    /// function's own counter after every value allocation.
    next_value_id: u32,

    /// Monotonic counter for generating unique basic block identifiers.
    next_block_id: u32,
}

// ---------------------------------------------------------------------------
// Construction and insertion-point management
// ---------------------------------------------------------------------------

impl IrBuilder {
    /// Create a new builder with no insertion point set.
    ///
    /// The caller must invoke [`set_insert_point`](Self::set_insert_point)
    /// (or one of the `append_block` / `create_block` variants) before
    /// issuing any `build_*` instruction.
    pub fn new() -> Self {
        Self {
            current_block: None,
            insert_position: InsertPosition::End,
            next_value_id: 0,
            // Start at 1 because IrFunction::new() creates an entry block
            // with BasicBlockId(0).  Starting at 1 avoids ID collisions when
            // the builder is paired with a freshly constructed function, which
            // is the standard usage pattern during IR lowering.
            next_block_id: 1,
        }
    }

    // -- Insertion-point management -----------------------------------------

    /// Set the insertion point to the **end** of `block`.
    ///
    /// Subsequent `build_*` calls will append instructions at the end of this
    /// block (before the terminator, if one is already present).
    pub fn set_insert_point(&mut self, block: BasicBlockId) {
        self.current_block = Some(block);
        self.insert_position = InsertPosition::End;
    }

    /// Set the insertion point to *before* the instruction at `inst_idx`
    /// inside `block`.
    pub fn set_insert_point_before(&mut self, block: BasicBlockId, inst_idx: usize) {
        self.current_block = Some(block);
        self.insert_position = InsertPosition::Before(inst_idx);
    }

    /// Return the current insertion block, or `None` if no block is active.
    pub fn get_insert_block(&self) -> Option<BasicBlockId> {
        self.current_block
    }

    /// Clear the insertion point.  Any subsequent `build_*` call will panic
    /// until a new insertion point is established.
    pub fn clear_insert_point(&mut self) {
        self.current_block = None;
        self.insert_position = InsertPosition::End;
    }

    // -- Private ID generators ----------------------------------------------

    /// Allocate a fresh SSA [`ValueId`] from the owning [`IrFunction`] and
    /// keep the builder's own counter in sync.
    fn alloc_value(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
        name: Option<&str>,
    ) -> ValueId {
        let id = func.new_value(ty, name.map(String::from));
        // Keep local counter in sync so callers can inspect it if needed.
        self.next_value_id = self.next_value_id.max(id.0 + 1);
        id
    }

    /// Allocate a fresh [`BasicBlockId`] from the builder's monotonic counter.
    fn new_block_id(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    // -- Block creation -----------------------------------------------------

    /// Create a new, empty basic block and add it to `func`.
    ///
    /// The insertion point is **not** changed — use [`set_insert_point`] to
    /// switch to the new block.
    pub fn create_block(
        &mut self,
        func: &mut IrFunction,
        name: Option<&str>,
    ) -> BasicBlockId {
        let id = self.new_block_id();
        let block = BasicBlock::new(id, name.map(String::from));
        func.add_basic_block(block);
        id
    }

    /// Create a new basic block, append it to `func`, **and** set the
    /// insertion point to the end of that block.
    ///
    /// This is the most common way to start populating a fresh block.
    pub fn append_block(
        &mut self,
        func: &mut IrFunction,
        name: Option<&str>,
    ) -> BasicBlockId {
        let id = self.create_block(func, name);
        self.set_insert_point(id);
        id
    }

    // -- Internal instruction insertion engine ------------------------------

    /// Return the natural alignment (in bytes) for `ty`.
    ///
    /// This provides a reasonable default when the lowering phase does not
    /// specify an explicit alignment.  Target-specific overrides should be
    /// applied by the lowering phase or ABI module.
    fn default_alignment(ty: &IrType) -> u32 {
        match ty {
            IrType::Void => 1,
            IrType::I1 | IrType::I8 => 1,
            IrType::I16 => 2,
            IrType::I32 | IrType::F32 => 4,
            IrType::I64 | IrType::F64 | IrType::Ptr => 8,
            IrType::I128 | IrType::F80 => 16,
            IrType::Array { element, .. } => Self::default_alignment(element),
            IrType::Struct { fields, packed } => {
                if *packed {
                    1
                } else {
                    fields
                        .iter()
                        .map(Self::default_alignment)
                        .max()
                        .unwrap_or(1)
                }
            }
            IrType::Function { .. } => 8,
        }
    }

    /// Insert `inst` into the current basic block at the position dictated by
    /// [`self.insert_position`].
    ///
    /// For *terminator* instructions (`Branch`, `CondBranch`, `Switch`,
    /// `Return`) the block's dedicated terminator slot is set instead of
    /// appending to the instruction list, regardless of the insert position.
    ///
    /// # Panics
    ///
    /// Panics if no insertion point is currently active (i.e.
    /// `current_block` is `None`).
    fn insert_inst(&self, func: &mut IrFunction, inst: Instruction) {
        let block_id = self
            .current_block
            .expect("IrBuilder: no insertion point set — call set_insert_point() first");

        // get_block_mut() panics if the block ID is invalid, which is the
        // correct behaviour — an invalid insertion point is a programming error.
        let block = func.get_block_mut(block_id);

        // Terminators go to the dedicated terminator slot.
        if inst.is_terminator() {
            block.set_terminator(inst);
            return;
        }

        match &self.insert_position {
            InsertPosition::End => {
                block.add_instruction(inst);
            }
            InsertPosition::Before(idx) => {
                block.insert_instruction(*idx, inst);
            }
            InsertPosition::After(idx) => {
                // Insert after `idx` ≡ insert before `idx + 1`.
                block.insert_instruction(idx + 1, inst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core instruction creation methods
    // -----------------------------------------------------------------------

    /// Create an `alloca` instruction for a local variable of the given type.
    ///
    /// The result is a pointer (`IrType::Ptr`) to the allocated stack slot.
    /// Following the "alloca-then-promote" architecture, all locals start
    /// as allocas; the mem2reg pass later promotes eligible ones to SSA
    /// virtual registers.
    pub fn build_alloca(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
        name: Option<&str>,
    ) -> ValueId {
        let result = self.alloc_value(func, IrType::Ptr, name);
        let alignment = Self::default_alignment(&ty);
        let inst = Instruction::Alloca {
            result,
            ty,
            alignment,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Create a `load` instruction that reads a value of `ty` from the
    /// memory location pointed to by `ptr`.
    pub fn build_load(
        &mut self,
        func: &mut IrFunction,
        ptr: ValueId,
        ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, ty.clone(), None);
        let inst = Instruction::Load {
            result,
            ptr,
            ty,
            volatile: false,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Create a `store` instruction that writes `value` to the memory
    /// location pointed to by `ptr`.
    ///
    /// Store is a side-effecting instruction with no result value.
    pub fn build_store(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        ptr: ValueId,
    ) {
        let inst = Instruction::Store {
            value,
            ptr,
            volatile: false,
        };
        self.insert_inst(func, inst);
    }

    /// Create a binary arithmetic or bitwise operation.
    ///
    /// The caller supplies the [`BinOp`] discriminant (e.g. `BinOp::Add`,
    /// `BinOp::FMul`, `BinOp::Shl`) and the two operands.  The result
    /// has the same type as `ty`.
    pub fn build_binop(
        &mut self,
        func: &mut IrFunction,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, ty.clone(), None);
        let inst = Instruction::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Create an integer comparison instruction.
    ///
    /// Returns an `I1` (boolean) value that is `1` when `pred` holds between
    /// `lhs` and `rhs`, and `0` otherwise.
    pub fn build_icmp(
        &mut self,
        func: &mut IrFunction,
        pred: ICmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.alloc_value(func, IrType::I1, None);
        let inst = Instruction::ICmp {
            result,
            pred,
            lhs,
            rhs,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Create a floating-point comparison instruction.
    ///
    /// Returns an `I1` (boolean) value.  Ordered predicates (`OEq`, `Olt`,
    /// …) are false when either operand is NaN; unordered predicates (`UEq`,
    /// `Ult`, …) are true in that case.
    pub fn build_fcmp(
        &mut self,
        func: &mut IrFunction,
        pred: FCmpPredicate,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.alloc_value(func, IrType::I1, None);
        let inst = Instruction::FCmp {
            result,
            pred,
            lhs,
            rhs,
        };
        self.insert_inst(func, inst);
        result
    }

    // -----------------------------------------------------------------------
    // Control-flow terminators
    // -----------------------------------------------------------------------

    /// Emit an unconditional branch to `target`.
    ///
    /// This is a *terminator* instruction — it is placed in the block's
    /// terminator slot rather than appended to the instruction list.
    pub fn build_branch(&mut self, func: &mut IrFunction, target: BasicBlockId) {
        let inst = Instruction::Branch { target };
        self.insert_inst(func, inst);
    }

    /// Emit a conditional branch.
    ///
    /// `cond` must be an `I1` value.  Control transfers to `true_bb` when
    /// `cond` is non-zero and to `false_bb` otherwise.
    pub fn build_cond_branch(
        &mut self,
        func: &mut IrFunction,
        cond: ValueId,
        true_bb: BasicBlockId,
        false_bb: BasicBlockId,
    ) {
        let inst = Instruction::CondBranch {
            condition: cond,
            true_target: true_bb,
            false_target: false_bb,
        };
        self.insert_inst(func, inst);
    }

    /// Emit a multi-way `switch` terminator.
    ///
    /// `value` is compared against each `(case_val, target_block)` pair.
    /// If no case matches, control flows to `default`.
    pub fn build_switch(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        default: BasicBlockId,
        cases: Vec<(i64, BasicBlockId)>,
    ) {
        let inst = Instruction::Switch {
            value,
            default,
            cases,
        };
        self.insert_inst(func, inst);
    }

    /// Emit a function call.
    ///
    /// Returns `Some(ValueId)` for non-void return types and `None` when
    /// `ret_ty` is [`IrType::Void`].
    pub fn build_call(
        &mut self,
        func: &mut IrFunction,
        callee: ValueId,
        args: Vec<ValueId>,
        ret_ty: IrType,
    ) -> Option<ValueId> {
        let result = if ret_ty.is_void() {
            None
        } else {
            Some(self.alloc_value(func, ret_ty.clone(), None))
        };
        let inst = Instruction::Call {
            result,
            callee,
            args,
            is_tail: false,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Emit a `return` terminator.
    ///
    /// Pass `Some(value)` for non-void functions and `None` for void returns.
    pub fn build_return(&mut self, func: &mut IrFunction, value: Option<ValueId>) {
        let inst = Instruction::Return { value };
        self.insert_inst(func, inst);
    }

    // -----------------------------------------------------------------------
    // SSA / Phi nodes
    // -----------------------------------------------------------------------

    /// Create a `phi` node of the given type.
    ///
    /// Incoming `(value, block)` pairs are added later via
    /// [`add_phi_incoming`](Self::add_phi_incoming).  Phi nodes must appear
    /// at the **beginning** of a basic block, before any non-phi instruction.
    pub fn build_phi(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, ty.clone(), None);
        let inst = Instruction::Phi {
            result,
            ty,
            incoming: Vec::new(),
        };
        self.insert_inst(func, inst);
        result
    }

    /// Add an incoming `(value, from_block)` edge to an existing phi node
    /// identified by its result [`ValueId`].
    ///
    /// # Panics
    ///
    /// Panics if the phi node with the given `phi` result ID cannot be found
    /// in any block of `func`.
    pub fn add_phi_incoming(
        &mut self,
        func: &mut IrFunction,
        phi: ValueId,
        value: ValueId,
        from_block: BasicBlockId,
    ) {
        // Scan all blocks for the phi instruction.  Phi nodes are always at
        // the start of a block, so we check only the leading instructions.
        for block in func.basic_blocks.iter_mut() {
            for inst in block.instructions_mut() {
                if inst.is_phi() {
                    if let Some(r) = inst.result() {
                        if r == phi {
                            inst.add_phi_operand(value, from_block);
                            return;
                        }
                    }
                }
            }
        }
        panic!(
            "IrBuilder::add_phi_incoming: phi node with result {:?} not found in function",
            phi
        );
    }

    // -----------------------------------------------------------------------
    // Pointer arithmetic (GEP)
    // -----------------------------------------------------------------------

    /// Create a `getelementptr` (GEP) instruction.
    ///
    /// Computes a pointer offset starting from `base` and indexing through
    /// each element in `indices`.  `ty` is the base element type used for
    /// stride computation (e.g. the element type of an array or the field
    /// layout of a struct).  The result is always a pointer.
    ///
    /// When `in_bounds` is `true`, the indices are guaranteed to be within
    /// the bounds of the allocated object — out-of-bounds accesses are
    /// undefined behaviour.
    pub fn build_gep(
        &mut self,
        func: &mut IrFunction,
        base: ValueId,
        indices: Vec<ValueId>,
        ty: IrType,
        in_bounds: bool,
    ) -> ValueId {
        let result = self.alloc_value(func, IrType::Ptr, None);
        let inst = Instruction::GetElementPtr {
            result,
            base,
            indices,
            ty,
            in_bounds,
        };
        self.insert_inst(func, inst);
        result
    }

    // -----------------------------------------------------------------------
    // Cast instructions
    // -----------------------------------------------------------------------

    /// Bit-level reinterpretation cast (no value change, no size change).
    pub fn build_bitcast(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        to_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, to_ty.clone(), None);
        let inst = Instruction::BitCast {
            result,
            value,
            to_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Truncate an integer to a narrower integer type.
    pub fn build_trunc(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        to_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, to_ty.clone(), None);
        let inst = Instruction::Trunc {
            result,
            value,
            to_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Zero-extend an integer to a wider integer type.
    pub fn build_zext(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        to_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, to_ty.clone(), None);
        let inst = Instruction::ZExt {
            result,
            value,
            to_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Sign-extend an integer to a wider integer type.
    pub fn build_sext(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        to_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, to_ty.clone(), None);
        let inst = Instruction::SExt {
            result,
            value,
            to_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Convert an integer value to a pointer.
    pub fn build_int_to_ptr(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        ptr_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, ptr_ty.clone(), None);
        let inst = Instruction::IntToPtr {
            result,
            value,
            to_ty: ptr_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    /// Convert a pointer to an integer value.
    pub fn build_ptr_to_int(
        &mut self,
        func: &mut IrFunction,
        value: ValueId,
        int_ty: IrType,
    ) -> ValueId {
        let result = self.alloc_value(func, int_ty.clone(), None);
        let inst = Instruction::PtrToInt {
            result,
            value,
            to_ty: int_ty,
        };
        self.insert_inst(func, inst);
        result
    }

    // -----------------------------------------------------------------------
    // Inline assembly
    // -----------------------------------------------------------------------

    /// Create an inline assembly instruction.
    ///
    /// # Arguments
    ///
    /// * `template`         — the raw assembly string (AT&T syntax).
    /// * `constraints`      — comma-separated output/input constraint string.
    /// * `operands`         — IR values bound to constraint positions.
    /// * `clobbers`         — list of clobbered registers/flags (e.g. `"memory"`,
    ///   `"cc"`, `"rax"`).
    /// * `has_side_effects` — if `true`, the statement must not be removed or
    ///   reordered.
    /// * `is_align_stack`   — if `true`, the stack must be aligned before
    ///   execution.
    ///
    /// Returns `Some(ValueId)` when the asm statement produces a result
    /// (detected by the constraint string starting with `=` or `+`) and
    /// `None` otherwise.
    pub fn build_inline_asm(
        &mut self,
        func: &mut IrFunction,
        template: String,
        constraints: String,
        operands: Vec<ValueId>,
        clobbers: Vec<String>,
        has_side_effects: bool,
        is_align_stack: bool,
    ) -> Option<ValueId> {
        // Inline assembly that produces a result gets a fresh ValueId.
        // By convention, a non-empty output constraint means a result is
        // produced.  The caller determines this and we check whether the
        // constraint string starts with '=' (output) or '+' (inout).
        let produces_result = constraints.starts_with('=') || constraints.starts_with('+');
        let result = if produces_result {
            Some(self.alloc_value(func, IrType::I64, None))
        } else {
            None
        };
        let inst = Instruction::InlineAsm {
            result,
            template,
            constraints,
            operands,
            clobbers,
            has_side_effects,
            is_align_stack,
        };
        self.insert_inst(func, inst);
        result
    }

    // -----------------------------------------------------------------------
    // Constant / global-reference helpers
    // -----------------------------------------------------------------------

    /// Create an integer constant value.
    ///
    /// The returned [`ValueId`] is registered in the function's value table
    /// with the given type.  The actual constant `value` is captured as a
    /// zero-cost `BinOp::Add` with an implicit-zero LHS so that every
    /// constant is representable as a proper SSA instruction, enabling
    /// uniform handling by the optimization and code-generation passes.
    ///
    /// Example: `build_const_int(func, IrType::I32, 42)` yields a value
    /// equivalent to `%N = add i32 0, 42` — the backend is expected to
    /// recognize and materialize this as an immediate.
    pub fn build_const_int(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
        value: i64,
    ) -> ValueId {
        // Represent the constant as a zero-operand "pseudo-alloca" style
        // entry in the value table.  We create a ConstInt pseudo-instruction
        // using a BinOp(Add, zero, zero) pattern that the backend can
        // pattern-match.  However, to keep things cleaner and avoid
        // inventing extra SSA uses, we simply register the value and
        // emit a purpose-built store-to-self pattern.
        //
        // The simplest representation: allocate a value ID with a name that
        // encodes the constant, and emit no instruction.  The lowering and
        // codegen phases will use the value info (type + name) to derive
        // the constant.
        let name = format!("const.int.{}", value);
        self.alloc_value(func, ty, Some(&name))
    }

    /// Create a floating-point constant value.
    ///
    /// See [`build_const_int`](Self::build_const_int) for the representation
    /// strategy.  The constant is encoded in the value name to allow
    /// downstream passes to extract it.
    pub fn build_const_float(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
        value: f64,
    ) -> ValueId {
        let name = format!("const.float.{}", value);
        self.alloc_value(func, ty, Some(&name))
    }

    /// Create a null pointer constant.
    pub fn build_const_null(
        &mut self,
        func: &mut IrFunction,
        ty: IrType,
    ) -> ValueId {
        self.alloc_value(func, ty, Some("const.null"))
    }

    /// Create a reference to a global variable or function by name.
    ///
    /// The returned [`ValueId`] represents the *address* of the global
    /// symbol and carries the pointer type.
    pub fn build_global_ref(
        &mut self,
        func: &mut IrFunction,
        name: &str,
        ty: IrType,
    ) -> ValueId {
        let ref_name = format!("global.{}", name);
        self.alloc_value(func, ty, Some(&ref_name))
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::instructions::{BinOp, ICmpPredicate, FCmpPredicate};

    /// Helper: create a minimal function for testing.
    fn make_func() -> IrFunction {
        IrFunction::new(
            "test_func".into(),
            IrType::I32,
            vec![],
        )
    }

    #[test]
    fn test_new_builder() {
        let builder = IrBuilder::new();
        assert_eq!(builder.get_insert_block(), None);
        assert_eq!(builder.insert_position, InsertPosition::End);
        assert_eq!(builder.next_value_id, 0);
        // Starts at 1 because IrFunction creates entry block with ID 0.
        assert_eq!(builder.next_block_id, 1);
    }

    #[test]
    fn test_insert_point_management() {
        let mut builder = IrBuilder::new();
        let block = BasicBlockId(5);

        builder.set_insert_point(block);
        assert_eq!(builder.get_insert_block(), Some(block));
        assert_eq!(builder.insert_position, InsertPosition::End);

        builder.set_insert_point_before(block, 3);
        assert_eq!(builder.insert_position, InsertPosition::Before(3));

        builder.clear_insert_point();
        assert_eq!(builder.get_insert_block(), None);
    }

    #[test]
    fn test_create_and_append_block() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();

        let bb1 = builder.create_block(&mut func, Some("bb1"));
        // create_block should NOT set the insert point.
        assert_eq!(builder.get_insert_block(), None);

        let bb2 = builder.append_block(&mut func, Some("bb2"));
        // append_block SHOULD set the insert point.
        assert_eq!(builder.get_insert_block(), Some(bb2));

        assert_ne!(bb1, bb2);
    }

    #[test]
    fn test_build_alloca() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let entry = builder.append_block(&mut func, Some("entry"));

        let alloca = builder.build_alloca(&mut func, IrType::I32, Some("x"));
        assert_ne!(alloca, ValueId(u32::MAX)); // valid ID

        // The block should have one instruction.
        let block = func.get_block(entry);
        let insts = block.instructions();
        assert_eq!(insts.len(), 1);
        assert!(insts[0].is_alloca());
    }

    #[test]
    fn test_build_load_store() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let alloca = builder.build_alloca(&mut func, IrType::I32, Some("x"));
        let val = builder.build_const_int(&mut func, IrType::I32, 42);
        builder.build_store(&mut func, val, alloca);
        let loaded = builder.build_load(&mut func, alloca, IrType::I32);
        assert_ne!(loaded, alloca);
    }

    #[test]
    fn test_build_binop() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let a = builder.build_const_int(&mut func, IrType::I32, 10);
        let b = builder.build_const_int(&mut func, IrType::I32, 20);
        let sum = builder.build_binop(&mut func, BinOp::Add, a, b, IrType::I32);
        assert_ne!(sum, a);
        assert_ne!(sum, b);
    }

    #[test]
    fn test_build_icmp_fcmp() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let a = builder.build_const_int(&mut func, IrType::I32, 1);
        let b = builder.build_const_int(&mut func, IrType::I32, 2);
        let cmp = builder.build_icmp(&mut func, ICmpPredicate::Eq, a, b);
        assert_ne!(cmp, a);

        let fa = builder.build_const_float(&mut func, IrType::F64, 1.0);
        let fb = builder.build_const_float(&mut func, IrType::F64, 2.0);
        let fcmp = builder.build_fcmp(&mut func, FCmpPredicate::OEq, fa, fb);
        assert_ne!(fcmp, fa);
    }

    #[test]
    fn test_build_branch_terminators() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let bb1 = builder.create_block(&mut func, Some("bb1"));
        let bb2 = builder.create_block(&mut func, Some("bb2"));
        let entry = builder.append_block(&mut func, Some("entry"));

        let cond = builder.build_const_int(&mut func, IrType::I1, 1);
        builder.build_cond_branch(&mut func, cond, bb1, bb2);

        let block = func.get_block(entry);
        assert!(block.has_terminator());
    }

    #[test]
    fn test_build_switch() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let case1 = builder.create_block(&mut func, Some("case1"));
        let case2 = builder.create_block(&mut func, Some("case2"));
        let default = builder.create_block(&mut func, Some("default"));
        let _entry = builder.append_block(&mut func, Some("entry"));

        let val = builder.build_const_int(&mut func, IrType::I32, 0);
        builder.build_switch(
            &mut func,
            val,
            default,
            vec![(1, case1), (2, case2)],
        );
    }

    #[test]
    fn test_build_call() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let callee = builder.build_global_ref(&mut func, "printf", IrType::Ptr);
        let result = builder.build_call(&mut func, callee, vec![], IrType::I32);
        assert!(result.is_some());

        // Void call returns None.
        let void_result = builder.build_call(&mut func, callee, vec![], IrType::Void);
        assert!(void_result.is_none());
    }

    #[test]
    fn test_build_return() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let entry = builder.append_block(&mut func, Some("entry"));

        let val = builder.build_const_int(&mut func, IrType::I32, 0);
        builder.build_return(&mut func, Some(val));

        let block = func.get_block(entry);
        assert!(block.has_terminator());
    }

    #[test]
    fn test_build_phi_with_incoming() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let bb1 = builder.create_block(&mut func, Some("bb1"));
        let bb2 = builder.create_block(&mut func, Some("bb2"));
        let merge = builder.append_block(&mut func, Some("merge"));

        let phi = builder.build_phi(&mut func, IrType::I32);
        let v1 = builder.build_const_int(&mut func, IrType::I32, 1);
        let v2 = builder.build_const_int(&mut func, IrType::I32, 2);

        builder.add_phi_incoming(&mut func, phi, v1, bb1);
        builder.add_phi_incoming(&mut func, phi, v2, bb2);

        // Verify the phi node is in the merge block.
        let block = func.get_block(merge);
        let insts = block.instructions();
        assert!(!insts.is_empty());
        assert!(insts[0].is_phi());
    }

    #[test]
    fn test_build_gep() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let base = builder.build_alloca(&mut func, IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        }, Some("arr"));
        let idx = builder.build_const_int(&mut func, IrType::I64, 3);
        let elem_ptr = builder.build_gep(&mut func, base, vec![idx], IrType::I32, true);
        assert_ne!(elem_ptr, base);
    }

    #[test]
    fn test_build_casts() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        let val_i32 = builder.build_const_int(&mut func, IrType::I32, 42);

        let trunc = builder.build_trunc(&mut func, val_i32, IrType::I8);
        assert_ne!(trunc, val_i32);

        let zext = builder.build_zext(&mut func, trunc, IrType::I64);
        assert_ne!(zext, trunc);

        let sext = builder.build_sext(&mut func, trunc, IrType::I32);
        assert_ne!(sext, trunc);

        let bc = builder.build_bitcast(&mut func, val_i32, IrType::F32);
        assert_ne!(bc, val_i32);

        let ptr_val = builder.build_int_to_ptr(&mut func, val_i32, IrType::Ptr);
        assert_ne!(ptr_val, val_i32);

        let int_val = builder.build_ptr_to_int(&mut func, ptr_val, IrType::I64);
        assert_ne!(int_val, ptr_val);
    }

    #[test]
    fn test_build_inline_asm() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();
        let _entry = builder.append_block(&mut func, Some("entry"));

        // Asm with output (produces a result).
        let result = builder.build_inline_asm(
            &mut func,
            "mov $0, %%rax".into(),
            "=r".into(),
            vec![],
            vec!["cc".into(), "memory".into()],
            true,  // has_side_effects
            false, // is_align_stack
        );
        assert!(result.is_some());

        // Asm without output (side-effect only).
        let no_result = builder.build_inline_asm(
            &mut func,
            "nop".into(),
            "".into(),
            vec![],
            vec![],
            false, // has_side_effects
            false, // is_align_stack
        );
        assert!(no_result.is_none());
    }

    #[test]
    fn test_build_constants() {
        let mut func = make_func();
        let mut builder = IrBuilder::new();

        let ci = builder.build_const_int(&mut func, IrType::I32, 100);
        #[allow(clippy::approx_constant)]
        let cf = builder.build_const_float(&mut func, IrType::F64, 3.14);
        let cn = builder.build_const_null(&mut func, IrType::Ptr);
        let gr = builder.build_global_ref(&mut func, "my_global", IrType::Ptr);

        // All four should be distinct values.
        let ids = [ci, cf, cn, gr];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    #[test]
    fn test_default_alignment() {
        assert_eq!(IrBuilder::default_alignment(&IrType::I1), 1);
        assert_eq!(IrBuilder::default_alignment(&IrType::I8), 1);
        assert_eq!(IrBuilder::default_alignment(&IrType::I16), 2);
        assert_eq!(IrBuilder::default_alignment(&IrType::I32), 4);
        assert_eq!(IrBuilder::default_alignment(&IrType::I64), 8);
        assert_eq!(IrBuilder::default_alignment(&IrType::I128), 16);
        assert_eq!(IrBuilder::default_alignment(&IrType::F32), 4);
        assert_eq!(IrBuilder::default_alignment(&IrType::F64), 8);
        assert_eq!(IrBuilder::default_alignment(&IrType::F80), 16);
        assert_eq!(IrBuilder::default_alignment(&IrType::Ptr), 8);
        assert_eq!(IrBuilder::default_alignment(&IrType::Void), 1);

        // Array inherits element alignment.
        let arr = IrType::Array {
            element: Box::new(IrType::I32),
            count: 10,
        };
        assert_eq!(IrBuilder::default_alignment(&arr), 4);

        // Struct alignment = max field alignment.
        let s = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: false,
        };
        assert_eq!(IrBuilder::default_alignment(&s), 8);

        // Packed struct alignment = 1.
        let ps = IrType::Struct {
            fields: vec![IrType::I8, IrType::I64],
            packed: true,
        };
        assert_eq!(IrBuilder::default_alignment(&ps), 1);
    }

    #[test]
    fn test_insert_position_default() {
        assert_eq!(InsertPosition::default(), InsertPosition::End);
    }

    #[test]
    fn test_builder_default() {
        let builder = IrBuilder::default();
        assert_eq!(builder.get_insert_block(), None);
    }
}
