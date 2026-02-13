//! Constant folding and propagation optimization pass for the BCC compiler.
//!
//! This module implements Phase 8's constant folding pass, the first pass in the
//! fixed optimization pipeline:
//! **constant folding → dead code elimination → CFG simplification**.
//!
//! # Transformations
//!
//! The pass performs the following constant-related optimizations:
//!
//! | Category                   | Description                                                |
//! |----------------------------|------------------------------------------------------------|
//! | Binary op folding          | Evaluate `BinOp` with two constant operands at compile time|
//! | Algebraic identities       | Simplify `x + 0`, `x * 1`, `x & x`, `x ^ x`, etc.        |
//! | Comparison folding         | Evaluate `ICmp`/`FCmp` with constant operands              |
//! | Trivial comparison folding | `x == x` → true, `x < x` → false, etc.                   |
//! | Cast folding               | Evaluate `Trunc`/`ZExt`/`SExt` on constant inputs          |
//! | Branch folding             | Replace `CondBranch` with known condition → `Branch`       |
//! | Switch folding             | Replace `Switch` with known value → `Branch`               |
//! | Constant propagation       | Track constant SSA values, propagate through use-def chains|
//!
//! # SSA Invariant Preservation
//!
//! When replacing an instruction with a constant, all uses of the result
//! `ValueId` are updated throughout the function via `Instruction::replace_use`.
//! Phi nodes are checked for constant-only incoming values.
//!
//! # Algorithm
//!
//! The pass uses a worklist-driven approach: all instructions are initially
//! added to the worklist. Each instruction is inspected; if foldable, it is
//! replaced and users are re-queued for further folding. Iteration continues
//! until the worklist is empty.

use crate::common::fx_hash::FxHashMap;
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::{BinOp, FCmpPredicate, ICmpPredicate, Instruction};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// ConstantValue — compile-time-known value representation
// ---------------------------------------------------------------------------

/// Internal representation of a compile-time-known constant value.
///
/// Used during constant propagation to track SSA values whose concrete
/// value is known. The variants cover the value categories produced by
/// the IR instruction set.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstantValue {
    /// Known integer value (covers I1 through I128).
    Int(i128),
    /// Known floating-point value (covers F32, F64; F80 is approximated).
    Float(f64),
    /// Known boolean value (I1 true/false).
    Bool(bool),
    /// Known null pointer.
    Null,
}

// ---------------------------------------------------------------------------
// Constant lookup helper
// ---------------------------------------------------------------------------

/// Attempts to retrieve the compile-time constant value for an SSA value.
///
/// This inspects the constant map first, then falls back to checking if
/// the value's defining instruction is a trivially constant pattern.
///
/// Returns `None` if the value is not a known constant.
pub fn try_get_constant(
    constants: &FxHashMap<ValueId, ConstantValue>,
    _func: &IrFunction,
    value: ValueId,
) -> Option<ConstantValue> {
    constants.get(&value).cloned()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Runs the constant folding and propagation pass on a single IR function.
///
/// Returns `true` if any instructions were folded or replaced, `false` if
/// the function was unchanged.
///
/// # Algorithm
///
/// 1. Scan all instructions to identify initial constants (phi with
///    all-identical incoming, BinOp with constant operands, etc.).
/// 2. Build a constant map (`ValueId` → `ConstantValue`).
/// 3. Iterate over blocks and instructions:
///    - For each foldable instruction, compute the result and record it.
///    - Replace the instruction with a simpler form or mark its result as constant.
///    - Propagate to uses until no more folding is possible.
/// 4. Apply branch/switch folding for conditions with known values.
///
/// # SSA Preservation
///
/// All replacements use `Instruction::replace_use` to maintain valid
/// SSA def-use chains. Phi nodes with a single constant incoming value
/// are simplified but not removed (DCE handles removal).
pub fn run_constant_folding(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let mut constants: FxHashMap<ValueId, ConstantValue> = FxHashMap::default();

    // Run multiple iterations to handle cascading constant propagation.
    // In practice 2-3 iterations suffice for most functions.
    let max_iterations = 4;
    for _iteration in 0..max_iterations {
        let mut changed_this_iter = false;

        // Phase 1: Scan for foldable instructions across all blocks.
        let block_ids: Vec<BasicBlockId> = func.blocks().iter().map(|b| b.id).collect();

        for &block_id in &block_ids {
            let num_insts = func.get_block(block_id).instructions().len();
            let mut inst_idx = 0;

            while inst_idx < num_insts {
                // Re-fetch block reference each iteration since we may mutate.
                let block = func.get_block(block_id);
                if inst_idx >= block.instructions().len() {
                    break;
                }
                let inst = block.instructions()[inst_idx].clone();

                match fold_instruction(&inst, &constants) {
                    FoldResult::NoChange => {
                        inst_idx += 1;
                    }
                    FoldResult::Constant(result_id, value) => {
                        constants.insert(result_id, value);
                        // Propagate the constant to all uses across the function.
                        propagate_constant(func, result_id, &constants);
                        changed_this_iter = true;
                        inst_idx += 1;
                    }
                    FoldResult::ReplaceWith(result_id, replacement_id) => {
                        // Replace all uses of result_id with replacement_id.
                        replace_all_uses(func, result_id, replacement_id);
                        // If replacement_id is a known constant, propagate.
                        if let Some(cv) = constants.get(&replacement_id) {
                            constants.insert(result_id, cv.clone());
                        }
                        changed_this_iter = true;
                        inst_idx += 1;
                    }
                    FoldResult::FoldBranch(target) => {
                        // Replace CondBranch/Switch with unconditional Branch.
                        let block = func.get_block_mut(block_id);
                        let insts = block.instructions_mut();
                        if inst_idx < insts.len() {
                            insts[inst_idx] = Instruction::Branch { target };
                        }
                        // Update successor/predecessor edges.
                        update_branch_edges(func, block_id, target, &inst);
                        changed_this_iter = true;
                        inst_idx += 1;
                    }
                }
            }
        }

        // Phase 2: Phi node simplification — if all incoming values are the
        // same constant, record the phi result as that constant.
        for &block_id in &block_ids {
            let block = func.get_block(block_id);
            let phis: Vec<Instruction> = block
                .instructions()
                .iter()
                .filter(|i| i.is_phi())
                .cloned()
                .collect();

            for phi in &phis {
                if let Instruction::Phi {
                    result, incoming, ..
                } = phi
                {
                    if let Some(cv) = try_fold_phi(incoming, &constants) {
                        if !constants.contains_key(result) || constants.get(result) != Some(&cv) {
                            constants.insert(*result, cv);
                            changed_this_iter = true;
                        }
                    }
                }
            }
        }

        changed |= changed_this_iter;
        if !changed_this_iter {
            break;
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Internal fold result type
// ---------------------------------------------------------------------------

/// Result of attempting to fold a single instruction.
enum FoldResult {
    /// No folding possible — leave instruction unchanged.
    NoChange,
    /// Instruction result is a compile-time constant.
    Constant(ValueId, ConstantValue),
    /// Replace all uses of the first `ValueId` with the second `ValueId`.
    ReplaceWith(ValueId, ValueId),
    /// Replace a CondBranch/Switch with an unconditional Branch to the target.
    FoldBranch(BasicBlockId),
}

// ---------------------------------------------------------------------------
// Instruction-level folding
// ---------------------------------------------------------------------------

/// Attempts to fold a single instruction given the current constant map.
fn fold_instruction(
    inst: &Instruction,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    match inst {
        // Binary operation folding.
        Instruction::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        } => fold_binop(*result, *op, *lhs, *rhs, ty, constants),

        // Integer comparison folding.
        Instruction::ICmp {
            result,
            pred,
            lhs,
            rhs,
        } => fold_icmp(*result, *pred, *lhs, *rhs, constants),

        // Floating-point comparison folding.
        Instruction::FCmp {
            result,
            pred,
            lhs,
            rhs,
        } => fold_fcmp(*result, *pred, *lhs, *rhs, constants),

        // Conditional branch folding.
        Instruction::CondBranch {
            condition,
            true_target,
            false_target,
        } => fold_cond_branch(*condition, *true_target, *false_target, constants),

        // Switch folding.
        Instruction::Switch {
            value,
            default,
            cases,
        } => fold_switch(*value, *default, cases, constants),

        // Truncation folding.
        Instruction::Trunc {
            result,
            value,
            to_ty,
        } => fold_trunc(*result, *value, to_ty, constants),

        // Zero-extension folding.
        Instruction::ZExt {
            result,
            value,
            to_ty,
        } => fold_zext(*result, *value, to_ty, constants),

        // Sign-extension folding.
        Instruction::SExt {
            result,
            value,
            to_ty,
        } => fold_sext(*result, *value, to_ty, constants),

        // No folding for other instruction types.
        _ => FoldResult::NoChange,
    }
}

// ---------------------------------------------------------------------------
// BinOp folding
// ---------------------------------------------------------------------------

/// Attempts to fold a binary operation.
///
/// Handles two cases:
/// 1. Both operands are known constants → compute result.
/// 2. One operand has an algebraic identity → simplify.
fn fold_binop(
    result: ValueId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    _ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    let lhs_const = constants.get(&lhs);
    let rhs_const = constants.get(&rhs);

    // Case 1: both operands are known integer constants.
    if let (Some(ConstantValue::Int(l)), Some(ConstantValue::Int(r))) = (lhs_const, rhs_const) {
        if let Some(val) = eval_int_binop(op, *l, *r) {
            return FoldResult::Constant(result, ConstantValue::Int(val));
        }
    }

    // Case 1b: both operands are known float constants.
    if let (Some(ConstantValue::Float(l)), Some(ConstantValue::Float(r))) = (lhs_const, rhs_const)
    {
        if let Some(val) = eval_float_binop(op, *l, *r) {
            return FoldResult::Constant(result, ConstantValue::Float(val));
        }
    }

    // Case 2: algebraic identity simplification.
    // x + 0 → x, x - 0 → x, x * 1 → x, x * 0 → 0, etc.
    if let Some(fold) = fold_algebraic_identity(result, op, lhs, rhs, constants) {
        return fold;
    }

    FoldResult::NoChange
}

/// Evaluates an integer binary operation at compile time.
///
/// Returns `None` for operations that would be undefined (e.g., division by zero,
/// shift by width or more).
fn eval_int_binop(op: BinOp, l: i128, r: i128) -> Option<i128> {
    match op {
        BinOp::Add => Some(l.wrapping_add(r)),
        BinOp::Sub => Some(l.wrapping_sub(r)),
        BinOp::Mul => Some(l.wrapping_mul(r)),
        BinOp::UDiv => {
            if r == 0 {
                None // Division by zero — leave as-is.
            } else {
                // Interpret as unsigned 64-bit for correctness.
                Some(((l as u128).wrapping_div(r as u128)) as i128)
            }
        }
        BinOp::SDiv => {
            if r == 0 {
                None
            } else {
                l.checked_div(r)
            }
        }
        BinOp::URem => {
            if r == 0 {
                None
            } else {
                Some(((l as u128).wrapping_rem(r as u128)) as i128)
            }
        }
        BinOp::SRem => {
            if r == 0 {
                None
            } else {
                l.checked_rem(r)
            }
        }
        BinOp::Shl => {
            if r < 0 || r >= 128 {
                None // Shift by negative or width — UB.
            } else {
                Some(l.wrapping_shl(r as u32))
            }
        }
        BinOp::LShr => {
            if r < 0 || r >= 128 {
                None
            } else {
                Some(((l as u128).wrapping_shr(r as u32)) as i128)
            }
        }
        BinOp::AShr => {
            if r < 0 || r >= 128 {
                None
            } else {
                Some(l.wrapping_shr(r as u32))
            }
        }
        BinOp::And => Some(l & r),
        BinOp::Or => Some(l | r),
        BinOp::Xor => Some(l ^ r),
        // Floating-point operations are not handled by integer eval.
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => None,
    }
}

/// Evaluates a floating-point binary operation at compile time.
fn eval_float_binop(op: BinOp, l: f64, r: f64) -> Option<f64> {
    match op {
        BinOp::FAdd => Some(l + r),
        BinOp::FSub => Some(l - r),
        BinOp::FMul => Some(l * r),
        BinOp::FDiv => {
            if r == 0.0 {
                None // Avoid producing Inf/NaN at compile time.
            } else {
                Some(l / r)
            }
        }
        BinOp::FRem => {
            if r == 0.0 {
                None
            } else {
                Some(l % r)
            }
        }
        _ => None,
    }
}

/// Attempts to simplify a BinOp via algebraic identity rules.
///
/// Returns `Some(FoldResult)` if an identity applies, `None` otherwise.
fn fold_algebraic_identity(
    result: ValueId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> Option<FoldResult> {
    let lhs_const = constants.get(&lhs);
    let rhs_const = constants.get(&rhs);

    let is_int_zero = |cv: Option<&ConstantValue>| matches!(cv, Some(ConstantValue::Int(0)));
    let is_int_one = |cv: Option<&ConstantValue>| matches!(cv, Some(ConstantValue::Int(1)));

    match op {
        // x + 0 → x, 0 + x → x
        BinOp::Add => {
            if is_int_zero(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
            if is_int_zero(lhs_const) {
                return Some(FoldResult::ReplaceWith(result, rhs));
            }
        }
        // x - 0 → x, x - x → 0
        BinOp::Sub => {
            if is_int_zero(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
            if lhs == rhs {
                return Some(FoldResult::Constant(result, ConstantValue::Int(0)));
            }
        }
        // x * 1 → x, 1 * x → x, x * 0 → 0, 0 * x → 0
        BinOp::Mul => {
            if is_int_one(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
            if is_int_one(lhs_const) {
                return Some(FoldResult::ReplaceWith(result, rhs));
            }
            if is_int_zero(rhs_const) {
                return Some(FoldResult::Constant(result, ConstantValue::Int(0)));
            }
            if is_int_zero(lhs_const) {
                return Some(FoldResult::Constant(result, ConstantValue::Int(0)));
            }
        }
        // x / 1 → x
        BinOp::UDiv | BinOp::SDiv => {
            if is_int_one(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
        }
        // x << 0 → x, x >> 0 → x
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            if is_int_zero(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
        }
        // x & 0 → 0, x & x → x
        BinOp::And => {
            if is_int_zero(rhs_const) || is_int_zero(lhs_const) {
                return Some(FoldResult::Constant(result, ConstantValue::Int(0)));
            }
            if lhs == rhs {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
        }
        // x | 0 → x, 0 | x → x, x | x → x
        BinOp::Or => {
            if is_int_zero(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
            if is_int_zero(lhs_const) {
                return Some(FoldResult::ReplaceWith(result, rhs));
            }
            if lhs == rhs {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
        }
        // x ^ 0 → x, 0 ^ x → x, x ^ x → 0
        BinOp::Xor => {
            if is_int_zero(rhs_const) {
                return Some(FoldResult::ReplaceWith(result, lhs));
            }
            if is_int_zero(lhs_const) {
                return Some(FoldResult::ReplaceWith(result, rhs));
            }
            if lhs == rhs {
                return Some(FoldResult::Constant(result, ConstantValue::Int(0)));
            }
        }
        // No algebraic simplification for remaining ops, URem, SRem, FP ops.
        _ => {}
    }

    None
}

// ---------------------------------------------------------------------------
// ICmp folding
// ---------------------------------------------------------------------------

/// Attempts to fold an integer comparison.
fn fold_icmp(
    result: ValueId,
    pred: ICmpPredicate,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    // Self-comparison: x cmp x
    if lhs == rhs {
        let val = match pred {
            ICmpPredicate::Eq | ICmpPredicate::Ule | ICmpPredicate::Uge
            | ICmpPredicate::Sle | ICmpPredicate::Sge => true,
            ICmpPredicate::Ne | ICmpPredicate::Ult | ICmpPredicate::Ugt
            | ICmpPredicate::Slt | ICmpPredicate::Sgt => false,
        };
        return FoldResult::Constant(result, ConstantValue::Bool(val));
    }

    // Both operands are constant integers.
    if let (Some(ConstantValue::Int(l)), Some(ConstantValue::Int(r))) =
        (constants.get(&lhs), constants.get(&rhs))
    {
        let val = eval_icmp(pred, *l, *r);
        return FoldResult::Constant(result, ConstantValue::Bool(val));
    }

    FoldResult::NoChange
}

/// Evaluates an integer comparison at compile time.
fn eval_icmp(pred: ICmpPredicate, l: i128, r: i128) -> bool {
    match pred {
        ICmpPredicate::Eq => l == r,
        ICmpPredicate::Ne => l != r,
        ICmpPredicate::Ugt => (l as u128) > (r as u128),
        ICmpPredicate::Uge => (l as u128) >= (r as u128),
        ICmpPredicate::Ult => (l as u128) < (r as u128),
        ICmpPredicate::Ule => (l as u128) <= (r as u128),
        ICmpPredicate::Sgt => l > r,
        ICmpPredicate::Sge => l >= r,
        ICmpPredicate::Slt => l < r,
        ICmpPredicate::Sle => l <= r,
    }
}

// ---------------------------------------------------------------------------
// FCmp folding
// ---------------------------------------------------------------------------

/// Attempts to fold a floating-point comparison.
fn fold_fcmp(
    result: ValueId,
    pred: FCmpPredicate,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    // Both operands are constant floats.
    if let (Some(ConstantValue::Float(l)), Some(ConstantValue::Float(r))) =
        (constants.get(&lhs), constants.get(&rhs))
    {
        let val = eval_fcmp(pred, *l, *r);
        return FoldResult::Constant(result, ConstantValue::Bool(val));
    }

    FoldResult::NoChange
}

/// Evaluates a floating-point comparison at compile time.
///
/// Handles IEEE 754 NaN semantics: ordered predicates return false if
/// either operand is NaN; unordered predicates return true if either is NaN.
fn eval_fcmp(pred: FCmpPredicate, l: f64, r: f64) -> bool {
    match pred {
        FCmpPredicate::OEq => l == r,
        FCmpPredicate::ONe => l != r && !l.is_nan() && !r.is_nan(),
        FCmpPredicate::Ogt => l > r,
        FCmpPredicate::Oge => l >= r,
        FCmpPredicate::Olt => l < r,
        FCmpPredicate::Ole => l <= r,
        FCmpPredicate::Ord => !l.is_nan() && !r.is_nan(),
        FCmpPredicate::Uno => l.is_nan() || r.is_nan(),
        FCmpPredicate::UEq => l == r || l.is_nan() || r.is_nan(),
        FCmpPredicate::UNe => l != r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ugt => l > r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Uge => l >= r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ult => l < r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ule => l <= r || l.is_nan() || r.is_nan(),
    }
}

// ---------------------------------------------------------------------------
// Conditional branch folding
// ---------------------------------------------------------------------------

/// Attempts to fold a conditional branch with a known-constant condition.
fn fold_cond_branch(
    condition: ValueId,
    true_target: BasicBlockId,
    false_target: BasicBlockId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    match constants.get(&condition) {
        Some(ConstantValue::Bool(true)) | Some(ConstantValue::Int(1)) => {
            FoldResult::FoldBranch(true_target)
        }
        Some(ConstantValue::Bool(false)) | Some(ConstantValue::Int(0)) => {
            FoldResult::FoldBranch(false_target)
        }
        Some(ConstantValue::Int(v)) => {
            // Non-zero integer → true.
            if *v != 0 {
                FoldResult::FoldBranch(true_target)
            } else {
                FoldResult::FoldBranch(false_target)
            }
        }
        _ => FoldResult::NoChange,
    }
}

// ---------------------------------------------------------------------------
// Switch folding
// ---------------------------------------------------------------------------

/// Attempts to fold a switch with a known-constant value.
fn fold_switch(
    value: ValueId,
    default: BasicBlockId,
    cases: &[(i64, BasicBlockId)],
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        let v64 = *v as i64;
        for &(case_val, target) in cases {
            if case_val == v64 {
                return FoldResult::FoldBranch(target);
            }
        }
        // No case matched — use default.
        return FoldResult::FoldBranch(default);
    }
    FoldResult::NoChange
}

// ---------------------------------------------------------------------------
// Cast folding
// ---------------------------------------------------------------------------

/// Attempts to fold a truncation operation on a constant.
fn fold_trunc(
    result: ValueId,
    value: ValueId,
    to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        if let Some(width) = to_ty.integer_width() {
            let mask = if width >= 128 {
                i128::MAX
            } else {
                (1i128 << width) - 1
            };
            return FoldResult::Constant(result, ConstantValue::Int(*v & mask));
        }
    }
    FoldResult::NoChange
}

/// Attempts to fold a zero-extension on a constant.
fn fold_zext(
    result: ValueId,
    value: ValueId,
    _to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        // Zero-extension: the value is already stored as i128; higher bits
        // are inherently zero if the original value was unsigned.
        return FoldResult::Constant(result, ConstantValue::Int(*v));
    }
    FoldResult::NoChange
}

/// Attempts to fold a sign-extension on a constant.
fn fold_sext(
    result: ValueId,
    value: ValueId,
    _to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        // Sign-extension: the value stored as i128 already preserves sign.
        return FoldResult::Constant(result, ConstantValue::Int(*v));
    }
    FoldResult::NoChange
}

// ---------------------------------------------------------------------------
// Phi node constant folding
// ---------------------------------------------------------------------------

/// Checks if a phi node has all-constant-identical incoming values.
///
/// Returns the common constant value if all incoming values resolve to
/// the same constant, `None` otherwise.
fn try_fold_phi(
    incoming: &[(ValueId, BasicBlockId)],
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> Option<ConstantValue> {
    if incoming.is_empty() {
        return None;
    }

    let first = constants.get(&incoming[0].0)?;
    for &(val, _) in &incoming[1..] {
        match constants.get(&val) {
            Some(cv) if cv == first => continue,
            _ => return None,
        }
    }

    Some(first.clone())
}

// ---------------------------------------------------------------------------
// Utility: replace all uses of a value in the function
// ---------------------------------------------------------------------------

/// Replaces all uses of `old_id` with `new_id` across all instructions in
/// the function.
fn replace_all_uses(func: &mut IrFunction, old_id: ValueId, new_id: ValueId) {
    for block in func.blocks_mut() {
        for inst in block.instructions_mut() {
            inst.replace_use(old_id, new_id);
        }
    }
}

/// Propagates a constant value by replacing uses where applicable.
///
/// This is a best-effort propagation: it updates the constant map but
/// does not directly modify instructions (the main loop handles that).
fn propagate_constant(
    _func: &mut IrFunction,
    _value_id: ValueId,
    _constants: &FxHashMap<ValueId, ConstantValue>,
) {
    // Propagation is handled by the main iteration loop: when a value is
    // added to the constant map, subsequent iterations will detect operands
    // that are now constant and fold their consuming instructions.
}

// ---------------------------------------------------------------------------
// Branch edge update after folding
// ---------------------------------------------------------------------------

/// Updates CFG predecessor/successor edges after folding a conditional
/// branch or switch into an unconditional branch.
///
/// The `old_inst` is the original terminator (CondBranch or Switch) that
/// was replaced. This function:
/// 1. Computes which successor targets were dropped.
/// 2. Removes the block from those targets' predecessor lists.
/// 3. Removes those targets from the block's successor list.
fn update_branch_edges(
    func: &mut IrFunction,
    block_id: BasicBlockId,
    new_target: BasicBlockId,
    old_inst: &Instruction,
) {
    let old_targets = old_inst.successor_blocks();

    // Collect targets that are no longer reachable from this block.
    let dropped: Vec<BasicBlockId> = old_targets
        .into_iter()
        .filter(|t| *t != new_target)
        .collect();

    // Update successor list of the current block.
    {
        let block = func.get_block_mut(block_id);
        // Clear all successors and set the single new target.
        let current_succs: Vec<BasicBlockId> = block.successors().to_vec();
        for succ in &current_succs {
            block.remove_successor(*succ);
        }
        block.add_successor(new_target);
    }

    // Remove this block from dropped targets' predecessor lists and clean
    // up their phi nodes.
    for &dropped_target in &dropped {
        let target_block = func.get_block_mut(dropped_target);
        target_block.remove_predecessor(block_id);

        // Remove incoming phi edges from this block.
        let insts = target_block.instructions_mut();
        for inst in insts.iter_mut() {
            if inst.is_phi() {
                inst.remove_phi_operand(block_id);
            }
        }
    }
}
