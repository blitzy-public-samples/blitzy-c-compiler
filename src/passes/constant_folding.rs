//! Constant folding and propagation optimization pass for the BCC compiler.
//!
//! This module implements Phase 8's constant folding pass — the first pass in
//! the fixed optimization pipeline:
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
//! | Cast folding               | Evaluate `Trunc`/`ZExt`/`SExt`/`BitCast` on constants      |
//! | Pointer cast folding       | Evaluate `IntToPtr`/`PtrToInt` on constant inputs          |
//! | Branch folding             | Replace `CondBranch` with known condition → `Branch`       |
//! | Switch folding             | Replace `Switch` with known value → `Branch`               |
//! | Phi folding                | Simplify phi nodes with all-identical constant inputs      |
//! | Constant propagation       | Track constant SSA values, propagate through use-def chains|
//!
//! # SSA Invariant Preservation
//!
//! When replacing an instruction with a constant, all uses of the result
//! `ValueId` are updated throughout the function via `Instruction::replace_use`.
//! Phi nodes are checked for constant-only incoming values. Def-use chains
//! remain valid because replacements substitute value references rather than
//! removing instructions (DCE handles removal).
//!
//! # Algorithm
//!
//! The pass uses a worklist-driven approach: all instructions are initially
//! added to the worklist. Each instruction is inspected; if foldable, it is
//! replaced and users are re-queued via `Instruction::uses()` for further
//! folding. Iteration continues until the worklist is empty. An `FxHashSet`
//! provides O(1) deduplication of worklist entries.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
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
/// Inspects the constant map first, then falls back to checking if the
/// value's defining type suggests a trivially determinable constant (e.g.,
/// a void-typed value or a known-zero).
///
/// Returns `None` if the value is not a known constant.
pub fn try_get_constant(
    constants: &FxHashMap<ValueId, ConstantValue>,
    func: &IrFunction,
    value: ValueId,
) -> Option<ConstantValue> {
    // Primary lookup: check the propagation map.
    if let Some(cv) = constants.get(&value) {
        return Some(cv.clone());
    }

    // Secondary: inspect the value's name — named constants created by
    // `IrBuilder::build_const_int`, `build_const_float`, and
    // `build_const_null` encode their value in the name string.
    // This is the *only* mechanism through which literal constants enter
    // the folding pipeline, because these pseudo-values have no defining
    // instruction.
    if let Some(info) = func.get_value_info(value) {
        if let Some(ref name) = info.name {
            if let Some(int_str) = name.strip_prefix("const.int.") {
                // Parse as i64 first (matches `build_const_int` signature),
                // then widen to i128 for the constant map.
                if let Ok(v) = int_str.parse::<i64>() {
                    return Some(ConstantValue::Int(v as i128));
                }
            } else if let Some(flt_str) = name.strip_prefix("const.float.") {
                if let Ok(v) = flt_str.parse::<f64>() {
                    return Some(ConstantValue::Float(v));
                }
            } else if name == "const.null" {
                return Some(ConstantValue::Null);
            }
        }
    }

    None
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
// Public entry point
// ---------------------------------------------------------------------------

/// Runs the constant folding and propagation pass on a single IR function.
///
/// Returns `true` if any instructions were folded or replaced, `false` if
/// the function was unchanged.
///
/// # Algorithm
///
/// 1. Initialize a worklist containing every instruction location
///    `(BasicBlockId, inst_index)` in the function.
/// 2. Use an `FxHashSet` to track which locations are currently enqueued,
///    preventing redundant re-processing.
/// 3. Dequeue each instruction, attempt folding:
///    - If the result is a compile-time constant, record it in the
///      constant map and enqueue all user instructions of the result.
///    - If the instruction can be simplified (algebraic identity),
///      replace uses and enqueue users.
///    - If a branch/switch condition is constant, replace the terminator
///      with an unconditional branch and update CFG edges.
/// 4. After the worklist drains, perform a phi-node simplification pass:
///    phi nodes whose incoming values are all the same constant are
///    folded and their users are re-processed.
///
/// # SSA Preservation
///
/// All replacements use `Instruction::replace_use` to maintain valid
/// SSA def-use chains. Phi nodes with a single constant incoming value
/// are simplified but not removed (DCE handles removal).
pub fn run_constant_folding(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let mut constants: FxHashMap<ValueId, ConstantValue> = FxHashMap::default();

    // -----------------------------------------------------------------
    // Phase 0: Seed the constant map from named constant values.
    //
    // `IrBuilder::build_const_int`, `build_const_float`, and
    // `build_const_null` create pseudo-values that carry their value
    // in the name string (e.g., "const.int.42").  These have *no*
    // defining instruction, so the worklist-driven Phase 2 would never
    // discover them.  We scan the entire value registry and populate
    // the constant map up-front so that all downstream fold functions
    // can look up operands via `constants.get(...)`.
    // -----------------------------------------------------------------
    for vi in func.local_values.iter() {
        if let Some(ref name) = vi.name {
            if let Some(int_str) = name.strip_prefix("const.int.") {
                if let Ok(v) = int_str.parse::<i64>() {
                    constants.insert(vi.id, ConstantValue::Int(v as i128));
                }
            } else if let Some(flt_str) = name.strip_prefix("const.float.") {
                if let Ok(v) = flt_str.parse::<f64>() {
                    constants.insert(vi.id, ConstantValue::Float(v));
                }
            } else if name == "const.null" {
                constants.insert(vi.id, ConstantValue::Null);
            }
        }
    }

    // -----------------------------------------------------------------
    // Phase 1: Build worklist with all instruction locations.
    // -----------------------------------------------------------------
    let mut worklist: Vec<(BasicBlockId, usize)> = Vec::new();
    let mut in_worklist: FxHashSet<(BasicBlockId, usize)> = FxHashSet::default();

    for block in func.blocks() {
        let bid = block.id;
        for idx in 0..block.instructions().len() {
            let key = (bid, idx);
            worklist.push(key);
            in_worklist.insert(key);
        }
    }

    // -----------------------------------------------------------------
    // Phase 2: Worklist-driven constant folding and propagation.
    // -----------------------------------------------------------------
    while let Some(key) = worklist.pop() {
        in_worklist.remove(&key);
        let (block_id, inst_idx) = key;

        // Guard against stale indices after branch folding may have
        // shortened an instruction list.
        let block = func.get_block(block_id);
        if inst_idx >= block.instructions().len() {
            continue;
        }

        let inst = block.instructions()[inst_idx].clone();

        // Skip instructions with side effects that cannot be folded
        // (e.g., Call, Store, InlineAsm). Terminators are handled
        // specially below.
        if inst.has_side_effects() && !inst.is_terminator() {
            continue;
        }

        match fold_instruction(&inst, &constants, func) {
            FoldResult::NoChange => {
                // Nothing to do — instruction is not foldable with
                // currently known constants.
            }
            FoldResult::Constant(result_id, value) => {
                constants.insert(result_id, value);
                // Enqueue all instructions that consume this result so
                // they can be re-examined with the new constant knowledge.
                enqueue_users_of(func, result_id, &mut worklist, &mut in_worklist);
                changed = true;
            }
            FoldResult::ReplaceWith(result_id, replacement_id) => {
                // Replace all uses of `result_id` with `replacement_id`
                // across the entire function.
                replace_all_uses(func, result_id, replacement_id);

                // If the replacement is itself a known constant,
                // propagate that knowledge to the result's entry.
                if let Some(cv) = constants.get(&replacement_id).cloned() {
                    constants.insert(result_id, cv);
                }

                // Enqueue users of the original result for re-folding.
                enqueue_users_of(func, result_id, &mut worklist, &mut in_worklist);
                changed = true;
            }
            FoldResult::FoldBranch(target) => {
                // Replace CondBranch/Switch terminator with an
                // unconditional Branch to the known target.
                {
                    let blk = func.get_block_mut(block_id);
                    let insts = blk.instructions_mut();
                    if inst_idx < insts.len() {
                        insts[inst_idx] = Instruction::Branch { target };
                    }
                }
                // Update predecessor/successor CFG edges for dropped
                // targets and clean up phi incoming edges.
                update_branch_edges(func, block_id, target, &inst);
                changed = true;
            }
        }
    }

    // -----------------------------------------------------------------
    // Phase 3: Phi-node simplification — if all incoming values of a
    // phi resolve to the same constant, record it and re-propagate.
    // Also handles trivial phis (all same ValueId, not necessarily
    // constant) via value-level replacement.
    // -----------------------------------------------------------------
    let mut phi_changed = true;
    // Track phis that have already been trivially eliminated so we
    // do not re-detect the same phi on the next while-loop iteration
    // (the instruction itself remains in the block — DCE removes it).
    let mut folded_phi_results: FxHashSet<ValueId> = FxHashSet::default();
    while phi_changed {
        phi_changed = false;
        let block_ids: Vec<BasicBlockId> = func.blocks().iter().map(|b| b.id).collect();

        for &block_id in &block_ids {
            let block = func.get_block(block_id);
            let phis: Vec<Instruction> = block
                .instructions()
                .iter()
                .filter(|i| i.is_phi())
                .cloned()
                .collect();

            for phi in &phis {
                if let Some(operands) = phi.phi_operands() {
                    let result_id_opt = phi.result();
                    // Skip phis we have already folded/eliminated.
                    if let Some(rid) = result_id_opt {
                        if folded_phi_results.contains(&rid) {
                            continue;
                        }
                    }
                    // --- Case A: all incoming values are the same constant ---
                    if let Some(cv) = try_fold_phi(operands, &constants) {
                        if let Some(result_id) = result_id_opt {
                            let prev = constants.get(&result_id);
                            if prev != Some(&cv) {
                                // If a stale constant existed, remove it
                                // before inserting the correct one.
                                if prev.is_some() {
                                    constants.remove(&result_id);
                                }
                                constants.insert(result_id, cv);
                                folded_phi_results.insert(result_id);
                                // Enqueue users for further folding.
                                enqueue_users_of(func, result_id, &mut worklist, &mut in_worklist);
                                phi_changed = true;
                                changed = true;
                            }
                        }
                    }
                    // --- Case B: trivial phi — all same ValueId (may not
                    //     be a constant, but the phi is still redundant) ---
                    else if let Some(result_id) = result_id_opt {
                        if let Some(common_val) = try_trivial_phi(operands, result_id) {
                            // Replace every use of the phi result with the
                            // single incoming value.
                            replace_all_uses(func, result_id, common_val);
                            folded_phi_results.insert(result_id);
                            // Propagate any constant knowledge.
                            if let Some(cv) = constants.get(&common_val).cloned() {
                                constants.insert(result_id, cv);
                            }
                            // Enqueue users for further folding.
                            enqueue_users_of(func, common_val, &mut worklist, &mut in_worklist);
                            phi_changed = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Drain any newly enqueued items from the phi pass.
        while let Some(key) = worklist.pop() {
            in_worklist.remove(&key);
            let (block_id, inst_idx) = key;

            let block = func.get_block(block_id);
            if inst_idx >= block.instructions().len() {
                continue;
            }
            let inst = block.instructions()[inst_idx].clone();

            if inst.has_side_effects() && !inst.is_terminator() {
                continue;
            }

            match fold_instruction(&inst, &constants, func) {
                FoldResult::NoChange => {}
                FoldResult::Constant(result_id, value) => {
                    constants.insert(result_id, value);
                    enqueue_users_of(func, result_id, &mut worklist, &mut in_worklist);
                    changed = true;
                    phi_changed = true;
                }
                FoldResult::ReplaceWith(result_id, replacement_id) => {
                    replace_all_uses(func, result_id, replacement_id);
                    if let Some(cv) = constants.get(&replacement_id).cloned() {
                        constants.insert(result_id, cv);
                    }
                    enqueue_users_of(func, result_id, &mut worklist, &mut in_worklist);
                    changed = true;
                    phi_changed = true;
                }
                FoldResult::FoldBranch(target) => {
                    {
                        let blk = func.get_block_mut(block_id);
                        let insts = blk.instructions_mut();
                        if inst_idx < insts.len() {
                            insts[inst_idx] = Instruction::Branch { target };
                        }
                    }
                    update_branch_edges(func, block_id, target, &inst);
                    changed = true;
                    phi_changed = true;
                }
            }
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Worklist helpers
// ---------------------------------------------------------------------------

/// Enqueues all instructions that *use* the given `ValueId` into the
/// worklist, using `FxHashSet`-based deduplication to avoid redundant work.
///
/// Scans every instruction in every block via `Instruction::uses()` to
/// locate consumers. This is O(n) in the total instruction count; for
/// large functions a pre-built use-list would be more efficient, but for
/// typical C translation units the linear scan is adequate.
fn enqueue_users_of(
    func: &IrFunction,
    value_id: ValueId,
    worklist: &mut Vec<(BasicBlockId, usize)>,
    in_worklist: &mut FxHashSet<(BasicBlockId, usize)>,
) {
    for block in func.blocks() {
        let bid = block.id;
        for (idx, inst) in block.instructions().iter().enumerate() {
            if inst.uses().contains(&value_id) {
                let key = (bid, idx);
                if !in_worklist.contains(&key) {
                    in_worklist.insert(key);
                    worklist.push(key);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction-level folding dispatcher
// ---------------------------------------------------------------------------

/// Attempts to fold a single instruction given the current constant map.
///
/// Dispatches to specialised folding functions based on the instruction
/// variant. Instructions with side effects or that are not amenable to
/// constant folding return `FoldResult::NoChange`.
fn fold_instruction(
    inst: &Instruction,
    constants: &FxHashMap<ValueId, ConstantValue>,
    func: &IrFunction,
) -> FoldResult {
    match inst {
        // ----- Arithmetic / bitwise / shift -----
        Instruction::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        } => fold_binop(*result, *op, *lhs, *rhs, ty, constants),

        // ----- Integer comparison -----
        Instruction::ICmp {
            result,
            pred,
            lhs,
            rhs,
        } => fold_icmp(*result, *pred, *lhs, *rhs, constants),

        // ----- Floating-point comparison -----
        Instruction::FCmp {
            result,
            pred,
            lhs,
            rhs,
        } => fold_fcmp(*result, *pred, *lhs, *rhs, constants),

        // ----- Conditional branch -----
        Instruction::CondBranch {
            condition,
            true_target,
            false_target,
        } => fold_cond_branch(*condition, *true_target, *false_target, constants),

        // ----- Switch -----
        Instruction::Switch {
            value,
            default,
            cases,
        } => fold_switch(*value, *default, cases, constants),

        // ----- Truncation -----
        Instruction::Trunc {
            result,
            value,
            to_ty,
        } => fold_trunc(*result, *value, to_ty, constants),

        // ----- Zero-extension -----
        Instruction::ZExt {
            result,
            value,
            to_ty,
        } => fold_zext(*result, *value, to_ty, constants),

        // ----- Sign-extension -----
        Instruction::SExt {
            result,
            value,
            to_ty,
        } => fold_sext(*result, *value, to_ty, constants),

        // ----- Bit-cast (reinterpret bits) -----
        Instruction::BitCast {
            result,
            value,
            to_ty,
        } => fold_bitcast(*result, *value, to_ty, constants, func),

        // ----- Integer-to-pointer conversion -----
        Instruction::IntToPtr {
            result,
            value,
            to_ty: _,
        } => fold_int_to_ptr(*result, *value, constants),

        // ----- Pointer-to-integer conversion -----
        Instruction::PtrToInt {
            result,
            value,
            to_ty,
        } => fold_ptr_to_int(*result, *value, to_ty, constants),

        // ----- Phi nodes are handled in Phase 3, not here -----
        Instruction::Phi { .. } => FoldResult::NoChange,

        // ----- Unconditional branches and all other instructions -----
        Instruction::Branch { .. } => FoldResult::NoChange,
        _ => FoldResult::NoChange,
    }
}

// ===========================================================================
// BinOp folding
// ===========================================================================

/// Attempts to fold a binary operation.
///
/// Two primary cases:
/// 1. Both operands are known constants → compute result at compile time.
/// 2. One operand has an algebraic identity → simplify without needing
///    both operands to be constant.
/// Truncates an i128 constant to the bit-width of the given IR type,
/// then sign-extends back to i128 so that subsequent signed comparisons
/// and arithmetic see the correct value.  This is essential for
/// compile-time evaluation of sequences like the software popcount
/// algorithm, where intermediate multiplications overflow the 32-bit
/// (or 64-bit) range and must wrap at the type boundary.
fn truncate_to_type_width(val: i128, ty: &IrType) -> i128 {
    match ty {
        IrType::I1 => val & 1,
        IrType::I8 => (val as i8) as i128,
        IrType::I16 => (val as i16) as i128,
        IrType::I32 => (val as i32) as i128,
        IrType::I64 => (val as i64) as i128,
        // I128 and non-integer types: no truncation needed.
        _ => val,
    }
}

fn fold_binop(
    result: ValueId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    let lhs_const = constants.get(&lhs);
    let rhs_const = constants.get(&rhs);

    // Case 1: both operands are known integer constants.
    if let (Some(ConstantValue::Int(l)), Some(ConstantValue::Int(r))) = (lhs_const, rhs_const) {
        if let Some(val) = eval_int_binop(op, *l, *r) {
            // CRITICAL: truncate the result to the IR type's bit-width.
            // Without this, intermediate results grow beyond the type
            // boundary (e.g., i128 product of two i32 values) and
            // subsequent shifts/comparisons produce wrong answers.
            let truncated = truncate_to_type_width(val, ty);
            return FoldResult::Constant(result, ConstantValue::Int(truncated));
        }
    }

    // Case 1b: both operands are known float constants.
    if let (Some(ConstantValue::Float(l)), Some(ConstantValue::Float(r))) = (lhs_const, rhs_const) {
        if let Some(val) = eval_float_binop(op, *l, *r) {
            return FoldResult::Constant(result, ConstantValue::Float(val));
        }
    }

    // Case 2: algebraic identity simplification (works even when only
    // one operand is constant or both operands are the same SSA value).
    if let Some(fold) = fold_algebraic_identity(result, op, lhs, rhs, constants) {
        return fold;
    }

    FoldResult::NoChange
}

/// Evaluates an integer binary operation at compile time.
///
/// Returns `None` for operations that would be undefined behaviour in C
/// (division by zero, shift by ≥ width), leaving them as-is for the
/// backend to emit a trap or the original operation.
fn eval_int_binop(op: BinOp, l: i128, r: i128) -> Option<i128> {
    match op {
        BinOp::Add => Some(l.wrapping_add(r)),
        BinOp::Sub => Some(l.wrapping_sub(r)),
        BinOp::Mul => Some(l.wrapping_mul(r)),
        BinOp::UDiv => {
            if r == 0 {
                None // Division by zero — undefined.
            } else {
                Some(((l as u128).wrapping_div(r as u128)) as i128)
            }
        }
        BinOp::SDiv => {
            if r == 0 {
                None
            } else {
                // Signed division; checked_div returns None on overflow
                // (e.g., i128::MIN / -1).
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
            if !(0..128).contains(&r) {
                None // Shift by negative or ≥ width — UB.
            } else {
                Some(l.wrapping_shl(r as u32))
            }
        }
        BinOp::LShr => {
            if !(0..128).contains(&r) {
                None
            } else {
                // Logical (unsigned) right shift.
                Some(((l as u128).wrapping_shr(r as u32)) as i128)
            }
        }
        BinOp::AShr => {
            if !(0..128).contains(&r) {
                None
            } else {
                // Arithmetic (signed) right shift — preserves sign bit.
                Some(l.wrapping_shr(r as u32))
            }
        }
        BinOp::And => Some(l & r),
        BinOp::Or => Some(l | r),
        BinOp::Xor => Some(l ^ r),
        // Floating-point operations — not handled by integer eval.
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => None,
    }
}

/// Evaluates a floating-point binary operation at compile time.
///
/// Returns `None` for division/remainder by zero to avoid producing
/// `Inf`/`NaN` during compilation (preserves the original semantics).
fn eval_float_binop(op: BinOp, l: f64, r: f64) -> Option<f64> {
    match op {
        BinOp::FAdd => Some(l + r),
        BinOp::FSub => Some(l - r),
        BinOp::FMul => Some(l * r),
        BinOp::FDiv => {
            if r == 0.0 {
                None
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

/// Attempts to simplify a `BinOp` via algebraic identity rules.
///
/// Returns `Some(FoldResult)` if an identity applies, `None` otherwise.
/// These transformations are valid even when only one operand (or neither)
/// is a known constant, because they rely on structural properties of the
/// operation (e.g., `x - x` is always 0 regardless of `x`'s value).
fn fold_algebraic_identity(
    result: ValueId,
    op: BinOp,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> Option<FoldResult> {
    let lhs_const = constants.get(&lhs);
    let rhs_const = constants.get(&rhs);

    /// Returns `true` if the constant value is integer zero.
    fn is_int_zero(cv: Option<&ConstantValue>) -> bool {
        matches!(cv, Some(ConstantValue::Int(0)))
    }
    /// Returns `true` if the constant value is integer one.
    fn is_int_one(cv: Option<&ConstantValue>) -> bool {
        matches!(cv, Some(ConstantValue::Int(1)))
    }

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
        // x & 0 → 0, 0 & x → 0, x & x → x
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
        // No algebraic simplification for URem, SRem, or FP operations.
        _ => {}
    }

    None
}

// ===========================================================================
// ICmp folding
// ===========================================================================

/// Attempts to fold an integer comparison.
///
/// Handles self-comparison (`x cmp x`) as a special case because the
/// result is always deterministic regardless of the actual value of `x`.
fn fold_icmp(
    result: ValueId,
    pred: ICmpPredicate,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    // Self-comparison: x cmp x — result depends only on the predicate.
    if lhs == rhs {
        let val = match pred {
            ICmpPredicate::Eq
            | ICmpPredicate::Ule
            | ICmpPredicate::Uge
            | ICmpPredicate::Sle
            | ICmpPredicate::Sge => true,
            ICmpPredicate::Ne
            | ICmpPredicate::Ult
            | ICmpPredicate::Ugt
            | ICmpPredicate::Slt
            | ICmpPredicate::Sgt => false,
        };
        return FoldResult::Constant(result, ConstantValue::Bool(val));
    }

    // Both operands are constant integers — evaluate the comparison.
    if let (Some(ConstantValue::Int(l)), Some(ConstantValue::Int(r))) =
        (constants.get(&lhs), constants.get(&rhs))
    {
        let val = eval_icmp(pred, *l, *r);
        return FoldResult::Constant(result, ConstantValue::Bool(val));
    }

    FoldResult::NoChange
}

/// Evaluates an integer comparison at compile time.
///
/// Unsigned predicates reinterpret the i128 operands as u128 before
/// comparing, matching the C-level unsigned comparison semantics.
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

// ===========================================================================
// FCmp folding
// ===========================================================================

/// Attempts to fold a floating-point comparison.
///
/// Note: unlike integer comparisons, `x == x` is **not** always true for
/// floats because NaN ≠ NaN. Therefore self-comparison folding is only
/// done when both operands are known constants.
fn fold_fcmp(
    result: ValueId,
    pred: FCmpPredicate,
    lhs: ValueId,
    rhs: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    // Both operands are constant floats — evaluate the comparison.
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
/// Ordered predicates (`O*`) return `false` when either operand is NaN.
/// Unordered predicates (`U*`) return `true` when either operand is NaN.
/// `Ord` checks that neither operand is NaN; `Uno` checks that at least
/// one operand is NaN.
fn eval_fcmp(pred: FCmpPredicate, l: f64, r: f64) -> bool {
    match pred {
        // Ordered predicates — false if any operand is NaN.
        FCmpPredicate::OEq => l == r,
        FCmpPredicate::ONe => l != r && !l.is_nan() && !r.is_nan(),
        FCmpPredicate::Ogt => l > r,
        FCmpPredicate::Oge => l >= r,
        FCmpPredicate::Olt => l < r,
        FCmpPredicate::Ole => l <= r,
        FCmpPredicate::Ord => !l.is_nan() && !r.is_nan(),
        // Unordered predicates — true if any operand is NaN.
        FCmpPredicate::Uno => l.is_nan() || r.is_nan(),
        FCmpPredicate::UEq => l == r || l.is_nan() || r.is_nan(),
        FCmpPredicate::UNe => l != r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ugt => l > r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Uge => l >= r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ult => l < r || l.is_nan() || r.is_nan(),
        FCmpPredicate::Ule => l <= r || l.is_nan() || r.is_nan(),
    }
}

// ===========================================================================
// Conditional branch folding
// ===========================================================================

/// Attempts to fold a conditional branch whose condition is a known constant.
///
/// If the condition resolves to true (non-zero) or false (zero), the
/// conditional branch is replaced with an unconditional `Branch` to the
/// appropriate target. The unused edge is cleaned up by
/// `update_branch_edges`.
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
            // Any non-zero integer is truthy in C.
            if *v != 0 {
                FoldResult::FoldBranch(true_target)
            } else {
                FoldResult::FoldBranch(false_target)
            }
        }
        _ => FoldResult::NoChange,
    }
}

// ===========================================================================
// Switch folding
// ===========================================================================

/// Attempts to fold a switch whose selector value is a known constant.
///
/// Iterates case values looking for a match; falls through to the default
/// target if no case matches. The original multi-way branch is replaced
/// with an unconditional `Branch`.
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
        // No case matched — branch to default.
        return FoldResult::FoldBranch(default);
    }
    FoldResult::NoChange
}

// ===========================================================================
// Cast folding
// ===========================================================================

/// Folds a truncation on a constant operand.
///
/// Truncation discards the upper bits: result = value & ((1 << width) - 1).
fn fold_trunc(
    result: ValueId,
    value: ValueId,
    to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        if let Some(width) = to_ty.integer_width() {
            let mask = if width >= 128 {
                i128::MAX // Full-width — no truncation needed.
            } else {
                (1i128 << width) - 1
            };
            return FoldResult::Constant(result, ConstantValue::Int(*v & mask));
        }
        // Truncating to I1 (boolean): result is the LSB.
        if *to_ty == IrType::I1 {
            return FoldResult::Constant(result, ConstantValue::Bool((*v & 1) != 0));
        }
    }
    FoldResult::NoChange
}

/// Folds a zero-extension on a constant operand.
///
/// Zero-extension fills upper bits with zeros. Since we store values in
/// i128 and the original value was already zero in its upper bits (assuming
/// correct semantics from earlier passes), the numeric value is unchanged.
fn fold_zext(
    result: ValueId,
    value: ValueId,
    _to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        return FoldResult::Constant(result, ConstantValue::Int(*v));
    }
    if let Some(ConstantValue::Bool(b)) = constants.get(&value) {
        // Bool → integer: true = 1, false = 0.
        return FoldResult::Constant(result, ConstantValue::Int(if *b { 1 } else { 0 }));
    }
    FoldResult::NoChange
}

/// Folds a sign-extension on a constant operand.
///
/// The value stored as i128 already preserves the sign from narrower
/// integer types (e.g., i8 -1 is stored as i128 -1). No additional
/// transformation is required.
fn fold_sext(
    result: ValueId,
    value: ValueId,
    _to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    if let Some(ConstantValue::Int(v)) = constants.get(&value) {
        return FoldResult::Constant(result, ConstantValue::Int(*v));
    }
    if let Some(ConstantValue::Bool(b)) = constants.get(&value) {
        // Bool → signed integer: true = -1 (all bits set), false = 0.
        // This matches C semantics where `(signed)true` sign-extends the 1-bit.
        return FoldResult::Constant(result, ConstantValue::Int(if *b { -1 } else { 0 }));
    }
    FoldResult::NoChange
}

/// Folds a bitcast on a constant operand.
///
/// Bitcast reinterprets the bits of a value as a different type of the
/// same bit-width. For integers, the value is unchanged. For float↔int
/// conversions, the bit pattern is preserved.
fn fold_bitcast(
    result: ValueId,
    value: ValueId,
    to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
    func: &IrFunction,
) -> FoldResult {
    match constants.get(&value) {
        Some(ConstantValue::Int(v)) => {
            // Integer-to-integer bitcast (different signedness, same width).
            if to_ty.integer_width().is_some() || *to_ty == IrType::I1 {
                return FoldResult::Constant(result, ConstantValue::Int(*v));
            }
            // Integer-to-float bitcast (e.g., i64 → f64 reinterpretation).
            match to_ty {
                IrType::F32 => {
                    let bits = (*v as u32).to_ne_bytes();
                    let f = f32::from_ne_bytes(bits);
                    return FoldResult::Constant(result, ConstantValue::Float(f as f64));
                }
                IrType::F64 => {
                    let bits = (*v as u64).to_ne_bytes();
                    let f = f64::from_ne_bytes(bits);
                    return FoldResult::Constant(result, ConstantValue::Float(f));
                }
                IrType::Ptr => {
                    // Integer → Pointer bitcast: treat as null if zero.
                    if *v == 0 {
                        return FoldResult::Constant(result, ConstantValue::Null);
                    }
                    // Non-zero integer → pointer: preserve as integer.
                    return FoldResult::Constant(result, ConstantValue::Int(*v));
                }
                _ => {}
            }
        }
        Some(ConstantValue::Float(f)) => {
            // Float-to-integer bitcast (e.g., f64 → i64 reinterpretation).
            let src_ty = func.get_value_type(value);
            match src_ty {
                IrType::F32 => {
                    let bits = (*f as f32).to_ne_bytes();
                    let int_val = u32::from_ne_bytes(bits) as i128;
                    return FoldResult::Constant(result, ConstantValue::Int(int_val));
                }
                IrType::F64 => {
                    let bits = f.to_ne_bytes();
                    let int_val = u64::from_ne_bytes(bits) as i128;
                    return FoldResult::Constant(result, ConstantValue::Int(int_val));
                }
                _ => {}
            }
        }
        Some(ConstantValue::Null) => {
            // Null pointer bitcast to integer → 0.
            if to_ty.integer_width().is_some() {
                return FoldResult::Constant(result, ConstantValue::Int(0));
            }
            // Null pointer bitcast to another pointer type → still null.
            if *to_ty == IrType::Ptr {
                return FoldResult::Constant(result, ConstantValue::Null);
            }
        }
        _ => {}
    }
    FoldResult::NoChange
}

/// Folds an integer-to-pointer conversion on a constant operand.
///
/// If the integer value is zero, the result is a null pointer.
/// Otherwise, the integer value is preserved (the pointer "address"
/// is the integer value).
fn fold_int_to_ptr(
    result: ValueId,
    value: ValueId,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    match constants.get(&value) {
        Some(ConstantValue::Int(0)) => FoldResult::Constant(result, ConstantValue::Null),
        Some(ConstantValue::Int(v)) => {
            // Non-zero integer → pointer: preserve the address value.
            FoldResult::Constant(result, ConstantValue::Int(*v))
        }
        _ => FoldResult::NoChange,
    }
}

/// Folds a pointer-to-integer conversion on a constant operand.
///
/// Null pointers convert to integer zero. Non-null constant pointer
/// values (rare in practice) preserve their numeric representation.
fn fold_ptr_to_int(
    result: ValueId,
    value: ValueId,
    to_ty: &IrType,
    constants: &FxHashMap<ValueId, ConstantValue>,
) -> FoldResult {
    match constants.get(&value) {
        Some(ConstantValue::Null) => FoldResult::Constant(result, ConstantValue::Int(0)),
        Some(ConstantValue::Int(v)) => {
            // Pointer stored as integer — mask to target width if known.
            if let Some(width) = to_ty.integer_width() {
                let mask = if width >= 128 {
                    i128::MAX
                } else {
                    (1i128 << width) - 1
                };
                return FoldResult::Constant(result, ConstantValue::Int(*v & mask));
            }
            FoldResult::Constant(result, ConstantValue::Int(*v))
        }
        _ => FoldResult::NoChange,
    }
}

// ===========================================================================
// Phi node constant folding
// ===========================================================================

/// Checks if a phi node has all-constant-identical incoming values.
///
/// Returns the common constant value if every incoming edge resolves to
/// the same constant in the propagation map. Returns `None` if any
/// incoming value is unknown or differs from the others.
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

/// Checks if a phi node is trivial — all incoming values are the exact
/// same `ValueId`, regardless of whether that value is a known constant.
///
/// When all incoming edges carry the same SSA value, the phi is redundant
/// and can be replaced by that value (the phi result becomes an alias).
/// This handles the common pattern produced by mem2reg where a phi
/// receives the same definition from every predecessor.
///
/// Returns `None` if the phi has no incoming edges, or if incoming
/// values differ, or if the common value is the phi result itself
/// (self-referencing cycle — leave for more advanced passes).
fn try_trivial_phi(incoming: &[(ValueId, BasicBlockId)], result_id: ValueId) -> Option<ValueId> {
    if incoming.is_empty() {
        return None;
    }

    let first = incoming[0].0;
    // Avoid self-referencing: if the phi feeds itself, skip.
    if first == result_id {
        return None;
    }
    for &(val, _) in &incoming[1..] {
        if val != first {
            return None;
        }
    }
    Some(first)
}

// ===========================================================================
// Utility: replace all uses of a value in the function
// ===========================================================================

/// Replaces all uses of `old_id` with `new_id` across every instruction
/// in every block of the function.
///
/// This preserves SSA def-use chain validity by updating operand references
/// in-place via `Instruction::replace_use`.
fn replace_all_uses(func: &mut IrFunction, old_id: ValueId, new_id: ValueId) {
    for block in func.blocks_mut() {
        for inst in block.instructions_mut() {
            inst.replace_use(old_id, new_id);
        }
    }
}

// ===========================================================================
// Branch edge update after folding
// ===========================================================================

/// Updates CFG predecessor/successor edges after folding a conditional
/// branch or switch into an unconditional branch.
///
/// The `old_inst` is the original terminator (CondBranch or Switch) that
/// was replaced. This function:
/// 1. Computes which successor targets were dropped.
/// 2. Removes the block from those targets' predecessor lists.
/// 3. Replaces the block's successor list with the single new target.
/// 4. Cleans up phi nodes in dropped targets by removing incoming edges
///    from this block.
fn update_branch_edges(
    func: &mut IrFunction,
    block_id: BasicBlockId,
    new_target: BasicBlockId,
    old_inst: &Instruction,
) {
    // Determine which blocks were reachable from the old terminator.
    let old_targets = old_inst.successor_blocks();

    // Collect targets that are no longer reachable after folding.
    let dropped: Vec<BasicBlockId> = old_targets
        .into_iter()
        .filter(|t| *t != new_target)
        .collect();

    // Update successor list of the current block: clear all and set the
    // single new unconditional target.
    {
        let block = func.get_block_mut(block_id);
        let current_succs: Vec<BasicBlockId> = block.successors().to_vec();
        for succ in &current_succs {
            block.remove_successor(*succ);
        }
        block.add_successor(new_target);
    }

    // Remove this block from each dropped target's predecessor list and
    // clean up phi nodes that referenced this block as an incoming edge.
    for &dropped_target in &dropped {
        let target_block = func.get_block_mut(dropped_target);
        target_block.remove_predecessor(block_id);

        // Remove incoming phi edges originating from this block.
        let insts = target_block.instructions_mut();
        for inst in insts.iter_mut() {
            if inst.is_phi() {
                inst.remove_phi_operand(block_id);
            }
        }
    }
}
