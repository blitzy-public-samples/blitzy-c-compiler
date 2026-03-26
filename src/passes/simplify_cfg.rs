//! Control-Flow Graph Simplification Pass
//!
//! This optimization pass simplifies the control-flow graph of a function through
//! four transformations:
//!
//! 1. **Block merging**: When block B has a single predecessor A, and A has a single
//!    successor B (and A's terminator is an unconditional branch to B), merge B's
//!    instructions into A and remove B.
//!
//! 2. **Empty block elimination**: When a block contains only an unconditional branch
//!    (no other instructions, or only phi nodes that are trivial), redirect all
//!    predecessors to jump directly to the target, then remove the empty block.
//!
//! 3. **Branch chain simplification (jump threading)**: When block A branches to B,
//!    and B contains only an unconditional branch to C, redirect A to branch
//!    directly to C (short-circuiting B).
//!
//! 4. **Trivial phi elimination**: When a phi node has all identical incoming values
//!    (or all incoming values are the phi result itself plus one other value),
//!    replace the phi with the single non-self value.
//!
//! The pass preserves SSA invariants and returns `true` if any modifications
//! were made (for fixpoint iteration).

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};

/// Maximum number of hops to follow when resolving branch chains, preventing
/// infinite loops on cyclic empty-block graphs.
const MAX_CHAIN_HOPS: usize = 10;

// ===========================================================================
// Helper functions
// ===========================================================================

/// Returns the target [`BlockId`] if `inst` is an unconditional branch, or
/// `None` for every other instruction kind.
///
/// Used throughout all four transformations to identify unconditional branches
/// in terminators.
#[inline]
fn is_unconditional_branch(inst: &Instruction) -> Option<BlockId> {
    if let Instruction::Branch { target, .. } = inst {
        Some(*target)
    } else {
        None
    }
}

/// Returns `true` if the block is "empty" — its only non-phi instruction is
/// an unconditional branch.
///
/// A block is considered empty when it does no real work: it may have phi
/// nodes at the start (selecting values from predecessors), but the only
/// computational instruction is a branch to the next block.
#[inline]
fn is_empty_block(block: &BasicBlock) -> bool {
    let non_phi_count = block
        .instructions
        .iter()
        .filter(|inst| !inst.is_phi())
        .count();
    if non_phi_count != 1 {
        return false;
    }
    // The single non-phi instruction must be an unconditional branch terminator.
    match block.terminator() {
        Some(t) if t.is_terminator() => is_unconditional_branch(t).is_some(),
        _ => false,
    }
}

/// Returns the branch target of a *pure trampoline* block — a block whose
/// only instruction is an unconditional branch (no phi nodes, no other
/// instructions).
///
/// This stricter check (compared to [`is_empty_block`]) is used by branch
/// chain simplification, which can only safely follow chains through blocks
/// that neither define nor select any values.
#[inline]
fn get_pure_branch_target(block: &BasicBlock) -> Option<BlockId> {
    if block.instructions.len() == 1 {
        is_unconditional_branch(&block.instructions[0])
    } else {
        None
    }
}

/// Replaces all occurrences of `old_block` with `new_block` inside a
/// terminator instruction's branch targets.
///
/// Handles [`Branch`](Instruction::Branch), [`CondBranch`](Instruction::CondBranch),
/// and [`Switch`](Instruction::Switch).  Non-terminator instructions are
/// silently ignored.
fn replace_block_reference_in_terminator(
    inst: &mut Instruction,
    old_block: BlockId,
    new_block: BlockId,
) {
    match inst {
        Instruction::Branch { target, .. } => {
            if *target == old_block {
                *target = new_block;
            }
        }
        Instruction::CondBranch {
            then_block,
            else_block,
            ..
        } => {
            if *then_block == old_block {
                *then_block = new_block;
            }
            if *else_block == old_block {
                *else_block = new_block;
            }
        }
        Instruction::Switch { default, cases, .. } => {
            if *default == old_block {
                *default = new_block;
            }
            for (_case_val, target) in cases.iter_mut() {
                if *target == old_block {
                    *target = new_block;
                }
            }
        }
        Instruction::IndirectBranch {
            possible_targets, ..
        } => {
            for t in possible_targets.iter_mut() {
                if *t == old_block {
                    *t = new_block;
                }
            }
        }
        _ => {}
    }
}

/// Replaces all occurrences of `old_pred` with `new_pred` in the predecessor
/// block references of a phi instruction's incoming list.
///
/// If the instruction is not a [`Phi`](Instruction::Phi), this is a no-op.
fn replace_phi_predecessor(inst: &mut Instruction, old_pred: BlockId, new_pred: BlockId) {
    if let Instruction::Phi { incoming, .. } = inst {
        for (_val, block_ref) in incoming.iter_mut() {
            if *block_ref == old_pred {
                *block_ref = new_pred;
            }
        }
    }
}

/// Returns `true` if `value` is referenced as an operand anywhere in
/// `func`.  Uses [`Instruction::operands()`] for read-only traversal.
///
/// This quick check is used before committing to a whole-function rewrite
/// in [`replace_value_in_function`], allowing us to skip the rewrite when
/// a dead phi result is already unused.
#[inline]
fn is_value_used(func: &IrFunction, value: Value) -> bool {
    for block in func.blocks() {
        for inst in block.instructions.iter() {
            if inst.operands().contains(&value) {
                return true;
            }
        }
    }
    false
}

/// Replaces every *use* of `old_value` with `new_value` across all
/// instructions in every block of `func`.
///
/// This is the core operation for trivial phi elimination: once a phi is
/// determined to always produce the same value, all references to the phi's
/// result must be rewritten to the replacement value.
///
/// Only operands (uses) are updated — result definitions are untouched.
fn replace_value_in_function(func: &mut IrFunction, old_value: Value, new_value: Value) {
    for block in func.blocks.iter_mut() {
        for inst in block.instructions.iter_mut() {
            // Use operands_mut() to get mutable references to all Value operands.
            for operand in inst.operands_mut() {
                if *operand == old_value {
                    *operand = new_value;
                }
            }
        }
    }
}

/// Rewrites all [`BlockId`] references inside a single instruction according
/// to the provided `index_map` (old index → new index).
///
/// Handles every instruction variant that contains [`BlockId`] values:
/// [`Branch`](Instruction::Branch), [`CondBranch`](Instruction::CondBranch),
/// [`Switch`](Instruction::Switch), [`Phi`](Instruction::Phi), and
/// [`InlineAsm`](Instruction::InlineAsm) (goto targets).
fn update_block_ids_in_instruction(inst: &mut Instruction, index_map: &FxHashMap<usize, usize>) {
    match inst {
        Instruction::Branch { target, .. } => {
            if let Some(&new_idx) = index_map.get(&target.index()) {
                *target = BlockId(new_idx as u32);
            }
        }
        Instruction::CondBranch {
            then_block,
            else_block,
            ..
        } => {
            if let Some(&new_idx) = index_map.get(&then_block.index()) {
                *then_block = BlockId(new_idx as u32);
            }
            if let Some(&new_idx) = index_map.get(&else_block.index()) {
                *else_block = BlockId(new_idx as u32);
            }
        }
        Instruction::Switch { default, cases, .. } => {
            if let Some(&new_idx) = index_map.get(&default.index()) {
                *default = BlockId(new_idx as u32);
            }
            for (_case_val, target) in cases.iter_mut() {
                if let Some(&new_idx) = index_map.get(&target.index()) {
                    *target = BlockId(new_idx as u32);
                }
            }
        }
        Instruction::Phi { incoming, .. } => {
            for (_val, block_ref) in incoming.iter_mut() {
                if let Some(&new_idx) = index_map.get(&block_ref.index()) {
                    *block_ref = BlockId(new_idx as u32);
                }
            }
        }
        Instruction::InlineAsm { goto_targets, .. } => {
            for target in goto_targets.iter_mut() {
                if let Some(&new_idx) = index_map.get(&target.index()) {
                    *target = BlockId(new_idx as u32);
                }
            }
        }
        Instruction::IndirectBranch {
            possible_targets, ..
        } => {
            for target in possible_targets.iter_mut() {
                if let Some(&new_idx) = index_map.get(&target.index()) {
                    *target = BlockId(new_idx as u32);
                }
            }
        }
        // Return instructions do not carry BlockId references, but we
        // match them explicitly for documentation completeness.
        Instruction::Return { .. } => {}
        _ => {}
    }
}

/// Removes all blocks in the `removed` set from the function and re-indexes
/// every surviving block so that block indices remain dense (0, 1, 2, …).
///
/// After removal and compaction, every [`BlockId`] reference in every
/// instruction, every predecessor/successor list entry, and every block's
/// `index` field is updated to match the new numbering.  Dominance
/// information (idom, dominance frontier) is also remapped.
///
/// # Panics
///
/// Panics in debug builds if `removed` contains index 0 (the entry block
/// must never be removed).
fn remove_and_reindex(func: &mut IrFunction, removed: &FxHashSet<usize>) {
    if removed.is_empty() {
        return;
    }
    debug_assert!(
        !removed.contains(&0),
        "remove_and_reindex: entry block (index 0) must never be removed"
    );

    let old_len = func.blocks.len();

    // Build old_index → new_index mapping for surviving blocks.
    let mut index_map: FxHashMap<usize, usize> = FxHashMap::default();
    let mut new_idx = 0usize;
    for old_idx in 0..old_len {
        if !removed.contains(&old_idx) {
            index_map.insert(old_idx, new_idx);
            new_idx += 1;
        }
    }

    // Remove blocks from the Vec in reverse order so that earlier indices
    // remain valid while later blocks are removed.
    let mut indices_to_remove: Vec<usize> = removed.iter().copied().collect();
    indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for idx in indices_to_remove {
        func.blocks.remove(idx);
    }

    // Update every surviving block's metadata and instruction references.
    for block in func.blocks.iter_mut() {
        // Block's own index.
        if let Some(&new) = index_map.get(&block.index) {
            block.index = new;
        }

        // Predecessor and successor lists.
        block.predecessors = block
            .predecessors
            .iter()
            .filter_map(|&old| index_map.get(&old).copied())
            .collect();
        block.successors = block
            .successors
            .iter()
            .filter_map(|&old| index_map.get(&old).copied())
            .collect();

        // All BlockId references in instructions.
        for inst in block.instructions.iter_mut() {
            update_block_ids_in_instruction(inst, &index_map);
        }

        // Dominance information (may be stale after CFG changes, but keep
        // it consistent to avoid later confusion).
        if let Some(idom_val) = block.idom {
            block.idom = index_map.get(&idom_val).copied();
        }
        block.dominance_frontier = block
            .dominance_frontier
            .iter()
            .filter_map(|&old| index_map.get(&old).copied())
            .collect();
    }
}

// ===========================================================================
// Pass 1 — Trivial phi node elimination
// ===========================================================================

/// Simplifies trivial phi nodes in all blocks.
///
/// A phi node is *trivial* when all of its non-self-referential incoming
/// values are identical.  Self-referential incoming edges (where the incoming
/// value equals the phi's own result) are ignored because they represent
/// back-edges that circularly depend on the phi definition itself.
///
/// Examples of trivial phis:
/// - `%3 = phi [%2, %bb1], [%2, %bb2]`  →  `%3` is always `%2`
/// - `%1 = phi [%1, %bb0], [%2, %bb1]`  →  only non-self value is `%2`
///
/// The pass iterates until no more trivial phis are found, handling cascading
/// simplifications (removing one trivial phi may make another trivial).
///
/// Returns `true` if any phi nodes were eliminated.
fn simplify_trivial_phis(func: &mut IrFunction) -> bool {
    let mut any_changed = false;

    loop {
        // Collect all trivial phis: (phi_result, replacement_value).
        let mut trivial_phis: Vec<(Value, Value)> = Vec::new();

        for block in func.blocks() {
            for inst in block.instructions.iter() {
                if let Instruction::Phi {
                    result, incoming, ..
                } = inst
                {
                    let mut unique_value: Option<Value> = None;
                    let mut is_trivial = true;

                    for &(val, _block_id) in incoming {
                        // Skip self-referential incoming edges.
                        if val == *result {
                            continue;
                        }
                        match unique_value {
                            None => unique_value = Some(val),
                            Some(uv) if uv == val => {
                                // Same value — still trivial.
                            }
                            Some(_) => {
                                // Different non-self values — non-trivial.
                                is_trivial = false;
                                break;
                            }
                        }
                    }

                    if is_trivial {
                        if let Some(replacement) = unique_value {
                            trivial_phis.push((*result, replacement));
                        }
                        // If unique_value is None, all incoming values are self-
                        // referential (dead phi).  In well-formed SSA this shouldn't
                        // happen, but we leave it for DCE to clean up.
                    }
                }
            }
        }

        if trivial_phis.is_empty() {
            break;
        }

        any_changed = true;

        // Replace all uses of each trivial phi's result with its replacement.
        // Skip rewrites for results that are already unused (dead phis).
        for &(old_val, new_val) in &trivial_phis {
            if is_value_used(func, old_val) {
                replace_value_in_function(func, old_val, new_val);
            }
        }

        // Remove the now-dead phi instructions.
        let trivial_results: FxHashSet<Value> =
            trivial_phis.iter().map(|&(result, _)| result).collect();
        for block in func.blocks_mut() {
            block.instructions.retain(|inst| {
                if let Some(result) = inst.result() {
                    // Remove phi instructions whose result was identified as trivial.
                    !(inst.is_phi() && trivial_results.contains(&result))
                } else {
                    true
                }
            });
        }
    }

    any_changed
}

// ===========================================================================
// Pass 2 — Block merging (single predecessor / single successor)
// ===========================================================================

/// Merges basic blocks connected by a single-predecessor/single-successor
/// unconditional branch edge.
///
/// When block B has exactly one predecessor A, A has exactly one successor B,
/// and A's terminator is an unconditional branch to B:
///
/// 1. Phi nodes in B (which must be trivial since B has only one predecessor)
///    are resolved — each phi `%x = phi [%v, A]` is replaced with `%v`.
/// 2. A's terminator (branch to B) is removed.
/// 3. B's remaining (non-phi) instructions are appended to A.
/// 4. A inherits B's successor list, and B's successors' predecessor lists
///    are updated to reference A instead of B.
/// 5. B is removed and block indices are compacted.
///
/// The entry block (index 0) is never removed, though it *can* absorb its
/// successor.
///
/// Returns `true` if any blocks were merged.
fn merge_blocks(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let mut removed: FxHashSet<usize> = FxHashSet::default();

    // Outer loop: repeat until no more merges are found.  Each merge can
    // expose new merge opportunities (e.g., A absorbs B, then A-C becomes
    // a new single-pred/single-succ pair).
    loop {
        let mut merged_this_pass = false;
        let num_blocks = func.block_count();

        for b_idx in 1..num_blocks {
            // Skip already-removed blocks.
            if removed.contains(&b_idx) {
                continue;
            }

            // B must have exactly one predecessor.
            let b_preds = &func.blocks[b_idx].predecessors;
            if b_preds.len() != 1 {
                continue;
            }
            let a_idx = b_preds[0];
            if removed.contains(&a_idx) {
                continue;
            }

            // A must have exactly one successor (which is B), connected by
            // an unconditional branch.
            let a_succs = &func.blocks[a_idx].successors;
            if a_succs.len() != 1 || a_succs[0] != b_idx {
                continue;
            }
            if let Some(term) = func.blocks[a_idx].terminator() {
                if is_unconditional_branch(term) != Some(BlockId(b_idx as u32)) {
                    continue;
                }
            } else {
                continue;
            }

            // --- Merge B into A ---

            // 1. Resolve B's phi nodes (trivial because B has only one pred).
            let b_phi_replacements: Vec<(Value, Value)> = func.blocks[b_idx]
                .instructions
                .iter()
                .filter(|inst| inst.is_phi())
                .filter_map(|inst| {
                    if let Instruction::Phi {
                        result, incoming, ..
                    } = inst
                    {
                        incoming.first().map(|&(val, _)| (*result, val))
                    } else {
                        None
                    }
                })
                .collect();

            for &(old_val, new_val) in &b_phi_replacements {
                replace_value_in_function(func, old_val, new_val);
            }

            // 2. Remove A's terminator (the branch to B).
            if func.blocks[a_idx].has_terminator() {
                func.blocks[a_idx].instructions.pop();
            }

            // 3. Move B's non-phi instructions to A.
            let b_insts: Vec<Instruction> = func.blocks[b_idx]
                .instructions
                .drain(..)
                .filter(|inst| !inst.is_phi())
                .collect();
            func.blocks[a_idx].instructions.extend(b_insts);

            // 4. A inherits B's successor list.
            let b_succs: Vec<usize> = func.blocks[b_idx].successors.clone();
            func.blocks[a_idx].successors.clear();
            for &s in &b_succs {
                func.blocks[a_idx].add_successor(s);
            }

            // 5. Update B's successors' predecessor lists: replace B → A.
            for &s_idx in &b_succs {
                if s_idx < num_blocks && !removed.contains(&s_idx) {
                    func.blocks[s_idx].remove_predecessor(b_idx);
                    func.blocks[s_idx].add_predecessor(a_idx);
                }
            }

            // 6. Update phi nodes in B's successors: predecessor B → A.
            for &s_idx in &b_succs {
                if s_idx < num_blocks && !removed.contains(&s_idx) {
                    for inst in func.blocks[s_idx].instructions_mut().iter_mut() {
                        replace_phi_predecessor(inst, BlockId(b_idx as u32), BlockId(a_idx as u32));
                    }
                }
            }

            // 7. Mark B for removal.
            func.blocks[b_idx].predecessors.clear();
            func.blocks[b_idx].successors.clear();
            removed.insert(b_idx);

            merged_this_pass = true;
            changed = true;
        }

        if !merged_this_pass {
            break;
        }
    }

    if !removed.is_empty() {
        remove_and_reindex(func, &removed);
    }

    changed
}

// ===========================================================================
// Pass 3 — Empty block elimination
// ===========================================================================

/// Removes empty blocks that contain only an unconditional branch (with
/// optional leading phi nodes).
///
/// For each empty block E that branches to target T, every predecessor P
/// of E is redirected to branch directly to T.  Phi nodes in T are updated
/// to replace incoming edges from E with edges from each predecessor P.
/// If E itself contains phi nodes, the values those phis would have selected
/// for each predecessor P are propagated into T's phis.
///
/// The entry block (index 0) is never removed.
///
/// Returns `true` if any blocks were removed.
fn eliminate_empty_blocks(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let mut removed: FxHashSet<usize> = FxHashSet::default();

    loop {
        let mut eliminated_this_pass = false;
        let num_blocks = func.block_count();

        for e_idx in 1..num_blocks {
            if removed.contains(&e_idx) {
                continue;
            }
            if !is_empty_block(&func.blocks[e_idx]) {
                continue;
            }

            // Determine the branch target T.
            let t_idx = match func.blocks[e_idx]
                .terminator()
                .and_then(is_unconditional_branch)
            {
                Some(bid) => bid.index(),
                None => continue,
            };
            if t_idx >= num_blocks || removed.contains(&t_idx) {
                continue;
            }
            // Self-loops are not eliminable.
            if e_idx == t_idx {
                continue;
            }

            // Collect information about E before mutation.
            let e_preds: Vec<usize> = func.blocks[e_idx].predecessors.clone();
            if e_preds.is_empty() {
                // Unreachable block — mark for removal directly.
                removed.insert(e_idx);
                eliminated_this_pass = true;
                changed = true;
                continue;
            }

            // Build phi resolution table for E's phis (if any).
            // Maps: phi_result_value -> { predecessor_index -> selected_value }
            let phi_resolution: FxHashMap<Value, FxHashMap<usize, Value>> = func.blocks[e_idx]
                .instructions
                .iter()
                .filter_map(|inst| {
                    if let Instruction::Phi {
                        result, incoming, ..
                    } = inst
                    {
                        let mut pred_to_val: FxHashMap<usize, Value> = FxHashMap::default();
                        for &(val, block_id) in incoming {
                            pred_to_val.insert(block_id.index(), val);
                        }
                        Some((*result, pred_to_val))
                    } else {
                        None
                    }
                })
                .collect();

            // Safety check: do NOT eliminate E if doing so would create
            // conflicting phi entries from the same predecessor in T.
            //
            // When both arms of a conditional branch are trivial (just
            // unconditional branches to the same merge block), eliminating
            // both would place two phi copies with DIFFERENT values in the
            // same predecessor block.  The second copy would overwrite the
            // first, destroying the conditional semantics.
            //
            // Example: block0 → block1(then) → block3(merge), block0 → block2(else) → block3
            //   phi in block3: m = phi(a from block1, b from block2)
            //   After eliminating block1: m = phi(a from block0, b from block2)
            //   After eliminating block2: m = phi(a from block0, b from block0) ← CONFLICT!
            //
            // We detect this by checking if any of E's predecessors P already
            // has an incoming entry in T's phis with a DIFFERENT value than
            // what we would insert.
            let mut would_conflict = false;
            for inst in func.blocks[t_idx].instructions.iter() {
                if let Instruction::Phi { incoming, .. } = inst {
                    // Build a map of existing incoming predecessor → value.
                    let existing: FxHashMap<usize, Value> = incoming
                        .iter()
                        .filter(|&&(_, bid)| bid != BlockId(e_idx as u32))
                        .map(|&(v, bid)| (bid.index(), v))
                        .collect();

                    // For each E-predecessor P, compute what value P would
                    // provide to T's phi if E were bypassed.
                    for &(val, block_ref) in incoming.iter() {
                        if block_ref != BlockId(e_idx as u32) {
                            continue;
                        }
                        // val is what E currently provides to T's phi.
                        for &p_idx in &e_preds {
                            let resolved_val = if let Some(pred_map) = phi_resolution.get(&val) {
                                pred_map.get(&p_idx).copied().unwrap_or(val)
                            } else {
                                val
                            };
                            // Check if P already has a DIFFERENT incoming value in this phi.
                            if let Some(&existing_val) = existing.get(&p_idx) {
                                if existing_val != resolved_val {
                                    would_conflict = true;
                                    break;
                                }
                            }
                        }
                        if would_conflict {
                            break;
                        }
                    }
                }
                if would_conflict {
                    break;
                }
            }
            if would_conflict {
                continue;
            }

            // Safety check: do NOT eliminate E if any of E's phi results
            // are used in instructions outside of E and outside of T's phi
            // incoming values.
            //
            // When E has a phi like `v617 = phi [v1, bb1], [v2, bb14], [v3, bb15]`
            // and v617 is used in a return statement at a distant block (e.g.,
            // `return v617` at the function exit), eliminating E would remove
            // the phi instruction but leave the dangling reference to v617.
            // The existing phi resolution only handles T's phi nodes, not
            // arbitrary uses of E's phi results throughout the function.
            let e_phi_results: FxHashSet<Value> = func.blocks[e_idx]
                .instructions
                .iter()
                .filter_map(|inst| {
                    if let Instruction::Phi { result, .. } = inst {
                        Some(*result)
                    } else {
                        None
                    }
                })
                .collect();

            if !e_phi_results.is_empty() {
                let mut phi_used_outside = false;
                // Check if any phi result is used in blocks other than E itself
                for (bi, blk) in func.blocks.iter().enumerate() {
                    if bi == e_idx {
                        continue;
                    }
                    for inst in blk.instructions.iter() {
                        // For phi nodes in T, the phi_resolution already handles the
                        // incoming values. But check uses in non-phi instructions.
                        let is_phi_incoming_in_t = bi == t_idx && inst.is_phi();
                        if is_phi_incoming_in_t {
                            // Phi incoming values in T are handled by the resolution.
                            // But check if ANY phi result from E is used as a non-incoming
                            // operand (e.g., used in the phi's own result).
                            continue;
                        }
                        // Check all operands of this instruction for uses of E's phi results.
                        let operands = inst.operands();
                        for &op in &operands {
                            if e_phi_results.contains(&op) {
                                phi_used_outside = true;
                                break;
                            }
                        }
                        if phi_used_outside {
                            break;
                        }
                    }
                    if phi_used_outside {
                        break;
                    }
                }
                if phi_used_outside {
                    continue;
                }
            }

            // --- Bypass E ---

            // 1. Redirect each predecessor P to target T directly.
            for &p_idx in &e_preds {
                if p_idx >= num_blocks || removed.contains(&p_idx) {
                    continue;
                }
                if let Some(term) = func.blocks[p_idx].terminator_mut() {
                    replace_block_reference_in_terminator(
                        term,
                        BlockId(e_idx as u32),
                        BlockId(t_idx as u32),
                    );
                }
                func.blocks[p_idx].remove_successor(e_idx);
                func.blocks[p_idx].add_successor(t_idx);
            }

            // 2. Update T's predecessor list.
            for &p_idx in &e_preds {
                func.blocks[t_idx].add_predecessor(p_idx);
            }
            func.blocks[t_idx].remove_predecessor(e_idx);

            // 3. Update phi nodes in T: replace incoming edges from E with
            //    edges from each predecessor P of E.  If E had its own phi
            //    nodes, resolve the value through them.
            for inst in func.blocks[t_idx].instructions_mut().iter_mut() {
                if let Instruction::Phi { incoming, .. } = inst {
                    let mut new_entries: Vec<(Value, BlockId)> = Vec::new();

                    incoming.retain(|&(val, block_ref)| {
                        if block_ref == BlockId(e_idx as u32) {
                            // Expand the single E-entry into per-predecessor entries.
                            for &p_idx in &e_preds {
                                let resolved = if let Some(pred_map) = phi_resolution.get(&val) {
                                    // val is a phi result in E — look up what E's phi
                                    // would produce when coming from P.
                                    pred_map.get(&p_idx).copied().unwrap_or(val)
                                } else {
                                    // val is not from E's phis — propagate directly.
                                    val
                                };
                                new_entries.push((resolved, BlockId(p_idx as u32)));
                            }
                            false // remove the original E-entry
                        } else {
                            true // keep
                        }
                    });

                    incoming.extend(new_entries);
                }
            }

            // 4. Mark E for removal.
            func.blocks[e_idx].predecessors.clear();
            func.blocks[e_idx].successors.clear();
            removed.insert(e_idx);

            eliminated_this_pass = true;
            changed = true;
        }

        if !eliminated_this_pass {
            break;
        }
    }

    if !removed.is_empty() {
        remove_and_reindex(func, &removed);
    }

    changed
}

// ===========================================================================
// Pass 4 — Branch chain simplification (jump threading)
// ===========================================================================

/// Simplifies chains of unconditional branches through pure trampoline
/// blocks (blocks containing a single unconditional branch and nothing else).
///
/// When A's terminator branches to B, and B is a pure trampoline that
/// branches to C, A is redirected to branch directly to C.  Chains are
/// followed up to [`MAX_CHAIN_HOPS`] to avoid infinite traversal on
/// cyclic empty-block graphs.
///
/// Returns `true` if any branch targets were redirected.
fn simplify_branch_chains(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let num_blocks = func.block_count();

    for block_idx in 0..num_blocks {
        // Snapshot the terminator to avoid borrow conflicts.
        let term_snapshot = match func.blocks[block_idx].terminator() {
            Some(t) => t.clone(),
            None => continue,
        };

        match &term_snapshot {
            Instruction::Branch { target, .. } => {
                let mut final_target = *target;
                let mut hops = 0;

                // Follow chain through pure trampoline blocks,
                // checking phi conflicts at each hop.
                while hops < MAX_CHAIN_HOPS {
                    let t_idx = final_target.index();
                    if t_idx >= num_blocks {
                        break;
                    }
                    match get_pure_branch_target(&func.blocks[t_idx]) {
                        Some(next) if next != final_target => {
                            if would_chain_redirect_conflict_phi(
                                func,
                                t_idx,
                                next.index(),
                                block_idx,
                            ) {
                                break;
                            }
                            final_target = next;
                            hops += 1;
                        }
                        _ => break,
                    }
                }

                if final_target != *target {
                    // Rewrite the terminator.
                    if let Some(Instruction::Branch { target: t, .. }) =
                        func.blocks[block_idx].terminator_mut()
                    {
                        *t = final_target;
                    }
                    // Update predecessor / successor edges.
                    let old_succ = target.index();
                    let new_succ = final_target.index();
                    func.blocks[block_idx].remove_successor(old_succ);
                    func.blocks[block_idx].add_successor(new_succ);
                    if old_succ < num_blocks {
                        func.blocks[old_succ].remove_predecessor(block_idx);
                    }
                    if new_succ < num_blocks {
                        func.blocks[new_succ].add_predecessor(block_idx);
                        // Propagate phi incoming: add entry from block_idx using
                        // the same values that the intermediate block contributed.
                        propagate_phi_for_chain(func, old_succ, new_succ, block_idx);
                    }
                    changed = true;
                }
            }

            Instruction::CondBranch {
                then_block,
                else_block,
                ..
            } => {
                let mut then_final = *then_block;
                let mut else_final = *else_block;

                // Resolve then-branch chain, checking phi conflicts at each hop.
                let mut hops = 0;
                while hops < MAX_CHAIN_HOPS {
                    let t_idx = then_final.index();
                    if t_idx >= num_blocks {
                        break;
                    }
                    match get_pure_branch_target(&func.blocks[t_idx]) {
                        Some(next) if next != then_final => {
                            // Guard: don't follow chain if it would create
                            // conflicting phi entries in the target.
                            if would_chain_redirect_conflict_phi(
                                func,
                                t_idx,
                                next.index(),
                                block_idx,
                            ) {
                                break;
                            }
                            then_final = next;
                            hops += 1;
                        }
                        _ => break,
                    }
                }

                // Resolve else-branch chain, checking phi conflicts at each hop.
                hops = 0;
                while hops < MAX_CHAIN_HOPS {
                    let t_idx = else_final.index();
                    if t_idx >= num_blocks {
                        break;
                    }
                    match get_pure_branch_target(&func.blocks[t_idx]) {
                        Some(next) if next != else_final => {
                            if would_chain_redirect_conflict_phi(
                                func,
                                t_idx,
                                next.index(),
                                block_idx,
                            ) {
                                break;
                            }
                            else_final = next;
                            hops += 1;
                        }
                        _ => break,
                    }
                }

                if then_final != *then_block || else_final != *else_block {
                    // Update then-branch edge.
                    if then_final != *then_block {
                        let old_t = then_block.index();
                        let new_t = then_final.index();
                        func.blocks[block_idx].remove_successor(old_t);
                        func.blocks[block_idx].add_successor(new_t);
                        if old_t < num_blocks {
                            func.blocks[old_t].remove_predecessor(block_idx);
                        }
                        if new_t < num_blocks {
                            func.blocks[new_t].add_predecessor(block_idx);
                            propagate_phi_for_chain(func, old_t, new_t, block_idx);
                        }
                    }
                    // Update else-branch edge.
                    if else_final != *else_block {
                        let old_e = else_block.index();
                        let new_e = else_final.index();
                        func.blocks[block_idx].remove_successor(old_e);
                        func.blocks[block_idx].add_successor(new_e);
                        if old_e < num_blocks {
                            func.blocks[old_e].remove_predecessor(block_idx);
                        }
                        if new_e < num_blocks {
                            func.blocks[new_e].add_predecessor(block_idx);
                            propagate_phi_for_chain(func, old_e, new_e, block_idx);
                        }
                    }
                    // Rewrite the terminator.
                    if let Some(Instruction::CondBranch {
                        then_block: tb,
                        else_block: eb,
                        ..
                    }) = func.blocks[block_idx].terminator_mut()
                    {
                        *tb = then_final;
                        *eb = else_final;
                    }
                    changed = true;
                }
            }

            // Switch chains could also be resolved, but they are rare and
            // not worth the complexity for the basic pass.
            _ => {}
        }
    }

    changed
}

/// When a branch chain A → B → … → C is short-circuited to A → C, any phi
/// nodes in C that reference the old intermediate block B need an additional
/// incoming entry from A with the same value that B contributed.
/// Returns `true` if redirecting `from_block → final_target` (bypassing
/// `old_intermediate`) would create conflicting phi entries in `final_target`.
///
/// A conflict exists when `final_target` has a phi that already has an
/// incoming entry from `from_block` with a value DIFFERENT from the value
/// that `old_intermediate` contributes.
fn would_chain_redirect_conflict_phi(
    func: &IrFunction,
    old_intermediate: usize,
    final_target: usize,
    from_block: usize,
) -> bool {
    if final_target >= func.blocks.len() {
        return false;
    }
    let old_bid = BlockId(old_intermediate as u32);
    let from_bid = BlockId(from_block as u32);

    for inst in func.blocks[final_target].instructions.iter() {
        if !inst.is_phi() {
            break;
        }
        if let Instruction::Phi { incoming, .. } = inst {
            // Find the value contributed by the old intermediate block.
            let intermediate_val = incoming
                .iter()
                .find(|&&(_, b)| b == old_bid)
                .map(|&(v, _)| v);
            // Find the value already contributed by from_block (if any).
            let existing_val = incoming
                .iter()
                .find(|&&(_, b)| b == from_bid)
                .map(|&(v, _)| v);
            if let (Some(new_v), Some(old_v)) = (intermediate_val, existing_val) {
                if new_v != old_v {
                    return true;
                }
            }
        }
    }
    false
}

fn propagate_phi_for_chain(
    func: &mut IrFunction,
    old_intermediate: usize,
    final_target: usize,
    from_block: usize,
) {
    if final_target >= func.blocks.len() {
        return;
    }
    let old_bid = BlockId(old_intermediate as u32);
    let new_bid = BlockId(from_block as u32);

    let mut additions: Vec<(usize, Value, BlockId)> = Vec::new();

    for (inst_idx, inst) in func.blocks[final_target].instructions.iter().enumerate() {
        if !inst.is_phi() {
            break;
        }
        if let Instruction::Phi { incoming, .. } = inst {
            // Find the value contributed by the old intermediate block.
            for &(val, block_ref) in incoming {
                if block_ref == old_bid {
                    additions.push((inst_idx, val, new_bid));
                    break;
                }
            }
        }
    }

    // Apply additions.
    for (inst_idx, val, block_ref) in additions {
        if let Instruction::Phi { incoming, .. } =
            &mut func.blocks[final_target].instructions[inst_idx]
        {
            // Only add if not already present from this predecessor.
            if !incoming.iter().any(|&(_, b)| b == block_ref) {
                incoming.push((val, block_ref));
            }
        }
    }
}

// ===========================================================================
// Public entry point
// ===========================================================================

/// Runs the CFG simplification pass on a single IR function.
///
/// Applies four transformations in a fixed order chosen for maximum
/// effectiveness:
///
/// 1. **Trivial phi elimination** — reduces phi nodes first, enabling more
///    block merges.
/// 2. **Block merging** — collapses single-predecessor / single-successor
///    pairs into a single block.
/// 3. **Empty block elimination** — redirects predecessors past blocks that
///    only contain a branch.
/// 4. **Branch chain simplification** — short-circuits remaining chains of
///    unconditional branches.
///
/// Returns `true` if any changes were made (for fixpoint iteration by the
/// pass manager).
pub fn run_simplify_cfg(func: &mut IrFunction) -> bool {
    let mut any_changed = false;

    // First, strip dead code after the first terminator in each block.
    // This is essential for correctness: after lowering, a block may contain
    // an IndirectBranch followed by a dead unconditional Branch (from the
    // while(1) back-edge in computed-goto patterns).  Every function that
    // calls `block.terminator()` — which returns the LAST terminator —
    // would see the dead Branch instead of the IndirectBranch, computing
    // wrong successor edges and causing phi copies to be missing for some
    // targets.  By stripping dead post-terminator instructions first, we
    // guarantee that every block has exactly one terminator as its last
    // instruction, making all subsequent passes correct.
    any_changed |= strip_dead_code_after_first_terminator(func);

    any_changed |= recompute_cfg_and_cleanup_phis(func);
    any_changed |= simplify_trivial_phis(func);
    any_changed |= merge_blocks(func);
    any_changed |= eliminate_empty_blocks(func);
    any_changed |= simplify_branch_chains(func);

    any_changed
}

/// Removes all instructions that follow the first terminator in each basic
/// block.  In well-formed IR a block should end with exactly one terminator,
/// but the lowering phase can produce blocks where an `IndirectBranch` is
/// immediately followed by a dead `Branch` (e.g., the back-edge of a
/// `while(1)` loop enclosing a computed-goto dispatch).  The dead trailing
/// instructions are unreachable and confuse later analyses that use
/// `block.terminator()` (which returns the *last* instruction).
fn strip_dead_code_after_first_terminator(func: &mut IrFunction) -> bool {
    let mut changed = false;
    for block_idx in 0..func.block_count() {
        let instructions = func.blocks[block_idx].instructions_mut();
        // Find the position of the first terminator instruction.
        if let Some(first_term_pos) = instructions.iter().position(|inst| inst.is_terminator()) {
            // If there are instructions after the first terminator, remove them.
            if first_term_pos + 1 < instructions.len() {
                instructions.truncate(first_term_pos + 1);
                changed = true;
            }
        }
    }
    changed
}

// ===========================================================================
// Pre-pass — CFG edge recomputation and stale phi incoming cleanup
// ===========================================================================

/// Recomputes predecessor and successor lists from actual block terminators,
/// then removes phi incoming entries whose source block is no longer a
/// predecessor.
///
/// This fixes a class of bugs where branch folding (in constant_folding.rs)
/// converts a conditional branch to an unconditional one but does not update
/// the stored CFG edges or phi incoming vectors.  The stale entries can cause
/// incorrect phi resolution during block merging.
///
/// Returns `true` if any phi incoming entries were removed.
fn recompute_cfg_and_cleanup_phis(func: &mut IrFunction) -> bool {
    let num_blocks = func.block_count();

    // Step 1: Compute actual successor sets from terminators.
    let mut actual_succs: Vec<Vec<usize>> = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let mut succs = Vec::new();
        if let Some(term) = func.blocks[block_idx].terminator() {
            match term {
                Instruction::Branch { target, .. } => {
                    succs.push(target.index());
                }
                Instruction::CondBranch {
                    then_block,
                    else_block,
                    ..
                } => {
                    succs.push(then_block.index());
                    if else_block.index() != then_block.index() {
                        succs.push(else_block.index());
                    }
                }
                Instruction::Switch { default, cases, .. } => {
                    succs.push(default.index());
                    for &(_val, blk) in cases {
                        if !succs.contains(&blk.index()) {
                            succs.push(blk.index());
                        }
                    }
                }
                Instruction::IndirectBranch {
                    possible_targets, ..
                } => {
                    for target in possible_targets {
                        let idx = target.index();
                        if !succs.contains(&idx) {
                            succs.push(idx);
                        }
                    }
                }
                _ => {}
            }
        }
        actual_succs.push(succs);
    }

    // Step 2: Compute actual predecessor sets from the successor sets.
    let mut actual_preds: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for (block_idx, succs) in actual_succs.iter().enumerate() {
        for &s in succs {
            if s < num_blocks && !actual_preds[s].contains(&block_idx) {
                actual_preds[s].push(block_idx);
            }
        }
    }

    // Step 3: Update predecessor and successor lists.
    for (block_idx, (succ, pred)) in actual_succs
        .iter()
        .zip(actual_preds.iter())
        .enumerate()
        .take(num_blocks)
    {
        func.blocks[block_idx].successors = succ.clone();
        func.blocks[block_idx].predecessors = pred.clone();
    }

    // Step 4: Remove stale phi incoming entries.
    let mut any_removed = false;
    for (block_idx, preds) in actual_preds.iter().enumerate().take(num_blocks) {
        let instructions = func.blocks[block_idx].instructions_mut();
        for inst in instructions.iter_mut() {
            if let Instruction::Phi { incoming, .. } = inst {
                let before_len = incoming.len();
                incoming.retain(|&(_val, blk)| preds.contains(&blk.index()));
                if incoming.len() != before_len {
                    any_removed = true;
                }
            }
        }
    }

    any_removed
}
