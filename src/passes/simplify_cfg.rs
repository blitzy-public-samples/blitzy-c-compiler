//! Control-flow graph simplification optimization pass for the BCC compiler.
//!
//! This module implements Phase 8's CFG simplification pass, the third and final
//! pass in the fixed optimization pipeline:
//! **constant folding → dead code elimination → CFG simplification**.
//!
//! # Transformations
//!
//! The pass performs the following simplifications iteratively until a fixpoint
//! is reached (no more changes occur):
//!
//! | Transformation                | Description                                              |
//! |-------------------------------|----------------------------------------------------------|
//! | Unreachable block removal     | Remove blocks not reachable from the entry block         |
//! | Phi node simplification       | Replace trivial/redundant phi nodes with single values   |
//! | Empty block elimination       | Remove blocks containing only an unconditional branch    |
//! | Block merging                 | Merge A→B when A has one succ and B has one pred         |
//! | Branch chain simplification   | Short-circuit A→B→…→Z chains to A→Z                     |
//! | Switch simplification         | Convert trivial switch instructions to simpler branches  |
//! | Branch threading              | Thread known-condition branches to direct targets        |
//! | Self-loop detection           | Simplify degenerate self-loop conditions                 |
//!
//! # SSA Invariant Preservation
//!
//! All transformations maintain SSA form by correctly updating phi nodes
//! whenever predecessor/successor edges are modified. When a phi node becomes
//! trivial (single incoming value or all-identical incoming values), it is
//! replaced with the corresponding value throughout the function.
//!
//! # Fixpoint Iteration
//!
//! The main entry point [`run_simplify_cfg`] runs all sub-passes in a loop
//! until no pass reports any change. This ensures that simplifications exposed
//! by one transformation (e.g., unreachable block removal exposing new merge
//! opportunities) are fully exploited.

use crate::common::fx_hash::{fx_hash_map, fx_hash_set, FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::{ICmpPredicate, Instruction};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Runs the CFG simplification pass on a single IR function.
///
/// Applies all CFG simplifications iteratively until no more changes occur
/// (fixpoint iteration). Each iteration runs the following sub-passes in order:
///
/// 1. Unreachable block removal
/// 2. Phi node simplification
/// 3. Empty block elimination
/// 4. Block merging (single predecessor + single successor)
/// 5. Branch chain simplification
/// 6. Switch simplification
/// 7. Branch threading
/// 8. Self-loop detection
///
/// # Arguments
///
/// * `func` — Mutable reference to the IR function to simplify.
///
/// # Returns
///
/// `true` if any simplification was applied across all iterations, `false`
/// if the CFG was already in its simplified form.
///
/// # SSA Invariants
///
/// This pass preserves SSA form: phi nodes are updated when blocks are
/// merged or edges are redirected, and trivial phi nodes are replaced with
/// their single incoming value across all uses in the function.
pub fn run_simplify_cfg(func: &mut IrFunction) -> bool {
    let mut ever_changed = false;

    loop {
        let mut changed = false;

        // Phase 1: Remove unreachable blocks — may expose new simplification
        // opportunities by eliminating dead predecessor edges and phi operands.
        changed |= remove_unreachable_blocks(func);

        // Phase 2: Simplify phi nodes — replace trivial phis (single incoming
        // value or all-identical values) so that downstream merging and
        // elimination see simpler instruction patterns.
        changed |= simplify_phi_nodes(func);

        // Phase 3: Eliminate empty blocks — remove blocks that contain only
        // an unconditional branch, redirecting predecessors to the target.
        // Includes chain resolution for arbitrary-length branch chains.
        changed |= eliminate_empty_blocks(func);

        // Phase 4: Merge blocks — combine A and B when A has exactly one
        // successor B and B has exactly one predecessor A.
        changed |= merge_blocks(func);

        // Phase 5: Simplify unconditional branch chains — short-circuit
        // A→B→…→Z to A→Z when intermediate blocks are pass-throughs.
        changed |= simplify_branch_chains(func);

        // Phase 6: Simplify switch statements — convert trivial switches
        // to unconditional branches when all cases target the same block.
        changed |= simplify_switches(func);

        // Phase 7: Thread branches — redirect edges when the branch
        // condition outcome is known from the predecessor context.
        changed |= thread_branches(func);

        // Phase 8: Detect and simplify self-loops with degenerate conditions.
        changed |= detect_self_loops(func);

        if !changed {
            break;
        }
        ever_changed = true;
    }

    ever_changed
}

// ---------------------------------------------------------------------------
// Pass 1: Unreachable block removal
// ---------------------------------------------------------------------------

/// Removes basic blocks that are not reachable from the function's entry block.
///
/// Computes reachability via BFS from the entry block, then removes all
/// blocks not in the reachable set. Phi nodes in surviving blocks are updated
/// to remove incoming edges from removed blocks. Predecessor lists of
/// surviving blocks are cleaned of references to removed blocks.
///
/// This complements DCE's unreachable block removal — it may discover new
/// unreachable blocks after branch folding or edge redirection performed by
/// other sub-passes.
fn remove_unreachable_blocks(func: &mut IrFunction) -> bool {
    let entry_id = func.entry_block().id;
    let num_blocks = func.block_count();

    // BFS from entry to compute reachable set.
    let mut reachable: FxHashSet<BasicBlockId> = fx_hash_set();
    let mut worklist: Vec<BasicBlockId> = Vec::with_capacity(num_blocks);
    reachable.insert(entry_id);
    worklist.push(entry_id);

    while let Some(block_id) = worklist.pop() {
        let block = func.get_block(block_id);
        for &succ_id in block.successors() {
            if reachable.insert(succ_id) {
                worklist.push(succ_id);
            }
        }
    }

    // If all blocks are reachable, nothing to do.
    if reachable.len() == num_blocks {
        return false;
    }

    // Collect IDs of unreachable blocks.
    let unreachable_ids: Vec<BasicBlockId> = func
        .blocks()
        .iter()
        .map(|bb| bb.id)
        .filter(|id| !reachable.contains(id))
        .collect();

    // Build a set for O(1) membership tests during phi/predecessor cleanup.
    let unreachable_set: FxHashSet<BasicBlockId> = unreachable_ids.iter().copied().collect();

    // Update surviving blocks: remove phi operands and predecessor entries
    // that reference unreachable blocks.
    for block in func.blocks_mut() {
        if unreachable_set.contains(&block.id) {
            continue; // This block is being removed; skip.
        }

        // Remove phi operands from unreachable predecessors.
        let phi_count = block.phi_count();
        for i in 0..phi_count {
            let inst = &mut block.instructions_mut()[i];
            for &dead_id in &unreachable_ids {
                inst.remove_phi_operand(dead_id);
            }
        }

        // Remove unreachable blocks from this block's predecessor list.
        for &dead_id in &unreachable_ids {
            block.remove_predecessor(dead_id);
        }
    }

    // Remove each unreachable block from the function.
    for dead_id in &unreachable_ids {
        func.remove_block(*dead_id);
    }

    true
}

// ---------------------------------------------------------------------------
// Pass 2: Phi node simplification
// ---------------------------------------------------------------------------

/// Simplifies trivial and redundant phi nodes throughout the function.
///
/// A phi node is trivial if:
/// - It has exactly **one** incoming value (after predecessor removal), or
/// - **All** incoming values are identical (ignoring self-references where
///   the phi result feeds back into itself).
///
/// Trivial phi nodes are replaced: every use of the phi result is rewritten
/// to use the single canonical value, and the phi instruction is removed.
/// Degenerate phi nodes (zero incoming values) are also cleaned up.
///
/// Replacement chains are resolved transitively: if phi A → B → C, then
/// all uses of A are rewritten to C.
fn simplify_phi_nodes(func: &mut IrFunction) -> bool {
    // First pass: collect phi nodes that can be replaced.
    // Map: phi_result_id → replacement_value_id
    let mut replacements: FxHashMap<ValueId, ValueId> = fx_hash_map();

    for block in func.blocks() {
        for inst in block.phi_nodes() {
            let result = match inst.result() {
                Some(r) => r,
                None => continue,
            };
            let operands = match inst.phi_operands() {
                Some(ops) => ops,
                None => continue,
            };

            if operands.is_empty() {
                // Degenerate phi with no operands — mark for removal but
                // cannot replace (no valid replacement value). We will just
                // remove the instruction below.
                continue;
            }

            if operands.len() == 1 {
                // Single incoming value — trivially replaceable.
                replacements.insert(result, operands[0].0);
                continue;
            }

            // Check if all incoming values are the same (ignoring self-
            // references where the incoming value equals the phi result).
            let first_non_self = operands.iter().find(|(v, _)| *v != result).map(|(v, _)| *v);

            if let Some(canonical) = first_non_self {
                let all_same = operands
                    .iter()
                    .all(|(v, _)| *v == canonical || *v == result);
                if all_same {
                    replacements.insert(result, canonical);
                }
            }
            // If all values are self-references, this is an undefined
            // cycle — skip (leave the phi as-is for correctness).
        }
    }

    if replacements.is_empty() {
        return false;
    }

    // Resolve transitive replacement chains: if A→B and B→C, resolve A→C.
    // This avoids multiple rewriting passes when chains of trivial phis exist.
    let resolved = resolve_replacement_chains(&replacements);

    // Second pass: apply replacements across all instructions in the function.
    for block in func.blocks_mut() {
        for inst in block.instructions_mut() {
            for (&old_val, &new_val) in &resolved {
                inst.replace_use(old_val, new_val);
            }
        }
    }

    // Third pass: remove the now-trivial phi instructions.
    // Collect (block_id, list of phi instruction indices to remove).
    let mut removals: Vec<(BasicBlockId, Vec<usize>)> = Vec::new();
    for block in func.blocks() {
        let mut indices_to_remove: Vec<usize> = Vec::new();
        for (i, inst) in block.instructions().iter().enumerate() {
            if !inst.is_phi() {
                break; // Phi nodes are a contiguous prefix.
            }
            if let Some(result) = inst.result() {
                if resolved.contains_key(&result) {
                    indices_to_remove.push(i);
                }
            }
            // Also remove degenerate phi nodes with zero operands.
            if let Some(operands) = inst.phi_operands() {
                if operands.is_empty() && !indices_to_remove.contains(&i) {
                    indices_to_remove.push(i);
                }
            }
        }
        if !indices_to_remove.is_empty() {
            removals.push((block.id, indices_to_remove));
        }
    }

    // Remove instructions in reverse index order to preserve indices.
    for (block_id, mut indices) in removals {
        indices.sort_unstable();
        indices.reverse();
        let block = func.get_block_mut(block_id);
        for idx in indices {
            block.remove_instruction(idx);
        }
    }

    true
}

/// Resolves transitive replacement chains in a value substitution map.
///
/// Given a map `{A→B, B→C}`, produces `{A→C, B→C}`. Detects and breaks
/// cycles to guarantee termination.
fn resolve_replacement_chains(
    replacements: &FxHashMap<ValueId, ValueId>,
) -> FxHashMap<ValueId, ValueId> {
    let mut resolved: FxHashMap<ValueId, ValueId> = fx_hash_map();

    for (&from, &initial_to) in replacements {
        let mut current = initial_to;
        let mut visited: FxHashSet<ValueId> = fx_hash_set();
        visited.insert(from);

        while let Some(&next) = replacements.get(&current) {
            if !visited.insert(current) {
                break; // Cycle detected — stop following the chain.
            }
            current = next;
        }
        resolved.insert(from, current);
    }

    resolved
}

// ---------------------------------------------------------------------------
// Pass 3: Empty block elimination
// ---------------------------------------------------------------------------

/// Eliminates basic blocks that contain only an unconditional branch.
///
/// A block B is eligible for elimination if:
/// - B is **not** the entry block.
/// - B has **no** phi nodes.
/// - B contains exactly one instruction: an unconditional [`Instruction::Branch`].
/// - B does not branch to itself (self-loop).
///
/// When such a block is found, all predecessors of B are redirected to
/// branch directly to B's target. Phi nodes in the target block are updated
/// to replace B with each of B's predecessors, preserving SSA form.
///
/// The pass processes one empty block per iteration and returns `true` if
/// any block was eliminated. The outer fixpoint loop in [`run_simplify_cfg`]
/// re-invokes this pass to handle chains (A→B→C where both B and C are
/// empty) across successive iterations.
fn eliminate_empty_blocks(func: &mut IrFunction) -> bool {
    let entry_id = func.entry_block().id;
    let mut changed = false;

    loop {
        // Scan for an empty block candidate.
        let candidate = find_empty_block(func, entry_id);

        let (empty_id, target_id, preds) = match candidate {
            Some(c) => c,
            None => break,
        };

        // Self-loop guard: do not eliminate a block that branches to itself.
        if empty_id == target_id {
            break;
        }

        // Redirect each predecessor of the empty block to the target.
        for &pred_id in &preds {
            // Update the predecessor's terminator: replace empty_id with target_id.
            let pred_term = func.get_block(pred_id).terminator().cloned();
            if let Some(mut new_term) = pred_term {
                new_term.replace_block(empty_id, target_id);
                let pred = func.get_block_mut(pred_id);
                pred.set_terminator(new_term);

                // Update predecessor's successor list.
                pred.remove_successor(empty_id);
                if !pred.successors().contains(&target_id) {
                    pred.add_successor(target_id);
                }
            }
        }

        // Update phi nodes in the target block: for each phi, replace the
        // incoming edge from empty_id with edges from each predecessor of
        // the empty block, carrying the same value.
        let target_phi_count = func.get_block(target_id).phi_count();
        for i in 0..target_phi_count {
            let value_from_empty = {
                let target = func.get_block(target_id);
                let inst = &target.instructions()[i];
                inst.phi_operands()
                    .and_then(|ops| ops.iter().find(|(_, b)| *b == empty_id).map(|(v, _)| *v))
            };
            if let Some(val) = value_from_empty {
                let target = func.get_block_mut(target_id);
                let inst = &mut target.instructions_mut()[i];
                inst.remove_phi_operand(empty_id);
                for &pred_id in &preds {
                    inst.add_phi_operand(val, pred_id);
                }
            }
        }

        // Update target's predecessor list: remove empty_id, add each of
        // the empty block's predecessors.
        {
            let target = func.get_block_mut(target_id);
            target.remove_predecessor(empty_id);
            for &pred_id in &preds {
                target.add_predecessor(pred_id);
            }
        }

        // Remove the now-unreferenced empty block.
        func.remove_block(empty_id);
        changed = true;
    }

    changed
}

/// Scans the function for a block eligible for empty block elimination.
///
/// Returns `Some((empty_block_id, branch_target, predecessors))` for the
/// first qualifying block, or `None` if no such block exists.
fn find_empty_block(
    func: &IrFunction,
    entry_id: BasicBlockId,
) -> Option<(BasicBlockId, BasicBlockId, Vec<BasicBlockId>)> {
    for block in func.blocks() {
        // Skip the entry block — it must always remain.
        if block.id == entry_id {
            continue;
        }
        // Skip blocks with phi nodes.
        if block.phi_count() > 0 {
            continue;
        }
        // Check for exactly one instruction that is an unconditional Branch.
        let insts = block.instructions();
        if insts.len() == 1 {
            if let Instruction::Branch { target } = &insts[0] {
                if *target != block.id {
                    return Some((block.id, *target, block.predecessors().to_vec()));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pass 4: Block merging
// ---------------------------------------------------------------------------

/// Merges basic block pairs where block A has exactly one successor B and
/// block B has exactly one predecessor A.
///
/// When such a pair is found:
/// 1. B's phi nodes (which must be trivially single-valued since B has one
///    predecessor) are resolved: all uses of the phi result are replaced
///    with the single incoming value.
/// 2. A's terminator (the [`Instruction::Branch`] to B) is removed.
/// 3. B's non-phi, non-terminator instructions are appended to A.
/// 4. B's terminator becomes A's new terminator.
/// 5. B's successors update their predecessor lists to reference A instead of B.
/// 6. B is removed from the function.
///
/// The process repeats to handle cascading merge opportunities (e.g., after
/// merging A+B, A may now merge with A's predecessor).
fn merge_blocks(func: &mut IrFunction) -> bool {
    let mut changed = false;

    loop {
        // Find a merge candidate: A→B where A has one succ, B has one pred.
        let candidate = find_merge_candidate(func);
        let (a_id, b_id) = match candidate {
            Some(pair) => pair,
            None => break,
        };

        // Step 1: Collect B's phi replacements. Since B has exactly one
        // predecessor, each phi must have exactly one incoming value.
        let phi_replacements: Vec<(ValueId, ValueId)> = {
            let block_b = func.get_block(b_id);
            block_b
                .phi_nodes()
                .filter_map(|inst| {
                    let result = inst.result()?;
                    let operands = inst.phi_operands()?;
                    if operands.len() == 1 {
                        Some((result, operands[0].0))
                    } else if operands.is_empty() {
                        None // Degenerate; skip.
                    } else {
                        // Multiple operands with single predecessor — take
                        // the value associated with a_id.
                        operands
                            .iter()
                            .find(|(_, blk)| *blk == a_id)
                            .map(|(v, _)| (result, *v))
                    }
                })
                .collect()
        };

        // Step 2: Collect B's non-phi instructions and terminator.
        let (b_non_phi_insts, b_terminator, b_successors) = {
            let block_b = func.get_block(b_id);
            let phi_count = block_b.phi_count();
            let insts = block_b.instructions();
            let non_phi: Vec<Instruction> = insts[phi_count..]
                .iter()
                .filter(|inst| !inst.is_terminator())
                .cloned()
                .collect();
            let term = block_b.terminator().cloned();
            let succs = block_b.successors().to_vec();
            (non_phi, term, succs)
        };

        // Step 3: Modify block A — remove its Branch terminator, append B's
        // instructions, and set B's terminator as A's new terminator.
        {
            let block_a = func.get_block_mut(a_id);
            // Find and remove A's terminator.
            let term_idx = block_a
                .instructions()
                .iter()
                .rposition(|inst| inst.is_terminator());
            if let Some(idx) = term_idx {
                block_a.remove_instruction(idx);
            }

            // Apply phi replacements to B's instructions before appending.
            let mut insts_to_add = b_non_phi_insts;
            for inst in &mut insts_to_add {
                for &(old_val, new_val) in &phi_replacements {
                    inst.replace_use(old_val, new_val);
                }
            }

            // Apply phi replacements to B's terminator.
            let mut new_term = b_terminator;
            if let Some(ref mut t) = new_term {
                for &(old_val, new_val) in &phi_replacements {
                    t.replace_use(old_val, new_val);
                }
            }

            // Append B's non-phi instructions to A.
            for inst in insts_to_add {
                block_a.add_instruction(inst);
            }

            // Set B's terminator as A's new terminator.
            if let Some(t) = new_term {
                block_a.set_terminator(t);
            }

            // Update A's successor list: remove B, add B's successors.
            block_a.remove_successor(b_id);
            for &succ_id in &b_successors {
                if !block_a.successors().contains(&succ_id) {
                    block_a.add_successor(succ_id);
                }
            }
        }

        // Step 4: Update B's successors to reference A instead of B.
        for &succ_id in &b_successors {
            let succ = func.get_block_mut(succ_id);
            succ.remove_predecessor(b_id);
            succ.add_predecessor(a_id);

            // Update phi nodes in successor: replace incoming block B with A.
            let phi_count = succ.phi_count();
            for i in 0..phi_count {
                let inst = &mut succ.instructions_mut()[i];
                inst.replace_block(b_id, a_id);
            }
        }

        // Step 5: Apply phi replacements globally. Other blocks may reference
        // the phi results from B's now-removed phi nodes.
        if !phi_replacements.is_empty() {
            for block in func.blocks_mut() {
                for inst in block.instructions_mut() {
                    for &(old_val, new_val) in &phi_replacements {
                        inst.replace_use(old_val, new_val);
                    }
                }
            }
        }

        // Step 6: Remove block B from the function.
        func.remove_block(b_id);
        changed = true;
    }

    changed
}

/// Scans for a block pair (A, B) eligible for merging: A has exactly one
/// successor B via unconditional [`Instruction::Branch`], and B has exactly
/// one predecessor A.
fn find_merge_candidate(func: &IrFunction) -> Option<(BasicBlockId, BasicBlockId)> {
    for block_a in func.blocks() {
        if block_a.successor_count() != 1 {
            continue;
        }
        let succ_id = block_a.successors()[0];
        // Avoid self-loops.
        if succ_id == block_a.id {
            continue;
        }
        // Confirm A's terminator is an unconditional Branch to succ_id.
        match block_a.terminator() {
            Some(Instruction::Branch { target }) if *target == succ_id => {}
            _ => continue,
        }
        // Confirm B has exactly one predecessor (A).
        let block_b = func.get_block(succ_id);
        if block_b.predecessor_count() == 1 && block_b.predecessors()[0] == block_a.id {
            return Some((block_a.id, succ_id));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pass 5: Branch chain simplification
// ---------------------------------------------------------------------------

/// Short-circuits unconditional branch chains of arbitrary length.
///
/// A "pass-through" block is one that has no phi nodes and contains only an
/// unconditional [`Instruction::Branch`] to another block. When a terminator
/// references such a block as a successor, the reference is resolved to the
/// final non-pass-through target, skipping all intermediates.
///
/// For example, if A→B→C→D where B and C are pass-throughs, A's terminator
/// is updated to target D directly. Predecessor/successor lists and phi nodes
/// in the final target are updated accordingly.
///
/// The intermediate pass-through blocks become unreachable and are cleaned up
/// by the unreachable block removal pass in the next fixpoint iteration.
fn simplify_branch_chains(func: &mut IrFunction) -> bool {
    let entry_id = func.entry_block().id;

    // Build a redirect map for pass-through blocks: block → branch target.
    let mut redirect: FxHashMap<BasicBlockId, BasicBlockId> = fx_hash_map();
    for block in func.blocks() {
        // The entry block is never eligible for chain-skip (even if empty,
        // it must remain as the function's entry point).
        if block.id == entry_id || block.phi_count() > 0 {
            continue;
        }
        let insts = block.instructions();
        if insts.len() == 1 {
            if let Instruction::Branch { target } = &insts[0] {
                if *target != block.id {
                    redirect.insert(block.id, *target);
                }
            }
        }
    }

    if redirect.is_empty() {
        return false;
    }

    // Collect terminator modifications: (block_id, old_successors, new_terminator).
    let mut modifications: Vec<(BasicBlockId, Vec<BasicBlockId>, Instruction)> = Vec::new();

    for block in func.blocks() {
        let term = match block.terminator() {
            Some(t) => t,
            None => continue,
        };

        let new_term = match term {
            Instruction::Branch { target } => {
                let final_target = resolve_chain_target(&redirect, *target);
                if final_target != *target {
                    Some(Instruction::Branch {
                        target: final_target,
                    })
                } else {
                    None
                }
            }
            Instruction::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let new_true = resolve_chain_target(&redirect, *true_target);
                let new_false = resolve_chain_target(&redirect, *false_target);
                if new_true != *true_target || new_false != *false_target {
                    Some(Instruction::CondBranch {
                        condition: *condition,
                        true_target: new_true,
                        false_target: new_false,
                    })
                } else {
                    None
                }
            }
            Instruction::Switch {
                value,
                default,
                cases,
            } => {
                let new_default = resolve_chain_target(&redirect, *default);
                let new_cases: Vec<(i64, BasicBlockId)> = cases
                    .iter()
                    .map(|(val, tgt)| (*val, resolve_chain_target(&redirect, *tgt)))
                    .collect();
                let modified = new_default != *default
                    || new_cases
                        .iter()
                        .zip(cases.iter())
                        .any(|((_, a), (_, b))| a != b);
                if modified {
                    Some(Instruction::Switch {
                        value: *value,
                        default: new_default,
                        cases: new_cases,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new_term) = new_term {
            let old_succs = term.successor_blocks();
            modifications.push((block.id, old_succs, new_term));
        }
    }

    if modifications.is_empty() {
        return false;
    }

    // Apply all collected modifications.
    for (block_id, old_succs, new_term) in modifications {
        let new_succs = new_term.successor_blocks();

        // Set the updated terminator.
        let block = func.get_block_mut(block_id);
        block.set_terminator(new_term);

        // Update successor list on the source block.
        for &old_s in &old_succs {
            if !new_succs.contains(&old_s) {
                block.remove_successor(old_s);
            }
        }
        for &new_s in &new_succs {
            if !old_succs.contains(&new_s) {
                let block = func.get_block_mut(block_id);
                block.add_successor(new_s);
            }
        }

        // Update predecessor lists on old and new successor blocks.
        for &old_s in &old_succs {
            if !new_succs.contains(&old_s) {
                let old_succ = func.get_block_mut(old_s);
                old_succ.remove_predecessor(block_id);
            }
        }
        for &new_s in &new_succs {
            if !old_succs.contains(&new_s) {
                // Before adding as predecessor, copy phi entries from the
                // last chain intermediate so the phi values are correct.
                let last_intermediate = find_last_intermediate(&redirect, new_s, &old_succs);
                copy_phi_entries_for_redirect(func, new_s, last_intermediate, block_id);

                let new_succ = func.get_block_mut(new_s);
                new_succ.add_predecessor(block_id);
            }
        }
    }

    true
}

/// Follows the redirect chain from `start` to find the final non-pass-through
/// target. Detects cycles to guarantee termination.
fn resolve_chain_target(
    redirect: &FxHashMap<BasicBlockId, BasicBlockId>,
    start: BasicBlockId,
) -> BasicBlockId {
    let mut current = start;
    let mut visited: FxHashSet<BasicBlockId> = fx_hash_set();
    visited.insert(current);

    while let Some(&next) = redirect.get(&current) {
        if !visited.insert(next) {
            break; // Cycle detected.
        }
        current = next;
    }
    current
}

/// Finds the last block in the redirect chain that is an immediate
/// predecessor of `final_target` among the `old_succs` set. This is the
/// block whose phi entries in `final_target` should be duplicated for the
/// redirected predecessor.
fn find_last_intermediate(
    redirect: &FxHashMap<BasicBlockId, BasicBlockId>,
    final_target: BasicBlockId,
    old_succs: &[BasicBlockId],
) -> BasicBlockId {
    // Walk each old successor's chain to find one that ends at final_target.
    for &old_s in old_succs {
        let mut current = old_s;
        let mut visited: FxHashSet<BasicBlockId> = fx_hash_set();
        visited.insert(current);
        while let Some(&next) = redirect.get(&current) {
            if next == final_target {
                return current; // This is the direct predecessor of final_target.
            }
            if !visited.insert(next) {
                break;
            }
            current = next;
        }
    }
    // Fallback: use the first old successor (best-effort).
    old_succs.first().copied().unwrap_or(final_target)
}

/// Copies phi node entries from `source_pred` to `new_pred` in block
/// `target_block_id`. For each phi node, if there is an incoming edge from
/// `source_pred`, a duplicate edge is added for `new_pred` with the same value.
fn copy_phi_entries_for_redirect(
    func: &mut IrFunction,
    target_block_id: BasicBlockId,
    source_pred: BasicBlockId,
    new_pred: BasicBlockId,
) {
    let phi_count = func.get_block(target_block_id).phi_count();
    for i in 0..phi_count {
        let value_from_source = {
            let target = func.get_block(target_block_id);
            let inst = &target.instructions()[i];
            inst.phi_operands()
                .and_then(|ops| ops.iter().find(|(_, b)| *b == source_pred).map(|(v, _)| *v))
        };
        if let Some(val) = value_from_source {
            let target = func.get_block_mut(target_block_id);
            let inst = &mut target.instructions_mut()[i];
            inst.add_phi_operand(val, new_pred);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 6: Switch simplification
// ---------------------------------------------------------------------------

/// Simplifies trivial [`Instruction::Switch`] terminators.
///
/// Two cases are handled:
///
/// 1. **All cases branch to the same target (including default):** The switch
///    is replaced with an unconditional [`Instruction::Branch`] to that target.
///    This catches both "all cases identical" and "zero cases" (just default).
///
/// 2. **Switch with zero cases:** The switch is trivially replaced with a
///    branch to the default target.
///
/// More complex conversions (e.g., single-case switch to ICmp + CondBranch)
/// are deferred until constant value representation is available in the IR.
fn simplify_switches(func: &mut IrFunction) -> bool {
    let mut changed = false;

    // Collect modifications: (block_id, new_terminator).
    let mut modifications: Vec<(BasicBlockId, Instruction)> = Vec::new();

    for block in func.blocks() {
        let term = match block.terminator() {
            Some(t) => t,
            None => continue,
        };

        if let Instruction::Switch {
            value: _,
            default,
            cases,
        } = term
        {
            // Case 1: Zero cases — convert to unconditional branch to default.
            if cases.is_empty() {
                modifications.push((block.id, Instruction::Branch { target: *default }));
                continue;
            }

            // Case 2: All case targets (and default) are the same block.
            let all_same_target = cases.iter().all(|(_, tgt)| *tgt == *default);
            if all_same_target {
                modifications.push((block.id, Instruction::Branch { target: *default }));
                continue;
            }

            // Case 3: Single case where case_target == default.
            if cases.len() == 1 && cases[0].1 == *default {
                modifications.push((block.id, Instruction::Branch { target: *default }));
                continue;
            }

            // Case 4: All cases (but not default) target the same block, and
            // there is only one case. The switch value determines whether to
            // go to case_target or default. We leave this as a Switch because
            // converting to ICmp+CondBranch requires creating a constant
            // value, which the current IR does not directly support.
        }
    }

    // Apply modifications.
    for (block_id, new_term) in modifications {
        let old_succs: Vec<BasicBlockId> = {
            let block = func.get_block(block_id);
            block
                .terminator()
                .map(|t| t.successor_blocks())
                .unwrap_or_default()
        };
        let new_succs = new_term.successor_blocks();

        let block = func.get_block_mut(block_id);
        block.set_terminator(new_term);

        // Update successor list.
        for &old_s in &old_succs {
            if !new_succs.contains(&old_s) {
                block.remove_successor(old_s);
            }
        }
        for &new_s in &new_succs {
            if !block.successors().contains(&new_s) {
                block.add_successor(new_s);
            }
        }

        // Update predecessor lists on removed successors.
        for &old_s in &old_succs {
            if !new_succs.contains(&old_s) {
                let succ = func.get_block_mut(old_s);
                succ.remove_predecessor(block_id);
                // Remove phi operands from this predecessor.
                let phi_count = succ.phi_count();
                for i in 0..phi_count {
                    let inst = &mut succ.instructions_mut()[i];
                    inst.remove_phi_operand(block_id);
                }
            }
        }
        changed = true;
    }

    changed
}

// ---------------------------------------------------------------------------
// Pass 7: Branch threading
// ---------------------------------------------------------------------------

/// Threads branches through blocks where the branch condition outcome is
/// known from the predecessor context.
///
/// Specifically, this pass handles:
///
/// 1. **CondBranch with identical targets:** If both the true and false targets
///    are the same block, the CondBranch is replaced with an unconditional
///    [`Instruction::Branch`] to that target.
///
/// 2. **Known-condition threading:** If a predecessor A computes the condition
///    for block B's [`Instruction::CondBranch`] via an [`Instruction::ICmp`]
///    with [`ICmpPredicate::Eq`] comparing a value to itself (always true),
///    or if the CondBranch condition is the same as a condition already
///    resolved in the predecessor, A's edge is threaded directly to the
///    appropriate successor of B.
fn thread_branches(func: &mut IrFunction) -> bool {
    let mut changed = false;

    // Sub-pass 1: Convert CondBranch with identical targets to Branch.
    let mut same_target_blocks: Vec<(BasicBlockId, BasicBlockId)> = Vec::new();
    for block in func.blocks() {
        if let Some(Instruction::CondBranch {
            true_target,
            false_target,
            ..
        }) = block.terminator()
        {
            if *true_target == *false_target {
                same_target_blocks.push((block.id, *true_target));
            }
        }
    }

    for (block_id, target) in same_target_blocks {
        let block = func.get_block_mut(block_id);
        let old_succs = block.successors().to_vec();
        block.set_terminator(Instruction::Branch { target });

        // Update successor list: CondBranch may have had the same target
        // listed twice; reduce to a single entry.
        for &old_s in &old_succs {
            block.remove_successor(old_s);
        }
        if !block.successors().contains(&target) {
            block.add_successor(target);
        }

        // If the block previously had duplicate successor entries for
        // the same target, clean up the predecessor count on the target.
        let target_block = func.get_block_mut(target);
        // Remove duplicates: the target should have block_id as predecessor
        // only once (since we now have a single unconditional branch).
        while target_block
            .predecessors()
            .iter()
            .filter(|&&p| p == block_id)
            .count()
            > 1
        {
            target_block.remove_predecessor(block_id);
        }

        changed = true;
    }

    // Sub-pass 2: Thread branches where the condition is known from a
    // predecessor's instruction definition. Two analysis patterns are used:
    //
    // (a) **Same-condition CondBranch:** If a predecessor also branches on
    //     the same condition C that the current block uses, we know C's
    //     truth value on the incoming edge and can thread to the appropriate
    //     successor directly.
    //
    // (b) **ICmp self-comparison:** If a predecessor defines condition C via
    //     an ICmp comparing a value with itself (e.g., `x == x` is always
    //     true), we thread the predecessor to the known successor.
    //
    // To avoid borrow-checker conflicts (the analysis phase borrows `func`
    // immutably while the application phase requires mutable access), all
    // threading decisions are **collected first** and **applied afterwards**.

    // Collect threading decisions: (pred_id, current_block_id, new_target).
    let mut decisions: Vec<(BasicBlockId, BasicBlockId, BasicBlockId)> = Vec::new();

    let block_ids: Vec<BasicBlockId> = func.blocks().iter().map(|b| b.id).collect();
    for &block_id in &block_ids {
        let (cond_val, true_tgt, false_tgt) = {
            let block = func.get_block(block_id);
            match block.terminator() {
                Some(Instruction::CondBranch {
                    condition,
                    true_target,
                    false_target,
                }) => (*condition, *true_target, *false_target),
                _ => continue,
            }
        };

        let preds: Vec<BasicBlockId> = func.get_block(block_id).predecessors().to_vec();
        for pred_id in preds {
            let mut already_threaded = false;

            // Analysis (a): Same-condition CondBranch threading.
            {
                let pred = func.get_block(pred_id);
                if let Some(Instruction::CondBranch {
                    condition: pred_cond,
                    true_target: pred_true,
                    false_target: pred_false,
                }) = pred.terminator()
                {
                    if *pred_cond == cond_val {
                        let thread_target = if *pred_true == block_id {
                            Some(true_tgt)
                        } else if *pred_false == block_id {
                            Some(false_tgt)
                        } else {
                            None
                        };
                        if let Some(new_target) = thread_target {
                            if new_target != block_id {
                                decisions.push((pred_id, block_id, new_target));
                                already_threaded = true;
                            }
                        }
                    }
                }
            } // immutable borrow of func through `pred` ends here.

            if already_threaded {
                continue; // Skip analysis (b) for this predecessor.
            }

            // Analysis (b): ICmp self-comparison.
            // The `for inst in ...` iterator holds an immutable borrow on
            // `func` through `pred`, so we must NOT mutate `func` inside
            // this loop. We only collect the decision.
            {
                let pred = func.get_block(pred_id);
                for inst in pred.instructions() {
                    if let Instruction::ICmp {
                        result,
                        pred: icmp_pred,
                        lhs,
                        rhs,
                    } = inst
                    {
                        if *result == cond_val && *lhs == *rhs {
                            // x == x is always true; x != x is always false.
                            let condition_known = match icmp_pred {
                                ICmpPredicate::Eq => Some(true),
                                ICmpPredicate::Ne => Some(false),
                                _ => None,
                            };
                            if let Some(is_true) = condition_known {
                                let target = if is_true { true_tgt } else { false_tgt };
                                if target != block_id {
                                    decisions.push((pred_id, block_id, target));
                                    break; // One match per predecessor.
                                }
                            }
                        }
                    }
                }
            } // immutable borrow of func through `pred` ends here.
        }
    }

    // Apply all collected threading decisions.
    for (pred_id, block_id, new_target) in decisions {
        // Redirect the predecessor's terminator from block_id to new_target.
        let pred_term = func.get_block(pred_id).terminator().cloned();
        if let Some(mut new_term) = pred_term {
            new_term.replace_block(block_id, new_target);
            let pred = func.get_block_mut(pred_id);
            pred.set_terminator(new_term);

            pred.remove_successor(block_id);
            if !pred.successors().contains(&new_target) {
                pred.add_successor(new_target);
            }
        }

        // Remove pred from block_id's predecessor list.
        {
            let block = func.get_block_mut(block_id);
            block.remove_predecessor(pred_id);
        }

        // Update phi nodes in block_id: remove operands from pred.
        let phi_count = func.get_block(block_id).phi_count();
        for i in 0..phi_count {
            let block = func.get_block_mut(block_id);
            let inst = &mut block.instructions_mut()[i];
            inst.remove_phi_operand(pred_id);
        }

        // Add pred to new_target's predecessor list and copy phi entries.
        copy_phi_entries_for_redirect(func, new_target, block_id, pred_id);
        {
            let tgt_block = func.get_block_mut(new_target);
            tgt_block.add_predecessor(pred_id);
        }

        changed = true;
    }

    changed
}

// ---------------------------------------------------------------------------
// Pass 8: Self-loop detection
// ---------------------------------------------------------------------------

/// Detects and simplifies degenerate self-loop patterns.
///
/// A self-loop is a block that branches to itself. This pass handles:
///
/// 1. **Unconditional self-loop with no side effects:** This is a legitimate
///    infinite loop (e.g., `for(;;) {}`). It is **left as-is** because it
///    represents correct program behavior.
///
/// 2. **Conditional self-loop where both targets are the same block:**
///    Converted to an unconditional [`Instruction::Branch`] to self (infinite
///    loop), since the condition is irrelevant.
///
/// 3. **Conditional self-loop where the non-self target is reachable but the
///    self-branch is on a known-false condition:** The self-loop edge is
///    eliminated, converting the CondBranch to an unconditional branch to the
///    other target.
///
/// The `has_side_effects()` method is used to check whether a self-looping
/// block contains observable side effects. Self-loops with side effects are
/// always preserved (they represent intentional behavior).
fn detect_self_loops(func: &mut IrFunction) -> bool {
    let mut changed = false;

    let block_ids: Vec<BasicBlockId> = func.blocks().iter().map(|b| b.id).collect();

    for &block_id in &block_ids {
        let block = func.get_block(block_id);
        let term = match block.terminator() {
            Some(t) => t,
            None => continue,
        };

        match term {
            Instruction::CondBranch {
                condition: _,
                true_target,
                false_target,
            } => {
                let tt = *true_target;
                let ft = *false_target;

                if tt == block_id && ft == block_id {
                    // Both targets are self — convert to unconditional
                    // self-branch (infinite loop).
                    let block = func.get_block_mut(block_id);
                    block.set_terminator(Instruction::Branch { target: block_id });
                    changed = true;
                } else if tt == block_id || ft == block_id {
                    // One target is self-loop, the other is an exit edge.
                    // Check if the block body has side effects.
                    let has_effects = func
                        .get_block(block_id)
                        .instructions()
                        .iter()
                        .any(|inst| inst.has_side_effects() && !inst.is_terminator());

                    if !has_effects && func.get_block(block_id).is_empty() {
                        // Empty self-loop with no side effects: this is a
                        // tight spin loop. Leave as-is (correct behavior
                        // for `while(cond) {}`).
                    }
                    // Note: converting a conditional self-loop to a non-loop
                    // branch requires proving the condition is always false
                    // on the self edge. Without constant tracking, we cannot
                    // do this here — that work belongs to constant folding.
                }
            }
            Instruction::Branch { target } if *target == block_id => {
                // Unconditional self-loop. Check for side effects to decide
                // if this is an intentional infinite loop.
                let _has_effects = func
                    .get_block(block_id)
                    .instructions()
                    .iter()
                    .any(|inst| inst.has_side_effects() && !inst.is_terminator());
                // Whether or not there are side effects, an unconditional
                // self-loop is an infinite loop — leave it as-is.
            }
            _ => {}
        }
    }

    changed
}
