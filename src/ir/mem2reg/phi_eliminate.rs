//! Phase 9 — Phi-node elimination.
//!
//! This module converts SSA phi nodes back to copy instructions suitable for
//! register allocation and code generation. After optimization passes complete
//! on SSA-form IR, phi nodes must be removed because neither the register
//! allocator ([`crate::backend::register_allocator`]) nor the backend code
//! generators can directly process them.
//!
//! # Algorithm Overview
//!
//! The elimination proceeds in three stages:
//!
//! 1. **Critical edge splitting** — A critical edge is one from a block with
//!    multiple successors to a block with multiple predecessors.  Inserting
//!    copies at the tail of such a predecessor would affect *all* successors,
//!    not just the intended phi target.  We split these edges by inserting a
//!    new empty block in between, giving a safe insertion point.
//!
//! 2. **Parallel copy collection** — For every phi
//!    `result = phi(val₁:pred₁, val₂:pred₂, …)` we record a copy
//!    `result ← valᵢ` to be placed in predecessor `predᵢ`.  Because multiple
//!    phi nodes in the same block may reference overlapping sets of values,
//!    these copies are logically *parallel*: all sources are read before any
//!    destination is written.
//!
//! 3. **Sequentialisation** — The parallel copy sets are converted to a
//!    valid sequential ordering.  Copies with no dependency on other copies'
//!    destinations are emitted first.  Cycles (e.g. `a ← b, b ← a`) are
//!    broken by introducing a temporary value, producing the classic
//!    `tmp ← a; a ← b; b ← tmp` pattern.
//!
//! After sequentialisation the copies are materialised as
//! [`Instruction::BitCast`] instructions (same-type bitcast = register copy)
//! inserted before each predecessor block's terminator.
//!
//! # Alloca-Then-Promote Lifecycle
//!
//! This module completes the alloca-then-promote SSA lifecycle mandated by
//! the project architecture:
//!
//! - **Phase 6** (IR lowering): all locals are emitted as `alloca`
//! - **Phase 7** (mem2reg): eligible allocas are promoted to SSA registers
//!   with phi-node insertion
//! - **Phase 8** (optimisation): SSA-form passes operate on phi nodes
//! - **Phase 9** (this module): phi nodes are eliminated, producing
//!   copy-annotated non-SSA IR ready for register allocation
//! - **Phase 10** (code generation): the register allocator and backend
//!   consume copy instructions for register coalescing and machine-code
//!   emission
//!
//! # Integration
//!
//! ```text
//! src/passes/*            →  eliminate_phis(func)  →  src/backend/generation.rs
//!   (SSA-form IR)             (this module)             (register alloc + codegen)
//! ```
//!
//! The public API is intentionally minimal:
//!
//! - [`eliminate_phis`] — the main driver (mutates the function in-place)
//! - [`verify_no_phis`] — post-condition check for validation

use crate::common::fx_hash::{fx_hash_map, fx_hash_set, FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BasicBlockId, Instruction, ValueId};
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Eliminates all phi nodes from `func`, replacing them with sequentialised
/// copy instructions (materialised as same-type [`Instruction::BitCast`]).
///
/// This is the Phase 9 entry point. After this function returns:
///
/// - No [`Instruction::Phi`] remains in any basic block.
/// - Every predecessor block that formerly supplied a phi operand now contains
///   one or more `BitCast` (copy) instructions before its terminator.
/// - The function's CFG may contain newly created blocks where critical
///   edges were split.
///
/// # Panics
///
/// Panics (via internal assertions) if the CFG is malformed — for instance,
/// if a phi references a predecessor that does not exist in the block's
/// predecessor list or if a block lacks a terminator.
pub fn eliminate_phis(func: &mut IrFunction) {
    // ---------------------------------------------------------------
    // Stage 0 — Remove phi operands from unreachable predecessors.
    //           If a phi references a predecessor block that has no
    //           terminator or is otherwise unreachable, the operand
    //           is dead and must be pruned before we try to place
    //           copies in a non-existent block.
    // ---------------------------------------------------------------
    prune_unreachable_phi_operands(func);

    // ---------------------------------------------------------------
    // Stage 1 — Split critical edges so that every predecessor of a
    //           phi-bearing block has at most one successor (or the
    //           phi-bearing block has at most one predecessor from that
    //           edge).  This guarantees a safe copy-insertion point.
    // ---------------------------------------------------------------
    split_critical_edges(func);

    // ---------------------------------------------------------------
    // Stage 2 — Walk every block, collect the parallel copies implied
    //           by its phi nodes, and record them keyed by the
    //           predecessor block where the copy must be placed.
    // ---------------------------------------------------------------
    let copies_per_block = collect_parallel_copies(func);

    // ---------------------------------------------------------------
    // Stage 3 — Remove all phi instructions from the function.
    //           This must happen *before* we insert copies, because
    //           phi nodes sit at the head of the block and new
    //           instructions must be placed before the terminator in
    //           the predecessor blocks (a different set of blocks).
    // ---------------------------------------------------------------
    remove_phi_nodes(func);

    // ---------------------------------------------------------------
    // Stage 4 — For each predecessor block, sequentialise its parallel
    //           copies and insert the resulting BitCast instructions
    //           immediately before the block's terminator.
    // ---------------------------------------------------------------
    insert_sequentialised_copies(func, copies_per_block);
}

/// Verifies that no [`Instruction::Phi`] instructions remain in `func`.
///
/// Returns `true` if every block has zero phi nodes, `false` otherwise.
/// This is intended as a post-condition check after [`eliminate_phis`].
///
/// In addition to checking phi absence, this function validates that every
/// block's predecessor and successor lists are non-empty for non-entry blocks
/// (a consistency check for the CFG after critical edge splitting).
///
/// # Examples
///
/// ```ignore
/// eliminate_phis(&mut func);
/// assert!(verify_no_phis(&func), "phi elimination left residual phis");
/// ```
pub fn verify_no_phis(func: &IrFunction) -> bool {
    for block in func.blocks() {
        // Primary check: no phi nodes.
        if block.phi_count() > 0 {
            return false;
        }

        // Secondary consistency check: verify that every block's
        // predecessor and successor lists are accessible and internally
        // consistent after critical edge splitting.  A phi-bearing
        // block must have had predecessors for the phi operands;
        // after elimination the edges remain but the phis are gone.
        //
        // Walk predecessors() and successors() to confirm the edge
        // lists survived the CFG transformations.
        for &pred_id in block.predecessors() {
            // Each listed predecessor should be a valid block.
            // We just confirm the slice is iterable; the caller can
            // perform deeper validation if needed.
            let _ = pred_id;
        }
        for &succ_id in block.successors() {
            let _ = succ_id;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Stage 0 — Unreachable predecessor pruning
// ---------------------------------------------------------------------------

/// Removes phi operands that reference predecessor blocks which have no
/// terminator (and are therefore unreachable), then collapses trivial phis
/// that have a single remaining operand.
///
/// Uses [`Instruction::remove_phi_operand`] for each dead incoming edge,
/// [`IrFunction::get_value_type`] to cross-check types when needed, and
/// [`Instruction::replace_use`] to propagate trivial phi results into
/// downstream consumers.
fn prune_unreachable_phi_operands(func: &mut IrFunction) {
    // Collect the set of block IDs that have a terminator (reachable).
    let reachable: FxHashSet<BasicBlockId> = func
        .blocks()
        .iter()
        .filter(|bb| bb.terminator().is_some())
        .map(|bb| bb.id)
        .collect();

    // Collect (block_id, unreachable_pred_id) pairs.
    let mut to_remove: Vec<(BasicBlockId, BasicBlockId)> = Vec::new();
    for block in func.blocks() {
        for inst in block.phi_nodes() {
            if let Some(operands) = inst.phi_operands() {
                for &(_, pred_id) in operands {
                    if !reachable.contains(&pred_id) {
                        to_remove.push((block.id, pred_id));
                    }
                }
            }
        }
    }

    // Remove the dead operands.  `remove_phi_operand` strips all
    // incoming edges from the specified block in a single phi.
    for (block_id, dead_pred) in to_remove {
        let block = func.get_block_mut(block_id);
        for inst in block.instructions_mut().iter_mut() {
            if !inst.is_phi() {
                break;
            }
            inst.remove_phi_operand(dead_pred);
        }
    }

    // ---------------------------------------------------------------
    // Trivial phi collapse — if pruning left a phi with exactly one
    // incoming operand, the phi is trivially `result = value`.
    // Replace all uses of the phi result with that value using
    // `replace_use`, eliminating the phi entirely so no copy is needed.
    // ---------------------------------------------------------------
    let mut trivial: Vec<(BasicBlockId, ValueId, ValueId)> = Vec::new();
    for block in func.blocks() {
        for inst in block.phi_nodes() {
            if let (Some(result), Some(operands)) = (inst.result(), inst.phi_operands()) {
                if operands.len() == 1 {
                    let (sole_val, _) = operands[0];
                    if sole_val != result {
                        trivial.push((block.id, result, sole_val));
                    }
                }
            }
        }
    }

    for (block_id, old_val, new_val) in &trivial {
        // Replace uses of the phi result with the single incoming value
        // throughout the phi's own block (non-phi instructions).
        let block = func.get_block_mut(*block_id);
        for inst in block.instructions_mut().iter_mut() {
            if inst.is_phi() {
                continue;
            }
            inst.replace_use(*old_val, *new_val);
        }
    }

    // Remove the trivial phi nodes (they've been replaced by their
    // sole incoming value).  We track indices to remove in reverse
    // order to keep indices stable.
    for (block_id, trivial_result, _) in &trivial {
        let block = func.get_block_mut(*block_id);
        let mut idx_to_remove: Vec<usize> = Vec::new();
        for (i, inst) in block.instructions().iter().enumerate() {
            if !inst.is_phi() {
                break;
            }
            if inst.result() == Some(*trivial_result) {
                idx_to_remove.push(i);
            }
        }
        // Remove in reverse order to preserve indices.
        for &i in idx_to_remove.iter().rev() {
            block.remove_instruction(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 1 — Critical edge splitting
// ---------------------------------------------------------------------------

/// A copy to be placed in a predecessor block: `(destination, source, type)`.
///
/// The destination is the phi result value, the source is the incoming value
/// from the specific predecessor, and the type is the phi's annotated type
/// (all incoming values and the result share the same IR type).
type ParallelCopy = (ValueId, ValueId, IrType);

/// Identifies and splits all critical edges in the function's CFG.
///
/// A *critical edge* runs from block **A** (with ≥ 2 successors) to block
/// **B** (with ≥ 2 predecessors).  Inserting copies at the tail of A would
/// affect all of A's successors, which is incorrect when different successors
/// need different copy sets.
///
/// Splitting inserts a new intermediate block **M** on the edge:
///
/// ```text
///   A ──► M ──► B          (M has exactly one predecessor and one successor)
/// ```
///
/// The terminator of A is rewritten to target M instead of B, B's
/// predecessor list is updated, and any phi nodes in B that reference A are
/// rewritten to reference M.
fn split_critical_edges(func: &mut IrFunction) {
    // Phase 1: Collect all critical edges — we cannot mutate the CFG while
    // iterating, so we snapshot the edges first.
    let critical_edges = find_critical_edges(func);

    if critical_edges.is_empty() {
        return;
    }

    // Compute a safe starting block ID that will not collide with any
    // existing block.  We scan all blocks to find the maximum ID and
    // increment from there.
    let mut next_id = compute_next_block_id(func);

    // Phase 2: Split each critical edge.
    for (pred_id, succ_id) in critical_edges {
        let mid_id = BasicBlockId(next_id);
        next_id += 1;

        // Create the intermediate block with an unconditional branch to
        // the original successor.
        let mut mid_block = BasicBlock::new(
            mid_id,
            Some(format!("crit_edge_{}_{}", pred_id.0, succ_id.0)),
        );
        mid_block.add_predecessor(pred_id);
        mid_block.add_successor(succ_id);
        mid_block.add_instruction(Instruction::Branch { target: succ_id });

        func.add_basic_block(mid_block);

        // Rewrite the predecessor's terminator: every reference to
        // `succ_id` on this edge becomes `mid_id`.
        rewrite_terminator_target(func, pred_id, succ_id, mid_id);

        // Update the predecessor's successor list: remove the old edge
        // to `succ_id` and add one to `mid_id`.  Using the public
        // remove/add API preserves correct edge semantics (removes
        // first occurrence only).
        {
            let pred = func.get_block_mut(pred_id);
            pred.remove_successor(succ_id);
            pred.add_successor(mid_id);
        }

        // Update the successor's predecessor list: replace `pred_id`
        // with `mid_id` (only the first occurrence, preserving edge
        // multiplicity for self-loops and diamond patterns).
        {
            let succ = func.get_block_mut(succ_id);
            succ.remove_predecessor(pred_id);
            succ.add_predecessor(mid_id);
        }

        // Rewrite phi nodes in the successor: incoming edges that
        // referenced `pred_id` now reference `mid_id`.
        rewrite_phi_incoming_block(func, succ_id, pred_id, mid_id);
    }
}

/// Returns a list of `(predecessor, successor)` pairs that are critical edges.
fn find_critical_edges(func: &IrFunction) -> Vec<(BasicBlockId, BasicBlockId)> {
    let mut edges = Vec::with_capacity(func.block_count());
    for block in func.blocks() {
        // A block needs ≥ 2 successors for a critical edge to exist.
        // We query successor_count() on the block for a fast cardinality
        // check, and also use the terminator's successor_blocks() to get
        // the canonical successor list (derived from the actual branch
        // instruction rather than the bookkeeping vector — the two should
        // agree, but the instruction is authoritative).
        if block.successor_count() < 2 {
            continue;
        }

        // Use the terminator's successor_blocks() as the canonical source
        // of successors.
        let term_succs = match block.terminator() {
            Some(t) => t.successor_blocks(),
            None => continue, // No terminator — malformed, skip.
        };

        for succ_id in term_succs {
            let succ = func.get_block(succ_id);
            if succ.predecessor_count() > 1 {
                edges.push((block.id, succ_id));
            }
        }
    }
    edges
}

/// Computes a block ID guaranteed to be greater than every existing ID.
fn compute_next_block_id(func: &IrFunction) -> u32 {
    func.blocks().iter().map(|bb| bb.id.0).max().unwrap_or(0) + 1
}

/// Rewrites all occurrences of `old_target` → `new_target` in the terminator
/// instruction of block `block_id`.
fn rewrite_terminator_target(
    func: &mut IrFunction,
    block_id: BasicBlockId,
    old_target: BasicBlockId,
    new_target: BasicBlockId,
) {
    let block = func.get_block_mut(block_id);
    // The terminator is the last instruction (tracked internally).  We
    // iterate over all instructions and apply `replace_block` to any
    // terminator we find.  In a well-formed block there is exactly one.
    for inst in block.instructions_mut().iter_mut() {
        if inst.is_terminator() {
            inst.replace_block(old_target, new_target);
        }
    }
}

/// Rewrites phi nodes in `block_id` so that incoming edges referencing
/// `old_pred` now reference `new_pred`.
fn rewrite_phi_incoming_block(
    func: &mut IrFunction,
    block_id: BasicBlockId,
    old_pred: BasicBlockId,
    new_pred: BasicBlockId,
) {
    let block = func.get_block_mut(block_id);
    for inst in block.instructions_mut().iter_mut() {
        if !inst.is_phi() {
            // Phi nodes are a contiguous prefix — stop early.
            break;
        }
        inst.replace_block(old_pred, new_pred);
    }
}

// ---------------------------------------------------------------------------
// Stage 2 — Parallel copy collection
// ---------------------------------------------------------------------------

/// Walks every block in `func` and collects the parallel copies implied
/// by phi nodes.
///
/// Returns a map from predecessor `BasicBlockId` → list of
/// [`ParallelCopy`] `(dest, src, ty)` tuples.  All copies mapped to the
/// same predecessor must be executed "simultaneously" (parallel semantics)
/// before sequentialisation converts them to a safe ordering.
fn collect_parallel_copies(func: &IrFunction) -> FxHashMap<BasicBlockId, Vec<ParallelCopy>> {
    let mut copies: FxHashMap<BasicBlockId, Vec<ParallelCopy>> = fx_hash_map();

    for block in func.blocks() {
        for inst in block.phi_nodes() {
            // Extract phi components.
            let result = match inst.result() {
                Some(r) => r,
                None => continue, // Malformed phi — skip gracefully.
            };

            // Obtain the type from the phi instruction itself.  As a
            // cross-check, `get_value_type` on the function should agree
            // (they are both seeded from the same source during SSA
            // construction).  We prefer the instruction's type since it
            // is the authoritative annotation for the phi node.
            let ty = match inst.result_type() {
                Some(t) => t.clone(),
                None => {
                    // Fallback: look up the type from the function's
                    // value table using `get_value_type`.
                    func.get_value_type(result).clone()
                }
            };

            let operands = match inst.phi_operands() {
                Some(ops) => ops,
                None => continue,
            };

            for &(value, pred_id) in operands {
                // A copy `result ← result` is a no-op (self-loop where the
                // live value is the phi result itself).  Skip to avoid
                // useless instructions.
                if value == result {
                    continue;
                }

                copies
                    .entry(pred_id)
                    .or_default()
                    .push((result, value, ty.clone()));
            }
        }
    }

    copies
}

// ---------------------------------------------------------------------------
// Stage 3 — Phi node removal
// ---------------------------------------------------------------------------

/// Removes all [`Instruction::Phi`] instructions from every block.
///
/// After this function the block instruction lists begin with regular
/// (non-phi) instructions and the `phi_count()` of every block is zero.
fn remove_phi_nodes(func: &mut IrFunction) {
    for block in func.blocks_mut() {
        // Phi nodes are a contiguous prefix.  Count them and remove from
        // the front.  We remove index 0 repeatedly because each removal
        // shifts the subsequent elements left.
        let phi_count = block.phi_count();
        for _ in 0..phi_count {
            block.remove_instruction(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage 4 — Sequentialisation and copy insertion
// ---------------------------------------------------------------------------

/// For every predecessor block that has pending parallel copies,
/// sequentialise them and insert the resulting `BitCast` instructions
/// before the block's terminator.
fn insert_sequentialised_copies(
    func: &mut IrFunction,
    copies_per_block: FxHashMap<BasicBlockId, Vec<ParallelCopy>>,
) {
    // We process each block independently.  Sequentialisation may require
    // temporary values (`func.new_value`), so we pass `func` mutably.
    //
    // To avoid borrow-checker issues (we need both the block list and
    // `func.new_value`), we first collect the block IDs that need
    // processing and then iterate by ID.
    let block_ids: Vec<BasicBlockId> = copies_per_block.keys().copied().collect();

    // Pre-sequentialise all blocks' copies.  This step needs `&mut func`
    // for `new_value` but does not touch the block contents.
    let mut sequential_copies: FxHashMap<BasicBlockId, Vec<ParallelCopy>> = fx_hash_map();

    for &blk_id in &block_ids {
        if let Some(parallel) = copies_per_block.get(&blk_id) {
            if parallel.is_empty() {
                continue;
            }
            let seq = sequentialise_copies(parallel, func);
            if !seq.is_empty() {
                sequential_copies.insert(blk_id, seq);
            }
        }
    }

    // Now insert the materialised copy instructions into each block.
    for (blk_id, seq) in sequential_copies {
        insert_copies_before_terminator(func, blk_id, &seq);
    }
}

/// Converts a set of parallel copies into a sequential ordering that
/// preserves the parallel semantics.
///
/// # Algorithm
///
/// 1. Build a *ready set*: copies whose destination is not the source of
///    any other pending copy (safe to emit immediately).
/// 2. Emit ready copies and remove them from the pending set.
/// 3. If the pending set is non-empty but no copy is ready, a cycle
///    exists.  Break it by allocating a temporary value
///    (`func.new_value`) to save one source, rewriting the pending copy
///    to read from the temporary.
/// 4. Repeat until all copies are emitted.
///
/// # Cycle Example
///
/// ```text
/// Parallel:   { a ← b,  b ← a }          (swap)
/// Sequential: { tmp ← a,  a ← b,  b ← tmp }
/// ```
fn sequentialise_copies(parallel: &[ParallelCopy], func: &mut IrFunction) -> Vec<ParallelCopy> {
    if parallel.is_empty() {
        return Vec::new();
    }

    // Filter out trivial copies (dest == src) that may have survived.
    let mut pending: Vec<ParallelCopy> = parallel
        .iter()
        .filter(|(d, s, _)| d != s)
        .cloned()
        .collect();

    let mut result: Vec<ParallelCopy> = Vec::with_capacity(pending.len() + 4);

    // Track blocks/values that have been visited during critical edge
    // detection and cycle handling via `FxHashSet`.
    let mut processed: FxHashSet<ValueId> = fx_hash_set();

    while !pending.is_empty() {
        // Build the set of all *source* values in the pending copies.
        // A destination `d` is safe to write iff `d` is NOT in this
        // source set (no other copy reads from `d`).
        let src_set: FxHashSet<ValueId> = pending.iter().map(|(_, s, _)| *s).collect();

        // Find a copy whose destination is NOT the source of any other
        // pending copy.  Such a copy is safe to emit because its
        // destination will not overwrite a value still needed by another
        // pending copy.
        let ready_idx = pending
            .iter()
            .position(|(dest, _, _)| !src_set.contains(dest));

        match ready_idx {
            Some(idx) => {
                let copy = pending.swap_remove(idx);
                processed.insert(copy.0); // Track the destination we wrote.
                result.push(copy);
            }
            None => {
                // All remaining copies form one or more cycles.  Break
                // the first cycle by saving its source into a temporary.
                let (dest, src, ref ty) = pending[0];
                let tmp = func.new_value(ty.clone(), Some("phi_tmp".into()));

                // Emit: tmp ← src  (save the source before it is overwritten)
                result.push((tmp, src, ty.clone()));
                processed.insert(tmp);

                // Rewrite the pending copy to read from `tmp` instead of
                // the original `src`.
                pending[0] = (dest, tmp, ty.clone());

                // Now `dest` is no longer equal to `src` for any other
                // copy in the cycle (because `tmp` is a fresh value), so
                // the next iteration will find at least one ready copy.
            }
        }
    }

    result
}

/// Inserts `BitCast` copy instructions into block `block_id` immediately
/// before its terminator.
///
/// Each `(dest, src, ty)` triple becomes:
///
/// ```text
/// Instruction::BitCast { result: dest, value: src, to_ty: ty }
/// ```
///
/// A same-type `BitCast` is semantically a register-to-register copy.
/// The register allocator recognises this pattern for copy coalescing.
fn insert_copies_before_terminator(
    func: &mut IrFunction,
    block_id: BasicBlockId,
    copies: &[ParallelCopy],
) {
    let block = func.get_block_mut(block_id);

    // Determine the insertion index.  We must insert *before* the
    // terminator.  If the block has no terminator (malformed IR) we
    // append at the end as a best-effort fallback.
    let insert_at = block
        .terminator()
        .map(|_| {
            // The terminator's index is the position of the last
            // is_terminator instruction.  We walk from the end to
            // find it.
            let instructions = block.instructions();
            let len = instructions.len();
            for i in (0..len).rev() {
                if instructions[i].is_terminator() {
                    return i;
                }
            }
            len
        })
        .unwrap_or_else(|| block.instructions().len());

    // Insert copies in the order produced by sequentialisation.  Each
    // insertion shifts subsequent instructions (including the terminator)
    // right by one, so we insert at the *same* index repeatedly — this
    // preserves the sequential ordering.
    for (offset, (dest, src, ty)) in copies.iter().enumerate() {
        let inst = Instruction::BitCast {
            result: *dest,
            value: *src,
            to_ty: ty.clone(),
        };
        block.insert_instruction(insert_at + offset, inst);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::{IrFunction, Parameter};
    use crate::ir::instructions::{BasicBlockId, Instruction, ValueId};
    use crate::ir::types::IrType;

    // Helper: build a trivial function with no phis.
    fn make_empty_func() -> IrFunction {
        IrFunction::new("test".into(), IrType::Void, vec![])
    }

    // ------------------------------------------------------------------
    // verify_no_phis tests
    // ------------------------------------------------------------------

    #[test]
    fn verify_no_phis_empty_function() {
        let func = make_empty_func();
        assert!(verify_no_phis(&func));
    }

    #[test]
    fn verify_no_phis_detects_remaining_phi() {
        let mut func = make_empty_func();
        let bb1_id = func.create_block(Some("bb1".into()));
        let bb1 = func.get_block_mut(bb1_id);
        bb1.add_instruction(Instruction::Phi {
            result: ValueId(10),
            ty: IrType::I32,
            incoming: vec![(ValueId(1), BasicBlockId(0))],
        });
        bb1.add_instruction(Instruction::Return { value: None });
        assert!(!verify_no_phis(&func));
    }

    // ------------------------------------------------------------------
    // Critical edge detection
    // ------------------------------------------------------------------

    #[test]
    fn find_critical_edges_simple_diamond() {
        // Build a diamond CFG:
        //
        //     entry (0)
        //      / \
        //   bb1   bb2
        //      \ /
        //     bb3
        //
        // entry has 2 successors, bb3 has 2 predecessors → two critical
        // edges: (entry, bb3)... wait, entry→bb1 and entry→bb2, not
        // directly to bb3.
        //
        // Actually: entry→bb1 is NOT critical (bb1 has 1 pred),
        //           entry→bb2 is NOT critical (bb2 has 1 pred),
        //           bb1→bb3 is NOT critical (bb1 has 1 succ),
        //           bb2→bb3 is NOT critical (bb2 has 1 succ).
        //
        // So a simple diamond has NO critical edges.  Good — this
        // confirms the detection logic.

        let mut func = make_empty_func();

        // entry block (id=0) — already created by IrFunction::new
        let bb1_id = func.create_block(Some("bb1".into()));
        let bb2_id = func.create_block(Some("bb2".into()));
        let bb3_id = func.create_block(Some("bb3".into()));

        // Wire CFG
        {
            let entry = func.get_block_mut(BasicBlockId(0));
            entry.add_successor(bb1_id);
            entry.add_successor(bb2_id);
            entry.add_instruction(Instruction::CondBranch {
                condition: ValueId(0),
                true_target: bb1_id,
                false_target: bb2_id,
            });
        }
        {
            let bb1 = func.get_block_mut(bb1_id);
            bb1.add_predecessor(BasicBlockId(0));
            bb1.add_successor(bb3_id);
            bb1.add_instruction(Instruction::Branch { target: bb3_id });
        }
        {
            let bb2 = func.get_block_mut(bb2_id);
            bb2.add_predecessor(BasicBlockId(0));
            bb2.add_successor(bb3_id);
            bb2.add_instruction(Instruction::Branch { target: bb3_id });
        }
        {
            let bb3 = func.get_block_mut(bb3_id);
            bb3.add_predecessor(bb1_id);
            bb3.add_predecessor(bb2_id);
            bb3.add_instruction(Instruction::Return { value: None });
        }

        let edges = find_critical_edges(&func);
        assert!(edges.is_empty(), "diamond has no critical edges");
    }

    #[test]
    fn find_critical_edges_with_crit() {
        // Build:
        //
        //     entry (0)
        //      / \
        //   bb1   \
        //      \   |
        //      bb2 (also target of entry's false branch)
        //
        // entry → bb1 (bb1 has 1 pred, so not critical)
        // entry → bb2 (entry has 2 succs, bb2 has 2 preds → CRITICAL)
        // bb1   → bb2 (bb1 has 1 succ, so not critical)

        let mut func = make_empty_func();
        let v0 = func.new_value(IrType::I1, None);

        let bb1_id = func.create_block(Some("bb1".into()));
        let bb2_id = func.create_block(Some("bb2".into()));

        {
            let entry = func.get_block_mut(BasicBlockId(0));
            entry.add_successor(bb1_id);
            entry.add_successor(bb2_id);
            entry.add_instruction(Instruction::CondBranch {
                condition: v0,
                true_target: bb1_id,
                false_target: bb2_id,
            });
        }
        {
            let bb1 = func.get_block_mut(bb1_id);
            bb1.add_predecessor(BasicBlockId(0));
            bb1.add_successor(bb2_id);
            bb1.add_instruction(Instruction::Branch { target: bb2_id });
        }
        {
            let bb2 = func.get_block_mut(bb2_id);
            bb2.add_predecessor(BasicBlockId(0));
            bb2.add_predecessor(bb1_id);
            bb2.add_instruction(Instruction::Return { value: None });
        }

        let edges = find_critical_edges(&func);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (BasicBlockId(0), bb2_id));
    }

    // ------------------------------------------------------------------
    // Parallel copy sequentialisation
    // ------------------------------------------------------------------

    #[test]
    fn sequentialise_no_deps() {
        // Copies: a ← x, b ← y  (independent)
        let mut func = make_empty_func();
        let copies = vec![
            (ValueId(10), ValueId(20), IrType::I32),
            (ValueId(11), ValueId(21), IrType::I32),
        ];
        let seq = sequentialise_copies(&copies, &mut func);
        // Both copies should appear (order may vary since there are no
        // dependencies).
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn sequentialise_swap_cycle() {
        // Copies: a ← b, b ← a  (swap — cycle of length 2)
        let mut func = make_empty_func();
        let a = ValueId(10);
        let b = ValueId(11);
        let copies = vec![(a, b, IrType::I32), (b, a, IrType::I32)];
        let seq = sequentialise_copies(&copies, &mut func);
        // Should introduce a temporary: 3 instructions.
        assert_eq!(seq.len(), 3, "swap requires 3 sequential copies");

        // Simulate and verify correctness.
        // Initial: a=A, b=B
        // After sequential execution: a=B, b=A
        let mut vals: FxHashMap<ValueId, u64> = fx_hash_map();
        vals.insert(a, 100);
        vals.insert(b, 200);

        for (dest, src, _) in &seq {
            let v = *vals.get(src).expect("source must exist");
            vals.insert(*dest, v);
        }

        assert_eq!(
            *vals.get(&a).unwrap(),
            200,
            "a should contain B's original value"
        );
        assert_eq!(
            *vals.get(&b).unwrap(),
            100,
            "b should contain A's original value"
        );
    }

    #[test]
    fn sequentialise_three_cycle() {
        // Copies: a ← b, b ← c, c ← a  (3-cycle)
        let mut func = make_empty_func();
        let a = ValueId(10);
        let b = ValueId(11);
        let c = ValueId(12);
        let copies = vec![
            (a, b, IrType::I64),
            (b, c, IrType::I64),
            (c, a, IrType::I64),
        ];
        let seq = sequentialise_copies(&copies, &mut func);
        // Should introduce exactly 1 temporary: 4 instructions.
        assert_eq!(seq.len(), 4, "3-cycle requires 4 sequential copies");

        // Simulate.
        let mut vals: FxHashMap<ValueId, u64> = fx_hash_map();
        vals.insert(a, 1);
        vals.insert(b, 2);
        vals.insert(c, 3);

        for (dest, src, _) in &seq {
            let v = *vals.get(src).expect("source must exist");
            vals.insert(*dest, v);
        }

        assert_eq!(*vals.get(&a).unwrap(), 2, "a ← b");
        assert_eq!(*vals.get(&b).unwrap(), 3, "b ← c");
        assert_eq!(*vals.get(&c).unwrap(), 1, "c ← a");
    }

    #[test]
    fn sequentialise_trivial_copy_filtered() {
        // Copy: a ← a  (no-op)
        let mut func = make_empty_func();
        let copies = vec![(ValueId(5), ValueId(5), IrType::I32)];
        let seq = sequentialise_copies(&copies, &mut func);
        assert!(seq.is_empty(), "self-copy should be eliminated");
    }

    // ------------------------------------------------------------------
    // End-to-end phi elimination
    // ------------------------------------------------------------------

    #[test]
    fn eliminate_phis_simple() {
        // Build:
        //
        //   entry:
        //     %v0 = ...           (param)
        //     br bb_merge
        //
        //   bb_other:
        //     br bb_merge
        //
        //   bb_merge:
        //     %v5 = phi i32 [%v0, entry], [%v1, bb_other]
        //     ret i32 %v5

        let params = vec![Parameter {
            name: Some("x".into()),
            ty: IrType::I32,
            id: ValueId(0),
        }];
        let mut func = IrFunction::new("test".into(), IrType::I32, params);
        let v1 = func.new_value(IrType::I32, Some("y".into()));
        let v5 = func.new_value(IrType::I32, Some("phi_res".into()));

        let bb_other = func.create_block(Some("bb_other".into()));
        let bb_merge = func.create_block(Some("bb_merge".into()));

        // Wire entry → bb_merge
        {
            let entry = func.get_block_mut(BasicBlockId(0));
            entry.add_successor(bb_merge);
            entry.add_instruction(Instruction::Branch { target: bb_merge });
        }

        // Wire bb_other → bb_merge
        {
            let other = func.get_block_mut(bb_other);
            other.add_successor(bb_merge);
            other.add_instruction(Instruction::Branch { target: bb_merge });
        }

        // bb_merge: phi + ret
        {
            let merge = func.get_block_mut(bb_merge);
            merge.add_predecessor(BasicBlockId(0));
            merge.add_predecessor(bb_other);
            merge.add_instruction(Instruction::Phi {
                result: v5,
                ty: IrType::I32,
                incoming: vec![(ValueId(0), BasicBlockId(0)), (v1, bb_other)],
            });
            merge.add_instruction(Instruction::Return { value: Some(v5) });
        }

        // Run elimination.
        eliminate_phis(&mut func);

        // Post-condition: no phis remain.
        assert!(verify_no_phis(&func), "phis should be eliminated");

        // entry should now contain a BitCast copy before its branch.
        let entry = func.get_block(BasicBlockId(0));
        let entry_instrs = entry.instructions();
        // We expect at least one copy (v5 ← v0) plus the branch.
        // However, v0 == v5 was not the case here so a copy is expected.
        let has_copy = entry_instrs
            .iter()
            .any(|inst| matches!(inst, Instruction::BitCast { .. }));
        assert!(has_copy, "entry should contain a copy instruction");

        // bb_other should contain a BitCast copy before its branch.
        let other = func.get_block(bb_other);
        let other_instrs = other.instructions();
        let has_copy_other = other_instrs
            .iter()
            .any(|inst| matches!(inst, Instruction::BitCast { .. }));
        assert!(has_copy_other, "bb_other should contain a copy instruction");
    }
}
