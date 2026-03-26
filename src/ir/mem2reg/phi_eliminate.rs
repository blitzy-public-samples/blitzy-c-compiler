//! Phase 9: Phi-Node Elimination
//!
//! Converts phi nodes to copy operations at predecessor block ends, then
//! sequentializes the parallel copies for register allocation consumption.
//!
//! ## Algorithm
//!
//! For each phi node `%result = phi [%v1, %pred1], [%v2, %pred2], ...`:
//! 1. **Parallel copy insertion**: At the end of each predecessor block,
//!    insert a copy: `result ← incoming_value` (logically parallel)
//! 2. **Sequentialization**: Convert parallel copies to sequential copies
//!    that respect data dependencies (break cycles with temporaries)
//! 3. **Phi removal**: Remove the original phi instructions
//!
//! ## Timing
//!
//! This pass runs AFTER optimization passes (Phase 8) and BEFORE code
//! generation (Phase 10). It produces an IR form that the register
//! allocator can directly consume — no phi nodes remain.
//!
//! ## Critical Edges
//!
//! If a predecessor block has multiple successors AND the phi's block has
//! multiple predecessors, the edge is "critical." Copies cannot be placed
//! at the end of the predecessor without affecting the other successor.
//! In this case, the critical edge must be split by inserting a new block.
//!
//! ## Copy Representation
//!
//! Copies are represented as `Instruction::BitCast` with the same source
//! and target type (identity cast). The backend recognizes these as
//! register-to-register moves and emits appropriate machine instructions.
//!
//! ## Cycle Breaking
//!
//! When parallel copies form a dependency cycle (e.g., `A←B, B←A`), the
//! sequentializer breaks the cycle by introducing a temporary value:
//! `temp←B; A←B's_src_via_temp; B←temp`. This ensures that all values
//! are read before any destination is overwritten.

use crate::common::diagnostics::Span;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

// ===========================================================================
// PhiCopy — copy operation generated during phi elimination
// ===========================================================================

/// A copy operation generated during phi elimination.
///
/// Represents `dest ← src` where both are SSA values. These copies are
/// first generated as *parallel* copies (all sources read before any
/// destination is written), then sequentialized into an executable order
/// that respects data dependencies.
///
/// The `ty` field carries the IR type of the copied value so that the
/// backend can emit type-correct copy (move) instructions. The `span`
/// field preserves the source location of the original phi instruction
/// for debug information and diagnostic messages.
#[derive(Debug, Clone)]
pub struct PhiCopy {
    /// Destination value (the phi result that receives the copied data).
    pub dest: Value,
    /// Source value (the incoming operand from a specific predecessor).
    pub src: Value,
    /// IR type of the copied value — propagated from the phi instruction.
    pub ty: IrType,
    /// Source location of the original phi instruction.
    pub span: Span,
}

// ===========================================================================
// PhiInfo — internal representation of a collected phi instruction
// ===========================================================================

/// Information about a single phi instruction, extracted during collection.
///
/// Groups the phi's result, type, source span, and all incoming
/// `(value, predecessor_block_index)` pairs for downstream processing
/// by the parallel copy generator.
#[derive(Debug)]
struct PhiInfo {
    /// SSA result value produced by this phi instruction.
    result: Value,
    /// IR type of the phi result and all incoming values.
    ty: IrType,
    /// Incoming `(value, predecessor_block_index)` pairs.
    incoming: Vec<(Value, usize)>,
    /// Source location span from the original phi instruction.
    span: Span,
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// Phase 9: Eliminate all phi nodes from a function.
///
/// Converts phi nodes to copy operations placed at predecessor block ends,
/// sequentializes parallel copies to respect data dependencies (breaking
/// cycles with temporary values), and removes the original phi instructions.
///
/// After this pass, the function contains **no** phi instructions. Copy
/// operations are represented as `Instruction::BitCast` with identical
/// source and target types (identity casts), which the backend translates
/// to register-to-register moves.
///
/// # Algorithm
///
/// 1. Eliminate trivial phis (all incoming values identical) as an optimization.
/// 2. Collect remaining phi nodes from all blocks.
/// 3. Split critical edges to ensure safe copy placement.
/// 4. Re-collect phi nodes (incoming pairs updated by edge splitting).
/// 5. Generate parallel copy sets per predecessor block.
/// 6. Sequentialize parallel copies (topological order + cycle breaking).
/// 7. Insert sequentialized copies before each predecessor's terminator.
/// 8. Remove all phi instructions from all blocks.
///
/// # Panics
///
/// Does not panic — gracefully handles functions with zero blocks or
/// zero phi nodes by returning immediately.
pub fn eliminate_phi_nodes(func: &mut IrFunction) {
    // Early exit for empty functions or declaration-only functions.
    if func.block_count() == 0 {
        return;
    }

    // Step 1: Eliminate trivial phis (optimization — reduces unnecessary copies).
    // Loop until no more trivial phis can be found, because replacing one
    // trivial phi may expose another.
    while eliminate_trivial_phis(func) {}

    // Step 2: Collect all remaining phi nodes organized by block.
    let phi_nodes = collect_phi_nodes(func);
    if phi_nodes.is_empty() {
        return;
    }

    // Step 3: Identify and split all critical edges that affect phi blocks.
    // A critical edge is one where the predecessor has >1 successor AND the
    // phi's block has >1 predecessor. Splitting inserts an intermediate block
    // so copies can be safely placed without affecting other control-flow paths.
    //
    // EXCEPTION: Do NOT split critical edges where the predecessor ends with
    // IndirectBranch (computed goto). IndirectBranch jumps to a runtime-
    // computed address (from BlockAddress instructions), and splitting would
    // create trampoline blocks that are unreachable because the runtime target
    // still points to the original label. Instead, copies for IndirectBranch
    // predecessors are inserted inline before the terminator. This is safe
    // because each phi has a unique result value — copies for different target
    // blocks don't interfere with each other.
    let mut edges_to_split: Vec<(usize, usize)> = Vec::new();
    for (&block_idx, phis) in phi_nodes.iter() {
        for phi in phis {
            for &(_, pred_idx) in &phi.incoming {
                if is_critical_edge(func, pred_idx, block_idx)
                    && !edges_to_split.contains(&(pred_idx, block_idx))
                {
                    // Skip splitting if predecessor contains an IndirectBranch.
                    // We check ALL instructions (not just the last terminator)
                    // because a block may have an IndirectBranch followed by
                    // dead code (e.g., a loop back-edge branch emitted after
                    // a computed goto). The dead branch is the last instruction,
                    // so `terminator()` would miss the IndirectBranch.
                    let pred_is_indirect = func.get_block(pred_idx).map_or(false, |b| {
                        b.instructions
                            .iter()
                            .any(|inst| matches!(inst, Instruction::IndirectBranch { .. }))
                    });
                    if !pred_is_indirect {
                        edges_to_split.push((pred_idx, block_idx));
                    }
                }
            }
        }
    }
    for &(pred, succ) in &edges_to_split {
        split_critical_edge(func, pred, succ);
    }

    // Step 4: Re-collect phi nodes after edge splitting (incoming BlockId
    // references were updated in-place by split_critical_edge).
    let phi_nodes = collect_phi_nodes(func);
    if phi_nodes.is_empty() {
        return;
    }

    // Step 5: Generate parallel copy sets — one set per predecessor block.
    let parallel_copies = generate_parallel_copies(&phi_nodes);

    // Step 6: Sequentialize each parallel copy set into an executable order,
    // breaking dependency cycles with fresh temporary values.
    let mut next_value = func.value_count;
    let mut sequentialized: FxHashMap<usize, Vec<PhiCopy>> = FxHashMap::default();
    for (&pred_idx, copies) in parallel_copies.iter() {
        let seq = sequentialize_copies(copies, &mut next_value);
        sequentialized.insert(pred_idx, seq);
    }
    func.value_count = next_value;

    // Step 7: Insert sequentialized copies into predecessor blocks, just
    // before each block's terminator instruction.
    for (&pred_idx, copies) in sequentialized.iter() {
        if let Some(block) = func.get_block_mut(pred_idx) {
            insert_copies_before_terminator(block, copies);
        }
    }

    // Step 8: Remove all phi instructions from all blocks.
    remove_all_phis(func);
}

/// Run phi elimination on all functions in the module.
///
/// This is the Phase 9 entry point called by the compilation pipeline
/// after optimization passes (Phase 8) and before code generation
/// (Phase 10). Iterates all function definitions and eliminates their
/// phi nodes in place.
///
/// Function declarations (without bodies) are skipped automatically
/// since they have no basic blocks.
pub fn run_phi_elimination(module: &mut IrModule) {
    for func in module.functions.iter_mut() {
        // Only process function definitions (functions with bodies).
        // Declarations have no blocks and are skipped by the early exit
        // in eliminate_phi_nodes, but checking is_definition is explicit.
        if func.is_definition {
            eliminate_phi_nodes(func);
        }
    }
}

/// Remove trivial phi nodes — phi nodes where all incoming values
/// (ignoring self-references) are identical.
///
/// A trivial phi like `%3 = phi [%1, %bb0], [%1, %bb1]` can be replaced
/// directly with `%1` without generating any copy instructions. Self-
/// referential incoming values (`%3` appearing in its own incoming list)
/// are ignored because they represent "use the value from the same
/// iteration," which is always the same as the non-self value.
///
/// # Returns
///
/// `true` if at least one trivial phi was eliminated. The caller should
/// re-invoke this function in a loop until it returns `false`, because
/// eliminating one trivial phi may expose others.
///
/// # Examples
///
/// - `%3 = phi [%1, %bb0], [%1, %bb1]` → replace all uses of `%3` with `%1`
/// - `%3 = phi [%3, %bb0], [%1, %bb1]` → replace all uses of `%3` with `%1` (self-ref ignored)
/// - `%3 = phi [%3, %bb0], [%3, %bb1]` → replace all uses of `%3` with `Value::UNDEF`
pub fn eliminate_trivial_phis(func: &mut IrFunction) -> bool {
    // Phase 1: Scan all phi instructions and identify trivial ones.
    let mut trivial: Vec<(Value, Value)> = Vec::new(); // (phi_result, replacement_value)

    for block in func.blocks() {
        // Scan ALL instructions in the block — not just the phi prefix —
        // because IR lowering may place non-phi instructions (e.g. casts
        // for ternary operand type unification) before phi nodes in merge
        // blocks.  Using `phi_instructions()` (which uses `take_while`)
        // would miss phis that follow a non-phi instruction.
        for inst in block.instructions.iter() {
            if let Instruction::Phi {
                result, incoming, ..
            } = inst
            {
                let mut unique_value: Option<Value> = None;
                let mut is_trivial = true;

                for (val, _) in incoming {
                    // Skip self-references — they don't contribute a unique value.
                    if *val == *result {
                        continue;
                    }
                    match unique_value {
                        None => {
                            unique_value = Some(*val);
                        }
                        Some(uv) => {
                            if uv != *val {
                                is_trivial = false;
                                break;
                            }
                        }
                    }
                }

                if is_trivial {
                    // If all incoming values were self-references (no unique value found),
                    // the phi is dead — replace with UNDEF.
                    let replacement = unique_value.unwrap_or(Value::UNDEF);
                    trivial.push((*result, replacement));
                }
            }
        }
    }

    if trivial.is_empty() {
        return false;
    }

    // Phase 1.5: Resolve trivial-phi chains transitively.
    //
    // When trivial phis form a chain (e.g. A→B and B→C), applying
    // replacements in arbitrary order can leave dangling references.
    // For example, if B→C is applied first and then A→B, all uses of A
    // become B — but B's phi is removed in Phase 3, leaving B undefined.
    //
    // To fix this, we follow the replacement map transitively: if A→B
    // and B→C, we resolve A directly to C before applying any
    // replacements.  This guarantees that regardless of application
    // order, every value maps to its final non-trivial-phi target.
    {
        let trivial_map: FxHashMap<Value, Value> = trivial.iter().copied().collect();
        for pair in trivial.iter_mut() {
            let mut target = pair.1;
            // Follow the chain: target → next → next → ...
            // Guard against cycles (shouldn't happen, but be defensive).
            let mut steps = 0u32;
            while let Some(&next) = trivial_map.get(&target) {
                if next == target || steps > 1000 {
                    break;
                }
                target = next;
                steps += 1;
            }
            pair.1 = target;
        }
    }

    // Phase 2: Replace all uses of each trivial phi's result with its
    // (now transitively resolved) replacement value throughout the
    // entire function.
    for &(old_val, new_val) in &trivial {
        replace_value_uses(func, old_val, new_val);
    }

    // Phase 3: Remove the trivial phi instructions from their blocks.
    let trivial_results: FxHashSet<u32> = trivial.iter().map(|(v, _)| v.0).collect();
    for block in func.blocks_mut() {
        block.instructions_mut().retain(|inst| {
            if let Instruction::Phi { result, .. } = inst {
                !trivial_results.contains(&result.0)
            } else {
                true
            }
        });
    }

    true
}

// ===========================================================================
// Phi collection
// ===========================================================================

/// Collect all phi instructions from a function, organized by block index.
///
/// Returns a map from block index to a vector of [`PhiInfo`] structs,
/// each capturing the phi's result, type, incoming pairs (with
/// predecessor block indices converted from `BlockId`), and source span.
///
/// Blocks with no phi instructions are not present in the returned map.
fn collect_phi_nodes(func: &IrFunction) -> FxHashMap<usize, Vec<PhiInfo>> {
    let mut result: FxHashMap<usize, Vec<PhiInfo>> = FxHashMap::default();

    for (block_idx, block) in func.blocks().iter().enumerate() {
        let mut phis_in_block: Vec<PhiInfo> = Vec::new();

        // Scan ALL instructions — not just the phi prefix — because IR
        // lowering may place non-phi instructions (e.g. type casts for
        // ternary operand unification) before phi nodes in merge blocks.
        for inst in block.instructions.iter() {
            if let Instruction::Phi {
                result: phi_result,
                ty,
                incoming,
                span,
            } = inst
            {
                let incoming_pairs: Vec<(Value, usize)> = incoming
                    .iter()
                    .map(|(val, blk_id)| (*val, blk_id.index()))
                    .collect();

                phis_in_block.push(PhiInfo {
                    result: *phi_result,
                    ty: ty.clone(),
                    incoming: incoming_pairs,
                    span: *span,
                });
            }
        }

        if !phis_in_block.is_empty() {
            result.insert(block_idx, phis_in_block);
        }
    }

    result
}

// ===========================================================================
// Critical edge detection and splitting
// ===========================================================================

/// Detect if the edge from `pred` to `succ` is a critical edge.
///
/// An edge is critical if the predecessor block has more than one successor
/// AND the successor block has more than one predecessor. Placing copy
/// instructions at the end of a block with multiple successors would
/// incorrectly execute them for all outgoing edges, so critical edges
/// must be split before copy insertion.
fn is_critical_edge(func: &IrFunction, pred: usize, succ: usize) -> bool {
    let pred_block = match func.get_block(pred) {
        Some(b) => b,
        None => return false,
    };
    let succ_block = match func.get_block(succ) {
        Some(b) => b,
        None => return false,
    };
    pred_block.successor_count() > 1 && succ_block.predecessor_count() > 1
}

/// Split a critical edge by inserting a new basic block between `pred` and `succ`.
///
/// The new block contains only an unconditional branch to `succ`. The
/// predecessor's terminator is updated to branch to the new block instead,
/// and all CFG edges (predecessor/successor lists) and phi instructions in
/// `succ` are updated accordingly.
///
/// # Returns
///
/// The index of the newly created intermediate block.
fn split_critical_edge(func: &mut IrFunction, pred: usize, succ: usize) -> usize {
    let new_block_idx = func.block_count();

    // Create the intermediate block with an unconditional branch to succ.
    let mut new_block = BasicBlock::new(new_block_idx);
    new_block.add_predecessor(pred);
    new_block.add_successor(succ);
    new_block.push_instruction(Instruction::Branch {
        target: BlockId(succ as u32),
        span: Span::dummy(),
    });
    func.add_block(new_block);

    // Update pred's terminator: retarget all references to succ → new_block_idx.
    retarget_terminator(&mut func.blocks[pred], succ, new_block_idx);
    func.blocks[pred].remove_successor(succ);
    func.blocks[pred].add_successor(new_block_idx);

    // Update succ's predecessor list: replace pred with new_block_idx.
    func.blocks[succ].remove_predecessor(pred);
    func.blocks[succ].add_predecessor(new_block_idx);

    // Update phi instructions in succ: replace incoming (_, BlockId(pred))
    // entries with (_, BlockId(new_block_idx)) so that copies placed in the
    // new block are correctly associated with the phi's incoming edge.
    for inst in func.blocks[succ].instructions_mut().iter_mut() {
        if let Instruction::Phi { incoming, .. } = inst {
            for (_, block_id) in incoming.iter_mut() {
                if block_id.index() == pred {
                    *block_id = BlockId(new_block_idx as u32);
                }
            }
        }
    }

    new_block_idx
}

/// Retarget a block's terminator instruction to replace all references
/// to `old_target` with `new_target`.
///
/// Handles `Branch`, `CondBranch`, and `Switch` terminators. For
/// `CondBranch` and `Switch`, both/all target references matching
/// `old_target` are updated (handles the rare case where the same
/// block appears multiple times as a target).
fn retarget_terminator(block: &mut BasicBlock, old_target: usize, new_target: usize) {
    if let Some(term) = block.terminator_mut() {
        match term {
            Instruction::Branch { target, .. } => {
                if target.index() == old_target {
                    *target = BlockId(new_target as u32);
                }
            }
            Instruction::CondBranch {
                then_block,
                else_block,
                ..
            } => {
                if then_block.index() == old_target {
                    *then_block = BlockId(new_target as u32);
                }
                if else_block.index() == old_target {
                    *else_block = BlockId(new_target as u32);
                }
            }
            Instruction::Switch { default, cases, .. } => {
                if default.index() == old_target {
                    *default = BlockId(new_target as u32);
                }
                for (_, target) in cases.iter_mut() {
                    if target.index() == old_target {
                        *target = BlockId(new_target as u32);
                    }
                }
            }
            Instruction::IndirectBranch {
                possible_targets, ..
            } => {
                for target in possible_targets.iter_mut() {
                    if target.index() == old_target {
                        *target = BlockId(new_target as u32);
                    }
                }
            }
            _ => {
                // Return or other non-branching terminators — nothing to retarget.
            }
        }
    }
}

// ===========================================================================
// Parallel copy generation
// ===========================================================================

/// For each predecessor block, collect all copies that need to happen when
/// control flows from that predecessor to a phi-containing block.
///
/// Returns a map from predecessor block index to a vector of [`PhiCopy`]
/// structs. Each copy records `dest ← src` where `dest` is the phi result
/// and `src` is the incoming value from that specific predecessor.
///
/// Identity copies (where `dest == src`) are filtered out since they are
/// no-ops and would waste a copy instruction.
fn generate_parallel_copies(
    phi_nodes: &FxHashMap<usize, Vec<PhiInfo>>,
) -> FxHashMap<usize, Vec<PhiCopy>> {
    let mut copies: FxHashMap<usize, Vec<PhiCopy>> = FxHashMap::default();

    for (_block_idx, phis) in phi_nodes.iter() {
        for phi in phis {
            for &(incoming_value, pred_idx) in &phi.incoming {
                // Skip identity copies — they are no-ops.
                if incoming_value == phi.result {
                    continue;
                }

                copies.entry(pred_idx).or_default().push(PhiCopy {
                    dest: phi.result,
                    src: incoming_value,
                    ty: phi.ty.clone(),
                    span: phi.span,
                });
            }
        }
    }

    copies
}

// ===========================================================================
// Parallel copy sequentialization
// ===========================================================================

/// Sequentialize a set of parallel copies into sequential copies that can
/// be executed in order without violating data dependencies.
///
/// **Parallel semantics**: all sources are read BEFORE any destination is
/// written. **Sequential semantics**: each copy reads its source, then
/// writes its destination, potentially overwriting a value needed by a
/// later copy.
///
/// # Algorithm
///
/// 1. Filter out identity copies (`dest == src`).
/// 2. Repeatedly find a copy whose destination is **not** used as a source
///    by any remaining copy — this copy is safe to execute because it won't
///    clobber a value needed later.
/// 3. If no safe copy exists, all remaining copies form one or more cycles
///    (e.g., `A←B, B←A`). Break a cycle by:
///    - Saving one copy's source into a fresh temporary: `temp ← copy.src`
///    - Replacing that copy's source with the temporary.
///    - The cycle is now broken because `temp` is only a source, never a
///      destination, enabling the algorithm to find safe copies again.
/// 4. Repeat until all copies are emitted.
///
/// # Parameters
///
/// - `copies`: Parallel copy set for a single predecessor block.
/// - `next_value`: Mutable counter for allocating fresh temporary `Value`s.
///   Incremented each time a cycle-breaking temporary is needed.
///
/// # Returns
///
/// The sequentialized copy list in the order they must be executed.
fn sequentialize_copies(copies: &[PhiCopy], next_value: &mut u32) -> Vec<PhiCopy> {
    let mut result: Vec<PhiCopy> = Vec::with_capacity(copies.len() + 2);

    // Work with a mutable clone of the copies, filtering out identity copies.
    let mut remaining: Vec<PhiCopy> = copies.iter().filter(|c| c.dest != c.src).cloned().collect();

    while !remaining.is_empty() {
        // Build the set of values that are still needed as sources by remaining copies.
        let needed_as_src: FxHashSet<u32> = remaining.iter().map(|c| c.src.index()).collect();

        // Find a copy whose destination is NOT needed as a source by any remaining copy.
        // Such a copy is safe to execute because overwriting its destination won't
        // clobber a value that a later copy needs to read.
        let safe_idx = remaining
            .iter()
            .position(|c| !needed_as_src.contains(&c.dest.index()));

        if let Some(idx) = safe_idx {
            // Safe copy found — emit it and remove from the remaining set.
            result.push(remaining.remove(idx));
        } else {
            // All remaining copies form cycles. Break one cycle by introducing
            // a temporary to save a source value before it gets overwritten.
            //
            // Pick the first remaining copy and save its source to a temp:
            //   temp ← copy.src
            // Then replace copy.src with temp:
            //   copy.dest ← temp
            //
            // This makes `temp` a source-only value (never a destination),
            // which breaks the cycle and allows the algorithm to proceed.
            let temp = Value(*next_value);
            *next_value += 1;

            let copy = &remaining[0];
            let save_copy = PhiCopy {
                dest: temp,
                src: copy.src,
                ty: copy.ty.clone(),
                span: copy.span,
            };
            result.push(save_copy);

            // Replace the source of the original copy with the temporary.
            remaining[0].src = temp;
            // The cycle is now broken — the next iteration will find safe copies.
        }
    }

    result
}

// ===========================================================================
// Copy insertion into blocks
// ===========================================================================

/// Insert copy instructions into a predecessor block, just before its
/// terminator instruction.
///
/// Each [`PhiCopy`] is translated to an `Instruction::BitCast` with the
/// same source and target type (identity cast), which the backend treats
/// as a register-to-register move. The copies are inserted in the order
/// determined by the sequentializer, which ensures data dependencies are
/// respected.
///
/// If the block has no terminator (malformed IR), copies are appended
/// at the end as a defensive fallback.
fn insert_copies_before_terminator(block: &mut BasicBlock, copies: &[PhiCopy]) {
    if copies.is_empty() {
        return;
    }

    // Determine the insertion point: right before the FIRST terminator
    // in the block.  A well-formed block has exactly one terminator as
    // its last instruction, but after IR lowering a block may contain
    // dead code after an IndirectBranch (computed goto) — e.g. an
    // unconditional branch emitted by the for-loop lowering that follows
    // the computed goto.  Using `block.terminator()` (which returns the
    // LAST instruction if it is a terminator) would place copies between
    // the IndirectBranch and the dead branch, making them dead code too.
    // Instead, we scan forward to find the first terminator and insert
    // copies before it, ensuring they execute on all outgoing edges.
    let insert_base = block
        .instructions()
        .iter()
        .position(|inst| inst.is_terminator())
        .unwrap_or(block.instructions().len());

    // Insert each copy at incrementing positions to maintain order.
    // After inserting at position P, the next insert at P+1 places the
    // second copy right after the first, pushing the terminator further.
    for (i, copy) in copies.iter().enumerate() {
        let copy_inst = Instruction::BitCast {
            result: copy.dest,
            value: copy.src,
            to_type: copy.ty.clone(),
            source_unsigned: false,
            span: copy.span,
        };
        block.insert_instruction(insert_base + i, copy_inst);
    }
}

// ===========================================================================
// Phi removal
// ===========================================================================

/// Remove all phi instructions from all blocks in the function.
///
/// Called after parallel copies have been inserted at predecessor block
/// ends. Filters out every instruction where `is_phi()` returns `true`,
/// preserving all non-phi instructions in their original order.
fn remove_all_phis(func: &mut IrFunction) {
    for block in func.blocks_mut() {
        block.instructions_mut().retain(|inst| !inst.is_phi());
    }
}

// ===========================================================================
// Value use replacement (helper for trivial phi elimination)
// ===========================================================================

/// Replace all uses of `old_value` with `new_value` throughout the entire
/// function.
///
/// Scans every instruction in every block. For each instruction, iterates
/// its mutable operand references (via `operands_mut()`) and replaces any
/// occurrence of `old_value` with `new_value`. This covers all operand
/// positions including phi incoming values, branch conditions, load/store
/// addresses, arithmetic operands, and call arguments.
fn replace_value_uses(func: &mut IrFunction, old_value: Value, new_value: Value) {
    for block in func.blocks_mut() {
        for inst in block.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if *operand == old_value {
                    *operand = new_value;
                }
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{BlockId, Instruction, Value};
    use crate::ir::types::IrType;

    /// Helper: create a minimal function with the given number of blocks.
    /// Each block has an index set and is appended to the function.
    fn make_func(name: &str, num_extra_blocks: usize) -> IrFunction {
        let mut func = IrFunction::new(name.to_string(), vec![], IrType::Void);
        for _ in 0..num_extra_blocks {
            let bb = BasicBlock::new(0); // index corrected by add_block
            func.add_block(bb);
        }
        func
    }

    #[test]
    fn test_empty_function() {
        let mut func = make_func("empty", 0);
        // Should not panic — gracefully handles a function with only an entry block.
        eliminate_phi_nodes(&mut func);
        assert_eq!(func.block_count(), 1);
    }

    #[test]
    fn test_no_phi_nodes() {
        let mut func = make_func("no_phi", 1);
        // Add a return terminator to entry block.
        func.blocks[0].push_instruction(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        func.blocks[1].push_instruction(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        eliminate_phi_nodes(&mut func);
        // Should complete without error — no phi nodes to eliminate.
        assert_eq!(func.block_count(), 2);
    }

    #[test]
    fn test_trivial_phi_elimination() {
        let mut func = make_func("trivial", 1);
        let v0 = Value(0);
        let v1 = Value(1);

        // Block 0: entry → branch to block 1
        func.blocks[0].push_instruction(Instruction::Branch {
            target: BlockId(1),
            span: Span::dummy(),
        });
        func.blocks[0].add_successor(1);
        func.blocks[1].add_predecessor(0);

        // Block 1: phi [v0, bb0] — trivial since there's only one incoming value
        func.blocks[1].add_phi(Instruction::Phi {
            result: v1,
            ty: IrType::I32,
            incoming: vec![(v0, BlockId(0))],
            span: Span::dummy(),
        });
        func.blocks[1].push_instruction(Instruction::Return {
            value: Some(v1),
            span: Span::dummy(),
        });

        let changed = eliminate_trivial_phis(&mut func);
        assert!(changed);
        // The trivial phi should be removed.
        assert!(func.blocks[1].phi_instructions().next().is_none());
    }

    #[test]
    fn test_trivial_phi_all_same_value() {
        let mut func = make_func("trivial_same", 2);
        let v0 = Value(0);
        let v1 = Value(1);

        // Block 0 → block 2, Block 1 → block 2
        func.blocks[0].push_instruction(Instruction::Branch {
            target: BlockId(2),
            span: Span::dummy(),
        });
        func.blocks[0].add_successor(2);

        func.blocks[1].push_instruction(Instruction::Branch {
            target: BlockId(2),
            span: Span::dummy(),
        });
        func.blocks[1].add_successor(2);

        func.blocks[2].add_predecessor(0);
        func.blocks[2].add_predecessor(1);

        // Block 2: phi [v0, bb0], [v0, bb1] — trivial, all same
        func.blocks[2].add_phi(Instruction::Phi {
            result: v1,
            ty: IrType::I32,
            incoming: vec![(v0, BlockId(0)), (v0, BlockId(1))],
            span: Span::dummy(),
        });
        func.blocks[2].push_instruction(Instruction::Return {
            value: Some(v1),
            span: Span::dummy(),
        });

        let changed = eliminate_trivial_phis(&mut func);
        assert!(changed);
        // Phi should be removed, and v1's use in Return should be replaced with v0.
        assert!(func.blocks[2].phi_instructions().next().is_none());
        if let Some(Instruction::Return { value, .. }) = func.blocks[2].terminator() {
            assert_eq!(*value, Some(v0));
        } else {
            panic!("Expected Return terminator");
        }
    }

    #[test]
    fn test_sequentialize_no_cycle() {
        // A←X, B←Y — no dependencies, order doesn't matter.
        let copies = vec![
            PhiCopy {
                dest: Value(0),
                src: Value(10),
                ty: IrType::I32,
                span: Span::dummy(),
            },
            PhiCopy {
                dest: Value(1),
                src: Value(11),
                ty: IrType::I32,
                span: Span::dummy(),
            },
        ];
        let mut next = 100;
        let seq = sequentialize_copies(&copies, &mut next);
        assert_eq!(seq.len(), 2);
        // No temporaries should be needed.
        assert_eq!(next, 100);
    }

    #[test]
    fn test_sequentialize_two_cycle() {
        // A←B, B←A — classic two-element cycle.
        let copies = vec![
            PhiCopy {
                dest: Value(0),
                src: Value(1),
                ty: IrType::I32,
                span: Span::dummy(),
            },
            PhiCopy {
                dest: Value(1),
                src: Value(0),
                ty: IrType::I32,
                span: Span::dummy(),
            },
        ];
        let mut next = 100;
        let seq = sequentialize_copies(&copies, &mut next);
        // Should produce 3 copies: temp←src, dest←other, dest←temp
        assert_eq!(seq.len(), 3);
        // One temporary should be allocated.
        assert_eq!(next, 101);
    }

    #[test]
    fn test_sequentialize_three_cycle() {
        // A←B, B←C, C←A — three-element cycle.
        let copies = vec![
            PhiCopy {
                dest: Value(0),
                src: Value(1),
                ty: IrType::I32,
                span: Span::dummy(),
            },
            PhiCopy {
                dest: Value(1),
                src: Value(2),
                ty: IrType::I32,
                span: Span::dummy(),
            },
            PhiCopy {
                dest: Value(2),
                src: Value(0),
                ty: IrType::I32,
                span: Span::dummy(),
            },
        ];
        let mut next = 100;
        let seq = sequentialize_copies(&copies, &mut next);
        // 3 original copies + 1 temp save = 4 copies.
        assert_eq!(seq.len(), 4);
        assert_eq!(next, 101);

        // Verify: the first emitted copy should save some source to a temp.
        assert_eq!(seq[0].dest, Value(100)); // temp
    }

    #[test]
    fn test_sequentialize_identity_filtered() {
        // A←A — identity copy, should be filtered out.
        let copies = vec![PhiCopy {
            dest: Value(0),
            src: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        }];
        let mut next = 100;
        let seq = sequentialize_copies(&copies, &mut next);
        assert!(seq.is_empty());
        assert_eq!(next, 100);
    }

    #[test]
    fn test_critical_edge_detection() {
        let mut func = make_func("crit", 2);

        // Block 0 (entry): conditional branch to blocks 1 and 2.
        func.blocks[0].push_instruction(Instruction::CondBranch {
            condition: Value(0),
            then_block: BlockId(1),
            else_block: BlockId(2),
            span: Span::dummy(),
        });
        func.blocks[0].add_successor(1);
        func.blocks[0].add_successor(2);

        // Block 1: branch to block 2.
        func.blocks[1].push_instruction(Instruction::Branch {
            target: BlockId(2),
            span: Span::dummy(),
        });
        func.blocks[1].add_successor(2);
        func.blocks[1].add_predecessor(0);

        // Block 2: predecessors are 0 and 1.
        func.blocks[2].add_predecessor(0);
        func.blocks[2].add_predecessor(1);
        func.blocks[2].push_instruction(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        // Edge 0→2 is critical: block 0 has 2 successors, block 2 has 2 predecessors.
        assert!(is_critical_edge(&func, 0, 2));
        // Edge 1→2 is NOT critical: block 1 has only 1 successor.
        assert!(!is_critical_edge(&func, 1, 2));
        // Edge 0→1 is NOT critical: block 1 has only 1 predecessor.
        assert!(!is_critical_edge(&func, 0, 1));
    }

    #[test]
    fn test_full_phi_elimination() {
        // Build a diamond CFG:
        //   bb0 (entry) → bb1, bb2
        //   bb1 → bb3
        //   bb2 → bb3
        //   bb3: phi [v1, bb1], [v2, bb2]
        let mut func = make_func("diamond", 3);
        let v0 = Value(0); // condition
        let v1 = Value(1); // value from bb1
        let v2 = Value(2); // value from bb2
        let v3 = Value(3); // phi result in bb3
        func.value_count = 4;

        // bb0: cond branch → bb1 / bb2
        func.blocks[0].push_instruction(Instruction::CondBranch {
            condition: v0,
            then_block: BlockId(1),
            else_block: BlockId(2),
            span: Span::dummy(),
        });
        func.blocks[0].add_successor(1);
        func.blocks[0].add_successor(2);

        // bb1: branch → bb3
        func.blocks[1].add_predecessor(0);
        func.blocks[1].push_instruction(Instruction::Branch {
            target: BlockId(3),
            span: Span::dummy(),
        });
        func.blocks[1].add_successor(3);

        // bb2: branch → bb3
        func.blocks[2].add_predecessor(0);
        func.blocks[2].push_instruction(Instruction::Branch {
            target: BlockId(3),
            span: Span::dummy(),
        });
        func.blocks[2].add_successor(3);

        // bb3: phi + return
        func.blocks[3].add_predecessor(1);
        func.blocks[3].add_predecessor(2);
        func.blocks[3].add_phi(Instruction::Phi {
            result: v3,
            ty: IrType::I32,
            incoming: vec![(v1, BlockId(1)), (v2, BlockId(2))],
            span: Span::dummy(),
        });
        func.blocks[3].push_instruction(Instruction::Return {
            value: Some(v3),
            span: Span::dummy(),
        });

        eliminate_phi_nodes(&mut func);

        // After elimination: no phi instructions remain anywhere.
        for block in func.blocks() {
            assert!(
                block.phi_instructions().next().is_none(),
                "Block {} still has phi instructions",
                block.index
            );
        }

        // bb1 should have a copy instruction (BitCast) before its Branch.
        let bb1 = &func.blocks[1];
        assert!(
            bb1.instructions().len() >= 2,
            "bb1 should have at least a copy + branch"
        );
        let copy_in_bb1 = &bb1.instructions()[0];
        assert!(
            matches!(copy_in_bb1, Instruction::BitCast { .. }),
            "Expected BitCast copy in bb1, got {:?}",
            copy_in_bb1
        );

        // bb2 should have a copy instruction (BitCast) before its Branch.
        let bb2 = &func.blocks[2];
        assert!(
            bb2.instructions().len() >= 2,
            "bb2 should have at least a copy + branch"
        );
        let copy_in_bb2 = &bb2.instructions()[0];
        assert!(
            matches!(copy_in_bb2, Instruction::BitCast { .. }),
            "Expected BitCast copy in bb2, got {:?}",
            copy_in_bb2
        );
    }
}
