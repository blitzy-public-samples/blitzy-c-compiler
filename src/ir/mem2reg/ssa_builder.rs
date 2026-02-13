//! SSA renaming pass for the BCC compiler's alloca-then-promote SSA construction.
//!
//! This module implements the **variable renaming phase** of the mem2reg SSA
//! construction algorithm (Phase 7). It operates after phi-node placement
//! (computed by [`crate::ir::mem2reg::dominance_frontier`]) and transforms the
//! IR from containing phi nodes with empty operands into valid SSA form where
//! every use of a promoted variable references its correct reaching definition.
//!
//! # Algorithm Overview
//!
//! The renaming algorithm traverses the dominator tree in **pre-order**,
//! maintaining a per-variable *reaching-definition stack*:
//!
//! 1. **Phi processing:** For each phi node placed in the current block,
//!    push the phi's result [`ValueId`] onto the corresponding alloca slot's
//!    reaching-definition stack.
//!
//! 2. **Instruction walk:** Process instructions in program order:
//!    - **Store to promoted alloca:** Push the stored value onto the slot's
//!      stack (this is a new definition).
//!    - **Load from promoted alloca:** Map the load's result to the current
//!      top of the slot's stack (the reaching definition).
//!    - **Other instructions:** No special handling during the walk.
//!
//! 3. **Successor phi filling:** For each CFG successor, fill in the
//!    incoming `(value, predecessor_block)` pair for every phi node using
//!    the current reaching definition from the stack top.
//!
//! 4. **Recurse into dominator tree children** — definitions pushed in
//!    this block are visible to all dominated blocks.
//!
//! 5. **Restore stack depths** — pop definitions made in this block so
//!    sibling subtrees do not observe them.
//!
//! After the dominator tree walk completes, a global replacement pass
//! rewrites all uses of load results to their reaching definitions, and
//! a cleanup pass removes the now-dead alloca, load, and store instructions.
//!
//! # Undef Handling
//!
//! If a variable is read before any definition on a given code path (e.g.,
//! uninitialized variable), the reaching-definition stack is empty for that
//! slot. The renamer uses a per-slot **undef sentinel** value, correctly
//! modelling C's undefined behavior for uninitialized reads.
//!
//! # Def-Use Chains
//!
//! After renaming, the pass constructs a def-use map:
//! `FxHashMap<ValueId, Vec<(BasicBlockId, usize)>>` — mapping each SSA value
//! to the list of `(block, instruction_index)` pairs that use it. This is
//! consumed by subsequent optimization passes (dead code elimination, constant
//! propagation).
//!
//! # References
//!
//! R. Cytron et al., "Efficiently Computing Static Single Assignment Form
//! and the Control Dependence Graph," *ACM TOPLAS*, 13(4):451–490, 1991.

use crate::common::fx_hash::{fx_hash_map, FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::Instruction;
use crate::ir::mem2reg::dominator_tree::DominatorTree;
use crate::ir::types::IrType;

// ---------------------------------------------------------------------------
// AllocaSlot — metadata for a single promotable alloca
// ---------------------------------------------------------------------------

/// Information about a single promotable alloca variable being renamed
/// during SSA construction.
///
/// Each `AllocaSlot` corresponds to one alloca instruction that the mem2reg
/// analysis determined is eligible for promotion (scalar type, never
/// address-taken, no volatile accesses). The slot index in the
/// [`SsaRenamer::alloca_slots`] vector serves as the canonical identifier
/// for this variable throughout the renaming algorithm.
///
/// # Fields
///
/// | Field             | Purpose                                           |
/// |-------------------|---------------------------------------------------|
/// | `alloca_value_id` | Original alloca instruction's result [`ValueId`]  |
/// | `ty`              | IR type of the allocated variable                 |
/// | `name`            | Optional variable name for debug info / IR dumps  |
/// | `undef_value`     | Sentinel value for reads before any definition    |
#[derive(Clone, Debug)]
pub struct AllocaSlot {
    /// The original alloca instruction's result value ID. Used to identify
    /// load/store instructions that reference this promoted variable: a
    /// `Load { ptr, .. }` or `Store { ptr, .. }` where `ptr == alloca_value_id`
    /// is a load from or store to this promoted alloca.
    pub alloca_value_id: ValueId,

    /// The IR type of the allocated variable (not the pointer type). For
    /// example, if the alloca was `%p = alloca i32`, the type is `IrType::I32`.
    /// This is used when creating phi instructions and undef sentinels.
    pub ty: IrType,

    /// Optional human-readable name derived from the source-level variable
    /// name. Propagated to phi results and undef values for debug output.
    pub name: Option<String>,

    /// The "undef" sentinel [`ValueId`] for this slot. When a variable is
    /// read before any definition on a given execution path (e.g., an
    /// uninitialized local variable), the reaching-definition stack is
    /// empty and this sentinel is used as the reaching definition. This
    /// correctly models C's undefined behavior for uninitialized reads.
    ///
    /// Created by [`rename_variables()`] via [`IrFunction::new_value()`]
    /// during initialization.
    pub undef_value: ValueId,
}

// ---------------------------------------------------------------------------
// SsaRenamer — state for the SSA renaming pass
// ---------------------------------------------------------------------------

/// State container for the SSA variable renaming pass.
///
/// `SsaRenamer` maintains all mutable state needed during the dominator tree
/// walk: per-variable reaching-definition stacks, the mapping from blocks to
/// their placed phi nodes, and the value replacement map built during renaming.
///
/// # Lifetime
///
/// An `SsaRenamer` is created at the start of [`rename_variables()`],
/// populated during the dominator tree walk, and consumed (its accumulated
/// data applied) at the end of the pass. It is not intended to be reused
/// across multiple functions.
///
/// # Stack Scoping
///
/// The `reaching_defs` stacks are scoped to the dominator tree: definitions
/// pushed while processing block B are visible to all blocks dominated by B,
/// and are popped when the algorithm backtracks past B. This is achieved by
/// saving stack depths before processing each block and restoring them
/// afterward.
pub struct SsaRenamer {
    /// Per-alloca reaching-definition stack. Indexed by alloca slot index
    /// (position in [`alloca_slots`]). Each inner `Vec<ValueId>` is a stack
    /// where the top element is the current reaching definition for that
    /// variable. Empty stacks indicate the variable has no definition on the
    /// current code path (→ use the undef sentinel).
    pub reaching_defs: Vec<Vec<ValueId>>,

    /// Mapping from block ID to the phi nodes placed in that block during
    /// SSA construction. Each entry is `(alloca_slot_index, phi_result_ValueId)`.
    /// Used during the dominator tree walk to:
    /// 1. Push phi results onto reaching-definition stacks.
    /// 2. Locate phi instructions in successor blocks for operand filling.
    pub phi_map: FxHashMap<BasicBlockId, Vec<(usize, ValueId)>>,

    /// Information about each promotable alloca being renamed. The index
    /// in this vector is the canonical "slot index" used throughout the
    /// renaming algorithm.
    pub alloca_slots: Vec<AllocaSlot>,

    /// Map from alloca result [`ValueId`] to its slot index in `alloca_slots`.
    /// Used for O(1) lookup when processing Load/Store instructions to
    /// determine if they reference a promoted alloca.
    alloca_ptr_to_slot: FxHashMap<ValueId, usize>,

    /// Accumulated value replacements: maps each load result that loaded from
    /// a promoted alloca to the SSA value that should replace it (the reaching
    /// definition at the load's program point).
    value_map: FxHashMap<ValueId, ValueId>,
}

impl SsaRenamer {
    /// Creates a new `SsaRenamer` for the given promotable alloca slots.
    ///
    /// Initializes empty reaching-definition stacks (one per slot), builds
    /// the alloca-pointer-to-slot lookup map, and prepares empty phi and
    /// value maps. The undef sentinel values in each `AllocaSlot` must
    /// already be allocated before calling this constructor.
    ///
    /// # Arguments
    ///
    /// * `alloca_slots` — metadata for each promotable alloca, with undef
    ///   sentinels already created via [`IrFunction::new_value()`].
    pub fn new(alloca_slots: Vec<AllocaSlot>) -> Self {
        let num_slots = alloca_slots.len();
        let mut alloca_ptr_to_slot: FxHashMap<ValueId, usize> = fx_hash_map();
        for (idx, slot) in alloca_slots.iter().enumerate() {
            alloca_ptr_to_slot.insert(slot.alloca_value_id, idx);
        }

        SsaRenamer {
            reaching_defs: vec![Vec::new(); num_slots],
            phi_map: fx_hash_map(),
            alloca_slots,
            alloca_ptr_to_slot,
            value_map: fx_hash_map(),
        }
    }

    /// Returns the current reaching definition for the given alloca slot.
    ///
    /// If the reaching-definition stack for this slot is non-empty, returns
    /// the top element (the most recent definition). If the stack is empty
    /// (variable read before any definition on this path), returns the
    /// slot's undef sentinel value.
    #[inline]
    fn current_def(&self, slot: usize) -> ValueId {
        if let Some(&top) = self.reaching_defs[slot].last() {
            top
        } else {
            self.alloca_slots[slot].undef_value
        }
    }

    /// Returns `true` if the given [`ValueId`] is the result of a promoted
    /// alloca instruction (i.e., it's a pointer that loads/stores target).
    #[inline]
    fn is_promoted_alloca(&self, ptr: ValueId) -> bool {
        self.alloca_ptr_to_slot.contains_key(&ptr)
    }

    /// Returns the alloca slot index for a promoted alloca pointer, or
    /// `None` if the pointer is not a promoted alloca.
    #[inline]
    fn get_slot(&self, ptr: ValueId) -> Option<usize> {
        self.alloca_ptr_to_slot.get(&ptr).copied()
    }
}

// ---------------------------------------------------------------------------
// Instruction action classification (internal)
// ---------------------------------------------------------------------------

/// Classification of an instruction's role during SSA renaming.
///
/// Used to pre-compute what action the renamer should take for each
/// instruction in a block, so that data can be collected from an immutable
/// borrow of the function and then applied via mutable operations afterward.
enum InstrAction {
    /// A phi node — already handled in the phi-processing phase.
    Phi,
    /// A store to a promoted alloca: push the stored value as a new definition.
    PromotedStore {
        /// Alloca slot index.
        slot: usize,
        /// The value being stored (becomes the new reaching definition).
        value: ValueId,
    },
    /// A load from a promoted alloca: replace with the current reaching definition.
    PromotedLoad {
        /// Alloca slot index.
        slot: usize,
        /// The load instruction's result ValueId (to be replaced).
        result: ValueId,
    },
    /// Any instruction not directly involved in alloca promotion.
    Other,
}

// ---------------------------------------------------------------------------
// rename_variables — main entry point
// ---------------------------------------------------------------------------

/// Renames variables in `func` to produce valid SSA form.
///
/// This is the main entry point for the SSA renaming pass — the final step
/// of Phase 7 ("alloca-then-promote" SSA construction). It performs:
///
/// 1. **Phi instruction creation:** Inserts phi instructions at the
///    locations specified by `phi_placements` (computed by the dominance
///    frontier pass).
///
/// 2. **Dominator tree walk:** Traverses the dominator tree in pre-order,
///    maintaining per-variable reaching-definition stacks, filling phi
///    operands, and recording load-to-reaching-def mappings.
///
/// 3. **Use replacement:** Applies the accumulated value replacement map
///    to rewrite all uses of promoted load results to their SSA reaching
///    definitions.
///
/// 4. **Dead instruction removal:** Removes the now-dead alloca, load, and
///    store instructions for promoted variables.
///
/// 5. **Def-use chain construction:** Builds and returns a map from each
///    SSA value to the list of `(block, instruction_index)` pairs that use it.
///
/// # Arguments
///
/// * `func` — the IR function to transform (mutated in place).
/// * `dom_tree` — pre-computed dominator tree for `func`.
/// * `promotable_allocas` — metadata for each alloca eligible for promotion.
///   The `undef_value` field must be pre-populated (e.g., by the caller
///   allocating sentinel values via `func.new_value()`).
/// * `phi_placements` — output of the dominance frontier pass: maps each
///   block ID to the list of `(alloca_slot_index, ir_type)` entries
///   indicating which phi nodes to place there.
///
/// # Returns
///
/// A def-use map: `FxHashMap<ValueId, Vec<(BasicBlockId, usize)>>` mapping
/// each SSA value to the instruction locations that use it. This is consumed
/// by subsequent optimization passes.
///
/// # Panics
///
/// Panics if:
/// - A block referenced in `phi_placements` does not exist in `func`.
/// - The dominator tree is inconsistent with `func`'s CFG.
pub fn rename_variables(
    func: &mut IrFunction,
    dom_tree: &DominatorTree,
    promotable_allocas: &[AllocaSlot],
    phi_placements: &FxHashMap<BasicBlockId, Vec<(usize, IrType)>>,
) -> FxHashMap<ValueId, Vec<(BasicBlockId, usize)>> {
    // Handle trivial case: nothing to promote.
    if promotable_allocas.is_empty() {
        return fx_hash_map();
    }

    // ---- Step 1: Create AllocaSlots with undef sentinels ----
    // The caller should have already populated undef_value in each slot.
    // We clone the slots into SsaRenamer.
    let alloca_slots: Vec<AllocaSlot> = promotable_allocas.to_vec();

    // ---- Step 2: Create SsaRenamer ----
    let mut renamer = SsaRenamer::new(alloca_slots);

    // ---- Step 3: Insert phi instructions and build phi_map ----
    // For each block in phi_placements, create Phi instructions with empty
    // incoming lists and insert them at the beginning of the block. Record
    // the mapping from block to (slot_index, phi_result_value_id).
    for (&block_id, placements) in phi_placements.iter() {
        let mut phi_entries: Vec<(usize, ValueId)> = Vec::with_capacity(placements.len());

        for &(slot_index, ref ir_type) in placements.iter() {
            // Derive a debug name for the phi result from the alloca's name.
            let phi_name = renamer.alloca_slots[slot_index]
                .name
                .as_ref()
                .map(|n| format!("{}.phi", n));

            // Allocate a fresh SSA value for the phi result.
            let phi_result = func.new_value(ir_type.clone(), phi_name);

            // Create the Phi instruction with empty incoming edges.
            let phi_inst = Instruction::Phi {
                result: phi_result,
                ty: ir_type.clone(),
                incoming: Vec::new(),
            };

            // Insert the phi at the beginning of the block, after any
            // existing phi nodes (maintaining the phi-first invariant).
            let block = func.get_block_mut(block_id);
            let insert_pos = block.phi_count();
            block.insert_instruction(insert_pos, phi_inst);

            phi_entries.push((slot_index, phi_result));
        }

        renamer.phi_map.insert(block_id, phi_entries);
    }

    // ---- Step 4: Recursive dominator tree walk ----
    let entry_block_id = func.entry_block_id;
    rename_block(&mut renamer, func, entry_block_id, dom_tree);

    // ---- Step 5: Apply value replacement map ----
    // Iterate all blocks and instructions, replacing every use of a promoted
    // load result with its reaching definition. We use the dominator tree
    // preorder traversal for systematic coverage.
    let preorder: Vec<BasicBlockId> = dom_tree.preorder().to_vec();
    for &block_id in &preorder {
        let block = func.get_block_mut(block_id);
        let instr_count = block.instructions().len();
        for i in 0..instr_count {
            let instructions = block.instructions_mut();
            for (&old_val, &new_val) in renamer.value_map.iter() {
                instructions[i].replace_use(old_val, new_val);
            }
        }
    }

    // ---- Step 6: Remove promoted alloca/load/store instructions ----
    let promoted_alloca_ids: FxHashSet<ValueId> = renamer
        .alloca_slots
        .iter()
        .map(|slot| slot.alloca_value_id)
        .collect();
    remove_promoted_instructions(func, &promoted_alloca_ids, &renamer.value_map);

    // ---- Step 7: Build def-use chains ----
    let def_use_map = build_def_use_map(func, dom_tree);

    def_use_map
}

// ---------------------------------------------------------------------------
// rename_block — recursive dominator tree walk
// ---------------------------------------------------------------------------

/// Processes a single basic block during the SSA renaming dominator tree walk.
///
/// This function implements the core of the renaming algorithm for one block:
///
/// 1. **Save stack depths** for later restoration.
/// 2. **Process phi nodes** — push each phi result onto its slot's stack.
/// 3. **Walk instructions** — handle stores (push new def), loads (record
///    replacement), skip phis and other instructions.
/// 4. **Fill successor phi operands** with current reaching definitions.
/// 5. **Recurse** into dominator tree children.
/// 6. **Restore stack depths** — pop definitions made in this block.
///
/// # Arguments
///
/// * `renamer` — mutable renaming state (stacks, maps).
/// * `func` — the IR function being transformed.
/// * `block_id` — the block to process.
/// * `dom_tree` — the dominator tree for traversal.
fn rename_block(
    renamer: &mut SsaRenamer,
    func: &mut IrFunction,
    block_id: BasicBlockId,
    dom_tree: &DominatorTree,
) {
    let num_slots = renamer.alloca_slots.len();

    // ---- 1. Save stack depths for unwinding after recursion ----
    let saved_depths: Vec<usize> = renamer
        .reaching_defs
        .iter()
        .map(|stack| stack.len())
        .collect();

    // ---- 2. Process phi nodes in this block ----
    // If this block has phi nodes for promoted allocas, push each phi's
    // result onto the corresponding slot's reaching-definition stack.
    if let Some(phi_entries) = renamer.phi_map.get(&block_id).cloned() {
        for &(slot_idx, phi_value_id) in &phi_entries {
            renamer.reaching_defs[slot_idx].push(phi_value_id);
        }
    }

    // ---- 3. Walk instructions and classify actions ----
    // We collect instruction actions from an immutable borrow of the block,
    // then apply mutations (stack pushes, value_map insertions) afterward
    // to satisfy the Rust borrow checker.
    let actions: Vec<InstrAction> = {
        let block = func.get_block(block_id);
        let instructions = block.instructions();
        let mut actions = Vec::with_capacity(instructions.len());

        for inst in instructions.iter() {
            if inst.is_phi() {
                actions.push(InstrAction::Phi);
                continue;
            }

            match inst {
                Instruction::Store {
                    value, ptr, volatile, ..
                } if !*volatile && renamer.is_promoted_alloca(*ptr) => {
                    let slot = renamer.get_slot(*ptr).unwrap();
                    actions.push(InstrAction::PromotedStore {
                        slot,
                        value: *value,
                    });
                }
                Instruction::Load {
                    result, ptr, volatile, ..
                } if !*volatile && renamer.is_promoted_alloca(*ptr) => {
                    let slot = renamer.get_slot(*ptr).unwrap();
                    actions.push(InstrAction::PromotedLoad {
                        slot,
                        result: *result,
                    });
                }
                _ => {
                    actions.push(InstrAction::Other);
                }
            }
        }

        actions
    };

    // ---- Apply actions: update reaching_defs and value_map ----
    // Process actions in instruction order so that definitions from earlier
    // instructions are visible to later ones within the same block.
    for action in &actions {
        match action {
            InstrAction::Phi => {
                // Already handled in step 2.
            }
            InstrAction::PromotedStore { slot, value } => {
                // A store to this promoted alloca creates a new definition.
                renamer.reaching_defs[*slot].push(*value);
            }
            InstrAction::PromotedLoad { slot, result } => {
                // A load from this promoted alloca: map the load result to
                // the current reaching definition (top of stack).
                let reaching_def = renamer.current_def(*slot);
                renamer.value_map.insert(*result, reaching_def);
            }
            InstrAction::Other => {
                // No action needed during the walk.
            }
        }
    }

    // ---- 4. Fill phi operands in successor blocks ----
    // For each CFG successor of this block, find the phi nodes placed
    // there for promoted allocas and add the current reaching definition
    // as an incoming value from this block.
    let successors: Vec<BasicBlockId> = func.get_block(block_id).successors().to_vec();
    for succ_id in successors {
        fill_successor_phis(renamer, func, block_id, succ_id);
    }

    // ---- 5. Recurse into dominator tree children ----
    let children: Vec<BasicBlockId> = dom_tree.children(block_id).to_vec();
    for child_id in children {
        rename_block(renamer, func, child_id, dom_tree);
    }

    // ---- 6. Restore reaching-definition stack depths ----
    // Pop all definitions pushed during this block's processing so that
    // sibling subtrees in the dominator tree do not see them.
    for i in 0..num_slots {
        renamer.reaching_defs[i].truncate(saved_depths[i]);
    }
}

// ---------------------------------------------------------------------------
// fill_successor_phis — phi operand filling
// ---------------------------------------------------------------------------

/// Fills phi node operands in a successor block with reaching definitions
/// from the current block.
///
/// For each phi node placed in `succ_id` that corresponds to a promoted
/// alloca, adds `(current_reaching_def, from_block_id)` as an incoming
/// edge. The reaching definition is the top of the slot's stack at the
/// exit point of `from_block_id`.
///
/// # Arguments
///
/// * `renamer` — the renaming state (read-only access for stacks and phi_map).
/// * `func` — the IR function (mutable access to insert phi operands).
/// * `from_block_id` — the predecessor block providing the incoming value.
/// * `succ_id` — the successor block containing the phi nodes to fill.
fn fill_successor_phis(
    renamer: &SsaRenamer,
    func: &mut IrFunction,
    from_block_id: BasicBlockId,
    succ_id: BasicBlockId,
) {
    // Look up the phi entries for the successor block.
    let phi_entries = match renamer.phi_map.get(&succ_id) {
        Some(entries) => entries.clone(),
        None => return, // No promoted-alloca phi nodes in this successor.
    };

    // For each phi node in the successor, add the current reaching definition
    // from the predecessor block.
    let block = func.get_block_mut(succ_id);
    for &(slot_idx, phi_value_id) in &phi_entries {
        let reaching_def = renamer.current_def(slot_idx);

        // Find the phi instruction with this result ValueId and add the
        // incoming edge. We search through the phi nodes at the front of
        // the block.
        let instructions = block.instructions_mut();
        for inst in instructions.iter_mut() {
            if !inst.is_phi() {
                break; // Past the phi prefix — no more phis to check.
            }
            if inst.result() == Some(phi_value_id) {
                inst.add_phi_operand(reaching_def, from_block_id);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// remove_promoted_instructions — dead instruction cleanup
// ---------------------------------------------------------------------------

/// Removes alloca, load, and store instructions that were promoted to SSA
/// form during the renaming pass.
///
/// After SSA renaming, the original alloca instructions, loads from promoted
/// allocas, and stores to promoted allocas are dead — their effects have been
/// replaced by phi nodes, reaching definitions, and the value replacement map.
/// This function removes those instructions from the function's basic blocks.
///
/// # Arguments
///
/// * `func` — the IR function to clean up (mutated in place).
/// * `promoted_allocas` — set of alloca result [`ValueId`]s that were promoted.
///   A load/store is considered promoted if its pointer operand is in this set.
/// * `load_replacements` — the value map from the renaming pass. A load whose
///   result is in this map was replaced and should be removed.
///
/// # Instruction Removal Strategy
///
/// Instructions are removed in reverse index order within each block to
/// avoid index invalidation during removal. This is critical because
/// [`BasicBlock::remove_instruction()`] shifts subsequent instructions left.
pub fn remove_promoted_instructions(
    func: &mut IrFunction,
    promoted_allocas: &FxHashSet<ValueId>,
    load_replacements: &FxHashMap<ValueId, ValueId>,
) {
    // Collect (block_index_in_func, instruction_indices_to_remove) for each block.
    // We use block indices into func.basic_blocks rather than BasicBlockIds to
    // simplify iteration over the mutable vec.
    let block_count = func.basic_blocks.len();

    for block_idx in 0..block_count {
        // Collect indices of instructions to remove from this block.
        let mut indices_to_remove: Vec<usize> = Vec::new();

        {
            let block = &func.basic_blocks[block_idx];
            let instructions = block.instructions();

            for (i, inst) in instructions.iter().enumerate() {
                let should_remove = if inst.is_alloca() {
                    // Remove promoted alloca instructions. Use is_alloca()
                    // for fast classification, then check result membership.
                    inst.result()
                        .map_or(false, |r| promoted_allocas.contains(&r))
                } else {
                    match inst {
                        // Remove loads from promoted allocas (their results
                        // are now in the value replacement map).
                        Instruction::Load { result, ptr, .. } => {
                            promoted_allocas.contains(ptr)
                                || load_replacements.contains_key(result)
                        }
                        // Remove stores to promoted allocas.
                        Instruction::Store { ptr, .. } => {
                            promoted_allocas.contains(ptr)
                        }
                        _ => false,
                    }
                };

                if should_remove {
                    indices_to_remove.push(i);
                }
            }
        }

        // Remove in reverse order to preserve index validity.
        for &idx in indices_to_remove.iter().rev() {
            func.basic_blocks[block_idx].remove_instruction(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// build_def_use_map — def-use chain construction
// ---------------------------------------------------------------------------

/// Builds a def-use map for the function after SSA renaming.
///
/// Scans every instruction in every reachable block (in dominator tree
/// pre-order) and records which SSA values each instruction uses. The
/// result maps each [`ValueId`] to the list of `(BasicBlockId, instruction_index)`
/// pairs that reference it.
///
/// This map is consumed by subsequent optimization passes:
/// - **Dead code elimination:** An instruction whose result has no uses
///   (and no side effects) can be removed.
/// - **Constant propagation:** Values with a single definition can have
///   their uses rewritten when the definition is a known constant.
///
/// # Arguments
///
/// * `func` — the IR function to analyze (read-only).
/// * `dom_tree` — the dominator tree (used for pre-order block ordering).
///
/// # Returns
///
/// A `FxHashMap<ValueId, Vec<(BasicBlockId, usize)>>` mapping each value
/// to its use sites.
fn build_def_use_map(
    func: &IrFunction,
    dom_tree: &DominatorTree,
) -> FxHashMap<ValueId, Vec<(BasicBlockId, usize)>> {
    let mut def_use_map: FxHashMap<ValueId, Vec<(BasicBlockId, usize)>> = fx_hash_map();

    // Iterate blocks in dominator tree pre-order for deterministic output.
    let preorder = dom_tree.preorder();
    for &block_id in preorder {
        let block = func.get_block(block_id);
        let instructions = block.instructions();

        for (inst_idx, inst) in instructions.iter().enumerate() {
            // For phi nodes, use phi_operands() to capture the full
            // (value, source_block) structure; for other instructions,
            // use the generic uses() accessor.
            if inst.is_phi() {
                if let Some(operands) = inst.phi_operands() {
                    for &(val, _source_block) in operands {
                        def_use_map
                            .entry(val)
                            .or_insert_with(Vec::new)
                            .push((block_id, inst_idx));
                    }
                }
            } else {
                let used_values = inst.uses();
                for used_val in used_values {
                    def_use_map
                        .entry(used_val)
                        .or_insert_with(Vec::new)
                        .push((block_id, inst_idx));
                }
            }
        }
    }

    def_use_map
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::fx_hash::fx_hash_set;
    use crate::ir::function::IrFunction;
    use crate::ir::mem2reg::dominator_tree::DominatorTree;

    /// Helper: create a minimal function with an entry block.
    fn make_function(name: &str) -> IrFunction {
        IrFunction::new(name.to_string(), IrType::Void, vec![])
    }

    /// Helper: create an alloca slot with the given value ID and type.
    fn make_slot(
        func: &mut IrFunction,
        alloca_id: ValueId,
        ty: IrType,
        name: Option<&str>,
    ) -> AllocaSlot {
        let undef_value = func.new_value(ty.clone(), name.map(|n| format!("{}.undef", n)));
        AllocaSlot {
            alloca_value_id: alloca_id,
            ty,
            name: name.map(String::from),
            undef_value,
        }
    }

    #[test]
    fn test_ssa_renamer_new_empty() {
        let renamer = SsaRenamer::new(vec![]);
        assert!(renamer.reaching_defs.is_empty());
        assert!(renamer.phi_map.is_empty());
        assert!(renamer.alloca_slots.is_empty());
    }

    #[test]
    fn test_ssa_renamer_current_def_returns_undef_when_empty() {
        let mut func = make_function("test");
        let alloca_id = func.new_value(IrType::Ptr, Some("ptr_x".into()));
        let slot = make_slot(&mut func, alloca_id, IrType::I32, Some("x"));
        let undef = slot.undef_value;
        let renamer = SsaRenamer::new(vec![slot]);

        // Stack is empty → should return undef sentinel.
        assert_eq!(renamer.current_def(0), undef);
    }

    #[test]
    fn test_ssa_renamer_current_def_returns_top_of_stack() {
        let mut func = make_function("test");
        let alloca_id = func.new_value(IrType::Ptr, Some("ptr_x".into()));
        let slot = make_slot(&mut func, alloca_id, IrType::I32, Some("x"));
        let mut renamer = SsaRenamer::new(vec![slot]);

        let val1 = ValueId(100);
        let val2 = ValueId(101);

        renamer.reaching_defs[0].push(val1);
        assert_eq!(renamer.current_def(0), val1);

        renamer.reaching_defs[0].push(val2);
        assert_eq!(renamer.current_def(0), val2);
    }

    #[test]
    fn test_ssa_renamer_is_promoted_alloca() {
        let mut func = make_function("test");
        let alloca_id = func.new_value(IrType::Ptr, Some("ptr_x".into()));
        let slot = make_slot(&mut func, alloca_id, IrType::I32, Some("x"));
        let renamer = SsaRenamer::new(vec![slot]);

        assert!(renamer.is_promoted_alloca(alloca_id));
        assert!(!renamer.is_promoted_alloca(ValueId(999)));
    }

    #[test]
    fn test_rename_variables_empty_promotable() {
        let mut func = make_function("test");
        let dom_tree = DominatorTree::compute(&func);
        let phi_placements: FxHashMap<BasicBlockId, Vec<(usize, IrType)>> = fx_hash_map();

        let def_use = rename_variables(&mut func, &dom_tree, &[], &phi_placements);
        assert!(def_use.is_empty());
    }

    #[test]
    fn test_rename_single_block_store_load() {
        // Build a function with a single block:
        //   %p = alloca i32
        //   store i32 %val, ptr %p
        //   %x = load i32, ptr %p
        //   ret void
        let mut func = make_function("test_single");

        // Create values.
        let val = func.new_value(IrType::I32, Some("val".into()));
        let alloca_result = func.new_value(IrType::Ptr, Some("p".into()));
        let load_result = func.new_value(IrType::I32, Some("x".into()));

        // Build instructions in the entry block.
        let entry_id = func.entry_block_id;
        {
            let block = func.get_block_mut(entry_id);
            block.add_instruction(Instruction::Alloca {
                result: alloca_result,
                ty: IrType::I32,
                alignment: 4,
            });
            block.add_instruction(Instruction::Store {
                value: val,
                ptr: alloca_result,
                volatile: false,
            });
            block.add_instruction(Instruction::Load {
                result: load_result,
                ptr: alloca_result,
                ty: IrType::I32,
                volatile: false,
            });
            block.set_terminator(Instruction::Return { value: Some(load_result) });
        }

        // Create alloca slot.
        let undef_val = func.new_value(IrType::I32, Some("x.undef".into()));
        let slot = AllocaSlot {
            alloca_value_id: alloca_result,
            ty: IrType::I32,
            name: Some("x".into()),
            undef_value: undef_val,
        };

        let dom_tree = DominatorTree::compute(&func);
        let phi_placements: FxHashMap<BasicBlockId, Vec<(usize, IrType)>> = fx_hash_map();

        let _def_use = rename_variables(&mut func, &dom_tree, &[slot], &phi_placements);

        // After renaming:
        // - The alloca, store, and load should be removed.
        // - The return should use `val` instead of `load_result`.
        let block = func.get_block(entry_id);
        let instructions = block.instructions();

        // Only the return instruction should remain.
        assert_eq!(instructions.len(), 1, "Expected only the return instruction");
        match &instructions[0] {
            Instruction::Return { value: Some(v) } => {
                assert_eq!(*v, val, "Return should use the stored value, not the load result");
            }
            other => panic!("Expected Return instruction, got: {:?}", other),
        }
    }

    #[test]
    fn test_remove_promoted_instructions_basic() {
        let mut func = make_function("test_remove");

        let alloca_result = func.new_value(IrType::Ptr, Some("p".into()));
        let store_val = func.new_value(IrType::I32, Some("val".into()));
        let load_result = func.new_value(IrType::I32, Some("x".into()));

        let entry_id = func.entry_block_id;
        {
            let block = func.get_block_mut(entry_id);
            block.add_instruction(Instruction::Alloca {
                result: alloca_result,
                ty: IrType::I32,
                alignment: 4,
            });
            block.add_instruction(Instruction::Store {
                value: store_val,
                ptr: alloca_result,
                volatile: false,
            });
            block.add_instruction(Instruction::Load {
                result: load_result,
                ptr: alloca_result,
                ty: IrType::I32,
                volatile: false,
            });
            block.set_terminator(Instruction::Return { value: None });
        }

        let mut promoted: FxHashSet<ValueId> = fx_hash_set();
        promoted.insert(alloca_result);

        let mut load_map: FxHashMap<ValueId, ValueId> = fx_hash_map();
        load_map.insert(load_result, store_val);

        remove_promoted_instructions(&mut func, &promoted, &load_map);

        // Only the return should remain.
        let block = func.get_block(entry_id);
        assert_eq!(block.instructions().len(), 1);
        assert!(block.instructions()[0].is_terminator());
    }

    #[test]
    fn test_alloca_slot_fields() {
        let slot = AllocaSlot {
            alloca_value_id: ValueId(0),
            ty: IrType::I32,
            name: Some("my_var".to_string()),
            undef_value: ValueId(100),
        };

        assert_eq!(slot.alloca_value_id, ValueId(0));
        assert_eq!(slot.undef_value, ValueId(100));
        assert_eq!(slot.name.as_deref(), Some("my_var"));
    }

    #[test]
    fn test_renamer_stack_save_restore() {
        // Verify that reaching_defs stacks can be saved and restored,
        // simulating the dominator tree scoping behavior.
        let mut func = make_function("test");
        let alloca_id = func.new_value(IrType::Ptr, None);
        let slot = make_slot(&mut func, alloca_id, IrType::I32, None);
        let mut renamer = SsaRenamer::new(vec![slot]);

        // Push a definition at the "outer" scope.
        renamer.reaching_defs[0].push(ValueId(10));
        let saved_depth = renamer.reaching_defs[0].len();

        // Push a definition at the "inner" scope (dominated block).
        renamer.reaching_defs[0].push(ValueId(20));
        assert_eq!(renamer.current_def(0), ValueId(20));

        // Restore to the outer scope.
        renamer.reaching_defs[0].truncate(saved_depth);
        assert_eq!(renamer.current_def(0), ValueId(10));
    }
}
