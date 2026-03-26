//! SSA Renaming Pass
//!
//! Performs the renaming phase of SSA construction. After phi nodes have been
//! inserted at the iterated dominance frontier locations, this pass:
//!
//! 1. Walks the dominator tree in preorder
//! 2. Maintains a reaching-definition stack per promoted alloca variable
//! 3. Replaces loads from promoted allocas with the current reaching definition value
//! 4. Replaces stores to promoted allocas by pushing new value definitions
//! 5. Fills phi-node operands in successor blocks with the current reaching definition
//! 6. After processing dominator subtree, restores stacks (pops pushed definitions)
//!
//! This converts the alloca-based IR produced by Phase 6 (lowering) into
//! proper SSA form for consumption by optimization passes (Phase 8).
//!
//! # Algorithm — Cytron et al. SSA Renaming
//!
//! The renaming algorithm is a depth-first walk of the dominator tree.
//! For each basic block visited:
//!
//! - Phi nodes placed by the dominance frontier pass receive the phi result
//!   value as a new definition for their corresponding alloca.
//! - Load instructions from promoted allocas are replaced with the current
//!   reaching definition (the top of the per-variable definition stack).
//! - Store instructions to promoted allocas push the stored value as a new
//!   definition onto the stack.
//! - Successor blocks' phi-node operands are filled with the current reaching
//!   definition for the appropriate alloca.
//! - After recursing into all dominator-tree children, definition stacks are
//!   restored to their pre-block depths, ensuring sibling blocks in the
//!   dominator tree see the correct (ancestor) definitions.
//!
//! # Zero-Dependency
//!
//! This module depends only on `crate::` internal modules and the Rust
//! standard library — no external crates are used.

use crate::common::diagnostics::Span;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::types::IrType;

use super::dominator_tree::DominatorTree;

// ===========================================================================
// Internal helper: compact instruction-level data extracted from a block
// ===========================================================================

/// Compact representation of instruction data relevant to SSA renaming.
///
/// By extracting only the fields we need (and copying `Value` — a `Copy`
/// type — rather than borrowing the instruction), we avoid holding a
/// shared borrow on the function while mutating `self` (the renamer).
enum InstAction {
    /// Phi node — already handled separately; skip.
    Phi,
    /// Load from a pointer: `(result_value, pointer_value)`.
    Load { result: Value, ptr: Value },
    /// Store a value to a pointer: `(stored_value, pointer_value)`.
    Store { value: Value, ptr: Value },
    /// Any other instruction — no action needed during renaming.
    Other,
}

// ===========================================================================
// SsaRenamer — the public renaming context
// ===========================================================================

/// SSA renaming state for the mem2reg pass.
///
/// Maintains per-variable definition stacks and tracks which phi nodes
/// need to be filled with operands from each predecessor block.
///
/// # Usage
///
/// ```ignore
/// // 1. Insert phi nodes at dominance frontiers
/// let (phi_map, next_val) = insert_phi_nodes(&mut func, &phi_locs, &alloca_types, next_val);
///
/// // 2. Rename: replace loads/stores with SSA values
/// let mut renamer = SsaRenamer::new(alloca_types.clone(), phi_map, next_val);
/// renamer.rename(&mut func, &dom_tree);
///
/// // 3. Remove dead loads, stores, and allocas
/// cleanup_promoted_instructions(&mut func, &alloca_types);
/// ```
pub struct SsaRenamer {
    /// For each promoted alloca (identified by its [`Value`]), a stack of
    /// reaching definitions.  The top of the stack is the current SSA value
    /// for that variable.  Entries are pushed on stores and phi-node
    /// processing, and popped when leaving a dominator subtree.
    reaching_defs: FxHashMap<Value, Vec<Value>>,

    /// Map from alloca [`Value`] to its [`IrType`].
    /// Used for type lookups when creating phi instructions and for
    /// determining which allocas are promotable.
    alloca_types: FxHashMap<Value, IrType>,

    /// Map from `(block_index, alloca_value)` → phi-node result [`Value`].
    /// Tracks which phi nodes exist in which blocks for which variables.
    /// Populated by [`insert_phi_nodes`] before renaming begins.
    phi_nodes: FxHashMap<(usize, Value), Value>,

    /// Next available SSA value number.  Incremented if new values are
    /// ever needed during renaming (reserved for future use; currently
    /// all values are created by [`insert_phi_nodes`]).
    next_value: u32,

    /// Value replacement map built during the renaming walk.
    ///
    /// Maps each promoted-load result [`Value`] to its reaching-definition
    /// replacement.  Applied in a single post-renaming pass over all
    /// instructions via [`apply_replacements`](SsaRenamer::apply_replacements).
    replacements: FxHashMap<Value, Value>,
}

// ===========================================================================
// SsaRenamer — construction
// ===========================================================================

impl SsaRenamer {
    /// Create a new SSA renamer.
    ///
    /// # Parameters
    ///
    /// - `promotable_allocas`: Map from alloca [`Value`] → allocated [`IrType`].
    ///   Only allocas in this map are considered for promotion.
    /// - `phi_locations`: Map from `(block_index, alloca_value)` →
    ///   phi-result [`Value`].  These phi nodes must already be inserted
    ///   into the function (via [`insert_phi_nodes`]).
    /// - `next_value`: The next available SSA value number in the function
    ///   (equal to `IrFunction::value_count` after phi insertion).
    pub fn new(
        promotable_allocas: FxHashMap<Value, IrType>,
        phi_locations: FxHashMap<(usize, Value), Value>,
        next_value: u32,
    ) -> Self {
        // Pre-populate reaching_defs with an empty stack for every
        // promotable alloca so that `current_def` always finds an entry.
        let mut reaching_defs: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
        for &alloca_val in promotable_allocas.keys() {
            reaching_defs.insert(alloca_val, Vec::new());
        }

        SsaRenamer {
            reaching_defs,
            alloca_types: promotable_allocas,
            phi_nodes: phi_locations,
            next_value,
            replacements: FxHashMap::default(),
        }
    }
}

// ===========================================================================
// SsaRenamer — core renaming algorithm
// ===========================================================================

impl SsaRenamer {
    /// Perform SSA renaming on the function.
    ///
    /// Walks the dominator tree in preorder starting from the entry block,
    /// replacing loads/stores of promoted allocas with SSA values and phi
    /// references.  After the walk, a single pass applies the collected
    /// value replacements to all remaining instructions.
    ///
    /// # Parameters
    ///
    /// - `func`: The IR function to transform (mutated in place).
    /// - `dom_tree`: Pre-computed dominator tree for `func`.
    pub fn rename(&mut self, func: &mut IrFunction, dom_tree: &DominatorTree) {
        if func.block_count() == 0 {
            return;
        }

        // Walk the dominator tree starting from the entry block (index 0).
        self.rename_block(func, 0, dom_tree);

        // Apply all collected load-result → reaching-def replacements.
        self.apply_replacements(func);

        // Ensure the function's value_count is up-to-date.
        if self.next_value > func.value_count {
            func.value_count = self.next_value;
        }
    }

    /// Rename a single basic block and recursively rename its dominator
    /// tree children.
    ///
    /// # Algorithm
    ///
    /// 1. Record current stack depths (for restoration after subtree).
    /// 2. Process phi nodes in this block — push phi results as new defs.
    /// 3. Process non-phi instructions:
    ///    - **Load** from promoted alloca → record replacement with
    ///      the current reaching definition.
    ///    - **Store** to promoted alloca → push stored value (resolved
    ///      through the replacement map) as a new definition.
    ///    - **Other** → leave unchanged (operands fixed in the
    ///      post-renaming replacement pass).
    /// 4. Fill phi-node operands in successor blocks.
    /// 5. Recursively rename dominator tree children.
    /// 6. Pop definitions to restore stacks to pre-block state.
    fn rename_block(&mut self, func: &mut IrFunction, block_idx: usize, dom_tree: &DominatorTree) {
        // Guard against out-of-bounds (should not happen with a valid
        // dominator tree, but defensive coding costs nothing).
        if block_idx >= func.block_count() {
            return;
        }

        // ------------------------------------------------------------------
        // 1. Save stack depths for restoration after this subtree.
        // ------------------------------------------------------------------
        let saved_depths = self.save_stack_depths();

        // ------------------------------------------------------------------
        // 2. Process phi nodes placed in this block by insert_phi_nodes.
        //    Each phi result becomes the new reaching definition for its
        //    corresponding alloca.
        // ------------------------------------------------------------------
        let block_phi_defs: Vec<(Value, Value)> = self
            .phi_nodes
            .iter()
            .filter_map(|(&(bidx, alloca_val), &phi_result)| {
                if bidx == block_idx {
                    Some((alloca_val, phi_result))
                } else {
                    None
                }
            })
            .collect();

        for (alloca_val, phi_result) in block_phi_defs {
            self.push_def(alloca_val, phi_result);
        }

        // ------------------------------------------------------------------
        // 3. Process non-phi instructions.
        //
        //    We extract the minimal data we need (Value copies) into an
        //    intermediate Vec to release the immutable borrow on `func`
        //    before mutating `self`.
        // ------------------------------------------------------------------
        let actions: Vec<InstAction> = {
            let instructions = func.blocks()[block_idx].instructions();
            instructions
                .iter()
                .map(|inst| {
                    if inst.is_phi() {
                        InstAction::Phi
                    } else {
                        match inst {
                            Instruction::Load { result, ptr, .. } => InstAction::Load {
                                result: *result,
                                ptr: *ptr,
                            },
                            Instruction::Store { value, ptr, .. } => InstAction::Store {
                                value: *value,
                                ptr: *ptr,
                            },
                            _ => InstAction::Other,
                        }
                    }
                })
                .collect()
        }; // immutable borrow of `func` released here

        for action in &actions {
            match action {
                InstAction::Phi => {
                    // Already handled in step 2.
                }
                InstAction::Load { result, ptr } => {
                    if self.alloca_types.contains_key(ptr) {
                        // This load reads a promoted alloca.  Record that
                        // the load's result value should be replaced by the
                        // current reaching definition everywhere it appears.
                        let reaching = self.current_def(ptr);
                        self.replacements.insert(*result, reaching);
                    }
                }
                InstAction::Store { value, ptr } => {
                    if self.alloca_types.contains_key(ptr) {
                        // This store writes to a promoted alloca.  Resolve
                        // the stored value through the replacement map
                        // (handles the case where the stored value is itself
                        // a promoted-load result) and push as a new def.
                        let actual_val = self.resolve_value(*value);
                        self.push_def(*ptr, actual_val);
                    }
                }
                InstAction::Other => {
                    // Non-load/store instructions are left unchanged during
                    // renaming.  Their operands are fixed by the post-walk
                    // apply_replacements pass.
                }
            }
        }

        // ------------------------------------------------------------------
        // 4. Fill phi-node operands in successor blocks.
        // ------------------------------------------------------------------
        self.fill_successor_phis(func, block_idx);

        // ------------------------------------------------------------------
        // 5. Recursively rename dominator tree children.
        // ------------------------------------------------------------------
        let children: Vec<usize> = dom_tree.children(block_idx).to_vec();
        for child_idx in children {
            self.rename_block(func, child_idx, dom_tree);
        }

        // ------------------------------------------------------------------
        // 6. Restore definition stacks to pre-block depths.
        // ------------------------------------------------------------------
        self.restore_stack_depths(&saved_depths);
    }
}

// ===========================================================================
// SsaRenamer — successor phi filling
// ===========================================================================

impl SsaRenamer {
    /// Fill phi-node operands in successor blocks.
    ///
    /// For each successor of `current_block`, if that successor contains a
    /// phi node for a promoted alloca, this method adds the current reaching
    /// definition as an incoming `(Value, BlockId)` pair from
    /// `current_block`.
    fn fill_successor_phis(&self, func: &mut IrFunction, current_block: usize) {
        // Copy successor list to release the immutable borrow on `func`.
        let successors: Vec<usize> = func.blocks()[current_block].successors().to_vec();
        let block_id = BlockId(current_block as u32);

        for succ_idx in successors {
            // Find all promoted-alloca phi nodes in this successor.
            let phi_entries: Vec<(Value, Value)> = self
                .phi_nodes
                .iter()
                .filter_map(|(&(bidx, alloca_val), &phi_result)| {
                    if bidx == succ_idx {
                        Some((alloca_val, phi_result))
                    } else {
                        None
                    }
                })
                .collect();

            for (alloca_val, phi_result) in phi_entries {
                let reaching_def = self.current_def(&alloca_val);

                // Locate the phi instruction in the successor and append
                // the incoming pair.
                if let Some(succ_block) = func.get_block_mut(succ_idx) {
                    for inst in succ_block.instructions_mut().iter_mut() {
                        match inst {
                            Instruction::Phi {
                                result, incoming, ..
                            } if *result == phi_result => {
                                incoming.push((reaching_def, block_id));
                                break;
                            }
                            _ if !inst.is_phi() => {
                                // Past the phi-node prefix — stop searching.
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// SsaRenamer — definition-stack management
// ===========================================================================

impl SsaRenamer {
    /// Record current stack depths for all promoted allocas.
    ///
    /// Used before processing a block so that the stacks can be restored
    /// to this state after the dominator subtree has been fully processed.
    fn save_stack_depths(&self) -> FxHashMap<Value, usize> {
        let mut saved: FxHashMap<Value, usize> = FxHashMap::default();
        for (alloca_val, stack) in self.reaching_defs.iter() {
            saved.insert(*alloca_val, stack.len());
        }
        saved
    }

    /// Restore definition stacks to previously saved depths.
    ///
    /// Truncates each stack to the depth recorded by [`save_stack_depths`],
    /// effectively popping all definitions pushed during the current block's
    /// processing.  This ensures that sibling blocks in the dominator tree
    /// see only the definitions from their common ancestor.
    fn restore_stack_depths(&mut self, saved: &FxHashMap<Value, usize>) {
        for (alloca_val, &depth) in saved.iter() {
            if let Some(stack) = self.reaching_defs.get_mut(alloca_val) {
                stack.truncate(depth);
            }
        }
    }

    /// Get the current reaching definition for a promoted alloca.
    ///
    /// Returns the top of the definition stack for the given alloca value.
    /// If no definition has been pushed (read-before-write), returns
    /// [`Value::UNDEF`] — the sentinel for undefined SSA values.
    #[inline]
    fn current_def(&self, alloca_value: &Value) -> Value {
        self.reaching_defs
            .get(alloca_value)
            .and_then(|stack| stack.last().copied())
            .unwrap_or(Value::UNDEF)
    }

    /// Push a new definition for a promoted alloca onto its
    /// reaching-definition stack.
    #[inline]
    fn push_def(&mut self, alloca_value: Value, new_value: Value) {
        self.reaching_defs
            .entry(alloca_value)
            .or_default()
            .push(new_value);
    }

    /// Resolve a value through the replacement map.
    ///
    /// If `value` is a promoted-load result that has already been mapped
    /// to a reaching definition, return that definition.  Otherwise,
    /// return `value` unchanged.
    ///
    /// This is used when processing stores whose stored value is itself
    /// the result of a promoted load — the actual reaching definition
    /// (not the dead load result) must be pushed onto the def stack.
    #[inline]
    fn resolve_value(&self, value: Value) -> Value {
        self.replacements.get(&value).copied().unwrap_or(value)
    }
}

// ===========================================================================
// SsaRenamer — post-renaming replacement application
// ===========================================================================

impl SsaRenamer {
    /// Apply all collected value replacements to every instruction in the
    /// function.
    ///
    /// After the dominator-tree renaming walk, the `replacements` map
    /// contains `promoted_load_result → reaching_definition` entries.
    /// This method iterates every instruction's operands and rewrites
    /// any reference to a promoted-load result with its replacement.
    fn apply_replacements(&self, func: &mut IrFunction) {
        if self.replacements.is_empty() {
            return;
        }

        for block in func.blocks_mut() {
            for inst in block.instructions_mut().iter_mut() {
                // Obtain mutable references to all value operands and
                // rewrite those that appear in the replacement map.
                for operand in inst.operands_mut() {
                    if let Some(&replacement) = self.replacements.get(&*operand) {
                        *operand = replacement;
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Free function: insert_phi_nodes
// ===========================================================================

/// Insert phi nodes at the specified dominance-frontier locations.
///
/// For each `(block_index, alloca_value)` pair in `phi_locations`, this
/// function creates a new [`Instruction::Phi`] with the alloca's [`IrType`],
/// assigns it a fresh [`Value`] number, and inserts it at the beginning of
/// the specified block (after existing phi nodes, before non-phi instructions).
///
/// # Parameters
///
/// - `func`: The IR function to modify.
/// - `phi_locations`: Set of `(block_index, alloca_value)` pairs indicating
///   where phi nodes are needed.
/// - `alloca_types`: Map from alloca [`Value`] → [`IrType`] (provides the
///   type for each phi instruction).
/// - `next_value`: The next available SSA value number.
///
/// # Returns
///
/// A tuple of:
/// - `FxHashMap<(usize, Value), Value>` — maps `(block_index, alloca_value)`
///   to the phi-node result [`Value`].
/// - `u32` — the updated next-value counter.
pub fn insert_phi_nodes(
    func: &mut IrFunction,
    phi_locations: &FxHashSet<(usize, Value)>,
    alloca_types: &FxHashMap<Value, IrType>,
    mut next_value: u32,
) -> (FxHashMap<(usize, Value), Value>, u32) {
    let mut phi_map: FxHashMap<(usize, Value), Value> = FxHashMap::default();

    // Sort locations for deterministic insertion order (block index first,
    // then alloca value index).  This ensures reproducible IR output.
    let mut sorted_locs: Vec<(usize, Value)> = phi_locations.iter().copied().collect();
    sorted_locs.sort_by_key(|&(bidx, val)| (bidx, val.0));

    for (block_idx, alloca_val) in sorted_locs {
        // Look up the alloca's type; skip if unknown (defensive).
        let ir_type = match alloca_types.get(&alloca_val) {
            Some(ty) => ty.clone(),
            None => continue,
        };

        // Allocate a fresh SSA value for the phi result.
        let phi_result = Value(next_value);
        next_value += 1;

        // Create the phi instruction with an empty incoming list.
        // Incoming pairs are filled later by SsaRenamer::fill_successor_phis.
        let phi_inst = Instruction::Phi {
            result: phi_result,
            ty: ir_type,
            incoming: Vec::new(),
            span: Span::dummy(),
        };

        // Insert at the correct position in the block (after existing phis).
        if let Some(block) = func.get_block_mut(block_idx) {
            block.add_phi(phi_inst);
        }

        // Record the mapping for SsaRenamer.
        phi_map.insert((block_idx, alloca_val), phi_result);
    }

    // Update the function's value counter.
    func.value_count = func.value_count.max(next_value);

    (phi_map, next_value)
}

// ===========================================================================
// Free function: cleanup_promoted_instructions
// ===========================================================================

/// Remove dead instructions left over after SSA renaming.
///
/// After the renaming pass has replaced all uses of promoted loads with
/// their reaching definitions, this function removes:
///
/// - **Load** instructions whose pointer operand is a promoted alloca
///   (their results have been replaced by reaching definitions).
/// - **Store** instructions whose pointer operand is a promoted alloca
///   (their values have been pushed onto definition stacks).
/// - **Alloca** instructions for promoted allocas from the entry block
///   (the allocations are no longer needed — the values live in SSA
///   virtual registers).
///
/// Instruction ordering for all remaining (non-promoted) instructions is
/// preserved.
///
/// # Parameters
///
/// - `func`: The IR function to clean up (mutated in place).
/// - `promotable_allocas`: Map from alloca [`Value`] → [`IrType`], the
///   same set used during renaming.
pub fn cleanup_promoted_instructions(
    func: &mut IrFunction,
    promotable_allocas: &FxHashMap<Value, IrType>,
) {
    if promotable_allocas.is_empty() {
        return;
    }

    // Build a fast-lookup set of promoted alloca values.
    let promoted_set: FxHashSet<Value> = promotable_allocas.keys().copied().collect();

    for block in func.blocks_mut() {
        block.instructions_mut().retain(|inst| {
            match inst {
                // Remove loads from promoted allocas.
                Instruction::Load { ptr, .. } => !promoted_set.contains(ptr),

                // Remove stores to promoted allocas.
                Instruction::Store { ptr, .. } => !promoted_set.contains(ptr),

                // Remove the alloca instructions themselves.
                Instruction::Alloca { result, .. } => !promoted_set.contains(result),

                // Keep all other instructions.
                _ => true,
            }
        });
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::{FunctionParam, IrFunction};
    use crate::ir::instructions::{Instruction, Value};
    use crate::ir::mem2reg::dominator_tree::DominatorTree;
    use crate::ir::types::IrType;

    /// Helper: create a simple function with one block and some allocas/loads/stores.
    fn make_simple_function() -> IrFunction {
        // int f(int param) { int x = param; return x; }
        //
        // entry:
        //   %0 = param
        //   %1 = alloca i32          ; x
        //   store i32 %0, ptr %1     ; x = param
        //   %2 = load i32, ptr %1    ; read x
        //   ret i32 %2
        let params = vec![FunctionParam::new("param".into(), IrType::I32, Value(0))];
        let mut func = IrFunction::new("f".into(), params, IrType::I32);
        func.value_count = 1; // param is Value(0)

        let alloca = Instruction::Alloca {
            result: Value(1),
            ty: IrType::I32,
            alignment: None,
            span: Span::dummy(),
        };
        let store = Instruction::Store {
            value: Value(0),
            ptr: Value(1),
            volatile: false,
            span: Span::dummy(),
        };
        let load = Instruction::Load {
            result: Value(2),
            ptr: Value(1),
            ty: IrType::I32,
            volatile: false,
            span: Span::dummy(),
        };
        let ret = Instruction::Return {
            value: Some(Value(2)),
            span: Span::dummy(),
        };

        func.value_count = 3; // Values 0, 1, 2
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca);
        entry.push_instruction(store);
        entry.push_instruction(load);
        entry.push_instruction(ret);

        func
    }

    #[test]
    fn test_ssa_renamer_single_block() {
        let mut func = make_simple_function();

        // Build dominator tree (single-block function).
        let dom_tree = DominatorTree::build(&func);

        // Alloca %1 is promotable: type I32
        let mut alloca_types: FxHashMap<Value, IrType> = FxHashMap::default();
        alloca_types.insert(Value(1), IrType::I32);

        // No phi nodes needed for a single-block function.
        let phi_locations: FxHashSet<(usize, Value)> = FxHashSet::default();
        let vc = func.value_count;
        let (phi_map, next_val) = insert_phi_nodes(&mut func, &phi_locations, &alloca_types, vc);

        assert!(phi_map.is_empty());
        assert_eq!(next_val, 3);

        // Rename.
        let mut renamer = SsaRenamer::new(alloca_types.clone(), phi_map, next_val);
        renamer.rename(&mut func, &dom_tree);

        // After renaming, the return instruction should use Value(0)
        // (the parameter) instead of Value(2) (the promoted load result).
        let entry = func.entry_block();
        let last = entry.instructions().last().unwrap();
        if let Instruction::Return { value, .. } = last {
            // %2 (load result) should have been replaced by %0 (param,
            // which was stored to the alloca).
            assert_eq!(*value, Some(Value(0)));
        } else {
            panic!("Expected Return instruction");
        }

        // Cleanup: remove the dead alloca, store, and load.
        cleanup_promoted_instructions(&mut func, &alloca_types);

        // After cleanup, only the return instruction should remain.
        let entry = func.entry_block();
        assert_eq!(
            entry.instructions().len(),
            1,
            "Expected only the return instruction after cleanup"
        );
    }

    #[test]
    fn test_insert_phi_nodes_basic() {
        let params = vec![FunctionParam::new("p".into(), IrType::I32, Value(0))];
        let mut func = IrFunction::new("g".into(), params, IrType::I32);
        func.value_count = 1;

        // Add a second block.
        let bb1 = BasicBlock::new(1);
        func.add_block(bb1);

        // Alloca %1, type I32
        let mut alloca_types: FxHashMap<Value, IrType> = FxHashMap::default();
        alloca_types.insert(Value(1), IrType::I32);

        // Phi needed at block 1 for alloca %1.
        let mut phi_locations: FxHashSet<(usize, Value)> = FxHashSet::default();
        phi_locations.insert((1, Value(1)));

        let (phi_map, next_val) = insert_phi_nodes(&mut func, &phi_locations, &alloca_types, 2);

        assert_eq!(phi_map.len(), 1);
        assert!(phi_map.contains_key(&(1, Value(1))));
        let phi_result = phi_map[&(1, Value(1))];
        assert_eq!(phi_result, Value(2));
        assert_eq!(next_val, 3);

        // Verify the phi instruction exists in block 1.
        let block1 = func.get_block(1).unwrap();
        let phi_insts: Vec<_> = block1.phi_instructions().collect();
        assert_eq!(phi_insts.len(), 1);
        if let Instruction::Phi {
            result,
            ty,
            incoming,
            ..
        } = &phi_insts[0]
        {
            assert_eq!(*result, Value(2));
            assert_eq!(*ty, IrType::I32);
            assert!(incoming.is_empty()); // Filled later by rename
        } else {
            panic!("Expected Phi instruction");
        }
    }

    #[test]
    fn test_cleanup_promoted_instructions() {
        let mut func = make_simple_function();

        let mut alloca_types: FxHashMap<Value, IrType> = FxHashMap::default();
        alloca_types.insert(Value(1), IrType::I32);

        // Before cleanup: 4 instructions (alloca, store, load, ret).
        assert_eq!(func.entry_block().instructions().len(), 4);

        cleanup_promoted_instructions(&mut func, &alloca_types);

        // After cleanup: only ret remains.
        assert_eq!(func.entry_block().instructions().len(), 1);
        assert!(matches!(
            func.entry_block().instructions()[0],
            Instruction::Return { .. }
        ));
    }

    #[test]
    fn test_current_def_undef() {
        let alloca_types: FxHashMap<Value, IrType> = {
            let mut m = FxHashMap::default();
            m.insert(Value(0), IrType::I32);
            m
        };
        let phi_nodes = FxHashMap::default();
        let renamer = SsaRenamer::new(alloca_types, phi_nodes, 1);

        // No definition pushed yet — should return UNDEF.
        assert_eq!(renamer.current_def(&Value(0)), Value::UNDEF);
    }

    #[test]
    fn test_push_and_current_def() {
        let alloca_types: FxHashMap<Value, IrType> = {
            let mut m = FxHashMap::default();
            m.insert(Value(0), IrType::I32);
            m
        };
        let phi_nodes = FxHashMap::default();
        let mut renamer = SsaRenamer::new(alloca_types, phi_nodes, 1);

        // Push two definitions.
        renamer.push_def(Value(0), Value(10));
        assert_eq!(renamer.current_def(&Value(0)), Value(10));

        renamer.push_def(Value(0), Value(20));
        assert_eq!(renamer.current_def(&Value(0)), Value(20));
    }

    #[test]
    fn test_save_restore_stack_depths() {
        let alloca_types: FxHashMap<Value, IrType> = {
            let mut m = FxHashMap::default();
            m.insert(Value(0), IrType::I32);
            m.insert(Value(1), IrType::I64);
            m
        };
        let phi_nodes = FxHashMap::default();
        let mut renamer = SsaRenamer::new(alloca_types, phi_nodes, 2);

        // Push some definitions.
        renamer.push_def(Value(0), Value(10));
        renamer.push_def(Value(1), Value(11));

        // Save.
        let saved = renamer.save_stack_depths();

        // Push more definitions.
        renamer.push_def(Value(0), Value(20));
        renamer.push_def(Value(1), Value(21));

        assert_eq!(renamer.current_def(&Value(0)), Value(20));
        assert_eq!(renamer.current_def(&Value(1)), Value(21));

        // Restore.
        renamer.restore_stack_depths(&saved);

        assert_eq!(renamer.current_def(&Value(0)), Value(10));
        assert_eq!(renamer.current_def(&Value(1)), Value(11));
    }

    #[test]
    fn test_resolve_value() {
        let alloca_types: FxHashMap<Value, IrType> = {
            let mut m = FxHashMap::default();
            m.insert(Value(0), IrType::I32);
            m
        };
        let phi_nodes = FxHashMap::default();
        let mut renamer = SsaRenamer::new(alloca_types, phi_nodes, 1);

        // No replacement — returns the value unchanged.
        assert_eq!(renamer.resolve_value(Value(5)), Value(5));

        // Add a replacement.
        renamer.replacements.insert(Value(5), Value(10));
        assert_eq!(renamer.resolve_value(Value(5)), Value(10));

        // Value not in the map — unchanged.
        assert_eq!(renamer.resolve_value(Value(99)), Value(99));
    }
}
