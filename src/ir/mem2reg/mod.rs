//! Memory-to-register promotion (mem2reg) subsystem — Phases 7 & 9 of the BCC pipeline.
//!
//! This module implements the SSA construction and elimination passes that form
//! the core of the mandated **alloca-then-promote** architecture:
//!
//! 1. **Phase 6** (in `crate::ir::lowering`) emits all local variables as `alloca`
//!    instructions in the function's entry block.
//! 2. **Phase 7** (this module) promotes eligible allocas to SSA virtual registers
//!    via dominance-frontier-based phi-node insertion and dominator-tree-ordered
//!    variable renaming.
//! 3. **Phase 9** (`phi_eliminate`) converts SSA phi nodes back to copy operations
//!    at predecessor block terminators for consumption by the register allocator.
//!
//! # Processing Pipeline
//!
//! ```text
//! identify_promotable_allocas  →  DominatorTree::compute  →  DominanceFrontier::compute
//!         →  compute_phi_placements (IDF)  →  build phi placement map
//!         →  SSA rename (insert phis, rename, cleanup)
//! ```
//!
//! # Promotability Criteria
//!
//! An alloca instruction is eligible for promotion to SSA registers when all of
//! the following conditions hold:
//!
//! - The allocated type is **scalar** (integer, floating-point, or pointer).
//!   Aggregate types (structs, arrays) larger than a register are not promoted.
//! - The alloca's result pointer is **never address-taken**: it only appears as
//!   the `ptr` operand of `Load` and `Store` instructions — never in
//!   `GetElementPtr`, `Call` arguments, `BitCast`, `Phi`, or any other context
//!   that would allow the address to escape.
//! - No **volatile** loads or stores reference the alloca. Volatile accesses
//!   have visible side effects that cannot be modeled in SSA form.
//!
//! # Submodules
//!
//! - [`dominator_tree`] — Lengauer-Tarjan dominator tree (O(n·α(n)))
//! - [`dominance_frontier`] — Dominance frontier and iterated DF computation
//! - [`ssa_builder`] — SSA variable renaming via reaching-definition stacks
//! - [`phi_eliminate`] — Phase 9 phi-node elimination to copies

// ── Submodule declarations ──────────────────────────────────────────────────

/// Dominator tree computation using the Lengauer-Tarjan algorithm.
/// Provides [`DominatorTree`] which is the foundational data structure for
/// both dominance frontier computation and SSA renaming traversal order.
pub mod dominator_tree;

/// Dominance frontier computation — implements the Cytron et al. 1991
/// algorithm for computing dominance frontiers and the iterated dominance
/// frontier (IDF) worklist algorithm for phi-node placement.
pub mod dominance_frontier;

/// SSA variable renaming — walks the dominator tree in pre-order to rename
/// promoted alloca uses to their reaching definitions, fill phi-node operands,
/// and build def-use chains. This is the final step of Phase 7.
pub mod ssa_builder;

/// Phase 9 phi-node elimination — converts SSA phi nodes back to copy
/// operations at predecessor block terminators for consumption by the
/// register allocator and backend code generator.
pub mod phi_eliminate;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use dominance_frontier::{compute_iterated_frontier, compute_phi_placements, DominanceFrontier};
pub use dominator_tree::DominatorTree;
pub use phi_eliminate::{eliminate_phis, verify_no_phis};
pub use ssa_builder::{rename_variables, AllocaSlot, SsaRenamer};

// ── Imports ─────────────────────────────────────────────────────────────────

use crate::common::fx_hash::{fx_hash_map, fx_hash_set, FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::Instruction;
use crate::ir::types::IrType;

// ── AllocaInfo — metadata for a single promotable alloca ────────────────────

/// Information about a single alloca instruction identified as eligible for
/// promotion to SSA register form during Phase 7.
///
/// Instances of `AllocaInfo` are produced by [`identify_promotable_allocas()`]
/// and consumed by [`promote_allocas_to_registers()`] to orchestrate
/// dominator tree computation, dominance frontier analysis, phi-node
/// placement, and SSA variable renaming.
///
/// # Fields
///
/// | Field        | Purpose                                               |
/// |--------------|-------------------------------------------------------|
/// | `alloca_id`  | The alloca instruction's result [`ValueId`] (pointer) |
/// | `ty`         | Allocated type (e.g., `I32`), not the pointer type    |
/// | `name`       | Optional variable name for debug info / IR dumps      |
/// | `def_blocks` | Blocks containing stores to this alloca (def sites)   |
/// | `use_blocks` | Blocks containing loads from this alloca (use sites)  |
#[derive(Clone, Debug)]
pub struct AllocaInfo {
    /// The result [`ValueId`] of the `Alloca` instruction. This is a pointer
    /// value that loads/stores reference. During SSA renaming, references to
    /// this pointer are replaced with direct SSA values.
    pub alloca_id: ValueId,

    /// The IR type of the allocated variable — **not** the pointer type.
    /// For example, if the alloca was `%p = alloca i32`, this field holds
    /// `IrType::I32`. This type is used when creating phi instructions and
    /// undef sentinels during SSA construction.
    pub ty: IrType,

    /// Optional human-readable name derived from the source-level variable
    /// declaration. Propagated to phi results and undef values for debug
    /// output and DWARF info generation.
    pub name: Option<String>,

    /// Set of basic blocks that contain `Store` instructions writing to
    /// this alloca. These are the "definition sites" in the SSA sense —
    /// dominance frontier computation uses them to determine where phi
    /// nodes must be placed.
    pub def_blocks: FxHashSet<BasicBlockId>,

    /// Set of basic blocks that contain `Load` instructions reading from
    /// this alloca. These are the "use sites" — blocks where the variable's
    /// value is consumed.
    pub use_blocks: FxHashSet<BasicBlockId>,
}

// ── promote_allocas_to_registers — main Phase 7 entry point ─────────────────

/// Promotes eligible alloca instructions to SSA virtual registers.
///
/// This is the **primary Phase 7 entry point** of the BCC compilation
/// pipeline. It transforms a function from lowered IR (where all local
/// variables are memory allocations accessed via load/store) to SSA form
/// (where eligible variables are direct values connected by phi nodes at
/// control-flow merge points).
///
/// # Algorithm
///
/// The promotion follows the classical Cytron et al. 1991 approach:
///
/// 1. **Identify promotable allocas:** Scan the entry block for alloca
///    instructions whose type is scalar and whose address is never taken.
///    Collect definition blocks (stores) and use blocks (loads) for each.
///
/// 2. **Compute dominator tree:** Build the Lengauer-Tarjan dominator tree
///    in O(n·α(n)) time.
///
/// 3. **Compute dominance frontiers:** For each block in the CFG, compute
///    its dominance frontier set using the walk-up algorithm.
///
/// 4. **Compute phi-node placements:** For each promotable alloca, compute
///    the iterated dominance frontier (IDF) of its definition blocks. The
///    IDF gives the exact set of blocks requiring phi nodes.
///
/// 5. **SSA renaming:** Walk the dominator tree in pre-order, inserting phi
///    instructions at placement blocks, renaming loads to their reaching
///    definitions, filling phi operands, and removing dead alloca/load/store
///    instructions.
///
/// # Arguments
///
/// * `func` — The IR function to transform. Modified in-place: eligible
///   alloca, load, and store instructions are removed and replaced by phi
///   nodes and direct SSA value references.
///
/// # Post-conditions
///
/// - All promoted allocas are removed from the entry block.
/// - All loads/stores to promoted allocas are replaced with SSA values.
/// - Phi nodes are inserted at dominance frontier merge points.
/// - Non-promoted allocas (aggregates, address-taken, volatile) remain
///   unchanged.
/// - The function's value registry may contain new values (phi results,
///   undef sentinels).
///
/// # Complexity
///
/// Dominated by dominator tree computation O(n·α(n)) and dominance frontier
/// computation O(|E|·depth(domtree)), where n is the number of blocks and
/// |E| is the CFG edge count. Practical performance is near-linear for the
/// structured control flow typical of C functions.
pub fn promote_allocas_to_registers(func: &mut IrFunction) {
    // ── Step 1: Identify promotable allocas ──────────────────────────────
    let alloca_infos = identify_promotable_allocas(func);
    if alloca_infos.is_empty() {
        // Nothing to promote — the function either has no allocas or all
        // of them are non-promotable (aggregates, address-taken, volatile).
        return;
    }

    // ── Step 2: Compute dominator tree ──────────────────────────────────
    let dom_tree = DominatorTree::compute(func);

    // ── Step 3: Compute dominance frontiers ─────────────────────────────
    let df = DominanceFrontier::compute(func, &dom_tree);

    // ── Step 4: Compute phi-node placements via iterated DF ─────────────
    // Collect the set of definition blocks for each promotable alloca.
    let def_blocks_per_alloca: Vec<FxHashSet<BasicBlockId>> = alloca_infos
        .iter()
        .map(|info| info.def_blocks.clone())
        .collect();

    // compute_phi_placements returns Vec<FxHashSet<BasicBlockId>> — a
    // parallel vector where entry [i] is the set of blocks requiring phi
    // nodes for alloca slot i.
    let phi_placement_sets = compute_phi_placements(&df, &def_blocks_per_alloca);

    // ── Step 5: Build phi placement map for SSA renamer ─────────────────
    // Convert from per-alloca sets to the per-block map format expected
    // by rename_variables: FxHashMap<BasicBlockId, Vec<(slot_index, IrType)>>
    let phi_placement_map = build_phi_placement_map(&alloca_infos, &phi_placement_sets);

    // ── Step 6: Create AllocaSlot descriptors with undef sentinels ──────
    // Each slot needs an "undef" sentinel value that represents reads of
    // the variable before any definition on a given path. This correctly
    // models C's undefined behavior for uninitialized local variables.
    let alloca_slots: Vec<AllocaSlot> = alloca_infos
        .iter()
        .map(|info| {
            let undef_name = info
                .name
                .as_ref()
                .map(|n| format!("{}.undef", n));
            let undef_value = func.new_value(info.ty.clone(), undef_name);
            AllocaSlot {
                alloca_value_id: info.alloca_id,
                ty: info.ty.clone(),
                name: info.name.clone(),
                undef_value,
            }
        })
        .collect();

    // ── Step 7: SSA renaming ────────────────────────────────────────────
    // rename_variables performs all remaining work:
    //   - Insert phi instructions at placement blocks
    //   - Walk dominator tree in pre-order to rename variables
    //   - Fill phi operands with reaching definitions
    //   - Replace load results with SSA reaching definitions
    //   - Remove dead alloca/load/store instructions
    //   - Build and return a def-use map (consumed by optimisation passes)
    let _def_use_map = rename_variables(func, &dom_tree, &alloca_slots, &phi_placement_map);
}

// ── identify_promotable_allocas — promotability analysis ────────────────────

/// Scans the function's entry block for alloca instructions eligible for
/// promotion to SSA register form.
///
/// An alloca is promotable when it meets all of the following criteria:
///
/// 1. **Scalar type**: The allocated type satisfies [`IrType::is_scalar()`]
///    — integer, floating-point, or pointer. Aggregate types (structs,
///    arrays) cannot be held in a single register and are not promoted.
///
/// 2. **Address not taken**: The alloca's result pointer appears only as the
///    `ptr` operand of `Load` and `Store` instructions. Any other use
///    (GEP, Call argument, BitCast, Phi, etc.) means the address may escape
///    to code that expects a valid memory location.
///
/// 3. **No volatile accesses**: Neither loads from nor stores to the alloca
///    carry the `volatile` flag. Volatile accesses have observable side
///    effects (e.g., memory-mapped I/O) that cannot be represented in pure
///    SSA form.
///
/// # Arguments
///
/// * `func` — The IR function to analyse (immutable; only reads instructions).
///
/// # Returns
///
/// A `Vec<AllocaInfo>` containing metadata for each promotable alloca found
/// in the entry block, ordered by their position in the instruction stream.
/// The index in this vector becomes the canonical "alloca slot index" used
/// throughout the remainder of the mem2reg pipeline.
///
/// Returns an empty vector if no allocas are promotable.
pub fn identify_promotable_allocas(func: &IrFunction) -> Vec<AllocaInfo> {
    let entry_block = func.entry_block();
    let mut promotable: Vec<AllocaInfo> = Vec::new();

    for inst in entry_block.instructions() {
        // We only care about Alloca instructions in the entry block.
        // By convention, Phase 6 lowering places all allocas here.
        let (alloca_id, alloca_ty) = match inst {
            Instruction::Alloca { result, ty, .. } => (*result, ty.clone()),
            _ => continue,
        };

        // ── Criterion 1: Scalar type ────────────────────────────────────
        // Only scalar types (integers, floats, pointers) are promotable.
        // Aggregates (structs, arrays) require memory representation.
        if !alloca_ty.is_scalar() {
            continue;
        }

        // ── Criterion 2: Address not taken ──────────────────────────────
        // The alloca pointer must only be used as the target of
        // Load { ptr } or Store { ptr } — no other context.
        if is_address_taken(func, alloca_id) {
            continue;
        }

        // ── Criterion 3: No volatile accesses ───────────────────────────
        // Volatile loads/stores have observable side effects that would be
        // lost if the alloca were promoted to a register.
        if has_volatile_access(func, alloca_id) {
            continue;
        }

        // ── Collect definition and use sites ────────────────────────────
        let (def_blocks, use_blocks) = collect_def_use_blocks(func, alloca_id);

        // ── Extract variable name from value registry ───────────────────
        let name = func
            .get_value_info(alloca_id)
            .and_then(|vi| vi.name.clone());

        promotable.push(AllocaInfo {
            alloca_id,
            ty: alloca_ty,
            name,
            def_blocks,
            use_blocks,
        });
    }

    promotable
}

// ── is_address_taken — escape analysis for a single alloca ──────────────────

/// Determines whether the alloca pointer's address "escapes" — i.e., is used
/// in any context other than being the pointer operand of a Load or Store.
///
/// If the address escapes (through a GEP, Call, BitCast, Phi, or being stored
/// as a *value*), the alloca cannot be safely promoted because external code
/// or data structures may hold a pointer to the memory location.
///
/// # Arguments
///
/// * `func`      — The IR function containing the alloca.
/// * `alloca_id` — The result [`ValueId`] of the alloca instruction to check.
///
/// # Returns
///
/// `true` if the alloca's address is taken or escapes, `false` if all uses
/// are safe Load/Store accesses through the pointer operand.
fn is_address_taken(func: &IrFunction, alloca_id: ValueId) -> bool {
    for block in func.blocks() {
        for inst in block.instructions() {
            // Skip instructions that don't reference this alloca at all.
            // The uses() method returns all ValueIds read by the instruction.
            if !inst.uses().contains(&alloca_id) {
                continue;
            }

            match inst {
                // A load FROM the alloca (ptr position) is safe — the alloca
                // address is being dereferenced, not captured.
                Instruction::Load { ptr, .. } if *ptr == alloca_id => {
                    // Safe: reading the alloca's stored value.
                }

                // A store TO the alloca (ptr position) is safe — writing a
                // new value into the alloca's memory. However, if the alloca's
                // address is being stored as the VALUE (i.e., the pointer
                // itself is being written to some other memory location), the
                // address escapes.
                Instruction::Store { ptr, value, .. } if *ptr == alloca_id => {
                    // The alloca is the store target — check if the value being
                    // stored is also the alloca pointer (self-referential).
                    // In practice this is degenerate, but handle it for correctness.
                    if *value == alloca_id {
                        return true;
                    }
                    // Otherwise safe: storing some other value into the alloca.
                }

                // Any other use means the address escapes:
                //   - Store { value: alloca_id, ptr: other } → pointer stored to memory
                //   - GetElementPtr { base: alloca_id, .. } → sub-element addressing
                //   - Call { args: [.., alloca_id, ..], .. } → passed to function
                //   - BitCast { value: alloca_id, .. } → type-punned pointer
                //   - Phi { incoming: [.., (alloca_id, _), ..], .. } → merged pointer
                //   - IntToPtr, PtrToInt, etc. → pointer escapes
                _ => {
                    return true;
                }
            }
        }
    }
    false
}

// ── has_volatile_access — volatile load/store check ─────────────────────────

/// Checks whether any load from or store to the alloca has the `volatile`
/// flag set.
///
/// Volatile accesses have externally observable ordering and side-effect
/// semantics (e.g., memory-mapped I/O). Promoting a volatile alloca to an
/// SSA register would silently drop these guarantees, producing incorrect
/// code. Volatile allocas must therefore remain in memory.
///
/// # Arguments
///
/// * `func`      — The IR function containing the alloca.
/// * `alloca_id` — The result [`ValueId`] of the alloca instruction to check.
///
/// # Returns
///
/// `true` if any load from or store to the alloca is volatile.
fn has_volatile_access(func: &IrFunction, alloca_id: ValueId) -> bool {
    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                Instruction::Load {
                    ptr, volatile: true, ..
                } if *ptr == alloca_id => {
                    return true;
                }
                Instruction::Store {
                    ptr, volatile: true, ..
                } if *ptr == alloca_id => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

// ── collect_def_use_blocks — definition and use site gathering ──────────────

/// Scans all blocks in the function to identify which blocks contain stores
/// to (definitions of) and loads from (uses of) the specified alloca.
///
/// The returned sets feed directly into the dominance-frontier-based
/// phi-node placement algorithm: phi nodes must be placed at the iterated
/// dominance frontier of the definition blocks so that all uses see a
/// valid reaching definition.
///
/// # Arguments
///
/// * `func`      — The IR function to scan.
/// * `alloca_id` — The alloca instruction's result [`ValueId`].
///
/// # Returns
///
/// A tuple `(def_blocks, use_blocks)`:
/// - `def_blocks`: Blocks containing at least one `Store` with `ptr == alloca_id`.
/// - `use_blocks`: Blocks containing at least one `Load` with `ptr == alloca_id`.
fn collect_def_use_blocks(
    func: &IrFunction,
    alloca_id: ValueId,
) -> (FxHashSet<BasicBlockId>, FxHashSet<BasicBlockId>) {
    let mut def_blocks: FxHashSet<BasicBlockId> = fx_hash_set();
    let mut use_blocks: FxHashSet<BasicBlockId> = fx_hash_set();

    for block in func.blocks() {
        for inst in block.instructions() {
            match inst {
                Instruction::Store { ptr, .. } if *ptr == alloca_id => {
                    def_blocks.insert(block.id);
                }
                Instruction::Load { ptr, .. } if *ptr == alloca_id => {
                    use_blocks.insert(block.id);
                }
                _ => {}
            }
        }
    }

    (def_blocks, use_blocks)
}

// ── build_phi_placement_map — convert per-alloca sets to per-block map ──────

/// Converts phi-node placement information from the per-alloca representation
/// (output of [`compute_phi_placements()`]) into the per-block representation
/// consumed by [`rename_variables()`].
///
/// The dominance frontier pass produces `Vec<FxHashSet<BasicBlockId>>` — for
/// each alloca slot index, the set of blocks where phi nodes are needed. The
/// SSA renamer expects `FxHashMap<BasicBlockId, Vec<(usize, IrType)>>` — for
/// each block, the list of `(alloca_slot_index, allocated_type)` entries
/// describing which phi nodes to create there.
///
/// This transposition is a simple O(total_phi_placements) operation.
///
/// # Arguments
///
/// * `alloca_infos`       — Metadata for each promotable alloca (provides types).
/// * `phi_placement_sets` — Per-alloca phi block sets from dominance frontier computation.
///
/// # Returns
///
/// A map from block ID to the list of `(slot_index, ir_type)` entries for
/// phi nodes that must be created in that block.
fn build_phi_placement_map(
    alloca_infos: &[AllocaInfo],
    phi_placement_sets: &[FxHashSet<BasicBlockId>],
) -> FxHashMap<BasicBlockId, Vec<(usize, IrType)>> {
    let mut map: FxHashMap<BasicBlockId, Vec<(usize, IrType)>> = fx_hash_map();

    for (alloca_idx, phi_blocks) in phi_placement_sets.iter().enumerate() {
        let alloca_ty = &alloca_infos[alloca_idx].ty;
        for &block_id in phi_blocks {
            map.entry(block_id)
                .or_insert_with(Vec::new)
                .push((alloca_idx, alloca_ty.clone()));
        }
    }

    map
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::basic_block::{BasicBlock, BasicBlockId};
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::Instruction;
    use crate::ir::types::IrType;

    /// Helper: creates a minimal function with a single entry block containing
    /// one scalar alloca, a store, and a load — the simplest promotable case.
    fn build_single_alloca_func() -> IrFunction {
        let mut func = IrFunction::new("test_func".into(), IrType::Void, vec![]);
        let entry_id = func.entry_block_id;

        // Create alloca: %0 = alloca i32
        let alloca_result = func.new_value(IrType::Ptr, Some("x.addr".into()));
        let alloca_inst = Instruction::Alloca {
            result: alloca_result,
            ty: IrType::I32,
            alignment: 4,
        };

        // Create a constant value to store
        let const_val = func.new_value(IrType::I32, Some("const".into()));

        // Store to alloca: store %const_val, %alloca_result
        let store_inst = Instruction::Store {
            value: const_val,
            ptr: alloca_result,
            volatile: false,
        };

        // Load from alloca: %load_result = load i32, %alloca_result
        let load_result = func.new_value(IrType::I32, Some("x".into()));
        let load_inst = Instruction::Load {
            result: load_result,
            ptr: alloca_result,
            ty: IrType::I32,
            volatile: false,
        };

        // Return
        let ret_inst = Instruction::Return { value: Some(load_result) };

        // Build the entry block
        let entry = func.get_block_mut(entry_id);
        entry.add_instruction(alloca_inst);
        entry.add_instruction(store_inst);
        entry.add_instruction(load_inst);
        entry.set_terminator(ret_inst);

        func
    }

    #[test]
    fn test_identify_scalar_alloca_is_promotable() {
        let func = build_single_alloca_func();
        let infos = identify_promotable_allocas(&func);

        assert_eq!(infos.len(), 1, "Expected exactly one promotable alloca");
        assert!(infos[0].ty.is_scalar(), "Alloca type should be scalar (i32)");
        assert!(!infos[0].def_blocks.is_empty(), "Should have at least one def block");
        assert!(!infos[0].use_blocks.is_empty(), "Should have at least one use block");
    }

    #[test]
    fn test_aggregate_alloca_not_promotable() {
        let mut func = IrFunction::new("test_agg".into(), IrType::Void, vec![]);
        let entry_id = func.entry_block_id;

        // Aggregate alloca: %0 = alloca {i32, i32}
        let agg_type = IrType::Struct(vec![IrType::I32, IrType::I32]);
        let alloca_result = func.new_value(IrType::Ptr, Some("pair.addr".into()));
        let alloca_inst = Instruction::Alloca {
            result: alloca_result,
            ty: agg_type,
            alignment: 4,
        };

        let ret_inst = Instruction::Return { value: None };

        let entry = func.get_block_mut(entry_id);
        entry.add_instruction(alloca_inst);
        entry.set_terminator(ret_inst);

        let infos = identify_promotable_allocas(&func);
        assert!(infos.is_empty(), "Aggregate alloca should not be promotable");
    }

    #[test]
    fn test_address_taken_not_promotable() {
        let mut func = IrFunction::new("test_escaped".into(), IrType::Void, vec![]);
        let entry_id = func.entry_block_id;

        // Scalar alloca: %0 = alloca i32
        let alloca_result = func.new_value(IrType::Ptr, Some("x.addr".into()));
        let alloca_inst = Instruction::Alloca {
            result: alloca_result,
            ty: IrType::I32,
            alignment: 4,
        };

        // GEP from the alloca — this takes the address
        let gep_result = func.new_value(IrType::Ptr, Some("gep".into()));
        let gep_idx = func.new_value(IrType::I32, None);
        let gep_inst = Instruction::GetElementPtr {
            result: gep_result,
            base: alloca_result,
            indices: vec![gep_idx],
            ty: IrType::I32,
            in_bounds: true,
        };

        let ret_inst = Instruction::Return { value: None };

        let entry = func.get_block_mut(entry_id);
        entry.add_instruction(alloca_inst);
        entry.add_instruction(gep_inst);
        entry.set_terminator(ret_inst);

        let infos = identify_promotable_allocas(&func);
        assert!(infos.is_empty(), "Address-taken alloca should not be promotable");
    }

    #[test]
    fn test_volatile_access_not_promotable() {
        let mut func = IrFunction::new("test_volatile".into(), IrType::Void, vec![]);
        let entry_id = func.entry_block_id;

        let alloca_result = func.new_value(IrType::Ptr, Some("v.addr".into()));
        let alloca_inst = Instruction::Alloca {
            result: alloca_result,
            ty: IrType::I32,
            alignment: 4,
        };

        // Volatile load
        let load_result = func.new_value(IrType::I32, Some("v".into()));
        let load_inst = Instruction::Load {
            result: load_result,
            ptr: alloca_result,
            ty: IrType::I32,
            volatile: true,
        };

        let ret_inst = Instruction::Return { value: Some(load_result) };

        let entry = func.get_block_mut(entry_id);
        entry.add_instruction(alloca_inst);
        entry.add_instruction(load_inst);
        entry.set_terminator(ret_inst);

        let infos = identify_promotable_allocas(&func);
        assert!(infos.is_empty(), "Volatile alloca should not be promotable");
    }

    #[test]
    fn test_no_allocas_returns_empty() {
        let mut func = IrFunction::new("test_empty".into(), IrType::Void, vec![]);
        let entry_id = func.entry_block_id;

        let ret_inst = Instruction::Return { value: None };
        func.get_block_mut(entry_id).set_terminator(ret_inst);

        let infos = identify_promotable_allocas(&func);
        assert!(infos.is_empty(), "Function with no allocas should have empty list");
    }

    #[test]
    fn test_promote_on_trivial_function() {
        // Ensure promote_allocas_to_registers doesn't panic on a simple function.
        let mut func = build_single_alloca_func();
        promote_allocas_to_registers(&mut func);
        // After promotion, the entry block should have no Alloca instructions
        // for promoted variables (they're removed by the SSA renamer).
    }

    #[test]
    fn test_build_phi_placement_map_empty() {
        let alloca_infos: Vec<AllocaInfo> = vec![];
        let phi_sets: Vec<FxHashSet<BasicBlockId>> = vec![];
        let map = build_phi_placement_map(&alloca_infos, &phi_sets);
        assert!(map.is_empty(), "Empty inputs should produce empty map");
    }

    #[test]
    fn test_build_phi_placement_map_single_alloca() {
        let mut def_blocks = fx_hash_set();
        def_blocks.insert(BasicBlockId(1));
        let info = AllocaInfo {
            alloca_id: ValueId(0),
            ty: IrType::I32,
            name: Some("x".into()),
            def_blocks,
            use_blocks: fx_hash_set(),
        };

        let mut phi_set = fx_hash_set();
        phi_set.insert(BasicBlockId(2));
        phi_set.insert(BasicBlockId(3));

        let map = build_phi_placement_map(&[info], &[phi_set]);

        assert!(map.contains_key(&BasicBlockId(2)));
        assert!(map.contains_key(&BasicBlockId(3)));
        assert_eq!(map[&BasicBlockId(2)].len(), 1);
        assert_eq!(map[&BasicBlockId(2)][0].0, 0); // slot index
    }

    #[test]
    fn test_collect_def_use_blocks_basic() {
        let func = build_single_alloca_func();
        let entry_id = func.entry_block_id;

        // The alloca is ValueId(0) in our helper
        let alloca_id = ValueId(0);
        let (defs, uses) = collect_def_use_blocks(&func, alloca_id);

        assert!(defs.contains(&entry_id), "Entry block should be a def site");
        assert!(uses.contains(&entry_id), "Entry block should be a use site");
    }
}
