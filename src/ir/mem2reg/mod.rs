//! # Mem2Reg — SSA Construction via Alloca Promotion
//!
//! This module implements the "alloca-then-promote" SSA construction strategy,
//! a non-negotiable architectural mandate that mirrors the LLVM approach.
//!
//! ## Phase 7: SSA Construction
//!
//! After IR lowering (Phase 6) places all local variables as `alloca`
//! instructions in the entry basic block, this module promotes eligible
//! allocas to SSA virtual registers:
//!
//! 1. Identify **promotable allocas** (scalar, non-address-taken, non-volatile)
//! 2. Compute the **dominator tree** using the Lengauer-Tarjan algorithm
//! 3. Compute **dominance frontiers** for phi-node placement
//! 4. Compute the **iterated dominance frontier** (IDF) of each variable's
//!    definition sites to determine exactly which blocks need phi nodes
//! 5. Insert **phi nodes** at IDF locations
//! 6. Perform **SSA renaming** — replace loads/stores with direct value references
//!    via a dominator-tree walk with reaching-definition stacks
//! 7. **Clean up** — remove the now-dead alloca, load, and store instructions
//!
//! After this pass, the function's IR is in proper SSA form for consumption
//! by optimization passes (Phase 8).
//!
//! ## Phase 9: Phi Elimination
//!
//! After optimization passes (Phase 8), phi nodes are converted to parallel
//! copies at predecessor block ends, then sequentialized for register
//! allocation consumption.  The [`eliminate_phi_nodes`] function is re-exported
//! here for convenient access via `crate::ir::mem2reg::eliminate_phi_nodes`.
//!
//! ## Architecture
//!
//! - [`dominator_tree`] — Lengauer-Tarjan dominator tree computation
//! - [`dominance_frontier`] — Iterated dominance frontier for phi placement
//! - [`ssa_builder`] — SSA renaming with reaching-definition stacks
//! - [`phi_eliminate`] — Phase 9 phi-node elimination
//!
//! ## Promotability Criteria
//!
//! An alloca is eligible for SSA promotion if and only if:
//! - It allocates a **scalar type** (integer, float, or pointer) — NOT
//!   aggregates or arrays
//! - Its address is **never taken** — it is only used as the pointer operand
//!   of Load and Store instructions, never passed to calls, GEP, or other
//!   instructions
//! - It is never used in a **volatile** load or store
//! - It resides in the **entry block** (standard alloca placement from Phase 6)
//!
//! Allocas that fail any criterion remain as memory operations and are not
//! promoted.
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::` internal modules and the Rust
//! standard library — no external crates are used.  All hash maps and sets
//! use [`FxHashMap`](crate::common::fx_hash::FxHashMap) and
//! [`FxHashSet`](crate::common::fx_hash::FxHashSet) as mandated by the
//! project's zero-dependency architecture.

// ============================================================================
// Submodule declarations — all public for crate-wide access
// ============================================================================

pub mod dominance_frontier;
pub mod dominator_tree;
pub mod phi_eliminate;
pub mod ssa_builder;

// ============================================================================
// Re-export Phase 9 entry point for convenient crate-wide access
// ============================================================================

/// Re-export of [`phi_eliminate::eliminate_phi_nodes`] for convenient access
/// via `crate::ir::mem2reg::eliminate_phi_nodes`.
pub use phi_eliminate::eliminate_phi_nodes;

// ============================================================================
// Imports — crate-internal only, zero external crates
// ============================================================================

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlock;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

use self::dominance_frontier::{compute_dominance_frontiers, compute_iterated_dominance_frontier};
use self::dominator_tree::DominatorTree;
use self::ssa_builder::{cleanup_promoted_instructions, insert_phi_nodes, SsaRenamer};

// ============================================================================
// Alloca Promotability Analysis
// ============================================================================

/// Determine if an alloca instruction is eligible for SSA promotion.
///
/// An alloca is promotable if **all** of the following hold:
///
/// - It allocates a **scalar type** (integer, float, or pointer) — arrays
///   and structs are not eligible because they cannot be represented as
///   a single SSA value.
/// - Its address is **never taken** — the alloca result value is used
///   exclusively as the `ptr` operand of `Load` and `Store` instructions.
///   If the alloca value appears as a call argument, a GEP base, the
///   *value* operand of a store, or in any other instruction, the address
///   has escaped and promotion is unsafe.
/// - It is never used in a **volatile** `Load` or `Store` — volatile
///   accesses must remain as memory operations to preserve their
///   ordering and side-effect semantics.
///
/// # Parameters
///
/// - `func`: The function containing the alloca.
/// - `alloca_value`: The [`Value`] produced by the `Alloca` instruction
///   (i.e., the pointer to the allocated memory).
///
/// # Returns
///
/// `true` if the alloca can be safely promoted to an SSA virtual register,
/// `false` otherwise.
///
/// # Algorithm
///
/// 1. Locate the `Alloca` instruction in the entry block and extract its
///    type.  If the type is not scalar, return `false`.
/// 2. Scan **every** instruction in **every** block of the function.
/// 3. For each instruction that uses `alloca_value` as an operand:
///    - `Load { ptr }` where `ptr == alloca_value`: valid use; check
///      volatile flag.
///    - `Store { ptr }` where `ptr == alloca_value` AND `value != alloca_value`:
///      valid use; check volatile flag.
///    - Any other use (including `Store { value == alloca_value }`,
///      which means the alloca address is being stored to memory):
///      address taken — return `false`.
pub fn is_promotable_alloca(func: &IrFunction, alloca_value: Value) -> bool {
    // Step 1: Find the alloca's type from the entry block.
    let alloca_type = match find_alloca_type(func, alloca_value) {
        Some(ty) => ty,
        None => return false, // Not an alloca or not found in entry block
    };

    // Step 2: The alloca must allocate a scalar type.
    if !alloca_type.is_scalar() {
        return false;
    }

    // Step 3: Scan all instructions in all blocks for uses of alloca_value.
    for block in func.blocks() {
        for inst in block.instructions() {
            // Skip the defining alloca instruction itself.
            if let Instruction::Alloca { result, .. } = inst {
                if *result == alloca_value {
                    continue;
                }
            }

            // Check each instruction variant that could reference alloca_value.
            match inst {
                Instruction::Load {
                    ptr, volatile: vol, ..
                } => {
                    if *ptr == alloca_value {
                        // Valid use as load pointer — but volatile loads
                        // prevent promotion (must remain as memory ops).
                        if *vol {
                            return false;
                        }
                    }
                }
                Instruction::Store {
                    value,
                    ptr,
                    volatile: vol,
                    ..
                } => {
                    if *ptr == alloca_value {
                        // Valid use as store pointer — check volatile.
                        if *vol {
                            return false;
                        }
                    }
                    if *value == alloca_value {
                        // The alloca's address appears as the stored VALUE.
                        // This means the pointer is being written to memory,
                        // allowing it to escape to unknown code paths.
                        return false;
                    }
                }
                _ => {
                    // For any other instruction type (Call, GEP, BinOp, etc.),
                    // check if alloca_value appears in its operand list.
                    // Any appearance means the address has been taken.
                    if instruction_uses_value(inst, alloca_value) {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Collect all promotable alloca instructions from a function's entry block.
///
/// Iterates the entry block (`blocks[0]`), identifies each `Alloca`
/// instruction, checks promotability via [`is_promotable_alloca`], and
/// collects the promotable ones into a map from alloca [`Value`] to
/// allocated [`IrType`].
///
/// Only allocas in the entry block are considered — this matches the
/// "alloca-then-promote" convention where Phase 6 (IR lowering) places
/// all local variable allocas in the function entry block.
///
/// # Parameters
///
/// - `func`: The function whose allocas are to be analyzed.
///
/// # Returns
///
/// A [`FxHashMap`] mapping each promotable alloca's result [`Value`] to
/// its allocated [`IrType`].  Returns an empty map if no allocas are
/// promotable or if the function has no blocks.
pub fn collect_promotable_allocas(func: &IrFunction) -> FxHashMap<Value, IrType> {
    let mut promotable: FxHashMap<Value, IrType> = FxHashMap::default();

    if func.block_count() == 0 {
        return promotable;
    }

    // Only scan the entry block — allocas should be placed here by Phase 6.
    let entry: &BasicBlock = func.entry_block();
    for inst in entry.instructions() {
        if !inst.is_alloca() {
            continue;
        }

        if let Instruction::Alloca { result, ty, .. } = inst {
            // Check full promotability (scalar type, not address-taken,
            // not volatile).
            if is_promotable_alloca(func, *result) {
                promotable.insert(*result, ty.clone());
            }
        }
    }

    promotable
}

// ============================================================================
// Helper Functions — Private
// ============================================================================

/// Find the [`IrType`] of an `Alloca` instruction by its result value.
///
/// Searches the entry block for the `Alloca` instruction that produces
/// `alloca_value` and returns a clone of its allocated type.
///
/// Returns `None` if:
/// - The function has no blocks.
/// - No `Alloca` instruction in the entry block produces `alloca_value`.
fn find_alloca_type(func: &IrFunction, alloca_value: Value) -> Option<IrType> {
    if func.block_count() == 0 {
        return None;
    }
    let entry: &BasicBlock = func.entry_block();
    for inst in entry.instructions() {
        if let Instruction::Alloca { result, ty, .. } = inst {
            if *result == alloca_value {
                return Some(ty.clone());
            }
        }
    }
    None
}

/// Check if an instruction uses a specific [`Value`] in any of its operands.
///
/// This is a helper for address-taken detection in [`is_promotable_alloca`].
/// It examines the instruction's operand list (via [`Instruction::operands`])
/// for any occurrence of `target`.
///
/// # Returns
///
/// `true` if `target` appears in the instruction's operand list.
#[inline]
fn instruction_uses_value(inst: &Instruction, target: Value) -> bool {
    inst.operands().contains(&target)
}

/// Find all basic blocks that contain a `Store` to the given alloca.
///
/// These are the "definition sites" for the promoted variable — blocks
/// where a new value is assigned.  Used to seed the iterated dominance
/// frontier computation for phi-node placement.
///
/// # Parameters
///
/// - `func`: The function to scan.
/// - `alloca_val`: The alloca [`Value`] whose stores we're looking for.
///
/// # Returns
///
/// A [`FxHashSet`] of block indices containing at least one `Store`
/// with `ptr == alloca_val`.
fn find_def_blocks(func: &IrFunction, alloca_val: Value) -> FxHashSet<usize> {
    let mut def_blocks: FxHashSet<usize> = FxHashSet::default();

    for (block_idx, block) in func.blocks().iter().enumerate() {
        for inst in block.instructions() {
            if let Instruction::Store { ptr, .. } = inst {
                if *ptr == alloca_val {
                    def_blocks.insert(block_idx);
                    // One definition per block is sufficient for IDF computation.
                    break;
                }
            }
        }
    }

    def_blocks
}

/// Find all basic blocks that contain a `Load` from the given alloca.
///
/// These are the "use sites" for the promoted variable — blocks where
/// the variable's value is read.  While the standard IDF-based phi
/// placement algorithm does not strictly require use-site information,
/// this data is available for future pruned-SSA optimizations (avoiding
/// phi placement at blocks where the variable is never live-in).
///
/// # Parameters
///
/// - `func`: The function to scan.
/// - `alloca_val`: The alloca [`Value`] whose loads we're looking for.
///
/// # Returns
///
/// A [`FxHashSet`] of block indices containing at least one `Load`
/// with `ptr == alloca_val`.
#[allow(dead_code)] // Retained for future pruned-SSA optimization passes.
fn find_use_blocks(func: &IrFunction, alloca_val: Value) -> FxHashSet<usize> {
    let mut use_blocks: FxHashSet<usize> = FxHashSet::default();

    for (block_idx, block) in func.blocks().iter().enumerate() {
        for inst in block.instructions() {
            if let Instruction::Load { ptr, .. } = inst {
                if *ptr == alloca_val {
                    use_blocks.insert(block_idx);
                    // One use per block is sufficient.
                    break;
                }
            }
        }
    }

    use_blocks
}

/// Remove promoted alloca, load, and store instructions after SSA promotion.
///
/// Delegates to [`ssa_builder::cleanup_promoted_instructions`] which
/// retains all non-promoted instructions and removes:
/// - `Alloca` instructions whose result is in the promotable set
/// - `Load` instructions whose pointer operand is a promoted alloca
/// - `Store` instructions whose pointer operand is a promoted alloca
///
/// # Parameters
///
/// - `func`: The function to clean up (mutated in place).
/// - `promotable`: The same map of promotable allocas used during promotion.
fn remove_promoted_instructions(func: &mut IrFunction, promotable: &FxHashMap<Value, IrType>) {
    cleanup_promoted_instructions(func, promotable);
}

// ============================================================================
// Single-Block Optimization
// ============================================================================

/// Promote allocas in a function with exactly one basic block.
///
/// When a function has no control flow (only one block), phi nodes are
/// never needed.  This special case is handled by a linear scan that
/// tracks the last stored value for each promoted alloca and replaces
/// loads with the tracked reaching definition.
///
/// This is significantly faster than the full dominator-tree-based
/// algorithm for the common case of trivial helper functions and
/// compiler-generated thunks.
///
/// # Algorithm
///
/// 1. Walk the entry block's instructions in order.
/// 2. For each `Store` to a promoted alloca, record the stored value
///    (resolved through prior load replacements) as the current definition.
/// 3. For each `Load` from a promoted alloca, record a replacement mapping
///    from the load result to the current definition (or `Value::UNDEF` if
///    no store has been seen — undefined behavior in C, read-before-write).
/// 4. Apply all collected replacements to every instruction's operands.
/// 5. Remove the dead alloca/load/store instructions.
fn promote_single_block(func: &mut IrFunction, promotable: &FxHashMap<Value, IrType>) {
    // Phase 1: Scan the entry block and collect replacement mappings.
    // We build two maps:
    // - current_val: alloca_value → most recent stored Value
    // - replacements: load_result → reaching definition Value
    let replacements: FxHashMap<Value, Value>;
    {
        let entry: &BasicBlock = func.entry_block();
        let mut current_val: FxHashMap<Value, Value> = FxHashMap::default();
        let mut repl: FxHashMap<Value, Value> = FxHashMap::default();

        for inst in entry.instructions() {
            match inst {
                Instruction::Store { value, ptr, .. } if promotable.contains_key(ptr) => {
                    // Resolve the stored value through prior replacements
                    // (handles the case where the stored value is itself
                    // the result of a promoted load from earlier in the block).
                    let actual = repl.get(value).copied().unwrap_or(*value);
                    current_val.insert(*ptr, actual);
                }
                Instruction::Load { result, ptr, .. } if promotable.contains_key(ptr) => {
                    // Replace this load's result with the current reaching
                    // definition for the alloca.  If no store has been seen
                    // yet, the variable is read before written (UB in C) —
                    // use Value::UNDEF as the sentinel.
                    let reaching = current_val.get(ptr).copied().unwrap_or(Value::UNDEF);
                    repl.insert(*result, reaching);
                }
                _ => {}
            }
        }

        replacements = repl;
    } // immutable borrow on func released here

    // Phase 2: Apply replacements to all remaining instruction operands.
    if !replacements.is_empty() {
        let entry = func.entry_block_mut();
        for inst in entry.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if let Some(&replacement) = replacements.get(operand) {
                    *operand = replacement;
                }
            }
        }
    }

    // Phase 3: Remove the now-dead promoted alloca/load/store instructions.
    remove_promoted_instructions(func, promotable);
}

// ============================================================================
// Phase 7 Driver — promote_allocas_to_ssa
// ============================================================================

/// Phase 7: Promote eligible alloca instructions to SSA form.
///
/// This is the main entry point for the mem2reg pass on a single function.
/// It implements the full "alloca-then-promote" SSA construction strategy:
///
/// 1. **Collect promotable allocas** from the entry block — scalar type,
///    non-address-taken, non-volatile.
/// 2. **Early exit** if no allocas are promotable (no work to do).
/// 3. **Single-block optimization**: if the function has only one basic block,
///    use the fast linear-scan algorithm (no phi nodes needed).
/// 4. **Build the dominator tree** using the Lengauer-Tarjan algorithm —
///    O(n × α(n)) for efficiency on large kernel functions.
/// 5. **Compute dominance frontiers** for phi-node placement.
/// 6. **Compute iterated dominance frontier (IDF)** for each promoted variable
///    based on its definition sites (blocks containing stores).
/// 7. **Insert phi nodes** at IDF locations — one phi per variable per block.
/// 8. **Perform SSA renaming** — walk the dominator tree, replace loads with
///    reaching definitions, push store values as new definitions, fill phi
///    operands in successor blocks.
/// 9. **Clean up** — remove dead alloca/load/store instructions.
///
/// After this pass, the function's IR is in proper SSA form for optimization
/// passes (Phase 8).
///
/// # Parameters
///
/// - `func`: The IR function to transform (mutated in place).
///
/// # Edge Cases
///
/// - Functions with zero blocks (declarations): returns immediately.
/// - Functions with no promotable allocas: returns immediately.
/// - Functions with one basic block: uses the fast single-block path
///   (no dominator tree or phi nodes needed).
/// - Functions where all allocas are aggregate/volatile/address-taken:
///   those allocas remain as memory operations.
pub fn promote_allocas_to_ssa(func: &mut IrFunction) {
    // Guard: empty function (declaration or malformed).
    if func.block_count() == 0 {
        return;
    }

    // Step 1: Identify all promotable allocas in the entry block.
    let promotable = collect_promotable_allocas(func);

    // Step 2: Early exit — nothing to promote.
    if promotable.is_empty() {
        return;
    }

    // Step 3: Single-block optimization — no control flow, no phi nodes.
    if func.block_count() == 1 {
        promote_single_block(func, &promotable);
        return;
    }

    // Step 4: Build the dominator tree (Lengauer-Tarjan algorithm).
    let dom_tree = DominatorTree::build(func);

    // Step 5: Compute dominance frontiers for all blocks.
    let df = compute_dominance_frontiers(func, &dom_tree);

    // Step 6: For each promotable alloca, compute the iterated dominance
    // frontier of its definition sites.  The IDF gives exactly the blocks
    // where phi nodes must be inserted for that variable.
    let mut phi_locations: FxHashSet<(usize, Value)> = FxHashSet::default();

    for (&alloca_val, _alloca_ty) in promotable.iter() {
        // Find all blocks containing a store to this alloca (definition sites).
        let def_blocks = find_def_blocks(func, alloca_val);

        // Also consider the entry block as a definition site if there are
        // use blocks — even without an explicit store, the alloca's initial
        // undefined value acts as a definition for reads that precede any store.
        // The IDF computation handles this correctly because if use blocks
        // exist outside the entry, the entry block's "implicit undef definition"
        // creates demand for phi nodes at merge points.

        // Compute the IDF — blocks where phi nodes for this variable are needed.
        let idf = compute_iterated_dominance_frontier(&def_blocks, &df);

        for &block_idx in idf.iter() {
            phi_locations.insert((block_idx, alloca_val));
        }
    }

    // Step 7: Insert phi nodes at IDF locations and get the phi result map.
    let next_value = func.value_count;
    let (phi_map, updated_next_value) =
        insert_phi_nodes(func, &phi_locations, &promotable, next_value);

    // Step 8: Perform SSA renaming — walk the dominator tree, replacing
    // loads/stores of promoted allocas with SSA value references.
    let mut renamer = SsaRenamer::new(promotable.clone(), phi_map, updated_next_value);
    renamer.rename(func, &dom_tree);

    // Step 9: Remove the now-dead promoted alloca/load/store instructions.
    remove_promoted_instructions(func, &promotable);
}

// ============================================================================
// Module-Level Entry Point — run_mem2reg
// ============================================================================

/// Run the mem2reg pass on all functions in the module.
///
/// This is the Phase 7 entry point called by the compilation pipeline
/// after IR lowering (Phase 6) and before optimization passes (Phase 8).
///
/// Iterates all function definitions in the module and applies
/// [`promote_allocas_to_ssa`] to each one.  Function declarations
/// (without bodies) are skipped — they have no basic blocks and
/// no allocas to promote.
///
/// # Parameters
///
/// - `module`: The IR module containing all functions to transform.
///
/// # Example
///
/// ```ignore
/// use bcc::ir::mem2reg::run_mem2reg;
///
/// // After Phase 6 (IR lowering):
/// run_mem2reg(&mut module);
/// // Now the module's functions are in SSA form, ready for Phase 8.
/// ```
pub fn run_mem2reg(module: &mut IrModule) {
    for func in module.functions.iter_mut() {
        // Only process function definitions — declarations have no
        // basic blocks and no allocas to promote.
        if func.is_definition {
            promote_allocas_to_ssa(func);
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::basic_block::BasicBlock;
    use crate::ir::function::{FunctionParam, IrFunction};
    use crate::ir::instructions::{BlockId, Instruction, Value};
    use crate::ir::types::IrType;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Create a simple single-block function:
    /// ```text
    /// int f(int param) {
    ///     int x = param;
    ///     return x;
    /// }
    /// ```
    /// IR:
    /// ```text
    /// entry:
    ///   %1 = alloca i32        ; x
    ///   store i32 %0, ptr %1   ; x = param
    ///   %2 = load i32, ptr %1  ; read x
    ///   ret i32 %2
    /// ```
    fn make_simple_function() -> IrFunction {
        let params = vec![FunctionParam::new(
            "param".to_string(),
            IrType::I32,
            Value(0),
        )];
        let mut func = IrFunction::new("f".to_string(), params, IrType::I32);
        func.value_count = 1; // param = Value(0)

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

        func.value_count = 3;
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca);
        entry.push_instruction(store);
        entry.push_instruction(load);
        entry.push_instruction(ret);

        func
    }

    /// Create a function with two variables and one block:
    /// ```text
    /// int g(int a, int b) {
    ///     int x = a;
    ///     int y = b;
    ///     return x + y; // simplified as load x, load y, add, ret
    /// }
    /// ```
    fn make_two_var_function() -> IrFunction {
        let params = vec![
            FunctionParam::new("a".to_string(), IrType::I32, Value(0)),
            FunctionParam::new("b".to_string(), IrType::I32, Value(1)),
        ];
        let mut func = IrFunction::new("g".to_string(), params, IrType::I32);
        func.value_count = 2;

        // %2 = alloca i32 (x)
        let alloca_x = Instruction::Alloca {
            result: Value(2),
            ty: IrType::I32,
            alignment: None,
            span: Span::dummy(),
        };
        // %3 = alloca i32 (y)
        let alloca_y = Instruction::Alloca {
            result: Value(3),
            ty: IrType::I32,
            alignment: None,
            span: Span::dummy(),
        };
        // store i32 %0, ptr %2  (x = a)
        let store_x = Instruction::Store {
            value: Value(0),
            ptr: Value(2),
            volatile: false,
            span: Span::dummy(),
        };
        // store i32 %1, ptr %3  (y = b)
        let store_y = Instruction::Store {
            value: Value(1),
            ptr: Value(3),
            volatile: false,
            span: Span::dummy(),
        };
        // %4 = load i32, ptr %2  (read x)
        let load_x = Instruction::Load {
            result: Value(4),
            ptr: Value(2),
            ty: IrType::I32,
            volatile: false,
            span: Span::dummy(),
        };
        // %5 = load i32, ptr %3  (read y)
        let load_y = Instruction::Load {
            result: Value(5),
            ptr: Value(3),
            ty: IrType::I32,
            volatile: false,
            span: Span::dummy(),
        };
        // ret i32 %4 (simplified — normally would add %4+%5)
        let ret = Instruction::Return {
            value: Some(Value(4)),
            span: Span::dummy(),
        };

        func.value_count = 6;
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca_x);
        entry.push_instruction(alloca_y);
        entry.push_instruction(store_x);
        entry.push_instruction(store_y);
        entry.push_instruction(load_x);
        entry.push_instruction(load_y);
        entry.push_instruction(ret);

        func
    }

    /// Create a function with a non-promotable alloca (address taken via call).
    fn make_address_taken_function() -> IrFunction {
        let params = vec![FunctionParam::new(
            "param".to_string(),
            IrType::I32,
            Value(0),
        )];
        let mut func = IrFunction::new("h".to_string(), params, IrType::I32);
        func.value_count = 1;

        // %1 = alloca i32
        let alloca = Instruction::Alloca {
            result: Value(1),
            ty: IrType::I32,
            alignment: None,
            span: Span::dummy(),
        };
        // call @foo(%1)  — address escapes via function argument
        let call = Instruction::Call {
            result: Value(2),
            callee: Value(10),    // some function
            args: vec![Value(1)], // alloca passed as argument
            return_type: IrType::Void,
            span: Span::dummy(),
        };
        let ret = Instruction::Return {
            value: None,
            span: Span::dummy(),
        };

        func.value_count = 3;
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca);
        entry.push_instruction(call);
        entry.push_instruction(ret);

        func
    }

    /// Create a function with a volatile load from an alloca.
    fn make_volatile_function() -> IrFunction {
        let params = vec![FunctionParam::new(
            "param".to_string(),
            IrType::I32,
            Value(0),
        )];
        let mut func = IrFunction::new("vol".to_string(), params, IrType::I32);
        func.value_count = 1;

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
        let volatile_load = Instruction::Load {
            result: Value(2),
            ptr: Value(1),
            ty: IrType::I32,
            volatile: true, // volatile!
            span: Span::dummy(),
        };
        let ret = Instruction::Return {
            value: Some(Value(2)),
            span: Span::dummy(),
        };

        func.value_count = 3;
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca);
        entry.push_instruction(store);
        entry.push_instruction(volatile_load);
        entry.push_instruction(ret);

        func
    }

    /// Create a function with an aggregate (struct) alloca.
    fn make_aggregate_alloca_function() -> IrFunction {
        use crate::ir::types::StructType;

        let mut func = IrFunction::new("agg".to_string(), vec![], IrType::Void);
        func.value_count = 0;

        let struct_ty = IrType::Struct(StructType::new(vec![IrType::I32, IrType::I64], false));
        let alloca = Instruction::Alloca {
            result: Value(0),
            ty: struct_ty,
            alignment: None,
            span: Span::dummy(),
        };
        let ret = Instruction::Return {
            value: None,
            span: Span::dummy(),
        };

        func.value_count = 1;
        let entry = func.entry_block_mut();
        entry.push_instruction(alloca);
        entry.push_instruction(ret);

        func
    }

    // -----------------------------------------------------------------------
    // is_promotable_alloca tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_promotable_scalar_alloca() {
        let func = make_simple_function();
        // %1 is an i32 alloca only used in non-volatile load/store.
        assert!(is_promotable_alloca(&func, Value(1)));
    }

    #[test]
    fn test_not_promotable_address_taken() {
        let func = make_address_taken_function();
        // %1 has its address passed to a call — not promotable.
        assert!(!is_promotable_alloca(&func, Value(1)));
    }

    #[test]
    fn test_not_promotable_volatile() {
        let func = make_volatile_function();
        // %1 has a volatile load — not promotable.
        assert!(!is_promotable_alloca(&func, Value(1)));
    }

    #[test]
    fn test_not_promotable_aggregate() {
        let func = make_aggregate_alloca_function();
        // %0 is a struct alloca — not scalar, not promotable.
        assert!(!is_promotable_alloca(&func, Value(0)));
    }

    #[test]
    fn test_nonexistent_alloca() {
        let func = make_simple_function();
        // Value(99) doesn't exist — should return false gracefully.
        assert!(!is_promotable_alloca(&func, Value(99)));
    }

    // -----------------------------------------------------------------------
    // collect_promotable_allocas tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_collect_simple() {
        let func = make_simple_function();
        let promotable = collect_promotable_allocas(&func);
        assert_eq!(promotable.len(), 1);
        assert!(promotable.contains_key(&Value(1)));
        assert_eq!(promotable.get(&Value(1)), Some(&IrType::I32));
    }

    #[test]
    fn test_collect_two_vars() {
        let func = make_two_var_function();
        let promotable = collect_promotable_allocas(&func);
        // Both %2 (x) and %3 (y) should be promotable.
        assert_eq!(promotable.len(), 2);
        assert!(promotable.contains_key(&Value(2)));
        assert!(promotable.contains_key(&Value(3)));
    }

    #[test]
    fn test_collect_none_promotable() {
        let func = make_address_taken_function();
        let promotable = collect_promotable_allocas(&func);
        assert!(promotable.is_empty());
    }

    #[test]
    fn test_collect_empty_function() {
        let mut func = IrFunction::new("empty".to_string(), vec![], IrType::Void);
        func.is_definition = false;
        func.blocks.clear();
        let promotable = collect_promotable_allocas(&func);
        assert!(promotable.is_empty());
    }

    // -----------------------------------------------------------------------
    // find_def_blocks / find_use_blocks tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_def_blocks_single() {
        let func = make_simple_function();
        let defs = find_def_blocks(&func, Value(1));
        // Store to %1 is in block 0 (entry).
        assert_eq!(defs.len(), 1);
        assert!(defs.contains(&0));
    }

    #[test]
    fn test_find_use_blocks_single() {
        let func = make_simple_function();
        let uses = find_use_blocks(&func, Value(1));
        // Load from %1 is in block 0 (entry).
        assert_eq!(uses.len(), 1);
        assert!(uses.contains(&0));
    }

    #[test]
    fn test_find_no_defs() {
        let func = make_simple_function();
        // Value(99) doesn't appear in any store.
        let defs = find_def_blocks(&func, Value(99));
        assert!(defs.is_empty());
    }

    // -----------------------------------------------------------------------
    // promote_allocas_to_ssa tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_promote_single_block_simple() {
        let mut func = make_simple_function();

        // Before promotion: entry has alloca, store, load, ret.
        assert_eq!(func.entry_block().instruction_count(), 4);

        promote_allocas_to_ssa(&mut func);

        // After promotion: alloca, store, and load should be removed.
        // Only the ret instruction remains.
        let entry = func.entry_block();
        assert_eq!(entry.instruction_count(), 1);

        // The return should now use Value(0) (the parameter) directly
        // instead of Value(2) (the promoted load result).
        if let Instruction::Return { value, .. } = &entry.instructions()[0] {
            assert_eq!(*value, Some(Value(0)));
        } else {
            panic!("Expected Return instruction");
        }
    }

    #[test]
    fn test_promote_two_vars_single_block() {
        let mut func = make_two_var_function();

        promote_allocas_to_ssa(&mut func);

        // After promotion: only the ret instruction should remain
        // (both allocas and their loads/stores removed).
        let entry = func.entry_block();

        // The return should use Value(0) (param a) since %4 = load from %2
        // which stored Value(0).
        if let Instruction::Return { value, .. } = entry.instructions().last().unwrap() {
            assert_eq!(*value, Some(Value(0)));
        } else {
            panic!("Expected Return instruction");
        }
    }

    #[test]
    fn test_promote_no_promotable_allocas() {
        let mut func = make_address_taken_function();
        let original_count = func.entry_block().instruction_count();

        promote_allocas_to_ssa(&mut func);

        // Nothing should change — no promotable allocas.
        assert_eq!(func.entry_block().instruction_count(), original_count);
    }

    #[test]
    fn test_promote_empty_function() {
        let mut func = IrFunction::new("empty".to_string(), vec![], IrType::Void);
        func.blocks.clear();

        // Should not panic on empty function.
        promote_allocas_to_ssa(&mut func);
    }

    // -----------------------------------------------------------------------
    // run_mem2reg tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_mem2reg_module() {
        let mut module = IrModule::new("test.c".to_string());

        // Add a function definition with a promotable alloca.
        let mut func = make_simple_function();
        func.is_definition = true;
        module.functions.push(func);

        // Add a declaration (should be skipped).
        let mut decl_func = IrFunction::new("extern_fn".to_string(), vec![], IrType::Void);
        decl_func.is_definition = false;
        decl_func.blocks.clear();
        module.functions.push(decl_func);

        run_mem2reg(&mut module);

        // First function should have been promoted.
        let promoted_func = &module.functions[0];
        assert_eq!(promoted_func.entry_block().instruction_count(), 1);

        // Second function (declaration) should be unchanged.
        assert!(!module.functions[1].is_definition);
    }

    // -----------------------------------------------------------------------
    // Multi-block promotion test
    // -----------------------------------------------------------------------

    #[test]
    fn test_promote_multi_block() {
        // Build a simple if-then-else CFG:
        //
        //   entry (bb0):
        //     %1 = alloca i32
        //     store i32 %0, ptr %1
        //     br i1 %cond, label %bb1, label %bb2
        //
        //   bb1:
        //     store i32 42, ptr %1  (using a made-up constant value %10)
        //     br label %bb3
        //
        //   bb2:
        //     store i32 99, ptr %1  (using a made-up constant value %11)
        //     br label %bb3
        //
        //   bb3:
        //     %2 = load i32, ptr %1
        //     ret i32 %2
        //
        // After mem2reg, bb3 should have a phi node merging the values
        // from bb1 and bb2.

        let params = vec![FunctionParam::new(
            "param".to_string(),
            IrType::I32,
            Value(0),
        )];
        let mut func = IrFunction::new("ifelse".to_string(), params, IrType::I32);
        func.value_count = 1;

        // Clear default entry block and build from scratch.
        func.blocks.clear();

        // bb0 (entry)
        let mut bb0 = BasicBlock::with_label(0, "entry".to_string());
        bb0.push_instruction(Instruction::Alloca {
            result: Value(1),
            ty: IrType::I32,
            alignment: None,
            span: Span::dummy(),
        });
        bb0.push_instruction(Instruction::Store {
            value: Value(0),
            ptr: Value(1),
            volatile: false,
            span: Span::dummy(),
        });
        // Use a made-up condition value %12 for the branch
        bb0.push_instruction(Instruction::CondBranch {
            condition: Value(12),
            then_block: BlockId(1),
            else_block: BlockId(2),
            span: Span::dummy(),
        });
        bb0.add_successor(1);
        bb0.add_successor(2);

        // bb1 (then)
        let mut bb1 = BasicBlock::with_label(1, "then".to_string());
        bb1.add_predecessor(0);
        bb1.push_instruction(Instruction::Store {
            value: Value(10), // some constant
            ptr: Value(1),
            volatile: false,
            span: Span::dummy(),
        });
        bb1.push_instruction(Instruction::Branch {
            target: BlockId(3),
            span: Span::dummy(),
        });
        bb1.add_successor(3);

        // bb2 (else)
        let mut bb2 = BasicBlock::with_label(2, "else".to_string());
        bb2.add_predecessor(0);
        bb2.push_instruction(Instruction::Store {
            value: Value(11), // some constant
            ptr: Value(1),
            volatile: false,
            span: Span::dummy(),
        });
        bb2.push_instruction(Instruction::Branch {
            target: BlockId(3),
            span: Span::dummy(),
        });
        bb2.add_successor(3);

        // bb3 (merge)
        let mut bb3 = BasicBlock::with_label(3, "merge".to_string());
        bb3.add_predecessor(1);
        bb3.add_predecessor(2);
        bb3.push_instruction(Instruction::Load {
            result: Value(2),
            ptr: Value(1),
            ty: IrType::I32,
            volatile: false,
            span: Span::dummy(),
        });
        bb3.push_instruction(Instruction::Return {
            value: Some(Value(2)),
            span: Span::dummy(),
        });

        func.blocks.push(bb0);
        func.blocks.push(bb1);
        func.blocks.push(bb2);
        func.blocks.push(bb3);
        func.value_count = 13; // accommodate all made-up values

        // Run mem2reg.
        promote_allocas_to_ssa(&mut func);

        // After promotion:
        // - The alloca and all loads/stores to it should be removed.
        // - bb3 should have a phi node (inserted by the IDF algorithm).
        // - The return in bb3 should use the phi's result (not the old load).

        // Verify alloca was removed from bb0.
        let bb0_insts = func.blocks[0].instructions();
        let has_alloca = bb0_insts.iter().any(|i| i.is_alloca());
        assert!(!has_alloca, "Alloca should be removed after promotion");

        // Verify bb3 has a phi node.
        let bb3_insts = func.blocks[3].instructions();
        let phi_count = bb3_insts.iter().filter(|i| i.is_phi()).count();
        assert_eq!(phi_count, 1, "bb3 should have exactly one phi node");

        // Verify the phi has incoming values from bb1 and bb2.
        if let Instruction::Phi { incoming, .. } = &bb3_insts[0] {
            assert_eq!(
                incoming.len(),
                2,
                "Phi should have 2 incoming pairs (from bb1 and bb2)"
            );
        } else {
            panic!("First instruction in bb3 should be a Phi");
        }
    }
}
