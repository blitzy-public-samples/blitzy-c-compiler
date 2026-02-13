//! Dead code elimination optimization pass for the BCC compiler.
//!
//! This module implements Phase 8's dead code elimination (DCE) pass, the second
//! pass in the fixed optimization pipeline:
//! **constant folding → dead code elimination → CFG simplification**.
//!
//! # Transformations
//!
//! | Transformation                | Description                                                    |
//! |-------------------------------|----------------------------------------------------------------|
//! | Dead instruction removal      | Remove instructions with unused results and no side effects    |
//! | Dead alloca elimination       | Remove allocas whose results have zero uses                    |
//! | Unreachable block removal     | Remove blocks not reachable from the entry block               |
//! | Phi cleanup                   | Remove incoming edges from deleted blocks in surviving phis    |
//! | Cascading elimination         | Removing one instruction may make its operands dead            |
//!
//! # Side-Effect Classification
//!
//! An instruction is **never dead** (even with unused results) if it has side
//! effects:
//!
//! - `Store` — writes to memory
//! - `Call` — may modify global state
//! - `InlineAsm` — always assumed to have side effects
//! - Terminators (`Branch`, `CondBranch`, `Switch`, `Return`) — affect control flow
//! - Volatile `Load` — must not be removed
//!
//! All other instructions are considered side-effect-free and are dead if their
//! result `ValueId` has zero uses.
//!
//! # Algorithm
//!
//! 1. **Build use-count map:** Scan all instructions in all blocks to count
//!    references to each `ValueId`.
//! 2. **Identify dead instructions:** Find instructions that produce a result
//!    with zero uses and have no side effects.
//! 3. **Worklist-driven removal:** Remove dead instructions iteratively. When
//!    an instruction is removed, decrement operand use counts; if any operand's
//!    count drops to zero, its defining instruction may also be dead.
//! 4. **Unreachable block removal:** BFS from the entry block; remove blocks
//!    not in the reachable set; clean up phi nodes in surviving blocks.
//!
//! # SSA Invariant Preservation
//!
//! - When removing unreachable blocks, their incoming edges are stripped from
//!   phi nodes in successor blocks via `Instruction::remove_phi_operand`.
//! - Degenerate phi nodes (single incoming value after cleanup) are replaced
//!   with their sole operand via `Instruction::replace_use`.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::basic_block::BasicBlockId;
use crate::ir::function::{IrFunction, ValueId};
use crate::ir::instructions::Instruction;
use crate::ir::module::{Constant, IrModule};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Runs the dead code elimination pass on a single IR function.
///
/// Returns `true` if any instructions or blocks were removed, `false` if
/// the function was unchanged.
///
/// # Algorithm
///
/// 1. Remove unreachable blocks (BFS from entry).
/// 2. Build use-count map for all values.
/// 3. Iteratively remove dead instructions (worklist).
/// 4. Repeat until fixpoint.
pub fn run_dead_code_elimination(func: &mut IrFunction) -> bool {
    let mut changed = false;

    // Phase 1: Unreachable block removal.
    changed |= remove_unreachable_blocks(func);

    // Phase 2: Dead instruction elimination (worklist-based).
    changed |= eliminate_dead_instructions(func);

    changed
}

// ---------------------------------------------------------------------------
// Optional module-level dead global elimination
// ---------------------------------------------------------------------------

/// Runs dead global elimination on an IR module.
///
/// Removes global variables that are never referenced by any function and
/// function declarations that are never called.  Externally visible globals
/// (external, weak, common linkage) are always retained because they may be
/// referenced by other translation units linked later.
///
/// # Algorithm
///
/// 1. **Root collection:** All function definition names are roots.  All
///    names referenced from function bodies (via `local_values` names and
///    inline assembly templates) are added to the referenced set.  Module-
///    level inline assembly blocks are treated as opaque — any global whose
///    name appears in their template text is conservatively marked live.
///
/// 2. **Transitive closure:** Global variable initializers may contain
///    [`Constant::GlobalRef`] references to other globals or functions.
///    We iterate to a fixpoint, propagating liveness through initializer
///    references so that a chain like `a → b → c` keeps all three alive
///    when `a` is live.
///
/// 3. **Pruning:** Internal-linkage globals with zero references are
///    removed.  Function declarations not in the referenced set are removed.
///
/// Returns `true` if any globals or declarations were removed.
pub fn run_dead_global_elimination(module: &mut IrModule) -> bool {
    let mut changed = false;

    // ------------------------------------------------------------------
    // Step 1: Collect root symbol names referenced from function bodies.
    // ------------------------------------------------------------------
    let mut referenced_names: FxHashSet<String> = FxHashSet::default();

    for func in &module.functions {
        // Every defined function is itself a root — it will be emitted.
        referenced_names.insert(func.name.clone());

        // Scan the function's SSA value registry.  During IR lowering, when
        // a global address is materialised, the resulting ValueId is typically
        // given the global's symbol name.  Capturing all named values is a
        // conservative superset that guarantees we never accidentally remove
        // a referenced global.
        for val_info in &func.local_values {
            if let Some(ref name) = val_info.name {
                referenced_names.insert(name.clone());
            }
        }

        // Scan inline assembly templates inside function bodies.  Template
        // strings may embed global symbol names (e.g., `".pushsection ..."`
        // or raw references in kernel code).  We conservatively mark every
        // global whose name appears as a substring of any template.
        collect_inline_asm_refs_from_function(func, &module.globals, &module.declarations, &mut referenced_names);
    }

    // Module-level inline assembly blocks are emitted verbatim and may
    // reference any symbol.  We conservatively keep every global or
    // declaration whose name appears in any module-level asm template.
    for asm_block in &module.inline_asm_blocks {
        for global in &module.globals {
            if asm_block.template.contains(&global.name) {
                referenced_names.insert(global.name.clone());
            }
        }
        for decl in &module.declarations {
            if asm_block.template.contains(&decl.name) {
                referenced_names.insert(decl.name.clone());
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: Transitive closure via Constant::GlobalRef in initializers.
    // ------------------------------------------------------------------
    // A global variable's initializer may reference other globals through
    // Constant::GlobalRef (e.g., `int *p = &other_global;`).  We propagate
    // liveness: if global A is referenced and A's initializer contains a
    // GlobalRef to B, then B is also referenced.  Repeat until fixpoint.
    loop {
        let prev_size = referenced_names.len();
        for global in &module.globals {
            if referenced_names.contains(&global.name) {
                if let Some(ref init) = global.initializer {
                    collect_global_refs_from_constant(init, &mut referenced_names);
                }
            }
        }
        // Fixpoint reached when no new names were added.
        if referenced_names.len() == prev_size {
            break;
        }
    }

    // ------------------------------------------------------------------
    // Step 3: Prune unreferenced internal globals and declarations.
    // ------------------------------------------------------------------

    // Remove unreferenced global variables with internal linkage.
    // Externally visible globals are always kept — removing them would
    // break cross-translation-unit references resolved at link time.
    let original_global_count = module.globals.len();
    module.globals.retain(|g| {
        if g.is_externally_visible() {
            return true;
        }
        referenced_names.contains(&g.name)
    });
    if module.globals.len() != original_global_count {
        changed = true;
    }

    // Remove function declarations that are never referenced.
    // Declarations are prototypes for external functions (e.g., `printf`).
    // If no function body references the declared name, the declaration is
    // dead and can be safely removed — it only occupies symbol table space.
    let original_decl_count = module.declarations.len();
    module.declarations.retain(|d| referenced_names.contains(&d.name));
    if module.declarations.len() != original_decl_count {
        changed = true;
    }

    changed
}

// ---------------------------------------------------------------------------
// Global elimination helpers
// ---------------------------------------------------------------------------

/// Recursively collects all symbol names referenced via [`Constant::GlobalRef`]
/// within a constant tree (used for global variable initializers).
///
/// Array and struct constants may contain nested `GlobalRef` entries at
/// arbitrary depth.  This function descends into all aggregate children
/// to capture every referenced name.
fn collect_global_refs_from_constant(constant: &Constant, names: &mut FxHashSet<String>) {
    match constant {
        Constant::GlobalRef { name } => {
            names.insert(name.clone());
        }
        Constant::Array { elements, .. } => {
            for elem in elements {
                collect_global_refs_from_constant(elem, names);
            }
        }
        Constant::Struct { fields, .. } => {
            for field in fields {
                collect_global_refs_from_constant(field, names);
            }
        }
        // Scalar constants (Int, Float, String, Null, Zero) do not reference
        // other globals by name.
        Constant::Int { .. }
        | Constant::Float { .. }
        | Constant::String { .. }
        | Constant::Null { .. }
        | Constant::Zero { .. } => {}
    }
}

/// Scans inline assembly templates within a function's body for references
/// to global variable and function declaration symbol names.
///
/// Inline asm template strings may embed raw symbol names (e.g., in
/// `.pushsection`/`.popsection` directives or operand expressions).
/// Since parsing arbitrary assembly is infeasible, we use a conservative
/// substring check: if any global or declaration name appears within a
/// template, we mark it as referenced.
///
/// Only `InlineAsm` instructions are examined; other instruction types
/// reference globals solely through `ValueId` operands, which are
/// already captured via the `local_values` scan.
fn collect_inline_asm_refs_from_function(
    func: &IrFunction,
    globals: &[crate::ir::module::GlobalVariable],
    declarations: &[crate::ir::module::FunctionDecl],
    names: &mut FxHashSet<String>,
) {
    for block in func.blocks() {
        for inst in block.instructions() {
            if let Instruction::InlineAsm { template, .. } = inst {
                // Check if any global variable name appears in the template.
                for global in globals {
                    if template.contains(&global.name) {
                        names.insert(global.name.clone());
                    }
                }
                // Check if any function declaration name appears in the template.
                for decl in declarations {
                    if template.contains(&decl.name) {
                        names.insert(decl.name.clone());
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unreachable block removal
// ---------------------------------------------------------------------------

/// Removes basic blocks that are not reachable from the entry block.
///
/// Uses BFS to compute the reachable set, then removes all blocks not in
/// that set. For each removed block, phi nodes in its successors are cleaned
/// up to remove the now-invalid incoming edges.
///
/// Returns `true` if any blocks were removed.
fn remove_unreachable_blocks(func: &mut IrFunction) -> bool {
    let reachable = compute_reachable_blocks(func);

    // Collect IDs of unreachable blocks.
    let unreachable_ids: Vec<BasicBlockId> = func
        .blocks()
        .iter()
        .filter(|b| !reachable.contains(&b.id))
        .map(|b| b.id)
        .collect();

    if unreachable_ids.is_empty() {
        return false;
    }

    // For each unreachable block, clean up phi nodes in successor blocks
    // that are reachable (i.e., still alive).
    for &dead_id in &unreachable_ids {
        // Get successors of the dead block before removal.
        let successors: Vec<BasicBlockId> = func.get_block(dead_id).successors().to_vec();

        for &succ_id in &successors {
            if reachable.contains(&succ_id) {
                // Remove dead block from successor's predecessor list.
                let succ = func.get_block_mut(succ_id);
                succ.remove_predecessor(dead_id);

                // Remove incoming phi edges from the dead block.
                let insts = succ.instructions_mut();
                for inst in insts.iter_mut() {
                    if inst.is_phi() {
                        inst.remove_phi_operand(dead_id);
                    }
                }
            }
        }
    }

    // Remove unreachable blocks from the function.
    for &dead_id in &unreachable_ids {
        func.remove_block(dead_id);
    }

    // Simplify degenerate phi nodes that may have been created.
    simplify_degenerate_phis(func);

    true
}

/// Computes the set of basic blocks reachable from the entry block via BFS.
fn compute_reachable_blocks(func: &IrFunction) -> FxHashSet<BasicBlockId> {
    let mut reachable = FxHashSet::default();
    let mut worklist: Vec<BasicBlockId> = Vec::new();

    let entry_id = func.entry_block().id;
    reachable.insert(entry_id);
    worklist.push(entry_id);

    while let Some(block_id) = worklist.pop() {
        let block = func.get_block(block_id);

        // Follow successor edges from the terminator.
        if let Some(term) = block.terminator() {
            for succ in term.successor_blocks() {
                if reachable.insert(succ) {
                    worklist.push(succ);
                }
            }
        }

        // Also follow explicit successor list in case it diverges from
        // the terminator (defensive).
        for &succ in block.successors() {
            if reachable.insert(succ) {
                worklist.push(succ);
            }
        }
    }

    reachable
}

// ---------------------------------------------------------------------------
// Dead instruction elimination (worklist-based)
// ---------------------------------------------------------------------------

/// Iteratively removes dead instructions from the function.
///
/// An instruction is dead if:
/// 1. It produces a result (`ValueId`).
/// 2. That result has zero uses across all other instructions.
/// 3. The instruction has no side effects.
///
/// When an instruction is removed, its operands' use counts are decremented.
/// If any operand's count drops to zero and its defining instruction is
/// side-effect-free, that instruction becomes dead too (cascading elimination).
///
/// Returns `true` if any instructions were removed.
fn eliminate_dead_instructions(func: &mut IrFunction) -> bool {
    let mut changed = false;

    // Build the initial use-count map.
    let mut use_counts: FxHashMap<ValueId, usize> = FxHashMap::default();
    for block in func.blocks() {
        for inst in block.instructions() {
            for used_val in inst.uses() {
                *use_counts.entry(used_val).or_insert(0) += 1;
            }
        }
    }

    // Iterate until no more dead instructions are found.
    let mut found_dead = true;
    while found_dead {
        found_dead = false;

        let block_ids: Vec<BasicBlockId> = func.blocks().iter().map(|b| b.id).collect();

        for &block_id in &block_ids {
            let mut idx = 0;
            loop {
                let block = func.get_block(block_id);
                if idx >= block.instructions().len() {
                    break;
                }
                let inst = &block.instructions()[idx];

                // Check if this instruction is dead.
                if is_dead_instruction(inst, &use_counts) {
                    // Record operands before removal (for use-count decrement).
                    let operand_ids = inst.uses();

                    // Remove the instruction.
                    let block = func.get_block_mut(block_id);
                    block.remove_instruction(idx);

                    // Decrement use counts for all operands.
                    for op in operand_ids {
                        if let Some(count) = use_counts.get_mut(&op) {
                            *count = count.saturating_sub(1);
                        }
                    }

                    found_dead = true;
                    changed = true;
                    // Don't increment idx — the next instruction is now at idx.
                } else {
                    idx += 1;
                }
            }
        }
    }

    changed
}

/// Determines whether an instruction is dead (removable).
///
/// An instruction is dead if:
/// - It produces a result value.
/// - That result has zero uses.
/// - It has no observable side effects.
fn is_dead_instruction(inst: &Instruction, use_counts: &FxHashMap<ValueId, usize>) -> bool {
    // Instructions without results are never "dead" in the DCE sense
    // (they're side-effecting terminators/stores).
    let result_id = match inst.result() {
        Some(id) => id,
        None => return false,
    };

    // Side-effecting instructions are never dead.
    if inst.has_side_effects() {
        return false;
    }

    // Check use count: zero uses means dead.
    let count = use_counts.get(&result_id).copied().unwrap_or(0);
    count == 0
}

// ---------------------------------------------------------------------------
// Degenerate phi simplification
// ---------------------------------------------------------------------------

/// Simplifies phi nodes that have become degenerate (single incoming value
/// or all-identical incoming values) after block removal.
///
/// For each degenerate phi, replaces all uses of the phi's result with
/// the sole incoming value.
fn simplify_degenerate_phis(func: &mut IrFunction) {
    // Collect degenerate phi replacements: (phi_result, replacement_value).
    let mut replacements: Vec<(ValueId, ValueId)> = Vec::new();

    for block in func.blocks() {
        for inst in block.instructions() {
            if let Instruction::Phi {
                result, incoming, ..
            } = inst
            {
                if incoming.len() == 1 {
                    // Single incoming value — phi is trivially equal to it.
                    replacements.push((*result, incoming[0].0));
                } else if incoming.len() > 1 {
                    // Check if all incoming values are the same.
                    let first_val = incoming[0].0;
                    if incoming.iter().all(|(v, _)| *v == first_val) {
                        replacements.push((*result, first_val));
                    }
                }
            }
        }
    }

    // Apply replacements across the entire function.
    for (old_id, new_id) in &replacements {
        for block in func.blocks_mut() {
            for inst in block.instructions_mut() {
                inst.replace_use(*old_id, *new_id);
            }
        }
    }
}
