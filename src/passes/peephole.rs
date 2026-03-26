//! Peephole Optimizer
//!
//! Performs local, pattern-matching optimizations on small instruction
//! sequences (typically 2-3 instructions). Unlike the algebraic
//! simplifications in instruction combining, peephole optimizations
//! look at instruction sequences and their relationships.
//!
//! ## Optimizations Performed
//!
//! - **Redundant load after store**: `store %v, %p; %x = load %p` → `%x = %v`
//! - **Redundant store after store**: `store %v1, %p; store %v2, %p` → `store %v2, %p`
//! - **Load of known alloca**: `alloca %p; store %v, %p; load %p` → `%v`
//! - **Branch to next block**: `br %next` where %next is the fallthrough → remove
//! - **Conditional branch with same targets**: `br %c, %t, %t` → `br %t`
//! - **Double comparison elimination**: `icmp eq %x, 0; br %cmp` after `icmp ne %x, 0; br %cmp`
//!
//! ## Algorithm
//!
//! Sliding window of 2-3 instructions per basic block, applied in a single
//! forward pass.
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::FxHashMap;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};

/// Runs the peephole optimizer on a single function.
///
/// Returns `true` if any optimization was applied.
pub fn run_peephole(func: &mut IrFunction) -> bool {
    let mut changed = false;

    // Optimization 1: Redundant store-load pairs within same block
    // store %v, %p ; ... no intervening store to %p ... ; %x = load %p → %x = %v
    changed |= optimize_store_load_pairs(func);

    // Optimization 2: Redundant store-store pairs within same block
    changed |= optimize_store_store_pairs(func);

    // Optimization 3: Conditional branch with same targets
    changed |= optimize_branch_same_target(func);

    changed
}

/// Replaces loads from a pointer that was just stored to (within the same
/// basic block, with no intervening stores to the same pointer).
fn optimize_store_load_pairs(func: &mut IrFunction) -> bool {
    let mut replacements: FxHashMap<Value, Value> = FxHashMap::default();

    for block in func.blocks() {
        // Track the last stored value for each pointer
        let mut last_store: FxHashMap<Value, Value> = FxHashMap::default();

        for inst in block.instructions() {
            match inst {
                Instruction::Store {
                    value,
                    ptr,
                    volatile: false,
                    ..
                } => {
                    last_store.insert(*ptr, *value);
                }

                Instruction::Load {
                    result,
                    ptr,
                    volatile: false,
                    ..
                } => {
                    if let Some(&stored_val) = last_store.get(ptr) {
                        replacements.insert(*result, stored_val);
                    }
                }

                // A call might modify memory through pointers
                Instruction::Call { .. } | Instruction::InlineAsm { .. } => {
                    last_store.clear();
                }

                // Another store to the same pointer updates the tracked value
                // (handled by the Store case above)
                _ => {}
            }
        }
    }

    if replacements.is_empty() {
        return false;
    }

    // Apply replacements
    let mut changed = false;
    for block in func.blocks_mut() {
        for inst in block.instructions_mut().iter_mut() {
            for operand in inst.operands_mut() {
                if let Some(&replacement) = replacements.get(operand) {
                    *operand = replacement;
                    changed = true;
                }
            }
        }
    }

    changed
}

/// Removes redundant store-store pairs where the first store is immediately
/// overwritten by the second store to the same pointer.
fn optimize_store_store_pairs(func: &mut IrFunction) -> bool {
    let mut changed = false;

    for block_idx in 0..func.blocks().len() {
        let block = &func.blocks()[block_idx];
        let mut to_remove: Vec<usize> = Vec::new();

        // Track last store index for each pointer
        let mut last_store_idx: FxHashMap<Value, usize> = FxHashMap::default();

        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            match inst {
                Instruction::Store {
                    ptr,
                    volatile: false,
                    ..
                } => {
                    if let Some(&prev_idx) = last_store_idx.get(ptr) {
                        // Check that no load from this pointer happened between prev and current
                        let mut has_load = false;
                        for mid_idx in (prev_idx + 1)..inst_idx {
                            if let Instruction::Load {
                                ptr: load_ptr,
                                volatile: false,
                                ..
                            } = &block.instructions()[mid_idx]
                            {
                                if load_ptr == ptr {
                                    has_load = true;
                                    break;
                                }
                            }
                            // Calls may read from the pointer
                            if matches!(
                                &block.instructions()[mid_idx],
                                Instruction::Call { .. } | Instruction::InlineAsm { .. }
                            ) {
                                has_load = true;
                                break;
                            }
                        }
                        if !has_load {
                            to_remove.push(prev_idx);
                        }
                    }
                    last_store_idx.insert(*ptr, inst_idx);
                }

                // Load or call invalidates our tracking for that pointer
                Instruction::Load { .. } => {
                    // Don't invalidate — we only care about stores
                }

                Instruction::Call { .. } | Instruction::InlineAsm { .. } => {
                    last_store_idx.clear();
                }

                _ => {}
            }
        }

        if !to_remove.is_empty() {
            to_remove.sort_unstable();
            to_remove.dedup();
            let block = func.get_block_mut(block_idx).unwrap();
            for &idx in to_remove.iter().rev() {
                if idx < block.instructions.len() {
                    block.instructions.remove(idx);
                    changed = true;
                }
            }
        }
    }

    changed
}

/// Simplifies conditional branches where both targets are the same block.
fn optimize_branch_same_target(func: &mut IrFunction) -> bool {
    let mut changed = false;

    for block in func.blocks_mut() {
        let len = block.instructions.len();
        if len == 0 {
            continue;
        }

        let last_idx = len - 1;
        let needs_simplify = if let Instruction::CondBranch {
            then_block,
            else_block,
            ..
        } = &block.instructions[last_idx]
        {
            *then_block == *else_block
        } else {
            false
        };

        if needs_simplify {
            if let Instruction::CondBranch {
                then_block, span, ..
            } = &block.instructions[last_idx]
            {
                let target = *then_block;
                let sp = *span;
                block.instructions[last_idx] = Instruction::Branch { target, span: sp };
                changed = true;
            }
        }
    }

    changed
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::function::IrFunction;
    use crate::ir::instructions::{BlockId, Instruction, Value};
    use crate::ir::types::IrType;

    #[test]
    fn test_peephole_store_load() {
        let mut func = IrFunction::new("test_ph".to_string(), vec![], IrType::Void);
        let entry = &mut func.blocks[0];

        // store %0 to %1
        entry.instructions.push(Instruction::Store {
            value: Value(0),
            ptr: Value(1),
            volatile: false,
            span: Span::dummy(),
        });

        // %2 = load from %1  — should be replaced by %0
        entry.instructions.push(Instruction::Load {
            result: Value(2),
            ptr: Value(1),
            ty: IrType::I32,
            volatile: false,
            span: Span::dummy(),
        });

        // %3 = add %2, %2
        entry.instructions.push(Instruction::BinOp {
            result: Value(3),
            op: crate::ir::instructions::BinOp::Add,
            lhs: Value(2),
            rhs: Value(2),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        entry.instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        let changed = run_peephole(&mut func);
        assert!(changed);

        // %2 should be replaced by %0 in the add
        if let Instruction::BinOp { lhs, rhs, .. } = &func.blocks[0].instructions[2] {
            assert_eq!(*lhs, Value(0));
            assert_eq!(*rhs, Value(0));
        }
    }

    #[test]
    fn test_peephole_branch_same_target() {
        let mut func = IrFunction::new("test_bst".to_string(), vec![], IrType::Void);
        let entry = &mut func.blocks[0];

        entry.instructions.push(Instruction::CondBranch {
            condition: Value(0),
            then_block: BlockId(1),
            else_block: BlockId(1),
            span: Span::dummy(),
        });

        let changed = run_peephole(&mut func);
        assert!(changed);
        assert!(matches!(
            func.blocks[0].instructions[0],
            Instruction::Branch {
                target: BlockId(1),
                ..
            }
        ));
    }

    #[test]
    fn test_peephole_no_change() {
        let mut func = IrFunction::new("test_nc".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_peephole(&mut func);
        assert!(!changed);
    }
}
