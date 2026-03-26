//! Aggressive Dead Code Elimination (ADCE)
//!
//! A more aggressive variant of dead code elimination that uses reverse
//! reachability from "essential" instructions (stores, calls, returns,
//! branches) to mark live code. Everything not reachable from an essential
//! instruction is dead and can be removed.
//!
//! ## Differences from Simple DCE
//!
//! Simple DCE only removes instructions whose results have zero uses.
//! ADCE additionally removes entire chains of computation that feed only
//! into dead code — even if intermediate values appear to have uses,
//! those uses may themselves be dead.
//!
//! ## Algorithm
//!
//! 1. Mark all "essential" instructions as live:
//!    - Stores (including volatile)
//!    - Calls (may have side effects)
//!    - Returns
//!    - Branches (control flow)
//!    - InlineAsm
//!    - StackSave/StackRestore
//! 2. Walk backwards through use-def chains: for each live instruction,
//!    mark all operand-producing instructions as live
//! 3. Remove all instructions not marked live
//!
//! ## Zero-Dependency
//!
//! Uses only `crate::ir::*`, `crate::common::fx_hash`, and `std`.

use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::ir::function::IrFunction;
use crate::ir::instructions::{Instruction, Value};
use std::collections::VecDeque;

/// Checks if an instruction is "essential" — has side effects that prevent
/// removal regardless of whether its result is used.
fn is_essential(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Store { .. }
            | Instruction::Call { .. }
            | Instruction::Return { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Switch { .. }
            | Instruction::IndirectBranch { .. }
            | Instruction::InlineAsm { .. }
            | Instruction::StackSave { .. }
            | Instruction::StackRestore { .. }
    ) || matches!(inst, Instruction::Load { volatile: true, .. })
}

/// Runs aggressive dead code elimination on a single function.
///
/// Returns `true` if any instructions were removed.
pub fn run_adce(func: &mut IrFunction) -> bool {
    // Phase 1: Build def map (Value → (block_idx, inst_idx))
    let mut def_map: FxHashMap<Value, (usize, usize)> = FxHashMap::default();
    for (block_idx, block) in func.blocks().iter().enumerate() {
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            if let Some(result) = inst.result() {
                def_map.insert(result, (block_idx, inst_idx));
            }
        }
    }

    // Phase 2: Mark essential instructions and walk backwards
    let mut live: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();

    // Seed with essential instructions
    for (block_idx, block) in func.blocks().iter().enumerate() {
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            if is_essential(inst) && live.insert((block_idx, inst_idx)) {
                worklist.push_back((block_idx, inst_idx));
            }
        }
    }

    // Walk backwards through use-def chains
    while let Some((block_idx, inst_idx)) = worklist.pop_front() {
        let block = &func.blocks()[block_idx];
        if inst_idx >= block.instructions().len() {
            continue;
        }
        let inst = &block.instructions()[inst_idx];

        // Mark all operand-producing instructions as live
        for operand in inst.operands() {
            if let Some(&(def_block, def_inst)) = def_map.get(&operand) {
                if live.insert((def_block, def_inst)) {
                    worklist.push_back((def_block, def_inst));
                }
            }
        }
    }

    // Also keep all Alloca instructions alive (they define stack slots)
    for (block_idx, block) in func.blocks().iter().enumerate() {
        for (inst_idx, inst) in block.instructions().iter().enumerate() {
            if inst.is_alloca() {
                live.insert((block_idx, inst_idx));
            }
        }
    }

    // Phase 3: Remove dead instructions
    let mut changed = false;
    for block_idx in 0..func.blocks().len() {
        let block = func.get_block_mut(block_idx).unwrap();
        let original_len = block.instructions.len();
        let mut new_insts = Vec::with_capacity(original_len);

        for (orig_idx, inst) in block.instructions.drain(..).enumerate() {
            if live.contains(&(block_idx, orig_idx)) {
                new_insts.push(inst);
            } else {
                changed = true;
            }
        }

        block.instructions = new_insts;
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
    use crate::ir::instructions::{BinOp, Instruction, Value};
    use crate::ir::types::IrType;

    #[test]
    fn test_adce_removes_dead_chain() {
        let mut func = IrFunction::new("test_adce".to_string(), vec![], IrType::Void);
        let entry = &mut func.blocks[0];

        // %0 = add i32 %param, %param  — dead (not used by return)
        entry.instructions.push(Instruction::BinOp {
            result: Value(10),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // %1 = mul i32 %0, %0  — dead (feeds dead %2)
        entry.instructions.push(Instruction::BinOp {
            result: Value(11),
            op: BinOp::Mul,
            lhs: Value(10),
            rhs: Value(10),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // return void — essential
        entry.instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        let changed = run_adce(&mut func);
        assert!(changed);
        // Only the return should remain
        assert_eq!(func.blocks[0].instructions.len(), 1);
        assert!(func.blocks[0].instructions[0].is_terminator());
    }

    #[test]
    fn test_adce_preserves_live_chain() {
        let mut func = IrFunction::new("test_adce2".to_string(), vec![], IrType::I32);
        let entry = &mut func.blocks[0];

        // %0 = add i32 %param, %param  — live (used by return)
        entry.instructions.push(Instruction::BinOp {
            result: Value(10),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(0),
            ty: IrType::I32,
            span: Span::dummy(),
        });

        // return %0 — essential
        entry.instructions.push(Instruction::Return {
            value: Some(Value(10)),
            span: Span::dummy(),
        });

        let changed = run_adce(&mut func);
        assert!(!changed);
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_adce_empty_function() {
        let mut func = IrFunction::new("test_adce3".to_string(), vec![], IrType::Void);
        func.blocks[0].instructions.push(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });
        let changed = run_adce(&mut func);
        assert!(!changed);
    }
}
